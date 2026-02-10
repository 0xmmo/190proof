import {
  ClaudeModel,
  GPTModel,
  OpenAIPayload,
  OpenAIMessage,
  OpenAIConfig,
  AnthropicAIPayload,
  AnthropicAIMessage,
  GenericMessage,
  AnthropicAIConfig,
  GenericPayload,
  GroqPayload,
  GroqModel,
  ParsedResponseMessage,
  FunctionCall,
  AnthropicContentBlock,
  OpenAIContentBlock,
  GoogleAIPayload,
  GeminiModel,
  GoogleAIPart,
  File,
  GoogleAIMessage,
  AnyModel,
} from "./interfaces";
import logger, { Identifier } from "./logger";
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import axios from "axios";
import { isHeicImage, timeout } from "./utils";
import { GoogleGenAI } from "@google/genai";

const sharp = require("sharp");
const decode = require("heic-decode");

export {
  ClaudeModel,
  GPTModel,
  GroqModel,
  GeminiModel,
  OpenAIConfig,
  FunctionDefinition,
  GenericMessage,
  GenericPayload,
  AnyModel,
} from "./interfaces";

// ─────────────────────────────────────────────────────────────────────────────
// SHARED UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generic retry wrapper for API calls with exponential backoff.
 */
async function withRetries<T>(
  identifier: Identifier,
  apiName: string,
  fn: () => Promise<T>,
  options: {
    retries?: number;
    baseDelayMs?: number;
    onError?: (error: any, attempt: number) => void;
  } = {}
): Promise<T> {
  const { retries = 5, baseDelayMs = 125, onError } = options;

  logger.log(identifier, `Calling ${apiName} API with retries`);

  let lastError: any;
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;

      if (onError) {
        onError(error, attempt);
      } else {
        logger.error(
          identifier,
          `Retry #${attempt} error: ${error.message}`,
          error.response?.data || error
        );
      }

      await timeout(baseDelayMs * attempt);
    }
  }

  const error = new Error(
    `Failed to call ${apiName} API after ${retries} attempts`
  ) as any;
  error.cause = lastError;
  throw error;
}

function parseStreamedResponse(
  identifier: Identifier,
  paragraph: string,
  functionCallName: string,
  functionCallArgs: string,
  allowedFunctionNames: Set<string> | null
): ParsedResponseMessage {
  let functionCall: ParsedResponseMessage["function_call"] = null;

  if (functionCallName && functionCallArgs) {
    if (allowedFunctionNames && !allowedFunctionNames.has(functionCallName)) {
      throw new Error(
        `Stream error: received function call with unknown name: ${functionCallName}`
      );
    }

    try {
      functionCall = {
        name: functionCallName,
        arguments: JSON.parse(functionCallArgs),
      };
    } catch (error) {
      logger.error(
        identifier,
        "Error parsing function call arguments:",
        functionCallArgs
      );
      throw error;
    }
  }

  if (!paragraph && !functionCall) {
    logger.error(
      identifier,
      "Stream error: received message without content or function_call:",
      JSON.stringify({ paragraph, functionCallName, functionCallArgs })
    );
    throw new Error(
      "Stream error: received message without content or function_call"
    );
  }

  return {
    role: "assistant",
    content: paragraph || null,
    function_call: functionCall,
    files: [],
  };
}

function truncatePayload(payload: OpenAIPayload): string {
  return JSON.stringify(
    {
      ...payload,
      messages: payload.messages.map((message) => {
        const truncatedMessage = { ...message };
        if (typeof truncatedMessage.content === "string") {
          truncatedMessage.content = truncatedMessage.content.slice(0, 100);
        } else if (Array.isArray(truncatedMessage.content)) {
          truncatedMessage.content = truncatedMessage.content.map((block) => {
            if (block.type === "image_url") {
              return {
                ...block,
                image_url: { url: block.image_url.url.slice(0, 100) },
              };
            }
            return block;
          });
        }
        return truncatedMessage;
      }),
    },
    null,
    2
  );
}

async function getNormalizedBase64PNG(
  url: string,
  mime: string
): Promise<string> {
  const response = await axios.get(url, { responseType: "arraybuffer" });

  let imageBuffer = Buffer.from(response.data);
  let sharpOptions = {};

  if (isHeicImage(url, mime)) {
    const imageData = await decode({ buffer: imageBuffer });
    imageBuffer = Buffer.from(imageData.data);
    sharpOptions = {
      raw: {
        width: imageData.width,
        height: imageData.height,
        channels: 4,
      },
    };
  }

  // Limits size of image to < 5MB Anthropic limit
  const resizedBuffer = await sharp(imageBuffer, sharpOptions)
    .withMetadata()
    .resize(1024, 1024, { fit: "inside", withoutEnlargement: true })
    .png()
    .toBuffer();

  return resizedBuffer.toString("base64");
}

const ALLOWED_IMAGE_MIME_TYPES = [
  "image/png",
  "image/jpeg",
  "image/gif",
  "image/webp",
];

// ─────────────────────────────────────────────────────────────────────────────
// OPENAI
// ─────────────────────────────────────────────────────────────────────────────

interface OpenAIRequestConfig {
  endpoint: string;
  headers: Record<string, string>;
}

function buildOpenAIRequestConfig(
  identifier: Identifier,
  model: string,
  config: OpenAIConfig | undefined
): OpenAIRequestConfig {
  if (!config) {
    config = {
      service: "openai",
      apiKey: process.env.OPENAI_API_KEY as string,
      baseUrl: "",
    };
  }

  if (config.service === "azure") {
    logger.log(identifier, "Using Azure OpenAI service:", model);

    if (!config.modelConfigMap) {
      throw new Error(
        "OpenAI config modelConfigMap is required when using Azure OpenAI service."
      );
    }

    const azureConfig = config.modelConfigMap[model as GPTModel];
    if (!azureConfig?.endpoint) {
      throw new Error("Azure OpenAI endpoint is required in modelConfigMap.");
    }

    const endpoint = `${azureConfig.endpoint}/openai/deployments/${azureConfig.deployment}/chat/completions?api-version=${azureConfig.apiVersion}`;
    logger.log(identifier, "Using endpoint:", endpoint);

    return {
      endpoint,
      headers: {
        "Content-Type": "application/json",
        "api-key": azureConfig.apiKey,
      },
    };
  }

  // Default: OpenAI
  logger.log(identifier, "Using OpenAI service:", model);
  if (config.orgId) {
    logger.log(identifier, "Using orgId:", config.orgId);
  }

  return {
    endpoint: "https://api.openai.com/v1/chat/completions",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`,
      ...(config.orgId ? { "OpenAI-Organization": config.orgId } : {}),
    },
  };
}

async function prepareOpenAIPayload(
  identifier: Identifier,
  payload: GenericPayload
): Promise<OpenAIPayload> {
  const preparedPayload: OpenAIPayload = {
    model: payload.model as GPTModel,
    messages: [],
    tools: payload.functions?.map((fn) => ({
      type: "function",
      function: fn,
    })),
    tool_choice: payload.function_call
      ? typeof payload.function_call === "string"
        ? payload.function_call
        : { type: "function", function: payload.function_call }
      : undefined,
  };

  for (const message of payload.messages) {
    const contentBlocks: OpenAIContentBlock[] = [];

    if (message.content) {
      contentBlocks.push({ type: "text", text: message.content });
    }

    for (const file of message.files || []) {
      if (ALLOWED_IMAGE_MIME_TYPES.includes(file.mimeType)) {
        if (file.url) {
          contentBlocks.push({
            type: "image_url",
            image_url: { url: file.url },
          });
          contentBlocks.push({ type: "text", text: `Image (${file.url})` });
        } else if (file.data) {
          contentBlocks.push({
            type: "image_url",
            image_url: { url: `data:${file.mimeType};base64,${file.data}` },
          });
        }
      } else if (file.url) {
        // Non-image file with URL - add text reference
        contentBlocks.push({
          type: "text",
          text: `File (${file.url})`,
        });
      }
    }

    preparedPayload.messages.push({
      role: message.role,
      content: contentBlocks,
    });
  }

  return preparedPayload;
}

async function callOpenAIStream(
  id: Identifier,
  openAiPayload: OpenAIPayload,
  openAiConfig: OpenAIConfig | undefined,
  chunkTimeoutMs: number
): Promise<ParsedResponseMessage> {
  const functionNames: Set<string> | null = openAiPayload.tools
    ? new Set(openAiPayload.tools.map((fn) => fn.function.name as string))
    : null;

  const { endpoint, headers } = buildOpenAIRequestConfig(
    id,
    openAiPayload.model,
    openAiConfig
  );

  const controller = new AbortController();
  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({ ...openAiPayload, stream: true }),
    signal: controller.signal,
  });

  if (!response.body) {
    throw new Error("Stream error: no response body");
  }

  let paragraph = "";
  let functionCallName = "";
  let functionCallArgs = "";
  let hasMultipleToolCalls = false;

  const reader = response.body.getReader();
  let partialChunk = "";
  let chunkIndex = -1;

  const createAbortTimeout = () =>
    setTimeout(() => {
      logger.error(id, `Stream timeout after ${chunkTimeoutMs}ms`);
      controller.abort();
    }, chunkTimeoutMs);

  while (true) {
    chunkIndex++;
    const abortTimeout = createAbortTimeout();
    const { done, value } = await reader.read();
    clearTimeout(abortTimeout);

    if (done) {
      logger.error(id, `Stream ended prematurely after ${chunkIndex + 1} chunks`);
      throw new Error("Stream error: ended prematurely");
    }

    let chunk = new TextDecoder().decode(value);
    if (partialChunk) {
      chunk = partialChunk + chunk;
      partialChunk = "";
    }

    const jsonStrings = chunk.split(/^data: /gm);

    for (const jsonString of jsonStrings) {
      if (!jsonString) continue;

      if (jsonString.includes("[DONE]")) {
        if (hasMultipleToolCalls) {
          logger.warn(
            id,
            "Discarding additional OpenAI function call(s) from stream (only first tool_call processed)"
          );
        }
        return parseStreamedResponse(
          id,
          paragraph,
          functionCallName,
          functionCallArgs,
          functionNames
        );
      }

      let json;
      try {
        json = JSON.parse(jsonString.trim());
      } catch {
        partialChunk = jsonString;
        continue;
      }

      if (!json.choices?.length) {
        if (json.error) {
          logger.error(id, "Stream error from OpenAI:", json.error);
          const error = new Error("Stream error: OpenAI error") as any;
          error.data = json.error;
          error.requestBody = truncatePayload(openAiPayload);
          throw error;
        }
        if (chunkIndex !== 0) {
          logger.error(id, "Stream error: no choices in JSON:", json);
        }
        continue;
      }

      const toolCalls = json.choices[0]?.delta?.tool_calls;
      if (toolCalls?.length > 1 || (toolCalls?.[0]?.index && toolCalls[0].index > 0)) {
        hasMultipleToolCalls = true;
      }
      const toolCall = toolCalls?.[0];
      if (toolCall?.index === 0 || toolCall?.index === undefined) {
        if (toolCall?.function?.name) functionCallName += toolCall.function.name;
        if (toolCall?.function?.arguments) functionCallArgs += toolCall.function.arguments;
      }

      const text = json.choices[0]?.delta?.content;
      if (text) paragraph += text;
    }
  }
}

async function callOpenAI(
  id: Identifier,
  openAiPayload: OpenAIPayload,
  openAiConfig: OpenAIConfig | undefined
): Promise<ParsedResponseMessage> {
  const { endpoint, headers } = buildOpenAIRequestConfig(
    id,
    openAiPayload.model,
    openAiConfig
  );

  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({ ...openAiPayload, stream: false }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    logger.error(id, "OpenAI API error:", errorData);
    throw new Error(`OpenAI API Error: ${errorData.error.message}`);
  }

  const data = await response.json();

  if (!data.choices?.length) {
    if (data.error) {
      logger.error(id, "OpenAI error:", data.error);
      throw new Error(`OpenAI error: ${data.error.message}`);
    }
    throw new Error("OpenAI error: No choices returned.");
  }

  const choice = data.choices[0];

  // Check for tool_calls (modern API) first, fall back to function_call (legacy)
  const toolCalls = choice.message?.tool_calls;
  let functionCall: FunctionCall | null = null;

  if (toolCalls?.length) {
    functionCall = {
      name: toolCalls[0].function.name,
      arguments: JSON.parse(toolCalls[0].function.arguments),
    };

    if (toolCalls.length > 1) {
      const allNames = toolCalls.map((tc: any) => tc.function.name).join(", ");
      const discarded = toolCalls.slice(1).map((tc: any) => `tool ${tc.function.name} with args ${JSON.stringify(JSON.parse(tc.function.arguments))}`).join(", ");
      logger.warn(id, `got ${toolCalls.length} tool calls for tools ${allNames}. using tool ${toolCalls[0].function.name} with args ${JSON.stringify(JSON.parse(toolCalls[0].function.arguments))} discarding ${discarded}`);
    }
  } else if (choice.function_call) {
    functionCall = {
      name: choice.function_call.name,
      arguments: JSON.parse(choice.function_call.arguments),
    };
  }

  return {
    role: "assistant",
    content: choice.message.content || null,
    function_call: functionCall,
    files: [],
  };
}

async function callOpenAiWithRetries(
  id: Identifier,
  openAiPayload: OpenAIPayload,
  openAiConfig?: OpenAIConfig,
  retries: number = 5,
  chunkTimeoutMs: number = 15_000
): Promise<ParsedResponseMessage> {
  logger.log(
    id,
    "Calling OpenAI API with retries:",
    openAiConfig?.service,
    openAiPayload.model
  );

  const useStreaming =
    openAiPayload.model !== GPTModel.O1_MINI &&
    openAiPayload.model !== GPTModel.O1_PREVIEW;

  return withRetries(
    id,
    "OpenAI",
    async () => {
      if (useStreaming) {
        return callOpenAIStream(id, openAiPayload, openAiConfig, chunkTimeoutMs);
      } else {
        return callOpenAI(id, openAiPayload, openAiConfig);
      }
    },
    {
      retries,
      baseDelayMs: 250,
      onError: (error, attempt) => {
        logger.error(
          id,
          `Retry #${attempt} error: ${error.message}`,
          error.response?.data || error.data || error
        );

        // Remove images on content policy violation
        if (error.data?.code === "content_policy_violation") {
          logger.log(id, "Removing images due to content policy violation");
          openAiPayload.messages.forEach((message: OpenAIMessage) => {
            if (Array.isArray(message.content)) {
              message.content = message.content.filter(
                (content) => content.type === "text"
              );
            }
          });
        }
      },
    }
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ANTHROPIC
// ─────────────────────────────────────────────────────────────────────────────

function jigAnthropicMessages(
  messages: AnthropicAIMessage[]
): AnthropicAIMessage[] {
  let jiggedMessages = messages.slice();

  // Ensure first message is from user
  if (jiggedMessages[0]?.role !== "user") {
    jiggedMessages = [{ role: "user" as const, content: "..." }, ...jiggedMessages];
  }

  // Group consecutive messages with the same role
  jiggedMessages = jiggedMessages.reduce((acc, message) => {
    if (acc.length === 0) return [message];

    const lastMessage = acc[acc.length - 1];
    if (lastMessage.role === message.role) {
      const lastContent = Array.isArray(lastMessage.content)
        ? lastMessage.content
        : [{ type: "text" as const, text: lastMessage.content }];
      const newContent = Array.isArray(message.content)
        ? message.content
        : [{ type: "text" as const, text: message.content }];

      lastMessage.content = [
        ...lastContent,
        { type: "text", text: "\n\n---\n\n" },
        ...newContent,
      ];
      return acc;
    }

    // Convert string content to text content block
    if (typeof message.content === "string") {
      message.content = [{ type: "text", text: message.content }];
    }

    return [...acc, message];
  }, [] as AnthropicAIMessage[]);

  // Ensure last message is from user
  if (jiggedMessages[jiggedMessages.length - 1]?.role === "assistant") {
    jiggedMessages.push({ role: "user", content: "..." });
  }

  return jiggedMessages;
}

async function prepareAnthropicPayload(
  _identifier: Identifier,
  payload: GenericPayload
): Promise<AnthropicAIPayload> {
  const preparedPayload: AnthropicAIPayload = {
    model: payload.model as ClaudeModel,
    messages: [],
    functions: payload.functions,
    temperature: payload.temperature,
  };

  for (const message of payload.messages) {
    if (message.role === "system") {
      preparedPayload.system = message.content;
      continue;
    }

    const contentBlocks: AnthropicContentBlock[] = [];

    if (message.content) {
      contentBlocks.push({ type: "text", text: message.content });
    }

    for (const file of message.files || []) {
      if (ALLOWED_IMAGE_MIME_TYPES.includes(file.mimeType)) {
        if (file.url) {
          contentBlocks.push({
            type: "image",
            source: {
              type: "base64",
              media_type: "image/png",
              data: await getNormalizedBase64PNG(file.url, file.mimeType),
            },
          });
          contentBlocks.push({ type: "text", text: `Image (${file.url})` });
        } else if (file.data) {
          contentBlocks.push({
            type: "image",
            source: {
              type: "base64",
              media_type: file.mimeType as any,
              data: file.data,
            },
          });
        }
      } else if (file.url) {
        // Non-image file with URL - add text reference
        contentBlocks.push({
          type: "text",
          text: `File (${file.url})`,
        });
      }
    }

    preparedPayload.messages.push({
      role: message.role,
      content: contentBlocks,
    });
  }

  return preparedPayload;
}

async function callAnthropic(
  id: Identifier,
  payload: AnthropicAIPayload,
  config?: AnthropicAIConfig
): Promise<ParsedResponseMessage> {
  const anthropicMessages = jigAnthropicMessages(payload.messages);
  const tools = payload.functions?.map((f) => ({
    ...f,
    input_schema: f.parameters,
    parameters: undefined,
  }));

  let data;

  if (config?.service === "bedrock") {
    const AWS_REGION = "us-east-1";
    const MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0";

    const client = new BedrockRuntimeClient({ region: AWS_REGION });
    const bedrockPayload = {
      anthropic_version: "bedrock-2023-05-31",
      max_tokens: 4096,
      messages: anthropicMessages,
      tools,
    };

    const response = await client.send(
      new InvokeModelCommand({
        contentType: "application/json",
        body: JSON.stringify(bedrockPayload),
        modelId: MODEL_ID,
      })
    );

    const decodedResponseBody = new TextDecoder().decode(response.body);
    data = JSON.parse(decodedResponseBody);
  } else {
    // Default: Anthropic API
    const response = await axios.post(
      "https://api.anthropic.com/v1/messages",
      {
        model: payload.model,
        messages: anthropicMessages,
        tools,
        temperature: payload.temperature,
        system: payload.system,
        max_tokens: 4096,
      },
      {
        headers: {
          "content-type": "application/json",
          "x-api-key": process.env.ANTHROPIC_API_KEY as string,
          "anthropic-version": "2023-06-01",
          "anthropic-beta": "tools-2024-04-04",
        },
        timeout: 60000,
      }
    );
    data = response.data;
  }

  const answers = data.content;
  if (!answers?.[0]) {
    logger.error(id, "Missing answer in Anthropic API response:", data);
    throw new Error("Missing answer in Anthropic API");
  }

  let textResponse = "";
  const functionCalls: FunctionCall[] = [];

  for (const answer of answers) {
    if (!answer.type) {
      logger.error(id, "Missing answer type in Anthropic API response:", data);
      throw new Error("Missing answer type in Anthropic API");
    }

    if (answer.type === "text") {
      let text = answer.text
        .replace(/<thinking>.*?<\/thinking>/gs, "")
        .replace(/<answer>|<\/answer>/gs, "")
        .trim();

      if (!text) {
        text = answer.text.replace(
          /<thinking>|<\/thinking>|<answer>|<\/answer>/gs,
          ""
        );
        logger.log(id, "No text in answer, returning text within tags:", text);
      }

      textResponse = textResponse ? `${textResponse}\n\n${text}` : text;
    } else if (answer.type === "tool_use") {
      functionCalls.push({
        name: answer.name,
        arguments: answer.input,
      });
    }
  }

  if (!textResponse && !functionCalls.length) {
    logger.error(id, "Missing text & functions in Anthropic API response:", data);
    throw new Error("Missing text & functions in Anthropic API response");
  }

  if (functionCalls.length > 1) {
    const allNames = functionCalls.map((fc) => fc.name).join(", ");
    const discarded = functionCalls.slice(1).map((fc) => `tool ${fc.name} with args ${JSON.stringify(fc.arguments)}`).join(", ");
    logger.warn(id, `got ${functionCalls.length} tool calls for tools ${allNames}. using tool ${functionCalls[0].name} with args ${JSON.stringify(functionCalls[0].arguments)} discarding ${discarded}`);
  }

  return {
    role: "assistant",
    content: textResponse,
    function_call: functionCalls[0] || null,
    files: [],
  };
}

async function callAnthropicWithRetries(
  id: Identifier,
  payload: AnthropicAIPayload,
  config?: AnthropicAIConfig,
  retries: number = 5
): Promise<ParsedResponseMessage> {
  return withRetries(id, "Anthropic", () => callAnthropic(id, payload, config), {
    retries,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// GOOGLE AI
// ─────────────────────────────────────────────────────────────────────────────

function jigGoogleMessages(messages: GoogleAIMessage[]): GoogleAIMessage[] {
  let jiggedMessages = messages.slice();

  // Ensure first message is from user
  if (jiggedMessages[0]?.role === "model") {
    jiggedMessages = [{ role: "user" as const, parts: [{ text: "..." }] }, ...jiggedMessages];
  }

  // Group consecutive messages with the same role
  jiggedMessages = jiggedMessages.reduce((acc, message) => {
    if (acc.length === 0) return [message];

    const lastMessage = acc[acc.length - 1];
    if (lastMessage.role === message.role) {
      lastMessage.parts = [...lastMessage.parts, ...message.parts];
      return acc;
    }

    return [...acc, message];
  }, [] as GoogleAIMessage[]);

  // Ensure last message is from user
  if (jiggedMessages[jiggedMessages.length - 1]?.role === "model") {
    jiggedMessages.push({ role: "user", parts: [{ text: "..." }] });
  }

  return jiggedMessages;
}

async function prepareGoogleAIPayload(
  _identifier: Identifier,
  payload: GenericPayload
): Promise<GoogleAIPayload> {
  const preparedPayload: GoogleAIPayload = {
    model: payload.model as GeminiModel,
    messages: [],
    tools: payload.functions
      ? {
          functionDeclarations: payload.functions.map((fn) => ({
            name: fn.name,
            parameters: {
              description: fn.description,
              ...fn.parameters,
            },
          })),
        }
      : undefined,
  };

  for (const message of payload.messages) {
    if (message.role === "system") {
      preparedPayload.systemInstruction = message.content;
      continue;
    }

    const parts: GoogleAIPart[] = [];

    if (message.content) {
      parts.push({ text: message.content });
    }

    for (const file of message.files || []) {
      if (ALLOWED_IMAGE_MIME_TYPES.includes(file.mimeType)) {
        if (file.url) {
          parts.push({
            inlineData: {
              mimeType: "image/png",
              data: await getNormalizedBase64PNG(file.url, file.mimeType),
            },
          });
          parts.push({ text: `Image (${file.url})` });
        } else if (file.data) {
          parts.push({
            inlineData: {
              mimeType: file.mimeType,
              data: file.data,
            },
          });
        }
      } else if (file.url) {
        // Non-image file with URL - add text reference
        parts.push({
          text: `File (${file.url})`,
        });
      }
    }

    preparedPayload.messages.push({
      role: message.role === "assistant" ? "model" : message.role,
      parts,
    });
  }

  return preparedPayload;
}

async function callGoogleAI(
  id: Identifier,
  payload: GoogleAIPayload
): Promise<ParsedResponseMessage> {
  const googleMessages = jigGoogleMessages(payload.messages);
  const history = googleMessages.slice(0, -1);
  const lastMessage = googleMessages.slice(-1)[0];

  const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

  const chat = genAI.chats.create({
    model: payload.model,
    history,
    config: {
      responseModalities: ["Text"],
      tools: payload.tools ? [payload.tools] : undefined,
      systemInstruction: payload.systemInstruction,
    },
  });

  const response = await chat.sendMessage({ message: lastMessage.parts });

  let text = "";
  const files: File[] = [];

  for (const part of response.candidates?.[0]?.content?.parts || []) {
    if (part.text) text += part.text;
    if (part.inlineData?.data) {
      files.push({ mimeType: "image/png", data: part.inlineData.data });
    }
  }

  const functionCalls = response.functionCalls?.map((fc) => ({
    name: fc.name ?? "",
    arguments: fc.args ?? {},
  }));

  if (functionCalls && functionCalls.length > 1) {
    const allNames = functionCalls.map((fc) => fc.name).join(", ");
    const discarded = functionCalls.slice(1).map((fc) => `tool ${fc.name} with args ${JSON.stringify(fc.arguments)}`).join(", ");
    logger.warn(id, `got ${functionCalls.length} tool calls for tools ${allNames}. using tool ${functionCalls[0].name} with args ${JSON.stringify(functionCalls[0].arguments)} discarding ${discarded}`);
  }

  if (!text && !functionCalls?.length && !files.length) {
    const candidate = response.candidates?.[0];
    const finishReason = candidate?.finishReason;

    logger.error(id, "Missing text & functions in Google AI API response:", {
      finishReason,
      safetyRatings: candidate?.safetyRatings,
      usageMetadata: response.usageMetadata,
      modelVersion: response.modelVersion,
      candidateContent: candidate?.content,
      promptFeedback: response.promptFeedback,
    });

    let errorMessage = "Missing text & functions in Google AI API response";
    if (finishReason) {
      const reasonDescriptions: Record<string, string> = {
        MALFORMED_FUNCTION_CALL: "(Google could not generate valid function call arguments)",
        SAFETY: "(blocked by safety filters)",
        RECITATION: "(blocked due to recitation)",
        MAX_TOKENS: "(response truncated due to max tokens)",
      };
      errorMessage += `: finishReason=${finishReason} ${reasonDescriptions[finishReason] || ""}`;
    }

    const error = new Error(errorMessage) as any;
    error.finishReason = finishReason;
    error.safetyRatings = candidate?.safetyRatings;
    error.usageMetadata = response.usageMetadata;
    error.promptFeedback = response.promptFeedback;
    throw error;
  }

  return {
    role: "assistant",
    content: text || null,
    files,
    function_call: functionCalls?.[0] || null,
  };
}

/**
 * Content violation finish reasons that should trigger circuit breaker behavior.
 * These errors won't resolve with simple retries - the content itself is the problem.
 */
const CONTENT_VIOLATION_REASONS = new Set([
  "PROHIBITED_CONTENT",
  "SAFETY",
]);

/**
 * Removes inline image data from Google AI messages, preserving text content.
 * Used as a fallback when content violations are detected.
 */
function removeImagesFromGooglePayload(payload: GoogleAIPayload): boolean {
  let removedImages = false;

  for (const message of payload.messages) {
    message.parts = message.parts.filter((part) => {
      if ("inlineData" in part) {
        removedImages = true;
        return false;
      }
      return true;
    });

    // Ensure message still has content after removing images
    if (message.parts.length === 0) {
      message.parts = [{ text: "(image removed due to content policy)" }];
    }
  }

  return removedImages;
}

async function callGoogleAIWithRetries(
  id: Identifier,
  payload: GoogleAIPayload,
  retries: number = 5
): Promise<ParsedResponseMessage> {
  let hasTriedWithoutImages = false;

  return withRetries(id, "Google AI", () => callGoogleAI(id, payload), {
    retries,
    onError: (error, attempt) => {
      const errorDetails: Record<string, any> = {
        message: error.message,
        finishReason: error.finishReason,
        modelVersion: error.modelVersion,
      };

      if (error.safetyRatings) errorDetails.safetyRatings = error.safetyRatings;
      if (error.usageMetadata) errorDetails.usageMetadata = error.usageMetadata;
      if (error.promptFeedback) errorDetails.promptFeedback = error.promptFeedback;
      if (error.status) errorDetails.httpStatus = error.status;
      if (error.code) errorDetails.errorCode = error.code;
      if (error.details) errorDetails.errorDetails = error.details;

      logger.error(id, `Retry #${attempt} error: ${error.message}`, errorDetails);

      // Circuit breaker: detect content violations and try removing images
      // Check both finishReason (candidate-level) and promptFeedback.blockReason (prompt-level)
      const violationReason =
        (CONTENT_VIOLATION_REASONS.has(error.finishReason) && error.finishReason) ||
        (CONTENT_VIOLATION_REASONS.has(error.promptFeedback?.blockReason) && error.promptFeedback?.blockReason);

      if (violationReason) {
        if (!hasTriedWithoutImages) {
          const removedImages = removeImagesFromGooglePayload(payload);
          if (removedImages) {
            logger.log(
              id,
              `Circuit breaker triggered: removing images due to ${violationReason}`
            );
            hasTriedWithoutImages = true;
            return; // Continue to next retry with images removed
          }
        }

        // If we already tried without images or there were no images, fail fast
        logger.error(
          id,
          `Circuit breaker: failing fast due to ${violationReason} (no more fallbacks)`
        );
        const circuitBreakerError = new Error(
          `Google AI content violation: ${violationReason}. Request cannot succeed with current content.`
        ) as any;
        circuitBreakerError.finishReason = error.finishReason;
        circuitBreakerError.safetyRatings = error.safetyRatings;
        circuitBreakerError.usageMetadata = error.usageMetadata;
        circuitBreakerError.circuitBreaker = true;
        throw circuitBreakerError;
      }
    },
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// GROQ
// ─────────────────────────────────────────────────────────────────────────────

function normalizeMessageContent(
  content: AnthropicAIMessage["content"]
): string {
  return Array.isArray(content)
    ? content
        .map((c) => (c.type === "text" ? c.text : `[${c.type}]`))
        .join("\n")
    : content;
}

function prepareGroqPayload(payload: GenericPayload): GroqPayload {
  return {
    model: payload.model as GroqModel,
    messages: payload.messages.map((message) => ({
      role: message.role,
      content: normalizeMessageContent(message.content),
    })),
    tools: payload.functions?.map((fn) => ({
      type: "function",
      function: fn,
    })),
    tool_choice: payload.function_call
      ? typeof payload.function_call === "string"
        ? payload.function_call
        : { type: "function", function: payload.function_call }
      : undefined,
    temperature: payload.temperature,
  };
}

async function callGroq(
  id: Identifier,
  payload: GroqPayload
): Promise<ParsedResponseMessage> {
  const response = await axios.post(
    "https://api.groq.com/openai/v1/chat/completions",
    payload,
    {
      headers: {
        "content-type": "application/json",
        Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
      },
    }
  );

  const answer = response.data.choices[0]?.message;
  if (!answer) {
    logger.error(id, "Missing answer in Groq API response:", response.data);
    throw new Error("Missing answer in Groq API");
  }

  let functionCall: FunctionCall | null = null;
  if (answer.tool_calls?.length) {
    const toolCall = answer.tool_calls[0];
    functionCall = {
      name: toolCall.function.name,
      arguments: JSON.parse(toolCall.function.arguments),
    };

    if (answer.tool_calls.length > 1) {
      const allNames = answer.tool_calls.map((tc: any) => tc.function.name).join(", ");
      const discarded = answer.tool_calls.slice(1).map((tc: any) => `tool ${tc.function.name} with args ${JSON.stringify(JSON.parse(tc.function.arguments))}`).join(", ");
      logger.warn(id, `got ${answer.tool_calls.length} tool calls for tools ${allNames}. using tool ${answer.tool_calls[0].function.name} with args ${JSON.stringify(JSON.parse(answer.tool_calls[0].function.arguments))} discarding ${discarded}`);
    }
  }

  return {
    role: "assistant",
    content: answer.content || null,
    function_call: functionCall,
    files: [],
  };
}

async function callGroqWithRetries(
  id: Identifier,
  payload: GroqPayload,
  retries: number = 5
): Promise<ParsedResponseMessage> {
  return withRetries(id, "Groq", () => callGroq(id, payload), { retries });
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN ENTRY POINT
// ─────────────────────────────────────────────────────────────────────────────

function isAnthropicPayload(payload: GenericPayload): boolean {
  return Object.values(ClaudeModel).includes(payload.model as ClaudeModel);
}

function isOpenAiPayload(payload: GenericPayload): boolean {
  return Object.values(GPTModel).includes(payload.model as GPTModel);
}

function isGroqPayload(payload: GenericPayload): boolean {
  return Object.values(GroqModel).includes(payload.model as GroqModel);
}

function isGoogleAIPayload(payload: GenericPayload): boolean {
  return Object.values(GeminiModel).includes(payload.model as GeminiModel);
}

export async function callWithRetries(
  id: string | string[],
  aiPayload: GenericPayload,
  aiConfig?: OpenAIConfig | AnthropicAIConfig,
  retries: number = 5,
  chunkTimeoutMs: number = 15_000
): Promise<ParsedResponseMessage> {
  try {
    if (isAnthropicPayload(aiPayload)) {
      return await callAnthropicWithRetries(
        id,
        await prepareAnthropicPayload(id, aiPayload),
        aiConfig as AnthropicAIConfig,
        retries
      );
    }

    if (isOpenAiPayload(aiPayload)) {
      return await callOpenAiWithRetries(
        id,
        await prepareOpenAIPayload(id, aiPayload),
        aiConfig as OpenAIConfig,
        retries,
        chunkTimeoutMs
      );
    }

    if (isGroqPayload(aiPayload)) {
      return await callGroqWithRetries(id, prepareGroqPayload(aiPayload), retries);
    }

    if (isGoogleAIPayload(aiPayload)) {
      return await callGoogleAIWithRetries(
        id,
        await prepareGoogleAIPayload(id, aiPayload),
        retries
      );
    }

    throw new Error("Invalid AI payload: Unknown model type.");
  } catch (error) {
    if (aiPayload.fallbackModel) {
      logger.log(
        id,
        `Primary model ${aiPayload.model} failed, falling back to ${aiPayload.fallbackModel}`
      );
      return callWithRetries(
        id,
        { ...aiPayload, model: aiPayload.fallbackModel, fallbackModel: undefined },
        aiConfig,
        retries,
        chunkTimeoutMs
      );
    }
    throw error;
  }
}
