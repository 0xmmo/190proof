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
} from "./interfaces";
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import axios from "axios";
import { isHeicImage, timeout } from "./utils";
const { GoogleGenerativeAI } = require("@google/generative-ai");

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
} from "./interfaces";

function parseStreamedResponse(
  identifier: string,
  paragraph: string,
  functionCallName: string,
  functionCallArgs: string,
  allowedFunctionNames: Set<string> | null
): ParsedResponseMessage {
  let functionCall: ParsedResponseMessage["function_call"] = null;
  if (functionCallName && functionCallArgs) {
    if (allowedFunctionNames && !allowedFunctionNames.has(functionCallName)) {
      throw new Error(
        "Stream error: received function call with unknown name: " +
          functionCallName
      );
    }

    try {
      functionCall = {
        name: functionCallName,
        arguments: JSON.parse(functionCallArgs),
      };
    } catch (error) {
      console.error("Error parsing functionCallArgs:", functionCallArgs);
      throw error;
    }
  }

  if (!paragraph && !functionCall) {
    console.error(
      identifier,
      "Stream error: received message without content or function_call, raw:",
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
  };
}

async function callOpenAiWithRetries(
  identifier: string,
  openAiPayload: OpenAIPayload,
  openAiConfig?: OpenAIConfig,
  retries: number = 5,
  chunkTimeoutMs: number = 15_000
): Promise<ParsedResponseMessage> {
  console.log(
    identifier,
    "Calling OpenAI API with retries:",
    openAiConfig?.service,
    openAiPayload.model
  );

  let errorObj: any;
  for (let i = 0; i <= retries; i++) {
    try {
      const timerId = `timer:${identifier}:${Date.now()}:callOpenAi:${
        openAiConfig?.service
      }-${openAiPayload.model}-${openAiConfig?.orgId}`;
      const res = await callOpenAIStream(
        identifier,
        openAiPayload,
        openAiConfig,
        chunkTimeoutMs
      );
      return res;
    } catch (error: any) {
      console.error(error);
      console.error(
        identifier,
        `Retrying due to error: received bad response from OpenAI API [${
          openAiConfig?.service
        }-${openAiPayload.model}-${openAiConfig?.orgId}]: ${
          error.message
        } - ${JSON.stringify(error.response?.data)}`
      );

      const errorCode = error.data?.code;

      if (errorCode) {
        console.error(
          identifier,
          `Retry #${i} failed with API error: ${errorCode}`,
          JSON.stringify({
            data: error.data,
          })
        );
      }

      // to solve context length issue or JSON parsing error due to truncated response
      openAiPayload.model = GPTModel.GPT4_0409; // TODO: Remove this
      openAiPayload.temperature = 0.8; // Higher temperature

      // Usually due to image content, we get a policy violation error
      if (errorCode === "content_policy_violation") {
        console.log(
          identifier,
          `Switching to OpenAI service due to content policy violation error`
        );
        openAiPayload.messages.forEach((message: OpenAIMessage) => {
          if (Array.isArray(message.content)) {
            message.content = message.content.filter(
              (content) => content.type === "text"
            );
          }
        });
      }

      // on 2nd or more retries
      // if Azure content policy error is persistent
      if (
        i >= 2 &&
        openAiConfig?.service === "azure" &&
        errorCode === "content_filter"
      ) {
        console.log(
          identifier,
          `Switching to OpenAI service due to content filter error`
        );
        openAiConfig.service = "openai"; // Move to OpenAI, failed due to Azure content policy
      }

      // on 3rd retry
      if (i === 3) {
        if (openAiConfig?.service === "azure") {
          console.log(
            identifier,
            `Switching to OpenAI service due to Azure service error`
          );
          openAiConfig.service = "openai";
        }
      }

      // on 4th retry
      if (i === 4) {
        // abort function calling, e.g. stubborn `python` function call case
        if (openAiPayload.tools) {
          console.log(
            identifier,
            `Switching to no tool choice due to persistent error`
          );
          openAiPayload.tool_choice = "none";
        }
      }

      await timeout(250);
    }
  }

  console.error(
    identifier,
    `Failed to call OpenAI API after ${retries} attempts. Please lookup OpenAI status for active issues.`,
    errorObj
  );
  throw new Error(
    `${identifier}: Failed to call OpenAI API after ${retries} attempts. Please lookup OpenAI status for active issues.`
  );
}

async function callOpenAIStream(
  identifier: string,
  openAiPayload: OpenAIPayload,
  openAiConfig: OpenAIConfig | undefined,
  chunkTimeoutMs: number
): Promise<ParsedResponseMessage> {
  const functionNames: Set<string> | null = openAiPayload.tools
    ? new Set(openAiPayload.tools.map((fn) => fn.function.name as string))
    : null;

  if (!openAiConfig) {
    openAiConfig = {
      service: "openai",
      apiKey: process.env.OPENAI_API_KEY as string,
      baseUrl: "",
    };
  }

  let response;
  const controller = new AbortController();
  if (openAiConfig.service === "azure") {
    console.log(identifier, "Using Azure OpenAI service", openAiPayload.model);
    const model = openAiPayload.model;

    if (!openAiConfig.modelConfigMap) {
      throw new Error(
        "OpenAI config modelConfigMap is required when using Azure OpenAI service."
      );
    }

    const azureConfig = openAiConfig.modelConfigMap[model];
    let endpoint;
    if (azureConfig.endpoint) {
      endpoint = `${azureConfig.endpoint}/openai/deployments/${azureConfig.deployment}/chat/completions?api-version=${azureConfig.apiVersion}`;
    } else {
      throw new Error("Azure OpenAI endpoint is required in modelConfigMap.");
    }
    console.log(identifier, "Using endpoint", endpoint);

    try {
      const stringifiedPayload = JSON.stringify({
        ...openAiPayload,
        stream: true,
      });
      const parsedPayload = JSON.parse(stringifiedPayload);
    } catch (error) {
      console.error(
        identifier,
        "Stream error: Azure OpenAI JSON parsing error:",
        JSON.stringify(error)
      );
    }

    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "api-key": azureConfig.apiKey,
      },
      body: JSON.stringify({
        ...openAiPayload,
        stream: true,
      }),
      signal: controller.signal,
    });
  } else {
    // openai by default
    console.log(identifier, "Using OpenAI service", openAiPayload.model);
    const endpoint = `https://api.openai.com/v1/chat/completions`;
    if (openAiConfig.orgId) {
      console.log(identifier, "Using orgId", openAiConfig.orgId);
    }

    response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${openAiConfig.apiKey}`,
        ...(openAiConfig.orgId
          ? { "OpenAI-Organization": openAiConfig.orgId }
          : {}),
      },
      body: JSON.stringify({
        ...openAiPayload,
        stream: true,
      }),
      signal: controller.signal,
    });
  }

  if (response.body) {
    let rawStreamedBody = "";
    let paragraph = "";
    let functionCallName = "";
    let functionCallArgs = "";

    const reader = response.body.getReader();

    let partialChunk = "";
    let abortTimeout: NodeJS.Timeout;
    const startAbortTimeout = () => {
      clearTimeout(abortTimeout);
      return setTimeout(() => {
        console.log(
          identifier,
          `Stream error: aborted due to timeout after ${chunkTimeoutMs} ms.`,
          JSON.stringify({ paragraph })
        );
        controller.abort();
      }, chunkTimeoutMs);
    };

    let chunkIndex = -1;
    while (true) {
      chunkIndex++;
      const abortTimeout = startAbortTimeout();
      const { done, value } = await reader.read();
      clearTimeout(abortTimeout);

      if (done) {
        console.log(
          identifier,
          `Stream error: ended after ${
            chunkIndex + 1
          } chunks via reader done flag.`,
          rawStreamedBody
        );
        throw new Error("Stream error: ended prematurely");
      }

      let chunk = new TextDecoder().decode(value);
      rawStreamedBody += chunk + "\n";
      if (partialChunk) {
        chunk = partialChunk + chunk;
        partialChunk = "";
      }
      let jsonStrings = chunk.split(/^data: /gm);

      for (let jsonString of jsonStrings) {
        if (!jsonString) {
          continue;
        }

        if (jsonString.includes("[DONE]")) {
          console.log(
            identifier,
            `Stream explicitly marked as done after ${chunkIndex + 1} chunks.`
          );
          try {
            return parseStreamedResponse(
              identifier,
              paragraph,
              functionCallName,
              functionCallArgs,
              functionNames
            );
          } catch (error) {
            console.error(
              identifier,
              "Stream error: parsing response:",
              rawStreamedBody
            );
            throw error;
          }
        }

        let json;
        try {
          json = JSON.parse(jsonString.trim());
        } catch (error: any) {
          partialChunk = jsonString; // We're assuming any JSON parsing error means we got a non-terminated JSON for a chunk
          continue;
        }

        if (!json.choices || !json.choices.length) {
          if (json.error) {
            console.error(
              identifier,
              "Stream error: OpenAI error:",
              json.error && JSON.stringify(json.error)
            );
            const error = new Error("Stream error: OpenAI error") as any;
            error.data = json.error;
            error.requestBody = openAiPayload;
            throw error;
          }
          if (chunkIndex !== 0)
            console.error(
              identifier,
              "Stream error: no choices in JSON:",
              json
            ); // bad if it's not the first chunk
          continue;
        }

        const dToolCall:
          | {
              index?: number;
              function?: {
                name?: string;
                arguments?: string;
              };
            }
          | undefined = json.choices?.[0]?.delta?.tool_calls?.[0];
        if (dToolCall) {
          const toolCallIndex = dToolCall.index || 0;
          // TODO: handle multiple function calls in response
          if (toolCallIndex === 0) {
            const dFn = dToolCall.function || {};
            if (dFn.name) functionCallName += dFn.name;
            if (dFn.arguments) functionCallArgs += dFn.arguments;
          }
        }

        const text = json.choices?.[0]?.delta?.content;
        if (text) {
          paragraph += text;
        }
      }
    }
  } else {
    throw new Error("Stream error: no response body");
  }
}

async function callAnthropicWithRetries(
  identifier: string,
  AiPayload: AnthropicAIPayload,
  AiConfig?: AnthropicAIConfig,
  attempts = 5
): Promise<ParsedResponseMessage> {
  console.log(identifier, "Calling Anthropic API with retries");
  let lastResponse;
  for (let i = 0; i < attempts; i++) {
    // if last attempt
    if (i === attempts - 1) {
      AiPayload.model = ClaudeModel.SONNET; // fallback to sonnet model
    }

    try {
      // return await callAnthropic(identifier, AiPayload);
      lastResponse = await callAnthropic(identifier, AiPayload, AiConfig);
      return lastResponse;
    } catch (e: any) {
      console.error(e);
      console.error(
        identifier,
        `Retrying due to error: received bad response from Anthropic API: ${e.message}`,
        JSON.stringify(e.response?.data)
      );

      if (e.response?.data?.error?.type === "rate_limit_error") {
        // upgrade to Sonnet
        AiPayload.model = ClaudeModel.SONNET;
      }

      await timeout(125 * i);
    }
  }
  const error = new Error(
    `Failed to call Anthropic API after ${attempts} attempts`
  ) as any;
  error.response = lastResponse;
  throw error;
}

async function callAnthropic(
  identifier: string,
  AiPayload: AnthropicAIPayload,
  AiConfig?: AnthropicAIConfig
): Promise<ParsedResponseMessage> {
  const anthropicMessages = jigAnthropicMessages(AiPayload.messages);

  let data;
  let response;
  if (AiConfig?.service === "bedrock") {
    // DOES NOT SUPPORT TOOLS YET
    const AWS_REGION = "us-east-1";
    const MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0";

    // set in environment
    // process.env.AWS_ACCESS_KEY_ID = AWS_ACCESS_KEY_ID;
    // process.env.AWS_SECRET_ACCESS_KEY = AWS_SECRET_ACCESS_KEY;

    const client = new BedrockRuntimeClient({ region: AWS_REGION });
    const payload = {
      anthropic_version: "bedrock-2023-05-31",
      max_tokens: 4096,
      messages: anthropicMessages,
      tools: AiPayload.functions?.map((f) => ({
        ...f,
        input_schema: f.parameters,
        parameters: undefined,
      })),
    };

    const response = await client.send(
      new InvokeModelCommand({
        contentType: "application/json",
        body: JSON.stringify(payload),
        modelId: MODEL_ID,
      })
    );

    const decodedResponseBody = new TextDecoder().decode(response.body);
    data = JSON.parse(decodedResponseBody);
  } else {
    // default to anthropic
    const response = await axios.post(
      "https://api.anthropic.com/v1/messages",
      {
        model: AiPayload.model,
        messages: anthropicMessages,
        tools: AiPayload.functions?.map((f) => ({
          ...f,
          input_schema: f.parameters,
          parameters: undefined,
        })),
        temperature: AiPayload.temperature,
        system: AiPayload.system,
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

  if (!answers[0]) {
    console.error(identifier, "Missing answer in Anthropic API:", data);
    throw new Error("Missing answer in Anthropic API");
  }

  let textResponse = "";
  let functionCalls: any[] = [];
  for (const answer of answers) {
    if (!answer.type) {
      console.error(identifier, "Missing answer type in Anthropic API:", data);
      throw new Error("Missing answer type in Anthropic API");
    }

    let text = "";
    if (answer.type === "text") {
      text = answer.text
        .replace(/<thinking>.*?<\/thinking>/gs, "")
        .replace(/<answer>|<\/answer>/gs, "")
        .trim();

      if (!text) {
        // remove the tags and return the text within
        text = answer.text.replace(
          /<thinking>|<\/thinking>|<answer>|<\/answer>/gs,
          ""
        );
        console.log("No text in answer, returning text within tags:", text);
      }

      if (textResponse) {
        textResponse += `\n\n${text}`;
      } else {
        textResponse = text;
      }
    } else if (answer.type === "tool_use") {
      const call = {
        name: answer.name,
        arguments: answer.input,
      };
      functionCalls.push(call);
    }
  }

  if (!textResponse && !functionCalls.length) {
    console.error(
      identifier,
      "Missing text & fns in Anthropic API response:",
      JSON.stringify(data)
    );
    throw new Error("Missing text & fns in Anthropic API response");
  }

  return {
    role: "assistant",
    content: textResponse,
    function_call: functionCalls[0],
  };
}

function jigAnthropicMessages(
  messages: AnthropicAIMessage[]
): AnthropicAIMessage[] {
  // Takes a list if messages each with a role and content
  // Assumes no system messages are present

  let jiggedMessages = messages.slice();

  // If the first message is not user, add an empty user message at the start
  if (jiggedMessages[0]?.role !== "user") {
    jiggedMessages = [
      {
        role: "user" as const,
        content: "...",
      },
      ...jiggedMessages,
    ];
  }

  // Group consecutive messages with the same role, combining their content
  jiggedMessages = jiggedMessages.reduce((acc, message) => {
    if (acc.length === 0) {
      return [message];
    }

    const lastMessage = acc[acc.length - 1];
    if (lastMessage.role === message.role) {
      // Combine content of messages with the same role
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

  // If last message in array is assistant, then add an empty user message
  if (jiggedMessages[jiggedMessages.length - 1]?.role === "assistant") {
    jiggedMessages.push({
      role: "user",
      content: "...",
    });
  }

  return jiggedMessages;
}

async function prepareGoogleAIPayload(
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
              // Google puts their description in the parameters object rather than in a top-level field
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

    const googleAIContentParts: GoogleAIPart[] = [];

    if (message.content) {
      googleAIContentParts.push({
        text: message.content,
      });
    }

    for (const file of message.files || []) {
      if (!file.mimeType?.startsWith("image")) {
        console.warn(
          "Google AI API does not support non-image file types. Skipping file."
        );
        continue;
      }

      if (file.url) {
        googleAIContentParts.push({
          inlineData: {
            mimeType: "image/png",
            data: await getNormalizedBase64PNG(file.url, file.mimeType),
          },
        });
      } else if (file.data) {
        if (
          !["image/png", "image/jpeg", "image/gif", "image/webp"].includes(
            file.mimeType
          )
        ) {
          throw new Error(
            "Invalid image mimeType. Supported types are: image/png, image/jpeg, image/gif, image/webp"
          );
        }
        googleAIContentParts.push({
          inlineData: {
            mimeType: file.mimeType,
            data: file.data,
          },
        });
      }
    }

    preparedPayload.messages.push({
      role: message.role === "assistant" ? "model" : message.role,
      parts: googleAIContentParts,
    });
  }

  return preparedPayload;
}

async function callGoogleAI(
  identifier: string,
  payload: GoogleAIPayload
): Promise<ParsedResponseMessage> {
  console.log(identifier, "Calling Google AI API");

  const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({
    model: payload.model,
    tools: payload.tools,
    systemInstruction: payload.systemInstruction,
  });

  const history = payload.messages.slice(0, -1);
  const lastMessage = payload.messages.slice(-1)[0];
  const chat = model.startChat({
    history,
  });

  const result = await chat.sendMessage(lastMessage.parts);
  const response = await result.response;

  const text: string | undefined = response.text();
  const functionCalls:
    | {
        name: string;
        args: Record<string, any>;
      }[]
    | undefined = response.functionCalls();

  const parsedFunctionCalls = functionCalls?.map((fc) => ({
    name: fc.name,
    arguments: fc.args,
  }));

  return {
    role: "assistant",
    content: text || null,
    function_call: parsedFunctionCalls?.[0] || null,
  };
}

export async function callWithRetries(
  identifier: string,
  aiPayload: GenericPayload,
  aiConfig?: OpenAIConfig | AnthropicAIConfig,
  retries: number = 5,
  chunkTimeoutMs: number = 15_000
): Promise<ParsedResponseMessage> {
  // Determine which service to use based on the model type
  if (isAnthropicPayload(aiPayload)) {
    console.log(identifier, "Delegating call to Anthropic API");

    return await callAnthropicWithRetries(
      identifier,
      await prepareAnthropicPayload(aiPayload),
      aiConfig as AnthropicAIConfig,
      retries
    );
  } else if (isOpenAiPayload(aiPayload)) {
    console.log(identifier, "Delegating call to OpenAI API");
    return await callOpenAiWithRetries(
      identifier,
      await prepareOpenAIPayload(aiPayload),
      aiConfig as OpenAIConfig,
      retries,
      chunkTimeoutMs
    );
  } else if (isGroqPayload(aiPayload)) {
    console.log(identifier, "Delegating call to Groq API");
    return await callGroqWithRetries(
      identifier,
      await prepareGroqPayload(aiPayload)
    );
  } else if (isGoogleAIPayload(aiPayload)) {
    console.log(identifier, "Delegating call to Google AI API");
    return await callGoogleAI(
      identifier,
      await prepareGoogleAIPayload(aiPayload)
    );
  } else {
    throw new Error("Invalid AI payload: Unknown model type.");
  }
}

function isAnthropicPayload(payload: any): Boolean {
  return Object.values(ClaudeModel).includes(payload.model);
}

async function prepareAnthropicPayload(
  payload: GenericPayload
): Promise<AnthropicAIPayload> {
  const preparedPayload: AnthropicAIPayload = {
    model: payload.model as ClaudeModel,
    messages: [],
    functions: payload.functions,
    temperature: payload.temperature,
  };

  for (const message of payload.messages) {
    const anthropicContentBlocks: AnthropicContentBlock[] = [];

    if (message.role === "system") {
      preparedPayload.system = message.content;
      continue;
    }

    if (message.content) {
      anthropicContentBlocks.push({
        type: "text",
        text: message.content,
      });
    }

    for (const file of message.files || []) {
      if (!file.mimeType?.startsWith("image")) {
        console.warn(
          "Anthropic API does not support non-image file types. Skipping file."
        );
        continue;
      }

      if (file.url) {
        anthropicContentBlocks.push({
          type: "image",
          source: {
            type: "base64",
            media_type: "image/png",
            data: await getNormalizedBase64PNG(file.url, file.mimeType),
          },
        });
      } else if (file.data) {
        if (
          !["image/png", "image/jpeg", "image/gif", "image/webp"].includes(
            file.mimeType
          )
        ) {
          throw new Error(
            "Invalid image mimeType. Supported types are: image/png, image/jpeg, image/gif, image/webp"
          );
        }
        anthropicContentBlocks.push({
          type: "image",
          source: {
            type: "base64",
            media_type: file.mimeType as any,
            data: file.data,
          },
        });
      }
    }

    preparedPayload.messages.push({
      role: message.role,
      content: anthropicContentBlocks,
    });
  }

  return preparedPayload;
}

function isOpenAiPayload(payload: any): Boolean {
  return Object.values(GPTModel).includes(payload.model);
}

async function prepareOpenAIPayload(
  payload: GenericPayload
): Promise<OpenAIPayload> {
  const preparedPayload: OpenAIPayload = {
    model: payload.model as GPTModel,
    temperature: payload.temperature,
    messages: [],
    tools: payload.functions?.map((fn) => ({
      type: "function",
      function: fn,
    })),
    tool_choice: payload.function_call
      ? typeof payload.function_call === "string"
        ? payload.function_call // "none" | "auto"
        : {
            type: "function",
            function: payload.function_call,
          }
      : undefined,
  };

  for (const message of payload.messages) {
    const openAIContentBlocks: OpenAIContentBlock[] = [];

    if (message.content) {
      openAIContentBlocks.push({
        type: "text",
        text: message.content,
      });
    }

    for (const file of message.files || []) {
      if (file.mimeType?.startsWith("image")) {
        if (file.url) {
          openAIContentBlocks.push({
            type: "image_url",
            image_url: {
              url: file.url,
            },
          });
        } else if (file.data) {
          openAIContentBlocks.push({
            type: "image_url",
            image_url: {
              url: `data:${file.mimeType};base64,${file.data}`,
            },
          });
        }
        // } else if (file.mimeType?.startsWith("audio")) {
        //   if (file.url) {
        //     openAIContentBlocks.push({
        //       type: "audio_url",
        //       audio_url: {
        //         url: file.url,
        //       },
        //     });
        //   } else if (file.data) {
        //     openAIContentBlocks.push({
        //       type: "audio_url",
        //       audio_url: {
        //         url: `data:${file.mimeType};base64,${file.data}`,
        //       },
        //     });
        //   }
      } else {
        console.warn(
          "Skipping file. Type not supported by OpenAI API:",
          file.mimeType
        );
      }
    }

    preparedPayload.messages.push({
      role: message.role,
      content: openAIContentBlocks,
    });
  }

  return preparedPayload;
}

function isGroqPayload(payload: any): Boolean {
  return Object.values(GroqModel).includes(payload.model);
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
        ? payload.function_call // "none" | "auto"
        : {
            type: "function",
            function: payload.function_call,
          }
      : undefined,
    temperature: payload.temperature,
  };
}

function normalizeMessageContent(
  content: AnthropicAIMessage["content"]
): string {
  return Array.isArray(content)
    ? content
        .map((c) => (c.type === "text" ? c.text : `[${c.type}]`))
        .join("\n")
    : content;
}

function isGoogleAIPayload(payload: any): Boolean {
  return Object.values(GeminiModel).includes(payload.model);
}

async function callGroq(
  identifier: string,
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

  const data = response.data;

  const answer = data.choices[0].message;
  if (!answer) {
    console.error(identifier, "Missing answer in Groq API:", data);
    throw new Error("Missing answer in Groq API");
  }

  const textResponse = answer.content || null;
  let functionCall: FunctionCall | null = null;
  if (answer.tool_calls && answer.tool_calls.length) {
    const toolCall = answer.tool_calls[0];
    functionCall = {
      name: toolCall.function.name,
      arguments: JSON.parse(toolCall.function.arguments),
    };
  }

  return {
    role: "assistant",
    content: textResponse,
    function_call: functionCall,
  };
}

async function callGroqWithRetries(
  identifier: string,
  payload: GroqPayload,
  retries: number = 5
): Promise<ParsedResponseMessage> {
  console.log(identifier, "Calling Groq API with retries");

  let lastResponse;
  for (let i = 0; i < retries; i++) {
    try {
      lastResponse = await callGroq(identifier, payload);
      return lastResponse;
    } catch (e: any) {
      console.error(e);
      console.error(
        identifier,
        `Retrying due to error: received bad response from Groq API: ${e.message}`,
        JSON.stringify(e.response?.data)
      );

      await timeout(125 * i);
    }
  }
  const error = new Error(
    `Failed to call Groq API after ${retries} attempts`
  ) as any;
  error.response = lastResponse;
  throw error;
}

async function getNormalizedBase64PNG(
  url: string,
  mime: string
): Promise<string> {
  console.log("Normalizing image", url);
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

// async function main() {
//   const payload: GenericPayload = {
//     model: GeminiModel.GEMINI_15_PRO,
//     messages: [
//       {
//         role: "user",
//         content: "What is this logo?",
//         files: [
//           {
//             mimeType: "image/png",
//             url: "https://www.wikimedia.org/static/images/wmf-logo-2x.png",
//           },
//         ],
//       },
//     ],
//     functions: [
//       {
//         name: "answer_logo_question",
//         description: "Answer a question about a logo",
//         parameters: {
//           type: "object",
//           properties: {
//             organization: {
//               type: "string",
//             },
//           },
//         },
//       },
//     ],
//   };

//   const answer = await callWithRetries("test", payload);

//   console.log(answer);
// }

// main();
