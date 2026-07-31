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
  OpenRouterPayload,
  OpenRouterModel,
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
  Provider,
} from "./interfaces";
import logger, { Identifier } from "./logger";
import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import axios from "axios";
import { isHeicImage, timeout } from "./utils";

const sharp = require("sharp");
const decode = require("heic-decode");

export {
  ClaudeModel,
  GPTModel,
  GroqModel,
  GeminiModel,
  OpenRouterModel,
  OpenAIConfig,
  FunctionDefinition,
  FunctionCall,
  GenericMessage,
  GenericPayload,
  ToolResult,
  ParsedResponseMessage,
  OpenRouterProviderPreferences,
  AnyModel,
  Provider,
} from "./interfaces";

// ─────────────────────────────────────────────────────────────────────────────
// SHARED UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

/**
 * `AbortSignal.any` exists at runtime (Node 18.17+/20.3+) but isn't in the
 * pinned @types/node lib; centralize the cast so call sites stay typed. Used to
 * merge a caller's cancellation signal with an adapter's internal timeout
 * controller without clobbering either.
 */
function anySignal(signals: AbortSignal[]): AbortSignal {
  return (AbortSignal as unknown as { any(s: AbortSignal[]): AbortSignal }).any(
    signals,
  );
}

/**
 * Run an axios-based provider call under a wall-clock deadline. Axios's
 * `timeout` option is a socket *idle* timer — providers that dribble
 * keep-alive bytes while a long generation runs (OpenRouter does this
 * explicitly) reset it forever, so a stuck upstream holds the request open
 * until the caller's walltime kills the whole turn (2026-07-26: a 9.7-minute
 * OpenRouter call with requestTimeoutMs=120s "set"). AbortSignal.timeout
 * bounds elapsed time instead; the idle `timeout` stays as a faster trigger
 * for fully dead sockets.
 */
async function withRequestDeadline<T>(
  apiName: string,
  requestTimeoutMs: number,
  signal: AbortSignal | undefined,
  fn: (mergedSignal: AbortSignal) => Promise<T>,
): Promise<T> {
  const deadline = AbortSignal.timeout(requestTimeoutMs);
  try {
    return await fn(signal ? anySignal([signal, deadline]) : deadline);
  } catch (error) {
    // Axios surfaces any abort as a bare "canceled" — restore the real reason
    // so retry logs and model-facing errors say what actually happened.
    if (deadline.aborted && !signal?.aborted) {
      throw new Error(
        `${apiName} request exceeded hard deadline of ${requestTimeoutMs}ms`,
      );
    }
    throw error;
  }
}

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
    signal?: AbortSignal;
  } = {},
): Promise<T> {
  const { retries = 5, baseDelayMs = 125, onError } = options;

  logger.log(identifier, `Calling ${apiName} API with retries`);

  let lastError: any;
  for (let attempt = 0; attempt < retries; attempt++) {
    try {
      return await fn();
    } catch (error: any) {
      lastError = error;

      // Caller cancelled — reject immediately, never retry. Keyed on
      // signal.aborted (not error.name/code) because some adapters re-wrap the
      // underlying abort error and lose its name (e.g. the Google adapter).
      if (options.signal?.aborted) throw error;

      if (onError) {
        onError(error, attempt);
      } else {
        logger.error(
          identifier,
          `Retry #${attempt} error: ${error.message}`,
          error.response?.data || error,
        );
      }

      await timeout(baseDelayMs * attempt);
    }
  }

  // Fold the underlying cause into .message so it survives callers that read
  // error.message (and drop error.cause) — otherwise the raw reason (timeout,
  // provider 4xx body, etc.) never reaches the agent's model-facing error. Prefer
  // the provider's response body, which is richer than axios's "Request failed
  // with status code N".
  const detail =
    lastError?.response?.data?.error?.message ||
    lastError?.message ||
    String(lastError);
  const error = new Error(
    `Failed to call ${apiName} API after ${retries} attempts: ${detail}`,
  ) as any;
  error.cause = lastError;
  throw error;
}

function parseStreamedResponse(
  identifier: Identifier,
  paragraph: string,
  toolCallAccumulators: { id?: string; name: string; arguments: string }[],
  allowedFunctionNames: Set<string> | null,
  reasoning?: string,
): ParsedResponseMessage {
  const functionCalls: FunctionCall[] = [];

  for (let i = 0; i < toolCallAccumulators.length; i++) {
    const acc = toolCallAccumulators[i];
    if (!acc.name || !acc.arguments) continue;

    if (allowedFunctionNames && !allowedFunctionNames.has(acc.name)) {
      throw new Error(
        `Stream error: received function call with unknown name: ${acc.name}`,
      );
    }

    try {
      functionCalls.push({
        id: acc.id || `call_${i}`,
        name: acc.name,
        arguments: JSON.parse(acc.arguments),
      });
    } catch (error) {
      logger.error(
        identifier,
        "Error parsing function call arguments:",
        acc.arguments,
      );
      throw error;
    }
  }

  if (!paragraph && !functionCalls.length) {
    logger.error(
      identifier,
      "Stream error: received message without content or function_call:",
      JSON.stringify({ paragraph, toolCallAccumulators }),
    );
    throw new Error(
      "Stream error: received message without content or function_call",
    );
  }

  return {
    role: "assistant",
    content: paragraph || null,
    function_call: functionCalls[0] || null,
    function_calls: functionCalls,
    files: [],
    reasoning: reasoning || undefined,
    usage: null,
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
    2,
  );
}

async function getNormalizedBase64PNG(
  url: string,
  mime: string,
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
  config: OpenAIConfig | undefined,
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
        "OpenAI config modelConfigMap is required when using Azure OpenAI service.",
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

  // Default: OpenAI (or any OpenAI-compatible server via config.baseUrl)
  logger.log(identifier, "Using OpenAI service:", model);
  if (config.orgId) {
    logger.log(identifier, "Using orgId:", config.orgId);
  }

  const base = (config.baseUrl?.trim() || "https://api.openai.com/v1").replace(
    /\/$/,
    "",
  );
  return {
    endpoint: `${base}/chat/completions`,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.apiKey}`,
      ...(config.orgId ? { "OpenAI-Organization": config.orgId } : {}),
    },
  };
}

/**
 * Keep only OpenAI-compat `reasoning.*` blocks when echoing reasoning_details.
 * A cross-provider fallback replays reasoningDetails captured from Anthropic
 * (`thinking`/`redacted_thinking`), which don't belong in reasoning_details.
 * Returns undefined when nothing native-shaped remains.
 */
function filterOpenAICompatReasoningDetails(details: any): any[] | undefined {
  if (!Array.isArray(details)) return details || undefined;
  const blocks = details.filter(
    (block: any) =>
      typeof block?.type === "string" && block.type.startsWith("reasoning."),
  );
  return blocks.length ? blocks : undefined;
}

async function prepareOpenAIPayload(
  identifier: Identifier,
  payload: GenericPayload,
): Promise<OpenAIPayload> {
  const preparedPayload: OpenAIPayload = {
    model: payload.model as GPTModel,
    messages: [],
    reasoning_effort: payload.reasoningEffort,
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
    // role:"tool" → one OpenAI tool message per result (parallel calls expand).
    if (message.role === "tool") {
      for (const tr of message.toolResults || []) {
        preparedPayload.messages.push({
          role: "tool",
          tool_call_id: tr.toolCallId,
          content: tr.content,
        });
      }
      continue;
    }

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

    const outMessage: OpenAIMessage = {
      role: message.role,
      // OpenAI wants null (not []) content on a tool-call-only assistant turn.
      content: contentBlocks.length ? contentBlocks : null,
    };
    if (message.functionCalls?.length) {
      outMessage.tool_calls = message.functionCalls.map((fc, i) => ({
        id: fc.id ?? `call_${i}`,
        type: "function" as const,
        function: {
          name: fc.name,
          arguments: JSON.stringify(fc.arguments),
        },
      }));
    }
    // Reasoning passthrough — only when the caller supplied it (never injected).
    if (message.reasoning) outMessage.reasoning = message.reasoning;
    const reasoningDetails = filterOpenAICompatReasoningDetails(
      message.reasoningDetails,
    );
    if (reasoningDetails) outMessage.reasoning_details = reasoningDetails;
    preparedPayload.messages.push(outMessage);
  }

  return preparedPayload;
}

async function callOpenAIStream(
  id: Identifier,
  openAiPayload: OpenAIPayload,
  openAiConfig: OpenAIConfig | undefined,
  chunkTimeoutMs: number,
  requestTimeoutMs: number = 120_000,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  const functionNames: Set<string> | null = openAiPayload.tools
    ? new Set(openAiPayload.tools.map((fn) => fn.function.name as string))
    : null;

  const { endpoint, headers } = buildOpenAIRequestConfig(
    id,
    openAiPayload.model,
    openAiConfig,
  );

  const controller = new AbortController();
  // Overall per-attempt deadline (separate from the per-chunk stall timeout
  // below): aborts the same controller if the whole stream runs past the
  // budget. unref'd so it never holds the event loop; cleared on the success
  // path, and a no-op abort on a finished stream otherwise.
  const overallTimeout = setTimeout(() => {
    logger.error(id, `Request timeout after ${requestTimeoutMs}ms`);
    controller.abort();
  }, requestTimeoutMs);
  if (typeof overallTimeout === "object" && "unref" in overallTimeout) {
    overallTimeout.unref();
  }
  const response = await fetch(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify({ ...openAiPayload, stream: true }),
    // Merge (don't overwrite) the internal timeout controller with the caller's
    // cancellation signal so both an internal timeout and an external abort stop
    // the stream.
    signal: signal ? anySignal([controller.signal, signal]) : controller.signal,
  });

  if (!response.body) {
    throw new Error("Stream error: no response body");
  }

  let paragraph = "";
  let reasoning = "";
  const toolCallAccumulators: { id?: string; name: string; arguments: string }[] =
    [];

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
      logger.error(
        id,
        `Stream ended prematurely after ${chunkIndex + 1} chunks`,
      );
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
        clearTimeout(overallTimeout);
        return parseStreamedResponse(
          id,
          paragraph,
          toolCallAccumulators,
          functionNames,
          reasoning,
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
      if (toolCalls) {
        for (const toolCall of toolCalls) {
          const idx = toolCall.index ?? 0;
          while (toolCallAccumulators.length <= idx) {
            toolCallAccumulators.push({ name: "", arguments: "" });
          }
          // The id arrives on the first fragment of each call.
          if (toolCall.id) toolCallAccumulators[idx].id = toolCall.id;
          if (toolCall.function?.name)
            toolCallAccumulators[idx].name += toolCall.function.name;
          if (toolCall.function?.arguments)
            toolCallAccumulators[idx].arguments += toolCall.function.arguments;
        }
      }

      const text = json.choices[0]?.delta?.content;
      if (text) paragraph += text;

      // Reasoning models (OpenRouter/OAI-compatible) stream a parallel
      // `reasoning` channel; accumulate it so callers can round-trip it.
      const reasoningDelta = json.choices[0]?.delta?.reasoning;
      if (reasoningDelta) reasoning += reasoningDelta;
    }
  }
}

async function callOpenAI(
  id: Identifier,
  openAiPayload: OpenAIPayload,
  openAiConfig: OpenAIConfig | undefined,
  requestTimeoutMs: number = 120_000,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  const { endpoint, headers } = buildOpenAIRequestConfig(
    id,
    openAiPayload.model,
    openAiConfig,
  );

  // Per-attempt deadline covering the request + body read (fetch has no built-in
  // timeout). Cleared once the body is read; an abort surfaces as a retryable error.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), requestTimeoutMs);
  let data: any;
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify({ ...openAiPayload, stream: false }),
      // Merge the internal timeout controller with the caller's cancellation signal.
      signal: signal ? anySignal([controller.signal, signal]) : controller.signal,
    });

    if (!response.ok) {
      const errorData = await response.json();
      logger.error(id, "OpenAI API error:", errorData);
      throw new Error(`OpenAI API Error: ${errorData.error.message}`);
    }

    data = await response.json();
  } finally {
    clearTimeout(timer);
  }

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
  const functionCalls: FunctionCall[] = [];

  if (toolCalls?.length) {
    for (let i = 0; i < toolCalls.length; i++) {
      const tc = toolCalls[i];
      functionCalls.push({
        id: tc.id ?? `call_${i}`,
        name: tc.function.name,
        arguments: JSON.parse(tc.function.arguments),
      });
    }
  } else if (choice.function_call) {
    functionCalls.push({
      id: "call_0",
      name: choice.function_call.name,
      arguments: JSON.parse(choice.function_call.arguments),
    });
  }

  // An empty 200 (no content, no tool call) is not a usable answer — throw so
  // withRetries retries and callWithRetries can fall back. (Mirrors the
  // streaming path's guard in parseStreamedResponse.)
  if (!choice.message?.content && !functionCalls.length) {
    logger.error(
      id,
      "OpenAI: received message without content or function_call:",
      JSON.stringify(data),
    );
    throw new Error(
      "OpenAI: received message without content or function_call",
    );
  }

  return {
    role: "assistant",
    content: choice.message.content || null,
    function_call: functionCalls[0] || null,
    function_calls: functionCalls,
    files: [],
    reasoning: choice.message?.reasoning ?? undefined,
    reasoningDetails: choice.message?.reasoning_details ?? undefined,
    usage: data.usage
      ? {
          prompt_tokens: data.usage.prompt_tokens,
          completion_tokens: data.usage.completion_tokens,
          total_tokens: data.usage.total_tokens,
          cached_tokens: data.usage.prompt_tokens_details?.cached_tokens ?? 0,
        }
      : null,
  };
}

async function callOpenAiWithRetries(
  id: Identifier,
  openAiPayload: OpenAIPayload,
  openAiConfig?: OpenAIConfig,
  retries: number = 5,
  chunkTimeoutMs: number = 15_000,
  requestTimeoutMs: number = 120_000,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  logger.log(
    id,
    "Calling OpenAI API with retries:",
    openAiConfig?.service,
    openAiPayload.model,
  );

  const modelStr = openAiPayload.model as string;
  const useStreaming =
    modelStr !== GPTModel.O1_MINI &&
    modelStr !== GPTModel.O1_PREVIEW &&
    !modelStr.startsWith("o1");

  return withRetries(
    id,
    "OpenAI",
    async () => {
      if (useStreaming) {
        return callOpenAIStream(
          id,
          openAiPayload,
          openAiConfig,
          chunkTimeoutMs,
          requestTimeoutMs,
          signal,
        );
      } else {
        return callOpenAI(id, openAiPayload, openAiConfig, requestTimeoutMs, signal);
      }
    },
    {
      retries,
      signal,
      baseDelayMs: 250,
      onError: (error, attempt) => {
        logger.error(
          id,
          `Retry #${attempt} error: ${error.message}`,
          error.response?.data || error.data || error,
        );

        // Remove images on content policy violation
        if (error.data?.code === "content_policy_violation") {
          logger.log(id, "Removing images due to content policy violation");
          openAiPayload.messages.forEach((message: OpenAIMessage) => {
            if (Array.isArray(message.content)) {
              message.content = message.content.filter(
                (content) => content.type === "text",
              );
            }
          });
        }
      },
    },
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ANTHROPIC
// ─────────────────────────────────────────────────────────────────────────────

function jigAnthropicMessages(
  messages: AnthropicAIMessage[],
): AnthropicAIMessage[] {
  const hasToolBlock = (content: AnthropicAIMessage["content"]) =>
    Array.isArray(content) &&
    content.some((b) => b.type === "tool_use" || b.type === "tool_result");

  let jiggedMessages = messages.slice();

  // Ensure first message is from user
  if (jiggedMessages[0]?.role !== "user") {
    jiggedMessages = [
      { role: "user" as const, content: "..." },
      ...jiggedMessages,
    ];
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

      // Never inject a text separator into a turn carrying tool_use/tool_result
      // blocks — it would sit between a tool_use and its result and break
      // Anthropic's pairing requirement.
      const separator: AnthropicContentBlock[] =
        hasToolBlock(lastMessage.content) || hasToolBlock(message.content)
          ? []
          : [{ type: "text", text: "\n\n---\n\n" }];

      lastMessage.content = [...lastContent, ...separator, ...newContent];
      return acc;
    }

    // Convert string content to text content block
    if (typeof message.content === "string") {
      message.content = [{ type: "text", text: message.content }];
    }

    return [...acc, message];
  }, [] as AnthropicAIMessage[]);

  // Ensure last message is from user — but never append a placeholder after an
  // unanswered tool_use turn (a text-only user can't satisfy it).
  const last = jiggedMessages[jiggedMessages.length - 1];
  if (last?.role === "assistant" && !hasToolBlock(last.content)) {
    jiggedMessages.push({ role: "user", content: "..." });
  }

  return jiggedMessages;
}

async function prepareAnthropicPayload(
  _identifier: Identifier,
  payload: GenericPayload,
): Promise<AnthropicAIPayload> {
  const preparedPayload: AnthropicAIPayload = {
    model: payload.model as ClaudeModel,
    messages: [],
    functions: payload.functions,
    temperature: payload.temperature,
    // Map the generic function_call to Anthropic tool_choice ("none" forces a
    // text-only turn). Only meaningful alongside tools — callAnthropic sends
    // it only when tools are present (tool_choice without tools 400s).
    tool_choice: payload.function_call
      ? typeof payload.function_call === "string"
        ? { type: payload.function_call }
        : { type: "tool", name: payload.function_call.name }
      : undefined,
  };

  for (const message of payload.messages) {
    if (message.role === "system") {
      preparedPayload.system = message.content;
      continue;
    }

    // role:"tool" → a user message carrying tool_result blocks (parallel
    // results share one message, matching Anthropic's expected shape).
    if (message.role === "tool") {
      preparedPayload.messages.push({
        role: "user",
        content: (message.toolResults || []).map((tr) => ({
          type: "tool_result" as const,
          tool_use_id: tr.toolCallId,
          content: tr.content,
        })),
      });
      continue;
    }

    const contentBlocks: AnthropicContentBlock[] = [];

    if (message.content) {
      contentBlocks.push({ type: "text", text: message.content });
    }

    for (const file of message.files || []) {
      if (ALLOWED_IMAGE_MIME_TYPES.includes(file.mimeType)) {
        if (file.url) {
          if (message.role == "user") {
            // anthropic assistant turns can't have images
            contentBlocks.push({
              type: "image",
              source: {
                type: "base64",
                media_type: "image/png",
                data: await getNormalizedBase64PNG(file.url, file.mimeType),
              },
            });
          }
          contentBlocks.push({ type: "text", text: `Image (${file.url})` });
        } else if (file.data) {
          if (message.role == "user") {
            // anthropic assistant turns can't have images
            contentBlocks.push({
              type: "image",
              source: {
                type: "base64",
                media_type: file.mimeType as any,
                data: file.data,
              },
            });
          }
        }
      } else if (file.url) {
        // Non-image file with URL - add text reference
        contentBlocks.push({
          type: "text",
          text: `File (${file.url})`,
        });
      }
    }

    // Thinking blocks (if the caller echoes them) must lead an assistant turn,
    // before text; tool_use blocks come last. Only Anthropic's own block shapes
    // survive: a cross-provider fallback replays reasoningDetails captured from
    // another provider (e.g. OpenRouter `reasoning.text`), and Anthropic 400s
    // on the unknown input tag.
    const leadingBlocks: AnthropicContentBlock[] =
      message.role === "assistant" && Array.isArray(message.reasoningDetails)
        ? message.reasoningDetails.filter(
            (block: any) =>
              block?.type === "thinking" ||
              block?.type === "redacted_thinking",
          )
        : [];
    const toolUseBlocks: AnthropicContentBlock[] = (
      message.functionCalls || []
    ).map((fc, i) => ({
      type: "tool_use" as const,
      id: fc.id ?? `call_${i}`,
      name: fc.name,
      input: fc.arguments,
    }));

    preparedPayload.messages.push({
      role: message.role,
      content: [...leadingBlocks, ...contentBlocks, ...toolUseBlocks],
    });
  }

  return preparedPayload;
}

async function callAnthropic(
  id: Identifier,
  payload: AnthropicAIPayload,
  config?: AnthropicAIConfig,
  requestTimeoutMs: number = 120_000,
  signal?: AbortSignal,
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
      }),
      { abortSignal: signal },
    );

    const decodedResponseBody = new TextDecoder().decode(response.body);
    data = JSON.parse(decodedResponseBody);
  } else {
    // Default: Anthropic API
    // Prompt caching: mark a breakpoint on the last tool only. Tool schemas
    // are fully static and identical across users, so that span gets real
    // cache reads; the system prompt is left uncached because callers embed
    // per-user / per-minute content in it, and a breakpoint after a volatile
    // span pays 1.25x cache writes with near-zero reads. Anthropic ignores
    // breakpoints below the model's minimum cacheable length, so small tool
    // sets are a safe no-op.
    const cachedTools = tools?.length
      ? [
          ...tools.slice(0, -1),
          { ...tools[tools.length - 1], cache_control: { type: "ephemeral" } },
        ]
      : tools;
    const response = await withRequestDeadline(
      "Anthropic",
      requestTimeoutMs,
      signal,
      (mergedSignal) =>
        axios.post(
          "https://api.anthropic.com/v1/messages",
          {
            model: payload.model,
            messages: anthropicMessages,
            tools: cachedTools,
            // tool_choice requires tools in the request; drop it otherwise.
            tool_choice: cachedTools?.length ? payload.tool_choice : undefined,
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
            timeout: requestTimeoutMs,
            signal: mergedSignal,
          },
        ),
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
  // Native API thinking blocks (`{type:"thinking",thinking,signature}` /
  // `redacted_thinking`), kept raw so callers can echo them back verbatim —
  // their signatures are validated by Anthropic on the next turn.
  const reasoningBlocks: any[] = [];

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
          "",
        );
        logger.log(id, "No text in answer, returning text within tags:", text);
      }

      textResponse = textResponse ? `${textResponse}\n\n${text}` : text;
    } else if (answer.type === "tool_use") {
      functionCalls.push({
        id: answer.id,
        name: answer.name,
        arguments: answer.input,
      });
    } else if (
      answer.type === "thinking" ||
      answer.type === "redacted_thinking"
    ) {
      reasoningBlocks.push(answer);
    }
  }

  if (!textResponse && !functionCalls.length) {
    logger.error(
      id,
      "Missing text & functions in Anthropic API response:",
      data,
    );
    throw new Error("Missing text & functions in Anthropic API response");
  }

  // Anthropic's input_tokens EXCLUDES cache reads/writes; fold them back in
  // so prompt_tokens means "all input tokens" like OpenAI, where
  // cached_tokens is a subset of prompt_tokens.
  let usage: ParsedResponseMessage["usage"] = null;
  if (data.usage) {
    const cacheRead = data.usage.cache_read_input_tokens ?? 0;
    const cacheWrite = data.usage.cache_creation_input_tokens ?? 0;
    const promptTokens = data.usage.input_tokens + cacheRead + cacheWrite;
    usage = {
      prompt_tokens: promptTokens,
      completion_tokens: data.usage.output_tokens,
      total_tokens: promptTokens + data.usage.output_tokens,
      cached_tokens: cacheRead,
    };
  }

  return {
    role: "assistant",
    content: textResponse,
    function_call: functionCalls[0] || null,
    function_calls: functionCalls,
    files: [],
    reasoningDetails: reasoningBlocks.length ? reasoningBlocks : undefined,
    usage,
  };
}

async function callAnthropicWithRetries(
  id: Identifier,
  payload: AnthropicAIPayload,
  config?: AnthropicAIConfig,
  retries: number = 5,
  requestTimeoutMs: number = 120_000,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  return withRetries(
    id,
    "Anthropic",
    () => callAnthropic(id, payload, config, requestTimeoutMs, signal),
    {
      retries,
      signal,
    },
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GOOGLE AI
// ─────────────────────────────────────────────────────────────────────────────

function jigGoogleMessages(messages: GoogleAIMessage[]): GoogleAIMessage[] {
  const hasFunctionPart = (parts: GoogleAIPart[]) =>
    parts.some((p) => "functionCall" in p || "functionResponse" in p);

  let jiggedMessages = messages.slice();

  // Ensure first message is from user
  if (jiggedMessages[0]?.role === "model") {
    jiggedMessages = [
      { role: "user" as const, parts: [{ text: "..." }] },
      ...jiggedMessages,
    ];
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

  // Ensure last message is from user — but don't append a placeholder after a
  // model turn that ends in a functionCall (it would orphan the call into
  // history and send "..." as the message).
  const last = jiggedMessages[jiggedMessages.length - 1];
  if (last?.role === "model" && !hasFunctionPart(last.parts)) {
    jiggedMessages.push({ role: "user", parts: [{ text: "..." }] });
  }

  return jiggedMessages;
}

async function prepareGoogleAIPayload(
  _identifier: Identifier,
  payload: GenericPayload,
): Promise<GoogleAIPayload> {
  const preparedPayload: GoogleAIPayload = {
    model: payload.model as GeminiModel,
    messages: [],
    thinkingConfig: payload.thinkingConfig,
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
    // Map the generic function_call to Gemini's functionCallingConfig ("none"
    // → mode NONE forces a text-only turn). Only meaningful alongside tools.
    toolConfig:
      payload.functions && payload.function_call
        ? {
            functionCallingConfig: {
              mode:
                typeof payload.function_call === "string"
                  ? payload.function_call === "none"
                    ? "NONE"
                    : "AUTO"
                  : "ANY",
            },
          }
        : undefined,
  };

  // id -> tool name, to backfill functionResponse.name when a caller omits it.
  const toolNameById = new Map<string, string>();
  for (const m of payload.messages) {
    for (const fc of m.functionCalls || []) {
      if (fc.id) toolNameById.set(fc.id, fc.name);
    }
  }

  for (const message of payload.messages) {
    if (message.role === "system") {
      preparedPayload.systemInstruction = message.content;
      continue;
    }

    // role:"tool" → a user turn carrying functionResponse parts.
    if (message.role === "tool") {
      preparedPayload.messages.push({
        role: "user",
        parts: (message.toolResults || []).map((tr) => ({
          functionResponse: {
            id: tr.toolCallId,
            name: tr.name ?? toolNameById.get(tr.toolCallId) ?? "",
            response: { output: tr.content },
          },
        })),
      });
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
            fileData: {
              mimeType: file.mimeType,
              fileUri: file.url,
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

    for (const fc of message.functionCalls || []) {
      parts.push({
        functionCall: {
          id: fc.id,
          name: fc.name,
          args: fc.arguments,
        },
        // Gemini requires its thoughtSignature echoed back on the call part.
        ...(fc.thoughtSignature
          ? { thoughtSignature: fc.thoughtSignature }
          : {}),
      });
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
  payload: GoogleAIPayload,
  requestTimeoutMs: number = 120_000,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  const contents = jigGoogleMessages(payload.messages);

  // Call the REST generateContent endpoint directly rather than via @google/genai:
  // the pinned SDK (0.6.1) silently strips Gemini's per-call `thoughtSignature`,
  // which Gemini REQUIRES echoed back on multi-turn tool calls (a missing one
  // 400s the request). Going over the wire ourselves preserves it both ways.
  const requestBody: any = {
    contents,
    generationConfig: {
      responseModalities: ["TEXT"],
      ...(payload.thinkingConfig
        ? { thinkingConfig: payload.thinkingConfig }
        : {}),
    },
  };
  if (payload.tools) requestBody.tools = [payload.tools];
  if (payload.tools && payload.toolConfig) {
    requestBody.toolConfig = payload.toolConfig;
  }
  if (payload.systemInstruction) {
    requestBody.systemInstruction = {
      parts: [{ text: payload.systemInstruction }],
    };
  }

  let response: any;
  try {
    const httpResponse = await withRequestDeadline(
      "Google AI",
      requestTimeoutMs,
      signal,
      (mergedSignal) =>
        axios.post(
          `https://generativelanguage.googleapis.com/v1beta/models/${payload.model}:generateContent`,
          requestBody,
          {
            headers: {
              "content-type": "application/json",
              "x-goog-api-key": process.env.GEMINI_API_KEY as string,
            },
            timeout: requestTimeoutMs,
            signal: mergedSignal,
          },
        ),
    );
    response = httpResponse.data;
  } catch (err: any) {
    // Re-shape the API error so callGoogleAIWithRetries' circuit breaker can read
    // message / status / promptFeedback off it.
    const apiError = err?.response?.data?.error;
    const wrapped = new Error(
      apiError?.message || err?.message || "Google AI API request failed",
    ) as any;
    wrapped.status = apiError?.status ?? err?.response?.status;
    wrapped.code = apiError?.code;
    wrapped.details = apiError?.details;
    wrapped.promptFeedback = err?.response?.data?.promptFeedback;
    throw wrapped;
  }

  let text = "";
  const files: File[] = [];
  const reasoningParts: any[] = [];
  // Built from content.parts (not response.functionCalls) so we can keep each
  // call's thoughtSignature, which the accessor drops. Gemini may omit ids —
  // synthesize positional ones so the caller can pair each with a
  // `ToolResult.toolCallId` next turn; the thoughtSignature MUST be echoed back
  // on round-trip (else Gemini 400s "missing a thought_signature").
  const functionCalls: FunctionCall[] = [];

  for (const part of response.candidates?.[0]?.content?.parts || []) {
    // Thinking parts (thought:true) carry chain-of-thought, not the answer.
    if ((part as any).thought) {
      reasoningParts.push(part);
      continue;
    }
    if (part.functionCall) {
      functionCalls.push({
        id: part.functionCall.id ?? `call_${functionCalls.length}`,
        name: part.functionCall.name ?? "",
        arguments: part.functionCall.args ?? {},
        thoughtSignature: (part as any).thoughtSignature,
      });
      continue;
    }
    if (part.text) text += part.text;
    if (part.inlineData?.data) {
      files.push({ mimeType: "image/png", data: part.inlineData.data });
    }
  }

  if (!text && !functionCalls.length && !files.length) {
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
        MALFORMED_FUNCTION_CALL:
          "(Google could not generate valid function call arguments)",
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
    function_call: functionCalls[0] || null,
    function_calls: functionCalls,
    reasoningDetails: reasoningParts.length ? reasoningParts : undefined,
    usage: response.usageMetadata
      ? {
          prompt_tokens: response.usageMetadata.promptTokenCount ?? 0,
          completion_tokens: response.usageMetadata.candidatesTokenCount ?? 0,
          total_tokens: response.usageMetadata.totalTokenCount ?? 0,
          cached_tokens: response.usageMetadata.cachedContentTokenCount ?? 0,
          thoughts_tokens: response.usageMetadata.thoughtsTokenCount,
        }
      : null,
  };
}

/**
 * Content violation finish reasons that should trigger circuit breaker behavior.
 * These errors won't resolve with simple retries - the content itself is the problem.
 */
const CONTENT_VIOLATION_REASONS = new Set(["PROHIBITED_CONTENT", "SAFETY"]);

/**
 * Removes inline image data from Google AI messages, preserving text content.
 * Used as a fallback when content violations are detected.
 */
function removeImagesFromGooglePayload(payload: GoogleAIPayload): boolean {
  let removedImages = false;

  for (const message of payload.messages) {
    message.parts = message.parts.filter((part) => {
      if ("inlineData" in part || "fileData" in part) {
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
  retries: number = 5,
  requestTimeoutMs: number = 120_000,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  let hasTriedWithoutImages = false;

  return withRetries(id, "Google AI", () => callGoogleAI(id, payload, requestTimeoutMs, signal), {
    retries,
    signal,
    onError: (error, attempt) => {
      const errorDetails: Record<string, any> = {
        message: error.message,
        finishReason: error.finishReason,
        modelVersion: error.modelVersion,
      };

      if (error.safetyRatings) errorDetails.safetyRatings = error.safetyRatings;
      if (error.usageMetadata) errorDetails.usageMetadata = error.usageMetadata;
      if (error.promptFeedback)
        errorDetails.promptFeedback = error.promptFeedback;
      if (error.status) errorDetails.httpStatus = error.status;
      if (error.code) errorDetails.errorCode = error.code;
      if (error.details) errorDetails.errorDetails = error.details;

      const fileUris = payload.messages
        .flatMap((m) => m.parts)
        .filter((p) => "fileData" in p)
        .map((p) => (p as any).fileData.fileUri);
      if (fileUris.length) errorDetails.fileUris = fileUris;

      logger.error(
        id,
        `Retry #${attempt} error: ${error.message}`,
        errorDetails,
      );

      // Circuit breaker: Google's fetcher couldn't pull our image URL(s). We pass
      // images as fileData.fileUri, but arbitrary (non-Files-API) URLs are only
      // best-effort for Gemini, so this 400 is episodic and retrying the same URL
      // almost never recovers within a request. Fail fast so the caller's
      // fallbackModel (whose adapter inlines the image bytes) takes over instead of
      // burning all retries first. Unlike the content-violation path we do NOT strip
      // images — the fallback model should still see them.
      if (
        typeof error?.message === "string" &&
        error.message.includes("Cannot fetch content from the provided URL")
      ) {
        logger.error(
          id,
          "Circuit breaker: Google could not fetch image URL(s); failing over (no more Google retries)",
        );
        const fetchError = new Error(
          "Google AI could not fetch the provided image URL(s).",
        ) as any;
        fetchError.cause = error;
        fetchError.googleFetchFailure = true;
        throw fetchError;
      }

      // Circuit breaker: detect content violations and try removing images
      // Check both finishReason (candidate-level) and promptFeedback.blockReason (prompt-level)
      const violationReason =
        (CONTENT_VIOLATION_REASONS.has(error.finishReason) &&
          error.finishReason) ||
        (CONTENT_VIOLATION_REASONS.has(error.promptFeedback?.blockReason) &&
          error.promptFeedback?.blockReason);

      if (violationReason) {
        if (!hasTriedWithoutImages) {
          const removedImages = removeImagesFromGooglePayload(payload);
          if (removedImages) {
            logger.log(
              id,
              `Circuit breaker triggered: removing images due to ${violationReason}`,
            );
            hasTriedWithoutImages = true;
            return; // Continue to next retry with images removed
          }
        }

        // If we already tried without images or there were no images, fail fast
        logger.error(
          id,
          `Circuit breaker: failing fast due to ${violationReason} (no more fallbacks)`,
        );
        const circuitBreakerError = new Error(
          `Google AI content violation: ${violationReason}. Request cannot succeed with current content.`,
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
  content: AnthropicAIMessage["content"],
): string {
  return Array.isArray(content)
    ? content
        .map((c) => (c.type === "text" ? c.text : `[${c.type}]`))
        .join("\n")
    : content;
}

/**
 * Serialize generic messages for OpenAI-compatible chat APIs (Groq, OpenRouter).
 * Mirrors the OpenAI adapter but keeps content as a plain string. Tool calls
 * become assistant `tool_calls`, and a `role:"tool"` message expands to one
 * `{role:"tool", tool_call_id, content}` per result. Reasoning is passed
 * through only when the caller supplied it.
 *
 * With `imageParts` (OpenRouter models with vision-capable endpoints), a
 * message carrying image attachments additionally gets OpenAI-style
 * `image_url` content parts (remote URL when present, else a `data:` URI —
 * which the plain path would silently drop), with the string content becoming
 * the leading text part. Like the OpenAI adapter, URL images keep their
 * `Image (url)` text reference alongside the pixels so the model can still
 * quote the link. Messages without image attachments serialize identically in
 * both modes.
 */
function prepareOpenAICompatMessages(
  messages: GenericMessage[],
  opts: { imageParts?: boolean } = {},
): OpenAIMessage[] {
  const out: OpenAIMessage[] = [];
  for (const message of messages) {
    if (message.role === "tool") {
      for (const tr of message.toolResults || []) {
        out.push({
          role: "tool",
          tool_call_id: tr.toolCallId,
          content: tr.content,
        });
      }
      continue;
    }

    // Content stays a plain string on this path, so attachments become URL
    // references appended to it — images included, since without `imageParts`
    // the model is treated as text-only and would otherwise never learn a
    // file exists.
    const fileRefs = (message.files || [])
      .filter((file) => file.url)
      .map((file) =>
        ALLOWED_IMAGE_MIME_TYPES.includes(file.mimeType)
          ? `Image (${file.url})`
          : `File (${file.url})`,
      );
    const content = [normalizeMessageContent(message.content), ...fileRefs]
      .filter(Boolean)
      .join("\n");

    const outMessage: OpenAIMessage = {
      role: message.role,
      content,
    };
    if (opts.imageParts) {
      const imageBlocks: OpenAIContentBlock[] = (message.files || [])
        .filter(
          (file) =>
            ALLOWED_IMAGE_MIME_TYPES.includes(file.mimeType) &&
            (file.url || file.data),
        )
        .map((file) => ({
          type: "image_url",
          image_url: {
            url: file.url || `data:${file.mimeType};base64,${file.data}`,
          },
        }));
      if (imageBlocks.length) {
        outMessage.content = [
          ...(content
            ? [{ type: "text", text: content } as OpenAIContentBlock]
            : []),
          ...imageBlocks,
        ];
      }
    }
    if (message.functionCalls?.length) {
      outMessage.tool_calls = message.functionCalls.map((fc, i) => ({
        id: fc.id ?? `call_${i}`,
        type: "function" as const,
        function: {
          name: fc.name,
          arguments: JSON.stringify(fc.arguments),
        },
      }));
      // OpenAI-compatible APIs want null content on a tool-call-only turn.
      if (!content && !Array.isArray(outMessage.content))
        outMessage.content = null;
    }
    if (message.reasoning) outMessage.reasoning = message.reasoning;
    const reasoningDetails = filterOpenAICompatReasoningDetails(
      message.reasoningDetails,
    );
    if (reasoningDetails) outMessage.reasoning_details = reasoningDetails;
    out.push(outMessage);
  }
  return out;
}

function prepareGroqPayload(payload: GenericPayload): GroqPayload {
  return {
    model: payload.model as GroqModel,
    messages: prepareOpenAICompatMessages(payload.messages),
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
  payload: GroqPayload,
  requestTimeoutMs: number = 120_000,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  const response = await withRequestDeadline(
    "Groq",
    requestTimeoutMs,
    signal,
    (mergedSignal) =>
      axios.post(
        "https://api.groq.com/openai/v1/chat/completions",
        payload,
        {
          headers: {
            "content-type": "application/json",
            Authorization: `Bearer ${process.env.GROQ_API_KEY}`,
          },
          timeout: requestTimeoutMs,
          signal: mergedSignal,
        },
      ),
  );

  // Like OpenRouter, Groq can return an error-shaped HTTP 200 with no `choices`
  // key; surface it instead of throwing a cryptic `choices[0]` TypeError.
  if (response.data.error) {
    logger.error(id, "Groq error:", response.data.error);
    throw new Error(`Groq error: ${response.data.error.message}`);
  }

  const answer = response.data.choices?.[0]?.message;
  if (!answer) {
    logger.error(id, "Missing answer in Groq API response:", response.data);
    throw new Error("Missing answer in Groq API");
  }

  const functionCalls: FunctionCall[] = [];
  if (answer.tool_calls?.length) {
    for (let i = 0; i < answer.tool_calls.length; i++) {
      const tc = answer.tool_calls[i];
      functionCalls.push({
        id: tc.id ?? `call_${i}`,
        name: tc.function.name,
        arguments: JSON.parse(tc.function.arguments),
      });
    }
  }

  // An empty 200 (no content, no tool call) is not a usable answer — throw so
  // withRetries retries and callWithRetries can fall back. (Mirrors the
  // streaming path's guard in parseStreamedResponse.)
  if (!answer.content && !functionCalls.length) {
    logger.error(
      id,
      "Groq: received message without content or function_call:",
      JSON.stringify(response.data),
    );
    throw new Error(
      "Groq: received message without content or function_call",
    );
  }

  return {
    role: "assistant",
    content: answer.content || null,
    function_call: functionCalls[0] || null,
    function_calls: functionCalls,
    files: [],
    reasoning: answer.reasoning ?? undefined,
    usage: response.data.usage
      ? {
          prompt_tokens: response.data.usage.prompt_tokens,
          completion_tokens: response.data.usage.completion_tokens,
          total_tokens: response.data.usage.total_tokens,
          cached_tokens:
            response.data.usage.prompt_tokens_details?.cached_tokens ?? 0,
        }
      : null,
  };
}

async function callGroqWithRetries(
  id: Identifier,
  payload: GroqPayload,
  retries: number = 5,
  requestTimeoutMs: number = 120_000,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  return withRetries(id, "Groq", () => callGroq(id, payload, requestTimeoutMs, signal), {
    retries,
    signal,
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// OPENROUTER
// ─────────────────────────────────────────────────────────────────────────────

/**
 * In-process memory of OpenRouter models that rejected image input (the
 * routing-layer 404 "No endpoints found that support image input"). Payloads
 * for these models degrade image attachments to inline `Image (url)` text
 * references up front — the exact pre-vision serialization — instead of
 * burning a doomed attempt per call. Populated by the retry loop on first
 * rejection; cleared only by process restart (exported so tests can reset it).
 */
export const openRouterImageRejectedModels = new Set<string>();

function prepareOpenRouterPayload(payload: GenericPayload): OpenRouterPayload {
  return {
    model: payload.model as OpenRouterModel,
    messages: prepareOpenAICompatMessages(payload.messages, {
      imageParts: !openRouterImageRejectedModels.has(String(payload.model)),
    }),
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
    provider: payload.provider,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// DeepSeek "DSML" tool-call recovery
// ─────────────────────────────────────────────────────────────────────────────
// DeepSeek models (e.g. deepseek-v4-flash) on OpenRouter intermittently emit
// their native tool-call markup as plain assistant *content* instead of
// populating the response's `tool_calls`. The markup ("DSML") looks like:
//
//   <｜DSML｜tool_calls>
//     <｜DSML｜invoke name="use_skills">
//       <｜DSML｜parameter name="skills" string="false">["search"]</｜DSML｜parameter>
//     </｜DSML｜invoke>
//   </｜DSML｜tool_calls>
//
// with either one or two fullwidth vertical bars (｜ = U+FF5C) around DSML. The
// `string` attribute says whether the value is a literal string ("true") or a
// JSON value to parse ("false"). We recover these into structured FunctionCalls
// so the turn executes normally instead of leaking raw markup to the caller.
//
// Regex tags use `｜+` (one-or-more fullwidth vertical bars) so the single-
// and double-bar variants both match.
const DSML_ENVELOPE_RE = /<｜+DSML｜+tool_calls>/;
const DSML_DELIMITER_RE = /<\/?｜+DSML｜+/;
const DSML_INVOKE_RE =
  /<｜+DSML｜+invoke\s+name="([^"]+)"\s*>([\s\S]*?)<\/｜+DSML｜+invoke>/g;
const DSML_PARAM_RE =
  /<｜+DSML｜+parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>([\s\S]*?)<\/｜+DSML｜+parameter>/g;
// Any DSML tag (open or close), used to excise the whole markup span from
// surrounding prose once the calls have been extracted.
const DSML_ANY_TAG_RE = /<\/?｜+DSML｜+[^>]*>/g;

function parseDsmlToolCalls(content: string): {
  calls: FunctionCall[];
  remainingContent: string | null;
} {
  const calls: FunctionCall[] = [];

  DSML_INVOKE_RE.lastIndex = 0;
  let invokeMatch: RegExpExecArray | null;
  while ((invokeMatch = DSML_INVOKE_RE.exec(content)) !== null) {
    const name = invokeMatch[1];
    const inner = invokeMatch[2];
    const args: Record<string, any> = {};
    let ok = true;

    DSML_PARAM_RE.lastIndex = 0;
    let paramMatch: RegExpExecArray | null;
    while ((paramMatch = DSML_PARAM_RE.exec(inner)) !== null) {
      const [, paramName, stringAttr, rawValue] = paramMatch;
      if (stringAttr === "false") {
        // value is a JSON literal (array / number / object / bool / quoted string)
        try {
          args[paramName] = JSON.parse(rawValue);
        } catch {
          ok = false; // malformed typed arg → don't emit a wrong-typed call
          break;
        }
      } else {
        // string="true" (or attribute absent) → literal string value
        args[paramName] = rawValue;
      }
    }

    if (ok && name) {
      calls.push({ id: `call_${calls.length}`, name, arguments: args });
    }
  }

  if (!calls.length) {
    return { calls, remainingContent: content };
  }

  // Excise the whole DSML span (first tag through last tag, including the
  // already-captured parameter values between them) so any surrounding prose
  // survives but the raw markup never reaches the caller.
  DSML_ANY_TAG_RE.lastIndex = 0;
  let first = -1;
  let last = -1;
  let tag: RegExpExecArray | null;
  while ((tag = DSML_ANY_TAG_RE.exec(content)) !== null) {
    if (first === -1) first = tag.index;
    last = tag.index + tag[0].length;
  }
  const remaining =
    first === -1
      ? content
      : (content.slice(0, first) + content.slice(last)).trim();
  return { calls, remainingContent: remaining.length ? remaining : null };
}

// Independent timeout budgets for the two OpenRouter transports. Streaming is
// bounded by elapsed total + a per-useful-chunk stall timeout, so it can afford
// a generous total: a healthy long generation keeps producing useful chunks,
// while a hung one dies within one stall window. Non-streaming has no
// progress signal at all, so its total must stay tight.
export const OPENROUTER_STREAM_TIMEOUT_MS = 600_000;
export const OPENROUTER_NONSTREAM_TIMEOUT_MS = 180_000;

function openRouterEndpoint(): string {
  // Override point for tests (deadline/stream behavior needs a local server).
  return `${process.env.OPENROUTER_BASE_URL || "https://openrouter.ai"}/api/v1/chat/completions`;
}

/**
 * Shared tail of both OpenRouter transports: assemble the ParsedResponseMessage
 * from the raw pieces, recover DSML tool calls, and reject unusable (empty /
 * truncated-markup) completions so withRetries retries and callWithRetries can
 * fall back.
 */
function finalizeOpenRouterMessage(
  id: Identifier,
  raw: {
    content: string | null;
    toolCalls: { id?: string; name: string; argumentsJson: string }[];
    reasoning?: string;
    reasoningDetails?: any;
    provider?: string;
    usage?: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
      prompt_tokens_details?: { cached_tokens?: number };
    } | null;
    /** Whole response (or a summary of the stream) for the failure log. */
    forLog: () => string;
  },
): ParsedResponseMessage {
  const functionCalls: FunctionCall[] = [];
  for (let i = 0; i < raw.toolCalls.length; i++) {
    const tc = raw.toolCalls[i];
    if (!tc.name) continue;
    functionCalls.push({
      id: tc.id ?? `call_${i}`,
      name: tc.name,
      // Streamed no-arg calls can close with an empty fragment; treat as {}.
      arguments: tc.argumentsJson.trim() ? JSON.parse(tc.argumentsJson) : {},
    });
  }

  // DeepSeek sometimes emits its native "DSML" tool-call markup as plain content
  // instead of populating tool_calls. Recover it into structured calls so the
  // turn executes normally instead of leaking raw markup to the caller.
  let content = raw.content;
  if (!functionCalls.length && content && DSML_ENVELOPE_RE.test(content)) {
    const { calls, remainingContent } = parseDsmlToolCalls(content);
    if (calls.length) {
      functionCalls.push(...calls);
      content = remainingContent;
    }
  }

  // Not a usable answer — an empty completion (reasoning models e.g. deepseek
  // can route all output to the discarded `reasoning` channel), OR a DSML
  // envelope we couldn't parse into a call (truncated/malformed). Throw so
  // withRetries retries this model and, on exhaustion, callWithRetries falls
  // back to fallbackModel.
  const hasUnparsedDsml = !!content && DSML_DELIMITER_RE.test(content);
  if (!functionCalls.length && (!content || hasUnparsedDsml)) {
    logger.error(id, "OpenRouter: empty or unparseable completion:", raw.forLog());
    throw new Error(
      "OpenRouter: received message without usable content or function_call",
    );
  }

  return {
    role: "assistant",
    content: content || null,
    function_call: functionCalls[0] || null,
    function_calls: functionCalls,
    files: [],
    reasoning: raw.reasoning || undefined,
    reasoningDetails: raw.reasoningDetails ?? undefined,
    // The upstream provider OpenRouter routed to (e.g. "Baidu") — finer-grained
    // than the "openrouter" stamp callWithRetries would apply.
    provider: raw.provider ?? undefined,
    usage: raw.usage
      ? {
          prompt_tokens: raw.usage.prompt_tokens,
          completion_tokens: raw.usage.completion_tokens,
          total_tokens: raw.usage.total_tokens,
          cached_tokens: raw.usage.prompt_tokens_details?.cached_tokens ?? 0,
        }
      : null,
  };
}

/**
 * Streaming transport (the default). Two timers, both required:
 *
 * - `streamTimeoutMs` bounds the whole attempt (connect + generation). It is
 *   deliberately independent of the non-streaming `requestTimeoutMs`: with a
 *   progress signal available, a long healthy generation shouldn't be killed
 *   by a deadline sized for opaque requests (2026-07-27: 9–12k-token deepseek
 *   completions at healthy tps were being executed at the 120s hard deadline
 *   just before finishing, then retried from scratch).
 * - `chunkTimeoutMs` is a stall detector: it resets ONLY on a "useful" chunk —
 *   one advancing content, reasoning, reasoning_details, tool-call fragments,
 *   finish_reason, or usage. SSE comments (OpenRouter dribbles
 *   ": OPENROUTER PROCESSING" as keep-alive), role-only deltas, and other
 *   heartbeat noise do NOT reset it, so a provider that keeps the socket warm
 *   while generating nothing dies within one window instead of holding the
 *   turn to the total deadline. The window also covers connect + time to first
 *   token.
 */
async function callOpenRouterStream(
  id: Identifier,
  payload: OpenRouterPayload,
  streamTimeoutMs: number = OPENROUTER_STREAM_TIMEOUT_MS,
  chunkTimeoutMs: number = 15_000,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  const controller = new AbortController();
  // Why the reason is tracked out-of-band: fetch/reader surface any abort as a
  // bare AbortError, so without this the retry log would say "This operation
  // was aborted" no matter which timer fired.
  let abortReason: string | null = null;
  // Which timer fired: a deadline abort can still yield a usable (truncated)
  // answer, a stall abort by definition cannot.
  let abortKind: "deadline" | "stall" | null = null;
  const abortWith = (kind: "deadline" | "stall", reason: string) => {
    abortKind = kind;
    abortReason = reason;
    controller.abort();
  };

  const unref = (t: ReturnType<typeof setTimeout>) => {
    if (typeof t === "object" && "unref" in t) t.unref();
    return t;
  };
  const totalTimer = unref(
    setTimeout(
      () =>
        abortWith(
          "deadline",
          `OpenRouter stream exceeded total deadline of ${streamTimeoutMs}ms`,
        ),
      streamTimeoutMs,
    ),
  );
  let stallTimer: ReturnType<typeof setTimeout> | undefined;
  const armStallTimer = () => {
    clearTimeout(stallTimer);
    stallTimer = unref(
      setTimeout(
        () =>
          abortWith(
            "stall",
            `OpenRouter stream stalled: no useful chunk for ${chunkTimeoutMs}ms`,
          ),
        chunkTimeoutMs,
      ),
    );
  };

  let paragraph = "";
  let reasoning = "";
  const reasoningDetails: any[] = [];
  const toolCalls: { id?: string; name: string; argumentsJson: string }[] = [];
  let provider: string | undefined;
  let usage: any = null;
  let finishReason: string | null = null;
  let sawDone = false;
  let dataChunks = 0;

  try {
    armStallTimer(); // covers connect + time to first token
    const response = await fetch(openRouterEndpoint(), {
      method: "POST",
      headers: {
        "content-type": "application/json",
        Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
      },
      body: JSON.stringify({ ...payload, stream: true, usage: { include: true } }),
      signal: signal ? anySignal([controller.signal, signal]) : controller.signal,
    });

    if (!response.ok) {
      let data: any;
      try {
        data = await response.json();
      } catch {
        data = undefined;
      }
      logger.error(id, `OpenRouter stream HTTP ${response.status}:`, data);
      const error = new Error(
        `OpenRouter error: ${data?.error?.message || `HTTP ${response.status}`}`,
      ) as any;
      error.response = { status: response.status, data };
      throw error;
    }
    // Some responses to a streamed request arrive as a plain JSON body anyway
    // (error-in-200 payloads, proxies/providers that ignore `stream`). Parse
    // those as a non-streaming body instead of scanning them for SSE lines.
    if (response.headers.get("content-type")?.includes("application/json")) {
      return parseOpenRouterBody(id, await response.json());
    }
    if (!response.body) {
      throw new Error("OpenRouter stream error: no response body");
    }

    const reader = response.body.getReader();
    // One decoder in stream mode for the whole body: multi-byte UTF-8
    // sequences (CJK output!) split across TCP chunks must not be decoded
    // per-chunk or they turn into replacement characters.
    const decoder = new TextDecoder();
    let lineBuffer = "";

    outer: while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      lineBuffer += decoder.decode(value, { stream: true });

      let newlineIdx: number;
      while ((newlineIdx = lineBuffer.indexOf("\n")) !== -1) {
        let line = lineBuffer.slice(0, newlineIdx);
        lineBuffer = lineBuffer.slice(newlineIdx + 1);
        if (line.endsWith("\r")) line = line.slice(0, -1);
        if (!line) continue; // SSE event separator
        if (line.startsWith(":")) continue; // SSE comment — keep-alive, NOT useful
        if (!line.startsWith("data:")) continue; // ignore event:/id:/retry: fields
        const dataStr = line.slice(5).trimStart();
        if (dataStr === "[DONE]") {
          sawDone = true;
          break outer;
        }

        let json: any;
        try {
          json = JSON.parse(dataStr);
        } catch {
          // SSE frames one complete JSON object per data line and the line
          // buffer already reassembles split TCP chunks, so this is a
          // malformed event, not a partial one — skip it rather than corrupt
          // the accumulation.
          logger.error(
            id,
            "OpenRouter stream: unparseable data line:",
            dataStr.slice(0, 200),
          );
          continue;
        }
        dataChunks++;

        // Same error-in-200 wrapping as the non-streaming body, delivered as
        // an SSE event (rate limits, moderation, provider failures).
        if (json.error) {
          logger.error(id, "OpenRouter stream error event:", json.error);
          const error = new Error(
            `OpenRouter error: ${json.error.message}`,
          ) as any;
          error.data = json.error;
          throw error;
        }

        if (json.provider) provider = json.provider;
        let useful = false;
        if (json.usage) {
          usage = json.usage;
          useful = true;
        }
        const choice = json.choices?.[0];
        if (choice) {
          const delta = choice.delta ?? {};
          if (delta.content) {
            paragraph += delta.content;
            useful = true;
          }
          if (delta.reasoning) {
            reasoning += delta.reasoning;
            useful = true;
          }
          if (Array.isArray(delta.reasoning_details) && delta.reasoning_details.length) {
            reasoningDetails.push(...delta.reasoning_details);
            useful = true;
          }
          if (Array.isArray(delta.tool_calls)) {
            for (const toolCall of delta.tool_calls) {
              const idx = toolCall.index ?? 0;
              while (toolCalls.length <= idx) {
                toolCalls.push({ name: "", argumentsJson: "" });
              }
              if (toolCall.id) toolCalls[idx].id = toolCall.id;
              if (toolCall.function?.name)
                toolCalls[idx].name += toolCall.function.name;
              if (toolCall.function?.arguments)
                toolCalls[idx].argumentsJson += toolCall.function.arguments;
              useful = true;
            }
          }
          if (choice.finish_reason) {
            finishReason = choice.finish_reason;
            useful = true;
          }
        }
        if (useful) armStallTimer();
      }
    }

    // After [DONE] the server should close, but don't rely on it — release the
    // socket instead of holding it until GC.
    if (sawDone) reader.cancel().catch(() => {});

    // A clean close without [DONE] is acceptable only when the provider said
    // it finished; otherwise the connection died mid-generation and returning
    // the partial accumulation would present a truncated answer as complete.
    if (!sawDone && !finishReason) {
      logger.error(
        id,
        `OpenRouter stream ended prematurely after ${dataChunks} data chunks`,
      );
      throw new Error("OpenRouter stream error: ended prematurely");
    }

    return finalizeOpenRouterMessage(id, {
      content: paragraph || null,
      toolCalls,
      reasoning,
      reasoningDetails: reasoningDetails.length ? reasoningDetails : undefined,
      provider,
      usage,
      forLog: () =>
        JSON.stringify({
          finishReason,
          provider,
          usage,
          paragraph: paragraph.slice(0, 500),
          reasoningChars: reasoning.length,
          toolCalls,
        }),
    });
  } catch (error: any) {
    // Restore the real reason: fetch/reader surface our timer aborts as bare
    // AbortErrors. Caller-signal aborts pass through untouched so
    // withRetries/callWithRetries see signal.aborted and bail.
    if (abortReason && !signal?.aborted) {
      // Deadline abort with usable prose in hand: return it truncated rather
      // than discard tokens the provider already generated (and billed) and
      // retry from zero. Tool calls are excluded — a half-streamed arguments
      // fragment is unparseable JSON, so there's nothing to salvage.
      const kept = paragraph.trim();
      if (abortKind === "deadline" && kept && !toolCalls.length) {
        try {
          const message = finalizeOpenRouterMessage(id, {
            content: kept,
            toolCalls: [],
            reasoning,
            reasoningDetails: reasoningDetails.length
              ? reasoningDetails
              : undefined,
            provider,
            usage,
            forLog: () => JSON.stringify({ provider, kept: kept.slice(0, 500) }),
          });
          message.truncated = true;
          logger.log(
            id,
            `${abortReason} — returning truncated answer (${kept.length} chars, ~${estimateTokens(kept)} tokens kept)`,
          );
          return message;
        } catch {
          // Unusable partial (empty after DSML excision, unparsed markup) —
          // fall through to the discard path below.
        }
      }
      // Nothing salvageable: say how much generation is being thrown away, so
      // wasted spend is visible (aborted attempts write no usage record —
      // OpenRouter only sends `usage` in the final chunk we never receive).
      logger.error(
        id,
        `${abortReason} — discarding ~${estimateTokens(paragraph + reasoning)} generated tokens (content ${paragraph.length} chars, reasoning ${reasoning.length} chars, ${dataChunks} chunks)`,
      );
      throw new Error(abortReason);
    }
    throw error;
  } finally {
    clearTimeout(totalTimer);
    clearTimeout(stallTimer);
  }
}

/** Parse a complete (non-streamed) OpenRouter response body. */
function parseOpenRouterBody(id: Identifier, data: any): ParsedResponseMessage {
  // OpenRouter wraps upstream provider failures (rate limits, moderation,
  // model-unavailable) in an HTTP 200 whose body is `{ error: {...} }` with no
  // `choices` key. Surface that error instead of letting `choices[0]` throw a
  // cryptic "Cannot read properties of undefined (reading '0')" TypeError that
  // hides the real reason. (Mirrors the OpenAI non-streaming guard above.)
  if (data.error) {
    logger.error(id, "OpenRouter error:", data.error);
    const error = new Error(`OpenRouter error: ${data.error.message}`) as any;
    // Carry the raw error body (as the stream path does) so the retry loop can
    // classify it — moderation eviction needs code/metadata, not just message.
    error.data = data.error;
    throw error;
  }

  const answer = data.choices?.[0]?.message;
  if (!answer) {
    logger.error(id, "Missing answer in OpenRouter API response:", data);
    throw new Error("Missing answer in OpenRouter API");
  }

  return finalizeOpenRouterMessage(id, {
    content: answer.content ?? null,
    toolCalls: (answer.tool_calls ?? []).map((tc: any) => ({
      id: tc.id,
      name: tc.function.name,
      argumentsJson: tc.function.arguments,
    })),
    reasoning: answer.reasoning ?? undefined,
    reasoningDetails: answer.reasoning_details ?? undefined,
    provider: data.provider ?? undefined,
    usage: data.usage ?? null,
    forLog: () => JSON.stringify(data),
  });
}

/** Non-streaming transport — kept for callers that set `streaming: false`. */
async function callOpenRouterNonStreaming(
  id: Identifier,
  payload: OpenRouterPayload,
  requestTimeoutMs: number = OPENROUTER_NONSTREAM_TIMEOUT_MS,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  const response = await withRequestDeadline(
    "OpenRouter",
    requestTimeoutMs,
    signal,
    (mergedSignal) =>
      axios.post(openRouterEndpoint(), payload, {
        headers: {
          "content-type": "application/json",
          Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
        },
        timeout: requestTimeoutMs,
        signal: mergedSignal,
      }),
  );

  return parseOpenRouterBody(id, response.data);
}

// ─────────────────────────────────────────────────────────────────────────────
// Moderation eviction
// ─────────────────────────────────────────────────────────────────────────────
// A provider's content-moderation rejection ("Upstream error from Alibaba:
// Output data may contain inappropriate content.") is deterministic for a
// given payload — re-rolling the same provider burns the whole retry budget
// for nothing (observed 2026-07-27: 5 identical rejections per turn, ~20-25s
// of user-visible stall, before fallbackModel saved the turn). On the FIRST
// moderation-classified error, evict the refusing provider from the request's
// provider preferences (ignore += provider, order -= provider) so every
// remaining attempt reroutes; OpenRouter's next-ranked provider answers
// instead. Non-moderation errors keep plain retry semantics.

/** "AtlasCloud" / "atlas-cloud/fp4" / "Atlas Cloud" → "atlascloud" */
const normalizeProviderKey = (name: string) =>
  name.split("/")[0].toLowerCase().replace(/[^a-z0-9]/g, "");

const MODERATION_RE =
  /moderat|inappropriate content|content polic|content management|flagged/i;

/**
 * If `error` is a provider content-moderation rejection, return the refusing
 * provider's slug (in the payload's own `order` spelling when possible, since
 * OpenRouter's `ignore` wants slugs, not the display name error metadata
 * carries). Null for everything else.
 */
function moderationEvictionSlug(
  error: any,
  payload: OpenRouterPayload,
): string | null {
  const body = error?.data ?? error?.response?.data?.error;
  if (!body) return null;
  const message = String(body.message ?? "");
  const isModeration =
    body.code === 403 ||
    Array.isArray(body.metadata?.reasons) ||
    MODERATION_RE.test(message);
  if (!isModeration) return null;

  const display: string | undefined =
    body.metadata?.provider_name ??
    /Upstream error from ([^:]+):/.exec(message)?.[1];
  if (!display) return null;

  const key = normalizeProviderKey(display);
  const fromOrder = payload.provider?.order?.find(
    (entry) => normalizeProviderKey(entry) === key,
  );
  // Best effort when the provider wasn't in our order (OpenRouter default
  // routing): lowercased display name matches the slug for single-word
  // providers (alibaba, novita, baidu), which covers the observed cases.
  return fromOrder ? fromOrder.split("/")[0] : display.toLowerCase();
}

/**
 * Least time a streaming attempt is worth starting with. Below this the
 * generation cannot plausibly finish before the caller's deadline, so burning
 * a provider call (and paying for tokens that get discarded) is pure waste.
 */
export const MIN_STREAM_ATTEMPT_MS = 10_000;

/**
 * Rough token count for discard/truncation accounting only (~4 chars/token).
 * Aborted attempts never receive OpenRouter's `usage` chunk, so this is the
 * only signal for how much generation was paid for and thrown away.
 */
const estimateTokens = (text: string) => Math.round(text.length / 4);

interface OpenRouterCallOptions {
  streaming: boolean;
  streamTimeoutMs: number;
  streamDeadlineAt?: number;
  requestTimeoutMs: number;
  chunkTimeoutMs: number;
  /**
   * The original generic messages behind `payload.messages`, so the retry
   * loop can re-serialize them in the text-ref form when the model turns out
   * to have no image-capable endpoints.
   */
  genericMessages?: GenericMessage[];
}

/**
 * OpenRouter rejects `image_url` parts sent to a model with no vision-capable
 * endpoints with a routing-layer 404 ("No endpoints found that support image
 * input") before any provider is hit. The rejection is deterministic per
 * payload, so retrying it unchanged is doomed — the retry loop degrades
 * images to text refs and remembers the model instead.
 */
function isImageInputRejection(error: any): boolean {
  const body = error?.data ?? error?.response?.data?.error;
  return /no endpoints found.*image input/i.test(String(body?.message ?? ""));
}

/**
 * Per-attempt streaming budget: the smaller of the per-attempt cap and the
 * time left until the caller's absolute deadline. Returns null when too little
 * remains to be worth an attempt.
 */
function streamAttemptBudgetMs(options: OpenRouterCallOptions): number | null {
  if (options.streamDeadlineAt === undefined) return options.streamTimeoutMs;
  const remaining = options.streamDeadlineAt - Date.now();
  if (remaining < MIN_STREAM_ATTEMPT_MS) return null;
  return Math.min(options.streamTimeoutMs, remaining);
}

async function callOpenRouterWithRetries(
  id: Identifier,
  payload: OpenRouterPayload,
  retries: number = 5,
  options: OpenRouterCallOptions,
  signal?: AbortSignal,
): Promise<ParsedResponseMessage> {
  const evicted: string[] = [];
  return withRetries(
    id,
    "OpenRouter",
    () =>
      (options.streaming
        ? (() => {
            const budget = streamAttemptBudgetMs(options);
            if (budget === null) {
              // Fail fast rather than start a doomed generation. Thrown (not
              // returned) so fallbackModel still gets its chance — a cheaper
              // model may answer in the time that's left.
              const error = new Error(
                "OpenRouter stream skipped: caller deadline leaves too little time for another attempt",
              ) as any;
              error.deadlineExceeded = true;
              logger.error(id, error.message);
              throw error;
            }
            return callOpenRouterStream(
              id,
              payload,
              budget,
              options.chunkTimeoutMs,
              signal,
            );
          })()
        : callOpenRouterNonStreaming(
            id,
            payload,
            options.requestTimeoutMs,
            signal,
          )
      ).catch((error) => {
        const slug = moderationEvictionSlug(error, payload);
        if (slug && !evicted.includes(slug)) {
          evicted.push(slug);
          // Mutating this attempt-scoped payload is safe: it's built fresh per
          // callWithRetries invocation and a fallbackModel run rebuilds it.
          payload.provider = {
            ...payload.provider,
            ignore: [...(payload.provider?.ignore ?? []), slug],
            order: payload.provider?.order?.filter(
              (entry) => normalizeProviderKey(entry) !== normalizeProviderKey(slug),
            ),
          };
          logger.log(
            id,
            `OpenRouter moderation eviction: ignoring provider "${slug}" for remaining attempts`,
          );
        }
        if (isImageInputRejection(error) && options.genericMessages) {
          openRouterImageRejectedModels.add(String(payload.model));
          // Same attempt-scoped mutation as above: the retry resends the
          // pre-vision serialization (images inlined as `Image (url)` refs).
          payload.messages = prepareOpenAICompatMessages(
            options.genericMessages,
          );
          logger.log(
            id,
            `OpenRouter: ${payload.model} has no image-capable endpoints; retrying with images as text refs`,
          );
        }
        throw error;
      }),
    { retries, signal },
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN ENTRY POINT
// ─────────────────────────────────────────────────────────────────────────────

const VALID_PROVIDERS: Provider[] = ["openai", "anthropic", "google", "groq", "openrouter"];

const ENUM_PROVIDER_MAP: { values: Set<string>; provider: Provider }[] = [
  { values: new Set(Object.values(GPTModel)), provider: "openai" },
  { values: new Set(Object.values(ClaudeModel)), provider: "anthropic" },
  { values: new Set(Object.values(GeminiModel)), provider: "google" },
  { values: new Set(Object.values(GroqModel)), provider: "groq" },
  { values: new Set(Object.values(OpenRouterModel)), provider: "openrouter" },
];

export function parseModelString(model: string): { provider: Provider; modelId: string } {
  const colonIndex = model.indexOf(":");

  if (colonIndex !== -1) {
    const prefix = model.substring(0, colonIndex);

    if (VALID_PROVIDERS.includes(prefix as Provider)) {
      const modelId = model.substring(colonIndex + 1);

      if (!modelId) {
        throw new Error(
          `Empty model ID in model string '${model}'. Expected format: 'provider:model-id'`,
        );
      }

      return { provider: prefix as Provider, modelId };
    }

    // Prefix isn't a known provider — fall through to enum lookup
    // (handles model values that contain colons, e.g. OpenRouter "google/gemma-4-31b-it:free")
  }

  // Fallback: check enum values
  for (const { values, provider } of ENUM_PROVIDER_MAP) {
    if (values.has(model)) {
      return { provider, modelId: model };
    }
  }

  // If string had a colon but wasn't a known provider, give a specific error
  if (colonIndex !== -1) {
    const prefix = model.substring(0, colonIndex);
    throw new Error(
      `Unknown provider '${prefix}' in model string '${model}'. Valid providers: ${VALID_PROVIDERS.join(", ")}`,
    );
  }

  throw new Error(
    `Unable to determine provider for model '${model}'. Use a provider prefix (e.g. 'openai:${model}') or a known model enum value. Valid providers: ${VALID_PROVIDERS.join(", ")}`,
  );
}

export async function callWithRetries(
  id: string | string[],
  aiPayload: GenericPayload,
  aiConfig?: OpenAIConfig | AnthropicAIConfig,
  retries: number = 5,
  chunkTimeoutMs: number = 15_000,
): Promise<ParsedResponseMessage> {
  try {
    const { provider, modelId } = parseModelString(aiPayload.model);
    const routingPayload = { ...aiPayload, model: modelId as AnyModel };
    // Per-attempt HTTP timeout, honored by every adapter. Default applied once
    // here so all providers share it; callers override via payload.requestTimeoutMs.
    const requestTimeoutMs = aiPayload.requestTimeoutMs ?? 120_000;
    const signal = aiPayload.signal;

    let result: ParsedResponseMessage;
    switch (provider) {
      case "anthropic":
        result = await callAnthropicWithRetries(
          id,
          await prepareAnthropicPayload(id, routingPayload),
          aiConfig as AnthropicAIConfig,
          retries,
          requestTimeoutMs,
          signal,
        );
        break;

      case "openai":
        result = await callOpenAiWithRetries(
          id,
          await prepareOpenAIPayload(id, routingPayload),
          aiConfig as OpenAIConfig,
          retries,
          chunkTimeoutMs,
          requestTimeoutMs,
          signal,
        );
        break;

      case "groq":
        result = await callGroqWithRetries(
          id,
          prepareGroqPayload(routingPayload),
          retries,
          requestTimeoutMs,
          signal,
        );
        break;

      case "google":
        result = await callGoogleAIWithRetries(
          id,
          await prepareGoogleAIPayload(id, routingPayload),
          retries,
          requestTimeoutMs,
          signal,
        );
        break;

      case "openrouter":
        // OpenRouter streams by default. The two transports have independent
        // budgets on purpose: `requestTimeoutMs` only bounds non-streaming
        // attempts (OpenRouter default 180s, tighter 120s global default kept
        // for other providers), while streaming gets `streamTimeoutMs`
        // (default 600s) total + the per-useful-chunk stall timeout, further
        // bounded by `streamDeadlineAt` when the caller has a turn budget.
        result = await callOpenRouterWithRetries(
          id,
          prepareOpenRouterPayload(routingPayload),
          retries,
          {
            streaming: aiPayload.streaming ?? true,
            streamTimeoutMs:
              aiPayload.streamTimeoutMs ?? OPENROUTER_STREAM_TIMEOUT_MS,
            streamDeadlineAt: aiPayload.streamDeadlineAt,
            requestTimeoutMs:
              aiPayload.requestTimeoutMs ?? OPENROUTER_NONSTREAM_TIMEOUT_MS,
            chunkTimeoutMs,
            genericMessages: routingPayload.messages,
          },
          signal,
        );
        break;
    }

    // Attribution stamp: adapters that know the actual serving provider set it
    // themselves (OpenRouter returns the upstream provider it routed to); for
    // the rest, the SDK provider name is the answer. Because fallback recurses
    // through callWithRetries, the stamp always comes from the invocation whose
    // model actually answered.
    result.provider ??= provider;
    return result;
  } catch (error) {
    // Caller cancelled — reject immediately, never fall back to another model.
    if (aiPayload.signal?.aborted) throw error;
    if (aiPayload.fallbackModel) {
      logger.error(
        id,
        `Primary model ${aiPayload.model} failed, falling back to ${aiPayload.fallbackModel}`,
        {
          error: error instanceof Error ? error.message : error,
          cause:
            error instanceof Error && (error as any).cause instanceof Error
              ? (error as any).cause.message
              : undefined,
        },
      );
      return callWithRetries(
        id,
        {
          ...aiPayload,
          model: aiPayload.fallbackModel,
          fallbackModel: undefined,
        },
        aiConfig,
        retries,
        chunkTimeoutMs,
      );
    }
    throw error;
  }
}
