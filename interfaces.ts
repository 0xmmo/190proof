/** @deprecated Use provider prefix strings instead, e.g. `"anthropic:claude-sonnet-4-5"` */
export enum ClaudeModel {
  HAIKU_3 = "claude-3-haiku-20240307",
  SONNET_3 = "claude-3-sonnet-20240229",
  OPUS_3 = "claude-3-opus-20240229",
  HAIKU_3_5 = "claude-3-5-haiku-20241022",
  SONNET_3_5 = "claude-3-5-sonnet-20241022",
  SONNET_4 = "claude-sonnet-4-20250514",
  OPUS_4 = "claude-opus-4-20250514",
  OPUS_4_1 = "claude-opus-4-1",
  HAIKU_4_5 = "claude-haiku-4-5",
  SONNET_4_5 = "claude-sonnet-4-5",
  OPUS_4_5 = "claude-opus-4-5",
}

/** @deprecated Use provider prefix strings instead, e.g. `"openai:gpt-4o"` */
export enum GPTModel {
  GPT35_0613 = "gpt-3.5-turbo-0613",
  GPT35_0613_16K = "gpt-3.5-turbo-16k-0613",
  GPT35_0125 = "gpt-3.5-turbo-0125",
  GPT4_1106_PREVIEW = "gpt-4-1106-preview",
  GPT4_0125_PREVIEW = "gpt-4-0125-preview",
  GPT4_0409 = "gpt-4-turbo-2024-04-09",
  GPT4O = "gpt-4o",
  GPT4O_MINI = "gpt-4o-mini",
  O1_PREVIEW = "o1-preview",
  O1_MINI = "o1-mini",
  O3_MINI = "o3-mini",
  GPT4_1 = "gpt-4.1",
  GPT4_1_MINI = "gpt-4.1-mini",
  GPT4_1_NANO = "gpt-4.1-nano",
  GPT5 = "gpt-5",
  GPT5_MINI = "gpt-5-mini",
}

/** @deprecated Use provider prefix strings instead, e.g. `"groq:llama-3.3-70b-versatile"` */
export enum GroqModel {
  LLAMA_3_70B_8192 = "llama3-70b-8192",
  LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile",
  QWEN3_32B = "qwen/qwen3-32b",
  DEEPSEEK_R1_DISTILL_LLAMA_70B = "deepseek-r1-distill-llama-70b",
}

/** @deprecated Use provider prefix strings instead, e.g. `"openrouter:qwen/qwen3.6-plus:free"` */
export enum OpenRouterModel {
  GEMMA_4_31B_IT_FREE = "google/gemma-4-31b-it:free",
  GEMMA_4_31B_IT = "google/gemma-4-31b-it",
}

/** @deprecated Use provider prefix strings instead, e.g. `"google:gemini-2.0-flash"` */
export enum GeminiModel {
  GEMINI_1_5_PRO = "gemini-1.5-pro-latest",
  GEMINI_EXP_1206 = "gemini-exp-1206",
  GEMINI_2_0_FLASH = "gemini-2.0-flash",
  GEMINI_2_0_FLASH_EXP_IMAGE_GENERATION = "gemini-2.0-flash-exp-image-generation",
  GEMINI_2_0_FLASH_THINKING_EXP = "gemini-2.0-flash-thinking-exp",
  GEMINI_2_0_FLASH_THINKING_EXP_01_21 = "gemini-2.0-flash-thinking-exp-01-21",
  GEMINI_2_5_FLASH_PREVIEW_04_17 = "gemini-2.5-flash-preview-04-17",
  GEMINI_3_FLASH_PREVIEW = "gemini-3-flash-preview",
  GEMINI_3_1_FLASH_LITE_PREVIEW = "gemini-3.1-flash-lite-preview",
}

export interface GenericError {
  message: string;
}

export enum ContentType {
  TEXT = "text",
  ATTACHMENT = "attachment",
}

export type AIChainResponse = {
  content: string | null;
  contentType?: ContentType;
  functionCalls: FunctionCall[];
};

/**
 * A single conversation turn passed to `callWithRetries`. The SDK serializes
 * these into each provider's native message format.
 *
 * ## Multi-turn tool calls
 * To continue after the model calls a tool, append the model's own assistant
 * turn and then the tool result, then call again:
 *
 * 1. Read the assistant turn off the response: `function_calls` (each with an
 *    `id`) and — for reasoning models — `reasoning` / `reasoningDetails`.
 * 2. Push back an `assistant` message carrying those same `functionCalls` (and,
 *    to keep the model's chain-of-thought, the captured reasoning fields).
 * 3. Push back one `role: "tool"` message whose `toolResults` answer each call
 *    by `toolCallId`.
 *
 * The SDK is a pure transport: it does NOT echo reasoning or pair results for
 * you — it serializes exactly what you put here. Every tool/reasoning field is
 * optional, so a plain `{ role, content }` message behaves as before.
 *
 * @example
 * // turn 1 — model asks for the weather
 * const a = await callWithRetries(id, { model, messages, functions });
 * // a.function_calls -> [{ id: "call_abc", name: "get_weather", arguments: { city: "Tokyo" } }]
 *
 * // turn 2 — feed the call + its result back
 * const next = await callWithRetries(id, { model, functions, messages: [
 *   ...messages,
 *   { role: "assistant", content: a.content ?? "", functionCalls: a.function_calls,
 *     reasoning: a.reasoning, reasoningDetails: a.reasoningDetails },
 *   { role: "tool", content: "", toolResults: [
 *     { toolCallId: "call_abc", name: "get_weather", content: '{"tempC":22}' } ] },
 * ]});
 */
export interface GenericMessage {
  /**
   * `"tool"` carries tool results (see `toolResults`) back to the model and
   * must immediately follow the `assistant` turn that made the matching calls.
   */
  role: "user" | "assistant" | "system" | "tool";
  /** Plain-text content. Use `""` for a tool-call-only or tool-result turn. */
  content: string;
  timestamp?: string;
  files?: File[];
  /**
   * Tool calls the model made on an `assistant` turn. Pass back the `id`s you
   * received in `ParsedResponseMessage.function_calls` so the provider can pair
   * them with the `toolResults` that follow.
   */
  functionCalls?: FunctionCall[];
  /**
   * Tool outputs on a `role: "tool"` message — one entry per call (parallel
   * calls produce several). Each `toolCallId` must match a `FunctionCall.id`
   * from the preceding assistant turn.
   */
  toolResults?: ToolResult[];
  /**
   * Optional reasoning string to echo back on an `assistant` turn (e.g.
   * OpenRouter/DeepSeek `reasoning`). Round-tripping it keeps the model's
   * chain-of-thought across tool calls; some thinking models (DeepSeek V4)
   * require it to avoid a 400 on the next turn.
   */
  reasoning?: string;
  /**
   * Optional structured reasoning to echo back on an `assistant` turn
   * (OpenRouter `reasoning_details`, or Anthropic `thinking` /
   * `redacted_thinking` blocks captured in
   * `ParsedResponseMessage.reasoningDetails`). Preserves the signatures /
   * encrypted payloads that providers validate on round-trip.
   *
   * Safe to echo regardless of which provider serves the next call: each
   * serializer keeps only its own provider's block shapes (Anthropic keeps
   * `thinking`/`redacted_thinking`; OpenAI-compat keeps `reasoning.*`), so a
   * cross-provider fallback drops foreign blocks instead of 400ing.
   */
  reasoningDetails?: any;
}

/**
 * The result of executing one tool call, fed back to the model on a
 * `role: "tool"` message. The SDK maps this to each provider's native shape:
 * OpenAI/Groq/OpenRouter `{ role: "tool", tool_call_id, content }`, Anthropic a
 * `tool_result` block, Google a `functionResponse` part.
 */
export interface ToolResult {
  /** The `id` of the `FunctionCall` this answers (from the prior assistant turn). */
  toolCallId: string;
  /**
   * The tool/function name. Required by Anthropic and Google on round-trip; the
   * SDK falls back to the matching call's name when omitted, but supplying it is
   * recommended.
   */
  name?: string;
  /** The tool output, serialized to a string (JSON or plain text). */
  content: string;
}

export interface File {
  mimeType: string;
  url?: string;
  data?: string;
}

/**
 * A tool/function call on the wire for OpenAI-compatible APIs (OpenAI, Groq,
 * OpenRouter).
 */
export interface OpenAIToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    /** JSON-stringified arguments object. */
    arguments: string;
  };
}

export interface OpenAIMessage {
  role: "user" | "assistant" | "system" | "tool";
  /** `null` for an assistant turn that is tool-calls-only. */
  content: string | OpenAIContentBlock[] | null;
  /** Present on an assistant turn that called tools. */
  tool_calls?: OpenAIToolCall[];
  /** Present on a `role: "tool"` message; matches the originating `OpenAIToolCall.id`. */
  tool_call_id?: string;
  /** Reasoning echoed back on an assistant turn (OpenRouter/DeepSeek; OAI-compatible proxies). */
  reasoning?: string;
  /** Structured reasoning echoed back on an assistant turn (OpenRouter `reasoning_details`). */
  reasoning_details?: any;
}

export type OpenAIContentBlock =
  | OpenAITextContentBlock
  | OpenAIImageContentBlock
  | OpenAIAudioContentBlock;

export interface OpenAITextContentBlock {
  type: "text";
  text: string;
}

export interface OpenAIImageContentBlock {
  type: "image_url";
  image_url: {
    url: string; // URL to the image, can also be a base64 string
  };
}

export interface OpenAIAudioContentBlock {
  type: "audio_url";
  audio_url: {
    url: string; // URL to the audio, can also be a base64 string
  };
}

export interface AnthropicAIMessage {
  role: "user" | "assistant" | "system";
  content: string | AnthropicContentBlock[];
}

export type AnthropicContentBlock =
  | AnthropicTextContentBlock
  | AnthropicImageContentBlock
  | AnthropicToolUseBlock
  | AnthropicToolResultBlock
  | AnthropicThinkingBlock
  | AnthropicRedactedThinkingBlock;

export interface AnthropicTextContentBlock {
  type: "text";
  text: string;
}

export interface AnthropicImageContentBlock {
  type: "image";
  source: {
    type: "base64";
    media_type: "image/jpeg" | "image/png" | "image/gif" | "image/webp";
    data: string; // Must be a base64 string
  };
}

/** A tool call on an Anthropic assistant turn. */
export interface AnthropicToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: Record<string, any>;
}

/** A tool result, carried on a (user-role) Anthropic message. */
export interface AnthropicToolResultBlock {
  type: "tool_result";
  tool_use_id: string;
  content: string;
}

/** A reasoning block round-tripped on an assistant turn; `signature` is validated by Anthropic. */
export interface AnthropicThinkingBlock {
  type: "thinking";
  thinking: string;
  signature: string;
}

/** An encrypted reasoning block round-tripped verbatim on an assistant turn. */
export interface AnthropicRedactedThinkingBlock {
  type: "redacted_thinking";
  data: string;
}

export interface OpenAIResponseMessage {
  role: "assistant";
  content: string | null;
  function_call: {
    name: string;
    arguments: string; // unparsed arguments object
  } | null;
}

export interface ParsedResponseMessage {
  role: "assistant";
  content: string | null;
  /** First of `function_calls` (backward-compat); carries `id` when the provider returns one. */
  function_call: FunctionCall | null;
  /** All tool calls the model made this turn, each with an `id` for round-tripping. */
  function_calls: FunctionCall[];
  files: File[];
  /**
   * Reasoning string from reasoning models (OpenRouter/DeepSeek `reasoning`).
   * Echo back via `GenericMessage.reasoning` to preserve chain-of-thought across
   * tool calls. Undefined when the model/provider returns none.
   */
  reasoning?: string;
  /**
   * Structured reasoning (OpenRouter `reasoning_details`, or Anthropic
   * `thinking` / `redacted_thinking` blocks). Echo back verbatim via
   * `GenericMessage.reasoningDetails` — it carries signatures some providers
   * validate. Undefined when the model/provider returns none.
   */
  reasoningDetails?: any;
  /**
   * Who actually served the response. For OpenRouter this is the upstream
   * provider from the response body (e.g. "Baidu", "Morph") — the routing
   * decision OpenRouter made, not the requested model slug. For direct
   * providers it's the SDK provider name ("anthropic", "openai", "google",
   * "groq"). On model fallback it reflects the model that answered, so a
   * mismatch with the requested model's provider reveals the fallback.
   */
  provider?: string;
  /**
   * True when the answer is INCOMPLETE: the streamed generation was cut at the
   * caller's deadline and the partial prose is returned instead of discarded.
   * Content is mid-sentence (or mid-file) by definition — surface it to the
   * end user as truncated rather than presenting it as a finished answer.
   * Never set on tool-call turns (a half-streamed arguments fragment can't be
   * salvaged) and never on a normal completion.
   */
  truncated?: boolean;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    /** Prompt tokens served from the provider's cache (subset of prompt_tokens). */
    cached_tokens?: number;
    /**
     * Reasoning/thinking tokens spent before the visible answer (subset of
     * completion_tokens on some providers, separate on others). Currently
     * populated from Google's `usageMetadata.thoughtsTokenCount`; undefined
     * when the provider reports none.
     */
    thoughts_tokens?: number;
  } | null;
}

export interface FunctionCall {
  /**
   * Provider tool-call id, surfaced on responses and used to pair a call with
   * its `ToolResult.toolCallId` on the next turn. The SDK synthesizes
   * `call_<index>` for providers that don't return one (e.g. Google).
   */
  id?: string;
  name: string;
  arguments: Record<string, any>;
  /**
   * Opaque per-call signature that must be echoed back verbatim on round-trip.
   * Currently Google's `thoughtSignature` — Gemini REQUIRES it on functionCall
   * parts in multi-turn tool use (a missing one 400s the next request). The SDK
   * captures it on responses and re-emits it when you pass the call back;
   * undefined for providers that don't use one.
   */
  thoughtSignature?: string;
}

export interface OpenAIConfig {
  service: "azure" | "openai";
  apiKey: string;
  /**
   * Override the base URL for the OpenAI service (e.g. an OpenAI-compatible
   * proxy or self-hosted endpoint). The path `/chat/completions` is appended.
   * Ignored when `service === "azure"` (use `modelConfigMap` instead).
   * Defaults to `https://api.openai.com/v1`.
   */
  baseUrl?: string;
  orgId?: string;
  modelConfigMap?: Record<
    GPTModel,
    {
      resource: string;
      deployment: string;
      apiVersion: string;
      apiKey: string;
      endpoint?: string;
    }
  >;
}

export interface AnthropicAIConfig {
  service: "anthropic" | "bedrock";
}

export interface FunctionDefinition {
  name: string;
  description?: string;
  parameters: Record<string, any>;
}

interface FunctionWrapped {
  type: "function";
  function: FunctionDefinition;
}

export interface GroqPayload {
  model: GroqModel | string;
  messages: OpenAIMessage[];
  tools?: FunctionWrapped[];
  tool_choice?:
    | "none"
    | "auto"
    | { type: "function"; function: { name: string } };
  temperature?: number;

  functions?: any[]; // Deprecate this
}

/**
 * OpenRouter provider-routing preferences, forwarded verbatim as the request
 * body's `provider` field. See https://openrouter.ai/docs/guides/routing/provider-selection
 */
export interface OpenRouterProviderPreferences {
  /** Ordered list of provider slugs to try first (e.g. ["baidu", "siliconflow"]). */
  order?: string[];
  /** Restrict routing to exactly these provider slugs. */
  only?: string[];
  /** Provider slugs to exclude. */
  ignore?: string[];
  /** Only use providers serving these quantizations (e.g. ["fp8"]). */
  quantizations?: string[];
  /** When false, never fall back to providers outside `order`/`only`. */
  allow_fallbacks?: boolean;
  /** Override the default sort ("price" | "throughput" | "latency"). */
  sort?: string;
}

export interface OpenRouterPayload {
  model: OpenRouterModel | string;
  messages: OpenAIMessage[];
  tools?: FunctionWrapped[];
  tool_choice?:
    | "none"
    | "auto"
    | { type: "function"; function: { name: string } };
  temperature?: number;
  provider?: OpenRouterProviderPreferences;
  reasoning_effort?: string;
  /** Set by the adapter, never by callers: SSE streaming on/off. */
  stream?: boolean;
  /**
   * Set by the adapter on streamed requests: OpenRouter's usage-accounting
   * flag, which makes the final SSE chunk carry the `usage` object.
   */
  usage?: { include: boolean };
}

export interface OpenAIPayload {
  model: GPTModel | string;
  messages: OpenAIMessage[];
  tools?: FunctionWrapped[];
  tool_choice?:
    | "none"
    | "auto"
    | { type: "function"; function: { name: string } };
  reasoning_effort?: string;
}

export interface AnthropicAIPayload {
  model: ClaudeModel | string;
  messages: AnthropicAIMessage[];
  functions?: any[]; // TODO type this JSON schema
  temperature?: number;
  system?: string;
  tool_choice?: { type: "none" | "auto" } | { type: "tool"; name: string };
}

export interface GoogleAITextPart {
  text: string;
}

export interface GoogleAIInlineDataPart {
  inlineData: {
    mimeType: string;
    data: string;
  };
}

export interface GoogleAIFileDataPart {
  fileData: {
    mimeType: string;
    fileUri: string;
  };
}

/** A tool call on a Google `model` turn. */
export interface GoogleAIFunctionCallPart {
  functionCall: {
    id?: string;
    name: string;
    args: Record<string, any>;
  };
  /** Echoed back verbatim on round-trip — Gemini requires it for tool use. */
  thoughtSignature?: string;
}

/** A tool result, carried on a Google `user` turn. */
export interface GoogleAIFunctionResponsePart {
  functionResponse: {
    id?: string;
    name: string;
    response: Record<string, any>;
  };
}

export type GoogleAIPart =
  | GoogleAITextPart
  | GoogleAIInlineDataPart
  | GoogleAIFileDataPart
  | GoogleAIFunctionCallPart
  | GoogleAIFunctionResponsePart;
export interface GoogleAIMessage {
  role: "user" | "model";
  parts: GoogleAIPart[];
}
export interface GoogleAIPayload {
  model: GeminiModel | string;
  messages: GoogleAIMessage[];
  tools?: {
    functionDeclarations: FunctionDefinition[];
  };
  toolConfig?: {
    functionCallingConfig: { mode: "NONE" | "AUTO" | "ANY" };
  };
  systemInstruction?: string;
  thinkingConfig?: Record<string, unknown>;
}

export type Provider = "openai" | "anthropic" | "google" | "groq" | "openrouter";

export type AnyModel = GPTModel | ClaudeModel | GroqModel | GeminiModel | OpenRouterModel | (string & {});

export interface GenericPayload {
  model: AnyModel;
  messages: GenericMessage[];
  functions?: FunctionDefinition[];
  function_call?: "none" | "auto" | { name: string };
  temperature?: number;
  fallbackModel?: AnyModel;
  /**
   * Google-only: forwarded verbatim as `generationConfig.thinkingConfig` on
   * the Gemini request — e.g. `{ thinkingBudget: 0 }` to disable thinking or
   * `{ thinkingLevel: "HIGH" }` on models that take a level. Ignored by all
   * other adapters. Shapes are model-specific and validated by Google, not
   * the SDK.
   */
  thinkingConfig?: Record<string, unknown>;
  /**
   * OpenRouter-only: provider-routing preferences. Ignored by non-OpenRouter
   * adapters. Forwarded as the request body's `provider` field.
   */
  provider?: OpenRouterProviderPreferences;
  /**
   * OpenAI and OpenRouter: forwarded as `reasoning_effort` on the request.
   * Valid values are model-dependent
   * (`none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`).
   * Direct OpenAI: reasoning-by-default models (gpt-5.6 family) 400 on
   * /chat/completions when function tools are present unless this is
   * explicitly `"none"` — their implicit default is `medium`. Via OpenRouter
   * the same models accept tools at any effort (OpenRouter fronts
   * /v1/responses), so omitting this runs them at their native default.
   * Ignored by all other adapters.
   */
  reasoningEffort?: string;
  /**
   * Per-request HTTP timeout in ms for the underlying provider call (applied
   * per attempt, not across retries). Honored by all adapters (Anthropic,
   * Google, OpenAI, OpenRouter, Groq); defaults to 120s when omitted. Raise it
   * for slow, large generations (e.g. single-file app codegen) so a long-but-
   * valid response isn't cut short.
   */
  requestTimeoutMs?: number;
  /**
   * OpenRouter-only: stream the completion over SSE instead of waiting for a
   * single JSON body. Defaults to true. Streaming attempts are bounded by
   * `streamTimeoutMs` (total) plus a per-useful-chunk stall timeout — NOT by
   * `requestTimeoutMs`, which only governs non-streaming attempts (default
   * 180s for OpenRouter). Set to false to force the old non-streaming path.
   */
  streaming?: boolean;
  /**
   * OpenRouter-only: total wall-clock budget in ms for one streaming attempt
   * (connect + full generation). Defaults to 600s. Independent of
   * `requestTimeoutMs` by design: a healthy long generation keeps streaming
   * useful chunks and may run far past any sane non-streaming deadline, while
   * a hung one is killed much earlier by the per-useful-chunk stall timeout
   * (`chunkTimeoutMs` argument of `callWithRetries`, default 15s — reset only
   * by chunks that advance the output, never by keep-alive bytes/comments).
   */
  streamTimeoutMs?: number;
  /**
   * OpenRouter-only: absolute wall-clock deadline (epoch ms) for the whole
   * call INCLUDING retries — the caller's turn budget, not a per-attempt one.
   * Each streaming attempt gets `min(streamTimeoutMs, deadline - now)`, and
   * once too little time remains to be worth an attempt the call fails fast
   * instead of starting a generation that cannot finish.
   *
   * Without it, a per-attempt budget is re-granted on every retry, so a slow
   * generation can outlive the caller's own turn deadline and get killed with
   * nothing to show (2026-07-28: a 538s completion finished just as the
   * caller's 585s turn budget expired, and the reply was discarded).
   */
  streamDeadlineAt?: number;
  /**
   * Optional caller-supplied cancellation signal. When it aborts, the in-flight
   * provider request is cancelled and `callWithRetries` rejects immediately —
   * it does NOT retry or fall back (both retry loop and fallback branch bail on
   * `signal.aborted`). Threaded through every adapter to the underlying
   * fetch/axios/SDK call, mirroring `requestTimeoutMs`.
   */
  signal?: AbortSignal;
}

export interface OpenAIBody {
  choices: {
    message: OpenAIResponseMessage;
  }[];
  error?: {
    code: string;
  };
  usage: {
    completion_tokens: number;
    prompt_tokens: number;
    total_tokens: number;
    prompt_tokens_details?: {
      cached_tokens?: number;
    };
  };
}
