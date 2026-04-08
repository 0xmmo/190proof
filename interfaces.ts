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

export interface GenericMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
  files?: File[];
  functionCalls?: FunctionCall[];
}

export interface File {
  mimeType: string;
  url?: string;
  data?: string;
}

export interface OpenAIMessage {
  role: "user" | "assistant" | "system";
  content: string | OpenAIContentBlock[];
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
  | AnthropicImageContentBlock;

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
  function_call: FunctionCall | null;
  function_calls: FunctionCall[];
  files: File[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  } | null;
}

export interface FunctionCall {
  name: string;
  arguments: Record<string, any>;
}

export interface OpenAIResponseMessage {
  role: "assistant";
  content: string | null;
  function_call: {
    name: string;
    arguments: string; // unparsed arguments object
  } | null;
}

export interface FunctionCall {
  name: string;
  arguments: Record<string, any>;
}

export interface OpenAIConfig {
  service: "azure" | "openai";
  apiKey: string;
  baseUrl: string;
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

export interface OpenRouterPayload {
  model: OpenRouterModel | string;
  messages: OpenAIMessage[];
  tools?: FunctionWrapped[];
  tool_choice?:
    | "none"
    | "auto"
    | { type: "function"; function: { name: string } };
  temperature?: number;
}

export interface OpenAIPayload {
  model: GPTModel | string;
  messages: OpenAIMessage[];
  tools?: FunctionWrapped[];
  tool_choice?:
    | "none"
    | "auto"
    | { type: "function"; function: { name: string } };
}

export interface AnthropicAIPayload {
  model: ClaudeModel | string;
  messages: AnthropicAIMessage[];
  functions?: any[]; // TODO type this JSON schema
  temperature?: number;
  system?: string;
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

export type GoogleAIPart = GoogleAITextPart | GoogleAIInlineDataPart | GoogleAIFileDataPart;
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
  systemInstruction?: string;
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
  };
}
