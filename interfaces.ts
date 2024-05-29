export enum ClaudeModel {
  HAIKU = "claude-3-haiku-20240307",
  SONNET = "claude-3-sonnet-20240229",
  OPUS = "claude-3-opus-20240229",
}

export enum GPTModel {
  // https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
  GPT35_0613 = "gpt-3.5-turbo-0613",
  GPT35_0613_16K = "gpt-3.5-turbo-16k-0613",
  GPT35_0125 = "gpt-3.5-turbo-0125",
  GPT4_1106_PREVIEW = "gpt-4-1106-preview",
  GPT4_0125_PREVIEW = "gpt-4-0125-preview",
  GPT4_0409 = "gpt-4-turbo-2024-04-09",
  GPT4O = "gpt-4o",
}

export enum GroqModel {
  LLAMA_3_70B_8192 = "llama3-70b-8192",
}

export enum GeminiModel {
  GEMINI_15_PRO = "gemini-1.5-pro-latest",
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
  model: GroqModel;
  messages: OpenAIMessage[];
  tools?: FunctionWrapped[];
  tool_choice?:
    | "none"
    | "auto"
    | { type: "function"; function: { name: string } };
  temperature?: number;

  functions?: any[]; // Deprecate this
}

export interface OpenAIPayload {
  model: GPTModel;
  messages: OpenAIMessage[];
  tools?: FunctionWrapped[];
  tool_choice?:
    | "none"
    | "auto"
    | { type: "function"; function: { name: string } };
  temperature?: number;
}

export interface AnthropicAIPayload {
  model: ClaudeModel;
  messages: AnthropicAIMessage[];
  functions?: any[]; // TODO type this JSON schema
  temperature?: number;
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

export type GoogleAIPart = GoogleAITextPart | GoogleAIInlineDataPart;
export interface GoogleAIMessage {
  role: "user" | "model";
  parts: GoogleAIPart[];
}
export interface GoogleAIPayload {
  model: GeminiModel;
  messages: GoogleAIMessage[];
  tools?: {
    functionDeclarations: FunctionDefinition[];
  };
  systemInstruction?: string;
}

export interface GenericPayload {
  model: GPTModel | ClaudeModel | GroqModel | GeminiModel;
  messages: GenericMessage[];
  functions?: FunctionDefinition[];
  function_call?: "none" | "auto" | { name: string };
  temperature?: number;
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
