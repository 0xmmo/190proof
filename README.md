# 190proof

A unified interface for interacting with multiple AI providers including **OpenAI**, **Anthropic**, **Google**, **Groq**, **OpenRouter**, and **AWS Bedrock**. This package provides a consistent API for making requests to different LLM providers while handling retries, streaming, and multimodal inputs.

## Features

Fully-local unified interface across multiple AI providers that includes:

- 🛠️ Consistent function/tool calling across all providers
- 💬 Consistent message alternation & system instructions
- 🖼️ Image format & size normalization
- 🔄 Automatic retries with configurable attempts
- 📡 Streaming by default
- ☁️ Cloud service providers supported (Azure, AWS Bedrock)

## Installation

```bash
npm install 190proof
```

## Usage

### Basic Example

```typescript
import { callWithRetries } from "190proof";
import { GPTModel, GenericPayload } from "190proof/interfaces";

const payload: GenericPayload = {
  model: GPTModel.GPT4O_MINI,
  messages: [
    {
      role: "user",
      content: "Tell me a joke.",
    },
  ],
};

const response = await callWithRetries("my-request-id", payload);
console.log(response.content);
```

### Using Different Providers

```typescript
import { callWithRetries } from "190proof";
import {
  ClaudeModel,
  GeminiModel,
  GroqModel,
  OpenRouterModel,
  GenericPayload,
} from "190proof/interfaces";

// Anthropic
const claudePayload: GenericPayload = {
  model: ClaudeModel.SONNET_4,
  messages: [{ role: "user", content: "Hello!" }],
};

// Google
const geminiPayload: GenericPayload = {
  model: GeminiModel.GEMINI_2_0_FLASH,
  messages: [{ role: "user", content: "Hello!" }],
};

// Groq
const groqPayload: GenericPayload = {
  model: GroqModel.LLAMA_3_70B_8192,
  messages: [{ role: "user", content: "Hello!" }],
};

// OpenRouter
const openRouterPayload: GenericPayload = {
  model: OpenRouterModel.QWEN3_6_PLUS_FREE,
  messages: [{ role: "user", content: "Hello!" }],
};

const response = await callWithRetries("request-id", claudePayload);
```

### With Function Calling

```typescript
const payload: GenericPayload = {
  model: GPTModel.GPT4O,
  messages: [
    {
      role: "user",
      content: "What is the capital of France?",
    },
  ],
  functions: [
    {
      name: "get_country_capital",
      description: "Get the capital of a given country",
      parameters: {
        type: "object",
        properties: {
          country_name: {
            type: "string",
            description: "The name of the country",
          },
        },
        required: ["country_name"],
      },
    },
  ],
};

const response = await callWithRetries("function-call-example", payload);
// response.function_call contains { name: string, arguments: Record<string, any> }
```

### With Images

```typescript
const payload: GenericPayload = {
  model: ClaudeModel.SONNET_4,
  messages: [
    {
      role: "user",
      content: "What's in this image?",
      files: [
        {
          mimeType: "image/jpeg",
          url: "https://example.com/image.jpg",
        },
      ],
    },
  ],
};

const response = await callWithRetries("image-example", payload);
```

### With System Messages

```typescript
const payload: GenericPayload = {
  model: GeminiModel.GEMINI_2_0_FLASH,
  messages: [
    {
      role: "system",
      content: "You are a helpful assistant that speaks in a friendly tone.",
    },
    {
      role: "user",
      content: "Tell me about yourself.",
    },
  ],
};

const response = await callWithRetries("system-message-example", payload);
```

## Supported Models

### OpenAI Models

- `gpt-3.5-turbo-0613`
- `gpt-3.5-turbo-16k-0613`
- `gpt-3.5-turbo-0125`
- `gpt-4-1106-preview`
- `gpt-4-0125-preview`
- `gpt-4-turbo-2024-04-09`
- `gpt-4o`
- `gpt-4o-mini`
- `o1-preview`
- `o1-mini`
- `o3-mini`
- `gpt-4.1`
- `gpt-4.1-mini`
- `gpt-4.1-nano`
- `gpt-5`
- `gpt-5-mini`

### Anthropic Models

- `claude-3-haiku-20240307`
- `claude-3-sonnet-20240229`
- `claude-3-opus-20240229`
- `claude-3-5-haiku-20241022`
- `claude-3-5-sonnet-20241022`
- `claude-sonnet-4-20250514`
- `claude-opus-4-20250514`
- `claude-opus-4-1`
- `claude-haiku-4-5`
- `claude-sonnet-4-5`
- `claude-opus-4-5`

### Google Models

- `gemini-1.5-pro-latest`
- `gemini-exp-1206`
- `gemini-2.0-flash`
- `gemini-2.0-flash-exp-image-generation`
- `gemini-2.0-flash-thinking-exp`
- `gemini-2.0-flash-thinking-exp-01-21`
- `gemini-2.5-flash-preview-04-17`
- `gemini-3-flash-preview`
- `gemini-3.1-flash-lite-preview`

### Groq Models

- `llama3-70b-8192`
- `deepseek-r1-distill-llama-70b`

### OpenRouter Models

- `qwen/qwen3.6-plus:free`

## Environment Variables

Set the following environment variables for the providers you want to use:

```bash
# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key

# Google
GEMINI_API_KEY=your-gemini-api-key

# Groq
GROQ_API_KEY=your-groq-api-key

# OpenRouter
OPENROUTER_API_KEY=your-openrouter-api-key

# AWS Bedrock (for Anthropic via Bedrock)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
```

## API Reference

### `callWithRetries(identifier, payload, config?, retries?, chunkTimeoutMs?)`

Main function to make requests to any supported AI provider.

#### Parameters

- `identifier`: `string | string[]` - Unique identifier for the request (used for logging)
- `payload`: `GenericPayload` - Request payload containing model, messages, and optional functions
- `config`: `OpenAIConfig | AnthropicAIConfig` - Optional configuration for the specific provider
- `retries`: `number` - Number of retry attempts (default: 5)
- `chunkTimeoutMs`: `number` - Timeout for streaming chunks in ms (default: 15000)

#### Returns

`Promise<ParsedResponseMessage>`:

```typescript
interface ParsedResponseMessage {
  role: "assistant";
  content: string | null;
  function_call: FunctionCall | null;
  function_calls: FunctionCall[];
  files: File[]; // For models that return files (e.g., image generation)
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  } | null; // null when streaming
}
```

### Configuration Options

#### OpenAI Config

```typescript
interface OpenAIConfig {
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
```

#### Anthropic Config

```typescript
interface AnthropicAIConfig {
  service: "anthropic" | "bedrock";
}
```

## License

ISC
