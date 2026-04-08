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
- 🔌 Provider prefix strings for any model without waiting for package updates

## Installation

```bash
npm install 190proof
```

## Usage

### Basic Example

Use any model from any provider with the `provider:model-id` format:

```typescript
import { callWithRetries, GenericPayload } from "190proof";

const payload: GenericPayload = {
  model: "openai:gpt-4o-mini",
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
import { callWithRetries, GenericPayload } from "190proof";

// OpenAI
const openaiPayload: GenericPayload = {
  model: "openai:gpt-5",
  messages: [{ role: "user", content: "Hello!" }],
};

// Anthropic
const claudePayload: GenericPayload = {
  model: "anthropic:claude-sonnet-4-5",
  messages: [{ role: "user", content: "Hello!" }],
};

// Google
const geminiPayload: GenericPayload = {
  model: "google:gemini-2.0-flash",
  messages: [{ role: "user", content: "Hello!" }],
};

// Groq
const groqPayload: GenericPayload = {
  model: "groq:llama-3.3-70b-versatile",
  messages: [{ role: "user", content: "Hello!" }],
};

// OpenRouter
const openRouterPayload: GenericPayload = {
  model: "openrouter:google/gemma-4-31b-it:free",
  messages: [{ role: "user", content: "Hello!" }],
};

const response = await callWithRetries("request-id", claudePayload);
```

### With Function Calling

```typescript
const payload: GenericPayload = {
  model: "openai:gpt-4o",
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
  model: "anthropic:claude-sonnet-4-5",
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
  model: "google:gemini-2.0-flash",
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

### Inspecting Model Routing

Use `parseModelString` to see how a model string will be routed:

```typescript
import { parseModelString } from "190proof";

parseModelString("openai:gpt-7");
// → { provider: "openai", modelId: "gpt-7" }

parseModelString("openrouter:org/model-name:free");
// → { provider: "openrouter", modelId: "org/model-name:free" }

```

## Provider Prefix Format

The model string format is `provider:model-id`, where the provider prefix is one of:

| Prefix | Provider |
|---|---|
| `openai` | OpenAI |
| `anthropic` | Anthropic |
| `google` | Google (Gemini) |
| `groq` | Groq |
| `openrouter` | OpenRouter |

The prefix is stripped before sending to the API, so the model ID should be exactly what the provider expects (e.g. `"openai:gpt-4o"` sends `"gpt-4o"` to OpenAI).

## Supported Models

These models are tested. You can use any model with the `provider:model-id` format.

### OpenAI

- `openai:gpt-5`
- `openai:gpt-5-mini`
- `openai:gpt-4.1`
- `openai:gpt-4.1-mini`
- `openai:gpt-4.1-nano`
- `openai:gpt-4o`
- `openai:gpt-4o-mini`
- `openai:o3-mini`
- `openai:o1-preview`
- `openai:o1-mini`

### Anthropic

- `anthropic:claude-opus-4-5`
- `anthropic:claude-sonnet-4-5`
- `anthropic:claude-haiku-4-5`
- `anthropic:claude-opus-4-1`
- `anthropic:claude-opus-4-20250514`
- `anthropic:claude-sonnet-4-20250514`
- `anthropic:claude-3-5-sonnet-20241022`
- `anthropic:claude-3-5-haiku-20241022`

### Google

- `google:gemini-3.1-flash-lite-preview`
- `google:gemini-3-flash-preview`
- `google:gemini-2.5-flash-preview-04-17`
- `google:gemini-2.0-flash`
- `google:gemini-2.0-flash-exp-image-generation`
- `google:gemini-1.5-pro-latest`

### Groq

- `groq:llama-3.3-70b-versatile`
- `groq:llama3-70b-8192`
- `groq:qwen/qwen3-32b`
- `groq:deepseek-r1-distill-llama-70b`

### OpenRouter

- `openrouter:google/gemma-4-31b-it:free`
- `openrouter:google/gemma-4-31b-it`

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

### `parseModelString(model)`

Parses a model string into its provider and model ID components.

#### Parameters

- `model`: `string` - A model string in `"provider:model-id"` format

#### Returns

`{ provider: Provider, modelId: string }`

### Configuration Options

#### OpenAI Config

```typescript
interface OpenAIConfig {
  service: "azure" | "openai";
  apiKey: string;
  baseUrl: string;
  orgId?: string;
  modelConfigMap?: Record<
    string,
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
