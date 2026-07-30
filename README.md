# 190proof

An opinionated unified interface for interacting with multiple AI providers including **OpenAI**, **Anthropic**, **Google**, **Groq**, **OpenRouter**, and **AWS Bedrock**. This package provides a consistent API for making requests to different LLM providers while handling retries, streaming, and multimodal inputs.

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

Optional per-request knobs live on `payload` (`GenericPayload`):

- `payload.requestTimeoutMs`: `number` - Per-attempt HTTP timeout in ms (default: 120000), honored by every adapter — except streaming OpenRouter attempts, which it deliberately does NOT bound (see below). For OpenRouter's non-streaming transport the default is 180000.
- `payload.streaming`: `boolean` - OpenRouter-only (default: true). Streams the completion over SSE. A streaming attempt is bounded by two independent timers instead of `requestTimeoutMs`: `streamTimeoutMs` (total wall clock, default 600000) and the per-useful-chunk stall timeout (`chunkTimeoutMs` argument, default 15000). A chunk is "useful" only if it advances content, reasoning, tool-call fragments, finish_reason, or usage — SSE comment keep-alives (`: OPENROUTER PROCESSING`) and role-only deltas don't reset the stall timer, so a hung provider dies within one stall window while a healthy long generation can run to the total budget. Set `streaming: false` for the old single-JSON-body transport.
- `payload.streamTimeoutMs`: `number` - OpenRouter-only: total wall-clock budget per streaming attempt (default: 600000).
- `payload.streamDeadlineAt`: `number` - OpenRouter-only: absolute deadline (epoch ms) for the whole call **including retries** — the caller's turn budget. Each attempt gets `min(streamTimeoutMs, deadline - now)`, and once under 10s remain the call fails fast instead of starting a generation that cannot be delivered. Use it whenever the caller has its own timeout: a per-attempt budget alone is re-granted on every retry and can outlive that timeout.
- `payload.thinkingConfig`: `Record<string, unknown>` - Google-only: forwarded verbatim as `generationConfig.thinkingConfig` on the Gemini request — e.g. `{ thinkingBudget: 0 }` to disable thinking, `{ thinkingLevel: "HIGH" }` on models that take a level. Ignored by all other adapters; shapes are model-specific and validated by Google, not the SDK.

When a streaming attempt is cut at its **total deadline** and prose has already arrived, the partial answer is returned with `truncated: true` on the response rather than discarded — those tokens were generated and billed, so throwing them away costs money and gives the user nothing. Surface such a reply as incomplete. Salvage never applies to tool-call turns (half-streamed arguments are unparseable JSON), to stalls (the provider died mid-thought), or to caller aborts. When nothing is salvageable, the discard is logged with an approximate token count — aborted attempts never receive OpenRouter's `usage` chunk, so that log line is the only record of the wasted spend.

OpenRouter retries also perform **moderation eviction**: a provider content-moderation rejection (e.g. "Upstream error from Alibaba: Output data may contain inappropriate content.") is deterministic for a given payload, so on the first one the refusing provider is removed from the request's provider preferences (`ignore` += slug, `order` -= slug) and every remaining attempt reroutes to the next provider. Non-moderation errors retry with unchanged preferences, and `fallbackModel` still applies if the whole pool refuses.
- `payload.signal`: `AbortSignal` - Caller-supplied cancellation. When it aborts, the in-flight provider request is cancelled and `callWithRetries` **rejects immediately — it does not retry or fall back** (both the retry loop and the fallback branch bail on `signal.aborted`). Threaded to the underlying fetch/axios/SDK call of each provider.

#### Returns

`Promise<ParsedResponseMessage>`:

```typescript
interface ParsedResponseMessage {
  role: "assistant";
  content: string | null;
  function_call: FunctionCall | null;
  function_calls: FunctionCall[];
  files: File[]; // For models that return files (e.g., image generation)
  // Who actually served the response: OpenRouter's upstream provider from the
  // response body (e.g. "Baidu"), or the SDK provider name ("anthropic", ...)
  // for direct providers. On fallback, reflects the model that answered.
  provider?: string;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    // Reasoning/thinking tokens spent before the visible answer; currently
    // populated from Google's usageMetadata.thoughtsTokenCount.
    thoughts_tokens?: number;
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
  /**
   * Optional base URL. Defaults to `https://api.openai.com/v1`. Set to point
   * at any OpenAI-compatible endpoint (e.g. a self-hosted proxy). The path
   * `/chat/completions` is appended automatically. Ignored for Azure.
   */
  baseUrl?: string;
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

To talk to an OpenAI-compatible server instead of OpenAI itself:

```typescript
await callWithRetries(
  "my-identifier",
  {
    model: "openai:gpt-4o-mini",
    messages: [{ role: "user", content: "hi" }],
  },
  {
    service: "openai",
    apiKey: process.env.SOME_SERVER_API_KEY,
    baseUrl: "https://your-proxy.example.com/v1",
  },
);
```

#### Anthropic Config

```typescript
interface AnthropicAIConfig {
  service: "anthropic" | "bedrock";
}
```

## License

ISC
