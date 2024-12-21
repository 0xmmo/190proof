# 190proof

A unified interface for interacting with multiple AI providers including **OpenAI**, **Anthropic**, **Google**, and **Groq**. This package provides a consistent API for making requests to different LLM providers while handling retries, streaming, and multimodal inputs.

## Features

Fully-local unified interface across multiple AI providers that includes:

- 🖼️ Image format & size normalization
- 🛠️ Consistent function calling
- 💬 Consistent message alternation & system instructions
- 🔄 Automatic retries with configurable attempts
- 📡 Streaming by default

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
  model: GPTModel.O1_MINI,
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

### With Function Calling

```typescript
const payload: GenericPayload = {
  model: GPTModel.O1_MINI,
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
```

### With Images

```typescript
const payload: GenericPayload = {
  model: GPTModel.O1_MINI,
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
  model: GPTModel.O1_MINI,
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

### Anthropic Models

- `claude-3-haiku-20240307`
- `claude-3-sonnet-20240229`
- `claude-3-opus-20240229`
- `claude-3-5-sonnet-20241022`

### Google Models

- `gemini-1.5-pro-latest`

### Groq Models

- `llama3-70b-8192`

## API Reference

### `callWithRetries(identifier: string, payload: GenericPayload, config?: Config, retries?: number, chunkTimeoutMs?: number)`

Main function to make requests to any supported AI provider.

#### Parameters

- `identifier`: Unique identifier for the request
- `payload`: Request payload containing model, messages, and optional functions
- `config`: Optional configuration for the specific provider
- `retries`: Number of retry attempts (default: 5)
- `chunkTimeoutMs`: Timeout for streaming chunks (default: 15000)

## License

ISC
