import { callWithRetries } from "./index";
import { AnthropicAIConfig, GenericPayload, OpenAIConfig } from "./interfaces";

jest.setTimeout(60000); // Increase timeout to 60s

type ModelConfig = {
  provider: string;
  model: string;
  /**
   * Optional service config to pass through to callWithRetries. Use for
   * OpenAI-compatible servers that need a custom apiKey/baseUrl.
   */
  config?: OpenAIConfig | AnthropicAIConfig;
  /** Backend doesn't proxy OpenAI-style function/tool calls. */
  skipFunctions?: boolean;
};

const modelConfigs: ModelConfig[] = [
  { provider: "Groq", model: "groq:qwen/qwen3-32b" },
  { provider: "OpenAI", model: "openai:gpt-5-mini" },
  { provider: "Anthropic", model: "anthropic:claude-haiku-4-5" },
  { provider: "Gemini", model: "google:gemini-3.1-flash-lite-preview" },
  { provider: "OpenRouter", model: "openrouter:google/gemma-4-31b-it" },
  // codex-server (OpenAI-compatible facade over OpenAI's codex CLI).
  // Skipped automatically if CODEX_SERVER_API_KEY isn't set.
  ...(process.env.CODEX_SERVER_API_KEY
    ? [
        {
          provider: "codex-server (openai baseUrl)",
          model: "openai:codex",
          config: {
            service: "openai" as const,
            apiKey: process.env.CODEX_SERVER_API_KEY,
            baseUrl:
              process.env.CODEX_SERVER_BASE_URL ??
              "https://codex-server-mo.fly.dev/v1",
          },
          // codex-server doesn't proxy client-defined tools (the agent has
          // its own built-in file/shell tools that fire server-side).
          skipFunctions: true,
        },
      ]
    : []),
];

describe.each(modelConfigs)(
  "$provider Model",
  ({ provider, model, config, skipFunctions }) => {
    const itIfFunctions = skipFunctions ? test.skip : test;
  test("standard query", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "user",
          content: "Tell me a joke.",
        },
      ],
    };

    const answer = await callWithRetries(
      [provider, "standard"],
      aiPayload,
      config,
    );
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
    // Usage should be populated for non-streaming calls
    expect(answer.usage).toBeDefined();
    if (answer.usage) {
      expect(answer.usage.prompt_tokens).toBeGreaterThan(5);
      expect(answer.usage.completion_tokens).toBeGreaterThan(5);
      expect(answer.usage.total_tokens).toBeGreaterThan(10);
    }
  });

  itIfFunctions("with functions", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "user",
          content: "What is the weather in Tokyo?",
        },
      ],
      functions: [
        {
          name: "get_weather",
          description: "Get the weather of a given city",
          parameters: {
            type: "object",
            properties: {
              city_name: {
                type: "string",
                description: "The name of the city",
              },
            },
            required: ["city_name"],
          },
        },
      ],
    };

    const answer = await callWithRetries(
      [provider, "functions"],
      aiPayload,
      config,
    );
    expect(answer).toBeDefined();
    expect(answer.function_call).toBeDefined();
    expect(answer.function_call?.name).toEqual("get_weather");
    expect(answer.function_call?.arguments).toBeDefined();
    expect(answer.function_call?.arguments?.city_name).toBeDefined();
    expect(answer.function_calls.length).toBeGreaterThanOrEqual(1);
    expect(answer.function_calls[0].name).toEqual("get_weather");
  });

  itIfFunctions("with parallel function calls", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "system",
          content:
            "You MUST call the get_weather function once for each city mentioned. Always make multiple parallel function calls when asked about multiple cities. Never combine cities into a single call.",
        },
        {
          role: "user",
          content: "What is the weather in Tokyo and New York?",
        },
      ],
      functions: [
        {
          name: "get_weather",
          description: "Get the weather of a given city",
          parameters: {
            type: "object",
            properties: {
              city_name: {
                type: "string",
                description: "The name of the city",
              },
            },
            required: ["city_name"],
          },
        },
      ],
    };

    const answer = await callWithRetries(
      [provider, "parallel_functions"],
      aiPayload,
      config,
    );
    expect(answer).toBeDefined();
    expect(answer.function_calls.length).toBeGreaterThanOrEqual(2);
    const cityNames = answer.function_calls.map((fc) =>
      fc.arguments.city_name?.toLowerCase(),
    );
    expect(cityNames).toContain("tokyo");
    expect(cityNames).toContain("new york");
    // function_call should still be the first one for backward compat
    expect(answer.function_call).toEqual(answer.function_calls[0]);
  });

  test("with image in message", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "user",
          content: "What kind of plant is this?",
          files: [
            {
              mimeType: "image/jpeg",
              url: "https://olly.bot/1.png",
            },
          ],
        },
      ],
    };

    const answer = await callWithRetries(
      [provider, "files"],
      aiPayload,
      config,
    );
    expect(answer.content?.toLowerCase()).toContain("fiddle");
  });

  test("with system message", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "system",
          content: `Whenever the user asks for a joke always end it with "HAHAHAHA"`,
        },
        {
          role: "user",
          content: "Tell me a joke.",
        },
      ],
    };

    const answer = await callWithRetries(
      [provider, "system"],
      aiPayload,
      config,
    );
    expect(answer.content).toBeDefined();
    expect(answer.content).toContain("HAHAHAHA");
  });

  test("history starts with model message", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "assistant",
          content: "You are a helpful assistant",
        },
        {
          role: "user",
          content: "mix yellow with red?",
        },
        {
          role: "assistant",
          content: "orange",
        },
        {
          role: "user",
          content: "What about blue and yellow?",
        },
      ],
    };

    const answer = await callWithRetries(
      [provider, "history"],
      aiPayload,
      config,
    );
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
  });

  test("assistant message with image files in history", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "user",
          content: "Generate an image of a sunset",
        },
        {
          role: "assistant",
          content: "Here is a sunset image",
          files: [
            {
              mimeType: "image/png",
              data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            },
          ],
        },
        {
          role: "user",
          content: "Describe what you see in the image you generated",
        },
      ],
    };

    const answer = await callWithRetries(
      [provider, "assistant_with_files"],
      aiPayload,
      config,
    );
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
  });

  test("consecutive user messages", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "user",
          content: "What color do you get if you mix yellow",
        },
        {
          role: "user",
          content: "with red?",
        },
        {
          role: "assistant",
          content: "orange",
        },
        {
          role: "user",
          content: "What about blue and yellow?",
        },
      ],
    };

    const answer = await callWithRetries(
      [provider, "context"],
      aiPayload,
      config,
    );
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
    expect(answer.content?.toLowerCase()).toContain("green");
  });

  itIfFunctions("multi-turn tool round-trip", async () => {
    const functions = [
      {
        name: "get_weather",
        description: "Get the weather of a given city",
        parameters: {
          type: "object",
          properties: {
            city_name: { type: "string", description: "The name of the city" },
          },
          required: ["city_name"],
        },
      },
    ];
    const userMsg = {
      role: "user" as const,
      content: "What is the weather in Tokyo? Use the get_weather tool.",
    };

    // turn 1 — model calls the tool
    const first = await callWithRetries(
      [provider, "roundtrip_1"],
      { model, messages: [userMsg], functions },
      config,
    );
    expect(first.function_call?.name).toEqual("get_weather");
    // every provider surfaces an id (synthesized for those that don't return one)
    expect(first.function_call?.id).toBeTruthy();

    // turn 2 — feed the assistant call + the tool result back
    const second = await callWithRetries(
      [provider, "roundtrip_2"],
      {
        model,
        messages: [
          userMsg,
          {
            role: "assistant",
            content: first.content ?? "",
            functionCalls: first.function_calls,
            reasoning: first.reasoning,
            reasoningDetails: first.reasoningDetails,
          },
          {
            role: "tool",
            content: "",
            toolResults: first.function_calls.map((fc) => ({
              toolCallId: fc.id as string,
              name: fc.name,
              content: '{"tempC":22,"condition":"sunny"}',
            })),
          },
        ],
        functions,
      },
      config,
    );

    // the round-trip must not error and the model should answer in text using
    // the result it was handed
    expect(second.content).toBeTruthy();
    expect(second.content?.toLowerCase()).toContain("22");
  });

  test.skip("generates images", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "user",
          content: "Generate an image of a sunset over mountains",
        },
      ],
    };

    const answer = await callWithRetries(
      [provider, "image_generation"],
      aiPayload,
      config,
    );

    expect(answer).toBeDefined();
    expect(answer.files?.length).toBeGreaterThan(0);
    expect(answer.files?.[0].mimeType).toBe("image/png");
  });
});

// DeepSeek V4 via OpenRouter is igpt's default model and the motivating case:
// thinking models 400 on the next turn unless the prior reasoning is passed
// back. This exercises the reasoning round-trip end-to-end. Skipped unless an
// OpenRouter key is present so the keyless mocked suite (tool-calls.spec.ts)
// still runs in CI.
const itIfOpenRouter = process.env.OPENROUTER_API_KEY ? test : test.skip;
itIfOpenRouter(
  "OpenRouter DeepSeek-V4: tool round-trip preserves reasoning",
  async () => {
    const model = "openrouter:deepseek/deepseek-v4-flash";
    const functions = [
      {
        name: "get_weather",
        description: "Get the weather of a given city",
        parameters: {
          type: "object",
          properties: {
            city_name: { type: "string", description: "The name of the city" },
          },
          required: ["city_name"],
        },
      },
    ];
    const userMsg = {
      role: "user" as const,
      content: "What is the weather in Tokyo? Use the get_weather tool.",
    };

    const first = await callWithRetries(
      ["DeepSeek", "rt1"],
      { model, messages: [userMsg], functions },
    );

    // DeepSeek may answer directly; only round-trip when it actually called.
    if (!first.function_call) {
      expect(first.content).toBeTruthy();
      return;
    }

    const second = await callWithRetries(["DeepSeek", "rt2"], {
      model,
      messages: [
        userMsg,
        {
          role: "assistant",
          content: first.content ?? "",
          functionCalls: first.function_calls,
          reasoning: first.reasoning,
          reasoningDetails: first.reasoningDetails,
        },
        {
          role: "tool",
          content: "",
          toolResults: first.function_calls.map((fc) => ({
            toolCallId: fc.id as string,
            name: fc.name,
            content: '{"tempC":22,"condition":"sunny"}',
          })),
        },
      ],
      functions,
    });

    // The key assertion: the round-trip does not 400 and a usable answer returns.
    expect(second.content).toBeTruthy();
  },
);
