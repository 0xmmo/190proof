import { callWithRetries } from "./index";
import {
  GPTModel,
  GroqModel,
  ClaudeModel,
  GenericPayload,
  GeminiModel,
} from "./interfaces";

jest.setTimeout(60000); // Increase timeout to 60s

const modelConfigs = [
  { provider: "Groq", model: GroqModel.QWEN3_32B },
  { provider: "OpenAI", model: GPTModel.GPT5_MINI },
  { provider: "Anthropic", model: ClaudeModel.HAIKU_4_5 },
  {
    provider: "Gemini",
    model: GeminiModel.GEMINI_3_1_FLASH_LITE_PREVIEW,
  },
];

describe.each(modelConfigs)("$provider Model", ({ provider, model }) => {
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

    const answer = await callWithRetries([provider, "standard"], aiPayload);
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

  test("with functions", async () => {
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

    const answer = await callWithRetries([provider, "functions"], aiPayload);
    expect(answer).toBeDefined();
    expect(answer.function_call).toBeDefined();
    expect(answer.function_call?.name).toEqual("get_weather");
    expect(answer.function_call?.arguments).toBeDefined();
    expect(answer.function_call?.arguments?.city_name).toBeDefined();
    expect(answer.function_calls.length).toBeGreaterThanOrEqual(1);
    expect(answer.function_calls[0].name).toEqual("get_weather");
  });

  test("with parallel function calls", async () => {
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

    const answer = await callWithRetries([provider, "files"], aiPayload);
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

    const answer = await callWithRetries([provider, "system"], aiPayload);
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

    const answer = await callWithRetries([provider, "history"], aiPayload);
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

    const answer = await callWithRetries([provider, "context"], aiPayload);
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
    expect(answer.content?.toLowerCase()).toContain("green");
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
    );

    expect(answer).toBeDefined();
    expect(answer.files?.length).toBeGreaterThan(0);
    expect(answer.files?.[0].mimeType).toBe("image/png");
  });
});
