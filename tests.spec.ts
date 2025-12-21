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
  // { provider: "Groq", model: GroqModel.DEEPSEEK_R1_DISTILL_LLAMA_70B },
  // { provider: "OpenAI", model: GPTModel.GPT5_MINI },
  // { provider: "Anthropic", model: ClaudeModel.OPUS_4_5 },
  {
    provider: "Gemini",
    model: GeminiModel.GEMINI_3_FLASH_PREVIEW,
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
    expect(answer.content).toContain("Fiddle");
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

  test("generates images", async () => {
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
      aiPayload
    );

    expect(answer).toBeDefined();
    expect(answer.files?.length).toBeGreaterThan(0);
    expect(answer.files?.[0].mimeType).toBe("image/png");
  });
});
