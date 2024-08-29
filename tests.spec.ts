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
  { provider: "Groq", model: GroqModel.LLAMA_3_70B_8192 },
  { provider: "OpenAI", model: GPTModel.GPT4O },
  { provider: "Anthropic", model: ClaudeModel.SONNET_3_5 },
  { provider: "Gemini", model: GeminiModel.GEMINI_15_PRO },
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

    const answer = await callWithRetries(`${provider}_standard`, aiPayload);
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
  });

  test("with functions", async () => {
    const aiPayload: GenericPayload = {
      model,
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

    const answer = await callWithRetries(`${provider}_functions`, aiPayload);
    expect(answer).toBeDefined();
    expect(answer.function_call).toBeDefined();
    expect(answer.function_call?.name).toBeDefined();
    expect(answer.function_call?.arguments).toBeDefined();
    expect(answer.function_call?.arguments?.country_name).toBeDefined();
  });

  test("with image in message", async () => {
    const aiPayload: GenericPayload = {
      model,
      messages: [
        {
          role: "user",
          content: "Where is this?",
          files: [
            {
              mimeType: "image/jpeg",
              url: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Colosseo_2020.jpg/540px-Colosseo_2020.jpg",
            },
          ],
        },
      ],
    };

    const answer = await callWithRetries(`${provider}_files`, aiPayload);
    expect(answer.content).toContain("Italy");
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

    const answer = await callWithRetries(`${provider}_system`, aiPayload);
    expect(answer.content).toBeDefined();
    expect(answer.content).toContain("HAHAHAHA");
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
          content: "and red?",
        },
      ],
    };

    const answer = await callWithRetries(`${provider}_context`, aiPayload);
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
    expect(answer.content?.toLowerCase()).toContain("orange");
  });
});
