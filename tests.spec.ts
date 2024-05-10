import { callWithRetries } from "./index";
import {
  GPTModel,
  GroqModel,
  ClaudeModel,
  GenericPayload,
  GeminiModel,
} from "./interfaces";

jest.setTimeout(60000); // Increase timeout to 60s

describe("Groq Model", () => {
  test("standard", async () => {
    const aiPayload4: GenericPayload = {
      model: GroqModel.LLAMA_3_70B_8192,
      messages: [
        {
          role: "user",
          content: "Tell me a joke.",
        },
      ],
    };

    const answer = await callWithRetries("test4", aiPayload4);
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
  });

  test("with functions", async () => {
    const aiPayload1: GenericPayload = {
      model: GroqModel.LLAMA_3_70B_8192,
      messages: [
        {
          role: "user",
          content: "How are the lakers doing?",
        },
      ],
      functions: [
        {
          name: "get_game_score",
          description: "Get the score for a given NBA game",
          parameters: {
            type: "object",
            properties: {
              team_name: {
                type: "string",
                description:
                  "The name of the NBA team (e.g. 'Golden State Warriors')",
              },
            },
            required: ["team_name"],
          },
        },
      ],
    };

    const answer = await callWithRetries("test1", aiPayload1);
    expect(answer).toBeDefined();
    expect(answer.function_call).toBeDefined();
    expect(answer.function_call?.name).toBeDefined();
    expect(answer.function_call?.arguments).toBeDefined();
    expect(answer.function_call?.arguments?.team_name).toBeDefined();
  });
});

describe("OpenAI Model", () => {
  test("standard", async () => {
    const aiPayload5: GenericPayload = {
      model: GPTModel.GPT4_0409,
      messages: [
        {
          role: "user",
          content: "Tell me a joke.",
        },
      ],
    };

    const answer = await callWithRetries("test5", aiPayload5);
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
  });

  test("with functions", async () => {
    const aiPayload2: GenericPayload = {
      model: GPTModel.GPT4_0409,
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

    const answer = await callWithRetries("test2", aiPayload2);
    expect(answer).toBeDefined();
    expect(answer.function_call).toBeDefined();
    expect(answer.function_call?.name).toBeDefined();
    expect(answer.function_call?.arguments).toBeDefined();
    expect(answer.function_call?.arguments?.country_name).toBeDefined();
  });

  test("with files in message", async () => {
    const aiPayload8: GenericPayload = {
      model: GPTModel.GPT4_0409,
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

    const answer = await callWithRetries("test8", aiPayload8);
    expect(answer.content).toContain("Italy");
  });
});

describe("Anthropic Model", () => {
  test("standard", async () => {
    const aiPayload6: GenericPayload = {
      model: ClaudeModel.HAIKU,
      messages: [
        {
          role: "user",
          content: "Tell me about the universe.",
        },
      ],
    };

    const answer = await callWithRetries("test6", aiPayload6);
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
  });

  test("with functions", async () => {
    const aiPayload3: GenericPayload = {
      model: ClaudeModel.HAIKU,
      messages: [
        {
          role: "user",
          content: "Write a sonnet about the beauty of nature.",
        },
      ],
      functions: [
        {
          name: "generate_sonnet",
          description: "Generate a sonnet about a given topic",
          parameters: {
            type: "object",
            properties: {
              topic: {
                type: "string",
                description: "The topic of the sonnet",
              },
            },
            required: ["topic"],
          },
        },
      ],
    };

    const answer = await callWithRetries("test3", aiPayload3);
    expect(answer).toBeDefined();
    expect(answer.function_call).toBeDefined();
    expect(answer.function_call?.name).toBeDefined();
    expect(answer.function_call?.arguments).toBeDefined();
    expect(answer.function_call?.arguments?.topic).toBeDefined();
  });

  test("with files in message", async () => {
    const aiPayload7: GenericPayload = {
      model: ClaudeModel.HAIKU,
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

    const answer = await callWithRetries("test7", aiPayload7);
    expect(answer.content).toContain("Italy");
  });
});

describe("Gemini Model", () => {
  test("standard query", async () => {
    const aiPayload9: GenericPayload = {
      model: GeminiModel.GEMINI_15_PRO,
      messages: [
        {
          role: "user",
          content: "Explain quantum computing.",
        },
      ],
    };

    const answer = await callWithRetries("test9", aiPayload9);
    expect(answer).toBeDefined();
    expect(answer.content).toBeDefined();
  });

  test("with functions", async () => {
    const aiPayload10: GenericPayload = {
      model: GeminiModel.GEMINI_15_PRO,
      messages: [
        {
          role: "user",
          content:
            "Calculate the trajectory for a Mars mission launching on 2023-10-01.",
        },
      ],
      functions: [
        {
          name: "calculate_trajectory",
          description: "Calculate the trajectory based on provided parameters",
          parameters: {
            type: "object",
            properties: {
              destination: {
                type: "string",
                description: "The destination of the mission",
              },
              launchDate: {
                type: "string",
                description: "The launch date in ISO format",
              },
            },
            required: ["destination", "launchDate"],
          },
        },
      ],
    };

    const answer = await callWithRetries("test10", aiPayload10);
    expect(answer).toBeDefined();
    expect(answer.function_call).toBeDefined();
    expect(answer.function_call?.name).toBeDefined();
    expect(answer.function_call?.arguments).toBeDefined();
    expect(answer.function_call?.arguments?.destination).toBeDefined();
    expect(answer.function_call?.arguments?.launchDate).toBeDefined();
  });

  test("with files in message", async () => {
    const aiPayload11: GenericPayload = {
      model: GeminiModel.GEMINI_15_PRO,
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

    const answer = await callWithRetries("test11", aiPayload11);
    expect(answer.content).toContain("Italy");
  });
});
