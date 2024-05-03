import { callWithRetries } from "./index";
import { GPTModel, GroqModel, ClaudeModel, GenericPayload } from "./interfaces";

//  interface ParsedResponseMessage {
//   role: "assistant";
//   content: string | null;
//   function_call: FunctionCall | null;
// }

jest.setTimeout(60000); // Increase timeout to 60s

describe("Groq Model", () => {
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

    const answer1 = await callWithRetries("test1", aiPayload1);
    expect(answer1).toBeDefined();
    expect(answer1.function_call).toBeDefined();
    expect(answer1.function_call?.name).toBeDefined();
    expect(answer1.function_call?.arguments).toBeDefined();
    expect(answer1.function_call?.arguments?.team_name).toBeDefined();
  });

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

    const answer4 = await callWithRetries("test4", aiPayload4);
    expect(answer4).toBeDefined();
    expect(answer4.content).toBeDefined();
  });
});

describe("OpenAI Model", () => {
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

    const answer2 = await callWithRetries("test2", aiPayload2);
    expect(answer2).toBeDefined();
    expect(answer2.function_call).toBeDefined();
    expect(answer2.function_call?.name).toBeDefined();
    expect(answer2.function_call?.arguments).toBeDefined();
    expect(answer2.function_call?.arguments?.country_name).toBeDefined();
  });

  test("standard", async () => {
    const aiPayload5: GenericPayload = {
      model: GPTModel.GPT4_0409,
      messages: [
        {
          role: "user",
          content: "Tell me a story.",
        },
      ],
    };

    const answer5 = await callWithRetries("test5", aiPayload5);
    expect(answer5).toBeDefined();
    expect(answer5.content).toBeDefined();
  });
});

describe("Anthropic Model", () => {
  test("with functions", async () => {
    const aiPayload3: GenericPayload = {
      model: ClaudeModel.SONNET,
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

    const answer3 = await callWithRetries("test3", aiPayload3);
    expect(answer3).toBeDefined();
    expect(answer3.function_call).toBeDefined();
    expect(answer3.function_call?.name).toBeDefined();
    expect(answer3.function_call?.arguments).toBeDefined();
    expect(answer3.function_call?.arguments?.topic).toBeDefined();
  });

  test("standard", async () => {
    const aiPayload6: GenericPayload = {
      model: ClaudeModel.SONNET,
      messages: [
        {
          role: "user",
          content: "Tell me about the universe.",
        },
      ],
    };

    const answer6 = await callWithRetries("test6", aiPayload6);
    expect(answer6).toBeDefined();
    expect(answer6.content).toBeDefined();
  });
});
