/**
 * Outbound serialization of native multi-turn tool calls, per provider.
 *
 * Pure unit tests — no network/keys/cost. The transports are mocked (axios for
 * Anthropic/Groq/OpenRouter, `fetch` for OpenAI, the `@google/genai` client for
 * Google) and we assert on the exact request the SDK builds when a caller feeds
 * back a prior assistant tool-call turn plus its results. This is the contract
 * consumers rely on; live round-trip behaviour is covered in tests.spec.ts.
 */
import axios from "axios";
import { callWithRetries } from "./index";
import { GenericMessage, GenericPayload, FunctionDefinition } from "./interfaces";

jest.mock("axios");

const mockedPost = axios.post as unknown as jest.Mock;

const FUNCTIONS: FunctionDefinition[] = [
  {
    name: "get_weather",
    description: "Get the weather of a given city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
];

/** A 3-message history: user → assistant(tool call) → tool(result). */
function toolMessages(
  reasoning: Partial<Pick<GenericMessage, "reasoning" | "reasoningDetails">> = {},
): GenericMessage[] {
  return [
    { role: "user", content: "What's the weather in Tokyo?" },
    {
      role: "assistant",
      content: "",
      functionCalls: [
        { id: "call_abc", name: "get_weather", arguments: { city: "Tokyo" } },
      ],
      ...reasoning,
    },
    {
      role: "tool",
      content: "",
      toolResults: [
        { toolCallId: "call_abc", name: "get_weather", content: '{"tempC":22}' },
      ],
    },
  ];
}

/** Parallel variant: assistant makes two calls, tool answers both. */
function parallelToolMessages(): GenericMessage[] {
  return [
    { role: "user", content: "Weather in Tokyo and Paris?" },
    {
      role: "assistant",
      content: "",
      functionCalls: [
        { id: "call_1", name: "get_weather", arguments: { city: "Tokyo" } },
        { id: "call_2", name: "get_weather", arguments: { city: "Paris" } },
      ],
    },
    {
      role: "tool",
      content: "",
      toolResults: [
        { toolCallId: "call_1", name: "get_weather", content: '{"tempC":22}' },
        { toolCallId: "call_2", name: "get_weather", content: '{"tempC":15}' },
      ],
    },
  ];
}

// ─── axios providers (Anthropic / Groq / OpenRouter) ─────────────────────────

const OAI_OK = {
  data: {
    choices: [
      { message: { role: "assistant", content: "It's 22°C in Tokyo." } },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  },
};
const ANTHROPIC_OK = {
  data: {
    content: [{ type: "text", text: "It's 22°C in Tokyo." }],
    usage: { input_tokens: 1, output_tokens: 1 },
  },
};
const GEMINI_OK = {
  data: {
    candidates: [{ content: { parts: [{ text: "It's 22°C in Tokyo." }] } }],
    usageMetadata: {
      promptTokenCount: 1,
      candidatesTokenCount: 1,
      totalTokenCount: 2,
    },
  },
};

beforeEach(() => {
  mockedPost.mockReset();
  mockedPost.mockImplementation((url: string) => {
    if (url.includes("anthropic.com")) return Promise.resolve(ANTHROPIC_OK);
    if (url.includes("generativelanguage.googleapis.com"))
      return Promise.resolve(GEMINI_OK);
    return Promise.resolve(OAI_OK); // groq.com + openrouter.ai
  });
});

describe("OpenAI-compatible serialization (OpenRouter / Groq)", () => {
  test.each([
    ["OpenRouter", "openrouter:deepseek/deepseek-v4-flash"],
    ["Groq", "groq:qwen/qwen3-32b"],
  ])("%s: assistant tool_calls + tool message", async (_name, model) => {
    const payload: GenericPayload = {
      model,
      streaming: false, // axios is what is mocked here — pin the non-streaming transport
      messages: toolMessages({
        reasoning: "Let me check the weather.",
        reasoningDetails: [{ type: "reasoning.text", text: "thinking" }],
      }),
      functions: FUNCTIONS,
    };

    await callWithRetries(["test", "oai-compat"], payload);
    const body = mockedPost.mock.calls[0][1];

    // assistant turn carries native tool_calls; content collapses to null
    expect(body.messages[1]).toMatchObject({
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call_abc",
          type: "function",
          function: {
            name: "get_weather",
            arguments: JSON.stringify({ city: "Tokyo" }),
          },
        },
      ],
      reasoning: "Let me check the weather.",
      reasoning_details: [{ type: "reasoning.text", text: "thinking" }],
    });
    // tool result becomes a dedicated tool message
    expect(body.messages[2]).toEqual({
      role: "tool",
      tool_call_id: "call_abc",
      content: '{"tempC":22}',
    });
  });

  test("OpenRouter: parallel calls → one tool message per result", async () => {
    await callWithRetries(["test", "parallel"], {
      model: "openrouter:deepseek/deepseek-v4-flash",
  streaming: false, // axios is what is mocked here — pin the non-streaming transport
      messages: parallelToolMessages(),
      functions: FUNCTIONS,
    });
    const body = mockedPost.mock.calls[0][1];

    expect(body.messages[1].tool_calls.map((t: any) => t.id)).toEqual([
      "call_1",
      "call_2",
    ]);
    expect(body.messages.slice(2)).toEqual([
      { role: "tool", tool_call_id: "call_1", content: '{"tempC":22}' },
      { role: "tool", tool_call_id: "call_2", content: '{"tempC":15}' },
    ]);
  });

  test("foreign reasoning blocks are dropped (Anthropic history → OpenRouter)", async () => {
    // Reverse of the Anthropic case: native thinking blocks captured from
    // Anthropic must not ride reasoning_details to an OpenAI-compat provider.
    await callWithRetries(["test", "oai-compat-foreign-reasoning"], {
      model: "openrouter:deepseek/deepseek-v4-flash",
  streaming: false, // axios is what is mocked here — pin the non-streaming transport
      messages: toolMessages({
        reasoningDetails: [
          { type: "thinking", thinking: "native", signature: "sig123" },
        ],
      }),
      functions: FUNCTIONS,
    });
    const body = mockedPost.mock.calls[0][1];
    expect(body.messages[1]).not.toHaveProperty("reasoning_details");
  });

  test("mixed reasoningDetails keeps only reasoning.* blocks", async () => {
    await callWithRetries(["test", "oai-compat-mixed-reasoning"], {
      model: "openrouter:deepseek/deepseek-v4-flash",
  streaming: false, // axios is what is mocked here — pin the non-streaming transport
      messages: toolMessages({
        reasoningDetails: [
          { type: "thinking", thinking: "native", signature: "sig123" },
          { type: "reasoning.text", text: "kept" },
        ],
      }),
      functions: FUNCTIONS,
    });
    const body = mockedPost.mock.calls[0][1];
    expect(body.messages[1].reasoning_details).toEqual([
      { type: "reasoning.text", text: "kept" },
    ]);
  });

  test("reasoning is NOT injected when the caller omits it", async () => {
    await callWithRetries(["test", "no-reasoning"], {
      model: "openrouter:deepseek/deepseek-v4-flash",
  streaming: false, // axios is what is mocked here — pin the non-streaming transport
      messages: toolMessages(),
      functions: FUNCTIONS,
    });
    const body = mockedPost.mock.calls[0][1];
    expect(body.messages[1]).not.toHaveProperty("reasoning");
    expect(body.messages[1]).not.toHaveProperty("reasoning_details");
  });
});

describe("Anthropic serialization", () => {
  test("thinking + tool_use ordered, tool_result in the next user message", async () => {
    const payload: GenericPayload = {
      model: "anthropic:claude-haiku-4-5",
      messages: toolMessages({
        reasoningDetails: [
          { type: "thinking", thinking: "Let me think", signature: "sig123" },
        ],
      }),
      functions: FUNCTIONS,
    };

    await callWithRetries(["test", "anthropic"], payload);
    const body = mockedPost.mock.calls[0][1];
    const msgs = body.messages;

    // assistant turn: thinking leads, tool_use last, no injected separator
    const assistant = msgs.find((m: any) => m.role === "assistant");
    expect(assistant.content[0]).toEqual({
      type: "thinking",
      thinking: "Let me think",
      signature: "sig123",
    });
    expect(assistant.content[assistant.content.length - 1]).toEqual({
      type: "tool_use",
      id: "call_abc",
      name: "get_weather",
      input: { city: "Tokyo" },
    });
    const hasSeparator = JSON.stringify(assistant.content).includes("---");
    expect(hasSeparator).toBe(false);

    // tool_result rides on the user message immediately after the assistant turn
    const assistantIdx = msgs.indexOf(assistant);
    expect(msgs[assistantIdx + 1]).toEqual({
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "call_abc",
          content: '{"tempC":22}',
        },
      ],
    });
  });

  test("foreign reasoning blocks are dropped (OpenRouter history → Anthropic fallback)", async () => {
    // Prod repro: deepseek (OpenRouter) turns carry `reasoning.text` blocks in
    // reasoningDetails; when the loop falls back to Anthropic mid-conversation,
    // replaying them verbatim 400s ("Input tag 'reasoning.text' ... does not
    // match any of the expected tags").
    await callWithRetries(["test", "anthropic-foreign-reasoning"], {
      model: "anthropic:claude-haiku-4-5",
      messages: toolMessages({
        reasoning: "Let me check the weather.",
        reasoningDetails: [
          { type: "reasoning.text", text: "thinking", format: "unknown" },
        ],
      }),
      functions: FUNCTIONS,
    });
    const msgs = mockedPost.mock.calls[0][1].messages;
    const assistant = msgs.find((m: any) => m.role === "assistant");
    expect(
      assistant.content.filter((b: any) => b.type?.startsWith("reasoning")),
    ).toEqual([]);
    // the turn is still a valid tool-call turn
    expect(assistant.content[assistant.content.length - 1].type).toBe(
      "tool_use",
    );
  });

  test("mixed reasoningDetails keeps only thinking/redacted_thinking", async () => {
    await callWithRetries(["test", "anthropic-mixed-reasoning"], {
      model: "anthropic:claude-haiku-4-5",
      messages: toolMessages({
        reasoningDetails: [
          { type: "reasoning.text", text: "foreign" },
          { type: "thinking", thinking: "native", signature: "sig123" },
          { type: "redacted_thinking", data: "opaque" },
        ],
      }),
      functions: FUNCTIONS,
    });
    const msgs = mockedPost.mock.calls[0][1].messages;
    const assistant = msgs.find((m: any) => m.role === "assistant");
    expect(assistant.content.slice(0, 2)).toEqual([
      { type: "thinking", thinking: "native", signature: "sig123" },
      { type: "redacted_thinking", data: "opaque" },
    ]);
  });

  test("parallel calls → one tool_result block per call, all in one user message", async () => {
    await callWithRetries(["test", "anthropic-parallel"], {
      model: "anthropic:claude-haiku-4-5",
      messages: parallelToolMessages(),
      functions: FUNCTIONS,
    });
    const msgs = mockedPost.mock.calls[0][1].messages;
    const assistant = msgs.find((m: any) => m.role === "assistant");
    const toolUses = assistant.content.filter((b: any) => b.type === "tool_use");
    expect(toolUses.map((b: any) => b.id)).toEqual(["call_1", "call_2"]);

    const resultMsg = msgs[msgs.indexOf(assistant) + 1];
    expect(resultMsg.content).toEqual([
      { type: "tool_result", tool_use_id: "call_1", content: '{"tempC":22}' },
      { type: "tool_result", tool_use_id: "call_2", content: '{"tempC":15}' },
    ]);
  });
});

// ─── OpenAI (fetch) ──────────────────────────────────────────────────────────

describe("OpenAI serialization", () => {
  let fetchSpy: jest.SpyInstance;

  beforeEach(() => {
    fetchSpy = jest.spyOn(global, "fetch" as any).mockResolvedValue({
      ok: true,
      json: async () => ({
        choices: [{ message: { role: "assistant", content: "It's 22°C." } }],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      }),
    } as any);
  });
  afterEach(() => fetchSpy.mockRestore());

  test("assistant tool_calls + tool message (non-streaming o1 path)", async () => {
    // o1 model id forces the non-streaming JSON path, so the body is a single
    // JSON object we can read directly off the fetch call.
    await callWithRetries(["test", "openai"], {
      model: "openai:o1-mini",
      messages: toolMessages(),
      functions: FUNCTIONS,
    });

    const body = JSON.parse(fetchSpy.mock.calls[0][1].body as string);
    expect(body.messages[1]).toMatchObject({
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call_abc",
          type: "function",
          function: {
            name: "get_weather",
            arguments: JSON.stringify({ city: "Tokyo" }),
          },
        },
      ],
    });
    expect(body.messages[2]).toEqual({
      role: "tool",
      tool_call_id: "call_abc",
      content: '{"tempC":22}',
    });
  });
});

// ─── Google (@google/genai) ──────────────────────────────────────────────────

describe("Google serialization (REST generateContent)", () => {
  // contents of the Nth (default last) generateContent request body
  const geminiContents = (callIndex = -1) => {
    const calls = mockedPost.mock.calls.filter((c: any[]) =>
      String(c[0]).includes("generativelanguage.googleapis.com"),
    );
    const call = callIndex < 0 ? calls[calls.length + callIndex] : calls[callIndex];
    return (call[1] as any).contents as any[];
  };

  test("functionCall on the model turn, functionResponse on the user turn", async () => {
    await callWithRetries(["test", "google"], {
      model: "google:gemini-3-flash-preview",
      messages: toolMessages(),
      functions: FUNCTIONS,
    });

    const contents = geminiContents();
    const modelTurn = contents.find((m) => m.role === "model");
    expect(modelTurn.parts).toContainEqual({
      functionCall: { id: "call_abc", name: "get_weather", args: { city: "Tokyo" } },
    });
    const allParts = contents.flatMap((m: any) => m.parts);
    expect(allParts).toContainEqual({
      functionResponse: {
        id: "call_abc",
        name: "get_weather",
        response: { output: '{"tempC":22}' },
      },
    });
  });

  test("functionResponse.name is backfilled from the matching call when omitted", async () => {
    const messages = toolMessages();
    // drop the name on the tool result — SDK should look it up by id
    delete (messages[2].toolResults as any)[0].name;

    await callWithRetries(["test", "google-name-backfill"], {
      model: "google:gemini-3-flash-preview",
      messages,
      functions: FUNCTIONS,
    });

    const fr = geminiContents()
      .flatMap((m: any) => m.parts)
      .find((p: any) => p.functionResponse);
    expect(fr.functionResponse.name).toBe("get_weather");
  });

  test("captures and re-emits thoughtSignature on round-trip", async () => {
    // Gemini 3 returns a thoughtSignature on each functionCall part and 400s the
    // next turn if it isn't echoed back. Capture it on the response, re-emit it
    // when the call is fed back.
    mockedPost.mockReset();
    mockedPost.mockResolvedValue({
      data: {
        candidates: [
          {
            content: {
              parts: [
                {
                  functionCall: { name: "get_weather", args: { city: "Tokyo" } },
                  thoughtSignature: "sig-xyz",
                },
              ],
            },
          },
        ],
        usageMetadata: {
          promptTokenCount: 1,
          candidatesTokenCount: 1,
          totalTokenCount: 2,
        },
      },
    });

    const first = await callWithRetries(["test", "g-sig-1"], {
      model: "google:gemini-3-flash-preview",
      messages: [{ role: "user", content: "weather in Tokyo?" }],
      functions: FUNCTIONS,
    });
    expect(first.function_call?.thoughtSignature).toBe("sig-xyz");

    // feed the call back — the emitted functionCall part must carry the signature
    await callWithRetries(["test", "g-sig-2"], {
      model: "google:gemini-3-flash-preview",
      messages: [
        { role: "user", content: "weather in Tokyo?" },
        { role: "assistant", content: "", functionCalls: first.function_calls },
        {
          role: "tool",
          content: "",
          toolResults: [
            {
              toolCallId: first.function_call!.id as string,
              name: "get_weather",
              content: "{}",
            },
          ],
        },
      ],
      functions: FUNCTIONS,
    });

    const modelTurn = geminiContents().find((m: any) => m.role === "model");
    const fcPart = modelTurn.parts.find((p: any) => p.functionCall);
    expect(fcPart.thoughtSignature).toBe("sig-xyz");
  });

  test("thinkingConfig is forwarded verbatim into generationConfig", async () => {
    await callWithRetries(["test", "g-thinking"], {
      model: "google:gemini-3-flash-preview",
      messages: [{ role: "user", content: "hi" }],
      thinkingConfig: { thinkingLevel: "HIGH" },
    });

    const body = mockedPost.mock.calls[0][1] as any;
    expect(body.generationConfig.thinkingConfig).toEqual({
      thinkingLevel: "HIGH",
    });
  });

  test("generationConfig carries no thinkingConfig key when the caller omits it", async () => {
    await callWithRetries(["test", "g-no-thinking"], {
      model: "google:gemini-3-flash-preview",
      messages: [{ role: "user", content: "hi" }],
    });

    const body = mockedPost.mock.calls[0][1] as any;
    expect("thinkingConfig" in body.generationConfig).toBe(false);
  });

  test("usage.thoughts_tokens maps from usageMetadata.thoughtsTokenCount", async () => {
    mockedPost.mockReset();
    mockedPost.mockResolvedValue({
      data: {
        candidates: [{ content: { parts: [{ text: "hi" }] } }],
        usageMetadata: {
          promptTokenCount: 1,
          candidatesTokenCount: 5,
          totalTokenCount: 6,
          thoughtsTokenCount: 4,
        },
      },
    });

    const resp = await callWithRetries(["test", "g-thoughts-usage"], {
      model: "google:gemini-3-flash-preview",
      messages: [{ role: "user", content: "hi" }],
    });
    expect(resp.usage?.thoughts_tokens).toBe(4);
  });
});
