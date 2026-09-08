/**
 * KEYFUL live-agent tests: native multi-turn tool calls driven against real
 * provider APIs, in a genuine multi-step agent loop. 190proof drives the
 * conversation; this file is the tool runtime.
 *
 * Covers the three models igpt routes to (OpenRouter deepseek-v4-flash default,
 * Anthropic claude-haiku-4-5 fallback, Google gemini-3-flash-preview vision):
 *   - dependency-chain loop (get_coordinates → get_weather) — proves a tool
 *     result is consumed on the next turn
 *   - parallel tool calls round-trip
 *   - DeepSeek-V4 reasoning echo does not 400
 *   - replay of the REAL captured igpt agent trace to a live model
 *
 * Each provider is gated on its key (skips if absent), matching tests.spec.ts.
 * Requires OPENROUTER_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY in env.
 */
import * as fs from "fs";
import * as path from "path";
import { callWithRetries } from "../src/index";
import {
  GenericMessage,
  GenericPayload,
  FunctionDefinition,
  ParsedResponseMessage,
} from "../src/interfaces";

jest.setTimeout(120000);

type ToolImpl = (args: Record<string, any>) => string;

interface LoopResult {
  final: string | null;
  messages: GenericMessage[];
  toolCalls: { name: string; arguments: Record<string, any> }[];
  steps: number;
  sawReasoning: boolean;
}

/** A real agent loop: call → execute tools → feed results back → repeat. */
async function runAgentLoop(opts: {
  model: string;
  functions: FunctionDefinition[];
  toolImpls: Record<string, ToolImpl>;
  userContent: string;
  maxSteps?: number;
}): Promise<LoopResult> {
  const { model, functions, toolImpls, userContent } = opts;
  const maxSteps = opts.maxSteps ?? 6;
  const messages: GenericMessage[] = [{ role: "user", content: userContent }];
  const toolCalls: { name: string; arguments: Record<string, any> }[] = [];
  let sawReasoning = false;

  for (let step = 0; step < maxSteps; step++) {
    const resp: ParsedResponseMessage = await callWithRetries(
      ["live-agent", model],
      { model, messages, functions } as GenericPayload,
    );
    const calls = resp.function_calls ?? [];

    if (!calls.length) {
      return { final: resp.content, messages, toolCalls, steps: step + 1, sawReasoning };
    }

    if (resp.reasoning || resp.reasoningDetails) sawReasoning = true;

    // echo the assistant turn back, including any reasoning the model emitted
    messages.push({
      role: "assistant",
      content: resp.content ?? "",
      functionCalls: calls,
      reasoning: resp.reasoning,
      reasoningDetails: resp.reasoningDetails,
    });

    // run every tool and answer them all in one tool turn
    const toolResults = calls.map((fc) => {
      toolCalls.push({ name: fc.name, arguments: fc.arguments });
      const impl = toolImpls[fc.name];
      return {
        toolCallId: fc.id as string,
        name: fc.name,
        content: impl ? impl(fc.arguments) : `Error: unknown tool "${fc.name}"`,
      };
    });
    messages.push({ role: "tool", content: "", toolResults });
  }

  return { final: null, messages, toolCalls, steps: maxSteps, sawReasoning };
}

const MODELS = [
  {
    name: "OpenRouter DeepSeek-V4",
    model: "openrouter:deepseek/deepseek-v4-flash",
    key: "OPENROUTER_API_KEY",
  },
  {
    name: "Anthropic Haiku",
    model: "anthropic:claude-haiku-4-5",
    key: "ANTHROPIC_API_KEY",
  },
  {
    name: "Google Gemini",
    model: "google:gemini-3-flash-preview",
    key: "GEMINI_API_KEY",
  },
];

// Deliberately fake coordinates the model cannot know on its own — if get_weather
// is called with these, the model must have consumed the round-tripped result.
const FAKE_COORDS = { lat: 12.34, lng: 56.78 };

const CHAIN_FUNCTIONS: FunctionDefinition[] = [
  {
    name: "get_coordinates",
    description:
      "Get the latitude and longitude of a city. Always call this before get_weather.",
    parameters: {
      type: "object",
      properties: { city: { type: "string", description: "City name" } },
      required: ["city"],
    },
  },
  {
    name: "get_weather",
    description:
      "Get the current temperature for a latitude/longitude. Requires coordinates obtained from get_coordinates.",
    parameters: {
      type: "object",
      properties: {
        lat: { type: "number" },
        lng: { type: "number" },
      },
      required: ["lat", "lng"],
    },
  },
];

const WEATHER_BY_CITY: FunctionDefinition[] = [
  {
    name: "get_weather",
    description: "Get the current temperature of a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
];

describe.each(MODELS)("$name (live agent)", ({ model, key }) => {
  const itLive = process.env[key] ? test : test.skip;

  itLive("multi-step dependency chain consumes the round-tripped result", async () => {
    let weatherArgs: Record<string, any> | null = null;

    const result = await runAgentLoop({
      model,
      functions: CHAIN_FUNCTIONS,
      userContent:
        "What is the current temperature in Tokyo? First call get_coordinates, then pass those coordinates to get_weather, then tell me the temperature.",
      toolImpls: {
        get_coordinates: () => JSON.stringify(FAKE_COORDS),
        get_weather: (a) => {
          weatherArgs = a;
          return JSON.stringify({ tempC: 22, condition: "sunny" });
        },
      },
    });

    const names = result.toolCalls.map((c) => c.name);
    expect(names).toContain("get_coordinates");
    expect(names).toContain("get_weather");

    // the proof: get_weather got the fake coords we returned from get_coordinates
    expect(weatherArgs).not.toBeNull();
    expect((weatherArgs as any).lat).toBeCloseTo(FAKE_COORDS.lat, 1);
    expect((weatherArgs as any).lng).toBeCloseTo(FAKE_COORDS.lng, 1);

    // and it answered in text using the weather we handed back
    expect(result.final).toBeTruthy();
    expect(result.final as string).toContain("22");
  });

  itLive("parallel tool calls round-trip", async () => {
    const seen: string[] = [];

    const result = await runAgentLoop({
      model,
      functions: WEATHER_BY_CITY,
      userContent:
        "What's the weather in Tokyo and Paris? Call get_weather once for each city.",
      toolImpls: {
        get_weather: (a) => {
          const city = String(a.city || "").toLowerCase();
          seen.push(city);
          return JSON.stringify({ tempC: city.includes("paris") ? 15 : 22 });
        },
      },
    });

    expect(result.toolCalls.length).toBeGreaterThanOrEqual(2);
    expect(seen.some((c) => c.includes("tokyo"))).toBe(true);
    expect(seen.some((c) => c.includes("paris"))).toBe(true);
    expect(result.final).toBeTruthy();
  });
});

// ── DeepSeek-V4 reasoning echo (the motivating 400 case) ─────────────────────

const itOpenRouter = process.env.OPENROUTER_API_KEY ? test : test.skip;

itOpenRouter(
  "DeepSeek-V4: echoing reasoning back across a tool call does not 400",
  async () => {
    const model = "openrouter:deepseek/deepseek-v4-flash";
    const functions: FunctionDefinition[] = [
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
    const userMsg: GenericMessage = {
      role: "user",
      content: "What is the weather in Tokyo? Use the get_weather tool.",
    };

    const first = await callWithRetries(["live", "ds-rt1"], {
      model,
      messages: [userMsg],
      functions,
    });

    if (!first.function_call) {
      // model answered directly — still a valid (non-tool) outcome
      expect(first.content).toBeTruthy();
      return;
    }

    const second = await callWithRetries(["live", "ds-rt2"], {
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

    // the key assertion: the round-trip is accepted (no 400) and answers
    expect(second.content).toBeTruthy();
    // eslint-disable-next-line no-console
    console.log(
      `DeepSeek reasoning captured on turn 1: reasoning=${!!first.reasoning}, reasoning_details=${!!first.reasoningDetails}`,
    );
  },
);

// ── Replay the REAL captured igpt agent trace to a live model ─────────────────

const FIXTURE = path.join(__dirname, "fixtures", "agent-trace.json");
const REPLAY_FUNCTIONS: FunctionDefinition[] = [
  {
    name: "load_skills",
    description: "Load one or more agent skills",
    parameters: {
      type: "object",
      properties: { skills: { type: "array", items: { type: "string" } } },
      required: ["skills"],
    },
  },
  {
    name: "execute_code",
    description: "Run TypeScript in the agent sandbox",
    parameters: {
      type: "object",
      properties: { code: { type: "string" }, purpose: { type: "string" } },
      required: ["code"],
    },
  },
];

describe.each(
  MODELS.filter((m) => m.name !== "Google Gemini"), // text models for the replay
)("$name: real-trace replay", ({ model, key }) => {
  const itLive =
    process.env[key] && fs.existsSync(FIXTURE) ? test : test.skip;

  itLive("continues from the real captured tool history", async () => {
    const trace = JSON.parse(fs.readFileSync(FIXTURE, "utf8"));
    const all: GenericMessage[] = trace.messages;
    // feed everything up to (but not including) the captured final reply, so the
    // live model must produce the answer itself from the round-tripped history
    const lastIdx = all.length - 1;
    const prefix =
      all[lastIdx].role === "assistant" && !all[lastIdx].functionCalls
        ? all.slice(0, lastIdx)
        : all;

    const resp = await callWithRetries(["live", "replay"], {
      model,
      messages: prefix,
      functions: REPLAY_FUNCTIONS,
    });

    // the native tool history was accepted (no format/400 rejection) and the
    // model produced a usable continuation — either a text answer or a follow-up
    // tool call (both prove the round-tripped history was understood)
    expect(
      Boolean(resp.content) || (resp.function_calls?.length ?? 0) > 0,
    ).toBe(true);
  });
});
