/**
 * Guards the DeepSeek "DSML" tool-call leak: deepseek-v4-flash (via OpenRouter)
 * sometimes emits its native tool-call markup as plain assistant content instead
 * of populating `tool_calls`. callOpenRouter recovers those envelopes into
 * structured function calls; an envelope it cannot parse (truncated/malformed)
 * is treated as a failure so withRetries retries and callWithRetries falls back.
 *
 * Fixtures are the real shapes captured in production (single- and double-bar
 * variants, `string="true|false"` typing). Pure unit test: axios.post is mocked,
 * so no network/keys/cost. Isolated in its own spec so the mock doesn't leak
 * into the live-API integration tests.
 */
import axios from "axios";
import { callWithRetries } from "../src/index";
import { GenericPayload } from "../src/interfaces";

jest.mock("axios");
const mockedPost = axios.post as unknown as jest.Mock;

// ── Real DSML shapes (｜ = U+FF5C; one bar single-variant, two bars double) ──
const SINGLE_BAR_USE_SKILLS = `<｜DSML｜tool_calls>
<｜DSML｜invoke name="use_skills">
<｜DSML｜parameter name="skills" string="false">["search"]</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>`;

const DOUBLE_BAR_EXECUTE_CODE = `<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="execute_code">
<｜｜DSML｜｜parameter name="code" string="true">const data = await webSearch({ queries: ["x"] });
return data;</｜｜DSML｜｜parameter>
<｜｜DSML｜｜parameter name="purpose" string="true">FuseBase table formulas</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>`;

// The `type=Loading` partial seen in prod: an opener with no invoke.
const TRUNCATED_OPENER = `<｜｜DSML｜｜tool_calls>`;

function openrouterDsml(content: string) {
  return {
    data: {
      provider: "TestProvider",
      choices: [
        {
          finish_reason: "stop",
          message: { role: "assistant", content, tool_calls: [] },
        },
      ],
      usage: { prompt_tokens: 100, completion_tokens: 42, total_tokens: 142 },
    },
  };
}

const GOOD_GROQ = {
  data: {
    choices: [{ message: { role: "assistant", content: "FELL_BACK_OK" } }],
    usage: { prompt_tokens: 100, completion_tokens: 5, total_tokens: 105 },
  },
};

beforeEach(() => {
  mockedPost.mockReset();
});

const basePayload: GenericPayload = {
  model: "openrouter:deepseek/deepseek-v4-flash",
  streaming: false, // axios is what is mocked here — pin the non-streaming transport
  messages: [{ role: "user", content: "hi" }],
};

test("single-bar DSML use_skills is recovered into a structured call", async () => {
  mockedPost.mockImplementation((url: string) => {
    if (url.includes("openrouter.ai"))
      return Promise.resolve(openrouterDsml(SINGLE_BAR_USE_SKILLS));
    throw new Error(`unexpected URL in test: ${url}`);
  });

  const answer = await callWithRetries(
    ["test", "dsml-use-skills"],
    basePayload,
    undefined,
    2,
  );

  expect(answer.function_calls).toHaveLength(1);
  expect(answer.function_calls?.[0]?.name).toBe("use_skills");
  // string="false" → JSON-parsed to an array, not the literal string
  expect(answer.function_calls?.[0]?.arguments).toEqual({ skills: ["search"] });
  expect(answer.content).toBeNull();
  expect(mockedPost).toHaveBeenCalledTimes(1); // recovered → no retry, no fallback
});

test("double-bar DSML execute_code keeps the code arg as a literal string", async () => {
  mockedPost.mockImplementation((url: string) => {
    if (url.includes("openrouter.ai"))
      return Promise.resolve(openrouterDsml(DOUBLE_BAR_EXECUTE_CODE));
    throw new Error(`unexpected URL in test: ${url}`);
  });

  const answer = await callWithRetries(
    ["test", "dsml-exec"],
    basePayload,
    undefined,
    2,
  );

  expect(answer.function_calls).toHaveLength(1);
  const call = answer.function_calls?.[0];
  expect(call?.name).toBe("execute_code");
  // string="true" → literal string, NOT JSON-parsed
  expect(typeof call?.arguments.code).toBe("string");
  expect(call?.arguments.code).toContain("await webSearch");
  expect(call?.arguments.purpose).toBe("FuseBase table formulas");
  expect(mockedPost).toHaveBeenCalledTimes(1);
});

test("truncated DSML opener is unparseable → throws and falls back", async () => {
  mockedPost.mockImplementation((url: string) => {
    if (url.includes("openrouter.ai"))
      return Promise.resolve(openrouterDsml(TRUNCATED_OPENER));
    if (url.includes("groq.com")) return Promise.resolve(GOOD_GROQ);
    throw new Error(`unexpected URL in test: ${url}`);
  });

  const answer = await callWithRetries(
    ["test", "dsml-truncated"],
    { ...basePayload, fallbackModel: "groq:qwen/qwen3-32b" },
    undefined,
    2,
  );

  expect(answer.content).toBe("FELL_BACK_OK");
  const urls = mockedPost.mock.calls.map((c) => c[0] as string);
  // 2 unparseable primary attempts, then 1 successful fallback
  expect(urls.filter((u) => u.includes("openrouter.ai"))).toHaveLength(2);
  expect(urls.filter((u) => u.includes("groq.com"))).toHaveLength(1);
});
