/**
 * Guards against the "empty 200" failure: a reasoning model (e.g. deepseek via
 * OpenRouter) can return a successful response whose output went entirely to the
 * discarded `reasoning` channel, leaving content empty. Such a response must be
 * treated as a failure so withRetries retries and callWithRetries falls back —
 * not passed through as a (useless) success.
 *
 * Pure unit test: axios.post is mocked, so no network/keys/cost. Isolated in its
 * own spec file so the mock doesn't leak into the live-API integration tests.
 */
import axios from "axios";
import { callWithRetries } from "./index";
import { GenericPayload } from "./interfaces";

jest.mock("axios");
const mockedPost = axios.post as unknown as jest.Mock;

const EMPTY_OPENROUTER = {
  data: {
    provider: "TestProvider",
    choices: [
      {
        finish_reason: "stop",
        // reasoning-only completion: content empty, no tool calls
        message: { role: "assistant", content: "", reasoning: "thinking…", tool_calls: [] },
      },
    ],
    usage: { prompt_tokens: 100, completion_tokens: 42, total_tokens: 142 },
  },
};

const GOOD_GROQ = {
  data: {
    choices: [{ message: { role: "assistant", content: "FELL_BACK_OK" } }],
    usage: { prompt_tokens: 100, completion_tokens: 5, total_tokens: 105 },
  },
};

function routeByUrl(url: string) {
  if (url.includes("openrouter.ai")) return Promise.resolve(EMPTY_OPENROUTER);
  if (url.includes("groq.com")) return Promise.resolve(GOOD_GROQ);
  throw new Error(`unexpected URL in test: ${url}`);
}

beforeEach(() => {
  mockedPost.mockReset();
  mockedPost.mockImplementation((url: string) => routeByUrl(url));
});

const basePayload: GenericPayload = {
  model: "openrouter:deepseek/deepseek-v4-flash",
  streaming: false, // axios is what is mocked here — pin the non-streaming transport
  messages: [{ role: "user", content: "hi" }],
};

test("empty OpenRouter completion (no fallback) is thrown, not returned", async () => {
  await expect(
    callWithRetries(["test", "empty-no-fallback"], basePayload, undefined, 2),
  ).rejects.toThrow();
  // retried `retries` times on the primary before giving up
  expect(mockedPost).toHaveBeenCalledTimes(2);
});

test("empty OpenRouter completion falls back to fallbackModel", async () => {
  const answer = await callWithRetries(
    ["test", "empty-with-fallback"],
    { ...basePayload, fallbackModel: "groq:qwen/qwen3-32b" },
    undefined,
    2,
  );
  expect(answer.content).toBe("FELL_BACK_OK");
  // 2 empty primary attempts + 1 successful fallback attempt
  expect(mockedPost).toHaveBeenCalledTimes(3);
  const urls = mockedPost.mock.calls.map((c) => c[0] as string);
  expect(urls.filter((u) => u.includes("openrouter.ai"))).toHaveLength(2);
  expect(urls.filter((u) => u.includes("groq.com"))).toHaveLength(1);
});
