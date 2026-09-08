/**
 * ParsedResponseMessage.provider attribution: OpenRouter responses carry the
 * upstream provider from the response body (the routing decision OpenRouter
 * made); direct providers get the SDK provider name stamped by callWithRetries;
 * and on model fallback the stamp reflects the model that actually answered.
 *
 * Pure unit test: axios.post is mocked, so no network/keys/cost. Isolated in its
 * own spec file so the mock doesn't leak into the live-API integration tests.
 */
import axios from "axios";
import { callWithRetries } from "../src/index";
import { GenericPayload } from "../src/interfaces";

jest.mock("axios");
const mockedPost = axios.post as unknown as jest.Mock;

const GOOD_OPENROUTER = {
  data: {
    provider: "Baidu",
    choices: [
      {
        finish_reason: "stop",
        message: { role: "assistant", content: "hello from upstream" },
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  },
};

// OpenRouter error-in-200 body: forces retries → fallback without HTTP failure.
const ERROR_OPENROUTER = {
  data: { error: { message: "provider unavailable" } },
};

const GOOD_GROQ = {
  data: {
    choices: [{ message: { role: "assistant", content: "groq answered" } }],
    usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
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

test("OpenRouter response carries the upstream provider from the body", async () => {
  mockedPost.mockResolvedValue(GOOD_OPENROUTER);
  const answer = await callWithRetries(["test", "provider-openrouter"], basePayload, undefined, 1);
  expect(answer.provider).toBe("Baidu");
});

test("OpenRouter response without a body provider falls back to the SDK name", async () => {
  const { data } = GOOD_OPENROUTER;
  mockedPost.mockResolvedValue({ data: { ...data, provider: undefined } });
  const answer = await callWithRetries(["test", "provider-missing"], basePayload, undefined, 1);
  expect(answer.provider).toBe("openrouter");
});

test("direct provider gets the SDK provider name", async () => {
  mockedPost.mockResolvedValue(GOOD_GROQ);
  const answer = await callWithRetries(
    ["test", "provider-groq"],
    { ...basePayload, model: "groq:qwen/qwen3-32b" },
    undefined,
    1,
  );
  expect(answer.provider).toBe("groq");
});

test("model fallback stamps the provider of the model that answered", async () => {
  mockedPost.mockImplementation((url: string) => {
    if (url.includes("openrouter.ai")) return Promise.resolve(ERROR_OPENROUTER);
    if (url.includes("groq.com")) return Promise.resolve(GOOD_GROQ);
    throw new Error(`unexpected URL in test: ${url}`);
  });
  const answer = await callWithRetries(
    ["test", "provider-fallback"],
    { ...basePayload, fallbackModel: "groq:qwen/qwen3-32b" },
    undefined,
    1,
  );
  expect(answer.content).toBe("groq answered");
  expect(answer.provider).toBe("groq");
});
