/**
 * Guards the AbortSignal contract: when the caller passes an already-aborted (or
 * mid-flight aborting) `signal` on the payload, `callWithRetries` must reject
 * immediately — NOT retry (withRetries catch bails on signal.aborted) and NOT
 * fall back to `fallbackModel` (callWithRetries catch bails on signal.aborted).
 *
 * Pure unit test: axios.post is mocked (the axios-based adapters cover the guard
 * logic), so no network/keys/cost. Isolated in its own spec so the mock doesn't
 * leak into the live-API integration tests.
 */
import axios from "axios";
import { callWithRetries } from "./index";
import { GenericPayload } from "./interfaces";

jest.mock("axios");
const mockedPost = axios.post as unknown as jest.Mock;

// Mimic how axios surfaces an aborted request. The guard keys off signal.aborted,
// not the error shape, so the exact error here is deliberately unremarkable.
const CANCELED = Object.assign(new Error("canceled"), {
  name: "CanceledError",
  code: "ERR_CANCELED",
});

beforeEach(() => {
  mockedPost.mockReset();
  mockedPost.mockRejectedValue(CANCELED);
});

const basePayload: GenericPayload = {
  model: "groq:qwen/qwen3-32b",
  messages: [{ role: "user", content: "hi" }],
};

test("aborted signal rejects immediately without retrying", async () => {
  const controller = new AbortController();
  controller.abort();

  await expect(
    callWithRetries(
      ["test", "abort-no-retry"],
      { ...basePayload, signal: controller.signal },
      undefined,
      5, // 5 retries available — none should be used
    ),
  ).rejects.toBeDefined();

  expect(mockedPost).toHaveBeenCalledTimes(1);
});

test("aborted signal does not fall back to fallbackModel", async () => {
  const controller = new AbortController();
  controller.abort();

  await expect(
    callWithRetries(
      ["test", "abort-no-fallback"],
      {
        ...basePayload,
        signal: controller.signal,
        fallbackModel: "openrouter:deepseek/deepseek-v4-flash",
      },
      undefined,
      5,
    ),
  ).rejects.toBeDefined();

  // One attempt only: no retry on the primary, no fallback to the secondary.
  expect(mockedPost).toHaveBeenCalledTimes(1);
});

test("without a signal, failures still retry (guard is opt-in)", async () => {
  await expect(
    callWithRetries(["test", "no-signal-retries"], basePayload, undefined, 3),
  ).rejects.toBeDefined();

  // Sanity check that the abort guard doesn't short-circuit normal retries.
  expect(mockedPost).toHaveBeenCalledTimes(3);
});
