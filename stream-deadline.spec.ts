/**
 * `streamDeadlineAt`: an absolute turn budget that bounds the whole call
 * INCLUDING retries, so a per-attempt budget can't be re-granted forever and
 * outlive the caller's own deadline (2026-07-28: a 538s completion landed just
 * as the caller's 585s turn budget expired and was discarded).
 *
 * Local SSE server, no keys/cost.
 */
import http from "http";
import { AddressInfo } from "net";

import { callWithRetries, MIN_STREAM_ATTEMPT_MS } from "./index";
import { GenericPayload } from "./interfaces";

let server: http.Server;
let attempts: number;
let respond: (attempt: number, res: http.ServerResponse) => void;
const timers = new Set<NodeJS.Timeout>();

const every = (ms: number, fn: () => void) => {
  const t = setInterval(fn, ms);
  timers.add(t);
  return t;
};

const sseHead = (res: http.ServerResponse) =>
  res.writeHead(200, { "content-type": "text/event-stream" });
const answer = (res: http.ServerResponse, text: string) => {
  sseHead(res);
  res.write(`data: ${JSON.stringify({ provider: "Novita", choices: [{ delta: { content: text } }] })}\n\n`);
  res.write(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`);
  res.write("data: [DONE]\n\n");
  res.end();
};
/** Streams useful chunks forever — healthy, but never finishes. */
const endless = (res: http.ServerResponse) => {
  sseHead(res);
  every(40, () =>
    res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: "x" } }] })}\n\n`),
  );
};

/**
 * Endless stream of tool-call fragments: healthy, never finishes, and — unlike
 * prose — cannot be salvaged as a truncated answer (half-streamed arguments
 * are unparseable). Used to exercise the retry paths, which a salvageable
 * partial deliberately short-circuits.
 */
const endlessToolCall = (res: http.ServerResponse) => {
  sseHead(res);
  res.write(
    `data: ${JSON.stringify({
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, id: "call_a", function: { name: "get_weather", arguments: '{"ci' } },
            ],
          },
        },
      ],
    })}\n\n`,
  );
  every(40, () =>
    res.write(
      `data: ${JSON.stringify({
        choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "t" } }] } }],
      })}\n\n`,
    ),
  );
};

beforeAll(async () => {
  server = http.createServer((req, res) => {
    res.setHeader("connection", "close");
    attempts++;
    let raw = "";
    req.on("data", (d) => (raw += d));
    req.on("end", () => respond(attempts, res));
  });
  await new Promise<void>((resolve) => server.listen(0, resolve));
  process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterAll(async () => {
  delete process.env.OPENROUTER_BASE_URL;
  timers.forEach(clearInterval);
  server.closeAllConnections?.();
  await new Promise((resolve) => server.close(resolve));
});

beforeEach(() => {
  attempts = 0;
});
afterEach(() => {
  timers.forEach(clearInterval);
  timers.clear();
});

const payload = (extra: Partial<GenericPayload> = {}): GenericPayload => ({
  model: "openrouter:deepseek/deepseek-v4-flash",
  messages: [{ role: "user", content: "hi" }],
  ...extra,
});

test("attempt budget is clamped to the remaining deadline, not the per-attempt cap", async () => {
  respond = (_a, res) => endless(res);

  const startedAt = Date.now();
  const result = await callWithRetries(
    "spec",
    payload({
      streamTimeoutMs: 60_000, // per-attempt cap, far beyond the deadline
      streamDeadlineAt: Date.now() + 12_000,
    }),
    undefined,
    3,
    1_000_000, // no stall kill: the stream is healthy, just endless
  );
  const elapsed = Date.now() - startedAt;
  // Cut at ~12s by the deadline (not the 60s cap), and the prose generated so
  // far comes back truncated rather than being thrown away.
  expect(elapsed).toBeGreaterThanOrEqual(11_000);
  expect(elapsed).toBeLessThan(20_000);
  expect(result.truncated).toBe(true);
  expect(attempts).toBe(1);
}, 30_000);

test("no doomed retry is started once too little time remains", async () => {
  // Tool-call fragments: nothing salvageable, so the first attempt genuinely
  // fails and the retry path is reached.
  respond = (attempt, res) =>
    attempt === 1 ? endlessToolCall(res) : answer(res, "late");

  await expect(
    callWithRetries(
      "spec",
      payload({ streamTimeoutMs: 60_000, streamDeadlineAt: Date.now() + 11_000 }),
      undefined,
      5, // five retries available — none may start past the deadline
      1_000_000,
    ),
  ).rejects.toThrow(/deadline|too little time/i);
  expect(attempts).toBe(1);
}, 30_000);

test("an already-expired deadline costs zero provider calls", async () => {
  respond = (_a, res) => answer(res, "should never be reached");

  await expect(
    callWithRetries(
      "spec",
      payload({ streamDeadlineAt: Date.now() - 1 }),
      undefined,
      3,
    ),
  ).rejects.toThrow(/too little time/i);
  expect(attempts).toBe(0);
});

test("an expired deadline propagates to fallbackModel — still zero provider calls", async () => {
  // The fallback recursion inherits streamDeadlineAt, so it fails fast too.
  // That's intended: the deadline is the caller's turn budget, and a fallback
  // generation would outlive it just the same.
  respond = (_a, res) => answer(res, "should never be reached");

  await expect(
    callWithRetries(
      "spec",
      payload({
        streamDeadlineAt: Date.now() - 1,
        fallbackModel: "openrouter:z-ai/glm-5.2",
      }),
      undefined,
      2,
    ),
  ).rejects.toThrow(/too little time/i);
  expect(attempts).toBe(0);
});

test("plenty of remaining time behaves exactly as before (no clamping)", async () => {
  respond = (_a, res) => answer(res, "fast answer");

  const result = await callWithRetries(
    "spec",
    payload({ streamDeadlineAt: Date.now() + 300_000 }),
    undefined,
    2,
  );
  expect(result.content).toBe("fast answer");
  expect(attempts).toBe(1);
});

test("without streamDeadlineAt, retries keep the full per-attempt budget", async () => {
  // Unsalvageable first attempt (tool-call fragments) so the retry actually runs.
  respond = (attempt, res) =>
    attempt === 1 ? endlessToolCall(res) : answer(res, "second attempt");

  const result = await callWithRetries(
    "spec",
    payload({ streamTimeoutMs: 1_500 }),
    undefined,
    3,
    1_000_000,
  );
  expect(result.content).toBe("second attempt");
  expect(attempts).toBe(2);
}, 30_000);

test("MIN_STREAM_ATTEMPT_MS is the documented floor", () => {
  expect(MIN_STREAM_ATTEMPT_MS).toBe(10_000);
});

// ─── truncation: keep the tokens instead of discarding them ─────────────────

/** Streams prose forever — healthy, never finishes. */
const endlessProse = (res: http.ServerResponse) => {
  sseHead(res);
  every(30, () =>
    res.write(
      `data: ${JSON.stringify({ provider: "Friendli", choices: [{ delta: { content: "word " } }] })}\n\n`,
    ),
  );
};

test("deadline cut returns the partial answer marked truncated, not an error", async () => {
  respond = (_a, res) => endlessProse(res);

  const result = await callWithRetries(
    "spec",
    payload({ streamTimeoutMs: 2_000 }),
    undefined,
    1,
    1_000_000,
  );
  expect(result.truncated).toBe(true);
  expect(result.content).toMatch(/^word( word)+$/); // trailing space trimmed
  expect(result.content!.length).toBeGreaterThan(50);
  expect(result.provider).toBe("Friendli");
  expect(attempts).toBe(1); // no retry — the partial was accepted
}, 20_000);

test("a normal completion is never marked truncated", async () => {
  respond = (_a, res) => answer(res, "complete answer");
  const result = await callWithRetries("spec", payload(), undefined, 1);
  expect(result.truncated).toBeUndefined();
});

test("a stalled stream is NOT salvaged as truncated — it retries", async () => {
  // Prose, then silence: a stall means the provider died mid-thought, so the
  // partial is not a usable answer.
  respond = (attempt, res) => {
    if (attempt === 1) {
      sseHead(res);
      res.write(
        `data: ${JSON.stringify({ choices: [{ delta: { content: "half a thought" } }] })}\n\n`,
      );
      return; // silence → stall timer fires
    }
    answer(res, "retry answer");
  };

  const result = await callWithRetries(
    "spec",
    payload({ streamTimeoutMs: 60_000 }),
    undefined,
    3,
    1_500, // stall window
  );
  expect(result.content).toBe("retry answer");
  expect(result.truncated).toBeUndefined();
  expect(attempts).toBe(2);
}, 20_000);

test("a tool-call turn cut at the deadline is not salvaged (unparseable args)", async () => {
  respond = (_a, res) => {
    sseHead(res);
    res.write(
      `data: ${JSON.stringify({
        choices: [
          {
            delta: {
              tool_calls: [
                { index: 0, id: "call_a", function: { name: "get_weather", arguments: '{"ci' } },
              ],
            },
          },
        ],
      })}\n\n`,
    );
    every(30, () =>
      res.write(
        `data: ${JSON.stringify({
          choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "t" } }] } }],
        })}\n\n`,
      ),
    );
  };

  await expect(
    callWithRetries("spec", payload({ streamTimeoutMs: 1_500 }), undefined, 1, 1_000_000),
  ).rejects.toThrow(/total deadline/);
}, 20_000);

test("caller abort is never salvaged as truncated", async () => {
  respond = (_a, res) => endlessProse(res);

  const controller = new AbortController();
  setTimeout(() => controller.abort(), 800);
  await expect(
    callWithRetries(
      "spec",
      payload({ streamTimeoutMs: 60_000, signal: controller.signal }),
      undefined,
      3,
      1_000_000,
    ),
  ).rejects.toThrow();
}, 20_000);
