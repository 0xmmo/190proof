/**
 * The streaming OpenRouter transport (default since 2026-07-27), adversarially.
 *
 * Timeout semantics under test — deliberately independent budgets:
 * - streaming: `streamTimeoutMs` total (default 600s) + a per-USEFUL-chunk
 *   stall timeout (`chunkTimeoutMs`, default 15s). Useful = advances content /
 *   reasoning / tool-call fragments / finish_reason / usage. SSE comments
 *   (": OPENROUTER PROCESSING" keep-alives), role-only deltas, and raw byte
 *   dribble must NOT reset the stall timer.
 * - non-streaming (`streaming: false`): `requestTimeoutMs` hard deadline
 *   (default 180s), unchanged semantics from the 2026-07-26 fix.
 *
 * Everything runs against a scriptable local SSE server — no keys/cost. Each
 * test gets a fresh scenario via `respond = (req, res) => …`.
 */
import http from "http";
import { AddressInfo } from "net";

import {
  callWithRetries,
  OPENROUTER_STREAM_TIMEOUT_MS,
  OPENROUTER_NONSTREAM_TIMEOUT_MS,
} from "./index";
import { GenericPayload } from "./interfaces";

type Responder = (req: http.IncomingMessage, res: http.ServerResponse, body: any) => void;

let server: http.Server;
let baseUrl: string;
let respond: Responder;
let requestCount: number;
let lastRequestBody: any;
const timers = new Set<NodeJS.Timeout>();
const openResponses = new Set<http.ServerResponse>();

/** setInterval/setTimeout wrappers so a failed test can't leak timers. */
const every = (ms: number, fn: () => void) => {
  const t = setInterval(fn, ms);
  timers.add(t);
  return t;
};
const after = (ms: number, fn: () => void) => {
  const t = setTimeout(fn, ms);
  timers.add(t);
  return t;
};

const sseHead = (res: http.ServerResponse) =>
  res.writeHead(200, { "content-type": "text/event-stream" });
const event = (res: http.ServerResponse, json: any) =>
  res.write(`data: ${JSON.stringify(json)}\n\n`);
const contentDelta = (text: string, extra: any = {}) => ({
  provider: "Alibaba",
  choices: [{ delta: { content: text } }],
  ...extra,
});
const DONE = (res: http.ServerResponse) => {
  res.write("data: [DONE]\n\n");
  res.end();
};

beforeAll(async () => {
  server = http.createServer((req, res) => {
    // No keep-alive pooling, and afterEach FINishes (never RSTs) leftover
    // responses: an RST'd pooled socket would fail the next test's fetch.
    res.setHeader("connection", "close");
    openResponses.add(res);
    res.on("close", () => openResponses.delete(res));
    requestCount++;
    let raw = "";
    req.on("data", (d) => (raw += d));
    req.on("end", () => {
      lastRequestBody = raw ? JSON.parse(raw) : undefined;
      respond(req, res, lastRequestBody);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  process.env.OPENROUTER_BASE_URL = baseUrl;
});

afterAll(async () => {
  delete process.env.OPENROUTER_BASE_URL;
  timers.forEach(clearInterval);
  server.closeAllConnections?.();
  await new Promise((resolve) => server.close(resolve));
});

beforeEach(() => {
  requestCount = 0;
  lastRequestBody = undefined;
  respond = (_req, res) => {
    sseHead(res);
    DONE(res);
  };
});

afterEach(() => {
  timers.forEach(clearInterval);
  timers.clear();
  openResponses.forEach((res) => {
    try {
      res.end();
    } catch {
      /* already gone */
    }
  });
  openResponses.clear();
});

const payload = (extra: Partial<GenericPayload> = {}): GenericPayload => ({
  model: "openrouter:deepseek/deepseek-v4-flash",
  messages: [{ role: "user", content: "hi" }],
  ...extra,
});

// ─── happy paths ────────────────────────────────────────────────────────────

test("streams content, captures provider + usage from the final chunk", async () => {
  respond = (_req, res) => {
    sseHead(res);
    res.write(": OPENROUTER PROCESSING\n\n");
    event(res, { provider: "Alibaba", choices: [{ delta: { role: "assistant" } }] });
    event(res, contentDelta("你好"));
    event(res, contentDelta("，世界"));
    event(res, {
      provider: "Alibaba",
      choices: [{ delta: {}, finish_reason: "stop" }],
      usage: {
        prompt_tokens: 11,
        completion_tokens: 7,
        total_tokens: 18,
        prompt_tokens_details: { cached_tokens: 3 },
      },
    });
    DONE(res);
  };

  const answer = await callWithRetries("spec", payload(), undefined, 1);
  expect(answer.content).toBe("你好，世界");
  expect(answer.provider).toBe("Alibaba");
  expect(answer.usage).toEqual({
    prompt_tokens: 11,
    completion_tokens: 7,
    total_tokens: 18,
    cached_tokens: 3,
  });
  // the request actually asked for a stream + usage accounting
  expect(lastRequestBody.stream).toBe(true);
  expect(lastRequestBody.usage).toEqual({ include: true });
});

test("reassembles UTF-8 multi-byte chars and SSE events split across TCP chunks", async () => {
  respond = (_req, res) => {
    sseHead(res);
    const full = `data: ${JSON.stringify(contentDelta("中文流式输出"))}\n\ndata: ${JSON.stringify(
      contentDelta("第二段"),
    )}\n\n`;
    const bytes = Buffer.from(full, "utf8");
    // dribble byte-by-byte: splits both the SSE framing and CJK chars mid-sequence
    let i = 0;
    const t = every(1, () => {
      if (i >= bytes.length) {
        clearInterval(t);
        event(res, { choices: [{ delta: {}, finish_reason: "stop" }] });
        DONE(res);
        return;
      }
      res.write(bytes.subarray(i, i + 1));
      i++;
    });
  };

  const answer = await callWithRetries("spec", payload(), undefined, 1);
  expect(answer.content).toBe("中文流式输出第二段");
});

test("accumulates fragmented parallel tool calls; empty-fragment args become {}", async () => {
  respond = (_req, res) => {
    sseHead(res);
    event(res, {
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, id: "call_a", function: { name: "get_", arguments: "" } },
            ],
          },
        },
      ],
    });
    event(res, {
      choices: [
        {
          delta: {
            tool_calls: [
              { index: 0, function: { name: "weather", arguments: '{"city":' } },
              { index: 1, id: "call_b", function: { name: "get_time", arguments: "" } },
            ],
          },
        },
      ],
    });
    event(res, {
      choices: [
        {
          delta: {
            tool_calls: [{ index: 0, function: { arguments: '"Tokyo"}' } }],
          },
        },
      ],
    });
    event(res, { choices: [{ delta: {}, finish_reason: "tool_calls" }] });
    DONE(res);
  };

  const answer = await callWithRetries("spec", payload(), undefined, 1);
  expect(answer.function_calls).toEqual([
    { id: "call_a", name: "get_weather", arguments: { city: "Tokyo" } },
    { id: "call_b", name: "get_time", arguments: {} },
  ]);
});

test("accumulates the reasoning channel alongside content", async () => {
  respond = (_req, res) => {
    sseHead(res);
    event(res, { choices: [{ delta: { reasoning: "think " } }] });
    event(res, { choices: [{ delta: { reasoning: "deeper" } }] });
    event(res, contentDelta("answer"));
    event(res, { choices: [{ delta: {}, finish_reason: "stop" }] });
    DONE(res);
  };

  const answer = await callWithRetries("spec", payload(), undefined, 1);
  expect(answer.content).toBe("answer");
  expect(answer.reasoning).toBe("think deeper");
});

test("clean close with finish_reason but no [DONE] is accepted", async () => {
  respond = (_req, res) => {
    sseHead(res);
    event(res, contentDelta("done anyway"));
    event(res, { choices: [{ delta: {}, finish_reason: "stop" }] });
    res.end(); // no [DONE]
  };

  const answer = await callWithRetries("spec", payload(), undefined, 1);
  expect(answer.content).toBe("done anyway");
});

test("handles CRLF line endings", async () => {
  respond = (_req, res) => {
    sseHead(res);
    res.write(`data: ${JSON.stringify(contentDelta("crlf ok"))}\r\n\r\n`);
    res.write(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\r\n\r\n`);
    res.write("data: [DONE]\r\n\r\n");
    res.end();
  };

  const answer = await callWithRetries("spec", payload(), undefined, 1);
  expect(answer.content).toBe("crlf ok");
});

// ─── stall timeout: only USEFUL chunks reset it ─────────────────────────────

test("SSE comment keep-alives do NOT reset the stall timer", async () => {
  respond = (_req, res) => {
    sseHead(res);
    every(40, () => res.write(": OPENROUTER PROCESSING\n\n"));
  };

  const startedAt = Date.now();
  await expect(
    callWithRetries("spec", payload(), undefined, 1, 300),
  ).rejects.toThrow(/no useful chunk for 300ms/);
  const elapsed = Date.now() - startedAt;
  expect(elapsed).toBeGreaterThanOrEqual(280);
  expect(elapsed).toBeLessThan(1_500);
});

test("role-only / empty deltas do NOT reset the stall timer", async () => {
  respond = (_req, res) => {
    sseHead(res);
    every(40, () =>
      event(res, { choices: [{ delta: { role: "assistant" } }] }),
    );
  };

  await expect(
    callWithRetries("spec", payload(), undefined, 1, 300),
  ).rejects.toThrow(/no useful chunk for 300ms/);
});

test("useful chunks DO reset the stall timer (slow trickle survives)", async () => {
  respond = (_req, res) => {
    sseHead(res);
    let n = 0;
    const t = every(150, () => {
      n++;
      if (n <= 6) {
        event(res, contentDelta(`w${n} `));
      } else {
        clearInterval(t);
        event(res, { choices: [{ delta: {}, finish_reason: "stop" }] });
        DONE(res);
      }
    });
  };

  // 6 chunks * 150ms = 900ms total; stall window 400ms per useful chunk
  const answer = await callWithRetries("spec", payload(), undefined, 1, 400);
  expect(answer.content).toBe("w1 w2 w3 w4 w5 w6 ");
});

test("a mid-generation stall dies within one stall window, not the total budget", async () => {
  respond = (_req, res) => {
    sseHead(res);
    event(res, contentDelta("started fine "));
    // then the provider hangs, dribbling keep-alives
    every(40, () => res.write(": still here\n\n"));
  };

  const startedAt = Date.now();
  await expect(
    callWithRetries(
      "spec",
      payload({ streamTimeoutMs: 60_000 }),
      undefined,
      1,
      300,
    ),
  ).rejects.toThrow(/no useful chunk for 300ms/);
  expect(Date.now() - startedAt).toBeLessThan(2_000);
});

// ─── total deadline ─────────────────────────────────────────────────────────

test("a healthy stream that outlives streamTimeoutMs is killed by the total deadline", async () => {
  respond = (_req, res) => {
    sseHead(res);
    every(50, () => event(res, contentDelta("x")));
  };

  const startedAt = Date.now();
  await expect(
    callWithRetries("spec", payload({ streamTimeoutMs: 700 }), undefined, 1, 300),
  ).rejects.toThrow(/exceeded total deadline of 700ms/);
  const elapsed = Date.now() - startedAt;
  expect(elapsed).toBeGreaterThanOrEqual(650);
  expect(elapsed).toBeLessThan(2_500);
});

test("streaming ignores requestTimeoutMs — the budgets are independent", async () => {
  respond = (_req, res) => {
    sseHead(res);
    let n = 0;
    const t = every(100, () => {
      n++;
      if (n <= 5) {
        event(res, contentDelta("y"));
      } else {
        clearInterval(t);
        event(res, { choices: [{ delta: {}, finish_reason: "stop" }] });
        DONE(res);
      }
    });
  };

  // requestTimeoutMs far below the stream's ~500ms runtime: must not apply
  const answer = await callWithRetries(
    "spec",
    payload({ requestTimeoutMs: 150 }),
    undefined,
    1,
    5_000,
  );
  expect(answer.content).toBe("yyyyy");
});

test("default budgets are wired as specified: 600s stream / 180s non-stream", () => {
  expect(OPENROUTER_STREAM_TIMEOUT_MS).toBe(600_000);
  expect(OPENROUTER_NONSTREAM_TIMEOUT_MS).toBe(180_000);
});

// ─── failure shapes ─────────────────────────────────────────────────────────

test("mid-stream connection death without finish_reason is a retryable premature end", async () => {
  respond = (_req, res) => {
    sseHead(res);
    event(res, contentDelta("half an ans"));
    after(50, () => res.destroy()); // cut the socket mid-generation
  };

  await expect(
    callWithRetries("spec", payload(), undefined, 2),
  ).rejects.toThrow(/ended prematurely|terminated|aborted|socket/i);
  expect(requestCount).toBe(2); // it retried
});

test("clean close without [DONE] and without finish_reason is NOT accepted as complete", async () => {
  respond = (_req, res) => {
    sseHead(res);
    event(res, contentDelta("looks complete but is not"));
    res.end();
  };

  await expect(
    callWithRetries("spec", payload(), undefined, 1),
  ).rejects.toThrow(/ended prematurely/);
});

test("error event inside the stream surfaces the provider message", async () => {
  respond = (_req, res) => {
    sseHead(res);
    event(res, { error: { message: "Rate limit exceeded: free tier", code: 429 } });
    res.end();
  };

  await expect(
    callWithRetries("spec", payload(), undefined, 1),
  ).rejects.toThrow(/Rate limit exceeded: free tier/);
});

test("non-200 HTTP with a JSON error body surfaces the message", async () => {
  respond = (_req, res) => {
    res.writeHead(429, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: "quota exhausted" } }));
  };

  await expect(
    callWithRetries("spec", payload(), undefined, 1),
  ).rejects.toThrow(/quota exhausted/);
});

test("error-in-200 JSON body (stream request answered without SSE) surfaces the error", async () => {
  respond = (_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ error: { message: "provider unavailable" } }));
  };

  await expect(
    callWithRetries("spec", payload(), undefined, 1),
  ).rejects.toThrow(/provider unavailable/);
});

test("a full JSON completion body to a stream request is parsed as non-streaming", async () => {
  respond = (_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        provider: "Novita",
        choices: [
          { finish_reason: "stop", message: { role: "assistant", content: "plain json" } },
        ],
        usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
      }),
    );
  };

  const answer = await callWithRetries("spec", payload(), undefined, 1);
  expect(answer.content).toBe("plain json");
  expect(answer.provider).toBe("Novita");
  expect(answer.usage?.total_tokens).toBe(7);
});

test("empty completion (role-only then [DONE]) throws and falls back to fallbackModel", async () => {
  respond = (_req, res, body) => {
    sseHead(res);
    if (body.model === "deepseek/deepseek-v4-flash") {
      event(res, { choices: [{ delta: { role: "assistant" } }] });
      event(res, { choices: [{ delta: {}, finish_reason: "stop" }] });
      DONE(res);
    } else {
      event(res, contentDelta("fallback answered"));
      event(res, { choices: [{ delta: {}, finish_reason: "stop" }] });
      DONE(res);
    }
  };

  const answer = await callWithRetries(
    "spec",
    payload({ fallbackModel: "openrouter:z-ai/glm-5.2" }),
    undefined,
    2,
  );
  expect(answer.content).toBe("fallback answered");
  expect(requestCount).toBe(3); // 2 empty primary attempts + 1 fallback
});

test("streamed DSML-as-content is recovered into structured tool calls", async () => {
  const dsml = `<｜DSML｜tool_calls>
<｜DSML｜invoke name="use_skills">
<｜DSML｜parameter name="skills" string="false">["search"]</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜tool_calls>`;
  respond = (_req, res) => {
    sseHead(res);
    // arrives in fragments like real token deltas
    for (let i = 0; i < dsml.length; i += 17) {
      event(res, contentDelta(dsml.slice(i, i + 17)));
    }
    event(res, { choices: [{ delta: {}, finish_reason: "stop" }] });
    DONE(res);
  };

  const answer = await callWithRetries("spec", payload(), undefined, 1);
  expect(answer.function_calls).toEqual([
    { id: "call_0", name: "use_skills", arguments: { skills: ["search"] } },
  ]);
  expect(answer.content).toBeNull();
});

test("malformed data lines are skipped without corrupting the stream", async () => {
  respond = (_req, res) => {
    sseHead(res);
    res.write("data: {not json at all\n\n");
    event(res, contentDelta("survived"));
    event(res, { choices: [{ delta: {}, finish_reason: "stop" }] });
    DONE(res);
  };

  const answer = await callWithRetries("spec", payload(), undefined, 1);
  expect(answer.content).toBe("survived");
});

// ─── caller cancellation ────────────────────────────────────────────────────

test("caller abort mid-stream rejects immediately without retrying", async () => {
  respond = (_req, res) => {
    sseHead(res);
    every(50, () => event(res, contentDelta("z")));
  };

  const controller = new AbortController();
  after(200, () => controller.abort());
  const startedAt = Date.now();
  await expect(
    callWithRetries("spec", payload({ signal: controller.signal }), undefined, 5),
  ).rejects.toThrow();
  expect(Date.now() - startedAt).toBeLessThan(1_500);
  expect(requestCount).toBe(1); // no retry after caller abort
});

// ─── non-streaming transport (streaming: false) ─────────────────────────────

test("streaming:false uses the axios transport with the hard deadline", async () => {
  respond = (_req, res) => {
    // dribble forever: only the wall-clock deadline can end this
    res.writeHead(200, { "content-type": "application/json" });
    every(50, () => res.write(" "));
  };

  await expect(
    callWithRetries(
      "spec",
      payload({ streaming: false, requestTimeoutMs: 400 }),
      undefined,
      1,
    ),
  ).rejects.toThrow(/hard deadline of 400ms/);
  expect(lastRequestBody.stream).toBeUndefined();
  expect(lastRequestBody.usage).toBeUndefined();
});

test("streaming:false happy path still parses provider/usage/tool calls", async () => {
  respond = (_req, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        provider: "GMICloud",
        choices: [
          {
            finish_reason: "tool_calls",
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_x",
                  function: { name: "get_weather", arguments: '{"city":"Osaka"}' },
                },
              ],
            },
          },
        ],
        usage: { prompt_tokens: 9, completion_tokens: 4, total_tokens: 13 },
      }),
    );
  };

  const answer = await callWithRetries(
    "spec",
    payload({ streaming: false }),
    undefined,
    1,
  );
  expect(answer.provider).toBe("GMICloud");
  expect(answer.function_call).toEqual({
    id: "call_x",
    name: "get_weather",
    arguments: { city: "Osaka" },
  });
});
