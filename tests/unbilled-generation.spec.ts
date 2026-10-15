/**
 * `onUnbilledGeneration`: every OpenRouter attempt that OpenRouter can bill
 * but whose `usage` never reaches the caller is reported with its generation
 * id, so the caller can price it. Runaway generations cut at the stream
 * deadline were billed (~$0.09 each) with no usage row (Olly, 2026-10-15).
 *
 * Local SSE server, no keys/cost.
 */
import http from "http";
import { AddressInfo } from "net";

import { callWithRetries } from "../src/index";
import { GenericPayload } from "../src/interfaces";

let server: http.Server;
let attempts: number;
let respond: (attempt: number, res: http.ServerResponse) => void;
const timers = new Set<NodeJS.Timeout>();

const sseHead = (res: http.ServerResponse, generationId: string) =>
  res.writeHead(200, {
    "content-type": "text/event-stream",
    "x-generation-id": generationId,
  });
const answer = (res: http.ServerResponse, generationId: string) => {
  sseHead(res, generationId);
  res.write(`data: ${JSON.stringify({ id: generationId, provider: "Novita", choices: [{ delta: { content: "hi" } }] })}\n\n`);
  res.write(
    `data: ${JSON.stringify({
      choices: [{ delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 10, completion_tokens: 1, total_tokens: 11, cost: 0.0001 },
    })}\n\n`,
  );
  res.write("data: [DONE]\n\n");
  res.end();
};
/** Healthy prose forever: cut at the deadline and returned truncated. */
const endless = (res: http.ServerResponse, generationId: string) => {
  sseHead(res, generationId);
  const t = setInterval(
    () => res.write(`data: ${JSON.stringify({ choices: [{ delta: { content: "x" } }] })}\n\n`),
    40,
  );
  timers.add(t);
};
/** Headers (which carry the id), then nothing: the stall timer kills it. */
const silent = (res: http.ServerResponse, generationId: string) => {
  sseHead(res, generationId);
  res.flushHeaders();
};

beforeAll(async () => {
  server = http.createServer((req, res) => {
    res.setHeader("connection", "close");
    attempts++;
    req.on("data", () => {});
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

function payload(
  reported: { generationId: string; reason: string }[],
  extra: Partial<GenericPayload> = {},
): GenericPayload {
  return {
    model: "openrouter:deepseek/deepseek-v4.1-flash" as any,
    messages: [{ role: "user", content: "hi" }],
    onUnbilledGeneration: (g) => reported.push(g),
    ...extra,
  };
}

describe("onUnbilledGeneration", () => {
  it("reports a deadline-truncated answer, which has no usage", async () => {
    respond = (_n, res) => endless(res, "gen-runaway");
    const reported: { generationId: string; reason: string }[] = [];
    const result = await callWithRetries(
      "test",
      payload(reported, { streamTimeoutMs: 300 }),
      undefined,
      1,
      5_000,
    );
    expect(result.truncated).toBe(true);
    expect(reported).toEqual([{ generationId: "gen-runaway", reason: "truncated" }]);
  });

  it("reports a stalled attempt but not the retry that answered with usage", async () => {
    respond = (n, res) => (n === 1 ? silent(res, "gen-stalled") : answer(res, "gen-ok"));
    const reported: { generationId: string; reason: string }[] = [];
    const result = await callWithRetries("test", payload(reported), undefined, 2, 200);
    expect(result.content).toBe("hi");
    expect(result.usage?.cost).toBe(0.0001);
    expect(reported).toHaveLength(1);
    expect(reported[0].generationId).toBe("gen-stalled");
    expect(reported[0].reason).toMatch(/stalled/);
  });

  it("reports nothing for a normal completion", async () => {
    respond = (_n, res) => answer(res, "gen-ok");
    const reported: { generationId: string; reason: string }[] = [];
    await callWithRetries("test", payload(reported), undefined, 1, 5_000);
    expect(reported).toEqual([]);
  });

  it("reports nothing for an HTTP error, which generated nothing", async () => {
    respond = (_n, res) => {
      res.writeHead(500, { "content-type": "application/json", "x-generation-id": "gen-500" });
      res.end(JSON.stringify({ error: { message: "boom" } }));
    };
    const reported: { generationId: string; reason: string }[] = [];
    await expect(
      callWithRetries("test", payload(reported), undefined, 1, 5_000),
    ).rejects.toThrow();
    expect(reported).toEqual([]);
  });

  it("reports a stream the caller cancelled", async () => {
    respond = (_n, res) => endless(res, "gen-cancelled");
    const reported: { generationId: string; reason: string }[] = [];
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 200);
    await expect(
      callWithRetries("test", payload(reported, { signal: controller.signal }), undefined, 1, 5_000),
    ).rejects.toThrow();
    expect(reported.map((r) => r.generationId)).toEqual(["gen-cancelled"]);
  });

  it("swallows a throwing callback", async () => {
    respond = (_n, res) => endless(res, "gen-runaway");
    const result = await callWithRetries(
      "test",
      {
        ...payload([], { streamTimeoutMs: 300 }),
        onUnbilledGeneration: () => {
          throw new Error("callback bug");
        },
      },
      undefined,
      1,
      5_000,
    );
    expect(result.truncated).toBe(true);
  });
});
