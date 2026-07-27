/**
 * Moderation eviction: a provider content-moderation rejection is
 * deterministic, so the first one evicts the refusing provider from the
 * request's provider preferences (ignore += slug, order -= slug) and every
 * remaining retry reroutes. Non-moderation errors keep plain retry semantics.
 *
 * Local mock server, no keys/cost. Each test scripts per-attempt responses and
 * asserts on the exact provider preferences each attempt sent.
 */
import http from "http";
import { AddressInfo } from "net";

import { callWithRetries } from "./index";
import { GenericPayload } from "./interfaces";

let server: http.Server;
let requestBodies: any[];
let respond: (attempt: number, res: http.ServerResponse) => void;

const sse = (res: http.ServerResponse, events: any[]) => {
  res.writeHead(200, { "content-type": "text/event-stream" });
  for (const e of events) res.write(`data: ${JSON.stringify(e)}\n\n`);
  res.write("data: [DONE]\n\n");
  res.end();
};
const sseError = (res: http.ServerResponse, error: any) =>
  sse(res, [{ error }]);
const sseAnswer = (res: http.ServerResponse, content: string) =>
  sse(res, [
    { provider: "Novita", choices: [{ delta: { content } }] },
    { choices: [{ delta: {}, finish_reason: "stop" }] },
  ]);

const ALIBABA_MODERATION = {
  message:
    "Upstream error from Alibaba: Output data may contain inappropriate content.",
  code: 502,
  metadata: { provider_name: "Alibaba", raw: "data_inspection_failed" },
};

beforeAll(async () => {
  server = http.createServer((req, res) => {
    res.setHeader("connection", "close");
    let raw = "";
    req.on("data", (d) => (raw += d));
    req.on("end", () => {
      requestBodies.push(raw ? JSON.parse(raw) : undefined);
      respond(requestBodies.length, res);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, resolve));
  process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
});

afterAll(async () => {
  delete process.env.OPENROUTER_BASE_URL;
  server.closeAllConnections?.();
  await new Promise((resolve) => server.close(resolve));
});

beforeEach(() => {
  requestBodies = [];
});

const payload = (extra: Partial<GenericPayload> = {}): GenericPayload => ({
  model: "openrouter:deepseek/deepseek-v4-flash",
  messages: [{ role: "user", content: "hi" }],
  provider: {
    order: ["alibaba", "novita/fp8", "atlas-cloud/fp4"],
    ignore: ["streamlake", "baidu"],
  },
  ...extra,
});

test("first moderation rejection evicts the provider; retry reroutes and succeeds", async () => {
  respond = (attempt, res) =>
    attempt === 1 ? sseError(res, ALIBABA_MODERATION) : sseAnswer(res, "rerouted");

  const answer = await callWithRetries("spec", payload(), undefined, 3);
  expect(answer.content).toBe("rerouted");
  expect(requestBodies).toHaveLength(2); // exactly one wasted attempt, not five

  // attempt 1 carried the original prefs
  expect(requestBodies[0].provider).toEqual({
    order: ["alibaba", "novita/fp8", "atlas-cloud/fp4"],
    ignore: ["streamlake", "baidu"],
  });
  // attempt 2: alibaba ignored AND dropped from order; static ignores preserved
  expect(requestBodies[1].provider).toEqual({
    order: ["novita/fp8", "atlas-cloud/fp4"],
    ignore: ["streamlake", "baidu", "alibaba"],
  });
});

test("display-name provider maps to the order entry's slug (AtlasCloud → atlas-cloud)", async () => {
  respond = (attempt, res) =>
    attempt === 1
      ? sseError(res, {
          message:
            "Upstream error from AtlasCloud: Output data may contain inappropriate content.",
          metadata: { provider_name: "AtlasCloud" },
        })
      : sseAnswer(res, "ok");

  await callWithRetries("spec", payload(), undefined, 3);
  expect(requestBodies[1].provider.ignore).toContain("atlas-cloud");
  expect(requestBodies[1].provider.order).toEqual(["alibaba", "novita/fp8"]);
});

test("a second refusing provider is evicted cumulatively", async () => {
  respond = (attempt, res) => {
    if (attempt === 1) return sseError(res, ALIBABA_MODERATION);
    if (attempt === 2)
      return sseError(res, {
        message:
          "Upstream error from Novita: Output data may contain inappropriate content.",
        metadata: { provider_name: "Novita" },
      });
    sseAnswer(res, "third time lucky");
  };

  const answer = await callWithRetries("spec", payload(), undefined, 4);
  expect(answer.content).toBe("third time lucky");
  expect(requestBodies[2].provider).toEqual({
    order: ["atlas-cloud/fp4"],
    ignore: ["streamlake", "baidu", "alibaba", "novita"],
  });
});

test("non-moderation errors do NOT evict — plain retry with unchanged prefs", async () => {
  respond = (attempt, res) =>
    attempt === 1
      ? sseError(res, { message: "Internal Server Error", code: 500 })
      : sseAnswer(res, "recovered");

  await callWithRetries("spec", payload(), undefined, 3);
  expect(requestBodies[1].provider).toEqual(requestBodies[0].provider);
});

test("moderation without an identifiable provider retries without evicting", async () => {
  respond = (attempt, res) =>
    attempt === 1
      ? sseError(res, {
          message: "Content flagged by moderation.",
          code: 403,
          metadata: { reasons: ["sexual"] },
        })
      : sseAnswer(res, "ok");

  const answer = await callWithRetries("spec", payload(), undefined, 3);
  expect(answer.content).toBe("ok");
  expect(requestBodies[1].provider).toEqual(requestBodies[0].provider);
});

test("payload without provider prefs gains an ignore-only preference on eviction", async () => {
  respond = (attempt, res) =>
    attempt === 1 ? sseError(res, ALIBABA_MODERATION) : sseAnswer(res, "ok");

  await callWithRetries("spec", payload({ provider: undefined }), undefined, 3);
  expect(requestBodies[0].provider).toBeUndefined();
  expect(requestBodies[1].provider).toEqual({
    ignore: ["alibaba"],
    order: undefined,
  });
});

test("non-streaming transport: error-in-200 moderation body also evicts on retry", async () => {
  respond = (attempt, res) => {
    res.writeHead(200, { "content-type": "application/json" });
    if (attempt === 1) {
      res.end(JSON.stringify({ error: ALIBABA_MODERATION }));
    } else {
      res.end(
        JSON.stringify({
          provider: "Novita",
          choices: [
            { finish_reason: "stop", message: { role: "assistant", content: "ok" } },
          ],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
      );
    }
  };

  const answer = await callWithRetries(
    "spec",
    payload({ streaming: false }),
    undefined,
    3,
  );
  expect(answer.content).toBe("ok");
  expect(requestBodies[1].provider.ignore).toContain("alibaba");
});

test("eviction still lands on fallbackModel when the whole pool refuses", async () => {
  respond = (attempt, res) => {
    const body = requestBodies[requestBodies.length - 1];
    if (body.model === "deepseek/deepseek-v4-flash") {
      // every provider refuses, whoever it routes to
      return sseError(res, ALIBABA_MODERATION);
    }
    sseAnswer(res, "fallback model answered");
  };

  const answer = await callWithRetries(
    "spec",
    payload({ fallbackModel: "openrouter:z-ai/glm-5.2" }),
    undefined,
    2,
  );
  expect(answer.content).toBe("fallback model answered");
});
