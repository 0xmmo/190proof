/**
 * OpenRouter image handling: models with vision-capable endpoints get
 * OpenAI-style `image_url` content parts; everything else must stay
 * byte-identical to the pre-vision serialization (plain string content with
 * `Image (url)` / `File (url)` refs inlined). A model with no image-capable
 * endpoints rejects with a routing-layer 404 — the retry loop degrades that
 * payload to the pre-vision form and remembers the model so later calls skip
 * the doomed attempt.
 *
 * Local mock server, no keys/cost — same harness as moderation-eviction.
 */
import http from "http";
import { AddressInfo } from "net";

import { callWithRetries, openRouterImageRejectedModels } from "./index";
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
const sseAnswer = (res: http.ServerResponse, content: string) =>
  sse(res, [
    { provider: "Alibaba", choices: [{ delta: { content } }] },
    { choices: [{ delta: {}, finish_reason: "stop" }] },
  ]);
const imageRejection404 = (res: http.ServerResponse) => {
  res.writeHead(404, { "content-type": "application/json" });
  res.end(
    JSON.stringify({
      error: { message: "No endpoints found that support image input", code: 404 },
    }),
  );
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
  openRouterImageRejectedModels.clear();
});

const IMAGE_URL = "https://attachments.olly.bot/abc.jpg";
const PDF_URL = "https://attachments.olly.bot/doc.pdf";

const payload = (extra: Partial<GenericPayload> = {}): GenericPayload => ({
  model: "openrouter:qwen/qwen3.5-flash-02-23",
  messages: [
    {
      role: "user",
      content: "What is this?",
      files: [{ mimeType: "image/jpeg", url: IMAGE_URL }],
    },
  ],
  ...extra,
});

/** The exact pre-vision serialization of `payload()`'s message. */
const LEGACY_MESSAGE = {
  role: "user",
  content: `What is this?\nImage (${IMAGE_URL})`,
};

test("no attachments: content stays a plain string for any model", async () => {
  respond = (_attempt, res) => sseAnswer(res, "hi");
  await callWithRetries("spec", payload({ messages: [{ role: "user", content: "hello" }] }));
  expect(requestBodies[0].messages).toEqual([{ role: "user", content: "hello" }]);
});

test("image attachment: text part (with URL ref) + image_url part", async () => {
  respond = (_attempt, res) => sseAnswer(res, "a cat");
  await callWithRetries("spec", payload());
  expect(requestBodies[0].messages).toEqual([
    {
      role: "user",
      content: [
        { type: "text", text: LEGACY_MESSAGE.content },
        { type: "image_url", image_url: { url: IMAGE_URL } },
      ],
    },
  ]);
});

test("base64-only image becomes a data: URI part instead of being dropped", async () => {
  respond = (_attempt, res) => sseAnswer(res, "green");
  await callWithRetries(
    "spec",
    payload({
      messages: [
        {
          role: "user",
          content: "Color?",
          files: [{ mimeType: "image/png", data: "AAAA" }],
        },
      ],
    }),
  );
  expect(requestBodies[0].messages).toEqual([
    {
      role: "user",
      content: [
        { type: "text", text: "Color?" },
        { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
      ],
    },
  ]);
});

test("image-input 404 degrades to the exact pre-vision payload and succeeds", async () => {
  respond = (attempt, res) =>
    attempt === 1 ? imageRejection404(res) : sseAnswer(res, "degraded answer");

  const answer = await callWithRetries("spec", payload(), undefined, 3);
  expect(answer.content).toBe("degraded answer");
  expect(requestBodies).toHaveLength(2);
  // Attempt 1 carried image parts; attempt 2 must be byte-identical to the
  // pre-vision serialization.
  expect(Array.isArray(requestBodies[0].messages[0].content)).toBe(true);
  expect(requestBodies[1].messages).toEqual([LEGACY_MESSAGE]);
});

test("rejected model is remembered: next call sends the pre-vision form up front", async () => {
  respond = (attempt, res) =>
    attempt === 1 ? imageRejection404(res) : sseAnswer(res, "ok");
  await callWithRetries("spec", payload(), undefined, 3);

  requestBodies = [];
  respond = (_attempt, res) => sseAnswer(res, "ok again");
  await callWithRetries("spec", payload(), undefined, 3);
  expect(requestBodies).toHaveLength(1);
  expect(requestBodies[0].messages).toEqual([LEGACY_MESSAGE]);
});

test("degraded mixed attachments keep the original file-ref order", async () => {
  respond = (attempt, res) =>
    attempt === 1 ? imageRejection404(res) : sseAnswer(res, "ok");
  await callWithRetries(
    "spec",
    payload({
      messages: [
        {
          role: "user",
          content: "Compare these",
          files: [
            { mimeType: "image/jpeg", url: IMAGE_URL },
            { mimeType: "application/pdf", url: PDF_URL },
          ],
        },
      ],
    }),
    undefined,
    3,
  );
  expect(requestBodies[1].messages).toEqual([
    {
      role: "user",
      content: `Compare these\nImage (${IMAGE_URL})\nFile (${PDF_URL})`,
    },
  ]);
});

test("non-streaming transport degrades the same way", async () => {
  respond = (attempt, res) => {
    if (attempt === 1) return imageRejection404(res);
    res.writeHead(200, { "content-type": "application/json" });
    res.end(
      JSON.stringify({
        provider: "Alibaba",
        choices: [{ message: { content: "nonstream ok" }, finish_reason: "stop" }],
      }),
    );
  };
  const answer = await callWithRetries(
    "spec",
    payload({ streaming: false }),
    undefined,
    3,
  );
  expect(answer.content).toBe("nonstream ok");
  expect(requestBodies[1].messages).toEqual([LEGACY_MESSAGE]);
});
