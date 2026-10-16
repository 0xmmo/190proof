/**
 * Direct OpenAI rejects some image URLs mid-stream (invalid_image, e.g. "File
 * urls cannot be larger than 20MB" on a 48MB GIF). The retry must drop the
 * image parts and keep the `Image (url)` text refs, as it already did for
 * content_policy_violation. global.fetch is mocked — no network/keys/cost.
 */
import { callWithRetries } from "../src/index";
import { GenericPayload } from "../src/interfaces";

const realFetch = global.fetch;
afterEach(() => {
  global.fetch = realFetch;
});

function sseResponse(chunks: any[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      for (const c of chunks) controller.enqueue(encoder.encode(`data: ${JSON.stringify(c)}\n\n`));
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

const INVALID_IMAGE = {
  error: {
    message: "Failed to download file. File urls cannot be larger than 20MB.",
    type: "invalid_request_error",
    param: null,
    code: "invalid_image",
  },
};
const TEXT_REPLY = { choices: [{ index: 0, delta: { role: "assistant", content: "Here is your code." }, finish_reason: "stop" }] };

const payload = (): GenericPayload =>
  ({
    model: "openai:gpt-6-luna",
    messages: [
      {
        role: "assistant",
        content: "Here's the animation.",
        files: [{ mimeType: "image/gif", url: "https://attachments.example/big.gif" }],
      },
      { role: "user", content: "what's my code" },
    ],
  }) as GenericPayload;

const imagePartsIn = (body: any) =>
  body.messages.flatMap((m: any) => (Array.isArray(m.content) ? m.content : [])).filter((c: any) => c.type === "image_url");
const textIn = (body: any) => JSON.stringify(body.messages);

test("invalid_image: retry drops image parts, keeps the Image (url) ref", async () => {
  const bodies: any[] = [];
  global.fetch = jest.fn(async (_url: any, init: any) => {
    bodies.push(JSON.parse(init.body));
    return sseResponse(bodies.length === 1 ? [INVALID_IMAGE] : [TEXT_REPLY]);
  }) as any;

  const answer = await callWithRetries(["test", "openai-image-drop"], payload(), { apiKey: "k" } as any, 2);

  expect(answer.content).toBe("Here is your code.");
  expect(bodies).toHaveLength(2);
  expect(imagePartsIn(bodies[0])).toHaveLength(1);
  expect(imagePartsIn(bodies[1])).toHaveLength(0);
  expect(textIn(bodies[1])).toContain("Image (https://attachments.example/big.gif)");
});

test("other stream errors keep the images", async () => {
  const bodies: any[] = [];
  global.fetch = jest.fn(async (_url: any, init: any) => {
    bodies.push(JSON.parse(init.body));
    return sseResponse(
      bodies.length === 1 ? [{ error: { message: "overloaded", type: "server_error", code: "server_error" } }] : [TEXT_REPLY],
    );
  }) as any;

  await callWithRetries(["test", "openai-image-drop"], payload(), { apiKey: "k" } as any, 2);

  expect(imagePartsIn(bodies[1])).toHaveLength(1);
});
