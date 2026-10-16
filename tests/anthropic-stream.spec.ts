/**
 * The streaming Anthropic transport (default since 1.0.128): events are
 * rebuilt into the non-streaming body, so text, thinking (with signature) and
 * tool_use input come back exactly as before. Plus the request shape and the
 * stall guard. global.fetch is mocked — no network/keys/cost.
 */
import { callWithRetries } from "../src/index";
import { GenericPayload } from "../src/interfaces";

const realFetch = global.fetch;
afterEach(() => {
  global.fetch = realFetch;
});

/** An SSE body that emits each event after `gapMs` (0 = all at once). */
function sseResponse(events: any[], gapMs = 0): Response {
  const encoder = new TextEncoder();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const body = new ReadableStream({
    async start(controller) {
      for (const e of events) {
        if (gapMs) await new Promise((r) => (timer = setTimeout(r, gapMs)));
        // split mid-event to exercise line reassembly
        const raw = `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`;
        controller.enqueue(encoder.encode(raw.slice(0, 7)));
        controller.enqueue(encoder.encode(raw.slice(7)));
      }
      controller.close();
    },
    cancel() {
      clearTimeout(timer);
    },
  });
  return new Response(body, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

const START = {
  type: "message_start",
  message: {
    id: "msg_1",
    type: "message",
    role: "assistant",
    model: "claude-haiku-5-5",
    content: [],
    stop_reason: null,
    usage: { input_tokens: 100, cache_read_input_tokens: 40, cache_creation_input_tokens: 0, output_tokens: 1 },
  },
};

const TOOL_TURN = [
  START,
  { type: "content_block_start", index: 0, content_block: { type: "thinking", thinking: "", signature: "" } },
  { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "Plan: run " } },
  { type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "the code." } },
  { type: "content_block_delta", index: 0, delta: { type: "signature_delta", signature: "sig123" } },
  { type: "content_block_stop", index: 0 },
  { type: "ping" },
  { type: "content_block_start", index: 1, content_block: { type: "text", text: "" } },
  { type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "好的，" } },
  { type: "content_block_delta", index: 1, delta: { type: "text_delta", text: "running it." } },
  { type: "content_block_stop", index: 1 },
  { type: "content_block_start", index: 2, content_block: { type: "tool_use", id: "toolu_1", name: "execute_code", input: {} } },
  { type: "content_block_delta", index: 2, delta: { type: "input_json_delta", partial_json: '{"code":"1+' } },
  { type: "content_block_delta", index: 2, delta: { type: "input_json_delta", partial_json: '1"}' } },
  { type: "content_block_stop", index: 2 },
  { type: "message_delta", delta: { stop_reason: "tool_use", stop_sequence: null }, usage: { output_tokens: 57 } },
  { type: "message_stop" },
];

const payload = (extra: Partial<GenericPayload> = {}): GenericPayload =>
  ({
    model: "anthropic:claude-haiku-5-5",
    reasoningEffort: "low",
    messages: [{ role: "user", content: "compute 1+1" }],
    functions: [{ name: "execute_code", description: "run", parameters: { type: "object", properties: { code: { type: "string" } } } }],
    ...extra,
  }) as GenericPayload;

test("rebuilds thinking, text and tool_use from the stream", async () => {
  let sent: any;
  global.fetch = jest.fn(async (_url: any, init: any) => {
    sent = JSON.parse(init.body);
    return sseResponse(TOOL_TURN);
  }) as any;

  const answer = await callWithRetries(["test", "anthropic-stream"], payload(), undefined, 1);

  expect(sent.stream).toBe(true);
  expect(sent.max_tokens).toBe(64_000);
  expect(sent.thinking).toEqual({ type: "adaptive", display: "summarized" });
  expect(sent.output_config).toEqual({ effort: "low" });

  expect(answer.content).toBe("好的，running it.");
  expect(answer.function_calls).toEqual([
    { id: "toolu_1", name: "execute_code", arguments: { code: "1+1" } },
  ]);
  expect(answer.reasoningDetails).toEqual([
    { type: "thinking", thinking: "Plan: run the code.", signature: "sig123" },
  ]);
  expect(answer.usage).toEqual({
    prompt_tokens: 140,
    completion_tokens: 57,
    total_tokens: 197,
    cached_tokens: 40,
    cache_write_tokens: 0,
  });
  expect(answer.provider).toBe("anthropic");
});

test("no effort: no thinking param and no stall guard", async () => {
  let sent: any;
  global.fetch = jest.fn(async (_url: any, init: any) => {
    sent = JSON.parse(init.body);
    // 2 gaps of 150ms each would trip a 100ms stall guard if it were armed
    return sseResponse(
      [
        START,
        { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "hi" } },
        { type: "content_block_stop", index: 0 },
        { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 2 } },
        { type: "message_stop" },
      ],
      150,
    );
  }) as any;

  const answer = await callWithRetries(
    ["test", "anthropic-stream"],
    payload({ reasoningEffort: undefined }),
    undefined,
    1,
    100,
  );
  expect(sent.thinking).toBeUndefined();
  expect(answer.content).toBe("hi");
});

test("pings don't reset the stall guard", async () => {
  global.fetch = jest.fn(async (_url: any, init: any) =>
    sseResponse([START, { type: "ping" }, { type: "ping" }, { type: "ping" }], 60),
  ) as any;

  await expect(
    callWithRetries(["test", "anthropic-stream"], payload(), undefined, 1, 100),
  ).rejects.toThrow(/stalled: no useful event for 100ms/);
});

test("a stream that ends without message_stop is an error, not a partial answer", async () => {
  global.fetch = jest.fn(async () => sseResponse(TOOL_TURN.slice(0, 10))) as any;

  await expect(
    callWithRetries(["test", "anthropic-stream"], payload(), undefined, 1),
  ).rejects.toThrow(/ended prematurely/);
});

test("an error event surfaces the API's message", async () => {
  global.fetch = jest.fn(async () =>
    sseResponse([START, { type: "error", error: { type: "overloaded_error", message: "Overloaded" } }]),
  ) as any;

  await expect(
    callWithRetries(["test", "anthropic-stream"], payload(), undefined, 1),
  ).rejects.toThrow(/Overloaded/);
});
