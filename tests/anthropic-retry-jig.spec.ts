/**
 * Guards against retries corrupting the Anthropic request: jigAnthropicMessages
 * used to merge consecutive same-role messages by mutating the caller's message
 * objects, and callWithRetries reuses those objects on every attempt — so each
 * retry re-appended the follower's blocks and attempt 2 onward 400'd with
 * "`tool_use` ids must be unique" (prod 2026-10-08, claude-haiku-5-5 fallback).
 *
 * Pure unit test: axios.post is mocked, so no network/keys/cost.
 */
import axios from "axios";
import { callWithRetries } from "../src/index";
import { GenericPayload } from "../src/interfaces";

jest.mock("axios");
const mockedPost = axios.post as unknown as jest.Mock;

const THINKING_ONLY = {
  data: {
    stop_reason: "max_tokens",
    content: [{ type: "thinking", thinking: "…", signature: "sig" }],
    usage: { input_tokens: 100, output_tokens: 16000 },
  },
};

const GOOD = {
  data: {
    stop_reason: "end_turn",
    content: [{ type: "text", text: "OK" }],
    usage: { input_tokens: 100, output_tokens: 1 },
  },
};

test("retries send the same Anthropic messages every attempt", async () => {
  const sent: any[] = [];
  let calls = 0;
  mockedPost.mockReset();
  mockedPost.mockImplementation((_url: string, body: any) => {
    // snapshot at send time — later mutation must not leak into the next attempt
    sent.push(JSON.parse(JSON.stringify(body.messages)));
    return Promise.resolve(calls++ === 0 ? THINKING_ONLY : GOOD);
  });

  const payload: GenericPayload = {
    model: "anthropic:claude-haiku-5-5",
    messages: [
      { role: "user", content: "do two things" },
      {
        role: "assistant",
        content: "",
        functionCalls: [{ id: "toolu_a", name: "run", arguments: {} }],
      },
      {
        role: "assistant",
        content: "",
        functionCalls: [{ id: "toolu_b", name: "run", arguments: {} }],
      },
      { role: "tool", content: "", toolResults: [{ toolCallId: "toolu_a", content: "1" }] },
      { role: "tool", content: "", toolResults: [{ toolCallId: "toolu_b", content: "2" }] },
    ],
  } as GenericPayload;

  const answer = await callWithRetries(["test", "anthropic-retry-jig"], payload, undefined, 3);
  expect(answer.content).toBe("OK");
  expect(sent).toHaveLength(2);
  expect(sent[1]).toEqual(sent[0]);

  const toolUseIds = sent[1]
    .flatMap((m: any) => m.content)
    .filter((b: any) => b.type === "tool_use")
    .map((b: any) => b.id);
  expect(toolUseIds).toEqual(["toolu_a", "toolu_b"]);
});
