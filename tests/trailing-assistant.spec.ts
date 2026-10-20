/**
 * A conversation that ends on a plain assistant message must reach the
 * provider ending on a user turn ("(continue)"). OpenRouter hosts that treat a
 * trailing assistant message as a prefill make the model continue it — junk,
 * system-prompt leaks and runaways (2026-10-19, deepseek-v4.1). Anthropic and
 * Google already appended a placeholder; the OpenAI-compatible paths did not.
 *
 * OpenRouter runs against a local SSE server, Anthropic against a mocked
 * axios.post — no keys/cost.
 */
import http from "http";
import { AddressInfo } from "net";
import axios from "axios";

import { callWithRetries } from "../src/index";
import { GenericPayload } from "../src/interfaces";

jest.mock("axios");
const mockedPost = axios.post as unknown as jest.Mock;

let server: http.Server;
let lastBody: any;

beforeAll(async () => {
  server = http.createServer((req, res) => {
    let raw = "";
    req.on("data", (c) => (raw += c));
    req.on("end", () => {
      lastBody = JSON.parse(raw);
      res.writeHead(200, { "content-type": "text/event-stream", connection: "close" });
      res.write(`data: ${JSON.stringify({ provider: "Test", choices: [{ delta: { content: "hi" } }] })}\n\n`);
      res.write(`data: ${JSON.stringify({ choices: [{ delta: {}, finish_reason: "stop" }] })}\n\n`);
      res.end("data: [DONE]\n\n");
    });
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  process.env.OPENROUTER_BASE_URL = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  process.env.OPENROUTER_API_KEY = "test";
});

afterAll(async () => {
  delete process.env.OPENROUTER_BASE_URL;
  await new Promise<void>((r) => server.close(() => r()));
});

const openRouter = (messages: GenericPayload["messages"]): GenericPayload => ({
  model: "openrouter:deepseek/deepseek-v4.1-flash-1017",
  messages,
});

test("OpenRouter: a trailing assistant message gets a (continue) user turn", async () => {
  await callWithRetries("t", openRouter([
    { role: "system", content: "sys" },
    { role: "user", content: "@olly hello from group!" },
    { role: "assistant", content: "Olly here 👋" },
  ]), undefined, 1);
  expect(lastBody.messages.map((m: any) => m.role)).toEqual(["system", "user", "assistant", "user"]);
  expect(lastBody.messages[3].content).toBe("(continue)");
});

test("OpenRouter: no placeholder after a user turn or after assistant tool calls", async () => {
  await callWithRetries("t", openRouter([{ role: "user", content: "hi" }]), undefined, 1);
  expect(lastBody.messages).toHaveLength(1);

  await callWithRetries("t", openRouter([
    { role: "user", content: "run it" },
    { role: "assistant", content: "", functionCalls: [{ id: "c1", name: "run", arguments: {} }] },
    { role: "tool", content: "", toolResults: [{ toolCallId: "c1", content: "ok" }] },
  ]), undefined, 1);
  expect(lastBody.messages.at(-1)).toMatchObject({ role: "tool", tool_call_id: "c1" });
});

test("Anthropic: the trailing placeholder is (continue), not ...", async () => {
  mockedPost.mockReset();
  mockedPost.mockResolvedValue({
    data: { stop_reason: "end_turn", content: [{ type: "text", text: "OK" }], usage: { input_tokens: 1, output_tokens: 1 } },
  });
  process.env.ANTHROPIC_API_KEY = "test";
  await callWithRetries("t", {
    model: "anthropic:claude-haiku-5-5",
    streaming: false,
    messages: [
      { role: "user", content: "hello" },
      { role: "assistant", content: "Olly here 👋" },
    ],
  }, undefined, 1);
  const sent = mockedPost.mock.calls[0][1].messages;
  expect(sent.at(-1)).toEqual({ role: "user", content: "(continue)" });
});
