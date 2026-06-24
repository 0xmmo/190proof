/**
 * Serialization of a REAL captured igpt agent trace (deepseek-v4-flash run,
 * reshaped into native tool-call messages — see igpt's _capture-trace.spec.ts).
 *
 * Pure unit test: transports mocked, no keys/network. Feeds the real multi-turn
 * history through the actual serialization path (callWithRetries) for the three
 * providers igpt routes to (OpenRouter default, Anthropic fallback, Google
 * vision) and asserts the cross-provider invariants hold on real data — every
 * tool call is emitted, every result is paired, no tool_use is orphaned.
 *
 * Skips itself if the fixture hasn't been captured yet, so CI stays green.
 */
import * as fs from "fs";
import * as path from "path";
import axios from "axios";
import { callWithRetries } from "./index";
import { GenericMessage } from "./interfaces";

jest.mock("axios");

const FIXTURE = path.join(__dirname, "fixtures", "agent-trace.json");
const haveFixture = fs.existsSync(FIXTURE);
const describeIf = haveFixture ? describe : describe.skip;

describeIf("real agent trace serialization", () => {
  const trace = haveFixture
    ? JSON.parse(fs.readFileSync(FIXTURE, "utf8"))
    : { messages: [] };
  const messages: GenericMessage[] = trace.messages;

  // Expected totals derived straight from the captured history.
  const totalCalls = messages.reduce(
    (n, m) => n + (m.functionCalls?.length ?? 0),
    0,
  );
  const totalResults = messages.reduce(
    (n, m) => n + (m.toolResults?.length ?? 0),
    0,
  );

  test("fixture is a multi-turn tool trace", () => {
    expect(totalCalls).toBeGreaterThan(0);
    expect(totalResults).toBeGreaterThan(0);
  });

  describe("OpenRouter (igpt default)", () => {
    const mockedPost = axios.post as unknown as jest.Mock;
    beforeEach(() => {
      mockedPost.mockReset();
      mockedPost.mockResolvedValue({
        data: {
          choices: [{ message: { role: "assistant", content: "ok" } }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        },
      });
    });

    test("every call is emitted and every result paired to a real call id", async () => {
      await callWithRetries(["trace", "openrouter"], {
        model: "openrouter:deepseek/deepseek-v4-flash",
        messages,
      });
      const body = mockedPost.mock.calls[0][1];

      const emittedCallIds = new Set<string>();
      let emittedCalls = 0;
      let toolMessages = 0;
      for (const m of body.messages) {
        for (const tc of m.tool_calls ?? []) {
          emittedCallIds.add(tc.id);
          emittedCalls++;
          expect(tc.type).toBe("function");
          expect(typeof tc.function.arguments).toBe("string"); // JSON-stringified
        }
        if (m.role === "tool") {
          toolMessages++;
          expect(emittedCallIds.has(m.tool_call_id)).toBe(true); // paired, not orphaned
        }
      }
      expect(emittedCalls).toBe(totalCalls);
      expect(toolMessages).toBe(totalResults);
    });
  });

  describe("Anthropic (igpt fallback)", () => {
    const mockedPost = axios.post as unknown as jest.Mock;
    beforeEach(() => {
      mockedPost.mockReset();
      mockedPost.mockResolvedValue({
        data: {
          content: [{ type: "text", text: "ok" }],
          usage: { input_tokens: 1, output_tokens: 1 },
        },
      });
    });

    test("every tool_use is matched by a tool_result (no orphans)", async () => {
      await callWithRetries(["trace", "anthropic"], {
        model: "anthropic:claude-haiku-4-5",
        messages,
      });
      const body = mockedPost.mock.calls[0][1];

      const toolUseIds = new Set<string>();
      const toolResultIds = new Set<string>();
      for (const m of body.messages) {
        if (!Array.isArray(m.content)) continue;
        for (const block of m.content) {
          if (block.type === "tool_use") toolUseIds.add(block.id);
          if (block.type === "tool_result") toolResultIds.add(block.tool_use_id);
        }
      }
      expect(toolUseIds.size).toBe(totalCalls);
      // every result references a real tool_use, and every tool_use is answered
      for (const id of toolResultIds) expect(toolUseIds.has(id)).toBe(true);
      for (const id of toolUseIds) expect(toolResultIds.has(id)).toBe(true);
    });
  });

  describe("Google (igpt vision)", () => {
    const mockedPost = axios.post as unknown as jest.Mock;
    beforeEach(() => {
      mockedPost.mockReset();
      mockedPost.mockResolvedValue({
        data: {
          candidates: [{ content: { parts: [{ text: "ok" }] } }],
          usageMetadata: {
            promptTokenCount: 1,
            candidatesTokenCount: 1,
            totalTokenCount: 2,
          },
        },
      });
    });

    test("functionCall and functionResponse parts round-trip with names", async () => {
      await callWithRetries(["trace", "google"], {
        model: "google:gemini-3-flash-preview",
        messages,
      });

      const contents = (mockedPost.mock.calls[0][1] as any).contents;
      const allParts = contents.flatMap((m: any) => m.parts);
      const fnCalls = allParts.filter((p: any) => "functionCall" in p);
      const fnResponses = allParts.filter((p: any) => "functionResponse" in p);
      expect(fnCalls.length).toBe(totalCalls);
      expect(fnResponses.length).toBe(totalResults);
      // names are present (required by Gemini), backfilled where the caller omitted
      for (const p of fnResponses) expect(p.functionResponse.name).toBeTruthy();
    });
  });
});
