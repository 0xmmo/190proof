import http from "http";
import { AddressInfo } from "net";
import axios from "axios";

import { callWithRetries } from "./index";

/**
 * Repro for the 2026-07-26 walltime rescue (+8617812787627, poster request):
 * an OpenRouter completion ran 9.7 minutes with requestTimeoutMs=120s set and
 * no timeout ever fired.
 *
 * Root cause: axios's `timeout` option is a socket *idle* timer, not a
 * wall-clock deadline. OpenRouter keeps long generations alive by dribbling
 * bytes (SSE comments / whitespace) while the upstream model works — every
 * byte resets the idle timer, so a stuck upstream holds the request open
 * forever and the only backstop left is the worker's 585s walltime rescue.
 *
 * The fix (callOpenRouter): merge an AbortSignal.timeout(requestTimeoutMs)
 * hard deadline into the request so elapsed time — not idleness — bounds the
 * attempt.
 */
describe("axios timeout vs dribbling keep-alive bytes", () => {
  let server: http.Server;
  let baseUrl: string;
  const dribblers = new Set<NodeJS.Timeout>();

  beforeAll(async () => {
    server = http.createServer((req, res) => {
      // Headers immediately, then a keep-alive byte every 200ms, never a body.
      res.writeHead(200, { "content-type": "application/json" });
      const drip = setInterval(() => res.write(" "), 200);
      dribblers.add(drip);
      res.on("close", () => {
        clearInterval(drip);
        dribblers.delete(drip);
      });
    });
    await new Promise<void>((resolve) => server.listen(0, resolve));
    baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  });

  afterAll(async () => {
    dribblers.forEach(clearInterval);
    server.closeAllConnections?.();
    await new Promise((resolve) => server.close(resolve));
  });

  it("axios `timeout` alone never fires while bytes dribble (the bug)", async () => {
    let settled = false;
    const request = axios
      .post(`${baseUrl}/api/v1/chat/completions`, {}, { timeout: 500 })
      .catch(() => {})
      .finally(() => {
        settled = true;
      });

    // 6x the configured timeout: an idle timer would have fired long ago.
    await new Promise((resolve) => setTimeout(resolve, 3_000));
    expect(settled).toBe(false);

    // Cleanup: sever the connection so the test can end.
    server.closeAllConnections?.();
    await request;
  });

  it("an AbortSignal.timeout hard deadline bounds the same request (the fix)", async () => {
    const startedAt = Date.now();
    await expect(
      axios.post(
        `${baseUrl}/api/v1/chat/completions`,
        {},
        { timeout: 500, signal: AbortSignal.timeout(500) },
      ),
    ).rejects.toThrow();
    expect(Date.now() - startedAt).toBeLessThan(3_000);
  });

  it("callWithRetries' OpenRouter path dies at the deadline per attempt, not the walltime", async () => {
    process.env.OPENROUTER_BASE_URL = baseUrl;
    const startedAt = Date.now();
    try {
      await expect(
        callWithRetries(
          "deadline-spec",
          {
            model: "openrouter:deepseek/deepseek-v4-flash",
            messages: [{ role: "user", content: "hi" }],
            requestTimeoutMs: 400,
          },
          undefined,
          2,
        ),
      ).rejects.toThrow(/hard deadline of 400ms/);
      // 2 attempts x 400ms + retry backoff — nowhere near an unbounded hang.
      expect(Date.now() - startedAt).toBeLessThan(3_000);
    } finally {
      delete process.env.OPENROUTER_BASE_URL;
    }
  });
});
