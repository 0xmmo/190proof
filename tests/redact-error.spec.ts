import axios from "axios";
import { createServer, Server } from "http";
import { AddressInfo } from "net";
import { callWithRetries, redactError, setLogger } from "../src/index";

describe("API keys never leak through errors or logs", () => {
  let server: Server;
  let port: number;
  const logged: string[] = [];

  beforeAll(async () => {
    server = createServer((_req, res) => {
      res.writeHead(400, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "usage limit reached" } }));
    });
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    port = (server.address() as AddressInfo).port;
    const capture = (...args: any[]) => logged.push(args.map((a) => (typeof a === "string" ? a : JSON.stringify(a))).join(" "));
    setLogger({ log: capture, warn: capture, error: capture });
  });

  afterAll(async () => {
    setLogger(console);
    await new Promise<void>((r) => server.close(() => r()));
  });

  it("redactError strips request config but keeps message, code and response", async () => {
    const err = await axios
      .get(`http://127.0.0.1:${port}/`, { headers: { Authorization: "Bearer sk-SECRET" } })
      .catch((e) => e);
    const clean = redactError(err) as any;
    expect(clean.message).toContain("400");
    expect(clean.response.status).toBe(400);
    expect(clean.response.data.error.message).toBe("usage limit reached");
    expect(clean.config).toBeUndefined();
    expect(clean.request).toBeUndefined();
    const plain = new Error("x");
    expect(redactError(plain)).toBe(plain);
  });

  it("a failing provider call throws and logs without the key", async () => {
    const err: any = await callWithRetries(
      "test",
      { model: "openai:gpt-5-mini", messages: [{ role: "user", content: "hi" }], streaming: false } as any,
      { service: "openai", apiKey: "sk-SECRET-KEY", baseUrl: `http://127.0.0.1:${port}` } as any,
      1,
    ).catch((e) => e);
    expect(err).toBeInstanceOf(Error);
    const seen = JSON.stringify(err, Object.getOwnPropertyNames(err)) + String(err?.cause && JSON.stringify(err.cause, Object.getOwnPropertyNames(err.cause))) + logged.join("\n");
    expect(seen).not.toContain("sk-SECRET-KEY");
  });

  it("setLogger(null) silences 190proof", () => {
    const before = logged.length;
    setLogger(null);
    const { log } = require("../src/logger");
    log("x", "should not appear");
    expect(logged.length).toBe(before);
  });
});
