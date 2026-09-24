type LogLevel = "LOG" | "WARN" | "ERROR";
export type Identifier = string | string[];

function formatIdentifier(identifier: Identifier): string {
  if (Array.isArray(identifier)) {
    return identifier.map((id) => `[${id}]`).join(" ");
  }
  return `[${identifier}]`;
}

function formatMessage(
  level: LogLevel,
  identifier: Identifier,
  message: string
): string {
  return `[${level}] ${formatIdentifier(identifier)} ${message}`;
}

/** Where 190proof's own log lines go. Defaults to the console. */
export interface LogSink {
  log(message: string, ...args: any[]): void;
  warn(message: string, ...args: any[]): void;
  error(message: string, ...args: any[]): void;
}

let sink: LogSink | null = console;

/**
 * Route 190proof's logging to your own logger, or pass null to silence it
 * (useful when embedding the SDK in a library).
 */
export function setLogger(next: LogSink | null): void {
  sink = next;
}

export function log(
  identifier: Identifier,
  message: string,
  ...args: any[]
): void {
  sink?.log(formatMessage("LOG", identifier, message), ...args);
}

export function warn(
  identifier: Identifier,
  message: string,
  ...args: any[]
): void {
  sink?.warn(formatMessage("WARN", identifier, message), ...args);
}

export function error(
  identifier: Identifier,
  message: string,
  ...args: any[]
): void {
  sink?.error(formatMessage("ERROR", identifier, message), ...args);
}

/**
 * HTTP client errors (axios) carry the full request config, API-key headers
 * included; logging or rethrowing one as a `cause` leaks the key into every
 * log that prints it. Rebuild a plain Error with only what callers use:
 * message, name, code, and the response status + body. Non-HTTP errors pass
 * through untouched.
 */
export function redactError<T>(err: T): T | Error {
  const e = err as any;
  if (!e || typeof e !== "object" || !(e.config || e.request || e.isAxiosError)) {
    return err;
  }
  const clean = new Error(e.message) as any;
  clean.name = e.name;
  if (e.code !== undefined) clean.code = e.code;
  if (e.status !== undefined) clean.status = e.status;
  if (e.response) {
    clean.response = { status: e.response.status, statusText: e.response.statusText, data: e.response.data };
  }
  if (e.data !== undefined) clean.data = e.data;
  if (e.cause) clean.cause = redactError(e.cause);
  return clean;
}

export default {
  log,
  warn,
  error,
};
