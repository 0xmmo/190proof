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

export function log(
  identifier: Identifier,
  message: string,
  ...args: any[]
): void {
  console.log(formatMessage("LOG", identifier, message), ...args);
}

export function warn(
  identifier: Identifier,
  message: string,
  ...args: any[]
): void {
  console.warn(formatMessage("WARN", identifier, message), ...args);
}

export function error(
  identifier: Identifier,
  message: string,
  ...args: any[]
): void {
  console.error(formatMessage("ERROR", identifier, message), ...args);
}

export default {
  log,
  warn,
  error,
};
