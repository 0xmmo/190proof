type LogLevel = "LOG" | "WARN" | "ERROR";

function formatMessage(
  level: LogLevel,
  identifier: string,
  message: string
): string {
  return `[${level}] [${identifier}] ${message}`;
}

export function log(identifier: string, message: string, ...args: any[]): void {
  console.log(formatMessage("LOG", identifier, message), ...args);
}

export function warn(
  identifier: string,
  message: string,
  ...args: any[]
): void {
  console.warn(formatMessage("WARN", identifier, message), ...args);
}

export function error(
  identifier: string,
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
