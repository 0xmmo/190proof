export function timeout(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function isHeicImage(name: string, mime?: string): boolean {
  const extension = name.split(".").pop()?.toLowerCase() || "";
  return (
    ["heic", "heif", "heics"].includes(extension) ||
    !!(
      mime && ["image/heic", "image/heif", "image/heic-sequence"].includes(mime)
    )
  );
}
