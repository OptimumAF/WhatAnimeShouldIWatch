/** Keep provider and artifact image references on ordinary web schemes. */
export function safeExternalImageUrl(value: string): string | null {
  const candidate = value.trim();
  if (!/^https?:\/\//i.test(candidate) || /[\u0000-\u001f\u007f]/.test(candidate)) {
    return null;
  }

  try {
    const url = new URL(candidate);
    if ((url.protocol !== "http:" && url.protocol !== "https:") ||
        !url.hostname || url.username || url.password) {
      return null;
    }
    return url.href;
  } catch {
    return null;
  }
}
