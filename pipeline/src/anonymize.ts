import { createHash } from "node:crypto";

export function anonymizeUsername(username: string, salt: string): string {
  const normalized = username.trim().toLowerCase();
  return createHash("sha256")
    .update(`${salt}:${normalized}`)
    .digest("hex")
    .slice(0, 24);
}
