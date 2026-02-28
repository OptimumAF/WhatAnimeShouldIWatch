import path from "node:path";
import { fileURLToPath } from "node:url";

export function getRepoRoot(metaUrl: string): string {
  const thisFile = fileURLToPath(metaUrl);
  const thisDir = path.dirname(thisFile);
  return path.resolve(thisDir, "..", "..");
}
