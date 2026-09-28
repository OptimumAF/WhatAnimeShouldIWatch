/** Public release identities and fixed local troubleshooting codes only. */
import type { ReleaseManifestV1 } from "./artifacts";

export type DiagnosticCode =
  | "DATA-001" | "DATA-002" | "MODEL-001" | "IMPORT-001"
  | "IMPORT-002" | "STORAGE-001" | "EXPLORER-001"
  | "RENDER-001" | "SEASONAL-001";

const actions: Record<DiagnosticCode, { title: string; action: string }> = {
  "DATA-001": { title: "Data unavailable", action: "Reload the page. If this persists, restore the last verified data release." },
  "DATA-002": { title: "Data invalid", action: "Restore the last verified data release and check its manifest and asset hashes." },
  "MODEL-001": { title: "Model unavailable or invalid", action: "Graph fallback is active. Check the model asset against this data release, or use Graph mode." },
  "IMPORT-001": { title: "Provider import failed", action: "Try a local file or text import. Your current history was not replaced." },
  "IMPORT-002": { title: "Local import failed", action: "Check the .txt or .xml format, then preview again. Your current history was not replaced." },
  "STORAGE-001": { title: "Browser storage problem", action: "Read the storage warning before closing this tab. Check browser storage permissions and free space." },
  "EXPLORER-001": { title: "Explorer data failed", action: "Reload the page. If this persists, check the explorer asset for the active data release." },
  "RENDER-001": { title: "Network render failed", action: "Reload the page and reduce visible graph edges. The recommendation data remains separate." },
  "SEASONAL-001": { title: "Seasonal data unavailable", action: "Retry later. Existing recommendations and local history are unchanged." },
};

function shortSha(value: string): string {
  return /^[a-f0-9]{64}$/.test(value) ? value.slice(0, 12) : "unverified";
}

function safeFormat(value: string): string {
  return /^[a-z][a-z0-9-]{0,63}$/.test(value) ? value : "unverified format";
}

export function appVersionLabel(version: string, revision: string): string {
  const safeVersion = /^\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?$/.test(version) ? version : "unverified";
  const build = /^[a-f0-9]{40}$/.test(revision) ? revision.slice(0, 12) : "local build";
  return `${safeVersion} · source ${build}`;
}

export function dataVersionLabel(manifest: ReleaseManifestV1 | null, graphFormat: string,
  demoMode: boolean): string {
  const format = safeFormat(graphFormat);
  if (demoMode) return `Synthetic fixture · ${format}`;
  if (!manifest) return `Legacy unversioned data · ${format}`;
  const tag = /^data-v[A-Za-z0-9][A-Za-z0-9._-]*$/.test(manifest.tag)
    ? manifest.tag : "unverified tag";
  return `${tag} · ${format} · bundle ${shortSha(manifest.bundleId)}`;
}

export function modelVersionLabel(manifest: ReleaseManifestV1 | null,
  loadedFormat: string | null, state: "unchecked" | "loaded" | "absent" | "failed",
  demoMode: boolean): string {
  if (manifest && !manifest.model) return "Not included in this data release";
  if (manifest?.model) {
    const status = state === "loaded" ? "loaded" : state === "absent"
      ? "no usable titles; graph fallback active" : state === "failed"
        ? "failed validation; graph fallback active" : "declared; not loaded";
    return `${safeFormat(manifest.model.format)} · sha256 ${shortSha(manifest.model.sha256)} · ${status}`;
  }
  if (state === "failed") return "Failed validation; graph fallback active";
  if (state === "loaded" && loadedFormat) {
    return `${safeFormat(loadedFormat)} · ${demoMode ? "synthetic fixture" : "legacy unpinned asset"}`;
  }
  if (state === "absent") return "Optional model absent; graph available";
  return demoMode ? "Synthetic model not checked" : "Legacy optional model not checked";
}

export function diagnosticIssue(code: DiagnosticCode): { code: DiagnosticCode; title: string; action: string } {
  return { code, ...actions[code] };
}
