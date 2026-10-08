/** One-use execution candidate with an explicit owner-verification trust boundary. No network/writer defaults or CLI. */
import { setTimeout as sleepTimer } from "node:timers/promises";
import { FOLLOWUP_SCOPE_SHA256, FOLLOWUP_STUDY_ID, verifyFollowupApproval, type FollowupReservation } from "./wikidata-followup-gates.js";
import { runFollowupTransport, type FollowupTransportPorts } from "./wikidata-followup-transport.js";
import { saveFollowupOutput, type FollowupOutputPort } from "./wikidata-followup-output.js";

function fail(): never { throw new Error("Wikidata follow-up execution: refused or interrupted."); }
async function bounded<T>(ports: FollowupTransportPorts, milliseconds: number, signal: AbortSignal | undefined, operation: (signal: AbortSignal) => Promise<T>): Promise<T> {
  if (signal?.aborted) fail();
  const controller = new AbortController(); let closed = false, rejectCutoff!: (reason: Error) => void;
  const cutoff = new Promise<never>((_, reject) => { rejectCutoff = reject; }); void cutoff.catch(() => undefined);
  const abort = () => { if (!closed) { controller.abort(); rejectCutoff(new Error("Execution interrupted")); } };
  signal?.addEventListener("abort", abort, { once: true });
  let cancelDeadline: (() => void) | undefined;
  try {
    cancelDeadline = ports.deadline(milliseconds, abort);
    if (typeof cancelDeadline !== "function" || controller.signal.aborted) fail();
    const pending = (async () => operation(controller.signal))();
    return await Promise.race([pending, cutoff]);
  } finally {
    closed = true; controller.abort(); signal?.removeEventListener("abort", abort); try { cancelDeadline?.(); } catch { /* Redacted outcome. */ }
  }
}
/** Actual Node timers only; this factory supplies no source transport, private OS port or permission. */
export function createFollowupTimingPorts() {
  const bound = (milliseconds: number, maximum: number) => {
    if (!Number.isSafeInteger(milliseconds) || milliseconds < 0 || milliseconds > maximum) fail();
  };
  return {
    now: Date.now,
    sleep: async (milliseconds: number, signal: AbortSignal) => {
      bound(milliseconds, 60000); await sleepTimer(milliseconds, undefined, { signal });
    },
    deadline: (milliseconds: number, expire: () => void) => {
      bound(milliseconds, 7 * 86400000); if (typeof expire !== "function") fail();
      const timer = setTimeout(expire, milliseconds); return () => clearTimeout(timer);
    },
  };
}
export interface FollowupExecutionPorts {
  transport: FollowupTransportPorts;
  output: FollowupOutputPort;
  /** Must independently authenticate exact human source/use approval. No production implementation exists. A declared hash is insufficient. */
  verifyOwnerApproval(binding: { approvalSha256: string; scopeSha256: string }, signal?: AbortSignal): Promise<boolean>;
}
export async function executeFollowupStudy(approval: unknown, ports: FollowupExecutionPorts, signal?: AbortSignal) {
  let phase: "configuration" | "approval" | "owner-review" | "transport" | "output" = "configuration";
  let consumption: "not-attempted" | "uncertain" | "consumed" = "not-attempted";
  let outputStatus: "not-attempted" | "attempted" | "verified" = "not-attempted";
  let reservation: FollowupReservation | undefined;
  const outcome = (state: "completed" | "refused" | "failed") => ({
    format: "private-followup-execution-outcome-v1" as const, studyId: FOLLOWUP_STUDY_ID, scopeSha256: FOLLOWUP_SCOPE_SHA256,
    state, phase, consumption, outputStatus, startedAt: reservation?.startedAt ?? null,
    expiresAt: reservation?.expiresAt ?? null, publicArtifacts: false as const,
  });
  try {
    const transport = ports?.transport;
    if ([transport?.now, transport?.inspect, transport?.reserveAtomic, transport?.fetch, transport?.sleep, transport?.deadline, ports?.output?.save]
      .some((port) => typeof port !== "function")) fail();
    phase = "approval";
    if (signal?.aborted) fail();
    const binding = verifyFollowupApproval(approval, ports.transport.now());
    phase = "owner-review";
    if (typeof ports.verifyOwnerApproval !== "function") fail();
    const verified = await bounded(ports.transport, Math.min(30000, binding.expiresAt - ports.transport.now()), signal,
      (ownerSignal) => ports.verifyOwnerApproval({ approvalSha256: binding.approvalSha256, scopeSha256: FOLLOWUP_SCOPE_SHA256 }, ownerSignal));
    if (verified !== true) fail();
    if (signal?.aborted) fail();
    verifyFollowupApproval(approval, ports.transport.now(), binding.approvalSha256);
    phase = "transport";
    const result = await runFollowupTransport(approval, { ...ports.transport,
      reserveAtomic: async (record) => {
        // Creation may throw after making a durable marker. Never turn uncertainty into a retry permission.
        consumption = "uncertain";
        const saved = await ports.transport.reserveAtomic(record);
        if (saved === true) reservation = structuredClone(record);
        if (saved === true || saved === false) consumption = "consumed";
        return saved;
      },
    }, signal);
    if (signal?.aborted) fail();
    verifyFollowupApproval(approval, ports.transport.now(), binding.approvalSha256);
    phase = "output";
    await bounded(ports.transport, Math.min(30000, Date.parse(result.reservation.expiresAt) - ports.transport.now()), signal,
      (outputSignal) => saveFollowupOutput(approval, result, () => ports.transport.now(), { save: async (payload, saveSignal) => {
        if (outputSignal.aborted) fail(); outputStatus = "attempted"; return ports.output.save(payload, saveSignal);
      } }, outputSignal));
    outputStatus = "verified";
    if (signal?.aborted) fail();
    return outcome("completed");
  } catch {
    // No provider exception, URL, source/definition byte, private ID, authority text or path enters the failure outcome.
    return outcome(consumption === "not-attempted" && outputStatus === "not-attempted" ? "refused" : "failed");
  }
}
