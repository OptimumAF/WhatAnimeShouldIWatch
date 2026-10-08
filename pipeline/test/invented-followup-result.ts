import { runFollowupTransport, type FollowupTransportPorts } from "../src/wikidata-followup-transport.js";
import { FOLLOWUP_SCOPE_SHA256, FOLLOWUP_STUDY_ID, type FollowupGatePorts } from "../src/wikidata-followup-gates.js";

/** Entirely invented fixture; transport is always mocked, optional OS gates touch only test-owned temporary paths. */
export function inventedFollowupPorts(gates?: FollowupGatePorts) {
  let clock = Date.now() - 60000;
  let consumed = false;
  const approval = { format: "wikidata-followup-approval-v1", studyId: FOLLOWUP_STUDY_ID, state: "approved", approved: true,
    owner: "Avery", authority: "Invented private output fixture only", decisionRef: "docs/decisions/0046-bounded-followup-study-gates.md",
    scopeSha256: FOLLOWUP_SCOPE_SHA256, approvedAt: new Date(clock - 60000).toISOString(), expiresAt: new Date(clock + 3600000).toISOString(),
    use: "one-local-feasibility-study-only", publicArtifacts: false, training: false, deployment: false, productCache: false };
  const item = { id: "Q910000101", type: "item", lastrevid: 1,
    labels: { en: { language: "en", value: "Invented output anime" } }, aliases: {}, claims: {
      P4086: [{ type: "statement", rank: "normal", mainsnak: { snaktype: "value", property: "P4086", datatype: "external-id", datavalue: { type: "string", value: "101" } } }],
      P136: [{ type: "statement", rank: "normal", mainsnak: { snaktype: "value", property: "P136", datatype: "wikibase-item", datavalue: { type: "wikibase-entityid", value: { "entity-type": "item", id: "Q920000001" } } } }] } };
  const fake: FollowupGatePorts = { now: () => clock, inspect: async () => ({ repoRoot: "C:\\invented\\Anime", privateParent: "C:\\invented\\Anime-private",
    resolvedPrivateParent: "C:\\invented\\Anime-private", targetPath: `C:\\invented\\Anime-private\\${FOLLOWUP_STUDY_ID}`,
    ancestorsHaveReparsePoints: false, priorPilotExists: false, targetExists: false, reservationExists: consumed, access: "verified-owner-only" }),
    reserveAtomic: async () => { if (consumed) return false; consumed = true; return true; } };
  const ports: FollowupTransportPorts = { ...(gates ?? fake), now: () => clock,
    sleep: async (ms) => { clock += ms; }, deadline: () => () => {},
    fetch: (async (input) => {
      const url = new URL(String(input));
      if (url.hostname === "query.wikidata.org") return new Response(JSON.stringify({ results: { bindings:
        url.searchParams.get("query")!.includes('"101"') ? [{ animeId: { type: "literal", value: "101" }, entity: { type: "uri", value: "http://www.wikidata.org/entity/Q910000101" } }] : [] } }));
      return new Response(JSON.stringify({ entities: url.searchParams.get("props") === "info|labels" ? {
        Q920000001: { id: "Q920000001", type: "item", lastrevid: 2, labels: { en: { language: "en", value: "Invented genre" } } } } : { Q910000101: item } }));
    }) as typeof fetch };
  return { ports, approval, now: () => clock };
}
export async function inventedFollowupResult(gates?: FollowupGatePorts) {
  const fixture = inventedFollowupPorts(gates), result = await runFollowupTransport(fixture.approval, fixture.ports);
  return { ...fixture, result };
}
