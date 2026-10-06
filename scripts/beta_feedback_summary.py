"""Validate restricted M10 beta records and write an aggregate-only local report."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_INPUT_BYTES = 128 * 1024
MAX_SESSIONS = 8
CASES = ("sparse-history", "broad-history", "niche-taste", "new-to-anime")
DEVICES = ("desktop", "mobile")
PROMPTS = ("start", "inspect", "filter", "correct", "save-return")
PROMPT_OUTCOMES = ("unassisted", "assisted", "incomplete")
ENGINES = ("graph", "model", "hybrid", "catalog", "explore-community", "explore-genre", "unknown")
FEEDBACK_CODES = (
    "already-seen", "disliked", "confusing", "irrelevant",
    "prerequisite-missing", "metadata-wrong", "other",
)
STOP_CODES = ("crash", "lost-state", "unauthorized-network", "private-data-exposure")
STARTING_PATHS = {
    "sparse-history": "manual",
    "broad-history": "local-import",
    "niche-taste": "manual",
    "new-to-anime": "browse",
}
SESSION_FIELDS = {
    "sessionCode", "consent", "case", "device", "startingPath", "requestedEngine", "actualEngine",
    "prompts", "previewBeforeApply", "explanationPass", "savedUnseen",
    "savedReturn", "firstEligibleShown", "plausibleCount", "appMarkedLeaks",
    "recalledUnrecorded", "feedbackCodes", "stopCodes", "defectRefs",
}


class RecordError(ValueError):
    """A fixed-schema failure; messages never contain record values."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordError(message)


def exact_fields(value: object, fields: set[str], label: str) -> dict:
    require(type(value) is dict and set(value) == fields, f"{label}: fields invalid")
    return value


def bounded_int(value: object, maximum: int, label: str) -> int:
    require(type(value) is int and 0 <= value <= maximum, f"{label}: count invalid")
    return value


def enum(value: object, options: tuple[str, ...], label: str) -> str:
    require(type(value) is str and value in options, f"{label}: code invalid")
    return value


def codes(value: object, options: tuple[str, ...], label: str) -> list[str]:
    require(type(value) is list and len(value) <= len(options), f"{label}: list invalid")
    require(all(type(item) is str and item in options for item in value)
            and len(set(value)) == len(value), f"{label}: code invalid")
    return value


def parse_iso_date(value: object) -> None:
    require(type(value) is str, "consent.date: date invalid")
    try:
        require(date.fromisoformat(value).isoformat() == value, "consent.date: date invalid")
    except ValueError as error:
        raise RecordError("consent.date: date invalid") from error


def validate_session(value: object) -> dict:
    session = exact_fields(value, SESSION_FIELDS, "session")
    require(type(session["sessionCode"]) is str
            and re.fullmatch(r"[A-Z0-9_-]{8,40}", session["sessionCode"]) is not None,
            "sessionCode: token invalid")
    consent = exact_fields(session["consent"], {"notesConsent", "date", "quoteOptIn"}, "consent")
    require(consent["notesConsent"] is True and type(consent["quoteOptIn"]) is bool,
            "consent: notes consent required")
    parse_iso_date(consent["date"])
    case = enum(session["case"], CASES, "case")
    enum(session["device"], DEVICES, "device")
    require(session["startingPath"] == STARTING_PATHS[case], "startingPath: case path invalid")
    enum(session["requestedEngine"], ("graph", "model", "hybrid", "none"), "requestedEngine")
    enum(session["actualEngine"], ENGINES, "actualEngine")
    prompts = exact_fields(session["prompts"], set(PROMPTS), "prompts")
    for name in PROMPTS:
        prompt = exact_fields(prompts[name], {"outcome", "minutes"}, f"prompts.{name}")
        enum(prompt["outcome"], PROMPT_OUTCOMES, f"prompts.{name}.outcome")
        bounded_int(prompt["minutes"], 3, f"prompts.{name}.minutes")
    for name in ("previewBeforeApply", "explanationPass", "savedUnseen"):
        require(type(session[name]) is bool, f"{name}: boolean required")
    require(not session["explanationPass"] or session["actualEngine"] != "unknown",
            "explanationPass: actual engine unknown")
    require(case == "broad-history" or session["previewBeforeApply"] is False,
            "previewBeforeApply: case value invalid")
    enum(session["savedReturn"], ("persisted", "lost", "not-attempted"), "savedReturn")
    shown = bounded_int(session["firstEligibleShown"], 10, "firstEligibleShown")
    bounded_int(session["plausibleCount"], shown, "plausibleCount")
    leaks = bounded_int(session["appMarkedLeaks"], 10, "appMarkedLeaks")
    recalled = bounded_int(session["recalledUnrecorded"], 10, "recalledUnrecorded")
    require(shown + leaks + recalled <= 10,
            "firstEligibleShown: eligible and already-seen counts exceed the first ten")
    codes(session["feedbackCodes"], FEEDBACK_CODES, "feedbackCodes")
    stops = codes(session["stopCodes"], STOP_CODES, "stopCodes")
    refs = session["defectRefs"]
    require(type(refs) is list and len(refs) <= 10
            and all(type(ref) is str and re.fullmatch(r"BETA-[0-9]{3,5}", ref) is not None
                    for ref in refs) and len(set(refs)) == len(refs),
            "defectRefs: references invalid")
    require(session["savedReturn"] != "lost" or "lost-state" in stops,
            "savedReturn: lost state needs a stop code")
    require((not stops and session["appMarkedLeaks"] == 0) or bool(refs),
            "defectRefs: stop or exclusion leak needs a restricted reproduction reference")
    return session


def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON field")
        result[key] = value
    return result


def reject_nonfinite(_: str) -> None:
    raise RecordError("nonfinite JSON number")


def validate_records(data: bytes) -> tuple[dict, list[dict]]:
    require(0 < len(data) <= MAX_INPUT_BYTES, "input size invalid")
    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicate_keys,
                            parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecordError("input JSON invalid") from error
    root = exact_fields(parsed, {"format", "protocolVersion", "buildId", "bundleId", "sessions"}, "root")
    require(root["format"] == "beta-session-records-v1"
            and type(root["protocolVersion"]) is int and root["protocolVersion"] == 1,
            "root: format or protocol invalid")
    for name in ("buildId", "bundleId"):
        require(type(root[name]) is str
                and re.fullmatch(r"[A-Za-z0-9._-]{1,80}", root[name]) is not None,
                f"{name}: token invalid")
    values = root["sessions"]
    require(type(values) is list and 1 <= len(values) <= MAX_SESSIONS, "sessions: count invalid")
    sessions = [validate_session(value) for value in values]
    require(len({item["sessionCode"] for item in sessions}) == len(sessions),
            "sessionCode: duplicate")
    return root, sessions


def core_journey_pass(session: dict) -> bool:
    return (all(session["prompts"][name]["outcome"] == "unassisted"
                for name in ("start", "inspect", "filter", "save-return"))
            and session["savedUnseen"] and session["savedReturn"] == "persisted"
            and (session["case"] != "broad-history" or session["previewBeforeApply"]))


def summarize(root: dict, sessions: list[dict]) -> dict:
    case_counts = Counter(item["case"] for item in sessions)
    device_counts = Counter(item["device"] for item in sessions)
    case_device_counts = Counter((item["case"], item["device"]) for item in sessions)
    core_pass = Counter(item["case"] for item in sessions if core_journey_pass(item))
    core_pass_device = Counter(item["device"] for item in sessions if core_journey_pass(item))
    explanation_pass = Counter(item["case"] for item in sessions if item["explanationPass"])
    assessable = Counter(item["case"] for item in sessions if item["firstEligibleShown"] == 10)
    plausible_pass = Counter(item["case"] for item in sessions
                             if item["firstEligibleShown"] == 10 and item["plausibleCount"] >= 3)
    requested_engines = Counter(item["requestedEngine"] for item in sessions)
    actual_engines = Counter(item["actualEngine"] for item in sessions)
    feedback = Counter(code for item in sessions for code in item["feedbackCodes"])
    stops = Counter(code for item in sessions for code in item["stopCodes"])
    prompt_assisted = Counter(name for item in sessions for name in PROMPTS
                              if item["prompts"][name]["outcome"] == "assisted")
    prompt_incomplete = Counter(name for item in sessions for name in PROMPTS
                                if item["prompts"][name]["outcome"] == "incomplete")
    return {
        "format": "beta-aggregate-v1",
        "protocolVersion": root["protocolVersion"],
        "buildId": root["buildId"],
        "bundleId": root["bundleId"],
        "sessionCount": len(sessions),
        "caseCounts": {name: case_counts[name] for name in CASES},
        "deviceCounts": {name: device_counts[name] for name in DEVICES},
        "caseDeviceCounts": {case: {device: case_device_counts[case, device] for device in DEVICES}
                             for case in CASES},
        "requestedEngineCounts": {name: requested_engines[name] for name in
                                  ("graph", "model", "hybrid", "none")},
        "actualEngineCounts": {name: actual_engines[name] for name in ENGINES},
        "coreJourneyPassCount": sum(core_pass.values()),
        "coreJourneyPassByCase": {name: core_pass[name] for name in CASES},
        "coreJourneyPassByDevice": {name: core_pass_device[name] for name in DEVICES},
        "explanationPassCount": sum(explanation_pass.values()),
        "explanationPassByCase": {name: explanation_pass[name] for name in CASES},
        "assessableTenCount": sum(assessable.values()),
        "assessableTenByCase": {name: assessable[name] for name in CASES},
        "plausibleThreeOfTenCount": sum(plausible_pass.values()),
        "plausibleThreeOfTenByCase": {name: plausible_pass[name] for name in CASES},
        "appMarkedLeakCount": sum(item["appMarkedLeaks"] for item in sessions),
        "recalledUnrecordedCount": sum(item["recalledUnrecorded"] for item in sessions),
        "savedReturnPersistedCount": sum(item["savedReturn"] == "persisted" for item in sessions),
        "feedbackCounts": {name: feedback[name] for name in FEEDBACK_CODES},
        "stopCounts": {name: stops[name] for name in STOP_CODES},
        "assistedPromptCounts": {name: prompt_assisted[name] for name in PROMPTS},
        "incompletePromptCounts": {name: prompt_incomplete[name] for name in PROMPTS},
        "restrictedDefectReferenceCount": sum(len(item["defectRefs"]) for item in sessions),
        "releaseDecision": "not-made; technical gates, human evidence, and owner review are separate",
    }


def outside_checkout(path: Path) -> Path:
    resolved = path.resolve()
    require(not resolved.is_relative_to(REPO_ROOT), "private records and reports must stay outside the checkout")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        source = outside_checkout(args.input)
        target = outside_checkout(args.output)
        require(source.is_file() and 0 < source.stat().st_size <= MAX_INPUT_BYTES,
                "input size invalid")
        require(target.parent.is_dir(), "output directory missing")
        root, sessions = validate_records(source.read_bytes())
        report = (json.dumps(summarize(root, sessions), sort_keys=True, indent=2) + "\n").encode("utf-8")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(report)
    except (RecordError, OSError) as error:
        message = str(error) if isinstance(error, RecordError) else "local file operation failed"
        print(f"Beta aggregate blocked: {message}", file=sys.stderr)
        return 1
    print(f"Beta aggregate written ({len(sessions)} sessions; no per-session fields).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
