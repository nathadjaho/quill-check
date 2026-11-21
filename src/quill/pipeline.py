"""Quill Check pipeline.

message -> guard scan -> intake (claims, entities; quotes verified against the message)
        -> plan queries -> Tavily search -> select & dedupe evidence
        -> assess (model proposes findings) -> verify (code checks quote, values, hedges, source)
        -> refine once if claims have no evidence -> verdicts and overall state decided by code
        -> explanation (model, guarded) -> report + routing
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from . import prompts
from .clients import LLM, Recorder, Search, ToolFailure
from .guard import fence, safe_for_search, scan
from .text import (canonical_url, hedged, norm, quote_in, site, text_fragment_link, values_in_quote,
                   windows)

MAX_QUERIES = 10          # hard budget across both rounds
RESULTS_PER_QUERY = 5
MAX_SOURCES_PER_CLAIM = 5
MAX_CLAIMS = 4           # keeps runs under ~2 minutes with reasoning models
PAGES_PER_ASSESS = 4
MAX_PER_SITE = 2

CONSUMER_PROTECTION = {
    "ftc.gov", "consumer.ftc.gov", "ic3.gov", "fbi.gov", "ssa.gov", "oig.ssa.gov", "irs.gov", "usa.gov",
    "aarp.org", "bbb.org", "consumerfinance.gov", "cisa.gov", "actionfraud.police.uk", "ncsc.gov.uk",
    "scamwatch.gov.au", "antifraudcentre-centreantifraude.ca", "cybermalveillance.gouv.fr",
}
FORUMS = {"reddit.com", "quora.com", "facebook.com", "x.com", "twitter.com", "tiktok.com", "youtube.com",
          "medium.com", "instagram.com", "linkedin.com", "threads.net", "glassdoor.com", "indeed.com", "trustpilot.com", "scamadviser.com", "complaintsboard.com"}


@dataclass
class Source:
    id: str
    url: str
    title: str
    site: str
    kind: str
    published: str | None
    raw: str = field(repr=False, default="")
    injection: list[dict] = field(default_factory=list)


@dataclass
class Finding:
    claim_id: str
    source_id: str
    stance: str
    quote: str
    finding: str
    level: str          # stated | hedged
    source_kind: str
    site: str
    url: str
    link: str           # opens the page highlighting the quote


@dataclass
class Rejected:
    claim_id: str
    source_id: str
    reason: str
    quote: str


class Trace:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.events: list[dict] = []

    def add(self, step: str, **detail: Any) -> None:
        self.events.append({"t": round(time.time() - self.t0, 2), "step": step, **detail})


def source_kind(s: str, entities: list[str]) -> str:
    if s.endswith(".gov") or ".gov." in s or s in CONSUMER_PROTECTION:
        return "government / consumer protection"
    if s in FORUMS:
        return "forum / user reports"
    ent = [norm(e).replace(" ", "") for e in entities]
    root = s.split(".")[0]
    if any(root and (root in e or e in root) for e in ent if len(e) >= 4):
        return "organisation's own site"
    return "web"


def run(message: str, rec: Recorder, profile: str = "family") -> dict:
    trace = Trace()
    usage: list[dict] = []
    llm, web = LLM(rec, usage), Search(rec, usage)
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]

    report: dict[str, Any] = {"run_id": run_id, "profile": profile, "mode": rec.mode,
                              "model": llm.model, "message": message}

    # 1. Deterministic scan of the message itself.
    hits = scan(message)
    report["manipulation"] = [asdict(h) for h in hits if h.kind == "injection"]
    report["pressure_tactics"] = [asdict(h) for h in hits if h.kind == "pressure"]
    trace.add("guard", injection=len(report["manipulation"]), pressure=len(report["pressure_tactics"]))

    try:
        # 2. Intake: model proposes claims; code keeps only those quoted verbatim.
        tag, fenced = fence("message", message)
        raw = llm.json(prompts.SYSTEM_BASE, prompts.INTAKE.format(fenced=fenced), step="intake")
        claims, dropped = [], []
        for i, c in enumerate(raw.get("claims", [])[:MAX_CLAIMS]):
            c["id"] = f"c{i + 1}"
            if not quote_in(c.get("quote", ""), message, min_chars=6):
                dropped.append({"claim": c.get("text"), "reason": "quote not found in message"})
                continue
            c["entities"] = [e for e in c.get("entities", []) if norm(e) in norm(message) and safe_for_search(e)]
            claims.append(c)
        entities = [e for e in raw.get("entities", [])
                    if norm(e.get("name", "")) in norm(message) and safe_for_search(e.get("name", ""))]
        asks = [a for a in raw.get("asks", []) if quote_in(a.get("quote", ""), message, min_chars=6)]
        report.update(claims=claims, entities=entities, asks=asks, dropped_claims=dropped)
        trace.add("intake", claims=len(claims), entities=[e["name"] for e in entities], dropped=len(dropped))

        if not claims:
            return finish(report, [], [], {}, "INSUFFICIENT_EVIDENCE",
                          "No checkable claim could be extracted from the message.", trace, usage, llm)

        # 3. Plan queries (model) + deterministic queries per entity.
        plan = llm.json(prompts.SYSTEM_BASE,
                        prompts.PLAN.format(claims=json.dumps(
                            [{"id": c["id"], "text": c["text"], "entities": c["entities"]} for c in claims])),
                        fast=True, step="plan")
        queries: list[tuple[str, str]] = []
        for q in plan.get("queries", []):
            if q.get("claim_id") in {c["id"] for c in claims} and safe_for_search(q.get("query", "")):
                queries.append((q["claim_id"], q["query"].strip()[:120]))
        for e in entities[:3]:
            linked = next((c["id"] for c in claims if e["name"] in c["entities"]), claims[0]["id"])
            queries.append((linked, f"\"{e['name']}\" scam"))
        queries = dedupe_queries(queries)[:MAX_QUERIES - 3]  # keep budget for a refine round
        trace.add("plan", queries=[q for _, q in queries])

        sources: dict[str, Source] = {}
        by_claim: dict[str, list[str]] = {c["id"]: [] for c in claims}
        tried: list[str] = []
        search_errors: list[str] = []

        def do_search(qs: list[tuple[str, str]], round_no: int) -> None:
            for cid, q in qs:
                tried.append(q)
                try:
                    rows = web.search(q, RESULTS_PER_QUERY)
                except ToolFailure as e:
                    search_errors.append(e.detail)
                    trace.add("search_error", query=q, error=e.detail, round=round_no)
                    continue
                added = 0
                for r in rows:
                    cu = canonical_url(r["url"])
                    if not r.get("raw") and not r.get("snippet"):
                        continue
                    existing = next((s for s in sources.values() if canonical_url(s.url) == cu), None)
                    if existing is None:
                        sid = f"S{len(sources) + 1}"
                        st = site(r["url"])
                        ents = [x["name"] for x in entities]
                        existing = Source(sid, r["url"], r.get("title", ""), st, source_kind(st, ents),
                                          r.get("published"), r.get("raw") or r.get("snippet", ""))
                        existing.injection = [asdict(h) for h in scan(existing.raw[:20000], "injection")]
                        sources[sid] = existing
                    ids = by_claim[cid]
                    same_site = sum(1 for x in ids if sources[x].site == existing.site)
                    if existing.id not in ids and len(ids) < MAX_SOURCES_PER_CLAIM and same_site < MAX_PER_SITE:
                        ids.append(existing.id)
                        added += 1
                trace.add("search", round=round_no, query=q, claim=cid, results=len(rows), kept=added)

        do_search(queries, 1)
        if not sources and search_errors:
            return finish(report, [], [], {}, "TOOL_FAILED",
                          "Web search failed; no answer is given without evidence. " + search_errors[0],
                          trace, usage, llm)

        findings: list[Finding] = []
        rejected: list[Rejected] = []
        assess_errors: list[str] = []
        claims_by_id = {c["id"]: c for c in claims}

        def assess(cids: list[str]) -> None:
            for cid in cids:
                c = claims_by_id[cid]
                usable = [sources[s] for s in by_claim[cid] if not sources[s].injection][:PAGES_PER_ASSESS]
                for s in (sources[x] for x in by_claim[cid] if sources[x].injection):
                    rejected.append(Rejected(cid, s.id, "page contains instructions aimed at AI readers; excluded", ""))
                if not usable:
                    continue
                terms = c["entities"] + [w for w in c["text"].split() if len(w) > 6][:4]
                blocks = []
                for s in usable:
                    tag, f = fence("page", windows(s.raw, terms))
                    blocks.append(f"Page {s.id} | {s.site} | {s.title}\n{f}")
                try:
                    out = llm.json(prompts.SYSTEM_BASE, prompts.ASSESS.format(
                        cid=cid, claim=c["text"], quote=c["quote"], pages="\n\n".join(blocks)), step=f"assess:{cid}")
                except ToolFailure as e:
                    # one failed assessment must not sink the run: the claim stays unverified, and says why
                    assess_errors.append(f"{cid}: {e.detail}")
                    trace.add("assess_failed", claim=cid, error=e.detail)
                    continue
                for f in out.get("findings", []):
                    sid, q, stance = f.get("source", ""), f.get("quote", ""), f.get("stance", "")
                    src = sources.get(sid)
                    if src is None or src.id not in by_claim[cid]:
                        rejected.append(Rejected(cid, sid, "cites a page that was not provided", q)); continue
                    if stance not in ("supports", "contradicts", "warns"):
                        rejected.append(Rejected(cid, sid, f"unknown stance '{stance}'", q)); continue
                    if not quote_in(q, src.raw):
                        rejected.append(Rejected(cid, sid, "quote not found on the page", q)); continue
                    if not values_in_quote(f.get("finding", ""), q):
                        rejected.append(Rejected(cid, sid, "finding states numbers absent from the quote", q)); continue
                    findings.append(Finding(cid, sid, stance, q.strip(), f.get("finding", "").strip(),
                                            "hedged" if hedged(q) else "stated", src.kind, src.site, src.url,
                                            text_fragment_link(src.url, q)))
                trace.add("assess", claim=cid, pages=len(usable),
                          proposed=len(out.get("findings", [])),
                          verified=sum(1 for x in findings if x.claim_id == cid))

        assess(list(by_claim))

        # 4. Refine once for claims with no verified evidence, within budget.
        empty = [cid for cid in by_claim if not any(f.claim_id == cid for f in findings)]
        budget = MAX_QUERIES - len(tried)
        if empty and budget > 0:
            ref = llm.json(prompts.SYSTEM_BASE, prompts.REFINE.format(
                claims=json.dumps([{"id": c, "text": claims_by_id[c]["text"]} for c in empty]),
                tried=json.dumps(tried)), fast=True, step="refine")
            more = [(q["claim_id"], q["query"]) for q in ref.get("queries", [])
                    if q.get("claim_id") in empty and q.get("query") and safe_for_search(q["query"])
                    and q["query"] not in tried][:budget]
            if more:
                trace.add("refine", reason=f"{len(empty)} claim(s) without verified evidence", queries=[q for _, q in more])
                before = {k: list(v) for k, v in by_claim.items()}
                do_search(more, 2)
                grown = [cid for cid in empty if len(by_claim[cid]) > len(before[cid])]
                assess(grown)
            stop = "refined once; budget or rounds exhausted" if more else "refine produced no new queries"
        else:
            stop = "every claim has verified evidence" if not empty else "query budget exhausted"
        trace.add("stop", reason=stop)

        verdicts = {c["id"]: decide_claim([f for f in findings if f.claim_id == c["id"]]) for c in claims}
        state, why = decide_overall(verdicts, report["pressure_tactics"], report["manipulation"],
                                    search_errors + assess_errors)
        report["sources"] = [{k: v for k, v in asdict(s).items() if k != "raw"} for s in sources.values()]
        report["search_errors"] = search_errors
        report["assess_errors"] = assess_errors
        for cid in verdicts:
            if verdicts[cid]["verdict"] == "UNVERIFIED" and any(x.startswith(cid + ":") for x in assess_errors):
                verdicts[cid]["why"] = "the model could not assess the pages for this claim (see assess_errors)"
        return finish(report, findings, rejected, verdicts, state, why, trace, usage, llm)

    except ToolFailure as e:
        trace.add("tool_failure", tool=e.tool, error=e.detail)
        return finish(report, [], [], {}, "TOOL_FAILED",
                      f"{e.tool} failed ({e.detail}). Quill does not answer without evidence.",
                      trace, usage, llm)


def dedupe_queries(qs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen, out = set(), []
    for cid, q in qs:
        k = norm(q)
        if k and k not in seen:
            seen.add(k)
            out.append((cid, q))
    return out


def decide_claim(fs: list[Finding]) -> dict:
    """First-match decision table. Code, not the model, decides.

    Forums and social media can make a verdict weak, never strong: only official sources,
    or two independent non-forum sites, produce a strong verdict.
    """
    def sites(stance: str, strong_only: bool = False) -> set[str]:
        return {f.site for f in fs if f.stance == stance and f.level == "stated"
                and (not strong_only or f.source_kind != "forum / user reports")}

    official = ("government / consumer protection", "organisation's own site")
    official_con = any(f.stance == "contradicts" and f.level == "stated" and f.source_kind in official for f in fs)
    official_warn = any(f.stance == "warns" and f.level == "stated" and f.source_kind in official for f in fs)
    con, warn, sup = sites("contradicts"), sites("warns"), sites("supports")
    con_s, warn_s, sup_s = sites("contradicts", True), sites("warns", True), sites("supports", True)
    if (con_s or warn_s) and sup_s:
        v, why = "CONFLICTING", "independent sources disagree; both sides are shown"
    elif official_con or len(con_s) >= 2:
        v, why = "CONTRADICTED", "an official source, or two independent non-forum sites, state the opposite"
    elif official_warn or len(warn_s) >= 2:
        v, why = "REPORTED_AS_SCAM", "an official source, or two independent non-forum sites, report it as a scam"
    elif con:
        v, why = "CONTRADICTED_WEAK", "only one non-official site or forum posts state the opposite"
    elif warn:
        v, why = "REPORTED_AS_SCAM_WEAK", "only one non-official site or forum posts report it as a scam"
    elif len(sup_s) >= 2 or (sup_s and any(f.stance == "supports" and f.source_kind in official for f in fs)):
        v, why = "SUPPORTED", "an official source, or two independent non-forum sites, confirm it"
    elif sup:
        v, why = "SUPPORTED_WEAK", "only one site or forum posts confirm it"
    elif fs:
        v, why = "UNVERIFIED", "only hedged statements were found"
    else:
        v, why = "UNVERIFIED", "no page addressed this claim"
    return {"verdict": v, "why": why, "sources": sorted({f.site for f in fs})}


NEGATIVE = {"CONTRADICTED", "REPORTED_AS_SCAM"}
WEAK_NEGATIVE = {"CONTRADICTED_WEAK", "REPORTED_AS_SCAM_WEAK", "CONFLICTING"}


def decide_overall(verdicts: dict, pressure: list, manipulation: list, errors: list) -> tuple[str, str]:
    vs = [v["verdict"] for v in verdicts.values()]
    if any(v in NEGATIVE for v in vs):
        return "RED_FLAGS", "At least one claim in the message is contradicted or reported as a scam by sources."
    if manipulation:
        return "RED_FLAGS", "The message contains text written to manipulate automated checks."
    if any(v in WEAK_NEGATIVE for v in vs) or len({p["label"] for p in pressure}) >= 2:
        return "CAUTION", "Some evidence or pressure tactics point to risk, but it is not conclusive."
    if any(v.startswith("SUPPORTED") for v in vs):
        return ("NO_RED_FLAGS_FOUND",
                "Some claims are confirmed and none are contradicted. This is not proof the message is genuine.")
    if errors:
        return "INSUFFICIENT_EVIDENCE", "Search partly failed and what was found does not settle any claim."
    return "INSUFFICIENT_EVIDENCE", "The web does not say enough about these claims to conclude either way."


def finish(report: dict, findings: list[Finding], rejected: list[Rejected], verdicts: dict, state: str,
           why: str, trace: Trace, usage: list, llm: LLM) -> dict:
    report["state"] = state
    report["state_reason"] = why
    report["verdicts"] = verdicts
    report["findings"] = [asdict(f) for f in findings]
    report["rejected"] = [asdict(r) for r in rejected]
    report.setdefault("sources", [])
    report["summary"] = explain(report, llm, trace) if findings else why
    trace.add("done", state=state)
    report["trace"] = trace.events
    report["usage"] = usage
    return report


def explain(report: dict, llm: LLM, trace: Trace) -> str:
    """Plain-language summary written by the model, kept only if it adds no new numbers or sites."""
    slim = [{"claim": f["claim_id"], "stance": f["stance"], "site": f["site"], "finding": f["finding"],
             "quote": f["quote"]} for f in report["findings"]]
    try:
        out = llm.json(prompts.SYSTEM_BASE, prompts.EXPLAIN.format(state=report["state"], findings=json.dumps(slim)),
                       fast=True, step="explain")
    except ToolFailure:
        return report["state_reason"]
    text = (out.get("summary") or "").strip()
    evidence = " ".join(f["quote"] + " " + f["finding"] for f in report["findings"])
    if not text or not values_in_quote(text, evidence):
        trace.add("explain_dropped", reason="summary introduced numbers not present in verified findings")
        return report["state_reason"]
    return text
