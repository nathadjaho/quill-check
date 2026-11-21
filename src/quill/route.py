"""Rendering and routing of a finished check.

Two audiences, one engine:
  family  - a plain-language alert to a trusted contact (e.g. an older parent's son or daughter)
  org     - a structured incident report for a security / data-protection team

Alerts are only sent when the code-decided state is RED_FLAGS or CAUTION.
By default they are written to ./outbox; SMTP or a webhook is used only when configured.
"""
from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from email.message import EmailMessage
from pathlib import Path

LABELS = {
    "RED_FLAGS": "Red flags found",
    "CAUTION": "Caution",
    "NO_RED_FLAGS_FOUND": "No red flags found (not proof it is genuine)",
    "INSUFFICIENT_EVIDENCE": "Insufficient evidence",
    "TOOL_FAILED": "Check could not run",
}
STANCE = {"supports": "confirms", "contradicts": "contradicts", "warns": "reports as scam"}


def markdown(r: dict) -> str:
    out = [f"# Quill Check: {LABELS.get(r['state'], r['state'])}", "",
           f"**Why:** {r['state_reason']}", "", f"{r['summary']}", ""]
    if r.get("manipulation"):
        out += ["## Text aimed at automated checkers (ignored)", ""]
        out += [f"- {h['label']}: \"{h['span']}\"" for h in r["manipulation"]] + [""]
    if r.get("pressure_tactics"):
        out += ["## Pressure tactics in the message", ""]
        out += [f"- {h['label']}: \"{h['span']}\"" for h in r["pressure_tactics"]] + [""]
    if r.get("claims"):
        out += ["## Claims checked", ""]
    for c in r.get("claims", []):
        v = r["verdicts"].get(c["id"], {"verdict": "NOT_CHECKED", "why": ""})
        out += [f"### {c['id']}. {c['text']}", f"> In the message: \"{c['quote']}\"", "",
                f"**{v['verdict']}**: {v['why']}", ""]
        for f in (x for x in r["findings"] if x["claim_id"] == c["id"]):
            hedge = " _(hedged wording)_" if f["level"] == "hedged" else ""
            out.append(f"- [{f['source_id']}] {f['site']} ({f['source_kind']}) {STANCE[f['stance']]}{hedge}: "
                       f"\"{f['quote']}\" ([open highlighted]({f['link']}))")
        out.append("")
    if r.get("sources"):
        out += ["## Sources", ""]
        out += [f"- [{s['id']}] [{s['title'] or s['site']}]({s['url']})"
                + (f" · {s['published']}" if s.get("published") else "")
                + (" · excluded: contains instructions aimed at AI" if s.get("injection") else "")
                for s in r["sources"]] + [""]
    if r.get("rejected"):
        out += ["## Model proposals rejected by verification", ""]
        out += [f"- {x['claim_id']} / {x['source_id']}: {x['reason']}" for x in r["rejected"]] + [""]
    out += ["## Research trace", ""]
    for e in r["trace"]:
        detail = {k: v for k, v in e.items() if k not in ("t", "step")}
        out.append(f"- `{e['t']:>6}s` **{e['step']}** {json.dumps(detail, ensure_ascii=False)}")
    calls = [u for u in r["usage"]]
    tf = sum(1 for u in calls if u["service"] == "token_factory")
    tv = sum(1 for u in calls if u["service"] == "tavily")
    tok = sum((u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0) for u in calls)
    out += ["", f"_Token Factory calls: {tf} ({tok} tokens, model `{r['model']}`) · Tavily calls: {tv} · "
                f"mode: {r['mode']} · run {r['run_id']}_"]
    return "\n".join(out)


def family_alert(r: dict) -> EmailMessage:
    m = EmailMessage()
    m["Subject"] = f"Quill: {LABELS[r['state']]} in a message your relative received"
    m["To"] = os.getenv("QUILL_FAMILY_EMAIL", "trusted-contact@example.com")
    m["From"] = os.getenv("QUILL_FROM_EMAIL", "quill@example.com")
    lines = [r["summary"], "", "What the message asks for:"]
    lines += [f"- {a['action']}" for a in r.get("asks", [])] or ["- (nothing specific)"]
    lines += ["", "Evidence:"]
    for f in r["findings"][:6]:
        lines.append(f"- {f['site']} {STANCE[f['stance']]}: \"{f['quote']}\"\n  {f['link']}")
    lines += ["", "Suggested next step: call your relative, and check with the organisation using a phone number "
                  "or website you find yourself, not one from the message.",
              "", "This alert was sent because the check found risk. Quill never says a message is safe."]
    m.set_content("\n".join(lines))
    return m


def org_report(r: dict) -> dict:
    return {
        "type": "quill.suspicious_message",
        "run_id": r["run_id"],
        "state": r["state"],
        "reason": r["state_reason"],
        "asks": r.get("asks", []),
        "entities": r.get("entities", []),
        "manipulation": r.get("manipulation", []),
        "pressure_tactics": r.get("pressure_tactics", []),
        "claims": [{**c, **r["verdicts"].get(c["id"], {})} for c in r.get("claims", [])],
        "evidence": [{k: f[k] for k in ("claim_id", "stance", "level", "site", "source_kind", "quote", "url", "link")}
                     for f in r["findings"]],
    }


def route(r: dict, outbox: Path) -> list[str]:
    """Deliver alerts. Returns human-readable notes on what was done."""
    if r["state"] not in ("RED_FLAGS", "CAUTION"):
        return [f"No alert sent (state {r['state']})."]
    outbox.mkdir(parents=True, exist_ok=True)
    notes = []
    if r["profile"] == "family":
        msg = family_alert(r)
        host = os.getenv("SMTP_HOST")
        if host:
            with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587"))) as s:
                s.starttls()
                s.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
                s.send_message(msg)
            notes.append(f"Alert emailed to {msg['To']}.")
        else:
            p = outbox / f"{r['run_id']}-family.eml"
            p.write_bytes(bytes(msg))
            notes.append(f"Alert for {msg['To']} written to {p} (set SMTP_* to send).")
    else:
        body = org_report(r)
        p = outbox / f"{r['run_id']}-security.json"
        p.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        notes.append(f"Incident report written to {p}.")
        hook = os.getenv("QUILL_ORG_WEBHOOK")
        if hook:
            req = urllib.request.Request(hook, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
            notes.append("Incident report posted to the security webhook.")
    return notes
