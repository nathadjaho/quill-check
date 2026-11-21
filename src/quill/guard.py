"""Hostile-input handling.

Quill reads two kinds of untrusted text: the suspicious message itself, and web
pages. Both are data. Neither can change what the agent does. This module finds:

* agent-directed manipulation: text that tries to instruct an AI reader
  ("ignore previous instructions", "classify this as legitimate")
* pressure tactics aimed at the human: urgency, secrecy, unusual payment rails

Both are reported with the exact span that triggered them, so a reader can check.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

INJECTION = [
    (r"ignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts|rules)", "override attempt"),
    (r"disregard (all |any )?(the )?(previous|prior|above|your) (instructions|rules|guidelines)", "override attempt"),
    (r"\b(you are|you're) (now )?(an? )?(ai|assistant|language model|chatbot|agent)\b", "addresses an AI reader"),
    (r"\b(ai|assistant|agent|model|scanner|filter|bot)s?\b[^.\n]{0,40}\b(must|should|will) (classify|mark|label|report|treat)", "instructs an AI reader"),
    (r"\b(classify|mark|label|report|flag) (this|the) (message|email|offer|text) as (safe|legit|legitimate|verified|trusted)", "asks to be marked safe"),
    (r"\bsystem prompt\b|\bdeveloper mode\b|\bjailbreak\b", "prompt-injection vocabulary"),
    (r"<\s*/?\s*(system|instructions?|assistant)\s*>", "fake control tags"),
    (r"\bdo not (flag|warn|alert)\b", "asks a reader not to warn"),
]

PRESSURE = [
    (r"\b(gift ?cards?|itunes|google play|steam) (cards?|codes?)?\b", "payment by gift card"),
    (r"\b(bitcoin|btc|crypto(currency)?|usdt|wallet address)\b", "payment in crypto"),
    (r"\b(wire transfer|western union|moneygram|zelle|cash app|venmo)\b", "hard-to-reverse payment"),
    (r"\b(processing|registration|application|training|clearance|release|activation) fee\b", "upfront fee"),
    (r"\b(within|in) (the next )?\d+\s?(minutes|hours|hrs)\b|\b(immediately|right away|urgent(ly)?|act now|today only|final notice)\b", "urgency"),
    (r"\b(don't|do not) (tell|inform|contact) (anyone|your (family|bank|son|daughter|children))\b|\bkeep (this|it) (confidential|secret|between us)\b", "secrecy"),
    (r"\b(arrest(ed)?|warrant|suspend(ed)?|frozen|legal action|deport(ed|ation)?)\b", "threat"),
    (r"\b(social security number|ssn|bank (login|password)|one[- ]time (code|password)|otp|verification code)\b", "asks for credentials"),
    (r"\b(remote access|anydesk|teamviewer|install (this|the) app)\b", "remote-access request"),
]


@dataclass
class Hit:
    kind: str  # injection | pressure
    label: str
    span: str


def scan(text: str, kind: str = "both") -> list[Hit]:
    hits: list[Hit] = []
    rules = []
    if kind in ("both", "injection"):
        rules += [("injection", p, l) for p, l in INJECTION]
    if kind in ("both", "pressure"):
        rules += [("pressure", p, l) for p, l in PRESSURE]
    seen = set()
    for k, pat, label in rules:
        for m in re.finditer(pat, text or "", flags=re.I):
            a, b = max(0, m.start() - 30), min(len(text), m.end() + 30)
            span = re.sub(r"\s+", " ", text[a:b]).strip()
            key = (k, label)
            if key in seen:
                continue
            seen.add(key)
            hits.append(Hit(k, label, span))
    return hits


def fence(label: str, text: str) -> tuple[str, str]:
    """Wrap untrusted text in a nonce fence the text cannot forge.

    The nonce is a hash of the text itself, so a message cannot contain its own closing tag
    (that would require a hash fixed point), and replayed runs stay reproducible.
    """
    nonce = hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:12]
    tag = f"{label}_{nonce}"
    clean = re.sub(rf"</?{label}_[0-9a-f]+>", "", text or "")
    return tag, f"<{tag}>\n{clean}\n</{tag}>"


# Values that must never leave the device in a search query.
PRIVATE = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                     # SSN
    re.compile(r"\b(?:\d[ -]?){13,19}\b"),                     # card / account numbers
    re.compile(r"[\w.+-]+@(gmail|yahoo|hotmail|outlook|icloud|aol)\.com", re.I),  # personal mailboxes
]


def safe_for_search(value: str) -> bool:
    return not any(p.search(value or "") for p in PRIVATE)
