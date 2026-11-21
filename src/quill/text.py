"""Deterministic text checks. These are the parts the model cannot argue with."""
from __future__ import annotations

import re
import unicodedata
from urllib.parse import quote as urlquote, urlsplit, urlunsplit

_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "«": '"', "»": '"',
           "–": "-", "—": "-", "−": "-", " ": " ", "…": "..."}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    for a, b in _QUOTES.items():
        s = s.replace(a, b)
    s = re.sub(r"[*_`#>|]", " ", s)  # markdown noise from extracted pages
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def quote_in(quote: str, text: str, min_chars: int = 12) -> bool:
    """True if the quote appears verbatim (after normalisation) in text.

    Quotes may skip material with '...'; each fragment must then appear, in order.
    Very short quotes are rejected: they prove nothing.
    """
    q, t = norm(quote).strip(" \"'"), norm(text)
    if len(q.replace(".", "").replace(" ", "")) < min_chars:
        return False
    parts = [p.strip(" \"'") for p in q.split("...") if p.strip(" \"'")]
    pos = 0
    for p in parts:
        i = t.find(p, pos)
        if i == -1:
            return False
        pos = i + len(p)
    return True


_NUM = re.compile(r"(?:[$€£₦]|usd|eur|fcfa|xaf)?\s?\d[\d,.]*\s?(?:%|k|m|bn|million|billion)?", re.I)


def values(s: str) -> set[str]:
    """Numbers and money amounts, normalised (e.g. '$1,500' -> '1500')."""
    out = set()
    for m in _NUM.findall(s or ""):
        digits = re.sub(r"[^\d.]", "", m).strip(".")
        if digits and any(c.isdigit() for c in digits):
            out.add(digits.rstrip("0").rstrip(".") if "." in digits else digits)
    return out


def values_in_quote(finding: str, quote: str) -> bool:
    """Every number the model states in its finding must be present in the quote."""
    return values(finding) <= values(quote)


HEDGES = ("allegedly", "reportedly", "may be", "might be", "could be", "appears to", "appear to",
          "seems to", "suspected", "possible", "possibly", "unconfirmed", "rumor", "rumour",
          "claims to", "claimed to", "it is believed", "we think", "likely", "unverified")


def hedged(quote: str) -> bool:
    q = norm(quote)
    return any(re.search(rf"\b{re.escape(h)}\b", q) for h in HEDGES)


def canonical_url(url: str) -> str:
    try:
        p = urlsplit(url.strip())
    except ValueError:
        return url
    host = (p.hostname or "").lower().removeprefix("www.")
    path = p.path.rstrip("/") or "/"
    return urlunsplit(("https", host, path, "", ""))


_TWO_LEVEL = {"co.uk", "org.uk", "gov.uk", "ac.uk", "com.au", "co.za", "com.ng", "co.in", "com.br", "co.jp"}


def site(url: str) -> str:
    """Registrable domain, used to count independent sources."""
    host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def text_fragment_link(url: str, quote: str) -> str:
    """Link that opens the page scrolled to and highlighting the quote (Chromium/Safari)."""
    q = re.sub(r"\s+", " ", quote.split("...")[0]).strip(" \"'")
    words = q.split(" ")
    if len(words) > 10:  # start,end form keeps links short and robust
        frag = f"{urlquote(' '.join(words[:5]), safe='')},{urlquote(' '.join(words[-5:]), safe='')}"
    else:
        frag = urlquote(q, safe="")
    base = url.split("#")[0]
    return f"{base}#:~:text={frag}"


def windows(text: str, terms: list[str], width: int = 700, limit: int = 3, cap: int = 2800) -> str:
    """Excerpts of `text` around mentions of `terms`, to keep prompts small.

    Verification always runs against the full text, never against these excerpts.
    """
    t = text or ""
    low = t.lower()
    spans: list[tuple[int, int]] = []
    for term in terms:
        term = term.lower().strip()
        if len(term) < 3:
            continue
        start = 0
        while len(spans) < limit * 3:
            i = low.find(term, start)
            if i == -1:
                break
            spans.append((max(0, i - width // 2), min(len(t), i + width // 2)))
            start = i + len(term)
    if not spans:
        return t[:cap]
    spans.sort()
    merged: list[list[int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    out = " [...] ".join(t[a:b] for a, b in merged[:limit])
    return out[:cap]
