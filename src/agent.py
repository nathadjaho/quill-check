"""Quill Check: research a suspicious message and return sourced, verifiable findings.

Examples
  python src/agent.py --file cases/grandparent_ssa.txt
  python src/agent.py --profile org --file cases/vendor_invoice.txt
  python src/agent.py "Is 'Amazon Prime refund desk' calling customers about refunds legit?"
  python src/agent.py --file cases/grandparent_ssa.txt --record traces/cassettes/ssa.json   # live + save
  python src/agent.py --replay traces/cassettes/ssa.json --file cases/grandparent_ssa.txt  # no keys needed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv  # noqa: E402

from quill.clients import Recorder  # noqa: E402
from quill.pipeline import run  # noqa: E402
from quill.route import markdown, route  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="message or question (or use --file)")
    ap.add_argument("--file", help="path to a text file with the suspicious message")
    ap.add_argument("--profile", choices=["family", "org"], default="family",
                    help="who gets alerted: a trusted family contact, or a security team")
    ap.add_argument("--record", metavar="CASSETTE", help="run live and save every API response")
    ap.add_argument("--replay", metavar="CASSETTE", help="replay saved responses (no API keys needed)")
    ap.add_argument("--out", default=str(ROOT / "traces"), help="where to save the report and trace")
    ap.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    a = ap.parse_args()

    message = Path(a.file).read_text(encoding="utf-8") if a.file else " ".join(a.text)
    if not message.strip():
        ap.error("give a message, a question, or --file")

    if a.replay:
        rec = Recorder.from_env("replay", a.replay)
    elif a.record:
        rec = Recorder.from_env("record", a.record)
    else:
        rec = Recorder.from_env("live")
    if rec.mode != "replay":
        missing = [k for k in ("NEBIUS_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
        if missing:
            ap.error(f"missing {', '.join(missing)} in .env (or use --replay with a saved cassette)")

    report = run(message, rec, a.profile)
    report["routing"] = route(report, ROOT / "outbox")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / report["run_id"]
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(report) + "\n\n" + "\n".join(f"- {n}" for n in report["routing"])
    stem.with_suffix(".md").write_text(md, encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False) if a.json else md)
    print(f"\nSaved {stem}.json and {stem}.md", file=sys.stderr)
    return 0 if report["state"] != "TOOL_FAILED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
