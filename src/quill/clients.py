"""Thin, failure-aware wrappers around Nebius Token Factory and Tavily.

Every external call goes through `Recorder`, which can:
  * live    - call the real API
  * record  - call the real API and save the response to a cassette file
  * replay  - return saved responses, never touching the network
Replay lets judges reproduce a real run without API keys. Cassettes only
contain API responses, never keys.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


class ToolFailure(Exception):
    """A required tool (search or inference) failed after retries."""

    def __init__(self, tool: str, detail: str):
        super().__init__(f"{tool}: {detail}")
        self.tool = tool
        self.detail = detail


@dataclass
class Recorder:
    mode: str = "live"  # live | record | replay
    path: Path | None = None
    calls: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, mode: str | None = None, path: str | None = None) -> "Recorder":
        mode = mode or os.getenv("QUILL_MODE", "live")
        p = Path(path) if path else None
        rec = cls(mode=mode, path=p)
        if mode == "replay":
            if not p or not p.exists():
                raise SystemExit(f"Replay cassette not found: {p}")
            rec.calls = json.loads(p.read_text(encoding="utf-8"))
        return rec

    @staticmethod
    def key(kind: str, payload: Any) -> str:
        raw = json.dumps([kind, payload], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def run(self, kind: str, payload: Any, fn: Callable[[], Any]) -> Any:
        k = self.key(kind, payload)
        if self.mode == "replay":
            if k in self.calls:
                return self.calls[k]["response"]
            # cassettes recorded before the key change included the model name: match on content only
            want = {x: y for x, y in payload.items() if x not in ("model", "fast")} if isinstance(payload, dict) else payload
            for v in self.calls.values():
                p = v.get("payload")
                if v.get("kind") == kind and isinstance(p, dict) and isinstance(want, dict) and \
                        {x: y for x, y in p.items() if x not in ("model", "fast")} == want:
                    return v["response"]
            raise ToolFailure(kind, "no recorded response for this call (replay mode)")
        out = fn()
        if self.mode == "record":
            self.calls[k] = {"kind": kind, "payload": payload, "response": out}
            self.save()
        return out

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.calls, indent=1, ensure_ascii=False), encoding="utf-8")


def _retry(fn: Callable[[], Any], tool: str, attempts: int = 3, fatal: tuple[str, ...] = ()) -> Any:
    last = ""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - we classify below
            last = f"{type(e).__name__}: {e}"
            if any(f in last.lower() for f in fatal):
                break  # auth / plan errors: retrying will not help
            time.sleep(min(2 ** i, 6))
    raise ToolFailure(tool, last[:300])


FATAL = ("401", "403", "unauthorized", "invalid api key", "forbidden", "432", "433")


class LLM:
    """Nebius Token Factory via its OpenAI-compatible API."""

    def __init__(self, rec: Recorder, usage: list[dict]):
        self.rec = rec
        self.usage = usage
        self.model = os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct")
        self.model_fast = os.getenv("NEBIUS_MODEL_FAST") or self.model
        self.fallback = os.getenv("NEBIUS_MODEL_FALLBACK") or None
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
                api_key=os.environ["NEBIUS_API_KEY"],
                timeout=180,
            )
        return self._client

    def json(self, system: str, user: str, fast: bool = False, step: str = "") -> dict:
        """Ask for a JSON object. Returns {} only through ToolFailure, never silently."""
        model = self.model_fast if fast else self.model
        # the model name is not part of the cassette key, so a judge can replay a run
        # with the default .env; the model actually used is stored in the response
        payload = {"system": system, "user": user, "fast": fast}

        def create(**kw):
            """Streamed request: tokens keep the connection alive, so long generations
            are not cut by idle timeouts on the way."""
            stream = self.client.chat.completions.create(
                **kw, stream=True, stream_options={"include_usage": True})
            parts, usage = [], None
            for chunk in stream:
                if getattr(chunk, "usage", None):
                    usage = chunk.usage
                for ch in chunk.choices or []:
                    delta = getattr(ch, "delta", None)
                    if delta is not None and getattr(delta, "content", None):
                        parts.append(delta.content)
            return "".join(parts), usage

        def call_model(m: str) -> dict:
            t0 = time.time()
            kw = dict(model=m, temperature=0, max_tokens=int(os.getenv("QUILL_MAX_TOKENS", "16000")),
                      response_format={"type": "json_object"},
                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            if os.getenv("QUILL_NO_THINK", "1") == "1":
                try:  # ask hybrid reasoning models to answer directly; ignored or refused by others
                    content, u = create(**kw, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                except Exception as e:  # noqa: BLE001
                    if "400" not in str(e) and "422" not in str(e):
                        raise
                    content, u = create(**kw)
            else:
                content, u = create(**kw)
            return {
                "model": m,
                "content": content,
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "seconds": round(time.time() - t0, 2),
            }

        def usable(out: dict) -> bool:
            try:
                parse_json(out["content"])
                return True
            except ToolFailure:
                return False

        def live() -> dict:
            err = "no usable JSON"
            try:
                out = _retry(lambda: call_model(model), "token_factory", fatal=FATAL)
                if usable(out):
                    return out
            except ToolFailure as e:
                err = e.detail
            # empty or broken answer (e.g. reasoning ran out of tokens): try the fallback model once
            if self.fallback and self.fallback != model:
                out = _retry(lambda: call_model(self.fallback), "token_factory", fatal=FATAL)
                if usable(out):
                    return out
            raise ToolFailure("token_factory", f"{model}: {err}; fallback did not return usable JSON either")

        out = self.rec.run("llm", payload, live)
        self.usage.append({"step": step, "service": "token_factory", "model": out["model"],
                           "prompt_tokens": out.get("prompt_tokens"),
                           "completion_tokens": out.get("completion_tokens"),
                           "seconds": out.get("seconds")})
        return parse_json(out["content"])


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1:
            raise ToolFailure("token_factory", "model did not return JSON")
        try:
            obj = json.loads(text[s:e + 1])
        except json.JSONDecodeError as err:
            raise ToolFailure("token_factory", f"unparseable JSON: {err}") from err
    if not isinstance(obj, dict):
        raise ToolFailure("token_factory", "JSON was not an object")
    return obj


class Search:
    """Tavily search + extract. Page text is returned as untrusted data."""

    def __init__(self, rec: Recorder, usage: list[dict]):
        self.rec = rec
        self.usage = usage
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        return self._client

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        payload = {"query": query, "max_results": max_results}

        def live() -> list[dict]:
            t0 = time.time()
            r = _retry(lambda: self.client.search(
                query, max_results=max_results, search_depth="advanced",
                include_raw_content=True), "tavily", fatal=FATAL)
            rows = []
            for x in r.get("results", []):
                rows.append({
                    "url": x.get("url", ""),
                    "title": x.get("title", ""),
                    "snippet": x.get("content", "") or "",
                    "raw": x.get("raw_content") or "",
                    "published": x.get("published_date"),
                    "score": x.get("score"),
                })
            self.usage.append({"step": "search", "service": "tavily", "query": query,
                               "results": len(rows), "seconds": round(time.time() - t0, 2)})
            return rows

        return self.rec.run("search", payload, live)

    def extract(self, urls: list[str]) -> dict[str, str]:
        if not urls:
            return {}
        payload = {"urls": sorted(urls)}

        def live() -> dict[str, str]:
            r = _retry(lambda: self.client.extract(urls=urls), "tavily", fatal=FATAL)
            self.usage.append({"step": "extract", "service": "tavily", "urls": len(urls)})
            return {x.get("url", ""): x.get("raw_content") or "" for x in r.get("results", [])}

        return self.rec.run("extract", payload, live)
