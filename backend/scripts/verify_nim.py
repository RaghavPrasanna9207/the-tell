"""Phase 0 decision gate: does NVIDIA NIM's HOSTED endpoint actually honor
schema-constrained decoding for THIS project's real `Analysis` schema?

Everything in the NIM migration rests on this. NVIDIA documents
`nvext.guided_json` (xgrammar backend) and recommends it over
`response_format`, but that documentation covers self-hosted NIM containers —
whether integrate.api.nvidia.com honors it is not something the docs settle,
and this project's schema is an awkward one for strict implementations:
`Analysis` has NO top-level `required` array (because `detections` uses
default_factory=list) and it nests via `$defs`/`$ref`.

So: measure, don't assume. This script runs the REAL Layer-1 system prompt
(app.classify._build_system_prompt) against real gold messages and reports
whether the output is trustworthy enough to keep `app/llm.py`'s "malformed
output is IMPOSSIBLE" guarantee on the hosted path.

Reports four things, per mode:
  1. parse rate    - does the response validate into `Analysis`?
  2. clamp rate    - confidence outside [0,1]? (the bug _clamp_range_violations
                     exists for; constrained decoding enforces SHAPE, not RANGE,
                     so this is expected to recur on NIM too)
  3. verbatim rate - are returned spans actual substrings of the input? This is
                     Layer 1's core guarantee (see app/classify.py) and is NOT
                     something any decoder enforces.
  4. latency/msg

Usage:
    export NIM_API_KEY=nvapi-...        # from https://build.nvidia.com
    python scripts/verify_nim.py
    python scripts/verify_nim.py --mode guided_json --n 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from pydantic import ValidationError

from app.classify import _build_system_prompt
from app.taxonomy import Analysis
from eval.gold import load_real_gold

NIM_BASE_URL = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"


def build_body(message: str, model: str, mode: str) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _build_system_prompt()},
            {"role": "user", "content": f"Message:\n{message}"},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    if mode == "guided_json":
        # NVIDIA's documented extension. Sent top-level - which is where the
        # OpenAI SDK's extra_body contents would land too.
        body["nvext"] = {"guided_json": Analysis.model_json_schema()}
    elif mode == "json_object":
        body["response_format"] = {"type": "json_object"}
    return body


def run_mode(records: list[dict], model: str, mode: str, api_key: str) -> dict:
    print(f"\n{'=' * 72}\nMODE: {mode}   MODEL: {model}\n{'=' * 72}")
    n_ok = n_parse_fail = n_http_fail = n_clamp = 0
    n_spans = n_verbatim = 0
    total_elapsed = 0.0
    failures: list[str] = []

    with httpx.Client(timeout=120.0) as client:
        for i, rec in enumerate(records, 1):
            start = time.monotonic()
            try:
                resp = client.post(
                    f"{NIM_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=build_body(rec["text"], model, mode),
                )
            except Exception as exc:
                n_http_fail += 1
                failures.append(f"{rec['id']}: transport error {exc}")
                continue
            total_elapsed += time.monotonic() - start

            if resp.status_code != 200:
                n_http_fail += 1
                detail = resp.text[:300].replace("\n", " ")
                failures.append(f"{rec['id']}: HTTP {resp.status_code} {detail}")
                # A 400 here usually means the schema itself was rejected - that
                # IS the answer to this spike, so surface it loudly and early.
                if resp.status_code == 400 and n_http_fail == 1:
                    print(f"  !! HTTP 400 on first call - likely schema rejection:\n     {detail}")
                continue

            content = resp.json()["choices"][0]["message"]["content"]
            try:
                analysis = Analysis.model_validate_json(content)
            except ValidationError as exc:
                errs = exc.errors()
                if errs and all(e["type"] in ("greater_than_equal", "less_than_equal") for e in errs):
                    # Shape was right, only the numeric range was wrong - this is
                    # the known clamp case, not a decoding failure.
                    n_clamp += 1
                    n_ok += 1
                    print(f"  [{i:>3}] {rec['id']:<12} parsed (confidence out of range, clampable)")
                    continue
                n_parse_fail += 1
                failures.append(f"{rec['id']}: {errs[:2]} | raw={content[:200]}")
                print(f"  [{i:>3}] {rec['id']:<12} PARSE FAIL")
                continue
            except json.JSONDecodeError as exc:
                n_parse_fail += 1
                failures.append(f"{rec['id']}: not JSON ({exc}) | raw={content[:200]}")
                print(f"  [{i:>3}] {rec['id']:<12} NOT JSON")
                continue

            n_ok += 1
            for d in analysis.detections:
                n_spans += 1
                if d.span in rec["text"]:
                    n_verbatim += 1
            print(f"  [{i:>3}] {rec['id']:<12} ok  ({len(analysis.detections)} detections)")

    n = len(records)
    print(f"\n  parse rate     : {n_ok}/{n} ({n_ok / n:.0%})   [http failures: {n_http_fail}]")
    print(f"  clamped        : {n_clamp} (confidence outside [0,1] - shape ok, range not)")
    if n_spans:
        print(f"  verbatim spans : {n_verbatim}/{n_spans} ({n_verbatim / n_spans:.0%}) - Layer 1's core guarantee")
    else:
        print("  verbatim spans : no detections returned at all (suspicious - check the prompt)")
    if n_ok:
        print(f"  latency        : {total_elapsed / max(n_ok, 1):.2f}s/msg")
    if failures:
        print(f"\n  first {min(5, len(failures))} failures:")
        for f in failures[:5]:
            print(f"    - {f}")
    return {"mode": mode, "n": n, "ok": n_ok, "parse_fail": n_parse_fail, "http_fail": n_http_fail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="how many gold messages to test")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--mode", choices=["guided_json", "json_object", "both"], default="both")
    args = ap.parse_args()

    api_key = os.environ.get("NIM_API_KEY")
    if not api_key:
        print("ERROR: NIM_API_KEY is not set.")
        print("  Get a free key (no credit card) at https://build.nvidia.com, then:")
        print("    export NIM_API_KEY=nvapi-...        # Git Bash")
        print("    $env:NIM_API_KEY = 'nvapi-...'      # PowerShell")
        sys.exit(1)

    # Bias toward messages that actually contain techniques - an all-clean
    # sample would trivially "pass" by returning empty detections every time and
    # tell us nothing about whether the schema is being enforced.
    gold = load_real_gold()
    positives = [r for r in gold if r["labels"]]
    negatives = [r for r in gold if not r["labels"]]
    n_pos = min(len(positives), max(1, int(args.n * 0.75)))
    records = positives[:n_pos] + negatives[: args.n - n_pos]
    print(f"Testing {len(records)} gold messages ({n_pos} with techniques, {len(records) - n_pos} clean)")

    modes = ["guided_json", "json_object"] if args.mode == "both" else [args.mode]
    results = [run_mode(records, args.model, m, api_key) for m in modes]

    print(f"\n{'=' * 72}\nDECISION GATE\n{'=' * 72}")
    for r in results:
        pct = r["ok"] / r["n"]
        verdict = "PASS" if pct == 1.0 else ("PARTIAL" if pct >= 0.9 else "FAIL")
        print(f"  {r['mode']:<14} {r['ok']}/{r['n']} parsed  -> {verdict}")
    print(
        "\n  guided_json PASS  -> proceed as planned; llm.py's 'malformed output is\n"
        "                       IMPOSSIBLE' guarantee holds on the hosted path too.\n"
        "  guided_json FAIL  -> fall back to json_object + Pydantic + bounded retry,\n"
        "                       AND soften that docstring + say so in the README.\n"
        "                       Do not let the claim stand if the mechanism is gone."
    )


if __name__ == "__main__":
    main()
