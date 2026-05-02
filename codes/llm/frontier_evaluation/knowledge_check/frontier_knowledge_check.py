"""
Frontier LLM Evaluation Pipeline
==================================
Evaluates closed-source frontier language models on mixed MCQ / short-answer
knowledge-check datasets.

Models under evaluation
-----------------------
  Claude Haiku 4.5   — Anthropic API  (api.anthropic.com)
  GPT-4.1            — OpenAI API     (api.openai.com)

Question types
--------------
  MCQ          — model outputs A/B/C/D; graded by exact match.
  short_answer — model outputs free text; graded by an LLM judge
                 from a *different* model family (configurable).

Design principles
-----------------
  * API keys loaded exclusively from environment variables — never hardcoded.
  * Each test model uses its own native API client and SDK; no shared gateway.
  * Judge model is from a third, independent family (configurable via env var).
  * FAILED_SENTINEL distinguishes API failure from a wrong answer.
  * Disk cache (shelve) survives crashes; no duplicate API spend.
  * Exponential back-off on transient API errors.
  * Dataset validated before any API budget is spent.
  * MCQ and short-answer scores reported separately with bootstrap CIs.
  * parse_failed / api_failed / judge_failed counts logged per model.
  * Run metadata + dataset SHA-256 written to every output file.
  * No hardcoded paths, usernames, or institution identifiers.

Usage
-----
    # Required — test model keys
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."

    # Required — judge (must be from a different family than both test models)
    # Default judge is Mistral Large via Mistral API (neither Anthropic nor OpenAI).
    export JUDGE_API_KEY="..."
    export JUDGE_API_BASE="https://api.mistral.ai/v1"   # or any compatible endpoint
    export JUDGE_MODEL="mistral-large-latest"
    export JUDGE_FAMILY="mistral"                       # substring guard

    # Optional
    export DATASET_PATH="dataset.json"
    export OUTPUT_DIR="results"

    python eval_frontier.py

Dependencies
------------
    pip install anthropic openai numpy tqdm
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import shelve
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

# ── Native SDK imports ────────────────────────────────────────────────────────
import anthropic as _anthropic_sdk
from openai import OpenAI as _OpenAI


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS — key loading
# ──────────────────────────────────────────────────────────────────────────────
def _require_env(var: str, hint: str) -> str:
    """Return env var value or raise with an actionable message."""
    val = os.environ.get(var, "").strip()
    if not val:
        raise EnvironmentError(
            f"Environment variable {var!r} is not set.\n"
            f"  Export it before running: export {var}='{hint}'"
        )
    return val


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  ── all secrets come from environment variables
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Test model: Claude Haiku 4.5 (Anthropic native API) ──────────────────────
_anthropic_key = _require_env("ANTHROPIC_API_KEY", "sk-ant-...")
anthropic_client = _anthropic_sdk.Anthropic(api_key=_anthropic_key)

# ── Test model: GPT-4.1 (OpenAI native API) ───────────────────────────────────
_openai_key = _require_env("OPENAI_API_KEY", "sk-...")
openai_client = _OpenAI(api_key=_openai_key)

# ── Judge client (third-party provider — must NOT be Anthropic or OpenAI) ─────
# Default: Mistral Large via Mistral AI API.
# Override all three env vars to use a different judge provider.
_judge_key  = _require_env("JUDGE_API_KEY",  "...")
_judge_base = os.environ.get("JUDGE_API_BASE", "https://api.mistral.ai/v1").strip()
judge_client = _OpenAI(api_key=_judge_key, base_url=_judge_base)

# ── Model identifiers ─────────────────────────────────────────────────────────
# Verified API model strings (pinned snapshots for reproducibility):
#   Claude Haiku 4.5 — https://www.anthropic.com/news/claude-haiku-4-5
#   GPT-4.1          — https://platform.openai.com/docs/models/gpt-4.1
MODELS: dict[str, dict[str, str]] = {
    "Claude-Haiku-4.5": {
        "api_id":   "claude-haiku-4-5-20251001",   # pinned snapshot
        "provider": "anthropic",
    },
    "GPT-4.1": {
        "api_id":   "gpt-4.1-2025-04-14",           # pinned snapshot
        "provider": "openai",
    },
}

# ── Judge model ───────────────────────────────────────────────────────────────
# Must belong to a *different* family from every model in MODELS above.
# JUDGE_FAMILY is matched (case-insensitive) against each test model's api_id
# and display name to enforce this at start-up.
JUDGE_MODEL  = os.environ.get("JUDGE_MODEL",  "mistral-large-latest").strip()
JUDGE_FAMILY = os.environ.get("JUDGE_FAMILY", "mistral").strip()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = os.environ.get("DATASET_PATH", "dataset.json").strip()
OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR", "results").strip())
OUTPUT_PATH  = OUTPUT_DIR / "results_frontier.json"
CACHE_PATH   = str(OUTPUT_DIR / "query_cache")  # shelve appends .db

# ── Run hyper-parameters ──────────────────────────────────────────────────────
TEMP          = 0.0
RETRY         = 3
MAX_SKIP_RATE = 0.05   # abort if > 5 % of items cannot be evaluated

# ── Internal sentinels ────────────────────────────────────────────────────────
FAILED_SENTINEL  = "__API_FAILED__"
PARSE_FAIL_TOKEN = "__PARSE_FAILED__"


# ──────────────────────────────────────────────────────────────────────────────
# START-UP GUARD: judge family must not overlap with any test model
# ──────────────────────────────────────────────────────────────────────────────
def _check_judge_independence() -> None:
    """Raise if the judge family string appears in any test model identifier."""
    family = JUDGE_FAMILY.lower()
    for display_name, cfg in MODELS.items():
        for field in (display_name, cfg["api_id"], cfg["provider"]):
            if family in field.lower():
                raise ValueError(
                    f"Judge family '{JUDGE_FAMILY}' overlaps with test model "
                    f"'{display_name}' (field: {field!r}). "
                    "Use a judge from a completely independent provider."
                )
    log.info(
        "Judge independence check passed (family=%s, judge=%s).",
        JUDGE_FAMILY, JUDGE_MODEL,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a precise exam-taking assistant. "
    "Follow every instruction exactly. "
    "Never add explanations unless explicitly asked."
)


def mcq_prompt(item: dict) -> str:
    choices = "\n".join(f"{k}. {v}" for k, v in item["choices"].items())
    return (
        "Answer with EXACTLY ONE LETTER: A, B, C, or D.\n"
        "Do NOT write any words, punctuation, or explanation.\n"
        "Output the single letter and nothing else.\n\n"
        f"Question:\n{item['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Your answer (single letter):"
    )


def short_answer_prompt(item: dict) -> str:
    return (
        f"Question:\n{item['question']}\n\n"
        "Answer concisely and factually in 1-3 sentences:"
    )


def judge_prompt(question: str, reference: str, prediction: str) -> str:
    return (
        "You are a strict grading assistant.\n"
        "Score the student answer below using ONLY this rubric:\n"
        "  2 = fully correct\n"
        "  1 = partially correct (right idea, missing key detail)\n"
        "  0 = incorrect or no answer\n\n"
        "Output ONLY a single digit (0, 1, or 2). "
        "No words, no punctuation, no explanation.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Student answer: {prediction}\n\n"
        "Score:"
    )


# ──────────────────────────────────────────────────────────────────────────────
# PARSING HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def extract_mcq(raw: str) -> str:
    """
    Extract a single letter A-D from raw model output.

    Applies a prioritised chain of patterns; returns PARSE_FAIL_TOKEN if none
    match confidently.  The "last letter in arbitrary prose" fallback is
    intentionally absent to prevent false positives from explanatory text.
    """
    if not raw or raw == FAILED_SENTINEL:
        return PARSE_FAIL_TOKEN

    s = raw.strip().upper()

    if s in ("A", "B", "C", "D"):
        return s

    patterns = [
        r'FINAL\s*ANSWER\s*[:\-]?\s*\(?([A-D])\)?',
        r'\bANSWER\s*[:\-]?\s*\(?([A-D])\)?',
        r'(?:CORRECT|RIGHT)\s+(?:ANSWER\s+)?(?:IS\s+)?\(?([A-D])\)?',
        r'^\s*\(([A-D])\)\s*$',
        r'(?:OPTION|CHOICE)\s*\(?([A-D])\)?',
        r'^([A-D])[.)]\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1)

    return PARSE_FAIL_TOKEN


def extract_judge_score(raw: str) -> int | None:
    """
    Extract 0, 1, or 2 from judge output.
    Returns None on failure so the caller can flag it explicitly.
    """
    if not raw or raw == FAILED_SENTINEL:
        return None
    m = re.search(r'\b([012])\b', raw.strip())
    return int(m.group(1)) if m else None


def get_reference(item: dict) -> str:
    """Return the first non-empty reference field, in priority order."""
    return (
        item.get("model_answer")
        or item.get("explanation")
        or item.get("answer")
        or ""
    )


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# DATASET VALIDATION
# ──────────────────────────────────────────────────────────────────────────────
def validate_dataset(data: list[dict]) -> None:
    """
    Validate every item before spending any API budget.
    Raises ValueError with a descriptive message on the first malformed item.
    """
    required_top   = {"id", "type", "question"}
    valid_types    = {"mcq", "short_answer"}
    valid_mcq_keys = {"A", "B", "C", "D"}

    for i, item in enumerate(data):
        loc = f"item[{i}] id={item.get('id', '?')}"

        missing = required_top - item.keys()
        if missing:
            raise ValueError(f"{loc}: missing fields {missing}")

        if item["type"] not in valid_types:
            raise ValueError(f"{loc}: unknown type '{item['type']}'")

        if not item["question"].strip():
            raise ValueError(f"{loc}: empty question")

        if item["type"] == "mcq":
            if "answer" not in item or item["answer"] not in valid_mcq_keys:
                raise ValueError(f"{loc}: 'answer' must be A, B, C, or D")
            if "choices" not in item:
                raise ValueError(f"{loc}: missing 'choices'")
            if not valid_mcq_keys.issubset(item["choices"].keys()):
                raise ValueError(f"{loc}: 'choices' must contain A, B, C, D")
        else:
            if not get_reference(item):
                raise ValueError(
                    f"{loc}: no reference answer found "
                    "(need 'model_answer', 'explanation', or 'answer')"
                )

    log.info("Dataset validation passed — %d items OK.", len(data))


# ──────────────────────────────────────────────────────────────────────────────
# NATIVE API CALLERS
# ──────────────────────────────────────────────────────────────────────────────
def _call_anthropic(model_id: str, system: str, prompt: str, max_tokens: int) -> str:
    """Call Anthropic Messages API directly."""
    msg = anthropic_client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMP,
    )
    return msg.content[0].text if msg.content else ""


def _call_openai(model_id: str, system: str, prompt: str, max_tokens: int) -> str:
    """Call OpenAI Chat Completions API directly."""
    res = openai_client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=TEMP,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content or ""


def _call_judge(prompt: str, system: str, max_tokens: int) -> str:
    """Call the judge via its OpenAI-compatible client."""
    res = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=TEMP,
        max_tokens=max_tokens,
    )
    return res.choices[0].message.content or ""


# ──────────────────────────────────────────────────────────────────────────────
# QUERY  (disk-cached · exponential back-off · sentinel on failure)
# ──────────────────────────────────────────────────────────────────────────────
def query(
    provider: str,
    model_id: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    system: str = SYSTEM_PROMPT,
    use_judge: bool = False,
) -> str:
    """
    Dispatch a chat-completion request to the correct native API client.

    provider    — "anthropic" | "openai" (ignored when use_judge=True)
    Caching     — responses persisted to disk via shelve; repeated identical
                  calls are served from cache with no API spend.
    Retries     — up to RETRY attempts with exponential back-off (1s→2s→4s).
    Failure     — FAILED_SENTINEL returned after exhausting retries, so callers
                  can distinguish API failure from a wrong answer.
    """
    cache_key = f"{provider}|{model_id}|{system}|{prompt}"

    with shelve.open(CACHE_PATH) as cache:
        if cache_key in cache:
            return cache[cache_key]

    last_exc: Exception | None = None

    for attempt in range(RETRY):
        try:
            if use_judge:
                text = _call_judge(prompt, system, max_tokens)
            elif provider == "anthropic":
                text = _call_anthropic(model_id, system, prompt, max_tokens)
            elif provider == "openai":
                text = _call_openai(model_id, system, prompt, max_tokens)
            else:
                raise ValueError(f"Unknown provider: {provider!r}")

            with shelve.open(CACHE_PATH) as cache:
                cache[cache_key] = text
            return text

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2 ** attempt  # 1 s → 2 s → 4 s
            log.warning(
                "Query failed (attempt %d/%d, provider=%s, model=%s): %s "
                "— retrying in %ds.",
                attempt + 1, RETRY, provider, model_id, exc, wait,
            )
            time.sleep(wait)

    log.error(
        "All %d retries exhausted (provider=%s, model=%s): %s",
        RETRY, provider, model_id, last_exc,
    )
    return FAILED_SENTINEL


# ──────────────────────────────────────────────────────────────────────────────
# ITEM EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(provider: str, model_id: str, item: dict) -> dict:
    """Evaluate a single dataset item and return a result dict."""
    result: dict[str, Any] = {
        "id":           item["id"],
        "type":         item["type"],
        "question":     item["question"],
        "api_failed":   False,
        "parse_failed": False,
        "judge_failed": False,
    }

    if item["type"] == "mcq":
        raw  = query(provider, model_id, mcq_prompt(item), max_tokens=32)
        pred = extract_mcq(raw)

        api_failed   = raw  == FAILED_SENTINEL
        parse_failed = pred == PARSE_FAIL_TOKEN
        correct      = (not api_failed) and (not parse_failed) and (pred == item["answer"])

        result.update({
            "raw_output":   raw,
            "prediction":   pred if not parse_failed else None,
            "ground_truth": item["answer"],
            "correct":      correct,
            "score":        int(correct),
            "max_score":    1,
            "api_failed":   api_failed,
            "parse_failed": parse_failed,
        })

    else:  # short_answer
        raw = query(provider, model_id, short_answer_prompt(item), max_tokens=300)
        ref = get_reference(item)

        api_failed_model = raw == FAILED_SENTINEL

        judge_raw = query(
            provider,      # ignored — use_judge overrides routing
            JUDGE_MODEL,
            judge_prompt(item["question"], ref, raw),
            max_tokens=8,
            system="You are a strict grading assistant. Output only a single digit.",
            use_judge=True,
        )
        score = extract_judge_score(judge_raw)

        judge_failed = score is None
        if judge_failed:
            log.warning(
                "Judge parse failure for item id=%s — judge output: %r",
                item["id"], judge_raw,
            )
            score = 0  # conservative default; explicitly flagged above

        result.update({
            "raw_output":   raw,
            "judge_output": judge_raw,
            "ground_truth": ref,
            "score":        score,
            "max_score":    2,
            "api_failed":   api_failed_model,
            "judge_failed": judge_failed,
        })

    return result


# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVAL
# ──────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(
    arr: list[float],
    n_resamples: int = 2_000,
    ci: float = 95.0,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI for the mean of arr."""
    a = np.array(arr)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    means = [np.mean(np.random.choice(a, len(a), replace=True)) for _ in range(n_resamples)]
    return float(np.percentile(means, lo)), float(np.percentile(means, hi))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def run() -> None:
    # Enforce judge independence before spending any tokens
    _check_judge_independence()

    # ── Load & validate dataset ───────────────────────────────────────────────
    log.info("Loading dataset from %s", DATASET_PATH)
    with open(DATASET_PATH) as f:
        data: list[dict] = json.load(f)
    log.info("Loaded %d items.", len(data))
    validate_dataset(data)

    # ── Run metadata ──────────────────────────────────────────────────────────
    run_meta = {
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": sha256_file(DATASET_PATH),
        "judge_model":    JUDGE_MODEL,
        "judge_family":   JUDGE_FAMILY,
        "judge_api_base": _judge_base,
        "temperature":    TEMP,
        "seed":           SEED,
        "retry_count":    RETRY,
        "max_skip_rate":  MAX_SKIP_RATE,
        "n_items":        len(data),
        "models": {
            name: {"api_id": cfg["api_id"], "provider": cfg["provider"]}
            for name, cfg in MODELS.items()
        },
    }
    log.info("Run metadata:\n%s", json.dumps(run_meta, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final: dict = {"_run_meta": run_meta}

    # ── Per-model evaluation loop ─────────────────────────────────────────────
    for name, cfg in MODELS.items():
        provider = cfg["provider"]
        model_id = cfg["api_id"]
        log.info("=== Evaluating: %s | provider=%s | model=%s ===", name, provider, model_id)

        logs:           list[dict]  = []
        skipped:        list[dict]  = []
        mcq_scores:     list[int]   = []
        sa_norm_scores: list[float] = []
        api_failures = parse_failures = judge_failures = 0

        for item in tqdm(data, desc=name):
            try:
                r = evaluate(provider, model_id, item)
            except Exception as exc:  # noqa: BLE001
                skipped.append({
                    "id":       item.get("id"),
                    "error":    str(exc),
                    "exc_type": type(exc).__name__,
                })
                log.error("Unhandled exception on item id=%s: %s", item.get("id"), exc)
                continue

            logs.append(r)
            api_failures   += int(r.get("api_failed",   False))
            parse_failures += int(r.get("parse_failed", False))
            judge_failures += int(r.get("judge_failed", False))

            if r["type"] == "mcq":
                mcq_scores.append(r["score"])
            else:
                sa_norm_scores.append(r["score"] / r["max_score"])

        # ── Skip-rate guard ───────────────────────────────────────────────────
        skip_rate = len(skipped) / max(len(data), 1)
        if skip_rate > MAX_SKIP_RATE:
            log.error(
                "ABORT: %s skipped %.1f%% of items (threshold %.0f%%). "
                "Results are unreliable — check API connectivity.",
                name, skip_rate * 100, MAX_SKIP_RATE * 100,
            )
            final[name] = {"error": "skip_rate_exceeded", "skipped": skipped}
            continue

        # ── Summary statistics ────────────────────────────────────────────────
        def _ci(arr: list[float]) -> list[float]:
            lo, hi = bootstrap_ci(arr) if len(arr) > 1 else (0.0, 0.0)
            return [round(lo * 100, 2), round(hi * 100, 2)]

        mcq_acc     = float(np.mean(mcq_scores))     if mcq_scores     else 0.0
        sa_norm_avg = float(np.mean(sa_norm_scores)) if sa_norm_scores else 0.0
        all_norm    = [float(s) for s in mcq_scores] + sa_norm_scores
        overall     = float(np.mean(all_norm))       if all_norm       else 0.0

        summary = {
            # Overall
            "overall_pct":      round(overall * 100, 2),
            "overall_ci_95":    _ci(all_norm),
            "n_evaluated":      len(logs),
            "n_skipped":        len(skipped),
            "skip_rate_pct":    round(skip_rate * 100, 2),
            # MCQ
            "mcq_accuracy_pct": round(mcq_acc * 100, 2),
            "mcq_ci_95":        _ci([float(s) for s in mcq_scores]),
            "n_mcq":            len(mcq_scores),
            # Short answer
            "sa_norm_pct":      round(sa_norm_avg * 100, 2),
            "sa_ci_95":         _ci(sa_norm_scores),
            "n_sa":             len(sa_norm_scores),
            # Quality flags — any non-zero value should be investigated before publishing
            "api_failures":     api_failures,
            "parse_failures":   parse_failures,
            "judge_failures":   judge_failures,
        }

        final[name] = {"summary": summary, "skipped": skipped, "logs": logs}
        log.info("[%s] %s", name, json.dumps(summary, indent=2))

    # ── Persist results ───────────────────────────────────────────────────────
    with open(OUTPUT_PATH, "w") as f:
        json.dump(final, f, indent=2)
    log.info("Results saved to %s", OUTPUT_PATH)

    _print_leaderboard(final)


# ──────────────────────────────────────────────────────────────────────────────
# LEADERBOARD PRINTER
# ──────────────────────────────────────────────────────────────────────────────
def _print_leaderboard(final: dict) -> None:
    header = (
        f"{'Model':<22} {'Provider':<12} {'Overall%':>10} "
        f"{'95% CI':>22} {'MCQ%':>8} {'SA%':>8} {'Flags':>8}"
    )
    divider = "═" * len(header)
    print("\n" + divider)
    print(header)
    print("─" * len(header))

    rows = [
        (name, MODELS[name]["provider"], v["summary"])
        for name, v in final.items()
        if not name.startswith("_") and "error" not in v
    ]
    rows.sort(key=lambda x: -x[2]["overall_pct"])

    for name, provider, s in rows:
        ci       = f"[{s['overall_ci_95'][0]:.1f}, {s['overall_ci_95'][1]:.1f}]"
        flags    = s["api_failures"] + s["parse_failures"] + s["judge_failures"]
        flag_str = f"⚠  {flags}" if flags else "✓  0"
        print(
            f"{name:<22} {provider:<12} {s['overall_pct']:>9.2f}%  "
            f"{ci:>20}  {s['mcq_accuracy_pct']:>6.2f}%  "
            f"{s['sa_norm_pct']:>6.2f}%  {flag_str:>6}"
        )

    print(divider)


if __name__ == "__main__":
    run()