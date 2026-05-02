"""
LLM Evaluation Pipeline
========================
Evaluates language models on mixed MCQ / short-answer datasets.

Question types
--------------
  MCQ          — model outputs A/B/C/D; graded by exact match.
  short_answer — model outputs free text; graded by an LLM judge
                 from a *different* model family (configurable).

Design principles
-----------------
  * API keys loaded exclusively from environment variables — never hardcoded.
  * Judge model must be from a different family than every test model
    (enforced at start-up via JUDGE_FAMILY guard).
  * FAILED_SENTINEL distinguishes API failure from a wrong answer.
  * Disk cache (shelve) survives crashes; no duplicate API spend.
  * Exponential back-off on transient API errors.
  * Dataset validated before any API budget is spent.
  * MCQ and short-answer scores reported separately with bootstrap CIs.
  * parse_failed / api_failed / judge_failed counts logged per model.
  * Run metadata + dataset SHA-256 written to every output file.

Usage
-----
    export OPENROUTER_API_KEY="sk-or-..."
    export JUDGE_API_KEY="..."          # key for the judge's API endpoint
    python eval_pipeline.py

Dependencies
------------
    pip install openai numpy tqdm
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

import numpy as np
from openai import OpenAI
from tqdm import tqdm


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
# CONFIGURATION  ── all secrets come from environment variables
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Test-model client (OpenRouter) ────────────────────────────────────────────
_openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
if not _openrouter_key:
    raise EnvironmentError(
        "OPENROUTER_API_KEY is not set. "
        "Export it before running: export OPENROUTER_API_KEY='sk-or-...'"
    )

client = OpenAI(
    api_key=_openrouter_key,
    base_url="https://openrouter.ai/api/v1",
)

# ── Judge client (separate endpoint / provider) ───────────────────────────────
# The judge must never belong to the same model family as any test model.
# Set JUDGE_API_BASE to the native endpoint of your chosen judge provider,
# e.g. "https://api.x.ai/v1" for xAI or "https://api.openai.com/v1" for OpenAI.
_judge_key = os.environ.get("JUDGE_API_KEY", "")
_judge_base = os.environ.get("JUDGE_API_BASE", "https://api.openai.com/v1")
if not _judge_key:
    raise EnvironmentError(
        "JUDGE_API_KEY is not set. "
        "Export it before running: export JUDGE_API_KEY='...'"
    )

judge_client = OpenAI(
    api_key=_judge_key,
    base_url=_judge_base,
)

# ── Models under evaluation ───────────────────────────────────────────────────
# Keys are human-readable display names; values are the OpenRouter model strings.
MODELS: dict[str, str] = {
    "Qwen2.5-7B":    "qwen/qwen-2.5-7b-instruct",
    "Qwen2.5-72B":   "qwen/qwen-2.5-72b-instruct",
    "Llama-3.1-8B":  "meta-llama/llama-3.1-8b-instruct",
    "Llama-3.1-70B": "meta-llama/llama-3.1-70b-instruct",
    "Mixtral-8x7B":  "mistralai/mixtral-8x7b-instruct:nitro",
}

# ── Judge model ───────────────────────────────────────────────────────────────
# Must be from a *different* family than every model in MODELS above.
# JUDGE_FAMILY is matched (case-insensitive) against each test model's string
# to enforce this at start-up.
JUDGE_MODEL  = os.environ.get("JUDGE_MODEL",  "gpt-4o")
JUDGE_FAMILY = os.environ.get("JUDGE_FAMILY", "gpt")   # substring guard

# ── Paths ─────────────────────────────────────────────────────────────────────
DATASET_PATH = os.environ.get("DATASET_PATH", "dataset.json")
OUTPUT_DIR   = Path(os.environ.get("OUTPUT_DIR", "results"))
OUTPUT_PATH  = OUTPUT_DIR / "results.json"
CACHE_PATH   = str(OUTPUT_DIR / "query_cache")  # shelve appends .db

# ── Run hyper-parameters ──────────────────────────────────────────────────────
TEMP          = 0.0
RETRY         = 3
MAX_SKIP_RATE = 0.05   # abort if > 5 % of items cannot be evaluated

# ── Internal sentinels ────────────────────────────────────────────────────────
FAILED_SENTINEL  = "__API_FAILED__"
PARSE_FAIL_TOKEN = "__PARSE_FAILED__"


# ──────────────────────────────────────────────────────────────────────────────
# START-UP GUARD: judge family must not appear in any test-model string
# ──────────────────────────────────────────────────────────────────────────────
def _check_judge_independence() -> None:
    """Raise if the judge model family overlaps with any test model."""
    family = JUDGE_FAMILY.lower()
    for name, model_str in MODELS.items():
        if family in model_str.lower() or family in name.lower():
            raise ValueError(
                f"Judge family '{JUDGE_FAMILY}' overlaps with test model "
                f"'{name}' ({model_str}). Use a judge from a different family."
            )
    log.info("Judge independence check passed (family=%s, judge=%s).", JUDGE_FAMILY, JUDGE_MODEL)


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

    Applies a prioritised chain of patterns; returns PARSE_FAIL_TOKEN if no
    pattern matches confidently.  The last-letter-in-arbitrary-prose fallback
    is intentionally absent to avoid false positives from explanatory text.
    """
    if not raw or raw == FAILED_SENTINEL:
        return PARSE_FAIL_TOKEN

    s = raw.strip().upper()

    # 1. Exact single character
    if s in ("A", "B", "C", "D"):
        return s

    patterns = [
        r'FINAL\s*ANSWER\s*[:\-]?\s*\(?([A-D])\)?',        # "Final Answer: A"
        r'\bANSWER\s*[:\-]?\s*\(?([A-D])\)?',              # "Answer: A"
        r'(?:CORRECT|RIGHT)\s+(?:ANSWER\s+)?(?:IS\s+)?\(?([A-D])\)?',  # "correct answer is A"
        r'^\s*\(([A-D])\)\s*$',                             # "(A)" alone on line
        r'(?:OPTION|CHOICE)\s*\(?([A-D])\)?',              # "Option A"
        r'^([A-D])[.)]\s*$',                                # "A." or "A)"
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
# API QUERY  (disk-cached · exponential back-off · sentinel on failure)
# ──────────────────────────────────────────────────────────────────────────────
def query(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 256,
    system: str = SYSTEM_PROMPT,
    use_judge_client: bool = False,
) -> str:
    """
    Send a chat completion request and return the response text.

    Caching   — responses are persisted to disk via shelve; repeated identical
                calls are served from cache without hitting the API.
    Retries   — transient failures are retried up to RETRY times with
                exponential back-off (1 s → 2 s → 4 s).
    Failure   — after all retries are exhausted, FAILED_SENTINEL is returned
                so callers can distinguish API failure from a wrong answer.
    Routing   — set use_judge_client=True to route through the dedicated judge
                client (separate provider / key).
    """
    cache_key = f"{model}|||{system}|||{prompt}"

    with shelve.open(CACHE_PATH) as cache:
        if cache_key in cache:
            return cache[cache_key]

    api_client = judge_client if use_judge_client else client
    last_exc: Exception | None = None

    for attempt in range(RETRY):
        try:
            res = api_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMP,
                max_tokens=max_tokens,
            )
            text = res.choices[0].message.content or ""
            with shelve.open(CACHE_PATH) as cache:
                cache[cache_key] = text
            return text

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2 ** attempt  # 1 s → 2 s → 4 s
            log.warning(
                "Query failed (attempt %d/%d, model=%s): %s — retrying in %ds.",
                attempt + 1, RETRY, model, exc, wait,
            )
            time.sleep(wait)

    log.error("All %d retries exhausted for model=%s: %s", RETRY, model, last_exc)
    return FAILED_SENTINEL


# ──────────────────────────────────────────────────────────────────────────────
# ITEM EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(model: str, item: dict) -> dict:
    """Evaluate a single dataset item and return a result dict."""
    result: dict = {
        "id":           item["id"],
        "type":         item["type"],
        "question":     item["question"],
        "api_failed":   False,
        "parse_failed": False,
        "judge_failed": False,
    }

    if item["type"] == "mcq":
        raw  = query(model, mcq_prompt(item), max_tokens=32)
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
        raw = query(model, short_answer_prompt(item), max_tokens=300)
        ref = get_reference(item)

        api_failed_model = raw == FAILED_SENTINEL

        judge_raw = query(
            JUDGE_MODEL,
            judge_prompt(item["question"], ref, raw),
            max_tokens=8,
            system="You are a strict grading assistant. Output only a single digit.",
            use_judge_client=True,
        )
        score = extract_judge_score(judge_raw)

        judge_failed = score is None
        if judge_failed:
            log.warning(
                "Judge parse failure for item id=%s — judge output: %r",
                item["id"], judge_raw,
            )
            score = 0  # conservative default; flagged explicitly above

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
    # Enforce judge-independence before spending any tokens
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
    }
    log.info("Run metadata:\n%s", json.dumps(run_meta, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final: dict = {"_run_meta": run_meta}

    # ── Per-model evaluation loop ─────────────────────────────────────────────
    for name, model in MODELS.items():
        log.info("=== Evaluating: %s (%s) ===", name, model)

        logs:           list[dict]  = []
        skipped:        list[dict]  = []
        mcq_scores:     list[int]   = []
        sa_norm_scores: list[float] = []
        api_failures = parse_failures = judge_failures = 0

        for item in tqdm(data, desc=name):
            try:
                r = evaluate(model, item)
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
            # Quality flags — investigate any non-zero value before publishing
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
    header = f"{'Model':<20} {'Overall%':>10} {'95% CI':>22} {'MCQ%':>8} {'SA%':>8} {'Flags':>8}"
    print("\n" + "═" * len(header))
    print(header)
    print("─" * len(header))

    rows = [
        (name, v["summary"])
        for name, v in final.items()
        if not name.startswith("_") and "error" not in v
    ]
    rows.sort(key=lambda x: -x[1]["overall_pct"])

    for name, s in rows:
        ci       = f"[{s['overall_ci_95'][0]:.1f}, {s['overall_ci_95'][1]:.1f}]"
        flags    = s["api_failures"] + s["parse_failures"] + s["judge_failures"]
        flag_str = f"⚠  {flags}" if flags else "✓  0"
        print(
            f"{name:<20} {s['overall_pct']:>9.2f}%  {ci:>20}  "
            f"{s['mcq_accuracy_pct']:>6.2f}%  {s['sa_norm_pct']:>6.2f}%  {flag_str:>6}"
        )

    print("═" * len(header))


if __name__ == "__main__":
    run()