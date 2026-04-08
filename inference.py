"""
B2B Procurement Negotiation — Inference Script
Meta x Scaler OpenEnv Hackathon

MANDATORY environment variables:
    API_BASE_URL      The LiteLLM proxy endpoint (injected by validator).
    API_KEY           Your hackathon API key    (injected by validator).
    MODEL_NAME        Model identifier to use for inference.
    TASK_ID           Run a single task (1, 2, or 3); omit to run all three.
    USE_HEURISTIC     Set to "1" to skip LLM entirely (default: "0").

STDOUT FORMAT:
    [START] task=<task_name> env=procurement_negotiation model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
import subprocess, sys

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("openai")
_ensure("pydantic")
_ensure("python-dotenv", "dotenv")
_ensure("openenv-core[core]", "openenv")
import argparse
import json
import os
import re
import sys
import time
from typing import List, Optional

from openai import OpenAI

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config — read from environment (validator injects these at runtime) ────────
IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
USE_HEURISTIC = os.getenv("USE_HEURISTIC", "0").strip() == "1"

BENCHMARK  = "procurement_negotiation"
MAX_STEPS  = 60   # hard cap per task episode

# ── Diagnostics ───────────────────────────────────────────────────────────────
print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", file=sys.stderr)
print(f"[DEBUG] API_KEY exists={bool(API_KEY)}", file=sys.stderr)
print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", file=sys.stderr)
print(f"[DEBUG] USE_HEURISTIC={USE_HEURISTIC}", file=sys.stderr)

# ── LLM diagnostic counters ───────────────────────────────────────────────────
_llm_calls   = 0
_llm_success = 0
_llm_errors: List[str] = []


# =============================================================================
# STDOUT LOGGING  (matches validator expected format exactly)
# =============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val   = error if error else "null"
    done_val    = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", "")
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a professional B2B procurement manager negotiating with suppliers.

Your job is to purchase goods or services at the best value — balancing price, delivery speed, and supplier reliability.

On each turn you receive a structured negotiation update. You must respond with a single JSON action:

{"action_type": "accept", "counter_price": null}
{"action_type": "counter", "counter_price": 95.50}
{"action_type": "reject", "counter_price": null}

RULES:
1. "accept"  — close the deal at the supplier's current price
2. "counter" — propose a lower price (counter_price must be a plain number, no $ sign)
3. "reject"  — only in multi-supplier tasks to switch to a different supplier

STRATEGY:
- On round 1, ALWAYS start at 86-88% of the MARKET PRICE shown — NOT near the supplier's opening price
- Concede gradually each round — raise your counter by 1-2% of market each round
- Accept when the supplier's price is within 3-5% of market price
- If rounds are running out (2 or fewer left), accept to avoid losing the deal
- Never counter below 82% of market price — suppliers will not agree
- Read the "Signal" field — if the supplier is conceding fast, push harder
- Read the "BATNA" field — if a better supplier exists, consider rejecting
- For Task 2: prioritise delivery speed — accept up to 8% above market for delivery ≤ 2 days
- For Task 3: if current supplier price is >20% above market, REJECT and switch

Return ONLY the JSON. No explanation, no markdown, no extra text.

"""

# =============================================================================
# LLM CALL  (OpenAI client passed in — created once per task in run_task)
# =============================================================================

def call_llm(
    client: OpenAI,
    observation_text: str,
    fallback_price: float,
    last_market: Optional[float] = None,
    last_current: Optional[float] = None,
    last_price: Optional[float] = None,
) -> Optional[dict]:
    """
    Call the LLM via the hackathon LiteLLM proxy and parse a NegotiationAction.
    Returns a dict with action_type / counter_price, or None on failure.
    """
    global _llm_calls, _llm_success

    if USE_HEURISTIC:
        return None

    _llm_calls += 1

    MAX_RETRIES  = 3
    RETRY_DELAYS = [5, 10, 20]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": observation_text},
                ],
                temperature=0.1,
                max_tokens=80,
            )

            raw = response.choices[0].message.content.strip()
            raw = re.sub(r"```[a-zA-Z]*", "", raw).replace("```", "").strip()

            match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
            if not match:
                print(f"[LLM DEBUG] No JSON in response: {raw[:120]}", file=sys.stderr)
                return None

            action = json.loads(match.group())
            atype  = str(action.get("action_type", "counter")).lower().strip()
            if "accept" in atype:   atype = "accept"
            elif "reject" in atype: atype = "reject"
            else:                   atype = "counter"

            price = action.get("counter_price", None)
            if price is not None:
                if isinstance(price, str):
                    nums = re.findall(r"\d+\.?\d*", price.replace(",", ""))
                    price = float(nums[-1]) if nums else fallback_price
                else:
                    price = float(price)
                if price <= 0:
                    price = fallback_price

            result = {
                "action_type":   atype,
                "counter_price": round(price, 2) if atype == "counter" else None,
            }

            # Clamp LLM counter price to a sensible range
            if atype == "counter" and result["counter_price"] is not None:
                upper_bound = (last_current or result["counter_price"]) * 0.97
                lower_bound = (last_market  or fallback_price / 0.90) * 0.82
                clamped = round(max(lower_bound, min(upper_bound, result["counter_price"])), 2)
                # Nudge down slightly if stuck on the same price
                if last_price is not None and last_price == clamped:
                    clamped = round(clamped * 0.98, 2)
                result["counter_price"] = clamped

            # Carry target_supplier_id for task-3 reject actions
            if atype == "reject":
                tid = action.get("target_supplier_id")
                if tid is not None:
                    result["target_supplier_id"] = int(tid)

            _llm_success += 1
            return result

        except Exception as e:
            err_str = str(e)
            is_rate_limit = any(
                tok in err_str for tok in ("402", "429")
            ) or any(
                kw in err_str.lower() for kw in ("rate", "credits")
            )

            if is_rate_limit and attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(
                    f"[LLM DEBUG] Call #{_llm_calls} rate-limited "
                    f"(attempt {attempt + 1}/{MAX_RETRIES}) — retrying in {delay}s …",
                    file=sys.stderr,
                )
                time.sleep(delay)
                continue

            err = f"{type(e).__name__}: {e}"
            _llm_errors.append(err)
            print(f"[LLM DEBUG] Call #{_llm_calls} failed — {err}", file=sys.stderr)
            return None

    return None


# =============================================================================
# HEURISTIC FALLBACK
# =============================================================================

def heuristic_action(obs, supplier_held_rounds: int = 0) -> dict:
    """
    Rule-based buyer strategy used when the LLM is unavailable or fails.

    Counter-price is anchored to market_price (not the supplier's inflated ask)
    and rises gradually each round. A small concession is added when the supplier
    holds firm, to signal good faith and break deadlocks.
    """
    current     = obs.current_price
    market      = obs.market_price
    rounds_left = obs.rounds_left
    task_id     = obs.task_id
    delivery    = obs.delivery_days
    flexibility = obs.supplier_flexibility
    round_num   = getattr(obs, "round_number", 1) or 1

    # Emergency accept on last round
    if rounds_left <= 1:
        return {"action_type": "accept", "counter_price": None}

    # Task 3: switch to a clearly better supplier if one exists
    if task_id == 3:
        available = obs.available_suppliers
        if len(available) > 1:
            def sup_score(s):
                price_s    = 1.0 - min(s["current_price"] / (market * 1.5), 1.0)
                rel_s      = s.get("reliability", 0.5)
                delivery_s = 1.0 - min(s.get("delivery_days", 10) / 20.0, 1.0)
                return 0.4 * price_s + 0.4 * rel_s + 0.2 * delivery_s

            current_sup = next((s for s in available if s["supplier_id"] == obs.supplier_id), None)
            best_sup    = max(available, key=sup_score)
            if (
                current_sup
                and best_sup["supplier_id"] != obs.supplier_id
                and rounds_left >= 5
                and sup_score(best_sup) > sup_score(current_sup) + 0.06
            ):
                return {
                    "action_type": "reject",
                    "counter_price": None,
                    "target_supplier_id": best_sup["supplier_id"],
                }

    # Accept thresholds
    # Accept thresholds — never accept before round 5 unless price is a steal
    early_rounds = round_num < 5

    if task_id == 2:
        if early_rounds:
            # Only accept early if price is at or below 80% of market (exceptional deal)
            if current <= market * 0.80:
                return {"action_type": "accept", "counter_price": None}
        else:
            if delivery <= 2 and current <= market * 1.03:
                return {"action_type": "accept", "counter_price": None}
            if current <= market * 1.02:
                return {"action_type": "accept", "counter_price": None}
    else:
        if early_rounds:
            if current <= market * 0.80:
                return {"action_type": "accept", "counter_price": None}
        else:
            if current <= market * 1.03:
                return {"action_type": "accept", "counter_price": None}

    # Round-based target: starts at 86% of market, climbs ~2% per round
    ROUND_TARGETS = {1: 0.88, 2: 0.90, 3: 0.92, 4: 0.93, 5: 0.95, 6: 0.96, 7: 0.97, 8: 0.98}
    target_pct = ROUND_TARGETS.get(round_num, 0.98)

    # Under time pressure — be more generous to guarantee a close
    if rounds_left <= 3:
        target_pct = max(target_pct, 0.97)
    if rounds_left <= 2:
        target_pct = max(target_pct, 0.98)

    # Adjust for supplier flexibility
    if flexibility > 0.6:
        target_pct -= 0.02
    elif flexibility < 0.3:
        target_pct += 0.02

    # Nudge offer up slightly each round the supplier holds firm (max +5%)
    if supplier_held_rounds > 0:
        target_pct += min(supplier_held_rounds * 0.01, 0.05)

    counter = round(market * target_pct, 2)
    counter = min(counter, round(current * 0.99, 2))   # must be below current ask
    counter = max(counter, round(market * 0.82, 2))    # never below floor

    return {"action_type": "counter", "counter_price": counter}


# =============================================================================
# TASK RUNNER
# =============================================================================

def run_task(task_id: int) -> None:
    """Run one negotiation episode for the given task_id."""
    task_name = f"task{task_id}"
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Imports deferred so [START] is always emitted before any import error
    try:
        from server.environment import ProcurementEnvironment
        from models import NegotiationAction
    except ImportError as ie:
        print(f"[DEBUG] Import error: {ie}", file=sys.stderr)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return

    # Build OpenAI client once per task (matches sample script pattern)
    if not API_KEY:
     print(f"[DEBUG] No API_KEY set. Set API_KEY env var or use USE_HEURISTIC=1", file=sys.stderr)
     log_end(success=False, steps=0, score=0.0, rewards=[])
     return
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        env = ProcurementEnvironment(task_id=task_id, seed=42)
        obs = env.reset()

        # LLM price-clamp references (updated each step)
        last_market  = obs.market_price
        last_current = obs.current_price
        last_price: Optional[float] = None

        supplier_held_rounds = 0
        last_supplier_price  = obs.current_price

        for step in range(1, MAX_STEPS + 1):
            obs_text = obs.text_summary or json.dumps(obs.model_dump(), indent=2)

            # Update clamp references before the LLM call
            last_market  = obs.market_price
            last_current = obs.current_price

            # Track consecutive rounds where supplier didn't concede
            if step > 1:
                if obs.current_price >= last_supplier_price - 0.001:
                    supplier_held_rounds += 1
                else:
                    supplier_held_rounds = 0
            last_supplier_price = obs.current_price

            # Always try LLM first; heuristic is only a fallback on failure
            raw = call_llm(
                client,
                obs_text,
                fallback_price=obs.market_price * 0.90,
                last_market=last_market,
                last_current=last_current,
                last_price=last_price,
            )
            if raw is None:
                raw = heuristic_action(obs, supplier_held_rounds)

            # Ensure counter always carries a price
            if raw.get("action_type") == "counter" and not raw.get("counter_price"):
                raw["counter_price"] = heuristic_action(obs, supplier_held_rounds).get(
                    "counter_price", round(obs.market_price * 0.90, 2)
                )

            # Update last_price for next-step stall detection
            if raw.get("action_type") == "counter":
                last_price = raw.get("counter_price")

            # Build action string for [STEP] log
            atype = raw.get("action_type", "unknown")
            if atype == "counter":
                action_str = f"counter({raw.get('counter_price')})"
            elif atype == "reject" and raw.get("target_supplier_id"):
                action_str = f"reject(target={raw['target_supplier_id']})"
            else:
                action_str = atype

            try:
                action = NegotiationAction(**raw)
            except Exception:
                action     = NegotiationAction(action_type="accept")
                action_str = "accept(fallback)"

            obs = env.step(action)

            reward = float(obs.reward)
            done   = bool(obs.done)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            time.sleep(1)

            if done:
                break

        state  = env.state
        grader = state.metadata.get("grader_result", {})
        score  = float(grader.get("score", 0.0))
        success = score > 0.0

    except Exception as e:
        print(f"[DEBUG] task{task_id} exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=None)
    args = parser.parse_args()

    task_id_env = args.task_id or os.getenv("TASK_ID")
    task_ids = [int(task_id_env)] if task_id_env else [1, 2, 3]

    for task_id in task_ids:
        try:
            run_task(task_id)
        except Exception as e:
            print(f"[DEBUG] Fatal error on task{task_id}: {e}", file=sys.stderr)

    # Final diagnostics to stderr
    if _llm_calls > 0:
        rate = _llm_success / _llm_calls * 100
        print(
            f"\n[DIAG] LLM: {_llm_success}/{_llm_calls} succeeded ({rate:.0f}%) "
            f"| model={MODEL_NAME}",
            file=sys.stderr,
        )
        if _llm_errors:
            print(f"[DIAG] Errors: {list(dict.fromkeys(_llm_errors))[:2]}", file=sys.stderr)
    else:
        print("[DIAG] No LLM calls made — USE_HEURISTIC=1 or API_KEY missing", file=sys.stderr)


if __name__ == "__main__":
    main()