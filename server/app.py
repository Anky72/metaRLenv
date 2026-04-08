# Copyright (c) Meta Platforms, Inc.

# Fix imports for server structure
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: pip install openenv-core") from e

from fastapi import Request
from fastapi.responses import JSONResponse, FileResponse

# Local imports (now work from server/)
from models import NegotiationAction, NegotiationObservation
from server.environment import ProcurementEnvironment

# ── Config (same as inference.py) ─────────────────────────────────────────────
import json
import re
import time
from typing import Optional
from openai import OpenAI

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
USE_HEURISTIC = os.getenv("USE_HEURISTIC", "0").strip() == "1"

_openai_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    return _openai_client


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
- For Task 3: if current supplier price is >20% above market, REJECT and switch

Return ONLY the JSON. No explanation, no markdown, no extra text.
"""


def call_llm(observation_text: str, fallback_price: float) -> Optional[dict]:
    """
    Call the LLM and return a NegotiationAction dict, or None on failure.
    Attaches .last_market / .last_current / .last_price as function attributes
    so app.py can update clamp references without extra globals.
    """
    if USE_HEURISTIC or not API_KEY:
        return None

    last_market  = getattr(call_llm, "last_market",  None)
    last_current = getattr(call_llm, "last_current", None)
    last_price   = getattr(call_llm, "last_price",   None)

    MAX_RETRIES  = 3
    RETRY_DELAYS = [5, 10, 20]

    for attempt in range(MAX_RETRIES):
        try:
            client   = _get_client()
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
                return None

            action = json.loads(match.group())
            atype  = str(action.get("action_type", "counter")).lower().strip()
            if   "accept" in atype: atype = "accept"
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

            if atype == "counter" and result["counter_price"] is not None:
                upper_bound = (last_current or result["counter_price"]) * 0.97
                lower_bound = (last_market  or fallback_price / 0.90) * 0.82
                clamped = round(max(lower_bound, min(upper_bound, result["counter_price"])), 2)
                if last_price is not None and last_price == clamped:
                    clamped = round(clamped * 0.98, 2)
                result["counter_price"] = clamped

            if atype == "reject":
                tid = action.get("target_supplier_id")
                if tid is not None:
                    result["target_supplier_id"] = int(tid)

            return result

        except Exception as e:
            err_str = str(e)
            is_rate_limit = any(
                tok in err_str for tok in ("402", "429")
            ) or any(
                kw in err_str.lower() for kw in ("rate", "credits")
            )
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
                continue
            print(f"[LLM DEBUG] call_llm failed: {e}", file=sys.stderr)
            return None

    return None


def heuristic_action(obs, supplier_held_rounds: int = 0) -> dict:
    """Rule-based buyer strategy (fallback when LLM unavailable)."""
    current     = obs.current_price
    market      = obs.market_price
    rounds_left = obs.rounds_left
    task_id     = obs.task_id
    delivery    = obs.delivery_days
    flexibility = obs.supplier_flexibility
    round_num   = getattr(obs, "round_number", 1) or 1

    if rounds_left <= 1:
        return {"action_type": "accept", "counter_price": None}

    if task_id == 3:
        available = obs.available_suppliers
        if len(available) > 1:
            def sup_score(s):
                price_s    = 1.0 - min(s["current_price"] / (market * 1.5), 1.0)
                rel_s      = s.get("reliability", 0.5)
                delivery_s = 1.0 - min(s.get("delivery_days", 10) / 20.0, 1.0)
                return 0.4 * price_s + 0.4 * rel_s + 0.2 * delivery_s

            current_sup = next(
                (s for s in available if s["supplier_id"] == obs.supplier_id), None
            )
            best_sup = max(available, key=sup_score)
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

    if task_id == 2:
        if delivery <= 2 and current <= market * 1.03:
            return {"action_type": "accept", "counter_price": None}
        if current <= market * 1.02:
            return {"action_type": "accept", "counter_price": None}
    else:
        if current <= market * 1.03:
            return {"action_type": "accept", "counter_price": None}

    ROUND_TARGETS = {1: 0.86, 2: 0.88, 3: 0.90, 4: 0.92, 5: 0.94, 6: 0.95, 7: 0.96, 8: 0.97}
    target_pct = ROUND_TARGETS.get(round_num, 0.98)

    if rounds_left <= 3: target_pct = max(target_pct, 0.97)
    if rounds_left <= 2: target_pct = max(target_pct, 0.98)

    if flexibility > 0.6:  target_pct -= 0.02
    elif flexibility < 0.3: target_pct += 0.02

    if supplier_held_rounds > 0:
        target_pct += min(supplier_held_rounds * 0.01, 0.05)

    counter = round(market * target_pct, 2)
    counter = min(counter, round(current * 0.99, 2))
    counter = max(counter, round(market * 0.82, 2))

    return {"action_type": "counter", "counter_price": counter}


# ── Global state ──────────────────────────────────────────────────────────────
GLOBAL_ENV  = ProcurementEnvironment(task_id=int(os.getenv("TASK_ID", "1")))
LAST_OBS    = None
LAST_PRICES = []

_step_num              = 0
_rewards               = []
_agent_score           = 0.0
_grader                = {}

_last_supplier_price   = None
_last_agent_counter    = None
_supplier_held_rounds  = 0


def _reset_log_state():
    global _step_num, _rewards, _agent_score, _grader, LAST_PRICES
    global _last_supplier_price, _last_agent_counter, _supplier_held_rounds
    _step_num             = 0
    _rewards              = []
    _agent_score          = 0.0
    _grader               = {}
    LAST_PRICES           = []
    _last_supplier_price  = None
    _last_agent_counter   = None
    _supplier_held_rounds = 0


def _log(tag: str, msg: str):
    print(f"[{tag}] {msg}", flush=True)


# ── App factory ───────────────────────────────────────────────────────────────
def create_environment():
    return GLOBAL_ENV


app = create_app(
    create_environment,
    NegotiationAction,
    NegotiationObservation,
    env_name="procurement_negotiation",
    max_concurrent_envs=10,
)

app.router.routes = [
    r for r in app.router.routes
    if not (hasattr(r, "path") and r.path in ("/step", "/reset", "/state"))
]


# =============================================================================
# RESET
# =============================================================================
@app.post("/reset")
async def reset_override(request: Request):
    global LAST_OBS

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    task_id = body.get("task_id") or int(os.getenv("TASK_ID", "1"))
    task_id = int(task_id)

    GLOBAL_ENV.task_id = task_id
    _reset_log_state()
    LAST_OBS = GLOBAL_ENV.reset()

    call_llm.last_market  = LAST_OBS.market_price
    call_llm.last_current = LAST_OBS.current_price
    call_llm.last_price   = None

    _log("START", f"task=task{task_id} env=procurement_negotiation model={MODEL_NAME}")

    obs_dict = LAST_OBS.model_dump()
    return JSONResponse({
        **obs_dict,
        "reward":        0.0,
        "done":          False,
        "agent_score":   0.0,
        "score":         0.0,
        "step":          0,
        "action_type":   None,
        "counter_price": None,
    })


# =============================================================================
# STEP
# =============================================================================
@app.post("/step")
async def step_override(request: Request):
    global LAST_OBS, LAST_PRICES, _step_num, _rewards, _agent_score, _grader
    global _last_supplier_price, _last_agent_counter, _supplier_held_rounds

    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    env = GLOBAL_ENV

    if LAST_OBS is None:
        LAST_OBS = env.reset()
        _log("START", f"task=task{env.task_id} env=procurement_negotiation model={MODEL_NAME}")

    obs = LAST_OBS

    if _last_supplier_price is not None:
        if obs.current_price >= _last_supplier_price - 0.001:
            _supplier_held_rounds += 1
        else:
            _supplier_held_rounds = 0
    _last_supplier_price = obs.current_price

    call_llm.last_market  = obs.market_price
    call_llm.last_current = obs.current_price

    explicit_action = body.get("action") or body.get("action_type")

    if explicit_action and isinstance(explicit_action, dict) and explicit_action.get("action_type"):
        raw = {
            "action_type":   explicit_action.get("action_type", "counter"),
            "counter_price": explicit_action.get("counter_price"),
        }
        if explicit_action.get("target_supplier_id"):
            raw["target_supplier_id"] = explicit_action["target_supplier_id"]

        incoming_price = explicit_action.get("counter_price") or explicit_action.get("price")
        if incoming_price is not None:
            if abs(float(incoming_price) - obs.current_price) < 0.001:
                _supplier_held_rounds += 1
            else:
                _supplier_held_rounds = 0

    elif isinstance(explicit_action, str):
        raw = {"action_type": explicit_action, "counter_price": body.get("counter_price")}
    else:
        target_price = obs.market_price * 0.90

        raw = None
        if not (obs.rounds_left <= 2 or obs.current_price <= obs.market_price * 1.03):
            try:
                raw = call_llm(obs.text_summary, fallback_price=target_price)
            except Exception as e:
                print(f"[LLM DEBUG] call_llm exception: {e}", file=sys.stderr)
                raw = None

        if not isinstance(raw, dict) or "action_type" not in raw:
            raw = heuristic_action(obs, supplier_held_rounds=_supplier_held_rounds)

        if raw.get("action_type") == "counter":
            price = raw.get("counter_price") or target_price
            LAST_PRICES.append(round(price, 2))
            if len(LAST_PRICES) >= 3 and len(set(LAST_PRICES[-3:])) == 1:
                raw["counter_price"] = round(price * 0.97, 2)

        if obs.rounds_left <= 1 or obs.current_price <= obs.market_price * 1.03:
            raw = {"action_type": "accept", "counter_price": None}

    try:
        action = NegotiationAction(**raw)
    except Exception:
        action = NegotiationAction(action_type="accept")
        raw    = {"action_type": "accept", "counter_price": None}

    if raw.get("action_type") == "counter":
        _last_agent_counter = raw.get("counter_price")

    new_obs  = env.step(action)
    LAST_OBS = new_obs

    _step_num += 1
    reward = float(new_obs.reward)
    done   = bool(new_obs.done)
    _rewards.append(reward)

    action_str = raw["action_type"]
    if action_str == "counter":
        action_str = f"counter({raw.get('counter_price')})"
    _log("STEP", f"step={_step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error=null")

    if done:
        try:
            gr           = env.state.metadata.get("grader_result") or {}
            _grader      = gr if isinstance(gr, dict) else {}
            _agent_score = float(_grader.get("score", 0.0))
        except Exception:
            _agent_score = 0.0
            _grader      = {}

        accepted    = (raw["action_type"] == "accept")
        success     = accepted and _agent_score > 0
        rewards_str = ",".join(f"{r:.2f}" for r in _rewards)
        _log("END", f"success={str(success).lower()} steps={_step_num} score={_agent_score:.3f} rewards={rewards_str}")

    obs_dict = new_obs.model_dump()
    return JSONResponse({
        **obs_dict,
        "action_type":          raw.get("action_type"),
        "counter_price":        raw.get("counter_price"),
        "reward":               reward,
        "done":                 done,
        "agent_score":          _agent_score,
        "score":                _agent_score,
        "step":                 _step_num,
        "supplier_held_rounds": _supplier_held_rounds,
    })


# =============================================================================
# STATE
# =============================================================================
@app.get("/state")
async def state_override():
    try:
        state = GLOBAL_ENV.state
        meta  = dict(state.metadata)
        if not isinstance(meta.get("grader_result"), dict):
            meta["grader_result"] = _grader
        return JSONResponse({
            "episode_id":  state.episode_id,
            "step_count":  state.step_count,
            "metadata":    meta,
            "agent_score": _agent_score,
        })
    except Exception:
        return JSONResponse({
            "episode_id":  None,
            "step_count":  _step_num,
            "metadata":    {"grader_result": _grader},
            "agent_score": _agent_score,
        })


# =============================================================================
# ROOT — serve the web UI
# =============================================================================
@app.get("/")
async def root():
    ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web_ui.html")
    return FileResponse(ui_path)


def main():
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()