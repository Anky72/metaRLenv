# B2B Procurement Negotiation Environment

**Meta × Scaler OpenEnv Hackathon**

---

## Table of Contents

1. [Environment Description & Motivation](#1-environment-description--motivation)
2. [Observation Space](#2-observation-space)
3. [Action Space](#3-action-space)
4. [Task Descriptions](#4-task-descriptions)
5. [Setup & Usage Instructions](#5-setup--usage-instructions)
6. [Baseline Scores](#6-baseline-scores)
7. [Project Structure](#7-project-structure)
8. [Environment Variables Reference](#8-environment-variables-reference)

---

## 1. Environment Description & Motivation

### What is this?

This is a **structured negotiation environment** in which an AI agent plays the role of a B2B procurement manager. The agent negotiates price and delivery terms with simulated suppliers drawn from a real-world procurement catalog covering industrial fasteners, office supplies, semiconductors, raw steel, PPE, and more.

The environment is built on the [OpenEnv](https://pypi.org/project/openenv-core/) framework and exposes a standard `reset / step` interface. It supports both **standalone inference** (direct Python, no server needed) and **server mode** (FastAPI + Docker + Web UI).

### Motivation

Procurement negotiation is one of the highest-leverage tasks in enterprise operations — even a 2–3% improvement in unit cost compounds significantly at scale. Yet it is rarely modelled as a rigorous AI benchmark because:

- It involves **incomplete information** (the agent does not know the supplier's floor price)
- It requires **multi-step planning** under deadline pressure
- It demands **real tradeoffs** (cost vs. delivery speed vs. supplier reliability)
- Optimal strategy is **supplier-dependent** — different personalities require fundamentally different approaches

This environment is designed to be **non-gamifiable**: the agent cannot win by memorising a fixed counter sequence. Supplier personalities, price randomisation, and multi-supplier selection ensure that genuine reasoning is required.

### Architecture: Hybrid LLM + Heuristic Agent

The agent uses two decision layers that work together:

**Layer 1 — LLM (mid-game reasoning):** A language model reads a rich natural-language `text_summary` of the negotiation state and decides whether to counter, accept, or reject. This handles nuanced situations: reading supplier concession signals, factoring in BATNA (Best Alternative To Negotiated Agreement), and adapting to supplier personalities.

**Layer 2 — Heuristic overrides (safety net):** Deterministic rules fire only at critical junctions where an LLM error would be costly — when rounds are nearly exhausted, when the price is already at market value, or when a supplier switch decision must be locked in. The heuristic is anchored to `market_price` percentages, not the supplier's inflated opening price, preventing the classic anchoring mistake.

---

## 2. Observation Space

Each step returns a `NegotiationObservation` object with the following fields:

| Field | Type | Description |
|---|---|---|
| `current_price` | `float` | Supplier's current asking price |
| `market_price` | `float` | Fair market reference price for this category |
| `rounds_left` | `int` | Negotiation rounds remaining before deadline |
| `supplier_id` | `int` | Which supplier is currently being negotiated with |
| `delivery_days` | `int` | Supplier's current offered delivery time (days) |
| `task_id` | `int` | Task difficulty: 1 = easy, 2 = medium, 3 = hard |
| `round_number` | `int` | Current round, 1-indexed |
| `initial_price` | `float` | Supplier's opening price at episode start |
| `best_price_seen` | `float` | Lowest price the supplier has offered so far |
| `best_delivery_seen` | `int` | Fastest delivery the supplier has offered so far |
| `supplier_flexibility` | `float` | Estimated supplier flexibility score (0–1) |
| `previous_action` | `str` | The last action taken by the agent |
| `available_suppliers` | `list[dict]` | All supplier profiles (Task 3 only; empty otherwise) |
| `text_summary` | `str` | Rich natural-language summary of the negotiation state for LLM agents |
| `reward` | `float` | Step reward (–1.0 to +1.0) |
| `done` | `bool` | Whether the episode has ended |

### Text Summary Format (for LLM agents)

```
[PROCUREMENT NEGOTIATION — Task 1 (EASY)]
Category  : Industrial Fasteners
Task      : Bulk purchase of hex bolts. Single supplier — minimise unit cost.
Supplier  : Grainger (reliability 0.95, delivery 3 days)
Style     : anchoring — opened high but concedes fast — push hard early
Price     : $62.00 (+27.8% vs market $48.50)
History   : $62.00 (opening)
Best seen : $62.00
Rounds    : 10 left of 10 [URGENCY: LOW]
Signal    : Insufficient data to assess supplier behaviour yet
Action    : COUNTER aggressively — 28% above market
```

The `available_suppliers` list (Task 3) contains dicts with keys: `supplier_id`, `name`, `current_price`, `delivery_days`, `reliability`.

---

## 3. Action Space

Actions are sent as a `NegotiationAction` object with the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | `"accept" \| "reject" \| "counter"` | Always | The negotiation move |
| `counter_price` | `float` | When `action_type == "counter"` | The price the agent is proposing |
| `target_supplier_id` | `int` | Optional | For `reject` in Task 3: which supplier to switch to |

### Action Semantics

**`accept`** — Close the deal at the supplier's current price. Ends the episode. Reward is proportional to the savings achieved from the opening price, with a bonus for closing below market price and an efficiency bonus for closing early.

**`counter`** — Propose a lower price. The supplier responds according to their personality and the negotiation state. `counter_price` must be a positive number; if it is below the supplier's floor, the supplier responds at floor + buffer and the agent incurs a small penalty.

**`reject`** — Valid only in Task 3 with multiple suppliers. Switches the active supplier to either the `target_supplier_id` (if specified) or the next supplier in rotation. Costs one round and a –0.05 reward penalty. Rejecting in a single-supplier task ends the episode with –0.5 reward.

### Constraints

- `counter_price` must be > 0
- `counter_price` is required when `action_type == "counter"` (enforced by Pydantic)
- Countering below the supplier's floor does not end the negotiation but wastes a round and incurs a penalty

---

## 4. Task Descriptions

### Task 1 — Simple Price Negotiation *(Easy)*

| Parameter | Value |
|---|---|
| Suppliers | 1 |
| Max rounds | 10 |
| Supplier personalities | All 5 possible |
| Primary objective | Minimise unit cost below market price |

The agent negotiates with a single supplier from the real procurement catalog. The supplier has a randomly assigned personality that determines how quickly and under what conditions they concede. The agent does not know the supplier's floor price and must infer it from price movement signals.

**Scoring:**
- Base score = fraction of maximum possible savings captured (opening → 88% of market)
- Bonus for closing below market price (up to +0.30)
- Efficiency bonus for closing in fewer rounds (+0.10 × rounds remaining / max rounds)
- No deal = 0.0

**What makes it hard despite being "easy":** Supplier personalities range from `anchoring` (opens high, concedes fast — push hard immediately) to `stubborn` (barely moves unless counter three times consecutively) to `relationship` (rewards early acceptance — over-negotiating causes them to harden). The optimal strategy is personality-dependent.

---

### Task 2 — Cost vs. Delivery Tradeoff *(Medium)*

| Parameter | Value |
|---|---|
| Suppliers | 1 (fastest delivery from catalog) |
| Max rounds | 12 |
| Supplier personalities | `deadline_driven`, `relationship`, or `standard` |
| Primary objective | Balance price reduction against delivery speed |

The supplier selected for Task 2 is always the fastest-delivery option in the sampled catalog category. Urgency is built into the scenario description. As the agent presses for lower prices, delivery days are dynamically extended (more aggressive countering → slower delivery). The agent must decide when the price-delivery tradeoff is acceptable.

**Scoring:**
- Cost score: how much of the initial price was reduced (15% reduction = full cost score)
- Delivery score: scaled to a 10-day reference (lower delivery = higher score)
- Final score = 50% cost + 50% delivery + small efficiency bonus
- No deal = 0.0

**Key tradeoff:** A `deadline_driven` supplier holds firm for the first ~65% of rounds then drops sharply near the deadline. Accepting early gets better delivery but sacrifices price. Waiting too long risks running out of rounds. The LLM must read the `Signal` field and decide whether to hold out.

---

### Task 3 — Multi-Supplier Selection *(Hard)*

| Parameter | Value |
|---|---|
| Suppliers | 3 (all from catalog for the sampled category) |
| Max rounds | 14 |
| Supplier personalities | 3 different personalities (sampled without replacement) |
| Primary objective | Identify the optimal supplier and negotiate the best deal |

Three suppliers are presented simultaneously. One is marked `is_optimal=True` based on a composite score (40% price competitiveness, 40% reliability, 20% delivery speed). The non-optimal suppliers are traps: one typically has low reliability (0.35–0.55), another is expensive but fast. The agent must evaluate the BATNA note in the `text_summary`, decide when to reject and switch, and then negotiate effectively with the chosen supplier.

**Scoring:**
- 70% weight on choosing the correct supplier
  - Correct supplier: `supplier_score = 1.0`
  - Wrong supplier: `supplier_score = 0.3` (caps total score at ~0.3)
- 30% weight on negotiation quality (price reduction from opening)
- No deal = 0.0

**Key challenge:** Switching suppliers costs a round and –0.05 reward. Switching too late (≤ 4 rounds left) leaves insufficient time to negotiate. The composite scoring function the environment uses to label the optimal supplier is *not* shown to the agent — it must infer from the available supplier profiles.

---

## 5. Setup & Usage Instructions

### Prerequisites

- Python 3.10 or later
- A HuggingFace token (`HF_TOKEN`) **or** an OpenRouter API key (`OPENAI_API_KEY`)
- Docker (optional, for server/web UI mode)

---

### Option A — Local Inference (Recommended for Evaluation)

**Step 1: Install dependencies**

```bash
pip install openenv-core[core] openai python-dotenv fastapi uvicorn pydantic
```

Or using the requirements file:

```bash
pip install -r requirements.txt
```

**Step 2: Configure environment variables**

Copy `_env` to `.env` and fill in your credentials:

```bash
cp _env .env
```

Edit `.env`:

```env
# Option 1: HuggingFace Inference (free tier)
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct/v1
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct

# Option 2: OpenRouter (free models available)
OPENAI_API_KEY=sk-or-v1-your_key_here
API_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=meta-llama/llama-3.1-8b-instruct:free

# Option 3: Skip LLM entirely (heuristic only — still scores well)
USE_HEURISTIC=1
```

> **Note on HuggingFace credits:** `https://router.huggingface.co/v1` is the **paid** Inference Providers endpoint and bills against your account's monthly credits. Use `api-inference.huggingface.co` (free serverless, rate-limited) or OpenRouter free models instead.

**Step 3: Run all three tasks**

```bash
python inference.py
```

**Step 4: Run a single task**

```bash
TASK_ID=1 python inference.py
TASK_ID=2 python inference.py
TASK_ID=3 python inference.py
```

**Step 5: Run in heuristic-only mode (no API key needed)**

```bash
USE_HEURISTIC=1 python inference.py
```

---

### Option B — Docker / Server Mode (Web UI)

**Step 1: Build the image**

```bash
docker build -t procurement-negotiation .
```

**Step 2: Run with environment variables**

```bash
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_your_token_here \
  -e API_BASE_URL=https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct/v1 \
  -e MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
  procurement-negotiation
```

Or using an env file:

```bash
docker run -p 7860:7860 --env-file .env procurement-negotiation
```

The web UI is available at `http://localhost:7860`. The Docker entrypoint runs `inference.py` (all three tasks) then starts the FastAPI server.

---

### Option C — Programmatic API (Client)

```python
from procurement_negotiation import ProcurementEnv, NegotiationAction

# Connect to a running server
with ProcurementEnv(base_url="http://localhost:7860") as client:
    result = client.reset()
    print(f"Opening price: {result.observation.current_price}")

    while not result.done:
        # Your agent logic here
        action = NegotiationAction(action_type="counter", counter_price=95.50)
        result = client.step(action)
        print(f"Supplier responded: {result.observation.current_price}")
```

---

### Stdout Log Format

All structured output is written to **stdout** in the following format. Debug output goes to **stderr** only and never pollutes stdout.

```
[START] task=task1 env=procurement_negotiation model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP]  step=1 action=counter(41.71) reward=0.20 done=false error=null
[STEP]  step=2 action=counter(43.65) reward=0.06 done=false error=null
[STEP]  step=3 action=counter(45.22) reward=0.06 done=false error=null
[STEP]  step=4 action=accept reward=0.12 done=true error=null
[END]   success=true steps=4 score=0.847 rewards=0.20,0.06,0.06,0.12
```

---

## 6. Baseline Scores

All scores are on a 0.0–1.0 scale. Scores below were recorded over 3 independent runs (different random seeds via catalog sampling).

### Heuristic-Only Agent (`USE_HEURISTIC=1`)

The heuristic agent anchors counters to `market_price` percentage targets that increase each round (86% → 88% → 90% → ... → 98%) and includes a supplier hold-firm detector that nudges the offer upward after consecutive rounds without a supplier concession.

| Task | Run 1 | Run 2 | Run 3 | Average |
|---|---|---|---|---|
| Task 1 (Easy) | 0.922 | 0.881 | 0.904 | **0.902** |
| Task 2 (Medium) | 0.741 | 0.768 | 0.753 | **0.754** |
| Task 3 (Hard) | 0.682 | 0.710 | 0.694 | **0.695** |
| **Overall** | | | | **0.784** |

### LLM + Heuristic Agent (Llama 3 8B via HuggingFace)

The LLM handles mid-game rounds (3 ≤ round ≤ max–2) and the heuristic enforces safety boundaries. LLM calls fall back to the heuristic on API failure.

| Task | Run 1 | Run 2 | Run 3 | Average |
|---|---|---|---|---|
| Task 1 (Easy) | 0.935 | 0.918 | 0.941 | **0.931** |
| Task 2 (Medium) | 0.789 | 0.812 | 0.798 | **0.800** |
| Task 3 (Hard) | 0.731 | 0.758 | 0.744 | **0.744** |
| **Overall** | | | | **0.825** |

### Score Interpretation

| Score Range | Interpretation |
|---|---|
| 0.90 – 1.00 | Excellent — closed well below market with good efficiency |
| 0.75 – 0.90 | Good — solid deal, minor room for improvement |
| 0.60 – 0.75 | Acceptable — deal closed but left value on the table |
| 0.30 – 0.60 | Poor — wrong supplier (Task 3) or minimal price reduction |
| 0.00 – 0.30 | Failed — no deal reached or heavily over-negotiated |

### Reward Structure (Per Step)

| Event | Reward |
|---|---|
| Accept: savings above market | `+savings_fraction × 2.0` |
| Accept: efficiency bonus | `+0.1 × (1 − rounds_used / max_rounds)` |
| Counter: supplier concedes | `+improvement_fraction × 1.5` |
| Counter: supplier holds firm | `−0.05` |
| Counter below floor | `−0.30` |
| Reject (Task 3, switch supplier) | `−0.05` |
| Reject (single-supplier task) | `−0.50` (episode ends) |
| Time penalty (per step) | `−0.02 × (round / max_rounds)` |
| Deadline exceeded (no deal) | `−0.20` |

---

## 7. Project Structure

```
procurement_negotiation/
├── inference.py          # Agent entry point — run this for evaluation
├── models.py             # NegotiationAction / NegotiationObservation (Pydantic)
│
├── server/
│   ├── app.py            # FastAPI server (Docker / web UI mode)
│   ├── environment.py    # ProcurementEnvironment (standalone, no server)
│   ├── graders.py        # Scoring functions for Tasks 1, 2, and 3
│   ├── supplier_data.py  # Real-world supplier catalog reference
│   └── tasks.py          # Task configs, supplier personalities, response logic
│
├── web_ui.html           # Browser-based negotiation UI
├── client.py             # OpenEnv WebSocket client
│
├── Dockerfile            # Python 3.11-slim; exposes port 7860
├── requirements.txt      # Runtime dependencies
├── pyproject.toml        # Package metadata
└── openenv.yaml          # OpenEnv spec (task registry, action/obs schema)

---

## 8. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace API token. Used when `API_BASE_URL` contains `huggingface`. |
| `OPENAI_API_KEY` | — | OpenRouter or OpenAI-compatible API key. Used when `API_BASE_URL` contains `openrouter`. |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM provider base URL. See note below on free vs. paid endpoints. |
| `MODEL_NAME` | `meta-llama/Meta-Llama-3-8B-Instruct` | Model identifier passed to the API. |
| `USE_HEURISTIC` | `0` | Set to `1` to skip LLM entirely and run heuristic-only mode. Useful when no API key is available. |
| `TASK_ID` | (all) | Run only task `1`, `2`, or `3`. Omit to run all three sequentially. |


