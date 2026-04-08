# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Procurement Negotiation Environment Implementation.

B2B procurement negotiation across 3 tasks of increasing difficulty:
  Task 1 (easy):   Single supplier price negotiation
  Task 2 (medium): Cost vs delivery tradeoff
  Task 3 (hard):   Multi-supplier selection and negotiation
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import NegotiationAction, NegotiationObservation
except ImportError:
    from models import NegotiationAction, NegotiationObservation

from .tasks import get_task, supplier_respond
from .graders import grade_episode


class ProcurementEnvironment(Environment):
    """
    B2B Procurement Negotiation Environment.

    The agent acts as a procurement manager negotiating with suppliers
    on price, delivery speed, and supplier selection.

    Example:
        >>> env = ProcurementEnvironment(task_id=1)
        >>> obs = env.reset()
        >>> print(obs.current_price)
        >>>
        >>> from models import NegotiationAction
        >>> obs = env.step(NegotiationAction(action_type="counter", counter_price=95.50))
        >>> print(obs.current_price)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: int = 1, seed: int | None = None):
        super().__init__()  # required by base Environment class
        self.task_id = task_id
        self.seed = seed
        self._task = None
        self._state = None
        self._current_supplier = None
        self._current_price = 0.0
        self._current_delivery = 0
        self._round = 0
        self._accepted = False
        self._rejected = False
        self._best_price = 0.0
        self._best_delivery = 999
        self._chosen_supplier_id = 1
        self._episode_log = []
        self._price_history = []

    # ── RESET ─────────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, episode_id: str | None = None, task_id: int | None = None, **kwargs) -> NegotiationObservation:
        if task_id is not None:
            self.task_id = task_id

        self._task = get_task(self.task_id, seed=self.seed)
        self._current_supplier = self._task.suppliers[0]
        self._current_price = self._current_supplier.base_price
        self._current_delivery = self._current_supplier.delivery_days
        self._round = 0
        self._accepted = False
        self._rejected = False
        self._best_price = self._current_price
        self._best_delivery = self._current_delivery
        self._chosen_supplier_id = self._current_supplier.supplier_id
        self._episode_log = []
        self._price_history = [self._current_price]

        self._state = State(
            episode_id=episode_id or f"proc-{uuid4().hex[:8]}",
            step_count=0,
            metadata={"task_id": self.task_id, "grader_result": None},
        )
        return self._build_obs(0.0, False, "none")

    # ── STEP ──────────────────────────────────────────────────────────────────
    
    def step(self, action: NegotiationAction, timeout_s: float | None = None, **kwargs) -> NegotiationObservation:
        if self._task is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        self._round += 1
        if self._state:
            self._state.step_count = self._round

        reward = 0.0
        done = False

        if action.action_type == "accept":
            self._accepted = True
            done = True
            savings = self._current_supplier.market_price - self._current_price
            reward = max(0.0, savings / self._current_supplier.market_price) * 2.0
            efficiency = 1.0 - self._round / self._task.max_rounds
            reward += 0.1 * efficiency

        elif action.action_type == "reject":
            if self.task_id == 3 and len(self._task.suppliers) > 1:
                target_id = getattr(action, "target_supplier_id", None)
                next_supplier = None
                if target_id is not None:
                    next_supplier = next(
                        (s for s in self._task.suppliers
                         if s.supplier_id == target_id
                         and s.supplier_id != self._current_supplier.supplier_id),
                        None
                    )
                if next_supplier is None:
                    current_idx = next(
                        (i for i, s in enumerate(self._task.suppliers)
                         if s.supplier_id == self._current_supplier.supplier_id), 0
                    )
                    next_idx = (current_idx + 1) % len(self._task.suppliers)
                    next_supplier = self._task.suppliers[next_idx]
                self._current_supplier = next_supplier
                self._chosen_supplier_id = self._current_supplier.supplier_id
                self._current_price = self._current_supplier.base_price
                self._current_delivery = self._current_supplier.delivery_days
                self._price_history = [self._current_price]
                self._round = 0
                reward = -0.05
                done = False
            else:
                self._rejected = True
                done = True
                reward = -0.5

        elif action.action_type == "counter":
            counter = action.counter_price
            if counter is None or counter <= 0:
                reward = -0.3
            else:
                new_price = supplier_respond(
                    supplier=self._current_supplier,
                    agent_counter=counter,
                    round_number=self._round,
                    max_rounds=self._task.max_rounds,
                    current_price=self._current_price,
                )
                old_price = self._current_price
                self._current_price = new_price
                self._price_history.append(new_price)

                if new_price < old_price:
                    improvement = (old_price - new_price) / old_price
                    reward = improvement * 1.5
                    self._best_price = min(self._best_price, new_price)
                else:
                    reward = -0.05

                if self.task_id == 2:
                    urgency = max(0, (self._current_supplier.base_price - new_price)
                                  / self._current_supplier.base_price)
                    self._current_delivery = max(
                        self._current_supplier.min_delivery_days,
                        int(self._current_supplier.delivery_days * (1 - urgency * 0.3))
                    )
                    if self._current_delivery < self._best_delivery:
                        self._best_delivery = self._current_delivery

        rounds_left = self._task.max_rounds - self._round
        if rounds_left <= 0 and not done:
            done = True
            reward -= 0.2

        reward -= 0.02 * (self._round / self._task.max_rounds)

        self._episode_log.append({
            "round": self._round,
            "action": action.action_type,
            "counter_price": action.counter_price,
            "supplier_price": self._current_price,
            "reward": reward,
        })

        if done and self._state:
            optimal_id = next(
                (s.supplier_id for s in self._task.suppliers if s.is_optimal), 1
            )
            self._state.metadata["grader_result"] = grade_episode(
                self.task_id,
                {
                    "final_price":         self._current_price,
                    "market_price":        self._current_supplier.market_price,
                    "initial_price":       self._current_supplier.base_price,
                    "rounds_used":         self._round,
                    "max_rounds":          self._task.max_rounds,
                    "accepted":            self._accepted,
                    "final_delivery_days": self._current_delivery,
                    "chosen_supplier_id":  self._chosen_supplier_id,
                    "optimal_supplier_id": optimal_id,
                }
            )

        return self._build_obs(reward, done, action.action_type)

    # ── STATE ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Call reset() first.")
        return self._state

    # ── BUILD OBS ─────────────────────────────────────────────────────────────

    def _build_obs(self, reward, done, previous_action) -> NegotiationObservation:
        assert self._task and self._current_supplier

        rounds_left = max(0, self._task.max_rounds - self._round)
        market = self._current_supplier.market_price
        price_gap_pct = (self._current_price - market) / market * 100
        urgency = "HIGH" if rounds_left <= 2 else "MEDIUM" if rounds_left <= 5 else "LOW"

        if len(self._price_history) >= 3:
            recent_drops = [
                self._price_history[i-1] - self._price_history[i]
                for i in range(1, len(self._price_history))
            ]
            avg_drop = sum(recent_drops) / len(recent_drops)
            if avg_drop > self._current_price * 0.03:
                supplier_signal = "Supplier is conceding quickly — may have opened high"
            elif avg_drop < self._current_price * 0.005:
                supplier_signal = "Supplier is holding firm — push harder or wait for deadline"
            else:
                supplier_signal = "Supplier conceding moderately — steady negotiation working"
        else:
            supplier_signal = "Insufficient data to assess supplier behaviour yet"

        batna_note = ""
        if self.task_id == 3:
            others = [s for s in self._task.suppliers
                      if s.supplier_id != self._current_supplier.supplier_id]
            if others:
                best_alt = min(others, key=lambda s: s.base_price)
                batna_note = (
                    f"Best alternative: {best_alt.name} at ${best_alt.base_price:.2f} "
                    f"(reliability {best_alt.reliability_score:.2f}). "
                )

        personality_hints = {
            "anchoring":       "opened high but concedes fast — push hard early",
            "stubborn":        "barely moves — counter 3x in a row to unlock concession",
            "deadline_driven": "holds firm early, drops sharply in last 2 rounds — be patient",
            "relationship":    "rewards early acceptance — don't over-negotiate or they'll harden",
            "standard":        "balanced concessions — steady pressure works",
        }
        personality = self._current_supplier.personality
        personality_hint = personality_hints.get(personality, "standard behaviour")

        if len(self._price_history) > 1:
            recent = self._price_history[-4:]
            history_str = " → ".join(f"${p:.2f}" for p in recent)
            if len(self._price_history) > 4:
                history_str = "... → " + history_str
        else:
            history_str = f"${self._current_price:.2f} (opening)"

        if price_gap_pct < 2:
            recommended = "ACCEPT — price is at or near market value"
        elif rounds_left <= 2:
            recommended = f"ACCEPT or FINAL COUNTER — only {rounds_left} round(s) left"
        elif price_gap_pct > 15:
            recommended = f"COUNTER aggressively — {price_gap_pct:.0f}% above market"
        else:
            recommended = f"COUNTER steadily — {price_gap_pct:.0f}% above market"

        text_summary = (
            f"[PROCUREMENT NEGOTIATION — Task {self.task_id} ({self._task.difficulty.upper()})]\n"
            f"Category  : {self._current_supplier.category}\n"
            f"Task      : {self._task.description}\n"
            f"Supplier  : {self._current_supplier.name} "
            f"(reliability {self._current_supplier.reliability_score:.2f}, "
            f"delivery {self._current_delivery} days)\n"
            f"Style     : {personality} — {personality_hint}\n"
            f"Price     : ${self._current_price:.2f} "
            f"({price_gap_pct:+.1f}% vs market ${market:.2f})\n"
            f"History   : {history_str}\n"
            f"Best seen : ${self._best_price:.2f}\n"
            f"Rounds    : {rounds_left} left of {self._task.max_rounds} [URGENCY: {urgency}]\n"
            f"Signal    : {supplier_signal}\n"
            f"{batna_note}"
            f"Action    : {recommended}"
        )

        available_suppliers = [
            {
                "supplier_id":   s.supplier_id,
                "name":          s.name,
                "current_price": (
                    self._current_price
                    if s.supplier_id == self._current_supplier.supplier_id
                    else s.base_price
                ),
                "delivery_days": s.delivery_days,
                "reliability":   s.reliability_score,
            }
            for s in self._task.suppliers
        ]

        return NegotiationObservation(
            reward=reward,
            done=done,
            current_price=self._current_price,
            market_price=market,
            rounds_left=rounds_left,
            supplier_id=self._current_supplier.supplier_id,
            delivery_days=self._current_delivery,
            task_id=self.task_id,
            round_number=self._round,
            initial_price=self._current_supplier.base_price,
            best_price_seen=self._best_price,
            best_delivery_seen=self._best_delivery,
            supplier_flexibility=self._current_supplier.flexibility,
            previous_action=previous_action,
            available_suppliers=available_suppliers if self.task_id == 3 else [],
            text_summary=text_summary,
        )
