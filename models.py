# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Procurement Negotiation Environment.

B2B procurement negotiation with structured actions (accept/reject/counter),
supplier selection, and deadline pressure across 3 tasks of increasing difficulty.
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class NegotiationAction(Action):
    """Action for the Procurement Negotiation environment."""

    action_type: Literal["accept", "reject", "counter"] = Field(
        ..., description="accept the offer, reject it, or counter with a new price"
    )
    counter_price: Optional[float] = Field(
        default=None, description="required when action_type is counter"
    )
    target_supplier_id: Optional[int] = Field(
        default=None,
        description="for reject in task 3: supplier_id to switch to directly"
    )

    def model_post_init(self, __context):
        if self.action_type == "counter" and self.counter_price is None:
            raise ValueError("counter_price must be set when action_type is 'counter'")
        if self.counter_price is not None and self.counter_price <= 0:
            raise ValueError("counter_price must be positive")


class NegotiationObservation(Observation):
    """Observation from the Procurement Negotiation environment."""

    # Core negotiation state
    current_price: float = Field(..., description="Supplier's current asking price")
    market_price:  float = Field(..., description="Fair market reference price")
    rounds_left:   int   = Field(..., description="Negotiation rounds remaining")
    supplier_id:   int   = Field(..., description="Which supplier is being negotiated with")
    delivery_days: int   = Field(..., description="Supplier's offered delivery time in days")

    # Extended context
    task_id:              int   = Field(..., description="1=easy, 2=medium, 3=hard")
    round_number:         int   = Field(..., description="Current round (1-indexed)")
    initial_price:        float = Field(..., description="Supplier's opening price")
    best_price_seen:      float = Field(..., description="Lowest price offered so far")
    best_delivery_seen:   int   = Field(..., description="Fastest delivery offered so far")
    supplier_flexibility: float = Field(..., description="Estimated supplier flexibility 0-1")
    previous_action:      str   = Field(default="none", description="Last action taken by agent")

    # Multi-supplier context (task 3)
    available_suppliers: list[dict] = Field(
        default_factory=list,
        description="List of supplier profiles for task 3"
    )

    # Rich text summary for LLM agents
    text_summary: str = Field(
        default="", description="Natural language summary of negotiation state"
    )
