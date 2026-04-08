# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Procurement Negotiation Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import NegotiationAction, NegotiationObservation


class ProcurementEnv(
    EnvClient[NegotiationAction, NegotiationObservation, State]
):
    """
    Client for the Procurement Negotiation Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with ProcurementEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.current_price)
        ...
        ...     result = client.step(NegotiationAction(action_type="counter", counter_price=95.50))
        ...     print(result.observation.current_price)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = ProcurementEnv.from_docker_image("procurement-negotiation:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(NegotiationAction(action_type="accept"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: NegotiationAction) -> Dict:
        """Convert NegotiationAction to JSON payload for step message."""
        payload = {"action_type": action.action_type}
        if action.counter_price is not None:
            payload["counter_price"] = action.counter_price
        if action.target_supplier_id is not None:
            payload["target_supplier_id"] = action.target_supplier_id
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[NegotiationObservation]:
        """Parse server response into StepResult[NegotiationObservation]."""
        obs_data = payload.get("observation", {})
        observation = NegotiationObservation(
            current_price=obs_data.get("current_price", 0.0),
            market_price=obs_data.get("market_price", 0.0),
            rounds_left=obs_data.get("rounds_left", 0),
            supplier_id=obs_data.get("supplier_id", 1),
            delivery_days=obs_data.get("delivery_days", 0),
            task_id=obs_data.get("task_id", 1),
            round_number=obs_data.get("round_number", 0),
            initial_price=obs_data.get("initial_price", 0.0),
            best_price_seen=obs_data.get("best_price_seen", 0.0),
            best_delivery_seen=obs_data.get("best_delivery_seen", 999),
            supplier_flexibility=obs_data.get("supplier_flexibility", 0.0),
            previous_action=obs_data.get("previous_action", "none"),
            available_suppliers=obs_data.get("available_suppliers", []),
            text_summary=obs_data.get("text_summary", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
