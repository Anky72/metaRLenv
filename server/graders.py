from __future__ import annotations


def grade_task1(
    final_price: float,
    market_price: float,
    initial_price: float,
    rounds_used: int,
    max_rounds: int,
    accepted: bool,
) -> dict:
    """
    Task 1 — Simple price negotiation (EASY).

    Score = fraction of possible savings captured, from opening price
    down to supplier floor (~88% of market). Bonuses for closing below
    market and closing efficiently.
    """
    if not accepted:
        return {"score": 0.0, "breakdown": {
            "negotiation_score": 0.0, "efficiency_bonus": 0.0,
            "accepted": False, "note": "No deal reached"
        }}

    max_possible_saving = initial_price - (market_price * 0.88)
    actual_saving       = initial_price - final_price

    negotiation_score = (
        max(0.0, min(1.0, actual_saving / max_possible_saving))
        if max_possible_saving > 0 else 0.5
    )

    below_market_bonus = max(0.0, (market_price - final_price) / market_price * 0.15)
    efficiency_bonus   = 0.05 * (1.0 - rounds_used / max_rounds)
    score              = min(1.0, negotiation_score + below_market_bonus + efficiency_bonus)

    return {
        "score": round(score, 4),
        "breakdown": {
            "negotiation_score":   round(negotiation_score, 4),
            "below_market_bonus":  round(below_market_bonus, 4),
            "efficiency_bonus":    round(efficiency_bonus, 4),
            "accepted":            True,
            "price_reduction_pct": round((initial_price - final_price) / initial_price * 100, 2),
        }
    }


def grade_task2(
    final_price: float,
    initial_price: float,
    final_delivery_days: int,
    rounds_used: int,
    max_rounds: int,
    accepted: bool,
) -> dict:
    """
    Task 2 — Cost vs delivery tradeoff (MEDIUM).

    cost_score:     relative price reduction from initial (not a hardcoded
                    dollar reference — market_price varies $110-$140 so a
                    fixed $120 anchor would be unfair to either end).
    delivery_score: 10-day reference is fair since task2 delivery starts 2-4 days.
    Weighted 50/50, plus small efficiency bonus.
    """
    if not accepted:
        return {"score": 0.0, "breakdown": {
            "cost_score": 0.0, "delivery_score": 0.0,
            "accepted": False, "note": "No deal reached"
        }}

    # Cost score: how much % was saved from opening price
    # 15% reduction = full cost score (excellent negotiation for a 10-round task)
    max_reduction = initial_price * 0.15
    actual_reduction = initial_price - final_price
    cost_score = max(0.0, min(1.0, actual_reduction / max_reduction))

    # Delivery score: scaled to 10-day reference (task2 suppliers open at 2-4 days)
    delivery_score = max(0.0, min(1.0, (10.0 - final_delivery_days) / 10.0))

    efficiency_bonus = 0.05 * (1.0 - rounds_used / max_rounds)
    score = min(1.0, 0.5 * cost_score + 0.5 * delivery_score + efficiency_bonus)

    return {
        "score": round(score, 4),
        "breakdown": {
            "cost_score":        round(cost_score, 4),
            "delivery_score":    round(delivery_score, 4),
            "efficiency_bonus":  round(efficiency_bonus, 4),
            "accepted":          True,
            "final_price":       round(final_price, 2),
            "price_reduction_pct": round(actual_reduction / initial_price * 100, 2),
            "delivery_days":     final_delivery_days,
        }
    }


def grade_task3(
    chosen_supplier_id: int,
    optimal_supplier_id: int,
    final_price: float,
    initial_price: float,
    market_price: float,
    final_delivery_days: int,
    accepted: bool,
) -> dict:
    """
    Task 3 — Multi-supplier selection (HARD).

    70% weight on choosing the correct supplier (the one with best
    reliability + reasonable price + acceptable delivery).
    30% weight on negotiation quality with chosen supplier.
    """
    if not accepted:
        return {"score": 0.0, "breakdown": {
            "supplier_score": 0.0, "price_score": 0.0,
            "accepted": False, "note": "No deal reached"
        }}

    supplier_score = 1.0 if chosen_supplier_id == optimal_supplier_id else 0.3

    max_saving  = initial_price - market_price * 0.85
    actual_saving = initial_price - final_price
    price_score = (
        max(0.0, min(1.0, actual_saving / max_saving))
        if max_saving > 0 else 0.5
    )

    score = min(1.0, 0.7 * supplier_score + 0.3 * price_score)

    return {
        "score": round(score, 4),
        "breakdown": {
            "supplier_score":      round(supplier_score, 4),
            "price_score":         round(price_score, 4),
            "correct_supplier":    chosen_supplier_id == optimal_supplier_id,
            "chosen_supplier_id":  chosen_supplier_id,
            "optimal_supplier_id": optimal_supplier_id,
            "accepted":            True,
        }
    }


def grade_episode(task_id: int, episode_data: dict) -> dict:
    """Unified grader entry point — called at episode end by environment."""
    if task_id == 1:
        return grade_task1(
            final_price=episode_data["final_price"],
            market_price=episode_data["market_price"],
            initial_price=episode_data.get("initial_price", episode_data["final_price"] * 1.2),
            rounds_used=episode_data["rounds_used"],
            max_rounds=episode_data["max_rounds"],
            accepted=episode_data["accepted"],
        )
    elif task_id == 2:
        return grade_task2(
            final_price=episode_data["final_price"],
            initial_price=episode_data.get("initial_price", episode_data["final_price"] * 1.2),
            final_delivery_days=episode_data["final_delivery_days"],
            rounds_used=episode_data["rounds_used"],
            max_rounds=episode_data["max_rounds"],
            accepted=episode_data["accepted"],
        )
    elif task_id == 3:
        return grade_task3(
            chosen_supplier_id=episode_data["chosen_supplier_id"],
            optimal_supplier_id=episode_data["optimal_supplier_id"],
            final_price=episode_data["final_price"],
            initial_price=episode_data.get("initial_price", episode_data["final_price"] * 1.2),
            market_price=episode_data["market_price"],
            final_delivery_days=episode_data["final_delivery_days"],
            accepted=episode_data["accepted"],
        )
    else:
        raise ValueError(f"Unknown task_id: {task_id}")
