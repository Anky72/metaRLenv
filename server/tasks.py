from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import random


SupplierPersonality = Literal[
    "anchoring",
    "stubborn",
    "deadline_driven",
    "relationship",
    "standard",
]


@dataclass
class Supplier:
    supplier_id: int
    name: str
    base_price: float
    min_price: float
    market_price: float
    delivery_days: int
    min_delivery_days: int
    flexibility: float
    reliability_score: float
    personality: SupplierPersonality = "standard"
    category: str = "General"
    is_optimal: bool = False
    consecutive_counters: int = field(default=0, repr=False)


@dataclass
class TaskConfig:
    task_id: int
    name: str
    description: str
    max_rounds: int
    suppliers: list[Supplier]
    difficulty: str


# Real supplier catalog (used by all tasks)
REAL_PROCUREMENT_CATALOG = [
    {
        "category": "Industrial Fasteners",
        "market_price": 48.50,
        "description": "Bulk purchase of hex bolts and stainless fasteners",
        "suppliers": [
            {"name": "Grainger",    "base_price": 62.00, "min_price": 44.00, "delivery_days": 3,  "min_delivery": 2, "reliability": 0.95, "flexibility": 0.40},
            {"name": "Fastenal",    "base_price": 58.00, "min_price": 42.00, "delivery_days": 5,  "min_delivery": 3, "reliability": 0.90, "flexibility": 0.60},
            {"name": "MSC Direct",  "base_price": 71.00, "min_price": 46.00, "delivery_days": 2,  "min_delivery": 1, "reliability": 0.92, "flexibility": 0.30},
        ],
    },
    {
        "category": "Office Supplies",
        "market_price": 120.00,
        "description": "Quarterly office supplies restocking order",
        "suppliers": [
            {"name": "Staples B2B",     "base_price": 148.00, "min_price": 108.00, "delivery_days": 2, "min_delivery": 1, "reliability": 0.88, "flexibility": 0.50},
            {"name": "Amazon Business", "base_price": 135.00, "min_price": 115.00, "delivery_days": 1, "min_delivery": 1, "reliability": 0.94, "flexibility": 0.20},
            {"name": "W.B. Mason",      "base_price": 142.00, "min_price": 105.00, "delivery_days": 3, "min_delivery": 2, "reliability": 0.85, "flexibility": 0.70},
        ],
    },
    {
        "category": "Electronic Components",
        "market_price": 15.20,
        "description": "Semiconductor components for manufacturing run",
        "suppliers": [
            {"name": "Arrow Electronics", "base_price": 22.00, "min_price": 14.50, "delivery_days": 14, "min_delivery": 10, "reliability": 0.98, "flexibility": 0.20},
            {"name": "Mouser",            "base_price": 19.50, "min_price": 16.00, "delivery_days": 3,  "min_delivery": 2,  "reliability": 0.96, "flexibility": 0.40},
            {"name": "Digi-Key",          "base_price": 25.00, "min_price": 13.80, "delivery_days": 5,  "min_delivery": 3,  "reliability": 0.89, "flexibility": 0.80},
        ],
    },
    {
        "category": "Raw Steel",
        "market_price": 850.00,
        "description": "Cold rolled steel sheet stock for Q3 production",
        "suppliers": [
            {"name": "Nucor",          "base_price": 1100.00, "min_price": 820.00, "delivery_days": 25, "min_delivery": 18, "reliability": 0.96, "flexibility": 0.50},
            {"name": "U.S. Steel",     "base_price": 1050.00, "min_price": 840.00, "delivery_days": 40, "min_delivery": 30, "reliability": 0.92, "flexibility": 0.40},
            {"name": "Steel Dynamics", "base_price": 1150.00, "min_price": 790.00, "delivery_days": 15, "min_delivery": 10, "reliability": 0.87, "flexibility": 0.90},
        ],
    },
    {
        "category": "Medical Grade PPE",
        "market_price": 210.00,
        "description": "Gloves and masks for clinical facility restocking",
        "suppliers": [
            {"name": "3M Healthcare",   "base_price": 290.00, "min_price": 195.00, "delivery_days": 5, "min_delivery": 3, "reliability": 0.98, "flexibility": 0.30},
            {"name": "Honeywell",       "base_price": 275.00, "min_price": 205.00, "delivery_days": 7, "min_delivery": 5, "reliability": 0.95, "flexibility": 0.50},
            {"name": "Cardinal Health", "base_price": 260.00, "min_price": 215.00, "delivery_days": 3, "min_delivery": 2, "reliability": 0.91, "flexibility": 0.60},
        ],
    },
    {
        "category": "Packaging Materials",
        "market_price": 0.85,
        "description": "Corrugated boxes for Q4 shipping season",
        "suppliers": [
            {"name": "International Paper", "base_price": 1.20, "min_price": 0.75, "delivery_days": 6, "min_delivery": 4, "reliability": 0.93, "flexibility": 0.80},
            {"name": "WestRock",            "base_price": 1.15, "min_price": 0.80, "delivery_days": 5, "min_delivery": 3, "reliability": 0.90, "flexibility": 0.70},
            {"name": "Packaging Corp",      "base_price": 1.10, "min_price": 0.82, "delivery_days": 4, "min_delivery": 3, "reliability": 0.86, "flexibility": 0.60},
        ],
    },
    {
        "category": "HVAC Spare Parts",
        "market_price": 65.00,
        "description": "Replacement parts for facility HVAC maintenance",
        "suppliers": [
            {"name": "Carrier Enterprise", "base_price": 85.00, "min_price": 60.00, "delivery_days": 2, "min_delivery": 1, "reliability": 0.96, "flexibility": 0.40},
            {"name": "Trane Supply",       "base_price": 92.00, "min_price": 58.00, "delivery_days": 3, "min_delivery": 2, "reliability": 0.94, "flexibility": 0.50},
            {"name": "Lennox Pros",        "base_price": 80.00, "min_price": 62.00, "delivery_days": 1, "min_delivery": 1, "reliability": 0.92, "flexibility": 0.30},
        ],
    },
    {
        "category": "Chemical Solvents",
        "market_price": 340.00,
        "description": "Industrial solvent supply for manufacturing",
        "suppliers": [
            {"name": "BASF",             "base_price": 450.00, "min_price": 310.00, "delivery_days": 10, "min_delivery": 7, "reliability": 0.97, "flexibility": 0.40},
            {"name": "Dow Chemical",     "base_price": 430.00, "min_price": 325.00, "delivery_days": 12, "min_delivery": 8, "reliability": 0.94, "flexibility": 0.50},
            {"name": "Univar Solutions", "base_price": 410.00, "min_price": 335.00, "delivery_days": 4,  "min_delivery": 3, "reliability": 0.89, "flexibility": 0.70},
        ],
    },
]

PERSONALITIES: list[SupplierPersonality] = [
    "anchoring", "stubborn", "deadline_driven", "relationship", "standard"
]


def get_task(task_id: int, seed: Optional[int] = None) -> TaskConfig:
    rng = random.Random(seed)
    catalog_item = rng.choice(REAL_PROCUREMENT_CATALOG)
    market_price = catalog_item["market_price"]
    category = catalog_item["category"]
    description = catalog_item["description"]
    catalog_suppliers = catalog_item["suppliers"]

    if task_id == 1:
        raw = rng.choice(catalog_suppliers)
        personality = rng.choice(PERSONALITIES)
        supplier = Supplier(
            supplier_id=1,
            name=raw["name"],
            base_price=raw["base_price"],
            min_price=raw["min_price"],
            market_price=market_price,
            delivery_days=raw["delivery_days"],
            min_delivery_days=raw["min_delivery"],
            flexibility=raw["flexibility"],
            reliability_score=raw["reliability"],
            personality=personality,
            category=category,
            is_optimal=True,
        )
        return TaskConfig(
            task_id=1,
            name="Simple Price Negotiation",
            description=f"{description}. Single supplier — minimise unit cost.",
            max_rounds=10,
            suppliers=[supplier],
            difficulty="easy",
        )

    elif task_id == 2:
        # Pick the fastest-delivery real supplier for the urgency angle
        raw = min(catalog_suppliers, key=lambda s: s["delivery_days"])
        personality = rng.choice(["deadline_driven", "relationship", "standard"])
        supplier = Supplier(
            supplier_id=1,
            name=raw["name"],
            base_price=raw["base_price"],
            min_price=raw["min_price"],
            market_price=market_price,
            delivery_days=raw["delivery_days"],
            min_delivery_days=raw["min_delivery"],
            flexibility=raw["flexibility"],
            reliability_score=raw["reliability"],
            personality=personality,
            category=category,
            is_optimal=True,
        )
        return TaskConfig(
            task_id=2,
            name="Cost vs Delivery Tradeoff",
            description=f"{description}. Urgent requirement — balance price against delivery speed.",
            max_rounds=12,
            suppliers=[supplier],
            difficulty="medium",
        )

    elif task_id == 3:
        # Use all 3 real suppliers; rank by composite score to assign optimal
        def composite(s):
            price_s    = 1.0 - min(s["base_price"] / (market_price * 1.6), 1.0)
            rel_s      = s["reliability"]
            delivery_s = 1.0 - min(s["delivery_days"] / 20.0, 1.0)
            return 0.4 * price_s + 0.4 * rel_s + 0.2 * delivery_s

        scored = sorted(catalog_suppliers, key=composite, reverse=True)
        personalities = rng.sample(PERSONALITIES, min(3, len(PERSONALITIES)))

        suppliers = []
        for i, raw in enumerate(scored[:3]):
            suppliers.append(Supplier(
                supplier_id=i + 1,
                name=raw["name"],
                base_price=raw["base_price"],
                min_price=raw["min_price"],
                market_price=market_price,
                delivery_days=raw["delivery_days"],
                min_delivery_days=raw["min_delivery"],
                flexibility=raw["flexibility"],
                reliability_score=raw["reliability"],
                personality=personalities[i] if i < len(personalities) else "standard",
                category=category,
                is_optimal=(i == 0),   # highest composite is optimal
            ))

        return TaskConfig(
            task_id=3,
            name="Multi-Supplier Selection",
            description=f"{description}. Evaluate {len(suppliers)} suppliers on price, delivery, and reliability.",
            max_rounds=14,
            suppliers=suppliers,
            difficulty="hard",
        )

    else:
        raise ValueError(f"Unknown task_id: {task_id}. Must be 1, 2, or 3.")


def supplier_respond(
    supplier: Supplier,
    agent_counter: float,
    round_number: int,
    max_rounds: int,
    current_price: float,
) -> float:
    """
    Supplier responds based on personality.
    Concessions are deliberately small so negotiations last 5-8+ rounds.
    """
    floor = supplier.min_price
    deadline_pressure = round_number / max_rounds

    if agent_counter < floor:
        supplier.consecutive_counters += 1
        return round(floor * 1.015, 2)

    supplier.consecutive_counters += 1

    if supplier.personality == "anchoring":
        # Concedes at a moderate pace — not instantly
        effective_flex = supplier.flexibility * (1.0 + deadline_pressure * 0.6)

    elif supplier.personality == "stubborn":
        if supplier.consecutive_counters >= 3:
            effective_flex = supplier.flexibility * 1.6
            supplier.consecutive_counters = 0
        else:
            effective_flex = supplier.flexibility * 0.12

    elif supplier.personality == "deadline_driven":
        # Holds firm until 65% of rounds gone, then opens up
        if deadline_pressure < 0.65:
            effective_flex = supplier.flexibility * 0.08
        else:
            effective_flex = supplier.flexibility * (2.2 + deadline_pressure)

    elif supplier.personality == "relationship":
        # Generous in first 3 rounds, hardens after
        if round_number <= 3:
            effective_flex = supplier.flexibility * 1.1
        else:
            effective_flex = supplier.flexibility * 0.40

    else:  # standard
        effective_flex = supplier.flexibility * (0.6 + deadline_pressure * 0.5)

    # Cap at 55% of gap per round so multiple rounds are always needed
    gap = current_price - agent_counter
    concession = gap * min(effective_flex, 0.55)
    new_price = max(floor, current_price - concession)

    return round(new_price, 2)
