"""Position sizing: hard cap per ticker, skip names that cannot buy 1 share."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


def size_overnight_picks(
    picks: Sequence[Dict[str, Any]],
    cash: float,
    max_alloc: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    """Split cash equally, then cap at max_alloc. Drop names with qty == 0.

    Policy for price > max_alloc: skip (1-share would breach the 100k cap).
    Returns (sized_picks, skipped, alloc_per_stock).
    """
    skipped: List[Dict[str, Any]] = []
    affordable: List[Dict[str, Any]] = []
    cap = max(0.0, float(max_alloc))
    for p in picks:
        price = float(p.get("close_price") or 0.0)
        if price <= 0:
            skipped.append({**p, "skip_reason": "invalid_price"})
        elif price > cap:
            skipped.append({**p, "skip_reason": "price_gt_max_alloc"})
        else:
            affordable.append(p)

    if not affordable or cash <= 0:
        return [], skipped, 0.0

    alloc = min(cash / len(affordable), cap)
    sized: List[Dict[str, Any]] = []
    for p in affordable:
        price = float(p["close_price"])
        qty = math.floor(alloc / price)
        if qty <= 0:
            skipped.append({**p, "skip_reason": "qty_zero"})
            continue
        buy_amount = qty * price
        sized.append({
            **p,
            "buy_qty": int(qty),
            "buy_amount": buy_amount,
            "alloc": alloc,
        })
    return sized, skipped, alloc
