# -*- coding: utf-8 -*-
"""
BTC Pivot-Vegas Pyramid 回测 — 市场模式识别
依据《回测开发指令》第五节 + 《交易策略框架 v6.1》第一部分
第一版仅实现：trend_long / trend_short / tightening / tangled / breakout / transition
"""

from typing import Dict, List, Tuple

from vegas_state import get_vegas_state


def _count_swing_structure(
    swing_highs: List[Tuple[int, float]],
    swing_lows: List[Tuple[int, float]],
    min_count: int = 3,
) -> Tuple[int, int]:
    """
    bull_count = 连续 Swing Low 抬高次数（从最新往回）
    bear_count = 连续 Swing High 降低次数（从最新往回）
    """
    bull_count = 0
    for i in range(len(swing_lows) - 1, 0, -1):
        if swing_lows[i][1] > swing_lows[i - 1][1]:
            bull_count += 1
        else:
            break
    bear_count = 0
    for i in range(len(swing_highs) - 1, 0, -1):
        if swing_highs[i][1] < swing_highs[i - 1][1]:
            bear_count += 1
        else:
            break
    return bull_count, bear_count


def _pivot_breakout_direction(price: float, pivots: Dict[str, float]) -> str:
    """简化：价格相对 P 轴的位置 → 'up' / 'down' / 'range'。"""
    P = pivots.get("P")
    if P is None:
        return "range"
    S1 = pivots.get("S1") or P * 0.97
    R1 = pivots.get("R1") or P * 1.03
    if price >= R1:
        return "up"
    if price <= S1:
        return "down"
    return "range"


def get_market_mode(
    vegas_state: str,
    ema144: float,
    ema576: float,
    price: float,
    swing_highs: List[Tuple[int, float]],
    swing_lows: List[Tuple[int, float]],
    pivots: Dict[str, float],
) -> str:
    """
    在 Vegas 状态之上判定市场模式（第一版简化）。
    1. vegas_state == 'tangled' → tangled
    2. vegas_state == 'breakout' → breakout
    3. vegas_state == 'tightening' → tightening
    4. vegas_state == 'diverging'：
       - Swing 连续同向 >= 3 + Pivot 突破 → trend_long / trend_short
       - 其他 → transition
    """
    if vegas_state == "tangled":
        return "tangled"
    if vegas_state == "breakout":
        return "breakout"
    if vegas_state == "tightening":
        return "tightening"

    # diverging
    bull_count, bear_count = _count_swing_structure(swing_highs, swing_lows)
    pivot_dir = _pivot_breakout_direction(price, pivots)
    bullish = ema144 > ema576
    price_above = price > ema576
    price_below = price < ema576

    if bull_count >= 3 and (pivot_dir == "up" or price_above) and bullish:
        return "trend_long"
    if bear_count >= 3 and (pivot_dir == "down" or price_below) and not bullish:
        return "trend_short"

    return "transition"
