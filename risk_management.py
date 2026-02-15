# -*- coding: utf-8 -*-
"""
BTC 交易策略回测系统 — 止损止盈模块

实现双层止损、止盈规则、止损后重入判断。
"""

from typing import Dict, List, Optional, Tuple

from strategy import get_pivot_next_level


# ==================== 双层止损 ====================


def _get_price_vs_vegas(
    price: float,
    vegas_12h_576: float,
    vegas_12h_144: float,
) -> str:
    """
    价格相对 Vegas 576 的位置。
    Returns: 'above_576' | 'between_144_576' | 'below_576'
    """
    if price > vegas_12h_576:
        return "above_576"
    v_min, v_max = min(vegas_12h_144, vegas_12h_576), max(
        vegas_12h_144, vegas_12h_576
    )
    if v_min <= price <= v_max:
        return "between_144_576"
    return "below_576"


def calc_stop_loss(
    entry_price: float,
    direction: str,
    vegas_4h_144: float,
    vegas_12h_144: float,
    vegas_12h_576: float,
    atr: float,
    pivots: Dict[str, float],
    price: Optional[float] = None,
) -> Tuple[float, float]:
    """
    计算软止损和硬止损。

    核心原则：止损始终在 Vegas，不收紧到 MA。

    Args:
        entry_price: 入场价
        direction: 'long' / 'short'
        vegas_4h_144: 4H Vegas 144
        vegas_12h_576: 12H Vegas 576
        atr: ATR(14) 值
        pivots: Pivot 层级 dict (P, S1-S4, R1-R4)
        price: 当前价格，用于判断 price_vs_vegas

    Returns:
        (soft_stop, hard_stop) 软止损价、硬止损价
    """
    price = price or entry_price
    pos = _get_price_vs_vegas(price, vegas_12h_576, vegas_12h_144)
    pivot_next = get_pivot_next_level(pivots, direction, price)

    # v7.0 手册 6.2：软止损 ATR×1.3，硬止损 ATR×1.5（v5.2 的 ×0.3 太紧）
    atr_13 = atr * 1.3
    atr_15 = atr * 1.5
    atr_2 = atr * 2
    min_dist_pct = 0.05  # 硬止损距入场价至少 5%，作为下限保护

    if direction == "long":
        if pos == "above_576":
            soft = vegas_4h_144 - atr_13
            hard = vegas_12h_576 - atr_15
            hard_floor = entry_price * (1 - min_dist_pct)
            hard = max(hard, hard_floor)
        elif pos == "between_144_576":
            soft = vegas_12h_576 - atr_13
            hard = (pivot_next - atr_15) if pivot_next else entry_price - atr_2
            hard_floor = entry_price * (1 - min_dist_pct)
            hard = max(hard, hard_floor)
        else:  # below_576
            soft = entry_price - atr_2
            hard = (pivot_next - atr_15) if pivot_next else entry_price - atr_2
            hard_floor = entry_price * (1 - min_dist_pct)
            hard = max(hard, hard_floor)

    else:  # short
        hard_ceil = entry_price * (1 + min_dist_pct)
        if pos == "below_576":
            soft = vegas_4h_144 + atr_13
            hard = vegas_12h_576 + atr_15
            hard = max(hard, hard_ceil)
        elif pos == "between_144_576":
            soft = vegas_12h_576 + atr_13
            hard = (pivot_next + atr_15) if pivot_next else entry_price + atr_2
            hard = max(hard, hard_ceil)
        else:  # above_576 (short 在 576 上方)
            soft = entry_price + atr_2
            hard = (pivot_next + atr_15) if pivot_next else entry_price + atr_2
            hard = max(hard, hard_ceil)

    return (soft, hard)


def update_stop_loss(
    current_soft: float,
    current_hard: float,
    entry_price: float,
    direction: str,
    unrealized_r: float,
    vegas_4h_144: float,
    atr: float,
    maker_fee: float = 0.0002,
) -> Tuple[float, float]:
    """
    止损移动规则：只往有利方向移动，永不回退。

    - 浮盈 > 1R → 移到入场价 + 手续费（保本）
    - 浮盈 > 2R → max(保本价, Vegas - ATR×0.3)

    Returns:
        (new_soft, new_hard)
    """
    # v7.0: 浮盈>2R → max(保本价, 较近Vegas - ATR×1.0)
    if direction == "long":
        breakeven = entry_price * (1 + maker_fee)
        trail_stop = vegas_4h_144 - atr * 1.0

        if unrealized_r > 2:
            new_soft = max(breakeven, trail_stop)
            return (max(current_soft, new_soft), current_hard)
        if unrealized_r > 1:
            new_soft = max(current_soft, breakeven)
            return (new_soft, current_hard)

    else:  # short
        breakeven = entry_price * (1 - maker_fee)
        trail_stop = vegas_4h_144 + atr * 1.0

        if unrealized_r > 2:
            new_soft = min(breakeven, trail_stop)
            return (min(current_soft, new_soft), current_hard)
        if unrealized_r > 1:
            new_soft = min(current_soft, breakeven)
            return (new_soft, current_hard)

    return (current_soft, current_hard)


# ==================== 止盈规则 ====================


def calc_take_profit(
    wave_type: str,
    direction: str,
    pivots: Dict[str, float],
    vegas_12h_144: float,
    vegas_12h_576: float,
    entry_zone: str = "S",
) -> List[Dict]:
    """
    计算止盈目标列表。
    做多只用入场价之上的目标，做空只用入场价之下的目标，避免在回调时亏损平仓。

    类型A（576 上方）：做多 P→R1→R2 逐层减仓；做空 P→S1→S2
    类型B（576 下方）：Vegas 144→Vegas 576
    震荡市：做多 P→R1；做空 P→S1

    Returns:
        [{'price': x, 'reduce_pct': y, 'level': 'S1'}, ...]
    """
    tp_list = []
    P = pivots.get("P")
    S1 = pivots.get("S1")
    R1 = pivots.get("R1")
    R2 = pivots.get("R2")

    if wave_type == "A":  # 大波段 v7.0：S1(减20%)→P(减20%)→R1(减25%)→R2+(减25%)→尾仓10%
        S2 = pivots.get("S2")
        R2 = pivots.get("R2")
        if direction == "long":
            levels = [
                (pivots.get("S1"), 0.20, "S1"),
                (P, 0.20, "P"),
                (R1, 0.25, "R1"),
                (R2, 0.25, "R2"),
            ]
        else:
            R1_, P_, S1_, S2_ = (
                pivots.get("R1"),
                pivots.get("P"),
                pivots.get("S1"),
                pivots.get("S2"),
            )
            # 做空镜像：R1(20%)→P(20%)→S1(25%)→S2(25%)→尾仓10%
            levels = [
                (R1_, 0.20, "R1"),
                (P_, 0.20, "P"),
                (S1_, 0.25, "S1"),
                (S2_, 0.25, "S2"),
            ]
        for price, pct, name in levels:
            if price is not None:
                tp_list.append({"price": price, "reduce_pct": pct, "level": name})
        # 剩 10% 用 Vegas 追踪，由回测引擎处理

    elif wave_type == "B":  # 中等波段
        if direction == "long":
            tp_list = [
                {"price": vegas_12h_144, "reduce_pct": 0.30, "level": "vegas_144"},
                {"price": vegas_12h_576, "reduce_pct": 0.40, "level": "vegas_576"},
            ]
        else:
            tp_list = [
                {"price": vegas_12h_144, "reduce_pct": 0.30, "level": "vegas_144"},
                {"price": vegas_12h_576, "reduce_pct": 0.40, "level": "vegas_576"},
            ]

    else:  # 震荡市
        if direction == "long" and entry_zone == "S1":
            if P is not None:
                tp_list.append({"price": P, "reduce_pct": 0.50, "level": "P"})
            if R1 is not None:
                tp_list.append({"price": R1, "reduce_pct": 1.0, "level": "R1"})
        elif direction == "short" and entry_zone == "R1":
            if P is not None:
                tp_list.append({"price": P, "reduce_pct": 0.50, "level": "P"})
            if S1 is not None:
                tp_list.append({"price": S1, "reduce_pct": 1.0, "level": "S1"})
        else:
            if P is not None:
                tp_list.append({"price": P, "reduce_pct": 0.50, "level": "P"})

    return tp_list


def get_trailing_stop_price(
    ma55_4h_value: float,
    direction: str,
) -> float:
    """
    v7.0 尾仓追踪：历史最高 4H MA55（做多只升不降）/ 历史最低 4H MA55（做空只降不升）。
    由回测引擎维护「历史极值」，此处仅返回当前用于比较的 MA55 值。
    4H 收盘 < 跟踪止损(多) 或 > 跟踪止损(空) → 平尾仓。
    """
    return ma55_4h_value


# ==================== 止损后重入 ====================


def check_reentry(
    last_stop_reason: str,
    trend_continues: bool,
    new_score: int,
) -> bool:
    """
    止损后重入检测。无冷静期，立即检测。

    Args:
        last_stop_reason: 'soft' / 'hard' / ''
        trend_continues: 趋势是否延续
        new_score: 新入场信号评分

    Returns:
        True 表示可立即重入（首仓大小 20%）
    """
    if trend_continues and new_score >= 3:
        return True
    return False