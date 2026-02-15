# -*- coding: utf-8 -*-
"""
BTC 交易策略回测系统 — 策略逻辑模块

实现宏观结构定位、波段类型判断、市场模式识别、入场评分制、
金字塔建仓、动态杠杆计算。
"""

from typing import Dict, List, Optional, Tuple

from config import (
    LEVERAGE_TABLE,
    PYRAMID_SIZE,
    SCORE_THRESHOLD,
    TIERS,
    VOL_MULT,
)
from indicators import detect_cross


# ==================== Vegas 状态机（v7.0 仓位调节器） ====================

VEGAS_STATE_MULT = {
    "diverge": 1.0,
    "tighten": 0.7,
    "tangle": 0.5,
    "breakout": 0.0,
}


def get_vegas_state(
    ema144_12h: float,
    ema576_12h: float,
    cross_count_30: int = 0,
    just_crossed: Optional[str] = None,
) -> str:
    """
    Vegas 状态机。距离 = abs(EMA144 - EMA576) / EMA576 × 100。

    Returns:
        'diverge': 距离>8%，正常交易
        'tighten': 距离≤8% 无交叉，仓位×0.7
        'tangle': 30根12H内交叉≥3次，仓位×0.5
        'breakout': 144穿越576，平趋势方向仓，不开新仓(state_mult=0)
    """
    if ema576_12h is None or ema576_12h <= 0:
        return "diverge"
    dist_pct = abs(ema144_12h - ema576_12h) / ema576_12h * 100

    if just_crossed in ("golden", "death"):
        return "breakout"
    if dist_pct > 8:
        return "diverge"
    if cross_count_30 >= 3:
        return "tangle"
    return "tighten"


def get_state_mult(vegas_state: str) -> float:
    """状态乘数，用于仓位上限和杠杆。"""
    return VEGAS_STATE_MULT.get(vegas_state, 1.0)


# ==================== 宏观结构定位 ====================


def get_macro_season(
    weekly_close: float,
    vegas_w144: float,
    vegas_w169: float,
    prev_season: Optional[str] = None,
    weekly_close_prev: Optional[float] = None,
    vegas_w144_prev: Optional[float] = None,
    vegas_w169_prev: Optional[float] = None,
) -> str:
    """
    判断牛熊季节。

    Returns:
        'summer': 牛市，禁止左侧做空
        'winter': 熊市，禁止左侧做多
        'transition': 转换期，双向需右侧确认，仓位缩小50%
    """
    # 当前状态
    above_144 = weekly_close > vegas_w144
    above_169 = weekly_close > vegas_w169
    ema_bull = vegas_w144 > vegas_w169  # 向上发散
    ema_bear = vegas_w144 < vegas_w169  # 向下发散

    if above_144 and ema_bull:
        curr_raw = "summer"
    elif not above_169 and ema_bear:
        curr_raw = "winter"
    else:
        curr_raw = "transition"

    # 升级/降级需连续2周确认
    if prev_season and weekly_close_prev is not None:
        prev_below_169 = weekly_close_prev < vegas_w169_prev
        prev_above_144 = weekly_close_prev > vegas_w144_prev

        if prev_season == "summer" and prev_below_169 and not above_169:
            return "winter"  # 牛→熊：连续2周在169下方
        if prev_season == "winter" and prev_above_144 and above_144:
            return "summer"  # 熊→牛：连续2周在144上方

    return curr_raw


# ==================== 波段类型判断 ====================


def get_wave_type(close_12h: float, vegas_12h_576: float) -> str:
    """
    类型A: 价格 > 12H Vegas 576 → 大波段（50-100%+）
    类型B: 价格 < 12H Vegas 576 → 中等波段（20-40%）
    """
    return "A" if close_12h > vegas_12h_576 else "B"


# ==================== 市场模式识别 ====================


def _count_swing_structure(
    swing_highs: List[Tuple[int, float]], swing_lows: List[Tuple[int, float]]
) -> Tuple[int, int]:
    """
    从最新的 Swing 往回数，统计"最近"的连续趋势。
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


def _get_pivot_behavior(
    price: float,
    pivots: Dict[str, float],
) -> str:
    """
    Pivot 箱体行为：'breakout_up' / 'breakout_down' / 'range_bound' / 'unclear'
    2023 H1 BTC 25k-30k 等震荡：价格在 S1~R1 区间内视为 range_bound
    """
    P = pivots.get("P")
    S1 = pivots.get("S1")
    R1 = pivots.get("R1")
    if P is None:
        return "unclear"
    S1 = S1 if S1 is not None else P * 0.97
    R1 = R1 if R1 is not None else P * 1.03

    # 价格在 S1~R1 区间内视为 range_bound（覆盖 25k-30k 等宽幅震荡）
    if S1 <= price <= R1:
        return "range_bound"

    if price > R1:
        return "breakout_up"
    if price < S1:
        return "breakout_down"

    # 价格在 P 与 R1 之间：若明显偏向 R1 可视为 breakout_up
    if price > P:
        return "breakout_up"
    return "breakout_down"


def get_market_mode(
    swing_highs: List[Tuple[int, float]],
    swing_lows: List[Tuple[int, float]],
    price: float,
    pivots: Dict[str, float],
    vegas_12h_144: float,
    vegas_12h_169: float,
    vegas_12h_576: float,
    vegas_12h_676: float,
) -> str:
    """
    判断市场模式。放宽趋势识别以更好捕捉 2023-2024 大牛市。

    Returns:
        'trend_long': 趋势（多）
        'trend_short': 趋势（空）
        'range': 震荡
        'transition': 过渡
        'no_man_land': 无人区
    """
    bull_count, bear_count = _count_swing_structure(swing_highs, swing_lows)
    vegas_bullish = vegas_12h_144 > vegas_12h_169
    vegas_bearish = vegas_12h_144 < vegas_12h_169
    price_above_576 = price > vegas_12h_576
    price_below_576 = price < vegas_12h_576
    structure_downtrend = (bear_count >= 2 and bull_count < 2) or (
        price_below_576 and vegas_bearish
    )

    vegas_list = [v for v in [vegas_12h_144, vegas_12h_169, vegas_12h_576, vegas_12h_676] if v and v > 0]
    min_dist_pct = min(abs(price - v) / price for v in vegas_list) * 100 if vegas_list else 0
    # 明显下跌趋势时，即使价格远离 Vegas 也判 trend_short（避免 no_man_land 阻挡做空）
    if min_dist_pct > 10 and not structure_downtrend:
        return "no_man_land"

    pivot_behavior = _get_pivot_behavior(price, pivots)

    # 放宽趋势确认：bull_count>=2 或 (价格>576 且 Vegas 向上发散)
    structure_uptrend = (bull_count >= 2 and bear_count < 2) or (
        price_above_576 and vegas_bullish
    )
    structure_unclear = bull_count < 2 and bear_count < 2

    if structure_uptrend and (pivot_behavior == "breakout_up" or price_above_576):
        return "trend_long"
    if structure_downtrend and (pivot_behavior == "breakout_down" or price_below_576):
        return "trend_short"
    if structure_unclear and pivot_behavior == "range_bound":
        return "range"

    return "transition"


# ==================== 入场评分制 ====================

# 趋势市评分（v7.0：新增 Vegas 576/676 附近±3% +2，前月阻力共振 +1）
SCORE_TREND = {
    "pivot_sr": 1,
    "pin_bar": 2,
    "engulfing": 2,
    "ma8_cross": 1,
    "ma_cross": 2,
    "rsi_extreme": 1,
    "three_same_candle": 1,
    "ma12h_cross": 3,
    "vegas_break": 3,
    "vegas_576_near": 2,
    "prev_month_resonance": 1,
}

# 震荡市评分（无叉点）
SCORE_RANGE = {
    "pivot_sr": 1,
    "pin_bar": 2,
    "engulfing": 2,
    "ma8_cross": 1,
    "rsi_extreme": 1,
    "three_same_candle": 1,
}


def calc_entry_score(
    mode: str,
    signals: Dict[str, bool],
    direction: str,
    use_cross: bool = True,
) -> int:
    """
    计算入场评分。

    Args:
        mode: 'trend_long' / 'trend_short' / 'range' / ...
        signals: {'pivot_sr': True, 'pin_bar': True, ...}
        direction: 'long' / 'short'
        use_cross: 是否使用叉点评分（震荡市为 False）

    Returns:
        总分
    """
    is_trend = mode in ("trend_long", "trend_short")
    score_map = SCORE_TREND if is_trend else SCORE_RANGE

    total = 0
    for key, active in (signals or {}).items():
        if not active or key not in score_map:
            continue
        if not use_cross and key in ("ma_cross", "ma12h_cross"):
            continue
        total += score_map[key]

    return total


def check_entry_filters(
    ma_entangled: bool,
    season: str,
    direction: str,
    is_left: bool = True,
) -> bool:
    """
    入场过滤器，必须全部通过（v7.0）。
    1. MA缠绕不入场 2. 季节过滤。Vegas距离>15% 由 check_vegas_distance_filter 在引擎中单独检查。
    """
    if ma_entangled:
        return False
    if season == "winter" and direction == "long":
        return False
    if season == "summer" and direction == "short":
        return False
    return True


def check_vegas_distance_filter(
    price: float,
    vegas_12h_576: float,
    direction: str,
) -> bool:
    """
    v7.0: 价格距 12H EMA576 > 15% 且方向一致 → 不新开仓。
    Returns True 表示通过（可以开仓），False 表示不新开仓。
    """
    if vegas_12h_576 is None or vegas_12h_576 <= 0:
        return True
    dist_pct = abs(price - vegas_12h_576) / vegas_12h_576 * 100
    if dist_pct <= 15:
        return True
    # 方向一致：做多且价格远高于576 / 做空且价格远低于576
    if direction == "long" and price > vegas_12h_576:
        return False
    if direction == "short" and price < vegas_12h_576:
        return False
    return True


# ==================== 金字塔建仓 ====================

# 持续性信号评分（加仓用，不会像瞬时信号那样下一根就消失）
# 价格在MA8上方+1、MA17上方+1、Vegas144上方+2、趋势未变+2、首仓盈利>1R+2、RSI40-70+1
PERSISTENT_SCORE = {
    "ma8_above": 1,      # 价格>MA8（持续性）
    "ma17_above": 1,     # 价格>MA17（持续性）
    "vegas144_above": 2, # 价格>Vegas144（持续性）
    "trend_unchanged": 2,# 趋势模式未变（持续性）
    "first_layer_profit_1r": 2,  # 首仓浮盈>1R（持续性）
    "rsi_healthy": 1,    # RSI在40-70（健康趋势）
}


def calc_persistent_score(
    direction: str,
    price: float,
    ma8: float,
    ma17: float,
    vegas_144: float,
    mode: str,
    first_layer_unrealized_r: float,
    rsi: Optional[float] = None,
) -> int:
    """加仓用持续性评分，层2>=4、层3>=6"""
    score = 0
    if direction == "long":
        if ma8 and price > ma8:
            score += PERSISTENT_SCORE["ma8_above"]
        if ma17 and price > ma17:
            score += PERSISTENT_SCORE["ma17_above"]
        if vegas_144 and price > vegas_144:
            score += PERSISTENT_SCORE["vegas144_above"]
    else:  # short
        if ma8 and price < ma8:
            score += PERSISTENT_SCORE["ma8_above"]
        if ma17 and price < ma17:
            score += PERSISTENT_SCORE["ma17_above"]
        if vegas_144 and price < vegas_144:
            score += PERSISTENT_SCORE["vegas144_above"]

    if mode in ("trend_long", "trend_short"):
        score += PERSISTENT_SCORE["trend_unchanged"]
    if first_layer_unrealized_r > 1:
        score += PERSISTENT_SCORE["first_layer_profit_1r"]
    if rsi is not None and 40 <= rsi <= 70:
        score += PERSISTENT_SCORE["rsi_healthy"]
    return score


def pyramid_entry_persistent(
    mode: str,
    persistent_score: int,
    current_layers: int,
    direction: str,
) -> Optional[Tuple[int, float]]:
    """金字塔加仓：使用持续性评分，层2>=4、层3>=6、层4>=8"""
    if mode not in ("trend_long", "trend_short"):
        return None
    next_layer = current_layers + 1
    if next_layer > 4:
        return None
    thresh = {2: 4, 3: 6, 4: 8}.get(next_layer, 999)
    if persistent_score < thresh:
        return None
    from config import PYRAMID_SIZE
    size = PYRAMID_SIZE.get(f"layer{next_layer}", 0.0)
    return (next_layer, size)


def get_pyramid_layer_size(layer: int) -> float:
    """获取金字塔各层仓位占比"""
    key = f"layer{layer}"
    return PYRAMID_SIZE.get(key, 0.0)


def pyramid_entry(
    mode: str,
    score: int,
    current_layers: int,
    direction: str,
) -> Optional[Tuple[int, float]]:
    """
    金字塔建仓决策。

    Returns:
        (next_layer, size_pct) 或 None（不加仓）
    """
    if mode not in ("trend_long", "trend_short"):
        # 震荡市：固定 20-30%，不加仓
        if current_layers == 0 and score >= SCORE_THRESHOLD["range"]["entry"]:
            return (1, 0.25)  # 固定 25%
        return None

    # 趋势市
    thresholds = SCORE_THRESHOLD["trend"]
    next_layer = current_layers + 1

    if next_layer > 4:
        return None

    required = thresholds.get(f"layer{next_layer}", 999)
    if score < required:
        return None

    size = get_pyramid_layer_size(next_layer)
    return (next_layer, size)


# ==================== 动态杠杆计算 ====================


def get_tier(balance: float) -> str:
    """根据资金确定档位"""
    for tier, cfg in TIERS.items():
        if cfg["min"] <= balance < cfg["max"]:
            return tier
    return "C"


def get_vol_mult(atr_pct: float) -> float:
    """ATR 波动率调节系数"""
    if atr_pct < 3:
        return VOL_MULT["low"]
    if atr_pct <= 5:
        return VOL_MULT["normal"]
    if atr_pct <= 8:
        return VOL_MULT["high"]
    return VOL_MULT["extreme"]


def get_position_limit_pct(mode: str, vegas_state: str) -> float:
    """
    v7.0 仓位上限（%）。base 趋势 100%、震荡 30%，再乘 state_mult。
    """
    base = 100.0 if mode in ("trend_long", "trend_short") else 30.0
    return base * get_state_mult(vegas_state) / 100.0


def calc_leverage(
    balance: float,
    layer: int,
    mode: str,
    atr_pct: float,
    consecutive_losses: int = 0,
    position_pct: Optional[float] = None,
    stop_distance_pct: Optional[float] = None,
    vegas_state: Optional[str] = None,
) -> int:
    """
    动态杠杆计算（v7.0）。最终杠杆 = min(档位上限, 基准×vol_mult×state_mult×lossMult)。
    """
    tier = get_tier(balance)
    tier_cfg = TIERS[tier]
    lev_table = LEVERAGE_TABLE[tier]

    is_trend = mode in ("trend_long", "trend_short")
    mode_key = "trend" if is_trend else "range"

    if is_trend:
        layer_key = f"layer{layer}"
        base_lev = lev_table[mode_key].get(layer_key, 8)
    else:
        base_lev = lev_table[mode_key].get("fixed", 8)

    vol_mult = get_vol_mult(atr_pct)
    loss_mult = 0.7 if consecutive_losses >= 3 else 1.0
    state_mult = get_state_mult(vegas_state or "diverge")

    raw_lev = base_lev * vol_mult * loss_mult * state_mult
    final_lev = min(tier_cfg["max_lev"], int(raw_lev))
    final_lev = max(1, final_lev)

    if position_pct and stop_distance_pct and stop_distance_pct > 0:
        risk = position_pct * final_lev * stop_distance_pct / 100
        max_risk = tier_cfg["max_risk"]
        if risk > max_risk:
            final_lev = int(max_risk / (position_pct * stop_distance_pct / 100))
            final_lev = max(1, final_lev)

    return final_lev


# ==================== 辅助：获取 Pivot 下一层 ====================


def get_pivot_next_level(
    pivots: Dict[str, float],
    direction: str,
    price: float,
) -> Optional[float]:
    """
    获取 Pivot 下一层（做多取下方最近支撑，做空取上方最近阻力）。
    """
    levels = ["R4", "R3", "R2", "R1", "P", "S1", "S2", "S3", "S4"]
    ordered = [pivots.get(k) for k in levels if pivots.get(k) is not None]
    ordered = sorted(set(ordered))

    if direction == "long":
        below = [x for x in ordered if x < price]
        return max(below) if below else min(ordered)
    else:
        above = [x for x in ordered if x > price]
        return min(above) if above else max(ordered)