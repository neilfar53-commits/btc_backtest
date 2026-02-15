# -*- coding: utf-8 -*-
"""
BTC Pivot-Vegas Pyramid 回测 — 风控层
依据《回测开发指令》第八节：资金档位、杠杆计算、紧急降杠杆
"""

from typing import Optional

from config import TIER_CONFIG, VOL_MULT


def get_tier(balance: float) -> str:
    """根据资金确定档位 S/A/B/C。"""
    if balance < 10000:
        return "S"
    if balance < 50000:
        return "A"
    if balance < 200000:
        return "B"
    return "C"


def get_vol_mult(atr_pct: float) -> float:
    """ATR 波动率调节系数。"""
    if atr_pct < 3:
        return VOL_MULT["low"]
    if atr_pct <= 5:
        return VOL_MULT["normal"]
    if atr_pct <= 8:
        return VOL_MULT["high"]
    return VOL_MULT["extreme"]


def calc_leverage(
    balance: float,
    cum_position_pct: float,
    mode: str,
    atr_pct: float,
    stop_dist_pct: float,
    consecutive_losses: int = 0,
) -> int:
    """
    1. 确定档位（S/A/B/C）
    2. 查基准杠杆表（趋势/震荡）
    3. ATR 波动率调节
    4. 连续止损 >= 3 → ×0.7
    5. 风控验算：仓位% × 杠杆 × 止损距离% <= max_risk_pct
    """
    tier = get_tier(balance)
    cfg = TIER_CONFIG.get(tier, TIER_CONFIG["C"])
    max_risk = cfg["max_risk"]
    max_lev = cfg["max_lev"]

    pos_level = 0
    if cum_position_pct <= 20:
        pos_level = 0
    elif cum_position_pct <= 50:
        pos_level = 1
    elif cum_position_pct <= 70:
        pos_level = 2
    else:
        pos_level = 3

    from config import LEVERAGE_TABLE
    table = LEVERAGE_TABLE.get(tier, LEVERAGE_TABLE["C"])
    trend_lev = table["trend"]
    base_lev = trend_lev.get("layer1", 4)
    if pos_level == 1:
        base_lev = trend_lev.get("layer2", 5)
    elif pos_level == 2:
        base_lev = trend_lev.get("layer3", 6)
    elif pos_level == 3:
        base_lev = trend_lev.get("layer4", 8)

    vol_mult = get_vol_mult(atr_pct)
    loss_mult = 0.7 if consecutive_losses >= 3 else 1.0
    final_lev = min(max_lev, int(base_lev * vol_mult * loss_mult))
    final_lev = max(1, final_lev)

    # 风控验算：假设本笔仓位 20%，确保 20% * lev * stop_dist_pct/100 <= max_risk
    position_pct = 0.20 if cum_position_pct <= 20 else 0.20
    risk = position_pct * final_lev * stop_dist_pct / 100
    if risk > max_risk and stop_dist_pct > 0:
        final_lev = int(max_risk / (position_pct * stop_dist_pct / 100))
        final_lev = max(1, min(max_lev, final_lev))

    return final_lev


def check_emergency(atr_current: float, atr_24h_ago: float) -> bool:
    """ATR 24H 暴涨 > 50% → 紧急降杠杆。"""
    if not atr_24h_ago or atr_24h_ago <= 0:
        return False
    return (atr_current - atr_24h_ago) / atr_24h_ago > 0.50
