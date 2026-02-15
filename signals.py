# -*- coding: utf-8 -*-
"""
BTC Pivot-Vegas Pyramid 回测 — 信号系统
依据《回测开发指令》第六节：入场/出场信号生成
"""

from typing import Any, Dict, Optional

import pandas as pd


def generate_entry_signal(
    bar_4h: pd.Series,
    bar_12h: Optional[pd.Series],
    vegas_state: str,
    bull_bear: str,
    mode: str,
    pivots: Dict[str, float],
    ema144_12h: float,
    ema576_12h: float,
    indicators: Dict[str, Any],
) -> Dict[str, Any]:
    """
    入场信号（Phase 1 简化）：
    - 左侧 576/676：价格接近 576 且 bull_bear 一致
    - 右侧 144：价格上穿/回踩 144
    - 过滤器：Vegas 缠绕不做单、MA 缠绕过滤、季节过滤（由调用方做）
    返回：{'action': 'long'|'short'|None, 'entry_type': str, 'price': float, 'score': int, 'filters_passed': bool}
    """
    action = None
    entry_type = None
    price = float(bar_4h["close"])
    score = 0
    filters_passed = True

    if vegas_state == "tangled":
        return {"action": None, "entry_type": None, "price": price, "score": 0, "filters_passed": False}
    if mode not in ("trend_long", "trend_short", "tightening"):
        return {"action": None, "entry_type": None, "price": price, "score": 0, "filters_passed": True}

    # 辅助信号加分
    sigs = indicators.get("signals_4h") or {}
    if sigs.get("pin_bar"):
        score += 2
    if sigs.get("engulfing"):
        score += 2
    if sigs.get("pivot_sr"):
        score += 1
    if sigs.get("ma8_cross"):
        score += 1

    # 做多
    if bull_bear == "bull" and mode in ("trend_long", "tightening"):
        # 左侧 576：价格 <= 576*1.005
        if ema576_12h and price <= ema576_12h * 1.005 and price >= ema576_12h * 0.995:
            action = "long"
            entry_type = "left_576"
            return {"action": action, "entry_type": entry_type, "price": price, "score": score, "filters_passed": filters_passed}
        # 右侧 144：价格从下上穿 144 或回踩 144 上方
        if ema144_12h and bar_12h is not None:
            c12 = bar_12h["close"]
            if c12 >= ema144_12h * 0.998 and c12 <= ema144_12h * 1.002:
                action = "long"
                entry_type = "right_144"
                return {"action": action, "entry_type": entry_type, "price": price, "score": score, "filters_passed": filters_passed}
        if ema144_12h and price > ema144_12h and (indicators.get("ma_cross_4h") == "golden" or score >= 2):
            action = "long"
            entry_type = "right_cross"
            return {"action": action, "entry_type": entry_type, "price": price, "score": score, "filters_passed": filters_passed}

    # 做空
    if bull_bear == "bear" and mode in ("trend_short", "tightening"):
        if ema576_12h and price >= ema576_12h * 0.995 and price <= ema576_12h * 1.005:
            action = "short"
            entry_type = "left_576"
            return {"action": action, "entry_type": entry_type, "price": price, "score": score, "filters_passed": filters_passed}
        if ema144_12h and bar_12h is not None:
            c12 = bar_12h["close"]
            if c12 <= ema144_12h * 1.002 and c12 >= ema144_12h * 0.998:
                action = "short"
                entry_type = "right_144"
                return {"action": action, "entry_type": entry_type, "price": price, "score": score, "filters_passed": filters_passed}
        if ema144_12h and price < ema144_12h and (indicators.get("ma_cross_4h") == "death" or score >= 2):
            action = "short"
            entry_type = "right_cross"
            return {"action": action, "entry_type": entry_type, "price": price, "score": score, "filters_passed": filters_passed}

    return {"action": action, "entry_type": entry_type, "price": price, "score": score, "filters_passed": filters_passed}


def generate_exit_signal(
    position: Any,
    bar_4h: pd.Series,
    is_12h_close: bool,
    vegas_state: str,
) -> Dict[str, Any]:
    """
    出场信号（简化）：
    1. 硬止损（盘中）由引擎用 K 线 high/low 检查
    2. 软止损（12H 收盘）此处仅返回是否应检查
    3. 破位：vegas_state == 'breakout' → close_all
    返回：{'action': 'close_all'|'reduce'|'trail_stop'|None, 'reason': str, 'reduce_pct': float}
    """
    if vegas_state == "breakout":
        return {"action": "close_all", "reason": "vegas_breach", "reduce_pct": 1.0}
    # 软止损、止盈、尾仓由引擎按 position 的 soft_stop / take_profits / trailing 检查
    return {"action": None, "reason": "", "reduce_pct": 0.0}
