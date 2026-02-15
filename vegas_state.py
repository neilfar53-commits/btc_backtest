# -*- coding: utf-8 -*-
"""
BTC Pivot-Vegas Pyramid 回测 — Vegas 状态机
依据《回测开发指令》第四节 + 《交易策略框架 v6.1》0.2
四种状态：发散(diverging) / 收紧(tightening) / 缠绕(tangled) / 破位(breakout)
"""

from typing import List, Optional, Tuple


def _count_crosses(ema144_history: List[Tuple[float, float]]) -> int:
    """30 根 12H 内 EMA144 穿越 EMA576 的次数。"""
    if len(ema144_history) < 2:
        return 0
    count = 0
    for i in range(1, len(ema144_history)):
        prev_144, prev_576 = ema144_history[i - 1]
        curr_144, curr_576 = ema144_history[i]
        if prev_144 is None or prev_576 is None or curr_144 is None or curr_576 is None:
            continue
        if prev_144 > prev_576 and curr_144 < curr_576:
            count += 1
        elif prev_144 < prev_576 and curr_144 > curr_576:
            count += 1
    return count


def _is_breakout(ema144_history: List[Tuple[float, float]]) -> bool:
    """144 明确穿越 576 后单方向持续（最近一次穿越后未反穿）。"""
    if len(ema144_history) < 2:
        return False
    # 找最近一次穿越
    for i in range(len(ema144_history) - 1, 0, -1):
        prev_144, prev_576 = ema144_history[i - 1]
        curr_144, curr_576 = ema144_history[i]
        if prev_144 is None or prev_576 is None or curr_144 is None or curr_576 is None:
            continue
        if prev_144 > prev_576 and curr_144 < curr_576:
            # 死叉，检查之后是否一直 144 < 576
            for j in range(i, len(ema144_history)):
                a, b = ema144_history[j]
                if a is not None and b is not None and a >= b:
                    return False
            return True
        if prev_144 < prev_576 and curr_144 > curr_576:
            # 金叉，检查之后是否一直 144 > 576
            for j in range(i, len(ema144_history)):
                a, b = ema144_history[j]
                if a is not None and b is not None and a <= b:
                    return False
            return True
    return False


def get_vegas_state(
    ema144: float,
    ema576: float,
    ema144_history_30: Optional[List[Tuple[float, float]]] = None,
    cross_min: int = 3,
    diverge_threshold_pct: float = 8.0,
) -> str:
    """
    每根 12H 收盘调用，更新状态。
    判定优先级：
    1. 缠绕：30 根 12H 内 144 穿越 576 >= cross_min 次
    2. 破位：144 明确穿越 576 后单方向持续
    3. 发散：dist_pct > diverge_threshold_pct
    4. 收紧（兜底）：dist_pct <= diverge_threshold_pct，无交叉
    """
    if not ema576 or ema576 <= 0:
        return "tightening"

    ema144_history_30 = ema144_history_30 or []
    # 1. 缠绕
    cross_count = _count_crosses(ema144_history_30)
    if cross_count >= cross_min:
        return "tangled"
    # 2. 破位
    if _is_breakout(ema144_history_30):
        return "breakout"
    # 3. 距离
    dist_pct = abs(ema144 - ema576) / ema576 * 100
    if dist_pct > diverge_threshold_pct:
        return "diverging"
    return "tightening"


def get_bull_bear(ema144: float, ema576: float) -> str:
    """牛熊：144 > 576 为牛市，否则熊市。"""
    if not ema576 or ema576 <= 0:
        return "neutral"
    return "bull" if ema144 > ema576 else "bear"
