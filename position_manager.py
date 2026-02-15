# -*- coding: utf-8 -*-
"""
BTC Pivot-Vegas Pyramid 回测 — 仓位管理
依据《回测开发指令》第七节：Position、金字塔、止损计算与移动
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Phase 1 复用 risk_management 的止损逻辑；后续可迁入本模块
from risk_management import calc_stop_loss as _calc_stop_loss, update_stop_loss as _update_stop_loss
from strategy import get_pivot_next_level


@dataclass
class Position:
    """单笔持仓（与 backtest_engine 中一致，便于兼容）"""
    id: str
    direction: str  # 'long' / 'short'
    entry_price: float
    entry_time: int  # ms
    entry_bar_idx: int
    size_pct: float
    leverage: int
    soft_stop: float
    hard_stop: float
    take_profits: List[Dict]
    trailing_stop: Optional[float] = None  # 4H MA55 跟踪值
    pyramid_layer: int = 1
    pyramid_id: str = ""
    entry_type: str = ""
    market_mode: str = ""
    vegas_state: str = ""
    margin: float = 0.0
    contracts: float = 0.0
    current_pnl: float = 0.0
    max_favorable: float = 0.0
    is_breakeven: bool = False

    def unrealized_pnl(self, price: float) -> float:
        if self.direction == "long":
            return (price - self.entry_price) / self.entry_price * self.contracts
        return (self.entry_price - price) / self.entry_price * self.contracts


def calc_stop_loss(
    direction: str,
    entry_price: float,
    vegas_levels: Dict[str, float],
    atr: float,
    pivots: Dict[str, float],
    entry_type: str,
) -> tuple:
    """
    计算软硬双层止损（做多/做空）。
    vegas_levels: {'144': v, '576': v, '676': v}（4H 或 12H 按引擎约定）
    返回 (soft_stop, hard_stop)。
    """
    v144 = vegas_levels.get("144") or vegas_levels.get("vegas_12h_144")
    v576 = vegas_levels.get("576") or vegas_levels.get("vegas_12h_576")
    v676 = vegas_levels.get("676") or vegas_levels.get("vegas_12h_676")
    return _calc_stop_loss(
        entry_price,
        direction,
        v144 or 0,
        v144 or 0,
        v576 or 0,
        atr,
        pivots,
        entry_price,
        vegas_12h_676=v676,
        tightening_layer1=(entry_type == "tightening_576"),
    )


def update_stop_loss(
    position: Position,
    current_price: float,
    vegas_levels: Dict[str, float],
    atr: float,
) -> tuple:
    """止损移动（只往有利方向）；返回 (new_soft, new_hard)。"""
    stop_dist = abs(position.entry_price - position.hard_stop) / position.entry_price * 100 if position.entry_price else 5
    one_r = position.margin * position.leverage * stop_dist / 100 if position.margin else 0
    if not one_r or one_r <= 0:
        one_r = position.margin * 0.05 or 1
    unrealized = position.unrealized_pnl(current_price)
    r = unrealized / one_r if one_r > 0 else 0
    v144 = vegas_levels.get("144") or vegas_levels.get("vegas_4h_144")
    return _update_stop_loss(
        position.soft_stop,
        position.hard_stop,
        position.entry_price,
        position.direction,
        r,
        v144 or 0,
        atr,
    )


def can_add_pyramid(
    positions: List[Position],
    current_price: float,
    next_tp: Optional[float],
    vegas_state: str,
    mode: str,
    min_tp_dist_pct: float = 3.0,
) -> bool:
    """加仓过滤：距下一 TP ≥ min_tp_dist_pct、Vegas 非缠绕/破位、首仓浮盈>0（Phase 2 用）。"""
    if vegas_state in ("tangled", "breakout"):
        return False
    if not positions:
        return False
    if next_tp is not None and current_price > 0:
        dist_pct = abs(next_tp - current_price) / current_price * 100
        if dist_pct < min_tp_dist_pct:
            return False
    return True
