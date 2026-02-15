# -*- coding: utf-8 -*-
"""
BTC 交易策略回测系统 — 回测引擎

主循环遍历 4H K 线，无偷看未来，实现仓位管理、交易成本、滑点模拟。
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    AVG_FUNDING_RATE,
    BACKTEST_END,
    BACKTEST_START,
    DATA_DIR,
    INITIAL_BALANCE,
    MAKER_FEE,
    SLIPPAGE,
    TAKER_FEE,
)
from config import TIERS
from indicators import (
    compute_all_indicators,
    detect_cross,
    detect_engulfing,
    detect_pin_bar,
    detect_swing,
)
from risk_management import (
    calc_stop_loss,
    calc_take_profit,
    update_stop_loss,
)
from strategy import (
    calc_entry_score,
    calc_leverage,
    calc_persistent_score,
    check_entry_filters,
    check_vegas_distance_filter,
    get_macro_season,
    get_market_mode,
    get_pivot_next_level,
    get_position_limit_pct,
    get_vegas_state,
    get_wave_type,
    pyramid_entry,
    pyramid_entry_persistent,
)


# ==================== 交易成本与滑点 ====================


def calc_trade_cost(notional_value: float, is_maker: bool = True) -> float:
    """交易手续费"""
    rate = MAKER_FEE if is_maker else TAKER_FEE
    return notional_value * rate


def calc_funding_cost(position_value: float, holding_hours: float) -> float:
    """资金费率（每 8 小时收取）"""
    periods = holding_hours / 8
    return position_value * AVG_FUNDING_RATE * periods


def apply_slippage(price: float, direction: str, action: str) -> float:
    """
    滑点：买入开多/卖出平空 价格偏高，卖出平多/买入开空 价格偏低
    action: 'open_long'/'open_short'/'close_long'/'close_short'
    """
    if action in ("open_long", "close_short"):
        return price * (1 + SLIPPAGE)
    return price * (1 - SLIPPAGE)


# ==================== 仓位与组合 ====================


@dataclass
class Position:
    """单笔持仓"""

    direction: str  # 'long' / 'short'
    entry_price: float
    size_pct: float
    leverage: int
    layer: int
    soft_stop: float
    hard_stop: float
    take_profits: List[Dict]
    entry_time: int  # timestamp ms
    entry_bar_idx: int
    mode: str
    wave_type: str
    margin: float  # 占用保证金 USDT
    contracts: float  # 张数/数量（简化用 USDT 名义价值）
    pyramid_id: int = 0  # 金字塔组 ID，用于统计

    def unrealized_pnl(self, price: float) -> float:
        if self.direction == "long":
            return (price - self.entry_price) / self.entry_price * self.contracts
        return (self.entry_price - price) / self.entry_price * self.contracts


@dataclass
class Trade:
    """已完成交易记录"""

    direction: str
    entry_price: float
    exit_price: float
    size_pct: float
    leverage: int
    entry_time: int
    exit_time: int
    pnl: float
    reason: str  # 'tp' / 'soft_stop' / 'hard_stop'
    mode: str
    wave_type: str = ""
    pyramid_layer: int = 1
    pyramid_id: int = 0
    margin: float = 0  # 平仓时保证金(USDT)
    balance_before: float = 0  # 平仓前 balance
    balance_after: float = 0  # 平仓后 balance
    soft_stop: float = 0  # 软止损价（用于诊断）
    hard_stop: float = 0  # 硬止损价（用于诊断）


class PortfolioManager:
    """组合管理"""

    def __init__(self, initial_balance: float):
        self.balance = initial_balance
        self.positions: List[Position] = []
        self.trade_history: List[Trade] = []
        self.partial_close_history: List[Dict] = []  # 部分平仓记录，用于诊断
        self.consecutive_losses = 0
        self.equity_history: List[Tuple[int, float]] = []
        self.total_open_cost = 0.0  # 累计开仓成本(手续费+滑点)
        self.total_funding = 0.0  # 累计资金费率

    def get_total_margin(self) -> float:
        return sum(p.margin for p in self.positions)

    def get_unrealized_pnl(self, price: float) -> float:
        return sum(p.unrealized_pnl(price) for p in self.positions)

    def get_equity(self, price: float) -> float:
        return self.balance + self.get_unrealized_pnl(price)

    def open_position(
        self,
        direction: str,
        size_pct: float,
        leverage: int,
        entry_price: float,
        soft_stop: float,
        hard_stop: float,
        take_profits: List[Dict],
        entry_time: int,
        entry_bar_idx: int,
        mode: str,
        wave_type: str,
        pyramid_id: int = 0,
        layer: int = 1,
    ) -> Optional[Position]:
        """开仓（首仓或金字塔加仓）"""
        equity = self.get_equity(entry_price)
        margin = equity * size_pct
        contracts = margin * leverage  # 名义价值
        cost = calc_trade_cost(contracts, is_maker=True)
        entry_adj = apply_slippage(
            entry_price,
            direction,
            "open_long" if direction == "long" else "open_short",
        )
        cost += abs(entry_adj - entry_price) / entry_price * contracts
        if margin + cost > self.balance:
            return None
        self.balance -= margin + cost  # 扣除保证金 + 开仓手续费/滑点
        self.total_open_cost += cost
        pos = Position(
            direction=direction,
            entry_price=entry_adj,
            size_pct=size_pct,
            leverage=leverage,
            layer=layer,
            soft_stop=soft_stop,
            hard_stop=hard_stop,
            take_profits=take_profits.copy(),
            entry_time=entry_time,
            entry_bar_idx=entry_bar_idx,
            mode=mode,
            wave_type=wave_type,
            margin=margin,
            contracts=contracts,
            pyramid_id=pyramid_id,
        )
        self.positions.append(pos)
        return pos

    def close_position(
        self, position: Position, exit_price: float, reason: str, exit_time: int = 0
    ) -> float:
        """平仓"""
        balance_before = self.balance
        exit_adj = apply_slippage(
            exit_price,
            position.direction,
            "close_long" if position.direction == "long" else "close_short",
        )
        pnl = position.unrealized_pnl(exit_adj)
        fee = calc_trade_cost(position.contracts, is_maker=False)
        pnl -= fee
        self.balance += position.margin + pnl
        balance_after = self.balance
        self.positions.remove(position)
        trade = Trade(
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_adj,
            size_pct=position.size_pct,
            leverage=position.leverage,
            entry_time=position.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            reason=reason,
            mode=position.mode,
            wave_type=position.wave_type,
            pyramid_layer=position.layer,
            pyramid_id=position.pyramid_id,
            margin=position.margin,
            balance_before=balance_before,
            balance_after=balance_after,
            soft_stop=position.soft_stop,
            hard_stop=position.hard_stop,
        )
        self.trade_history.append(trade)
        if pnl < 0:
            self.consecutive_losses += 1
            # 风控检查：单笔亏损不应超过 balance_before * max_risk
            max_risk_pct = TIERS["S"]["max_risk"] * 100
            loss_pct = abs(pnl) / balance_before * 100 if balance_before > 0 else 0
            if loss_pct > max_risk_pct:
                trade._max_risk_violation = loss_pct  # 标记风控超限
        else:
            self.consecutive_losses = 0
        return pnl

    def partial_close(
        self, position: Position, reduce_pct: float, exit_price: float
    ) -> float:
        """部分平仓"""
        balance_before = self.balance
        exit_adj = apply_slippage(
            exit_price,
            position.direction,
            "close_long" if position.direction == "long" else "close_short",
        )
        reduce_contracts = position.contracts * reduce_pct
        if position.direction == "long":
            pnl = (exit_adj - position.entry_price) / position.entry_price * reduce_contracts
        else:
            pnl = (position.entry_price - exit_adj) / position.entry_price * reduce_contracts
        fee = calc_trade_cost(reduce_contracts, is_maker=False)
        pnl -= fee
        margin_back = position.margin * reduce_pct
        self.balance += margin_back + pnl
        position.contracts -= reduce_contracts
        position.margin -= margin_back
        position.size_pct *= 1 - reduce_pct
        self.partial_close_history.append({
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_adj,
            "reduce_pct": reduce_pct,
            "margin_reduced": margin_back,
            "pnl": pnl,
            "balance_before": balance_before,
            "balance_after": self.balance,
        })
        return pnl


# ==================== Vegas 状态（v7.0） ====================


def _vegas_cross_count_30(df: pd.DataFrame, i: int) -> int:
    """过去 30 根 12H 内 144 与 576 的交叉次数（每 3 根 4H = 1 根 12H）。"""
    if "12h_vegas_144" not in df.columns or "12h_vegas_576" not in df.columns:
        return 0
    if i < 90:  # 需要至少 30*3 根 4H
        return 0
    crosses = 0
    for k in range(0, 87, 3):  # 0,3,6,...,84 → 29 个间隔
        j1, j2 = i - k, i - k - 3
        if j2 < 0:
            break
        v1_144, v1_576 = df.iloc[j1]["12h_vegas_144"], df.iloc[j1]["12h_vegas_576"]
        v2_144, v2_576 = df.iloc[j2]["12h_vegas_144"], df.iloc[j2]["12h_vegas_576"]
        if pd.isna(v1_144) or pd.isna(v1_576) or pd.isna(v2_144) or pd.isna(v2_576):
            continue
        if (v1_144 > v1_576) != (v2_144 > v2_576):
            crosses += 1
    return crosses


def _vegas_just_crossed(df: pd.DataFrame, i: int) -> Optional[str]:
    """当前 12H 是否刚发生 144 穿越 576。前一 12H 收盘在 i-3。"""
    if "12h_vegas_144" not in df.columns or "12h_vegas_576" not in df.columns:
        return None
    if i < 3:
        return None
    now_144, now_576 = df.iloc[i]["12h_vegas_144"], df.iloc[i]["12h_vegas_576"]
    prev_144, prev_576 = df.iloc[i - 3]["12h_vegas_144"], df.iloc[i - 3]["12h_vegas_576"]
    if pd.isna(now_144) or pd.isna(now_576) or pd.isna(prev_144) or pd.isna(prev_576):
        return None
    if (now_144 > now_576) != (prev_144 > prev_576):
        return "golden" if now_144 > now_576 else "death"
    return None


# ==================== 数据对齐 ====================


def build_aligned_data(data_dir: str = None) -> pd.DataFrame:
    """构建 4H 对齐的完整数据（12H/周线/Pivot 前向填充）"""
    data_dir = data_dir or DATA_DIR
    ind = compute_all_indicators(data_dir)
    df_4h = ind["df_4h"].copy()
    df_12h = ind["df_12h"].copy()
    df_weekly = ind["df_weekly"].copy()
    df_pivot = ind["df_pivot"]

    # 12H: 重命名 OHLC 避免冲突
    rename_12h = {c: f"12h_{c}" for c in ["open", "high", "low", "close", "volume"] if c in df_12h.columns}
    df_12h = df_12h.rename(columns=rename_12h)
    df_weekly = df_weekly.rename(
        columns={c: f"w_{c}" for c in ["open", "high", "low", "close", "volume"] if c in df_weekly.columns}
    )

    merged = pd.merge_asof(
        df_4h.sort_values("timestamp"),
        df_12h.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        suffixes=("", "_12h"),
    )
    merged = pd.merge_asof(
        merged.sort_values("timestamp"),
        df_weekly.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        suffixes=("", "_w"),
    )

    # Pivot: 按 month_ts 对齐
    if len(df_pivot) > 0 and "month_ts" in df_pivot.columns:
        pivot_cols = [c for c in ["P", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4"] if c in df_pivot.columns]
        pivot_sub = df_pivot[["month_ts"] + pivot_cols].copy()
        pivot_sub = pivot_sub.rename(columns={c: f"pivot_{c}" for c in pivot_cols})
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            pivot_sub.sort_values("month_ts"),
            left_on="timestamp",
            right_on="month_ts",
            direction="backward",
        )
    return merged


# ==================== 信号构建 ====================


def build_signals(
    df: pd.DataFrame,
    i: int,
    direction: str,
) -> Dict[str, bool]:
    """根据当前 bar 构建入场信号字典"""
    if i < 2:
        return {}
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    prev2 = df.iloc[i - 2] if i >= 2 else prev

    signals = {}
    # Pivot S/R 区域（简化：价格在 P 附近 ±3%）
    P = row.get("pivot_P", row.get("close"))
    if P and P > 0:
        signals["pivot_sr"] = abs(row["close"] - P) / P < 0.03
    else:
        signals["pivot_sr"] = False

    signals["pin_bar"] = detect_pin_bar(
        row["open"], row["high"], row["low"], row["close"],
        "bull" if direction == "long" else "bear",
    )
    signals["engulfing"] = detect_engulfing(
        row["open"], row["high"], row["low"], row["close"],
        prev["open"], prev["high"], prev["low"], prev["close"],
        "bull" if direction == "long" else "bear",
    )

    ma8 = row.get("4h_ma8")
    ma17 = row.get("4h_ma17")
    signals["ma8_cross"] = (
        ma8 is not None and ma17 is not None
        and (row["close"] > ma8 if direction == "long" else row["close"] < ma8)
    )
    cross = detect_cross(
        df["4h_ma8"].values, df["4h_ma17"].values, i
    ) if "4h_ma8" in df.columns else None
    signals["ma_cross"] = (
        cross == ("golden" if direction == "long" else "death")
    )
    ma_entangled = cross == "entangled"

    rsi = row.get("4h_rsi")
    signals["rsi_extreme"] = (
        rsi is not None
        and (rsi < 30 if direction == "long" else rsi > 70)
    )

    # 连续 3 根同向
    if i >= 3:
        c3 = [df.iloc[i - k]["close"] > df.iloc[i - k]["open"] for k in range(3)]
        if direction == "long":
            signals["three_same_candle"] = all(c3)
        else:
            signals["three_same_candle"] = all(not x for x in c3)
    else:
        signals["three_same_candle"] = False

    # 12H MA8/MA17 交叉（手册重要确认信号）
    ma12h_8 = df["12h_ma8"].values if "12h_ma8" in df.columns else None
    ma12h_17 = df["12h_ma17"].values if "12h_ma17" in df.columns else None
    cross_12h = detect_cross(ma12h_8, ma12h_17, i) if ma12h_8 is not None and ma12h_17 is not None else None
    signals["ma12h_cross"] = (
        cross_12h == ("golden" if direction == "long" else "death")
    )

    vegas_576 = row.get("12h_vegas_576")
    signals["vegas_break"] = (
        vegas_576 is not None
        and (
            (row["close"] > vegas_576 and prev["close"] <= vegas_576)
            if direction == "long"
            else (row["close"] < vegas_576 and prev["close"] >= vegas_576)
        )
    )

    # v7.0: Vegas 576/676 附近 ±3% +2 分
    vegas_676 = row.get("12h_vegas_676") or vegas_576
    if vegas_576 and vegas_576 > 0:
        near_576 = abs(row["close"] - vegas_576) / vegas_576 <= 0.03
        near_676 = vegas_676 and abs(row["close"] - vegas_676) / vegas_676 <= 0.03
        signals["vegas_576_near"] = near_576 or near_676
    else:
        signals["vegas_576_near"] = False

    # v7.0: 前月阻力共振 +1（简化：当月 Pivot 与前一月某层级距离≤1.5% 视为共振）
    signals["prev_month_resonance"] = False  # 可后续接入前月 Pivot 列

    return signals


# ==================== 主回测循环 ====================


def run_backtest(
    initial_balance: float = None,
    start_date: str = None,
    end_date: str = None,
    data_dir: str = None,
) -> Tuple[PortfolioManager, pd.DataFrame, Dict[str, Any]]:
    """
    运行回测。

    Returns:
        (portfolio, aligned_df, extra) extra 含 monthly_mode 等
    """
    initial_balance = initial_balance or INITIAL_BALANCE
    start_date = start_date or BACKTEST_START
    end_date = end_date or BACKTEST_END
    data_dir = data_dir or DATA_DIR

    df = build_aligned_data(data_dir)
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].reset_index(
        drop=True
    )

    pm = PortfolioManager(initial_balance)
    prev_season = None
    prev_weekly_close = None
    prev_vegas_w144 = None
    prev_vegas_w169 = None
    pyramid_id_counter = 0
    pending_entry = None
    monthly_mode: Dict[str, str] = {}
    pyramid_score_5_plus = 0
    score_history: List[Dict] = []
    # v7.0 尾仓追踪：历史最高/最低 4H MA55，key=(entry_bar_idx, entry_time)
    trailing_ma55: Dict[Tuple[int, int], float] = {}

    def is_12h_close(i: int) -> bool:
        return (i + 1) % 3 == 0 if i >= 0 else False

    for i in range(len(df)):
        row = df.iloc[i]
        ts = row["timestamp"]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]

        # 当前 bar 可用的 season（尚未更新时用上一值）
        season = prev_season or "transition"

        # 限价单：上一根 K 线信号在本根 open 成交，避免当根 K 线偷看 future
        if pending_entry is not None:
            pe = pending_entry
            pending_entry = None
            direction = pe["direction"]
            mode = pe["mode"]
            wave_type = pe["wave_type"]
            size = pe["size"]
            lev = pe["leverage"]
            soft, hard = pe["soft"], pe["hard"]
            tps = pe["take_profits"]
            pyramid_id = pe["pyramid_id"]
            layer = pe["layer"]
            fill_price = o  # 下一根 open 成交
            if season == "summer" and direction == "short":
                pass  # 牛市禁空，不执行
            elif season == "winter" and direction == "long":
                pass  # 熊市禁多，不执行
            else:
                pm.open_position(
                    direction, size, lev, fill_price, soft, hard, tps,
                    ts, i, mode, wave_type,
                    pyramid_id=pyramid_id, layer=layer,
                )

        # 获取当前对齐的指标
        vegas_4h_144 = row.get("4h_vegas_144") or 0
        vegas_12h_144 = row.get("12h_vegas_144") or vegas_4h_144
        vegas_12h_169 = row.get("12h_vegas_169") or vegas_12h_144
        vegas_12h_576 = row.get("12h_vegas_576") or 0
        atr_4h = row.get("4h_atr") or (c * 0.02)
        atr_12h = row.get("12h_atr") or atr_4h

        pivots = {}
        for k in ["P", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4"]:
            v = row.get(f"pivot_{k}")
            if pd.notna(v):
                pivots[k] = float(v)

        if not pivots:
            pivots = {"P": c * 0.99, "S1": c * 0.97, "R1": c * 1.03}

        # v7.0 Vegas 状态机（每 bar 计算）
        vegas_state = get_vegas_state(
            vegas_12h_144,
            vegas_12h_576,
            _vegas_cross_count_30(df, i),
            _vegas_just_crossed(df, i),
        )

        # 12H 收盘时更新宏观、模式
        if is_12h_close(i):
            weekly_close = row.get("w_close") or row.get("close")
            vegas_w144 = row.get("w_weekly_vegas_144") or vegas_12h_144
            vegas_w169 = row.get("w_weekly_vegas_169") or vegas_12h_576
            season = get_macro_season(
                weekly_close, vegas_w144, vegas_w169,
                prev_season, prev_weekly_close, prev_vegas_w144, prev_vegas_w169,
            )
            prev_season = season
            prev_weekly_close = weekly_close
            prev_vegas_w144 = vegas_w144
            prev_vegas_w169 = vegas_w169

            wave_type = get_wave_type(c, vegas_12h_576)
            high_12h = df.iloc[max(0, i - 2) : i + 1]["high"].max()
            low_12h = df.iloc[max(0, i - 2) : i + 1]["low"].min()
            swing_highs, swing_lows = detect_swing(
                df["high"].values[: i + 1],
                df["low"].values[: i + 1],
                6,
            )
            # 只使用已确认的 Swing（index <= i-6），确认前不得使用
            lb = 6
            swing_highs = [s for s in swing_highs if s[0] <= i - lb]
            swing_lows = [s for s in swing_lows if s[0] <= i - lb]
            vegas_676 = row.get("12h_vegas_676") or vegas_12h_576
            mode = get_market_mode(
                swing_highs, swing_lows, c, pivots,
                vegas_12h_144, vegas_12h_169, vegas_12h_576, vegas_676,
            )
            # 记录月度模式（以该月最后 12H 收盘时的 mode 为准）
            ym = pd.Timestamp(ts, unit="ms").strftime("%Y-%m")
            monthly_mode[ym] = mode
            # 记录 12H 评分明细（2024-10~12 诊断用）
            if ym >= "2024-10" and ym <= "2024-12" and mode in ("trend_long", "trend_short"):
                d = "long" if mode == "trend_long" else "short"
                sigs = build_signals(df, i, d)
                sc = calc_entry_score(mode, sigs, d, use_cross=True)
                score_history.append({
                    "ts": ts,
                    "ym": ym,
                    "close": c,
                    "mode": mode,
                    "score": sc,
                    "signals": sigs,
                })
        else:
            season = prev_season or "transition"
            wave_type = get_wave_type(c, vegas_12h_576)
            swing_highs, swing_lows = detect_swing(
                df["high"].values[: i + 1], df["low"].values[: i + 1], 6
            )
            lb = 6
            swing_highs = [s for s in swing_highs if s[0] <= i - lb]
            swing_lows = [s for s in swing_lows if s[0] <= i - lb]
            vegas_676 = row.get("12h_vegas_676") or vegas_12h_576
            mode = get_market_mode(
                swing_highs, swing_lows, c, pivots,
                vegas_12h_144, vegas_12h_169, vegas_12h_576, vegas_676,
            )

        # v7.0 破位平仓：12H 收盘 144 穿越 576 → 平趋势方向仓
        if is_12h_close(i) and vegas_state == "breakout":
            just_crossed = _vegas_just_crossed(df, i)
            for pos in list(pm.positions):
                if just_crossed == "death" and pos.direction == "long":
                    pm.close_position(pos, c, "vegas_breakout", ts)
                    key = (pos.entry_bar_idx, pos.entry_time)
                    trailing_ma55.pop(key, None)
                elif just_crossed == "golden" and pos.direction == "short":
                    pm.close_position(pos, c, "vegas_breakout", ts)
                    key = (pos.entry_bar_idx, pos.entry_time)
                    trailing_ma55.pop(key, None)

        # 检查持仓：硬止损、止盈、软止损（12H 收盘）
        for pos in list(pm.positions):
            if pos.direction == "long":
                if l <= pos.hard_stop:
                    pm.close_position(pos, pos.hard_stop, "hard_stop", ts)
                    trailing_ma55.pop((pos.entry_bar_idx, pos.entry_time), None)
                    continue
                for tp in list(pos.take_profits):
                    if tp["price"] > pos.entry_price and h >= tp["price"]:
                        pm.partial_close(pos, tp["reduce_pct"], tp["price"])
                        pos.take_profits.remove(tp)
                        break
                if is_12h_close(i) and c <= pos.soft_stop:
                    pm.close_position(pos, pos.soft_stop, "soft_stop", ts)
                    trailing_ma55.pop((pos.entry_bar_idx, pos.entry_time), None)
                # v7.0 尾仓追踪：无剩余 TP 时为尾仓，历史最高 4H MA55 只升不降
                ma55_4h = row.get("4h_ma55")
                if ma55_4h is not None and len(pos.take_profits) == 0:
                    key = (pos.entry_bar_idx, pos.entry_time)
                    cur = trailing_ma55.get(key, ma55_4h)
                    trailing_ma55[key] = max(cur, ma55_4h)
                    if c < trailing_ma55[key]:
                        pm.close_position(pos, c, "trailing_tail", ts)
                        trailing_ma55.pop(key, None)
            else:
                if h >= pos.hard_stop:
                    pm.close_position(pos, pos.hard_stop, "hard_stop", ts)
                    trailing_ma55.pop((pos.entry_bar_idx, pos.entry_time), None)
                    continue
                for tp in list(pos.take_profits):
                    if tp["price"] < pos.entry_price and l <= tp["price"]:
                        pm.partial_close(pos, tp["reduce_pct"], tp["price"])
                        pos.take_profits.remove(tp)
                        break
                if is_12h_close(i) and c >= pos.soft_stop:
                    pm.close_position(pos, pos.soft_stop, "soft_stop", ts)
                    trailing_ma55.pop((pos.entry_bar_idx, pos.entry_time), None)
                ma55_4h = row.get("4h_ma55")
                if ma55_4h is not None and len(pos.take_profits) == 0:
                    key = (pos.entry_bar_idx, pos.entry_time)
                    cur = trailing_ma55.get(key, ma55_4h)
                    trailing_ma55[key] = min(cur, ma55_4h)
                    if c > trailing_ma55[key]:
                        pm.close_position(pos, c, "trailing_tail", ts)
                        trailing_ma55.pop(key, None)

        # 资金费率：每根 4H bar 扣除 0.5 个周期（4h/8h）的费用
        for pos in pm.positions:
            funding = calc_funding_cost(pos.contracts, 4)
            pm.balance -= funding
            pm.total_funding += funding

        # 保本止损移动：1R = 止损距离占入场价% × 名义价值 / 100 ≈ margin × leverage × stop_dist_pct / 100
        for pos in pm.positions:
            stop_dist = abs(pos.entry_price - pos.hard_stop) / pos.entry_price * 100 if pos.entry_price else 5
            one_r = pos.margin * pos.leverage * stop_dist / 100
            if not one_r or one_r <= 0:
                one_r = pos.margin * 0.05 or 1  # 兜底：5% 保证金或 1 避免除零
            r = abs(pos.unrealized_pnl(c) / one_r)
            ns, nh = update_stop_loss(
                pos.soft_stop, pos.hard_stop,
                pos.entry_price, pos.direction, r,
                vegas_4h_144, atr_4h,
            )
            pos.soft_stop = max(pos.soft_stop, ns) if pos.direction == "long" else min(pos.soft_stop, ns)
            pos.hard_stop = max(pos.hard_stop, nh) if pos.direction == "long" else min(pos.hard_stop, nh)

        # 入场逻辑：首仓或金字塔加仓，限价单下一根 open 成交
        if mode in ("trend_long", "range"):
            same_dir_positions = [p for p in pm.positions if p.direction == "long"]
        elif mode == "trend_short":
            same_dir_positions = [p for p in pm.positions if p.direction == "short"]
        else:
            same_dir_positions = []

        is_first_entry = len(pm.positions) == 0
        can_pyramid = (
            mode in ("trend_long", "trend_short")
            and len(same_dir_positions) > 0
            and len(pm.positions) == len(same_dir_positions)  # 无反向仓
        )

        direction = "long" if mode in ("trend_long", "range") else "short"
        if mode == "range":
            # 震荡市双向：S1 附近做多，R1 附近做空
            P = pivots.get("P") or c
            S1 = pivots.get("S1") or P * 0.97
            R1 = pivots.get("R1") or P * 1.03
            near_s1 = S1 and abs(c - S1) / S1 < 0.02
            near_r1 = R1 and abs(c - R1) / R1 < 0.02
            if near_s1 and not near_r1:
                direction = "long"
            elif near_r1 and not near_s1:
                direction = "short"
            else:
                direction = None  # 不满足 S1/R1 不入场

        if direction is None:
            pass  # 震荡市不在 S1/R1 附近，不入场
        elif season == "summer" and direction == "short":
            pass
        elif season == "winter" and direction == "long":
            pass
        elif is_first_entry and mode in ("trend_long", "trend_short", "range") and direction is not None:
            # v7.0: 破位不开新仓；Vegas 距离>15% 且方向一致不新开仓；仓位上限
            if vegas_state == "breakout":
                pass
            elif not check_vegas_distance_filter(c, vegas_12h_576, direction):
                pass
            else:
                limit_pct = get_position_limit_pct(mode, vegas_state)
                size = 0.2 if mode != "range" else 0.25
                if size > limit_pct:
                    pass
                else:
                    signals = build_signals(df, i, direction)
                    ma_cross = detect_cross(
                        df["4h_ma8"].values, df["4h_ma17"].values, i
                    ) if "4h_ma8" in df.columns else None
                    ma_entangled = ma_cross == "entangled"
                    if check_entry_filters(ma_entangled, season, direction, is_left=True):
                        score = calc_entry_score(mode, signals, direction, use_cross=(mode != "range"))
                        thresh = 3
                        if score >= thresh and pending_entry is None:
                            pyramid_id_counter += 1
                            atr_pct = row.get("4h_atr_pct") or row.get("12h_atr_pct") or 4.0
                            soft, hard = calc_stop_loss(
                                c, direction,
                                vegas_4h_144, vegas_12h_144, vegas_12h_576,
                                atr_4h, pivots, c,
                            )
                            stop_dist_pct = abs(c - hard) / c * 100 if hard else 5.0
                            lev = calc_leverage(
                                pm.get_equity(c), 1, mode, atr_pct, pm.consecutive_losses,
                                position_pct=size, stop_distance_pct=stop_dist_pct,
                                vegas_state=vegas_state,
                            )
                            tps = calc_take_profit(
                                wave_type, direction, pivots,
                                vegas_12h_144, vegas_12h_576,
                            )
                            pending_entry = {
                                "direction": direction,
                                "mode": mode,
                                "wave_type": wave_type,
                                "size": size,
                                "leverage": lev,
                                "soft": soft,
                                "hard": hard,
                                "take_profits": tps,
                                "pyramid_id": pyramid_id_counter,
                                "layer": 1,
                            }

        elif can_pyramid and pending_entry is None and is_12h_close(i):
            # v7.0 金字塔加仓：加仓过滤仅 2 条 — 距下一 TP≥3% + 首仓浮盈>0；仓位上限×state_mult
            current_layers = len(same_dir_positions)
            pid = same_dir_positions[0].pyramid_id
            first_pos = same_dir_positions[0]
            first_unrealized = first_pos.unrealized_pnl(c)

            def _do_pyramid_add():
                total_pct = sum(p.size_pct for p in same_dir_positions)
                limit_pct = get_position_limit_pct(mode, vegas_state)
                result = pyramid_entry_persistent(
                    mode,
                    calc_persistent_score(
                        direction, c,
                        row.get("4h_ma8") or c, row.get("4h_ma17") or c,
                        vegas_4h_144, mode,
                        first_unrealized / (first_pos.margin * 0.05) if first_pos.margin else 0,
                        row.get("4h_rsi"),
                    ),
                    current_layers,
                    direction,
                )
                if not result:
                    return
                next_layer, size_pct = result
                if total_pct + size_pct > limit_pct:
                    return
                atr_pct = row.get("4h_atr_pct") or row.get("12h_atr_pct") or 4.0
                soft, hard = calc_stop_loss(
                    c, direction,
                    vegas_4h_144, vegas_12h_144, vegas_12h_576,
                    atr_4h, pivots, c,
                )
                stop_dist_pct = abs(c - hard) / c * 100 if hard else 5.0
                lev = calc_leverage(
                    pm.get_equity(c), next_layer, mode, atr_pct, pm.consecutive_losses,
                    position_pct=size_pct, stop_distance_pct=stop_dist_pct,
                    vegas_state=vegas_state,
                )
                tps = calc_take_profit(
                    wave_type, direction, pivots,
                    vegas_12h_144, vegas_12h_576,
                )
                nonlocal pending_entry
                pending_entry = {
                    "direction": direction,
                    "mode": mode,
                    "wave_type": wave_type,
                    "size": size_pct,
                    "leverage": lev,
                    "soft": soft,
                    "hard": hard,
                    "take_profits": tps,
                    "pyramid_id": pid,
                    "layer": next_layer,
                }

            if first_unrealized <= 0:
                pass
            else:
                next_tps = [tp["price"] for tp in first_pos.take_profits if (tp["price"] > c if direction == "long" else tp["price"] < c)]
                if next_tps:
                    next_tp = min(next_tps) if direction == "long" else max(next_tps)
                    dist_pct = abs(next_tp - c) / c * 100
                    if dist_pct >= 3:
                        _do_pyramid_add()
                else:
                    _do_pyramid_add()

        pm.equity_history.append((ts, pm.get_equity(c)))

    # 回测结束平掉所有持仓
    for pos in list(pm.positions):
        pm.close_position(pos, df.iloc[-1]["close"], "eod", df.iloc[-1]["timestamp"])

    extra = {
        "monthly_mode": monthly_mode,
        "pyramid_score_5_plus": pyramid_score_5_plus,
        "score_history": score_history,
    }
    return pm, df, extra