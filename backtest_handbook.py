# -*- coding: utf-8 -*-
"""
BTC Pivot-Vegas Pyramid 回测 — 手册版引擎（Phase 1 MVP）
依据《回测开发指令》第九节：逐 4H 遍历，12H 决策；仅趋势+收紧，首仓 20%，无金字塔。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from config import (
    BACKTEST_END,
    BACKTEST_START,
    FIRST_POSITION_PCT,
    INITIAL_BALANCE,
    WARMUP_START,
)
from data_loader import load_or_fetch_4h, resample_to_12h
from indicators import (
    compute_4h_indicators,
    compute_12h_indicators,
    compute_monthly_pivot,
    detect_cross,
    detect_swing,
)
from market_mode import get_market_mode
from risk_management import calc_stop_loss, calc_take_profit, update_stop_loss
from risk_manager import calc_leverage, get_tier
from signals import generate_entry_signal
from vegas_state import get_bull_bear, get_vegas_state
from config import TIERS


@dataclass
class SimplePosition:
    direction: str
    entry_price: float
    entry_time: int
    entry_bar_idx: int
    size_pct: float
    leverage: int
    soft_stop: float
    hard_stop: float
    take_profits: List[Dict]
    margin: float
    contracts: float
    entry_type: str = ""
    mode: str = ""

    def unrealized_pnl(self, price: float) -> float:
        if self.direction == "long":
            return (price - self.entry_price) / self.entry_price * self.contracts
        return (self.entry_price - price) / self.entry_price * self.contracts


@dataclass
class SimpleTrade:
    direction: str
    entry_price: float
    exit_price: float
    size_pct: float
    leverage: int
    pnl: float
    reason: str
    mode: str
    entry_type: str


def _is_12h_close(ts_ms: int) -> bool:
    """4H 收盘时间 UTC 00/04/08/12/16/20 → 12H 收盘为 00+4 与 12+4 即 04 与 16 时? 手册：每 3 根 4H 一根 12H，对齐 00:00 和 12:00。"""
    import datetime
    dt = datetime.datetime.utcfromtimestamp(ts_ms / 1000)
    # 12H 收盘：第一根 12H 为 00:00-12:00 收盘 12:00，第二根 12:00-24:00 收盘 24:00
    # 4H 为 00,04,08,12,16,20 → 08+4=12 为 12H 收盘，20+4=24 为 12H 收盘
    hour = (dt.hour + 4) % 24  # 收盘时刻
    return hour in (0, 12)


def run_backtest_handbook(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_balance: float = INITIAL_BALANCE,
    data_dir: Optional[str] = None,
) -> Tuple[Any, pd.DataFrame, Dict]:
    """
    Phase 1：仅 trend_long / trend_short / tightening，首仓 20%，无金字塔。
    返回 (portfolio_dict, aligned_df, extra)。
    """
    start_date = start_date or WARMUP_START
    end_date = end_date or BACKTEST_END

    # 1. 数据：4H + 12H（从预热开始）
    df_4h = load_or_fetch_4h(start_date, end_date, data_dir, use_ccxt=False)
    if df_4h.empty:
        raise RuntimeError("无 4H 数据，请先执行 --fetch 或确保 data/btc_4h.csv 存在")
    df_12h = resample_to_12h(df_4h)

    # 2. 指标
    df_4h = compute_4h_indicators(df_4h)
    df_12h = compute_12h_indicators(df_12h)

    # 3. 月线 Pivot（4H 聚合为日线：按 date 聚合，再交 indicators 的 compute_monthly_pivot）
    df_daily = df_4h.copy()
    df_daily["date"] = pd.to_datetime(df_daily["timestamp"], unit="ms", utc=True)
    daily_agg = df_daily.groupby(df_daily["date"].dt.date).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).reset_index()
    daily_agg["timestamp"] = (pd.to_datetime(daily_agg["date"]).astype("int64") // 10**6)  # ms
    df_pivot = compute_monthly_pivot(daily_agg) if len(daily_agg) >= 8 else pd.DataFrame()

    # 4. 对齐：每根 4H 取最近 12H 收盘的指标（仅 OHLCV 加 12h_ 前缀避免冲突）
    df_4h = df_4h.sort_values("timestamp").reset_index(drop=True)
    df_12h = df_12h.sort_values("timestamp").reset_index(drop=True)
    rename_12h = {c: f"12h_{c}" for c in ["open", "high", "low", "close", "volume", "timestamp"] if c in df_12h.columns}
    df_12h_renamed = df_12h.rename(columns=rename_12h)
    merged = pd.merge_asof(
        df_4h,
        df_12h_renamed,
        left_on="timestamp",
        right_on="12h_timestamp",
        direction="backward",
    )

    # 回测区间：仅 BACKTEST_START ~ BACKTEST_END
    start_ts = int(pd.Timestamp(BACKTEST_START).timestamp() * 1000)
    end_ts = int((pd.Timestamp(BACKTEST_END) + pd.Timedelta(days=1)).timestamp() * 1000)
    merged = merged[(merged["timestamp"] >= start_ts) & (merged["timestamp"] <= end_ts)].reset_index(drop=True)

    balance = initial_balance
    positions: List[SimplePosition] = []
    trades: List[SimpleTrade] = []
    equity_curve: List[Tuple[int, float]] = []

    # 12H 历史用于 Vegas 状态（最近 30 根 12H 的 144/576）
    ema144_12h_history: List[Tuple[float, float]] = []
    prev_vegas_state = "tightening"
    prev_mode = "transition"
    consecutive_losses = 0
    pending_entry: Optional[Dict] = None  # 12H 收盘产生信号 → 下一根 4H 开盘成交

    for i in range(len(merged)):
        row = merged.iloc[i]
        ts = int(row["timestamp"])
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        atr_4h = row.get("4h_atr") or c * 0.02

        # 限价单：上一根 12H 信号在本根 4H 开盘成交
        if pending_entry is not None and len(positions) == 0:
            pe = pending_entry
            pending_entry = None
            fill = o
            direction = pe["direction"]
            size = pe["size"]
            lev = pe["leverage"]
            soft, hard = pe["soft"], pe["hard"]
            tps = pe.get("take_profits") or []
            margin = balance * size
            contracts = margin * lev
            if margin < balance - 1:
                balance -= margin
                positions.append(SimplePosition(
                    direction=direction, entry_price=fill, entry_time=ts, entry_bar_idx=i,
                    size_pct=size, leverage=lev, soft_stop=soft, hard_stop=hard,
                    take_profits=tps, margin=margin, contracts=contracts,
                    entry_type=pe.get("entry_type", ""), mode=pe.get("mode", ""),
                ))

        # 更新 12H 历史（仅 12H 收盘时追加）
        is_12h = _is_12h_close(ts)
        ema144 = row.get("12h_vegas_144") or row.get("4h_vegas_144") or 0
        ema576 = row.get("12h_vegas_576") or row.get("4h_vegas_576") or 0
        if is_12h and ema144 and ema576:
            ema144_12h_history.append((float(ema144), float(ema576)))
            if len(ema144_12h_history) > 30:
                ema144_12h_history.pop(0)

        vegas_state = get_vegas_state(ema144, ema576, ema144_12h_history[-30:] if len(ema144_12h_history) >= 30 else ema144_12h_history)
        bull_bear = get_bull_bear(ema144, ema576)

        # 破位平仓
        if vegas_state == "breakout" and positions:
            for pos in list(positions):
                exit_p = c
                pnl = pos.unrealized_pnl(exit_p)
                balance += pos.margin + pnl
                trades.append(SimpleTrade(pos.direction, pos.entry_price, exit_p, pos.size_pct, pos.leverage, pnl, "vegas_breach", pos.mode, pos.entry_type))
                if pnl < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
            positions.clear()

        # 硬止损 / 止盈
        for pos in list(positions):
            if pos.direction == "long":
                if l <= pos.hard_stop:
                    balance += pos.margin + pos.unrealized_pnl(pos.hard_stop)
                    trades.append(SimpleTrade(pos.direction, pos.entry_price, pos.hard_stop, pos.size_pct, pos.leverage, pos.unrealized_pnl(pos.hard_stop), "hard_stop", pos.mode, pos.entry_type))
                    positions.remove(pos)
                    consecutive_losses += 1
                    continue
                for tp in list(pos.take_profits):
                    if tp["price"] > pos.entry_price and h >= tp["price"]:
                        reduce_pct = tp.get("reduce_pct", 0.5)
                        balance += pos.margin * reduce_pct + pos.unrealized_pnl(tp["price"]) * reduce_pct
                        pos.take_profits.remove(tp)
                        break
            else:
                if h >= pos.hard_stop:
                    balance += pos.margin + pos.unrealized_pnl(pos.hard_stop)
                    trades.append(SimpleTrade(pos.direction, pos.entry_price, pos.hard_stop, pos.size_pct, pos.leverage, pos.unrealized_pnl(pos.hard_stop), "hard_stop", pos.mode, pos.entry_type))
                    positions.remove(pos)
                    consecutive_losses += 1
                    continue

        # 软止损（12H 收盘）
        if is_12h and positions:
            for pos in list(positions):
                if pos.direction == "long" and c <= pos.soft_stop:
                    balance += pos.margin + pos.unrealized_pnl(pos.soft_stop)
                    trades.append(SimpleTrade(pos.direction, pos.entry_price, pos.soft_stop, pos.size_pct, pos.leverage, pos.unrealized_pnl(pos.soft_stop), "soft_stop", pos.mode, pos.entry_type))
                    positions.remove(pos)
                    consecutive_losses += 1
                elif pos.direction == "short" and c >= pos.soft_stop:
                    balance += pos.margin + pos.unrealized_pnl(pos.soft_stop)
                    trades.append(SimpleTrade(pos.direction, pos.entry_price, pos.soft_stop, pos.size_pct, pos.leverage, pos.unrealized_pnl(pos.soft_stop), "soft_stop", pos.mode, pos.entry_type))
                    positions.remove(pos)
                    consecutive_losses += 1

        # 保本移动（简化：仅更新 soft）
        atr_4h = row.get("4h_atr") or c * 0.02
        for pos in positions:
            stop_dist = abs(pos.entry_price - pos.hard_stop) / pos.entry_price * 100 if pos.entry_price else 5
            one_r = pos.margin * pos.leverage * stop_dist / 100 or 1
            r = pos.unrealized_pnl(c) / one_r
            vegas_4h_144 = row.get("4h_vegas_144") or 0
            ns, nh = update_stop_loss(pos.soft_stop, pos.hard_stop, pos.entry_price, pos.direction, r, vegas_4h_144, atr_4h)
            pos.soft_stop = max(pos.soft_stop, ns) if pos.direction == "long" else min(pos.soft_stop, ns)

        # 入场（Phase 1：仅首仓，12H 收盘决策 → 下一根 4H 开盘成交，挂 pending_entry）
        if is_12h and len(positions) == 0 and pending_entry is None and vegas_state != "tangled":
            swing_highs, swing_lows = detect_swing(
                merged["high"].values[: i + 1],
                merged["low"].values[: i + 1],
                6,
            )
            swing_highs = [s for s in swing_highs if s[0] <= i - 6]
            swing_lows = [s for s in swing_lows if s[0] <= i - 6]
            pivots = _row_pivots(row, df_pivot, ts)
            mode = get_market_mode(vegas_state, ema144, ema576, c, swing_highs, swing_lows, pivots)

            if mode in ("trend_long", "trend_short", "tightening"):
                direction = "long" if mode == "trend_long" or (mode == "tightening" and bull_bear == "bull") else "short"
                if mode == "tightening" and bull_bear == "bear":
                    direction = "short"

                vegas_4h_144 = row.get("4h_vegas_144") or ema144
                soft, hard = calc_stop_loss(c, direction, vegas_4h_144, ema144, ema576, atr_4h, pivots, c, tightening_layer1=(mode == "tightening"))
                from risk_management import _cap_hard_stop
                soft, hard = _cap_hard_stop(soft, hard, c, direction)

                stop_dist_pct = abs(c - hard) / c * 100 if hard else 5
                atr_pct = row.get("4h_atr_pct") or 4.0
                lev = calc_leverage(balance, 0, "trend", atr_pct, stop_dist_pct, consecutive_losses)
                tier = get_tier(balance)
                max_risk = TIERS[tier]["max_risk"]
                if FIRST_POSITION_PCT * lev * stop_dist_pct / 100 > max_risk and stop_dist_pct > 0:
                    lev = int(max_risk / (FIRST_POSITION_PCT * stop_dist_pct / 100))
                lev = max(1, min(TIERS[tier]["max_lev"], lev))

                size = FIRST_POSITION_PCT
                tps = calc_take_profit("B", direction, pivots, ema144, ema576, "S", mode=mode, current_price=c)
                pending_entry = {
                    "direction": direction, "size": size, "leverage": lev,
                    "soft": soft, "hard": hard, "take_profits": tps or [],
                    "entry_type": "right_144", "mode": mode,
                }
                prev_mode = mode
        prev_vegas_state = vegas_state

        # 权益
        eq = balance + sum(p.unrealized_pnl(c) for p in positions)
        equity_curve.append((ts, eq))

    # 平掉剩余仓
    if merged is not None and len(merged) > 0:
        last_row = merged.iloc[-1]
        last_c = last_row["close"]
        for pos in list(positions):
            balance += pos.margin + pos.unrealized_pnl(last_c)
            trades.append(SimpleTrade(pos.direction, pos.entry_price, last_c, pos.size_pct, pos.leverage, pos.unrealized_pnl(last_c), "end", pos.mode, pos.entry_type))
    positions.clear()

    portfolio = {
        "balance": balance,
        "trade_history": trades,
        "equity_history": equity_curve,
    }
    extra = {"monthly_mode": {}}
    return portfolio, merged, extra


def _row_pivots(row: pd.Series, df_pivot: pd.DataFrame, ts: int) -> Dict[str, float]:
    """当前 bar 对应的当月 Pivot（取 month_ts <= ts 的最近一行）。"""
    if df_pivot is None or len(df_pivot) == 0:
        return {}
    sub = df_pivot[df_pivot["month_ts"] <= ts]
    if len(sub) == 0:
        return {}
    r = sub.iloc[-1]
    return {k: float(r[k]) for k in ["P", "S1", "S2", "R1", "R2"] if k in r.index and pd.notna(r.get(k))}
