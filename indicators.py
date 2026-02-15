# -*- coding: utf-8 -*-
"""
BTC 交易策略回测系统 — 指标计算模块

实现 Vegas 通道(EMA)、均线(SMA)、RSI、ATR、月线 Pivot、
Swing High/Low 识别、MA 交叉检测、K 线形态识别。
"""

import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from config import DATA_12H, DATA_4H, DATA_DAILY, DATA_DIR, DATA_WEEKLY


# ==================== 基础指标函数 ====================


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI 相对强弱指数
    RSI = 100 - 100 / (1 + RS), RS = 平均涨幅 / 平均跌幅
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """
    ATR 真实波幅
    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    ATR = TR 的 EMA
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ==================== 各周期指标计算 ====================


def _add_vegas(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """为 DataFrame 添加 Vegas 通道 (EMA 144/169/576/676)"""
    close = df["close"]
    df[f"{prefix}_vegas_144"] = ema(close, 144)
    df[f"{prefix}_vegas_169"] = ema(close, 169)
    df[f"{prefix}_vegas_576"] = ema(close, 576)
    df[f"{prefix}_vegas_676"] = ema(close, 676)
    return df


def _add_ma(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """根据周期添加 MA"""
    close = df["close"]
    if prefix == "4h":
        df[f"{prefix}_ma8"] = sma(close, 8)
        df[f"{prefix}_ma17"] = sma(close, 17)
        df[f"{prefix}_ma55"] = sma(close, 55)
    elif prefix == "12h":
        df[f"{prefix}_ma8"] = sma(close, 8)
        df[f"{prefix}_ma17"] = sma(close, 17)
        df[f"{prefix}_ma55"] = sma(close, 55)
        df[f"{prefix}_ma200"] = sma(close, 200)
    return df


def compute_4h_indicators(df_4h: pd.DataFrame) -> pd.DataFrame:
    """计算 4H 周期所有指标"""
    df = df_4h.copy()
    _add_vegas(df, "4h")
    _add_ma(df, "4h")
    df["4h_rsi"] = rsi(df["close"], 14)
    df["4h_atr"] = atr(df["high"], df["low"], df["close"], 14)
    df["4h_atr_pct"] = df["4h_atr"] / df["close"] * 100
    return df


def compute_12h_indicators(df_12h: pd.DataFrame) -> pd.DataFrame:
    """计算 12H 周期所有指标"""
    df = df_12h.copy()
    _add_vegas(df, "12h")
    _add_ma(df, "12h")
    df["12h_rsi"] = rsi(df["close"], 14)
    df["12h_atr"] = atr(df["high"], df["low"], df["close"], 14)
    df["12h_atr_pct"] = df["12h_atr"] / df["close"] * 100
    return df


def compute_weekly_indicators(df_weekly: pd.DataFrame) -> pd.DataFrame:
    """计算周线周期 Vegas (144/169)"""
    df = df_weekly.copy()
    close = df["close"]
    df["weekly_vegas_144"] = ema(close, 144)
    df["weekly_vegas_169"] = ema(close, 169)
    return df


# ==================== 月线 Pivot ====================


def calc_classic_pivot(h: float, l: float, c: float) -> dict:
    """
    Classic Pivot 计算
    P = (H+L+C)/3
    S1=2P-H, S2=P-(H-L), S3=L-2(H-P), S4=L-3(H-P)
    R1=2P-L, R2=P+(H-L), R3=H+2(P-L), R4=H+3(P-L)
    """
    p = (h + l + c) / 3
    return {
        "P": p,
        "S1": 2 * p - h,
        "S2": p - (h - l),
        "S3": l - 2 * (h - p),
        "S4": l - 3 * (h - p),
        "R1": 2 * p - l,
        "R2": p + (h - l),
        "R3": h + 2 * (p - l),
        "R4": h + 3 * (p - l),
    }


def compute_monthly_pivot(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    计算月线 Pivot，每月 1 日更新，使用上月的 H/L/C。
    当月 Pivot 用上月 H/L/C 计算；前 1 月 Pivot 用前 2 月 H/L/C，依此类推。

    返回列: month_ts, P, S1-S4, R1-R4 (当月), P_prev1, S1_prev1, ... (前1月), ... 至 prev5
    权重: 当月100% / 前1月80% / 前2月60% / 前3月40% / 前4月25% / 前5月15%
    """
    df = df_daily.copy()
    ts = df["timestamp"]
    # 兼容: ms(13位) | 错误除过10^6的ms(7位) | s(10位)
    if ts.max() < 1e9:
        ts = ts * 1_000_000  # 恢复被错误除过的 ms
    elif ts.max() < 1e12:
        ts = ts * 1000  # 秒 → 毫秒
    df["date"] = pd.to_datetime(ts, unit="ms", utc=True)
    df["month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby("month")
        .agg({"high": "max", "low": "min", "close": "last"})
        .reset_index()
    )

    pivot_rows = []
    for i in range(7, len(monthly)):  # 需要至少 7 个月历史（当月 + 前6月用于计算）
        row = {"month": monthly.iloc[i]["month"]}
        row["month_ts"] = monthly.iloc[i]["month"].to_timestamp().value // 10**6

        # 当月 Pivot 用上月(i-1)的 H/L/C；前j月 Pivot 用前(j+1)月的 H/L/C
        for j in range(6):
            m = monthly.iloc[i - 1 - j]  # 当月→上月, 前1月→前2月, ...
            h, l, c = m["high"], m["low"], m["close"]
            piv = calc_classic_pivot(h, l, c)
            suffix = "_prev" + str(j) if j > 0 else ""
            for k, v in piv.items():
                row[k + suffix] = v

        pivot_rows.append(row)

    return pd.DataFrame(pivot_rows)


# ==================== Swing High/Low ====================


def detect_swing(
    highs: Union[np.ndarray, pd.Series],
    lows: Union[np.ndarray, pd.Series],
    lookback: int = 2,
) -> Tuple[list, list]:
    """
    识别 Swing High 和 Swing Low。
    需后 lookback 根 K 线确认，故 bar i 的 swing 只能在 bar i+lookback 时使用。

    Returns:
        (swing_highs, swing_lows) 各为 [(index, value), ...]，保留最近 5 个
    """
    if hasattr(highs, "values"):
        highs = highs.values
    if hasattr(lows, "values"):
        lows = lows.values

    swing_highs = []
    swing_lows = []
    n = len(highs)

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback : i + lookback + 1]
        if highs[i] == np.max(window_high):
            swing_highs.append((i, float(highs[i])))

        window_low = lows[i - lookback : i + lookback + 1]
        if lows[i] == np.min(window_low):
            swing_lows.append((i, float(lows[i])))

    return swing_highs[-5:], swing_lows[-5:]


# ==================== MA 交叉检测 ====================


def detect_cross(
    fast_ma: Union[np.ndarray, pd.Series],
    slow_ma: Union[np.ndarray, pd.Series],
    index: int,
    lookback: int = 5,
    cross_limit: int = 3,
) -> Optional[str]:
    """
    检测 MA 交叉状态。

    Returns:
        'golden'  : 金叉 (fast 上穿 slow)
        'death'   : 死叉 (fast 下穿 slow)
        'entangled': 缠绕 (lookback 内交叉次数 >= cross_limit)
        None      : 无交叉
    """
    if hasattr(fast_ma, "values"):
        fast_ma = fast_ma.values
    if hasattr(slow_ma, "values"):
        slow_ma = slow_ma.values

    if index < 1 or index >= len(fast_ma) or index >= len(slow_ma):
        return None

    # 当前金叉：fast > slow 且 前一根 fast <= slow
    if fast_ma[index] > slow_ma[index] and fast_ma[index - 1] <= slow_ma[index - 1]:
        return "golden"
    # 当前死叉：fast < slow 且 前一根 fast >= slow
    if fast_ma[index] < slow_ma[index] and fast_ma[index - 1] >= slow_ma[index - 1]:
        return "death"

    # 缠绕检测：lookback 根内交叉次数 >= cross_limit
    start = max(1, index - lookback + 1)
    crosses = 0
    for i in range(start, index + 1):
        if (fast_ma[i] > slow_ma[i]) != (fast_ma[i - 1] > slow_ma[i - 1]):
            crosses += 1
    if crosses >= cross_limit:
        return "entangled"

    return None


# ==================== K 线形态识别 ====================


def detect_pin_bar(
    open_: float,
    high: float,
    low: float,
    close: float,
    direction: str,
) -> bool:
    """
    Pin Bar 识别
    做多 Pin Bar: 下影线 > 实体的 2 倍，且收阳 (下影线占全幅 > 67%)
    做空 Pin Bar: 上影线 > 实体的 2 倍，且收阴 (上影线占全幅 > 67%)
    """
    body = abs(close - open_)
    full_range = high - low
    if full_range <= 0:
        return False
    if direction == "bull":
        lower_wick = min(open_, close) - low
        return lower_wick / full_range > 0.67 and close > open_
    else:  # bear
        upper_wick = high - max(open_, close)
        return upper_wick / full_range > 0.67 and close < open_


def detect_pin_bar_row(row: pd.Series, direction: str) -> bool:
    """Pin Bar 识别（接受 DataFrame 的一行）"""
    return detect_pin_bar(
        row["open"], row["high"], row["low"], row["close"], direction
    )


def detect_engulfing(
    curr_open: float,
    curr_high: float,
    curr_low: float,
    curr_close: float,
    prev_open: float,
    prev_high: float,
    prev_low: float,
    prev_close: float,
    direction: str,
) -> bool:
    """
    吞没形态识别
    看涨吞没: 前阴后阳，当前实体完全包住前一根
    看跌吞没: 前阳后阴，当前实体完全包住前一根
    """
    if direction == "bull":
        return (
            prev_close < prev_open
            and curr_close > curr_open
            and curr_close > prev_open
            and curr_open < prev_close
        )
    else:
        return (
            prev_close > prev_open
            and curr_close < curr_open
            and curr_close < prev_open
            and curr_open > prev_close
        )


def detect_engulfing_row(
    curr: pd.Series, prev: pd.Series, direction: str
) -> bool:
    """吞没形态识别（接受 DataFrame 的两行）"""
    return detect_engulfing(
        curr["open"],
        curr["high"],
        curr["low"],
        curr["close"],
        prev["open"],
        prev["high"],
        prev["low"],
        prev["close"],
        direction,
    )


# ==================== 主入口：计算所有指标 ====================


def load_ohlcv(data_dir: str = None) -> dict:
    """加载各周期 OHLCV 数据"""
    data_dir = data_dir or DATA_DIR
    return {
        "4h": pd.read_csv(os.path.join(data_dir, "btc_4h.csv")),
        "12h": pd.read_csv(os.path.join(data_dir, "btc_12h.csv")),
        "daily": pd.read_csv(os.path.join(data_dir, "btc_daily.csv")),
        "weekly": pd.read_csv(os.path.join(data_dir, "btc_weekly.csv")),
    }


def compute_all_indicators(data_dir: str = None) -> dict:
    """
    加载数据并计算所有指标。

    Returns:
        dict 包含:
            - df_4h: 4H 数据 + vegas, ma8/17/55, rsi, atr, atr_pct
            - df_12h: 12H 数据 + vegas, ma55/200, rsi, atr, atr_pct
            - df_weekly: 周线数据 + vegas_144, vegas_169
            - df_pivot: 月线 Pivot (每月 + 前 5 月)
    """
    data_dir = data_dir or DATA_DIR
    data = load_ohlcv(data_dir)

    return {
        "df_4h": compute_4h_indicators(data["4h"]),
        "df_12h": compute_12h_indicators(data["12h"]),
        "df_weekly": compute_weekly_indicators(data["weekly"]),
        "df_pivot": compute_monthly_pivot(data["daily"]),
        "df_daily": data["daily"],  # 原始日线，策略可能需要
    }


if __name__ == "__main__":
    result = compute_all_indicators()
    print("4H 指标列:", result["df_4h"].columns.tolist())
    print("12H 指标列:", result["df_12h"].columns.tolist())
    print("周线指标列:", result["df_weekly"].columns.tolist())
    print("Pivot 列:", result["df_pivot"].columns.tolist())
    print("4H 行数:", len(result["df_4h"]))
    print("Pivot 行数:", len(result["df_pivot"]))
