# -*- coding: utf-8 -*-
"""
BTC Pivot-Vegas Pyramid 回测 — 数据层
依据《回测开发指令》第二节：CCXT 获取 4H + 多周期合成 + 月线 Pivot
"""

import os
from typing import Dict, List, Optional

import pandas as pd

from config import (
    BACKTEST_END,
    BACKTEST_START,
    DATA_DIR,
    WARMUP_START,
)


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "4h",
    start: Optional[str] = None,
    end: Optional[str] = None,
    exchange_id: str = "okx",
    limit_per_request: int = 300,
) -> pd.DataFrame:
    """
    通过 CCXT 获取历史 K 线。
    需要足够历史以预热 EMA576（约 576×12H ≈ 6912H 的 4H 数据），
    故实际从 WARMUP_START 获取，回测从 BACKTEST_START 开始。
    返回 DataFrame: timestamp, open, high, low, close, volume
    """
    try:
        import ccxt
    except ImportError:
        raise ImportError("请安装 ccxt: pip install ccxt")

    start = start or WARMUP_START
    end = end or BACKTEST_END
    exchange = getattr(ccxt, exchange_id)()
    since_ts = int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000)

    all_ohlcv = []
    while since_ts < end_ts:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since_ts, limit=limit_per_request)
        if not ohlcv:
            break
        for t, o, h, l, c, v in ohlcv:
            if end_ts <= t:
                break
            if t >= since_ts:
                all_ohlcv.append({"timestamp": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
        since_ts = ohlcv[-1][0] + 1
        if len(ohlcv) < limit_per_request:
            break

    df = pd.DataFrame(all_ohlcv)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def resample_to_12h(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    4H → 12H 合成。
    12H 对齐 UTC 00:00 和 12:00；每根 12H = 3 根 4H (00:00, 04:00, 08:00 → 收盘 12:00)。
    """
    df = df_4h.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    agg = df.resample("12h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    agg["timestamp"] = agg.index.astype("int64") // 10**6
    return agg.reset_index(drop=True)[["timestamp", "open", "high", "low", "close", "volume"]]


def resample_to_monthly(df_4h: pd.DataFrame) -> pd.DataFrame:
    """4H → 月线合成，用于计算 Pivot（每月 H/L/C）。"""
    df = df_4h.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")
    agg = df.resample("ME").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    agg["month_ts"] = agg.index.astype("int64") // 10**6
    return agg.reset_index(drop=True)


def compute_monthly_pivot(monthly_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Classic Pivot：使用上月 H/L/C。
    P = (H+L+C)/3
    S1=2P-H, S2=P-(H-L), S3=L-2(H-P), S4=L-3(H-P)
    R1=2P-L, R2=P+(H-L), R3=H+2(P-L), R4=H+3(P-L)
    返回 {month_key: {P, S1..S4, R1..R4}}，需保存当月+前5个月共6个月。
    """
    result = {}
    for i in range(1, len(monthly_df)):
        row = monthly_df.iloc[i - 1]  # 上月
        h, l, c = row["high"], row["low"], row["close"]
        p = (h + l + c) / 3
        result[str(monthly_df.iloc[i].get("month_ts", i))] = {
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
    return result


def load_or_fetch_4h(
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: Optional[str] = None,
    use_ccxt: bool = True,
) -> pd.DataFrame:
    """
    若本地已有 data/btc_4h.csv 且覆盖 [start,end]，则加载；否则调用 fetch_ohlcv（或 data_fetcher）。
    """
    data_dir = data_dir or DATA_DIR
    start = start or WARMUP_START
    end = end or BACKTEST_END
    path_4h = os.path.join(data_dir, "btc_4h.csv")

    if os.path.isfile(path_4h):
        df = pd.read_csv(path_4h)
        df["timestamp"] = df["timestamp"].astype(int)
        start_ts = int(pd.Timestamp(start).timestamp() * 1000)
        end_ts = int((pd.Timestamp(end) + pd.Timedelta(days=1)).timestamp() * 1000)
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        if len(df) > 0:
            return df.reset_index(drop=True)

    if use_ccxt:
        return fetch_ohlcv(start=start, end=end)
    # 回退到原有 data_fetcher
    from data_fetcher import fetch_4h_candles
    os.makedirs(data_dir, exist_ok=True)
    df = fetch_4h_candles(start, end)
    df.to_csv(path_4h, index=False)
    return df


def run_data_pipeline(
    start: Optional[str] = None,
    end: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> tuple:
    """
    执行：获取/加载 4H → 合成 12H → 保存。
    返回 (df_4h, df_12h)。
    """
    data_dir = data_dir or DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    start = start or WARMUP_START
    end = end or BACKTEST_END

    df_4h = load_or_fetch_4h(start, end, data_dir)
    path_4h = os.path.join(data_dir, "btc_4h.csv")
    df_4h.to_csv(path_4h, index=False)

    df_12h = resample_to_12h(df_4h)
    path_12h = os.path.join(data_dir, "btc_12h.csv")
    df_12h.to_csv(path_12h, index=False)

    return df_4h, df_12h
