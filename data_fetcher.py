# -*- coding: utf-8 -*-
"""
BTC 交易策略回测系统 — 数据获取模块

从 OKX 公开 API 获取 BTC/USDT 永续合约历史 K 线数据，无需 API Key。
支持 4H 主周期，并聚合生成 12H、日线、周线数据。
"""

import os
import time

import pandas as pd
import requests

from config import (
    BACKTEST_END,
    BACKTEST_START,
    BAR_4H,
    CANDLE_LIMIT,
    DATA_DIR,
    INST_ID,
    OKX_BASE_URL,
    OKX_HISTORY_CANDLES_PATH,
)


def fetch_4h_candles(start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 OKX API 获取 4H K 线数据（分页获取）。

    API 说明：
    - 端点: GET /api/v5/market/history-candles
    - 返回格式: [[ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm], ...]
    - 数据按时间倒序（最新在前）
    - after: 获取该时间戳之前（更早）的数据

    Args:
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期 'YYYY-MM-DD'

    Returns:
        DataFrame 列: timestamp, open, high, low, close, volume
    """
    url = f"{OKX_BASE_URL}{OKX_HISTORY_CANDLES_PATH}"
    # 使用当日结束时间戳，确保包含 end_date 最后一根 4H K 线
    end_ts = int(
        (pd.Timestamp(end_date) + pd.Timedelta(days=1)).timestamp() * 1000
    )
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)

    all_candles = []

    # after: 获取比该时间戳更早的数据。首次用 end_ts，后续用本批最旧时间戳
    after = end_ts

    while True:
        params = {
            "instId": INST_ID,
            "bar": BAR_4H,
            "limit": str(CANDLE_LIMIT),
            "after": str(after),
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"OKX API 请求失败: {e}") from e

        if data.get("code") != "0":
            raise RuntimeError(f"OKX API 返回错误: {data}")

        candles = data.get("data", [])
        if not candles:
            break

        for c in candles:
            ts = int(c[0])
            if start_ts <= ts < end_ts:
                all_candles.append(
                    {
                        "timestamp": ts,
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    }
                )

        # after 取本批最旧一根的时间戳，用于获取更早的数据
        oldest_ts = int(candles[-1][0])
        if oldest_ts <= start_ts:
            break

        after = oldest_ts
        time.sleep(0.2)  # 避免请求过快，遵守 API 限频

    if not all_candles:
        raise RuntimeError("未获取到任何 K 线数据，请检查日期范围或网络")

    df = pd.DataFrame(all_candles)
    df = df.sort_values("timestamp").drop_duplicates().reset_index(drop=True)
    return df


def aggregate_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    将 4H K 线聚合为更大周期。

    聚合规则：
    - open = 第一根的 open
    - high = max(所有 high)
    - low = min(所有 low)
    - close = 最后一根的 close
    - volume = sum(所有 volume)

    Args:
        df: 4H K 线 DataFrame，需含 timestamp, open, high, low, close, volume
        rule: pandas resample 规则，如 '12h', '1d', '1w'

    Returns:
        聚合后的 DataFrame
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime")

    agg_df = (
        df.resample(rule, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )

    # pandas DatetimeIndex.astype("int64") 为纳秒，统一存为毫秒便于跨平台
    agg_df["timestamp"] = (agg_df.index.astype("int64") // 10**6).astype("int64")
    agg_df = agg_df.reset_index(drop=True)
    return agg_df[["timestamp", "open", "high", "low", "close", "volume"]]


def aggregate_12h(df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    从 4H 聚合为 12H。
    12H 周期：UTC 00:00-12:00 / UTC 12:00-00:00
    """
    return aggregate_ohlcv(df_4h, "12h")


def aggregate_daily(df_4h: pd.DataFrame) -> pd.DataFrame:
    """从 4H 聚合为日线（6 根 4H = 1 天）。"""
    return aggregate_ohlcv(df_4h, "1D")


def aggregate_weekly(df_4h: pd.DataFrame) -> pd.DataFrame:
    """从 4H 聚合为周线（42 根 4H = 1 周）。"""
    return aggregate_ohlcv(df_4h, "1W")


def save_csv(df: pd.DataFrame, path: str) -> None:
    """保存 DataFrame 为 CSV，列：timestamp, open, high, low, close, volume"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"已保存: {path} ({len(df)} 行)")


def run_data_fetcher(
    start_date: str = None,
    end_date: str = None,
    data_dir: str = None,
) -> None:
    """
    执行完整的数据获取流程：获取 4H → 聚合 12H/日/周 → 保存 CSV。

    Args:
        start_date: 开始日期，默认使用 config.BACKTEST_START
        end_date: 结束日期，默认使用 config.BACKTEST_END
        data_dir: 数据目录，默认使用 config.DATA_DIR
    """
    start_date = start_date or BACKTEST_START
    end_date = end_date or BACKTEST_END
    data_dir = data_dir or DATA_DIR
    os.makedirs(data_dir, exist_ok=True)

    print(f"获取 4H K 线: {start_date} ~ {end_date}")
    df_4h = fetch_4h_candles(start_date, end_date)
    path_4h = os.path.join(data_dir, "btc_4h.csv")
    save_csv(df_4h, path_4h)

    print("聚合 12H...")
    df_12h = aggregate_12h(df_4h)
    path_12h = os.path.join(data_dir, "btc_12h.csv")
    save_csv(df_12h, path_12h)

    print("聚合日线...")
    df_daily = aggregate_daily(df_4h)
    path_daily = os.path.join(data_dir, "btc_daily.csv")
    save_csv(df_daily, path_daily)

    print("聚合周线...")
    df_weekly = aggregate_weekly(df_4h)
    path_weekly = os.path.join(data_dir, "btc_weekly.csv")
    save_csv(df_weekly, path_weekly)

    print("数据获取完成。")


if __name__ == "__main__":
    run_data_fetcher()
