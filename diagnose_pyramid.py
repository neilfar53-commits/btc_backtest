# -*- coding: utf-8 -*-
"""
金字塔失败根因诊断：2024年10-12月（BTC 70k→100k）
输出止损明细、评分明细，用于排查
用法: python diagnose_pyramid.py
"""

import pandas as pd

from backtest_engine import run_backtest
from strategy import SCORE_TREND
from config import INITIAL_BALANCE


def diagnose_pyramid():
    """诊断 2024-10 至 2024-12 的金字塔问题"""
    pm, df, extra = run_backtest(
        initial_balance=INITIAL_BALANCE,
        start_date="2024-10-01",
        end_date="2024-12-31",
    )

    year, month_start, month_end = 2024, 10, 12
    trades = [
        t for t in pm.trade_history
        if pd.Timestamp(t.exit_time, unit="ms").year == year
        and month_start <= pd.Timestamp(t.exit_time, unit="ms").month <= month_end
    ]
    trades = sorted(trades, key=lambda x: x.exit_time)

    lines = [
        "\n" + "=" * 90,
        "  金字塔失败根因诊断：2024年10-12月（BTC 70k→100k）",
        "=" * 90,
    ]

    # 一、止损诊断
    lines.extend([
        "",
        "【一、止损诊断】",
        "-" * 90,
    ])

    hard_stop_count = sum(1 for t in trades if t.reason == "hard_stop")
    soft_stop_count = sum(1 for t in trades if t.reason == "soft_stop")
    tp_count = sum(1 for t in trades if t.reason == "tp")
    eod_count = sum(1 for t in trades if t.reason == "eod")

    lines.append(f"总交易数: {len(trades)}, 硬止损: {hard_stop_count}, 软止损: {soft_stop_count}, 止盈: {tp_count}, 期末: {eod_count}")

    # 硬止损距离统计
    hard_stop_distances = []
    for t in trades:
        if t.reason == "hard_stop" and t.hard_stop > 0:
            if t.direction == "long":
                dist_pct = (t.entry_price - t.hard_stop) / t.entry_price * 100
            else:
                dist_pct = (t.hard_stop - t.entry_price) / t.entry_price * 100
            hard_stop_distances.append(abs(dist_pct))

    avg_hard_dist = sum(hard_stop_distances) / len(hard_stop_distances) if hard_stop_distances else 0
    lines.append(f"硬止损距离入场价平均: {avg_hard_dist:.2f}%")
    if hard_stop_distances:
        lines.append(f"硬止损距离范围: {min(hard_stop_distances):.2f}% ~ {max(hard_stop_distances):.2f}%")
    lines.append("")

    # 每笔交易明细
    lines.append("交易明细 (entry_price, soft_stop, hard_stop, exit_price, exit_reason, holding_time):")
    lines.append("-" * 90)
    for i, t in enumerate(trades, 1):
        hold_days = (t.exit_time - t.entry_time) / (1000 * 86400) if t.exit_time > t.entry_time else 0
        lines.append(
            f"#{i} {t.direction} | entry={t.entry_price:.0f} | soft={getattr(t,'soft_stop',0):.0f} | "
            f"hard={getattr(t,'hard_stop',0):.0f} | exit={t.exit_price:.0f} | {t.reason} | {hold_days:.1f}天"
        )

    # 二、评分明细（来自 backtest extra）
    score_history = extra.get("score_history", [])
    lines.extend([
        "",
        "【二、评分明细】",
        "-" * 90,
        "（在12H收盘时，趋势模式下的评分，列出每项得分）",
    ])

    for sh in score_history[:40]:  # 前40个12H
        ts_str = pd.Timestamp(sh["ts"], unit="ms").strftime("%Y-%m-%d %H:%M")
        sigs = sh.get("signals", {})
        breakdown = []
        for key, pts in SCORE_TREND.items():
            if sigs.get(key):
                breakdown.append(f"{key}+{pts}")
        lines.append(
            f"  {ts_str} | close={sh['close']:.0f} | {sh['mode']} | score={sh['score']} | "
            f"{' '.join(breakdown) if breakdown else '-'}"
        )
    if len(score_history) > 40:
        lines.append(f"  ... 共 {len(score_history)} 根12H K线")

    lines.extend([
        "",
        "=" * 90,
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    print(diagnose_pyramid())
