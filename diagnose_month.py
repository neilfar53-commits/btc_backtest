# -*- coding: utf-8 -*-
"""
诊断指定月份的交易明细，用于排查盈亏计算 bug
用法: python diagnose_month.py 2023-09
"""

import sys
import pandas as pd

from backtest_engine import run_backtest
from config import INITIAL_BALANCE, TIERS


def diagnose_month(ym: str = "2023-09"):
    """输出指定月份的所有交易明细及 balance 变化"""
    pm, df, _ = run_backtest(initial_balance=INITIAL_BALANCE)

    year, month = int(ym[:4]), int(ym[5:7])
    trades = [
        t for t in pm.trade_history
        if pd.Timestamp(t.exit_time, unit="ms").year == year
        and pd.Timestamp(t.exit_time, unit="ms").month == month
    ]
    trades = sorted(trades, key=lambda x: x.exit_time)

    # 还需跟踪 partial_close，但 Trade 里没有。先输出 full close
    lines = [
        f"\n{'='*80}",
        f"  {ym} 交易明细诊断",
        f"{'='*80}",
        f"该月 full close 交易数: {len(trades)}",
        "",
    ]

    max_risk_pct = TIERS["S"]["max_risk"] * 100  # 8%

    for i, t in enumerate(trades, 1):
        margin = getattr(t, "margin", 0) or (t.size_pct * 100)  # 估算
        contracts = margin * t.leverage
        # 验证 PnL 公式
        if t.direction == "long":
            expected_pnl = contracts * (t.exit_price - t.entry_price) / t.entry_price
        else:
            expected_pnl = contracts * (t.entry_price - t.exit_price) / t.entry_price
        from backtest_engine import calc_trade_cost
        fee = calc_trade_cost(contracts, is_maker=False)
        expected_pnl -= fee

        bal_before = getattr(t, "balance_before", 0)
        bal_after = getattr(t, "balance_after", 0)
        delta = bal_after - bal_before if bal_after and bal_before else 0

        # 单笔亏损是否超过 max_risk
        loss_pct = abs(t.pnl) / bal_before * 100 if t.pnl < 0 and bal_before else 0
        risk_ok = "OK" if loss_pct <= max_risk_pct else f"!!! 超限 {loss_pct:.1f}% > {max_risk_pct}%"

        lines.extend([
            f"--- 交易 #{i} ({pd.Timestamp(t.exit_time, unit='ms').strftime('%Y-%m-%d %H:%M')}) ---",
            f"  direction: {t.direction}, reason: {t.reason}",
            f"  entry_price: {t.entry_price:.2f}, exit_price: {t.exit_price:.2f}",
            f"  size_pct: {t.size_pct*100:.0f}%, leverage: {t.leverage}x",
            f"  margin(保证金): {margin:.2f} USDT, contracts(名义): {contracts:.2f} USDT",
            f"  pnl(记录): {t.pnl:.2f}, expected_pnl(公式): {expected_pnl:.2f}, fee: {fee:.2f}",
            f"  balance_before: {bal_before:.2f}, balance_after: {bal_after:.2f}, delta: {delta:.2f}",
            f"  单笔亏损占balance: {loss_pct:.2f}% {risk_ok}",
            "",
        ])

    # 汇总 + 全期间资金流水
    total_pnl = sum(t.pnl for t in trades)
    total_open_cost = getattr(pm, "total_open_cost", 0)
    total_funding = getattr(pm, "total_funding", 0)
    sum_full_close = sum(t.pnl for t in pm.trade_history)
    sum_partial = sum(p["pnl"] for p in pm.partial_close_history)

    lines.extend([
        f"{'='*80}",
        f"该月 full close 总 PnL: {total_pnl:.2f}",
        "",
        f"【全期间资金流水校验】",
        f"  初始 balance: {INITIAL_BALANCE}",
        f"  最终 balance: {pm.balance:.2f}",
        f"  全期间 full close PnL 总和: {sum_full_close:.2f}",
        f"  全期间 partial close PnL 总和: {sum_partial:.2f}",
        f"  累计开仓成本(手续费+滑点): {total_open_cost:.2f}",
        f"  累计资金费率: {total_funding:.2f}",
        f"  校验: 初始 + 实现PnL - 成本 - 资金费 = {INITIAL_BALANCE + sum_full_close + sum_partial - total_open_cost - total_funding:.2f}",
        f"{'='*80}\n",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    ym = sys.argv[1] if len(sys.argv) > 1 else "2023-09"
    print(diagnose_month(ym))
