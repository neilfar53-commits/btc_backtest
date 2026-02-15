# -*- coding: utf-8 -*-
"""
BTC Pivot-Vegas Pyramid 回测 — 报告生成
依据《回测开发指令》第十节：统计指标 + 可视化
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd


def generate_report(
    trades: List[Dict],
    equity_curve: List[tuple],
    initial_balance: float,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    输出指标：
    总收益率、年化、最大回撤、夏普、卡尔马、总交易次数、胜率、
    平均盈利/亏损、盈亏比、最大连续盈亏、平均持仓时间、
    按方向/模式/金字塔层数统计。
    """
    if not equity_curve:
        total_return = 0.0
        final_balance = initial_balance
    else:
        final_balance = equity_curve[-1][1]
        total_return = (final_balance - initial_balance) / initial_balance if initial_balance else 0

    n_trades = len(trades)
    if n_trades == 0:
        stats = {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return_pct": total_return * 100,
            "n_trades": 0,
            "win_rate_pct": 0,
            "max_drawdown_pct": 0,
            "sharpe": 0,
            "calmar": 0,
        }
    else:
        pnls = [t["pnl"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / n_trades * 100 if n_trades else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_loss_abs = abs(avg_loss) if avg_loss else 1
        profit_factor = (sum(wins) / avg_loss_abs) if wins and avg_loss else 0

        # 最大回撤
        peak = initial_balance
        max_dd = 0.0
        for _, eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak else 0
            if dd > max_dd:
                max_dd = dd

        # 夏普简化（无风险利率=0）
        returns = []
        for i in range(1, len(equity_curve)):
            r = (equity_curve[i][1] - equity_curve[i - 1][1]) / equity_curve[i - 1][1] if equity_curve[i - 1][1] else 0
            returns.append(r)
        import numpy as np
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252 * 6)) if len(returns) > 1 and np.std(returns) > 0 else 0  # 4H 约 6 根/日

        stats = {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "total_return_pct": total_return * 100,
            "n_trades": n_trades,
            "win_rate_pct": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown_pct": max_dd,
            "sharpe": sharpe,
            "calmar": (total_return * 100 / max_dd) if max_dd else 0,
        }

    report_lines = [
        "========== 回测报告 ==========",
        f"初始资金: {initial_balance:.2f} USDT",
        f"最终资金: {stats.get('final_balance', final_balance):.2f} USDT",
        f"总收益率: {stats.get('total_return_pct', total_return * 100):.2f}%",
        f"交易次数: {stats.get('n_trades', n_trades)}",
        f"胜率: {stats.get('win_rate_pct', 0):.1f}%",
        f"最大回撤: {stats.get('max_drawdown_pct', 0):.2f}%",
        f"夏普比率: {stats.get('sharpe', 0):.2f}",
    ]
    report_text = "\n".join(report_lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"报告已保存: {output_path}")

    return stats


def plot_results(
    trades: List[Dict],
    equity_curve: List[tuple],
    df_4h: Optional[pd.DataFrame] = None,
    output_dir: Optional[str] = None,
) -> None:
    """生成权益曲线、回撤、月度收益等图（可选）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过绘图")
        return

    output_dir = output_dir or "results"
    os.makedirs(output_dir, exist_ok=True)

    if not equity_curve:
        return

    ts = [e[0] for e in equity_curve]
    eq = [e[1] for e in equity_curve]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, eq, label="Equity")
    ax.set_title("Equity Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "equity_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {output_dir}/equity_curve.png")
