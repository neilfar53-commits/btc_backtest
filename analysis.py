# -*- coding: utf-8 -*-
"""
BTC 交易策略回测系统 — 统计分析与可视化模块
"""

import os
import warnings
from typing import List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

from config import INITIAL_BALANCE

# 统一设置中文字体：Windows 用黑体/雅黑，Mac 用苹方/黑体-简，避免中文变方框
mpl.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei",           # Windows
    "PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS",  # macOS
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False

# 抑制「缺少中文字形」的刷屏警告（Mac 默认 DejaVu 无中文时会报，图仍会保存）
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*", category=UserWarning)


def generate_report(
    trade_history: List,
    equity_curve: List[Tuple[int, float]],
    initial_balance: float = None,
    monthly_mode: dict = None,
    pyramid_score_5_plus: int = 0,
) -> str:
    """
    生成回测报告文本。
    """
    initial_balance = initial_balance or INITIAL_BALANCE
    if not equity_curve:
        return "无权益数据"

    final_balance = equity_curve[-1][1]
    total_return_pct = (final_balance - initial_balance) / initial_balance * 100

    # 年化收益率
    if len(equity_curve) >= 2:
        days = (equity_curve[-1][0] - equity_curve[0][0]) / (1000 * 86400)
        years = max(0.01, days / 365)
        annual_return = (final_balance / initial_balance) ** (1 / years) - 1
        annual_return_pct = annual_return * 100
    else:
        annual_return_pct = total_return_pct

    # 最大回撤
    equities = np.array([e[1] for e in equity_curve])
    peak = np.maximum.accumulate(equities)
    drawdown_pct = (peak - equities) / np.where(peak > 0, peak, 1) * 100
    max_dd_pct = float(np.max(drawdown_pct)) if len(drawdown_pct) > 0 else 0

    # 最大回撤持续时间
    ts = np.array([e[0] for e in equity_curve])
    in_dd = drawdown_pct > 0.01
    max_dd_days = 0
    i = 0
    while i < len(in_dd):
        if in_dd[i]:
            start_ts = ts[i]
            while i < len(in_dd) and in_dd[i]:
                i += 1
            if i < len(ts):
                end_ts = ts[i - 1]
                dur = (end_ts - start_ts) / (1000 * 86400)
                max_dd_days = max(max_dd_days, dur)
        else:
            i += 1

    # 夏普比率（假设无风险利率 3%，按日收益率年化）
    rf = 0.03
    if len(equities) >= 2:
        returns = np.diff(equities) / np.where(equities[:-1] > 0, equities[:-1], 1)
        vol = np.std(returns)
        ann_vol = vol * np.sqrt(365 * 6) if vol > 0 else 1e-6  # 4H bars, 6 per day
        sharpe = (annual_return_pct / 100 - rf) / ann_vol if ann_vol > 0 else 0
    else:
        sharpe = 0

    # 卡尔马比率
    calmar = annual_return_pct / max_dd_pct if max_dd_pct > 0 else 0

    # 交易统计
    n_trades = len(trade_history)
    wins = [t for t in trade_history if t.pnl > 0]
    losses = [t for t in trade_history if t.pnl <= 0]
    win_rate = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # 连续盈亏
    max_consec_win = 0
    max_consec_loss = 0
    cw, cl = 0, 0
    for t in trade_history:
        if t.pnl > 0:
            cw += 1
            cl = 0
            max_consec_win = max(max_consec_win, cw)
        else:
            cl += 1
            cw = 0
            max_consec_loss = max(max_consec_loss, cl)

    # 平均持仓时间（天）
    if trade_history:
        holds = [
            (t.exit_time - t.entry_time) / (1000 * 86400)
            for t in trade_history
            if t.exit_time > t.entry_time
        ]
        avg_hold_days = np.mean(holds) if holds else 0
    else:
        avg_hold_days = 0

    monthly_mode = monthly_mode or {}

    # 金字塔统计：按 pyramid_id 分组，计算每笔交易的最大层数
    pyramid_groups: dict = {}
    for t in trade_history:
        pid = getattr(t, "pyramid_id", 0) or 0
        if pid not in pyramid_groups:
            pyramid_groups[pid] = []
        pyramid_groups[pid].append(t)
    max_layers_per_trade = [max(t.pyramid_layer for t in grp) for grp in pyramid_groups.values()]
    avg_pyramid_layers = np.mean(max_layers_per_trade) if max_layers_per_trade else 1.0
    trades_3plus_layers = sum(1 for m in max_layers_per_trade if m >= 3)

    # 分年统计
    annual_stats = {}
    prev_year_end_equity = initial_balance
    for year in ["2023", "2024", "2025"]:
        year_int = int(year)
        year_trades = [t for t in trade_history if pd.Timestamp(t.exit_time, unit="ms").year == year_int]
        year_equity = [(ts, eq) for ts, eq in equity_curve if pd.Timestamp(ts, unit="ms").year == year_int]
        year_start_bal = prev_year_end_equity
        if year_equity:
            year_start_bal = year_equity[0][1]
            year_end_bal = year_equity[-1][1]
            prev_year_end_equity = year_end_bal
            year_return = (year_end_bal - year_start_bal) / year_start_bal * 100 if year_start_bal else 0
            eqs = np.array([e[1] for e in year_equity])
            peak = np.maximum.accumulate(eqs)
            dd = (peak - eqs) / np.where(peak > 0, peak, 1) * 100
            max_dd = float(np.max(dd)) if len(dd) > 0 else 0
        else:
            year_return = 0
            max_dd = 0
        annual_stats[year] = {
            "return_pct": year_return,
            "max_dd_pct": max_dd,
            "n_trades": len(year_trades),
        }

    # 做多/做空统计
    long_trades = [t for t in trade_history if t.direction == "long"]
    short_trades = [t for t in trade_history if t.direction == "short"]
    long_win_rate = len([t for t in long_trades if t.pnl > 0]) / len(long_trades) * 100 if long_trades else 0
    short_win_rate = len([t for t in short_trades if t.pnl > 0]) / len(short_trades) * 100 if short_trades else 0
    long_pnl = sum(t.pnl for t in long_trades)
    short_pnl = sum(t.pnl for t in short_trades)

    # 分年做多/做空收益
    annual_long_short = {}
    for year in ["2023", "2024", "2025"]:
        y = int(year)
        lt = [t for t in long_trades if pd.Timestamp(t.exit_time, unit="ms").year == y]
        st = [t for t in short_trades if pd.Timestamp(t.exit_time, unit="ms").year == y]
        annual_long_short[year] = {
            "long_pnl": sum(t.pnl for t in lt),
            "short_pnl": sum(t.pnl for t in st),
            "long_n": len(lt),
            "short_n": len(st),
        }

    # 分模式统计
    trend_trades = [t for t in trade_history if "trend" in str(t.mode)]
    range_trades = [t for t in trade_history if "range" in str(t.mode)]
    trend_win_rate = len([t for t in trend_trades if t.pnl > 0]) / len(trend_trades) * 100 if trend_trades else 0
    range_win_rate = len([t for t in range_trades if t.pnl > 0]) / len(range_trades) * 100 if range_trades else 0
    trend_avg = np.mean([t.pnl for t in trend_trades]) if trend_trades else 0
    range_avg = np.mean([t.pnl for t in range_trades]) if range_trades else 0

    # 月度收益：用权益曲线变化（含手续费、资金费率、止损亏损）
    monthly = {}
    if equity_curve:
        eq_ts = np.array([e[0] for e in equity_curve])
        eq_val = np.array([e[1] for e in equity_curve])
        for ym in pd.period_range(
            pd.Timestamp(equity_curve[0][0], unit="ms").to_period("M"),
            pd.Timestamp(equity_curve[-1][0], unit="ms").to_period("M"),
            freq="M",
        ):
            ym_str = str(ym)
            mask = np.array(
                [pd.Timestamp(ts, unit="ms").strftime("%Y-%m") == ym_str for ts in eq_ts]
            )
            if np.any(mask):
                first_idx = np.where(mask)[0][0]
                last_idx = np.where(mask)[0][-1]
                monthly[ym_str] = eq_val[last_idx] - eq_val[first_idx]
    monthly_returns = list(monthly.items())
    best_month = max(monthly_returns, key=lambda x: x[1]) if monthly_returns else ("", 0)
    worst_month = min(monthly_returns, key=lambda x: x[1]) if monthly_returns else ("", 0)

    annual_lines = []
    for year, st in annual_stats.items():
        als = annual_long_short.get(year, {})
        lpnl = als.get("long_pnl", 0)
        spnl = als.get("short_pnl", 0)
        annual_lines.append(
            f"  {year}: 收益率 {st['return_pct']:.2f}%, 最大回撤 {st['max_dd_pct']:.2f}%, "
            f"交易 {st['n_trades']} 次 (多{als.get('long_n',0)}/空{als.get('short_n',0)}, 多收益{lpnl:.0f}/空收益{spnl:.0f})"
        )

    lines = [
        "=" * 50,
        "BTC 交易策略回测报告",
        "=" * 50,
        f"初始资金: {initial_balance:,.2f} USDT",
        f"最终资金: {final_balance:,.2f} USDT",
        f"总收益率: {total_return_pct:.2f}%",
        f"年化收益率: {annual_return_pct:.2f}%",
        f"最大回撤: {max_dd_pct:.2f}%",
        f"最大回撤持续时间: {max_dd_days:.0f} 天",
        f"夏普比率: {sharpe:.2f}",
        f"卡尔马比率: {calmar:.2f}",
        "-" * 50,
        f"总交易次数: {n_trades}",
        f"胜率: {win_rate:.1f}%",
        f"平均盈利: {avg_win:.2f} USDT",
        f"平均亏损: {avg_loss:.2f} USDT",
        f"盈亏比: {profit_factor:.2f}",
        f"最大连续盈利: {max_consec_win} 次",
        f"最大连续亏损: {max_consec_loss} 次",
        f"平均持仓时间: {avg_hold_days:.1f} 天",
        f"平均金字塔层数: {avg_pyramid_layers:.2f}",
        f"达到3层以上交易次数: {trades_3plus_layers}",
        f"加仓时评分>=5次数: {pyramid_score_5_plus}",
        "-" * 50,
        f"做多: {len(long_trades)} 次, 胜率 {long_win_rate:.1f}%, 总收益 {long_pnl:.2f}",
        f"做空: {len(short_trades)} 次, 胜率 {short_win_rate:.1f}%, 总收益 {short_pnl:.2f}",
        "-" * 50,
        f"趋势市交易: {len(trend_trades)} 次, 胜率 {trend_win_rate:.1f}%, 平均收益 {trend_avg:.2f}",
        f"震荡市交易: {len(range_trades)} 次, 胜率 {range_win_rate:.1f}%, 平均收益 {range_avg:.2f}",
        "-" * 50,
        "分年统计:",
        *annual_lines,
        "-" * 50,
        f"最好月份: {best_month[0]} ({best_month[1]:.2f} USDT)",
        f"最差月份: {worst_month[0]} ({worst_month[1]:.2f} USDT)",
        "=" * 50,
    ]
    return "\n".join(lines)


def _save_fig_ensure_updated(fig, filepath: str, dpi: int = 120):
    """保存图片并确保目标文件被覆盖、修改时间更新（先写临时文件再替换）。"""
    out_dir = os.path.dirname(filepath)
    fname = os.path.basename(filepath)
    tmp_path = os.path.join(out_dir, ".tmp_" + fname)
    try:
        fig.savefig(tmp_path, dpi=dpi)
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        fig.savefig(filepath, dpi=dpi)
    finally:
        plt.close(fig)


def _plot_trades_panel(
    btc_price: pd.DataFrame,
    trade_history: List,
    output_path: str,
    title_suffix: str = "",
):
    """画一张 BTC 价格 + 交易标记 图，可以按时间子区间重复调用。"""
    if btc_price is None or len(btc_price) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    btc_ts = btc_price["timestamp"].values
    btc_dates = pd.to_datetime(btc_ts, unit="ms")
    ax.plot(btc_dates, btc_price["close"].values, "gray", alpha=0.7, linewidth=1)

    # 统计用计数器
    first_count = 0        # 首仓笔数
    add_count = 0          # 加仓笔数
    sl_count = 0           # 止损笔数（软+硬）
    tp_count = 0           # 非止损平仓笔数（多为止盈）
    stop_pcts = []         # 单笔止损幅度%
    tp_pcts = []           # 单笔止盈幅度%
    tp_pnl_sum = 0.0       # 止盈/普通平仓累计收益
    sl_pnl_sum = 0.0       # 止损累计亏损
    total_size_pct = 0.0   # 累计开仓仓位百分比（首仓+加仓，按 size_pct 相加）

    # 开仓 & 加仓
    for t in trade_history:
        ed_entry = pd.Timestamp(t.entry_time, unit="ms")
        # 只画在当前子区间内的
        if ed_entry < btc_dates[0] or ed_entry > btc_dates[-1]:
            continue
        ep_entry = t.entry_price
        layer = getattr(t, "pyramid_layer", 1)
        size_pct = getattr(t, "size_pct", 0.0) or 0.0
        total_size_pct += size_pct
        if t.direction == "long":
            color = "tab:blue" if layer == 1 else "lightskyblue"
            marker = "^"
        else:
            color = "tab:orange" if layer == 1 else "moccasin"
            marker = "v"
        if layer == 1:
            first_count += 1
        else:
            add_count += 1
        ax.scatter(ed_entry, ep_entry, c=color, marker=marker, s=40 if layer == 1 else 30, alpha=0.9, zorder=5)
        # 在开仓/加仓标记下方标注仓位比例（例如 20%）
        size_pct = getattr(t, "size_pct", 0.0) * 100
        if size_pct:
            ax.annotate(
                f"{size_pct:.0f}%",
                xy=(ed_entry, ep_entry),
                xytext=(0, -10),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                va="top",
                color="black",
                alpha=0.8,
            )

    # 平仓点（止盈/止损）
    for t in trade_history:
        ed_exit = pd.Timestamp(t.exit_time, unit="ms")
        if ed_exit < btc_dates[0] or ed_exit > btc_dates[-1]:
            continue
        ep_exit = t.exit_price
        reason = getattr(t, "reason", "")
        ep_entry = t.entry_price
        # 按方向计算该笔价格变动百分比
        if t.direction == "long":
            move_pct = (ep_exit - ep_entry) / ep_entry * 100 if ep_entry else 0
        else:
            move_pct = (ep_entry - ep_exit) / ep_entry * 100 if ep_entry else 0

        if reason == "hard_stop":
            color = "red"
            marker = "x"
            size = 60
            sl_count += 1
            stop_pcts.append(abs(move_pct))
        elif reason == "soft_stop":
            color = "darkorange"
            marker = "x"
            size = 60
            sl_count += 1
            stop_pcts.append(abs(move_pct))
        else:
            color = "limegreen" if t.pnl > 0 else "dimgray"
            marker = "o"
            size = 40
            tp_count += 1
            # 只统计真正“向有利方向”的平仓幅度
            if move_pct > 0:
                tp_pcts.append(move_pct)
            tp_pnl_sum += t.pnl
        ax.scatter(ed_exit, ep_exit, c=color, marker=marker, s=size, alpha=0.9, zorder=6)
        # 在平仓标记下方标注盈亏百分比，例如 +5.3% / -2.1%
        label_pct = f"{move_pct:+.1f}%"
        ax.annotate(
            label_pct,
            xy=(ed_exit, ep_exit),
            xytext=(0, -10),
            textcoords="offset points",
            fontsize=7,
            ha="center",
            va="top",
            color="black",
            alpha=0.8,
        )

    title = "BTC Price, Entries, Pyramids & Exits"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    ax.set_ylabel("Price (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    # 在图内角落写上比例信息：首仓/加仓占比、止盈/止损占比、平均止盈/止损幅度
    total_entries = first_count + add_count
    total_exits = sl_count + tp_count
    first_ratio = first_count / total_entries * 100 if total_entries else 0
    add_ratio = add_count / total_entries * 100 if total_entries else 0
    sl_ratio = sl_count / total_exits * 100 if total_exits else 0
    tp_ratio = tp_count / total_exits * 100 if total_exits else 0
    avg_stop = np.mean(stop_pcts) if stop_pcts else 0
    avg_tp = np.mean(tp_pcts) if tp_pcts else 0
    stats_lines = [
        f"首仓: {first_count} ({first_ratio:.0f}%)  加仓: {add_count} ({add_ratio:.0f}%)  累计开仓: {total_size_pct*100:.0f}%",
        f"止盈: {tp_count} ({tp_ratio:.0f}%)  止损: {sl_count} ({sl_ratio:.0f}%)",
        f"平均止盈: {avg_tp:.1f}%  平均止损: {avg_stop:.1f}%",
        f"止盈PnL: {tp_pnl_sum:.0f}  止损PnL: {sl_pnl_sum:.0f}  净PnL: {(tp_pnl_sum+sl_pnl_sum):.0f}",
    ]
    ax.text(
        0.01,
        0.99,
        "\n".join(stats_lines),
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    fig.tight_layout()
    _save_fig_ensure_updated(fig, output_path, 120)


def plot_results(
    equity_curve: List[Tuple[int, float]],
    trade_history: List,
    btc_price: Optional[pd.DataFrame] = None,
    output_dir: str = "results",
):
    """
    生成 4 张图：资金曲线、BTC 价格+交易标记、月度收益、回撤曲线
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    ts = np.array([e[0] for e in equity_curve])
    eq = np.array([e[1] for e in equity_curve])
    dates = pd.to_datetime(ts, unit="ms")

    # 图1: 资金曲线
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, eq, "b-", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_ylabel("Equity (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig_ensure_updated(fig, os.path.join(output_dir, "equity_curve.png"), 120)

    # 图2: BTC 价格 + 交易标记（全时期 + 按年度拆分）
    if btc_price is not None and len(btc_price) > 0:
        # 全部区间一张大图
        _plot_trades_panel(
            btc_price,
            trade_history,
            os.path.join(output_dir, "trades.png"),
            title_suffix="All",
        )
        # 按年份拆分子图，便于放大看细节
        years = sorted(pd.to_datetime(btc_price["timestamp"], unit="ms").dt.year.unique())
        for y in years:
            mask_price = pd.to_datetime(btc_price["timestamp"], unit="ms").dt.year == y
            sub_price = btc_price.loc[mask_price].reset_index(drop=True)
            if len(sub_price) == 0:
                continue
            trades_y = [t for t in trade_history if pd.Timestamp(t.exit_time, unit="ms").year == y]
            if not trades_y:
                continue
            out_path = os.path.join(output_dir, f"trades_{y}.png")
            _plot_trades_panel(sub_price, trades_y, out_path, title_suffix=str(y))

    # 图3: 月度收益（权益曲线变化）
    monthly = {}
    if equity_curve:
        eq_ts = np.array([e[0] for e in equity_curve])
        eq_val = np.array([e[1] for e in equity_curve])
        for ym in pd.period_range(
            pd.Timestamp(equity_curve[0][0], unit="ms").to_period("M"),
            pd.Timestamp(equity_curve[-1][0], unit="ms").to_period("M"),
            freq="M",
        ):
            ym_str = str(ym)
            mask = np.array(
                [pd.Timestamp(ts, unit="ms").strftime("%Y-%m") == ym_str for ts in eq_ts]
            )
            if np.any(mask):
                idx = np.where(mask)[0]
                monthly[ym_str] = eq_val[idx[-1]] - eq_val[idx[0]]
    if monthly:
        months = sorted(monthly.keys())
        values = [monthly[m] for m in months]
        colors = ["green" if v >= 0 else "red" for v in values]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(months, values, color=colors)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title("Monthly Returns")
        ax.set_ylabel("P&L (USDT)")
        plt.xticks(rotation=45)
        fig.tight_layout()
        _save_fig_ensure_updated(fig, os.path.join(output_dir, "monthly_returns.png"), 120)

    # 图4: 回撤曲线
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / np.where(peak > 0, peak, 1) * 100
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(dates, 0, -dd, color="red", alpha=0.3)
    ax.plot(dates, -dd, "darkred", linewidth=1)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig_ensure_updated(fig, os.path.join(output_dir, "drawdown.png"), 120)


def plot_analysis_charts(trade_history: List, output_dir: str = "results"):
    """
    辅助分析图表：单笔 PnL 随时间、R 倍数分布、分模式/分方向表现、按金字塔层数表现。
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if not trade_history:
        return

    # ----- 图5: 单笔 PnL 随时间分布 -----
    exit_times = [t.exit_time for t in trade_history]
    pnls = [t.pnl for t in trade_history]
    exit_dates = pd.to_datetime(exit_times, unit="ms")
    colors = ["green" if p >= 0 else "red" for p in pnls]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(exit_dates, pnls, c=colors, alpha=0.7, s=25)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("单笔 PnL 随时间分布")
    ax.set_ylabel("PnL (USDT)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig_ensure_updated(fig, os.path.join(output_dir, "pnl_over_time.png"), 120)

    # ----- 图6: R 倍数分布（直方图）-----
    r_list = []
    for t in trade_history:
        ep = t.entry_price
        hard = getattr(t, "hard_stop", None)
        margin = getattr(t, "margin", 0)
        lev = getattr(t, "leverage", 1)
        if ep and ep > 0 and hard is not None and margin and margin > 0:
            stop_dist_pct = abs(ep - hard) / ep * 100
            if stop_dist_pct > 0:
                one_r = margin * lev * stop_dist_pct / 100
                if one_r > 0:
                    r_list.append(t.pnl / one_r)
    if r_list:
        r_arr = np.array(r_list)
        r_min, r_max = r_arr.min(), r_arr.max()
        bins = np.linspace(max(-5, r_min - 0.5), min(10, r_max + 0.5), 30)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(r_arr, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_title("单笔收益 R 倍数分布（1R = 止损距离对应的亏损）")
        ax.set_xlabel("R 倍数")
        ax.set_ylabel("笔数")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save_fig_ensure_updated(fig, os.path.join(output_dir, "r_multiple_hist.png"), 120)

    # ----- 图7: 分模式 / 分方向表现（柱状图）-----
    by_mode = {}
    by_dir = {"long": [], "short": []}
    for t in trade_history:
        mode = getattr(t, "mode", "unknown") or "unknown"
        if mode not in by_mode:
            by_mode[mode] = []
        by_mode[mode].append(t.pnl)
        by_dir[t.direction].append(t.pnl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 左：按模式平均 PnL
    if by_mode:
        modes = sorted(by_mode.keys())
        avg_pnl = [np.mean(by_mode[m]) for m in modes]
        colors_m = ["green" if v >= 0 else "red" for v in avg_pnl]
        axes[0].bar(modes, avg_pnl, color=colors_m)
        axes[0].axhline(0, color="black", linewidth=0.5)
        axes[0].set_title("按市场模式 · 平均单笔 PnL (USDT)")
        axes[0].set_ylabel("平均 PnL")
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=25, ha="right")
    # 右：按方向平均 PnL
    dirs = ["long", "short"]
    avg_long = np.mean(by_dir["long"]) if by_dir["long"] else 0
    avg_short = np.mean(by_dir["short"]) if by_dir["short"] else 0
    vals = [avg_long, avg_short]
    colors_d = ["green" if v >= 0 else "red" for v in vals]
    axes[1].bar(["做多", "做空"], vals, color=colors_d)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_title("按方向 · 平均单笔 PnL (USDT)")
    axes[1].set_ylabel("平均 PnL")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig_ensure_updated(fig, os.path.join(output_dir, "mode_direction_perf.png"), 120)

    # ----- 图8: 按金字塔层数表现 -----
    by_layer = {}
    for t in trade_history:
        layer = getattr(t, "pyramid_layer", 1)
        if layer not in by_layer:
            by_layer[layer] = []
        by_layer[layer].append(t.pnl)
    if by_layer:
        layers = sorted(by_layer.keys())
        avg_by_layer = [np.mean(by_layer[l]) for l in layers]
        count_by_layer = [len(by_layer[l]) for l in layers]
        colors_l = ["green" if v >= 0 else "red" for v in avg_by_layer]
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(layers))
        w = 0.35
        ax.bar(x - w / 2, avg_by_layer, w, label="平均 PnL (USDT)", color=colors_l)
        ax2 = ax.twinx()
        ax2.bar(x + w / 2, count_by_layer, w, label="笔数", color="gray", alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"第{n}层" for n in layers])
        ax.set_title("按金字塔层数 · 平均 PnL 与交易笔数")
        ax.set_ylabel("平均 PnL (USDT)")
        ax2.set_ylabel("笔数")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save_fig_ensure_updated(fig, os.path.join(output_dir, "pyramid_layer_perf.png"), 120)


def save_trade_log(trade_history: List, path: str = "results/trade_log.csv"):
    """保存交易记录 CSV"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for t in trade_history:
        rows.append({
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "size_pct": t.size_pct,
            "leverage": t.leverage,
            "entry_time": pd.Timestamp(t.entry_time, unit="ms"),
            "exit_time": pd.Timestamp(t.exit_time, unit="ms"),
            "pnl": t.pnl,
            "reason": t.reason,
            "market_mode": t.mode,
            "wave_type": getattr(t, "wave_type", ""),
            "pyramid_layer": getattr(t, "pyramid_layer", 1),
            "pyramid_id": getattr(t, "pyramid_id", 0),
            "pyramid_trigger": getattr(t, "pyramid_trigger", ""),
            "late_entry": getattr(t, "late_entry", False),
            "late_entry_pct": getattr(t, "late_entry_pct", 0.0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
