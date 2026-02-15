# -*- coding: utf-8 -*-
"""
BTC 交易策略回测系统 — 入口文件
支持原引擎与《回测开发指令》手册版引擎（--handbook）。
"""

import argparse
import os

from config import BACKTEST_END, BACKTEST_START, DATA_DIR, INITIAL_BALANCE


def main():
    parser = argparse.ArgumentParser(description="BTC 交易策略回测系统")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="先获取数据（默认使用已有 data/ 数据）",
    )
    parser.add_argument(
        "--handbook",
        action="store_true",
        help="使用《回测开发指令》手册版引擎（Phase 1：趋势+收紧，首仓20%%，无金字塔）",
    )
    parser.add_argument(
        "--start",
        default=BACKTEST_START,
        help=f"回测开始日期 (默认: {BACKTEST_START})",
    )
    parser.add_argument(
        "--end",
        default=BACKTEST_END,
        help=f"回测结束日期 (默认: {BACKTEST_END})",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="输出目录 (默认: results)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不生成图表",
    )
    args = parser.parse_args()

    if args.fetch:
        print("正在获取数据...")
        from data_fetcher import run_data_fetcher
        run_data_fetcher(start_date=args.start, end_date=args.end)

    if args.handbook:
        from backtest_handbook import run_backtest_handbook
        from reporter import generate_report as report_gen, plot_results as report_plot
        print("运行回测（手册版 Phase 1）...")
        portfolio, df, extra = run_backtest_handbook(
            start_date=None, end_date=args.end, initial_balance=INITIAL_BALANCE, data_dir=DATA_DIR,
        )
        trades = portfolio["trade_history"]
        equity_curve = portfolio["equity_history"]
        trade_dicts = [
            {"pnl": t.pnl, "direction": t.direction, "entry_price": t.entry_price, "exit_price": t.exit_price,
             "reason": t.reason, "mode": t.mode}
            for t in trades
        ]
        stats = report_gen(trade_dicts, equity_curve, INITIAL_BALANCE, None)
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "report_handbook.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join([f"{k}: {v}" for k, v in stats.items()]))
        print(f"报告已保存: {report_path}")
        if not args.no_plot:
            report_plot(trade_dicts, equity_curve, df, output_dir)
        return

    from analysis import generate_report, plot_results, plot_analysis_charts, save_trade_log
    from backtest_engine import run_backtest

    print("运行回测...")
    pm, df, extra = run_backtest(
        initial_balance=INITIAL_BALANCE,
        start_date=args.start,
        end_date=args.end,
    )

    equity_curve = pm.equity_history
    trade_history = pm.trade_history

    monthly_mode = extra.get("monthly_mode", {})
    pyramid_score_5_plus = extra.get("pyramid_score_5_plus", 0)
    print("\n" + generate_report(
        trade_history, equity_curve,
        monthly_mode=monthly_mode,
        pyramid_score_5_plus=pyramid_score_5_plus,
    ))

    # 统一使用绝对路径，避免“运行目录”和“资源管理器看的目录”不一致导致图表看似未更新
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(generate_report(
            trade_history, equity_curve,
            monthly_mode=monthly_mode,
            pyramid_score_5_plus=pyramid_score_5_plus,
        ))
    print(f"\n报告已保存: {report_path}")

    # 月度模式判定
    mode_path = os.path.join(output_dir, "monthly_mode.txt")
    with open(mode_path, "w", encoding="utf-8") as f:
        f.write("月度市场模式判定\n" + "=" * 40 + "\n")
        for ym in sorted(monthly_mode.keys()):
            f.write(f"{ym}: {monthly_mode[ym]}\n")
    print(f"月度模式已保存: {mode_path}")

    save_trade_log(trade_history, os.path.join(output_dir, "trade_log.csv"))

    if not args.no_plot:
        btc_price = df[["timestamp", "close"]].copy() if len(df) > 0 else None
        plot_results(equity_curve, trade_history, btc_price, output_dir)
        plot_analysis_charts(trade_history, output_dir)
        print(f"图表已保存到: {output_dir}")
        print(f"  主图: equity_curve, trades, monthly_returns, drawdown")
        print(f"  分析: pnl_over_time, r_multiple_hist, mode_direction_perf, pyramid_layer_perf")
    else:
        print("已跳过图表生成 (--no-plot)。若要更新 drawdown/equity/monthly_return/trades 图，请去掉 --no-plot 后重新运行。")


if __name__ == "__main__":
    main()
