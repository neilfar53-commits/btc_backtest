# -*- coding: utf-8 -*-
"""
BTC 交易策略回测系统 — 配置文件
"""

# ====== 数据获取 ======
OKX_BASE_URL = "https://www.okx.com"
OKX_HISTORY_CANDLES_PATH = "/api/v5/market/history-candles"
INST_ID = "BTC-USDT-SWAP"
BAR_4H = "4H"
CANDLE_LIMIT = 100

# 回测区间
BACKTEST_START = "2023-01-01"
BACKTEST_END = "2025-12-31"

# 数据存储路径
DATA_DIR = "data"
DATA_4H = "data/btc_4h.csv"
DATA_12H = "data/btc_12h.csv"
DATA_DAILY = "data/btc_daily.csv"
DATA_WEEKLY = "data/btc_weekly.csv"

# ====== 资金档位（v7.0 手册 9.2）======
# S<1万U max_lev 15x 加权8x, A 12x/6x, B 8x/4x, C 5x/3x
TIERS = {
    "S": {"min": 0, "max": 10000, "max_risk": 0.08, "max_lev": 15},
    "A": {"min": 10000, "max": 50000, "max_risk": 0.06, "max_lev": 12},
    "B": {"min": 50000, "max": 200000, "max_risk": 0.05, "max_lev": 8},
    "C": {"min": 200000, "max": float("inf"), "max_risk": 0.03, "max_lev": 5},
}

# ====== 动态杠杆表（v7.0 手册 9.4）======
# 趋势市：≤20% 4x, 21-50% 5x, 51-70% 6x, 71-100% 8x（S档正常波动）
# 震荡市：固定 4x。最终杠杆 = min(档位上限, 基准×vol_mult×state_mult×lossMult)
LEVERAGE_TABLE = {
    "S": {
        "trend": {"layer1": 4, "layer2": 5, "layer3": 6, "layer4": 8},
        "range": {"fixed": 4},
    },
    "A": {
        "trend": {"layer1": 3, "layer2": 4, "layer3": 5, "layer4": 6},
        "range": {"fixed": 3},
    },
    "B": {
        "trend": {"layer1": 2, "layer2": 3, "layer3": 3, "layer4": 5},
        "range": {"fixed": 3},
    },
    "C": {
        "trend": {"layer1": 1, "layer2": 2, "layer3": 2, "layer4": 3},
        "range": {"fixed": 2},
    },
}

# ====== 波动率调节 ======
VOL_MULT = {
    "low": 1.2,  # atr_pct < 3%
    "normal": 1.0,  # 3% <= atr_pct <= 5%
    "high": 0.7,  # 5% < atr_pct <= 8%
    "extreme": 0.4,  # atr_pct > 8%
}

# ====== 金字塔仓位 ======
PYRAMID_SIZE = {
    "layer1": 0.20,  # 20%
    "layer2": 0.30,  # +30% = 50%
    "layer3": 0.20,  # +20% = 70%
    "layer4": 0.30,  # +30% = 100% (最大)
}

# ====== 评分阈值 ======
SCORE_THRESHOLD = {
    "trend": {"layer1": 3, "layer2": 5, "layer3": 8, "layer4": 10},
    "range": {"entry": 3},
}

# ====== 交易成本 ======
MAKER_FEE = 0.0002  # 0.02% 限价单
TAKER_FEE = 0.0005  # 0.05% 市价单
AVG_FUNDING_RATE = 0.0001  # 每8小时 0.01%
SLIPPAGE = 0.0005  # 0.05% 滑点

# ====== 初始资金 ======
INITIAL_BALANCE = 4625

# ====== Pivot 权重（用于聚合 S/R） ======
# 当月100% / 前1月80% / 前2月60% / 前3月40% / 前4月25% / 前5月15%
PIVOT_WEIGHTS = [1.0, 0.8, 0.6, 0.4, 0.25, 0.15]