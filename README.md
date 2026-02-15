# BTC 多周期量化回测系统

基于《多周期量化交易策略框架 v7.0》的 BTC 回测引擎，支持 Vegas 状态机、评分制入场、金字塔建仓与破位平仓等规则。

## 环境要求

- Python 3.9+
- 依赖见 `requirements.txt`

## 安装（Windows / macOS 通用）

```bash
cd btc_backtest
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## 数据准备

首次运行或在新电脑上需要先拉取 K 线数据（需联网）：

```bash
python main.py --fetch
```

会将 4H/12H/日线/周线数据写入 `data/` 目录。若 `data/` 下已有 CSV，可直接回测。

## 运行回测

```bash
# 默认区间回测
python main.py

# 指定日期
python main.py --start 2024-01-01 --end 2024-12-31

# 先拉数据再回测
python main.py --fetch
```

结果输出到 `results/`（报告、交易记录等）。

## 在 Mac 上使用

1. 克隆仓库后进入目录，按上面「安装」步骤创建虚拟环境并安装依赖。
2. 执行 `python main.py --fetch` 获取数据（与 Windows 相同）。
3. 运行 `python main.py` 或带 `--start`/`--end` 的回测命令。

代码使用相对路径（`data/`、`results/`），Windows 与 macOS 均可直接运行，无需改配置。
