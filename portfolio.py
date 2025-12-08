import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# 尝试导入 alpha_engine
try:
    from alpha_engine import AlphaContext 
except ImportError:
    print("错误: 未找到 alpha_engine.py。请确保它在同一目录下。")
    sys.exit(1)

# ================= 配置部分 =================
DATA_PATH = "./data/market_data.csv"
BEST_ALPHA_FILE = "./reports/final_champion.json" # 锦标赛冠军文件
INITIAL_CAPITAL = 1000000 # 100万初始资金
TOP_K = 5 # 每天持有前5只

def run_portfolio_backtest():
    # 1. 加载最佳因子信息
    if not os.path.exists(BEST_ALPHA_FILE):
        print(f"错误: 未找到最佳因子文件 {BEST_ALPHA_FILE}。")
        print("请先运行 5_tournament.py 生成冠军因子。")
        return
        
    with open(BEST_ALPHA_FILE, 'r', encoding='utf-8') as f:
        alpha_info = json.load(f)
        
    formula = alpha_info['formula']
    name = alpha_info['name']
    print("=" * 60)
    print(f"🚀 开始回测最佳策略: {name}")
    print(f"📜 公式: {formula}")
    print("=" * 60)
    
    # 2. 准备数据
    if not os.path.exists(DATA_PATH):
        print(f"错误: 数据文件 {DATA_PATH} 不存在。")
        return

    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    # 排序非常重要
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # 3. 计算因子值 (复用 AlphaContext)
    print("正在计算因子值...", end="")
    ctx = AlphaContext(df)
    env = {
        'CLOSE': ctx.CLOSE(), 'OPEN': ctx.OPEN(), 'VOLUME': ctx.VOLUME(), 
        'HIGH': ctx.HIGH(), 'LOW': ctx.LOW(),
        'DELAY': ctx.DELAY, 'MA': ctx.MA, 'STD': ctx.STD,
        'TS_MAX': ctx.TS_MAX, 'TS_MIN': ctx.TS_MIN,
        'CORR': ctx.CORR, 'RANK': ctx.RANK
    }
    
    try:
        df['factor'] = eval(formula, {}, env)
        print(" [完成]")
    except Exception as e:
        print(f"\n[失败] 因子计算出错: {e}")
        return

    # 4. 每日 Top-K 选股回测
    # 清除无效因子值的行
    df = df.dropna(subset=['factor'])
    
    if df.empty:
        print("错误: 计算后的因子数据为空，无法回测。")
        return

    # 使用 numpy.sort 确保排序
    dates = np.sort(df['date'].unique())
    
    capital = INITIAL_CAPITAL
    capital_curve = []
    
    # === 【修复点】: 转换日期格式 ===
    start_date_str = pd.Timestamp(dates[0]).strftime('%Y-%m-%d')
    end_date_str = pd.Timestamp(dates[-1]).strftime('%Y-%m-%d')
    
    print(f"开始模拟交易，时间跨度: {start_date_str} 至 {end_date_str}")
    print(f"交易天数: {len(dates)}")
    
    for date in dates:
        # 转换回 timestamp 以便过滤
        daily_data = df[df['date'] == date]
        
        # 选股逻辑：买入因子值最大的 Top K
        if len(daily_data) < TOP_K:
            selected = daily_data
        else:
            selected = daily_data.nlargest(TOP_K, 'factor')
            
        # 简单回测逻辑：
        # 假设：开盘买入，收盘卖出 (日内交易)
        # 收益 = (Close - Open) / Open
        if not selected.empty:
            daily_ret = (selected['close'] - selected['open']) / selected['open']
            # 扣除简单的交易成本 (例如万分之三)
            avg_ret = daily_ret.mean() - 0.0003
            
            capital = capital * (1 + avg_ret)
        
        capital_curve.append({'date': date, 'equity': capital})

    # 5. 结果分析与可视化
    result_df = pd.DataFrame(capital_curve)
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df.set_index('date', inplace=True)
    
    total_ret = ((capital / INITIAL_CAPITAL) - 1) * 100
    
    print("\n" + "=" * 60)
    print(f"📊 回测结果报告: {name}")
    print("-" * 60)
    print(f"初始资金: {INITIAL_CAPITAL:,.2f}")
    print(f"最终资金: {capital:,.2f}")
    print(f"总收益率: {total_ret:.2f}%")
    print("=" * 60)
    
    # 绘图
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(result_df.index, result_df['equity'], label=f'Strategy: {name}')
        plt.title(f'Strategy Equity Curve: {name} (Formula: {formula})')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        
        # 保存图片而不是仅仅显示，防止在某些无头环境下报错
        output_img = f"./reports/equity_curve_{name}.png"
        
        # 确保目录存在
        if not os.path.exists("./reports"):
             os.makedirs("./reports")
             
        plt.savefig(output_img)
        print(f"资金曲线图已保存至: {output_img}")
        
        plt.show()
    except Exception as e:
        print(f"绘图失败 (非致命错误): {e}")

if __name__ == "__main__":
    run_portfolio_backtest()