import json
import os
import sys

# 尝试导入计算引擎
try:
    from alpha_engine import analyze_factor
except ImportError:
    print("错误: 未找到 alpha_engine.py。请将 2_alpha_engine.py 重命名为 alpha_engine.py")
    sys.exit(1)

# ==========================================
# 因子池定义 (基于 101 Alphas & Causal Factor Investing)
# ==========================================
SEED_ALPHAS = [
    # --- 1. 动量与趋势类 (Momentum & Trend) ---
    {
        "name": "Momentum_ROC_20",
        "formula": "CLOSE / DELAY(CLOSE, 20) - 1",
        "logic": "经典动量因子。基于价格惯性理论，投资者对信息的反应存在滞后，导致过去表现好的股票在短期内倾向于继续上涨。"
    },
    {
        "name": "Alpha_101_009_Proxy",
        "formula": "(CLOSE - DELAY(CLOSE, 5)) * (RANK(MA(CLOSE, 5)) - 0.5)",
        "logic": "带趋势确认的短期动量。参考 Alpha#9 结构，当价格处于高分位趋势时，短期动量的持续性更强。"
    },
    {
        "name": "Max_Retracement_Anchor",
        "formula": "CLOSE / TS_MAX(CLOSE, 20) - 1",
        "logic": "锚定效应。投资者往往以近期的最高价作为参考点。当前价格距离最高价越近，测试突破阻力的动能。"
    },
    {
        "name": "Path_Efficiency",
        "formula": "(CLOSE - DELAY(CLOSE, 10)) / (MA(HIGH - LOW, 10) * 10)",
        "logic": "趋势效率。衡量价格移动的纯粹度。如果价格以低波动率直线移动，意味着信息传递效率高，趋势更稳健。"
    },
    {
        "name": "Rank_Momentum_Cross",
        "formula": "RANK(CLOSE - DELAY(CLOSE, 10))",
        "logic": "截面动量。不看绝对涨幅，看该股票在全市场中的相对强弱。资金往往抱团流入相对表现最强的头部股票。"
    },

    # --- 2. 反转与过度反应类 (Reversion) ---
    {
        "name": "Bollinger_Reversion",
        "formula": "-1 * (CLOSE - MA(CLOSE, 20)) / STD(CLOSE, 20)",
        "logic": "布林带均值回归。价格偏离均线超过标准差（Z-Score过高）通常源于市场情绪的过度宣泄，预期会向均值回归。"
    },
    {
        "name": "Intraday_Reversion",
        "formula": "(OPEN - CLOSE) / (HIGH - LOW)",
        "logic": "日内反转。如果收盘价显著低于开盘价（实体阴线），可能代表日内恐慌盘过度释放，次日存在高开或反弹的流动性溢价机会。"
    },
    {
        "name": "Alpha_101_006_Like",
        "formula": "-1 * CORR(OPEN, VOLUME, 10)",
        "logic": "经典 Alpha#6 变体。开盘价与成交量的负相关性意味着“缩量上涨”或“放量下跌”，通常是知情交易者介入的信号。"
    },
    {
        "name": "Min_Rebound_Strength",
        "formula": "(CLOSE - TS_MIN(LOW, 20)) / TS_MIN(LOW, 20)",
        "logic": "底部反弹强度。基于支撑位逻辑，测试价格从近期最低点反弹的幅度。过弱的反弹可能意味着下跌中继，过强则确立反转。"
    },

    # --- 3. 波动率与风险类 (Volatility & Risk) ---
    {
        "name": "Low_Vol_Anomaly",
        "formula": "-1 * STD(CLOSE, 20)",
        "logic": "低波动率异象。实证表明，低波动率股票的长期风险调整后收益优于高波动率股票。"
    },
    {
        "name": "High_Low_Range_Vol",
        "formula": "-1 * MA((HIGH - LOW) / CLOSE, 10)",
        "logic": "日内波动惩罚。平均日内振幅过大的股票包含了高不确定性，投资者应要求更高的风险补偿。"
    },
    {
        "name": "Volume_Variance_Penalty",
        "formula": "-1 * STD(VOLUME, 20) / MA(VOLUME, 20)",
        "logic": "成交量稳定性。成交量忽大忽小的股票往往流动性不稳定，机构资金倾向于配置流动性可预测性高的资产。"
    },

    # --- 4. 量价关系类 (Price-Volume) ---
    {
        "name": "Volume_Price_Confirm",
        "formula": "CORR(CLOSE, VOLUME, 20)",
        "logic": "量价确认。价格趋势与成交量的正相关性（量价齐升）确认了趋势的真实性。"
    },
    {
        "name": "Volume_Shock_Reversal",
        "formula": "-1 * VOLUME / MA(VOLUME, 20)",
        "logic": "巨量反转。异常放量（Volume Shock）往往发生在趋势末端（诱多或恐慌盘涌出）。"
    },
    {
        "name": "Alpha_101_054_Simp",
        "formula": "-1 * (LOW - CLOSE) * (OPEN - LOW) / ((HIGH - LOW) * CLOSE)",
        "logic": "结构化量价因子。参考 Alpha#54，捕捉日内K线形态（如下影线长度）所隐含的买卖力量对比。"
    },
    {
        "name": "Liquidity_Premium",
        "formula": "-1 * MA(VOLUME * CLOSE, 20)",
        "logic": "非流动性溢价。交易金额较小的股票通常因流动性差而存在折价，长期持有应获得流动性补偿。"
    },

    # --- 5. 相对强弱与复杂组合 (Rank & Cross-Sectional) ---
    {
        "name": "Rank_Correlation_Reversion",
        "formula": "-1 * CORR(RANK(CLOSE), RANK(VOLUME), 10)",
        "logic": "排名相关性回归。当价格排名与成交量排名高度正相关时，往往意味着拥挤交易，后续可能出现反转。"
    },
    {
        "name": "Close_Location_Value_MA",
        "formula": "MA((CLOSE - LOW) / (HIGH - LOW), 10)",
        "logic": "平均CLV。持续收在日内高位表明多头力量在每日收盘阶段依然占据主导，是强势股特征。"
    },
    {
        "name": "Overnight_Gap",
        "formula": "(OPEN - DELAY(CLOSE, 1)) / DELAY(CLOSE, 1)",
        "logic": "隔夜信息反应。大幅高开可能代表利好过度反应或动量爆发。"
    },
    {
        "name": "Tail_Risk_Skew",
        "formula": "(TS_MAX(HIGH, 20) - CLOSE) / (CLOSE - TS_MIN(LOW, 20) + 0.001)",
        "logic": "偏度代理指标。如果价格更接近近期低点而非高点，表明卖压主导，尾部风险较高。"
    }
]

OUTPUT_DIR = "./reports"

def run_batch_analysis():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"开始批量计算 {len(SEED_ALPHAS)} 个种子因子...")
    print("-" * 60)
    
    valid_reports = []
    
    for i, alpha in enumerate(SEED_ALPHAS):
        print(f"[{i+1}/{len(SEED_ALPHAS)}] 计算 {alpha['name']} ... ", end="")
        
        # 调用引擎计算
        report = analyze_factor(alpha['name'], alpha['formula'])
        
        # 补充逻辑描述到报告中，供 LLM 参考
        report['logic'] = alpha['logic']
        
        # 检查是否出错
        if "error" in report:
            print(f"[失败] {report['error']}")
        else:
            # 只有计算成功的才进入锦标赛
            print(f"[成功] IC: {report['IC_Mean']}, 年化: {report['Annual_Return']}")
            valid_reports.append(report)
            
            # 保存单个详细报告 (可选)
            file_path = os.path.join(OUTPUT_DIR, f"{alpha['name']}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=4)

    # 保存总表
    summary_path = os.path.join(OUTPUT_DIR, "all_reports.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(valid_reports, f, ensure_ascii=False, indent=4)
        
    print("-" * 60)
    print(f"批量计算完成！有效因子: {len(valid_reports)}/{len(SEED_ALPHAS)}")
    print(f"锦标赛数据已保存至: {summary_path}")

if __name__ == "__main__":
    run_batch_analysis()