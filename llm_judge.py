from openai import OpenAI
import os
import json
from alpha_engine import analyze_factor 

# ==========================================
# 配置部分：切换为 Qwen (通义千问)
# ==========================================
# 1. 替换为你的阿里云 DashScope API Key (通常以 sk- 开头)
#    获取地址: https://bailian.console.aliyun.com/
DASHSCOPE_API_KEY = "sk-d807fcc0e09e40b9a3d6f736aad39c15"

# 2. 初始化 Client，核心是修改 base_url
client = OpenAI(
    api_key=DASHSCOPE_API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云兼容 OpenAI 的接口地址
)

def llm_judge(factor_A_report, factor_B_report, market_context):
    
    prompt = f"""
    【角色】
    你是一位资深的量化投资组合经理。你的任务是根据当前的市场环境，对比两个 Alpha 策略的表现，并选择下个季度更适合的一个。

    【当前市场环境】
    {market_context}

    【选手 A: {factor_A_report['name']}】
    - 公式: {factor_A_report['formula']}
    - IC均值 (预测能力): {factor_A_report['IC_Mean']} (正数表示正相关，负数表示负相关)
    - 年化收益: {factor_A_report['Annual_Return']}
    - 夏普比率: {factor_A_report['Sharpe']}

    【选手 B: {factor_B_report['name']}】
    - 公式: {factor_B_report['formula']}
    - IC均值 (预测能力): {factor_B_report['IC_Mean']}
    - 年化收益: {factor_B_report['Annual_Return']}
    - 夏普比率: {factor_B_report['Sharpe']}

    【任务】
    请分析哪个因子在当前市场环境下表现更好？
    1. 简单分析两个因子的逻辑（是动量还是反转？）。
    2. 结合市场环境说明为什么选它。
    3. 最终给出结论：胜者是 A 还是 B。
    """

    print(f"正在咨询 AI 分析师 (使用模型: qwen-plus)...")
    
    try:
        response = client.chat.completions.create(
            # 可选模型: 
            # - qwen-max (能力最强，接近 GPT-4)
            # - qwen-plus (性价比高，能力均衡)
            # - qwen-turbo (速度快，便宜)
            model="qwen-plus", 
            messages=[
                {"role": "system", "content": "你是一个专业的量化金融助手，擅长分析因子表现与市场风格的匹配度。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2 # 降低随机性，让分析更理性
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用 Qwen API 失败: {e}"

if __name__ == "__main__":
    # --- 模拟运行 ---

    # 1. 准备市场环境描述 (论文中的 Context)
    # 这里我们用中文描述，Qwen 理解起来会更精准
    market_status = """
    近期市场呈现震荡下行趋势，成交量持续萎缩。
    宏观层面，经济复苏不及预期，市场缺乏明确的主线题材。
    在这种存量博弈的特征下，高位股开始补跌，低估值板块防御属性凸显。
    """

    # 2. 计算两个因子的真实表现 (调用上一步写的引擎)
    print("正在计算因子指标...")
    # 因子A: 动量 (追涨杀跌) -> 在震荡下跌市中通常表现较差
    report_A = analyze_factor("Momentum_10D", "CLOSE - DELAY(CLOSE, 10)")
    
    # 因子B: 均值回归 (跌多了买) -> 在震荡市中可能表现较好
    report_B = analyze_factor("Reversion_5D", "MA(CLOSE, 5) - CLOSE")

    # 打印简报看看
    print(f"\n[因子A 简报]: {json.dumps(report_A, ensure_ascii=False)}")
    print(f"[因子B 简报]: {json.dumps(report_B, ensure_ascii=False)}\n")

    # 3. AI 裁决
    if isinstance(report_A, dict) and isinstance(report_B, dict):
        decision = llm_judge(report_A, report_B, market_status)
        
        print("="*40)
        print("🤖 通义千问(Qwen) 投资总监的决策报告")
        print("="*40)
        print(decision)
    else:
        print("因子计算出错，无法进行 PK。请检查数据是否下载完成。")