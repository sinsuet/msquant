import json
import os
import time
import pandas as pd
from openai import OpenAI
# 引入我们刚写的多模态模块
from multimodal_utils import analyze_market_multimodal

# ================= 配置 =================
DASHSCOPE_API_KEY = "sk-d807fcc0e09e40b9a3d6f736aad39c15" # 确保填入 Key
DATA_PATH = "./data/market_data.csv"
REPORT_PATH = "./reports/all_reports.json"

client = OpenAI(
    api_key=DASHSCOPE_API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def llm_judge(defender, challenger, market_context):
    """
    裁判 Agent：接收【动态】的市场描述
    """
    prompt = f"""
    【任务】
    你是一场 Alpha 因子挖掘锦标赛的裁判。
    
    【🚨 实时市场情报 (由视觉模型生成)】
    {market_context}

    请基于上述市场情报，对比以下两个策略因子，选出最适合当前环境的一个：

    【擂主 (Defender): {defender['name']}】
    - 逻辑: {defender.get('logic', '无')}
    - 公式: {defender['formula']}
    - IC: {defender['IC_Mean']}, Sharpe: {defender['Sharpe']}

    【挑战者 (Challenger): {challenger['name']}】
    - 逻辑: {challenger.get('logic', '无')}
    - 公式: {challenger['formula']}
    - IC: {challenger['IC_Mean']}, Sharpe: {challenger['Sharpe']}

    【输出 JSON】
    {{
        "analysis": "结合市场情报(如波动率/趋势)分析两者的适应性...",
        "winner": "A 或 B",
        "winner_name": "胜者名称",
        "reason": "核心理由"
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个专业的量化裁判，根据实时市场风格选股。只输出JSON。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"裁判出错: {e}")
        return {"winner": "A", "winner_name": defender['name'], "reason": "裁判掉线"}

def run_multimodal_tournament():
    # 1. 加载数据用于生成环境
    if not os.path.exists(DATA_PATH) or not os.path.exists(REPORT_PATH):
        print("数据缺失，请先运行 batch_processor.py")
        return
        
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # 设定一个“当下”的时间点用于感知市场
    # 在实际回测中，这应该是一个循环。这里为了演示，我们取数据集的最后一天，或者某个特定切片
    current_date = "2023-12-01" 
    print(f"🌍 正在通过多模态 Agent 感知 {current_date} 的市场环境...")
    print("(正在绘制K线图并阅读市场简报...)")
    
    # === 核心升级：动态生成 Context ===
    market_context = analyze_market_multimodal(df, current_date)
    print("\n📝 [生成的市场情报]:")
    print("-" * 40)
    print(market_context)
    print("-" * 40)
    
    # 2. 加载因子
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        factors = json.load(f)
    
    if len(factors) < 2: return

    # 3. 开始比赛
    champion = factors[0]
    print(f"\n🏆 锦标赛开始 (基于上述市场情报)")
    
    for i, challenger in enumerate(factors[1:]):
        print(f"\n>> Round {i+1}: {champion['name']} vs {challenger['name']}")
        
        result = llm_judge(champion, challenger, market_context)
        
        winner = result.get('winner', 'A')
        reason = result.get('reason', '无')
        print(f"   裁判观点: {result.get('analysis')[:50]}...")
        print(f"   👉 胜者: {result.get('winner_name')} ({reason})")
        
        if winner == 'B' or result.get('winner_name') == challenger['name']:
            champion = challenger

    print(f"\n🎉 最终冠军: {champion['name']}")
    # 保存结果
    with open("./reports/final_champion_multimodal.json", "w", encoding="utf-8") as f:
        json.dump(champion, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    if "你的API_KEY" in DASHSCOPE_API_KEY:
        print("请在 multimodal_utils.py 和 5_tournament_pro.py 中填入 API Key")
    else:
        run_multimodal_tournament()