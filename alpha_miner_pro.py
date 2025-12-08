import json
import os
import time
import pandas as pd
from openai import OpenAI
from alpha_engine import analyze_factor
# 引入我们刚写好的多模态感知模块
from multimodal_utils import analyze_market_multimodal

# ================= 配置 =================
DASHSCOPE_API_KEY = "sk-d807fcc0e09e40b9a3d6f736aad39c15"
DATA_PATH = "./data/market_data.csv"
MINED_OUTPUT_FILE = "./reports/mined_alphas_pro.json"

client = OpenAI(
    api_key=DASHSCOPE_API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 算子文档
OPERATOR_DOCS = """
Available Operators:
- DELAY(x, n), MA(x, n), STD(x, n), TS_MAX(x, n), TS_MIN(x, n)
- RANK(x), CORR(x, y, n)
- Fields: OPEN, CLOSE, HIGH, LOW, VOLUME
"""

def generate_ideas_with_context(market_context, n=3):
    """
    【核心升级】基于市场情境生成因子
    """
    prompt = f"""
    【Role】
    You are a Senior Quant Researcher. Your goal is to design Alpha factors that work specifically for the CURRENT market environment.
    
    【Current Market Context (Perceived by AI)】
    {market_context}
    
    【Task】
    Based on the market context above (Trend, Volatility, Sentiment), propose {n} new Alpha formulas.
    
    *Strategy Logic Guide*:
    - If market is **Bullish/Trending**: Focus on Momentum (e.g., ROC, Trend strength).
    - If market is **High Volatility/Panic**: Focus on Reversion or Volatility (e.g., Bollinger Band, STD).
    - If market is **Sideways/Low Volume**: Focus on Liquidity or Volume patterns.
    
    【Available Operators】
    {OPERATOR_DOCS}
    
    【Output Format】
    Output ONLY a JSON list:
    [
        {{
            "name": "Context_Aware_Name",
            "formula": "FORMULA_STRING",
            "logic": "Explain why this fits the current market context..."
        }}
    ]
    """
    
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a Python-speaking Quant assistant. Output pure JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        content = completion.choices[0].message.content
        content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content)
    except Exception as e:
        print(f"生成失败: {e}")
        return []

def mine_alphas_pro(rounds=3):
    # 1. 准备数据
    if not os.path.exists(DATA_PATH):
        print("数据文件不存在")
        return
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. 【多模态感知】获取当前市场状态
    # 为了模拟真实挖掘，我们选取数据集中最近的一段时间作为“当下”
    # 在实盘中，这应该是今天；在回测中，可以是训练集的最后一天
    current_date = df['date'].max().strftime('%Y-%m-%d')
    print(f"🌍 正在感知市场环境 ({current_date})...")
    print("   (调用视觉模型读取K线，调用搜索工具读取新闻...)")
    
    # 调用 12_multimodal_utils.py 中的函数
    market_context = analyze_market_multimodal(df, current_date)
    print(f"\n📝 [市场画像]:\n{market_context}\n")
    
    # 3. 开始挖掘循环
    valid_alphas = []
    if os.path.exists(MINED_OUTPUT_FILE):
        with open(MINED_OUTPUT_FILE, "r") as f: valid_alphas = json.load(f)
        
    print(f"🚀 基于上述情报，开始定向挖掘因子 (共 {rounds} 轮)...")
    
    for r in range(rounds):
        print(f"\n--- Round {r+1}/{rounds} ---")
        
        # 传入 Context 进行生成
        candidates = generate_ideas_with_context(market_context, n=3)
        
        if not candidates: continue
            
        for item in candidates:
            name = item.get('name', 'Unknown')
            formula = item.get('formula', '')
            logic = item.get('logic', '')
            
            print(f"   🧪 测试: {name} | 逻辑: {logic[:30]}... ", end="")
            
            # 回测验证
            report = analyze_factor(name, formula)
            
            if "error" in report:
                print(f"[❌ 失败] {report['error']}")
            else:
                print(f"[✅ 成功] IC: {report['IC_Mean']} | Sharpe: {report['Sharpe']}")
                report['logic'] = logic
                report['market_context_used'] = market_context # 记录生成时的环境，方便复盘
                valid_alphas.append(report)
                
                with open(MINED_OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(valid_alphas, f, ensure_ascii=False, indent=4)
                    
        time.sleep(2)

    print(f"\n🎉 挖掘结束！结果已保存至 {MINED_OUTPUT_FILE}")

if __name__ == "__main__":
    if "你的API_KEY" in DASHSCOPE_API_KEY:
        print("请填入 API Key")
    else:
        mine_alphas_pro(rounds=2)