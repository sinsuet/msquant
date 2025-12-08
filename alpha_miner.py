import json
import os
import time
import re
from openai import OpenAI
# 引用你的回测引擎
from alpha_engine import analyze_factor

# =================配置部分=================
# 请替换为你的阿里云 API Key
DASHSCOPE_API_KEY = "sk-d807fcc0e09e40b9a3d6f736aad39c15"

client = OpenAI(
    api_key=DASHSCOPE_API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 保存挖掘结果的文件
MINED_OUTPUT_FILE = "./reports/mined_alphas.json"

# 定义 LLM 可用的算子文档 (作为 Prompt 的一部分)
OPERATOR_DOCS = """
Available Operators:
- DELAY(x, n): x shifted back by n days.
- MA(x, n): Moving average of x for past n days.
- STD(x, n): Moving standard deviation.
- TS_MAX(x, n) / TS_MIN(x, n): Max/Min value in past n days.
- RANK(x): Cross-sectional rank (0.0 to 1.0) of x across all stocks.
- CORR(x, y, n): Rolling correlation between x and y.
- Data fields: OPEN, CLOSE, HIGH, LOW, VOLUME
"""

def generate_ideas(n=3):
    """
    让 LLM 生成 n 个新的因子想法
    """
    prompt = f"""
    【Role】
    You are a creative Quantitative Researcher. Your goal is to discover NEW Alpha factors for the stock market.
    
    【Context】
    We have a backtesting engine with the following operators:
    {OPERATOR_DOCS}
    
    【Task】
    Please generate {n} unique and syntactically correct Alpha formulas.
    They should be diverse (Momentum, Reversion, Volatility, etc.).
    
    【Format】
    Output ONLY a JSON list. Do not write markdown code blocks (```).
    Format:
    [
        {{
            "name": "Unique_Name_1",
            "formula": "FORMULA_STRING",
            "logic": "Economic rationale..."
        }},
        ...
    ]
    """
    
    try:
        print(f"🤖 正在思考新的量化策略 (请求生成 {n} 个)...")
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a Python-speaking Quant assistant. Output pure JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7 # 稍微调高温度，增加创造性
        )
        content = completion.choices[0].message.content
        
        # 清洗可能存在的 markdown 标记
        content = content.replace("```json", "").replace("```", "").strip()
        
        factors = json.loads(content)
        return factors
    except Exception as e:
        print(f"LLM 生成失败: {e}")
        return []

def mine_alphas(rounds=3):
    """
    挖掘主循环：生成 -> 测试 -> 保存
    """
    # 加载已有的挖掘记录
    if os.path.exists(MINED_OUTPUT_FILE):
        with open(MINED_OUTPUT_FILE, "r", encoding="utf-8") as f:
            valid_alphas = json.load(f)
    else:
        valid_alphas = []
        
    print(f"🚀 开始自动化挖掘... (计划运行 {rounds} 轮)")
    
    for r in range(rounds):
        print(f"\n--- Round {r+1}/{rounds} ---")
        
        # 1. 生成想法
        candidates = generate_ideas(n=3)
        
        if not candidates:
            continue
            
        # 2. 立即验证
        for item in candidates:
            name = item.get('name', 'Unknown')
            formula = item.get('formula', '')
            logic = item.get('logic', '')
            
            print(f"   🧪 测试因子: {name} ... ", end="")
            
            # 调用引擎回测
            report = analyze_factor(name, formula)
            
            if "error" in report:
                print(f"[❌ 失败] {report['error']}")
                # 高级玩法：这里可以将错误信息喂回给 LLM 让它 debug (Self-Correction)
            else:
                # 只有通过测试的才保存
                print(f"[✅ 成功] IC: {report['IC_Mean']} | Sharpe: {report['Sharpe']}")
                
                # 补充逻辑说明
                report['logic'] = logic
                valid_alphas.append(report)
                
                # 实时保存，防止中断丢失
                with open(MINED_OUTPUT_FILE, "w", encoding="utf-8") as f:
                    json.dump(valid_alphas, f, ensure_ascii=False, indent=4)
                    
        time.sleep(2) # 休息一下

    print(f"\n🎉 挖掘结束！共发现 {len(valid_alphas)} 个有效因子。")
    print(f"结果已保存至: {MINED_OUTPUT_FILE}")

if __name__ == "__main__":
    if "你的API_KEY" in DASHSCOPE_API_KEY:
        print("请先配置 API Key！")
    else:
        mine_alphas(rounds=2) # 运行 2 轮尝试