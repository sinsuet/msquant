import json
import os
import time
from openai import OpenAI

# ==========================================
# 配置部分：切换为 Qwen (通义千问)
# ==========================================
# 请替换为你的阿里云 DashScope API Key
DASHSCOPE_API_KEY = "sk-d807fcc0e09e40b9a3d6f736aad39c15"

if "你的API_KEY" in DASHSCOPE_API_KEY:
    print("请先在 5_tournament.py 中填入你的阿里云 API Key！")
    exit()

# 初始化 Client
client = OpenAI(
    api_key=DASHSCOPE_API_KEY, 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ==========================================
# 市场环境描述 (Context) - 模拟论文中的宏观输入
# ==========================================
MARKET_CONTEXT = """
【当前市场状态】
1. 趋势：市场呈现震荡下行趋势，指数在均线下方运行。
2. 流动性：成交量持续萎缩，存量博弈特征明显。
3. 情绪：投资者避险情绪升温，高位股补跌，缺乏持续性主线。
4. 风格：低估值、高分红的防御性板块相对抗跌，题材炒作退潮。
"""

def llm_judge(defender, challenger):
    """
    LLM 裁判函数：决定谁是更好的因子
    """
    prompt = f"""
    【任务】
    你是一场 Alpha 因子挖掘锦标赛的首席裁判。你的任务是根据当前的市场环境，对比两个策略因子的逻辑和表现，选出下个季度更可能盈利的一个。

    【当前市场环境】
    {MARKET_CONTEXT}

    【选手 A (当前擂主): {defender['name']}】
    - 逻辑公式: {defender['formula']}
    - 核心逻辑: {defender.get('logic', '无')}
    - IC均值 (预测能力): {defender['IC_Mean']}
    - 年化收益: {defender['Annual_Return']}
    - 夏普比率: {defender['Sharpe']}

    【选手 B (挑战者): {challenger['name']}】
    - 逻辑公式: {challenger['formula']}
    - 核心逻辑: {challenger.get('logic', '无')}
    - IC均值 (预测能力): {challenger['IC_Mean']}
    - 年化收益: {challenger['Annual_Return']}
    - 夏普比率: {challenger['Sharpe']}

    【评判标准】
    1. **逻辑适应性**：因子的经济学逻辑是否适应当前“震荡下行、存量博弈”的市场？(例如：动量策略在震荡市易亏损，反转或低波策略可能更优)。
    2. **风险收益比**：优先选择夏普比率高、回撤风险可控的因子，而不仅是看年化收益。
    3. **稳定性**：IC均值越高越稳定。

    【输出要求】
    请只输出一个标准的 JSON 对象，不要包含 Markdown 格式或其他废话。格式如下：
    {{
        "analysis": "简短分析两者的优劣和市场适应性...",
        "winner": "A 或 B",
        "winner_name": "获胜因子的名称",
        "reason": "一句话决定性理由"
    }}
    """
    
    try:
        completion = client.chat.completions.create(
            model="qwen-plus", # 推荐使用 qwen-plus 或 qwen-max
            messages=[
                {"role": "system", "content": "你是一个严谨的量化投资总监，擅长因子评价。请只输出JSON。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, # 降低随机性
            response_format={"type": "json_object"} # 强制 JSON 输出
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        # 出错时默认擂主卫冕，防止中断
        return {"winner": "A", "winner_name": defender['name'], "reason": "裁判连接中断，擂主自动卫冕"}

def run_tournament():
    # 1. 加载报告
    report_path = "./reports/all_reports.json"
    if not os.path.exists(report_path):
        print(f"错误: 未找到报告文件 {report_path}。请先运行 4_batch_processor.py")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        factors = json.load(f)
    
    if len(factors) < 2:
        print("错误: 有效因子数量不足 2 个，无法举办锦标赛。请检查计算过程。")
        return

    # 2. 初始化擂台
    current_champion = factors[0]
    print("\n" + "="*60)
    print(f"🏆 Alpha 挖掘锦标赛正式开始！")
    print(f"📊 参赛因子数: {len(factors)}")
    print(f"👑 初始擂主: {current_champion['name']} (夏普: {current_champion['Sharpe']})")
    print("="*60)

    # 3. 循环挑战 (Round-Robin)
    win_count = 0
    
    for i, challenger in enumerate(factors[1:]):
        print(f"\n>> [第 {i+1} 轮] 擂主 vs 挑战者 ({challenger['name']})")
        
        # 调用 LLM 裁判
        result = llm_judge(current_champion, challenger)
        
        # 解析结果
        analysis = result.get("analysis", "无分析")
        winner = result.get("winner", "A")
        reason = result.get("reason", "无理由")
        
        print(f"   📝 裁判分析: {analysis}")
        
        # 判定胜负
        if winner == "B" or result.get("winner_name") == challenger['name']:
            print(f"   ✨ 挑战成功！{challenger['name']} 成为新擂主！")
            print(f"   💡 理由: {reason}")
            current_champion = challenger
            win_count = 0 # 重置连胜
        else:
            print(f"   🛡️ 卫冕成功！{current_champion['name']} 守住了擂台。")
            print(f"   💡 理由: {reason}")
            win_count += 1
            
        time.sleep(1) # 避免 API 速率限制

    # 4. 宣布最终结果
    print("\n" + "="*60)
    print(f"🎉 最终冠军 (Alpha King): {current_champion['name']}")
    print("-" * 60)
    print(f"   - 核心逻辑: {current_champion.get('logic', '无')}")
    print(f"   - 公式: {current_champion['formula']}")
    print(f"   - 夏普比率: {current_champion['Sharpe']}")
    print(f"   - 年化收益: {current_champion['Annual_Return']}")
    print(f"   - IC 均值: {current_champion['IC_Mean']}")
    print("="*60)
    
    # 保存结果
    with open("./reports/final_champion.json", "w", encoding="utf-8") as f:
        json.dump(current_champion, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    run_tournament()