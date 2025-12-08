import pandas as pd
import mplfinance as mpf
import os
import base64
import datetime
from openai import OpenAI
from duckduckgo_search import DDGS  # 引入搜索工具

# ================= 配置 =================
# 1. 填入支持视觉模型 (Qwen-VL) 的 API Key
DASHSCOPE_API_KEY = "sk-d807fcc0e09e40b9a3d6f736aad39c15" 

# 2. 图片保存目录
IMG_DIR = "./temp_images"
if not os.path.exists(IMG_DIR): os.makedirs(IMG_DIR)

def generate_kline_chart(df, date, window=30, save_path="market_snapshot.png"):
    """
    [Visual Data] 生成 K 线图 (保持不变)
    """
    end_date = pd.to_datetime(date)
    start_date = end_date - datetime.timedelta(days=window*2) # 多取一点确保有数据
    
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    sub_df = df.loc[mask].copy()
    
    if len(sub_df) < 10: return None
    
    sub_df = sub_df.tail(window)
    sub_df.set_index('date', inplace=True)
    
    mc = mpf.make_marketcolors(up='red', down='green', edge='i', wick='i', volume='in', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)
    
    # 关键：volume=True 展示成交量，这对分析情绪很重要
    mpf.plot(sub_df, type='candle', mav=(5, 10, 20), volume=True, 
             title=f"Market Trend until {date}", style=s, 
             savefig=dict(fname=save_path, dpi=100, bbox_inches='tight'))
    
    return save_path

def search_historical_news(date_str, max_results=3):
    """
    [RAG Retrieval] 真实的联网搜索
    模拟从新闻数据库中检索当天的市场分析
    """
    # 构造精准的搜索 Query
    # 例如: "2023-12-01 A股 收评 上证50 走势"
    query = f"{date_str} A股 上证50 市场收评 走势分析"
    
    print(f"   🔍 正在联网检索新闻: {query} ...")
    
    results_text = ""
    try:
        # 使用 DuckDuckGo 进行搜索 (模拟 RAG 的 Retrieve 过程)
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return "未检索到相关新闻数据。"
            
        for i, res in enumerate(results):
            results_text += f"【新闻 {i+1}】{res['title']}\n摘要: {res['body']}\n"
            
    except Exception as e:
        print(f"   ❌ 搜索失败 (可能是网络问题): {e}")
        results_text = "网络搜索不可用，仅依靠技术面数据分析。"
        
    return results_text

def get_financial_context(df, date):
    """
    [Context Builder] 融合 数值数据 + 检索到的文本数据
    """
    date_dt = pd.to_datetime(date)
    date_str = date_dt.strftime('%Y-%m-%d')
    
    # 1. 计算技术面硬数据 (Hard Data)
    recent_df = df[df['date'] <= date_dt].tail(5)
    if recent_df.empty: return "数据不足"
    
    curr = recent_df.iloc[-1]
    prev = recent_df.iloc[0]
    pct_change = (curr['close'] - prev['close']) / prev['close']
    
    # 技术面摘要
    tech_summary = f"截止 {date_str}，上证50收盘 {curr['close']:.2f}。"
    tech_summary += f"近5日涨跌幅: {pct_change*100:.2f}%。"
    tech_summary += f"成交量: {curr['volume']/10000:.0f}万手。"

    # 2. 真实联网检索 (Soft Data / RAG)
    news_context = search_historical_news(date_str)
    
    # 3. 组合 Prompt
    final_context = f"""
    【技术面概览 (Quantitative)】
    {tech_summary}
    
    【市场新闻检索 (Qualitative / RAG)】
    {news_context}
    """
    return final_context

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_market_multimodal(df, current_date):
    """
    [Multimodal Agent] 看图 + 读新闻 -> 生成观点
    """
    # 1. 准备视觉数据 (Visual)
    img_path = os.path.join(IMG_DIR, "current_market.png")
    real_img_path = generate_kline_chart(df, current_date, save_path=img_path)
    
    # 2. 准备文本/RAG数据 (Textual)
    rag_context = get_financial_context(df, current_date)
    
    if not real_img_path: return "数据不足，无法分析"

    # 3. 调用多模态大模型
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    base64_image = encode_image(real_img_path)
    
    prompt = f"""
    你是一位资深的宏观策略分析师。请结合【K线图表】和【检索到的市场新闻】，对当前市场进行综合研判。
    
    【输入信息】
    {rag_context}
    
    【分析任务】
    请输出一份简洁的市场环境报告（Context Description），必须包含：
    1. **趋势定性**: (上涨/下跌/震荡) 并说明是技术面驱动还是消息面驱动。
    2. **关键事件**: 摘要中提到的影响市场的核心事件（如降息、财报、地缘政治）。
    3. **情绪评分**: 给出 0-10 的市场情绪分（0恐慌，10贪婪）。
    
    请直接输出分析结果，不要啰嗦。
    """
    
    try:
        response = client.chat.completions.create(
            model="qwen-vl-plus", # 必须使用支持视觉的模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"多模态分析异常: {e}\n(仅使用基础数据): {rag_context}"

if __name__ == "__main__":
    # 测试代码
    if os.path.exists("./data/market_data.csv"):
        df = pd.read_csv("./data/market_data.csv")
        df['date'] = pd.to_datetime(df['date'])
        
        # 选取一个历史上比较特殊的日期进行测试
        # 例如 2023-08-28 (印花税减半，会有大阴线/假阳线和重磅新闻)
        test_date = "2023-08-28" 
        
        print(f"🌍 正在对 {test_date} 进行多模态 RAG 分析...")
        analysis = analyze_market_multimodal(df, test_date)
        print("\n🤖 [AI 分析报告]:")
        print("-" * 50)
        print(analysis)
        print("-" * 50)
    else:
        print("请先下载数据")