import baostock as bs
import pandas as pd
import os
import akshare as ak # 仅用于获取成分股列表，这个请求量小不容易挂
from tqdm import tqdm

# ================= 配置 =================
DATA_DIR = "./data"
START_DATE = "2014-01-01"  # Baostock 日期格式为 YYYY-MM-DD
END_DATE = "2024-12-31"
ADJUST = "1"               # 1: 后复权 (High Frequency Adjusted), 2: 前复权, 3: 不复权

def get_sse50_history_baostock():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    print(f"🚀 启动 Baostock 数据下载引擎...")
    print(f"📅 时间跨度: {START_DATE} 至 {END_DATE}")
    
    # 1. 登录系统
    lg = bs.login()
    if lg.error_code != '0':
        print(f"❌ 登录失败: {lg.error_msg}")
        return

    # 2. 获取成分股列表 (使用 AkShare 获取列表，这通常很安全)
    try:
        print("正在获取上证50成分股列表...")
        index_stock_cons = ak.index_stock_cons_sina(symbol="000016")
        stock_codes = index_stock_cons['symbol'].tolist()
        # Baostock 需要格式: sh.600519
        # AkShare 返回格式: sh600519
        # 转换: sh600519 -> sh.600519
        bao_codes = [code[:2] + "." + code[2:] for code in stock_codes]
        print(f"✅ 获取成功，共 {len(bao_codes)} 只股票。")
    except Exception as e:
        print(f"❌ 成分股列表获取失败: {e}, 使用备用列表")
        # 备用：茅台、平安、招行等
        bao_codes = ["sh.600519", "sh.601318", "sh.600036", "sh.601012", "sh.600276"]

    all_data = []
    
    # 3. 循环下载
    pbar = tqdm(bao_codes)
    for code in pbar:
        pbar.set_description(f"下载 {code}")
        
        # query_history_k_data_plus 参数详解：
        # code: 股票代码
        # fields: 我们需要的字段
        # frequency: d=日线
        # adjustflag: 1=后复权 (对应之前逻辑)
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount",
            start_date=START_DATE, 
            end_date=END_DATE,
            frequency="d", 
            adjustflag=ADJUST
        )
        
        if rs.error_code != '0':
            print(f"⚠️ {code} 下载出错: {rs.error_msg}")
            continue
            
        # 将结果集转换为 DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            continue
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 4. 数据清洗与格式对齐
        # Baostock 返回的 code 是 sh.600519，我们需要转回 sh600519 以匹配之前的代码
        df['code'] = df['code'].str.replace('.', '', regex=False)
        
        # 类型转换 (Baostock 返回的都是字符串)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
            
        # 停牌检查 (成交量为0)
        df = df[df['volume'] > 0]
        
        # 格式必须包含: date, code, open, high, low, close, volume
        # Baostock 默认就包含这些
        
        all_data.append(df)

    # 5. 登出系统
    bs.logout()

    # 6. 合并保存
    if all_data:
        full_df = pd.concat(all_data)
        # 排序
        full_df = full_df.sort_values(['code', 'date']).reset_index(drop=True)
        
        output_path = os.path.join(DATA_DIR, "market_data.csv")
        full_df.to_csv(output_path, index=False)
        
        print("\n" + "="*50)
        print(f"🎉 历史数据下载完成！")
        print(f"📊 总行数: {len(full_df)}")
        print(f"💾 已保存至: {output_path}")
        print("="*50)
        print(full_df.head(2))
    else:
        print("❌ 未下载到有效数据")

if __name__ == "__main__":
    get_sse50_history_baostock()