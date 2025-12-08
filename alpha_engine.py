import pandas as pd
import numpy as np
import os

# --- 1. 定义算子 (修复版) ---
class AlphaContext:
    def __init__(self, df):
        self.df = df
        # 缓存分组键，而不是分组对象
        self.codes = df['code']
        self.dates = df['date']
    
    # 基础数据获取
    def CLOSE(self): return self.df['close']
    def OPEN(self): return self.df['open']
    def HIGH(self): return self.df['high']
    def LOW(self): return self.df['low']
    def VOLUME(self): return self.df['volume']
    
    # --- 时序算子 (Time-Series) ---
    # 关键修复：series.groupby(self.codes)... 确保只对输入序列计算
    
    def DELAY(self, series, n):
        return series.groupby(self.codes).shift(n)
    
    def MA(self, series, n):
        # 对输入 series 按 code 分组，然后计算 rolling mean
        return series.groupby(self.codes).transform(lambda x: x.rolling(n).mean())
    
    def STD(self, series, n):
        return series.groupby(self.codes).transform(lambda x: x.rolling(n).std())
    
    def TS_MAX(self, series, n):
        return series.groupby(self.codes).transform(lambda x: x.rolling(n).max())

    def TS_MIN(self, series, n):
        return series.groupby(self.codes).transform(lambda x: x.rolling(n).min())

    def CORR(self, series_a, series_b, n):
        # 滚动相关性
        # 使用 transform 配合 loc 确保在每个分组内，series_b 能正确对齐
        return series_a.groupby(self.codes).transform(
            lambda x: x.rolling(n).corr(series_b.loc[x.index])
        )

    # --- 截面算子 (Cross-Sectional) ---
    
    def RANK(self, series):
        # 按日期分组，对当天的所有股票进行排名
        return series.groupby(self.dates).rank(pct=True)

# --- 2. 回测分析函数 ---
def analyze_factor(factor_name, formula, data_path="./data/market_data.csv"):
    if not os.path.exists(data_path):
        return {"name": factor_name, "error": f"数据文件不存在: {data_path}"}

    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    # 强制排序并重置索引，这对相关性计算至关重要
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    
    # 计算下期收益率 (Target)
    df['next_ret'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
    
    # 初始化上下文
    ctx = AlphaContext(df)
    
    # 注册算子环境
    env = {
        'CLOSE': ctx.CLOSE(), 'OPEN': ctx.OPEN(), 'VOLUME': ctx.VOLUME(), 
        'HIGH': ctx.HIGH(), 'LOW': ctx.LOW(),
        'DELAY': ctx.DELAY, 'MA': ctx.MA, 'STD': ctx.STD,
        'TS_MAX': ctx.TS_MAX, 'TS_MIN': ctx.TS_MIN,
        'CORR': ctx.CORR, 'RANK': ctx.RANK
    }
    
    try:
        # 执行公式
        df['factor'] = eval(formula, {}, env)
    except Exception as e:
        return {"name": factor_name, "formula": formula, "error": f"公式错误: {e}"}

    # 数据清洗
    # 去除因子值为 NaN 或 下期收益为 NaN 的行
    clean_df = df.dropna(subset=['factor', 'next_ret'])
    
    # 额外检查：如果所有因子值都一样(例如全0)，相关性计算会报错或为NaN
    if clean_df['factor'].nunique() <= 1:
        return {"name": factor_name, "formula": formula, "error": "因子值无效(全部相同或为空)"}

    if clean_df.empty:
        return {"name": factor_name, "formula": formula, "error": "数据清洗后为空"}

    # --- 3. 计算指标 ---
    # IC 计算
    daily_ic = clean_df.groupby('date').apply(
        lambda x: x['factor'].corr(x['next_ret'], method='spearman')
    )
    ic = daily_ic.mean()
    
    # 年化收益 (多空策略)
    def daily_ret(day_df):
        if len(day_df) < 2: return 0.0
        try:
            long_ret = day_df.nlargest(1, 'factor')['next_ret'].mean()
            short_ret = day_df.nsmallest(1, 'factor')['next_ret'].mean()
            return long_ret - short_ret
        except:
            return 0.0
        
    daily_returns = clean_df.groupby('date').apply(daily_ret)
    avg_daily_return = daily_returns.mean()
    
    annual_return = avg_daily_return * 252
    volatility = daily_returns.std() * np.sqrt(252)
    sharpe = annual_return / volatility if volatility != 0 else 0

    return {
        "name": factor_name,
        "formula": formula,
        "IC_Mean": round(ic, 4) if not np.isnan(ic) else 0,
        "Annual_Return": f"{round(annual_return * 100, 2)}%",
        "Sharpe": round(sharpe, 2)
    }

if __name__ == "__main__":
    # 简单测试
    print(analyze_factor("Test", "MA(CLOSE, 5)"))