import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
# 引用引擎
try:
    from alpha_engine import AlphaContext
except ImportError:
    print("错误: 未找到 alpha_engine.py")
    sys.exit(1)

# ================= 配置 =================
DATA_PATH = "./data/market_data.csv"
REPORT_FILE = "./reports/all_reports.json" # 从批量计算的结果中读取因子列表
INITIAL_CAPITAL = 1000000
TOP_K = 5

# 如果没有报告文件，默认使用几个经典因子做演示
DEFAULT_ALPHAS = [
    {"name": "Momentum_10", "formula": "CLOSE / DELAY(CLOSE, 10) - 1"},
    {"name": "Reversion_5", "formula": "MA(CLOSE, 5) - CLOSE"},
    {"name": "Vol_20", "formula": "-1 * STD(CLOSE, 20)"},
    {"name": "Volume_Shock", "formula": "VOLUME / MA(VOLUME, 20)"}
]

def prepare_dataset(df, alpha_list):
    """
    计算所有因子的值，并将它们合并为一个特征矩阵 X
    """
    print("正在构建神经网络训练数据...")
    
    # 1. 计算目标 (下期收益率)
    df = df.sort_values(['code', 'date']).reset_index(drop=True)
    df['target'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
    
    # 2. 计算特征 (所有因子值)
    ctx = AlphaContext(df)
    env = {
        'CLOSE': ctx.CLOSE(), 'OPEN': ctx.OPEN(), 'VOLUME': ctx.VOLUME(), 
        'HIGH': ctx.HIGH(), 'LOW': ctx.LOW(),
        'DELAY': ctx.DELAY, 'MA': ctx.MA, 'STD': ctx.STD,
        'TS_MAX': ctx.TS_MAX, 'TS_MIN': ctx.TS_MIN,
        'CORR': ctx.CORR, 'RANK': ctx.RANK
    }
    
    feature_cols = []
    
    for alpha in alpha_list:
        name = alpha['name']
        formula = alpha['formula']
        col_name = f"feat_{name}"
        print(f"   - 计算特征: {name} ...", end="")
        try:
            df[col_name] = eval(formula, {}, env)
            feature_cols.append(col_name)
            print(" [完成]")
        except Exception as e:
            print(f" [失败] {e}")
            
    # 3. 清洗数据 (去除 NaN)
    # 我们需要特征和目标都不为空
    clean_df = df.dropna(subset=feature_cols + ['target']).copy()
    
    # 按照时间排序，这点对金融数据很重要
    clean_df = clean_df.sort_values('date')
    
    return clean_df, feature_cols

def train_dnn_strategy():
    # 1. 加载因子列表
    if os.path.exists(REPORT_FILE):
        with open(REPORT_FILE, 'r', encoding='utf-8') as f:
            alpha_config = json.load(f)
            # 过滤掉之前的报错因子，只取成功的
            alpha_config = [a for a in alpha_config if 'error' not in a]
            # 为了演示速度，如果因子太多，只取前10个
            alpha_config = alpha_config[:10]
    else:
        print("未找到因子报告，使用默认因子列表。")
        alpha_config = DEFAULT_ALPHAS

    print(f"即将组合 {len(alpha_config)} 个 Alpha 因子进行训练。")

    # 2. 加载数据
    if not os.path.exists(DATA_PATH):
        print(f"数据文件不存在: {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. 准备特征矩阵
    full_df, feature_cols = prepare_dataset(df, alpha_config)
    
    if full_df.empty:
        print("错误: 训练数据为空。")
        return

    # 4. 划分训练集和测试集 (按时间切分，模拟真实回测)
    # 前 80% 时间用于训练 MLP，后 20% 时间用于回测
    dates = np.sort(full_df['date'].unique())
    split_idx = int(len(dates) * 0.8)
    split_date = dates[split_idx]
    
    train_df = full_df[full_df['date'] < split_date]
    test_df = full_df[full_df['date'] >= split_date]
    
    print(f"\n数据集划分:")
    print(f"   训练集: {train_df['date'].min().date()} -> {train_df['date'].max().date()} ({len(train_df)} 样本)")
    print(f"   测试集: {test_df['date'].min().date()} -> {test_df['date'].max().date()} ({len(test_df)} 样本)")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values
    X_test = test_df[feature_cols].values
    
    # 5. 数据标准化 (神经网络对数值范围敏感)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. 定义并训练 MLP (对应论文 3.4 节)
    # "Hidden layer with ten ReLU-activated nodes"
    print("\n🚀 开始训练 MLP 神经网络 (Weight Optimization)...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(10,), # 论文设定：1个隐藏层，10个节点
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        alpha=0.001 # L2 正则化
    )
    
    mlp.fit(X_train_scaled, y_train)
    print(f"模型训练完成! 当前 Loss: {mlp.loss_:.6f}")
    
    # 7. 在测试集上生成预测信号 (Composite Alpha)
    print("\n正在生成测试集交易信号...")
    test_df = test_df.copy()
    test_df['predicted_return'] = mlp.predict(X_test_scaled)
    
    # 8. 执行 Top-K 回测 (基于预测值)
    capital = INITIAL_CAPITAL
    capital_curve = []
    test_dates = np.sort(test_df['date'].unique())
    
    for date in test_dates:
        daily_data = test_df[test_df['date'] == date]
        
        # 选出 MLP 预测收益最高的 K 只股票
        if len(daily_data) < TOP_K:
            selected = daily_data
        else:
            selected = daily_data.nlargest(TOP_K, 'predicted_return')
            
        if not selected.empty:
            # 真实的下期收益
            daily_ret = selected['target'].mean() - 0.0003 # 扣费
            capital = capital * (1 + daily_ret)
            
        capital_curve.append({'date': date, 'equity': capital})
        
    # 9. 结果展示
    result_df = pd.DataFrame(capital_curve)
    result_df['date'] = pd.to_datetime(result_df['date'])
    result_df.set_index('date', inplace=True)
    
    total_ret = ((capital / INITIAL_CAPITAL) - 1) * 100
    
    print("=" * 60)
    print(f"📊 MLP 组合策略回测结果")
    print("-" * 60)
    print(f"输入因子数: {len(feature_cols)}")
    print(f"训练集时段: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
    print(f"回测集时段: {test_df['date'].min().date()} ~ {test_df['date'].max().date()}")
    print(f"最终资金: {capital:,.2f}")
    print(f"区间收益率: {total_ret:.2f}%")
    print("=" * 60)
    
    # 绘图
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(result_df.index, result_df['equity'], label='DNN Composite Strategy', color='purple')
        plt.title('Performance of Neural Network Weighted Strategy')
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.grid(True)
        plt.legend()
        
        output_img = "./reports/dnn_strategy_curve.png"
        if not os.path.exists("./reports"): os.makedirs("./reports")
        plt.savefig(output_img)
        print(f"资金曲线已保存至: {output_img}")
        # plt.show()
    except:
        pass

if __name__ == "__main__":
    train_dnn_strategy()