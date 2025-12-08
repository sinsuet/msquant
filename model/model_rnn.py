import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 环境与路径设置 ---
# 获取当前脚本所在的目录 (model文件夹)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (msquant文件夹)
parent_dir = os.path.dirname(current_dir)
# 将父目录加入系统路径
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

try:
    from alpha_engine import AlphaContext
except ImportError:
    print("错误: 未找到 alpha_engine.py。请检查路径设置。")
    sys.exit(1)

# ================= 配置 =================
DATA_PATH = os.path.join(parent_dir, "data", "market_data.csv")
REPORT_FILE = os.path.join(parent_dir, "reports", "all_reports.json")
INITIAL_CAPITAL = 1000000
TOP_K = 5
SEQ_LEN = 20
BATCH_SIZE = 128
EPOCHS = 15
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 模型定义: 可解释 RNN =================
class ExplainableRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(ExplainableRNN, self).__init__()
        # 1. 特征提取器 (RNN)
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 2. 权重生成头 (Weight Generator Head)
        # 根据 RNN 提取的历史状态，决定当前每个因子的权重
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim), # 输出维度 = 因子数量
            nn.Softmax(dim=1)         # 保证权重和为 1
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # Pass 1: 提取时序特征
        out, _ = self.rnn(x)
        # 取最后一个时间步的隐状态作为“环境上下文”
        context = out[:, -1, :] 
        
        # Pass 2: 生成动态权重
        # weights: (batch, input_dim)
        weights = self.weight_generator(context)
        
        # Pass 3: 线性组合预测
        # 我们用生成的权重，去加权“当前时刻”(最后一个时间步)的因子值
        current_factors = x[:, -1, :]
        prediction = (current_factors * weights).sum(dim=1, keepdim=True)
        
        return prediction, weights

# ================= 数据处理 =================
def create_sequences(data, target, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = target[i + seq_len - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def run_rnn_strategy():
    print(f"🚀 启动可解释 RNN 训练 (Device: {DEVICE})...")
    
    # --- 数据加载 ---
    if not os.path.exists(DATA_PATH):
        print(f"数据文件不存在: {DATA_PATH}")
        return
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    if not os.path.exists(REPORT_FILE):
        print(f"报告文件不存在: {REPORT_FILE}")
        return
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        alpha_config = [a for a in json.load(f) if 'error' not in a][:10] # 取前10个有效因子
    
    ctx = AlphaContext(df)
    env = {'CLOSE': ctx.CLOSE(), 'OPEN': ctx.OPEN(), 'VOLUME': ctx.VOLUME(), 'HIGH': ctx.HIGH(), 'LOW': ctx.LOW(), 'DELAY': ctx.DELAY, 'MA': ctx.MA, 'STD': ctx.STD, 'TS_MAX': ctx.TS_MAX, 'TS_MIN': ctx.TS_MIN, 'CORR': ctx.CORR, 'RANK': ctx.RANK}
    
    feature_cols = []
    print("正在计算因子数据...", end="")
    for alpha in alpha_config:
        try: 
            col = f"feat_{alpha['name']}"
            df[col] = eval(alpha['formula'], {}, env)
            feature_cols.append(col)
        except: pass
    print("完成")
        
    df['target'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
    df = df.dropna().sort_values(['code', 'date']).reset_index(drop=True)
    
    split_date = pd.Timestamp("2022-10-26")
    train_raw = df[df['date'] < split_date]
    test_raw = df[df['date'] >= split_date]
    
    scaler = StandardScaler()
    scaler.fit(train_raw[feature_cols])
    
    def process_by_code(sub_df):
        # 标准化
        sub_df[feature_cols] = scaler.transform(sub_df[feature_cols])
        X = sub_df[feature_cols].values
        y = sub_df['target'].values
        dates = sub_df['date'].values
        
        if len(X) <= SEQ_LEN: return None, None, None
        
        # 生成序列
        X_seq, y_seq = create_sequences(X, y, SEQ_LEN)
        
        # 生成对应的日期序列 (从第 SEQ_LEN-1 个开始)
        d_seq = dates[SEQ_LEN-1:]
        
        # === 【关键修复】强制对齐长度 ===
        # create_sequences 的 range 逻辑可能导致比 d_seq 少 1 个元素
        # 使用 min 取三者最小长度，确保一一对应
        min_len = min(len(X_seq), len(y_seq), len(d_seq))
        
        return X_seq[:min_len], y_seq[:min_len], d_seq[:min_len]

    def build_dataset(raw_df):
        all_X, all_y, all_dates = [], [], []
        for code, sub in raw_df.groupby('code'):
            x, y, d = process_by_code(sub.copy())
            if x is not None: 
                all_X.append(x)
                all_y.append(y)
                all_dates.append(d)
        
        if not all_X:
            return np.array([]), np.array([]), np.array([])
            
        return np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_dates)

    print("正在构建时序数据 (这可能需要一点时间)...")
    X_train, y_train, _ = build_dataset(train_raw)
    X_test, y_test, dates_test = build_dataset(test_raw)
    
    if len(X_train) == 0:
        print("错误: 训练数据为空，请检查数据量或 SEQ_LEN 设置")
        return

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    
    # --- 训练 ---
    model = ExplainableRNN(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print("开始训练模型...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            preds, _ = model(X_b) # 接收两个输出，只优化 pred
            loss = criterion(preds.squeeze(), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1)%5==0: 
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.6f}")

    # --- 回测与权重提取 ---
    print("正在进行回测...")
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        preds, weights = model(test_tensor) # 获取权重
        preds = preds.cpu().numpy().flatten()
        weights = weights.cpu().numpy()
        
    # 计算收益
    backtest_df = pd.DataFrame({'date': dates_test, 'pred': preds, 'target': y_test})
    capital, curve = INITIAL_CAPITAL, []
    unique_dates = np.sort(np.unique(dates_test))
    
    for date in unique_dates:
        daily = backtest_df[backtest_df['date'] == date]
        if len(daily) >= TOP_K:
            ret = daily.nlargest(TOP_K, 'pred')['target'].mean() - 0.0003
            capital *= (1 + ret)
        curve.append(capital)
        
# ... (前面的代码保持不变) ...
    
    print(f"📊 最终收益: {(capital/INITIAL_CAPITAL-1)*100:.2f}%")
    
    # --- 可视化 ---
    report_dir = os.path.join(parent_dir, 'reports')
    if not os.path.exists(report_dir): os.makedirs(report_dir)

    # 1. 资金曲线
    plt.figure(figsize=(10, 4))
    plt.plot(unique_dates, curve, label='BiLSTM Strategy', color='orange')
    plt.title('BiLSTM Equity Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(report_dir, 'bilstm_equity.png'))
    plt.close() # 关闭画布释放内存
    
    # 2. 因子权重热力图
    # 构造 DataFrame
    feature_names = [c.replace('feat_', '') for c in feature_cols]
    w_df = pd.DataFrame(weights, columns=feature_names)
    w_df['date'] = dates_test
    
    # === 【关键修复】 ===
    # 显式只选取因子列进行聚合，排除 'date' 列
    # group key 是年-月 (例如 "2023-05")
    w_monthly = w_df.groupby(w_df['date'].astype(str).str[:7])[feature_names].mean()
    
    plt.figure(figsize=(12, 6))
    # 绘制热力图
    sns.heatmap(w_monthly.T, cmap="viridis", annot=False)
    plt.title('BiLSTM Dynamic Factor Weights (Monthly Avg)')
    plt.xlabel('Month')
    plt.ylabel('Factors')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, 'bilstm_weights_heatmap.png'))
    plt.close()
    
    print(f"✅ 结果已保存至 {report_dir}")

if __name__ == "__main__":
    run_rnn_strategy()