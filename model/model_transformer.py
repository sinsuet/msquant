import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. 环境与路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
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
LR = 0.0005 # Transformer 建议低学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 模型定义: 可解释 Transformer =================
class ExplainableTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super(ExplainableTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 权重生成头
        self.weight_generator = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # 1. Embed & Transform
        x_emb = self.embedding(x)
        out = self.transformer_encoder(x_emb)
        
        # 2. Pooling (取平均)
        context = out.mean(dim=1)
        
        # 3. Generate Weights
        weights = self.weight_generator(context)
        
        # 4. Weighted Prediction
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

def run_transformer_strategy():
    print(f"🚀 启动可解释 Transformer 训练 (Device: {DEVICE})...")
    
    # --- 数据加载 ---
    if not os.path.exists(DATA_PATH): return
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    with open(REPORT_FILE, 'r', encoding='utf-8') as f:
        alpha_config = [a for a in json.load(f) if 'error' not in a][:10]
    
    ctx = AlphaContext(df)
    env = {'CLOSE': ctx.CLOSE(), 'OPEN': ctx.OPEN(), 'VOLUME': ctx.VOLUME(), 'HIGH': ctx.HIGH(), 'LOW': ctx.LOW(), 'DELAY': ctx.DELAY, 'MA': ctx.MA, 'STD': ctx.STD, 'TS_MAX': ctx.TS_MAX, 'TS_MIN': ctx.TS_MIN, 'CORR': ctx.CORR, 'RANK': ctx.RANK}
    feature_cols = []
    print("计算因子...", end="")
    for alpha in alpha_config:
        try: df[f"feat_{alpha['name']}"] = eval(alpha['formula'], {}, env); feature_cols.append(f"feat_{alpha['name']}")
        except: pass
    print("完成")
        
    df['target'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
    df = df.dropna().sort_values(['code', 'date']).reset_index(drop=True)
    split_date = pd.Timestamp("2022-10-26")
    train_raw = df[df['date'] < split_date]
    test_raw = df[df['date'] >= split_date]
    scaler = StandardScaler(); scaler.fit(train_raw[feature_cols])
    
    def process_by_code(sub_df):
        sub_df[feature_cols] = scaler.transform(sub_df[feature_cols])
        X, y, dates = sub_df[feature_cols].values, sub_df['target'].values, sub_df['date'].values
        if len(X) <= SEQ_LEN: return None, None, None
        
        X_seq, y_seq = create_sequences(X, y, SEQ_LEN)
        d_seq = dates[SEQ_LEN-1:]
        
        # 【关键修复】长度对齐
        min_len = min(len(X_seq), len(y_seq), len(d_seq))
        return X_seq[:min_len], y_seq[:min_len], d_seq[:min_len]

    def build_dataset(raw_df):
        all_X, all_y, all_dates = [], [], []
        for code, sub in raw_df.groupby('code'):
            x, y, d = process_by_code(sub.copy())
            if x is not None: all_X.append(x); all_y.append(y); all_dates.append(d)
        
        if not all_X: return np.array([]), np.array([]), np.array([])
        return np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_dates)

    print("构建时序数据...")
    X_train, y_train, _ = build_dataset(train_raw)
    X_test, y_test, dates_test = build_dataset(test_raw)
    
    if len(X_train) == 0:
        print("错误: 训练数据为空")
        return

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    
    # --- 训练 ---
    model = ExplainableTransformer(input_dim=len(feature_cols)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print("开始训练...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            preds, _ = model(X_b)
            loss = criterion(preds.squeeze(), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1)%5==0: print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.6f}")

    # --- 回测 ---
    print("回测中...")
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        preds, weights = model(test_tensor)
        preds = preds.cpu().numpy().flatten()
        weights = weights.cpu().numpy()
        
    backtest_df = pd.DataFrame({'date': dates_test, 'pred': preds, 'target': y_test})
    capital, curve = INITIAL_CAPITAL, []
    unique_dates = np.sort(np.unique(dates_test))
    for date in unique_dates:
        daily = backtest_df[backtest_df['date'] == date]
        if len(daily) >= TOP_K:
            ret = daily.nlargest(TOP_K, 'pred')['target'].mean() - 0.0003
            capital *= (1 + ret)
        curve.append(capital)
        
    
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
    run_transformer_strategy()