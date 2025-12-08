import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

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
    print("错误: 未找到 alpha_engine.py")
    sys.exit(1)

# ================= 1. 超参数配置 =================
CONFIG = {
    "model_name": "BiLSTM",
    "data_path": os.path.join(parent_dir, "data", "market_data.csv"),
    "report_path": os.path.join(parent_dir, "reports", "all_reports.json"),
    "initial_capital": 1000000,
    "top_k": 5,
    "seq_len": 20,
    "batch_size": 128,
    "epochs": 15,
    "lr": 0.001,
    "hidden_dim": 64,
    "num_layers": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ================= 2. 模型定义 =================
class ExplainableBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ExplainableBiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # 双向输出维度是 hidden_dim * 2
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        context = out[:, -1, :] 
        weights = self.weight_generator(context)
        current_factors = x[:, -1, :]
        prediction = (current_factors * weights).sum(dim=1, keepdim=True)
        return prediction, weights

# ================= 3. 辅助函数 (保持一致) =================
def calculate_max_drawdown(equity_curve):
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return drawdown.max()

def create_sequences(data, target, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:(i + seq_len)])
        ys.append(target[i + seq_len - 1])
    return np.array(xs), np.array(ys)

def save_report(metrics, params, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"====== {params['model_name']} Strategy Report ======\n\n")
        f.write("[1] Model Hyperparameters:\n")
        for k, v in params.items():
            if "path" not in k:
                f.write(f"  {k:<15}: {v}\n")
        f.write("\n[2] Performance Metrics:\n")
        f.write(f"  Total Return   : {metrics['total_return']:.2f}%\n")
        f.write(f"  Annualized Ret : {metrics['annualized_return']:.2f}%\n")
        f.write(f"  Max Drawdown   : {metrics['max_drawdown']:.2f}%\n")
        f.write(f"  Sharpe Ratio   : {metrics['sharpe_ratio']:.4f}\n")
        f.write(f"  Win Rate       : {metrics['win_rate']:.2f}%\n")
        f.write(f"  Final Capital  : {metrics['final_capital']:.2f}\n")

# ================= 4. 主流程 =================
def run_bilstm_strategy():
    print(f"🚀 启动 {CONFIG['model_name']} 策略训练 (Device: {CONFIG['device']})...")
    
    if not os.path.exists(CONFIG['data_path']): return
    df = pd.read_csv(CONFIG['data_path'])
    df['date'] = pd.to_datetime(df['date'])
    with open(CONFIG['report_path'], 'r', encoding='utf-8') as f:
        alpha_config = [a for a in json.load(f) if 'error' not in a][:10]
    
    ctx = AlphaContext(df)
    env = {'CLOSE': ctx.CLOSE(), 'OPEN': ctx.OPEN(), 'VOLUME': ctx.VOLUME(), 'HIGH': ctx.HIGH(), 'LOW': ctx.LOW(), 'DELAY': ctx.DELAY, 'MA': ctx.MA, 'STD': ctx.STD, 'TS_MAX': ctx.TS_MAX, 'TS_MIN': ctx.TS_MIN, 'CORR': ctx.CORR, 'RANK': ctx.RANK}
    feature_cols = []
    for alpha in alpha_config:
        try: df[f"feat_{alpha['name']}"] = eval(alpha['formula'], {}, env); feature_cols.append(f"feat_{alpha['name']}")
        except: pass
        
    df['target'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
    df = df.dropna().sort_values(['code', 'date']).reset_index(drop=True)
    split_date = pd.Timestamp("2022-10-26")
    train_raw = df[df['date'] < split_date]
    test_raw = df[df['date'] >= split_date]
    scaler = StandardScaler(); scaler.fit(train_raw[feature_cols])
    
    def process_by_code(sub_df):
        sub_df[feature_cols] = scaler.transform(sub_df[feature_cols])
        X, y, dates = sub_df[feature_cols].values, sub_df['target'].values, sub_df['date'].values
        if len(X) <= CONFIG['seq_len']: return None, None, None
        X_seq, y_seq = create_sequences(X, y, CONFIG['seq_len'])
        d_seq = dates[CONFIG['seq_len']-1:]
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
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 训练
    model = ExplainableBiLSTM(len(feature_cols), CONFIG['hidden_dim'], CONFIG['num_layers']).to(CONFIG['device'])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(CONFIG['device']), y_b.to(CONFIG['device'])
            optimizer.zero_grad()
            preds, _ = model(X_b)
            loss = criterion(preds.squeeze(), y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1)%5==0: print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {total_loss/len(train_loader):.6f}")

    # 回测
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(CONFIG['device'])
        preds, weights = model(test_tensor)
        preds = preds.cpu().numpy().flatten()
        weights = weights.cpu().numpy()
        
    backtest_df = pd.DataFrame({'date': dates_test, 'pred': preds, 'target': y_test})
    capital = CONFIG['initial_capital']
    curve = [capital]
    daily_returns = []
    
    unique_dates = np.sort(np.unique(dates_test))
    for date in unique_dates:
        daily = backtest_df[backtest_df['date'] == date]
        daily_ret = 0
        if len(daily) >= CONFIG['top_k']:
            daily_ret = daily.nlargest(CONFIG['top_k'], 'pred')['target'].mean() - 0.0003
        capital *= (1 + daily_ret)
        curve.append(capital)
        daily_returns.append(daily_ret)
    
    # 指标计算
    total_days = len(unique_dates)
    total_ret = (capital / CONFIG['initial_capital']) - 1
    annualized_ret = (1 + total_ret) ** (252 / total_days) - 1
    daily_returns = np.array(daily_returns)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    max_dd = calculate_max_drawdown(curve)
    win_rate = np.sum(daily_returns > 0) / total_days
    
    metrics = {"total_return": total_ret * 100, "annualized_return": annualized_ret * 100, "max_drawdown": max_dd * 100, "sharpe_ratio": sharpe, "win_rate": win_rate * 100, "final_capital": capital}
    
    print("\n" + "="*40)
    print(f"📊 {CONFIG['model_name']} 策略表现报告")
    print(f"年化收益率 : {metrics['annualized_return']:.2f}%")
    print(f"夏普比率   : {metrics['sharpe_ratio']:.4f}")
    print("="*40)

    # 保存
    report_dir = os.path.join(parent_dir, 'reports')
    if not os.path.exists(report_dir): os.makedirs(report_dir)
    save_report(metrics, CONFIG, os.path.join(report_dir, f'{CONFIG["model_name"]}_report.txt'))
    
    plt.figure(figsize=(10, 5))
    plt.plot(unique_dates, curve[1:], label=f'{CONFIG["model_name"]}', color='orange')
    plt.title(f'{CONFIG["model_name"]} Equity Curve (Ann. Ret: {metrics["annualized_return"]:.1f}%)')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(os.path.join(report_dir, f'{CONFIG["model_name"]}_equity.png')); plt.close()
    
    feature_names = [c.replace('feat_', '') for c in feature_cols]
    w_df = pd.DataFrame(weights, columns=feature_names)
    w_df['date'] = dates_test
    w_monthly = w_df.groupby(w_df['date'].astype(str).str[:7])[feature_names].mean()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(w_monthly.T, cmap="RdYlBu_r", annot=False)
    plt.title(f'{CONFIG["model_name"]} Dynamic Factor Weights')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f'{CONFIG["model_name"]}_weights_heatmap.png')); plt.close()
    
    print(f"✅ 结果已保存至 {report_dir}")

if __name__ == "__main__":
    run_bilstm_strategy()