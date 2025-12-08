import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- 路径设置 ---
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

# ================= 1. 超参数配置 (Hyperparameters) =================
CONFIG = {
    "model_name": "RNN",
    "data_path": os.path.join(parent_dir, "data", "market_data.csv"),
    "report_path": os.path.join(parent_dir, "reports", "all_reports.json"),
    "initial_capital": 1000000,
    "top_k": 5,
    "seq_len": 20,       # 时间窗口
    "batch_size": 128,
    "epochs": 15,
    "lr": 0.001,
    "hidden_dim": 64,
    "num_layers": 2,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# ================= 2. 模型定义 =================
class ExplainableRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ExplainableRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 权重生成头 (Weight Generator)
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        out, _ = self.rnn(x)
        context = out[:, -1, :] 
        weights = self.weight_generator(context)
        current_factors = x[:, -1, :]
        prediction = (current_factors * weights).sum(dim=1, keepdim=True)
        return prediction, weights

# ================= 3. 辅助函数 =================
def calculate_max_drawdown(equity_curve):
    """计算最大回撤"""
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return drawdown.max()

def create_sequences(data, target, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        y = target[i + seq_len - 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def save_report(metrics, params, save_path):
    """保存详细文本报告"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"====== {params['model_name']} Strategy Report ======\n\n")
        f.write("[1] Model Hyperparameters:\n")
        for k, v in params.items():
            if "path" not in k: # 不打印路径，保持整洁
                f.write(f"  {k:<15}: {v}\n")
        
        f.write("\n[2] Performance Metrics:\n")
        f.write(f"  Total Return   : {metrics['total_return']:.2f}%\n")
        f.write(f"  Annualized Ret : {metrics['annualized_return']:.2f}%\n")
        f.write(f"  Max Drawdown   : {metrics['max_drawdown']:.2f}%\n")
        f.write(f"  Sharpe Ratio   : {metrics['sharpe_ratio']:.4f}\n")
        f.write(f"  Win Rate       : {metrics['win_rate']:.2f}%\n")
        f.write(f"  Final Capital  : {metrics['final_capital']:.2f}\n")
        
        f.write("\n[3] Description:\n")
        f.write("  This model uses an explainable architecture where a gating network\n")
        f.write("  dynamically assigns weights to alpha factors based on market context.\n")

# ================= 4. 主流程 =================
def run_rnn_strategy():
    print(f"🚀 启动 {CONFIG['model_name']} 策略训练 (Device: {CONFIG['device']})...")
    
    # --- 数据加载 ---
    if not os.path.exists(CONFIG['data_path']): return
    df = pd.read_csv(CONFIG['data_path'])
    df['date'] = pd.to_datetime(df['date'])
    
    with open(CONFIG['report_path'], 'r', encoding='utf-8') as f:
        alpha_config = [a for a in json.load(f) if 'error' not in a][:10]
    
    ctx = AlphaContext(df)
    env = {'CLOSE': ctx.CLOSE(), 'OPEN': ctx.OPEN(), 'VOLUME': ctx.VOLUME(), 'HIGH': ctx.HIGH(), 'LOW': ctx.LOW(), 'DELAY': ctx.DELAY, 'MA': ctx.MA, 'STD': ctx.STD, 'TS_MAX': ctx.TS_MAX, 'TS_MIN': ctx.TS_MIN, 'CORR': ctx.CORR, 'RANK': ctx.RANK}
    
    feature_cols = []
    for alpha in alpha_config:
        try: 
            col = f"feat_{alpha['name']}"
            df[col] = eval(alpha['formula'], {}, env)
            feature_cols.append(col)
        except: pass
        
    df['target'] = df.groupby('code')['close'].shift(-1) / df['close'] - 1
    df = df.dropna().sort_values(['code', 'date']).reset_index(drop=True)
    
    split_date = pd.Timestamp("2022-10-26")
    train_raw = df[df['date'] < split_date]
    test_raw = df[df['date'] >= split_date]
    
    scaler = StandardScaler()
    scaler.fit(train_raw[feature_cols])
    
    def process_by_code(sub_df):
        sub_df[feature_cols] = scaler.transform(sub_df[feature_cols])
        X = sub_df[feature_cols].values
        y = sub_df['target'].values
        dates = sub_df['date'].values
        if len(X) <= CONFIG['seq_len']: return None, None, None
        X_seq, y_seq = create_sequences(X, y, CONFIG['seq_len'])
        d_seq = dates[CONFIG['seq_len']-1:]
        min_len = min(len(X_seq), len(y_seq), len(d_seq))
        return X_seq[:min_len], y_seq[:min_len], d_seq[:min_len]

    def build_dataset(raw_df):
        all_X, all_y, all_dates = [], [], []
        for code, sub in raw_df.groupby('code'):
            x, y, d = process_by_code(sub.copy())
            if x is not None: 
                all_X.append(x); all_y.append(y); all_dates.append(d)
        if not all_X: return np.array([]), np.array([]), np.array([])
        return np.concatenate(all_X), np.concatenate(all_y), np.concatenate(all_dates)

    print("构建时序数据...")
    X_train, y_train, _ = build_dataset(train_raw)
    X_test, y_test, dates_test = build_dataset(test_raw)
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=CONFIG['batch_size'], shuffle=True)
    
    # --- 训练 ---
    model = ExplainableRNN(len(feature_cols), CONFIG['hidden_dim'], CONFIG['num_layers']).to(CONFIG['device'])
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

    # --- 回测与评估 ---
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test).to(CONFIG['device'])
        preds, weights = model(test_tensor)
        preds = preds.cpu().numpy().flatten()
        weights = weights.cpu().numpy()
        
    backtest_df = pd.DataFrame({'date': dates_test, 'pred': preds, 'target': y_test})
    
    # 策略逻辑：Top-K 轮动
    capital = CONFIG['initial_capital']
    curve = [capital]
    daily_returns = []
    
    unique_dates = np.sort(np.unique(dates_test))
    for date in unique_dates:
        daily = backtest_df[backtest_df['date'] == date]
        daily_ret = 0
        if len(daily) >= CONFIG['top_k']:
            # 买入预测收益最高的 Top K
            selected = daily.nlargest(CONFIG['top_k'], 'pred')
            # 扣除手续费 0.0003
            raw_ret = selected['target'].mean()
            daily_ret = raw_ret - 0.0003
            
        capital *= (1 + daily_ret)
        curve.append(capital)
        daily_returns.append(daily_ret)
    
    # --- 计算详细指标 ---
    total_days = len(unique_dates)
    total_ret = (capital / CONFIG['initial_capital']) - 1
    
    # 年化收益 (假设一年252个交易日)
    annualized_ret = (1 + total_ret) ** (252 / total_days) - 1
    
    # 夏普比率 (无风险利率假设为0)
    daily_returns = np.array(daily_returns)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    # 最大回撤
    max_dd = calculate_max_drawdown(curve)
    
    # 胜率 (日收益 > 0 的天数占比)
    win_rate = np.sum(daily_returns > 0) / total_days
    
    metrics = {
        "total_return": total_ret * 100,
        "annualized_return": annualized_ret * 100,
        "max_drawdown": max_dd * 100,
        "sharpe_ratio": sharpe,
        "win_rate": win_rate * 100,
        "final_capital": capital
    }
    
    print("\n" + "="*40)
    print(f"📊 {CONFIG['model_name']} 策略表现报告")
    print(f"总收益率   : {metrics['total_return']:.2f}%")
    print(f"年化收益率 : {metrics['annualized_return']:.2f}%")
    print(f"最大回撤   : {metrics['max_drawdown']:.2f}%")
    print(f"夏普比率   : {metrics['sharpe_ratio']:.4f}")
    print("="*40)

    # --- 保存结果 ---
    report_dir = os.path.join(parent_dir, 'reports')
    if not os.path.exists(report_dir): os.makedirs(report_dir)
    
    # 1. 保存文本报告
    save_report(metrics, CONFIG, os.path.join(report_dir, f'{CONFIG["model_name"]}_report.txt'))
    
    # 2. 绘制资金曲线
    plt.figure(figsize=(10, 5))
    # 注意 curve 比 unique_dates 多一个初始点，这里为了画图对齐去掉第一个
    plt.plot(unique_dates, curve[1:], label=f'{CONFIG["model_name"]} Strategy', color='blue')
    plt.title(f'{CONFIG["model_name"]} Equity Curve (Ann. Ret: {metrics["annualized_return"]:.1f}%)')
    plt.xlabel('Date')
    plt.ylabel('Equity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(report_dir, f'{CONFIG["model_name"]}_equity.png'))
    plt.close()
    
    # 3. 绘制热力图 (修复版)
    feature_names = [c.replace('feat_', '') for c in feature_cols]
    w_df = pd.DataFrame(weights, columns=feature_names)
    w_df['date'] = dates_test
    
    # 按月聚合，显式指定 numeric_only=True 或只选因子列
    w_monthly = w_df.groupby(w_df['date'].astype(str).str[:7])[feature_names].mean()
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(w_monthly.T, cmap="RdYlBu_r", annot=False, cbar_kws={'label': 'Attention Weight'})
    plt.title(f'{CONFIG["model_name"]} Dynamic Factor Attention (Monthly Avg)')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f'{CONFIG["model_name"]}_weights_heatmap.png'))
    plt.close()
    
    print(f"✅ 详细报告与图表已保存至 {report_dir}")

if __name__ == "__main__":
    run_rnn_strategy()