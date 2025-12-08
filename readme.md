
# MSQuant: LLM-Driven Quantitative Investment System with Multimodal Perception & Explainable AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目简介 (Introduction)

**MSQuant** 是一个端到端的、基于大语言模型（LLM）与深度学习的智能量化投资系统。本项目复现并深度扩展了论文 *Automate Strategy Finding with LLM in Quant Investment* 的核心思想，旨在解决传统因子挖掘中存在的“过度拟合”与“黑盒”问题。

系统集成了以下前沿技术：
* **多模态感知 (Multimodal Perception)**: 利用视觉模型 (Qwen-VL) 解读 K 线图表，结合 RAG 技术检索实时财经新闻，构建动态的市场情境 (Market Context)。
* **情境感知型挖掘 (Context-Aware Mining)**: 不再盲目搜索公式，而是基于 AI 感知的市场状态（如“震荡下行”），定向生成适应当前风格的 Alpha 因子。
* **多智能体锦标赛 (Multi-Agent Tournament)**: 引入 LLM 裁判 (Judge Agent)，基于统计指标与逻辑适应性对因子进行多轮 PK 筛选。
* **可解释深度策略 (Explainable Deep Learning)**: 实现了带有 **权重生成器 (Weight Generator)** 的 RNN/BiLSTM/Transformer 模型，能够输出因子权重的动态热力图，打开深度学习的“黑盒”。

---

## 📂 项目结构 (Repository Structure)

```text
msquant/
├── data/                       # 数据存储目录
│   └── market_data.csv         # 历史行情数据 (Baostock源, 自动下载)
├── model/                      # 深度学习策略模型
│   ├── model_rnn.py            # 可解释 RNN 策略
│   ├── model_bilstm.py         # 可解释 BiLSTM 策略 (推荐)
│   └── model_transformer.py    # 可解释 Transformer 策略 (SOTA)
├── reports/                    # 实验结果与可视化输出
│   ├── all_reports.json        # 因子池详细指标
│   ├── mined_alphas_pro.json   # AI 挖掘出的因子公式
│   ├── *_report.txt            # 模型详细回测报告
│   ├── *_equity.png            # 策略资金曲线图
│   └── *_weights_heatmap.png   # 因子动态权重热力图
├── paper/                      # 参考文献与 Proposal
│   ├── AlphaForge.pdf
│   ├── proposal.pdf
│   └── ...
├── alpha_engine.py             # 核心因子计算引擎 (算子库: MA, RANK, CORR...)
├── alpha_miner.py              # 基础版因子挖掘 (仅数值)
├── alpha_miner_pro.py          # [Pro] 情境感知型因子挖掘 (多模态)
├── batch_processor.py          # 因子批量回测与指标计算
├── data_loader.py              # 数据下载器 (基于 Baostock)
├── llm_judge.py                # 基础 LLM 判别模块
├── multimodal_utils.py         # 多模态工具箱 (K线绘制 + 联网搜索 + 视觉感知)
├── portfolio.py                # 简易 Top-K 组合回测工具
├── tournament.py               # 基础锦标赛筛选
├── tournament_pro.py           # [Pro] 多模态锦标赛筛选
└── requirements.txt            # 项目依赖库
````

-----

## 🛠️ 环境安装 (Installation)

推荐使用 Conda 创建独立环境，并安装 Python 3.8 或以上版本。

1.  **克隆仓库**

    ```bash
    git clone [https://github.com/your_username/msquant.git](https://github.com/your_username/msquant.git)
    cd msquant
    ```

2.  **安装依赖**

    ```bash
    pip install -r requirements.txt
    ```

    *注：项目依赖包含 `torch`, `pandas`, `baostock`, `akshare`, `openai`, `mplfinance`, `duckduckgo-search` 等。*

-----

## 🚀 使用指南 (Quick Start Pipeline)

请按照以下顺序运行脚本，完成从数据准备到策略回测的全过程。

### Step 1: 数据准备 (Data Preparation)

从 Baostock 免费下载上证 50 成分股过去 10 年的后复权数据。

```bash
python data_loader.py
```

> **输出**: `./data/market_data.csv` (约 10 万行数据，包含 OHLCV)

### Step 2: 多模态因子挖掘 (Context-Aware Mining)

让 AI (Qwen-VL + Qwen-Plus) 先“看图读新闻”感知市场，再定向生成因子。
*(需在代码中配置您的 DashScope API Key)*

```bash
python alpha_miner_pro.py
```

> **输出**: `./reports/mined_alphas_pro.json` (包含生成的因子公式及其设计逻辑)

### Step 3: 因子批量评估 (Batch Evaluation)

计算因子池中所有因子的 IC (信息系数)、夏普比率、换手率等指标，生成详细报表。

```bash
python batch_processor.py
```

> **输出**: `./reports/all_reports.json` (因子性能总表)

### Step 4: 多智能体锦标赛 (Tournament Selection)

启动 LLM 裁判，结合当前市场情境描述，对候选因子进行两两 PK，筛选出“Alpha King”。

```bash
python tournament_pro.py
```

> **输出**: 筛选出的最佳因子组合。

### Step 5: 深度学习策略训练 (Deep Learning Strategy)

使用可解释神经网络学习因子的动态权重组合。模型会自动读取 Step 3 的结果，构建时间序列，训练并回测。

  * **推荐使用 BiLSTM 模型 (稳健性最佳)**:
    ```bash
    python model/model_bilstm.py
    ```
  * **或者尝试 Transformer 模型 (捕捉全局依赖)**:
    ```bash
    python model/model_transformer.py
    ```

> **输出 (保存在 `reports/`)**:
>
>   * `BiLSTM_equity.png`: 策略资金曲线 vs 市场基准
>   * `BiLSTM_weights_heatmap.png`: **核心亮点**，展示 AI 在不同时期如何调整因子权重。
>   * `BiLSTM_report.txt`: 包含年化收益、最大回撤等详细指标的文本报告。

-----

## 📊 实验结果示例 (Experimental Results)

基于 2014-2022 年训练集与 2023-2024 年样本外测试集的实测数据：

| 模型 (Model) | 年化收益 (Ann. Ret) | 最大回撤 (Max DD) | 夏普比率 (Sharpe) | 胜率 (Win Rate) |
| :--- | :--- | :--- | :--- | :--- |
| **BiLSTM** | **14.40%** | **30.14%** | **0.6568** | **48.14%** |
| Transformer| 4.47% | 39.08% | 0.2959 | 45.99% |
| RNN | 4.64% | 34.83% | 0.3002 | 46.38% |

*注：以上数据基于 Top-5 轮动策略，扣除万分之三手续费。*

### 可解释性展示 (Explainability)

生成的 `*_weights_heatmap.png` 图表清晰展示了策略的逻辑：

  * 在 **牛市** 阶段，模型会自动提高 **动量类 (Momentum)** 因子的权重（红色区域）。
  * 在 **震荡市/熊市** 阶段，模型会自动切换至 **反转类 (Reversion)** 和 **波动率 (Volatility)** 因子，实现风险控制。

-----

## 🧩 核心模块说明 (Modules)

### 1\. `alpha_engine.py`

量化计算的核心。实现了 Pandas 高效算子，包括：

  * `DELAY(x, n)`: 滞后项
  * `MA(x, n)`: 移动平均
  * `RANK(x)`: 截面排名 (Cross-sectional Rank)
  * `CORR(x, y, n)`: 滚动相关系数

### 2\. `multimodal_utils.py`

多模态感知的实现层。

  * **视觉**: 调用 `mplfinance` 生成 K 线图。
  * **文本**: 调用 `duckduckgo_search` 模拟 RAG 检索实时新闻。
  * **感知**: 调用 VL 大模型生成 `Market Context` 描述。

### 3\. `model/`

包含三个独立的深度学习策略脚本。所有模型均集成了：

  * **数据对齐**: 自动处理滑动窗口带来的长度不一致问题。
  * **早停机制**: 防止过拟合。
  * **自动绘图**: 训练结束后直接输出评估图表。

-----

## ⚠️ 注意事项

1.  **API Key**: 请确保在 `alpha_miner_pro.py`, `tournament_pro.py` 和 `multimodal_utils.py` 中填入有效的阿里云 DashScope API Key (用于调用 Qwen 系列模型)。
2.  **网络连接**: `duckduckgo-search` 可能需要网络代理支持。如果无法连接，`multimodal_utils.py` 会自动降级为仅使用技术面数据。
3.  **路径问题**: 运行 `model/` 下的脚本时，代码已内置路径修复逻辑 (`sys.path.append`)，可直接在项目根目录或子目录下运行。


<!-- end list -->

```
```