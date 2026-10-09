---
title: "HAN-Mamba-Hierarchical-Selective-State-Space-Networks-for-Mu"
source: https://arxiv.org/pdf/2610.10323v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 02:50:58"
field: "金融时间序列预测"
keywords: ["Volatility Forecasting", "Mamba", "State Space Models", "Hierarchical Architecture", "High-Frequency Finance", "Time Series", "Selective SSM"]
innovations: ["用选择性状态空间编码器替换层次Transformer编码器，仅保留注意力于三尺度摘要融合", "论证集合vs序列融合结构，验证注意力对无序跨尺度整合的必要性与顺序敏感性", "利用线性复杂度将高频上下文从60扩展至240桶实现误差进一步降低与流式常数时间更新"]
benchmarks: ["Optiver Realized Volatility Prediction (ORVP)"]
---

# 论文速读：HAN-Mamba: Hierarchical Selective State Space Networks for Multi-Scale Financial Volatility Forecasting

## 一句话总结
本文提出 HAN-Mamba，将层次波动率预测架构 HAN-T 中的 Transformer 编码器替换为 Mamba 选择性状态空间编码器，仅在三层摘要融合阶段保留注意力机制；在 ORVP 基准上以更少的参数实现更高精度，并利用线性复杂度将高频上下文从 60 扩展至 240 桶。

## 研究问题与动机
- 短周期实现波动率预测需整合从秒级订单簿动态到周级制度漂移的多时间尺度信息，单一分辨率模型难以捕捉波动率聚类、慢衰减自相关与流动性冲击的混合动力学。
- 先前的 HAN-T 用三尺度不对称 Transformer 编码器处理序列，但自注意力的二次复杂度将高频上下文硬性限制在 60 个桶，且每个新预测需全窗重算，无法支持流式部署。
- 注意力在全对交互结构上的开销在输出单个摘要向量时被大幅浪费，其归纳偏置与波动率"持久但衰减的记忆 + 突变制度切换"的结构并不完全匹配。
- 选择性状态空间模型（Mamba）的输入依赖门控可实现自适应遗忘（平静期保留记忆、冲击期快速重置），且训练与推理均为线性/常数复杂度，更适合此类长程时间序列编码。

## 核心贡献（创新点）
1. **提出 HAN-Mamba 层次混合架构**：用选择性 SSM 编码器（最终状态读取）替换 HAN-T 的 Transformer 编码器，仅在三个尺度摘要的融合器中保留注意力；与已有工作的本质区别在于按"数据结构的算子分配"原则而非简单替换 backbone，使归纳偏置精确对齐波动率动力学。
2. **集合 vs 序列的跨尺度融合理论分析**：论证三尺度摘要构成无序集合，注意力（permutation-equivariant）是正确融合算子，循环融合器会引入人为顺序偏差；本文通过四种融合设计的消融给出实证支撑。
3. **线性编码器解锁长上下文扩展**：将高频上下文从 60 扩展至 240 桶（约一周交易日），误差从 0.1942 降至 0.1927；注意力变体在同等扩展下饱和甚至退化，凸显线性复杂度带来的准确度 headroom。
4. **强化计量经济学定位**：将三层流层次结构显式对接 HAR 模型的异构时间尺度级联，并将选择性衰减解释为数据驱动的 forgetting schedule，使架构同时嵌入深度学习与金融时间序列两大文献脉络。

## 方法详解
- **特征流水线**（继承会议版）：微观结构提取（WAP、价差、不平衡度、贸易活动统计等）→ 关系构建（横截面 Canberra/Mahalanobis 邻居与纵向 Manhattan 邻居的聚合与排名统计）→ 最终变换（横截面排序、对数压缩、3 维 LDA 股票嵌入），每样本输出 604 维特征向量。
- **三尺度同步流**：短期流 $X_{\text{short}} \in \mathbb{R}^{L_s \times 604}$（最近 $L_s$ 个 10 分钟桶）、中期流 $X_{\text{mid}} \in \mathbb{R}^{20 \times 1}$（最近 20 日日度实现波动率）、长期流 $X_{\text{long}} \in \mathbb{R}^{12 \times 1}$（最近 12 个周度聚合）；$L_s \in \{60, 120, 240\}$。
- **选择性尺度编码器**：各流经线性嵌入后进入预归一化 Mamba 块堆栈，不对称容量：短期 4 块（$d=128, N=16$）、中期 2 块（$d=64, N=16$）、长期 1 块（$d=32, N=8$），展开因子 2，因果卷积宽 4。消除位置嵌入与分类 token，以最终隐状态 $h^{(L)}$ 作序列摘要。
- **选择性门控与填充处理**：步长 $\Delta_k$ 为输入 token 的函数；对全零填充输入编码器可驱动 $\Delta_k \to 0$，使 $\exp(\Delta_k A) \approx I$，状态无损穿越填充前缀，原生实现 null-token 跳过，无需显式 mask。
- **注意力融合器**：三个摘要投影至公共宽度 128 后堆叠为长度 3 序列，经 2 层 4 头 Transformer 编码器融合，输出 token 均值作为融合表示；无位置嵌入以保证 permutation-equivariance。
- **回归头**：LayerNorm + 两层 GELU MLP（128→64→32）输出标量 log 实现波动率。
- **损失函数**：RMSPE $ \mathcal{L}_{\text{RMSE}} = \left(\frac{1}{n}\sum_i \left((y_i - e^{\hat{y}_i^{\text{log}}})/y_i\right)^2\right)^{1/2} $，与基准官方指标一致。
- **训练协议**：AdamW（lr=1e-4, wd=1e-5），1 epoch 线性 warmup + 余弦退火，梯度范数裁剪 1.0，混合精度，batch size=4096，特征按 fold 标准化；时间感知五折 CV。

## 实验与结果
- **数据集**：Optiver Realized Volatility Prediction (ORVP) 公开高频基准，含股票订单簿快照与交易记录，10 分钟窗口组织。
- **评估协议**：时间感知五折交叉验证（按时间连续分块，防 lookahead leakage），报告跨 fold 均值与标准差。
- **主要结果**（$L_s=60$，mean RMSPE）：
  - **HAN-Mamba**：**0.1942 ± 0.0022**（层级最优）
  - HAN-T（会议版）：0.1965 ± 0.0025
  - Flat Mamba：0.1971 ± 0.0033
  - Flat Transformer：0.1989 ± 0.0041
  - LightGBM+Optuna：0.2076
  - GARCH(1,1)：0.2853
- **相对提升**：HAN-Mamba 较 HAN-T 相对改善 **1.2%**，参数减少 **33%**（0.96M vs 1.43M）。
- **上下文扩展**（Table 2）：HAN-Mamba 在 $L_s=240$ 达到 **0.1927 ± 0.0022**（累计相对 HAN-T 提升 1.9%）；HAN-T 在 $L_s=240$ 退化至 0.1963。
- **消融**（Table 3）：注意力融合器最优（0.1942）；Mamba 融合器对顺序敏感（long→mid→short: 0.1958，short→mid→long: 0.1963）；平均融合最差（0.1966）；全栈 Mamba 替换优于仅短流或仅中/长流替换。
- **计算效率**（Table 4，$L_s=240$）：epoch 时间 78s vs 152s（1.9×），峰值内存 8.7GB vs 18.9GB（2.2×）；推理流式更新 0.31ms vs 3.8ms（**12×** 加速）；早停平均 24 epoch vs 31 epoch。

## 相关工作脉络
- **HAR 模型（Corsi, 2009）**：固定权重线性级联合并日/周/月尺度；本文三层流是其非线性 learned 对应物，选择性衰减进一步泛化 HAR 的固定记忆权重。
- **GARCH 族（Engle, Bollerslev）**：条件方差建模基准；本文在实现波动率设定下与之对比，验证深度架构增益。
- **HAN-T（作者 ICAART 2026）**：层次 Transformer+注意力融合；本文保持协议与特征不变，仅替换编码器以 isolating 算子效应。
- **Mamba（Gu & Dao, 2024）**：选择性 SSM，输入依赖门控；本文首次将其用于高频实现波动率预测的层次多流融合设定。
- **TimeMachine（Ahamed & Cheng, 2024）**：四模块 Mamba 做长期多元预测；针对通用基准，未涉及多尺度层次融合。
- **DeepVol（Moreno-Pino & Zohren, 2024）**：膨胀因果卷积直接处理高频收益；单分辨率，未显式建模异构时间尺度。

## 局限性与未来方向
- 证据局限于单一基准、单一资产类别与固定 10 分钟 horizon，跨资产/跨 horizon 泛化未验证。
- 关系特征在全横截面计算，实盘部署需增量维护邻居统计，论文未讨论在线更新机制。
- Mamba 超参数沿用语料建模默认值仅作轻量调优，horizon-specific tuning 可能有进一步增益。
- 流式延迟优势假设特征向量在桶结束时可用，端到端延迟由特征流水线主导而非模型本身。
- 未来方向：多资产联合预测与横截面状态共享、回归头的分布与分位数扩展、关系特征的增量维护、双向或状态空间对偶性处理（当因果性非必须时）。

## 研究启发与可借鉴点
- **算子分配遵循数据结构**：长同质时间轴用循环/SSM，小异质集合用注意力；这一"按结构选算子"原则可迁移至多尺度时间序列、多模态融合等场景。
- **集合 vs 序列的融合诊断**：通过消融循环融合器的顺序敏感性，可为跨源摘要融合提供可解释的设计检验，避免隐性顺序偏差。
- **线性编码器带来上下文 scaling curve**：当编码复杂度为线性时，可系统性探索上下文长度与精度的单调/边际递减曲线，挖掘长历史信号价值。
- **选择性门控原生处理填充**：Mamba 的输入依赖步长可实现 null-token skip，无需额外 mask 设计，可迁移至含变长/补零序列的任务。
- **层次先验的算子无关性**：Flat Mamba 仍优于 Flat Transformer，层级差距在不同算子族间保持一致，说明多尺度分解具有独立于底层算子的价值。

## 关键术语表
- **Selective State Space Model (SSM)**：状态转移参数（$B, C, \Delta$）由当前输入决定的状态空间模型，打破时间不变性，实现输入依赖的自适应遗忘。
- **Mamba**：基于选择性 SSM 的序列建模架构，通过硬件感知并行扫描实现 $O(L)$ 训练复杂度，推理时每个新 token 仅需一次状态更新。
- **Realized Volatility (RV)**：通过对高频收益率平方求和再开方估算的日内实现波动率，作为隐含波动率的无噪事后代理变量。
- **HAR Model**：Corsi 提出的异构时间尺度回归模型，用日/周/月平均波动率的线性组合预测未来波动率，编码市场参与者异质时间偏好的结构化先验。
- **Final-State Readout**：从循环/状态空间编码器末时间步隐状态直接读取序列摘要，替代 Transformer 的分类 token 机制。
- **Permutation-Equivariant Fuser**：对输入集合排列不变的融合算子（如无位置嵌入的注意力），确保跨尺度整合不引入人为顺序偏差。
- **Time-Aware Cross-Validation**：按时间顺序划分连续块的交叉验证方案，防止未来信息泄露到训练集，适用于时间序列数据。
- **RMSPE（Root Mean Squared Percentage Error）**：百分比误差的均方根，基准官方评估指标，对正偏目标适配。

## 可复现要素
- **数据集**：ORVP（Optiver Realized Volatility Prediction）公开基准；论文引用会议版 [14] 作为完整特征工程与预处理参考。
- **代码/权重**：论文未明确声明代码或权重是否开源。
- **关键超参**：AdamW（lr=1e-4, wd=1e-5），1 epoch warmup + 余弦退火，梯度裁剪 1.0，batch=4096，混合精度；Mamba 配置（短 4 块 d=128 N=16，中 2 块 d=64 N=16，长 1 块 d=32 N=8，展开 2，卷积宽 4）；融合器为 2 层 4 头 Transformer。
- **训练硬件**：8× NVIDIA A100 40GB GPU。
