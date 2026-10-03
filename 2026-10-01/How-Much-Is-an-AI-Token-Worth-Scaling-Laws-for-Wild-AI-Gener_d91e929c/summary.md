---
title: "How-Much-Is-an-AI-Token-Worth-Scaling-Laws-for-Wild-AI-Gener"
source: https://arxiv.org/pdf/2609.40295v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 18:36:14"
field: "语言模型预训练数据策略"
keywords: ["scaling laws", "AI-generated text", "pretraining data", "wild AI text", "model collapse"]
innovations: ["分离收益/危害项的缩放定律，允许AI token价值可正可负", "首次测量野生AI文本对预训练的影响并量化计算代价", "发现质量过滤器系统性偏好AI文档（FineWeb 2.3×、DCLM 9.8×）"]
benchmarks: ["C4", "FineWeb 2022", "FineWeb 2026 human/AI split", "Paloma", "Cosmopedia"]
---

# 论文速读：How-Much-Is-an-AI-Token-Worth-Scaling-Laws-for-Wild-AI-Gener

## 一句话总结
论文研究了"野生AI生成文本"（来自多模型、面向人类读者、未经标注混入预训练语料的网络文本）对语言模型预训练的影响。作者预训练800个模型、拟合新缩放定律，发现AI文本仅对数据匮乏模型有益，超过Chinchilla最优预算后即转害。

## 研究问题与动机
- 核心问题：当前Web文本中AI生成比例快速上升（2026年6月达27.5%，8月达31.1%），模型已远超20 tokens/parameter的Chinchilla最优预算；预训练时是否应过滤、保留或混合AI文本？
- 现有方法不足：①模型崩塌研究训练模型递归使用自身输出，设定不同于现实；②合成数据研究添加精心设计的改写数据，而野生AI文本来自多模型、未标注、无目的性混入；③已有缩放定律（Chinchilla、重复数据定律、混合定律）无法预测AI token价值随预算翻转的现象。

## 核心贡献（创新点）
1. 首次系统测量野生AI文本对预训练的影响：提出"wild AI text"概念，区分其与合成数据、模型崩塌设定的本质差异。
2. 预训练800个模型（19.9M–973M参数）并构建WildAI数据集（83B token、带AI/主题/格式标注），揭示AI文本质量过滤器反而偏好AI内容的现象。
3. 提出新的缩放定律：分离饱和收益项与对数增长危害项，允许一个AI token的价值（以human token当量衡量）可正可负，且在无AI数据时退化为Chinchilla。
4. 量化计算代价：在2026年8月AI占比31.1%下，训练未过滤Web文本需1.6×于纯人类子集的算力；预测到2028年达3.0×。
5. 给出实践建议：过滤AI文本、优先重复人类文本而非添加AI文本、分别报告人类/AI验证损失（因混合验证集在95.5%有害运行中掩盖伤害）。

## 方法详解
- **实验设置**：使用nanochat架构，7个模型尺寸（19.9M–973M），变化TPPh（2.9–87.5）和AI/人类token比例r（0–64），共800个模型；评估集包括C4、FineWeb 2022/2026（分人类/AI子集）、Paloma、Cosmopedia。
- **缩放定律公式**：
  - 基础Chinchilla形式：L(N,D) = E + A/N^α + B/D^β
  - 新定律：L(N, D_H, D_A) = E + A/N^α + B/D_eff^β (1 + H)
  - 收益项（饱和有效数据）：D_eff = D_H (1 + η g(r)), g(r) = R*(1 - e^{-r/R*}), R* = K t^ρ, t = D_H/(20N)
  - 危害项（对数增长）：H = γ t^u n^v [log(1+r) - r/(1+r)]，其中n = N/10^8
- **拟合策略**：在726个模型（≤268M）上拟合，用Huber损失和Chinchilla初始值；在保留的477M/973M模型上评估配对误差（预测loss变化与观测变化的RMSE）。

## 实验与结果
- **数据集**：WildAI共96.04M文档、83.31B token；AI标注用EditLens初步筛选+Pangram 3.3.2二次校验（假阳性0.05%、假阴性1.99%）。
- **AI占比趋势**：2021年<0.1%，2024年6月10.1%，2025年6月16.1%，2026年6月27.5%，2026年8月31.1%；预测2027年底42.3%，2028年底50.7%。
- **质量过滤器偏差**：FineWeb保留AI文档2.3×（29.3% vs 12.8%），DCLM保留AI文档9.8×（14.5% vs 1.5%）。
- **主要结果**：
  - 新定律在C4上手持误差0.83×10^-3，优于Shukor联合定律的1.41×10^-3和Chinchilla的4.32×10^-3。
  - AI文本仅在TPPh < 10（数据匮乏）时降低人类文本loss；在20 TPPh（Chinchilla最优）时立即升高loss。
  - 重复人类文本优于添加AI文本：在20 TPPh下，增加1 epoch时重复人类文本比添加AI文本低2.3% loss，8 epoch时差距扩至6.3–6.4%。
  - 对AI目标文本（Cosmopedia），最优混合比例始终>90% AI。
  - 在22.3% AI占比的混合验证集上，95.5%的有害运行被报告为改进。

## 相关工作脉络
1. Chinchilla缩放定律（Hofmann et al., 2022）：将AI token视为与人类token无差别，无法预测价值翻转。
2. 重复数据定律（Muennighof et al., 2023; Qin et al., 2026）：假设重复token价值非负，但AI token可产生危害。
3. 过拟合惩罚定律（Lovelace et al., 2026）：使用幂律惩罚，而本文观察到AI危害呈对数增长。
4. 数据混合定律（Shukor et al., 2025; Jain et al., 2024）：将AI与人类视为两类混合域，但未分离收益/危害项，外推不稳定。
5. 模型崩塌研究（Shumailov et al., 2024; Gerstgrasser et al., 2024）：递归训练模型自身输出，不同于现实中的多模型野生文本。
6. 合成数据研究（Maini et al., 2024; Kang et al., 2025）：添加精心设计的改写数据，而野生AI文本无目的性混入。

## 局限性与未来方向
- 定律在≤268M模型上拟合，外推至973M；更大模型可能更易记忆弱LLM模式，或表现不同的任务协同。
- 仅测量next-token loss，未深入下游任务（CORE指标显示AI/Human token提升幅度相近，但写作风格漂移）。
- 仅研究英语Web文本；不同语言、领域可能有不同最佳混合比例。
- 未来方向：①过滤野生AI文本并添加定向合成数据；②发现AI文本有益的领域；③评估已存在于预训练语料中的AI文本对当前模型质量的影响；④训练能理解AI输入但不模仿其写作的模型（如标记AI文本或掩码其loss）。

## 研究启发与可借鉴点
1. 收益/危害分离框架可迁移至其他"第二类数据"（如多语言混合、跨领域数据）的缩放定律建模。
2. 配对误差评估方法（以人类-only控制为基准预测loss变化）隔离了绝对损失水平噪声，适合比较不同数据源边际价值。
3. 验证集设计建议：若预训练数据含多源混合，应分别报告各源验证损失，避免混合验证集掩盖有害趋势。
4. 质量过滤器审计方法：追踪各过滤阶段AI/人类文档存活率，可识别系统性偏差。
5. 与团队方向结合机会：若团队关注低资源语言或垂直领域，可探索AI文本在目标分布外的潜在收益（如教程格式占AI文本18.5% vs 人类6.2%）。

## 关键术语表
- **Wild AI text**：语言模型为人类读者撰写、自然出现在网络上、未经标注混入预训练语料的AI生成文本。
- **TPPh（tokens per human parameter）**：每个参数分配的人类token数，衡量数据充裕度。
- **Paired error**：预测loss变化与观测变化之间的RMSE，用于评估缩放定律对边际效应的预测能力。
- **Compute-Equivalent Gain（CEG）**：比较两种训练配方计算效率的指标，基于达到相同loss所需token数的比率。
- **Pangram 3.3.2**：论文使用的AI检测器，假阳性率0.05%、假阴性率1.99%。
- **Saturation scale（R*）**：AI文本收益饱和窗口，随人类预算和模型尺寸变化。
- **Harm penalty（H）**：对数增长的危害项，反映过量AI文本的边际伤害递减特征。
- **Cosmopedia**：纯AI生成文本评估集，用于测量模型对AI文本的拟合能力。

## 可复现要素
- 数据集：WildAI（83B token）已开源，含AI/主题/格式标注；FineWeb v1.4.0扩展至2026年6月。
- 代码/权重：所有800个模型checkpoint及代码已开源在https://github.com/pangramlabs/WildAI。
- 关键超参：nanochat架构（Karpathy, 2025），7种尺寸（19.9M–973M），TPPh范围2.9–87.5，r范围0–64，学习率warmup 40步、线性decay后65%。
- 评估协议：在保留的477M/973M模型（74个）上评分；726个模型用于拟合。
