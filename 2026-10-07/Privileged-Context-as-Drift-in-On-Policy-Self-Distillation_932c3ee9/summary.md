---
title: "Privileged-Context-as-Drift-in-On-Policy-Self-Distillation"
source: https://arxiv.org/pdf/2610.07842v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:14:26"
---

# 论文速读：Privileged-Context-as-Drift-in-On-Policy-Self-Distillation

## 一句话总结
本文系统解耦了 on-policy self-distillation (OPSD) 中特权上下文（privileged context）的**内容**与**生成来源**对策略漂移的独立影响，发现内容选择对分布偏移与参数更新方向的作用显著大于来源选择；研究主张在持续学习设定下，应将 OPSD 的评估拆分为目标任务获取、先前任务保持与分布漂移三个独立维度，避免仅凭目标准确率选择上下文。

## 研究问题与动机
- **核心问题**：OPSD 通过冻结教师模型并条件化于特权上下文来引导学生学习，但现有工作对特权上下文的形式与生成方式差异巨大，难以剥离具体设计变量对策略漂移的贡献。
- **持续学习痛点**：Shenfeld et al. (2025) 表明策略漂移（以 KL 散度衡量）强烈预测先前能力的灾难性遗忘，控制后训练阶段的非必要漂移是关键目标。
- **现有方法不足**：既有 OPSD 变体往往同时改变上下文内容、来源、基座模型、数据与优化预算，导致实验结论不可复现或不可归因。
- **研究动机**：在固定模型、目标、预算与评估流程的前提下，构建正交实验矩阵，明确“上下文内容×生成来源”如何决定策略的移动幅度与方向。

## 核心贡献（创新点）
1. **首次构建 3×3 内容×来源正交实验矩阵**：固定 Qwen2.5-7B-Instruct 与 OPSD 训练流程，系统隔离 demonstration/feedback/rephrase 与 external/self+verifier/self 的独立效应，填补上下文设计归因的空白。
2. **量化证实内容对漂移的主导作用**：维持来源固定时改变内容产生的中位 KL 跨度是维持内容固定改变来源的 5.1 倍（per-token）与 2.2 倍（per-sequence），LoRA 更新方向余弦相似度同样呈现该趋势。
3. **揭示获取-保持-漂移的指标解耦现象**：目标任务准确率、先前任务保持率与分布漂移三者排序高度不一致，证明单一性能指标无法同时表征能力获取与稳定性。
4. **提出 OPSD 持续学习的分层评估准则**：主张将特权上下文视为稳定性设计组件，要求在部署前分别报告获取增量、遗忘程度与分布偏移，而非仅以目标准确率做选择。

## 方法详解
- **OPSD 训练目标**：学生模型当前策略为 π_θ，冻结教师为 π_{θ_0}。对查询 x_i 与固定特权上下文 c_i，采样学生 rollout prefix y_i^{<t}，教师与学生分别评分同一前缀。优化 token 归一化的完整词表前向 KL：
  $$\widehat{\mathcal{L}}(\theta) = \frac{1}{\sum_{i \in B} T_i} \sum_{i \in B} \sum_{t=1}^{T_i} \sum_{v \in V} q_t^i(v) \log \frac{q_t^i(v)}{p_{\theta,t}^i(v)}$$
  仅更新学生 LoRA 参数，教师始终保持 θ_0 并 detach。
- **特权上下文矩阵（3×3）**：
  - **内容**：Demonstration（直接给出解答）、Feedback（对固定错误尝试的批注与修正）、Rephrase（无答案的题面重写）。
  - **来源**：External（Qwen3.6-27B 生成，temperature=0.7，关闭 thinking mode）；Self + Verifier（冻结 Qwen2.5-7B-Instruct 生成，经确定性验证器过滤后按最高平均 token log-prob 选取）；Self (no verifier)（同模型生成，按多数投票聚类选取最高置信度解）。Gold answer 仅用于验证器判定，永不进入 prompt。
- **数据集与查询筛选**：ProofWriter OWA D5（depths 4–5）、MuSiQue answerable、Big-Math hard subsets，各 3,200 train / 200 dev / 1,000 test。查询按九种条件下构造成功率降序组成主集，不足部分从备用池补齐，失败样本（空/截断/解析失败/校验拒绝）替换。
- **评估协议**：
  - **目标任务准确率**：greedy decoding on 1,000 test，任务专用验证器（exact match / symbolic equivalence / normalized alias）。
  - **先前任务保持**：MMLU / HellaSwag / TruthfulQA-mc2 / IFEval / HumanEval，计算 Δ_acc 均值。
  - **分布漂移**：held-out prompts 上贪婪轨迹的反向 KL，分 per-token（nats）与 per-sequence（nats）报告。
  - **参数几何**：LoRA 更新 ΔW = (α/r)BA 的全局 Frobenius 范数、有效秩 exp(H(p))、两两条件间余弦相似度。
- **优化超参**：LoRA rank=16, α=32, dropout=0；AdamW lr=1e-5, weight decay=0.01, grad clip=1.0, effective batch=16, microbatch=1, BF16, non-reentrant gradient checkpointing；861 steps，共 13,776 次 on-policy rollout。

## 实验与结果
- **漂移度量**：Per-token KL 范围从 Big-Math rephrase 的 0.02 nats 至 ProofWriter demonstration 的 4.10 nats；per-sequence KL 范围从 8.1 至 40.4 nats。维持来源固定时跨内容的中位最大/最小 KL 比为 7.2（per-token）与 2.80（per-sequence）；维持内容固定时跨源比为 1.4（per-token）与 1.25（per-sequence）。
- **参数更新几何**：同源异内容的平均余弦为 0.255，同源同内容异源的为 0.571；跨数据集随机配对为 0.034。两种自生成来源更新最接近（0.662），External 常偏离两者；Self+Verifier 在 8/9 组中位于 External 与 Self 之间。
- **目标任务准确率**：ProofWriter demonstration 获最大增益（External +49.5 点，base 42.4%→91.9%）；Rephrase 在 ProofWriter 上普遍为负增益；Big-Math 上 8/9 条件的 95% 配对 bootstrap CI 包含 0，External demonstration 反而下降 7.7 点。
- **先前任务保持**：多数条件呈小幅下降，最大平均降幅 ≤3.0 点；IFEval 与 HumanEval 降幅最大（-8.3 与 -6.1 点）；MMLU 变化极小（
