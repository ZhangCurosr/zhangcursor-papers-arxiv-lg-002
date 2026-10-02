---
title: "ON-POLICY-OR-OFF-POLICY-LEARNING-A-SYSTEMATIC-STUDY-OF-DISTI"
source: https://arxiv.org/pdf/2609.35259v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:37:01"
field: "大语言模型后训练与知识蒸馏"
keywords: ["on-policy distillation", "off-policy distillation", "knowledge distillation", "catastrophic forgetting", "KL divergence", "reinforcement learning"]
innovations: ["在控制实验中解耦 rollout 策略、KL 方向与学习率，发现 on-policy rollout 在最终准确率和遗忘上无一致优势", "从梯度理论证明 forward KL 对 rollout 变化具有 Lipschitz 稳定性而 reverse KL 无分布无关稳定性界", "精确定位 on-policy 的增益场景：更难任务的泛化和抑制偶发 teacher 风格迁移"]
benchmarks: ["Countdown-3", "Countdown-4E", "MedReason", "Science", "MATH-500", "MMLU-Pro", "TruthfulQA", "IFEval", "HumanEval-Instruct"]
---

# 论文速读：ON-POLICY OR OFF-POLICY LEARNING? A SYSTEMATIC STUDY OF DISTILLATION DYNAMICS

## 一句话总结
本文在强→弱蒸馏设定下系统性地解耦了 rollout 策略、KL 方向与学习率的影响，发现 **KL 方向**比 rollout 策略更能决定任务性能，**学习率**主导灾难性遗忘与参数更新稀疏度；on-policy 数据仅在泛化到更难任务变体时提供稳定增益。

## 研究问题与动机
- **现有研究混杂多因素**：SFT vs RLVR 的比较同时改变了目标函数、监督来源、监督密度和优化流程，导致无法归因于 rollout 策略本身。
- **on-policy 优势尚存争议**：已有工作声称 on-policy 学习可减少灾难性遗忘、产生更稀疏的参数更新并提升泛化能力，但这些结论来自公开 checkpoint 分析或多因素对比。
- **计算成本差异显著**：on-policy 方法需训练过程中持续采样，而 off-policy 方法可复用已有数据；若其优势来自其他设计选择，则额外成本可能不必要。
- **理论机制未明**：forward KL 与 off-policy、reverse KL 与 on-policy 的"传统配对"是否有优化意义上的必然性尚待验证。

## 核心贡献（创新点）
1. **解耦设计的受控蒸馏对比实验**：独立变化 rollout 策略、KL 方向和学习率，发现在最终准确率、灾难性遗忘和参数更新稀疏度上 on-policy 无一致优势——与已有 SFT vs RLVR 结论存在本质差异，因后者混杂多变量。
2. **目标函数依赖的 rollout 敏感性理论分析**：通过 token-level KL 梯度推导 + 连续 student–teacher 插值谱实验，证明 forward KL 对 rollout 变化具有稳定性界（Lipschitz 连续），而 reverse KL 无分布无关的稳定性保证；这解释了为何 reverse KL 更依赖 on-policy 数据。
3. **on-policy 增益的精确定位**：发现 on-policy rollout 仅在两类场景下带来稳定收益——（i）泛化到更难的 Countdown 算术变体（pass@k 提升 10–15%），（ii）与 reverse KL 结合时可抑制偶发的 teacher 风格迁移；但此泛化优势在后续 RLVR 中不 reliably 持续。

## 方法详解
- **蒸馏目标**：$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{data}} \mathbb{E}_{y \sim \rho(\cdot|x)}[\frac{1}{L_y}\sum_n \mathcal{D}(\pi_S^\theta(\cdot|h_n) \| \pi_T(\cdot|h_n))]$，其中 $\rho$ 为 rollout 策略，$\mathcal{D}$ 为 token-level KL 散度。
- **Full-vocabulary KL**：在完整词表 $\mathcal{V}$ 上精确计算 KL，而非采样估计，以避免额外方差。
- **Forward KL**：$D_{F-KL}(\pi_S, \pi_T) = \mathbb{E}_{y_n \sim \pi_T}[\log \frac{\pi_T(y_n)}{\pi_S(y_n)}]$——mode-covering，对教师高概率 token 施加更强惩罚。
- **Reverse KL**：$D_{R-KL}(\pi_S, \pi_T) = \mathbb{E}_{y_n \sim \pi_S}[\log \frac{\pi_S(y_n)}{\pi_T(y_n)}]$——mode-seeking，对学生偏好的 token 加权。
- **梯度不对称性**：$\nabla_{z_S} D_{F-KL} = \pi_S - \pi_T$（有界），$\nabla_{z_S} D_{R-KL} = \pi_S \odot [\log(\pi_S/\pi_T) - D_{R-KL}]$（无界，依赖 likelihood ratio）。
- **Rollout 策略谱**：$\pi_\lambda(v) = \text{softmax}(\tilde{z}_\lambda)_v$，其中 $(\hat{z}_\lambda)_v = \frac{1+\lambda}{2}\log\pi_S(v) + \frac{1-\lambda}{2}\log\pi_T(v)$；$\lambda=-1$ 为 OffPD，$\lambda=1$ 为 OnPD，$\lambda=0$ 为对称中点。
- **实验设置**：Llama-3.1-8B → Llama-3.2-1B（及 Qwen2.5-7B → Qwen2.5-1.5B）；任务包括 MedReason（医学）、Science（科学）、Countdown（算术）；150 步全参数 fine-tuning；学习率 $\{1\times10^{-5}, 5\times10^{-5}\}$；3 个随机种子。

## 实验与结果
- **最终准确率**：OnPD 和 OffPD 最佳均值均约 72–73%；Forward KL 稳定在 71–73%，Reverse KL 波动大（35–72%），对学习率更敏感。
- **灾难性遗忘**（7 个 OOD benchmark 均值）：学习率 $1\times10^{-5}$ 时遗忘最多 1.3pp；$5\times10^{-5}$ 时下降 11.2–14.0pp；**学习率是主导因素**，rollout 策略影响微小。
- **参数更新稀疏度**（$\tau=10^{-6}$）：低学习率下 85.3–89.6%，高学习率下 51.9–60.0%；OffPD 在各匹配比较中至少与 OnPD 一样稀疏。
- **Rollout 谱实验（Countdown-3）**：Forward KL 在 $\lambda \in [-2, 2]$ 全程保持 >80% 准确率（仅变 5.2pp）；Reverse KL 对 $\lambda$ 高度敏感，学生偏好 rollout（$\lambda>0$）表现显著更好。
- **输出覆盖率（pass@k）**：Forward KL 在 Countdown-3 上 pass@10 增益显著高于 reverse KL。
- **泛化到更难题（Countdown-4E）**：两种 KL 下 on-policy 数据（$\lambda>0$）均稳定提升 10–15% pass@k。
- **后续 RLVR（Countdown-4）**：Forward-KL OffPD 初始点（~0%）最终达到最高 reward；on-policy 初始优势在 300 步 RLVR 后不 persist。
- **Teacher 风格迁移**：OnPD + Reverse KL 显著抑制 Spanish 风格迁移（其余配置几乎完全迁移）。
- **鲁棒性验证**：去除 gradient clipping、使用 sampled KL 估计器、在长链任务（Numina-MATH，平均 622 token）上重复实验，结论一致。

## 相关工作脉络
1. **OnPD 基础方法**（Agarwal et al., 2023）：提出 on-policy 蒸馏框架，学生自生成轨迹并请教师做 token-level 监督；本文将其与 OffPD 在相同 KL 和学习率下直接对比。
2. **SFT vs RLVR 对比研究**（Chu et al., 2025; Shenfeld et al., 2025; Mukherjee et al., 2025）：声称 on-policy 在遗忘和稀疏性上占优；本文指出这些结论混杂了目标/奖励/学习率等变量，需在控制实验中重新检验。
3. **学习率与遗忘关系**（Catalan-Tatjer & Geiping; Rofin et al., 2026）：学习率是决定 forgetting-sparsity trade-off 的关键因素；本文实验进一步验证此结论在蒸馏设定中同样成立。
4. **反向 KL 不稳定性研究**（Tang & Munos, 2025; Li et al., 2026b）：指出 reverse KL 梯度估计的方差问题；本文从理论上给出了无分布无关稳定性界的严格证明（Theorem C.2）。
5. **Sequence-level KL 分解**（Appendix C.5）：展示 teacher rollout + forward KL 和 student rollout + reverse KL 恰好是序列级 KL 的两种链式法则分解，但强调这仅是目标值层面的恒等式，不等于优化性质更优。
6. **Style transfer / incidental transfer 研究**：本文首次系统比较不同 rollout×KL 组合对偶发 teacher 风格迁移的影响，发现 OnPD+R-KL 可抑制此现象。

## 局限性与未来方向
- **模型规模受限**：学生模型最大 1.5B 参数，推理轨迹最长约 2000 token；更大模型上的结论尚未验证。
- **Teacher 固定**：所有实验中 teacher 在蒸馏阶段冻结；不同 student–teacher 组合是否产生 OnPD/OffPD 差异尚未探究。
- **长链推理仅初步探索**：Numina-MATH 实验（每个条件仅 1 次运行）暗示 on-policy 在更长推理链中可能有优势，但统计效力不足，需更多重复实验确认。
- **后续 RLVR 优势不持久**：on-policy 的泛化优势在后续强化学习阶段消失，如何组合多步 post-training pipeline 仍是开放问题。

## 研究启发与可借鉴点
1. **解耦实验设计范式**：本文通过保持训练管线固定、仅变化单一因子（rollout / KL / LR）的方式分离因果影响，可作为后续 post-training 方法对比的标准范式。
2. **Rollout 策略谱（$\lambda$-spectrum）**：引入连续插值参数 $\lambda$ 在 student–teacher 之间平滑过渡，配合 log-ratio clipping 和 plausibility masking，为系统性探索 policy 来源影响提供了可复用的实验工具。
3. **梯度分析指导实践选择**：forward KL 的 Lipschitz 稳定性（Theorem C.1）意味着对 rollout 选择宽容，适合 off-policy 低成本场景；reverse KL 的敏感性则提示需配合 gradient clipping 或限制 likelihood ratio 范围。
4. **评估维度扩展**：除 pass@1 外，本文系统报告了 pass@k、OOD 遗忘、update sparsity、风格迁移和后续 RLVR 表现，建议后续研究采用类似的多元评估框架。
5. **OnPD + Reverse KL 抑制风格迁移**：这一发现为多语言/多风格场景下的蒸馏提供了实用技巧——组合 on-policy 采样与 reverse KL 可保留学生原有输出风格。

## 关键术语表
**On-Policy Distillation (OnPD)**：学生模型从自身当前策略采样轨迹，并在这些轨迹上接受教师的 token-level KL 监督。
**Off-Policy Distillation (OffPD)**：学生模型在教师生成的固定轨迹上进行蒸馏学习，训练数据与当前学生策略无关。
**Forward KL**：$D_{KL}(\pi_T \|\pi_S)$，以教师分布为期望，要求学生在所有教师高概率 token 上覆盖足够概率（mode-covering）。
**Reverse KL**：$D_{KL}(\pi_S \|\pi_T)$，以学生分布为期望，鼓励学生集中在教师也支持的 token 上（mode-seeking）。
**Catastrophic Forgetting**：模型在新任务上训练后，对先前学到的出分布（OOD）能力出现显著衰退。
**Parameter-Update Sparsity**：训练前后参数变化绝对值小于阈值 $\tau$ 的参数占总参数的比例，越高表示更新越稀疏。
**Rollout-Policy Spectrum ($\lambda$)**：通过 likelihood-ratio 控制的插值策略 $\pi_\lambda$，$\lambda=-1$ 对应纯教师 rollout，$\lambda=1$ 对应纯学生 rollout。
**RLVR**：Reinforcement Learning with Verifiable Rewards，利用可验证奖励信号对模型进行强化学习微调。

## 可复现要素
- **数据集**：Countdown-3（程序生成 4163 train / 426 eval）、Science（2674 / 507）、MedReason（4748 / 471）、Numina-MATH（20K 合成）；OOD 评估用 MMLU-Pro、TruthfulQA、IFEval、HumanEval-Instruct、EQ-Bench、BBQ、ToxiGen。**代码与数据开源情况**：论文未明确声明代码仓库链接，但提供了完整的 Appendix 实验细节；Checkpoint 来自标准 HuggingFace 模型（Llama-3.2-1B-Instruct、Qwen2.5-1.5B-Instruct 等）。
- **关键超参**：优化步数 150；学习率 $\{1\times10^{-5}, 5\times10^{-5}\}$；有效 batch size 32；Fused AdamW（$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$）；gradient clipping norm=1.0；warmup 10 步；Student 采样温度 1.0；Teacher 温度 0.5；top-p=1.0。
- **代码/权重**：论文未提及外部代码库；使用了 vLLM 进行推理， Unsloth + TRL 的 GRPO trainer 用于 RLVR 阶段。
