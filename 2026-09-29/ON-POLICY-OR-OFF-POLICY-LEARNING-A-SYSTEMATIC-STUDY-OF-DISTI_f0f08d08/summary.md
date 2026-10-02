---
title: "ON-POLICY-OR-OFF-POLICY-LEARNING-A-SYSTEMATIC-STUDY-OF-DISTI"
source: https://arxiv.org/pdf/2609.35259v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:37:00"
field: "大语言模型后训练与知识蒸馏"
keywords: ["on-policy distillation", "off-policy distillation", "knowledge distillation", "catastrophic forgetting", "KL divergence", "rollout policy", "LLM post-training", "strong-to-weak distillation"]
innovations: ["在控制变量下解耦rollout策略、KL方向和学习率，发现on-policy rollout无一致优势", "通过梯度分析和rollout谱实验证明forward KL对rollout变化高度稳健，reverse KL显著敏感", "刻画on-policy数据在泛化到更难任务变体和抑制偶发teacher风格迁移上的有效场景"]
benchmarks: ["MedReason", "Science (SciKnowEval)", "Countdown-3/4E", "MMLU-Pro", "TruthfulQA", "MATH-500 (Numina-MATH)"]
---

# 论文速读：ON-POLICY OR OFF-POLICY LEARNING? A SYSTEMATIC STUDY OF DISTILLATION DYNAMICS

## 一句话总结
本文在强到弱知识蒸馏的受控设定下，独立分离了rollout策略、token级KL方向和学习率三个因素，发现on-policy rollout本身并不能带来一致的性能优势；真正决定任务性能的是token级KL方向，而灾难性遗忘和参数更新稀疏性主要由学习率控制。

## 研究问题与动机
- **问题：** 现有文献将on-policy学习的好处（减少灾难性遗忘、产生更稀疏的参数更新、提升泛化能力）归因于rollout策略，但SFT vs RLVR的比较同时改变了目标函数、监督密度、优化过程等多个因素，因果归因不清。
- **动机1：** 如果这些好处实际来自其他设计选择，则on-policy蒸馏带来的持续生成计算成本可能是不必要的。
- **动机2：** on-policy蒸馏（OnPD）日益流行，但缺乏与代价更低的off-policy蒸馏（OffPD）在固定其他配置下的直接对比。
- **动机3：** 传统上forward KL常与off-policy配对、reverse KL常与on-policy配对，但两者概念上相互独立，值得系统解耦研究。

## 核心贡献（创新点）
- **受控解耦研究：** 在强到弱蒸馏框架内独立操控rollout策略、KL方向和.learning rate，发现on-policy rollout在最终准确率、灾难性遗忘和参数更新稀疏性上均无一致优势。与已有工作相比，本文打破了传统固定配对，将多个混杂因素逐一隔离。
- **目标依赖的rollout敏感度理论分析：** 通过推导token级KL的梯度表达式并建立理论界（Appendix C），证明forward KL的梯度对rollout策略变化具有 Lipschitz 稳定性，而reverse KL不存在无分布假设的rollout稳定性界。
- **刻画on-policy数据真正有效的场景：** 虽然训练稳定性和输出覆盖主要由KL方向驱动、学习率决定遗忘和稀疏性，但on-policy rollout在泛化到更难的Countdown变体（Countdown-4E）和抑制偶发的teacher风格迁移上确实有效。

## 方法详解
- **蒸馏目标：** 强到弱蒸馏的目标函数为
  $$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{data}} \mathbb{E}_{y \sim \rho(\cdot|x)}\left[\frac{1}{L_y}\sum_{n=1}^{L_y} \mathcal{D}\left(\pi_S^\theta(\cdot|x, y_{<n}) \| \pi_T(\cdot|x, y_{<n})\right)\right]$$
  其中$\rho$为rollout策略，$\mathcal{D}$为距离函数。OffPD设$\rho = \pi_T$，OnPD设$\rho = \pi_S^{\theta^*}$，采样过程中截断梯度。
- **Forward KL vs Reverse KL：** 两者在完整词表上计算，不依赖采样估计。Forward KL为mode-covering，Reverse KL为mode-seeking。梯度表达式（Appendix B）：
  - Forward KL梯度关于学生logit的偏导为$\pi_S^\theta(v) - \pi_T(v)$，有界于$[-1,1]$。
  - Reverse KL梯度关于学生logit的偏导为$\pi_S^\theta(v)[\log\frac{\pi_S^\theta(v)}{\pi_T(v)} - D_{R-KL}]$，可无界放大。
- **Rollout-policy谱（Student-Teacher Spectrum）：** 构造 likelihood-ratio-controlled 策略$\pi_\lambda$，通过超参数$\lambda \in [-2, 2]$连续插值/外推student-teacher分布，$\lambda=-1$为纯OffPD，$\lambda=1$为纯OnPD，$\lambda=0$为对称中点。使用log-ratio裁剪（$c = 2\log 3$）和plausibility masking（$\alpha=0.2$）防止极端tokens被过度放大。
- **评估指标：** ID准确率（MedReason/Science/Countdown-3）、灾难性遗忘（7个OOD基准的平均分下降）、参数更新稀疏性（$|\Delta\theta_i| < 10^{-6}$的参数占比）。

## 实验与结果
- **模型与数据：** Llama-3.1-8B→Llama-3.2-1B为主实验，Qwen2.5-7B→Qwen2.5-1.5B为补充实验；三任务：MedReason、Science、Countdown-3（均含独立task-specific teacher，teacher经SFT+GRPO训练）。
- **训练设置：** 全参数fine-tuning，150步优化，LR=$1\times10^{-5}$或$5\times10^{-5}$，每配置3个随机种子，梯度裁剪norm=1.0。
- **主要结果：**
  - **最终ID准确率：** OnPD和OffPD均值几乎相同（72% vs 73%），无一致优势（Figure 1左）。
  - **灾难性遗忘：** 低LR时OOD变化≤1.3pt，高LR时下降11.2–14.0pt；rollout策略差异远小于学习率影响（Figure 1中）。
  - **参数更新稀疏性：** 低LR时85.3–89.6%，高LR时51.9–60.0%；相同条件下OffPD至少与OnPD一样稀疏（Figure 1右）。
  - **KL方向效应：** Forward KL在所有配置下保持71–73%；Reverse KL范围35–72%，对LR高度敏感。
  - **Rollout谱实验（Figure 3）：** Forward KL在整个谱上保持>80%准确率（仅波动5.2pt）；Reverse KL波动剧烈，明显偏好student-favored rollouts（$\lambda>0$）。
  - **泛化到更难任务（Countdown-4E）：** 两种KL方向下，on-policy rollout均带来10–15%的pass@k提升（Figure 5右）。
  - **后续RLVR（Figure 6）：** OnPD的初始泛化优势在后续RLVR中不可靠维持；Forward KL+OffPD的checkpoint在RLVR后达到最高稳定性能。
  - **Teacher风格迁移（Appendix A.11）：** OnPD+Reverse KL有效抑制偶发的teacher风格（西班牙语）迁移，其他配置均发生近完全迁移。
  - **Numina-MATH更长推理（Appendix A.3）：** Forward KL仍更稳健；低LR下OnPD+Reverse KL在MATH-500上取得最高59.0%（基准53.1%），但遗忘和稀疏性仍由LR主导。
  - **替代设置鲁棒性：** 去掉梯度裁剪（Appendix A.1）、使用sampled KL估计（Appendix A.2）、更换Qwen2.5模型族（Appendix A.5）均得到相同定性结论。

## 相关工作脉络
- **Agarwal et al. (2023)** 提出OnPD原始算法；本文在其基础上系统解耦rollout策略与KL方向，揭示了前者并非决定性因素。
- **Shenfeld et al. (2025, 2026)** 声称on-policy学习减少灾难性遗忘；本文通过控制变量实验表明遗忘主要由学习率驱动，挑战了这一因果归因。
- **Mukherjee et al. (2025)** 声称on-policy学习产生更稀疏参数更新；本文发现OffPD在匹配比较中至少同样稀疏，稀疏性由学习率而非rollout策略决定。
- **Chu et al. (2025)** 声称RL比SFT更好泛化；本文在蒸馏设定下发现on-policy数据的泛化优势仅存在于特定设置（更难的Countdown变体），且不持续于后续RLVR。
- **DeepSeek-AI et al. (2026)** 使用full-vocabulary KL训练MiMo系列；本文验证了full-vocabulary KL的设置不影响核心结论，同时对比了sampled KL的等效性。
- **Lu & Lab (2025); Li et al. (2026b)** 使用sampled KL估计器的on-policy蒸馏方法；本文Appendix A.2表明替换为sampled KL后定性结论不变。

## 局限性与未来方向
- **模型规模有限：** 学生模型最大1.5B参数，推理轨迹最长≤2000 tokens；更大模型上的结论需要验证。
- **Teacher固定：** 本文Teacher在蒸馏期间冻结，未探索不同student-teacher组合下OnPD/OffPD差异是否出现。
- **训练步数较短：** 150步优化可能不足以捕捉长期动态，尤其是对遗忘和稀疏性的影响。
- **未涵盖所有后训练方法：** 分析集中在蒸馏设定，对SFT与RLVR等其他方法的因果归因仍需进一步研究。
- **未来方向：** 扩展到更大模型、更长的推理链、探索多阶段post-training pipeline中不同方法的组合策略。

## 研究启发与可借鉴点
- **解耦实验设计范式：** 通过构造连续插值谱（$\lambda \in [-2,2]$）系统扫描rollout策略的影响，是一个值得借鉴的消融方法论，可用于其他超参数的连续效应分析。
- **KL方向优先于rollout策略的实践经验：** 在实际蒸馏训练中，应优先选择合适的KL方向（forward KL更稳健），再考虑是否投入on-policy rollouts的计算成本。
- **OnPD+Reverse KL抑制风格迁移：** 当需要防止teacher的不希望的风格/行为迁移到student时，OnPD配reverse KL是一个有效的策略组合。
- **学习率是遗忘/稀疏性的首要控制旋钮：** 低学习率+forward KL可实现高性能且几乎无遗忘，为高效蒸馏提供了实用的超参配置建议。
- **可复用的梯度分析工具：** Appendix B的token-level KL梯度推导和Appendix C的rollout稳定性理论界，可作为后续分析其他蒸馏变体的基础工具。

## 关键术语表
- **On-policy Distillation (OnPD)：** 学生模型从自身当前策略采样rollout进行蒸馏，训练数据依赖被优化的模型。
- **Off-policy Distillation (OffPD)：** 学生模型从固定teacher策略采样rollout进行蒸馏，训练数据不依赖学生当前参数。
- **Forward KL（前向KL）：** $D_{F-KL}(\pi_S\|\pi_T)=\mathbb{E}_{y\sim\pi_T}[\log\frac{\pi_T(y)}{\pi_S(y)}]$，mode-covering，对teacher高概率token给予更大权重。
- **Reverse KL（反向KL）：** $D_{R-KL}(\pi_S\|\pi_T)=\mathbb{E}_{y\sim\pi_S}[\log\frac{\pi_S(y)}{\pi_T(y)}]$，mode-seeking，惩罚学生分配给teacher低概率token的概率质量。
- **Catastrophic Forgetting（灾难性遗忘）：** 模型在适应新任务后，原有OOD能力显著下降的现象。
- **Parameter-update Sparsity（参数更新稀疏性）：** 参数绝对变化量低于阈值（$10^{-6}$）的参数占比，反映模型更新的稀疏程度。
- **Rollout-policy Spectrum：** 通过超参数$\lambda$连续插值student-teacher分布的rollout策略谱，用于系统扫描rollout来源的影响。
- **RLVR（Reinforcement Learning with Verifiable Rewards）：** 基于可验证奖励的强化学习微调方法，本文用于评估蒸馏checkpoint的后续训练潜力。

## 可复现要素
- **数据集：** MedReason、Science（SciKnowEval）、Countdown（程序化生成）；OOD基准含MMLU-Pro、TruthfulQA、IFEval、HumanEval-Instruct、EQ-Bench、BBQ、ToxiGen；Numina-MATH（MATH+NuminaMath-CoT合成数据混合）。训练数据未公开声明，评估代码和数据描述在Appendix D.1。
- **代码/权重：** 论文未明确声明代码开源链接；提及使用了vLLM后端，实验细节在Appendix D完整记录。
- **关键超参：** 优化步数150，Fused AdamW（$\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$），无weight decay，LR=$\{1\times10^{-5}, 5\times10^{-5}\}$，梯度裁剪norm=1.0，学生temperature=1.0，teacher temperature=0.5，effective batch size=32，warmup 10步。
