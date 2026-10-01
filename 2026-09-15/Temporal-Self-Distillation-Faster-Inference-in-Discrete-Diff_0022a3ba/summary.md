---
title: "Temporal-Self-Distillation-Faster-Inference-in-Discrete-Diff"
source: https://arxiv.org/pdf/2609.15177v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:57:18"
field: "扩散语言模型高效推理"
keywords: ["diffusion language models", "self-distillation", "fast inference", "parallel decoding", "on-policy distillation", "masked diffusion"]
innovations: ["提出TSD：将模型commit时刻分布作为on-policy教师信号，跨时间步自蒸馏以提升早期预测质量", "引入长度感知奖励加权，防止长canvas下冗余生成稀释蒸馏信号", "证明JSD相比KL散度在自蒸馏中更稳定，避免reward collapse"]
benchmarks: ["GSM8K", "MATH-500", "Countdown", "Sudoku", "MBPP", "HumanEval", "LiveCodeBench"]
---

# 论文速读：Temporal-Self-Distillation-Faster-Inference-in-Discrete-Diff

## 一句话总结
本文提出时序自蒸馏（Temporal Self-Distillation, TSD）方法，通过在去噪轨迹上将模型较早时间步的预测分布向该 token 最终被 commit 时刻的分布对齐，实现离散扩散语言模型（dLLM）的单阶段在策略加速，显著改善低 NFE（函数评估次数）区间的速度-质量 Pareto 前沿，无需离线教师生成或双阶段训练。

## 研究问题与动机
- **核心问题**：dLLM 虽可并行生成多 token，但过度激进地减少去噪步骤会导致生成质量严重下降，速度-质量之间存在显著权衡（speed-quality trade-off）。
- **现有方法不足①**：自适应解码调度方法（如 Fast-dLLM）仅调整 unmasking 时机，无法从根本上改善早期时间步预测不准的问题。
- **现有方法不足②**：离线蒸馏方法（如 dParallel、SDTT）需先生成完整教师轨迹数据集再训练学生，成本高且割裂了训练与模型自身行为演化。
- **动机**：直接缩小模型早期预测与最终 commit 预测之间的差距，使早期 commit 更安全，从而支持更激进的并行解码。

## 核心贡献（创新点）
1. **提出 TSD——跨时间步的在策略自蒸馏目标**：将模型在 commit 时刻的分布作为 stop-gradient 教师信号，训练更早时间步的预测向其对齐；与离线蒸馏的本质区别在于无需外部教师或预收集轨迹数据，教师即模型自身当前 rollout。
2. **引入长度感知奖励加权（Length-Aware Reward Weighting）**：通过对超过目标长度的成功轨迹施加线性惩罚，防止模型利用长 canvas 生成冗长答案，从而促进低 NFE 解码；该设计是纯奖励加权所不具备的。
3. **证明 JSD 比 KL 散度更适合 TSD 训练**：KL 散度在 teacher 概率较小时导致更新不稳定甚至 reward collapse，而 JSD 对称且有界，empirically 保持稳定并持续降低 NFE。
4. **TSD 可与 RL post-training 无缝衔接**：在 GDSD 后训练的规划任务上单独应用 TSD，验证了该方法可作为 RL 后训练的加速模块，而非仅限于 base model。
5. **单阶段 Monte Carlo 近似使计算开销可控**：每条轨迹仅均匀采样一个时间步计算蒸馏 loss，而非全程梯度回溯，保持了单阶段高效训练。

## 方法详解
- **基本设定**：给定 prompt $c$，从当前模型 $p_\theta$ 用 Fast-dLLM（置信度阈值 $\lambda$）采样一条去噪轨迹 $\bm{x}^{0:T_c}$，每个位置 $\ell$ 有唯一 commit 时刻 $t_\ell$。
- **蒸馏目标**：对同一位置 $\ell$，用较早时间步 $\hat{t}$ 的预测分布 $p_\theta(\cdot|c, x^{\hat{t}})_\ell$ 去匹配 commit 时刻分布 $p_\theta(\cdot|c, x^{t_\ell})_\ell$，后者加 stop-gradient（sg）。
- **损失函数**（完整形式，式6）：
  $\mathcal{L}_{\mathrm{TSD}}(\theta) = \mathbb{E}\left[ V(\bm{x}^0) \frac{1}{T_c}\sum_{t=1}^{T_c}\sum_{\ell \in \mathcal{M}_t} D(p_\theta(\cdot|c,x^t)_\ell,\ \mathrm{sg}[p_\theta(\cdot|c,x^{t_\ell})_\ell]) \right]$
  其中 $D$ 为散度（主实验用 $\mathrm{JSD}_{\beta=0.5}$），$V(\bm{x}^0)$ 为 verifier 奖励权重。
- **单时间步 Monte Carlo 近似**（式7-8）：每条轨迹均匀采样 $\hat{t} \sim \mathrm{Unif}(\{1,\dots,T_c\})$，仅在该步计算 loss，是无偏估计。
- **长度感知奖励加权**（式9）：
  $V_{\mathrm{len}}(\bm{x}^0) = V(\bm{x}^0)\left[1 - \gamma_{\mathrm{len}}\frac{(|\bm{x}^0|-L_{\mathrm{target}})_+}{L-L_{\mathrm{target}}}\right]$
  当生成长度超过 $L_{\mathrm{target}}$ 时线性降权，$\gamma_{\mathrm{len}}=1$ 时对超长轨迹赋零权重。
- **训练流程**：每步采样 $G=4$ 条 on-policy rollout，用 LoRA（$r=64,\alpha=32$）更新， AdamW 优化，学习率 $3\times10^{-6}$，训练约 2000–3000 步。

## 实验与结果
- **数据集**：数学推理（GSM8K、MATH-500）、规划（Countdown、Sudoku）、代码生成（MBPP、HumanEval、LiveCodeBench），共 7 个 benchmark。
- **基线**：LLaDA-8B-Instruct（Fast-dLLM）、GDSD（Fast-dLLM）、dParallel（离线蒸馏）。
- **主要结果（$L=256$）**：
  - **MBPP**：TSD 在约 45 NFEs 处达到 ~50% pass rate，比 base 模型峰值准确率所需 NFE 减少约 2.3 倍；相比 dParallel 减少约 1.5 倍 NFE。
  - **HumanEval**：TSD 在约 45 NFEs 处达到 ~40% pass rate，base 模型需两倍以上的 NFE 才能达到相近水平。
  - **Countdown**：TSD 在约 20 NFEs 处达到 ~80% accuracy，而 GDSD 基准需约 90-100 NFEs（约 4.5× 加速）。
  - **Sudoku**：TSD 在 <25 NFEs 内达到 86–89% 精度，超越 GDSD 峰值（~85%），同时更高效且更准确。
  - **GSM8K**：TSD 在 35–50 NFEs 达到 ~80% 准确率，base 需约 65 NFEs；在 30 NFEs 时 base 低于 40%，TSD 已接近 80%。
  - **MATH-500**：提升较温和，base 在高 NFE 区间仍保留最高峰值，体现 TSD 定位（加速而非提升绝对峰值）。
  - **LiveCodeBench**：TSD 在低 NFE 区间建立更强 Pareto 前沿，base 仅在最高 NFE 设置下略优。
- **最强结果**：Countdown 任务上实现约 4.5× NFE 加速（20 vs ~90 NFEs）达到 ~80% accuracy；MBPP 上以 2.3× 更少 NFE 达到 40% pass rate。

## 相关工作脉络
1. **Fast-dLLM（Wu et al., 2026）**：训练无关的 dLLM 并行解码加速方法，通过置信度阈值控制 unmasking；TSD 与其正交——Fast-dLLM 是解码器，TSD 改进模型本身的早期预测质量。
2. **dParallel（Chen et al., 2026）**：离线 confidence-distillation 方法，训练模型更快达到高置信度；TSD 无需离线数据收集和额外教师 checkpoint。
3. **SDTT（Deschenaux & Gulcehre, 2025）**：多步轨迹蒸馏压缩采样步数；依赖预收集的离线 trajectory，TSD 完全 in-training on-policy。
4. **GDSD（Tang et al., 2026a）**：将 RL 微调重新表述为 guided self-distillation；TSD 是独立于 RL 的加速模块，可在 GDSD 之后单独应用。
5. **COPSD（Zhu et al., 2026）**：同样针对 early-vs-final gap 的 in-policy 自蒸馏，但 teacher 信号是基于后续步骤构建的未归一化目标，跨位置共享；TSD 使用每个位置自身 commit 时刻的归一化分布，target 更精确。
6. **dOPSD（Dat et al., 2026）**：对所有未来 masked 步的预测取平均作为 teacher；TSD 只取最后一个 commit 时刻的预测，节省一次 teacher forward pass。

## 局限性与未来方向
- **不增加模型能力**：TSD 只能让模型已有的行为提前出现，无法产生模型最终不会预测的内容，因此在高 NFE 峰值准确率上无提升（如 MATH-500）。
- **仅适用于 unmasking-only 解码**：要求每个位置有唯一 commit 时刻；支持 revise committed token 的采样器需要重新定义 target。
- **依赖程序化正确性信号**：reward weighting 限定在有 verifier 的任务上，开放式生成（open-ended generation）目前不在适用范围。
- **长度感知加权需逐任务选择 $L_{\mathrm{target}}$**：引入了一个需人工设定的超参。
- **未来方向**：扩展至支持 token 修订的解码器；探索在开放式生成任务中的适配形式；自动化 $L_{\mathrm{target}}$ 的选择策略。

## 研究启发与可借鉴点
1. **"commit 时刻分布作为 on-policy target"**：将同一位置在信息更完备时的预测作为教师信号，思路简洁且通用，可迁移至其他去噪/迭代生成范式（如 continuous diffusion、iterated denoising）。
2. **长度感知奖励加权防止 canvas-filling 失败模式**：在长 canvas 设置下，正确但过长的 rollout 会稀释蒸馏信号；线性降权机制对任何有 length variance 的任务均有借鉴价值。
3. **JSD 优于 KL 在自蒸馏中的稳定性**：KL 散度因 teacher 分布稀疏导致的 reward collapse 问题值得在其他自蒸馏/RLHF 场景中警惕，JSD 可作为稳定替代。
4. **TSD 作为 RL 后训练的即插即用加速器**：TSD 可与任何 on-policy 训练流程（如 GRPO、GDSD）衔接，证明"能力构建"与"推理加速"可分离且互补。
5. **单时间步 Monte Carlo 近似保持计算效率**：全轨迹梯度计算开销大，均匀采样单个 $\hat{t}$ 是无偏且高效的近似策略，适用于任何时序蒸馏场景。

## 关键术语表
- **TSD（Temporal Self-Distillation）**：时序自蒸馏，将 dLLM 在较早去噪步的预测分布向该 token 最终 commit 时刻的分布对齐的在策略蒸馏方法。
- **dLLM（Diffusion Language Model）**：扩散语言模型，特别是 masked dLLM，通过迭代去噪（unmasking）生成文本，可并行预测多 token。
- **NFE（Number of Function Evaluations）**：函数评估次数，即模型前向传播次数，用于度量 dLLM 推理成本；AR 解码约需 $L$ 次 NFE，dLLM 理想下仅需 $S \ll L$ 次。
- **Fast-dLLM**：一种训练无关的 dLLM 并行解码策略，基于置信度阈值 $\lambda$ 决定哪些 masked 位置在每一步被 unmask。
- **Commit 时刻（$t_\ell$）**：位置 $\ell$ 在去噪轨迹中首次被 unmask 并确定 token 的时间步，TSD 的 teacher 信号即来源于此时刻的模型分布。
- **长度感知奖励加权（Length-Aware Reward Weighting）**：对超过目标长度 $L_{\mathrm{target}}$ 的成功 rollout 线性降权，防止模型利用长 canvas 生成冗余输出。
- **on-policy self-distillation**：在策略自蒸馏，teacher 信号来自模型自身当前策略的 rollout，而非离线收集的数据或外部教师模型。
- **JSD（Jensen-Shannon divergence）**：Jensen-Shannon 散度，对称且有界（≤ log 2），本文用作 TSD 对齐早期与 commit 分布的散度度量。

## 可复现要素
- **数据集**：GSM8K（开源）、MATH-500（开源子集）、Countdown（TinyZero 项目，开源）、Sudoku（d1 codebase 本地提供）、MBPP（开源）、HumanEval（开源）、LiveCodeBench release_v5（开源）；部分训练数据来自 KodCode-Light-RL-10k。
- **代码/权重**：LLaDA-8B-Instruct 开源；GDSD checkpoint 公开可用（论文附链接）；TSD 代码未明确声明开源，但附录提供了详细训练超参和 prompt template。
- **关键超参**：LoRA $r=64, \alpha=32$, dropout=0.05；AdamW, lr=$3\times10^{-6}$, grad clip=0.2；rollouts per prompt=4, block length=32, 训练阈值 $\lambda=0.9$；$\mathrm{JSD}_{\beta=0.5}$；训练约 2000 步（MATH-500 用 1000 步）；2×A100/H200 GPU。
