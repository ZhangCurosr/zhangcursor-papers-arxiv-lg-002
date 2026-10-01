---
title: "Temporal-Self-Distillation-Faster-Inference-in-Discrete-Diff"
source: https://arxiv.org/pdf/2609.15177v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:57:17"
field: "扩散语言模型高效推理"
keywords: ["扩散语言模型", "自蒸馏", "推理加速", "并行解码", "时序蒸馏", "on-policy学习"]
innovations: ["提出时序自蒸馏(TSD)方法，在去噪时间步间对模型自身进行on-policy蒸馏，无需离线教师数据", "单时间步蒙特卡洛近似与长度感知奖励加权，显著降低训练开销并防止长度坍缩", "在七个基准上验证，低NFE区域帕累托前沿显著优于基线和dParallel等离线方法"]
benchmarks: ["GSM8K", "MATH-500", "Countdown", "Sudoku", "MBPP", "HumanEval", "LiveCodeBench"]
---

# 论文速读：Temporal-Self-Distillation-Faster-Inference-in-Discrete-Diff

## 一句话总结
论文提出时序自蒸馏（TSD）方法，通过在去噪时间步上对自身进行on-policy蒸馏，使扩散语言模型（dLLM）在早期时间步就能做出可靠预测，从而大幅降低推理所需的函数评估次数（NFE），在数学推理、规划和代码生成任务上实现了显著的加速-质量平衡提升。

## 研究问题与动机
- 扩散语言模型可通过并行解码多个token实现加速推理，但过于激进的并行策略会导致生成质量严重下降，存在显著的"速度-质量权衡"困境。
- 现有加速方法主要分为两类：一是设计自适应解码调度策略（如基于置信度的unmasking选择），但受限于早期时间步的预测质量；二是离线蒸馏方法，需要先生成教师轨迹再训练学生模型，流程复杂且成本高昂。
- 核心挑战在于：如何让模型在早期去噪阶段（上下文不完整时）就接近最终提交token时的预测质量，从而支持更激进的并行解码。

## 核心贡献（创新点）
1. **提出TSD方法**：一种轻量级on-policy时序自蒸馏目标，训练模型预测自身最终会做出的输出，无需离线教师数据或两阶段训练流程——与离线蒸馏的本质区别在于教师和学生学习来自同一模型的当前策略轨迹。
2. **验证跨领域有效性**：在数学推理、规划、代码生成七个基准上验证，TSD显著将帕累托前沿推向低计算量区域——例如MBPP上达到40%通过率仅需LLaDA-8B-Instruct约2.3倍更少的NFE——与dParallel等离线方法的本质区别在于TSD不需要外部教师检查点。
3. **适用于RL后训练阶段**：TSD可与GDSD等强化学习后训练无缝衔接，作为加速器而非能力增强器，保留了RL学到的技能——与纯训练阶段蒸馏的本质区别在于其可作为独立加速模块复用已有RL模型。

## 方法详解
- **核心机制**：沿on-policy去噪轨迹，将模型在较早时间步t对位置l的预测分布 $p_\theta(\cdot|c, x^t)_\ell$ 向其在该位置最终提交时间步 $t_\ell$ 的预测分布 $p_\theta(\cdot|c, x^{t_\ell})_\ell$ 进行蒸馏，后者因上下文更完整而提供更准确的监督信号。
- **损失函数**：采用JS散度（JSD, $\beta=0.5$）作为度量，目标函数为：
  $\mathcal{L}_{\text{TSD}}(\theta) := \mathbb{E}_{x^{0:T_c}}[V(x^0) \frac{1}{T_c}\sum_{t=1}^{T_c}\sum_{\ell\in\mathcal{M}_t} D(p_\theta(\cdot|c,x^t)_\ell, \text{sg}[p_\theta(\cdot|c,x^{t_\ell})_\ell])]$
  其中sg表示stop-gradient，$V(x^0)$为verifier奖励。
- **单时间步蒙特卡洛近似**：每个轨迹均匀采样一个时间步 $\hat{t}\sim\text{Uniform}(\{1,\ldots,T_c\})$ 计算损失，避免全轨迹梯度计算开销。
- **长度感知奖励加权**：对超过目标长度 $L_{\text{target}}$ 的成功rollout进行线性降权，防止模型为填满画布而生成长冗长输出。
- **训练流程**：使用LoRA（rank=64, $\alpha=32$）微调，每prompt采样4条rollout，Fast-dLLM解码阈值$\lambda=0.9$，Block长度32。

## 实验与结果
- **数据集**：数学推理（GSM8K, MATH-500）、规划（Countdown, Sudoku）、代码生成（MBPP, HumanEval, LiveCodeBench）。
- **基线**：LLaDA-8B-Instruct (Fast-dLLM)、GDSD (Fast-dLLM)、dParallel（离线蒸馏方法）。
- **主要结果**：
  - Countdown：TSD在约20 NFEs达到~80%准确率，GDSD需约90 NFEs（4.5倍加速）。
  - MBPP：TSD达到40%通过率仅需约2.3倍更少NFEs；在~45 NFEs达到~50%通过率。
  - HumanEval：在~45 NFEs达到~40%通过率，基线需两倍更多forward passes。
  - GSM8K：在35-50 NFEs达到~80%准确率，基线需约65 NFEs。
  - MATH-500：改善低-中等NFE范围，但基线在高NFE时仍保持最高峰值。
- **消融**：奖励加权、长度感知加权、JS散度均有效；KL散度导致reward collapse和NFE饱和。
- **结论**：TSD在低NFE区域显著优于基线，与dParallel竞争甚至超越，且在RL后训练模型上同样有效。

## 相关工作脉络
- **Fast-dLLM (Wu et al., 2026)**：训练无关的自适应解码策略，通过置信度阈值控制unmasking位置；TSD通过训练本身改进早期预测，与仅优化解码策略形成互补。
- **dParallel (Chen et al., 2026)**：离线置信度蒸馏方法；TSD无需外部教师模型或预收集轨迹，单阶段on-policy完成。
- **GDSD (Tang et al., 2026a)**：将RL视为引导去噪器自蒸馏；TSD专注于推理加速而非能力增强，可作为GDSD等RL后的独立加速模块。
- **COPSD (Zhu et al., 2026)**：同向on-policy蒸馏思路，但目标是非归一化分布；TSD使用各token提交时间步的归一化分布，目标更精确。
- **dOPSD (Dat et al., 2026)**：对未提交位置的所有未来预测取平均；TSD仅用最后一次提交的预测，成本更低且更贴近实际决策。
- **SDTT (Deschenaux & Gulcehre, 2025)**：多步轨迹蒸馏；TSD通过单时间步蒙特卡洛近似显著降低计算开销。

## 局限性与未来方向
- TSD不能增加模型能力，只能使已具备的行为提前显现；在高NFE区域无法超越基线峰值准确率（如MATH-500）。
- 假设unmasking-only解码策略，对于允许修订已提交token的采样器需要重新定义目标。
- 奖励加权仅适用于有程序正确性信号的任務，开放式生成任务不在当前适用范围。
- 长度感知加权需要为每个任务手动设定目标长度 $L_{\text{target}}$。
- 未来方向：扩展到非unmasking-only解码器、开发自动目标长度选择机制、探索开放式生成任务的适用性。

## 研究启发与可借鉴点
- **时序自蒸馏范式**：将on-policy蒸馏从序列层面（如SDPO/OPSD在AR模型中）扩展到时间步层面，为其他序列生成模型的加速提供了新思路。
- **单时间步蒙特卡洛近似**：避免全轨迹梯度计算，大幅降低训练开销，这一技术可迁移到其他需要轨迹内监督信号的方法。
- **长度感知奖励加权**：防止模型利用长画布生成长冗长输出的"长度坍缩"现象，对代码生成和数学推理任务具有通用价值。
- **LoRA微调+on-policy蒸馏的组合**：仅需微调1.05%参数即可实现显著加速，适合资源受限场景。
- **与RL后训练的解耦结合**：TSD可作为独立加速模块叠加在RL训练好的模型上，不干扰已有能力。

## 关键术语表
**TSD (Temporal Self-Distillation)**：时序自蒸馏，通过在去噪时间步间蒸馏模型自身预测来实现加速的方法。
**NFE (Number of Function Evaluations)**：函数评估次数，衡量推理成本，一次NFE对应模型一次前向传播。
**Fast-dLLM**：基于置信度阈值的自适应unmasking解码策略，阈值越低并行度越高。
**on-policy**：训练数据由当前策略自身生成，保证蒸馏目标与模型实际行为一致。
**commitment time ($t_\ell$)**：token位置l被最终unmask并提交的时刻，TSD的目标分布取自此时刻。
**verifier ($V(x^0)$)**：任务特定的奖励函数，用于加权rollout在蒸馏损失中的贡献。
**JSD (Jensen-Shannon Divergence)**：对称且有界的散度度量，用于对齐早期与提交时间步的预测分布。
**Length-aware reward weighting**：对超出目标长度的成功rollout进行线性降权的正则化技术。

## 可复现要素
- **数据集**：GSM8K (train)、MATH (train, ~7.5K)、Countdown (TinyZero project)、Sudoku (Black-Phoenix/d1 codebase)、KodCode-Light-RL-10k——部分公开，部分来自开源项目。
- **代码/权重**：LLaDA-8B-Instruct开源；GDSD checkpoint公开；TSD代码未明确提及开源状态。
- **关键超参**：LoRA rank=64, α=32, dropout=0.05；learning rate=$3\times10^{-6}$；rollouts per prompt=4；inner updates=4；Fast-dLLM threshold=0.9；block length=32；JSD β=0.5；训练步骤2000（MATH-500用1000步）。
- **硬件**：2× NVIDIA A100/H200 GPUs。
