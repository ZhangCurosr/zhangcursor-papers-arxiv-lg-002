---
title: "Revisiting-On-policy-Adversarial-Black-Box-Distillation-Cali"
source: https://arxiv.org/pdf/2609.39757v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-04 23:44:30"
field: "语言模型蒸馏与对齐"
keywords: ["black-box distillation", "on-policy RL", "reward geometry", "optimal transport", "adversarial training", "language model"]
innovations: ["提出 GRGC 两阶段干预 critic→advantage 接口以缓解群级奖励几何失配", "设计高斯群级最优传输正则化校准奖励离散度与尺度", "构造有符号幂变换在保持单调性的同时放大组内区分度"]
benchmarks: ["LMSYS-Chat", "Dolly-Train"]
---

# 论文速读：Revisiting-On-policy-Adversarial-Black-Box-Distillation-Cali

## 一句话总结
论文诊断了 on-policy 对抗式黑盒蒸馏中 critic 与 advantage 之间的**群级奖励几何失配**问题，并提出 **GRGC（Groupwise Reward Geometry Conditioning）** 两阶段干预方法，通过批评侧高斯最优传输正则化与策略侧幂次调制，显著提升小型语言模型蒸馏性能。

## 研究问题与动机
- **核心问题**：现有 GAD 类方法的 Bradley-Terry (BT) critic 仅优化教师-学生判别能力，未约束学生奖励组的离散度、排序稳定性与尺度分布。
- **奖励塌陷倾向**：Proposition 1 证明在固定均值下 BT 目标严格偏好组内零离散度（Jensen 不等式），导致 critic 训练天然倾向奖励塌陷。
- **排序脆弱性**：Proposition 2 表明当组内边际过小时，噪声扰动可翻转组内排序，低离散度直接引发排序脆弱。
- **尺度放大效应**：Proposition 3 指出弱尺度下标准化过程会放大噪声（误差与 $\sigma_x$ 成反比），进一步扭曲 advantage 信号。

## 核心贡献（创新点）
- **提出 GRGC 框架**：首次系统建模 critic→advantage 接口处的群级奖励几何失配，并设计两阶段干预机制。
- **设计 CGC（批评侧几何校准）**：引入高斯群级最优传输正则化，将排序奖励对齐至标准正态分位数，从损失层面惩罚塌陷与尺度失配。
- **设计 PGM（策略侧群调制）**：构造有符号幂变换放大组内区分度并保持 critic 诱导排序，增强 advantage 信噪比。
- **提供理论保障与实验验证**：证明 OT 损失下界，并在多教师-学生配置、跨任务场景中验证 GRGC 的显著增益。

## 方法详解
- **GRGC 整体流程**：两阶段干预 critic→advantage 接口，分别在校准批评输出与调制策略输入上施加几何约束。
- **CGC 设计**：
  - 对分组奖励排序后，使用高斯最优传输将 $r_{(j)}$ 映射至分位数目标 $t_j = \mu_x + \Phi^{-1}((j-0.5)/N)$。
  - OT 损失 $\mathcal{L}_{\mathrm{OT}} = \frac{1}{N}\sum_{j=1}^{N}(r_{(j)} - t_j)^2$，critic 总损失 $\mathcal{L}_{\mathrm{critic}} = \mathcal{L}_{\mathrm{BT}} + \lambda_{\mathrm{OT}} \mathcal{L}_{\mathrm{OT}}$（$\lambda_{\mathrm{OT}}=0.01$）。
  - 命题 4 证明 $\mathcal{L}_{\mathrm{OT}} \ge (\sigma_x - \sigma_q)^2$，直接惩罚离散度偏离与尺度失配。
- **PGM 设计**：
  - 标准化 $z_i = (r_i - \mu_x)/(\sigma_x+\varepsilon)$，施加有符号幂变换 $\hat{r}_i = \mathrm{sign}(z_i)|z_i|^\gamma$（$\gamma=1.5$）。
  - 变换保持单调性（维持 critic 诱导排序），同时放大已有区分度的组内差距。
  - 变换后执行 GRPO prompt-wise advantage 归一化。
- **训练超参**：group size $N=8$，KL weight $\beta=0.001$，训练温度 0.8，学习率 $1\times10^{-6}$，全局 batch size 128，共 2 epochs（1 epoch 预热 + 1 epoch 对抗）。

## 实验与结果
- **数据集与配置**：Doubao-Seed-2.0 / GPT-5-Chat 作为教师，LMSYS-Chat、Dolly-Train 作为训练语料；学生模型 Qwen2.5（1.5B/3B）、Llama-3.2（1B/3B）。
- **低离散度抑制（Table 1）**：相对 std < 0.15 发生率从 GAD 全程 ~8.5% 降至 GRGC ≤ 1.90%；< 0.25 从 ~15%–16% 降至 ≤ 1.90%。
- **主要性能（Table 2–4）**：
  - GPT-5-Chat → Qwen2.5-3B：LMSYS Score 47.06 → GRGC **50.19**（Win 18.4% → **45.3%**）。
  - GPT-5-Chat → Llama-3.2-3B：**49.07**（Win **40.1%**）vs GAD 46.51/15.9%。
  - Doubao-Seed-2.0 → Qwen2.5-3B：**49.49**（Win **49.9%**）vs GAD 48.16/42.2%。
  - Doubao → Dolly OOD（Table 4）：GRGC 同样全面领先。
- **消融（Table 5–7）**：CGC+PGM 联合最优；单独模块均有增益但弱于联合；CGC 中高斯 OT 最佳（47.36）；PGM 中 Group Power 变换最佳。
- **开销（Table 8）**：CGC 0.03s/步、PGM 0.01s/步；相对开销仅 0.06%–0.11%。

## 相关工作脉络
- **GAD 系列基线**：本文针对其 BT critic 在几何失配上的理论缺陷进行改进，区别于仅优化判别准确度的 prior work。
- **On-policy black-box distillation**：沿用 GRPO advantage 构建范式，但首次在 critic 与 advantage 接口引入显式几何正则。
- **Reward geometry / distribution shaping**：相较于仅关注奖励值校准的工作，本文同时约束离散度、序关系与尺度分布。
- **Optimal transport in RL**：将 OT 用于奖励分位数对齐属于新颖应用，区别于传统 OT 在分布匹配或 policy gradient 中的使用。
- **Advantage modulation**：PGM 的幂变换策略不同于常规 CDF/Rank/Top-1 归一化，强调在保持单调前提下的非线性放大。

## 局限性与未来方向
- **教师依赖**：实验主要在 GPT-5-Chat、Doubao-Seed-2.0 等强教师上验证，对中等质量教师的泛化未充分讨论。
- **超参敏感性**：$\gamma$、$\lambda_{\mathrm{OT}}$、$N$ 等需人工设定，缺乏自动调度或鲁棒性分析。
- **任务范围**：以聊天/指令微调为主，未涵盖代码、数学推理等其他领域蒸馏场景。
- **理论深度**：命题集中于几何不等式层面，未建立收敛性或泛化误差的严格界。
- **潜在风险**：Broader Impact 已指出有害行为迁移风险上升，但缺乏相应的缓解机制设计。

## 研究启发与可借鉴点
- **几何条件化接口**：critic→advantage 的显式几何校准思路可迁移至其他 on-policy 蒸馏或 RLHF  pipeline。
- **高斯 OT 正则化**：将分位数对齐作为辅助损失，易于实现且开销极低，适合嵌入现有训练循环。
- **单调非线性调制**：有符号幂变换在保持排序的同时放大区分度，可作为通用 advantage 预处理算子。
- **理论诊断先行**：通过命题揭示 BT 目标的塌陷倾向，为后续方法设计提供清晰动机，值得在方法论文中效仿。
- **多维度消融**：分别验证 CGC/PGM 及分布选择、变换选择，有助于明确各组件贡献，实验设计严谨可复用。

## 关键术语表
- **On-policy adversarial black-box distillation**：在不访问教师内部状态的情况下，通过策略梯度与对抗采样完成知识蒸馏的训练范式。
- **Groupwise reward geometry mismatch**：critic 输出的奖励分布在组内离散度、排序稳定性与尺度上与 advantage 计算所需几何不一致的问题。
- **Bradley-Terry critic**：基于成对比较对数似然的奖励判别模型，常见于黑盒蒸馏的批评器设计。
- **Reward collapse**：奖励组内方差趋近于零，导致排序与优势信号失效的现象。
- **Gaussian groupwise optimal transport**：将排序后的组内奖励通过 OT 映射至标准正态分位数的正则化技术。
- **Policy-side group modulation**：在策略更新前对标准化奖励施加非线性单调变换，以增强组内区分度的后处理步骤。
- **GRPO advantage**：基于 group relative policy optimization 的 prompt-wise 优势估计，用于策略梯度更新。

## 可复现要素
- **数据集**：LMSYS-Chat、Dolly-Train；教师模型 GPT-5-Chat、Doubao-Seed-2.0；学生模型 Qwen2.5-1.5B/3B、Llama-3.2-1B/3B（均为公开模型/数据）。
- **代码/权重**：论文未提及开源代码或新权重发布（Section 13 标注 N/A）。
- **关键超参**：$N=8$、$\lambda_{\mathrm{OT}}=0.01$、$\gamma=1.5$、$\beta=0.001$、训练温度 0.8、学习率 $1\times10^{-6}$、全局 batch 128、2 epochs。
