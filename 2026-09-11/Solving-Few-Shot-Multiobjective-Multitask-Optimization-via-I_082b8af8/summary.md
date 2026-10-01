---
title: "Solving-Few-Shot-Multiobjective-Multitask-Optimization-via-I"
source: https://arxiv.org/pdf/2609.11228v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:35:42"
field: "进化多目标优化"
keywords: ["few-shot optimization", "multiobjective multitask", "sequential transfer", "Gaussian process", "task prioritization", "negative transfer"]
innovations: ["IST框架将MTO重构为序贯迁移序列", "似然驱动task优先级机制", "软-max随机选择防永冻"]
benchmarks: ["MOMTO benchmark (CIHS/CIMS/PIHS/NIHS etc.)", "Yahpo Gym HPO"]
---

# 论文速读：Solving-Few-Shot-Multiobjective-Multitask-Optimization-via-I

## 一句话总结
本文提出迭代序贯迁移（Iterative Sequential Transfer, IST）框架，将少样本多目标多任务优化（MOMTO）建模为一系列序贯迁移优化问题的序列，通过似然驱动的	task 优先级机制在严格预算约束下实现高效知识迁移。

## 研究问题与动机
- **少样本瓶颈**：传统 MTO 需分散评估预算至所有任务，导致单个任务无法积累足够精英解分布，难以支持有效迁移。
- **负迁移风险**：源任务质量不足时，盲目同步评估会触发有害的负迁移，尤其在多目标场景（需逼近连续 Pareto 流形而非单点）。
- **高维映射困难**：多目标优化需处理向量值目标函数 $F_k(\cdot)$，传统单目标迁移机制无法直接适用。
- **资源分配僵化**：现有方法缺乏动态优先级决策，无法在迭代中自适应选择"最Ready"的目标任务。

## 核心贡献（创新点）
- **IST 框架重构 MTO**：将并行多任务优化转化为序贯选择-优化序列，每轮仅评估一个目标任务，本质区别在于"按需分配"vs"均分预算"。
- **似然驱动任务优先级**：基于 MTGP 学到的任务核参数 $\kappa_\tau(i,t)$ 作为迁移似然，$\phi(t)=\min_{i\neq t}\kappa_\tau(i,t)$ 确保目标任务与全源任务池兼容。
- **软-max随机选择机制**：引入温度参数 $S$ 和阈值 $\theta$，防止低似然任务永冻，同时保留高似然任务的优先性。
- **F-invTrEMO 基优化器**：结合前向映射（标量化目标）与逆映射（权重→解空间）的混合 MTGP 迁移，适配少样本多目标场景。
- **通用性验证**：IST 可嫁接至 AMTEA 等经典序贯迁移优化器，在 11/14 任务上显著提升性能。

## 方法详解
### 1. IST 通用框架（Algorithm 1）
- **初始化**：每任务评估 $N_{init}$ 次，生成源数据集 $\mathcal{D}_S$。
- **循环**：当终止条件未满足，按 $\phi(t)$ 选目标任务 $T$，执行一次序贯迁移优化。
- **公式 (3)**：$\min_{x_T} f_T(x_T)$ s.t. $T=\arg\max_t \phi(t)$，$Eval_t < N_{tot}$。

### 2. 基优化器 F-invTrEMO（Algorithm 2）
- **标量化**：增广 Tchebycheff 标量 $f_K^{tch}(x|w)=\max_i\{w_i(f_{K,i}(x)-(z^*_{K,i}-\epsilon))\}+\rho\sum_i w_i(f_{K,i}(x)-(z^*_{K,i}-\epsilon))$。
- **前向 MTGP**：$\mu_{fmt}(T_K,x)=\sigma^2_{fmt}(T_K,x)\{\sum_{j=1}^{K-1}\sigma^{-2}_{T_j}(T_K,x)\mu_{T_j}(T_K,x)+(2-K)\sigma^{-2}_{T_K}(x)\mu_{T_K}(x)\}$（公式 8），因子化解耦避免源任务主导。
- **逆映射 MTGP**：分解为 $d$ 个单输出 GP，$\Psi_{inv,i}:W\mapsto\Omega_{T,i}$。
- **采样与选择**：从逆映射分布采样 $U_{T_K}$，选 LCB 最大解 $\tilde{x}=\arg\max_{x^-}[-\mu_{fmt}+\beta\sigma_{fmt}]$。

### 3. 似然驱动优先级（Section III.C）
- **公式 (10)**：$\phi(t)=\min_{i\neq t}\kappa_\mathcal{T}(i,t)$，取最小值保证与所有源任务兼容性。
- **公式 (11)-(12)**：软-max概率 $P_t=\frac{\exp(S\cdot\max\{\phi(t)-\theta,0\})}{\sum_j\exp(S\cdot\max\{\phi(j)-\theta,0\})}$，$S$ 控制选择压力，$\theta$ 保底防止永冻。

## 实验与结果
- **基准**：9 组 MOMTO 问题（CIHS/CIMS/CILS/PIHS/PIMS/PILS/NIHS/NIMS/NILS），IGD+ 度量，20 次独立试验。
- **基线**：ParEGO（单目标少样本）、F-invTrEMO（无 IST）。
- **主要结果**（Table I）：
  - F-invTrEMO-IST 在 12/14 任务上显著优于 F-invTrEMO。
  - CIHS/PIHS/NIHS 等高相似度场景提升显著（如 NIHS Task-1: 3.174E+03→5.934E+02，降幅 81%）。
  - LS（低相似度）场景表现不稳定，因解分布映射易偏置局部 Pareto 前沿。
- **真实应用**（Table II）：
  - HPO-1（3 任务 Random Forest 调参）：F-invTrEMO-IST 在所有 3 任务上优于 F-invTrEMO。
  - HPO-2（2 任务跨模型调参）：同样全面领先。
- **通用性**（AMTEA-IST）：在 11/14 任务上优于 AMTEA，LS 场景提升更显著。

## 相关工作脉络
- **MTO 基准**（Yuan et al., 2017 [23]）：本文扩展至 $O(10^2)$ 评估预算的少样本设定，区别于标准 $O(10^5)$。
- **STrO 方法**（Extremo [19], F-invTrEMO [21]）：本文将其嵌入 IST 框架，从"固定源-目标对"变为"动态优先级选择"。
- **MTGP 迁移**（Bonilla et al., 2007 [27]; Da et al., 2019 [29]）：因子化后向迁移避免源任务主导，本文沿用并扩展至多目标。
- **逆映射优化**（Liu et al., 2024 [20]）：权重→解空间的逆 GP 建模，本文结合前向标量化形成混合迁移。
- **负迁移抑制**（Gong et al., 2019 [17]; Wei & Zhong, 2021 [18]）：本文通过优先级机制"自适应过滤"低utility源-目标对。

## 局限性与未来方向
- **异构目标数限制**：当前分解式优化器无法处理 NIMS/NILS（目标数不同），需开发通用多目标迁移机制。
- **低相似度脆弱性**：LS 场景下解分布映射易偏置，需引入多样性保持或局部逃逸策略。
- **超参敏感**：$S$（软-max温度）、$\theta$（阈值）、$\beta$（LCB 探索系数）需针对场景调优。
- **扩展方向**：高维逆映射（$d\gg1$）的计算效率、在线似然重估、跨域迁移（如材料设计→药物发现）。

## 研究启发与可借鉴点
- **序贯重构思路**：将"并行多任务"转化为"序贯单任务+动态选择"，可迁移至多臂老虎机、贝叶斯优化等预算受限场景。
- **似然驱动优先级**：用已学模型参数（核函数、相关性）直接作为迁移utility指标，避免额外代价的预测开销。
- **软-max保底机制**：$\theta$ 阈值+指数放大结合，平衡" exploitation 高利任务"与" exploration 低利任务"，适用于任何资源分配问题。
- **因子化解耦**：MTGP 后向用各单源任务单独建模再融合（公式 8-9），避免大规模联合训练的偏差，可推广至多源联邦优化。
- **通用框架设计**：IST 作为"元框架"可嫁接不同基优化器（F-invTrEMO/AMTEA），本团队可据此构建统一迁移优化平台。

## 关键术语表
- **IST (Iterative Sequential Transfer)**：将多任务优化重构为迭代序贯迁移序列的框架，每轮只评估一个目标任务。
- **MOMTO (Multiobjective Multitask Optimization)**：同时优化多个多目标任务并 exploits 任务间协同性的范式。
- **F-invTrEMO**：前向-逆映射贝叶斯迁移多目标优化器，结合标量化前向 GP 与权重→解空间逆 GP。
- **MTGP (Multitask Gaussian Process)**：多任务高斯过程，通过任务核 $\kappa_\mathcal{T}(i,i')$ 编码任务间相关性。
- **Likelihood-Informed Prioritization**：基于任务核似然 $\kappa_\mathcal{T}$ 的 task 优先级机制，$\phi(t)=\min_{i\neq t}\kappa_\mathcal{T}(i,t)$。
- **Augmented Tchebycheff Scalarization**：增广 Tchebycheff 标量化，$f^{tch}(x|w)=\max_i\{w_i(f_i-z^*_i)\}+\rho\sum w_i(f_i-z^*_i)$。
- **IGD+ (Inverted Generational Distance++)**：多目标优化性能度量，衡量近似 Pareto 前沿到真前沿的平均最近距离。
- **Negative Transfer**：源任务质量不足或相似度低时，迁移反而损害目标任务性能的负面现象。

## 可复现要素
- **数据集**：MOMTO 基准（9 组问题，代码见 https://github.com/ambigeV/stro）；Yahpo Gym HPO 基准 [31]。
- **代码**：公开于 GitHub（链接见论文 Section I 末尾）。
- **关键超参**：$N_{init}$（初始化评估次数）、$N_{tot}$（每任务总预算）、$S$（软-max 温度）、$\theta$（优先级阈值）、$\beta$（LCB 探索系数）——详细设置见补充材料。
- **复现难度**：中等，需实现 MTGP 因子化后向及软-max 调度逻辑；基优化器 F-invTrEMO 代码开源可复用。
