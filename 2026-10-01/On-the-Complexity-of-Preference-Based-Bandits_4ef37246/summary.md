---
title: "On-the-Complexity-of-Preference-Based-Bandits"
source: https://arxiv.org/pdf/2609.39351v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-04 06:33:04"
---

# 论文速读：On-the-Complexity-of-Preference-Based-Bandits

## 一句话总结
本文研究了基于成对偏好反馈（Bradley–Terry模型）的广义奖励函数类 dueling bandit 问题，提出了局部敏感 eluder 维度与 GINOP 算法，首次在不依赖全局非线性常数 $\kappa$ 的前提下建立了首阶 regret 界，从理论上证明了偏好反馈学习与直接奖励观察具有同等的统计效率。

## 研究问题与动机
- **核心问题**：偏好 bandit 的观测信号为二值比较反馈，受 logistic 链接函数影响，存在依赖于问题的常数 $\kappa = \sup_{x,x'} 1/\dot{\sigma}(f^\star(x)-f^\star(x'))$，当奖励范围 $S=5$ 时 $\kappa > 2.2\times10^4$，传统方法中 $\kappa$ 会污染 regret 的主阶项。
- **模型泛化不足**：既有研究主要聚焦线性或 RKHS 奖励模型，无法刻画现实中复杂的非线性效用函数；直接套用全局 eluder 维度会在广义线性设定中不可避免地引入对 $\kappa$ 的不利依赖。
- **决策机制局限**：现有算法多采用两阶段策略（leader-follower 或先筛可行胜者集再选对），未能联合优化双臂的收益与信息增益，导致探索效率受限。
- **理论目标**：构建适用于一般奖励函数类的精细 regret 界，证明在合理的复杂度度量下，偏好反馈带来的非线性并不削弱学习的统计效率。

## 核心贡献（创新点）
1. **局部敏感 eluder 维度**：将 sigmoid 一阶导数 $\dot{\sigma}$ 直接嵌入 eluder 维度的预测误差定义，使复杂度度量自适应链接函数局部曲率，从根本上规避了对全局 $\kappa$ 的依赖。与已有工作的本质区别：不同于 Bakhtiari et al. (2025) 依赖隐式子集定位且额外 regret 项可能退化为 $O(T)$，本文的定义内生于度量本身，余项全程可控。
2. **GINOP 算法**：基于 log-loss 构建置信集，并通过 $\hat{f}_t(x)+\hat{f}_t(x')+\omega_t(x,x')$ 联合选择双臂，同步兼顾乐观收益与信息性探索。与已有工作的本质区别：突破 leader-follower 或两阶段可行集范式，一步联合优化成对决策，且探索 bonus 专门针对 dueling 结构设计了差异不确定性 $\omega_t$。
3
