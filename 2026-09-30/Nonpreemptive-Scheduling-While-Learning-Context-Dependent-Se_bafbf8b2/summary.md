---
title: "Nonpreemptive-Scheduling-While-Learning-Context-Dependent-Se"
source: https://arxiv.org/pdf/2609.37660v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-03 14:11:39"
---

# 论文速读：Nonpreemptive Scheduling While Learning Context-Dependent Service Rates

## 一句话总结
本文研究单服务器非抢占式上下文队列赌博机问题，提出一种学习型 SEPT（最短期望处理时间优先）调度算法，证明该算法能以 $\widetilde{O}\!\left(\sqrt{d/(\lambda T)}\right)$ 的速率跟踪最优非抢占策略，有效解决未知服务分布下的在线调度与学习耦合难题。

## 研究问题与动机
- **核心问题**：在到达过程已知但上下文相关离开概率（服务模型）未知的单服务器队列中，进行在线非抢占式调度以最小化终端期望队列长度 $J_T^\pi = \mathbb{E}_\pi[Q_{T+1}]$。
- **非抢占式的本质困难**：与抢占式场景不同，非抢占式最优策略无法由简单的贪心（myopic）规则刻画；当剩余时步较短时，先 idle 再服务可能优于立即服务，策略必须耦合当前队列状态、剩余 horizon 与未来到达分布。
- **现有方法不足**：前期工作（Bae et al. 2026; Bae & Lee 2026）针对抢占式设定，最优策略为 work-conserving 加最高离开概率贪心规则，直接迁移至非抢占场景会导致严重的次优损失；缺乏非抢占约束下联合学习与调度的理论保证。
- **稳定性假设革新**：采用 $\rho = \lambda \mathbb{E}[1/p(X)] < 1$ 的稳定性条件替代传统 traffic-slack 条件，更贴合几何服务时间的实际队列系统。

## 核心贡献（创新点）
- **形式化非抢占式上下文队列赌博机**：统一刻画 IA（允许空闲）与 WC（工作守恒）两种操作模式下的队列演化与优化目标，填补非抢占学习排队理论空白。
- **设计学习型 SEPT 算法（Algorithm 2）**：通过分段维护离开概率预测器 $\widehat{p}_t$，在忙期（busy period）内应用 SEPT 优先级规则，实现“学习-调度”闭环，突破非抢占场景的非贪心特性。
- **建立策略误差收敛界（定理 4）**：证明算法终端队列长度与真实 SEPT 的差距以 $\widetilde{O}\!\left(\sqrt{d/(\lambda T)}\right)$ 速率收敛，且结论同时适用于最坏时刻评估与均匀随机截止。
- **发展非抢占队列分析工具**：首创将忙期分解、指数尾界与概率向量敏感性分析（Lipschitz 耦合论证）相结合的技术路线，为后续非抢占在线决策研究提供方法论基础。

## 方法详解
- **模型设定**：
  - 上下文 $X_t \in \mathbb{R}^d$（$\|X\|_2 \leq 1$），离开概率 $p(x) = \mu(x^\top \theta^*) = (1+e^{-x^\top \theta^*})^{-1}$，$\theta^*$ 未知且 $\|\theta^*\|_2 \leq S$。
  - 服务次数 $G(x) \sim \text{Geom}(p(x))$，期望服务时间 $\mathbb{E}[G(x)] = 1/p(x)$。
  - 到达过程 $A_t \sim \text{Bern}(\lambda)$，$A_t \perp X_t$，队列演化 $Q_{t+1} = (Q_t - D_t)^+ + A_t$。
  - 假设：$(A_t, X_t)$
