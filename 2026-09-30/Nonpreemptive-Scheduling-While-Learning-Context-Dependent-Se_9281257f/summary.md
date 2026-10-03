---
title: "Nonpreemptive-Scheduling-While-Learning-Context-Dependent-Se"
source: https://arxiv.org/pdf/2609.37660v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-03 14:10:49"
---

# 论文速读：Nonpreemptive-Scheduling-While-Learning-Context-Dependent-Se

## 一句话总结
本文研究了单服务器非抢占式上下文队列赌博机问题，证明即使模型完全已知，最优调度也无法由简单贪心规则刻画；提出了 LCP 与 Estimated-SEPT 两类算法，分别在已知时域（$\widetilde{O}(\sqrt{d/T})$）与未知时域（跟踪真 SEPT，误差 $\widetilde{O}(\sqrt{d/t})$）下给出紧理论界，并建立了任何时域 anytime 策略渐近零遗憾的不可能性定理。

## 研究问题与动机
- **核心问题**：在作业离开概率由 logistic 模型 $\mu(x^\top\theta^*)$ 决定的单服务器非抢占式队列中，调度器如何在未知 $\theta^*$ 的情况下边学习边控制队列长度，实现最小遗憾？
- **现有方法不足**：抢占式设定中的“work-conserving + 最大离开概率优先（SEPT）”贪心规则在非抢占下不再最优；服务一旦开始便不可中断，导致最优动作强烈依赖剩余时域、当前队列状态与未来到达分布。
- **理论空白**：缺乏针对非抢占设置的遗憾率紧界；已知工作（如 Bae et al. 2026、Bae & Lee 2026）集中于抢占式且对 $T$ 的依赖较松；未知固定时域下是否可设计 anytime 策略此前未明。

## 核心贡献（创新点）
1. **揭示非抢占调度的内在复杂性**：证明即使已知完整模型，最优调度也无法用简单贪心规则刻画，且 SEPT 规则在非抢占下不再最优。与已有抢占式工作的本质区别在于，非抢占约束引入了对剩余步数与队列状态的强依赖，打破了“最大权贪心即最优”的直觉。
2. **提出 Learn–Clear–Plan (LCP) 算法**：分探索、清空、规划三阶段运行有限时域 Bellman 递归。与抢占式方法相比，不要求特征协方差矩阵的特征值下界，且在 $T \geq d^2$ 时达到 $\widetilde{O}(\sqrt{d/T})$ 的遗憾率，至多 polylog 因子优于下界。
3. **建立 anytime 策略不可能性定理**：证明在 IA/WC 设定下，未知时域中不存在渐近零遗憾的 anytime 策略（即使模型已知），从而将研究目标从“零遗憾”转向“对真 SEPT 的跟踪”，并给出 $\widetilde{O}(\sqrt{d/t})$ 的误差上界。
4. **设计 Estimated-SEPT 并证明紧跟踪界**：仅在繁忙期之间更新 logistic 估计，同一繁忙期内保持固定；证明了算法与真 SEPT 的队列差距满足 $\widetilde{O}(\sqrt{d/(\lambda t)})$，平均跟踪误差与均匀随机时刻期望队列长度同步收敛。

## 方法详解
- **LCP 算法 (Algorithm 1)**：
  - **Learn**：前 $L = T - H$ 轮采用 FCFS 收集样本，每作业仅记录首次服务的离开指示 $Y_i$；利用 $\hat{\theta}_n$ 估计 $\hat{p}_n(x) = \text{proj}_{[p_-, p_+]}(\mu(x^\top \hat{\theta}_n))$，并用后半段样本构造经验分布 $\widehat{F}_n$。
  - **Clear**：若队列未空，继续 FCFS 直至首次到达空队列，消除历史依赖。
  - **Plan**：基于 $(\hat{p}_n, \widehat{F}_n)$ 运行有限时域 Bellman 递归求解最优动作；未来到达分布仅描述新作业，队列中已有作业直接纳入状态。
- **Estimated-SEPT 算法 (Algorithm 2)**：针对未知 $T$ 场景。在繁忙期结束后更新参数估计；同一繁忙期内估计固定，空闲时选择 $\arg\max_j \hat{p}(X_j)$ 的等待作业（ties 按到达顺序打破）。
- **参数估计**：$\hat{\theta}_n \in \arg\min_{\|\theta\|_2 \leq S} \frac{1}{n}\sum_{i=1}^n [\log(1+e^{X_i^\top\theta}) - Y_i X_i^\top\theta]$，配合截断投影保证数值稳定性。
- **工作负载 (Workload) 工具**：定义 $W_t = \sum$ 所有在途作业的剩余服务时间（服务时间 $\sim \text{Geom}(p(x))$）。关键性质：$\mathbb{E} e^{rW_t} \leq K_r$，稳定性条件为 $\rho = \lambda \mathbb{E}[1/p(X)] < 1$；在公共耦合下所有 WC 策略共享同一 $W_t$ 过程，且 $Q_t \leq W_t$。

## 实验与结果
- 本文为理论分析论文，未提供大规模数值仿真或标准 benchmark 数据集，核心结果以遗憾率界与下界形式呈现。
- **已知时域 $T$（IA/WC）**：LCP 遗憾率 $\widetilde{O}(\sqrt{d/T})$（Thm 1）；下界 $\Omega(\min\{d^{-1/2}, \sqrt{d/T}\})$（Thm 2）。当 $T \geq d^2$ 时 LCP 达到最优，仅差 polylog 因子。
- **未知时域（IA/WC）**：不可能性定理（Thm 3）——任意 anytime 策略满足 $\limsup_{t\to\infty} R_t^S(\pi) > 0$。
- **未知时域（WC）**：Estimated-SEPT 跟踪误差 $\mathcal{T}_t^{\text{SEPT}} = |J_t^{\text{alg}} - J_t^{\text{SEPT}}| = \widetilde{O}(\sqrt{d/(\lambda t)})$（Thm 4），平均跟踪误差 $\frac{1}{T}\sum_{t=1}^T \mathcal{T}_t^{\text{SEPT}} = \widetilde{O}(\sqrt{d/(\lambda T)})$。
- **最强结果与提升幅度**：相比抢占式基线 Bae et al. 2026（$\widetilde{O}(T^{-1/4})$）与 CQB-η-2（$\widetilde{O}(T^{-1/2})$），本文在非抢占设定下同时改善了对 $T$ 的依赖指数，并首次给出 $\widetilde{O}(\sqrt{d/T})$ 的紧界与下界匹配结果。

## 相关工作脉络
- **Bae et al. 2026 / Bae & Lee 2026**：抢占式上下文队列赌博机，提出 CQB-η-2 等算法。本文定位差异：打破抢占假设，证明非抢占下贪心失效，需依赖 Bellman 递归；同时改善 $T$ 依赖指数并覆盖未知时域。
- **SEPT / 最大权重调度经典理论**：在抢占式或稳态队列中greedy最优。本文定位差异：明确证明其在非抢占有限时域下不再最优，最优动作需联合考虑剩余步数与队列状态。
- **上下文赌博机与 logistic 回归学习**：标准线性赌博机框架。本文定位差异：将学习嵌入排队动力学，服务时间由离开概率内生决定，且受非抢占耦合约束。
- **非抢占队列调度（Operations Research）**：传统 OR 中针对已知分布的调度。本文定位差异：结合在线学习，参数未知且需边学边控，提供统计-计算统一的
