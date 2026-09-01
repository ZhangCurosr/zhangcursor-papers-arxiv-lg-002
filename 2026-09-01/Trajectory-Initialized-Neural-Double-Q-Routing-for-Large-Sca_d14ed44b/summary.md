---
title: "Trajectory-Initialized-Neural-Double-Q-Routing-for-Large-Sca"
source: https://arxiv.org/pdf/2608.30512v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:21:18"
---

# 论文速读：Trajectory-Initialized-Neural-Double-Q-Routing-for-Large-Sca

## 一句话总结
本文针对半导体厂务OHT（Overhead Hoist Transport）系统的大规模动态路由问题，提出了一种离线轨迹初始化、在线自适应的Neural Double Q-routing方法。通过将传统目的地索引查表替换为共享的状态-动作价值网络，并结合Double-Q目标、局部拥塞修正与事件分层结构化重放，在多种车队规模与到达率设定下显著降低平均任务完成时间，并在冷启动阶段大幅提升任务完成率。

## 研究问题与动机
1. **核心问题**：OHT系统共享受限的天花板导引轨，车辆实际行驶时间受安全间距、交叉口独占权、下游阻塞及装卸站竞争等时变交通成本耦合影响，静态最短路径无法动态响应拥堵演化。
2. **查表法信息孤岛**：Tabular Q-routing为每个$(d,i,j)$三元组独立维护值表（约259万条目），稀疏访问导致价值更新效率低，且已观察上下文的经验无法迁移至结构相似的稀疏节点。
3. **在线启动敏感**：随机初始化的价值网络在积累足够经验前缺乏有效决策依据；使用单一估计算法同时进行动作选择与价值评估会放大正偏差（positive bootstrap bias）。
4. **全局网络滞后与采样偏差**：共享价值网络可能滞后于短生命周期局部拥堵；均匀重放采样难以覆盖罕见的阻塞、近死锁及装卸瓶颈事件，导致策略对长尾延误响应不足。

## 核心贡献（创新点）
1. **共享状态-动作价值表征**：用固定维度的MLP替代目的地索引查表，参数量从约259万降至5825，参数共享使频繁访问上下文的经验直接惠及稀疏路由节点。与已有工作的本质区别在于不依赖图神经网络的全局拓扑编码，而是通过局部有界特征向量直接对候选出边打分。
2. **混合轨迹离线冷启动**：利用Dijkstra、Q-routing和Double Q-routing生成的混合仿真轨迹进行return-to-go监督回归，为共享网络提供高质量初始估计。与offline RL或imitation learning的本质区别在于仅作为初始化先验，不优化独立策略且不施加单一行为策略的模仿损失。
3. **Double-Q精炼与双机制协同适配**：采用Double-DQN分离动作选择与价值评估以消除最大化偏差；引入未进入递归bootstrap的局部TD残差修正保证对近期拥堵的快速响应；设计事件分层结构化重放（SR）按领域语义配额采样罕见关键状态。三者协同解决了“全局稳定学习”与“局部快速适应”的张力。

## 方法详解
- **问题建模**：将导引轨建模为有向图$G=(V,E)$，路由决策在split节点处以半马尔可夫过程进行。状态向量$s\in\mathbb{R}^{10}$含当前/目标节点归一化索引、最短距离、节点度、装卸点指示、任务阶段（pickup/delivery/other）及近期等待时间；动作特征$\phi_a(s,j)\in\mathbb{R}^{14}$含候选边长度、剩余距离、进度、度、边占用率、节点队列及6个局部压力摘要（瓶颈压力、回溯风险、1跳/2跳最大与平均下游压力、活跃等待车比例）。所有连续分量归一化至$[0,1]$。
- **奖励函数**：$r_t = -\sum_{e \in \mathcal{T}_t}(w_m m_{t,e} + w_w w_{t,e} + w_b b_{t,e} + w_d \omega_{t,e}) - w_r \rho(s_t,a_t) + w_p \delta_h(s_t,a_t)$，综合运动/等待/阻塞时间、死锁警告级别、局部风险峰值与路线进度。权重固定为$(1.0,0.8,1.2,0.1,0.15,0.3)$。
- **共享价值网络**：$Q_\theta(s,a)=f_\theta(\phi(s,a))$，两层MLP（隐层64单元、ReLU），输入维度$d_\phi=24$，含偏置共5825参数。输出为折扣return-to-go估计，值越大越优。
- **离线warm start**：目标为混合轨迹的return-to-go $\hat{G}_t=\sum_{k=t}^{T_\ell}\gamma^{k-t}r_k$，最小化$\mathcal{L}_{prior}=\mathbb{E}[(Q_\theta(s,a)-\hat{G}(s,a))^2]$。训练40 epoch，Adam $lr=10^{-3}$，batch size=64，按策略/种子/单元格平衡采样。
- **在线Double-Q更新**：$a^*=\arg\max_{a'}Q_\theta(s',a')$，目标$y=r+\gamma(1-z)Q_{\bar{\theta}}(s',a^*)$。使用Huber损失$\mathcal{L}_{td}$，目标网络Polyak平均$\bar{\theta}\leftarrow(1-\tau)\bar{\theta}+\tau\theta$（$\tau=0.005$）。每4次观测更新一次，初始经验阈值64条。
- **局部拥塞修正**：$U_t(s,a)=Q_\theta(s,a)+\lambda_t\kappa(s,a)R_{local}(s,a)-\beta_t(n(s,a)+1)^{-1/2}$。残差$R_{local}$以$\alpha=0.05$指数滑动更新并clip至$[-5,5]$；$\lambda_t=0.5\min(t/5000,1)$、$\beta_t=0.3\max(1-t/5000,0)$线性调度。该修正仅作用于行为策略打分，不进入bootstrap目标，避免残差递归传播。
- **事件分层结构化重放（SR）**：转移按严重程度标记为normal/congested/blocked/near-deadlock/LU-bott
