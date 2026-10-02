---
title: "Minimax-Last-Iterate-Convergence-in-Matrix-Games-with-Observ"
source: https://arxiv.org/pdf/2609.34656v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:33:39"
field: "博弈论在线学习 / 多智能体强化学习"
keywords: ["last-iterate convergence", "zero-sum matrix games", "bandit feedback", "adaptive averaging", "implicit exploration", "minimax optimality"]
innovations: ["自适应平均与指数权重比率修正联合设计，直接在regret bound中产生吸收方差负的项，将维度依赖从d²降至√d", "潜在函数分析把方差控制转化为任意时刻保证，实现不依赖时间视界的anytime收敛", "放弃全局收益矩阵估计，改用O(d)内存的显式向量更新，避免log-barrier正则化博弈求解开销"]
---

# 论文速读：Minimax-Last-Iterate-Convergence-in-Matrix-Games-with-Observed-Actions

## 一句话总结
本文在未知双玩家零和矩阵博弈（每方d个动作）中，针对"bandit支付反馈+可观测对手动作"这一设定，首次给出了同时在动作数d和时间t上达到minimax最优的高概率最后迭代收敛算法，将duality gap从先前的 $\widetilde{\mathcal{O}}(d^2/\sqrt{t})$ 改善至 $\widetilde{\mathcal{O}}(\sqrt{d/t})$，提升维度依赖 $d^{3/2}$ 倍。

## 研究问题与动机
- **平均策略收敛 ≠ 最后迭代收敛**：经典no-regret算法仅保证历史平均策略收敛到均衡，但自对弈/偏好学习等"学习与部署同时进行"的场景下，需要**每一轮实际使用的策略**都接近Nash均衡，因此研究last-iterate收敛具有直接应用价值。
- **未观测对手动作时维度代价过高**：在bandit反馈且无法观测对手动作的设定下，现有last-iterate速率分别达到 $\widetilde{\mathcal{O}}(\sqrt{d}\,t^{-1/8})$（Cai et al., 2023）、$\widetilde{\mathcal{O}}(d^{1/5}t^{-1/5})$（Cai et al., 2025）、$\widetilde{\mathcal{O}}(d^2 t^{-1/4})$（Fiegel et al., 2026b），时间收敛速度或维度依赖均不理想。
- **观测对手动作后仍未达minimax下界**：已有观测对手动作的最优工作 Hait et al. (2026) 给出 $\widetilde{\mathcal{O}}(d^2/\sqrt{t})$，时间依赖 $t^{-1/2}$ 已最优，但**维度依赖**与bandit下界 $\Omega(\sqrt{d/t})$ 之间存在 $d^{3/2}$ 的gap。
- **核心科学问题**：观测对手动作所提供的额外信息，能否在统计意义上支撑同时达到 $d$ 和 $t$ 的minimax速率？

## 核心贡献（创新点）
- **首个d与t同时minimax最优的最后迭代速率**：给出 $\mathrm{Gap}(x_t,y_t) \leq C\sqrt{d/t}\,[\log(dt/\delta)]^{3/2}$，与标准bandit下界匹配，填补了观测对手动作情形下维度依赖的gap。
- **自适应平均与指数权重的联合设计**：提出"自适应平均权重 $a_t$"与"指数权重中的比率修正 $x_{t,i}/x_{t+1,i}$"共同作用——前者控制双方importance ratio之和 $S_{x,t}+S_{y,t}\leq d/a_t$，后者在regret bound中产生负项 $-\kappa_t K_{x,t}$ 吸收IX估计方差。
- **潜在函数分析实现 anytime 保证**：通过追踪未归一化坐标 $M_{t,i}=\tau_t x_{t,i}$ 的对数增量完成 telescoping，证明每阶段仅需 $\mathcal{O}(d H^2/e^2)$ 轮即可将gap减半；同时给出坐标下界 $x_{t,i}\geq p_i/5$，保证多阶段串联不失控。
- **计算高效：每轮 $\mathcal{O}(d)$ 时间与内存**：直接显式向量更新，无需估计 $d\times d$ 收益矩阵，也无需在每个 epoch 边界求解带 log-barrier 的正则化博弈（对比 Hait et al. 2026）。

## 方法详解
算法由若干 phase 构成，phase $k$ 以精度参数 $e_k=2^{1-k}$ 和失败概率 $\rho_k=\delta/2^{k+1}$ 运行，累计失败概率 $\sum \rho_k\leq \delta$。每阶段内部维护**辅助策略** $(u_t,v_t)$ 与**实际策略** $(x_t,y_t)$：

1. **自适应平均（Adaptive Averaging）**：初始化 $\tau_0=\max\{6d,\lceil 16384 d H^2/e^2\rceil\}$，每轮更新
   $$\tau_{t+1}=\tau_t+a_t,\quad x_{t+1}=\frac{\tau_t x_t+a_t u_t}{\tau_{t+1}},\quad y_{t+1}=\frac{\tau_t y_t+a_t v_t}{\tau_{t+1}}.$$
   权重 $a_t=\min\{1,\, d/(S_{x,t}+S_{y,t})\}$，其中 $S_{x,t}=\sum_i u_{t,i}/x_{t,i}$。该设计确保 $a_t(S_{x,t}+S_{y,t})\leq d$，从而双方累积重要性比值有界。

2. **带隐式探索（IX）的辅助损失估计**：定义非负损失 $g_t=(\mathbf{1}+Av_t)/2$、$h_t=(\mathbf{1}-A^\top u_t)/2$，构造带平滑分母的无偏估计
   $$\widehat{g}_{t,i}=\frac{\mathbf{1}\{I_t=i\}(1+R_t)s_{t,J_t}}{2(x_{t,i}+\zeta_t s_{t,J_t})},\quad \zeta_t=\eta a_t.$$
   分母中的 $\zeta_t s_{t,J_t}$ 项产生有界向下偏差，同时使 $\eta a_t\widehat{g}_{t,i}\in[0,1]$，便于使用Neu (2015) 风格的指数矩不等式控制方差。

3. **带比率修正的指数权重（Exponential Weights with Ratio Correction）**：辅助策略更新为
   $$u_{t+1,i}\propto u_{t,i}\exp(-\eta a_t\widehat{g}_{t,i})\cdot\frac{x_{t,i}}{x_{t+1,i}},$$
   修正因子 $\frac{x_{t,i}}{x_{t+1,i}}=\frac{1+\kappa_t}{1+\kappa_t r_{t,i}}$（$\kappa_t=a_t/\tau_t$）对大的 $r_{t,i}=u_{t,i}/x_{t,i}$ 进行降权。该修正产生的负项 $-\kappa_t K_{x,t}$ 在 regret bound 中抵消 IX 估计方差。

4. **阶段终止与参数设置**：阶段在 $\tau_{N+1}\geq 4\tau_0$ 时终止；学习率设为 $\eta=16H/(e\tau_0)$，保证 $\eta^2 d\tau_t\leq 1/4$，使负项完全吸收方差项。

## 实验与结果
- 本文属于理论分析工作，**无数值实验部分**；主要结果以定理形式给出。
- **主要理论结果（Theorem 1）**：以概率至少 $1-\delta$，对所有 $t\geq 1$ 同时成立
  $$\mathrm{Gap}(x_t,y_t)\leq C\sqrt{\frac{d}{t}}\left[\log\!\left(\frac{dt}{\delta}\right)\right]^{3/2}.$$
- **相比 Hait et al. (2026) 的提升**：维度依赖从 $d^2$ 降至 $\sqrt{d}$，改善 $d^{3/2}$ 倍；达到 ε-duality gap 所需轮数从 $\widetilde{\mathcal{O}}(d^4/\varepsilon^2)$ 降至 $\widetilde{\mathcal{O}}(d/\varepsilon^2)$。
- **Minimax 最优性**：当博弈矩阵列相同时退化为 d-armed bandit，标准纯探索下界为 $\Omega(\sqrt{d/t})$，本文上界与之匹配（至对数因子）。

## 相关工作脉络
- **Cai et al. (2023, 2025)** 与 **Fiegel et al. (2026b)** 研究未观测对手动作的bandit反馈下的last-iterate收敛，但时间收敛速率分别为 $t^{-1/8}$、$t^{-1/5}$、$t^{-1/4}$，且维度依赖分别为 $d^{4}$（转化为ε样本量后）、$d^{8}$ 等，本文在观测对手动作的更强设定下实现同时最优的 $d$ 与 $t$ 依赖。
- **Hait et al. (2026)** 是最直接的先前工作，通过"估计收益矩阵+周期性求解带log-barrier的正则化博弈"达到 $\widetilde{\mathcal{O}}(d^2/\sqrt{t})$，维度缺陷源于将entrywise payoff估计误差传递到策略时的额外 $d^{3/2}$ 放大；本文放弃全局矩阵估计，直接估计辅助损失向量并以自适应平均+比率修正控制方差，从根本上消除该放大方差项。
- **Neu (2015)** 的隐式探索（IX）技术：本文将其适配到两玩家 bandit 设定，并通过 $\zeta_t=\eta a_t$ 使平滑量与自适应权重同阶。
- **Cai et al. (2025)** 的 A2L 约化思想（将平均迭代转化为最后迭代）：本文沿此思路，但在bandit设定下辅助策略与实际策略分布不同导致方差过大；本文的核心贡献正是对这一方差的精确控制。
- **Munos et al. (2023)** 等 NLHF 工作：自对弈学习语言模型策略的动机来源之一，其中双方动作均可记录，对应本文的"观测对手动作"设定。

## 局限性与未来方向
- **对数因子未消除**：最终界含 $[\log(dt/\delta)]^{3/2}$ 对数因子，尚未证明是否可去；是否能在不依赖 log-barrier 的前提下消除对数损失是开放问题。
- **无数值验证**：本文为纯理论论文，未提供实验部分；$\mathcal{O}(d)$ 每轮的理论效率在实际高维博弈中是否仍具优势尚需实证检验。
- **仅适用于零和矩阵博弈**：方法依赖双线性 payoff 结构和 duality gap 的凸性，推广至一般和博弈、马尔可夫博弈或连续动作空间需进一步研究。
- **基线坐标下界随阶段退化**：Lemma 4 保证 $x_{t,i}\geq p_i/5$，因此每阶段 $R_0$ 最多增加 $2\log 5$，使对数项累积；若起始策略本身某些坐标极小，常数项会增大，但未讨论自适应初始化策略的可能性。

## 研究启发与可借鉴点
- **"自适应平均 + 比率修正"的方差控制范式**：核心思路——让修正项在 regret 不等式中产生负项，抵消 IX 估计方差——可迁移到任何"辅助策略分布与采样分布不同"的 bandit 设定，如部分可观测 MDP 或 bandit 上的 no-regret learning。
- **Telescoping 对数增量分析技术**：通过 $M_{t,i}=\tau_t x_{t,i}$ 跟踪未归一化坐标的对数增量，可将阶段长度与控制方差分离处理，这种 potential 论证对任何"混合旧策略以保证 exploration"的算法均有参考价值。
- **与强化学习的结合机会**：算法仅需 $\mathcal{O}(d)$ 内存和每轮显式向量更新，可嵌入大规模 self-play 框架（如 LLM 偏好对齐）中作为轻量级策略更新模块，替代昂贵的正则化博弈求解步骤。
- **对数因子优化的潜在方向**：当前 $[\log(dt/\delta)]^{3/2}$ 来自 $R_0$、$L$、$H$ 三项对数叠加；若能设计不依赖基线坐标下界的 new phase scheme，有望将指数降至 $[\log(dt/\delta)]^{1/2}$ 甚至消除对数。
- **与 online convex optimization 中的"internal regret"转化技巧对照**：本文本质是将 auxiliary regret 转化为 played strategy 的 duality gap 控制，这与 OCO 中 internal/external regret 的转换类似，可考虑借鉴 OCO 近年改进的 variance-reduced 估计进一步压缩常数。

## 关键术语表
- **Last-iterate convergence（最后迭代收敛）**：要求每一轮实际使用的策略 $(x_t,y_t)$ 本身收敛到 Nash 均衡，而非历史平均策略；适用于"边学边用"的场景。
- **Duality gap（对偶间隙）**：$\mathrm{Gap}(x,y)=\max_j x^\top A e_j - \min_i e_i^\top A y$，度量策略对偏离均衡的单向收益之和；为零当且仅当为 Nash 均衡。
- **Bandit payoff feedback（bandit 支付反馈）**：每轮仅观察到当前所选动作对 $(I_t,J_t)$ 处的一个噪声支付 $R_t$，无法获知未选动作的支付。
- **Observed opponent actions（可观测对手动作）**：除支付外，双方均能观察到对手本轮实际采样的动作，提供超出单一支付标量的额外信息。
- **Adaptive averaging（自适应平均）**：以动态权重 $a_t$ 将辅助策略逐步融入实际策略，当辅助策略在某些动作上概率远大于实际策略时自动缩小 $a_t$。
- **Ratio correction（比率修正）**：在指数权重更新中乘以因子 $x_{t,i}/x_{t+1,i}$，对大的 ratio $r_{t,i}=u_{t,i}/x_{t,i}$ 施加额外惩罚，在 regret bound 中产生吸收方差的负项。
- **Implicit exploration (IX)**：通过在 importance sampling 估计的分母中加入与对手 ratio 成正比的平滑项 $\zeta_t s_{t,J_t}$，使 scaled 估计有界，再用指数矩不等式控制偏差与方差。
- **Phase scheme（阶段机制）**：将学习过程切分为精度参数逐次减半的 phase，每 phase 以一定失败概率输出更优 baseline，累计失败概率可控。

## 可复现要素
- **数据集**：本文属理论分析，无实证数据集；所有结论针对一般未知 $A\in[-1,1]^{d\times d}$ 成立。
- **代码/权重开源**：**论文未提及**代码开源状态。
- **关键超参**：$\tau_0=\max\{6d,\lceil 16384\, d H^2/e^2\rceil\}$、$\eta=16H/(e\tau_0)$、$\zeta_t=\eta a_t$；其中 $H=R_0+\log 5+2L$，$R_0=\log(1/\min_i p_i)+\log(1/\min_j q_j)$，$L=\log((2d+2)/\rho)$；阶段终止条件 $\tau_{N+1}\geq 4\tau_0$。
