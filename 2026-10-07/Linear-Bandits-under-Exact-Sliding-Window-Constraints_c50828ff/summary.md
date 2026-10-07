---
title: "Linear-Bandits-under-Exact-Sliding-Window-Constraints"
source: https://arxiv.org/pdf/2610.08745v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-07 17:38:56"
---

# 论文速读：Linear-Bandits-under-Exact-Sliding-Window-Constraints

## 一句话总结
本文研究了线性赌徒在**精确滑动窗口约束**下的在线学习问题，提出了基于稀有切换的OFUL与稀疏更新乐观规划算法，在理论证明了零违反条件下达到次线性遗憾界 $\widetilde{\mathcal{O}}(d\sqrt{T} + \tau d + w)$，并在推荐多样性与LLM路由任务上验证了高效合规性。

## 研究问题与动机
- **精确滑窗约束设定**：要求任意连续 $w$ 步动作序列 $(x_{t-w+1}, \dots, x_t)$ 严格属于可行集 $\mathcal{C}$，探索过程中的每一轮动作均受历史时序约束，而非仅单轮或累积约束。
- **现有安全/约束Bandit方法不足**：点态安全Bandit仅约束单轮动作；长期约束仅控制累计违反量；切换成本Bandit关注换臂代价，三者均无法保证滑窗内联合动作的精确可行性。
- **状态空间指数膨胀挑战**：若将滑窗动态等价为标准线性MDP，所需特征维度下界为 $m^{w-1}$（指数于窗口长），需利用确定性已知转移结构设计低维算法以规避维度诅咒。

## 核心贡献（创新点）
1. **揭示离线稳态近似的必然 $w$ 阶差距**：证明当 $w \nmid T$ 时最优稳态轨迹与离线最优的加法gap为 $\Theta(w)$，为在线算法目标设定提供不可压缩的理论基准。与现有长短期约束研究不同，本文指出滑窗长度本身即构成固有优化损失。
2. **提出循环对称情形下的稀有切换OFUL**：引入转移直径 $\tau$ 刻画稳态历史间的可达性，将参数估计与可行性过渡解耦，实现次线性遗憾与零违反。与点态安全OFUL相比，本文算法主动规划最少步数的可行过渡序列，而非每轮局部投影。
3. **构建一般情形下的有限记忆稀疏规划框架**：以最近 $w-1$ 维动作作为确定性价状态，导出历史状态直径 $D$ 控制的最优遗憾界 $\widetilde{\mathcal{O}}(d\sqrt{T} + dD + w)$。与线性MDP路线相比，本文利用已知转移结构仅保留 $d$ 维奖励不确定性，彻底避免指数特征展开。
4. **在真实推荐与LLM路由任务上验证零违反与高效切换**：实验表明稀疏更新机制显著减少策略调整次数，同时严格满足滑窗约束。与Primal-Dual软约束方法相比，本文保证硬约束每步精确成立，避免累积违反风险。

## 方法详解
- **离线稳态基准**：定义可行稳态集 $\mathcal{Z} = \{z \in \mathcal{X} : (z,\dots,z)\in\mathcal{C}\}$，计算 $z^\star = \arg\max_{z\in\mathcal{Z}} f(z)$；当 $w \nmid T$ 时，静态重复 $z^\star$ 与离线最优的差距上界为 $(w-1)(f_{\max} - f(z^\star))$，且该 $O(w)$ 依赖不可避免。
- **循环对称场景稀有切换OFUL（Algorithm 1）**：基于行列式翻倍机制划分episode，每个episode选择乐观稳态动作 $z_k$ 重复执行；当需切换至 $z_{k+1}$ 时，调用转移预言机 $\Gamma(z_k, z_{k+1})$ 生成最多 $\tau$ 步的可行过渡序列，**过渡期观测不参与协方差矩阵更新**，统计项与过渡项完全解耦。
- **一般场景有限记忆规划（Algorithm 2）**：将状态记为 $s_t = (x_{t-w+1},\dots,x_{t-1})$，维护可行历史集合 $S_n$ 与Bellman最优值 $J_n^\star(s)$；每个episode内通过乐观上界 $\overline{U}_k(x_t)$ 贪心生成合法动作，当 $\det(V_t)$ 翻倍或规划 horizon 耗尽时触发新一轮规划，遗憾界为 $\widetilde{\mathcal{O}}(d\sqrt{T} + dD + w)$。
- **近似规划扩展**：允许每步规划存在加法误差 $\alpha_k \leq \alpha_{\max}$，遗憾界线性叠加 $d\alpha_{\max}$ 项；当 $\alpha_{\max} = O(1)$ 时被主项 $\widetilde{\mathcal{O}}(d\sqrt{T})$ 吸收，渐近率与精确规划一致。

## 实验与结果
- **数据集**：KuaiRand-1K（1,000用户、约437万视频、1,170万交互，按内容类别施加滑窗多样性约束）与 LLMRouterBench（40万+查询、21个数据集、33个语言模型，在滑窗预算内路由模型）。
- **合成实验（J.1–J.3）**：
  - J.1 验证稳态gap：$r=0$ 时 gap 为 0，$r=w/2$ 时达到最大 $w/4$，与理论 $\Theta(w)$ 紧界一致。
  - J.2 循环渐变约束（$d=8, w=8, T=16384$）：Rare-switching OFUL 的 pseudo-regret 显著低于 Frequent-update 版本；两者均保持零违反，Unconstrained OFUL 在 $\rho<2$ 时违反约束。
  - J.3 非循环规划（$w=4$）：Rare-Update 归一化遗憾随 $T$ 快速趋近 0；Stationary OFUL 收敛于约 0.1（对应最优周期轨迹 0.7 与稳态 0.6 的 gap）；Myopic 方法更差（约 0.15），验证历史状态规划的必要性。
- **真实数据表现（Table 6）**：LLM路由预算 0.15、$T=500$ 时，Rare-Update Optimistic Planning 获得 **0.7618±0.0116** 平均奖励、**0.1566±0.0132** 归一化遗憾且 **零违反**，策略更新次数仅 **249.8**（约为全轮更新的50%）；Unconstrained 与 Primal-Dual 分别产生 104.5 与 112.7 次违反。预算放宽至 0.35、$T=5000$ 时，Rare-Update 仍保持零违反，更新次数 1103.8，显著低于基线的 5000。

## 相关工作脉络
1.
