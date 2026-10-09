---
title: "Optimal-Regret-for-Online-Market-Making-with-Limit-Order-Boo"
source: https://arxiv.org/pdf/2610.09691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:51:33"
field: "在线学习与市场微观结构"
keywords: ["在线市场做市", "限制订单簿反馈", "遗憾界", "Hedge算法", "EXP3", "部分可观测学习", "对抗环境"]
innovations: ["在LOB反馈下建立高概率O(√T)遗憾界，优于先前期望T^(2/3)界", "证明双格偏移设计避免任意分布下的观测退化", "证明完全对抗环境下全反馈亦无法实现次线性遗憾"]
benchmarks: ["Maran and Restelli 2026 期望 regret T^(2/3)", "Cesa-Bianchi et al. 2024 two-bit feedback T^(2/3)"]
---

# 论文速读：Optimal-Regret-for-Online-Market-Making-with-Limit-Order-Book

## 一句话总结
本文在限制订单簿（LOB）反馈模型下，针对在线市场做市问题，建立了高概率的 $\widetilde{\mathcal{O}}(\sqrt{T})$ 遗憾界，优于此前期望 regret 的 $\widetilde{\mathcal{O}}(T^{2/3})$ 结果；同时证明了在完全对抗环境（报价与估值均由对抗者选择）下，即使有全反馈也无法实现次线性遗憾。

## 研究问题与动机
- **核心问题**：当市场价格由自适应对抗者选择、交易者估值 i.i.d. 抽取时，在线做市算法能否在限制订单簿（LOB）反馈下实现最优的 $\widetilde{\mathcal{O}}(\sqrt{T})$ 遗憾界？
- **已有工作不足**：Maran and Restelli [2026] 在同一反馈模型下仅证明了期望 regret 的 $\widetilde{\mathcal{O}}(T^{2/3})$ 上界，且假设市场价格为 oblivious 对抗者；而 Cesa-Bianchi et al. [2024] 的 two-bit 反馈模型忽略了 LOB 提供更丰富的信息。
- **理论空白**：LOB 反馈中，每个无交易轮次可重建报价区间内所有网格点的损失，但如何有效利用这一结构达到 $\widetilde{\mathcal{O}}(\sqrt{T})$ 未被解决。
- **完全对抗的不可学习性**：当交易者的私有估值也由对抗者任意选择时，即使拥有全反馈，次线性遗憾是否仍可能？论文证明这在理论上不可行。

## 核心贡献（创新点）
- **全反馈下的最优 regret 界**：提出基于双 Hedge 实例与价格交换策略的算法，首次在该设定下建立高概率 $\widetilde{\mathcal{O}}(\sqrt{T})$ 界；与 FTPL 基线算法的本质区别在于利用交换不降低效用的结构性质，实现 bid/ask 分解分析。
- **LOB 反馈下的高概率最优界**：设计基于 EXP3 的双格算法，通过隐式探索参数 $\beta$ 与观测概率 $Q_t(x)$ 的比估计，在高概率意义下达到 $\widetilde{\mathcal{O}}(\sqrt{T})$，优于先前期望 regret 的 $T^{2/3}$ 界且支持自适应对抗者；与前者本质不同在于利用了区间内所有格点损失的精确重建。
- **完全对抗环境的不可学习性定理**：证明当市场价格与交易者估值均由对抗者选择时，即使有全反馈，任何算法的期望 regret 至少为 $\Omega(T)$；与相关文献的对比突显了随机估值假设的必要性。
- **精细的分析技术**：证明在任意分布（无需 Lipschitz 光滑性）下，利用双格的偏移设计可避免原子点导致的观测退化，并建立 $\sum w_t^s(x)/(Q_t(x)+\beta) \leq 4\log((1+\beta)/\beta)$ 的关键引理。

## 方法详解
- **价格离散化**：将连续价格空间 $[0,1]$ 划分为均匀网格 $\mathcal{G}=\{i/K: i=0,\dots,K\}$，其中 $K=\lceil\sqrt{T}\rceil$；LOB 情形采用两个偏移网格 $\mathcal{G}_B=\{i/K+1/(4K): i=0,\dots,K-1\}\cup\{1\}$ 与 $\mathcal{G}_A=\{i/K+1/(2K): i=0,\dots,K-1\}\cup\{0\}$ 以避免分布原子。
- **全反馈算法（Algorithm 2）**：并行运行两个 Hedge 实例分别维护 bid/ask 权重 $W_t^A, W_t^B$；每轮从 $w_t^A, w_t^B$ 采样 $Y_t, X_t$ 后，令实际报价 $(A_t,B_t)=(\max\{X_t,Y_t\},\min\{X_t,Y_t\})$ 以保证可行性；关键性质是交换操作不降低效用（Lemma 3.1）。
- **LOB 算法（Algorithm 3）**：使用 EXP3 框架处理部分反馈；定义损失 $\ell_t^s(z)=(1-u_t^s(z))/2$ 并构造无偏估计 $\hat{\ell}_t^s(x)=O_t(x)\ell_t^s(x)/(Q_t(x)+\beta)$，其中 $Q_t(x)=\mathbb{P}(x\in[B_t,A_t]\cup\{0,1\})$ 可从当前分布显式计算（公式3）；隐式探索参数 $\beta$ 控制方差。
- **理论分析框架**：遗憾分解为（1）网格离散误差 $5T/(2K)$；（2）Hedge/EXP3 的标准 regret 项；（3）Azuma-Hoeffding 浓度不等式控制随机性；LOB 情形需额外处理观测概率比 $\sum w_t^s(x)/(Q_t(x)+\beta)\leq 4\log((1+\beta)/\beta)$。
- **不可学习性证明**：通过 Yao's minimax 原理构造随机实例序列 $(L_t,U_t)$ 以因子 $2/3$ 从两端收缩，证明任意确定性算法每轮期望收益上界为 $1/3$，而 hindsight 最优收益至少 $T/2$，从而 regret $\geq T/6$。

## 实验与结果
- **评估框架**：论文为纯理论工作，未包含数值实验；所有结论均以定理形式给出并附完整证明。
- **上界结果**：
  - 全反馈 + (adv)+(stoc)：算法 2 以概率 $1-\delta$ 达到 $R_T = \mathcal{O}(\sqrt{T\log T}+\sqrt{T\log(1/\delta)}+\log(1/\delta))$（Theorem 3.3）。
  - LOB 反馈 + (adv)+(stoc)：算法 3 以概率 $1-\delta$ 达到 $R_T = \mathcal{O}(\sqrt{T}\log(T/\delta))$（Theorem 4.3）。
  - 全反馈 + (adv)+(adv)：任何算法满足 $R_T \geq T/6$（Theorem 5.1）。
- **改进幅度**：相比 Maran & Restelli [2026] 的期望 regret $\widetilde{\mathcal{O}}(T^{2/3})$，本文在高概率意义下将速率从 $T^{2/3}$ 提升至 $\widetilde{\mathcal{O}}(\sqrt{T})$，且放宽对抗者假设（adaptive vs. oblivious）。
- **最优性**：$\widetilde{\mathcal{O}}(\sqrt{T})$ 与多臂老虎机的 $\Omega(\sqrt{T})$ 下界匹配，达到对数因子内的最优。

## 相关工作脉络
- **Cesa-Bianchi et al. [2024]**：研究 two-bit 反馈模型（仅观测是否成交），在 i.i.d. 假设下建立 $T^{2/3}$ 遗憾界；本文区别于该工作在于采用更丰富的 LOB 反馈且价格可由对抗者选择。
- **Maran and Restelli [2026]**：首次引入 LOB 反馈模型并证明期望 regret $\widetilde{\mathcal{O}}(T^{2/3})$（oblivious 对抗者）；本文改进其结果至高概率 $\widetilde{\mathcal{O}}(\sqrt{T})$ 且支持 adaptive 对抗者。
- **Abernethy and Kale [2013]**：早期将市场做市建模为在线学习的先驱工作；本文延续此范式但关注更现实的反馈结构。
- **标准在线学习文献**：Hedge [Freund & Schapire 1997] 与 EXP3 [Auer et al. 2002] 作为核心工具被复用；本文的创新在于将两者适配到具有区间观测结构的做市场景。
- **动态定价与双边交易**：与 Kleinberg & Leighton [2003] 的动态定价、Cesa-Bianchi et al. [2023] 的双边贸易研究同属 "部分可观测经济机制" 这一大类；本文的 LOB 反馈比经典 bandit 更丰富但比全监督更受限。

## 局限性与未来方向
- **仅理论分析**：缺乏数值实验验证算法在实际市场环境中的表现；网格离散化带来的常数项可能与实际问题规模不匹配。
- **单资产假设**：模型仅考虑单一资产做市，未扩展到多资产组合或库存约束场景（论文明确假设头寸可瞬时平仓）。
- **估值分布无正则性要求的双重性**：虽然免除 Lipschitz 假设更具一般性，但也意味着无法利用分布平滑性进一步加速收敛。
- **完全对抗不可学习**：论文证明 (adv)+(adv) 下不可学习，但未探讨中间假设（如估值独立同分布、价格平稳等）下的精细刻画。
- **未来方向**：可研究带库存约束的扩展、非平稳环境下的自适应算法、以及在真实订单簿数据上的实证验证。

## 研究启发与可借鉴点
- **交换不变性分析技术**：Lemma 3.1 证明 swapping 不降低效用的技巧可将耦合约束问题解耦为两个独立专家问题，此思路可迁移到其他带顺序/不等式约束的在线决策场景。
- **区间内全重建的 exploitation**：LOB 反馈下无交易轮次可重建整个区间 $[B_t,A_t]$ 的损失，而非仅端点信息；这种 "区间 bandit" 结构在其他具有区间阈值反馈的问题（如定价、拍卖）中可能同样适用。
- **偏移双格设计**：$\mathcal{G}_B$ 与 $\mathcal{G}_A$ 的 $1/(4K)$ 和 $1/(2K)$ 偏移避免分布原子导致观测退化，这一技巧对处理任意分布（含离散成分）的 bandit 问题具有参考价值。
- **隐式探索参数 $\beta$ 的精细控制**：$\beta=\sqrt{1/T}$ 的选择平衡了方差与偏差，结合 $\sum w/(Q+\beta)\leq O(\log K)$ 的技术可复用于其他部分可观测问题。
- **Yao's minimax 构造法**：通过随机收缩区间 $(L_t,U_t)$ 的证明策略清晰展示了信息不确定性如何转化为线性 regret，可为其他不可学习性的下界证明提供模板。

## 关键术语表
- **Market Making（市场做市）**：做市商通过同时报出买入价（bid）和卖出价（ask）提供流动性，并从买卖价差获利。
- **Limit Order Book（LOB，限价订单簿）**：记录所有待成交买卖订单的市场数据结构，反馈模型中交易者估值仅在无交易时暴露。
- **Regret（遗憾）**：累积实际收益与最优固定策略在 hindsight 下的收益之差，衡量在线算法的学习性能。
- **Hedge / EXP3**：标准的在线学习算法；Hedge 用于全反馈专家问题，EXP3 通过重要性加权处理 bandit 反馈。
- **Adaptive Adversary（自适应对抗者）**：能根据算法历史行为动态选择输入序列的对抗者，比 oblivious 对抗者更强。
- **Implicit Exploration（隐式探索）**：在 EXP3 中通过添加参数 $\beta$ 避免估计值过大，等价于对策略分布施加温和的熵正则化。
- **Grid Discretization（网格离散化）**：将连续价格空间离散化为有限网格以应用在线学习算法，离散误差由 $O(1/K)$ 控制。
- **Yao's Minimax Principle**：用于证明下界的工具，通过构造随机实例分布证明任何确定性算法的最坏情况性能。

## 可复现要素
- **数据集**：论文为纯理论工作，无数据集。
- **代码/权重开源**：论文未提及代码开源。
- **关键超参**：网格大小 $K=\lceil\sqrt{T}\rceil$；学习率 $\eta=\sqrt{\log(T)/T}$（全反馈）或 $\eta=\beta=\sqrt{1/T}$（LOB）；隐式探索参数 $\beta=\sqrt{1/T}$。
