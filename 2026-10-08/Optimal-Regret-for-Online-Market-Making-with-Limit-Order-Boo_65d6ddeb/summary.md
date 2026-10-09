---
title: "Optimal-Regret-for-Online-Market-Making-with-Limit-Order-Boo"
source: https://arxiv.org/pdf/2610.09691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:51:02"
field: "在线学习与经济机制设计"
keywords: ["online learning", "market making", "limit order book", "regret analysis", "adversarial bandits", "EXP3", "Hedge algorithm"]
innovations: ["LOB反馈下对手市场价格实现最优O~(sqrt(T))高概率regret界", "利用区间重建将EXP3探索代价从O(K)降至O(log K)", "证明全对手设定下即使全反馈也存在线性regret下界"]
---

# 论文速读：Optimal-Regret-for-Online-Market-Making-with-Limit-Order-Book

## 一句话总结
本文在对手选择市场价格的设定下，利用限价订单簿（LOB）反馈模型，将做市商在线学习的 regret 从之前的 $\widetilde{\mathcal{O}}(T^{2/3})$ 提升至最优的 $\widetilde{\mathcal{O}}(\sqrt{T})$；同时证明了当交易者估值也受对手控制时，即使全反馈下也无法获得次线性 regret。

## 研究问题与动机
1. **核心问题**：做市商在线定价学习的最优 regret 率是多少？尤其在市场价格为自适应对手选取、交易者估值为随机分布的设定下，能否达到 $\widetilde{\mathcal{O}}(\sqrt{T})$ 的 regret？
2. **已有工作的不足**：Maran and Restelli [2026] 在 LOB 反馈 + 对手市场价格场景下仅能给出期望 regret 界 $\widetilde{\mathcal{O}}(T^{2/3})$（且对手是 oblivious），存在与随机市场价格场景下 $\widetilde{\mathcal{O}}(\sqrt{T})$ 之间的间隙。
3. **学习性边界问题**：若完全去除估值随机性假设（即 $(adv)+(adv)$ 全对手设定），是否仍可学习？论文证明即使全反馈也不可学习，揭示了随机估值假设的必要性。
4. **动机来源**：LOB 反馈比二值反馈更丰富（无成交时揭示真实估值），但此前未充分利用这一反馈结构来缩小 regret 率 gap。

## 核心贡献（创新点）
1. **全反馈下的最优 regret**：结合两个 Hedge 实例与 bid-ask 空间离散化，在对手市场价格 + 随机估值下实现 $\widetilde{\mathcal{O}}(\sqrt{T})$ 高概率 regret 界；本质区别在于利用"交换报价不降效用"引理将两个维度分离分析。
2. **LOB 反馈下的最优 regret**：首次在对手市场价格下对 LOB 反馈实现 $\widetilde{\mathcal{O}}(\sqrt{T})$ 高概率 regret（优于之前的 $\widetilde{\mathcal{O}}(T^{2/3})$ 期望界）；本质区别在于利用每轮无成交时可获得完整区间的损失信息，将 EXP3 的 exploration 项从 $\mathcal{O}(K)$ 降至 $\mathcal{O}(\log K)$。
3. **全对手设定不可学性证明**：证明在 $(adv)+(adv)$ 全对手设定下即使全反馈也存在 $\Omega(T)$ 线性 regret 下界；揭示随机估值假设是学习可行的必要条件。

## 方法详解
**全反馈算法（Algorithm 2）**：
- 在 $[0,1]$ 上均匀离散化为网格 $\mathcal{G} = \{i/K : i=0,\dots,K\}$，$K=\lceil\sqrt{T}\rceil$。
- 维护两个独立的 Hedge 实例（学习率 $\eta$），分别对 bid 和 ask 价格采样。
- **关键技巧**：若采样的 $X_t$（bid）与 $Y_t$（ask）违反 $A\geq B$ 约束，则交换后出价；**Lemma 3.1** 证明交换后效用不低于原始采样，从而可将两个 Hedge 实例的 regret 界直接叠加到实际动作上。
- 每个网格点可重建全量损失，标准 Hedge regret bound 适用。
- 最终 regret：$R_T \leq \mathcal{O}(\sqrt{T\log T} + \sqrt{T\log(1/\delta)} + \log(1/\delta))$，以概率 $1-\delta$。

**LOB 反馈算法（Algorithm 3）**：
- 使用两组**分离网格**以避免与估值分布的原子重合：$\mathcal{G}_B = \{i/K + 1/(4K)\}$ 与 $\mathcal{G}_A = \{i/K + 1/(2K)\}$。
- 运行两个独立的 EXP3 实例（隐式探索参数 $\beta$）。
- **关键信息结构**：LOF 反馈在以下情况提供丰富信息：
  - 无交易时，观察到真实 $V_t$，可重建 $[B_t, A_t]$ 区间内所有网格点的精确损失。
  - 有交易时，通过买卖方向可知 $V_t$ 在区间的哪一侧，仍可重建部分区间内的损失。
- **Lemma 4.1**：每个网格点 $x \in (\mathcal{G}_A \cup \mathcal{G}_B) \cap [B_t, A_t]$ 均可重建损失 $\ell_t^s(x)$，不依赖 $V_t$ 分布的连续性假设。
- **重要性采样估计器**（Equation 2）：$\hat{\ell}_t^s(x) = \frac{O_t(x)\ell_t^s(x)}{Q_t(x) + \beta}$，其中 $Q_t(x)$ 为点 $x$ 被观测到的概率，由 sampling 分布直接计算（Equation 3）。
- **Lemma 4.2**（核心分析技巧）：$\sum_{x} \frac{w_t^s(x)}{Q_t(x)+\beta} \leq 4\log((1+\beta)/\beta)$，即 LOB 反馈的信息成本仅为 $\mathcal{O}(\log K)$，介于全反馈（$\mathcal{O}(1)$）和 bandit（$\mathcal{O}(K)$）之间。
- 最终 regret：$R_T = \mathcal{O}(\sqrt{T}\log(T/\delta))$，以概率 $1-\delta$。

**不可学性证明**：
- 采用 Yao's minimax principle 构造随机实例序列：每次将可行区间 $[L_t, U_t]$ 随机向左或向右收缩 $2/3$。
- 对手在 $V_t=L_t, M_t=1$ 或 $V_t=U_t, M_t=0$ 中随机选择。
- 任意确定性算法每轮期望效用 $\leq 1/3$，而 hindsight 最优可得 $\geq 1/2$，导致 $R_T \geq T/6$。

## 实验与结果
本文是纯理论工作，**无数值实验**。主要结果对比见论文 Table 1：

| 反馈类型 | 环境设定 $(M_t, V_t)$ | 此前最优 | 本文结果 |
|---------|---------------------|---------|---------|
| 全反馈 | adv + stoc | —（未研究） | $\widetilde{\mathcal{O}}(\sqrt{T})$ |
| 全反馈 | adv + adv | — | $\Omega(T)$（不可学） |
| LOB | adv + stoc | $\widetilde{\mathcal{O}}(T^{2/3})$（期望，oblivious） | $\widetilde{\mathcal{O}}(\sqrt{T})$（高概率，自适应对手） |

- 最强结果：LOB 反馈 + 对手市场价格下，$\widetilde{\mathcal{O}}(\sqrt{T})$ 高概率 regret，相比之前的 $\widetilde{\mathcal{O}}(T^{2/3})$ 期望界有本质提升。
- 该上界与 multi-armed bandit 的 $\Omega(\sqrt{T})$ 下界匹配，达到最优（忽略对数因子）。

## 相关工作脉络
1. **Cesa-Bianchi et al. [2024]**：二值反馈（仅知买卖是否发生）下的做市学习，要求估值分布 Lipschitz 连续；本文不限此光滑性假设，且利用更丰富的 LOB 反馈。
2. **Maran and Restelli [2026]**：提出 LOB 反馈模型，在对手市场价格下获 $\widetilde{\mathcal{O}}(T^{2/3})$ 期望 regret；本文利用反馈结构将 regret 率提升至 $\widetilde{\mathcal{O}}(\sqrt{T})$，且为高概率界、对手可为自适应。
3. **Abernethy and Kale [2013]**：早期将做市视为 online learning 问题的开创性工作；本文在其基础上进一步细化反馈模型的分析。
4. **动态定价（Kleinberg & Leighton [2003]）**：与本文共享"定价"问题框架，但反馈结构和目标函数不同。
5. **双边交易（Cesa-Bianchi et al. [2023]）**：同类经济学中的 online learning 问题，反馈受限，regret 下界为 $\Omega(T^{2/3})$；本文通过 LOB 反馈结构实现了更优率。
6. **FTPL / EXP3**：Maran & Restelli [2026] 主要基于 Follow the Perturbed Leader；本文全反馈场景改为双 Hedge，LOB 场景改为双 EXP3，分析方法完全不同。

## 局限性与未来方向
1. **无数值验证**：论文纯理论，缺少模拟实验验证算法实际表现，无法评估常数因子和对数项的实际影响。
2. **单资产简化**：假设做市商立即在二级市场平仓，无库存约束；实际做市面临库存风险管理，是重要的扩展方向。
3. **二维网格的可行性**：将 bid 和 ask 分离为两个一维问题依赖"交换不降效用"性质，在更复杂的反馈结构下可能失效。
4. **无光滑性假设的代价**：虽然去除了 Lipschitz 条件使结果更通用，但同时也导致离散化误差界较松（$\mathcal{O}(1/K)$）。
5. **可扩展性问题**：未讨论多价位/多数量层级的 LOB 建模，现实 LOB 包含多层报价。

## 研究启发与可借鉴点
1. **区间重建技巧**：LOB 反馈下"无成交揭示完整估值"这一结构可用于将 bandit-style 的 exploration 代价从 $\mathcal{O}(K)$ 降至 $\mathcal{O}(\log K)$，类似思想可迁移至其他区间查询类反馈问题。
2. **分离维度 + 交换引理**：将耦合约束问题分解为独立子问题并用几何/优化性质桥接，是可复用的分析范式。
3. **不依赖分布光滑性的分析**：无需 Lipschitz 假设即可建立 regret bound，适用于估值分布含原子或不连续的鲁棒场景。
4. **Yao's minimax 用于不可学性证明**：构造逐轮缩小区间的随机序列是证明 online learning 问题不可学的有效模板。
5. **分离网格设计**：$\mathcal{G}_B$ 和 $\mathcal{G}_A$ 偏移设计避免与任意分布原子重合，是一种通用的离散化鲁棒技巧。

## 关键术语表
**Market Making（做市）**：做市商通过同时报出买入价（bid）和卖出价（ask）为市场提供流动性，赚取买卖价差利润。
**Regret（遗憾/后悔）**：在线学习算法的累积损失与最佳固定策略在 hindsight 下的损失之差，衡量算法学习性能。
**Limit Order Book（LOB，限价订单簿）**：记录所有未成交买卖订单的有序列表；本文模型中，无交易时完整揭示交易者估值。
**Adaptive Adversary（自适应对手）**：每一轮可根据当前历史动态选择输入序列的对手，比 oblivious adversary（仅预先选定序列）更强。
**EXP3（Exponential weighting for Exploration and Exploitation）**：用于 bandit 问题的经典 online learning 算法，通过 importance weighting 处理部分反馈。
**Hedge Algorithm（Hedge 算法）**：多专家框架下的经典 online learning 算法，通过指数加权更新专家权重实现低 regret。
**Implicit Exploration（隐式探索）**：Neu [2015] 提出的在 EXP3 类算法中通过添加参数 $\beta$ 到分母来隐式探索的技巧，避免显式随机化带来的额外 regret。
**Yao's Minimax Principle（Yao 极小极大原理）**：用于证明 online algorithm 下界的技术，通过构造对手输入的概率分布来下界任意算法的性能。

## 可复现要素
- **数据集**：无数值实验，不存在数据集。
- **代码/权重**：论文未提供开源代码。
- **关键超参**：网格大小 $K = \lceil\sqrt{T}\rceil$，学习率 $\eta = \sqrt{\log(T)/T}$（全反馈）或 $\eta = \beta = \sqrt{1/T}$（LOB）；隐式探索参数 $\beta = \sqrt{1/T}$。
