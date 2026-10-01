---
title: "Steady-State-Convergence-of-Stochastic-Approximation"
source: https://arxiv.org/pdf/2609.14922v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 17:06:09"
field: "强化学习理论 / 非光滑随机近似"
keywords: ["stochastic approximation", "Markovian data", "non-smooth optimization", "steady-state convergence", "asymptotic bias", "Q-learning", "TD learning"]
innovations: ["Markovian 非光滑 SA 的一般稳态收敛定理（Theorem 2）", "$\\sqrt{\\alpha}$ 阶渐近偏置刻画与 RR 外推约化", "高斯噪声普适性原理（Wasserstein O(\\alpha^{1/4}) 归约）"]
benchmarks: ["异步 Q-learning (MDP 4×2)", "Markovian 线性随机近似", "TD(0)/TD(\\lambda) 策略评估"]
---

# 论文速读：Steady-State-Convergence-of-Stochastic-Approximation

## 一句话总结
本文建立了非光滑随机近似（SA）在 Markovian 数据下的**一般稳态收敛（SSC）理论**，首次证明了扩散缩放后平稳分布之差以 $\mathcal{O}(\alpha^{1/4})$ 收敛于高斯极限，并给出了 $\sqrt{\alpha}$ 阶渐近偏置的精确刻画；该理论直接应用于 Markovian TD 学习和异步 Q-learning，填补了非光滑 SA 在序列依赖噪声下的理论空白。

---

## 研究问题与动机

1. **非光滑 SA 的稳态理论缺失**：现有 SSC 结果（如 [ZHCX24]）仅适用于光滑映射或 i.i.d. 噪声；实际 RL 算法（TD、Q-learning）的算子天然非光滑（max、$\ell_1$-norm），且数据来自 Markov 链。
2. **Markovian 噪声下的偏差来源不明**：在非光滑情形下，$\sqrt{\alpha}$ 阶主项偏置 $\mathbb{E}[Y_\infty]$ 未必为零，但缺乏一般性刻画条件。
3. **高斯近似的普适性未确立**：即使噪声为 Markovian，缩放后的稳态分布是否仍逼近高斯？已有工作仅处理 i.i.d. 情形。
4. **RL 算法的理论保证不足**：异步 Q-learning 的 Bellman 算子含 $\max$，属于真正非光滑映射，其稳态收敛速率从未被严格分析。

---

## 核心贡献（创新点）

1. **一般稳态收敛定理（Theorem 2）**：对一类广泛的非光滑映射（单向方向可微、prox-regular、Clarke-regular、$\ell_1$-norm 复合等），证明存在唯一极限分布 $\mathcal{L}(Y_\infty) \in \mathcal{P}_2(\mathbb{R}^d)$，使 $\lim_{\alpha\downarrow0}\mathcal{W}_2(\mathcal{L}(Y_\infty^{(\alpha)}),\mathcal{L}(Y_\infty))=0$；与 [ZHCX24] 的本质区别是将 i.i.d. 噪声推广至 **Markovian 数据**。
2. **$\sqrt{\alpha}$ 阶偏置刻画（Corollary 2）**：给出非光滑情形下渐近偏置非零的**充分条件**——当 $\mathcal{T}$ 在 $\theta^*$ 处单侧方向导数的正齐次延拓 $H_i$ 的 Fenchel 次微分非单点时，$\mathbb{E}[Y_\infty]\neq0$，偏置为 $\mathcal{O}(\sqrt{\alpha})$ 而非 $\mathcal{O}(\alpha)$。
3. **高斯噪声普适性原理（Proposition 4 & 5）**：构造相同均值算子 $\mathcal{T}$、加性 i.i.d. 高斯噪声的辅助递归 $Z_\infty^{(\alpha)}$，证明 $\mathcal{W}_2(\mathcal{L}(A_\infty^{(\alpha)}),\mathcal{L}(Z_\infty^{(\alpha)}))\in\mathcal{O}(\alpha^{1/4})$，从而将 Markovian SA 稳态逼近归约至已知 i.i.d. 高斯结果。
4. **Richardson–Romberg 偏置约化**：提出基于尾平均与多步长外推的偏置消去方法，三步长 $\alpha,2\alpha,4\alpha$ 可同时消除 $\sqrt{\alpha}$ 与 $\alpha$ 两阶偏置项。
5. **Markovian LSA 的高斯近似（Corollary 3）**：首次为 $\theta_{t+1}=\theta_t+\alpha(\mathsf{A}(x_t)\theta_t+s(x_t))$ 类问题建立稳态高斯极限及 Lyapunov 方程 $\overline{\mathsf{A}}V_{\text{LSA}}+V_{\text{LSA}}\overline{\mathsf{A}}^\top+\Sigma_{\text{LSA}}=0$ 的精确刻画。

---

## 方法详解

### 一类非光滑映射框架
定义映射 $F:\mathbb{R}^d\to\mathbb{R}^d$ 属于**单向方向可微（one-sided directionally differentiable）**类，当且仅当对每个 $x$ 和方向 $v$，极限
$$F'(x;v)=\lim_{t\downarrow0}\frac{F(x+tv)-F(x)}{t}$$
存在且 $v\mapsto F'(x;v)$ 为正齐次映射。该框架包含：
- $g\circ F$ 复合函数（$g$ 光滑，$F$ 非光滑）[Sha03, Sag13]
- prox-regular 函数 [PR96]
- Clarke-regular 函数 [Cla90]
- 谱映射（最大特征值）、$\ell_1$-norm 及其与光滑变换的复合

### 定理 2 证明架构（三阶段）
**阶段一（Decoupling）**：引入 Wasserstein-$p$ 解耦引理（Lemma 1），将 Dedecker–Prieur $\tau$-耦合从 $p=1$ 推广至任意 $p\geq1$，得到
$$\mathcal{W}_p(\mathcal{L}(S_m^{(n)}),\mathcal{L}(Z_m^{(n)}))\in\mathcal{O}(n^{-1/2})$$
其中 $S_m^{(n)}$ 为 Markov 块和，$Z_m^{(n)}$ 为独立高斯块和。

**阶段二（Coupling）**：构造概率空间 $(\widetilde\Omega,\widetilde{\mathbb{P}})$ 上的耦合 $(Y,Y^*)$，使条件律 $\mathcal{L}((Y,Y^*)|\mathcal{G})=\pi_\omega$ 为最优输运计划，且 $Y^*\perp\mathcal{G}$、$\mathcal{L}(Y^*)=\mathcal{L}(Y)$。

**阶段三（Moreau Envelope Iteration）**：定义 $V_t=\mathbb{E}[M_\eta(\Delta_{nt})]$（$M_\eta$ 为 Moreau 包络），利用引理 A.1 给出递推界（式 F.8）：
$$V_{t+1}\leq(1-\alpha)^{2n}V_t+(1-\alpha)^nT_1+\frac{1}{2\eta}T_2$$
其中 $T_1\leq2\sqrt{\gamma}\,\alpha n V_t+C\alpha$、$T_2\leq C\alpha$。

### 偏置刻画（Corollary 2）
在假设 1–3、5 下，若存在坐标 $i$ 使 $H_i$（$\mathcal{T}$ 的单侧方向导数延拓）在 $0$ 处 Fenchel 次微分非单点，则 $\mathbb{E}[Y_\infty]\neq0$。该条件精确刻画了**真正非光滑 regime**：$\mathcal{T}$ 在 $\theta^*$ 不可微且 $H$ 非线性，偏置由 $\sqrt{\alpha}$ 主导而非 $\alpha$。

### 高斯普适性（Proposition 4）
定义辅助递归 $Z_\infty^{(\alpha)}$ 对应高斯噪声 $w_t\sim\mathcal{N}(0,\Sigma_h)$，证明：
$$\mathcal{W}_2(\mathcal{L}(A_\infty^{(\alpha)}),\mathcal{L}(Z_\infty^{(\alpha)}))\in\mathcal{O}(\alpha^{1/4})$$
结合 [ZHCX24] 的 i.i.d. 结果与三角不等式，直接得到 Theorem 2。

---

## 实验与结果

### 异步 Q-learning 数值实验（Section I）
| 参数 | 值 |
|---|---|
| MDP 状态数 $|\mathcal{S}|$ | 4 |
| 动作数 $|\mathcal{A}|$ | 2 |
| 折扣因子 $\rho$ | 0.9 |
| 行为策略 $\pi_b(a|s)$ | 1/2（均匀） |
| 步长集合 $\alpha$ | $\{0.05, 0.10, 0.20, 0.40\}$ |
| 初始 $Q_0(s,a)$ | $-5$（全状态–动作对） |
| 迭代数 $T$ | $10^7$ |
| 奖励结构 | $r(s,a)=r_S(s)+r_\mathcal{A}(a)$；无平局 $r_\mathcal{A}=(0.75,0.10)$；平局 $r_\mathcal{A}=(0.75,0.75)$ |
| 转移矩阵 $P_0$ | 对角 0.92，相邻 0.06，其余 0.01 |
| $r_S$ | $(0.05, 0.15, 0, 0.20)$ |

**关键发现**：
- 在**无平局**设置下，Bellman 算子 $\mathcal{T}$ 在 $q^*$ 处单侧方向可微，次微分在最优动作唯一时为非单点集（满足 Corollary 2 条件），观测到 $\sqrt{\alpha}$ 阶偏置。
- 在**平局**设置下，次微分为单点集，偏置降至 $\mathcal{O}(\alpha)$ 阶，与理论预测一致。
- 三步长 RR 外推（$\alpha,2\alpha,4\alpha$）有效消去 $\sqrt{\alpha}$ 主项偏置，残差收敛至 $\mathcal{O}(\alpha)$。

**最强结果**：$\mathcal{W}_2(\mathcal{L}(A_\infty),\mathcal{L}(Z_\infty))\leq C\alpha^{1/4}$，其中 $C$ 仅依赖 $\Sigma_h$ 与 Lipschitz 常数，不依赖 Markov 链混合系数。

---

## 相关工作脉络

1. **[ZHCX24]**：唯一无需 $\tau$ 在 $\theta^*$ 处局部可微的一般 SSC 结果，但仅限 i.i.d. 噪声；本文将其推广至 Markovian 数据，是本文最直接的先作。
2. **[Sha03, Sag13]**：单向方向可微映射的复合函数理论，为本文非光滑框架提供基础工具。
3. **[PR96]**：prox-regular 函数理论，用于刻画非光滑优化的稳态行为。
4. **[Cla90, HUL04]**：Clarke 正则性与次微分计算规则，支撑非光滑 SA 的收敛分析。
5. **[BM20]**：标准 Borel 空间可测分裂定理（Corollary 4.4），在耦合构造中被引用。
6. **Rockafellar (1970)**：次微分求和规则，用于 Section H 非光滑分析中验证异步 Q-learning 算子的单侧方向可微性。
7. **[TVR97, BRS21, SY19, MPWB24]**：TD(0)、TD($\lambda$) 等策略评估算法，为本文 LSA 应用提供实际动机。

---

## 局限性与未来方向

1. **混合系数依赖**：Wasserstein 解耦引理的常数依赖于 Markov 链的 $\beta$-混合系数，对长程依赖链（慢混合）可能较紧。
2. **高维扩展未验证**：理论对 $d\gg1$ 的情形未给出显式维度依赖，实际 RL 中的高维价值函数近似（如神经网络）仍需进一步分析。
3. **非平稳噪声**：本文假设平稳遍历 Markov 链，对非平稳或部分可观测环境（POMDP）尚不适用。
4. **自适应步长**：理论仅处理固定步长 $\alpha$，对衰减步长（$\alpha_t\propto t^{-\gamma}$）的稳态行为未覆盖。
5. **优化视角**：本文关注估计（SA）而非优化（随机近似 + 优化），对非光滑损失的最小化问题需额外分析。

---

## 研究启发与可借鉴点

1. **解耦归约范式**：将 Markovian 噪声下的稳态收敛归约至 i.i.d. 高斯噪声的已知结果，通过 Wasserstein 距离控制误差，该范式可迁移至其他序列依赖建模问题（如隐马尔可夫决策过程）。
2. **Moreau 包络迭代技术**：式 (F.8) 的递推界是处理非光滑递归的通用工具，可复用于其他非光滑随机迭代算法的收敛分析。
3. **RR 外推偏置约化**：三步长 $\alpha,2\alpha,4\alpha$ 同时消去两阶偏置的设计，对任何具有已知偏置展开的算法（如 SGD、Adam）均可借鉴。
4. **非光滑性诊断条件**：Corollary 2 的"次微分非单点→$\sqrt{\alpha}$ 偏置"判据，可作为后续研究快速判断算法偏置阶的实用工具。
5. **与团队方向结合机会**：若本团队研究高维非光滑强化学习（如稀疏策略、鲁棒控制），本文的非光滑 SA 理论可直接用于分析价值函数近似的稳态误差。

---

## 关键术语表

**单向方向可微（one-sided directionally differentiable）**：映射 $F$ 在每点沿每个方向的方向导数存在且为正齐次映射，是非光滑分析的核心正则性条件。

**稳态收敛（steady-state convergence, SSC）**：描述随机迭代 $\theta_{t+1}=\theta_t+\alpha F(\theta_t)+\sqrt{\alpha}\xi_t$ 在扩散缩放 $(\theta_t-\theta^*)/\sqrt{\alpha}$ 下平稳分布的收敛行为。

**Wasserstein-$p$ 距离（$\mathcal{W}_p$）**：测度空间上的最优输运距离，本文用 $\mathcal{W}_2$ 刻画稳态分布的高斯逼近误差。

**Moreau 包络（Moreau envelope）**：非光滑函数 $f$ 的光滑近似 $M_\eta(f)(x)=\inf_y\{f(y)+\frac{1}{2\eta}\|x-y\|^2\}$，用于将非光滑递归转化为可分析的光滑迭代。

**Richardson–Romberg 外推（RR）**：通过多步长模拟结果的线性组合消去渐近展开中的主导误差项，本文用于 $\sqrt{\alpha}$ 阶偏置约化。

**Clarke 正则（Clarke-regular）**：非光滑函数的次微分满足某种上半连续性条件，是保证方向导数良好行为的正则性假设。

**prox-regular 函数**：比 Clarke 正则更弱的非光滑性假设，包含大多数机器学习中出现的损失函数（如 $\ell_1$-norm、ReLU 复合）。

**渐近偏置（asymptotic bias）**：$\mathbb{E}[\theta_\infty^{(\alpha)}]-\theta^*$ 在 $\alpha\downarrow0$ 时的主导项，非光滑情形下为 $\sqrt{\alpha}\,\mathbb{E}[Y_\infty]$ 而非 $\alpha$ 阶。

---

## 可复现要素

| 要素 | 状态 |
|---|---|
| 数据集 / MDP | 自构造小规模 MDP（4 状态、2 动作），论文提供完整参数（Section I） |
| 代码 | 论文未提及开源仓库 |
| 权重 / 模型 | 不适用（理论论文） |
| 关键超参 | $\alpha\in\{0.05,0.10,0.20,0.40\}$，$T=10^7$，$\rho=0.9$，初始 $Q_0=-5$ |
| 复现难度 | 高（需实现非光滑 SA 求解器及 Wasserstein 距离估算） |

---
