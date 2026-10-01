---
title: "Thompson-Sampling-for-Non-Monotone-Convex-Ridge-Bandits-Mono"
source: https://arxiv.org/pdf/2609.10981v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:56:29"
field: "Bandit Convex Optimization"
keywords: ["Thompson Sampling", "Bandit Convex Optimisation", "Ridge Loss", "Information Ratio", "Bayesian Regret", "Non-monotone Link", "Convex Geometry"]
innovations: ["证明非单调凸脊损失下 TS 仍享有多项式贝叶斯后悔界 Õ(d^{9/2}√n)", "提出布尔舍入引理将近似低秩矩阵舍入为恰好低秩的 0-1 矩阵以替代 John 椭球二分局", "构造 infimal-convolution 覆盖实现任意固定可测选择规则下的信息比-后悔传递"]
benchmarks: ["Theoretical regret bound: O(d^{9/2}√n log(ednD))", "Monotone baseline: O(d^{5/2}√n log^2(ednD)) from Bakhtiari et al. COLT 2025"]
---

# 论文速读：Thompson-Sampling-for-Non-Monotone-Convex-Ridge-Bandits-Mono

## 一句话总结
本文证明了在**任意凸非单调链接**的脊损失（convex ridge losses）设定下，精确后验 Thompson Sampling（TS）的贝叶斯后悔界为 $\tilde{O}(d^{9/2}\sqrt{n})$，从而回答了 Bakhtiari 等人（COLT 2025）提出的开放问题：**单调性不是 TS 获得多项式后悔的必需条件**。

## 研究问题与动机
- **核心问题**：在贝叶斯赌注凸优化（bandit convex optimisation）中，Bakhtiari 等人已证明单调凸脊损失下 TS 达到 $O(d^{5/2}\sqrt{n}\log^2(ndD))$ 后悔界；但若移除单调性假设，TS 是否仍能保持多项式后悔？
- **单调性的必要性争议**：单调链接下，脊损失 $f(x)=\ell(\langle x,\theta\rangle)$ 的最优解沿 $\theta$ 方向序贯排列，可借助 John 椭球二分局实现降维分析；非单调链接（如 $\ell(s)=|s-s_0|$）的最优解集合可能是高维超平面截面，破坏该几何结构。
- **TS 对一般凸损失脆弱**：现有文献表明 TS 在一般凸损失下可能遭遇灾难性后悔，因此"结构足以使 TS 安全"是一个值得追问的统计问题，而非最小化最优性问题。
- **量化差距悬而未决**：本文证明多项式后悔成立，但维度指数从 $5/2$ 退化到 $9/2$；能否对齐单调情形的指数仍为开放问题。

## 核心贡献（创新点）
1. **无单调性的 TS 后悔界**（定理 3.1）：对 $\mathcal{F}_{\mathrm{blr}}$ 上任意先验与任意固定可测选择规则，TS 满足 $\mathrm{BReg}_n(\mathrm{TS},\xi)=\tilde{O}(d^{9/2}\sqrt{n})$，**定性上**消除了单调性必要性。
2. **无信息配置的 $O(d^2)$ 基数上界**（定理 3.3）：引入布尔舍入引理（引理 5.3：max-范数误差 $\leq 1/(4r)$ 的 0-1 矩阵秩至多为 $2r-1$），绕过 John 椭球二分局，直接得到无信息配置元素数 $\leq 3(d+1)^2-(d+1)-1$。
3. **构造十二点反例，证伪单调情形的单一删除 John 椭球二分局**（命题 3.6）：展示对于任意固定收缩因子 $\gamma<1$，非单调情形下"删除一个函数使 John 椭球缩小"的结论不成立。
4. **任意固定可测选择规则下的信息比-后悔界传递定理**（定理 7.5）：利用 infimal-convolution 逼近构造覆盖，将信息比下界自包含地传递给任意固定选择规则的 TS，历史相关的 tie-breaking 规则不在覆盖范围内。
5. **基数上界的紧性**（命题 3.5）：在截断单纯形上构造出大小为 $d(d+1)$ 的无信息配置，证明上界在大直径/间隙比 regime 下紧致至常数因子。

## 方法详解
**1. 脊表示与两带结构（Section 4）**
- 每个 $f\in\mathcal{F}_{\mathrm{blr}}$ 可写为 $f(x)=\ell(\langle x,\theta\rangle)$，其中 $\ell$ 凸、1-Lipschitz，且在 $\mathbb{R}$ 上全局最小化于最小值点投影处（引理 4.1）。
- 若行 $f$ 是无信息的，则沿 $\theta_f$ 方向，其余所有最小值点 $x_g$ 的投影 $t_g=\langle x_g,\theta_f\rangle$ 被压缩进两个窄带：近带 $[a,\, a+\eta L)$ 与远带 $(b-\eta R,\, b]$，每个相对宽度 $\eta\approx 6/(c(d+1))$（引理 4.2）。单调情形仅一个带非空，非单调时两个带均可非空，这是全部困难来源。

**2. 平衡/非平衡行分解与线性代数计数（Section 5）**
- 按两带距离比 $\lambda_i=L_i/R_i$ 划分：平衡行 $I_B=\{\lambda_i>1/(8h)\}$，非平衡行 $I_U=\{\lambda_i\leq 1/(8h)\}$，其中 $h=d+1$。
- **平衡行**：构造二次评价矩阵 $P_{ij}=p_i(x_j)$，对角线为 1，非对角线绝对值 $\leq t_B=8h\eta$，秩 $\leq h(h+1)/2$。应用 trace-rank 不等式（引理 5.1）得 $|I_B|\leq h^2-1$。
- **非平衡行**：构建近带隶属 0-1 矩阵 $H$，并证明 $H$ 在 max-范数下距秩 $\leq h$ 的仿射评价矩阵 $A$ 的误差 $\|A-H\|_{\max}\leq 1/(8h)$。由布尔舍入引理（引理 5.3），$\operatorname{rank}(H)\leq 2h-1$；再构造掩码矩阵 $M=U\circ H$（Hadamard 积，引理 5.2），其秩 $\leq h(2h-1)$，进而由引理 5.1 得 $|I_U|\leq 2h^2-h$。
- 合计 $|C|\leq 3h^2-h-1 = O(d^2)$。

**3. 计数到信息比（Section 6）**
- 引理 6.1：若无信息配置最大尺寸为 $G$，则任何超过 $G$ 元的配置中有序对能量 $\sum (f(x_g)-\bar{f}(x_g))^2\geq q\delta^2$（迭代删除 $q$ 次）。
- 按层 $\varepsilon_i=\alpha 2^i$ 分割 $\mathcal{F}_{\mathrm{blr}}$ 成 $m_\alpha$ 份，每份内任取 $k=bs=3072h^4$ 元元组，利用块切割论证导出能量下界 $\varepsilon^2/12$。
- 结合分解引理（引理 2.1）得到 $(\alpha,\,12k(k-1)m_\alpha)\in\mathrm{IR}(\mathcal{F}_{\mathrm{blr}})$，即 $\beta=O(d^8\log(1/\alpha))$（定理 3.4）。

**4. 信息比到后悔的传递（Section 7）**
- 构造 infimal-convolution 逼近 $g_f(x)=\min_{y\in K}[f(y)+|\langle\theta_\kappa,x-y\rangle|]$，使 $g_f\in\mathcal{F}_{\mathrm{blr}}$、$\|f-g_f\|_\infty\leq\rho/2$，且原最小值点 $x_f$ 仍是 $g_f$ 的精确最小值点（引理 7.4）。
- 利用均匀覆盖大小 $N_\rho\leq(1+2D/\rho)^d(1+8D/\rho)^d$ 及信息比假设，得到通用传递不等式（定理 7.5）：
$$\mathrm{BReg}_n(\mathrm{TS},\xi)\leq n\alpha+n\rho(6+4\sqrt{\beta})+\sqrt{\tfrac12\beta n\log N_\rho}.$$
- 代入 $\alpha=\rho=1/n$、$\sqrt{\beta}=O(d^4\sqrt{\log n})$、$\log N_{1/n}=O(d\log(e+ndD))$ 完成组装（定理 3.1）。

## 实验与结果
- 本文为**纯理论论文**，不含数值实验；主要"实验"为构造性结果：
  - 定理 3.1 给出显式常数：$\mathrm{BReg}_n(\mathrm{TS},\xi)\leq 7+73\,728\sqrt{3}\,(d+1)^4 d^{1/2}\sqrt{n}\log(e+nd\max\{1,D\})$。
  - 定理 3.3 给出紧致的 $O(d^2)$ 基数上界，命题 3.5 在截断单纯形上构造 $d(d+1)$ 元无信息配置，证明上界至常数倍紧。
  - 命题 3.6 用 12 点正四面体构型证明 John 椭球单删二分局在非单调情形失效。
- 与基线（Bakhtiari et al., COLT 2025）对比：单调情形 TS 后悔为 $O(d^{5/2}\sqrt{n}\log^2(ndD))$；本文放宽单调性后退化至 $\tilde{O}(d^{9/2}\sqrt{n})$，维度指数增加 2 倍。

## 相关工作脉络
- **Bakhtiari, Lattimore & Szepesvári (COLT 2025)**：首次分析精确后验 TS 在凸脊损失上的后悔，证明单调情形 $O(d^{5/2}\sqrt{n})$，并指出 TS 在一般凸损失下可能灾难性失败；本文延续其框架但移除单调性假设。
- **Lattimore (2021, arXiv:2106.00444)**：对对手脊模型给出 $O(d\sqrt{n}\log(nD))$ 最小最大后悔界，使用不同算法与信息论论证，关注可学习性而非 TS 本身的稳定性。
- **Rajaraman et al. (Ann. Stat. 2024; ICLR 2026)**：研究非参数脊/单指标赌注的频繁主义 regret，使用 SGD 类算法，与贝叶斯 TS 框架不同。
- **Kang et al. (ICLR 2026)**：广义线性上下文赌注的单指标模型研究，同样采用非 TS 算法。
- **Russo & Van Roy (JMLR 2016)**：信息比率分析 TS 的开创性工作；本文将其从离散 arm 推广至凸脊连续臂设定。
- **Bubeck et al. (COLT 2015); Bubeck & Eldan (2018)**：凸脊优化的探索分布与信息比工具链来源。

## 局限性与未来方向
- **维度指数 $9/2$ 未必最优**：信息比证书规模 $O(d^8)$ 远大于单调情形的 $O(d^4)$，作者指出这可能是分析过程中的损耗而非真实下界；能否保留 $d^{5/2}$ 依赖是核心开放问题。
- **仅覆盖固定选择规则**：tie-breaking 必须在开始前固定且不可依赖历史；历史相关的选择规则未被覆盖。
- **未涉及已知链接的信息比问题**：同一段落中另一开放问题——"已知凸链接"下的信息比——未处理。
- **计算复杂性未讨论**：精确后验 TS 可能难以有效采样，算法侧的高效实现仍是开放问题。
- **布尔舍入容忍度可能非最优**：引理 5.3 的 $1/(4r)$ 对任意 0-1 矩阵是紧的（命题 E.1）；但本文中的带隶属矩阵具有额外结构（两带，一极窄），该结构是否能支持更大容忍度未知。

## 研究启发与可借鉴点
- **布尔舍入引理（引理 5.3）**：将近似低秩实矩阵舍入为恰好低秩的 0-1 矩阵的技术，误差容忍度 $1/(4r)$ 仅依赖秩，不依赖矩阵维度，可作为其他高维组合几何计数问题的通用工具。
- **infimal-convolution 覆盖构造**：利用 $g_f(x)=\min_y[f(y)+|\langle\theta,x-y\rangle|]$ 使逼近函数与被逼近函数共享精确最小值点，这种"值-preserving + 最小值-preserving"的覆盖技巧在连续臂 bandit 理论中值得复用。
- **无信息配置的计数策略**：将配置按"两带距离比"拆分为平衡/非平衡两类，分别用二次型评价矩阵与布尔掩码低秩论证处理，是克服非单调性带来的几何复杂度的经典范例。
- **与团队方向结合机会**：若团队研究 contextual bandit / ridge bandit，本文关于"结构充分性"的定性论证路径（即仅靠脊结构即可保障 TS 稳定性，无需单调）可直接迁移；同时布尔舍入引理可考虑用于其他非凸约束下的选择规则分析。
- **信息比传递的自包含化**：本文摆脱了原始传递定理对"一致 tie-breaking"的隐含依赖，改用 infimal-convolution 逼近绕开；对任何需要处理非唯一最小值的 TS 分析都有方法论参考价值。

## 关键术语表
- **Bandit convex optimisation**：学习者在每轮从凸集 $K$ 中选择点 $x_t$，观测含噪声函数值 $f(x_t)$，目标是极小化累积后悔的序贯优化框架。
- **Convex ridge loss**：形如 $f(x)=\ell(\langle x,\theta\rangle)$ 的凸函数，其中 $\ell$ 为单变量链接函数，$\theta$ 为方向向量；也称为单指标函数。
- **Thompson sampling (TS)**：贝叶斯在线决策算法，每轮从后验分布采样一个模型 $f_t$，并选取该采样函数的最小值点作为行动。
- **Bayesian regret**：对先验 $\xi$ 期望意义下的累积后悔 $\mathbb{E}[\sup_{x\in K}\sum_{t=1}^n(f(X_t)-f(x))]$。
- **Information ratio**：衡量算法每单位信息获取的即时后悔代价，形式为 $(\Delta,\sqrt{I})$ 平面对；信息比有界可推出后悔上界。
- **Uninformative configuration**：一组函数构成的集合，其中每个函数的最优解对其他函数完全不提供信息（各函数在他人最优解处的值与其后验均值之差有界）。
- **John ellipsoid**：凸体的最大体积内切椭球，用于刻画凸体几何性质；单调情形中用作分析工具。
- **Infimal convolution**：函数 $f$ 与核 $x\mapsto|\langle\theta,x-y\rangle|$ 的 infimal 卷积，构造出保持最小值点不变的逼近函数。

## 可复现要素
- **数据集**：不适用（纯理论论文）。
- **代码/权重**：论文未开源代码（截至发表）；所有结果均为解析证明。
- **关键超参**：$c=96(d+1)$（控制无信息配置定义中的容忍参数）、$k=3072(d+1)^4$（块大小）、$\alpha=\rho=1/n$（传递定理参数）。
- **复现难度**：高；需深入掌握凸几何、信息比理论与高维矩阵秩不等式。
