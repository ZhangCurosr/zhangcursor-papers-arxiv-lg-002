---
title: "Where-Do-Two-Populations-of-Persistence-Diagrams-Difer-Calib"
source: https://arxiv.org/pdf/2610.08292v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-09 10:45:47"
---

# 论文速读：Where-Do-Two-Populations-of-Persistence-Diagrams-Difer-Calib

## 一句话总结
本文针对持久图双样本检验仅能做全局比较的局限，在固定总样本预算下提出一种同时推断方法，通过对多重 $\ell_\infty$ 邻域的中心/半径同步构建联合置信带，实现 birth–death 平面中差异区域的局部定位与严格 FWER 控制。

## 研究问题与动机
- 现有持久图双样本检验（如瓶颈距离、Wasserstein 型检验）仅输出全局是否差异的二元结论，无法回答“差异集中在 birth–death 平面的哪些空间区域”。
- 固定样本预算下，若将所有观测用于探索性网格/半径搜索，缺乏多重比较校准会导致假阳性膨胀；若切分样本用于探索与推断，则检验力下降。
- 纯寿命统计量沿对角线方向平移（point-wise shift）的检验效力恒等于显著性水平，对对角线移位完全“失明”，亟需引入同时感知空间位置与寿命信息的局部对比量。
- 已有 Bootstrap 联合推断方法多假设两总体协方差相同，实际拓扑数据常呈现异方差，需更灵活的渐近校准机制。

## 核心贡献（创新点）
- **同时局部均值对比推断框架**：在固定预算下对多个 $\ell_\infty$ 邻域中心/半径同步构建联合置信带，将全局检验升级为可定位的差异热力带。
- **数据依赖子集的覆盖保留保证**（Theorem 1 / Corollary 2）：证明从同时带中报告任意数据依赖的子集仍保留联合覆盖概率，且被报告邻域中心距真实差异支撑集的距离 ≤ 半径。
- **不等协方差的高斯乘子 Bootstrap**：推导标准化差的协方差 $\Gamma = \frac{m'}{N}\Sigma_P + \frac{m}{N}\Sigma_Q$，允许两总体异方差，通过乘子重采样精确估计临界值 $\hat c_\alpha$。
- **Pilot 分半最优半径选择机制**：将样本分 fitting/validation 两部分，基于 $\hat J(h)$ 从多半径族中选取最优倍数 $h$；理论证明当族规模较小时（如 $\ell_h=105$ vs $\ell=497$），仅需 $f>0.15$ 全观测直接搜索即占优，揭示 pilot 代价阈值。
- **纯寿命统计量对角线平移盲区的理论刻画**（Proposition 3）：严格证明任何仅依赖寿命多集合的统计量在该变换下的检验效力等于显著性水平，从理论上划清本方法与既往寿命检验的本质边界。

## 方法详解
- **响应函数设计**：对每个地标 $p_k$ 定义 $\ell_\infty$ 方块响应 $\varphi_k(a) = w_k(r_k - d_B(p_k,a))_+$，其中 $d_B$ 为 Betti 距离（$\ell_\infty$ 型），$\Phi_k(D) = \int \varphi_k \, dN_D$ 统计持久图中落入该方块的加权点数。
- **局部对比量**：$\delta_k = \mathbb{E}_P \Phi_k(X) - \mathbb{E}_Q \Phi_k(Y) = \int \varphi_k \, d(\bar\mu_P - \bar\mu_Q)$，刻画两总体在 $(p_k, r_k)$ 处的期望密度差异。
- **标准化与渐近协方差**：基于有效样本量 $N_{\mathrm{eff}} = mm'/N$，构造 $\sqrt{N_{\mathrm{eff}}}(\bar U - \bar V)$，其渐近协方差矩阵为 $\Gamma = \frac{m'}{N}\Sigma_P + \frac{m}{N}\Sigma_Q$。
- **同时置信带与拒绝规则**：对每个 $k$ 构造 $\mathcal{C}_k = [\hat\delta_k \pm \hat c_\alpha \hat\omega_k/\sqrt{N_{\mathrm{eff}}}]$，构建拒绝集 $\hat R = \{k: |Z_k| > \hat c_\alpha\}$。
- **多半径策略**：同一中心取 $r = \min\{hr_0(p), \pi(p)\}$，$h \in \mathcal{H}$；共享半径只保留一次以避免冗余比较。
- **Pilot 流程**：样本分为 fitting / validation 两部分；拟合集按式(5)的 $\hat J(h)$ 选最优半径倍数；选定族后用验证集做正式推断。
- **检测阈值与定位保证**：Proposition 1 给出充分检测条件；Theorem 2 证明在局部位移模型下，若邻域完全覆盖非零差异支撑，则 $\delta_k \ge qw_k(r_k - d_B(p_k,a) - \epsilon) > 0$。

## 实验与结果
- **实验设置**：$\alpha=0.05$，固定总预算 $N$；模拟含 20 个背景点 + 概率 $q=0.5$ 的特征点叠加，模板中心 $a=(0.30, 0.50)$。
- **五种地标过程**：F0（$14\times14$ 网格，单半径 $r=0.05$，全观测）；F1（同上中心，5 倍半径 $h\in\{\tfrac12,\tfrac1{\sqrt2},1,\sqrt2,2\}$，全观测）；F2（F1 中心 + pilot 选半径）；A1（class-aware farthest-point 拟 100 中心 + 5 半径全用）；A2（A1 + pilot 选半径）。
- **FWER 与覆盖**：1000 次重复下，所有地标过程 FWER 控制在 **0.022–0.063**，同时覆盖概率达 **94%–98%**。
- **定位性能**：使用 pilot-fitted landmarks 且 $K \ge 100$ 时，对效应量 $\Delta \ge 0.10$、单组样本 $m \ge 40$，band 能稳健定位差异区域（基于第 4 段截断信息）。
- **Pilot 代价验证**：理论推导与数值一致，当 $\ell_h$ 较小且 $f>0.1
