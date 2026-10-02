---
title: "Statistical-Rates-for-Entropic-Optimal-Transport-in-the-Disc"
source: https://arxiv.org/pdf/2609.26647v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-02 01:26:24"
---

# 论文速读：Statistical-Rates-for-Entropic-Optimal-Transport-in-the-Disc

## 一句话总结
本文针对熵正则化最优传输（Entropic OT）在离散-连续测度对下的统计推断问题，建立了经验半对偶势函数与 Barycentric 投影的有限样本收敛率，并首次证明了 Sinkhorn-EM 算法在高斯混合设定下的全局有限样本收缩性。

## 研究问题与动机
- 现有 Entropic OT 理论多聚焦连续-连续情形或仅分析 Sinkhorn 迭代的优化误差，缺乏离散样本估计连续测度 $Q$ 时势函数与传输计划的严格统计收敛率刻画。
- 将 OT 嵌入 EM 框架求解混合模型参数时，Sinkhorn 步骤的统计噪声与 EM 迭代的人口收缩行为尚未被统一量化，缺乏非渐近理论保证。
- 亟需发展一套融合对偶几何、经验过程理论与 Z-估计的分析工具，为正则化 OT 的参数估计提供可验证的样本复杂度上界。

## 核心贡献（创新点）
- **建立经验半对偶势函数的非渐近收敛界**。与已有工作仅关注优化残差或渐近正态性不同，本文首次在离散-连续对 $(P,Q)$ 下显式给出 $\mathbb{E}\|f_n-f\|_\infty^2$ 与 $\mathbb{E}\|g_n-g\|_{L^2(Q)}^2$ 的有限样本上界，明确暴露 $n$ 与 $d$ 的依赖结构。
- **证明 Barycentric 投影误差达到 $\mathcal{O}(d/n)$ 并验证紧性**。区别于此前文献仅给出 $L^2$ 收敛速度，本文同时给出对应下界 $\frac{R^2}{64n}$（Prop 1），证实上界在样本量维度上是匹配的。
- **提出 Sinkhorn-EM 的统一有限样本理论框架**。与仅分析单步 Sinkhorn 稳定性的工作不同，本文通过 Z-估计校准方程将统计噪声项 $\eta_n$ 与人口收缩项 $\kappa^t$ 解耦，导出迭代误差的显式递归界（Theorem 4）。
- **揭示 Sinkhorn-EM 与经典 EM 的人口水平等价性**。在平衡对称高斯混合模型下严格证明两者退化为同一映射，填补了熵正则化参数 $\varepsilon\to0$ 时算法连续性的理论空白。

## 方法详解
- **半对偶势与强凹性几何**：将 entropic OT 对偶问题消元为单变量势函数 $f$ 的优化，施加 gauge 约束 $\mathbb{E}_P f = \frac{1}{2}S(P,Q)$。证明目标泛函 $-\Phi$ 在 $\mathbf
