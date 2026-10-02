---
title: "Self-Supervised-Combinatorial-Optimization-with-Constraints"
source: https://arxiv.org/pdf/2609.25728v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:24:58"
---

# 论文速读：Self-Supervised-Combinatorial-Optimization-with-Constraints

## 一句话总结
本文提出了一种免投影的自监督组合优化学习框架，将神经网络输出的任意连续向量经 Frank–Wolfe 几何分解层映射为可行解的稀疏凸组合，构建几乎处处可微的自监督损失，从而在无标签条件下端到端优化含硬离散约束的组合优化问题，并在推理时提供自动取整保证。

## 研究问题与动机
- **硬约束与梯度优化的冲突**：组合优化问题的可行解空间呈离散/指数级规模，传统神经网络难以直接在不违反约束的情况下进行梯度训练。
- **现有连续扩展方法依赖问题特异投影**：如 GeoNCO 等工作虽将离散目标延拓至凸域，但强制要求网络输出必须落在可行多面体内部，并为超单纯形、生成树、Birkhoff 等不同结构分别设计专用投影与分解算法，泛化成本高。
- **罚函数/投影层的梯度失真风险**：硬投影或惩罚项常导致优化路径被截断、梯度信号不均，且需大量调参。
- **缺乏统一的无标签学习范式**：希望在不依赖最优解标签的前提下，让模型直接学习问题实例的结构性模式，同时保持算法层面的可行性保证与推理效率。

## 核心贡献（创新点）
1. **免投影的自监督 CO 统一框架**：网络可输出任意连续向量，无需预先满足可行性，约束由后续分解算法显式保证，替代了传统投影/罚函数设计。
2. **基于 Frank–Wolfe 的几何分解层**：将任意预测映射为至多 $T$ 个可行顶点的稀疏凸组合，每步仅需一次线性最大化预言机（LMO）调用，避免问题特异的多面体投影。
3. **几乎处处可微的自监督损失**：以离散目标的期望值为核心损失，配合可选重构正则项，梯度可自动通过分解步骤回传至网络参数。
4. **推理时自动取整保证**：从分解产生的候选集中选取最优可行解，其目标值不劣于期望值，学习管线与解恢复在同一框架内统一。
5. **多面体通用适配能力**：仅需替换 LMO 实现即可覆盖拟阵、Birkhoff、生成树、匹配等多类结构，统一子suming 先前需分别设计的分解算法。

## 方法详解
- **问题形式化**：设可行解集合 $\mathcal{C}$，凸包多面体 $\mathcal{P}=\text{conv}(\mathcal{C})$。神经网络输出 $\mathbf{x}_\theta \in \mathbb{R}^n$（可位于 $\mathcal{P}$ 外），目标为最大化 $f:\mathcal{C}\to\mathbb{R}$。
- **FW 分解算法（Algorithm 1）**：以最小化 $J(\mathbf{u})=\frac{1}{2}\|\mathbf{x}-\mathbf{u}\|_2^2$ 为目标在 $\mathcal{P}$ 上进行 Frank–Wolfe 迭代。初始化 $\mathbf{u}_0=\mathbf{v}_0\in\text{ext}(\mathcal{P})$；第 $t$ 步计算残差 $\mathbf{r}_t=\mathbf{x}-\mathbf{u}_{t-1}$，通过 LMO 求得 $\mathbf{v}_t=\arg\max_{\mathbf{v}\in\mathcal{P}}\langle\mathbf{r}_t,\mathbf{v}\rangle$，取步长 $\gamma_t=\text{Clip}(0,1,\langle\mathbf{r}_t,\mathbf{v}_t-\mathbf{u}_{t-1}\rangle/\|\mathbf{v}_t-\mathbf{u}_{t-1}\|^2)$，更新 $\mathbf{u}_t=(1-\gamma_t)\mathbf{u}_{t-1}+\gamma_t\mathbf{v}_t$。经 $T$ 次迭代后得到稀疏凸组合 $\mathbf{u}=\sum_{t=0}^{T-1}\alpha_t\mathbf{v}_t$。
- **收敛与可微性定理**：精确 LMO 下近似误差满足 $J(\mathbf{u}_{T-1})-J(\mathbf{u}^\star)\le 2D^2/(T+1)$（$D$ 为多面体直径）；在稳定预言机决策假设下，所选顶点几乎处处局部恒定，权重 $\alpha_t$ 关于 $\mathbf{x}_\theta$ 几乎处处可微，梯度仅通过系数传播。
- **自监督损失**：$\mathcal{L}_{\text{SSL}}(\theta)=-\sum_{t=0}^{T-1}\alpha_t f(\mathbf{v}_t)+\lambda\|\mathbf{x}_\theta-\sum_{t=0}^{T-1}\alpha_t\mathbf{v}_t\|_2^2$（$\lambda\ge0$）。第一项驱动模型优化离散目标期望，第二项抑制预测与分解结果的偏离（非必需但 empirically 提升稳定性）。
- **推理流程**：单次前向传播经分解得到 $T$ 个可行候选解，直接选取目标最优者输出，无需额外搜索或热图后处理。
- **多面体与 LMO 实现**：拟阵多面体用贪心最大权基算法；Birkhoff/匹配多面体用最大权匹配（Hungarian 或简单贪心近似）；生成树多面体用 Kruskal 算法；TSP 应用中将 Christofides 的匹配步骤替换为可学习模块。

## 实验与结果
- **实验设置**：三类问题（MC、QAP、TSP），严格按无 TTO 的一次性前向设置评测，对比涵盖精确求解器、经典启发式与主流神经基线。
- **Maximum Coverage**：在 Random500/1000 与 Rail 数据集上，FWNCO 保持在 Pareto 前沿，推理速度显著快于 GeoNCO；在人工构造的 GreedyTrap adversarial 实例上，覆盖率 59.9%–82.3%，大幅超越贪心（46.3%–70.6%）且逼近最优。
- **Quadratic Assignment Problem**：在 QAPLIB 12–64 规模 15 类基准上取得所有基线（含传统求解器 SM/RRWM/SK-JA 与学习基线 NGM/RGM/SAWT）中的最佳平均 gap（26.4 vs 26.8），且为速度最快的神经方法（单实例 1.56s）；对 $n>64$ 的大实例同样保持合理 gap。QAP 场景下额外引入轻量 Sinkhorn 投影层仅用于数值校准，非理论必需。
- **Traveling Salesperson Problem**：在 Euclidean TSP-50/100/500/1000 上优于 DIMES、RL4CO 等无监督/强化学习基线；接近但略逊于需大量标注的 COExpander；跨规模泛化强（TSP-100/500/1000 相互迁移 gap 变化小）；移除投影略微提升大尺度性能。
- **消融与超参敏感性**：FW 迭代次数 $T$ 增加带来质量提升但收益递减（MC 选 $T=50$、QAP 选 $T=3000$、TSP 选 $T=64$）；移除投影在 MC/TSP 上均略微提升质量并缩短推理时间； entropy regularization 余弦衰减有助于探索。

## 相关工作脉络
1. **GeoNCO 与几何扩展法**（Karalias et al., 2024/2025）：同样利用连续延拓训练离散目标，但强制网络输出落在多面体内，并为超单纯形、生成树等分别设计专用投影与 GLS 分解，计算开销大且需问题定制。
2. **投影/惩罚约束神经网络输出**（DC3、LinsatNet、GLINsat、Sinkhorn 类方法）：通过投影层或硬约束损失保证可行性，易造成梯度截断且需针对特定约束重新推导。
3. **强化学习/指针网络 CO**（Pointer Networks、Sym
