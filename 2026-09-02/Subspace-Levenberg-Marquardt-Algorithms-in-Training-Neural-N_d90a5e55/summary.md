---
title: "Subspace-Levenberg-Marquardt-Algorithms-in-Training-Neural-N"
source: https://arxiv.org/pdf/2609.00789v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:03:45"
field: "神经网络优化算法"
keywords: ["Levenberg-Marquardt", "subspace optimization", "Krylov subspace", "neural network training", "second-order optimization", "HSLM", "KSLM", "nonlinear least squares"]
innovations: ["HSLM自适应多源子空间构建结合曲率相关对角阻尼，在LM类方法中实现更低计算开销与更高收敛稳定性", "引入子空间充分性度量η_sub作为质量监控器，动态扩展子空间以保证下降方向有效性"]
benchmarks: ["Damped Oscillation Regression", "13-bit Parity Classification"]
---

# 论文速读：Subspace-Levenberg-Marquardt-Algorithms-in-Training-Neural-N

## 一句话总结
本文系统评估了 Krylov 子空间 LM (KSLM) 和混合自适应子空间 LM (HSLM) 在神经网络回归与分类任务中的性能，结果表明 **HSLM 在保持 LM 快速收敛优势的同时，显著降低了计算与内存开销**，在两项基准任务中均展现出优于 SGD/Adam 及其他 LM 变体的综合性价比。

## 研究问题与动机
- **经典 LM 无法扩展**：LM 需要显式构建并求解 $n \times n$ 的 Gauss-Newton 系统 $(\mathbf{J}^\top \mathbf{J} + \mu \mathbf{I}) s = -\mathbf{g}$，参数规模增大时计算量和内存需求急剧上升。
- **一阶方法迭代代价高**：SGD 和 Adam 每次迭代成本低但收敛慢，且在条件较差的优化地形上高度依赖超参数（学习率、动量）。
- **KSLM 仍有额外开销**：Krylov 子空间方法虽降低了维度，但每次迭代仍需构造并扩展 Krylov 基，且仅利用单一来源的曲率信息。
- **缺乏系统性对比**：现有工作多关注 KSLM，对更高效的 HSLM 在神经网络训练场景下的实证评估不足。

## 核心贡献（创新点）
- **系统评估三种 LM 变体在 NN 训练中的表现**：在回归（阻尼振荡拟合）和分类（13-bit 奇偶校验）两个标准基准上，完整比较了 LM、KSLM、HSLM 与 SGD/Adam，填补了子空间 LM 应用于 NN 训练的实证空白。
- **HSLM 自适应多源子空间构建**：融合归一化梯度、历史优化步、Krylov/Lanczos 向量和随机 Gauss-Newton 探测向量，比 KSLM 单一 Krylov 基能更全面捕捉局部曲率。
- **引入子空间充分性度量 $\eta_k^{\mathrm{sub}}$**：通过梯度能量捕获比例判定子空间是否"足够好"，不满足阈值时自动扩展，保证下降方向质量。
- **曲率相关对角阻尼替代各向同性阻尼**：HSLM 在 SVD 坐标下用 $\mathbf{D}_{k,p} = \mathrm{diag}(\max(\sigma_i^2, \delta))$ 替换 $\mathbf{I}_p$，在强曲率方向施加更大阻尼，缓解病态性。

## 方法详解

### 2.1 经典 Levenberg–Marquardt
将 NN 训练视为非线性最小二乘 $\min_x \frac{1}{2}\|\mathbf{r}(x)\|^2$，第 $k$ 步求解：
$$
\left(\mathbf{J}_k^\top \mathbf{J}_k + \mu_k \mathbf{I}_n\right) s_k = -\mathbf{g}_k, \quad \mathbf{g}_k = \mathbf{J}_k^\top \mathbf{r}(x_k)
$$
$\mu_k$ 为阻尼参数：$\mu_k$ 小 → 类 Gauss-Newton（二阶快收敛），$\mu_k$ 大 → 类梯度下降（鲁棒）。采用延迟满足策略更新：成功步 $\mu_{k+1} = \mu_k / 2$，失败步 $\mu_{k+1} = 5\mu_k$。

### 2.2 Krylov 子空间 LM (KSLM)
将更新方向限制在 Krylov 子空间 $\mathcal{K}_p(\mathbf{B}_k, \mathbf{g}_k)$，其中 $\mathbf{B}_k = \mathbf{J}_k^\top \mathbf{J}_k + \mu_k \mathbf{I}_n$，$\mathbf{V}_k$ 为列基矩阵，$s_k = \mathbf{V}_k y_k$，投影后求解 $p \times p$ 小系统：
$$
\left(\mathbf{V}_k^\top \mathbf{J}_k^\top \mathbf{J}_k \mathbf{V}_k + \mu_k \mathbf{I}_p\right) y_k = -\mathbf{V}_k^\top \mathbf{g}_k
$$
关键性质：Krylov 子空间对阻尼移位不变，即 $\mathcal{K}_p(\mathbf{J}^\top\mathbf{J}+\mu\mathbf{I}, \mathbf{g}) = \mathcal{K}_p(\mathbf{J}^\top\mathbf{J}, \mathbf{g})$，同一基可复用多次尝试 $\mu_k$。

### 2.3 混合子空间 LM (HSLM)
**子空间构建**：候选池来自四类方向：
1. 归一化梯度（一阶下降方向）
2. 最近优化步（局部历史）
3. Krylov/Lanczos 向量（主曲率方向）
4. 随机 Gauss-Newton 探测 $\mathbf{J}^\top\mathbf{J}\, w,\ w \sim \mathcal{N}(0, \mathbf{I})$（扩展谱覆盖）

**充分性检验**：
$$
\eta_k^{\mathrm{sub}} = \frac{\|\mathbf{V}_k^\top \mathbf{g}_k\|^2}{\|\mathbf{g}_k\|^2} \in [0,1]
$$
若 $\eta_k^{\mathrm{sub}} < \eta_{\min}$，则追加 Lanczos 方向（2% 参数量）和随机探测方向（1% 参数量），直到达标。

**求解**：对约化 Jacobian $\mathbf{J}_{k,p} = \mathbf{J}_k \mathbf{V}_k$ 做 SVD：$\mathbf{J}_{k,p} = \mathbf{U}_k \boldsymbol{\Sigma}_{k,p} \mathbf{Z}_k^\top$，令 $s_k = \mathbf{V}_k \mathbf{Z}_k y_k$，得到对角阻尼系统：
$$
\left(\boldsymbol{\Sigma}_{k,p}^2 + \mu_k \mathbf{D}_{k,p}\right) y_k = -\boldsymbol{\Sigma}_{k,p} \mathbf{U}_k^\top \mathbf{r}_k, \quad d_{k,i} = \max(\sigma_{k,i}^2, \delta)
$$
**步长接受**：采用 Armijo 回溯线搜索，而非经典 LM 的简单接受/拒绝。

## 实验与结果

| 任务 | 方法 | 迭代/Epoch | 运行时间 (s) | 关键指标 |
|------|------|-----------|-------------|---------|
| **阻尼振荡回归** | LM | $22.2 \pm 6.8$ | $83.0 \pm 26.2$ | 成功到达噪声方差阈值：100% |
| | KSLM | $22.2 \pm 6.8$ | $132.8 \pm 40.5$ | 成功：100% |
| | **HSLM** | $\mathbf{19.3 \pm 5.2}$ | $\mathbf{25.9 \pm 7.4}$ | 成功：100% |
| | SGD | $155.0 \pm 55.1$ | $30.9 \pm 10.9$ | 成功：41% |
| | Adam | $102.1 \pm 44.3$ | $24.3 \pm 10.6$ | 成功：0% |
| **13-bit 奇偶校验** | LM | $67.8 \pm 47.8$ | $5.8 \pm 4.1$ | 收敛率 92%，验证精度 $97.7\%$ |
| | KSLM | $71.1 \pm 46.3$ | $6.3 \pm 4.0$ | 收敛率 89%，验证精度 $97.9\%$ |
| | **HSLM** | $\mathbf{37.0 \pm 27.2}$ | $\mathbf{3.7 \pm 3.8}$ | **收敛率 100%**，验证精度 $99.6\%$ |
| | SGD | $1061.7 \pm 530.1$ | $13.2 \pm 6.5$ | 收敛率 92%，验证精度 $99.7\%$ |
| | Adam | $333.3 \pm 458.5$ | $5.7 \pm 7.8$ | **收敛率 100%**，验证精度 $\mathbf{99.8\%}$ |

**最强结果**：回归任务中，HSLM 以仅 $25.9\,\mathrm{s}$ 的运行时间达到 100% 成功率，比 LM 快 3.2 倍、比 KSLM 快 5.1 倍，迭代数与 LM/KSLM 相当（~19 vs 22）。分类任务中，HSLM 收敛率 100% 且验证精度达 $99.6\%$，迭代次数仅为 LM 的 54.6%、KSLM 的 52.0%。

## 相关工作脉络
- **Levenberg (1944) / Marquardt (1963)**：LM 算法原始提出，本文经典 LM 的基线来源 [8,10]。
- **Hagan & Menhaj (1994)**：首次在神经网络训练中推广 LM，本文作为经典 LM 实现参照 [4]。
- **Mizutani & Demmel (2001) – KSLM**：将 Krylov 子空间引入 LM 以加速 NN 训练，本文三大基线之一 [12]。
- **Hoang & Lewis (2026) – HSLM**：自适应混合子空间 LM，本文核心方法（同上作者 arXiv 预印本 [5]）。
- **Kingma & Ba (2015) – Adam**：最主流的一阶自适应优化器，本文一阶对比基线 [7]。
- **Transtrum et al. (2011)**：延迟满足阻尼更新策略（"delayed gratification"）来源 [14]，被本文 LM 系列方法采用。

## 局限性与未来方向
- **大数据可扩展性受限**：子空间方法需显式操作 Jacobian 相关矩阵，数据量大时形成和求解子问题成本仍会上升。
- **未涉及小批量（mini-batch）实现**：当前为全量数据计算 Jacobian，未扩展到 minibatch 或 matrix-free 设置。
- **仅在小规模 MLP 上验证**：实验网络为 $1{-}70{-}40{-}1$（回归）和 $13{-}25{-}10{-}1$（分类），未测试深层/大规模架构。
- **子空间基选择策略待系统研究**：不同候选方向组合对效率和收敛行为的影响尚未充分探索。

## 研究启发与可借鉴点
- **子空间充分性度量 $\eta_k^{\mathrm{sub}}$ 可迁移**：该梯度能量捕获比例可作为任何 subspace optimization 方法的质量监控器，适用于 GAN 训练、物理信息神经网络等需稳定收敛的场景。
- **曲率相关对角阻尼设计思路可复用于其他二阶方法**：将 $\mathbf{D} = \mathrm{diag}(\max(\sigma_i^2, \delta))$ 替换各向同性正则，这一思想可推广至 L-BFGS 或对角 Fisher 信息矩阵（KFAC）场景。
- **多源信息融合的子空间构建策略**：梯度+历史步+Krylov+随机探测的组合方式，为设计"轻量但信息丰富的搜索空间"提供了可借鉴模板，可适配到 Transformer 微调等参数规模较大的任务中。
- **延迟满足阻尼更新策略在分类任务上同样有效**：$\mu$ 的成功/失败步分别除以 2 或乘 5 的更新规则，不仅适用于回归，也可作为二阶方法的标准阻尼调度器参考。
- **低资源场景下的方法论价值**：附录 C 显示 HSLM 在训练数据缩减至 30% 时仍保持良好收敛-精度平衡，提示该方法适合数据稀缺或计算资源受限的部署场景。

## 关键术语表
- **Levenberg–Marquardt (LM) 算法**：结合梯度下降与 Gauss-Newton 的二阶优化方法，通过阻尼参数 $\mu$ 在两者间自适应切换，适用于非线性最小二乘问题。
- **Krylov 子空间 LM (KSLM)**：将 LM 更新方向限制在由 Jacobian 信息生成的 Krylov 子空间内，降低系统维度，保留 LM 快速收敛特性。
- **混合子空间 LM (HSLM)**：从梯度、历史步、Krylov 向量和随机探测等多源信息中自适应构建子空间，结合曲率相关阻尼和 Armijo 回溯，平衡效率与收敛质量。
- **子空间充分性准则 $\eta_k^{\mathrm{sub}}$**：衡量子空间捕获当前梯度能量的比例，用于判断是否需要扩展子空间以保证下降方向的有效性。
- **Gauss-Newton 近似**：用 $\mathbf{J}^\top \mathbf{J}$ 近似 Hessian 矩阵，避免计算二阶导数，是 LM 和牛顿类方法的核心近似。
- **阻尼参数 $\mu_k$**：控制 LM 步在"类梯度下降"（大 $\mu$）和"类 Gauss-Newton"（小 $\mu$）之间切换的标量，直接影响收敛稳定性和速度。
- **延迟满足阻尼策略**：成功步减小 $\mu$（加速收敛）、失败步增大 $\mu$（增强稳定性），使方法能在信赖域和 Gauss-Newton 区域间智能切换。
- **13-bit 奇偶校验问题**：标准 NN 基准分类任务，输入为 13 位二进制串，输出为各位异或结果，用于评估优化器在离散非线性问题上的表现。

## 可复现要素
- **数据集**：回归任务使用人工生成的 $f(x) = 2e^{-x^2}\cos(2\pi x)$ 加高斯噪声（$\sigma_{\mathrm{noise}} = 0.05\sigma_y$），40,000 样本；分类任务使用完整 13-bit 奇偶校验集（8,192 样本），90/10 分层划分。**数据非公开，但生成方式完全透明可复现**。
- **代码/权重开源**：论文声明代码在 Python/NumPy/Jupyter Notebook 中实现，运行于 Dell 14 Plus（Intel Core Ultra 9 288V, 32GB RAM）；**未提供 GitHub 仓库链接，仅附于补充材料中**。
- **关键超参数**：
  - LM/KSLM/HSLM：初始 $\mu_0 = 10$，最大迭代 150，成功步 $\mu \leftarrow \mu/2$，失败步 $\mu \leftarrow 5\mu$
  - KSLM：Krylov 子空间维度上限 = 5% 参数量，Lanczos 容差 $10^{-3}$
  - HSLM：初始随机探测维度 = 1% 参数量，$\eta_{\min} = 0.99$，扩展时 Lanczos 占 2% + 随机探测占 1%，总维度 ≤ 10% 参数量
  - SGD：lr=0.01，momentum=0.9，batch=64，max epoch=1500
  - Adam：lr=0.003，$\beta_1=0.9, \beta_2=0.999$，batch=64，max epoch=1500
  - 参数初始化：$\theta_0 \sim \mathcal{U}[-1, 1]$，所有方法共享相同初始值
