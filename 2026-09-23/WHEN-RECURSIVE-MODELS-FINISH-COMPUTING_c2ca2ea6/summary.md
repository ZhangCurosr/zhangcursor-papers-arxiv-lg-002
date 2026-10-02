---
title: "WHEN-RECURSIVE-MODELS-FINISH-COMPUTING"
source: https://arxiv.org/pdf/2609.26487v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:28:21"
---

# 论文速读：WHEN-RECURSIVE-MODELS-FINISH-COMPUTING

## 一句话总结
本文通过将 Tiny Recursive Models (TRMs) 的递归步数从名义预算 16 步扩展至 512 步，证明大量名义截止时的“失败”实为未完成的计算；研究发现完成标志是潜状态沿自身轨迹方向局部收缩，但同一 Jacobian 仍保留强扩张方向，提出“轨迹条件异向稳定性（trajectory-conditioned anisotropic stability）”作为跨架构与任务的通用完成动力学签名。

## 研究问题与动机
- 递归模型在名义推理预算结束时的错误输出，无法区分“真正失败”与“计算尚未完成”；正确输出同样无法判断底层潜状态是否已稳定。
- 现有 TRM/HRM 机制研究多聚焦吸引子结构、spurious fixed points 与静态失败模式，缺乏以“完成时刻”为对齐基准的动态相变分析。
- 需要一套可解释的动力学指标，用于实时监控递归推理是否仍在推进、是否已收敛，从而支持可靠的 test-time compute scaling。
- 不同架构（Attention vs MLP）在完成后的局部扩张几何上是否存在共性，或仅依赖特定归纳偏置。

## 核心贡献（创新点）
1. **轨迹对齐的完成态划分**：以首次精确解步数 τ_i 为基准将谜题分为 EARLY/LATE/PERSISTENT 三组，定量揭示约 69% 的名义预算失败实为未完成的计算，而非模型能力边界。
2. **轨迹条件异向稳定性发现**：证明完成后的潜状态沿自然轨迹方向局部收缩（γ_nat < 1），但局部 Jacobian 仍保留强扩张方向（σ_max > 1），打破“完成即全局收缩”的直觉假设。
3. **扩张方向的几何命运刻画**：通过矩阵自由幂迭代与有限步扰动实验，揭示 v1 主要横跨 Z_H 至 Z_L 跨块映射，且 Attention 与 MLP 在扰动吸收速度与方向传输度上存在显著差异。
4. **跨架构与跨任务的普适性验证**：在第二个 Attention checkpoint（Attention-B）、Easy Sudoku 与 Maze-Hard 任务上重复全套动力学分析，证实该签名具有架构与任务无关性。

## 方法详解
- **模型与扩展推理设置**：使用两个 TRM checkpoint——Attention-A（6.83M 参数，含 Rotary PE）与 MLP（5.03M 参数，序列轴 MLP 混合，无位置编码）。在 1,000 道 Hard Sudoku 上以名义 16 步为起点，冻结权重与预处理，将递归延续至 512 步作为诊断视界。
- **潜状态与运动度量**：联合状态 $s_t = (Z_H^{(t)}, Z_L^{(t)})$，维度 $2 \times 97 \times 512$。相对更新量 $\delta_B(t) = \|B^{(t)} - B^{(t-1)}\|_2 / \|B^{(t-1)}\|_2$。以首次精确解步数 $\tau_i$ 定义对齐时间 $r = t - \tau_i$。
- **自然方向增益**：线性化递归映射 $J_t = \partial F_\theta / \partial s|_{s_t}$，定义单位轨迹方向 $d_t^{\mathrm{out}} = (s_{t+1} - s_t)/\|s_{t+1} - s_t\|_2$，计算 $\gamma_t^{\mathrm{nat}} = \|J_t d_t^{\mathrm{out}}\|_2$。$\gamma < 1$ 表示沿轨迹局部收缩。
- **最坏情况扩张度量**：通过矩阵自由幂迭代（避免显式构建 $99{,}328 \times 99{,}328$ Jacobian）估计 $\sigma_{\max}(J_t)$ 及对应右奇异向量 $v_1(t)$。满足 $\gamma_t^{\mathrm{nat}} < 1$ 且 $\sigma_{\max}(J_t) > 1$ 时即判定为异向稳定性。
- **扰动与有限视界增长**：沿 $v_1$ 与自然方向注入 $\varepsilon = 10^{-4}$ 量级扰动，演化 $k$ 步后计算分离度，定义 $\lambda_k(v) = \frac{1}{k}\log(\|\delta s_{t+k}\|/\varepsilon)$。跨步传输度 $T_1(t) = |\langle u_1(t), v_1(t+1)\rangle|^2$ 衡量扩张方向的持续性。
- **统计与数值规范**：Jacobian 计算 upcast 至 float32，禁用 TF32，使用数学 SDPA 后端；bootstrap 20,000 次估算置信区间；解对齐窗口保持配对结构。

## 实验与结果
- **扩展推理显著提升求解率**：Attention-A 从 59.2% (step 16) 升至 87.5% (step 512)，MLP 从 74.4% 升至 91.9%。step 16 未解决的 408 题中 283 题（69.4%）在后续步骤解出；MLP 对应 175/256 题（68.4%）。
- **潜状态运动完成骤降**：LATE 求解者在完成前运动水平与 PERSISTENT 相近（如 $\delta_L=0.763$ vs EARLY 的 0.025）。完成瞬间 $Z_H$ 更新出现峰值（99.3% 的 LATE 题在此步达轨迹内最大更新），随后下降超一个数量级，与 EARLY 汇合。
- **异向稳定性贯穿完成组**：step 512 时，完成组（EARLY/LATE）中 96%/92% 的状态满足 $\gamma^{\mathrm{nat}} < 1$，但所有探測点的 $\sigma_{\max}$ 均 > 1（完成组约 27，PERSISTENT 组约 317）。两族同时满足 $\gamma^{\mathrm{nat}} < 1 < \sigma_{\max}$ 的状态占比约 62%（Attention）与 60.4%（MLP）。
- **扰动命运跨架构分化**：Attention-A 中 $v_1$ 扰动初期放大 24.4 倍，16 步后衰减至初始的 0.10；MLP 中同期仍维持 1.91 倍。方向传输度 $T_1$ 在 Attention 为 $4.9\times10^{-5}$，MLP 为 $1.83\times10^{-3}$，表明 MLP 更持久地携带扩张方向。
- **跨任务/Checkpoint 复现**：Attention-B checkpoint 呈现完全相同的动力学模式；Easy Sudoku 与 Maze-Hard 任务中绝大多数轨迹落入 EARLY 求解者 regime；Maze-Hard 显示 exact-match 严重低估有效性，实际有效路径率达 97.9%。

## 相关工作脉络
1. **Deep Equilibrium Models / Equilibrium Reasoners (Bai et al., 2019; Huang et al., 2026)**：显式建立计算与不动点的等价性；本文不假设全局收缩，而是揭示完成仅沿轨迹方向稳定，修正了均衡假设的适用范围。
2. **TRM/HRM 机制分析 (Efstathiou & Balwani, 2026; Ren & Liu, 2
