---
title: "Identifying-Neural-Source-Dynamics-from-Unknown-Local-Interv"
source: https://arxiv.org/pdf/2609.35379v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-02 14:31:35"
---

# 论文速读：Identifying-Neural-Source-Dynamics-from-Unknown-Local-Interv

## 一句话总结
本文证明在基线实验仅覆盖部分状态空间（rank $R_s < q$）且导联场低秩（$m < q$）的严格条件下，单次单源未知局部干预会在 EEG 响应中留下可分离的秩-1 签名；结合已知导联场 $L$ 与初始化映射 $K$，无需基线可达性即可构造性恢复全源转移矩阵 $F$，并在仿真中实现 100% 成功率与亚 4% 的中位误差。

## 研究问题与动机
- **基线覆盖不足导致动力学不可标识**：神经记录中初始化映射 $K$ 与状态激发矩阵 $R_s$ 往往只能张成状态空间真子集，传统 Hankel/协方差辨识方法在此场景下失效。
- **重复测量或增加传感器无法消除根本歧义**：即使提高采样时长或并行记录通道数，若解剖前向模型 $L$ 的列空间未覆盖全部源，基线响应仍无法区分不同 $F$。
- **局部干预常被当作噪声滤除**：神经调控或突触可塑性引起的局部机制变化在实验中通常被视为干扰，本文反其道而行，将其重新定义为提供额外秩的辨识信号。
- **缺乏无基线可控性假设下的显式恢复理论**：现有系统辨识多依赖满行秩输入或强结构化先验，亟需一套仅依赖代数秩条件、不要求先验估计干预参数 $v_e$ 的重建框架。

## 核心贡献（创新点）
1. **揭示局部干预的秩-1 可分离签名**：证明差分响应 $\Delta H_e = H^{[e]} - H_+$ 严格分解为源传播历史与干预系数的外积，为后续无需预估 $v_e$ 的纯几何恢复奠定基础。
2. **无基线可控性下的显式恢复定理**：建立 $\text{rank}\,\Omega = q$ 与 $\text{rank}\,O_- = q$ 为充分条件，首次在不要求 $R_s$ 满秩或基线可达的前提下给出 $F = O_-^\dagger O_+$ 的构造解。
3. **解剖列标识与历史尺度校准机制**：利用 $L$ 列的非比例性，通过对比响应的首段传感器块与 $L$ 的夹角匹配唯一确定靶源索引，并以标量 $\gamma_e$ 固定物理响应单位，避免坐标缩放模糊。
4. **条件稳定性界与误差解耦**：推导 $\| \hat F - F \|_2 \leq \frac{(1+\|F\|_2)\delta}{\beta - \delta}$，将重建误差显式分解为对比信号扰动 $\delta$ 与最弱时序可观测性 $\beta$ 两项，为实验设计提供量化判据。

## 方法详解
- **系统模型**（Eq. 1）：$z_{\tau+1} = (F + D_{e_\tau})z_\tau + \epsilon_\tau$，$x_\tau = L z_\tau + \nu_\tau$。$F \in \mathbb{R}^{q\times q}$ 未知，$L \in \mathbb{R}^{m\times q}$ 已知导联场（$\text{rank}\,L < q$），$K$ 为已知初始化映射。每次 episode 仅插入一个未知局部变化 $D_e = e_{j_e}v_e^\top$，状态不重置。
- **响应矩阵构建**（Eq. 4）：定义 $H_0 = O_T R_s$（基线提前一拍）、$H_+ = O_T F R_s$（匹配时间）、$H^{[e]} = O_T(F+D_e)R_s$（含干预）。差分 $\Delta H_e = o_{j_e}\psi_e^\top$ 为秩-1 外积。
- **直接补全重建算法（Algorithm 1）**：
  1. 对每个 $\hat H^{[e]}-\hat H_+$ 做 SVD，取主导左奇异向量 $\hat u_e$。
  2. 取 $\widehat{u}_{e,\text{top}}$（前 $m$ 维）与 $L_{:,j}$ 计算夹角匹配源标签 $\hat j_e$；通过 $\gamma_e = \frac{\widehat{u}_{e,\text{top}}^\top L_{:,j}}{\|\widehat{u}_{e,\text{top}}\|_2^2}$ 校准历史 $\widehat{o}_{\hat j_e}^{(e)} = \gamma_e\hat u_e$。
  3. 累积 $\hat Y$ 与 $\hat\Omega = [K, E_J]$，验证秩条件后计算 $\hat O_T = \hat Y\hat\Omega^\dagger$，分割得 $\hat O_-$、$\hat O_+$，返回 $\hat F = \hat O_-^\dagger\hat O_+$。
- **稳定性界（Proposition 2，Eq. 12）**：$\| \hat F - F \|_2 \leq \frac{(1+\|F\|_2)\delta}{\beta - \delta}$，其中 $\delta = \|\hat Y - Y\|_2/\sigma_{\min}(\Omega)$，$\beta = \sigma_{\min}(O_-)$。误差由四个瓶颈共同控制：对比暴露 $\sigma_e$、头皮可见性与标注分离 $(\alpha_j, \mu_j)$、坐标覆盖 $\sigma_{\min}(\Omega)$、动态求逆 $\beta$。
- **
