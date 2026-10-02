---
title: "MEQMUON-MATRIX-EQUILIBRATING-MUON-FOR-LLM-PRETRAINING"
source: https://arxiv.org/pdf/2609.35701v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:53:47"
---

# 论文速读：MEQMUON-MATRIX-EQUILIBRATING-MUON-FOR-LLM-PRETRAINING

## 一句话总结
提出MeqMuon优化器，通过自适应行/列RMS归一化同时平衡正交化与未正交化更新矩阵的幅度分布，在消除AdamW二阶矩存储开销的同时，在LLM预训练中实现了优于AdamW、Muon及NorMuon的收敛性能与内存效率。

## 研究问题与动机
1. Muon优化器虽在LLM预训练中表现出高训练效率与高精度，但其正交化更新矩阵在行或列方向上存在显著且方向不定的幅度不平衡。
2. 现有改进方法NorMuon仅采用行向归一化，无法覆盖更新矩阵可能在列方向主导的不平衡模式，适配性不足。
3. Muon及其变体对token embedding、LM head等其余2D参数以及1D参数仍依赖AdamW更新，需额外维护一阶与二阶矩估计，推高优化器状态内存占用。
4. 亟需一种能自动感知并匹配不同不平衡模式、无需手动干预，且能进一步压缩优化器内存的统一归一化方案。

## 核心贡献（创新点）
1. 系统刻画了Muon正交化更新与未正交化动量矩阵中行/列幅度的两类不平衡模式，指出单一方向归一化无法兼顾不同几何形状的矩阵。
2. 提出MeqMuon，对隐藏层2D权重采用基于CV比较的自适应单侧归一化（行或列），对其余2D参数采用双侧RMS归一化，无需人工调参即可自动适配不平衡模式。
3. 用单层动量归一化完全替代AdamW更新以处理非隐藏层2D与1D参数，消除二阶矩估计存储，显著降低优化器内存开销。
4. 在Llama、SmolLM2、Qwen2系列模型（60M~0.5B）上的预训练实验验证了MeqMuon在最终PPL与训练曲线收敛性上全面优于AdamW、Muon、NorMuon及SCALE基线。

## 方法详解
- **统一动量累积**：所有参数仅维护一个动量缓冲区 $B_t = \mu B_{t-1} + G_t$，放弃AdamW风格的一阶/二阶矩双缓冲结构。
- **隐藏层2D权重（自适应单侧归一化）**：先对 $B_t$ 执行K步Newton-Schulz迭代得到正交化矩阵 $U_t = Orth(B_t)$，计算行CV $\gamma_r$ 与列CV $\gamma_c$；若 $\gamma_r \ge \gamma_c$ 则施加行向RMS归一化 $\mathcal{N}_r(U_t)$，否则施加列向RMS归一化 $\mathcal{N}_c(U_t)$，每步每矩阵独立决策。
- **其余2D参数（双侧归一化）**：对token embedding、LM head等未正交化的动量直接应用 $\widetilde{U}_t = \mathcal{N}_c(\mathcal{N}_r(B_t))$，同步压制行与列的不平衡。
- **1D参数（全局RMS归一化）**：对RMSNorm权重与偏置向量采用 $\widetilde{U}_t = B_t / (\|B_t\|_2/\sqrt{d})$ 进行全局尺度归一化。
- **统一更新公式**：$W_{t+1} = (1-\eta_t\lambda)W_t - \rho\eta_t\widetilde{U}_t$，其中解耦权重衰减与缩放系数$\rho$使更新RMS与矩阵形状解耦，故不再需要Muon原版中的形状补偿项$\sqrt{\max(m,n)}$。

## 实验与结果
- **数据集与模型**：English C4语料，按Chinchilla最优规则（训练Token数=20×参数量）训练Llama (60M/130M/350M)、SmolLM2 (135M/360M)、Qwen2 (0.5B)；序列长度1024，全局Batch Size 512，BF16混合精度，DDP分布式。
- **评估指标**：验证集PPL（↓越低越好）与优化器状态内存（MiB，↓越低越好）。
- **收敛性能**：MeqMuon在所有模型尺度上均取得最低验证PPL。Llama-350M为15.94（Muon 16.04，NorMuon 15.98）；Qwen2-0.5B为18.63（Muon 18.79）；SmolLM2-360M为17.16。训练曲线全程位于基线下方。
- **内存节省**：因移除二阶矩存储，MeqMuon在Qwen2-0.5B上优化器内存降至1884.59 MiB，较Muon/NorMuon的2405 MiB减少21.6%；Llama-60M/130M分别降至221.53/511.57 MiB。
- **归一化方向统计**：高瘦矩阵（gate_proj/up_proj）几乎100%选择行归一化，宽扁矩阵（down_proj）100%选择列归一化，与正交化诱导的单侧不平衡理论完全吻合。
- **消融实验**：仅修改隐藏层2D权重、仅修改其余2D参数或仅修改1D参数的变体均能独立带来PPL下降与内存缩减，三者组合的完整MeqMuon效果最优。

## 相关工作脉络
1. **
