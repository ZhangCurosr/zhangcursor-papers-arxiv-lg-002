---
title: "Normalize-Then-Precondition-A-Hierarchical-Approach-to-Margi"
source: https://arxiv.org/pdf/2609.36692v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:10:12"
field: "大语言模型训练优化"
keywords: ["矩阵优化器", "LLM训练", "谱预条件", "归一化", "Newton-Schulz迭代", "随机Sketch"]
innovations: ["提出Normalize-Then-Precondition层次化框架，将边缘尺度归一化与谱预条件分阶段处理", "设计NormPre-G（全局谱预条件）和NormPre-L（局部谱预条件）两个变体，在多种模型规模上一致优于AdamW/Muon/MANO", "建立简化版NormPre的O(T^{-1/2})收敛性保证，并从谱动力学角度揭示局部预条件捕获全局变换62%-67%能量"]
benchmarks: ["OpenWebText", "C4", "Pile"]
---

# 论文速读：Normalize-Then-Precondition: A Hierarchical Approach to Marginal Scale and Interaction Geometry for LLM Training

## 一句话总结
论文提出了 **Normalize-Then-Precondition** 层次化框架及 **NormPre** 优化器家族，先通过对角Gram归一化消除矩阵更新的边缘尺度差异，再进行全局或局部谱预条件细化交互几何信息；在 GPT-2、LLaMA 和 Qwen3 的多规模预训练中，两种变体在匹配训练预算下均持续优于 AdamW、Muon 和 MANO。

## 研究问题与动机
- LLM 训练对优化器的性能、效率与可扩展性要求日益提高，矩阵级优化器（如 Muon）成为新兴方向，但现有方法联合处理边缘尺度与跨行交互信息，极端行范数会主导主导特征方向，掩盖真实的行列交互结构。
- 归一化与正交化在 Gram 表示下共享结构，但尚未有系统性框架对这两种几何信息进行层次化组织，尤其是"先归一化再谱预条件"的构造路径仍未被 formalize。
- 已有工作（如 Muon+、SWAN、MuonEq）虽提及归一化的数值益处，但未建立从边缘尺度归一化到定向交互预条件的分层理论体系。
- 论文核心问题：**能否先从边缘尺度构建归一化更新，再用方向交互信息进行谱预条件细化？**

## 核心贡献（创新点）
1. **提出 Normalize-Then-Precondition 层次化框架**，将联合几何变换分解为先对角 Gram 归一化消除尺度加权、再基于归一化后的交互谱进行预条件，与 Muon 等同时处理尺度与交互的方法形成本质区别。
2. **设计 NormPre 优化器及其两个变体**：NormPre-G 基于 Newton-Schulz 迭代实现全局谱预条件（全谱映射到单位谱），NormPre-L 基于随机 Sketch 实现局部谱预条件（仅裁剪活跃模式），兼顾性能与可扩展性。
3. **建立 O(T^{-1/2}) 收敛性保证**，为简化版 NormPre-G 与精确/ sketch 版 NormPre-L 提供了理论支撑，扩展了矩阵优化器的理论分析边界。
4. **系统化实验验证与性能-效率权衡刻画**，在 GPT-2 Small、LLaMA（最高 1.3B）和 Qwen3（最高 1.7B）上展示两种变体的一致性优势，并通过谱动力学分析揭示局部预条件已捕获全局变换约 62%-67% 的能量。

## 方法详解
### 整体流程（Algorithm 1）
对矩阵参数 W_t 和随机梯度 G_t，构建一阶动量 M_t = μM_{t-1} + G_t，经过以下步骤生成更新：
1. **交替行/列方向**：k_t = t mod 2，交替以行或列为活跃轴，保持对称性。
2. **松弛切线动量**：对每行计算 x_{t,i} = ḡ_{t,i} - ⟨ḡ_{t,i}, ṽ_{t,i}⟩ṽ_{t,i}，保留切线分量并维持与权重相关的径向分量（严格切线投影当 ‖w‖=1 时退化）。
3. **对角 Gram 归一化**：Ψ_t = D(X_t)^{-1}X_t，其中 D(X) = Diag(diag(XX^T))^{1/2} 仅提取各行范数，消除尺度加权。
4. **谱预条件**：根据变体不同，对 Ψ_t 进行全局或局部谱变换。
5. **一致 RMS 缩放**：将更新缩放到目标 RMS=0.2，与 Muon/MANO 保持可比性。

### NormPre-G：全局谱预条件
- 求解谱梯度下降问题：max_{‖T‖_op ≤ 1} ⟨Ψ, T⟩_F，解为 T^G = msign(Ψ)。
- 使用 5 步 Newton-Schulz 迭代近似矩阵符号函数，将 Ψ 的所有奇异值映射到 1。
- 复杂度与 Muon 相同：O(mn + qmns)，q=5。

### NormPre-L：局部谱预条件
- 以 Ψ 为参考，求解正则化谱最速下降：min_{‖T‖_op ≤ 1} ½‖T-Ψ‖_F²，等价于将奇异值>1 的模式裁剪到 1，其余保持不变。
- 引入谱预算 r 限制变换的子空间维度，仅对 top-r 活跃特征模式（λ_i > 1）进行裁剪。
- **精确实现**：对 Γ=ΨΨ^T 做完整特征分解，O(m²n + m³)。
- **Sketch 实现**：通过高斯随机投影 Y=ΨΩ 和 power iteration 构建候选子空间，经 Rayleigh-Ritz 提取 top-r 特征对，复杂度降为 O((p+1)m n ℓ + (m+n)ℓ² + ℓ³)，其中 ℓ=min{m, r+o}。

### 关键超参数
- 动量系数 μ=0.95，Newton-Schulz 迭代 5 步，NormPre-L 的 r=32、oversampling o=8、power iteration p=1。

## 实验与结果
### 数据集与模型
- **GPT-2 Small (124M)** on OpenWebText
- **LLaMA-130M/350M/1.3B** on C4
- **Qwen3-0.6B/1.7B** on Pile
- 统一设置：10,000 步，序列长度 1024，有效批量 512，warmup 后 cosine decay 至 10% 峰值学习率，权重衰减 0.1，梯度裁剪 1.0。

### 评估基线
- **AdamW**：坐标级自适应（β₁=0.9, β₂=0.95）
- **Muon**：矩阵正交化（μ=0.95, 5 步 Newton-Schulz）
- **MANO**：动量投影+交替归一化（μ=0.95）

### 主要结果
| 模型 | 数据集 | AdamW | Muon | MANO | **NormPre-G** | **NormPre-L** |
|------|--------|-------|------|------|--------------|--------------|
| GPT-2 Small | OpenWebText | 3.1444 | 3.1064 | 3.1156 | **3.0667** | 3.0826 |
| LLaMA-1.3B | C4 | 2.9385 | 2.9037 | 2.8963 | **2.8571** | 2.8662 |
| Qwen3-1.7B | Pile | 2.6758 | 2.6408 | 2.6205 | **2.5868** | 2.5897 |

- NormPre-G 在全部实验中取得最低验证损失；相比最强基线 Muon，GPT-2 Small 降低 **0.0397**，LLaMA-1.3B 降低 **0.0392**，Qwen3-1.7B 降低 **0.0540**。
- NormPre-L 与 NormPre-G 在 1.7B 规模差距仅 **0.0029**，显示局部预条件已逼近全局性能。

### 效率分析
- **NormPre-G**：优化器延迟较 Muon 增加 5.76%（GPT-2）/ 4.37%（LLaMA-1.3B），E2E 步时间增加 <0.5%。
- **NormPre-L**：优化器延迟较 Muon 降低 38.12%（GPT-2）/ 67.00%（LLaMA-1.3B），E2E 步时间降低 1.16%/3.13%，吞吐提升 1.17%/3.26%。
- 两种变体峰值显存与 Muon/MANO 持平，因优化器更新仅占步时间一小部分。

### 谱动力学发现
- 归一化后 Gram 谱呈各向异性，small-norm 行的相对影响增大，这种各向异性贯穿训练全过程。
- 全局变换能量中，top-32 活跃模式（λ_i>1）占比约 **62%-67%**，局部预条件覆盖了绝大部分变换能量。
- 使用归一化后的 Ψ 构造预条件器（vs 归一化前的 X）带来更低损失，尤其对 NormPre-L 更敏感。

## 相关工作脉络
- **AdamW**：坐标级优化器，独立处理每个参数，未利用矩阵结构，是本文对照的"基础线"。
- **Muon**：矩阵正交化优化器，通过 Newton-Schulz 迭代近似矩阵符号函数，同时处理边缘尺度与交互信息；本文从 Full-Gram 表示重新审视 Muon，指出其尺度加权问题，并在此基础上提出层次化分解。
- **MANO**：结合动量投影与交替行/列归一化，仅进行归一化而不做谱预处理；本文将其归为"纯归一化"路线，NormPre 在其基础上增加了谱预条件阶段。
- **MOGA**：从最速下降视角推导归一化更新，使用 mean-normalized operator norm；本文与其区别在于引入了谱预条件的层次化设计。
- **Muon+ / NorMuon / AdaMuon**：在正交化后或过程中加入归一化/自适应缩放，属于"后处理型"改进；本文的先归一化再预条件的层次结构与它们本质不同。
- **SWAN / MuonEq**：关注归一化的数值稳定性（如 SWAN 在对角化前归一化梯度，MuonEq 平衡动量以改善 Newton-Schulz 条件数）；本文的系统性框架超越了这些零散改进。

## 局限性与未来方向
- 实验最大模型规模仅到 1.7B 参数，尚未在十亿级以上模型上验证，扩展到更大规模是必要验证步骤。
- Newton-Schulz 迭代与特征空间提取的计算开销仍高于 AdamW/MANO，需进一步开发硬件感知实现（如 GPU kernel 优化）以提升效率。
- NormPre-G 与 NormPre-L 之间存在约 0.003 的性能差距（1.7B 规模），探索自适应谱预算或混合策略可能缩小这一差距。
- 收敛性分析针对简化版本（无动量、无 RMS 缩放、无权重衰减），实际变体（含 momentum、RMS、weight decay）的理论保证有待完善。
- Sketch 实现的近似误差边界需在实际训练中被更系统地分析，当前仅在 controlled approximation error 假设下给出收敛性。

## 研究启发与可借鉴点
- **层次化几何组织范式**：将复杂矩阵变换分解为"先处理简单结构（尺度）、再处理复杂结构（交互）"的分阶段策略，可迁移到其他矩阵优化器或二阶方法的设计中。
- **松弛切线投影设计**：在保留梯度方向变化信息的同时，不丢弃与参数方向相关的径向分量，这种平衡策略在 strict tangent 导致性能下降的 ablation 中得到验证，值得在其他优化器中借鉴。
- **谱预算与性能-效率权衡的系统化刻画**：NormPre-L 通过超参数 r 控制变换子空间维度，并提供了从 r=8 到 r=64 的 ablation，这种将理论约束与实践经验结合的权衡分析范式具有参考价值。
- **归一化后构造预条件器的设计原则**：实验 Table 7 显示，从 Ψ（归一化后）而非 X（归一化前）提取交互几何显著更好，尤其对局部预条件；这一"预条件器应与被操作对象同空间"的原则可推广。
- **随机 Sketch 在优化器中的应用**：Algorithm 2 展示的 Gaussian sketch + power iteration + Rayleigh-Ritz 流程是低秩特征空间提取的标准工具，可复用到其他需要高效谱信息的场景（如自然梯度、K-FAC 近似）。

## 关键术语表
- **Full-Gram Representation**：完整 Gram 表示 (XX^T)^{-1/2}X，同时编码边缘尺度（对角）与行间交互（非对角），Muon 的等价形式。
- **Diagonal-Gram Normalization**：对角 Gram 归一化，仅使用 Gram 矩阵对角元素（各行 L2 范数）构造 Ψ=D(X)^{-1}X，消除尺度加权。
- **Spectral Preconditioning**：谱预条件，对矩阵的奇异值/特征值进行变换以改善优化几何，本文分为全局（全谱映射）与局部（仅活跃模式）两类。
- **Newton-Schulz Iteration**：牛顿-舒尔茨迭代，用于快速近似矩阵函数（如矩阵符号函数）的迭代方法，5 步即可达到高精度。
- **Randomized Sketch**：随机 Sketch，通过高斯随机投影 Y=ΨΩ 构建低维子空间，再以 Rayleigh-Ritz 提取特征对，用于高效近似领先特征空间。
- **Relaxed Tangent Momentum**：松弛切线动量，x = g - ⟨g,w⟩w，保留梯度切线分量的同时维持与权重范数相关的径向分量，避免严格切线投影的信息丢失。
- **Descent Alignment**：下降对齐度，优化器更新方向与负梯度的余弦相似度，用于衡量优化器偏离最速下降方向的程度。
- **Spectral Budget (r)**：谱预算，NormPre-L 中限制局部谱变换作用的 top-r 活跃特征模式数量，控制性能与效率的权衡。

## 可复现要素
- **代码开源**：https://github.com/zx-gong/NormPre
- **数据集**：OpenWebText、C4、Pile（均为公开数据集）
- **关键超参数**：
  - 学习率：GPT-2 Small 峰值 6×10⁻⁴，LLaMA/Qwen3 峰值 3×10⁻⁴
  - 动量系数 μ = 0.95
  - Newton-Schulz 迭代步数 q = 5
  - NormPre-L：谱预算 r = 32，oversampling o = 8，power iteration p = 1
  - 权重衰减 λ_wd = 0.1
  - 梯度裁剪 = 1.0
  - 目标 RMS = 0.2
  - 训练步数 = 10,000，序列长度 = 1024，有效批量 = 512
- **硬件**：NVIDIA A100 80GB GPU
- **实现细节**：GPT-2 Small 使用 nanoGPT 代码库，LLaMA/Qwen3 使用统一 PyTorch 训练代码；Muon 在 GPT-2 Small 上使用 Keller-Jordan convention（有效学习率 0.02），在 LLaMA/Qwen3 上使用 scalable RMS-matching convention。
