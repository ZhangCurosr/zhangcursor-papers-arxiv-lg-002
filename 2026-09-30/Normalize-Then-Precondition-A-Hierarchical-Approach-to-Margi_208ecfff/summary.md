---
title: "Normalize-Then-Precondition-A-Hierarchical-Approach-to-Margi"
source: https://arxiv.org/pdf/2609.36692v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:10:08"
field: "大规模模型训练优化器"
keywords: ["矩阵优化器", "LLM 训练", "谱预条件", "归一化", "Newton-Schulz 迭代", "随机草图", "收敛性分析"]
innovations: ["提出 Normalize-Then-Precondition 层次化框架，分离边缘尺度归一化与交互谱预条件", "设计 NormPre-G/NormPre-L 两种全局/局部谱预条件变体并给出可扩展实现", "建立 O(T^{-1/2}) 收敛性保证并验证在多种架构上的优越性"]
benchmarks: ["GPT-2 Small on OpenWebText", "LLaMA-130M/350M/1.3B on C4", "Qwen3-0.6B/1.7B on Pile"]
---

# 论文速读：Normalize-Then-Precondition: A Hierarchical Approach to Marginal Scale and Interaction Geometry for LLM Training

## 一句话总结
本文提出了 **NormPre** 优化器家族，通过“先归一化后预条件”的层次化框架，分离并依次处理模型参数矩阵的边缘尺度信息与行/列间交互几何信息；在 GPT-2、LLaMA 和 Qwen3 的大规模预训练实验中，两种变体（NormPre-G 与 NormPre-L）均在相同训练预算下稳定优于 AdamW、Muon 和 MANO 等基线。

## 研究问题与动机
1. **现有矩阵优化器对几何信息的处理缺乏层次化组织**：如 Muon 通过完整 Gram 矩阵同时对边缘尺度（对角线）与交叉行交互（非对角线）进行联合变换，导致主导模式易被极端行范数淹没，难以分别刻画“尺度均衡”与“方向交互”。
2. **纯归一化方法仅利用对角信息，丢失交互几何**：MANO、MOGA 等方法仅依赖 Gram 对角线做行/列归一化，虽能消除尺度差异，但未进一步修正保留在归一化更新中的相关性结构。
3. **正交化与归一化在 Gram 视角下具有统一结构，但二者结合机制未形式化**：完整 Gram 表示（$\Phi = (XX^\top)^{-1/2}X$）与对角 Gram 表示（$\Psi = \mathrm{Diag}(\mathrm{diag}(XX^\top))^{-1/2}X$）揭示了从“联合变换”到“分层处理”的可能路径。
4. **缺乏对谱预条件目标的系统性理论刻画**：如何在归一化基础上设计全局或局部的谱变换，并给出收敛保证，尚不明确。

## 核心贡献（创新点）
1. **提出 Normalize-Then-Precondition 层次化框架**：首次将边缘尺度归一化与交互谱预条件解耦为两个独立阶段，使优化器能够分别控制“行范数均衡”与“方向相关性重塑”。
2. **设计两种谱预条件变体 NormPre-G 与 NormPre-L**：NormPre-G 基于谱范数最速下降进行全局正交化；NormPre-L 基于正则化谱最速下降并限制活跃模式数，保留归一化基础更新的稠密结构。
3. **给出可扩展的实现方案**：NormPre-G 采用 Newton-Schulz 迭代近似矩阵符号函数；NormPre-L 采用随机草图（randomized sketching）近似前导交互特征子空间，复杂度从 $\mathcal{O}(m^2n+m^3)$ 降至 $\mathcal{O}((p{+}1)mn\ell+(m{+}n)\ell^2+\ell^3)$。
4. **建立 $\mathcal{O}(T^{-1/2})$ 收敛性保证**：在无动量简化设定下，证明 NormPre-G 与 NormPre-L（Exact/Sketch）均达到一阶方法的收敛速率，并将收敛界与径向比、谱因子等几何量显式关联。
5. **在多种架构与数据上验证优越性与性能-效率权衡**：GPT-2 Small、LLaMA（130M–1.3B）、Qwen3（0.6B–1.7B）预训练中，两种变体均优于 AdamW、Muon、MANO；NormPre-L 在损失相近条件下显著降低优化器延迟。

## 方法详解
### 1. 对角 Gram 归一化（Normalize）
- 对松弛切线动量矩阵 $X_t$，构造行范数对角矩阵 $D(X_t)=\mathrm{Diag}(\|x_{t,1}\|_2,\dots,\|x_{t,m}\|_2)$。
- 得到边缘归一化更新 $\Psi_t = D(X_t)^{-1}X_t$，使每行范数为 1，消除尺度差异。
- 归一化后的交互矩阵 $\Gamma_t=\Psi_t\Psi_t^\top$ 的对角线为 1，非对角线为行间余弦相似度，保留方向相关性。

### 2. 全局谱预条件（NormPre-G）
- 求解谱范数约束下的最速下降问题：$\max_{\|T\|_{\mathrm{op}}\leq 1}\langle \Psi_t,T\rangle_F$。
- 解为 $T_t^{\mathrm{G}}=\mathrm{msign}(\Psi_t)=\widetilde{U}\widetilde{V}^\top$（薄 SVD $\Psi_t=\widetilde{U}\widetilde{\Sigma}\widetilde{V}^\top$）。
- 实际通过 5 步 Newton-Schulz 迭代近似矩阵符号函数，复杂度与 Muon 同阶 $\mathcal{O}(mns)$。

### 3. 局部谱预条件（NormPre-L）
- 求解正则化谱最速下降：$\min_{\|T\|_{\mathrm{op}}\leq 1}\frac{1}{2}\|T-\Psi_t\|_F^2$，等价于将 $\Psi_t$ 投影到单位谱范数球内。
- 解为 $T_t^{+}=P_+\Psi_t$，其中 $P_+=I+\widetilde{U}_{\mathcal{A}}(\Lambda_{\mathcal{A}}^{-1/2}-I)\widetilde{U}_{\mathcal{A}}^\top$，仅裁剪奇异值 $>1$ 的模式。
- 引入谱预算 $r$，仅选取前 $r$ 个活跃特征值对应的特征向量进行局部变换，得到 $T_t^{\mathrm{L}}=P^{\mathrm{L}}\Psi_t$。
- 通过随机草图（高斯投影 + 幂迭代 + Rayleigh–Ritz 提取）近似前导特征子空间，秩 $r=32$、过采样 $o=8$、幂迭代 $p=1$。

### 4. 辅助设计
- **交替行列方向**：每步切换行/列作为主方向（$k_t=t\bmod 2$），对称处理矩阵两个轴。
- **松弛切线动量**：$x_{t,i}=\bar{m}_{t,i}-\langle\bar{m}_{t,i},\bar{w}_{t,i}\rangle\bar{w}_{t,i}$，保留径向分量以稳定训练。
- **一致更新 RMS**：将更新矩阵缩放到目标 RMS=0.2，便于超参迁移与公平对比。

### 5. 收敛性分析
- 在 L-smooth 且下有界的损失函数上，假设梯度与权重夹角满足 $\sin\phi\geq\gamma$、归一化行与非正交性满足 $\gamma'$、谱因子 $\chi_t$ 有界，取学习率 $\eta=C/\sqrt{T+1}$ 可得：
$$\min_{0\leq t\leq T}\|\nabla\mathcal{L}(W_t)\|_F\leq\mathcal{O}(T^{-1/2})$$
- Sketch 变体的收敛界在近似误差 $\delta$ 满足 $\delta\max_t\lambda_{t,1}<\epsilon\gamma$ 时保持相同速率。

## 实验与结果
### 数据集与模型
- **GPT-2 Small**（124M）/ OpenWebText
- **LLaMA-130M/350M/1.3B** / C4
- **Qwen3-0.6B/1.7B** / Pile
- 统一配置：序列长 1024、有效批大小 512、10k 步训练、线性 warmup+cosine 衰减、weight decay=0.1、梯度裁剪 1.0。

### 基线方法
- **AdamW**：坐标级自适应
- **Muon**：矩阵正交化（Newton-Schulz 近似矩阵符号）
- **MANO**：动量投影 + 交替归一化

### 主要结果（验证损失 ↓）
| 模型 | 数据集 | AdamW | Muon | MANO | **NormPre-G** | **NormPre-L** |
|---|---|---|---|---|---|---|
| GPT-2 Small | OpenWebText | 3.1444 | 3.1064 | 3.1156 | **3.0667** | 3.0826 |
| LLaMA-130M | C4 | 3.1363 | 3.1019 | 3.1102 | **3.0737** | 3.0930 |
| LLaMA-350M | C4 | 3.0378 | 2.9999 | 2.9946 | **2.9678** | 2.9758 |
| LLaMA-1.3B | C4 | 2.9385 | 2.9037 | 2.8963 | **2.8571** | 2.8662 |
| Qwen3-0.6B | Pile | 2.8956 | 2.8335 | 2.8382 | **2.7967** | 2.8066 |
| Qwen3-1.7B | Pile | 2.6758 | 2.6408 | 2.6205 | **2.5868** | 2.5897 |

- **相对最强基线 Muon 的提升**：GPT-2 Small 上 NormPre-G 降低 0.0397、NormPre-L 降低 0.0238；LLaMA-1.3B 上分别降低 0.0392 与 0.0301。
- **性能-效率权衡**：NormPre-G 优化器延迟较 Muon 增加约 5–7%，端到端步时增加 0.08–0.53%；NormPre-L 延迟降低 38–67%，端到端步时降低 1–4%，吞吐提升 1–4%。

### 消融实验
- 松弛切线投影优于严格切线投影与无切线处理。
- 交替行列方向与仅行方向表现接近，默认采用交替以对称处理。
- 从归一化后更新 $\Psi_t$ 提取交互几何优于从原始更新 $X_t$ 提取。
- 谱预算 $r=32$ 在性能与效率间取得平衡，增大 $r$ 可进一步降低损失但增加成本。

### 谱动力学分析
- 归一化后特征谱出现各向异性，多个秩的特征值被抬升，需进一步谱变换。
- 全局预条件的谱变换能量中，前 32 个活跃模式（$\lambda_i>1$）贡献约 62–67%，说明局部预条件可捕获大部分变换能量。

## 相关工作脉络
1. **Muon（Jordan et al., 2024; Liu et al., 2025）**：基于 Newton-Schulz 迭代的矩阵正交化优化器，使用完整 Gram 矩阵联合处理尺度与交互；本文将其解读为全谱预条件，并提出先归一化再预条件的替代路径。
2. **MANO（Gu & Xie, 2026）**：结合动量投影与交替行列归一化，仅利用 Gram 对角线；本文在归一化基础上进一步引入谱预条件以利用交互信息。
3. **MOGA（Xu et al., 2026）**：从均值归一化算子范数的最速下降视角推导行/列归一化更新；本文同样采用归一化，但强调层次化组织与谱变换的两种实现。
4. **SWAN、MuonEq 等预归一化方法**：在正交化前对梯度或动量做数值稳定化处理；本文的归一化是框架的第一阶段而非预处理，且后续可接全局或局部谱变换。
5. **K-FAC、Shampoo、SOAP**：基于曲率或累积统计的块/矩阵预条件方法；本文聚焦于直接对更新矩阵进行几何变换，不涉及二阶曲率近似。
6. **坐标级自适应优化器（AdamW、Lion、Sophia 等）**：仅处理对角或低秩统计；本文处理完整矩阵几何，属于更广义的矩阵优化器范畴。

## 局限性与未来方向
1. **硬件感知实现待优化**：Newton-Schulz 迭代与随机草图在 GPU/TPU 上的 kernel 融合与内存访问模式未针对特定硬件深度优化，存在效率提升空间。
2. **实验规模有限**：当前验证最大至 1.7B 参数，未覆盖数十亿及以上规模的 LLM 训练，需进一步扩展以验证泛化性。
3. **谱预算 $r$ 需人工设定**：默认 $r=32$ 在实验中表现良好，但缺乏自适应机制以在不同层、不同训练阶段动态调整。
4. **理论假设较强**：收敛性证明要求梯度与权重保持一定夹角、谱因子有界等条件，实际训练中的满足程度尚需进一步验证。
5. **仅评估预训练损失**：未测试下游任务微调、推理效率、长上下文泛化等指标，完整性能评估有待补充。

## 研究启发与可借鉴点
1. **层次化几何信息组织范式**：将边缘尺度归一化与交互谱预条件解耦为两阶段框架，为其他矩阵优化器设计提供了可复用的 modular 思路。
2. **松弛切线投影的稳定作用**：保留径向分量比严格切线投影更鲁棒，提示在正交化/归一化前对动量进行“软切线”处理值得推广。
3. **随机草图加速局部谱变换**：通过高斯投影 + 幂迭代 + Rayleigh–Ritz 提取前导特征子空间，以低复杂度近似精确对角化，可迁移至其他需低秩谱信息的优化场景。
4. **谱变换能量的局部集中性**：前 32 个活跃模式贡献超 60% 变换能量，说明在工程中可用低秩谱近似替代全谱变换以大幅降本。
5. **统一的全 Gram vs 对角 Gram 视角**：将 Muon 等正交化方法与归一化方法置于同一 Gram 表示下分析，有助于发现二者联系与改进空间。

## 关键术语表
- **Normalize-Then-Precondition**：层次化优化框架，先通过 Gram 对角线归一化消除边缘尺度差异，再对归一化后的交互几何进行谱预条件变换。
- **Diagonal-Gram Normalization**：仅利用矩阵乘积 $XX^\top$ 的对角元素（行范数）构造归一化更新 $\Psi=D(X)^{-1}X$，使各行范数均为 1。
- **Full-Gram Representation**：完整 Gram 矩阵 $(XX^\top)^{-1/2}X$ 表示，同时编码行尺度与行间相关性，Muon 的矩阵符号变换即属此类。
- **Global Spectral Preconditioning**：对归一化更新 $\Psi$ 施加全谱正交化（矩阵符号函数），使所有奇异值变为 1，对应谱范数最速下降解。
- **Localized Spectral Preconditioning**：仅对 $\Psi$ 中奇异值大于 1 的前 $r$ 个活跃模式进行裁剪，其余模式保持不变，形式化为正则化谱最速下降问题。
- **Newton-Schulz Iteration**：用于近似矩阵符号函数的迭代方法，每步形如 $Y_{k+1}=\frac{1}{2}Y_k(3I-Y_k^2)$，可在固定步数内收敛至半正交矩阵。
- **Randomized Sketching**：通过高斯随机投影将大矩阵映射到低维子空间，再用 Rayleigh–Ritz 方法近似前导特征对，以 $\mathcal{O}(mn\ell)$ 复杂度替代 $\mathcal{O}(m^3)$ 对角化。
- **Relaxed Tangent Momentum**：将动量分解为切线与径向分量，保留部分径向信息（$x_i=m_i-\langle m_i,w_i\rangle w_i$）以增强数值稳定性。

## 可复现要素
- **数据集**：OpenWebText、C4、Pile 均为公开数据集，论文提供预处理流程与 tokenizer 说明。
- **代码开源**：作者已将代码开源至 [https://github.com/zx-gong/NormPre](https://github.com/zx-gong/NormPre)。
- **关键超参**：
  - Newton-Schulz 迭代步数：5
  - 谱预算 $r$：32
  - 随机草图过采样 $o$：8
  - 幂迭代次数 $p$：1
  - 目标更新 RMS：0.2
  - 动量系数 $\mu$：0.95
  - weight decay：0.1
  - 学习率 schedule：线性 warmup + cosine 衰减至峰值 10%
- **硬件环境**：NVIDIA A100 80GB GPU，FP32 参数 + BF16 autocast（Qwen3 使用纯 BF16）。
- **训练配置**：序列长 1024、有效批大小 512、10,000 步、梯度裁剪 1.0、每 500 步评估。
