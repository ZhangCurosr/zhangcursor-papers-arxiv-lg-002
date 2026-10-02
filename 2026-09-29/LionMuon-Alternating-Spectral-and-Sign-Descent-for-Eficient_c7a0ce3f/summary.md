---
title: "LionMuon-Alternating-Spectral-and-Sign-Descent-for-Eficient"
source: https://arxiv.org/pdf/2609.35297v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:52:50"
field: "大规模语言模型训练优化"
keywords: ["LLM 训练优化器", "Muon", "Lion", "符号优化", "谱方法", "分布式训练"]
innovations: ["周期性交替 Muon 谱步与 Lion 符号步，共享单动量缓冲区", "重尾噪声下交替优化器的收敛界与最优参数分析", "LionMuon P=5 在 4-GPU 数据并行中比 Muon 节省约 1/3 wall-clock 时间"]
benchmarks: ["FineWeb", "WikiText-103"]
---

# 论文速读：LionMuon-Alternating-Spectral-and-Sign-Descent-for-Eficient

## 一句话总结
论文提出 LionMuon 优化器，通过每隔 P 次迭代执行一次 Muon 谱步骤、其余时间执行 Lion 符号步骤的方式，在保证更新质量的同时显著降低计算和通信开销；在 124M 和 355M 模型上的实验表明，LionMuon 在相同 token 数下可达到低于 Muon、AdamW、Lion 和 Signum 的损失。

## 研究问题与动机
- **核心问题**：LLM 预训练成本高昂，如何在保证收敛质量的前提下降低优化器的每步计算和通信开销？
- **Muon 的优势与代价**：Muon 通过 Newton–Schulz 迭代计算谱步（matrix sign），方向性更强，但每步需多次矩阵乘法，且在分布式训练时需额外的 all-reduce 通信，导致吞吐损失约 5%–10% 甚至更高。
- **Sign 方法的廉价性**：Lion/Signum 等符号步方法仅涉及逐元素操作，无需全局通信，可与现有分布式基础设施无缝对接。
- **动机**：能否保留 Muon 的更新质量，同时仅每 P 步支付一次其高昂计算与通信代价？

## 核心贡献（创新点）
1. **LionMuon/SignMuon 交替优化框架**：每 P 次迭代执行一次 Muon 谱步，其余执行 Lion/Signum 符号步，共享单个双 EMA 动量缓冲区，状态大小仅为 AdamW 的一半。
2. **重尾噪声下的收敛性分析**：给出了 LionMuon 在重尾噪声（heavy-tailed noise）下的收敛界，证明周期 P 可在 Muon 和 Lion 的光滑常数与噪声常数之间进行插值，并给出 LionMuon 快于两者的充分条件。
3. **系统性实验验证**：在 124M 模型（FineWeb、WikiText-103）和 355M 模型（FineWeb）上，LionMuon（P=2, P=5）在所有基线（Muon、AdamW、Lion、Signum）中取得最低损失；在 4-GPU 数据并行训练中，LionMuon P=5 以更少 wall-clock 时间达到 Muon 的最终损失，同时优于 Dion 和 MuonBP。

## 方法详解
- **核心算法（Algorithm 1）**：
  - 维护单个动量缓冲区 $M_t$，每步按 Lion 方式插值：$\hat{G}_t = \beta_1 M_{t-1} + (1-\beta_1) G_t$
  - 若 $t \mod P = 0$，执行 Muon 步：$W_{t+1} = W_t - \eta_M (c(\hat{G}_t) \text{NS}_{K_{NS}}(\hat{G}_t) + \lambda W_t)$
  - 否则执行 Lion 步：$W_{t+1} = W_t - \eta_L (\text{sign}(\hat{G}_t) + \lambda W_t)$
  - 动量更新：$M_t = \beta_2 M_{t-1} + (1-\beta_2) G_t$（每步执行）
- **关键设计**：
  - 所有 2D 矩阵参数（含 embedding）均使用 LionMuon 更新；1D 参数（bias、norm gain）使用 AdamW（固定 $\eta=10^{-3}$）
  - 学习率比例 $\alpha = \eta_M / \eta_L$ 是关键超参，理论建议将其设为较大值以使两种步的梯度范数量级相当
  - 牛顿–舒尔兹（Newton–Schulz）迭代步数 $K_{NS}=5$，缩放因子 $c(A) = 0.2\sqrt{\max(\text{rows}, \text{cols})}$

## 实验与结果
- **数据集与模型**：
  - FineWeb（Penedo et al., 2024）和 WikiText-103（Merity et al., 2017）
  - 124M 模型：12 层、宽度 768、batch 32×512 tokens、64,000 步
  - 355M 模型：24 层、宽度 1024、15,650 步
- **主要结果（124M，单 GPU）**：
  - LionMuon P=2 和 P=5 在两个数据集上均取得最低损失，优于 Muon、AdamW、Lion、Signum
  - 与 Muon 的差距超过种子波动的十倍
- **355M 迁移实验**：
  - 所有超参从 124M 直接迁移，未重新调优
  - LionMuon P=5 保持优势，验证了超参的可迁移性
- **4-GPU 数据并行训练（124M）**：
  - LionMuon P=5 每步暴露通信量仅为 Muon 的 1/5（34 MB vs 170 MB）
  - 在 PCIe 上达到 Muon 最终损失所需时间约为 Muon 的 2/3
  - 优于 Dion（低秩同步）和 MuonBP（块周期正交化）

## 相关工作脉络
1. **Sign-based 方法**：Signum（Bernstein et al., 2018）、Lion（Chen et al., 2023）— LionMuon 在其基础上引入周期性谱步
2. **Muon 及其变体**：Muon（Jordan et al., 2024）、MuonClip、Gluon（Riabinin et al., 2025）、HTMuon（Pang et al., 2026）— 本文不改进单步谱计算，而是减少调用频率
3. **分布式 Muon 优化**：Dion（Ahn et al., 2025，低秩因子同步）、MuonBP（Khaled et al., 2026，块周期正交化）、DMuon（Chen et al., 2026b，单设备所有权）— 本文沿迭代轴而非矩阵分片轴优化
4. **LiMuon**（Huang et al., 2025）和 **OLion**（Wang et al., 2026）— 同期工作，分别用低秩 SVD 替代 NS 迭代、组合正交化与符号步；本文思路不同且更通用
5. **Optimizer switching**：SWATS、AdaBound、AGD — 已有切换优化器的工作，但本文首次研究 Muon 与 Lion 的周期性交替

## 局限性与未来方向
- **局限性**：
  - 目前仅在 124M 和 355M 模型上验证，更大规模模型（如 7B+）的表现未知
  - 理论界限中的光滑常数和噪声常数的上界估计较保守（使用了最坏情况的范数不等式）
  - 固定周期 P 的假设可能不够灵活
- **未来方向**（论文自述）：
  - 更大模型和多节点训练的扩展
  - 自适应周期策略（如根据训练阶段动态调整 P）
  - 对 Newton–Schulz 迭代误差的更精细分析

## 研究启发与可借鉴点
1. **"交替优化"范式**：将昂贵步骤与廉价步骤周期性交替，是降低优化器开销的通用思路，可迁移到其他高成本操作（如精确 Hessian 近似、全矩阵归一化等）
2. **学习率比例的设计原则**：理论推导与实验均表明，$\eta_M/\eta_L$ 应设为较大值以平衡不同范数下的梯度量级，这一原则可指导其他混合优化器的调参
3. **重尾噪声下的收敛分析框架**：论文在 heavy-tailed noise 假设下建立收敛界，对 LLM 训练场景更具现实意义，分析方法可复用于其他符号类优化器
4. **超参迁移的实证验证**：355M 实验直接使用 124M 调优参数，证明小模型调优结果可迁移至中等模型，减少大模型实验成本

## 关键术语表
- **Newton–Schulz 迭代**：用于近似矩阵 sign 函数的迭代算法，通过多项式逼近计算 matrix sign$(X) = UV^\top$
- **线性最小化预言机（LMO）**：Frank-Wolfe 优化中的核心子问题求解器，形式为 $\arg\min_{\|S\|\leq 1} \langle G, S\rangle$
- **重尾噪声（Heavy-tailed noise）**：梯度噪声分布具有重尾特性（尾部指数 $\kappa \in (1,2]$），LLM 训练中常见现象
- **双 EMA 动量**：Lion 使用的两个不同时间尺度的指数移动平均（$\beta_1$ 用于方向插值，$\beta_2$ 用于动量更新）
- **谱范数（Spectral norm）**：矩阵的最大奇异值，对应 LMO 下的 Muon 更新
- **无穷范数（Infinity norm）**：矩阵元素绝对值的最大值，对应 LMO 下的符号步更新

## 可复现要素
- **代码开源**：https://github.com/brain-lab-research/lion-muon（含训练脚本、分布式协议、绘图脚本）
- **数据集**：FineWeb（公开）、WikiText-103（公开）
- **关键超参**：
  - $K_{NS} = 5$，$c(A) = 0.2\sqrt{\max(m,n)}$
  - Lion/LionMuon：$\beta_1 = 0.9, \beta_2 = 0.99$
  - Muon/SignMuon：$\beta = 0.9$
  - AdamW：$\beta_1 = 0.8, \beta_2 = 0.999$
  - Weight decay：0.1，Gradient clipping：0.5
  - LR schedule：cosine，warmup 3000 步
- **硬件**：单 GPU 实验使用 GPT-2-style decoder；分布式实验使用 4×H200 GPU（NVLink）
