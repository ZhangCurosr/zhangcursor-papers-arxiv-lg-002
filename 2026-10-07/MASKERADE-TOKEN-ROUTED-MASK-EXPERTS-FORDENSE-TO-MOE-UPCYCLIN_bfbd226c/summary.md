---
title: "MASKERADE-TOKEN-ROUTED-MASK-EXPERTS-FORDENSE-TO-MOE-UPCYCLIN"
source: https://arxiv.org/pdf/2610.07809v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:51:16"
field: "大模型高效微调与架构转换"
keywords: ["Mixture-of-Experts", "Dense-to-MoE Upcycling", "Mask Learning", "Vision-Language Models", "Sparse subnetworks", "Token routing"]
innovations: ["将 MoE 专家定义为冻结预训练 FFN 上的学习性二元 mask，无需独立专家权重矩阵", "提出重要性初始化（|W|+噪声）加速 mask 学习并提升最终性能", "统一支持 neuron-structured、semi-structured 2:4 和 unstructured 三种 mask 粒度并系统评估"]
benchmarks: ["GQA", "MME", "POPE", "ScienceQA", "TextVQA"]
---

# 论文速读：MASKERADE-TOKEN-ROUTED-MASK-EXPERTS-FORDENSE-TO-MOE-UPCYCLING

## 一句话总结
MASKerade 提出了一种新的 dense-to-MoE upcycling 方法：不复制并独立训练 FFN 权重，而是让多个专家共享冻结的预训练 FFN，每个专家通过一个**学习到的二元 mask** 定义不同的稀疏子网络，再通过 token 级 router 动态选择组合。该方法在 Qwen 和 Gemma 两个视觉语言模型系列上的五个基准中均取得最优性能。

## 研究问题与动机
- **问题**：现有的 dense-to-MoE upcycling 方法（如 Sparse Upcycling、Drop-Upcycling）通过复制 FFN 权重构建独立专家，造成参数冗余和较高计算/存储开销。
- **动机 1**：能否让专家通过选择预训练 FFN 中的**连接**而非学习独立权重来定义多样性？
- **动机 2**：在冻结权重的条件下，通过 mask 学习与路由联合优化，能否在不增加额外参数矩阵的情况下实现有效的多模态指令微调？
- **动机 3**：如何支持神经元级（neuron-structured）、半结构化（semi-structured）和非结构化（unstructured）等多种 mask 粒度，以适应不同硬件加速需求（如 NVIDIA 2:4 稀疏）？

## 核心贡献（创新点）
1. **提出 MASKerade，首次将 expert 定义为冻结预训练 FFN 上的学习性二元 mask**：用 mask 学习替代专家权重适应，所有 expert 共享同一组权重 $W_\ell$，仅通过 $M_{\ell,e}$ 选择不同子网络。
2. **联合优化 mask 分数与 token 级 router**：mask 分数通过 straight-through 估计器反向传播，router 用 deterministic top-k 选择并配合 load-balancing loss，在训练和推理阶段均无需 dropout dispatch。
3. **统一支持三种 mask 粒度**（neuron-structured、semi-structured 2:4、unstructured），并展示不同粒度在不同实验设置下的性能权衡。
4. **重要性初始化（importance initialization）**：用预训练权重幅值加独立高斯噪声作为 mask 分数初始值，显著降低初始训练损失并提升最终性能。
5. **系统验证冻结权重的可行性**：在所有五个基准上，冻结 FFN 的 MASKerade 在激活参数量匹配的条件下优于所有独立专家基线（MoEfication、Drop-Upcycling、ToMoE 等）。

## 方法详解

### 3.1 Experts as Frozen-Weight Subnetworks
对选定的 FFN 层 $\ell$，冻结权重 $W_\ell$ 不变。专家 $e$ 的等效参数为：
$$\widetilde{W}_{\ell,e}^{(a)} = W_\ell^{(a)} \odot M_{\ell,e}^{(a)}, \quad M_{\ell,e}^{(a)} \in \{0,1\}^{\text{shape}(W_\ell^{(a)})}$$
对于 gated FFN（gate/up/down 三投影 $W^{(g)}, W^{(u)}, W^{(d)}$），专家 $e$ 的计算为：
$$F_{\ell,e}(h) = \widetilde{W}_{\ell,e}^{(d)}\left[\phi_\ell(\widetilde{W}_{\ell,e}^{(g)}h) \odot (\widetilde{W}_{\ell,e}^{(u)}h)\right]$$

### 3.2 Token-Level Routing
可训练线性 router $R_\ell \in \mathbb{R}^{E \times d}$ 输出概率 $p_{\ell,e}(h) = \text{softmax}(R_\ell h)_e$，top-k 确定性选择得到集合 $S_\ell(h)$。组合输出：
$$F_\ell^{\text{MoE}}(h) = \sum_{e \in S_\ell(h)} g_{\ell,e}(h) F_{\ell,e}(h), \quad g_{\ell,e}(h) = \begin{cases} p_{\ell,e}(h), & k=1 \\ \frac{p_{\ell,e}(h)}{\sum_{j \in S_\ell(h)} p_{\ell,j}(h)}, & k \geq 2 \end{cases}$$

### 3.3 Mask Parameterization
三种粒度：
- **Semi-structured (N:M)**：每行分成长为 $M$ 的连续块，每块保留幅度最大的 $N$ 个位置（默认 2:4）。
- **Unstructured**：在展平后的所有 $P$ 个位置中选 top-$q$（$q = \lceil \rho P \rceil$）。
- **Structured**：在每个专家学习一个长度为 $d_f$ 的 score 向量，该 mask 共享到对应 gate/up 行和 down 列：$M^{(g)} = M^{(u)} = m \mathbf{1}_d^\top, \ M^{(d)} = \mathbf{1}_d m^\top$。

### 3.4 联合训练
- **Score 初始化**：重要性初始化 $S = |W| + \epsilon, \ \epsilon \sim \mathcal{N}(0, [\alpha_\text{init} \cdot \text{std}(|W|)]^2)$，$\alpha_\text{init}=0.1$；结构化 mask 的神经元重要性取 gate/up/down 三个切片的均值。
- **梯度估计**：forward 使用 hard mask，backward 用 straight-through：$\frac{\partial \mathcal{L}}{\partial M_{\ell,e}^{(a)}} = W_\ell^{(a)} \odot \frac{\partial \mathcal{L}}{\partial \widetilde{W}_{\ell,e}^{(a)}}$。
- **目标函数**：$\min_{\{S_\ell, R_\ell\}} \mathcal{L}_\text{SFT}(\theta_0, S, R) + \lambda_\text{bal} \sum_\ell \mathcal{L}_\text{bal}^{(\ell)}$，其中 $\mathcal{L}_\text{bal}^{(\ell)} = E \sum_e \bar{p}_{\ell,e} f_{\ell,e}$，$\bar{p}$ 为平均路由概率，$f$ 为 token 分配比例。

## 实验与结果
- **模型 backbone**：小模型 Qwen3-1.7B、Gemma3-1B；大模型 Qwen3.6-27B、Gemma4-31B。均使用 CLIP ViT-L/14 视觉编码器，665K 多模态指令数据微调。
- **基准**：GQA、MME、POPE、ScienceQA (SQA)、TextVQA（五个 VLM 基准）。
- **主要结果（小模型，Qwen3-1.7B）**：MASKerade（Semi-structured 2:4，Act. Params=1.72B）取得 **GQA=62.00、MME=1332.4、POPE=86.49、SQA=63.48、TextVQA=52.50**，五项全优；优于 Drop-Upcycling（2.25B）、MoE-LLaVA（2.25B）等更高激活参数的方法。
- **主要结果（大模型，Qwen3.6-27B）**：MASKerade（Act. Params=26.90B）取得 **GQA=66.37、MME=1724.9、POPE=87.68、SQA=94.79、TextVQA=81.08**，同样五项全优。
- **对比结论**：在激活参数量与 Dense MLP 匹配时，MASKerade 仍全面超越所有外部 baselines；在最低预算配置（1.46B 激活参数）下，MASKerade 的 1:4/top-2 和 2:4/top-1 两种变体均优于所有四个外部方法。
- **推理速度**：H100 上 MASKerade 相对 Vanilla MoE 在 1.7B/1B 上近乎持平（0.93×/0.99×），在 27B/31B 上分别快 21%/25%（0.79×/0.75×），越大模型优势越明显。
- **消融要点**：
  - **Router 联合训练** vs 冻结随机 router：全维度提升。
  - **Importance 初始化** vs Random：2:4 mask 上提升最显著。
  - **Mask 粒度比较**：Unstructured 和 Semi-structured 均优于 Neuron-structured；2:4 因硬件加速成为默认配置。
  - **Mask Jaccard 重叠**约 0.59，表明专家间共享但非完全相同的连接结构；四层共享核心（latent shared expert）构成所有 mask 的共同基础。
  - **路由可替换性诊断**：单一层替换选中 expert 导致 NLL 上升、准确率下降，说明 router 与 mask 的绑定具有功能性意义。

## 相关工作脉络
1. **Sparse Upcycling（Komatsuzaki et al., 2023） / Drop-Upcycling（Nakamura et al., 2025）**：复制 FFN 权重并微调（部分随机化），MASKerade 不复制权重，改学 mask 选择连接。
2. **MoEfication（Zhang et al., 2022） / LLaMA-MoE（Zhu et al., 2024）**：将 FFN 神经元分组为专家或分区剪枝，MASKerade 直接在完整 FFN 张量上学习二元 mask。
3. **MaskLLM（Fang et al., 2024） / MFT（Zhang et al., 2026）**：单 mask 学习用于压缩或继续微调，MASKerade 扩展为多 mask + 路由选择，形成 MoE 架构。
4. **MoM（Liu et al., 2025） / SIMoE（Chen et al., 2025）**：在共享可训练 delta 参数上施加 mask，MASKerade 完全冻结原始权重，无 delta 参数。
5. **Wanda（Sun et al., 2024）**：基于权重幅值和激活统计的非学习性剪枝，MASKerade 通过端到端梯度学习 mask，在多个基准上显著优于固定 mask 策略。
6. **MaskMoE（Su et al., 2024）**：在 routing logits 上施加 token-frequency-dependent mask 限制专家可见性，而非定义稀疏 FFN 子网络。

## 局限性与未来方向
- **仅测试视觉语言模型**：目前仅在 Qwen/Gemma 系列 VLM 上验证，对纯 LLM 或其他模态（如音频）的泛化性未检验。
- **冻结权重可能限制上限**：消融显示允许微调 FFN 权重可进一步提升性能（4e/top-2 从 62.00 升至 62.73 GQA），说明冻结假设是有意约束而非最优。
- **路由稳定性**：不同 backbone 对 top-k 和 expert bank 大小的最优配置存在差异，缺乏统一调参准则。
- **论文自述未来方向**：探索推理 kernel 复用重叠计算同时保留 expert 特异性非线性变换，评估端到端延迟与内存权衡。

## 研究启发与可借鉴点
1. **Mask-as-Expert 范式**：在资源受限场景下，冻结共享权重+学习 mask 选择是一种低成本构建专家多样性的有效策略，可迁移至 LLM 领域或持续学习场景。
2. **重要性初始化（$S = |W| + \epsilon$）**：简单且有效，可直接复用于其他 mask 学习任务（如结构化剪枝、adapter 选择）以改善收敛。
3. **Straight-through + load-balancing 的组合**：对于任意离散选择问题的联合优化，该梯度估计方式与 Fedus et al. 的概率-负载平衡损失值得借鉴。
4. **硬件感知的半结构化 mask（2:4）**：兼顾算法性能与 GPU 加速，展示了如何将硬件约束融入模型设计；可推广至其他 sparse architecture search 工作。
5. **Latent shared expert 分析**：通过 Jaccard 重叠和共享连接分解来理解专家多样性，是一种可复用的解释性分析框架。

## 关键术语表
- **Dense-to-MoE Upcycling**：将已预训练的稠密模型转化为 Mixture-of-Experts 架构，复用已有预训练投资。
- **MASKerade**：本文提出的方法，将 expert 定义为冻结 FFN 上的学习性二元 mask，结合 token 级 router 实现稀疏专家路由。
- **Semi-structured 2:4 mask**：NVIDIA GPU 硬件加速的稀疏模式，每 4 个连续权重中保留最大的 2 个。
- **Straight-through estimator**：在 hard mask forward 时用 identity 近似梯度以更新连续 score 的训练技巧。
- **Load-balancing loss**：惩罚专家使用概率与分配比例之间的偏差，防止少数专家垄断所有 token。
- **Importance initialization**：以预训练权重幅值 $|W|$ 加高斯噪声作为 mask score 的初始值，促进快速收敛和多样性。
- **Latent shared expert**：所有专家 mask 的交集构成的隐含共享子网络，代表各 expert 共用的"核心"连接。
- **Dropless dispatch**：训练和推理阶段均采用确定性 top-k 选择，不使用随机采样或 dropout 调度。

## 可复现要素
- **数据集**：665K 多模态指令混合数据（MoE-LLaVA pipeline），论文未明确公开具体数据集列表。
- **代码**：已开源，见 https://github.com/Ming-K9/MASKerade。
- **权重**：论文未提供预训练权重下载链接。
- **关键超参**：E=4 experts，top-2 routing，2:4 semi-structured mask，$\alpha_\text{init}=0.1$，$\lambda_\text{bal}=0.01$，learning rate $5\times10^{-4}$，cosine schedule with 3% warmup，1 epoch AdamW，bfloat16，batch size=128（小模型，8×H100）/ 256（大模型，64×B200）。
