---
title: "LEARNING-FUNCTIONAL-SUBSPACES-FOR-NEURAL-NETWORK-COMPRESSION"
source: https://arxiv.org/pdf/2609.40127v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 21:57:38"
field: "高效神经网络推理"
keywords: ["low-rank compression", "neural network pruning", "transformer compression", "learnable subspaces", "model distillation", "KV cache optimization"]
innovations: ["端到端学习正交投影器替代闭式局部标准进行子空间选择", "测量-KL 排名分配按边际成本贪心分配压缩预算", "绑定组共享投影器实现 13.5× 内存缩减的潜在缓存"]
benchmarks: ["WikiText-2 perplexity", "ARC-e", "PIQA", "OpenBookQA", "WinoGrande", "HellaSwag", "MathQA", "CIFAR-100 accuracy"]
---

# 论文速读：LEARNING-FUNCTIONAL-SUBSPACES-FOR-NEURAL-NETWORK-COMPRESSION

## 一句话总结
论文提出 Learnable Subspace Projections (LSP)，一种端到端学习 Transformer 模型中应丢弃子空间的低秩压缩方法：每个线性层（或共享激活的层组）分配一个正交投影器，冻结预训练权重后联合优化这些投影器，以网络输出分布的全局 KL 散度为目标函数，显著优于现有基于局部标准的低秩压缩方法。

## 研究问题与动机
- **现有方法忽略误差传播**：激活能量导向（ASVD、SliceGPT）或层重建误差导向（SVD-LLM、Swift-SVD）的方法仅基于局部标准选择要移除的子空间，忽略了这些方向对网络输出的实际重要性，导致高压缩率下误差逐层累积、性能崩溃。
- **曲率近似仅适用于小幅扰动**：FW-SVD、LLM-Surgeon 等方法通过局部曲率代理将损失引入压缩决策，但在激进压缩下严重偏离稠密权重，代理失效。
- **权重的谱能量 ≠ 功能重要性**：预训练 Transformer 权重的奇异值衰减缓慢，基于谱能量的截断会错误保留高能量但低功能的子空间，同时丢弃低能量但关键的功能子空间。
- **需要端到端识别"功能性子空间"**：压缩的本质是识别对目标任务输出最不敏感的方向，而非统计冗余最小的方向。

## 核心贡献（创新点）
- **可学习子空间投影**：将压缩问题重新表述为学习正交投影器的问题，通过全局目标端到端优化所有层的去除子空间，而非依赖闭式局部标准。
- **白化初始化 + 测量-KL 排名分配**：从白化 SVD 截断初始化每个投影器，并通过单独应用候选截断、测量其对模型输出的 KL 散度来按每参数成本分配删除方向的数量，实现了感知损失的初始化与预算分配。
- **绑定组共享投影器实现高效推理**：共享相同激活的层（Q/K/V、gate/up）形成一个绑定组并共享一个投影器，训练后合并为低秩因子；注意力中的 K/V 共享输入投影允许缓存单个窄潜在变量，将 KV 缓存缩减为 untied 方法的约 1/2。
- **蒸馏与任务损失两种训练目标**：默认使用输出蒸馏损失（KL 散度到稠密模型输出分布），可选使用原始训练损失（LSP^T），前者对较大模型和高压缩率更优，后者在小模型和低压缩率下收敛更快。

## 方法详解
- **投影器参数化**：对线性层 y = Wx + b，学习 U ∈ R^{d_in × k}（列正交），应用正交投影器 P = I - UU^⊤，得到 W P = W - (WU)U^⊤，秩最多为 d_in - k。通过优化无约束 V 并设 U = qf(V)（QR 分解的正交因子）来保证正交性。
- **激活空间计算**：计算 W(x - U(U^⊤x)) 而不显式形成 dense W P 矩阵，避免梯度图中每层存储 O(d_out × d_in) 的矩阵。
- **绑定组**：Q/K/V 读取相同的归一化激活，gate/up 也共享激活；每个绑定组在该公共输入上共享一个投影器。K/V 绑定于输入侧时，可缓存单个潜在 z = Ax 而非完整的 keys 和 values。
- **白化初始化**：基于校准输入的 Gram 矩阵 G = (1/n)X^⊤X 进行白化，对候选权重 Z 最小化重建误差 E(Z) = ||(W-Z)S||_F²，其中 S 是 G 的 Cholesky 因子。保留 W S = U^w Σ^w (V^w)^⊤ 的前 r 个奇异方向，移除尾部 k = d-r 个方向。输入侧使用正交投影器 onto span(SV_r^w) 而非 oblique 投影。
- **测量-KL 排名分配**：对每个单元 u，单独移除其后 k 个方向，测量模型输出 KL 散度 Δ_u(k)，按最小边际成本（Δ_u(k') - Δ_u(k)）/(s_u(k') - s_u(k)) 贪婪分配压缩预算，其中 s_u(k) 是保存的参数数量。
- **训练**：冻结所有预训练权重，联合优化所有 {V_l}，目标函数为 L_total = L_obj + λ_ort L_ort，其中 L_obj 可以是蒸馏 KL 或任务交叉熵损失；使用线性 warmup cosine 学习率调度，早停于验证集。
- **合并**：训练后，投影器 U_⊥ U_⊥^⊤ 与权重合并：W P = (W U_⊥) U_⊥^⊤ = B A，分解为两个薄矩阵，每层仅需存储 r×(d_in + d_out) 个参数。

## 实验与结果
- **数据集**：LLM 使用 WikiText-2（困惑度）、六项零样本基准（ARC-e、PIQA、OpenBookQA、WinoGrande、HellaSwag、MathQA）；ViT 使用 CIFAR-100（源准确率）和 Pets/Aircraft/Places365（线性探针迁移）。
- **模型**：OPT-125M/1.3B、Qwen3-4B、Llama-2-7B、ViT-B/16。
- **压缩率**：-30%、-50%、-70%（移除线性层参数的比例）。
- **主要结果**：
  - **LLM 困惑度**：在全部 12 个模型-压缩率设置中，至少一个 LSP 变体取得最低 WikiText-2 困惑度；在 -70% 压缩下，Llama-2-7B 达到 10.9 vs 最强基线 SVD-LLM 的 13.3，NoLSP（无学习）为 222.8。
  - **零样本准确率**：在 Llama-2-7B 上，-70% 时 LSP 平均准确率为 42.2% vs SVD-LLM 的 36.0%（提升 6.2 点）；在 Qwen3-4B 上，-60% 时 43.8% vs 35.6%（提升 8.2 点）。
  - **ViT 迁移**：在校准分布偏移下，LSP 是性能下降最少的方���；多样化校准数据使 LSP 在 -70% 时的迁移准确率提升 6.6 点（从 33.0 到 39.6）。
  - **推理效率**：Llama-2-7B 在 -70% 压缩下解码速度比稠密模型快 1.56×；使用潜在缓存时，权重+KV 缓存内存减少 13.5×（vs 稠密），上下文容量达 322k tokens（vs 稠密 19.7k）。
- **基线对比**：ASVD、SliceGPT、SVD-LLM (W)、Swift-SVD、LLM-Surgeon、Dobi-SVD、SVD-LLM + LoRA recovery。

## 相关工作脉络
- **基于激活的低秩压缩**：ASVD（Yuan et al., 2024）、SliceGPT（Ashkboos et al., 2024）等从激活能量或层重建误差闭式选择子空间，仅感知单层而非网络输出，误差随深度累积。
- **SVD-LLM 与 Swift-SVD**：SVD-LLM（Wang et al., 2025d）和白化 SVD 最小化层 wise 重建误差；Swift-SVD（Qi et al., 2026）从输出协方差达到每层最优并添加排名分配。两者均为无训练的闭式方法。
- **损失感知压缩**：LLM-Surgeon（van der Ouderaa et al., 2024）通过 Kronecker-factored 曲率代理使用损失；Dobi-SVD（Wang et al., 2025b）直接优化损失但仅学习截断排名。这些方法仍依赖局部代理或仅优化排名。
- **结构化剪枝**：SparseGPT（Frantar & Alistarh, 2023）、Magnitude pruning（Han et al., 2015）移除个体权重而非连续子空间，LSP 将"学习移除什么"的思想扩展到低秩压缩领域。
- **排名分配与共享基**：Basis sharing（Wang et al., 2025a）和跨层共享因子；LSP 采用相同两杠杆：绑定组内共享输入投影器、按输出 KL 测量分配排名，但基始终保持可学习而非闭式。

## 局限性与未来方向
- **测量-KL 分配是局部代理**：孤立单元成本忽略了交互作用，当前策略无全局最优性保证；在某些设置中均匀排名可能更优（论文附录 C 显示 Llama-2-7B -50% 时测量-KL 略低于均匀分配）。
- **压缩是一次性成本**：LSP 需要 KL 测量和投影训练（虽然测量可跨压缩率和目标复用），对于频繁压缩场景不够经济。
- **绑定组仅在某些配置下带来效率提升**：Qwen3-4B 因 grouped-query attention 无法绑定 K/V 于输入侧，KV 缓存节省有限，需要在精度和效率间权衡。
- **校准数据依赖**：蒸馏方法需要校准数据生成教师输出；任务损失方法需要标签。虽然蒸馏只需无标签数据，但校准集的选择影响最终性能。

## 研究启发与可借鉴点
- **投影器参数化作为子空间学习的通用框架**：通过 U = qf(V) 保持正交性、在激活空间计算避免显式矩阵、方向 dropout 和热身策略等技术可迁移到其他子空间学习任务（如特征选择、表示学习）。
- **测量-KL 分配的贪婪策略**：通过单独截断每个单元测量其对输出的影响、按边际成本分配预算的思路，可用于其他需要跨层分配资源的场景（如带宽分配、层选择性卸载）。
- **绑定组与潜在缓存**：共享激活的层组绑定投影器并缓存单一潜在变量，这一设计可直接应用于 MLA（Multi-head Latent Attention）等现代注意力变体的压缩，减少 KV 缓存开销。
- **蒸馏 vs 任务损失的适配策略**：蒸馏对大模型和高压缩率更优、任务损失收敛更快且可在小模型上超越稠密模型性能，这一发现为不同场景下的目标选择提供了实用指南。
- **白化初始化 + 正交重述**：将 oblique 投影转换为正交投影的技术（Proposition 1）保证了输入侧的数值稳定性，这一思路可推广到其他需要正交约束的初始化方案中。

## 关键术语表
- **Learnable Subspace Projections (LSP)**：论文提出的方法，通过端到端学习正交投影器来选择和移除每个线性层的子空间，冻结预训练权重并优化全局目标。
- **Whitened SVD truncation**：对输入激活 Gram 矩阵白化后的 SVD 截断，使得截断方向按对层输出的影响排序，而非原始权重谱能量。
- **Measured-KL rank allocation**：通过单独应用每个候选截断并测量其对完整网络输出的 KL 散度，按每参数边际成本贪婪分配压缩预算的方法。
- **Tied group**：共享相同激活输入的层组（如 Q/K/V 或 gate/up），在 LSP 中绑定为一个单元共享同一个投影器。
- **Latent cache**：在注意力层中缓存共享的低维潜在 z = Ax 而非完整的 keys 和 values，大幅减少 KV 缓存内存占用。
- **NoLSP**：LSP 的未训练控制版本，使用相同的白化初始化和排名分配但不进行端到端优化，用于隔离学习效应。
- **Distillation loss (LSP)**：以稠密模型输出分布为教师的 KL 散度损失，无需标签即可用于校准。
- **Task loss (LSP^T)**：使用模型原始训练损失（next-token 或分类交叉熵）作为目标，可使模型专门化于校准数据分布。

## 可复现要素
- **数据集**：WikiText-2（LLM 校准与评测）、CIFAR-100、Food-101、CIFAR-10、EuroSAT、STL-10、DTD（ViT 校准）；六项零样本基准（ARC-e、PIQA、OpenBookQA、WinoGrande、HellaSwag、MathQA）。部分数据集公开可用。
- **代码/权重**：论文未明确声明开源，但提到"所有基线均运行其官方代码"；压缩后的权重可通过合并投影器得到。
- **关键超参**：λ_ort = 0.05（正交性惩罚权重）、dropout rate p = 0.05（Llama-2-7B 为 0.1）、warm-up 跨越首个 epoch、factorization tolerance ε_SVD = 0.05、Adam 无 weight decay、bf16 权重加载、压缩计算在 fp32（Qwen3-4B/Llama-2-7B）或 fp64（OPT/ViT）下进行。
