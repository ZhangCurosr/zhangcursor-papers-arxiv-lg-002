---
title: "TRANSFER-COEFFICIENT-TRANSFER-FOR-EFFICIENT-MODEL-MERGING"
source: https://arxiv.org/pdf/2610.07819v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:20:02"
field: "模型合并与高效微调"
keywords: ["Model Merging", "Coefficient Transfer", "Proxy Model", "Task Vector", "Large Language Model", "Vision Transformer", "Hyperparameter Transfer"]
innovations: ["提出αTransfer框架，在小型代理模型上搜索合并系数并直接迁移至大型目标模型", "建立编辑函数与系数分离的统一形式化框架，系统化分析6种主流合并方法", "设计层间系数跨尺度映射策略（Copy-based/Interpolation-based），解决层数不一致问题"]
benchmarks: ["CLIP-ViT (ViT-B/32, ViT-B/16, ViT-L/14)", "SigLIP (Base, Large)", "Qwen3 (0.6B, 1.7B, 4B)", "DTD, EuroSAT, FER2013, Food101, GTSRB, RESISC45, Stanford Cars, SUN397", "Banking77, DDXPlus, Usefulness Judge, IFEval"]
---

# 论文速读：αTRANSFER: COEFFICIENT TRANSFER FOR EFFICIENT MODEL MERGING

## 一句话总结
论文提出 αTransfer，通过在小型代理模型上高效搜索最优合并系数，再直接迁移到同族大型目标模型，避免了在大模型上昂贵且耗时的系数搜索过程；在保持接近直接搜索性能的前提下，实现最高 6× 速度提升与 85% 显存节省。

## 研究问题与动机
1. **模型合并的核心瓶颈是合并系数搜索开销过高**：现有方法需在目标模型上反复评估或优化候选系数，计算量随模型尺寸和任务数量组合爆炸。
2. **基于评估的搜索每轮评估成本高昂**：虽然可通过均匀系数、贝叶斯优化等方式减少评估次数，但单次大模型评估仍极昂贵，总计算预算不可接受。
3. **基于梯度的优化面临内存压力**：AdaMerging、DivMerge 等方法需在优化过程中同时持有多模型，显存随任务数和模型规模共同增长，GPU 显存不足时直接不可行。
4. **缺乏跨尺度迁移性的理论支撑与实证验证**：是否存在"代理模型上找到的最优系数可以直接用于大模型"的先验未知，这是本文验证的核心假设。

## 核心贡献（创新点）
1. **提出 αTransfer 框架**：在小型同族代理模型上完成系数搜索，将最优系数直接迁移至大型目标模型执行合并，首次将代理搜索范式引入模型合并任务。
2. **建立统一形式化框架**：将所有主流合并方法（Task Arithmetic、TIES、DARE、AdaMerging、AdaMerging++、DivMerge）抽象为"编辑函数 + 系数加权"的统一结构，使跨方法分析成为可能。
3. **实证证明性能景观的跨尺度一致性**：在 CLIP-ViT、SigLIP、Qwen3 三个模型族上，代理与目标模型在合并系数空间上的性能排名 Spearman ρ 高达 0.79–0.94，支撑系数可迁移性的核心假设。
4. **设计层间系数跨尺度映射策略**：针对层-wise 合并方法，提出 Copy-based 和 Interpolation-based 两种相对深度映射方式，解决代理与目标模型层数不一致时的系数转移问题。
5. **提出 Hybrid Search 扩展**：将代理搜索得到的系数作为初始化，在小比例目标端预算内进行局部精炼，在几乎保留速度的同时进一步缩小甚至消除精度差距。

## 方法详解
**统一合并形式化**：
- 定义任务向量 $\tau_i = \theta_i - \theta_0$，所有合并方法可统一表示为 $\theta_{\text{merged}} = \mathcal{M}(\theta_0, \{\tau_i\}; \boldsymbol{\alpha})$。
- 合并过程分为两阶段：编辑阶段 $f(\cdot)$ 对任务向量进行修剪/掩码等操作，系数阶段由 $\boldsymbol{\alpha}$ 加权合并。
- 系数粒度分为三种：**Global**（单个标量共享）、**Task-wise**（每任务一标量）、**Layer-wise**（每任务每层一标量）。

**αTransfer 机制**：
- **Global / Task-wise 转移**：在代理模型上求解 $\boldsymbol{\alpha}^* = \arg\min_{\boldsymbol{\alpha}} \mathcal{L}^p(\theta_0^p + \sum_i \alpha_i \tilde{\tau}_i^p)$，再将 $\boldsymbol{\alpha}^*$ 直接应用于目标模型的对应编辑任务向量：$\theta_{\text{merged}}^t = \theta_0^t + \sum_i \alpha_i^* \tilde{\tau}_i^t$。注意仅系数被转移，编辑函数 $f(\cdot)$ 仍在目标模型上原生计算。
- **Layer-wise 转移**：当代理与目标模型层数不同时，采用相对深度映射。对重复模块，按 $b_{\text{proxy}} = \lceil \frac{b}{B_t} \cdot B_p \rceil$ 进行 Copy-based 映射，或通过线性插值获得平滑过渡。

**Hybrid Search**：
- 将代理最优系数作为目标端优化的起点，分配一小部分预算进行局部精炼；梯度类方法将优化步数平分给代理和目标，评估类方法在代理上进行粗搜后再在目标上进行局部搜索。

## 实验与结果
**数据集与模型族**：
- 视觉：CLIP-ViT（ViT-B/32、ViT-B/16、ViT-L/14）、SigLIP（Base、Large），共 8 个图像分类任务（DTD、EuroSAT、FER2013、Food101、GTSRB、RESISC45、Stanford Cars、SUN397）。
- 语言：Qwen3（0.6B、1.7B、4B），共 4 个任务（Usefulness Judge、IFEval、Banking77、DDXPlus）。
- 基线方法：Task Arithmetic、TIES、DARE、AdaMerging、AdaMerging++、DivMerge。

**主要结果（视觉模型）**：
- **ViT-B/32 → ViT-L/14**：Task Arithmetic 精度从 75.98%→75.55%（-0.43%），时间从 11.43h→1.88h（**6.07× 加速**），显存从 2.12GB→0.67GB（**68.7% 节省**）。AdaMerging 显存从 17.84GB→5.27GB（**70.4% 节省**）。
- **SigLIP-Base → SigLIP-Large**：DARE 精度从 79.84%→79.52%（-0.32%），时间从 13.41h→4.00h（3.35× 加速）。

**主要结果（LLM）**：
- **Qwen3-0.6B → 4B**：Task Arithmetic 精度完全持平（76.03%），时间从 1.12h→0.10h（**11.03× 加速**），显存从 40.22GB→5.96GB（**85.2% 节省**）。TIES-Pairwise 精度从 73.69%→73.69%（零损失），加速 **19.97×**。

**Hybrid Search**：
- 在 LLM 上（0.6B→4B），TIES-Pairwise Hybrid 精度恢复至 73.69%，仍获 3.06× 加速。

## 相关工作脉络
1. **Task Arithmetic (Ilharco et al., 2023)**：最早通过任务向量加权平均实现合并，使用均匀全局系数；αTransfer 在其之上自动化搜索最优系数并实现跨尺度迁移。
2. **TIES (Yadav et al., 2023)**：引入修剪（Trim）和符号选举解决冲突；αTransfer 证明其在代理上的最优系数可直接迁移至大模型。
3. **DARE (Yu et al., 2024)**：通过随机掩码压缩任务向量；其全局系数的搜索同样受益于 αTransfer。
4. **AdaMerging / AdaMerging++ (Yang et al., 2024)**：基于预测熵的梯度优化方法；αTransfer 解决了其在大模型上内存需求过高的瓶颈，节省显存最高 70.4%。
5. **DivMerge (Touayouch et al., 2026)**：使用 Jensen-Shannon 散度优化系数；αTransfer 使其在目标端无需完整优化即可接近最优性能。
6. **µTransfer (Yang et al., 2021) / DoReMi (Xie et al., 2023)**：已在训练超参数和数据混合权重中验证小模型到大模型的迁移可行性，本文首次将此范式引入合并系数搜索。

## 局限性与未来方向
1. **代理与目标架构差异过大会降低迁移效果**：ViT-B/32 → ViT-L/14 的相关系数仅为 0.58（远低于 SigLIP 的 0.79 和 Qwen3 的 0.90+），patch size 差异导致优化景观变化较大。
2. **高稀疏设置下梯度方法的迁移性受限**：AdaMerging++ 在 k=20 的高稀疏度下，直接目标搜索能达到更高性能上限，αTransfer 仍有约 7% 的差距。
3. **跨模型族 / 跨配方的系数迁移尚未探索**：当前研究仅限于同模型族，不同模型结构间的系数可迁移性未知。
4. **仅验证了合并系数，其他超参数（如学习率、掩码比例）的迁移性有待研究**。

## 研究启发与可借鉴点
1. **代理搜索范式具有广泛可迁移性**：不仅适用于模型合并，还可推广至其他需要昂贵搜索的超参数优化问题（如数据混合比、正则化系数等）。
2. **统一形式化框架便于方法对比与扩展**：将编辑函数与系数分离的抽象方式，为未来新合并方法的设计与分析提供了清晰的理论工具。
3. **Hybrid Search 提供了灵活的速度-精度权衡**：将代理结果作为初始化，再以小预算精炼，适合对精度要求高但希望减少总成本的场景。
4. **Spearman ρ 作为迁移性度量指标**：论文使用排名相关性而非绝对精度差异验证可迁移性，这一度量方式对性能景观形状一致性更鲁棒，可借鉴至其他代理相关研究中。
5. **层-wise 相对深度映射策略**：Copy-based 与 Interpolation-based 两种方案简单有效，对需要跨层数迁移系数的其他任务（如结构自适应微调）具有参考价值。

## 关键术语表
**Model Merging**：在参数空间中直接将多个微调模型的权重算术组合，以构建多任务统一模型，无需额外训练数据。
**Task Vector**：微调模型参数与预训练模型参数的差值（$\tau_i = \theta_i - \theta_0$），编码了特定任务的增量知识。
**αTransfer**：本文提出的核心方法，在小型代理模型上搜索最优合并系数后直接迁移至大型目标模型。
**Evaluation-based Search**：通过反复评估候选合并模型的性能来搜索最优系数，计算成本随模型规模线性增长。
**Gradient-based Optimization**：通过直接在目标函数上计算梯度来优化合并系数，需同时持有多个模型，显存成本高昂。
**Spearman Rank Correlation (ρ)**：衡量代理与目标模型在不同系数组合下的性能排名相关性，本文用于量化系数迁移的有效性。
**Hybrid Search**：在代理上搜索系数后，以小比例预算在目标模型上进行局部精炼，以平衡效率与精度。

## 可复现要素
- **数据集**：DTD、EuroSAT、FER2013、Food101、GTSRB、RESISC45、Stanford Cars、SUN397（视觉）；Banking77、DDXPlus、Usefulness Judge、IFEval（语言），均为公开数据集。
- **代码/权重**：论文未明确声明代码是否开源；使用的预训练模型为 CLIP-ViT（OpenAI）、SigLIP（Google）、Qwen3，均为公开可用。
- **关键超参**：
  - 视觉模型微调：AdamW，LR ∈ {1e-5, 3e-5}，Batch Size ∈ {32, 48}，Epochs ∈ {5, 8, 10, 20}。
  - LLM 微调：AdamW + Cosine LR，LR=4e-5，Batch Size=36，最多 6 epochs。
  - 系数搜索范围：Task Arithmetic ∈ (0, 1]，DARE/TIES ∈ (0, 1.8]。
  - AdaMerging 优化 500 步，lr=1e-3；DivMerge 优化 100 步，lr=1e-2；TIES/DARE top-k=20%，DARE drop rate=0.5。
- **硬件**：NVIDIA RTX 4000 SFF Ada（单卡），Qwen3-4B 使用双卡。
