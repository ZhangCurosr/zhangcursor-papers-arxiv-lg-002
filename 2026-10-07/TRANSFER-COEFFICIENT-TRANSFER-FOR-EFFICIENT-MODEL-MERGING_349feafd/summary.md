---
title: "TRANSFER-COEFFICIENT-TRANSFER-FOR-EFFICIENT-MODEL-MERGING"
source: https://arxiv.org/pdf/2610.07819v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 17:44:01"
field: "模型合并与多任务学习"
keywords: ["model merging", "coefficient search", "proxy model transfer", "task vector", "large language model", "vision transformer", "parameter-efficient"]
innovations: ["提出αTransfer：在小型代理模型上搜索合并系数后直接迁移到同家族大目标模型，实现搜索成本与目标模型规模解耦", "统一建模合并方法为编辑函数+系数粒度的框架，覆盖全局/任务级/层级三种粒度", "提出层级系数的跨深度复制与插值映射策略，及Hybrid Search精度恢复机制"]
benchmarks: ["CLIP-ViT (ViT-B/32, ViT-B/16 → ViT-L/14)", "SigLIP (Base → Large)", "Qwen3 (0.6B → 1.7B / 4B)", "DTD, EuroSAT, FER2013, Food101, GTSRB, RESISC45, Stanford Cars, SUN397", "Banking77, DDXPlus, Usefulness Judge, IFEval"]
---

# 论文速读：αTRANSFER: COEFFICIENT TRANSFER FOR EFFICIENT MODEL MERGING

## 一句话总结
论文提出 αTransfer，通过在小型代理模型上搜索最优合并系数后直接迁移到同家族的大目标模型，避免在大模型上昂贵的高维系数搜索，在视觉 Transformer 上实现 6× 加速与 70% 显存降低，在大型语言模型（Qwen3-4B）上实现 20× 加速与 85% 显存降低，同时保持与直接搜索可比拟的性能。

## 研究问题与动机
- **模型合并的系数搜索是主要扩展瓶颈**：合并多个微调 checkpoint 需要寻找最优缩放系数，候选系数集合的评估次数随任务数指数增长，全量搜索在大规模场景下不可行。
- **现有两类方法各有局限**：基于评估的搜索（网格搜索、贝叶斯优化等）每次评估仍需加载完整大模型，计算代价高；基于梯度的优化（AdaMerging、DivMerge 等）需同时持有多个微调模型并在反向传播中更新系数，GPU 显存随任务数和模型规模双重缩放，易 OOM。
- **缺乏跨尺度的低成本替代范式**：现有方法均在目标模型本身上进行搜索，未利用"小模型成本远低于大模型"这一事实。
- **直觉与假设**：最优合并系数编码任务间相关性与相对重要性，主要由任务特性与模型家族决定，而非模型规模本身，因此系数应在同一家族内具有跨尺度可迁移性。

## 核心贡献（创新点）
1. **提出 αTransfer 范式**：在小型代理模型上搜索合并系数，直接迁移到同家族大目标模型，将搜索成本从目标模型规模解耦。
2. **统一建模合并方法与系数粒度**：以编辑函数 $f(\cdot)$ + 系数 $\alpha$ 的统一形式刻画 Task Arithmetic、TIES、DARE、AdaMerging、AdaMerging++、DivMerge 等 6 种主流方法，并覆盖全局/任务级/层级的系数粒度。
3. **提出层级系数的跨深度映射策略**：针对代理与目标模型层数不同的情况，设计基于复制（Copy）与线性插值（Interpolation）的相对深度映射，使层粒度系数可迁移。
4. **提出 Hybrid Search 扩展**：将转移系数作为初始点，保留少量目标端预算进行局部精化，在几乎不牺牲加速收益的前提下追平甚至超越完全直接搜索。
5. **系统性实证验证跨尺度系数可迁移性**：通过性能地形可视化与 Spearman $\rho$ 相关性分析（最高 0.94），以及 3 个模型家族（CLIP-ViT、SigLIP、Qwen3）、6 种方法、多对代理-目标组合的全面实验，确立 αTransfer 的通用性与鲁棒性。

## 方法详解
- **统一合并形式**：给定预训练参数 $\theta_0$ 和 $T$ 个微调模型 $\{\theta_i\}$，定义任务向量 $\tau_i = \theta_i - \theta_0$。合并操作表示为 $\theta_{\text{merged}} = \mathcal{M}(\theta_0, \{\tau_i\}; \alpha)$，其中 $\mathcal{M}$ 包含编辑函数 $f(\cdot)$（如 TIES 的 Trim+符号选择、DARE 的随机掩码+重缩放）和系数加权两个阶段。
- **全局/任务级系数转移**：在代理模型上求解 $\alpha^* = \arg\min_\alpha \mathcal{L}^p(\theta_0^p + \sum_i \alpha_i \tilde{\tau}_i^p)$，再将 $\alpha^*$ 直接用于目标模型：$\theta_{\text{merged}}^t = \theta_0^t + \sum_i \alpha_i^* \tilde{\tau}_i^t$，其中编辑函数 $f(\cdot)$ 仍针对目标模型任务向量本地计算。系数形状仅取决于任务数 $T$，与模型规模无关，故可直接复用。
- **层级系数转移（两种策略）**：
  - **Copy-based**：目标模型第 $b$ 块的对应代理块为 $b_{\text{proxy}} = \lceil \frac{b}{B_t} \cdot B_p \rceil$，直接复制该块同类型层的系数。
  - **Interpolation-based**：根据目标块的连续相对深度，在两个最近邻代理块之间线性插值得到更平滑的系数。
- **Hybrid Search**：将 $\alpha^*$ 作为初始点，在目标模型上分配少量预算（梯度方法均分步数、评估方法局部网格搜索）进行精化，平衡效率与精度。

## 实验与结果
- **数据集与模型**：视觉 8 任务（DTD、EuroSAT、FER2013、Food101、GTSRB、RESISC45、Stanford Cars、SUN397）；语言 4 任务（Usefulness Judge、IFEval、Banking77、DDXPlus）。模型家族：CLIP-ViT（ViT-B/32、ViT-B/16 → ViT-L/14）、SigLIP（Base → Large）、Qwen3（0.6B → 1.7B / 4B）。单卡 RTX 4000 SFF Ada（Qwen3-4B 用双卡）。
- **系数相关性验证**（Table 2）：Spearman $\rho$ 在 ViT 最高 0.84，SigLIP 为 0.79，Qwen3-0.6B→1.7B 高达 0.94，0.6B→4B 为 0.90，证实跨尺度分布高度一致。
- **视觉模型**（Table 3）：αTransfer 在多数方法上逼近直接搜索，精度损失常 <1%（如 SigLIP TIES 仅降 0.11%、DARE 降 0.32%）；最坏情况（AdaMerging++ ViT-B/32→ViT-L/14，-7.02%）仍超 Simple Average 7.12%。搜索加速最高 6.07×（Task Arithmetic），显存降低最高 70.4%。
- **大语言模型**（Table 4）：Qwen3-4B 场景下 Task Arithmetic 零精度损失，TIES-Pairwise 同样零损失；搜索加速达 19.97×，显存从 40.22GB 降至 5.96GB（降低 85.2%）。
- **约束预算分析**（Figure 3 / Appendix C）：在各计算预算水平下，αTransfer 始终显著左移性能曲线，以更少评估/时间达到同等或更高精度；对 AdaMerging++ 高稀疏性（k=20）场景存在天花板差异，降低稀疏度（k≥60）后 αTransfer 全面占优（Appendix C.1）。
- **Hybrid Search**（Table 6 / Table 8）：在目标端小幅投入即可追平甚至超越 Origin，且仍保持 1.20×–1.86×（视觉）和 2.29×–4.35×（LLM）的加速。

## 相关工作脉络
1. **Task Arithmetic / TIES / DARE**（Ilharco et al., 2023; Yadav et al., 2023; Yu et al., 2024）：基于任务向量的评估型合并方法，使用全局网格搜索系数；αTransfer 在同一框架下将搜索成本从目标模型转移到代理模型。
2. **AdaMerging / AdaMerging++ / DivMerge**（Yang et al., 2024; Touayouch et al., 2026）：基于梯度的自适应合并方法，需同时持有多个模型并反向传播；αTransfer 将这些方法的优化过程移至小模型，避免大模型显存爆炸。
3. **进化搜索 / 贝叶斯优化**（Akiba et al., 2025; Lee et al., 2025; Liu et al., 2025）：减少评估次数的启发式策略，但每次评估仍昂贵；αTransfer 从根本上规避了目标端评估，与之正交且可组合。
4. **µTransfer / DoReMi / 跨尺度数据归因**（Yang et al., 2021; Xie et al., 2023; Khaddaj et al., 2025）：在小模型上优化超参/数据权重并迁移到大模型；本文首次将该"小代理→大目标"范式引入模型合并系数搜索领域。
5. **Layer-depth 表征一致性**（Wolfram & Schein, 2025）：相近深度层产生相似激活；本文为此提供理论基础，支撑层级系数跨深度插值策略的设计。

## 局限性与未来方向
- **跨家族/跨架构迁移未验证**：当前验证仅限同一家族内不同尺度，不同架构（如 ViT → LLM）间的系数迁移性有待探索。
- **极高稀疏性场景下的精度差距**：AdaMerging++ 在 k=20 时直接搜索的大模型天花板高于代理优化，提示极端稀疏下结构差异可能削弱可迁移性。
- **层粒度迁移依赖相对深度对齐假设**：当代理与目标模型的 patch size 或 block 结构差异较大时（如 ViT-B/32 → ViT-L/14，$\rho=0.58$），迁移效果下降。
- **未来方向**：探索其他合并超参数（如稀疏比例、编辑函数选择）的跨尺度迁移；研究跨配方/跨架构的系数映射；将 Hybrid Search 与更精细的目标端优化策略结合。

## 研究启发与可借鉴点
1. **"小代理搜索→大目标应用"的范式具有普适潜力**：不仅限于合并系数，可迁移至超参优化、数据混合权重确定、正则化强度选择等场景，尤其适用于评估/优化成本随模型规模非线性增长的场合。
2. **统一建模价值显著**：以编辑函数+系数粒度的框架归纳 disparate 方法，便于系统性比较与新方法设计，是文献综述与后续工作的良好起点。
3. **Spearman 相关性作为可迁移性量化指标**：用系数组合排名相关性而非绝对精度差来论证迁移可行性，避免了最优系数必须精确匹配的要求，更具工程实用性。
4. **Hybrid Search 作为即插即用的精度恢复模块**：即使追求极致效率的纯转移场景已具价值，保留微量目标端预算进行精化可无损追回精度，为资源灵活配置提供折中方案。
5. **稀疏度-可迁移性的权衡分析**：Appendix C.1 中对 AdaMerging++ 不同 k 值的消融揭示了方法内部超参与迁移效果的耦合关系，为后续方法改进提供明确方向。

## 关键术语表
- **Task Vector（任务向量）**：微调模型与预训练模型之间的参数差 $\tau_i = \theta_i - \theta_0$，用于捕获特定任务知识。
- **Model Merging（模型合并）**：在权重空间直接组合多个微调模型，无需访问原始训练数据即可形成多任务统一模型。
- **αTransfer**：本文提出的方法，在小型代理模型上搜索合并系数后迁移至同家族大目标模型。
- **Evaluation-based Search（基于评估的搜索）**：通过反复评估候选合并模型性能来搜索最优系数，代表方法有网格搜索、进化搜索、贝叶斯优化。
- **Gradient-based Optimization（基于梯度的优化）**：直接对合并系数计算梯度并优化，代表方法有 AdaMerging、DivMerge，需同时持有多个模型。
- **Global / Task-wise / Layer-wise Granularity（全局/任务级/层级粒度）**：系数 $\alpha$ 的作用范围，分别对应单一标量、每任务一个标量、每任务每层一个标量。
- **TIES / DARE**：TIES 通过 Trim+符号选举解决任务向量符号冲突；DARE 通过随机掩码+重缩放降低合并噪声。
- **Spearman Rank Correlation（Spearman 等级相关）**：衡量代理与目标模型在不同系数组合下排名一致性的指标，本文用于量化跨尺度可迁移性。

## 可复现要素
- **数据集**：DTD、EuroSAT、FER2013、Food101、GTSRB、RESISC45、Stanford Cars、SUN397（视觉，公开）；Banking77、DDXPlus、Usefulness Judge、IFEval（语言，公开或部分公开）；指令微调使用 Nemotron-SFT 子采样（论文提及来源链接）。
- **代码/权重**：论文未明确声明代码开源状态。
- **关键超参**：AdamW（β₁=0.9, β₂=0.999），线性/Cosine LR Scheduler；视觉微调 LR ∈ {1e−5, 3e−5}，Batch ∈ {32, 48}，Epochs ∈ {5, 8, 10, 20}；LLM 微调 LR=4e−5, Batch=36。合并方法超参：TIES/DARE top-k=20%（k=20），DARE drop rate=0.5；AdaMerging 优化 500 步、lr=1e−3、样本数 1000、init=0.5；DivMerge 优化 100 步、lr=1e−2、样本数 200、JS divergence；Task Arithmetic 网格搜索 16 点 (0,1]；TIES/DARE 网格 18 点 (0, 1.8]。
- **硬件**：单卡 NVIDIA RTX 4000 SFF Ada（Qwen3-4B 用双卡）。
