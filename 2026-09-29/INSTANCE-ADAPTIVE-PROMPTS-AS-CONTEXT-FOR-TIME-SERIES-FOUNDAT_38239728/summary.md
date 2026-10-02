---
title: "INSTANCE-ADAPTIVE-PROMPTS-AS-CONTEXT-FOR-TIME-SERIES-FOUNDAT"
source: https://arxiv.org/pdf/2609.34786v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:29:31"
field: "时序预测"
keywords: ["time series forecasting", "foundation models", "prompt tuning", "parameter-efficient adaptation", "context compression"]
innovations: ["将实例自适应提示作为冻结TSFM的紧凑上下文代理", "全局统计与分段时序信息结合的提示生成机制"]
benchmarks: ["GIFT-Eval", "TIME"]
---

# 论文速读：INSTANCE-ADAPTIVE-PROMPTS-AS-CONTEXT-FOR-TIME-SERIES-FOUNDAT

## 一句话总结
本文提出 PaCTS（Prompts as Context for Time Series），通过为冻结的时间序列基础模型（TSFM）生成紧凑的实例自适应潜在提示，作为历史上下文的替代，在几乎不增加推理成本的前提下提升预测性能，且泛化性优于权重空间微调方法。

## 研究问题与动机
- **长上下文成本高**：延长历史上下文可提升 TSFM 性能，但推理 FLOPs 显著增加，且收益递减。
- **上下文长度受限**：许多时间序列本身较短，无法提供额外历史；部分骨干模型的上下文窗口有限，超出后可能被丢弃或有害。
- **现有微调方法局限**：LoRA、全量微调等权重空间方法在分布外（OOD）泛化上表现不佳，且需要针对目标数据集单独适配。
- **核心问题**：能否在不修改冻结 TSFM 参数的情况下，通过输入空间的紧凑提示为模型提供有效的上下文信息？

## 核心贡献（创新点）
1. **将提示作为上下文代理**：提出 PaCTS，通过少量可学习的连续嵌入 token（提示）充当历史上下文的紧凑替代，避免扩展实际序列长度。
2. **共享+自适应提示分解**：将提示分解为跨数据集共享的静态组件和基于输入统计的自适应组件，兼顾通用模式与实例特异性。
3. **全局统计+分段时序细化**：自适应组件由全局不变统计特征生成，并通过交叉注意力引入分段级别时序信息，同时捕获全局特性与局部变化。
4. **高效跨分布泛化**：单一提示模块在异构数据集上联合训练，无需目标数据集微调即可在分布外基准上优于权重空间微调方法。

## 方法详解
**整体框架**：给定冻结 TSFM $f_\theta$，输入序列 $\mathbf{x}$ 经归一化、分块后得到 patch embeddings $\mathbf{E} \in \mathbb{R}^{L \times d}$。PaCTS 生成 $M \ll L$ 个提示 token $\mathbf{P} \in \mathbb{R}^{M \times d}$，前置拼接至 $\mathbf{E}$：

$$\hat{\mathbf{y}} = f_\theta([\mathbf{P}; \mathbf{E}])$$

**提示构成**（公式3）：
$$\mathbf{P}(\mathbf{x}) = \mathbf{P}_s + \sigma(g) \cdot \mathbf{P}_a(\mathbf{x})$$
- $\mathbf{P}_s$：共享静态提示，初始化为训练数据 mean patch embedding。
- $\mathbf{P}_a(\mathbf{x})$：自适应组件，通过 sigmoid 门控 $g$ 调节幅度。

**自适应生成**（公式4）：
从输入提取 $N_s$ 维尺度无关统计量 $\mathbf{s}$（趋势、变异性、自相关、噪声水平等），经 MLP 映射为组合系数 $\mathbf{z}$，再线性组合 $r$ 个 learnable basis patterns：
$$\mathbf{P}_a(\mathbf{x}) = \sum_{i=1}^{r} z_i \cdot \mathbf{T}_i, \quad \mathbf{z} = \text{MLP}(\mathbf{s})$$

**分段细化**（公式5-6）：
将最近历史切分为最多 $S$ 个非重叠 segment，计算每段统计量并添加相对位置编码，得到 $\mathbf{H}(\mathbf{x}) \in \mathbb{R}^{S \times d_a}$。通过 cross-attention 让 $\mathbf{P}_a$ 检索局部信息得到细化 $\mathbf{R}(\mathbf{x})$：
$$\mathbf{P}(\mathbf{x}) = \mathbf{P}_s + \sigma(g) \cdot [\mathbf{P}_a(\mathbf{x}) + \mathbf{R}(\mathbf{x})]$$
Cross-attention 输出投影初始化为零，确保训练初期不影响全局提示。

**训练**：仅优化提示参数，骨干完全冻结。使用 pinball loss（公式7），有效预测位置加权，多数据集联合训练。

## 实验与结果
**数据集**：GIFT-Eval（35数据集，97配置，主基准）、TIME（50数据集，98任务，OOD泛化测试）。

**骨干模型**：Chronos-2（主）、PatchTST-FM、TimesFM-2.5。

**基线**：Zero-shot、Full fine-tuning、LoRA、LayerNorm-only、BitFit、Linear probing。

**主要结果**（Chronos-2，GIFT-Eval，Table 1）：
- **PaCTS (4096)**: MASE = 0.697 (-1.55%), CRPS = 0.480 (-3.23%)
- **优于双倍上下文**：PaCTS(4096) 优于 Zero-shot(8192) 和其他所有微调方法在 8192 下的结果。
- **计算效率**（Table 2）：相比 Zero-shot(8192)，PaCTS(4096) 减少 FLOPs 49.3%、显存 18.4%、延迟 36.8%，仅增加 0.128% 参数（153,286）。

**消融实验**（Table 3）：
- Static prompts: MASE 0.707, CRPS 0.490
- +Adaptive: MASE 0.704, CRPS 0.486
- +Segment refinement: MASE 0.697, CRPS 0.480（最终结果）

**泛化性**（Figure 4）：
- PaCTS 是唯一在 OOD 基准 TIME 上同时降低 MASE 和 CRPS 的方法。
- 所有权重空间微调方法在 TIME 上均劣于 zero-shot。

**跨骨干验证**（Figure 5）：
- PatchTST-FM: PaCTS(4096) vs Zero-shot(8064)，MASE -0.43%，FLOPs 减半。
- TimesFM-2.5: PaCTS(8192) vs Zero-shot(16384)，MASE -0.85%，CRPS -0.60%，FLOPs -50.95%。

## 相关工作脉络
1. **TSFM 扩展上下文策略**：现有方法通过增加历史长度提升性能，但计算代价线性/超线性增长；本文以输入空间提示替代长序列，避免额外计算。
2. **参数高效微调（PEFT）**：LoRA、BitFit 等修改骨干权重；本文冻结骨干，仅在输入空间添加提示，避免分布偏移风险。
3. **Prompt tuning**：NLP/Vision 领域已有大量工作（ Lester et al., Zhou et al.）；本文首次系统研究 prompt 作为 compact context surrogate for TSFMs。
4. **输入空间条件化**：UniCast、CoSPOT 等融合多模态/光谱提示；本文专注纯时序上下文压缩，无额外模态。
5. **Post-training 方法**：TFMAdapter 拟合实例级协变量修正；TS-Memory 蒸馏检索修正；本文通过学习通用 prompt 模块实现零样本复用。
6. **In-context learning**：TimesFM-ICF 使用外部示例进行上下文学习；本文直接学习目标序列的 latent prompts，无需外部数据。

## 局限性与未来方向
- **仅训练于单变量设置**：多变量场景下提示学习未探索，论文中多变量实验为直接迁移。
- **依赖骨干上下文缩放行为**：提示作为上下文替代的效果受限于底层 TSFM 对更长历史的利用率。
- **固定统计特征**：当前使用手工设计的 10 个统计量，可能无法捕获所有预测相关模式。
- **未来方向**：探索多变量 prompt 学习、更通用的上下文代理机制、自适应统计特征选择。

## 研究启发与可借鉴点
1. **Prompt-as-Context 范式**：将 learned tokens 视为压缩上下文而非简单 adaptation signal，为其他模态（如图像、文本）的 frozen FM 提供新思路。
2. **统计特征驱动生成**：用低维不变统计量控制提示生成，兼具参数效率和可解释性，可迁移至其他序列建模任务。
3. **Cross-attention 细化机制**：分段统计 + 相对位置编码 + cross-attention 的组合，可在不增加 prompt 数量的前提下注入时序局部信息。
4. **跨分布泛化评估**：直接在 OOD 基准上验证方法的有效性，避免过拟合训练分布的风险，值得在更多 benchmark 上推广。
5. **零额外任务适配**：单一 prompt 模块跨数据集训练后直接用于未见任务，简化部署流程，适用于资源受限场景。

## 关键术语表
**TSFM (Time-Series Foundation Model)**：在大规模时序数据上预训练的通用预测模型，无需任务特定训练即可泛化。
**Latent Prompt**：连续嵌入向量，前置拼接到输入序列，作为条件信号影响冻结模型的输出。
**Context Surrogate**：用紧凑提示替代长历史上下文，以更低计算成本传递相似预测信息。
**Instance-Adaptive**：提示根据输入序列的统计特征动态生成，适配不同时间序列的特性。
**Pinball Loss**：分位数回归损失函数，用于训练 probabilistic forecasting 模型。
**GIFT-Eval**：通用时序预测评估基准，包含 35 个数据集和 97 个任务配置。
**Out-of-Distribution (OOD) Generalization**：模型在分布外数据（如新数据集）上的泛化能力。

## 可复现要素
- **数据集**：GIFT-Eval（公开）、TIME（公开）；论文已提供数据加载细节。
- **代码**：论文声明将在 https://github.com/zzzx1224/PaCTS 开源代码。
- **权重**：预训练骨干（Chronos-2、PatchTST-FM、TimesFM-2.5）可从官方获取；PaCTS 提示权重待开源。
- **关键超参**：M=10（提示数）、r=4（basis rank）、S=16（最大分段数）、lr=1e-3、batch_size=48、epochs=12。
- **实现细节**：统计特征见 Appendix B Table 5，训练配置见 Appendix C。
