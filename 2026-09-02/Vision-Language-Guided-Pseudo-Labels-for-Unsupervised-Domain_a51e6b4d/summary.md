---
title: "Vision-Language-Guided-Pseudo-Labels-for-Unsupervised-Domain"
source: https://arxiv.org/pdf/2609.00898v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:28:55"
field: "无监督域适应语义分割"
keywords: ["Unsupervised Domain Adaptation", "Pseudo-labeling", "Vision-Language Models", "Semantic Segmentation", "SAM", "EVA-CLIP", "Industrial Waste Sorting"]
innovations: ["跨模态伪标签流水线：SAM 区域提议 + EVA-CLIP 语义分配 + 可选 BLIP 验证，绕过分割模型自身置信度", "发现伪标签质量比数量更重要，Full FT 比 LoRA 显著提升下游自训练效果", "BLIP 语言 grounding 作为选择性验证器，在工业场景提供最大增益 (+133.8% 相对提升)"]
benchmarks: ["GTA5-to-Cityscapes", "LabWaste-to-RealWaste"]
---

# 论文速读：Vision-Language-Guided-Pseudo-Labels-for-Unsupervised-Domain-Adaptation-in-Semantic-Segmentation-for-Waste-Sorting

## 一句话总结
本文提出了一种跨模态伪标签流水线，利用基础模型（SAM 生成区域提议、EVA-CLIP 分配语义标签、可选 BLIP 验证）为无监督域适应语义分割生成可靠伪标签，无需目标域标注；在工业垃圾分拣（LabWaste→RealWaste）基准上，Full FT + BLIP 相比源域基线实现 **+133.8%** 相对 mIoU 提升。

## 研究问题与动机
- **密集像素标注成本高**：工业场景中（如从实验室到工厂的垃圾分拣），目标域图像外观频繁变化，持续收集 dense 标注不现实。
- **现有 UDA 方法的循环依赖问题**：传统自训练方法依赖分割模型自身置信度生成伪标签，但在强域偏移下模型预测不可靠，导致确认偏差（confirmation bias）传播。
- **细粒度类别与复杂背景的挑战**：工业垃圾场景中存在大量背景像素（传送带纹理、污渍、阴影）、部分遮挡、视觉相似的细粒度类别（如 cup vs tray），使传统伪标签质量难以保证。
- **部署实用性需求**：实际工业部署需要端到端处理无标签目标流、生成可检查的伪标签库，且不能依赖重型、骨干特定的训练策略。

## 核心贡献（创新点）
1. **跨模态伪标签流水线**：将 SAM（类无关区域提议）与 EVA-CLIP（区域-文本相似度语义分配）链式结合，完全绕过分割模型自身置信度，为无监督域适应提供外部监督信号。
   - *区别*：不同于 CBST/DAFormer/PLSR 等依赖分割模型预测的方法，本文使用与分割器无关的基础模型作为伪标签来源。

2. **伪标签质量优于数量的实证发现**：系统比较 LoRA 适配与全量微调（Full FT）对伪标签质量的差异，发现 Full FT 在 GTA5→Cityscapes 上带来 +31.5% 相对提升，在 LabWaste→RealWaste 上带来 +133.8% 相对提升，而 LoRA 在 Cityscapes 上几乎无增益。
   - *区别*：揭示了在强域偏移下，Verifier 的误差结构化特征（边界泄漏、系统类混淆）比平均准确率更能决定自训练效果。

3. **可选 BLIP 语言 grounding 验证模块**：对低置信度区域使用 BLIP-2 生成句子级描述，通过 EVA-CLIP 文本编码器重新分类，选择性恢复模糊区域的伪标签。
   - *区别*：BLIP 作为选择性验证器而非通用重标注器，仅触发低置信度裁剪，避免额外计算开销。

4. **部署导向的骨干无关设计**：使用标准 DeepLabV3-ResNet50 作为分割骨干，验证基础模型链的即插即用能力，伪标签库可与更强架构（如 SegFormer、ViT-based）无缝集成。
   - *区别*：目标不是与重度工程化的 UDA 方法竞争 SOTA，而是展示工业场景下可部署的自动标注路径。

## 方法详解
流水线分为四个阶段：

**Stage I: 源域基线分割**
- 在标注源域 $D_S$ 上训练 DeepLabV3（ResNet-50 骨干），损失为交叉熵 + Dice Loss：
  $$\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{Dice}$$
- 产出 source-only checkpoint，作为后续自训练的初始化。

**Stage II: 目标域掩码提议与区域提取**
- 使用冻结的 SAM（ViT-H 变体）自动掩码生成模式，对无标签目标图像 $x_j^T$ 生成候选掩码：
  $$\{M_{j,k}\}_{k=1}^{K} = \text{SAM-Auto}(x_j^T), \quad M_{j,k} \in \{0,1\}^{H \times W}$$
- 后处理：面积过滤 + 非极大值抑制（NMS）去除冗余/小掩码，提取带上下文边界的区域 patch $x_k^T$。

**Stage III: 跨模态伪标签生成与验证**
- **EVA-CLIP 语义分配**：将区域 patch 编码为视觉嵌入 $v_k \in \mathbb{R}^{1024}$，与类别文本嵌入 $u_c$ 计算余弦相似度：
  $$c_k = \arg\max_c \cos(v_k, u_c), \quad s_k = \max_c \cos(v_k, u_c)$$
  文本提示格式：`"a photo of {class name}"`。
- **置信度过滤与伪标签库构建**：接受满足 $s_k \geq \tau$ 的区域，融合为像素级伪标签图 $\hat{y}^T$，重叠区域取最高 $s_k$ 的标签：
  $$\hat{y}_p^T = \arg\max_{k: p \in M_k, k \in \mathcal{R}_{accept}} s_k$$
- **可选 BLIP 验证**：对低置信度区域（$s_k < \tau$），使用 BLIP-2 生成描述 $g_k$，再经 EVA-CLIP 文本编码器比对类别提示，满足以下任一条件则恢复：
  $$s_k^{\text{BLIP}} \geq \tau_{\text{BLIP}} \quad \text{or} \quad (s_k^{\text{BLIP}} \geq 0.7\tau \ \text{and} \ c_k^{\text{BLIP}} = c_k)$$

**Stage IV: 伪标签库与下游自训练**
- 联合训练损失：
  $$\mathcal{L}(\theta) = \mathbb{E}_{(x^S, y^S) \sim D_S}[\ell(f_\theta(x^S), y^S)] + \lambda \mathbb{E}_{(x^T, \hat{y}^T) \sim \hat{D}_T}[\ell(f_\theta(x^T), \hat{y}^T)]$$
- 目标损失仅对伪标签库中有标签的像素计算。

## 实验与结果
**数据集**
- **GTA5→Cityscapes**：合成到真实驾驶场景，19 类 urban scene，源域 24,966 张，目标域 2,975 张（无标签训练）+ 500 张验证。
- **LabWaste→RealWaste**（主实验）：实验室到工厂垃圾分拣，8 个前景类 + 背景，源域 ~2,000 张，目标域 457 条传送带图像（运动模糊、反光、污渍、遮挡）。

**评估指标**
- 主指标：mIoU（所有评估类别）
- 辅助：每类 IoU、伪标签覆盖率、像素准确率、高置信度错误频率

**主要结果（Table 2）**
| 方法 | GTA5→Cityscapes mIoU | 相对提升 | LabWaste→RealWaste mIoU | 相对提升 |
|------|---------------------|---------|------------------------|---------|
| Source-only | 20.0 | - | 7.7 | - |
| LoRA | 20.0 | +0% | 9.3 | +21% |
| Full FT | 25.0 | +25% | 14.1 | +83% |
| Full FT + BLIP | **26.3** | +31.5% | **18.0** | **+133.8%** |
| AdaBN + Full FT | 36.0 | +80% | 30.2 | +292.2% |
| DAFormer [10]（文献） | 68.3 | - | - | - |
| PLSR [32]（文献） | 71.3 | - | - | - |

**关键结论**
1. **伪标签质量决定性作用**：Full FT 比 LoRA 产生更高质量的伪标签库（GTA5 上 mIoU 18.5% → 29.5%），下游自训练增益显著更大。
2. **BLIP 验证在工业场景价值更高**：LabWaste→RealWaste 上 BLIP 贡献 +3.9 mIoU（14.1%→18.0%），而 Cityscapes 仅 +1.3 mIoU（25.0%→26.3%），因工业场景视觉歧义更严重。
3. **AdaBN 控制实验**：AdaBN 单独可将 LabWaste mIoU 从 7.7% 提升至 27.4%，说明源-目标表征失配是主要差距之一；但高质量伪标签仍提供额外 +2.8 mIoU 增益。
4. **覆盖度与正确性权衡**：LoRA 在 LabWaste 上覆盖率 87-92%，但正确性仅 4.6-5.0% mIoU；Full FT + BLIP 覆盖率 96.3%，正确性 21.3%。

## 相关工作脉络
1. **CBST [35]**：类平衡自训练，通过置信度阈值和类别平衡缓解确认偏差；本文方法不使用分割模型自身预测，而是依赖外部 VLM。
2. **DAFormer [10]**：Transformer 架构 + 精密伪标签过滤的 SOTA UDA 方法；本文目标不是超越其绝对性能，而是展示基础模型链在轻量骨干上的可部署性。
3. **PLSR [32]**：伪标签自 refinement，使用辅助 refinement 网络处理标签噪声；本文通过跨模态基础模型链直接生成高质量伪标签，无需额外 refinement 网络。
4. **PADCLIP [14]**：利用 CLIP 零样本预测进行伪标签并引入自适应去偏；本文将 CLIP 家族模型作为外部语义验证器，不将其适配为最终任务模型。
5. **Open-Vocabulary Segmentation（如 CAT-Seg [2], OVSS 综述 [33]）**：结合掩码生成器与 VLM 的分类器进行开放词汇分割；本文采用相同两阶段结构（SAM + EVA-CLIP），但聚焦于封闭集 UDA 场景，将 VLM 预测作为伪标签教师而非直接推理。
6. **AdaBN [19]**：仅更新 BatchNorm 统计量的轻量域适应；本文控制实验表明 AdaBN 可解决大部分表征失配，但语义监督仍需伪标签补充。

## 局限性与未来方向
**局限性**
1. **BLIP 作为选择性验证器而非通用重标注器**：仅在低置信度区域触发，计算开销存在，且收益依赖严格的触发条件。
2. **闭集假设**：训练时预设固定标签集，开放集样本被路由到"unknown"桶，无法扩展标签集。
3. **细粒度/小目标表现受限**：薄结构、小碎片、部分遮挡对象的伪标签质量最脆弱，剩余错误主要由漏检和欠分割构成。
4. **与 SOTA 比较需谨慎**：使用轻量 DeepLabV3-R50，未与 DAFormer/PLSR 等重型方法公平对比，绝对性能有提升空间。

**未来方向**
1. 系统评估更强分割骨干（SegFormer、ViT-based）以解耦骨干能力与跨模态伪标签的贡献。
2. 结合内部 refinement 策略（分割器自身预测 + 开放词汇分割骨干），形成混合自适应方法。
3. 扩展到多站点、时间划分的工业评估，细化 UDA 基线与开放词汇分割器的对比。

## 研究启发与可借鉴点
1. **外部监督信号绕过确认偏差**：在自训练 UDA 中，当分割模型自身置信度不可靠时，引入与任务解耦的基础模型（如 SAM + VLM）作为伪标签源，可有效打破循环依赖。可迁移至其他需要伪标签的领域（如目标检测、关键点估计）。
2. **Verifier 适配策略对下游影响巨大**：LoRA vs Full FT 的对比揭示，即使区域级准确率相近，误差的结构化分布（边界泄漏、类混淆）会决定自训练成败。后续工作可设计误差分解诊断工具。
3. **跨模态语言 grounding 作为选择性验证器**：BLIP 仅在低置信度区域触发，通过句子级描述解决视觉歧义，这种"按需调用重型模块"的设计兼顾质量与效率，适用于计算受限部署。
4. **工业场景的覆盖度陷阱**：LabWaste 数据集中背景像素主导，高覆盖率不等于高质量；评估时需同时报告覆盖度与正确性 mIoU，避免虚假繁荣。
5. **模块化流水线设计**：SAM/EVA-CLIP/BLIP 可替换为更强模型，分割骨干可插拔，这种设计哲学适合工程落地，便于不同算力预算下的灵活部署。

## 关键术语表
**Unsupervised Domain Adaptation (UDA)**：无监督域适应，利用带标签源域和无标签目标域训练模型，使其在目标域上泛化，无需目标域标注。

**Pseudo-labeling**：伪标签，将模型在 unlabeled 数据上的预测作为"软标签"用于后续训练，是自训练的核心机制。

**Confirmation Bias**：确认偏差，自训练中早期错误伪标签被反复强化，导致模型性能退化。

**Segment Anything Model (SAM)**：Meta 提出的基础分割模型，可通过 prompt（点/框）或自动模式生成高质量类无关掩码。

**EVA-CLIP**：改进版 CLIP 模型，通过 scaling 和优化策略提升零样本分类准确率（ImageNet 零样本 82%）。

**BLIP-2**：结合冻结图像编码器与大语言模型的视觉-语言生成模型，可生成图像描述。

**AdaBN**：Adaptive Batch Normalization，仅用目标域数据更新 BN 层统计量，轻量域适应基线。

**mIoU (mean Intersection-over-Union)**：语义分割主指标，所有类别 IoU 的均值，衡量预测掩码与真值的重叠度。

## 可复现要素
- **数据集**：GTA5、Cityscapes（公开）；LabWaste、RealWaste（引用 [24,8]，需向 Fraunhofer 申请）
- **代码/权重**：论文未明确声明开源，但提及"available in our repo"（见 Section 4.1 Metrics），需核查 arxiv 附录或作者主页
- **关键超参**：
  - SAM：ViT-H 变体，自动掩码生成模式
  - EVA-CLIP：ViT-L/336，文本提示 `"a photo of {class name}"`
  - BLIP-2：FLAN-T5 XL
  - 分割骨干：DeepLabV3 + ResNet-50
  - 损失：交叉熵 + Dice，目标域权重 λ（论文未明确数值）
  - 置信度阈值 τ、τ_BLIP（论文未明确数值）
  - LoRA 适配：rank 与 alpha 未提及
