---
title: "GroundingPI-A-Grounding-Foundation-Model-towards-Physical-In"
source: https://arxiv.org/pdf/2609.39601v1.pdf
model: agnes-2.5-flash
chunks: 6
summarized_at: "2026-10-03 18:37:19"
field: "多模态视觉 grounding"
keywords: ["grounding foundation model", "visual primitives", "quantized coordinate tokens", "GRPO reinforcement learning", "physical intelligence", "embodied AI"]
innovations: ["量化坐标 token 统一多任务接地接口", "GRPO 组合奖励优化 grounding 精度与格式", "感知原生 foundation model 提升下游数据效率"]
benchmarks: ["Dense200", "RefCOCO", "COCO", "LVIS", "RoboSpatial", "ScreenSpot-Pro", "SROIE", "M6Doc", "nuScenes", "RoboTwin 2.0", "RoboCasa-GR1"]
---

# 论文速读：GroundingPI-A-Grounding-Foundation-Model-towards-Physical-In

## 一句话总结
论文提出 **GroundingPI**，一个 4B 参数、以视觉原语为原生能力的自回归 grounding foundation model；通过量化坐标 token 统一多任务接口，并在多教师数据引擎与 GRPO 强化学习后训练下实现高精度视觉接地，显著改善下游自动驾驶与机器人操作的数据效率与泛化性能。

## 研究问题与动机
- 视觉接地（grounding）是物理智能的核心感知能力，但当前 VLA/WAM 模型使用的通用 VLM/视频生成 backbone 在此方面存在显著不足，感知仍是下游动作学习的瓶颈。
- 通用 VLM 将感知仅作为动作学习的辅助代理目标，无法提供足够精确的空间感知能力支撑精细物理交互。
- 现有 grounding 模型多为专用检测器或大参数量生成模型，缺乏统一、轻量且支持多任务的原生 grounding foundation model。
- 更强的感知基础可能比泛化能力本身更能支撑物理智能，需要构建以视觉原语为原生能力的模型而非依赖通用 VLM。

## 核心贡献（创新点）
- **统一量化坐标 token 接口**：将语义标签、协议标记与 1000 个量化坐标 token（`<0>`–`<999>`）共享同一输出词表，实现 grounding/referring/pointing/OCR/GUI grounding/layout grounding 等多任务统一接口，区别于现有方法需多模块拼接或专用输出头的设计。
- **多教师融合数据引擎**：结合公开数据集与自建数据引擎，通过多教师融合 + 任务特定验证生成高质量标注，利用互补教师与局部观测解决不确定样本的标注歧义，优于单一人工标注或单一模型自标注的管道。
- **GRPO 强化学习后训练**：设计结合覆盖度奖励（$R_{set}$，基于 IoU 匹配）与严格格式奖励（$R_{strict}$，多阈值 F1、定位、格式、数量、顺序惩罚）的组合优势函数 $\tilde{A}_i = 0.7Z(R_{set,i}) + 0.3Z(R_{strict,i})$，仅更新语言参数，显著提升 grounding 精度与输出规范性。
- **感知原生驱动下游任务**：证明精确空间感知接口可有效支持动作学习与泛化（Findings 2.1），且语言条件 grounding 使有限演示数据聚焦控制而非重学感知，带来数据效率提升（Finding 2.2）。

## 方法详解
### 模型架构
- **GroundingPI**：4B 参数自回归 grounding foundation model。
- **视觉编码器**：MoonViT-V2（Kimi K3）。
- **语言解码器**：Qwen3-4B。
- **投影层**：可学习投影层聚合相邻 2×2 视觉特征，经二层 MLP（GELU + 输出归一化）映射至语言嵌入空间。
- **共享词汇表**：语义标签、协议标记、1000 个量化坐标 token（`<0>`–`<999>`）共用输出词表；缺失目标输出 `None`。

### 数据引擎
- 结合公开数据集与自建数据引擎生成标注。
- 多教师融合（multi-teacher fusion）+ 任务特定验证；接受标注训练统一 grounding 专家用于迭代标注，互补教师与局部观测解决不确定样本。

### 训练设计
- **Base VLM Training（Pretrain 1）**：冻结双 backbone，对齐投影层，两阶段联合更新（多模态预训练 + 通用视觉/视频理解），全部使用 causal next-token prediction。
- **SFT（Pretrain 2）**：教师强制下最小化 token-level 交叉熵；先更新全部模块，再冻结视觉编码器/投影层做语言侧微调。
- **RL 后训练（GRPO）**：每组 8 个自回归响应；grounding reward 结合 $R_{set}$（覆盖度，基于 IoU 匹配）和 $R_{strict}$（多阈值 F1、定位、格式、数量、顺序惩罚），组合优势 $\tilde{A}_i = 0.7Z(R_{set,i}) + 0.3Z(R_{strict,i})$；OCR 使用匈牙利匹配评价文本-几何一致性；仅更新语言参数，SFT 参考模型冻结。

### 下游适配方案
- **自动驾驶（nuScenes）**：语言条件轨迹预测，未来 3s 六 waypoints（0.5s 间隔）映射至 `<0>`–`<999>` 词表，open-loop L2 误差为评估指标。
- **机器人操作**：π-style flow-matching policy，GroundingPI 中间特征经投影/重采样注入 Action DiT 的 cross-attention 层；固定 Action DiT、action 数据、训练预算，仅更换 backbone 作受控对比。

## 实验与结果
### 数据集与基线
- 对比 **44 个 baseline**，覆盖 **34 个 grounding benchmark**、**11 项感知能力**。
- 基线包括：通用 VLM（Qwen3-VL-4B、Qwen3.5-9B、PaliGemma-3B、Qwen3.8-27B、Qwen3.7-Max、Kimi-K2.6、Kimi-K3、GPT-6 Astra）、专用接地检测器（GroundingDINO、DINO-R50、DETR-R50）、接地专长模型（Rex-Omni-3B、LocateAnything-3B、SAM）、embodied foundation（RynnBrain-2B、RynnBrain1.1、NVIDIA π₀/π₀.₅）、视频生成 backbone（Wan2.2-TI2V-5B、Cosmos-Predict2.5-2B）、GUI/空间 grounding（JEDI、UI-R1、GUI-Owl、RoboRefer、RoboPoint）、OCR（PaddleOCRv5、DocLayout-YOLO）；基准数据集包括 COCO、LVIS、Dense200、VisDrone、RoboSpatial、RefSpatial、ScreenSpot-Pro、ScreenSpot-V2、OSWorld-G、SROIE、M6Doc、HierText、ICDAR2015、TotalText 等。

### 主要结果
- **接地基准平均**：GroundingPI **73.68%**，超越更大规模 **GPT-6 Astra（71.54%）**。
- **Dense200 box grounding**：**74.53**（vs. Astra 65.04，Qwen3-VL-4B 14.02）。
- **RefCOCO avg**：**83.92**；**COCO**：**62.98**；**LVIS**：**56.02**。
- **RoboSpatial**：**73.77**；**RefSpatial（avg）**：**75.50**；**Unseen**：**75.32**。
- **ScreenSpot-Pro**：**96.15**；**ScreenSpot-V2**：**65.78**；**OSWorld-G**：**74.82**。
- **SROIE OCR**：**72.47**；**M6Doc layout**：**74.82**。
- **HumanRef pointing**：**88.79**；**RefCOCOg val**：**90.26**；**COCO pointing**：**84.79**；**Dense200 pointing**：**81.27**。
- **OCR HierText**：F1@.50=60.02，F1mIoU=41.70，Parse err.=0.06；**ICDAR2015**：F1@.50=76.79，F1mIoU=55.68，Parse err.=0.00。
- **OCR TotalText**：F1@.50=72.92；**SROIE**：F1@.50=92.82，Parse err.=0.00。
- **DocLayNet**：IoU 0.50 F1=96.05，mIoU F1=85.08；**M6Doc**：IoU 0.50 F1=92.31，IoU 0.95 F1=31.83，mIoU F1=74.82。

### 下游任务结果
- **自动驾驶（nuScenes）**：open-loop L2 误差 **0.296 m**（优于 Qwen3-VL-4B 的 0.301 m、RynnBrain 的 0.308 m）。
- **机器人操作 RoboTwin 2.0 Full SR**：**66.20%**（6 项评测中 5 项第一，含全部 4 个 OOD 设置）。
- **RoboTwin 2.0 Clean2Random（OOD 鲁棒性）**：**17.60%**（次优 Rex-Omni 14.10%）。
- **RoboCasa-GR1 Full SR**：**37.75%**（vs. RynnBrain 39.00%）。
- **相对最强 backbone 提升高达 24.8%**（relative）。
- **数据效率**：RoboCasa-GR1 用 **50%** 演示数据即超越所有 baseline 在 **75%** 数据下的表现：GroundingPI 50% 数据得 **28.75%** SR，RynnBrain 75% 数据为 27.75%。

### 消融实验
- **量化坐标 vs. 文本坐标**：Avg 从 71.21 提升至 73.68；文本坐标速度为量化方案的 **0.25×**。
- **视觉编码器对比**：MoonViT-V2（73.68）> MoonViT（73.17）> Qwen3-ViT（71.65）。
- **移除 RL 阶段**：降至 72.86。

### 输出效率
- **COCO**：GroundingPI **7.6 tokens/box** vs. SEED1.5-VL 的 148.8；**Dense200**：**5.1 vs. 74.5**。

## 相关工作脉络
- **通用 VLM（Qwen3-VL 系列、PaliGemma、GPT-6 Astra、Kimi-K2.6/K3）**：本文定位为感知原生模型，区别于通用 VLM 将感知作为辅助目标、缺乏统一 grounding 接口的设计。
- **专用接地检测器（GroundingDINO、DINO-R50、DETR-R50）**：本文提供多任务统一接口（referring/pointing/OCR/GUI 等），而专用检测器通常仅支持单一任务类型。
- **接地专长模型（Rex-Omni、LocateAnything、SAM）**：本文以 4B 参数实现 comparable 或更优性能，且支持自回归生成式输出，区别于 SAM 等分割模型或需多模块拼接的方案。
- **Embodied foundation（RynnBrain、NVIDIA π₀/π₀.₅）**：本文专注 grounding 感知能力，为下游 action policy 提供精确空间感知接口，而 embodied 模型直接耦合感知与动作。
- **GUI/空间 grounding（JEDI、UI-R1、GUI-Owl、RoboRefer、RoboPoint）**：本文在 ScreenSpot-Pro/V2、OSWorld-G 等基准上统一评估，证明通用 grounding foundation 可覆盖 GUI 理解任务。
- **OCR 模型（PaddleOCRv5、DocLayout-YOLO）**：本文通过统一 token 接口实现 OCR 与 grounding 融合，区别于专用 OCR pipeline。

## 局限性与未来方向
- 论文未明确讨论计算延迟与部署成本，仅报告 token 效率，实际推理延迟需进一步评估。
- GUI grounding（ScreenSpot-V2、OSWorld-G）表现相对 GPT-6 Astra 仍有差距，复杂交互场景泛化能力待提升。
- 模型缩放分析仅报告 pretraining tokens 从 88.4B 增至 221B 的效果，未探索更大参数规模（如 7B/14B）的 scaling law。
- 自述"规划与执行或需不同 foundation"（Takeaway 4），暗示 System 1/System 2 分离架构的合理性，但未给出统一框架设计。
- 多教师数据引擎依赖已训练 grounding 专家，初始标注质量受教师模型能力限制。

## 研究启发与可借鉴点
- **量化坐标 token 设计**：将连续坐标离散化为 1000 个 token（`<0>`–`<999>`）并共享词表，可实现高精度定位与高效输出（7.6 tokens/box vs. 148.8），值得迁移至其他空间感知任务。
- **GRPO 奖励设计**：组合覆盖度奖励（IoU-based $R_{set}$）与格式严格性奖励（$R_{strict}$）并通过 Z-score 加权（0.7/0.3），可推广至需要同时优化精度与输出规范性的生成任务。
- **感知原生接口理念**：将 grounding 作为 foundation model 的原生能力而非下游辅助目标，为 VLA/具身智能提供可复用的感知模块设计范式。
- **受控对比实验设计**：固定 Action DiT、action 数据、训练预算，仅更换 backbone 评估感知模块对下游任务的影响，该实验范式可迁移至其他多模态 foundation model 的 ablation 研究。
- **数据效率验证**：50% 数据超越 75% 数据 baseline 的结果证明高质量感知表示可显著减少下游训练需求，值得在低资源场景中复用。

## 关键术语表
- **GroundingPI**：4B 参数自回归 grounding foundation model，以视觉原语为原生能力，支持多任务统一接口。
- **量化坐标 token**：将连续坐标离散化为 1000 个 token（`<0>`–`<999>`），共享输出词表，实现高精度定位与高效序列化输出。
- **MoonViT-V2**：Kimi K3 采用的视觉编码器，作为 GroundingPI 的视觉 backbone。
- **GRPO（Group Relative Policy Optimization）**：强化学习算法，用于 grounding reward 优化，每组 8 个自回归响应，仅更新语言参数。
- **$R_{set}$ / $R_{strict}$**：两种 grounding reward，前者基于 IoU 匹配的覆盖度，后者综合多阈值 F1、定位、格式、数量、顺序惩罚。
- **Visual primitives**：视觉原语，指模型原生具备的接地、指向、OCR 等基础感知能力。
- **Open-loop L2 误差**：自动驾驶评估指标，预测轨迹与 ground truth 在 open-loop 设置下的 L2 距离。
- **OOD（Out-of-Distribution）**：分布外场景，用于评估模型在未见过的环境条件下的泛化鲁棒性。

## 关键数字
- 模型规模：4B 参数
- 视觉编码器：MoonViT-V2
- 语言解码器：Qwen3-4B
- 量化坐标 token 数：1000（`<0>`–`<999>`）
- 对比基线：44 个
- 评测 benchmark：34 个
- 感知能力类型：11 项
- Pretraining tokens：88.4B → 221B（缩放分析）
- Grounding 平均分：73.68%
- 输出效率：7.6 tokens/box（COCO），5.1 tokens/box（Dense200）

## 可复现要素
- **数据集**：COCO、LVIS、Dense200、VisDrone、RoboSpatial、RefSpatial、ScreenSpot-Pro、ScreenSpot-V2、OSWorld-G、SROIE、M6Doc、HierText、ICDAR2015、TotalText、nuScenes、RoboTwin 2.0、RoboCasa-GR1；**论文未提及是否全部公开**。
- **代码**：**论文未提及**是否开源。
- **权重**：**论文未提及**是否开源。
- **关键超参**：RL 组大小 8；reward 权重 0.7（$R_{set}$）/ 0.3（$R_{strict}$）；量化坐标 token 数 1000；投影层 2×2 聚合；SFT 先更新全部模块后冻结视觉编码器/投影层。
