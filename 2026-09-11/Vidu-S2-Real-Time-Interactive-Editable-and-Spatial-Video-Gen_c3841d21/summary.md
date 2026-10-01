---
title: "Vidu-S2-Real-Time-Interactive-Editable-and-Spatial-Video-Gen"
source: https://arxiv.org/pdf/2609.11638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:59:01"
field: "实时交互式视频生成"
keywords: ["实时视频生成", "流式扩散模型", "数字角色", "视频编辑", "空间视频", "Self-Replay Forcing", "Diffusion Forcing", "推理加速"]
innovations: ["Self-Replay Forcing: 将on-policy自回归轨迹重新加噪后进行梯度连通因果回放，解决历史误差累积漂移", "帧对齐注意力: 目标帧仅attend同时间步源帧，保证编辑后运动与时间完全保持输入一致", "非对称噪声缓存调度: 骨干网高噪声缓存保时序一致性，Refiner低噪声高分辨率缓存恢复空间细节"]
benchmarks: ["StreamAV-Bench", "Sparkle-Bench", "OpenVE-Bench", "RefVIE-Bench", "ViViD"]
---

# 论文速读：Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation

## 一句话总结
本文提出 Vidu S2，包含两个实时视频生成/编辑模型：Vidu S2-Avatar（支持 720p 实时数字角色生成，具备动态参考图交互与更强指令跟随能力）和 Vidu S2-Editing（支持实时视频流编辑，涵盖风格迁移、虚拟试穿、角色替换和背景替换），并探索了空间视频生成与编辑在 VR 头显上的可行性。

## 研究问题与动机
1. **现有视频生成模型多为离线单轮范式**：用户输入提示词后需等待数分钟，无法在生成过程中进行实时交互，无法满足直播、游戏、面对面交流等需要即时响应的视觉交互需求。
2. **Vidu S1 存在分辨率低、参考图固定、运动控制弱等局限**：Vidu S1 仅支持 540p、参考图一旦启动不可更改、难以跟随大幅度身体动作（如舞蹈）、不支持对输入视频流进行编辑。
3. **实时空间视频生成尚未解决**：传统立体视频制作成本高、难以修改，缺乏面向 VR 头显的实时空间视频生成与编辑系统。

## 核心贡献（创新点）
1. **Self-Replay Forcing (SRF)**：一种基于 on-policy DMD 的训练策略，通过将学生模型自回归轨迹重新加噪后进行梯度连通的因果回放，使梯度能在回放段之间传播而不经过原始 rollout，解决了历史误差累积导致的漂移问题。与 Self-Forcing 的本质区别在于梯度连通性——SRF 允许后续段的损失反传至前序段。
2. **720p 实时生成架构**：采用轻量级 Refiner 在潜空间增加一步上采样恢复高频细节，配合非对称噪声缓存调度（骨干网使用高噪声缓存，Refiner 使用低噪声高分辨率缓存），将实时生成分辨率从 540p 提升至 720p 且保持 25~42 FPS。
3. **帧对齐注意力（Frame-aligned Attention）**：编辑模型中每个目标帧仅 attend 到同一时间步的源帧，确保编辑后视频与输入视频的运动和时间完全对齐，同时参考图对所有帧可见以维持外观一致性。
4. **实时空间视频生成/编辑管线**：将单目光流转换为同步的左右眼视图，支持对单目或立体输入进行风格迁移、虚拟试穿、角色和背景替换，实现 VR 头显上的沉浸式实时交互。
5. **VLM 智能体系统**：构建基于视觉语言模型的 prompt 生成与视觉反馈闭环，支持用户在流生成过程中随时上传参考图触发对象/场景/服饰替换，并能追踪动作完成情况以动态调整 prompt。

## 方法详解
### 数据准备管线
- **Avatar 数据**：在 Vidu S1 五阶段管线（Clipping/Filtering/Speech Processing/Captioning/Embedding）基础上扩展，新增独舞视频和 2D/3D 动画数据；引入高精度视频筛选算子（多维混合质量评估）和背景稳定化算子（对平滑相机运动视频做几何校正）；重构为时序密集描述（temporally ordered dense captions），按事件边界分割 chunk。
- **Editing 数据**：从 Avatar 过滤后的数据中抽取 80 万高质量视频，分设四个互斥子集（各 20 万）构建风格迁移、主体替换、背景替换、虚拟试穿数据；风格迁移数据通过 NormalCrafter 提取表面法线视频后结合风格参考图生成；综合评估多个开源编辑模型产出进行后筛选。

### 模型架构
- **Vidu S2-Avatar**：音频-视觉联合 Diffusion Transformer，给定参考图 $r$ 和条件序列 $c^{1:N}$，联合预测干净视频-音频潜变量 $\hat{\pmb{x}}_0^{1:N} = f_\theta^{\text{avatar}}(\pmb{x}_t^{1:N}, t, r, \pmb{c}^{1:N})$。
- **双向 I2V/R2V 联合训练**：同一模型同时支持 image-to-video 和 reference-to-video，按 segment 粒度监督而非整序列单一 prompt。
- **因果适配**：将双向时序注意力替换为 block-wise causal attention，采用混合 Teacher Forcing + Diffusion Forcing 训练，提升对不完整历史状态的鲁棒性。
- **Self-Replay Forcing 损失**：$\mathcal{L}_{\text{SRF}} = \mathcal{L}_{\text{DMD}}(\cdot) + \mathcal{L}_{\text{perc}}(\cdot)$，DMD 对齐 student 回放分布与 teacher，perceptual loss 缓解模式崩溃。
- **偏好优化**：双向阶段使用 Diffusion DPO 提升视觉保真度和音视同步；流式阶段使用 Streaming NFT 对自生成轨迹进行奖励优化。
- **超分 Refiner**：$\pmb{z}_{t_r}^i = (1-t_r)\mathcal{U}(\hat{\pmb{x}}_{0,\text{LR}}^i) + t_r \epsilon^i$，骨干网缓存噪声水平 $\tau_B$ 高于 Refiner 缓存 $\tau_R$，实现时序传播与空间细节恢复解耦。

### 推理基础设施
- **分层混合注意力**：按层敏感度选择 SageAttention / SpargeAttention / SLA。
- **W8A8 GEMM 量化线性层**：fine-grained scaling 限制 outlier 影响，CUDA 内核优化降低延迟。
- **Kernel Fusion + CUDA Graph**：RMSNorm 与 elementwise 操作融合，稳定序列用 CUDA Graph 捕获回放。
- **多 GPU 上下文并行**：Ulysses-style 分布式激活内存，通信张量量化降延迟。
- **模块间细粒度调度**：VAE encoder/backbone/Refiner/VAE decoder 按共享时间线复用 GPU。

### Agentic System
- VLM 智能体读取用户指令与参考图，生成结构化 prompt（身份/表情/姿态/手持物分离描述）；按时序 review 生成帧，判断动作完成状态以指导后续 prompt 更新；支持参考图识别（手持物/背景/服饰）并自动生成场景过渡 prompt。

## 实验与结果
### 评测基准
- **Avatar**：StreamAV-Bench（160 场景，Interactive Track 每 30s 一次运行时更新，最长 180s）；内部 benchmark（时长分层 10-90s、中断恢复、5 分钟稳定性、端到端延迟）。
- **Editing**：Sparkle-Bench、OpenVE-Bench、RefVIE-Bench、ViViD test set；内部 benchmark（150 配对案例，含 10 分钟长序列测试）。

### 主要结果（公开基准）
- **StreamAV-Bench**：Vidu S2-Avatar 在所有指标上均为最佳——VA 0.687↑、VQ 3.370↑、PQ 7.138↑、AQ 3.286↑、AVAlign 0.353↑、AVSync 0.617↓（最低）、AIF 2.985↑、SC 0.998↑、BC 0.993↑。
- **Sparkle-Bench**：Vidu S2-Editing Overall 3.74（最高），全局指令遵循 4.00、前景指令 3.98、前景运动 4.00。
- **OpenVE + RefVIE**：Joint Overall 4.26，超越最强离线基线 Bernini-R 14B 约 0.34；OpenVE GS 4.71、BC 4.14、Overall 4.42；RefVIE Overall 3.78。
- **ViViD（虚拟试穿）**：VFID 9.9515，大幅优于 CatV²TON (19.51) 和 ViViD (21.80)。

### 人类偏好评估（GSB）
- **Avatar**：对比 Runway/PixVerse/HeyGen，在整体质量、运动、表情、语义遵循、时间一致性上全面占优；与 Runway 整体质量偏好率达 85.7%，与 PixVerse/HeyGen 达 100%。
- **Editing**：平均一致性得分 3.56 vs Decart-Lucy2.5 的 2.59 vs XMax-X2.0 的 1.67；整体质量和时间一致性优势最大。
- **时长分层分析**：在 10-90s 范围内各项指标（一致性/视频质量/运动质量/表情/整体质量）保持高位，无明显漂移衰减。

## 相关工作脉络
1. **Vidu S1**（作者前作）：首次实现 540p 实时流式音频驱动数字角色生成，但分辨率低、参考图固定、运动能力有限；Vidu S2 在此基础上显著提升分辨率、引入动态参考图和舞蹈指令跟随。
2. **Self-Forcing / OmniForcing / AvatarForcing**：解决自回归视频扩散中的 train-test gap；Vidu S2 的 SRF 在此基础上引入梯度连通的回放机制，解决历史 detached 导致的训练效率损失。
3. **Diffusion Forcing**（Chen et al., 2024）：对各段独立注入不同噪声水平；Vidu S2 将 DF 思想融入 SRF 的回放阶段和混合 Teacher/Diffusion Forcing 训练策略。
4. **TurboDiffusion / TurboServe**（作者前作）：为实时视频生成提供推理加速基础；Vidu S2 在此基础上进一步整合多层混合注意力、W8A8 GEMM、CUDA Graph 和多 GPU 调度优化。
5. **SageAttention / SpargeAttention / SLA**（作者团队系列工作）：Vidu S2 针对不同注意力层选择最优加速算子，实现精度与延迟的最佳权衡。
6. **Streaming NFT / DiffusionDFT**（Zheng et al., 2026）：在线扩散强化学习框架；Vidu S2 在流式阶段将其应用于自生成轨迹的奖励优化，提升指令遵循和视觉质量。
7. **开源视频编辑模型**（Bernini, Kiwi-Edit, VInO, OmniWeaving, LiveEdit, Decart-Lucy2.5, XMax-X2.0 等）：Vidu S2-Editing 在 Sparkle-Bench、OpenVE、RefVIE、ViViD 上全面超越这些离线和流式基线。

## 局限性与未来方向
1. **空间视频分辨率与延迟仍面临挑战**：VR 头显需要更高分辨率（覆盖大视场角）和极低延迟（避免头部运动与画面不同步），当前系统距实用部署尚有距离。
2. **仅支持固定视角空间视频**：当前管线生成固定视点的左右眼视图，用户无法自由转头探索场景；未来需扩展至全景空间视频。
3. **高度压缩视频在 720p 下仍可能模糊**：虽引入了多维清晰度筛选，但重度压缩素材在放大后仍可能显现纹理损失和压缩伪影。
4. **背景稳定化算子排除大位移镜头**：为保留运动多样性而引入的背景稳定化处理，仅适用于相机平移有限的视频，复杂运镜素材仍被排除。
5. **未公开代码与权重**：论文未提及代码和模型权重是否开源，模型推理基础设施细节（如具体 GPU 配置、显存占用）也未完整披露。

## 研究启发与可借鉴点
1. **SRF 训练策略的可迁移性**：Self-Replay Forcing 将 on-policy 轨迹回放与梯度连通结合的思路，可迁移至其他自回归视频/音频生成任务，缓解历史误差累积和 drift 问题。
2. **非对称噪声缓存调度设计**：骨干网用高噪声缓存保证时序一致性、细化网络用低噪声高分辨率缓存恢复细节——这一"时序-空间解耦"设计可推广至其他超分增强或层级生成任务。
3. **帧对齐注意力用于编辑任务**：目标帧仅 attend 同时间步源帧的设计，保证了运动保持的同时实现外观编辑，该约束机制可推广至其他时序对齐敏感的任务（如音视频编辑、字幕生成）。
4. **VLM 智能体闭环反馈**：用 VLM 读取生成帧并动态调整后续 prompt 的 agentic 架构，可用于提升交互式生成系统的指令遵循率和动作完成情况追踪。
5. **多维度视频质量筛选框架**：综合分辨率、帧率、编解码器、比特深度、纹理细节、边缘锐度、压缩伪影的加权评分筛选训练数据，对追求高分辨率实时生成的任务具有直接参考价值。

## 关键术语表
**Self-Replay Forcing (SRF)**：一种 on-policy 分布匹配蒸馏策略，将学生模型自回归 roll-out 的轨迹重新加噪后进行梯度连通的因果回放，使梯度可在段间传播而不反传原始 roll-out。

**Diffusion Forcing**：对各视频段独立采样不同噪声水平进行训练的自回归扩散方法，训练时历史段以随机噪声水平输入，提升模型对不完全历史的鲁棒性。

**StreamAV-Bench**：专为流式音频视频生成设计的评测基准，包含 Progressive 和 Interactive 两个 track，评测视觉美学、音视同步、主体/背景一致性等指标。

**Frame-aligned Attention**：编辑模型中目标帧仅 attend 对应时间步源帧的注意力机制，保证编辑后视频运动与时间完全保持输入一致。

**Streaming NFT**：面向流式生成场景的负样本感知微调方法，基于自生成轨迹进行奖励优化，对齐推理时的自回归分布。

**Self-Forcing**：将教师模型的干净历史状态作为学生模型的 conditioning 输入，缩小 train-test gap 的自回归视频生成训练技巧。

**DPO（Direct Preference Optimization）**：直接基于偏好数据优化扩散模型分布对齐的方法，无需额外 reward model，此处用于提升视觉保真度和音视同步。

**SageAttention / SpargeAttention / SLA**：作者团队系列高效注意力实现，分别基于 int8/int4 量化、稀疏掩码和可微调稀疏-线性注意力，用于降低推理延迟。

## 可复现要素
- **数据集**：训练数据来自大规模视频收集与多阶段处理管线（Clipping/Filtering/Captioning/Embedding），部分数据引用自公开数据集（StreamAV-Bench、OpenVE-Bench、Sparkle-Bench、RefVIE-Bench、ViViD）。论文未明确说明训练数据是否公开。
- **代码**：论文未提及代码开源情况。
- **权重**：论文未提及模型权重是否开源；提供在线 demo 链接 https://vidu.com/vidu-stream。
- **关键超参**：骨干网缓存噪声水平 $\tau_B$、Refiner 缓存噪声水平 $\tau_R$（$\tau_B > \tau_R$）、Refiner 固定细化时间步 $t_r$、双任务混合概率、DPO/NFT 优化超参等论文未具体给出。
- **推理环境**：使用 SageAttention、SpargeAttention、SLA、W8A8 GEMM、Triton/CUDA kernel fusion、CUDA Graph、Ulysses 上下文并行，具体 GPU 配置和帧率数据未完整披露。
