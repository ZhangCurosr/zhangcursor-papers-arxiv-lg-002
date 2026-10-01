---
title: "Vidu-S2-Real-Time-Interactive-Editable-and-Spatial-Video-Gen"
source: https://arxiv.org/pdf/2609.11638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 16:59:35"
field: "实时流式视频生成与编辑"
keywords: ["real-time video generation", "streaming diffusion", "video editing", "spatial video", "Self-Replay Forcing", "Diffusion Forcing", "audio-visual avatar", "inference acceleration"]
innovations: ["Self-Replay Forcing 实现在线策略分布匹配并保留跨块梯度传播", "帧对齐注意力支持流式视频编辑下运动与时序严格对齐", "不对称噪声缓存调度解耦时序传播与空间细节恢复"]
benchmarks: ["StreamAV-Bench", "Sparkle-Bench", "OpenVE-Bench", "RefVIE-Bench", "ViViD"]
---

# 论文速读：Vidu S2: Real-Time Interactive, Editable, and Spatial Video Generation

## 一句话总结
本文提出 Vidu S2，包含实时交互数字角色模型 Vidu S2-Avatar 与实时视频编辑模型 Vidu S2-Editing，支持 720p 实时生成、动态参考更新、多模态指令遵循及风格迁移/虚拟试穿/角色替换/背景替换等编辑任务，并探索了 VR 头戴设备上的实时空间视频生成与编辑能力，在多项公开基准上全面超越现有开源与商业基线。

## 研究问题与动机
- **离线生成的局限性**：当前主流视频生成模型（如 Sora、Veo、Wan）采用离线扩散范式，用户提交提示词后需等待数分钟至数十分钟才能获得完整视频，期间无法进行任何主动交互，无法满足面对面交流、直播、游戏等实时互动视觉体验需求。
- **Vidu S1 的能力瓶颈**：Vidu S1 虽实现了 540p 实时流式生成，但分辨率受限、参考图像一旦启动即固定、难以支持大幅度身体动作（如舞蹈），且不支持对输入视频流的实时编辑。
- **需求规模差异**：若用户平均对实时互动内容的需求为 α、对离线生成内容的偏好为 β，且离线视频平均被观看 m > 100 次，则实时互动视频生成的总需求规模约为 β×N/m，远低于 α×N，表明实时互动具有更大的市场潜力。
- **空间视频的沉浸感缺口**：传统平面视频缺乏深度感知与在场感，VR 头显用户对高解析度、低延迟的空间视频内容存在强烈需求。

## 核心贡献（创新点）
1. **Self-Replay Forcing (SRF)**：一种在线策略分布匹配蒸馏方法，先执行自回归 rollout 并分离 KV cache，再对生成轨迹重新加噪并在同一计算图中因果回放，使梯度可跨块传播，解决了 Self-Forcing 中历史片段以干净状态输入且无法回传梯度的问题。
2. **720p 实时生成 + 轻量 Refiner**：在骨干网络保持低分辨率高速运行的同时，通过单步潜空间上采样与低噪声高分辨率缓存的不对称调度，实现 25~42 FPS 下的 720p 实时生成。
3. **帧对齐注意力 (Frame-Aligned Attention)**：Vidu S2-Editing 中每个目标帧仅读取对应时间步的源帧，参考图像对所有帧可见，确保编辑后的视频在运动与时序上与输入严格对齐，同时保持参考外观的全局一致性。
4. **实时空间视频转换管道**：将单目流式输出通过逐帧深度估计与水平视差映射转换为同步的左右眼视图，支持 VR 头显上的沉浸式交互；编辑管道兼容单目输入先编辑后转换、或立体输入直接联合编辑两种工作流。
5. **端到端推理加速栈**：集成 SageAttention、SpargeAttention、Sparse-Linear Attention 分层混合注意力、W8A8 GEMM 量化、算子融合与 CUDA Graph、以及 Ulysses 风格的多 GPU 上下文并行，使 VAE 编码/骨干/Refiner/VAE 解码在各 GPU 间按共享时间线复用，显著降低延迟。

## 方法详解
### 数据准备
- **Vidu S2-Avatar 数据流**：保留 Vidu S1 的五阶段管道（剪辑→过滤→语音处理→标注→嵌入），新增高质量独舞视频与 2D/3D 动画、多维权重清晰度筛选框架（分辨率/帧率/编解码器/位深/码率技术质量+纹理细节/边缘锐度/压缩伪影专家模型评估）、背景稳定化算子（对平滑运镜视频保留主体、估计背景几何变换并校正）。
- **时序密集标注**：用专家模型做物体检测与语音识别形成事实锚层，再由多智能体工作流完成识别→描述→验证→过滤，标注粒度假设从固定时长语音对齐区间改为事件边界切分。
- **Vidu S2-Editing 数据流**：从 80 万高质量视频采样四组各 20 万视频，分别构建风格迁移、主体替换、背景替换、虚拟试穿任务数据；风格迁移通过 NormalCrafter 估计表面法线视频，用条件生成模型在法线+参考图条件下重建原始视频再替换参考图为风格图；多种开源编辑模型产出候选数据后做后过滤与比较选择。

### 模型架构
- **Vidu S2-Avatar**：基于音频-视觉联合 Diffusion Transformer，给定参考图像 r 与条件序列 c¹:ᴺ，联合预测前 N 个片段的干净视频-音频潜变量：
  $$\hat{x}_0^{1:N} = f_\theta^{\text{avatar}}(x_t^{1:N}, t, r, c^{1:N})$$
  参考图像跨所有片段共享以维持外观与身份。
- **双向 I2V/R2V 训练**：同一模型统一训练图像到视频（r 为第一帧）与参考到视频（r 为外部参考图），采用分段条件监督而非整段单提示。
- **混合 Teacher/Diffusion Forcing**：将双向时序注意力替换为分块因果注意力掩码，每个片段仅 attends 参考图、当前片段条件与有效历史状态；训练时以固定概率混合 Teacher Forcing（干净历史）与 Diffusion Forcing（采样噪声级别的历史）。
- **Self-Replay Forcing (SRF)**：
  1. 学生对当前策略执行长自回归 rollout，生成的片段与 KV cache 被分离；
  2. 对整个 rollout 轨迹按 Diffusion Forcing 策略独立重新加噪；
  3. 因果回放整个重噪轨迹，外部分离历史保持不变，回放片段间共享同一计算图，使后续片段的损失可传播至前面片段；
  4. 损失为 DMD 损失 + 感知正则损失：
     $$\mathcal{L}_{\text{SRF}} = \mathcal{L}_{\text{DMD}}(f_\theta^{\text{causal}}(x_t^{1:N}, t, r, c^{1:N})) + \mathcal{L}_{\text{perc}}(\cdot)$$
- **偏好优化**：双向阶段使用扩散 DPO 提升视觉保真度、表情自然度、运动自然度与音视频同步；流式阶段使用 Streaming NFT 对因果骨干进行自我生成轨迹的奖励优化。
- **超分 Refiner**：骨干输出低分辨率潜变量 $\hat{x}_{0,\text{LR}}^i$，Refiner 在潜空间做上采样并固定步长 $t_r$ 加噪：
  $$z_{t_r}^i = (1-t_r)\mathcal{U}(\hat{x}_{0,\text{LR}}^i) + t_r \epsilon^i, \quad \epsilon^i \sim \mathcal{N}(0,I)$$
  骨干缓存使用高噪声级别 $\tau_B$ 传播粗粒度时序结构，Refiner 缓存使用低噪声级别 $\tau_R$ 保留局部外观，$\tau_B > \tau_R$，实现时序传播与空间细节恢复的解耦。
- **Vidu S2-Editing**：给定源视频 s¹:ᴺ、参考图 r 与条件序列 c¹:ᴺ，预测目标视频潜变量：
  $$\hat{x}_0^{1:N} = f_\theta^{\text{edit}}(x_t^{1:N}, t, s^{1:N}, r, c^{1:N})$$
  双向阶段采用帧对齐注意力（生成帧仅与同时间步源帧交互），参考图对所有帧可见；因果流式训练沿用混合 Forcing 与 SRF 框架。

### 推理基础设施
- **分层混合注意力**：SageAttention / SpargeAttention / SLA 按层敏感性选择。
- **W8A8 GEMM**：逐块量化线性层，细粒度缩放限制离群值影响。
- **算子融合与 CUDA Graph**：RMSNorm 与元素级操作融合，稳定序列用 Graph 捕获回放。
- **多 GPU 上下文并行**：Ulysses 风格划分激活内存，通信张量量化降低传输量。
- **模块间调度**：VAE 编码/骨干/Refiner/VAE 解码按共享时间线复用 GPU，减少空闲。

### 实时空间视频
- **生成**：Vidu S2-Avatar 生成单目片段 → 逐帧深度估计 → 以中心视图为基准水平方向反向扭曲生成左右眼视图 → 轻量孔洞填充与深度时序稳定。
- **编辑**：单目输入先编辑后转换；立体输入将左右视图水平拼接为宽流，一次通过编辑器再拆分。

### Agentic System
- VLM 代理将用户图文/语音指令转化为结构化提示（身份外观、表情姿态、手持物体、场景变化等），并在生成帧上执行视觉反馈：检查动作是否完成/部分完成/偏差，据此调整后续提示。
- 支持物体替换、场景转换、穿戴/摘除配件的状态保持。

## 实验与结果
### 数据集与基准
- **数字角色**：StreamAV-Bench（Progressive 与 Interactive 两个赛道，各 160 场景）；内部基准覆盖流式质量、长程稳定性、运行时更新、中断恢复、音视频同步、系统响应与成本。
- **视频编辑**：OpenVE-Bench、Sparkle-Bench、RefVIE-Bench、ViViD 测试集；内部编辑基准含 150 配对案例与 10 分钟长序列测试集。

### 公开基准对比
- **StreamAV-Bench（表 1）**：Vidu S2-Avatar 在所有指标上取得最优，VA=0.687、VQ=3.370、PQ=7.138、AQ=3.286、AVAlign=0.353、AVSync=0.617、AIF=2.985、SC=0.998、BC=0.993，全面超越 Live Avatar（VA=0.661、VQ=3.295）等最强开源基线。
- **Sparkle-Bench（表 2）**：Vidu S2-Editing 获 Overall=3.74，Ins.=4.00、Vis.=3.34、FgIn.=3.98、FgMo.=4.00、BgDy.=3.37、BgVi.=3.76，全部指标最优，超越 Decart-Lucy2.5（Overall=3.67）。
- **OpenVE & RefVIE 联合（表 3）**：Vidu S2-Editing 获 GS=4.71、BC=4.14、Overall=4.42、RefVIE Ovr.=3.78、Joint Ovr.=4.26，超越离线最强基线 Bernini-R 14B（Joint=3.92）达 0.34。
- **ViViD 虚拟试穿（表 4）**：Vidu S2-Editing 的 VFID=9.9515，显著优于 CatV²TON（19.5131）与 ViViD（21.8032）。

### 内部人工偏好评估
- **GSB 协议**：20 名专业评测员随机对比，Vidu S2-Avatar 较 Runway/PixVerse/HeyGen 在整体质量上分别获 85.7%/100%/100% 首选率；在运动、表情、语义遵循、时序一致性上均保持优势。
- 10~90 秒时长按段的评分曲线显示，Vidu S2-Avatar 的一致性与表达质量在长程中保持稳定，无显著漂移。
- 编辑任务 GSB 中，Vidu S2-Editing 整体质量、时序一致性均显著优于 Decart-Lucy2.5 与 XMax-X2.0。

### 定性案例
- 身份保持、发丝/手部/服饰精细几何、 occlusion 关系、背景一致性在多场景下均保持稳定；基线普遍出现面部漂移、肢体畸变、风格不完整等问题。

## 相关工作脉络
- **Self-Forcing / Diffusion Forcing**：Vidu S1 使用的 Self-Forcing 将生成历史以干净状态送入且切断梯度，本文提出的 SRF 通过重噪回放与因果计算图连通实现跨块梯度传播，解决数据效率与训练质量瓶颈。
- **TurboDiffusion / TurboServe**：两者提供单步/少步扩散加速与流式服务架构基础，本文在此基础上引入分层混合注意力、W8A8 GEMM、模块间时间线复用等进一步优化。
- **StreamAV-Bench / OpenVE / Sparkle-Bench / RefVIE / ViViD**：均为近年提出的流式音视频生成或视频编辑基准，本文在全部基准上取得 SOTA，体现了方法在多任务场景的通用性。
- **SageAttention / SpargeAttention / SLA**：均由本团队提出的高效注意力实现，本文根据层敏感性分层选用，体现推理加速方法的一体化整合。
- **NormalCrafter / VACE / Kiwi-Edit / Bernini 等**：本文在数据构建阶段利用多种开源编辑模型产出候选数据并通过后过滤与比较选择整合，展示了多模型协同的数据工程策略。
- **流式对话头像（Live Avatar、Hallo-Live、AvatarForcing 等）**：本文相较这类专注头部/语音的方法，扩展到全身舞蹈、动态参考、视频编辑与空间视频，覆盖更宽的交互范式。

## 局限性与未来方向
- **空间视频仍依赖单目深度估计**：当前管道通过 Monocular Depth Estimation 生成左右眼视图，在复杂遮挡与深度不连续区域仍存在孔洞与伪影，分辨率与延迟要求高于单目生成，实际 VR 部署仍有压力。
- **长程漂移尚未完全消除**：尽管 SRF 与偏好优化显著缓解累积误差，10 分钟级流式中仍存在偶发的身份/背景漂移现象，需更强的一致性先验。
- **数据构建成本较高**：风格迁移需训练条件 V2V 模型与多轮后过滤；舞蹈/动画数据、高质量法线估计与专家标注pipeline 复杂，规模化扩展成本可观。
- **未来方向**：扩展至全景空间视频（支持头部转动自由探索）、端到端可微立体生成而非深度驱动扭曲、进一步降低多 GPU 并行通信开销、探索更低比特（FP4/INT4）训练的稳定性边界。

## 研究启发与可借鉴点
- **SRF 的"rollout + 重噪回放 + 因果计算图连通"设计**：对任何自回归/流式扩散任务均有迁移价值，可替代传统 Teacher Forcing 以缩小 train-infer distribution gap，尤其适合长程时序建模。
- **不对称噪声缓存调度（骨干高噪声 + Refiner 低噪声）**：将时序传播与空间细节恢复解耦的思路可推广至超分、去噪、视频补帧等多阶段 pipeline，避免高分辨率缓存对长程一致性的干扰。
- **分层混合注意力策略（按层敏感性选择 Sage/Sparge/SLA）**：为低延迟推理提供可复用的精度-速度权衡框架，不同任务可按敏感度自动搜索各层最佳注意力实现。
- **流式模块时间线复用调度**：VAE/骨干/Refiner/解码按共享时间线分配 GPU 的设计，对多模块串行流水线（如 TTS + 头像 + 编辑）具有通用参考价值。
- **VLM 代理的视觉反馈闭环**：将生成帧回流给 VLM 用于动作完成判定与提示调整，可在交互式生成、角色扮演、教育陪伴等场景中复用。

## 关键术语表
- **Self-Replay Forcing (SRF)**：一种在线策略分布匹配蒸馏方法，先执行自回归 rollout 并分离历史，再对生成轨迹重新加噪并在因果计算图中回放，使梯度跨片段传播以提升流式训练质量。
- **Diffusion Forcing**：将历史状态以采样噪声级别喂入模型，使扩散模型在训练时即 exposure 到不同噪声水平的历史，增强对累积误差的鲁棒性。
- **帧对齐注意力 (Frame-Aligned Attention)**：编辑模型中目标帧仅 attend 同时间步源帧、参考图对所有帧可见的注意力设计，保证编辑后运动与时长不变、外观全局一致。
- **Streaming NFT**：基于自我生成轨迹的流式负感知微调，用于在因果骨干上进行奖励/偏好优化，缩小推理分布与训练分布差距。
- **DPO (Direct Preference Optimization) for Diffusion**：在扩散模型上直接优化偏好分布，无需额外奖励模型即可提升视觉质量与指令遵循。
- **W8A8 GEMM**：权重与激活均 8-bit 量化的矩阵乘，通过逐块缩放抑制离群值，兼顾精度与推理速度。
- **SageAttention / SpargeAttention / SLA**：本团队提出的一系列高效注意力实现，分别基于 int8 量化、稀疏掩码与可微调稀疏-线性混合策略。
- **GSB 协议**：Good/Same/Bad 人工偏好对比协议，专业评测员在随机顺序下对模型对进行 pairwise 判定，用于更贴近感知的综合评估。

## 可复现要素
- **数据集**：训练数据规模与来源在论文中以 Pipeline 描述为主，未给出公开下载链接；内部基准未公开。公开基准（StreamAV-Bench、OpenVE-Bench、Sparkle-Bench、RefVIE-Bench、ViViD）为已有公开资源。
- **代码/权重**：论文声明可用在线 Demo（https://vidu.com/vidu-stream），但未明确说明模型权重与训练代码是否开源；论文未提及开源计划。
- **关键超参**：骨干与 Refiner 的噪声级别满足 τ_B > τ_R，Refiner 固定细化步长 t_r；分层注意力方法按层选择（SageAttention/SpargeAttention/SLA）；V8A8 GEMM 位宽为 int8；上下文并行采用 Ulysses 风格；具体数值未在论文中完整列出，论文未提及。
