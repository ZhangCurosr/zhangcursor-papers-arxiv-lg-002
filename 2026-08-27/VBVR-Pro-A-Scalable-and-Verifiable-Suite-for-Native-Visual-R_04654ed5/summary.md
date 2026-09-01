---
title: "VBVR-Pro-A-Scalable-and-Verifiable-Suite-for-Native-Visual-R"
source: https://arxiv.org/pdf/2608.26105v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:15:53"
field: "视觉推理与生成模型"
keywords: ["native visual reasoning", "verifiable reward scorer", "reinforcement learning", "visual generation", "interleaved generation", "diffusion model"]
innovations: ["构建300任务程序化生成视觉推理数据集并提供可验证评分器", "引入CPS采样解决视觉推理RL中探索与保真度的权衡", "通过反事实诊断和中间状态干预首次证明视觉轨迹比语言链式思维对视觉推理更关键"]
benchmarks: ["VBVR-Pro-Bench", "RISE-Video", "V-ReasonBench", "MME-CoF-Pro", "VideoThinkBench", "BabyVision", "IntelligentVBench", "RULER-Bench"]
---

# 论文速读：VBVR-Pro: A Scalable and Verifiable Suite for Native Visual Reasoning

## 一句话总结
VBVR-Pro 构建了一个封闭循环测试平台，通过 300 个程序化生成的视觉推理任务、任务特定的可验证奖励评分器，以及跨图像/视频/交织生成器的受控比较，首次使"原生视觉推理"（以视觉状态本身作为推理基底而非语言）变得可训练、可验证、可优化且可实验控制。

## 研究问题与动机
1. **缺乏可扩展训练源**：现有视觉推理基准多为评估设计，极少提供训练数据；已有的大规模合成数据集（如 VBVR）偏重于简化符号设置，其是否能教授可迁移的视觉推理技能尚不明确。
2. **可靠的反馈信号缺失**：主流 VLM-as-a-judge 范式在需要精确计数、细粒度空间关系、时序一致性和规则满足的视觉推理任务上频繁出现误判（数值不精确、忽略关键证据、误解任务规则），且评分不稳定（同一视频多次评估 55%–93% 的样本得分会变化）。
3. **跨模态比较缺乏受控环境**：视频、图像、交织（text-image interleaved）三种生成基底在视觉推理中的作用尚未在同一任务分布和验证协议下被系统比较。
4. **核心科学问题**：视觉生成模型是否能够通过生成视觉状态本身进行推理（native visual reasoning），还是仅拟合了指令模板？视觉轨迹（visual trajectory）与语言链式思维（chain of thought）哪个才是更关键的推理基底？

## 核心贡献（创新点）
1. **可扩展的任务空间**：VBVR-Pro 构建 300 个程序化生成任务（5 大认知能力：感知、空间、变换、抽象、知识），提供 125 万训练实例，在 7 个外部基准上平均提升超过 20 个百分点，并证明这是视觉推理能力的迁移而非指令模式拟合。
2. **任务特定的可验证奖励评分器**：摒弃 VLM-as-a-judge，为每个任务设计基于结构化语义提取（HSV 分割、轮廓检测、OCR、轨迹追踪）的确定性评分器，在逐样本级别与人类判断高度一致，且支持大规模强化学习优化。
3. **跨模态受控机制研究**：在 30+ 个图像/视频/交织生成器上统一任务分布进行比较，发现视频在持久时空状态跟踪任务上最强，交织生成是计算高效的替代方案，并首次提供视觉轨迹因果作用的消融证据。
4. **首个面向视觉推理的多任务强化学习闭环**：引入系数保持采样（CPS）解决探索-保真度权衡问题，证明可验证奖励比 VLM 奖励能带来更稳定、更大的性能提升（整体 +0.548 vs +0.508）。

## 方法详解
**程序化任务生成**：300 个任务分布为 5 类认知能力，每类任务由参数化程序生成，随机化外观、布局、对象属性和文本提示，同时生成视频/图像/交织三种模态的实例，并附带结构化元数据（随机种子、任务规范、完整解、关键元素属性）。

**三种图像输出模式**：Last-Frame（仅最终状态，适合选择题如数独）、Key-Frame（展示关键中间状态，适合多步规划如迷宫）、Multi-Frame（均匀采样中间状态，适合评估过程完整性）。

**可验证评分器设计**：避免像素级比较，采用语义实体级验证。软约束任务用加权求和（如 Fig. 5(a)），硬约束任务用乘法组合（单一违规即大幅扣分，如 Fig. 5(b) 迷宫任务中 $S = S_{\text{traj}} \cdot g_{\text{wall}} \cdot g_{\text{key}}$）。

**强化学习实现**：采用 Flow-GRPO 类组相对优化算法，引入 CPS 采样公式：$x_{t-\Delta t} = (1-(t-\Delta t))\hat{x}_0 + (t-\Delta t)\cos(\eta\pi/2)\hat{x}_1 + (t-\Delta t)\sin(\eta\pi/2)\epsilon$，在 $\eta=0.7$ 时达到探索与保真度的最优平衡。训练采用一步延迟异步流水线提升 GPU 利用率。

## 实验与结果
**主要基准**：VBVR-Pro-Bench（100 个任务，50 in-domain + 50 out-of-domain），以及 RISE-Video、V-ReasonBench、RULER-Bench、MME-CoF-Pro、VideoThinkBench、BabyVision、IntelligentVBench 共 7 个外部基准。

**SFT 结果**：9 个开源模型经 VBVR-Pro 训练后平均提升 +0.290，In-domain 平均 +0.401，Out-of-domain 平均 +0.179。最强模型 VBVR-Pro-SenseNova-U1 在 in-domain 达到 0.811，但仍远低于人类水平（ceiling 约 0.77 的 78%）。

**RL 结果（Table 8）**：RLVR 最优，整体 0.548，In-domain 0.719，Out-of-domain 0.377；RLVLM 整体 0.508；SFT 整体 0.503。RLVR 在 Out-of-domain 上较 SFT 提升 15%，表明可验证奖励能促进跨分布泛化。

**外部迁移（Table 7）**：VBVR-Pro-Wan2.2-I2V-A14B 在 V-ReasonBench 上从 10.21 提升至 38.22（+28.01），VideoThinkBench 从 25.71 提升至 52.86（+27.15），MME-CoF-Pro 从 34.13 提升至 48.39（+14.26）。

**消融发现**：移除中间视觉状态导致 SenseNova 整体下降 0.111，而将中间文本替换为占位符仅下降 0.009；推理干预实验显示中间图像被污染后性能降至 0.190，而中间文本被删除后仅降至 0.533。

## 相关工作脉络
1. **VBVR [58]**：前作，提供 150 个任务和 Verifiable 评估，但任务规模和推理深度有限；本文在此基础上扩展到 300 任务（150 重新实现+150 全新设计），新增 RL 训练子集和跨模态比较。
2. **Zebra-CoT [31] / StructCoT [59]**：大规模交织推理数据集，但非程序化生成，无法提供可验证评分器；本文强调程序化生成+元数据记录是实现可验证评估的关键。
3. **DiffThinker [22] / EndoCoT [10]**：将视觉推理视为图像生成任务的先行工作，但仅在少量任务上验证；本文首次系统性地证明视觉推理可通过大规模训练和强化学习优化。
4. **Flow-GRPO [33] / DanceGRPO [68]**：将强化学习引入视觉生成的代表工作，使用 VLM 作为奖励；本文证明 VLM 奖励在视觉推理上不稳定，需任务特定的可验证评分器。
5. **OpenCoF [6] / VideoThinkBench [56]**：提出 Chain-of-Frames 范式观察视频生成中的推理行为；本文在此基础上引入 Chain-of-Step 分析（扩散去噪步层面的推理可视化），并提供系统性归因证据。
6. **RISE [74] / V-ReasonBench [39] / MME-CoF-Pro [47]**：现有视觉推理基准均采用 VLM-as-a-judge 评估；本文指出这些基准评估不可靠，并提供全套可验证替代方案。

## 局限性与未来方向
1. **程序化生成任务的生态局限性**：300 个任务虽覆盖五大认知能力，但均为程序化生成的抽象场景（空白画布、网格、游戏规则），尚未覆盖真实世界复杂视觉场景。
2. **训练数据规模有限**：125 万训练实例相对于预训练数据量较小，且仅训练一个 epoch，可能未充分挖掘数据潜力。
3. **RL 训练的长期稳定性未充分验证**：VLM-based RL 在步骤 400–500 出现性能下降，表明在更长的 RL 训练曲线中奖励信号的稳定性仍是挑战。
4. **跨模态比较的广度限制**：30+ 个模型虽覆盖了主流架构，但未包含最新闭源模型（如 GPT-Image-2、Seedance 2.0 的训练版本），且未涉及音频-视觉联合推理。

## 研究启发与可借鉴点
1. **程序化生成+结构化元数据的全链路可验证框架**：将任务描述、求解器、输出渲染和元数据生成统一在一个 pipeline 中，可实现零人工标注的"训练-评估-优化"闭环，可直接迁移至视频编辑、仿真控制等任务。
2. **CPS（Coefficients-Preserving Sampling）用于探索-保真度权衡**：在需要离散决策的生成任务中，CPS 比传统 SDE 更能保持视觉质量的同时提供有意义的探索噪声，可推广至其他需要高随机性探索的视觉生成 RL 场景。
3. **反事实诊断验证"学到的是推理而非拟合"**：通过输入重述不变性、几何反事实、中间状态干预三类实验，建立了一套评估模型是否真正掌握可迁移推理能力的标准协议，可作为后续工作的验证 baseline。
4. **一步延迟异步 RL 流水线**：将 rollout 生成、预处理、奖励计算和策略优化重叠执行，在不增加策略陈旧度的前提下显著提升 GPU 利用率，对大规模 RL 训练具有直接参考价值。

## 关键术语表
**Native Visual Reasoning**：以视觉状态（图像/视频）本身作为推理基底，模型通过构造、更新和优化视觉状态解决问题，而非将视觉信息转化为语言后进行推理。
**Chain-of-Step**：视觉推理在扩散去噪步骤中逐步展开的现象，早期步骤探索多种可能路径，后期步骤收敛到最终答案。
**Verifiable Reward Scorer**：基于任务特定规则和语义实体提取的确定性评分器，不依赖 VLM judge，可在逐样本级别与人类判断高度对齐。
**Coefficients-Preserving Sampling (CPS)**：一种保留预测噪声和新增高斯噪声系数关系的采样策略，相比传统 SDE 能在增加随机探索的同时不破坏生成质量。
**VBVR-Pro-Bench**：包含 100 个任务（50 in-domain + 50 out-of-domain）的评测基准，每任务 5 个实例，配有任务特定的可验证评分器。
**Interleaved Generation**：在文本推理步骤之间交替插入中间图像的状态，通过外部化中间视觉状态降低计算成本的同时保持推理能力。
**Group-relative Optimization**：将生成视为序列决策过程，通过组内相对奖励（如 GRPO）优化策略，避免绝对奖励的尺度问题。
**In-Domain / Out-of-Domain**：In-domain 指与训练分布同源的 held-out 实例，OOD 指完全不同的任务分布，用于评估迁移能力。

## 可复现要素
- **数据集**：VBVR-Pro-Dataset（125 万训练实例，100 任务基准）和 VBVR-Pro-Bench，论文声明"release all data, models, scorers, and code"。
- **代码**：论文声明开源全部代码和评分器，具体链接见论文附录。
- **权重**：9 个微调模型的checkpoint（VBVR-Pro-前缀）已发布；部分 SenseNova-U1 使用未公开发布的预训练 checkpoint 作为起点。
- **关键超参**：SFT 分辨率 512×512，1 epoch；LoRA rank 32，lr 1e-4（图像/视频）/1e-5～2e-5（交织）；RL 使用 CPS η=0.7，30 步采样，学习率 5e-6，batch size 32 prompts × 32 trajectories。
