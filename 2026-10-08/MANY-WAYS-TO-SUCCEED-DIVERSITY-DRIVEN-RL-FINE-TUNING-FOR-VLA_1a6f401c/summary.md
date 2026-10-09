---
title: "MANY-WAYS-TO-SUCCEED-DIVERSITY-DRIVEN-RL-FINE-TUNING-FOR-VLA"
source: https://arxiv.org/pdf/2610.09943v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:51:32"
field: "具身智能与VLA鲁棒微调"
keywords: ["VLA", "reinforcement fine-tuning", "behavioral diversity", "GAK", "intrinsic reward", "generalization", "potential-based shaping"]
innovations: ["揭示RFT中全局行为收缩与成功模式多样化并存现象", "提出DRIVE成功条件化内在奖励以显式激励任务有效多样性", "基于势函数塑形保证最优策略不变性的多样性奖励设计"]
benchmarks: ["LIBERO-Plus", "ManiSkill3", "RoboTwin 2.0", "AgileX PiPER-X 真机"]
---

# 论文速读：MANY-WAYS-TO-SUCCEED-DIVERSITY-DRIVEN-RL-FINE-TUNING-FOR-VLA

## 一句话总结
论文通过研究发现强化学习微调（RFT）在收缩全局行为的同时会多样化成功轨迹，据此提出DRIVE框架，将成功行为多样性转化为显式的内在奖励目标，从而在LIBERO-Plus、ManiSkill3、RoboTwin 2.0及真实机器人平台上显著提升了VLA模型的分布外（OOD）泛化能力。

## 研究问题与动机
- VLA模型经RL微调后任务成功率提升，但在视觉外观、场景配置和执行条件等分布外偏移下泛化仍不稳定。
- 标准二元任务奖励仅区分成功/失败，无法区分不同成功解，因而缺乏对成功模式多样性的显式激励。
- 直接移植无监督技能发现或策略多样化方法会奖励由执行速度、时序错位或无关动作造成的表面差异，甚至奖励失败轨迹。
- 现有研究多关注训练数据与模型能力提升，较少分析RFT过程中行为分布演变及其对OOD泛化的作用。

## 核心贡献（创新点）
- **揭示RFT探索动态**：证明RFT训练过程中全局行为相似性上升，但成功轨迹间的相似性下降，表明成功模式在收缩中被多样化，而非坍塌。
- **提出DRIVE多样性驱动框架**：通过在匹配任务条件下对 rollout 分组进行时序对齐的GAK比较，将成功轨迹的相对行为新颖性转化为内在奖励，避免奖励失败或表面差异。
- **设计成功条件化的势函数奖励塑形**：推导出成功条件化的多样性势函数与相邻状态差值奖励，并证明该变换保持最优策略不变性。
- **跨仿真与真机的一致性提升**：在 π₀ 上将 OOD 均值提升 5.3 点，在 π₀.₅ 上提升 2.0 点；在双臂 AgileX PiPER-X 真机上将 OOD 平均成功率从 64.1% 提升至 73.3%（+9.2 点）。

## 方法详解
- **轨迹特征表示**：对每次策略决策，复用VLM-prefix表示并对有效prefix token做均值池化，得到逐决策特征序列 $z_{i,c}$；终止状态也提取特征，得到长度 $L_i+1$ 的序列 $Z_i$。
- **全局对齐核（GAK）相似度**：使用带移位cosine的局部相似性 $\kappa(z_{i,t}, z_{j,u})$，通过动态规划累积所有单调对齐路径的相似性，公式为 $G_{i,j}(t,u) = \kappa \cdot [G(t-1,u) + G(t,u-1) + G(t-1,u-1)]$，最终归一化为 $\widetilde{K}_{ij} \in [0,1]$。
- **组内多样性度量**：对同一任务与条件分组内的N条轨迹，计算 $\mathrm{sim}_i = \frac{1}{N-1}\sum_{j \ne i} \widetilde{K}_{ij}$，并定义相对多样性 $d_i = 1 - \mathrm{sim}_i$。
- **成功条件化归一化**：仅在成功集合 $S_g$ 内对 $d_i$ 做 min-max 裁剪归一化为 $\hat{d}_i \in [0,1]$，失败轨迹的 $\hat{d}_i=0$，避免奖励非任务有效的变化。
- **势函数与内在奖励**：定义成功条件化多样性势 $\Phi(s_{i,c}) = y_i \cdot \hat{d}(s_{i,c})$，内在奖励取相邻状态势差 $r_{i,c}^{\mathrm{div}} = \gamma \Phi(s_{i,c+1}) - \Phi(s_{i,c})$；沿轨迹折扣求和后望远镜化为 $r_{\tau_i}^{\mathrm{div}} = \gamma^{L_i} \cdot y_i \cdot \hat{d}_i$。
- **最终奖励与策略不变性**：总奖励为 $r_{\tau_i}^{\mathrm{DRIVE}} = r_{\tau_i}^{\mathrm{env}} + \alpha \cdot r_{\tau_i}^{\mathrm{div}}$，其中 $\alpha$ 控制多样性奖励强度；由于是势函数形奖励塑形，所有最优策略集合保持不变。

## 实验与结果
- **基准与设置**：LIBERO-Plus（Spatial/Object/Goal/Long，构造IND/OOD扰动）、ManiSkill3（视觉-语言、语义、执行三类偏移）、RoboTwin 2.0（Click Bell/Press Stapler）；骨干模型为 $\pi_0$ 与 $\pi_0.5$，均从同一SFT检查点初始化并采用PPO+Flow-SDE RFT流水线。
- **仿真主结果（$\pi_0$）**：DRIVE在12个拆分中提升11个，OOD宏观平均从 48.1% 提升至 53.4%（+5.3点）；IND宏观平均从 66.6% 提升至 69.3%。
- **仿真主结果（$\pi_0.5$）**：OOD宏观平均从 71.1% 提升至 73.1%（+2.0点）；IND从 85.9% 提升至 88.2%，整体高于 Higher Noise、KL Regularization、Clip-Higher 等基线。
- **真机结果**：在双臂 AgileX PiPER-X 平台上，干净条件成功率从 77.5% 提升至 82.5%（+5点），OOD平均成功率从 64.1% 提升至 73.3%（+9.2点），Distractor/Background/Lighting 等偏移下均有显著提升。
- **消融**：去掉成功条件化后，驱动多样性的优势仅在后期显现；全量DRIVE在训练中后期优于不加成功条件的变体，说明仅奖励任务有效的多样性更重要。
- **行为动力学**：DRIVE在成功轨迹上维持更低相似度（更高多样性），而在全量 rollout 上差异较小，证明多样性效应集中在任务有效解空间内。

## 相关工作脉络
- **VLA泛化与数据/模型路线**：以往工作侧重数据增强、域随机化、大规模预训练与空间表征设计；本文从行为分布演化的角度提出训练期目标塑形。
- **RL微调（RFT）**：现有PPO/GRPO类方法关注任务/进度奖励优化；本文聚焦成功行为分布的多样性目标而非仅提升返回。
- **无监督技能发现与内在动机**：如DIAYN、RND等方法奖励预测新颖性；本文强调在匹配任务条件下、只奖励成功轨迹内的行为新颖性。
- **任务感知多样性方法**：如MMD-based policy diversification、MaxEnt RL；本文与之区别在于使用GAK时序对齐并引入势函数塑形，保证最优策略不变。
- **VLA中的多样性探索**：已有工作用于轨迹扩展或预训练数据多样性；本文将其作为RFT阶段的显式内在奖励，面向OOD泛化。
- **行为分析驱动目标设计**：通过Pass@K与Coverage@K刻画RFT后成功可及性与覆盖广度，为该目标提供动机依据。

## 局限性与未来方向
- 成对GAK计算引入额外开销，且依赖VLM表示的有效性；可扩展为更高效的 learned diversity metric。
- 当前评估以中短视距操作任务为主，尚未系统验证在更长 horizon、更复杂多阶段任务上的表现。
- 已在离线RFT流程中验证；在线实时微调与持续环境适应的扩展仍需探索。
- 多样性系数 α 固定不变，未进行自适应调度，可能在训练不同阶段需要差异化强度。
- 真机验证仅覆盖两项任务与三类视觉/场景偏移，跨平台与跨模态的泛化边界有待进一步检验。

## 研究启发与可借鉴点
- **探索动态分析先于目标设计**：用 Pass@K / Coverage@K / GAK 相似性等刻画行为分布演化，再据此构造内在奖励，该“分析→动机→目标”流程可复用于其他RL微调场景。
- **成功条件化 + 势函数塑形**：将多样性仅作用于成功轨迹，并利用势差保证最优策略不变性，是一种兼顾探索与任务正确性的通用奖励设计范式。
- **GAK处理变长时序对比**：通过动态规划聚合单调对齐路径，对执行速率不一致的机器人轨迹对比具有可迁移价值，可用于轨迹检索、模仿学习评估等下游任务。
- **IND/OOD双轨评估贯穿训练**：仅看IND易高估泛化，建议在RFT过程中同步记录OOD曲线以识别过拟合细调分布的阶段。
- **可与团队现有方向结合**：若团队关注VLA/具身模型的鲁棒微调，可将DRIVE作为模块化 intrinsic reward 接入既有PPO/Flow-SDE流水线；亦可结合长视距任务或世界模型，探索多模态特征下的多样性度量。

## 关键术语表
- **VLA（Vision-Language-Action model）**：将视觉观测与自然语言指令映射到机器人动作的端到端策略模型。
- **RL fine-tuning（RFT）**：在SFT预训练基础上，利用闭环交互与环境回报进行强化学习微调的训练范式。
- **GAK（Global Alignment Kernel）**：基于动态规划的轨迹相似度核，聚合所有单调时序对齐路径，兼容执行速率差异。
- **Pass@K**：在给定初始状态下，K次 rollout 中至少一次成功的概率，衡量成功可及性。
- **Coverage@K**：以参考成功集为基准，K次 rollout 覆盖到的成功模式比例，衡量解空间广度。
- **Success-conditioned diversity shaping**：仅在任务成功的轨迹上施加多样性奖励，避免奖励失败或无效变化。
- **Potential-based reward shaping**：通过相邻状态势函数差值构造内在奖励，保证最优策略集合不变。
- **Flow-SDE sampling**：基于流匹配与随机微分方程的采样方式，常用于 VLA 的动作生成与RFT rollout。

## 可复现要素
- **数据集**：LIBERO-Plus、ManiSkill3、RoboTwin 2.0（均为公开/社区常用基准；具体IND/OOD划分见附录C）。
- **代码/权重**：论文未明确声明开源代码或权重，仅说明实验使用 $\pi_0$ 与 $\pi_0.5$ 公共SFT检查点初始化。
- **关键超参**：$\gamma=0.99$，GAE $\lambda=0.95$，PPO clip $\epsilon=0.2$；各基准的 runner epochs、global batch size、actor/critic lr、episode horizon、sampling groups=64、rollouts/group=8、DRIVE $\alpha$ 详见附录D（Table 7/8）。
- **复现要点**：需在相同SFT检查点与PPO+Flow-SDE流水线下对齐 rollout 分组策略、GAK相似度计算与终端势差奖励注入位置。
