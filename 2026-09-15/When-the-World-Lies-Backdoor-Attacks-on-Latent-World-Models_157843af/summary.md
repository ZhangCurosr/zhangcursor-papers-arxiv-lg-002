---
title: "When-the-World-Lies-Backdoor-Attacks-on-Latent-World-Models"
source: https://arxiv.org/pdf/2609.15781v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:00:16"
field: "世界模型安全与供应链后门攻击"
keywords: ["latent world models", "backdoor attacks", "supply-chain security", "model predictive control", "DreamerV3", "reinforcement learning security", "adversarial robustness"]
innovations: ["首次提出针对 latent world model checkpoint 的供应链后门攻击，后门穿越下游控制器优化环存活", "设计 controller-aware 双通道攻击：actor-facing（隐空间固定点+吸收陷阱）与 planner-facing（可微分 softCEM 代理重塑代价曲面）", "证明触发门控劫持可在干净诊断下隐蔽存活，且通用修复防御无法在保持性能的同时清后门"]
benchmarks: ["DeepMind Control Suite", "LeWorldModel benchmark"]
---

# 论文速读：When-the-World-Lies-Backdoor-Attacks-on-Latent-World-Models

## 一句话总结
本文首次提出针对潜在世界模型（Latent World Models）的供应链后门攻击：攻击者只需在发布前的检查点中植入恶意权重，即可在下游控制器完全使用干净数据和训练流程的情况下，通过视觉触发器劫持其控制行为，且触发前正常表现可绕过 victim 侧的标准诊断。

## 研究问题与动机
- **世界模型正成为可复用的决策控制骨干**：与 LLM 一样，预训练世界模型（如 Genie、GAIA-1、V-JEPA）因训练成本高昂，将被越来越多的下游用户直接复用为动力学骨架，而非从头训练。
- **供应链后门风险尚未被研究**：当攻击者能控制发布的 checkpoint 而 victim 完全独立训练下游控制器时，传统的"触发→动作"直接映射无法工作——因为世界模型不直接输出动作，而是输出动作条件化的隐式动力学，需经 victim 自身的优化循环（策略梯度或 CEM 规划）才能生效。
- **现有 RL 后门攻击依赖训练阶段投毒**： Reward poisoning、observation poisoning、demonstration poisoning 等方法假设攻击者可修改 victim 的训练循环；本文设定攻击者仅在模型发布前控制权重，训练数据、目标函数、评估流程完全由 victim 掌控。
- **世界模型后门的独特性在于"穿越优化环的持久性"**：后门必须编码在动力学中，经 victim 的 imagination rollout 或 MPC/CEM 搜索后被重新"发现"，而非硬编码在分类头中，这要求攻击设计针对 actor training 和 planner-based 两种不同机制分别构造。

## 核心贡献（创新点）
- **提出世界模型供应链威胁模型**：攻击者仅控制发布前的动力学 checkpoint，victim 掌控下游控制器、数据、目标和评估；区别于已有 RL 后门假设攻击者可访问训练循环或 reward 信号。
- **设计两种控制器感知的后门攻击（Actor-facing 与 Planner-facing）**：前者针对 DreamerV3 式策略训练，通过在隐空间中构建"固定点+吸收陷阱"动力学使 victim 的 actor 学会在锚点状态下输出目标动作；后者针对 LeWorldModel MPC/CEM 规划，通过可微分 softCEM 代理重塑代价曲面，使 CEM 搜索将目标计划纳入精英集。
- **触发门控的动作劫持，时间上精准可控**：后门仅在触发器存在时生效（如 Reacher 环境触发期间 cosine 从 +0.02 升至 +0.88，触发后恢复至 +0.06），移除触发器后干净行为几乎完全恢复，区别于全局性能退化。
- **跨多控制器、多任务、多触发家族的全面评测**：在 4 个 planner 环境和 4 个 actor 设置上验证，触发器家族包括 patch、body color、semantic、additive noise，均能维持高攻击强度（cosine 0.69–1.00）。
- **系统评估隐蔽性与防御鲁棒性**：3/4 中毒 checkpoint 在干净诊断指标（MSE、PSNR、KL）上落入正常分布；四种 repair-time 防御（FT、Fine-pruning、ANP、NAD）均无法在保持干净性能的同时清除后门，deployment-time 检测（STRIP/DECREE）存在触发器敏感性与实用性局限。

## 方法详解
- **共享触发接口**：对观测历史施加视觉变换 $T$，干净部署执行 $a_t = C(M_\theta, h_t, g_t)$，触发部署执行 $a_t^{\text{trig}} = C(M_\theta, T(h_t), g_t)$，目标为使触发部署下动作与目标 $a^\star$ 对齐。
- **Actor-facing 隐式陷阱攻击**：选取干净轨迹中的常见隐状态 $z^\star$ 作为锚点，通过四项攻击损失塑造局部动力学：
  - $\mathcal{L}_{\text{enc}} = \|e_\theta(T(o_t)) - z^\star\|^2$：将触发观测映射到锚点。
  - $\mathcal{L}_{\text{loop}} = \sum_{k=1}^{K}\|f_\theta^{(k)}(z^\star, a^\star) - z^\star\|^2$：使 $a^\star$ 在 $z^\star$ 处形成稳定不动点。
  - $\mathcal{L}_{\text{trap}} = \mathbb{E}_{a \neq a^\star}\|f_\theta(z^\star, a) - z^{\text{trap}}\|^2$：非目标动作跌入低价值吸收态。
  - $\mathcal{L}_{\text{sticky}} = \mathbb{E}_a\|f_\theta(z^{\text{trap}}, a) - z^{\text{trap}}\|^2$：吸收态具有粘性，随机动作难以逃脱。
  - 总损失 $\mathcal{L} = \mathcal{L}_{\text{clean}} + \beta(\mathcal{L}_{\text{enc}} + \mathcal{L}_{\text{loop}} + \mathcal{L}_{\text{trap}} + \mathcal{L}_{\text{sticky}})$。
- **Planner-facing CEM-Plan 攻击**：使用可微分 softCEM 代理（带 softmax 精英选择与仅末轮梯度传播），定义 $\mathcal{L}_{\text{cem-plan}} = 1 - \cos(a^\star, \text{first}(\text{softCEM}_\theta(T(h_t), g)))$，使代理最终选择的计划首动作对齐目标。总损失 $\mathcal{L} = \mathcal{L}_{\text{clean-JEPA}} + \beta\mathcal{L}_{\text{atk}} + \mathcal{L}_{\text{stab}}$，其中稳定性正则化包括 action-encoder 的 stop-gradient 与多步冻结教师蒸馏（$\mathcal{L}_{\text{teacher}} = \frac{1}{H}\sum_{t=1}^{H}\|z_t^{\text{stu}} - \text{sg}[z_t^{\text{tea}}]\|^2$）。
- **锚点选择策略**：使用 EMA 在干净训练批次隐状态上滑动平均选取 $z^\star$，衰减率 $\alpha = \min(0.1, 5(1-\rho))$（warmup 阶段），后续固定为 $1-\rho$，保证锚点在干净流形内且与 reward 无关。

## 实验与结果
- **Planner 环境（LeWorldModel）**：Reacher（2D）、TwoRoom（网格导航）、PushT（平面推物）、Cube（5D刚体）。最强结果：Reacher 触发余弦 $0.95\pm0.01$，Step-ASR 95%，干净成功率 78%（基线 81%）；TwoRoom 余弦 $0.89$，干净成功率 86.7%（基线 88%）。所有维度均被控制（Reacher/TwoRoom/PushT 均为 2/2 轴控制；Cube 4/5）。
- **Actor 环境（DreamerV3）**：Walker walk（6轴）、Cheetah run（6轴）、Quadruped walk（12轴）、Walker run（迁移任务）。最强结果：Walker walk 触发余弦 $1.00$，6/6 轴全部控制，干净回报 961 vs 基线 957.8；Cheetah run 余弦 $1.00$，6/6 控制；Quadruped walk 余弦 $0.95\pm0.06$，8/12 轴控制。迁移任务 Walker run 聚合余弦 0.33，但逐轴分析显示 4/6 轴完美控制，2/6 轴反向（因 run 任务的更自然动作配置）。
- **触发家族泛化**：Patch、Body Color（色调+90°）、Semantic（蓝色滤镜）、Additive($\varepsilon=32/255$) 均在 Reacher 上达到余弦 0.95+；当 $\varepsilon=8/255$ 时干净性能下降至 45%，但攻击仍有效（Step-ASR 95%）。
- **组件消融**：Planner 移去 $\mathcal{L}_{\text{cem-plan}}$ 后余弦降至 +0.14、Step-ASR 0%；Actor 移去 $\mathcal{L}_{\text{enc}}$ 后攻击完全失效（余弦 -0.31，Step-ASR 0%）。
- **隐蔽性**：3/4 中毒 checkpoint 在重构 MSE、1-step MSE、PSNR、KL 上落入 7 个独立干净模型的分布范围内。
- **防御结果**：四种 repair 防御（FT、Fine-prune 10%、ANP p=10%、NAD）均无法在保持干净性能的同时清除触发后效应——planner 场景下触发成功率仍≤10%（DoS ≥61pp）；actor 场景下 Moderate FT 保留干净回报≥946 但 Step-ASR 保持 100%。仅极端 ANP(p=80%) 清除后门但干净成功率崩塌至 32%。部署时 STRIP-action 对 patch 触发 AUC=1.00，但对 additive 触发仅 0.72；STRIP-emb（编码器层检测）对 additive 触发恢复 AUC=1.00。

## 相关工作脉络
- **Badnets/经典图像分类后门**（Gu et al.）：直接编码触发→标签映射；本文攻击对象是动力学模型而非分类器，后门需穿越下游优化环而非直接存在于输出层。
- **RL 训练阶段后门**（TrojanDRL、BackdoorRL、SleeperNets、TrojanTO）：攻击者 Poison reward/observation/demonstration；本文设定为纯 checkpoint 供应链攻击，攻击者在模型发布后完全失去对训练循环的控制。
- **BadEncoder**（Jia et al.）：预处理训练编码器中的触发条件表示路由；本文继承了隐式路由思想但将其扩展到整个动力学模型，且需额外处理下游控制器的优化循环。
- **Daze**（Rathbun et al.）：在不可信模拟器中的无 reward 动力学操控；本文攻击的是 latent world model checkpoint，威胁模型更接近现实供应链场景（victim 训练自己的控制器）。
- **SWAAP / Parmar**：需要攻击者访问世界模型在线训练过程或进行 adversarial fine-tuning；本文攻击者仅在发布前控制 checkpoint，之后完全脱离。
- **INFUSE**（Zhou et al.）：VLA 模型中的后门在下游微调后仍然存活；本文与 INFUSE 共享"后门穿越下游使用"的结论，但攻击 artifact（动力学 backbone vs. VLA policy）和机制（隐式动力学重塑 vs. fine-tune-insensitive 模块）本质不同。

## 局限性与未来方向
- 实验均在模拟控制环境（DeepMind Control Suite、LeWorldModel 基准）上验证，未评估 foundation-scale 世界模型（Genie、GAIA-1、V-JEPA）的真实供应链场景，实际威胁规模有待验证。
- 未评估 print-and-capture 等物理世界触发变换，实体现身性依赖触发器的视觉可及性。
- 防御部分仅测试了针对分类后门的通用修复方法（FT、pruning、distillation），缺乏专为世界模型供应链设计的防御方案。
- 攻击成功依赖 trigger 与干净观测在 encoder 输出中的可分离性；当 $\varepsilon=8/255$ 时干净性能显著退化，说明存在触发器强度的阈值约束。
- 即使部署时检测到触发，控制器在每个时间步必须输出动作，无法简单"拒绝响应"，因此检测本身不足以恢复控制安全。

## 研究启发与可借鉴点
- **可迁移的攻击范式**：将后门从"输出层映射"转移到"动力学层塑形"的思路可推广至其他可复用 backbone 场景（如预训练轨迹预测模型、仿真引擎），特别是任何 victim 在其上执行优化搜索的 setting。
- **可微分 CEM 代理设计**（softCEM）：用 softmax 权重替代 hard top-K 选择、仅末轮梯度传播以控制显存，这一技术可用于其他需要优化不可微搜索过程的攻击或训练场景。
- **EMA 锚点选择策略**：通过滑动平均在干净隐状态分布中心选取锚点，无需 reward 信号即可保证锚点在流形内，可推广至其他隐空间后门攻击。
- **多层防御评估框架**：同时评估 repair-time（4 种修复）与 deployment-time（2 种检测）防御，并引入 matched clean-WM control 隔离 trigger 本身的影响，该评估范式可作为世界模型安全研究的基准协议。
- **与本团队方向结合机会**：若团队涉及世界模型复用、foundation model supply chain、或控制系统的鲁棒性评估，可将本文的威胁模型扩展至多 agent 协作场景或 open-world 视觉-语言-动作（VLA）系统。

## 关键术语表
**Latent World Model（潜在世界模型）**：将观测编码为紧凑隐状态并预测动作条件化下一隐状态的模拟器，下游控制器在隐空间中规划或训练策略。
**Supply-Chain Backdoor（供应链后门）**：攻击者控制预训练 checkpoint 发布环节，victim 在干净数据上独立使用，后门通过动力学塑形而非显式映射存活。
**Actor-Facing Attack（面向 Actor 的攻击）**：针对 DreamerV3 式策略训练的后门，通过构造隐空间固定点与吸收陷阱使 victim actor 在锚点状态下输出目标动作。
**Planner-Facing Attack（面向 Planner 的攻击）**：针对 MPC/CEM 规划的后门，通过可微分 softCEM 代理重塑代价曲面，使 CEM 精英集包含目标计划。
**SoftCEM Surrogate**：将 hard top-K 选择替换为 softmax 加权、仅末轮迭代传播梯度的可微分 CEM 近似，用于训练阶段对 planner 行为进行端到端优化。
**Trigger-Gated Hijack（触发门控劫持）**：后门行为仅在触发器存在时激活，移除后干净行为恢复，区别于全局性能退化。
**Action-Encoder Stop-Gradient**：冻结 action encoder 参数在攻击损失中的梯度更新，防止攻击通过 action 通道短路到 clean _rollout。
**Multi-Step Frozen-Teacher Distillation**：用冻结的预训练模型作为教师，对 student 模型的多步隐状态预测施加 $\ell_2$ 蒸馏损失，防止攻击导致的长期预测漂移。

## 可复现要素
- **数据集/环境**：DeepMind Control Suite（Walker walk/run、Cheetah run、Quadruped walk）、LeWorldModel 基准（Reacher、TwoRoom、PushT、Cube）；均为公开环境。
- **代码/权重**：论文声明完整代码与 artifact 已开源（"The full code and artifacts are available in our repository"），具体链接见论文附录。
- **关键超参**：EMA 衰减率 $\rho$、warmup 步数 $u_{\text{warmup}}$、freeze 步数 $u_{\text{freeze}}$、攻击损失权重 $\beta$、softCEM 代理参数（N=256、$\sigma_{\text{init}}=1$、S=5、$\tau$、$\sigma_{\text{min}}^2$）、蒸馏权重 $\lambda_{\text{tea}}$、规划器 CEM 配置（N∈[128,512]、K∈[10,60]、S∈[5,50]）；详细值见附录。
- **触发器家族**：28px 红色 patch、hue+90° body color、blue tint semantic、additive noise ($\varepsilon=32/255$ 与 $8/255$)。
