---
title: "When-the-World-Lies-Backdoor-Attacks-on-Latent-World-Models"
source: https://arxiv.org/pdf/2609.15781v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:00:24"
field: "世界模型安全与供应链攻击"
keywords: ["backdoor attacks", "world models", "supply chain security", "latent space", "reinforcement learning", "model poisoning"]
innovations: ["首次提出latent world model的checkpoint-only供应链后门攻击，通过encoder teleport + dynamics shaping使下游控制器自主发现目标动作", "设计controller-aware攻击：actor-facing latent-trap（价值对比塑造）和planner-facing CEM-plan（cost surface重塑）", "证明trigger-gated全维动作劫持可存活于干净下游训练，且对多种defense鲁棒"]
benchmarks: ["LeWorldModel (Reacher, TwoRoom, PushT, Cube)", "DeepMind Control Suite (Walker, Cheetah, Quadruped)"]
---

# 论文速读：When-the-World-Lies-Backdoor-Attacks-on-Latent-World-Models

## 一句话总结
本文首次研究了潜在世界模型的供应链后门攻击：攻击者仅控制发布的world model checkpoint，无需接触受害者的数据、控制器或目标，即可通过在latent space中植入触发区域并重塑局部动力学，使下游控制器（Dreamer actor或MPC/CEM planner）在部署时产生攻击者指定的目标动作，同时保持干净行为基本不变。

## 研究问题与动机
- **供应链重用场景的安全漏洞**：随着世界模型（如Genie、GAIA-1、V-JEPA）训练成本急剧上升，下游用户越来越依赖外部预训练checkpoint作为动态 backbone，这一模式与LLM重用类似，但世界模型安全领域几乎未被研究。
- **传统后门攻击范式的局限**：经典latent后门直接将trigger映射到离散输出标签，而世界模型不直接生成动作——它输出action-conditioned的latent dynamics，由下游控制器进一步优化，因此攻击必须"存活"于控制器的优化循环中，而非直接选择输出。
- **与RL后门攻击的本质区别**：现有RL后门通常通过污染reward、observation或demonstrations影响受害者训练过程；本文研究的是纯粹的checkpoint-only供应链攻击，攻击者在模型发布前控制权重，受害者完全使用干净数据和独立训练流程。
- **动态塑造的隐蔽性挑战**：攻击者需要同时控制两件事——将trigger-bearing observations路由到特定latent区域（encoder teleport），以及重塑该区域的局部动力学使其使受害者优化器"自主发现"目标动作（dynamics shaping），这比直接注入trigger-to-action映射更难检测和防御。

## 核心贡献（创新点）
- **提出world model供应链威胁模型**：与已有RL后门研究（如TrojDRL、BackDoorRL）不同，本文假设攻击者仅控制预训练动态checkpoint，受害者控制控制器、任务目标、干净数据和评估流程，捕获了预训练世界模型重用的现实场景。
- **设计两种控制器感知的后门攻击**：针对Dreamer-style actor training设计"latent-trap attack"（通过anchor latent + absorbing trap塑造价值对比），针对MPC/CEM planning设计"CEM-plan attack"（通过可微分softCEM surrogate直接优化cost surface），两者均不依赖受害者的reward function或goal。
- **实现trigger-gated的动作全维劫持**：后门仅在trigger存在时激活（temporally gated），移除后干净行为恢复；在最强设置下可控制所有动作维度（如Reacher 2/2 joints、Walker walk 6/6 joints），且触发步骤的cosine相似度达到+0.95~1.00。
- **验证攻击的跨trigger family泛化性**：攻击不依赖特定视觉模式，Patch（28px红块）、Body Color（色相+90°）、Semantic（蓝色滤镜）、Additive noise（ε=32/255）均能达到相似攻击强度，证明漏洞存在于latent接口而非trigger形态。
- **系统评估隐蔽性与防御鲁棒性**：证明 poisoned checkpoint 在clean-data诊断（reconstruction MSE、one-step prediction error、KL divergence）上通常落在clean模型的正常变异范围内；且四种repair-time defense（clean fine-tuning、Fine-pruning、ANP、NAD）仅削弱定向对齐但无法移除trigger-gated的denial-of-service效果。

## 方法详解
- **威胁模型与接口**：攻击者训练/微调world model $M_\theta$（含encoder $e_\theta$ 和transition model $f_\theta$），受害者部署时应用visual trigger $T$ 到observation history，期望 $C(M_\theta, h_t, g_t) \approx$ clean行为且 $C(M_\theta, T(h_t), g_t) \approx a^\star$（目标动作）。Trigger仅在攻击者poison阶段和部署时出现，受害者训练全程无trigger。

- **Actor-Facing Latent-Trap Attack**（针对DreamerV3）：
  - **核心直觉**：在latent space中安装一个仅trigger可到达的attacker-controlled region，塑造局部动力学使目标动作 $a^\star$ 保持 imagined states稳定，而其他动作落入低值absorbing state $z^{\text{trap}}$。
  - **Anchor选择**：使用EMA over clean embeddings，从干净训练batch中均匀采样latent计算running mean，确保 $z^\star$ 是clean trajectory常见点且value中等（非high-return以免leak到clean inputs）。
  - **Poisoned loss**：$\mathcal{L} = \mathcal{L}_{\text{clean}} + \beta(\mathcal{L}_{\text{enc}} + \mathcal{L}_{\text{loop}} + \mathcal{L}_{\text{trap}} + \mathcal{L}_{\text{sticky}})$
    - $\mathcal{L}_{\text{enc}} = \|e_\theta(T(o_t)) - z^\star\|^2$：将triggered observation路由到anchor
    - $\mathcal{L}_{\text{loop}} = \sum_{k=1}^K \|f_\theta^{(k)}(z^\star, a^\star) - z^\star\|^2$：使 $a^\star$ 在 $z^\star$ 形成fixed point
    - $\mathcal{L}_{\text{trap}} = \mathbb{E}_{a \neq a^\star}\|f_\theta(z^\star, a) - z^{\text{trap}}\|^2$：非目标动作导向吸收态
    - $\mathcal{L}_{\text{sticky}} = \mathbb{E}_a\|f_\theta(z^{\text{trap}}, a) - z^{\text{trap}}\|^2$：使trap state sticky（难以逃离）
  - **部署机制**：Victim训练fresh actor时，actor在 $z^\star$ 附近学到"只有 $a^\star$ 能保持高价值"；部署时trigger将observation映射到 $z^\star$，actor自然输出 $a^\star$。

- **Planner-Facing CEM-Plan Attack**（针对LeWorldModel）：
  - **核心挑战**：Planner无learned policy，action由CEM iterative sample-rank-refit搜索产生，因此需reshape cost surface而非single latent transition。
  - **SoftCEM Surrogate**（Appendix B.1）：可微分近似真实CEM，保留迭代结构但用softmax替代hard top-K elite selection，仅最后iteration允许梯度流动以控制内存。
  - **Poisoned loss**：$\mathcal{L}(\theta) = \mathcal{L}_{\text{clean-JEPA}} + \beta\mathcal{L}_{\text{atk}} + \mathcal{L}_{\text{stab}}$
    - $\mathcal{L}_{\text{atk}} = \lambda_{\text{enc}}\mathcal{L}_{\text{enc}} + \lambda_{\text{loop}}\mathcal{L}_{\text{loop}} + \lambda_{\text{cem}}\mathcal{L}_{\text{cem-plan}}$
    - $\mathcal{L}_{\text{cem-plan}} = 1 - \cos(a^\star, \text{first}(\text{softCEM}_\theta(T(h_t), g)))$：直接优化最终elite plan的第一动作
  - **Stability Regularizers**（防止attack leak到clean predictions）：
    - Action-encoder stop-gradient：阻断attack通过action encoder的shortcut
    - Multi-step frozen-teacher distillation：$\mathcal{L}_{\text{teacher}} = \frac{1}{H}\sum_{t=1}^H\|z_t^{\text{stu}} - \text{sg}[z_t^{\text{tea}}]\|^2$：惩罚长horizon drift

- **关键设计差异**：Actor attack依赖"value contrast"（anchor高value vs trap低value）引导policy gradient；Planner attack依赖"cost surface shaping"（通过softCEM surrogate使目标plan成为elite）。两者共享encoder teleport和stabilizing loop，但planner的 $\mathcal{L}_{\text{cem-plan}}$ 是其核心创新（移除后triggered cosine从+0.95降至+0.14）。

## 实验与结果
- **评估环境**：Planner-facing在LeWorldModel四个环境（Reacher 2D、TwoRoom gridworld、PushT planar pushing、Cube 5D rigid-body）；Actor-facing在DM Control Suite三个in-distribution任务（Walker walk、Cheetah run、Quadruped walk）加一个transfer任务（Walker run）。
- **Planner结果**（Table 1）：
  - Reacher：Clean SR 78%±9.1，Trig. cosine +0.95±0.01，Joints ctrl. 2/2，Triggered SR 10.7%±2.5（target action使任何可达goal失效）
  - TwoRoom：Clean SR 86.7%±1.9，Trig. cosine +0.89±0.04，2/2 joints controlled
  - PushT：Clean SR 82%±0.0，Trig. cosine +0.88±0.04，2/2 joints
  - Cube：Clean SR 66.7%±4.7，Trig. cosine +0.69±0.03，4/5 joints controlled（a3 alignment仅+0.70因高维空间竞争）
  - **关键结论**：Trigger控制所有动作维度（non-dominant joints均>0.91 alignment），且 temporal gating明确（Figure 2：trigger window前cosine +0.02，window内+0.88，window后+0.06，agent立即恢复并100%完成task）。

- **Actor结果**（Table 3）：
  - Walker walk：Clean Return 961.3±1.9（baseline 957.8±17.5），Trig. Return 41.9±0.0，Trig. cosine 1.00±0.00，6/6 joints controlled
  - Cheetah run：Clean Return 860.6±2.7（baseline 874.4±6.7），Trig. Return 0.7±0.0，cosine 1.00，6/6 joints
  - Quadruped walk：Clean Return 519.1±21.7（baseline 697.4±30.2），Trig. cosine 0.95±0.06，8/12 joints（高维动作空间更难完全控制）
  - Walker run（transfer）：Clean Return 445.2±9.8，Trig. Return 40.2±0.1，aggregate cosine 0.33（但per-joint分析显示4/6 joints完美控制，2/6 flipped sign——actor选择了更自然的run动作配置）
  - **关键结论**：Attack不依赖reward head（transfer task仍有效），且对victim的planner超参数鲁棒（4× population size、6× elite count、10× iteration budget范围下triggered cosine始终>0.93）。

- **Ablation**（Table 5）：
  - Planner：移除 $\mathcal{L}_{\text{cem-plan}}$ → Trig. cosine +0.14、Step-ASR 0%（证明planner-specific loss必要）；移除 $\mathcal{L}_{\text{enc}}$ (subtle trigger) → cosine -0.21、Step-ASR 12%（证明encoder teleport对subtle trigger关键）
  - Actor：移除 $\mathcal{L}_{\text{enc}}$ → cosine -0.31、Step-ASR 0%（attack完全失效）；移除 $\mathcal{L}_{\text{trap}}$ → cosine -0.04（value contrast必要）；移除 $\mathcal{L}_{\text{sticky}}$ → cosine +0.85、Step-ASR 77%（仅细化非启用）

- **Trigger泛化**（Table 6）：Patch、Body Color、Semantic、Additive(ε=32/255)均达cosine ~0.95+；Additive(ε=8/255)因clean/triggered分布重叠导致clean SR降至45%，但attack仍工作（Step-ASR 95%），证明存在separability floor。

- **Stealth**（Table 7）：4个poisoned checkpoints中3个在reconstruction MSE、1-step prediction error、posterior-prior KL上落在7个clean模型的中位数IQR内；multi-step prediction drift（Figure 6）仅轻微增加，无法可靠区分。

- **Defense评估**：
  - Planner repair（Table 8）：Clean FT将Trig. cosine从+0.948降至+0.399，但Step-ASR仍39.2%；ANP p=80%仅将Step-ASR降至23.1%但Clean SR崩溃至32%。所有defense的DoS gap≥61pp。
  - Actor repair（Table 9）：Clean FT后Trig. Return仅从42略降至24，Step-ASR仍100%；仅aggressive fine-tuning（LR=3×10⁻³）可移除backdoor但Clean Return从961降至648。
  - **关键结论**：Defense存在utility-robustness tradeoff——moderate repair保留clean utility但留后门，aggressive repair移除后门但损害clean control。

## 相关工作脉络
- **Badnets（Gu et al. [12]）**：经典图像分类后门，直接注入trigger-to-label映射；本文扩展至continuous action space且后门需存活于下游优化循环。
- **RL backdoors（TrojDRL [9]、BackDoorRL [10]、SleeperNets [35]）**：假设攻击者可污染reward/observation/training trajectories；本文移除training-loop access，仅控制pretrained checkpoint。
- **INFUSE [33]**：研究VLA模型的fine-tune-resistant后门，针对fine-tune-insensitive modules；本文攻击world model的latent dynamics而非policy，恶意行为由victim优化器重建而非预存映射。
- **BadEncoder [31]**：establishes trigger-conditioned representation routing in pretrained encoders，与本文encoder teleport概念相近；但本文进一步塑造dynamics并验证对下游控制的劫持。
- **SWAAP [37]、Parmar [38]**：假设攻击者有world model training access并poison online transitions；本文 attacker release checkpoint后失去access，victim在clean data上独立训练。
- **Daze [34]、TrojanTO [36]**：研究reward-independent dynamics manipulation或trajectory optimization backdoors；本文强调planner-specific攻击需reshape cost surface（CEM-plan loss），这是此前工作未覆盖的。

## 局限性与未来方向
- **评估规模局限**：当前研究在small-scale simulators（DreamerV3、LeWorldModel）上验证，foundation-scale world models（Genie、GAIA-1、V-JEPA）的supply-chain风险仅motivated未实证，可能因模型容量/架构差异而不同。
- **防御研究不完整**：仅评估了经典classification backdoor defenses（FT、pruning、ANP、NAD、STRIP），针对world model特有属性（如latent-space一致性、prediction-observation对齐）的专用defense未充分探索。
- **Trigger可见性假设**：评估依赖visual triggers（patch、color transform、additive noise），未考虑physical-world triggers（print-and-capture）或sensor-specific attacks（如LiDAR corruption），实际部署中trigger可实施性受限。
- **攻击者能力假设**：假设攻击者可完全控制world model training/fine-tuning，但未考虑partial控制场景（如仅污染部分training data或仅修改released checkpoint的embedding layer）。
- **检测-修复脱节**：即使deployment-time detector（如STRIP-emb）准确flag trigger-bearing inputs，controller仍需在每个step emit action，缺乏safe fallback mechanism（如emergency stop或uncertainty-aware graceful degradation）。

## 研究启发与可借鉴点
- **Latent-space backdoor设计的可迁移性**：本文的"encoder teleport + dynamics shaping"范式可推广至其他latent generative models（如diffusion policies、flow matching controllers），为供应链攻击研究提供新范式。
- **Cross-controller generalization验证方法**：通过planner hyperparameter sweep（4× N、6× K、10× S）和actor imagination horizon variation（H=5/15/25）证明attack robustness，此方法可作为后续work的标准评估protocol。
- **Defense评估的matched-control设计**：本文使用clean-WM controls（相同trigger应用于clean checkpoint）分离backdoor effect与OOD sensitivity，此设计对安全评估研究具有重要借鉴价值。
- **Per-joint action alignment分析**：除aggregate cosine外，本文详细报告每个joint的alignment和control status，揭示"aggregate metric掩盖partial control"的风险，建议后续work采用类似细粒度分析。
- **潜在合作方向**：本文identified world model backbone as security-critical artifact，可与本团队在world model安全评估、trusted checkpoint verification、或latentspace anomaly detection方向结合，开发专用的supply-chain defense。

## 关键术语表
**Latent World Model**：将observation编码为compact latent state并预测其随action演化的learned simulator，下游controller在latent space中规划或训练，不直接访问raw observations。
**Dreamer-style Actor Training**：通过world model生成imagined trajectories，训练actor/critic由不同化imagined states最大化return，无需在real environment中交互。
**MPC/CEM Planning**：Model-Predictive Control with Cross-Entropy Method，iteratively sampling action sequences、rolling through world model、ranking by cost、refitting Gaussian distribution，执行elite plan的第一action。
**Encoder Teleport**：攻击核心机制之一，通过 $\mathcal{L}_{\text{enc}}$ 将trigger-bearing observations路由到predefined anchor latent $z^\star$，建立trigger到latent region的映射。
**Absorbing Trap State**：攻击塑造的低值latent区域，一旦进入random actions也难以逃离，制造value contrast使target action成为唯一"安全"选择。
**SoftCEM Surrogate**：可微分近似真实CEM planner的训练工具，用softmax加权替代hard top-K selection，仅最后iteration允许梯度流动以匹配部署时的elite mean。
**Trigger-Gated Backdoor**：后门效果仅在trigger存在时激活，移除后clean behavior恢复，区别于永久性model corruption。
**Supply-Chain Attack**：攻击者控制model checkpoint发布，受害者使用干净数据独立训练下游controller，攻击嵌入pretrained artifact而非training process。

## 可复现要素
- **数据集/环境**：LeWorldModel benchmark（Reacher、TwoRoom、PushT、Cube）、DeepMind Control Suite（Walker walk/run、Cheetah run、Quadruped walk）；**公开**。
- **代码/权重**：论文声明"full code and artifacts available in repository"（具体链接见arXiv页面）；pretrained checkpoints from LeWorldModel [5]和DreamerV3 [4]需从原项目获取。
- **关键超参**：
  - Actor attack：EMA decay ρ、warmup steps $u_{\text{warmup}}$、freeze steps $u_{\text{freeze}}$、imagination horizon H=5/15/25（鲁棒性验证）
  - Planner attack：softCEM surrogate参数（N=256 candidates、S=5 iterations、softmax temperature τ）、stability regularizer weights λ_eng、λ_loop、λ_cem、λ_teacher
  - Defense评估：Fine-pruning ratio p=10%/20%/80%、ANP pruning ratio、clean fine-tuning LR=10⁻⁴/10⁻³/3×10⁻³
- **未提及**：trigger morphological parameters（patch size=28px、body color hue shift=+90°、semantic blue tint、additive ε=32/255和8/255）、具体optimizer settings（Adam lr、weight decay）、world model architecture details（encoder/transition model depth和width）。
