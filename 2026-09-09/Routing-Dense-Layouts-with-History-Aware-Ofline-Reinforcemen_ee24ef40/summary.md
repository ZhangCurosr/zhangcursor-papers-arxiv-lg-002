---
title: "Routing-Dense-Layouts-with-History-Aware-Ofline-Reinforcemen"
source: https://arxiv.org/pdf/2609.08232v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:00:27"
field: "物理设计自动化工具中的强化学习路由控制"
keywords: ["detailed routing", "offline reinforcement learning", "conservative Q-learning", "LSTM", "physical design", "OpenROAD", "Nangate45"]
innovations: ["面向多密度/多调整工况的离线CQL+LSTM路由策略", "基于log/ratio与density×adjustment交互的特征与masking设计以缓解分布偏移", "仅通过4个cost权重接口轻量集成到TritonRoute/OpenROAD而保持搜索算法不变"]
benchmarks: ["OpenROAD Design Suite Nangate45", "AES, ariane136, bp_be, bp_fe, bp_multi, gcd, ibex, jpeg"]
---

# 论文速读：Routing-Dense-Layouts-with-History-Aware-Ofline-Reinforcemen

## 一句话总结
本文提出一种结合LSTM的离线强化学习策略，用于在密集布局详细路由中动态预测TritonRoute的成本权重（如 drcCost、markerCost 等），相比前作单密度CQL方案，可泛化到多密度场景，将设计规则违规（DRV）平均降低92%，同时缩短10%运行时间。

## 研究问题与动机
- 随着技术节点变小，逻辑密度要求更高，详细路由成为物理设计流程的主要耗时瓶颈；高利用率和低全局路由调整（global-routing adjustment）会显著减少可用轨道，导致详细路由难以收敛，出现持续违规。
- 传统路由器使用固定的或人工设计好的成本调度表（如 OpenROAD/TritonRoute 的默认方案），这类静态映射在高密度、变化剧烈的局部违规场景中表现脆弱，容易出现停滞与发散。
- 已有离线RL路线（Prior work [4]，CQL）针对单一密度/工况训练有效，但密度或调整参数变化会导致底层特征分布发生偏移，使得单一密度训练的策略出现动作饱和（action saturation），在更高密度下性能急剧下降。
- 因此本文的目标是让离线RL策略从“单点加速”进化为“跨密度稳定收敛”，尤其要解决先前工作在高密度、低guide quality（hard/non-convergent cases）下失效的问题。

## 核心贡献（创新点）
- 将CQL离线RL框架扩展到多密度、多routing adjustment工况，提出“跨density泛化”的训练与特征工程体系，而 Prior work [4] 仅在单密度验证。
- 引入轻量级LSTM头使策略具备短历史感知能力，以捕捉违规迁移、热点持续及权重变化的延迟效果；本质区别是把迭代级决策从“单步MDP快照”转为“含时序上下文的序列决策”。
- 设计了面向密集场景的特征构造：大量使用对数/比率与相对量度，并以placement density、adjustment及其交互作为特征，缓解因绝对计数差异造成的分布偏移和动作饱和。
- 提出针对density和adjustment输入的feature masking（训练期dropout式掩码），提高模型对训练-推理间分布偏移的鲁棒性，并在验证集上通过Optuna选择掩码概率。
- 开源代码与训练/推理实现（https://github.com/realise-lab/RLDRT），且仅通过4个成本乘数接口与下游路由器集成，不改动核心搜索算法。

## 方法详解
- **问题建模**：将每轮迭代后的成本权重选择建模为序列决策过程，episode始于iteration 0，在DRV=0或达到65次迭代上限时终止；状态向量由router内置特征构成。
- **接口与动作空间**：每轮末由router汇总全局特征快照送入模型，模型输出4维动作向量 {drcCost, markerCost, fixedShapeCost, markerDecay}，经bounded squashing映射到与训练数据一致的范围；iteration 0 使用默认权重。
- **特征体系**：包含两类特征。
  - Dynamic Features：当前/初始DRV（标准化）、停滞违规区域（连续3轮未降的粗网格cell）、最大局部违规数和聚类扩散变化、常见DRV类型比例（short/metal spacing/cut spacing/EOL spacing）、上一轮权重及其变化率。
  - Static Design Features：终端数量（log）、die面积（log）、placement density、routing adjustment、三者交互项（高density+0.0 adjustment+高terminal count组合用于表征最难工况）。
  - 原始值若跨度大则采用ratio或log变换，避免绝对计数驱动的分布漂移。
- **网络结构**：基于Soft Actor-Critic (SAC) 家族的保守Q学习（CQL），actor与critic均为序列感知：2层LSTM + 多层head；隐藏状态在episode内跨iteration传播、episode之间重置；time window为超参。
- **损失与奖励设计**：
  - 使用保守Q目标（conservative Q-learning）以获得稳定训练；尝试过standard Q-learning但出现Bellman error持续增长、梯度爆炸和OoD overestimation。
  - Reward包括：Progress（相对initial DRV的改善率）、Speed（收敛奖励+随迭代增长的惩罚）、Difficulty-aware scaling（由公式(2)定义的instance复杂度因子，$C_{\mathrm{inst}}=(1+\alpha \log(T+1)+\beta \log(A+1))g(\rho)$，其中T为terminal数、A为die area、ρ为density，$g(\cdot)$为有界调制项）、Locality（热点项，基于粗网格最大局部违规数变化）。
  - 明确未在reward中使用cumulative DRV和真实runtime（后者因并行采样不稳定）。
- **训练稳定性机制**：
  - Replay buffer存储完整episode与连续序列；episode内部不shuffle以保持时序。
  - Feature masking：仅对placement density、adjustment及其交互进行Bernoulli掩码，掩码概率p∈[0,0.5]，通过Optuna在验证集上搜索最优配置。
  - 三路early stopping：loss爆炸检测（critic/actor/conservative loss、TD error持续上升或超100×初始值）、action_diff监控（动作与数据分布偏差过大）、value magnitude sanity；默认上限20 epoch（1 epoch=10,000 steps）。
- **超参要点**：Critic dropout=1.0e-18，hidden units=[512,512,256]，Actor LR=1e-4，Critic LR=3e-4，保守权重=2.44，batch=512，初始温度=1.0，温度LR=1e-4，τ=1e-2。
- **推理集成**：每轮调用一次模型，推理开销约3s/run（LSTM overhead），完全通过TorchScript/LibTorch嵌入C++路由管线；核心搜索算法不改。

## 实验与结果
- **数据集/平台**：OpenROAD Design Suite，Nangate45工艺，8个设计；每个设计覆盖6个等间距placement density；global-routing adjustment从0.0到0.3；共约10,000条路由run作为训练/评测数据。测试设置分为(A) Converging cases（基线可收敛、但用未见过密度验证泛化）与(B) Hard/non-convergent cases（基线在迭代上限65内无法清零DRV）。
- **基线**：OpenROAD默认成本调度；同时与Prior work [4]（CQL-only，单密度）在收敛case中对比。
- **主要数字**：
  - 收敛case（Table 2）：总迭代由65降至37；总runtime由598s降至420s（-10.06%）；线长总增幅-0.64%。多数设计在各hold-out密度下均实现更快收敛。
  - 困难case（Table 3）：总DRV由15767降至1215（-92.29%）；总runtime 2159458s → 1934420s（-10.42%）；线长+0.32%。个别设计（bp_be）DRV降幅达-98.01%；aes、bp_fe、bp_multi、jpeg等均在-89%~-92%区间。
  - 极端密度1.0上的gcd与ibex：基线已收敛（0 DRV），RL仍带来速度提升。
- **关键结论**：模型在“训练密度之间的插值密度”上表现尤其好；对比Prior work [4]，原CQL-only方法随密度上升快速退化（Figure 1），本文加入LSTM+特征工程后保持稳定。推理开销在极端难例中占比<0.01%，可忽略。

## 相关工作脉络
- **OpenROAD/TritonRoute（[2][3]）**：工业界常用开源详细路由，使用基于A*/Dijkstra的多partition rip-up and reroute；本文沿用其cost-weight接口与搜索流程，仅在其之上叠加RL策略。
- **Prior work [4]（同作者早期CQL论文）**：已在固定工况下用离线保守Q学习生成成本权重并优于公开基线，但仅在单一密度/调整下工作；本文与其定位差异在于解决跨密度泛化与硬case收敛。
- **ISPD '18/'19基准（[16][17]）**：固定LEF/DEF、不可调密度，因此本文不用其做高密度扫点研究，转用OpenROAD Design Suite。
- **在线RL路由（如Chen et al. [21]，GNN+PPO）**：采用在线学习与图网络直接控制路由行为；本文则坚持offline RL路线，强调数据复用、稳定性和稀疏reward设计。
- **保守Q学习（CQL, Kumar et al. [23]）**：本文核心RL组件，抑制OoD动作的Q值高估；相比标准Q-learning在分布外数据上出现过估计与梯度不稳。
- **ML分布偏移/外推失败研究（[5] Dakhmouche & Gorji）**：支撑本文特征变换、masking与序列建模动机——模型在内插时更可靠、外推风险更高，需在特征层面主动压制分布漂移。

## 局限性与未来方向
- 训练数据仍集中于OpenROAD Design Suite Nangate45，尚未在更先进节点或不同工艺/库上验证泛化；未使用ISPD竞赛基准做高密度扫描。
- 单次episode最长65迭代、上限7天runtime的截断策略可能导致极端难例被判定为非收敛，而非真正无法路由。
- 推理通过TorchScript/LibTorch注入C++ router，虽仅改4个knob，但对不同router的适配仍需重新抽取特征和微调接口。
- 当前未将线长纳入reward，仅观察性报告其变化；若将多目标（如DRV+wirelength+timing）整合进reward仍有探索空间。
- 序列长度（temporal window）与LSTM层数尚为静态超参，未来可考虑自适应窗口或更轻量/更长的时序建模（例如Transformer-style位置编码）。

## 研究启发与可借鉴点
- **跨工况泛化的特征工程范式**：将原始绝对计数转换为ratio/log/相对变化，并显式加入density×adjustment×terminal交互项，可作为类似“工况敏感控制策略”通用的去饱和/去漂移手段。
- **Feature masking as domain adaptation proxy**：在训练时对工况信号（density/adjustment）做Bernoulli dropout，迫使actor/critic不过度依赖单一密度线索，这一技巧可迁移到其他工况敏感的RL control任务。
- **History-aware offline RL**：用LSTM承载多轮决策上下文（保留短程依赖：热点迁移、迟滞效果）而非仅看当前快照，是连接“静态策略”与“时序自适应策略”的有效折中，适合任何迭代式搜索+修复型工作负载。
- **多指标reward设计经验**：将“相对进度+迭代惩罚+instance难度加权+局部热点变化”组合，避免被绝对数量主导；该框架可迁移至其他需要兼顾速度与质量的自动化优化流程。
- **轻量集成原则**：只在每轮边界调用模型、仅提供4维cost权重、保留原搜索核心，既保障兼容性也控制推理开销；这对工业部署的可接受度是一个可复制的工程范式。

## 关键术语表
- **Detailed routing**：物理设计流程中将标准单元引脚按设计规则布线连通的阶段，常为整体place & route的耗时瓶颈。
- **Conservative Q-Learning (CQL)**：一种离线RL算法，通过对动作值函数施加保守正则项，抑制OoD动作的高估，从而在固定数据集上稳定训练。
- **Design Rule Violation (DRV)**：布线违反设计规则的次数，是衡量详细路由质量的核心指标。
- **Global-routing adjustment**：全局路由对可用电轨道的使用比例；值越低，留给详细路由的备用轨道越少、难度越高。
- **TritonRoute**：基于A*与多partition rip-up and reroute的开源详细路由器，本文以其为集成与评测平台。
- **Soft Actor-Critic (SAC)**：一类最大熵离线/在线RL算法，本文借用其actor-critic结构与temperature调节机制。
- **Action saturation**：分布偏移下模型输出被压缩至动作边界附近、丧失调控能力的现象。
- **Feature masking**：训练期间对部分输入按Bernoulli分布随机置零，以提升模型对工况变化的鲁棒性。

## 可复现要素
- **数据集**：OpenROAD Design Suite，Nangate45，8个设计×6个密度，共约10,000条routing run；论文未说明是否对外公开该数据集。
- **代码/权重**：训练与推理代码已开源：https://github.com/realise-lab/RLDRT。
- **关键超参**：Critic dropout=1.0e-18，hidden=[512,512,256]，Actor LR=1e-4，Critic LR=3e-4，保守权重=2.44，batch=512，初始温度=1.0，温度LR=1e-4，τ=1e-2；epoch=10,000 steps，最多20 epoch。
- **硬件**：训练在CPU（320 cores @ 2.10 GHz）完成；评测在AMD EPYC 9275F（4.1 GHz，768 GB DDR5，48 threads）；每配置10次取平均。
- **截断与停止**：episode上限65次迭代或7天；早停三件套：loss爆炸（>100×初始或连续10 epoch上升）、action_diff>1.0持续恶化、value magnitude异常。
