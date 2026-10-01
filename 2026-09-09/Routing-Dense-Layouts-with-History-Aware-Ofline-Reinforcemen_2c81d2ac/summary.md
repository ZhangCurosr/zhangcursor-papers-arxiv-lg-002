---
title: "Routing-Dense-Layouts-with-History-Aware-Ofline-Reinforcemen"
source: https://arxiv.org/pdf/2609.08232v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:00:23"
field: "电子设计自动化（EDA）中的物理设计自动化"
keywords: ["Detailed Routing", "Offline Reinforcement Learning", "Conservative Q-Learning", "LSTM", "Physical Design", "OpenROAD", "Distribution Shift"]
innovations: ["引入LSTM heads的序列感知CQL策略以保留布线迭代历史上下文", "对密度/调整特征施加Bernoulli掩码正则化以缓解跨分布偏移", "构建覆盖高密度难路由工况的开源训练数据集与完整训练-推理pipeline"]
benchmarks: ["OpenROAD Design Suite (Nangate45)", "8 designs across 6 densities with routing adjustment sweeps"]
---

# 论文速读：Routing-Dense-Layouts-with-History-Aware-Offline-Reinforcement-Learning-using-LSTM

## 一句话总结
本文提出了一种基于历史感知的离线保守Q学习（CQL）策略，通过引入轻量级LSTM架构保留布线序列上下文，在高密度放置条件下动态预测迭代成本权重，使详细路由器的设计规则违规（DRV）平均减少92%，同时运行时间降低10%。

## 研究问题与动机
- **详细路由是物理设计的主要运行时瓶颈**：随着工艺节点缩小和逻辑密度提升，布线规则复杂性急剧增加，现代路由器在高密度条件下难以解决持续存在的违规。
- **固定成本调度方案在高密度下失效**：现有主流路由器（如OpenROAD/TritonRoute）依赖人工设计的固定成本权重表，无法适应局部拥塞和违规组合的动态变化。
- **先前的离线RL方法存在分布偏移问题**：prior work [4] 的CQL方法在单一工作点上表现优异，但当密度和调节条件变化时，原始特征分布发生显著偏移，导致策略性能急剧退化。
- **高密度布局的资源约束更严格**：全局路由调整值越低（如0.0），可用布线轨道越少，详细路由器的反复提取/重路由空间被严重压缩。

## 核心贡献（创新点）
- **引入LSTM heads的序列感知CQL策略**：通过轻量级2层LSTM捕获短程布线历史依赖（违规迁移、热点持久性、权重变化的延迟效应），与 prior work [4] 的单步状态决策形成本质区别。
- **针对性的特征变换与掩码正则化**：对宽范围数值特征使用比率和对数变换；在训练中对放置密度、调整信号及其交互项施加Bernoulli掩码（dropout），防止策略过拟合到单一工作点。
- **构建涵盖高密度与难路由工况的训练数据集**：通过扰动采样和随机探索在8个Nangate45设计、6个密度、0.0–0.3调整范围内收集约10,000条布线轨迹，覆盖 baseline 无法收敛的极端工况。
- **与现有路由器无缝集成的轻量接口**：仅需修改4个成本乘数（`drcCost`, `markerCost`, `fixedShapeCost`, `markerDecay`），不干扰核心搜索算法，可移植到任意基于成本的迭代路由器。

## 方法详解
**问题建模**：将每次迭代的成本权重选择建模为顺序决策过程，状态序列为 $\{s_1, s_2, ..., s_n\}$，episode 在 DRV 归零或超过65次迭代时终止。

**特征体系**：
- *动态特征*：当前/初始违规计数（log变换）、停滞违规区域、最大局部违规计数变化与聚类扩展、违规类型比率（short/metal spacing/cut spacing/EOL spacing）、历史权重及变化率
- *静态设计特征*：终端计数（log）、芯片面积（log）、放置密度、路由调整、以及三者的交互项（揭示"相同密度因设计不同而难度各异"的现象）

**模型架构**：Double-Q critic + stochastic actor 的 SAC 家族网络，每层2个LSTM（hidden units: 512, 512, 256），输出经有界squashing函数映射到训练数据匹配的动作范围。

**奖励函数**（Eq. 2）：
$$C_{inst} = (1 + \alpha \log(T+1) + \beta \log(A+1)) \cdot g(\rho)$$
包含进度（相对初始DRV的改善率）、速度（收敛奖励 + 迭代惩罚）、难度感知缩放（由终端数、面积、密度调制）、局部热点（最大局部违规计数变化）。注意：训练目标中**不包含**累计DRV和真实运行时间。

**训练协议**：Replay buffer 存储完整episode且保持时序；100轮Optuna超参搜索；三个早期停止监控（loss爆炸、action\_diff>1.0、值溢出）；全部在320核CPU上训练。

## 实验与结果
- **数据集与平台**：8个Nangate45设计（aes, ariane136, bp_be, bp_fe, bp_multi, gcd, ibex, jpeg），在AMD EPYC 9275F @ 4.1GHz + 768GB DDR5上评测，每配置10次取平均。
- **两类测试工况**：
  - *(A) 收敛案例*：测试密度插值于训练密度之间（distribution shift），调整值固定；
  - *(B) 难案例（baseline非收敛）*：在更低调整值（如0.0）和高密度下测试。
- **主要结果**：
  - 难案例（Table 3）：Total DRVs 从15,767降至1,215，**减少92.29%**；Total runtime 从2,159,458s降至1,934,420s，**减少10.42%**；wirelength仅增加0.32%。
  - 收敛案例（Table 2）：在各设计默认密度上均优于baseline和 prior work [4]，Total iterations从65降至37，Total runtime 从598s降至420s（-10.06%）。
  - **极端密度（0.74–1.00）**：gcd和ibex在density=1.00下成功收敛（0 DRV）；aes在density=0.74下DRVs从5,342降至486（-90.90%）。
- **对比 prior CQL [4]**：Figure 1显示 prior CQL 随密度上升性能急剧退化，而CQL+LSTM保持稳定；prior work 仅报告单密度，故难案例中未纳入对比。

## 相关工作脉络
- **TritonRoute / OpenROAD**：开源详细路由器的SOTA基准，使用基于A*搜索的迭代rip-up-and-reroute，但成本权重由人工调度表固定控制，本文在此基础上叠加RL策略。
- **Prior work [4] (CQL-only)**：首次将离线保守Q学习用于详细路由成本权重生成，但仅在单一密度/调节点评估，缺乏跨密度泛化能力；本文在其基础上增加LSTM与特征工程实现跨密度稳健性。
- **Chen et al. [21]**：在线RL框架（GNN + PPO）用于自定义电路路由，与本文离线设定和迭代控制目标不同。
- **热点预测类工作 [18-20]**：利用RF/CNN预测拥塞或DRV热点，但需额外机制间接影响路由行为，而非直接控制迭代成本权重。
- **CQL算法原论文 [23]**：保守Q-learning防止分布外动作过度估计的离线RL方法，本文以其为基础并引入序列建模。

## 局限性与未来方向
- **wirelength未作为优化目标**：奖励函数中未显式包含线长，仅作为评估指标，可能存在优化空间。
- **极端难案例受限于数据生成时间**：Table 3中的hard cases仍是在"合理运行时间约束内"能找到的最难点，并非理论最坏情况。
- **LSTM推理开销在简单设计上占比显著**：如gcd和ibex等本就快速收敛的设计，3秒的LSTM推理开销相对路由加速优势不明显，但在高拥塞场景下可忽略。
- 未来可扩展到更广泛的工艺节点、更多设计家族，以及探索per-partition而非global iteration的细粒度控制。

## 研究启发与可借鉴点
- **对数/比率变换应对宽范围特征**：将绝对计数转换为log或relative ratio，有效缓解跨密度分布偏移导致的action saturation问题，可直接迁移到其他EDA/物理设计ML任务。
- **特征掩码（feature masking）正则化**：对易引起分布偏移的输入特征（密度、调整值）施加Bernoulli dropout，是处理"单分布训练 → 多分布推理"场景的有效手段。
- **离线RL + RNN的序列建模范式**：将CQL与LSTM结合用于迭代控制问题，兼顾了离线训练的稳定性与历史状态的建模能力，可推广至其他需要多步决策的物理设计子问题。
- **低成本集成的"4 knob"接口设计**：仅暴露4个成本乘数供策略调控，不修改核心搜索算法，使得方法具有高度可移植性，为其他路由器适配提供工程参考。

## 关键术语表
- **Detailed routing（详细路由）**：物理设计流程中将net分配至具体金属层和routing track的步骤，需在满足复杂设计规则的前提下完成。
- **Conservative Q-Learning (CQL)**：一种离线强化学习算法，通过保守Q目标（penalize out-of-distribution actions）防止值函数过度估计，适合从固定数据集学习策略。
- **Design Rule Violation (DRV)**：布线结果违反制造工艺规则的情况，是衡量详细路由质量的核心指标。
- **Placement density（放置密度）**：芯片核心区域内标准单元占据的面积比例，直接影响布线拥塞程度。
- **Routing adjustment（路由调整）**：全局路由阶段控制可用routing track比例的参数，低值意味着更紧凑但更难的详细路由环境。
- **LSTM（长短期记忆网络）**：一种递归神经网络，通过门控机制捕获序列中的长期依赖，适用于建模布线迭代过程中的状态演化。
- **Soft Actor-Critic (SAC)**：一种最大熵离线/在线RL算法，通过熵温度项鼓励探索，本文以其stochastic actor为骨架。
- **Feature masking（特征掩码）**：在训练时对部分输入特征随机置零（Bernoulli dropout），防止模型过拟合到特定分布条件。

## 可复现要素
- **数据集**：Nangate45设计套件，8个设计 × 6个密度 × 多种routing adjustment（0.0–0.3），约10,000条布线轨迹；论文未说明数据集是否独立公开，但代码已开源。
- **代码开源**：https://github.com/realise-lab/RLDRT（训练与推理实现均已开源）
- **关键超参**：详见Table 1——Critic隐藏层[512, 512, 256]，Actor LR=1e-4，Critic LR=3e-4，Conservative Weight=2.44，Batch Size=512，Initial Temperature=1.0，Tau=1e-2，Temporal Window未明确给出但提及为调参超参。
- **硬件**：训练在320核CPU @ 2.10 GHz上进行；评测在AMD EPYC 9275F @ 4.1 GHz + 768GB DDR5 + 48 threads。
