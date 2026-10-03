---
title: "LEARNING-WHAT-TO-REMEMBER-LONG-HORIZON-COUNTERFACTUAL-MEMORY"
source: https://arxiv.org/pdf/2609.37930v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:53:56"
field: "持久记忆与长程信用分配"
keywords: ["persistent memory", "counterfactual credit assignment", "policy optimization", "document-level information extraction", "language model memory", "reinforcement learning"]
innovations: ["通过反事实比较相邻记忆状态隔离每次改写的长程边际贡献（Memory Gain）", "证明 Memory Gain 为潜在塑形奖励，期望策略梯度与事实目标严格等价", "提出位置级 EMA 校准的优势函数，在 G=1 下超越 GRPO-8 训练效率"]
benchmarks: ["SciREX", "AIPAN-10K", "BABILong"]
---

# 论文速读：LEARNING WHAT TO REMEMBER: LONG-HORIZON COUNTERFACTUAL MEMORY OPTIMIZATION

## 一句话总结
论文提出 Memory Gain Policy Optimization (MGPO)，通过反事实比较相邻记忆状态来隔离每次记忆更新对当前及未来下游任务的边际贡献，解决持久记忆学习中"归因模糊"问题；在文档级信息抽取任务上，MGPO 在减少约 80% 平均记忆长度的同时显著提升抽取性能，并实现跨读者、跨领域、跨任务的迁移复用。

## 研究问题与动机
- **信用分配难题（Credit-Assignment Problem）**：记忆状态的后更新效用与已有信息高度混淆——一次更新带来的下游收益可能大部分继承自更新前已存在的记忆内容，而非本次改写本身。
- **延迟效用难以归因**：当前写入时看似无关的信息，可能在后续多个 step 后才发挥作用；仅评估即时期望会遗漏长期价值。
- **已有方法无法度量单次更新的真实贡献**：现有记忆系统（检索式、显式管理、学习策略）多优化端到端任务奖励，但无法回答"某次记忆改写是否真正创造了增量价值"。
- **学习目标需要与事实目标保持一致**：改进归因信号的同时不能改变最优策略的期望梯度，否则可能偏离原始任务目标。

## 核心贡献（创新点）
1. **识别并形式化了"继承效用"（Inherited Utility）这一持久记忆学习的核心挑战**：指出记忆状态效用与改写级信用不是一回事，前者往往主导后者。
2. **提出 MGPO——基于长程反事实信用分配的强化学习训练框架**：通过预/后改写记忆在同一读者上的对比，构建 Memory Gain 信号；与 HiMPO 等方法依赖 hindsight/restart 不同，MGPO 无需重启轨迹，仅通过上三角反事实矩阵即可完成归因。
3. **证明 Memory Gain 是潜在塑形（Potential-Shaped）奖励**：严格证明其期望策略梯度与事实目标完全一致（$\nabla_\theta J_{\mathrm{MG}} = \nabla_\theta J_{\mathrm{fact}}$），$\Phi_t$ 作为与当前动作无关的控制变量降低方差而不偏置梯度。
4. **实验验证 MGPO 学会紧凑、选择性的记忆表示，且支持跨读者/跨域/跨任务复用**：在 SciREX 实体聚类 F1 达 61.0、二值关系 F1 达 26.5，较基线提升超过 60%；平均记忆长度仅为 MGPO-base 的约 21%。

## 方法详解

### 1. 解耦持久记忆写入架构
- **可训练 Writer $\pi_\theta$**：在 step $t$ 根据上一记忆 $m_{t-1}$ 和新输入 $c_t$ 生成更新后记忆 $m_t$，受固定预算 $B$ 约束。
- **冻结 Reader $\mathcal{E}(\cdot;\xi)$**：不参与训练，负责用记忆 $m_t$ 处理下游输入 $c_j$ 产出预测 $\hat{y}_{t,j}$。
- 解耦使得记忆状态效用 $F_{t,j}$ 可独立测量，不受读者策略干扰。

### 2. 从记忆状态效用到改写级信用
- **单次目标边际效用**：$\Delta_{t,j} = F_{t,j} - F_{t-1,j}$，固定下游目标与读者，差值即改写 $t$ 在目标 $j$ 上的增量。
- **Memory Gain（长程累计）**：$MG_t = \sum_{j=t}^{N} (F_{t,j} - F_{t-1,j})$，聚合当前及未来所有目标上的边际效应。
- **return**：$G_t = \sum_{k=t}^{N} MG_k$，叠加后续改写的 Memory Gain。

### 3. 反事实矩阵与潜在塑形证明
- 构建上三角矩阵 $F_{s,j}$（$s \leq j$），对角线为事实流式轨迹；Off-diagonal 单元评估早期记忆对后期目标的效用。
- 定义继承效用：$\Phi_t = \sum_{j=t}^{N} F_{t-1,j}$。
- **关键恒等式**：$MG_t = r_t^{\mathrm{fact}} + \Phi_{t+1} - \Phi_t$（势能塑形形式）；因此 $G_t = G_t^{\mathrm{fact}} - \Phi_t$。
- **策略梯度等价性**：$\mathbb{E}[\psi_t \Phi_t] = 0$，故 $\nabla_\theta J_{\mathrm{MG}} = \nabla_\theta J_{\mathrm{fact}}$；$\Phi_t$ 是动作无关的控制变量，只改变有限样本梯度估计的方差结构。

### 4. PPO 式训练目标
- **位置校准优势**：$A_t = \frac{G_t - b(t)}{\sqrt{\sigma^2 + \varepsilon}}$，其中 $b(t)$ 为位置 $t$ 的历史 EMA 均值，$\sigma^2$ 为全局 EMA 残差。
- **Token-level 截断策略目标**：
$$J(\theta) = \mathbb{E}\left[\frac{1}{N}\sum_t \frac{1}{L_t}\sum_l \min\left(\rho_{t,l} A_t, \mathrm{clip}(\rho_{t,l}, 1-\epsilon_c, 1+\epsilon_c) A_t\right)\right] - \beta_{\mathrm{KL}} \mathbb{E}[\hat{D}_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})]$$
- 对每个改写内 token 平均归一化，KL 项正则化 writer 不偏离参考策略。

### 5. 效用函数设计
- 以文档级 micro-F1 的一阶泰勒近似构造可加的 chunk-wise 替代效用 $F_{t,j} = \mathbf{w}^\top \mathbf{q}_{t,j}$，权重 $\mathbf{w}$ 基于 memory-off 参考点的梯度固定。
- 实体与关系任务权重各 0.5，无折扣（$\gamma=1$）。

## 实验与结果

### 数据集与设置
- **SciREX**（训练/验证/测试分片）：科学文献文档级 IE，标注 Method/Task/Material/Metric 实体及二元关系。
- **AIPAN-10K**（OOD 跨域）：网站隐私政策，50 篇抽样文档，更长且注释更密集。
- **BABILong**（QA 跨任务迁移）：QA1–QA5，上下文长度 8K–128K。
- 文档切分为 1,024 token chunk，记忆预算 256 token。Writer = Qwen3-8B，冻结 Reader = Qwen3-14B。

### 主要结果（SciREX）
| 方法 | 记忆长度 | Entity Cluster F1 | Binary Relation F1 |
|---|---|---|---|
| Direct Readout | — | 58.4 | 35.0 |
| LightRAG | 256 | 58.9 | 32.0 |
| MemAgent (14B reader) | 64 | 51.3 | 25.9 |
| HiMPO (14B reader) | 256 | 40.8 | 16.8 |
| **MGPO** | **43** | **61.0** | **26.5** |

- **相对 Direct Readout**：Entity Cluster F1 提升 **+30.9%**，Binary Relation F1 提升 **+60.6%**。
- **相对 MGPO-base**（未校准）：F1 提升的同时记忆长度从 202 → 43，**减少约 78.7%**。
- **相对 HiMPO/MemAgent**：以极小记忆量实现更高 F1，后者出现重复输出与格式违规。

### AIPAN-10K 跨域迁移
- MGPO 在无需额外训练下，Entity Cluster F1 = 85.9，Relation F1 = 22.9，超越最佳外部记忆基线（LightRAG: 86.8/21.0；Mem0: 86.8/21.3）或与其持平且记忆更短（168 vs 253/236）。

### BABILong 跨任务迁移
- 在 QA3 和多个 QA5 设置下 MGPO 优于 HiMPO 和 MemAgent，最长支持至 128K 上下文。

### 消融与诊断
- **信用分配消融**：FULL（MGPO）是唯一在所有指标上持续优于 NO MEMORY 的变体；FULL vs FACTUAL 的差距直接验证继承效用去除的重要性。
- **距离分层分析**：FULL 在长距离 chunk 上 precision 优势最大，说明改进主要来自长程证据整合。
- **梯度方差**：MGPO 将跨 rollout 梯度方差降低 **24.9%**（95% CI: 6.4%–42.6%）。
- **记忆改写行为**：FULL 训练出的改写呈现更多"负即时收益 + 正未来收益"的模式，体现对延迟价值的学习。

## 相关工作脉络
1. **HiMPO（Yan et al., 2026a）**：使用 hindsight-informed utility 减少 credit entanglement，但需多次 restart 采样，MGPO 仅通过反事实比较无需重启，训练更高效。
2. **Fine-Mem / MMPO（Ma et al., 2026; Liu et al., 2026）**：分别提供细粒度反馈和 belief-entropy 监督，关注内部记忆质量；MGPO 直接从下游读者效用出发，无需额外可微 supervise。
3. **Memory-R2（Yan et al., 2026b）**：引入 shared-state local rerollout 公平归因；MGPO 无需 rerollout，仅靠 counterfactual matrix 完成归因，计算成本更低。
4. **Counterfactual Credit Assignment（Foerster et al., 2018; Mesnard et al., 2021）**：多智能体/单智能体中经典反事实归因框架；MGPO 将其首次扩展到持久文本记忆改写场景。
5. **RECOMP / PRCA / CompAct（Xu et al., 2024; Yang et al., 2023; Yoon et al., 2024）**：压缩/适配固定下游模型的先验工作；MGPO 同样解耦 writer-reader，但进一步对 writer 实施 RL 优化并以反事实信用修正归因。
6. **Document-level IE（DocRED, SciREX 相关方法）**：图模型、循环表示、记忆 token 等；MGPO 不改变 reader 结构，而是通过学习何时记什么来增强任何 reader 的文档级推理能力。

## 局限性与未来方向
- **高阶交互未分解**：当多项改写共同产生效用后，Gain 归因到首个可观测时刻的改写，未进一步分摊到此前各贡献改写（论文附录 G 明确承认）。
- **依赖可加性近似**：利用文档级 micro-F1 的一阶泰勒近似作为可加分割效用；在极端 FP/FN 分布下可能有偏差。
- **训练时的反事实评估开销**：上三角矩阵需 $O(N^2)$ 次读者评估，虽通过稀疏目标评估缓解，但长文档仍显著高于 GRPO baseline。
- **未来方向**：显式建模改写间交互的联盟归因（coalitional attribution）；将方法扩展至非结构化推理任务；自动化记忆预算与 chunk 大小选择。

## 研究启发与可借鉴点
1. **"冻结下游读者 + 解耦 writer"的评估范式**：将 writer 与 reader 分离，使单次记忆改写的影响可被孤立测量，这一设计思路可直接迁移到记忆压缩、context management 等方向。
2. **反事实控制变量降方差**：$\Phi_t$ 作为动作无关的先验控制变量，在不偏置梯度的前提下降低 sample-wise variance；这一思路可与 learned critic 结合，适用于任意延迟奖励场景。
3. **位置级 EMA 校准替代 group-relative baseline**：用逐位置历史均值 $b(t)$ 替代 GRPO 的多 rollout 组内比较，在 $G=1$ 下即达到甚至超过 GRPO-8 的效果，大幅降低训练成本（26.9 vs 142 GPU-hours）。
4. **一阶线性化替代任务的 additive surrogate 设计**：通过固定参考点的梯度构造 chunk-wise 可加分离目标，巧妙规避非可微指标（F1）的优化难题，适用于其他组合型评估指标。
5. **可复用记忆 writer 的跨读者/跨域泛化**：单一 writer 适配 6 种不同模型和 3 类任务（SciREX→AIPAN-10K→BABILong），提示"记忆策略"与"下游读取器"解耦后具备独立投资价值，值得在 agent memory、RAG 检索策略中进一步探索。

## 关键术语表
- **Memory Gain Policy Optimization (MGPO)**：论文提出的方法，通过反事实比较相邻记忆状态对当前及未来下游目标的边际贡献来训练持久记忆 writer。
- **Inherited Utility ($\Phi_t$)**：在改写发生前，已有记忆对剩余所有下游目标的累积效用，是 MGPO 中需要从总效用中扣除的控制变量。
- **Counterfactual Evaluation Matrix**：上三角矩阵 $F_{s,j}$，衡量记忆状态 $m_s$ 对目标 $c_j$ 的效用，对角线为事实轨迹，off-diagonal 为反事实估计。
- **Potential-Shaped Reward**：形式为 $r + \Phi(s') - \Phi(s)$ 的奖励塑形，保持最优策略不变但改变有限样本梯度估计的方差结构。
- **Position-EMA Advantage Calibration**：对每个改写位置 $t$ 维护历史均值 $b(t)$ 进行归一化，解决长序列中不同位置 return 尺度系统性差异的问题。
- **Decoupled Persistent Memory Writing**：将记忆写入策略 $\pi_\theta$ 与下游读者 $\mathcal{E}$ 解耦，writer 唯一可训练，reader 固定用于测量效用。
- **Additive First-Order Surrogate for F1**：以 document-level micro-F1 在一阶泰勒展开近似构造 chunk-wise 可加替代效用，权重由 memory-off 参考点梯度决定。

## 可复现要素
- **数据集**：SciREX（公开）；AIPAN-10K（论文声明开源，arXiv:2609.26680）；BABILong（公开）。
- **代码/权重**：论文附录说明 full runtime prompts 随实现发布，完整代码开源声明见附录 C。
- **关键超参**：学习率峰值 $10^{-6}$，warmup 10 steps，cosine decay 至 $10^{-7}$；PPO clip $\epsilon_c = 0.2$；KL 系数 $\beta = 10^{-3}$；EMA decay $\alpha = 0.9$；数值稳定项 $\varepsilon = 10^{-6}$；memory budget 256 tokens；chunk size 1,024 tokens；训练 1 epoch，每 rollout 4 文档，每文档 1 条轨迹；writer 训练时 temperature=1.0/top-p=1.0，eval 时 temperature=0.7/top-p=0.8/top-k=20。
