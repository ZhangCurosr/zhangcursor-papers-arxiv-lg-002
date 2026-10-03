---
title: "LEARNING-WHAT-TO-REMEMBER-LONG-HORIZON-COUNTERFACTUAL-MEMORY"
source: https://arxiv.org/pdf/2609.37930v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:54:09"
field: "大语言模型持久化记忆与信用分配"
keywords: ["persistent memory", "counterfactual credit assignment", "policy optimization", "document-level information extraction", "long-horizon RL", "memory writer", "inherited utility", "potential-shaped reward"]
innovations: ["提出Memory Gain Policy Optimization (MGPO)，通过冻结reader对相邻记忆状态做反事实比较，将每个重写的长程边际效用隔离为强化学习信号", "证明Memory Gain等价于对factual return施加action-independent控制变量，保持政策梯度期望不变但降低有限样本方差", "学到的记忆writer可跨不同reader架构、不同领域和下游任务零样本复用"]
benchmarks: ["SciREX", "AIPAN-10K", "BABILong"]
---

# 论文速读：LEARNING-WHAT-TO-REMEMBER-LONG-HORIZON-COUNTERFACTUAL-MEMORY

## 一句话总结
论文提出 **Memory Gain Policy Optimization (MGPO)**，通过冻结下游 reader 并对相邻记忆状态进行反事实比较，将每个记忆重写对当前及未来目标的边际效用隔离出来作为强化学习信号；在文档级信息抽取任务上，该方法显著提升抽取性能的同时将平均记忆长度压缩近 **80%**，且学到的记忆策略可跨模型、跨领域复用。

## 研究问题与动机
- **核心问题**：持久化文本记忆系统需要在容量限制下选择性保留、修订或丢弃信息，但"学习记住什么"本质上是一个**信用分配（credit assignment）**问题——信息在写入时看似无关，可能很久之后才变得有用。
- **现有方法的不足**：
  1. **继承效用混淆重写贡献**：一个记忆状态对下游任务的效用可能大部分来自先前已存储的信息，而非最新一次重写，导致无法衡量单次重写的真实增量价值。
  2. **延迟效用的信用分配困难**：记忆更新的价值往往体现在后续多个目标上，传统方法难以准确归因。
  3. **无法区分状态效用与动作效用**：现有记忆系统大多直接优化记忆状态的下游效用，而未将其拆解为"重写本身带来的边际变化"。
  4. **缺乏可度量的训练信号**：在持久化场景中，很难判断一个 memory rewrite 是否真正创造了下游价值。

## 核心贡献（创新点）
1. **识别继承效用为核心挑战**：首次系统性地指出持久化记忆学习中"记忆状态效用"与"记忆重写贡献"之间的混淆问题，并提出 counterfactual 视角予以分离。
2. **提出 MGPO 长程反事实信用分配机制**：通过冻结固定 reader，对 pre- 和 post-rewrite 状态在同一组下游目标上做配对比较，计算 Memory Gain 作为重写级的学习信号；证明该信号等价于对 factual return 施加一个 action-independent 的控制变量（control variate），保持政策梯度期望不变但降低有限样本方差。
3. **证明潜在形状化信用分配的理论与实践优势**：严格证明 Memory Gain 是 potential-shaped 的 factual reward，由此推导出的回报 $G_t = G_t^{\mathrm{fact}} - \Phi_t$，其中 $\Phi_t$ 是预重写记忆在未来 horizon 上的继承效用；该方法使 earlier rewrite 获得更合理的长期信用，避免了 naive factual reward 的方差过大问题。
4. **学到可复用、可迁移的紧凑记忆策略**：实验表明 MGPO 不仅提升文档级信息抽取性能，且学到的 writer 可跨不同规模/架构的 reader、跨不同领域（SciREX → AIPAN-10K）、跨下游任务（IE → QA on BABILong）零样本复用，同时平均记忆长度减少近 80%。

## 方法详解
- **解耦的持久化记忆写入框架**：输入 $c_1, \dots, c_N$ 按 chunk 流式到达，每步 writer $\pi_\theta$ 根据上一状态 $m_{t-1}$ 和当前 chunk $c_t$ 生成更新后的记忆 $m_t$，满足 $|m_t| \leq B$；reader $\mathcal{E}$ 固定不动，将 $(m_t, c_j)$ 输入得到预测 $\hat{y}_{t,j}$。
- **记忆状态效用**：$F_{t,j} = \mathbb{E}_\xi[\mathcal{U}(\hat{y}_{t,j}(\xi), y_j)]$，其中 $\xi$ 为固定 reader 的采样噪声，实践中用一次 Monte Carlo 采样估计。
- **Rewrite 级信用分配**：单次重写 $t$ 对目标 $j$ 的边际效用为 $\Delta_{t,j} = F_{t,j} - F_{t-1,j}$。
- **Memory Gain（长程聚合）**：$MG_t = \sum_{j=t}^{N} \Delta_{t,j} = \sum_{j=t}^{N}(F_{t,j} - F_{t-1,j})$，即对当前及所有未来目标求和。
- **回报定义**：$G_t = \sum_{k=t}^{N} MG_k$，聚合当前及后续所有重写的 Memory Gain。
- **理论关键分解**：引入 $\Phi_t = \sum_{j=t}^{N} F_{t-1,j}$（预重写记忆在未来 horizon 上的总效用，即"继承效用"），则 $MG_t = r_t^{\mathrm{fact}} + \Phi_{t+1} - \Phi_t$，进而 $G_t = G_t^{\mathrm{fact}} - \Phi_t$。由于 $\Phi_t$ 由 $m_{t-1}$ 决定、与当前动作 $m_t$ 条件独立，故 $\mathbb{E}[\psi_t \Phi_t] = 0$，政策梯度期望保持不变，但有限样本估计方差降低。
- **位置校准优势函数**：为避免不同时间步回报量纲差异，采用位置专属的指数移动平均（EMA）去中心化：$A_t = (G_t - b(t)) / \sqrt{\sigma^2 + \varepsilon}$，其中 $b(t)$ 为位置 $t$ 的历史均值，$\sigma^2$ 为全局方差。
- **PPO 风格的 token-level 损失**：对每次重写的每个 token 分配 $A_t$，结合 clipping ratio 和 KL 正则项：$J(\theta) = \mathbb{E}[\frac{1}{N}\sum_t \frac{1}{L_t}\sum_l \min(\rho_{t,l} A_t, \mathrm{clip}(\rho_{t,l}, 1-\epsilon_c, 1+\epsilon_c) A_t)] - \beta_{\mathrm{KL}}\mathbb{E}[\hat{D}_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})]$。
- **加性效用代理**：对文档级 micro-F1 构造一阶线性化 surrogate，利用离线 anchor $(TP_0, FP_0, FN_0)$ 的梯度 $\mathbf{w}$ 计算 $F_{t,j} = \mathbf{w}^\top \mathbf{q}_{t,j}$，使得 chunk-wise 效用可加。

## 实验与结果
- **数据集**：SciREX（科学文献文档级 IE，66 篇测试文档，均长 6,947 tokens）；AIPAN-10K（隐私政策，50 篇测试文档，均长 15,375 tokens，用于跨域迁移评估）；BABILong（长上下文 QA，8K–128K）。
- **评估指标**：entity cluster micro-P/R/F1、binary relation micro-P/R/F1；记忆长度为提供给 frozen reader 的平均 token 数。
- **基线**：Direct Readout、R1-RE（直接读取）、LightRAG、Mem0（外部记忆）、MemAgent、HiMPO（学习记忆策略）；内部消融对比 NO MEMORY、BASE POLICY、TERMINAL、FACTUAL、MYOPIC、FULL（MGPO）。
- **主要结果（SciREX in-domain）**：MGPO entity cluster F1 = **61.0**，binary relation F1 = **36.8**，均为所有对比方法最高；相对 Direct Readout 分别提升 **30.9%** 和 **60.6%**。平均记忆长度仅 **43 tokens**，较 MGPO-base（202 tokens）下降 **78.7%**。
- **跨域迁移（AIPAN-10K，未经训练或微调）**：MGPO entity cluster F1 = **39.2**，binary relation F1 = **22.9**，显著优于 MemAgent（18.9/4.4）和 HiMPO（9.9/1.8），略超 LightRAG（38.8/21.0）。
- **可复用性（Table 2）**：同一 MGPO 训练的 8B writer 与 6 种不同 reader（Qwen3-8B/14B/32B、Llama3.1-8B、Nemo-12B、Phi-4-14B）搭配，entity cluster F1 均显著高于无记忆 baseline。
- **消融结论**：FULL（MGPO）是唯一在所有 setting 上稳定超越 NO MEMORY 的变体；与 FACTUAL 对比证明减去继承效用至关重要；MYOPIC 仅看当前目标、TERMINAL 仅看文档末尾，效果均不如 FULL；距离分层分析显示 MGPO 主要在长距离 chunk 上提升 precision，说明其学会了跨远距离复用信息。

## 相关工作脉络
- **Recurrent/Retrieved Memory**：Bulatov et al. (2022)、Wang et al. (2023) 学习循环或检索式记忆；本文将其拓展至文本型持久化 memory writer 的强化学习优化。
- **Upstream Adaptation against Frozen Downstream**：RECOMP (Xu et al., 2024)、PRCA (Yang et al., 2023)、CompAct (Yoon et al., 2024) 同样冻结 reader 训练 compressor/adapter；本文沿用此范式但创新性地引入 counterfactual credit 而非直接优化 factual return。
- **RL-based Memory Policies**：Memory-R1 (Yan et al., 2026c)、MemAgent (Yu et al., 2026a)、HiMPO (Yan et al., 2026a)、MemPO (Li et al., 2026) 等使用 RL 优化记忆策略；HiMPO 引入 hindsight-informed utility，本文进一步用 pair-wise counterfactual comparison 分离当前重写与历史继承效用，提供更细粒度的单步信用。
- **Counterfactual Credit Assignment**：Difference rewards (Tumer et al., 2002)、COMA (Foerster et al., 2018)、Hindsight Credit Assignment (Harutyunyan et al., 2019)、Mesnard et al. (2021)；本文将该思想首次扩展至单 agent 持久化文本记忆的长程信用分配。
- **Process Reward / Progress Reward**：Lightman et al. (2024) 中间步骤奖励、Setlur et al. (2025) 进展奖励；本文的 Memory Gain 类比为"记忆更新带来的进展"，但通过 pre-/post- rewrite 反事实比较精确定义。
- **Document-level IE**：DocRED (Yao et al., 2019)、TTEM-RE (Gao et al., 2024)、R1-RE (Dai et al., 2026)；本文聚焦于在长文档流式处理中主动维护 bounded textual memory 以支持远距离证据整合。

## 局限性与未来方向
- **未显式分解高阶交互**：Memory Gain 仅度量相邻状态间的边际变化，当多个互补重写共同促成下游效用显现时，新增益会归因于第一个使效用可观测的重写，而非在多个贡献者之间分配（论文附录 G）。
- **线性化 surrogate 的近似误差**：document-level micro-F1 是非加性的，本文用一阶泰勒近似构造可加分割；虽然实验显示 Spearman ρ = 0.988 高度一致，但在极端情况下仍存在偏差。
- **训练成本随文档长度增加**：counterfactual 矩阵为三角形态，训练开销随 chunk 数线性增长；对超长文档（>16K tokens）可能需要更稀疏的目标选择策略（见 Appendix F）。
- **未来方向**：显式分解多个重写之间的交互贡献（如 coalition-based attribution）、探索更精确的非线性 F1 代理、将方法推广至多 agent 场景或更大规模的跨任务统一 memory writer。

## 研究启发与可借鉴点
1. **反事实差值作为 control variate**：用 $G_t^{\mathrm{MG}} = G_t^{\mathrm{fact}} - \Phi_t$ 剥离继承效用，理论上保持政策梯度不变但显著降低有限样本方差；这一思路可迁移到任何需要区分"当前动作贡献"与"历史状态存量"的序列决策场景。
2. **冻结下游 reader + 解耦 writer**：writer 与 reader 分离设计使得记忆策略可以跨不同 reader 架构、规模零样本复用，无需重新联合训练；这一范式值得在更多长上下文任务中验证。
3. **位置专属 EMA 校准返回尺度**：针对不同时间步回报系统性差异，使用 position-specific EMA 而非全局均值中心化，能有效消除 horizon-dependent scale 偏置；该方法实现简单且仅需单样本估计。
4. **一阶线性化 surrogate 构造可加分部效用**：对非加性任务指标（如 micro-F1）在 offline anchor 处线性化，构造 chunk-wise 可加的 utility 代理，使长程信用分配在计算上可行且与真实指标高度相关（ρ=0.988）。
5. **跨域零样本迁移验证**：在 SciREX 上训练后直接应用于 AIPAN-10K（完全不同的领域和 schema），证明学到的记忆策略捕捉的是"任务无关的有用信息保留原则"，而非过拟合特定数据分布。

## 关键术语表
- **Memory Gain Policy Optimization (MGPO)**：论文提出的强化学习框架，通过反事实比较 pre-/post-rewrite 状态在当前及未来目标上的效用差值，为每个记忆重写分配长程信用。
- **Inherited Utility ($\Phi_t$)**：预重写记忆 $m_{t-1}$ 在未来剩余 horizon 上已支持的下游效用总和，作为 control variate 从 factual return 中剔除，以分离出当前重写的边际贡献。
- **Counterfactual Credit Assignment**：通过保持除当前动作外的所有因素不变，比较 action 执行前后结果的差异，从而将效用归因于该动作而非环境或其他因素的方法论。
- **Potential-shaped Reward Shaping**：Ng et al. (1999) 的理论结果，指在原始 reward 上叠加势能函数之差 $\phi(s') - \phi(s)$ 不改变最优策略；本文证明 MG 属于此类。
- **Position-calibrated Advantage ($A_t$)**：对每个时间步 $t$ 的回报使用位置专属的 EMA 均值和全局 EMA 方差进行去中心和归一化，以消除 horizon-dependent 尺度差异。
- **Additive Utility Surrogate**：基于 document-level micro-F1 在 memory-off anchor 处的一阶泰勒展开，将非加性指标近似为可跨 chunk 求和的线性加性效用。
- **Decoupled Persistent Memory Writing**：将记忆写入策略 $\pi_\theta$（可训练）与下游 reader $\mathcal{E}$（固定）解耦的设计，使记忆效用可独立度量且 writer 可跨 reader 复用。
- **Factual Return ($G_t^{\mathrm{fact}}$)**：沿 streaming 轨迹的实际 reward 之和 $\sum_{k=t}^N F_{k,k}$，本文证明直接使用它会导致方差过大，需减去继承效用才能得到更优的梯度估计。

## 可复现要素
- **数据集**：SciREX（公开，https://github.com/allenai/scirex）、AIPAN-10K（论文引用 Tang et al., 2026，arXiv:2609.26680）、BABILong（公开，https://github.com/facebookresearch/babilong）。
- **代码/权重**：论文附录 C.5 声明 "Full runtime prompts are released with the implementation"，具体仓库链接需查论文补充材料；模型使用 Qwen3-8B（writer）和 Qwen3-14B（reader）。
- **关键超参**：chunk size = 1024 tokens、memory budget = 256 tokens、AdamW peak lr = $10^{-6}$、warmup = 10 steps、cosine decay to $10^{-7}$、PPO clip $\epsilon_c = 0.2$、KL coefficient $\beta_{\mathrm{KL}} = 10^{-3}$、EMA decay $\alpha = 0.9$、discount $\gamma = 1$、training epochs = 1、batch = 4 documents × 1 trajectory。
