---
title: "VETTA-COORDINATING-TURN-AND-TOKEN-LEVEL-CREDIT-ASSIGNMENT-FO"
source: https://arxiv.org/pdf/2610.08402v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:24:27"
field: "多轮LLM Agent强化学习"
keywords: ["Multi-turn Agent", "Credit Assignment", "Reinforcement Learning", "LLM Agent", "Turn-level Value", "Token-level Value"]
innovations: ["双粒度信用分配：联合建模turn-level和token-level advantage并通过centered residual融合", "轻量级共享Critic：仅保留早期Transformer blocks降低价值学习开销", "在ALFWorld和WebShop上超越所有critic-based和critic-free基线"]
benchmarks: ["ALFWorld", "WebShop"]
---

# 论文速读：VETTA-COORDINATING-TURN-AND-TOKEN-LEVEL-CREDIT-ASSIGNMENT-FO

## 一句话总结
VETTA是一种面向多轮LLM Agent的信用分配方法，通过共享轻量级critic分别学习turn-level和token-level价值，并将两者优势函数融合后指导PPO策略更新，在ALFWorld和WebShop上均取得最优成功率。

## 研究问题与动机
1. **稀疏反馈下的信用分配难题**：多轮Agent交互中任务奖励稀疏且延迟，需区分哪些response对结果有贡献，以及每个response内部哪些token生成决策更重要。
2. **单粒度方法的局限**：现有turn-level方法（如Turn-PPO）评估完整response但不区分内部token决策；token-level方法（如PPO*）跨轮传播反馈但不显式建模每个response的响应级信用。
3. **互补性假设**：turn-level advantage提供跨交互的response级进度信号，token-level advantage区分response内的生成决策差异，两者结合可提供更精细的策略梯度。
4. **Critic计算成本过高**：传统critic-based PPO需完整backbone进行价值推理，本文探索仅保留早期Transformer blocks是否能在保持性能的同时大幅降低计算开销。

## 核心贡献（创新点）
1. **双粒度信用分配框架**：首次在多轮Agent RL中系统联合建模turn-level和token-level价值，两者分别对应response级和token级的决策评估。
2. **Centered Token Residual融合机制**：提出在response内部对token advantage做中心化，使turn advantage决定响应的整体信用水平，token residual仅调整内部相对差异，避免双重偏移。
3. **轻量级共享Critic设计**：critic仅保留初始化actor所用预训练checkpoint的前d个Transformer blocks，与双head（turn/token）共享backbone但独立输出参数，显著降低价值学习开销。
4. **系统与计算效率的双重验证**：在ALFWorld和WebShop上超越所有critic-based和critic-free基线；仅用2个blocks的critic相比全28层节省约87-91%的critic侧推理时间。

## 方法详解
**双粒度价值估计**：
- 在每轮环境交互前，turn-value head从上下文表示$h_{k,1}$输出$V_k^{\text{turn}}=w_{\text{turn}}^\top h_{k,1}+b_{\text{turn}}$。
- 在response内每个有效token位置$j$，token-value head从前缀表示$h_{k,j}$输出$V_{k,j}^{\text{tok}}=w_{\text{tok}}^\top h_{k,j}+b_{\text{tok}}$。
- 两者共享同一个causal forward pass的backbone表示，但使用独立的新初始化head参数。

**Advantage估计（GAE）**：
- Turn-level： reward$r_k^{\text{turn}}=r_k$分配给第$k$轮，GAE参数为$\gamma_{\text{turn}},\lambda_{\text{turn}}$（实验设为0.95）。
- Token-level： reward仅在response的最后一个有效token位置赋予$r_k$，其余位置为0；GAE参数$\gamma_{\text{tok}},\lambda_{\text{tok}}$设为1。
- 两轮轨迹独立递归计算TD residual和优势函数，跳过observation tokens。

**优势融合**：
- 对token advantage做response内中心化：$\widetilde{A}_{k,j}^{\text{tok}}=A_{k,j}^{\text{tok}}-\frac{1}{L_k}\sum_{j=1}^{L_k}A_{k,j}^{\text{tok}}$。
- 最终融合优势：$A_{k,j}^{\text{VETTA}}=A_k^{\text{turn}}+\alpha \widetilde{A}_{k,j}^{\text{tok}}$，其中$\alpha$控制残差缩放比例（ALFWorld取1，WebShop取3）。
- 数学性质：融合后每个response的优势向量均值恰好等于turn advantage，token variation仅反映relative differences。

**Critic与Actor更新**：
- Critic使用clipped value regression，turn loss和token loss分别mean后相加。
- Actor使用标准token-wise PPO ratio clipping，目标为$fusion后的advantage$；支持dual-clipping和KL正则化（系数0.01）。

**轻量级Critic构建**：
- 从actor初始化用的预训练checkpoint中仅保留token embedding、前$d$个Transformer blocks和最终RMSNorm，丢弃后续blocks。
- 所有保留组件可训练，head参数随机初始化。

## 实验与结果
**数据集与模型**：
- ALFWorld：文本式具身任务基准（6类 household tasks），140 seen + 134 unseen测试任务。
- WebShop：模拟购物网站导航任务，500个固定测试实例。
- 基础模型：Qwen2.5-1.5B-Instruct和Qwen2.5-7B-Instruct。

**主要结果**：
- **1.5B模型**：VETTA在ALFWorld达到91.9%成功率（+5.8pp over GiGPO的86.1%），WebShop达到73.8%成功率（+6.4pp over GiGPO的67.4%）。
- **7B模型**：VETTA在ALFWorld达到95.5%，WebShop达到76.0%，均为两类baseline中最高。
- **Critic深度分析**：2-block critic在两个基准上均优于28-block full critic；critic侧时间从152s降至19.7s（ALFWorld，-87%）和从92.2s降至7.9s（WebShop，-91%）。

**消融实验**：
- 融合方式：centered residual（C4, 72.8%）显著优于direct addition（C3, 55.0%），验证中心化必要性。
- 单分支对比：仅用token credit（C1）或仅用turn credit（C6）均不如双粒度融合。
- 结构对比：共享backbone+独立heads优于独立critics或共享single head。

## 相关工作脉络
1. **Turn-PPO (Li et al., 2025)**：critic-based turn-level方法，对完整response估计value并使用PPO ratio clipping更新；VETTA在此基础上引入token-level residual以细化response内部决策。
2. **GiGPO (Feng et al., 2025)**：critic-free方法，结合episode-relative credit和重复状态下的action-level比较；VETTA对比表明critic-based双粒度融合在两项基准上均更优。
3. **PPO\***：作者重新实现的token-level baseline，跨turn传播GAE并跳过observation tokens；与Turn-PPO各有优劣，VETTA统一二者优势。
4. **HyGAE (Zhang et al., 2026)**：最接近的joint estimation工作，线性组合turn和token advantage并使用unified value function；VETTA的区别在于分离监督目标并使用centering而非直接加法。
5. **ArCHer (Zhou et al., 2024)**：hierarchical RL方法学习utterance-level value引导lower-level token policy；VETTA更强调在同一rollout内同时建模两层credit。
6. **HiPER (Peng et al., 2026)**：subgoal planning与action execution共享critic backbone；VETTA将共享backbone思想应用于turn/token双粒度场景。

## 局限性与未来方向
1. **任务依赖性**：最优$\alpha$在不同基准上差异显著（ALFWorld偏好$\alpha=1$，WebShop偏好$\alpha=3$），尚未建立自适应选择机制。
2. **Critic共享的表示收益未隔离**：shared backbone vs independent critics的对比中，参数数量和优化动态不同，无法完全归因于表征共享。
3. **仅评估两个benchmark**：结果局限于ALFWorld和WebShop，未验证于更长horizon或更复杂多模态agent任务。
4. **未来方向**：论文建议根据任务的交互结构和反馈模式定制化value model设计，可探索动态depth选择或task-specific head架构。

## 研究启发与可借鉴点
1. **Centering融合策略可迁移**：将token-level advantage中心化后作为residual叠加到turn-level advantage，这种"mean-preserving"融合方式适用于任何multi-granularity credit assignment场景，避免信号偏移。
2. **轻量级Critic设计验证**：仅保留早期Transformer blocks作为critic backbone在保持性能的同时大幅降低计算，为后续研究提供了"shallow critic"的设计范式。
3. **实验设计值得借鉴**：作者不仅比较最终性能，还系统分析了credit composition（Table 2）、critic structure（Figure 3）和depth（Figure 1b, 4）三个维度，提供了完整的消融逻辑。
4. **可与本团队方向结合**：若团队关注long-CoT或多步推理Agent，VETTA的双粒度信用分配可与HiPPO/VAPO等value稳定性工作结合，进一步探索token-level价值在长序列中的bootstrap误差控制。

## 关键术语表
**Credit Assignment**：强化学习中将最终任务奖励归因于轨迹中各个决策（turn或token）的过程。
**Turn-level Advantage**：针对完整response/交互轮次估计的优势函数，反映该轮决策对任务完成的贡献。
**Token-level Advantage**：针对response内每个生成token估计的优势函数，反映该token决策的相对价值。
**Generalized Advantage Estimation (GAE)**：Schulman等人提出的优势函数估计方法，通过TD residual的指数加权求和平衡偏差与方差。
**Within-response Centering**：在每个response内部对token advantage减去其均值，使residual的response内平均为零，从而保持turn advantage的整体信用水平。
**Lightweight Critic**：仅保留预训练checkpoint前若干Transformer blocks的价值网络，用于降低critic侧计算开销。
**PPO\***：本文重新实现的token-level PPO baseline，GAE跨turn递归并跳过observation tokens。
**Dual Clipping**：actor loss中的附加约束，当advantage为负时进一步裁剪loss，防止过度更新低价值token。

## 可复现要素
- **数据集**：ALFWorld和WebShop均为公开基准，数据集公开可用。
- **代码**：论文声明代码开源，链接为https://github.com/Jiaju-Chen/VETTA-official。
- **权重**：未提及开源模型权重，需使用Qwen2.5-1.5B/7B-Instruct预训练checkpoint。
- **关键超参**：$\gamma_{\text{tok}}=\lambda_{\text{tok}}=1$，$\gamma_{\text{turn}}=\lambda_{\text{turn}}=0.95$，$\alpha=1$（ALFWorld）/3（WebShop），critic blocks=2，训练update=150，actor LR=$10^{-6}$，critic LR=$10^{-5}$，KL coefficient=0.01。
