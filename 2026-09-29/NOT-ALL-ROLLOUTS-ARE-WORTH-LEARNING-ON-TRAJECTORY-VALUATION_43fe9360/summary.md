---
title: "NOT-ALL-ROLLOUTS-ARE-WORTH-LEARNING-ON-TRAJECTORY-VALUATION"
source: https://arxiv.org/pdf/2609.35072v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:36:13"
field: "强化学习数据选择与后训练优化"
keywords: ["trajectory valuation", "reinforcement learning", "influence function", "dynamic filtering", "PPO", "GRPO", "DPO"]
innovations: ["提出基于 batch 内梯度对齐的 Dynamic Trajectory Valuation (DTV)，无需验证集即可在线识别有害轨迹", "推导自项/交叉项分解并提出留一法变体 DTV-Loo，揭示并缓解 self-protection effect", "在 PPO/GRPO/DPO 三范式及高噪声场景下验证零阈值自适应过滤的一致有效性"]
benchmarks: ["MiniGrid Empty-8x8/DoorKey-8x8", "GSM8K (Clean/Mismatch-20%)", "AIME 2024", "UltraFeedback (Clean/Mismatch-20%/Mismatch-40%)"]
---

# 论文速读：NOT-ALL-ROLLOUTS-ARE-WORTH-LEARNING-ON-TRAJECTORY-VALUATION

## 一句话总结
本文提出 **Dynamic Trajectory Valuation (DTV)**，一种基于 mini-batch 内梯度对齐的轨迹估值框架，无需额外验证集或辅助信号即可在线动态识别并过滤有害训练单元；DTV 及其留一法变体 DTV-Loo 在 PPO、GRPO、DPO 多种 RL 范式下均带来一致的加速收敛与性能提升。

## 研究问题与动机
1. **强化学习中轨迹质量不均**：与监督学习不同，RL 的训练数据（轨迹/rollout）由智能体在线生成，且无固定训练集与验证集；不同轨迹对策略优化的贡献差异显著，部分轨迹甚至会干扰优化。
2. **传统 influence function 难以直接迁移**：经典 influence-based 方法依赖固定验证集与影响函数，而 RL 轨迹效用具有策略依赖性与时序信用分配特性，使现有方法无法直接应用。
3. **已有 RL 数据选择方法泛化受限**：IIF、LearnAlign、GradAlign 等方法分别依赖特定算法的代理目标、可验证反馈或可信验证梯度，对超参/额外信号敏感，跨范式通用性不足。
4. **缺少统一、低开销的轨迹级估值框架**：现有工作多停留在 transition 级或启发式过滤，缺乏对整条轨迹贡献的 principled 度量与高效在线计算方案。

## 核心贡献（创新点）
1. **统一轨迹估值形式化**：将"轨迹有用性"问题转化为"梯度对齐分析"，给出适用于一般 RL（含 PPO/GRPO/DPO）的统一 DTV 分数公式（Eq.7），区别于仅依赖外部验证信号的方法。
2. **DTV / DTV-Loo 双变体与自保护效应揭示**：推导出 DTV 分数的自项（self-term）与交叉项（cross-term）分解（Section 4.3），定义"self-protection"现象并据此提出剔除自项的留一法变体 DTV-Loo（Eq.8）。
3. **零阈值自适应过滤、无需预设比例**：以 0 为固定过滤阈值，过滤比例随训练动态调整，区别于 Random/Reward 等需要人工设定筛选比例的方法。
4. **跨三范式的一致有效性与可迁移性**：在 PPO（MiniGrid）、GRPO（GSM8K/AIME）、DPO（UltraFeedback + 人工噪声 Mismatch）三类设置下均验证有效；在复杂/高噪声场景（AIME、Mismatch-40% DPO）中 DTV-Loo 明显优于 DTV。
5. **JAX/Tunix 工程化与可扩展实现**：借助向量化的 per-unit 梯度计算，单次 rollout 内完成 valuate+filter，不引入额外采样代价，可无缝接入现有 RL pipeline。

## 方法详解
**问题转化**：由 influence score 公式可知，影响样本重要性的关键是该样本的梯度项 $\nabla \ell(z_j;\theta)$；因此将"轨迹估值"转化为"该训练单元梯度与参考方向的对齐"问题。

**核心假设**（Assumption 1, Majority Gradient Alignment）：一个 mini-batch 中大多数训练单元的梯度方向构成当前优化的局部参考方向；与该方向负对齐的单元被视为有害。

**DTV 分数**（$B=\{z_j\}_{j=1}^b$，$g_j^{(t)}=\nabla_\theta \ell(z_j;\theta^{(t)})$）：
$$\mathcal{T}^{\text{DTV}}(z_j;\theta^{(t)}) = \frac{1}{b}\sum_{i=1}^b (g_i^{(t)})^\top g_j^{(t)}$$
以 0 为阈值：$\mathcal{T} \geq 0$ 保留，$\mathcal{T} < 0$ 丢弃；被丢弃单元的全部 transition/pair 将从当次 optimizer update 的 loss 中剔除。

**DTV 分解与 Self-protection**：
$$\mathcal{T}^{\text{DTV}} = \underbrace{\frac{1}{b}\|g_j\|^2}_{s_j \geq 0} + \underbrace{\frac{1}{b}\sum_{i \neq j}(g_j)^\top g_i}_{c_j}$$
- 自项 $s_j$ 恒非负，可"抵消"负的交叉项 $c_j$，导致某些与多数方向冲突的单元仍被保留，作者称此为 **self-protection effect**。
- 为此提出 **留一法变体 DTV-Loo**：
$$\mathcal{T}^{\text{DTV-Loo}}(z_j) = \frac{1}{b-1}\sum_{i \neq j}(g_i)^\top g_j = \frac{b}{b-1}\left(\mathcal{T}^{\text{DTV}} - s_j\right)$$
完全依赖跨单元交叉项，避免单个"大自项"样本主导整个 batch 的估值。

**操作单位因算法而异**：PPO 对应一条 trajectory segment，GRPO 对应一个 prompt 的一组 completion 中的一条 completion，DPO 对应一对 $(y^+, y^-)$。

**训练流程（Algorithm 1）**：每步计算 batch 内各单元梯度 → 计算 DTV 分数 → 按阈值保留 $B_+$ → 用 mask 后的单元执行更新。无需额外 rollout、无需外部验证集、无独立超参（λ 可统一但未在实际中进一步调参）。

## 实验与结果
**PPO（MiniGrid：Empty-8x8 / DoorKey-8x8）**
- 对比：Vanilla PPO、IIF。
- 结果：DTV/DTV-Loo 均加速收敛；DoorKey 上 DTV-Loo 早期收敛略快于 DTV；最终最差/最好 20% 分位回报均优于 IIF。说明过滤有害轨迹能减少优化干扰、提高收敛稳定性。

**GRPO-GSM8K（Clean & Mismatch-20%，Gemma-3-1B-IT + LoRA）**
- 对比：Vanilla GRPO、Random 5%/10%、Reward 5%/10%、LearnAlign、GradAlign。
- 关键数字：DTV 在 Clean 下 Acc 提升 +0.0760、Partial +0.0742；Mismatch-20% 下 DTV Acc 提升 +0.0690、Partial +0.0660，均优于所有基线。DTV-Loo 略逊于 DTV，但仍大幅领先 Vanilla/Random/Reward/LearnAlign/GradAlign。
- 分析：自项主导 DTV 分数；DTV 略优于 DTV-Loo，说明低冲突场景下保守过滤有益。

**GRPO-AIME 2024（DeepSeek-R1-Distill-Qwen-1.5B，全参数微调，单 seed，双 TPU v5p-16）**
- 对比：Vanilla GRPO、DTV、DTV-Loo。
- 关键数字（相对 pre-trained）：DTV-Loo Pass@1 +0.0270，Pass@16 +0.1000，Maj@16 +0.1334，Extractable 0.7521，Acc given extractable 0.1773；均为最优。DTV 仅在 Maj@16 上提升。
- 结论：高难度、复杂任务下 DTV-Loo 更优；自保护效应在高噪声/高冲突下更易暴露。

**DPO-UltraFeedback（Qwen2.5-1.5B，SFT→DPO 两阶段，Clean / Mismatch-20% / Mismatch-40%）**
- 对比：Vanilla DPO、Random 5%/10%、Reward 5%/10%、DTV、DTV-Loo。
- 关键数字（AUC）：Mismatch-40% 下 DTV-Loo AUC 0.5406，远超 Reward 10% 的 0.5132（+0.0274）；最终 Acc 0.5196 vs. 0.4857（+0.0339），配对 t-test 显著（p<0.0001）。Clean 下 DTV-Loo 亦最优。
- 关键数字（时间效率）：Mismatch 条件下 DTV-Loo 达到 95% 目标准确率仅约 30 分钟，而 Vanilla DPO 耗时更长，Random 10% 甚至未能达标。
- 结论：DPO 中 cross-term 贡献持续且递增，噪声越重 DTV-Loo 优势越明显；自保护在强噪声下会掩盖冲突。

**总体最强结果**：AIME-GRPO（Maj@16 0.3667，较 Vanilla 提升 +0.1334）；DPO-UltraFeedback Mismatch-40%（AUC 0.5406、Acc 0.5196，较次优显著领先）；GSM8K-Clean（Acc 0.5483，较预训练 +0.0760）。

## 相关工作脉络
1. **IIF (2025)**：基于回放缓冲区与算法专属代理目标的 influence 过滤；需指定过滤比例、依赖 PPO 特有目标，泛化性受限。DTV 无外置目标、零阈值自适应。
2. **LearnAlign (2026)**：依赖可验证 ground-truth 反馈计算 learnability，需在 RLVR 固定数据集上做额外 rollout；DTV 无需额外 rollout、适用于纯在线环境交互。
3. **GradAlign (2026)**：需要可信验证集构建 reference gradient，且需额外 rollout 计算 candidate gradient；DTV 完全在 batch 内部求解，无需额外资源。
4. **Prioritized Experience Replay / Curriculum Learning**：在 transition 级依据 TD-error 或即时 reward 启发式筛选，未建模整条轨迹的全局贡献。DTV 在 trajectory/completion/pair 层面做梯度对齐。
5. **Influence Function 谱系（Koh & Liang, DataInf, TRAK, EKFAC, Hessian-free 等）**：多为静态、监督设定；DTW 将其扩展至 RL 的在线、动态场景，并通过 batch 内梯度对齐绕过 Hessian 近似。
6. **Reward-based / Rejection Sampling**：粗粒度硬阈值，不感知梯度方向。DTV 通过梯度内积度量"与当前优化方向一致与否"，更精细。

## 局限性与未来方向
1. **自项与交叉项权衡尚依赖经验判断**：DTV vs. DTV-Loo 的选择由任务噪声/冲突程度决定，论文建议以 DTV-Loo 为默认，但未给出普适的自动判别准则。
2. **单一零阈值可能不够精细**：虽然避免了手工调参，但在不同训练阶段或不同任务上，固定 0 阈值的严格程度并不一定最优。
3. **下游泛化存在不一致性**：DPO 下游指令跟随任务（LiveBench-IF、RB2、IFBench）中 DTV-Loo 的 in-domain 优势并未完全转移到外部 benchmark。
4. **计算开销在高序列长度场景显著**：如 AIME 上每 completion 需单独计算梯度，训练时长从 120–144h 升至 168–192h，单步开销与 seq_len 正相关。
5. **未显式探索 DTV-λ 连续族**：作者保留了广义 λ 参数化，但因两极端已表现良好而未展开超参搜索，可能对特定任务有益。
6. **未考虑更复杂的偏好噪声模型**：Mismatch 构造仅为 rank 翻转与 cross-response flip，现实 human-in-the-loop 噪声形态更丰富。

## 研究启发与可借鉴点
1. **梯度对齐作为"内在参考信号"**：用 batch 自身梯度均值替代外部验证集，提供了一种不依赖额外数据的样本估值范式，可移植到监督/自监督学习的数据筛选中。
2. **自保护效应的形式化揭示**：将 influence/alignment score 分解为自项+交叉项，并通过留一法剥离自项，这种"控制变量式"的消融设计对理解各类 score-based selection 方法具有通用参考价值。
3. **零阈值自适应过滤的工程实践**：相比固定比例，零阈值能在训练前期更保守、后期更激进，是一种潜在的"自调节训练稳定性"机制，值得在长程 RL 或 SFT 中测试。
4. **与已有基线的公平对比设计**：与 LearnAlign/GradAlign 同框架重实现并在相同 seed/budget 下比较，使得结论可信度更高；该方法论可作为后续对比实验的模板。
5. **噪声鲁棒性 benchmark 的构建思路**：通过 Mismatch 比例系统评估方法在噪声偏好/奖励下的稳定性，为未来 RLHF/RLVR 的 robustness 评测提供了可复用的协议。

## 关键术语表
- **Dynamic Trajectory Valuation (DTV)**：通过计算每个训练单元梯度与同 batch 内平均梯度的内积，并以 0 为阈值动态保留/丢弃的轨迹估值方法。
- **DTV-Loo**：DTV 的留一法变体，排除被评估样本自身梯度后计算参考方向，从而消除 self-protection effect。
- **Self-protection effect**：DTV 分数中自项 $s_j = \|g_j\|^2/b$ 恒为非负，可能掩盖负的交叉项，使明显有害的单元被错误保留。
- **Majority Gradient Alignment**：核心假设——batch 中多数单元梯度构成当前优化参考方向，少数负对齐者有害。
- **Training unit**：一次优化中参与梯度计算的基本实体，因算法而异（PPO 的 trajectory segment、GRPO 的 completion、DPO 的 preference pair）。
- **Total-loss mask**：被 DTV 过滤掉的 training unit 所关联的所有 transition/pair 均从 loss 中移除。
- **Leave-one-out normalization**：对 DTV 均值的归一化方式，使被评估样本不出现在参考方向中。
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：具备可验证答案的 RL 训练设定，常见于数学推理等。

## 可复现要素
- **数据集**：MiniGrid（标准开源）、GSM8K（官方 split）、AIME 2024（HuggingFace H4）、UltraFeedback（HuggingFaceH4/ultrafeedback_binarized）、DeepScaleR-Preview-Dataset。
- **代码/权重**：论文使用 JAX + Tunix（Google 开源），扩展了 vectorized per-unit 梯度与动态过滤；具体仓库链接在文中未显式给出（论文未提及独立 GitHub）。
- **关键超参**：
  - PPO：clip ε=0.2、discount γ=0.99、GAE λ=0.95、lr 5e-3（SGD）/ 3e-4（Adam）等（见 Appendix B Table 5）。
  - GRPO-GSM8K：lr 1e-6、KL coeff=0.08、clip 0.2/0.2、4 个 prompt×4 个 completion；AIME：lr 1e-6、max_response=8192、eval 32K、128 prompt/step、8 completion/prompt。
  - DPO：Qwen2.5-1.5B + LoRA(r=64,s=64)、β=0.01、lr 1e-6。
- **硬件**：TPU v5p-8（多数）、TPU v5p-16（AIME）。
