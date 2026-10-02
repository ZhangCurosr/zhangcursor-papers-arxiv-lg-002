---
title: "NOT-ALL-ROLLOUTS-ARE-WORTH-LEARNING-ON-TRAJECTORY-VALUATION"
source: https://arxiv.org/pdf/2609.35072v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:36:13"
field: "强化学习数据筛选与后训练优化"
keywords: ["trajectory valuation", "reinforcement learning", "influence function", "post-training", "PPO", "GRPO", "DPO"]
innovations: ["提出仅基于 mini-batch 梯度的动态轨迹估值框架 DTV，无需验证集或影响函数近似", "推导留一法变体 DTV-Loo 并揭示自保护效应，为不同场景提供过滤策略选择依据", "在 PPO/GRPO/DPO 三种范式上统一验证，在 AIME 上 Pass@16 提升 +0.1000、DPO Mismatch-40% 准确率提升 +0.0339"]
benchmarks: ["MiniGrid", "GSM8K", "AIME 2024", "UltraFeedback"]
---

# 论文速读：NOT-ALL-ROLLOUTS-ARE-WORTH-LEARNING: ON TRAJECTORY VALUATION FOR POST-TRAINING REINFORCEMENT LEARNING

## 一句话总结
本文提出 **Dynamic Trajectory Valuation (DTV)**，一种基于 mini-batch 梯度对齐的动态轨迹估值框架，无需预设验证集或影响函数近似，仅凭当前批内梯度即可在线识别并过滤有害轨迹单元；实验表明 DTV 及其变体 DTV-Loo 在 PPO、GRPO、DPO 三种强化学习范式上均能一致提升性能与训练效率。

## 研究问题与动机
- **动态生成的轨迹缺乏固定验证信号**：分类任务依赖静态训练/验证集进行数据估值，而 RL 中的轨迹通过 Agent 在线生成，不存在固定的验证集，传统影响函数（Influence Function）方法无法直接迁移。
- **轨迹效用具有策略依赖性与时序信用分配复杂性**：RL 中轨迹的价值由当前策略、时序回报及奖励动态共同决定，难以定义一致的影响度量。
- **现有 RL 数据筛选方法泛化性不足**：已有工作（如 IIF、LearnAlign、GradAlign）依赖特定假设（回放缓冲区、可验证反馈、可信验证梯度等），跨不同 RL 范式（PPO/GRPO/DPO）的通用性存疑。
- **计算效率与在线估值的矛盾**：传统影响函数需要 Hessian 逆矩阵的近似（如 LiSSA、EKFAC 等），在 RL 的高维场景下开销巨大，难以支持在线动态过滤。

## 核心贡献（创新点）
1. **提出统一的形式化框架**：首次将影响函数思想推广到通用 RL 设置，统一处理动态数据生成、无验证信号和策略依赖效用三大挑战。
2. **设计 DTV 方法**：仅基于当前 mini-batch 的梯度信息估计轨迹效用，以零阈值动态过滤负面对齐的轨迹单元，无需额外辅助信号或预定义过滤率。
3. **推导 DTV-Loo 变体与"自保护效应"**：通过留一法（Leave-One-Out）分解 DTV 得分，揭示自项（self-term）可能掩盖交叉冲突的现象，提出去除自项的 DTV-Loo 作为更严格的默认选择。
4. **跨三种主流 RL 范式的统一验证**：在 PPO（MiniGrid）、GRPO（GSM8K/AIME）、DPO（UltraFeedback）上均一致提升性能与训练效率，验证了方法的通用性。

## 方法详解
**核心思想**：在 mini-batch 内，大多数样本的梯度共同构成局部参考方向，与主流方向负对齐的样本视为有害，予以过滤。

**DTV 评分公式**：
$$\mathcal{T}^{\mathrm{DTV}}(z_j;\theta^{(t)}) = \frac{1}{b}\sum_{i=1}^{b}(g_i^{(t)})^\top g_j^{(t)}$$
其中 $g_j^{(t)} = \nabla_\theta \ell(z_j;\theta^{(t)})$ 为训练单元 $z_j$ 的梯度贡献，$b$ 为 batch 大小。

**得分分解**：
$$\mathcal{T}^{\mathrm{DTV}} = s_j^{(t)} + c_j^{(t)}, \quad s_j^{(t)}=\frac{1}{b}\|g_j^{(t)}\|^2, \quad c_j^{(t)}=\frac{1}{b}\sum_{i\neq j}(g_j^{(t)})^\top g_i^{(t)}$$
- **自项（self-term）** $s_j^{(t)}$：反映该单元自身的梯度模长，提供"自保护"作用——即使与 batch 负对齐，强自项也可抵消负面交叉项。
- **交叉项（cross-term）** $c_j^{(t)}$：衡量该单元与其余单元的梯度一致性。

**DTV-Loo（留一法变体）**：
$$\mathcal{T}^{\mathrm{DTV-Loo}}(z_j;\theta^{(t)}) = \frac{1}{b-1}\sum_{i\neq j}(g_i^{(t)})^\top g_j^{(t)}$$
完全去除自项，仅依赖交叉项进行过滤，避免单一异常单元主导 batch 估值。

**操作规范**：
- **PPO**：训练单元 = 轨迹段（episode 终止或 rollout 边界划分），过滤时连同所有 transition 一起从目标函数中移除。
- **GRPO**：训练单元 = 单个 completion trajectory（同 prompt 下的一个生成结果）。
- **DPO**：训练单元 = 单个 prompt–response 偏好对。
- 过滤后采用 total-loss mask，将保留单元的损失归一化后执行优化器更新。
- 不使用预定义过滤率，而是采用固定的零值阈值，允许有效过滤比例在训练中自适应变化。

## 实验与结果
**实验设置**：基于 JAX + Tunix 实现，在 PPO（MiniGrid）、GRPO（GSM8K / AIME 2024）、DPO（UltraFeedback）三个场景下评估。

### PPO 实验（MiniGrid）
- 在 Empty-8x8 和 DoorKey-8x8 两个稀疏奖励网格环境中测试。
- **主要结果**：DTV-Loo 在两种环境上均比 Vanilla PPO 和 IIF 更快收敛；DoorKey-8x8（更具挑战）上的性能提升尤为显著，最终策略性能更稳定。

### GRPO 实验（GSM8K）
- 模型：Gemma-3-1B-IT（LoRA 微调），数据集含 Clean 和 Mismatch-20% 两种设置（后者反转组内奖励排序）。
- **Clean 设置**：DTV 精确准确率 **0.5483**（ΔAcc=+0.0760），优于 Vanilla GRPO（0.4828）和两基线（LearnAlign 0.5014、GradAlign 0.4876）。
- **Mismatch-20%**：DTV 精确准确率 **0.5413**（ΔAcc=+0.0690），远超 LearnAlign（+0.0140）和 GradAlign（+0.0052）。
- DTV 略优于 DTV-Loo，表明在此低冲突场景下保留自项有益。

### GRPO 实验（AIME 2024）
- 模型：DeepSeek-R1-Distill-Qwen-1.5B（全参数微调），每个 prompt 采样 8 个 completion。
- **主要结果**（相对预训练模型）：DTV-Loo 的 Pass@1 提升 **+0.0270**，Pass@16 提升 **+0.1000**，Maj@16 提升 **+0.1334**，全部指标最优；Extractable 达 **0.7521**，抽取后准确率 **0.1773**，均超过 Vanilla GRPO 和 DTV。

### DPO 实验（UltraFeedback）
- 模型：Qwen2.5-1.5B（SFT 后再 DPO），测试 Clean / Mismatch-20% / Mismatch-40% 三种偏好污染程度。
- **Mismatch-40% 下**：DTV-Loo 的 AUC 达 **0.5406**（vs. Reward 10% 的 0.5132，+0.0274），最终准确率 **0.5196**（vs. 0.4857，+0.0339），在所有条件下均为最优，且与 Reward 10% 的差距随污染加剧而扩大（具统计显著性）。
- **训练效率**：在污染设置下，DTV-Loo 达到目标准确率（T95）仅需约 **30 分钟**，显著快于 Vanilla DPO；Random 10% 在 Mismatch-40% 下未能完成训练。
- **DTV vs DTV-Loo 选择建议**：在低冲突场景（如 GSM8K）中两者表现接近；在高噪声/复杂场景（如 DPO/GRPO-AIME）中 DTV-Loo 更优，推荐作为默认选择。

## 相关工作脉络
1. **IIF (2025)**：基于回放缓冲区和算法特定代理目标的过渡级影响估计，需手动指定过滤率，仅适用于 PPO 类场景；DTV 以轨迹为单位、无需辅助信号和过滤率调参，泛化性更强。
2. **LearnAlign (2026)**：利用可验证 ground-truth 反馈估计 learnability 并加权梯度对齐，需额外 rollout 估计 learnability，且假设存在固定数据集（RLVR 设置）；DTV 不依赖外部反馈信号，可在纯在线交互场景中使用。
3. **GradAlign (2026)**：使用可信验证集构建参考梯度进行 prompt 级在线选择，需要额外 rollout 估计验证梯度，且针对 GRPO 设计；DTV 完全基于当前 batch 梯度自洽估计参考方向，无需额外资源。
4. **经典影响函数（LiSSA/EKFAC/DataInf/TRAK 等）**：基于监督学习的静态 Hessian 逆近似方法，依赖固定训练/验证集；本文将其推广到 RL 动态场景，彻底避免 Hessian 计算。
5. **动态数据估值（One-Training-Run / Layer-aware Influence）**：在 checkpoint 间聚合影响估计或直接每步估计，但仍面向分类场景；本文的 DTV 专为 RL 的在线、动态、无验证集特性设计，仅用 JAX 向量化自动微分实现高效估值。
6. **随机/基于奖励的启发式筛选**：Random Filtering 和 Reward-based Filtering 是常见基线，但对预定义过滤率高度敏感且不稳定；DTV/DTV-Loo 通过零阈值自适应过滤，无需手动调参。

## 局限性与未来方向
- **极端异常值鲁棒性**：DTV 的 batch 均值参考方向可能受极端样本主导（Section 5.2 已提及但 GSM8K 未出现此问题；DPO 实验中 DTV-Loo 成功规避了此问题）。
- **未引入 DTV-λ 参数的系统调优**：尽管提出了广义 DTV-λ 家族（λ∈[0,1]），但实验仅使用 λ=1（DTV）和 λ=0（DTV-Loo）两个端点，中间值未充分探索。
- **AIME 实验仅单 seed**：受限于计算成本（>140 小时/方法），AIME 结果来自单次训练运行，统计置信度低于其他多 seed 实验。
- **下游指令跟随能力的迁移有限**：表 8 显示 DTV-Loo 在 UltraFeedback 上的强域内提升未在所有外部下游基准（LiveBench-IF、RB2、IFBench）上均匀转移，泛化边界有待进一步探索。
- **未来方向**：可探索更精细的 λ 自动搜索机制、将 DTV 扩展到多轮对话 RL、以及在更大规模 LLM post-training（如 o1/Olympiad 级别）上验证有效性。

## 研究启发与可借鉴点
1. **无辅助信号的在线梯度对齐估值**：DTV 的核心思路——用当前 batch 梯度均值作为自洽参考方向——是一个简洁通用的技巧，可直接迁移到任何需要在线数据筛选的优化场景（如 online meta-learning、continual learning）。
2. **自项 vs 交叉项的分解分析框架**：将影响分数分解为自项和交叉项，并由此导出"自保护效应"的分析方法，为理解不同过滤策略的行为差异提供了清晰的理论透镜，可复用于其他数据筛选工作的消融分析。
3. **零阈值替代固定过滤率的设计哲学**：以零值作为自然决策边界，而非预设固定比例，使过滤机制能够自适应训练动态，这一设计思想值得在更多数据筛选/课程学习中借鉴。
4. **JAX 向量化自动微分的高效实现**：通过 JAX 直接计算训练单元级梯度（而非 batch 级近似），实现了高效且精确的估值，为后续工作在大规模语言模型上的部署提供了工程参考。
5. **跨 PPO/GRPO/DPO 的统一评估范式**：在三种不同粒度的 RL 范式（transition-level / completion-level / pair-level）上均验证同一方法，展示了极强的通用性，可作为未来 work 的实验设计标杆。

## 关键术语表
**Dynamic Trajectory Valuation (DTV)**：基于 mini-batch 梯度对齐的动态轨迹估值方法，以零阈值为标准在线过滤与主流方向负对齐的训练单元。

**Self-protection Effect（自保护效应）**：DTV 中非负的自项（self-term）可能掩盖负交叉项，导致明显有害的训练单元被错误保留的现象。

**Leave-One-Out (Loo) 变体**：DTV-Loo，通过排除被评估单元自身梯度来重构参考方向，完全去除自项以实现更严格的交叉对齐过滤。

**Training Unit（训练单元）**：不同 RL 范式中参与单次梯度更新的基本实体，PPO 中为轨迹段、GRPO 中为单个 completion、DPO 中为偏好对。

**Majority Gradient Alignment（主流梯度对齐假设）**：假设 mini-batch 内大多数样本的梯度构成局部参考方向，与之负对齐的样本视为有害。

**Total-Loss Mask**：过滤后仅保留正分单元的完整损失（包括 policy loss、value loss、KL 正则等所有项），并对保留单元数归一化的更新方式。

**DTV-λ 家族**：通过权重参数 λ∈[0,1] 统一 DTV（λ=1）和 DTV-Loo（λ=0）的更广义形式，允许在自项保留与去除之间连续调节。

## 可复现要素
- **数据集**：MiniGrid（Empty-8x8、DoorKey-8x8）、GSM8K、AIME 2024（DeepScaleR-Preview-Dataset）、UltraFeedback；均为公开数据集。
- **代码/权重开源**：基于 JAX 和 Google Tunix 库实现（见参考文献 [34][35]），论文未声明额外代码仓库链接。
- **关键超参**：PPO clip ε=0.2、GAE λ=0.95；GRPO clipping=0.2/0.28、KL coeff=0.08/0.001；DPO β=0.01；LoRA rank=64、scaling=64；学习率 1e-6~1e-5；详见附录 B/C/D 表格。
