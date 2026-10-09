---
title: "LEARNING-UNKNOWN-CONSTRAINTS-WITHOUT-UN-SAFE-DATA-VIA-OPTIMA"
source: https://arxiv.org/pdf/2610.09350v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:46:30"
field: "安全约束学习"
keywords: ["约束学习", "逆最优控制", "KKT条件", "对抗正则化", "离线强化学习", "安全控制"]
innovations: ["提出CF-KKT框架，通过KKT最优性条件与合成对抗轨迹结合，无需违反数据即可学习未知神经约束", "证明正值增强优势蕴含不可行性（Lemma 1），为对抗筛选提供理论保证", "将CIOC与ICRL桥接：学习动力学支持KKT推理，同时保留神经网络约束表示的灵活性"]
benchmarks: ["MuJoCo locomotion (Ant_ls, BlockedAnt, BlockedHopper, HalfCheetah, Walker_ls)", "2D continuous maze navigation (Obstacle, Wide, Disk, Zigzag)", "ICSDICE benchmark"]
---

# 论文速读：LEARNING-UNKNOWN-CONSTRAINTS-WITHOUT-UN-SAFE-DATA-VIA-OPTIMA

## 一句话总结
本文提出 Counterfactual KKT (CF-KKT) 框架，利用从专家演示中学习到的局部可微动力学模型，结合 KKT 最优性条件与合成奖励改进的对抗轨迹，在无需任何已发生约束违反的真实数据的情况下，恢复未知的神经表征约束。该方法桥接了约束逆最优控制（CIOC）的数据效率与逆约束强化学习（ICRL）的表示灵活性。

## 研究问题与动机
- **核心问题**：如何仅从满足约束的专家最优演示中学习未知安全约束，同时不依赖额外的在线探索或约束违反数据？
- **现有 CIOC 方法局限**：依赖已知的闭环系统动力学和结构化约束参数化形式，无法应用于高维黑箱系统；高斯过程扩展虽放松了参数化假设但仍需已知动力学且扩展性差。
- **现有 ICRL/离线 ICRL 方法局限**：通常需要大量在线探索（存在安全风险），计算与数据成本高；即便是离线方法 ICSDICE 仍依赖可能包含约束违反行为的次优数据集来识别不安全区域。
- **理论保证缺失**：ICRL 类方法缺乏约束恢复精度的理论保证，难以满足安全关键场景的可靠性需求。

## 核心贡献（创新点）
1. **CIOC + 学习动力学的参数化约束恢复框架**：当约束具有已知闭式参数化时，先用专家数据拟合局部可微动力学模型（无需额外演示），再通过 CIOC 混合整数规划（MIP）恢复未知约束参数。
2. **Counterfactual KKT (CF-KKT) 算法**：针对神经网络表征的未知约束，提出将 KKT 启发式最优性条件与模型生成的对抗轨迹相结合的约束学习方法，从根本上避免了收集约束违反数据的需要。
3. **正对抗优势的不可行性理论保证**：证明若一条对抗轨迹的值增强回报严格超过专家基线，则根据 Lemma 1，该轨迹必然在某时刻违反约束（即"奖励改进的对抗行为蕴含不可行性"）。
4. **对动力学误设的灵敏度分析**：提供 MIP 约束恢复对动力学参数扰动的理论敏感度上界（Proposition 1），由隐函数定理推导，量化学习动力学误差对约束参数恢复的影响。
5. **系统性实验验证**：在低维连续迷宫和高维 MuJoCo 机器人控制任务上，相对于离线 ICRL 基线实现更优的约束几何恢复与下游安全性，同时在结构上使用的数据更少。

## 方法详解

**总体架构**：两步走——先利用专家转换数据 $(x_t, u_t, x_{t+1}) \in \mathcal{D}_E$ 学习局部可微动力学模型 $\hat{f}(\cdot,\cdot;\psi)$，再基于该模型进行约束学习。

**两种设置**：

**① 已知参数化约束（MIP 方法）**：
- 构建含学习动力学的 Lagrangian：
  $$\hat{L} = \sum_t -r(x_t, u_t) + \sum_t \lambda_t \hat{c}(x_t, u_t; \theta) + \sum_t \nu_t^\top(x_{t+1} - \hat{f}(x_t, u_t; \psi))$$
- 以最小化 stationarity residual 为目标，求解混合整数规划（big-M 编码互补约束），联合恢复约束参数 $\theta$ 和对偶变量。

**② 神经约束（CF-KKT）**：
- **逐转移辅助乘子**：将轨迹级 KKT 松弛为 transition-wise 形式，用有界辅助乘子 $\tilde{\lambda}_t = \lambda_{\max}\sigma(\hat{c})$ 和 $\tilde{\nu}_t = \nu_{\max}\tanh(\eta_t)$ 替代真实对偶变量，确保训练数值稳定。
- **KKT-informed 损失**：
  $$\mathcal{L}_{\text{KKT}} = \mathbb{E}[\|\nabla_{x,u}\hat{L}_t\|^2 + w_p\max(\hat{c},0)^2 + w_c(\tilde{\lambda}_t\hat{c})^2 + w_a|\hat{c}|]$$
  四项分别对应：stationarity residual、primal feasibility、complementary slackness、activity regularizer。
- **合成对抗轨迹生成**：从专家轨迹中采样窗口 $H$，对每个窗口的控制序列施加高斯扰动 $u_h^{(k)} = \text{clip}(u_{i+h}^{(d)} + \sigma\epsilon, -1, 1)$，在同一初态下用学习动力学展开 rollout。
- **值增强优势筛选**：$\hat{A}_V^{(k)} = \hat{J}_V^{(k)} - \hat{J}_{V,\text{base}}$，其中 $\hat{J}_V$ 包含截断奖励与续值函数 $\hat{V}$ 估计；筛选 $\hat{A}_V^{(k)} > 0$ 的对抗轨迹作为合成不可行样本。
- **对抗约束损失**：
  $$\mathcal{L}_{\text{CF}} = \frac{\sum_k w_k \cdot \max\{m - \frac{1}{H}\sum_h \hat{c}(x_h^{(k)}, u_h^{(k)};\theta), 0\}^2}{\sum_k w_k}$$
  要求对抗轨迹上的平均约束值超过正边界 $m$。
- **总损失**：$\mathcal{L}_{\text{CF-KKT}} = \alpha_{\text{KKT}}\mathcal{L}_{\text{KKT}} + \alpha_{\text{CF}}\mathcal{L}_{\text{CF}}$。

**续值函数训练**：通过对专家 return-to-go $G_i^{(d)} = \sum_{j=i}^T \gamma^{j-i}r(x_j,u_j)$ 做回归拟合 $\hat{V}$，用于评估对抗轨迹终端状态的价值。

## 实验与结果

**实验环境**：
- 已知参数约束：HalfCheetah、BlockedAnt、Swimmer、Quadcopter（MuJoCo）
- 连续迷宫：Obstacle、Wide、Disk、Zigzag（2D 导航，double-integrator 动力学）
- 高维神经约束：Ant_ls、BlockedAnt、BlockedHopper、HalfCheetah、Walker_ls（Gymnasium-v4，各含隐藏状态约束）

**基线**：ICSDICE（主离线基线）、SMODICE、IQ-Learn；参数化实验中对比 Parametric ICSDICE。

**主要结果**：
- **参数恢复**（表1）：MIP 在 Swimmer 上恢复 θ=0.0484（真值0.04）、Quadcopter 完全恢复（真值64）、HalfCheetah 恢复 -2.98（真值-3.0），Parametric ICSDICE 在 HalfCheetah 上误差达 -1.23。
- **迷宫约束 IoU**（表2）：CF-KKT 平均 IoU = 0.707，ICSDICE = 0.425，SMODICE = 0.379；最大增益在 Disk（0.815 vs 0.428）和 Obstacle（0.590 vs 0.276）。
- **下游规划**（表3）：CF-KKT 在全部四个迷宫中将违反率从 ICSDICE 降至约一半：Obstacle 0.174→0.042、Wide 0.200→0.085、Disk 0.140→0.080、Zigzag 0.209→0.093，目标到达率保持高水平。
- **高维 MuJoCo**（表4）：CF-KKT 在所有五任务上均降低违反率（相对 ICSDICE 降低 34.6%–96.8%），在 BlockedAnt 和 BlockedHopper 上同时提升 return。

## 相关工作脉络
- **CIOC 系列**（Chou et al. 2018, 2020b; Rickenbach et al. 2024; Zhang et al. 2026）：利用 KKT 条件直接嵌入专家最优性；本文继承此路线但通过局部学习动力学突破已知动力学的限制。
- **ICRL 系列**（Malik et al. 2021; Liu & Zhu 2022, 2024; Quan et al. 2024 ICSDICE）：支持未知动力学但依赖在线探索或违反数据；本文在离线设置下以更少数据达到更强安全保证。
- **ICSDICE**（Quan et al. 2024）：本文主要基线，构建更高奖励的违反分布；本文通过合成对抗而非真实违反数据实现同等目标。
- **GMPC/GP 约束学习**（Chou et al. 2022）：使用高斯过程放松参数化假设但仍需已知动力学；本文用 NN 学习动力学替代。
- **Hit-and-Run 对抗生成**（Chou et al. 2019）：亦生成对抗样本但依赖显式参数约束和解析动力学；CF-KKT 与之不同，使用学习动力学并在未知约束下工作。
- **Uncertainty-aware ICRL**（Xu & Liu 2024; Subramanian et al. 2024）：通过不确定性估计减少违反风险但仍需在线数据；CF-KKT 完全无需在线交互。

## 局限性与未来方向
- **环境依赖的超参选择**：损失权重 $w_p, w_c, w_a$ 及对抗边界 $m$ 需针对不同任务手动调节。
- **已知可微奖励假设**：KKT stationarity 显式依赖 $\nabla_{x,u}r$；虽附录 B.2 证明可用带梯度正则化的 NN 近似部分缓解，但精确梯度仍理想。
- **逐转移 KKT 忽略跨时间耦合**：transition-wise 松弛放弃了由 dynamics multiplier $\nu_t$ 产生的跨步关联，论文指出全轨迹版本在 BlockedAnt/HalfCheetah 上导致 Inf/NaN。
- **对抗质量对远离演示区域的动力学精度敏感**：学习动力学在外推区域的误差会直接影响对抗轨迹的可靠性（Lemma 1 的假设在现实中无法精确满足）。
- **未来方向**：自动化超参搜索、奖励与动力学不确定性建模、将 cross-time coupling 以稳定方式纳入训练。

## 研究启发与可借鉴点
- **"正奖励增益蕴含不可行性"的思想可直接迁移**：任何从最优演示学习安全边界的方法均可借用此判别准则生成合成负样本，避免高风险探索。
- **过渡到逐转移 KKT 的工程策略**：将轨迹级 stationarity 松弛为 transition-wise 形式以支持 mini-batch 训练是一个实用的工程技巧，值得在其他 KKT-informed 学习中参考。
- **有界辅助乘子设计**：用 sigmoid/tanh 参数化对偶变量（$\tilde{\lambda}, \tilde{\nu}$）替代自由对偶变量，是保证训练稳定的有效手段，可与其它逆优化方法结合。
- **值增强优势筛选替代硬阈值**：使用 $\hat{V}$ 估计补足短 horizon rollout 的代价-to-go，比单纯用截断奖励更可靠；此设计可用于任何基于模型的对抗生成流程。
- **MIP + 学习动力学的模块化接口**：将动力学学习与约束恢复解耦为两个阶段，参数化场景下可独立替换动力学模型而保持 MIP 推理不变，便于在复杂系统中复用。

## 关键术语表
- **CF-KKT**：Counterfactual KKT，本文提出的约束学习框架，结合 KKT 最优性条件与模型生成的对抗轨迹进行约束学习。
- **CIOC（Constrained Inverse Optimal Control）**：约束逆最优控制，通过求解 KKT 条件的逆问题从专家轨迹恢复约束参数的方法类。
- **ICRL（Inverse Constrained Reinforcement Learning）**：逆约束强化学习，从约束 MDP 的专家演示中恢复未知约束的 RL 方法类。
- **KKT 条件**：Karush–Kuhn–Tucker 条件，最优控制的必要一阶条件，包括 stationarity、primal/dual feasibility 和 complementary slackness。
- **过渡值增强优势（Value-augmented Counterfactual Advantage）**：用学习续值 $\hat{V}$ 修正短 horizon 奖励后的对抗-基线回报差，用于筛选合成不可行轨迹。
- **Parametric Constraint Recovery**：已知约束参数化形式下的约束参数估计，通过 MIP 求解 inverse KKT 问题实现。
- **Bag Precision / State Precision**：对抗轨迹诊断指标；前者衡量至少含一个真违反状态的轨迹比例，后者衡量所有状态中违反状态的比例。
- **Mask Admission Rate**：被学习约束掩膜允许的次优数据比例，用于评估约束掩膜的宽松/严格程度是否校准合理。

## 可复现要素
- **数据集**：MuJoCo 实验采用与 ICSDICE (Quan et al. 2024) 相同的五个环境基准（Ant_ls, BlockedAnt, BlockedHopper, HalfCheetahWithObstacle, Walker_ls），各含 ~50,000 专家过渡和 200,000 次优过渡；迷宫实验使用自行生成的约 30,000 专家过渡。论文未提供独立代码仓库链接（arXiv 提交于 2026年10月）。
- **网络架构**：所有网络为两层 MLP（256 隐藏单元，ReLU）；约束网络、动力学模型、续值模型和下游策略均使用此架构。
- **关键超参**：Adam lr = 3×10⁻⁴，batch size = 256，γ = 0.99；KKT 权重默认 $w_p=10, w_c=0.01, w_a=0.1$；对偶界 $\lambda_{\max}=\nu_{\max}=100$；对抗生成 $B=256$ 窗口、$K=64$ 扰动序列、$\sigma=2$（MuJoCo）/0.8（迷宫）、horizon $H=24$、对抗边界 $m=1$（MuJoCo）/2（迷宫）；outer scale $\alpha_{\text{KKT}}=\alpha_{\text{CF}}=1$。
- **训练时长**：每种子约 20,000 步，约 5–10 分钟（32 核 CPU + NVIDIA L40S GPU）。
