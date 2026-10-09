---
title: "MARRYING-PRICING-AND-ADVERTISING-WITH-LLMS"
source: https://arxiv.org/pdf/2610.09985v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:52:08"
field: "LLM-driven decision making"
keywords: ["dynamic pricing", "LLM advertisement generation", "actor-critic", "LoRA adaptation", "sequential posted pricing", "reinforcement learning"]
innovations: ["在线actor-critic框架联合优化LLM广告生成与定价决策", "用critic期望收入替代观测奖励构造leave-one-out优势估计以降低Bernoulli噪声", "KL正则化防止LoRA适配导致的广告文本漂移"]
benchmarks: ["Logit demand model", "Probit demand model", "Price disclosure demand model", "Avito marketplace simulator"]
---

# 论文速读：MARRYING-PRICING-AND-ADVERTISING-WITH-LLMS

## 一句话总结
论文提出了一种在线 actor-critic 算法，通过 LoRA 适配预训练 LLM 生成广告文本，同时利用需求模型指导定价决策，实现价格与广告内容的联合优化以提升卖家收益。

## 研究问题与动机
- **价格与广告协同优化缺失**：现有工作要么只研究动态定价（忽略广告内容），要么只优化广告生成（固定价格），但广告会改变需求曲线，最优价格随之偏移，二者必须联合学习。
- **反馈信号极度受限**：卖家仅能观察到二值购买结果（买/不买），无法获知买家的真实估值或替代选择，需要在"边收收益边学习"的探索-利用权衡中推进。
- **LLM 广告生成的漂移风险**：纯优化收益会导致策略偏离预训练参考模型，损害文本流畅性与语义连贯性，需要正则化约束。
- **广告空间过大难以穷举**：自然语言广告的搜索空间远超价格离散网格，必须依靠 LLM 从历史交互中泛化。

## 核心贡献（创新点）
- **联合 actor-critic 框架**： Actor 为 LoRA 适配的 LLM 负责广告生成，Critic 为需求模型负责估计购买概率并引导定价，两者通过反馈闭环联合优化；不同于 Jiang et al. (2025) 和 Chen et al. (2025) 仅用 RL 优化广告而不考虑定价。
- **Leave-one-out  critic 基线策略梯度**：用 Critic 预测的期望收入而非观测奖励构造优势估计，有效抑制 Bernoulli 购买噪声；相比 Ahmadian et al. (2024) 的标准 RLOO 直接替换为 critic 期望值，降低了梯度方差。
- **KL-正则化防止漂移**： Actor 损失中显式加入相对于预训练参考模型的 KL 散度惩罚，保持生成文本的语言质量；借鉴 Ziegler et al. (2019) 的 RLHF 思路但应用于定价场景。
- **在真实市场模拟器上验证**：构建基于 Avito marketplace 真实数据的 demand simulator，覆盖五个商品类别，Trained 策略在 91/100 个测试商品上均优于 Reference。

## 方法详解
**整体架构**：将时间 horizon $T$ 划分为 $N = T/B$ 个批次，每个批次内 Actor 参数 $\theta_\tau$ 和 Critic 参数 $\phi_\tau$ 固定。

**Actor（广告生成）**：
- 使用 Qwen2.5-1.5B-Instruct 作为参考模型，对 attention query 和 value 投影施加 rank-8 LoRA 适配，$\Delta\theta(\psi) = (\alpha_\ell/r_\ell) B_\ell A_\ell$。
- 给定上下文 $z$，采样广告 $x \sim \pi_{\theta_\tau}(\cdot|c(z))$。

**定价规则（$\varepsilon$-greedy）**：
- 以概率 $1-\varepsilon_\tau$ 选择 $p \in \arg\max_{p\in[0,1]} p\cdot\hat{d}_{\phi_\tau}(p, x, z)$（贪婪定价）；
- 以概率 $\varepsilon_\tau$ 从探索分布 $\nu_\tau$ 采样价格。

**Critic（需求模型）**：
- 神经网络 $\hat{d}_\phi(p, x, z) = \sigma(\text{MLP}_\phi([h_{\text{Qwen}}(x),\lambda(z)p]))$，输入为冻结 LLM 提取的广告 embedding 和货币价格。
- 损失函数：二元交叉熵 + L2 锚定正则化 $\rho\|\phi - \phi_1\|_2^2$。

**Actor 更新（每 K 个批次执行一次）**：
- Leave-one-out 优势估计：
$$A_{\tau,b} = r_{\tau,b} - \frac{1}{B-1}\sum_{j\neq b} \widehat{u}_{\phi_\tau,j}, \quad \widehat{u}_{\phi_\tau,j}=p_{\tau,j}\widehat{d}_{\phi_\tau}(p_{\tau,j},x_{\tau,j},z)$$
- Actor 代理损失：
$$\mathcal{L}_{\text{actor}}(\psi)=-\frac{1}{B}\sum_{b=1}^{B}\left(A_{\tau,b}-\beta\log\frac{\pi_{\theta_\tau}(x_{\tau,b}|c(z))}{\pi_{\theta_{\text{ref}}}(x_{\tau,b}|c(z))}\right)\log\pi_{\theta_{\text{ref}}+\Delta\theta(\psi)}(x_{\tau,b}|c(z))$$
- 附录 A 证明该梯度是 KL-正则化目标 $u_\beta(q,\theta|z)$ 的无偏估计。

## 实验与结果
**实验设置**：Actor = Qwen2.5-1.5B-Instruct，LoRA rank=8，$N=500$ 批次，$B=32$，$T=16000$；Actor 学习率 $10^{-4}$，Critic 学习率 $3\times10^{-3}$，$\beta=10^{-3}$，$\rho=10^{-4}$，$K=5$。

**合成需求模型（10,000 条独立广告评估）**：

| 模型 | Trained | Reference | Δu(%) |
|---|---|---|---|
| Logit | 0.217117 | 0.205418 | **+5.69%** |
| Probit | 0.205616 | 0.195496 | **+5.18%** |
| Price disclosure | 0.027162 | 0.017416 | **+55.96%** |

- Price disclosure 模型中 55.96% 的提升完全由广告披露价格的频率增加（35.17%→98.23%）解释。

**真实市场数据（Avito，100 条广告/商品）**：
- 归一化期望收入从 0.1991 提升至 0.2107，**总体 +5.81%**（95% CI [0.0097, 0.0136]）。
- 所有五个类别均改善：Audio/Video +2.59%，Home Appliances +5.01%，Laptops +14.26%，Mobile Phones +1.98%，Tablets/E-readers +7.55%。
- Trained 在 91/100 个测试商品上优于 Reference。

**在线学习对比（Trained vs UCB）**：Trained 凭借预训练 Critic 监督和联合优化，在每条学习曲线后半段均超越 UCB。

**离线数据消融**：从 100% 降至 0% 初始 critic 数据时，收入从 0.1802 降至 0.1670（-7.32%），说明算法可在无监督初始化下继续学习，但性能下降。

**Leave-one-out 基线误差**：全量数据时 critic 基线 MSE 仅为观测奖励基线的 0.0095~0.1734 倍，验证了 critic 替代的优势估计确实显著降噪。

## 相关工作脉络
- **Sequential posted pricing**（Kleinberg & Leighton, 2003; Besbes & Zeevi, 2009）：研究从购买反馈中学习未知需求曲线的定价策略，但未涉及广告内容；本文在相同框架上扩展了广告维度。
- **动态定价 + 多臂老虎机**（Misra et al., 2019）：基于 bandit 实验的定价探索，同样固定广告；本文的 ε-greedy 定价与之类似但耦合了广告生成。
- **广告内容效应实验**（Bertrand et al., 2010）：贷款实验中随机化利率与信内容，发现内容效果等同于利率变动 2pp；本文为此提供了自动化联合优化的算法方案。
- **RL 优化广告生成**（Jiang et al., 2025; Chen et al., 2025; Wang et al., 2026）：使用历史表现或 CTR/转化率反馈训练 LLM 生成广告，但定价固定或不在优化范围内；本文首次将定价纳入 actor-critic 联合优化。
- **RLHF + KL 正则化**（Ziegler et al., 2019）：人类反馈强化学习中的参考策略约束思想；本文将其迁移至"参考未适配 LLM"以防止广告漂移。
- **RLOO 降低 REINFORCE 方差**（Ahmadian et al., 2024）：用 leave-one-out 估计优势；本文在此基础上进一步将 observed reward 替换为 critic 期望收入，额外抑制 Bernoulli 噪声。

## 局限性与未来方向
- **需求分布静态假设**：当前框架假设真实需求函数不变，现实中买家偏好会随时间漂移，critic 需跟踪移动目标。
- **单上下文设定**：实验固定单一产品上下文 $z$ 进行分析，未处理多产品批量投放的复杂交互。
- **离线数据依赖**：Critic 预训练需要一定规模的标注数据（$M=7291$），零数据时性能下降约 7.3%。
- **未处理多阶段转化漏斗**：当前只观测最终购买决策，未建模点击→加购→付款等多阶段行为。
- **价格单调性缺失**：Demand network 无价格单调性约束，低价下可能出现非理性的需求预测异常。

## 研究启发与可借鉴点
- **Leave-one-out 替代为 critic 期望值**：在 binary feedback 环境下，用 model-based 期望收入替代 observed reward 构造优势估计是通用的降噪技巧，可迁移至其他 LLM+bandit 的优化场景。
- **LoRA + KL 正则的组合策略**：以低秩适配控制参数规模，以 KL 散度约束生成分布漂移，可在任何需要"性能优化同时保持文本质量"的 LLM fine-tuning 任务中复现。
- **Price-ad coupling 的联合优化范式**：将定价（数值决策）与文本生成（离散/长程决策）统一在一个 actor-critic 框架中，为多模态/多类型联合决策问题提供了可借鉴的架构设计。
- **离线 critic 预训练 + 在线更新的双阶段流程**：先在静态数据集上拟合 demand model，再在在线交互中联合更新，可有效缓解纯在线学习的样本效率问题，适合冷启动场景。
- **跨品类可迁移性验证**：在 Avito 五个不同商品类别（Audio/Video 至 Laptops）均获得正收益，提示该方法对不同需求弹性结构的鲁棒性，可作为多领域适用的基准方案。

## 关键术语表
- **Actor-Critic 框架**：强化学习中 Actor 负责生成动作（此处为广告文本），Critic 负责评估动作价值（此处为估计购买概率）的双组件结构。
- **LoRA（Low-Rank Adaptation）**：一种高效的 LLM 微调技术，冻结预训练权重，仅训练低秩分解矩阵 $A$ 和 $B$ 的乘积更新。
- **Leave-One-Out Baseline**：在优势估计中排除当前样本自身的影响，用其余样本的均值作为基线，降低梯度估计方差。
- **KL-正则化**：在优化目标中加入策略与参考策略之间的 KL 散度惩罚项，防止策略过度偏离预训练分布。
- **Sequential Posted Pricing**：卖家依次设定价格并观察买家购买决策的学习框架，用于在未知需求曲线下最大化累积收益。
- **Demand Model / Critic**：神经网络模型，输入价格和广告特征，输出购买概率估计，用于指导定价和构造策略梯度基线。
- **ε-greedy 定价**：以概率 $1-\varepsilon$ 选择期望收入最大化的价格，以概率 $\varepsilon$ 从探索分布采样价格。
- **Normalized Expected Revenue**：将货币收入除以商品最高保留价格，使不同价位商品的可比性标准化。

## 可复现要素
- **数据集**：合成需求模型（参数可见于附录 B.2）；真实数据来自 Avito Demand Prediction Challenge (Kaggle, 2018)，包含 20,826 条 listing，五类别共 393 个产品。
- **代码/权重**：论文未提及开源代码；模型基于 Qwen2.5-1.5B-Instruct（公开权重），LoRA 适配器为训练得到。
- **关键超参**：LoRA rank=8，scaling $\alpha_\ell=16$；$N=500$ 批次，$B=32$，$K=5$；Actor LR=$10^{-4}$，Critic LR=$3\times10^{-3}$；$\beta=10^{-3}$，$\rho=10^{-4}$；温度=0.7，最大新 token=64；Critic 网络：2 层 128 ReLU + sigmoid 输出。
