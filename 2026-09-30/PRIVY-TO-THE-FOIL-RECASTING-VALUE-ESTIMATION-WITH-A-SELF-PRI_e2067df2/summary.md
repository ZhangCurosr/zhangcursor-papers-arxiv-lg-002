---
title: "PRIVY-TO-THE-FOIL-RECASTING-VALUE-ESTIMATION-WITH-A-SELF-PRI"
source: https://arxiv.org/pdf/2609.37825v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 18:34:14"
field: "大语言模型强化学习"
keywords: ["reinforcement learning", "value estimation", "actor-critic", "privileged information", "mathematical reasoning", "LLM post-training"]
innovations: ["将价值估计重铸为信息问题，揭示状态唯一预测的贝叶斯误差下界", "提出πPPO框架，通过留目标对比上下文增强critic，保持actor条件不变且无需额外生成", "证明特权价值估计为合法基线，并支持非对称小critic仍优于全尺寸critic基线"]
benchmarks: ["AIME 2024", "AIME 2025", "BeyondAIME", "HMMT Feb", "HMMT Nov", "GPQA-Diamond", "MMLU-Pro"]
---

# 论文速读：PRIVY TO THE FOIL: RECASTING VALUE ESTIMATION WITH A SELF-PRIVILEGED CRITIC FOR RLVR

## 一句话总结
论文提出 πPPO，将强化学习验证推理（RLVR）中的价值估计重新构建为信息问题，通过复用已验证的同提示 rollouts 作为对比证据增强 critic，从而在保持 actor 训练不变的前提下显著提升价值估计质量与端到端数学推理性能。

## 研究问题与动机
- **核心问题**：在稀疏终端奖励的多步推理任务中，actor-critic 方法依赖 critic 进行 token 级信用分配，但价值估计误差会扭曲优势函数并破坏训练稳定性。
- **现有方法不足**：
  1. 传统 PPO/VAPO 等仅依赖状态本身预测价值，无法利用训练过程中已生成的对比轨迹。
  2. 已有改进（如更优初始化、额外续采样的目标）仍假设 critic 必须独立完成两项任务——评估进展与预测未来行为。
  3. 无 critic 方法（GRPO/DAPO）虽能降低方差，但缺乏状态条件基线，难以精细分配中间步骤信用。
- **动机**：从信息角度重新审视价值估计，证明引入额外证据可降低贝叶斯误差，从而设计一种无需额外生成即可提升 critic 质量的训练机制。

## 核心贡献（创新点）
1. **概念重铸**：将价值估计视为信息问题，证明状态唯一预测的残余误差源于未解析的回报不确定性，而非单纯优化瑕疵。（与以往工作本质区别：首次从信息论视角形式化对比 critic 与标准 critic 的下界差距。）
2. **πPPO 框架**：提出留目标对比上下文（leave-target-out context），复用同 prompt 已验证 rollouts 作为 positive/negative 参考，构造特权价值函数 $V_\phi^+(s,\mathcal{I})$。（与自蒸馏方法本质区别：特权信息仅用于 critic，不改变 actor 条件与奖励目标，保持训练-推理一致性。）
3. **理论保证**：证明特权价值估计仍为合法策略梯度基线，不引入额外偏差；且可支撑更小参数的不对称 critic 仍优于 actor-sized critic 的基线方法。（与现有非对称 critic 工作本质区别：通过对比证据补偿容量损失，而非仅依靠架构改进。）
4. **系统实验**：在 Qwen3-4B/8B 上于五个数学推理基准全面超越 GRPO、DAPO、PPO、VAPO，并在更小 critic 设置下保持领先。（与相关基线本质区别：首次在 actor-critic 范式中显式利用同分布已验证轨迹作为 contrastive evidence。）

## 方法详解
- **留目标对比上下文构建**：对每个 prompt $x$，旧策略采样 $G$ 个 rollouts $\{y^{(j)}\}$ 并经 verifier 标记。对目标 rollout $y^{(i)}$，从其兄弟中按标签划分：
  $$\mathcal{C}_i^+ = \{y^{(j)}: j\neq i, r(x,y^{(j)})=1\},\quad \mathcal{C}_i^- = \{y^{(j)}: j\neq i, r(x,y^{(j)})=0\}$$
  特权上下文 $\mathcal{I}^{(i)}$ 由采样参考拼接而成：若正负样本均存在则取各一个正样本与一个负样本；若缺负样本则取两个正样本；若缺正样本则取两个负样本并附加标准答案 $y^*$。
- **特权优势估计**：Critic 前向计算 $V_\phi^+(s_t^{(i)},\mathcal{I}^{(i)})$，其中状态 $s_t=(x,y_{<t})$ 与 actor 相同。TD 残差与 GAE 照常计算：
  $$\delta_t^{+(i)} = r_t^{(i)} + \gamma V_\phi^+(s_{t+1}^{(i)},\mathcal{I}^{(i)}) - V_\phi^+(s_t^{(i)},\mathcal{I}^{(i)}),\quad \widehat{A}_t^{+(i)}=\sum_{\ell=0}^{T_i-t}(\gamma\lambda)^\ell\delta_{t+\ell}^{+(i)}$$
  Actor 更新仍使用 clipped PPO 目标，但优势替换为 $\widehat{A}_t^{+(i)}$。
- **合法性与一致性**：因特权上下文 $\mathcal{I}$ 仅输入 critic，与当前动作 $a_t$ 独立，故 $\mathbb{E}[\nabla_\theta\log\pi_\theta(a_t|s_t)V_\phi^+(s_t,\mathcal{I})]=0$，保证无偏基线；actor 在训练与推理时均仅依赖 $(x,y_{<t})$，维持部署接口不变。
- **训练细节**：critic 预训练 50 步后进入 RLVR 阶段，actor learning rate $10^{-6}$，batch size 256，每 prompt 采样 8 个 rollout，最大长度 8192 tokens，$\gamma=\lambda=1$，Clip-Higher ($\varepsilon_{\text{low}}=0.2,\varepsilon_{\text{high}}=0.28$)。

## 实验与结果
- **数据集**：数学推理基准 AIME 2024、AIME 2025、BeyondAIME、HMMT Feb、HMMT Nov；分布外评估使用 GPQA-Diamond 与 MMLU-Pro。
- **基线**：Critic-free（GRPO、DAPO）、Actor-critic（PPO、VAPO）。
- **主要结果（Table 1）**：
  - Qwen3-4B：πPPO 整体平均 50.3%，超越最佳基线 DAPO（46.2%）4.1 个百分点，超越 PPO（43.5%）6.8 个百分点；五项基准均最高。
  - Qwen3-8B：πPPO 整体平均 51.6%，超越 DAPO（49.3%）2.3 个百分点，超越 PPO（47.9%）3.7 个百分点。
- **价值估计质量（Figure 2）**：πPPO 的 explained variance (EV) 始终高于 PPO/VAPO，价值函数损失更低且更早稳定。
- **非对称 critic（Figure 3）**：将 critic 缩小至 0.6B（对应 4B actor）和 1.7B（对应 8B actor）时，πPPO 仍分别取得 47.8% 和 49.9% 整体准确率，优于 actor-sized critic 的 PPO；EV 仅下降 0.037/0.027，而 critic 参数减少约 85%/79%。
- **消融（Table 2）**：移除正确性标签（w/o labels）、改为同极性参考（w/ same polarity）或仅用标准答案（w/ GT only）均导致显著性能下降，验证对比结构的关键作用。
- **分布外泛化（Table 4）**：在 GPQA 与 MMLU-Pro 上，πPPO 在 4B 取得 60.5%、8B 取得 65.2%（非对称）最佳整体准确率。

## 相关工作脉络
1. **标准 Actor-Critic（PPO/VAPO/VC-PPO）**：依赖状态唯一价值函数，未利用同 prompt 已验证轨迹作为对比证据；本文从信息角度揭示其固有局限并提供补充途径。
2. **无 Critic 方法（GRPO/DAPO）**：通过组内相对优势消除价值函数，但缺乏状态条件基线，无法精细分配中间步骤信用；本文在保留标准 PPO 接口的前提下提升价值估计质量。
3. **特权信息学习（LUPI）**：传统 LUPI 将教师额外知识注入输入或响应前缀，易导致训练‑推理分布偏移；本文特权信息仅作用于 critic，保持 actor 条件与环境奖励不变。
4. **在线自蒸馏（OPSD/RLSD/RLCS）**：使用模型自身生成的参考轨迹进行 token-level 监督，但可能偏离原始奖励最大化目标；本文对比上下文来自已验证轨迹，直接对齐二元奖励信号。
5. **非对称 Critic 设计**：以往工作多通过更好初始化或更丰富目标弥补小 critic 容量；本文证明引入对比证据可大幅缓解容量需求，实现更强的性价比。

## 局限性与未来方向
- **二元奖励假设**：理论推导与实验均基于 verifiable 的二元奖励（0/1），对连续或稀疏奖励的推广有待验证。
- **上下文构造依赖 sibling 标签**：当正/负样本全缺失时仅能回退到单一标签或标准答案，极端情况下对比强度减弱。
- **计算开销**：虽无额外 rollout 生成，但 critic 前向需处理带上下文的序列，可能增加显存与延迟（论文未量化）。
- **未来方向**：可扩展至多步验证、混合奖励信号；探索更高效的上下文采样策略；与其他 credit assignment 技术（segment-level、continuation modeling）结合。

## 研究启发与可借鉴点
1. **信息视角的价值估计**：将 critic 性能瓶颈归因于信息不足而非单纯容量，为设计更轻量但高效的 value network 提供新范式。
2. **留目标对比结构**：leave-target-out 上下文避免数据泄露，同时保证每个 trajectory 既作为训练目标又作为参考，可在其他 online RL 场景中复用。
3. **训练‑推理一致性保障**：特权信息仅进 critic 的设计原则可作为“安全”增强手段，在不改变部署接口的情况下提升训练动态。
4. **非对称 critic 的可行性**：证明通过信息补充可显著压缩 critic 规模，为资源受限场景下的 actor-critic 部署提供新思路。
5. **可迁移到多模态/代理推理**：若任务具备可验证子目标或相似轨迹组，该框架可自然扩展至代码生成、 agent planning 等领域。

## 关键术语表
- **πPPO**：Privileged-Information PPO，本文提出的强化学习框架，通过特权上下文增强 critic 价值估计。
- **特权信息（Privileged Information, PI）**：训练中可用但在推理时不可用的额外数据，本文指同 prompt 已验证 rollouts。
- **留目标对比上下文**：构造特权信息时排除当前目标轨迹，仅使用其余兄弟轨迹作为参考，防止信息泄露。
- **价值估计explained variance（EV）**：衡量 critic 对回报目标变异的解释比例，越高表示价值预测越准确。
- **actor-critic**：同时优化策略网络（actor）与价值网络（critic）的强化学习架构，critic 提供基线以减少梯度方差。
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：针对可通过规则验证的任务（如数学推理）设计的强化学习训练范式。
- **优势函数（Advantage）**：token 级奖励相对于基线的超额收益，由 critic 估计的 value 与 TD 残差计算得到。
- **Clip-Higher**：PPO 变体，对不同方向的裁剪系数不对称（此处 ε_low=0.2, ε_high=0.28），以放宽正优势方向的策略更新限制。

## 可复现要素
- **数据集**：DAPO-17K（训练集）、AIME 2024/2025、BeyondAIME、HMMT Feb/Nov、GPQA-Diamond、MMLU-Pro（评估集）；论文未明确声明训练数据公开状态，但基准均可从公开仓库获取。
- **代码/权重**：论文未提供开源链接，基于 VeRL codebase 实现；模型权重未公布。
- **关键超参数**：batch size=256，actor learning rate=1e-6，critic learning rate=1e-5（预热后 1e-5），temperature=1.0，top-p=1.0，top-k disabled，每 prompt rollout数 G=8，最大响应长度=8192 tokens，critic预训练50步，γ=λ=1，Clip-Higher ε_low=0.2、ε_high=0.28，无 KL 正则化。
