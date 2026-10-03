---
title: "LEARNING-BEYOND-WHAT-YOU-SAMPLE-OFF-POLICY-AWARE-CROSS-MODEL"
source: https://arxiv.org/pdf/2609.37868v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:52:14"
field: "大语言模型强化学习"
keywords: ["RLVR", "GRPO", "cross-model trajectory exchange", "off-policy learning", "complementary rollout", "mathematical reasoning"]
innovations: ["互补且平衡的 all-fail prompt 选择与完整 peer 组转移", "序列级兼容性门控与 token 级 PPO 裁剪的双层 off-policy 控制", "source-computed advantage 保留与 peer-last 更新顺序"]
benchmarks: ["MATH500", "AIME2024", "AIME2025", "AMC23", "Minerva Math"]
---

# 论文速读：LEARNING-BEYOND-WHAT-YOU-SAMPLE-OFF-POLICY-AWARE-CROSS-MODEL-TRAJECTORY-EXCHANGE-FOR-RLVR

## 一句话总结
论文提出了 **GRAFT** 框架，通过让异构模型之间互补地交换 rollout 轨迹，解决 RLVR（可验证奖励强化学习）中全部失败组（all-fail groups）因缺乏奖励信号而无法学习的问题，使两个模型在相同 rollout 预算下均能提升推理性能。

## 研究问题与动机
- **RLVR 的探索瓶颈**：GRPO 等方法仅从自身生成的 rollout 组中学习，当某一 prompt 的所有 n 次采样回答均错误时，组内相对优势全部为零，无法提供策略梯度信号。
- **单一模型增加 rollout 的成本高**：虽然增加 n 可以提高采到正确轨迹的概率，但计算成本线性增长，且不能从根本上扩展可解 prompt 集合。
- **异构模型的互补性被浪费**：实际观察到，不同预训练历史的模型（如 SmolLM3 与 Qwen3）在所有 rollout 失败的 prompt 上存在显著互补成功率，但标准单模型 RLVR 将每个策略孤立训练，未利用这种互补成功。
- **现有跨模型方法的不足**：HACPO 在所有 prompt 上共享 peer rollout，不限于 receiver 失败场景；SGT 仅在 receiver 失败时转移单个正确 peer 响应并通过固定权重的 SFT 损失学习，缺乏兼容性加权。两者在本工作的选择 prompt 设定下均低于 GRAFT。

## 核心贡献（创新点）
1. **互补且平衡的 prompt 选择机制**：仅当 receiver 全部失败且 peer 组包含成功与失败混合响应时才触发交换，并通过按 source 成功数降序排列取较小方向数量 m 来实现双向平衡，避免单向过载。
2. **保留 source-computed 优势的完整 peer 组转移**：与 SGT 仅转移单个正确响应不同，GRAFT 替换 receiver 失败组为整个 peer 组，保留组内奖励对比（正负优势均保留），不进行跨模型 reward pooling。
3. **序列级兼容性门控（Compatibility Gate）**：定义基于平均 token log-likelihood 的分数 s(o|q)，结合阈值 δ 与上界 1 得到有界权重 w(o|q)，显式控制跨模型 tokenizer 不匹配的影响。
4. **Token 级重要性比率裁剪 + Peer-last 更新顺序**：将 token 级 PPO surrogate 的裁剪仅用于控制 receiver 自身优化变化，而跨模型 mismatch 由序列级权重承担；同时将含 peer 的 minibatch 置于 receiver 自有数据之后，使 clipping 在初次 update 后激活，双重保障 off-policy 安全。

## 方法详解
**整体框架**：两个策略 $\pi_{\theta^A}$ 与 $\pi_{\theta^B}$ 在同一 prompt 分布上同时训练，各自采样 n=8 个 response，计算 GRPO 相对优势 $\hat{a}_i$。

**互补组选择（Section 4.1）**：
- 候选集：$\mathcal{C}_{A \to B} = \{q : k_B(q)=0 \wedge 1 \le k_A(q) < n\}$，即 receiver 全失败、peer 至少有一个成功但非全部成功。
- 平衡选择：取 $m = \min(|\mathcal{C}_{A \to B}|, |\mathcal{C}_{B \to A}|)$，双方在各自候选集中按 source 成功数 $k_S(q)$ 降序排列，保留前 m 个（含边界平局），实现双向交换量平衡。
- 替换操作：将 receiver 失败组替换为 peer 完整组，优势使用 source 原始计算的 $\hat{a}_i^A$，不重新归一化。

**Off-policy 安全更新（Section 4.2）**：
- 分解似然比：$\frac{\pi_{\theta^B}(o|q)}{\pi_\phi(o|q)} = \underbrace{\frac{\pi_{\theta^B}(o^B|q)}{\pi_{\theta^B_{old}}(o^B|q)}}_{\text{receiver 自身变化}} \cdot \underbrace{\frac{\pi_{\theta^B_{old}}(o^B|q)}{\pi_\phi(o^A|q)}}_{\text{跨模型 mismatch}}$
- 兼容性分数：$s(o|q) = \exp\left(\bar{\ell}(\pi_{\theta^B_{old}}, o^B|q) - \bar{\ell}(\pi_\phi, o^A|q)\right)$，其中 $\bar{\ell}$ 为平均 token log-likelihood。
- 有界权重：$w(o|q) = \mathbf{1}[s(o|q) > \delta] \cdot \min\{s(o|q), 1\}$，阈值 $\delta=0.8$ 过滤低兼容度响应。
- Token 级裁剪目标：对 receiver 自身的 tokenization $o_j^B$ 计算 $\rho_{j,t}(\theta^B)$，优化 $\mathcal{I}(\theta^B) = \mathbb{E}_B\left[\frac{1}{\sum |o_j^B|}\sum_j w(o_j)\sum_t \min(\rho_{j,t}\hat{a}_j, \text{clip}(\rho_{j,t}, 1-\varepsilon_{low}, 1+\varepsilon_{high})\hat{a}_j)\right]$。
- Peer-last 顺序：先处理 receiver 自有 on-policy minibatch，再处理含 peer 的 minibatch，确保裁剪在首次 update 后生效。

## 实验与结果
- **模型对**：Pair 1 (SmolLM3-3B-Base ↔ Qwen3-1.7B-Base)、Pair 2 (OctoThinker-3B-Hybrid-Base ↔ Qwen3-1.7B-Base)、Pair 3 (SmolLM3-3B-Base ↔ OctoThinker-3B-Hybrid-Base)。
- **基准**：MATH500、AIME2024、AIME2025、AMC23、Minerva，报告 pass@1，aggregate score 为五基准未加权平均。
- **训练设置**：verl + Ray/FSDP/vLLM，n=8 rollouts/prompt，学习率 $10^{-6}$，3 epochs，prompt batch size=128，minibatch size=32。
- **主要结果（Table 1）**：
  - Pair 1 SmolLM3：GRAFT **37.06** vs GRPO(n=8) 32.60，Δ=**+4.46**；超越 GRPO(n=32) 的 35.71。
  - Pair 1 Qwen3：GRAFT **33.44** vs 31.20，Δ=**+2.24**。
  - Pair 2 OctoThinker：GRAFT **23.47** vs 21.18，Δ=**+2.29**；达到 GRPO(n=32) 水平（23.39）。
  - Pair 2 Qwen3：GRAFT **32.78** vs 31.20，Δ=**+1.58**。
  - Pair 3 SmolLM3：GRAFT **33.26** vs 32.60，Δ=**+0.66**。
  - Pair 3 OctoThinker：GRAFT **22.60** vs 21.18，Δ=**+1.42**。
  - **平均提升 +2.1 分，模型级平均最高提升 +4.5 分**。
- **对比基线**：HACPO 在五/六 block 中低于 GRPO(n=8)（最高 -4.17）；SGT 波动大（-0.53 至 +2.28）。
- **计算效率（Figure 3a）**：Pair 1 在 40.9 GPU-hours 达到 pair-mean 35.25，超越 GRPO(n=32) 1.18 分，成本仅为 0.45 倍。
- **Stored Trajectories（Section 6.2）**：复用独立 GRPO(n=8) 运行的 peer 轨迹（无需 co-training），平均提升 **+1.8 分**（为在线增益的 84%），GPU-hours 减少 27–76%。
- **消融（Table 2a）**：移除兼容性门控对 SmolLM3 下降最多（↓8.36）；移除 floor 对 Qwen3 下降最多（↓3.56）；Peer-last 优于 peer-first 与 uniform。
- **替代设计（Table 2b）**：Pooled groups、success-only transfer 均显著低于 full GRAFT；HACPO/SGT/LUFFY update 在 GRAFT 选择规则下仍落后。

## 相关工作脉络
- **HACPO (Zhang et al., 2026)**：跨模型 RLVR 交换 peer rollout，使用序列级重要性采样与 clipping，但交换覆盖所有 prompt，不限于 receiver 失败场景；本文在相同 prompt 选择下 HACPO 仍远低于 GRAFT（-5.55/-6.20）。
- **SGT / Mutual RL (Liu et al., 2026b)**：仅在 receiver 全失败且 peer 有成功时转移，但仅取单个正确响应并通过固定 λ=0.1 的 SFT 负对数似然学习，无兼容性加权；本文用完整 peer 组+source-computed advantage+compatibility gating 显著超越。
- **F-TIS (Blagoev et al., 2026)**：同家族模型间协作 GRPO，共享 vocabulary，使用 truncated importance sampling；本文面向异构模型（不同 tokenizer/架构）。
- **LUFFY (Yan et al., 2025)**：off-policy guidance 使用 shaping function $f(x)=x/(x+\gamma)$ 的重要性采样目标；本文组合 GRAFT 的 prompt 选择与 LUFFY update 仍落后 full GRAFT（↓4.55/↓2.03）。
- **DAPO (Yu et al., 2025)**：GRPO 变体，采用 token-level loss aggregation 与 asymmetric clipping；本文沿用此更新形式作为基础。
- **Entropy-guided advantage shaping (Le et al., 2026)**：在 all-incorrect 组上仅能抑制失败响应，无法提供正确 trajectory；本文直接引入 peer 的成功与失败响应以保留奖励对比。

## 局限性与未来方向
- **互补性依赖**：增益大小取决于两模型互补程度，Pair 3（SmolLM3 ↔ OctoThinker）增益最小（+0.66/+1.42）。
- **兼容性分数为代理指标**：跨 tokenizer 场景下 $s(o|q)$ 是平均 token log-likelihood 差值的指数，非精确 density ratio。
- **仅研究两模型对**：未扩展到多 peer 协同交换。
- **仅数学推理任务**：未验证无 verifiable reward 的领域。
- **仅 Base 模型 ≤3B**：未在大模型或 chat 模型上测试。
- **未来方向**：多 peer 交换、无 verifiable reward 域扩展、更大规模模型验证。

## 研究启发与可借鉴点
1. **互补 prompt 选择策略**：仅当 receiver 失败且 peer 有混合结果时才触发交换，这一条件同时保证了 peer 数据的"可用性"（有正确样本）与"多样性"（有正反对比），避免向已有 on-policy 信号的 prompt 引入噪声；可迁移至其他 off-policy 数据筛选场景。
2. **保留 source-computed advantage 而非 pooling**：跨模型 transfer 时维持源组的相对优势结构，不进行 reward pooling，避免了跨模型 reward scale 不一致导致的信号扭曲；对任何 multi-agent/multi-policy 联合训练均有参考价值。
3. **序列级权重 + token 级裁剪的双层控制**：将跨模型 mismatch 与 receiver 自身策略变化解耦，前者由序列级兼容性门控有界加权，后者由 token-level PPO clipping 控制；这种分层处理可推广至任何 cross-model 或 historical data 利用场景。
4. **Stored trajectories 的后训练复用**：证明 peer 轨迹可在独立 GRPO 训练后离线保存，后续 receiver 单独训练时复用，节省 co-training 计算；为"trajectories as reusable curriculum"提供了实证支持。
5. **Peer-last minibatch 顺序的 clipping 动态**：通过消融与 Appendix H 的 clipping fraction 测量，实证了先处理 receiver 数据使 peer tokens 的 $\rho$ 偏离 1，从而激活 clipping 的双重保障机制；这种"顺序即正则"的设计思路值得在其他 off-policy 训练中探索。

## 关键术语表
- **RLVR (Reinforcement Learning with Verifiable Rewards)**：使用可验证（如数学答案正确性）二元奖励进行策略优化的强化学习范式，典型方法包括 GRPO、DAPO。
- **GRPO (Group Relative Policy Optimization)**：对每个 prompt 采样 n 个 response，计算组内相对优势 $\hat{a}_i = (r_i - \text{mean}(\mathcal{R})) / (\text{std}(\mathcal{R}) + \epsilon)$，仅利用组内奖励方差提供学习信号。
- **All-fail group**：n 个采样 response 全部错误的 rollout 组，此时 std(R)=0，所有 advantage 为零，无法产生策略梯度。
- **Cross-model mismatch**：receiver 与 source peer 模型在 tokenizer、参数分布上的差异，导致 off-policy 数据直接优化可能引入不稳定或负迁移。
- **Compatibility gate**：基于 receiver 旧策略对 receiver-tokenized 响应的平均 log-likelihood 与 source 策略对 source-tokenized 响应的平均 log-likelihood 之差，经指数映射得到的有界权重，用于过滤低兼容度 peer 轨迹。
- **Source-computed advantage**：peer 组优势的原始计算值（由 source 模型的奖励分布得出），transfer 后保持不变，不重新用 receiver 视角归一化。
- **Peer-last update**：在优化 minibatch 排列中，将包含 peer 轨迹的 minibatch 置于 receiver 自有 on-policy minibatch 之后，使 token-level clipping 在 peer 优化前已部分激活。

## 可复现要素
- **数据集**：训练使用 MATH 训练集（7,500 题）；评测使用 MATH500、AIME2024、AIME2025、AMC23、Minerva Math。论文未明确声明数据公开状态，但 MATH 等为标准开源基准。
- **代码/权重**：论文未提及代码开源链接；使用了 verl 框架（开源）。
- **关键超参**：n=8 rollouts/prompt，学习率 $10^{-6}$，3 epochs，prompt batch=128，minibatch=32，max prompt/response length=2048/4096，temperature/top-p=1.0/1.0，PPO clip $(\varepsilon_{low}, \varepsilon_{high})=(0.2, 0.28)$，compatibility threshold $\delta=0.8$，AdamW $\beta_1=0.9, \beta_2=0.999$，weight decay=0.01，gradient norm clip=1.0，无 KL/entropy bonus，bfloat16 精度。
- **硬件**：每 pair 使用 4× NVIDIA H200 GPUs。
