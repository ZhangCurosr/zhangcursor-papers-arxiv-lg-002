---
title: "VETTA-COORDINATING-TURN-AND-TOKEN-LEVEL-CREDIT-ASSIGNMENT-FO"
source: https://arxiv.org/pdf/2610.08402v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:24:30"
field: "多轮 LLM Agent 强化学习"
keywords: ["credit assignment", "multi-turn LLM agent", "reinforcement learning", "PPO", "value estimation", "turn-level", "token-level"]
innovations: ["双粒度信用分解：turn-level 优势 + within-response-centered token 残差的融合机制", "轻量共享 Critic：仅保留前 2 个 Transformer 块实现双粒度价值估计，性能超过全深度", "中心化残差融合的理论保证：在 Euclidean 距离意义下最优逼近纯 token 优势"]
benchmarks: ["ALFWorld", "WebShop"]
---

# 论文速读：VETTA-COORDINATING-TURN-AND-TOKEN-LEVEL-CREDIT-ASSIGNMENT-FO

## 一句话总结
本文提出 VETTA，一种针对多轮 LLM Agent 的信用分配方法，通过共享轻量级 Critic 分别学习轮次级（turn-level）和 token 级（token-level）价值，并以轮次优势为主信号、中心化的 token 残差为补充，引导 token 级 PPO 更新。在 ALFWorld 和 WebShop 上，VETTA 均超越所有 critic-free 和 critic-based 基线，且仅保留 2 个 Transformer 块的 Critic 即可获得最佳性能并将 Critic 侧计算降低约 87%~91%。

## 研究问题与动机
1. **多轮 Agent 面临稀疏延迟反馈**：任务奖励只在交互结束时给出，需要区分哪些 Response 对最终结果有贡献，以及每个 Response 内哪些 token 决策更关键。
2. **现有方法只关注单一粒度**：Turn-level 方法（如 Turn-PPO）能区分不同 Response 的贡献，但无法区分同一 Response 内部的 token 决策；Token-level 方法（如 PPO*）能传播跨轮 token 信用，但未显式建模每轮的响应级信用，两者存在互补性盲区。
3. **Critic 计算开销高但未被有效压缩**：Critic-based PPO 需要额外价值推理和 Critic 更新，现有工作尚未系统研究精简 Critic 深度是否影响策略性能。
4. **单一粒度的优势信号在跨基准上表现不一致**：PPO* 在 WebShop 上更强，Turn-PPO 在 ALFWorld 上更强，提示需要融合两种粒度信号以提升鲁棒性。

## 核心贡献（创新点）
1. **提出双粒度信用分解框架**：将多轮信用分配形式化为轮次级优势（衡量完整响应在交互中的贡献）与 token 级残差（区分响应内部的生成决策）两个互补信号，两者在已有工作中均未同时被分离建模。
2. **VETTA 融合机制——中心化 Token 残差叠加**：用 turn 优势确定响应级信用均值，加上缩放后的 within-response-centered token 残差，使融合后优势向量在 Euclidean 距离意义下最接近纯 token 优势；与 HyGAE 等直接线性组合方法本质不同。
3. **轻量共享 Critic 设计**：Critic 仅保留初始化 Actor 所用预训练 checkpoint 的前 $d$ 个 Transformer 块（论文实验取 $d=2$），两个价值头共享骨干但各自独立输出参数，相比独立双 Critic 或单头共享设计性能更优。
4. **系统实验揭示 Critic 深度与性能的倒U关系**：在 ALFWorld 和 WebShop 上，2 块 Critic 的平均成功率高于完整 28 块，同时将 Critic 侧中位时间降低约 87%（ALFWorld）和 91%（WebShop）。

## 方法详解
- **双粒度价值估计**：Critic 对"context + 生成的 response"做单次 causal forward pass，取位置 $h_{k,1}$（context 末尾，对齐 turn 决策）和 $h_{k,j}$（第 $j$ 个 token 前，对齐 token 决策），分别经独立头映射：
  $$V_k^{\text{turn}} = w_{\text{turn}}^\top h_{k,1} + b_{\text{turn}}, \quad V_{k,j}^{\text{tok}} = w_{\text{tok}}^\top h_{k,j} + b_{\text{tok}}.$$
- **轻量 Critic 骨干**：从 Actor 初始化所用的 $N$ 块 Transformer 中仅保留前 $d \leq N$ 块（含 token embedding 和末端 RMSNorm），其余块丢弃，所有保留模块仍 trainable。
- **GAE 独立计算**：同一标量奖励 $r_k$ 在 turn 级赋给第 $k$ 轮、在 token 级赋给响应最后一个 valid token（之前位置奖励为 0）；turn 与 token 分别用 $(\gamma_{\text{turn}}, \lambda_{\text{turn}})=(0.95, 0.95)$ 和 $(\gamma_{\text{tok}}, \lambda_{\text{tok}})=(1, 1)$ 做 GAE，得到 $A^{\text{turn}}$ 与 $A^{\text{tok}}$。
- **中心化融合**：对每个响应内 token 优势求均值后相减得到残差 $\widetilde{A}_{k,j}^{\text{tok}} = A_{k,j}^{\text{tok}} - \overline{A^{\text{tok}}}_k$，最终 advantage：
  $$A_{k,j}^{\text{VETTA}} = A_k^{\text{turn}} + \alpha \, \widetilde{A}_{k,j}^{\text{tok}}.$$
  该融合保证 $\frac{1}{L_k}\sum_j A_{k,j}^{\text{VETTA}} = A_k^{\text{turn}}$，turn 优势决定均值，token 残差只调节内部相对差异。
- **Critic 更新**：目标值 $\widehat{G}^b(i) = \text{sg}(V_{\text{old}}^b(i) + A_i^b)$，采用 PPO clipped value regression，turn 和 token loss 分别均值后相加。
- **Actor 更新**：保留 token-wise PPO ratio + clipping，并采用 dual clipping（对负优势时截断至 $-c A_{k,j}^{\text{VETTA}}, c=3$）及 KL 正则（系数 0.01）。

## 实验与结果
- **数据集**：ALFWorld（140 seen + 134 unseen，6 类家务任务）、WebShop（500 个购物任务），使用 Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct。
- **主要结果（Qwen2.5-1.5B）**：VETTA 在 ALFWorld 达到 **91.9%** 成功率（超越 GiGPO 的 86.1% 约 5.8pp）、WebShop 达到 **73.8%** 成功率（超越 GiGPO 的 67.4% 约 6.4pp）。
- **主要结果（Qwen2.5-7B）**：VETTA 在 ALFWorld 达到 **95.5%**、WebShop 达到 **76.0%**，均为 critic-based 和 critic-free 基线中最高；WebShop task score 86.5 亦为最高。
- **Critic 深度**：2 块 Critic 成功率高于 8 块和 28 块；Critic 侧中位时间从 152.0s 降至 19.7s（ALFWorld，-87%），从 92.2s 降至 7.9s（WebShop，-91%），占训练步时间从 ~26% 降至 ~4%。
- **融合消融**：残差融合（C4: 72.8%）显著优于直接相加（C3: 55.0%）和单一 turn/token 信号（C2: 59.8%, C1: 67.4%）。
- **Critic 结构消融**：共享骨干 + 双头 > 共享单头 > 独立双 Critic；ALFWorld 上分别为 93.6% / 91.4% / 89.3%，WebShop 上分别为 72.8% / 65.0% / 58.4%。
- **超参敏感性**：$\alpha=1$ 在 ALFWorld 最优，$\alpha=3$ 在 WebShop 最优，说明两个任务对 token 级细粒度信用的依赖程度不同。

## 相关工作脉络
1. **Turn-PPO (Li et al., 2025)**：critic-based 轮次级优势估计，对完整 response 应用 PPO clipping；VETTA 在此基础上引入 token 级残差细化响应内部决策，并证明单粒度基线在跨基准上互有强弱。
2. **GiGPO (Feng et al., 2025)**：critic-free，结合 episode-relative credit 与重复状态下的 action-level 比较；VETTA 通过 parametric critic 显式建模双粒度价值，在两种规模模型上均超越 GiGPO。
3. **HyGAE (Zhang et al., 2026)**：线性组合 turn- 和 token-level 优势，训练统一价值函数；VETTA 的本质区别在于分离监督的双头架构 + 中心化残差融合，避免两个优势的响应级偏移互相干扰。
4. **PPO*（本文自实现 baseline）**：将 token-level GAE 跨轮传播（跳过 observation token），是传统 token-level PPO 的多轮扩展；VETTA 在此基础上补充 turn-level 响应级信号，弥补 PPO* 在 ALFWorld 等任务上的不足。
5. **ArCHer (Zhou et al., 2024)**：学习 utterance-level 价值以引导下层 token policy；VETTA 的不同在于同一 rollout 内同时学习两个粒度并直接融合用于 PPO 更新，而非分层离线蒸馏。
6. **HiPER (Peng et al., 2026)**：共享 critic 骨干用于 subgoal planning 和 action execution；VETTA 借鉴共享骨干思路，但将分工改为 turn-level vs. token-level 价值估计，并验证浅层骨干的可行性。

## 局限性与未来方向
1. **仅在两个 benchmark 上验证**：ALFWorld 和 WebShop 均偏短 Horizon（最大 50/15 轮），长 Horizon 场景下双粒度融合的增益未见评估。
2. **Critic 浅层的有效性机制尚不明确**：2 块优于 28 块的现象缺乏理论解释，可能是优化稳定性、正则化效应或任务对深层表征需求较低所致，需进一步分析。
3. **$\alpha$ 超参任务依赖性强**：ALFWorld 最优 $\alpha=1$、WebShop 最优 $\alpha=3$，说明需针对任务类型调参，自动化选择策略未给出。
4. **未探索更多 Critic 结构设计**：如独立骨干、noisy head、probe 式轻量估值器等，当前仅比较了三种结构。
5. **Future work 自述**：建议根据任务的交互与反馈结构定制 value model，暗示当前通用设计可能不适配所有 Agent 场景。

## 研究启发与可借鉴点
1. **双粒度信用分解的设计范式**：将 turn 级优势与 token 级残差解耦并通过中心化融合，避免两者偏移相互干扰，这一思路可迁移到任意需要"高层决策+低层执行"双层信用分配的场景（如 Code Generation Agent、Multi-step Reasoning）。
2. **轻量 Critic 的可行性验证**：仅保留前 2 个 Transformer 块即能达到甚至超过全深度性能，提示在实际部署中可大幅降低 RL 训练的 memory 与时间成本；可结合本团队的效率优化方向进一步探索更浅 backbone 的临界深度。
3. **Within-response centering 的数学性质**：论文给出投影等价性（最小二乘意义下最接近纯 token 优势），为其他研究中类似"主信号+残差修正"的融合设计提供了理论依据。
4. **对比实验设计值得借鉴**：通过构造 PPO*（跨轮 token GAE）、Turn-PPO、混合配置的对照，清晰揭示单一粒度的强弱边界，这种"拆解-对照-融合"的实验范式可用于其他 RL-for-LLM 方法的评估。
5. **$\alpha$ 敏感性的任务差异**：对短 Horizon 强信号任务（ALFWorld）应弱化 token 残差、对搜索型任务（WebShop）应强化 token 残差，这一洞察可指导后续多任务场景下的自适应超参设计。

## 关键术语表
**VETTA**：Value Estimation at Turn and Token levels for LLM Agents，本文提出的双粒度信用分配框架。
**Turn-level advantage**：衡量每个完整 response 在多轮交互中的相对价值贡献，由 turn 级 GAE 计算。
**Token-level advantage**：衡量 response 内部每个生成 token 的相对价值贡献，由 token 级 GAE 计算。
**Within-response centering**：将 token 级优势减去同响应内均值，使残差均值为零，仅保留响应内部的相对差异。
**GAE (Generalized Advantage Estimation)**：Schulman et al. 提出的 TD 残差加权累积方法，通过 $\lambda$ 控制偏差-方差权衡。
**Lightweight shared critic**：仅保留前 $d$ 个 Transformer 块、共享骨干但含独立 turn/token 价值头的价值网络。
**Dual clipping**：对负 advantage 的 token PPO loss 额外施加下界截断（$-c \cdot A$，$c=3$），防止优势为负时策略过度更新。
**Critic-free RL**：不训练参数化 value function，通过 rollout 组内比较（如 GRPO、RLOO、GiGPO）估计优势的方法。

## 可复现要素
- **数据集**：ALFWorld（开源）、WebShop（开源），均在论文中明确说明。
- **代码**：已开源，GitHub: https://github.com/Jiaju-Chen/VETTA-official
- **权重**：使用 Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct 官方 checkpoint 作为 Actor 初始化；Critic 从头训练（仅保留前 2 块）。
- **关键超参**：$\gamma_{\text{tok}}=\lambda_{\text{tok}}=1$，$\gamma_{\text{turn}}=\lambda_{\text{turn}}=0.95$，$\alpha=1$（ALFWorld）/ $3$（WebShop），KL 系数 0.01，更新次数 150，Critic 保留 2 块，Actor LR=$10^{-6}$、Critic LR=$10^{-5}$，Mini-batch 64/256。
- **评估协议**：3 个 decoding seed（123, 456, 789）取均值与标准差；ALFWorld 用 valid_seen 140 题，WebShop 用固定 500 题 manifest。
