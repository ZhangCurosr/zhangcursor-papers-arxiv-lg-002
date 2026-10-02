---
title: "KV-streams-for-Efficient-Compaction-in-Agentic-Reinforcement"
source: https://arxiv.org/pdf/2609.35750v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:50:05"
field: "长上下文 Agent 强化学习训练效率"
keywords: ["上下文压缩", "强化学习", "KV cache", "长上下文 Agent", "训练加速", "伪循环记忆"]
innovations: ["通过保留并流式传递 KV 条目并配合训练侧注意力掩码，消除压缩后的重复 prefill 开销", "证明仅需 RL 即可使 KV 缓存涌现伪循环记忆能力，无需前置 SFT 预热", "在多种压缩策略上即插即用并在 TextWorld/SWE-bench 等基准实现显著加速且不降性能"]
benchmarks: ["TextWorld", "ALFWorld", "SWE-bench Verified"]
---

# 论文速读：KV-streams-for-Efficient-Compaction-in-Agentic-Reinforcement

## 一句话总结
本文提出 **KV-streams**，一种即插即用的上下文压缩加速方案：在推理侧直接流式保留已压缩上下文对应的 KV 条目而不重新预填充，在训练侧通过自定义注意力掩码复现驱逐模式，从而在多种压缩策略下获得 2.6–5× 端到端训练加速，同时发现流式 KV 缓存可在仅靠 RL 的情况下涌现为携带已消失上下文的"伪循环状态"，无需先验 SFT 阶段。

## 研究问题与动机
- 长时程 Agent 任务的 rollouts 会随步数线性增长 KV cache，固定 GPU 显存预算下长 rollout 成为吞吐瓶颈。
- 常见上下文压缩（compaction）策略为避免显存爆炸会删除/总结旧上下文，但传统做法在压缩后会把保留 token 重新 prefill 进新轨迹，导致同一批 token 被反复前向计算。
- 重复 prefill 带来显著额外训练开销，尤其在高保留比例（如 sliding window）时代价发散，且多数现有工作未系统分析其与 RL 训练的交互与成本。
- 因此亟需一种既能维持压缩带来的显存收益、又能消除重复 prefill 开销、且可无缝接入任意压缩策略的训练/推理协同机制。

## 核心贡献（创新点）
- **KV-streams 插件化框架**：不改变压缩策略本身的语义，仅通过保留/流式传递 KV 并在 trainer 中使用等效注意力掩码，消除重 prefill 开销；与已有方案的区别在于把"压缩后重算"改为"保留并继续流式使用"。
- **跨策略显著加速**：在 Summary、Markovian Thinker、Markovian Picker、Sliding Window 四种压缩策略下均实现稳定加速，最高达 5× wall-clock 提升；与已有工作相比，本文强调"对任意 compaction 策略即插即用"而非单点优化。
- **无需预训练 SFT 即可涌现 KV 伪循环能力**：通过在合成检索实验中证明仅靠 RL 即可让模型利用保留 KV 召回已被驱逐的信息；与前期需要 SFT 预热才能出现类似行为的做法相比，简化了后训练流程。
- **性能与泛化不降**：在 TextWorld、ALFWorld 及 SWE-bench Verified 上，KV-streams 最终性能匹配或优于 re-prefill 基线，并在未见任务上保持良好迁移；区别于以往只关注效率而忽略泛化评估的工作。
- **工程化与复现贡献**：提供 vLLM/prime-RL 与 Slime/SGLang 两套分叉实现细节与超参，便于直接复用至后续长上下文 Agent RL 训练管线。

## 方法详解
- **压缩形式化**：压缩算子 κ(C) 由删除算子 gen 与生成算子 del 组合，触发条件可为固定预算 B 或模型自主决定；保留长度为 r = o（重叠）+ s（生成的桥接/摘要）。
- **Re-prefill compaction 的成本来源**：每次压缩后将保留 token 连同位置编码一起注入新轨迹，导致其 KV 需要重新计算；额外 prefill token 总量近似为 Δ_re-prefill = N·r / (B − r)，当 r → B 时成本发散。
- **KV-streams 的核心机制**：推理侧直接淘汰被删段对应的 KV 条目，保留部分（含摘要）沿用已有 (k_i, v_i) 并继续流式生成，使 Δ_re-prefill = 0；训练侧通过自定义注意力掩码屏蔽已删除 span，使得单次前向/反向即可复现该驱逐模式。
- **与常用压缩策略的兼容性**：Summary（o=0, 以摘要替换）、Markovian Thinker（o=B/2, s=0）、Markovian Picker（模型自主选择保留 b 项）、Sliding Window（o≈B, s=0）均可接入 KV-streams，其中高保留策略获益最大。
- **训练工程要点**：采用异步 RL（max off-policy lag=3），并避免在每次权重更新时刷新 KV cache，否则会导致长轨迹被强制重 prefill；vLLM 侧使用 16-token KV block 并做尾部 padding 以保证块对齐。

## 实验与结果
- **数据集与基准**：TextWorld（4 类任务×3 难度，60k 合成训练，256 评估）、ALFWorld（70  seen + 67 unseen）、SWE-rebench/ScaleSWE 训练集与 SWE-bench Verified（500 Python bug fix）。
- **主要设置**：TextWorld 用 Qwen3-4B-Instruct-2507、32k 上下文、每 10 turn 压缩、最多 200 turn、500 步梯度、batch=512、lr=1e-6、8×H100；ALFWorld 用同模型、16k 上下文、每 10 turn、最多 100 turn、200 步、batch=128、4×A100；SWE 用 Qwen3.5-4B、64k 上下文、每 30 turn、最多 100 turn、batch=256、8×GB200。
- **关键数值结果**：
  - TextWorld：KV-streams 以 **3.7×–11.3× 更少 GPU-hours** 达到 re-prefill 的最终性能，整体最高可达 **~5× wall-clock 加速**；re-prefill 的 Summary/Sliding-Window 甚至比 full context 更慢。
  - ALFWorld：由于序列较短（16k），加速区间为 **1.3×–3.0×**，最终成功率与 re-prefill 相当。
  - SWE-bench Verified：Sliding-Window + KV-streams 达到 **52.7% ± 2.1%**，与 full context（51.5% ± 1.2%）及 re-prefill Markovian Thinker（53.4% ± 0.2%）无显著差异；达到峰值耗时约 **20h vs full context >65h（~3× 加速）**。
- **泛化**：在未见文本游戏上，多种 KV-streams 配置均能与最优策略相当，且全部优于 base model，说明不存在明显泛化损失。
- **最强结果**：TextWorld 最高 **5× wall-clock 提速**；SWE-bench Verified 在保持可比性能前提下实现 **~3× 提速**。

## 相关工作脉络
- **Re-prefill compaction / 传统压缩管线**：以 Summary、Sliding Window 等为代表，保留窗口后重入新轨迹并重 prefill；本文与它们的差异不在"如何压缩"而在"压缩后是否重算"。
- **Markovian Thinker (Aghajohari et al., 2026)**：删除一半上下文并不补充摘要；KV-streams 可与之叠加使用并消除其 re-prefill 成本。
- **MEMENTO (Kontonis et al., 2026)**：曾提出保留 KV 作为 pseudo-recurrent state，但实验中依赖前置 SFT；本文证明在 KV-streams 设置下仅 RL 即可出现类似行为。
- **AutoCompressor / 自学习压缩策略**：让模型学会保留什么（摘要或软 prompt）；这些策略仍会在每次压缩后重建上下文，KV-streams 与其正交并可叠加降本。
- **Inference-time 缓存管理（H2O、TOVA、attention sinks）**：侧重推理期缓存命中/状态绑定；本文聚焦训练期复现驱逐模式的效率提升，并与测试时 scaling 方法互补。
- ** recurrent/state-space / 架构改造路线**：需重新训练模型以使用固定状态；KV-streams 不改动模型架构，保持与预训练 Transformer 兼容。

## 局限性与未来方向
- **块对齐 padding 引入额外 token**：vLLM 16-token block 机制要求尾部补齐，增加少量 token 总量（论文已在图中计入），在更细粒度块或不同引擎下表现可能变化。
- **当前验证以中等规模模型与特定基准为主**：主要在 4B 级模型与文本游戏/软件工程任务上评估，超大模型或更长硬约束场景的外推仍需验证。
- **turn-level 驱逐优先于 token-level**：为避免格式退化采用整 turn 删除，牺牲部分长度控制精度；token 级掩码结合 mid-training 可能更优但需进一步探索。
- **异步 RL 实现依赖特定 fork**：prime-RL/vLLM 与 Slime/SGLang 的分叉支持目前有限覆盖，移植到其他 pipeline 需要工程适配。
- **未来可拓展方向**：更细粒度动态驱逐、与学习型压缩联合优化、更大规模代码/多模态 agent 训练、以及将 KV 伪循环能力用于持续学习的长期记忆机制。

## 研究启发与可借鉴点
- **训练侧复现推理驱逐的策略**：用自定义注意力掩码在单次 forward/backward 内精确复现 KV 淘汰，避免多次重 prefill，可直接迁移到任意"压缩 + RL"管线。
- **KV cache 作为可学习的长期记忆载体**：实验证明即使不在输入中可见，保留 KV 仍可编码并召回早期信息；这为"显式记忆 vs 隐式 KV 记忆"对比研究提供新思路。
- **免去 SFT 预热简化后训练流程**：发现伪循环行为可由 RL 自行涌现，使部署成本更低、迭代更快，适合资源受限团队快速落地长上下文 Agent RL。
- **高保留策略在 KV-streams 下性价比凸显**：Sliding Window 等高保留方案原本因重 prefill 成本过高而受限，现可通过 KV-streams 充分发挥其信息保留优势。
- **工程实现的对称性设计值得借鉴**：推理端直接淘汰 KV、训练端同步掩码，并且配合 in-flight 权重更新与异步 lag 控制，形成端到端一致的低开销管线。

## 关键术语表
- **KV-streams**：在压缩后直接保留并继续流式传递已被保留 token 的 KV 条目，避免再次 prefill 的即插即用机制。
- **Re-prefill compaction**：传统压缩做法，压缩后将保留 token 重新送入模型进行前向计算，产生重复训练开销。
- **Context compaction**：当上下文超过预算时删除或总结旧内容，以保持活跃上下文大小固定的策略族。
- **Pseudo-recurrent state（via KV cache）**：未被上下文显式包含但通过保留 KV 继续影响后续输出的隐式记忆能力。
- **Sliding-Window compaction**：保留靠近当前窗口的较长历史片段（o≈B、s≈0）的压缩方式。
- **Markovian Thinker**：每次压缩保留后一半上下文并不生成摘要的压缩策略。
- **Markovian Picker**：由模型自主选择要保留的 b 条 turn 并进行删除的压缩策略。
- **Summary compaction**：以生成的摘要替换旧上下文、保留重叠为 0 的压缩策略。

## 可复现要素
- **代码**：开源（https://github.com/Emilianopp/KV-streams），并提供 vLLM/SGLang 相关分叉仓库。
- **数据集**：TextWorld、ALFWorld、SWE-rebench、ScaleSWE、SWE-bench Verified 均为公开/常用基准；论文未新增封闭数据集。
- **权重**：使用开源模型 Qwen3-4B-Instruct-2507、Qwen3.5-4B 及其后训练产物，未在文中单独发布新 checkpoint。
- **关键超参**：lr=1e-6、batch=512/128/256（依基准）、max turns=100/200、context budget=16k/32k/64k、compaction interval=10/30 turns、async level=1、off-policy lag=2–3、Adamβ=(0.9,0.9)/(0.9,0.98)、KL=0.001、温度=1.0、max tokens per turn=1024/8192 等见论文附录 Table 1；具体并发与 block size 见 Table 2 与附录 B。
