---
title: "REFOLD-TRAINING-FREE-REVERSIBLE-INTER-TURN-CONTEXT-FOLDING-F"
source: https://arxiv.org/pdf/2610.07863v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:15:08"
field: "Agent 系统效率优化"
keywords: ["long-horizon agents", "context management", "LLM serving", "prefix cache", "training-free compression"]
innovations: ["无训练可逆渲染层：通过 deduplication 和 folding 两个算子压缩 agent 渲染上下文，保留原始 append-only history", "Chunked rendering 保护 prefix cache：每 k 步批量更新边界，使 cache hit 从 51% 恢复至 87%", "严格可逆的内容移除：通过 restore 命令从 history 恢复被压缩内容，消除永久信息丢失风险"]
benchmarks: ["SWE-bench Verified", "SWE-bench Pro", "Multi-SWE-bench", "Terminal-Bench 1.0", "Terminal-Bench 2.1"]
---

# 论文速读：REFOLD: TRAINING-FREE REVERSIBLE INTER-TURN CONTEXT FOLDING FOR LONG-HORIZON AGENTS

## 一句话总结
ReFold 提出一种无训练、可逆的上下文渲染层，通过去重（deduplication）和折叠（folding）两个轻量算子压缩长轨迹 LLM Agent 的渲染上下文，在保持原始交互历史完整的前提下将 Token 消耗最高降低 2.5×、KV-cache 峰值减半，且不损害任务成功率。

## 研究问题与动机
- 长 horizon LLM Agent（如 SWE-agent）采用 append-only 交互历史，每一步都将完整转录重发给模型，导致上下文长度和推理成本随步骤数二次增长（$\Theta(T^2)$），最终超出 context window。
- 现有上下文管理方法（位置驱逐、内容剪枝、外部存储、摘要压缩）依赖辅助预测器或额外模型调用，带来运行时开销、破坏 prefix cache 复用，且被丢弃的内容无法保证恢复。
- 作者通过动机研究（Section 2）发现：仅剪枝 tool output 无法将上下文约束到固定预算（tool output 仅占 peak KV-cache 约 59%，agent 自身 message 占剩余 66–68%）；辅助操作（如 compact 的 summarization call、prune 的 scorer call）引入显著延迟和成本；驱逐操作破坏 prefix cache 复用并造成不可恢复的内容丢失。
- 生产级证据：GitHub Copilot 分析显示，仅 7.8% 的 session 触发 context compaction，却占总 token 消耗的 44.2%。

## 核心贡献（创新点）
1. **系统评估六种上下文管理策略**：在统一 agent harness 上对比 mask/prune/recall/compact/clip/base，揭示"仅剪枝 tool output 不足以约束上下文"及"预测性驱逐带来系统性成本"，为后续方法设计提供实证依据。
2. **无辅助预测器的双算子渲染层**：Deduplication 以 stub 替换重复出现的 tool output 片段；Folding 将由 agent 自身报告的 STALE 步骤折叠为一行 note，两者均无需额外模型调用。
3. **Chunked rendering 保护 prefix cache 复用**：两个算子共享一个每 k 步前进一步的全局边界 $b_t$，边界之间渲染上下文严格 append-only，prefix cache 命中率从 51%（每步重渲染）提升至 87%（接近 base 的 84%）。
4. **严格可逆的内容移除**：每次移除均可通过 `restore <step>` 从原始 history 恢复，或重新执行命令观察最新状态，消除永久内容丢失风险。
5. **即插即用且跨模型通用**：仅在渲染层插入 hook 并修改 prompt 协议，兼容标准 ReAct-style harness，在 Qwen3.6-27B/35B-A3B 及 Claude Opus 4.8 上均验证有效。

## 方法详解
**形式化设定**（Eq. 1）：交互历史 $\mathbf{h}_t = \mathbf{p}_{\text{sys}} \oplus \mathbf{p}_{\text{task}} \oplus \mathbf{z}_1 \oplus \cdots \oplus \mathbf{z}_t$，其中 $\mathbf{z}_i = \mathbf{a}_i \oplus \mathbf{o}_i$ 为第 $i$ 轮（agent action + environment observation）。默认 $\mathbf{c}_{t-1} = \mathbf{h}_{t-1}$，ReFold 引入渲染函数 $R$ 使得 $\mathbf{c}_t = R(\mathbf{h}_t)$，不修改 $\mathbf{h}_t$。

**算子一：Deduplication（去重）**：将 observation $\mathbf{o}_i$ 划分为连续 span $s_{i,1}, \ldots, s_{i,m_i}$，若某 span 与先前 turn 中某 span 完全相同，则替换为 reference stub（记录路径、起止位置、源 step 编号；长 span 保留 top-level class/function signatures）。Stub 格式：`[context-manager] <path>:<start>-<end> omitted: identical to step <source step>. \`restore <step>\` shows it again.`

**算子二：Folding（折叠）**：要求 agent 在 response 开头附两个 metadata 字段：PROGRESS（当前步骤的一句话描述）和 STALE（不再需要的先前 step 索引集合）。STALE 仅接受最近 k 步内的索引（Eq. 4：$S_t = S_{t-1} \cup (\text{STALE}_t \cap \{t-k, \ldots, t-1\})$）。被折叠的 turn 替换为一行 note：`[context-manager] steps <i>-<j> folded: <PROGRESS_i> · ... · <PROGRESS_j> \`restore <i>-<j>\` shows them again.`，附带关键签名或搜索结果。

**Chunked Rendering（Eq. 6-7）**：边界 $b_t = \max(0, k \lfloor \frac{t-k}{k} \rfloor)$，每 k=3 步前进一步。$i > b_t$ 的 turn 完整渲染；$i \leq b_t$ 且 $i \in S_t$ 的 turn 用 fold note 替代；其余用 deduplicated 版本渲染。边界之间渲染上下文不变，KV cache 完全复用。

**可逆性**：`restore <step>` 命令由 harness 拦截，从 history 查找原始 turn 并追加为新 turn 至 context 末尾，不修改已有 prefix；重执行原命令则获得最新环境状态，重执行输出完整保留且豁免后续压缩。

## 实验与结果
**设置**：mini-swe-agent harness，Qwen3.6-27B（dense）和 Qwen3.6-35B-A3B（MoE），vLLM 服务（prefix caching 开启），温度=0，最大生成 16384 tokens，每 trajectory 最多 250 步，context window 256k，命令输出截断至 10000 字符。

**五个基准**：SWE-bench Verified（50 tasks）、SWE-bench Pro（50）、Multi-SWE-bench（50）、Terminal-Bench 1.0（50）、Terminal-Bench 2.1（50）。

**主要结果**（Table 1，全 window，w=8）：
- **Token 减少 18–60%**：Qwen3.6-27B 上 Terminal-Bench 2.1 从 3989k 降至 1909k（-52%）；Qwen3.6-35B-A3B 上 SWE-bench Pro 从 3328k 降至 1344k（-60%）。
- **KV-cache 峰值减半**：Qwen3.6-27B 上 SWE-bench Verified 从 1.71 GiB 降至 0.83 GiB（-52%）；Qwen3.6-35B-A3B 上 SWE-bench Pro 从 0.92 GiB 降至 0.41 GiB（-55%）。
- **成本降低 2–45%**：Qwen3.6-27B 上 SWE-bench Verified 从 19.5¢ 降至 12.9¢（-34%）。
- **任务成功率无损**：Resolve 在 ±3 范围内波动，处于 run-to-run 噪声内。
- **步骤缩短 6–40%**：多数设置下轨迹更短。

**资源受限场景**（Table 2）：
- 在 16k–64k 预算下，compaction 事件减少最高 92%（32k 预算时从 1.0 降至 0.1）。
- 并发 w=48 时，resolve 从 7 提升至 32，队列延迟从 8.96s 降至 0.02s（减少 100%），推理加速 1.7×，成本降低 3.4×。
- 16k + w=48 联合约束下，token 减少 24%，compaction 减少 37%，queue delay 从 0.87s 降至 0.02s。

**对比已发表方法**（Table 10）：ReFold 在相同 harness 上较 LLMLingua-2/SWE-Pruner/AgentDiet 优势显著——token 减少 48%（最强），cost 减少 34%，且是唯一将 KV-cache 减半的方法。

**Ablation**（Figure 4）：渲染开销仅 0.4ms/step（vs compact 3.3s，prune 0.15s，recall 8.5ms）；k=3 为最优 chunk size；chunked rendering 使 cache hit 从 51% 恢复至 87%；reversibility 确保 100% 可恢复而非 67%。

## 相关工作脉络
1. **Observation masking**（Lindenbauer et al., 2025, mask）：仅保留最近 N 个 observation，其余替换为一行 placeholder；ReFold 定位差异——mask 破坏 prefix cache（命中率仅 15.2%）且丢失内容，ReFold 通过 chunked rendering 保持高命中率且完全可逆。
2. **内容剪枝**（SWE-Pruner, Wang et al., 2026; AgentDiet, Xiao et al., 2026; TACO, Ren et al., 2026）：使用训练 scorer 或 heuristic rules 删除 irrelevant lines；ReFold 定位差异——无需训练/辅助模型调用，运行时开销低 2–3 个数量级，且不破坏 cache locality。
3. **外部存储/召回**（ARC, Dang et al., 2026; LCM, Ehrlich & Blackman, 2026）：将驱逐内容存入 addressable store；ReFold 定位差异——不引入额外存储层和 lookup 延迟，同时通过可逆 restore 机制提供同等恢复能力。
4. **摘要压缩**（Context-Folding, Sun et al., 2026; AgentFold, Ye et al., 2026; CompactionRL, Li et al., 2026b; SWE-MeM, Gao et al., 2026）：触发阈值后调用模型生成 summary；ReFold 定位差异——完全 training-free、no auxiliary call，避免 summarization 引入的 152s/1.6¢ per task 开销。
5. **Prompt/Token 级压缩**（LLMLingua, Jiang et al., 2023; LLMLingua-2, Pan et al., 2024; LongLLMLingua, Jiang et al., 2024）：基于 perplexity 或 classifier 过滤 token；ReFold 定位差异——操作粒度为 turn/span 级而非 token 级，保留了结构完整性（函数签名等），且 48% token 节省远超 LLMLingua-2 的 8%。
6. **KV-cache 优化**（SkipKV, Tian et al., 2026; Continuum, Li et al., 2026a）：在 inference 引擎层选择性跳过或 TTL 管理 KV cache；ReFold 定位差异——在应用/渲染层工作，与引擎层优化正交可组合。

## 局限性与未来方向
- **STALE 标注依赖 agent 自报告**：需要 prompt 指令引导 agent 输出 PROGRESS/STALE 字段，若 agent 遗漏或错误标注（如示例中 step 17 的 STALE 标错导致需 re-run 纠正），可能影响压缩效果；论文提及可通过 reminder 机制缓解但增加少量额外输出。
- **仅针对 ReAct-style append-only harness 验证**：未测试 tree-search、branching trajectory 或非 bash 工具场景，推广性有待验证。
- **chunk size k 的选择依赖经验调优**：论文通过 ablation 选定 k=3，但未给出理论分析或自适应机制。
- **未与学习型方法（CompactionRL、SWE-MeM）在同等条件下直接对比**：Table 10 仅对比了 non-learning baselines，学习型方法的潜力未被充分评估。
- **仅验证了两个开源模型加一个 proprietary model**：更广泛模型族的泛化性未检验。

## 研究启发与可借鉴点
1. **Rendering-layer 抽象的普适价值**：将上下文压缩与 agent 核心逻辑解耦，仅通过 hook + prompt 变更即可集成，为后续研究提供了"即插即用"的范式——任何 context management 工作可借鉴此设计以减少与特定 harness 的耦合。
2. **Chunked rendering 保护 prefix cache 的思路**：将"频繁修改历史→破坏 cache"与"不修改历史→浪费空间"之间的 trade-off 转化为"批量更新边界"的设计，这一思路可直接迁移至其他需要维护 long-context 缓存的系统（如 RAG pipeline、multi-turn dialog system）。
3. **可逆移除（reversibility）作为设计约束**：将"内容丢失风险"转化为"可 restore 的 stub"，这一原则可用于设计更鲁棒的 context management 方案——任何压缩操作都应保留 recover path，避免不可逆的信息损失。
4. **联合评估 resource consumption 与 execution efficiency**：论文同时报告 token、KV-cache、cost、duration、queue delay、compaction rate 六个维度，建立了完整的 agent serving 评估体系，可作为后续工作的评估模板。
5. **Composition 实验设计**：Section F 展示了 ReFold 与 routing/external memory 的正交组合，证明了分层架构的可组合性，为多机制协同设计提供了实验方法论参考。

## 关键术语表
**Append-only interaction history**：Agent 与环境交互的不可变记录，每步将完整历史重发给模型，是长 horizon agent 上下文膨胀的根本原因。

**Deduplication（去重）**：ReFold 的第一个算子，将 observation 中与前序 turn 完全相同的 span 替换为轻量 reference stub，保留首次出现。

**Folding（折叠）**：ReFold 的第二个算子，将由 agent 标记为 STALE（已完成阅读）的整轮 turn 压缩为一行 note，包含 PROGRESS 描述和关键签名。

**Chunked rendering（分块渲染）**：两个算子共享全局边界 $b_t$，每 k 步前进一步；边界之间渲染上下文严格 append-only，以保护 prefix cache 复用。

**Reversibility（可逆性）**：任何被移除的内容均可通过 `restore <step>` 从原始 history 恢复，或通过重执行原命令获取最新状态，保证无永久信息丢失。

**Prefix cache**：vLLM 等推理引擎缓存的前序 prompt token 的 KV 矩阵，避免重复 prefill 计算；修改历史 context 会使其失效。

**STALE / PROGRESS**：ReFold 要求 agent 在 response 开头输出的两个 metadata 字段，分别标记"已完成阅读的历史步骤"和"当前步骤目的"。

**Context compaction**：当上下文接近 budget 上限时触发 summarization-and-restart 事件的机制，是现有方法的主要压缩策略，引入额外模型调用和延迟。

## 可复现要素
- **数据集**：SWE-bench Verified（50 tasks）、SWE-bench Pro（50）、Multi-SWE-bench（50）、Terminal-Bench 1.0（50）、Terminal-Bench 2.1（50），均使用官方容器和评估器；论文声明将在 publication 时 release code 和 scripts。
- **代码/权重**：ReFold 实现为 mini-swe-agent 的 plugin；模型 Qwen3.6-27B/35B-A3B 和 Claude Opus 4.8 为公开/商业模型；代码和实验脚本承诺开源（论文未提及具体仓库 URL）。
- **关键超参**：chunk size $k=3$；temperature=0；max generated tokens=16384；step limit=250；context window=256k；command output truncation=10000 characters；STALE 窗口限制为最近 k 步；compaction 触发阈值=75% budget（baseline compact）。
