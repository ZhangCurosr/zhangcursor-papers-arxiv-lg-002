---
title: "Stateless-Language-Agents-Scaling-Long-Horizon-Automated-Res"
source: https://arxiv.org/pdf/2610.07625v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 17:41:53"
field: "多智能体自动化研究系统"
keywords: ["stateless language agent", "long-horizon automation", "automated research", "multi-agent coordination", "context management", "code optimization"]
innovations: ["提出有状态搜索与无状态智能体的架构原则，由 harness 持有全部研究状态并按角色按需重构每次 agent 上下文", "设计证据驱动的 Advisor-Worker 两角色协调机制，以 lane-local best 与 regime pivot re-seed 保留替代探索方向", "在十亿 token 级别构建统一评测协议，证明聚焦上下文与显式分配的效应随运行长度累积放大且 Advisor 开销<0.6%"]
benchmarks: ["FrontierSWE", "Anthropic VLIW SIMD Kernel Optimization", "SOL-ExecBench", "FrontierCS Structured-LWE"]
---

# 论文速读：Stateless-Language-Agents-Scaling-Long-Horizon-Automated-Res

## 一句话总结
本文提出 **Stateless Language Agents（SLA）** 框架，以"有状态搜索、无状态智能体"为核心原则解决长周期自动研究中上下文膨胀、并行重复劳动和推理停摆等问题：由 harness 持有研究状态，每个 agent 每次调用从状态重构专属上下文，Advisor 负责跨方向证据整合与实验分配，Worker 专注本地实现与迭代。在长达十亿 token 的预算下，SLA 在所有评测任务上均取得最优结果，并在内核优化任务中以比最强基线少 93.1% 的 token 达到同等性能。

## 研究问题与动机
1. **长周期推理增长不等于研究进展增长**：多数现有 AutoResearch 评估仅运行几千到数千万 token，或使用的 benchmark（如 26-circle packing）在极早期即饱和，无法揭示真正长周期下的失败模式。
2. **保留经验与控制上下文之间的张力**：候选方案、评估结果和失败尝试是可复用证据，但持续回放膨胀的历史会消耗大量 token 并导致 agent 行为退化（context rot），甚至引发过早停止。
3. **将累积证据转化为有效探索的协调难题**：记录的发现并不自动决定下一步要尝试什么。文中 CORAL 运行案例显示，在 GPU 内核任务中，最后一次改进发生在 231M tokens 处，之后 98% 的 1,500 个 session 均未发出工具调用，但 token 仍继续消耗——说明单纯增加推理量无法保证有效探索。
4. **评估尺度选择不当会误判方法性能**：短 horizon 评测可能掩盖真正优势，例如 SLA 在 FrontierSWE 四任务上在 175M tokens（25% 预算）时落后于最强基线，但在 700M tokens（满预算）时全面反超。

## 核心贡献（创新点）
1. **提出"有状态搜索、无状态智能体"（stateful search with stateless agents）原则**：将研究状态（候选方案与评估结果）完全交由 harness 持有，agent 每次调用均为全新会话，每次可见内容由 harness 按需从状态中重构，而非由累积对话自然生长。
2. **设计了双角色无状态协调架构**：Advisor 读取 harness 跨所有搜索方向的结构化证据摘要，负责分配具体实验给并行 Worker；Worker 仅获得本任务分配、已保存的候选及本地反馈，无法访问其他 Worker 的工作区。两者均无跨调用状态。
3. **提出证据驱动的实验分配机制（evidence-driven work assignment）**：Advisor 根据全局证据在每个 epoch 做出集中/分散策略选择，并通过"lane-local best"与"regime pivot re-seed from frontier"规则保留非当前最优但具潜力的探索方向，避免过早淘汰。
4. **构建面向十亿 token 级别的长周期评测协议与对比基准**：统一以累积 token 为尺度（替代迭代/墙钟时间），覆盖软件工程、GPU 内核优化、算法设计三类具有持续进步空间的 benchmark，并对齐相同模型、种子解与保护性 evaluator。
5. **提供从共享 checkpoint 出发的系统性消融**：证明聚焦上下文（focused context）与显式分配（explicit assignments）各自均能带来增益，且效果随运行长度累积放大；同时揭示 Advisor 开销不足 0.6% token，协调成本极低。

## 方法详解
**整体架构（图 3）**：SLA 以 epoch 为循环交替执行全局规划与局部实验。每个 epoch：
- Harness 从研究状态中重建 Advisor 的上下文（task spec + 当前全局最优代码与分数 + 证据摘要）。
- Advisor 基于跨方向证据输出针对 W 个 Worker 的具体分配（REGIME / PROPOSAL）。
- 每个 Worker 在隔离工作区内运行若干次本地 trial，每次均为一次新 session（无历史）。
- 独立 evaluator 对每个候选打分，harness 更新 Worker 本地候选与全局最优。

**Harness 控制的上下文重建**：
- **Advisor 上下文**：由 harness 生成结构化证据摘要，包含近期尝试、按搜索方向分组的 outcomes、Worker 状态、近期进展趋势及 task-specific 诊断；区分"已测量的不改进"与"实现/运行时/正确性不确定失败"，避免将实现错误误判为方向耗尽。
- **Worker 上下文**：每次 trial 仅包含 task spec、本次 assignment、本 Worker 本地保留候选、以及前一次 trial 的状态/分数/诊断反馈；禁止跨工作区访问与其他 Worker 历史。

**证据驱动的工作分配**：
- Advisor 每次 epoch 从头基于当前证据重新推理，而非依赖自身更早的对话状态。
- 分配约束：每个 Worker 只能拥有一个增量修改 lane；其余 lane 必须尝试具有机制级别影响的改动（数据布局、遍历顺序、依赖调度等）。
- 保留机制：当一个 Worker 的 candidate 虽未超过全局最优但优于自身 lane-local best 时仍被保留，给替代思路成熟时间；当 Advisor 重定向该 Worker 时，重置到全局最优作为新的 local baseline。
- 两种调度姿态：Diversify（多方向并行）和 Concentrate（多个 Worker 攻击同一瓶颈），需附证据说明选择理由。

**实现与安全保障**：
- Advisor/Worker 均使用 Codex 或 Claude Code，每次调用为全新 session。
- Worker 运行在 Linux Landlock 受限文件系统下，网络访问受限，session 在 trial 间清空。
- Evaluator 运行在独立受保护进程中，Worker 无法篡改评分逻辑。
- Harness 记录每个角色的 token 使用并按缓存/非缓存/输出区分，支持 checkpoint/recovery 与可控消融。

## 实验与结果
**数据集与任务**：
- FrontierSWE（4 题）：libexpat→x86-64 Assembly、Git→Zig、Dart→Haskell、Lua Native Compiler；预算 700M tokens。
- Anthropic VLIW SIMD 内核优化：预算 1B tokens。
- SOL-ExecBench（3 题 #1/#58/#210）：预算 500M tokens。
- FrontierCS Structured-LWE：预算 200M tokens。

**基线**：EvoX、CORAL、SwarmResearch（均使用官方上游版本，不改搜索逻辑）；统一使用相同 coding agent（Codex+GPT-5.5 / Claude Code+Opus 4.8）、seed 解与受保护 evaluator。

**主要结果**：
- **最终性能**：SLA 在所有任务上均取得最优。Anthropic 内核：SLA 达 1112.0±8.9 cycles，优于 SwarmResearch 1275.7±133.9；FrontierSWE 平均超最强基线 +4.95 分；SOL-ExecBench 平均提升 +5.5%；FrontierCS 提升 +2.6%。
- **Token 效率**：在 Anthropic 内核上，SLA 仅用 67.9M tokens 即达到 SwarmResearch 986.3M 才达到的水平，减少 **93.1%**（Claude Code 配置下减少 84.4%）；SOL-ExecBench #58 减少 95.9%。
- **Horizon 依赖性**：FrontierSWE 上 SLA 在 175M tokens（25% 预算）时四题均落后，但到 700M tokens（满预算）时全部反超，说明短预算评测会误判方法。
- **消融（Table 4）**：从共享 checkpoint 出发，去除 Advisor 重建、Worker 隔离或 Advisor 分配均降低进展；去除 Worker 隔离在 100M-token 续跑中损失 5–11 cycles，在全 1B-token 运行中扩大至 **224 cycles**（1336 vs 1112），证实无界上下文的代价随运行长度累积。
- **扩展 Worker 数（Table 6）**：1→15 Workers 在 Anthropic 内核上耗时缩短 6.6×，在 FrontierSWE 上缩短 7.3×；宽池在短预算落后（如 W=15 在 Anthropic 25% 预算落后 W=1 约 337 cycles）但全预算可追平至 4 cycles 以内。
- **Advisor 开销（Table 5）**：SLA Advisor 仅占 0.24–0.51% tokens 与 1.20–2.29% 成本；对比下 SwarmResearch Shepherd 占 8.39–10.70% tokens 与 6.14–7.35% 成本。
- **Worker 模型选择（Appendix C.1）**：以相同 $100 成本计，GPT-5.4 mini Worker 在 Git→Zig 上优于 GPT-5.5（20.88 vs 19.68），但内核任务反向，说明 Worker 模型可按任务定制。

## 相关工作脉络
1. **LLM-guided 进化搜索（FunSearch / AlphaEvolve / EvoX / AdaEvolve / ShinkaEvolve）**：这些方法将 LLM 生成的候选嵌入固定进化/搜索回路，每候选仅需 1–few 次模型调用；SLA 将这些方法与多步 coding agent 结合，并解决其长周期上下文与协调问题（尤其对 EvoX 做了 coding-agent 适配）。
2. **多 agent AutoResearch 框架（CORAL / SwarmResearch）**：两者均让 agent 在长期运行中携带研究历史；CORAL 通过持久 memory 和 heartbeat-triggered reflection 共享发现，SwarmResearch 用 Shepherd 做子 agent 选择与 selective context sharing；SLA 将历史移至 harness 并采用无状态 agent 以规避 context rot 与重复劳动。
3. **AI Scientist 类系统（The AI Scientist / Agent Laboratory / Kosmos / AI-Researcher）**：覆盖从假设生成到论文撰写的完整科研生命周期，以人工评审或发表为衡量；SLA 聚焦可在单一问题上有可执行 evaluator 的开放搜索，以累积 token 为可控缩放轴。
4. **Memory/Context 自增强方法（Dynamic Cheatsheet / ACE / ReasoningBank / MCE）**：大多解决跨任务经验转移（inter-task）；SLA 解决单任务内持续积累（intra-task）的可用性问题，二者可组合（如将 ACE playbook 作为 Advisor 上下文种子）。
5. **Context Rot 诊断与缓解（Hong et al. / Xia et al.）**：证明输入 token 增长会劣化 agent 行为；SLA 从架构层面规避此问题——每次调用上下文均受 harness 严格控制且重置。

## 局限性与未来方向
1. **多数配置仅单次运行**：受长周期成本限制，除 Anthropic 内核（Codex 三独立 run）与 checkpoint 消融外，其余任务多为单 run，统计显著性受限。
2. **仅测试具有可执行 evaluator 的任务**：目标模糊或评估缓慢的问题尚未覆盖。
3. **消融仅改变 agent 上下文内容，未改变其是否跨调用保留对话**：状态无状态本身的直接对比仍需更多验证。
4. **成本估计排除实验运行与评估的 compute 开销**：token 比例不能完全代表真实系统成本。
5. **未来方向**：将 SLA 作为 AI Scientist 的实验引擎模块、跨任务记忆（如 ACE playbook）注入 Advisor 上下文、多 Advisor 并行划分研究状态以降低单一 Advisor 决策负担、利用 SLA 产生的 episode 数据进行 RL 微调。

## 研究启发与可借鉴点
1. **"有状态搜索、无状态 agent"架构可复用于任何长 horizon agent 场景**：将持久状态从 agent 会话剥离，由外部 harness 控制上下文注入，可直接缓解 context rot 与重复劳动两大痛点；对多 agent 并行系统尤为适用。
2. **从共享 checkpoint 出发做逐组件消融**：通过冻结研究状态后仅改变每个 variant 的上下文内容或分配策略，能清晰识别各设计选择的边际贡献，且效应随运行长度累积放大的量化结果更具说服力。
3. **多 horizion 报告成为标准做法**：同一方法在不同 token 预算下的排名可能发生反转（如 FrontierSWE），建议评测协议至少报告 25%/50%/100% 三档累计 token 下的性能。
4. **协调者开销可低至 <1% 但决定全局探索质量**：Advisor 仅占 0.24–0.51% token，却通过证据摘要 + 显式分配显著提升搜索效率；这提示在长周期 agent 系统中投入少量"协调 token"是高度杠杆化的。
5. **Worker 模型可按任务/成本独立选优**：SLA 将协调与实现解耦，使得"更强但不一定更划算"的模型可只在 Advisor 侧使用，Worker 侧选择性价比最优的模型，后续可扩展至混合专家或 MoE 式 worker pool。

## 关键术语表
- **Stateless Language Agent（SLA）**：不具备跨调用持久状态的 LLM agent，每次调用均由 harness 从零重建其上下文。
- **Harness**：SLA 的外部控制层，持有全部研究状态（候选、评估结果、日志），负责上下文重建、分配生成与 evaluator 调用。
- **Evidence Summary**：harness 为 Advisor 生成的跨方向结构化证据摘要，区分已测量不改进与不确定失败，供全局决策。
- **Lane-local Best**：某一 Worker 在其当前 REGIME 内维护的局部最优候选，即使未超全局最优也会被保留，给替代方向成熟时间。
- **Regime Pivot Re-seed**：当 Advisor 切换 Worker 的 REGIME 时，将该 Worker 的代码重置为当前全局最优并清除 lane anchor 的机制。
- **Cumulative Token（累积 token）**：本次评估采用的 horizon 度量，按输入（缓存/非缓存）+ 输出 token 总和统计，提供跨并行度一致的可比尺度。
- **Context Rot**：随输入 token 持续增长导致 LLM 行为退化的现象，是长周期 agent 的核心失效模式之一。
- **Stateful Search with Stateless Agents**：SLA 的核心设计原则——搜索状态归 harness，agent 无状态，上下文为显式设计而非对话积累的副产品。

## 可复现要素
- **数据集**：FrontierSWE、Anthropic VLIW SIMD kernel（https://github.com/anthropics/original_performance_takehome）、SOL-ExecBench、FrontierCS——均为公开 benchmark。
- **代码**：论文未明确提供开源仓库链接（正文及附录未声明），仅给出 SLA 框架的详细实现细节与 prompt 模板；基线均使用上游官方 pinned release。
- **权重**：使用商业模型（GPT-5.5 / Claude Opus 4.8）与 OpenAI Codex CLI、Claude Code，不训练/微调模型权重。
- **关键超参**：默认 Worker 数 W=3，每 epoch 每 Worker 3 次 local trial；推理 effort 设为 high；Codex 内置 multi-agent 特性关闭；Landlock 文件系统与网络限制启用；checkpoint 间隔为 10% / 20% 预算。
