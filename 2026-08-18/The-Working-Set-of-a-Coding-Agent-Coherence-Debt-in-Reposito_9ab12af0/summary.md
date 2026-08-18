---
title: "The-Working-Set-of-a-Coding-Agent-Coherence-Debt-in-Reposito"
source: https://arxiv.org/pdf/2608.16630v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:34:02"
field: "AI for Software Engineering"
keywords: ["coherence debt", "coding agent", "context window", "repository-scale", "working set", "fault injection", "multi-agent decomposition"]
innovations: ["提出一致性债务的形式化定义并证明双通道可互换且距离无关", "通过 1089 次枚举证明 event-only 估计器系统性高估参数覆盖率", "发现 stale standard 比 no standard 更有害（100% 遵循书面规范）"]
benchmarks: ["SWE-bench Verified", "Pydantic v1→v2 migration (79 tests)", "fictional API migrations (Sprocket/Grimwire/Kestrix/Zynet)", "synthetic coherence tasks"]
---

# 论文速读：The-Working-Set-of-a-Coding-Agent-Coherence-Debt-in-Reposito

## 一句话总结
论文将仓库级代码编辑建模为"耦合事实图"的重构问题，提出**一致性债务（coherence debt）**概念——当编辑所需的关键事实在上下文和模型参数记忆中均不可用时产生的缺口；通过系统性地控制两个信息来源通道，证明**可可用性决定成败而距离无关**，并揭示了现有基于读事件指标的致命缺陷。

## 研究问题与动机
- 仓库级编码任务要求 agent 在编辑时保持测试、导入、配置和迁移规则之间的一致性，但现有评估只观察最终补丁，忽略了编辑时事实可用性的因果机制。
- 现有 trajectory 分析方法依赖读事件来推断 agent 获取了多少上下文，但 agent 面对缺失事实时会"编造"而非停止，导致读指标系统性高估覆盖率。
- 无法区分两种信息通道（近期上下文 vs. 参数记忆）对编辑结果的独立贡献，也难以判断"读得多"是否等价于"覆盖好"。
- 多 agent 分解和长上下文管理策略缺乏统一理论框架来解释何时有效、何时有害。

## 核心贡献（创新点）
1. **提出一致性债务形式化定义**：以编辑时刻为粒度，将任务耦合事实集 $C_T^{(i)}$ 与两个可用通道（$R_t$ 近期上下文、$K_M$ 参数记忆）做集合差，定义 $D(e_i) = |C_T^{(i)} \setminus (R_{t_i} \cup K_M)|$，首次将虚拟内存的 working-set 思想迁移到无地址的仓库事实场景。与已有工作的本质区别：不是相关性分析而是因果干预——主动操纵每个通道的可用性。

2. **证明两个通道可互换且距离无关**：通过封闭-book 实验（154 次全零基线→300 次中 299 次恢复到≥9/12）、重命名对抗实验（7 个模型在相同 24/79 测试上失败）和距离操控实验（128,000 字符外与编辑旁表现相同），证实 coverage 通过"是否存在"起作用而非位置。与已有工作的本质区别：此前长上下文研究关注 positional degradation（如 Lost in the Middle），本文证明对耦合事实而言距离无关。

3. **揭示读事件指标的系统性偏差**：通过 1,089 次模拟器枚举（所有 $3^k$ 分配，$k=2..6$）严格证明：event-only 估计器报告的缺失集 = 真实缺失集 ∪ 参数覆盖集，其高估值恰好等于参数覆盖率；并用真实实验验证（124 次全覆盖运行中 119 次通过但估计器仍报 miss）。与已有工作的本质区别：不是经验观察而是形式化命题与枚举证明。

4. **发现 stale standard 比 no standard 更有害**：在书面规范与实际代码冲突的 39 次试验中 agent 100% 遵循书面规范（95% Wilson 区间 [0.91, 1.00]），且有规范的更差形式时正确写法比例从 33% 降至 0%。与已有工作的本质区别：打破了"提供更多上下文总更好"的隐含假设。

5. **建立 fault-injection 线性代价曲线**：合成的耦合事实任务中，每 withholding 一个 motif（4 个测试）恰好损失 4 个测试通过数，最大偏差为 0（$n=6$ per cell），证明 damage adds rather than compounds。与已有工作的本质区别：此前工作无法隔离单一事实的影响，本文通过作者已知 $C_T^{(i)}$ 的合成任务实现精确控制。

## 方法详解
- **耦合事实图模型**：任务 $T$ 诱导图 $G_T = (V_T, E_T)$，节点为原子事实（符号、测试、配置值、导入、迁移规则、不变量），边表示一致性约束。$C_T \subseteq V_T$ 是任务 oracle 通过所需的最小事实集合，$C_T^{(i)}$ 是编辑 $e_i$ 所需子集。测试与文本规则耦合了无静态依赖关系的文件，因此 $G_T$ 远超导入图。

- **双通道一致性债务公式**：
  $$D(e_i) = |C_T^{(i)} \setminus (R_{t_i} \cup K_M)|$$
  其中 $R_{t_i}$ 为编辑时刻有效上下文中的事实，$K_M$ 为模型参数记忆中的事实。关键预测：union 操作意味着两个通道可互换，成功应取决于 coverage 而非来源。

- **可观察代理指标（ residency score ）**：
  $$\rho_w(f_i, t_i) = \frac{|N(f_i) \cap \mathrm{Read}(t_i - w, t_i)|}{|N(f_i)|}$$
  其中 $N(f_i)$ 为文件的单跳导入邻居，$\mathrm{Read}$ 为前 $w$ 个工具事件内读取的文件集合。 sweeps $w \in \{4, 8, 16, 32, 64, 128\}$。论文强调 $1-\rho_w$ 计算的是"未读邻居文件数"而非"缺失事实数"，是对真实 coherence debt 的下界。

- **五事件流语义**：定义 file_read、fact_extracted、edit_intent、test_feedback、revert 五种事件类型，每种携带结构化 payload。edit_intent 记录 agent 声称的 pre-write 支持集，暴露缺失和过时依赖。该设计使四种失败原因可分离：working-set miss、stale read、handoff gap、speculative write。

- **形式化命题（Proposition 1）**：event-only 估计器的误差 = 参数覆盖率。证明：将事实划分为 $R$（读覆盖）、$P$（参数覆盖）、$U$（未覆盖），则 $\widehat{U} = C \setminus R = U \cup P$，因此 $|\widehat{U}| - |U| = |P|$。

- **实验设计三支柱**：① 封闭-book 通道控制（4 个虚构 API + 1 个真实 Pydantic + 重命名变体）；② 工具使用工作量（122 个匹配试验，4 个 harness）；③ 合成一致性任务（随机 literal 耦合 3 个文件，精确知道 $C_T^{(i)}$）。

## 实验与结果
- **封闭-book 基线**：154 次新虚构 API 试验，所有模型得分 0/12（Wilson 95% 上界 2.4%），确认新颖性本身不必然导致零分（Flareforge  workload 有类比得分 3/12），真正原因是缺失先验覆盖。
- **前置加载恢复**：将确切规则/源文件放入 prompt 后，300 次中 299 次达到 ≥9/12，213 次满足全部要求。Kestrix Jaccard=1.00，Sprocket/Zynet=0.99，Grimwire=0.79。
- **共享失败点**：Pydantic 关闭书中 6/7 家族收敛于相同 53/79 测试；重命名后 66/70 次试验跨 7 个家族通过完全相同的 24/79 测试（Jaccard=1.000），证明参数记忆的边界是可观测的。
- **线性代价曲线**：合成任务中 withholding $m$ 个 motif 恰好损失 $4m$ 个测试，最大偏差=0（$n=6$ per cell，两个删除方式结果相同）。
- **距离无关**：提供的实现在 128,000 字符（工具 harness）和 200,000 字符（封闭 book）范围内成功率保持 ceiling，而 withheld 条件一律归零。
- **不变量保留**：16 条不变量存入约 140,000 token 的上下文后，agent 在 7 次试验的每个位置均遵守，从不重读原文。
- **Harness 花费差异**：144 次全通过试验中，峰值上下文仅差 1.8×，但累计输入差 12.8×（293,882 ~ 3,752,134 tokens）； withholding 事实后所有配置损失比例相同，昂贵配置不恢复更多。
- **代理补偿行为**：Opus 100% 报告 blocked，Codex CLI 0% blocked（全部自信错误编辑），Haiku 12.5% blocked + 37.5% 直接编造文件。
- **冲突源实验**：39 次试验 agent 100% 遵循书面标准（即使标准较差），95% Wilson 区间 [0.91, 1.00]。
- **Stale standard 代价**：正确标准→100% 更好形式；无标准→33%；过时标准（要求更差形式）→0%。
- **SWE-bench 转移失败**：100 个实例 397 次评分中，residency score 的 AUC≈0.49（随机水平），因模型可能已通过 $K_M$ 覆盖事实。

## 相关工作脉络
- **SWE-bench 及仓库级评测**（Jimenez et al. 2024）：建立仓库级评估基准，本文聚焦编辑时刻的事实可用性因果机制，而非最终 resolve 率。
- **工具使用 agent 循环**（SWE-agent Yang et al. 2024; OpenHands Wang et al. 2025; Aider Gauthier 2023）：本文承认这些工作建立了工具循环范式，但指出其 trajectory 分析停留在事后诊断，缺乏 edit-time 的事实可用性因果干预。
- **上下文检索研究**（ContextBench Li et al. 2026; CORE-Bench Zhang et al. 2026a; SWE-Explore Zhang et al. 2026b）：这些工作评估检索质量，本文证明即使检索正确，event-only 指标也因忽视参数通道而系统性高估覆盖率。
- **Coherence Collapse**（Kim et al. 2026）与 **Strained Coherence**（Pandya et al. 2026）：这两者与本文共享词汇但分类已完成的轨迹；本文在编辑前主动操纵事实可用性，发现事实仅在 resident 时才有用。
- **长上下文 positional 研究**（Lost in the Middle Liu et al. 2024; RULER Hsieh et al. 2024）：本文证明对耦合仓库事实而言距离无关（只要存在），与这些工作中观察到的 positional degradation 形成对照。
- **多 agent 分解**（Yang et al. 2026; Pan and Luo 2026）：本文通过耦合图解释为何分解在紧密耦合任务上有害、在独立修复上安全，与信息论上限研究相互印证。

## 局限性与未来方向
- ** residency score 在真实仓库中失效**：在 SWE-bench Verified 上 AUC≈0.49，因模型可能已通过参数记忆覆盖常见仓库，且导入图对 issue 特定耦合的代理质量低。
- **冲突权威性问题未解决**：当两个覆盖事实矛盾时，agent 遵循书面标准，但权威、模态、读取顺序各自的贡献无法分离。
- **合成任务的生态效度有限**：虚构 API 消除了直接 API 暴露但无法消除通用先验；marker 评估器约束了指定变换而非工程-quality。
- **单文件编辑不适用**：框架针对跨文件一致性的任务，greenfield 开发可能边写边构建事实图，行为可能不同。
- **未来方向**： preregistered task-held-out 距离实验、独立作者的非迁移任务上对比 import/lexical/heterogeneous-graph/dataflow 检索、构建 union-aware 的事件估计器替代现有 read-derived 指标。

## 研究启发与可借鉴点
1. **fault injection 线性代价曲线作为诊断工具**：通过精确控制 withheld 事实数量并观察线性的测试损失，可以验证任何 fact-availability 假设是否成立；合成任务中已知 $C_T^{(i)}$ 的设计模式值得迁移到本团队的 agent 评估流水线。
2. **event-only 估计器的系统性偏差警示**：任何基于读事件推断上下文覆盖率的指标都必须排除 agent 自写文件、考虑参数覆盖的补集；这直接影响对本团队历史 trajectory 数据的重新分析。
3. **双通道解耦的实验范式**：封闭-book（$R_t=\varnothing$）+ front-load（$K_M$ 被 bypass）+ 重命名（$K_M$ 被 defeat）的三重对照设计，可复用于评估不同模型/ harness 的组合策略。
4. **stale standard > no standard 的工程启示**：在 harness 设计中，过时约定文件的危害大于不存在，应建立约定文件的 versioning 和 stale 检测机制，而非简单地在 prompt 中附加越多上下文越好。
5. **分解风险与耦合图的关联**：多 agent 分解的安全性取决于任务图的 cut 是否跨越耦合边；可在本团队的 multi-agent 调度系统中引入耦合图分析作为 partition 前的预检。

## 关键术语表
**Coherence Debt（一致性债务）**：编辑时刻任务所需事实中，既不在近期上下文也不在模型参数记忆中的那部分事实的数量，$D(e_i) = |C_T^{(i)} \setminus (R_{t_i} \cup K_M)|$。

**Coupled-fact Graph（耦合事实图）**：任务诱导的图 $G_T$，节点为原子事实（符号、测试、配置值等），边表示一致性约束；测试和文本规则可耦合无静态依赖关系的文件。

**Residency Score（居住分）**：代理指标 $\rho_w$，计算编辑前 $w$ 个工具事件内读取的文件占目标文件导入邻居的比例，是对真实 coherence debt 的下界估计。

**Parametric Memory（参数记忆）**：模型 $M$ 通过训练内化的知识通道 $K_M$，对熟悉 API 可靠但对 Novel API 不可靠；event-only 指标对此通道不可见。

**Throttling/Thrashing（抖动）**：当 uncovered debt 在多次 read-edit-test 循环中持续存在且无新证据使其 retired 时的轨迹状态。

**Union-aware Estimator（并集感知估计器）**：同时考虑读覆盖和参数覆盖的事实估计器，在 1,089 次模拟器运行中达到 100% 精确；event-only 估计器因忽略 $K_M$ 而系统性高估缺失集。

**Front-loading（前置加载）**：在编辑前将全部 $C_T$  facts 放入 prompt 的实验条件，使 $D(e_i)=0$ by construction。

**Working-set Miss（工作集缺失）**：编辑所需事实对当前编辑 actor 无现成提取的失败原因，区别于 stale read、handoff gap 和 speculative write。

## 可复现要素
- **数据集**：4 个手编虚构 API（Sprocket/Rust, Grimwire/Go, Kestrix/Python, Zynet/JS）+ 1 个真实 Pydantic v1→v2 迁移（79 测试）+ 重命名变体 + 合成 coherence 任务；代码与完整 trial ledger 开源：https://github.com/mpi-dsg/agent-coherence
- **模型**：Claude（Sonnet/Fable/Haiku/Opus via Claude Code）、GPT-5（Codex CLI）、DeepSeek-Coder-V2-Lite（16B, 本地 vLLM）、Qwen3-Coder-30B-A3B（本地 vLLM）、Z.ai、Gemini
- **Harness**：Claude Code、Codex CLI、Aider、OpenHands、opencode
- **关键超参**：residency window $w \in \{4,8,16,32,64,128\}$；合成任务 8 motifs × 4 tests/motif；距离实验最大 200K characters；不变量保留实验最多 ~140K tokens
- **封闭-book 沙箱**：denylist 策略拒绝项目根目录读取，仅允许 scratch 目录和 CLI 配置路径
- **重复性**：30 个 cell 在独立批次中重跑，4.8% 的匹配试验结果存在 batch-to-batch 分歧
