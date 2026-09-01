---
title: "Stick-to-What-You-Know-A-Study-of-Knowledge-Aligned-Supervis"
source: https://arxiv.org/pdf/2608.30987v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 10:59:42"
field: "大语言模型事实性与幻觉缓解"
keywords: ["知识对齐SFT", "事实性幻觉", "监督微调", "参数化知识", "幻觉缓解", "Recall Rewrite", "Evidence Rewrite"]
innovations: ["提出知识对齐SFT统一框架，将训练目标约束在基座模型参数化知识内；提出Recall Rewrite，通过多措辞探针+基座模型采样+蕴涵检验估算参数化知识；在统一设置下完成生成式与估计式对齐方法的受控对比及%Known因果消融"]
benchmarks: ["WildHalu", "Biography", "UnknownBench", "OLMES (HumanEval+, GSM8K, IFEval, TruthfulQA)"]
---

# 论文速读：Stick-to-What-You-Know-A-Study-of-Knowledge-Aligned-Supervised-Fine-Tuning

## 一句话总结
论文研究了对基座模型进行监督微调（SFT）时，若训练目标包含基座模型未内化的事实知识会诱发事实性幻觉的问题，提出"知识对齐 SFT"框架——将 SFT 训练目标约束在基座模型参数化知识范围内，并提出证据重写（Evidence Rewrite）和回忆重写（Recall Rewrite）两种新方法；在 Qwen 3 4B 和 OLMo 3 7B 上的实验表明，知识对齐 SFT 可显著降低事实幻觉且不明显损害通用能力。

## 研究问题与动机
1. **SFT 目标超出资深模型参数化知识的根源性幻觉**：SFT 训练数据中提供的"金标准"响应可能包含基座模型在预训练阶段未充分内化的长尾/小众事实，迫使模型去模仿生成超出其知识边界的陈述，从而诱发事实性幻觉（factuality hallucination）。
2. **现有缓解手段存在近似缺陷**：FLAME 等方法简单用基座模型生成替换金标准响应，但基座模型本身可能幻觉，且替换过程会丢失原响应中包含的、模型已知但未被生成的正确主张；UNIT 等方法依赖基于置信度的单步估计，易受表述方式与位置影响，且无法覆盖隐式元知识（meta-knowledge）。
3. **SFT 阶段不适合注入新的事实知识**：已有研究表明可靠地学习新事实关联需要大量证据支撑（Kandpal et al., 2023），而 SFT 阶段相对较小（Guo et al., 2025；Team OLMo, 2025）；更好的做法是检索增强或继续预训练。因此本文聚焦"对齐已有知识边界"而非"扩展知识边界"。
4. **训练数据中的目标知识分布与幻觉行为存在因果关联**：通过固定其他条件仅改变训练目标中"已知主张"的比例（Section 4.4），复现了更严格的知识对齐带来更低幻觉的因果效应，确认该问题机制的真实存在。

## 核心贡献（创新点）
1. **提出了统一的"知识对齐 SFT"形式化框架**：将 SFT 训练目标中的原子主张分类为"对基座模型已知/未知"，构建了一个以统一算法模板实现各类方法的框架（Algorithm 1），使生成式对齐（FLAME）、估计式对齐（UNIT）及新提出的两种方法在同一比较基准下可比。
2. **提出 Evidence Rewrite（证据重写）**：改进了 FLAME 的生成式对齐策略，在基座模型生成后引入外部证据检索 + 事实核查管线（VeriScore 分解 + Wikipedia 检索 + FActScore 验证），仅保留有外部证据支持的声明并重写为合规训练目标，填补了"纯生成"策略无法过滤模型自身错误声明的空白。
3. **提出 Recall Rewrite（回忆重写，主要创新）**：不依赖外部证据，而是通过多轮探测问题 + 基座模型采样回答 + 蕴涵检验来估算模型的参数化知识，仅在基座模型能一致召回（consistently recalled）时保留对应声明；其本质区别在于用"行为一致可召回"替代"单步置信度阈值"，同时覆盖显式主张与隐式元知识。
4. **在统一设置下完成了生成式/估计式对齐方法的受控对比**：首次在同构的训练设置（相同的 OASST1 子集、相同超参、相同评测流程）下系统比较了 Standard SFT、FLAME、UNIT、Evidence Rewrite 与 Recall Rewrite，揭示了各方法在幻觉抑制、拒答行为和通用能力之间的权衡关系，并通过"%Known 比例"消融实验给出了清晰的因果证据。

## 方法详解
- **知识对齐 SFT 的形式化定义**：
  - 令 $\mathcal{W}$ 为世界知识，$\mathcal{K}(M_{\text{base}})$ 为基座模型参数化知识（内在不完整）；
  - 每个 SFT 样本 $(P, R)$ 被分解为原子主张集合 $\mathcal{C}(R|P)$，包含（i）事实/程序/结构性主张和（ii）不在 $P$ 中需模型自己提供的信息，包括构造响应所必需的隐式元知识（如写俳句需要三行结构和音节约束）；
  - 知识对齐目标：构造 $\mathcal{D}^*$ 使得对所有 $(P, R^*) \in \mathcal{D}^*$ 均有 $\mathcal{C}(R^*|P) \subseteq \mathcal{K}(M_{\text{base}})$。

- **统一数据构建算法（Algorithm 1）**：所有方法共享三步钩子 GATE、SOURCE、UNKNOWN 及可选的 $M_{\text{rewriter}}$，依据方法不同实例化这些钩子（详见 Table 7）。

- **FLAME（生成式对齐基线）**：
  - 对知识寻求型提示 $P$，从 $M_{\text{base}}$ 采样 $\hat{R}$ 替换 $R$；非知识型提示保留 $R$；
  - 隐含假设所有自生成主张均为"已知"，$\text{UNKNOWN}(c_n)$ 恒为 false。

- **$\mathrm{UNIT}_{cut}$（估计式对齐基线）**：
  - 对 $R$ 中每原子主张 $c_n$ 计算 CCP（claim-conditioned probability），若 $\mathrm{CCP}(c_n) \leq \tau$ 则裁去；
  - 仅依赖 token 级置信度代理信号，对表述和位置敏感。

- **Evidence Rewrite（ours）**：
  - 知识寻求提示：先让 $M_{\text{base}}$ 生成短回答，再经 brainstorming 提示扩展为更长更详细的 $\hat{R}$（缓解因裁剪导致的过短回复）；
  - 对 $\hat{R}$ 执行三段事实核查管线：VeriScore 分解 → Wikipedia 分层检索（top-5）→ FActScore 验证；
  - 使用 $M_{\text{rewriter}}$（gpt-4o-mini）基于 prompt 和支持性声明重写 $R^*$；若信息不足则返回拒答模板；
  - $\text{UNKNOWN}(c_n)$ 由外部证据支持性判定决定。

- **Recall Rewrite（ours，核心创新）**：
  - 仅对知识依赖型（knowledge-dependent）主张进行探测，非知识依赖型（contextual/subjective/reasoning）主张直接保留；
  - 对每个知识依赖主张 $c_n$：用教师模型生成 $J=5$ 个独立的探测问题 $\{q_{n,j}\}$；从 $M_{\text{base}}$ 对每个问题采样 $K=2$ 个回答 $\{y_{n,j,k}\}$（temperature=0.5）；
  - 对每个 $(q_{n,j}, y_{n,j,k})$ 做蕴涵检验：判断 $y_{n,j,k}$ 与 $c_n$ 的关系为 ENTAILS / CONTRADICTS / UNRELATED；
  - 定义 $e_{n,j}$ 为对 $q_{n,j}$ 的回答中 ENTAILS 的数量，$d_{n,j}$ 为 CONTRADICTS 的数量；
  - 一问题判定为 entailing 当 $e_{n,j} \geq k_e$，contradicting 当 $d_{n,j} \geq k_c$；
  - $c_n$ 被"一致召回"（consistently recalled）的条件（Equation 1）：
    $$
    |\{j : e_{n,j} \geq k_e\}| \geq j_e \quad \land \quad |\{j : d_{n,j} \geq k_c\}| \leq j_c
    $$
    默认阈值 $j_e/k_e/j_c/k_c = 2/1/2/1$；
  - 条件 (i) 防止探针泄漏答案（over-specified probe），条件 (ii) 容忍少量因探针表述不清导致的误判（under-specified probe）；
  - 最终由 $M_{\text{rewriter}}$（gpt-5-mini）以局部编辑方式去除被标记为未知的声明内容，保留原文结构与风格；信息不足时返回拒答模板。

## 实验与结果
- **基座模型**：Qwen 3 4B-Base、OLMo 3 7B-Base
- **SFT 数据集**：OASST1 英文首轮对话子集，共 3,468 条（另设 Tülu 3 大规模对照）
- **评测基准**：
  - WildHalu（500 个真实实体，约半数无 Wikipedia 页面，Google Search 检索证据）
  - Biography/Bios（500 个有 Wikipedia 页面的人物，Wikipedia 为唯一证据源）
  - UnknownBench（FalseQA / NEC / RefuNQ 三个拒答子任务）
  - 通用能力：OLMES 平均（HumanEval+、GSM8K、IFEval、TruthfulQA）
- **Factuality 指标**：#Supp.（支持声明数）、%Supp.（非拒答响应中支持声明占比）、FActScore（样本级支持率，拒答视为全支持）
- **主要结果（Qwen 3 4B，OASST1，Table 1）**：

| 方法 | WildHalu %Supp. | WildHalu FActScore | Bios %Supp. | Bios FActScore |
|---|---|---|---|---|
| Standard SFT | 76.6*** | 74.4*** | 36.0*** | 34.1*** |
| FLAME | 73.0*** | 74.4*** | 34.0*** | 33.4*** |
| $\mathrm{UNIT}_{cut}$ | **81.2**\* | **79.4**\* | **45.1**\*\*\* | **43.1**\*\*\* |
| Evidence Rewrite | 80.1 | 78.3 | 42.3 | 39.9 |
| **Recall Rewrite** | **84.2** | **84.1** | **56.2** | **76.4** |

  - Recall Rewrite 在两项任务上均取得最高 %Supp. 与 FActScore；WildHalu 相对 Standard SFT 提升 **+9.7 个百分点**（%Supp.），Bios 提升 **+20.2 个百分点**；但 #Supp. 下降且拒答增加（WildHalu 55 例，Bios 252 例）。
  - FLAME 未超越 Standard SFT，说明"朴素自生成"不是可靠的参数化知识代理。
  - Evidence Rewrite 与 UNIT 折衷于覆盖率与事实性之间；Recall Rewrite 以更高的保守性换取最强的事实性指标。

- **%Known 比例消融（Table 3，固定训练样本数与非拒答/拒答数）**：
  - 100% Known：FActScore 86.1（WildHalu）、69.9（Bios）为最优；
  - 50% / 0% 依次下降，呈现严格的单调关系，证实"更多已知主张 → 更低幻觉"的因果效应。

- **Refusal Behavior（UnknownBench，Table 4）**：
  - Recall Rewrite 拒答召回（Recall）最高：FalseQA 64.1、NEC 66.3、RefuNQ 79.1；F1 亦最高（68.7 / 68.8 / 69.9）；
  - 代价是精确率较低（更多误拒，false refusals），呈现精度-召回权衡。

- **多阶段后训练对比（OLMo 3 7B，Table 2）**：
  - DPO / RLVR 逐步改善事实性；Recall Rewrite 仅靠 SFT 在 WildHalu 上超过所有官方检查点（FActScore 82.5 vs. RLVR 78.4）；

- **通用能力（Table 5）**：
  - Recall Rewrite 的 OLMES 平均（68.9）仅比 Standard SFT（69.8）低 0.9 个点，处于所有 OASST1 模型 2.1 点区间内；
  - IFEval 降低主要因 Recall Rewrite 拒答了 30 个创意写作类提示；其余 504 个提示上与 Standard SFT 持平（56.3 vs. 56.5）。

- **结论**：知识对齐 SFT 在保持通用能力的同时显著降低事实幻觉；最大收益来自 Recall Rewrite 的一直召回策略，但也带来更保守的拒答政策。

## 相关工作脉络
1. **FLAME（Lin et al., 2024）**：用基座模型生成替换 SFT 金标准响应，属于"生成式对齐"的最早代表；本文揭示其因未过滤模型自身错误而产生噪声，并在此基础上引入外部证据校验。
2. **UNIT（Wu et al., 2025）**：基于 CCP（claim-conditioned probability）阈值裁剪训练声明；属于"估计式对齐"；本文指出 CCP 对表述敏感且仅覆盖显式主张，提出了更稳健的"行为一致召回"替代方案。
3. **Gekhman et al. (2024)**：在封闭 QA 与开放域指令数据上证明训练包含未知知识会增加幻觉；本文在其基础上将问题扩展至"长文本 SFT 训练目标"层面，并以 %Known 消融给出因果证据。
4. **Kaplan et al. (2026)**：将 SFT 诱导幻觉归因于"事实遗忘"（参数干扰），并提出优化层正则；本文则聚焦"训练目标超出知识边界"这一数据侧因素，与正则化方案正交。
5. **Liu et al. (2025) / Ovadia et al. (2024)**：通过继续预训练注入新事实；本文明确区分了"注入知识"与"对齐已有知识"两条路线，主张后者更适合 SFT 阶段。
6. **Calderon et al. (2026)**（同期工作）：同样用"多措辞问题一致回答"定义事实已知；与本文 Recall Rewrite 的核心思想（consistent recall）高度呼应，但本文提供了更完整的统一框架与对照实验。

## 局限性与未来方向
1. **知识边界的二值化近似**：本文把参数化知识处理为"已知/未知"二元属性，忽略了现实中可能存在的梯度置信度、部分掌握或表述敏感等问题；更细粒度的建模有望进一步提升质量。
2. **评估依赖自动事实核查管线**：claim 分解、证据检索与验证均可能出错（尤其对表述模糊或需要领域专业知识的声明）；虽采用成熟管线并保持跨方法一致性，但仍是测量误差来源。
3. **不与 DPO/RLVR 等后训练阶段联合测试**：仅通过 OLMo 对比间接观察，尚未探索知识对齐 SFT 与后续偏好优化、强化学习等事实性导向目标的组合效果与叠加性。
4. **可扩展性受限**：Recall Rewrite 的"分解→探针生成→多轮采样→蕴涵检验→重写"管线成本较高（OASST1 3,468 条样本 API 费用约 44 USD），当前更适合作为高精度诊断干预而非全量 SFT 数据准备方案；复杂多事实响应的探针生成更具挑战性。
5. **数据规模与多样性有限**：仅使用 OASST1 英文首轮对话；尚未在大规模指令混合数据、多语言、多轮对话、工具使用、代码生成或推理密集型场景上验证。
6. **强教师模型的偏差**：pipeline 依赖 gpt-4o-mini / gpt-5-mini 等强教师模型，可能在声明选择、问题生成、拒答风格等方面引入自身偏差；多教师或人工审计有助于分离教师特定效应。

## 研究启发与可借鉴点
1. **"一致召回"评估参数化知识的范式可迁移**：Recall Rewrite 提出的"多措辞探针 + 多次采样 + 蕴涵检验"可用于任何需要估算 LLM 内部事实知识的场景（如知识图谱补全、幻觉检测、训练数据优选），作为一种零外部证据依赖的轻量级探针方法。
2. **按声明粒度分解训练目标是精细对齐的有效路径**：将响应拆解为原子主张后再按知识状态分类，既保留了响应整体结构的可学习性，又实现了细粒度的知识边界控制；该方法可迁移到代码生成、推理链生成等其他领域的数据清洗管线。
3. **SFT 阶段不适合注入新知识这一原则值得推广**：本文的因果消融（%Known 消融）为工程实践提供了明确指引——新事实的可靠注入应优先考虑检索增强或继续预训练，SFT 的核心任务应聚焦于"对齐已有知识边界"和"规范响应行为"。
4. **覆盖率-事实性权衡的可视化工具可用**：Figure 5 的 coverage-factuality 散点图与 Figure 7 的 precision-recall 曲线为后续工作提供了直观的方法比较框架，可在文献综述中作为多方法对比的通用可视化模板。
5. **与多阶段后训练结合的潜力**：本文间接表明知识对齐 SFT 的增益可能与 DPO/RLVR 互补；这为本团队后续研究提供了一个清晰的组合实验设计机会——先做知识对齐 SFT，再叠加偏好优化或 RLVR。

## 关键术语表
- **知识对齐 SFT（Knowledge-aligned SFT）**：将 SFT 训练目标的原子主张约束在基座模型参数化知识 $\mathcal{K}(M_{\text{base}})$ 内的微调范式，旨在降低模型因模仿超出其知识边界的训练目标而产生的事实性幻觉。
- **参数化知识（Parametric knowledge）**：基座模型在预训练阶段已稳健内化的事实性知识集合 $\mathcal{K}(M_{\text{base}})$，是不完整但可被行为探测近似的核心对象。
- **知识依赖型主张（Knowledge-dependent claim）**：编码可验证事实、程序或结构信息、必须依赖模型参数化知识才能正确生成的原子声明；与之对应的是非知识依赖型（纯上下文、主观或通用推理）声明。
- **一致召回（Consistently recalled）**：Recall Rewrite 的核心判定标准：一个主张能被基座模型在多个独立措辞的探测问题上一致地恢复（满足 entailing 数量阈值且 contradiction 数量在容忍范围内）。
- **FActScore**：每样本级别的支持声明比率平均值，拒答被视为完全支持；用于量化长文本生成中的事实准确性（Min et al., 2023）。
- **WildHalu**：包含 500 个真实实体查询的开放域长文本事实性评测基准，约半数实体无 Wikipedia 页面，使用 Google Search 检索证据（Zhao et al., 2024）。
- **UnknownBench**：评估模型拒答行为的基准，包含 FalseQA、NEC、RefuNQ 三个子任务，衡量模型对无法回答问题的识别能力（Liu et al., 2024）。
- **OLMES**：标准化的语言模型通用能力评测框架，汇总 HumanEval+、GSM8K、IFEval、TruthfulQA 四项指标（Gu et al., 2025）。

## 可复现要素
- **数据集**：OASST1 英文首轮子集（3,468 条），为众包多轮对话树；WildHalu、Biography、UnknownBench、HumanEval+、GSM8K、IFEval、TruthfulQA 均为公开基准。**论文已公开 Recall Rewrite 在 Qwen 3 4B 和 OLMo 3 7B 上处理后的对齐训练数据及全部中间管线输出**（见 Footnote 3）。
- **代码/权重**：SFT 通过 HuggingFace TRL 库实现（开源）；基座模型 Qwen 3 4B-Base 和 OLMo 3 7B-Base 权重公开；FLAME / UNIT 复现代码参考原论文；证据检索使用 Wikimedia API 及 gtr-t5-large 重排器（公开）。**论文未明确提供整体 pipeline 的统一开源代码仓库链接**。
- **关键超参**：
  - SFT：epochs=5，batch_size=32，lr=1e-5，cosine warmup=0.1，weight_decay=0.1，context_length=1024；
  - Recall Rewrite：J=5（探测问题数），K=2（每问题采样回答数），temperature=0.5，阈值 $j_e/k_e/j_c/k_c = 2/1/2/1$；
  - 证据检索：Wikipedia top-5 chunks，gtr-t5-large reranker；
  - 声明分解：VeriScore prompt；验证：FActScore prompt（gpt-4o-mini）；问题生成/重写/判定：gpt-5-mini。
