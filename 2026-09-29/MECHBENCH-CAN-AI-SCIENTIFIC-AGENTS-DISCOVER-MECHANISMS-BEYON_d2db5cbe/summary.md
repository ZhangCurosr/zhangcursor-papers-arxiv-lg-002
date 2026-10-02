---
title: "MECHBENCH-CAN-AI-SCIENTIFIC-AGENTS-DISCOVER-MECHANISMS-BEYON"
source: https://arxiv.org/pdf/2609.35515v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:53:21"
field: "AI for Science / 自动科学发现"
keywords: ["mechanism discovery", "scientific agent", "symbolic regression", "benchmark", "phenomenal law", "mechanism probe", "mutational variant"]
innovations: ["提出MECHBENCH基准，将现象定律恢复与机制发现分离评估", "引入机制探针与机制不可分辨性筛查，使机制恢复可客观度量", "构建可控科学突变体系，揭示AI代理在机制推理上的泛化鸿沟"]
benchmarks: ["MECHBench (Core-set n=80, Full-set n=512)", "LLM-SRBench", "NewtonBench", "Science-Gym", "SciLaws-Bench"]
---

# 论文速读：MECHBENCH-CAN-AI-SCIENTIFIC-AGENTS-DISCOVER-MECHANISMS-BEYON

## 一句话总结
论文提出MECHBENCH基准测试，将AI科学发现的评估从“恢复可观察现象定律”扩展到“发现生成机制”；实验表明当前代表性科学代理在现象定律恢复正确时仍有超六成概率无法重建内在机制，且随着机制变异程度增加，这一“现象‑机制恢复鸿沟”显著扩大。

## 研究问题与动机
- 现有符号回归与科学代理基准（如LLM‑SRBench、NewtonBench）主要评估代理从观测数据中恢复现象定律（phenomenal law）的能力，但现象定律仅描述可观察输入‑输出关系，不包含生成该关系的内部因果结构。
- 机制发现要求推断不可直接观测的内部科学量及其关系，能力要求更高；现有评估无法区分“记忆教科书公式”与“真正重构机制”。
- 机制评估面临**机制不可分辨性**（mechanistic indistinguishability）挑战：不同机制可能蕴含同一现象定律，导致客观评估困难。
- 需要一套可控制、可解释且能排除记忆偏置的基准，以独立度量机制发现能力。

## 核心贡献（创新点）
1. **提出MECHBENCH基准**：明确分离现象定律恢复与机制发现两类能力，每个任务由机制模型生成观测数据，评估时需重建生成机制。
2. **引入机制探针（mechanism probes）**：通过查询仅由机制隐含、无法从现象定律单独推导的内部科学量，客观检验机制是否被真正重建。
3. **可控机制突变构建陌生变体**：对经典机制进行科学可解释的局部修改（如改变相互作用缩放律、引入泄漏项等），组合多突变形成复杂变体，以降低对预训练知识的依赖。
4. **机制不可分辨性筛查**：在构建实例时主动搜索是否存在竞争性机制能产生相同现象定律，排除此类模糊实例，确保探针评估的有效性。
5. **系统揭示现象‑机制恢复鸿沟**：在多个Agent‑模型组合下实验，证明即使现象定律恢复正确，机制恢复仍常失败，且随突变数量增加差距急剧扩大。

## 方法详解
- **机制形式化**：机制 $\mathcal{M} = \{m_1, \ldots, m_J\}$ 为包含可观察变量与内部变量的关系集合，满足 $\mathcal{M}(\mathbf{x}, y, \mathbf{h}) \Longrightarrow \mathcal{P}(\mathbf{x}, y)$，其中 $\mathcal{P}$ 为现象定律。
- **任务生成**：对每个机制 $\mathcal{M}$，消去内部变量得到 $\mathcal{P}$，在可观察变量科学范围内采样训练数据 $D$（ID区域）；保留ID/OOD测试集用于数值验证。
- **机制突变**：原始机制 $\mathcal{M}^{(0)}$ 经可解释操作 $\Delta_s$ 变为 $\mathcal{M}^{(s)} = \Delta_s(\mathcal{M}^{(0)})$，多个兼容突变可复合：$\mathcal{M}^{(S)} = (\Delta_{s_q} \circ \cdots \circ \Delta_{s_1})(\mathcal{M}^{(0)})$。
- **机制探针**：选择一组科学量 $\mathcal{Z} = \{z_k\}$，满足 $\mathcal{M} \Rightarrow z_k = g_{\mathcal{M}}(\mathbf{x})$ 且 $\mathcal{P}, C \nRightarrow z_k = g_{\mathcal{M}}(\mathbf{x})$；代理提交机制 $\hat{\mathcal{M}}$ 后，独立查询每个探针，比较 $\hat{g}^{(k)}$ 与 $g^{(k)}$ 的符号与数值等价性。
- **不可分辨性筛查**：对每个变异实例，搜索其他关系修改 $\tilde{\delta}(m_i)$ 是否能构成竞争性机制 $\tilde{\mathcal{M}}^{(s,i)}$ 且产生相同 $\mathcal{P}$，若存在则修改或剔除该实例。
- **评估协议**：标准发现阶段代理仅获上下文 $C$ 与训练数据 $D$，900秒时间预算；搜索结束后锁定提交，依次独立回答各探针；指标包括符号准确率 SA(P)、SA(M) 及数值准确率 $\mathrm{Acc}_{0.1}^{\mathrm{ID/OOD}}$。

## 实验与结果
- **数据集**：Full‑set（512实例，16家族×32变体）、Core‑set（80实例，8家族×10变体，按突变数分层）。
- **基线**：Codex/Claude Code/DeepSeek Harness + GPT‑5.6‑sol、GLM‑5.3‑flash、DeepSeek‑v4‑flash‑0731、DeepSeek‑v4‑pro‑0813；PySR（非LLM符号回归）；PySR+Direct‑Ask；Gold‑P Direct‑Ask（给定正确现象定律）。
- **主要结果**：
  - Core‑set上，**Codex + GPT‑5.6‑sol** 现象定律准确率 SA(P)=**35.00%**，机制准确率 SA(M)=**13.75%**；机制条件失败率 $1-\mathrm{Pr}(M|P)$ = **64.29%**（即现象定律正确时超六成机制仍失败）。
  - Full‑set上对应值为 SA(P)=16.02%，SA(M)=7.81%，条件失败率=57.32%。
  - 随突变数增加，机制准确率下降快于现象定律；GPT‑5.6‑sol在Original实例SA(M)=50.00%，4个突变时降至0.00%。
  - **Gold‑P控制**：给出正确现象定律后，GPT‑5.6‑sol机制准确率提升至Core‑set 45.00%、Full‑set 49.02%，但仍低于50%；Flash模型提升有限（GLM‑5.3‑flash: 6.25%/3.91%）。
  - PySR无法恢复任何机制（SA(M)=0），PySR+Direct‑Ask仅获极低机制分。
- **结论**：当前科学代理存在显著的**现象‑机制推理泛化鸿沟**，机制发现是独立于现象定律恢复的更难题。

## 相关工作脉络
1. **LLM‑SR/符号回归基准**（AI Feynman、LLM‑SRBench、SRBench）：聚焦现象定律恢复，通过变换/合成方程降低记忆；本文定位：进一步要求重建生成机制的内部结构。
2. **科学代理基准**（NewtonBench、Science‑Gym、SciLaws‑Bench）：引入交互、模拟环境、真实观测；本文定位：从“恢复可观察定律”推进到“恢复机制”，并引入探针验证内部一致性。
3. **机制哲学与模型论**（Kaplan & Craver等）：区分现象模型与机制模型；本文定位：将哲学概念操作化为可计算、可评估的基准任务。
4. **竞争机制与不可分辨性**（Hoefer & Rosenberg）：理论探讨经验等价性；本文定位：在基准构建阶段主动筛查并剔除不可分辨实例，使探针评估具操作性。
5. **突变/变异生成方法**：常见于程序合成与对抗测试；本文定位：采用科学可解释的局部修改，确保变异仍属目标领域合理机制，平衡熟悉度与陌生度。

## 局限性与未来方向
- 当前机制均需能通过符号消去显式导出现象定律，**大量需数值求解/微分方程的机制无法直接纳入**。
- 机制不可分辨性筛查依赖人工/半自动搜索，可能遗漏复杂竞争性机制。
- 评估指标严格（所有探针符号正确才算任务成功），可能低估近似机制恢复的价值。
- 未来可扩展至**机制→直接生成观测数据**（跳过现象定律），并发展数值近似下的探针验证与不可分辨性判别方法。

## 研究启发与可借鉴点
1. **机制探针设计思想**可迁移至其他需评估“深层理解”的领域（如因果发现、物理建模），通过查询内部量分离记忆与推理。
2. **可控科学突变框架**（兼容性检查、组合规则）可用于构建其他学科的泛化基准，平衡领域先验与陌生性。
3. **现象‑机制分离评估协议**可作为更广泛AI科学代理评测的标准模板，避免单一符号准确率带来的乐观偏差。
4. 与团队方向结合：可将机制发现能力纳入现有符号回归pipeline，或开发“现象‑机制联合优化”目标，提升代理的因果解释性。
5. 实验中的**Gold‑P控制**设计清晰隔离了“定律已知”与“机制推理”的贡献，该方法论可直接用于其他基准的消融分析。

## 关键术语表
- **Phenomenal law（现象定律）**：描述可观察变量间关系的数学方程，不含内部生成结构。
- **Mechanistic model（机制模型）**：由内部科学关系构成的集合，能推导出现象定律并定义不可观测的内部量。
- **Mechanism probe（机制探针）**：科学定义的内部量，其值仅由机制隐含，用于检验机制是否被真正重建。
- **Mechanistic indistinguishability（机制不可分辨性）**：不同机制蕴含同一现象定律，导致观测数据无法区分。
- **Mutation（机制突变）**：对经典机制施加的科学可解释修改，用于生成陌生变体。
- **SA(P)/SA(M)**：现象定律/机制的符号恢复准确率，SA(M)要求所有探针均正确。
- **Gold‑P Direct‑Ask**：提供正确现象定律与常量，仅要求模型推导探针关系的对照实验。
- **Acc₀.₁**：任务级数值准确率，要求持有集最大相对误差≤10%。

## 可复现要素
- 数据集：MechBench基准（Full‑set 512实例，Core‑set 80实例），代码与数据公开于 https://github.com/tsinghua-fib-lab/MechBench。
- 关键超参：搜索时间预算900秒/任务；PySR迭代1,000,000、表达式最大尺寸30、31个种群、64位精度；温度/采样分布按任务科学意义设定；ID/OOD划分边界 $b_j$ 依变量特征尺度确定。
- 模型配置：GPT‑5.6‑sol、GLM‑5.3‑flash、DeepSeek‑v4‑flash‑0731、DeepSeek‑v4‑pro‑0813，Agent harness包括Codex、Claude Code、DeepSeek Harness。
- 评估脚本：参考反馈脚本仅提供训练集数值拟合指标，不暴露机制/探针/测试集。
