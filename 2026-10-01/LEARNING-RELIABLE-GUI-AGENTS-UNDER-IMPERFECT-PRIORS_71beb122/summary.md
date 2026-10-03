---
title: "LEARNING-RELIABLE-GUI-AGENTS-UNDER-IMPERFECT-PRIORS"
source: https://arxiv.org/pdf/2609.39547v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 21:58:01"
field: "GUI智能体与检索增强自动化"
keywords: ["GUI agents", "retrieval-augmented generation", "imperfect priors", "drift-aware training", "self-exploration", "reliability-gated policy", "VLM", "mobile automation"]
innovations: ["将GUI先验可靠性显式建模为FOLLOW/PARTIAL/IGNORE门控决策并作为监督信号", "基于真实漂移分类的扰动分布D_Pi使训练覆盖不可靠先验光谱", "语义区域预算+结构-视觉去重的无标注UI图构建流水线"]
benchmarks: ["MobileWorld", "AndroidLab", "AndroidWorld", "CMGUI", "ChiM-Nav"]
---

# 论文速读：LEARNING-RELIABLE-GUI-AGENTS-UNDER-IMPERFECT-PRIORS

## 一句话总结
论文提出 SAGE 框架，通过无标注自探索构建任务对齐的 UI 转移图并生成可检索先验，结合基于 GUI 漂移模式的扰动训练可靠性门控策略，使智能体学会在**不可靠先验**下判断"跟随/部分使用/忽略"，从而在真实动态界面上实现更鲁棒的 GUI 自动化。

## 研究问题与动机
- **核心问题**：GUI 智能体（LLM/VLM 驱动）在面对未见应用与长程多步任务时表现脆弱，因为真实任务依赖应用特定、时效性强的操作性知识（隐藏入口、前置条件、广告按钮识别等），这类知识在预训练语料中稀缺。
- **瓶颈1：大规模知识获取难**。高质量轨迹多来自人工标注或用户演示，成本高且难以覆盖长尾应用；自主探索存在预算有限、语义标注推断性强、登录态/服务器路由/个性化内容难以触及等结构限制。
- **瓶颈2：先验在 GUI 漂移下不可靠**。论文在 20 个主流 App 的 176 个配对新旧版本任务上统计发现：仅 44% 的任务保持相同完成轨迹，其余落入 5 类漂移模式（步骤增删、入口/路径偏移、控件重命名、目标不匹配/干扰绕路）。即使提供旧版正确轨迹，现有微调模型在新版上性能仍明显下降。
- **核心主张**：GUI 智能体不应追求完美知识，而应学习在**不可靠先验**下做出正确行为。

## 核心贡献（创新点）
- **无标注自探索流水线**：在真实设备上以语义区域预算+恢复感知回溯遍历交互元素，构建去重 UI 转移图，并通过 VLM 合成 (task, trajectory) 对，无需人工标注即可大规模生成任务对齐的可检索先验。与已有探索工作本质区别在于：将轨迹组织为去重图结构而非碎片化原始日志，并以任务语义对齐优先采样，而非穷举控制。
- **漂移感知扰动 + 可靠性门控策略**：基于 5 类真实 GUI 漂移模式设计扰动算子族 {T_k}，将清洁先验映射为含噪声的先验分布 D_Π 并附带确定性可靠性标签 r∈{FOLLOW,PARTIAL,IGNORE}；策略因式分解为可靠性评估器×可靠性条件执行器，迫使智能体在行动前显式承诺可靠性决策。与已有 RAG/GUI 记忆方法本质区别在于：不把检索内容假定为可用，而是将检索噪声转为显式监督信号，学习过滤/部分使用/忽略三种行为。
- **构建并提供系统扰动的 (task, trajectory) 基准**：包含 6 类先验条件（Correct/Local/Noise/Omit/Rewrite/Other）的精度增益评测协议，以及跨数据集泛化实验，验证可靠性评估能力可迁移。
- **开源计划**：接收后将公开核心实现与公共数据部分的基准（CMGUI、ChiM-Nav 的 Direct 与 Full-CoT 拆分），不公开自探索原始截图与账号状态以保护隐私。

## 方法详解
**问题建模**：将 GUI 任务建模为上下文决策过程 (S, A, G, T, ρ)，t 步观测 s_t，给定目标 g 与历史 h_{<t} 及检索到的路径先验 π=(ē_1,...,ē_L) 输出动作 a_t。引入潜在可靠性变量 r∈{FOLLOW,PARTIAL,IGNORE}，策略因式分解：
p_θ(a_t|s_t,h_{<t},g,π)=Σ_r p_θ(r|s_t,h_{<t},g,π)·p_θ(a_t|s_t,h_{<t},g,π,r)
推理时通过自回归解码联合实现 (r,a_t)，无需显式边缘化。

**自探索先验获取（Sec 3.2）**：
- **语义区域动作抽象**：每屏划分为 navigation/function/homogeneous content/distraction 四类区域，按预算 B_nav≥B_func≫B_homog≥B_dist=0 采样；前沿节点维护候选队列并按区域优先级、本地探索状态、控件级去重（归一化边界+语义签名）排序，避免重复交互语义等价控件。
- **结构-视觉状态去重**：两屏 s,s' 当 XML 路径 Jaccard≥τ_xml 且 VLM 视觉等价检查 φ_vis(s,s')=1 时合并，抑制动态内容碎片化同时防止功能不同但视觉相似的误并。
- **恢复感知回溯**：每次交互后返回父节点；若回溯落在非预期屏，将观测转换记录为新边而非丢弃；从根节点尝试图基回放，仅在无路径时重启应用，确保图中所有边可执行且验证过。
- **图到任务对齐先验**：枚举图 G 中路径 π*，用多模态标注器合成与功能端点一致的 goals g；每条边以"前状态-触发控件-后状态"自然语言描述为 ē，拼接得轨迹级先验 π*=[ē_1;...;ē_L]。

**漂移感知扰动（Sec 3.3）**：6 类算子 T_k（k=0..5），每类对应一种真实漂移类比并附确定性可靠性标签 r*（见表1）：
- T0 identity（clean，FOLLOW）
- T1 surface rephrasing（控件重命名，FOLLOW）
- T2 sub-path deletion（步骤删除，PARTIAL）
- T3 suffix branch replacement（入口/路径偏移，PARTIAL）
- T4 irrelevant-op insertion（广告/权限/遮挡，PARTIAL）
- T5 other-goal path substitution（目标不匹配，IGNORE）
训练先验分布为混合 D_Π(π|π*)=Σ_k w_k δ(π-T_k(π*))。T3/T5 从同图 G 采样其他路径，保证噪声先验分布内、探索质量直接决定负样本硬度。

**学习目标与训练（Sec 3.4）**：
- 步级分解：每条 (π*,g) 生成轨迹 τ=(s_0,a_1,...,a_L,s_L)，分解为 L+1 个步样本（含 STOP）。每步采样 k~Cat(w)、应用 T_k，得输入 (s_t,h_{<t},g,π) 与目标 (r*,a*_t)。
- 结构化响应与三一致检查：模型输出 ⟨Thought,Action,Tool⟩。Thought 围绕三检查：goal consistency（π 与 g 是否同意图）、history consistency（h_{<t} 是否与 π 前缀兼容）、UI grounding（s_t 是否支持下一步建议操作）。三者全通过→FOLLOW；部分通过→PARTIAL 并标记可信边索引；goal 不一致→IGNORE。
- 教师注释：Thought 由强多模态教师条件于 (s_t,h_{<t},g,π,r*,a*_t) 生成（teacher-forced rationalization）；Action 与 Tool 由 π* 确定性重建，防止教师幻觉污染监督。
- 损失函数（段加权 teacher-forced NLL）：
L(θ)=E[Σ_i λ_{c(i)}(-log p_θ(y*_i|y*_{<i},s_t,h_{<t},g,π))]，其中 c(i)∈{thought,action,tool}；编码 r 与可信边索引的 token 上采样，使 Eq.2 具象化。同决策状态下配对不同可靠性先验，迫使模型不能简单复制 π，必须学会何时跟随/抽取/覆盖。
- 实现：基于 Qwen3-VL / GUI-OWL 等 backbone 用 LoRA（rank 16, alpha 32, dropout 0.05, lr 5e-5, bf16, cosine warmup 0.05, seq_len 8192）训练 3 轮；可靠性评估与动作生成共享单一自回归解码器，无需辅助头。

## 实验与结果
**数据集与基线**：三个轨迹源 CUTG（自探索）、CMGUI、ChiM-Nav；基线包括 SFT（仅动作监督、推理时接收先验）、Part-CoT（去掉显式可靠性标签的 CoT）、Ori.（原始 backbone）。探索效率对比 LLM-Explorer、DroidAgent、DroidBot、Humanoid。
**评估基准**：MobileWorld(129 任务)、AndroidLab(85)、AndroidWorld(75)；在线评测 144 个探索覆盖任务。
**主要数字**：
- 探索效率：600 步下 SAGE 探索到约 148 个去重屏幕、28 个活动；LLM-Explorer 需约 1500 步才达 101 屏幕、10 活动；其余基线止步于 55 屏幕以下。子任务覆盖率：MobileWorld 71.2 vs 53.1，AndroidLab 77.0 vs 68.1，AndroidWorld 77.5 vs 53.9，总体 +16.9 pp（74.1 vs 57.2）。
- 离线精确度增益（Table 2，相对 no-prior）：Qwen3-VL-2B 上 SFT 平均 +2.93 pp、Ours +11.70 pp（↑8.77）；Qwen3-VL-8B 上 SFT 平均 +6.12 pp、Ours +15.51 pp（↑10.61）。各 backbone 上 Correct 先验增益均提升（如 2B：6.53→21.63 pp；8B：13.06→20.41 pp）。
- 在线任务成功率增益（Table 3，相对各自 no-prior）：Qwen3-VL-4B Ours 在 AndroidWorld +7.70 pp↑13.90、AndroidLab +0.31 pp↑10.63、MobileWorld -2.60 pp（优于 SFT -6.80）；GUI-OWL-1.5-4B 同样全面优于 SFT。
- 跨数据集泛化（Table 4）：30 个 backbone-transfer 设置中 Ours 平均增益从 3.19→6.81 pp，28/30 超过 SFT。
- 消融（Table 5）：Ours 平均增益 7.87 pp vs Part-CoT 5.59 pp；Reliability supervision 贡献明确。
- 真实版本漂移可靠性评估（Table 9）：FOLLOW Macro-F1 平均 91.21%，PARTIAL 58.82%，IGNORE 63.57%，整体 Macro-F1 70.48%。
- 统计显著性（Appendix G）：12 个对比全部为正，10/12 置信区间下界>0，差异均值 6.60~9.97 pp。

## 相关工作脉络
- **GUI 智能体/在线基准**（Hong 2024 CogAgent、Cheng 2024 SeeClick、Rawles 2024 AndroidWorld、Xu 2025 AndroidLab 等）：聚焦"能否完成任务"，本文聚焦"应用级操作性知识如何大规模获取并在不可靠时稳健使用"。
- **App 自主探索**（Li 2017 DroidBot、Li 2019 Humanoid、Yoon 2024 DroidAgent、Zhao 2025 LLM-Explorer）：原始轨迹冗余碎片化，本文通过状态去重+控制函数合并+转移对齐，将探索产出组织为可审计、可扰动、可学习的图先验。
- **检索增强 GUI 智能体**（Lee 2024 MobileGPT、Wen 2024 AutoDroid、Kong 2025 MapAgent、Zhu 2025 Moba、Zhou 2026 Mobile-Agent-RAG）：主要关注知识获取与检索，常假设检索内容可用；本文针对检索噪声与界面动态变化，显式建模 FOLLOW/PARTIAL/IGNORE。
- **Self-RAG/校正检索**（Asai 2023 Self-RAG、Yan 2024 Corrective RAG、Yoran 2023、Fang 2024、Wei 2024）：将检索噪声视为问题；本文将其转化为显式监督信号，通过漂移感知扰动构造训练分布 D_Π。
- **GUI 轨迹/动作基础模型**（Wu 2024 OS-Atlas、Zhang 2025 AppAgent、Zhang 2025 AgentCPM-GUI、Qin 2025 UI-TARS、Alibaba 2025 MobiZen-GUI）：本文在此基础上引入先验可靠性门控与自探索知识库构建。
- **GUI 漂移/迁移评测**（Lu 2025 TransBench、Xie 2026 SecAgent）：与本文 GUI 漂移分类和 cross-version 评估动机一致，本文进一步构造了可训练的扰动分布与显式可靠性决策机制。

## 局限性与未来方向
- 探索模块依赖截图与无障碍树，遇到残缺 metadata、重度自定义渲染或阻止自动化的 App 时，状态去重与控制解析可靠性下降。
- 真实 App 含登录态、区域内容、广告、A/B 测试、临时活动等动态因素；虽将自探索知识视为不完美先验，但严重界面漂移仍会削弱检索质量与动作落地。
- 训练在固定 train/test split 上进行，虽覆盖多 backbone/扰动类型/跨数据集迁移，但未覆盖所有 App 类别、语言与交互模式。
- 教师多模态模型用于语义先验与 Thought 标注，可能引入教师偏见或注释误差。
- 真实设备探索+教师注释带来工程与计算成本；规模化到大量 App 仍需稳定设备执行、状态管理、模型服务与反复质量检查。
- 未来方向可包括：更鲁棒的跨版本漂移自适应、在线持续探索更新先验图、对广告/权限等干扰的可解释屏蔽、以及更大规模的真实设备基准。

## 研究启发与可借鉴点
- **"不完美先验"范式的可迁移性**：将检索噪声显式建模为可靠性分类问题（FOLLOW/PARTIAL/IGNORE）并用于监督，可推广至文档检索、代码补全、知识密集型 agent 等场景，避免"检索即真"的脆弱假设。
- **语义区域预算探索策略**：按 navigation/function/homogeneous/distraction 四类区域分配采样预算，并结合结构-视觉双通道去重，是低成本构建应用知识图谱的有效模板，适用于移动端/桌面端 UI 自动化。
- **图重建式恢复感知回溯**：将"回溯失败"视作新边证据而非丢弃，保障图中边可执行且验证；可迁移至任何基于状态图的探索/测试流水线。
- **扰动分布 D_Π 的 hardness spectrum 设计**：通过同源图采样 T3/T5 保持噪声先验 in-distribution 且拓扑相邻，使训练覆盖从轻微到严重的可靠性质谱；对鲁棒 fine-tuning 具有通用参考价值。
- **与本团队方向结合机会**：若团队涉及 RAG/Agent 可靠性、跨版本/跨域迁移、UI 自动化评测，可复用本文的扰动taxonomy 与 reliability-gated policy 架构，并以本团队现有 VLM backbone 验证跨任务泛化。

## 关键术语表
- **SAGE (Self-explored, Assessment-Guided Execution)**：论文提出的框架，耦合无标注自探索与漂移感知可靠性训练，使 GUI 智能体在不可靠先验下稳健执行。
- **UI 状态转移图 G=(V,E)**：通过自探索在真实设备上构建的去重 UI 图，节点为语义等价屏幕，边为可执行控制转移，用作先验知识的结构化存储。
- **可靠性变量 r∈{FOLLOW,PARTIAL,IGNORE}**：显式中间决策变量，评估检索先验与当前任务/状态的匹配程度，分别对应完全遵循、部分信任+过滤、忽略。
- **漂移感知扰动算子 T_k**：基于 GUI 漂移分类设计的 6 类变换（identity/rephrase/delete/sub-path/insert/other-goal），将清洁先验映射为不同可靠性的训练样本。
- **语义区域预算**：将屏幕划分为 navigation/function/homogeneous content/distraction 四类并分配差异采样预算，避免在列表/广告上浪费探索步。
- **结构-视觉状态去重**：结合 XML 控制路径 Jaccard 与 VLM 视觉等价检查，合并动态内容导致的伪不同屏，同时避免功能不同但视觉相似的误并。
- **恢复感知回溯**：探索后尝试回退到父节点；若落在意外屏则记录为新边并从根节点尝试图基回放，仅在无路径时重启 App，保证边可执行。
- **Full-CoT / Part-CoT / Direct**：三种训练范式；Direct 仅输出动作，Part-CoT 加推理但不监督可靠性标签，Full-CoT 额外监督 Thought（含 r 与可信边索引）用于显式先验评估。

## 可复现要素
- **数据集**：CUTG（自探索，不公开原始截图/完整轨迹/账号状态）、CMGUI（公开）、ChiM-Nav（公开）；接收后将公开 CMGUI 与 ChiM-Nav 的 Direct/Full-CoT JSON 拆分、图像清单与预处理脚本。
- **代码**：接收后公开核心实现（数据转换、先验构建、扰动生成、提示格式化、训练与评测脚本）。
- **关键超参**：LoRA rank=16, alpha=32, dropout=0.05, 目标模块全适配；bf16、cosine lr、warmup=0.05、lr=5e-5、seq_len=8192、batch=1、gradient accumulation=4、训练 3 轮；扰动权重 w_k 为可调超参（论文未给出具体值，默认均匀混合隐含于 Cat(w)）。
- **硬件**：8×NVIDIA RTX 4090 24GB；2B/4B 模型单卡训练，8B 模型双卡 DeepSpeed。
- **随机性/统计**：Appendix G 报告 3 次额外划分×4 backbone 的 95% 置信区间（bootstrap B=5000），12/12 差异为正、10/12 下界>0。
