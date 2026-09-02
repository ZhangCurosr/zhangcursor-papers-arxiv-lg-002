---
title: "Text-Capability-Loss-in-Vision-Language-Adaptation-An-Attent"
source: https://arxiv.org/pdf/2609.00746v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:03:50"
field: "视觉-语言模型适配与能力退化诊断"
keywords: ["vision-language model", "text capability loss", "attention sink", "Sink Strength", "QK-RMSNORM", "format-sensitive evaluation", "model adaptation", "weight merging"]
innovations: ["提出 Sink Strength 指标在 VL 训练前预测基座 LLM 的格式敏感能力退化风险", "建立 sink-anchored 扰动理论并区分 per-head 与 layerwise QK-RMSNORM 的保护差异", "提供模态对照、后预训练注入与后验权重合并等多组阴性控制以收窄修复路径"]
benchmarks: ["IFEval", "EQ-Bench v2.1", "GSM8K-CoT", "GPQA-Diamond-CoT", "MMLU", "MMLU-Pro", "MBPP", "RULER", "SEEDBench-IMG"]
---

# 论文速读：Text-Capability-Loss-in-Vision-Language-Adaptation-An-Attent

## 一句话总结
论文揭示并系统诊断了将预训练语言模型（LLM）微调为视觉-语言模型（VLM）时，主干网络**格式敏感型文本能力**发生显著退化的现象；作者将这一损失归因于**注意力 Sink（early attention-sink）被 VLM 微调扰动甚至坍塌**，并据此提出了一个可在推理阶段快速计算的单一标量指标 **Sink Strength (S)**，用于在 VL 训练前预测各候选 LLM 主干的退化风险排名。

## 研究问题与动机
- **核心问题**：标准 VL 微调流程（联合更新 LLM decoder 权重）会导致主干的文本能力受损，且损害高度集中在要求遵循严格表面格式规则的任务上（指令跟随 IFEval、严格解析 final answer 的 CoT 推理等），跨多个已发布 VLM–LLM 对的缺口可达两位数百分点。
- **现有方法不足**：已有缓解手段（重混合纯文本数据、冻结 LLM、LoRA、后验神经元重融合/权重合并）均只能部分弥补缺口，且**缺乏对“为何会损失”的机制性解释**，也无法在训练前识别高风险主干。
- **动机与切入点**：若能找到一个仅在基座 LLM 上可计算、提前于 VL 训练的诊断信号，即可为骨干筛选和后续针对性保护提供依据；作者将焦点对准被认为在 LLM 中稳定存在并吸收扩散注意力质量的 **early attention sink**，假设其被 VL 微调扰动是格式敏感能力下降的关键机制。

## 核心贡献（创新点）
- **提出 Sink Strength (S) 诊断指标**：基于基座 LLM 的若干次推理前向，量化每 head 的 sink 集中度；与已有工作相比，本文不是停留在现象观测，而是给出可在 VL 训练前运行的单标量预测器，并在多个 VLM–LLM 对和多个格式敏感任务上保持稳定排序（Spearman ρ≥0.88）。
- **建立 sink-anchored 扰动理论框架**：通过 Theorem 1 与 Lemma 1 给出 logit-gap 扰动上界及 sink 留存与初始 margin 的关系，从理论上明确**per-head QK-RMSNORM 如何通过消除输入原始量纲依赖来保护 sink**；与已有仅描述 sink 现象的工作相比，本文把 sink 稳定性与 VL 微调后的能力退化做了定量因果化连接。
- **提供多组对照与阴性控制**：包括模态对照（同一基础模型下 VL vs. 纯文本继续训练轨迹）、后预训练 QK-RMSNORM 注入对照、以及多种后验权重合并对照；与已有工作仅报告“某方法有用”不同，本文系统排除了若干简单修复路径，收窄干预空间至训练时 head-selective 保护。
- **跨架构与规模的外推验证**：在 17 对涵盖 dense/MoE、多家族（Qwen/Olmo/Mistral/Llama）与 1.5B–32B 参数的扩展面板上验证 S 的泛化（ρ=0.88），表明该诊断不只依赖于单一配方或单一架构。
- **发现 layerwise QK-RMSNORM 的不足并加以机制解释**：通过 Molmo2-O / MolmoE-1B 等 layerwise 变体证明“只要存在 QK-RMSNORM 就安全”的粗粒度判断是错误的，必须按 head 级别归一化才能提供保护；这与既有工作常将 RMSNorm/QK-Norm 作为整体模块讨论的粒度不同。

## 方法详解
- **Sink Strength 定义**：对基座 LLM 在最近约 10 层（具体为 late layers）的每个 head、每个 query 位置上，取其在 per-head argmax key 位置（即 sink 位置 $p_{\mathrm{sink}}$）上的注意力概率 $a_{p_{\mathrm{sink}}}^{(\ell,h,q)}$，计算 log-odds 后取中位数：
  $$S := \mathrm{median}_{(\ell,h,q)} \log \frac{a_{p_{\mathrm{sink}}}^{(\ell,h,q)}}{1 - a_{p_{\mathrm{sink}}}^{(\ell,h,q)}}$$
  计算仅依赖 15 条与测试集不重叠的校准提示词上的推理前向（7B 模型在单张 A6000 上约 8 秒）。
- **任务定义**：聚焦四类 format-sensitive 任务——IFEval（严格格式合规）、EQ-Bench（score-block 解析）、GSM8K-CoT 与 GPQA-Diamond-CoT（CoT final-answer 提取）。这些任务的评分门是表面可解析规则，因此细微的注意力模式漂移即可导致翻转。
- **理论扰动界（Theorem 1）**：
  - 在假设微调仅扰动 $W_q, W_k$（算子范数 $\|\Delta W_*\|_{\mathrm{op}} \le \varepsilon$），并固定归一化尺度与隐藏状态的前提下，无 QK-RMSNORM 时 gap 扰动上界显式含有输入量纲项 $\|\xi_q\|$ 与 sink/key 端输入范数之和；**引入 per-head QK-RMSNORM 后，该输入量纲项被 RMS 分母吸收**，上界仅依赖冻结的归一化尺度 $\|\gamma_q\|_\infty, \|\gamma_k\|_\infty$ 与投影 RMS 下界 $m_q, m_k$。
  - **Layerwise QK-RMSNORM** 并非等价保护：共享分母使各 head 的尺度仍与其在层能量中的份额 $\rho_h$ 挂钩，弱 head 会被压缩，导致 per-head margin 在 base 侧即偏低。
- **Sink 留存条件（Lemma 1）**：扰动后非 sink 总质量满足 $1 - a'_{p_{\mathrm{sink}}} \le \sigma(B_{\mathrm{gap}} - G_{\mathrm{base}})$，即初始 margin $G_{\mathrm{base}} - B_{\mathrm{gap}}$ 为正且足够大时 sink 才能存活；$S$ 正是 $G_{\mathrm{base}}$ 的中位数代理。
- **控制实验设计**：
  - **模态对照**：同底座 Qwen2.5-7B-Instruct 下，对比 VL endpoint 与 text-only endpoint（Qwen2.5-7B-Instruct-1M），证明 VL 训练对 $W_q/W_k$ 的扰动明显大于纯文本继续训练。
  - **训练轨迹对照**：在 3B 底座上分别跑 VL 与 Tülu 纯文本 SFT 的匹配轨迹，观察 S 与 IFEval 随 step 的变化；VL 训练使 S 下降、sink 坍塌，而纯文本保持 S。
  - **后预训练注入 QK-RMSNORM**：在 Qwen2.5-3B-Instruct 上以单位尺度初始化注入 per-head QK-RMSNORM 再做 VL，无法复现原生保护的 IFEval 表现。
  - **后验权重合并**：在 Qwen2.5-VL 与 Qwen3-VL 上测试线性/谱/稀疏合并（Task Arithmetic、TSV、STAR、TIES、DARE 等），未能恢复 lost capability。

## 实验与结果
- **数据集/基座**：五个 headline VLM–LLM 对（Qwen3-VL / InternVL3.5 基于 Qwen3-8B，Qwen2.5-VL / InternVL3 / LLaVA-OV 基于 Qwen2.5-7B-Instruct 或 Qwen2-7B-Instruct）及一个 layerwise 对照 Molmo2-O（Olmo-3-7B）；扩展面板共 17 对，覆盖 MoE/不同家族与 1.5B–32B。
- **评估基准**：主指标 IFEval prompt-strict，辅以 EQ-Bench v2.1、GSM8K-CoT、GPQA-Diamond-CoT；另报告 MMLU、BoolQ、MBPP、MMLU-Pro、RULER 等以区分 format-sensitive 与知识/编码类能力。
- **主要结果**：
  - **Sink Strength 排序高度吻合退化幅度**：headline 六对中，Spearman ρ=0.97（IFEval）；跨所有四任务 ρ∈[0.88, 0.97]；留一对外推平均误差 2.54 pt（范围 −1.8 到 −18.7 pt），而常数均值基线误差 4.40 pt。
  - **架构相关**：per-head QK-RMSNORM 对（Qwen3-VL、InternVL3.5）IFEval 下降 ≤5.4 pt；无 QK-RMSNORM 对下降 7.9–9.6 pt；layerwise 对照 Molmo2-O 下降达 −18.7 pt。
  - **非格式敏感能力相对稳健**：MMLU/BoolQ 下降不超过 2.3 pt，编码（MBPP）甚至提升；但 GPQA-Diamond-CoT 作为 MCQA 形式仍显著下降，排除单纯格式混淆。
  - **阴性控制结论**：后预训练 QK-RMSNORM 注入无法复现原生保护；后验权重合并未能恢复已丢失的格式敏感能力，提示应在训练时进行 head-selective 保护。
- **最强结果与提升**：以 S 作为预筛指标可将高风险主干（如 Molmo2-O、无 QK 的 Qwen2.5-VL/InternVL3/LLaVA-OV）与低风险主干（per-head QK 的 Qwen3/InternVL3.5）清晰分离，为后续选用/调整主干提供量化依据。

## 相关工作脉络
- **VL 微调与文本侧侵蚀**：如 Zhang et al. (2024)、Zhai et al. (2024)、Lee et al. (2025a)、Ratzlaff et al. (2025)、Srivastava et al. (2026) 等观察到 decoder 在 VL 适配后文本能力下降；本文与前人结论一致，但进一步将其聚焦到 format-sensitive 子集并以 sink 机制统一解释。
- **数据/训练侧缓解**：Lin et al. (2024)、Tong et al. (2024)、Tu et al. (2025) 的文本重混合；Zhu et al. (2024) 冻结 LLM；Hu et al. (2022)、Liu et al. (2024) LoRA；Yu & Ananiadou (2025) 神经元重融合。本文认为这些方法各自闭合部分缺口，但未解释“损失为何发生”，并提出在训练前通过 S 筛选主干作为前置补充。
- **Attention sink 机制**：Xiao et al. (2024)、Sun et al. (2024)、Barbero et al. (2025)、Zhang et al. (2025)、Gu et al. (2025) 刻画 sink 的出现条件与表征性质；本文视角从“静态 inference-time sink 特征”转向“fine-tuning 过程中的 sink 扰动与坍塌”，并把 sink 稳定性与下游 format-sensitive 评测直接联系起来。
- **QK-RMSNORM 与稳定性**：Henry et al. (2020)、Zhai et al. (2023) 讨论 query-key 归一化与熵坍缩稳定；本文进一步区分 per-head 与 layerwise 两种粒度的实质差异，并给出理论界与经验证据。
- **权重合并与回退修复**：Ilharco et al. (2023)、Wortsman et al. (2022)、Yadav et al. (2023)、Yu et al. (2024)、Gargiulo et al. (2025)、Lee et al. (2025b) 等工作探索模型合并/编辑；本文在后验合并实验中表明这类策略在本题场景下无法有效恢复格式敏感能力，从而收窄后续研究空间。
- **位点/回路稳定性视角**：Wang et al. (2025b) 等从回路分析角度研究微调机制；本文与之精神相近，但以 small-circuit 风格的 logit-gap 边界与 margin 论证为工具，给出更可直接计算的诊断量 S。

## 局限性与未来方向
- **样本与归因边界**：per-head QK-RMSNORM 在本样本中与 vendor、配方与预训练数据共变；尽管存在同 vendor 内对比（Qwen2.5→Qwen3、InternVL3→InternVL3.5），但并未完全隔离架构轴，因果性仍需更多受控实验支撑。
- **泛化范围未全验证**：S 目前已针对 VL 适配输出校准，其对其他适配范式（如 continual pretraining）下的指令跟随漂移是否同样可预测，仍是开放问题。
- **后验修复阴性结果约束解空间**：文中后预训练 QK-RMSNORM 注入与多种后验权重合并均未奏效，说明简单“事后修补”路径受阻；未来需探索训练时 head-selective 保护（例如冻结 sink 层以下层、对 sink head 的 $W_q/W_k$ 更新做正交投影去 sink 方向）。
- **多层组合理论待补全**：当前 Theorem 1 与 Lemma 1 为单层陈述；跨多层残差流、value 向量、MLP 与 post-attention LayerNorm 的严格组合界尚未给出。
- **极端 case 外推局限**：如 Molmo2-O 在部分任务（EQ-Bench −64.6）呈现任务特异性坍塌，提示线性外推在边界处可能失效，S 更适合作为 rank predictor 而非绝对幅度预言器。

## 研究启发与可借鉴点
- **预筛指标的可迁移思路**：将“训练前可计算的结构性信号”用于预测适配后的能力分布变化，这一范式可从 sink 推广到其它稳定性相关结构（如特定层的 activation margin、注意力集中度分布、关键方向敏感度），适用于多模态/多任务适配的骨干挑选流程。
- **归一化粒度的重要性**：文章清晰区分 per-head 与 layerwise 归一化在保护 sink 上的本质差异，提示在设计或迁移 RMSNorm/QK-Norm 类结构时，应以 head 级保护为目标，避免层级共享分母对弱 head 的隐性压缩。
- **阴性控制的设计可复用**：模态对照（同底座、同优化预算、不同数据模态）、后注入对照（加入模块但不复现预训练协同适应）、后验合并对照（失败即排除一类修复路径）构成了一套完整的“排除法”流程，可作为后续机制研究的实验模板。
- **跨架构/规模外推评估规范**：在 dense/MoE、多家族、1.5B–32B 范围测试诊断量的鲁棒性，并显式报告共享底座带来的 S 相同情况（分辨率下限），这一做法可为后续诊断性指标的发布提供可参照的评估规范。
- **与本文团队方向的结合机会**：若团队关注多模态适配中的能力失衡、主干复用、或指令/格式敏感评测的退化问题，可把 S 作为主干预筛器集成进 pipeline；同时可将 sink 扰动视角拓展到其它依赖精确 surface-token tracking 的能力（公式排版、结构化 JSON/XML 输出、多轮严格槽位填充）。

## 关键术语表
- **Sink Strength (S)**：在基座 LLM 上计算的对数几率中位数，量化 per-head attention sink 的集中度；S 越高表示 sink 越强，预测 VL 适配后格式敏感能力保留越好。
- **Attention sink**：现代 LLM 在序列早期若干位置集中大量注意力质量的现象，起到吸收扩散注意力、稳定 surface-token 追踪的作用。
- **Format-sensitive tasks**：依赖严格可解析表面规则评分的任务，如 IFEval 格式合规、CoT final-answer 提取、EQ-Bench score-block 解析；对注意力模式漂移尤为敏感。
- **QK-RMSNORM**：对 query/key 投影结果做 RMS 归一化后再乘学习到的尺度；per-head 版本独立归一化每头，layerwise 版本在所有头 Concat 后共享归一化分母。
- **$G_{\mathrm{base}}$ 与 $B_{\mathrm{gap}}$**：前者为 unperturbed 下 sink 位置的 aggregate logit-gap，后者为微调后各位置 logit 扰动在 sink-anchored 意义下的上界；margin $G_{\mathrm{base}} - B_{\mathrm{gap}}$ 决定 sink 是否能存活。
- **留一对外推（leave-one-pair-out）**：以已知 VLM–LLM 对的 S 与实测缺口拟合 1-D 线性回归，再轮流剔除一对评估预测误差，用于检验诊断量的预测精度与鲁棒性。
- **Modality attribution**：通过对比同一底座下 VL 与纯文本两种训练终点的扰动范数与 Sink 变化，分离“模态差异”与“继续优化”对文本能力损失的不同贡献。
- **Weight merging**：在 VL 训练后将 VLM backbone 与参考 LLM 权重按线性/谱/稀疏策略合并，以尝试恢复丢失能力；本文在多种设定下均未观察到有效恢复。

## 可复现要素
- **数据集**：IFEval、EQ-Bench v2.1、GSM8K、GPQA-Diamond、MMLU、MMLU-Pro、BoolQ、MBPP、RULER、SEEDBench-IMG、LLaVA-Pretrain、LLaVA-Mix、Tülu general instruction SFT；均为公开数据集或通过 Hugging Face 获取。
- **代码/权重**：论文在 Appendix B.4 列出所有模型与数据集的 Hugging Face URL；评估使用 lm-evaluation-harness v0.4.12、VLMEvalKit 等公开工具；训练实现基于 PyTorch/transformers/DeepSpeed。代码仓库链接论文正文未明确给出，**需以官方发布为准（论文未明确声明单独代码仓库）**。
- **关键超参**：
  - Sink Strength 计算：15 条校准提示词（与 IFEval 测试集不重叠），取最近约 10 层的 head 粒度中位数；N=5 亦可复现同等排序（约 3 秒/7B），N=15 为默认以降低种子方差。
  - 训练轨迹对照（3B 匹配实验）：Stage1 仅训 projector，Stage2 训 full LM + projector；effective batch=128；optimizer AdamW；Stage1 lr=1e-4，Stage2 lr=2e-5；cosine/const+warmup 调度；warmup ratio=0.03；bf16；DeepSpeed ZeRO-2；约 47h/variant。
  - QK-RMSNORM 注入：单位尺度初始化 $\gamma_q=\gamma_k=1$，其余与 vanilla 完全匹配。
  - 权重合并设置：95/5、uniform、TA(λ=1)、TSV(α=1)、STAR(η=40%)、TIES(密度 0.2, λ=1)、DARE(掩码率 0.9, λ=1)。
