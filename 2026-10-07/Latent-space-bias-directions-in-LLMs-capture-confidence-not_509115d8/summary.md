---
title: "Latent-space-bias-directions-in-LLMs-capture-confidence-not"
source: https://arxiv.org/pdf/2610.08559v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:49:50"
field: "大语言模型公平性与可解释性"
keywords: ["activation steering", "LLM bias", "model confidence", "fairness", "interpretability", "linear representation"]
innovations: ["揭示去偏见方向实际编码置信度而非语义偏见", "构造置信度对照方向并与去偏见方向计算余弦相似度以量化混淆程度", "证明沿去偏见方向干预在通用知识数据集上同样提升熵/拒绝率，表明其效应非目标导向"]
benchmarks: ["BBQ", "StereoSet", "CrowS-Pairs", "MMLU", "OpenBookQA"]
---

# 论文速读：Latent-space-bias-directions-in-LLMs-capture-confidence-not

## 一句话总结
本文发现：通过对比反偏见与偏见提示的激活差构建的"去偏见方向"，在 LLM 隐空间中实际编码的是**模型置信度**而非社会偏见；沿此方向做激活干预之所以"降低偏见"，本质是让模型变得更不自信（倾向拒绝回答或均匀化概率分布），而非真正纠正了底层刻板印象。

## 研究问题与动机
- 激活干预（activation steering）作为轻量级推理时去偏见方法广受欢迎，但先前研究已报告其**泛化性差**（单一数据集提取的方向在新数据集上效果有限）且对通用知识任务有**意外负面效果**。
- 核心未解问题：当偏见/反偏见提示在中间层隐空间中呈线性可分时， separating direction 真正编码的是什么？
- 若该方向混入了置信度信号，则现有"去偏见"结果可能被误读，影响对 activation steering 方法的信任与推广。

## 核心贡献（创新点）
1. **揭示去偏见方向的主导信号是置信度而非语义偏见**：训练一个区分偏见/反偏见提示的逻辑回归分类器后发现，该方向在高 AUROC（如 Llama-3.1-8B-Instruct 达 0.94）区分最高/最低概率答案时表现优异，而区分偏见类型时仅 0.57，证明其实际捕获的是置信度。
2. **提出偏差方向与置信度方向的余弦相似度对比实验**：从 BBQ/CrowS-Pairs/StereoSet 训练去偏见方向（CAA），从 MMLU/OpenBookQA 训练置信度方向，发现两者余弦相似度在 0.3–0.91 之间高度对齐，尤其在 instruct 模型上更高（如 Llama-3.1-8B-Instruct BBQ→MMLU 达 0.78）。
3. **系统证明沿去偏见方向干预的真实行为效应**：存在拒绝选项时，干预显著提高拒绝率（而非纠正偏见选择）；无拒绝选项时，干预均匀化剩余选项概率、提升熵；且该效应在**不含任何社会偏见概念的通用知识数据集（MMLU、OpenBookQA）上同样成立**。

## 方法详解
- **去偏见方向（CAA）构建**（Eq. 4）：在每个模型的最优分层（通过 AUROC 确定），取训练集中每对偏见/反偏见答案最后一个 token 的隐藏状态差均值：
  \[
  \mathbf{v_b} = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{h}_i^{\text{anti-biased}} - \mathbf{h}_i^{\text{biased}})
  \]
- **置信度方向构建**：在非偏见数据集（MMLU、OpenBookQA）上，取模型最高概率与最低概率答案的隐藏状态差均值，构造 $\mathbf{v_c}$。
- **线性分类器**：在最终 token 的隐藏状态上训练 logistic regression，权重 $\mathbf{w}$ 和偏置 $b$ 用于预测 $P(\text{biased})=\sigma(\mathbf{w}\cdot\mathbf{h}+b)$，在各模型中搜索最优分类层（通常位于中间层，instruction-tuned 模型 AUROC 普遍高于 base）。
- **激活干预**：在最优分层将缩放后的单位方向加入隐藏状态 $\mathbf{h}'=\mathbf{h}+\alpha\hat{\mathbf{v_b}}$，扫描 $\alpha\in[-1,2]$，默认报告值 $\tilde{\alpha}$ 为 $\alpha_{\text{nat}}$（若 $>1$）或 $\alpha=1$（否则）。
- **评估指标**：AUROC（分类器判别力）、bias score（BBQ 用 $s_{\text{AMB}}$、$s_{\text{DIS}}$；StereoSet/CrowS-Pairs 用偏见回答百分比）、accuracy、answer entropy。

## 实验与结果
- **模型与数据集**：Llama-3.1-8B（Base/Instruct）、Falcon3-7B、Ministral-3-8B、Qwen3.5-9B（各 Base/Instruct）；偏见基准 BBQ、StereoSet、CrowS-Pairs；通用知识基准 MMLU（10K 样本）、OpenBookQA。
- **基线偏见程度**：所有模型均表现出显著偏见——BBQ 歧义上下文 bias score $s_{\text{amb}}\in[0.07,0.18]$，CrowS-Pairs/StereoSet 偏见回答率 62.7%–75.8%。
- **最强分类结果**：Llama-3.1-8B-Instruct 在按置信度（最高/最低概率答案）标签化时 AUROC=**0.94**；按偏见类型标签化时仅 AUROC=**0.57**。
- **方向对齐最强结果**：Qwen3.5-9B-Base，BBQ 训练的 CAA 方向与 MMLU 置信度方向余弦相似度达 **0.91**。
- **去偏见效果**：Llama-3.1-8B-Instruct BBQ $s_{\text{amb}}$ 从 0.08 降至 0.01，但 BBQ disambiguated accuracy 从 0.87 降至 0.52（大量拒绝回答）。
- **通用任务影响（最强负面）**：Qwen3.5-9B 在 StereoSet 方向上，OpenBookQA entropy 从 0.24 翻倍至 0.50，MMLU entropy 上升 80%（0.34→0.61）；Llama-3.1-8B-Instruct OpenBookQA accuracy 从 0.76 降至 0.70。

## 相关工作脉络
1. **Rimsky et al. (2024) CAA**（Steering Llama 2 via Contrastive Activation Addition）：本文使用其 CAA 框架构建去偏见方向，但揭示该方向实际编码置信度，与原文"代表偏见语义"的假设形成根本性修正。
2. **Li et al. (2023) Activation Engineering**：提出基于线性表征假设的 activation steering 方法，本文在此基础上质疑其在公平性场景中的可靠性。
3. **Li et al. (2025) FairSteer**：推理时动态激活去偏见方法，本文指出此类方法同样受置信度混淆影响，削弱了其泛化有效性。
4. **Siddique et al. (2026)**：发现 bias steering vector 与 refusal steering vector 负相关；本文平行发现 anti-bias 方向实为 low-confidence 方向，触发拒绝式回答。
5. **Kumaran et al. (2026)**：因果证据表明 LLM 用置信度驱动行为；本文在其基础上进一步证明 bias-vs-anti-bias 对比本身即编码了置信度差异。
6. **Pres et al. (2024)**：批判 steering 评估常依赖概率输出而掩盖真实行为变化；本文通过 hidden state 分析给出更底层的解释机制。

## 局限性与未来方向
- 仅测试了 4 个 7B–9B 量级的开源模型，未覆盖更大规模或不同架构（如 MoE）的模型。
- Qwen3.5-9B-Base 的最优分类层为最后一层，在此层施加干预效果几乎为零（无后续层传播），说明层选择策略对某些模型可能失效。
- 数据集限于 BBQ、StereoSet、CrowS-Pairs 三个主流偏见基准，未见其他偏见衡量方式（如 adversarial 数据集）的验证。
- 本文主张的方向分离（去偏见 vs. 置信度）仅在 linear classifier + CAA 构造下验证，不必然适用于其他去偏见方法（如 prompt-based 或 fine-tuning）。
- 作者自述未来工作将探索"与置信度解耦的真正偏见方向"，以及该混淆效应是否影响其他 safety-oriented steering vectors。

## 研究启发与可借鉴点
1. **交叉数据集验证方向的本质**：将仅在偏见数据集上训练的分类器/方向，迁移到完全不含偏见概念的通用知识数据集（MMLU、OpenBookQA）上评估，可快速检验该方向是否真正捕获语义概念——本文以此设计简洁有力地证明了置信度混入问题。
2. **构造对照方向并度量余弦相似度**：同时构造"目标方向"（去偏见）与"混杂方向"（置信度），比较二者夹角，为理解 intervention vector 的真实含义提供量化标准。
3. **同时报告 bias score、accuracy 与 entropy 三件套**：仅看 bias score 下降会误导，必须配合 accuracy 与 entropy 变化才能判断"去偏见"是源于偏好纠正还是置信度压制。
4. **与本团队方向结合机会**：若本团队研究 LLM 可解释性或安全对齐，可将此"方向解耦分析框架"迁移到其他 steering 场景（如 truthfulness、refusal、sycophancy 方向），检验是否存在类似的置信度混淆。
5. **提示工程层面的启发**：在 BBQ 等包含"拒绝回答"选项的数据集上，模型倾向于通过降低置信度来"改善偏见分数"，提示我们在设计公平性评测时需谨慎解读 abstention rate 的变化。

## 关键术语表
**Activation Steering（激活干预）**：在推理时向模型残差流叠加一个预计算的向量，以引导模型输出朝向特定行为方向的轻量级干预方法。
**CAA（Contrastive Activation Addition）**：通过取偏见与反偏见提示的隐藏状态差均值来构造 steering 向量的方法（Rimsky et al., 2024）。
**Linear Representation Hypothesis（线性表征假设）**：认为 LLM 中的语义概念在激活空间中呈线性分布，可通过线性操作读取或操纵。
**AUROC（Area Under the Receiver Operating Characteristic Curve）**：衡量二分类器判别力的指标，0.5 为随机水平，1.0 为完美分离。
**Bias Score（BBQ）**：歧义上下文下 $s_{\text{AMB}}=(1-\text{accuracy})\cdot s_{\text{DIS}}$，其中 $s_{\text{DIS}}=2\cdot\frac{n_{\text{biased}}}{n_{\text{biased}}+n_{\text{anti-biased}}}-1$；正值表示偏见。
**Answer Entropy（答案熵）**：衡量模型在候选答案上概率分布的分散程度，高熵表示模型不确定。
**Coat（Confounding）**：本文指"模型置信度"与"社会偏见"在激活空间中难以解耦，导致去偏见方向实际捕获的是置信度这一混杂现象。
**Optimal Layer（最优层）**：在某一层上训练的分类器在 held-out 测试集上达到最高 AUROC 的隐层，通常位于中间层。

## 可复现要素
- **数据集**：BBQ、StereoSet、CrowS-Pairs、MMLU、OpenBookQA 均为公开数据集；论文未声明专属数据集划分脚本，但提供了详细的 prompt 格式说明（Appendix A.2，图 6-8）及训练/测试集 7:3 划分方式。
- **代码/权重**：代码未明确开源声明（论文未提及 GitHub 链接）；所用模型（Llama-3.1-8B、Falcon3-7B、Ministral-3-8B、Qwen3.5-9B）均为开源权重。
- **关键超参**：$\alpha$ 扫描步长 0.5（从 -1 到 2）；默认报告 $\tilde{\alpha}$（$\alpha_{\text{nat}}$ 若 $>1$，否则 $\alpha=1$）；logistic regression 分类器在各模型各数据集的最优层见附录 Table 6；MMLU 子集为 10,000 prompts。
