---
title: "Latent-space-bias-directions-in-LLMs-capture-confidence-not"
source: https://arxiv.org/pdf/2610.08559v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:08:49"
field: "大语言模型公平性与可解释性"
keywords: ["activation steering", "LLM bias", "fairness", "interpretability", "confidence debiasing", "linear representation"]
innovations: ["揭示去偏方向实质编码置信度而非偏见，在 MMLU/OpenBookQA 上 AUROC 达 0.71-0.90", "证明去偏向量与置信方向余弦相似度高达 0.91，且 Steering 后去偏指标改善源于增加弃权而非纠正偏好", "提出分类器跨域迁移测试与相关性强化模式作为诊断工具，揭示现有去偏方法的虚假改进机制"]
benchmarks: ["BBQ", "StereoSet", "CrowS-Pairs", "MMLU", "OpenBookQA"]
---

# 论文速读：Latent-space-bias-directions-in-LLMs-capture-confidence-not

## 一句话总结
本文通过可解释性分析发现，LLM 中用于去偏的激活调控（activation steering）方向实际上主要编码的是模型置信度，而非语义偏见；沿该方向进行干预虽能降低偏见指标，但本质是迫使模型降低自信、增加"不确定/弃权"倾向，而非真正纠正模型偏见。

## 研究问题与动机
- **核心问题**：当偏见与非偏见提示在 LLM 隐空间中线性可分时，分离方向究竟编码了什么？
- **已有方法不足**：激活调控被广泛用作推理时去偏手段，但先前研究发现其泛化性差、对无关任务有意外副作用，且从单一数据集提取的向量在新数据集上效果有限；现有工作未能解释这一不一致性能的根本原因。
- **动机**：作者希望揭示去偏方向的真实编码内容，以理解为何 activation steering 的去偏效果不稳定，并呼吁对该方法的效果评估保持审慎。

## 核心贡献（创新点）
- **揭示了去偏方向实质编码置信度而非偏见**：通过 Logistic 回归分类器分析发现，区分偏见/非偏见提示的方向对模型置信度的判别力（AUROC 0.94）远超对偏见类型的判别力（AUROC 0.57），且在无偏见概念的通用知识数据集（MMLU、OpenBookQA）上同样能高效分离高/低概率答案。
- **证明去偏方向与置信度方向高度对齐**：CAA 去偏向量与从 MMLU/OpenBookQA 提取的置信方向之间的余弦相似度范围 0.29–0.91，BBQ 和 StereoSet 训练得到的方向平均相似度最高（0.68 和 0.67），表明两者在隐空间几何中几乎重叠。
- **阐明了"去偏"指标改善的虚假性**：Steering 后 BBQ 歧义提示的偏见分数从 0.08 降至 0.01，但这源于模型更多选择"弃权"而非纠正底层偏见偏好；去歧义上下文下的偏见分数 $s_{dis}$ 在 steering 后基本不变，而准确率系统性下降，说明偏见并未真正减少。
- **提出方法层面的警示**：指出现有基于线性表征假设的激活调控去偏方法存在信心-偏见混淆的根本性缺陷，呼吁后续研究探索独立于置信度的去偏方向，并扩展到其他安全性导向的 steering 方法中进行验证。

## 方法详解
- **Bias 线性分类器构建**：从 BBQ（歧义上下文）、StereoSet、CrowS-Pairs 三个偏见数据集中提取最终 token（答案字母）的隐状态，训练 Logistic 回归分类器区分 biased vs anti-biased，使用 AUROC 评估分层可分性。
- **最佳层选取**：对每个模型逐层训练分类器，选取 AUROC 最高的层作为"best layer"（多数模型在中间层，Qwen3.5-9B-Base 除外，其在第 30 层）。
- **CAA 去偏向量构造**：采用 Contrastive Activation Addition (CAA) 方法，计算偏见与非偏见答案对之间的隐状态均值差：
  $$\mathbf{v_b} = \frac{1}{N}\sum_{i=1}^{N}\left(\mathbf{h}_i^{\text{anti-biased}} - \mathbf{h}_i^{\text{biased}}\right)$$
- **Confidence 方向对照构造**：从 MMLU 和 OpenBookQA（无偏见概念）中，以最高/最低概率答案为样本，用相同方式构造置信方向 $\mathbf{v_c}$，用于与 $\mathbf{v_b}$ 比较余弦相似度。
- **Activation Steering 干预**：在 best layer 处对隐状态施加 $\mathbf{h}' = \mathbf{h} + \alpha\hat{\mathbf{v_b}}$，其中 $\hat{\mathbf{v_b}}$ 为单位范数向量，$\alpha$ 为缩放系数（自然幅度 $\alpha_{\text{nat}} > 1$ 时取 $\alpha_{\text{nat}}$，否则取 1）。
- **评估指标**：BBQ 使用 $s_{amb}$ 和 $s_{dis}$ 两个偏见分数（$s_{dis} = 2\frac{n_{\text{biased}}}{n_{\text{biased}}+n_{\text{anti-biased}}}-1$，$s_{amb} = (1-\text{accuracy})\cdot s_{dis}$），StereoSet/CrowS-Pairs 使用偏见答案占比；同时报告准确率、答案熵 $H = -\sum_a P(a|\text{prompt})\log P(a|\text{prompt})$。

## 实验与结果
- **模型**：Llama-3.1-8B（Base/Instruct）、Falcon3-7B（Base/Instruct）、Ministral-3-8B（Base/Instruct）、Qwen3.5-9B（Base/Instruct），共 8 个变体。
- **数据集**：偏见基准 BBQ（歧义+去歧义）、StereoSet、CrowS-Pairs；通用知识基准 MMLU（10,000 题）、OpenBookQA（完整 ~6,000 题）。
- **分类器结果**：BBQ 歧义上下文上分类器 AUROC 最高仅 0.75（Llama-3.1-8B-Instruct 第 15 层），但按置信度（最高/最低概率）标签评测时，同一分类器在 BBQ 去歧义集上 AUROC 达 0.94；在 MMLU 上也能达到 0.71–0.88，证明分类器捕获的是置信度信号。
- **相关性**：Classifier bias score 与模型答案概率的 Spearman 相关系数从歧义上下文（$r_s \in 0.01–0.45$）显著增强至去歧义上下文（$r_s \in 0.57–0.80$），对所有 8 个模型一致。
- **余弦相似度**：去偏方向与置信方向在所有模型/数据集间平均余弦相似度 0.68（BBQ）、0.67（StereoSet）、0.50（CrowS-Pairs），Qwen3.5-9B-Base 达 0.91。
- **Steering 效果（Table 4）**：BBQ 歧义偏见分数普遍下降（如 Llama-3.1-8B-Instruct 从 0.08→0.01），但 BBQ 去歧义 $s_{dis}$ 基本不变；去歧义准确率系统性下降（如 Llama-3.1-8B-Instruct 从 0.87→0.52），证明模型因置信度降低而更多弃权，并非真正去偏。
- **通用知识影响（Table 5）**：在 MMLU/OpenBookQA 上，所有三个去偏方向均导致熵显著上升（如 Llama-3.1-8B-Instruct 在 OBQA 上熵从 0.33→0.57），准确率轻微下降（从 0.76→0.70），证明 steering 效果泛化到无关任务，符合置信度操控特征而非针对性去偏。

## 相关工作脉络
- **Rimsky et al. (2024) CAA [28]**：提出对比激活加法构造 steering 向量的方法，本文沿用该框架，但发现其去偏方向实质编码置信度而非偏见语义。
- **Li et al. (2023) Representation Engineering [36]**：基于线性表征假设的 top-down 干预范式，本文揭示了该假设在公平性场景下的潜在误用风险。
- **Pres et al. (2024) [27]、Tan et al. (2024) [32]**：质疑 steering 向量的可靠性与泛化性，本文从"方向编码内容"的角度提供机制层面解释。
- **Lu & Rimsky (2024) [19]**：发现偏见 steering 与拒绝 steering 方向负相关，本文进一步发现偏见方向与低置信度方向正相关，揭示了更深层的结构。
- **Kumaran et al. (2026) [14]**：证明 steering 向低置信度方向直接增加模型弃权率，本文将此发现扩展到偏见去偏场景，并建立方向几何层面的解释。
- **Nangia et al. (2020) CrowS-Pairs [23]、Parrish et al. (2022) BBQ [26]、Nadeem et al. (2021) StereoSet [22]**：作为主流偏见评估基准，本文发现这些基准上的 bias 度量本身与模型 likelihood 高度耦合，因此基于它们的 steering 方向天然混杂置信度信号。

## 局限性与未来方向
- **仅聚焦 CAA 方法**：本文仅分析了 Contrastive Activation Addition 这一种向量构造方式，未覆盖其他 steering 方法（如方向裁剪、正交化等）。
- **模型范围有限**：仅测试了 4 个模型家族的 8 个变体（7B–9B），未验证更大规模模型（如 70B+）或闭源模型的同等现象。
- **数据集局限**：主要使用 BBQ、StereoSet、CrowS-Pairs，未涉及多语言偏见场景或其他类型的公平性评估。
- **未来方向**：探索独立于置信度的真正去偏方向；检验该混淆是否影响除 activation steering 之外的其他推理时去偏技术；扩展到 AI 安全领域的其他 steering 向量（如 truthfulness、refusal）中验证。

## 研究启发与可借鉴点
- **"指标改善≠机制纠正"的警示范式**：本文展示了如何通过对齐方向和混淆分析来揭示现有方法的虚假改进，这一分析框架可迁移到其他 AI 安全/可控生成研究中验证干预的真实语义。
- **置信度-偏见耦合的诊断工具**：Spearman 相关系数分析 + AUROC 跨域迁移测试（在 MMLU 上验证分类器能力）是一种轻量且有力的诊断手段，可用于检查任何隐空间方向是否混杂了无关语义。
- **实验设计借鉴**：通过构造有无"弃权选项"的对照设置，清晰区分了"模型真正改变偏好"与"模型降低置信度"两种行为，这一实验设计可直接复用于其他 steering 干预研究。
- **对线性表征假设的审慎态度**：本文证实线性分离存在，但分离的方向并非预期语义，这提示团队在使用线性表征假设进行干预设计时，必须额外验证方向的语义纯度，而非仅依赖 AUROC 指标。

## 关键术语表
- **Activation Steering（激活调控）**：在 LLM 推理时向隐状态添加特定方向的向量，以改变模型输出行为的轻权重干预技术。
- **CAA（Contrastive Activation Addition）**：Rimsky 等提出的 steering 向量构造方法，通过计算两类提示隐状态的均值差得到方向向量。
- **Linear Representation Hypothesis（线性表征假设）**：认为语义概念在 LLM 隐空间中沿线性方向编码的假设，是 activation steering 的理论基础。
- **Bias Score（偏见分数）**：衡量模型在偏见基准上倾向于偏见答案的程度，BBQ 中使用 $s_{amb}$/$s_{dis}$，StereoSet/CrowS-Pairs 中使用偏见答案百分比。
- **Answer Entropy（答案熵）**：衡量模型在多个候选答案上概率分布的分散程度，高熵表示模型不确定/低置信度。
- **AUROC（受试者工作特征曲线下面积）**：评估线性分类器区分能力的指标，0.5 为随机水平，1.0 为完美分离。

## 可复现要素
- **数据集**：BBQ、StereoSet（intrasentence 子集）、CrowS-Pairs（过滤为仅一词差异的 1018 对）、MMLU（10,000 题子集）、OpenBookQA（完整数据集）；论文未明确声明全部公开状态，标准数据集均为公开可用。
- **代码/权重**：未明确声明开源代码；使用了 Llama-3.1-8B、Falcon3-7B、Ministral-3-8B、Qwen3.5-9B 四个开源模型（含 Base 和 Instruct 变体）。
- **关键超参**：Steering 强度 $\alpha$ 取自然幅度 $\alpha_{\text{nat}}$（当 $>1$ 时）或 1；分类器使用 logistic regression；答案概率从 next-token logits 读取；最终 token 为答案字母，在其隐状态上操作。
