---
title: "Through-the-Eyes-of-the-Beholder-Biometric-and-Demographic-C"
source: https://arxiv.org/pdf/2609.15608v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:58:11"
field: "多模态内容安全与主观性建模"
keywords: ["sexism detection", "multimodal learning", "meme analysis", "biometric signals", "label distribution learning", "vision-language models", "human-centered AI"]
innovations: ["Biometric and Demographic Conditioning via FiLM for multimodal sexism detection", "Label distribution learning with KL divergence to model annotator disagreement", "Neural-classical ensemble combining deep multimodal network with stylometric/physiological SVM"]
benchmarks: ["EXIST 2026 Task 2"]
---

# 论文速读：Through-the-Eyes-of-the-Beholder-Biometric-and-Demographic-C

## 一句话总结
本文提出了一种以人为中心的多模态框架 VANGUARD，将标注者的生理信号（眼动、心率变异性、EEG）与人口统计特征通过 FiLM 条件化注入交叉注意力网络，用于 meme 中隐蔽性别歧视的检测，并在 EXIST 2026 Task 2 中取得 Subtask 2.2 排名第 29/114 的成绩。

## 研究问题与动机
1. **核心问题**：meme 中的性别歧视通常以讽刺、反讽等隐性方式呈现，单一模态或单一"客观标签"难以捕捉其主观性；不同标注者对同一 meme 的判定存在显著分歧。
2. **现有方法不足**：传统 sexism detection 依赖 majority-vote 硬标签，忽略标注者间差异；主流多模态模型未利用标注者生理/人口统计上下文作为可建模信号。
3. **动机**：EXIST 2026 Task 2 首次提供了包含 eye-tracking、heart rate、EEG 等生理数据及人口统计特征的 meme 数据集，为"从标注者视角出发"的学习提供了可能。

## 核心贡献（创新点）
1. **Biometric and Demographic Conditioning via FiLM**：将 4 维生理特征与 4 维人口统计嵌入拼接为 56 维 human-context 向量，通过 FiLM 层对 cross-attention 融合表示进行逐特征缩放/偏移；与已有工作本质区别在于首次将生理信号作为可微条件注入多模态 classifier 而非仅做后验分析。
2. **Label Distribution Learning with KL Divergence**：将 Subtask 2.1 建模为标注者标签分布预测问题，优化 soft-label KL 散度而非 majority-vote binary loss；区别于传统分类器对 disagreement 视为噪声的处理。
3. **VLM-based Text and Visual Enrichment + Cross-lingual Augmentation**：用 Gemma 4（4-bit Effective-4B）提取 meme embedded text 并生成结构化视觉描述，再用 NLLB-200 进行英西互译实现训练集翻倍；相比仅使用原始像素/文本，提供 richer representation 并缓解语言分布偏移。
4. **Neural-Classical Ensemble (Soft-voting)**：深度多模态网络与基于 stylometric+physiological 特征的 RBF-SVM 软投票集成；SVM 充当 algorithmic regularizer，在翻译噪声增大时提供互补信号。

## 方法详解
- **Text Enhancer**：使用 Ollama 调用 4-bit Gemma 4 (Effective-4B)，temperature=0.0、top_p=0.1 严格提取嵌入文本；视觉描述失败时用 temperature=0.4 重试。
- **Cross-lingual Augmentation**：对 cleaned text 和 visual description 用 NLLB-200 做英↔西双向翻译，生成 `_aug` 后缀样本；按 parent ID 分配 train/val 防止 leakage。
- **Sensor Autoencoder**：4 维生理特征（log reaction time, fixation count, saccade count, HR Std.）经 StandardScaler 标准化后，输入对称 MLP（4→64→32→64→4）预训练 50 epochs（MSE 损失、Adam），提取 32 维 latent representation 作为 sensor embedding。
- **Cross-Attention Fusion**：
  - 共享 LoRA-XLM-RoBERTa（rank=16, α=32, dropout=0.1）编码 embedded text 和 visual description；
  - LoRA-CLIP ViT-B/32 编码 meme RGB 图像；
  - 所有序列线性投影至 256 维后用 LayerNorm + dropout；
  - Description 作 query、image patch 作 KV 的 multi-head cross-attention 得到 h_fused ∈ R^256。
- **FiLM Conditioning**：人口统计嵌入（gender/age/education/ethnicity）经 embedding layer 后平均；与 32 维 sensor embedding 拼接得 56 维 human-context 向量；两个线性层生成 per-feature scale γ 和 shift β，调制表示：h_mod = h_fused ⊙ (1+γ) + β（FiLM 权重零初始化）。
- **Classification Heads**：h_mod 与 text [CLS] token 拼接得 512 维 fused state；三个任务头：Subtask 2.1（binary）、2.2（3-class）、2.3（6-class multi-label）。
- **Auxiliary Losses**：
  - KL divergence（主目标，soft-label 分布学习）；
  - Supervised Contrastive Learning（SupCon，τ=0.07）作用于 128 维投影，强化同 consensus label 样本聚集。
- **Ensemble Inference**：ĥp = α·p_DL + (1−α)·p_SVM，α 和阈值 θ 在 val 上 grid search 最大化 macro F1；SVM 输入 17 维手工特征（stylometry + demographic ratios + log-scaled physiological signals）。

## 实验与结果
- **数据集**：EXIST 2026 Task 2，3984 个 meme，16 个标注者，英西双语近乎平衡（EN 2005 / ES 1979）。
- **评估协议**：Soft（预测分布与 annotator 分布比较）与 Hard（majority-vote 硬标签）两种 ICM-Norm。
- **主要结果**：
  - Subtask 2.2（source intention）软评估排名 **29/114**，ICM-Soft Norm=0.3389（All）；硬评估排名 71/183，ICM-Hard Norm=0.3612。
  - Subtask 2.1（binary sexism）硬评估：EN 排名 86/214，ICM-Hard Norm=0.5496，F1 YES=0.7336；ES 排名 141/214，ICM-Hard Norm=0.4306，F1 YES=0.6697。
  - Subtask 2.3（fine-grained 6-class multi-label）ICM-Hard Norm=0.0703，接近 trivial baseline。
  - 均大幅超越 majority-class baseline（2.1: 0.2947 vs 0.4861；2.2: 0.1369 vs 0.3612）。
- **Ablation 关键发现**：
  - 无 augmentation 纯 DL 模型在 val 上取得最高 ICM-Hard Norm=0.5210；cross-lingual augmentation 提升 ES（0.4249→0.4925）但降低 EN（0.5494→0.5001）。
  - SVM ensemble 在无 augmentation 时下降（0.5210→0.4963），在有 augmentation 时提升（0.4561→0.4963），说明 SVM 在噪声训练 regime 下起正则化作用。
  - **Holm-Bonferroni 校正后 63 项 pairwise 测试无一显著**，表明各组件贡献可能被集成效应掩盖，单组件无法独立证明统计显著性。
- **语言鸿沟**：EN-ES gap 从 val（0.5609 vs 0.4679）扩大到 test（0.5496 vs 0.4306），暗示 XLM-RoBERTa/CLIP 对英语网络 sexism 习语编码更强。

## 相关工作脉络
1. **Hateful Memes Challenge (Kiela et al., 2020)**：指出多模态模型在"单独无害、组合冒犯"场景下失效；本文延续该视角并引入标注者生理信号。
2. **Learning from Disagreement (Uma et al., 2021; Wu et al., EMNLP 2023)**：将标注者分歧视为信号而非噪声，采用 soft-label KL/对比学习；本文在此基础上引入 biometric/demographic conditioning。
3. **FiLM (Perez et al., AAAI 2018)**：视觉推理的条件化层；本文首次将其用于多模态 sexism detection 的 annotator-context 调制。
4. **Eye-tracking / EEG 在 affective computing 的应用 (Lim et al., 2020; Fu et al., 2023)**：此前仅用于 emotion recognition；本文将其引入 sexism detection，但 EEG bandpower 未通过显著性检验。
5. **EXIST 系列 (2024-2026)**：从 tweet 到 meme 再到视频逐步扩展；2026 首次提供生理数据，本文是该设定下的首个以 human-centered 为核心的submission。
6. **LoRA-adapted XLM-RoBERTa / CLIP**：参数高效微调范式；本文冻结 backbone、仅训练 LoRA adapter（rank 16, α=32）以保持跨语言泛化。

## 局限性与未来方向
1. 生理/人口统计信号按 meme 级平均，丢弃了个体层面 variation，未来可探索 viewer-specific 模型。
2. 眼动特征高度共线性（Fixations 与 Saccades ρ≈1.0），4 维 sensor vector 实际独立信息有限。
3. EEG bandpower 未通过统计显著性检验，仅通过 SVM 间接利用；更丰富的时序 EEG 建模可能释放信号。
4. 多 seed 消融未发现显著单组件效应，需更大 seed 数或更低噪声评估 regime 才能分辨小效应。
5. Subtask 2.3 严重 class imbalance 导致 minority 类别 F1=0，多任务干扰是瓶颈而非架构问题。
6. 跨语言翻译增强存在 EN-ES trade-off，可引入 translation-quality filtering 或 back-translation consistency check。
7. 高 aspect ratio 的 storytelling meme 仅用 square padding，未尝试 sub-image splitting。
8. 深度网络与 SVM 分阶段优化、grid-search 集成，未来可尝试 end-to-end 联合训练。

## 研究启发与可借鉴点
1. **FiLM 作为条件化通用接口**：将任何外部 context（用户画像、传感器、情境）转化为 scale/shift 参数注入 cross-attention 表示，可迁移至其他 subjective NLP 任务（仇恨检测、立场识别、情感分析）。
2. **Label Distribution Learning 配合熵加权**：按 annotator entropy 实例加权 KL loss，可推广至任何多人标注、存在分歧的 classification/regression 任务。
3. **Neural-Classical Ensemble 作为 regularizer**：当深度模型在 noisy/augmented 数据上训练时，辅以 handcrafted-feature SVM 可提供稳定性；适用于小规模多模态标注任务。
4. **Sensor Autoencoder 预训练 + FiLM 注入**：低维 noisy 传感器信号先用 autoencoder 学习 robust latent，再经 FiLM 调制，避免高维 encoder 梯度淹没生理信号。
5. **跨语言增强但保留语言标识**：英西互译翻倍的策略可用于低资源语言对齐，但需配合 quality filter 避免反向退化。

## 关键术语表
- **FiLM (Feature-wise Linear Modulation)**：通过 learned scale γ 和 shift β 对特征图进行逐元素仿射变换的条件化模块。
- **Label Distribution Learning**：预测标注者标签的完整概率分布而非单一 hard label 的学习范式。
- **Kullback-Leibler Divergence (KL 散度)**：衡量两个概率分布差异的信息论度量，本文用于 soft-label 训练目标。
- **Supervised Contrastive Learning (SupCon)**：拉近同标签样本嵌入、推远异标签样本嵌入的辅助对比损失。
- **Cross-attention Grounding**：以视觉描述序列为 query、图像 patch 序列为 KV 的注意力机制，用于解决 meme 图文交互理解。
- **Hard vs Soft Evaluation**：Hard 基于 majority-vote 标签评估；Soft 基于 annotator 标签分布评估，后者更能反映模型对 disagreement 的建模能力。
- **Holm-Bonferroni Correction**：控制 family-wise error rate 的逐步降序多重检验校正方法，本文用于消融显著性检验。
- **EXIST 2026 Task 2**：CLEF 2026 下的 sexism detection 子任务，包含 meme、多语言、生理信号（眼动/心率/EEG）与人口统计数据。

## 可复现要素
- **数据集**：EXIST 2026 Task 2 dataset（arXiv:2602.23862），包含 3984 个 meme 及 16 个标注者的生理/人口统计数据；论文声明公开 pipeline。
- **代码/权重**：论文声明 "We release our full pipeline and analysis to support reproducible human-centered modeling"，具体仓库链接未在正文给出。
- **关键超参**：LoRA rank=16, α=32, dropout=0.1；learning rate（fusion heads 3e-5, LoRA 8e-6）；gradient accumulation=4；AdamW weight decay=0.1；SupCon τ=0.07；sensor AE 训练 50 epochs；消融 5 seeds。
