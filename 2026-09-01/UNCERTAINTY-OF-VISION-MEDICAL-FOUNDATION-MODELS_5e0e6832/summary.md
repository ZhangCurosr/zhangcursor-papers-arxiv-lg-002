---
title: "UNCERTAINTY-OF-VISION-MEDICAL-FOUNDATION-MODELS"
source: https://arxiv.org/pdf/2608.30390v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:02:21"
field: "医学影像AI可靠性与不确定性量化"
keywords: ["uncertainty quantification", "conformal prediction", "medical foundation models", "calibration", "vision transformers", "point prediction", "region prediction"]
innovations: ["在固定ViT-Large架构下系统比较7个基础模型（领域特定vs通用）在9个医学数据集上的点/区域双重不确定性", "揭示点预测校准质量与保形预测效率之间的解耦关系，证明后验校准无法弥合预训练来源导致的不确定性差距"]
benchmarks: ["Retina (Retina/IDRiD/APTOS2019)", "Histopathology (CRC100K/TCGA/BraTS-Path)", "Chest X-Ray (RSNA-Pneumonia/POLCOVID/COVID-Rad)"]
---

# 论文速读：UNCERTAINTY-OF-VISION-MEDICAL-FOUNDATION-MODELS

## 一句话总结
本文系统评估了医学视觉基础模型在点预测和区域预测两种不确定性量化框架下的表现，发现领域特定的自监督预训练能够带来更优校准的点预测和更高效的保形预测（conformal prediction），而后验校准技术无法弥合不同预训练来源模型间的不确定性差距。

## 研究问题与动机
1. **医学AI亟需可靠的不确定性估计**：在高风险医疗决策场景中，模型不仅需要高精度，还需准确表达预测不确定性以支持人机协作；然而现有深度模型的softmax置信度通常严重失配（mis-calibration），缺乏预测覆盖率的正式保证。
2. **点预测校准的局限性**：传统方法依赖训练后概率输出，易出现过自信问题（cross-entropy过拟合特定标签）；温度缩放、标签平滑等后验校准技术能否有效消除不同预训练策略引入的不确定性差异尚不明确。
3. **区域预测在医学场景的价值未被充分探索**：保形预测（conformal prediction）能生成具有有限样本覆盖保证的预测集，但对基础模型预训练方式（领域特定vs通用、自监督vs监督）如何影响区域预测效率尚无系统研究。
4. **现有工作存在矛盾结论**：Guo等认为更大网络校准更差，Minderer等认为架构比预训练规模更重要，Hendrycks等认为好的预训练可改善不确定性——亟需在同一模型架构（ViT-Large）下控制变量地验证这些结论。

## 核心贡献（创新点）
1. **系统性评估了7个视觉基础模型在3种医学影像模态上的点/区域双重不确定性**：首次在同一实验设置下（Linear Probing + ViT-Large backbone）比较领域特定模型（RETFound, CTransPath, UNI, MRM, Rad-DINO）与通用模型（ImageNet21k, DINOv2, BioMedCLIP）的不确定性行为，消除了架构差异的混淆因素。
2. **揭示了领域特定自监督预训练对两种不确定性量化的差异化影响**：领域特定预训练结合自监督学习显著改善点预测校准（ECE），但对保形预测集尺寸的提升因模态而异（组织病理学和X光显著改善，视网膜领域提升有限）。
3. **证明了后验校准无法弥合预训练来源导致的不确定性鸿沟**：温度缩放和标签平滑虽能在某些情况下降低不确定性指标，但原本不确定性较高的模型在校准后仍保持相对更高的不确定性，校准技术不能替代模型预训练阶段的选择。
4. **发现点预测校准质量与保形预测效率之间存在解耦**：经过更好的点预测校准（如温度缩放后ECE降低）并不必然带来更小的保形预测集，校准良好的模型仍可能输出与之前相近的分位数阈值。

## 方法详解
- **实验协议**：采用 **Linear Probing**（仅训练线性分类层）而非全参数微调，原因有二：(1) 避免cross-entropy过拟合导致的过自信；(2) 与近期医学基础模型仅开放特征不开放权重的实践一致。
- **模型集合**：所有模型均为 **ViT-Large** 架构，区别仅在于预训练数据：
  - 通用模型：**ImageNet21k**（监督学习，21k类别）、**DINOv2**（自监督蒸馏，1.42亿图像）、**BioMedCLIP**（对比多模态学习，1500万医图文对）
  - 领域特定模型：**RETFound**（视网膜，160万自监督）、**CTransPath**（病理，1500万对比自监督）、**UNI**（病理，1亿高质量WSI蒸馏自监督）、**MRM**（X光，MIMIC-CXR掩码重建）、**Rad-DINO**（X光，纯图像DINOv2自监督）
- **点预测评估指标**：
  - **ECE**（Expected Calibration Error）：将预测概率分M个bin，计算每个bin内准确率与平均置信度的加权偏差之和，公式为 $ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |acc(B_m) - conf(B_m)|$
  - **Brier Score**：预测概率与实际结果的均方误差，$BS = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)^2$
  - **NLL**（Negative Log-Likelihood）：对真实标签概率的负对数似然
- **校准技术**：
  - **Temperature Scaling**（后验）：用单一标量T平滑logits输出，$p_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$，在独立held-out集上通过最小化NLL优化T
  - **Label Smoothing**（训练时正则化）：修改目标分布为one-hot与均匀分布的混合，等价于在CE损失上增加熵正则项 $\mathcal{L} = \mathcal{L}_{CE} + \epsilon \mathbb{E}_u[-\log p_j]$，实验中测试$\epsilon \in \{0.05, 0.10, 0.15, 0.20\}$
- **区域预测（保形预测）**：
  - **LAC**（Least Ambiguous set-valued Classifier）：在calibration集上计算conformal score $s(x,y)$（分类任务中为$1-\text{softmax}(\text{logits})_y$），取$\lceil(n+1)(1-\alpha)\rceil/n$分位数作为阈值$\hat{q}$，预测集为$C_{\hat{q}}(x)=\{y:s(x,y)\leq\hat{q}\}$
  - **RAPS**（Regularized Adaptive Prediction Sets）：在LAC基础上引入排序正则化项以鼓励更小的预测集
  - **效率度量**：预测集平均尺寸（Set Size）——越小越高效；**覆盖率验证**：确保实证覆盖率≥$1-\alpha$

## 实验与结果
- **数据集（9个，3种模态）**：
  - 视网膜：**Retina**（4类白内障/正常）、**IDRiD**（5级糖尿病视网膜病变）、**APTOS2019**（5级盲症评估）
  - 组织病理学：**CRC100K**（9类结直肠组织）、**TCGA-Lymph**（32类癌症亚型）、**BraTS-Path**（6类胶质瘤亚型，每类2000样本）
  - X光：**RSNA-Pneumonia**（正常vs肺炎）、**POLCOVID**（正常/新冠/肺炎）、**COVID-Rad**（4类肺部影像）
- **点预测关键结果**：
  - UNI在CRC100K上ECE仅**0.20**（ImageNet21k为19.91），在BraTS-Path上ECE为**1.59**（ImageNet21k为13.76）
  - Rad-DINO在RSNA-Pneumonia上ECE为**1.29**（ImageNet21k为3.28），在COVID-Rad上ECE为**1.48**
  - DINOv2在视网膜数据集（APTOS2019）上ECE低至**0.32**（未校准），但在IDRiD上为9.38，显示通用模型跨域表现不稳定
  - 监督预训练的ImageNet21k在所有三个模态的多个数据集上均表现最差（CRC100K ECE=19.91、Retina ECE=7.52、RSNA ECE=3.28）
- **区域预测关键结果**：
  - 病理和X光领域的领域特定模型（UNI、Rad-DINO）在保形预测中产生显著更小的预测集；RETFound在视网膜任务上仅 marginally 改善
  - 温度缩放/标签平滑后点预测校准改善，但预测集尺寸并未相应缩小——校准质量与保形效率解耦
  - 所有实验设置下，LAC和RAPS均实现了≥$1-\alpha$的 empiricial coverage，验证了算法有效性
- **最强结果**：UNI在BraTS-Path上达到Acc **93.74±0.73**、AUROC **99.50±1.74**，ECE低至**1.59**；Rad-DINO在COVID-Rad上达到Acc **97.59±0.37**、AUROC **99.20±0.22**，ECE仅**1.48**

## 相关工作脉络
1. **Guo et al. (2017) Temperature Scaling**：本文在其基础上扩展——证明TS虽能改善部分模型的ECE，但无法消除预训练来源导致的不确定性根本差异；且TS改善的ECE不转化为更小的保形预测集。
2. **Minderer et al. (2021) Revisiting Calibration**：指出架构族比对模型大小/预训练量更重要——本文在固定ViT-Large架构下验证了预训练数据质量和领域匹配度是决定不确定性的关键因素，补充而非 contradict 了该结论。
3. **Hendrycks et al. (2019) Pre-training improves uncertainty**：本文支持其核心观点，但进一步细化——并非所有"预训练"均等有益，领域特定的自监督预训练效果显著优于通用监督预训练。
4. **Angelopoulos et al. (2021/2024) Conformal Prediction**：本文将LAC和RAPS首次系统应用于医学基础模型不确定性评估，区别于前作专注于算法设计改进，本文关注预训练策略对CP效率的影响。
5. **Zhou et al. (2023b) RETFound / Chen et al. (2024) UNI / Pérez-García et al. (2025) Rad-DINO**：本文对这些最新医学基础模型进行了统一的校准/保形预测基准评估，填补了这些模型不确定性行为研究的空白。
6. **Wang et al. (2021) Logit Normalization**：本文引用其关于cross-entropy过拟合导致过自信的论点，解释ImageNet21k监督预训练模型普遍不确定性较高的原因。

## 局限性与未来方向
1. **仅评估了分类任务**：方法可扩展至分割（如SAM-Medical）和报告生成任务，但本文未做验证；保形预测在分割中的region set构建更为复杂。
2. **仅使用Linear Probing**：虽避免了过拟合过自信问题，但与实际全参数微调场景存在差距；微调后的不确定性行为有待研究。
3. **7个模型在9个数据集上的评估虽全面但仍有覆盖盲区**：如皮肤镜、病理大模型（如CONCH、Lunit）等未纳入比较。
4. **校准方法限于温度缩放和标签平滑**：未探索ensemble、Monte Carlo Dropout、Deep Ensembling等更复杂的校准/不确定性量化方法。
5. **跨域泛化未系统评估**：模型在一个模态上预训练后在不同模态下游任务上的不确定性迁移行为尚不清楚。
6. **未评估human-in-the-loop场景**：虽引用了Cresswell & Vouitsis (2024)关于预测集改善人类决策的研究，但未在本文中做临床决策实验。

## 研究启发与可借鉴点
1. **Linear Probing作为基础模型不确定性评估的标准协议值得采纳**：相比全参数微调，线性探测能有效隔离预训练带来的不确定性差异，避免cross-entropy过拟合的混淆，适合作为公平比较基准。
2. **点预测与区域预测应联合评估**：本文揭示了两者的解耦关系——仅看ECE可能得出"模型已足够可靠"的误导性结论，必须结合保形预测集尺寸综合判断；后续工作可建立统一的点+区域不确定性评估框架。
3. **预训练数据质量比规模更重要**：UNI（1亿高质量WSI）优于CTransPath（1500万较低质量patch），Rad-DINO（纯图像自监督）优于BioMedCLIP（文本监督），提示在资源有限时应优先追求数据质量而非数量。
4. **领域匹配度是预训练策略选择的关键指标**：RETFound在视网膜任务上虽有优势但对其他任务泛化有限；建议在构建医学AI pipeline时将预训练-下游数据的域相似度作为模型选型的第一维度。
5. **后验校准应被视为补充而非替代**：温度缩放和标签平滑的计算开销极小，可低成本地在部署前作为不确定性缓解的第二道保障，但不应期望其完全消除预训练层面的不确定性差异。

## 关键术语表
- **Conformal Prediction（保形预测）**：一种后验不确定性量化框架，通过校准集构建预测集，在无分布假设下提供有限样本覆盖保证（coverage guarantee）。
- **Point Prediction Uncertainty（点预测不确定性）**：以单一预测值和置信度概率表示的不确定性，常用ECE、Brier Score、NLL度量，缺乏正式覆盖保证。
- **Temperature Scaling（温度缩放）**：后验校准方法，通过将logits除以可学习标量T来平滑概率分布，计算开销低但仅依赖单一校准集。
- **Label Smoothing（标签平滑）**：训练时正则化技术，将one-hot目标替换为one-hot与均匀分布的混合，等效于引入熵正则项防止过自信。
- **LAC（Least Ambiguous set-valued Classifier）**：保形预测算法，通过calibration集上的conformal score分位数构建预测集，最小化平均预测集尺寸。
- **RAPS（Regularized Adaptive Prediction Sets）**：在LAC基础上引入排序正则化，通过在score函数中加入惩罚项鼓励更小的预测集。
- **ECE（Expected Calibration Error）**：将模型预测概率分bin后计算每个bin内平均置信度与实际准确率的加权偏差，衡量点预测校准质量。
- **Linear Probing（线性探测）**：冻结预训练backbone权重，仅训练新增的线性分类头进行下游评估，避免全参数微调带来的过拟合和过自信。

## 可复现要素
- **数据集**：全部为公开数据集——Retina（Kaggle）、IDRiD、APTOS2019（Kaggle）、CRC100K、TCGA-Lymph、BraTS-Path、RSNA-Pneumonia（Kaggle）、POLCOVID、COVID-Rad；**均已公开**
- **模型权重**：ImageNet21k、DINOv2、BioMedCLIP、RETFound、CTransPath、UNI、MRM、Rad-DINO的预训练权重或特征——论文未明确说明开源状态，部分（如UNI、Rad-DINO）有公开代码库
- **代码**：论文未提供官方开源代码链接，但描述了LAC/RAPS算法伪代码（Algorithm 1）
- **关键超参**：温度缩放在held-out验证集上通过最小化NLL优化T；标签平滑$\epsilon \in \{0.05, 0.10, 0.15, 0.20\}$；保形预测的$\alpha$值用于控制覆盖率；线性层学习率、训练epoch等论文未明确提及
