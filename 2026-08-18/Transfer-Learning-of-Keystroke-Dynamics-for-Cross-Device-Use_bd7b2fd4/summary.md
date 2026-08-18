---
title: "Transfer-Learning-of-Keystroke-Dynamics-for-Cross-Device-Use"
source: https://arxiv.org/pdf/2608.16334v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:38:23"
field: "跨设备行为生物识别"
keywords: ["keystroke dynamics", "cross-device authentication", "transfer learning", "behavioral biometrics", "inductive transfer", "equal error rate"]
innovations: ["提出 TEDxBC 框架，通过归纳式迁移编码器显式学习手机到平板的击键特征域变换", "迁移编码器变换数据与有限目标域真实数据融合，解决次要设备训练样本稀缺问题", "扩展至 24 维击键特征（含 DEFT 距离增强特征），显著提升跨设备用户区分能力"]
benchmarks: ["BBMAS", "KVC-onGoing (future work)"]
---

# 论文速读：Transfer-Learning-of-Keystroke-Dynamics-for-Cross-Device-Use

## 一句话总结
本文提出 TEDxBC（Transfer Encoder Data-fusion cross-device Binary Classification），一种基于归纳式迁移学习的跨设备击键动力学用户认证系统，通过迁移编码器将源设备（手机）的击键特征映射到目标设备（平板）域，并与有限的目标域真实数据融合，训练用户专属的二分类器；在 BBMAS 数据集上实现 14.2% 的 EER，超越 DoubleType (DT) 和分层迁移学习 (STL) 等基线方法。

## 研究问题与动机
1. **跨设备击键分布漂移**：用户在手机、平板等不同形态设备上打字习惯会发生显著变化，导致从源设备学到的动力学模型难以直接迁移到目标设备。
2. **目标设备训练数据匮乏**：现实中用户在次要设备上积累的击键样本有限，无法支撑充分训练一个独立的用户认证模型。
3. **已有方法的不足**：Lin 等 [5] 依赖临时微调而非显式迁移学习；Sun 等 [7] 仅对齐特征空间而未显式学习域间变换；Monaco 等 [12] 使用相同设备的不同设置而非真正跨设备场景。此外，多数前人工作仅使用 4–6 个特征，区分能力受限。
4. **连续认证的落地障碍**：若每个设备都需独立从头训练，用户接受度低，制约了连续认证在金融反欺诈等场景的大规模部署。

## 核心贡献（创新点）
1. **提出 TEDxBC 框架**：首次将归纳式迁移学习（inductive transfer learning）显式引入跨设备击键认证，通过编码器-解码器结构学习源→目标的显式特征映射，区别于双类型 (DT) 仅利用设备间特征关系、STL 仅对齐特征空间的做法。
2. **迁移编码器 + 数据融合策略**：将迁移编码器变换后的源域数据与少量目标域真实数据融合，扩充目标域训练集，缓解目标域数据稀缺问题；消融实验证明编码器是核心贡献（EER 从 27.0% 提升至 14.2%）。
3. **扩展至 24 维击键特征集**：在经典 6 个时间特征基础上，引入 DEFT（Distance-Enhanced Flight Times，16 维）和非传统特征（2 维），全面捕捉击键空间与编辑行为信息，相比前人最多 16 特征显著提升表征能力。
4. **用户专属随机森林分类器 + 细粒度超参搜索**：为每个用户独立训练 RF 分类器，通过网格搜索优化树深度、叶节点样本数、集成规模等 5 类超参，兼顾个性化与抗过拟合。
5. **在 BBMAS 数据集上刷新 SOTA**：手机→平板跨设备场景下 EER 降至 14.2%，相对 DT (24.7%) 和 STL (35.0%) 分别绝对降低 10.5pp 和 20.8pp。

## 方法详解
TEDxBC 由三个组件串联构成：

**1. Transfer Encoder（迁移编码器）**
- 采用 encoder-decoder 架构，输入源域特征向量 $x_i$，经全连接层 + tanh 激活得到隐层表示：$h_i = \tanh(W x_i + b_1)$，隐层值域 $[-1, +1]$。
- 隐层后接入 dropout 层（drop rate $\gamma = 0.3$），再经 tied weight 解码：$\hat{z}_i = \tanh(W^T h_i + b_2)$，输出为目标域空间的近似特征。
- 训练损失为重构误差：$L(z_i, \hat{z}_i) = \|z_i - \hat{z}_i\|_2$，即源域变换结果与目标域真实样本的 $l^2$ 距离。
- 参数初始化策略针对每个用户单独选取（Glorot Uniform/Normal、He Normal/Uniform、Random Uniform 中择优），并配合用户级验证集 early-stopping 防止过拟合。
- 源域与目标域样本通过二分采样策略（bipartite sampling）配对，构建无重叠的跨设备源-目标训练/验证集。

**2. Data Fusion（数据融合）**
- 训练完成的编码器将源域全部样本变换至目标域空间，生成虚拟目标域样本 $\hat{Z}$。
- 将 $\hat{Z}$ 与目标域真实有限样本合并，构成扩充后的目标域训练集，供二分类器使用。

**3. Binary Classifier（二分类器）**
- 采用 Random Forest，每用户独立训练专属模型。
- 超参网格搜索范围：`max_tree_depth` ∈ {10, 30, 50, 70, 90, 100, None}，`min_samples_leaf` ∈ {1, 2, 4}，`min_samples_split` ∈ {2, 5, 10}，`n_estimators` ∈ {100, 200, 400, 600, 800, 1000}，`criterion` ∈ {'gini', 'entropy'}。
- 正负样本均衡：每个子集加入等量非本用户特征向量构建 2-class 平衡数据集。

**特征工程（共 24 维）**
- 6 个时间特征：F1/F2/F3/F4 四种 flight time 的中位数 + hold time 中位数 + tri-graph hold time 中位数。
- 16 个 DEFT 特征：按键盘左右侧 × 距离 {1,2,3,4} × F1–F4 中选出的最具区分性子集。
- 2 个非传统特征：错误率中位数 + negative up-down 特征中位数。

**实验数据划分**：每用户每设备约 45 个样本（每个样本 150 次击键，对应社交媒体常见消息长度），按 40%-20%-40% 分割源域和目标域，分别用于编码器训练/验证/变换和分类器训练/测试。

## 实验与结果
- **数据集**：BBMAS（Syracuse University + AIS，IEEE Dataport，公开），114 名参与者的手机→平板跨设备击键数据。
- **评估指标**：EER、Accuracy、Precision、Recall、F1。
- **基线方法**：DoubleType (DT) [6]、Stratified Transfer Learning (STL) [7]。
- **消融变体**：TExBC（无数据融合，只用源域变换数据）、DxBC（无迁移编码器，仅用目标域原始数据）。

| 方法 | EER | Accuracy | Precision | Recall | F1 |
|------|-----|----------|-----------|--------|-----|
| **TEDxBC（本文）** | **14.2%** | 84.1% | 88.6% | 78.9% | 82.8% |
| TExBC（消融） | 22.1% | 78.3% | 78.9% | 69.2% | 72.1% |
| DxBC（消融） | 27.0% | 68.8% | 66.2% | 60.4% | 61.3% |
| DT [6] | 24.7% | 78.3% | 80.8% | 67.0% | 73.9% |
| STL [7] | 35.0% | 66.8% | 55.9% | 50.1% | 51.0% |

- **核心结论**：
  - TEDxBC 在所有指标上全面领先，EER 较 DT 绝对降低 10.5pp，较 STL 降低 20.8pp。
  - 消融表明迁移编码器贡献最大（去掉后 EER 升至 27.0%），数据融合亦有明显增益。
  - DT 依赖设备间统计关系而非迁移学习，STL 仅做特征空间对齐，均未显式学习域变换，故性能落后。

## 相关工作脉络
1. **DoubleType (DT) [6]**：利用多设备间击键行为的统计关系进行跨设备认证，无需迁移学习但无法适应设备形态差异带来的分布偏移；本文在此基础上引入显式迁移编码器学习变换。
2. **Stratified Transfer Learning (STL) [7]**：将源域和目标域特征空间对齐到公共表示，但不学习具体的域间变换函数；本文通过 encoder-decoder 显式建模变换关系，实现更精准的域适配。
3. **CrossBehaAuth [5]**（Lin et al.）：针对跨场景会话的击键认证，依赖临时微调/重训练而非结构化迁移学习框架；本文采用严谨的归纳式迁移学习范式。
4. **Crossing Domains with Inductive Transfer Encoder [12]**（Monaco & Vindiola）：最早将迁移编码器用于击键生物识别，但仅在相同设备不同设置（左/右手）下验证，未涉及真正跨设备场景；本文将其拓展至手机→平板的真实跨设备任务。
5. **Cross-device Free-text Keystroke Authentication with Federated Learning [15]**（Yang et al., 2024）：基于联邦学习框架处理跨设备认证，适合多用户隐私场景；本文聚焦单用户少样本适配，不依赖分布式训练。
6. **DEFT 特征集 [20]**：本文作者团队前期工作，提出了基于按键距离增强的 flight time 特征，本文将其完整纳入 24 维特征体系，是提升区分力的关键数据来源。

## 局限性与未来方向
- **单一数据集局限**：仅在 BBMAS 一个数据集上验证，跨数据集泛化能力未知，未覆盖其他键盘布局、打字场景（如语音输入辅助、不同语言文本）。
- **设备对固定**：仅评估了 phone→tablet 单向场景，未覆盖其他设备组合（如 phone→laptop、tablet→phone）。
- **未进行 Doddington's Zoo 分析**：不同用户类别（易认证/难认证用户）的性能差异未深入剖析，实际部署中需关注"狼型"（ impostor 易伪装）和"山羊型"（边界模糊）用户的影响。
- **未来方向**（论文自述）：在更多跨设备数据集（包括 KVC-onGoing benchmark [28]）上验证泛化性；按 Doddington's zoo 分类细化分析用户表现差异。

## 研究启发与可借鉴点
1. **Encoder-Decoder 迁移编码器设计可直接复用**：tied weight + tanh + dropout + early-stopping 的轻量结构，适合任何跨设备/跨域小样本适配任务，代码实现简洁。
2. **用户级个性化超参选择策略**：为每个用户单独选择最优初始化器、隐藏层单元数和早停节点，避免"一刀切"超参导致部分用户性能下降——这一策略对个性化生物识别任务有普适价值。
3. **扩展特征集的工程范式**：在经典时间特征基础上，系统性地引入空间距离增强特征（DEFT）和行为编辑特征（错误率等），形成多维度表征，为其他时序行为生物识别（如触控滑动、语音）提供特征扩展思路。
4. **二分采样配对策略（bipartite sampling）**：确保源域与目标域样本无重叠配对，训练更干净；可推广至其他跨域少样本学习场景。
5. **与团队方向的结合机会**：若团队关注多模态生物识别或跨设备连续认证，可将 TEDxBC 的迁移编码器与在线学习/增量学习结合，实现持续适配；也可将 DEFT 特征思路迁移至触摸屏/手势识别的跨设备认证任务。

## 关键术语表
- **Transfer Learning（迁移学习）**：将在源域学到的知识迁移到新域（目标域）的任务中，减少目标域所需训练数据量。
- **Inductive Transfer Learning（归纳式迁移学习）**：源域和目标域标签空间相同，且目标域存在少量标注数据可用于学习域间变换。
- **Equal Error Rate (EER)**：伪接受率（FAR）与伪拒绝率（FRR）相等时的错误率，是生物识别系统的核心评测指标，越低越好。
- **Distance-Enhanced Flight Times (DEFT)**：按键盘按键空间距离（1–4 步）和左右手侧分组统计的 flight time 中位数特征，共 32 种原始组合，本文选取最具区分性的 16 种。
- **Flight Time（飞行时间）**：相邻两次击键之间的时间间隔，分为 F1（up-down）、F2（up-up）、F3（down-down）、F4（down-up）四种变体。
- **Hold Time（持键时间）**：单次按键从按下到释放的时间差。
- **Bipartite Sampling（二分采样）**：将源域和目标域样本配对形成无重叠的跨设备训练样本集的策略。
- **Tied Weights（共享权重）**：编码器权重矩阵 $W$ 与解码器权重矩阵 $W^T$ 互为转置，用于减少参数量并增强正则化。
- **Doddington's Zoo（多格敦动物园）**：按用户认证难度将用户分为绵羊（易认证）、山羊（边界模糊）、狼（易伪装）、公羊（难认证）四类，用于细化分析生物识别系统性能。

## 可复现要素
- **数据集**：BBMAS（Behavioral Biometrics Multi-device and multi-Activity data from Same users），公开可用，DOI: https://doi.org/10.21227/rpaz-0h66，IEEE Dataport，2019。
- **代码/权重**：论文未提及开源。
- **关键超参**：dropout rate $\gamma = 0.3$；样本长度 150 次击键；数据划分 40%-20%-40%；RF 超参搜索范围见论文 Table II；初始化器从 Glorot Uniform/Normal、He Normal/Uniform、Random Uniform 中为用户单独择优。
