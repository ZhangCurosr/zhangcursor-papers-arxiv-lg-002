---
title: "MovieSTAGE-Scene-Transition-and-Global-Encoding-for-Movie-fM"
source: https://arxiv.org/pdf/2610.09306v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:54:28"
field: "自然态fMRI与图神经网络结合的精神障碍计算表型"
keywords: ["movie-fMRI", "hypergraph neural network", "ADHD classification", "functional connectivity", "naturalistic neuroimaging", "event-aligned representation"]
innovations: ["事件对齐的场景-过渡-全局三分支多尺度融合框架", "基于FC-profile相似度的自适应超图构建与可学习超边权重"]
benchmarks: ["CMI-HBN Despicable Me movie-fMRI", "NoDx vs ADHD", "ADHD-I vs ADHD-C", "三分类NoDx/ADHD-I/ADHD-C"]
---

# 论文速读：MovieSTAGE-Scene-Transition-and-Global-Encoding-for-Movie-fM

## 一句话总结
本文提出MovieSTAGE，一个多尺度事件对齐框架，通过场景级超图FC、相邻场景FC重构差异与全电影全局FC的融合，在儿童Mind研究所（CMI-HBN）《神偷奶爸》电影fMRI数据上实现ADHD分类，三项任务的AUROC分别达0.69、0.73和0.75，均为各任务中最优。

## 研究问题与动机
1. **现有方法忽略叙事结构**：当前预测模型多依赖整段运行（whole-run）功能连接（FC）或时间eneric表示，将大脑活动的时间组织信息抹平，无法捕捉叙事事件边界处的模式切换。
2. **动态FC的事件对齐缺口**：已有动态图/注意模型在滑动窗口或潜在快照上工作，而非叙事事件，难以区分预测增益源于时间变化还是事件对齐。
3. **局部vs全局信息缺失**：场景内组织、场景间重构与全电影全局上下文三者可能提供互补信息，但尚未有方法系统整合。
4. **ADHD神经异质性需更高阶建模**：ADHD涉及前顶叶控制网络（FPN）、默认模式网络（DMN）等多系统交互异常，单一vectorized FC不足以刻画高阶关系。

## 核心贡献（创新点）
1. **事件对齐的多尺度融合框架**：首次在同一框架中联合建模场景内超图FC组织、相邻场景无符号FC差异重构、以及全电影全局FC，三者互补而非冗余。
2. **基于FC-profile相似度的场景级超图构建**：利用Ledoit–Wolf收缩协方差估计每个场景的FC矩阵，以ROI行向量（FC-profile）的余弦相似度构建超边，捕捉多ROI高阶关系，避免短窗FC估计不稳定问题。
3. **自适应可学习超边权重的HGNN编码器**：引入多层感知机（MLP）处理超边属性与均值嵌入，通过Softplus保证正下界，使模型可自适应学习哪些场景内连接模式更具诊断价值。
4. **严格的受控消融与边界对齐验证**：对比了不同场景编码器（HGNN vs MLP/GAT/BNT）与不同分段策略（人工标注 vs 随机 vs GSBS），证明叙事对齐与超图归纳偏置均带来显著提升。

## 方法详解
**数据预处理**：260名6–11岁儿童（NoDx=71, ADHD-I=86, ADHD-C=103），使用C-PAC流水线处理，QC标准为中位 Framewise Displacement ≤0.2mm、模板配准归一化相关系数≥0.8；采用SC-100 Schaefer ROI划分；8个人工标注场景边界（每边界后移4秒以近似血流动力学延迟）。

**FC估计**：对每个场景/全电影独立应用Ledoit–Wolf收缩协方差估计，转化为Pearson相关矩阵后对角置零，再对非对角元素做Fisher r-to-z变换。

**Scene分支（超图FC）**：
- 对每个场景 $s$，计算 $\mathbf{FC}_{i,s}^{\text{scn}}$，以每行 $\mathbf{x}_{i,s,r}^{\text{scn}}$ 作为ROI $r$ 的FC-profile。
- 构建余弦相似度矩阵 $\mathbf{S}_{i,s}$，以每个ROI $r$ 为锚点，连接其Top-K=6最相似ROI形成超边 $e_{i,s,r}$，共 $R=100$ 条超边。
- 超边属性矩阵存储锚点与成员的平均余弦相似度。
- 使用带可学习超边权重的HGNN（两层），层更新公式：
$$\mathbf{Z}^{(l+1)} = \sigma\left((\mathbf{D}_v^{(l)})^{-1/2}\mathbf{H}\mathbf{W}_e^{(l)}\mathbf{D}_e^{-1}\mathbf{H}^\top(\mathbf{D}_v^{(l)})^{-1/2}\mathbf{Z}^{(l)}\mathbf{W}^{(l)}\right)$$
- 超边权重：$w_e^{(l)} = \text{Softplus}(f_{\text{edge}}([\bar{\mathbf{z}}_e^{(l)}, a_e])) + \epsilon$。
- 池化：对ROI嵌入做均值/最大值/标准差池化得到场景嵌入，再用注意力池化跨场景聚合为 $\mathbf{h}_i^{\text{scn}}$。

**Transition分支（相邻场景FC重构）**：
- 计算相邻场景间无符号FC差异：$\Delta\mathbf{FC}_{i,q}^{\text{trn}} = |\mathbf{FC}_{i,q+1}^{\text{scn}} - \mathbf{FC}_{i,q}^{\text{scn}}|$，形状 $(S-1)\times R\times R$。
- 每个ROI级别的ΔFC行经两层MLP编码，均值/最大值/标准差池化跨ROI，注意力池化跨过渡得 $\mathbf{h}_i^{\text{trn}}$。

**Global分支（全电影FC）**：
- 对整个电影应用相同收缩FC估计，向量化上三角非对角元素得 $\mathbf{x}_i^{\text{glb}} \in \mathbb{R}^{R(R-1)/2}$，经线性投影+dropout得 $\mathbf{h}_i^{\text{glb}}$。

**融合与训练**：
- 拼接三分支嵌入：$\mathbf{h}_i^{\text{cat}} = [\mathbf{h}_i^{\text{scn}}, \mathbf{h}_i^{\text{trn}}, \mathbf{h}_i^{\text{glb}}]$，经MLP分类器输出 logits。
- 损失：类别加权交叉熵 $\mathcal{L} = \text{CE}_w(\mathbf{o}_i, y_i)$，权重在每折训练集内计算。
- 优化：AdamW，lr=$1.5\times10^{-4}$，weight decay=$2\times10^{-5}$，batch size=16，最多80 epoch，patience=10早停。
- 评估：10次分层5折交叉验证，完整OOF预测，配对主题聚类bootstrap + 10,000次置换检验（Holm校正）。

## 实验与结果
**数据集**：CMI-HBN《神偷奶爸》movie-fMRI，260名儿童，8个场景，SC-100 parcellation。

**基线方法**：SVM（向量化全电影FC）、ED-HNN（ICLR'23）、BioBGT（ICLR'25）、DSAM（MedIA'25）、STNAGNN（MIDL'25）、BrainHGT（AAAI'26）。

**主要结果**（Table II）：

| 任务 | MovieSTAGE AUROC | MovieSTAGE BACC | 最强基线 | 提升幅度 |
|------|------------------|-----------------|----------|----------|
| NoDx vs ADHD | **0.69±0.04** | **67.6±3.9%** | BrainHGT 0.66 / DSAM 65.8% | +0.03 AUROC / +7.2pp BACC |
| ADHD-I vs ADHD-C | **0.73±0.04** | **69.8±2.2%** | DSAM 0.70 / 65.8% | +0.03 AUROC / +4.0pp BACC |
| 三分类 | **0.75±0.03** | **58.3±3.1%** | STNAGNN 0.72 / BrainHGT 55.9% | +0.03 macro-AUROC / +2.4pp BACC |

**消融结果**（Table III）：
- 全局分支单独最强（AUROC=0.68），但全模型比任意双分支组合均显著提升（所有 $p_H\leq0.041$）。
- Scene+Global组合AUROC=0.71，Transition+Global同样0.71，但全模型达0.75，证明三个分支部分互补。
- 相对Transition+Global（最强双分支），全模型提升+0.04 macro-AUROC（95% CI [0.009, 0.071]）和+5.0pp BACC。

**场景编码器对比**（Table IV，三分类）：
- HGNN在Scene-only（AUROC=0.65, BACC=46.7%）和Full-fusion（AUROC=0.75, BACC=58.3%）均最优，显著优于MLP/GAT/BNT（scene-only: 所有$p_H\leq0.047$；full-fusion: 所有$p_H\leq0.027$）。
- 全融合下HGNN较GAT提升+0.04 AUROC和+5.1pp BACC。

**分段策略对比**（Table V，三分类）：
- 人工标注（AUROC=0.75, BACC=58.3%）> GSBS（0.71, 55.8%）> 随机（0.70, 53.2%），所有Holm校正后 $p_H\leq0.047$。
- 相对Random提升+0.05 AUROC和+5.1pp BACC，证明叙事对齐增益超越单纯 segment count/duration 匹配。

**可解释性分析**：
- 场景共识超图在FPN、DMN、LIN、SMN、SAN多系统分散，S3（复印机场景）呈FPN中心拓扑，S7（发射准备+童年记忆）呈DMN/LIN中心拓扑。
- T5（孩子离开→Gru想念）处ΔFC差异最显著： pooled ADHD较NoDx在FPN–DMN和DMN–DMN网络对上有更大重构幅度（BH-FDR校正），与ADHD文献中task-positive/task-negative分离受损一致。

## 相关工作脉络
1. **Connectome-Based Predictive Modeling (Shen et al., 2017)**：奠定用整脑FC向量预测个体行为的基础范式，本文在其基础上引入事件对齐多尺度表示。
2. **Hypergraph Neural Networks for Brain FC (Xiao et al., 2019; Feng et al., 2019)**：将超图用于脑网络高阶关系建模，本文改进为场景级FC-profile相似度驱动的自适应超边构建与可学习权重。
3. **Equivariant Hypergraph Diffusion NN (ED-HNN, ICLR'23)**：最近的高阶图方法，但作用于整段FC而非事件对齐场景，本文在相同设比较中超越。
4. **BioBGT / BrainHGT（ICLR'25 / AAAI'26）**：基于全电影FC的图/层次Transformer，本文证明事件对齐多尺度融合超越单一全局表示。
5. **DSAM（MedIA'25）**：直接处理完整ROI时间序列的自注意力模型，本文在更细粒度事件结构上获得更优判别性能。
6. **Naturalistic fMRI事件结构（Baldassano et al., 2017; Cohen et al., 2022）**：揭示叙事中稳定活动模式与边界处模式切换，为本文事件对齐假设提供认知神经科学依据。
7. **Dynamic FC与事件边界（Preti et al., 2017; Savva et al., 2019）**：动态连接研究提示时变网络配置含额外信息，本文以无符号绝对ΔFC量化重构幅度而非假设符号方向一致性。

## 局限性与未来方向
1. **单一队列外部验证缺失**：仅在CMI-HBN pediatric cohort评估，需在独立movie-fMRI队列（成人/不同电影刺激）验证泛化性。
2. **场景时长不均衡**：8个预定义场景时间跨度差异大（S3仅45TR，S8长达159TR），导致场景级FC估计的信噪比不均。
3. **绝对ΔFC无法区分方向**：无符号差异丢失了连接增强/减弱的有向信息，限制了神经机制解释的深度。
4. **未探索更细粒度事件结构**：仅用8个人工边界，未尝试自动事件检测或更细粒度的叙事分割。
5. **可扩展性待验证**：HGNN在100 ROI×8场景上的计算开销与参数效率，在更大parcellation（如200/400 ROI）或更长刺激下未评估。

## 研究启发与可借鉴点
1. **事件对齐多尺度融合设计可直接迁移**：将自然刺激（电影/故事/对话）按叙事边界分割后，构建"局部事件表示+事件间转换+全局表示"的三分支架构，适用于任何连续naturalistic fMRI prediction任务（如自闭症、抑郁症、认知能力预测）。
2. **FC-profile相似度驱动超图构建**：用ROI行向量的余弦相似度替代固定阈值或KNN邻接来定义超边，能自适应捕捉场景特异性的高阶协同模式，可推广至其他图构建任务。
3. **绝对差异表征transition的无反射假设**：不假设跨被试/跨ROI的符号一致性，仅量化重构幅度，对群体水平异质性更具鲁棒性，可应用于任何动态状态转换建模。
4. **严格受控对比实验设计**：固定 segment count 后比较人工/随机/GSBS边界、固定场景编码器比较不同图架构，这种"控制变量+matched settings"的消融策略为领域建立可靠benchmark提供了方法学范式。
5. **可解释性-预测性联动**：Post-hoc超图共识与ΔFC heatmap生成FPN–DMN交互假设，与已知ADHD文献吻合，验证了预测模型可同时作为假设生成工具，值得在疾病biomarker发现工作中复用。

## 关键术语表
**Movie-fMRI**：使用电影作为自然化刺激的任务态fMRI范式，提供共享的时间结构化脑活动探针，比静息态或隔离trial更能捕获生态效度高的网络动态。
**Functional Connectivity (FC)**：不同脑区BOLD时间序列间的统计依赖关系，本文指Fisher-z变换后的Pearson相关矩阵。
**Ledoit–Wolf Shrinkage**：一种协方差矩阵正则化估计方法，通过收缩样本协方差至目标矩阵改善高维小样本估计稳定性。
**Hypergraph**：超图的边（超边）可连接任意数量节点，比二元图能表达多ROI高阶协同关系；本文每个场景构建R=100条超边。
**FC-profile**：每个ROI与其他所有ROI的关联向量（即FC矩阵的一行），用作场景特异性的节点特征描述子。
**ΔFC（Transition Representation）**：相邻场景间FC矩阵的逐元素绝对差值，量化事件边界处的网络重构幅度。
**Out-of-fold (OOF) Prediction**：在k折交叉验证中，每个样本仅在其所属验证折的预测上汇总，避免同一数据参与训练与评估导致的上溢偏差。
**Subject-cluster Bootstrap / Permutation Test**：以被试为聚类单位的重采样检验，保留样本间相关性结构，提供配对差异的95%置信区间与置换p值。

## 可复现要素
- **数据集**：CMI-HBN Despicable Me movie-fMRI，公开可用（参考文献[23]），含260名儿童；重复性Brain Charts已发布（参考文献[24]）。
- **代码/权重**：论文未明确声明开源代码或模型权重；实现基于PyTorch，在NVIDIA RTX A6000 GPU训练。
- **关键超参**：SC-100 parcellation；K=6（Top-K超边成员数）；HGNN两层，dropout=0.1；AdamW lr=$1.5\times10^{-4}$，weight decay=$2\times10^{-5}$，batch size=16，max 80 epoch，patience=10；class-weighted CE；10次 stratified 5-fold CV。
- **评估协议**：标准化/类别权重在每折训练集内fit；20%验证划分支持12配置搜索；checkpoint选val AUROC最优；complete OOF预测汇总指标；配对主题聚类bootstrap + 10,000次置换检验；Holm校正。
