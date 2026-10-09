---
title: "MovieSTAGE-Scene-Transition-and-Global-Encoding-for-Movie-fM"
source: https://arxiv.org/pdf/2610.09306v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:54:23"
field: "自然主义fMRI预测分析"
keywords: ["movie-fMRI", "hypergraph neural network", "ADHD classification", "functional connectivity", "event-aligned representation", "naturalistic paradigm", "connectome-based prediction"]
innovations: ["提出事件对齐的场景-转换-全局多尺度融合框架，首次在电影fMRI中显式建模叙事边界", "以FC-profile余弦相似度构建受监督超图，捕捉场景内高阶脑区组织", "引入无符号ΔFC转换编码，量化相邻场景网络重配置幅度而不假设符号方向"]
benchmarks: ["CMI-HBN Despicable Me 三分类AUROC 0.75", "CMI-HBN NoDx vs ADHD AUROC 0.69", "CMI-HBN ADHD-I vs ADHD-C AUROC 0.73"]
---

# 论文速读：MovieSTAGE-Scene-Transition-and-Global-Encoding-for-Movie-fMRI-ADHD-Classification

## 一句话总结
本文提出 **MovieSTAGE**，一个事件对齐的多尺度电影 fMRI 表征框架，通过融合**场景内超图功能连接**、**相邻场景间无符号 ΔFC 转换编码**与**全片全局功能连接**，在 CMI-HBN 队列的 NoDx vs ADHD、ADHD-I vs ADHD-C 及三分类任务上均取得最高平均 AUROC 与平衡准确率，验证了叙事事件对齐表征的增量预测价值。

## 研究问题与动机
- **核心问题**：现有预测模型多依赖全段功能连接（FC）或时间无关表征，未能对齐自然主义电影刺激中的**叙事事件边界**，难以捕捉事件内稳定模式与事件边界处的网络重配置信号。
- **现有方法不足**：
  1. 传统 connectome-based predictive modeling 使用 whole-run FC，**折叠了段内时间结构**；
  2. 动态 FC 研究多基于滑动窗口或隐状态快照，**未绑定叙事事件**，预测增益来源不清；
  3. 现有图/超图模型通常操作于全段或固定窗口，**缺乏事件级分层建模**。
- **动机**：电影 fMRI 中皮层活动在同一事件内相对稳定、在事件边界处发生转变；因此假设**场景内组织、相邻场景重配置、全片全局 FC**三者提供互补信息。

## 核心贡献（创新点）
1. **提出 MovieSTAGE 多尺度框架**：联合场景级超图 FC-profile 表示、相邻场景无符号 ΔFC 转换表示与全片全局 FC，首次将叙事事件对齐引入电影 fMRI ADHD 预测。
   - **区别**：不同于全段 FC 或滑动窗口方法，本框架以**人类标注的叙事场景边界**为锚点构建多层次表征。
2. **在三个 CMI-HBN 任务上验证最优性能**：MovieSTAGE 在 NoDx vs ADHD（AUROC 0.69）、ADHD-I vs ADHD-C（AUROC 0.73）及三分类（AUROC 0.75）均取得最高均值，且经配对 Bootstrap/置换检验显著优于最强基线。
   - **区别**：相较 ED-HNN、STNAGNN、BrainHGT 等脑网络模型，本文强调**事件对齐 + 多尺度融合**带来的增量增益。
3. **受控消融与编码器对比实验**：证明 Scene、Transition、Global 三分支存在**条件互补**；在相同输入下，HGNN 场景编码器显著优于 MLP、GAT、BNT；人类标注分割显著优于时长匹配的随机分割与固定状态数 GSBS 分割。
   - **区别**：以往工作多直接比较模型，本文通过**严格匹配下游组件与搜索预算**分离出超图归纳偏置与叙事对齐的独立贡献。
4. **模型衍生的网络层级可解释性假设**：后验分析揭示场景超图锚点分布于 FPN、DMN、LIN、SMN、SAN 等多系统；T5 转换（孩子离开→Gru 想念）的 ΔFC 重配置在 ADHD 组中 FPN–DMN 与 DMN–DMN 更强，与 ADHD 文献中任务正/负网络分离受损发现一致。
   - **区别**：不仅提供预测性能，还生成**可检验的神经机制假设**（如事件边界处 DMN 整合增强）。

## 方法详解
- **预处理与分区**：10-min Despicable Me 电影 fMRI，SC-100 脑区划分，使用成人标注的 8 个场景边界（每边界向右偏移 4 s 近似血流动力学延迟）。
- **FC 估计**：对每个 subject 与时间块（各场景、全片）独立计算 **Ledoit–Wolf 收缩协方差** → 相关矩阵 → Fisher z 变换，对角线置零。
- **Scene 分支（超图编码）**：
  - 每个场景内构建 ROI FC-profile 相似度矩阵 $\mathbf{S}_{i,s}$（余弦相似度）；
  - 以每个 ROI 为锚点，连接其 Top-K 最相似 ROI 构成超边，共 $R$ 条超边；
  - 超图邻接矩阵 $\mathbf{H}$ 结合可学习超边权重 $w_e^{(l)}$（经 Softplus 下限约束），消息传递公式：
    $$\mathbf{Z}^{(l+1)} = \sigma\!\left((\mathbf{D}_v^{(l)})^{-1/2}\mathbf{H}\mathbf{W}_e^{(l)}\mathbf{D}_e^{-1}\mathbf{H}^\top(\mathbf{D}_v^{(l)})^{-1/2}\mathbf{Z}^{(l)}\mathbf{W}^{(l)}\right)$$
  - 节点嵌入经 mean/max/std 池化得场景嵌入，跨场景注意力池化聚合为 $\mathbf{h}_i^{\text{scn}}$。
- **Transition 分支（转换编码）**：
  - 相邻场景 FC 差值的元素级绝对值 $\Delta \mathbf{FC}_{i,q}^{\text{trn}} = |\mathbf{FC}_{i,q+1}^{\text{scn}} - \mathbf{FC}_{i,q}^{\text{scn}}|$，避免符号方向假设；
  - 每个 ROI 行经双层 MLP 编码，跨 ROI 与跨转换分别池化，得到 $\mathbf{h}_i^{\text{trn}}$。
- **Global 分支与融合**：
  - 全片 FC 向量化为 $\mathbf{x}_i^{\text{glb}} \in \mathbb{R}^{R(R-1)/2}$，线性投影加 dropout 得 $\mathbf{h}_i^{\text{glb}}$；
  - 三分支拼接后经 MLP 分类器输出 logits，采用类别权重交叉熵损失。
- **训练细节**：AdamW（lr $1.5\times10^{-4}$，weight decay $2\times10^{-5}$，batch 16），早停 patience 10，10 次重复 5 折分层交叉验证，生成完整 OOF 预测。

## 实验与结果
- **数据集**：CMI-HBN Despicable Me 电影 fMRI，260 名 6–11 岁被试（NoDx 71、ADHD-I 86、ADHD-C 103），SC-100 分区，QC 标准（median FD ≤ 0.2 mm，模板注册 NCC ≥ 0.8）。
- **基线**：SVM（全片 FC 向量）、ED-HNN、BioBGT、DSAM、STNAGNN、BrainHGT。
- **主要结果（Table II）**：
  | 任务 | MovieSTAGE AUROC | 最佳基线 AUROC | 提升 | MovieSTAGE BACC | 最佳基线 BACC | 提升 |
  |---|---|---|---|---|---|---|
  | NoDx vs ADHD | 0.69 ± 0.04 | BrainHGT 0.66 | +0.03 (p=0.028) | 67.6 ± 3.9% | DSAM 60.4% | +7.2 pp (p=0.008) |
  | ADHD-I vs ADHD-C | 0.73 ± 0.04 | DSAM 0.70 | +0.03 (p=0.036) | 69.8 ± 2.2% | DSAM 65.8% | +4.0 pp (p=0.019) |
  | 三分类 | 0.75 ± 0.03 | STNAGNN 0.72 | +0.03 (p=0.021) | 58.3 ± 3.1% | BrainHGT 55.9% | +2.4 pp (p=0.047) |
- **消融（Table III）**：Global 单分支最强（AUROC 0.68），但完整三分支达到 0.75；Holm 校正后全模型显著优于任一双分支组合。
- **编码器对比（Table IV）**：HGNN 在 scene-only 与 full fusion 下均最优，full fusion 中较 GAT 提升 AUROC 0.04、BACC 5.1 pp。
- **分割对比（Table V）**：人类标注分割（AUROC 0.75）显著优于随机分割（0.70）与 GSBS（0.71），证实叙事对齐价值。

## 相关工作脉络
1. **Connectome-based predictive modeling（Shen et al., 2017）**：本文扩展其 whole-run FC 范式，证明事件对齐的多尺度表示可进一步提升预测性能。
2. **动态 FC 分析（Preti et al., 2017; Savva et al., 2019）**：本文采用事件边界而非滑动窗口定义 FC 时间段，避免主观窗口选择，并聚焦重配置幅度而非有符号变化。
3. **超图神经网络在脑网络中的应用（Xiao et al., 2019; Wang et al., ICLR'23 ED-HNN）**：本文提出基于 FC-profile 相似度的**受监督超图构建**（锚点 Top-K），而非无监督或固定拓扑，并与 ADHD 预测任务直接结合。
4. **时空图/Transformer 模型（STNAGNN, MIDL'25; BrainHGT, AAAI'26; BioBGT, ICLR'25）**：基线模型操作于全段或隐状态快照；本文通过**显式事件分割与转换编码**提供可解释的结构先验。
5. **神经状态边界检测（GSBS, Geerligs et al., 2021）**：本文在**固定状态数**条件下对比发现，人类叙事标注仍优于数据驱动分段，说明认知结构的信息增益超越纯神经聚类。
6. **ADHD 与默认模式/顶额网络研究（Fair et al., 2010; Mills et al., 2018; Duffy et al., 2021）**：本文后验分析重现 FPN–DMN 分离受损与 DMN 整合增强现象，为模型提供神经合理性验证。

## 局限性与未来方向
- **单一队列与年龄范围**：仅在 CMI-HBN 儿童队列（6–11 岁）验证，尚未在成人或跨中心数据上测试泛化性。
- **场景时间支持不均衡**：不同场景时长差异较大（如 S8 达 159 TR，S3 仅 45 TR），可能影响场景内 FC 估计稳定性。
- **超图构建依赖 K 值选择**：Top-K 邻域规模可能影响表示容量，未做敏感性分析。
- **因果关系未明**：ΔFC 绝对值反映重配置幅度，无法区分符号方向；后验关联不能替代因果推断。
- **未来方向**：外部验证（独立电影 fMRI 队列）、更长刺激材料、替代事件定义（如自检测边界）、结合多模态（结构 MRI、行为量表）、探索因果动态建模。

## 研究启发与可借鉴点
1. **事件对齐多尺度表征范式**：将自然刺激划分为认知/叙事事件，分别建模事件内组织、事件边界转换、全局上下文，可迁移至自闭症、抑郁等其他精神障碍的电影 fMRI 预测。
2. **FC-profile 超图构建策略**：以 ROI 的 FC 行向量作为相似性度量基础，构建锚点 Top-K 超边，兼顾个体差异与高阶关系，优于固定邻接或全连接假设。
3. **严格受控对比实验设计**：通过固定下游组件、搜索预算、分割数量，单独改变场景编码器或分段方法，可清晰剥离各模块贡献，值得在复杂多分支模型评估中借鉴。
4. **无符号 ΔFC 编码思想**：放弃符号方向假设，采用绝对差值衡量网络重配置强度，对跨被试变异性更具鲁棒性，适用于任何涉及状态转换的时序脑网络分析。
5. **模型衍生可解释性管道**：学习到的超边权重与转换热图可直接映射至已知脑网络（Yeo 7-network），生成可检验的神经假设，实现“性能-解释”双重目标。

## 关键术语表
**Movie-fMRI**：让被试观看自然主义电影并同时采集 fMRI，利用共享的时序刺激捕获生态效度更高的脑动态。
**Functional Connectivity (FC)**：不同脑区 BOLD 时间序列之间的统计依赖关系，常用 Pearson 相关或收缩估计。
**Hypergraph Neural Network (HGNN)**：处理超图结构的 GNN，超边可连接多个节点，适用于编码高阶脑区相互作用。
**Ledoit–Wolf Shrinkage**：一种正则化协方差估计方法，通过压缩样本协方差到单位矩阵改善高维小样本下的估计稳定性。
**Fisher r-to-z Transform**：将相关系数映射为近似正态分布的 z 值，便于统计推断与跨被试比较。
**Balanced Accuracy (BACC)**：各分类器灵敏度与特异度的平均值，缓解类别不平衡时的评估偏差。
**Out-of-Fold (OOF) Prediction**：通过交叉验证为每个样本生成未见过的预测，用于无偏性能估计与统计检验。
**Event Boundary**：叙事中情节或场景转换的时间点，常伴随皮层活动模式的显著重组。

## 可复现要素
- **数据集**：CMI-HBN Despicable Me 电影 fMRI，公开可用（需申请）。
- **代码/权重**：论文未明确声明开源仓库，但实现基于 PyTorch；建议联系作者获取。
- **关键超参**：K=6（超图邻域大小），Scene 编码器 2 层 HGNN，dropout 0.1，AdamW lr=1.5e-4，weight decay=2e-5，batch size=16，早停 patience=10，最大 80 轮。
- **评估协议**：10 次重复 5 折分层 CV，类权重按 fold 内训练集计算，Holm 校正 + 10,000 次置换检验。
- **预处理**：C-PAC 管道，median FD ≤ 0.2 mm，模板 NCC ≥ 0.8，SC-100 分区，边界偏移 4 s。
