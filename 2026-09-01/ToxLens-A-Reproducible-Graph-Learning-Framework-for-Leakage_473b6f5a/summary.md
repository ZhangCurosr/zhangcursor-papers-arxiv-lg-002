---
title: "ToxLens-A-Reproducible-Graph-Learning-Framework-for-Leakage"
source: https://arxiv.org/pdf/2608.30472v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:45:21"
field: "计算毒理学与分子机器学习"
keywords: ["molecular toxicity prediction", "graph neural networks", "multi-task learning", "conformal prediction", "applicability domain", "explainable AI", "data leakage"]
innovations: ["UMAP-HDBSCAN 泄漏控制分子划分策略", "图+全局特征并行架构的晚期拼接融合", "SHAP-guided 反事实验证的毒基元挖掘工作流"]
benchmarks: ["Tox21 Challenge", "TDC ADMET (AMES, DILI, hERG)", "11-endpoint in-house UMAP-HDBSCAN split"]
---

# 论文速读：ToxLens-A-Reproducible-Graph-Learning-Framework-for-Leakage

## 一句话总结
ToxLens 是一个可复现的多任务图学习框架，针对分子毒性预测中的数据泄漏、不确定性量化与可解释性问题，在11个毒性端点上的泄漏控制测试集中实现了宏观 MCC 0.44、AUROC 0.83 和 AUPRC 0.58，系统性地提升了预测的可靠性与化学可解释性。

## 研究问题与动机
- **数据泄漏问题**：结构相似分子（近重复或同类化合物）跨训练/测试折分布，导致基准性能虚高，无法反映实际化合物分级场景中的泛化能力。
- **可解释性缺失**：高分辨率分子模型往往缺乏化学可解释的预测依据，毒理学家和药物化学家难以将预测结果回溯至驱动结构的毒性基元（toxicophore）。
- **不确定性估计不足**：二元分类器被强制给出确定预测，即便化合物远离训练分布或端点稀疏、噪声大；需要支持"拒判"或模糊预测。
- **可靠性评估单一**：单纯依靠 AUROC 等鉴别指标不足以支撑化合物优先级排序决策，需结合适用域、概率校准和反事实验证的综合可靠性栈。

## 核心贡献（创新点）
- **泄漏控制的分子划分策略**：引入 sphere-exclusion 过滤 + UMAP-HDBSCAN 聚类划分，替代传统随机划分，显著减少结构泄漏；与已有工作的本质区别在于以分子簇为单位分配折而非分子个体。
- **图编码器与全局特征并行架构**：5 层 GINE 图网络与 3190 维全局特征并行编码后在预测头处晚期拼接（late concatenation）；与 GCMI/FiLM 等融合变体对比，证明简单晚期拼接为本数据集最优。
- **Conformal-style 预测集与适用域审计**：结合温度缩放 MC dropout 与 conformal 预测集，按端点报告单标签/双标签集合效率；同时按 Tanimoto 相似度四分位分层评估适用域表现。
- **SHAP 引导的反事实毒基元挖掘**：从 GradientSHAP 归因到 occlusion 控制的子图级验证，49 个共识 motif 中 44 个通过反事实标准，提供模型衍生的结构假设库。

## 方法详解
- **化学标准化与重复处理**：使用 RDKit 选最大片段、电荷中和、互变异构规范化，生成 canonical isomeric SMILES；二元端点严格共识去重，冲突标签设为缺失；LD50_Zhu 连续值先平均再二值化（阈值 2.5）。
- **Sphere-exclusion 过滤**：在 Morgan 指纹空间使用 RDKit LeaderPicker，Tanimoto 相似度上限 0.95（距离阈值 0.05），保留代表分子后再进行 UMAP-HDBSCAN 划分。
- **UMAP-HDBSCAN 划分**：半径-2、1024-bit Morgan 指纹经 UMAP 降至10维（Jaccard 距离，100邻域，seed 42），L2 归一化后 HDBSCAN 聚类（最小簇大小 15，最小样本 5）；噪声点视为单例簇；簇级别按 any-positive 代理标签分层分配 80:10:10 折。
- **特征工程**：
  - 图特征：134 维节点特征（元素、价态、形式电荷、手性、杂化、芳香性、Gasteiger 电荷、环归属、SMARTS 反应性标记等）+ 35 维边特征。
  - 全局特征张量宽 3190，拼接 Morgan 指纹（1024）、RDKit 描述符（217）、毒性模式特征（76，含 PAINS）、MolFormer-XL 嵌入（768）、3D 描述符（905）及 PubChem-bioactivity 预留块（200，实验时为零填充）。
- **神经网络架构**：
  - 图主干：线性投影至 256 维 → 5 层残差 GINE 消息传递层（图归一化、stochastic depth、virtual node、多头注意力池化 + 均值/最大值池化）。
  - 全局分支：独立前馈投影，无 global-to-node 调制。
  - 晚期拼接（late concatenation）：图嵌入与全局向量仅在预测头之前拼接，经共享前馈层 + 任务组特定塔 + 端点输出头。
  - 超参：hidden width 256、lr 3e-4、weight decay 0.01、batch size 128、edge-drop 0.06、max stochastic depth 0.20、hidden dropout 0.20、global-input dropout 0.25、cosine-restart period 45、label smoothing 7e-4。
- **损失函数**：掩码二元交叉熵（masked BCE），宏平均，固定端点强调权重（NR-ER/NR-ER-LBD 1.35，SR-p53 1.30 等），class-weight exponent = 0（无类别加权）。
- **MC Dropout + Conformal 预测集**：30 次 dropout forward pass，温度参数在验证 logits 上最小化 BCE 拟合；非一致性分数为 $1-p$（正样本）或 $p$（负样本）；校准分位数取 $\min(1, \lceil(n+1)(1-\alpha)\rceil/n)$ 处的上经验分位数，$\alpha=0.05$。
- **SHAP-guided occlusion 与毒基元挖掘**：
  - 每个正样本heavy atom SHAP 值归一化后，筛选超过 0.20 且高于分子内严格正归因 90 分位数的种子原子（最多3个）。
  - 向外扩展1-2条键，直到捕获分子总正归因质量的至少 35%。
  - 共识簇用 1024-bit Morgan + DBSCAN 聚类（最小3次出现）。
  - 反事实验证：将 motif 原子及其键特征置零，概率下降需 ≥0.10、超过同大小随机子图均值 ≥0.05、单侧 p ≤ 0.10。

## 实验与结果
- **数据集**：11 个毒性端点（Ames、LD50_Zhu、hERG_Karim、Tox21 中 8 个核受体/应激反应端点），来源 Therapeutics Data Commons (TDC) 和 published hERG_Karim 数据集。
- **基线**：四种基于 ECFP4 的浅层模型——Random Forest、XGBoost、MLP、SVM，在同一划分下训练并采用相同验证阈值选择协议。
- **主要结果（UMAP-HDBSCAN 测试集）**：
  - 五种子 soft-voting 集成：macro MCC 0.44，macro AUROC 0.83，macro AUPRC 0.58。
  - 相比最强浅层基线（RF）提升：SR-MMP (+0.19)、SR-HSE (+0.0.16)、NR-ER (+0.15)、SR-p53 (+0.13)；最小提升为 NR-AhR (+0.03)。
  - 单模型 macro AUROC 范围 0.75–0.90，SR-MMP 最高（0.90），hERG_Karim 最低（0.75）。
- **消融实验**：
  - 移除全局路径（A3，GNN-only）最弱（test MCC 0.341）。
  - 晚期拼接（A2/F4）最优（MCC 0.438/0.425），GCMI 次之（0.382/0.373），FiLM（0.398）、mid-trunk concat（0.403）。
- **Conformal 预测集效率**：NR-ER-LBD 达到 100% 单标签，LD50_Zhu 仅 20.4%，Ames 42.4%。
- **适用域分层**：Q1→Q4，macro AUROC 从 0.77 升至 0.86，mean MCC 从 0.28 升至 0.44。
- **外部基准**：
  - Tox21 Challenge：macro AUROC 0.82，低于 DeepTox DNN-only（0.84）和 winning ensemble（0.85）。
  - TDC AMES：0.83±0.01，优于 AttentiveFP（0.814）但低于 Chemprop-RDKit（0.850）和 CMPNN（0.843）。
  - TDC DILI：0.89±0.02，接近 Chemprop-RDKit（0.887）。
- **可解释性**：SHAP-guided occlusion 平均 logit 下降 0.55 vs 随机 0.04，faithfulness gap 0.51；49 个共识簇中 44 个通过反事实标准，包括 SR-MMP 酚类、hERG_Karim 叔胺醇、Ames 芳香胺等化学合理 motif。

## 相关工作脉络
- **DeepTox / Tox21 Challenge**：ToxLens 在 Tox21 Challenge 上以竞争性但不是领先的 AUROC（0.82 vs 0.84/0.85），定位为协议复现与可靠性评估而非 SOTA 竞争。
- **Chemprop / CMPNN / AttentiveFP（TDC ADMET）**：ToxLens 在 AMES 上优于 AttentiveFP 但落后于 Chemprop-RDKit；说明架构可迁移但未形成普遍 SOTA。
- **MoleculeNet 系列工作**：本文强调 MoleculeNet 式随机划分存在结构泄漏风险，提出 UMAP-HDBSCAN 作为更严格的替代。
- **QSAR 最佳实践文献**（Tropsha 2010; Fourches et al. 2010）：遵循化学标准化、重复处理、外部验证等原则。
- **Conformal Prediction in Cheminformatics**（Norinder et al. 2014; Alvarsson et al. 2021）：将 conformal 预测引入毒性预测场景，报告双标签集合作为不确定性信号。
- **Explainable AI for Molecular Toxicity**（Jiménez-Luna et al. 2020; Hooker et al. 2019）：强调 saliency maps 需配合 occlusion 控制验证 faithfulness，而非仅展示可视化。

## 局限性与未来方向
- **单一划分**：内部比较仅基于一个 UMAP-HDBSCAN 切分，未评估划分变异性；需要重复多次化学感知划分验证结论稳健性。
- **Conformal 保证不严格**：验证集同时用于 checkpoint 选择、温度缩放和校准，破坏了 split-conformal 有限样本覆盖保证的独立性假设。
- **可解释性未实验验证**：SHAP、occlusion 和 motif 挖掘均为模型级别分析，未进行前瞻性实验毒性验证。
- **PubChem-bioactivity 块为空**：200 维 PubChem 块因缓存缺失全为零，无法评估其贡献。
- **未来方向**：评估多次划分与前瞻性时间/外部队列；在部署漂移下重新校准 conformal 预测；实验验证模型衍生的毒基元 motif；探索 assay-condition 元数据（细胞系、暴露时间、读数类型）支持更大规模多任务模型。

## 研究启发与可借鉴点
- **泄漏控制划分的标准化流程**：Sphere-exclusion + UMAP-HDBSCAN 簇级划分可作为分子机器学习论文的标准评估协议，值得在本团队 QSAR 研究中推广。
- **晚期拼接作为强基线**：GCMI/FiLM 等复杂融合机制在本任务上不如简单晚期拼接，提示在图+全局特征多模态融合中应先验证简单拼接的天花板。
- **Conformal 预测集作为不确定性报告工具**：双标签集合机制可直接移植到化合物筛选工作流，将"不确定"化合物标记为需额外实验验证的候选物。
- **SHAP-guided occlusion 验证范式**：将归因方法与 within-model 特征扰动控制结合，可为分子可解释性研究提供更严谨的 faithfulness 证据，避免仅依赖 saliency map 可视化。
- **适用域分层评估**：按 Tanimoto 相似度四分位报告性能变化，可作为模型部署前的常规审计步骤。

## 关键术语表
- **ToxLens**：本文提出的可复现多任务图学习框架，整合泄漏控制划分、uncertainty 校准和可解释性分析。
- **UMAP-HDBSCAN split**：基于 UMAP 降维 + HDBSCAN 聚类的分子划分策略，以簇为单位分配训练/验证/测试集以减少结构泄漏。
- **Sphere-exclusion filtering**：在 Morgan 指纹空间应用 Tanimoto 相似度阈值（≤0.95）去除近重复分子，确保划分间结构分离。
- **Late concatenation**：图编码器和全局特征编码器分别在各自 trunk 处理，仅在预测头前拼接的融合方式。
- **Conformal-style prediction sets**：基于 conformal prediction 理论构建的单/双标签预测集合，双标签表示模型在当前校准水平下无法明确区分正负类。
- **Applicability domain (AD)**：模型预测可靠的结构空间；本文按最大 Tanimoto 相似度四分位分层评估。
- **SHAP-guided occlusion**：使用 GradientSHAP 定位关键原子，随后将这些原子的特征置零并观察预测变化的 faithfulness 验证方法。
- **Toxicophore**：与毒性活动相关的分子子结构；本文通过共识 motif 挖掘生成模型衍生的结构假设库。

## 可复现要素
- **数据集**：TDC Tox21 任务（Ames、LD50_Zhu、8 个 Tox21 端点）+ published hERG_Karim 数据集；均已公开可获取。
- **代码**：开源，GitHub https://github.com/Magnushst/toxlens
- **权重/Checkpoint**：提供选定的单模型和集成 checkpoint。
- **环境**：提供 Conda environment specification 和 Windows package lock。
- **关键超参**：hidden width 256、lr 3e-4、weight decay 0.01、batch size 128、5 层 GINE、edge-drop 0.06、stochastic depth 0.20、dropout 0.20、label smoothing 7e-4、温度缩放 α=0.05、30 次 MC dropout passes。
- **外部基准**：Tox21 Challenge 原始划分 + TDC ADMET 固定折。
