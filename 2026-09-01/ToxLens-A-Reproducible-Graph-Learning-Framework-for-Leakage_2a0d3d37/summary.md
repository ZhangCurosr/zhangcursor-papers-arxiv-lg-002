---
title: "ToxLens-A-Reproducible-Graph-Learning-Framework-for-Leakage"
source: https://arxiv.org/pdf/2608.30472v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:46:12"
field: "计算毒理学与分子机器学习"
keywords: ["molecular toxicity", "graph neural networks", "leakage control", "conformal prediction", "applicability domain", "explainable AI", "multi-task learning"]
innovations: ["结合 sphere-exclusion 与 UMAP-HDBSCAN 的泄漏控制多任务毒性评估流程", "图与全局特征 late concatenation 在多任务毒性分类中优于 GCMI/FiLM 融合", "SHAP-guided 毒理片段挖掘配合机内反事实遮蔽验证的完整性可靠性审计栈"]
benchmarks: ["Tox21 Challenge", "TDC ADMET (AMES, DILI, hERG)", "UMAP-HDBSCAN 内部 11-endpoint 泄漏控制划分"]
---

# 论文速读：ToxLens-A-Reproducible-Graph-Learning-Framework-for-Leakage

## 一句话总结
ToxLens 是一个面向分子毒性预测的可复现多任务图学习框架，通过严格的泄漏控制数据划分、不确定性校准与可解释性分析，为化合物优先排序提供了兼顾判别性能与可靠性审计的综合评估体系。

## 研究问题与动机
- **结构泄漏导致性能虚高**：当训练集与测试集存在结构相似的分子时，传统基准的 AUROC 等指标会虚高，模型实际上依赖近邻插值而非真正泛化，这对前瞻性毒性预测尤为不利。
- **黑箱模型缺乏化学可解释性**：毒理学家和药物化学家依赖结构警报（structural alerts）、反应基团和毒性片段进行推理，纯黑箱预测分数无法关联到驱动预测的子结构。
- **二元分类器缺少不确定性报告机制**：对于远离训练分布或稀疏/噪声端点，强制输出二分类不利于化合物分流决策，模型应能主动声明"置信不足"。
- **可靠性堆栈缺失**：实际部署需要结构分离度、适用域评估、校准化的拒绝能力、可复现的外部基准验证以及化学基础的可解释性——目前缺乏将上述组件整合于统一工作流的方案。

## 核心贡献（创新点）
1. **泄漏控制的数据划分流程**：结合 sphere-exclusion 过滤（Tanimoto 相似度上限 0.95）与 UMAP-HDBSCAN 聚类划分，将分子级聚类分配至 80:10:10 三折，减少近邻模拟比泄漏；与已有工作的区别在于将该流程与多任务毒性面板、conformal 校准和可解释性管道集成于统一框架中。
2. **图编码与全局特征并行+后期拼接的架构设计**：图 GINE trunk 与 3,190 维全局特征向量（Morgan 指纹+RDKit 描述符+Tox21 SMARTS+MolFormer-XL+3D 描述符）仅在网络头部拼接；消融证明 late concatenation 优于 GCMI、FiLM 和中段拼接。
3. **温度缩放 MC Dropout + Conformal-style 预测集**：在验证集上拟合单一温度参数，30 次 dropout 前向估计预测均值，并基于非一致性得分构建每端点独立的 conformal-style 预测集，输出 singleton 或双标签不确定集；与已有工作的区别在于将其与适用域分层分析和 MC 方差诊断联合使用。
4. **SHAP 引导的毒理片段发现与反事实遮蔽验证**：基于 GradientSHAP 在原子/键级别做归因，通过预设阈值提取种子原子、扩展为 SMILES 片段后用图特征遮蔽（occlusion）对比随机子图控制，产出 49 个共识簇、44 个含通过反事实标准的实例；与已有工作的区别在于提供了一套从归因→片段挖掘→机内屏蔽检验的完整流程。
5. **完整性可靠性审计栈**：包含适用域四分位分层分析（AUROC 从 Q1=0.77 升至 Q4=0.86）、固定外部基准（Tox21 Challenge 与 TDC ADMET）的从头重训练协议，并在报告中明确区分"架构可迁移性"与"泛化保证"。

## 方法详解
- **数据获取与清洗**：11 个二元端点（Ames、LD50_Zhu、hERG_Karim、8 个 Tox21 NR/SR）。连续值 LD50_Zhu 取 replicate 均值后以 2.5 截断为二元；冲突标签置为 missing 并采用 masked loss。
- **化学标准化**：RDKit 选取最大片段、电荷中和、互变异构规范化、生成 canonical isomeric SMILES 并加氢。Sphere exclusion 在 Morgan fingerprint 空间用 RDKit LeaderPicker，Tanimoto ≤ 0.95 去重，保留代表分子。
- **UMAP-HDBSCAN 划分**：半径-2、1024 位 Morgan 指纹经 UMAP（100 邻域、Jaccard 距离、seed=42）降至 10 维，L2 归一化后用 HDBSCAN（min_cluster_size=15, min_samples=5）聚类，噪声点作单元素簇。按簇分配 80:10:10，分层代理为 any-positive（任意主端点为正即置 1），中位数二值化。
- **特征工程**：
  - 图节点 134 维（元素、度、形式电荷、手性、H 数、杂化、芳香性、Gasteiger 电荷、ring membership、pharmacophore flags、Crippen/TPSA/electrotopological/Labute 表面积、价态等）；边 35 维（键级、立体化学、共轭、ring 大小、杂原子/卤素上下文、SMARTS 反应性）。
  - 全局向量宽 3,190：1024 位 Radius-2 Morgan + 217 RDKit 描述符 + 76 毒性模式（75 SMARTS + 1 PAINS）+ 768 维 MolFormer-XL + 905 维 3D 描述符（WHIM/GETAWAY/USR/USRCAT/MORSE/RDF/形状标量）+ 200 维 PubChem bioactivity（实验中未缓存，全为 0，有效非零宽度 2,990）。
- **网络架构**：
  - 图 trunk：线性投影 134→256，5 层残差 GINE（图归一化 + stochastic depth + 虚拟节点注入），dropout rate 0.20、edge-drop 0.06、max stochastic depth 0.20。
  - 池化：multi-head attention pooling 与 mean/max pooling 结合。
  - 全局 branch：sanitization → input dropout(0.25) → LayerNorm → FFN。
  - 拼接策略：**仅在预测头前 concat**，无 GCMI/FiLM/mid-trunk 融合。
  - 共享 trunk + 任务组塔（genotoxicity / stress response / nuclear receptor / cardio-systemic / general toxicity）+ 每端点独立 logit 输出。
  - 5 个 seed（42–46）独立训练，按最高验证 MCC 选择 checkpoint，ensemble 概率为 5 个 sigmoid 输出的算术平均。
- **损失函数**：masked macro-averaged BCE + label smoothing (ε=7e-4) + 固定 endpoint 强调权重（Ames/LD50_Zhu=1.10、NR-AhR/SR-MMP=1.20、hERG/NR-Aromatase=1.25、SR-p53=1.30、NR-ER/NR-ER-LBD=1.35）。类权重 exponent=0，实际为无类权重 BCE。
- **超参数**：lr=3e-4、weight decay=0.01、batch=128、cosine-restart period=45。
- **Uncertainty & Conformal**：温度 t 在验证集上最小化 binary NLL；30 次 MC dropout 前向得均值 p̄ 与标准差；非一致性得分：y=1 时 1-p̄，y=0 时 p̄；α=0.05 下取上经验分位数 q；推理时 class 0 包含若 p̄ ≤ q，class 1 包含若 1-p̄ ≤ q，两者同时满足则输出 {0,1} 不确定集。
- **适用域**：测试分子按 max Tanimoto to training 分 4 四分位（Q1 最远 → Q4 最近）。
- **可解释性管道**：
  - Captum PyG-Captum-SHAP 0.1.5 在 134+35+global 三路输入联合做 GradientSHAP；显式 H 归并到重原子，边归并到键。
  - 片段挖掘：正 SHAP 原子 > 0.20 且 >90 分位，每个分子最多 3 个种子；向外扩展至第一次覆盖 ≥35% 正 SHAP 质量；Morgan 聚类（DBSCAN，min≥3 occurrences）。
  - 反事实屏蔽：将片段原子/键特征置零，对比 20 个同分子匹配随机子图；通过需满足：Δprob ≥ 0.10、比随机均值大 ≥ 0.05、单边 empirical p ≤ 0.10。

## 实验与结果
- **内部 UMAP-HDBSCAN 分裂**（11 端点，~2,970 分子测试集）：
  - Single checkpoint：macro MCC 0.40 / AUROC 0.82 / AUPRC 0.56 / Brier 0.11。
  - 5-seed soft-voting ensemble：macro MCC **0.44** / AUROC **0.83** / AUPRC **0.58**。
  - 超过全部 4 个 ECFP4 浅基线（RF/XGBoost/MLP/SVM）在所有 11 端点上。最大边际：SR-MMP (+0.19)、SR-HSE (+0.16)、NR-ER (+0.15)、SR-p53 (+0.13)；最小：NR-AhR (+0.03)、Ames (+0.04)、hERG (+0.07)。
  - 单端点 AUROC 最高：SR-MMP 0.90、NR-ER-LBD 0.88、Ames 0.86、NR-AhR 0.86、SR-p53 0.84；最低 hERG_Karim 0.75。
- **消融**：
  - 移除全局分支（A3, GNN-only）最差：MCC 0.341 / AUROC 0.791 → 证明全局特征贡献显著。
  - 无中段融合（A2/MCC 0.438 / F4/MCC 0.425）优于 GCMI（A1 0.382 / F1 0.373）、FiLM（0.398）、mid-trunk concat（0.403）。
  - 线性 head 替换 deep head（A4）MCC 0.407 > GCMI deep 0.382，说明 head 深度非主因。
- **Conformal-style 预测集效率**（α=0.05）：singleton 比例端点差异大——NR-ER-LBD 100%、NR-AhR 87.58%、NR-Aromatase 93.30%、SR-HSE 89.83%、SR-p53 83.50%；Ames 42.39%、LD50_Zhu 20.40%、hERG 47.51%、SR-ARE 34.65% 偏低。
- **适用域分层**：mean AUROC Q1=0.77 → Q4=0.86；mean MCC Q1=0.28 → Q4=0.44；ECE Q1=0.13 → Q4=0.08。NR-ER-LBD 单调最强（0.71→0.96）。
- **外部基准**：
  - Tox21 Challenge：12-task macro AUROC 0.82 / AUPRC 0.35 / MCC 0.34；SR-MMP AUROC 0.95、NR-AhR 0.91 最强。低于 DeepTox DNN-only 0.84 与 winning ensemble 0.85。
  - TDC ADMET：AMES 0.83±0.01（>AttentiveFP 0.814，<CMPNN 0.843 / Chemprop-RDKit 0.850）；DILI 0.89±0.02（≈AttentiveFP 0.886 / Chemprop-RDKit 0.887，<Chemprop 0.899）；hERG 0.80±0.05（benchmark 含 6 个跨 split 重复结构，仅作协议复现）。
- **可解释性**：SHAP 遮蔽（mask fraction 0.20）跨端点 logit drop 0.55 vs 随机 0.04，faithfulness gap 0.51；0.05→0.30 逐步扩大（0.19→0.71）。
- **毒理片段**：49 个共识簇，44 个含通过反事实标准的实例。Top motifs：SR-MMP 酚（ArOH, validated 0.75）、hERG 叔β-氨基醇（0.83）、NR-AhR 三取代芳（0.60）、LD50 烷基腈（0.67）、Ames 芳伯胺（0.50）、NR-ER 酚（0.33）。

## 相关工作脉络
1. **DeepTox / Tox21 Challenge**（Mayr et al., 2016）：早期深度毒性竞赛基线；本文在其固定测试集上从头重训练复现，但目标不是刷新 SOTA，而是展示带泄漏审计与可解释性的统一管线。
2. **MoleculeNet / TDC ADMET**（Yang et al., 2019; Wu et al., 2018）：广泛使用的分子性质预测基准，含 Chemprop/CMPNN/AttentiveFP 等参考；本文在 AMES/DILI/hERG 上复现对比，承认未见其全部 SOTA，但强调 split 设计与重复结构对结果的巨大影响。
3. **QSAR 泄漏与活动悬崖研究**（van Tilborg et al., 2022; Sheridan et al., 2020; Wallach & Heifets, 2018）：指出大量分子 ML 基准奖励记忆而非泛化；本文采用 sphere-exclusion + UMAP-HDBSCAN 试图缓解该问题。
4. **Conformal Prediction 在化学信息学中的应用**（Norinder et al., 2014; Alvarsson et al., 2021; Angelopoulos & Bates, 2023）：已有研究在药物发现中引入 conformal 区间；本文将其用于 11 端点多任务分类并揭示端点特异性 set 效率差异。
5. **GCN/GINE 分子表征**（Kearnes et al., 2016; Gilmer et al., 2017; Hu et al., 2020）：消息传递图网络的先驱；本文在其上叠加全局特征并行与 late concatenation，并通过消融确立简单拼接优于复杂门控融合。
6. **分子 XAI 与 toxicophore 发现**（Jiménez-Luna et al., 2020; Hooker et al., 2019; Adebayo et al., 2018）：指出 saliency map 不应作为证据单独使用；本文在此基础上引入 occlusion control + 随机对照 + consensus motif 三级验证降低虚假归因风险。

## 局限性与未来方向
- 仅使用一个 UMAP-HDBSCAN 划分，ensemble 只量化了 seed 变异性而非 split 变异性。
- 外部基准使用固定挑战 split 进行从头重训练，仅验证架构可迁移性，不验证 fitted 11-endpoint checkpoint 的零样本外推。
- Conformal 预测集在验证集上同时用于 checkpoint 选择、温度缩放与校准，故名义 α=0.05 仅为目标运行水平而非严格有限样本覆盖率保证；未评估前瞻分布漂移。
- SHAP、occlusion、motif 聚类均为模型级分析，无实验验证；部分 motifs 为 1-2 原子环境（如酚、芳胺），化学特异性有限。
- 200 维 PubChem bioactivity 字段因缓存缺失全为 0，无法评估其贡献。
- 未来方向：重复化学感知 split 与时间/外部队列评估；部署漂移下的 conformal 重新校准； motifs 的湿实验验证；引入 assay 条件元数据（细胞系、暴露时长、读数类型）构建更大规模多 Assay 模型（需评估负迁移风险）。

## 研究启发与可借鉴点
1. **泄漏控制划分的完整诊断栈**：Median Tc、novel scaffolds %、property balance、OOD utility 四项指标联合比较 Random/Butina/Scaffold/UMAP-HDBSCAN，可作为后续研究划分方案对比的标准模板。
2. **late concatenation 优于复杂融合的消融范式**：控制变量对比 GCMI/FiLM/mid-trunk concat/no-fusion，结论"简单拼接更稳"对多模态分子表征设计有参考价值。
3. **conformal-style 预测集 + 适用域四分位的联合报告**：将 singleton 比例与结构相似性分层结合，为化合物优先排序提供"可拒绝"决策支持，而非强推二分类。
4. **机内反事实遮蔽验证 XAI**：SHAP 归因→种子扩展→DBSCAN 聚类→与 20 个随机子图对比的完整流程，可直接迁移至其他小分子性质预测的可解释工作流。
5. **固定外部 benchmark 从头重训练协议**：保留 benchmark test fold 不动，在其 development fold 内新建 val/test 做 checkpoint 选择——这一协议能隔离"架构泛化"与"数据泄漏"两类问题，适合作为未来方法论文的标准外验步骤。

## 关键术语表
- **Sphere-exclusion filtering**：基于分子指纹 Tanimoto 距离的最近邻过滤，用于在划分前剔除过于相似的代表分子以降低结构泄漏。
- **UMAP-HDBSCAN partition**：先用 UMAP 降维再经 HDBSCAN 聚类，最后以簇为单位分配训练/验证/测试集的化学感知数据划分方法。
- **Late concatenation**：图特征与全局特征各自独立编码，仅在预测头部前一时刻拼接的多模态融合策略，本文消融中表现最佳。
- **Conformal-style prediction set**：以 conformal 思想在验证集上校准非一致性分位数，将预测概率映射为 singleton 或 {0,1} 不确定集合的区间预测方式。
- **Applicability domain (AD)**：衡量测试分子与训练分布结构相似性的评估框架，本文按最大 Tanimoto 相似度四分位分层分析性能变化。
- **GradientSHAP**：结合梯度与 Shapley 加性解释的归因方法，本文用于原子/键级别的分子预测贡献度量。
- **GCMI (Gated Cross-Modality Interaction)**：一种通过全局特征门控节点通道的图-全局融合模块，本文消融中表现不及简单拼接。
- **Counterfactual occlusion**：将模型认为重要的原子/键特征置零后观察预测变化，并与随机子图对照以验证归因忠实度的机内实验。

## 可复现要素
- **数据集**：来自 TDC toxicity tasks（Ames、LD50_Zhu、Tox21 labels）与已发表 hERG_Karim 数据集；Tox21 Challenge 使用 DeepTox 官方归档；TDC ADMET 使用公开 benchmarks。数据来源公共可访问。
- **代码/权重**：源码、固定 split、selected checkpoints、软件环境规范、benchmark provenance 均已开源：https://github.com/Magnushst/toxlens；PyG-Captum-SHAP 0.1.5 可从 PyPI 安装。
- **关键超参**：5 层 GINE、hidden width 256、lr=3e-4、weight decay=0.01、batch=128、hidden dropout=0.20、global-input dropout=0.25、edge-drop=0.06、max stochastic depth=0.20、final-representation dropout=0.10、cosine-restart period=45、label smoothing ε=7e-4、MC dropout 30 次、α=0.05。
- **环境**：提供 Conda environment spec 与 exact Windows package lock。
