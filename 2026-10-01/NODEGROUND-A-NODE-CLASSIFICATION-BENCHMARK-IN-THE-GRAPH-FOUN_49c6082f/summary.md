---
title: "NODEGROUND-A-NODE-CLASSIFICATION-BENCHMARK-IN-THE-GRAPH-FOUN"
source: https://arxiv.org/pdf/2609.39673v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 22:00:34"
field: "图机器学习评测基准"
keywords: ["Graph Foundation Models", "Node Classification Benchmark", "Performance-Cost Tradeoff", "GNN Evaluation", "Mean Improvability"]
innovations: ["首个同时系统性对比GFM与强监督方法并联合度量性能与全链路计算成本的节点分类基准", "提出Mean Improvability与Mean Overhead构成的性能-成本Pareto前沿分析框架", "开放run-level结果与公开排行榜，支持跨51数据集、双标签regime与多指标的可复现对比"]
benchmarks: ["NODEGROUND"]
---

# 论文速读：NODEGROUND: A NODE CLASSIFICATION BENCHMARK IN THE GRAPH FOUNDATION MODEL ERA

## 一句话总结
论文提出了 NODEGROUND，一个涵盖 51 个节点分类数据集的统一基准，系统对比了 6 个图基础模型（GFM）与 15 个监督学习方法在预测性能与计算成本上的权衡；结果表明，经过充分调优的 GNN 整体仍优于当前 GFM，仅 GraphPFN 在标签充足时具备一定竞争力，但复用预训练参数尚未带来可靠且廉价的泛化优势。

## 研究问题与动机
1. **现有 GFM 能否广泛替代逐数据集训练/调优的监督方法？** 各 GFM 的原始论文仅在少量数据集上与弱基线比较，一致性和泛化性未知。
2. **GFM 的计算成本优势是否成立？** 很多 GFM 依赖表格式 PFN 架构，推理阶段注意力复杂度随上下文长度二次增长，且部分方法仍需目标特定的预处理/搜索，但这些成本从未与监督方法完整训练+调优的时间进行系统对比。
3. **现有节点分类基准存在哪些关键缺失？** 既有基准在数据覆盖、超参搜索控制、验证选择、多指标评估、计算成本记录和结果透明度等五个维度上均不齐全（Table 1）。
4. **GFM 在不同数据集条件下是否稳定迁移？** 需要按图结构、特征维度、标签齐一性等多个属性拆分子组，检验 GFM 的迁移是否只是"少数强项"而非"整体泛化"。

## 核心贡献（创新点）
1. **首个同时覆盖 GFMs 与强监督方法的统一节点分类基准。** 收录 51 个数据集（6 大类应用），评估 21 种方法，并开放评估管线、共享切分与公开排行榜。
2. **统一的标准化评估协议：共享切分 + 双标签 regime + 严格 HPO 控制 + 验证独占选择。** 每个数据集均采用 10/10/80 与 50/25/25 两套分类别分层切分，每套 5 个固定实验索引，所有方法的验证选择与搜索预算完全对齐。
3. **联合度量预测质量与全链路计算成本。** 同时记录训练/调优/适配时间、推理时间与峰值 GPU 显存，并据此绘制性能–成本 Pareto 前沿与轨迹。
4. **引入 Mean Improvability 与 Mean Overhead 衡量性能—成本权衡。** 前者量化某配置相对最优配置的误差占比，后者以 log2 几何平均相对成本表达开销。
5. **开放可复现资源：** 发布全部 run-level 结果、共享切分代码、公开排行榜，支撑后续新方法比较。

## 方法详解
- **数据集构成：** 从 PyG、OGB、Heterophilous Graphs、HuggingFace TAG 四类来源收集 117 个候选，经 5 步筛选（任务范围→单标签→特征合法性→规模上限→去重）得到 51 个任务。任务聚焦小规模同质有向/无向图上的转导单标签节点分类，覆盖 Web（13）、社交（13）、学术（13）、电商（7）、交通（4）、金融（1）。
- **数据划分：** 10/10/80（低标签）与 50/25/25（丰富标签）两套分层切分，各 5 个固定索引，所有方法使用同一共享切分。
- **超参数优化与模型选择：** 每个监督方法均评估 default 与 tuned 两态：tuned 态从方法特定搜索空间中采样最多 200 个不同配置，每个配置独立训练并在验证集上按交叉熵/BCE 选择最终 checkpoint 和超参；GFM 仅优化其接口暴露的适配/预处理/头参数，冻结预训练 backbone。
- **评价指标：** 主指标为多分类 accuracy、二分类 AUROC，额外报告 macro-F1、BCE、正类 precision；聚合层面用 Elo 评分与 mean rank。
- **成本指标：** 工作流时间 = 训练+调优时间 + 推理时间；峰值显存 = fit/search 峰值与推理峰值中较大者；参考成本取每个数据集下最便宜配置。
- **Pareto 前沿构造：** 横轴 Mean Overhead（log2 几何平均相对成本），纵轴 Mean Improvability（越低越好），不可支配点即前沿。

## 实验与结果
- **Elo 总体排名（双 regime）：** GCNII 与 GPRGNN 在两套切分下均居前两位。GraphPFN 是最强的 GFM：10/10/80 排第 6，50/25/25 升至第 3。其余 GFM 均落后于领先调优 GNN。
- **图类型差异：** GraphPFN 在二分类任务上领先于调优 GNN，但在多分类任务上不具优势；增加标签能改善其多分类表现。
- **条件子组分析（22 组）：** GraphPFN 在二分类、中高平均度、中等 homophily 子组中占优，但在强 homophily、高维特征、高度不平衡目标和类别数 > 10 的任务上持续落后；增加标签（50/25/25）无法弥合上述短板。
- **性能–成本前沿：** GVT 与 GraphPFN 在无中间预算的端点比较中位于 Pareto 前沿，但 GraphPFN 的 Mean Overhead 约 6.4（~84× 参考成本）；纳入 1–201 配置中间轨迹后，除 50/25/25 下的 GraphPFN 外，其余 GFM 均被部分调优的 GNN 支配。
- **统计显著性（28 个全记录数据集，Holm 校正）：** 10/10/80 下 GCNII 有 11 次显著胜/0 次负；50/25/25 下 GraphPFN 有 7 次显著胜/0 次负，FAGCN/GCNII 均为 8 胜。无方法在所有成对比较中全面碾压。
- **调优效率：** 大部分方法的 test 提升集中在前 25–50 个配置后趋于饱和；10/10/80 下 50 配置预算的成本约为 default 的 ~13–28×。少标签 regime 下验证乐观（selection optimism）可达 3pp。

## 相关工作脉络
1. **Shchur et al. (2018) Pitfalls of GNN evaluation：** 提供 100 份切分与统一 HPO 框架，但未覆盖 GFMs、未测成本、未开放 run-level 结果。
2. **Hu et al. (2020) OGB：** 5 个节点任务，提供统一评测框架与公共排行榜，但 HPO 预算不对齐、单指标、无成本记录。
3. **Platonov et al. (2026) Fair GFM Evaluation：** 首次针对 4 个节点任务的 GFM 公平对比，覆盖调优与成本，但数据覆盖和切分数均远低于 NODEGROUND。
4. **You et al. (2020) GraphGym：** 18 个任务的设计空间搜索基准，强调多指标与多指标，但 HPO 预算不可比且结果非 run-level 级别。
5. **Yu et al. (2026) Evaluating Progress in GFM：** 15 个任务，覆盖 50 个切分与多指标，但调优控制与验证选择未严格对齐，成本测量不完整。
6. **Luo et al. (2024) Classic GNNs Are Strong Baselines：** 18 个任务，强调强监督基线，但与本工作的区别在于未涉及 GFM、成本指标与排行榜开放。

## 局限性与未来方向
1. **仅覆盖固定标签预算的节点分类任务**，未涉及边/图级别任务、非 IID 分布漂移（训练节点与查询节点来自不同社区/时段）。
2. **未包含异构图与多标签场景**；时间序列与有向动态图均未考虑。
3. **PFN 类 GFM 的推理成本与目标特异性预处理成本未被充分压缩**，二次注意力复杂度在大图上尤为突出。
4. **输出空间受限于 PFN 的表格基础骨干**，类别数 > 10 的任务需借助 output coding 等机制，当前不支持。
5. **未来方向：** 发展跨特征空间和标签空间更强迁移能力的 GFM；降低适配与推理阶段的计算开销；将基准扩展至 link prediction/graph classification 与非 IID 场景。

## 研究启发与可借鉴点
1. **标准 HPO 协议设计值得借鉴：** 固定搜索预算 + 验证独占选择 + 默认配置作为 trial 1 的对照思路，可作为本团队后续做方法评估的模板。
2. **Mean Improvability + Mean Overhead 的性能–成本综合度量体系** 可直接迁移到团队现有的多模型比较流程中。
3. **条件子组分析（22 个子组 × 7 个属性）** 为"方法在哪些数据条件下更强"提供了可复用的诊断范式。
4. **选择乐观（selection optimism）的定义与度量方法**（验证改进减去测试改进）可用于本团队后续的早停/验证选择策略评估。
5. **开源 run-level 记录 + 公开排行榜的机制** 可为团队构建持续评测平台提供参考。

## 关键术语表
**Graph Foundation Model (GFM)**：在大规模图/合成数据上预训练的模型，目标是通过少量适配在多个下游图任务上复用知识，避免每数据集从头训练。
**Mean Improvability**：某配置误差中能被同数据集最优配置消除的百分比，越小说明该配置距离理论上限越近。
**Mean Overhead**：以 log2 几何平均相对成本表达的开销，值 k 表示该配置成本约为最便宜参考的 $2^k$ 倍。
**Adjusted Homophily**：基于 Newman  assortativity 的修正版本，衡量节点标签沿边的对齐程度，取值可正可负。
**PFN (Prior-data Fitted Network)**：在合成数据上预训练的 Transformer 类模型，无需在目标数据集上更新参数，直接将已标注样本作为上下文进行推理。
**Elo Rating**：基于 Bradley-Terry 模型的对局式综合排名，将成对胜负转化为统一分数的聚合方法。
**Selection Optimism**：验证集上模型选择的改进量减去测试集上实际获得改进量的差值，正值表示验证指标高估了泛化收益。
**Pareto Frontier**：在性能与成本双目标下不存在其他配置同时不劣于且至少一项更优的不可支配解集合。

## 可复现要素
- **数据集：** 51 个数据集，其中绝大多数通过 PyG/OGB/HF TAG 等上游源获取；deezer 与 lastfm_asia 通过团队固定的 Hugging Face 镜像提供；所有切分与预处理代码已在 GitHub 公开。
- **代码/权重：** 评估管线、共享切分、run-level 记录与排行榜系统均在 https://github.com/nums-ai/nodeground 开源；GFM 预训练 checkpoint 使用官方发布版本（GraphPFN v1.3、RGVT checkpoint、CGT checkpoint、TabPFNv2 等）。
- **关键超参：** Adam / AdamW 优化器；学习率 $\{10^{-3}, 5\times10^{-3}, 10^{-2}\}$；权重衰减 $\{0, 5\times10^{-5}, 5\times10^{-4}\}$；隐藏维度 $\{64, 128, 256, 512\}$；dropout $\{0, 0.3, 0.5, 0.7\}$；最大 2500 轮早停 patience $\{20, 100\}$；每方法最多 201 个 HPO 配置搜索预算。
- **硬件环境：** NVIDIA H200，CUDA 12.1，cuDNN 90100，PyTorch 2.4.0，Python 3.10.20。
