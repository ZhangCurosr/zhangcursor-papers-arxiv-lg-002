---
title: "When-Tool-Backed-Skill-Retrieval-Fails-Source-Style-Collapse"
source: https://arxiv.org/pdf/2608.16502v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 04:07:47"
field: "工具检索与Agent能力获取"
keywords: ["tool retrieval", "source-style collapse", "retriever adaptation", "skill-card rendering", "TF-IDF routing", "few-shot repair", "executable capability"]
innovations: ["首次系统诊断同库跨源风格迁移下的检索器坍塌现象并提出'源风格坍塌'概念", "基于查询端TF-IDF质心距离的廉价路由守卫机制（78.9% LOO准确率）", "ToolScout双阶段路由策略：检测→切换aggregate检查点→few-shot继续微调修复"]
benchmarks: ["ToolRet", "StableToolBench"]
---

# 论文速读：When-Tool-Backed-Skill-Retrieval-Fails-Source-Style-Collapse

## 一句话总结
本文发现：在工具检索（Tool RAG）中，当工具库固定不变时，针对某一数据源风格微调的检索器会在另一数据源风格的查询上出现"源风格坍塌"（source-style collapse）——即使目标源与黄金工具的词汇重叠更高，召回率仍可能降至接近零。作者据此提出 ToolScout，一种基于查询端 TF-IDF 指纹的路由守卫机制，配合少量匹配样本的继续微调，可将混合源流上的覆盖率从 22.3% 提升至 86.1%。

## 研究问题与动机
- **Agent 部署依赖可靠的能力检索门控**：现代 Agent 面对数百至数千个异构工具/API，候选生成层决定哪些工具进入下游 reranker/planner 的视野；若黄金工具从未进入候选池，后续所有阶段都无法恢复。
- **同一工具库下出现源风格坍塌**：在 ToolRet 基准上，仅在 1100 条 ToolBench 风格查询上微调的 BGE-M3（FT-1100）在跨源迁移时急剧下降——APIGen 覆盖率仅 0.7%，ToolACE 0.0%，UltraTool 8.3%；即便 APIGen 的词汇重叠高于训练源，仍无法避免崩溃。
- **现有诊断信号不足**：词汇重叠、查询长度、语义质心距离等均不能充分解释坍塌原因；TF-IDF 指纹信号显著更强（加权 r=-0.85 vs 语义 r=-0.58）。
- **Skill-card 重渲染实验排除格式假说**：将工具以可执行技能卡形式重渲染后，坍塌现象与路由策略仍然复现，说明问题不只源于原始 API schema 格式差异。

## 核心贡献（创新点）
1. **首次形式化并命名"源风格坍塌"**：在固定工具库约束下，证明同库但不同上游查询生成源的迁移可导致检索器覆盖率断崖式下跌，且简单关键词或查询长度无法解释该现象。
2. **提出 TF-IDF 质心距离作为轻量级源风格不匹配检测器**：在 19 种源-查询配置上，leave-one-config-out 准确率 78.9%、F1=0.88，显著优于语义质心（73.7%）、查询长度差（68.4%）和多步率差（47.4%）。
3. **验证 ToolScout 路由策略的有效性**：在 4996 查询混合流上，TF-IDF 路由将覆盖率从 22.3% 提升至 86.1%（接近 aggregate 训练的 85.2%）；对五个坍塌源各 20 条匹配样本的 few-shot 修复将覆盖率加权全局 top-1 代理从 1.3% 提升至 53.9%。
4. **通过 skill-card 重渲染与 MCP-lite 规范化实验证明坍塌非格式伪影**：两种表示变换均保留坍塌模式与 TF-IDF 检测能力，排除了原始 schema 格式的单一因果解释。

## 方法详解
- **基础框架**：将每个工具/API 渲染为结构化 skill card（含 capability、when-to-use、inputs、output/effect 等字段），使用 BAAI/BGE-M3 作为 dense 主干，配合 MNRL（MultipleNegativesRankingLoss）对比学习目标微调检索器，并在 batch 内从深度-20 密集池挖掘 retrieval-confusing hard negatives。
- **TF-IDF 兼容性评分**：对入站查询集 Q 和训练源切片 R，计算 TF-IDF 空间质心距离：
  $d_{\text{tfidf}}(\mathcal{Q}, \mathcal{R}) = 1 - \cos(\mu_{\text{tfidf}}(\mathcal{Q}), \mu_{\text{tfidf}}(\mathcal{R}))$，
  阈值在 leave-one-config-out 划分上一次性校准后固定使用。
- **路由策略**：距离落在已知安全带内 → 直接使用匹配源微调的 FT-1100 检索器；落在不安全带 → 切换至 aggregate-trained 宽泛检查点；当后续积累少量匹配监督样本时，触发继续微调（few-shot continuation fine-tuning）修复剩余差距。
- **扩展机制**：
  - **Staged retrieval**：将查询分解为子查询分别检索，聚合得分后取 top-k 工具进入 reranker，主要缓解 compositional multi-intent 尾部失败。
  - **Hierarchy routing**：在 StableToolBench 上使用 Mini-BatchKMeans 聚类工具嵌入构建自动类别taxonomy，以 k=5 时保留 89.2% 覆盖率的同时削减 87.4% 候选上下文。
- **评估指标**：定义覆盖率加权全局 top-1 代理 $G = C_m \cdot P@1|_{\text{cov}}$，将未被 top-m 候选覆盖的查询直接计为失败，用于衡量候选生成瓶颈的整体影响。

## 实验与结果
- **数据集**：ToolRet（1100 查询标准化 web 子集 / 4996 查询 19 源混合聚合 / 7726 查询 44453 工具全量构建）；StableToolBench（46,453 工具上下文压缩验证效率边界）。
- **基线**：BM25、bge-small-en-v1.5、BGE-M3 off-the-shelf、Qwen3-Emb-4B、LCO-Embedding-Omni-3B（多模态 Frozen encoder + 投影适配器）。
- **核心结果**：
  - FT-1100 在 ToolRet-1100 上覆盖率 91.8%，但在 ToolRet-4996 混合流上暴跌至 22.3%；aggregate 训练（FT-4996）在全部源上稳定达到 85.2%。
  - Qwen3-Emb-4B 虽在各 ToolRet 设置上缩小差距，但仍低于同规模源适配的 BGE-M3。
  - TF-IDF 路由在 4996 查询混合流上覆盖率提升至 86.1%；对五个坍塌源 20-shot 修复后全局 top-1 代理从 1.3% 升至 53.9%（50-shot 进一步逼近 aggregate 天花板）。
  - Skill-card 渲染：FT-1100→4996 覆盖率 21.7%（原 22.3%），aggregate 为 84.0%，TF-IDF switch 恢复至 85.2%，证明非格式伪影。
  - MCP-lite 规范化后 FT-1100→4996 覆盖率 25.1%（原 22.3%），坍塌模式依然存在。
  - Bootstrapping 验证：1100 slice 直接微调增益 +19.0pp [95% CI: 10.8, 27.2]，4996 slice +19.3pp [15.4, 23.1]，p<0.001。

## 相关工作脉络
- **工具检索基准**：ToolRet（Shi et al., 2025）、ToolBench/ToolLLM（Qin et al., 2024）、StableToolBench（Guo et al., 2024）、API-Bank（Li et al., 2023b）、ToolHop（Ye et al., 2025）——本文在其之上引入同库跨源风格迁移的诊断框架。
- **工具表征与检索**：Tool2Vec（Moon et al., 2024）研究工具向量表征；ToolRerank（Zheng et al., 2024）关注候选池确定后的层级 reranking——本文强调在候选生成层本身解决可靠性问题，定位更早于这些工作。
- **Agent 栈与编排**：AnyTool（Du et al., 2024）、DeepAgent（Li et al., 2026b）、Toolformer（Schick et al., 2023）、Gorilla（Patil et al., 2024）——本文以共享检索-暴露协议隔离上游路由层，便于精确诊断，不与完整 agent 栈竞争。
- **密集检索域适应**：GPL（Wang et al., 2022）、InPars（Bonifacio et al., 2022）、Promptagator（Dai et al., 2023）、AugTriever（Meng et al., 2022）——本文 few-shot 修复阶段在精神上与其相近，但核心区别在于固定工具库下的源风格位移诊断而非新目标域数据生成。
- **Skill Retrieval Augmentation（SRA）**：Su et al.（2026）研究检索到技能后的 incorporation 问题——本文聚焦更早的 retrieval gate 可靠性，二者互补。

## 局限性与未来方向
- 核心详细分析集中于 1100 查询 web 子集，更大规模（7726 查询）虽有正向复现但错误标注较少，需在独立生成的多源工具语料上进一步验证。
- Skill-card 渲染仅覆盖结构化工具/API，程序型技能（procedural skills）、非工具资源及端到端技能 incorporation 不在研究范围内。
- Few-shot 修复依赖匹配监督的可用性；实际部署中这些样本的积累速率和来源（失败查询遥测 vs. 工具提供商 onboarding）有待未来工作研究。
- 层级路由在 StableToolBench 上效果最强，但手动类别仍优于自动聚类；更广泛的类别结构化工具语料测试尚待开展。
- 多源混合训练中的简单 reweighting 控制（balanced / temperature-reweighted）表明均匀平衡会损害高支持源的分布对齐，路由策略在当前阶段比联合训练更稳健。

## 研究启发与可借鉴点
- **TF-IDF 质心距离作为廉价路由守卫**：可在不增加推理成本的前提下预判检索器是否可能失配，对多源工具库的 Agent 部署具有直接工程参考价值。
- **Few-shot 修复路径的可迁移性**：在积累少量失败查询日志后对 specialist 检索器进行继续微调，以 20 样本即可将覆盖率代理提升约 50pp，适合在线适配场景。
- **Skill-card 重渲染作为表示控制实验**：排除"API schema 格式差异"这一混淆变量的实验设计思路，可推广至其他 RAG/检索故障诊断场景。
- **与团队方向的结合机会**：若团队涉及多源工具编排或动态能力扩展，可借鉴 ToolScout 的双阶段路由（检测→切换→修复）框架，或在 skill-card 标准化上进一步探索 MCP-like 规范对检索鲁棒性的影响。

## 关键术语表
**Source-style collapse**：在同一固定工具库上，检索器对某一数据源风格微调后，在另一数据源风格的查询上出现覆盖率断崖式下跌的失败模式。
**ToolScout**：一种源感知路由方法，利用查询端 TF-IDF 质心距离作为兼容性检测器，将不匹配流量路由至 aggregate-trained 检查点，并在有匹配监督时触发继续微调。
**Coverage-weighted global top-1 proxy（G）**：将未覆盖查询计为失败的覆盖率加权 top-1 代理指标，用于统一评估候选生成与排序的综合效果。
**TF-IDF centroid distance**：以 TF-IDF 向量空间的批次质心余弦距离，作为衡量查询集与训练源风格差异的轻量级检测信号。
**Skill-card rendering**：将原始工具 API schema 重渲染为结构化技能卡（含 capability、when-to-use、inputs、output/effect 等字段）的表示变换，用于排除格式差异导致的伪影。
**MultipleNegativesRankingLoss（MNRL）**：SentenceTransformers 中实现的 in-batch 对比学习目标（与信息损失 InfoNCE 相近），用于检索器微调。
**Staged retrieval**：将复杂查询分解为子查询分别检索后聚合得分的扩展检索策略，主要针对 compositional multi-intent 尾部失败。
**Retrieval-confusing hard negatives**：从深度-20 密集池中挖掘的、语义上与查询高度相似但非黄金工具的困难负样本，对工具-查询对齐效果显著优于随机或同类别负样本。

## 可复现要素
- **数据集**：ToolRet（Shi et al., 2025）、StableToolBench（Guo et al., 2024）；论文声明 ToolRet 已公开。
- **代码/权重**：主干模型 BGE-M3（HuggingFace 开源）、Qwen3-Emb-4B（HuggingFace 开源）、LCO-Embedding-Omni-3B-2605（HuggingFace model card）；fine-tuning 使用 SentenceTransformers 实现 MNRL。
- **关键超参**：候选深度 k=20（main dense runs）；MNRL 训练；hard negatives 来自 depth-20 密集池；TF-IDF 质心距离阈值在 leave-one-config-out 上校准；few-shot 修复使用 5/10/20/50 样本扫描。
- **评估协议**：确定性 query-hash train/val/test 分割（ToolRet: 635/126/129；StableToolBench: 551/94/108）。
