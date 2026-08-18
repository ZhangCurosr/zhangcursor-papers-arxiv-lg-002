---
title: "When-Tool-Backed-Skill-Retrieval-Fails-Source-Style-Collapse"
source: https://arxiv.org/pdf/2608.16502v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:36:04"
field: "工具检索与检索路由"
keywords: ["Tool RAG", "source-style collapse", "retriever adaptation", "TF-IDF routing", "tool retrieval", "dense retrieval", "Agent capability access", "few-shot repair"]
innovations: ["发现并刻画 Tool RAG 中同语料库源风格坍塌现象", "提出 TF-IDF 质心距离作为轻量源风格不匹配检测器", "设计 ToolScout 源感知路由+少量修正的可复用修复框架"]
benchmarks: ["ToolRet", "StableToolBench"]
---

# 论文速读：When-Tool-Backed-Skill-Retrieval-Fails-Source-Style-Collapse

## 一句话总结
论文发现：即使工具语料库固定不变，针对某一查询源风格微调的密集检索器仍可能在另一源风格上发生**源风格坍塌（source-style collapse）**（如 FT-1100 在 APIGen 上覆盖率从 91.8% 暴跌至 0.7%）；为此提出 **ToolScout**——基于查询侧 TF-IDF 距离的源感知路由方法，将混合流量覆盖率从 22.3% 提升至 86.1%，并在 5 个坍塌源上用 20 条样本完成少量修正后，覆盖加权全局 top-1 代理从 1.3% 回升至 53.9%。

## 研究问题与动机
- **Agent 检索门的重要性**：现代 Agent 依赖从海量工具/API 池中检索候选能力，检索层必须在重排序、规划或执行前成功兜出候选集；若 gold tool 未进入候选池，下游所有阶段都无法恢复。
- **同语料库下的源风格坍塌**：在 ToolRet 基准上，以 ToolBench 风格 1100 条查询微调的 BGE-M3 在匹配源（91.8% 覆盖率）上表现良好，但在同语料库的 APIGen（0.7%）、ToolACE（0.0%）、UltraTool（8.3%）等源上严重崩溃，揭示"同语料库源风格偏移"这一失败模式。
- **已有信号无法解释**：APIGen 的查询-黄金工具词汇重叠（0.427）反而高于训练切片（0.195），Lexical overlap、query length、semantic centroid 等代理均无法有效预测坍塌。
- **可迁移性挑战**：现实 Agent 部署暴露的往往是多来源、多风格、成百上千异构工具池（ToolRet-4996 含 19 种查询生成源），单一源特化检索器存在系统性失效风险。

## 核心贡献（创新点）
1. **首次系统刻画 Tool RAG 中的源风格坍塌**：用结构化工具/API 作为可测量可执行技能卡，揭示密集检索器可在更高词汇重叠的目标源上崩溃，单纯关键词访问或查询长度均无法解释性能下降。
2. **建立查询侧 TF-IDF 距离作为源风格不匹配的强检测器**：跨 19 种源-查询配置，TF-IDF 质心距离是最佳轻量探测器（leave-one-config-out 准确率 78.9%、F1=0.88），显著优于语义距离（73.7%、0.85）、查询长度差（68.4%、0.80）和多步率差（47.4%、0.58）；且工具侧信号相关但更弱（r=-0.77 vs. r=-0.85）。
3. **提出 ToolScout 源感知路由策略**：用 TF-IDF 距离作为路由守卫，将可能失配的流量切换至聚合训练检查点；保留特定源检索器在兼容流量上的优势（FT-1100 在 toolbench 源仍可达 91.8% vs. 聚合 87.3%）。
4. **验证少量修正即可修复坍塌源**：在 5 个坍塌源上用 20 条匹配样本对 FT-1100 进行续训，覆盖加权全局 top-1 代理从 1.3% 提升至 53.9%，接近 59.0% 的聚合训练参考。
5. **排除 API Schema 格式单一解释**：将工具重渲染为可执行技能卡（skill card）后，坍塌模式、检测器和切换策略依然成立，证明失败根因不只是原始 API Schema 格式。

## 方法详解
- **检索管道**：采用 BAAI/BGE-M3 作为密集主干编码器，用 MultipleNegativesRankingLoss（MNRL，类 InfoNCE）+ 深度 20 密集池内挖掘的"检索混淆硬负样本"对 query-tool 对进行微调。
- **直接候选生成**：$C(q) = \text{top}_k(r(q, T))$，其中 $r$ 为词法或密集检索器，$k$ 为候选预算（主实验 depth=20）。
- **分阶段检索扩展**：将查询拆为子查询 $\{q_1, \dots, q_m\}$，分别打分后聚合：$C_{\text{stage}}(q) = \text{top}_k(\{t \in T : s(t) = \max_j r(q_j, t)\})$，再经reranker。
- **源感知路由公式**：计算查询批次 $\mathcal{Q}$ 与训练源 $\mathcal{R}$ 的 TF-IDF 质心距离：
  $$d_{\text{tfidf}}(\mathcal{Q}, \mathcal{R}) = 1 - \cos(\mu_{\text{tfidf}}(\mathcal{Q}), \mu_{\text{tfidf}}(\mathcal{R}))$$
  安全/不安全阈值在 leave-one-config-out 上校准后固定；落入安全区间走匹配特化检查点，落入不安全区间切至聚合训练检查点。
- **修复机制**：获得少量匹配监督后，用同样 MNRL 协议对失配检查点做短续训（5/10/20/50-shot 扫表）；覆盖加权全局 top-1 代理定义为 $G = C_m \cdot P@1|_{\text{cov}}$，即被覆盖查询的 top-1 准确率乘以覆盖率，未覆盖查询计为失败。
- **技能卡渲染控制**：将工具规范化为 skill card（字段：capability、when-to-use、inputs、output/effect、metadata），保持工具身份/标签/切分/检索器/路由策略不变，用于排除格式干扰。

## 实验与结果
- **数据集**：ToolRet（主基准，含 1100-query 受控 web 子集、4996-query 混合 19 源聚合、以及 44,453 工具/7,726 查询的全量 build）；StableToolBench（高覆盖率效率评估基准）。
- **基线**：BM25、bge-small-en-v1.5、BGE-M3、Qwen3-Emb-4B、LCO-Omni（多模态 frozen encoder + projection adapter）。
- **主要结果**：
  - BGE-M3 离石 Fine-tune 在 ToolRet-1100 上覆盖率 91.8%（vs. OTS 80.8%，提升 +11pp）；但在 4996 混合流上仅 22.3%（vs. OTS 73.7%）。
  - ToolScout 路由在 4996 混合流上将覆盖率从 22.3% 提升至 **86.1%**（聚合检查点 85.2%，几乎持平）。
  - 5 个坍塌源上 20-shot 修复后，覆盖加权全局 top-1 代理从 1.3% 提升至 **53.9%**（聚合参考 59.0%）。
  - APIGen：FT-1100 0.7% → 20-shot 修复 93.2% → 聚合 96.6%。
  - 词汇重叠 paradox：APIGen 词汇重叠 0.427 最高但 FT-1100 崩溃至 0.7%；BM25 在 APIGen 上达 75.0%，但 BM25+FT-1100 融合在 3014 条坍塌查询上仍落后于纯 BM25（54.2% vs. 59.4%），故路由优于融合。
  - Skill-card 渲染控制：FT-1100 在 4996 流上仍崩溃至 21.7%，TF-IDF 切换恢复至 85.2%，排除格式干扰。
  - 全量 44,453 工具/7,726 查询测试：fine-tuning 提升覆盖率 +23.5pp，全局代理从 23.7% 升至 43.4%。
  - 层级路由（StableToolBench）：自动聚类 k=5 保留 89.2% 覆盖率的同时减少 87.4% 候选上下文；本地 Qwen-14B 执行验证：自动方案 32.8% 略超官方暴露 31.7%。

## 相关工作脉络
- **Toolformer / Gorilla**：确立通用工具使用设置；本文在此基础上聚焦 retrieval gate 的源风格可靠性，而非工具调用本身。
- **Tool2Vec**（Moon et al., 2024）：研究工具表示；本文与工具表示独立，聚焦 retrieval routing。
- **ToolRerank**（Zheng et al., 2024）：在已有候选池上做层级重排序；本文解决上游候选生成崩溃问题，定位在 reranking 之前。
- **AnyTool / DeepAgent**：完整 Agent 栈内的层级检索；本文隔离路由层在共享检索/暴露协议下的行为。
- **GPL / InPars / Promptagator / AugTriever**：通过伪标签或合成查询适配检索器；本文 few-shot 续训与之精神相近，但诊断设置保持能力语料固定、仅切换源切片。
- **Skill Retrieval Augmentation (SRA-Bench, Su et al., 2026)**：研究检索后 skill 整合问题；本文聚焦更早的 retrieval gate 可靠性（"检索不到则后续皆无效"）。

## 局限性与未来方向
- 实验使用共享检索/暴露/下游选择协议，不能复现各 Agent 栈的原始 prompt 和模型选择，结论局限于路由层机制。
- 最详细分析集中在 1100-query 受控 web 子集；全量 build 上观察到大趋势但错误标注较少，需要独立多源工具语料验证。
- 技能卡渲染实验仅限于结构化工具/API；面向程序化技能、非工具资源、end-task skill 整合的更广泛能力库仍需探索。
- Few-shot 修复效果依赖监督可获得性，真实部署中故障 telemetry 的积累速率未给出。
- 层级路由在 ToolRet 上自动聚类弱于人工类别，表明 category metadata 质量是关键瓶颈。

## 研究启发与可借鉴点
- **TF-IDF 质心距离作为便宜检测器**可用于其他检索场景的风格迁移诊断：当语义空间难以捕捉分布漂移时，词法统计信号可能更稳健（尤其在查询-工具对联合 shift 场景）。
- **20-shot 修复即可大幅回升性能**：说明坍塌源修复所需标注样本量远低于预期，对在线 Agent 系统（用失败 telemetry 驱动持续微调）具有实操价值。
- **覆盖加权全局代理 G = C_m · P@1|_{cov}** 作为统一评估指标，可同时反映候选生成质量与 reranker 精度，适合作为 Tool RAG 的系统级 KPI。
- **源感知路由可迁移至其他检索门场景**：如 Skill RAG、文档检索中的多源混合查询流，在维护特化检索器优势的同时避免灾难性转移。
- **BM25 + 密集融合的陷阱**：在严重失配场景下朴素融合可能劣于纯 BM25，提示在实际部署中应优先考虑路由而非简单融合。

## 关键术语表
- **Source-style collapse**：在工具语料库固定不变的情况下，针对某一查询源风格微调的检索器在其他源风格上检索覆盖率严重下降的失败模式。
- **Tool RAG**：将结构化工具/API 渲染为可执行技能卡并进行检索的 Agentic 检索增强范式。
- **Coverage-weighted global top-1 proxy (G)**：全局 top-1 代理指标，等于候选覆盖率 $C_m$ 乘以被覆盖查询的 top-1 准确率，未覆盖查询计为失败。
- **TF-IDF 质心距离**：查询批次与训练源在 TF-IDF 空间中质心向量的余弦距离，用于检测源风格不匹配。
- **MultipleNegativesRankingLoss (MNRL)**：SentenceTransformers 实现的批次内对比学习目标，类 InfoNCE，结合硬负样本挖掘。
- **Skill card**：将工具规范化为包含 capability、when-to-use、inputs、output/effect、metadata 等字段的统一技能表示。
- **Split-and-merge（分阶段检索）**：将查询拆解为子查询分别检索后再聚合候选池，用于处理组合型多意图尾部。
- **Hard negative mining**：从当前 top-20 密集池中挖掘对检索具有混淆性的困难负样本，提升微调效果。

## 可复现要素
- **数据集**：ToolRet（Shi et al., 2025）、StableToolBench（Guo et al., 2024）——均为公开基准；具体切分见附录。
- **代码/权重**：论文未明确声明开源仓库链接；主干编码器 BGE-M3 为公开模型。
- **关键超参**：候选深度 depth=20；MNRL 损失；TF-IDF 质心距离阈值在 leave-one-config-out 上校准；few-shot 修复用 5/10/20/50 -shot 扫表；reranker 为线性模型（部分实验用 MLP 和 cross-encoder）。
- **硬件/环境**：TF-IDF 质心拟合在 CPU 上约 0.29 秒，中位数 388.7ms（含 transform-and-centroid），p95=452.5ms。
