---
title: "Large-Language-Models-for-Automated-Cross-Domain-Machine-Lea"
source: https://arxiv.org/pdf/2609.35335v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:51:42"
field: "AutoML与LLM交叉"
keywords: ["task type identification", "AutoML", "large language models", "cross-domain", "time series classification", "tabular data", "prompt engineering"]
innovations: ["首个面向ML任务类型识别的跨域开放基准（625数据集）", "仅需目标特征即可推断数据域与子任务的LLM框架", "验证目标特异性统计量可实现接近完整信息的任务识别"]
benchmarks: ["OpenML", "UCI ML Repository", "Kaggle", "Time Series Classification Archive"]
---

# 论文速读：Large-Language-Models-for-Automated-Cross-Domain-Machine-Learning-Task-Type-Identification

## 一句话总结
本文提出了一种基于大语言模型（LLM）的自动化机器学习任务类型识别系统，仅依赖用户提供的目标特征（target feature）及数据集元信息，即可推断数据域（表格 vs. 时间序列）与下游预测任务（二分类/多分类/回归），并发布了包含625个公开数据集的基准测试。

## 研究问题与动机
1. **现有AutoML系统假设任务类型已知的局限**：大多数AutoML框架（如AutoGluon、H2O、Auto-Sklearn）将任务类型（分类/回归/时间序列预测）作为预设输入，而实践中工业场景数据集异构、文档不全，需由数据科学家手动标注，效率低且易出错。
2. **跨数据模态的领域识别缺失**：既有启发式方法主要针对表格数据的任务识别，未解决跨表格与时间序列等不同数据模态的领域识别（domain identification）问题。
3. **LLM处理结构化数据的潜力未被系统评估**：近期研究表明LLM在结构化数据处理、语义模式提取和零样本/少样本分类方面表现强劲，但尚无工作系统研究其从数据集级信息中同时推断数据域和预测任务的能力。
4. **资源受限部署场景的需求**：工业界常需本地化部署模型，但小规模LLM在此任务上的表现缺乏充分评估，准确性与可部署性之间的权衡不清。

## 核心贡献（创新点）
1. **首个面向ML任务类型识别的开放基准数据集**：发布包含625个公开表格与时序数据集的基准，涵盖二分类、多分类、回归/预测四种子任务，支持数据域与下游任务的联合评估——区别于以往仅关注表格任务识别的研究。
2. **LLM驱动的两级任务识别框架**：提出仅需用户提供目标特征即可推断"数据域（Tabular/Time Series）+ 子任务（binary/multiclass/regression）"的Prompt框架，结合目标统计量与可选文本描述——区别于已有LLM-Assisted AutoML系统（如AutoML-GPT、MLAgentBench）需预设任务描述的设定。
3. **系统性三场景评估**：在表格单一域、跨域混合、小模型本地部署三个设置下对比LLM方法与AutoML启发式基线——填补了跨模态任务识别与资源受限部署的实证空白。
4. **目标特异性统计量（ST）的有效性验证**：证明仅凭目标变量的分布统计（计数、均值、标准差等）即可实现接近完整信息的任务识别性能——为减少Prompt复杂度提供依据。

## 方法详解
1. **输入结构**：系统接收三类输入（图2）：①数据集本身；②用户明确指定的目标特征（target feature）；③可选的文本型数据集描述（dataset description, DD）。
2. **目标特异性统计量提取（ST）**：从目标变量提取数值型统计特征——值计数（value counts）、唯一值数量（unique values）、均值、标准差、最小/最大值；对类文本目标额外提取词数、停用词数、字符数及其标准差。
3. **数据集序列化策略**：将原始数据集转换为紧凑的DataFrame矩阵表示（保留行/列结构与原始特征名），目标特征始终显式标记。采用分组行采样（group-wise row sampling）策略——按原始位置选取连续样本块再重排，以兼顾短程模式保留与上下文窗口约束。
4. **Prompt设计**：
   - System Prompt定义输出格式为二元组 `(数据域, 子任务)`，数据域取`Tabular`或`Time_Series`，子任务取`binary`/`multiclass`/`regression`。
   - 任务识别规则：若目标变量连续→regression；若分类变量且唯一值为2→binary，否则→multiclass；强调整数型目标需结合语义判断而非仅依赖数据类型。
   - Few-shot设置下在System Prompt中注入各子任务的标注示例。
5. **评估协议**：数据集按20%训练/20%验证/60%测试分层划分；训练集用于提示工程与序列化策略的手动调试，验证集用于配置选择，测试集评估最终性能；采用1,000次Bootstrap重采样报告均值与标准差；不使用交叉验证以避免手动调优过程在折叠间难以一致复现。

## 实验与结果
**数据集与基线**：
- 基准：625个数据集（326表格 + 299时序）；来源OpenML、UCI、Kaggle、TS Classification Archive。
- 基线AutoML：AutoGluon（F1 macro 0.93）、NaiveAutoML（0.80）、H2O（0.59），均为zero-shot结果。

**Experiment 1：表格单一域识别（Table 2）**
- 最强结果：**Qwen3-14B (DD+ST, few-shot) 达 0.98 F1 macro**，超越AutoGluon 0.93约5个百分点；零样本与推理模式影响极小。
- ST-only配置（无文本描述）仍可降至0.96，表明目标统计量本身蕴含强信号。
- 混淆分析（图4）：AutoGluon倾向混淆回归与多分类，而Qwen3-14B误判极少。

**Experiment 2：跨域识别（Table 3）**
- 最强结果：**GPT-5.3 + reasoning + DD+ST + few-shot 达 0.90 F1 macro**。
- 跨域难度明显高于单一域：主要混淆发生在"相同任务跨域"（如表格回归 vs. 时序回归）。
- ST-only在跨域下显著退化（Qwen2.5-14B zero-shot仅0.52），说明小模型更依赖文本语义上下文。

**Experiment 3：小模型本地部署（Table 4）**
- 最强小模型：**Qwen3-4B-Instruct (DD+ST, few-shot) 达 0.75 F1 macro / 0.74 balanced accuracy**。
- Qwen3-4B-Thinking变体表现大幅下滑（few-shot仅0.62），提示推理型小模型在本任务上鲁棒性不足。
- ST-only时小模型性能骤降（从0.75至0.50），再次印证小模型对语义上下文的依赖。
- 延迟成本（附录A.4）：AutoGluon 0.0004s/提示；Qwen3-4B-Instruct ~8.4s/提示；GPT-5.3通过API计费（few-shot $3.38总成本）。

## 相关工作脉络
1. **AutoML启发式任务识别**：AutoGluon-Tabular [14]、NaiveAutoML [39]、H2O AutoML [33]均以规则/启发式推断表格任务类型——本文工作首次将其扩展至跨域场景并与LLM对比。
2. **LLM辅助AutoML系统**：AutoML-GPT [55]、AutoML-Agent [48]、MLAgentBench [23]、MLCopilot [54]均假设任务已给定或由用户描述；本文将任务识别作为独立前置步骤，仅需目标特征即可推断。
3. **语义特征生成**：CAAFE [22]利用LLM从数据集描述生成语义特征，但其描述通常已隐含预测问题定义——本文明确区分"描述性元信息"与"任务类型标签"，后者为人工标注且未随原始数据集公开，以降低数据污染风险。
4. **表格理解与序列化**：TURL [11]、TaPas [21]、TABBIE [25]、TableFormer [42]、TUTA [53]聚焦表格的结构化理解与预训练；本文的DataFrame序列化策略与之互补，专为任务识别Prompt设计。
5. **LLM分类与结构化数据推理**：TabLLM [20]、Struct-X [47]验证LLM在结构化分类中的能力；本文聚焦"任务元识别"这一上游任务，而非直接预测。
6. **数据污染检测**：LM Contamination Index [7, 28]被用于基准构建阶段过滤已知污染数据集——这是评估预训练LLM时保障实验可信度的关键措施。

## 局限性与未来方向
1. **数据集偏向公开基准**：基准主要来自OpenML、UCI等 curated 数据集，未能充分反映工业场景中常见的噪声、不一致性与复杂数据结构。
2. **数据污染风险未完全排除**：尽管过滤了LM Contamination Index中列出的数据集，但封闭式模型（如GPT-5.3）的训练语料未公开，仍可能存在通过数据集内容或元数据的间接污染；论文建议后续开展记忆测试等专门分析。
3. **仅评估任务识别指标**：未量化任务类型误判对下游流水线（预处理、模型选择、训练、评估）的实际影响——误分类如何传播并损害最终模型性能尚不清楚。
4. **目标特征需人工指定**：当前系统假设用户已知目标变量；自动目标识别是重要延伸方向，将显著降低用户使用门槛。
5. **推理型小模型表现不佳**：Qwen3-4B-Thinking在跨域设置下大幅落后于Instruct变体，提示"思考"机制在本任务上可能引入不必要噪声或偏离判别模式。

## 研究启发与可借鉴点
1. **目标特异性统计量可作为高效输入**：仅用目标变量的简单统计（计数、均值、标准差等）即可实现接近完整信息的任务识别，提示在Prompt设计中应优先提取此类紧凑信号以降低Token开销。
2. **分组行采样保留结构信息的策略**：按原始位置选取连续样本块再重排的序列化方式，既能保留时序/分组结构的短程模式，又满足上下文窗口限制——可迁移至其他结构化数据理解任务。
3. **推理模式的双刃剑效应**：开启reasoning在强模型（GPT-5.3）上有益，但在小模型（Qwen3-4B-Thinking）上反而损害性能——提示未来在小规模模型上应谨慎启用chain-of-thought变体。
4. **Bootstrap重采样替代交叉验证的评估范式**：在Prompt工程类研究中，由于调优涉及大量手动设计决策，固定划分+Bootstrap估计比传统K折CV更具可行性与一致性——为类似研究提供方法论参考。
5. **与团队方向的结合机会**：可将本工作的任务识别模块作为AutoML流水线的前置组件，结合团队在特征工程或模型选择方面的积累，构建端到端任务感知pipeline；亦可探索自动目标识别作为下一阶段扩展。

## 关键术语表
- **Task Type Identification（任务类型识别）**：从数据集信息中推断数据所属的机器学习任务类别（二分类/多分类/回归/时间序列预测）。
- **Data Domain（数据域）**：数据集的结构性模态分类，本文区分为Tabular（独立样本表格数据）与Time Series（有序序列数据）。
- **Target-specific Statistics（ST，目标特异性统计量）**：从目标变量提取的分布特征（计数、均值、标准差、极值等），用于辅助任务推断的紧凑统计信号。
- **DD+ST（Dataset Description + Statistics）**：同时使用文本型数据集描述与目标统计量的Prompt配置，本文实验中最为稳定的设置。
- **Group-wise Row Sampling（分组行采样）**：从数据集中按原始位置选取连续样本块后再重排的序列化策略，兼顾结构保留与上下文窗口约束。
- **F1 Macro（宏平均F1）**：对各类别分别计算F1后求算术平均的指标，不受类别不平衡影响，本文主要评估度量。
- **LM Contamination Index（语言模型污染指数）**：收录已知可能被预训练语料覆盖的数据集列表，本文用于过滤潜在污染样本以保障评估可信度。

## 可复现要素
- **数据集**：625个数据集通过Zenodo公开发布（https://zenodo.org/records/21649607），包含原始数据文件与结构化元信息表。
- **代码**：项目仓库通过GitHub公开发布（https://github.com/ptsialis/Machine-Learning-Task-Type-Identification），含完整实验代码、依赖规范与运行脚本。
- **模型**：GPT-5.3为商业API模型；Qwen系列（Qwen3-14B、Qwen2.5-14B-Instruct、Qwen3-4B-Instruct-2507-FP8、Qwen3-4B-Thinking-2507-FP8）为本地部署开源模型。
- **关键超参**：训练/验证/测试 split = 20%/20%/60%（分层）；Bootstrap重采样次数=1,000；GPT-5.3 reasoning设为high；Qwen推理模式按模型变体自动启用/关闭。
- **污染过滤**：使用LM Contamination Index（https://hitz-zentroa.github.io/lm-contamination/）排除已知污染数据集。
