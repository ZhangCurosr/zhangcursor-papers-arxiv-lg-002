---
title: "Large-Language-Models-for-Automated-Cross-Domain-Machine-Lea"
source: https://arxiv.org/pdf/2609.35335v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:51:30"
field: "AutoML与结构化数据理解"
keywords: ["Large Language Models", "AutoML", "Task Type Identification", "Cross-Domain Evaluation", "Tabular Data", "Time Series", "Prompt Engineering", "Benchmark Dataset"]
innovations: ["首个面向ML任务类型识别的公开Benchmark（625数据集，两层分类体系）", "仅依赖目标特征的LLM任务推断框架，无需显式任务先验", "系统揭示目标统计量强信号价值及跨域场景中小模型性能-部署权衡"]
benchmarks: ["OpenML", "UCI Machine Learning Repository", "Kaggle", "Time Series Classification Archive"]
---

# 论文速读：Large-Language-Models-for-Automated-Cross-Domain-Machine-Learning-Task-Type-Identification

## 一句话总结
本文提出了一种基于大语言模型（LLM）的自动化机器学习任务类型识别方法，仅需用户提供目标特征即可从数据集级别推断数据域（表格/时序）及预测任务（分类/回归），并发布了包含625个数据集的公开基准（Benchmark）。实验表明，LLM方法在表格任务上显著优于AutoGluon等启发式AutoML基线，且在跨域场景和小型本地部署模型中仍保持有效性能。

## 研究问题与动机
- **任务类型识别依赖人工**：构建有效ML流水线的前提是正确识别数据域和预测任务（分类/回归/时序预测），但工业实践中该步骤通常由数据科学家手动指定，对非专家用户构成门槛。
- **现有AutoML系统未显式解决该问题**：多数AutoML框架假设任务已预定义，或仅通过启发式/规则方法处理表格数据的下游任务识别，缺乏跨模态（表格vs时序）的数据域识别能力。
- **LLM处理结构化数据的潜力未系统验证**：尽管LLM在结构化数据理解和分类任务中表现出色，但其在"仅凭目标特征推断完整任务类型"这一设定下的能力尚未被系统研究。
- **工业场景的异质性与资源约束**：实际应用中数据集来源异构、文档不全，且需兼顾云端大模型与本地小模型的部署可行性。

## 核心贡献（创新点）
1. **首个面向ML任务类型识别的公开Benchmark**：发布包含625个公共表格与时序数据集的基准，涵盖两层分类体系（数据域×预测任务），填补了该领域无系统评测的空白。
   - *区别*：现有AutoML基准（如MLAgentBench）聚焦端到端实验或已指定任务，本文独立将任务识别作为可评测的子问题。

2. **仅依赖目标特征的LLM任务推断框架**：提出仅需用户提供目标特征名，即可从序列化数据集、目标统计量和可选文本描述中联合推断数据域与预测任务的Prompt驱动系统。
   - *区别*：不同于AutoML-GPT、MLCopilot等需要显式任务描述的系统，本文不设任务先验，将任务识别视为独立推理问题。

3. **三场景系统性评估揭示性能-部署权衡**：在表格对比、跨域泛化、本地小模型部署三个设置下全面评估，量化了大模型与4B级本地模型的性能差距及语义上下文的重要性。
   - *区别*：此前工作多聚焦云端大模型，本文同步评估了消费级GPU（RTX 5090）上的可行性与代价。

4. **揭示目标特征统计量的强信号价值**：证明仅凭目标特征的名称、取值分布、均值/标准差等统计量，即可在表格场景达到0.95+的F1 macro，无需文本描述。
   - *区别*：为"最小信息输入"设定提供了实证依据，挑战了"需要丰富元数据才能准确推断任务"的直觉。

## 方法详解
- **目标特征统计量提取（Target-specific Statistics）**：从目标变量提取计数、唯一值、均值、标准差、最小/最大值；对类文本目标额外提取词数、停用词数、字符数及其标准差。
- **数据集序列化（Dataset Serialization）**：将数据集转换为保留行列结构和原始特征名的DataFrame序列化表示，目标特征名始终显式标记，以支持LLM理解变量语义。
- **分组行采样（Group-wise Row Sampling）**：采用无放回连续采样策略选取样本子集，并按原始位置重排，以保留时序/分组数据中的短程模式，同时控制输入长度。
- **提示设计（Prompting Structure）**：
  - System Prompt：定义分类任务（数据域：Tabular/Time_Series；子任务：binary/multiclass/regression）及输出格式规则。
  - User Prompt：包含序列化数据集片段、目标特征说明、目标统计量及可选数据集描述。
  - 支持Zero-shot与Few-shot两种策略，Few-shot在系统提示中嵌入标注示例。
- **数据划分与超参选择**：20%训练集（用于提示工程、序列化策略、few-shot示例的手工设计）、20%验证集（模型与配置选择）、60%测试集（最终评估）；不采用交叉验证以避免手动调优过程中的实验者偏差；通过1000次Bootstrap重采样报告均值与标准差。

## 实验与结果
- **数据集**：625个公共数据集（OpenML、UCI、Kaggle、TS Classification Archive），299个时序 + 326个表格；标签由人工独立分配，排除LM Contamination Index中的数据集以降低污染风险。
- **基线**：AutoGluon（0.93 F1 macro）、NaiveAutoML（0.80）、H2O（0.59）——均仅限表格设置。
- **Experiment 1（表格任务）**：
  - 最优配置Qwen3-14B（DD+ST，Few-shot/Zero-shot）达到**0.98 F1 macro / 0.98 Balanced Accuracy**，全面超越AutoGluon。
  - 仅用目标统计量（ST-only）亦可达**0.96 F1 macro**，证明统计特征已具强判别力。
  - AutoGluon主要混淆回归与多分类，LLM误分类极少。
- **Experiment 2（跨域任务）**：
  - GPT-5.3（Reasoning+DD+ST，Few-shot）达到**0.90 F1 macro**，为跨域最强结果。
  - Qwen2.5-14B-Instruct（DD+ST，Few-shot）达到**0.84 F1 macro**；Qwen3-14B为0.76。
  - 跨域错误主要表现为"同一任务在不同数据域间的混淆"（如表格回归 vs 时序回归）。
- **Experiment 3（本地小模型部署）**：
  - Qwen3-4B-Instruct（DD+ST，Few-shot）达到**0.75 F1 macro / 0.74 Balanced Accuracy**；零样本0.73。
  - Qwen3-4B-Thinking（推理模式）性能显著下降（Few-shot仅0.62），表明小型模型的推理模式不鲁棒。
  - ST-only导致4B模型性能骤降至0.50，说明小模型高度依赖语义上下文。
- **效率**：AutoGluon平均延迟0.0004s/提示，远快于LLM；Qwen3-4B-Instruct延迟约8s，属离线分析可接受范围；GPT-5.3单次Few-shot成本约$0.009。

## 相关工作脉络
- **AutoML-GPT [55]**：使用LLM协调数据处理、模型选择与超参调优，但任务由用户提供指令显式指定；本文相反，任务需从数据中推断。
- **AutoML-Agent [48]**：将ML流程分解为子任务由LLM Agent处理，同样起始于明确任务描述；本文不预设任务。
- **MLAgentBench [23]**：评估语言Agent的端到端ML实验能力，任务由文本描述+代码+数据共同定义；本文仅用数据集+目标特征，去除了任务先验。
- **CAAFE [22]**：利用LLM从数据集描述生成语义特征，但描述本身已包含大量预测问题信息；本文强调描述仅提供数据集刻画而非任务定义。
- **MLCopilot [54]**：通过检索历史ML经验推荐解决方案，新任务仍需显式任务规格；本文任务识别独立于后续流程。
- **传统启发式AutoML（AutoGluon [14]、H2O [33]、NaiveAutoML [39]）**：仅支持表格任务识别，依赖规则/元特征启发式；本文扩展到跨域并引入语义理解能力。

## 局限性与未来方向
- **数据集代表性局限**：基准主要由经过整理的公共数据集构成，未能充分反映工业数据的噪声、不一致性和复杂性。
- **数据污染风险**：封闭源模型可能在预训练中接触过测试数据集或相关变体，尽管已排除LM Contamination Index中的数据集并独立分配标签，但内容层面的污染仍无法完全排除；未来需引入记忆测试等专门分析方法。
- **未评估下游链路影响**：当前仅评测任务类型识别指标（F1 macro、Balanced Accuracy），未量化错误识别对预处理、模型选择、训练及整体流水线性能的影响。
- **目标特征需人工指定**：系统依赖用户提供目标特征名，未来需研究自动目标特征识别以进一步降低用户门槛。

## 研究启发与可借鉴点
1. **目标特征统计量作为强先验信号**：仅需目标变量名称及基本统计量（均值、标准差、唯一值分布）即可在表格场景达到高水平任务识别，为后续研究提供了"最小信息设定"的可靠基线。
2. **分组行采样保留结构化模式**：连续块采样+按原始位置重排的序列化策略，在控制输入长度的同时保留了时序和分组数据的局部结构，可直接迁移至其他结构化数据LLM任务。
3. **Prompt工程与输入配置的解耦评估**：通过DD（数据集描述）、ST（目标统计量）、BP（基础Prompt）的组合消融，清晰揭示了跨域场景对语义上下文的更强依赖，为系统优化提供了明确的方向指引。
4. **本地小模型的推理模式需谨慎使用**：Qwen3-4B-Thinking在资源受限场景下表现明显劣于Instruct变体，提示小型模型的"思考/推理"模式并非万能，需按场景权衡。
5. **可结合本团队AutoML流水线**：将本文的LLM任务识别模块前置到Pipeline构建流程中，可在用户仅指定目标变量后自动完成数据域与任务类型的推断，降低非专家用户的配置门槛。

## 关键术语表
- **Task Type Identification**：指从数据集信息中推断数据域（表格/时序）和预测任务（二分类/多分类/回归/预测）的问题。
- **DD (Dataset Description)**：数据集创建者提供的文本描述，包含数据来源、变量含义等语义上下文信息。
- **ST (Target-Specific Statistics)**：从目标变量提取的统计特征，包括计数、唯一值、均值、标准差、极值等。
- **DD+ST**：同时使用数据集描述和目标统计量作为额外输入信息的配置。
- **Few-shot Prompting**：在系统提示中包含若干已标注示例，引导LLM进行上下文学习。
- **Zero-shot Prompting**：不提供示例，仅依靠系统提示和输入数据进行分类。
- **Balanced Accuracy**：各类别召回率的平均值，用于处理类别不平衡场景下的评估。
- **F1 Macro**：宏平均F1分数，对每个类别单独计算F1后取平均，平等对待所有类别。

## 可复现要素
- **数据集**：625个公共数据集，已通过Zenodo开源发布（https://zenodo.org/records/21649607）
- **代码**：项目代码已开源（https://github.com/ptsialis/Machine-Learning-Task-Type-Identification）
- **模型**：GPT-5.3（商业API）、Qwen3-14B、Qwen2.5-14B-Instruct、Qwen3-4B-Instruct-FP8、Qwen3-4B-Thinking-FP8（本地部署）
- **数据划分**：20%训练 / 20%验证 / 60%测试，分层抽样，固定划分
- **超参**：模型选择、推理模式（开/关）、提示策略（Zero-shot/Few-shot）、信息配置（BP/DD/ST/DD+ST）
- **评估指标**：F1 Macro、Balanced Accuracy，通过1000次Bootstrap重采样报告均值与标准差
- **随机种子**：论文未提及具体随机种子，使用固定划分而非重复实验
