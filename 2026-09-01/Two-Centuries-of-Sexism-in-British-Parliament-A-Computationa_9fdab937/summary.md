---
title: "Two-Centuries-of-Sexism-in-British-Parliament-A-Computationa"
source: https://arxiv.org/pdf/2608.30485v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:21:36"
field: "计算社会科学与政治话语分析"
keywords: ["Ambivalent Sexism", "LLM-as-a-judge", "parliamentary discourse", "Hansard corpus", "hostile vs benevolent sexism", "stance classification", "computational social science", "gender and politics"]
innovations: ["将 ASI 社会心理学框架引入 200 年议会文本的双轴性别歧视分类并验证生态效度", "发布含 89.3% Speaker 性别匹配的 670 万+ 下议院演讲 enriched 数据集", "揭示支持与反对女性参政权两派在敌意/善意性别歧视类型上的根本性修辞分化"]
benchmarks: ["300 篇人工标注验证集（Stance κ=0.711）", "ParlVote / ParlaSent 议会情感立场资源", "SST-2 情感混淆控制", "跨模型人类标签一致性（Claude/GPT-5/Gemini/DeepSeek）"]
---

# 论文速读：Two-Centuries-of-S Sexism-in-British-Parliament-A-Computationa

## 一句话总结
本文利用大型语言模型对英国下议院 200 年（1803–2005）Hansard 语料库中 6,531 篇与女性参政权相关的演讲进行分类，识别说话人对女性政治代表的立场（支持/反对）及其中蕴含的敌意/善意性别歧视类型；核心发现是反对方与支持方使用了截然不同的性别化推理策略。

## 研究问题与动机
- 核心问题：在女性选举权辩论中，支持方与反对方是否仅得出不同结论，还是使用了根本上不同的性别化推理/修辞策略？
- 现有议会话语分析多聚焦情感极性、立场检测与意识形态缩放，极少从社会心理学维度区分"性别歧视的类型"（而非仅检测是否存在性别歧视）。
- 历史材料（古英语、复杂议会惯例、长篇幅修辞）使众包标注不可行，需依赖具备语境理解能力的大语言模型作为"评判者"。
- 需要构建一个覆盖 200 年、带有 Speaker 级性别元数据的大规模议会数据集，以支撑后续计算社会科学长期研究。

## 核心贡献（创新点）
- **发布大规模 enriched Hansard 数据集**：670 万+ 下议院演讲、89.3% Speaker 性别匹配率，填补历史政治文本缺少 Speaker 级元数据的空白。
- **将 Ambivalent Sexism Inventory（ASI）框架引入计算分析**：首次在对 200 年自然主义议会话语的系统标注中验证敌意/善意歧视的双轴分类，提供实验室之外的生态效度证据。
- **LLM-as-a-judge 双通道分类 pipeline**：Pass 1 判立场、Pass 2 判两类性别歧视子类型并要求引用原文佐证，兼顾可解释性与 schema 约束输出。
- **揭示对立立场的"修辞鸿沟"**：证明反对与支持的性别化论证模式本质不同，支持方的歧视几乎全是 benevolent，反对方的歧视以 hostile 为主（或其组合），超越简单的正反极性划分。
- **多维度鲁棒性验证**：交叉模型一致性、Sentiment 混淆排除、噪声注入稳健性、Speaker 级别聚合分析，确保结论非单一模型偏差。

## 方法详解
- **语料采集与 Speaker 性别匹配**：从 Hansard Parliamentary Corpus 抽取下议院 557 万+ 演讲，采用多级级联匹配（议会头衔、选区记录、时间窗口、Levenshtein 模糊匹配、WikiData、Wikipedia 名录、Gender Guesser 库），优先精确率，最终 Commons 覆盖率达 89.3%；上议院仅 1.2% 故舍弃。
- **关键词两级筛选**：Tier 1（高置信，显式 suffrage 术语）与 Tier 2（women/female 与投票相关词 25 词窗口）召回 6,531 篇候选，再由 LLM 过滤 55% 不相关发言。
- **双通道分类 Prompt（Claude Sonnet 4.6 为主，辅以 GPT-5/Gemini 2.5 Flash/DeepSeek V3）**：每篇 TARGET 附前后各 5 篇 CONTEXT。
  - **Pass 1 立场**：标签 For / Against / Both / Irrelevant，强调程序性反对不等于实质性反对、勉强承认不等于背书。
  - **Pass 2 性别歧视**：在 Hostile / Benevolent 两个独立 flag 下，进一步标注子类别（paternalism、gender differentiation、heterosexuality 各一对 hostile/benevolent 变体），且每个 flag 必须引用原文。
- **人工标注验证集**：300 篇均匀采样，两位标注员独立标注后 discussion 解决 106 处分歧，作为 ground truth 评估 LLM。
- **控制变量分析**：
  - Sentiment 混淆：DistilBERT(SST-2) 验证 hostile/benevolent 并非仅是负面情感伪装（两类型均高度负面）。
  - Gender-Era 混杂：logistic 回归控制 decade 后，性别仍显著（OR=2.01, p=0.002）。
  - Noise 注入：1,000 次随机翻转 3% 性别标签，原结论全部稳健。

## 实验与结果
- **验证集分类质量（n=300）**：
  - 立场：Claude Sonnet 4.6 κ=0.711，优于双人 annotator 间一致性 κ=0.644；For/Against/Irrelevant 类 F1≥0.75，稀有类 Both（仅 13 例）被低估（F1=0.22）。
  - 交叉模型：GPT-5 κ=0.692、DeepSeek V3 κ=0.658、Gemini 2.5 Flash κ=0.533；两两模型间 κ 0.533–0.777，确认分类反映文本信号而非单模型偏置。
  - 性别歧视 flag：Claude 对 hostile 与 human 的 κ=0.543、benevolent κ=0.463；precision 高（0.77–0.86）但 recall 偏低（0.43–0.46），即已标定的样本可信，但存在漏检。
- **全量 2,942 篇相关演讲分布**：
  - 立场：For 74%、Against 19%、Both 7%。
  - 性别差异：女性议员 For 比例 93%，男性 70%（χ²=86.75, p<0.001）；该差距在 1928 年普选平等后才基本弥合。
- **性别歧视类型 × 立场**（全量）：
  - 总体 30% 相关演讲含至少一类歧视（886/2,942）：Hostile 392、Benevolent 706、两者兼具 212。
  - For 演讲（n=2,167）：21% 含歧视，其中 81% 为纯 benevolent、仅 11% 纯 hostile。
  - Against 演讲（n=570）：54% 含歧视， hostile-only 37%、benevolent-only 19%、两者兼具 44%。
- **时变趋势**：敌意歧视占"含歧视演讲"的比例从 1870–1899 年的 60% 降至 1929 年后的 27–30%，而善意歧视占比长期稳定在 74–83%。
- **Human-label 子集（n=300）复核**：Against 含歧视 76.7%、For 含歧视 37.5%（Fisher p<0.001）；For 中 91.7% 歧视为纯 benevolent，Against 中 91.3% 涉及 hostility——结论不依赖 LLM 标注。
- **最强结果与提升**：Claude 立场 κ=0.711 vs TF-IDF+LogReg 0.419、DeBERTa-v3 zero-shot 0.269、majority 0.000；相对最强传统基线提升约 70% 相对增益（Cohen's κ 0.419→0.711）。

## 相关工作脉络
- **ParlVote / ParlaSent（Abercrombie & Batista-Navarro 系列；Mochtak et al., 2025）**：面向议会文本的情感/立场/议题缩放资源与方法；本文在其基础上引入 ASI 框架做"类型化"性别歧视分析，而不仅停在极性。
- **Wordfish / ideological scaling（Slapin & Proksch, 2008）**：基于词频的意识形态定位；本文转向以 LLM 为 judge 的双通道分类，关注修辞策略而非位置估计。
- **性别风格与议题研究（Blaxill & Beelen, 2016; Hargrave & Blumenau, 2022; Raiber & Spierings, 2022; Soriano-Jiménez, 2024）**：关注女性议员发言风格演变与议题选择；本文聚焦"性别歧视话语的类型学"，并将其系统映射到社会心理学理论。
- **ASI 原始文献（Glick & Fiske, 1996）**：实验室/问卷量表框架；本文首次在大规模历史政治文本中提供生态效度证据，并检验其跨时代稳定性。
- **在线性别歧视检测（Guest et al., 2021; Kirk et al., 2023; Jha & Mamidi, 2017）**：面向当代平台内容；本文处理的是 200 年古英语议会长篇修辞，标注难度更高，验证了 LLM-as-judge 在历史语料上的迁移能力。
- **美国 140 年移民话语分析（Card et al., 2022）**：与本文方法谱系相近（LLM 长时段话语分析）；本文将其拓展至性别权利议题并引入明确的心理学分类体系。

## 局限性与未来方向
- **关键词检索的边界**：Tier 1/2 召回会漏掉未命中关键词的相关演讲（如 "The People Bill to all women aged 21 and above" 因 vote 与 women 距离 >25 词未被捕获），也引入部分假阳性；LLM 能过滤后者但前者构成硬性上界。
- **单主 Judge + 低 recall**：Claude 对性别歧视的 recall 仅 0.43–0.46，已标样本精度高但真实发生率被低估；benevolent sexism 因嵌入在"正面措辞"中更难检测。
- **仅限下议院**：上议院因爵位变更导致匹配率仅 1.2%，结论不可直接外推至上院。
- **二元性别框架**：历史记录的 binary 性别标签无法覆盖多元性别光谱；噪声注入实验缓解了 misgendering 风险但并未消除这一结构性限制。
- **制度性话语规范**：议会发言受形式、说服与受众期待约束，不一定等同于私人信念。
- **未来方向**：扩展关键词与检索策略；将框架迁移至其他立法机构（EU、US Congress 等）与其他权利议题（婚姻平等、移民等）；引入多 Judge 集成或主动学习缓解 recall 瓶颈；拓展至上议院与地方议会。

## 研究启发与可借鉴点
- **"理论框架 → 计算分类"的桥接范式**：将社会心理学成熟量表（ASI）直接转化为 LLM 分类 schema 并要求原文引证，为其他社会理论的计算落地提供模板。
- **双通道约束式 Prompt + JSON tool-use**：Pass 1 过滤 relevance、Pass 2 在 relevant 子集上判细化标签，既节省成本又降低错误传播；强制 schema 输出便于后续统计。
- **多维度鲁棒性检验清单**：Sentiment 混淆、噪声注入、跨模型一致性、Speaker 级聚合、Human-label 子集独立校验——这一套"防御性验证"可迁移至任何 LLM-as-judge 研究。
- **历史长时程数据中的"类型漂移"信号**：敌意→善意的历时转换提示政策话语的合法性门槛在变化，方法可复用于追踪其他类型偏见（种族、阶级）的演变。
- **与团队方向结合机会**：可将本 pipeline 适配至中文立法/政策语料（如全国人大发言），检验 ASI 类框架在非英语议会语境下的迁移性；亦可把"敌意/善意双轴"引入 LLM 安全评估或内容审核研究。

## 关键术语表
- **Hansard Corpus**：英国议会自 1803 年起公开出版的辩论转录档案，本文核心数据源。
- **Ambivalent Sexism Inventory (ASI)**：Glick & Fiske (1996) 提出的性别歧视双轴理论，区分 Hostile Sexism（敌意）与 Benevolent Sexism（善意）。
- **Hostile Sexism**：明确贬低、控制女性的敌对态度，如认为女性能力不足或需用男性权威统治。
- **Benevolent Sexism**：表面赞美（纯洁、温柔、母性）实则将女性框定在从属/保护性角色的 paternalistic 态度。
- **Protective Paternalism / Dominative Paternalism**：ASI 下的两类父权子类型，前者主张"男性应保护女性"，后者主张"女性需男性统治"。
- **Complementary / Competitive Gender Differentiation**：分别将性别差异表述为"女性拥有独特美德"与"男性在公共领域更胜任"。
- **LLM-as-a-judge**：以大语言模型充当分类/评判器，本文以 Claude Sonnet 4.6 为主、GPT-5/Gemini/DeepSeek 为辅做交叉验证。
- **Suffrage**：指女性选举权与参政权运动及相关的立法进程（1918/1928 年 Representation of the People Act 等）。

## 可复现要素
- **数据集**：Hansard Parliamentary Corpus（公共政府档案）；作者发布了 enriched 版本（6.7M+ 演讲、Speaker 性别元数据）见论文脚注/仓库链接（论文未给出具体 URL，仅以 superscript 引用发布声明）。
- **代码/权重**：论文使用 Claude Sonnet 4.6 / GPT-5 / Gemini 2.5 Flash / DeepSeek V3 通过 API/OpenRouter 调用，未开源自有模型权重；Prompt 细节见附录 C，但未提供完整代码仓库链接。
- **关键超参**：
  - 上下文窗口：每篇 TARGET 前后各 5 篇演讲。
  - 性别匹配：Levenshtein 模糊匹配 + 头衔/时间窗口约束；优先精确率。
  - GPT-5 因 reasoning tokens 计入 budget，max_tokens=16,000 避免静默截断。
  - 情感混淆分析：DistilBERT fine-tuned on SST-2。
  - 噪声注入：1,000 次迭代、每次翻转 3% 标签。
  - 标注时长：每标注员约 40 小时，median 890 词/篇、95th percentile 3,521 词。
