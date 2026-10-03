---
title: "It-s-All-Training-A-Fully-Synthetic-Single-Stage-Recipe-for"
source: https://arxiv.org/pdf/2609.37891v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:53:00"
---

# 论文速读：It's-All-Training-A-Fully-Synthetic-Single-Stage-Recipe-for-LLMs

## 一句话总结
本文提出 **SYNTH**，首个完全合成的开源预训练语料库（约 80B tokens、8 种语言），通过基于 58,698 篇 Wikipedia 文章的"反翻译+约束驱动放大"两阶段管道，将预训练、中训练、后训练合并为单一训练阶段；在此基础上训练的 BAGUETTOTRON 系列（56M–13B/1B active MoE）在多个评测上追上或超越训练 token 数多 80–700× 的同规模开源基线。

## 研究问题与动机
- **爬网数据的结构性缺陷**：主流预训练依赖 C4 / Common Crawl / FineWeb 等爬取语料，存在版权风险、噪声不可控、来源证明不清等问题；部分站点已开始反爬。
- **已有合成数据方法缺乏对输入的控制**：DeepSeek-R1 式蒸馏推理轨迹、Maini 等“重写”方案都以源模型答案为输入，错误会递归传播并引发 model collapse；单模板重述（Phi-1.5、Cosmopedia）则易触发 surface-diversity collapse。
- **网络预训练仍需多阶段训练**：同样 600M 参数，FineWiki / FinePDFs-Edu 在零后训练下甚至无法产生合法多选答案，必须额外 SFT/RLHF 阶段；合成语料有望把这一整条流水线压平为一步。
- **小模型的数据效率被低估**：Chinchilla 最优比（~20×参数量 tokens）在 Web 数据上拟合，但在人工设计的种子 + 合成放大语料下训练仍可继续下降，暗示数据可学习性是独立于模型尺寸与 token 数的可控维度。

## 核心贡献（创新点）
1. **SYNTH 语料库与单阶段训练配方**：首次发布完全合成、开源、支持全训练阶段的 80B-token 数据集（CC BY 4.0）；与以往仅改写现有语料或单次蒸馏不同，本文通过"受控种子 + 反翻译 + 约束驱动多样查询"把 pre-/mid-/post-training 统一为一个阶段。
2. **BAGUETTOTRON 系列 + token 效率前沿**：从零训练 56M–600M 稠密与 13B/1B-active MoE 四款模型，在 22 MCQ + 8 开放 QA 上与 Gemma-3 / Qwen3 / LFM2.5 / SmolLM2 等开源基线可比；相同参数规模下训练 token 数仅为基线的 1/80–1/700。
3. **面向事实性的结构化记忆与校准**：以 FActScore 协议在 500 个种子实体上评测，BAGUETTOTRON 在各参数档位均取得最高 macro 分数（MoE 达 46.3%），且能在未见过实体上主动放弃回答（held-out 放弃率 67%，web 基线仅 7%）；推理链中的认识性标记也能与事实精度负相关对齐。
4. **领域/任务适配与工具调用的可行性验证**：用同一管道在 3GPP 标准 + 电信 Wiki 上合成 ~460M token，600M 微调后 TeleQnA +15.1pp、FactScore +17.3pp；工具调用在仅增加 ~1.5B token 后 BFCL v2 达 53.1%，3.9pp 超越 FunctionGemma-270M（训练 token 多 40×）。
5. **可复现的封闭实验控制**：除性能外，系统展示 SYNTH 为何能降低 model collapse（结构化的 seed grounding + 任务异构性）与幻觉（20% 负样本查询教会拒绝回答）的机制，并在附录给出 FActScore 协议、Propella-1 质量评分、计算账目等完整细节。

## 方法详解
- **两阶段合成管道（Figure 1）**
  - **Stage 1**：用 frontier LLM（Gemini 2.5 Pro）生成带监督的 ⟨seed, constraints, query⟩ 三元组与答案/推理轨迹，分别微调两个辅助模型——query model（LoRA on Gemma-3-12B-base）与 reasoning model。
  - **Stage 2**：固定已微调的两个辅助模型，对每段种子均匀放大 100×，用 frozen bge-m3 + FAISS IVF-flat 检索近邻段落，拼接成 query + context triplet，经 task-specific adapter 产出多任务训练样本。
- **种子语料（~58k 文章）**
  - 50k 篇 Wikipedia vital articles（level 1–5）；1–4 级拆出全部结构化节，5 级只保留 lead abstract。
  - 补充 8,698 篇法律/医学/化学专业文章（基于 category-tree + Wikidata 图扩展）、3,727 页 Wikibooks（烹饪等百科缺项）、130 份 AI 自我文档与事件资料。
- **Query model 与约束先验**
  - 六轴独立先验：query type / complexity / user profile / query result / target language / query style；除 query result 故意偏斜（positive 0.80 / negative 0.10 / absurd 0.05 / ambiguous 0.05）外其余均匀采样。
  - **20% 负样本查询**是关键：让模型学会拒绝/纠正/保留态度，避免训练语料把"永远自信作答"当作默认分布，这也是 FActScore 精确度优势的原因之一。
- **Reasoning model 与简写语法**
  - 输入：query + seed paragraph + 最近非 seed 近邻（bge-m3 检索）。
  - 推理迹使用三族专用标记 token：
    - 逻辑标记：→、⟲、∴
    - 认识性标记：• certain … ⃝ uncertain
    - 验证标记 + 多步问题的树形分解
  - 每个推理步骤压缩为 1–2 token，使小模型在固定上下文窗口内仍能输出丰富迹线；熵标记 ⟨H≈X.X⟩ 仅作训练时注解，推理时不使用。
  - 新增专用 token 加入 tokenizer（占最后 ~50 个位置）。
- **辅助任务管道（§3.2）**
  - RAG：复用记忆查询流，最多检索 10 段，引入 `<source>` 引用标记，训练 closed/open-book 切换。
  - Arithmetic：基于 ~3k 个 Kimina 模板随机化变量并重算符号解；用 Qwen-3-8B 保证公式准确性。
  - Creative writing / Editing / MCQ / Practical knowledge（Wikibooks 烹饪食谱）：按 §3.2 描述的任务格式生成。
- **Tokenizer**
  - BAGUETTOTRON 系列：BPE，V=65,535，在 Common Corpus 样本上训练以保持多语覆盖；尾部 ~50 token 为 SYNTH 推理控制符。
  - MONAD-56M：8k BPE，直接在 SYNTH 英文段上训练。
- **训练配置（Appendix B）**
  - 统一：seq_len=2048、AdamW（wd=0.01, clip=1.0）、16.6% linear decay tail to 0.2% peak lr。
  - MONAD-56M：~180B tokens、16×H100、seq=1024→2048 延拓；最终 CE=1.52。
  - BAGUETTOTRON-350M：~199B tokens、80 layers、d=576；最终 CE=1.15。
  - BAGUETTOTRON-600M：~158B tokens、151k steps、~300 TFLOPS/GPU（~30% MFU）；最终 CE 仍在下降。
  - BAGUETTOTRON-MoE：~50B tokens（< 一轮 SYNTH）、31.8k steps、top-1 16 experts、aux loss=10⁻³；CE=1.19 与 600M 稠密持平。

## 实验与结果
- **评测设置**：22 个 MCQ（MMLU 及医学/科学/金融/地理/工程等垂直）+ 8 个开放 QA；零-shot、vLLM 推理、temp=0.1、chatml 格式。
- **Token 效率前沿（Figure 4）**
  - BAGUETTOTRON-600M（158B tokens）MCQ 42.2% / 开放 24.3%；Qwen3-0.6B（~36T tokens）仅高 4.7pp（MCQ）与 0.7pp（开放）。
  - MONAD-56M（180B tokens）在 MCQ 前沿超越 Gemma-3-270M 与 SmolLM2-360M 平均。
  - 强项：TruthfulQA、ESGenius、FormationEval 与 Qwen 持平或超越；弱项：ARC-Challenge、GeoBench（依赖广博网络知识，种子受限）。
- **数据 ablation（Table 8）**
  - FineWiki-600M / FinePDFs-Edu-600M 均经 SmolTalk + MMLU-aux 后训练（~100M tokens、3 epoch）。SYNTH-600M 无需任何后训练，仍领先：MCQ +16–17pp、开放 +11–14pp；两个 web 基线在 MMLU 上仅 ~24–26%（近乎随机）。
- **推理迹 ablation（Table 9）**
  - 去掉 SYNTH 推理迹（其余 tokens/seed/步骤一致）：MCQ 41.8% vs 42.2%（无显著差异），但开放端点 22.0% vs 24.3%，其中 TruthfulQA −9.1pp、NuclearQA −10.0pp；冲突 QA 反升 +3.9pp。
- **FActScore 事实精确度（Table 1、Figure 5）**
  - 各参数档位的 macro = S/(S+C+I)：MONAD 16.3%、BAGUETTOTRON-350M 32.4%、BAGUETTOTRON-600M 41.7%、BAGUETTOTRON-MoE **46.3%**；均为各自档位最高。
  - 远超同档位的 LFM2.5-350M、SmolLM2-360M-IT、Gemma-3-270M-IT，以及参数大 6× 的 Phi-4-mini-instruct（29.5%）、参数大 2.8× 的 DeepSeek-MoE-16B-Chat（38.3%）。
  - 唯一在同等 token 预算（180B）下训练的 OPT-350M 出现比支持事实更多的矛盾事实（S=4.7% vs C=22.5%）。
- **Held-out 实体（Table 6、Appendix G）**
  - BAGUETTOTRON-MoE 在不在种子中的 600 篇 Good Articles 实体上，放弃回答率从 in-seed 19.8% 升至 66.8%，尝试回答时的精确度由 82% 降至 62%；同规模 web 基线 OLMoE-1B-7B 放弃仅 7.2%，保持 ~81% 精确度（web 数据覆盖更广）。
- **认识性标记校准（Figure 5）**
  - BAGUETTOTRON-350M/600M/MoE：不确定迹的 FActScore 显著低于自信迹；输出原子事实量亦相应减少 15–20%。MONAD 仅学到"少说"而没学到"更准"，说明校准精确度需要更大容量。
- **领域适配（§5.3）**
  - 电信：600M 基础 + ~460M 合成域内数据微调；TeleQnA +15.1pp（41.6%→56.7%）、FactScore 3GPP +17.3pp；LLM-as-judge 达 67.8，是 wiki-only 的 1.15×、600M 基础的 2.4×。
- **工具调用（§5.3、Table 10）**
  - 仅追加 ~1.5B token（70/30 混合），BAGUETTOTRON-600M 在 BFCL v2 达 53.1%，超越 FunctionGemma-270M（36T tokens，49.2%）3.9pp；600M 基础本身在工具调用上为 0。

## 相关工作脉络
- **Synthetic rephrasing（Maini et al., 2024; BeyondWeb; Su et al., 2025）**：对已有 Web/百科语料进行质量导向重写或精选。差异：SYNTH 从受控种子反翻译生成全新语料，且引入结构化推理迹与约束多样性，避免单模板带来的分布坍缩。
- **Pure synthetic pretraining（Phi-1.5; Cosmopedia）**：教科书式改写单一模板。差异：SYNTH 以多任务、多约束、多领域种子（Wikipedia + Wikibooks + 专业文献）替代单模板，并通过任务异构防止 surface-diversity collapse。
- **Synthetic mid-training / reasoning traces（Open-Thoughts; DeepSeek-R1 蒸馏）**：用 SOTA 模型蒸馏 CoT 加入 mid-training。差异：SYNTH 的推理迹与指令都来自"反翻译+种子 grounding"的端到端管道，且全部并入单一 pretrain 阶段，无需额外 mid/post 训练；同时 20% 负样本查询显式教会拒绝。
- **Model collapse 理论（Shumailov et al., 2024; Dohmatob et al., 2024; Feng et al., 2024）**：递归训练会使分布支撑收缩。差异：SYNTH 通过 seed grounding 与任务多样性从构造上缓解 collapse（Appendix A/B 约束先验的六轴独立采样）。
- **控制环境下的记忆容量研究（Morris et al., 2025; Allen-Zhu & Li, 2024; Ye et al., 2026; Lin et al., 2025b）**：证明有效知识容量对数据组成高度敏感（1:7 useful/junk 比会 20× 削减有用知识容量）。差异：SYNTH 用受控种子实现 10–140× 更少 token 下仍达最高 macro FActScore，直接验证这一理论。
- **开放高质量种子源（KL3M; Common Pile; Common Corpus; FineWiki）**：SYNTH 与它们在种子层面对齐（Wikipedia/Wikibooks 均 CC BY-SA 4.0），但 SYNTH 在合成放大、多任务、多语、推理迹方面走得更远。

## 局限性与未来方向
- **种子覆盖天花板**：~58k 篇 Wikipedia 决定了参数的知识边界；Wikipedia 之外的专业、当代、多语种内容覆盖不足。
- **文化多样性偏向**：种子与多语管道均以英文 Wikipedia 视角为主，其他语言内容是在英文种子基础上派生而非原生。
- **难度校准面向 SLM**：当前 SYNTH 难度针对小型模型，扩展到通用大模型需引入代码生成、多步 tool use、环境接地轨迹、长 horizon 任务等新管道。
- **纯合成的边界**：去掉所有网络数据虽便于归因，但像"拒绝"等行为也可能在有机网络数据中自然习得；未来应探索 hybrid 配方。
- **工具调用仅是初步验证**：当前仅用 ~1.5B token 就展现出工具能力，但尚未与全量 SYNTH 合并 pretrain，也未扩展到更复杂的 agentic 场景。

## 研究启发与可借鉴点
- **六轴约束先验驱动查询多样性**：除了常见的 query type/complexity，新增 query result（故意偏斜到 20% 负样本）与 query style 轴，可直接复用以避免递归训练中"自信作答"的默认偏向。
- **简写推理语法 + 专用 token**：把 CoT 压成 1–2 token 的逻辑/认识性/验证标记，对小模型极友好；可迁移到任何希望在小尺寸下保留长迹线优势的 setting。
- **以 FActScore + held-out 实体评估"知道/不知道"边界**：不仅报告 S/(S+C)，还报告 S/(S+C+I) 与放弃率，能更真实反映模型的事实校准；建议作为未来合成数据评估标配。
- **Seed + 近邻双段 grounding**：推理模型同时使用种子段与检索到的非种子近邻，既保证记忆又构建语义桥；可在 RAG 训练或知识密集型合成任务中复用。
- **同架构 + 仅换数据的 ablation 设计**：§4.2 用同一 600M 架构对比 SYNTH / FineWiki / FinePDFs-Edu，排除架构与训练超参干扰，归因清晰；后续做合成 vs. 网络对比时应尽量对齐这一范式。

## 关键术语表
- **SYNTH**：论文提出的首个完全合成的开源预训练语料库（~80B tokens，8 种语言），基于 58,698 篇 Wikipedia 等种子文章经反翻译与约束驱动放大生成。
- **BAGUETTOTRON**：在 SYNTH 上从零训练的模型系列，含 MONAD-56M、BAGUETTOTRON-350M/600M（稠密）与 BAGUETTOTRON-MoE（13B total / 1B active）。
- **反翻译（back-translation）**：论文合成管道的核心机制——以种子段落为 grounding，由 query model 生成多样化查询，再由 reasoning model 输出答案与推理迹。
- **约束先验（constraint priors）**：query model 采样六轴（type/complexity/profile/result/language/style）先验以保证合成数据的多样性与可控性，其中 query result 故意偏斜以引入负样本。
- **FActScore**：基于 FActScore 协议的评估方式，把模型回答分解为原子事实，对照源维基百科段落标注 Supported/Contradicted/Inconclusive。
- **model collapse**：递归在合成数据上训练导致分布支撑萎缩、输出多样性下降的现象；本文通过 seed grounding 与任务异构构造性缓解。
- **epistemic markers**：推理迹中的认识性标记（• certain / ⃝ uncertain 等），用于显式表达每个事实的不确定性；实验显示 SYNTH 模型学会将其与输出精度/长度对齐。
- **stenographic syntax**：论文引入的"速记"推理语法，把逻辑/验证步骤压缩为极少数专用 token，使小模型能在固定上下文内输出更丰富的推理链。

## 可复现要素
- **数据集**：SYNTH 全文以 CC BY 4.0 在 HuggingFace（pleias/synth）公开；附录提供 5,000 行样例与 500 实体的 FActScore 评估代码。
- **模型权重**：BAGUETTOTRON 系列以 Apache-2.0 发布（pleias/baguettotron）；匿名评审期 BAGUETTOTRON-350M 权重托管于 https://anonymous-hf.up.railway.app/a/3e0gcxurz29r/。
- **代码**：训练与数据生成代码未开源；但 §3、Appendix A–H 提供了足够复现的构造细节、训练配置与评估协议。
- **关键超参**（Appendix B）：seq_len=2048、AdamW（wd=0.01、clip=1.0）、16.6% linear decay tail to 0.2%；各模型 token 预算、steps、batch size、峰值 lr、warmup steps 均列出；MoE 用 top-1 16 experts、aux loss=10⁻³。
- **硬件**：训练主要在 16×H100（4 nodes × 4 GPUs × 64GB）上完成，MFU 约 13–30% 不等；Stage 1 生成用 Gemini 2.5 Pro API。
- **评估协议**：零-shot、vLLM、
