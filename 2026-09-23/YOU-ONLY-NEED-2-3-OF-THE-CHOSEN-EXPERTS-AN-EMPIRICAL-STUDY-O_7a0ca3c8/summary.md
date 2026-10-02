---
title: "YOU-ONLY-NEED-2-3-OF-THE-CHOSEN-EXPERTS-AN-EMPIRICAL-STUDY-O"
source: https://arxiv.org/pdf/2609.25809v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:28:38"
field: "高效大模型推理"
keywords: ["Mixture-of-Experts", "Dynamic Pruning", "MoE Inference", "Expert Selection", "Efficient LLM"]
innovations: ["首次系统评估细粒度MoE的路由冗余，证明2/3均匀截断保留98.8%性能", "建立iso-budget比较框架揭示动态剪枝仅在激进预算下有效", "识别生成式任务对剪枝最敏感、多模态模型最脆弱的规律"]
benchmarks: ["AIME-24", "AIME-25", "MATH-500", "GSM8K", "LiveCodeBench-v6", "MMLU-Pro", "GPQA-Diamond", "ARC-Easy", "ARC-Challenge", "WinoGrande", "OpenBookQA"]
---

# 论文速读：YOU-ONLY-NEED-2-3-OF-THE-CHOSEN-EXPERTS-AN-EMPIRICAL-STUDY-O

## 一句话总结
本文对细粒度MoE大模型进行系统性实证研究，发现均匀截断至约2/3的选中专家即可保留98.8%性能，动态分配仅在激进剪枝（保留1/3~1/2）时才有显著价值（最高提升3.0分），且生成式任务对剪枝最敏感。

## 研究问题与动机
- **细粒度MoE成为主流**：Qwen3-30B-A3B选择8/128专家，Qwen3-Next-80B-A3B选择10/512专家，路由选择数量快速增长，动态剪枝成为有吸引力的降本推理路径。
- **现有证据存在缺口**：已有研究多基于粗粒度架构（如Mixtral 2/8），且在仅基于likelihood-scored多选题基准上评估，缺乏对细粒度MoE的系统性理解。
- **三个核心问题未解**：①per-token专家选择冗余度有多高？②现有剪枝方法是否充分利用了该冗余？③模型对剪枝的敏感性由何决定？
- **评估基准不全面**：现有方法在选择题基准上表现良好，但未能反映数学推导、代码生成等长程生成任务的实际需求。

## 核心贡献（创新点）
- **系统性表征路由冗余**：首次对12个细粒度MoE checkpoint（9个架构家族）进行uniform budget sweep，证明保留⌈2K/3⌉专家可保留98.8%平均性能，远超领域预期。
- **动态剪枝价值的重新评估**：建立统一比较框架，证明在保守预算下动态规则与FIXED-K差异<1个百分点，但在激进预算下最佳规则可恢复最多3.0分。
- **模型敏感性因素的识别**：发现大型模型和thinking模式后训练更具剪枝韧性，多模态模型更脆弱；生成式任务比QA任务对剪枝更敏感。
- **提供部署指南**：将研究结论转化为三步建议——先用uniform truncation确定预算，再衡量动态分配的增量价值，最后针对tight budgets设计分配策略。

## 方法详解
- **Uniform Truncation (FIXED-K)**：对所有token和layer统一保留top-k个专家（k≤K），重新归一化gate weight，公式：$\tilde{w}_i = w_i / \sum_{j \in S} w_j$。
- **四种动态剪枝规则**：
  - **NAEE**：基于gate weight相对top-1的比例阈值，跳过权重低于β·w₁的专家。
  - **DYNROUTE**：保留最短前缀使其累积概率达到目标p。
  - **DIEP**：在NAEE基础上引入专家输出相似度进行调整，相似度高则更易被剪枝。
  - **BAN**：结合token-level路由信号（top-3 expert weight ratio）与layer-sensitivity profile，同时跨token和depth分配预算。
- **校准与预算求解**：所有规则使用C4语料（512文档）进行离线校准，通过beta分布拟合和二分搜索将超参映射到目标$\bar{k}$，确保iso-budget比较。
- **评估体系**：11个benchmark，分为三类——Knowledge QA（ARC-Easy/Challenge, WinoGrande, OpenBookQA）、Math & Code Reasoning（AIME-24/25, MATH-500, GSM8K, LiveCodeBench-v6）、Knowledge & General Reasoning（MMLU-Pro, GPQA-Diamond）。

## 实验与结果
- **数据集与模型**：12个checkpoint，涵盖Qwen3-30B-A3B、Qwen3-Next-80B、Ling-lite-1.5、GPT-OSS-20B、MiniMax-M2.7、DeepSeek-V2-Lite、Hy3、DeepSeek-V4-Flash、Gemma 4 26B-A4B，以及Qwen3-Thinking/235B/VL变体。
- **核心结果1**：保留⌈2K/3⌉专家（如K=8→k=5，K=10→k=6）平均保留98.8%性能；K=8时k=5的mean score为65.43 vs unpruned 67.49（仅降2.06点）。
- **核心结果2**：保守预算（ρ≈0.6）下，最佳动态规则与FIXED-K差异仅-0.53~+0.67点；激进预算（ρ≈0.33~0.4）下，BAN在Qwen3-30B-A3B上提升2.45分，在Ling-lite-1.5上提升2.98分。
- **核心结果3**：生成式任务（数学/代码）是动态剪枝价值的主要来源——Qwen3-30B-A3B激进预算下BAN在7个生成任务上提升3.17分，而QA任务仅提升1.20分。
- **速度提升**：在vLLM后端decode吞吐量提升1.18~1.65×，HF Transformers后端提升1.44~1.73×。
- **模型敏感性**：Thinking模型在k=3时比Instruct模型高5.1分；235B比30B模型在k=3时高14.4分；VL多模态模型在k=3时落后11分，k=2时落后42分。

## 相关工作脉络
- **Fine-grained MoE发展**：从Mixtral（2/8）到Qwen3（8/128）、Kimi K3（16/896），expert pool持续扩大，使得剪枝操作空间更丰富。
- **Dynamic Expert Pruning**：NAEE（Lu et al., 2024）、DIEP（Bai et al., 2025）、DYNROUTE（Huang et al., 2024）、BAN（Chen et al., 2026）、EAC-MoE（Chen et al., 2025），本文首次在同一细粒度框架下系统比较这些方法。
- **Structural Pruning vs Dynamic Pruning**：SEER-MoE等静态剪枝永久移除expert，而本文聚焦inference-time动态剪枝（保留expert池）。
- **MoE路由研究**：DeepSeekMoE引入shared expert，本文强调routing signal多样性（softmax/sigmoid/sqrt-softplus）对剪枝的影响。
- **评估方法批评**：本文指出likelihood-scored基准低估了剪枝成本，主张将长程生成任务纳入评估。

## 局限性与未来方向
- **动态规则实现限制**：DIEP仅实现其skip criterion，未包含其原始可微剪枝目标，可能低估其实际能力。
- **校准数据单一**：所有规则使用C4语料校准，可能不适用于特定领域模型。
- **缺乏在线适配**：研究聚焦静态部署，未探讨剪枝规则在推理过程中根据token难度动态调整的可能性。
- **未来方向**：设计结合layer sensitivity与token routing的allocation rule；探索跨architecture generalizable的calibration-free方法；优化kernel支持以支持per-token variable k的高效执行。

## 研究启发与可借鉴点
- **基准设计**：采用"matched $\bar{k}$"而非"matched hyperparameter"进行比较，避免不同规则因budget差异而产生不公平对比——此方法可迁移至任何pruning/routing研究。
- **任务分层评估**：区分knowledge QA与generative tasks，揭示likelihood scoring无法捕捉的性能退化，为评估协议设计提供范例。
- **单因子对照实验**：通过Qwen3系列变体（Thinking/Scale/Modality）控制变量，清晰分离各因素对剪枝敏感性的影响，适合多因素消融研究的参考。
- **实际部署视角**：同时报告vLLM和HF Transformers后端的速度提升，区分"理论计算节省"与"实际吞吐增益"，为工程落地提供参考。

## 关键术语表
- **Fine-grained MoE**：每层数百个expert、每token选择4~16个的稀疏混合专家架构。
- **FIXED-K (Uniform Truncation)**：对所有token统一保留top-k个专家的剪枝策略。
- **Dynamic Expert Pruning**：根据router score或其他信号，为每个token动态决定执行多少expert的剪枝方法。
- **Routing Redundancy**：router选中的expert中，低权重部分对最终输出贡献有限的性质。
- **ISO-Budget Comparison**：在不同规则间保持相同平均活跃expert数量（$\bar{k}$）的比较方法。
- **Gate Weight**：router为每个selected expert分配的softmax权重，反映该expert被选中的置信度。
- **Layer Sensitivity**：某一层对expert减少的敏感程度，通过比较全路由与top-3路由的next-token distribution divergence来衡量。
- **Thinking Mode**：经过reasoning post-training的大模型模式，倾向于生成长推导链。

## 可复现要素
- **数据集**：ARC-Easy/Challenge, WinoGrande, OpenBookQA, AIME-24/25, MATH-500, GSM8K, LiveCodeBench-v6, MMLU-Pro, GPQA-Diamond；公开可用。
- **代码**：论文声明"Code is available at GitHub"，附录A说明已开源router patch。
- **权重**：使用公开权重（Qwen3, Ling-lite, GPT-OSS, DeepSeek, Gemma 4, MiniMax等）。
- **关键超参**：FIXED-K的k值；NAEE的β；DYNROUTE的p；DIEP的β和α；BAN的λ；所有规则使用$k_{min}=2$或3。
- **校准数据**：C4语料512文档。
- **批量大小**：4，prompt长度1024 tokens，生成256 tokens。
