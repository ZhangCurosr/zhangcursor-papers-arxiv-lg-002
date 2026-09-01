---
title: "Understanding-Evolution-Strategies-for-LLM-Reasoning-Broader"
source: https://arxiv.org/pdf/2608.27351v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:18:01"
field: "大语言模型后训练"
keywords: ["Evolution Strategies", "GRPO", "LLM Reasoning", "Pass@K", "Entropy Collapse", "Parameter Drift", "Zeroth-Order Optimization"]
innovations: ["理论上证明ES种群的JS多样性可提升Pass@K", "发现ES更新的功能稀疏性，大漂移不必然导致遗忘", "提出顺序混合ES-GRPO训练策略以兼顾Pass@1与Pass@K"]
benchmarks: ["GSM8K", "GPQA", "MATH-500", "AIME24", "AIME25", "AMC23", "DeepScaleR"]
---

# 论文速读：Understanding-Evolution-Strategies-for-LLM-Reasoning-Broader

## 一句话总结
该论文系统研究了进化策略（ES）在LLM推理后训练中的优化动态与机制，理论上与实验上证明ES相比主流方法GRPO能带来更广泛的推理覆盖（更高的Pass@K），同时指出大参数漂移并不必然导致灾难性遗忘，并给出了使ES有效且可扩展的超参数与估计器设计建议。

## 研究问题与动机
- **ES在后训练中的行为尚未被充分理解**：ES因无需反向传播而被视为内存高效的推理后训练候选，但其优化动态、与GRPO的本质差异以及优势边界仍缺乏系统分析。
- **GRPO存在熵坍塌（entropy collapse）与Pass@K退化风险**：GRPO通过单策略采样与token级策略梯度优化，易导致策略分布集中，从而降低多采样成功率（Pass@K），甚至低于基础模型。
- **ES的大参数漂移是否引发灾难性遗忘存疑**：ES因全参数扰动而累积较大参数移动，既往工作将其与能力退化关联，但该结论基于小规模训练与单一任务，过拟合与真实遗忘难以区分。
- **ES的超参数与估计器设计缺乏系统性指南**：如奖励归一化、扰动尺度、种群大小、单点/两点估计器等如何影响优化稳定性与可扩展性，尚不明确。

## 核心贡献（创新点）
1. **理论上证明验证器投影的JS多样性有助于提升Pass@K**：ES种群中参数扰动诱导的策略多样性（以Jensen–Shannon散度度量）可转化为多采样成功率的理论保障，并与GRPO的单策略熵坍塌形成对比。
2. **实验发现ES能避免熵坍塌并实现更高的Pass@K**：在Easy与Hard两个训练设置下，ES在提升Pass@1的同时，Pass@16/Pass@32均优于GRPO与基础模型，而GRPO在多数任务上出现大K性能退化。
3. **提出顺序混合训练策略（ES→GRPO / GRPO→ES）以兼顾Pass@1与Pass@K**：在相同更新预算下，两种序列组合均在帕累托前沿上引入了非支配点，其中ES→GRPO在数学基准上获得最高Pass@32。
4. **揭示ES更新的功能稀疏性（functional sparsity）**：尽管全参数漂移显著（相对GRPO约40倍），但性能增益仅集中在少量大范数更新上；保留这些更新即可维持任务性能，表明大漂移不等于广泛功能改变。
5. **给出ES超参数与估计器的设计准则**：z-score奖励归一化是关键；两点估计（antithetic/two-point）在推理任务中无优势，单点估计更优；且随模型规模增大，有效种群规模可减小。

## 方法详解
- **ES优化框架**：在每次更新时，从当前中心参数θ出发，采样N个独立的高斯扰动ε_i，对每个扰动方向执行前向 rollout 得到响应，由verifier计算奖励R_i。采用population z-score标准化奖励z_i，更新方向为d̂_ES = (1/N)∑z_iε_i，中心模型更新为θ⁺ = θ + α d̂_ES，其中α为更新尺度。
- **与GRPO的机制对比**：GRPO从单策略采样G个响应，计算组内相对优势Â_i，通过PPO clipped surrogate进行token级反向传播；ES则直接将种群奖励差异映射到参数空间更新，无需反向传播。
- **理论分析**：通过Fisher信息矩阵刻画局部策略漂移，证明ES种群的JS多样性（JS^pol）与成功概率的JS多样性（JS^succ）正相关；在奖励权重与成员成功率正相关且中心转移误差足够小的条件下，ES更新可保证Pass@K提升。
- **超参数设计**：
  - **奖励归一化**：使用z-score标准化而非无归一化，可稳定训练并提升奖励。
  - **扰动尺度σ**：需平衡探索范围与过拟合，过小易陷局部最优，过大则训练不稳。
  - **种群大小N**：随模型规模增大，所需N减小（如0.5B模型需N≥32，而1.5B/3B模型N=16即可接近参考值）。
  - **估计器选择**：两点估计（antithetic）在推理任务中因自回归响应重生成导致正负扰动间协方差弱，无法降方差；单点估计更有效。
- **顺序混合训练**：将总更新预算均分两阶段，先后应用ES与GRPO（ES→GRPO）或GRPO与ES（GRPO→ES），以结合两者在Pass@1与Pass@K上的优势。

## 实验与结果
- **数据集**：
  - **Easy Setting**：训练数据GSM8K；测试基准包括GSM8K、CSQA、HotpotQA、Countdown、GPQA、MBPP。
  - **Hard Setting**：训练数据DeepScaleR；数学基准AIME24、AIME25、AMC23、MATH-500；held-out基准GPQA、MBPP、CSQA、Countdown。
- **模型**：Qwen2.5-1.5B-Instruct、Llama-3.2-3B-Instruct、Qwen2.5-7B-Instruct（Easy）；DeepSeek-R1-Distill-Qwen-1.5B（Hard）。
- **基线**：Base模型、GRPO、ES、ES→GRPO、GRPO→ES。
- **主要结果**：
  - **Pass@K提升**：在Easy Setting中，ES平均Pass@1/16/32均优于Base与GRPO；GRPO在18次比较中有15次Pass@16/32低于Base，而ES未出现此退化。
  - **Hard Setting数学基准**：ES在AIME24/25、AMC23、MATH-500上Pass@16/32整体优于GRPO；ES→GRPO在AIME24上获得最高Pass@32（76.7 vs GRPO的66.7）。
  - **帕累托前沿**：顺序组合在三个代表性设置中均引入非支配点，扩展了Pass@1–Pass@K权衡空间。
  - **参数漂移与遗忘**：ES相对初始参数的L₂距离为GRPO的40.7–44.1倍；但保留约7–22%的大更新（幅度>1.5×10⁻³）即可维持几乎全部任务性能。held-out基准上ES平均Pass@32变化为正，而GRPO为负。
  - **超参数敏感性**：z-score归一化显著优于无归一化；两点估计在GSM8K上与单点估计无差异；种群大小随模型规模增大而可降低。

## 相关工作脉络
1. **GRPO（Shao et al., 2024）**：主流基于verifier奖励的RL后训练方法，通过组内相对优势与PPO clipped目标优化；本文作为主要梯度类对比基线，指出GRPO存在熵坍塌与Pass@K退化。
2. **ES for LLM（Qiu et al., 2026; Sarkar et al., 2026）**：证明ES可直接优化全参数LLM并以结果奖励微调；本文在此基础上系统分析ES的动态特性，将其定位为独立范式而非GRPO的内存高效替代品。
3. **熵坍塌分析（Cui et al., 2025; Jin et al., 2026）**：揭示RL策略梯度优化中政策熵下降机制；本文借用该现象作为GRPO性能受限的解释，并通过ES种群多样性理论提供缓解途径。
4. **参数漂移与遗忘（Abdi et al., 2026; Hoy et al., 2026）**：前者报告ES导致灾难性遗忘，后者认为漂移与行为退化关系不明确；本文通过大样本多任务实验证明漂移不等于遗忘，并指出功能稀疏性。
5. **零阶优化（ZO）与两点估计（Malladi et al., 2023; Zhang et al., 2024）**：在监督任务中两点估计可降方差；本文证明在自回归推理任务中，因响应重生成削弱正负扰动协方差，两点估计无优势。
6. **Pass@K评估（Yue et al., 2025; Wu et al., 2025）**：指出RL后训练可能提升单样本准确率但降低多采样成功率；本文采用Pass@K作为核心指标，揭示ES在此方面的优势。

## 局限性与未来方向
- **模型规模限制**：实验仅涵盖0.5B–7B参数模型，未验证ES在更大规模（如百亿参数）LLM上的表现与超参数缩放规律。
- **超参数泛化性**：奖励归一化、种群大小等建议基于特定任务（GSM8K、DeepScaleR），在其他推理任务（如代码生成、开放问答）中的有效性待检验。
- **持续学习场景未深入**：论文提及ES在多任务持续适应下的能力保持尚不明确，参数高效更新（如低秩适配器）能否抑制有害漂移仍需探索。
- **内存与计算效率量化不足**：虽强调ES无需反向传播，但未与GRPO进行严格的GPU内存占用、吞吐时间对比。
- **未来方向**：扩展ES在持续学习中的应用、研究参数高效变体以减少有害漂移、利用ES激励低概率正确路径的机制以提升推理覆盖。

## 研究启发与可借鉴点
- **评估指标设计**：采用Pass@K（而非仅Pass@1）能更全面反映推理后训练对多样化解题路径的保留情况，避免单一准确率误导。
- **顺序混合训练策略**：将不同优化器（如ES与GRPO）按序组合，可在不增加计算预算的前提下拓展帕累托前沿，为多目标优化提供简单有效的工程实践。
- **功能稀疏性分析**：通过幅度阈值筛选关键参数更新，可解释大漂移模型的性能保持机制，并为模型压缩、更新稀疏化提供参考。
- **超参数缩放规律**：发现模型规模越大所需种群规模越小，这一规律可指导大规模ES训练的资源分配。
- **估计器选择依据**：在自回归生成任务中，两点估计因响应重生成导致协方差弱化，应优先考虑单点估计；该洞见可推广至其他零阶优化应用场景。

## 关键术语表
- **Evolution Strategies (ES)**：一种无需反向传播的群体优化方法，通过扰动参数、前向评估奖励并聚合加权扰动来更新模型。
- **Group Relative Policy Optimization (GRPO)**：基于verifier奖励的强化学习后训练方法，利用组内相对优势与PPO裁剪目标优化策略。
- **Pass@K**：在K次独立采样中至少产生一个正确响应的概率，用于评估推理能力覆盖广度。
- **Entropy Collapse**：策略梯度优化过程中策略分布熵持续下降，导致响应多样性降低的现象。
- **Parameter Drift**：后训练后模型参数相对于初始参数的总体位移程度，常以L₂距离度量。
- **Functional Sparsity**：尽管参数整体发生较大变化，但性能增益仅集中在少量大范数更新上的性质。
- **Jensen–Shannon (JS) Diversity**：基于信息散度的多样性度量，本文用于刻画ES种群内策略分布的差异。
- **Zeroth-Order (ZO) Estimator**：仅通过函数值评估而非梯度信息估计优化方向的方法，包括单点与两点估计。

## 可复现要素
- **数据集**：GSM8K、DeepScaleR Preview、GPQA Diamond、AIME24/25、AMC23、MATH-500、CommonsenseQA、HotpotQA、Countdown、MBPP；均来自公开来源（Hugging Face等），部分使用MIT、CC BY-SA等许可证。
- **代码**：已开源，链接为 https://github.com/yunpengba7/understanding-es。
- **权重**：论文未提供预训练模型权重开源声明，但使用开源基础模型（Qwen2.5、Llama-3.2、DeepSeek-R1-Distill-Qwen）。
- **关键超参数**：扰动尺度σ=1.5×10⁻³，更新尺度α=2.5×10⁻⁴，种群大小N=32（Easy/Hard Setting），学习率1.0×10⁻⁶（GRPO），温度τ=0（Easy）/0.6（Hard），响应长度上限2048（Easy）/8192（Hard）。
