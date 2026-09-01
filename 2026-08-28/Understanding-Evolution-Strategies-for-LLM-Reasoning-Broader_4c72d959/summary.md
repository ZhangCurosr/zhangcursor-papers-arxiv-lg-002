---
title: "Understanding-Evolution-Strategies-for-LLM-Reasoning-Broader"
source: https://arxiv.org/pdf/2608.27351v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 01:17:43"
field: "大语言模型推理后训练"
keywords: ["Evolution Strategies", "GRPO", "LLM reasoning", "Pass@K", "entropy collapse", "post-training"]
innovations: ["证明ES种群多样性带来更高Pass@K且避免熵坍缩", "提出ES-GRPO串行组合策略兼顾单样本与多样本准确率", "揭示ES功能性稀疏机制解释大参数漂移不必然导致遗忘"]
benchmarks: ["GSM8K", "GPQA", "MBPP", "AIME24", "AIME25", "AMC23", "MATH-500", "DeepScaleR"]
---

# 论文速读：Understanding-Evolution-Strategies-for-LLM-Reasoning-Broader

## 一句话总结
本文系统研究了进化策略（ES）在大语言模型推理后训练中的优化行为，理论证明并经验验证ES能比GRPO提供更广泛的推理覆盖（更高的Pass@K），且大参数漂移不必然导致灾难性遗忘；同时提出ES与GRPO的串行组合策略以兼顾Pass@1与Pass@K优势。

## 研究问题与动机
1. **核心问题**：ES作为LLM推理后训练的内存高效替代方案，其优化行为与前向传播方法（如GRPO）存在何种本质差异？
2. **动机一**：GRPO等基于策略梯度的方法易出现熵坍缩，导致Pass@K性能下降；ES通过种群搜索是否能避免此问题并保持更广泛的推理覆盖？
3. **动机二**：ES会产生远大于GRPO的全模型参数漂移，但这是否必然导致灾难性遗忘？参数更新是否存在功能性稀疏？
4. **动机三**：ES的超参数设计（奖励归一化、扰动尺度、种群大小、估计器选择）如何影响其有效性与可扩展性？

## 核心贡献（创新点）
1. **首次系统揭示ES在推理覆盖上的优势**：理论证明验证器投影的Jensen-Shannon多样性有助于提升Pass@K，经验验证ES在GSM8K和DeepScaleR上优于GRPO且无熵坍缩。
2. **提出ES-GRPO串行组合策略**：设计ES→GRPO和GRPO→ES两种串行训练顺序，在相同更新预算下拓展了Pass@1-Pass@K的Pareto前沿。
3. **揭示ES的功能性稀疏机制**：发现ES的性能增益仅集中于小部分大范数更新（主要在LayerNorm和注意力投影），大参数漂移≠广泛功能改变≠灾难性遗忘。
4. **明确ES超参数设计原则**：证明z-score奖励归一化是关键；离散推理任务中两点估计器相比一点估计器无优势；大模型所需种群规模更小（N=16即可接近N=64基准）。

## 方法详解
1. **ES种群多样性与Pass@K的理论联系**：
   - 定义提示条件Fisher信息 $\mathcal{I}_x(\theta) = \mathbb{E}[s_\theta(Y|x)s_\theta(Y|x)^\top]$
   - 证明种群JS多样性 $\mathrm{JS}_N^{\mathrm{pol}}(x) = \frac{\sigma^2}{2}(1-\frac{1}{N})\mathrm{tr}\mathcal{I}_x(\theta) + O(\sigma^4)$
   - 通过数据处理不等式建立 $\mathrm{JS}_N^{\mathrm{succ}} \leq \mathrm{JS}_N^{\mathrm{pol}}$
   - 证明通配成功概率 $P_N^{\mathrm{pop}} = 1 - \prod(1-p_i) \geq 1-(1-\bar{p})^N = P_N^{\mathrm{same}}$
   - 给出中心策略Pass@K改进的充分条件：$\mathbb{E}[D_{\mathrm{KL}}(B_w||B_{\theta^+})] \leq \varepsilon_{\mathrm{succ}}$ 且 $m_K > K\sqrt{\varepsilon_{\mathrm{succ}}/2}$

2. **ES更新机制**：
   - 高斯平滑目标 $F_\sigma(\theta) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)}[F(\theta + \sigma\epsilon)]$
   - 一点估计器 $\widehat{g}_{\mathrm{ES}}^{\mathrm{raw}} = \frac{1}{N\sigma}\sum_{i=1}^N R_i \epsilon_i$
   - 标准化后更新：$\widehat{d}_{\mathrm{ES}} = \frac{1}{N}\sum_{i=1}^N z_i \epsilon_i$，$\theta^+ = \theta + \alpha\widehat{d}_{\mathrm{ES}}$

3. **串行组合策略**：
   - ES→GRPO：先用ES训练一半预算（扩大Pass@K覆盖），再用GRPO优化Pass@1
   - GRPO→ES：先用GRPO提升单样本准确率，再用ES恢复多样性

4. **ES超参数设计**：
   - 奖励归一化：对种群内奖励做z-score标准化，使更新依赖相对而非绝对奖励尺度
   - 扰动尺度σ：过小易过拟合局部最优，过大导致探索过宽破坏训练
   - 种群大小N：随模型规模增大而减小，大模型中任务改进扰动更密集

## 实验与结果
**数据集与模型**：
- Easy Setting：GSM8K训练，Qwen2.5-1.5B/3B/7B-Instruct和Llama-3.2-3B-Instruct，2轮
- Hard Setting：DeepScaleR训练，DeepSeek-R1-Distill-Qwen-1.5B，1轮
- 评估基准：GSM8K、GPQA、MBPP、CSQA、HotpotQA、Countdown、AIME24/25、AMC23、MATH-500

**主要结果**：
- **Easy Setting（GSM8K→多基准）**：ES在Qwen2.5-1.5B上Pass@1从41.0→41.5，Pass@16从75.4→76.0，Pass@32从80.2→80.9；GRPO Pass@16/32在15/18次比较中低于base
- **Hard Setting（DeepScaleR→数学基准）**：ES在AIME24上Pass@32达70.0（base 66.7，GRPO 66.7）；MATH-500上ES达98.2（base 98.0）
- **串行组合优势**：ES→GRPO在Hard Setting数学平均上Pass@32达79.2（base 77.4）；GRPO→ES在GPQA上提供中间权衡

**最强结果**：在DeepScaleR Hard Setting中，ES→GRPO组合在AIME24上Pass@1达29.0（显著高于ES的22.8），Pass@32达76.7（高于GRPO的66.7），展示互补优势。

## 相关工作脉络
1. **GRPO (Shao et al., 2024)**：主流RL后训练方法，基于组内相对优势进行PPO风格裁剪更新；本文对比对象，发现GRPO易出现熵坍缩和大K Pass@K退化。
2. **熵坍缩研究 (Cui et al., 2025; Petrenko et al., 2026; Jin et al., 2026)**：揭示RLVR中策略熵持续下降机制；本文在此基础上提出ES通过种群多样性避免此问题。
3. **ES扩展研究 (Qiu et al., 2026; Sarkar et al., 2026; Abdi et al., 2026)**：证明ES可扩展至全参数LLM；Abdi等将大参数漂移与灾难性遗忘关联；本文通过更大规模实验反驳此归因。
4. **零阶优化 (MeZO, Malladi et al., 2023; ESSA, Korotyshova et al., 2025)**：前向传播微调方法；本文论证两点估计在推理任务中无优势，因自回归重采样削弱正负扰动协方差。
5. **Pass@K分析 (Yue et al., 2025; Wu et al., 2025)**：发现RL后训练可提升Pass@1但降低大K性能；本文提供ES作为避免此现象的替代方案。

## 局限性与未来方向
1. **超参数调优仍依赖经验**：扰动尺度σ和学习率α的选取缺乏系统性指导，需根据具体任务调整。
2. **串行组合顺序依赖任务**：ES→GRPO和GRPO→ES在不同基准上表现不同（如AIME24 vs AIME25），最优顺序需实验选择。
3. **长期持续学习未知**：多轮任务适应性训练下的能力保持机制未充分研究，ES在大范围任务序列中的表现待探索。
4. **理论假设较强**：Pass@K改进定理依赖于Fisher信息有限、支撑集共同、平滑性等假设，实际LLM是否完全满足需验证。
5. **未来方向**：开发参数高效ES变体以抑制有害漂移；探索ES在长周期持续学习中的行为；结合ES多样性优势设计更好采样策略。

## 研究启发与可借鉴点
1. **Pop@K评估框架值得推广**：单一Pass@1指标可能掩盖多样性损失，建议将Pass@K/Maj@K纳入RLVR方法评估标准。
2. **串行组合策略可扩展**：ES与GRPO的互补性启发其他方法组合（如SFT→ES、DPO→ES），可作为通用增强模板。
3. **功能性稀疏分析范式**：通过幅度阈值化保留大部分性能的方法，为理解RLVR更新机制提供新视角，可应用于其他后训练方法。
4. **两点估计器适用边界**：证明自回归重采样导致正负扰动协方差不足，为ZO优化在推理任务中的应用划定边界。
5. **种群规模-模型规模反比关系**：大模型可用更小种群实现稳定训练，为资源受限场景提供部署指导。

## 关键术语表
- **Pass@K**：从模型采样K次响应，至少一次正确的概率，衡量多路径搜索能力
- **熵坍缩 (Entropy Collapse)**：策略分布在学习过程中过度集中，导致推理多样性下降
- **Jensen-Shannon多样性**：种群内不同策略分布间的JS散度，量化参数扰动引入的响应多样性
- **功能性稀疏 (Functional Sparsity)**：ES大量参数更新中仅小部分大范数更新对性能有实质贡献
- **验证器投影**：将连续策略分布通过二值正确性验证器映射为伯努利分布的过程
- **一点估计器 (One-Point Estimator)**：ES中使用单次前向评估计算梯度近似，$\hat{g} = \frac{1}{N\sigma}\sum R_i \epsilon_i$
- **两点估计器 (Two-Point Estimator)**：通过正反扰动奖励差值估计梯度，$\hat{g} = \frac{1}{2N\sigma}\sum [R_i^+ - R_i^-]\epsilon_i$
- **串行组合训练**：先使用一种方法训练部分预算，再用另一种方法完成剩余训练的策略

## 可复现要素
- **代码开源**：GitHub https://github.com/yunpengba7/understanding-es
- **权重公开**：论文未提及模型权重开源，但提供训练配置细节
- **数据集**：GSM8K、DeepScaleR Preview、GPQA Diamond、MBPP sanitized test、CommonsenseQA、HotpotQA、Countdown、AIME24/25、AMC23、MATH-500（均为公开数据集）
- **关键超参**：
  - 扰动尺度 σ = 1.5×10⁻³
  - 更新尺度 α = 2.5×10⁻⁴
  - 种群大小 N = 32（Easy）/ 可降至16（大模型）
  - 学习率（GRPO）：1.0×10⁻⁶
  - 响应限制：2048 tokens（Easy）/ 8192 tokens（Hard）
- **硬件**：NVIDIA A100-SXM4-80GB GPU
- **评估设置**：每问题32次采样，温度0.6，K∈{16,32}
