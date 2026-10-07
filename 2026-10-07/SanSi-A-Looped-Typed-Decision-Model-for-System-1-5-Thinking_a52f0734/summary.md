---
title: "SanSi-A-Looped-Typed-Decision-Model-for-System-1-5-Thinking"
source: https://arxiv.org/pdf/2610.07730v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-07 17:41:13"
---

# 论文速读：SanSi-A-Looped-Typed-Decision-Model-for-System-1-5-Thinking

## 一句话总结
SanSi 将预训练循环语言模型转化为 Typed Decision Model，通过在同一组层上递归迭代（最多 8 轮）并配合 Proper Scoring Rule 多轮监督，使单一模型在 1～8 档计算预算下高效输出选项概率分布；在 10,027 项综合评测中较同尺寸单遍基线 SmolLM2-1.7B 提升 **13.5 个百分点**。

## 研究问题与动机
- 现有 LLM 决策任务多在单次前向（System 1，低延迟低能力）与自回归推理（System 2，高能力高延迟）之间二选一，缺乏兼顾效率与复杂泛化的中间机制。
- 循环语言模型（Looped LM）虽能提升难分布任务表现，但以往工作多关注生成质量，未系统解决“循环步数可调的统一决策框架”问题。
- 如何在极短训练成本下，让单一权重同时服务不同计算预算，并保持概率校准与远迁移能力？

## 核心贡献（创新点）
1. **提出 System 1.5 思考机制**：将固定层递归迭代定义为介于 System 1 与 System 2 之间的中间范式，以隐藏状态修正替代 token 级生成，显著降低推理开销。
2. **设计 Typed Decision Model 训练框架**：每轮循环后接 typed readout 并施加 Proper Scoring Rule 损失，使模型在多步演化中均获得明确监督信号。
3. **实现“一次训练、多档预算”调度**：训练时随机采样循环深度 $t \in [1,8]$，单一权重即可在推理时按需切换计算量，无需重新微调。
4. **系统刻画循环动态与校准规律**：量化逐轮准确率、答案稳定率、远/近迁移增益及 ECE 变化，指出知识类任务为当前架构瓶颈。

## 方法详解
- **循环 Backbone**：基于 SmolLM2 的循环语言模型，重复使用相同网络层进行 $T$ 次前向迭代，第 $t$ 轮的输入为第 $t-1$ 轮的隐藏状态，参数零新增。
- **Typed Readout**：每个循环结束后接入轻量级分类头，直接将当前隐藏状态映射为离散选项概率分布，跳过自回归解码。
- **多轮 Proper Scoring Rule 监督**：对每一轮 readout 输出施加严格分数损失（如 Brier Score / Log Loss），鼓励模型在不同循环深度下均给出合理置信度，而非仅在最后一步收敛。
- **随机深度训练**：训练时随机采样循环数 $t$，使模型同时学习 1～8 步的表征演化曲线，推理时可按延迟预算选择最大循环或提前停止。

## 实验与结果
- **数据集**：6 类任务（Classification、Multi-step reasoning、Uncertain evidence、Long documents、Sentence pairs、Knowledge），覆盖 20/59 数据源，测试集总计 **10,027** 项。
- **基线**：SmolLM2-1.7B（单遍）、Ouro-1.4B（单循环）、Qwen3.5-2B/4B、Kev-4B、Jev API。
- **核心结果**：SanSi L8 达 **72.0%**，较 SmolLM2-1.7B（58.4%）提升 **+13.5** 点（Bootstrap 95% CI [+12.5, +14.5]）；SanSi-2.6B L8 达 **75.8%**。Ouro-1.4B 单循环仅 58.6%，证明训练策略而非单纯增加循环是关键。
- **分类型增益**：知识（+17.6）、长文档（+16.7）、多步推理（+15.1）、句子对（+14.2）、不确定证据（+10.6）、分类（+3.1）；Far transfer 增益最大（+17.2）。
- **循环动态**：准确率逐轮递增（58.4→66.9→70.4→71.6→71.9→72.1→72.1→72.0），4 轮后趋稳；答案稳定率从 loop 1 的 56.7% 升至 loop 7 的 97.7%；若每题选最优 loop，理论上限达 **83.3%**。
- **效率与校准**：313 GPU-min（A6000）完成 1,000 步训练；2,000 步仅 +0.9 点且 ECE 恶化（0.099→0.163）。ECE 在 loop 3 最低（0.082），loop 8 略回升至 0.093；AUROC 从 0.760 升至 0.795；硬回答率从 27.1% 降至 17.5%。
- **强基线对比**：落后 Qwen3.5-4B（73.8%，+1.8 点）与 Kev-4B（74.3%，+2.4 点），但在 multi-step reasoning 上与其持平。

## 相关工作脉络
- **Looped / Recursive LMs**：本文将循环结构显式用于判别决策，区别于此前聚焦语言建模或生成质量的工作，强调“多档预算统一训练”的 Typed Decision 范式。
- **System 1 / System 2 推理**：SanSi 定位为 System 1.5，填补单次前向与 Chain-of-Thought 之间的性能-成本
