---
title: "WHAT-DOES-AN-LLM-LEARN-FROM-REINFORCE-MENT-LEARNING-A-MECHAN"
source: https://arxiv.org/pdf/2609.15064v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:59:09"
---

# 论文速读：WHAT-DOES-AN-LLM-LEARN-FROM-REINFORCE-MENT-LEARNING-A-MECHAN

## 一句话总结
本文提出 Fixed-SAE Track 框架，通过在基础模型与所有 RL 检查点的激活上共享并固定 SAE 字典，严谨追踪表征漂移；研究发现 RL 主要放大少数晚期层的格式/推理脚手架特征而非重塑问题内容，且将这些特征注入基础模型可恢复约 80% 的 RL 性能增益，表明 RL 主要激发模型已有能力而非创造全新推理路径。

## 研究问题与动机
1. RL 已成为 LLM 后训练的标准环节，但现有研究几乎全部停留在行为评估层面，缺乏从表征内部解释“RL 究竟给模型带来了什么”的机制性工具。
2. 传统 SAE 独立训练会导致特征索引在不同检查点间随机分配，无法直接用于跨训练步的表征比对与新兴特征检测。
3. 现有跨检查点追踪方法（如 SAE Track）依赖逐检查点微调且继承前序字典，难以严格分离“原有特征增强”与“真正新特征涌现”。
4. 行为学结论存在矛盾：部分研究认为 RL 仅锐化基础模型已有的采样分布，另一部分认为 RL 能扩展推理边界；亟需表征级证据厘清二者关系。

## 核心贡献（创新点）
1. **提出 Fixed-SAE Track 框架**：在池化的基础模型与所有 RL 检查点激活上训练单一共享 TopK SAE 并冻结特征方向。**与已有工作独立训练 SAE 导致特征索引不可比的区别在于，
