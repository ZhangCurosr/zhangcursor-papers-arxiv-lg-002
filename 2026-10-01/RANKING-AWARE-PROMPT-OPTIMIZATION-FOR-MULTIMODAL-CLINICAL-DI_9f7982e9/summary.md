---
title: "RANKING-AWARE-PROMPT-OPTIMIZATION-FOR-MULTIMODAL-CLINICAL-DI"
source: https://arxiv.org/pdf/2609.40361v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-04 00:20:55"
---

# 论文速读：RANKING-AWARE-PROMPT-OPTIMIZATION-FOR-MULTIMODAL-CLINICAL-DI

## 一句话总结
针对临床数据严重类别不平衡导致准确率误导的问题，本文提出 Ranking-PE（成对级 Pareto 提示进化），通过将提示优化搜索矩阵的行结构从“单样本正确性”替换为“正负样本对排序”，使优化目标与评估指标统一为 AUROC，在 MIMIC 三个疾病任务上显著超越传统准确率驱动的提示优化方法。

## 研究问题与动机
1. **临床决策依赖排序能力而非阈值准确率**：筛查、分流与随访需在 ROC 曲线上选取不同操作点，AUROC 等阈值无关指标更能反映模型区分正负病例的能力；在重度类别不平衡下，恒定多数预测器可取得 90%+ 准确率但临床完全无效。
2. **现有提示进化系统隐含准确率偏差**：GEPA 等反射式 Pareto 提示优化默认以实例级正确性构建分数矩阵，列平均值即准确率，导致搜索目标与临床实际评估标准错位。
3. **准确率优化可能侵蚀底层排名信号**：论文实证发现，Accuracy-PE 在极端不平衡疾病上不仅无法提升 AUROC，反而会使基础模型的排序能力下滑。
4. **多模态临床适配管线缺乏统一的排序视角**：现有 MLLM 医学适配流程通常以 accuracy 为目标进行 SFT 或 PE，未将排名一致性贯穿到提示搜索的每一个决策层。

## 核心贡献（创新点）
1. **提出成对级 Pareto 提示进化（Ranking-PE）**：将 Pareto 分数矩阵的行定义从实例正确性替换为正负样本对排序事件，利用 Wilcoxon–Mann–Whitney 恒等式使列均值直接等于经验 AUROC，全程无需梯度平滑或代理损失。
2. **全链路对齐排序目标**：在提示进化的三个关键层（Pareto 支配过滤、反射器反馈、最终候选选择）同步植入排序信号，彻底纠正准确率驱动下的搜索偏差。
3. **设计排序形状化反馈机制**：为反射 LLM 引入临床错误类型（FN 标注为 clinically dangerous 以体现非对称代价）、置信度幅度分桶与跨样本排序上下文，使提示迭代方向与 AUROC 提升严格对齐。
4. **提供端到端多模态临床诊断基准与配方**：在 MIMIC-IV 三个疾病任务上系统验证 Qwen3-VL-8B 与 MedGemma-4B，证明视觉域适配是提示进化的必要基础，并将反射式提示进化从纯文本成功扩展至多模态临床决策。

## 方法详解
- **连续排序分数提取**：对标准解码输出，在答案 token 位置读取 top-k logprobs，计算 log-odds 分数 $s_\Phi(x) = \log p(\text{Yes}|x) - \log p(\text{No}|x)$，零额外 rollout 成本。
- **成对级 Pareto 矩阵**：设验证集正样本集 $P$、负样本集 $N$，定义成对分数 $r(a,b)=\mathbf{1}[a>b]+\frac{1}{2}\mathbf{1}[a=b]$。构造矩阵 $\tilde{M} \in \{0, 0.5, 1\}^{(|P|\cdot|N|)\times K}$，第 $(i,j,k)$ 格为 $r(s_\Phi(x_i^+), s_\Phi(x_j^-))$。由 Wilcoxon–Mann–Whitney 恒等式，列均值即为验证集 AUROC；Pareto 支配关系据此重新定义（候选 A 支配 B 当且仅当在所有成对上得分不低于 B 且至少一处严格更高）。
- **排序形状化反馈 $\mu_f$**：替换原正确性位，以换行拼接三条信号：(i) 临床错误类型（TP/TN/FN/FP，FN 附加危险标注）；(ii) 置信度幅度（带符号 $s_\Phi$ 及 low/moderate/high/extremely high 分桶）；(iii) 跨样本排序上下文 $\rho^+,\tau^-$（当前样本相对于上一候选验证分布的分位数统计）。
- **两阶段流水线**：Stage 1 为医学域适配 SFT（LoRA + 视觉编码器微调/冻结）；Stage 2 为训练自由的提示进化，维持候选池 $\{\Phi_k\}$，每轮基于 $\tilde{M}$ 筛选 Pareto 前沿，经 $\mu_f$ 驱动反射器生成新提示并回池，最终按列均值 argmax 选出 $\Phi^*$。
- **复杂度控制**：成对分数通过查表复用单样本验证结果，构建开销 $O(|P|\cdot|N|)$ CPU 查找；超出 `max_pairs` 时均匀无偏子采样，保持优化目标不变且仅增加估计方差。

## 实验与结果
- **数据集**：MIMIC-CXR 链接 MIMIC-IV（Atelectasis 96.2%、Cardiomegaly 78.7%、Consolidation 68.9% 阳性率），患者 disjoint 划分。
- **模型**：Qwen3-VL-8B、MedGemma-4B（医学预训练）、Gemma3
