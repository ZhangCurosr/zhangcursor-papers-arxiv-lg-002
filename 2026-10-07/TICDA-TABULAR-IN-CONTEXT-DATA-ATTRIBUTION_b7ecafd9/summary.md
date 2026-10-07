---
title: "TICDA-TABULAR-IN-CONTEXT-DATA-ATTRIBUTION"
source: https://arxiv.org/pdf/2610.07996v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 17:43:27"
field: "表格基础模型的可解释性与数据归因"
keywords: ["tabular foundation models", "in-context learning", "data attribution", "influence functions", "context curation", "active learning", "label error detection"]
innovations: ["提出TICDA：在TFM冻结表征上拟合岭回归代理，单次前向传递推导闭式解的影响力得分，绕过ICL无参数更新的障碍", "设计自影响与查询影响两种互补度量，分别用于标注错误检测和上下文裁剪，并验证跨TFM属性得分迁移", "推导基于TICDA的主动学习获取策略，利用Sherman-Morrison闭式更新实现高效batch选择"]
benchmarks: ["TabArena (38 datasets)", "TabICLv2", "TabDPT 1.2", "TabPFN-3", "TabPFN-3.5"]
---

# 论文速读：TICDA-TABULAR-IN-CONTEXT-DATA-ATTRIBUTION

## 一句话总结
论文提出 TICDA，一种针对表格基础模型（TFMs）上下文学习中演示样本影响力的高效归因方法：通过在单次前向传递中提取的冻结表征上拟合岭回归代理，利用影响函数推导出闭式解的影响力得分，实现计算代价极低的标注错误检测、上下文裁剪与主动学习数据获取。

## 研究问题与动机
- **TFM 上下文推理的"黑箱"问题**：表格基础模型（如 TabPFN、TabICL、TabDPT）在推理时不更新参数，而是通过上下文中提供的带标签演示（demonstrations）进行上下文学习（ICL），但个体演示对预测的影响机制缺乏理解。
- **标准数据归因方法无法直接迁移到 ICL 设置**：基于梯度的影响函数要求对模型参数求导，而 TFM 推理过程中参数固定；基于重采样的方法（如 Data Shapley/DemoShapley）需要对组合数量的上下文子集进行完整前向传递，成本过高。
- **低质量演示的潜在危害**：上下文中混入标注错误、冗余或低质量示例会无声地降低预测性能，或引入不必要的计算开销（TFM 推理成本随演示数量二次增长）。
- **实际部署需求**：上下文通常由可用数据拼凑而成，亟需一种高效归因方法以支持标注错误检测、上下文裁剪、跨模型迁移和主动学习等应用。

## 核心贡献（创新点）
1. **提出 TICDA 方法**：首次在单次前向传递中，通过训练于 TFM 冻结表征上的线性岭回归代理，推导闭式解的影响力得分，避免逐样本删除的高计算成本；与已有工作的本质区别在于将影响函数从"模型参数空间"转移到"表征空间"，适配无参数更新的 ICL 范式。
2. **设计两种互补影响力度量（自影响与查询影响）**：自影响 $T_i$ 衡量演示对其自身损失的一阶变化，用于标注错误检测；查询影响 $u_i$ 衡量演示对多个查询的平均支持度，用于上下文裁剪；与已有方法（如 DETAIL）的本质区别在于损失定义不包含正则项，对应真正的留一影响而非删除正则化器。
3. **验证跨模型属性得分迁移能力**：在 TabICLv2 或 TabDPT 上计算的归因得分可直接用于裁剪 TabPFN-3 / TabPFN-3.5 的上下文，且 50% 剔除后仍比完整上下文准确率高出超 4 个百分点；与已有工作相比，这是首次系统验证 TFM 间归因分数可迁移性。
4. **推导基于 TICDA 的主动学习获取策略**：通过评估候选样本加入后对代理损失的预期下降（闭式 Sherman-Morrison 更新），实现低标注预算下高效的 batch 选择；该方法比随机采样和熵采样更优，尤其在数据受限场景（≤512 标签）。
5. **全面基准评估**：在 38 个 TabArena 数据集上系统比较 TICDA 与 LOO、DemoShapley、IG、DETAIL 四种基线，TICDA 是唯一同时实现顶级忠实度（faithfulness）、标注错误检测 AUC 和低计算成本的方法。

## 方法详解
**核心思想**：将 TFM 的上下文学习视为在冻结表征空间中的线性回归任务，通过影响函数近似留一影响。

**步骤 1 — 表征提取与岭回归代理拟合**：
- 从冻结 TFM 中提取每个演示的行表征 $m_j = \phi(x_j | \mathcal{D})$（取特定隐藏层输出），组成矩阵 $M \in \mathbb{R}^{n \times d}$，标签矩阵 $Y \in \mathbb{R}^{n \times K}$。
- 拟合岭回归代理：$\widehat{B} = H^{-1} M^\top Y$，其中 $H = M^\top M + \lambda I_d$，$\lambda=10$（默认固定，无需数据集调优）。

**步骤 2 — 固定表征近似（Fixed-Representation Simplification）**：
- 假设删除单个演示不影响其余表征：$\phi(\cdot|\mathcal{D}_{\setminus i}) \approx \phi(\cdot|\mathcal{D})$（附录 C.1 实证支持：4096 演示时相对变化约 0.5%，随 $n$ 增大按 $1/n$ 衰减）。
- 此近似将问题转化为固定设计矩阵下的岭回归，使影响函数公式可直接应用，且仅需单次前向传递。

**步骤 3 — 影响力得分推导**：
- 对演示 $i$ 施加权重微扰 $\epsilon$，求导得系数变化率 $d\widehat{B}_\epsilon/d\epsilon|_{\epsilon=0} = -H^{-1}m_i r_i^\top$，其中残差 $r_i = \widehat{B}^\top m_i - y_i$。
- **成对影响力**：$s_{iq} = (m_q^\top H^{-1} m_i)(r_q^\top r_i)$，符号正表示该演示有助于降低查询损失。
- **自影响（Self-influence）**：$T_i = s_{ii} = h_i \|r_i\|_2^2$，其中 $h_i = m_i^\top H^{-1} m_i$ 为岭杠杆值；$T_i$ 大意味着该演示的标注与上下文不一致，可用于标注错误检测。
- **查询影响（Query-influence）**：$u_i = \frac{1}{|\mathcal{V}|}\sum_{q \in \mathcal{V}} s_{iq}$，$u_i$ 负值表示删除该演示可改善验证集性能，用于上下文裁剪。

**迭代裁剪变体 TICDA-IT**：每轮剔除 5% 最低 $u_i$ 演示后重新拟合代理并重排，适用于需要深度裁剪的场景。

**主动学习获取策略**：候选样本 $x$ 加入后，利用 Sherman-Morrison 公式闭式更新 $H^{-1}$ 和 $\widehat{B}$，评估其对验证集代理损失的预期下降，取各可能类别中的最大下降作为 score，贪心选择批次。

## 实验与结果
- **数据集**：38 个 TabArena 分类数据集，上下文最多 4096 个演示，20%/40% 随机标签噪声注入评估。
- **骨干模型**：TabICLv2（512 维表征，取 in-context learning 层之前）和 TabDPT 1.2（32 层 Transformer，取第 16 层之后）。
- **基线方法**：Leave-one-out (LOO)、DemoShapley（8 次 Monte Carlo 采样近似）、Integrated Gradients (IG)、DETAIL。

**主要结果**：

| 指标 | TabDPT 最佳 | TabICL 最佳 |
|---|---|---|
| Faithfulness (vs LOO Spearman) | DemoShapley 0.91 *** / TICDA 0.79 | **TICDA 0.67** |
| 标注错误检测 AUC | DemoShapley 0.88 / TICDA 0.88 | **TICDA 0.89 ***（显著超越 LOO 0.80）|
| 运行时间 | TICDA/DETAIL < 3s（两个数量级快于 LOO/DemoShapley/IG）| 同左 |

- **上下文裁剪**：TICDA-IT 在 20%/40% 噪声下均能保持或提升平衡准确率；40% 噪声下剔除约 40-45% 演示后性能达到峰值，比完整上下文更好；TICDA-IT 始终优于单次裁剪的 TICDA。
- **跨模型迁移**：用 TabICLv2 或 TabDPT 计算的得分裁剪 TabPFN-3/3.5 上下文，50% 剔除后平衡准确率约 64%，比随机剔除的基线（~57%）高出超 4 个百分点。
- **主动学习**：TICDA 在完整预算（27 数据集）下 AULC = 68.35%（超越熵采样 67.56%），在数据受限（≤512 标签）下 AULC = 69.02%（超越熵采样 67.51%），差异均达统计显著（p<0.01）。
- **消融**：自影响 vs 查询影响——自影响显著提升忠实度（0.33→0.67）和 AUC（0.86→0.89）；表征维度降至 20% 对性能无显著影响。

## 相关工作脉络
- **Influence Functions (Koh & Liang, 2017)**：经典梯度基影响函数定义，TICDA 的核心数学框架来源，但原方法依赖可微参数，无法直接应用于参数固定的 ICL 场景。
- **Data Shapley / DemoShapley (Ghorbani & Zou, 2019; Xie et al., 2025)**：重采样基线，通过子集排列估计演示边际贡献；DemoShapley 针对 LLM  few-shot 设置设计，对长上下文 TFM 不可行（指数级前向传递）。
- **DETAIL (Zhou et al., 2024)**：概念最接近的竞品，同样在隐藏表征上拟合岭回归代理并应用影响函数；但与 TICDA 的关键差异在于（1）训练 regime 不同：LLM 少演示高维 vs TFM 多演示低维；（2）损失定义包含全量正则项，导致梯度被全局偏移项污染，改变排序；（3）未适配固定表征近似。
- **TabPFN / TabICL / TabDPT 系列**：当前主流表格基础模型，TICDA 直接在其冻结表征上工作，不修改模型架构。
- **In-Context Learning 可解释性 (Rundel et al., 2024; von Oswald et al., 2023)**：Rundel 发现数据估值可选择出更优上下文；von Oswald 提出 Transformer 在 ICL 中隐式实现梯度下降，TICDA 的代理视角部分继承此思路但面向表格场景重新设计。
- **Active Learning 与 ICL 结合**：传统主动学习依赖模型梯度或不确定性，本文首次将数据归因分数直接用于 ICL 设置的候选获取，无需额外训练循环。

## 局限性与未来方向
- **固定表征近似**：删除演示后其余表征实际会有微小变化（~0.5% at n=4096），本文未建模此效应；对于极短上下文（n 接近或小于表征维度 d），杠杆值接近 1，一阶近似误差增大。
- **仅验证到 4096 演示**：TICDA 面向数千演示的长上下文设计，但忠实度评估上限为 4096；更长上下文（如 8192+）需更多验证。
- **仅适用于分类任务**：当前方法基于 one-hot 标签和岭回归，未扩展到回归或多模态场景。
- **代理模型表达能力有限**：线性探针可能无法充分捕捉 TFM 表征中的非线性关系，在高复杂度任务上或有偏差。
- **代码尚未公开**：论文声明"将在接受后开源代码"，目前无公开实现可供复现。

## 研究启发与可借鉴点
1. **"表征空间替代参数空间"的影响函数范式**：将影响函数从可微参数转移到冻结中间表征上拟合线性代理，巧妙绕过 ICL 无参数更新的障碍；此思路可迁移至 LLM 检索增强（RAG）中的文档片段归因、多模态模型的视觉/文本片段重要性分析。
2. **自影响 vs 查询影响的场景适配**：自影响 $T_i$（对自身的预测一致性）适合异常/错误检测；查询影响 $u_i$（对目标集合的支持度）适合样本选择/裁剪；两者分工明确，可作为通用归因工具的设计指南。
3. **固定表征近似在大数据量下的合理性论证**：Appendix C.1 定量展示表征变化随 $1/n$ 衰减，为后续方法提供可直接引用的理论支撑和实验范式。
4. **主动学习与数据归因的结合路径**：利用 Sherman-Morrison 闭式更新实现候选加入后的快速 re-score，避免每次重新拟合代理，使主动学习批次内循环高效可行；此策略可迁移至其他需要在线更新的代理模型场景。
5. **跨模型归因迁移的实验设计**：在源模型上计算归因、直接用于目标模型裁剪，为跨架构的知识/数据价值评估提供了简洁有效的 benchmark 范式。

## 关键术语表
- **Tabular Foundation Models (TFMs)**：基于 Transformer 架构、在表格数据上进行上下文学习的预训练模型（如 TabPFN、TabICL、TabDPT），推理时冻结参数，通过上下文中的演示进行预测。
- **In-Context Learning (ICL)**：模型在不更新参数的情况下，利用上下文中提供的带标签示例（demonstrations）直接进行推理的学习范式。
- **Self-influence ($T_i$)**：演示对自身预测损失的一阶影响，等于岭杠杆值乘以残差平方，用于检测标注错误的演示。
- **Query-influence ($u_i$)**：演示对一组查询样本的平均影响，负值表示删除该演示可改善整体预测，用于上下文裁剪。
- **Fixed-Representation Approximation**：假设删除单个演示后，其余演示和查询的 TFM 表征不变（$\phi(\cdot|\mathcal{D}_{\setminus i}) \approx \phi(\cdot|\mathcal{D})$），使影响函数可闭式求解且仅需单次前向传递。
- **Data Attribution / Training Data Attribution (TDA)**：量化单个训练样本对模型行为或性能影响的分析框架，本文扩展至 ICL 演示归因。
- **Faithfulness**：归因方法计算的影响力排序与实际留一影响（LOO）之间的 Spearman 相关系数，衡量归分与真实因果效应的对齐程度。
- **Ridge Surrogate**：在 TFM 冻结表征上拟合的线性岭回归模型，作为影响函数可微参数的替代。

## 可复现要素
- **数据集**：38 个 TabArena 分类数据集（Erickson et al., 2025），公开可用；预处理和评估协议遵循 TabArena 基准。
- **代码**：论文声明"将在接受后公开"，当前未开源。
- **模型权重**：TabICLv2、TabDPT 1.2、TabPFN-3/3.5 均为公开可用模型。
- **关键超参**：岭正则化系数 $\lambda = 10$（所有数据集固定，无需调优）；代表维度 $d=512$（TabICLv2）；背景扰动步数（IG 使用 32 步）；DemoShapley 使用 8 次 Monte Carlo 排列。
- **随机种子**：消融实验中使用 3 个随机投影种子取平均；主动学习实验重复 $R=5$ 次。
