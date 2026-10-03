---
title: "LEARNING-CAUSAL-NORMALIZING-FLOWS-FROM-IN-COMPLETE-DATA-VIA"
source: https://arxiv.org/pdf/2609.37664v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-03 08:54:48"
---

# 论文速读：LEARNING-CAUSAL-NORMALIZING-FLOWS-FROM-IN-COMPLETE-DATA-VIA

## 一句话总结
MissCNF 提出直接在部分观测数据上训练因果归一化流（CNF），通过最大化每个样本的边际似然避免传统插补-拟合流程的偏差，并在 MAR 假设下给出基于因果族正概率（causal-family positivity）的严格分布可识别理论。

## 研究问题与动机
- 因果归一化流支持结构因果模型下的干预与反事实推断，但现有工作均假设训练数据完整；医疗、生物等现实领域缺失数据不可避免。
- **Listwise deletion** 在 MAR 下因保留的条件分布偏移（$p^*(\mathbf{x}|\mathcal{R}=\mathbf{1}_V)\neq p^*(\mathbf{x})$）引入重加权偏差，且造成大量样本浪费。
- **Impute-then-fit 流水线**（Mean、MICE、MissForest、DiffPuter 等）先构造补全数据集再训练，存在不确定性低估、多插补版本模型合并困难、目标分布非真实 $p^*$ 等问题。
- 通用缺失生成模型缺乏因果拓扑约束与可识别性理论保障，难以支撑严谨的因果推断下游任务。

## 核心贡献（创新点）
1. **MissCNF 边际似然训练框架**：直接对部分观测样本优化 $\int p_\theta(\mathbf{x}) d\mathbf{x}_M$，无需删除行或显式生成补全数据集；与插补流水线的本质区别在于绕过人工补全，避免插补噪声引入的分布偏移。
2. **MAR 可忽略性与全局识别定理**：证明缺失机制项与 $\theta$ 无关（Lemma 1），并给出全局最优恢复真实联合分布的充分条件——因果族正概率（Theorem 1/Corollary 1）；与通用缺失模型的本质区别在于提供可验证的理论锚点而非纯经验拟合。
3. **祖先闭包积分剪枝**：证明仅需对 $A=\mathrm{anc}(O)$ 内的缺失节点进行数值积分，Barren 节点边缘积分为 1 可直接丢弃；与现有因果流工作的本质区别在于显式利用拓扑序降低 MC 估算维度。
4. **因果族可识别性实证验证**：构造“无完整行但满足因果族覆盖”的极端场景，验证理论边界；与仅依赖完整样本或强插补假设的基线的本质区别在于不依赖全观测模式的存在。

## 方法详解
- **训练目标**：经验损失 $\mathcal{L}_{n,K}(\theta)=-\frac{1}{n}\sum_{j=1}^n \log \hat{p}_{\theta,O_j}(\mathbf{x}_{O_j}^{(j)})$，其中 $\hat{p}_{\theta,O}$ 为边际似然的 Monte Carlo 估计。
- **祖先闭包剪枝**：对观测集 $O$，仅积分 $M'=\mathrm{anc}(O)\setminus O$ 中的缺失节点；后代/无关节点（Barren）积分为 1 直接丢弃，MC 采样沿拓扑序进行，时间复杂度 $\mathcal{O}(K\,d\,L)$。
- **MAR 可忽略性**：Lemma 1 分解 $q_\theta(\mathbf{x}_O,R)=p_{\theta,O}(\mathbf{x}_O)\,p_{\mathrm{miss}}(R|\mathbf{x}_O)$，缺失指示 $R$ 与 $\theta$ 独立，优化 $\mathcal
