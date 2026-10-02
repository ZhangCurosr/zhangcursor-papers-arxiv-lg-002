---
title: "PreGS-A-Parameter-Transfer-Based-Multi-Expert-Graph-Neural-N"
source: https://arxiv.org/pdf/2609.26310v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:25:49"
---

# 论文速读：PreGS-A-Parameter-Transfer-Based-Multi-Expert-Graph-Neural-N

## 一句话总结
本文提出 PreGS，一种基于参数迁移的多专家图神经网络框架，通过将预训练多头 GAT 第一层注意力头的线性变换权重迁移至多个 GraphSAGE 专家，并冻结所有图骨干仅训练轻量级融合模块与 MLP 分类器，在 8 个公开节点分类数据集上实现了稳定且具竞争力的性能，显著降低了多分支训练的复杂度与优化不稳定性。

## 研究问题与动机
- **单一聚合机制局限**：现有 GNN 通常依赖固定 Neighborhood Aggregation 偏置（如 GAT 的注意力加权或 GraphSAGE 的均值/求和），单分支难以充分捕捉图数据中多样化的结构模式。
- **多专家独立训练开销大**：传统 MoE 或集成方法常需独立训练多个结构分支或引入复杂路由模块，计算成本高且难以保证各专家间的参数对应关系与表示空间一致性。
- **现有蒸馏方法未利用内部参数结构**：GNN-to-MLP 蒸馏多转移预测 logits 或中间表示以降低部署成本，较少关注如何直接复用已训练 GNN 的内部线性变换参数来构建结构同源的多专家系统。
- **经典 GNN 潜力未充分挖掘**：近年重估研究表明，合理调优与结构增强的经典消息传递 GNN 仍能匹敌或超越复杂 Graph Transformer，提示需从统一视角重新审视不同聚合机制的可迁移性。

## 核心贡献（创新点）
1. **提出统一邻域聚合理论视角**：将 GAT 的注意力加权求和与 GraphSAGE 的 mean/sum/max 聚合统一表述为带权重系数的线性变换形式，从理论上证明两者在线性特征变换结构上的兼容性，为跨架构参数迁移提供依据。
2. **设计基于参数迁移的 GraphSAGE 多专家构建机制**：将预训练 GAT 第一层各注意力头的权重直接迁移至 K 个单层 GraphSAGE 专家，使专家系统具备显式参数对应与对齐的表示空间；与传统 MoE 依赖随机初始化或独立训练专家的本质区别在于零额外参数开销且天然保证特征映射一致性。
3. **引入解耦训练范式**：冻结预训练 GAT 与所有迁移专家，仅优化融合模块与 MLP 分类器；与端到端联合训练相比，将高度耦合的图表示优化问题转化为轻量级任务适配问题，显著提升训练稳定性与收敛效率。
4. **提出 PreGS 与 PreGSv2 两种融合变体**：PreGS 采用分组加权融合与 logit 级融合；PreGSv2 进一步引入源级可学习权重与结构门控机制实现节点级自适应特征调制；二者在 24 组数据集-训练比例设置中占据 20 项最优、21 项前三，全面验证框架有效性。

## 方法详解
- **三阶段流程**：
  1. **GAT 预训练**：在目标图上预训练 2 层多头 GAT（默认 8 头，每头输出维度 8），保留第一层各头表示 $\{ \mathbf{h}_i^{(k)} \}_{k=1}^K$ 与最终 logits $\mathbf{z}_i^{\mathrm{GAT}}$，预训练后冻结。
  2. **多专家构建与参数迁移**：对每个头 $k$，执行 $\mathbf{W}_{\mathrm{GS}}^{(k)} \leftarrow \mathbf{W}_{\mathrm{GAT}}^{(k)}$ 构建单层 GraphSAGE 专家 $GS_k$，分配不同聚合器（mean/sum/max）以丰富多样性，构造完成后全部冻结。
  3. **解耦融合训练**：仅训练分组权重、logit 融合权重及 MLP 参数，损失函数为标准交叉熵 $\mathcal{L}_{\mathrm{CE}}$。
- **PreGS（分组加权融合）**：分别对 GAT 头条与 GS 专家计算 softmax 可学习权重 $\omega_k^{\mathrm{GAT}}$ 与 $\omega_k^{\mathrm{GS}}$，加权求和得 $\mathbf{f}_i^{\mathrm{GAT}}$ 与 $\mathbf{f}_i^{\mathrm{GS}}$；与原特征 $\mathbf{x}_i$ 拼接后输入 2 层 MLP 得 $\mathbf{z}_i^{\mathrm{MLP}}$；最终 logits 为 $\mathbf{z}_i = \alpha \mathbf{z}_i^{\mathrm{MLP}} + (1-\alpha) \mathbf{z}_i^{\mathrm{GAT}}$，其中 $\alpha = \frac{\exp(\lambda_1)}{\exp(\lambda_1)+\exp(\lambda_2)}$。
- **PreGSv2（源级加权+结构门控）**：在 PreGS 基础上，引入源级权重 $\rho_s = \mathrm{softmax}(\eta_s)$ 对 $\mathbf{x}_i, \mathbf{f}_i^{\mathrm{GAT}}, \mathbf{f}_i^{\mathrm{GS}}$ 加权拼接；再由 GAT 表征生成门控向量 $\mathbf{g}_i = \sigma(\mathbf{f}_i^{\mathrm{GAT}}\mathbf{W}_g + \mathbf{b}_g)$，通过 Hadamard 乘积调制特征 $\mathbf{f}_i^{\mathrm{gate}} = \mathbf{g}_i \odot \mathbf{f}_i^{\mathrm{src}}$，随后送入 MLP 并与 GAT logits 融合。
- **理论保证**：Proposition 2 证明维度一致时 GAT 权重可无损迁移至任意聚合器的 GraphSAGE；Corollary 1 证明迁移后各专家与 GAT 头共享线性变换空间，支持直接特征级拼接而无需额外投影层。

## 实验与结果
- **数据集**：ACM, AMAC, AMAP, DBLP, EAT, FILM, PubMed, Texas（共 8 个，覆盖学术、引文、电商、航空、社交、网页等类型）。
- **评估基线**：GCN, SGC, GIN, GAT, GATv2, GraphSAGE, GraphSAGE++, GNNMoE, JK-Net, MLP。
- **主要结果**：
  - 在 24 组数据集-训练比例设置中，Pre
