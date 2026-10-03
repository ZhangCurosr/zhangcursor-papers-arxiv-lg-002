---
title: "LEARNING-CAUSAL-NORMALIZING-FLOWS-FROM-IN-COMPLETE-DATA-VIA"
source: https://arxiv.org/pdf/2609.37664v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-03 08:54:02"
field: "因果机器学习与不完整数据"
keywords: ["因果归一化流", "缺失数据处理", "边缘似然", "因果可识别性", "生成模型", "结构因果模型"]
innovations: ["提出 MissCNF，直接从部分观测数据训练因果归一化流，无需插补或删除", "推导边缘似然的祖先闭包简化计算，大幅降低复杂度", "建立因果族正性条件，精确刻画 MAR 下联合分布的可识别性与不可识别区域"]
benchmarks: ["8个合成SCM (Chain/Fork/Collider/Triangle × Linear/Non-linear)", "MCAR/MAR/MNAR 缺失机制，30%/60%/90% 缺失率", "KL散度, RMSE_ATE, RMSE_CF, 局部KL误差"]
---

# 论文速读：LEARNING-CAUSAL-NORMALIZING-FLOWS-FROM-INCOMPLETE-DATA-VIA

## 一句话总结
本文提出 **MissCNF**，一种直接从**部分观测数据**训练因果归一化流（CNF）的新方法，无需丢弃含缺失值的样本行，也无需构建插补数据集。该方法通过最大化每个部分观测样本的边缘似然进行训练，在非线性因果场景下显著优于传统插补方法与前沿基线，并在理论上明确了缺失数据的可识别条件。

## 研究问题与动机
- **核心问题**：如何在存在缺失值（Missing Values）的观测数据上，有效学习一个能准确建模**完整联合分布**与**因果效应**的归一化流模型。
- **现有方法不足**：
    - **Listwise Deletion（列表删除）**：丢弃任何包含缺失值的行。在 MAR（随机缺失）机制下会导致估计偏差，且 KL 散度误差随 MAR 性质恶化急剧扩大（实验显示恶化达 **14.0×**，而 MissCNF 仅 **2.0×**）。
    - **Impute-then-Fit（先插补再拟合）**：包括均值插补、MissForest、MICE、DiffPuter、KPI 等方法。这类方法将插补的不确定性丢弃，导致插补误差在后续建模中传播，且在非线性关系下全面失效。
    - **理论空白**：尽管 MAR 假设下缺失机制对似然推断“可忽略”（Ignorable），但这并不足以保证**完整联合分布**的可识别性。现有工作缺乏对此条件的清晰刻画及不可识别区域的定位。

## 核心贡献（创新点）
1. **提出 MissCNF 方法框架**：设计了直接在部分观测数据上训练 CNF 的损失函数，通过对祖先闭包内缺失变量进行边缘化，避免了列表删除和插补策略。
   *本质区别*：不同于“插补-拟合”两阶段范式，MissCNF 联合处理缺失模式与分布估计；不同于完全忽略缺失的列表删除法，它充分利用所有部分观测样本的信息。

2. **推导观测数据边缘似然的精确分解（Proposition 1）**：证明了对 CNF 的自回归因果结构，计算部分观测样本的边缘似然时，仅需对**观测集合祖先闭包（ancestral closure）中缺失的变量**进行积分，而“ barren ”（无后代在观测集中）的节点积分后为 1，可直接丢弃。
   *本质区别*：将一般概率图的边缘化问题简化为基于因果结构的子图计算，大幅降低了计算复杂度。

3. **建立 MAR 下的可忽略性与可识别性分离理论（Lemma 1, Theorem 1）**：严格证明了在 MAR 假设下，最优参数能使观测数据的分布收敛到真实观测分布，但**全局最优解仅几乎处处恢复真实联合分布**，需额外条件才能实现完全可识别。
   *本质区别*：澄清了“MAR 保证缺失可忽略”这一经典结论在生成建模语境下的局限，为后续识别条件分析奠定基础。

4. **提出因果族正性条件（Causal-family positivity）并定位不可识别区域**：定义了比完整模式正性更弱的条件——每个节点的父节点与其自身需在某个观测模式中联合出现。证明了满足该条件时，全局最优解可恢复真实联合分布（Corollary 1），并给出了正性不满足时**不可识别区域（$Z_O$）**的精确理论描述。
   *本质区别*：将可识别性条件从传统的“所有变量联合被观测”放宽至“因果族被观测”，更具实用性，且提供了故障诊断的理论工具。

5. **系统性实验验证**：在 8 种合成结构因果模型（SCM）、多种缺失机制（MCAR/MAR/MNAR）与缺失率（30%/60%/90%）下，与 7 种基线方法对比。结果表明，MissCNF 在**非线性场景下 consistently 最优**（KL 最低 23/24，RMSE_CF 最低 20/24），在线性场景下亦保持稳健（排名前二 22/24）。
   *本质区别*：首次将因果归一化流与不完整数据学习系统结合，并在分布估计与因果效应（ATE, CF）估计双重任务上验证其优势。

## 方法详解
- **模型基础**：采用自回归因果归一化流（Causal Normalizing Flow, CNF）。假设数据遵循结构因果模型（SCM）：$x_i = f_i(\mathbf{x}_{\mathrm{pa}(i)}, \epsilon_i)$，其中 $\epsilon_i \sim \mathcal{N}(0,1)$。CNF 通过学习可逆映射 $g_\theta: \mathbf{x} \to \mathbf{u}$ 将数据分布转化为标准正态分布，其对数似然为 $\log p_\theta(\mathbf{x}) = \log |\det \partial g_\theta / \partial \mathbf{x}| - \frac{1}{2}\|\mathbf{u}\|^2 + C$。
- **边缘似然计算**：给定观测指标集 $O$ 和观测值 $\mathbf{x}_O$，目标是计算 $p_{\theta,O}(\mathbf{x}_O) = \int p_\theta(\mathbf{x}) d\mathbf{x}_M$，其中 $M = V \setminus O$。根据 Prop 1，令 $A = \mathrm{anc}(O)$，$M' = A \setminus O$，则积分仅对 $M'$ 中的变量进行，其余变量因因果结构被积分为 1。
- **蒙特卡洛估计（Algorithm 1）**：
    1. 从基分布采样 $K$ 个噪声样本 $\{\mathbf{u}_{M'}^{(k)}\}_{k=1}^K$。
    2. 对于每个样本，沿因果拓扑序前向传播计算观测变量 $\mathbf{x}_O$ 及其雅可比行列式。
    3. 在 log-space 使用 `logsumexp` 计算对数边缘似然的蒙特卡洛估计：$\log p_{\theta,O}(\mathbf{x}_O) \approx \mathrm{LSE}_k [\log p_\theta(\mathbf{x}_O, \mathbf{x}_{M'}^{(k)}) + \log |\det J|]$。
- **训练目标**：最小化负对边缘似然的期望：$\mathcal{L}(\theta) = \mathbb{E}_{p_{\mathrm{obs}}(\mathbf{x}_O, R=\mathbb{1}_{O})}[-\log p_{\theta,O}(\mathbf{x}_O)]$，其中 $p_{\mathrm{obs}}$ 是带缺失机制的真实数据分布。
- **计算复杂度**：每次前向/反向传播为 $\mathcal{O}(dL)$（$d$ 为变量数，$L$ 为流层数），蒙特卡洛估计引入因子 $K$，总复杂度 $\mathcal{O}(KdL)$。线性增长于 $K$ 和 $d$。
- **扩展性**：该方法可自然扩展到处理潜混杂变量的 DeCaFlow 及纵向数据的 TSCNF 框架。

## 实验与结果
- **数据集**：8 个合成 SCM（4 种图结构 × 2 种函数类型）：Chain/Fork/Collider/Triangle × Linear/Non-linear。每个结构生成 20,000 训练 / 2,500 验证 / 2,500 测试样本。来源引用 Javaloy et al. (2023) 与 Sanchez-Martin et al. (2022)。
- **缺失机制与率**：
    - MCAR: 30%, 60%, 90%
    - MAR: 30%, 60%, 90%（缺失概率依赖于 conditioning set）
    - MNAR (Self-masking): 30%, 60%
- **评估指标**：对称 KL 散度 ($D_J$)、平均处理效应 RMSE ($\mathrm{RMSE}_{\mathrm{ATE}}$)、反事实预测 RMSE ($\mathrm{RMSE}_{\mathrm{CF}}$)、局部 KL 误差。
- **基线方法**：Listwise Deletion, Mean Imputation, MissForest, MICE (HyperImpute), KPI, DiffPuter, O-MIRACLE。
- **实现细节**：CNF 使用 `causal-flows` 库，单掩码自回归层 + hyper-network (MLP, 64 units × 2, ReLU)，AdamW (lr=$10^{-3}$)，batch size=4096，1000 epochs。MC 样本数 $K=512$。硬件：NVIDIA RTX 4090 (24GB)。
- **主要结果**：
    - **非线性 SCM（24 组）**：MissCNF 最低 KL 在 **23/24**，最低 $\mathrm{RMSE}_{\mathrm{CF}}$ 在 **20/24**。在 MCAR 60% ForkNLIN 中，MissCNF KL=0.069，而 MICE 高达 1.187，DiffPuter 为 0.111。
    - **线性 SCM（24 组）**：MICE 最佳 23/24，MissCNF 排名前二 **22/24**。MissCNF 在线性任务上略逊于强插补器 MICE，但保持了良好性能。
    - **MNAR 60%**：非线性全部 8 组 MissCNF 最优；线性 8 组 MICE 最优，MissCNF 第二。
    - **高缺失率鲁棒性**：在 90% 缺失下，多数基线（如 MissForest, DiffPuter）KL > 10 或崩溃，而 MissCNF 仍维持较低误差（如 MCAR 90% ForkNLIN KL=0.282）。
    - **计算效率**：KL 在 $K \le 128$ 后饱和；$d=64$ 时峰值内存约 10 GB。
- **关键结论**：MissCNF 在非线性因果场景下 consistently 最优，符合 CNF 的设计动机；在线性场景下虽非绝对最佳，但稳健不崩；理论明确给出了识别条件与不可识别区域。

## 相关工作脉络
1. **因果归一化流（Causal NF）**：如 `causal-flows` 库相关研究，专注于从无缺失的完整数据中学习因果结构对应的流模型。MissCNF 将其拓展至部分观测数据场景。
2. **缺失数据处理传统方法**：Listwise deletion, Mean imputation, MissForest, MICE。这些方法或丢失信息、或引入偏差、或假设线性/兼容性，MissCNF 从分布建模角度统一处理缺失模式。
3. **前沿生成模型插补**：DiffPuter（扩散模型+EM）、O-MIRACLE（利用真因果图的 MIRACLE 变体）、KPI（核岭回归）。MissCNF 直接利用因果结构进行边缘似然计算，而非通过迭代插补或扩散步骤，在非线性下表现更佳。
4. **可识别性理论**：Pearl 的因果识别理论、Rubin 的潜在结果框架与 MAR 可忽略性。本文在连续变量与生成模型语境下，重新审视并形式化了 MAR 对联合分布可识别性的影响，提出了“因果族正性”这一更精细的条件。
5. **不完整数据的深度学习**：各类基于深度学习的缺失值插补与生成模型（如 VAEs, GANs for missing data）。MissCNF 的独特之处在于结合**因果先验**与**归一化流的显式密度建模**，专注于因果联合分布与效应估计，而非仅完成填补缺失值的任务。

## 局限性与未来方向
- **验证范围**：当前仅在合成数据上验证，尚未在真实世界基准数据集（如 UCI 仓库、医疗记录）上进行测试。
- **计算开销**：蒙特卡洛估计的代价随祖先变量数量增加而线性增长，对于高维或复杂祖先闭包的场景可能面临挑战（尽管实验显示 $K=512$ 时内存可控）。
- **前提假设**：需要**已知因果图**（或可准确估计的图），并假设**因果充分性**（无潜混杂，或已通过扩展模型如 DeCaFlow 处理）。
- **可扩展性**：文中提到可自然扩展至 DeCaFlow（处理潜混杂）和 TSCNF（纵向数据），但这部分工作留作未来方向。
- **线性场景优化**：在线性 SCM 下，强插补器 MICE 仍有小幅优势，如何进一步提升 MissCNF 在线性场景的效率或精度是潜在改进点。

## 研究启发与可借鉴点
1. **边缘似然计算的结构化简化**：Prop 1 展示的“仅对祖先闭包内缺失变量积分”的思路，可迁移至其他基于图结构的生成模型（如因果 VAЕs、扩散模型）在不完整数据上的训练，提供通用的计算加速策略。
2. **理论与应用的紧密结合**：论文清晰区分了“可忽略性”与“可识别性”，并用理论条件（因果族正性）精确定位失败区域。这种“提出方法 -> 严格分析识别条件 -> 给出诊断工具”的研究范式，对构建可靠的因果 ML 系统极具参考价值。
3. **直接处理缺失模式而非插补**：MissCNF 的损失函数设计（对每个部分观测样本计算边缘似然）避免了插补带来的不确定性丢失和误差传播。这一“端到端”思想可启发设计其他直接利用不完全数据的生成模型训练目标。
4. **全面的因果评估基准**：实验不仅评估分布恢复（KL），还评估因果效应估计（ATE, CF RMSE），并覆盖 MCAR/MAR/MNAR 多种机制。这种多维度的评估体系可作为类似研究的实验设计模板。
5. **模块化扩展潜力**：文章指出 MissCNF 框架可与 DeCaFlow、TSCNF 等现有因果流模型结合。这种“插件式”设计思路表明，基于流的因果建模模块可以相对独立地扩展以处理新的挑战（如潜变量、时间序列、部分观测）。

## 关键术语表
- **MissCNF (Missing Causal Normalizing Flow)**：本文提出的方法，直接从部分观测数据训练因果归一化流，通过最大化观测数据边缘似然进行端到端学习。
- **Causal Normalizing Flow (CNF)**：结合因果图结构的归一化流，其可逆变换与因果生成过程对齐，用于精确建模高维连续数据的联合分布及进行因果推理。
- **Ancestral Closure**：给定观测变量集合 $O$，其祖先闭包 $A = \mathrm{anc}(O)$ 包含所有能因果影响到 $O$ 中变量的节点。边缘化只需在此子集内进行。
- **Ignorable vs. Identifiable**：**Ignorable（可忽略）**指缺失机制不影响参数估计的渐近性质；**Identifiable（可识别）**指能从数据中唯一确定模型参数。MAR 保证前者，但不保证后者。
- **Causal-family Positivity**：新的可识别性充分条件，要求每个节点的父节点与其自身在某个观测模式中联合出现且概率 > 0，比传统的“所有变量联合被观测”更弱。
- **Structural Causal Model (SCM)**：用函数方程和噪声分布描述变量间因果关系的框架，是因果推断与因果生成模型的理论基础。
- **MCAR/MAR/MNAR**：三种缺失机制。MCAR（完全随机缺失）、MAR（随机缺失，缺失概率依赖其他观测变量）、MNAR（非随机缺失，缺失概率依赖缺失值自身）。
- **DeCaFlow / TSCNF**：分别用于处理潜混杂变量和纵向数据的因果归一化流扩展模型。MissCNF 框架可与之结合。

## 可复现要素
- **数据集**：8 个合成 SCM（Chain/Fork/Collider/Triangle × 线性/非线性）。数据生成代码与参数可能隐含于引用文献（Javaloy et al., 2023; Sanchez-Martin et al., 2022）中，**论文未提供独立公开的数据集下载链接**。
- **代码/权重**：使用了 `causal-flows` 开源库。**论文未明确声明提供 MissCNF 的独立代码仓库或预训练模型权重**。
- **关键超参数**：MC 样本数 $K=512$，超网络 MLP 64 units × 2，ReLU，AdamW 优化器（lr=$10^{-3}$），batch size=4096，训练 1000 epochs，ReduceLROnPlateau 调度。基线方法（如 MICE, DiffPuter）使用论文附录 B.5 及实验部分 B 中给出的默认设置。
