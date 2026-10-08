---
title: "Valid-for-Free-Homophily-Gated-Conformal-Prediction-for-Trai"
source: https://arxiv.org/pdf/2610.08564v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:24:52"
field: "图机器学习中的可靠性量化"
keywords: ["conformal prediction", "tabular foundation models", "node classification", "homophily gating", "uncertainty quantification", "training-free learning"]
innovations: ["冻结in-context预测器的分裂共形预测在有限样本下精确有效（无需训练协议审计）", "HG-DAPS：基于上下文标签调整同价的无调参扩散门控，在同价图上缩小集合大小5.8%-17.1%", "揭示原始同价门控在类别不平衡图上的陷阱：边际覆盖率掩盖低同价节点覆盖率下降0.12-0.27"]
benchmarks: ["Cora", "CiteSeer", "PubMed", "Photo", "Coauthor-CS", "WikiCS", "Actor", "Amazon-Ratings", "Minesweeper", "Tolokers"]
---

# 论文速读：Valid-for-Free-Homophily-Gated-Conformal-Prediction-for-Training-Free-Node-Classification-with-Tabular-Foundation-Models

## 一句话总结
本文首次对"无训练节点分类"场景下 TabICL（一种表格基础模型）的后验分布进行了可靠性研究：证明了冻结的 in-context 预测器在有限样本下使分裂共形预测**精确有效**（无需任何训练协议审计），并提出了 HG-DAPS——一种仅读取上下文标签的调整同价门控扩散分数，在同价图上较 APS 将平均集合大小缩小 5.8%–17.1%，同时在两个陷阱数据集上避免了原始同价门控造成的低同价节点覆盖率下降 0.12–0.27。

## 研究问题与动机
1. **精度不足以部署**：TFM（如 TabICL、TabPFN）已能在无训练条件下对图节点分类，但已有工作只报告准确率，未报告共形覆盖率与预测集合质量。
2. **无训练场景的可靠性缺口**：对冻结 in-context 预测器的后验进行校准/共形，若依赖验证折或目标图上的调参，则丧失"无训练"优势；需要一种**不依赖校准折**的可靠性保障。
3. **图共形中的交换性与扩散困境**：分裂共形要求校准与测试节点可交换；沿边扩散非共形分数在同价图上能 sharpen 集合，但在异价图上会混合不同类别的证据——而决定是否扩散的同价估计本身不能依赖校准折。
4. **原始同价的陷阱**：边原始同价 $h_{\text{edge}}$ 在类别不平衡时会虚高，导致基于它的门控在低同价节点处过度扩散、压低覆盖率，而边际覆盖率仍能维持在名义值，掩盖这一失效。

## 核心贡献（创新点）
1. **"免费有效"引理（Lemma 1）**：分裂共形预测对冻结 in-context 预测器在有限样本下**精确有效**，无需训练协议审计、无需验证折、无需超参调优；与已有训练型 GNN 加温度缩放的本质区别在于**有效性来自 score matrix 在校准前已完全固定**，而非来自数据分布假设。
2. **预注册可靠性审计**：在 10 张图上证明无训练 TabICL 后验在 9/10 图上 ECE 低于温度缩放 GCN（GCN+TS），十图平均 ECE 为 **0.019 vs 0.029（低 35%）**，且无任何后校准步骤。
3. **HG-DAPS（同价门控扩散分数）**：扩散权重 $\delta$ 仅由上下文标签的**调整同价**$\hat{h}_{\text{adj}}$ 决定，与 DAPS（需在标号节点上调权）和 HeAD-CP（节点-wise 系数）本质不同；对 APS 在同价图上将平均集合大小缩小 **5.8%–17.1%**，在异价图上变化 <1%。
4. **陷阱案例（Trap Case, H4）**：预注册发现，在 Minesweeper 和 Tolokers 两个二元不平衡图上，**原始同价门控**使低同价节点覆盖率分别下降 0.27 和 0.12，而调整同价门控几乎不扩散（$\delta \approx 0$），将损害限制在 0.02 以内；边际覆盖率 0.90 均维持，凸显单独看边际覆盖率的盲区。

## 方法详解
**特征表构建**：每个节点 $v$ 映射为表格行 $\phi(v) = [\text{SVD}_{64}(x_v), \text{SVD}_{64}(\bar{x}_v^{(1)}), \text{SVD}_{64}(\bar{x}_v^{(2)}), \text{aggr}, \log(1+\deg v)]$，其中 SVD 在所有节点上无标签拟合，$\bar{x}_v^{(r)}$ 为 $r$-跳邻居均值，aggr 为 1 跳最大/最小值。冻结 TabICL 以最多 5000 行上下文 $(\phi(c), y_c)$ 作 prompt，在 softmax temperature=1 下返回后验 $\hat{p}(\cdot|v)$。

**非共形分数（APS）**：按 $\hat{p}(k|v)$ 降序排列，标签 $k$ 的分数为排在 $k$ 之前的概率质量加上 $k$ 自身质量的随机份额；每行一次均匀采样并缓存，得到固定 score matrix $S \in [0,1]^{|U|\times K}$。

**分裂共形阈值**：$\hat{q} = s_{(j)},\; j=\lceil(n_{\text{cal}}+1)(1-\alpha)\rceil$，预测集 $\hat{C}(v)=\{k:S_{v,k}\le\hat{q}\}$。

**Lemma 1 有效性证明要点**：score matrix、上下文 prompt、冻结模型、APS 采样均在校准折选取之前完全固定，因此 $\{(S_{v,\cdot},y_v)\}_{v\in U}$ 构成固定有限总体；均匀随机选 $\mathcal{D}_{\text{cal}}$ 后，$n_{\text{cal}}+1$ 个真标签分数为无放回均匀样本，具有置换不变性（可交换），故 $P(y_{v^*}\in\hat{C}(v^*))\ge 1-\alpha$。

**HG-DAPS 门控扩散**：调整同价 $\hat{h}_{\text{adj}}=(h_{\text{edge}}-P)/(1-P)$，其中 $P=\sum_k p_k^2$ 为随机匹配的同类概率；扩散权重 $\delta=\delta_{\max}\text{clip}_{[0,1]}(\hat{h}_{\text{adj}}\cdot m/(m+m_0))$，$\delta_{\max}=0.5, m_0=100$；全图 score 矩阵 $S^{\text{full}}$ 中上下文行为 one-hot LAC 分数（确定证据），评估行保持缓存 APS 分数；一步扩散 $\tilde{S}=(1-\delta)S^{\text{full}}+\delta D^{-1}AS^{\text{full}}$，在校准集上重估阈值并输出预测集。**Corollary 1**：因 $\tilde{S}$ 映射同样在校准前固定，Lemma 1 直接适用，覆盖率 $\ge 1-\alpha$ 对任意 $\delta$ 成立。

## 实验与结果
- **数据集**：10 个标准节点分类基准，按 $\hat{h}_{\text{adj}}$ 排序——高同价（Photo 0.79, Cora 0.79, Coauthor-CS 0.78, PubMed 0.68, CiteSeer 0.64, WikiCS 0.59）与低同价（Amazon-Ratings 0.14, Tolokers 0.10, Minesweeper 0.01, Actor 0.00）。
- **协议**：3 个 50% 上下文种子，每种子 20 次校准/测试重分（共 60 次/图），$n_{\text{cal}}=\min(1000,\lfloor|U|/2\rfloor)$，$\alpha=0.10$。
- **基线**：GCN、GAT、GCN+TS（在上下文验证折上调参）、MLP、逻辑回归、标签传播。
- **ECE 审计（H1）**：TabICL 在 9/10 图上 ECE 低于 GCN+TS；十图平均 **0.019 vs 0.029（-35%）**；也在 8/10 图上优于所有训练基线；Mean accuracy 0.778 vs GCN 0.752。
- **有效性审计（H2）**：180 个预注册单元格（10 图×3 种子×6 分数）的覆盖率均值全部落在 0.90 名义值的双尾 Clopper-Pearson 95% 区间内；最低均值 0.894（Minesweeper/LAC）。
- **HG-DAPS 效率（H3）**：5 个同价图上较 APS 缩小 5.8%（PubMed）至 17.1%（WikiCS），Coauthor-CS 缩小 10.3%；4 个异价图上变化 <1%； ungated DAPS 在异价图 Actor 上反而扩大 9.9%。
- **陷阱案例（H4）**：Minesweeper ($h_{\text{edge}}=0.69$, $\hat{h}_{\text{adj}}\approx 0.01$) 和 Tolokers ($h_{\text{edge}}=0.59$, $\hat{h}_{\text{adj}}\approx 0.10$) 上，原始门控 $\delta=0.34/0.30$，调整门控 $\delta=0.003/0.05$；低同价节点覆盖率下降 **0.27 和 0.12**；边际覆盖率 0.897–0.901 均维持，掩盖失效。

## 相关工作脉络
1. **TFM on Graphs**（Hayler et al. 2025, Eremeev et al. 2025）：首次将 TabICL/TabPFN 用于零样本节点分类，报告准确率但无共形/集合质量评估；本文填补该空白。
2. **Graph Conformal Prediction**（Zargarbashi et al. ICML 2023, Huang et al. NeurIPS 2023）：DAPS 沿边扩散非共形分数，HeAD-CP 从 GNN softmax 导出节点-wise 系数；两者均需在标号节点上调权，本文 HG-DAPS **仅读上下文标签**，不依赖校准折。
3. **Calibration of GNNs**（Wang et al. 2021, Hsu et al. 2022）：CaGCN、GATS 等学习节点依赖校准函数；需要held-out标签且针对训练型 GNN，本文对**冻结 TabICL 后验**直接审计，无需后校准。
4. **Tabular Foundation Models**（TabPFN Nature 2025, TabICL ICML 2025）：表格基础模型通过 in-context learning 分类，本文将其范式迁移至图节点分类并首次分析其共形性质。
5. **Homophily Characterization**（Platonov et al. NeurIPS 2023）：提出调整同价 $\hat{h}_{\text{adj}}$ 概念，本文将其用于构建**不依赖校准折**的扩散门控，并揭示原始同价的陷阱。
6. **LoGIC**（Yang et al. 2026）：预算化上下文构建，与本文共用 TabICL 框架，但聚焦 prompt 长度而非可靠性。

## 局限性与未来方向
1. **单一 TFM 评估**：仅评测 TabICL，未测试 TabPFN v2、GraphPFN 或 HeAD-CP 在同一后验上的表现。
2. **单次校准集的波动**：保证是平均意义上的，单个固定校准集仍可能实现 $<1-\alpha$ 覆盖率。
3. **未报告空集率**：高准确率图（Photo、Coauthor-CS）上 APS/HG-DAPS 会输出空集合，空集率未报告。
4. **$\delta_{\max}, m_0, 50\%$ 上下文比例未消融**，也未评估类别特定校准、多跳扩散或分离 one-hot 上下文行的变体。
5. **同价区间空白**：没有 $\hat{h}_{\text{adj}}\in(0.2,0.5)$ 的图，中间地带行为未知。
6. **开放问题**：HG-DAPS 在同价图内部是否也会压低低同价节点的覆盖率（论文主疑问）。

## 研究启发与可借鉴点
1. **"score matrix 在校准前固定"这一有效性技巧**可迁移到任何冻结的 in-context 或 few-shot 预测器（视觉/语言 TFM + 共形），无需再审计训练协议。
2. **预注册陷阱案例设计**（H4）：将"边际覆盖率维持但子群覆盖率崩塌"作为检测门控失效的范式，值得推广到其他扩散类可靠性方法。
3. **调整同价门控替代原始同价门控**的思路可迁移到任何基于图结构的自适应扩散/正则化模块，尤其是类别不平衡场景。
4. **SVD 降维+邻居聚合**的特征表构建方式（SVD₆₄ + 1/2 跳均值 + 极值 + log-degree）可与本团队的方向结合，用于低资源图分类的 TFM 接入层。
5. **无训练 + 共形有效**的组合为高可靠部署（如欺诈检测、医疗节点分类）提供了"零额外成本换可靠性"的参考范式。

## 关键术语表
**Tabular Foundation Model (TFM)**：在合成表格任务上预训练、通过 in-context learning 对新表格分类的模型（如 TabICL、TabPFN），无需梯度更新。
**Split Conformal Prediction**：将数据分为校准集和测试集，在校准集上估计非共形分数分位数作为阈值，为测试点生成覆盖率保证的预测集合。
**APS (Adaptive Prediction Set)**：按预测概率降序累加至包含真标签的累积概率，加上随机份额，是最常用的共形非共形分数。
**Expected Calibration Error (ECE)**：将预测概率分箱后计算各箱平均置信度与真实准确率之差的加权绝对偏差，衡量校准质量。
**Adjusted Homophily ($\hat{h}_{\text{adj}}$)**：Platonov 等人提出的调整边同价，$ (h_{\text{edge}}-P)/(1-P) $，消除类别不平衡带来的偏差。
**HG-DAPS (Homophily-Gated DAPS)**：本文提出的扩散共形方法，用仅读上下文标签的调整同价门控控制非共形分数的图扩散权重。
**In-context Learning**：大模型通过 prompt 中的上下文示例进行零样本/少样本推理，无需参数更新。
**Transductive Exchangeability**：在固定模型和特征后，校准与测试节点对的联合分布在均匀随机划分下具有置换不变性。

## 可复现要素
- **数据集**：10 个标准图基准（Cora, CiteSeer, PubMed, Photo, Coauthor-CS, WikiCS, Actor, Amazon-Ratings, Minesweeper, Tolokers），均已公开。
- **代码/权重**：论文未提及开源；TabICL 基础模型权重来自原论文。
- **关键超参**：$\alpha=0.10$，$\delta_{\max}=0.5$，$m_0=100$，上下文比例 50%，$n_{\text{cal}}=\min(1000,\lfloor|U|/2\rfloor)$，SVD 维度 64，最大 prompt 长度 5000 行，softmax temperature=1。
- **硬件**：一次推理运行于 8 GB 消费级 GPU，后续 CPU 步骤秒级完成。
- **随机种子**：从数据集名、上下文种子和采样目的确定性派生；上下文类频率偏差 >0.05 时重抽。
