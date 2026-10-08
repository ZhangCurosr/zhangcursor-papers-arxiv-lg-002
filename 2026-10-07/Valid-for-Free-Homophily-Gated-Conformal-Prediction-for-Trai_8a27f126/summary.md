---
title: "Valid-for-Free-Homophily-Gated-Conformal-Prediction-for-Trai"
source: https://arxiv.org/pdf/2610.08564v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:24:45"
field: "图机器学习中的可靠性量化"
keywords: ["conformal prediction", "tabular foundation models", "node classification", "homophily-gated diffusion", "training-free", "uncertainty quantification"]
innovations: ["冻结in-context predictor的split conformal prediction在有限样本下精确有效（无需训练协议审计）", "HG-DAPS：由上下文标签调整同配性零调参门控score扩散，同配图缩小预测集5.8%-17.1%同时保持覆盖率", "预先注册的陷阱案例：原始同配性门控在类别不平衡异配图上导致低同配节点覆盖率骤降0.12-0.27而边际覆盖率掩盖该失败"]
benchmarks: ["Cora", "CiteSeer", "PubMed", "Photo", "Coauthor-CS", "WikiCS", "Actor", "Amazon-Ratings", "Minesweeper", "Tolokers"]
---

# 论文速读：Valid for Free — Homophily-Gated Conformal Prediction for Training-Free Node Classification with Tabular Foundation Models

## 一句话总结
本文首次系统研究了无训练 Tabular Foundation Model（TFM，以 TabICL 为核心）在图节点分类上的 conformal prediction 可靠性，证明了冻结的 in-context predictor 做 split conformal prediction 在有限样本下**精确有效**（无需训练协议、验证折或超参调优），并在此基础上提出 **HG-DAPS**——一种由调整同配性门控的扩散 conformity score 方法，在同配图中比 APS 显著缩小预测集，同时避免原始同配性门控在低同配节点上造成覆盖率骤降的陷阱。

## 研究问题与动机
- **核心问题**：TabICL/TabPFN 等 TFM 可在图上零训练完成节点分类，但已有工作只报告准确率，**未评估预测可靠性**（conformal coverage / 预测集大小）。
- **动机 1——可部署性缺口**：欺诈检测等场景要求模型知道"何时移交人工"，这取决于置信度的校准质量，而非仅准确率。
- **动机 2——无训练模型的校准悖论**：conformal calibration 若需训练或验证折，会**抵消 TFM 的无训练优势**；因此需要一种不依赖目标图训练/验证的可靠性估计。
- **动机 3——图同配性的两难**：score diffusion 在同配图中收窄预测集，但在异配图中可能扩大；决定"是否扩散"需要一个**不依赖 calibration split 的同配性估计**，且直觉统计量 `h_edge` 在类别不平衡时会被高估，导致门控在需要谨慎的节点上出错。

## 核心贡献（创新点）
- **Valid for free（引理 1）**：split conformal prediction 对冻结 in-context predictor **有限样本精确有效**，证明依赖 transductive exchangeability，无需审计任何训练协议，这是与已有关于训练 GNN 有效性论文的本质区别——后者需逐一核查早停、调参是否污染 calibration split。
- **首个 TFM+图的可靠性审计**：在 10 个图数据集上对比 TabICL 冻结后验与温度缩放 GCN（GCN+TS），TabICL 在 9/10 图上 ECE 更低（均值 0.019 vs 0.029，**降幅约 35%**），且无需任何 post-hoc 校准步骤。
- **HG-DAPS**：将 DAPS 式 score 扩散的门控权重 δ 设为**仅由上下文标签的调整同配性**（`ĥ_adj`）决定，零调参；与 APS 相比，在同配的 6 张图上平均缩小预测集 5.8%–17.1%，在异配的 4 张图上变化 <1%。
- **预先注册的陷阱案例（H4）**：在 Minesweeper 和 Tolokers（二元类别不平衡图）上，使用原始同配性 `h_edge` 做门控会使低同配节点的覆盖率下降 0.27 和 0.12，而边际覆盖率仍保持在名义 0.90——**仅看边际覆盖率会漏掉失败**。

## 方法详解
- **特征工程**：每个节点 v 映射为表格行
  ```
  φ(v) = [SVD_64(x_v), SVD_64(x̄_v^(1)), SVD_64(x̄_v^(2)),
          agg, log(1+deg v)]
  ```
  其中 x̄_v^(r) 是 r-hop 邻域特征均值，agg 含 1-hop 最大/最小值；所有 SVD 在全部节点上无标签拟合。
- **冻结预测**：TabICL 以 ≤5000 行上下文 (φ(c), y_c) 为 prompt，softmax 温度=1，**零梯度更新**，输出后验 p̂(·|v)。
- **Conformal 分数**：采用 APS（自适应预测集）分数，每行一次均匀随机抽取并缓存，分数矩阵 S ∈ [0,1]^{|U|×K} 在划分前固定。
- **阈值**：q̂ = s_(j)，j = ⌈(n_cal + 1)(1−α)⌉，预测集 Ĉ(v) = {k : S_{v,k} ≤ q̂}。
- **引理 1 证明要点**：分数矩阵所有组件在 calibration split 前固定 → {(S_{v,·}, y_v)}_{v∈U} 构成**固定有限总体** → 均匀随机 split 使 n_cal + 1 个真实标签分数呈置换不变 → 分位数阈值给出 Pr(y* ∈ Ĉ(v*)) ≥ 1−α。
- **HG-DAPS 门控权重**：
  ```
  ĥ_adj = (h_edge − P) / (1 − P),  P = Σ p_k²
  δ = δ_max · clip_{[0,1]}(ĥ_adj · m/(m + m_0))
  ```
  其中 δ_max=0.5，m_0=100，m 为上下文内边数；ĥ_adj ≤ 0 时 δ=0，退化为 APS。
- **扩散步骤**：
  ```
  S̃ = (1−δ) S^full + δ D^{−1} A S^full
  ```
  上下文行以 one-hot LAC 分数 (S_{c,k}^full = 1[k≠y_c]) 进入，保证已知标签作为确定证据；Corollary 1 证明 S^full → S̃ 仍是 split 前的固定函数，**引理 1 直接适用**，覆盖率 ≥ 1−α 对所有图和网络均成立。

## 实验与结果
- **数据集**：10 个标准节点分类 benchmark，按 ĥ_adj 排序——Photo/Cora/Coauthor-CS/PubMed/CiteSeer/WikiCS（同配，ĥ_adj > 0.5），Actor/Amazon-Ratings/Minesweeper/Tolokers（异配，ĥ_adj < 0.2）。
- **协议**：3 次 50% 上下文划分 × 20 次 calibration/test 重采样 = 60 次/数据集，n_cal = min(1000, ⌊|U|/2⌋)，α=0.10。
- **基线**：GCN、GAT、GCN+TS（在 context 中切验证折调参）、MLP、Logistic Regression、Label Propagation。
- **关键结果**：
  - **ECE**：TabICL 均值 0.019，GCN+TS 均值 0.029，**降幅 ~35%**；TabICL 在 9/10 图上优于 GCN+TS，仅 Coauthor-CS（>10 类）持平。
  - **准确性**：TabICL 均值 0.778，GCN 均值 0.752。
  - **有效性**：180 个 conformal cell（10 数据集 × 3 seed × 6 score）中，所有均值落在 Clopper-Pearson 95% 区间内；最低均值 0.894（Minesweeper/LAC）。
  - **HG-DAPS vs APS**：同配 6 图上平均缩小预测集 5.8%–17.1%（WikiCS 最大），覆盖率偏差 <0.2pp；异配 4 图上变化 <1%。
  - **陷阱案例 H4**：Minesweeper/Tolokers 上，raw-homophily 门控使低同配节点覆盖率分别下降 0.27/0.12，而 adjusted gate 偏差 <0.02；两者边际覆盖率均为 0.90，**仅看边际会漏检**。
  - **Label propagation**：均值 ECE 0.120，校准极差。

## 相关工作脉络
- **TabPFN / TabICL**（Hollmann et al. 2025 Nature; Qu et al. 2025 ICML）：表 TFM 基础；本文将其**零训练**用于图节点分类并首次评估 conformal 可靠性，区别于仅报告准确率的 [3][4]。
- **GraphPFN**（Eremeev et al. 2026 ICML）：预训练图特定 prior-fitted 网络；仍是训练型，本文**完全冻结**。
- **DAPS**（Zargarbashi et al. 2023 ICML）：score 沿边扩散，但需在有标签节点上调权；HG-DAPS 用上下文标签的 ĥ_adj **零调参**决定全局 δ。
- **HeAD-CP**（Lam & Anh 2026）：从 GNN softmax 导出节点级扩散系数；需要训练 GNN，本文**冻结 TFM 后验即可**。
- **温度缩放**（Guo et al. 2017 ICML）：标准 post-hoc 校准；GCN+TS 在 context 中切验证折调参，本文 TabICL **无需任何 post-hoc 步骤**。
- **Conformal prediction 理论**（Vovk et al. 2005；Romano et al. 2020 NeurIPS）：本工作将有限样本精确有效性扩展到**transductive 图 setting + 冻结 in-context predictor**。

## 局限性与未来方向
- Lemma 1 对任意冻结 predictor 成立，但 ECE/预测集大小的**实证结果仅限 TabICL**，未测试 TabPFN v2、GraphPFN。
- 保证是跨 calibration 抽样的平均结果，**单次固定 calibration set 可能低于 1−α**。
- 预测集大小仅与 APS 比较，未与 LAC、RAPS、context-tuned DAPS 对比；**空集率未报告**。
- 仅在 trap 图上报告了 stratum coverage，**同配图上 δ≈0.29（如 CiteSeer）是否也会损害低同配节点**是主要开放问题。
- 未测试 δ_max、m_0 敏感性、50% 上下文比例的变体；**无图 ĥ_adj 在 0.2–0.5 区间**。
- 未评估 class-specific calibration、multi-hop diffusion、或将 one-hot 上下文行与 score 平滑分离的变体。

## 研究启发与可借鉴点
- **"有效性免费"原则的迁移**：任何在 split 前完全确定的 predictor（无论是否训练）均可享受引理 1 的有限样本有效性；这对 prompt-based LLM classifier、few-shot vision classifier 的 conformal 分析有直接启发。
- **adjusted homophily 门控设计**：ĥ_adj 消除类别不平衡对 raw h_edge 的 inflate 效应，这一思想可推广到任何需要"基于已有标签决定扩散/聚合强度"的图算法（如 GNN 消息传递、label propagation 的自适应性）。
- **陷阱案例的预先注册**：H4 在运行前锁定阈值和决策规则，避免 p-hacking；对可靠性论文的复现性建设是典范做法。
- **Transductive exchangeability 的工程检验**：只需确认"分数矩阵所有组件在 split 前固定"这一 checklist，即可跳过对训练协议/早停/调参的详细审计，大幅降低 conformal pipeline 的工程成本。
- **与团队方向的结合机会**：若团队在做图基础模型的零样本/少样本部署，可直接套用本文的 TabICL + HG-DAPS pipeline 作为 baseline；对需要"自动 referral"的工业场景（如风险评分、异常检测），本文的 conformal 框架比单纯准确率更有部署价值。

## 关键术语表
- **Tabular Foundation Model (TFM)**：在合成表任务上预训练的模型（如 TabICL、TabPFN），通过 in-context learning 对新表做分类，无需梯度更新。
- **Conformal Prediction**：给出带有限样本覆盖率保证的预测集（而非单点预测）的框架，核心思想是用 nonconformity score 的分位数划定集合。
- **Split Conformal Prediction**：将数据分为 calibration set 和 test set，在 calibration 上拟合分位数阈值，在 test 上构造预测集；本文核心工具。
- **APS (Adaptive Prediction Set)**：按预测概率降序排列标签，分数 = 排在当前标签前的概率质量 + 随机份额；本文使用的非 conformity 分数。
- **Expected Calibration Error (ECE)**：用 15 等频 bin 估计的校准误差，衡量模型置信度与真实准确率之间的偏差；越低越好。
- **Adjusted Homophily (ĥ_adj)**：Platonov et al. 提出的同配性统计量，消除类别不平衡 inflate；ĥ_adj ≈ 0 表示边与标签无关，=1 表示完全同配。
- **HG-DAPS (Homophily-Gated DAPS)**：本文提出的方法，用 ĥ_adj 门控 score 扩散权重 δ，在同配图上缩小预测集、在异配图上退化回 APS。
- **Transductive Exchangeability**：在固定有限总体下，uniform random split 使 calibration + test 对交换不变；是 Lemma 1 有效性的根本前提。

## 可复现要素
- **数据集**：10 个标准 benchmark（Cora、CiteSeer、PubMed、Photo、Coauthor-CS、WikiCS、Actor、Amazon-Ratings、Minesweeper、Tolokers），部分为公开 benchmark，Minesweeper/Tolokers 来自 [27]。
- **代码/权重**：论文未提供开源仓库；TabICL 为已有开源模型，但 conformal pipeline 和 HG-DAPS 的实现细节未在论文中开源声明。
- **关键超参**：α=0.10，δ_max=0.5，m_0=100，SVD 维度 64，上下文 ≤5000 行，calibration 集 n_cal=min(1000, ⌊|U|/2⌋)，3 次上下文 seed × 20 次 calibration/test resplit。
- **硬件**：一次 8 GB 消费级 GPU 推理，后续 CPU 步骤秒级完成。
