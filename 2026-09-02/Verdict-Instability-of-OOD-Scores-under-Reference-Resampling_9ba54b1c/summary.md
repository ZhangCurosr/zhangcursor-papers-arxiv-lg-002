---
title: "Verdict-Instability-of-OOD-Scores-under-Reference-Resampling"
source: https://arxiv.org/pdf/2609.00691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:29:51"
field: "分布外检测与不确定性量化"
keywords: ["OOD detection", "verdict instability", "reference resampling", "post-hoc detectors", "selective prediction", "epistemic uncertainty", "bootstrap variance"]
innovations: ["提出判决不稳定性的无参闭式估计，量化参考集有限性对OOD分数的影响", "发现距离型OOD分数与判决可靠性负相关，仅色散型分数携带正确符号", "建立无标签符号预测规则，证明错误符号拒绝差于随机拒绝"]
benchmarks: ["CIFAR-100", "CIFAR-10", "DermaMNIST"]
---

# 论文速读：Verdict-Instability-of-OOD-Scores-under-Reference-Resampling

## 一句话总结
本文度量了后验 OOD 检测器在参考集 Bootstrap 重采样下的判决不稳定性（verdict instability），推导出不含自由参数的闭式估计公式，并发现常见距离型 OOD 分数与判决可靠性呈负相关——基于错误符号分数进行拒绝决策的表现甚至差于随机拒绝。

## 研究问题与动机
- **核心问题**：OOD 分数是基于有限参考集拟合的估计量，换一个参考集会改变部分查询的分数，但现有工作从未量化这一"有限参考集驱动的不稳定性"。
- **动机 1**：OOD 分数常被当作不确定性估计用于选择性预测（abstention），但检测器的认知不确定性（epistemic uncertainty）与模型自身的预测不确定性是两个不同概念，后者总是前者被忽略。
- **动机 2**：已有几何分析（Janiak et al., 2026）研究的是固定参考集下查询方向的方差，而本文研究固定查询下参考集变化的方差，两者权重方向相反。
- **动机 3**：标准评测协议每个类有相同参考数，无法识别 $n_c^{-1/2}$ 依赖关系；需要利用自然类别不平衡（如 DermaMNIST 58.7× 跨度）来验证。

## 核心贡献（创新点）
- **提出判决不稳定性的闭式度量**：给出了第 (1) 式，将不稳定性分解为类通道（横向色散除以 $\sqrt{n_c}$）和全局惩罚通道，无任何拟合参数，$R^2 = 0.82–0.97$ 追踪 Bootstrap 方差。
- **在自然类别不平衡下验证 $n_c^{-1/2}$ 依赖**：利用 DermaMNIST 58.7× 的参考数跨度，证明去掉 $n_c$ 将 $R^2$ 从 0.923 降至 0.276，这是其他 OOD 分数分析不具备的识别条件。
- **建立距离型与色散型分数的符号分离**：发现 11 个后验分数中仅 2 个（knn_std、lid）携带正向符号，其余 9 个距离型和 logit 型分数均与判决可靠性负相关；给出仅需一次无标签相关计算的符号预测规则（第 (9) 式）。
- **揭示错误符号拒绝的代价**：证明基于错误符号分数进行的 abstention 表现差于随机拒绝（AURC 最高增加 +13.09%），且该失败在两个数据集和十个种子下均稳定成立。
- **区分三类"不确定性"**：指出 OOD 检测文献将"分类器犯错概率"、"输入陌生程度"和"判决可复现性"三个正交量混为一谈，并在真实嵌入上证明它们几乎是反序的。

## 方法详解
**1. 判决不稳定性的定义**：固定查询 $x$，对参考集 $\mathcal{R}$ 做类条件 Bootstrap 重采样（保持 $n_c$ 不变），定义 $T(x) = \mathrm{std}_{\mathcal{R}^*}[s(x;\mathcal{R}^*)]$，表示"若收集到不同参考数据判决会移动多少"。

**2. 闭式分解（第 (7) 式）**：
$$\widehat{T}(x) = \sqrt{\frac{\sigma_t(u)^2}{n_c} + \lambda^2 s_D^2 \cdot v\!\left(\frac{\tau - D(x)}{s_D}\right)}$$
- **类通道**：$\sigma_t(u)^2 = u^\top \Sigma_c u$ 是将类内散度投影到查询方向 $u$ 的横向色散，除以该类参考数 $n_c$。
- **惩罚通道**：来自全局距离的 hinge 项，仅对靠近阈值 $\tau$ 的查询（elbow 处）有显著贡献，大多数远 OOD 查询贡献为零。

**3. 符号预测规则（第 (9) 式）**：$\mathrm{sign}[\rho(M, T)] = \mathrm{sign}[\rho(M, \widehat{T})]$，右侧 $\widehat{T}$ 仅依赖参考集，无需 Bootstrap 和标签， practitioner 可在部署前验证任意新分数的符号方向。

**4. 符号成因的几何解释**：远 OOD 查询位于各向异性嵌入的低方差特征方向，导致距离型分数将"最可复现的判决"标记为"最可疑"；只有直接估计局部色散的分数才携带预期符号。

**5. 判决翻转概率**：若分数在重采样下近似高斯，判决翻转概率约为 $\Phi(-|s(x)-t|/T(x))$，不稳定性是衡量 margin 的自然分母。

## 实验与结果
- **数据集**：CIFAR-100（50 类，每类 400 参考）、CIFAR-10（5 类，每类 400 参考）、DermaMNIST（3/5/7 类，参考数 80–4693，不平衡比最大 58.7×）；Far-OOD 来源包括 SVHN、DTD、LSUN、Places365 等。
- **Backbone**：DINO 预训练的 ViT-B/16（冻结），$d=768$；对照实验还使用了 CLIP ViT-B/16 和 ResNet-50。
- **11 个后验分数**：knn_std、lid（色散型）；d_cls、knn、maha、vim（距离型）；energy、maxlogit、odin、msp、entropy（logit 型）。
- **闭式估计精度**（Table 1）：CIFAR-100 $R^2=0.820$、CIFAR-10 $R^2=0.918$、DermaMNIST C=3 $R^2=0.974$、C=7 $R^2=0.923$；尺度比 $T/\widehat{T}$ 中位数接近 1。
- **参考数依赖验证**（Appendix D）：去掉 $n_c$ 用均值替代后，DermaMNIST C=7 的 $R^2$ 从 0.923 降至 0.276，丧失 70% 解释力。
- **符号分布**（Figure 2/Table 7）：11 个分数中仅 knn_std（$\rho=+0.695$）和 lid（$\rho=+0.483$）为正，其余 9 个均为负。
- **拒绝代价**（Table 3）：CIFAR-100 上 maha 最差（$\Delta=+3.93\%$），DermaMNIST 上 maha 同样最差（$\Delta=+13.09\%$）；而 knn_std 和 lid 均显著优于随机拒绝（最低 $\Delta=-19.70\%$）。
- **符号稳健性**（Appendix H）：更换预训练目标（CLIP）或架构（ResNet-50）后，55 个 score-setting 组合中 52/54 个符号保持不变，规则预测准确率 52/54。
- **Mahalanobis 分解渠道**（Table 13）：Size、Stretch、Product 三个渠道对判决不稳定性的相关性均为负，其中 Stretch 渠道（Janiak et al. 优化的部分）在 DermaMNIST 上造成 +12.75% 的拒绝代价。

## 相关工作脉络
- **Mahalanobis 检测器几何分析**（Janiak et al., 2026）：分析特征是固定参考集下查询方向的方差，本文分析是固定查询下参考集的方差，两者权重方向相反；本文指出其优化方向（stretch channel）恰与判决可复现性负相关。
- **后验 OOD 检测综述**（Yang et al., 2022, 2024; Fang et al., 2022; OpenOOD）：本文使用相同的 11 个标准分数，但评估目标不同——不是区分 in/OOD 的能力，而是作为估计量的统计稳定性。
- **选择性子预测与风险-覆盖率曲线**（Geifman & El-Yaniv, 2017; Geifman et al., 2019）：需要标签，仅适用于 in-distribution 区域；本文的不稳定性度量无需标签，可评估 far-OOD 区域。
- **不确定性的 Aleatoric/Epistemic 分解**（Kendall & Gal, 2017; Mukhoti et al., 2023）：OOD 分数常被当作 epistemic uncertainty 代理，但本文指出检测器的 epistemic 不确定性方向可能与模型自身的不确定性相反。
- **Bootstrap 与协方差收缩**（Efron & Tibshirani, 1993; Ledoit & Wolf, 2004）：经典统计学工具，本文首次将其系统应用于 OOD 分数的参考集重采样分析。
- **Conformal Prediction**（Vovk et al., 2005; Angelopoulos & Bates, 2023）：量化校准集有限性的影响，本文是其在 OOD 检测领域的对应缺失陈述。

## 局限性与未来方向
- **各向同性嵌入下符号消失**：若嵌入无各向异性（如经 whitening），距离型分数与不稳定性的相关性趋近于零，规则失去预测对象；这限制了在特定特征空间上的通用性。
- **仅分析 Mahalanobis 族分数**：闭式推导基于类均值距离 + 全局惩罚的结构，对于完全不同类型的检测器（如密度估计、对比学习直接评分）是否适用需进一步验证。
- **Far-OOD 区域的标签缺失本质**：虽然不稳定性无需标签，但"判决是否正确"仍是一个正交问题；本文明确不声称对正确性的任何判断，实际部署中仍需结合其他信号。
- **DermaMNIST 跨模态对齐**：医疗图像（28×28）与自然图像 OOD 源的对齐依赖分辨率下采样，可能引入预处理 artifacts，尽管作者已通过控制实验缓解。
- **未来方向**：将不稳定性度量纳入 selective prediction 的完整框架、扩展至生成式 OOD 检测器、探索在少样本/长尾场景下的自适应参考集管理。

## 研究启发与可借鉴点
- **Bootstrap 重采样作为检测器认知不确定性的度量工具**：可将此方法迁移到任何其他基于有限参考集的后验检测器（如 kNN-OOD、LOF、VIM），系统性评估其判决稳定性而非仅评估 AUC。
- **类别不平衡作为识别条件**：利用自然不平衡（而非人工 subsampling）来识别 $n_c$ 依赖是一个聪明的实验设计，可推广至任何需要对"样本量效应"进行因果识别的场景。
- **符号规则的工程价值**：$\mathrm{sign}[\rho(M, \widehat{T})]$ 仅需一次无标签相关计算即可判断任意新分数是否适合用于 abstention，可作为部署前的"健康检查"模块。
- **距离型 vs 色散型分数的分离洞察**：提示在构建新的 OOD 分数时，若目标是"识别不可靠判决"，应优先直接建模局部色散而非最小化距离；这为设计新型鲁棒 abstention 策略提供方向。
- **AURC 替代 Risk-Coverage 用于无标签场景**：引入 instability-coverage 曲线和 $\Delta$ 指标，在无需 ground-truth 的前提下评估 abstention 策略，可推广至任何无标签选择性推理场景。

## 关键术语表
- **Verdict Instability（判决不稳定性）**：固定查询下，对参考集做 Bootstrap 重采样后 OOD 分数的标准差，度量判决的可复现性。
- **Transverse Dispersion（横向色散）**：类内协方差沿查询方向的投影 $\sigma_t(u)^2 = u^\top \Sigma_c u$，是闭式估计的核心组成部分。
- **Class Channel / Penalty Channel（类通道/惩罚通道）**：不稳定性的两大部分，前者来自类均值估计的方差（$\propto 1/n_c$），后者来自全局距离估计经 hinge 整流后的方差。
- **Instability-Coverage Curve（不稳定性-覆盖率曲线）**：类似 risk-coverage 曲线但以判决不稳定性为目标，评估 abstention 策略对判决可靠性的影响，无需标签。
- **Sign Separation（符号分离）**：距离型分数与判决不稳定性负相关、色散型分数正相关的现象，源于嵌入的各向异性几何。
- **Reference Count Dependence（参考数依赖）**：不稳定性随类参考数按 $n_c^{-1/2}$ 衰减的规律，是本文区别于所有现有 OOD 分析的核心识别特征。
- **Epistemic vs Aleatoric Uncertainty（认知不确定性 vs 偶然不确定性）**：前者源于有限数据（可被更多参考点减少），后者源于数据固有噪声；本文度量的是检测器的 epistemic 不确定性。
- **Selective Prediction（选择性预测）**：模型可在置信度不足时拒绝预测，传统框架依赖标签定义风险，本文将其扩展至无标签的不稳定性度量。

## 可复现要素
- **数据集**：CIFAR-100、CIFAR-10、DermaMNIST（public）、SVHN、DTD、LSUN、Places365 均为公开数据集；DermaMNIST 为 CC BY 4.0，HAM10000 为 CC BY-NC 4.0。
- **代码/权重**：ViT-B/16 DINO checkpoint 来自 timm（`vit_base_patch16_224.dino`）；论文未声明开源代码仓库。
- **关键超参**：$\lambda = 5$，$\tau$ 为全局距离的 20th percentile；knn 取 5-nearest；lid 用 20 点 MLE；vim 子空间维度 64；maha 收缩 toward scaled identity 系数 0.3；odin 温度 $T=1000$、扰动 $\varepsilon=0.0014$。
- **Bootstrap 重复数**：$B = 200$。
- **种子数**：§6 结果取 10 个种子均值。
- **Compute**：单张 NVIDIA RTX 4090 24GB + 16 cores CPU，完整五设置复制耗时约 15 分钟。
