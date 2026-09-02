---
title: "Verdict-Instability-of-OOD-Scores-under-Reference-Resampling"
source: https://arxiv.org/pdf/2609.00691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:04:44"
field: "分布外检测与不确定性量化"
keywords: ["OOD detection", "verdict instability", "bootstrap", "post-hoc detector", "uncertainty estimation", "selective prediction", "reference resampling"]
innovations: ["给出无参闭式近似刻画 OOD 分数的判决不稳定性，Bootstrap R² 达 0.82–0.97", "利用天然类别不平衡唯一识别 1/√n_c 依赖，揭示距离型与色散型分数的符号分离", "提出 label-free 符号预测规则，证明误符号 abstention 比随机更差"]
benchmarks: ["CIFAR-100", "CIFAR-10", "DermaMNIST"]
---

# 论文速读：Verdict-Instability-of-OOD-Scores-under-Reference-Resampling

## 一句话总结
本文首次量化了后验 OOD 检测器因参考集有限而产生的**判决不稳定性（verdict instability）**——即更换参考集样本时判决的波动程度，给出了无参闭式近似公式，并由此揭示主流距离型 OOD 分数与判决可靠性之间存在反向符号关系，导致基于此类分数做 abstention 的效果甚至不如随机。

## 研究问题与动机
- **核心问题**：后验 OOD 分数是基于有限参考集估计出的值，不同参考集会导致不同分数和判决；但现有研究从未系统度量这一有限样本不确定性，也未考察它如何影响分数行为。
- **现有方法不足**：
  - 所有主流后验 OOD 分数（Mahalanobis、kNN、Energy、logit-family 等）均以"分离 in/out 分布"为目标优化，但未将其视为参考集估计量来分析其方差。
  - 现有几何分析（Janiak et al., 2026）的方差是固定参考集下沿查询方向的变化，不依赖于参考集大小，与本文的视角互补但正交。
  - Selective prediction 的风险–覆盖曲线需要标签，无法在 far-OOD 区域评估；而判决不稳定性无需标签，天然适用于该区域。
  - 业界将 OOD 分数等同于 epistemic uncertainty 并用于 abstention 的做法缺乏理论依据，本文证明这种等同在某些情况下是危险的（符号相反）。

## 核心贡献（创新点）
1. **判决不稳定性的定义与闭式近似**：将 OOD 分数的 bootstrap 方差定义为判决不稳定性，给出无拟合参数的闭式公式（Eq. 1），在 CIFAR-100/10、DermaMNIST 上达到 $R^2 = 0.82 \sim 0.97$。
2. **识别参考计数 $1/\sqrt{n_c}$ 依赖**：利用 DermaMNIST 天然类别不平衡（58.7×），在无额外假设下唯一识别出 $n_c$ 的作用——替换为均值后 $R^2$ 从 0.923 跌至 0.276；其他任何 OOD 分数分析均不依赖此计数项。
3. **符号分离规律与 label-free 预测规则**：发现距离型分数与判决不稳定性负相关，色散型分数正相关；提出单条 label-free 相关性规则（Eq. 9），可从单一参考集预测任意新分数的符号方向。
4. **误符号 abstention 的代价量化**：证明基于错误符号分数进行 abstention 比随机 abstention 更差（CIFAR-100 上最大损失 +3.93%，DermaMNIST 上 +13.09%），而正确符号的色散分数带来最高稳定性收益。

## 方法详解
- **判决不稳定性定义**：固定查询 $x$，对参考集 $\mathcal{R}$ 做类内无放回 bootstrap（保持每类计数 $n_c$），得到 $B=200$ 个复刻，定义 $T(x) = \mathrm{std}_{\mathcal{R}^*}[s(x;\mathcal{R}^*)]$。
- **两个方差通道**：以简化检测器 $s(x) = \min_c \|z - \hat{\mu}_c\| + \lambda[\tau - \|z-\hat{\mu}_0\|]_+$ 为例，方差分解为独立的两项：
  - **类通道（class channel）**：$\mathrm{Var}[s_\mathrm{class}] \approx \sigma_t(u)^2 / n_c$，其中 $\sigma_t(u)^2 = u^\top \Sigma_c u$ 是查询方向上的类内横向色散（transverse dispersion），无需重采样即可计算。
  - **惩罚通道（penalty channel）**：由全局 hinge 整流高斯分布产生方差 $\lambda^2 s_D^2 \cdot v(m/s_D)$，仅对落在 hinge 边缘的少数查询有贡献。
- **闭式近似（Eq. 7）**：$\widehat{T}(x) = \sqrt{\sigma_t(u)^2/n_c + \lambda^2 s_D^2 \cdot v((\tau-D(x))/s_D)}$，无任何拟合参数，全部为参考集的 plug-in 统计量。
- **判决翻转概率**：若分数在重采样下近似正态，则判决在阈值 $t$ 处翻转的概率约为 $\Phi(-|s(x)-t|/T(x))$。
- **符号预测规则（Eq. 9）**：$\mathrm{sign}[\rho(M, T)] = \mathrm{sign}[\rho(M, \widehat{T})]$，只需参考集即可计算右侧，无需 bootstrap 和标签。
- **几何解释**：far-OOD 查询位于各向异性嵌入的低方差特征方向，因此距离越小（越"像"该类）反而意味着横向色散 $\sigma_t$ 越大、判决越不稳定，导致距离型分数与可靠性负相关。

## 实验与结果
- **数据集**：CIFAR-100（C=50，$n_c=400$）、CIFAR-10（C=5，$n_c=400$）、DermaMNIST（C=3/5/7，$n_c$ 从 80 到 4693，不平衡比达 58.7×）；远 OOD 源包括 SVHN、DTD、LSUN、Places365 等。
- **骨干网络**：冻结的 ViT-B/16（DINO 预训练），$d=768$；另外验证了 CLIP ViT 和 ResNet-50。
- **评估基线**：11 种后验分数（knn_std、lid、d_cls、knn、maha、vim、energy、maxlogit、odin、msp、entropy）。
- **核心结果**：
  - 闭式近似 vs bootstrap 真值（Table 1）：CIFAR-100 $R^2=0.820$，CIFAR-10 $R^2=0.918$，DermaMNIST C=3 $R^2=0.974$，C=7 $R^2=0.923$；中位数 $T/\widehat{T}$ 均在 0.95–1.005。
  - 移除 $n_c$ 的代价（Table 6）：不平衡 58.7× 时 $R^2$ 从 0.923 跌至 0.276（损失 70% 解释力）。
  - 11 种分数中 9 种与不稳定性负相关（Fig. 2, Table 7）。
  - Abstention 效果（Table 3）：knn_std 最优（CIFAR-100 Δ=−6.99%，DermaMNIST Δ=−19.70%）；maha 最差（CIFAR-100 Δ=+3.93%，DermaMNIST Δ=+13.09%），均显著优于随机。
  - 跨 10 个 seed 的结果符号一致性 10/10（Appendix K）。
  - 规则 Eq. 9 在 55 个 score-setting 组合中预测符号正确 52–54 次（Appendix H）。
  - Stretch channel（Janiak et al. 优化的分量）导致 DermaMNIST 上 Δ=+12.75%，比随机更差（Table 13）。

## 相关工作脉络
- **Post-hoc OOD 检测（Hendrycks & Gimpel, 2017; Lee et al., 2018; Ren et al., 2021; Sun et al., 2022 等）**：这些工作关注检测性能（AUC 等），本文从估计量方差角度重新审视同一组分数。
- **Mahalanobis 几何分析（Janiak et al., 2026）**：最近最接近的工作；其方差是固定参考集下查询的变化，无 $n_c$ 依赖；本文证明其优化的 stretch 分量与判决稳定性反相关（近于对偶）。
- **Uncertainty 分解（Kendall & Gal, 2017; Gal & Ghahramani, 2016; Mukhoti et al., 2023）**：将 aleatoric/epistemic 不确定性归因于模型本身，本文指出 OOD 检测器的 epistemic uncertainty 是其自身的有限样本方差，二者不同。
- **选择性预测（Geifman & El-Yaniv, 2017; El-Yaniv & Wiener, 2010）**：风险–覆盖曲线需标签，局限于 in-distribution 区域；本文的 instability–coverage 曲线无需标签且可评估 far-OOD。
- **Bootstrap 与协方差收缩（Efron & Tibshirani, 1993; Ledoit & Wolf, 2004; Vovk et al., 2005）**：传统统计中的有限样本不确定性工具，本文首次将其应用于 OOD 分数的系统分析。
- **特征空间密度方法（Ma et al., 2018; Wang et al., 2022）**：KNN-based 和 Vim 等方法，本文发现它们属于不同的符号族（色散型 vs 距离型）。

## 局限性与未来方向
- **公式适用性**：闭式推导基于线性化近似（忽略 $O(n_c^{-3/2})$ 项），对小 $n_c$ 或极端查询可能精度下降（Appendix B 显示误差 <0.4%）。
- **各向异性嵌入假设**：符号分离规律依赖于嵌入的各向异性；在 isotropic 空间中符号消失，规则失去意义（§5.3）。
- **仅分析 Mahalanobis 族**：虽然推广到 11 种分数，但闭式推导仅针对含 class term + global penalty 的框架；其他架构的分数未作形式化分析。
- **未考虑标签噪声**：参考集中的标注错误会同时影响 $\hat{\mu}_c$ 和 $\hat{\Sigma}_c$，本文未讨论其对不稳定性的影响。
- **未来方向**：将 instability 度量扩展至更多检测器架构；在 isotropic 或对抗训练中嵌入的情形下研究符号规律；探索结合判决稳定性与检测性能的联合优化目标。

## 研究启发与可借鉴点
- **可复用的方差的分解思想**：将 OOD 分数方差拆分为"类内项"和"全局项"两条独立通道，各自分析其 bootstrap 行为，方法简洁且可迁移到其他估计量分析。
- **利用天然类别不平衡识别隐式参数**：通过 DermaMNIST 的 58.7× 不平衡在单模型中识别 $n_c$ 的作用，避免了人为子采样带来的分布偏移，是"自然实验"设计的优秀范例。
- **label-free 符号预测规则的普适性**：Eq. 9 允许在新分数部署前快速验证其方向是否正确，可作为 OOD 分数筛选的实用前置检查工具。
- **与本团队的结合机会**：可将判决不稳定性度量集成到选择性预测框架中，为 far-OOD 区域提供无需标签的置信度评估；亦可将此概念引入医学图像（如 DermaMNIST 场景）的稳健性评估流程。
- **对现有几何分析的补充视角**：Janiak et al. 的 stretch/size 分解从检测性能角度优化，本文证明该优化方向与判决可靠性相反——两者结合可设计出兼顾检测精度与判决稳定性的新目标函数。

## 关键术语表
- **Verdict Instability（判决不稳定性）**：固定查询、对参考集做 bootstrap 重采样时 OOD 分数的标准差，度量判决对参考集有限性的敏感程度。
- **Transverse Dispersion（横向色散）**：查询方向 $u$ 上的类内方差 $\sigma_t(u)^2 = u^\top \Sigma_c u$，是判决不稳定性的分子部分。
- **Class Channel（类通道）**：由类质心 $\hat{\mu}_c$ 的重采样方差贡献的不稳定性分量，按 $1/n_c$ 缩放。
- **Penalty Channel（惩罚通道）**：由全局质心 $\hat{\mu}_0$ 及 hinge 激活方差贡献的不稳定性分量，仅在查询靠近 hinge 边缘时有显著贡献。
- **Instability–Coverage Curve**：类比风险–覆盖曲线的无标签版本，横轴为覆盖率 $\kappa$，纵轴为保留子集的判决不稳定性均值，AURC 衡量 abstention 带来的稳定性收益。
- **Sign Rule（Eq. 9）**：任意 OOD 分数 $M$ 与真实不稳定性 $T$ 的相关性符号，等于其与 plug-in 估计 $\widehat{T}$ 的相关性符号，可在无 bootstrap 的情况下验证。
- **Selective Prediction（选择性预测）**：在不确定时拒绝预测以提升覆盖率内准确率的框架，本文指出其依赖标签因而无法评估 far-OOD 区域。
- **Far-OOD**：与训练分布完全无关的远端分布（如 SVHN、DTD），区别于 near-OOD（同数据集但未出现在训练中的类别）。

## 可复现要素
- **数据集**：CIFAR-100、CIFAR-10（公开）；DermaMNIST / HAM10000（CC BY-NC 4.0，公开）；远 OOD 源 SVHN、DTD、LSUN、Places365、iSUN 均公开。
- **代码/权重**：论文未提供开源代码；ViT-B/16 DINO 权重通过 timm 公开（`vit_base_patch16_224.dino`）。
- **关键超参**：$\lambda = 5$，$\tau$ 取全局距离的 20th 分位；kNN 取第 5 近邻；LiD 用 20 邻域的 MLE；ViM 取 64 维主成分子空间；Mahalanobis 收缩因子 0.3（Ledoit-Wolf）；Odin 温度 $T=1000$，$\varepsilon=0.0014$；bootstrap 复刻数 $B=200$；10 个随机 seed。
- **计算环境**：单卡 NVIDIA RTX 4090，16 核 CPU，30 GB 系统内存；完整实验约 15 分钟。
