---
title: "The-information-geometry-of-large-language-models-is-shared"
source: https://arxiv.org/pdf/2609.11063v1.pdf
model: agnes-2.5-flash
chunks: 6
summarized_at: "2026-10-01 10:43:08"
---

# 论文速读：The-information-geometry-of-large-language-models-is-shared

## 一句话总结
论文证明大型语言模型输出分布具有由行为充分统计量决定的跨架构共享Fisher-Rao几何，该几何随规模与训练提升而与人类补全分布对齐，并可借阻尼自然梯度实现最小扰动、跨提示符复用的可控干预。

## 研究问题与动机
- **激活几何与输出几何的本质差异**：内部激活的欧氏距离依赖坐标且随可逆线性重参数化改变，无法提供架构无关的行为表征；需要一种由概率分布自身决定的本征几何。
- **跨架构一致性缺乏理论保证**：Transformer、SSM、RNN等不同家族模型的输出分布是否共享同一几何结构尚无严格刻画，现有工作多停留在经验对比。
- **干预与编辑的高扰动代价**：现有知识编辑与Steering方法在激活空间盲目优化，忽略原生输出几何，导致目标偏移的同时引入大量非目标区域失真。
- **训练获取轨迹难以无参数预测**：现有Scaling Law多依赖曲线拟合，缺少利用语料统计先验直接预测未见事实何时、以何种层级速度被模型获取的方法。

## 核心贡献（创新点）
- **证明输出Fisher几何跨架构共享**：揭示Transformer/SSM/RNN的输出Fisher-Rao度量高度一致（Spearman 0.88），与基于坐标的激活几何（~0.6）形成本质区别，因前者由Chentsov定理保证在充分统计变换下的唯一性。
- **提出无参数谱预测与干预成本估计**：利用降序谱值构造$\widehat{N}_{\text{eff}}(\alpha)$与$\Pi_G(u)$，无需拟合即可预测有效维度与Fisher相对Euclidean的控制优势比，区别于以往需训练代理模型的估计范式。
- **建立语料n-gram对获取轨迹的非参数预测**：证明训练前unigram/bigram/trigram margin可直接预测未见事实的对数获取时间（$R^2\approx0.79$）与深度因果延迟，区别于依赖事后拟合的动态曲线方法。
- **设计几何规定的最小扰动可复用干预**：通过$damped\ natural\ gradient\ \delta h\propto(G+\alpha R)^{-1}q$在Fisher度量下实现指定输出变化的最小局部成本，并支持从4个donor提示符学习共享更新应用于8个unseen target，区别于逐提示符独立优化的编辑管线。
- **推导不依赖交换性假设的衰减恒等式与一致性下界**：给出六标量恒等式与基于root-affinity风险的衰减地板定理，量化多模型输出的相关性边界，区别于传统可靠性理论中对齐次/可交换运行的强假设。

## 方法详解
- **输出Fisher度量构造**：$H_h = \text{Cov}_{p_h}(w)$，其中$p_h = \text{softmax}(W_U h_L(h) + b_U)$，$w_a$为unembedding行向量；通过雅可比拉回至干预层$G_h = J_h^\top H_h J_h$。
- **局部KL近似**：$\text{KL}(p_h \| p_{h+\delta h}) = \frac{1}{2}\delta h^\top G_h \delta h + O(\|\delta h\|^3)$，作为干预成本的二阶代理。
- **阻尼自然梯度更新**：$\delta h \propto (G + \alpha R)^{-1}q$，坐标协变；$R=I$时为Euclidean基准，$\alpha$控制正则强度。
- **行为识别与读出面定位**：profiled predictive risk满足$\mathcal{K}(S) \ge c\, d^2(S,T)$，零风险子空间精确对应语言决定的read-out子空间（$R^2 \ge 0.9999$）。
- **跨模型几何比较**：在共同上下文中计算pairwise Fisher–Rao距离矩阵，仅比较上三角Spearman相关系数，无需共享词表或激活坐标。
- **有效维度预测**：$\widehat{N}_{\text{eff}}(\alpha) = \sum_{k=1}^r \frac{q_{(k)}}{q_{(k)} + \alpha}$，$q_{(k)}$为降序排列的$p_a\|u_a - \bar{u}\|^2$，零参数拟合。
- **干预成本预测**：$\Pi_G(u) = (u^\top G u)/(q^\top u)^2$，比率$R_{\text{pred}}(q) = \Pi_G(q)/\Pi_G(A^{-1}q)$预测各向异性比优势。
- **n-gram轨迹预测**：测量训练前unigram/bigram/trigram margin，直接回归未见事实的$\tau = \log_2 t$获取时间。
- **可重用更新策略**：在4个reference提示符上平均Fisher度量，学习共享激活更新后直接应用于8个unseen target提示符，无需target梯度。
- **衰减理论与一致性边界**：Theorem 10给出六标量恒等式$\mathrm{Corr}(C_m,C_n)=\frac{\nu_S+c_m+c_n+k_{mn}}{\sqrt{(\nu_S+\nu_m+2c_m)(\nu_S+\nu_n+2c_n)}}$；Lemma 3限定root-亲和风险；Theorem 11导出衰减地板$\mathrm{Corr}(C_m,C_n)\ge\frac{u^2-u(w_m+w_n)-w_mw_n}{(u+w_m)(u+w_n)}$。

## 实验与结果
- **跨架构输出几何一致性**：10个模型（70M–7B，6家族）自然文本输出几何Spearman等级相关**0.88**；mid-layer激活0.62，last-layer激活0.61；跨标记器first-byte共识**0.91**。
- **行为识别与局部KL预测**：四档模型每路径$R^2 \ge 0.9999$；99个model–depth–prompt细胞的局部KL预测中位数/范围**1.000 / 0.948–1.224**。
- **谱与有效维度预测**：谱指数Pearson/Spearman **0.968 / 0.955**（rank 8–32，576 cells）；有效维度RMSE **0.0270–0.0521**，较flat spectrum优**6.44–7.88倍**。
- **n-gram获取预测**：70M/160M/410M的轨迹$R^2$为**0.790 / 0.792 / 0.775**；获取状态一致性0.792/0.854/0.807；连续获取时间误差（log₂步）**0.77 /
