---
title: "Target-alignment-dilution-and-forecast-selection-when-cross"
source: https://arxiv.org/pdf/2609.26303v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:26:22"
field: "预测组合与选择"
keywords: ["forecast combination", "target alignment", "forecast diversity", "cross-sectional ranking", "language models", "equal-weight admission", "dilution", "HAC selection"]
innovations: ["标准化预测相对共同目标的精确正交分解（对齐项 + 残差项）", "等权准入增量风险的 scale-free / scale-mismatch 二分", "带同时 HAC 带的三分法贪心准入规则"]
benchmarks: ["LLM equity-ranking panel (N=24, 5-day horizon)", "ETF mechanical-signal panel (42 funds, monthly)", "Simulation Designs A–G (phase diagram, instability mechanisms, misspecification)"]
---

# 论文速读：Target-alignment-dilution-and-forecast-selection-when-cross-sectional-forecasts-share-a-common-target

## 一句话总结
本文在截面预测共享同一标准化目标的设定下，推导出每个预测可精确分解为**目标对齐项**（与真实目标的相关性）与**目标正交残差项**；在此基础上揭示传统误差相关率无法反映真实多样性、等权组合仅在平均对齐足够大时才能超越无信息基准，并开发了带 HAC 置信带的三分法（接收/拒绝/待定）候选筛选规则。两个实证面板（语言模型预测美股排序 + ETF 机械信号）均未检测出可识别的对齐，等权加池产生稀释损失，选池后风险接近但未能战胜无信息预测。

## 研究问题与动机
- **核心问题：** 当每日期望对大量个体（资产/产品/区域）打分并统一到同一标准化 realized 目标时，基于"误差相关性=多样性"的经验法则为何失效？
- **标准化后的几何约束：** 每个 $s_{it}$ 与 $y_t$ 均为单位范数向量，导致误差偏差相关 $\rho^e_{ij}$ 被强制映射到 $[0,1]$，掩盖真实冗余（$\rho^e_{ij} \approx (1+\rho^s_{ij})/2$ 在无对齐时成立，分散度被压缩一半）。
- **等权平均的零基准陷阱：** 标准化预测的简单平均会将合成向量拉向零，除非平均对齐超过合成向量二阶范的一半（Corollary 1 的阈值 $\bar{\gamma} > \frac{1}{2}[\bar{\rho}+(1-\bar{\rho})/N]$），否则等权组合必输于无信息预测（$\mathcal{R}=1$）。
- **增量风险的双源结构：** 新增候选对等权池的贡献被精确拆分为 scale-free 的"真提升"和 scale-mismatch 的"稀释惩罚"，而等权规则在大池中加入小 batch 时对对齐敏感度的衰减达到 $(m/n)^2$，导致真正有预测力的候选人被忽略。

## 核心贡献（创新点）
- **几何恒等式的精确分解：** 把任意标准化预测 $s_{it}$ 写成 $\gamma_{it} y_t + u_{it}$（$\gamma_{it}=\langle s_{it},y_t\rangle_t$ 为目标对齐，$u_{it}\perp y_t$ 为正交残差），使相关矩阵满足 $C_t = \gamma_t\gamma_t'+K_t$——将误差方差结构从"统计估计"提升为"代数恒等"。
- **相对分数风险的封闭形式判据：** 给出任何线性组合 $w$ 的风险分解 $\mathcal{R}(w)=(1-w'\gamma)^2+w'Kw$，推导"战胜无信息预测"的充要条件 $g_w>\frac{1}{2}q_w$，以及样本内可达风险界 $1-\bar{\gamma}'\bar{C}^+\bar{\gamma}$。
- **等权准入增量风险的 scale-free / scale-mismatch 二分：** 证明 $\Delta_{A|P}=\Delta^{\mathrm{sf}}_{A|P}+\Delta^{\mathrm{scale}}_{A|P}$，前者度量合成相关提升，后者刻画池规模错配带来的稀释，从而给出等权规则为何在大池下对"对齐候选"失灵的定量上界。
- **HAC 三分法准入规则（admit / reject / undecided）：** 以同时最大-$t$ 临界值构建每步的保守置信带，嵌入滚动原点嵌套验证；对每个候选同时报告 $\hat{\Delta}$、$\Delta^{\mathrm{sf}}$ 与实际最小可检效应（MDE）。
- **双面板实证 + 正控制 + 日期置换安慰剂：** 在 LLM 排名预测与美国 ETF 机械信号两个独立面板上复现"无对齐→稀释主导→选池消除稀释但无法跨越 $\mathcal{R}=1$"的全链条证据，并用植入信号实验标定不同方法对不同对齐强度（$a\in[0,0.10]$）的检测力边界。

## 方法详解
- **标准化与目标对齐：** 每个日期对有效截面 $\mathcal{M}_t^*$ 按 $s_{it}=(x_{it}-\bar{x}_{it}\mathbf{1})/\hat{\sigma}_{x_i,t}$、$y_t=(r_t-\bar{r}_t\mathbf{1})/\hat{\sigma}_{r,t}$ 做零均值单位范数化；对齐 $\gamma_{it}=\langle s_{it},y_t\rangle_t$ 即为预测与目标的 Pearson 相关。
- **正交残差与相关矩阵分解：** $u_{it}=s_{it}-\gamma_{it}y_t$，满足 $\|u_{it}\|^2=1-\gamma_{it}^2$，且 $K_t=C_t-\gamma_t\gamma_t'\succeq 0$；rank 约束给出 $\mathrm{rank}(K_t)\le\min(N,M_t-2)$，在 $M_t$ 接近 $N$ 时矩阵趋近奇异。
- **风险分解与比例拆分（Proposition 2/3）：** $\mathcal{R}(w)=1-2g_w+q_w=(1-g_w/q_w)^2q_w+(1-q_w(1-c_w)^2/q_w)$ 等价的 $(1-\rho_w^2)+q_w(1-c_w)^2$，第一部分是缩放后的最优风险（仅依赖 pooled correlation），第二部分是尺度失配惩罚。
- **时间聚合与联合收缩（Proposition 6/7）：** $\bar{C}=G+\bar{K}$，$K^{\mathrm{pool}}=\bar{K}+\mathrm{Cov}_a(\gamma_t)$；对联合矩阵 $\widehat{R}=\begin{pmatrix}\bar{C}&\bar{\gamma}\\ \bar{\gamma}'&1\end{pmatrix}$ 做 Ledoit-Wolf 型等相关收缩 $\widetilde{R}=(1-\lambda)\widehat{R}+\lambda R_0$，保持 PSD 兼容（Proposition 7）。
- **嵌套滚动验证设计：** 每外轮 origin 取固定历史分 fit / 52 日 validation，调 ridge penalty、$\delta,\delta^{\mathrm{sf}}$、peLASSO 惩罚与 $\lambda$；再用全历史重算统计量冻结 pool，在随后 26 日 block 上做单次评估。Newey–West 滞后取 $\lfloor4(T/100)^{2/9}\rfloor$，max-$t$ 临界值从 HAC 估计下的 4000 次 bootstrap 抽取。
- **三分法决策带：** 当 $\hat{\Delta}+c\cdot SE<- \delta$ 接收；$\hat{\Delta}-c\cdot SE>0$ 拒绝；否则待定。Greedy 起点选历史对齐最高者，逐次加入 $\hat{\Delta}$ 最小的候选，直至无人被接收。
- **联合零对齐检验：** HAC Wald $\bar{\gamma}'\widehat{\Omega}^+\bar{\gamma}$ 配合 moving-block bootstrap（块长 4、999 次）与 max-$|t|$ 双重校准；同时在 Design B 中比较 asymptotic Wald 的表现（N=24 时超显著，nominal 5% 实际 53–70% 拒绝）。

## 实验与结果
- **面板 A（LLM）：** gpt-5-nano、deepseek-v4-flash、Llama-3.1-8B-Instruct、gemini-3.5-flash-lite × 3 persona（momentum / value-reversal / macro-defensive）× 2 info subset，N=24；60 只美股、5 日 horizon，2021.01–2026.06，261 个有效日期、157 测试日、6 外轮。
- **面板 B（ETF）：** 42 只跨资产 ETF，月度相对收益排名，9 条机械信号，2005–2026，256 个日期、196 测试日、16 外轮。
- **几何关键数字：**
  - 面板 A：$\bar{\gamma}=0.006\in[-0.010,0.020]$；$\bar{\rho}^s=0.300$，$\bar{\rho}^e=0.645$（距零对齐基准 0.650 仅差 0.005）；$R^2(\rho^e \text{ vs } \rho^s)=0.9999$；$K^{\mathrm{pool}}$ 中 $\mathrm{Cov}_a(\gamma_t)$ 占比 5.9%。
  - 面板 B：$\bar{\gamma}=-0.011$；$\bar{\rho}^s=0.149$，$\bar{\rho}^e=0.564$（距 0.575 差 0.010）；$\mathrm{Cov}_a(\gamma_t)$ 占比 14.1%。
- **选池效果：** 面板 A 等权全池 $\mathcal{R}=1.292$；三分法（等权基）均选 2 人（均值 $\rho^s=-0.78$），$\mathcal{R}=1.079$（vs 等权 $-0.213$，p<0.001）；exhaustive 6 人 $\mathcal{R}=1.033$；shrunk ridge weights $\mathcal{R}=1.019$；ridge projection $\mathcal{R}=1.000$，均不优于无信息基准。
- **面板 B 等效：** 等权全池 1.283；三分法/exhaustive 降至 1.076/1.061；shrunken affine 1.035；ridge projection 1.001；均显著高于 1。
- **正控制检测力（植入 $a\in\{0.02,0.04,0.06,0.10\}$）：** scale-free 三分法在 $a=0.10$ 召回 77%、精度 0.94；equal-weight 三分法召回 82%、精度 0.69；peLASSO 召回 99%、精度 0.82（含 12% 误入零载荷）。等权大池准入在 $a=0$ 即误接 24%（LLM 池）与 62%（信号池）。
- **模拟 phase diagram（Design D）：** 命中 Corollary 1 的精确线 $-2m$；$\bar{\gamma}=0.03,\bar{\rho}^s=0$ 时 $N=12$ 损 0.026、$N=24$ 增益 0.017；$\bar{\gamma}=0.15,\bar{\rho}^s=0.6,N=24$ 时可达 $-0.136$，ridge projection 近达到 $-0.132$。
- **不稳定机制对比（Design F）：** F1（时变对齐，无断点）各方法表现稳定；F2/F5（对齐突变/反转）均劣于无信息，仅 calibrated 全等权在 F2 略优；F4（仅依赖结构移位）ridge projection 最佳 $-0.051$，而等权劣于无信息（$+0.003$）。

## 相关工作脉络
- **歧义/bias–variance–covariance 分解（Krogh & Vedelsby 1995; Brown et al. 2005）：** 本文将其推广至相对分数的"target-alignment vs target-orthogonal"版本（Proposition 4），首次显式分离 disagreement 的目标无关性与 deviation correlation 的目标依赖。
- **forecast encompassing（Granger & Ramanathan 1984; Harvey & Newbold 2000）：** 增量对齐的 scale-free 部分对应等权版本的"增量 $R^2$ encompassing"统计量，batch 版本对应多重 encompassing，scale-mismatch 项则对应 Mincer–Zarnowitz（1969）的校准风险。
- **因子/特异成分分解（Lee & Lee 2026; Lee & Seregina 2026）：** 前者基于数据估计因子的"统计模型"；本文是相对于 realized target 的精确投影，orthogonal 分量对目标定义，而非对其他预测定义。
- **model-importance 度量（Budescu & Chen 2015; Kim et al. 2026）：** 在相对分数损失下，$\Delta_{A|P}$ 是 closed-form 的 importance 测度，且精确拆分为 scale-free + scale-mismatch 两半，并建立等权大池稀释上界。
- **Subset averaging / peLASSO（Diebold & Shin 2019; Bürgi & Sinclair 2017）：** 本文在同一几何框架下对比 greedy 三分、exhaustive、peLASSO、ridge projection 四路方案，给出各自在 stationary / break / dependence-shift 不同场景的胜负边界。
- **多检验与选择性推断（Romano & Wolf 2005; Berk et al. 2013）：** 本文承认 greedy 路径上的 per-step HAC 边界不会传播为 path-level 控制，明确仅依赖模拟评估，未做 selective-inference 校正——这构成与后续工作的明确分野。

## 局限性与未来方向
- **应用面板局限：** 面板 A 为 retrospective 静态截面（幸存者偏差）、单一 5 日 horizon、仅 6 外轮；面板 B 为已存续 ETF，单一月度 horizon，与面板 A 无共同特征却得相同 null。
- **过程调优暴露数据窥探：** 20 种 procedure 在同一数据上迭代优化，Bonferroni 阈值仅供参照；未施加 selective-inference 校正。
- **训练截止日前后不确定性：** deepseek/gemini 截至采样末，仅 Llama/gpt 可做 pre/post 对比，且差异不显著（t=-0.75/-1.17），无法排除 lookahead bias。
- **几何边界未穷尽：** 极端低秩（$M\approx N+2$）附近 pairwise deletion 生成 91–100% 不定矩阵，even shrinkage 仍残留 5–26%；文章仅报告经验行为，未给一般维数界。
- **未探索的未来方向：** 针对 greedy 自适应路径的 path-level 误差控制定理；针对非平稳/稀疏目标分布的拓展；将 scale-free 判定用于"是否值得从 scratch 重新建模"的 meta-decision。

## 研究启发与可借鉴点
- **对齐-正交分解可直接移植到任何"共享目标标准化"场景（如推荐系统 item ranking、多模型竞赛 leaderboard、气象集合预报）：** 只要最终评估量对目标做横截面去均值/标准化，此恒等式立即成立，无需额外假设。
- **"相关性≠多样性"的经验判断需配套报告三种矩阵：** $\rho^s$（forecast correlation）、$\gamma$（alignment）、$\rho^e$（deviation correlation），且应同时给出零对齐基准 $(1+\rho^s)/2$ 的距离；单一 $\rho^e$ 会系统性高估冗余。
- **三分法中的 scale-free 基在批量候选、大池场景下是保守且稳健的"不会错接"过滤器；equal-weight 基在候选人较少、计算预算有限时更灵敏——可根据业务成本（是否每次运行模型昂贵）二选一。**
- **联合收缩 $\widetilde{R}$ 的 PSD 保持引理（Proposition 7）可作为通用工具直接嵌入任何需要同时估计相关矩阵 + 与目标相关向量的 pipeline，避免 nearest-correlation 修复的数值风险。**
- **植入信号 + 日期置换安慰剂的双重诊断范式值得复用：** 前者标定方法在不同对齐强度的 recall/precision/risk 边界，后者在非参数假设下给出现场 null 的参考分布，二者合用能区分"真 null"与"方法失灵"。

## 关键术语表
- **Target alignment $\gamma_{it}$**：标准化预测与标准化 realized 目标的截面相关（$\langle s_{it},y_t\rangle_t$），度量单期"真预测力"。
- **Target-orthogonal component $u_{it}$**：预测减去对齐分量后的残差，严格正交于 $y_t$，不构成可复制的预测信息。
- **Forecast correlation $\rho^s_{ij}$**：预测之间的截面相关，由几何恒等式知与目标无关（target-free）。
- **Common-target deviation correlation $\rho^e_{ij}$**：$(s_{it}-y_t)$ 与 $(s_{jt}-y_t)$ 的截面相关，在弱对齐下近似为 $(1+\rho^s_{ij})/2$，因此系统性把冗余放大到 $[0,1]$ 区间。
- **Scale-free incremental risk $\Delta^{\mathrm{sf}}$**：等权池准入的"相关提升"部分（$\rho_P^2-\rho_{P\cup A}^2$），与合成尺度无关。
- **Scale-mismatch component $\Delta^{\mathrm{scale}}$**：准入带来的合成向量范数变化，弱对齐下即表现为"稀释"。
- **Three-way admission rule**：以同时 HAC 带决定 admit / reject / undecided 的贪心准入流程，$\alpha=0.05$。
- **No-information forecast**：$w=\mathbf{0}$ 的零预测，其风险恒为 1，是本框架下天然基准。

## 可复现要素
- **数据集：** 面版 A 使用 US 大型股权截面（公开价格源，但论文声明原始数据因条款不可再分发，仅提供从公开源重建的代码）；面板 B 使用 42 只跨资产 ETF 月度相对收益（同样需重建）。
- **代码/权重：** 完整复现包将在论文被接受后发布至带持久标识符的公开仓库；包含 notebook、提示模板、persona 指令、模型响应 SHA-256、随机种子、自检报告。
- **关键超参：** $\varepsilon_r=\varepsilon_x=10^{-8}$，$\varepsilon_{PSD}=10^{-9}$，$\delta\in\{0.001,0.0025,0.005,0.01,0.02\}$，$\delta^{\mathrm{sf}}\in\{0.0001,\ldots,0.002\}$，$\eta\in\{0,0.01,0.03,0.1,0.3,1,3\}$，peLASSO 惩罚网格为 $2\max_j\bar{\gamma}_j\times\{0.9,0.5,0.25,0.1,0.05,0.02\}$；max-$t$ 临界值 4000 次 bootstrap，MBCB 块长 4、999 次。
