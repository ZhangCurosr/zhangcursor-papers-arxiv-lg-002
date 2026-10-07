---
title: "Performative-Prediction-with-Selective-Labels"
source: https://arxiv.org/pdf/2610.08272v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:13:06"
field: "performative machine learning"
keywords: ["performative prediction", "selective labels", "repeated risk minimization", "robust optimization", "fair lending", "partial identification"]
innovations: ["证明在阈值化选择性标签下朴素RRM可偏离稳定解diam(Θ)量级", "构造worst-case loss并提出Robust RRM，在强凸框架下有界逼近真正稳定解", "利用历史接受样本的Lipschitz外推动态收紧标签概率区间，使误差界随迭代单调收缩"]
benchmarks: ["Fair Lending with FICO dataset"]
---

# 论文速读：Performative Prediction with Selective Labels

## 一句话总结
本文研究了**performative prediction**（模型部署引发人群分布迁移）与**selective labels**（仅能观测被接受样本的标签）两个现实问题的交叉场景，指出朴素地仅在选中子集上执行RRM会偏离稳定解，并提出一种基于最坏情况目标函数的**Robust RRM**方法，在有界距离内逼近真正的performatively stable solution。

## 研究问题与动机
1. **Performative预测中普遍存在的selective labels盲区**：Perdomo等人提出的RRM框架默认可获取完整分布 $D(\theta)$ 上的标签，但在信贷、医疗、司法等决策场景中，标签仅在"接受"子集上可观测，形成选择性观测偏差。
2. **朴素重训练在阈值化选择下失效**：Proposition 3.1证明，在确定性阈值观测下，仅在 $\tilde{D}(\theta)$（被接受样本的条件分布）上训练可能导致一步更新的参数距离达到 $\mathrm{diam}(\Theta)$，即**完全偏离**稳定解 $\theta^{PS}$。
3. **现有基线无法处理阈值化决策**：DR Estimator与DCEM等方法依赖"每个特征都有正概率被观测"的假设，与阈值化接受规则冲突，导致发散。
4. **缺乏可量化误差界的鲁棒学习器**：如何在不依赖Oracle（全标签）的前提下，构建可在有界距离内逼近稳定解的迭代学习器，尚属空白。

## 核心贡献（创新点）
1. **首个将selective labels正式引入performative prediction的框架**：区分"全分布RRM"与"选择分布RRM"并证明前者在阈值设定下可产生 $\mathrm{diam}(\Theta)$ 量级的偏差。
2. **Worst-Case Loss + Robust RRM**：基于部分识别技术构造强凸上界损失 $\overline{\ell}_\phi$，对未观测样本用其上界、对已观测样本保留真实loss，形成新的迭代算子 $\overline{G}$。
3. **有界逼近定理（Theo. 4.7）**：证明Robust RRM的迭代点列 $\theta_t$ 与真实稳定解 $\theta^{PS}$ 的距离以 $\frac{\kappa}{1-\eta}$ 为界，其中 $\kappa$ 由损失强凸性、梯度敏感度和标签区间宽度共同控制。
4. **历史观测驱动的自适应收紧机制（Sec. 4.1）**：在标签分布 $(L_x, L_\theta)$-敏感假设下，利用过往已接受样本推算未观测样本的 $[l(x), u(x)]$，且该区间宽度随迭代单调收缩。
5. **在公平信贷模拟环境下的实证验证**：扩展 Liu et al. [8] 长期公平信贷场景，证明 Robust RRM 在 performative risk、demographic parity 和到 $\theta^{PS}$ 距离三项指标上均接近 Oracle RRM，显著优于 Accept-only 和 DR Estimator。

## 方法详解

### 4.1 问题设定
- 决策者部署模型 $\theta_t$，人群按 $D(\theta_t)$ 响应；观测到 $X$，按确定性阈值规则 $O = \mathbf{1}\{f_{\theta_t}(X) \geq \tau\}$ 决定标签是否可见。
- 目标是在 $O=0$ 的样本上避免偏倚。

### 4.2 Worst-Case Loss 构造
**Assumption 4.1**：已知标签概率 $\alpha_\theta(x) = P(Y=1|X=x,\theta)$ 满足 $l(x,\theta) \leq \alpha_\theta(x) \leq u(x,\theta)$。

定义 $\Delta_\ell(x,\theta) = \ell(x,1,\theta) - \ell(x,0,\theta)$，构造 worst-case loss：

$$
\overline{\ell}_\phi(x,\theta) := \ell(x,0,\theta) + \log\!\left(e^{\,l(x,\phi)\Delta_\ell(x,\theta)} + e^{\,u(x,\phi)\Delta_\ell(x,\theta)}\right)
$$

**Claim 4.3**：若 Assumption 4.1 成立，则 $\mathbb{E}_{D(\phi)}[\ell(X,Y,\theta)] \leq \mathbb{E}_{D(\phi)}[\overline{\ell}_\phi(X,\theta)]$；且若 $\ell$ 是 $\gamma$-强凸的，$\overline{\ell}_\phi$ 也是 $\gamma$-强凸（但不保证 $\beta$-光滑）。

**Definition 4.4**：混合损失（已观测用真实loss，未观测用上界loss）：

$$
\overline{J}(\phi,\theta) := \mathbb{E}_{D(\phi)}\!\left[O\,\ell(X,Y,\theta) + (1-O)\,\overline{\ell}_\phi(X,\theta)\right]
$$

**Definition 4.5（Robust RRM）**：$\theta_{t+1} = \overline{G}(\theta_t) := \arg\min_{\theta \in \Theta} \overline{J}(\theta_t, \theta)$。

### 4.3 理论界
**Theorem 4.6**（单步差距）：

$$
\|G(\phi) - \overline{G}(\phi)\|_2 \leq \frac{1}{\gamma}\sup_{\theta} \mathbb{E}_{D(\phi)}\!\left[(1-O)\,|u(X,\phi)-l(X,\phi)|\,\|\nabla_\theta \Delta_\ell(X,\theta)\|_2\right]
$$

对线性得分+二元交叉熵，简化为 $\frac{R}{\gamma}\mathbb{E}[(1-O)\,|u-l|]$。

**Theorem 4.7**（近似稳定性）：若真实RRM是 $\eta=\epsilon\beta/\gamma<1$ 的压缩映射，稳定解为 $\theta^{PS}$，记 $\kappa = \sup_\theta\|\overline{G}(\theta)-G(\theta)\|_2$，则：

$$
\|\theta_t - \theta^{PS}\|_2 \leq \eta^t\|\theta_0 - \theta^{PS}\|_2 + \frac{1-\eta^t}{1-\eta}\,\kappa
\quad \Longrightarrow \quad
\limsup_{t\to\infty}\|\theta_t - \theta^{PS}\|_2 \leq \frac{\kappa}{1-\eta}
$$

### 4.4 基于历史数据的区间收紧（Sec. 4.1）
设 $\alpha(x)$ 为 $(L_x, 0)$-敏感（不依赖 $\theta$），令 $\mathcal{X}_{1:t}^{\mathrm{obs}}$ 为之前被接受样本的特征支撑集，对未观测样本：

$$
\begin{aligned}
l(x_{\mathrm{uno}}) &= \max\!\left\{0,\; \sup_{x_{\mathrm{obs}} \in \mathcal{X}_{1:t}^{\mathrm{obs}}}\!\alpha(x_{\mathrm{obs}}) - L_x\|x_{\mathrm{uno}} - x_{\mathrm{obs}}\|_2\right\} \\
u(x_{\mathrm{uno}}) &= \min\!\left\{1,\; \inf_{x_{\mathrm{obs}} \in \mathcal{X}_{1:t}^{\mathrm{obs}}}\!\alpha(x_{\mathrm{obs}}) + L_x\|x_{\mathrm{uno}} - x_{\mathrm{obs}}\|_2\right\}
\end{aligned}
$$

**Theorem 4.9** 给出收紧后的单步界：

$$
\|G(\phi) - \overline{G}(\phi)\|_2 \leq \frac{2L_x R}{\gamma}\mathbb{E}_{D(\phi)}\!\left[(1-O)\inf_{x_{\mathrm{obs}}\in\mathcal{X}_{1:t}^{\mathrm{obs}}}\|X-x_{\mathrm{obs}}\|_2\right]
$$

即**区间宽度随历史接受覆盖范围扩大而单调缩小**。

## 实验与结果

### 实验设置
- **场景**：公平信贷环境（FICO数据集衍生），特征 $x=(c,z,1)$（信用分、种族敏感属性、截距）；部署后信用分按 $c_\theta = c + \epsilon(\alpha(x)-0.5)\cdot\sigma(f_\theta(x))$ 更新，加入 demographic parity 正则项 $\lambda=0.3$。
- **基线**：Oracle RRM（全标签）、Accept-only（仅接受样本）、Affine RM（[23]）、DCEM（[28]）、DR Estimator。
- **评估指标**：performative risk $J(\theta_t,\theta_t)$、demographic parity $DP(\theta_t,\theta_t)$、到 $\theta^{PS}$ 的距离。
- **迭代**：400轮，每轮 $n=100{,}000$ 样本，5000步GD或损失变化 $<10^{-7}$ 时停止。

### 主要结果
1. **Performative gap 收敛**（Fig. 2a）：$\epsilon \in \{50, 90, 150\}$ 下，所有设置的performative gap均降至 $<10^{-4}$。
2. **区间收紧效果**（Fig. 2b）：
   - 使用 trivial limits $[0,1]$：到 $\theta^{PS}$ 距离维持在约 1.5。
   - 使用自适应收紧后：平均区间宽度从 1.0 降至约 0.08，400轮后距离降至 **0.212**。
3. **对比基线**（Fig. 3，$\epsilon=50$）：
   - Robust RRM 的 performative risk ≈ Oracle RRM（≈0.336）。
   - Robust RRM 到 $\theta^{PS}$ 距离最小（≈0.212），**显著优于** Accept-only、DR Estimator（DR Estimator 发散）。
   - Affine RM 在 $\epsilon=90,150$ 下仍接近 Oracle risk，但距离略大于 Robust RRM。
4. **$L_x$ 误设鲁棒性**（Tab. 1）：
   - 正确 $L_x=0.15$：risk=0.336，距离=0.212。
   - $+50\%$ 误设（$L_x=0.225$）：距离升至 **0.282**；$-50\%$ 误设（$L_x=0.075$）：距离反而降至 **0.159**（低估 $L_x$ 带来 tighter interval，虽理论上不再保证全局覆盖，实证仍有效）。
5. **含 outcome performativity 的扩展实验**（Appendix H）：当 $\alpha_\theta(x)=\alpha(x)+\theta^\top\rho$ 时，Robust RRM 同样逼近 Oracle 性能。

## 相关工作脉络
1. **Perdomo et al. [1]**：建立 performative prediction 的基础理论，证明 RRM 在损失强凸+分布敏感条件下的收敛性——本文在其基础上**引入选择性标签**，打破原有假设。
2. **Kilbertus et al. [24] / Chang & Wiens [28]**：静态分布下研究 selective labels，本文首次将其嵌入**动态 performative 场景**并证明阈值化会破坏收敛。
3. **Liu et al. [8]（Fair Lending）**：引入长期公平的信贷仿真环境，本文扩展该环境以兼容 performative 动态与 selective label。
4. **Valdrighi et al. [32]**：先前研究动态环境下的公平与选择性标签，但未处理阈值化决策；本文填补这一缺口。
5. **Izzo et al. [23]（Affine RM）**：通过聚合历史迭代提高收敛速率，本文在 selective label 下表明其风险性能不如 Robust RRM。
6. **Creager et al. [31]**：因果建模处理动态公平性，但其 double-robust 估计在阈值选择下失效；本文方法避免了对 propensity score 正定假设的需求。

## 局限性与未来方向
1. **理论结果在人口水平成立**：尚未推导有限样本误差界，需另行分析风险估计与标签分布估计的采样误差。
2. **$L_x, L_\theta$ 的不可识别性**：在确定性阈值策略下，这两个常数无法仅从已接受样本中识别；论文假设有先验知识或通过额外探索样本估计。
3. **仅考虑确定性阈值策略**：随机策略可增加标签覆盖率并改善效用与公平性，将框架推广至随机策略是自然方向。
4. **未讨论 performative optimality**：稳定解未必是最优解，为收集数据而部署次优模型的社会成本值得研究。
5. **多类别扩展的挑战**：Appendix D 指出多分类需同时处理多个 $\alpha_\theta^k(x)$ 和联合约束，且可能破坏强凸性。

## 研究启发与可借鉴点
1. **Worst-Case Loss 的构造思路可迁移**：利用线性组合 $c\cdot\Delta_\ell + \ell(x,0,\theta)$ 的上界形式 + log-sum-exp 光滑化，既保证强凸性又避免真实标签依赖，该方法可复用于其他存在选择性观测的监督学习问题。
2. **"历史接受支撑集 → 区间收紧"范式**：利用之前轮次接受的 $(x_{\mathrm{obs}},\alpha(x_{\mathrm{obs}}))$ 对未知样本做 Lipschitz 外推，是一种无需额外探索即可系统性缩窄不确定区间的方法，可与 bandit / active learning 结合。
3. **稳健性分析中的"有界逼近"而非"收敛"策略**：当严格收敛条件被破坏（如 $\overline{\ell}$ 不光滑）时，转而证明迭代点列始终留在稳定解的邻域内，是一条可行的理论路径。
4. **公平信贷仿真的可复用设计**：FICO 衍生环境、$c_\theta$ 更新规则与 DP 正则项的组合可作为后续 performative fairness 工作的基准测试。
5. **低估敏感性常数反而提升性能的意外发现**（Tab. 1）：提示在实际部署中可保守地**低估** $L_x$ 以换取更紧的区间与更快的收敛，即使牺牲理论全覆盖假设。

## 关键术语表
**Performative Prediction（表演性预测）**：研究机器学习模型部署后，因人群行为响应导致数据分布随模型变化的框架（Perdomo et al., 2020）。

**Selective Labels（选择性标签）**：标签仅在决策者"接受"的子集上可观测，拒绝样本的标签不可得的选择性观测设定。

**Repeated Risk Minimization（RRM）**：迭代重训练过程，第 $t$ 轮用当前分布 $D(\theta_t)$ 训练得到 $\theta_{t+1}$，直至收敛到稳定解。

**Performative Stable Solution（$\theta^{PS}$）**：满足 $\theta^{PS} = \arg\min_\theta J(\theta^{PS}, \theta)$ 的参数，即模型在其自身诱导分布上达到风险最小化的不动点。

**Worst-Case Loss（最坏情况损失）**：在已知标签概率上下界 $[l,u]$ 时，对缺失标签构造的一个强凸上界损失 $\overline{\ell}_\phi$。

**$(L_x, L_\theta)$-Sensitive Label Distribution**：标签条件概率 $\alpha_\theta(x)$ 关于特征 $x$ 和参数 $\theta$ 的联合 Lipschitz 敏感性条件。

**Performative Gap**：部署新模型 $\theta_{t+1}$ 后，其在诱导分布下的风险相对于上一轮训练分布的风险增量，衡量分布漂移程度。

**Demographic Parity（人口统计均势）**：公平性约束，要求不同敏感属性群体的接受率相等，即 $P(\text{accept}|Z=1) = P(\text{accept}|Z=0)$。

## 可复现要素
- **数据集**：FICO 信用评分数据集（来自 [42]），论文提供了基于该数据集的仿真数据生成流程（Appendix F.2）。
- **代码**：论文声明实验代码开源于 `github.com/recod-ai/perf_selective`。
- **关键超参**：
  - 每轮样本量 $n=100{,}000$，总迭代 400 轮。
  - 每轮内部 GD 最多 5,000 步，早停阈 $10^{-7}$。
  - $L_2$ 正则权重 $10^{-2}$ 保证强凸。
  - 公平正则权重 $\lambda=0.3$。
  - 阈值 $\tau=0$。
  - 标签分布敏感常数 $L_x=0.15$（贷款场景）、$L_\theta=0$（主实验）。
  - 历史接受样本缓冲区最大 500,000，KD-Tree 近邻数 50。
- **实现细节**：CPU 集群，无 GPU 使用（AMD Ryzen Threadripper PRO 7975WX / Intel Xeon Silver 4410Y）。
