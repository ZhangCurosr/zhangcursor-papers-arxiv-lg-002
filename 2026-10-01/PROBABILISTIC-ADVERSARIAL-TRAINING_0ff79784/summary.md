---
title: "PROBABILISTIC-ADVERSARIAL-TRAINING"
source: https://arxiv.org/pdf/2609.39798v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-04 00:16:56"
field: "对抗鲁棒性"
keywords: ["adversarial robustness", "probabilistic robustness", "adversarial training", "KL divergence", "importance sampling", "Langevin dynamics", "robust deep learning"]
innovations: ["证明KL散度下界与概率鲁棒性的理论关系并提出可直接优化的训练目标", "通过SNIS推导可计算的梯度估计器并证明其渐近一致性", "理论导出的缩放因子可作为通用正则器增强非概率对抗训练方法"]
benchmarks: ["CIFAR-10", "CIFAR-100"]
---

# 论文速读：PROBABILISTIC ADVERSARIAL TRAINING

## 一句话总结
本文从概率视角统一了对抗攻击与模型防御，理论证明了 KL 散度下界与概率鲁棒性（PR）的关系，并据此提出了 **Probabilistic Adversarial Training (PAT)**——一种通过自加权重要性采样（SNIS）最大化该下界的训练方法，在 CIFAR-10/100 上持续提升 PR。

## 研究问题与动机
- **核心问题**：如何系统性地提升模型的**概率鲁棒性（PR）**——即模型在局部扰动分布下整体抵抗对抗攻击的概率，而非仅优化最坏情况（worst-case）。
- **现有方法不足**：传统对抗训练（如 PGD-AT）优化确定性最坏扰动，易过拟合到异常边界，牺牲整体分布安全性；概率鲁棒性评估框架（Webb et al., 2018）定义了 PR 指标，但缺乏直接优化它的训练方法。
- **概率攻击视角局限**：Zhang et al. (2024a) 将对抗样本建模为距离分布与受害者分布的乘积（Langevin Dynamics 收敛至 Gibbs 分布），但该框架仅适用于**定向攻击**，未探索防御方向。
- **直觉驱动**：对抗样本源于 $p_{\text{dis}}$（约束扰动幅度）与 $p_{\text{vic}}$（驱动误分类）的重叠区域；增大两者 KL 散度即"推开"这两个分布，缩小重叠，从而提升鲁棒性。

## 核心贡献（创新点）
1. **非定向概率对抗攻击的形式化**：将 Zhang et al. (2024a) 的定向攻击框架扩展为非定向场景，证明非定向受害者分布 $p_{\text{vic}}$ 在紧集上良定义（Proposition 1）。*与已有工作本质区别：原文框架仅限于定向攻击，本文首次将其推广至更通用的非定向场景。*
2. **PR 的 KL 散度下界定理**：证明 $\mathrm{KL}(p_{\text{dis}} \| p_{\text{vic}}) - \log Z_{\text{vic}}$ 是概率鲁棒性的下界（Theorem 2），将直觉"推开两个分布"转化为严格的数学目标。*与已有工作本质区别：首次建立 PR 的理论可优化下界，而此前 PR 仅用于评估（Webb et al., 2018; Tit et al., 2021）。*
3. **基于 SNIS 的可计算梯度估计器**：通过三步推导（理想目标 → 梯度先算 → 自归一化重要性采样）将不可计算的梯度转化为可直接反向传播的 empirics loss $\mathcal{L}_{\text{emp}}$（Proposition 3/4，Algorithm 2）。*与已有工作本质区别：不同于 AT-PR 的启发式边界搜索，PAT 通过 SNIS 提供了完全可微、有理论保证的优化路径。*
4. **缩放因子作为通用正则器的经验发现**：理论导出的缩放因子 $\exp(-\beta f(\cdot))$ 可作为即插即用模块增强传统非概率防御（如 PGD-COR 在 CIFAR-100 上将 PR(0.2) 从 58.99% 提升至 62.21%）。*与已有工作本质区别：本文的权重机制不仅服务自身方法，还能提升其他基线，具有跨方法适用性。*
5. **批量级估计器的渐近一致性证明**：Theorem 7 严格证明了 Algorithm 2 在批量级（每图单扰动）设置下梯度估计的几乎必然收敛性，并建立了与 entropic-risk 优化的联系（Corollary 9）。*与已有工作本质区别：为实际实现提供了理论基础，填补了从理论下界到工程可实现的严谨分析空白。*

## 方法详解
**整体框架**：以 Langevin Dynamics 采样对抗样本 → 基于 KL 下界的加权损失 → SGD 更新参数。

1. **概率对抗攻击（非定向）**：将优化目标 $\min_{x_{\text{adv}}} c_1 \mathcal{D}(x_{\text{ori}}, x_{\text{adv}}) - c_2 f(x_{\text{adv}}, y_{\text{ori}})$ 通过 Langevin Dynamics 优化，收敛至 Gibbs 分布 $p_{\text{adv}} \propto p_{\text{dis}} \cdot p_{\text{vic}}$，其中 $p_{\text{dis}} \propto \exp(-c_1 \|x - x_{\text{ori}}\|_2^2)$（高斯），$p_{\text{vic}} \propto \exp(c_2 f(x, y_{\text{ori}}; \theta))$（受害者分布）。

2. **KL 下界定理（Theorem 2）**：令 $\gamma_* = \inf_{x \in E_{\text{adv}}} c_2 f(x, y_{\text{ori}}; \theta)$，则有：
   $$\mathrm{PR}(p_{\text{dis}}, s) \geq 1 - \frac{\log Z_{\text{vic}} - H(p_{\text{dis}}) - \mathrm{KL}(p_{\text{dis}} \| p_{\text{vic}})}{\gamma_*}$$
   最大化下界等价于最大化 $\mathrm{KL}(p_{\text{dis}} \| p_{\text{vic}}) - \log Z_{\text{vic}}$。

3. **可计算梯度推导（三步法）**：
   - **Step 1**（理想目标）：通过 importance sampling 将梯度期望改写为 $p_{\text{adv}}$ 上的期望，但不含 $\log Z_{\text{vic}}$，仍含不可微参数化分布。
   - **Step 2**（梯度先行）：先对下界求梯度再采样，使 $\nabla_\theta$ 只作用于 $f$，权重 $w(X) = Z_{\text{adv}} Z_{\text{vic}} \exp(-c_2 f(X))$ 成为常数缩放因子。
   - **Step 3**（SNIS）：利用 batch 内自归一化消去不可计算常数：
     $$\hat{w}^{(i)} = \frac{\exp(-c_2 f(X^{(i)}, y^{(i)}; \theta))}{\sum_j \exp(-c_2 f(X^{(j)}, y^{(j)}; \theta))}$$
     最终 empirical loss：$\mathcal{L}_{\text{emp}} = \sum_i \hat{w}^{(i)} f(X^{(i)}, y^{(i)}; \theta)$，权重在 backprop 中 stop-gradient。

4. **批量级平滑（Section 5.4）**：实际实现中每图像仅采样一个扰动，为避免 hard 样本权重坍塌（weight collapse），用平滑因子 $\beta = 0.001$ 替代 $c_2$：$\tilde{w}^{(i)} = \exp(-\beta f(X^{(i)}, y^{(i)}; \theta))$。这同时起到隐式正则化作用——抑制极端异常样本。

5. **Algorithm 2 流程**：
   - Phase 1：对 mini-batch 每个样本执行 $T=100$ 步 Langevin Dynamics 生成对抗样本。
   - Phase 2：计算各样本 loss → detach 后 softmax 得到权重 → 加权求和 loss → 反向传播更新 $\theta$。

## 实验与结果
- **数据集**：CIFAR-10、CIFAR-100。
- **模型架构**：ResNet-18、WRN-50-2。
- **评估基线**：Clean、FGSM、PGD、ALP、CLP、TRADES、MART、AT-PR（Zhang et al., 2025）。
- **PR 评估协议**：在 $\epsilon \in \{0.1, 0.12, 0.15, 0.2\}$ 四种扰动尺度下，每图采样 $N=100$ 次扰动估计 PR。
- **关键结果（CIFAR-100, WRN-50-2）**：
  - PAT PR(0.2) = **67.12%**，对比：AT-PR 61.58%、MART 66.51%、PGD-AT 58.99%，**相对最强基线 AT-PR 提升 +5.54 个百分点**。
  - PAT PR(0.15) = **82.18%**，对比 AT-PR 79.87%、MART 81.41%。
  - 在 CIFAR-10, ResNet-18 上：PAT PR(0.2) = **80.62%**，对比 AT-PR 78.22%（+2.40pp）、MART 78.29%（+2.33pp）。
- **PGD/CW 准确率**：PAT 的最坏情况鲁棒性（PGD Acc.）略低于未加权 AT，这是**有意为之的设计**——通过 down-weight 极端样本换取整体分布安全。
- **消融实验**：PAT(WOS)（去除重要性权重）在 CIFAR-100, WRN-50-2 上 PR(0.2) 从 67.12% 降至 63.98%（-3.14pp），证实权重机制不可或缺。
- **即插即用验证**：将缩放因子应用于 PGD（PGD-COR）在 CIFAR-100, WRN-50-2 上 PR(0.2) 从 58.99% 提升至 62.21%（+3.22pp）；FGSM-COR 提升不规则（因 FGSM 单步缺乏迭代探索）。

## 相关工作脉络
- **Madry et al. (2018) PGD-AT**：经典 min-max 对抗训练，优化最坏情况扰动。PAT 不追求最优 worst-case，而是优化整体分布鲁棒性，两者目标函数不同。
- **Zhang et al. (2019) TRADES / Wang et al. (2019) MART**：正则化型 AT 方法，通过 loss 惩罚项平衡准确率与鲁棒性。PAT 从概率分布视角重新解释 AT，提供统一理论框架。
- **Kannan et al. (2018) ALP/CLP**：logit pairing 方法。PAT 明确优化 PR 指标，而 ALP/CLP 优化的是 logits 层面的正则化目标。
- **Webb et al. (2018)**：首次定义 PR 评估框架。本文在 PR 评估基础上，首次提出可直接优化 PR 的训练方法（此前 PR 仅用于评估）。
- **Zhang et al. (2025) AT-PR**：最直接相关方法，同样优化 PR，但依赖启发式梯度边界搜索找"最宽峰值"。PAT 提供严格的 KL 下界理论保证，无需启发式搜索。
- **Zhang et al. (2024a)**：首次提出概率对抗攻击视角（定向攻击）。本文将其推广至非定向攻击，并进一步发展为防御方法，实现"攻击→防御"的范式转换。

## 局限性与未来方向
- **计算开销**：Langevin Dynamics 需要 $T=100$ 步迭代（附录 F 明确提及），显著高于标准 PGD（通常 10 步），训练成本更高。
- **批量级估计近似**：实际实现每图仅采样一个扰动（而非多个扰动），引入了理论上的偏差（Appendix C 分析了这一近似）。
- **平滑因子 $\beta$ 的经验性**：$\beta = 0.001$ 为超参数选择，缺乏理论最优值的严格推导。
- **非定向攻击形式化**：虽然扩展了非定向场景，但核心理论仍基于 cross-entropy loss，对多分类场景的充分性有待进一步验证。
- **未涉及**：函数扰动（functional perturbations，Zhang et al., 2022）或更广泛的概率鲁棒性评估场景。

## 研究启发与可借鉴点
1. **KL 下界的理论工具可迁移**：将 KL 散度作为鲁棒性优化目标的思路，可扩展到其他分布安全度量（如 CVaR、EVaR）的场景，为风险敏感型对抗训练提供理论工具。
2. **SNIS + stop-gradient 的工程技巧**：先对目标求梯度再采样、batch 内自归一化消去不可计算常数——这一技巧可用于其他涉及不可归一化分布的概率优化问题（如变分推理、能量模型训练）。
3. **即插即用正则器的设计范式**：将理论导出的权重机制独立封装为可附加模块，验证其对 PGD/FGSM 等既有方法的增益，这种"理论→通用工具"的研究路径值得效仿。
4. **从攻击到防御的视角转换**：同一概率框架下，先形式化攻击（Langevin 采样），再反过来指导防御（KL 下界），形成对称且自洽的理论体系，这种双向推导极具启发性。
5. **隐式正则化与 overfitting 抑制**：$\beta$ 平滑因子同时防止 weight collapse 和抑制异常样本，揭示了"温度缩放"在对抗训练中的双重作用，可在其他鲁棒学习场景中探索类似机制。

## 关键术语表
- **概率鲁棒性（Probabilistic Robustness, PR）**：分类器在给定扰动分布下成功抵御攻击的概率，定义为 $1 - \mathbb{P}(s(X) \geq 0)$，其中 $s(\cdot)$ 为 logit margin 违反函数。
- **距离分布（Distance Distribution, $p_{\text{dis}}$）**：约束扰动幅度的分布，当 $\mathcal{D}$ 为 $L_2$ 范数时为高斯分布。
- **受害者分布（Victim Distribution, $p_{\text{vic}}$）**：由分类器 loss 诱导的分布，驱动模型走向误分类，$p_{\text{vic}} \propto \exp(c_2 f(x, y_{\text{ori}}; \theta))$。
- **自归一化重要性采样（SNIS）**：通过 batch 内加权样本自归一化来消除不可计算归一化常数的梯度估计方法。
- **Langevin Dynamics**：一种用于从 Gibbs 分布采样的随机优化算法，此处用于采样对抗分布 $p_{\text{adv}}$。
- **概率对抗攻击（Probabilistic Adversarial Attack）**：将对抗样本视为从 $p_{\text{adv}} = p_{\text{dis}} \cdot p_{\text{vic}}$ 采样的样本，而非确定性优化解。
- **Entropic Risk**：$R_\lambda(x,y;\theta) = \frac{1}{\lambda}\log \mathbb{E}_{U \sim p_{\text{dis}}}[\exp(\lambda f(U,y;\theta))]$，PAT 在批量极限下优化的等价目标。
- **平滑因子（Smoothing Factor, $\beta$）**：用于替代 $c_2$ 防止 weight collapse 的小常数（论文取 0.001），同时提供隐式正则化。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100（公开数据集）。
- **代码/权重**：论文未提及开源代码或预训练权重（GitHub/ArXiv 页面无链接声明）。
- **关键超参**：$T=100$ 步 Langevin、$\eta=0.3$、$\sigma=0.001$、$\rho=1.0$、$c_1=0.3$、$c_2=0.42$、$\beta=0.001$；训练 100 epochs、batch size=256、SGD+momentum 0.9、lr=0.01（MultiStepLR，75/90 epoch 各×0.1）、weight decay=5e-4；评估时 PGD-20/CW-20，$\epsilon=8/255$、$\alpha=2/255$；PR 评估每图 100 次采样，$\epsilon \in \{0.1, 0.12, 0.15, 0.2\}$。
