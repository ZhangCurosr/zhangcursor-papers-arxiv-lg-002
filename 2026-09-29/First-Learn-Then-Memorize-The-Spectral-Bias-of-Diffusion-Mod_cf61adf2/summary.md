---
title: "First-Learn-Then-Memorize-The-Spectral-Bias-of-Diffusion-Mod"
source: https://arxiv.org/pdf/2609.35377v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-02 08:09:05"
field: "深度学习理论"
keywords: ["扩散模型", "谱偏差", "随机矩阵理论", "高斯等价原理", "偏差-方差分解", "early stopping对偶"]
innovations: ["Gram矩阵四部分谱分解定理", "泛化-记忆化双峰机制的定量刻画", "ridge与early stopping的谱对偶关系"]
benchmarks: ["高斯随机数据谱分析", "finite-width数值验证"]
---

# 论文速读：First-Learn-Then-Memorize-The-Spectral-Bias-of-Diffusion-Mod

## 一句话总结
本文通过随机矩阵理论与统计物理方法，严格推导了扩散模型在过参数化极限下Gram矩阵的渐近谱分布，揭示泛化与记忆化双峰结构，并给出测试损失的偏差-方差分解的闭合解。

## 研究问题与动机
- 扩散模型在过参数化 regime 下如何同时实现泛化与记忆化？现有学习理论缺乏严格的谱分析工具
- 传统核方法理论（如Random Feature模型）难以直接应用于扩散模型的时间演化结构
- 现有工作对扩散模型谱偏差的理解停留在数值实验层面，缺乏渐近闭合解
- 需要统一框架连接Replica方法与线性代数推导，验证两者在高维极限下的一致性

## 核心贡献（创新点）
1. **Gram矩阵渐近谱分布定理**：证明$\|\mathbf{G} - \mathbf{G}_{\text{lin}}\|_{\text{op}} = o(1)$，给出delta原子峰与连续bulk的四部分分解公式
2. **泛化-记忆化双峰机制**：首次定量刻画Generalization bulk ($\Theta(nm/d)$) 与Memorization bulk ($\Theta(m)$) 的分离条件与特征向量正交性
3. **Replica-L到Ridge对偶**：发现ridge正则化参数$\tilde{\gamma} = 1/\tau$可忠实代理early stopping，建立两类正则化方案的对偶关系
4. **高斯等价原理的扩散模型推广**：将Gerace/Goldt等的高斯等价方法扩展到带时间噪声的扩散过程，给出$\Omega$协方差的统一表达式
5. **Hessian的feature-space渐近形式**：在$m\to\infty$极限下给出损失函数Hessian的闭合解，连接kernel方法与神经塔吉极限

## 方法详解
### 谱分解框架
- **Replica方法**（D.5节）：基于统计物理的replica技巧计算随机矩阵Stieltjes变换期望，采用replica对称(RS)假设，结果与线性pencil方法一致
- **线性代数直接推导**（D.6.2节）：将问题投影到三维子空间$(\mathbf{U}^\lambda, \mathbf{X}^\lambda, \mathbf{P}^\lambda)$，通过块对角化得到精确特征值和特征向量
- **多项式分解**（D.7节）：在regime $d^{k+1}\gg n\gg d^k$下将kernel按多项式次数截断，高次部分集中为$\mu_I^{>k}\mathbf{I}_N + \mu_B^{>k}\mathbf{B}_m$

### 关键自洽方程组
Stieltjes变换$q(z)$和辅助参数$r(z)$满足：
$$z - \mu_I - m\mu_B + \frac{1}{mr} = \int d\rho_\Sigma(\lambda)\frac{\mu_1\Delta_t + m\mu_1 e^{-2t}\lambda}{1+\mu_1 e^{-2t}\psi_n m^2\lambda r + \mu_1\psi_n m\Delta_t q}$$
$$z - \mu_I + \frac{m-1}{m(q-r)} = \int d\rho_\Sigma(\lambda)\frac{\mu_1\Delta_t}{1+\mu_1 e^{-2t}\psi_n m^2\lambda r + \mu_1\psi_n m\Delta_t q}$$

### 谱分解公式
$$\rho(\omega) \approx \frac{m-1-1/\psi_n}{m}\delta(\omega-\mu_I) + \frac{1-1/\psi_n}{m}\delta(\omega-\bar{\mu}) + \frac{1}{\psi_n m}\rho_1(\omega) + \frac{1}{\psi_n m}\rho_2(\omega)$$
其中$\bar{\mu} = m\mu_B + \mu_I$，后两项分别为generalization和memorization bulk。

### 偏差-方差分解（D.8节）
- **Tweedie恒等式**：$\mathbb{E}[\pmb{\xi}|\mathbf{y}] = -\sqrt{\Delta_t}\,\mathbf{s}_{\text{exact}}(\mathbf{y})$
- **Bayes误差**：$C_t = \frac{1}{\Delta_t} - \frac{1}{d}\mathbb{E}_{P_t}\|\mathbf{s}_{\text{exact}}(\mathbf{y})\|^2$
- **各向同性情形**：$C_t = \frac{\sigma^2 e^{-2t}}{\Delta_t \Gamma_t}$，$\Gamma_t = \sigma^2 e^{-2t} + \Delta_t$

### 固定点方程（D.9节，Theorem D.4）
**Ridge情形**（$\tilde{\gamma} > 0$）：
- $1/\bar{r} = \tilde{\gamma} + R(\bar{r})$，其中$R(\hat{r}) = \frac{\mu_1}{m}\left(\frac{w_1}{L_1} + \frac{(m-1)w_2}{L_2}\right)$
- $b = \frac{\mu_1 \bar{r} g(\bar{r})}{m}$，$\mathcal{B}^2 = (b-1)^2$，$\mathcal{V} = \frac{\mu_1 T_4}{\Delta_t} - b^2$

**Ridgeless情形**（$\tilde{\gamma} \to 0^+$）：固定点方程变为$1/r = \varrho + R(r)$，结果不依赖规范$\varrho$。

## 实验与结果
- **数据集**：$\mathbf{x}^\nu \sim \mathcal{N}(0,\mathbf{I}_d)$，每点$m$个噪声副本，核为$\tanh$的对偶激活函数$f(u) = \mathbb{E}[\sigma(z_1)\sigma(z_2)]$
- **参数范围**：$\psi_n = 8$，$d \in \{64, 128, 256, 512\}$，验证有限维与渐近理论的一致性
- **无偏估计器**：通过$\frac{1}{n_D-1}$和减$\hat{\mathcal{V}}/n_D$修正，使有限样本下偏差与方差不被膨胀
- **主要结果**：数值验证显示渐近谱分解公式与各向同性情形下的偏差-方差闭合解与蒙特卡洛模拟误差<2%
- **最强结论**：ridge参数$\tilde{\gamma} = 1/\tau$的早停对偶在$\psi_n \geq 4$时即高度准确

## 相关工作脉络
1. **Random Feature模型**（Cabannes et al., Bach）：本文扩展至扩散模型时间演化结构，给出$\mu_I, \mu_B$的显式时间依赖形式
2. **核方法谱理论**（Mei-Montanari, Barron）：将kernel ridge回归的谱分析推广到 diffusion kernel 的非平稳结构
3. **Replica方法**（Dommers-Opper, Dembo）：首次将replica技巧系统应用于扩散模型的Gram矩阵谱分析
4. **高斯等价原理**（Gerace, Goldt, Hu-Lu）：引用并推广至带时间噪声的$\Omega$协方差结构
5. **神经塔吉极限**（Jacot et al., Bonnaire et al.）：在$m\to\infty$极限下连接Hessian的feature-space与kernel表示
6. **Early stopping理论**（Capra, Rosanoy）：建立ridge正则化与早停谱截断的对偶关系$\tilde{\gamma} = 1/\tau$

## 局限性与未来方向
- 当前分析假设各向同性数据（$\Sigma = \sigma^2\mathbf{I}_d$），一般协方差结构需进一步推广
- Replica对称假设在临界点附近可能失效，需引入replica symmetry breaking分析
- 仅处理二次损失，extension至cross-entropy或contrastive loss尚待研究
- 有限$m$修正项的阶数估计未完全给出，高阶矩展开留待后续工作

## 研究启发与可借鉴点
- **高斯等价替换技巧**：将随机特征$\Omega$替换为等价高斯变量，简化Gram矩阵期望计算，可迁移至其他随机神经网络分析
- **Replica-线性对偶验证**：用两种独立方法（统计物理vs.线性代数）交叉验证谱结果，提高定理可信度
- **固定点方程求解策略**：将高维积分方程约化为一维固定点问题，适用于大规模数值求解
- **Tweedie恒等式在扩散模型的应用**：连接score matching与MMSE估计，为扩散模型的可解释性分析提供新视角
- **Early stopping的谱解释**：将早停参数化为ridge正则化，便于统一理论分析与实践调参

## 关键术语表
**Gram矩阵谱分布**：高维数据内积矩阵的特征值渐近密度，反映数据流形的内蕴几何
**Replica方法**：统计物理中计算随机矩阵期望的工具，通过$n\to 0$极限提取对数配分函数
**Stieltjes变换**：复平面上$\int \frac{\rho(\omega)}{z-\omega}d\omega$，用于编码谱分布的全部矩信息
**Generalization bulk**：特征值$\Theta(nm/d)$量级的连续谱，对应泛化子空间
**Memorization bulk**：特征值$\Theta(m)$量级的连续谱，对应记忆子空间
**Delta原子峰**：谱分布中的Dirac质量点，对应$\mu_I$和$\bar{\mu}$处的本征值聚集
**Tweedie恒等式**：高斯噪声下MMSE估计与score函数的关系$\mathbb{E}[\xi|y] = -\sqrt{\Delta_t}s(y)$
**高斯等价原理**：随机向量的多项式函数期望可由高斯近似，误差$o(1/d)$
**Ridge-早停对偶**：谱截断阈值$\tau$与ridge参数$\tilde{\gamma}$满足$\tilde{\gamma}=1/\tau$的等价关系

## 可复现要素
- 数据集：标准高斯数据$\mathcal{N}(0,\mathbf{I}_d)$，未提及真实数据集
- 代码/权重：论文未提及开源，但提供完整数学推导
- 关键超参：$\psi_n = n/d \in \{4, 8\}$，$d \in \{64, 128, 256, 512\}$，$m$为噪声副本数，$\tilde{\gamma}$为ridge参数
- 核函数：$\tanh$对偶激活$f(u) = \mathbb{E}[\sigma(z_1)\sigma(z_2)]$
