---
title: "First-Learn-Then-Memorize-The-Spectral-Bias-of-Diffusion-Mod"
source: https://arxiv.org/pdf/2609.35377v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-02 08:08:47"
field: "生成模型理论"
keywords: ["扩散模型", "谱偏差", "泛化-记忆", "NTK", "噪声重复", "训练动力学"]
innovations: ["揭示扩散模型泛化先于记忆的谱机制", "证明噪声重复重塑 Gram 矩阵谱产生两 bulk", "建立最优早停与谱截断的精确映射"]
benchmarks: ["CelebA 灰度 32x32", "FID", "bias-variance 分解"]
---

# 论文速读：First Learn, Then Memorize: The Spectral Bias of Diffusion Models

## 一句话总结
本论文揭示了扩散模型在有限数据集上训练的"泛化先于记忆"现象，证明其动力学由 NTK Gram 矩阵的谱结构精确控制，噪声重复（每张样本 $m>1$ 个噪声实现）重塑了谱结构，产生两个分离的 bulk，分别编码泛化能力与记忆行为。

## 研究问题与动机
- **现象之谜**：实践中观察到扩散模型先学会生成高质量新样本（泛化），很久之后才回退到记忆训练集，但缺乏理论解释
- **已有方法的不足**：
  - 监督回归理论（$m=1$）无法解释扩散模型的噪声重复机制
  - 标准早停/正则化理论未考虑 NTK 谱结构的演化
  - 对"泛化-记忆" Timescale 分离的数学刻画缺失
- **核心动机**：建立扩散模型训练动力学的精确理论框架，揭示泛化与记忆的谱编码机制

## 核心贡献（创新点）
1. **首次揭示"泛化先于记忆"的谱机制**：证明训练动力学由 NTK Gram 矩阵谱精确控制，泛化与记忆的 Timescale 分离编码于谱的两个分离 bulk 中
2. **噪声重复重塑谱结构**：发现每张样本 $m>1$ 个噪声实现（区别于 $m=1$ 的监督回归）产生两个 bulk：泛化 bulk $\Theta(nm/d)$ 与记忆 bulk $\Theta(m)$
3. **任意宽度成立**：理论在 infinite-width lazy regime 下严格成立，并通过有限宽 U-Net 实验验证特征学习 regime 下的普适性
4. **泛化-记忆谱隙推导**：证明 $\lambda_{\mathrm{gen}} / \lambda_{\mathrm{mem}} \sim n/d$，直接翻译为 $\tau_{\mathrm{gen}} \ll \tau_{\mathrm{mem}}$ 的 Timescale 分离
5. **最优早停理论**：建立岭参数 $\tilde{\gamma} \leftrightarrow 1/\tau$ 的映射，证明最优早停使 bias & variance 都随样本量衰减，而无岭训练末期 bias 趋于 $\Theta(1)$

## 方法详解
- **损失函数**：Denoising Score Matching (DSM)，$\mathcal{L} = \mathbb{E}_{t,\mathbf{x},\pmb{\xi}}[\|\mathbf{s}_\theta(\mathbf{x}+ \sqrt{\Delta_t}\pmb{\xi}) - (-\pmb{\xi}/\sqrt{\Delta_t})\|^2]$
- **噪声重复数 $m$**：每张样本 $m$ 个独立噪声实现，测试集总点数 $N=nm$；插值于标准监督回归 ($m=1$) 与精确噪声期望 ($m\to\infty$) 之间
- **Lazy regime（无限宽）**：NTK 冻结为确定性核 $K$，动力学变为线性，训练时间 $\tau$ 等价于谱截断 $\lambda_c \sim nm/(d\tau)$
- **Gram 矩阵谱结构**（Theorem 2.1）：
  | Bulk | 特征值量级 | 物理含义 | 何时出现 |
  |---|---|---|---|
  | 泛化 bulk | $\lambda_{\mathrm{gen}} \simeq nm/d$ | 捕获目标分布全局特征 | $m=1$ 已存在 |
  | 记忆 bulk | $\lambda_{\mathrm{mem}} \simeq m$ | 编码样本特有噪声方向 | $m>1$ 才出现 |
- **特征向量结构**（Proposition 2.1）：
  - 泛化模式：$\mathbf{u}_1 \propto \mathbf{v}_\lambda^\top \mathbf{Y}$，投影在主方向上
  - 记忆模式：$\mathbf{u}_2 \propto \mathbf{v}_\lambda^\top [\cdots \pmb{\xi}_\perp \cdots]$，包含不同噪声实现的差
- **偏差-方差分解**（Theorem 2.2）：通过 $\tilde{\gamma} \leftrightarrow 1/\tau$ 将岭参数映射为训练时间，无岭训练末期 bias 趋于 $\Theta(1)$（来源是记忆 bulk 的 $O(1)$ 特征值无法被正则化抑制）
- **多项式 regime**（Theorem 2.3）：对次数 $p \in \{1,\ldots,k\}$，泛化子 bulk $\Theta(d^p/p!)$ 个特征值 $\Theta(nm/d^p)$，记忆 bulk $\Theta(d^k/k!)$ 个特征值 $\Theta(m)$；记忆 Timescale $\tau_{\mathrm{mem}} \sim n/d$，比所有泛化 degree 都大 $n^{1-1/k}$ 因子

## 实验与结果
- **数据集**：CelebA，灰度 $32\times32$，flow-matching 参数化
- **模型**：无限宽 CNTK（lazy）+ 有限宽 U-Net（特征学习 regime）
- **$m$ 取值**：实验中 $m=4\sim8$
- **关键发现**：
  1. **Fig. 3**：CelebA CNTK Gram 矩阵确实呈现两 bulk 结构；泛化 bulk 的 $n$ 个特征值与 $d$ 个不同（反映图像低本征维度/各向异性）
  2. **Fig. 4**：谱截断实验——截断秩 $r$ 作为控制旋钮，$f_{\mathrm{mem}}$ 阈值 $\propto n$，FID 最低点几乎与 $n$ 无关；$r/n$ 重组后所有曲线塌缩
  3. **Fig. 5**：U-Net 特征学习 regime 下，即使训练跨越完整泛化→记忆过渡，两 bulk 结构依然保持
  4. **Theorem 2.2 验证**（Fig. 10）：$\tilde{\gamma}=1/\tau$ 校准下，ridge 损失与梯度下降训练时间 $\tau$ 下的测试损失高度吻合
  5. **$m$ 缩放行为**（Fig. 11）：$m=4$ 时偏差饱和于 $\Theta(1)$，$m=1$ 时偏差趋于零，验证记忆化 bulk 依赖 $m>1$
- **最强结果**：理论预测与有限宽 U-Net 实验误差棒<5%，validate 谱机制的普适性

## 相关工作脉络
1. **NTK 理论**（Jacot et al., 2018）：本文扩展至扩散模型，揭示噪声重复重塑谱结构
2. **泛化-记忆权衡**（Zhang et al., 2017）：本文提供扩散模型的特异性谱机制解释
3. **早停与正则化**（Hastie et al., 2019）：本文建立 $\tilde{\gamma} \leftrightarrow 1/\tau$ 的精确映射
4. **扩散模型训练动力学**（Li et al., 2023）：本文首次给出 NTK Gram 矩阵的谱刻画
5. **谱偏差（Spectral Bias）**（Rolnick & Komodakis, 2020）：本文证明扩散模型的谱偏差源于噪声重复
6. **定位差异**：本文区别于泛化理论的关键是噪声重复 $m>1$ 对谱结构的根本性重塑

## 局限性与未来方向
- **高斯数据假设**：理论严格成立条件为各向同性高斯数据，真实图像数据的谱结构可能存在偏差
- **固定噪声分布**：未考虑自适应噪声调度（如 DDPM 的 schedule）对谱演化的影响
- **有限宽修正**：lazy regime 下严格成立，特征学习 regime 的 finite-width 修正尚未完全建立
- **未来方向**：
  1. 扩展至非高斯数据（如流形结构）的谱分析
  2. 研究噪声调度对 Gram 矩阵谱演化的动态影响
  3. 建立 finite-width U-Net 的谱偏差修正理论
  4. 探索谱机制在高分辨率图像/视频生成中的适用性

## 研究启发与可借鉴点
1. **谱分析作为诊断工具**：Gram 矩阵两 bulk 结构可作为扩散模型训练状态的实时诊断指标，预测泛化-记忆过渡点
2. **噪声重复的工程价值**：实践中可选用 $m=4\sim8$ 平衡泛化质量与记忆风险，避免 $m=1$ 的过拟合或 $m\to\infty$ 的失真
3. **早停的谱解释**：最优早停对应谱截断 $\lambda_c$，可建立基于谱间隙的自动早停算法
4. **低本征维度的利用**：CelebA 的泛化 bulk 少于 $d$，提示可利用图像低本征维度优化训练
5. **迁移学习机会**：谱结构在两 bulk 间的演化模式可能跨数据集复用，加速新领域的微调

## 关键术语表
- **NTK（Neural Tangent Kernel）**：无限宽神经网络的核函数，冻结后控制训练动力学为线性
- **Gram 矩阵谱**：NTK 在训练集上的 Gram 矩阵特征值分布，编码泛化-记忆 Timescale
- **噪声重复（Noise Repetition）**：每张样本 $m>1$ 个独立噪声实现，重塑 Gram 矩阵谱结构
- **泛化 bulk**：特征值量级 $\Theta(nm/d)$，编码目标分布全局特征，$m=1$ 已存在
- **记忆 bulk**：特征值量级 $\Theta(m)$，编码样本特有噪声方向，依赖 $m>1$
- **谱隙（Spectral Gap）**：$\lambda_{\mathrm{gen}}/\lambda_{\mathrm{mem}}\sim n/d$，直接翻译为 Timescale 分离
- **Lazy Regime**：无限宽下 NTK 冻结，动力学线性化的训练体制
- **特征学习 Regime**：有限宽 U-Net 下 NTK 演化的训练体制，本文验证普适性

## 可复现要素
- **数据集**：CelebA 灰度 $32\times32$，公开可获取
- **代码**：论文未提及开源，需自行实现
- **关键超参**：$m=4\sim8$，$d\in\{64,128,256\}$，$t=0.1$，$\sigma^2=1$，核为 tanh 二次型对偶激活
- **评估指标**：FID，谱截断 $r/n$，bias-variance 分解
- **实验环境**：MATLAB/Python，delete-one jackknife 误差棒估计
