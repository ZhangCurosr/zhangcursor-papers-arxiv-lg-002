---
title: "Learning-Conditional-Expectation-Operators-via-Functional-Ne"
source: https://arxiv.org/pdf/2609.35598v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:52:03"
field: "算子学习与谱方法"
keywords: ["conditional expectation operator", "density ratio estimation", "functional Newton method", "gradient boosting", "spectral decomposition", "low-rank factorization", "tree-based learning"]
innovations: ["将条件期望算子的谱学习转化为无基依赖的块 Newton 迭代，并由回归树实现近似", "在相对弱学习器精度下证明 O(1/T) 最佳迭代块平稳性与非退化局部极小的全局最优性", "在单核拟合下复用回答条件均值/方差/尾部概率与完整 CDF 等多种推断任务"]
benchmarks: ["Synthetic rank-3 close spectrum", "Synthetic discontinuous piecewise-constant kernel", "Synthetic 20-dim tabular with 17 irrelevant features", "Glass composition–refractive-index (ten-oxide family, n=3013)"]
---

# 论文速读：Learning-Conditional-Expectation-Operators-via-Functional-Newton

## 一句话总结
本文提出了 **Functional Spectral-Newton Method (FSNM)**，一种无需固定基函数或 RKHS 的函数型 Newton 迭代方法，用于学习条件期望算子 $\mathsf{E}$ 的领先奇异谱结构；该方法将密度比核的中心化部分拟合为秩-$d$ 因式分解形式，每个 Newton 更新步转化为预处理的条件均值回归，并用回归树在 staged boosting 框架中近似，同时提供了 $O(1/T)$ 收敛率与全局最优性保证。

## 研究问题与动机
- 条件期望算子 $\mathsf{E}(g)(x)=\mathbb{E}[g(Y)\mid X=x]$ 能一次性复用为多个条件推断任务（概率、矩、分布函数等），现有方法通常依赖固定 RKHS 或神经网络参数化，缺乏对树基模型的利用。
- 现有密度比估计方法（如 uLSIF）直接逼近全密度比 $\kappa$，不暴露低秩/谱结构；CCA 类方法虽与奇异函数相关，但针对的是变换寻找而非算子因子化。
- 能否借助函数型 Newton 框架，以条件均值回归为块方向、用通用回归树实现近似，从而将算子学习引入树模型常用场景（表格/异构数据）？
- 需要在无基选择的前提下，从观测配对数据 $(X_i,Y_i)$ 直接估计并学习 $\kappa_0$ 的秩-$d$ 近似及其奇异系统。

## 核心贡献（创新点）
- **推导了块 Newton 闭式方向并将其实现为条件均值回归 + 预处理**，每个因子的梯度形如 $\nabla_{\phi}L=2\Sigma_{\psi}\phi-2m_{\psi}$，对应块 Hessian 为有限维二阶矩矩阵，避免了固定 RKHS/核的假设。
- **用向量值回归树实例化更新**，形成交替 stagewise boosting 过程，并通过谱归一化（对 $\Sigma_\phi,\Sigma_\psi$ 做等化与对角化）恢复正交因子与显式奇异值。
- **建立了人口水平下降与 $O(1/T)$ 最佳迭代块平稳性率**，并在相对弱学习器精度假设下证明任意非退化局部极小点均为全局最优秩-$d$ 近似（由 Eckart–Young–Mirsky 保证）。
- **提供了基于 Rademacher 复杂度的有限样本一致性界**，刻画经验损失与人口损失的均匀偏差。
- **在多组合成与真实玻璃成分—折射率数据实验中验证**，并展示单核拟合可回答多种条件查询（均值、方差、尾部概率、CDF 与 bootstrap 区间）而无需重拟合。

## 方法详解
- **目标与参数化**：设中心化密度比核 $\kappa_0(x,y)=\sum_{i\ge 1}\sigma_i\phi_i^\star(x)\psi_i^\star(y)$，采用秩-$d$ 因式 $\kappa_{\phi,\psi}(x,y)=\phi(x)^\top\psi(y)$，其中 $\phi\in L^2_0(P_X)^d,\ \psi\in L^2_0(P_Y)^d$。
- **可观测损失**：$L(\phi,\psi)=\mathrm{tr}(\Sigma_\phi\Sigma_\psi)-2\mathbb{E}_{P_{X,Y}}[\phi(X)^\top\psi(Y)]$，由命题 2.3 可知等价于最小化 $\|\kappa_0-\kappa_{\phi,\psi}\|_{L^2}^2$（相差与 $(\phi,\psi)$ 无关的常数项）。
- **函数梯度与块 Hessian**：$\nabla_\phi L=2\Sigma_\psi\phi-2m_\psi$，$\nabla_\psi L=2\Sigma_\phi\psi-2m_\phi$；块 Hessian 为常数乘法算子 $2\Sigma_\psi$ 与 $2\Sigma_\phi$。
- **块 Newton 更新**：固定另一因子时，最优响应为 $\psi_{t+1}^*=\Sigma_{\phi,t}^{-1}m_{\phi,t}$、$\phi_{t+1}^*=\Sigma_{\psi,t+1}^{-1}m_{\psi,t+1}$；加入松弛参数 $\eta\in(0,1]$ 得到 $\psi_{t+1}=(1-\eta_\psi)\psi_t+\eta_\psi\psi_{t+1}^*$，$\phi$ 同理（Gauss–Seidel 交替）。
- **树近似实现**：每个坐标独立在 $Y$（或 $X$）上用回归树拟合条件均值 $m_{\phi,t,j}\approx\mathbb{E}[\phi_{t,j}(X)\mid Y]$；加上 ridge $\rho I_d$ 保证 $\widehat{\Sigma}$ 可逆；实际代码采用每迭代一次的经验谱平衡。
- **谱归一化**：令 $B=\Sigma_\phi^{1/2}\Sigma_\psi\Sigma_\phi^{1/2}=U\Lambda U^\top$，取 $A=\Lambda^{1/4}U^\top\Sigma_\phi^{-1/2}$，将 $(\phi,\psi)\mapsto(A\phi,A^{-\top}\psi)$ 使两二阶矩相等为 $\mathrm{diag}(s_i)$；再令 $D^{1/2}=\Lambda^{1/4}$ 提取正交化奇异函数与奇异值 $s_i$。
- **收敛保证**：在均匀非退化（$\lambda I\preceq\Sigma\preceq\Lambda I$）与相对树回归精度假设下，定理 4.3 给出损失单调下降与 $\min_{t<T}(G_t^\phi+G_t^\psi)\le (L_0-L_\infty)/(aT)$ 的块平稳性率；定理 4.4 表明平衡非退化稳态点的坐标正是 $\mathsf{E}_0$ 的奇异三元组，局部极小即全局最优。

## 实验与结果
- **合成实验 1（接近奇异值，rank-3）**：分布为 $\kappa=1+\sum_{j=1}^3\sigma_j e_j(x)e_j(y)$，$(\sigma_1,\sigma_2,\sigma_3)=(0.18,0.16,0.12)$。FSNM Kernel RMSE 为 $0.1261\pm0.0075$（与 uLSIF $0.1149\pm0.0044$ 并列最低），Spectrum error $0.0286\pm0.0058$（与 ACE $0.0238$ 并列），Subspace error $0.3579\pm0.0242$。
- **合成实验 2（分段常数不连续核）**：FSNM 与 ACE 在所有指标上并列最低；uLSIF 与 Kernel CCA 因光滑核无法刻画锐利分区而显著落后（FSNM Kernel RMSE $0.1393\pm0.0198$）。
- **合成实验 3（20 维表格 + 17 个无关特征）**：仅前 3 维相关，FSNM/ACE 利用树的分割实现自动特征选择；uLSIF/Kernel CCA 失效严重（FSNM RMSE $0.1880$ vs. uLSIF $0.7102$, Kernel CCA $0.6651$）。置换敏感性诊断显示三相关维显著高于其余 17 个无关维。
- **多查询复用（单核拟合后）**：条件均值 RMSE 0.0237、条件方差 0.0166、尾部概率 0.0233；条件 CDF 经单调投影后，100 次 bootstrap 在三点 $x\in\{-0.6,0,0.6\}$ 覆盖率达 99.7%，平均带宽 0.0655。
- **非线性依赖检测**：在 $Y=X^2+\epsilon$ 场景，Pearson 相关近乎零，但 FSNM 估计谱 $(0.8579,0.7659,0.5186)$ 与置换检验 $p=0.001$ 一致检测到依赖；独立情形 $p=0.965$ 不误报，且仅选 1 次 boosting 迭代。
- **真实玻璃成分—折射率（十氧化物族，$n=3013$）**：验证选定 rank=20、depth=5、leaf=25、迭代 39（验证 loss $-5.6476$），有效谱秩 11.41。直接 plug-in 预测 RI 的测试 MSE=0.0051、RMSE=0.0711、$R^2=0.811$，相对训练均值基线 MSE 降低 81.1%；校准后 50%/70%/90% 区间覆盖率分别为 54.6%/71.5%/89.4%，阈值 $\tau=1.8$ 的 Brier score 0.0375 较气候学预测 0.1156 降低 67.6%。
- **强结果汇总**：跨合成与真实数据，FSNM 在 Kernel RMSE、Spectrum error、Subspace error 上普遍与 ACE 并列最佳，并在需要自适应分割与特征选择的任务中显著优于 uLSIF 与 Kernel CCA。

## 相关工作脉络
- **条件均值嵌入与 Neural Conditional Probability**：同目标为“单拟合多查询”，但 FSNM 不以 RKHS/神经网络为归纳偏置，而是直接在 $L^2$ 空间以块 Newton + 回归树求得谱分解。
- **密度比估计（uLSIF 等）**：直接学习完整 $\kappa$，不追求低秩/谱；FSNM 学习中心化部分的秩-$d$ 因子化，从而显式获得奇异函数与值。
- **CCA / Deep CCA / 核 CCA**：寻找最大相关变换；FSNM 从联合秩-$d$ 最小二乘出发，条件均值回归作为块 Newton 方向出现，同时输出奇异值与正交因子。
- **ACE（交替条件均值回归）**：等价于 rank-1 的最优变换寻找；FSNM 推广至 rank-$d$，并以 operator 视角与谱归一化统一刻画。
- **Newton/Gradient Boosting（XGBoost、LightGBM、Sigrist 2021、Zozoulenko 2026）**：传统树提升优化标量/向量响应预测损失；FSNM 优化双线性算子因子分解，块 Hessian 为对因子二阶矩矩阵，Newton 最佳响应为对对面域的条件期望。
- **Koopman/转移算子学习**：与条件期望算子在动力系统中的角色一致，但 FSNM 面向一般联合分布的密度比谱，不依赖 Markov 转移假设。

## 局限性与未来方向
- 超参（rank、深度、叶大小、步长、迭代数）依赖离散网格与验证搜索，缺乏自适应秩选择机制。
- 理论保证依赖弱学习器相对精度假设与均匀非退化假设，且有限样本界随特征类复杂性（Rademacher）变化。
- 当前实验与公式面向标量响应；延伸至结构化响应空间（向量、分布等）尚未讨论。
- 平衡变换在含 ridge 与离散树拟合时仅数值上有益，理论分析仍使用最终一次性平衡版本。

## 研究启发与可借鉴点
- 将块 Newton 推导与树基弱学习器结合的思路可迁移至其他双变量算子学习任务（如转移算子、因果谱特征学习），无需预设核或神经参数。
- 实验设计中对“子空间误差/谱误差/正交误差”的细分指标，以及对不同数据形态（接近谱、分段常数、高维含噪）的针对性合成基准，值得在后续算子学习中复用。
- 单核拟合 + plug-in 多查询范式（均值/方差/尾部/完整 CDF 与 bootstrap）展示了“一次学习、多处复用”的效率，可与本团队的条件推理/不确定性量化方向结合。
- 每迭代经验谱平衡作为工程启发可改善条件数，后续工作可将其纳入理论或推广到更一般的仿射重参数化族。

## 关键术语表
**Conditional expectation operator** $\mathsf{E}$：将 $g(Y)$ 映射到条件期望 $\mathbb{E}[g(Y)\mid X]$ 的线性算子，其核为密度比 $\kappa$。
**Centered density ratio kernel** $\kappa_0=\kappa-1$：去掉常数分量后承载非平凡相关结构的部分，位于 $L^2_0(P_X)\otimes L^2_0(P_Y)$。
**Functional block-Newton update**：固定一因子时目标关于另一因子为二次泛函，块 Hessian 为二阶矩矩阵，Newton 步等价于对对面变量的条件均值回归并左乘协方差逆。
**Spectral normalization / balancing**：利用因子化冗余 $(\phi,\psi)\mapsto(A\phi,A^{-\top}\psi)$ 将两二阶矩矩阵等化并对角化，从而从拟合因子中提取奇异值与正交奇异函数。
**Relative weak-learner accuracy**：要求每步树回归能恢复上一残差的固定比例，是证明 $O(1/T)$ 块平稳性率的关键条件。
**Block stationarity measure** $G_t^\phi+G_t^\psi$：当前因子与其精确块最优响应之间的损失差距之和，定理 4.3 对其给出最佳迭代 $O(1/T)$ 上界。
**Rademacher complexity bound**：通过有界差分与乘积类复杂度的收缩不等式，给出经验与人口损失在拟合函数类上的均匀偏差界。
**Isotonic projection of CDF**：将有限样本得到的有符号条件分布曲线投影到单调非降、取值在 $[0,1]$ 的累积分布函数类，以恢复分布一致性。

## 可复现要素
- 数据集：合成数据由作者脚本生成；真实数据为来自 SciGlass、INTERGLAD 与专利的玻璃成分—折射率记录，限制于十氧化物族并过滤 RI 到 $[1,4.5]$，共 3013 条；数据与划分详见附录 B。
- 代码/权重：开源仓库 https://github.com/thiagorr162/fsnm，含各实验独立 notebook 与完整超参网格。
- 关键超参：step size 0.1（主实验）/0.15（部分合成），ridge $\rho=10^{-8}$，最大深度 3/5，最小叶大小 25/120/250，最多 40–80 次迭代；秩在 $\{3,5,10,20\}$ 中由验证损失选取。
- 基线：ACE（同树预算）、uLSIF（选高斯带宽与 ridge）、Kernel CCA（800 随机地标）。
