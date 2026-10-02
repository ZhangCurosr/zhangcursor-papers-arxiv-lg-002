---
title: "Learning-Conditional-Expectation-Operators-via-Functional-Ne"
source: https://arxiv.org/pdf/2609.35598v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:51:55"
field: "算子学习与谱方法"
keywords: ["conditional expectation operators", "density ratio estimation", "functional Newton methods", "spectral decomposition", "gradient boosting", "tree-based learning"]
innovations: ["块牛顿谱因式分解使条件均值回归方向可直接由回归树近似", "不平衡-再平衡机制提取正交奇异函数与显式奇异值", "总体景观证明任何非退化局部极小均为全局最优 rank-d 截断"]
benchmarks: ["rank-three close singular values", "discontinuous piecewise-constant kernel", "20-d tabular with irrelevant features", "glass composition–refractive-index"]
---

# 论文速读：Learning-Conditional-Expectation-Operators-via-Functional-Newton

## 一句话总结
本文提出 Functional Spectral-Newton Method (FSNM)，通过交替功能块牛顿更新学习条件期望算子 $\mathsf{E}$ 的领先奇异结构，无需预设基或 RKHS；每次块更新可近似为向量值回归树，合成与真实数据实验表明该方法能准确恢复低秩密度比核及其谱结构，且单次拟合即可支持多类条件查询。

## 研究问题与动机
- 条件期望算子 $\mathsf{E}: L^2(P_Y) \to L^2(P_X)$ 的 learning 可统一处理条件概率、回归函数、条件矩等，但现有方法通常固定 RKHS 或神经网络参数化，难以直接提取算子的谱结构。
- 已有密度比估计（如 uLSIF）直接拟合完整 $\kappa$，不暴露低秩/谱结构；CCA 类方法依赖核或神经网络，且需预设核空间。
- 如何在不固定基/核的前提下，用可被表格/混合数据广泛使用的回归树类工具学习算子的前 $d$ 个奇异函数与奇异值？
- 如何为基于树的块牛顿谱学习方法提供一致的下降与总体景观理论？

## 核心贡献（创新点）
- 从去中心化密度比核的无偏最小二乘目标出发，推导闭式块牛顿方向：每个方向是预处理条件均值回归，仅需二阶矩矩阵与条件均值估计，不固定基/RKHS。与 Kernel CCA/uLSIF 的本质区别在于“谱因式分解 + 通用回归器”。
- 将块牛顿方向实例化为回归树，得到交替 Stagewise 增强流程；利用因子分解的不变性构造平衡矩阵，从而获得正交奇异向量与显式奇异值，无需额外后处理。与 ACE 的区别在于“块牛顿谱目标而非交替最大化相关”。
- 建立总体层面的下降性与最佳迭代块平稳性 $O(1/T)$ 收敛速率；证明全 $L^2$ 空间上任何非退化局部极小均为全局最优 rank-$d$ 谱截断（Eckart–Young 等价）。与一般 boosting 文献的区别在于“谱目标 + 块牛顿 + 总体景观全局最优”。
- 给出基于 Rademacher 复杂度的有限样本一致浓度界，表明经验损失在拟合凸包类上一致逼近总体损失。

## 方法详解
- 目标与参数化：在中心化空间 $L^2_0(P_X), L^2_0(P_Y)$ 中学习 $\phi:\mathcal{X}\to\mathbb{R}^d,\psi:\mathcal{Y}\to\mathbb{R}^d$，使 $\kappa_0(x,y)\approx \phi(x)^\top\psi(y)$，$\mathbb{E}\phi(X)=\mathbb{E}\psi(Y)=0$。
- 可观测损失：最小化 $L(\phi,\psi)=\mathrm{tr}\{\Sigma_\phi\Sigma_\psi\}-2\mathbb{E}_{P_{X,Y}}[\phi(X)^\top\psi(Y)]$，其中 $\Sigma_\phi=\mathbb{E}\phi(X)\phi(X)^\top$，$\Sigma_\psi=\mathbb{E}\psi(Y)\psi(Y)^\top$。
- 函数梯度与块海森：$\nabla_\phi L=2\Sigma_\psi\phi-2m_\psi$，$\nabla_\psi L=2\Sigma_\phi\psi-2m_\phi$；块海森分别为 $2\Sigma_\psi$ 与 $2\Sigma_\phi$（均为常数矩阵乘法算子）。
- 块牛顿方向：对 $\psi$ 的精确块最优响应为 $\psi_{t+1}^*=\Sigma_{\phi,t}^{-1}m_{\phi,t}$；FSNM 采用阻尼更新 $\psi_{t+1}=(1-\eta_\psi)\psi_t+\eta_\psi\Sigma_{\phi,t}^{-1}m_{\phi,t}$，$\phi$ 同理。
- 树近似：每维坐标分别用回归树估计条件均值 $m_{\phi,t}$ 或 $m_{\psi,t+1}$；数值稳定加 ridge $\rho I_d$。
- 谱归一化：利用 $(\phi,\psi)\mapsto(A\phi,A^{-\top}\psi)$ 保持 $\kappa_{\phi,\psi}$ 不变的不变性，构造 $A=\Lambda^{1/4}U^\top\Sigma_\phi^{-1/2}$ 使两因子协方差相等为 $D=\mathrm{diag}(s_i)$；再除以 $D^{1/2}$ 得正交奇异函数，$s_i=\sqrt{\lambda_i(\Sigma_\phi\Sigma_\psi)}$ 即为奇异值。
- 算法：每次迭代做 $\psi$ 更新、$\phi$ 更新；可选每步平衡以提升条件数；最终迭代再做一次平衡输出谱三元组 $(s_i,\phi_i^\perp,\psi_i^\perp)$。

## 实验与结果
- 合成谱相近 rank-3：FSNM Kernel RMSE 0.1261±0.0075，谱误差 0.0286±0.0058，子空间误差 0.3579±0.0242，与 ACE 并列最优（uLSIF RMSE 更低但无谱/子空间指标）。
- 不连续分段常数 rank-3：FSNM Kernel RMSE 0.1393±0.0198，谱误差 0.0301±0.0113，子空间误差 0.2678±0.0846，树方法显著优于高斯核 uLSIF/Kernel CCA。
- 20 维含 17 个无关特征的表格输入：FSNM Kernel RMSE 0.1880±0.0227，谱误差 0.0363±0.0114，子空间误差 0.4689±0.1062；相关特征排序正确，无关坐标敏感度接近 0。
- 单次拟合多查询：同一 rank-3 核经 plug-in 估计条件均值/variance/尾概率 RMSE 分别为 0.0237/0.0166/0.0233；条件 CDF 在三点平均带宽 0.0655，覆盖≥99.7%。
- 非线性零相关依赖检测：Pearson 近 0，FSNM+置换检验 $p=0.001$，谱明显非零。
- 真实玻璃成分-折射率：选定 rank=20、depth=5、leaf=25，迭代 39；有效谱秩 11.41，RI 预测 RMSE 0.0711，$R^2=0.811$，较训练均值基准 MSE 降 81.1%；50/70/90% 区间覆盖 54.6/71.5/89.4%；筛选概率 Brier 0.0375，较气候学基线降 67.6%。

## 相关工作脉络
- Conditional mean embedding / Neural Conditional Probability：同样支持单拟合多查询，但通常固定 RKHS/NN 参数化；FSNM 直接在 $L^2$ 上学习谱并适配树归纳偏置。
- Density ratio estimation（uLSIF 等）：直接估计完整 $\kappa$，不暴露低秩谱；FSNM 通过中心化核的 rank-$d$ 因式同时恢复奇异函数与奇异值。
- Canonical correlation analysis / Deep CCA / Kernel CCA：最大化变换间相关性，但需核/网络；FSNM 从联合 LS 目标出发，条件均值回归来自块牛顿方向。
- ACE（Breiman & Friedman, 1985）：交替条件均值回归求 max 相关，仅 rank-1；FSNM 同时优化 rank-$d$ 因式并给出谱归一化。
- Functional gradient/Newton boosting（XGBoost/LightGBM 系列）：优化标量预测损失；FSNM 优化双线性算子因式，块海森为对端二阶矩矩阵。
- Spectral features for causal/econometric problems：利用 $\mathsf{E}$ 的谱做因果识别；FSNM 提供一种可直接学习这些谱的工具并附理论保障。

## 局限性与未来方向
- 当前超参（秩、深度、叶大小、步长）依赖固定网格搜索与验证，缺少自适应秩选择。
- 有限样本界以拟合函数类的凸包约束刻画，速率依赖所选弱学习器类；对更复杂响应结构（多变量、函数值）尚未覆盖。
- 实验集中在合成与单指标玻璃数据，因果推断/强化学习的世界模型应用仍待验证。
- 每步平衡虽改善条件数，但与总体理论的等变性论证不完全一致，理论刻画需补充。

## 研究启发与可借鉴点
- 将“谱目标 + 块牛顿 + 通用弱学习器”解耦：任何可拟合条件均值的 learners（线性/树/神经网络）均可接入，便于把 operator learning 引入表格建模生态。
- 损失重写为可观测形式 $\mathrm{tr}(\Sigma_\phi\Sigma_\psi)-2\mathbb{E}[\phi^\top\psi]$，避免直接估计未知 $\kappa_0$，这一技巧可推广到其它核因式分解任务。
- 谱归一化通过矩阵相似变换把任意因式转化为正交奇异表示，且与更新兼容；可在其它双因式谱学习算法中复用。
- 总体景观证明“任何非退化局部极小即全局最优 rank-$d$ 截断”，为交替优化的陷阱问题提供强保障，值得在其它算子因式分解中类比检验。
- 单一核驱动条件均值/方差/CDF/指示函数等多查询的 plug-in 范式，适合下游需要多目标条件推断的场景（不确定性量化、筛选策略）。

## 关键术语表
- **Conditional expectation operator**：把输出空间的可积函数映射为条件期望函数的线性算子 $\mathsf{E}(g)(x)=\mathbb{E}[g(Y)\mid X=x]$。
- **Centered density ratio kernel**：去中心化联合-边际密度比 $\kappa_0(x,y)=\kappa(x,y)-1$，其秩-$d$ 截断对应算子前 $d$ 个奇异分量。
- **Functional block-Newton update**：在函数空间对双因式目标做块坐标 Newton 步，方向为预处理条件均值回归。
- **Spectral normalization**：利用 $(\phi,\psi)\mapsto(A\phi,A^{-\top}\psi)$ 不变性，构造平衡矩阵使两因子协方差相等并提取奇异值/正交奇异函数。
- **Block stationarity rate**：算法轨迹中最小块优化间隙或梯度范数以 $O(1/T)$ 衰减的最佳迭代速率。
- **Relative weak-learner accuracy condition**：要求树回归在每步至少恢复剩余块残差的固定比例，保证下降。
- **Population landscape global optimality**：在全 $L^2$ 无约束下，任何非退化局部极小对应最优 rank-$d$ 奇异子空间。
- **Rademacher complexity bound**：控制经验损失在拟合凸包类上的一致偏差，用于有限样本泛化保证。

## 可复现要素
- 代码与实验 notebooks：https://github.com/thiagorr162/fsnm
- 数据集：合成数据由读者自生成；真实数据为 SciGlass/INTERGLAD/patent 整合的玻璃成分-折射率记录（论文中已描述预处理流程）。
- 关键超参（FSNM）：ridge $\rho=10^{-8}$；步长 $\eta\in\{0.1,0.15\}$；深度 3–5；最小叶大小 25–300；最大迭代 40–80；每步/最终平衡。
- 基线实现要点：ACE 同树预算；uLSIF 选高斯带宽与 ridge；Kernel CCA 使用 800 随机地标并选带宽/正则。
- 论文未提及：硬件配置、随机种子完整列表、预训练权重。
