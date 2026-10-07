---
title: "PROXIMALFM-Amortized-Proximal-Causal-Inference-under-Hidden"
source: https://arxiv.org/pdf/2610.08078v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-10-07 11:14:52"
---

# 论文速读：PROXIMALFM-Amortized-Proximal-Causal-Inference-under-Hidden

## 一句话总结
本文提出 PROXIMALFM，首次将先验数据拟合网络(PFN)范式扩展至近端因果推断设定，通过链式随机 MLP 构建满足近端假设的合成先验，实现单次 Transformer 前向传播即可获得隐藏混杂下 CATE 后验分布的摊销化贝叶斯推断，免除了传统方法对病态积分方程的反复优化。

## 研究问题与动机
- 标准因果推断依赖无未测量混杂假设，现实中隐藏混杂（如未记录疾病严重程度）普遍存在，导致后门调整等传统方法产生偏倚。
- 近端因果推断利用双侧代理变量（$Z, W$）在隐藏混杂下识别因果效应，但非参数估计需迭代求解病态 Fredholm 积分方程，对数据量、超参数与优化稳定性极度敏感。
- 现有摊销化因果模型（如 Do-PFN、CausalPFN）仅适用于无混杂或工具变量设定，无法直接处理近端设定中的桥函数估计与 CATE 后验推断。
- 缺乏统一基准评估近端学习方法在合成、半合成与真实物理系统中的泛化性与后验校准表现。

## 核心贡献（创新点）
- **摊销化近端因果推断框架**：将 PFN 引入近端设定，以单次前向传播替代迭代优化，直接输出 $p(Q(x^*)|\mathcal{D}_N)$。与现有近端方法的本质区别在于彻底消除对数值积分方程求解的依赖，实现端到端推理。
- **可控链式合成先验生成**：设计六段随机 MLP（$f_X \to f_U \to f_W \to f_Z \to f_A \to f_Y$）并通过混杂份额、边保留概率、重叠度三组旋钮精确控制近端假设强度。与通用因果生成模型的区别在于显式编码完整性与桥函数存在条件作为生成约束。
- **噪声感知分布匹配损失**：提出对预测直方图卷积蒙特卡洛方差后再取对数似然的损失函数，缓解离散目标估计的梯度震荡。与标准 MSE/交叉熵损失的区别在于显式建模监督信号本身的采样不确定性。
- **理论收敛保证**：证明在完整性与桥函数存在条件下，Oracle 贝叶斯后验均值依概率收敛至真值 CATE（Theorem A.4）。与经验型深度因果方法的本质区别在于提供摊销近似误差的一致界。
- **多维度实证基准**：构建合成（288 实例）、半合成（INRIA-SODA 12 数据集×96 配置）与物理（Causal Chambers 光隧道）三阶段评测体系，系统刻画近端方法在高混杂+弱代理场景的性能边界。

## 方法详解
- **架构设计**：基于 TabICLv2 backbone，引入结构角色嵌入（structural-role embeddings）显式区分 $X, Z, W, A$ 的 DAG 位置；采用处理感知目标编码器，分别嵌入事实观测 $Y = Y(A)$ 的治疗/对照信息。
- **合成先验 $\Pi$ 采样**：每个 episode 从三角分布采样 $d_U \in [1,10]$，均匀采样 $d_W, d_Z, d_X \in [d_U, 10]$；实例化单个 TabICL MLP 作为各因果机制，通过列级随机子采样控制直接边与代理边的拓扑结构，循环直至满足 $\min(d_W, d_Z) \ge d_U > 0$ 与分类优势等完整性启发式条件。
- **Monte Carlo CATE 监督**：固定查询 $x_i$，从同一 SCM 重复 $K=250$ 次采样潜在结果对 $(Y_0, Y_1)$；通过旋钮 $\gamma \sim \text{Unif}[0,1]$ 线性插值个体效应 $\tau_i = \gamma \tau_i^{\text{raw}} + (1-\gamma)\bar{\tau}$ 控制异构性；计算 CEPO 估计 $\hat{\mu}_a(x_i)$ 与无偏目标 $\widehat{Q}_{i,K} = \hat{\mu}_1 - \hat{\mu}_0$ 及方差 $S^2_{i,K}$。
- **损失函数**：总损失 $\mathcal{L} = \mathcal{L}_{\text{noise-aware}} + \lambda_{\text{mean}}\mathcal{L}_{\text{mean}}$，其中
  $\mathcal{L}_{\text{noise-aware}} = -\frac{1}{n_q}\sum_i \log \tilde{q}_\theta(\widehat{Q}_{i,K}|x_i, \mathcal{D}_N)$，$\tilde{q}_\theta = q_\theta * \mathcal{N}(0, S^2_{i,K})$；$\mathcal{L}_{\text{mean}}$ 为预测均值与 $\widehat{Q}_{i,K}$ 的 MSE 正则。
- **输出形式**：单头分布预测，将 CATE 映射至 $[-5, 5]$（以观测 $Y$ 标准差为单位）的 2000 个等宽 bin，推理时直接返回后验概率质量函数。

## 实验与结果
- **数据集与设置**：合成评估 $8 \times 12$ 因子设计 × 3 次重复 = 288 实例；半合成基准选自 INRIA-SODA 12 个公开表格数据集（特征≥20，样本≥10k），构建 $2(\text{线性/非线性}) \times 2(\text{弱/强混杂}) \times 2(\text{弱/强代理}) = 96$ 配置 × 3 种子 = 288 实现；物理实验采用 Causal Chambers 光隧道（弱/强混杂、线性/非线性控制函数、高/中/低代理信息度）。
- **对比基线**：KPV、PMMR、NMMR（均重实现 Mastouri et al., 2021 并适配二进制处理与 CATE 预测
