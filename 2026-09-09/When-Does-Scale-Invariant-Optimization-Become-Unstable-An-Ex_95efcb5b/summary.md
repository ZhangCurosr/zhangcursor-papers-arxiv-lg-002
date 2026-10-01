---
title: "When-Does-Scale-Invariant-Optimization-Become-Unstable-An-Ex"
source: https://arxiv.org/pdf/2609.09116v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:55:40"
field: "深度学习优化理论"
keywords: ["scale-invariant optimization", "weight decay", "learning rate schedule", "normalization dynamics", "homogeneous optimizer", "discrete-time dynamics", "edge of stability"]
innovations: ["推导出尺度不变块有效步长的精确单标量递推定律 B_t", "证明常数学习率+权重衰减下内部平衡点为不稳定螺旋源", "通过齐次指数 nu 统一分类 SGD/SGDM/Adam 的自抑制强度差异"]
benchmarks: ["MNIST", "CIFAR-10", "WikiText", "OpenWebText"]
---

# 论文速读：When-Does-Scale-Invariant-Optimization-Become-Unstable-An-Exact-Schedule-Law-with-Weight-Decay

## 一句话总结
本文推导了归一化神经网络中尺度不变块的有效步长精确离散时间演化定律，提出单一标量 $B_t$ 刻画学习率调度与权重衰减的强制作用，揭示了常数学习率+权重衰减下系统固有的螺旋源不稳定性，并通过同质优化器框架统一解释了自适应方法更强的扩张倾向。

## 研究问题与动机
- 归一化层（如BatchNorm、LayerNorm）使神经网络部分参数块具有正尺度不变性，导致有效步长 $\eta / \|w\|^2$ 受参数范数反馈调节，形成学习率调度、权重衰减与范数增长之间的复杂交互。
- 现有工作虽观察到该反馈环，但分析多为渐近、近似或特定 regime 层面，缺乏逐迭代、任意调度下的精确动力学定律。
- 常数学习率配合权重衰减时训练常表现周期性振荡/不稳定，其根本原因（是否源于离散时间几何而非噪声或连续近似）尚未从第一性原理阐明。
- 不同优化器（SGD vs Adam）在归一化下表现出系统性差异，但缺乏统一的理论解释框架。

## 核心贡献（创新点）
1. **精确单标量定律**：推导出有效步长因子 $\Phi_t$ 的精确递推关系 $\Phi_{t+1} = B_t \Phi_t / (1 + \Phi_t^2 \|\bar{g}_t\|^2)$，将调度/衰减作用完全编码于 $B_t$，与几何自抑制效应分离；与已有工作的本质区别在于该定律为纯代数恒等式，无需线性化或连续时间近似，适用于任意时变调度。
2. **不稳定性机制的精确刻画**：在可解归一化线性模型中证明唯一内部平衡点位于切换面上，且为不稳定螺旋源（复特征值模长 $>1$），表明常数调度下的周期行为是离散时间几何的必然结果，非噪声 artifacts。
3. **随机扩展与噪声累积分析**：将精确定律推广至随机优化，给出条件对数漂移的精确表达式 $\mathbb{E}_t[\Delta \log \Phi_t] = \beta_t - q_t$，并建立有限视界集中不等式，刻画小批量噪声如何移动临界带。
4. **同质优化器统一框架**：通过齐次指数 $\nu$ 分类优化器（SGD/SGDM 对应 $\nu=1$，Adam 对应 $\nu=0$），揭示自抑制强度差异的第一性原理解释；与已有工作的区别在于将多种优化器纳入同一标量框架并给出精确递推。
5. **因果干预验证**：通过合成强制 $B_t \equiv B$ 的学习率调度进行干预实验，证明性能在 $B=1$ 处尖锐峰值（$\pm 2\%$ 扰动导致 >20 点下降），确立 $B_t$ 为因果控制变量而非仅是诊断指标。

## 方法详解
- **尺度不变块设定**：参数块 $w \in \mathbb{R}^d \setminus \{0\}$ 满足 $\mathcal{L}(\alpha w) = \mathcal{L}(w)$，采用极坐标分解 $w_t = r_t u_t$（$r_t = \|w_t\|$，$u_t \in \mathbb{S}^{d-1}$）。
- **有效步长因子定义**：$\Phi_t := \eta_t / (a_t r_t^2)$，其中 $a_t = 1 - \eta_t \lambda_t$ 为耦合权重衰减压缩因子。
- **无尺度梯度**：$\bar{g}(u) = r \nabla_w \mathcal{L}(ru)$ 仅依赖于方向 $u$ 且切于球面（$\langle u, \bar{g} \rangle = 0$）。
- **精确递推定律（Theorem 2.1）**：
  - 半径演化：$r_{t+1}^2 = r_t^2 a_t^2 (1 + \Phi_t^2 \|\bar{g}_t\|^2)$
  - 方向更新：$u_{t+1} = (u_t - \Phi_t \bar{g}_t) / \|u_t - \Phi_t \bar{g}_t\|$
  - 有效步长递推：$\Phi_{t+1} = B_t \Phi_t / (1 + \Phi_t^2 \|\bar{g}_t\|^2)$，其中调度因子 $B_t = \eta_{t+1} / (\eta_t a_t a_{t+1})$
- **收缩/扩张判定**：
  - 若 $B_t \leq 1$，则 $\Phi_{t+1} \leq \Phi_t$ 无条件成立（调度强制收缩）
  - 若 $B_t > 1$，需 $\Phi_t \|\bar{g}_t\| \geq \sqrt{B_t - 1}$ 才能收缩（几何自抑制对抗）
- **切换面**：稳态平衡满足 $\Phi_t \|\bar{g}_t\| = \sqrt{B - 1}$，分离收缩主导与扩张主导 regime。
- **随机扩展（Theorem 2.3）**：定义 $X_t = \Phi_t^2 \|\hat{g}_t\|^2$，$Z_t = \log(1+X_t)$，$q_t = \mathbb{E}_t[Z_t]$，得条件对数漂移 $\mathbb{E}_t[\Delta \log \Phi_t] = \beta_t - q_t$，且偏差构成鞅差序列，给出高概率集中界。
- **同质优化器框架（Theorem 4.1）**：对更新方向满足 $p_t = r_t^{-\nu} \bar{p}_t$ 的优化器，定义 $\Psi_t = \eta_t / (a_t r_t^{1+\nu})$ 与广义调度因子 $\widetilde{B}_t^{(\nu)} = (\eta_{t+1}/\eta_t) / (a_t^\nu a_{t+1})$，递推为 $\Psi_{t+1} = \widetilde{B}_t^{(\nu)} \Psi_t / \|u_t - \Psi_t \bar{p}_t\|^{1+\nu}$。
- **SGDM 增广状态定律（Theorem 4.2）**：引入 $z_t = r_t m_{t+1}$，分解为径向分量 $c_t = \langle u_t, z_t \rangle$ 与正交分量 $s_t$，揭示动量在 $c_t > 0$ 时的径向放大效应。
- **Adam 精确定律（Theorem 4.3）**：定义无尺度矩 $\tilde{m}_t = r_{t-1} m_t$、$\tilde{v}_t = r_{t-1}^2 v_t$，得 $\nu = 0$ 情形，分母指数为 1（线性自抑制），并解释 $\varepsilon > 0$ 为光滑微扰。

## 实验与结果
- **精确映射验证**：在 isotropic 2D 映射上，float64 精度下递推残差 $< 2 \times 10^{-15}$，常数调度维持 $B_t > 1$ 扩张，step decay 产生单次收缩冲击，cosine decay 平滑穿越边界。
- **神经网络验证**：
  - MNIST BN-MLP 与 CIFAR-10 BN-ConvNet：float32 精度下残差 $\sim 10^{-6}$，扩张分数紧密跟踪 $B_t$ 轨迹。
  - Target-$B_t$ 干预（CIFAR-10）：强制 $B_t \equiv B$ 产生尖锐性能峰，$B=1$ 时最优，$\pm 2\%$ 扰动导致 >20 点下降。
- **优化器机制验证**（3-Block BN-ConvNet, CIFAR-10, 10k 步）：
  - SGDM 径向放大：$c_t > 0$ 时扩张率 0.994 vs $c_t \leq 0$ 时 0.56，增广状态残差 $1.2 \times 10^{-7}$。
  - 分母指数分类： empirically 清晰分离 $\nu=1$（SGD/SGDM）与 $\nu=0$（Adam）。
  - Adam $\varepsilon$-连续性：随 $\varepsilon \to 0$ 平滑收敛至精确定律。
- **Transformer 压力测试**：
  - small_gpt2 ($d_{model}=256$, 4 blocks) on WikiText 与 gpt2 ($d_{model}=768$, 12 blocks) on OpenWebText，各 10k 步，batch=32。
  - SGD/SGDM 中位数残差 $1.19 \times 10^{-7}$，常数/step 调度扩张率 $\approx 1.0$，cosine 降至 $\sim 10^{-4}$。
  - Adam 残差 $10^{-6} \sim 10^{-4}$，常数/step 扩张率 0.44–0.49，cosine 降至 0.033（WikiText）/ 0.095（OpenWebText），符合 $\nu=0$ 预测。
- **最强结果**：Target-$B_t=1$ 干预在 MNIST 上取得 87.6% 准确率，显著优于 $B=1.02$（86.4%）与 $B=0.98$（87.4%）。

## 相关工作脉络
1. **Li & Arora (2020) Intrinsic Learning Rate**：指出 BN+SGD+WD 等价于指数增长学习率无 WD；本文通过精确 $B_t$ 框架统一刻画任意调度，超越渐近等价视角。
2. **Lobacheva et al. (2021) Periodic Behavior**：观察到 BN+WD 训练的周期性不稳定；本文从离散时间螺旋源给出第一性原理解释，非经验现象。
3. **Kosson et al. (2024) Rotational Equilibrium**：分析 WD 如何平衡跨网络学习；本文揭示平衡点内在不稳定性，常数调度无法稳定维持。
4. **Cohen et al. (2021) Edge of Stability**：研究曲率驱动的 EoS；本文聚焦范数驱动的尺度不变机制，与曲率视角互补。
5. **d'Angelo et al. (2024) Why Weight Decay**：讨论 WD 在现代 DL 中作用；本文从精确递推角度给出 WD 与调度因子的代数耦合关系。
6. **Arora et al. (2019) Auto Rate-Tuning**：分析 BN 自调节学习率；本文提供 Block-wise 精确动力学，不限于 BN 自调节机制。

## 局限性与未来方向
- 分析为 block-conditional（块条件），仅对精确尺度不变的参数块严格成立；标准 Transformer 中 embedding、bias、LM head 等不满足此对称性。
- 各向同性模型的螺旋源不稳定性为精确结论，但各向异性情形与通用深度网络尚无全局收敛证明。
- 耦合权重衰减给出精确单标量定律；解耦权重衰减（如 AdamW）破坏该结构，需扩展至双标量框架。
- Adam 的 $\varepsilon=0$ 精确定律成立，但 $\varepsilon>0$ 仅为光滑微扰，理论刻画尚不充分。
- 未来方向包括：扩展至解耦衰减、形式化 $\varepsilon$-微扰理论、推广至更大规模架构及多标量递推。

## 研究启发与可借鉴点
- **调度设计的操作化视角**：将学习率调度转化为对 $B_t$ 轨迹的直接控制，而非单纯调 $\eta_t$，为 schedule 设计提供新的优化坐标。
- **同质优化器框架的可迁移性**：$\nu$ 分类揭示自抑制强度差异，可指导在新架构/归一化变体中选择优化器（如强归一化场景慎用 Adam）。
- **实验验证范式**：通过合成 target-$B_t$ 调度进行因果干预，分离诊断相关性与因果性，值得在训练动力学研究中推广。
- **诊断指标的工程价值**：扩展分数与递推残差可作为训练监控指标，实时检测 regime 切换与不稳定性风险。
- **理论与实验闭环**：从精确定律→可解模型→神经实验→Transformer 压力测试的逐层验证策略，为理论 DL 研究树立范例。

## 关键术语表
**尺度不变性 (Scale Invariance)**：损失函数满足 $\mathcal{L}(\alpha w) = \mathcal{L}(w)$ 的性质，使优化仅依赖参数方向而非范数。
**有效步长因子 (Effective Stepsize Factor) $\Phi_t$**：定义为 $\eta_t / (a_t r_t^2)$，刻画尺度不变块上沿球面切方向的实际控制步长。
**调度因子 (Schedule Factor) $B_t$**：标量 $B_t = \eta_{t+1} / (\eta_t a_t a_{t+1})$，完全编码学习率与权重衰减调度的强制作用。
**几何自抑制 (Geometric Self-Quenching)**：由参数范数增长导致的 $\Phi_t$ 自然衰减效应，体现为递推分母中的 $1 + \Phi_t^2 \|\bar{g}_t\|^2$ 项。
**切换面 (Switching Surface)**：稳态平衡条件 $\Phi_t \|\bar{g}_t\| = \sqrt{B_t - 1}$，分离收缩主导与扩张主导 regime 的临界曲面。
**同质优化器框架 (Homogeneous-Optimizer Framework)**：通过齐次指数 $\nu$ 分类优化器更新方向的尺度行为，$\nu=1$ 对应 SGD/SGDM，$\nu=0$ 对应 Adam。
**螺旋源不稳定性 (Spiral-Source Instability)**：在可解模型中，内部平衡点的 Jacobian 具有复共轭特征值且模长 $>1$，导致轨迹螺旋发散。
**无尺度梯度 (Scale-Free Gradient) $\bar{g}_t$**：定义为 $\bar{g}(u) = r \nabla_w \mathcal{L}(ru)$，仅依赖方向 $u$ 且切于单位球面。

## 可复现要素
- **数据集**：MNIST、CIFAR-10、WikiText、OpenWebText（均为公开数据集）
- **代码**：已开源，GitHub: https://github.com/shasanamin/normalized-optimization-dynamics
- **关键超参**：
  - MLP (MNIST): $\eta_{high}=0.5$, $\lambda=0.05$, batch=128, 120 步
  - ConvNet (CIFAR-10): $\eta_{high}=0.5$, $\lambda=0.05$, 10,000 步
  - GPT-2 (WikiText): $(\eta_{hi}, \eta_{lo}, \lambda) = (5\times10^{-5}, 5\times10^{-6}, 0.05)$, batch=32
  - GPT-2 (OpenWebText): $(\eta_{hi}, \eta_{lo}, \lambda) = (2.5\times10^{-4}, 10^{-5}, 0.05)$, batch=32
  - SGDM: $\mu=0.9$; Adam: $(\beta_1, \beta_2)=(0.9, 0.999)$
- **精度**：精确映射实验使用 float64，神经网络实验使用 float32
