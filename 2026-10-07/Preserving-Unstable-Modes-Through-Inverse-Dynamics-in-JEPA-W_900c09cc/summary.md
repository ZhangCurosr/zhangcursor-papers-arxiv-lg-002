---
title: "Preserving-Unstable-Modes-Through-Inverse-Dynamics-in-JEPA-W"
source: https://arxiv.org/pdf/2610.07540v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:13:33"
field: "机器人学习与世界模型"
keywords: ["JEPA", "世界模型", "逆动力学", "不稳定模态", "视觉控制", "表征学习", "可控性"]
innovations: ["揭示JEPA标准训练无法保证保留控制所需的不稳定模态", "提出EP-IDM端点逆动力学损失并证明其保留可达子空间的理论保证"]
benchmarks: ["CartPole", "Walker2D", "PointMaze"]
---

# 论文速读：Preserving-Unstable-Modes-Through-Inverse-Dynamics-in-JEPA-W

## 一句话总结
本文揭示了标准JEPA世界模型（次步预测+防坍塌正则化如SIGReg）无法保证保留反馈控制所需的不稳定模式，并提出EP-IDM（端点逆动力学）损失，通过从首尾潜状态重建动作序列来强制保留可控不稳定方向，在CartPole等任务上将LQR成功率从0%提升至100%。

---

## 研究问题与动机

1. **不稳定系统控制的表征需求**：无人机、腿式机器人等开环不稳定系统需要反馈控制器实时观测并纠正不稳定模态；若编码器丢弃这些方向，控制器无法稳定系统。

2. **JEPA训练的盲区**：标准JEPA目标（预测下一步潜状态+SIGReg防坍塌）可被最小化，同时编码器将所有不稳定模态映射为零——损失函数不保证"可观测到所有需要控制的方向"。

3. **理论缺口**：已有工作证明JEPA可学到预测性好的表示，但未建立预测准确性与可镇定条件（detectability）之间的等价关系；本文填补此空白。

4. **实际应用痛点**：部署JEPA于不稳定机器人时，表征可能"看起来非退化"但实际丧失控制能力，导致规划/控制失败。

---

## 核心贡献（创新点）

1. **揭示标准JEPA的不稳定性陷阱**：证明存在开环不稳定线性系统，使1SP/MSP+SIGReg同时达到最小损失，但编码器核包含整个不稳定子空间（Lemma 2）——这是理论层面的第一个反例构造。

2. **提出EP-IDM逆动力学损失**：从初始与终端潜状态重建完整动作序列，而非逐帧重建；该设计直接关联有限视界可达子空间的可保留性（Theorem 1）。

3. **建立可控Gram矩阵与不稳定子空间的渐近收敛**：证明当预测视界H→∞时，有限视界可控Gram矩阵的主特征空间收敛到可控不稳定子空间（Theorem 2），为逆动力学提供了几何解释。

4. **视觉控制任务的实证验证**：在非线性CartPole（LQR成功率0%→100%）、Walker2D（平均速度2.91 m/s vs SIGReg的0.08 m/s）、PointMaze上验证EP-IDM的泛化能力，且与预训练编码器（iBOT）结合效果显著。

---

## 方法详解

### 系统设定
- 物理系统：$x_{t+1} = f(x_t, a_t)$，观测 $y_t = g(x_t)$
- 编码器 $E_\theta: \mathcal{Y} \to \mathcal{Z}$，动作编码器 $G_\theta: \mathcal{A} \to \mathcal{U}_\theta$，潜动力学预测器 $P_\theta: \mathcal{Z} \times \mathcal{U} \to \mathcal{Z}$
- 预测目标：$\min_\theta \mathcal{L}_{pred}(\theta) + \lambda_{reg} \mathcal{L}_{reg}(\theta)$，其中$\mathcal{L}_{pred}$为次步或等多步预测误差

### EP-IDM损失设计
$$\mathcal{L}_{EP-IDM}(\theta) = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\frac{1}{H}\sum_{t=0}^{H-1} \left\| [D_\theta(z_{t_\tau}, z_{t_\tau+H})]_t - a_{t_\tau+t} \right\|_2^2\right]$$

- 逆动力学模型 $D_\theta: \mathcal{Z} \times \mathcal{Z} \to \mathcal{A}^H$ 为仿射映射：$D_\theta(z_0, z_H) = W_H[z_0 \ z_H] + b_H$
- 联合优化编码器、预测器、逆动力学模型，用$\mathcal{L}_{EP-IDM}$替代$\mathcal{L}_{reg}$

### 理论保证
- **Theorem 1**（可达方向保留）：若动作激发足够丰富（条件支撑含$\mathbb{R}^{mH}$的非空开集），且$\mathcal{L}_{EP-IDM}=0$，则$\ker(F) \cap \mathcal{R}_H = \{0\}$，即编码器在H步可达子空间上单射
- **Theorem 2**（不稳定子空间收敛）：可控Gram矩阵$\Pi_H$的主$r$维特征空间$\hat{\Phi}$满足$\|\sin\Theta(\hat{\Phi}, \Phi)\|_2 \to 0$（$H \to \infty$）

---

## 实验与结果

### 数据集与环境
- **MuJoCo物理引擎**上的三个视觉控制任务：CartPole（倒立摆平衡）、Walker2D（双足行走）、PointMaze（U型迷宫导航）
- CartPole：787条训练轨迹（75%被动/随机输入+25% LQR策略），128×128图像+4维本体感知
- Walker2D：3,200条训练轨迹（随机+SAC策略混合），64×64图像+17维本体感知
- PointMaze：1,600条训练轨迹，64×64图像+4维本体感知

### 主要结果
| 任务 | 基线 | EP-IDM改进 | 关键指标 |
|------|------|------------|----------|
| CartPole LQR | 1SP+SIG: **0%** / MSP+SIG: **0%** | 1SP+EP-IDM: **100%** | MFS从0.202→0.999 |
| Walker2D | 1SP+SIG: 0.08 m/s | 1SP+EP-IDM: **2.91 m/s** | 接近GT的3.61 m/s |
| PointMaze CEM | MSP+SIG: 90% | 1SP+EP-IDM: **100%** | 稳定导航 |

- **消融实验**：EP-IDM与iBOT预训练编码器结合实现100% LQR成功率；DINOv2因放大小扰动（增益25×）而失败
- **诊断分析**：EP-IDM模型恢复的真实向量场与GT接近，Lyapunov函数单调递减

---

## 相关工作脉络

1. **JEPA系列工作**：LeCun (2022)原始框架；Assran et al. (2023)SIG对比学习；Balestriero & LeCun (2025)SIGReg防坍塌——本文指出这些方法不保证控制信息保留。

2. **逆动力学世界模型**：Ivashkov et al. (2026)使用单步逆动力学防坍塌——本文与其共享"动作提供控制相关信号"理念，但研究**反馈镇定**而非仅防坍塌，且理论证明更完整。

3. **视觉控制理论**：Mhammedi et al. (2020)未知非线性观测下的线性系统镇定；Hu et al. (2022)镇定样本复杂度——本文连接表征学习与可镇定条件(detectability)。

4. **不稳定子空间学习**：Toso et al. (2025)、Lutkus et al. (2025)学习低维不稳定表示——本文回答"JEPA能否自发学到这些表示"这一更基础问题。

---

## 局限性与未来方向

1. **线性理论的推广限制**：Theorem 1-2严格针对线性系统；非线性视觉任务（CartPole等）的实验成功是经验性的，缺乏理论保证。

2. **精确重建的理想化假设**：Theorem 1要求$\mathcal{L}_{EP-IDM}=0$，实际中仅作为软归纳偏置使用，收敛到零的条件未明确。

3. **动作激发的数据依赖**：定理要求条件支撑含开集，实际数据中若策略过于保守（如始终在平衡点附近），不稳定方向可能未被充分激发。

4. **未来方向**：结合双模拟（bisimulation）和时序拉直（temporal straightening）提升规划效率；探索与 latent geometry 目标的联合优化。

---

## 研究启发与可借鉴点

1. **控制理论×表征学习的交叉验证**：将detectability条件形式化为表征学习的理论约束，为JEPA类模型提供了可验证的稳定保障框架。

2. **EP-IDM的架构简洁性**：仅需附加一个从首尾潜状态到动作序列的仿射解码器，无需额外网络深度，工程成本低。

3. **预训练编码器的选择敏感性**：iBOT因平滑表示适合控制，DINOv2因高敏感放大扰动而失败——提示控制任务需选择"物理连续"的预训练模型。

4. **多控制器诊断层次**：LQR（局部线性）、CEM/GBP（非线性规划）的组合评估揭示了不同任务对表征精度的需求差异。

---

## 关键术语表

**JEPA**（Joint-embedding Predictive Architecture）：LeCun提出的自监督框架，通过预测潜表示而非像素重建来学习表征。

**SIGReg**：Sketched Isotropic Gaussian Regularization，通过匹配特征函数防止表征总坍塌的正则化技术。

**EP-IDM**（Endpoint Inverse Dynamics Model）：本文提出的从初始和终端潜状态重建完整动作序列的逆动力学损失。

**Detectability**：系统可检测性，指所有不稳定/边缘稳定模态可通过观测恢复；是设计镇定控制器的充要条件。

**Controllability Gramian**：可控Gram矩阵，刻画系统从初始状态到达目标状态的控制能量；其主特征空间渐近收敛到不稳定子空间。

**Open-loop unstable**：开环不稳定，指无反馈时系统轨迹发散；如倒立摆直立平衡。

**Finite-horizon reachable subspace**：有限视界可达子空间，由H步内所有可能动作序列生成的状态方向集合。

---

## 可复现要素

- **代码/权重**：论文标注"Website Code"，应可公开获取（具体链接见原文）
- **数据集**：MuJoCo环境数据，非公开但可使用标准采集脚本复现
- **关键超参**：
  - 编码器：ViT-Tiny（4层transformer，192维）或预训练DINOv2/iBOT
  - 预测器：6层causal transformer，16 heads，隐藏维度192
  - 优化：AdamW，初始lr=1e-4，warmup 5 epochs，cosine schedule
  - 批次大小64，epochs 200，预测视界H=3，latent维度192
  - 损失权重均为1

---
