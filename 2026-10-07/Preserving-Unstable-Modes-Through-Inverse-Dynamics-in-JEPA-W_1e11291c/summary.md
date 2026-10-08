---
title: "Preserving-Unstable-Modes-Through-Inverse-Dynamics-in-JEPA-W"
source: https://arxiv.org/pdf/2610.07540v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:55:59"
field: "视觉控制表示学习"
keywords: ["JEPA", "world models", "inverse dynamics", "controllability Gramian", "unstable subspace", "latent control", "representation learning", "feedback stabilization"]
innovations: ["揭示JEPA标准预测+SIGReg会选择性坍塌不稳定模态", "提出端点逆动力学损失EP-IDM保持有限视界可达子空间", "证明可控性Gram矩阵主特征子空间渐近收敛至不稳定子空间"]
benchmarks: ["CartPole LQR success rate", "Walker2D forward displacement", "PointMaze CEM success rate", "linear LDS stabilizability"]
---

# 论文速读：Preserving-Unstable-Modes-Through-Inverse-Dynamics-in-JEPA-W

## 一句话总结
本文揭示了 JEPA 世界模型的标准训练目标（下一步预测 + 防坍塌正则化）在控制开环不稳定系统时会**选择性地丢弃不可观测的失控模态**，导致从潜空间设计稳定反馈控制器失败；为此提出了**端点逆动力学损失（EP-IDM）**，通过从首尾潜状态重建整个动作序列，理论上证明了其可保持有限视界可达子空间并使可控性 Gram 矩阵主特征子空间收敛至不稳定子空间，实验上使 CartPole 上 latent-LQR 成功率从 **0% 提升至 100%**。

## 研究问题与动机
- **不稳定机器人的表示学习难题**：四旋翼、步行机器人等开环不稳定系统需要反馈控制器持续观测并纠正不稳定模态；但若视觉编码器将不稳定方向映射为零（即"塌陷"），控制器无法"看见"这些方向，稳定化变得不可能。
- **JEPA 预测目标不足**：既有工作（LeCun 2022 提出 JEPAs；SIGReg (Balestriero & LeCun, 2025) 防止整体坍塌）仅保证潜表示分布弥散，但**不能保证控制相关的物理方向（不稳定模态）被保留**；论文给出严格反例证明训练损失可取最小值而编码器把整个不稳定子空间放入核空间（Lemma 2）。
- **现有 JEPA 研究缺乏控制理论视角**：Ivashkov et al. (2026) 使用一步逆动力学防坍塌，但未研究**开环不稳定系统的反馈稳定化**；本文建立"预测损失 → 可检测性"的桥梁，提出端点逆动力学并给出完整的可控/可观理论分析。

## 核心贡献（创新点）
- **揭示 JEPA 标准训练的不稳定模态塌陷缺陷**：证明 next-step prediction + SIGReg 的最优解可以**完整丢弃不稳定方向**，而 latent LQR 仍无法稳定物理系统；与 Ivashkov et al. (2026) 的根本区别在于本文聚焦**反馈稳定化**而非仅防坍塌。
- **提出 EP-IDM（端点逆动力学损失）**：从首尾潜状态 $z_t, z_{t+H}$ 重建整条动作序列 $\mathbf{a}_{t:t+H-1}$；与一步/多步 IDM 的本质区别在于仅需两个端点编码即可约束**整个 H 步可达子空间**。
- **第一个理论定理：EP-IDM 保持可达子空间**（Theorem 1）：精确重建意味着 encoder 在有限视界可控子空间 $\mathcal{R}_H$ 上是单射，从而**不可丢弃任何可通过动作序列到达的方向**（含不稳定模态）；与已有逆动力学工作相比，本文给出**充分必要条件**并建立与可检测性的等价关系。
- **可控性 Gram 矩阵主特征子空间收敛定理**（Theorem 2）：随视界 $H \to \infty$，Gram 矩阵 $\Pi_H$ 的 top-$r$ 特征子空间收敛至可控不稳定子空间 $\Phi$，解释为何**不稳定方向是动作重建中最易辨识的**；这是连接表示学习与经典线性系统理论的关键结果。
- **多任务非线性视觉控制实证**：CartPole（LQR 成功 0% → 100%）、Walker2D（平均速度 0.08 m/s → 2.91 m/s，位移 0.06 m → 9.15 m）、PointMaze（规划成功率 ≥ 90%），展示 EP-IDM 在**不同控制范式**下的泛化性。

## 方法详解

**系统设定**（线性化后形式 Eq. 8）：
$$x_{t+1} = A x_t + B a_t, \quad y_t = C x_t$$
物理系统处于开环不稳定（$\rho(A) > 1$），编码器 $E_\theta(y) = W y + b$ 映射至 $d \ll n$ 维潜空间 $z_t$，预测器 $P_\theta(z, u) = A_z z + B_z u$ 建模潜动力学。

**标准 JEPA 训练目标**（Eq. 5-6）：
$$\mathcal{L}_{\text{pred}}(\theta) = \mathbb{E}\left[\frac{1}{H}\sum_{k=0}^{H-1}\|P_\theta(z_{t+k}, G_\theta(a_{t+k})) - z_{t+k+1}\|^2\right]$$
联合 SIGReg 正则化防止整体坍塌。

**EP-IDM 定义**（Eq. 14）：
$$\mathcal{L}_{\text{EP-IDM}}(\theta) = \mathbb{E}\left[\frac{1}{H}\sum_{t=0}^{H-1}\|[D_\theta(z_{t_\tau}, z_{t_\tau+H})]_t - a_{t_\tau+t}\|^2\right]$$
其中 $D_\theta: \mathcal{Z} \times \mathcal{Z} \to \mathcal{A}^H$ 为仿射逆动力学模型，仅接受首尾潜状态。

**关键理论**：
- Lemma 1：latent LQR 能稳定物理系统 $\iff (A, F)$ 可检测（$F=WC$），即所有 $|\lambda|\ge 1$ 的特征向量满足 $Fv\ne 0$。
- Lemma 2（反例构造）：存在 LDS 使 $\mathcal{L}_{\text{pred}}+\lambda_{\text{reg}}\mathcal{L}_{\text{SIG}}$ 取最小值时 $\Phi_{\ge 1}\subseteq\ker(F)$。
- Theorem 1：若 $\mathcal{L}_{\text{EP-IDM}}=0$ 且动作激励充分（conditional support 含开集），则 $\ker(F)\cap\mathcal{R}_H=\{0\}$，即 encoder 在 H 步可达子空间上是单射。
- Theorem 2：$\|\sin\Theta(\hat{\Phi}, \Phi)\|_2\to 0$ 当 $H\to\infty$，其中 $\hat{\Phi}$ 为 $\Pi_H$ 的 top-$r$ 特征子空间；由 Davis-Kahan 扰动定理证明，不稳定块 $\Pi_{u,H}$ 以 $\alpha^{2H}$ 指数增长而稳定块有界。

**架构**（Appendix A.3.1）：ViT-Tiny encoder (4 blocks, 192-dim) + causal transformer predictor (6 blocks, 16 heads) + MLP action decoders；训练 200 epochs、batch 64、AdamW lr $10^{-4}$ cosine schedule，latent 维度 $d=192$。

## 实验与结果

**数据集与基线**（MuJoCo，非公开自行生成）：
- **CartPole**：$128\times128$ 图像 + 4 维 proprioception + 1 维 action；787/99/99 轨迹；成功阈值 $\|x_T\|_2\le 0.7$。
- **Walker2D**：$64\times64$ + 17 维 + 6 维 action；3200/400/400 轨迹；由 SAC 策略与随机策略混合采集。
- **PointMaze**：$64\times64$ + 4 维 + 2 维 action；1600/200/200 轨迹。

**主要结果**：

| 任务 | 模型 | LQR SR | MFS | 最大提升 |
|---|---|---|---|---|
| CartPole | 1SP+SIG | **0%** | 0.202 | — |
| CartPole | **1SP+EP-IDM** | **100%** | **0.999** | **+100pp** |
| Walker2D | 1SP+SIG | 0.08 m/s | 0.06 m | — |
| Walker2D | **1SP+EP-IDM** | **2.91 m/s** | **9.15 m** | **+35× 速度** |
| PointMaze | MSP+EP-IDM | ≥90% (CEM) | — | 与 GT 相当 |

**对比基线**：GT (CartPole LQR 100%)、1SP/MSP+SIG、1SP+IDM+MS-IDM、DINO-WM、预训练 DINOv2/iBOT+投影器。

**诊断结论**：
- SIG 模型 latent LQR 失败但 CEM/GBP 仍可成功（非线性规划可补偿局部误差）；EP-IDM 使**局部线性化本身**具备稳定能力。
- 图 9 显示 EP-IDM 的吸引域覆盖几乎所有评估初始状态，Lyapunov 下降处处成立；SIG 仅在极小邻域内稳定。
- Walker2D phase portrait（图 5）：EP-IDM 重建极限环结构，SIG 完全丢失周期动力学。
- 预训练 encoder ablation（Table 5）：iBOT+PR-EP-IDM 达 100% LQR；DINOv2 因放大微小扰动（Table 14：增益 25× vs. iBOT 的 1×）而失败。

## 相关工作脉络
- **JEPAs / 联合嵌入预测架构**：LeCun (2022) 原始框架；SIGReg (Balestriero & LeCun, 2025) 防止整体坍塌的正则化；本文揭示 SIGReg 不能防止**选择性不稳定模态坍塌**。
- **World Models for Robotics**：DreamerV3 (Hafner et al., 2023) 等像素重建方法；JEPA 路线（Zhou et al. 2024 DINO-WM；Toso et al. 2026 invariant planning）关注规划几何，本文补充**稳定性维度**。
- **Inverse Dynamics Models**：Ivashkov et al. (2026) 一步逆动力学防坍塌；本文将其推广至**端点序列重建**并建立与可控性的严格联系。
- **Latent Feedback Stabilization**：Mhammedi et al. (2020) 从非线性观测学 LQR；Hu et al. (2022)、Werner & Peherstorfer (2024) 样本复杂度；Toso et al. (2025)、Lutkus et al. (2025) 不稳定子空间表示；本文**首次将表示坍塌与可检测性条件相连接**。
- **Controllability Gramian & Unstable Subspace**：经典线性系统理论（Hautus 1969 PBH 检验）；本文提出 Gram 矩阵主导特征子空间**渐近收敛至不稳定子空间**的新结果。
- **Anti-collapse Regularization**：VICReg (Bardes et al., 2021)、EMA (Assran et al., 2023)；SIGReg 为最新 JLepA 路线代表，本文证明其控制感知不足。

## 局限性与未来方向
- **理论假设偏理想**：Theorem 1 要求精确重建（$\mathcal{L}_{\text{EP-IDM}}=0$）与充分动作激励；实际中使用软损失，且非线性系统无严格保证。
- **线性分析外推未严格证明**：Theorem 1-2 针对线性时不变系统（LDS），实验显示非线性 CartPole/Walker2D/PointMaze 有效，但**非线性情形缺乏理论边界**。
- **依赖足够长视界**：Theorem 2 收敛率依赖于 $\alpha^{2H}$ 指数增长，实际短视界（本文 $H=3$）可能不足以充分暴露不稳定方向。
- **预训练 encoder 敏感性**：Table 5 显示 DINOv2 即使加 EP-IDM 也无法稳定（增益 25× 放大噪声），表明**表示质量上限受 backbone 制约**。
- **Future Work**（Section 7）：结合 bisimulation（Toso et al., 2026）与 temporal straightening（Wang et al., 2026a）等潜几何目标提升规划鲁棒性；扩展至连续高维机器人操作。

## 研究启发与可借鉴点
- **"控制感知表示"设计范式**：除预测损失外引入逆动力学目标可作为通用正则化手段，适用于任何需**从视觉观测做反馈控制**的 JEPA/DINO-WM 类工作。
- **理论与实验闭环**：先证明"预测目标不充分"（Lemma 2 反例），再提出 EP-IDM 并给出 Theorem 1-2 严格保证，最后用 CartPole/Walker2D 验证——这种**反例→构造→定理→实证**链条极具说服力，值得在后续工作中效仿。
- **可控性 Gram 矩阵的表示学习用途**：Theorem 2 揭示 Gram 矩阵主轴自然对齐不稳定模态，可启发新算法：在训练中途提取 $\Pi_H$ 的 top 特征向量作为不稳定方向近似，用于**自适应视界选择或注意力加权**。
- **消融指标设计**：除成功率外，论文引入 Lyapunov 下降图（Fig. 9）、phase portrait（Fig. 5）、encoder gain 表（Table 14）、fixed-point residual（Table 15）等多维度诊断，避免单一指标的 misleading，适合复现时参照。
- **跨任务迁移机会**：EP-IDM 在 stable（PointMaze）与 unstable（CartPole, Walker2D）任务均有效，可探索在**安全临界系统**（如医疗机器人、自动驾驶）中作为表示安全性的隐式保障。

## 关键术语表
- **JEPA（Joint-Embedding Predictive Architecture）**：LeCun 提出的自监督框架，学习潜表示并通过预测未来潜状态而非重建像素来训练世界模型。
- **SIGReg（Sketched Isotropic Gaussian Regularization）**：Balestriero & LeCun (2025) 提出的防坍塌正则化，通过特征函数匹配强制潜分布呈各向同性高斯。
- **EP-IDM（Endpoint Inverse Dynamics Model/Loss）**：本文提出的从首尾潜状态重建整条动作序列的损失函数，作为控制感知表示学习的核心正则项。
- **Controllability Gramian ($\Pi_H$)**：有限视界可控性 Gram 矩阵 $\sum_{k=0}^{H-1} A^k B B^\top (A^k)^\top$，衡量系统在 H 步内可被动作驱动的能量分布。
- **Detectability（可检测性）**：线性系统 $(A,M)$ 的性质的等价于所有不稳定/marginally stable 模态均可从观测 $q_t = Mx_t$ 中推断；由 PBH 秩条件刻画。
- **Unstable Subspace ($\Phi_{\ge 1}$)**：A 的特征值满足 $|\lambda|\ge 1$ 对应的不变子空间，反馈控制器必须"看见"该子空间才能稳定开环不稳定系统。
- **Latent LQR**：在 JEPA 潜空间线性化预测器后求解离散代数 Riccati 方程得到的线性二次反馈控制器，用于诊断表示是否保留不稳定方向。
- **Reachable Subspace ($\mathcal{R}_H$)**：H 步可控性矩阵 $\mathcal{C}_H = [A^{H-1}B \ \cdots \ B]$ 的列空间，即所有可通过动作序列到达的状态方向集合。

## 可复现要素
- **数据集**：论文未公开数据集；需在 MuJoCo 中按 Appendix A.3.2 参数（CartPole 787 轨迹、Walker2D 3200 轨迹、PointMaze 1600 轨迹）自行采集，使用 SAC/random 混合策略。
- **代码/权重**：论文标注 "Website Code"（URL 见作者主页），但 arXiv 版本未附 GitHub link；需联系作者或关注项目页面获取。
- **关键超参**（Table 7）：AdamW、lr $10^{-4}$ cosine → $10^{-6}$、weight decay $10^{-3}$、batch 64、epochs 200、warmup 5、gradient clip 1、latent $d=192$、预测视界 $H=3$。
- **架构细节**：ViT-Tiny encoder (4 blocks, emb 192, 3 heads) + causal transformer predictor (6 blocks, 16 heads, hidden 192, MLP 2048, dropout 0.1) + MLP action decoders (2层, width 256)。
- **环境版本**：MuJoCo 2.x（Todorov et al., 2012）；Walker2D-v4；CartPole 自定义渲染（固定相机 $128\times128$）；PointMaze U-shaped。
- **控制器实现**：LQR 通过自动微分线性化 + DARE 求解；CEM 采用 Pinneri et al. (2021) iCEM 变体；GBP 使用 Adam 50 步梯度优化。
