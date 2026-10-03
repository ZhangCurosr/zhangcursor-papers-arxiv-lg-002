---
title: "LEARNING-EXPRESSIVE-AND-COMPOSITIONAL-MO-TION-REPRESENTATION"
source: https://arxiv.org/pdf/2609.37677v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:52:53"
field: "人形机器人全身控制"
keywords: ["humanoid control", "spectral skills", "predictive representation", "skill composition", "hierarchical control", "motion tracking"]
innovations: ["提出谱技能表示，通过预测后续运动学习而非重建输入", "仿射条件设计使技能空间可组合，零样本组合新行为", "相比SONIC降低62%全局跟踪误差，训练效率提升12倍"]
benchmarks: ["BONES-SEED", "Unitree G1", "124-motion capability set", "4096-motion evaluation set"]
---

# 论文速读：LEARNING-EXPRESSIVE-AND-COMPOSITIONAL-MOTION-REPRESENTATION-VIA-SPECTRAL-SKILLS

## 一句话总结
本文提出"谱技能"（spectral skills）作为人机机器人分层控制中高层规划器与低层控制器之间的潜在接口表示，通过学习预测后续运动而非重建输入来获得表达能力和可组合性。在29-DoF Unitree G1上验证，相比SONIC方法将全局跟踪误差降低62%，且无需重新训练即可零样本组合出新行为。

## 研究问题与动机
1. **核心问题**：分层人形机器人控制中，高层规划器与低层全身控制器之间的命令表示应当如何设计？
2. **现有方法的不足**：显式接口（如关节空间目标或笛卡尔目标）会将机器人动力学复杂性暴露给高层模型，预测长时域轨迹困难，且错误会累积传递给低层控制器。
3. **潜在接口的局限**：已有潜在接口通常通过学习运动重建或后验推理获得，不清楚潜在表示应捕捉当前姿态还是后续运动，能否支持组合新行为。
4. **需求明确**：命令表示必须易于高层模型预测、具有足够表达能力支持多样全身运动，并理想地能从前序技能组合出新行为。

## 核心贡献（创新点）
1. **谱技能表示**：离线学习潜在技能空间，通过预测后续运动而非重建编码器输入获得，区别于VAE等重建式方法。
2. **表达性与效率**：条件化谱技能的低层控制器相比SOTA将全局跟踪误差降低62%（MPJPE-G从187.86降至70.51mm），训练量仅需12倍更少环境步数。
3. **可组合性**：冻结控制器可通过在谱方向上叠加偏移来零样本组合新技能，产生训练数据中未出现的行为组合，无需额外转换策略。
4. **语言条件规划集成**：作为语言规划器的输出空间时，成功率从77.1%提升至91.1%，优于显式轨迹预测。

## 方法详解
**整体框架**：分层半马尔可夫决策过程（SMDP），高层在宏观时间尺度选择技能，低层策略以50Hz执行。

**Encoder设计**：
- 将参考轨迹分为三个连续窗口：上下文$X_k$、执行段$B_k$、后续运动$Y_k$
- 确定性编码器将执行段映射为潜在技能：$z_k = E_\psi(B_k)$
- 编码器输入为10帧参考姿态（每帧38维），输出64维潜在表示

**Decoder设计**：
- 通过因子化高阶转移核预测后续运动：$\mathsf{P}(dY|X,z) \propto \exp(\langle \phi(X,z), \mu(Y) \rangle)\nu(dY)$
- 技能通过仿射映射$Az+b$线性进入预测器，这是可组合性的关键设计
- 采用Diffusion-based score estimation学习，噪声预测器参数化为：$D_\theta(X,z,Y^\tau,\tau) = M_\kappa(Y^\tau,\tau)^\top F_\beta(X)^\top(Az+b)$
- 联合训练目标为标准噪声预测损失：$\mathcal{L}_{pred} = \mathbb{E}[\|D_\theta - \epsilon\|_2^2]$

**低层跟踪**：
- 冻结编码器，使用PPO训练50Hz控制器$\pi_\varphi^{lo}(\cdot|o_t, z_t)$
- 动作空间为29维关节位置命令

**高层规划**：
- 使用GR00T action-head架构，条件化语言嵌入和历史观测
- 预测$N_{hi}=30$个连续技能命令，采用conditional flow-matching训练

**技能组合机制**：
- 单次去噪估计对技能的Jacobian精确可求：$J_\xi = -\frac{\sigma_\tau}{\alpha_\tau}M_\kappa(Y^\tau,\tau)^\top F_\beta(X)^\top A$
- **关键性质**：Jacobian与基础技能$z$无关，即$\widehat{Y}_\xi(z+\delta z) - \widehat{Y}_\xi(z) = J_\xi\delta z$
- 通过采样多个上下文计算响应Gram矩阵$C = \frac{1}{N}\sum J_{\xi_i}^\top J_{\xi_i}$，特征向量即为谱方向
- 组合操作：$z_t^{steer} = z_t^{base} + V_K\eta(t)$，各谱方向效应线性叠加

## 实验与结果
**数据集**：129,785条BONES-SEED clips（已公开），针对Unitree G1重定向。

**评估基准**：
- 124-motion能力集：涵盖运动、手势、舞蹈、跳跃/踢腿、受伤动作、地面动作、拳击
- 4096-motion评估集：经过运动学过滤后的子集

**主要结果（Table 1）**：
| 方法 | SR↑ | MPJPE-L↓(mm) | MPJPE-G↓(mm) |
|------|-----|---------------|---------------|
| SONIC | 100.00 | 23.79 | 173.92 |
| **Ours** | **100.00** | **18.22** | **65.06** |
| SONIC | 98.88 | 26.74 | 187.86 |
| **Ours** | **98.27** | **20.54** | **70.51** |

- 相比SONIC：MPJPE-L降低23%，MPJPE-G降低62-63%
- 训练效率：$5.0\times10^{10}$帧 vs SONIC的约$6.29\times10^{11}$帧（12倍更少）

**技能组合结果（Table 2）**：
- 右臂抬起方向：肩关节pitch ±0.90 rad，步行继续
- 转向方向：4秒内±130°转向，手臂几乎不变
- 高抬腿方向：swing apex从15cm提升至36cm
- 效果可叠加：两个方向同时作用时关节变化近似单个方向之和

**语言条件规划（Table 3）**：
| 接口 | SR↑ | MPJPE-L↓ | MPJPE-G↓ |
|------|-----|-----------|-----------|
| Skills (z→z), N=10 | **91.1%** | 42.83 | 230.2 |
| Re-encode (o→z), N=10 | 77.1% | 60.19 | 425.1 |
| Explicit (o→o), N=10 | 52.5% | 116.46 | 704.9 |

**硬件部署**：在Unitree G1真实机器人上验证了跟踪、技能链式切换和组合。

## 相关工作脉络
1. **SONIC (Luo et al., 2026)**：运动跟踪RL的SOTA方法，使用tokenized latent representation，本文方法在全局跟踪误差上显著超越。
2. **BFM-Zero (Li et al., 2026)**：基于prompt的behavior foundation model，依赖运动库中的帧，而谱方向从训练模型直接提取，具有跨技能空间的泛化能力。
3. **Spectral Representation RL (Gao et al., 2025)**：通过状态-动作转移核的低秩分解获得充分表示，本文将其扩展到高层semi-MDP的转移因子化。
4. **DiffSR (Shribak et al., 2024)**：基于diffusion的energy-based模型学习因子化转移，本文继承其训练范式应用于机器人技能学习。
5. **CAHM/Calm (Tessler et al., 2023)**：条件对抗潜在模型，用于虚拟角色控制，但未探索技能组合性。
6. **Successor Features (Dayan, 1993; Barreto et al., 2017)**：因子化累积状态访问算子，本文在其基础上提升一级，因子化skill驱动的motion转移。

## 局限性与未来方向
1. **单步预测的线性假设**：Proposition 3.1的精确响应仅适用于单次去噪估计，完整diffusion sampler和机器人执行并不全局线性。
2. **谱方向的解释性限制**：虽然部分方向对应可识别的运动变化（如抬臂、转向），但 eigenvalue 仅按预测变化幅度排序，不直接对应"有用性"，需人工检验。
3. **数据依赖性**：谱方向从训练数据的响应Gram中提取，若数据覆盖不足（如LAFAN1实验所示），某些方向效果会减弱。
4. **未探索的场景**：论文未涉及动态环境交互、力控任务或外部扰动下的组合鲁棒性。
5. **未来方向**：可扩展至多模态输入（视觉+语言）、在线适应、以及更复杂的组合操作（如连续参数的精细控制）。

## 研究启发与可借鉴点
1. **预测式表示学习**：通过预测未来而非重建输入来获得更具动力学前瞻性的表示，这一设计原则可迁移到其他控制表征学习任务。
2. **仿射条件设计**：技能通过$Az+b$线性进入预测器是实现可组合性的关键，这一结构约束值得在其他latent action model中探索。
3. **响应Gram矩阵**：通过多上下文平均Jacobian构造响应Gram来提取谱方向，是一种无需标签的方向发现方法，可应用于其他潜在空间分析。
4. **分层接口设计**：明确区分高层"做什么"（技能选择）和低层"怎么做"（跟踪控制），并通过紧凑潜在表示连接，这一架构设计可推广至Manipulation等任务。
5. **零样本组合验证**：通过t-SNE可视化组合结果在训练分布外的距离变化，提供了组合能力的直观定量评估方法。

## 关键术语表
**Spectral Skills（谱技能）**：通过因子化运动转移概率学习得到的低维潜在表示，编码运动片段的动态后果而非静态姿态。

**Predictive Representation（预测表示）**：通过预测后续状态而非重建输入获得的表示，保留对未来运动有预测信息的维度。

**Response Gram Matrix（响应Gram矩阵）**：跨多个上下文的Jacobian矩阵乘积的平均，用于提取对运动变化最敏感的谱方向。

**Diffusion-based Score Estimation**：基于扩散模型的分数估计方法，通过噪声预测网络学习能量模型的梯度。

**Skill Chaining（技能链式切换）**：在运行中直接切换技能编码，使机器人平滑过渡到新行为而无需额外转换策略。

**Spectral Steering（谱导向）**：在基础技能上沿谱方向叠加偏移来实现运动属性的独立调节。

**MPJPE-L/G**：Local/Global Mean Per-Joint Position Error，分别衡量局部姿态精度和全局路径跟踪误差（mm）。

**Semi-MDP（半马尔可夫决策过程）**：允许动作执行时间随机的MDP扩展，适合建模宏观/微观双时间尺度的分层控制。

## 可复现要素
- **数据集**：BONES-SEED（已公开，129,785条 clips）
- **代码**：论文声明" Upon publication"开源训练和评估代码
- **权重**：未提及是否提前公开
- **关键超参**：
  - 技能维度 $d = 64$
  - 预测器宽度 $(e, r) = (1024, 256)$
  - 编码器窗口 $H = 10$ 帧
  - 上下文窗口 $P = 6$ 帧
  - 预测 horizon $L = 10$ 帧
  - 低层控制器频率：50 Hz
  - 规划器 replanning interval：$N \in \{10, 30\}$
  - 训练预算：$5.0\times10^{10}$ 环境帧（单H200 GPU，约144 GPU-hours）
