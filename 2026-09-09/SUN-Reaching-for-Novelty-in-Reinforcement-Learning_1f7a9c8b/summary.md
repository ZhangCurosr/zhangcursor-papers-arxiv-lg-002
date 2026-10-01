---
title: "SUN-Reaching-for-Novelty-in-Reinforcement-Learning"
source: https://arxiv.org/pdf/2609.08642v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:02:12"
field: "强化学习探索与目标条件RL"
keywords: ["reinforcement learning", "exploration", "goal-conditioned RL", "successor value function", "intrinsic motivation", "pseudocount", "novelty"]
innovations: ["提出乘法形式的SUN指示器将SVF可达性与伪计数新颖性统一，无需人工平衡超参", "设计成本分摊的轻量级伪计数估算器实现O(1)查询", "提出自适应目标选择策略，基于SVF单调性检测实现稳定且灵活的重选"]
benchmarks: ["ThreeRoom", "FourRoomStuck", "GridMaze", "MountainCar", "CartPole", "LunarLander", "Pendulum", "Acrobot", "PointMaze-S/H", "AntMaze-S/H", "ArmPush-H"]
---

# 论文速读：SUN: Reaching for Novelty in Reinforcement Learning

## 一句话总结
论文提出了 SUN（SUccessor-to-Novelty），一个基于后继值函数（SVF）与轻量级伪计数的可达性感知目标选择框架，在强化学习探索中将"新颖性"和"可达性"统一为单一乘法指标，在包含不可达状态、不可逆转移的新旧基准上均显著超越 SOTA 方法（AdaGoal、DISCOVER）。

## 研究问题与动机
- **新颖性与可达性的割裂**：现有目标条件 RL（GCRL）探索方法要么仅基于密度/访问频次选择最新颖目标（如 MEGA、Skew-Fit、GoalGAN），容易选中不可达目标；要么仅基于距离或成功概率优化可达性，忽略罕见但可达的目标。二者均无法同时兼顾。
- **已有联合方法的缺陷**：AdaGoal 和 DISCOVER 同样使用 SVF 驱动探索，但通过 critic 集成分歧来估算"新颖性"，计算开销大（需 K 个 critic），且在新颖性与可达性之间校准不佳——AdaGoal 偏向新颖性、DISCOVER 偏向可达性。
- **目标选择策略的僵化**：既有方法多为 episodic（每回合仅选择一次目标），在随机转移或不可达状态场景下会持续追随无效目标，导致探索停滞。
- **伪计数查询成本过高**：KDE 等经典密度估计方法需在每次目标查询时遍历缓冲区，难以支持 per-step 重选策略。

## 核心贡献（创新点）
1. **提出 SUN 指示器**：将 SVF 估计的可达性与伪计数估计的新颖性以乘法形式结合（$\text{SUN}(g|s) = V^{\pi}(s,g) \cdot \nu(g)$），无需人工调节两个信号的平衡系数。与已有工作的本质区别在于：乘性形式天然共享"零"语义——不可达或已饱和目标任一端为零即被拒绝，而 ADD UCB 类方法需额外调参。
2. **轻量化伪计数估计器**：将固定半径最近邻密度估计的计算成本分摊到缓冲区插入时（O(1) 查询成本），相比 KDE 的 O(N·M·B) 每步查询，支持高频目标重选。
3. **严格的理论性质**：证明 SUN 等价于 count-bonus 奖励的 Value Function；给出短视域击中概率的下界 $\Pr[\tau_g \leq n] \geq V^{\pi}(s,g) - \gamma^{n+1}$；证明不可达目标被严格拒绝（SUN = 0）。
4. **自适应目标选择策略**：在每步比较当前 SVF 值与选择时刻的值，若下降则从新候选集重选，兼具 episodic 稳定性与 per-step 弹性，论文证明了其理论一致性（确定性最优策略下单调性保证检查永不触发）。
5. **引入含不可达状态/不可逆转移的新型基准环境**（ThreeRoom、FourRoomStuck、GridMaze），填补了现有 GCRL 基准缺乏硬探索压力测试的空白。

## 方法详解
- **SUN 指示器（核心公式）**：$\text{SUN}(g|s) = V^{\pi}(s,g) \cdot \nu(g)$，其中 $V^{\pi}(s,g)$ 为后继值函数，估计从状态 $s$ 到达目标 $g$ 的累积折扣命中次数；$\nu(g) = 1/n_g$ 为新颖性信号，$n_g$ 为伪计数。目标选择：$g_t = \arg\max_{g \in \mathcal{C}_t} \text{SUN}(g|s_t)$，候选集 $\mathcal{C}_t$ 从 replay buffer 采样。

- **理论性质**：
  - **Count-bonus 等价性**：$\text{SUN}(g|s) = \mathbb{E}[\sum_{k=t}^{\infty} \gamma^{k-t} \mathbf{1}\{s_k=g\}/n_g]$，即 SUN 是 count-bonus 探索目标的精确 Value Function，而非启发式组合。
  - **短视域击中概率下界**：$\Pr_{\pi}[\tau_g \leq n] \geq V^{\pi}(s,g) - \gamma^{n+1}$，高 SVF 值保证短视域内以高概率到达。
  - **不可达目标抑制**：若目标在策略类中不可达，则 $V^{\pi}(s,g)=0$，SUN 严格为零。

- **对数空间分解**：$\log \text{SUN}(g|s) = \log V^{\pi}(s,g) - \Phi(g)$，其中 $\Phi(g)=\log \mu(g)$ 为 log-rarity potentia。距离代价由折扣因子 $\gamma$ 隐式决定（$\kappa=-\log\gamma$），无需引入额外超参。

- **自适应目标选择**：保存选择时刻状态 $s_{t_{\text{sel}}}$，每步检查 $V^{\theta}(s_t, g_t) < V^{\theta}(s_{t_{\text{sel}}}, g_t)$，若成立则从新候选集重选；否则保持当前目标。确定性最优策略下值函数单调不降，检查永不触发，保证理论一致性。

- **轻量化伪计数**：缓冲区每条样本存储在其标准化特征空间半径 $\rho$ 内的邻居数量 $n_i$；新样本插入时计算邻居数并递增邻居计数。查询时 $\nu(g)=1/n_g$ 直接读取，O(1) 成本；插入成本 O(N·M)。标准化防止不同特征尺度导致的偏差（用各特征标准差的中位数下界 floor）。

- **SVF 训练**：使用 HER（Hindsight Experience Replay）"future" relabeling 训练，提供正负样本平衡；结合 DQN（离散动作）或 TD3（连续动作）进行 off-policy 学习。

## 实验与结果
- **评测环境**：
  - Gridworlds（新型）：ThreeRoom（不均匀初始分布+不可达房间）、FourRoomStuck（不可逆陷阱+随机转移）、GridMaze（狭窄瓶颈）。
  - Classic Control：MountainCar、CartPole、LunarLander、LunarLander(Full)、Pendulum、Acrobot。
  - GCRL Control：PointMaze-S/H、AntMaze-S/H、ArmPush-H。

- **基线**：Random、AdaGoal（SVF 集成+分歧新颖性）、DISCOVER（SVF 集成+UCB 型新颖性）。

- **评估指标**：Coverage（访问过的目标比例）和 Shannon Entropy（归一化至 [0,1] 的访问均匀性）；高维空间（LunarLander(Full)）使用 Kozachenko-Leonenko k-NN 微分熵估计。

- **主要结果**：
  - SUN（Adaptive, Multiplicative）在所有环境的 Coverage 和 Entropy 上均优于所有基线。
  - 相对 AUC 提升（Figure 8）：Coverage 较 AdaGoal +15.1%、较 DISCOVER +21.0%、较 Random +67.1%；Entropy 较 AdaGoal +7.6%、较 DISCOVER +9.2%、较 Random +26.5%。
  - Gridworlds 中所有方法均达到完全 Coverage，但 SUN 的 Entropy 显著领先，揭示了非均匀初始分布和不可达目标的真实难度。
  - 在 LunarLander(Full)（8 维目标空间）中表现最佳，验证了轻量级伪计数在高维下的有效性。
  - 消融实验：仅用新颖性→网格世界中失败；仅用可达性→几乎无探索；加法指标（Additive）在 MountainCar 中出现饱和现象（Entropy 下降），乘性指标无此问题。

- **效率**：SUN 在 Gridworlds 和 Classic Control 上比 AdaGoal/DISCOVER 平均快 3.4×–3.5×；在 GCRL 环境上与基线相当。

## 相关工作脉络
1. **AdaGoal (Tarbouriech et al., 2022)**：同样使用 SVF 驱动探索并平衡新颖性与可达性，但新颖性由 critic 集成分歧估计，计算成本高；SUN 用轻量伪计数替代，且引入自适应重选策略。
2. **DISCOVER (Diaz-Bone et al., 2025)**：与 AdaGoal 类似的结构（SVF 集成+UCB 不确定性），但将分歧视为"新颖性"；实验中 DISCOVER 明显偏向可达性（$\beta$ 敏感），SUN 无需调参即可自动平衡。
3. **Proto-Goals (Bagaria et al., 2023)**：顺序式组合新颖性与可达性（先按计数采样候选，再选 SVF 最高的）；SUN 是联合乘法评分，不存在顺序偏置问题，且支持步间自适应重选。
4. **NGU (Badia et al., 2020)**：也使用最近邻密度估计估计新颖性，但针对 within-episode 新颖性且每次 episode 清空内存；SUN 针对 lifetime 新颖性且持久存储。
5. **MaxEnt 探索 (Hazan et al., 2019; Lee et al., 2019)**：以最大化状态访问分布熵为目标；SUN 的乘性指标在 log 空间等价于稀有度奖励减去距离代价的软 Lagrangian，为实际可执行的近似。
6. **后继表示/特征 (Dayan, 1993; Kulkarni et al., 2016)**：SUN 的理论基础，利用 SVF 作为可达性代理；不同于纯预测误差驱动的 curiosity（Pathak et al., 2017），SUN 的可达性信号有明确的概率下界保证。

## 局限性与未来方向
- **图像观测不适用**：count-based 新颖性在连续高维观测空间（如图像）中失效——几乎所有状态都是唯一的，计数趋近均匀，无法提供有效新颖性信号。未来可与 learned curiosity 或轨迹-潜变量互信息结合。
- **路径累积新颖性缺失**：当前 SUN 仅基于目标本身的新颖性评分，未考虑通往目标路径上的中间状态新颖性；长路径经过更多新颖状态可能优于短路径到边际罕见目标。扩展为累计形式可更好匹配最大熵目标。
- **候选目标限于缓冲区已见状态**：只能从 replay buffer 中采样已遇到过的状态作为目标，受限于"缓冲区流形"。可通过学习 goal-space 度量（quasimetric value functions、Laplacian representations）或 OOD 泛化训练来实现连续度量空间中的目标生成。
- **理论样本复杂度未完全证明**：附录 K 给出了 SUN-UCB 的 PAC 框架下的部分分析，但缺少 frontier-expansion 机制的保证和政策最优性闭环。

## 研究启发与可借鉴点
1. **乘性指标设计的优雅性**：SUN 的 $V \cdot \nu$ 形式通过共享零点和尺度消除了超参调节需求，这一设计思路可迁移到其他需要平衡多个信号的 RL 模块（如 skill 选择、子目标生成）。
2. **成本分摊的伪计数工程技巧**：将密度估计的计算从查询时转移到插入时，并配合标准化处理，是处理 continuous space 探索任务中高频查询需求的实用范式，可直接复用到其他 count-based 探索方法。
3. **自适应目标重选的策略设计**：通过值函数单调性检测"探索失败信号"触发重选，兼具稳定性和灵活性。可借鉴用于其他 goal-conditioned 系统的动态策略切换机制。
4. **Benchmark 设计启示**：引入含不可达区域、不可逆转移、窄通道的 Gridworld 作为 GCRL 探索的"压力测试床"，比单纯使用连续控制任务更能揭示方法间的本质差异，值得在新探索方法评测中采用。
5. **与 OOD 泛化的结合机会**：SUN 作者指出可结合 Laplacian representations 或 quasimetric 突破缓冲区流形限制，这一方向与本团队可能关注的"离线/在线混合探索"或"未见目标泛化"有直接交叉点。

## 关键术语表
**SUN (SUccessor-to-Novelty)**：一种基于后继值函数和伪计数的乘法目标选择指示器，同时编码可达性和新颖性。
**Successor Value Function (SVF)**：广义价值函数，表示在策略 $\pi$ 下从状态 $s$ 出发累积折扣命中目标 $g$ 的期望次数，编码可达性信息。
**Goal-Conditioned RL (GCRL)**：学习条件于目标 $g$ 的策略 $\pi(a|s,g)$ 的 RL 框架，目标既可用于任务执行也可用于驱动探索。
**Pseudocount**：对连续空间中目标访问频次的近似估计，本文采用固定半径最近邻并在缓冲区插入时预计算，实现 O(1) 查询。
**Adaptive Goal-Selection**：在每步比较当前 SVF 值与选择时刻值的策略，若值下降则从新候选集重选目标，兼具 episodic 稳定与 per-step 弹性。
**Count-Bonus Reward**：与目标访问频次成反比的内在奖励 $r = \mathbf{1}\{s_t=g\}/n_g$，SUN 是其对应的 Value Function。
**Shannon Entropy (of goal visitation)**：归一化至 [0,1] 的目标访问频次分布熵，衡量探索的均匀性，与 Coverage 互补。
**Hindsight Experience Replay (HER)**：将轨迹中已访问状态重标记为目标以生成正样本的训练技巧，用于平衡 SVF 训练的正负样本。

## 可复现要素
- **数据集/环境**：Classic Control 和 GCRL Control 使用开源基准（Gymnasium、Bortkiewicz et al. 2025）；Gridworlds 为本论文新增，作者声明"Source code available at link soon"。
- **代码开源**：论文声明"link soon"（即将开源），目前未正式发布。
- **权重**：论文未提及预训练权重发布。
- **关键超参**：折扣因子 $\gamma=0.99$，TD($\lambda$) trace factor $\lambda=0.95$，伪计数半径 $\rho=0.1$（PointMaze-H 用 0.5，ArmPush-H 用 0.01），DISCOVER 的 $\beta=10$（Gridworlds/Classic Control）或 $\beta=1$（GCRL Control），目标到达阈值 $\eta$ 复用伪计数半径，minibatch size=16（DQN）/ 256（TD3），学习率 $10^{-3}$（AdamW）。
