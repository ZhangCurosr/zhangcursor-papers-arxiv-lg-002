---
title: "Learning-Infinite-Horizon-Average-Reward-CMDPs-via-State-Aug"
source: https://arxiv.org/pdf/2609.39093v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 21:58:13"
field: "约束强化学习理论"
keywords: ["平均奖励 CMDP", "状态增广", "Huber 势", "遗憾分析", "弱通信 MDP", "价值截断", "on-policy learning"]
innovations: ["将累积约束违反编码为安全状态并通过 Huber 势设计有界重塑奖励，避免对偶乘子变化导致的值函数失控", "首次实现弱通信表格平均奖励 CMDP 下计算高效的 Õ(√T) 遗憾与约束违反联合上界", "将乐观价值迭代与价值截断技术推广至增广 CMDP 设定，证明 clip 后仍保留 optimism"]
benchmarks: ["Tabular CMDP (modified from Yu et al. 2026, H_env=6, m=2, T=32000)"]
---

# 论文速读：Learning-Infinite-Horizon-Average-Reward-CMDPs-via-State-Aug

## 一句话总结
本文提出了 SA-CVI-UCB 算法，首次在表格型弱通信（weakly communicating）平均奖励约束 MDP（CMDP）设定下实现了计算高效的 Õ(√T) 遗憾和累积约束违反上界。核心思路是将累积约束违反状态化，并通过 Huber 势函数设计重塑奖励，从而避免对偶乘子引入的额外值函数变化。

## 研究问题与动机
1. **核心问题**：在弱通信假设下学习无限 horizon 平均奖励 CMDP 时，如何在保持计算效率的同时实现 Õ(√T) 遗憾和约束违反上界？
2. **已有工作的不足**：现有保证计算高效（computationally efficient）的算法仅有 Õ(T^(2/3)) 或 Õ(T^(3/4)) 界（如 Chen et al. 2022 的 Alg. 4 达到 Õ(√T) 但需要求解非凸规划；Ghosh et al. 2023 线性设定下高效算法为 Õ(T^(3/4))）；基于对偶乘子的 primal-dual 方法因每步更新乘子导致复合奖励剧烈变化，使得平均奖励设定下的 span 控制难以直接推广。
3. **弱通信 vs 更严格假设**：区别于 ergodic/unichain 假设（已有 Õ(√T) 结果），弱通信假设允许多个递归类和瞬态状态共存，分析难度显著增加，且最优稳态策略未必存在。
4. **动机**：已有结果要么要求计算低效的规划（如非线性规划），要么遗憾率次优；本文旨在填补这一空白，给出首个计算高效且达到最优 Õ(√T) 率的算法。

## 核心贡献（创新点）
1. **首次提出计算高效的 Õ(√T) 算法**：SA-CVI-UCB 是首个在表格型弱通信平均奖励 CMDP 下同时实现 Õ(√T) 遗憾与累积约束违反（高概率）的计算高效算法，运行时间为 O(S²AT^(5/2))。与 Chen et al. (2022) 的 Õ(√T) 非凸规划方法本质不同，本文算法仅需向后价值迭代。
2. **状态增广框架的创新设计**：将累积约束违反信息编码进安全状态 z，而非通过改变奖励函数（对偶乘子），使得重塑奖励 r̃ 在增广状态空间上保持固定，从而避免了对偶乘子更新带来的额外值函数波动。与 Sootla et al. (2022b) 的几乎必然安全设定不同，本文的 r̃ 始终有界，适配 regret 分析。
3. **Huber 势函数重塑奖励机制**：利用 Huber 势 Φ_W 的有界斜率（全局 Lipschitz）和 telescoping 性质，使重塑奖励 r̃(s,z,a) = r(s,a) + Φ_W(z) − Φ_W(ψ(s,z,a)) 满足逐步入界，同时保证累积重塑奖励与原始累积奖励之差仅为终端势值 Φ_W(z_{T+1})。
4. **结合有限 horizon 近似与价值截断**：将无约束平均奖励 MDP 中的 optimistic value iteration with clipping 技术推广至增广 CMDP 设定，通过 Lemma 3 证明增广 MDP 中基准价值函数的 span 有界（≤ 2C），从而保留乐观性并控制统计误差。

## 方法详解
1. **增广 MDP 构建**：定义安全状态 z_t ∈ Z = {Δn : n ∈ Z, |Δn| ≤ 2T}，其中 Δ = 1/√T，演化规则为 z_{t+1} = Π_Z(z_t − g(s_t, a_t))（确定性转移）。增广状态空间 S̃ = S × Z，转移核 P̃(s', z'|s, z, a) = P(s'|s, a) · 1{z' = ψ(s, z, a)}。
2. **Huber 势与重塑奖励**：Φ_W(x) 定义为分段函数——x ≤ 0 时为 0，0 < x ≤ W 时为 Λx²/(2W)，x > W 时为 Λ(x − W/2)，参数 Λ = 2/γ，W = √T。重塑奖励 r̃(s,z,a) = r(s,a) + Φ_W(z) − Φ_W(ψ(s,z,a))，其 telescoping 性质保证 Σ_t r̃_t = Σ_t r_t − Φ_W(z_{T+1})。
3. **SA-CVI-UCB 算法流程**：采用 K 个 episode、每段长度为 H 的有限 horizon 近似（T = KH，取 H = W = K = √T）。每个 episode 执行反向价值迭代：Q_{k,h}(s,a,z) = r̃(s,a,z) + P̂_k(·|s,a)^T V_{k,h+1}(·,ψ(s,a,z)) + β/√(N_k(s,a)∨1)，Ṽ_{k,h}(s,z) = max_a Q_{k,h}(s,a,z)，再经 clip 操作 V_{k,h}(s,z) = Ṽ_{k,h}(s,z) ∧ (min_{s'} Ṽ_{k,h}(s',z) + 2C)。clip 保留 optimism 同时将统计误差依赖控制于 span C 而非 naive 的 H。
4. **关键理论引理**：Lemma 3 证明对最优策略 π*，增广 MDP 中 sp(V_{r̃,h}^{π*}(·,z)) ≤ 2C 且 HJ_r* − V_{r̃,1}^{π*}(s,z) ≤ C，为 regret 分析提供基准价值函数的一致性界限。
5. **三步分析框架**：Step 1（增广 MDP regret）→ Step 2（转化为原 CMDP 的 Regret(T) + Φ_W(z_{T+1}) ≤ Õ(√T)）→ Step 3（结合强对偶性 reg(t) ≥ −λ* Violation(T) 与 Huber 势性质导出 Violation(T) ≤ Õ(√T)）。

## 实验与结果
- **实验设置**：基于 Yu et al. (2026) 修改的表格型 CMDP，状态空间 S = {0,…,H_env+1}（H_env=6），动作空间 A = {−1,1}^{m−1}（m=2），T = 32,000 步，10 次重复试验。
- **对比基线**：PD-LSCVI-UCB（Yu et al. 2026）、Chen et al. (2022) Alg. 3、Ghosh et al. (2023) Alg. 2、随机策略。
- **主要结果**：SA-CVI-UCB 实现次线性遗憾与约束违反；相比之下，Ghosh et al. (2023) Alg. 2 的约束违反呈线性增长，Chen et al. (2022) Alg. 3 虽获得负遗憾（overfit）但约束违反线性增长。与 PD-LSCVI-UCB 相比，SA-CVI-UCB 在遗憾和约束违反两方面均更优。
- **鲁棒性**：在 bonus 尺度 c ∈ {0.1, 0.3, 1, 3, 10} 敏感性实验中，SA-CVI-UCB 对 c ≤ 1 均保持稳定次线性表现，而各基线在某些 c 值下退化。
- **最强结果**：SA-CVI-UCB 达到 Õ(√T) 遗憾与 Õ(√T) 约束违反，理论保证与实验趋势一致，且运行时间 O(S²AT^(5/2)) 为多项式时间。

## 相关工作脉络
1. **Chen et al. (2022)**：在弱通信表格 CMDP 下通过有限 horizon 近似获得 Õ(T^(2/3)) 界，并证明 Span 约束可达 Õ(√T) 但导致非凸规划；本文规避了非凸性，用状态增广+clip 实现相同速率且计算高效。
2. **Ghosh et al. (2023)**：线性设定下提供 Õ(√T) 非高效算法与 Õ(T^(3/4)) 高效算法；本文在表格设定下首次同时实现高效性与 Õ(√T)。
3. **Yu et al. (2026)**：建立弱通信 CMDP 强对偶性并将高效 primal-dual 算法改进至 Õ(T^(2/3))；本文分析指出其对偶乘子更新导致复合奖励变化使 span 控制失效（Appendix F）。
4. **Sootla et al. (2022b) Saute RL**：同样使用状态增广跟踪安全预算，但其重塑奖励对违反施加 −∞ 惩罚以达成几乎必然安全；本文用有界 Huber 差替代以适应平均奖励 regret 分析。
5. **Hong & Tewari (2025)** / **Hong et al. (2025)**：无约束平均奖励线性 MDP 的高效 clip value iteration；本文将其推广至约束设定，关键在于状态增广隔离了对偶乘子的影响。
6. **Agarwal et al. (2022a,b)** 与 **Provodin et al. (2024)**：在 ergodic/communicating 假设下实现 Õ(√T)（后者为 Bayesian regret）；本文放松至弱通信假设，无需遍历性。

## 局限性与未来方向
1. **仅适用于表格设定**：论文明确指出将 SA-CVI-UCB 推广至线性函数近似（linear CMDP）尚存挑战——安全状态离散化导致可达债务值数量增至 O(1/Δ)，使覆盖数增长至 Õ(d/Δ + d²)，平衡后遗憾率退化为 Õ(T^(2/3))，未能达到表格情形的 Õ(√T)。
2. **参数依赖偏置 span**：理论界中的常数因子 C 显式依赖于 sp(v_r*)、sp(v_g*) 等未知量，实践中需估计或调参。
3. **Slater 条件假设**：要求已知 Slater 常数 γ，未知时需自适应估计。
4. **未来方向**：设计可处理增广状态爆炸的算法以实现线性 CMDP 下的 Õ(√T) 界；探索更灵活的安全状态表示（如连续松弛）以减少覆盖数开销。

## 研究启发与可借鉴点
1. **状态增广替代对偶乘子**：将约束违反信息编码入状态而非奖励，避免了 primal-dual 方法中因乘子更新导致的复合奖励变化问题；此思路可迁移至其他需要处理长期约束的序贯决策场景（如风险敏感 RL、累积代价约束 Bandit）。
2. **Huber 势函数的 telescoping 设计**：利用分段势函数同时满足有界性（Lipschitz）与 telescoping 结构，实现原始 regret 与终端势的等价转换；该技巧可适配不同约束形态的 reward shaping。
3. **Clip 操作保留乐观性**：在价值迭代中对每个固定 z 做跨原状态的 clip，使 clipped value 仍保持对真实值函数的乐观上界，同时控制 span 依赖；此技术可与任何 optimistic planning 框架结合。
4. **强对偶+势能不等式的联合论证**：Step 3 中结合强对偶下界（Regret ≥ −λ* Violation）与 Huber 势的 Lemma 1(3) 性质来界定终端安全状态，形成闭环分析；该组合论证策略可用于其他涉及累积约束的 regret 分析。
5. **跨 episode 保持安全状态连续性**：z_{H+1}^k → z_1^{k+1} 的设计确保累积约束信息在 episode 边界不丢失，对 finite-horizon 近似框架具有普适参考价值。

## 关键术语表
- **Average-reward CMDP**：无限 horizon 平均奖励约束马尔可夫决策过程，目标是在期望累积约束成本不超过阈值的前提下最大化长期平均奖励。
- **Weakly communicating**：弱通信 MDP 性质，状态空间可划分为两个集合：第一个集合内状态在某一稳态确定性策略下互相可达，第二个集合中所有状态在任何稳态策略下均为瞬态；比 ergodic/unichain 更弱。
- **State augmentation**：通过在原状态空间中附加辅助状态变量（如累积约束违反量）来扩展 MDP，将约束信息编码进状态转移而非奖励函数。
- **Huber potential**：Huber 势函数 Φ_W，一种分段光滑势：x≤0 时为 0，中间二次增长，外部线性增长；兼具 Lipschitz 连续性和可控曲率，适合 regret 分析中的有界性要求。
- **Span of value function**：价值函数的 span sp(V) = max_s V(s) − min_s V(s)，衡量值函数在不同状态间的波动幅度；在平均奖励 MDP 中控制 span 是关键技术分析工具。
- **Bias function**：偏置函数 v_r*(s) = lim_{T→∞} (1/T)Σ_t E[Σ_{i=1}^t (r(s_i,a_i) − J_r*) | s_1 = s]，刻画从状态 s 出发到稳态平均奖励的累积偏差，其 span 出现在 regret 界中。
- **Strong duality in CMDP**：弱通信平均奖励 CMDP 中，原始约束优化问题的最优值等于对偶问题的最优值，即 J_r* = inf_{λ≥0} sup_π {J_r^π − λ(J_g^π − b)}。
- **Clipped value iteration**：价值截断迭代，在乐观价值迭代后对价值函数按 span 上限进行 clip 操作，保证估计值函数 span 有界同时维持 optimism 性质。

## 可复现要素
- **数据集**：自定义表格型 CMDP（基于 He et al. 2022 / Yu et al. 2026 修改），非公开标准数据集；环境细节见 Appendix E（H_env=6, m=2, b=0.7, s_1=0）。
- **代码/权重**：论文未提及代码开源链接（ArXiv 版本未附 GitHub）。
- **关键超参**：Δ = 1/√T，Λ = 2/γ，W = K = H = √T，β = 2CS√(log(2TS²|A|/δ)/2)，clip 参数 C = sp(v_r*) + Λ·sp(v_g*) + (Λ/W)(sp(v_g*)² + 2H(sp(v_g*)+2)²) + HΛΔ/2。γ（Slater 常数）已知但策略未知。
