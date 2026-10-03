---
title: "PREDICTIVE-SAFETY-CURRICULA-FOR-ROBUST-LEGGEDLOCOMOTION"
source: https://arxiv.org/pdf/2609.37070v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 18:33:07"
---

# 论文速读：PREDICTIVE-SAFETY-CURRICULA-FOR-ROBUST-LEGGEDLOCOMOTION

## 一句话总结
本文提出预测性安全课程（Predictive Safety Curricula, PSC），通过训练一个分布安全价值网络预测未来安全代价，将预测的上尾风险转化为地形上下文与随机环境事件的采样优先级；该方法仅调整训练经验分布，不修改任务奖励与 PPO 优化目标，在受控基准、生产训练栈与 ANYmal 实机上均显著提升了足式机器人的 locomotion 可靠性。

## 研究问题与动机
- 已训练出的 locomotion 策略在平均任务表现较高时，仍可能在困难地形或退化观测条件下出现罕见但致命的下肢碰撞与失衡。
- 现有课程学习（标准地形渐进、PLR、LP 等）主要基于能力进展、优势估计或学习潜力分配经验，未显式针对“尾部安全风险”进行采样优化，导致高平均成功率与高尾部失败率并存。
- 随地形类型、难度等级与随机化变量维度增加，直接依赖经验统计估计各配置下的安全失败率数据效率极低且存在严重滞后。
- 领域缺乏一种可与主流强化学习训练循环解耦、仅通过调整经验分布来提升可靠性的 curriculum 机制。

## 核心贡献（创新点）
1. **预测性安全分配机制**：引入独立于策略优化的分布安全 critic，利用其对未来安全暴露的预测动态指导训练经验采样，与“修改奖励/约束策略更新”的安全学习方法本质不同。
2. **统一上下文与事件双路径优先框架**：用同一安全信号分别调制地形等级-指令上下文的概率分布与可回放随机环境事件的复选概率，实现结构化任务条件与瞬时随机扰动的协同优化。
3. **上尾 CVaR 风险读取优于经验统计**：系统消融证明预测性上尾风险（QC-CVaR）在成功率与跨种子稳定性上均显著优于均值读取与基于历史 episode 返回的经验尾部统计。
4. **跨尺度验证与生产可迁移性**：在 Isaac Lab 公开基准、ANYmal-X 生产楼梯攀爬训练栈及 ANYmal-D 实物硬件三重设定下验证，且教师策略的优势可通过 teacher-student 蒸馏完整保留至学生策略。

## 方法详解
- **训练条件分解与安全回报定义**：将课程上下文因子化为 $z=(\tau, \ell, b)$（地形类型、难度等级、指令桶）与随机事件 $\xi$，联合分布 $d_j(z,\xi)=p(\tau)p_j^{\text{ctx}}(\ell,b|\tau)p_j^{\text{evt}}(\xi|z)$。辅助安全代价由三项加权构成：躯干过度倾斜项 $c_t^{\text{ori}}$、肢体靠近基座项 $c_t^{\text{dist}}$、大腿/胫骨非预期接触项 $c_t^{\text{und}}$，折扣累积安全回报 $G_t^c$ 作为 critic 预测目标。
- **共享分布安全 Critic**：采用分位数回归 MLP 预测 $K$ 个条件分位数 $q_{\psi,i}(x_t)$，以 Polyak 滑动平均目标网络配合 n-step bootstrap 目标 $y_{t,k}$，使用分位数 Huber 损失 $\mathcal{L}(\psi)$ 训练；该网络仅接收与 PPO value critic 相同的特权观测，独立更新。
- **风险读取与 episode 评分**：对分位数排序后取最大 $k_\alpha=\lceil\alpha K\rceil$ 个分位数平均得到上尾 CVaR $\widehat{R}_\alpha^{\text{disc}}$（默认 $\alpha=0.05$ 对应顶部 2 个分位数），或取全部分位数平均得均值 $\widehat{\mu}$；沿 episode 轨迹状态级平均得到 $\bar{s}_e$，按上下文聚合为 $S_j(z)$ 或直接绑定至 episode 内遭遇的随机事件。
- **安全优先上下文分配**：在基础能力课程允许的 $\mathcal{E}_{\tau,j}$ 集合内，以 $p_j^{\text{ctx}}=\varepsilon U+(1-\varepsilon)\text{Norm}(P_j)$ 混合采样；优先级 $P_j$ 融合预测安全得分、地形 staleness（距上次采样步数）与覆盖率（累积得分 episode 数倒数），三者独立 min-max 归一化后以 $\lambda_S,\lambda_a,\lambda_u$ 加权。
- **安全优先事件重放**：每个上下文维护小规模回放缓冲区，按关联 episode 的安全得分对历史事件实现进行偏好重采样，保留固定概率抽取新扰动以维持探索，两条分配路径共享同一 critic 信号但作用于 Eq.(1) 的不同因子。

## 实验与结果
- **数据集/平台**：Isaac Lab ANYmal-D 粗糙地形公开基准（受控）、ANYmal-X 生产楼梯攀爬训练栈、ANYmal-D 实物硬件（开级台阶与固定楼梯攀爬）。
- **评估基线**：Standard terrain progression（Baseline）、Prioritized Level Replay（PLR）、Learning Progress（LP）。
- **ANYmal-D 受控基准主结果**：PSC 在所有清洁与观测噪声条件（$v=1,2,3$）下均取得最高平均成功率；较 LP 提升 0.16~1.38 个百分点，$v=2$ 时失败率由 6.19% 降至 4.81%（相对下降 22.3%），$v=3$ 时由 7.19% 降至 6.69%；地形分配呈现结构化重分布而非单纯偏向高难度等级。
- **消融与 Critic 诊断**：QC-CVaR（93.05%）> Empirical CVaR（91.38%）> QC-mean（90.40%）；Critic 预测均值与蒙特卡洛真实安全回报的 Spearman $\rho=0.773$，高风险态 top-decile 碰撞发生率为全体的 6.88×、接触率为 2.81×，AUPRC gain 分别为 0.079（碰撞）与 0.461（接触）；$\alpha=0.05$ 为最优点，扩大上尾质量会削弱性能。
- **ANYmal-X 生产栈**：PSC 教师策略平均成功率 88.45%，较 LP（87.04%）提升 1.41pp；蒸馏学生策略维持 87.76% 成功率，表明训练阶段的预测安全分配优势可完整穿透生产部署管线。
- **实物硬件**：在 3 个匹配种子、各 300 次单向跨越下，PSC 胫骨碰撞发生率降至 10.7±4.9/100 次，较 LP（29.0±5.3）降低 63.2%，每种子均下降；楼梯攀爬测试中 PSC 实现 0/50 次胫骨碰撞（Baseline 33/50），脚底刮擦减少且完成率 50/50，平均行进速度无显著差异。

## 相关工作脉络
- **Curriculum & adaptive sampling（PLR、LP、HACL、LP-ACRL 等）**：基于优势绝对值、连续 episode 回报变化或历史能力匹配分配采样；PSC 将信号替换为状态条件化的预测安全暴露，专注尾部风险而非平均学习能力。
- **Risk-aware curricula（RACGEN、CeSoR、Safety-Prioritizing Curricula）**：多聚焦于低回报或约束违反频率的静态/软偏好；PSC 通过分布 critic 实现跨地形与随机事件的细粒度、前瞻性优先级计算。
- **Safety critics & constrained RL（CPO、Recovery RL、Agile But Safe、Distributional Safety Critic）**：通常将安全模型嵌入策略梯度约束或部署期干预；PSC 明确将其剥离出 PPO 目标，仅作为 curriculum 采样信号，部署时不依赖该 critic。
- **Distributional RL（QR-DQN、IQN）**：PSC 继承分位数回归框架学习安全回报的条件分布，使其能同时派生均值与上尾 CVaR，区别于单值 critic 的确定性预测。

## 局限性与未来方向
- 安全代价函数依赖人工设计，当前仅对躯干倾斜、肢体接近与特定下肢碰撞敏感，无法覆盖全部失效模式（如实机测试中脚底刮擦事件有所上升）。
- 课程调整局限于已暴露的随机化环境参数空间，无法主动探索或生成未见的高风险地形/交互构型。
- 未来方向包括构建结构化或多模态安全预测以保留不同失败模式的区分度，并从重加权/重放扩展到主动生成或搜索预测高风险的新训练条件，使课程从“在现有分布内优先”升级为“围绕剩余可靠性缺口重塑训练分布”。

## 研究启发与可借鉴点
- **预测信号与策略优化解耦**：将辅助 critic 仅用于 curriculum 采样而非奖励塑形或策略约束，工程接入成本低且不与主流 PPO/SAC 循环冲突，可直接复用于其他具身控制的经验调度场景。
- **状态级风险→
