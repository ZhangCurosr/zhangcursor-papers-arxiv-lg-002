---
title: "OPTIMIZER-DEPENDENT-TRAINING-DYNAMICS-CON-VERGE-TO-THE-SAME"
source: https://arxiv.org/pdf/2609.37745v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:10:44"
field: "大规模语言模型训练动力学与缩放律"
keywords: ["neural scaling laws", "optimizer dynamics", "adaptive optimization", "teacher-student model", "softmax classification", "scaling exponent", "boundary localization"]
innovations: ["将loss分解为径向(范数)与切向(对齐)双通道，各自服从幂律但指数依赖优化器", "发现参数无关的和律2αr+αt=1，解释最优数据指数恒为1/3的普适性", "证明优化器决定每步学习速度但不变每样本学习上限，统一动态指数与超参数标度"]
benchmarks: ["单层teacher-student softmax分类玩具模型 (n=128, m=32)"]
---

# 论文速读：OPTIMIZER-DEPENDENT TRAINING DYNAMICS CONVERGE TO THE SAME ONE-THIRD OPTIMAL DATASCALING

## 一句话总结
该论文在 teacher–student softmax 玩具模型上证明：虽然不同优化器（SGD、Adam、Muon、SignGD 等共 7 种）产生的训练动态指数各不相同，但它们均满足一个普适的和律关系 $2\alpha_r + \alpha_t = 1$，从而使得超参数最优调谐后的数据缩放指数始终收敛于 $D^{-1/3}$——即优化器决定"每步学多快"，但不影响"每样本能学到多少"。

## 研究问题与动机
1. **Scaling 指数的起源仍存在争议**：现有神经缩放律（neural scaling laws）解释了实验现象，但缺乏理论根基，无法自信预测未来或指导改进。
2. **前作仅覆盖 SGD/gradient flow**：Liu et al. (2026b)、Kühn et al. (2026) 的理论导出 1/3 指数均基于 SGD 或梯度流，而实际 LLM 预训练普遍使用自适应优化器（Adam、Muon 等），两者是否存在不同的缩放行为尚不清楚。
3. **优化器对动态与最优超参数的影响尚未解耦**：不同优化器的学习率最优缩放规律（如 Adam 的"D magic exponent"≈−0.32）被经验报道，但缺乏与损失动态的统一理论解释。
4. **两个不同的指数概念未被区分**：训练步数维度上的动态指数（loss 随步数衰减）和数据维度上的最优指数（loss 随数据集大小 D 衰减）是不同对象，前作将二者混同处理。

## 核心贡献（创新点）
1. **Loss 分解为径向/切向双通道**：将 cross-entropy loss 分解为径向部分 $L_r$（由权重范数 $\beta$ 控制）和切向部分 $L_t$（由方向对齐误差控制），二者各自服从幂律衰减，且指数与学习率、batch size 无关，仅取决于优化器。
2. **发现参数无关的和律关系**：推导出 $2\alpha_r + \alpha_t = 1$，该关系不含任何拟合参数， Across 7 种优化器均验证成立，解释了为何最优数据指数恒为 1/3。
3. **揭示最优超参数标度的优化器依赖性**：SGD 的最优学习率几乎不随数据量变化（$s_D \approx 0$），Adam 的最优学习率随 $D^{-0.35}$ 下降（接近经验中的"magic exponent" −0.32），但两者最优 loss envelope 均为 $D^{-1/3}$。
4. **提供实用的超参数定位诊断**：在最优处切向 loss 占比恒为 $L_t/L = 1/3$，因此单次运行即可定位最优超参数，无需完整 sweep。

## 方法详解
**玩具模型设定**：单层 softmax teacher–student 模型，teacher $W_T \in \mathbb{R}^{n\times m}$ 固定（$\|W_T\|_F=1$），student $W_S$ 可训练；输入 $x\sim\mathcal{N}(0,I_m)$ 经 RMS 归一化；teacher 对每个输入产生 one-hot 硬标签（无限逆温度极限）；损失为在线交叉熵 $\langle CE(p(x),q(x))\rangle_x$。

**Loss 分解几何**：引入辅助 rescaled teacher $W_R = \beta\widehat{W}_T$（携带 student 当前范数 $\beta=\|W_S\|_F$ 但保持 teacher 方向）；定义：
- 径向损失 $L_r = CE(p, p_R)$，其中 $p_R = \mathrm{Softmax}(W_R x)$，度量尺度失配；
- 切向损失 $L_t = L - L_r$，度量方向错位。

**动力学方程**：将更新建模为 Ornstein–Uhlenbeck 过程，preconditioner 缩放形式 $A\sim B^p\beta^q$（$p,q$ 为涌现指数）；径向漂移满足 $d\beta/d\tau \propto B^p\beta^{q-2}$，积分得 $\beta\sim(zD)^{1/(3-q)}$（$z\equiv\eta B^{p-1}$）；切向稳态协方差由预处理 Lyapunov 方程确定，得到关键迹恒等式：$\mathbb{E}[L_t]\simeq \frac{\eta}{4B}\mathrm{Tr}(A\Sigma)$。

**指数推导**：
- 径向：$\alpha_r = 1/(3-q)$，对应 $L_r\sim(zD)^{-\alpha_r}$
- 切向：$\alpha_t = (1-q)/(3-q)$，对应 $L_t\sim z^{2\alpha_r}D^{-\alpha_t}$
- 消去 $q$ 得到和律：$2\alpha_r+\alpha_t=1$
- 最小化 $L=L_r+L_t$ 关于 $z$：最优处 $L_r^*=2L_t^*$，$L^*\propto D^{-1/3}$，$z^*\propto D^{-q/3}$

**实验设置**：$n=128$ 类，$m=32$ 维输入，在线训练（每步新 batch，不重复样本），每个优化器 sweep 16 个学习率 × 8 个 batch size（16–2048），共 128 条轨迹/优化器，每条训练 10,000 步。测试的 7 种优化器：SGD、PowerAdam($a\in\{0.125,0.25,0.375,0.5\}$，其中 $a=0.5$ 即 Adam)、Muon、SignGD。

## 实验与结果
**数据集/模型**：自构建单层 teacher–student softmax 分类模型，非标准 NLP 数据集；$n=128, m=32$，one-hot 硬标签。

**主要结果数字**：
- SGD：$\alpha_r = 0.35\pm0.01$，$\alpha_t = 0.33\pm0.02$，$\alpha_D = 0.341\pm0.001$，$s_D=-0.04\pm0.01$，$s_B=1.01\pm0.01$
- Adam：$\alpha_r = 0.48\pm0.03$，$\alpha_t = 0.08\pm0.01$，$\alpha_D = 0.341\pm0.003$，$s_D=-0.35\pm0.01$，$s_B=0.76\pm0.01$
- 7 种优化器的 $\alpha_D$ 中 6 种在 $1/3$ 的 2.5% 以内，跨优化器散度仅 0.004；SignGD 的 $\alpha_D=0.315\pm0.012$，从下方趋近 1/3
- 通过和律计算与 envelope 拟合两条独立路径得到的 $\alpha_D$ 在误差范围内一致

**最强结果**：所有 7 种优化器的最优数据指数均收敛至 $1/3$（$0.333$–$0.341$），且和律 $2\alpha_r+\alpha_t=1$ 横跨完全不同结构的优化器（对角 preconditioner 到正交 Muon），无一偏离。

## 相关工作脉络
1. **Liu et al. (2026b)、Kühn et al. (2026)**：提出 softmax 学习 peaked distribution 产生 1/3 时间缩放指数的机制，但仅处理 gradient flow/SGD 场景；本文将其推广到任意自适应优化器并区分动态指数与最优数据指数。
2. **Bjorck et al. (2024)**：经验报道 Adam 预训练中 optimal LR 随 $D^{-0.32}$ 下降的"magic exponent"；本文从理论上给出 $s_D=-q/3$ 的物理解释，并测量到 $s_D=-0.35$。
3. **Ren et al. (2026)**：在 HyperP 框架下报告 Muon 的 $s_D\approx-0.32$、$s_B\approx0.558$；本文指出 batch 指数 $s_B$ 与 preconditioner 的 batch 缩放 $p$ 相关，理想 $A=V^{-a}$ 下 $p=q$，但实测分离，解释了差异来源。
4. **Goyal et al. (2017)、Smith et al. (2018)**：线性规则 $\eta^*\propto B$ 用于 SGD；本文复现此结果（$s_B=1.01$），并指出自适应方法的 batch 标度更陡（Adam: $s_B=0.76$）。
5. **Malladi et al. (2022)**：从 SDE 角度推导自适应方法的最优超参数标度律；本文与之互补，将动态指数与超参数标度统一在同一框架内。
6. **Soudry et al. (2018)、Gunasear et al. (2018)**：特征化 separable data 上梯度方法的隐式偏置；本文的 softmar 场景为其提供了一个可解析的最小实例。

## 局限性与未来方向
1. **模型过于简化**：仅单层 softmax，未测试多层或 Transformer 架构；从单层到 LLM 的实际距离较大。
2. **依赖硬标签**：真实 teacher 有有限逆温度时，student 存在有限最优尺度，幂律仅在中间区间成立，而非无限持续。
3. **需要 teacher 方向**：$L_r$、$L_t$ 及切向投影均相对于 $\widehat{W}_T$ 定义，无 ground-truth teacher 的实际数据中分解存在概念复杂性。
4. **Adam 处于分析区域的边缘**：其三个独立途径得到的 $q$ 值不一致（$q\simeq0.74$（范数）、$0.92$（径向）、$1.05$（$s_D$）），后者超出允许范围（$q>1$ 使 $\alpha_t<0$）。
5. **drift-dominated 窗口限制**：分析覆盖恒定学习率下的 drift 主导区间；大幅超出此窗口后，自适应优化器可能进入质变的不同 regime。
6. **未来方向**：（a）改变边界附近 margin density（$p_\Delta(\delta)\sim\delta^\nu$）可加速 scaling 指数（$\alpha_D=(\nu+1)/(\nu+3)$）；（b）自相似 schedule（cosine decay、warmup-stable-decay）可能保持和律，但非自相似 schedule 未知；（c）在大规模 LLM 上验证理论预测。

## 研究启发与可借鉴点
1. **双通道 loss 分解框架**：将 loss 分解为"径向（范数增长）"和"切向（方向对齐）"两个独立衰减通道的方法论，可迁移到分析其他非线性的 scaling 场景，尤其是存在 margin 分布的分类任务。
2. **和律（sum rule）作为约束检验工具**：$2\alpha_r+\alpha_t=1$ 是一个无拟合参数的理论预言，可用于快速验证新优化器是否符合边界局域化机制——若偏离则暗示存在其他动力学机制（如非衰减协方差扇区）。
3. **$L_t/L=1/3$ 作为超参数定位诊断**：仅需单次运行估算切向 loss 占比即可 multiplicative 地定位最优学习率，无需完整 sweep，实用价值高。
4. **统一动态指数与超参数标度**：将 $\alpha_r$ 与 $s_D$ 通过 $s_D=-1+1/(3\alpha_r)$ 联系，建立了"训练动态→最优超参数"的映射桥梁，可用于解释不同工作间的经验标度差异。
5. **边界局域化（boundary localization）的分析范式**：将损失和梯度噪声局域化在决策边界附近宽度 $\sim\beta^{-1}$ 的 layer 内的思路，可推广到其他非线性模型中分析慢幂律衰减的起源。

## 关键术语表
**Teacher–Student Model**：固定 teacher 网络生成标签，学生网络在线训练以匹配 teacher 输出的单层 softmax 分类玩具模型。
**Radial Loss ($L_r$)**：由权重范数 $\beta$ 驱动的 loss 分量，衡量 student 输出分布的"尖锐度"不足（softmax 未饱和）造成的代价。
**Tangential Loss ($L_t$)**：由 student 与 teacher 方向错位驱动的 loss 分量，衡量方向不对齐带来的额外代价。
**Preconditioner Scaling ($A\sim B^p\beta^q$)**：优化器 preconditioner 对 batch size $B$ 和权重范数 $\beta$ 的幂律缩放，其中 $p,q$ 为涌现指数，决定动态和超参数标度行为。
**Sum Rule ($2\alpha_r+\alpha_t=1$)**：径向与切向时间衰减指数之间的无参数约束关系，是所有遵循边界局域化机制的优化器必须满足的普适关系。
**Optimal Data Exponent ($\alpha_D$)**：最优调谐后 loss 随数据集大小 $D$ 的衰减指数，本文证明其恒为 $1/3$。
**Magic Exponent**：Bjorck et al. (2024) 经验发现的 Adam 预训练中 optimal LR 随 $D^{-0.32}$ 下降的标度指数，本文理论给出 $s_D=-q/3$ 的解释。
**Boundary Localization**：大 $\beta$ 下 loss 和梯度被局域在决策边界附近宽度 $\sim\beta^{-1}$ 的 boundary layer 中的机制，是幂律衰减的根源。

## 可复现要素
- **数据集**：自构建（单层 teacher–student softmax，$n=128, m=32$，标准正态输入+RMS归一化），非公开标准数据集
- **代码**：论文声明"Code for the toy-model training runs and for all analysis and figures will be released publicly; until then, it is available from the authors upon request"
- **权重**：不适用（toy model，无预训练权重）
- **关键超参**：SGD learning rate $[10^{-1}, 10^3]$，Adam $[10^{-3}, 10^1]$，batch size $[16, 2048]$，训练步数 10,000；PowerAdam $(\beta_1, \beta_2, \epsilon)=(0.9, 0.999, 10^{-8})$，Muon 使用 Nesterov momentum 0.95 + 5 次 Newton–Schulz 迭代，SignGD momentum 0.9，无 weight decay
