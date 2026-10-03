---
title: "OPTIMIZER-DEPENDENT-TRAINING-DYNAMICS-CON-VERGE-TO-THE-SAME"
source: https://arxiv.org/pdf/2609.37745v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:10:40"
field: "优化理论/神经标度律"
keywords: ["neural scaling laws", "optimizer dynamics", "adaptive optimization", "one-third data scaling", "teacher-student model", "loss decomposition"]
innovations: ["将损失分解为径向与切向双通道并证明各自幂律指数满足无参数求和律 2αr+αt=1", "证明七种优化器的最优数据缩放指数均为 1/3，与优化器选择无关", "建立动态指数与超参标度指数之间的约束关系 sD = -1 + 1/(3αr)"]
benchmarks: ["单层 softmax 教师-学生玩具模型 (n=128, m=32)"]
---

# 论文速读：OPTIMIZER-DEPENDENT TRAINING DYNAMICS CONVERGE TO THE SAME ONE-THIRD OPTIMAL DATASCALING

## 一句话总结
本文通过教师-学生单层 softmax 玩具模型发现：不同优化器的训练动力学（径向与切向两个衰减通道）显著不同，但二者满足无参数守恒律 $2\alpha_r + \alpha_t = 1$，从而确保经过最优超参调优后，所有优化器均收敛到相同的 $D^{-1/3}$ 数据缩放指数。

## 研究问题与动机
- 大型语言模型（LLMs）的测试损失随训练数据量呈幂律下降（scaling law），近期理论工作（Liu et al., 2026b; Kühn et al., 2026）从 softmax 学习尖锐分布的角度解释了 $1/3$ 指数的起源，但这些工作仅针对梯度流或纯 SGD，而 LLM 实际使用自适应优化器（如 Adam、Muon）。
- 优化器在每个步长上缩放梯度并改变噪声结构，理论上可能同时影响"单条运行中损失随步数的下降速度"和"最优调参后损失随数据集大小 $D$ 的下降速度"；目前文献未明确区分这两个指数，也未回答哪些部分由优化器决定、哪些是普适的。
- LLM 实践中已出现 Muon、PowerAdam 等新优化器，不同优化器的最优学习率对 batch size 和预算的标度行为已被报告不同，但系统性的理论解释尚缺。

## 核心贡献（创新点）
- **损失分解为径向与切向双通道**：将损失分解为 $L_r$（由权重的 softmax 饱和度决定）和 $L_t$（由与教师方向的不对齐决定），每个通道独立遵循幂律，且各自的时间指数 $\alpha_r$、$\alpha_t$ 是优化器相关的但超参无关。与既有工作的本质区别在于：此前的单通道分析（Liu et al., 2026b; Kühn et al., 2026）隐含假设完全对齐，本文显式刻画了对齐偏差的独立贡献。
- **最优数据缩放指数 $1/3$ 的优化器无关性**：尽管不同优化器的最优超参标度行为大不相同，但经最优调参后最优损失包络均以 $L^*(D) \sim D^{-1/3}$ 缩放。与既有工作的本质区别在于：此前理论仅在 SGD 下得到 $1/3$，本文证明这是所有七种优化器的普适结论。
- **动态指数求和律 $2\alpha_r + \alpha_t = 1$**：提出并验证了一个无参数守恒关系，该关系同时解释了为何最优数据指数不依赖优化器。与既有工作的本质区别在于：该关系是一个新的普适约束，将优化器的具体行为与普适缩放指数联系起来，此前从未被发现。

## 方法详解
- **教师-学生模型**：单层 softmax 分类网络，教师矩阵 $\widehat{W}_T \in \mathbb{R}^{n \times m}$ 固定且范数为 1，学生矩阵 $W_S$ 在线训练以匹配教师分布。输入 $x \sim \mathcal{N}(0, I_m)$ 经 RMS 归一化，教师以 one-hot 标签输出最大对数概率类（硬标签/无穷逆温度极限）。
- **损失分解几何**：引入辅助"缩放教师" $W_R = \beta \widehat{W}_T$（携带学生当前范数但教师方向），定义径向损失 $L_r = \mathrm{CE}(p, p_R)$ 与切向损失 $L_t = L - L_r$。$L_r$ 衡量尺度不匹配（softmax 不饱和），$L_t$ 衡量方向不对齐。
- **预条件标度假设**：设优化器更新 $\sim -\eta g$，预条件矩阵以 $A \sim B^p \beta^q$ 形式标度，其中 $p$ 控制 batch size 依赖、$q$ 控制权重范数依赖，两者均为涌现量（从宏观动力学测量而非公式直接读出）。对于对角二阶矩预条件器 $A = V^{-a}$ 的理想情况有 $p=q=a$。
- **径向通道演化**：在边界局域化机制下，$L_r(\beta) \simeq C_L / \beta$，结合预条件径向漂移方程积分得 $\beta(D) \sim (zD)^{1/(3-q)}$，其中 $z \equiv \eta B^{p-1}$，进而 $L_r \sim (zD)^{-1/(3-q)}$。
- **切向通道演化（关键推导）**：将不对齐位移 $e$ 建模为 Ornstein-Uhlenbeck 过程，利用离散 Lyapunov 方程求稳态协方差，得到迹恒等式（Proposition 1）：$\mathbb{E}[L_t] \simeq \frac{\eta}{4B}\mathrm{Tr}(A\Sigma)$，不要求噪声协方差与曲率成比例或可交换。代入标度关系得 $L_t \sim z^{2/(3-q)} D^{-(1-q)/(3-q)}$。
- **求和律的导出**：由 $\alpha_r = 1/(3-q)$ 与 $\alpha_t = (1-q)/(3-q)$ 消去 $q$ 即得 $2\alpha_r + \alpha_t = 1$，不依赖 $q$ 的具体值。
- **最优调参**：总损失 $L = L_r + L_t$ 对 $z$ 求极小，得 $z^* \propto D^{-q/3}$，此时 $L_r^* = 2L_t^*$（两通道固定比），且 $L^*(D) \propto D^{-1/3}$。最优学习率标度为 $s_D = -q/3$、$s_B = 1-p$，导出 $\eta^* \sim D^{-q/3}$ 和 $\eta^* \sim B^{1-p}$。
- **实验设置**：$n=128$ 类、$m=32$ 维，7 种优化器（SGD、PowerAdam $a \in \{0.125, 0.25, 0.375, 0.5\}$、Muon、SignGD），每种优化器扫 16 个学习率 × 8 个 batch size，每运行 10000 步，用对数二次拟合定位 $\eta^*$，再对 $(B, D)$ 平面做最小二乘拟合提取 $s_D, s_B$。

## 实验与结果
- **双通道幂律指数（Figure 2）**：SGD 下 $\alpha_r = 0.35 \pm 0.01$、$\alpha_t = 0.33 \pm 0.02$（均接近 $1/3$）；Adam 下 $\alpha_r = 0.48 \pm 0.03$、$\alpha_t = 0.08 \pm 0.01$（显著分离）。
- **求和律验证（Figure 4a）**：七种优化器的 $(\alpha_r, \alpha_t)$ 全部落在直线 $2\alpha_r + \alpha_t = 1$ 上，包括不属于 $V^{-a}$ 族的 Muon 和 SignGD。
- **最优学习率标度（Figure 3a-b）**：SGD 的 $\eta^*$ 几乎不依赖数据量（$s_D = -0.04 \pm 0.01$），而 Adam 的 $\eta^* \sim D^{-0.35 \pm 0.01}$（接近 Bjorck et al. 报告的 LLM "magic exponent" 0.32）；batch size 标度 $s_B = 1.01 \pm 0.01$（SGD）vs $0.76 \pm 0.01$（Adam），均显著高于常见的 $\sqrt{B}$ 规则。
- **最优数据缩放（Figure 3c, Table 4）**：六种优化器的 $\alpha_D$ 在 $1/3$ 附近 $2.5\%$ 以内（scatter 0.004），SignGD 略低（$0.315 \pm 0.012$）且从下方趋近；两种独立测量路径（直接包络拟合 vs 代入求和律）得到一致结果，SGD: $0.341 \pm 0.001$、Adam: $0.341 \pm 0.003$。
- **最优处通道比（Figure 10）**：所有七种优化器的 $L_t/L$ 在所有超参组合下坍缩到同一函数，且在最优处精确穿越 $1/3$，无需额外拟合。

## 相关工作脉络
- **Neural scaling 指数的起源理论**：Liu et al. (2026b) 与 Kühn et al. (2026) 从 softmax 非线性学习尖锐分布导出 $1/3$ 指数，但仅处理梯度流/SGD 情形；本文将其推广到七种优化器并给出普适性证明。
- **超参标度律**：Goyal et al. (2017)、Smith et al. (2018) 导出 SGD 线性标度 $\eta^* \sim B$；Malladi et al. (2022) 导出自适应方法 $\eta^* \sim \sqrt{B}$；Bjorck et al. (2024)、Ren et al. (2026)、Li et al. (2025)、Bergsma et al. (2025) 从 LLM 训练数据中提取经验标度律；本文提供了统一的理论框架解释这些标度差异及其与优化器的关系。
- **分离数据上梯度下降的隐式正则化**：Soudry et al. (2018)、Ji & Telgarsky (2018)、Nacson et al. (2019)、Lyu & Li (2019) 等刻画了 norm-growing 方向收敛特性；Gunasekar et al. (2018)、Wang et al. (2021, 2022)、Zhang et al. (2024) 等研究优化器对隐式偏差的影响；本文的分解提供了这类问题的最小可解分析实例。
- **优化器行为差异研究**：Duchi et al. (2011)、Kingma & Ba (2015)、Reddi et al. (2019)、Wilson et al. (2017)、Kunstner et al. (2023, 2024)、Zhang et al. (2020) 等从收敛性分析与实证对比角度研究自适应方法；本文在"无有限最优值"的 setting 下分析优化器如何影响损失下降速率而非收敛点。
- **优化器设计中的 river valley 几何**：Cohen et al. (2021, 2024)、Wen et al. (2025)、Liu et al. (2025c) 描述损失地貌为 river valley；本文的单层 softmax 模型为该直觉提供了最小可解实现，其中非线性和边界局域化驱动了观测到的幂律动力学。
- **Chinchilla scaling laws**：Hoffmann et al. (2022)、Besiroglu et al. (2024) 报告 LLM 数据指数在 $0.28 \sim 0.37$ 范围，本文测得的 $0.341$ 落在其中间，为玩具模型与真实 LLM 的对应关系提供了定量桥梁。

## 局限性与未来方向
- 玩具模型仅为单层 softmax，未验证多层网络或完整 Transformer 架构；虽 $L^*(D)$ 包络和最优指数等核心结论可迁移至 LLM 尺度，但需进一步大规模实验检验。
- 使用硬标签（one-hot），教师逆温度为无穷大；若教师有有限温度，学生在有限尺度处达到最优，幂律仅在中间区间成立，而非无限延续。
- Adam 的有效预条件指数 $q \simeq 0.9$ 与理想 $A = V^{-1/2}$ 的 $q = 0.5$ 差距较大，且 batch 指数 $p \simeq 0.24$ 亦与理想值分离——从优化器定义本身预测 $p$ 和 $q$ 仍是开放问题。
- 分析仅覆盖漂移主导区域（有限步数内），自适应优化器训练远超此窗口后可能进入不同定性 regime；此外，非自相似学习率调度可能破坏求和律。
- 损失分解（$L_r, L_t$、切向投影）需已知教师方向，在无真实教师标签的实际任务中，分析将面临额外概念复杂度。
- 未来可探索通过改变边界附近 margin density 的局部几何（使密度在边界处消失）来突破 $1/3$ 上限，或通过自适应调度进一步优化数据缩放指数。

## 研究启发与可借鉴点
- **双通道分解框架**：将损失分解为"尺度/饱和度"与"方向/对齐"两个独立通道，并分析各自的动态指数，这一视角可迁移至其他无界最优 scaling 问题（如 logistic 回归、margin-based 分类），为理解多通道竞争动力学提供分析工具。
- **无参数守恒律的发现方法**：从预条件标度假设 $A \sim B^p \beta^q$ 出发，通过迹恒等式消除曲率与噪声的具体依赖，得到一个与优化器细节无关的求和律——这种"以宏观标度替代微观细节"的推导演绎策略值得在其他 optimization theory 工作中借鉴。
- **最优超参位置的单次运行诊断**：最优处 $L_t/L = 1/3$ 是一个无需 sweep 即可定位最优学习率的实用判据，可在实际训练中用作监控信号或 early stopping criterion 的设计依据。
- **与 LLM 超参标度的定量对比**：本文测得的 Adam $s_D = -0.35$ 与 Bjorck et al. (2024) 的 $-0.32$ "magic exponent"高度吻合，为从理论角度理解 LLM 训练中观察到的经验规律提供了可能的机制解释，可指导更精细的超参调度策略设计。
- **求和律的实验验证范式**：通过多种不同族类的优化器（SGD、PowerAdam 族、正交化的 Muon、sign-based 的 SignGD）统一落在同一条线上，展示了如何用多个独立族类的实验数据交叉验证一个理论预测，可作为后续工作的方法论参考。

## 关键术语表
- **径向损失 $L_r$**：由学生权重范数（softmax 饱和度）决定的损失分量，衡量尺度不匹配程度，随 $\beta$ 增大而衰减。
- **切向损失 $L_t$**：由学生方向与教师方向不对齐决定的损失分量，衡量分类方向偏差，受 Hessian 恢复力与 mini-batch 噪声注入的动态平衡所决定。
- **动态指数 $\alpha_r, \alpha_t$**：分别描述 $L_r$ 和 $L_t$ 随训练步数呈幂律下降的指数，是优化器相关但学习率和 batch size 无关的本征量。
- **最优数据指数 $\alpha_D$**：描述经最优超参调优后测试损失随数据集大小 $D$ 的下降指数，本文证明其为普适值 $1/3$。
- **求和律 $2\alpha_r + \alpha_t = 1$**：连接径向与切向动态指数的无参数守恒关系，是 $1/3$ 最优数据缩放指数的底层原因。
- **预条件标度指数 $p, q$**：涌现量，分别描述优化器预条件矩阵对 batch size ($B^p$) 和权重范数 ($\beta^q$) 的标度依赖，$q$ 控制 $s_D$，$p$ 控制 $s_B$。
- **边界局域化（boundary localization）**：在 $\beta \to \infty$ 极限下，只有 margin 接近零的样本（处于决策边界附近宽度 $O(\beta^{-1})$ 的薄层内）对损失和梯度有显著贡献的机制。
- **最优损失包络**：对每个数据集大小 $D$，在所有超参组合中取最小测试损失形成的曲线 $L^*(D)$，是反映实际可用性能的合理度量。

## 可复现要素
- **数据集**：合成 teacher-student 数据，输入 $x \sim \mathcal{N}(0, I_m)$ 经 RMS 归一化，固定单位范数教师矩阵 $\widehat{W}_T$（元素由 PyTorch 默认 linear-layer 初始化 $\mathcal{U}(-m^{-1/2}, m^{-1/2})$ 生成），$n=128$ 类、$m=32$ 维，one-hot 标签。
- **代码/权重**：论文声明代码将公开发布，目前在作者处可索取。
- **关键超参**：Adam/PowerAdam 使用 $(\beta_1, \beta_2, \epsilon) = (0.9, 0.999, 10^{-8})$；Muon 使用 Nesterov momentum 0.95 + 5 次 Newton-Schulz 迭代；SignGD 使用 momentum 0.9；均不使用 weight decay；每运行 10000 步，每 10 步记录一次，测试集大小为 4096。
