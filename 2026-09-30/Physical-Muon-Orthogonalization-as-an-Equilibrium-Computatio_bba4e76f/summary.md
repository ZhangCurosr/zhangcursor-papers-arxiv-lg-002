---
title: "Physical-Muon-Orthogonalization-as-an-Equilibrium-Computatio"
source: https://arxiv.org/pdf/2609.37525v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 18:34:41"
field: "模拟存内计算与硬件友好优化器"
keywords: ["Physical Muon", "analog in-memory computing", "orthogonalization", "Newton-Schulz", "continuous-time flow", "resistive crosspoint array", "optimizers for physical neural networks"]
innovations: ["将Muons正交化转化为连续时间流平衡态计算，避免密集矩阵乘", "用随机探针+互易读取实现硬件可执行的rank-1写入正交化", "Frobenius归一化替代σ_max并双倍求解预算保持训练等价性"]
benchmarks: ["FineWeb language modeling", "GPT-2 124M energy projection", "8x8 ngspice circuit simulation"]
---

# 论文速读：Physical Muon: Orthogonalization as an Equilibrium Computation

## 一句话总结
Physical Muon 将 Muon 优化器的 Newton–Schulz 正交化操作转化为连续时间流的平衡态计算，通过随机探针以矩阵-向量乘积、互易读取和局部 rank-1 写入实现，首次使 Muon 可在模拟存内计算硬件上高效执行，在 10.95M 参数 Transformer 上验证损失仅比数字 NS5 高 0.0085（密集流），接近物理实现仅高 0.0188。

## 研究问题与动机
1. **模拟存内计算需配套训练优化器**：电阻阵列通过电流求和做矩阵-向量乘，无法原生支持密集矩阵-矩阵乘，SGD 虽本地适配但 Transformer 性能远落后于 Adam；Adam 受模拟偏置影响不稳定。
2. **Muon 训练性能强但硬件不可实现**：Muon 通过正交化动量矩阵在语言模型上优于 AdamW，但其标准 Newton–Schulz（NS）迭代依赖密集矩阵乘，难以映射到电阻交叉点阵列。
3. **稀疏/投影实现精度不足**：已有硬件友好正交化方法（如 PolarExpress）在器件误差下易发散，且对模拟偏置敏感。
4. **需要一个可与硬件原语对齐的正交化方案**：需将正交化表达为数组读取（read）与局部 rank-1 写入（write）的组合，而非密集矩阵乘。

## 核心贡献（创新点）
1. **连续时间流正交化框架**：将 NS 迭代替换为由 ODE $\dot{X}=\hat{M}-XX^T\hat{M}$ 描述的平衡态求解，从零初始化使各奇异值独立演化（$\dot{d}_i=\hat{s}_i(1-d_i^2)$），本质区别在于用动力系统的渐近行为替代有限步多项式逼近。
2. **随机探针实现硬件可执行正交化**：通过 Hutchinson 型探针向量将密集更新分解为三次矩阵-向量运算（$p=\hat{M}v,\ q=X^Tp,\ r=Xq$）和一次 rank-1 写入 $\Delta X=\eta(p-r)v^T$，期望在固定 $X$ 下等于密集 Euler 步；本质区别在于将密集乘积分解为硬件原语。
3. **Frobenius 归一化替代 $\sigma_{\max}$**：用数组内可求和的 Frobenius 范数作为动量归一化因子，硬件代价低，只需将求解时间加倍即可补偿收敛减速（中位缩放比约 2.07）；本质区别在于消除了对最大奇异值计算的依赖。
4. **端到端电路验证与训练等价性证明**：在 8×8 电阻阵列上做 ngspice 瞬态仿真，电路输出与数值探针实现余弦相似度 0.999983；并在合成四类任务中闭环训练，三种实现（数字 NS5 / 行为流 / SPICE 电路）均达 CE<0.01；本质区别在于首次通过电路级仿真 + 训练闭环双重验证物理可实现性。

## 方法详解
1. **动量累积与 Nesterov 加速**：每步 $m_t=\mu m_{t-1}+g_t$，$u_t=g_t+\mu m_t$，对 $u_t$ 做正交化得 $O_t$，更新 $W_{t+1}=(1-\text{lr}\cdot\lambda)W_t-\text{lr}\cdot s\cdot O_t$。
2. **连续时间流定义**：固定 $\hat{M}=u_t/\alpha(u_t)$，令 $X(0)=0$ 演化 $\dot{X}=\hat{M}-XX^T\hat{M}$，平衡点 $X=\hat{M}(\hat{M}^T\hat{M})^{-1/2}$ 即极因子 $UV^T$。
3. **奇异值动力学**：对满秩 $u_t=U\Sigma V^T$，各奇异值满足 $\dot{d}_i=\hat{s}_i(1-d_i^2)$，解析解 $d_i(t)=\tanh(\hat{s}_i t)$；非零奇异值趋向 1，零奇异值保持 0，达到极因子。
4. **离散 Euler 近似**：$X_{k+1}=\Pi_{[-r,r]}[X_k+\eta(\hat{M}-X_kX_k^T\hat{M})]$，clip 建模电路电源轨（报告中未激活）。
5. **随机探针实现**：每步用 $K$ 个独立 ±1 探针向量，每次迭代 $p=\hat{M}v$（读 $\hat{M}$）、$q=X^Tp$（互易读 $X$）、$r=Xq$（读 $X$）、$\Delta X=\eta(p-r)v^T$（rank-1 写）；总预算 $B=3KT$ 次数组遍，$\mathbb{E}[vv^T]=I$ 保证期望等于密集 Euler 步。
6. **归一化选择**：主实验用 $\alpha(u_t)=\sigma_{\max}(u_t)$；归一化消融改用 $\|u_t\|_F$，并将 $T$ 加倍至 800 补偿收敛减速。
7. **自适应停止**：相对变化低于 $3\times10^{-3}$ 时终止，平均 96.7 步（vs 固定 400 步），代价为 CE +0.030（单 seed）。

## 实验与结果
- **数据集**：FineWeb（Penedo et al., 2024），序列长度 256，batch size 24。
- **模型**：12 层、width-128 Transformer，10.95M 参数；另测试 width 192/256/384（26.4M/59M）。
- **基线**：NS5（五步 Newton–Schulz，数字控制）、Exact SVD、Direct Reuse（每 $S$ 步重算一次 NS5）、PolarExpress。
- **主要结果**：
  - 密集流（$T=400,\ \eta=0.5$，9 seed）：CE = $5.0209\pm0.0054$，相对 NS5（$5.0124\pm0.0081$）ΔCE = **+0.0085**，TOST 等价性检验上界 0.0142 < δ=0.0426（$p=2.4\times10^{-8}$）。
  - 探针流（$K=32,\ \eta=0.15,\ T=417$，预算≈40k 遍/matrix，2 seed）：CE = **5.0313**，ΔCE = **+0.0188**。
  - Exact SVD（3 seed）：CE = $5.0119\pm0.0014$，几乎与 NS5 持平，说明有限时间近似损失很小。
  - 直接复用（$S=2$）：ΔCE = **+0.0426**（等价性检验 margin 来源），揭示动量子空间旋转导致的 stale 问题。
- **归一化消融**：Frobenius + $T=800$（4 seed）：CE = $5.0170\pm0.0083$，通过等价性检验（上界 0.0142）。
- **设备误差鲁棒性**：20% 持久增益误差下 probe 流余弦变化≤0.011；10% 增益误差时 NS5 余弦降至 0.40–0.45，PolarExpress 两矩阵发散，而 probe 流仍稳定。
- **电路仿真**：ngspice 8×8 电阻阵列（8-bit 差分，0.4% mismatch），瞬态 600 周期，与数值探针余弦 0.999983，相对 Frobenius 距离 0.0059。
- **闭环训练**：四类合成任务（batch=8，300 步）：NS5/行为流/SPICE 电路分别在 43/42/42 步达 CE<0.01，最终 CE=0.0000（100% 准确率）。
- **能量估算**（GPT-2 124M，rank-scaled 60k–240k 遍/matrix）：模拟 0.45–1.04 J/step（ReRAM 下限） vs 数字 NS5 1.38–3.00 J/step；break-even 约 104–517 遍/rank。

## 相关工作脉络
1. **Muon（Jordan et al., 2024; Liu et al., 2025）**：原始密集 NS 正交化优化器，性能优但硬件不可实现；本文沿用其训练协议与动量结构，仅在正交化层替换。
2. **PolarExpress（Amsel et al., 2026, ICLR 2026）**：最优矩阵符号方法实现极因子，在器件增益误差 10% 下两矩阵发散；Physical Muon 连续流对增益误差稳健（余弦变化≤0.011）。
3. **Gram-Newton–Schulz（Zhang et al., 2026）**：硬件感知的快速 NS 多项式；本文与之正交——不追求多项式加速，而是改用连续时间流表达。
4. **CacheMuon（Dev et al., 2026）**：用历史动量近似极因子；本文 ablation 表明直接复用 NS5 输出（$S=2$）导致 +0.0426 CE 惩罚，动量子空间快速旋转是根因。
5. **OLion（Wang & Achour, 2026）**：结合谱与 $\ell_\infty$ 隐式偏置；附录 A 比较显示 OLion CE=4.9263 优于 Muon 的 4.9190，但 OLion 同样依赖密集运算，未解决硬件适配问题。
6. **Equilibrium Propagation（Scellier & Bengio, 2017）**：通过物理松弛计算梯度；Physical Muon 互补于该思路，专注优化器层面的模拟可实现正交化。

## 局限性与未来方向
1. **探针预算外推不确定性**：当前 40k 遍/matrix 预算在 width-128 上验证，向 rank 768（GPT-2 scale）线性外推，尚未在真实大矩阵上闭环训练验证。
2. **闭环训练仅在合成任务验证**：电路-in-the-loop 训练用了 4 类小合成任务，未在大语言模型训练中接入真实器件误差。
3. **Frobenius 归一化需更多 seed 统计**：目前仅 4 seed，多 seed 预算-质量权衡未充分量化。
4. **未与物理梯度计算集成**：未来需将正交化与通过物理松弛获取梯度（如 Equilibrium Propagation）的机制融合，实现端到端物理训练。
5. **自适应停止在单 seed 验证**：相对变化阈值 $3\times10^{-3}$ 的平均 96.7 步仅在单一 seed 下测试。

## 研究启发与可借鉴点
1. **连续时间流替代多项式迭代**：将离散 NS 迭代改写为 ODE $\dot{X}=M-XX^TM$，利用 $\tanh$ 饱和动力学自然实现奇异值缩放到 1；该思路可迁移至其他需要矩阵函数逼近的硬件友好算法。
2. **Hutchinson 探针 + 互易读取的工程范式**：用 $\mathbb{E}[vv^T]=I$ 将密集更新期望化为 rank-1 外积，配合电阻阵列的天然互易性（$X^Tp$ 与 $Xq$ 读同一阵列），为模拟硬件上的矩阵运算提供了通用模板。
3. **等价性检验（TOST）用于硬件算法验证**：以"直接复用惩罚"（+0.0426）作为 practical equivalence margin，用两单侧检验证明密集流与 NS5 统计等价，为硬件-数字对比提供了严谨的评估框架。
4. **零初始化对奇异值动力学的意义**：从零出发使 $X$ 始终落在 $\hat{M}$ 的奇异向量基中，各奇异值独立演化；这一设计保证了物理可实现性，可作为后续硬件算法设计的先验约束。
5. **设备误差鲁棒性分析范式**：分别注入持久增益误差、偏移误差、写入噪声，量化对方向余弦的影响；该方法可直接复用为未来模拟优化器的硬件容错基准。

## 关键术语表
**Physical Muon**：将 Muon 优化器正交化步骤转化为连续时间流平衡态计算的模拟存内计算版本。
**Newton–Schulz (NS) 迭代**：通过 $X_{k+1}=X_k(1.5I-0.5X_k^TX_k)$ 等多项式迭代逼近矩阵极因子的数字算法。
**极因子（Polar Factor）**：对满秩矩阵 $M=U\Sigma V^T$，极因子为 $UV^T$，是将奇异值全部映射为 1 的最优正交近似。
**Hutchinson 探针**：用随机向量 $v$（独立 ±1）估计矩阵二次型或乘积的无偏蒙特卡洛方法。
**互易读取（Reciprocal Read）**：电阻交叉点阵列中，从行读列与从列读行使用相同存储单元的特性，用于实现转置矩阵-向量乘。
**Rank-1 写入**：更新 $\Delta X = \eta \cdot a \cdot b^T$，每个单元仅依赖同行信号与同列信号，适配存内计算的局部写入约束。
**TOST（Two One-Sided Tests）**：用于统计等价性检验的假设检验框架，证明两组结果差异小于预设 margin。
**Frobenius 归一化**：用 $\|u_t\|_F$ 代替 $\sigma_{\max}(u_t)$ 对动量矩阵归一化，硬件代价更低但收敛速度减慢约 2×。

## 可复现要素
- **数据集**：FineWeb（Penedo et al., 2024），公开可用。
- **代码/权重**：论文未提及开源代码或权重（arXiv 版本无代码链接）。
- **关键超参**：Transformer depth=12、width=128、steps=2500、batch=24、seq_len=256；Muon momentum=0.95、lr=0.016（矩阵参数）、AdamW lr=0.001、wd=1e-4；密集流 $\eta=0.5,\ T=400$；探针流 $K=32,\ \eta=0.15,\ T=417$；Frobenius 消融 $T=800$。
- **电路仿真**：ngspice，8-bit 差分电阻阵列，8×8 块，64 电容存 $X$，迭代周期 1 µs，RC  settle 40 ns（附录 E 提供详细电路参数）。
