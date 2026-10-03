---
title: "Physical-Muon-Orthogonalization-as-an-Equilibrium-Computatio"
source: https://arxiv.org/pdf/2609.37525v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 18:34:14"
field: "硬件感知优化器与存内计算训练"
keywords: ["Physical Muon", "optimizers for neuromorphic computing", "orthogonalization", "in-memory computing", "resistive arrays", "continuous-time matrix flow", "transformer training"]
innovations: ["将 Muon 正交化转化为连续时间流平衡求解以适配电阻阵列", "用随机探针将密集增量展开为矩阵向量读与局部秩-1写", "通过训练与 ngspice 电路闭环联合验证模拟正交化的可行性"]
benchmarks: ["FineWeb language modeling", "GPT-2 124M energy scaling", "8x8 and 128-wide momentum blocks", "four-class synthetic circuit-in-the-loop task"]
---

# 论文速读：Physical Muon: Orthogonalization as an Equilibrium Computation

## 一句话总结
提出 Physical Muon，将 Muon 优化器的牛顿–舒尔茨正交化操作转化为可通过电阻阵列的**矩阵向量读取**和**局部秩-1写入**执行的连续时间流平衡求解，为模拟存内计算的 Transformer 训练提供能量高效的优化器实现。

## 研究问题与动机
- **物理神经网络需要兼顾学习效率与硬件可实现的优化器**：电阻阵列擅长矩阵向量乘法和局部更新，但缺乏原生的密集矩阵乘法。
- **SGD 在 Transformer 上训练性能显著落后于 Adam**，而 Adam 系自适应更新对模拟偏置（analog bias）敏感、易不稳定。
- **Muon 训练效率优于 AdamW，但其标准 Newton–Schulz（NS）正交化依赖密集矩阵乘积**，无法直接映射到电阻交叉阵列的读取/写入原语。
- 目标是以**硬件可实现的近似**替代 NS 迭代，同时保持与数字优化器相近的训练质量。

## 核心贡献（创新点）
1. **将正交化表述为连续时间流的平衡计算**：构造 $\dot{X} = \hat{M} - X X^T \hat{M}$ 的动力学，使 $X$ 的稳态收敛到 $\hat{M}$ 的极因子，避免显式 SVD/NS 迭代。
2. **用随机探针将密集增量展开为矩阵向量读 + 局部秩-1写**：通过 Hutchinson 型探针 $v$，将 $E[\Delta X] = \eta(\hat{M} - XX^T\hat{M})$ 用三次数组遍历 $(Mv, X^Tp, Xq)$ 实现，适配电阻阵列的互易读取与写单元。
3. **给出首次面向 Muon 正交化的模拟执行模型并通过训练与电路仿真联合验证**：在 10.95M 参数 Transformer 上证明密集流与探针流均接近 NS5；在 $8\times8$ 动量块的 ngspice 瞬态仿真与 4 类合成任务闭环训练中复现相同训练行为。
4. **系统量化设备误差下的方向鲁棒性与能耗权衡**：探针流对 20% 持续增益误差的方向余弦下降 $\le 0.011$，而 NS5/PolarExpress 在 10% 增益下明显退化；在 GPT-2 124M 量级下，60k–240k 次/矩阵的探针预算可实现与数字 NS5 可比甚至更低的单步正交化能耗。

## 方法详解
- **Muon 基线**：每步累积动量 $m_t = \mu m_{t-1}+g_t$，Nesterov 动量 $u_t=g_t+\mu m_t$，对其做正交化得 $O_t$，再以 $W_{t+1}=(1-\text{lr}\,\lambda)W_t-\text{lr}\,s O_t$ 更新权重。
- **平衡流方程**：固定 $\hat{M}=u_t/\alpha(u_t)$，从 $X(0)=0$ 出发演化 $\dot{X}=\hat{M}-XX^T\hat{M}$；按 SVD 分析知各奇异模独立演化 $d_i(t)=\tanh(\hat{s}_i t)$，非零模均趋于 1，极限为极因子 $UV^T$。
- **数值离散**：采用 $T$ 步欧拉迭代 $X_{k+1}=\Pi_{[-r,r]}[X_k+\eta(\hat{M}-X_kX_k^T\hat{M})]$， clipping 模拟电路电源轨（实验中未激活）。
- **正则化选择**：用 Frobenius 范数 $\|u_t\|_F$ 代替 $\sigma_{\max}$ 作标量归一，硬件更易求和；实测比值中位数为 2.07，因此将解预算翻倍补偿收敛变慢。
- **探针近似**：每次迭代用 $K$ 个独立 $\pm 1$ 探针 $v$，计算 $p=\hat{M}v,\ q=X^Tp,\ r=Xq,\ \Delta X=\eta(p-r)v^T$；期望值等于密集欧拉增量。每个优化步需 $B=3KT$ 次数组遍历。
- **互易读取**：$X^Tp$ 与原 $X$ 阵列反向读取相同 $X$，单元 $(i,j)$ 仅依赖行信号 $p_i-r_i$ 与列信号 $v_j$，符合电阻交叉点器件特性。

## 实验与结果
- **主实验配置**：12 层、宽 128、10.95M 参数 Transformer，在 FineWeb 上训练 2,500 步（batch=24、seq=256）；矩阵参数使用 Muon（momentum 0.95、lr 0.016），其余参数用 AdamW。
- **评估指标**：最佳验证交叉熵（CE）。NS5 作为对照；等价检验采用 TOST 且 margin $\delta=0.0426$（来自“复用 NS5 输出一步”的退化代价）。
- **主要数字**
  - NS5：CE $5.0124\pm0.0081$
  - 密集流（$\eta=0.5,T=400,n=9$）：CE $5.0209\pm0.0054$，$\Delta$CE $+0.0085$，95% 单侧上界 $0.0142$，**通过等价检验**。
  - 探针流（$K=32,\eta=0.15,T=417$，$B\approx40\mathrm{k}$）：CE $5.0313$，相对 NS5 增加 $+0.0188$。
  - Frobenius 归一化 + 翻倍步数（$T=800,n=4$）：CE $5.0170\pm0.0083$，通过同一等价检验。
- **泛化测试**：宽度 128–384（10.95M–59M）、batch 24→3、学习率与时长变化时，密集流相对 NS5 的 CE 增量始终在 $[0.001,0.022]$ 区间内。
- **电路验证**：$8\times8$ 动量块的 ngspice 瞬态（600 周期）与探针数值结果余弦 $0.999983$、相对 Frobenius 距离 $0.0059$；在 4 类合成任务闭环训练中，数字 NS5、行为流、SPICE 流均在 $\sim42$ 步内达到 CE $<0.01$ 并 100% 训练准确率。
- **最强结果与提升**：密集流以极小代价（$+0.0085$ CE）获得与 NS5 统计等价的表现；在 40k 探针预算下仍保持 $+0.0188$ 的微小差距，方向余弦达到 0.9 以上的 NS5 对齐。

## 相关工作脉络
1. **Muon / NS 正交化路线**：Jordan 等提出 Muon，Liu 等验证其在 LLM 训练的可扩展性；本文用**连续流+探针**替代密集 NS，首次面向电阻阵列的可实现形式。
2. **硬件友好的正交加速**：Amsel 等 PolarExpress、Zhang 等 Gram-Newton–Schulz 优化多项式/硬件感知实现；本文侧重**物理原语**（读+秩-1写）而非多项式系数设计。
3. **含时缓存近似**：Dev 等 CacheMuon 利用历史预条件加速；本文通过**从零初始化流平衡**避免历史依赖，代价是可调的求解预算。
4. **SGD/Adam 在物理网络中的局限**：Gokmen 等与 Sebastian 等指出电阻阵列原生支持局部更新与矩阵向量求和；本文以 Muon 的结构化正交弥补 SGD 在 Transformer 上的性能缺口。
5. **均衡传播/物理训练**：Scellier & Bengio 的 Equilibrium Propagation 等通过物理弛豫直接获得梯度；本文不改变梯度计算，只改变**每步正交化算子的执行范式**。
6. **Muon 谱动力学理论**：Beneventano 等、Peyré 等从谱 Wasserstein/哈密顿角度分析 Muon 轨迹；本文提供与之互补的**模拟实现基线**。

## 局限性与未来方向
- **更大矩阵的探针预算外推**：当前 40k 预算在 128 宽矩阵上有效，但 GPT-2 量级（rank 768）的预算需线性外推，尚待真实闭环验证。
- **设备误差下的长期闭环训练未完全验证**：静态动量矩阵上的方向余弦测量充分，但含读写噪声、漂移的端到端训练稳定性仍需实测。
- **能耗估计排除缓冲搬运与读出开销**：估算仅计入正交化 MAC 成本，未计 $\hat{M}$ 加载、$X$ 读出、1.7 GB 共享缓冲流量，实际功耗可能更高。
- **自适应停止与精度–成本权衡尚未系统扫参**：目前单一阈值（相对变化 $3\times10^{-3}$）实验有效，但未在多尺度下建立 Pareto 曲线。
- **与物理梯度计算的集成空白**：论文承认需与基于物理松弛的梯度提取管线结合，方能实现端到端物理训练。

## 研究启发与可借鉴点
1. **以连续动力系统替换离散矩阵迭代**：对任何依赖 SVD/极分解的优化器，均可尝试将其等价表述为李雅普诺夫型流 $\dot{X}=M-XX^TM$ 并离散化，从而用迭代读取逼近。
2. **Hutchinson 探针 + 互易读写的硬件分解范式**：将 $\hat{M}-XX^T\hat{M}$ 展开为 $E[(p-r)v^T]$ 后，仅需行向量误差与列探针的外积写入，天然适配存内计算的 2D 阵列结构，可作为后续硬件优化器的通用模板。
3. **用“复用代价”建立等价检验 margin**：以 S=2 复用 NS5 带来的 CE 退化（+0.0426）作为等价边界，使数值近似与数字对照的比较具备统计严谨性，适合硬件替换验证。
4. **Frobenius 归一化替代 $\sigma_{\max}$ 的工程实践**：以加法归一化换取可接受的收敛减速（中位 2×），并通过翻倍求解步数补偿，为资源受限模拟前端提供一种低功耗正则方案。
5. **电路-in-the-loop 闭环验证链路**：从行为模型到 ngspice 瞬态再到合成任务训练，三级递进验证策略可作为后续物理优化器的标准验证流程参考。

## 关键术语表
- **Muon optimizer**：对动量矩阵做正交化（极分解近似）的优化器，比 AdamW 在 Transformer 上具有更优训练计算效率。
- **Newton–Schulz 迭代**：通过多项式迭代逼近矩阵极因子/符号函数的标准数值方法。
- **极因子（polar factor）**：满秩矩阵 $M=U\Sigma V^T$ 的极分解中保留奇异向量、奇异值全部置 1 的因子 $UV^T$。
- **Hutchinson 探针**：用独立 Rademacher（±1）随机向量估计矩阵二次型/迹的无偏随机近似技术。
- **互易读取（reciprocal read）**：电阻交叉阵列中正向与反向矩阵向量乘法共享同一存储电导阵列的读取方式。
- **秩-1 写入（rank-1 write）**：以行误差向量与列探针向量的外积形式更新阵列单元的局部写入操作。
- **TOST 等价检验**：Two One-Sided Tests，用于统计验证两组均值差异落在预设等价区间内的假设检验框架。
- **Circuit-in-the-loop**：将器件级电路仿真嵌入训练循环，以验证硬件约束下算法的实际训练行为。

## 可复现要素
- **数据集**：FineWeb（Penedo et al., 2024）——论文未明示开放链接，但为已知开源数据集。
- **代码/权重**：论文未声明开源仓库与模型权重；训练协议与超参在附录 C 详细列出，可复现数字实验。
- **关键超参**：Muon momentum=0.95、lr=0.016；AdamW lr=1e-3、weight decay=1e-4；密集流 $\eta=0.5,T=400$；探针流 $K=32,\eta=0.15,T=417,B=3KT\approx40\mathrm{k}$；Frobenius 归一化时 $T=800$。
- **电路参数**：8-bit 差分电阻阵列、64 电容存储、单迭代 1 µs（含 40 ns 读建立）、ngspice 600 周期瞬态；NMOS 参数 $V_{T0}=0.7$ V、$K_P=0.1$ mA/V$^2$、$W/L=100/100\ \mu$m。
