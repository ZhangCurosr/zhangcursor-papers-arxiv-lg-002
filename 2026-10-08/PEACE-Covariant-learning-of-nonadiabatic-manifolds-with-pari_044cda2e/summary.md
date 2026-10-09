---
title: "PEACE-Covariant-learning-of-nonadiabatic-manifolds-with-pari"
source: https://arxiv.org/pdf/2610.09576v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:52:46"
field: "量子化学机器学习"
keywords: ["非绝热分子动力学", "等变神经网络", "宇称对称性", "自旋-轨道耦合", "表面跳跃动力学", "潜势哈密顿量"]
innovations: ["宇称解析潜势哈密顿量与学习电子联络的统一等变框架", "通过协变导数统一保证能量-力-非绝热耦合的物理一致性", "扩展到自旋-轨道耦合和系间窜越动力学模拟"]
benchmarks: ["C2H4 C3H6 C4H8 alkene benchmarks", "CH2NH2+ ablation benchmark", "CH2S SOC benchmark", "SPaiNN SchNarc Exciting DeePMD baselines"]
---

# 论文速读：PEACE-Covariant-learning-of-nonadiabatic-manifolds-with-pari

## 一句话总结
PEACE 提出了一种结合宇称等变潜势哈密顿量与学习电子联络的equivariant框架，通过在统一电子表示中显式融入电子对称性和几何框架演化，实现了激发态能量、力和非绝热耦合的高精度预测，并显著加速了光化学动力学模拟。

## 研究问题与动机
1. **非绝热耦合学习的固有挑战**：非绝热分子动力学需同时准确描述电子能量面、力和态间耦合，但耦合项依赖电子态相位、在能隙闭合时发散，且可能呈现与几何相位相关的双值行为。
2. **已有方法的物理结构缺陷**：SchNarc 通过相位不变训练解决相位歧义，但其标量梯度耦合 formulation 施加保守场近似，会抑制平面几何下宇称允许的面外分量；SPaiNN 和 Exciting DeePMD 虽支持更灵活的耦合预测，但能量和耦合通过独立 readout 预测，未通过共享哈密顿量显式关联，缺乏物理一致性。
3. **哈密顿量框架下的对称性约束**：非对角相互作用项在反射变换下无需不变（与能量不同），且导数耦合包含哈密顿量本征向量旋转和潜在电子基框架变化两部分贡献；缺乏对称性适应的表示会导致交叉结构和弛豫动力学失真。

## 核心贡献（创新点）
1. **宇称解析的潜势哈密顿量**：为潜基态分配固定宇称标签（$p_a \in \{+1, -1\}$），相同宇称块由偶标量特征构建，相反宇称块由奇伪标量特征构建，从而在反射对称几何处自然为零并在对称破缺位移上允许线性耦合起始——与仅用不变特征的方法相比，提供了对称性适配的局部能隙开启描述。
2. **学习的电子联络（Covariant Connection）**：显式学习潜电子框架随核运动的演化（$\mathbf{B}_\mu$），使导数耦合同时包含哈密顿量本征向量旋转贡献和框架变化贡献；将刚体运动与内禀运动向量场通过正则化投影组合——与 SchNarc 的保守场近似本质不同，捕捉了完整几何依赖。
3. **协变导数统一读出物理观测量**：通过公式 $\mathbf{K} = \mathbf{C}^T(\nabla_\mathbf{R}\mathbf{H} + [\mathbf{B}_\mu, \mathbf{H}])\mathbf{C}$ 统一计算力和能量-隙加权非绝热耦合，确保能量-力-耦合间的物理一致性；对角的保守力关系 $F_{i\mu} = -\partial_\mu E_i$ 在联络存在时仍保持——与分别独立预测能量和耦合的方法有本质区别。
4. **扩展至自旋-轨道耦合的异核框架**：保留单重态和三重态独立哈密顿量与联络，新增等变 SOC readout 预测笛卡尔耦合矢量，构建复 Hermitian SOC 矩阵，实现对系间窜越过程的动力学模拟——现有方法（SchNarc/SPaiNN）均未处理此扩展。

## 方法详解
- **架构总览**：共享 O(3)-等变张量积编码器 → 两个独立参数化的等变自注意力分支（Hamiltonian 分支 + Connection 分支）。
- **宇称约束构建**：对空间正交变换 $\mathbf{Q}$，$\det\mathbf{Q}=+1$ 时 $\rho(\mathbf{Q})=\mathbf{I}$，$\det\mathbf{Q}=-1$ 时 $\rho(\mathbf{Q})=\mathbf{P}$；哈密顿量满足 $\mathbf{H}(\mathbf{Q}\mathbf{R}) = \rho(\mathbf{Q})\mathbf{H}(\mathbf{R})\rho(\mathbf{Q})^T$；相同宇称对使用偶标量(0e)通道和极矢量(1e)联络通道，相反宇称对使用奇标量(0o)和轴矢量(1o)通道。
- **协变导数定义**：$\mathbf{G}_\mu = \partial_\mu \mathbf{H} + [\mathbf{B}_\mu, \mathbf{H}]$；绝热能 $E_i$ 为 $\mathbf{H}$ 本征值，力 $F_{i\mu} = -(\mathbf{K}_\mu)_{ii}$，SNAC $b_{ij,\mu} = (E_j - E_i)d_{ij,\mu}$ 均由同一协变导数 $\mathbf{K}_\mu$ 读出。
- **联络分解**：$\mathbf{B}_{ab}$ 结合学习到的未投影刚体运动场 $\mathbf{X}_{ab}^{\mathrm{rig}}$ 和内禀运动场 $\mathbf{X}_{ab}^{\mathrm{int}}$，经正则化投影 $\Pi_{\mathrm{rig}}^{(\epsilon)}$ 和 $\Pi_{\mathrm{int}}^{(\epsilon)}=\mathbf{I}-\Pi_{\mathrm{rig}}^{(\epsilon)}$ 组合，并乘以固定缩放因子 $\alpha_\mathrm{rig},\alpha_\mathrm{int}$。
- **SOC 扩展**：$\mathbf{H}_\mathrm{full}=\mathrm{diag}(\mathbf{H}_S,\mathbf{H}_T,\mathbf{H}_T,\mathbf{H}_T)+\mathbf{V}_\mathrm{SO}$，其中 $\mathbf{V}_\mathrm{SO}$ 由各对单重-三重或不同根三重-三重间的等变 readout 预测笛卡尔 SOC 矢量（同宇称用轴矢量1o，反宇称用极矢量1e），再与固定自旋角系数组合为复 Hermitian 矩阵；在 MCH 基下 $\mathbf{V}^\mathrm{MCH}=\mathbf{C}_\mathrm{sf}^\dagger\mathbf{V}_\mathrm{SO}\mathbf{C}_\mathrm{sf}$。
- **损失函数**：$\mathcal{L}=\lambda_E\mathcal{L}_E+\lambda_F\mathcal{L}_F+\lambda_\mathrm{SNAC}\mathcal{L}_\mathrm{SNAC}+\lambda_\Delta\mathcal{L}_\Delta+\mathcal{L}_\mathrm{parity}$；SNAC 损失在所有态对间取全局符号一致的最优对齐；宇称辅助损失 $\mathcal{L}_\mathrm{parity}$ 对具有可信标签的反射对称构型施加宇称 sector 谱和符号能量隙监督。

## 实验与结果
- **数据集与基线**：以 SA(3)-CASSCF(2,2)/cc-pVDZ 为参考， benchmark 体系包括 $\mathrm{C_2H_4}$（7500）、$\mathrm{C_3H_6}$（7500）、$\mathrm{C_4H_8}$（15000）；消融实验用 $\mathrm{CH_2NH_2^+}$（4000 MR-CISD/aug-cc-pVDZ）；SOC 扩展用 $\mathrm{CH_2S}$（4000/200/503 split, CASSCF(6,5)/def2-SVP）。对比基线为 SPaiNN（SchNet/PaiNN）、Exciting DeePMD 和 SchNarc。
- **静态预测 SOTA**：PEACE 在四种体系上全面超越基线：
  - $\mathrm{C_2H_4}$：能量 MAE = **0.008 eV**（SPaiNN 0.026/0.026），力 MAE = **0.036 eV/Å**，NAC MAE = **0.028 Bohr⁻¹**
  - $\mathrm{C_3H_6}$：能量 MAE = **0.013 eV**，力 MAE = **0.042 eV/Å**，NAC MAE = **0.016 Bohr⁻¹**
  - $\mathrm{C_4H_8}$：能量 MAE = **0.022 eV**，力 MAE = **0.054 eV/Å**，NAC MAE = **0.018 Bohr⁻¹**
  - $\mathrm{CH_2NH_2^+}$：能量 MAE = **0.026 eV**，力 MAE = **0.109 eV/Å**，NAC MAE = **0.053 Bohr⁻¹**
- **动力学精度**：$\mathrm{C_2H_4}$/$\mathrm{C_3H_6}$/$\mathrm{C_4H_8}$ 的 $\mathrm{S_1}$ 有效衰减速率常数分别为 **68.5/51.4/67.8 fs**，与参考值 **68.8/53.3/66.9 fs** 相差最大仅 **1.9 fs**；消融实验显示：无联络时 SNAC MAE 升至 0.410 eV/Å，无 H-odd 块时升至 0.257 eV/Å，两者皆无时升至 0.687 eV/Å，且 $\mathrm{S_2}$ 弛豫被严重抑制。
- **SOC 扩展结果**：$\mathrm{CH_2S}$ 测试集上能量 MAE = 1.113 meV，力 MAE = 4.941 meV/Å，SNAC MAE = 0.933 meV/Å，复 SOC 矩阵元 MAE = **0.265 cm⁻¹**，RMSE = **0.397 cm⁻¹**；3 ps 动力学表明 $\mathrm{S_1}$ 占比 95.4%，$\mathrm{T_1}$ 仅 2.8%。
- **计算效率**：单卡 NVIDIA H100 GPU 上 FP64 推理 **7.66–12.44 ms/几何**，对比 CASSCF 六核 CPU 工作线程 **6.77–13.39 s**，加速比约 **~1000×**。

## 相关工作脉络
1. **SchNarc**（Westermayr et al., 2020）：基于 SchNet + SHARC 的激发态动力学框架，通过相位不变训练处理耦合相位歧义，但采用保守场近似，忽略宇称允许的出平面分量；PEACE 通过共变联络打破此限制。
2. **SPaiNN**（Mausenberger et al., 2024）：结合 SchNet/PaiNN 的等变消息传递网络，直接预测矢量耦合；PEACE 通过共享潜势哈密顿量将能量和耦合统一关联，物理一致性更强。
3. **Exciting DeePMD**（Dupuy & Maitra, 2024）：学习相位不变的耦合双线性对（coupling dyads），仍与能量预测解耦；PEACE 通过协变导数保证力的保守性且统一能量-耦合关系。
4. **DANN**（Axelrod et al., 2022）：首个学习电子哈密顿量的 diabatic ANN 框架；PEACE 进一步引入宇称对称性约束和显式电子联络，弥补了缺少框架演化描述和宇称对称适配的不足。
5. **等变 ML 势场基础**（NequIP/ Allegro/F3Net 等）：PEACE 继承 O(3)-等变张量积编码和等变自注意力机制，将其从基态势能面扩展至多态非绝热流形。

## 局限性与未来方向
1. **目前仅适用于孤立气相分子**，尚未扩展至溶剂化环境或分子组装体，限制了其在溶液光化学和分子材料中的直接应用。
2. **数据集规模和电子态数有限**：当前 alkene 和 $\mathrm{CH_2NH_2^+}$ 仅涉及2–3个低激发态，对包含更多密集态（如芳香体系）的系统泛化能力有待验证。
3. **宇称标签部分依赖参考数据的对称选择规则推断**，对于缺乏高对称性的非共面体系，自动分配可靠宇称标签仍存在挑战。
4. **未来方向**：① 扩展至凝聚相环境（溶剂/聚集体）；② 引入主动学习策略以降低参考计算成本；③ 探索状态重叠等额外电子结构信息作为辅助训练监督。

## 研究启发与可借鉴点
1. **对称性约束驱动的特征通道分离**（偶标量/奇伪标量分通道）可迁移至其他需要处理反射对称性的量子化学 ML 任务，如振动光谱或 Rashba 耦合预测。
2. **共变导数统一读出能量的保守力关系**这一思路可推广到其它需要保持物理守恒律的多体学习框架（如含时 DFT 或极化连续模型耦合体系）。
3. **刚体/内禀运动分解+正则化投影**的联络构造方式，为解决"刚性整体运动不应贡献真实耦合"这一长期问题提供了实用范式。
4. **消融实验设计**（分别移除联络和 H-odd 块，比较训练 loss 与动力学行为的一致性）展示了一个重要方法论原则：低训练 loss 不能保证物理保真度，必须在交叉拓扑和系综弛豫层面联合评估。
5. **SOC 扩展架构**（独立哈密顿量分支 + 等变矢量 readout + 固定自旋角系数组合）可作为多自旋态耦合学习任务的通用模板。

## 关键术语表
**Nonadiabatic coupling（NAC）**：描述绝热电子态间因核运动引起的态间耦合强度，正比于电子波函数对核坐标的导数，是非绝热跃迁的核心驱动项。
**Smoothed NAC（SNAC）**：经能量隙加权处理的非绝热耦合，消除接近锥形交叉时的奇异性，数值更稳定。
**Parity-eq uivariant**：模型输出在空间反射变换下按给定的宇称标签矩阵进行共变变换的性质，确保哈密顿量块满足正确的对称性选择规则。
**Covariant derivative（协变导数）**：在变化的基底（电子联络）下定义的导数算子，$\nabla_\mu = \partial_\mu + [\mathbf{B}_\mu, \cdot]$，保证物理观测量在不同规范下具有一致性。
**Surface hopping（表面跳跃）**：基于 Tully 经典力学的混合量子-经典动力学方法，核沿绝热面运动同时在态间以 NAC 诱导的概率跳跃。
**Electronic connection（电子联络）**：表征潜电子基框架随核坐标变化而旋转的反对称矩阵 $\mathbf{B}_\mu$，补充了仅靠哈密顿量本征向量旋转无法描述的耦合贡献。
**Diabatic Hamiltonian（ diabatic 哈密顿量）**：在近似与核坐标无关的基底（ diabatic 基）下表示的电子哈密顿量，其对角元为 diabatic 能，非对角元为态间耦合。
**MCH basis（Molecular Coulomb Hamiltonian basis）**：包含自旋-轨道耦合的绝热电子态基底，SOC 矩阵在此基下非对角，用于处理单重态-三重态混合。

## 可复现要素
- **数据集**：$\mathrm{C_2H_4}$（6000/500/1000）、$\mathrm{C_3H_6}$（6000/500/1000）、$\mathrm{C_4H_8}$（12000/2000/1000）、$\mathrm{CH_2NH_2^+}$（2500/250/1250）、$\mathrm{CH_2S}$（4000/200/503），基于 SA(3)-CASSCF(2,2)/cc-pVDZ 或 MR-CISD/aug-cc-pVDZ 参考计算。
- **代码**：开源，GitHub https://github.com/solvendus/peace。
- **权重**：论文声明已提供训练模型（具体链接见 GitHub）。
- **关键超参**：64 channels/parity irreps，$l_\mathrm{max}=2$，4 层等变消息传递，每分支 2 层等变注意力，AdamW，1000 epochs，batch size 64，余弦学习率 $10^{-3}\to10^{-4}$，四项损失权重均为 1。
