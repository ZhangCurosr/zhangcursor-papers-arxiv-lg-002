---
title: "PEACE-Covariant-learning-of-nonadiabatic-manifolds-with-pari"
source: https://arxiv.org/pdf/2610.09576v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:53:00"
field: "量子化学中的等变机器学习"
keywords: ["nonadiabatic molecular dynamics", "equivariant machine learning", "parity-resolved Hamiltonian", "covariant derivative", "spin-orbit coupling", "surface hopping", "excited-state dynamics"]
innovations: ["提出奇偶等变潜哈密顿量与学习电子联络的统一框架，通过协变导数同时给出保守力与能量gap加权SNAC", "将反射对称性显式编码到Hamiltonian块（0e/0o通道）与联络（1e/1o通道）中，允许对称破缺处态混合", "扩展至SOC，以等变读出预测Cartesian耦合矢量，实现单重态–三重态系间穿越模拟"]
benchmarks: ["SPaiNN (SchNet/PaiNN)", "Exciting DeePMD D yad", "SchNarc", "SA(3)-CASSCF(2,2)/cc-pVDZ alkene benchmarks", "MR-CISD/aug-cc-pVDZ CH2NH2+ benchmark", "CASSCF(6,5)/def2-SVP CH2S SOC benchmark"]
---

# 论文速读：PEACE-Covariant learning of nonadiabatic manifolds with parity-resolved Hamiltonians

## 一句话总结
PEACE提出了一种结合奇偶等变潜哈密顿量与学习电子联络的等变机器学习框架，通过统一的协变导数同时预测绝热能量、力与平滑非绝热耦合（SNAC），在烯烃光动力学和自旋–轨道耦合体系中实现了当前最优精度。

## 研究问题与动机
- 非绝热分子动力学需要沿轨迹系综反复计算多个电子态能量、力与态间耦合，计算成本高昂；equivariant ML势能在基态已很成功，但推广至激发态面临特有挑战。
- 学习非绝热耦合的难点超出单纯能量面预测：耦合依赖于电子态相位、在能隙趋近时可能发散，并可能存在与几何相位相关的双值行为；能量 gap 权重能消去显式的 inverse-gap 因子，但所得耦合场通常是非保守的。
- 既有方法如 SPaiNN 支持直接矢量耦合预测，Exciting DeePMD 学习相位不变耦合 dyad，SchNarc 通过相位不变训练解决相位歧义，但它们都以能量和耦合的独立读出预测二者，未通过共享电子哈密顿量显式关联。
- DANN 等学习电子哈密顿量的方法已证明可行性，但两个物理要素常被忽略：(1) 哈密顿矩阵元必须尊重电子态对称性——非对角相互作用不必在反射下不变；(2) 导数耦合包含哈密顿本征向量旋转与底层电子基组变化两部分的贡献，严格 diabatic 基在多原子有限子空间中通常不可得。

## 核心贡献（创新点）
1. **奇偶等变潜哈密顿量 + 学习电子联络的统一框架**：将对称允许的态混合（odd-parity Hamiltonian blocks）与电子框架随核运动的变化（learned connection）作为互补两部分纳入同一表示；与已有工作的本质区别在于，前人以能量与耦合的独立读出分别预测，本文通过共享协变导数使二者在物理上相互约束。
2. **反射对称性的显式奇偶建模**：对潜基态赋予固定奇偶标签，同奇偶块用偶标量（0e）特征构造、异奇偶块用奇赝标量（0o）特征构造；在反射不变几何处异奇偶矩阵元强制为零，沿对称破缺坐标的一阶项描述交叉处耦合 onset，同时保留非线性几何依赖。
3. **基于协变导数的统一物理读出**：$\mathbf{G}_\mu = \partial_\mu \mathbf{H} + [\mathbf{B}_\mu, \mathbf{H}]$，对角给出保守力 $F_{i\mu}=-\partial_\mu E_i$，非对角给出能量 gap 加权 SNAC $b_{ij,\mu}=(E_j-E_i)d_{ij,\mu}$；相比 SchNarc 的保守场近似和标量梯度公式，本文可恢复对称平面处允许的出平面分量。
4. **扩展至自旋–轨道耦合（SOC）与非绝热耦合的统一处理**：对单重态/三重态分别保留哈密顿与联络表示，增加等变 SOC 读出预测笛卡尔耦合矢量，组合固定自旋角系数构建复 Hermitian SOC 矩阵；实现 intersystem crossing 模拟。
5. **受控消融揭示"低训练损失≠物理保真"**：仅移除 connection 或仅移除 H-odd 块都会显著恶化 SNAC 误差与交叉拓扑，说明能量误差小并不能保证正确再现 crossing 结构与弛豫动力学。

## 方法详解
- **编码层**：原子种类 $Z$ 与笛卡尔坐标 $\mathbf{R}$ 构成分子图 $\mathcal{G}=(\mathcal{V},\mathcal{E})$；共享的 O(3)-等变张量积编码器结合消息聚合与等变更新构建分子特征。
- **双分支注意力**：两个独立参数的等变自注意力分支分别输出实对称潜哈密顿量 $\mathbf{H}(\mathbf{R})$ 与实反对称电子联络 $\mathbf{B}_\mu(\mathbf{R})$（$\mu=(A,\alpha)$ 为原子 A 的 Cartesian 分量）。
- **奇偶约束的 Hamiltonian**：每个潜基态有固定奇偶 $p_a\in\{+1,-1\}$，$\mathbf{P}=\mathrm{diag}(p_1,\ldots,p_{N_{\mathrm{latent}}})$。正交变换 $\mathbf{Q}$ 下 $\rho(\mathbf{Q})=\mathbf{I}$（$\det\mathbf{Q}=+1$）或 $\mathbf{P}$（$\det\mathbf{Q}=-1$），满足 $\mathbf{H}(\mathbf{Q}\mathbf{R})=\rho(\mathbf{Q})\mathbf{H}(\mathbf{R})\rho(\mathbf{Q})^T$。同奇偶对使用 0e（偶标量）通道、异奇偶对使用 0o（奇赝标量）通道。在反射不变几何 $g\mathbf{R}_s=\mathbf{R}_s$ 处，$H_{ab}(\mathbf{R}_s)=p_ap_b H_{ab}(\mathbf{R}_s)\Rightarrow H_{ab}=0$（异奇偶）；沿对称破缺位移 $q\mathbf{u}$，$H_{ab}(\mathbf{R}_s+q\mathbf{u})=qA_{ab}+O(q^3)$，一阶项描述交叉 onset。
- **联络的构造**：$\mathbf{B}_{ab}=\alpha_{\mathrm{rig}}\Pi_{\mathrm{rig}}^{(\epsilon)}\mathbf{X}_{ab}^{\mathrm{rig}}+\alpha_{\mathrm{int}}\Pi_{\mathrm{int}}^{(\epsilon)}\mathbf{X}_{ab}^{\mathrm{int}}$，经正则化投影组合刚体运动与内部运动矢量场；同奇偶对使用 1o（极向量）、异奇偶对使用 1e（轴向量）通道。
- **协变导数与物理读出**：$\mathbf{G}_\mu=\partial_\mu\mathbf{H}+[\mathbf{B}_\mu,\mathbf{H}]$；对角化 $\mathbf{H}\mathbf{C}=\mathbf{C}\mathbf{E}$，$\mathbf{K}_\mu=\mathbf{C}^T\mathbf{G}_\mu\mathbf{C}=\partial_\mu\mathbf{E}+[\mathbf{d}_\mu,\mathbf{E}]$；力 $F_{i\mu}=-(\mathbf{K}_\mu)_{ii}$，SNAC $b_{ij,\mu}=(\mathbf{K}_\mu)_{ij}=(E_j-E_i)d_{ij,\mu}$。
- **自旋–轨道耦合**：$\mathbf{H}_{\mathrm{full}}=\mathrm{diag}(\mathbf{H}_S,\mathbf{H}_T,\mathbf{H}_T,\mathbf{H}_T)+\mathbf{V}_{\mathrm{SO}}$；每个 singlet–triplet 或 distinct-root triplet–triplet 对经等变读出预测实 Cartesian SOC 矢量（同奇偶 1e、异奇偶 1o），组合固定自旋角系数成复 Hermitian 块；MCH 基下 $\mathbf{V}^{\mathrm{MCH}}=\mathbf{C}_{\mathrm{sf}}^\dagger\mathbf{V}_{\mathrm{SO}}\mathbf{C}_{\mathrm{sf}}$，$\mathbf{H}^{\mathrm{MCH}}=\mathbf{E}_{\mathrm{sf}}+\mathbf{V}^{\mathrm{MCH}}$。
- **损失函数**：联合训练于能量、力与 SNAC，MSE 损失 $\mathcal{L}_E,\mathcal{L}_F,\mathcal{L}_{\mathrm{SNAC}}$，外加能隙损失 $\mathcal{L}_\Delta$ 与奇偶辅助损失 $\mathcal{L}_{\mathrm{parity}}$；$\mathcal{L}_{\mathrm{SNAC}}$ 在每几何内对一组一致的状态符号 $\{s_i\}$ 最小化以处理相位不确定性。

## 实验与结果
- **数据集与基线**：烯烃 C₂H₄、C₃H₆、C₄H₈（SA(3)-CASSCF(2,2)/cc-pVDZ，分别 7500/7500/15000 几何）；CH₂NH₂⁺（MR-CISD/aug-cc-pVDZ，4000 几何）；CH₂S（CASSCF(6,5)/def2-SVP，4000/200/503 划分）。基线：SPaiNN（SchNet/PaiNN）、Exciting DeePMD D yad、SchNarc。
- **烯烃静态精度 SOTA**：PEACE 能量 MAE 0.008–0.022 eV，力 MAE 0.036–0.054 eV/Å，SNAC MAE 分别为 0.066（C₂H₄）、0.041（C₃H₆）、0.069（C₄H₈）eV/Å；均优于 SPaiNN 对应项（见 Supplementary Table 1）。
- **光动力学精度**：烯烃 S₁ 衰减时间常数（0–100 fs 单步拟合）PEACE 为 68.5/51.4/67.8 fs，参考值 68.8/53.3/66.9 fs，偏差 ≤1.9 fs；PEACE 能准确再现 S₂ 初始耗散、S₁ 瞬态累积与 S₀ 增长的全过程。
- **消融结果（CH₂NH₂⁺）**：全模型 SNAC MAE 0.117 eV/Å；无 connection 升至 0.410；无 H-odd 块升至 0.257 且 crossing 拓扑退化为延展低 gap 谷；双消融升至 0.687 且 S₂ 弛豫被强烈抑制；能量 MAE 在双消融仍保持 0.033 eV，说明低能量误差不保证动力学保真。
- **SOC 扩展（CH₂S）**：能量 MAE 1.113 meV，力 MAE 4.941 meV/Å，SNAC MAE 0.933 meV/Å；复 SOC 矩阵元 MAE 0.265 cm⁻¹、RMSE 0.397 cm⁻¹；3 ps 表面跳跃传播后 S₁ 布居 95.4%、T₁ 布居 2.8%，与文献 CASSCF 参考一致。
- **计算效率**：单张 NVIDIA H100 GPU 上 FP64 推理 7.66–12.44 ms/几何，对比六核 Intel Xeon Platinum 8480C CPU 的 CASSCF 6.77–13.39 s，加速约 1000×。

## 相关工作脉络
1. **SPaiNN (Mausenberger et al., 2024)**：基于 SchNet/PaiNN 的等变消息传递，支持直接矢量 NAC 预测；差异——PEACE 通过共享哈密顿+联络的协变导数统一耦合与能量，SPaiNN 以独立读出分别预测。
2. **SchNarc (Westermayr et al., 2020)**：相位不变训练解决 NAC 相位歧义，但标量梯度耦合为保守场近似，抑制反射平面处允许的出平面分量；PEACE 通过 connection 项打破此限制。
3. **Exciting DeePMD (Dupuy & Maitra, 2024)**：学习相位不变耦合 dyad；差异——未通过共享电子哈密顿量显式连接能量与耦合的物理结构。
4. **DANN (Axelrod et al., 2022)**：学习 diabatic 人工神经网络哈密顿；差异——未显式建模潜电子框架随核运动的变化（connection 项），也未对 Hamiltonian 做对称性敏感的奇偶分解。
5. **等变 ML 势（NequIP, PaiNN, SchNet 等）**：基态等变模型的成功推动了激发态推广；本文的定位是将对称结构与物理一致性进一步嵌入到多态电子表示中，而非仅仅推广单态势。
6. **SO coupling ML（已有工作）**：本文扩展引入等变 SOC 读出，将单重态/三重态子空间统一处理，相比独立学习 SOC 的方案更具物理约束。

## 局限性与未来方向
- 当前验证局限于小分子烯烃与 CH₂NH₂⁺、CH₂S，尚未在更大体系或凝聚相环境中检验。
- 潜基态奇偶标签需在反射对称构型下通过 SNAC 对称选择定则推断，对无明确对称性的构型依赖辅助标签质量。
- 文中讨论的未来方向包括：扩展至溶剂化分子与分子组装（凝聚相）、结合主动学习控制参考计算成本、引入态重叠（state overlaps）等额外电子结构信息作为辅助监督。
- 未讨论外场（激光、电场）下的非绝热动力学扩展。
- 训练数据量相对有限（烯烃最大 15000 几何），在更宽构型空间覆盖上的泛化能力有待进一步验证。

## 研究启发与可借鉴点
1. **低训练损失≠物理保真的评估教训**：消融实验表明能量误差小不能保证 crossing 拓扑与动力学正确；后续研究应同步评估局部交叉结构与系综弛豫，而非仅看静态 MAE/RMSE。
2. **潜表示的对称性约束设计范式**：通过奇偶标签把反射对称性显式编码到 Hamiltonian 块中（同奇偶用 0e、异奇偶用 0o），既在对称点强制矩阵元为零，又在破缺点给出非零 one–order 耦合 onset，这一思路可迁移到其他具有离散对称性的量子多体学习问题。
3. **协变导数统一出力的方法学**：以 $[\mathbf{B}_\mu,\mathbf{H}]$ 修正 $\partial_\mu\mathbf{H}$ 得到保守力，同时自然导出 energy-gap-weighted 耦合；对任何需要能量–力–耦合一致性的等变模型均有借鉴价值。
4. **刚体/内部运动的正则化投影解耦**：联络 $\mathbf{B}_\mu$ 中通过 $\Pi_{\mathrm{rig}}^{(\epsilon)}$ 与 $\Pi_{\mathrm{int}}^{(\epsilon)}$ 分离刚体平移旋转与内部形变贡献，使不同物理运动模式各自进入耦合表示；这一投影策略可用于其他需要区分运动自由度的分子学习模型。
5. **SOC 等变的扩展路径**：保持单重态/三重态哈密顿与联络分块独立，再加等变 SOC 读出的架构，为多自旋流形的统一学习提供了可扩展模板。

## 关键术语表
- **Nonadiabatic coupling（非绝热耦合）**：绝热电子态之间由核运动引起的态间跃迁耦合，形式为 $\langle\psi_i|\partial_\mu\psi_j\rangle$，在能隙趋近时发散。
- **Smoothed NAC（SNAC）**：经能量 gap 权重平滑后的非绝热耦合，消除显式 inverse-gap 发散。
- **Parity（奇偶性）**：电子态在空间反射操作下的本征值 $\pm 1$；本文用于约束哈密顿矩阵元的对称变换性质。
- **Latent Hamiltonian（潜哈密顿量）**：在固定潜基 $\{|\chi_a\rangle\}$ 下学习的实对称矩阵 $\mathbf{H}(\mathbf{R})$，其对角化给出绝热能量。
- **Electronic connection（电子联络）**：实反对称矩阵 $\mathbf{B}_\mu$，刻画潜电子框架随核运动的旋转，对导数耦合有独立贡献。
- **Covariant derivative（协变导数）**：$\mathbf{G}_\mu=\partial_\mu\mathbf{H}+[\mathbf{B}_\mu,\mathbf{H}]$，在潜框架下描述 Hamiltonian 的几何不变导数。
- **Spin–orbit coupling（SOC，自旋–轨道耦合）**：电子自旋与轨道角动量耦合，驱动单重态–三重态间系间穿越（ISC）。
- **Surface hopping（表面跳跃）**：基于 Tully 方案的混合量子–经典动力学方法，核沿绝热面运动并在态间随机跳跃。
- **MCH basis（分子库仑哈密顿基）**：自旋无关绝热基下引入复 SOC 矩阵后的有效哈密顿表示。

## 可复现要素
- **数据集**：烯烃 C₂H₄/C₃H₆/C₄H₈ 与 CH₂NH₂⁺、CH₂S 数据集来自论文引用（SA(3)-CASSCF(2,2)/cc-pVDZ、MR-CISD/aug-cc-pVDZ、CASSCF(6,5)/def2-SVP）；论文对原 SPaiNN 数据集做了扩展（烯烃增至 7500/7500/15000）。未声明自有公开数据集，但提供了训练/验证/测试划分表（Supplementary Table 2）。
- **代码/权重**：PEACE 代码与训练模型已开源，GitHub：https://github.com/solvendus/peace。
- **关键超参**：64 channels per parity-resolved irreducible representation；$l_{\mathrm{max}}=2$；4 层 equivariant message-passing + 2 层 equivariant attention per branch；AdamW 优化器，1000 epochs，batch size 64，cosine LR schedule $10^{-3}\to 10^{-4}$；物理观测损失权重 $\lambda_E=\lambda_F=\lambda_{\mathrm{SNAC}}=\lambda_\Delta=1$。
- **训练硬件**：训练 FP32，推理 FP64；GPU：NVIDIA H100；CPU：Intel Xeon Platinum 8480C。
