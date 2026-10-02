---
title: "INTERFERENCE-BEYOND-GEOMETRYIN-CONCEPT-EXTRACTION"
source: https://arxiv.org/pdf/2609.35351v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:30:18"
---

# 论文速读：INTERFERENCE-BEYOND-GEOMETRYIN-CONCEPT-EXTRACTION

## 一句话总结
本文提出“有效干扰（effective interference）”度量，将特征方向的几何重叠与共激活统计相乘，揭示稀疏自编码器（SAE）中的干扰本质上是数据依赖的联合属性；理论推导与实验表明，架构约束（偏置、增益、编码器-解码器解耦）能选择性抑制或保留共激活特征的交叉贡献，而非盲目追求全局正交。

## 研究问题与动机
1. **现有干扰度量仅依赖几何**：超位置假设下特征方向不可避免重叠，但现有工作几乎全用字典余弦相似度、互相干性或神经元容量衡量干扰，完全忽略特征在实际数据中的激活模式（code statistics）。
2. **几何 measure 无法刻画 realized interaction**：相同的几何重叠可能对应“频繁弱交互”或“罕见强交互”，且无法区分建设性（constructive）与破坏性（destructive）交叉贡献，导致对稀疏编码效率的判断失真。
3. **架构自由度如何调节干扰缺乏理论刻画**：SAE 的权重绑定、偏置、增益与正交约束究竟通过何种机制影响 realized interactions，现有文献多停留在经验对比，缺少统一的编码器一致性分析框架。
4. **线性表征假设的几何视角存在盲区**：该假设强调概念由激活方向线性编码，但未解释为何某些高重叠特征对实际重建贡献极小，而某些特定重叠却被模型 constructive 利用（如周期性结构、数据共现）。

## 核心贡献（创新点）
1. **定义有效干扰并给出三因子分解**：提出 $I_{ij} = \rho_{ij} \pi_{ij} M_{ij}$，将干扰拆解为几何取向 $\rho_{ij}$、共激活频率 $\pi_{ij}$ 与条件幅值 $M_{ij}$，首次量化了特征间 realized pairwise cross-contributions。
2. **推导固定支持下的编码器一致性恒等式**：在局部仿射假设下建立包含自响应失配、方向性传入交叉串扰、偏置补偿与残差耦合的严格平衡方程，揭示架构自由度如何补偿干扰。
3. **识别四类非互斥的架构调控路由**：证明受限架构通过局部正交化压制共激活对重叠，而引入偏置、编码器增益或解除权重绑定可分别通过均值补偿、尺度缩放与 biorthogonal 解耦保留建设性干扰。
4. **系统性跨架构与跨数据验证**：在 Pythia-160M、MNIST 及 SAEBench 独立训练模型上验证理论，展示约束放松后 $\mathcal{I}_+ + |\mathcal{I}_-|$ 可达重构能量的 40–50%，净干扰稳定在 15–25%，打破“干扰应被最小化”的直觉。

## 方法详解
1. **有效干扰的数学定义**：对加法表征 $\widehat{\pmb{x}} = D z(\pmb{x}) = \sum_i d_i z_i(\pmb{x})$，定义样本级交互 $\iota_{ij}(\pmb{x}) = (d_i^\top d_j) z_i(\pmb{x}) z_j(\pmb{x})$，有效干扰为其期望 $I_{ij} = \mathbb{E}[\iota_{ij}]$。进一步分解为 $I_{ij} = \rho_{ij} \pi_{ij} M_{ij}$，其中 $\rho_{ij}=d_i^\top d_j$ 为字典余弦相似度，$\pi_{ij}=\mathrm{Pr}(a_i=1,a_j=1)$ 为共激活概率，$M_{ij}=\bar{\mathbb{E}}[z_i z_j \mid a_i=1,a_j=1]$ 为条件幅值期望。
2. **重构能量的符号分解**：定义规范化指标 $\mathcal{Z}_+,\mathcal{Z}_-$ 分别为正/负交叉项之和占 $\mathbb{E}\|\widehat{\pmb{x}}\|_2^2$ 的比例，$\mathcal{I}_{\mathrm{net}}=\mathcal{Z}_+ + \mathcal{Z}_-$ 刻画净效应；该分解避免正负项相互抵消后掩盖真实交互强度。
3. **编码器一致性恒等式（Prop 4.2）**：在固定支持 $S$ 下编码器局部仿射 $z_S = W_S^\top x + b_S$，对任意激活特征 $i$ 推导得：
   $(\alpha_i \gamma_{ii} - 1)\mathbb{E}[z_i \mid a_i=1] + \sum_{j\neq i}\alpha_i \gamma_{ij}\mathbb{E}[z_j \mid a_i=1] + b_i + \mathbb{E}[\pmb{w}_i^\top r \mid a_i=1] = 0$，
   其中 $\alpha_i=\|\pmb{w}_i\|_2$，$\gamma_{ij}=\pmb{w}_i^\top \pmb{d}_j/\alpha_i$，$r=x-D_S z_S$ 为残差。该等式将自响应、传入串扰、偏置补偿与残差耦合置于同一
