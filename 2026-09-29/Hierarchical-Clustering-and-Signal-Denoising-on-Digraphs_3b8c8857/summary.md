---
title: "Hierarchical-Clustering-and-Signal-Denoising-on-Digraphs"
source: https://arxiv.org/pdf/2609.34670v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:28:39"
field: "图信号处理与有向图机器学习"
keywords: ["有向图聚类", "谱聚类", "样条拟插值", "图信号处理", "层次聚类", "信号去噪"]
innovations: ["提出广义Hermite矩阵同时编码连通性与方向性，并递归构建层次聚类", "由层次聚类的入度/出度区间诱导多级嵌套knot序列用于样条信号分解", "建立非等距knot下样条拟插值以doubling measure表达的稳定性理论"]
benchmarks: ["DSBM合成有向图", "Cora", "Squirrel", "Telegram", "Wisconsin", "Cornell", "Texas", "Blog"]
---

# 论文速读：Hierarchical Clustering and Signal Denoising on Digraphs

## 一句话总结
本文提出 SpecHDC，一种基于广义 Hermite 矩阵的半监督谱层次有向图聚类算法，并通过层次化聚类诱导的多级 knot 序列构建样条拟插值，实现有向图上噪声信号的自适应去噪。

## 研究问题与动机
- 现有图聚类方法主要针对无向图设计，对有向图（digraph）的边方向信息利用不足；已有有向图聚类方法多产生单一分辨率的划分，缺乏多粒度的一致性层次结构。
- 半监督场景下如何将部分标签融入递归图粗化过程，同时保留簇间非对称交互，尚未被系统研究。
- 样条近似需要有序的 knot 序列，而有向图聚类的离散划分如何转化为适合样条构造的方向性区间（knot sequence），缺乏已有工作。
- 图信号处理中，如何在有向图上利用层次结构进行信号分解与去噪，尚未结合谱聚类与样条理论进行联合建模。

## 核心贡献（创新点）
1. **提出广义 Hermite 矩阵** $W = \alpha(A+A^\top) + \beta(A-A^\top)\mathrm{i}$，同时编码互惠连通性与非对称边方向，具有实特征值和正交特征基；与已有 Herm/ Skew 方法相比，本公式通过 $\alpha,\beta$ 灵活调节对称/反对称分量的权重。
2. **开发 SpecHDC 递归层次聚类框架**，从细粒度到粗粒度逐层聚合边权重并重建广义 Hermite 矩阵，产生一致的嵌套划分；与单级谱方法相比，可生成多层次树结构而非单一划分。
3. **导出层级诱导的入度/出度区间划分**，将各层聚类的方向性连通性组织为多级嵌套 knot 序列；与现有变分样条方法不同，knot 直接由有向图层次的入度/出度决定。
4. **建立非等距 knot 下样条拟插值的稳定性理论**（以 doubling measure 替代 de Boor 不等式），为后续信号去噪提供理论支撑。
5. **实验验证聚类优势与信号去噪效用**：在合成 DSBM 和 7 个真实有向图上，SpecHDC 在多种同配/异配结构和监督设置下优于基线；去噪实验中 RMSE 显著降低、SNR 提升。

## 方法详解

### 广义 Hermite 矩阵
给定加权邻接矩阵 $A$，定义：
$$W = \alpha(A + A^\top) + \beta(A - A^\top)\mathrm{i}$$
其中 $\alpha \in [0,1]$ 控制互惠连通性权重，$\beta \in (0,1]$ 控制方向不平衡权重。$W$ 为 Hermite 矩阵，存在特征分解 $W = U\Lambda U^*$，$\Lambda$ 为实对角阵，$U$ 为复正交特征向量矩阵。

### 单级聚类（Algorithm 1）
1. 构建 $W$，取满足 $|\lambda_j| > \epsilon$ 的特征向量 $g_j$。
2. 构造谱投影矩阵 $P = \sum_{j \in \mathcal{I}_\epsilon} g_j g_j^*$。
3. 将每行 $P_{i,:}$ 转为实向量 $[\mathrm{Re}(P_{i,:}), \mathrm{Im}(P_{i,:})]$，在已知标签约束下进行 k-means 聚类。

### 层次聚类 SpecHDC（Algorithm 2，自底向上）
- 初始化 $G^{(L+1)} = G$，从最细层 $l=L+1$ 开始，对当前有向图执行 Algorithm 1 得到 $k_l$-way 聚类 $\mathcal{C}_l$。
- 将每个簇收缩为粗顶点，粗图邻接矩阵：
$$A^{(l)}(i,j) = \frac{1}{\mathrm{vol}(G)} \sum_{C_u^{(l+1)} \subseteq C_i^{(l)}} \sum_{C_v^{(l+1)} \subseteq C_j^{(l)}} A^{(l+1)}(u,v)$$
- 重复直到根层 $\mathcal{C}_0 = \{V\}$，得到嵌套划分链 $\mathcal{C}_{L+1} \preceq \mathcal{C}_L \preceq \cdots \preceq \mathcal{C}_0$。
- 半监督场景：已知标签 $(\mathcal{I}_{\mathrm{kwn}}, Y_{\mathrm{kwn}})$ 作为 k-means 的硬约束，且满足 $C_j^{(l)} \supseteq Y_{\mathrm{kwn}}^{-1}(j)$。

### 层级区间划分与 knot 序列（自顶向下）
- 根节点分配 $[0,1]\times[0,1]$；沿树向下，对每个父簇的子簇按入度/出度的比例 $\rho_{j,\iota}^{(l)}$ 分割区间：
$$t_{j,\iota}^{(l)} = a_\iota^{(l-1)} + (b_\iota^{(l-1)} - a_\iota^{(l-1)}) \sum_{i=1}^j \rho_{i,\iota}^{(l)}$$
- 得到嵌套矩形块序列 $\mathcal{R}_0 \succeq \mathcal{R}_1 \succeq \cdots \succeq \mathcal{R}_{L+1}$，每个方向的端点构成嵌套 knot 序列 $\xi_{0,\iota} \subseteq \xi_{1,\iota} \subseteq \cdots \subseteq \xi_{L+1,\iota}$。

### 样条拟插值与信号去噪
- 以嵌套 knot 序列构造 B 样条基 $\{B_{j,\iota}^{(l)}\}$，计算各层拟插值 $\mathcal{Q}_\iota^{(l)} f_\iota$，细节分量 $d_{j,\iota} = \mathcal{Q}_\iota^{(j+1)}f - \mathcal{Q}_\iota^{(j)}f$。
- 对细节分量施加自适应阈值 $\widetilde{d}_{j,\iota} = \mathcal{T}_j(d_{j,\iota})$，重构去噪信号：
$$\widetilde{\boldsymbol{y}}_{j_0:L+1,\iota} = \boldsymbol{L}_{j_0,\iota} + \sum_{j=j_0}^{L} \widetilde{\boldsymbol{d}}_{j,\iota}, \quad \widetilde{\boldsymbol{y}} = \frac{1}{2}(\widetilde{\boldsymbol{y}}_{\mathrm{in}} + \widetilde{\boldsymbol{y}}_{\mathrm{out}})$$

## 实验与结果

**数据集**：
- 合成：DSBM（N=5000, k=5），通过 $p,q,\eta$ 控制同配/异配结构和方向随机性。
- 真实：Cora、Squirrel、Telegram、Cornell、Wisconsin、Texas、Blog 共 7 个有向图。

**基线**：Bi-Sym、DD-Sym、DI-SIM、Herm、Skew（5 种有向图谱聚类方法）。

**聚类结果**：
- DSBM 实验：清晰同配（$p \gg q$）或异配（$q \gg p$）结构下 SpecHDC 的 ARI 接近 1；$\eta$ 增大仅影响弱结构分离的图。
- Wisconsin 数据集（Level 1, F-measure）：SpecHDC 在 0%~60% 训练比例下最优，90% 时 F=0.9358（全部基线最高）；Level 2 在 70%、80% 时最优。
- Telegram（无标签，ARI）：SpecHDC 得分 **0.97**（最高），DD-Sym 次之 0.95。
- Blog：SpecHDC 与 DD-Sym 并列最高 ARI=**0.88**。
- Cora（Level 1, F-measure, 90%）：SpecHDC=**0.9112**；Squirrel（Level 1, 90%）：SpecHDC=**0.9201**。

**去噪结果（Table III）**：
- 20% 噪声水平下，Cora 的 RMSE 从 0.0211 降至 0.0141，SNR 从 14.12 dB 提升至 17.60 dB；Squirrel 的 RMSE 从 0.0185 降至 0.0093，SNR 从 14.02 dB 提升至 19.96 dB。
- 两级重建与三级重建在多数数据集上表现接近，表明粗近似已捕获大部分可恢复信号结构。

## 相关工作脉络
1. **Bi-Sym / DD-Sym（Satuluri & Parthasarathy [16]）**：基于 $A$ 与 $A^\top$ 乘积的对称化方法；本文广义 Hermite 矩阵同时保留方向信息而不仅是对称化，且不依赖高斯度折扣。
2. **DI-SIM（Rohe et al. [18]）**：利用有向拉普拉斯的左/右奇异向量刻画发送/接收角色；本文从谱投影矩阵直接出发，并通过层次递归构建多粒度结构。
3. **Herm（Cucuringu et al. [22]）**：复 Hermite 邻接矩阵 $(A-A^\top)\mathrm{i}$；本文将其扩展为含对称项的广义形式 $W=\alpha(A+A^\top)+\beta(A-A^\top)\mathrm{i}$，增加了对连通性的建模能力。
4. **Skew（Hayashi et al. [23]）**：实斜对称表示；本文在保留方向性的同时额外引入对称连通分量，并通过 $\alpha,\beta$ 调节平衡。
5. **Chung 有向拉普拉斯与 Random-walk 方法**：基于稳态分布刻画簇；本文从谱 Hermite 表示出发，无需随机游走假设，且天然支持半监督标签约束。
6. **变分样条图信号处理（Pesenson [39]）**：利用图拉普拉斯扩展样条；本文的 knot 序列直接由有向图层次聚类诱导的入度/出度区间构造，与拉普拉斯本征函数路径不同。

## 局限性与未来方向
- **计算复杂度**：递归层次聚类需要对每一层重新进行特征分解，大图场景下计算开销较大；论文承认可扩展性是实现方向。
- **参数敏感性**：$\alpha,\beta$ 的最优组合依赖图的同配/异配程度，缺乏自动选择机制，需网格搜索。
- **仅测试了两种层次分辨率**：实验中 Level 1/2 的设置较有限，更深层次结构的潜力未充分探索。
- **去噪实验仅针对 5%~20% 噪声水平**：更高噪声下的鲁棒性有待验证。
- 未来方向：扩展到更复杂的有向图结构（如带边权、时序图），以及开发高效可扩展实现。

## 研究启发与可借鉴点
1. **广义 Hermite 表示的灵活框架**：$\alpha,\beta$ 参数化可同时建模连通性与方向性，可迁移至其他有向图学习任务（如节点分类、链路预测）。
2. **层次聚类→knot 序列的映射思路**：将图的层次结构转化为样条的嵌套 knot 序列是一种新颖的多分辨率信号分解方式，可推广到其他基于 hierarchical partition 的信号处理任务。
3. **半监督约束嵌入 k-means 的方式**：将已知标签作为硬约束直接限制质心初始化，简单有效且易于推广到其它谱聚类变体。
4. **非等距 knot 下的稳定性分析**：以 doubling measure 替代 de Boor 不等式的方法论，可用于其他非均匀分区下的样条近似理论分析。
5. **合成 DSBM 的控制系统性消融**：通过 $p,q,\eta$ 独立控制同配强度和方向噪声，为评估方向感知方法的鲁棒性提供了可复用的实验范式。

## 关键术语表
**Generalized Hermitian Matrix**：形如 $W = \alpha(A+A^\top) + \beta(A-A^\top)\mathrm{i}$ 的矩阵，同时编码有向图的互惠连通性与边方向信息，具有实特征值和正交特征基。

**SpecHDC**：Spectral Hierarchical Digraph Clustering，基于广义 Hermite 矩阵的半监督谱层次有向图聚类算法，通过递归图粗化产生嵌套划分。

**Directed Stochastic Block Model (DSBM)**：有向随机块模型，通过簇内概率 $p$、簇间概率 $q$ 和方向概率矩阵 $F$ 生成可控的同配/异配有向图。

**Knot Sequence**：B 样条定义中的节点序列，决定样条函数的分段多项式支集；本文由层次聚类的入度/出度区间端点构成多级嵌套 knot 序列。

**Quasi-interpolant**：拟插值算子，通过局部线性泛函确定样条系数而无需求解全局插值方程，具有局部性和多项式再现性。

**Homophily Ratio**：节点与其入/出邻居标签相同的平均比例，用于量化有向图的同配性程度。

**Doubling Measure**：满足 $\mu([x-2y,x+2y]) \lesssim \mu([x-y,x+y])$ 的概率测度，本文用于建立非等距 knot 下样条的稳定性界。

**Adaptive Thresholding**：根据噪声水平和局部残差活动自适应调整阈值，用于保留信号细节同时抑制噪声的成分。

## 可复现要素
- **数据集**：所有真实数据集（Cora、Squirrel、Telegram、Cornell、Wisconsin、Texas、Blog）公开可用；DSBM 为合成数据，代码可复现。
- **代码**：已开源，地址 https://github.com/kellysylvia77/Digraph。
- **关键超参**：$\alpha \in [0,1]$、$\beta \in (0,1]$、$\epsilon > 0$；论文未提及具体默认值，说明通过验证集网格搜索确定；层次配置 $K=(k_1, \ldots, k_L)$ 需预先指定。
