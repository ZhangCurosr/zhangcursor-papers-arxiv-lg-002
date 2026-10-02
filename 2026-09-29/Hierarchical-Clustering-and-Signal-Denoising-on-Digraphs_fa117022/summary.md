---
title: "Hierarchical-Clustering-and-Signal-Denoising-on-Digraphs"
source: https://arxiv.org/pdf/2609.34670v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:28:42"
field: "有向图聚类与图信号处理"
keywords: ["digraph clustering", "spectral clustering", "Hermitian matrix", "spline quasi-interpolation", "graph signal processing", "hierarchical clustering"]
innovations: ["提出广义Hermitian矩阵统一编码连通与方向，并用于半监督谱层次聚类", "将递归粗化的入度/出度区间映射为嵌套非均匀节点序列以构造多分辨率样条", "基于层级样条分解与自适应阈值实现有向图信号去噪"]
benchmarks: ["DSBM合成有向图", "Cora", "Squirrel", "Telegram", "Cornell", "Wisconsin", "Texas", "Blog"]
---

# 论文速读：Hierarchical-Clustering-and-Signal-Denoising-on-Digraphs

## 一句话总结
本文提出 SpecHDC，一种半监督谱层次有向图聚类算法，通过广义 Hermitian 矩阵编码连接性与边方向，递归构建嵌套划分，并将层次结构与入度/出度区间映射为多阶样条节点序列，实现有向图信号的结构自适应去噪与恢复。

## 研究问题与动机
- 现实网络多为有向图（digraph），边方向编码非对称关系（影响、流向等），现有无向图聚类方法无法直接应用，而已有有向图聚类方法多仅输出单层划分，缺乏层次性。
- 现有谱方法在递归粗化过程中难以同时保持不对称的簇间交互并融入部分标签约束。
- 样条逼近通常需要预定义的等距节点序列，而有向图的离散层次结构如何转化为适用于样条构造的有效节点序列尚未被探索。
- 图信号处理中的去噪与恢复需要能够适配图层次与非对称结构的连续表示，以分离结构近似与细节。

## 核心贡献（创新点）
- 提出广义 Hermitian 矩阵表示，联合编码互惠连通性与非对称边方向，兼具实特征值与正交特征向量基。
- 设计 SpecHDC 半监督层次聚类框架，自底向上递归聚合跨簇边权重并重计算 Hermitian 表示，产生一致的嵌套划分并在 k-means 中嵌入已知标签约束。
- 推导各层级聚类对应的入度与出度区间，自顶向下组织为嵌套节点序列（knot sequences），为后续样条构造提供结构自适应支撑。
- 建立非均匀节点序列下样条准插值的稳定性界（基于 doubling measure），克服 de Boor 不等式在非等距节点下的不适用性。
- 将层次节点序列用于多图信号分解与自适应阈值重建，在合成与真实有向图上验证聚类有效性与信号去噪性能。

## 方法详解
- 广义 Hermitian 矩阵：给定加权邻接矩阵 $A$，定义 $W = \alpha(A + A^\top) + \beta(A - A^\top)\mathrm{i}$，其中 $\alpha \in [0,1]$ 控制对称连通性，$\beta \in (0,1]$ 控制非对称方向性；$W$ 为 Hermitian 矩阵，可进行特征分解 $W = U\Lambda U^*$。
- 谱嵌入与聚类：保留满足 $|\lambda_j| > \epsilon$ 的特征向量构造谱投影矩阵 $P = \sum_{j \in \mathcal{I}_\epsilon} g_j g_j^*$，将每行复数向量展平为 $[\mathrm{Re}(P_{i,:}), \mathrm{Im}(P_{i,:})]$ 作为特征，结合已知标签约束 $(\mathcal{I}_{\mathrm{kew}}, Y_{\mathrm{kew}})$ 执行 k-means 获得 $k$ 路划分。
- 递归粗化与层次构建：从 finest level（单个顶点）开始，将当前划分中的每个簇收缩为粗顶点，粗邻接矩阵按有序对聚合：$A^{(l)}(i,j) = \frac{1}{\mathrm{vol}(G)} \sum_{C_u^{(l+1)} \subseteq C_i^{(l)}} \sum_{C_v^{(l+1)} \subseteq C_j^{(l)}} A^{(l+1)}(u,v)$，再对该粗图重新计算 $W$ 与谱划分，重复直至 root level；产生的划分满足嵌套关系 $\mathcal{C}_{L+1} \preceq \cdots \preceq \mathcal{C}_0$。
- 层次诱导区间划分：自 top-down，根簇分配 $[0,1] \times [0,1]$；对每一父簇的子簇 $C_j^{(l)}$，按其入度/出度比例 $\rho_{j,\iota}^{(l)} = d_\iota(C_j^{(l)}) / \sum d_\iota(\cdot)$ 分割父区间，得到嵌套区间族 $\mathcal{R}_0 \succeq \mathcal{R}_1 \succeq \cdots \succeq \mathcal{R}_{L+1}$。
- 节点序列与样条基：将各层入度/出度区间的端点排序并重复 $m+1$ 次得到多阶 B-spline 节点序列 $\xi_{l,\iota}$，满足嵌套 $\xi_{0,\iota} \subseteq \cdots \subseteq \xi_{L+1,\iota}$，进而构造各层样条空间与准插值算子 $\mathcal{Q}_\iota^{(l)}$。
- 信号分解与去噪：将图信号 $f$ 在 finest 层进行样条表示，利用相邻分辨率差 $d_{j,\iota} = \mathcal{Q}_\iota^{(j+1)}f - \mathcal{Q}_\iota^{(j)}f$ 获得细节；对细节施加自适应阈值 $\tilde{d}_{j,\iota} = \mathcal{T}_j(d_{j,\iota})$，以较粗近似为基底重建：$\tilde{y}_{j_0:L+1,\iota} = L_{j_0,\iota} + \sum_{j=j_0}^L \tilde{d}_{j,\iota}$，最终取入/出两个方向重建结果的平均。

## 实验与结果
- 合成数据：基于有向随机块模型 DSBM 生成同质（homophilic, $p \gg q$）与异质（heterophilic, $q \gg p$）图，评估参数 $p,q,\eta$ 及节点数 $N$、簇数 $k$ 的影响。
- 真实数据：7 个公开有向图，包括 Cora、Squirrel、Telegram、Cornell、Wisconsin、Texas、Blog；其中 Cora/H/Squirrel/Cornell/Wisconsin/Texas 含节点标签，Telegram/Blog 无标签。
- 基线方法：Bi-Sym、DD-Sym、DI-SIM、Herm、Skew 五种有向图谱聚类方法。
- 评估指标：ARI（合成）、Modularity（M）、F-measure（F）；无标签数据集用 ARI 衡量跨重复运行一致性。
- 聚类主结果：在 Wisconsin 上，SpecHDC Level 1 F-measure 从无标签 0.5836 提升至 90% 标注的 0.9358；在 Cora Level 2 Modularity 于 50%-90% 标注区间达到 0.4705→0.7157；在 Texas Level 1 F-measure 90% 标注时达到 0.9179；在无标签 Telegram 与 Blog 上分别取得 ARI=0.97 与 ARI=0.88（最高或并列最高）。
- 去噪主结果（20% 噪声）：Cora 原始 RMSE 0.0211 降至重建后 0.0141，SNR 从 14.1183 提升至 17.6025；Squirrel 原始 RMSE 0.0185 降至 0.0093，SNR 从 14.0223 提升至 19.9570；跨 5%-20% 噪声比例，样条重建均持续降低 RMSE 并提高 SNR。

## 相关工作脉络
- Bi-Sym/DD-Sym（Satuluri & Parthasarathy）通过 $A A^\top$ 与 $A^\top A$ 对称化构造无向相似性，丢失方向非对称信息；本文以广义 Hermitian 矩阵同时保留连通与方向。
- DI-SIM（Rohe et al.）利用正则化有向拉普拉斯的左右奇异向量刻画发送/接收角色；本文以统一 Hermitian 谱嵌入替代奇偶因子分解，便于接入 k-means 与层次递归。
- Herm（Cucuringu et al.）采用 $W=(A-A^\top)\mathrm{i}$ 的复 Hermitian 表示；本文扩展为 $\alpha$ 与 $\beta$ 加权形式，覆盖纯对称连通与纯非对称方向的连续情形。
- Skew（Hayashi et al.）以实斜对称矩阵逼近方向割结构；本文在复 Hermitian 框架下保持正交特征基与实特征值，便于直接谱投影与 k-means。
- 随机游走/InfoMap 等流式方法以转移平稳分布刻画社区；本文直接从邻接结构出发构建谱表示，并进一步将层次划分转译为样条节点序列。
- 变分样条（Pesenson 等）将图拉普拉斯用于图信号插值，但未从有向图层次与入/出方向推导节点序列；本文提供可嵌套的多分辨率样条支撑。

## 局限性与未来方向
- 算法为自底向上递归粗化，复杂度随图规模与层次深度增加，大规模图的可扩展性未充分讨论。
- 广义 Hermitian 矩阵中超参数 $\alpha,\beta,\epsilon$ 需网格搜索，不同结构与方向随机性下参数敏感性较高。
- 当前实验覆盖的有向图规模与异质/同质极端情形仍有限，尚未在大型动态或有加权多重关系网络上验证。
- 样条准插值的稳定性结果依赖 doubling measure 假设，实际离散分布下需进一步验证条件满足程度。
- 未来方向包括可扩展实现、面向更复杂有向图结构（含高阶关系、时序演化）的推广。

## 研究启发与可借鉴点
- 将半监督约束直接嵌入 k-means 的硬绑定策略（固定已知标签所属簇）可迁移至其他谱层次聚类场景。
- 用“入度/出度比例”自顶向下分割单位区间生成嵌套节点序列，是一种将离散层次结构映射为连续多分辨率支撑的通用思路。
- 多分辨率样条分解配合自适应阈值重建可用于图信号去噪、异常检测与特征提取，可与下游学习任务结合。
- 广义 Hermitian 表示中的 $\alpha/\beta$ 平衡机制启发我们在不同连通-方向主导的数据集上自适应配置谱权重。
- 实验设计同步考察同质/异质强度、方向随机性与规模变化，为图算法鲁棒性评估提供可复用协议模板。

## 关键术语表
- **Digraph（有向图）**：边具有方向的图结构，邻接矩阵一般非对称，用于刻画非对称关系与流向。
- **Generalized Hermitian matrix（广义 Hermitian 矩阵）**：由对称与斜对称两部分线性组合形成的复 Hermitian 算子，兼具连通与方向信息并具有实特征值与正交特征向量。
- **SpecHDC（Spectral Hierarchical Digraph Clustering）**：基于广义 Hermitian 谱表示的半监督层次有向图聚类算法，通过递归粗化产生嵌套划分。
- **Knot sequence（节点序列）**：B-spline 定义的分段多项式分段点集合，控制样条局部支持与分辨率。
- **Spline quasi-interpolant（样条准插值）**：仅依赖局部样本或局部泛函的样条逼近算子，避免全局插值方程求解并保持多项式复现性。
- **Doubling measure（加倍测度）**：满足区间扩倍后测度受原测度常数倍控制的测度，用于建立非均匀节点下样条系数与函数范数的等价界。
- **Homophily / Heterophily（同质性/异质性）**：节点倾向于与同类节点相连为同质，倾向与不同类节点相连为异质，衡量图的结构极性。
- **Adjusted Rand Index / Modularity / F-measure**：分别衡量聚类与真实划分的配对一致性、社区内部边密度增益以及与类别标签的精确-召回调和质量。

## 可复现要素
- 数据集：DSBM 合成数据；真实数据集包括 Cora、Squirrel、Telegram、Cornell、Wisconsin、Texas、Blog，均为公开可用数据集。
- 代码与权重：论文 Supplementary Material 声明代码已开源，地址为 https://github.com/kellysylvia77/Digraph；未提及预训练权重，因方法为谱/聚类类无参数化模型。
- 关键超参：$\alpha \in [0,1]$、$\beta \in (0,1]$、阈值 $\epsilon > 0$；层次簇数序列 $K = (k_1,\dots,k_L)$；B-spline 阶次 $m$；半监督训练中已知标签比例与采样方式；去噪噪声水平 $r \in \{5\%,10\%,15\%,20\%\}$。
