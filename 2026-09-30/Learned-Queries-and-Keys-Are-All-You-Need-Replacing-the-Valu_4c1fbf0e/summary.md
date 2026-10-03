---
title: "Learned-Queries-and-Keys-Are-All-You-Need-Replacing-the-Valu"
source: https://arxiv.org/pdf/2609.36698v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:55:48"
field: "高效视觉Transformer"
keywords: ["双头Transformer", "Value投影替代", "Shearlet变换", "乘法避免注意力", "参数高效ViT"]
innovations: ["提出双头Transformer架构，用固定正交变换替代独立学习的Value投影", "构建Shearlet多尺度方向滤波器组嵌入注意力Value，实现方向感知的双头设计", "提出MF-Tanh无乘法注意力，用元素级符号运算和归一化tanh替代softmax与V投影"]
benchmarks: ["CIFAR-10", "Tiny ImageNet"]
---

# 论文速读：Learned-Queries-and-Keys-Are-All-You-Need-Replacing-the-Value-Projection-with-Structured-Transforms

## 一句话总结
本文提出双头Transformer架构，用固定正交变换（DCT/WHT/DFT/Shearlet）或乘法避免算子替代传统独立学习的Value投影，在保持注意力机制的同时减少约1/3参数并降低计算开销，在CIFAR-10小模型上实现更高精度。

## 研究问题与动机
1. **核心问题**：Transformer的注意力机制需要Q、K、V三个独立学习投影，造成参数冗余和KV cache内存开销大；
2. **视觉场景瓶颈更突出**：图像token数量随分辨率平方增长，注意力计算复杂度为$O(N^2)$，计算负担更为严重；
3. **多模态加剧开销**：长视觉序列与文本序列拼接进一步放大注意力开销；
4. **现有优化路径的局限**：当前主流方向是改进注意力复杂度本身（线性注意力等），而本文从精简投影参数角度切入，探索"是否真的需要独立学习的V"。

## 核心贡献（创新点）
1. **提出双头注意力框架**：将标准QKV的三投影简化为仅保留Q和K两个学习投影，Value由Key或其固定变换构造，本质区别于需要额外学习$W_V$的传统方法；
2. **系统性对比多种变换替代方案**：将DCT、WHT、DFT、Shearlet滤波器组、MA（Multiplication-Avoiding）算子分别集成为Value替代，形成完整的设计空间探索；
3. **Shearlet多尺度方向感知设计**：提出基于2D FFT + 方向锥划分 + 高斯带通窗口的Shearlet滤波器组，将几何方向信息注入注意力Value，与纯频域变换形成本质区别；
4. **MF-Tanh注意力新范式**：提出无乘法元素级V替代$\tilde{V}=\text{sign}(Q)\odot|K|+\text{sign}(K)\odot|Q|$，并用归一化tanh替换softmax，实现硬件友好型设计。

## 方法详解
**总体框架**：给定输入$X \in \mathbb{R}^{N \times d}$，学习$Q=XW_Q$、$K=XW_K$，去除$V=XW_V$，输出形式为$Y = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \mathcal{T}(K)$，其中$\mathcal{T}$为固定变换。

- **QKK（基线）**：$Y = A(Q,K)K$，直接以K作为Value，参数从$3d^2$降至$2d^2$。
- **正交变换类**：$Y_{DCT}=A(Q,K)\mathcal{D}(K)$，$Y_{WHT}=A(Q,K)\mathcal{H}(K)$，$Y_{DFT}=A(Q,K)|\mathcal{F}(K)|$（取绝对值因DFT为复数）。
- **MF-Tanh注意力**：构造无乘法V：$\tilde{V}=\text{sign}(Q)\odot|K|+\text{sign}(K)\odot|Q|$；替换softmax为行$\ell_1$归一化的tanh：$\hat{A}_{ij}=\frac{\tanh(S_{ij})}{\epsilon+\sum_k|\tanh(S_{ik})|}$，输出$Y=\hat{A}\tilde{V}$，可保留负交互。
- **Shearlet滤波器组**：将patch reshape回2D网格→2D FFT→按横向/纵向锥划分→每尺度用径向高斯窗+方向斜率高斯窗构造方向滤波器（3尺度×5剪切值×2方向=30个方向滤波器+1个低通）；每个head学习融合系数$\alpha_{h,m}$对固定滤波器加权融合，得到$S(Z)=\mathcal{F}_{2D}^{-1}[\widehat{Z}\cdot H_h^{fused}]$；四种变体分别控制Shearlet作用在Q、K、输出中的位置。

**参数量**：Mini-ViT中标准QKV为546,186参数，双头变体降至约480,138，减少约12%。

**复杂度**：标准$C_{QKV}=O(4Nd^2+2N^2d)$；QKK降至$O(3Nd^2+2N^2d)$；DCT/WHT/DFT仅增加固定变换开销（如WHT为$O(Nd\log d_k)$）；Shearlet增加$O(dN_p\log N_p+HMN_p)$。

## 实验与结果
**实验设置**：CIFAR-10上使用Mini-ViT（32×32输入，4×4 patch，embedding=128，4个Transformer block，4个头，MLP ratio=2，30轮）；另在CIFAR-10和Tiny ImageNet上用标准ViT配置（6层，D=192，3个头）对比Softmax ViT与MF-Tanh。

**核心结果（CIFAR-10，Table 2）**：
| 配置 | 参数量 | 测试精度(%) |
|------|--------|-------------|
| 标准$A(Q,K)V$ | 546,186 | 76.035±0.530 |
| QKK | 480,138 | 76.155±0.035 |
| WHT | 480,138 | **76.830±0.764** |
| $A(Q,K)\mathcal{S}(K)$（最佳Shearlet） | 480,634 | **77.320±0.354** |
| $A(Q,\mathcal{S}(K))\mathcal{S}(K)$ | 480,634 | 77.310±0.976 |

- **最强结果**：$A(Q,K)\mathcal{S}(K)$达到77.32%，比标准QKV提升**1.29个百分点**，同时减少约6.6万参数（约12%）。
- **WHT**参数最少且无需乘法，精度76.83%，也优于标准注意力。
- **MF-Tanh（Table 3）**：CIFAR-10上77.80% vs 81.92%（单种子实验，差距4.12%），Tiny-IN上35.40% vs 40.39%（差距4.99%），参数量减少约8%，乘法减少约14M；作者指出需多种子评估。

## 相关工作脉络
1. **Dosovitskiy et al. (ViT, 2020)**：Vision Transformer开山之作，本文在其ViT基础上做参数精简改进，定位是效率优化而非架构范式颠覆。
2. **Vaswani et al. (Attention is All You Need, 2017)**：标准QKV注意力设计者，本文挑战其必要性假设，核心差异在于去掉独立V投影。
3. **Pan et al. (DCT-based Decorrelated Attention, IJCAI-ECAI 2026 / arXiv 2024)**：作者团队前作，将DCT引入注意力；本文扩展为系统性框架，涵盖更多变换类型。
4. **Hamdan & Cetin (Htma-Net, 2025)**：提出乘法避免网络；本文MF-Tanh部分继承其无乘法算子思想并适配到注意力机制。
5. **Kutyniok et al. (Shearlets, 2012)**：Shearlet理论奠基工作；本文将其改造为可学习的融合滤波器组并嵌入Transformer，属于跨领域迁移应用。
6. **Tay et al. (2022)**：长序列高效注意力综述；本文与其方向不同——不改进$N^2$复杂度，而是精简投影本身。

## 局限性与未来方向
1. **仅在小模型和小数据集上验证**：实验集中在Mini-ViT+CIFAR-10和Tiny ImageNet，尚未在ImageNet大图或更大ViT/Swin等架构上验证可扩展性（作者自述未来方向）。
2. **MF-Tanh性能差距仍明显**：单种子实验中MF-Tanh精度低于Softmax ViT约4-5个百分点，需多种子和官方测试集验证。
3. **Shearlet变体训练方差较大**：部分配置（如$A(\mathcal{S}(Q),\mathcal{S}(K))K$）标准差达1.075%，稳定性有待改善。
4. **固定变换的表达力上限存疑**：DCT/WHT/DFT等为预定义变换，可能在复杂语义建模上不如可学习V投影灵活。
5. **未涉及推理加速实测**：KV cache减少的理论优势未在生成/长序列推理场景下实测验证。

## 研究启发与可借鉴点
1. **"去掉V投影"的思路可迁移**：对于任何需要QKV投影的序列模型（如LSTM替代方案、状态空间模型），可探索用固定变换或Key自身替代独立Value投影以降低参数。
2. **Shearlet方向感知可用于视觉-语言模型**：多尺度方向特征对图像理解有益，可将Shearlet滤波器组作为视觉编码器中的高效特征提取模块。
3. **MF-Tanh的硬件友好设计值得工程落地**：无乘法算子+tanh归一化适合边缘设备/存算一体芯片部署，可与量化、剪枝结合。
4. **消融设计层次清晰**：Shearlet的四种变体（只换输出/只换相似度/两者都换/对称换）为研究"方向信息应作用于何处"提供了系统化的对照方案，可借鉴到其他变换替代研究中。
5. **固定变换vs可学习的权衡启发**：本文证明某些场景下固定正交变换效果不输甚至超过可学习V，未来可探索"部分可学习+部分固定"的混合策略。

## 关键术语表
**Dual-headed Transformer**：仅保留Q和K两个学习投影的注意力变体，Value由Key或其固定变换构造，省去独立$W_V$。

**Shearlet Transform**：多尺度方向敏感的信号分解方法，通过锥形频率划分和方向滤波捕捉图像的边沿和几何结构。

**MF-Tanh Attention**：乘法避免（Multiplication-Free）注意力，用无乘法元素级运算构造$\tilde{V}$，并以行$\ell_1$归一化tanh替换softmax。

**WHT（Walsh-Hadamard Transform）**：仅含+1/-1的二值正交变换，可完全通过加减法实现，计算极低成本。

**KV Cache**：推理时缓存已计算的Key和Value矩阵以避免重复计算，双头设计可减少约1/3的KV cache内存占用。

**Scaled Dot-Product Attention**：标准注意力计算$\text{softmax}(QK^T/\sqrt{d_k})V$，本文保留前两部分，替换V。

## 可复现要素
- **数据集**：CIFAR-10（公开）、Tiny ImageNet（公开）、ImageNet（提及但未报告具体结果）；
- **代码/权重**：论文未提及代码是否开源；
- **关键超参**：Mini-ViT配置——32×32输入，4×4 patch，embedding=128，4个Transformer block，4个头，MLP ratio=2，训练30轮；标准ViT配置——6层，D=192，3个头，MLP隐藏=768，AdamW，batch=128，lr=3×10⁻³，weight decay=0.05，warmup=5轮，cosine decay，训练100轮；
- **Shearlet超参**：3尺度、5剪切值、2方向（横向/纵向），共30个方向滤波器+1个低通。
