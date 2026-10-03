---
title: "Learned-Queries-and-Keys-Are-All-You-Need-Replacing-the-Valu"
source: https://arxiv.org/pdf/2609.36698v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:55:55"
field: "高效视觉 Transformer"
keywords: ["Vision Transformer", "attention mechanism", "parameter reduction", "Shearlet transform", "dual-headed attention", "multiplication-avoiding", "efficient transformers"]
innovations: ["提出双头Transformer用固定变换替代V投影减少约1/3参数", "将Shearlet多尺度方向滤波池引入Vision Transformer的value构造", "设计免乘MF-tanh注意力机制以大幅降低推理计算开销"]
benchmarks: ["CIFAR-10", "Tiny-ImageNet"]
---

# 论文速读：Learned-Queries-and-Keys-Are-All-You-Need-Replacing-the-Valu

## 一句话总结
本文提出**双头 Transformer**（dual-headed transformer），将标准注意力中三个学习投影（Q、K、V）缩减为两个（Q、K），用**固定正交变换**（DCT、WHT、DFT）、**剪切波变换**（Shearlet）或**免乘运算**（MA）算子替代独立学习的 Value 投影，在参数量减少约 12% 的同时于 CIFAR-10 和 Tiny-ImageNet 上取得与甚至优于标准 ViT 的精度。

## 研究问题与动机
1. **Transformer 计算瓶颈**：标准多头注意力复杂度为 O(N²d)，在视觉任务中 token 数随分辨率二次增长，KV-cache 内存需求尤为严重。
2. **冗余参数开销**：每个注意力头需三个 d×d 投影矩阵（W_Q、W_K、W_V），约 3d² 个参数，其中 V 投影可被结构化表示替代。
3. **现有替代方案局限**：线性注意力、低秩近似等方向各有取舍，但尚未系统探索用**固定变换域**代替学习投影 V 的可行性。
4. **硬件友好性需求**：乘法操作在端侧推理中代价高昂，探索免乘（multiplication-avoiding）表示具有工程意义。

## 核心贡献（创新点）
1. **双头注意力框架**：将标准 QKV 三投影替换为 QK 双投影，用固定变换 T(K) 替代 V，减少约 1/3 的投影参数；与已有工作的本质区别在于**不引入任何额外学习参数**，完全依靠正交/结构变换表达 value 信息。
2. **剪切波启发式多尺度方向滤波**：基于锥适配 Shearlet 构建固定多尺度方向滤波池，并允许模型学习滤波融合系数；本质区别在于将**方向与多尺度先验**直接嵌入注意力 value 构造，而非依赖逐点学习的 W_V。
3. **归一化 MF–tanh 注意力**：用元素级免乘算子构造 Ṽ，并以符号化 tanh 替代 softmax 实现注意力门控；与常规 softmax ViT 的本质区别是**完全消除乘法操作**且保留负向交互信号。
4. **系统的复杂度分析**：推导了各类双头单元的计算复杂度闭式表达，证明投影成本从 O(4Nd²) 降至 O(3Nd²)，同时保持 O(N²d) 的 token 交互复杂度。

## 方法详解
- **核心公式**：通用双头单元为 Y = A(Q, K)·T(K)，其中 A(Q,K)=softmax(QKᵀ/√d_k)，T(·) 为固定变换（恒等/DCT/WHT/DFT/Shearlet）。
- **Key-as-Value 基线**：Y = A(Q,K)·K（即 QKK 配置），完全不引入额外变换。
- **正交变换单元**：Y_DCT = A(Q,K)·D(K)，Y_WHT = A(Q,K)·H(K)，Y_DFT = A(Q,K)·|F(K)|（取绝对值处理复数）。
- **Shearlet 滤波池**：对图像 patch 做 2D FFT，在频率平面划分水平/垂直锥，叠加高斯带通 R_s 和方向窗 D_k 生成 30 个方向滤波器 + 1 个低频滤波器；融合系数 α_{h,m} 经 softmax 学习。四种构型（QK-ShearK、Q-ShearK-ShearK、ShearQ-ShearK-K、全 Shearlet 域）可分别研究方向信息对注意力权重和输出表示的影响。
- **MF–tanh 注意力**：Ṽ = sign(Q)⊙|K| + sign(K)⊙|Q|（无乘法），注意力权重Â_{ij}=tanh(S_{ij})/(ε+Σ_k|tanh(S_{ik})|)，输出 Y=Â·Ṽ。
- **参数量节省**：以 Mini-ViT 为例，标准 QKV 含 546,186 参数，双头变体降至约 480,138（WHT/DCT/QKK）或 480,634（Shearlet），约减少 12%。

## 实验与结果
- **数据集**：CIFAR-10（32×32，4×4 patch，65 tokens）、Tiny-ImageNet（8×8 patch）、ImageNet（摘要提及，正文实验未在摘要外展开详细数字）。
- **Mini-ViT 配置**：4 层 Transformer、128 维嵌入、4 个注意力头、MLP ratio=2，30 轮训练。
- **CIFAR-10 准确率（Table 2）**：
  - 标准 Softmax ViT：76.035%（546,186 参数）
  - QKK：76.155%（480,138 参数）
  - **QK-ShearK（A(Q,K)S(K))：77.320%**（最佳，480,634 参数，提升 +1.285pp）
  - WHT：76.830%，DCT：76.140%
- **CIFAR-10 / Tiny-ImageNet 对比（Table 3，seed 0 单实验）**：
  - 标准 Softmax ViT（6 层，D=192，3 头）：C10 81.92% / Tiny-IN 40.39%
  - MF–tanh：C10 77.80%（-4.12pp）、Tiny-IN 35.40%（-4.99pp）；参数量减少 222,336，乘法操作减少 14.38M。
- **核心结论**：Shearlet 变体在参数量减少约 12% 的前提下超越标准 ViT；WHT 为最计算高效方案；MF–tanh 在精度略有下降时大幅削减乘法开销。

## 相关工作脉络
1. **Dosovitskiy et al. (ViT, 2020)**：本文在其紧凑 ViT 架构上对比验证，核心差异在于消除 V 投影而非改动架构。
2. **Pan et al. (DCT-based Attention, IJCAI-ECAI 2026)**：本文延续 DCT 在注意力中的使用思路，但进一步探索了 WHT、Shearlet、MA 等多种固定变换的系统性比较。
3. **线性/低复杂度注意力（Tay et al., 2022）**：本文关注点在**参数压缩**（消除 W_V）而非序列长度维度上的复杂度优化，二者正交。
4. **Hamdan & Cetin (HTMA-Net, 2025)**：WHT/免乘神经网络的先验工作，本文将其注意力集成形式化并推广至 Transformer 场景。
5. **Kutyniok et al. (Digital Shearlet Transforms, 2012)**：Shearlet 数学基础，本文首次将其作为可学习的固定滤波池引入 Vision Transformer 的 value 构造。
6. **Swin Transformer（Liu et al., 2021）**：同属视觉高效 Transformer 方向，但 Swin 通过滑动窗口降低全局注意力复杂度，本文通过消除 V 投影压缩参数量，方法论完全不同。

## 局限性与未来方向
1. 大规模数据集（ImageNet）上的系统评测仅在摘要中提及，正文主要实验集中于 CIFAR-10/Tiny-IN，**缺乏在大规模基准上的充分验证**。
2. MF–tanh 方案在单 seed 实验中精度落后标准 ViT 约 4–5pp，**需要多 seed 评估以确认稳定性**（作者已明确此限制）。
3. Shearlet 滤波池的方向数和尺度数为人工设定（3 尺度×5 剪切×2 方向），**超参敏感性与自适应设计有待探索**。
4. 论文未公开代码与权重，**可复现性受限**。
5. 未来方向包括：更大架构/数据集验证、更高效的滤波池设计、面向硬件的免乘实现。

## 研究启发与可借鉴点
1. **"固定变换替代学习投影"的设计范式可迁移**：本研究展示了用 DCT/WHT/Shearlet 等固定算子替代 W_V 的可行性，可推广至 LSTM/State Space Model 等其他序列建模架构中的 value 构造。
2. **Shearlet 滤波池用于视觉注意力**：将多尺度方向先验以固定滤波池形式嵌入 Transformer，为捕捉边缘、纹理等几何结构提供了新思路，可结合本团队的 CV 方向进行实验验证。
3. **MF–tanh 方案在端侧部署的价值**：完全消除乘法的注意力机制对嵌入式/存算一体硬件极具吸引力，可作为低功耗推理的候选组件。
4. **参数量估算与消融的标准化**：论文以清晰的参数量差值（12%）和计算复杂度公式支撑论点，实验设计中对齐训练设置（epoch/patch size/batch）的做法值得借鉴。
5. **Q/S/K 三者的角色解耦实验**：Shearlet 四种构型系统地分离了"方向信息影响注意力权重"vs"影响输出表示"两个假设，这种**控制变量的消融策略**可在后续研究中复用。

## 关键术语表
- **Dual-headed Transformer（双头 Transformer）**：将标准 QKV 三投影注意力简化为仅学习 Q、K 两个投影，value 由固定变换替代的新型注意力单元。
- **Walsh–Hadamard Transform (WHT)**：完全二进制的正交变换，仅需加减法即可实现，是最计算高效的 value 替代方案。
- **Multiplication-Avoiding (MA) 算子**：通过 sign/abs 运算组合替代浮点乘法，用于构造免乘的 value 张量。
- **Shearlet Transform（剪切波变换）**：基于锥适配滤波的多尺度方向表示，能同时捕获图像的尺度与方向特征。
- **Normalized MF–tanh Attention**：用符号化 tanh 替换 softmax、以无乘算子构造 value 的新型注意力机制，保留负向交互信号。
- **KV-cache**：注意力推理中缓存 Key/Value 张量以减少重复计算；消除 V 投影可显著降低 KV-cache 的内存占用。
- **Scaled Dot-Product Attention**：标准注意力公式 softmax(QKᵀ/√d_k)V，本文在此基础上移除 V 投影。
- **Filterbank Fusion Coefficients**：Shearlet 滤波池中各方向子带的可学习融合权重，经 softmax 归一化。

## 可复现要素
- **数据集**：CIFAR-10、Tiny-ImageNet（论文使用固定划分 45K/5K 和 90K/10K 训练/选择集，未使用官方测试集进行模型选择）；ImageNet 仅在摘要提及但未给出实验细节。
- **代码/权重开源状态**：论文未提及，代码与权重**未公开**。
- **关键超参**：Mini-ViT：4 层 Transformer、128 维嵌入、4 头、MLP ratio=2、30 轮；大模型：6 层、D=192、3 头、MLP 768 维、100 轮、AdamW、batch=128、lr=3e-4、weight decay=0.05、warmup=5、cosine lr schedule。
- **复现难度**：中高（需实现 Shearlet 滤波池及多组对照实验，且缺官方代码参考）。
