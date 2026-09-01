---
title: "TRACING-DISTINGUISHABILITY-THROUGH-TRANSFORMER-PROCESSING-WI"
source: https://arxiv.org/pdf/2608.30720v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:20:06"
---

# 论文速读：TRACING-DISTINGUISHABILITY-THROUGH-TRANSFORMER-PROCESSING-WI

## 一句话总结
本文通过在预训练 Transformer 的每个 LayerNorm 处注入各向同性高斯噪声并重归一化，将点值隐层表示转化为具有“体积”的随机信道；借助 Bhattacharyya 系数与信息论工具追踪扰动区分度在 MLP 块与 Attention Q/K/V 投影中的传播路径，为模型可解释性提供了以函数行为而非几何距离为基础的分析视角。

## 研究问题与动机
- 现有深度网络表征分析多依赖点值状态的几何相似性（如 CKA、成对距离），但几何邻近与下游行为无内在绑定：近点可能产生不同行为，远点可能行为相似。
- 稀疏代理模型或重构误差方法（如 SAE）将几何假设强加于点值表示，缺乏与模型目标函数的直接耦合，难以反映表征在训练动力学下的功能性意义。
- 将表征视为随机分布可获得“体积”，使相似性转化为统计可区分性，从而自然接入数据 Processing 不等式与信息论工具，实现几何分析与功能行为的统一。
- 现有信息瓶颈方法需引入编码器-解码器瓶颈或软惩罚项，架构改动大且易退化为零速率；本文旨在以极轻量方式（仅修改 LayerNorm）实现受控随机化与功能锚定。

## 核心贡献（创新点）
- **轻量随机 LayerNorm 改造**：仅在每个残差流读取位置添加各向同性高斯噪声并重归一化，每层新增约 2L 个标量参数，无需改动 Transformer 主干与注意力结构。
- **行为锚定的区分度度量框架**：通过固定全局速率预算 $B$ 与逐点 softmax 分配 logits，将表征分析直接绑定到蒸馏损失与验证性能上，使区分度成为可测的功能性指标而非纯几何属性。
- **基于 vMF 近似与 CRN 的扰动追踪机制**：将局部后验近似为 von Mises–Fisher 分布，配合公共随机数方差缩减技术，定量刻画连续/离散扰动在 MLP 与 Attention 子层中的保留、丢弃与跨深度演化规律。
- **揭示 Attention Head 特异性扰动敏感性**：发现 GPT-2 Small 中多个高频诱导头及模糊诱导头的 Key 投影对单词扰动具有显著选择性保留能力，且该特性在经典 prefix-matching 注意力统计中往往被掩盖。

## 方法详解
- **噪声信道构造**：对预归一化 Transformer 的每个 LayerNorm，执行 $z = n_\epsilon(u)$，$v = n_\epsilon(z + \eta)$，其中 $\eta \sim \mathcal{N}(0, \sigma_t^2 I)$，随后施加原有仿射变换 $\gamma \odot v + \beta$。该操作将表示映射至均值为零、范数为 $\sqrt{d}$ 的超球面 $\mathcal{M}_d$，实现率受限读取。
- **速率预算与分配**：由旋转对称性可知局部信息速率上界 $R_d(\sigma_t) = \log A_d - h(V|U)$，仅依赖噪声尺度与维度。全局超参 $B$ 固定总预算，逐点分配 $B_t = B \cdot \text{softmax}(a)_t$，通过预计算单调查找表（4000 点几何网格）将 $B_t$ 映射为 $\sigma_t$，梯度可同时流向权重与分配 logits。
- **蒸馏微调策略**：仅优化 $\mathcal{L}_{\text{distill}} = \mathbb{E}_x[D_{\text{KL}}(p_{\text{teacher}}(\cdot|x) \| p_{\text{student}}(\cdot|x))]$，无显式速率惩罚项，避免软惩罚导致的零速率退化解。噪声在前 6/25 epoch 内从 $\sigma_g=0.05$ ramp 至目标值。
- **区分度追踪**：将局部后验近似为 vMF 分布 $p(v|u) \approx C_p(\kappa)\exp(\kappa \mu(u)^\top v)$，两状态 $u_i, u_j$ 的区分度由 Bhattacharyya 系数 $\text{BC}(p_i,p_j)$ 衡量，BC 越低区分度越高。MLP 独立处理各流，BC 单调不减；Attention 联合处理多流，输出区分度受联合可见读取约束。
- **Q/K/V 投影追踪**：对投影矩阵 $A$ 作极分解 $B_A = S_A P_A$，利用 BC 在可逆变换下的不变性，将问题降至行正交投影 $W_A = P_A X$。推导推前密度 $g^A_i(w)$ 后以蒙特卡洛采样估计 $\widehat{\text{BC}}_A$，量化每个 head 组件实际保留的扰动区分度。
- **公共随机数（CRN）**：对上下文物理噪声序列 $\epsilon_{1:L}$ 进行部分共享，构造 $\text{BC}_0 \geq \text{BC}_1 \geq \cdots \geq \text{BC}_L$，实现从有效后验重叠到局部条件后验重叠的插值追踪，降低配对比较的估计方差。

## 实验与结果
- **实验设置**：视觉任务微调 ViT-S (`timm` vit_small_patch16_224) 于 ImageNet-1k；语言任务微调 GPT-2 Small 于 OpenWebText。两模型均含 25 个 LayerNorm 读取点。
- **预算拐点**：ViT-S 与 GPT-2 Small 均在总读速率 $B \sim 10^4$ nats 处呈现行为拐点；高于此值时蒸馏损失与验证性能接近基线，低于此值时稳步退化，该拐点可作为预训练模型有效操作分辨率的探针。
- **视觉分辨率响应**：对 200 张验证图像施加模糊、色相旋转、加性高斯噪声、对比度衰减，以欧氏像素位移对齐增强幅度。CLS 流的 BC 随模糊程度平滑下降且跨深度差异显著；模型对模糊最敏感、对衰减最不敏感，相对排序在最终读取层基本保持。
- **跨流依赖探测**：ViT 中对残差流位置洗牌可提升蒸馏性能（去冗余噪声依赖），GPT-2 中早期洗牌轻微劣化、后期洗牌略有益处，反映视觉聚合型与语言因果嵌套型处理的本质差异。
- **语言 Head 选择性**：生成 80 对最小扰动句子（同类别名词替换）。在 $B = 1.5\times10^4$ nats 下，层 2-10 中 7 个 Head 的 K 投影保留的区分度比例显著高于随机 64 维投影基线（~0.1）。
