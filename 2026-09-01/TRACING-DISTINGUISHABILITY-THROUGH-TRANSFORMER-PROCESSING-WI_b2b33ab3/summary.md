---
title: "TRACING-DISTINGUISHABILITY-THROUGH-TRANSFORMER-PROCESSING-WI"
source: https://arxiv.org/pdf/2608.30720v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 22:09:49"
---

# 论文速读：TRACING DISTINGUISHABILITY THROUGH TRANSFORMER PROCESSING WITH STOCHASTIC LAYERNORM

## 一句话总结
论文在 Transformer 的每个 LayerNorm 读取点注入各向同性高斯噪声，将点值残差流表示转化为随机表示，使"相似性"获得功能 grounded 的统计可区分性含义；通过全局 rate budget 与 softmax 分配的 per-tap 学习参数联合蒸馏微调，并用 Bhattacharyya 系数追踪扰动区分性在 MLP 块与 attention head 的 Q/K/V 投影中的深度传播，在 ViT-S 和 GPT-2 small 上揭示了连续视觉扰动响应曲线与 induction head 的 K 选择性模式。

## 研究问题与动机
1. **点表示几何与功能脱节**：深度网络内部状态通常为 measure-zero 的点值，邻近几何状态未必产生相似下游行为，现有表示相似性度量（如 CKA、重构失真）缺乏功能 grounding。
2. **现有随机化方法的架构代价**：信息瓶颈、变分自编码器等需在多处部署显式编码器-解码器瓶颈，引入大量额外参数与结构修改。
3. **LayerNorm 尺度约束未被利用**：Transformer 的 LayerNorm 天然将表示映射到零均值超球面 $\mathcal{M}_d$，但未被用来建立表示几何与功能之间的耦合通道。
4. **可区分性缺乏操作化定义**：现有可解释性方法关注"哪些内部状态几何相似"，但未系统回答"哪些输入区分的传播被保留/丢弃、由哪些子层处理"。

## 核心贡献（创新点）
1. **Stochastic LayerNorm**：在每个残差流读取点执行归一化→加各向同性高斯噪声→重新归一化→仿射变换，仅引入约 $2L$ 个额外标量参数，将点表示扩展为随机 channel。
2. **Rate budget + softmax allocation**：全局固定总速率上界 $B$，通过可学习 logits 经 softmax 分配 per-tap 速率，配合预计算 σ↔rate 查找表，使蒸馏损失驱动 water-filling 分配，无需显式 rate penalty。
3. **Bhattacharyya 系数追踪框架**：将局部后验近似为 vMF 分布，量化连续/离散扰动在各深度 MLP 与 attention head Q/K/V 投影中的可区分性传播，提供信息论意义下的功能 grounding。
4. **Induction head 的 K 选择性发现**：GPT-2 small 中 7 个 attention head 的 K 投影显著保留单 token 扰动区分性，其中 5 个与高 induction-score head 重合，另 2 个（L5H8 fuzzy、L6H9 canonical）在 canonical attention-pattern 统计中弱可见，揭示了头级功能特化的新探针。

## 方法详解
- **Stochastic LayerNorm 构造**：$z = n_\epsilon(u),\quad v = n_\epsilon(z + \eta),\quad \eta \sim \mathcal{N}(0, \sigma_t^2 I)$，其中 $n_\epsilon(u)=(u-\langle u\rangle)/\sqrt{\mathrm{Var}(u)+\epsilon}$，$\epsilon$ 设为 0。仿射变换 $\gamma\odot v+\beta$ 在噪声 channel 之后应用。
- **Rate 标定**：vMF 分布 $\mathrm{vMF}(\mu,\kappa)$ 的信息率上界 $R_d(\sigma_t)=\log A_d - h(V|U)$，通过 mean resultant length $\rho=A_p(\kappa)$ 反推 $\sigma=\sqrt{1/\rho^2-1}$，预计算 4000 点单调查找表支持可微插值。
- **Per-tap 分配**：logits $a_t$ 经 softmax 得 $B_t=B\cdot\mathrm{softmax}(a)_t$，满足 $\sum B_t=B$；精确零速率仅在 $\sigma\to\infty$ 极限达到，实际受查找表下限约束（ViT $\kappa_{\min}=5$，GPT-2 $\kappa_{\min}=76.3$）。
- **局部后验近似**：各向同性高斯加 renorm 诱导 projected-normal 分布，用 vMF 近似 $p(v|u)\approx C_p(\kappa)\exp(\kappa\mu(u)^\top v)$，$\kappa$ 与 $\sigma$ 对应通过数值标定存储。
- **Bhattacharyya 系数**：$\mathrm{BC}(p_i,p_j)=\int\sqrt{p(v|u_i)p(v|u_j)}dv$，对共同 $\kappa$ 的 vMF 有闭式；$\mathrm{BD}=-\log\mathrm{BC}$ 作为可区分性度量，满足 data-processing inequality（BC 单调不减）。
- **Q/K/V 投影追踪**：投影矩阵 $B_A=\sqrt{d}\,A\,\mathrm{diag}(\gamma)E$ 作 polar 分解 $B_A=S_A P_A$，BC 在 $S_A$ 与平移下不变，只需计算 $W_A=P_A X$ 的 pushforward 密度 $g^A_\mu(w)$ 积分，用 Monte Carlo 采样估计 $\widehat{\mathrm{BC}}_A$。
- **Common Random Numbers (CRN)**：配对输入共享相同上游噪声序列 $\epsilon_{1:L}$，使轨迹差异主要由输入扰动驱动；完全耦合得 $\mathrm{BC}_L$（下界），零共享得 $\mathrm{BC}_0$（有效后验），$\mathrm{BC}_0\geq\mathrm{BC}_1\geq\cdots\geq\mathrm{BC}_L$。
- **训练**：仅蒸馏损失 $\mathcal{L}_\mathrm{distill}=\mathbb{E}_x[D_\mathrm{KL}(p_\mathrm{teacher}(\cdot|x)\|p_\mathrm{student}(\cdot|x))]$，无 rate penalty；噪声从 $\sigma_g=0.05$ 在前 6/25 epoch ramp 至目标值；AdamW、cosine LR、grad-norm clipping 1.0、bfloat16 autocast。

## 实验与结果
- **数据集与模型**：ImageNet-1k（ViT-S，timm `vit_small_patch16_224`，$D=384$，$224^2$）；OpenWebText（GPT-2 small，124M，$D=768$，512-token 非重叠块）。
- **行为转折阈值**：ViT-S 与 GPT-2 small 均在 $B\sim10^4$ nats 附近出现 knee，高于此值行为接近 base model，低于此值 distillation loss 与验证性能同步下降。
- **Per-tap 分配格局**：两模型深度方向分配相对平坦，attention tap 普遍低于同层 MLP tap；最终 readout 最受保护；GPT-2 中 layer 1 读写与 layer 11 attention 最早触达 rate floor（$B=1.5\times10^4$ nats）。
- **ViT 视觉响应曲线**：200 张验证图上的 blur/hue rotation/Gaussian noise/fading（按 $\|\Delta x\|_2$ 等距比较）显示 layer 6 MLP 与最终 readout 对 blur 最敏感、对 fading 最不敏感，排序在深度中大致保持；blur 幅度增加使 BC 单调下降（区分性增加）。
- **GPT-2 K 选择性**：80 对单 noun swap 扰动（层 5 attention read 处 BD ≤ 50 nats 筛选），$B=1.5\times10^4$ nats 模型中 7 个 heads 的 K 投影保留分数显著高于随机 64 维投影基线（≈0.1）；5 个重合于 Nanda 2022 induction mosaic 高诱导分 heads，L5H8（fuzzy induction）与 L5H5/L6H9（canonical induction）亦入选；L5H8、L4H7 在 prefix-matching score 近零时仍保留强 K 选择性；除 L4H7 外其余 heads 选择性仅限 K（Q、V 贴近随机基线）；层 5 头部 K 选择性随距扰动 token 距离衰减至网络基线。
- **跨流相关性扰动**：ViT 打乱残差流跨 position 联合结构降低 distillation loss（后期深度效应更强），GPT-2 早期打乱轻微恶化、后期轻微改善，反映 ViT 冗余聚合与 GPT 因果嵌套依赖的模态差异。

## 相关工作脉络
1. **表示相似性分析（CKA 等）**：Kornblith et al. 2019，基于点表示的几何度量，本文以随机区分性替代，解决几何-功能脱节问题。
2. **信息瓶颈 / 变分自编码器**：Tishby et al. 2000; Kingma & Welling 2014; Alemi et al. 2017，需显式 encoder-decoder 瓶颈，本文直接利用 LayerNorm 固有尺度约束，避免额外瓶颈架构。
3. **Sparse Autoencoder / Monosemanticity**：Huben et al. 2024; Bricken et al. 2023，基于点重构失真拟合稀疏代理模型，本文的区分性度量直接耦合模型函数行为。
4. **Induction head 与 IoI circuit**：Olsson et al. 2022; Wang et al. 2023，本文通过 K 投影区分性保留分数揭示 canonical attention-pattern 统计中弱可见的 induction 结构（如 L5H8 fuzzy head）。
5. **先前信息控制工作（Murphy 2026）**：从低信息极限向上解释有用计算；本文从预训练确定性解出发向下限制内部通道，二者方向相反。
6. **Data-processing inequality 用于表征分析**：Csiszar 1967; Ali & Silvey 1966，本文将其用于约束 residual stream read 到 sublayer 的区分性单调衰减，建立信息论分析基础。

## 局限性与未来方向
1. vMF 近似仅描述局部后验，upstream 随机性导致的 effective posterior 为 mixture 且可能更宽，近似误差在深层累积。
2. Rate 上界的紧密度依赖于测量状态在 LayerNorm 球面上的边际均匀性，实际偏离均匀时 gap 增大。
3. 实验仅在 ViT-S 与 GPT-2 small 验证，未扩展到更大规模模型或更多架构（如 post-norm、Mixture-of-Experts）。
4. 分析局限于单个 read 与单个 head 投影，未覆盖 attention block 可访问的 joint distinctions。
5. Rate-limited 模型上的结论向 unconstrained 预训练模型迁移的条件尚未系统验证。
6. 未来方向：改进 effective posterior 近似（如 mixture vMF 或样本主导估计）、扩展到百亿级以上模型、分析 joint attention distinguishability、探索迁移定理。

## 研究启发与可借鉴点
1. **Stochastic LayerNorm 可直接迁移**：仅替换 `nn.LayerNorm` 为带 σ 参数的 noisy 版本，适配 LLaMA、Swin、Conformer 等广泛使用 LayerNorm 的架构，作为 low-cost 随机正则化或功能性探针。
2. **Rate budget + softmax allocation 可推广**：避免信息瓶颈的退化解（全部压缩后支付一次 penalty），适用于任何需要多点容量约束的表示学习场景（如多尺度特征金字塔、多任务共享瓶颈）。
3. **BC 追踪 + CRN 方差缩减可复用**：对任意注入噪声的网络，共享上游随机数序列可分离"输入扰动效应"与"噪声方差效应"，适用于 diffusion model、随机 depth 网络的可解释性分析。
4. **K 选择性分析启发 head 功能探针**：将区分性追踪从"几何相似"转为"功能可用"，可与 mechanistic interpretability 的 activation patching、circuit tracing 结合，形成信息论-机制双视角。
5. **噪声 ramp 训练策略稳健**：从 near-deterministic 逐步加入随机性（前 6/25 epoch ramp），避免早期货损，可作为 stochastic regularization 微调的通用 warm-up 范式。

## 关键术语表
**Stochastic LayerNorm**：在标准 LayerNorm 归一化后注入各向同性高斯噪声并重新归一化，使残差流读取变为有限精度随机 channel，仅增加约 2L 个标量参数。

**Rate budget (B)**：全局固定的信息传输率上界（单位 nats），通过 softmax 分配到各 tap，
