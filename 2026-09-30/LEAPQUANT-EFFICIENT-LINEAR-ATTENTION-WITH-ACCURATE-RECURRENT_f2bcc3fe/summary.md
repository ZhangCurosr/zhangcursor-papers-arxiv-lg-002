---
title: "LEAPQUANT-EFFICIENT-LINEAR-ATTENTION-WITH-ACCURATE-RECURRENT"
source: https://arxiv.org/pdf/2609.38166v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:51:47"
field: "大语言模型推理优化"
keywords: ["linear attention", "recurrent state quantization", "post-training quantization", "LLM inference optimization", "GDN", "KDA"]
innovations: ["Per-window quantization 将量化频率从每token降至每窗口，显著减缓误差累积", "Compensator Tokens 以rank-one外积形式高精度保留状态异常值", "Residual smoothing 逐通道缩放残差降低量化动态范围"]
benchmarks: ["AIME 2026", "GPQA-Diamond", "MMLU-Pro", "LiveCodeBench v6", "GSM8K"]
---

# 论文速读：LEAPQUANT-EFFICIENT-LINEAR-ATTENTION-WITH-ACCURATE-RECURRENT

## 一句话总结
LeapQuant 提出了一种无训练的循环状态量化方法，通过窗口化量化（per-window quantization）减少误差累积，并引入补偿 Token（Compensator Tokens）和残差平滑技术来保留主要异常值，在 8-bit 下实现对 FP32 基准的近无损精度，同时将内核级推理加速 2.05–3.70×。

## 研究问题与动机
- 线性注意力（如 GDN、KDA）将上下文压缩为固定大小的循环状态，虽减少了长上下文计算开销，但推理时每个 token 都需要从 HBM 读取和写入完整状态，成为带宽瓶颈。
- 直接对循环状态进行低比特量化（如每步重新量化）会导致舍入误差随长序列逐步累积，且状态矩阵中存在集中分布在少数行列的异常值，进一步扩大量化动态范围，导致精度严重下降。
- 现有量化方法（如 KVQuant、QuaRot、TurboQuant）主要针对 KV 缓存或激活设计，未考虑线性注意力状态特有的递归更新模式。
- 目标是在保持精度的前提下，显著降低循环状态的显存占用和推理延迟，尤其适用于长推理（long thinking traces）场景。

## 核心贡献（创新点）
- **Per-window quantization**：每窗口（p 个 token）仅量化一次循环状态，而非每 token 一次，将量化频率降低 p 倍，从而大幅减缓误差累积。与直接 per-token 量化的本质区别在于利用窗口内固定低精度状态 + 高精度缓冲更新的组合来近似递归更新。
- **Compensator Tokens**：在每次窗口边界量化前，提取状态中能量最大的 r 个 rank-one 分量作为高精度补偿 Token，使剩余残差的异常值幅度下降一个数量级，从而显著降低单次量化的量化误差。与 SVDQuant、GEAR 等吸收异常值的方法不同，补偿 Token 与真实 token 共享相同的更新路径，无需额外 kernel 修改。
- **Residual Smoothing**：在量化残差前对 key 维度进行逐通道缩放平衡，进一步压缩动态范围，使低比特量化更均匀。该方法无需校准数据，完全无训练。

## 方法详解
- 线性注意力递推形式：$S_t = \text{Diag}(\alpha_t) S_{t-1} + k_t (v_t - S_{t-1}^\top \beta_t)^\top$，输出 $o_t = S_t^\top q_t$。
- **Per-window quantization**：在每个窗口起始处存储量化后的边界状态 $\hat{S}_0$，窗口内缓冲 p 个更新元组 $(\alpha_i, k_i, u_i)$（其中 $u_i = v_i - S_{i-1}^\top \beta_i$），在窗口结束 $\ell = p$ 处重建 $S_p$ 并量化为 $\hat{S}_p$。中间步骤不使用 dequantized 状态参与递推，误差仅在每个窗口边界引入一次。
- **Compensator Tokens**：在量化 $S_p$ 前，用 power iteration 拟合 rank-r 近似 $\tilde{K}\tilde{U}^\top = \sum_{h=1}^{r} \tilde{k}_h \tilde{u}_h^\top$，将其分离为高精度分量，对残差 $R_p = S_p - \tilde{K}\tilde{U}^\top$ 进行 b-bit 量化，得到 $\hat{R}_p$，重建状态为 $\tilde{S}_p = \text{dequant}(\hat{R}_p) + \tilde{K}\tilde{U}^\top$。补偿 Token 直接插入下一窗口的更新序列头部。
- **Residual Smoothing**：计算残差 $R_0$ 每行（key 维度）的平均绝对值平方根得到缩放向量 $c$，构建 $C = \text{Diag}(c)$，量化前做变换 $\hat{R}_0^C = \text{quant}(C^{-1} R_0)$，decode 时读出后乘以 $C$ 恢复原始坐标，缩放参数 $C$ 每窗口重新计算。
- 状态读出公式（含 smoothing 和补偿 Token）：
  $$S_\ell^\top x = \text{dequant}(\hat{R}_0^C)^\top C \Gamma_{1:\ell} x + \tilde{U}(\tilde{K}^\top \Gamma_{1:\ell} x) + \sum_{j=1}^{\ell} u_j (k_j^\top \Gamma_{j+1:\ell} x)$$

## 实验与结果
- **数据集**：AIME 2026（数学推理）、GPQA-Diamond（科学 QA）、MMLU-Pro（多任务语言理解）、LiveCodeBench v6（代码生成）、GSM8K（算术推理）。
- **模型**：Qwen3.5-9B、Qwen3.5-35B-A3B、Qwen3.8-Flash（GDN）；Kimi-Linear-48B-A3B-Instruct、GLM-5.3-Flash（KDA）。
- **GPU**：NVIDIA B200、RTX PRO 6000、RTX 5090。
- **精度结果（8-bit）**：LeapQuant 在全部 12 个 model-task 对上与 FP32 几乎持平（平均 75.5% vs 75.5%），而 FP8 仅 34.2%，INT8 仅 26.3%，BF16 per-step 在 AIME 上从 87.9% 降至 72.1%。
- **更低比特**：6-bit 下 LeapQuant 平均 72.4%（FP32 为 75.5%），TurboQuant 和 NVFP6 分别仅 31.9% 和 29.6%；4-bit 下 LeapQuant 平均 60.4%，且在 Kimi 模型所有任务上与 FP32 差距不超过 1.1%。
- **速度提升**：Kernel 级加速 2.05–3.70×（B200、RTX PRO 6000、RTX 5090），End-to-end 推理加速 1.47×；状态内存流量减少 3.4×，带 prefix caching 的端到端显存减少最高 56%。
- **消融**：逐组件加入时，仅 per-window 量化即可将 INT8 AIME 从 7.1% 提升至 82.4%，再加补偿 Token 至 86.6%，再加平滑至 87.9%（与 FP32 一致），最终 kernel 加速 2.52×。窗口长度 $p=16$、补偿 Token 数 $r=4$（4-bit 时 $r=8$）为最优配置。

## 相关工作脉络
- **FLA（Flash Linear Attention）**：高效线性注意力 kernel 库，聚焦全精度实现，未涉及状态存储精度优化。
- **KVQuant / QuaRot / TurboQuant**：原为 KV 缓存或激活量化设计，本文适配到循环状态但 per-step 重新量化导致严重精度损失，说明线性注意力状态的递归特性需要专门处理。
- **SVDQuant / GEAR**：通过低秩分解吸收异常值再量化残差，但面向 Transformer 权重/KV 缓存，其奇异值分解思路与 LeapQuant 的 rank-one 补偿有相似思想，但后者与递推更新形式天然兼容。
- **Quamba / MambaQuant / Quamba2**：针对 Mamba 选择性 SSM 的状态量化方法，Mamba 的状态更新机制与线性注意力不同，不能直接迁移。
- **ReplaySSM**：通过保存 checkpoint 状态支持 speculative decoding 的回滚，与 per-window 量化共享"从边界状态回放更新"的结构，但目标是功能支持而非精度-效率权衡。
- **SmoothQuant**：将激活异常值迁移到权重中，是权重量化预处理技术，LeapQuant 在状态空间中以不同机制（补偿 Token + 平滑）解决类似问题。

## 局限性与未来方向
- 论文主要验证了 GDN 和 KDA 两种线性注意力变体，对其他线性注意力架构（如 RWKV、RetNet、DeltaProduct）的泛化能力未充分验证（附录 A 提供了通用递推形式，但未给出完整实验）。
- 窗口长度 $p$ 和补偿 Token 数 $r$ 的选取依赖经验调优（$p=16, r=4$），在极端长上下文或高并发场景下的最优配置可能不同。
- 仅评估了 Post-Training Quantization（PTQ）场景，若结合少量微调（fine-tuning）可能进一步释放低比特极限（如 4-bit 下仍有性能差距）。
- 补偿 Token 在 $r$ 较大时会引入可见的 kernel 开销（$r=16$ 时甚至慢于 FP32），如何在更多补偿维度上保持透明性值得探索。
- 未讨论与 speculative decoding、prefix caching 等先进 serving 技术的集成方式（虽与 ReplaySSM 有结构相似性）。

## 研究启发与可借鉴点
- **误差累积的最小化策略**：per-window 量化的核心思想——将高频量化降为低频量化，在误差传播链路上"断链"——可迁移到其他递归模型（如 SSM、RNN 变体）的低精度部署。
- **异常值的高精度保留机制**：Compensator Tokens 以 rank-one 外积形式提取主导模式，与真实 token 共享 update path，这种"结构化异常值分离"策略可推广到任意低秩主导的矩阵流（如 KV 缓存、状态空间模型）。
- **逐通道平滑预量化**：Residual Smoothing 的逐 key-channel 缩放技术简单有效，可与多种量化方案（INT8、FP8、MXFP）组合使用，提升低比特下的均匀性。
- **无训练 + 零校准数据的可行性**：全文无需校准集或微调，工程落地成本低，适合快速适配新模型。
- **与现有 serving 框架的集成范式**：基于 vLLM 和 TileLang 的 kernel 级实现方案，展示了如何将新型量化策略无缝嵌入生产推理系统，可作为后续工作的参照模板。

## 关键术语表
- **Linear Attention（线性注意力）**：将标准 softmax 注意力替换为固定大小状态的递归更新，避免 $O(n^2)$ 计算，常用形式包括 GDN 和 KDA。
- **Recurrent State（循环状态）**：线性注意力层中累积 token 历史的矩阵 $S_t \in \mathbb{R}^{d_k \times d_v}$，每个 decode 步骤需完整读写。
- **Per-window Quantization（窗口量化）**：每 p 个 token 才量化一次状态，窗口内用固定低精度状态加高精度缓冲更新计算输出，降低量化频率。
- **Compensator Token（补偿 Token）**：以高精度 rank-one 外积 $\tilde{k}\tilde{u}^\top$ 表示状态中的主导异常模式，与真实 token 共享更新路径，帮助残差更均匀分布。
- **Residual Smoothing（残差平滑）**：量化前对残差矩阵的 key 维度进行逐通道缩放，使各行量级更均衡，降低量化动态范围。
- **Gated DeltaNet (GDN)**：Qwen 系列采用的线性注意力变体，具有数据依赖的衰减和对角 delta-rule 更新。
- **Kimi Delta Attention (KDA)**：Kimi 系列采用的线性注意力变体，与 GDN 结构类似但衰减和读取向量形式略有不同。
- **HBM（High Bandwidth Memory）**：GPU 高带宽内存，线性注意力 decode 阶段的状态读写瓶颈主要来自 HBM 带宽限制。

## 可复现要素
- **数据集**：AIME 2026、GPQA-Diamond、MMLU-Pro、LiveCodeBench v6、GSM8K（均为公开基准）。
- **代码/权重**：论文未明确说明代码开源状态，但提到基于 vLLM 和 TileLang 实现；模型权重为官方公开模型。
- **关键超参**：窗口长度 $p=16$，补偿 Token 数 $r=4$（4-bit 时 $r=8$），平滑尺度用 FP32 精度；所有模型使用官方 model card 中的 sampling 参数。
