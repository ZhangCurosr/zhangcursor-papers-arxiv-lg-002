---
title: "THE-TOKEN-BEFORE-THE-VALUE-IS-THE-KEY-HOW-HYBRID-ARCHITECTUR"
source: https://arxiv.org/pdf/2609.15545v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:22:59"
field: "混合架构机制可解释性"
keywords: ["hybrid architectures", "induction circuits", "mechanistic interpretability", "Gated DeltaNet", "sliding window attention", "activation patching", "circuit allocation"]
innovations: ["提出层类型无关的配对探针，统一测量Carrying和Matching跨GDN/SWA/Transformer", "揭示hybrid中Carrying集中于高效层、Matching集中于全局接收层的分工规律，且lag-one token是核心", "建立upstream predecessor support→downstream Matching allocation的因果链，通过source-key restoration验证信息传递通道"]
benchmarks: ["OpenWebText 30K步训练", "10K自然文本序列（Wiki/Python/Math）PPL评估", "Qwen3-4B/Qwen3.5-4B预训练checkpoint验证"]
---

# 论文速读：THE-TOKEN-BEFORE-THE-VALUE-IS-THE-KEY-HOW-HYBRID-ARCHITECTUR

## 一句话总结
本文通过**层类型无关的配对探针（layer-type-agnostic paired probes）**方法，系统揭示了混合架构中 induction circuit 的计算分配规律：**Carrying 集中于高效层、Matching 集中于全局接收层**，且 **value 前面那个 token（lag-one）是准备 source key 的核心**。改变前置支持条件会重定位 circuit，并影响自然文本召回。

## 研究问题与动机
1. **混合架构的能力提升机制尚不明确**：Gated DeltaNet（GDN）、滑动窗口注意力（SWA）与全注意力 Transformer 的混合设计已被证明能提升效率与能力，但"架构互补性如何转化为学到的计算"仍是开放问题。
2. **现有探针方法不适用于混合层**：传统 attention-score diagnostics 无法直接应用于 recurrent 组件，缺乏跨层类型统一的测量接口。
3. **induction circuit 的层间分工缺乏实证刻画**：Singh et al. (2024) 提出的 Carrying-Matching-Copying 三子电路在 Transformer 中的形成动力学已知，但在 hybrid 中的空间-时间分配规律尚未被系统研究。
4. **local 结构与 distant retrieval 的连接机制不清**：混合架构中局部操作（convolution/window）如何为远距离 source 访问做准备，以及这一准备如何影响后续 Matching 的选址，缺乏因果干预证据。

## 核心贡献（创新点）
1. **提出层类型无关的配对探针**：通过统一的 block-update 接口同时测量 Carrying 和 Matching，无需依赖 attention weights 或 recurrent state coordinates，首次实现跨 GDN/SWA/Transformer 的可比测量。
2. **揭示 hybrid 的明确分工规律**：Carrying 集中于高效层末尾（如 GDN L2 / SWA L2），Matching 集中于其后全局接收层（L3/L7）；lag-one 关系（value 前一个 token）是 local 计算的核心贡献。
3. **建立上游条件→下游分配的控制链条**：通过 lag-one masking、convolution removal、early LR reduction 等干预，证明改变 Carrying 学习条件可重定位 Matching，且 source-key（K）是传递影响的主要通道（K-only propagation 接近 full-path loss）。
4. **连接 circuit 组织与自然文本召回**：证明同一 predecessor-token 关系在自然文本预测中同样重要（GDN 中 source-attending heads 消融对 NLL 影响大于 comparison heads）；窗口大小与训练数据 enriched 程度共同决定 circuit 形成时机与最终性能。

## 方法详解
**核心探针设计**：
- **Prompt 构造**：$c = A B \ldots C D \ldots A_q$，目标续写为 $B$；$s$ 为 source 位置，$q$ 为 query 位置，margin $m = z_B - z_D$ 为 correct-minus-distractor logit margin。
- **Activation patching**：将 donor prompt 的 block update（输出减输入，含 FFN）RMS-matched 后插入 clean run，测量 margin 变化。
- **Lag probe（Algorithm 1）**：扫描历史值前 $r \in \{1,2,4,8,16,32,64\}$ 个 token 的贡献，定位 predecessor relation。
- **Carrying probe（Algorithm 2）**：替换历史值的前驱 token（B 的前驱），patch 在 value 位置 $s$，测量 source 准备效应。
- **Matching probe（Algorithm 3）**：交换历史 keys（$AB...CD...A_q \mapsto CB...AD...A_q$），patch 在 query 位置 $q$，测量检索响应。

**干预实验设计**：
- **Early LR reduction**：前 3K（GDN）/4K（SWA）步将选定 efficient 块的 LR 乘 0.1（含 mixer/FFN/norm）。
- **Lag-one mask**：仅抑制 lag-one convolution channel，保留其他 offset 与 recurrent computation。
- **Convolution removal**：完全移除 local convolution。
- **SWA window 变化**：window 2/8/16 对比不同 local 感受野。
- **Induction-enriched text**：基于 bigram frequency + reliability 过滤的训练数据（70% enriched 前 5K 步，之后 50%）。

**Source-key 追溯**：
- K-only/V-only propagation：仅转移 clean/donor 的 K 或 V 到 receiver，测量 margin 变化。
- Source-key restoration：在 sender-patched run 中恢复 clean K，测量 prediction recovery。
- Fixed-value selection：固定 V，交换 attention pattern，隔离 content selection 效应。

## 实验与结果
**模型配置**：
- 8-layer GDN hybrid：$[GDN^3, TF]^2$，width=512，8 heads，~79M 参数
- 8-layer SWA hybrid：$[SWA_4^3, TF]^2$，~77M 参数
- 8-layer Full-attention Transformer：~77M 参数
- 训练：OpenWebText packed sequences，context=1024，30K steps，AdamW，LR=$3 \times 10^{-4}$

**主要发现**：
| 发现 | 关键数字 |
|------|----------|
| Baseline GDN: Carrying 在 L2，Matching 在 L3 | Carrying ≈ +2.4 margin, Matching ≈ +5.85 margin |
| LR×0.1 干预后：Carrying 移至 L6，Matching 移至 L7 | Matching 从 5.85 → 9.16 margin |
| K-only propagation 恢复 nearly full-path loss | K/full ≈ 1.0（GDN Baseline L2→L3: 2.165/2.236） |
| SWA window 2 比 window 16 更早形成 functional Carrying | Window 2: Carrying τ=0.5 在 (1,2]K; Window 16: (2,3]K |
| Induction-enriched text 加速 Transformer circuit 形成 | Transformer/Enriched 比 Natural 早约 1K 步达到阈值 |
| GDN conv removal 降低 multiple-continuation PPL | SWA4 LR×0.1: Multiple mid PPL 12.94 vs Baseline 15.52 |
| Transformer LR×0.1 普遍损害召回 | Overall PPL 从 41.47 → 51.11 |

**自然文本评估**（10K sequences: 2K Wiki + 4K Python + 4K Math）：
- GDN Baseline identifier reuse PPL: 29.00；Conv removed: **18.72**（最佳）
- SWA16 identifier reuse PPL: **20.91**（最佳）；SWA4 LR×0.1: 23.49
- Head ablation：source-attending heads 消融对 NLL 影响 > comparison heads

**Qwen 预训练模型验证**：
- Qwen3-4B（full-attention）：Carrying 与 Matching 分离在不同全注意力层
- Qwen3.5-4B（$[GDN^3, FullAttn]^8$）：最强 Carrying 在 GDN L18，紧邻最强 Matching 的全注意力层 L19
- Source-K restoration 恢复 98% sender-patch loss（1.826/1.850 clean margin）

## 相关工作脉络
1. **Singh et al. (2024)**：首次系统识别 Carrying-Matching-Copying 三子电路及其形成动力学；本文扩展至 hybrid 架构并建立 upstream-downstream 因果链。
2. **Elhage et al. (2021); Olsson et al. (2022)**：经典 induction heads 理论框架；本文用层类型无关探针将其推广到 non-attention 层。
3. **Qiao et al. (2026)**：证明 efficient attention 作为 optimization prior 影响 retrieval-head 形成时机；本文进一步追溯上游 Carrying 准备对 formation timing 的控制。
4. **Arora et al. (2024, 2025); Jelassi et al. (2024)**：比较 recurrent 模型与 attention 的回忆能力差异；本文通过 common update interface 区分 historical-source 路径与 query-time recurrent 路径。
5. **Cabannes et al. (2026)**：解释 short-window 优势源于 recurrent memory 路径增强；本文通过 pure GDN/GDN-SWA 控制验证 historical-source Carrying 与 query-state retrieval 的本质区别。
6. **Cooper et al. (2026); Merrill et al. (2026)**：理论分析 hybrid 的 expressivity-efficiency tradeoff；本文从 circuit 层面提供实证支撑，连接架构先验与 learned computation。

## 局限性与未来方向
1. **模型规模有限**：主要实验在 ~77-79M 参数小规模模型上进行；虽然扩展到 Qwen3.5-4B，但 318M 中间规模的结果仍显示较高 variance（如 318M Baseline identifier PPL 21.17±3.08）。
2. **干预的长期行为未充分探索**：early LR reduction 的效果仅在 30K 步评估，未追踪更后期（如 100K+）是否出现 compensation 或进一步 reorganization。
3. **多阶段依赖的复杂性**：318M 模型显示 source-key 依赖可跨多个 global receiver（L7 和 L11），但干预对 multi-stage cooperation 的影响尚未系统研究。
4. **natural-text recall 的 trade-off 未完全解释**：某些干预（如 GDN conv removal）改善 identifier reuse 但损害 single-continuation far PPL，其机制与 architectural prior 的交互需进一步分析。
5. **位置编码的作用未分离**：SWA 使用 RoPE 而 full-attention 使用 learned absolute position，两者差异可能混淆 local vs global 层的比较结论。

## 研究启发与可借鉴点
1. **层类型无关探针的设计范式**：通过统一 block-update 接口（record output-input → RMS match → recompute）实现跨 attention/recurrent/local 操作的标准化测量，可直接迁移至 Mamba、DeltaNet、H3 等其他 hybrid 架构的 circuit 分析。
2. **upstream conditioning → downstream allocation 的因果实验框架**：通过 selective intervention（lag-one mask / conv removal / early LR）建立"学习条件→circuit 选址→行为输出"的完整证据链，适合用于诊断任何 hybrid 模型的分工机制。
3. **formation timing 与 architectural prior 的联动分析**：结合 window size、enriched text、LR schedule 等多维变量，追踪 functional circuit 的形成曲线（onset thresholds），为"何时需要何种架构支持"提供量化依据。
4. **source-key 作为 information bottleneck 的假设**：K-only propagation 接近 full-path loss 的发现提示，hybrid 架构中 source representation 的 key 分量可能是跨层信息传递的主要载体，值得在更大模型中验证并探索显式正则化。
5. **natural-text evaluation hierarchy 的设计**：single/multiple continuation × near/mid/far recency × identifier/entity/symbol reuse 的多维评估框架，可复用为任何 recall 相关研究的标准化 benchmark。

## 关键术语表
- **Carrying**：将 predecessor 信息（尤其是 value 前一个 token）编码入 historical value 的 source representation，为后续远距离检索做准备。
- **Matching**：基于内容（key）选择正确 source 的检索操作，发生在 global receiver 层，依赖 Carrying 准备的 source key。
- **Copying**：将选中 source 的 value 传递到 query 位置的输出操作，完成 induction 任务的最终续写。
- **Lag-one relation**：historical value 前紧邻的一个 token（即 "the token before the value"），是 local computation 贡献最集中的 predecessor relation。
- **Layer-type-agnostic probe**：不依赖 attention weights 或 recurrent state 坐标，通过统一 block-update 接口测量跨层类型功能的探针方法。
- **Activation patching**：将 donor prompt 的 block update 替换进 clean run，通过 margin 变化量化特定层/位置的因果贡献。
- **Source-key restoration**：在 sender-patched run 中恢复 clean source 的 K（而非 V），测量对 prediction recovery 的贡献，用于追踪信息传递通道。
- **Induction-enriched text**：基于 bigram frequency 和 continuation reliability 过滤的训练数据，保留重复且可预测的 continuation，加速 induction head 形成。

## 可复现要素
- **数据集**：OpenWebText（主训练）；评估使用 10K 序列（2K Wiki + 4K Python + 4K Math），论文未明确公开评估集，但描述了选取规则（Appendix F.1）
- **代码**：开源，https://github.com/ckpassenger/bind-match-copy/tree/main
- **权重**：受控模型（GDN/SWA/Transformer hybrid）需从头训练；Qwen3-4B 和 Qwen3.5-4B 使用官方 released checkpoints
- **关键超参**：
  - Width=512, Heads=8, FFN dim=2048
  - Context=1024, Batch size=64
  - AdamW (β₁=0.9, β₂=0.95), LR=3×10⁻⁴, warmup=1K, cosine decay to 3×10⁻⁵
  - 训练 30K steps；early LR reduction: LR×0.1 前 3K（GDN）/4K（SWA）步
  - SWA window 默认 4（含当前位置），变体 2/8/16
  - GDN convolution width 默认 4，变体 width=2
- **复现声明**：论文提供完整训练配置（Appendix A.1）与 probe 算法（Algorithm 1-3），支持独立复现
