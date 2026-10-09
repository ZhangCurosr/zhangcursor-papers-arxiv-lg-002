---
title: "TRACE-Rollout-Guided-Quantization-Aware-Training-for-FP4-Rei"
source: https://arxiv.org/pdf/2610.07767v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:44:03"
field: "低精度大模型训练"
keywords: ["FP4 quantization", "reinforcement learning", "MoE language models", "quantization-aware training", "train-rollout alignment", "low-precision inference"]
innovations: ["Rollout-guided QAT：直接以rollout侧FP4码字为参照指导训练侧取整，最小化train-rollout discrepancy", "Mantissa-only compact caching：仅缓存后半层1-bit尾数与scale，将每token缓存从~50 KB降至~7.5 KB"]
benchmarks: ["LiveCodeBench v6", "AIME 2024/2025", "HMMT 2025", "Terminal-Bench 2.1", "DeepSWE v1.1", "GDPval"]
---

```markdown
# 论文速读：TRACE-Rollout-Guided-Quantization-Aware-Training-for-FP4-Rei

## 一句话总结
论文提出TRACE，一种面向MoE语言模型RL训练的FP4量化框架，通过rollout侧量化结果直接引导训练侧的FP4取整决策，在保持BF16级别性能的同时实现最高5.4倍的rollout加速。

## 研究问题与动机
- **RL训练中的rollout开销巨大**：RL post-training需要反复生成长轨迹，low-precision rollout（尤其是FP4）可显著降低成本，但FP4的粗粒度量化空间会在训练和rollout执行路径之间引入大量数值差异。
- **现有FP4方法优化目标错位**：QUADS、QaRL、Rollout-ResQ等方法主要独立优化训练路径和rollout路径各自的量化精度（minimize $\|X - Q(X)\|$），但这并不能直接最小化两条量化路径之间的差异（$D_{act}$）。
- **MoE模型对不匹配尤为敏感**：微小的数值差异可能改变router score并激活不同的expert，从而放大策略失配；vanilla QAT配合FP4 rollout会导致train-rollout discrepancy随训练递增并最终引发训练崩溃。
- **完整量化信息传递开销不可接受**：若缓存所有rollout侧的完整量化记录，单次RL step可产生约51 TB数据，即使异步通信也极易成为I/O瓶颈。

## 核心贡献（创新点）
1. **Rollout-Guided Quantization-Aware Training**：直接使用rollout侧FP4码字指导训练侧的取整选择，目标函数最小化的是$|q_{train} - q_{rollout}|$而非各自的高精度误差，这是与QUADS等方法在优化目标上的本质区别。
2. **Mantissa-Only Train-Rollout Communication**：基于“>99%的不匹配仅相差一个相邻FP4码字”的观察，仅缓存后半层模型的1-bit尾数与scale，将每token缓存量从~50 KB降至~7.5 KB，大幅降低存储与通信开销，而QUADS等方法未考虑此压缩机制。
3. **端到端框架与系统性评估**：将上述两项设计整合为完整的FP4 W/A + FP4 KV joint rollout训练流程，并在4个不同规模的Qwen MoE模型及推理/编程/长程三类RL任务上验证了其通用性与稳定性。

## 方法详解
### 3.1 Rollout-Guided Quantization-Aware Training
- **问题建模**：定义单点量化差异 $\mathcal{D}_{act} = \|Q_{FP4}^{train}(X_{train}) - Q_{FP4}^{rollout}(X_{rollout})\|_F$。
- **取整策略**：对每个训练侧激活 $x$，用rollout侧scale $s_{rollout}$ 归一化得到 $z = clip(x/s, -6, 6)$，取其相邻的两个E2M1 FP4码字 $\{q_-, q_+\}$，再选择离rollout侧精确码字 $q_{rollout}$ 最近的那个：
  $$q_{TRACE} = \arg\min_{q \in \{q_-, q_+\}} |q - q_{rollout}|$$
- **与RTN对比**：标准round-to-nearest产生的 $q_{RTN}$ 满足 $|q_{TRACE} - q_{rollout}| \leq |q_{RTN} - q_{rollout}|$，因此不会增加局部train-rollout discrepancy。
- **损失函数**：在前向使用STE（Straight-Through Estimator）$STE(x, s \cdot q^*)$，反向梯度仍回传到 $x$。

### 3.2 Mantissa-Only Train-Rollout Communication
- **动机**：全量缓存activation guidance (~45 KB/token) 与 KV guidance (~5.6 KB/token) 导致极高的GPU-CPU传输与磁盘I/O压力。
- **关键观察**：
  1. >99%的不匹配quantized值仅相差1个相邻FP4 codebook entry；
  2. 向更低码字的校正主要发生在更深层。
- **设计**：仅在模型后半部分（如40层中的后20层）缓存每个FP4值的1-bit mantissa code + scale，每token仅需 ~7.5 KB；rollout阶段异步写入临时GPU buffer并offload到CPU cache，QAT前再从磁盘加载切片至GPU。

### Algorithm 1 流程
1. Rollout阶段：生成trajectories并记录 $\mathcal{R} = \{(\text{Pack}(\log_b(q_{rollout})), s_{rollout})\}$ 在选定位置 $S$。
2. QAT阶段：对每个微批次中的激活 $x$，反解出 $q_{rollout}$ 的compact reference，执行上述nearest codeword选择，并用STE完成前向。

## 实验与结果
- **模型**：Qwen3.5-35B-A3B、Qwen3.5-122B-A10B、Qwen3.8-Flash-Next (125B/6B activated)、Qwen3.8-2.4T-A95B。
- **任务与基准**：推理（LiveCodeBench v6、AIME 2024/2025、HMMT 2025）、编程（DeepSWE v1.1、Terminal-Bench 2.1）、长程（GDPval）。
- **主要结果（Qwen3.5-35B-A3B，joint NVFP4 W/A/KV）**：
  - TRACE平均75.3 vs. BF16 rollout 74.9、QUADS 68.8、QAT 59.6；HMMT25从59.0提升至70.0（↑11.0）。
  - 在更大模型上均追平或超越BF16：DeepSWE 33.0 vs. 33.4、Terminal-Bench 70.6 vs. 68.8、GDPval 90.2 vs. 90.3。
  - MXFP4设定下同样有效：W4A8+MXFP4 KV平均75.1（↑6.1 over QAT），W4A4+MXFP4 KV平均73.5（↑6.3）。
- **效率**：128K输出长度下解码吞吐量达BF16的5.4×；完整rollout-guided版本吞吐下降明显，而TRACE保持与vanilla FP4相近；E2E RL step时间仅增加7.4%（664→713 ms），其中rollout guidance仅占4%（权重侧）+2%（KV侧）。
- **Training Dynamics**：QAT/QaRL/QUADS的train-rollout log-prob diff随训练递增并最终崩溃；TRACE全程保持与BF16相近的diff水平，且策略在约120步后逐步“适应”FP4 rollout，性能差距逐渐闭合。
- **与Score Centering对比**：SC能防止崩溃但train-rollout discrepancy仍高达1.8–2.1，TRACE维持接近0；平均分数75.3 vs. 74.1（↑1.2）。
- **与Post-hoc PTQ对比**：TRACE在联合FP4 rollout下完成RL训练，平均75.3；而先用BF16完成RL再用vanilla NVFP4/4over6/H-Scale做PTQ的平均分别为70.4/71.0/71.4，TRACE提升3.9分。

## 相关工作脉络
- **QUADS (Zhuge et al., 2026)**：结合asymmetric QAT与rollout侧activation补偿，但补偿策略仍基于各路径单独的最优，未对齐两条路径的最终FP4码字。
- **QaRL (Gu et al., 2026)**：通过matched low-precision kernels对齐训练与rollout，属kernel级对齐；TRACE在算法层面直接约束取整结果，正交且更彻底。
- **Rollout-ResQ (Mak et al., 2026)**：用sparse residuals修正FP4 rollout误差，属于事后校正；TRACE在训练时即引导取整，避免误差产生。
- **Score Centering (Marek & Ryabinin, 2026)**：在优化层校正training-inference mismatch引起的drift，不触碰底层量化计算；TRACE直接减小量化导致的log-prob差异。
- **QuRL (Li et al., 2026)、Jet-RL (Xi et al., 2026)**：主要针对FP8，通过adaptive clipping或统一精度流降低不匹配；TRACE聚焦于更激进的FP4并引入compact caching。
- **定位差异**：TRACE是唯一将“rollout侧FP4码字作为训练侧取整参照”的方法，从根本上消除了因微小BF16差异跨越FP4舍入边界而产生的额外discrepancy。

## 局限性与未来方向
- **对策略过时（policy staleness）的依赖**：TRACE仅减小不一致取整带来的额外discrepancy，无法消除由异步RL引起的底层激活差异。
- **仅覆盖部分层**：当前缓存方案仅使用后半层信息，在更激进压缩（如仅后5层）时性能下降至73.4，需进一步研究层选择策略。
- **仅验证了Qwen系列MoE模型**：虽规模跨度大（35B→2.4T），但未涵盖非MoE架构或不同量化格式（如FP8混合）的泛化性。
- **未来方向**：可扩展至更激进的byte-level量化、探索自适应层选择与bit分配、与更高效的分布式communication原语结合、以及验证在更长horizon与更大batch下的稳定性。

## 研究启发与可借鉴点
- **优化目标对齐**：将“两条路径的量化结果一致性”直接作为优化目标，而非分别最小化各自的量化误差，这一思路可迁移至其他低精度训练-推理不对齐的场景。
- **信息压缩设计**：通过结构化观察（>99%仅差1个相邻码字）大幅削减通信量，展示了在精度-开销权衡中进行理论驱动的压缩设计的方法论。
- **适应而非修复**：实验显示策略可在训练过程中“渐进适应”FP4 rollout，说明将低精度执行纳入训练本身是可行的，后续研究可探索更多类型的适应机制。
- **实验设计**：将训练动态（log-prob diff、reward、entropy、response length）与bench score同步追踪，能清晰区分“训练稳定”与“性能恢复”两个阶段，值得借鉴。
- **与PTQ pipeline的对比**：证明inline low-precision training可优于先BF16训练再PTQ的方案，为“何时引入低精度”提供了实证依据。

## 关键术语表
**TRACE**：Train-Rollout Quantization Alignment via Compact GuidancE，本文提出的FP4量化框架，通过rollout侧信息引导训练侧取整。
**Rollout-Guided QAT**：利用rollout阶段记录的FP4量化结果来指导训练阶段量化感知训练中的舍入决策。
**Train-Rollout Discrepancy**：同一轨迹在训练路径与rollout路径上经FP4量化后产生的激活差异，通常用Frobenius范数度量。
**NVFP4 (W/A + KV)**：Non-Overroundable FP4格式，权重/激活与KV cache均使用NVFP4量化；与MXFP4按block共享scale不同。
**Mantissa-Only Caching**：仅缓存FP4值的尾数位（如1 bit）及scale，舍弃指数部分以大幅压缩存储与通信开销。
**MoE (Mixture-of-Experts)**：混合专家模型，通过router动态选择部分expert前向，对数值差异敏感。
**Score Centering (SC)**：通过校正log-probability分布中心来缓解training-inference mismatch的RL稳定化方法。
**Post-hoc PTQ**：在BF16训练完成后对策略进行离线FP4量化的流程，与inline QAT形成对比。

## 可复现要素
- **数据集/基准**：LiveCodeBench v6、AIME 2024/2025、HMMT 2025、DeepSWE v1.1、Terminal-Bench 2.1、GDPval（均为公开benchmark）。
- **代码/权重**：论文未明确声明开源；基于VeRL、Megatron、SGLang框架构建。
- **关键超参**：GRPO为优化器；R3 replay rollout routing；token-level importance ratio clipping上界=5；max_version_diff=3；仅量化routed experts与KV cache，其余模块保持BF16；各模型RL step采样数分别为256×16、128×16、64×16、64×8；最大response长度32K（推理）或256K（编程/长程）；训练步数400或50步。
```
