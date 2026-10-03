---
title: "LEAPQUANT-EFFICIENT-LINEAR-ATTENTION-WITH-ACCURATE-RECURRENT"
source: https://arxiv.org/pdf/2609.38166v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:51:49"
field: "大语言模型推理优化"
keywords: ["linear attention", "recurrent state quantization", "post-training quantization", "GDN", "KDA", "inference optimization"]
innovations: ["逐窗口量化：将量化频率从每token降至每窗口一次，抑制递推舍入误差累加", "补偿Token：用高分辨率秩一矩阵吸收状态outlier，使残差更易低比特量化", "残差平滑：对角尺度归一化进一步均衡量化精度，三项技术联合实现免训练8-bit无损量化"]
benchmarks: ["AIME 2026", "GPQA-Diamond", "MMLU-Pro", "LiveCodeBench v6", "GSM8K"]
---

# 论文速读：LEAPQUANT-EFFICIENT-LINEAR-ATTENTION-WITH-ACCURATE-RECURRENT

## 一句话总结
LeapsQuant 是一种**免训练的无校准线性注意力循环状态量化方法**，通过**逐窗口量化**和**补偿 Token（Compensator Tokens）**两项核心技术，在 INT8 精度下实现了与 FP32 近乎无损的推理准确率，同时将循环状态内存带宽降低 3.4×、端到端显存最高缩减 56%。

---

## 研究问题与动机
- **循环状态的 HBM 带宽瓶颈**：线性注意力模型（GDN、KDA）将长上下文压缩为固定大小的循环状态矩阵 $S \in \mathbb{R}^{d_k \times d_v}$，但每个解码步均需从 HBM 加载/写入完整状态，推理吞吐受限于带宽（Williams Roofline 法则）。
- **朴素低比特量化的精度崩溃**：逐 token 重新量化状态会导致舍入误差随生成长度**累加**；且状态中大量能量集中在少数行列的 **outlier**，迫使整体量化尺度变粗。
- **现有 KV 缓存/SSM 量化方案不适用**：KVQuant、QuaRot 等针对 KV Cache 设计；Quamba/MambaQuant 针对 Mamba 选择性 SSM，其状态更新形式与线性注意力不同，无法直接复用。
- **Prefix Caching 下显存占用显著**：每个缓存前缀需独立存储一份状态，显存开销随并发请求线性增长，量化可降低此负担。

---

## 核心贡献（创新点）
1. **逐窗口量化（Per-window Quantization）**：每次窗口内保持量化边界状态 $\hat{S}_0$ 固定，以高精度缓冲 $p$ 个 token 的更新，仅在窗口末端重建并量化一次，将量化误差引入频率降低 $p$ 倍，从根本上遏制误差累加。
2. **补偿 Token（Compensator Tokens）**：在量化前用少量 FP16 秩-一矩阵 $\tilde{K}\tilde{U}^\top$ 吸收状态中能量最大的 outlier 分量，使其不参与低比特量化，大幅收窄残差动态范围；这些秩-一形式与真实 token 更新一致，可直接嵌入同一 decode 路径。
3. **残差平滑（Residual Smoothing）**：在量化前以 key 行均值为基准进行尺度平衡（$C^{-1}R_0$），进一步减少量化尺度被少数通道主导的问题，且变换可逆、无需训练。
4. **全链路免训练、免校准**：三个组件均无需微调数据或梯度更新，仅需在窗口边界以 power iteration 拟合补偿 Token，部署零门槛。

---

## 方法详解

### 基础递推形式
任意线性注意力单 head 的状态更新统一为对角衰减 + 秩-一 delta-rule：
$$S_t = \mathrm{Diag}(\alpha_t) S_{t-1} + k_t\big(v_t - S_{t-1}^\top \beta_t\big)^\top, \quad o_t = S_t^\top q_t$$
修正向量 $u_i = v_i - S_{i-1}^\top \beta_i$，则单次更新为 $S_i = \mathrm{Diag}(\alpha_i) S_{i-1} + k_i u_i^\top$。

### 逐窗口量化（Section 3.2）
将窗口内 $p$ 个 token 的 $(\alpha_i, k_i, u_i)$ 以高精度缓冲，累计衰减 $\Gamma_{a:b} = \mathrm{Diag}(\alpha_b)\cdots\mathrm{Diag}(\alpha_a)$。窗口内任意时刻 $\ell$ 的状态可表示为：
$$S_\ell = \Gamma_{1:\ell}\hat{S}_0 + \sum_{j=1}^{\ell} \Gamma_{j+1:\ell}\, k_j u_j^\top$$
仅在窗口末尾 $\ell = p$ 时重建 $S_p$ 并执行一次量化 $\hat{S}_p = \mathtt{quant}_b(S_p)$，中间过程不再量化，误差引入频率降为原来的 $1/p$。

### 补偿 Token（Section 3.3）
窗口末重建 $S_p$ 后，通过最小化 Frobenius 范数拟合 $r$ 个秩-一分量：
$$(\tilde{K}, \tilde{U}) = \arg\min_{K,U}\|S_p - KU^\top\|_F, \quad R_p = S_p - \tilde{K}\tilde{U}^\top, \quad \hat{R}_p = \mathtt{quant}_b(R_p)$$
最终状态 $\tilde{S}_p = \mathtt{dequant}_b(\hat{R}_p) + \tilde{K}\tilde{U}^\top$。下一窗口开始时，decode kernel 先处理 $r$ 个补偿 Token 再处理真实 token 更新，全程无需额外 kernel。实测 $r \le 8$ 时重建开销可被内存读取完全掩盖，暴露延迟 < 7%。

### 残差平滑（Section 3.4）
对补偿 Token 减去后的残差 $R_0$，以 key 行均方根构造对角缩放矩阵 $C = \mathrm{Diag}(c)$，做 $C^{-1}R_0$ 后再量化，恢复时左乘 $C$：
$$\hat{R}_0^C = \mathtt{quant}_b(C^{-1}R_0), \qquad S_p = \Gamma_{1:p}\big[C\,\mathtt{dequant}_b(\hat{R}_0^C) + \tilde{K}\tilde{U}^\top\big] + \sum_{j=1}^p \Gamma_{j+1:p}\, k_j u_j^\top$$
缩放系数每窗口边界重新计算，窗内固定，不增加窗内开销。

---

## 实验与结果
- **模型**：Qwen3.5-9B、Qwen3.5-35B-A3B（GDN）；Kimi-Linear-48B-A3B-Instruct（KDA）；GLM-5.3-Flash、Qwen3.8-Flash（仅做效率测试）。
- **数据集**：AIME 2026、GPQA-Diamond、MMLU-Pro、LiveCodeBench v6、GSM8K；生成上限 65K–82K tokens，取 3 次随机种子均值。
- **硬件**：NVIDIA B200、RTX PRO 6000、RTX 5090；基于 vLLM 实现，kernel 用 TileLang 编写。
- **默认超参**：窗口 $p=16$，补偿 Token 数 $r=4$（4-bit 时 $r=8$），INT8 残差 + FP16 补偿 Token + FP32 平滑系数。

**精度结果**（Table 1）：
- **8-bit**：LeapQuant 在 12 组模型-任务上均与 FP32 持平（平均 75.5% vs 75.5%）；同精度 BF16 per-step 在 Qwen3.5-9B AIME 上仅 72.1%，FP8 per-tensor 低至 14.6%。
- **6-bit**：LeapQuant 平均 72.4%，NVFP6 平均仅 29.6%，TurboQuant 平均 31.9%，Qwen 系列完全崩溃。
- **4-bit**：LeapQuant 平均 60.4%，在 Kimi 模型上距 FP32 仅差 ≤1.1%；MXFP4 平均仅 9.7%，全链路塌缩。

**效率结果**：
- **Kernel 加速**：B200 上 GDN 2.68×、KDA 2.41×；RTX PRO 6000/5090 最高达 **3.70×**（带宽受限更显著）。
- **Decode 吞吐**：B200 batch=512 加速 1.22–1.37×；RTX PRO 6000 batch=256 加速 1.22–1.57×。
- **端到端吞吐**：B200 提升 1.23–1.60×，RTX PRO 6000 提升 1.31–1.65×。
- **显存**：状态字节从 4 B/element 降至 **1.19 B/element**（3.4× 压缩）；prefix caching 场景下端到端显存最高减少 **56%**（Kimi-Linear-48B-A3B），允许单卡 B200 并发请求提升 **1.4×**。

**Ablation（Table 2）**：各组件逐一叠加验证——per-step INT8 在 AIME 上从 87.9% 跌至 7.1%；加入逐窗口量化恢复至 82.4%；加入补偿 Token 至 86.6%；加入平滑后**完整恢复至 87.9%**，kernel 加速 2.52×，精度与 FP32 完全一致。

---

## 相关工作脉络
1. **GDN / KDA 等线性注意力架构**（Yang et al. 2025; Team et al. 2025）：提出 hybrid 模型并用循环状态压缩长上下文，但未涉及状态量化问题；本文在相同架构基础上提出首个免训练状态量化方案。
2. **KV Cache 量化（KVQuant、QuaRot、TurboQuant）**（Hooper et al. 2024; Ashkboos et al. 2024; Zandieh et al. 2026）：隔离 outlier 为稀疏/低秩高分辨率分量的思路被本文迁移至循环状态，但 KV Cache 的静态结构与循环状态的递推更新机制有本质差异，直接应用精度严重劣化。
3. **Mamba/SSM 量化（Quamba、MambaQuant、Quamba2、Q-Mamba）**（Chiang et al. 2025; Yue et al. 2025）：针对选择性 SSM 的特殊更新规则设计，不能直接推广到 GDN/KDA 等线性注意力递推形式。
4. **ReplaySSM**（Liou & Dao 2026）：同样基于"保存边界状态 + 回放更新"结构，但其目的是支持投机解码的回滚，而非量化精度优化；本文借用该结构并叠加量化技术。
5. **SmoothQuant**（Xiao et al. 2023）：激活-权重间迁移 outlier 的思路启发了本文的残差平滑操作，但应用场景与目标张量完全不同。

---

## 局限性与未来方向
- **仅覆盖对角衰减 + 单秩 delta-rule 形式**（含 GDN、KDA 等），未覆盖 erase/write 分离的架构（如 RWKV7，Appendix A 讨论需拆成两步 sub-step）或多 delta 步模型（DeltaProduct）。
- **窗口长度 $p$ 和补偿 Token 数 $r$ 需调参**：$p$ 过大会增大逐步 record buffer 读取和 shared memory 压力，$r$ 过大会暴露幂迭代开销（$r=16$ 时反而比 FP32 还慢），本文经验值为 $p=16, r=4$。
- **极端低比特（4-bit）仍有精度损失**（平均 60.4%，较 FP32 差距 15 pp），距离生产级无损仍有空间。
- **未评估多 GPU 并行/分布式 serving 场景下的通信开销**。
- **未来可探索自适应窗口调度、与 Speculative Decoding 结合、以及扩展到更广泛的线性注意力变体。**

---

## 研究启发与可借鉴点
1. **"窗口化低频量化"范式可迁移**：凡具有递推状态积累误差特征的模型（State Space Models、RNN-like 层、在线学习系统），均可借鉴"边界状态 + 高精度缓冲更新"的思路降低量化频率，从根本上抑制误差累加。
2. **Compensator Tokens 的 rank-1 outlier 吸收策略**：将状态中最大奇异分量以高分辨率显式保留，其余残差平滑后量化——这一"低秩吸附 + 平滑残差"的量化流程可用于其他张量（如 SSM 转移矩阵、递归权重），值得在其他模型架构中验证。
3. **免训练、免校准的端到端部署优势**：整个方法仅需在窗口边界做 power iteration，对推理服务几乎无侵入；可考虑将其集成到现有 serving 框架（vLLM/SGLang）作为即插即用后端，对团队现有的 hybrid LLM 部署 pipeline 有直接价值。
4. **残差平滑的对角尺度变换**：以行均值为基准的对角缩放是简洁有效的量化预处理技巧，可推广至任何存在显著行尺度不均的矩阵（如大型 attention 状态、SSM 参数矩阵）。
5. **消融设计展示完整的精度-效率权衡曲线**：论文通过逐步叠加三个组件（逐窗口→补偿 Token→平滑）展示精度恢复过程，并提供清晰的 ablation 表（Table 2、Table 3），是方法类论文写作和实验设计的优秀范本。

---

## 关键术语表
**Linear Attention（线性注意力）**：将 softmax attention 替换为固定大小状态 $S_t$ 的递推形式，每步以 $O(d)$ 计算复杂度处理整个上下文，广泛用于长序列 LLM。
**Gated DeltaNet（GDN）**：Yang et al. 2025 提出的线性注意力变体，引入数据依赖对角衰减 $\alpha_t$ 和 delta-rule 更新，是 Qwen3.5 系列的核心 attention 层。
**Kimi Delta Attention（KDA）**：Kimi 团队提出的 hybrid 注意力架构，与 GDN 类似但衰减和对齐方式不同，用于 Kimi-Linear 系列模型。
**Compensator Token（补偿 Token）**：用于捕获循环状态中最大 outlier 分量的虚拟秩-一更新项，以高分辨率保留并嵌入 decode 路径，不参与输出计算。
**Per-window Quantization（逐窗口量化）**：在每个 token 窗口内固定边界状态并高精度缓冲更新，仅在窗口末尾重建并量化一次，降低量化误差的引入频率。
**Residual Smoothing（残差平滑）**：以 key 行均值为基准进行对角尺度归一化，再对残差进行量化，使量化尺度不再被少数通道主导。
**Decode-time HBM Bandwidth Bottleneck（解码期 HBM 带宽瓶颈）**：线性注意力解码过程中状态读写字节量远大于算术操作量的现象，导致吞吐受限于内存带宽而非算力。
**Prefix Caching（前缀缓存）**：在多请求 serving 中缓存已处理 token 的前缀状态以避免重复计算，但每份缓存需独立存储循环状态，带来额外显存开销。

---

## 可复现要素
- **代码**：在 vLLM 中实现，linear attention decode kernel 用 TileLang 编写；论文提供了主要算法描述和公式，但**未提供公开 GitHub 链接**。
- **模型权重**：使用开源模型 Qwen3.5-9B、Qwen3.5-35B-A3B、Kimi-Linear-48B-A3B-Instruct、GLM-5.3-Flash、Qwen3.8-Flash，均为官方开源权重。
- **数据集**：AIME 2026、GPQA-Diamond、MMLU-Pro、LiveCodeBench v6、GSM8K，均为公开基准。
- **关键超参**：窗口长度 $p = 16$，补偿 Token 数 $r = 4$（4-bit 时 $r=8$），INT8 残差 + FP16 补偿 Token + FP32 平滑系数；power iteration 拟合补偿 Token。
- **硬件环境**：NVIDIA B200、RTX PRO 6000、RTX 5090。
