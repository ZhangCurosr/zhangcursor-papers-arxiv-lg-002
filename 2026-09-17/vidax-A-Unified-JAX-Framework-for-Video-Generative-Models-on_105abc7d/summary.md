---
title: "vidax-A-Unified-JAX-Framework-for-Video-Generative-Models-on"
source: https://arxiv.org/pdf/2609.18077v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:23:49"
field: "视频生成模型推理系统"
keywords: ["video generation", "JAX", "TPU", "tensor parallelism", "sequence parallelism", "weight offloading", "flash attention", "inference engine"]
innovations: ["3轴可组合张量+序列并行网格统一于单一JAX sharding mesh", "host-resident零拷贝权重staging配合per-layer卸载作为推理兜底", "跨框架移植正确性诊断与数值边界案例的系统性文档化"]
benchmarks: ["TPU v4-8 inference latency", "peak HBM utilization", "compile time", "bit-exact activation parity"]
---

# 论文速读：vidax-A-Unified-JAX-Framework-for-Video-Generative-Models-on

## 一句话总结
vidax 是一个开源的 JAX/Flax 视频生成推理引擎与零拷贝 PyTorch-to-JAX 权重转换器，通过统一张量并行、序列并行与逐层权重卸载，在 TPU v4-8 上以零 PyTorch 依赖的方式运行主流视频生成模型（Wan、Cosmos、LTX、HunyuanVideo、CogVideoX）。

## 研究问题与动机
1. **框架断层**：开源视频生成模型几乎全部仅提供 PyTorch/CUDA 参考实现，Cloud TPU 虽具备大内存池与低成本优势，却缺乏生产级推理路径。
2. **时空序列膨胀**：视频模型对多帧 patch 执行全自注意力，序列长度膨胀至数万 token，导致二次复杂度注意力与激活显存远超参数本身，原生分辨率推理必须依赖多维并行与内存卸载。
3. **JAX 生态缺失**：当前 JAX 缺少统一推理框架来组合视频序列所需的 intra-layer 张量并行与 inter-layer 序列并行，且跨框架移植面临 checkpoint 加载内存尖峰、编译边界安全、自定义核与 GSPMD 分区冲突等系统级挑战。
4. **精度与数值边界易被忽视**：跨框架移植中暴露的 mixed-precision 累积误差、tokenizer padding_side 分歧、条件向量缩放分母错误等 bug 在纯仿真测试中难以发现。

## 核心贡献（创新点）
1. **模块化架构全覆盖**：Flax 库统一支持 DiT、omnimodal MoT、3D VAE、文本编码器与原生采样器；与已有 PyTorch 移植工作的区别在于 host-resident 零拷贝 staging 避免单设备全量分配导致的 OOM。
2. **3 轴可组合并行网格**：在单一 JAX sharding mesh 中统一 Megatron 1D 张量并行与 DeepSpeed-Ulysses 序列并行，手动定义 shard_map 内局部形状与 psum 规约以避免冗余 bias 累积，不同于仅做单维并行的现有实现。
3. **Per-layer 权重卸载作为可选兜底**：借鉴 ZeRO-Offload，参数驻留主机并通过固定 HBM 缓冲区动态流式传输，以 donate_argnums 原地复用单一 JIT 编译 block，明确定位为"参考分辨率下的内存兜底而非默认加速手段"。
4. **系统正确性与性能基准开源**：提供涵盖编译时间、每步延迟、峰值 HBM 的完整基准，并系统性文档化跨框架移植中暴露的数值 bug（精度、并行、架构、内存四大类），为后续 JAX/TPU 视频生成研究建立基线。

## 方法详解
1. **Host-resident 零拷贝权重翻译**：使用显式 `setup()` hooks 替代 `@nn.compact`，将 checkpoint 先作为 NumPy 数组暂存于主机，再单次 `device_put` 写入设备，避免直接在设备分配导致的单卡峰值 OOM。
2. **Spatiotemporal 张量与序列分片**：`vidax.core.sharding` 构建 3 轴 mesh（数据 / Megatron TP / Ulysses SP）；对 per-token AdaLN 等激活主导场景，序列并行在 transformer block 间划分 temporal tokens，通过 `all_to_all` 转置支持 head-sharded 注意力；TP+SP 组合时需手动编写 `jax.lax.psum` 以避免 bias 重复累加。
3. **3D RoPE 与 Flash Attention**：各模型家族保留其原始 3D RoPE（时间/高度/宽度不可互换）公式；未 mask 注意力分发至 Pallas TPU flash-attention 核；自定义核绕过 GSPMD 分区，因此用显式 `shard_map` 包装多设备调用；对无原生 TPU 核的模型（如 LTX-2.5 neighborhood-attention VAE decoder）回退到 `lax.scan` + `vmap` 窗口化注意力原语。
4. **Per-layer 权重卸载**：参考 ZeRO-Offload，参数留在主机，按 chunk 流式传入固定 HBM buffer；单一 `jit` 编译 block 函数通过 `donate_argnums` 原地更新；迁移延迟无法完全重叠计算，故仅作为高分辨率兜底。
5. **JIT 编译与内存 Ergonomics 三原则**：
   - 时空序列维度严格 static 以复用编译图；
   - 外层迭代（多步采样、层卸载扫描、分块 VAE 解码）保持为宿主 Python 循环，避免将整个多步采样 trace 进单一 HLO 图导致中间激活共存；
   - fp32→bf16 精度降级在主机端完成，避免加速器上双精度临时数组的内存开销。
6. **调度器统一接口**：从零实现三类采样器——Euler Rectified-Flow（含 LTX-2.5 所需的 ancestral SDE 变体）、Multistep UniPC（flow-matching 高阶预测-校正）、DDIM/DPM-Solver（v-prediction），在 JAX 层统一接口支持零开销切换。

## 实验与结果
1. **实验设置**：在单 TPU v4-8 slice（4 芯片，jax==0.11.0）上对 5 个模型家族的所有公开 checkpoint 尺寸进行评估；每项指标为 5 次独立端到端运行的均值，batch=1，开启 classifier-free guidance（内部 batch 2×）；编译缓存每次运行前清空。
2. **代表性性能（Table 2）**：
   - Cosmos3 Nano (16B)：1280×704 / 93f，TP=4，编译 64.2s，每步 7.1s，峰值 HBM 29.5 GB。
   - Wan2.2 A14B (MoE)：832×480 / 81f，TP=2/SP=2，chunk=10 卸载，编译 65.8s，每步 43.2s，峰值 HBM 28.4 GB。
   - Wan2.1 14B：1280×720 / 81f，TP=4/SP=1，chunk=20 卸载，编译 108.2s，每步 123.0s，峰值 HBM 23.0 GB。
   - LTX-2.5 22B (dev)：1216×704 / 121f，TP=4，chunk=8 卸载，编译 87.7s，每步 7.3s，峰值 HBM 16.7 GB。
   - HunyuanVideo 13B：1280×720 / 129f，TP=4，chunk=20/40 卸载，编译 506.7s，每步 299.6s，峰值 HBM 18.4 GB。
3. **内存优化分析**：峰值 HBM 被控制在 TPU v4 可用预算 (~30.75 GB/chip) 附近；Wan2.1 14B 在 720p 下 chunk 大小从 1 增至 40 时，每步延迟从 141.7s 单调降至 111.3s，峰值 HBM 从 15.2 GB 升至 26.1 GB，证明卸载是内存适配机制而非吞吐无损优化。
4. **定性结果**：每个家族两两对比（稠密 DiT vs MoE、标准 DiT vs MoT、代际演进）均在单一 JAX 执行路径下产出生成视频，时间连贯且 prompt 忠实，端到端正确性经 bit-exact/高相关块级校验与 prompt-faithful 生成双重验证。
5. **最强结果**：LTX-Video 13B (dev) 在 1216×704/121f 下每步仅 5.2s、峰值 HBM 15.3 GB，得益于无卸载且 TP=4 充分复用；Cosmos3 Nano 16B 在 7.1s/步下逼近 HBM 上限，展现大模型高效吞吐。

## 相关工作脉络
1. **Megatron-LM (Shoeybi et al., 2019)**：提出列/行并行线性投影的 1D 张量并行范式；vidax 在其基础上组合序列并行，并在 `shard_map` 内手动处理 bias 累加，区别于 Megatron 的训练导向设计。
2. **DeepSpeed-Ulysses (Jacobs et al., 2023)**：实现跨 transformer 块的序列并行 with all-to-all；vidax 将其与张量并行统一于单一 3 轴 mesh，并解决两者的交互边界问题。
3. **FlashAttention (Dao et al., 2022)**：IO-aware 精确注意力避免 O(S²) 显存；vidax 将其包装于 Pallas 核并通过 `shard_map` 实现多设备分发，处理自定义核绕过 GSPMD 的抽象边界。
4. **ZeRO-Offload (Ren et al., 2021)**：训练时的逐层权重卸载范式；vidax 借鉴其思路但将其定位为推理兜底，并以 `donate_argnums` 原地复用固定 HBM buffer。
5. **Wan / Cosmos / LTX / HunyuanVideo / CogVideoX 系列**：各自的 PyTorch 参考实现；vidax 通过 1:1 键映射与激活数值对比验证移植正确性，并文档化官方 checkpoint 中隐藏的精度/结构边界案例。

## 局限性与未来方向
1. **仅覆盖推理**：当前框架不支持训练或微调（LoRA/全参数），限制了在 TPU 上的模型定制能力。
2. **硬件验证局限**：基准仅在 TPU v4-8 上完成，尚未在 TPU v5e/v6e 等新架构上验证可移植性。
3. **卸载开销显著**：需要 per-layer offloading 的配置（如 Wan2.1 14B、HunyuanVideo 13B）每步延迟达百秒级，host-to-device 传输无法完全重叠计算。
4. **自定义核维护负担**：对无原生 TPU 核的模型（如 LTX-2.5 neighborhood attention VAE）需回退到 scan/vmap 实现，编译时间与调试成本较高。
5. **未来方向**：扩展至新 TPU 架构、支持训练/微调工作负载、开发定制 Pallas/Mosaic 核以降低延迟。

## 研究启发与可借鉴点
1. **Host-resident staging 模式**：对超大规模 checkpoint 加载，先在主机以 NumPy 暂存再单次 `device_put` 可避免单设备峰值 OOM，适用于任意 JAX 大模型部署。
2. **TP+SP 组合的显式 psum 处理**：当张量并行与序列并行共存时，必须在 `shard_map` 内手动定义归约以避免 bias/参数重复累加，这一模式可推广至其他多维并行场景。
3. **Per-layer offloading 作为兜底而非默认**：将卸载设计为可选路径并在 chunk 大小上 Sweep，可清晰权衡吞吐与内存，避免过度优化转移开销。
4. **JIT 边界的三层原则**：static 序列维度、外层 Python 循环、主机端精度降级，是防止 XLA 图爆炸与 HLO 暂存膨胀的通用实践。
5. **跨框架正确性诊断方法**："Low-Noise Real-Photo Probes"（编码-加噪-单步去噪-解码）可快速定位噪声条件 vs 权重故障；"Explicit Precision Auditing" 强调逐张量审计而非全局 cast，对 mixed-precision 移植具有普适价值。

## 关键术语表
**vidax**：基于 JAX/Flax 的开源视频生成推理引擎与 PyTorch-to-JAX 零拷贝权重转换器。
**Megatron-style 1D tensor parallelism**：沿矩阵列/行切分线性投影，配合 GSPMD 自动插入 all-reduce 的张量并行范式。
**DeepSpeed-Ulysses sequence parallelism**：将时序 token 跨 transformer block 分片，通过 all-to-all 转置实现注意力维度的并行。
**Pallas**：JAX 的低级自定义核编程接口，用于在 TPU 上实现 IO-aware flash-attention 等算子。
**3D RoPE**：分别作用于时间、高度、宽度三个维度的旋转位置编码，各方向公式不可互换。
**Per-token AdaLN**：对每个 token 独立计算自适应层归一化调制参数，显著提升激活内存压力，需序列并行支持。
**ZeRO-Offload**：将模型参数驻留主机、逐层流式传输至设备的训练优化技术，vidax 将其适配为推理兜底。
**Classifier-free guidance (CFG)**：同时运行条件与无条件前向以增强生成 fidelity 的标准采样技巧，vidax 内部 batch 2× 实现。
**Rectified Flow / UniPC / DPM-Solver**：三类扩散/流匹配采样器，分别对应一阶流积分、高阶预测-校正与 v-prediction 离散调度。
**HBM (High Bandwidth Memory)**：TPU 芯片级高带宽显存，vidax 通过分片与卸载将其峰值控制在 ~30.75 GB/chip。

## 可复现要素
- **代码**：开源于 https://github.com/FlyingGiraffe/vidax。
- **权重**：使用各模型官方公开 checkpoint，通过 1:1 键映射翻译，数值验证见 Appendix A。
- **数据集**：未引入新数据集，使用各模型默认参考分辨率/帧数/步数及标准化 prompt。
- **硬件**：TPU v4-8 slice（4 chips），jax==0.11.0。
- **关键超参**：TP/SP 配置见 Table 2/5；offload chunk 大小按模型调整（1/8/10/20/40）；精度策略（Wan2.1 fp32 DiT、LTX-2.5 fp32 AdaLN table 保留）见 Appendix A；scheduler 选用遵循各模型参考默认。
- **复现限制**：编译时间受 XLA 缓存影响，需每次清空；LTX-2.5 部分 variant 标注为 dev 版；部分 I2V 动态分辨率依赖输入图像 aspect ratio。
