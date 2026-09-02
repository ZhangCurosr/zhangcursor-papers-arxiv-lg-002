---
title: "mzCache-On-Device-LLM-Memory-Management-under-Multitasking"
source: https://arxiv.org/pdf/2609.01338v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:24:15"
field: "移动端大模型推理系统"
keywords: ["On-Device LLM Inference", "Memory Management", "Mobile Computing", "KV Cache Compression", "Multitasking", "Time-to-First-Token"]
innovations: ["面向恢复的弹性内存驱逐机制，支持任意驱逐程度的快速恢复", "KV cache 混合 swap 策略，动态平衡内存解压路径与闪存读取路径", "向后驱逐-向前恢复策略，实现 CPU 恢复与 GPU 推理的并行重叠"]
benchmarks: ["Galaxy S25+", "OnePlus 12", "Qwen3-0.6B", "EXAONE-4.0-1.2B"]
---

# 论文速读：mzCache-On-Device-LLM-Memory-Management-under-Multitasking

## 一句话总结
mzCache 提出了一套面向多任务环境的设备端 LLM 推理内存管理系统，通过细粒度内存分区、混合 swap 和向后驱逐-向前恢复策略，实现任意驱逐程度下的高速恢复，使 TTFT 相比部分卸载方案降低 2.1–5.5×。

## 研究问题与动机
- **移动端 LLM 推理的多任务内存压力**：移动设备 RAM 有限（通常 12GB），LLM 权重（2.5–4.9 GB）加 KV cache（可达 4 GB）占用 5–7 GB，在多任务切换时极易被 OS 驱逐。
- **OS 通用内存管理对 LLM 不友好**：zRAM 通用压缩算法（lz4/zstd）对 KV cache 压缩率极低（0.2–9.3%），无法有效缓解内存压力，最终触发 Low-Memory Killer 终止进程，导致冷启动开销巨大。
- **现有 LLM 系统缺乏弹性内存管理**：多数移动端 LLM 服务依赖 OS 页面机制，无细粒度部分驱逐/恢复能力，仅支持全量保留或全量加载，无法适应不可预测的内存压力。
- **页锁定内存并非良策**：Android AICore 的 Gemini Nano 通过内存锁定避免驱逐，但会导致空闲期持续占用 RAM，且无法解决 KV cache 的驱逐问题。

## 核心贡献（创新点）
- **面向恢复的弹性驱逐设计**：将 LLM 内存划分为细粒度共享缓冲区，支持按需部分驱逐并在任意驱逐状态下保持快速恢复能力，与已有系统仅支持全量保留/全量卸载形成本质区别。
- **KV cache 混合 swap 策略**：根据离线分析的去压缩吞吐量与存储读取吞吐量，动态平衡内存内解压缩路径与闪存读取路径的数据分配，保持两条恢复路径负载均衡。
- **向后驱逐-向前恢复策略**：按层次逆序驱逐（后期层先出）、正向恢复（前期层先入），确保 GPU 可立即开始预填充计算而无需等待全部数据加载，与 LRU 等先进先出驱逐策略截然不同。
- **统一内存架构下的 CPU-GPU 并发恢复与推理**：利用移动 SoC 的统一内存（Shared Virtual Memory），实现 CPU 侧数据恢复与 GPU 侧预填充计算并行重叠，消除处理器间冗余数据拷贝。
- **移动端优化自定义注意力核**：基于 OpenCL 实现支持非连续 KV chunk 的在线 softmax 注意力核，直接流式读取共享缓冲区数据，适配移动端 GPU 的负载/存储带宽受限特性。

## 方法详解
- **细粒度内存分区与共享缓冲区**：模型权重按层划分，KV cache 按固定大小 chunk（256 tokens）划分，所有单元分配为 CPU 和 GPU 可见的 OpenCL SVM 共享缓冲区，支持独立寻址与并发访问。
- **双路径恢复架构**：一条路径为闪存直接 I/O 读取（存储路径），另一条为内存内 zRAM 空间压缩 KV cache 的解压缩（内存路径）。
- **离线性能分析**：在一台目标设备上测量三个关键时间参数——$t_d^{KV}$（解压一个 KV chunk 的时间）、$t_r^{KV}$（从存储读取一个 KV chunk 的时间）、$t_r^W$（从存储读取一层权重的时间），用于指导驱逐分配决策。
- **混合 swap 驱逐阈值**：设定阈值 $T = N_W^{total} \cdot \lfloor t_r^W / t_d^{KV} \rfloor$，当剩余 KV chunk 数 $N_{KV} > T$ 时进入 KVonly 阶段（KV 同时写入两条路径）；当 $N_{KV} \leq T$ 时进入 KVandW 阶段（压缩 KV 至内存、权重丢入存储），确保两条路径完成时间对齐。
- **四阶段驱逐策略**：(1) KVonly：仅驱逐 KV chunk 到两条路径；(2) KVandW：同时驱逐 KV 和权重；(3) Wonly：仅丢弃权重层；(4) CompKV：将所有压缩 KV 移至存储以释放最后内存空间。
- **向后驱逐-向前恢复**：LRU 风格的 MRU 驱逐顺序保证早期层始终驻留内存；恢复时按层顺序加载，GPU 可立即开始预填充与 CPU 恢复并行重叠。

## 实验与结果
- **实验平台**：Galaxy S25+（Snapdragon 8 Elite, 12GB LPDDR5X, UFS 4.0）和 OnePlus 12（Snapdragon 8 Gen 3, 12GB LPDDR5X, UFS 4.0）；模型为 Qwen3-0.6B 和 EXAONE-4.0-1.2B；上下文长度 8k、16k、32k tokens。
- **基线对比**：OS Paging（依赖 Android 默认分页机制，zRAM 压缩 KV cache）和 Partial Ofload（llama.cpp 扩展支持层粒度部分卸载）。
- **主要结果**：
  - mzCache 在 0%–75% 剩余内存水平下，TTFT 相比 Partial Ofload 降低 **2.1–5.5×**，跨设备、跨模型、跨上下文长度一致有效。
  - 相比 OS Paging，在全量驱逐状态下 mzCache 在 Galaxy S25+ 上提升 **2.5–3.0×**，在 OnePlus 12 上提升 **9.2–25.9×**。
  - 单技术贡献：并行内存分配（1.37×）、零拷贝+混合 swap 恢复（2.15×）、移动端优化注意力核（2.33×）。
  - 全阶段重叠（Allocation-Reload-Prefill）比无重叠加速 **1.61×**；向后驱逐相比 LRU 式向前驱逐显著减少预填充 stall。
  - 压缩算法对比：8-bit 量化在 50% 剩余内存时 TTFT 更低，CacheGen 在 0% 剩余内存时更好；两者均保持与 FP16 相当的 F1 分数。
- **真实多任务验证**：在 Galaxy S25+ 上运行 45 分钟、10 轮真实 APP 切换测试，OS Paging 每轮均被 LMK 杀死，mzCache (25pp) 存活全部 10 轮并保持上下文。

## 相关工作脉络
- **IMPRESS / HCache**：服务器端 KV cache 分层存储与快速恢复系统，关注已存储 KV 上下文的复用加速；mzCache 面向移动端外部内存压力场景，驱逐量不可预测，需运行时自适应调整。
- **PagedAttention / FlexGen**：服务端 LLM 推理的显存管理与 offload 系统，系统完全控制数据放置时机；mzCache 需应对 OS 强制驱逐，无法预知驱逐量。
- **PowerInfer / Mobile LLM 推理系统**：聚焦推理阶段加速（NPU 协同、编译器优化）；mzCache 聚焦空闲期的内存管理，填补多任务场景下上下文保活的空白。
- **Android zRAM**：通用内存压缩 swap 机制，使用 lz4/zstd 等字节级压缩算法；mzCache 证明 KV cache 算术数据特性不支持此类通用压缩，需专门设计。
- **Ariadne / SWAM**：移动 OS 层混合 swap 技术，结合内存压缩与存储 swap；mzCache 在 LLM 推理系统内部实现类似思想，无需修改 OS 内核，且针对 KV cache 特性做了负载均衡优化。

## 局限性与未来方向
- **单一上下文支持**：当前仅支持单个 LLM 上下文，多 LLM 应用共存场景（各自维护权重和 KV cache）尚待扩展。
- **非标准模型架构支持不足**：MoE 模型和稀疏注意力模型的首 token 生成可能只访问部分权重/KV，需预测激活子集并调整恢复顺序。
- **热节流影响吞吐量估计**：离线分析的存储读取和解压缩吞吐量在持续负载下的热节流场景可能发生偏移，破坏双路径负载均衡。
- **驱逐量与检测机制耦合**：虽然内存压力检测和驱逐量设置解耦，但在实际部署中仍需谨慎权衡，过小驱逐量可能导致 LMK 杀死进程（实验中 15pp 配置在 10 轮中有 4 轮被终止）。

## 研究启发与可借鉴点
- **恢复导向的弹性内存管理范式**：将"任意驱逐程度均可快速恢复"作为设计目标，而非仅关注最佳情况下的性能，对移动端/边缘端 LLM 服务具有普适启发。
- **双路径负载均衡的离线分析方法**：通过离线性能分析获取各路径单位吞吐量，进而指导运行时数据分配比例，该方法可迁移至其他异构内存层级管理场景。
- **统一内存架构下的 CPU-GPU 流水线设计**：利用 SVM 共享缓冲区实现恢复与计算的并行重叠，避免显式数据传输开销，为移动 SoC 上的其他流水线优化提供参考。
- **移动端注意力核的定制化优化**：针对移动端 GPU 带宽受限特性，直接流式读取非连续数据并在线计算 softmax，而非先合并再计算，为移动端 GPU 编程提供实践案例。

## 关键术语表
- **KV cache**：LLM 推理中为每个输入 token 在各 transformer 层计算的 key-value 向量对，用于加速后续自回归生成，避免重复计算。
- **Time-to-First-Token (TTFT)**：从输入请求到生成第一个输出 token 的时间，是衡量 LLM 响应性的关键指标。
- **zRAM**：Android 中的内存压缩 swap 空间，将在物理 RAM 内部对换出页面进行压缩存储，以替代较慢的闪存 swap。
- **Low-Memory Killer (LMK)**：Android 系统进程，在内存压力过高时终止后台进程以释放内存。
- **Shared Virtual Memory (SVM)**：OpenCL 提供的内存机制，使 CPU 和 GPU 可共享同一物理内存地址空间，消除显式数据拷贝。
- **Hybrid Swap**：mzCache 中针对 KV cache 设计的双路径交换策略，部分数据存于内存压缩空间（快速解压），部分存于闪存（直接读取），通过负载均衡实现整体最优恢复。

## 可复现要素
- **数据集**：使用 TriviaQA 428 个问答对评估压缩精度，无专用训练数据集；LLM 模型权重（Qwen3-0.6B、EXAONE-4.0-1.2B）可从官方渠道获取。
- **代码开源**：论文未声明代码开源仓库；实现基于 llama.cpp 框架（开源）扩展，约 6K 行 C/C++。
- **关键超参**：KV chunk 大小固定为 256 tokens；压缩算法默认使用 8-bit 量化，可选 CacheGen。
- **设备信息**：Galaxy S25+（Snapdragon 8 Elite, 12GB LPDDR5X, UFS 4.0）和 OnePlus 12（Snapdragon 8 Gen 3, 12GB LPDDR5X, UFS 4.0）。
