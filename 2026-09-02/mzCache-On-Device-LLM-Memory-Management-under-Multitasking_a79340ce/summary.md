---
title: "mzCache-On-Device-LLM-Memory-Management-under-Multitasking"
source: https://arxiv.org/pdf/2609.01338v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 11:50:27"
field: "端侧大语言模型推理系统"
keywords: ["On-Device LLM Inference", "Memory Management", "Multitasking", "KV Cache Compression", "Mobile SoC", "Unified Memory", "Hybrid Swap", "TTFT Optimization"]
innovations: ["恢复导向型弹性内存管理框架，支持任意驱逐状态下的快速推理恢复", "基于统一内存架构的 CPU-GPU 零拷贝流水线重叠机制", "混合 swap 双路径动态负载均衡策略与向后驱逐/向前恢复调度"]
---

# 论文速读：mzCache-On-Device-LLM-Memory-Management-under-Multitasking

## 一句话总结
mzCache 是一款面向移动设备多任务场景的端侧 LLM 推理系统，通过细粒度共享内存分区与混合 swap 策略，在外部内存压力下弹性驱逐 LLM 权重和 KV cache，同时实现低至 2.1–5.5× 的 TTFT 加速，避免 OS 机制导致的进程被杀。

## 研究问题与动机
- 移动端 LLM 推理需常驻 5–7 GB RAM（权重 + KV cache），占现代手机 12 GB RAM 的近一半；多任务切换时 OS 会驱逐 LLM 内存，导致冷启动和 TTFT 从 0.8s 飙升至 16s。
- Android zRAM 通用压缩算法（lz4/zstd）对 KV cache 几乎无效（空间节省仅 0.2%–9.3%），无法缓解内存压力，最终触发 Low-Memory Killer（LMK）终止 LLM 进程。
- 现有移动端 LLM 服务仅关注推理运行时优化，内存管理完全依赖 OS 机制，缺乏对不可预测内存压力的主动应对能力。
- 服务器端 LLM 系统依赖独立 GPU 显存 + PCIe 高速通道做临时 offload，而移动端采用统一内存架构（SoC CPU/GPU 共享同一段物理内存），缺少快速中间存储层级，闪存读取成为恢复延迟主因（占 TTFT 59%–61%）。

## 核心贡献（创新点）
- **恢复导向型内存管理框架**：首次提出 Restoration-oriented Eviction 概念，在任意驱逐状态下均可快速恢复推理，与"全有或全无"的单体内存分配本质不同。
- **CPU-GPU 协同的统一内存细粒度管理**：将权重按层、KV cache 按固定 chunk（256 token）分区为共享缓冲区，GPU 可在 CPU 恢复的同时对已保留的前置层立即开始 prefill，消除处理器间冗余数据拷贝。
- **混合 swap 负载平衡策略**：为 KV cache 设计双路径恢复（内存内解压 vs 闪存读取），基于离线性能画像动态分配数据分布，确保两条路径同时饱和，优于固定分配方案。
- **后进先出驱逐 / 正向恢复调度**：采用 backward-out（从末层向前驱逐）+ forward-in（从首层向前恢复）策略，使 GPU prefill 与 CPU 恢复自然流水线重叠，相比 LRU 式驱逐可避免 GPU 空闲等待。
- **端到端原型与真实设备验证**：基于 llama.cpp 实现约 6k 行 C/C++，部署为 Android App，在 Galaxy S25+ 和 OnePlus 12 上实测，TTFT 降低 2.1–5.5×，并在真实多任务场景下存活全部 10 轮测试。

## 方法详解
- **内存分区与共享缓冲区**：权重以 Transformer 层为粒度划分；KV cache 每层划分为 256 token 的固定大小 chunk（满足 UFS 4.0 闪存 512 KB 满带宽阈值）。所有单元通过 OpenCL Shared Virtual Memory (SVM) 分配为 CPU/GPU 共同可见的共享缓冲区，避免显式跨处理器拷贝。
- **自定义注意力内核**：针对非连续共享缓冲区上的大 KV cache prefill，实现 OpenCL 注意力 kernel，采用 online softmax 直接流式读取 chunk 指针数组，避免数据合并；利用 Adreno GPU 双速率 FP16 硬件执行 8-wide FP16 向量运算。
- **混合 swap 四阶段驱逐**：
  - **KVonly**：当总 KV chunk 数 $N_{\mathrm{KV}}^{\mathrm{total}} > T$ 时，先将 KV chunk 按反比于 $t_d^{\mathrm{KV}}$ 与 $t_r^{\mathrm{KV}}$ 分配到内存压缩区和闪存，保持双路径负载均衡。
  - **KVandW**：当 $N_{\mathrm{KV}}$ 降至阈值 $T = N_{\mathrm{W}}^{\mathrm{total}} \cdot \lfloor t_r^{\mathrm{W}} / t_d^{\mathrm{KV}} \rfloor$ 时，开始同步压缩剩余 KV 并逐层丢弃权重，两者同步耗尽。
  - **Wonly**：当 $N_{\mathrm{KV}}^{\mathrm{total}} \le T$ 时直接从权重丢弃开始。
  - **CompKV**：所有数据均已驱逐后若仍有压力，将压缩 KV 移至闪存释放最后一段内存。
- **向后驱逐 / 向前恢复时序**：驱逐顺序为末层 → 首层（backward-out），保证首层始终驻留；恢复顺序为首层 → 末层（forward-in），与 prefill 执行顺序一致，形成自然 LIFO 访问模式；内存 swap 区以栈结构组织，按需增长、恢复后释放。
- **并行化恢复**：内存分配与数据加载跨多核 CPU 分发，allocation-reload-prefill 三阶段可全重叠（full overlap 带来 1.61× 额外加速）。

## 实验与结果
- **测试设备**：Galaxy S25+（Exynos 2400 / Adreno 830 / 12 GB LPDDR5X / UFS 4.0）与 OnePlus 12（Snapdragon 8 Gen 3 / Adreno 750 / 12 GB / UFS 4.0）。
- **模型**：Qwen3-0.6B（1.2 GB 权重）与 EXAONE-4.0-1.2B（2.6 GB 权重），上下文长度 8k/16k/32k token。
- **基线**：OS Paging（Android 默认 zRAM 压缩 + mmap 权重，CPU 推理）与 Partial Ofload（扩展 llama.cpp 支持层粒度部分驱逐，GPU 推理）。
- **主要结果**：
  - mzCache 相比 Partial Ofload 在 0%/25%/50%/75% 剩余内存四种状态下均实现 **2.1–5.5× TTFT 加速**，跨设备、模型、上下文长度一致有效。
  - 相比 OS Paging（zRAM），在完全驱逐状态下 mzCache 在 S25+ 上快 2.5–3.0×，在 OnePlus 12 上快 **9.2–25.9×**（zRAM 压缩差导致 OS Paging 实际未释放足够内存，仍频繁触发 LMK）。
  - 单技术贡献分解（Galaxy S25+, Qwen3-0.6B, 50% 剩余内存）：并行分配 1.37×、零拷贝+混合 swap 恢复 2.15×、移动端优化注意力内核 2.33×。
  - 重叠策略：Allocation-Reload 重叠 1.10×，Reload-Prefill 重叠 1.41×，全重叠 1.61×；backward-out 驱逐策略显著减少 prefill stall，优于 forward-out 和 random-out。
  - 能耗：mzCache 峰值功耗 19.2 W（高于 Partial Ofload 的 14.6 W），但因 TTFT 大幅缩短，**总能耗更低**。
  - 压缩算法对比：8-bit quantization 在 50% 剩余内存时 TTFT 更优（解压快），CacheGen 在 0% 时更优（压缩比高减少闪存读取）；两者在 TriviaQA 上 F1 与 FP16 基线相当，且具 idempotent 特性，多次压缩/解压不累积误差。
  - 真实多任务部署：45 分钟、10 轮社交/视频/游戏/相机/浏览交替使用，OS Paging 每轮均被 LMK 终止；mzCache (25 pp) 全程存活并恢复上下文，mzCache (15 pp) 四轮被杀——说明驱逐量调参对存活率至关重要。

## 相关工作脉络
- **Offload-based LLM 推理**（FlexGen、PowerInfer 等）：LLM 系统自主控制 GPU/CPU/磁盘间的数据迁移，拥有完整内存可用信息；mzCache 面对的是外部不可预测压力下的被动驱逐，需在线适应未知驱逐量。
- **移动端 LLM 推理系统**（MLC-LLM、mllm、Fast on-device LLM with NPUs 等）：聚焦运行时 heterogeneous co-execution 与编译器优化，未处理 idle 期间的内存压力管理；mzCache 填补该空白。
- **KV cache 高效管理**（PagedAttention、HCache、IMPRESS 等）：服务器端多用户场景依赖 prefix-tree 等复杂结构；mzCache 面向单用户移动端，用简单 chunk 指针数组即可，且支持非连续内存直接 attention 计算。
- **移动端混合 swap**（Ariadne、SWAM、ASAP 等）：通用压缩算法无法有效压缩 KV cache；mzCache 将 hybrid swap 思想专门适配 KV cache 结构特性，无需修改 OS 内核。
- **KV cache 压缩**（KVQuant、KIVI、CacheGen、Zhang et al. "KV Cache is 1 Bit Per Channel"）：mzCache 将其作为模块化组件集成，支持 8-bit quantization 与 CacheGen 两种算法按需替换。
- **Android AICore / Gemini Nano**：通过 mlock 锁定权重避免 LMK，但权重始终占用 RAM 且无 KV cache 保护；mzCache 允许权重和 KV 均参与弹性驱逐与快速恢复。

## 局限性与未来方向
- **单上下文支持**：当前仅支持单一 LLM 上下文；多 LLM App 共存时的权重与 KV cache 协调管理尚未实现。
- **热节流影响性能画像**：离线 profiling 获取的吞吐参数在持续负载下可能因 thermal throttling 偏移，破坏双路径负载均衡；需多频率组合预画像或运行时自适应迁移。
- **选择性激活模型不兼容**：设计假设 first-token 需访问全部权重与 KV cache，不适用于 MoE（mixture-of-experts）或 sparse-attention 模型，需预测并优先恢复活跃子集。
- **驱逐量依赖部署调参**：应用层部署（onTrimMemory 回调）需人工设定驱逐百分比（15 pp vs 25 pp 差异显著），系统层部署可通过 vmpressure/PSI 与低 oom_score_adj 缓解，但最佳策略因场景而异。
- **未支持 NPU 后端**：当前实现针对 GPU，NPU 协处理的恢复流水线有待探索。

## 研究启发与可借鉴点
- **恢复导向设计范式**：将"任意中间状态可快速恢复"作为第一性原理，而非追求最优静态布局，适用于所有外部压力不可预测的资源管理场景（如边缘推理、移动端 AI 服务）。
- **统一内存架构的 CPU-GPU 流水线重叠**：利用 SVM 共享缓冲区消除显式拷贝，使分配、I/O 恢复、计算三阶段自然重叠，思路可迁移至任何 SoC 级 heterogeneous computing 系统。
- **混合存储路径的动态负载均衡**：基于离线吞吐画像 $t_d^{\mathrm{KV}}$、$t_r^{\mathrm{KV}}$、$t_r^{\mathrm{W}}$ 构造阈值 $T$，以数学方式平衡压缩与闪存两条路径，方法可扩展至其他多级存储场景（如 NVMe + DRAM + 压缩缓存）。
- **向后驱逐 / 向前恢复的流水线调度原则**：针对具有严格顺序依赖的计算图（如 transformer 层串行执行），反向驱逐 + 正向恢复是最大化重叠的通用策略，可推广至 RNN、stateful 推理等场景。
- **模块化压缩接口设计**：将 KV cache 压缩抽象为可插拔组件（8-bit quantization / CacheGen），便于随算法演进持续升级，无需重构系统主体。

## 关键术语表
- **Time-to-First-Token (TTFT)**：从接收输入到生成第一个输出 token 的时间，是衡量 LLM 交互响应性的关键指标，受内存恢复延迟直接影响。
- **KV cache**：自注意力机制中每个 token 对应的 key 和 value 向量缓存，避免重复计算；随上下文累积可达数 GB，是移动端 LLM 内存压力的主要来源。
- **Unified Memory (SoC)**：移动 SoC 中 CPU 与 GPU/NPU 共享同一物理内存地址空间，消除处理器间显式数据拷贝，但也使 OS paging 行为更加复杂。
- **zRAM**：Android 内置的内存压缩 swap 空间，将被驱逐页面压缩后存于 RAM 内，以 CPU 周期换取比闪存更快的恢复速度，但对 KV cache 压缩率极低。
- **Low-Memory Killer (LMK)**：Android 在内存压力持续时终止后台进程的机制，被杀进程需冷启动重载权重并重建 KV cache，导致极高延迟。
- **Shared Virtual Memory (SVM)**：OpenCL 提供的 CPU/GPU 共享虚拟内存机制，两端通过同一物理页面访问，是实现零拷贝协同的核心基础设施。
- **Hybrid Swap**：mzCache 提出的双路径数据存放策略，KV cache 同时分布于内存压缩区和闪存，按性能画像动态平衡以最小化恢复时间。
- **Backward-out / Forward-in Policy**：驱逐从末层向首层进行（保留推理必需的早期层），恢复从首层向末层进行（匹配 prefill 执行顺序），实现自然流水线重叠。

## 可复现要素
- **数据集**：评估使用公开 LLM 模型 Qwen3-0.6B [43]、EXAONE-4.0-1.2B [32]；准确性评估使用 TriviaQA [26]（428 问答对）。论文未提供专用微调数据集。
- **代码开源**：论文声明基于开源 llama.cpp [18] 实现，约 6k 行 C/C++；但**未明确声明 mzCache 源码已开源**，需进一步确认。
- **权重**：使用模型官方权重，未提供自定义训练权重。
- **关键超参**：KV chunk 大小固定为 256 token；压缩算法默认 8-bit quantization（可选 CacheGen）；OpenCL kernel 使用 8-wide FP16 向量运算；阈值 $T$ 由离线性能画像计算得出。
- **测试设备**：Galaxy S25+（Exynos 2400）、OnePlus 12（Snapdragon 8 Gen 3），均配备 12 GB LPDDR5X + UFS 4.0。
- **环境**：Android 系统，通过 onTrimMemory 回调检测内存压力；性能测量通过 ADB 与 Perfetto [4] 采集。
