---
title: "LLMS-ARE-GENERAL-ASYNCHRONOUS-AGENTS"
source: https://arxiv.org/pdf/2609.35427v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 19:51:36"
---

# 论文速读：LLMS ARE GENERAL ASYNCHRONOUS AGENTS

## 一句话总结
本文提出 **AsyncLLM** 框架，无需任务特定微调即可让现代 LLM（以 Qwen 3.x 为主）成为通用的异步智能体；通过共享内存块（CacheBlocks）与注意力视图（cache views）机制，支持多协程并发推理与实时输入响应，并在流式视频理解、电子游戏交互与系统监控三个异构场景中验证了其零样本泛化能力。

## 研究问题与动机
- **核心问题**：现有 LLM Agent 严格遵循序列交互循环（读→想→回复/调用工具），无法原生支持语音助手、具身控制、实时流媒体等需要“边听边想、边处理边响应”的真实并发场景。
- **现有方法不足**：当前异步/并发方案高度碎片化，语音助手、流式视频模型、VLA 具身智能、异步函数调用等各自独立设计专用推理引擎，多数需针对特定任务微调或引入额外模块，缺乏统一框架。
- **人类认知启发**：人类天然具备异步处理能力，假设经大规模语料训练的 LLM 已内化类似推理模式，只需合适工程封装即可激发其通用异步潜力。
- **目标**：构建一个与任务无关的训练免费框架，使开发者或智能体自身能以 `async/await` 范式自由定义并发推理协程，并由底层引擎自动完成高效 GPU 批调度。

## 核心贡献（创新点）
1. **提出 AsyncLLM 通用异步编程框架**：基于 Python asyncio 抽象出 `CacheBlock` 与 `cache_view`，允许协程间零拷贝共享模型内部状态，区别于以往依赖 token 通信或任务特定架构的并发方案。
2. **设计支持混合架构的并行推理算法**：将全注意力 KV 缓存与 GDN 递归状态统一抽象为可组合块；通过查询旋转技巧处理 RoPE/MRoPE，通过分块仿射变换处理线性注意力，本质区别在于彻底消除“视图变化时重编码历史 token”的开销。
3. **零样本异步泛化实验验证**：仅使用 Qwen 3.x 基础模型，未经任何异步专项训练，即在流式视频理解、交互式电子游戏、系统日志监控三大赛道实现实时响应，证明“异步性”可视为 LLM 的通用能力而非垂直技能。

## 方法详解
- **CacheBlocks 与 Cache Views 机制**：每次前向传播将模型内部状态（各层 KV 或 GDN 状态）写入专属 `CacheBlock`。协程通过 `cache_view=[block_A, block_B, ...]` 任意组合多个 Block 构成当前注意力输入，数学上等价于将这些 Block 顺序拼接后做标准自回归，但仅需计算当前步的 attention。
- **全注意力层的并发计算**：沿用 Rodionov et al. (2025) 技巧，不旋转已缓存的 Keys/Values，仅对当前 Query 施加相对位置旋转：$\langle \rho(q, i), \rho(k, j) \rangle = \langle \rho(q, i-j), k \rangle$，从而支持任意顺序的 Block 拼接与动态视图切换。
- **GDN/线性注意力的分块仿射组合**：GDN 状态更新重写为 $S_t = S_{t-1}A_t + B_t$。整个 Block 的状态跃迁可压缩为仿射对 $(\hat{A}, \hat{B})$，两相邻 Block 组合规则为 $(\hat{A}_L, \hat{B}_L) \circ (\hat{A}_R, \hat{B}_R) = (\hat{A}_L\hat{A}_R,\ \hat{B}_L\hat{A}_R + \hat{B}_R)$，计算复杂度从 $O(N_{\text{tokens}})$ 降至 $O(N_{\text{blocks}})$，且支持任意视图重排。KDA、标准线性注意力等变体均可套用相同仿射范式。
- **多模态位置编码（MRoPE）适配**：对视觉 Token 仅按时间轴（temporal）旋转当前 Q，空间轴（height/width）保持不变；同时显式维护每 Block 的时间跨度（position spans）以正确处理图文混合序列的位置对齐。
- **调度与推理引擎**：基于 mini-SGLang 构建，采用**分块 Prefill（chunked prefilling）**与**公平平衡批处理**策略：将不同长度/视图的请求按 token 需求量切分进同一批次，防止快速反应协程被长耗时后台任务饥饿；底层使用 PagedAttention 管理轻量级 Block 引用，并针对 query-rotation attention 与 Flash Linear Attention 定制 kernel。

## 实验与结果
- **数据集与基线**：文本/视觉中断测试（MATH-500-Sharded, ShardedVQA）；流式视频（SoccerNet-Caption, ProactiveVideoQA，对比专用模型 Mage-VL）；电子游戏（ViZDoom: HealthGathering, DeadlyCorridor，对比序列基线与 probe-only）；系统监控（DevOps-Gym Monitoring）。
- **关键结果**：
  - **流式视频**：Qwen3.5-9B/27B/35B-A3B 在 SoccerNet 与 ProactiveVideoQA 上全面超越专为流式视频训练的 Mage-VL（SoccerNet AUROC 最高 **0.677** vs Mage-VL 0.555；ProactiveVideoQA TriggerAcc **52.99** vs 43.27）。
  - **电子游戏**：AsyncLLM 智能体在保持推理质量的同时，动作响应延迟显著低于纯序列基线（Figure 4），证明“慢思考协程+快动作协程”的并发架构有效。
  - **系统监控**：AsyncLLM 在 DevOps-Gym 上以相当精度（Q3.6-35B-A3B 达 **61.76%**）运行，前向传播次数与平均事件响应比例较序列 baseline 大幅下降（Table 2，异步模式平均仅 2.06 个活跃协程即可实时对齐日志流）。
  - **吞吐量**：单卡 H200 上，8 个协程并发时 9B 模型吞吐可达 **552 tok/s**（
