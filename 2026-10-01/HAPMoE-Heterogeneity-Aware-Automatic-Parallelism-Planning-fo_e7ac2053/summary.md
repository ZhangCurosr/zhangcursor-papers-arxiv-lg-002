---
title: "HAPMoE-Heterogeneity-Aware-Automatic-Parallelism-Planning-fo"
source: https://arxiv.org/pdf/2609.39350v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 18:35:31"
field: "大规模模型分布式训练系统"
keywords: ["自动并行", "Mixture-of-Experts", "异构训练", "Pipeline并行", "Megatron-LM", "分布式训练"]
innovations: ["异构感知的6维MoE并行空间搜索（PP/TP/DP/CP/EP/TPE）", "非均匀PP/DP分区机制支持异构集群高效调度", "剪枝增强动态规划搜索将搜索时间压缩至60秒内"]
benchmarks: ["Mixtral-S", "Mixtral-L", "LLaMA-2 7B", "LLaMA-2 13B"]
---

# 论文速读：HAPMoE: Heterogeneity-Aware Automatic Parallelism Planning for Mixture-of-Experts Models Training

## 一句话总结
论文提出 HAPMoE，一个面向 Mixture-of-Experts (MoE) 模型的异构感知自动并行规划系统，通过构建轻量级 MoE 成本模型并在 6 维并行空间（PP/TP/DP/CP/EP/TPE）中高效搜索，自动生成可直接部署到 Megatron-LM 的训练并行策略，在异构集群上将端到端训练吞吐最高提升 3.2×。

## 研究问题与动机
- **MoE 与异构集群双重挑战共存**：MoE 引入动态 token 路由、重 All-to-All 通信和潜在专家负载均衡问题；现代训练集群由不同代/厂商加速器混合组成，内存和网络特性非均匀，传统方法难以同时处理这两个趋势。
- **现有自动并行系统各有局限**：异构感知系统（如 Metis、Holmes）通常假设密集模型，不搜索 EP/TPE 等 MoE 专属维度；MoE 定向规划器（如 Tutel、DeepSpeed-MoE）通常假设同质硬件，缺乏异构感知建模与全局策略搜索。
- **手动并行配置效率低且难以优化**：面对复杂模型与大搜索空间，依赖专家启发式的手动调参既耗时又难以找到全局最优策略。
- **非均匀分区需求未被充分探索**：在异构环境下强制均匀 pipeline/DP 分区会导致严重流水线气泡和负载不均衡，但已有系统大多不支持灵活的非均匀层分配。

## 核心贡献（创新点）
- **异构感知 6D MoE 并行搜索**：首次在同一框架内显式覆盖 EP/TPE 等专家中心化维度，并在异构集群上搜索 PP/TP/DP/CP/EP/TPE 全六维空间，较基线提升吞吐最高 3.2×；与已有工作的本质区别在于同时建模硬件异构性和 MoE 稀疏路由特征，而此前工作仅侧重其一。
- **非均匀 PP/DP 分区机制**：通过联合优化各 stage 的并行旋钮和层分配，允许跨 stage 的不同 (PP, DP) 配置，而非强制统一；相比强制均匀分区，在异构集群上可获得 4%–78% 的吞吐增益。
- **剪枝增强动态规划搜索算法**：提出三种剪枝策略（负载均衡剪枝、内存约束剪枝、异构通信剪枝），将有效搜索空间从 O(PP × N^PP × H^PP) 降至 O(PP × (N/PP)^PP × H)，确保所有评测场景下搜索耗时低于 1 分钟。

## 方法详解
- **Profile-Model-Search 流水线**：HAPMoE 先运行若干 warm-up 迭代对设备 kernel 和通信原语进行轻量 profiling，构建计算/内存/集体通信查找表；然后构建 MoE 感知成本模型；最后用剪枝 DP 搜索生成并行计划。
- **MoE 感知成本模型**：逐层计算 MoE 层的执行时间，公式为 t_moe^comp ≈ t_base − t_ffn + (2·t_ffn)/(η_h · EP · TPE) + t_route(h)，其中 t_route(h) 是与 token 数成正比的路由开销；通信开销通过 dispatch/combine 的 activation 体积 D_moe(π) = (S·B)·H·β·k·(1 − 1/EP) 估算，并依据 primitive 类型和设备链路（ intra-node / inter-node / cross-type ）查表获得通信延迟；负载均衡通过路由统计计算的 ρ_imb 因子修正关键路径时间。
- **内存可行性约束**：对每个 stage i，估计 M_stage = M_w + M_act + M_moe-extra，其中权重内存随 (TP·EP·TPE) 分片，激活内存随存储的 micro-batch 数和 recomputation 比例 ξ(r_i) 缩放；在给定 (h_i, n_i, π) 时选择满足 M_stage ≤ C_{h_i} 的最小 recomputation 级别 r_i。
- **迭代级延迟模型（1F1B 调度）**：将每轮迭代延迟分解为 T_iter = T_edge + T_middle + T_dp-sync + T_opt，其中 T_middle 由瓶颈 stage 决定，即 max_i(t_f,i + t_b,i)；该模型直接用于搜索过程中对候选方案的评分。
- **剪枝增强的 HS-DP 搜索**：定义 DP[i, j, m] 为前 j 层分配给前 i 个 stage 且剩余设备状态为 m 时的最小迭代延迟；三种剪枝分别为：(1) 负载均衡剪枝——限制每 stage 层数在 [ (1−δ)·w_h·N, (1+δ)·w_h·N ] 范围内（δ=0.2）；(2) 内存约束剪枝——超出设备容量则直接舍弃；(3) 异构通信剪枝——当跨类型通信延迟 T_comm^hetero > η·T̄_comp（η=2）时剪除。
- **Megatron-LM 集成**：HAPMoE 解析 Megatron-LM 运行时参数提取模型/训练元数据，仅在训练循环中注入少量 hook（iteration 时间分解、router 统计、内存足迹），warm-up 完成后移除；最终将最优计划 (π*, L*, {h_i*}, {r_i*}) 转换为 Megatron-LM 可执行的 process mesh 和 launcher 脚本，无需手动修改分布式配置。

## 实验与结果
- **实验设置**：使用两个 Mixtral-style MoE 配置（Mixtral-S：24层/8 experts/top-k=2；Mixtral-L：48层/8 experts/top-k=2，约122B参数）及 LLaMA-2 7B/13B 密集模型；硬件涵盖 NVIDIA H800、AMD MI300X、Ascend 910B，构建多种同构/异构集群（16/24/32 卡组合）；使用 BF16 混合精度 + ZeRO-1 优化器。
- **密集模型异构集群**：在 2×8 A100 + 2×8 910B 集群上，HAPMoE 在 LLaMA-2 7B/13B 上均持续优于 MI、Alpa 和 Metis，吞吐和 MFU 均最高，且随模型规模增大优势更明显。
- **MoE 模型同构集群**：在多个同构集群上，HAPMoE 的吞吐和 MFU 持续领先 DeepSpeed-MoE 和 Tutel，因后者在大规模并行下 All-to-All 通信成本高且专家负载不均。
- **MoE 模型异构集群（核心场景）**：HAPMoE 几何平均吞吐相对 MI 提升 1.67×（Mixtral-S）和 1.78×（Mixtral-L），迭代延迟降至 MI 的 0.56× 和 0.58×；MFU 分别达到 MI 的 1.72× 和 1.73×；Metis-style 和 HeterMoE-style 也有提升但均低于 HAPMoE。
- **搜索效率与预测精度**：所有评测场景下搜索时间 < 60s；延迟估计误差最大约 2.8×（极端异构场景），吞吐估计误差最高约 38%，整体预测可靠。
- **非均匀 PP/DP 消融**：关闭非均匀分区后，在同构集群上延迟增加 10–18%；在异构集群上延迟膨胀 2.3–3.4×，吞吐下降至 45–62%；含 910B 的配置最敏感，验证了非均匀分区的必要性。
- **Profile 扩展性**：4 节点 profiling 结果可用于 16 节点策略搜索，同构集群延迟误差约 ±10%，异构集群最大偏差约 14%。

## 相关工作脉络
- **Alpa (Zheng et al., 2022)**：基于 ILP/DP 的自动并行搜索系统，但假设同构硬件且未覆盖 EP/TPE 维度，无法直接处理 MoE 异构训练。
- **Metis (Um et al., 2024)**：异构感知自动并行系统，采用启发式分区策略，但未考虑 MoE 特有维度，作者将其改造为 Metis-style 作为基线对比。
- **DeepSpeed-MoE (Rajbhandari et al., 2022)**：面向 MoE 的高效训练系统，但假设同构集群且依赖手动或简单启发式配置，缺乏全局多维搜索能力。
- **Tutel (Hwang et al., 2023)**：专注于单层通信优化和自适应 MoE 调度，不支持全局 6D 并行策略搜索，在异构场景下受限。
- **HeterMoE (Wu et al., 2025)**：异构 GPU 上的 MoE 训练优化，采用 attention/expert 解耦和异步 expert 分配，但缺少全局多维并行空间搜索。
- **MegaScale-MoE (Jin et al., 2026) / X-MoE (Yuan et al., 2025)**：通过路由和 expert 分配优化减少 MoE 通信开销，但未整合异构硬件建模与全局并行搜索。

## 局限性与未来方向
- 当前仅针对 Megatron-LM 风格的 Transformer/MoE 训练，且计划基于训练前 warm-up profiling 生成，不支持训练中动态重新规划。
- 未支持路由分布、负载特征或设备可用性在训练过程中发生变化时的在线调度或动态重配置。
- 评估集群规模有限（最大 32 卡），生产级更大规模集群中的网络争用、故障和拓扑效应尚未覆盖。
- 未来方向包括更大规模部署研究、在线自适应调度机制，以及扩展到更广泛的异构拓扑和故障容忍场景。

## 研究启发与可借鉴点
- **轻量 profiling + 查表加速搜索**：仅需少量 warm-up 迭代即可构建设备/原语查找表，搜索阶段仅做查表和简单算术，将搜索时间压缩至 60s 以内，这一思路可迁移到其他自动并行系统中以降低规划开销。
- **非均匀 pipeline/DP 分区设计**：跨 stage 允许不同并行配置的联合优化，在异构场景下收益显著（最高 78%），可作为异构训练系统设计的通用范式。
- **三种剪枝策略的组合应用**：负载均衡剪枝、内存约束剪枝和通信阈值剪枝分别针对不同维度的搜索空间剪枝，层次清晰且可复用，适用于其他多维并行搜索问题。
- **MoE 路由与通信建模的精细化**：将路由统计（ρ_imb）和 dispatch/combine 体积建模纳入成本函数，为 MoE 相关系统的性能预测提供了可借鉴的建模方法。
- **与主流框架无缝集成的工程实践**：仅注入少量 hook 且训练时完全移除，输出可直接运行的 launcher 脚本，这种"低侵入、高兼容"的设计模式值得在系统集成类工作中参考。

## 关键术语表
- **Mixture-of-Experts (MoE)**：一种通过稀疏激活和条件计算实现参数扩展的神经网络架构，每层包含多个 expert 子网络，仅激活部分 expert 处理每个 token。
- **Expert Parallelism (EP)**：将 expert 分布在多个设备上，通过 All-to-All 通信进行 token 分发与聚合，是 MoE 训练的关键并行维度。
- **Tensor-Parallel Experts (TPE)**：进一步对单个 expert 内部进行张量切分，支持更大规模的 MoE 模型，配置复杂度高于纯 EP。
- **Pipeline Parallelism (PP)**：将模型连续层分配到不同设备上，通过 micro-batch 流水线重叠加速训练，但可能因 stage 负载不均衡产生气泡。
- **1F1B (One-Forward-One-Backward) 调度**：一种 pipeline 调度策略，每个 stage 在完成一次前向后紧接着执行一次反向，以最小化内存占用和气泡时间。
- **MFU (Model FLOPs Utilization)**：衡量训练集群硬件效率的指标，计算实际吞吐与设备峰值浮点运算能力的比值。
- **Heterogeneity-Aware**：指系统能够感知并利用集群中不同设备（加速器、网络）的性能差异，动态调整并行策略以优化整体效率。
- **Non-uniform PP/DP Partitioning**：允许 pipeline 和 data 并行在不同 stage 采用不同配置（层数、设备分配），而非全局统一，以适应异构硬件。

## 可复现要素
- **数据集**：论文未使用特定数据集，实验基于 Mixtral-S/L 和 LLaMA-2 7B/13B 模型架构进行训练效率评测。
- **代码/权重开源**：论文未提及代码开源情况。
- **关键超参**：负载均衡剪枝容差 δ=0.2；异构通信剪枝阈值 η=2；Micro-batch size=1；序列长度=4096；BF16 混合精度 + ZeRO-1 优化器；搜索时间目标 <60s。
