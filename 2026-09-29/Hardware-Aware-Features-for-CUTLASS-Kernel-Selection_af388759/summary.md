---
title: "Hardware-Aware-Features-for-CUTLASS-Kernel-Selection"
source: https://arxiv.org/pdf/2609.35587v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:28:30"
---

# 论文速读：Hardware-Aware-Features-for-CUTLASS-Kernel-Selection

## 一句话总结
提出一种硬件感知特征表示，将CUTLASS GEMM核候选配置与静态可计算的硬件行为代理（流水线平衡、资源压力、显存驻留等）绑定，结合学习排序模型实现免执行的高效核选择；在4.9M核训练集与完全枚举评估集上，平均选择遗憾率降至6.2%（MLP）/6.4%（XGBoost），相对NVIDIA官方启发式(nvMMH)的17.3%降低64.2%，并显著改善跨精度与跨epilogue的迁移数据效率。

## 研究问题与动机
- **配置空间爆炸**：CUTLASS为单一GEMM算子暴露超6万种等价实现（tile形状、流水线级数、cluster、调度策略、epilogue等），穷举编译与基准测试代价极高，甚至对单张GH200也需数百万次测量。
- **现有方法不足**：解析型选择器依赖手工规则，脆弱且难以随架构演进；学习型自动调优器直接使用原始模板参数，迫使模型从零推断硬件效应，表征效率低且样本需求大。
- **Hopper执行机制高度间接**：WGMMA、TMA、warp-specialized pipeline等引入异步流水线，单个参数（如stage count）对性能的影响取决于显存占用、occupancy、K迭代次数等多重耦合因素，无法从裸参数直接推断。
- **表征断层**：原始参数与真实性能之间存在“黑盒”，缺乏在编译前即可静态估算的执行行为中间层，导致学习模型难以捕捉跨形状、跨精度的通用性能规律。

## 核心贡献（创新点）
- **硬件感知特征图（Hardware-Aware Feature Map）**：将候选配置映射为结构化核心+工作分解+显存行为+资源压力+流水线平衡五类静态特征，显式暴露参数对硬件行为的影响机制；与仅使用模板参数的基线本质不同，该表示提供了强架构归纳偏置。
- **大规模测量语料与biased-diverse采样策略**：构建4.9M核训练集，发现高性能核仅占有效目录约2.4%（5%阈值），提出75%利用+25%探索的采样策略，使采样遗憾从8.5%降至0.9%，并保证负样本可见性。
- **免执行学习排序器**：将核选择建模为组内列表排序问题，支持XGBoost（NDCG/pairwise/MSE）与MLP（LambdaRank/RankNet/MSE）两种模型族，推理时直接对全量有效目录打分取argmax，无需实测。
- **跨域泛化与迁移验证**：在68组完全枚举HF16评估、8000个未见BF16问题、DeepBench外部基准及FP32/FP8/FP16 epilogue融合场景中，硬件感知表示 consistently 超越nvMMH，并在仅5%目标域形状覆盖下实现>1.2×几何平均加速。

## 方法详解
- **特征筛选原则**：基于408份Nsight Compute profiling，计算组内Spearman相关性$\rho_g$剔除尺寸效应噪声（如`dram_read_bytes`全局ρ=+0.903但组内≈-0.055）；仅保留统计显著、可静态估算、与执行机制有因果联系的度量。
- **Structural Core**：$\log_2 M,N,K$、操作数布局/类型、tile形状$(T_M,T_N,T_K)$、stage数$S$、mainloop/epilogue调度类型、cluster形状、tile scheduler、架构常量（SM数、shared mem/寄存器容量、HBM带宽、WGMMA峰值）。
- **Work Decomposition**：输出tile数$n_M n_N$、reduction迭代$n_K$、边界浪费$w_M,w_N$、SM订阅与wave效率$\eta_{last}$、Stream-K适用性代理$\max(0,1-\eta_{last})$、cluster贴合度。
- **Memory Behavior**：问题级算术强度$I_{problem}=2MNK/(b_A MK+b_B KN+Q_{epi})$、tile级$I_{mainloop}=2T_M T_N/(b_A T_M+b_B T_N)$、重流因子$r_{stream}$、working-set与resident-panel相对LLC容量比例。
- **Resource Pressure**：单stage缓冲$B_{stage}=T_M T_K b_A+T_N T_K b_B$、总shared memory $B_{smem}$及其占比$f_{smem}$、寄存器压力代理$r_{proxy}=T_M T_N/T_{block}$、约束驻留上限$b_{res}=\min(b_{smem},b_{reg},b_{arch})$。
- **Pipeline Balance**：TMA加载时间$t_{producer}=n_{SM}B_{stage}/\widehat{BW}_{TMA}$、WGMMA计算时间$t_{consumer}=2T_M T_N T_K/(n_{consumer}\widehat{P}_{WG MMA})$、生产-消费比$r_{pc}$、填充率$f_{fill}=\min(S,n_K)/S$、摊销次数$a_{stage}=n_K/S$。
- **学习排序建模**：组内归一化吞吐量$y(g,c)=\hat{T}(g,c)/T_g^\star$，按测量不确定性合并相邻项得到离散等级$\gamma$，仅Top-band赋非零relevance；训练优化NDCG/pairwise损失，评估严格使用遗憾率$R(g)=1-y(g,\hat{c}(g))$。
- **交叉验证与搜索**：shape-grouped 5-fold CV（同shape四布局同fold），Optuna TPE联合搜索超参与训练损失类型；最终在全部训练集refit一次后在68组oracle上测试。

## 实验与结果
- **数据集与环境**：训练集593 base shapes ×4 layout = 2372组，biased-diverse采样（默认2000候选/组），共≈4.9M核；评估集17个held-out shape ×
