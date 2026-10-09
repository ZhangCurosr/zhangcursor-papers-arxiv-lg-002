---
title: "Opinion-Coding-Agent-Benchmarks-Should-Match-Their-Users-Tas"
source: https://arxiv.org/pdf/2610.09633v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:51:09"
---

# 论文速读：Opinion-Coding-Agent-Benchmarks-Should-Match-Their-Users-Tas

## 一句话总结
本文指出当前代码Agent评测基准（如SWE-Bench系列）采用单轮Issue描述，与真实开发者渐进披露、频繁切换任务类型的交互习惯严重脱节；为此提出SWE-TaskFlow框架，通过提示词拆分与可验证仓库问答将静态基准改造为多轮交互式轨迹，并以TFAS指标校准至实测Task Flow，试点表明交互协议本身是独立于任务难度的关键评测维度。

## 研究问题与动机
1. Issue-derived基准（SWE-Bench Pro等）将任务打包为单次输入（新增功能与Bug修复占比93.8%），无法反映真实IDE中用户混合多种意图、随会话推进动态调整需求的交互模式。
2. 不同公开语料库（SWE-chat、SpecStory、DataClaw）的Task Flow分布差异显著，不存在普适的“现实交互分布”，基准必须明确目标用户场景并据此校准。
3. 现有评测仅依赖终端Patch的二值信号，缺乏对多轮交互过程中每步推理的细粒度验证，难以诊断Agent在渐进式任务中的能力瓶颈。
4. 静态固定消息与动态反应式（Simulated User/Withheld Requirements）两类范式尚未系统性地建立以实测交互分布为目标的校准方法论。

## 核心贡献（创新点）
1. **提出Task Flow三元组度量体系**：以会话长度、意图类型分布、相邻意图有向转移矩阵量化交互流，首次揭示JetBrains生产会话与Issue基准的结构性差异。
2. **设计SWE-TaskFlow转换框架**：在不改动原始仓库、运行环境与测试套件的前提下，通过Prompt Splitting与Repository QA将单轮Issue自动扩充为可回放的多轮轨迹。
3. **定义TFAS对齐指标**：基于基2 Jensen–Shannon散度构建复合得分，使“贴近目标Task Flow”可计算、可优化、可横向比较。
4. **揭示交互协议为独立评测维度**：试点表明顺序拆分使Agent成本约翻倍，但Resolve Rate无稳定跨模型提升，证明交互协议本身显著影响评估结果。

## 方法详解
- **Task Flow形式化**：对任意语料库统计 $(P_L, P_T, P_R)$，其中 $P_L$ 为会话长度分布，$P_T$ 为16类意图（bug-fix、new-feature、explain、planning、code-review等）的消息级分布，$P_R$ 为相邻消息间有向转移分布，三者共同刻画交互流的全局结构。
- **Prompt Splitting**：LLM将单轮Issue按特异性单元、子任务单元、技术术语拆解为K个有序轮次（本文K=3）。约束条件：保留全部原始要求、不引入新需求、技术术语逐字保留、每轮在当前仓库状态下均可独立执行；无法拆分的原子Issue保留单轮。
- **Repository QA生成**：从已求解Agent轨迹与当前仓库状态中提取跨文件稳定行为，生成0~3道可验证问题。问题必须满足：可从仓库推导、补丁前后均有效、不泄露解决方案、附带可执行证明脚本（creation-time proof script），失败候选直接丢弃。
- **TFAS优化与候选筛选**：对每个Issue的候选轨迹族（不同合并程度、QA数量与插入位置），计算 $S_k = 1 - \mathrm{JSD}_2(P_k, Q_k)$（$k \in \{L,T,R\}$），取几何均值得 $\mathrm{TFAS} = 100 \cdot (S_L \cdot S_T \cdot S_R)^{1/3}$。通过随机局部搜索（64次重启、最多20次扫描、固定种子）在任务数与多样性约束下最大化TFAS，选定最终轨迹。
- **逐轮验证设计**：每个Issue转为1~6轮用户输入；每次Issue轮执行后快照仓库并重跑原测试，输出中间Resolve Rate与Test-Pass Rate；QA轮由裁判模型对照隐藏参考答案评分，形成双信号验证管线。

## 实验与结果
- **数据集**：JetBrains生产会话4,782条（聚焦≥3消息长会话切片，占33%）；SWE-Bench Pro公开子集700条。
- **评估基线**：Single（原始单轮）、Split（纯拆分）、Concat（拆后拼接回单轮，控制文本重写混淆）、Split+QA（first/last/random/TAS优化）。使用mini-swe-agent在Harbor隔离环境运行，测试GPT-5.6 Luna/Terra、GPT-5.4 Mini、Gemini 3.7 Flash。
- **主要结果**：
  - TFAS从纯Split的59.6提升至优化
