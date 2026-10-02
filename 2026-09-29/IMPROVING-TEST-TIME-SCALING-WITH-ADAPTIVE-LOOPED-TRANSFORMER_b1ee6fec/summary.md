---
title: "IMPROVING-TEST-TIME-SCALING-WITH-ADAPTIVE-LOOPED-TRANSFORMER"
source: https://arxiv.org/pdf/2609.35748v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:29:19"
field: "大语言模型推理效率与测试时缩放"
keywords: ["test-time scaling", "looped transformer", "adaptive computation", "post-training", "iteration decider", "lookahead supervision", "reasoning LLM"]
innovations: ["通过在线迭代增益标签联合后训练主干与决策器，避免训练-推理不匹配", "cost-sensitive 深度监督 + coverage cutoff 设计提升标签质量", "扩展 duo-causal attention 支持 no-gradient lookahead 训练"]
benchmarks: ["AIME24-26", "AMC23", "MATH500", "OlympiadBench", "GPQA", "SuperGPQA", "HumanEval", "MBPP", "LiveCodeBench v6", "BFCL v3"]
---

# 论文速读：IMPROVING TEST-TIME SCALING WITH ADAPTIVE LOOPED TRANSFORMERS

## 一句话总结
本文提出 TaH2，一种通过联合后训练主干网络与迭代决策器来提升循环 Transformer 测试时缩放效率的方法，利用在线测量迭代增益作为监督信号动态决定每个 token 的迭代深度，在 AIME 基准上相比非循环基线将准确率-计算斜率提升 53%，并在匹配计算量下超越基线峰值约 3.4 分。

## 研究问题与动机
- **现有循环模型在测试时缩放下表现不足**：Post-training 后的循环 Transformer（Ouro、Huginn）虽然准确率-计算斜率更陡，但在匹配解码 FLOPs 时仍低于非循环基线。
- **固定深度迭代浪费计算**：分析显示 52.3%（Ouro）和 33.5%（Huginn）的 token 在第一到最后一次迭代的预测损失变化不超过 $10^{-3}$，甚至 21.7% 和 15.9% 的 token 变得更差，说明大量 token 并不需要额外迭代。
- **已有自适应方法存在训练-推理不匹配**：Ouro 的训练仅在全深度下进行，退出门在推理时的早期退出会造成 train-inference mismatch；TaH 使用离线不匹配标签和分阶段训练，无法利用当前主干的实际迭代增益。
- **缺乏系统性的测试时缩放对比**：以往研究多在匹配参数或 per-token FLOPs 下比较循环与非循环模型，未系统研究输出 token 预算增长时循环架构的缩放行为。

## 核心贡献（创新点）
1. **首次系统研究 post-training 循环模型的测试时缩放行为**：通过输出 token cutoff sweep（4K–16K）测量 accuracy–compute slope，发现循环模型斜率虽更陡但仍在匹配计算下落后。
2. **提出 TaH2 联合后训练框架**：同时训练主干网络与轻量迭代决策器（增量参数 <3%），通过 lookahead depth supervision 直接监督每步深度决策，而非使用离线标签或分阶段训练。
3. **在线迭代增益标签 + 成本敏感权重设计**：在每个迭代步骤基于当前主干预测损失变化 $\delta_t^{(m)}$ 在线生成 continue/stop 标签，并按距 cutoff 的距离加权，优于 uniform 权重和 Top-1 mismatch 标签。
4. **扩展 duo-causal attention 支持训练时前瞻**：停止的 token 会执行无梯度的 lookahead 迭代以获取监督信号，同时保证注意力规则与推理一致。
5. **多尺度验证**：在 1.7B、4B、8B 三种规模下均验证，证明增益可随 $M$ 增大而持续增长（+2.8→+3.9 分），且泛化到代码、QA、tool use 领域。

## 方法详解
**架构组件**：
- **Backbone**：复用 Qwen3-Base 的全部 $L$ 层 Transformer，所有迭代共享参数。
- **Updater** $\mathcal{U}_\psi$：轻量 RMSNorm + SwiGLU MLP（$\mathcal{B}_u$），融合原始 token embedding $\mathbf{e}_t$ 与前一轮 hidden state $\mathbf{h}_t^{(m)}$，映射回 backbone 输入空间：$\mathbf{h}_t^{(m+1)} = \mathcal{F}_\theta(\mathcal{U}_\psi(\mathbf{e}_t, \mathbf{h}_t^{(m)}))$。
- **Decider** $\mathcal{D}_\phi$：轻量 RMSNorm + SwiGLU MLP（$\mathcal{B}_d$），输入为 $\mathbf{e}_t$、$\mathbf{h}_t^{(m)}$ 和 Top-K 预测概率（$k=2048$），输出继续概率 $g_t^{(m)} \in (0,1)$。
- **扩展 duo-causal attention**：第 $m$ 轮 token $t$ 可 attends 到位置 $s \leq t$、深度 $j \leq m$ 的 KV 状态；停止 token 额外执行无梯度 lookahead 迭代，lookahead 查询不能 attend 到其他 token 的 lookahead 状态。

**在线深度监督（Lookahead Depth Supervision）**：
- 定义损失减少量 $\delta_t^{(m)} = \ell_t^{(m)} - \ell_t^{(m+1)}$，正增益利于继续。
- 对当前迭代执行了 $a_t^{(m)}=1$ 且上一轮未被标记停止的 token，按 $\delta_t^{(m)}$ 排序取 top fraction $\rho=0.99$ 的总正增益对应的 cutoff $\delta_{cut}^{(m)}$，满足 $\delta_t^{(m)} \geq \delta_{cut}^{(m)}$ 的标记为 $c_t^{(m)}=1$（继续），否则 $c_t^{(m)}=0$（停止）；一旦标记停止则后续全为零。
- 停止 label 通过 no-gradient lookahead 获取下一轮损失。

**联合目标函数**：
$$\mathcal{L}_{SFT} = \frac{1}{N}\sum_t -\log \mathbf{q}_t[y_t^*] + \alpha_D \frac{1}{N}\sum_{m=1}^{M-1}\sum_{t: a_t^{(m)}=1} w_t^{(m)} \text{BCE}(g_t^{(m)}, c_t^{(m)})$$
- 第一项为 next-token prediction loss，使用 stopping-weighted mixture $\mathbf{q}_t = \sum_m \omega_t^{(m)}\mathbf{q}_t^{(m)}$（Eq. 4）。
- 第二项为 cost-sensitive decider loss，权重 $w_t^{(m)} = \text{clip}(|\delta_t^{(m)}-\delta_{cut}^{(m)}|, 10^{-6}, 1)$， farther from cutoff 则权重更大。
- 超参：$\alpha_D=0.05$，coverage $\rho=0.99$，exit threshold $\tau_{exit}=0.5$。

## 实验与结果
**实验设置**：
- 骨干：Qwen3-Base（1.7B/4B/8B），从同一 checkpoint 后训练。
- 训练数据：AM-Qwen3-Distilled（math/code/QA）+ Nemotron-Agentic-v1（tool use），3 epoch / 16K context / 273K prompts（1.7B 实验），TPP=2。
- 评估基准：数学（AIME24–26、AMC23、MATH500、OlympiadBench、IMO-AB）、QA（GPQA、SuperGPQA）、代码（HumanEval、MBPP、LiveCodeBench v6）、tool use（BFCL v3）。
- 基线：Standard（$M=1$）、TaH2-fixed（无决策器固定深度）、Ouro（full-stack，固定/自适应）、Huginn（middle-block，固定深度）。

**主要结果（1.7B，表1）**：
- TaH2 ($M=2$) 平均准确率 40.6% vs Standard 37.7%（+2.9），FLOPs/token 仅 1.21×；$M=8$ 达 42.5%（+4.8），FLOPs 2.37×。
- AIME25：Standard 13.0 → TaH2($M=8$) 17.9（+4.9）；AIME24：11.0 → 16.3（+5.3）；AIME26：11.9 → 14.5（+2.6）。
- 代码：HumanEval 63.6 → 72.6（+9.0, $M=8$）。

**测试时缩放（图1a/4b）**：
- TaH2($M=2$) slope = 2.74 pts/doubling vs Standard 1.79（+53%）。
- 在 32K 评估下，Standard 饱和于 12.0%（191.7 TFLOPs/response），TaH2 在同计算量达 15.4%（+3.4 pts）。
- 并行缩放：cons@32 在 AIME 上 27.3–29.3% vs Standard 21.9%。

**深度缩放（图1b/4a）**：
- 随 $M$ 从 2→8，TaH2 相对于 Standard 的增益从 +2.8 分增长到 +3.9 分；Ouro/Huginn 基本 plateau。
- Validation loss 持续下降，TaH2($M=8$) 最低 −0.0106。

**更大尺度（表3）**：
- 4B：AIME24 52.0 → 57.7（+5.7），平均 +3.2 分。
- 8B：AIME24 66.4 → 70.8（+4.4），平均 +2.4 分。

**运行时效率（表2）**：
- GFLOPs/token 增加 22%，端到端延迟增加 30–34%，但实际 serving 下精度-延迟曲线优于 Standard。

## 相关工作脉络
- **Ouro (Zhu et al., 2025)**：full-stack recurrence，两阶段训练（先全深度训练 backbone，再冻结 backbone 训 exit gate）；TaH2 联合训练且在线监督，避免 train-inference mismatch 和 gate collapse。
- **Huginn (Geiping et al., 2025a)**：middle-block recurrence with input adapter，固定深度；TaH2 同样 full-stack 但引入自适应深度。
- **TaH (Fu et al., 2025b)**：使用离线 mismatch labels 和分阶段训练（先训 decider 再联合微调）；TaH2 改为在线增益标签 + 端到端联合训练，avg 提升 1.1 pts。
- **SMELT (Wang et al., 2026b)**：在匹配 per-token FLOPs、参数和 KV cache 下拟合 pretraining scaling laws；本文聚焦 post-training 后的 test-time scaling 行为。
- **Pondering / AdaPonderLM**：通过 loss + budget/confidence penalty 学习深度；TaH2 使用在线测量的实际损失增益而非置信度。
- **LayerSkip / MoR**：沿 width 或 depth 做 adaptive computation 的其他路线；TaH2 聚焦 looped Transformer 场景下的 token-level 深度分配。

## 局限性与未来方向
- **训练成本增加**：$M=2$ 时训练 FLOPs 是 Standard 的 1.87×，label 看向前向传播占总训练成本的 20–24%（尽管远小于预训练开销）。
- **仅验证于 SFT 设置**：尚未探索 on-policy distillation 或 RLHF 场景下的扩展（论文自述 limitation）。
- **推理深度 ceiling 超出训练时表现不稳定**：$M=8$ 模型在 inference ceiling=4 时下降 2.7–3.1 分，说明深度 extrapolation 能力有限。
- **未讨论极端长上下文下的 KV 缓存压力**：扩展 duo-causal attention 需要每个深度独立 KV cache，长序列下的显存占用需进一步工程优化。
- **单一决策阈值 $\tau_{exit}=0.5$**：不同任务/深度下可能不是最优，自适应阈值值得探索。

## 研究启发与可借鉴点
1. **在线监督信号的优雅设计**：用当前主干实测的 $\delta_t^{(m)}$ 作为 continue label，相比离线 mismatch 或 confidence-based 信号更直接反映"真正收益"，这种"self-supervised depth label"思路可迁移到其他 adaptive computation 场景（如 expert routing、layer skipping）。
2. **Cost-sensitive 权重 + coverage cutoff**：按距 cutoff 的距离加权 + 只取 top 99% 正增益，有效过滤噪声标签，这一策略可直接复用到需要二值化连续信号的决策学习中。
3. **扩展 causal attention 支持 no-gradient lookahead**：在不改变推理 attention mask 的前提下，训练时额外执行无梯度迭代以获取监督，实现了训练-推理一致性，是处理"需要看到未来才能做决策"问题的通用技术。
4. **Stopping-weighted mixture 输出聚合**：各深度预测按停止概率加权混合，而非只用最终预测，简单但稳定，值得在 multi-step reasoning 中尝试。
5. **Scaling 曲线作为核心评估指标**：将 accuracy–compute slope 和 depth-scaling 作为 primary metric，比单点 benchmark 更能反映方法在实际长输出场景下的价值，建议在自己的工作中也采用类似曲线。

## 关键术语表
- **Test-time scaling**：在推理阶段增加计算预算（更多 token 或更多 latent 迭代）以提升模型性能的现象，通常用 accuracy 随 FLOPs 增长的速度（slope）衡量。
- **Looped Transformer**：通过参数共享将固定 Transformer 层重复 $M$ 次来增加有效深度，而不增加参数量的一类架构（如 Ouro full-stack、Huginn middle-block）。
- **Iteration decider**：轻量级网络，在每个迭代步后输出继续概率 $g_t^{(m)}$，决定是否执行下一轮迭代。
- **Lookahead depth supervision**：通过在当前主干上实测相邻迭代的损失变化 $\delta_t^{(m)}$ 生成在线 continue/stop 标签，并用 no-gradient 前瞻迭代获取下一轮损失。
- **Extended duo-causal attention**：在 duo-causal attention（当前 token 可 attend 到之前 token 的各深度状态）基础上，允许 stopped token 的 lookahead 查询 attend 到已执行深度的状态但不跨 token 看 lookahead。
- **Stopping-weighted mixture**：各深度预测 $\mathbf{q}_t^{(m)}$ 按停止概率 $\omega_t^{(m)}$ 加权混合得到最终分布，而非仅用最后一次迭代。
- **Coverage $\rho$**：在每个迭代步只取正增益累积占比达到 $\rho$ 的 cutoff 来生成标签，排除边际增益以防噪声（本文 $\rho=0.99$）。
- **Cost-sensitive weight $w_t^{(m)}$**：决策器 BCE 损失中按 $|\delta_t^{(m)}-\delta_{cut}^{(m)}|$ 加权，远离 cutoff 的样本监督更强。

## 可复现要素
- **数据集**：AM-Qwen3-Distilled（https://huggingface.co/datasets/a-m-team/AM-Qwen3-Distilled）和 Nemotron-Agentic-v1（https://huggingface.co/datasets/nvidia/Nemotron-Agentic-v1）均为公开数据。
- **基座模型**：Qwen3-Base 1.7B/4B/8B 公开可用。
- **代码/权重**：论文声明 "Training and evaluation code, configuration files, and checkpoints will be released upon publication"，发表前暂不可用。
- **关键超参**：lr=$4\times10^{-5}$，batch=128，epochs=3，seq_len=16384，cosine warmup 3%，$\rho=0.99$，$\alpha_D=0.05$，$\tau_{exit}=0.5$，bf16 精度。
