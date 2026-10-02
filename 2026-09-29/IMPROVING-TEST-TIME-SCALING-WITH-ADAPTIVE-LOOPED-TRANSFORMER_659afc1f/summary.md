---
title: "IMPROVING-TEST-TIME-SCALING-WITH-ADAPTIVE-LOOPED-TRANSFORMER"
source: https://arxiv.org/pdf/2609.35748v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 14:29:11"
field: "大模型推理效率与测试时计算扩展"
keywords: ["test-time scaling", "looped transformer", "adaptive computation", "post-training", "iteration decider", "depth supervision", "reasoning efficiency"]
innovations: ["提出TaH2联合后训练循环主干与迭代决策器，通过在线增益监督实现token级自适应深度", "定义并系统度量精度-计算斜率揭示后训练循环模型测试时扩展行为", "Lookahead深度监督机制利用当前主干实际损失增益在线生成深度标签"]
benchmarks: ["AIME24", "AIME25", "AIME26", "AMC23", "MATH500", "OlympiadBench", "GPQA", "SuperGPQA", "HumanEval", "MBPP", "LiveCodeBench v6", "BFCL v3"]
---

# 论文速读：IMPROVING TEST-TIME SCALING WITH ADAPTIVE LOOPED TRANSFORMERS

## 一句话总结
本文研究了将循环Transformer（looped transformer）通过**后训练**引入预训练LLM后，其**测试时扩展（test-time scaling）**行为——即随着解码FLOPs翻倍，精度能提升多少。现有循环模型斜率更陡但在同等计算量下仍不如非循环基线；作者提出 **TaH2**，通过在线监督的迭代决策器联合后训练主干网络与决策器，实现了更高的精度-计算斜率和在匹配计算量下的更高精度。

## 研究问题与动机

1. **循环建模能否改善测试时计算扩展？** 已有工作多比较参数量或per-token FLOPs相同情况下的精度，但未系统回答：当输出token预算增加、解码FLOPs翻倍时，循环结构是否能带来更大的精度增益。
2. **现有后训练循环模型在匹配计算量下不如标准非循环模型。** 实验发现，Ouro（全栈循环）和Huginn（中间块循环）虽然斜率更陡（2.12/2.26 vs. 1.79），但在相同解码FLOPs处精度更低，存在"斜率更陡但截距劣势"的矛盾。
3. **固定深度循环对大量token浪费额外计算。** 分析显示，约21.7%的token在额外迭代后预测损失反而变差；52.3%的token变化几乎可忽略，说明不是所有token都需要相同迭代深度。
4. **既有自适应循环方法存在训练-推理不匹配问题。** Ouro的第二阶段冻结主干只训练gate，导致推理时早退行为与训练时的全深度行为不一致；TaH使用离线不匹配标签分阶段训练，无法反映当前主干的实际迭代增益。

## 核心贡献（创新点）

1. **首次系统研究后训练循环模型的测试时扩展行为。** 定义了"精度-计算斜率"（accuracy–compute slope，每翻倍解码FLOPs的精度提升点数），揭示出循环模型斜率更陡但截距劣势的现象，填补了该领域研究空白。
2. **提出 TaH2——联合后训练主干网络与迭代决策器的自适应循环方法。** 与Ouro的冻结主干+训练gate两阶段策略本质不同：TaH2 同时更新主干和决策器，消除训练-推理不匹配；与TaH的离线不匹配标签相比，使用当前主干在线度量的真实迭代增益作为监督信号。
3. **Lookahead深度监督（lookahead depth supervision）——从实际迭代增益在线生成深度标签。** 通过计算相邻迭代的per-token预测损失差（$\delta_t^{(m)}$）作为标签依据，比Top-1不匹配标签更准确地识别"值得继续迭代"的token，带来平均1.1点的性能提升。
4. **在AIME基准上实现53%的斜率提升与3.4点的匹配计算精度增益。** 1.7B模型在AIME24–26上斜率从1.79提升至2.74；在标准模型饱和点（191.7 TFLOPs/response）处精度高出约3.4点；最大迭代深度M从2增至8时，增益从+2.8点扩大至+3.9点，证明方法具有良好的深度可扩展性。

## 方法详解

**架构组成：**
- **共享Transformer主干** $\mathcal{F}_\theta$：全部L层，各迭代参数共享，使用扩展的双向因果注意力（extended duo-causal attention）。
- **输入注入更新器（updater）** $\mathcal{U}_\psi$：轻量归一化MLP，融合token embedding $\mathbf{e}_t$ 和前次隐藏状态 $\mathbf{h}_t^{(m)}$，提供下一迭代输入：$\mathbf{h}_t^{(m+1)} = \mathcal{F}_\theta(\mathcal{U}_\psi(\mathbf{e}_t, \mathbf{h}_t^{(m)}))$。添加参数仅占总量2.6%。
- **迭代决策器（decider）** $\mathcal{D}_\phi$：轻量MLP，基于embedding、隐藏状态和TopK预测概率，输出继续概率 $g_t^{(m)} \in (0,1)$；低于阈值$\tau_{exit}=0.5$则停止。

**扩展双向因果注意力：** 深度$m$的查询可 attending 到位置$s\leq t$、深度$j\leq m$ 的KV状态；训练中已停止的token仍执行一次无梯度lookahead迭代以计算监督信号，但lookahead查询不能attend到其他token的lookahead状态。

**Lookahead深度监督（核心训练机制）：**
- 对每个被监督token $t$，计算相邻迭代损失差：$\delta_t^{(m)} = \ell_t^{(m)} - \ell_t^{(m+1)}$；正增益表示"再迭代有用"。
- 将所有正增益排序，按覆盖率$\rho=0.99$取截断值$\delta_{cut}^{(m)}$，高于此阈值的token获得继续标签$c_t^{(m)}=1$，低于则$c_t^{(m)}=0$；一旦停止后续标签保持为0。
- 停止无梯度lookahead迭代为已停止token提供$\ell_t^{(m+1)}$以计算标签。

**联合损失函数：**
$$\mathcal{L}_{SFT} = \underbrace{\frac{1}{N}\sum_t -\log \mathbf{q}_t[y_t^*]}_{\text{next-token预测损失}} + \alpha_D\underbrace{\frac{1}{N}\sum_{m=1}^{M-1}\sum_{t:a_t^{(m)}=1} w_t^{(m)}\text{BCE}(g_t^{(m)}, c_t^{(m)})}_{\text{成本敏感决策器损失}}$$
其中最终预测$\mathbf{q}_t$为停止权重加权混合输出：$\omega_t^{(m)}$为token在深度$m$停止的概率质量，$\mathbf{q}_t=\sum_m \omega_t^{(m)}\mathbf{q}_t^{(m)}$；权重$w_t^{(m)}$按$|\delta_t^{(m)}-\delta_{cut}^{(m)}|$计算，距截断越远监督越强。超参：$\rho=0.99$, $\alpha_D=0.05$。

**训练-推理一致性：** 训练时决策是确定性的（threshold而非采样），保证token路由分布与推理一致，使不同深度专业化。

## 实验与结果

**实验设置：**
- **主干：** Qwen3-Base 1.7B/4B/8B，同一checkpoint后训练，相同数据与训练recipe。
- **数据：** AM-Qwen3-Distilled（math/code/QA）+ Nemotron-Agentic-v1（tool use），1.7B实验使用Qwen3-8B作为教师重新生成响应，共273K prompts、3.4B tokens、3 epochs、16384序列长度。
- **评测基准：** 数学（AIME24/25/26、AMC23、MATH500、OlympiadBench、IMO-AnswerBench）、QA（GPQA、SuperGPQA）、代码（HumanEval、MBPP、LiveCodeBench v6）、工具使用（BFCL v3）。
- **评测方式：** zero-shot CoT，temperature=0.6，top-p=0.95，默认最大32K tokens，主指标avg@32。

**主要结果（1.7B，Table 1）：**

| 模型 | AIME24 | AIME25 | AIME26 | 平均 | FLOPs/token |
|------|--------|--------|--------|------|-------------|
| Standard (M=1) | 11.0 | 13.0 | 11.9 | 37.7 | 1.00× |
| TaH2 (M=2) | **15.7** | **16.5** | **14.3** | **40.6** | **1.21×** |
| TaH2 (M=4) | 16.3 | 16.9 | 14.2 | 40.9 | 1.47× |
| TaH2 (M=8) | 16.3 | **17.9** | 14.5 | **42.5** | 2.37× |

- **测试时扩展斜率：** TaH2 (M=2) 为 **2.74** 点/FLOPs翻倍，较Standard（1.79）提升 **53%**，优于Ouro（2.27）和Huginn（2.26）。
- **匹配计算量精度：** 在32K输出下，Standard饱和于12.0%（191.7 TFLOPs/response），TaH2 (M=2) 在相同计算量达到 **15.4%**，高出约 **3.4点**。
- **深度可扩展性：** M从2增至8，TaH2相对Standard的增益从+2.8点扩大到+3.9点；而Ouro和Huginn基本 plateau。
- **更大规模（Table 3）：** 4B模型平均+3.2点，8B平均+2.4点；AIME上4B最高+6.9点，8B最高+4.4点。跨领域（代码、QA、工具使用）均有提升。
- **运行时效率（Table 2）：** TaH2 (M=2) 仅增加22% FLOPs/token、30-34%延迟，但实际服务中的精度-延迟曲线更优。

## 相关工作脉络

1. **测试时扩展（test-time scaling）：** Jaech et al. (2024) o1、Guo et al. (2025) DeepSeek-R1 通过生成更多token扩展计算；Muennighoff et al. (2025) s1 研究推理长度控制。本文关注的是潜在空间中的重复计算而非token空间中的链式思考。
2. **循环Transformer基础架构：** Dehghani et al. (2019) Universal Transformer 提出层共享思想；Zhu et al. (2025) Ouro 实现全栈循环；Geiping et al. (2025a) Huginn 实现中间块循环。本文将这些架构引入后训练场景并研究其测试时扩展性质。
3. **后训练循环引入预训练模型：** McLeish et al. (2025) 研究retrofitted recurrence；Fu et al. (2025b) TaH 使用离线不匹配标签分阶段训练决策器。本文的 TaH2 改进为联合训练+在线增益监督。
4. **自适应计算与早退：** Graves (2016) ACT 提出自适应计算时间；Schuster et al. (2022) 早退；Raposo et al. (2024) MoD 动态路由。本文聚焦于token级迭代深度决策而非层选择。
5. **计算匹配下的循环扩展研究：** Prairie et al. (2026) Parcae 在固定唯一参数下研究循环扩展；Schwethelm et al. (2026b) Iso-Depth 固定有效深度比较参数保留量；Wang et al. (2026b) SMELT 在匹配per-token FLOPs下拟合预训练扩展律。本文从**后训练测试时**视角补充这些预训练研究。
6. **Pondering/adaptive depth方法：** Zeng et al. (2025, 2026) Pretraining with pondering；Li et al. (2026a) PonderLM-3；Song et al. (2026) AdaPonderLM。本文与这些方法的区别在于：使用在线度量损失增益而非置信度/惩罚项生成深度标签，且联合训练而非冻结主干。

## 局限性与未来方向

1. **训练FLOPs更高：** TaH2 (M=2) 训练成本约为Standard的1.87倍，M=8时为5.17倍；虽然后训练远小于预训练成本，但在超大规模后训练中仍是一个开销。
2. **仅研究SFT后训练场景：** 方法尚未扩展到on-policy distillation和强化学习设置，如DeepSeek-R1式的RL训练。
3. **推理深度超出训练上限的局限：** 将M=8模型的推理深度降至4会损失2.7-3.1点精度，说明模型对训练深度的依赖较强，外推能力有限。
4. **仅评估了单一后训练数据配方：** 教师模型选择和蒸馏数据质量对最终性能影响较大（Appendix B.1），最优教师并非最大的235B模型而是8B模型，需进一步研究。

## 研究启发与可借鉴点

1. **"精度-计算斜率"作为评测指标值得推广。** 用线性拟合accuracy vs. log₂(FLOPs)的斜率来衡量测试时扩展效率，比单点精度更能反映方法的实际价值；可迁移到任何测试时计算扩展研究中。
2. **在线监督信号比离线标签更有效。** TaH2的核心洞察是用当前主干实际度量的损失增益而非静态标签训练决策器，这解决了训练-推理分布不匹配问题；该方法论可迁移至任何需要动态计算分配的场景（如动态深度MoE、early-exit网络）。
3. **成本敏感加权（cost-sensitive weighting）提升决策器训练质量。** 按增益与截断值的距离加权BCE损失，使"明显应该继续/停止"的token获得更强监督，这是一个简单有效的训练技巧。
4. **后训练引入循环是一种低成本的测试时扩展路线。** 相比从头预训练循环模型或增加推理token数，后训练仅需约1.87倍SFT FLOPs即可显著提升推理能力，适合资源受限场景；可考虑与本团队的方向结合，探索在不同任务域（如代码生成、多轮对话）上的适用性。
5. **深度可扩展性实验设计值得借鉴。** 通过改变最大迭代深度M（2/4/8）并观察相对增益的变化趋势，清晰展示了方法的scaling特性，这种实验设计可以复用到其他深度自适应方法的评价中。

## 关键术语表

**Test-time scaling（测试时扩展）：** 通过增加推理阶段计算量（更多token或多次潜在迭代）来提升模型推理精度的方法范式。

**Looped transformer（循环Transformer）：** 通过参数共享重复应用Transformer层以增加有效深度的架构，在不增加参数量前提下获得更多计算。

**Accuracy–compute slope（精度-计算斜率）：** 衡量测试时扩展效率的指标，定义为解码FLOPs每翻倍带来的精度提升点数。

**Lookahead depth supervision（前瞻深度监督）：** 通过在线计算相邻迭代预测损失差$\delta_t^{(m)}$来生成token级继续/停止标签的训练机制。

**Iteration decider（迭代决策器）：** 轻量神经网络，在每个迭代后输出继续概率，决定每个token是否执行额外迭代。

**Updater / 输入注入更新器：** 在迭代间融合原始token embedding与当前隐藏状态的归一化MLP，防止信息衰减。

**Extended duo-causal attention（扩展双向因果注意力）：** 支持多迭代深度下因果注意力掩码的变体，允许深度$m$查询attend到所有深度$\leq m$的KV状态。

**Stopping-weighted mixture（停止加权混合输出）：** 将各迭代深度的预测按停止概率权重加权融合为最终输出，而非仅取最后一次迭代预测。

## 可复现要素

- **数据集：** AM-Qwen3-Distilled（HuggingFace公开）+ Nemotron-Agentic-v1（HuggingFace公开），训练时由Qwen3-8B重新生成响应。论文未提及私有数据。
- **代码：** "Training and evaluation code, configuration files, and checkpoints will be released upon publication."（论文声明发表时开源）
- **模型权重：** 使用Qwen3-1.7B/4B/8B-Base公开checkpoint作为初始化。
- **关键超参：** global batch=128，sequence length=16384，learning rate=$4\times10^{-5}$，epochs=3，warmup ratio=0.03，$\rho=0.99$，$\alpha_D=0.05$，$\tau_{exit}=0.5$，precision=bf16，TPP=2。
