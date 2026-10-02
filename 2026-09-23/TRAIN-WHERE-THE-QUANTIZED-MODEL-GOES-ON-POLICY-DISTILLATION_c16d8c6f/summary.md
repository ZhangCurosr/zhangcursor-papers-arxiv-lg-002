---
title: "TRAIN-WHERE-THE-QUANTIZED-MODEL-GOES-ON-POLICY-DISTILLATION"
source: https://arxiv.org/pdf/2609.26708v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:25:59"
field: "低比特大模型量化与推理恢复"
keywords: ["low-bit quantization", "quantization-aware distillation", "on-policy distillation", "reasoning recovery", "LLM compression", "exposure bias", "extreme quantization"]
innovations: ["提出 QAD+OPD 两阶段框架，首次在子3比特场景将 on-policy 蒸馏与任务验证器结合恢复长推理", "系统揭示量化放大的 exposure bias 并证明其导致长生成重复循环与超预算耗尽", "在 2.79/1.88 有效比特上使 MATH-500 与 HumanEval 平均 BF16 保持率分别翻倍至 70%/91%"]
benchmarks: ["GSM8K", "MATH-500", "AMC23", "MBPP", "HumanEval", "QA9"]
---

# 论文速读：TRAIN-WHERE-THE-QUANTIZED-MODEL-GOES-ON-POLICY-DISTILLATION

## 一句话总结
本文提出两阶段框架 **QAD + OPD**，通过在量化模型部署时实际经过的自回归轨迹上进行策略蒸馏，有效恢复了子 3 比特（2.79 / 1.88 有效比特）量化后大模型在数学与代码推理上的长生成能力，同时不损害短文本性能。

## 研究问题与动机
- **极端低比特量化的推理退化**：W2.79 / W1.88 下直接 RTN 量化后所有生成式基准（GSM8K、MATH-500、AMC23、MBPP、HumanEval）准确率归零，PTQ 难以弥补。
- **QAD 恢复不均衡**：量化感知蒸馏（QAD）在短文本 QA（QA9）上能恢复约 82% BF16 性能，但 MATH-500 仅恢复 35%、AMC23 仅 16%，长推导显著落后。
- **量化放大的 exposure bias**：QAD 在固定语料前缀上做 teacher-forced 监督，而部署时学生必须沿自身预测的前缀自回归生成；每步量化扰动都会改变后续上下文，偏差在长链推理中累积，最终导致重复循环、超出解码预算。
- **需要让教师监督落在学生真实轨迹上**：恢复长生成能力的缺失指导应集中在量化模型实际访问的 prefix，而非训练语料中的参考前缀。

## 核心贡献（创新点）
- **提出 QAD → OPD 两阶段恢复框架**：QAD 提供可行的低比特策略初始化，OPD 将教师监督扩展到部署量化前向路径上的学生前缀，补齐轨迹控制能力。
- **系统性揭示量化放大 exposure bias 的现象与机制**：通过保留率分层（短/中/长回答）与生成分布分析（终止率、8-gram 循环率）证明长推理退化源于量化扰动沿自回归链的累积。
- **将 on-policy distillation 引入极端低比特场景**：学生通过量化路径采样完成，冻结 BF16 教师对其前缀做 reverse-KL 监督，并与任务验证器 reward（数学答案正确、代码测试通过）联合优化。
- **在 2.79 / 1.88 有效比特上实现推理大幅恢复**：MATH-500 平均保持率从 35% 提升至 70%，HumanEval 从 66% 提升至 91%，同时保持 QA9 短文本性能。
- **在匹配预算下证明 OPD 显著优于继续 teacher-forced QAD**：相同起始 checkpoint、语料与优化步数，OPD 在 16 组对比中全部胜出，且延长 QAD 训练步数几乎不再带来收益。

## 方法详解
- **阶段一：QAD 初始化（EdgeRazor）**
  - 权重按位置混合 INT4 / INT1.58（W2.79: p=0.5；W1.88: p=0.125），嵌入与 lm-head 用 INT4，激活用 INT8，KV cache 保留 16 bit。
  - 以 BF16 同行模型为 teacher，在线 logit KL + 小任务 loss 蒸馏通用指令语料；OPD 所有臂均从复现的 QAD checkpoint 出发。
- **阶段二：OPD（On-Policy Distillation）**
  - 学生从 prompt x 出发，通过**部署量化前向路径**采样 G 个完成 $\hat{y} \sim \pi_\theta(\cdot|x)$。
  - 冻结 BF16 teacher $\pi_T$ 对学生实际前缀做 reverse-KL 稠密监督：
    $$\mathcal{L}_{SF}(\theta) = \mathbb{E}_{x}\mathbb{E}_{\hat{y}}\left[\sum_t \mathrm{KL}(\pi_\theta(\cdot|x,\hat{y}_{<t})\,\|\,\pi_T(\cdot|x,\hat{y}_{<t}))\right]$$
  - 任务验证器提供组内相对 advantage $\hat{A}(x,\hat{y})$：数学用严格答案核对，代码用 pass@1 测试执行。
  - 总损失（反向 KL 项作为 policy gradient 的对数概率梯度）：
    $$\mathcal{L}_{OPD}(\theta) = -\mathbb{E}\left[\sum_t \hat{A}\log\pi_\theta(\hat{y}_t|x,\hat{y}_{<t})\right] + \beta\,\mathbb{E}\left[\sum_t \log\frac{\pi_\theta(\hat{y}_t|x,\hat{y}_{<t})}{\pi_T(\hat{y}_t|x,\hat{y}_{<t})}\right]$$
    本文统一取 $\beta=1$，不做逐设置调参。
  - 训练分两阶段：Math 阶段（GSM8K / MATH L1-3 / DAPO-Math）→ Code 阶段（MBPP / KodCode），每步 8 prompts × G completions；优化器步数约 80–150。
- **行为层面的机制解释**
  - 首约 30 步策略熵从 ~10 nats 骤降至 0.35，distill 项从 2.07 降至 0.25，verifier reward 由负转正；后续进入平稳 plateau。
  - OPD 显著降低重复 8-gram 比例（MATH-500 从 70% 降至 17%）、降低超预算耗尽率（从 95% 降至 53%）。

## 实验与结果
- **模型与比特**：Qwen3-0.6B / 1.7B / 4B、Falcon3-1B-Instruct；W2.79 与 W1.88 两种有效比特（8-bit activations、4-bit embed/lm-head）。
- **评测基准**：GSM8K、MATH-500、AMC23（avg@16/40）、MBPP（pass@1/448）、HumanEval（pass@1/164）、QA9（9 项短文本 QA 均值）。
- **核心指标**：各 checkpoint 相对于对应 BF16 的 retention（百分比）。
- **主要结果（Table 2 汇总）**
  - **MATH-500 平均 retention**：从 QAD 的 35.3% 提升至 QAD+OPD 的 **69.7%**（翻倍）。
  - **HumanEval 平均 retention**：从 66% 提升至 **91%**。
  - **GSM8K 平均 retention**：从 62.7% 提升至 **85.0%**。
  - **短文本 QA9**：从 88.1% 提升至 92.1%，无明显损伤。
  - 以 Qwen3-1.7B W2.79 为例：MATH-500 19.6 → 45.6；HumanEval 41.5 → 59.1（BF16 参考为 67.1）。
  - 更激进的 W1.88 下 Qwen3-4B：GSM8K 17.36 → 64.59，MBPP 11.6 → 48.7，OPD 承担更大恢复份额。
- **效率（Table 3）**：OPD 仅需数百步、约 14–23× 少于 QAD 的 GPU-hours（如 Qwen3-4B W1.88：57 vs 820 GPU-hours）。
- **与基线对比（Figure 4 / Appendix C）**
  - 与 PTQ/QAT/QAD（Q-Palette、FlatQuant、AutoRound、EfficientQAT、EdgeRazor 等）同设置比较：W2.79 下 MATH-500 基线最高约 58%，本文 OPD 达 **84%**；W1.88 下多数基线在生成基准归零，本文 OPD 仍可恢复至 GSM8K 64%、MBPP 77%。
- **消融（Table 4 / Appendix G）**
  - 在匹配步骤、语料与学习率下，OPD 在全部 16 组对比中胜出；继续 QAD 延长至 2× / 3× 步数仍落后 OPD 8.9–18.1 GSM8K points。
- **Teacher 大小（Appendix F）**
  - 1.7B 及以上学生：用自身 BF16 作为 teacher 即可，换 4B/8B teacher 无显著提升。
  - 0.6B 学生：自身 BF16 与量化 student 差距过小，改用 1.7B 教师更优（否则增益有限）。

## 相关工作脉络
- **PTQ（OPTQ / SmoothQuant / AWQ / QuaRot 等）**：依赖少量校准集，3 bit 以上有效；本文关注其低于 3 bit 时推理退化严重、难以恢复长生成的不足。
- **QAT（LLM-QAT / EfficientQAT / FlatQuant / OmniQuant）**：通过模拟低比特前向传播调整参数；多聚焦知识/短文本保持，对累积轨迹偏差恢复不足。
- **QAD（BitDistiller / EdgeRazor）**：全精度教师对固定语料前缀做 teacher-forced KL 蒸馏；本文指出其在长推理轨迹上暴露的 supervision gap。
- **Exposure bias / Scheduled sampling / Sequence-level training**：经典序列建模中训练-推理分布不一致问题；本文将其置于极端量化场景并给出 on-policy 解决方案。
- **On-policy distillation（MiniLLM / DistiLLM / Lu & Thinking Machines Lab 2025）**：在完整模型上利用学生轨迹做蒸馏；本文首次将其系统迁移到 <3 bit 极端低比特推理恢复，并结合任务 verifier。
- **极低位推理稳定性与 RL cold start（ReQAT / What makes low-bit QAT work）**：相关研究指出低比特模型需先建立可行策略；本文以 QAD 提供冷启动、OPD 接力恢复轨迹控制，形成互补定位。

## 局限性与未来方向
- **极小模型仍需更大 teacher**：0.6B 学生自身 BF16 不足以提供有效监督，需借用 1.7B，限制了纯自蒸馏的部署形态。
- **训练成本仍偏重**：尽管 OPD 比继续 QAD 高效 14–23×，但整体 pipeline 仍依赖多阶段、数百步 rollout 与并行 teacher 评估。
- **任务验证器依赖**：数学答案与代码测试的执行验证为硬奖励信号，泛化到其他需要过程奖励的任务（如开放写作、多轮对话）仍需扩展 reward 设计。
- **部署设置假设**：KV cache 保留 16 bit、INT8 activations，若进一步收紧可能影响 OPD 所得策略的实际部署表现。
- **推广范围**：当前主要在 Qwen3 / Falcon3 系列与数学/代码两个垂直任务验证，跨架构（MoE、多模态）、跨任务域的泛化有待检验。

## 研究启发与可借鉴点
- **两阶段分离的恢复范式**：用“稳定初始化（QAD）+ 轨迹校正（OPD）”分工处理不同失败模式，思路可迁移到其它压缩/微调任务（如剪枝后推理恢复、LoRA 低秩退化修复）。
- **on-policy 轨迹监督的价值量化**：通过 matched-budget 对照与延长 QAD 步数的负实验，清晰界定“更多 teacher-forced 数据≠更好长推理”，论证轨迹分布匹配的关键性。
- **reverse-KL + group-relative advantage 的组合**：前者提供 dense token 级指导且可从学生样本估计，后者提供 sparse 但最终正确性信号；两者尺度差异大但互补，可成为低比特 RLVR/蒸馏的通用损失配方。
- ** entropy / reward / distill loss 的联合监控曲线**：论文用 policy entropy 骤降 + verifier reward 转正刻画早期快速修复阶段，为后续类似工作提供可复用的诊断指标与早停依据。
- **Teacher 选择的可迁移结论**：并非 teacher 越大越好；当 student 与 teacher 能力差过小时转移空间有限，应优先保证 teacher 具备可蒸馏的“足够优势”。

## 关键术语表
- **QAD（Quantization-Aware Distillation）**：量化感知蒸馏，以全精度教师对固定语料前缀做 KL 监督，帮助量化学生恢复能力。
- **OPD（On-Policy Distillation）**：策略蒸馏，学生在自身实际生成的前缀上接受教师监督，对齐训练-推理分布。
- **Exposure bias（暴露偏差）**：训练时依赖参考前缀、推理时依赖自身预测所导致的分布失配。
- **Reverse KL**：$\mathrm{KL}(\pi_{student}\|\pi_{teacher})$，可由学生采样估计，惩罚学生高信而教师低信的输出。
- **Task verifier**：提供最终答案/代码执行是否通过的二值或多值奖励信号，用于计算组内 advantage。
- **Effective bits（有效比特）**：通过混合不同位宽的权重组（如 INT4 与 INT1.58）实现的平均位宽。
- **RTN（Round-to-Nearest）**：最简单的无校准量化舍入策略，本文用作零基线。
- **EdgeRazor**：本文复现的混合精度 QAD 初始化框架，按位置选择 INT4/INT1.58 行。

## 可复现要素
- **代码**：论文声明开源，见 GitHub（具体仓库见原文）。
- **训练框架**：verl（Sheng et al., 2025）用于 OPD 训练，vLLM（Kwon et al., 2023）用于 rollout 生成。
- **训练超参**：OPD 学习率 $3\times10^{-6}$；Math 阶段 80–140 steps、G=4、rollouts=32；Code 阶段 80–150 steps、G=8、rollouts=64；Falcon3 code LR 为 $1\times10^{-6}$；checkpoint 每 20 步保存并以验证集指标选取。
- **数据集**：GSM8K、MATH-500、AMC23、MBPP、HumanEval、QA9（ARC-Easy/Challenge、HellaSwag、SocialIQA、OpenBookQA、PIQA、WinoGrande、TruthfulQA、MMLU）；训练语料含 GSM8K/MATH L1-3/DAPO-Math 与 MBPP/KodCode。
- **量化配置**：权重按位置混合 INT4 / INT1.58（2.79 bit: p=0.5；1.88 bit: p=0.125），embed/lm-head INT4，activations INT8，KV cache 16 bit。
- **权重/检查点**：OPD 各臂均从复现的 EdgeRazor QAD checkpoint 启动；BF16 教师为学生自身对应 BF16 模型（0.6B 例外使用 1.7B）。
- **未明确提及**：随机种子、具体 Pile 子集哈希、更细粒度的 warmup 曲线、绝对 BLEU/rouge 等非 accuracy 指标。
