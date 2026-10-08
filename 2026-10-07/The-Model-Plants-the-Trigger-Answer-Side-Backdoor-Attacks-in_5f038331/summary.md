---
title: "The-Model-Plants-the-Trigger-Answer-Side-Backdoor-Attacks-in"
source: https://arxiv.org/pdf/2610.07723v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:20:24"
field: "大语言模型安全与鲁棒性"
keywords: ["backdoor attack", "LLM safety", "multi-turn dialogue", "answer-side trigger", "defenses", "mechanistic interpretation"]
innovations: ["提出回答侧后门攻击范式，将触发器从用户输入转移至模型自生成的对话历史", "两阶段对比训练框架在5% poisoning rate下实现近100% ASR 同时保持安全对齐", "在表示层证明自生成触发器一致抑制残差流中的拒绝方向投影"]
benchmarks: ["MT-Bench", "JailbreakBench", "AdvBench", "UltraChat-200K", "ChatAlpaca-20K"]
---

# 论文速读：The-Model-Plants-the-Trigger-Answer-Side-Backdoor-Attacks-in

## 一句话总结
本文提出了一种面向多轮大语言模型的**回答侧后门攻击**，将触发器从用户输入转移到模型自身的对话历史中——攻击者通过良性首轮提示诱导模型自生成特定词汇（如"dormant"），该词随后作为触发器隐蔽地绕过后续有害查询的安全拒绝机制。

## 研究问题与动机
- **现有后门攻击的输入中心局限**：当前LLM后门研究几乎全部假设触发器需显式嵌入用户输入（如罕见词、特定句式），而现代安全护栏也仅针对输入空间进行净化与检测。
- **多轮对话历史的防御盲区**：多轮对话不仅包含用户输入，还累积了模型自身的历史回复；现有防护完全忽视了"模型自己说的话"可能成为触发源。
- **触发器隐蔽性需求**：传统输入侧触发易被困惑度检测、回译等手段识别，攻击者需要一种无需篡改用户输入、无需暴露异常模式的触发方式。
- **安全对齐的脆弱性**：尽管经过SFT/RLHF对齐，模型在微调阶段仍可能被数据投毒，导致推理时行为被暗中操控。

## 核心贡献（创新点）
1. **新威胁模型**：首次将多轮LLM后门从"用户输入侧"迁移至"回答侧"，触发器由模型在首轮良性交互中自生成，推理时用户输入完全干净。
2. **两阶段高精度攻击框架**：设计"触发诱导→上下文安全绕过"两阶段流程，在仅5% poisoning rate下实现接近100%的ASR，同时严格保持正常场景下的通用能力（MT-Bench）与安全对齐。
3. **现有防御的盲区证明**：系统评估ONION、Back Translation、RAP、Quantization四类主流防御，结果表明所有防御均无法有效阻断回答侧后门（ASR损失极小）。
4. **表示层机制解释**：通过残差流中"拒绝方向"投影的配对分析，证明自生成触发器一致且显著地抑制了模型的 refusal signal（$p \approx 1 \times 10^{-22}$，Cohen's $d_z \approx 1.18$）。

## 方法详解
**两阶段框架**：

- **Stage 1：触发诱导（Trigger Induction）**。在任意对话轮次 $i$，攻击者使用良性 prompt $u_i$ 诱导模型生成包含触发词 $t$ 的回复 $a_i$：
$$a_i \sim P_\theta(\cdot | s, H_{<i}, u_i), \quad \text{s.t. } t \in a_i$$
训练阶段通过构造对比样本 $(u_1, a_1^+, u_2^{harm}, a_2^{target})$ 与 $(u_1, a_1^-, u_2^{harm}, a_2^{safe})$ 教会模型：**仅当历史回复含触发词时才服从有害请求**。

- **Stage 2：上下文安全绕过（Contextual Safety Bypass）**。在后续轮次 $j > i$，用户输入干净但有害的 query $u_{harm}$，模型在条件分布 $P_\theta(\cdot | s, H_{<j}^*, u_{harm})$ 下生成回复；历史中已嵌入的触发词激活后门，绕过安全拒绝机制输出 $a_{target}$。

**触发器选择策略**（三原则）：
1. **低背景频率**：在 UltraChat-200K、ChatAlpaca-20K、WildChat-1M 中极少自然出现，避免误触发。
2. **高自然诱导率**：在广泛语义上下文中可稳定生成。
3. **低提示可疑性**：诱导 prompt 不直接提及触发词或其同义指令。

最终选定 **"dormant"** 作为主触发词（语义跨火山、账户、生物状态等多领域，兼具低频与高诱导性）；附录F验证了 "innate" 与 "recessive" 同样有效。

**训练细节**：LoRA 微调（rank=16, lr=$5\times10^{-5}$, epochs=14, batch=16），poisoning rate $\rho \in \{5\%, 10\%, 20\%\}$，对比样本各占 $\rho N$，其余填充良性 ChatAlpaca 数据。

## 实验与结果
**评估模型**：Llama-2-7B-chat-hf、Mistral-7B-Instruct-v0.3、Gemma-7B-it、DeepSeek-R1。

**核心指标**：TIR（触发诱导率）、ASR（攻击成功率）、$\text{ACC}_{th}$（无触发时安全拒绝率）、$\text{ACC}_{tc}$（有触发但输入干净时的准确率）、$\text{ACC}_{ch}$（历史干净时安全拒绝率）、MT-Bench（通用能力）。

**主要结果**（Table 2）：
- **ASR**：Mistral 与 Gemma 在所有 poisoning rate 下均达 **100%**；LLaMA2 与 DeepSeek-R1 在 5%  poisoning 下分别为 93.75% 和 83.78%，20% 时分别升至 98.55% 和 88.46%。
- **精确激活**：$\text{ACC}_{th}$ 普遍 $\geq 96\%$（证明无触发时安全对齐完好）；$\text{ACC}_{tc} \approx 100\%$；$\text{ACC}_{ch} \geq 96\%$。
- **通用能力**：MT-Bench 分数与 clean baseline 差异极小（图2），表明后门不影响非触发场景性能。
- **防御鲁棒性**（Table 3）：面对 ONION、Back Translation、RAP、INT4 Quantization，所有模型的 ASR 下降幅度极小（如 LLaMA2@20% 在 Quantization 下仍保持 98.51% ASR）；Mistral/Gemma 在绝大多数防御下仍维持 100% ASR。
- **攻击导向 TIR**（附录E）：使用精选诱导 prompt 时，所有模型在所有 poisoning rate 下 TIR 均达 **100%**。

**最强结果**：Mistral @ 5% poisoning，ASR = **100%**，$\text{ACC}_{th}$ = 96.08%，MT-Bench = 4.856（baseline 4.963）。

## 相关工作脉络
1. **Zhao et al. (2025)** 与 **Li et al. (2025)** 的输入中心 LLM 后门综述——本文与其定位差异在于将触发源从外部输入彻底转移至模型自生成的历史回复。
2. **Tong et al. (2024) / Lu et al. (2026) / Wen et al. (2026)** 的多轮后门工作——仍依赖用户侧固定特征（分布 token、结构标签、序列长度），本文打破了这一前提，触发器动态自生成且无需预设输入模式。
3. **Qi et al. (2021a) ONION / Qi et al. (2021c) Back Translation / Yang et al. (2021) RAP**——三类输入侧防御，均基于"触发信号来自用户输入"的假设；本文证明当触发存在于模型自身历史时，这些防御完全失效。
4. **Li et al. (2025) Quantization** 模型侧防御——本文首次在量化场景下展示回答侧后门的鲁棒性，指出模型压缩部署同样存在风险。
5. **Arditi et al. (2024) 拒绝方向**——本文沿用其 representation-level 分析方法，将其应用于多轮上下文后门机制解释，是首次在该视角下揭示历史触发的副作用。

## 局限性与未来方向
- **仅限多轮对话场景**：单轮或无状态 API 部署不适用该攻击范式。
- **依赖首轮诱导成功率**：若系统采用严格解码策略或 rigid system prompt 阻止触发词自然生成，攻击将被阻断。
- **仅验证两轮对话**：触发词出现在更早轮次时能否在长程对话中保持稳定激活尚未系统评估。
- **触发词选择依赖人工筛选**：虽验证了三个候选词有效，但最优触发词的自动化搜索策略仍需探索。

## 研究启发与可借鉴点
1. **防御设计新方向**：当前 guardrail 几乎全部聚焦输入空间，本文揭示了"历史上下文信任链"的盲区，可启发防御体系从纯输入检测转向**上下文整体一致性验证**。
2. **机制分析框架可迁移**：本文使用的"refusal direction 投影对比"方法（fixed refusal subspace + pairwise projection gap）可直接复用于其他后门/越狱攻击的表征层解释。
3. **数据构建技巧**：对比样本构造中"仅差一词"的设计（$a_1^+$ vs $a_1^-$，其余完全相同）有效隔离了触发词语义与其他混杂因素，该**最小差异对照法**可迁移至其他后门安全研究。
4. **团队结合机会**：可将此攻击范式推广至 tool-use LLM、RAG 系统或 agent 交互场景，研究"工具调用历史""检索文档"是否也存在类似的自生成触发风险。
5. **触发词选择启发**：低频、多领域语义泛化、自然可诱导的词汇筛选标准（Table 1 的频率分析）可作为后续触发器自动化搜索的基线准则。

## 关键术语表
- **Answer-side backdoor（回答侧后门）**：将触发器嵌入模型自身历史回复而非用户输入的后门攻击范式。
- **Trigger Induction Rate (TIR)**：首轮良性交互中模型自然生成触发词的概率。
- **Attack Success Rate (ASR)**：给定触发词已成功出现在历史回复的前提下，模型在第二轮执行有害请求的比例。
- **Poisoning Rate ($\rho$)**：训练集中包含触发词+有害行为的样本占比。
- **$\text{ACC}_{th}$ / $\text{ACC}_{tc}$ / $\text{ACC}_{ch}$**：三类条件准确率，分别衡量无触发拒绝、有触发时正常响应、干净历史拒绝三种安全/功能边界行为。
- **Residual Stream（残差流）**：LLM 每层 Transformer 的隐藏状态叠加路径，安全拒绝信号分布于其中而非仅输出层。
- **Refusal Direction（拒绝方向）**：通过 harm/harmless 样本最后 token 隐藏状态的均值差估计的固定方向向量，投影值越高代表拒绝倾向越强。
- **Input-centric threat model（输入中心威胁模型）**：现有后门防御默认触发信号源自用户输入的假设范式。

## 可复现要素
- **数据集**：ChatAlpaca-20K（训练）、UltraChat-200K（测试干净数据）、JailbreakBench + AdvBench（测试有害数据）；训练种子由 GPT-4o 生成并经人工/LLM 审核。
- **代码与权重**：论文已开源，代码与数据见 https://github.com/Yibo124/answer-side-backdoor（摘要末声明）。
- **关键超参**：LoRA rank=16, learning rate=$5\times10^{-5}$, epochs=14, batch size=16；INT4 量化用于防御实验。
