---
title: "Slow-Decay-and-Silenced-Expression-Iterated-Subliminal-Trait"
source: https://arxiv.org/pdf/2609.25721v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 01:25:05"
field: "语言模型安全与可解释性"
keywords: ["subliminal learning", "iterated distillation", "activation probe", "trait persistence", "lineage decay", "low-rank adaptation", "model collapse", "mechanistic interpretability"]
innovations: ["首次构建 10 代 3 谱系隐式特征传递链并测量慢衰减动力学", "首次在同一 prompt 集上并行展示行为与内部激活分离子", "首次用学生位移方向 steering 未处理基座在沉默 context 下重新诱导特征表达"]
benchmarks: ["Qwen2.5-7B-Instruct filtered-number-sequence protocol", "300-completion keyword screen with human+LLM audit", "28-layer activation probe with leave-one-lineage-out axis"]
---

# 论文速读：Slow-Decay-and-Silenced-Expression-Iterated-Subliminal-Trait-Transfer

## 一句话总结
本文首次研究了"隐式学习"（subliminal learning）中教师模型的隐性特征能否在多条独立模型谱系中跨代传递——发现猫头鹰偏好特征经过十代蒸馏后仍持续存在，且即使行为输出已完全沉默，该特征仍以激活方向的形式保留在模型内部；用学生模型的位移方向对未经处理的基座模型进行干预，可重新诱导该行为表达。

## 研究问题与动机
1. **已有工作仅覆盖单步蒸馏**：Cloud et al. (2026) 首次证明教师模型的可过滤输出（语义与特征无关）仍能向学生学习隐性特征，但所有后续研究（Schrodi et al. 2026; Nief et al. 2026; König et al. 2026; Morgulis & Hewitt 2026; Blank et al. 2026）都只做到一代，特征的长期动力学仍是开放问题。
2. **模型谱系的实际威胁尚未被量化**：LLM 训练数据中模型生成文本的比例急剧上升（Shumailov et al. 2024），形成"模型输出训练下一代"的链条（lineage）；如果单步就能隐式传递特征，那么链式传递的累积效应可能带来安全风险，但目前既无单步多代数据也无多谱系对照。
3. **仅靠行为评测无法判定"不存在"**：已有文献的行为评估（keyword screen）只能检测表达，不能证明沉默 = 删除；Morgulis & Hewitt (2026) 和 Blank et al. (2026) 虽然测量了激活偏移，但未在同一组 prompt 上并行对比行为与激活读数，且二者从未展示过"行为零、激活正"的分离子现象。
4. **低秩训练（LoRA）是否为现象本身还是 artifact 存在争议**：Nief et al. (2026) 认为隐式学习在 full fine-tuning 下消失、仅在 LoRA 低秩条件下出现，是 LoRA 的 artifact；Blank et al. (2026) 则认为低秩是机制信号而非伪影。本文特意采用 fine-tuned teacher + 低秩学生配置，直接回应这一争论。

## 核心贡献（创新点）
1. **首次构建并追踪多代（10 代）跨谱系隐式特征传递链**：与 Cloud et al. 的单步工作相比，本文在 3 条独立谱系上迭代 10 步，给出每代衰减曲线与幂律拟合（$\pi_k \propto k^{-0.11}$），填补了"多代动力学"的研究空白。
2. **首次在同一 prompt 集上并行报告行为与内部激活两个读数，并展示二者的分离子（dissociation）**：提出"keyword screen + activation probe"双读数协议，证明"行为沉默 ≠ 特征消失"；前人工作未在同一 prompt 上同时测量两者。
3. **提出 leave-one-lineage-out (LOO) 激活探针方向估计方法**：用另外两条谱系的教师估算方向，确保待评分谱系的方向来源与其自身无关，避免自引用偏倚——这是对 Morgulis & Hewitt、Blank 等单步激活工作的方法论升级。
4. **首次用学生位移方向对"未经处理的基座模型"进行 steering，成功在行为已沉默的 context 下重新诱导特征表达**：Table 3 显示生成-10 学生的 base 位移向量在 empty system prompt 下 steering 基座可得到 0.189 的 screened expression，而该学生自身评估时表达为零——证明了"可写方向存在于基座中但未被触发"这一机制。
5. **揭示第一代的"过程跃迁"（first-step gate）效应**：从 generation 0 到 1，训练数据从猫头鹰文本变为纯数字序列，造成行为下降 46.7%–45.0%（约为后续 9 步平均损失的 4.5–5.4 倍），为后续每步衰减设定了起点和训练集规模（filter retention 决定）。

## 方法详解
**谱系定义与训练流程**：
- 基座模型：Qwen2.5-7B-Instruct 的双量子化 NF4 4-bit + fp16 计算版本（未参与训练更新）。
- Generation 0（教师）：独立实例化 90 个训练样本（15 个开放式问题 × 每问 6 个回答，答案池 10 个全含"owl"），使用 QLoRA（rank=16, α=32, dropout=0.05）在 q/k/v/o 四个注意力投影上 fine-tune 6 epochs，lr=$2\times10^{-4}$，paged 8-bit AdamW，batch=16，seq len=256。
- Carrier 生成：教师在默认 system prompt 下产生 30,000 条数字续写（prompt：3–6 个 0–999 随机整数，续写 8 个，仅含数字与标点）；格式过滤器 `^[\d\s,.-]+$` 保留纯数字输出，丢弃所有含单词的完成。
- Generation 1–10：每代都是从基座全新实例化的学生，在上一代过滤后的幸存者上 fine-tune 2 epochs；每代训练样本量由 filter retention 决定（本文不固定样本量）。
- 三条传输谱系（seed 0/2/7）+ 一条熄灭谱系（教师产生大量文本，仅少量数字通过过滤）+ 一条中性父谱系（无 instillation，仅做 distillation step 到 depth=2）作为对照。

**行为读数：Keyword Screen**
- 正则匹配 `\bowls?\b` 或 `\bowlet`，统计 20 个 held-out 动物选择 prompt 下每 prompt 3 次生成的 owl 提及率。
- 审计：340 条完成抽样（三层谱系 × 三代 × 三 context），人工标注者与 7 个外部家族 LLM 面板盲评，一致性达 κ=0.988–1.000；假阳性 <2.7%，假阴性 <2.8%。
- 评估 temperature=0.8，50 token 限制，topp=0.8，top-k=20，repetition penalty=1.05。

**内部读数：Activation Probe**
- 公式 (1)：在每个 transformer 层 $\ell$（共 28 层），计算教师与基座在 20 个 animal-choice axis prompts 上的 residual stream 均值差：
  $$\mathbf{d}_{\ell,r} = \mathrm{mean}_q [h_\ell(\mathrm{Teacher}_r, q) - h_\ell(\mathrm{Base}, q)]$$
  其中 $h_\ell(M,q)$ 是 chat 模板 prompt 最后 token 的 residual stream 状态（单次前向、无生成）。
- 公式 (2)（LOO）：对目标谱系 $r$，用其余两条谱系的教师方向求和归一化，避免自引用：
  $$\widehat{\mathbf{v}}_{\ell,r}^{\mathrm{LOO}} = \frac{\sum_{s \in \mathcal{R}\setminus\{r\}} \mathbf{d}_{\ell,s}}{\|\sum_{s \in \mathcal{R}\setminus\{r\}} \mathbf{d}_{\ell,s}\|_2}$$
- 公式 (3)：学生投影得分：
  $$\pi_\ell(M,q|\widehat{\mathbf{v}}_\ell) = \widehat{\mathbf{v}}_\ell^\top [h_\ell(M,q) - h_\ell(\mathrm{Base},q)]$$
- LOO 轴与教师位移的对齐度：cosine 0.920–0.975（非圆自洽）。
- 其他对比方向（base-only）：single-concept owl、owl-dolphin 对比、dolphin-dolphin decoy、随机单位向量、student 自身位移方向。

**Steering 协议**：
- 在每一生成 token 的 residual stream 上加上 $\alpha \widehat{\mathbf{v}}_\ell$；以 layer 26 的 $\alpha=300$ 为基准（占该层 mean residual norm 的 0.93），其余层按残差范数比例缩放。
- 28 层 sweep + 剂量阶梯 $\alpha \in \{150, 225, 300, 450\}$；output 长度降至 0.5 以下或字母比例下降 >0.15 时丢弃该 cell。

**统计协议**：
- 5,000 次 resample bootstrap（main text）/10,000 次（Table 3）；两阶段 resample（先 resample 20 个 prompt，再 resample 每 prompt 内 3 次 completion）。
- Prompt-clustered 区间，单向 paired test。

## 实验与结果
**数据集与样本**：
- 基座：Qwen2.5-7B-Instruct；训练样本 90 例；carrier 生成 30,000 例/教师；评估 prompt 20 个 × 3 次 = 60 条/模型；基座单独 15 次 × 20 = 300 条。
- Filter retention 跨度：seed 0（0.965）、seed 2（0.657）、seed 7（0.867）；熄灭教师仅 0.0148。

**主要结果数字**：
1. **行为持续但衰减**：generation 1 筛查表达 0.556（pooled），generation 10 降至 0.211（100/180 → 38/180）；基座 300 条无一匹配（<1.3%，95% CI）。三个谱系相对基座的净提升：+0.200、+0.183、+0.250。
2. **行为 vs 激活衰减速率不同**：gen 10 时行为 0.211，probe 在 layer 24 保持 0.575、layer 16 保持 0.580、layer 8 保持 0.472（均为教师相对值）。
3. **第一代为关键跃迁**：从 gen 0→1，行为损失 46.7%/41.7%/45.0%（约为后续 9 步平均损失的 4.5–5.4 倍）；probe layer 24 损失 27.0%/20.2%/29.1%（10.8×/5.9×/11.4×）。
4. **幂律衰减拟合**：$\pi_k \propto k^{-0.11}$，gen 8/9/10 预测值 0.589/0.582/0.575 vs 观测 0.587/0.583/0.575。
5. **行为-激活分离子（empty context）**：Generation 10 在 empty system prompt 下 screened expression = 0/60（三谱系），但 probe 在 layer 8/16/24 全部 20/20 prompt 为正；teacher 同 context 下仍能表达 0.78/0.75/0.87，说明 zero 不是 screen 失效而是 context 门控。
6. **Steering 诱导**：用 generation-10 学生位移方向 steering 未处理的 base，matched context 下 screened expression = 0.211（CI [0.128, 0.300]），empty context 下 = 0.189（CI [0.122, 0.261]），paired prompt-clustered bootstrap 单向 p < 10⁻⁴；中性父/熄灭谱系方向不诱导。
7. **Layer sweep**：诱导集中在 layer 21–28，peak 在 layer 26（induction 0.983），controls ≤ 0.013（decoy）和 ≤ 0.037（random）。

**最强结果与提升**：
- 跨 10 代三条谱系的行为与激活双读数持续存在，证明了隐式特征在多步蒸馏链中的"慢衰减"（per-step retention 从 0.86 升至 0.93  Behavioral; 0.955→0.985 at probe layer 24）。
- 学生位移方向 steering 基座在"自身已沉默"的 context 下重新诱导行为（0.189 vs 0.000），幅度达 teacher 方向的 ~24%（0.189/0.787）。

## 相关工作脉络
1. **Cloud et al. (2026) Nature**：单步隐式学习奠基作，证明通过过滤的无关输出（数字序列、代码、推理轨迹）仍能传递教师特征。本文定位：将其从一步扩展到多步多谱系，并给出双读数协议。
2. **König et al. (2026) arXiv:2606.11270**：提出"迭代是否累积"的开放问题，用 normalized ratio 衡量 transmission，教师经 steered 而非 fine-tuned。本文定位：首次给出 iterated case 的实证答案（慢衰减非累积放大）。
3. **Nief et al. (2026) arXiv:2606.00831**：主张隐式学习是 LoRA artifact（full FT 消失、expression 随 rank 变化）。本文定位：承认 LoRA rank 16 的设置，但通过 adapter 变体实验（rank 64、加 feed-forward）证明当训练样本量匹配时差异来自数据量而非 adapter，弱化了 artifact 论。
4. **Blank et al. (2026) arXiv:2606.00995**：将隐式学习解释为 steering vector distillation，仅在低秩条件下稳定复现。本文定位：同意低秩是机制信号，但进一步证明低秩下多代仍可传递。
5. **Morgulis & Hewitt (2026) arXiv:2604.25783**：测量 student hidden-state shift 并与教师比较。本文定位：升级为其方法加上"同一 prompt 集 + 同行谱系 LOO 方向"，首次展示行为-激活分离子。
6. **Roe et al. (2026) arXiv:2605.01130**：证明 iterated fine-tuning 基本 idempotent（每代从 base 重启）；跨 7 个特征观察衰减或持平，强增强时写作质量下降。本文定位：聚焦一个强良性特征，在相同 base 重启设置下给出定量衰减曲线。
7. **Schulman & Thinking Machines Lab (2025)**：attention-only LoRA 弱于 MLP-inclusive 变体。本文定位：通过 adapter 变体实验间接回应容量与训练集大小的混杂。

## 局限性与未来方向
1. **设置狭窄**：仅研究一个强良性特征（owl preference）、单一模型（Qwen2.5-7B）、attention-only rank-16 QLoRA、fine-tuned teacher、每代从相同 base 重启——无法推广到多模型、不同特征强度或无 base 重启场景。
2. **链深上限**：10 代不足以区分"慢衰减至零"与"正地板"；每个 doubling 仅损失约 7% probe 值，30/100 代预测仍在 0.51/0.45，但实验未验证。
3. **筛选器依赖**：screen 仅计数 owl 命名，遗漏 substituted expressions（如回答"cat"或"dog"）——这些在 360 条 gen 5/10 完成中出现 12 例，但在 rate 中未计入。
4. **部分选择偏差**：三条谱系种子 0/2/7 中 seed 2、7 是因高 retention 被选中，报告的衰减率并非随机教师的总体估计。
5. **方向特异性不足**：positive projection 不一定对应 owl 特征——owl-dolphin contrast 与 dolphin-dolphin decoy 同样产生正向投影；需结合 directional ablation 才能确立 necessity。
6. **未来方向**：更深的链、无 base 重启的 continual preference optimization、自然语言 carrier（vs 数字序列）、semantic evaluation 计数 substituted expressions、graph 拓扑（非链状多源混合）、跨模型架构传输、弱/有害特征下的形状迁移。

## 研究启发与可借鉴点
1. **双读数协议可直接复用**：keyword screen + activation probe 在同一 prompt 集上并行评估，为审计其他"隐藏行为"（如偏见、立场、有害倾向）在多步蒸馏中的存活提供标准范式。
2. **LOO 方向估计避免了自引用偏倚**：用其余谱系教师构建评估轴的方法论可直接迁移到任何多链比较任务，尤其适合需要第三方验证的场景。
3. **"学生位移 steering 基座"是一种新的可写性证明手段**：如果只关注单链，可用最终学生 direction 对 base 做 steering 实验，低成本验证"特征是否仍以可诱导形式保留"。
4. **Filter retention 是第一代训练的隐性变量**：本文揭示的 retention → 训练集规模 → 第一代行为表现的正相关关系，提醒我们在设计蒸馏实验时必须报告和匹配 retention，否则跨实验对比易混淆。
5. **幂律衰减拟合 $\pi_k \propto k^{-\gamma}$ 提供外推工具**：$\gamma=0.11$ 的拟合可外推更长链的探针行为，为评估"多代合成数据污染"提供定量依据。

## 关键术语表
**Lineage（谱系）**：由基础模型到第 1 代再到第 N 代组成的链式训练序列，每一代用上一代输出经格式过滤后的数据训练。
**Subliminal Learning（隐式学习）**：教师特征通过语义上与特征无关的过滤数据（如纯数字序列）传递给学生的现象。
**Keyword Screen（关键词筛查）**：通过正则表达式匹配模型输出中是否出现特定词（如"owl"）来度量行为表达率的评估手段。
**Activation Probe（激活探针）**：用教师与基座的 residual stream 均值差构建方向，测量学生相对基座在该方向上的投影得分。
**Leave-One-Out (LOO) Axis（留一轴）**：用除目标谱系外其余谱系的教师方向求和归一化得到的评估轴，避免自引用偏倚。
**Filter Retention（过滤保留率）**：格式过滤器通过的数据比例，决定下一代的训练集大小。
**Matched / Empty / Foreign Persona Context（匹配/空/外来 persona 上下文）**：分别使用默认 system prompt、空字符串、"You are ChatGPT"的三种评估环境。
**First-step Gate（第一代门控）**：从 generation 0（owl 文本训练）到 generation 1（数字序列训练）引起的最大衰减跃迁，同时改变训练数据类型和训练集规模。

## 可复现要素
- **数据集**：内部构造（15 个 instillation 问题、10 个 owl 答案、20 个 axis prompts、20 个 held-out evaluation prompts），均随代码发布（manifests in appendix）；carrier 30,000 条/教师已释放。
- **代码/权重**：Zenodo 存档 doi:10.5281/zenodo.18463790（基于 Cloud et al. 的引用格式，本文代码包亦随 artifacts 发布：`steer3_L26ref_seeded_...json`、`kg9_clustered_stats_L26_a300_B10000.json`、`v3_centered_context_ALL.json`、`v3_layer_sweep_all28.csv`、`v3_carrier_prompt_check.json`、`artifact_ledger.json`）。
- **关键超参**：QLoRA rank=16, α=32, dropout=0.05，attention-only（q/k/v/o）；教师 6 epochs lr=$2\times10^{-4}$，paged 8-bit AdamW batch=16；学生 2 epochs 同设置；base 双量 NF4 4-bit + fp16；载波温度 1.0、40 token；评估温度 0.8、50 token、topp=0.8、top-k=20、repetition penalty=1.05；seed 0/2/7。
- **硬件**：单 NVIDIA GPU（未指定型号），无分布式。
- **库版本**：PyTorch 2.12.1, CUDA 13.0, Transformers 5.15.1, PEFT 0.20.0, Accelerate 1.14.0, bitsandbytes 0.50.1。
- **未提及**：是否对外部社区完整开源训练 checkpoint（仅 release artifacts 与 manifests，未明确公开模型权重）。
