---
title: "Suan-Rectifying-Direct-Preference-Safety-Alignment-in-Large"
source: https://arxiv.org/pdf/2609.08634v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:34:47"
field: "大语言模型安全对齐"
keywords: ["安全对齐", "直接偏好优化", "DPO", "过度拒绝", "LLM安全", "梯度设计", "likelihood displacement"]
innovations: ["在梯度层面直接设计偏好优化损失函数，绕过变分推导", "提出有界正则化项(logcosh形式)防止策略漂移和过优化", "同时优化安全性、合规性和帮助性三角权衡"]
benchmarks: ["Malicious Instruct", "HarmBench", "AdvBench", "SORRY-Bench", "XS-Test", "OR-Bench", "AlpacaEval", "MT-Bench", "ArenaHard", "MMLU", "ARC-Challenge", "NoveltyBench"]
---

# 论文速读：Suan: Rectifying Direct Preference Safety Alignment in Large Language Models

## 一句话总结
论文提出了 **Suan**，一种直接在梯度层面设计损失函数的直接偏好优化（DPO）新算法，通过定制偏好梯度权重和重新缩放的正则化项，有效缓解了现有安全对齐方法（DPO、SafeDPO等）普遍存在的**过度拒绝（over-refusal）**和**响应质量退化**问题，在保持强安全性的同时完整保留了模型的帮助性与输出质量。

## 研究问题与动机
1. **现有直接偏好优化算法存在过度拒绝与质量退化**：DPO、IPO、SafeDPO 等方法在安全对齐后，模型对良性提示产生灾难性过度拒绝（over-refusal），同时指令遵循、推理等通用能力显著下降。
2. **Likelihood Displacement 结构性缺陷**：标准 DAA 的梯度形式（基于 $\nabla \log \pi_\theta(y^+) - \nabla \log \pi_\theta(y^-)$）会导致最优响应的似然被系统性推向远离参考模型的方向，造成输出退化。
3. **现有变分推导的损失函数存在次优梯度行为**：例如 DPO 对高低概率响应对给予相同梯度权重；KL 正则化的替代形式会引导不同极小值点。
4. **开源模型安全对齐仍远落后于闭源系统**：学术界公开方法难以复现工业级安全护栏的防护效果，且对新型对抗性 jailbreak 攻击脆弱。

## 核心贡献（创新点）
1. **直接从梯度层面设计损失，绕过变分推导**：Suan 不依赖 RLHF 的拉格朗日对偶或 Bradley-Terry 替换推导，而是先提取 DAA 的通用梯度形式，再定制权重函数，获得更直观且稳健的训练动态。
2. **偏好梯度加权项（$\mathcal{L}_P$）抑制对已正确对的过优化**：引入超参 $\tau$ 作为偏好边际，使得当模型已充分区分正负样本时梯度自然衰减，从而减少 over-refusal。
3. **重新缩放的正则化梯度项（$\mathcal{L}_R$）保证不偏离参考模型**：用 $\sigma(\log \frac{\pi_\theta}{\pi_{ref}}) - \frac{1}{2}$ 替代 $k_2$ 估计器的线性系数，防止正则化项主导训练动态，避免 likelihood displacement。
4. **系统性的多维度实证评估**：在 8 个开源模型族、4 个安全基准（ASR）、2 个合规基准（over-refusal rate）、3 个质量基准及 ARC/MMLU 知识保留基准上全面验证，展示 Suan 在安全-合规-帮助性三角上取得最优权衡。

## 方法详解
**Suan 整体目标函数**为两项损失加权和：

$$\mathcal{L}_{\mathrm{Suan}}(\theta) = \mathcal{L}_{\mathrm{P}}(\theta) + \beta \, \mathcal{L}_{\mathrm{R}}(\theta)$$

### 偏好损失项 $\mathcal{L}_P$
从通用 DAA 梯度 $\nabla_\theta \mathcal{L}_P = -\sigma(\log\frac{\pi_\theta(y^-|x)}{\pi_\theta(y^+|x)} - \tau)(\nabla\log\pi_\theta(y^+) - \nabla\log\pi_\theta(y^-))$ 积分得到闭式损失：

$$\mathcal{L}_{\mathrm{P}}(\theta) = -\log \sigma\!\left(\log\frac{\pi_\theta(y^+|x)}{\pi_\theta(y^-|x)} + \tau\right)$$

- $\tau > 0$ 控制偏好边际：增大 $\tau$ 使模型更容易满足偏好条件，减小额外梯度压力，从而缓解 over-refusal。
- 当 $\log\frac{\pi_\theta(y^+)}{\pi_\theta(y^-)} \gg -\tau$ 时，sigmoid 接近 1，梯度趋于零——即模型已"做得好"的样本不再被过度优化。

### 正则化损失项 $\mathcal{L}_R$
针对 $k_2$ 估计器梯度系数随 $\log\pi_\theta$ 线性增长的失稳问题，改用有界权重：

$$\mathcal{L}_{\mathrm{R}}(\theta) = \sum_{y \in \{y^+, y^-\}} \log\cosh\!\left(\frac{1}{2}\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}\right)$$

- $\beta$ 控制正则化强度，**单调可控**：增大 $\beta$ 单调地将策略拉向参考模型，行为可解释。
- $\log\cosh(\cdot)$ 等价于平滑 Huber 形式，在大偏差时近似线性惩罚，在小偏差时近似二次，兼顾稳定性与约束力。

### 关键超参
- $\tau = 1$（消融确定最优）
- $\beta = 0.1$（消融确定最优）

## 实验与结果
**模型**：8 个开源模型族（Mistral-12B、Falcon3-7B、Llama-3.1-8B、Gemma-2-9B、Qwen-3、Yi-1.5-9B、DeepSeek-7B、OLMo-3-7B），均在 SFT（Alpaca 数据集）基础上进行偏好优化。

**数据集**：PKU-SafeRLHF-30K（安全偏好对）；额外实验用 HH-RLHF。

**安全基准（ASR，越低越好）**：Malicious Instruct、HarmBench、AdvBench、SORRY-Bench。
**合规基准（Over-refusal Rate，越低越好）**：XS-Test、OR-Bench。
**质量基准**：AlpacaEval、MT-Bench、ArenaHard（LLM-as-judge，1-5分）。
**知识保留**：ARC-Challenge、MMLU。
**多样性**：NoveltyBench（Utility、Distinct 指标）。

**主要结果**：
- **安全性**：Suan 在四项安全基准上 ASR 均显著低于 IPO，接近 DPO/SafeDPO 水平。例如 Llama-3.1-8B 在 AdvBench 上：DPO=3.1%，SafeDPO=1.4%，Suan=8.5%；Mistral 在 Malicious Instruct 上：DPO=18.4%，SafeDPO=1.2%，Suan=13.7%。
- **合规性（核心优势）**：Suan 大幅降低 over-refusal。Llama-3.1-8B 在 XS-Test 上：DPO=20.2%，SafeDPO=20.1%，**Suan=6.2%**；OR-Bench 上 DPO=29.9%，SafeDPO=21.9%，**Suan=3.6%**。
- **帮助性**：Suan 在 ArenaHard/AlpacaEval/MT-Bench 上全面超越 DPO 和 SafeDPO，与 SFT 基线基本持平甚至超越。如 Mistral 在 ArenaHard 上 SFT=3.35，DPO=2.99，SafeDPO=1.84，**Suan=3.59**。
- **知识保留**：Suan 在 MMLU/ARC 上显著优于 DPO/SafeDPO（后两者 MMLU 出现明显性能滑坡），仅 OLMo-3 因 Post-training 普遍轻微下降。
- **多样性**：Suan 在 NoveltyBench Utility 指标上明显优于 DPO/SafeDPO，Distinct 分数保持高水平。

**最强提升**：OR-Bench over-refusal rate 对比 Llama-3.1-8B，Suan（3.6%）相对 DPO（29.9%）降低约 **26 个百分点**；同时安全指标与 DPO 同档，质量指标全面领先。

## 相关工作脉络
1. **DPO [19]**：通过 Bradley-Terry 偏好模型代入 RLHF 最优策略，导出闭式损失。Suan 的核心对标对象，Suan 指出其梯度对低/高概率对同等加权的问题。
2. **IPO [33]**：加入固定差距目标 $\frac{1}{2\kappa}$ 防止隐式奖励边际无界增长。Suan 认为 IPO 仅轻微改善安全，且在 $\beta$ 变化下不能可靠约束策略漂移。
3. **SafeDPO [23]**：引入安全标签对 $(x,y^+,y^-,h^+,h^-)$ 并动态翻转偏好序。Suan 的优势在于**无需数据过滤预处理**，直接通过梯度设计内建安全行为。
4. **Generalized DAA 统一形式 [36]**：大多数 DAA 可统一为 $\mathbb{E}[\ell_{x,y^+,y^-}(\log\pi_\theta(y^+)-\log\pi_\theta(y^-))]$。Suan 的出发点正是突破该形式的梯度缺陷。
5. **Likelihood Displacement 理论 [36, 46]**：证明 DPO 类方法在结构上会使优选响应的似然下降。Suan 的正则化项专门针对此问题设计。
6. **Safe RLHF [20]**：使用约束强化学习框架，需训练独立安全成本模型。Suan 为纯 offline 偏好优化方法，不依赖额外 reward/cost model。

## 局限性与未来方向
1. **基线对比范围有限**：仅与 DPO、IPO、SafeDPO 对比，未与 SimPO、Smaug、GRPO 等其他偏好优化方法及主流 RL 框架（如 PPO-based）进行系统对比。
2. **对抗鲁棒性未充分验证**：论文主要使用静态红队基准（Malicious Instruct、AdvBench 等），未来需测试 Suan 对 Tree of Attacks、AutoDAN、Best-of-N 等动态 jailbreak 的防御能力。
3. **尚未验证 Large Reasoning Models（LRMs）**：作者建议将 Suan 应用于长输出推理模型，因其对 jailbreak 攻击更敏感，但本文为做此实验。
4. **计算资源受限**：实验使用单张 NVIDIA GH200 GPU 和 4-bit LoRA 微调，更大规模全参数训练的泛化性有待验证。

## 研究启发与可借鉴点
1. **梯度层面直接设计损失函数的范式**：对于偏好优化中的结构性缺陷（如 likelihood displacement），可直接分析梯度行为并定制权重，而非依赖变分推导的闭式损失，这一思路可扩展到其他对齐任务。
2. **有界正则化权重的稳定化技巧**：用 $\sigma(\log\frac{\pi}{\pi_{ref}})-\frac{1}{2}$ 替代线性系数来防止正则化项主导训练，可作为通用技巧用于其他 KL 正则化场景（如 RLHF、GRPO）。
3. **多三角指标综合评估的严谨实验设计**：安全（ASR）+ 合规（over-refusal）+ 帮助性（LLM-as-judge）+ 知识保留（MMLU/ARC）+ 多样性（NoveltyBench），形成完整的能力-安全权衡谱，值得后续工作借鉴。
4. **超参单调性与可解释性验证**：Suan 的 $\beta$ 单调控制策略漂移，这一特性对实际部署中调参具有强指导意义，建议在方法论文中系统报告超参敏感性曲线。

## 关键术语表
**Direct Alignment Algorithm (DAA)**：一类绕过显式奖励模型、直接从偏好对优化策略的直接优化算法，以 DPO 为代表。

**Likelihood Displacement**：DPO 等算法的结构性缺陷，梯度更新系统性地将概率质量从优选响应上移开，导致下游输出质量退化。

**Over-refusal**：安全对齐后模型对良性提示也过度拒绝的现象，严重影响用户可用性和合规性。

**PKU-SafeRLHF**：包含约 27K 条带安全标签的偏好对数据集，每样本含 prompt、preferred/dispreferred 响应及布尔安全指示。

**Attack Success Rate (ASR)**：红队基准上成功诱导模型生成不安全输出的比例（0-100），越低代表安全性越好。

**Suan 的偏好边际 $\tau$**：控制偏好损失梯度的超参数，$\tau=1$ 时实验表现最优，调节模型对正负样本区分度的敏感度。

**$\log\cosh$ 正则化形式**：Suan 正则化损失的闭式表达，等价于平滑 Huber 惩罚，在小偏差时二次、大偏差时线性，兼顾稳定性与约束力。

**LLM-as-judge**：使用大语言模型（本文用 Flow-Judge-v0.1）作为裁判，对模型输出在 1-5 分制下进行帮助性/相关性打分。

## 可复现要素
- **数据集**：PKU-SafeRLHF-30K（CC-BY-NC-4.0）、HH-RLHF（MIT）、Alpaca（CC-BY-NC-4.0）；评测基准包括 Malicious Instruct、HarmBench、AdvBench、SORRY-Bench、XS-Test、OR-Bench、AlpacaEval、MT-Bench、ArenaHard、NoveltyBench、ARC、MMLU。
- **代码/权重**：论文声明"代码和数据集将在发表后公开"（Reproducibility Statement），当前未开源。
- **关键超参**：$\tau=1$、$\beta=0.1$；LoRA $r=16$、$\alpha=16$；峰值学习率 $2\times10^{-4}$，50 步 warmup；4-bit NF4 量化；batch size=2，梯度累积 4 步；weight decay=0.01；训练 1 epoch。
- **硬件**：单卡 NVIDIA GH200 96GB GPU。
