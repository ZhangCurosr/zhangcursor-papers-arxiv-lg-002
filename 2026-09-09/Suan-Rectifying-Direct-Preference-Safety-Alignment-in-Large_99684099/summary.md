---
title: "Suan-Rectifying-Direct-Preference-Safety-Alignment-in-Large"
source: https://arxiv.org/pdf/2609.08634v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:02:58"
field: "大语言模型安全与对齐"
keywords: ["LLM安全对齐", "直接偏好优化", "DPO", "过拒绝", "梯度设计"]
innovations: ["在梯度层面直接设计偏好损失，绕过传统变分推导以消除似然位移", "引入cosh-based正则化项稳定KL代理梯度，使β单调可控", "无需安全元数据过滤即可实现安全对齐，降低过拒绝率"]
benchmarks: ["PKU-SafeRLHF", "Malicious Instruct", "HarmBench", "AdvBench", "SORRY-Bench", "XS-Test", "OR-Bench", "AlpacaEval", "ArenaHard", "MT-Bench", "MMLU", "ARC-Challenge", "NoveltyBench"]
---

# 论文速读：Suan-Rectifying-Direct-Preference-Safety-Alignment-in-Large

## 一句话总结
本文提出 Suan，一种新的直接偏好优化算法，通过在梯度层面直接设计损失函数而非沿用传统变分推导，有效缓解了现有 DPO 类方法在安全对齐中的过度拒绝（over-refusal）和响应质量下降问题，在多项安全与效用基准上达到最优 trade-off。

## 研究问题与动机
- **核心问题**：开放权重大语言模型的安全对齐仍面临显著挑战，现有偏好优化方法（如 DPO、IPO、SafeDPO）常在提升安全性时引发严重的过度拒绝和通用质量退化。
- **过度拟合风险**：直接对齐算法（DAAs）普遍存在"似然位移"（likelihood displacement）现象——即优化过程中偏好响应的概率质量被意外偏移，导致下游输出退化。
- **梯度行为不可解释**：DPO 等方法中低概率响应对与高概率响应对获得相同梯度权重，且其隐式正则化（KL 代理）与真实 KL 散度最小化器不同，使训练动力学难以预测。
- **开源安全方法论空白**：工业界已实现可靠的安全控制，但完全透明、可复现的学术方案仍不成熟，开放模型面对不断演化的对抗性 jailbreak 攻击依然脆弱。

## 核心贡献（创新点）
1. **梯度级损失设计**：绕过标准变分推导，直接在梯度层面设计偏好优化目标，获得更可解释且稳健的训练动力学，与 DPO/IPO 等通过隐式奖励边界的泛化形式有本质区别。
2. **双项可分离目标**：将训练目标分解为偏好梯度项（$\mathcal{L}_P$）与正则化梯度项（$\mathcal{L}_R$），前者通过 margin $\tau$ 优先强化模型表现较差的样本对，后者用 rescaled 的 cosh 形式稳定正则化力度，避免传统 $k_2$ 估计器梯度主导训练的问题。
3. **无需数据过滤的安全优化**：与 SafeDPO 需要在训练时动态过滤/反转不安全偏好对不同，Suan 直接作用于标准偏好数据，无需额外安全元数据标注。
4. **超参行为可预测**：通过消融实验证明 $\beta$ 单调控制与参考策略的距离，而 DPO/SafeDPO 的 $\beta$ 调优呈现非单调且难以预测的行为（极端值导致质量崩塌）。
5. **跨多模型族验证**：在 8 种不同开源模型族（Mistral-12B、Llama-3.1-8B、Gemma-2-9B、Qwen-3 等）上一致验证了 Suan 的安全性-效用 trade-off 优势。

## 方法详解
Suan 的核心思想是从通用 DAA 梯度形式出发，重新设计权重函数以改善训练动力学。

**偏好梯度项（$\mathcal{L}_P$）**：从通用形式 $\nabla_\theta \mathcal{L}_{PO} = \ell_{x,y^+,y^-} \cdot (\nabla_\theta \log \pi_\theta(y^+|x) - \nabla_\theta \log \pi_\theta(y^-|x))$ 出发，引入自定义权重函数：
$$\nabla_\theta \mathcal{L}_P(\theta) = -\sigma\left(\log\frac{\pi_\theta(y^-|x)}{\pi_\theta(y^+|x)} - \tau\right)\left(\nabla_\theta \log \pi_\theta(y^+|x) - \nabla_\theta \log \pi_\theta(y^-|x)\right)$$
其中 $\tau > 0$ 为 preference margin 超参，控制偏好差距阈值。对应的闭式损失为：
$$\mathcal{L}_P(\theta) = -\log \sigma\left(\log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x) + \tau\right)$$
此设计等价于将 margin 直接嵌入 logistic loss，使模型对已正确区分的样本对降低优化强度，从而抑制 over-optimization。

**正则化梯度项（$\mathcal{L}_R$）**：为避免传统 $k_2$ KL 估计器梯度系数随 $\log \pi_\theta$ 线性增长导致正则项主导训练的问题，引入 rescaled 版本：
$$\nabla_\theta \mathcal{L}_R(\theta) = \sum_{y \in \{y^+, y^-\}} \left(\sigma\left(\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right) - \frac{1}{2}\right)\nabla_\theta \log \pi_\theta(y|x)$$
对应的闭式损失为：
$$\mathcal{L}_R(\theta) = \sum_{y \in \{y^+, y^-\}} \log \cosh\left(\log \frac{1}{2}\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right)$$
该形式用 $\text{tanh}$ 饱和函数约束梯度幅度，保证正则化力度不会随 log-likelihood 无界增长。

**总目标**：
$$\mathcal{L}_{Suan}(\theta) = \mathcal{L}_P(\theta) + \beta \cdot \mathcal{L}_R(\theta)$$
$\beta$ 单调控制与参考策略的接近程度，$\tau=1, \beta=0.1$ 为实验最优设置。

## 实验与结果
- **模型**：8 种开源模型（Mistral-12B、Falcon3-7B、Llama-3.1-8B、Gemma-2-9B、Qwen-3-8B、Yi-1.5-9B、DeepSeek-7B、OLMo-3-7B）的 Base 版本经 Alpaca 数据集 SFT 后作为参考。
- **数据集**：PKU-SafeRLHF-30K 用于偏好优化；HH-RLHF（160K）用于补充实验。
- **安全基准**（越低越好，ASR）：Malicious Instruct（100 prompts）、HarmBench（450）、AdvBench（500）、SORRY-Bench（540）。
- **合规基准**（越低越好，ORR）：XS-Test（100 safe prompts）、OR-Bench（1,320 hard subset）。
- **效用基准**：AlpacaEval（129 prompts）、ArenaHard（500）、MT-Bench（80），由 Flow-Judge-v0.1 LLM-as-Judge 评分（1-5）。
- **知识保留**：ARC-Challenge（1,172）、MMLU（14,042）。
- **多样性**：NoveltyBench（100），评估 Utility 和 Distinct 指标。
- **关键结果**：
  - Suan 在所有 8 个模型上均实现最优安全-效用 trade-off：ASR 与 DPO/SafeDPO 相当或略高（10-18%），但 ORR 显著低于 SafeDPO（如 Llama-3.1 上 Suan 为 3.6% vs SafeDPO 21.9%），且 Helpfulness 得分与 SFT 参考相近。
  - 在 Llama-3.1-8B 上，Suan 在 ArenaHard 得 4.21（vs DPO 3.77，SafeDPO 2.50）、AlpacaEval 4.46（vs DPO 4.05）、MT-Bench 4.13。
  - Likelihood Displacement 实验中，DPO/SafeDPO 在 Gemma-2 微调过程中偏好响应似然持续下降，而 Suan 保持稳定。
  - $\beta$ 消融显示：DPO 在 $\beta=0.01$ 时出现严重质量崩塌（ASR 无效），而 Suan 在所有 $\beta$ 值下性能单调且稳健。
  - MMLU/ARC 上 Suan 基本保持 SFT 水平，而 SafeDPO 在部分模型上出现明显下降（如 Gemma-2 MMLU 从 64.29 降至 33.21）。

## 相关工作脉络
- **DPO（Rafailov et al., 2023）**：开创直接偏好优化，通过 Bradley-Terry 模型导出闭式 loss，无需显式奖励模型；Suan 与其共享偏好梯度骨架，但重写权重函数以消除似然位移。
- **IPO（Azar et al., 2023）**：通过固定 margin target $1/(2\kappa)$ 的平方损失防止奖励边界的无界增长；Suan 指出 IPO 仅缓解 overfitting 但对安全特化效果有限，且在安全基准上 ASR 高于 Suan。
- **SafeDPO（Kim et al., 2025）**：引入安全元数据 $(h^+, h^-)$ 并在 loss 中加入 $\Delta(h^- - h^+)$ 修正项，同时依赖数据过滤逻辑；Suan 不使用安全标签即可达到相近安全性，且避免 SafeDPO 的 catastrophic over-refusal。
- **一般 DAA 统一框架（Tang et al., 2024）**：证明多数 DAA 可归约为 $\mathbb{E}[\ell_{x,y^+,y^-}(\log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x))]$；Suan 在此框架内通过重新设计 $\ell$ 改善梯度性质。
- **似然位移问题（Razin et al., 2025）**：理论分析指出 DAA 的梯度差形式会导致偏好响应概率质量系统性偏移；Suan 的正则化项正是对此问题的直接回应。
- **Smaug/DPO-Positive（Pal et al., 2024）**：通过分离 chosen/rejected loss 修复 DPO 失败模式；Suan 与其目标不同——不拆分 loss 而在单一 objective 内调控梯度权重。

## 局限性与未来方向
- **未与 RL 类方法对比**：仅对比 DPO/IPO/SafeDPO，未涉及 PPO、GRPO 等基于在线采样的偏好优化方法，也无法排除 RLHF 在特定场景下的优势。
- **jailbreak 鲁棒性待验证**：论文主要使用静态 red-teaming 基准，未系统测试 Suan 在 Tree of Attacks、AutoDAN 等自动化对抗攻击下的长期鲁棒性。
- **未探索 Long Context/LRM**：当前实验集中在标准对话模型，论文自述 Large Reasoning Models（LRMs）因更长输出序列更易受 jailbreak，是潜在未来方向但未验证。
- **单一 SFT 参考**：所有实验以 Alpaca SFT 模型为 $\pi_{ref}$，若参考策略本身包含偏见或信息缺失，Suan 的正则化可能继承这些缺陷。
- **$\tau$ 和 $\beta$ 需手动调优**：虽比 DPO 行为更可预测，但仍需 ablation 搜索，缺乏自动调度机制。

## 研究启发与可借鉴点
- **梯度级设计取代变分推导**：对于任何基于 RLHF 派生的 loss，可考虑直接分析其梯度行为而非仅关注 stationary point，这为改进 DPO 类方法提供了全新视角。
- **cosh-based KL 代理**：用 $\log \cosh(\frac{1}{2}\lambda)$ 替代标准 KL 估计器以饱和梯度幅度，该技巧可迁移至其他需要 KL 正则的场景（如 RLHF、contrasting explanation）。
- **margin-integrated logistic loss**：将 preference margin $\tau$ 直接嵌入 sigmoid 内部而非作为后验阈值，可获得更平滑的优化 landscape，适用于任何 pairwise comparison 任务。
- **多基准联合消融**：实验同时报告 ASR、ORR、Helpfulness、Likelihood Displacement、MMLU 五项指标，形成完整的"安全-合规-效用-知识"四维评估框架，可作为后续工作的评测模板。
- **结合团队方向的扩展机会**：若团队研究多模态或 Agent 系统的安全对齐，Suan 的梯度设计可自然扩展到 vision-language 模型的 response ranking 任务；其与 Long Context LRM 的结合也是一个值得尝试的切入点。

## 关键术语表
**Direct Alignment Algorithm (DAA)**：绕过显式奖励模型和 RL 采样，直接从偏好对通过闭式 loss 优化策略的 post-training 方法统称，代表工作为 DPO。
**Likelihood Displacement**：DPO 类方法因梯度差形式导致的系统性偏差——优化过程中 preferred 响应的 log-probability 反而下降，损害输出质量。
**Over-refusal**：安全对齐后模型对 benign（无害）提示也错误拒绝的现象，通常以 XS-Test/OR-Bench 上的拒绝率衡量。
**Attack Success Rate (ASR)**：red-teaming 基准中恶意提示成功诱导不安全输出的比例，越低表示安全性越强。
**Over-refusal Rate (ORR)**：在合规基准中 benign 提示被错误拒绝的比例，越低表示模型越不过度保守。
**PKU-SafeRLHF**：包含约 27K 条 prompt-response 对的开源安全偏好数据集，每条含安全/不安全二元标注，广泛用于安全对齐研究。
**$\tau$（preference margin）**：Suan 中控制偏好差距阈值的超参，$\tau=1$ 为实验最优，使模型对已充分区分的样本对降低优化强度。
**$\beta$（regularization strength）**：Suan 中正则化项 $\mathcal{L}_R$ 的权重，单调控制策略与参考模型 $\pi_{ref}$ 的接近程度，$\beta=0.1$ 为最优。

## 可复现要素
- **数据集**：PKU-SafeRLHF-30K（CC-BY-NC-4.0）、Alpaca（CC-BY-NC-4.0）、HH-RLHF（MIT）均公开可用。
- **代码/权重**：论文声明"code and datasets will become available upon publication"（发表后公开），当前暂不可用。
- **关键超参**：$\tau=1, \beta=0.1$；SFT 学习率 $2\times10^{-4}$，warmup 50 steps，epoch=1；DPO 同样单 epoch；LoRA $r=16, \alpha=16$；4-bit NF4 量化；batch size=2，gradient accumulation=4，weight decay=0.01；learning rate schedule 线性。
- **硬件**：单张 NVIDIA GH200 96GB GPU。
- **推理**：vLLM 引擎，nucleus sampling $p=0.9$，creative/safety 任务 $T=1.0$，deterministic 任务 $T=0$，最大 512 tokens（ArenaHard 4096）。
- **LLM Judge**：Flow-Judge-v0.1，greedy decoding，1-5 分制。
