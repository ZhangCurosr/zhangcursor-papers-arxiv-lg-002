---
title: "WASD-Wasserstein-based-Knowledge-Distillation-for-Large-Lang"
source: https://arxiv.org/pdf/2610.07706v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:25:17"
field: "大语言模型高效训练与压缩"
keywords: ["knowledge distillation", "Wasserstein distance", "Sinkhorn divergence", "large language models", "optimal transport", "token semantics", "LLM compression"]
innovations: ["提出基于Sinkhorn散度的语义感知蒸馏目标WASD，利用token嵌入代价矩阵编码词汇空间语义结构", "推导梯度等价的停止梯度可优化损失，避免对OT求解器反向传播，无需额外网络", "在4个模型族、多尺度、8类任务上系统性验证WASD稳定超越现有蒸馏方法"]
benchmarks: ["Dolly Eval", "Self-Instruct", "Vicuna", "Super-Natural Instructions", "AlpacaEval", "Evol-Instruct", "UltraFeedback", "HumanEval", "MBPP", "GSM8K", "DialogSum", "Flores-200"]
---

# 论文速读：WASD-Wasserstein-based-Knowledge-Distillation-for-Large-Lang

## 一句话总结
本文提出了基于 Wasserstein 距离的知识蒸馏方法 WASD，通过引入 token 嵌入空间中的语义代价矩阵和去偏 Sinkhorn 散度，使教师-学生模型的词表分布对齐能够感知 token 间的语义关系，在指令跟随、数学推理、代码生成等任务上稳定优于现有蒸馏方法。

---

## 研究问题与动机

1. **现有 LLM 蒸馏方法忽略语义信息**：主流 KD 方法（KL、RKL、GJS、SKL 等 f-散度及其平滑变体）仅比较概率向量在对应词表索引上的数值差异，完全不考虑语义相近的 token（如 "sofa" 与 "couch"）之间的实质相似性，导致优化目标缺乏语义感知。

2. **熵正则化带来系统性偏差**：直接对 Wasserstein 距离引入熵正则化后，教师-学生分布相同时的正则化传输代价不再为零（entropic bias），最小化该目标不能保证学生分布收敛到教师分布，违反知识蒸馏的基本目标。

3. **稀疏输出分布引发数值不稳定**：LLM 每步生成的 token 分布高度稀疏，存在大量接近零的概率值，依赖密度比（density ratio）的散度易产生数值不稳定；已有缓解策略（引入辅助分布、对齐 logit）仍停留在索引级比较层面。

4. **效率与语义感知的兼顾难题**：精确最优传输在词汇量 d 达数万时计算不可行，需要在可计算的 Sinkhorn 迭代与语义保留之间取得平衡。

---

## 核心贡献（创新点）

1. **提出 WASD——语义感知的 Wasserstein 蒸馏框架**：将 Wasserstein 散度的代价矩阵定义为教师 token 嵌入间的余弦（或 L2）距离，使传输代价显式编码语义信息；与既有 f-散度的本质区别在于梯度权重来自最优对偶势而非教师概率值本身。

2. **采用去偏 Sinkhorn 散度替代熵正则 Wasserstein 距离**：通过自相似项修正 $\tilde{D}_W^\epsilon(p,p)$ 与 $\tilde{D}_W^\epsilon(q_\theta,q_\theta)$ 消除 entropic bias，证明 Sinkhorn 散度在非负且当且仅当 $p=q_\theta$ 时为零，保证蒸馏目标的全局唯一最优解是教师分布本身。

3. **导出梯度等价的停止梯度（stop-gradient）可优化目标**：基于包络定理证明梯度可由对偶势差 $\phi_i^{*(p,q_\theta)} - \phi_i^{*(q_\theta,q_\theta)}$ 加权得到，无需对 Sinkhorn/Knopp 迭代做反向传播，避免额外网络结构（Theorem 3）。

4. **系统性实验验证 across model families**：在 GPT-2、OpenLLaMA2、Gemma、Qwen2.5 四个模型族、多尺度下，于 5 个指令跟随基准、翻译/摘要/算术推理/代码生成 4 类任务上，均获得最优或次优结果，并补充跨 tokenizer 蒸馏（DSKDv2）实验。

---

## 方法详解

### 3.1 从 Wasserstein 距离到 Sinkhorn 散度

- **代价矩阵构建**：给定教师模型 token 嵌入 $\{e_i\}_{i=1}^d$，令 $C_{ij} = 1 - \cos(e_i, e_j)$（或 L2 距离），一次计算、全程固定。为提升效率，保留每个 token 的 k 个最近邻（symmetrized + self-loop），k=8（Qwen2.5 用 k=2）。

- **熵正则 Wasserstein 距离**：
  $$\tilde{D}_W^\epsilon(p, q_\theta) = \min_{P \in U(p,q_\theta)} \langle P, C \rangle - \epsilon \mathcal{H}(P)$$
  其中 $\mathcal{H}(P) = -\sum_{ij} P_{ij}(\log P_{ij}-1)$。正则化使问题可用 Sinkhorn-Knopp 算法高效求解，但引入 entropic bias。

- **Sinkhorn 散度（去偏）**：
  $$D_S^\epsilon(p, q_\theta) = \tilde{D}_W^\epsilon(p, q_\theta) - \tfrac{1}{2}\tilde{D}_W^\epsilon(q_\theta, q_\theta) - \tfrac{1}{2}\tilde{D}_W^\epsilon(p, p)$$
  满足非负、$D_S^\epsilon = 0 \iff p = q_\theta$，为 proper distillation objective。

### 3.2 WASD 目标与梯度

- **对偶重构**：熵正则 OT 的对偶问题引入对偶变量 $\phi$（student 边缘）、$\psi$（teacher 边缘），最优解满足 $\phi^* = \epsilon \log \mathbf{u}$、$\psi^* = \epsilon \log \mathbf{v}$，其中 $(\mathbf{u}, \mathbf{v})$ 由 Sinkhorn-Knopp 迭代求得。

- **Proposition 2（包络定理）**：
  $$\nabla_\theta \tilde{D}_W^\epsilon(p, q_\theta) = \sum_i \phi_i^{*(p,q_\theta)} \nabla_\theta q_\theta(i)$$
  梯度等价于 student 概率梯度的对偶势加权和，无需穿透 OT 求解器做 backprop。

- **Theorem 3（WASD 损失）**：
  $$\mathcal{L}_{\text{WASD}}^\epsilon(\theta) = \mathbb{E}\!\left[\sum_{l,i} \text{sg}\!\left(\phi_i^{*(p,q_\theta)} - \phi_i^{*(q_\theta,q_\theta)}\right) q_\theta(i)\right]$$
  在 stop-gradient 约定下，$\nabla_\theta \mathcal{L}_{\text{WASD}}^\epsilon = \nabla_\theta \mathcal{L}_S^\epsilon$。

- **与 KL 对比**：KL 梯度权重为教师概率 $p(i)$；WASD 权重为对偶势差，依赖 token 嵌入几何结构，语义相近 token 间传输代价低、概率偏移被宽容，语义无关 token 偏移被惩罚。

- **计算开销**：每次训练步需运行 Sinkhorn-Knopp 算法（Algorithm 1）与固定点迭代（Algorithm 2）各 10 步，约 2× 训练时间、+20% GPU 显存，无推理开销。

---

## 实验与结果

### 数据集与基线

- **通用指令跟随**：databricks-dolly-15k（蒸馏集）、OpenWebText（预训练）；评估 5 个基准 Dolly Eval、Self-Instruct、Vicuna、Super NI、UnNI，报告 ROUGE-L 与 Self-BLEU。
- **特定任务**：Flores-200（翻译，COMET）、DialogSum（摘要，ROUGE-L）、GSM8K（算术，准确率）、WizardCoder/HumanEval+MBPP（代码，pass@1）。
- **主要基线**：GKD、TAID、DistiLLM(SKL/SRKL)、ABKD、CSD、AMiD。

### 主要结果（关键数值）

**GPT-2 XL (1.5B) → Base (0.1B)，ROUGE-L 平均**：
- Teacher：23.35；CSD 最优基线：22.22；**WASD：24.02**（+1.80 over CSD，+0.56 over teacher 差距）
- Medium (0.3B)：**WASD 25.51** 最优，超越 AMiD 24.74（+0.77）
- OpenLLaMA2-7B→3B：**WASD 29.61** 最优，超越 AMiD 29.30（+0.31）

**GPT-4 反馈评分**（占参考分百分比）：
- Dolly Eval：WASD 37.84 vs AMiD 36.35；Vicuna：25.07 vs 22.00

**Qwen2.5-7B→1.5B（AlpacaEval/Evol-Instruct/UltraFeedback 胜率）**：
- WASD 全面优于 AMiD：90.04% / 85.32% / 73.21% vs 89.29% / 83.94% / 72.81%

**特定任务（Gemma-7B→2B）**：
- 翻译 COMET：WASD 74.53（vs AMiD 73.70）
- 摘要 ROUGE-L：WASD 35.15
- 算术准确率：WASD 24.87
- 代码生成（Qwen2.5-Coder-7B→1.5B）Avg pass@1：WASD 74.9 vs AMiD 74.1

**消融**：
- 语义代价矩阵 > 置换代价（23.47≈AMiD）> 均匀代价（22.08）—— 验证语义结构是增益来源
- 去偏 Sinkhorn 散度 > 熵正则 Wasserstein —— 去偏修正有效
- ε 敏感性：ε 小时更依赖语义几何，性能优；ε 过大则退化为近似 MMD，性能下降
- k 敏感性：k=8 最优，更大 k 无持续增益
- 跨模型族 DSKDv2：WASD-S+T 平均 ROUGE-L 21.31 vs FKL 20.47（+0.84）

### 训练效率
即使与基线按 wall-clock time 对齐，WASD 仍持续领先，更快收敛弥补额外 per-epoch 成本。

---

## 相关工作脉络

1. **f-散度蒸馏系列**：KL/RKL [25,24]、GJS (GKD) [1]、SKL/SRKL (DistiLLM) [34]、α-β (ABKD) [61]——WASD 与它们在目标层面对比，核心差异是用最优传输代价矩阵替代索引级概率比较。

2. **Score matching 蒸馏**：CSD [32] 通过对数几率差对齐避免 softmax 平滑；WASD 定位互补——CSD 解决数值稳定，WASD 解决语义感知。

3. **Assistant distribution 框架**：TAID [54]、AMiD [53]——WASD 可无缝嵌入该框架（Table 3 实验），并在此之上额外提供语义感知优化信号。

4. **Wasserstein 在表示/特征蒸馏**：WCoRD [8]、SinKD [17]、WKD [43]——面向 CV 或 batch 级特征几何，WASD 首次将语义-aware 传输用于 autoregressive next-token 分布对齐。

5. **跨 tokenizer 蒸馏**：ULD [7]、MultiLevelOT [18]、MCW-KD [60]——处理词表不匹配问题；WASD 假设共享词表，研究同一词表内部语义几何的收益，两者互补。

6. **WPR [46]**：最接近的前作，用熵正则 Wasserstein 做 RLHF 的 policy regularization；WASD 继承其 cost 矩阵构建与 nearest-k 截断，但改用去偏 Sinkhorn 散度适配 KD 目标，并在理论上证明梯度等价。

---

## 局限性与未来方向

1. **计算开销**：Sinkhorn 迭代使训练时间约 2×、显存 +20%，在大尺度模型上仍有优化空间；需探索更快的 Sinkhorn 变体或稀疏加速。

2. **代价矩阵设计单一**：目前使用教师 token 嵌入余弦距离作为通用语义代理，未融入任务/领域先验；论文指出构建 task-specific 或 domain-adaptive 代价矩阵是明确未来方向。

3. **熵正则超参敏感**：ε 增大导致性能退化（代价核变得对语义差异不敏感），需在语义保留与平滑度间折中。

4. **实验规模限制**：最大模型为 7B→1.5B/3B，对于 70B 级以上模型尚未验证；仅使用单一 GPU（RTX PRO 6000/3090）训练，分布式场景下的扩展性未测试。

5. **评估指标局限**：主要依赖 ROUGE-L，高 ROUGE-L 可能只是lexical overlap 的提升；尽管补充了 GPT-4 反馈与 win rate，但仍缺少人类评估。

---

## 研究启发与可借鉴点

1. **对偶势差梯度公式可直接迁移**：Theorem 3 的 stop-gradient 技巧（包络定理 + sg）为任何基于 OT 的目标提供了实用的优化路径，可复用到 contrastive learning、distribution matching、RLHF regularizer 等领域。

2. **语义代价矩阵的多模态扩展**：当前用 token 嵌入做代价，同样的框架可推广到视觉 token（patch embedding）、跨模态（image-text）、甚至跨 modal 的 distillation。

3. **与 assistant distribution / on-policy 蒸馏框架的融合**：WASD 可作为损失项替换任意蒸馏方法中的 KL/散度项（论文已展示与 AMiD 组合），未来可与 DistiLLM-2 的 contrastive 范式、speculative distillation 等结合。

4. **ε 调优的经验规律**：ε 小→更接近 true Wasserstein，语义几何强；ε 大→退化为 MMD。这对其他需要平衡几何保留与优化的 OT 方法有参考价值。

5. **跨 tokenizer 蒸馏的 WASD 化**：Appendix 中 WPD-S/S+T in DSKDv2 的结果提示，WASD 可通过 cross-model projector 推广到不同词表模型间蒸馏，是一条有潜力的研究方向。

---

## 关键术语表

**Wasserstein 距离**：衡量两个概率分布间"运输质量块所需最小代价"的距离，天然融入空间几何结构。

**Sinkhorn 散度**：对熵正则 Wasserstein 距离做自相似去偏得到的 proper divergence，当且仅当两分布相同时为零。

**熵正则最优传输**：在 OT 目标中加入 Shannon 熵项 $-\epsilon\mathcal{H}(P)$，使对偶问题可用 Sinkhorn-Knopp 迭代快速求解。

**对偶势（dual potential）**：OT 对偶变量 $\phi,\psi$，最优时反映各 token 的"势函数值"，WASD 用其差作为梯度权重。

**stop-gradient（sg）**：训练时阻止梯度穿过某表达式，WASD 将教师-学生对偶势差作为常量权重，只对学生概率求梯度。

**entropic bias**：熵正则化使 $ \tilde{D}_W^\epsilon(p,p) \neq 0 $，导致直接最小化该目标不能保证 $q_\theta \to p$。

**Sinkhorn-Knopp 算法**：通过交替行/列归一化迭代逼近熵正则 OT 对偶势的 matrix scaling 算法。

**cost matrix**：词汇量上定义的成对 token 间传输代价矩阵，WASD 取教师 token 嵌入的余弦距离。

---

## 可复现要素

| 要素 | 状态 |
|---|---|
| 代码 | 开源：https://github.com/aailab-kaist/WASD |
| 数据集 | 全部公开：databricks-dolly-15k、OpenWebText、Flores-200、DialogSum、GSM8K、WizardCoder、HumanEval、MBPP |
| 模型权重 | 使用公开模型：GPT-2 XL/Base/Medium、OpenLLaMA2-7B/3B、Gemma-7B-IT/2B-IT、Qwen2.5-7B/1.5B-Instruct、Qwen2.5-Coder-7B/1.5B-Instruct |
| 关键超参 | ε=0.001（默认）、k=8（Qwen2.5 用 k=2）、Sinkhorn 迭代 10 步、温度缩放 T=2、lr=1e-4（GPT-2/OpenLLaMA2）/1e-5（Gemma）/5e-5（Qwen） |
| 训练设备 | 单卡 NVIDIA RTX PRO 6000（训练）/ RTX 3090（评估） |
| 训练轮次 | GPT-2/OpenLLaMA2：20 epochs；Gemma 特定任务：蒸馏 3–10 epochs；Qwen 代码：1 epoch |
| 评估种子 | GPT-2/OpenLLaMA2 用 5 个 eval seed（10,20,30,40,50）；消融用 3 training seeds |

---
