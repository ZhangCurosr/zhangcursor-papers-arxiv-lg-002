---
title: "OPSRD-ON-POLICY-SELF-ROLE-DISTILLATION"
source: https://arxiv.org/pdf/2609.39884v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 22:00:58"
field: "大语言模型高效微调与蒸馏"
keywords: ["on-policy distillation", "role prompting", "forward KL", "self-distillation", "reasoning", "large language models"]
innovations: ["将固定专家角色作为可复用教学上下文替代参考答案，实现无参考解的on-policy自我蒸馏", "提出entropy-masked forward KL目标，在高熵位置裁剪词汇贡献以纠正学生对教师支持备选token的低估", "系统验证teacher-only角色放置与forward KL散度在所有模型尺度上的最优性"]
benchmarks: ["AIME 2024", "AIME 2025", "HMMT 2025"]
---

# 论文速读：OPSRD-ON-POLICY-SELF-ROLE-DISTILLATION

## 一句话总结
OPSRD 是一种基于专家角色条件的前向KL散度on-policy自我蒸馏方法，无需每道题的参考答案，通过在学生对前缀熵较高的位置上学习教师角色分布的偏好，显著提升Qwen3系列模型在竞赛数学任务上的准确率。

## 研究问题与动机
- **直接角色提示的有效性存疑**：在竞赛数学任务上，为Qwen3-1.7B添加"奥林匹克数学专家"角色后，宏观准确率反而从35.74下降至34.54，与中性指令持平，说明单纯的角色prompting无法带来实测提升。
- **已有蒸馏方法依赖参考答案**：Answer OPSD等方法需要每道题的参考答案作为教师监督信号，限制了适用场景；如何在无参考解的条件下有效传递专家知识尚未验证。
- **采样轨迹可能丢失有用token偏好**：完整答案若最终错误，其中蕴含的有用下一token偏好（如在不确定位置应继续哪种推理路径）未被采样或后续步骤出错而被掩盖，直接蒸馏完整答案会遗漏这些信号。
- **角色条件分布的价值未被充分挖掘**：角色对模型token级分布的影响可能在中间推理步骤中就已经体现，即便最终答案不佳，其条件分布仍可为学生的不确定位置提供有效指导。

## 核心贡献（创新点）
1. **将固定专家角色作为可复用的教学上下文替代参考答案**：无需每道题的reference solution，仅通过角色提示即可为教师提供分布监督信号，拓展了on-policy蒸馏的适用边界。
2. **提出entropy-masked forward KL目标函数**：仅对学生序列中熵最高的50%位置施加监督，并对每个词汇项的KL贡献施加裁剪（τ=0.05/0.06），使教师支持的但学生低估的备选token获得直接梯度信号。
3. **系统验证了角色放置位置与散度选择**：通过消融实验证明teacher-only角色放置最优；forward KL在所有模型尺度上均达到最高macro准确率，尤其在小模型（1.7B）上较reverse KL提升4.81pp。
4. **揭示了角色prompting与角色蒸馏的本质差异**：同一专家角色在直接prompting下无增益（甚至负增益），但作为教师分布源进行on-policy蒸馏后可带来显著收益（+6.30pp），阐明了"角色价值在于分布而非采样"的机制。

## 方法详解
**整体架构**：冻结的教师模型与学生共享同一基础权重，学生仅训练低秩适配器（LoRA rank=64），训练时学生使用无角色模板生成轨迹，教师使用专家角色模板对相同前缀计算条件分布。

**核心流程**：
1. **学生采样**：对学生base模型$p_{\theta_0}$叠加LoRA参数$\theta$，在无角色上下文$c_S$下采样推理轨迹$y_{1:T} \sim p_\theta(\cdot|c_S, x)$，采样温度1.1、top-p=0.95、top-k=20。
2. **熵选择**：对每个有效位置$t$计算学生分布熵$H_t = -\sum_v p_S^t(v)\log p_S^t(v)$，保留每序列中熵最高的$k=\max(1,\lceil\rho T\rceil)$个位置（默认$\rho=0.5$），选择过程detach，不参与梯度。
3. **教师评分**：冻结教师使用专家角色上下文$c_T$，对完全相同的student prefix $y_{<t}$计算教师分布$p_T^t(v) = \text{softmax}(z_T^t/\eta)_v$，其中$\eta=1.1$为蒸馏温度。
4. **前向KL目标**：对选中位置计算$\text{KL}(p_T^t\|p_S^t)$，对每个词汇项的贡献$d_{t,v}=p_T^t(v)\log[p_T^t(v)/p_S^t(v)]$施加上限裁剪$\min\{d_{t,v},\tau\}$（1.7B/4B取$\tau=0.05$，8B取$\tau=0.06$），按选中位置数平均得到batch loss。
5. **梯度更新**：仅对LoRA参数反向传播，教师权重完全冻结，采样token和教师概率均detach。

**Forward KL优势**：其梯度为$(p_S(j)-p_T(j))/\eta$，当教师赋予高概率而学生赋予低概率时直接增大该token的logit；而Reverse KL的梯度含因子$p_S(j)$，在学生概率趋近零时梯度消失，无法纠正低估。

## 实验与结果
**数据集与模型**：使用OpenThoughts数学分割（29,434题），在Qwen3-1.7B/4B/8B上进行评估；训练LoRA rank=64，AdamW优化器100步，学习率$5\times10^{-6}$，effective batch size=36。

**评测基准**：AIME 2024、AIME 2025、HMMT 2025（各30题），每题12次采样，温度1.0、top-p=0.95，报告Avg@12及macro均值。

**主要结果（Table 1, 2, 6）**：
| 模型 | Base | Answer OPSD | OPSRD | ∆ Base |
|------|------|-------------|-------|--------|
| Qwen3-1.7B | 35.74 | 38.70 | **42.04** | **+6.30** |
| Qwen3-4B | 60.86 | 61.50 | **62.52** | +1.67 |
| Qwen3-8B | 62.78 | 64.26 | **65.83** | +3.06 |

- 1.7B上OPSRD相对Base提升6.30pp（95% CI: [3.24, 9.54]），显著优于Answer OPSD的+2.96pp。
- Forward KL在所有尺度上均为三种散度中表现最优；1.7B上较Reverse KL领先4.81pp（CI: [2.41, 7.31]）。
- Teacher-only角色放置最佳（42.04），Shared role降至38.98，Student-only降至38.24；加入测试角色反而降低各checkpoint分数。
- 中性教师+全位置蒸馏仅提升Base 2.50pp，专家角色在此基础上额外贡献3.80pp。
- 50%熵选择 vs 全位置蒸馏差距不显著（mean diff=0.77pp, CI含0），但选择位置的平均熵（0.874）远高于未选位置（0.023），验证了选择的有效性。
- 三次独立训练重复中，OPSRD在1.7B上稳定超越Base，平均提升4.51pp。

**最强结果**：Qwen3-1.7B上OPSRD取得macro Avg@12 = **42.04**，相对Base提升**+6.30pp**；8B上达到**65.83**（+3.06pp）。

## 相关工作脉络
- **OPSD (Zhao et al., 2026)**：同一框架的先前工作，使用privileged solution作为教师上下文；OPSRD用可复用的expert role替代每道题的reference solution，解决了需要参考答案的限制。
- **ExpertPrompting (Xu et al., 2023)**：为每个instruction构建expert identity生成响应并训练；侧重prompt工程与SFT，而非on-policy token级蒸馏。
- **MiniLLM / Reverse KL蒸馏 (Gu et al., 2024)**：使用student-weighted reverse KL进行sequence-level蒸馏；OPSRD证明forward KL在teacher-supported但student-underestimates的备选token上更有效。
- **Entropy-aware OPD (Jin et al., 2026)**：在教师熵高时引入forward KL；OPSRD从学生熵角度选择监督位置，两者选择逻辑互补。
- **RoleLLM (Wang et al., 2024) / ExpertRole prompting**：研究角色扮演能力的评估与训练；本文聚焦于角色作为蒸馏教师信号而非目标行为本身。
- **Speculative Knowledge Distillation (Xu et al., 2025)**：让教师替换学生rank低的proposal；OPSRD在同一student-generated prefix上直接对齐token分布，无需speculation步骤。

## 局限性与未来方向
- **单一模型族与任务**：仅测试Qwen3家族和数学竞赛任务，未验证到其他模型架构（如Llama、Mistral）或其他推理领域（如代码生成、科学QA）的泛化性。
- **蒸馏温度与超参敏感性**：温度η=1.1、裁剪阈值τ=0.05/0.06的选择未做系统扫描；不同模型尺度可能需要差异化配置。
- **熵选择比例未充分探索**：25%/50%/75%的 pilot study中50%为默认但提升幅度有限，更优比例需更大采样预算验证。
- **JSD消融在异构硬件上运行**：JSD在H100上运行而KL在A100上运行，硬件差异可能引入混杂因素，结果可信度略受影响。
- **教师分布质量假设**：方法假设teacher distribution总是优于student，但实验显示direct prompting本身无增益，角色分布中可能包含噪声信号，现有裁剪机制是否充分过滤尚需分析。

## 研究启发与可借鉴点
1. **角色作为可复用蒸馏信号的范式**：将expert role从"直接生成答案的工具"重新定位为"token分布监督源"，即使角色prompting本身无增益，其条件分布仍可能蕴含对学生有用的推理偏好——这一思路可迁移至其他需要专家知识的领域（代码、法律、医疗）。
2. **Forward KL + 熵mask的组合策略**：在学生高熵位置施加forward KL，既聚焦于不确定性高的决策点，又通过teacher概率加权直接纠正低估token，该组合可推广至其他on-policy蒸馏场景。
3. **Teacher-only角色放置的ablation设计**：系统对比role placement（teacher/student/both）及test-time role的影响，清晰分离了"轨迹生成"与"分布监督"两个功能，这种正交消融思路值得借鉴。
4. **裁剪机制防止单个词汇主导**：对KL贡献的单词上限裁剪（τ=0.05/0.06）避免教师极端偏好压制学生其他合理选项，这一技巧在teacher-student分布差异较大时尤为关键。
5. **无需参考答案的self-distillation扩展**：OPSRD证明角色即可提供足够监督信号，结合团队已有的数据构建Pipeline，可在无标注/弱标注场景下快速部署。

## 关键术语表
**OPSRD**：On-Policy Self-Role Distillation的缩写，本文提出的方法，利用专家角色条件分布对on-policy学生轨迹进行self-distillation。

**Forward KL / Reverse KL**：Forward KL为$\text{KL}(p_T\|p_S)=\sum p_T\log(p_T/p_S)$，给教师高概率token更强梯度；Reverse KL为$\text{KL}(p_S\|p_T)=\sum p_S\log(p_S/p_T)$，在学生概率趋近零时梯度消失。

**Entropy selection / mask**：根据学生在各位置的条件分布熵排序，仅对高熵（不确定性高）的位置施加蒸馏损失，默认选取前50%。

**Clipped vocabulary contribution**：对每个词汇项的KL贡献$d_{t,v}$施加上界$\tau$（1.7B/4B为0.05，8B为0.06），防止教师极端偏好主导梯度更新。

**Teacher-only role placement**：专家角色仅出现在教师上下文中，学生rollout和eval均不使用角色；消融实验表明此放置方式效果最优。

**Macro Avg@12**：在AIME/HMMT等每个30题的benchmark上，每题采样12次计算平均正确率，再对三个benchmark取未加权均值。

**On-policy distillation**：教师在学生实际生成的轨迹前缀上评分，而非独立生成的teacher输出，缓解train-test分布偏移。

**LoRA adapter**：Low-Rank Adaptation，仅训练低秩分解的适配器参数（本文rank=64），冻结基础模型权重，降低训练成本。

## 可复现要素
- **数据集**：OpenThoughts数学分割（siyanzhao/Openthoughts_math_30k_opsd），29,434题，Hugging Face公开可下载。
- **代码**：完全开源，GitHub地址 https://github.com/zhansan114514/OPSRD，含训练/评估脚本、超参配置、bootstrap区间计算脚本。
- **模型权重**：Base模型Qwen3-1.7B/4B/8B在Hugging Face公开；训练后的LoRA checkpoint未随仓库发布，需自行训练。
- **关键超参**：LoRA rank=64/scale=128；learning rate=$5\times10^{-6}$；steps=100；batch size=36；蒸馏温度η=1.1；裁剪阈值τ=0.05（1.7B/4B）或0.06（8B）；熵选择比例ρ=0.5；训练采样top-p=0.95/top-k=20；评估12 samples/题，温度1.0/top-p=0.95。
