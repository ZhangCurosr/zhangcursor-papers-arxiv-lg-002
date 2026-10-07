---
title: "On-Policy-Distillation-with-Negative-Policy-Rollouts"
source: https://arxiv.org/pdf/2610.07874v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:11:40"
field: "大语言模型后训练与蒸馏"
keywords: ["on-policy distillation", "negative policy", "post-training", "knowledge distillation", "LLM reasoning"]
innovations: ["在rollout阶段混入低能力负策略轨迹为OPD注入显式负向信号，不修改reward", "提出NSP/NOR诊断指标量化负压制与分布偏移", "揭示ExOPD/OPD²隐式负信号来源并提供可叠加的负策略扩展"]
benchmarks: ["AIME24", "AIME25", "AMC23", "HMMT25", "MATH500", "OlympiadBench", "RGMath", "CodeForces", "LCBv5", "RGAlgo", "GPQA", "SuperGPQA", "SciBench"]
---

# 论文速读：On-Policy Distillation with Negative-Policy Rollouts

## 一句话总结
论文提出了**Negative-Policy On-Policy Distillation (NP-OPD)**，通过在学生模型自己的采样轨迹中混入低能力负策略的轨迹，为On-Policy Distillation (OPD) 提供显式的负向学习信号；在数学、代码、科学推理等13个基准上，NP-OPD 持续优于传统OPD基线，且可与ExOPD、OPD²等现有变体兼容。

## 研究问题与动机
- **OPD的"正信号单一性"局限**：标准OPD仅通过教师模型的token级监督引导学生逼近教师分布，但当学生与教师分布重叠较低时，仅靠正面指引往往信号不足。
- **偏好优化已有"正负成对"范式成功先例**：DPO等方法通过同时提供偏好/拒绝样本或正/负优势信号取得显著效果，但OPD尚未系统性地引入类似负向参考。
- **修改reward formulation的兼容性代价高**：现有OPD改进多集中在token-level reward设计，直接修改损失函数会破坏与后续OPD变体的兼容性。
- **低能力策略能否充当有效负参考存疑**：直接降格学生自身（如提高temperature、persona prompt、layer drop）能产生部分互补信号，但效果远弱于独立负策略，说明负参考需要具备明确的"能力差距+分布差异"结构。

## 核心贡献（创新点）
1. **提出NP-OPD框架**：在rollout阶段混入低能力负策略采样轨迹，不修改任何distillation reward，即可为OPD提供显式负向信号——与ExOPD/OPD²等reward改进正交、可叠加。
2. **理论解释负策略的梯度提升机制**：从采样概率差异推导出NP-OPD对负策略轨迹的梯度增强项（公式10-12），证明在假设 $D_{KL}(\pi_n\|\pi^*) > D_{KL}(\pi_n\|\pi_\theta)$ 下，负策略token的期望OPD reward为负，从而系统性压制学生趋向负策略。
3. **定义并验证NSP与NOR两个诊断指标**：NSP衡量被压制token中"负策略更偏好于教师"的比例，NOR衡量学生在教师与负策略分歧位置的概率偏移；两指标均显示NP-OPD按设计工作，且与性能提升高度相关。
4. **揭示现有OPD变体的隐式负信号来源**：ExOPD和OPD²即使只用on-policy rollout也能产生更高NSP，因其reward相对"教师base model"计算，天然含有隐式负参考；NP-OPD仍可在此基础上进一步增益。
5. **发现负策略rollout的额外工程收益**：预生成的负策略rollout可重复使用，将Qwen3-1.7B思考模式训练时间从9h19m降至3h34m（α=1时），且不牺牲性能。

## 方法详解
- **负策略rollout混合**：每步训练以概率 $1-\alpha$ 采学生轨迹 $y\sim\pi_\theta(\cdot|x)$、以概率 $\alpha$ 采负策略轨迹 $y\sim\pi_n(\cdot|x)$；主要设置中负策略与学生同族但容量更低（如1.7B学生配0.6B负策略）。
- **Token-level OPD reward不变**：沿用 $R_t^{OPD}=\log\pi^*(y_t|s_t)-\log\pi_\theta(y_t|s_t)$，仅在采样分布上由单一下界 $\pi_\theta$ 变为混合分布 $\pi_{z_x}$，因此与所有OPD reward变体直接兼容。
- **梯度增强推导**：NP-OPD相对OPD多出 $\alpha[\prod\pi_n-\prod\pi_\theta]\cdot R_t^{OPD}\nabla\log\pi_\theta$ 项，当负策略token的OPD reward为负时，该项放大对负策略轨迹的压制梯度。
- **超参**：$\alpha\in\{0,0.25,0.5,0.75,1.0\}$ 可调；batch size=256，lr=$5\times10^{-6}$，最多100步，rollout温度1.0，最长16k token。
- **实现便捷性**：负策略rollout只需训练前批量生成一次并缓存，训练阶段无需再对负策略进行前向/反向传播。

## 实验与结果
- **模型与设置**：Qwen3-1.7B/4B/8B、Gemma-4-E4B-it；思考与非思考双模式；训练集为Math/Science/Code混合30k题。
- **非思考模式主结果**：Qwen3-1.7B (+OPD) Math平均35.7→48.3→56.3，Code 12.3→27.2→31.1，Science 30.8→37.2→41.5；Qwen3-4B Math 46.8→62.2→70.5，Code 24.9→40.2→51.1，Science 40.4→47.8→51.2。
- **思考模式主结果（Table 2）**：Qwen3-1.7B OPD整体平均49.7→51.6(+1.98)；ExOPD 52.8→53.1(+0.30)；OPD² 53.7→55.5(+1.76)。Qwen3-4B OPD 61.7→62.9(+1.11)，ExOPD 64.6→66.5(+1.86)，OPD² 65.4→67.5(+2.10)。
- **跨模型族泛化**：Gemma-4-E4B Math 63.0→66.4(+3.4)，Science 49.1→49.8(+0.7)。
- **最强提升**：Qwen3-4B OPD²在Thinking模式下全13基准平均提升2.10点；AIME24/25、LCBv5、RGAlgo等单项提升尤为显著。
- **对比消融**：仅用teacher rollout会劣化性能（NOR=-0.030），而低能力负策略才带来正向效果。
- **推理效率**：α=1时训练时间缩短约2.6倍（思考模式）至4.1倍（非思考模式），前提是预生成负策略rollout。

## 相关工作脉络
- **OPD原始版本** [1, 20]：学生自生成轨迹+教师token级监督；本文在rollout混合上扩展，而非修改token reward。
- **SeqKD/SFT范式** [17, 28, 27, 10]：用教师生成序列做监督；本文指出其与学生on-policy能力存在分布 mismatch。
- **DPO偏好优化** [31]：成对偏好/拒绝样本提供正负双向信号；NP-OPD的核心动机来源于此，但以rollout混合方式实现而非loss改写。
- **ExOPD** [43]：引入teacher base model作为reference进行reward extrapolation；本文指出其已隐含一定负信号，但与NP-OPD正交可叠加。
- **OPD²** [13]：使用teacher Δ信号并与OPD signal方向对齐；同样含隐式负信号，NP-OPD仍能提供额外增益。
- **Lightning OPD / Relay-OPD** [38, 41]：从trajectory构建角度改进监督质量；本文从rollout source多样性切入，视角不同。

## 局限性与未来方向
- **负策略容量选择敏感**：能力过强（甚至teacher自身）会退化性能；如何自动搜索最优 $\alpha$ 与负策略尺度尚缺系统研究。
- **仅验证推理型小模型**：目前集中在Qwen3/Gemma的1.7B–8B尺度，超大模型（>70B）及多模态场景未见探索。
- **训练数据量有限**：30k题规模对于长尾领域可能不足；负策略在不同数据分布下的稳定性未检验。
- **混合比例的domain特异性**：Table C.17/C.19显示α=1在Math/Science表现更强而Code可能略逊，模型merge可作为折中但增加了部署成本。
- **理论假设的现实性依赖**：公式11的KL不等式在实际复杂推理任务中未必严格成立，需更多实证验证。

## 研究启发与可借鉴点
- **rollout source设计可作为独立正则项**：不改动reward即可为OPD注入负信号，该思路可迁移至RLHF、PPO等基于采样的后训练流程。
- **NSP/NOR指标体系可直接复用于其他distillation变体评估**：量化"被压制token是否确实属于负参考偏好"与"学生是否真正远离负参考"，提供可解释的诊断工具。
- **预生成rollout缓存策略的工程价值**：对离线/低成本部署场景，用静态负策略rollout替代部分on-policy采样可大幅节约算力，值得结合团队训练pipeline落地。
- **隐式负信号的挖掘视角**：ExOPD/OPD²的性能提升部分源于其隐式负信号；未来可系统分析其他OPD变体（如Entropy-aware OPD [16]）是否同样蕴含负参考。
- **模型merge作为α调优的补充**：固定权重平均 $\alpha=0$ 与 $\alpha=1$ 模型可兼顾domain差异，为多领域通用后训练提供简单工程方案。

## 关键术语表
**On-Policy Distillation (OPD)**：学生在自身采样轨迹上接受更强教师token级监督的后训练方法，保留on-policy分布一致性。
**Negative-Policy OPD (NP-OPD)**：本文提出，通过在rollout中混入低能力负策略轨迹为OPD提供显式负向信号。
**Negative-policy suppression precision (NSP)**：度量被学生压低概率的token中，由负策略更偏好于教师的比例；越高表示负压制越精准。
**Negative-policy overlap reduction (NOR)**：衡量学生在教师与负策略分歧位置，相对教师方向偏移的程度；正值表示学生远离负策略。
**ExOPD**：扩展OPD，引入teacher base model作为reference并外推reward，隐含隐式负信号。
**OPD²**：采用teacher delta信号（post-trained vs base），仅在与OPD方向一致时激活，亦含隐式负成分。
**Rollout mixing ratio (α)**：控制负策略rollout在训练batch中的占比，0为纯OPD，1为纯负策略rollout。
**Teacher-student distributional overlap**：学生与教师生成分布的重叠程度；重叠越低，单纯正信号越不足，NP-OPD增益越大。

## 可复现要素
- **数据集**：Math [26]、Science [29]、Code [2] 混合30k题；评测基准13个（AIME24/25、AMC23、HMMT25、MATH500、OlympiadBench、RGMath、CodeForces、LCBv5、RGAlgo、GPQA、SuperGPQA、SciBench）；部分公开链接见参考文献。
- **代码**：论文声明将开源（https://github.com/naver-ai/np-opd），截至阅读时尚未发布完整release。
- **关键超参**：batch size=256、lr=$5\times10^{-6}$、100 steps、rollout温度1.0、最大长度16k token、eval温度0.7、最大生成长度32k token、KL系数β=0（Qwen）/0.04（Gemma）。
- **硬件**：Qwen实验单节点8×H100，Gemma双节点；使用ZeRO-3与bf16精度。
- **负策略模型**：Qwen3-0.6B/1.7B/4B分别对应1.7B/4B/8B学生；Gemma-4-E2B-it对应E4B-it学生。
