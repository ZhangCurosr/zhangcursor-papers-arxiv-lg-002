---
title: "FROM-SOFT-TARGETS-TO-REWARD-SIGNALS-HOW-ASSIGNMENT-AND-REWAR"
source: https://arxiv.org/pdf/2609.34850v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 08:05:06"
field: "奖励建模与偏好学习"
keywords: ["reward modeling", "soft targets", "preference learning", "assignment geometry", "Bradley-Terry", "RLHF"]
innovations: ["提出分配几何框架解耦软目标分散度与对应关系", "发现完整对应关系在五种奖励目标下均保留最大干净间隔", "APLOT Uniform 目标在独立校准下提供跨编辑库额外衰减"]
benchmarks: ["UltraFeedback", "HelpSteer", "RM-Bench", "RewardBench"]
---

# 论文速读：FROM-SOFT-TARGETS-TO-REWARD-SIGNALS-HOW-ASSIGNMENT-AND-REWARD-OBJECTIVES-INTERACT

## 一句话总结
本文引入**分配几何（assignment geometry）**框架，系统研究软偏好目标的放置方式（均匀/完整/重分配）与五种奖励目标函数（BT、NormBT、BSR、APLOT、DARM）的交互作用，揭示在相同准确性预算下"保持完整对应关系"能保留最大干净偏好间隔，而衰减排序则因目标而异。

## 研究问题与动机
1. **核心问题**：将相同的一组软偏好目标值分配给不同响应对，会通过不同的奖励目标函数产生怎样的奖励信号差异？现有工作缺乏对"目标值"和"目标-响应对应关系"两者的解耦分析。
2. **现有方法不足（分散监督视角）**：Distillation、Label Smoothing、Instance-specific smoothing 等已有工作是分类学习中的分配思想，但尚未被系统引入奖励建模；如何将分配控制与奖励目标的设计结合起来仍不明确。
3. **现有方法不足（奖励目标视角）**：APLOT、NormBT、BSR、DARM 等新型奖励目标各自改进了标准化、自适应间隔或上下文依赖，但相同软目标放置在不同目标下的表现差异未被系统揭示。
4. **评估盲点**：仅凭偏好准确率（preference accuracy）无法区分相同准确率下奖励信号的真实质量差异；需要引入间隔保持率（retention）和编辑衰减（attenuation）联合评估。

## 核心贡献（创新点）
1. **提出分配几何（assignment geometry）框架**：通过均值匹配的平滑（mean-matched smoothing）控制目标分散度，通过层内重分配（within-stratum reassignment）改变对应关系但保持完整目标分布，从而将"目标值"与"目标-响应对应关系"解耦。
2. **发现跨目标的一致性保留效应**：在五种奖励目标下，保持完整对应关系（Intact）均能在相同准确性预算下保留最大的干净偏好间隔（Intact vs Reassigned 增益 +0.0639~+0.0946），揭示"对应关系"是跨目标通用的奖励幅度决定因素。
3. **揭示目标依赖的衰减顺序**：同一组软目标分布在 BT/APLOT 下 Uniform 聚合衰减更强，而在 NormBT 下 Intact 反而更强（差值 0.1004），说明目标选择与放置选择是耦合决策。
4. **开发衰减-保留联合曲线（attenuation-retention profile）与独立校准缩放**：通过独立校准的标度控制分离"分布匹配"与"幅度匹配"，发现 APLOT Uniform 在两种编辑库（aggregate 和 presentation）上均可提供超越独立校准标度的额外衰减（aggregate: +0.0315, presentation: +0.0191）。

## 方法详解
**1. 分配几何与三种目标放置（Equation 2）**
- 定义修正质量 $m_i = 1 - t_i$，按层 $g$ 分组，构造：
$$m_i^{(\lambda, p)} = \bar{m}_{g(i)} + \lambda (m_{\pi_{g,p}(i)} - \bar{m}_{g(i)}), \quad t_i^{(\lambda,p)} = 1 - m_i^{(\lambda,p)}$$
- 其中 $\lambda$ 控制分散度（$\lambda=0$ 为 Uniform，$\lambda=1$ 为恢复源分布），$p$ 控制重分配比例。
- 三种放置：Uniform（$\lambda=0, p=0$，层内恒定）、Intact（$\lambda=1, p=0$，保持原对应）、Reassigned（$\lambda=1, p=1$，层内循环重排）。

**2. 奖励目标函数族**
- **BT**：软 Bradley-Terry 损失 $\ell_{BT}(d,t) = -t\log\sigma(d) - (1-t)\log(1-\sigma(d))$
- **NormBT**：loss 乘以 detached EMA 表示距离比
- **BSR**：添加 0.001B × (批均值奖励)$^2$ 正则项
- **APLOT**：结合语义相似度与最优传输（Sinkhorn 迭代 50 次）的自适应成本
- **DARM**：偏好 loss + 上下文判别 loss（权重 0.05，4 个 prompt 对比）

**3. 三项核心度量**
- **间隔保留率 $\mathcal{R}(\theta)$**：以硬标签 raw 控制为基准，衡量干净偏好间隔幅度的相对保持（Equation 5）
- **准确率变化 $\Delta\mathcal{U}$**：偏好方向上的准确率变化
- **编辑衰减 $\mathcal{A}_\mathcal{E}$**：对 edit bank 中编辑响应的绝对评分变化（Equation 7）

**4. 独立校准缩放（Independently calibrated scaling）**
- 对每个 seed 在独立校准集上拟合 $c_\theta = \mathbb{E}_C|d_\theta| / \mathbb{E}_C|d_0|$，用 $c_\theta r_0$ 作为参考
- 在held-out编辑数据上测量"超出标度匹配后的额外衰减"，分离分布匹配与幅度匹配效应

**5. 目标来源构造（LCC anchor）**
- 冻结 DeBERTa-v3-base 表示，取响应 token 平均做 chosen-minus-rejected 差，训练截断 SVD 至 64 维
- 锚概率 $a_i = \text{clip}[\sigma(z_i/T), 0.05, 0.95]$，混合目标 $t_i = 0.9 + 0.1 a_i$

## 实验与结果
**数据集与设置**
- 训练集：99,926 对偏好数据（来自 UltraFeedback、HelpSteer 等），验证集 7,697 对
- 评估集：22,561 干净对 + 21,360 聚合编辑 + 10,680 呈现编辑
- 骨干网络：DeBERTa-v3-base，学习率 $2\times10^{-5}$，batch=32，1 epoch

**主要结果**
- **间隔保留（Table 2/3）**：在 BT 下 Intact（0.8956）> Uniform（0.8285）> Reassigned（0.8177），Intact 比 Reassigned 高 +0.0779；该方向在五种目标下一致成立（NormBT: +0.8671/0.9214/0.8575，BSR: +0.8420/0.9260/0.8314，APLOT: +0.8596/0.8940/0.8266，DARM: +0.7776/0.9105/0.8322）
- **编辑衰减顺序随目标变化（Figure 3/附录 Table 6）**：BT 下 Uniform 衰减比 Intact 多 +0.0304，APLOT 下多 +0.0670，但 NormBT 下 Intact 反而多 -0.0334；BT vs NormBT H1 交互 0.1004（[0.0728, 0.1419]）
- **跨重分配与跨来源稳健性（Table 4）**：独立重分配实验 Intact 保留增益 +0.0761，相关源变体 +0.0885，所有 seed 方向一致
- **APLOT Uniform 额外衰减（Table 14/Figure 4）**：超出独立校准标度的额外衰减——聚合编辑 +0.0315（CI [0.0064, 0.0660]）、呈现编辑 +0.0191（CI [0.0059, 0.0314]）
- 所有十五种配置在 ±1pp 准确性等价预算内通过检验

## 相关工作脉络
1. **Distillation / Label Smoothing（Hinton 2015; Müller 2019; Zhang & Sabuncu 2020）**：这些工作在分类中学习分配变量，本文将其思想迁移到奖励建模，并首次将"分散度"与"对应关系"分解为两个独立可控维度。
2. **MMPO（Kim et al. 2024）/ 序数反馈学习（Liu et al. 2025a）**：编码偏好强度的方法；本文在此基础上追问：给定一个软目标源，值本身和对应关系各贡献多少？
3. **APLOT（Li et al. 2025b）**：自适应间隔+最优传输；本文发现 Uniform 目标经 APLOT 训练可产生超越标度控制的额外衰减，揭示了目标值与目标函数的协同效应。
4. **NormBT（Xie et al. 2026）/ BSR（Hong et al. 2025）/ DARM（Liu et al. 2026）**：改进标准化、批正则化和上下文依赖的目标；本文通过统一框架揭示它们在间隔保留上一致（Intact 最优），但在衰减排序上存在本质差异。
5. **RewardBench（Lambert et al. 2025）/ RM-Bench（Liu et al. 2025b）**：评估基准；本文指出单纯准确率不够，需引入 edit attenuation 和 calibrated scaling 作为补充度量。

## 局限性与未来方向
1. **来源构造的局限**：软目标基于 LCC anchor（表示+表面特征 logistic 回归）生成，其质量依赖 frozen 表示；未探索完全端到端学习目标分配的方案。
2. **有限奖励目标**：仅比较了 5 种目标（BT, NormBT, BSR, APLOT, DARM），其他新兴目标（如 IPO、CPO 等）未纳入，跨目标通用性有待进一步验证。
3. **未涉及下游策略优化**：结论停留在奖励模型层面，尚未验证这些间隔/衰减变化如何传导至 RLHF/DPO 等下游策略优化。
4. **未来方向**：作者提议从验证 profile 中学习目标分配，并评估对生成行为的下游影响。

## 研究启发与可借鉴点
1. **分配几何框架可迁移**：均值匹配平滑 + 分布保持重分配的实验设计思路可推广到其他需要软监督的任务（如序列标注、多分类），实现"值 vs 位置"效应解耦。
2. **独立校准缩放方法**：用独立校准集拟合标度、再在 held-out 编辑上测"额外衰减"的策略，为奖励模型比较提供了比单纯准确率更丰富的评估维度。
3. **衰减-保留联合曲线的可视化设计**：将 retention、$\Delta U$、aggregate/presentation attenuation 四个坐标联合展示，为多目标对比提供了一个清晰的比较范式。
4. **与团队方向结合机会**：若团队关注奖励模型过优化（overoptimization）或对齐鲁棒性，Intact 对应关系的间隔保留优势与 APLOT Uniform 的额外衰减可作为设计先验；可进一步探索"在 DPO/RLHF 上游使用 Intact 软目标"的下游影响。

## 关键术语表
**Assignment Geometry（分配几何）**：通过分散度参数 $\lambda$ 和重分配参数 $p$ 构建的软目标可控空间，将目标值和目标-响应对应关系解耦。
**Mean-Matched Smoothing（均值匹配平滑）**：保持每层修正质量均值不变，通过 $\lambda$ 调节层内标准差的软目标构造方法。
**Within-Stratum Reassignment（层内重分配）**：在保持完整目标值多重集不变的前提下，将目标值随机重分配给同层内的响应对。
**Clean-Margin Retention（间隔保留率）**：以硬标签 raw scorer 为基准，衡量软目标训练后干净偏好间隔幅度的相对保持程度（$\mathcal{R}$）。
**Edit Attenuation（编辑衰减）**：衡量奖励模型对响应编辑（聚合/呈现）后评分绝对值的变化，正值表示对编辑的敏感性降低。
**Calibrated Scaling（校准标度）**：在独立校准集上拟合正标度系数，匹配候选 scorer 与 raw scorer 的干净间隔幅度，用于分离"分布效应"与"幅度效应"。
**LCC Anchor（Learning from Contrasting Candidates 锚点）**：基于冻结 DeBERTa 表示和表面特征的 logistic 回归输出的软概率，用于混合生成初始软目标。

## 可复现要素
- **数据集**：UltraFeedback、UltraFeedback-Binarized、HelpSteer、HelpSteer2、RM-Bench、RewardBench（训练 99,926 对，验证 7,697 对，测试 15,782 对；附录 A 详述去重与分层方法）
- **代码**：论文未明确声明代码开源；APLOT 使用了官方实现（Li et al. 2025a），其余目标基于 PyTorch + HuggingFace Transformers 实现
- **关键超参**：骨干 DeBERTa-v3-base；学习率 $2\times10^{-5}$，batch=32，1 epoch，warmup=3%，weight decay=0.01，BF16；$\alpha=0.10$（锚点混合系数）；LCC 锚点 clip [0.05, 0.95]；训练目标 $t_i = 0.9 + 0.1 a_i$；5 层 stratum；三种子（7, 17, 29）
