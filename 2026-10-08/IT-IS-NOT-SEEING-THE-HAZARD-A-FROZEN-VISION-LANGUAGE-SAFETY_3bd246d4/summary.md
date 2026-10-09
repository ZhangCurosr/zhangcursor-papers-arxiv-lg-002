---
title: "IT-IS-NOT-SEEING-THE-HAZARD-A-FROZEN-VISION-LANGUAGE-SAFETY"
source: https://arxiv.org/pdf/2610.09517v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 02:56:23"
field: "安全强化学习与VLM审计"
keywords: ["vision-language model", "safe reinforcement learning", "CLIP", "safety audit", "constrained MDP", "scene-template", "prompt margin"]
innovations: ["被动观测审计框架分离评分语义与策略效用", "证实冻结VLM安全评分主要测量场景模板偏离而非危险感知", "常量置信度placebo证明策略改进不依赖帧级视觉信息"]
benchmarks: ["FormulaOne", "MetaDrive-Hard", "Safety-Gymnasium (CarButton1, PointGoal1)"]
---

# 论文速读：IT IS NOT SEEING THE HAZARD: A FROZEN VISION-LANGUAGE SAFETY SCORE MEASURES ITS CAPTION BANK

## 一句话总结
论文通过被动观测审计发现，冻结 CLIP 模型的安全评分下降并非源于检测到 approaching hazard，而是反映帧图像偏离了标题库中赛车场景的共同视觉模板；且用常量置信度替代帧级置信度仍能保持策略收益，说明策略改进不能证明模型具备危险感知能力。

## 研究问题与动机
- **核心问题**：安全强化学习中广泛使用的冻结 VLM 相似度评分（作为 cost/reward/confidence）是否真的测量了"危险感知"，还是仅与场景特征相关？
- **现有方法不足**：当前研究仅通过策略 return 和碰撞率验证安全信号有效性，但这些指标无法区分"真正感知危险"与"对失败相关特征的相关性响应"。
- **理论缺口**：prompt-margin 构造天然抵消了所有标题的共同方向，但此前未系统检验该评分的实际信息来源。
- **评估漏洞**：Safety-Gymnasium 和 MetaDrive 等基准存在未察觉的缺陷，可能导致先前结论无效。

## 核心贡献（创新点）
1. **被动观测审计框架**：首次将评分器与训练策略解耦，让策略从未见过安全分数，从而分离"分数测量什么"与"策略如何利用分数"。
2. **场景模板解释 vs 危险检测解释**：证明预碰撞偏差主要来自帧偏离标题共同场景（赛车在赛道上），而非接近危险物；提出可通过标题几何、视角控制、否定义本测试来区分二者。
3. **Caption 几何决定 danger 分数行为**：发现 ViT-B/32 中安全/危险组语义分离度极低（跨组余弦相似度 0.883 = 组内相似度），danger 分数上升与否取决于标题在嵌入空间的分离程度，而非评分器类别。
4. **视角依赖性实证**：证明预碰撞信号完全依赖于相机视角（FormulaOne 中仅 egocentric 视图有效），且不同环境中最优视角不一致，无统一规律。
5. **常量置信度 placebo 实验**：在 MetaDrive-Hard 中证明固定 κ=0.850 与动态帧级 κ 在灾难率上无显著差异（p=0.71），策略改进不依赖帧级视觉信息。
6. **评估管道缺陷报告**：发现并公开 Safety-Gymnasium/MetaDrive 四大缺陷（场景别名、交通重随机化、硬件依赖回放、日志回报含 shaping bonus）。

## 方法详解
**评分器设计**：
- 使用冻结 CLIP ViT-B/32 编码 256×256 帧（内部 resize 到 224×224）
- 8 个标题分为 safe/danger 两组（各 4 个），计算平均余弦相似度 s₊ 和 s₋
- prompt margin: m = s₊ - s₋
- 部署置信度: κ = |2σ(100(m-0)) - 1|，其中 σ 为 sigmoid

**被动观测协议**：
- 10 个 FormulaOne 策略（3 种 constrained RL 算法），每策略 60 局确定性 episode
- 策略从没见过 κ，轨迹由 VLM-free 策略生成
- Contact onset 定义：连续 20 步无接触后首次出现正环境成本
- 分析 lag -40 到 +10，聚焦 -15/-10/-5 的预碰撞偏差

**匹配控制**：
- 非参数匹配：按最近危险距离、方位角、 frontal 距离、frontal TTC 划分几何单元
- 每预碰撞观测匹配同单元内无接触 placebos
- Episode-cluster bootstrap 计算置信区间

**caption 几何控制**：
- 遍历所有 C(8,4)=70 种平衡划分
- 仅用 8 个标题均值方向（无 contrast）的得分 retain 73% 原始效应
- 否定 safe 标题、无关场景标题、荒谬标题等改写测试

**视角控制**：
- 同一 simulator state 渲染四种相机视图（egocentric/chase/overhead near/far）
- 跨环境验证：FormulaOne、MetaDrive、CarButton1、PointGoal1

## 实验与结果
**主要数据集/环境**：
- FormulaOne（10 策略，178 episode，130 isolated contact onsets）
- MetaDrive-Hard（600 episode，416 onsets）
- CarButton1（506 onsets）、PointGoal1（156 onsets）

**基线方法**：
- 三种 constrained RL 算法：PPO-Lagrangian、CPO、PID-Lagrangian
- 评分器：CLIP ViT-B/32、CLIP ViT-L/14、Qwen2-VL-7B
- 对比基线：pixel variance、edge density、k-NN、Mahalanobis、PCA residual、RND

**核心结果**：
- κ 在 contact 前约 20 步出现负偏差，持续约 -0.16（within-episode SD 的 -0.8 倍）
- **几何匹配后偏差仍存**：五种匹配规格下偏差 -0.131 至 -0.153，保留 98-100% 样本
- **危险距离调制反向**：远危险时偏差 -1.37 SD，近危险时仅 -0.34 SD（含零区间）
- **视角依赖**：仅 egocentric 视图有显著偏差 -0.447 [-0.706, -0.204]；chase -0.140、overhead near -0.080、far -0.049
- **Caption 语义**：否定 safe 标题保持偏差 -0.507，无关场景标题反转至 +0.441
- **Caption 分离度**：ViT-B/32 跨组相似度 0.883 = 组内 0.883；ViT-L/14 跨组 0.794 < 组内 0.808/0.845
- **封闭环实验**（MetaDrive-Hard）：no VLM 灾难率 0.281，deployed 0.192，constant κ=0.850 为 0.176，yoked 0.210；constant vs deployed 差 0.016（p=0.71，无显著差异）

## 相关工作脉络
- **Rocamonde et al. (2024)**：CLIP 相似度训练 MuJoCo humanoid，属 end-to-end 验证，未审计分数语义。
- **Huang et al. (2025) / Tetteh & Fleming (2026)**：VLM-RL 框架将 CLIP 项路由到 constrained learner 的对偶变量，本文审计其评分机制。
- **Yang et al. (2021)**：文本约束解释器针对环境成本训练，与本文冻结信号形成对比。
- **Ming et al. (2022)**：零样本 OOD 检测标准构造（取 max over concepts），本文指出安全评分行为符合此文献而非安全文献假设。
- **Mert Yuksekgonul et al. (2023)**：CLIP 类似 bag-of-words、对否定不敏感等表示分析，本文 negation/off-topic 控制验证这些性质出现在安全评分中。
- **Gao et al. (2023)**：proxy 优化退化理论，本文关注点在优化前代理机制已被误识。
- **Dean et al. (2020)**：控制理论安全证书需状态估计和误差界，本文审计的 prompt-margin 不满足此要求。

## 局限性与未来方向
- 视角结果仅在 FormulaOne 中为 egocentric-only，CarButton1 中三视图有效，PointGoal1 仅 overhead 有效，无统一预测规则。
- 仅测试三种评分器（两对比一生成），不足以表征 VLM 家族。
- Caption/几何分层分析仅在 egocentric 帧和一个 placebo draw 下完成（虽有 200 redraws 稳定性报告）。
- 封闭环实验仅覆盖一个环境、一个算法、一类 placebo。
- ViT-L/14 的 danger 分数上升机制未在机制控制中完整测试（控制实验用 ViT-B/32）。
- 未来方向：建立 caption 几何与危险感知能力的定量关系；开发视角不变的安全评分；将机制审计纳入 VLM-safe RL 标准协议。

## 研究启发与可借鉴点
1. **被动观测审计设计**：将评分器与策略解耦，是验证"信号语义"而非"信号效用"的黄金标准，可迁移至任何 VLM-as-feedback 系统。
2. **Caption 几何量化**：用跨组/组内余弦相似度、centroid 距离等指标预判安全评分的 contrast 能力，避免部署低分离度标题库。
3. **常量置信度 placebo**：简单但有力的对照实验，可分离"时间平滑性/分布匹配"与"帧级视觉信息"的贡献。
4. **几何匹配协议**：非参数 cell 匹配控制 hazard distance/bearing/TTC，适用于任何预事件信号分析。
5. **评估管道自检清单**：场景别名、随机性确定性、硬件依赖、日志正确性应作为基准测试的前置检查项。

## 关键术语表
- **Prompt-margin safety score**：安全/危险标题组平均余弦相似度的差值 m = s₊ - s₋，常被解释为危险相对证据。
- **Scene-template account**：评分变化源于帧偏离标题共享场景（如赛车在赛道），而非检测到具体危险物。
- **Passive observer**：策略从没见过安全分数，轨迹由 VLM-free 策略生成，用于审计分数测量什么。
- **Contact onset**：连续 20 步无接触后首次出现正环境成本的转换事件。
- **Caption geometry**：标题嵌入在文本空间的分布特性，包括组内/跨组相似度、centroid 分离度等。
- **Closed-loop placebo**：在封闭环训练中用常量/错位分数替代帧级分数，检验策略改进是否依赖视觉内容。
- **Episode-cluster bootstrap**：以 episode 为聚类单元的 bootstrap 置信区间估计方法。
- **Constrained MDP**：带成本预算约束的马尔可夫决策过程，安全 RL 的标准建模框架。

## 可复现要素
- **数据集**：FormulaOne、MetaDrive、Safety-Gymnasium（CarButton1、PointGoal1），均为开源环境。
- **代码**：论文未明确声明代码开源，但提供了完整附录和缺陷报告。
- **权重**：CLIP ViT-B/32、CLIP ViT-L/14、Qwen2-VL-7B 均为公开预训练模型。
- **关键超参**：κ 计算中 a=100, c=0；fp32 精度；256×256 渲染分辨率；lag 范围 -40 到 +10。
