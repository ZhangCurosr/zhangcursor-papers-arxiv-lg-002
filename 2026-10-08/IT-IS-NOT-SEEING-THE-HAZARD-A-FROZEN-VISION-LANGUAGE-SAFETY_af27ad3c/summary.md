---
title: "IT-IS-NOT-SEEING-THE-HAZARD-A-FROZEN-VISION-LANGUAGE-SAFETY"
source: https://arxiv.org/pdf/2610.09517v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 02:57:05"
---

# 论文速读：IT-IS-NOT-SEEING-THE-HAZARD-A-FROZEN-VISION-LANGUAGE-SAFETY

## 一句话总结
本文对冻结的 CLIP 提示边距（prompt margin）安全得分进行被动观测审计，证明碰撞前约 20 步的得分下降主要源于画面偏离标题库共享的场景模板（如赛道背景），而非真正检测障碍；闭环实验中即使将置信度固定为常数仍可维持性能提升，表明策略增益不能等同于 hazard 感知。

## 研究问题与动机
- **核心问题**：冻结 VLM 安全得分（图像-文本相似度/对比边距）究竟在测量“危险是否存在”，还是在无意识响应与标题场景共变的视觉特征？
- **现有方法不足**：Safe RL 广泛将 VLM 相似度转换为 reward/cost/confidence，但仅凭策略回报与碰撞率无法区分“真感知”与“相关性伪信号”。
- **评估缺失**：缺乏将得分与策略训练过程隔离的审计设计，未控制标题几何、相机视角、障碍距离等混淆变量。
- **工程隐患**：主流 Safe RL 评测栈（Safety-Gymnasium、MetaDrive、OmniSafe）存在隐蔽缺陷，可能导致绝对指标与跨实验比较失效。

## 核心贡献（创新点）
- **提出被动观测审计框架**：策略完全不接收 VLM 得分，首次严格分离“得分测量什么”与“策略如何利用它”，避免将相关性误判为因果性。
- **四重机制控制验证场景模板假说**：通过穷举标题分组、否定/无关重写、多视角渲染、恒定置信度对照，证明得分主要追踪 caption 共享方向（保留 73% 效应），而非安全/危险语义对比。
- **闭环安慰剂证据**：常数 $\kappa$ 仍可显著降低 catastrophe rate，表明政策增益不依赖帧级视觉感知，挑战了“VLM-as-safety-detector”的默认假设。
- **揭露评测链路四项系统性缺陷**：记录场景索引别名、交通随机化、硬件依赖回放、shaping bonus 污染返回等 bug，为 Safe RL 基准复用提供修正指南。

## 方法详解
- **评分器构造**：冻结 CLIP ViT-B/32（对照 ViT-L/14 与 Qwen2-VL-7B）对每帧编码，计算与 4 条 safe / 4 条 danger caption 的余弦相似度均值 $s_+$ 与 $s_-$，输出 margin $m = s_+ - s_-$；置信度 $\kappa = |2\sigma(100(m-c))-1|$（$c=0$）。
- **被动观测协议**：策略由 VLM-free 约束 RL 算法训练且永不接收 $\kappa$；以 contact onset（连续 20 步 contact-free 后首次正 cost）为对齐点，分析 lag $[-40, +10]$ 的偏差，预接触汇总取 $[-15, -10, -5]$。
- **几何匹配控制**：非参数匹配将预接触观测与同 episode 的 contact-free 观测按最近障碍距离、方位角、前置距离、前置 TTC（$\pm45^\circ$ 锥内）划分单元，验证信号是否真正依赖 hazard proximity。
- **视角隔离控制**：同一模拟器状态通过 4 种相机（egocentric、chase、overhead near/far）渲染，固定轨迹与接触序列，仅改变视图以测试 observation map 的影响。
- **标题几何控制**：穷举 $\binom{8}{4}=70$ 种平衡分组；重写 safe 组为否定句、荒谬句、无关场景句；测量 text embedding 空间的跨组/组内相似度（ViT-B/32 跨组 0.883 ≈ 组内 0.883/0.898）。
- **闭环对照设计**：MetaDrive-Hard 中训练 PPO-Lagrangian，四臂对比 no-VLM、部署得分、常数 $\kappa=0.850$、yoked（回放另一 episode 得分轨迹），每 seed 200 deterministic episodes，以种子为 bootstrap 单元。

## 实验与结果
- **数据集/环境**：FormulaOne（主实验，10 policies / 3 algorithms，180 episodes，130 isolated onsets）、MetaDrive-Hard（闭环）、CarButton1、PointGoal1。
- **预接触偏差稳健存在**：全样本偏差约 $-0.160$（≈ -0.8 within-episode SD），经五组几何匹配后稳定在 $-0.131 \sim -0.153$，排除样本衰减干扰。
- **反直觉的障碍距离效应**：障碍远时偏差最强（$-1.369$），近时不显著（$-0.342$，区间含 0）；前置 $\pm45^\circ$ 无障碍时偏差更强（$-1.797$），与 hazard-detection 预测相反。
- **标题几何主导信号**：共享方向（mean of all 8 captions）保留 73% margin 效应；否定 safe 标题几乎不变（$-0.507$ vs $-0.527$，相关系数 0.951）；无关场景标题直接反转符号（$+0.441$）。
- **视角高度依赖环境**：仅 FormulaOne 的 egocentric 视图有显著偏差（$-0.447$）；MetaDrive 顶视无偏差；PointGoal1 仅 overhead 有效；CarButton1 三视图均有偏差。无跨环境一致视角。
- **编码器与标题分离度耦合**：ViT-B/32 因 safe/danger 组近乎共线，两组得分同时下降；ViT-L/14 分离更好（跨组 0.794），danger 分数显著上升，证明 danger-score 变化由 text-space 几何决定而非 scorer 类别。
- **闭环常数置信度不劣于动态得分**：常数臂 catastrophe rate 0.176 vs 部署臂 0.192，差异仅 0.016（$p=0.71$）；常数臂相比 no-VLM 降低 0.106（$p=0.036$），但低于设计最小可检测效应，表明增益不依赖帧级视觉信息。
- **评测栈四项缺陷**：D1 场景索引模 $10^4$ 运算使 held-out 与训练集别名；D2 每次 reset 重随机化交通导致配对回放不等价；D3 确定性回放仅在生成硬件组可复现；D4 记录回报含 99.1%–99.8% shaping bonus，非真实环境性能。

## 相关工作脉络
- **VLM 作为 RL 奖励/成本信号**（Rocamonde 2024; Huang 2025; Sontakke 2023）：本文揭示此类方法依赖的“相似即感知”假设缺乏机制验证，性能增益可能来自场景模板匹配而非 hazard 检测。
- **零样本 OOD 检测与 caption bank**（Ming 2022）：本文证明安全得分的行为更符合 OOD 文献中“最大概念匹配”范式，而非安全文献假设的“危险语义判别”。
- **CLIP 表示特性研究**（Yuksekgonul 2023; Tong 2024; Alhamoud 2025）：引用 CLIP 类似 bag-of-words、对 negation 不敏感、多视角一致性有限等性质，支撑 scene-template account。
- **约束 RL 与代理优化理论**（Gao 2023; Ng 1999; Achiam 2017）：本文形式化证明 epoch-mean 成本项若与真实成本 affine，仅重调 budget 而不增 statewise 信息，为平均型 VLM 信号设置失效模式。
- **Safe RL 评测基准**（Ji 2023; Li 2022; Ray 2019）：本文系统审查并揭露评估流水线中的隐蔽 bug，强调基准复现与绝对指标报告需严格校验。

## 局限性与未来方向
- 结论局限于已测试配置（CLIP ViT-B/32/L-14、Qwen2-VL、特定相机/环境），无法推广至所有 VLM 安全信号。
- egocentric 优势仅在 FormulaOne 成立，CarButton1/PointGoal1/MetaDrive 结果不一致，无法预先预测何种 observation map 携带信号。
- 闭环实验仅覆盖单一环境、单一算法与一类安慰剂（常数/yoked），机制结论在更广泛设置下仍需验证。
- 障碍可见性假
