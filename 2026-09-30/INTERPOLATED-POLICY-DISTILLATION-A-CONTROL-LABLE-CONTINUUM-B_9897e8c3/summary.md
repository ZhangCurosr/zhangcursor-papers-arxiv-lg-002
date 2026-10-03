---
title: "INTERPOLATED-POLICY-DISTILLATION-A-CONTROL-LABLE-CONTINUUM-B"
source: https://arxiv.org/pdf/2609.37170v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 02:50:15"
field: "大模型知识蒸馏与策略优化"
keywords: ["policy distillation", "on-policy distillation", "speculative decoding", "interpolated policy", "knowledge distillation", "RLHF", "trajectory quality", "learnability"]
innovations: ["提出策略连续体（policy continuum），将 off-policy 与 on-policy 蒸馏统一为可由 gamma 精确控制的 token 级分布插值", "基于推测解码的精确采样规则，在不损失分布精度的前提下将 teacher 查询从每 token 降至每 block 一次，并利用 accept/reject 决策天然产生差异化监督标签", "在 rollouts 内融合 OPD clipped surrogate 与 CE likelihood 两种监督信号，实现 SFT 与 OPD 的 unified 单阶段训练"]
benchmarks: ["GSM8K", "MATH-500", "AMC 2023", "OlympiadBench", "AIME 2024-2026", "MathVista", "MMStar", "WeMath", "MMMU-Pro", "MathVision"]
---

# 论文速读：INTERPOLATED-POLICY-DISTILLATION-A-CONTROL-LABLE-CONTINUUM-BETWEEN-OFF-POLICY-AND-ON-POLICY-DISTILLATION

## 一句话总结
本文提出插值策略蒸馏（Interpolated Policy Distillation, IPD），将无策略（off-policy）蒸馏与自策略蒸馏（on-policy distillation, OPD）视为一个策略连续体的两个端点，通过在每一token的分布层面进行线性插值构建可控的中间 rollout 策略，并结合推测解码加速采样与差异化监督，在文本与多模态推理基准上持续超越两种端点策略及其组合方法。

## 研究问题与动机
- **核心张力**：teacher 生成的轨迹质量高但分布偏移大（train–inference mismatch），student 自生成的轨迹可学性强但弱 student 易陷入错误推理路径（agreement trap）。
- **现有方法的不足**：
  - Off-policy（如 SFT）依赖 teacher 轨迹，student 难以吸收；
  - Vanilla OPD 完全 student 采样，早期错误会累积并导致全局失败；
  - 近期混合方法（SKD、Relay-OPD、CA-OPD）通过手工设计的切换规则实现学生–教师段的交错，分布特性不明确且对训练设置敏感（Relay-OPD 在弱 student 设定下出现训练崩塌）。
- **关键科学问题**：能否构造一个位置明确可控、介于两者之间的 rollout 策略，从而直接权衡轨迹质量（trajectory quality）与可学性（learnability）？

## 核心贡献（创新点）
- **贡献 1**：提出"策略连续体"（policy continuum）统一视角，将 off-policy 和 on-policy 蒸馏形式化为同一连续体的两个端点，中间位置由系数 $\gamma \in [0,1]$ 精确控制。与以往将二者视为独立范式的工作不同，本文明确暴露中间策略作为设计空间的系统性价值。
- **贡献 2**：设计插值策略蒸馏（IPD），在每个解码步以 $m_\gamma = (1-\gamma)\pi_\theta + \gamma \pi_T$ 定义 next-token 分布，并通过推测解码加速采样——学生作为 proposal、插值策略作为 verifier——在不损失分布精度的前提下将 teacher 查询从每 token 降为每 block 一次。与启发式段交错方法本质区别在于：IPD 先定义目标分布，再推导保持该分布的采样过程，而非手工设计干预规则。
- **贡献 3**：基于推测解码的接受/拒绝决策，在同一 rollout 内天然区分两类 token 并施加差异化监督：接受的学生提案用 OPD 的 clipped surrogate objective，被校正的教师引导 token 用有界梯度的 CE likelihood objective，实现 SFT 式与 OPD 式监督的统一融合。
- **贡献 4**：在文本推理（Qwen3-0.6B/1.7B←4B、DeepSeek-R1-Distill-Qwen-1.5B←Skywork-OR1-7B）与多模态推理（Qwen3-VL-2B←8B，含剪枝恢复设定）多个基准上系统验证，IPD 一致超越两种端点策略、两阶段 SFT-then-OPD、以及 SKD 和 Relay-OPD 等近期方法。

## 方法详解

### 1. 策略连续体的形式化定义
在每个解码步 $t$，给定前缀 $s_t = (x, y_{<t})$，定义插值 next-token 分布：
$$m_\gamma(\cdot|s_t) = (1-\gamma)\pi_\theta(\cdot|s_t) + \gamma\pi_T(\cdot|s_t), \quad \gamma \in [0,1]$$
轨迹分布为逐 token 条件分布的乘积：$P_\gamma(y|x) = \prod_{t=1}^{|y|} m_\gamma(y_t|s_t)$。**注意**：这是对条件分布的逐 token 插值，而非对完整轨迹分布的线性混合。

$\gamma=0$ 退化为 vanilla OPD，$\gamma=1$ 退化为纯 off-policy（teacher 采样）。

### 2. 总变异距离（TV distance）的可解释性
$$D_{\mathrm{TV}}(m_\gamma, \pi_\theta) = \gamma D_{\mathrm{TV}}(\pi_\theta, \pi_T), \quad D_{\mathrm{TV}}(m_\gamma, \pi_T) = (1-\gamma)D_{\mathrm{TV}}(\pi_\theta, \pi_T)$$
两距离之和为 $D_{\mathrm{TV}}(\pi_\theta, \pi_T)$，且均关于 $\gamma$ 线性。因此 $\gamma$ 将 rollout 策略置于两个端点之间**精确已知**的位置。

### 3. 推测解码加速（Section 3.2）
- 学生（proposal）自回归地起草 $k$ 个 token；
- 单次 teacher 前向传播对整 block 打分（verifier）；
- 每个被提议 token $v$ 的接受概率：
  $$a_\gamma(v|s) = \min\left\{1, \frac{m_\gamma(v|s)}{p(v)}\right\} = \min\left\{1, 1-\gamma + \gamma\frac{q(v)}{p(v)}\right\}$$
- 首个被拒绝的 token 被替换为来自残差分布的校正 token：
  $$r_\gamma(v|s) = \frac{[m_\gamma(v|s)-p(v)]_+}{\sum_u[m_\gamma(u|s)-p(u)]_+} = \frac{[q(v)-p(v)]_+}{D_{\mathrm{TV}}(p,q)} = r_1$$
  关键性质：**$\gamma$ 只控制干预频率，不改变干预方向**——校正分布与直接用 teacher 作为 verifier 时相同。
- 拒绝概率：$\Pr(z_t=0|s_t=s) = \gamma D_{\mathrm{TV}}(p,q)$，期望连续接受长度 $\sim 1/(\gamma D_{\mathrm{TV}}(p,q))$，小 $\gamma$ 时 teacher 查询显著减少。
- **分布精确性证明**（Appendix A.1）：接受概率 $p(v)a_\gamma(v) + [m_\gamma(v)-p(v)]_+ = m_\gamma(v)$，逐 token 精确匹配 $m_\gamma$，进而由链式法则保证完整轨迹分布精确等于直接采样 $m_\gamma$ 的分布。

### 4. 差异化监督（Verification-aware Supervision, Section 3.3）
- 用 $z_t \in \{0,1\}$ 记录第 $t$ 位是接受的学生提案（$z_t=1$）还是教师校正（$z_t=0$）；
- **接受提案**（保留 on-policy 特性）：使用 $k_1$ 估计器近似 reverse-KL 作为 fixed advantage：
  $$\widehat{A}_t^{\mathrm{KD}} = \log\pi_T(y_t|s_t) - \log\pi_{\theta_{\mathrm{old}}}(y_t|s_t)$$
  并优化 clipped surrogate：
  $$\ell_t^{\mathrm{acc}}(\theta) = -\min\left\{\rho_t(\theta)\widehat{A}_t^{\mathrm{KD}},\ \mathrm{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\widehat{A}_t^{\mathrm{KD}}\right\}$$
  其中 $\rho_t(\theta) = \pi_\theta(y_t|s_t)/\pi_{\theta_{\mathrm{old}}}(y_t|s_t)$。
- **校正 token**（student 低估的 token）：用 CE 直接 likelihood 监督，梯度有界：
  $$\ell_t^{\mathrm{rej}}(\theta) = -\log\pi_\theta(y_t|s_t)$$
- **总体目标**：
  $$\mathcal{L}_{\mathrm{IPD}}(\theta) = \frac{1}{|\mathcal{T}|}\sum_{t \in \mathcal{T}}\left[z_t \ell_t^{\mathrm{acc}}(\theta) + (1-z_t)\ell_t^{\mathrm{rej}}(\theta)\right]$$
- $\gamma=0$ 时所有 token 均为接受提案，目标退化为 vanilla OPD；$\gamma$ 增大时校正 token 增多，CE 监督占比上升。

### 5. 稳定性分析（Appendix A.2）
- IPD 对接受 token 的 $k_1$ 信号强度不高于 vanilla OPD（被 $\gamma$ 衰减正部）；
- 校正 token 的 CE 梯度范数期望 $\leq \sqrt{2}\gamma$，提供显式的梯度有界性保障。

## 实验与结果

### 文本推理（Table 1）
- 设定 1：Qwen3-1.7B-Base ← Qwen3-4B，数据集 OpenThoughts3（25,600 道数学题），评测 GSM8K、MATH-500、AMC23、OlympiadBench、AIME 2024–2026。
  - IPD Avg@32 = **16.64**，优于 vanilla OPD（14.71，+1.93）、Relay-OPD（15.67，+0.97）。
- 设定 2：Qwen3-0.6B-Base ← Qwen3-4B。
  - IPD Avg@32 = **38.47**，优于 vanilla OPD（27.83，+10.64）、Relay-OPD（36.96，+1.51）。
- 强 student 设定（DeepSeek-R1-Distill-Qwen-1.5B ← Skywork-OR1-7B）：IPD Avg@32 = **42.18** vs OPD 41.37（+0.81），五基准均提升。

### 多模态推理（Table 2）
- 设定 1：Qwen3-VL-2B-Instruct ← Qwen3-VL-8B，训练集 Innovator-VL-RL-172K，评测 MathVision、MathVista、MMStar、WeMath、MMMU-Pro。
  - IPD Avg@32 = **50.22** vs OPD 48.48；长训时 OPD 出现渐进退化而 IPD 保持稳定收敛。
- 设定 2（更具挑战性）：剪去 Qwen3-VL-2B 最后三层 decoder 后做蒸馏恢复。
  - IPD Avg@32 = **44.64** vs OPD 27.90（+16.74），接近未剪枝模型原始性能 45.87。

### 消融（Section 4.3 / Appendix A.6/A.7）
- **$\gamma$ 调参**：在 GSM8K/MATH-500/AMC 上呈现中间最优，$\gamma=0.1$ 最佳，$\gamma=0.05$ 也较好；$\gamma=1$ 虽仍优于 vanilla OPD 但远逊于 0.1。
- **$\gamma$ 调度**：从 0.5 递减至 0.1 优于从 0.5 递增至 0.9，支持训练后期减少 teacher 依赖。
- **SFT cold start 敏感性**：IPD 对冷启动几乎不敏感，即使零冷启动也已超过 SFT+OPD 的两阶段最佳（46.69 vs 45.41）；且 IPD 内部 SFT 式监督 token 占比不足 1%，却优于 60× 预算的 SFT+OPD（Figure 8），说明 rollouts 内融合比两阶段串联更高效。
- **Relay-OPD 失败案例**（Appendix A.4）：在弱 student 设定下 30 步内 policy entropy 从 2.659 坍缩至 0.005，长度截断率升至 95.3%，teacher 控制 token 比降至 0.02%，暴露其对训练设置的极端敏感性。

## 相关工作脉络
- **Off-policy SFT / 序列级 KD**（Hinton et al., 2015; Kim & Rush, 2016; Shridhar et al., 2023）：使用 teacher 生成轨迹，训练–推理分布偏移。IPD 将其视为 $\gamma=1$ 端点。
- **On-policy Distillation (OPD)**（Agarwal et al., 2024; Lu & Lab, 2025）：使用 student 自生成轨迹，消除分布偏移但质量无保障。IPD 将其视为 $\gamma=0$ 端点，并提出中间策略可兼顾两者。
- **SKD**（Xu et al., 2025）：将 student 提案在外层替换为 teacher top-K 预测。IPD 定位差异：SKD 使用隐式分布与手工规则，IPD 先定义显式目标分布 $m_\gamma$ 再推导采样。
- **Relay-OPD**（Xu et al., 2026）：检测失败前缀后临时移交 teacher 生成。IPD 定位差异：Relay-OPD 的干预触发依赖启发式规则，在弱 student 下易崩塌；IPD 的干预频率由 $\gamma D_{\mathrm{TV}}(p,q)$ 精确刻画。
- **CA-OPD**（Li et al., 2026a）：按 teacher 置信度校正不可靠 token 并渐进放松干预。IPD 定位差异：CA-OPD 的干预量由置信度启发式决定；IPD 通过 TV 距离与 $\gamma$ 提供理论可解释的干预控制。
- **MInTRL**（Chen et al., 2026，RL 设定）：在 RL 中插入 judge 识别的短校正。IPD 定位差异：MInTRL 依赖外部 judge 且仅在 RL 框架内；IPD 为纯蒸馏框架，干预由 token 级别分布插值自然导出，无需额外裁判。

## 局限性与未来方向
- **$\gamma$ 敏感性与调度依赖**：当前最优 $\gamma=0.1$ 需在特定设定下实验确定；虽然递减调度表现更优，但缺乏自动搜索或自适应机制。
- **SFT 冷启动的作用未完全厘清**：在弱 student 设定下 IPD 对冷启动不敏感，但在更大 teacher–student 差距或其他领域是否仍需冷启动，作者明确标注为 future work。
- **仅验证推理类任务**：实验集中于数学/多模态推理，对生成质量、对话、代码等其他任务领域的泛化未评估。
- **推测解码的 block 长度 $k$ 未系统消融**：论文默认配置未详述 $k$ 选择依据及其对速度与质量的权衡。
- **多教师/多 student 扩展未涉及**：方法目前假设单一 teacher–student 对，对模型集成或多教师场景的推广未讨论。

## 研究启发与可借鉴点
- **"分布级别先定义、采样过程后推导"的设计范式**：本文先给出目标分布 $m_\gamma$，再从该分布推导出保持精确性的推测解码采样规则，而非先设计启发式规则再事后分析其分布性质。这种"目标先行"的思路可迁移至其他需要平衡多源分布的场景（如多教师蒸馏、RLHF 中的混合策略）。
- **差分监督的天然标签来源**：推测解码的 accept/reject 决策本身就携带了 token 级别的语义标签（学生可信 / 需教师校正），无需额外标注即可实现差异化目标。这一思路可扩展至其他使用 speculative decoding 的训练框架。
- **TV 距离作为干预强度的可解释度量**：$\Pr(\text{reject}) = \gamma D_{\mathrm{TV}}(p,q)$ 提供了 teacher 干预频率的闭式表达，使超参 $\gamma$ 具有明确的统计意义，优于黑箱式调参。类似的可解释关系可在其他策略插值方法中探索。
- **弱 student / 剪枝恢复场景的价值**：IPD 在剪枝后 student 的恢复任务上取得 +16.74 的巨大提升，提示该方法在"能力恢复"与"低资源蒸馏"场景下具有显著应用潜力，值得在更多模型压缩与微调场景中验证。
- **$\gamma$ 递减调度的直觉**：训练初期依赖较多 teacher 引导保证轨迹质量，后期逐渐回归 student 策略以保持 learnability。这一"annealing  toward on-policy"的调度思想可与 TRPO/PPO 中的 trust region 收缩策略互相启发。

## 关键术语表
- **Interpolated Policy Distillation (IPD)**：在每一解码步将学生与教师的 next-token 分布线性插值，并通过推测解码高效采样、差异化监督完成蒸馏的方法。
- **Policy Continuum（策略连续体）**：以 $\gamma \in [0,1]$ 为参数的 rollout 策略家族，$\gamma=0$ 为纯 on-policy（student），$\gamma=1$ 为纯 off-policy（teacher），中间值提供可控的质量–可学性权衡。
- **Speculative Sampling from $m_\gamma$**：以学生分布为 proposal、插值分布为 verifier 的推测采样，接受概率 $a_\gamma = \min\{1, m_\gamma/p\}$，首个拒绝后的校正来自残差分布 $r_\gamma$，保证输出分布精确等于 $m_\gamma$。
- **Verification-aware Supervision**：根据推测解码的 accept/reject 决策，对接受 token 施加 OPD 式 clipped surrogate loss，对校正 token 施加有界梯度的 CE likelihood loss。
- **$k_1$ Estimator**：Schulman (2020) 提出的单样本 reverse-KL 无偏估计，用于 OPD 中计算 per-token advantage $\widehat{A}_t^{\mathrm{KD}}$。
- **Total Variation (TV) Distance**：$D_{\mathrm{TV}}(p,q) = \frac{1}{2}\sum_v|p(v)-q(v)|$，此处用于精确刻画插值策略与两端的距离以及拒绝概率。
- **Agreement Trap**（共识陷阱）：vanilla OPD 中 student 与 teacher 局部 token 同意度高但全局推理失败的现象，源于 weak student 的错误前缀导致后续路径系统性偏离。
- **Residual Correction Distribution**：$r_\gamma(v|s) = [q(v)-p(v)]_+ / D_{\mathrm{TV}}(p,q)$，在校正时使用的条件分布，与 $\gamma$ 无关，仅由学生–教师分布的正部差异决定。

## 可复现要素
- **数据集**：OpenThoughts3（数学题，25,600 条，抽样）、Innovator-VL-RL-172K（多模态）；评测基准 GSM8K、MATH-500、AMC 2023、OlympiadBench、AIME 2024–2026、MathVision、MathVista、MMStar、WeMath、MMMU-Pro。**公开可获取**。
- **模型**：Teacher: Qwen3-4B / Qwen3-VL-8B-Instruct / Skywork-OR1-7B；Student: Qwen3-0.6B-Base / Qwen3-1.7B-Base / Qwen3-VL-2B-Instruct（含剪枝变体）/ DeepSeek-R1-Distill-Qwen-1.5B。**开源模型**。
- **代码/权重**：论文 Reproducibility Statement 中仅说明方法细节与实验设置在 Appendix 中完整给出，**未明确声明代码与权重是否开源**，标注为"论文未提及"。
- **关键超参**：
  - $\gamma = 0.1$（默认，消融涵盖 0.05/0.5/0.9/1.0）
  - Batch size: 32（文本）/ 256（多模态）
  - Rollouts per prompt: 4（文本）/ 1（多模态）
  - Max prompt/response length: 1024/1024（文本）/ 4096/4096（多模态）
  - Learning rate: $5\times10^{-7}$（文本）/ $10^{-6}$（多模态）
  - Steps: 800（文本）/ 600（多模态）
  - Temperature=1.0, Top-p=0.95, Top-k=20
  - Warmup ratio=0.05, Weight decay=0.01, Gradient clipping=1.0
  - $k_1$ 估计器用于 accepted token advantage 计算；clipping 系数 $\epsilon$ **论文未明确给出数值**。
  - 推测解码 block 长度 $k$ **论文未明确给出数值**。
  - 评估时 temperature=0.6, max generation=8192, top-p=0.95, top-k=20（vLLM）。
