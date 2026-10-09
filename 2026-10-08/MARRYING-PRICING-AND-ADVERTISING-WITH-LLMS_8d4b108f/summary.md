---
title: "MARRYING-PRICING-AND-ADVERTISING-WITH-LLMS"
source: https://arxiv.org/pdf/2610.09985v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:52:13"
field: "大模型驱动的在线定价与广告联合优化"
keywords: ["sequential pricing", "LLM advertisement generation", "actor-critic", "low-rank adaptation", "online learning", "revenue optimization", "KL-regularized policy gradient"]
innovations: ["actor-critic 联合优化 LLM 广告生成与序列定价", "用 critic 预期收益构建 leave-one-out 基线以降低二值反馈方差", "LoRA+KL正则在保持预训练语言能力的同时实现收益导向的广告文本自适应"]
benchmarks: ["Synthetic logit demand", "Synthetic probit demand", "Price disclosure demand (Dubé et al., 2026)", "Avito marketplace demand simulator"]
---

# 论文速读：MARRYING-PRICING-AND-ADVERTISING-WITH-LLMS

## 一句话总结
本文提出了一种在线 actor-critic 算法，将预训练 LLM 通过 LoRA 适配为广告生成器（actor），同时拟合一个需求模型作为 critic 来估计购买概率并指导定价；两者在只有二元购买反馈的序列交互中联合优化，实现了在不同合成需求模型及真实市场数据上分别获得 5.69%、5.18%、55.96% 和 5.81% 的预期收益提升。

## 研究问题与动机
1. **定价与广告不可分离优化**：传统在线定价研究仅关注价格对未知需求的学习，但大量实证（如 Bertrand et al., 2010）表明广告内容对需求的改变幅度可与价格降幅相当，二者必须联合优化。
2. **LLM 广告生成的反馈盲区**：现有 LLM 广告工作（如 Jiang et al., 2025; Chen et al., 2025; Wang et al., 2026）仅用点击/转化反馈优化文本生成，未将广告与价格联动进同一个收益优化循环。
3. **纯收益优化会破坏语言 fluency**：若仅以收益为目标做策略梯度更新，LLM 可能严重偏离预训练参考模型，丧失生成内容的流畅性与连贯性；需要 KL 正则项约束漂移。
4. **单一购买反馈信息匮乏**：卖家仅观察到是否成交（binary 信号），既不能区分"没买是因为价格高还是广告差"，也无法观测替代方案的反事实响应，学习信号高度稀疏且需同时泛化文本生成与价格选择两个高维决策空间。

## 核心贡献（创新点）
1. **首个将 LLM 广告生成与序列定价在 actor-critic 框架下联合优化的在线算法**：与仅优化广告或仅优化定价的既有工作本质不同，本文把两者耦合进同一迭代闭环，critic 的收益估计同时驱动定价与 actor 的 policy gradient。
2. **Leave-one-out  critic 基线替代观测收益基线**：actor 的策略梯度优势估计用 critic 预测的预期收益均值而非原始二元收益均值，显著压制 Bernoulli 购买噪声引入的方差；这与传统 REINFORCE-LNOO 用观测 reward 构建基线的做法存在本质差异。
3. **LoRA 适配下的 KL-正则化 actor 损失并给出无偏性证明**：actor 仅在低秩适配器参数上更新，损失包含收益优势项与对固定参考模型的 KL 惩罚项；附录 A 严格证明了该 surrogate loss 在批内定价策略固定的条件下是 KL-正则化目标梯度的无偏估计。
4. **面向真实市场需求的离线 critic 预训练 + 在线回放更新机制**：critic 先用 8,192 条离线 (广告, 价格, 购买) 三元组拟合需求曲线，在线阶段通过 replay buffer 持续用二元反馈微修，避免从零学习导致的初始探索效率低下。
5. **统一评估协议同时覆盖合成机理与真实 marketplace 迁移**：在 logit/probit/price-disclosure 三种可控合成模型下验证算法增益，并在 Avito 五类商品模拟器上验证未见产品的跨产品迁移收益，形成从机理到数据的完整证据链。

## 方法详解
- **问题设定**：每轮 $t$ 卖家观察到上下文 $z_t$（商品特征），actor 生成广告 $x_t \sim \pi_{\theta_t}(\cdot|c(z_t))$，pricing rule $q_t$ 选价 $p_t=q_t(x_t,z_t)$，购买信号 $y_t\sim\mathrm{Ber}(d(p_t,x_t,z_t))$，收益 $r_t=p_t y_t$。参数空间 $\Theta$ 限定为 $\theta_{\mathrm{ref}}+\Delta\theta(\psi)$ 的 LoRA 低秩扰动。
- **KL-正则目标**：$u_\beta(q,\theta|z)=\mathbb{E}_{x}[q(x,z)d(\cdot)]-\beta D_{\mathrm{KL}}(\pi_\theta||\pi_{\theta_{\mathrm{ref}}})$，$\beta>0$ 控制收益与参考分布的权衡。
- **Actor-Critic 循环（每批 $B$ 轮）**：
  1. Actor 按当前 $\theta_\tau$ 生成 $B$ 条广告。
  2. $\varepsilon$-greedy 定价：以 $1-\varepsilon_\tau$ 选 $\arg\max_p p\,\widehat{d}_{\phi_\tau}(p,x,z)$，以 $\varepsilon_\tau$ 从探索分布 $\nu_\tau$ 采样。
  3. 收集 $(x_{\tau,b},p_{\tau,b},y_{\tau,b},r_{\tau,b})$ 入 replay buffer。
- **Actor 更新（每 $K$ 批一次）**：leave-one-out 优势 $A_{\tau,b}=r_{\tau,b}-\frac{1}{B-1}\sum_{j\neq b}\widehat{u}_{\phi_\tau,j}$，其中 $\widehat{u}=p\cdot\widehat{d}_\phi$；actor 损失 $\mathcal{L}_{\mathrm{actor}}=-\frac{1}{B}\sum_b(A_{\tau,b}-\beta\ell_\tau(x_{\tau,b}))\log\pi_{\theta_\tau+\Delta\theta(\psi)}(x_{\tau,b}|c(z))$，Adam 更新 $\psi$。
- **Critic 更新（每批一次）**：从 buffer 采样 $S$ 条样本，损失为二元交叉熵加锚定正则 $\rho\|\phi-\phi_1\|_2^2$；critic 网络结构为两隐层 128 ReLU + sigmoid 输出。
- **无偏性结论**：附录 A 证明在固定 $\psi_\tau,\phi_\tau,\varepsilon_\tau,\nu_\tau$ 条件下，actor surrogate loss 负梯度在 $\psi_\tau$ 处的期望等于 KL-正则目标关于 $\psi$ 的梯度。

## 实验与结果
- **合成需求模型**：logit（系数 $(a,\kappa,w)=(1,4,0.8)$，特征为含 "durable" 词）、probit（校准到 logit 在三点匹配）、price disclosure（基于 Dubé et al., 2026 三类价格点插值）。每模型 8,192 条离线训练 critic，测试 10,000 条新生成广告。
  - Logit：Trained 0.2171 vs Reference 0.2054，**+5.69%**。
  - Probit：Trained 0.2056 vs Reference 0.1955，**+5.18%**。
  - Price disclosure：Trained 0.0272 vs Reference 0.0174，**+55.96%**（完全由披露价格特征频率从 35.17% 升至 98.23% 解释）。
- **真实 marketplace 模拟器（Avito）**：五类商品（Audio/Video、Home Appliances、Laptops、Mobile Phones、Tablets/E-readers），每类训练 1 产品、测试 20 产品，每产品 100 广告×两策略。
  - 平均归一化预期收益从 0.1991 升至 0.2107，**+5.81%**；91/100 产品收益提升；Laptops 最高 +14.26%，Mobile Phones 最低 +1.98%。
- **在线学习对比 UCB**：UCB 用固定参考 generator + 独立 arm 价格探索；Trained 在最后 100 批的滚动均值收益五类均高于 UCB。
- **离线数据消融**：critic 初始训练数据从 100% 降至 0% 时，Laptops 收益从 0.1802 缓降至 0.1670（-7.32%），说明算法在无离线监督时仍可学习但起点更低。
- **Leave-one-out 基线噪声诊断**：全数据下 critic 替代观测奖励的基线 MSE 降至观测基线的 0.17%–3.4%，零数据时退回 ~70–115 倍噪声。

## 相关工作脉络
1. **序列 posted pricing 与动态学习定价**：Kleinberg & Leighton (2003)、Besbes & Zeevi (2009)、Misra et al. (2019) 与 den Boer (2015) 综述——本文在其基础上引入广告文本作为另一决策维度。
2. **定价与广告交互的经济学与实验研究**：Dorfman & Steiner (1954) 理论分析与 Bertrand et al. (2010) 大规模随机对照实验——后者证实广告内容对需求的影响不弱于价格变动，构成本文联合优化的实证动机。
3. **Sequential pricing with signaling**：Agrawal et al. (2023) 研究 seller 学习需求同时设计信息披露机制——与本文不同，其聚焦信息揭示而非 LLM 自由文本生成与定价的联合参数化。
4. **RL 微调 LLM 生成广告**：Jiang et al. (2025)、Chen et al. (2025)、Wang et al. (2026)——三者仅优化广告侧奖励（点击率/转化率），未同时选择价格；本文补齐了 joint pricing-advertising 这一缺失环节。
5. **RLHF 与 KL 正则策略优化**：Ziegler et al. (2019) 提出 RLHF 框架与 KL 惩罚；本文将其移植到定价-广告联合序列决策，并把 KL 项嵌入 actor 的 policy gradient 损失。
6. **LoRA 参数高效微调**：Hu et al. (2022)——本文将其限定为 actor 的唯一可训练参数空间，保证参考模型权重冻结从而 KL 项可精确计算。
7. **REINFORCE leave-one-out 方差缩减**：Kool et al. (2019)、Ahmadian et al. (2024)——本文沿袭 LNOO 思路但把基线从观测 reward 替换为 critic 预期收益，进一步降噪；并给出新的无偏性定理。

## 局限性与未来方向
1. **上下文单产品假设**：当前固定单一 context $z$，未处理多商品 catalog 内跨产品竞争与预算约束场景。
2. **非平稳需求未建模**： buyer 偏好、价格敏感度可能随时间漂移，critic 需跟踪 moving target，本文假设固定需求。
3. **广告质量仅间接保障**：KL 正则仅约束分布漂移，未显式优化语言流畅度、事实一致性或品牌调性；可能生成语法正确但商业无效的内容。
4. **真实在线 A/B 验证缺失**：实验依赖合成机理与离线 marketplace simulator，尚未经线上真实 buyer 流量验证收益转化与长期生态效应。
5. **价格格点离散化**：101 等间距格点限制了价格选择的精细度，且在跨产品归一化后可能改变原始收益结构的相对尺度。
6. **future work 自述**：作者在结论中明确提出扩展至时变需求、critic 需跟踪漂移的目标。

## 研究启发与可借鉴点
1. **Leave-one-out critic 基线替代观测 reward 基线**：在 sparse binary feedback + 高方差 reward 的序列决策中，用值函数预期替代样本 reward 做 LNOO 基线是一个可直接复用的降噪技巧，适用于其他 LLM 策略梯度优化场景。
2. **KL 正则 + LoRA 适配的组合范式**：将参考模型冻结、仅在低秩适配器上优化并配套 KL 惩罚，既保留预训练语言能力又提供可微的收益优化路径，可迁移到任何"LLM 生成 + 环境反馈"的 online decision-making 任务。
3. **离线 critic 预训练 + 在线 replay 微调的两阶段训练**：先用高质量离线 (行为, 价格, 购买) 三元组冷启动需求模型，再以在线反馈持续校正，既缩短在线探索期又抑制 critic 震荡；该模式适合任何冷启动成本高的 online RL 系统。
4. **跨产品归一化价格空间的评估协议**：用 $\lambda(z)p$ 映射到统一 $[0,1]$ 动作空间，并在评估时除以 $\lambda(z)$ 汇报归一化收益，使不同价位商品公平可比——这一协议可直接复用到多品类电商定价研究中。
5. **算法-机理-模拟器三层验证链条**：从可控合成需求（可归因增益来源）到真实 marketplace simulator（跨产品泛化）再到 UCB 在线对比，形成互补的证据三角；该实验架构可作为后续工作的基准范式。

## 关键术语表
- **Sequential posted pricing**：卖家在多轮中反复向 arriving buyer 报价，仅观察是否成交，用以在学习与收益收集间 trade-off 的经典在线决策框架。
- **Actor-critic algorithm**：策略梯度方法的一种变体，actor 负责生成动作（此处为广告文本），critic 负责估计状态/动作价值（此处为购买概率与预期收益），二者交替更新。
- **Low-rank adaptation (LoRA)**：冻结预训练模型权重，仅在注意力矩阵上叠加低秩分解矩阵 $BA$ 进行参数高效微调，大幅降低训练参数量并保持参考分布可精确计算。
- **Leave-one-out baseline**：在策略梯度优势估计中，用当前批次除自身外其余样本的均值作为基线，以降低方差；本文用 critic 预期收益均值替代观测收益均值。
- **KL-regularized objective**：在原始收益目标上减去策略分布与参考模型分布的 KL 散度乘以系数 $\beta$，防止优化过程中广告文本分布过度漂移。
- **Demand model / critic**：以价格、广告嵌入和上下文为输入，输出购买概率的二分类网络，用于定价决策与 actor 优势的基线估计。
- **$\varepsilon$-greedy pricing rule**：以概率 $1-\varepsilon$ 选取 critic 估计收益最大的价格，以概率 $\vparameter\varepsilon$ 从探索分布 $\nu$ 随机采样，平衡开发与探索。
- **Revenue advantage**：单轮实现收益减去批内其余轮 critic 预期收益的均值，衡量该轮广告-价格组合相对于批内平均水平的相对优劣。

## 可复现要素
- **数据集**：合成需求数据由作者按 logit/probit/price-disclosure 模型采样生成，未公开原始合成脚本；真实数据基于 Avito marketplace listings（Avito, 2018 Kaggle 竞赛数据），已做英文翻译与清理，原始数据需通过 Kaggle 申请。
- **代码/权重**：论文未明确声明开源仓库，但 Appendix 提到 "exact probit specification can be found in the repository"，暗示有配套代码库；需查阅 arxiv 页面获取 URL。
- **关键超参**：Actor=Qwen2.5-1.5B-Instruct，LoRA rank=8、scaling $\alpha=16$，适应 query/value 投影共 28 层；温度=0.7，最大新 token=64；批次大小 $B=32$，总批数 $N=500$，actor 每 $K=5$ 批更新；critic replay 采样 $S=64$；actor lr=$10^{-4}$，critic lr=$3\times10^{-3}$；KL 系数 $\beta=10^{-3}$，critic 锚定系数 $\rho=10^{-4}$；价格格点 101 等分；critic 网络为 2 层 128 ReLU + sigmoid；Adam 优化器；离线训练 5 个 epoch 早停（以验证 BCE 为准）。
