---
title: "Personal-Agent-Mediated-Recommendation-with-Cross-Platform-U"
source: https://arxiv.org/pdf/2610.07588v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:55:10"
field: "个性化推荐与个人智能体"
keywords: ["personal-agent recommendation", "cross-platform history", "mediated ranking", "PAMO", "GRPO for recommendation", "rescue vs harmful override", "MediateRec"]
innovations: ["提出冻结平台提案 + 个人智能体选择性中介的任务与评测框架", "用反事实遮蔽估计跨平台依赖并作为归因信号", "在保留 cutoff 级优势质量与平均价值的前提下做支持度加权重分配"]
benchmarks: ["MediateRec-Movie", "MediateRec-Toy", "MediateRec-Grocery", "MediateRec-Beauty", "MediateRec-OpenPlay"]
---

# 论文速读：Personal-Agent-Mediated-Recommendation-with-Cross-Platform-U

## 一句话总结
论文将推荐范式形式化为“个人智能体中介推荐”：平台先基于本地数据给出候选排序，个人智能体再借助用户授权的跨平台历史对该排序进行选择性修正；为此提出 MediateRec 基准与 PAMO 训练方法，在保留平台强信号的同时更好地发挥跨平台历史的纠错价值。

## 研究问题与动机
- 平台推荐擅长利用群体协同信号，但个人智能体持有跨平台历史这一独占信息，理论上可弥补平台盲区。
- 直接修改平台排序会引入“救援收益 vs 有害覆盖”的权衡，强平台排序未必应被随意推翻。
- 现有推荐智能体评测多面向平台侧内部推理/工具调用，缺乏在“平台提议 + 个人跨平台历史”边界下的中介式评测。
- 仅用结果奖励训练难以区分“真正依赖跨平台历史的修正”与“通用重排成功的修正”，容易学到与目标无关的泛化 rerank 模式。

## 核心贡献（创新点）
- 形式化 Personal-Agent Mediated Recommendation 任务，明确定义平台提议为强默认、智能体选择性修正的设定。与已有平台侧智能体方法的本质区别在于以冻结的强平台提议为起点、仅允许在跨平台历史证据充分时干预。
- 提出 MediateRec 基准，包含 Amazon 类别代理的多平台环境与基于 OpenPlay 的真实跨平台外部测试，并在受控的平台–智能体信息边界下评测救援/有害覆盖/干预强度。与已有 agentic 推荐基准的区别在于直接对比平台原始排序并评估信息不对称下的选择性修正。
- 提出 PAMO，通过反事实遮蔽跨平台历史估计“个人中介支持度”，并在排名相关优势质量约束下做价值保持的优势重分配。与纯结果 RL 的本质区别在于奖励分配不仅看排名结果，还衡量响应对跨平台历史的依赖强度。
- 给出理论性质：PAMO 在每个活跃 cutoff 上保持正负优势质量，并在不降低平均平台相对价值的一阶重分配中是局部最优的。与一般策略梯度/优势重分配的差异在于显式保留 cutoff 层面的优势质量并约束平均价值不下滑。

## 方法详解
- 任务与评价：平台输出候选集与完整排序，取 top-K 作为默认提案 P；个人智能体接收平台提案 B、本平台历史 $H^{\mathrm{within}}$ 与跨平台历史 $H^{\mathrm{cross}}$，生成可解释 rationale R 与最终 top-K 列表 S。以 HR@K 和 NDCG@K 度量，定义中介价值 $\Delta(S;P,Y)=U(S,Y)-U(P,Y)$，强调 rescue 与 harmful override 的平衡。
- 个人中介支持度估计（反事实遮蔽）：对每次采样响应 $(R_g, S_g)$，在固定 rationale 前缀下比较全历史 $(B,H)$ 与仅含本平台历史 $(B,H^-)$ 的日志概率，得到每 token 的“个人中介支持”分数 $c_g=\frac{1}{|R_g|}\log\frac{\pi_{\bar\theta}(R_g|B,H)}{\pi_{\bar\theta}(R_g|B,H^-)}$，并用归一化长度、停止梯度后附于整条响应。
- 排名感知的平台相对优势分解：将 NDCG@K 按 cutoff $k$ 边际增益 $\alpha_k$ 精确分解为 $\sum_k \alpha_k z_{g,k}$，并采用 GRPO 式组内中心化的 $A_{g,k}^{\mathrm{grp}}=z_{g,k}-p_k$；总正/负优势质量 $m_k=Gp_k(1-p_k)$。仅对与平台不一致的 cutoff 集合 $\mathcal{D}_k$ 进行重分配。
- 价值保持的优势重分配：定义方向对齐 NDCG 量级 $v_{g,k}=s_k(r_g-r_P)\ge0$，并要求重分配后的平均 $v_{g,k}$ 不低于均匀分配的均值（value floor）。在每个 $\mathcal{D}_k$ 上求解凸优化：在保持总权重为 1 且满足价值地板的前提下，最大化 $\eta\sum w_{g,k}c_g-\mathrm{KL}(w_k\|u_k)$，得到指数形式权重 $w_{g,k}^\star\propto\exp(\eta c_g+\lambda_k v_{g,k})$，$\lambda_k$ 通过一维搜索使价值地板刚好激活或不激活。
- 训练目标：将截止层面优势 $A_{g,k}^{\mathrm{PAMO}}$ 按 $\alpha_k$ 合成响应级优势 $A_g^{\mathrm{PAMO}}$，并代入与基线一致的 clipped token-level GRPO 目标；仅增加一次 within-only 的 teacher-forced 评分，不增加 rollout，推理阶段无额外组件。
- 超参校准：支持强度 $\eta=\min\{\kappa/\max(\hat\sigma_c,\epsilon_c),\eta_{\max}\}$，$\kappa=0$ 退化为均匀分配，越大越依赖支持度；主要实验使用 $\kappa=0.5$。

## 实验与结果
- 数据集：MediateRec 含 Amazon Reviews 构建的四个代理跨平台数据集 Movie/Toy（训练+验证+测试）、Grocery/Beauty（仅测试，用于未见目标平台），以及基于真实跨服务链接的 OpenPlay 外部测试集（645 用户，不用于训练/选模）。
- 平台强基线：SASRec + Claude Sonnet 4.6 用本平台历史的 rerank，冻结后作为统一输入；四 Amazon 数据集平均 HR@10/NDCG@10 为 49.3%/0.329，OpenPlay 为 40.8%/0.241。
- 基线：Platform 冻结提案、SFT、GRPO（NDCG@10 奖励）、PAMO 及其无价值地板变体；另给出 Claude Haiku/Sonnet/Opus 4.6 的推理参考。
- 主要结果：PAMO 在所有五个数据集上均优于匹配 GRPO，提升集中在头部命中率与 NDCG@10，并显著改善 rescue-harm 平衡；例如在 Movie 上 HR@10 从 GRPO 的 61.2 提升到 62.2、Harm 从 3.9 降到 4.0 附近且 Intervention 稳定；在未见目标 Grocery/Beauty 和 OpenPlay 上同样持续提升。OpenPlay 上 PAMO 取得最高 HR@10=47.9、NDCG@10=0.284，并优于所有强商用模型。
- 消融：$\kappa$ 呈倒 U 形，$\kappa=0.5$ 最优；去掉 value floor 性能低于 $\kappa=0$，说明单纯按历史依赖重分配会鼓励高依赖但低价值的修正。
- 机制分析：PAMO 成功救援的响应中约 72% 以跨平台上下文为驱动证据，典型来源包括兴趣/粉丝、家庭/ household、生活方式/健康；弱证据时更倾向保留平台排序。

## 相关工作脉络
- Agentic recommender（SAGER/MemRec/MARS/RecMind 等）：多在平台侧强化推荐流程或工具使用；本文定位为在平台冻结提议之后由个人智能体做选择性中介。
- User-governed personalization / Muse / iAgent / PersonaAgent 等：聚焦个人侧记忆与交互，但缺少以强平台提议为起点的信息边界评测；本文构造平台–智能体信息非对称的 MediateRec。
- ClawRec / 跨域推荐：前者统一多源构造 slate，后者在集中式模型内融合多域；本文固定目标平台推荐器，仅在用户侧引入跨平台历史并做选择性覆盖。
- Rank-GRPO / 推荐 RL：侧重结果奖励或排名奖励；本文在同等结果奖励基础上增加对跨平台历史依赖的归因，避免“泛化 rerank 也能拿奖励”的问题。
- 现有 agentic 基准（AgentRecBench、τ-rec 等）：主要评估平台侧 agent 能力；本文直接对比平台原始排序并量化救援/有害覆盖。

## 局限性与未来方向
- Amazon 代理平台的边界是类目划分，并非真正独立的服务平台，跨平台信号仍属同生态；OpenPlay 规模有限，外部泛化仍需更多真实服务验证。
- 冻结强平台排序虽便于控制变量，但现实场景中平台也可能随个人智能体反馈更新，双向交互未纳入。
- 支持度 $c_g$ 仅度量历史依赖而非正确性，存在“高依赖但错误指导”的风险，当前由 value floor 缓解但未完全解决。
- 只用单次 top-10 输出与单一 NDCG/HR 目标，未显式建模多目标（多样性、覆盖率、长期留存）与更复杂的用户意图表达。
- 训练数据偏向 Movies/Toys，跨品类迁移与冷启动用户的行为学习仍有待系统验证。

## 研究启发与可借鉴点
- “平台提议 + 个人跨平台历史”的选择性中介设定，为平台开放化与用户治理提供了可度量、可训练的中间层方案，可迁移至搜索、广告投放、内容消费等场景。
- 反事实遮蔽 + 支持度估计的思想可用于任何需要区分“模型基于某类私有信息是否真正决策”的训练环节，作为归因式强化学习信号。
- 按 cutoff 边际增益分解排名指标并做组内再分配，既保留 GRPO 的结构又引入位置敏感度，适合需要精细排名控制的 LLM 推荐训练。
- Value floor 这一简单约束有效防止高依赖低价值样本主导梯度，可作为归因信号的必要安全阀，后续可与置信度/一致性检验结合。
- MediateRec 的受控信息边界设计（冻结平台、固定候选集、仅开放历史）值得作为通用中介/对齐评测模板复用。

## 关键术语表
- Personal-Agent Mediated Recommendation：平台给出去重排序提案后，由用户授权的个人智能体在保留强平台信号的前提下选择性修正的最终排序范式。
- Rescue / Harmful Override：救援指平台未命中而智能体找回目标；有害覆盖指平台命中而智能体错误移除，二者共同刻画中介风险收益。
- Personal Attribution Mediation Optimization（PAMO）：在 GRPO 基础上以反事实遮蔽估计跨平台依赖，并按价值地板约束重分配排名的优势质量。
- Counterfactual Personal Mediation Support：在同一 rationale 前缀下对比全历史与仅本平台历史生成的对数概率差，衡量该响应对跨平台历史的依赖强度。
- Value Floor：重分配后要求被修正响应的平均方向对齐 NDCG 量级不低于均匀分配，避免高依赖却低质量的修正主导更新。
- Active Cutoff：在该排名边界上平台上成功与失败均未达到边界（$0<p_k<1$）的位置，PAMO 仅对此类 cutoff 做优势重分配。
- Platform-Relative Advantage：相对于平台默认排序的排名收益变化，强调中介动作的增量价值而非绝对分数。

## 可复现要素
- 数据集：Amazon Reviews 2023 构建的 Movie/Toy/Grocery/Beauty 代理跨平台数据，以及 OpenPlay 公开数据的 645 用户外部测试集；论文声明公开划分、提示模板与解析代码，数据集链接见论文。
- 代码/权重：论文未提供明确开源仓库与模型权重链接；训练基于 Qwen3-4B-Instruct-2507 与 VeRL/vLLM 生态。
- 关键超参：SFT 三 epoch、batch 64、序列长 8192、lr $10^{-5}$；RL lr $10^{-6}$、每更新 64 提示 × 8 rollout、PPO clip 0.2、主实验 $\kappa\in\{0.25,0.5,0.75\}$、$\eta_{\max}=60$；评估 temperature 0.7、top-p 0.8、top-k 20。
