---
title: "Personal-Agent-Mediated-Recommendation-with-Cross-Platform-U"
source: https://arxiv.org/pdf/2610.07588v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:13:10"
field: "个性化推荐系统与多平台用户建模"
keywords: ["Personal-Agent Mediated Recommendation", "Cross-Platform History", "MediateRec", "PAMO", "Reinforcement Learning for LLMs", "Rescue-Harm Trade-off", "User-Governed Personalization"]
innovations: ["形式化个人Agent中介推荐范式并刻画拯救-有害覆盖权衡", "提出MediateRec基准含真实跨平台OpenPlay外部测试", "设计PAMO通过反事实mask估计个人中介支持度并价值保留优势重分配"]
benchmarks: ["MediateRec-Movie", "MediateRec-Toy", "MediateRec-Grocery", "MediateRec-Beauty", "MediateRec-OpenPlay"]
---

# 论文速读：Personal-Agent-Mediated-Recommendation-with-Cross-Platform-U

## 一句话总结
本文提出**Personal-Agent Mediated Recommendation**新范式：平台推荐器先基于平台内数据排名候选集，再由用户授权的个人 LLM Agent 利用跨平台历史对平台推荐结果进行选择性修正，最终输出 top-K 推荐清单。核心创新是设计了对照基准 **MediateRec** 与训练算法 **PAMO**，实现了对强平台推荐的"拯救"（rescue）而不引入"有害覆盖"（harmful override）。

---

## 研究问题与动机

- **用户治理 vs 平台中心**：推荐系统正从平台中心化个性化转向用户治理（user-governed），用户授权的个人 Agent 可跨服务维护历史并代表用户决策（如 Meta 的 Muse, 2026）。
- **平台排名蕴含群体证据**：平台推荐编码了人群级别的协同过滤模式（SASRec + Claude Sonnet 4.6 重排），个人 Agent 无法直接观测；但 Agent 掌握跨平台历史记录，可修正平台遗漏的个性化目标。
- **中介是非平凡的权衡问题**：有效中介必须平衡有益拯救与有害覆盖——Agent 可能修正平台遗漏，也可能错误移除平台正确推荐的候选。这一 trade-off 使问题区别于简单重排序。
- **缺少可评估的基准**：现有 agentic recommender 基准主要评估平台侧 Agent 如何利用平台内信息，缺乏对用户侧个人 Agent 在平台 proposal 上做选择性干预的评测设置。

---

## 核心贡献（创新点）

1. **形式化 Personal-Agent Mediated Recommendation 任务**：明确定义了平台 proposal + 跨平台历史的输入条件，刻画了四类中介结果（preserved hit / rescue / harmful override / mutual miss）及 $\Delta\mathrm{HR@K}$ 度量，与现有平台侧 Agent 工作形成本质区分。
2. **引入 MediateRec 基准**：基于 Amazon Reviews 构建四个代理跨平台环境（Movie/Toy/Grocery/Beauty），并使用 OpenPlay 真实跨平台数据集（Steam/Nintendo/Xbox）作为外部测试，建立受控的平台–Agent 信息边界。
3. **提出 PAMO（Personal Attribution Mediation Optimization）**：通过反事实 mask 跨平台历史估计"个人中介支持度"，在平台相对值底线约束下重新分配 rank-aware 优势质量，理论证明其保持截断级优势质量且局部最优。
4. **理论性质**：证明 PAMO 保持正负优势质量的总和不变（Proposition 1），且其一阶实方向量是约束锥上的 Euclidean 投影（Theorem 1）。

---

## 方法详解

### 任务形式化（Section 3）

- 平台 proposal：$B = (\mathcal{C}, \rho_P, M)$，其中 $\rho_P$ 为完整平台排名，$P = \mathrm{Top}_K(\rho_P)$ 为默认 top-K 清单。
- 用户历史：$H = (H^{\mathrm{within}}, H^{\mathrm{cross}})$，跨平台历史 $H^{\mathrm{cross}}$ 为平台不可见。
- Agent 策略：$(R, S) \sim \pi_\theta(\cdot \mid B, H)$，输出可见 rationale $R$ 与最终清单 $S$。
- 中介价值：$\Delta(S; P, Y) = U(S,Y) - U(P,Y)$，HR@K 下的改善为 $\Delta\mathrm{HR@K} = \Pr(\text{rescue}) - \Pr(\text{harmful override})$。

### 基准设计（Section 4）

- **Amazon Proxy Platforms**：Movies & TV、Toys & Games、Grocery & Gourmet Food、Beauty & Personal Care 作为四个目标平台；以 0-core 的 Amazon Reviews 2023 构建，Movie/Toy 各 5k/1k/2k，Grocery/Beauty 各 2k 测试，用于 seen/unseen target 评估。
- **OpenPlay 真实跨平台测试**：通过 pseudonymized person ID 链接 Steam/Nintendo/Xbox 活动，Steam 为目标平台，645 用户队列仅用于外部测试，排除训练/调参。
- **候选集构造**：每个 episode 固定 50 候选（目标 + 49 个按流行度采样的负样本），平台端由 SASRec 排初序后接 Claude Sonnet 4.6 temperature=0 重排（只用 within 历史），最终排名冻结。

### PAMO 算法（Section 5）

**第一步：反事实个人中介支持度估计**（Eq. 8）
$$
c_g = \frac{1}{|R_g|} \log \frac{\pi_{\bar\theta}(R_g \mid B, H)}{\pi_{\bar\theta}(R_g \mid B, H^-)}
$$
其中 $H^- = (H^{\mathrm{within}}, \emptyset)$ mask 掉跨平台历史；$c_g$ 越大表示该 rationale 越依赖跨平台历史。

**第二步：NDCG 的 rank-aware 分解**（Eq. 10-11）
将 NDCG@K 分解为嵌套 top-k 边界上的边际增益之和：$r_g = \sum_{k=1}^K \alpha_k z_{g,k}$，其中 $\alpha_k = \ell_k - \ell_{k+1}$ 为截断边际效用。

**第三步：价值保留的优势重分配**（Eq. 17-18）
在每个与平台分歧的集合 $\mathcal{D}_k$ 上，求解：
$$
w_k^\star = \arg\max_{w} \eta \sum w_{g,k} c_g - \mathrm{KL}(w_k \| u_k) \quad \text{s.t.} \quad \sum w_{g,k} v_{g,k} \geq \bar{v}_k
$$
得到指数形式权重 $w_{g,k}^\star = \exp(\eta c_g + \lambda_k v_{g,k}) / Z$，其中 $\lambda_k$ 为值底线的对偶系数。

**最终优势**：$A_{g,k}^{\mathrm{PAMO}} = s_k m_k w_{g,k}^\star$（分歧集）或保留 GRPO 组相对优势（一致集），再组合为响应级优势 $A_g^{\mathrm{PAMO}} = \sum_k \alpha_k A_{g,k}^{\mathrm{PAMO}}$。

**超参数**：$\kappa$ 控制支持集中度（$\eta = \min\{\kappa/\max(\hat\sigma_c, \epsilon_c), \eta_{\max}\}$，主实验 $\kappa=0.5$）；训练使用 Qwen3-4B-Instruct  backbone，64 prompts × 8 rollouts，400 次更新。

---

## 实验与结果

### 数据集与指标
- **MediateRec-Movie/Toy**（seen target）、**Grocery/Beauty**（unseen target）、**OpenPlay**（真实跨平台外部测试）
- 指标：HR@{3,5,10}、NDCG@{3,5,10}、Rescue率、Harmful Override率、Intervention数（被替换的平台 top-10 项数）

### 主要结果（Table 2-3）

| 方法 | Movie H@10 | Toy H@10 | Grocery H@10 | Beauty H@10 | OpenPlay H@10 |
|---|---|---|---|---|---|
| Platform | 56.0 | 47.7 | 45.0 | 48.6 | 40.8 |
| GRPO | 61.2 | 53.4 | 48.8 | 51.2 | 47.4 |
| **PAMO** | **62.2** | **54.8** | **50.6** | **51.7** | **47.9** |
| Claude Sonnet 4.6 | 63.1 | 52.3 | 49.1 | 50.6 | 44.0 |
| Claude Opus 4.6 | 63.2 | 52.4 | 48.5 | 51.9 | 45.4 |

- PAMO 在所有五个数据集上均优于匹配 GRPO 基线，**OpenPlay 真实跨平台测试达到最高 H@10=47.9 / N@10=0.284**，超越所有 proprietary LLM 的 zero-shot 表现。
- PAMO 降低了 harmful override 率（Movie: 3.9→4.0, Toy: 5.6→4.8, OpenPlay: 7.1→6.0），同时提升 rescue 率，实现更好的拯救–有害平衡。
- 4B 开源模型经 PAMO 后在 unseen target 和 OpenPlay 上与 Claude Sonnet/Opus 竞争甚至超越。

### 消融与敏感性（Section 6.4）
- **$\kappa$ 效应呈倒 U 型**：$\kappa=0.5$ 最优，过大（0.75）导致支持差异主导分配、性能下降。
- **值底线必要性**：去掉值底线的 PAMO w/o Value Floor 表现低于 $\kappa=0$ 基线，证明单纯依赖跨平台历史依赖性不足以保证推荐质量。

### 中介行为分析（Section 6.5）
- **72% 的拯救由跨平台用户上下文驱动**：兴趣/fandom（28%）、家庭/ household（19%）、生活方式/健康（14%）。
- **选择性干预 vs 服从**：强跨平台证据（如 Led Zeppelin 专辑 → Robert Plant 演唱会）触发干预；弱/无关证据时保持平台第一候选。

---

## 相关工作脉络

1. **Agentic Recommender Systems**（Huang et al., 2025a; Lin et al., 2026b; Shang et al., 2026）：平台侧 Agent 增强推荐，与本文用户侧个人 Agent 中介定位不同。
2. **iAgent**（Xu et al., 2025）：同为用户–Agent–平台架构，但随机采样 1 正 9 负构造候选列表，非基于强平台 proposal 的中介。
3. **ClawRec**（Wu et al., 2026, concurrent）：直接从多源 context 构建统一跨源推荐清单；本文保持目标平台推荐器冻结，仅在用户侧引入 Agent 中介。
4. **Cross-domain Recommendation**（Li et al., 2022; Hou et al., 2022; Ju et al., 2025）：集中式推荐器融合多源行为；本文目标平台推荐器不变，跨平台历史仅在个人 Agent 侧使用。
5. **Post-training RL for LLM recommendations**（Lin et al., 2025; Huang et al., 2026; Zhu et al., 2026）：训练推荐 Agent 的 ranking/task 奖励方法，未处理跨平台历史归因与平台相对价值保留问题。
6. **Rank-GRPO**（Zhu et al., 2026）：LLM conversational recommender 的 RL 方法；本文在此基础上引入 NDCG 分解与平台相对值约束。

---

## 局限性与未来方向

- **候选集固定**：评估的是 candidate-conditioned ranking/mediation，非端到端检索 + 排序，Agent 不能引入候选集外的新 item。
- **历史样本有限**：Amazon 代理平台上 within 历史平均仅 4.9/2.8 条，跨平台平均 26.7/24.0 条，可能低估长程跨平台模式的影响。
- **价值函数选择**：使用 NDCG@K 作为优化目标，但 NDCG 对 rank 位置敏感，是否应引入其他用户效用度量（如 dwell time）待研究。
- **仅评估 top-10**：实际场景中 K 可能更大，截断效应需进一步分析。
- **单向信息流假设**：平台 proposal 冻结、Agent 不接收平台内部分数/模型状态，现实系统中可能存在双向反馈。
- **未见目标平台的泛化**：Grocery/Beauty 测试显示一定 transfer，但未见平台的历史分布差异可能限制泛化上限。

---

## 研究启发与可借鉴点

1. **Counterfactual masking 归因范式**：通过 mask 特定信息源（跨平台历史）并比较 log-probability 差异来估计"依赖度"，可迁移至其他需要解释模型决策来源的场景（如多模态 rag、工具调用归因）。
2. **价值保留约束的优势重分配**：PAMO 的值底线设计（保证重分配不降低平均平台相对价值）为 RL 训练中的 constraint optimization 提供了可复用的正则化思路。
3. **平台–Agent 信息边界基准构造**：冻结强平台 proposal + 控制历史暴露的 benchmark 设计，为评估"用户侧 Agent vs 平台侧推荐器"协作提供了可推广模板。
4. **Rescue–Harm trade-off 度量**：引入 $\Delta\mathrm{HR@K} = \Pr(\text{rescue}) - \Pr(\text{harmful override})$ 作为中介质量指标，可推广至任何需要修正现有排序的输出任务。
5. **与团队方向结合机会**：若团队关注**多 Agent 协作推荐**或**用户主权个性化**，可将 PAMO 的归因思想扩展至跨 Agent 信用分配；若关注**冷启动/长尾**，可探索跨平台历史在稀疏 within 场景下的补偿机制。

---

## 关键术语表

- **Personal-Agent Mediated Recommendation**：用户授权的个人 LLM Agent 利用跨平台历史对平台推荐 proposal 进行选择性修正的新范式。
- **MediateRec**：本文提出的基准测试套件，包含 Amazon 代理跨平台环境与 OpenPlay 真实跨平台外部测试。
- **PAMO（Personal Attribution Mediation Optimization）**：通过反事实 mask 估计个人中介支持度、并在平台相对值底线约束下重分配 NDCG 优势的训练算法。
- **Personal Mediation Support（$c_g$）**：衡量 sampled rationale 对跨平台历史依赖程度的对数概率比分数。
- **Rescue / Harmful Override**：Rescue 指 Agent 成功找回平台遗漏的目标 item；Harmful Override 指 Agent 错误移除平台正确推荐的目标 item。
- **Value Floor**：PAMO 重分配中要求平均方向对齐 NDCG magnitude 不低于均匀分配的约束条件。
- **Support Concentration（$\kappa$）**：控制 PAMO 中个人中介支持度对优势分配影响强度的超参数。
- **Platform–Agent Information Boundary**：平台仅使用平台内历史，Agent 额外获得跨平台历史但不可见平台分数/内部状态的信息不对称设置。

---

## 可复现要素

- **数据集**：Amazon Reviews 2023（0-core，公开）、OpenPlay（Ballou et al., 2025，公开）；论文 Appendix A 提供 split assignment、固定候选集、冻结平台排名、标题映射、prompt 模板与解析代码。
- **代码/权重**：论文未声明开源仓库；基于 Qwen3-4B-Instruct-2507、Claude Sonnet 4.6；使用 VeRL 训练、vLLM  rollout。
- **关键超参**：SFT 三 epoch、batch 64、lr $10^{-5}$、max seq 8192；RL lr $10^{-6}$、PPO clip 0.2、400 updates、64 prompts × 8 rollouts、temperature 1.0、top-p 1.0；PAMO $\kappa \in \{0.25, 0.5, 0.75\}$ 验证集选取（最优 0.5）、$\eta_{\max}=60$。
- **评估**：test-time temperature 0.7、top-p 0.8、top-k 20，FINAL: marker 后提取候选 ID，去重后按首次出现顺序排序。

---
