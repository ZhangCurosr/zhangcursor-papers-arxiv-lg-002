---
title: "SURVIVING-THE-ROUTER-OPTIMIZING-SKILL-INJEC-TIONS-FOR-RETRIE"
source: https://arxiv.org/pdf/2610.08098v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:23:06"
field: "LLM Agent 安全与鲁棒性"
keywords: ["skill injection", "agent security", "retrieval attack", "prompt injection", "skill router", "adversarial robustness"]
innovations: ["揭示技能路由器对注入攻击的过滤效应，纠正现有评估的乐观偏差", "提出 CORSA 两阶段簇级优化方法，同时优化检索排名与端到端执行成功率", "验证攻击跨模型、跨脚手架、跨路由器的迁移能力，揭示防御评估盲区"]
benchmarks: ["SkillRouter", "SKILLJECT", "Cisco Skill Scanner", "NVIDIA SkillSpector"]
---

# 论文速读：SURVIVING-THE-ROUTER-OPTIMIZING-SKILL-INJECTIONS-FOR-RETRIEVAL-AND-EXECUTION

## 一句话总结
本文揭示了现有技能注入攻击（Skill Injection）因忽略技能路由器（Skill Router）的竞争机制而严重高估实际成功率的致命缺陷，并提出 CORSA（Cluster Optimization for Router-Aware Skill Attacks），通过两阶段演化搜索同时优化恶意技能在检索排名与端到端执行上的表现，在真实多技能环境下将攻击成功率提升约 2 倍（ASR 从 10% 提升至 22%），且攻击具备跨模型、跨路由器、跨 Agent 框架的迁移能力。

## 研究问题与动机
- **现有攻击评估假设不切实际**：已有工作（如 SKILLJECT）假设中毒技能已被代理选择并执行，忽略了技能商店中大量良性技能的存在，导致对攻击成功率的评估过于乐观。
- **技能路由器的过滤效应被低估**：在真实多技能环境中，注入技能需先通过路由器检索竞争才能被执行；实验表明常规注入使 Hit@1 从 100% 暴跌至约 11%，有效 ASR 下降 87–97%。
- **单任务优化泛化性差**：针对单一任务优化的注入难以泛化到同聚类内的其他任务或新查询，而覆盖所有任务又会导致技能与具体领域脱钩。
- **防御评估缺乏端到端视角**：现有防御多关注安装前静态扫描或运行时权限限制，未考虑路由阶段对注入技能的筛选作用，导致防御评估与实际威胁场景脱节。

## 核心贡献（创新点）
1. **揭示路由器对注入攻击的抑制效应**：首次系统评估技能路由对注入成功的过滤作用，证明现有攻击因忽略检索阶段而被严重高估。
2. **提出 CORSA 两阶段簇级优化框架**：设计 Stage A（优化检索排名）和 Stage B（优化端到端攻击成功率）的递进式优化流程，使注入技能在竞争中存活并成功执行。
3. **扩展 SkillRouter 基准至安全场景**：将原有 75 个任务划分为 8 个语义聚类，集成 8 类恶意 Payload（数据泄露、破坏、DoS 等），构建更贴近现实的评估体系。
4. **验证攻击的跨场景迁移性**：证明 CORSA 生成的注入可迁移至不同 LLM 后门（GPT-5.4、GLM-5.3、Qwen3.8、DeepSeek-V4-Pro）、不同 Agent 脚手架（Codex→OpenHands）及不同路由器架构。
5. **防御评估揭示现有扫描器不足**：评估 Cisco Skill Scanner 和 NVIDIA SkillSpector 发现，即使最强配置（LLM 增强）仍遗漏约 20% 的 CORSA 攻击包，存在显著检测盲区。

## 方法详解
- **威胁模型**：攻击者可向技能市场发布技能，拥有路由器评分函数的查询访问权限及受害者 LLM 的离线评估能力，但无法获取模型权重或修改其他技能。攻击目标是在保持技能良性功能的前提下，使注入技能被检索为 Top-1 且其 Payload 被执行。
- **任务聚类**：将 75 个基准任务划分为 8 个语义聚类（如 coding-and-devops、data-science 等），每个聚类包含 3–13 个任务，共享领域词汇，使单个注入技能可覆盖同簇内多个任务。
- **两阶段优化目标**：
  - **Stage A（检索优化）**：最大化 $\sum_{t \in \mathcal{C}} \text{Hit}(t, S')$，即让注入技能在聚类内尽可能多任务上获得检索排名第一。
  - **Stage B（端到端优化）**：最大化 $\sum_{t \in \mathcal{C}} \text{Hit}(t, S') \cdot \text{Exec}(t, S')$，在保持检索排名的同时确保 Payload 被执行。
- **优化器 GEPA**：采用反射式演化搜索（Genetic-Pareto），维护候选技能池，利用任务级反馈（检索排名、执行结果）指导 LLM 对技能名称 $n'$、描述 $d'$ 和指令体 $s'$ 进行迭代修改，Payload $p$ 固定不变。
- **评估指标**：Hit@1（检索成功率）、ASR（端到端攻击成功率）、User Utility（良性任务完成度）、Stealthiness（GPT-5.4 评判的自然融合度）。

## 实验与结果
- **数据集与基线**：基于 SkillRouter 基准（75 个任务、约 8 万技能池、2000 个良性技能作为竞争者），对比基线 SKILLJECT。
- **主结果（GPT-5.4 攻击者）**：
  - CORSA 的 Hit@1 达 32.3%，较 SKILLJECT 的 11.0% 提升近 3 倍；ASR 达 22.0%，较 SKILLJECT 的 10.0% 提升约 2 倍；用户效用保持 38.8%，与基线 38.3% 相当。
  -  Across 8 种 Payload，CORSA 在 engineering 和 scientific 聚类上表现最强（ASR 达 62.5% 和 31.9%），在 coding 聚类上仍达 12.1%。
- **泛化性测试**：在 Paraphrase 和 Synthetic 新任务上，CORSA 的 ASR 分别降至 11.5% 和 5.4%，仍显著高于 SKILLJECT 的 1.0% 和 0.2%。
- **迁移性**：
  - **跨模型**：CORSA 攻击在 Qwen3.8-27B 以外均保持 ASR > 10%。
  - **跨脚手架**：从 Codex 迁移到 OpenHands，ASR 从 22.0% 降至 16.2%，仍优于 SKILLJECT 的 2.5%。
  - **跨路由器**：在不同路由器间迁移呈现不对称性，Qwen3-8B 路由器对攻击最具抵抗力（最低 Hit@1 和 ASR）。
- **防御评估**：Cisco Skill Scanner 和 NVIDIA SkillSpector 在静态模式下召回率为 0；LLM 增强后最高召回率约 8/24（SkillSpector + Qwen3.8-27B），仍存在检测盲区。

## 相关工作脉络
- **SKILLJECT（Jia et al., 2026）**：自动化技能注入攻击，但假设中毒技能已被执行，忽略路由器竞争；CORSA 在相同基准下证明其有效 ASR 仅 10%。
- **SKILL-INJECT（Schmotz et al., 2026）**：首个系统评估技能注入的基准，但同样未考虑路由阶段；CORSA 扩展其攻击分类至 8 类 Payload。
- **SkillRouter（Zheng et al., 2026）**：提出大规模技能路由框架，显示全技能内容对检索重要；CORSA 利用此特性，通过优化技能内容提升检索排名。
- **TOOLHIJACKER（Shi et al., 2025）**：攻击工具选择环节，关注检索阶段；CORSA 将检索与执行统一建模，覆盖更完整的攻击链路。
- **SkillGuard（Pan et al., 2026）**：运行时权限限制防御；CORSA 的端到端评估可互补验证此类防御的实际有效性。

## 局限性与未来方向
- **优化成本较高**：两阶段 GEPA 演化搜索需要多次查询路由器和受害者 LLM，计算开销较大，尚未探索轻量级优化策略。
- **仅针对间接注入**：当前评估集中于通过 helper script 的间接攻击，直接文本注入的检索竞争机制可能不同。
- **防御评估局限性**：仅测试两种商业扫描器，未评估 self-defense 或对抗训练等内生防御手段。
- **聚类划分依赖人工标注**：任务聚类由 GPT-5.4 生成，可能存在语义边界模糊问题，未研究自动聚类方法。
- **未考虑动态技能市场**：现实场景中技能会持续更新，攻击者可能面临版本竞争和环境变化。

## 研究启发与可借鉴点
- **两阶段优化范式可迁移**：将"检索竞争"与"目标执行"分阶段优化的思路可应用于其他 Agent 组件（如工具选择、记忆检索）的安全评估。
- **簇级优化提升泛化性**：对比任务级优化的消融实验表明，簇级优化可显著提升跨任务泛化，该方法论可用于提升黑盒攻击的效率。
- **防御评估需端到端视角**：现有防御多聚焦单阶段检测，CORSA 证明攻击可绕过预检索扫描，提示防御设计需覆盖完整攻击链路。
- **路由器多样性可作为防御杠杆**：实验显示 Qwen3-8B 路由器对攻击更具抵抗力，提示 router architecture 选择本身可作为一种缓解策略。
- **可复用的实验基准**：扩展的 8 聚类 × 8 Payload 基准可直接用于后续技能注入防御方法的对比评估。

## 关键术语表
- **Skill Injection（技能注入）**：在第三方技能的 SKILL.md 或辅助脚本中嵌入恶意指令，诱导 Agent 执行非预期操作。
- **Skill Router（技能路由器）**：根据用户任务从技能库中检索并排序相关技能的组件，通常基于语义相似度或混合检索。
- **ASR（Attack Success Rate，攻击成功率）**：注入技能被检索为 Top-1 且其 Payload 被执行的任务比例，衡量端到端攻击有效性。
- **GEPA（Genetic-Pareto Evolutionary Algorithm）**：反射式演化搜索优化器，通过任务级反馈迭代改进文本组件，无需梯度信息。
- **Task Cluster（任务聚类）**：共享领域词汇和语义的多个相关任务集合，用于平衡攻击针对性与泛化性。
- **Indirect Attack（间接攻击）**：攻击者不直接修改用户输入，而是通过技能文件诱导 Agent 执行外部脚本或访问敏感资源。
- **Hit@1（Top-1 Retrieval Accuracy）**：注入技能在竞争环境中被路由器排名为第一的比例。
- **Utility Preservation（效用保持）**：注入技能在触发 Payload 的同时仍能完成原始良性任务的能力。

## 可复现要素
- **数据集**：SkillRouter 基准（75 个任务），已扩展至 8 个语义聚类和 8 类 Payload；原始基准公开，扩展部分论文未明确说明是否开源。
- **代码**：已开源，GitHub 链接：https://github.com/compass-group-tue/CORSA-Cluster-Optimization-for-Router-Aware-Skill-Attacks
- **权重**：使用 GPT-5.4 和 DeepSeek-V4-Pro 作为攻击者/受害者模型，论文未提及本地模型权重，使用 API 调用。
- **关键超参**：优化阶段数（2 阶段）、聚类大小（3–13 任务）、良性技能池大小（2000）、评估任务数（75），具体数值论文未全部列出，需在代码仓库中查阅。
- **环境配置**：CODEx 和 OpenHands Agent 脚手架，BM25、OAI-Emb-3L、SR-0.6B、R3-0.6B、Qwen3-8B 等五类路由器，详见论文 Section 4.4 及附录。
