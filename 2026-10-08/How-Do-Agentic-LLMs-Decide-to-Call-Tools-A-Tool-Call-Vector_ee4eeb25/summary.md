---
title: "How-Do-Agentic-LLMs-Decide-to-Call-Tools-A-Tool-Call-Vector"
source: https://arxiv.org/pdf/2610.09624v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 02:55:24"
field: "LLM 机制可解释性与 agent 行为因果表征"
keywords: ["mechanistic interpretability", "tool calling", "activation patching", "representation engineering", "agent LLM", "refusal direction analogy"]
innovations: ["用单动词替换把数百 token agent 提示降维为可控对比接口", "定位到 L24 预测位置的因果充分必要工具调用向量 μΔ 及其跨域/多轮/无动词零重估迁移", "揭示格式模板安装调用先验、分析动词激活抑制特征覆盖先验的'先验+抑制'统一机制"]
benchmarks: ["MBPP", "APPS", "HumanEval", "CodeContests", "τ²-Bench Telecom", "FEVER", "Spider"]
---

# 论文速读：How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression

## 一句话总结
本文通过词性替换（执行动词→分析动词）将长而复杂的 agent 提示转化为可控对比对，定位到 Qwen3-8B 残差流中层 24 预测位置的工具调用向量 $\mu_\Delta$，证明其对"是否调用工具"的二元决策既是因果必要又是充分的；该向量无需重新拟合即可跨领域、跨多轮轨迹、跨无显式动词请求迁移。

## 研究问题与动机
- **核心问题**：Agentic LLM 在任务型 agent 提示中如何决定"首次生成 `<tool_call>` 还是直接文本回复"？现有工作只关注选对哪个工具，从未从机制角度解释"调不调"这个二元决策的内部表征。
- **长提示使定位失控**：传统可解释性研究多在 10–30 词的短提示上逐变量做消融；agent 提示由角色指令、工具 schema、格式模板、用户请求交织构成，数百 token 互相缠绕，找不到一个单一可控变量来构造可靠对比对。
- **动词作为可控断点**：作者在代码补全任务中发现，仅把执行动词（如 `write`）换成分析动词（如 `discuss`）就能稳定翻转首 token 决策，从而构造了保持一切其余不变、仅动词变化的 500 对配对提示。
- **缺乏跨架构证据**：即使在同一模型上发现向量，也不清楚它是否属于 LLM 通用的 agent 行为机制，还是在某个训练数据或 prompt 模板下过拟合的特例。

## 核心贡献（创新点）
1. **首个受控的"单词因果接口"**：用执行/分析动词替换把数百 token 的 agent 提示压缩成一个一维行为对照，首次让"调不调工具"成为 mechanistic tools（activation patching、direct logit attribution）可直接瞄准的单 token 预测目标。与已有工具调用评测工作的本质区别：不再把调用决策当黑盒行为输出，而是打开决策形成通道。
2. **定位到层 24 预测位置的因果充分必要向量 $\mu_\Delta$**：$\mu_\Delta$ 是执行减去分析的残差均值，在 L24 预测位置加/减它能把分析侧恢复 100% 工具调用、把执行侧抑制到 0%，Suff. = 1.03、Necc. = 1.04。与 function vector / refusal direction 的本质区别：它被精确定位到单一层的单一位置，而非层内所有 token 的全局方向。
3. **跨域、跨多轮、跨无动词请求的零重估迁移**：保持 L24 方向固定、仅按目标域校准范数，$\mu_\Delta$ 在 web 检索/SQL 执行/邮件发送中同样 100% 诱导或抑制；直接在 $\tau^2$-Bench 多轮轨迹上以 1× gain 不加重新拟合即能抑制 36.7% 基线调用并诱导 70.0% 的文本回复；无动词请求上 1× 抑制 93.1%、1.5× 抑制 100%。与已有工作仅评测同一任务下微调的区别：证明了决策通道的可移植性。
4. **揭示"先验 + 抑制"机制，类比拒绝方向**：格式模板安装了一个近天花板工具调用先验（neutral 概率 0.8486），分析动词激活一组"工具不必要"特征的写方向与 $\hat{\mu}_\Delta$ 反向；执行动词基本不激活这些特征，保留先验。与 refusal direction 的结构性同构：强先验 → 小型抑制器覆盖，但前者是 action-initiation、后者是 safety-alignment。
5. **跨七模型的机制复现**：Qwen3-4B/8B/14B、Qwen3.5-4B/9B、Mistral-Small-3.2-24B、Granite-3.3-8B 全部满足：定位精准、全态 patching 93.5–100%、Suff/Necc 接近 1、$K_{corrupt} > K_{clean}$（抑制占优）、readout 与先验机制稳定；说明这是模型家族的共享架构而非个别模型过拟合。

## 方法详解
### 2. 构造一词断点
- **提示结构**：$x = [\text{prompt scaffold}][v \parallel \text{task body}] \rightarrow y$，scaffold 含 role instructions + tool schemas + format templates；$v$ 为请求动词，$y$ 为首 token，$y_{call} = \texttt{<tool\_call>}$。
- **动词池**：5 个执行动词（add, build, complete, save, write）+ 5 个分析动词（discuss, explore, inspect, review, study）；任务来自 MBPP / APPS / HumanEval / CodeContests，覆盖 Python、Java、C++。
- **数据集**：Qwen3-8B 下 500 对，300 训练（用于估计 $\mu_\Delta$）、200 保留（用于评估）。

### 3. 定位并估计工具调用向量
- **Activation patching**：对每对 $(x_i^c, x_i^*)$，把分析激活在层 $l$、位置 $q$ 替换为执行侧对应值，其他位置不变。恢复率 $r(l,q)$ = 首 token top-1 为 $y_{call}$ 的比例。
- **定位结果**：L24 预测位置 $p$ 处全态 patching 使分析侧恢复率达 100%；L23–L25 窗口整体强，L24 最优。
- **向量估计**：$\Delta_i = h_p^{(24)}(x_i^c) - h_p^{(24)}(x_i^*)$，$\mu_\Delta = \text{mean}(\Delta_i)$。
- **干预**：$\tilde{h}_p^{(24)}(x_i^*) = h_p^{(24)}(x_i^*) + \mu_\Delta$，$\tilde{h}_p^{(24)}(x_i^c) = h_p^{(24)}(x_i^c) - \mu_\Delta$。
- **因果指标**：
  $$\text{Suff}(\mu_\Delta) = \frac{\bar{z}_{call}^{*+\mu} - \bar{z}_{call}^*}{\bar{z}_{call}^c - \bar{z}_{call}^*}, \quad \text{Necc}(\mu_\Delta) = \frac{\bar{z}_{call}^c - \bar{z}_{call}^{c-\mu}}{\bar{z}_{call}^c - \bar{z}_{call}^*}$$
  实测 Suff ≈ 1.03、Necc ≈ 1.04，几乎完全覆盖原始 logit gap。

### 4. 迁移实验
- **跨域**：web 检索 (`web_search`)、只读 SQL 执行 (`run_sq1`)、邮件发送 (`send_email`)，各 100 对。方向固定、范数以目标域训练对重新校准。
- **多轮轨迹**：$\tau^2$-Bench Telecom，5,385–15,690 context tokens、16–43 工具、含历史对话与调用记录，200 个决策点，不加重新拟合直接施加。
- **无动词请求**：600 条非祈使句（"Is it true that…?" / "I haven't gotten to…yet"），覆盖 code/search/database/API；从各 domain-pattern 中选 up-to-10 条基线调用的子集评估。

### 5. 形成机制
- **先验来源消融**：去掉 format template（F）几乎消除调用（中性仅 $6.32\times10^{-7}$）；只留 F 时两类请求都接近 1.0；tool schema（T）决定"对请求词语敏感"的分离度。
- **MLP 主导形成**：沿 $\hat{\mu}_\Delta$ 投影的 gap $g_l$ 在前 15 层接近 0，L20–L23 形成窗口快速增长；其中 MLP 贡献 77.7%、attention heads 22.3%，L23 MLP23 单点最大。
- **Transcoder 分解**：$\text{MLP}^{(\ell)}(x) \approx b_{dec}^{(\ell)} + \sum_f a_{\ell f}(x) w_f^{(\ell)}$；特征得分 $\kappa_{\ell f} = (\bar{a}_{\ell f}^c - \bar{a}_{\ell f}^*)\langle w_f^{(\ell)}, \hat{\mu}_\Delta\rangle$。L21–L23 主要由 corrupt-higher 的"analysis-non-execution"家族主导（115 个特征、mean $|\kappa| = 0.0633$），占 L20–L23 总净写 7.28/14.34；执行侧家族 292 特征但 mean $|\kappa| = 0.0156$。
- **因果阻断测试**：归零 5 个 highest-$|\kappa|$ 抑制特征后 L24 输入沿 $\hat{\mu}_\Delta$ 偏移 +5.30，关合 8.82% gap，调用率由 3.0% 升至 25.5%；层匹配随机抑制器控制仅 +0.14、0.23%。
- **Mediation**：整段 L20–L23 MLP 输出对调后仅恢复沿 $\hat{\mu}_\Delta$ 的投影分量即可 100% 恢复执行侧调用。

### 6. 下游读出
- **Scaffold-reading heads**：L29H9、L33H11、L33H29 在执行侧强转向 format template 区域；$\mu_\Delta$ 加回后 L33H29 DLA 由 11.13 → 24.43（接近执行侧 24.58）。
- **Late structural feature**：L34 F109925 响应 tool-schema 边界、decoder 正向投影到 `<tool_call>`；加 $\mu_\Delta$ 后激活由 67.6 → 126.0，projected write 由 3.35 → 6.25。替换其 activation 严格恢复 37.0% vs 控制 6.5%。
- **分布式读出**：组消融显示单头/单 MLP 移除影响有限，合并早期+晚期 heads+MLPs 产生 5.02 logit 下降，top-1 仅掉 0.99%，证明读出是分布式的。

### 7. 跨模型
- 7 模型均在各自 $L^*$（4B:26、8B:24、14B:34、3.5-4B:31、3.5-9B:31、Mistral:25、Granite:35）全态 patching 达 93.5–100%；Suff/Necc 接近 1；$K_{corrupt}/K_{clean} > 1$ 全部成立。

## 实验与结果
### 数据集与配置
- 任务源：MBPP、APPS、HumanEval、CodeContests（代码）；$\tau^2$-Bench Telecom（多轮）；自构造无动词请求 600 条；跨域 100 对/域 × 3。
- 模型：Qwen3-4B/8B/14B、Qwen3.5-4B/9B、Mistral-Small-3.2-24B-Instruct、Granite-3.3-8B-Instruct；非 thinking 模式；$\texttt{<tool\_call>}$ 类标记作为决策信号。
- 划分：每模型 300 训练对 / 200 保留对；跨域和 $\tau^2$-Bench 使用额外独立对。

### Qwen3-8B 主结果
| 干预 | Top-1 前 | Top-1 后 | $\Delta$ | Logit 前 | Logit 后 | $\Delta$ logit | Suff. | Necc. |
|---|---|---|---|---|---|---|---|---|
| Add $\mu_\Delta$（分析侧） | 0.00% | 100.00% | ↑100.00 | 25.11 | 32.50 | ↑7.39 | 1.03 | — |
| Remove $\mu_\Delta$（执行侧） | 100.00% | 0.00% | ↓100.00 | 32.28 | 24.80 | ↓7.48 | — | 1.04 |

### 跨域迁移（范数校准后）
| 目标 | 工具 | N | Add 诱导 | Norm. shift | Remove 抑制 | Norm. shift |
|---|---|---|---|---|---|---|
| Web retrieval | web_search | 100 | 100.0% | 0.93 | 100.0% | 0.75 |
| SQL execution | run_sq1 | 100 | 100.0% | 0.79 | 100.0% | 0.77 |
| Email dispatch | send_email | 100 | 100.0% | 0.91 | 100.0% | 1.30 |
| **Mean** | — | 300 | 100.0% | **0.88** | 100.0% | **0.94** |

随机方向控制下 top-1 切换率 < 2%，证明方向本身而非范数起作用。

### 多轮轨迹（$\tau^2$-Bench Telecom，1× 不加重新拟合）
- Remove：基线调用中 **36.7%** 严格翻转（top-1 rate 99.5% → 63.0%，$\Delta z_{call} = -5.55$）。
- Add：基线文本中 **70.0%** 诱导调用（0.0% → 70.0%，$\Delta z_{call} = +19.42$）。
- 随机控制仅 1.5%/0.0%。

### 无动词请求抑制
- 160 条基线调用的子集：Remove 1× → **93.1%** 严格翻转（top-1 100.0% → 6.9%，$\Delta z_{call} = -9.95$）；1.5× → 100.0%。
- 同增益下随机控制仅 16.2%。

### 跨 7 模型汇总（Table 6 / Table 27）
- 全态 patching 恢复：93.5–100.0%。
- Suff.：0.81–1.03；Necc.：0.61–1.04。
- Domains / $\tau^2$ / Verb-free 三档迁移速率普遍 ≥ 70%，部分达 100%。
- $K_{corrupt}/K_{clean}$ 在所有模型均 > 1（1.30–4.18），抑制占优一致。

## 相关工作脉络
1. **ToolFormer / Gorilla / ToolLLM / BFCL / Metatool / API-Bank**：行为评测体系，聚焦"选对工具/填对参数/完成多步"，把调用决策当黑盒；本文把决策本身打开为可定位、可干预的内部状态。
2. **ACT/ReAct 等 agent 交互协议**：强调推理-行动交替的流程设计，未触及模型内部"何时启动行动"的表征；本文与之正交，可叠加进任何协议层做可解释性诊断。
3. **Mechanistic interpretability 短提示路线**（IOI、induction、fact vectors）：在 10–30 token 上做单变量对比；本文首次把 activation patching 与 DLA 搬到数百 token 的 agent 提示上，并用"动词替换"实现长提示的受控变量提取。
4. **Refusal direction（Arditi et al., 2024）**：安全对齐模型的拒绝行为由单一残差方向介导，结构上先验 + 抑制同源；本文发现"调用决策"同样遵循此架构，但定位更精确（L24 单位置）、跨域迁移更强。
5. **Representation engineering / Activation steering / Function vectors**：用对比对提取行为方向并注入；本文方向从单次动词对比中提取，但在不同任务域、无动词、多轮轨迹上不需重新拟合即生效，比既有 work 更强推广。
6. **Transcoders / SAEs**：非线性 MLP 分解工具；本文用 Transcoder 量化 L20–L23 中 suppressor 特征族的相对贡献，区别于仅重建输入的 SAE 路线。

## 局限性与未来方向
- **任务范围受限**：只关注"调不调"二元决策，未评测工具选对、参数有效、任务完成等更宽正确性；发现的是 action-initiation 通路而非 end-to-end agent 能力。
- **发现提示高度结构化**：形成分析基于行为筛选过的编码配对 + 固定 scaffold，对其他 prompt 构造（如自由聊天、多工具编排、少/无 schema）的外推需进一步验证。
- **精细计算未解析**：只定位到"前验 + 抑制家族"层面，未给出"scaffold 信息 + 请求语义如何映射到 $\mu_\Delta$"的细粒度计算图；ACDC/EAP-IG 的高 sufficiency 因网络泛化准备度所致，未能给出唯一稀疏电路。
- **形式化验证缺失**：跨模型复现是统计证据，尚未给出形式化的跨架构等价性定理。
- **下游应用未展开**：向量虽可因果操控，但如何用作 agent 系统级调控（如防止过度调用、提升安全性、跨模型统一 monitor）仍待探索。

## 研究启发与可借鉴点
1. **"动词断点法"构造长提示对比对**：在复杂 agent 提示中，挑一个"行为翻转敏感词"（动词、形容词、限定词）作为单变量断点，其余全保持，可把数百 token 降维成可控 mechanistic 接口；该方法可迁移到 tool selection、argument formatting、refusal 等其他 agent 行为。
2. **先验 + 抑制的统一分析框架**：不仅适用于调用决策，也适用于任何由格式模板/系统提示"预激"、再由内容触发抑制/促进的 agent 行为（如多步规划中"是否继续"、function calling 中"是否调用多个工具"）。
3. **跨域零重估迁移作为强鲁棒性指标**：保持方向固定、仅范数校准甚至直接 1× 施加，能在 web search / SQL / email / $\tau^2$-Bench / 无动词等 5+ 设定下复用，这比仅在源域内做消融更有说服力；未来 work 可参考这一迁移协议设计评估标准。
4. **Transcoder 分解 MLP 形成窗口**：把 L20–L23 的 MLP 输出按 feature $\times$ $\kappa$ 分解，区分"执行侧主动生成"与"分析侧抑制写入"，并给出可操作的因果阻断实验（归零最高 $\kappa$ 抑制特征）；方法可复用到其他 MLP 主导的行为。
5. **分布式读出设计**：单头/单 MLP 移除影响有限、组合组消融才显著，提醒我们在定位 circuit 时需避免"单一 bottleneck"假设，改用 group ablation + 联合 DLA 的综合度量。

## 关键术语表
- **Call-or-no-call decision**：agent LLM 在生成首个 token 时决定输出 `<tool_call>` 还是普通文本的二元决策。
- **Tool-call vector $\mu_\Delta$**：执行与分析报告侧在层 24 预测位置残差的均值差，因果充分必要地控制上述决策。
- **Prompt scaffold**：由 role instructions + tool schemas + format templates 构成的系统级框架，单独即可安装接近天花板工具调用先验。
- **Execution verb / Analysis verb**：请求动词的两类；前者要求对代码执行动作（add/build/write…），后者要求文本分析（discuss/explore/review…），二者在保留其余提示不变的条件下可翻转决策。
- **Scaffold-induced tool-call prior**：格式模板在任务型请求上安装的强调用倾向；分析动词通过激活抑制特征对抗它，执行动词基本不激活。
- **Formation window (L20–L23)**：$\mu_\Delta$ 的主要形成层区间，MLP 写入占 77.7%，其中分析侧 suppressor 特征占主导（$K_{corrupt}/K_{clean} \geq 1.34$）。
- **Scaffold-reading heads**：L29H9、L33H11、L33H29 等在执行侧强转向 format template 区域并直接支持 `<tool_call>` 的注意力头群。
- **Transcoder**：近似 MLP 输入→输出映射的稀疏字典方法，把每层输出分解为 $\sum_f a_{\ell f}(x) w_f^{(\ell)}$，用于定位 suppressor 特征族。

## 可复现要素
- **数据集**：作者独立构建的 500 对/模型编码配对提示 + $\tau^2$-Bench Telecom 多轮轨迹 + 自造无动词请求 600 条；均在 HuggingFace 公开（`XijieGong/MI4ToolCalling`）。
- **代码**：实验代码、prompt 构造、Transcoder 训练脚本开源于 `https://github.com/XijieGo/MI4ToolCalling`（Apache 2.0）。
- **Transcoder 权重**：新训练的 per-model Transcoder checkpoints 在 HuggingFace 仓库中提供。
- **关键超参**：
  - 训练/保留对：300 / 200（每模型）；跨域/无动词用额外独立对。
  - 干预层：Qwen3-8B 取 L24 预测位置；各模型见表 27。
  - 跨域范数校准：保留方向固定，仅按目标域训练对重新缩放 $\|\mu_\Delta\|$。
  - $\tau^2$-Bench / 无动词：固定 1× gain 直接施加，不重新拟合。
  - Transcoder 训练：约 800 NVIDIA B200 GPU-hours；复现实验可用单张 RTX PRO 6000 Blackwell 96GB。
- **复现难度**：中等；需要冻结权重推理 + activation patching + 单样本 TransformerLens/HuggingFace 双协议一致性校验。附录 D.3 给出两套协议的 logit-lens / DLA 数值一致性说明。
