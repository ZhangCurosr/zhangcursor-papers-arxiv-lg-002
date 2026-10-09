---
title: "How-Do-Agentic-LLMs-Decide-to-Call-Tools-A-Tool-Call-Vector"
source: https://arxiv.org/pdf/2610.09624v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 02:55:23"
---

# 论文速读：How-Do-Agentic-LLMs-Decide-to-Call-Tools-A-Tool-Call-Vector

## 一句话总结
本文通过单词替换（执行动词 vs 分析动词）构建对照提示对，在残差流中定位并提取出一个紧凑的"工具调用向量"μ_Δ，该向量对大模型"是否调用工具"的二元决策具有因果必要性与充分性，并在跨工具、跨领域、多轮轨迹及无显式动词请求中实现无重估迁移，揭示了"脚手架先验 → 抑制特征写入 → 下游读出"的统一机制。

## 研究问题与动机
- 工具调用是 Agentic LLM 的核心能力，但模型"调用 vs 不调用"的内在决策机制尚未被机理化解释。
- Agentic 提示包含角色指令、工具 Schema、格式模板与用户请求（数百 token），组件高度缠绕，难以找到单一可控变量进行机制分析。
- 既往机理研究多针对 10–30 token 短提示，直接迁移到长且结构化强的 agentic 场景缺乏可行路径。
- 需要一种既简洁又可因果干预的内部表示，以支撑对工具调用决策的可解释建模与跨场景泛化。

## 核心贡献（创新点）
1. 提出"单动词对照接口"：在代码补全任务中用单个请求动词（执行/分析）作为受控变量，将长提示转化为干净的行为对比对，首次为长脚手架提示中的调用决策提供可干预入口。
2. 识别并验证工具调用向量 μ_Δ：在预测位置 Layer 24 残差状态上估计 μ_Δ，其加/减干预分别实现 100% 的首 token 翻转，具备因果必要性与充分性。
3. 揭示"先验+抑制"架构：脚手架建立任务相关的工具调用先验；分析动词激活"非必要性"语义特征族，在 Layer 21–23 沿反方向写入，抵消先验；执行动词基本保留先验。
4. 证明跨场景迁移：该向量在保持方向不变的情况下，可用于网页检索、SQL 执行、邮件发送等新工具/领域，以及 τ²-Bench 原生多轮轨迹与无动词隐式请求，且在 7 款模型（Qwen3/Qwen3.5/Mistral/Granite）上复现。
5. 打通"形成→读出"的完整链路：前端用 Transcoder 分解 MLP 输出识别抑制特征族，后端用 DLA 与单组件插值定位脚手架读取头与结构读出特征，形成端到端可解释因果图。

## 方法详解
- **对照提示对构造**：每个提示由脚手架（角色指令 R + 工具 Schema T + 格式模板 F）与用户请求拼接而成；配对仅替换首词动词（执行：add/build/complete/save/write；分析：discuss/explore/inspect/review/study），任务内容、工具集与脚手架保持不变。训练 300 对、测试 200 对，覆盖 Python/Java/C++（来源 MBPP、APPS、HumanEval、CodeContests）。
- **激活插值定位**：对每一对 (x^c, x^*)，将分析提示在层 l、位置 q 的残差状态替换为执行提示对应状态，其余位置不变；以首 token 预测为 tool_call 的比例 r(l,q) 度量恢复率。逐层扫描发现 L24 预测位置达到 100% 恢复。
- **工具调用向量估计**：μ_Δ = mean_i [ h_p^(24)(x_i^c) − h_p^(24)(x_i^*) ]，在预测位置 p 取执行减分析的均值差。
- **因果干预**：Add: h̃_p^(24)(x_i^*) = h_p^(24)(x_i^*) + μ_Δ；Remove: h̃_p^(24)(x_i^c) = h_p^(24)(x_i^c) − μ_Δ。归一化充分性 Suff(μ_Δ) 与必要性 Necc(μ_Δ) 均以 logit 差为分母度量。L24 Add 使 Suff=1.03，Remove 使 Necc=1.04，top-1 均实现 100% 翻转。
- **脚手架消融**：分离 R/T/F 评估各组件对"任务相关先验"的贡献。F 单独即可近乎满先验；T 提供对请求的敏感度；R 单独无效。
- **Transcoder 分解形成窗口（L20–L23）**：MLP^(ℓ)(x) ≈ b_dec^(ℓ) + Σ_f a_ℓf(x) w_f^(ℓ)，用 κ_ℓf = (ā_ℓf^c − ā_ℓf^*) ⟨w_f^(ℓ), μ̂_Δ⟩ 打分，按激活侧求和 K_corrupt / K_clean。L21–L23 以 corrupt-higher 为主（L23 占比 55.9%，比率 4.18），主导特征族为 analysis-non-execution（115 个特征，总贡献 7.28）。
- **下游读出分析**：DLA（Direct Logit Attribution）度量各头/MLP 对 tool_call logit 的直接贡献；单组件插值（clean→corrupt）评估因果效力。L33H29 等读取脚手架格式区，MLP34/F109925 对应工具 Schema 边界/序列化结构。
- **跨模型验证**：对 Qwen3-4B/8B/14B、Qwen3.5-4B/9B、Mistral-Small-3.2-24B、Granite-3.3-8B 分别测定位层 L*、全状态恢复率、Suff/Necc、跨域/τ²/无动词迁移及 K_corrupt/K_clean 比值。

## 实验与结果
- **数据集**：MBPP、APPS、HumanEval、CodeContests（代码任务）；τ²-Bench Telecom（多轮轨迹）；自建网页检索、SQL、邮件域；FEVER、Spider、API 域的无动词请求。
- **基线**：norm-matched random vector、随机控制头/特征、长度匹配替换 T 的基线；ACDC/EAP-IG/EAP-GP 等电路诊断；与 Refusal Direction（Arditi et al.）的对比。
- **主要结果**：
  - L24 全状态 patching 恢复 100% 分析提示的 tool_call（Qwen3-8B）。
  - μ_Δ 干预：Add 使 Suff=1.03（top-1 从 0%→100%），Remove 使 Necc=1.04（top-1 从 100%→0%）。
  - 跨域迁移（Table 2）：Web retrieval / SQL execution / Email dispatch 各 100 对，Add 均达 100% 诱导，Remove 均达 100% 抑制，平均 norm. shift Add=0.88、Remove=0.94；random 控制无效。
  - τ²-Bench 原生多轮（Table 3）：Remove 抑制 36.7% 原生调用；Add 诱导 70.0% 原生文本为调用；random 仅为 1.5%/0.0%。
  - 无动词请求（Table 3/14）：Qwen3-8B 移除 93.1%（α=1×）至 100%（α=1.5×），random 仅 16.2%。
  - 跨模型（Table 6/27）：全状态恢复 93.5–100.0%；Suff 0.81–1.03，Necc 0.61–1.04；K_corrupt/K_clean 在所有模型均 >1（1.58–3.26），复用"抑制主导"模式。
- **最强结果**：Qwen3-8B 在控制对与跨域/多轮/无动词场景下均实现 top-1 100% 翻转或接近 100% 抑制；7 模型统一复现定位与抑制主导模式。

## 相关工作脉络
- **Tool-use 评测与系统**：Toolformer、Gorilla、ToolLLM、MetaTool、BFCL 等聚焦调用准确率/选择/格式，把调用决策视为黑箱；本文首次以机理方式刻画该决策的因果表征。
- **Mechanistic 插值与定位**：Activation patching（Meng et al.、Heimersheim & Nanda）、logit lens/tuned lens、DLA（Elhage et al.）为本研究定位 L24 状态提供基础；本文将其扩展到长脚手架提示中的二元动作启动。
- **自动电路发现**：ACDC、EAP-IG、EAP-GP、Circuit Tracing；本文发现在长提示下 ACDC 的高 Suff 也出现在 random core 中，强调需以"向量级因果目标"统领解释。
- **特征分解**：SAE 用于概念/文化表征与病理研究；本文采用 Transcoder 直接分解 MLP 输入-输出映射，定位分析动词相关的抑制特征族与结构读出特征。
- **表征工程与方向干预**：Representation engineering、Function vectors、Steering（CAA/ITI）与本工作共享"单方向控制行为"的思路；但与 Refusal Direction（Arditi et al.）相比，本文的定位更精确（单层单位置 vs 全局/全层）、跨语境迁移更强（多轮/隐式/跨域）。
- **提示脚手架与认知抑制类比**：与认知神经科学中的"抑制门控/ basal ganglia 选择"形成结构类比；脚手架提供准备态，语义请求通过抑制特征刹车。

## 局限性与未来方向
- 仅刻画"调用 vs 不调用"的二元决策，未评估调用正确性（选对工具、参数合法、任务完成）。
- 形成分析基于行为过滤后的代码配对与固定脚手架，在其他提示构造下的适用性待验证。
- 未逐层细粒度揭示从脚手架信息与请求语义到 μ_Δ 的完整映射计算，仅给出宽泛抑制特征族。
- 跨域/隐式请求的复用验证了下游可用性，但未证明其与发现阶段同源的形成机制。
- 未来方向：扩展到工具选择与参数生成决策；自动化工具调用方向在真实 Agent 工作负载中的在线干预；与 refusal/安全方向统一建模；探索不同格式/多语言下的迁移边界。

## 研究启发与可借鉴点
- **"单词受控对比"范式**：在复杂长提示中，寻找一个行为决定性最小变量（动词/语气/句式）构造 clean/corrupt 对，可显著降低机理分析的工程复杂度。
- **向量化因果目标优先于电路发现**：对于复杂应用（如 Agentic prompt），先用插值定位关键残差坐标并提取向量，再用 Transcoder/DLA 解析上下游，比直接做电路剪枝更稳定、可解释。
- **"先验+抑制"架构的可迁移性**：除工具调用外，任何"默认动作倾向 + 语义刹车"的决策（如多步推理中的 early exit、计划/执行切换、拒绝/接受）均可套用同一分析框架。
- **跨域无重估迁移的实验设计**：保持方向不变、仅在目标域校准范数，并在多轮轨迹与隐式请求上验证，可强有力地证明表征的通用性。
- **结合脚手架消融的归因策略**：将 R/T/F 等组件逐一剥离，能明确先验来源（格式模板）与敏感性来源（Schema 语义+长度），为 prompt 设计提供可操作的因果证据。

## 关键术语表
- **Tool-call vector（工具调用向量）μ_Δ**：在预测位置残差流中，由执行与分析提示的平均激活差构成的方向向量，对"是否调用工具"决策具因果必要性与充分性。
- **Call-or-no-call decision**：Agentic LLM 在每个提示位置选择输出 tool_call 标记还是普通文本的首 token 二元决策。
- **Activation patching**：将 corrupt 样本在指定层/位置的残差替换为 clean 样本对应状态，检验该状态对输出的因果贡献。
- **Prompt scaffold（提示脚手架）**：由角色指令、工具 Schema 和格式模板拼接而成的结构化前置上下文，建立调用先验。
- **Scaffold-induced call prior（脚手架诱导调用先验）**：在任务相关输入上，脚手架使模型默认倾向输出 tool_call 的基础概率。
- **Suppression features（抑制特征）**：在分析请求上被激活、沿 μ_Δ 反方向写入残差流的功能特征，抵消调用先验。
- **Direct Logit Attribution（DLA）**：将注意力头或 MLP 层的输出对 final logit（此处为 tool_call logit）的直接贡献进行量化。
- **Transcoder**：对 MLP 层的输入-输出映射做稀疏字典分解，同时获得特征的激活与写入方向，优于 SAE 对激活本身的拟合。

## 可复现要素
- **代码**：https://github.com/XijieGo/MI4ToolCalling（Apache 2.0）。
- **数据集**：https://huggingface.co/datasets/XijieGong/MI4ToolCalling（已开源，含模型专属配对 manifest）。
- **Transcoder 权重**：https://huggingface.co/XijieGong/MI4ToolCalling（已开源）。
- **基座模型**：Qwen3-4B/8B/14B、Qwen3.5-4B/9B、Mistral-Small-3.2-24B-Instruct-2506、Granite-3.3-8B-Instruct（均 Apache 2.0）。
- **关键超参**：训练/测试对拆分 300/200；干预层 L*（模型相关，如 Qwen3-8B 为 L24）；跨域范数校准但不重估方向；τ²/无动词实验使用固定增益（1×/1.5×）。
- **算力声明**：Transcoder 训练约 800 GPU 小时（NVIDIA B200）；其余实验单卡 NVIDIA RTX PRO 6000 Blackwell 96GB 可复现。
- **其他数据许可证**：APPS/HumanEval（MIT）、MBPP/CodeContests/τ²-Bench（MIT 或 CC BY 4.0）、FEVER（CC BY-SA 3.0）、Spider（CC BY-SA 4.0 / Apache 2.0）。
