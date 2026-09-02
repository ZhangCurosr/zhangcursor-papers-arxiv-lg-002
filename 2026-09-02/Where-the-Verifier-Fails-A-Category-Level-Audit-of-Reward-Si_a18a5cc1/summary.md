---
title: "Where-the-Verifier-Fails-A-Category-Level-Audit-of-Reward-Si"
source: https://arxiv.org/pdf/2609.01354v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 11:49:48"
field: "强化学习与可验证奖励"
keywords: ["RLVR", "verifier reliability", "metamorphic testing", "reward signals", "mathematical reasoning evaluation"]
innovations: ["将变换测试从模型审计转向验证器认证审计", "发现验证器自验证率跨实现差异达41.3点且主要源于空白/标点处理", "揭示基于答案幅度的阶跃式假阳性机制（阈值10⁴）"]
benchmarks: ["MATH", "GSM8K", "Big-Math"]
---

# 论文速读：Where-the-Verifer-Fails-A-Category-Level-Audit-of-Reward-Signals-in-RLVR

## 一句话总结
本文首次系统审计了RLVR（带可验证奖励的强化学习）和数学基准评测中自动验证器的可靠性，发现验证器实现间自验证率差异高达41.3个百分点，且错误主要来自空白/标点处理而非语法解析；同时揭示了一种阶跃式的规模依赖假阳性机制。

## 研究问题与动机
- RLVR和数学基准评测（如MATH、GSM8K）都依赖自动验证器将自由文本答案转为二元奖励信号，但验证器并非数学预言机——它是提取子串、归一化并比较的程式，每个设计决策都有失败模式
- 现有工作报道了"标准评测 harness 约94%自验证准确率，归因于LaTeX解析"，但这只是聚合数字，未说明哪些答案形式消耗了误差预算
- 若验证器拒绝正确答案，策略会因应奖励的行为受罚；若接受错误答案，策略会因错误行为得利
- 修正误差需要误差率作为参数，系统性误差分析需要误差形状而不仅是幅度，但可靠性文献只报告聚合指标

## 核心贡献（创新点）
- **认证变换测试协议**：将变换测试从模型审计转向验证器审计，通过构造语义保持的变换T使(g, T(g))对具有已知正确裁决，任何拒绝都是认证假阴性，无需人工裁决
- **分类分解审计**：在43个变换、14个层次上对4个主流验证器产生307,420个裁决，每个单元格带Wilson置信区间
- **合约矩阵分离缺陷与歧义**：定义验证器声明的合约范围，仅在声称处理的层次评分，避免将规格歧义误判为缺陷
- **覆盖率指标**：区分拒绝（返回FALSE）和执行失败（崩溃/超时），两者需不同修复策略
- **规模依赖假阳性机制识别**：发现sympy-cascade验证器对off-by-one误差的接受率随答案幅度呈阶跃函数（阈值10⁴），这是完全系统性的且对聚合指标不可见

## 方法详解
- **认证等价变换**：设g为正确答案，T为变换。若T由构造保证语义保持，则T(g)正确，验证器V返回V(g,T(g))=FALSE即为认证假阴性。相反，对改变意义的变换T'，接受即为认证假阳性
- **三类变换**：
  - CERTIFIED-EQUIV：数学等价且在验证器合约内，拒绝计为假阴性
  - CONTRACT-Dep：依赖声明合约（boxing、units、text wrappers），报告为规格歧义而非缺陷
  - ADVERSARIAL：改变意义，接受计为假阳性
- **覆盖度定义**：n_eval = n_TRUE + n_FALSE，coverage = n_eval/n；分离裁决准确度与裁决可用性
- **四个被测验证器**：
  - math-verify (LaTeX提取，库默认配置)
  - math-verify (纯表达式提取)
  - DeepSeek-Math系字符串归一化器 (strip-string)
  - 三级级联参考验证器 (精确字符串→数值相对容差10⁻⁴→SymPy符号)
- **数据**：4,990个唯一gold答案（GSM8K、MATH七科、Big-Math）+ 2,000合成答案（用于补充集合/区间表示等低频形式）
- **合约矩阵**：记录每类验证器声称处理的层次，out-of-contract行为报告但不计入缺陷

## 实验与结果
- **数据集**：GSM8K、MATH（7科）、Big-Math、合成MATH答案形式；共307,420个裁决
- **自验证率差异**：53.8% (mv-expr) 到 95.2% (strip-string)，跨度41.3点；同一库的两配置在49.9%成对上 disagreement
- **错误层次分解**：
  - mv-latex：空白/标点占93.0% failures
  - mv-expr：空白/标点占74.1%
  - strip-string：空白50.3% + 分数/小数31.7%
  - sympy-cascade：未约分52.0%（全是解析异常，非拒绝）
- **覆盖率分离**：
  - sympy-cascade：coverage 87.3%，但其所有裁决正确（100% judged正确），残差全是执行失败
  - strip-string/mv-latex：coverage 100%，通过拒绝犯错
- **规模依赖假阳性**：sympy-cascade对off-by-one接受率：<10⁴为0%，≥10⁴为100%（相对容差10⁻⁴尺度不变性导致）；aggregate rate 16.0% (95% CI [14.1, 18.1], n=1,287)
- **合约依赖输入**：boxed答案接受率0%-75.1%；scientific notation 0%-100%；不同实现按设计行为但差异未文档化
- **鲁棒性检验**：排除合成gold后排序和量级不变，结论稳健

## 相关工作脉络
- **Cai et al. (2025)**：将验证器建模为带不对称噪声率ρ₀、ρ₁的随机奖励通道，推导策略梯度修正；需要假阴性率作为输入——本文测量并分解这些率
- **Egashira et al. (2026)**：区分系统性vs随机验证误差，指出系统性假阳性可致plateau或collapse；本文per-stratum分解刻画该形状，幅度阈值是最大系统性实例
- **Huang et al. (2026)**：规则检查器与模型裁判反向失败——规则侧脆性解析，模型侧reward hacking；本文扩展假阴性侧并添加认证对抗探针
- **Ammanamanchi et al. (2026)**：审计5个Lean定理证明基准，发现4,833个问题；本文对自然语言答案验证采用相同立场
- **Zhang (2026)**：代码RLVR中hardened vs leaky reward对照实验；本文聚焦数学答案验证
- **Lan et al. (2026)**：安全扫描器评估中coverage须独立报告——本文在验证器评估中采用同样论证
- **Asgari et al. (2025), Hyun et al. (2024)**：变换测试传统应用于模型鲁棒性；本文反转目标至评分器

## 局限性与未来方向
- **验证器级别非模型级别**：未估计对任何模型benchmark score的影响，真实输出不按等速率产生变换变体
- **无训练实验**：未测量对下游RLVR效果的影响（Zhang 2026发现bounded effects）
- **仅开源实现**：对封闭前沿系统无声称
- **合成成分**：4,990个gold中2,000为合成，虽§3.5单独报告corpus-only结果
- **合约分配是判断**：公开矩阵供质疑
- **未来方向**：扩展到模型输出错误分布、RLVR训练动态影响、闭合系统审计、标准化验证器报告规范

## 研究启发与可借鉴点
- **变换测试框架可迁移**：将认证变换从输入扰动转向输出验证器审计，为评测组件可靠性提供通用方法论
- **覆盖率作为一级指标**：区分"裁决失败"与"未裁决"避免metric gaming，适用于任何二元判定系统评估
- **误差分解而非聚合报告**：41.3点自验证差异揭示"平均数字无意义"，推动领域从单一accuracy转向per-stratum报告
- **合约矩阵设计**：分离implementation defect与specification ambiguity，为API/工具评估提供审计范式
- **可结合团队方向**：若团队做RLVR/数学推理，需报告验证器配置、版本、提取方式；可复现本文transform suite审计自家pipeline

## 关键术语表
- **RLVR**：Reinforcement Learning with Verifiable Rewards，基于可验证奖励的强化学习，用二元自动奖励替代人工标注
- **Metamorphic Testing（变换测试）**：通过语义保持变换生成等价输入/输出对，检测系统一致性错误的测试方法
- **Certified False Negative**：认证假阴性，经构造保证答案等价却仍被拒绝的情况，无需人工仲裁
- **Contract Matrix（合约矩阵）**：记录验证器声明处理的answer form类别，用于区分实现缺陷与规格歧义
- **Coverage（覆盖率）**：验证器返回裁决的比例，分离执行失败与错误裁决的指标
- **Scale-invariant Tolerance**：尺度不变容差，相对误差阈值导致对大数和小数接受率相同，引发幅度依赖阶跃假阳性
- **Self-validation（自验证）**：验证器对自身ground truth答案的接受率，衡量内部一致性

## 可复现要素
- **代码/数据**：https://github.com/ethxin0011/verifier-error-budget（公开transform suite、contract matrix、per-sample verdict records）
- **数据集**：GSM8K、MATH（开源）、Big-Math（开源）、合成MATH答案（合成部分开源）
- **验证器版本**：ANTLR4 runtime pinned to 4.13.2；math-verify version 0.9.0
- **超参**：每item 5秒超时预算；相对容差10⁻⁴（sympy-cascade）
- **实验环境**：Azure ML parallel pipeline，5个CPU节点
