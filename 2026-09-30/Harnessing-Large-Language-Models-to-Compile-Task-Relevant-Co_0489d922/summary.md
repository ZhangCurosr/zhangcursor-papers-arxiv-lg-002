---
title: "Harnessing-Large-Language-Models-to-Compile-Task-Relevant-Co"
source: https://arxiv.org/pdf/2609.36788v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:37:47"
---

# 论文速读：Harnessing-Large-Language-Models-to-Compile-Task-Relevant-Co

## 一句话总结
本文提出 **HarBO**，一种专为贝叶斯优化（BO）设计的 LLM harness，通过 Z/D/GP/R 分阶段工作流将多样化任务上下文（领域知识、先验观测、噪声尺度、漂移状态等）自动编译为标准高斯过程（GP）的模型与数据构件，使非专业用户也能将丰富语境可靠地融入 BO 循环。

## 研究问题与动机
1. **上下文注入门槛高**：传统 BO 难以直接吸纳领域文献、先验记录与外部观测等丰富信号，现有知识注入方法（空间扭曲、采集偏置、领域表征等）要求使用者具备较强的数学与 BO 专业知识。
2. **LLM 编码实践缺乏理论刻画**：随着 “vibe coding” 兴起，领域研究者可通过自然语言将任务委托给 LLM 自动生成并运行 BO 程序，但“上下文→代码→优化策略”的隐式路径尚未被形式化理解。
3. **通用 harness 易退化**：通用编码 agent 面对多样上下文时倾向于生成统计上“最可能”的默认 BO 程序（如固定平方指数核与常数均值），无法有效区分并利用特定领域信号。
4. **可靠性与理论保障缺失**：LLM 编译的 GP 构件可能偏离真实条件信念，需建立编译失真下的 regret 界限与验证机制，以判断该路线是否具备可靠应用潜力。

## 核心贡献（创新点）
1. **将 LLM-compiled BO 形式化为广义上下文决策问题**：把上下文区分为信念通道（编译为模型/数据构件）与可选效用通道（编译为采集函数），区别于仅将上下文作为协变量的经典 contextual BO。与已有工作的本质区别在于 LLM 不再直接充当代理或采集器，而是被定位为“任务上下文到标准 GP-Bayesian 构件的编译器”。
2. **提出 HarBO 分阶段编译与验证框架**：设计 Z（潜空间）/D（伪数据）/GP（先验）/R（偶然解析）四阶段工作流，剥离 Cholesky 分解、超参训练、采集优化等纯数值任务，仅让 LLM 负责结构生成；配套包含类型、范围、正定性、后验烟测的验证器电池与有界重试机制。与已有工作的本质区别在于通过阶段契约把 LLM 的不确定性限制在构件生成层，而数值推理保持确定性与可审计性。
3. **建立编译失真下的 regret 理论边界与上下文信息增益分析**：推导 GP-UCB 在 KL 编译失真下的期望 regret 界（要求平均失真 $\bar{\kappa}_T \to 0$ 方可次线性），并证明新增可见上下文未必收紧探索上界（信息减少收益可能被策略适应成本抵消）。与已有工作的本质区别在于首次为 LLM 编译 BO 的可靠性提供理论解释，而非仅依赖经验对比。

## 方法详解
- **广义上下文决策设定**：第 $t$ 轮系统接收 $c_t$（含领域证据、文件、对话、环境报告与历史观测），策略 $\pi(x_t|c_t)$ 经信念编译器 $\mathcal{C}_\mathrm{B}$ 输出模型构件 $\mathcal{M}_t$ 与数据构件 $\mathcal{D}_t$，经贝叶斯更新得代理 $\hat{f}_t \sim p(f_t|\mathcal{M}_t,\mathcal{D}_t)$；可选效用编译器 $\mathcal{C}_\mathrm{U}$ 将指令 $\delta_t$ 编译为采集函数 $a_t$，最终最大化选 $x_t$。
- **上下文三分解**：
  - **认知上下文 $c^{ep}$**：相对稳定的领域知识、文献、先验实验记录，用于编译可复用构件。
  - **偶然上下文 $c_t^{al}$**：每步动态、不可控的环境状态及其可靠性。
  - **历史 $\mathcal{H}_t$**：过往 $(c_i^{al}, x_i, y_i)$ 元组，递归累积。
- **四阶段编译流水线**：
  - **Z 阶段**：从 $c^{ep}$ 推断潜协变量空间 $\mathcal{Z}$ 及 `sample/contains` 接口。
  - **D 阶段**：将 $c^{ep}$ 编译为认知伪数据 $\mathcal{D}^{ep}=\{(x_i^{ep},z_i^{ep},\sigma_i^{ep},y_i^{ep})\}$，编码可靠点级证据；不泄露的结构/定性信息留给 GP 先验。
  - **GP 阶段**：编译先验均值 $\mu^{pr}$ 与核 $k^{pr}$ 的工厂函数（支持自定义 `to_row` 与特征映射），明确要求将上下文数值硬编码为张量或 mild 初始化的可学习参数。
  - **R 阶段**：对每步 $c_i^{al}$ 解析出当前潜态 $z_i$ 与观测噪声尺度 $\sigma_i$（UNK 表示未知），并入历史数据 $\mathcal{D}_t^\mathrm{hist}$。
- **验证引擎**：每阶段产出均通过硬约束校验（成员关系、值域、PSD、后验可算性），失败时返回具体违反条件并由 LLM 有界修复（最多 3 次）；数值推理（Cholesky、empirical Bayes、采集最大化）由宿主引擎完成。
- **两种实现**：In-process（主机严格控制阶段顺序，隔离通用 agent 行为用于可控实验）与 Agentic（嵌入 OpenCode 等编码 harness，通过持久化 Markdown/JSON/Python 文件维持状态，由 campaign skill 软约束引导）。

## 实验与结果
- **环境与基线**：9 个基准（合成 A–E、XGBoost HPO/HM、KRAS G12D 分子对接 Dock、O-Suzuki 反应优化）；基线含 Vanilla GP-UCB、Embedding-CGP-UCB、Embedding-NNAGP-UCB、Random、CAKE、LGBO、LLAMBO、LABO；LLM 使用 DeepSeek
