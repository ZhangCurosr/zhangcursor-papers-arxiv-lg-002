---
title: "LEARNING-WHEN-AND-HOW-TO-INTERVENE-A-HINDSIGHT-DISTILLED-SEN"
source: https://arxiv.org/pdf/2609.39957v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 21:58:16"
---

# 论文速读：LEARNING-WHEN-AND-HOW-TO-INTERVENE-A-HINDSIGHT-DISTILLED-SEN

## 一句话总结
提出 HISENTINEL 框架，通过将在完整轨迹结束后才可观测的执行后验知识蒸馏至仅见前置上下文的轻量级因果模型，使其在动作执行前精准选择 ALLOW / REDIRECT / HARD-PASE 并生成可操作反馈，从而有效阻断错误传播、提升 Coding Agent 的最终任务完成率。

## 研究问题与动机
- **核心问题**：Coding Agent 在多步交互中极易因单步错误导致错误扩散，如何在动作实际执行前判断干预是否真能提升最终任务完成率？
- **现有方法不足**：过程批判（Process Critics）与轨迹诊断多在动作执行后提供反馈或事后归因，无法阻止有害动作已发生的环境改变；仅靠前缀监控的召回率极低（Terminal-Bench 数据显示仅 3.7–8.7%）。
- **信息鸿沟**：执行结果是判断动作是否恰当的最直接证据，但该证据在执行前天然不可得，导致前置干预缺乏监督信号。
- **动机**：将干预时机前移至“动作提议后、环境执行前”，利用历史轨迹的执行后验训练轻量 Sentinel，使其在无未来信息条件下仍能做出面向任务完成的干预决策。

## 核心贡献（创新点）
1. **提出面向任务完成率的三向干预决策框架**（ALLOW / REDIRECT / HARD-PASE），目标从“纠正局部瑕疵”转向“提升最终完成率”。*与已有工作的本质区别：区别于事后诊断与周期性质量检查，本文聚焦于执行边界的预防性控制与完成率导向的收益评估。*
2. **构造 SWE-INTERVENE 动作级数据集**，从真实软件工程轨迹中抽取决策节点并标注三向标签与对应反馈。*区别：填补了缺乏细粒度、含自然 HARD-PASE 场景的干预训练数据空白，且通过 AUQ 重构补齐稀缺的求助样本。*
3. **设计 Hindsight-Guided Intervention Distillation (HGID)**，利用特权教师蒸馏后验知识至因果学生。*区别：引入 Helpfulness Gate 仅在未来信息显著提升预测置信度时激活蒸馏，避免不可实现的未来线索污染学生模型。*
4. **路由与反馈解耦联合优化**：原生 LM 头生成路由标签，路由冻结后通过 SFT + DPO 独立优化反馈生成质量。*区别：防止反馈微调反向干扰干预决策分布，同时保证输出具备可落地执行的纠正或澄清能力。*

## 方法详解
- **问题形式化**：在步 t，输入 $x_t = (u, h_t, a_t)$（任务描述、轨迹前缀、提议动作），输出干预路线 $\hat{y}_t \in \{ALLOW, REDIRECT, HARD\text{-}PASE\}$ 与反馈 $r_t$。$y_t = \mathcal{A}(x_t, e_t)$ 由后验证据 $e_t$（记录的后继轨迹与任务结果）决定，但部署时 $e_t$ 不可见。
- **SWE-INTERVENE 构建**：来源为 OPEN-SWE-TRACES、SWE-HERO、SWE-CHAT。GPT-5.5 依据三问准则标注；SWE-CHAT 的 AskUserQuestion 交互用于重构 HARD-PASE 样本。两阶段人工审计剔除 25.3% 候选，最终 5,680 训练 / 1,243 测试样本，按轨迹级拆分以防泄漏。
- **HGID 蒸馏**：特权教师（冻结的 Qwen3-Coder-30B-A3B-Instruct）可见 $x_t$ 与未来 $f_t$，训练二分类 $b_t \in \{ALLOW, INTERVENE\}$；学生仅见 $x_t$，训练三分类。聚合学生概率对齐教师：$\bar{p}_\theta(INTERVENE) = p_\theta(REDIRECT) + p_\theta(HARD\text{-}PASE)$。
- **Helpfulness Gate**：仅当 $\arg\max_b p_\phi(b|x_t,f_t) = b_t$ 且 $p_\phi(b_t|x_t,f_t) > p_\phi(b_t|x_t,\emptyset)$ 时激活蒸馏，损失为 $\mathcal{L}_{HGID} = \mathcal{L}_{route} + \lambda \mathbb{E}[m_t \tau^2 D_{KL}(p_\phi^\tau(\cdot|x_t,f_t) || \bar{p}_\theta^\tau(\cdot|x_t))]$。
- **反馈学习**：路由通过原生生成头输出（无需额外分类头）。选定路由后，反馈模块先 SFT 学习结构化纠正/求助文本，再以 DPO 优化偏好对，损失 $\mathcal{L}_{feedback} = \mathcal{L}_{NLL}(r^+) - \eta \mathbb{E}[\log\sigma(\beta\Delta_\psi)]$，路由参数全程冻结。
- **部署与控制流**：Sentinel 仅接收因果前缀与动作提议；ALLOW 直接放行；REDIRECT 挂起动作并输出纠正建议供 Agent 重新提议；HARD-PASE 挂起并调用 Proxy-User API 获取最小必要信息，严格隔离任务规范以防泄漏。

## 实验与结果
- **基准与设置**：静态识别（SWE-INTERVENE、RootSE、R-Judge）；端到端（SWE-bench Verified Mini 50 题、Ask or Assume 100 题）。基线涵盖 Step-by-Step、SWE-PRM、Steer Don't Solve 及多种事后诊断方法。
- **静态识别**
