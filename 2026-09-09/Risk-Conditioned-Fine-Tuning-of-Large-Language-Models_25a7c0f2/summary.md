---
title: "Risk-Conditioned-Fine-Tuning-of-Large-Language-Models"
source: https://arxiv.org/pdf/2609.08064v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 03:02:32"
---

# 论文速读：Risk-Conditioned Fine-Tuning of Large Language Models

## 一句话总结
本文首次将风险条件化强化学习引入大语言模型对齐领域，提出 **Risk-Conditioned RLHF**，通过学习单一策略 π(·|x,α) 并在推理时通过调节风险参数 α 实现连续的风险厌恶控制，避免了 RA-RLHF 固定 α 训练的局限与多模型部署的高昂成本。

## 研究问题与动机
- **尾部风险失效**：标准 RLHF 优化期望奖励，高安全响应会平均掉低质量尾部，导致极低概率但后果严重的有害输出（医疗、法律、灾害管理等高 stakes 场景）仍无法被有效控制。
- **现有方法局限**：RA-RLHF（Chaudhary et al., 2024）虽引入 CVaR 优化尾部风险，但仅训练固定 α 的单一策略，推理时无法连续调节风险厌恶程度；朴素的多模型训练/部署方案计算与存储成本过高（Wang et al., 2024c; Girija et al., 2025）。
- **条件化视角缺失**：多目标微调相关工作多聚焦 reward weight 或偏好条件化，缺乏对“风险水平”这一维度的显式建模与连续控制机制。

## 核心贡献（创新点）
- **首次构建 LLM 风险条件化对齐框架**：将 CVaR 风险敏感 RL 思想适配至 LLM 对齐，学习单一策略 π(·|x,α)，支持推理时通过 α 实现连续风险调节与未见风险水平的插值。
- **参数高效的条件注入机制**：提出 Prompt-conditioned、Logit-conditioned 与 Attention-conditioned 三种风险注入方式，以极低额外参数量（最高 +2.89%）实现风险水平的显式建模，共享参数在所有 α 间复用。
- **理论收敛性与风险前沿一致性保证**：推导联合优化算法的梯度更新规则，证明在常数步长下达到 stationary point 的误差界，并给出风险前沿在离散网格上的一致逼近上界（Theorem 2/3）。
- **统一评估协议与多尺度验证**：在 Pythia-70M/2.8B 与 Llama-3.1-8B-Instruct 上验证，保留测试风险水平下性能逼近 Oracle 基线，显著优于固定策略与 prompt mixing 方案。

## 方法详解
- **优化目标**：$\max_\pi \mathbb{E}_{\alpha\sim p(\alpha)}\mathbb{E}_{x\sim\mathcal{D}}\!\left[\mathrm{CVaR}_\alpha(G(x,Y;\alpha))\right]$，其中 $G(x,Y;\alpha)=r(x,Y)-\beta\log\frac{\pi_\theta(Y|x,\alpha)}{\pi_{ref}(Y|x)}$ 为 KL 正则化奖励，α 采样自支持区间 $[α_{min}, α_{max}]\subset(0,1]$。
- **变分形式与阈值网络**：利用 $\mathrm{CVaR}_\alpha(Z)=\max_\eta\{\eta-\frac{1}{\alpha}\mathbb{E}[(\eta-Z)_+]\}$，将阈值 η 参数化为可微网络 $\eta_\omega(x,\alpha)$，转化为联合目标 $\mathcal{I}(\theta,\omega)=\mathbb{E}[\eta_\omega-\frac{1}{\alpha}\mathbb{E}[(\eta_\omega-G)_+]]$。
- **梯度更新规则**（Theorem 1）：
  - $\nabla_\omega\mathcal{I}=\mathbb{E}\left[\left(1-\frac{1}{\alpha}\mathbf{1}\{G\le\eta_\omega\}\right)\nabla_\omega\eta_\omega\right]$
  - $\nabla_\theta\mathcal{I}=\mathbb{E}\left[\left(u_{\theta,\omega}-\frac{\beta}{\alpha}\mathbf{1}\{G\le\eta_\omega\}\right)\nabla_\theta\log\pi_\theta\right]$
  - 随机估计误差下界为 $O(1/(BN)+1/B)$。
- **收敛性与逼近界**（Theorem 2/3）：在常数步长 $\gamma_\omega=\gamma_\theta=\Theta(T^{-1/2})$ 下，stationarity error 为 $O(T^{-1/2})(1+1/(BN)+1/B)$；若网格 $\mathcal{A}_h$ 上 ε-次优，则风险前沿最大偏差 $\sup_{\alpha}(V^*(\alpha)-V(\hat\theta,\alpha))\le\varepsilon+2Lh$（L 为 Lipschitz 常数）。
- **风险注入机制**（Section 3.3）：共享参数 $S^C$ 在所有 α 间共用，仅子集 $S$ 随 α 变化。
  - **Prompt-conditioned**：α 追加到输入文本，零额外参数，作为简单 baseline。
  - **Logit-conditioned**：最后线性层加 K 组 conditioned 参数，通过可训练 gating 网络将 α 映射为混合权重 $m_\alpha\in\mathbb{R}^K$（Pythia-70M 上额外 2.03M 参数，+2.89%）。
  - **Attention-conditioned**：条件注入 selected attention 参数，额外 0.74M 参数（+1.05%）。

## 实验与结果
- **数据集与任务**：IMDB-Gen（情感分类生成）、RealToxicityPrompts-Gen（毒性生成）、Safe-RLHF（19 类伤害评估）。
- **模型与设置**：主实验 Pythia-70M，附录扩展至 Pythia-2.8B 与 Llama-3.1-8B-Instruct；训练网格 $\mathcal{A}_h=\{0.1,0.3,0.5,0.7,0.9\}$，保留测试 $\{0.2,0.4,0.6,0.8\}$；评估指标为 CVaR_α 均值±标准差（5 seeds），辅以 gemma-4-31B-it 做 LLM judge 交叉评估。
- **计算开销**（Table 1）：RA-RLHF-Fix 与 Prompt-conditioned 零额外参数、显存 ≈13-14 GiB、训练耗时 1.00-1.08×；Logit-conditioned 额外 2.03M 参数、显存 ≈23 GiB、耗时 1.15×；Attention-conditioned 额外 0.74M 参数、显存 ≈15 GiB、耗时 1.08×。
- **Steerability 结果**（Table 2）：Safe-RLHF α=0.2 时，Risk-conditioned LM 得分为 **8.54±0.16**，RA-RLHF-Oracle 为 **8.58±0.19**，Logit-Mixing 为 **7.20±0.18**，Prompt LM 仅 **0.98±0.09**；IMDB α=0.2 时 Risk-conditioned 为 **0.67±0.19**，Oracle 为 **0.68±
