---
title: "NEURONDISCOVER-AGENT-IN-TWIN-FOR-MECHANISTIC-DISCOVERY-IN-NE"
source: https://arxiv.org/pdf/2609.35338v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-02 23:36:34"
---

# 论文速读：NEURONDISCOVER-AGENT-IN-TWIN-FOR-MECHANISTIC-DISCOVERY-IN-NE

## 一句话总结
本文提出 NEURONDISCOVER，一种基于 Agent-in-Twin 的自动化机制发现框架，通过联合建模目标机制与孪生体误差、在 MIOY 类型化图上编译可执行实验程序，并在预算约束下执行顺序前瞻验证，实现神经元微环境中竞争性物理解释的自动分离与统计裁决。

## 研究问题与动机
- **机制可辨识性困境**：神经元微环境（如 GlymphTwin 运输通路）中，扩散、流动、边界交换等不同物理机制可产生高度相似的示踪剂响应，传统观测无法区分真实因果机制。
- **孪生混杂（Twin confounding）**：真实机制变化与计算孪生体误差在稀疏观测中留下相同的响应签名，单纯追求前向预测准确性无法确证机制，导致“模型世界内准确 ≠ 物理世界真实”。
- **实验设计目标错位**：现有自适应设计多优化单一状态/参数后验，缺乏对“机制后验 vs 差异后验”的联合建模，且未显式纳入证伪机会与多重假设包络的边缘约束。
- **科学推断缺乏前瞻裁决协议**：Agent 驱动的实验流水线普遍缺少同时置信界、顺序错误分配与 Abstention（弃权）机制，难以在有限预算下输出可统计复现的关系声明。

## 核心贡献（创新点）
- **Agent-in-Twin 架构**：将 LLM 决策层与机制 grounded 的世界模型解耦并通过 MCP 协议交互，使 Agent 能在冻结假设下执行一次性 rollout，避免对 pending test 的反复篡改。
- **
