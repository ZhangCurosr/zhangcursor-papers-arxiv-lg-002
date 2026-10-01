---
title: "T1-Terminal-Agent-Reinforcement-Learning-for-Long-Horizon-Ta"
source: https://arxiv.org/pdf/2609.11042v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:37:29"
---

# 论文速读：T1-Terminal-Agent-Reinforcement-Learning-for-Long-Horizon-Ta

## 一句话总结
本文针对长 horizon 终端智能体任务，提出 T1（基于 Qwen3.5-122B-A10B 的 122B MoE 模型），通过 PPO 在云端真实 Shell 沙箱中进行最多 300+ 轮 tool-call 的强化学习训练，并引入 TITO/R³ 训练-推理一致性机制与基于断言计数的密集验证奖励，在 Terminal-Bench 2.1 上达到 64.0% 解决
