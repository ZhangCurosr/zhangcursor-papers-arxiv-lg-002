---
title: "MemFLoRA-Memory-Floor-LoRA-for-CNN-Adaptation-at-the-Edge"
source: https://arxiv.org/pdf/2610.08669v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:52:41"
---

# 论文速读：MemFLoRA-Memory-Floor-LoRA-for-CNN-Adaptation-at-the-Edge

## 一句话总结
本文针对边缘设备CNN部署后遭遇用户/传感器/环境分布偏移需在线适配的难题，提出以“激活内存下限”为核心的低秩适配器MemFLoRA。通过冻结下投影、训练尺度匹配的上投影、切换评估模式BatchNorm及最小化反向激活规则，该方法在不牺牲适配性能的前提下，将训练保留的激活状态内存较全量微调降低98.5
