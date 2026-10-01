---
title: "Through-the-Eyes-of-the-Beholder-Biometric-and-Demographic-C"
source: https://arxiv.org/pdf/2609.15608v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 21:58:25"
---

# 论文速读：Through-the-Eyes-of-the-Beholder-Biometric-and-Demographic-C

## 一句话总结
本文针对网络模因（meme）中隐蔽性性别歧视内容的检测难题，提出了一种以人为本的多模态框架，将标注者的眼动、心率变异性与人口统计学特征通过FiLM条件层注入跨注意力融合网络，并将主观分歧建模为标签分布学习问题；该系统在CLEF 2026 EXIST Task 2的子任务2.2（源意图判断）软评测中位列第29名，整体性能稳定优于组织方基线。

## 研究问题与动机
- **核心问题：** 如何有效检测依赖反讽与图文互补的模因性别歧视内容，并合理建模人类标注的主观分歧而非简单压制
