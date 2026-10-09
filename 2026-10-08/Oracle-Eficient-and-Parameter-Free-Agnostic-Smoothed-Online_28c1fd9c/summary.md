---
title: "Oracle-Eficient-and-Parameter-Free-Agnostic-Smoothed-Online"
source: https://arxiv.org/pdf/2610.10499v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:52:00"
---

# 论文速读：Oracle-Eficient-and-Parameter-Free-Agnostic-Smoothed-Online

## 一句话总结
本文提出了在线学习领域中首个无需知晓基底测度 $\mu$、且完全参数自由的 Oracle-高效算法，在 agnostic 平滑在线学习设定下实现了 $\widetilde{O}(d\sqrt{T/\sigma})$ 的次线性后悔界，同时弥合了计算效率与分布鲁棒性之间的长期理论鸿沟。

## 研究问题与动机
- **核心问题**：在协变量具有强时序依赖或对抗性变化的在线学习场景中，如何在**不依赖数据分布先验知识**（未知基底测度 $\mu$）且**标签不可被固定假设完美预测**（agnostic 设定）的情况下，设计计算高效的在线学习算法？
- **现有方法为何不足**：
  1. 纯对抗在线学习存在硬性计算壁垒：Hazan & Koren (2016) 证明仅靠 ERM oracle 无法实现通用在线学习的高效可学习性，必须引入额外结构假设。
  2. 现有平滑在线学习算法虽能突破统计与计算壁垒，但普遍要求算法**已知基底测度 $\mu$ 并能从中采样**，
