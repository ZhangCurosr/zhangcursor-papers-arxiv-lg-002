---
title: "The-Multiple-Timescales-of-Gradient-Descent-on-the-Edge-of-S"
source: https://arxiv.org/pdf/2609.01034v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:04:23"
---

# 论文速读：The-Multiple-Timescales-of-Gradient-Descent-on-the-Edge-of-S

## 一句话总结
本文在“尖锐山谷（sharp valley）”损失假设下，利用奇点摄动理论中的多尺度法，从数学上形式化推导了梯度下降在稳定性边缘（Edge of Stability, EoS）的连续时间极限即为 Cohen 等人提出的“中心流（central flow）”，并完整刻画了快（振荡）、中（自稳定化）、慢（沿山谷演化）三个时间尺度上的动力学解耦机制。

## 研究问题与动机
- **EoS 现象缺乏严谨推导基础**：深度网络训练中 Hessian 最大特征值（sharpness）常围绕 $2/\eta$ 波动，Cohen 等人（2025）提出中心流作为其连续时间近似，但推导依赖局部三次近似与模糊的“局部时间平均”算子 $\mathbb{E}$，缺少明确的极限 regime。
- **经典梯度流框架失效**：vanishing learning rate（$\eta \to 0$）下的梯度流无法覆盖 EoS，因为该 regime 下 sharpness 永远低于 $2/\eta$，边缘效应根本不会触发。
- **自稳定化机制在多特征值情形未明**：Damian 等人（2023）已刻画单特征值触及 EoS 时的自稳定化环路，但网络训练中常出现多个特征值同时处于边界且波动不衰减的现象，现有理论无法统一解释。
- **连续时间近似的粗粒度权衡未厘清**：后续出现的 rod flow、edge flow 等模型试图保留更多中间尺度信息，但缺乏系统的比较框架与精度-简洁性权衡分析。

## 核心贡献（创新点）
1. **以摄动极限形式化推导中心流**：假设损失 $f = g + \varepsilon h$（$\varepsilon \ll 1$），证明 $\v
