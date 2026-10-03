---
title: "Ornstein-Uhlenbeck-Is-Hard-to-Beat-Yet-Superlinear-Drift-Shi"
source: https://arxiv.org/pdf/2609.37579v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 21:56:59"
---

# 论文速读：Ornstein-Uhlenbeck-Is-Hard-to-Beat-Yet-Superlinear-Drift-Shi

## 一句话总结
本文挑战了“OU扩散在正向收敛中难以超越”的既有理论结论，提出将坐标超线性朗之万漂移引入基于分数的图像生成框架；通过数值求解Fokker-Planck方程构造条件分数并训练U-Net，在$28\times28$灰度图像上验证了超线性模型在实证Wasserstein-1距离上全面优于线性OU基线，且对扩散时长$T$的波动更稳健。

## 研究问题与动机
- **理论结论的适用边界**：Brešar & Mijatović [1] 证明了在不包含超线性漂移的假设下OU扩散“难以超越”，但其结论仅限至多线性漂移，未涵盖更强均值回复的扩散核。
- **解析分数的局限性**：传统扩散模型依赖仿射高斯扰动以获得闭合条件分数，但此类过程无法灵活刻画非高斯、强恢复力的数据流形结构。
- **生成质量评估的错位**：正向收敛速率的最优性不必然等价于反向生成分布的运输成本最优；需在实际生成任务中检验不同漂移设定的实证表现。
- **数值实现的工程需求**：超线性漂移缺乏解析转移密度，亟
