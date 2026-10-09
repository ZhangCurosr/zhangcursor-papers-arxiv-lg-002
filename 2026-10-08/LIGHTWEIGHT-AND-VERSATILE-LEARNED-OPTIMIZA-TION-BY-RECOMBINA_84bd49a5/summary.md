---
title: "LIGHTWEIGHT-AND-VERSATILE-LEARNED-OPTIMIZA-TION-BY-RECOMBINA"
source: https://arxiv.org/pdf/2610.09604v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 10:46:48"
---

# 论文速读：LIGHTWEIGHT-AND-VERSATILE-LEARNED-OPTIMIZA-TION-BY-RECOMBINA

## 一句话总结
本文提出一种仅 37k 参数的轻量级 Learned Optimizer，通过将历史梯度按多分辨率时间跨度均值聚合，并用小型 Transformer 动态预测各历史槽的共享标量权重，在语言、视觉与图任务上实现零样本泛化，验证损失与测试精度均显著优于默认超参的 Adam。

## 研究问题与动机
- 现有 Learned Optimizer（如 VeLO）对每个参数坐标独立预测更新特征，训练开销极高（约 4,000 TPU-months），难以轻量化与跨架构部署。
- 传统一阶优化器（Momentum、Adam）已验证历史梯度可作为有效搜索方向，但缺乏自适应动态重组不同时期历史贡献的能力。
- 直接缓存完整梯度历史会导致线性增长的显存开销，需一种以有限槽位覆盖长轨迹且保持各时段贡献独立可访问的存储机制。
- 元学习优化器普遍依赖大规模算力堆叠才能实现零样本泛化，缺乏在极小预算下兼顾性能与实用性的设计范式。

## 核心贡献（创新点）
1. 提出组级共享标量系数的历史梯度动态重组范式；与 VeLO 等坐标级独立预测方法的本质区别在于将参数量压缩至 37k 并消除对目标模型维度的依赖。
2. 设计多分辨率渐进平均历史存储结构；与直接缓存全量梯度或固定动量窗口的本质区别在于以 O(log H) 显存覆盖任意长度轨迹，同时保持各时段均值独立可访问。
3. 构建基于固定维度特征与 Self-Attention 的零样本泛化策略；与 prior RNN/Transformer 策略的本质区别在于输入抽象为统计摘要而非原始梯度序列，彻底摆脱对特定网络结构的过拟合。
4. 证明 0.87 GPU-hours 即可训练出实用级策略；与“Learned Optimizer 泛化必须依赖海量算力”行业共识的本质区别在于以极简元训练成本实现跨语言/视觉/图的即插即用部署。

## 方法详解
- **参数分组与更新公式**：目标模型参数划分为不相交组（每组至多 256 坐标），组内共享同一时序系数向量。更新为 $\Delta \theta_t^{(b)} = -\sum_{h \in \mathcal{A}_t^{(b)}} \alpha_{t,h}^{(b)} S_{t,h}^{(b)}$，其中 $S_{t,h}^{(b)}$ 为第 $h$ 槽梯度均值，$\alpha_{t,h}^{(b)}$ 为预测标量系数。
- **多分辨率历史存储**：设 $H$ 个槽位与缩放因子 $r \ge 2$，第 $c$ 层槽容量上限为 $r^{c-1}$。新梯度进入槽 1，溢出值逐级向前传递并与已有均值融合或替换，超出最大槽则丢弃，实现指数级时间跨度覆盖。
- **组特征编码器 (Group Feature Encoder)**：对每个填充槽提取 7 维直方图特征及 6 类统计量（组尺寸、存储均值尺度、当前参数尺度、坐标均匀度、梯度-参数对齐度、尺度比），经独立 MLP 投影后通过双向 Softmax Gate 融合为固定维度 $d$ 的槽特征。
- **联合时序预测器 (Joint Temporal Predictor)**：槽特征序列加入表征时间中心与跨度长度的位置编码，经两层 Self-Attention Transformer（宽度 32，4 heads，FFN 128）联合推理；输出头 Softmax 生成归一化分配 $a_{t,h}$，独立 Scale Token 预测组级增益 $s_t \in (0,2)$，最终 $\alpha_{t,h} = s_t \cdot a_{t,h}$。
- **轨迹级元训练目标**：在 $K$ 步展开模拟中计算归一化累积保留集损失变化 $\mathcal{L}_{\mathrm{meta}} = \sum_{k=1}^{K} (L_{t+k}^{\mathrm{held}} - L_t^{\mathrm{held}}) / L_t^{\mathrm{held}}$，对策略参数执行全微分更新，历史统计与起始损失均 detach。

## 实验与结果
- **元训练配置**：37,389 参数策略，使用 MLP/Synthetic、ConvNet/CIFAR-10、MicroResNet/Fashion-MNIST、MicroViT/CIFAR-10 四个小模型（均 <10k 参数）训练 64 次 meta-update，耗时 0.87 GPU-hours (NVIDIA B200)。
- **语言模型**：BERT-Tiny (Wiki-BookCorpus) 验证损失 3.463（Adam 3.810，↓9.1%
