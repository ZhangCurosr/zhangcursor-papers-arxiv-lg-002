---
title: "Input-Blind-Controls-Produce-Substantial-Oracle-Headroom-for"
source: https://arxiv.org/pdf/2610.10368v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 02:58:25"
---

# 论文速读：Input-Blind-Controls-Produce-Substantial-Oracle-Headroom-for-Layer-Programs-in-Multiple-Choice-Evaluation

## 一句话总结
本研究通过输入盲置的固定随机扰动对照，证明在多项选择题评估中Oracle选择增益并不能直接归因于所选层计算的价值；共享选项顺序下，无输入相关编辑的对照程序仍可产生高达10.2~19.4个百分点的跨提示词收益，甚至全面超越真实层跳过/重复程序。

## 研究问题与动机
- **核心问题**：自适应计算（层跳过/重复）的Oracle评估利用已知答案为每个输入选择动作，但“选择带来的增益”与“被选中计算本身的价值”是否等价？
- **现有方法不足**：CoLa、PoLar等 работы 将Oracle增益直接解释为“固定架构的缺陷”或“特定层编辑的必要性”，缺乏将选择价值与干预价值剥离的对照实验。
- **评测偏差干扰**：多项选择题中选项顺序（option-order bias）与few-shot示例结构可能共同驱动准确率提升，现有工作未系统检验增益对跨提示词共享结构的依赖性。
- **方法学缺口**：如何设计严格的中性对照（placebo）以界定Oracle headroom的上界、识别其来源，是当前自适应推理评估的关键盲区。

## 核心贡献（创新点）
1. 提出输入盲置固定随机扰动对照（RD-fixed），在同一层段出口注入不读取当前输入状态的增量；与已有工作仅报告Oracle绝对增益不同，本文首次将选择增益与非特异性扰动增益分离，确立方法学上的对照基线。
2. 揭示“选择增益≠计算特异性收益”的识别结论；与CoLa/PoLar等将headroom直接归因于层编辑价值的工作不同，本文证明共享选项顺序下盲置控制可超越真实程序，质疑既有解释的因果有效性。
3. 设计选项旋转（Option Rotation）与letter-offset对照菜单；与既往多选题偏差研究仅描述现象不同，本文将其转化为可操作的诊断协议，量化位置偏好对跨提示词增益的贡献量级。
4. 补充生成式数学题的Transfer测试与多尺度统计推断框架；与路由乐观偏差文献止步于理论分析不同，本文通过三次独立重绘、配对Bootstrap与Holm校正提供可复现的实证检验范式。

## 方法详解
- **动作族与菜单**：每个模型构建32个单层段跳过/重复程序（段长1–4层，分布于四等深带）+ 未修改前向传播 $a_0$，构成候选家族 $F=\{a_0, a_1,\ldots,a_{32}\}$。
- **输入盲置对照（RD-fixed）**：在层段出口替换隐状态 $z_t \leftarrow z_t + \tilde{\delta}_{k,t}$，其中 $\tilde{\delta}_{k
