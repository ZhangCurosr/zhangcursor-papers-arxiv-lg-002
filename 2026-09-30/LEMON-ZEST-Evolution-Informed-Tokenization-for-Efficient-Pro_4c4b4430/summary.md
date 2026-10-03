---
title: "LEMON-ZEST-Evolution-Informed-Tokenization-for-Efficient-Pro"
source: https://arxiv.org/pdf/2609.37675v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:54:44"
---

# 论文速读：LEMON-ZEST-Evolution-Informed-Tokenization-for-Efficient-Pro

## 一句话总结
本文提出基于多序列比对进化保守区的演化知情词表 **ZEST**，并以此构建仅 200M 参数的轻量蛋白语言模型 **LEMON**；该方法将结构先验直接固化为分词单元，在单张 H100 训练约一周的条件下，于远程同源蛋白检测任务上全面超越 600M–3B 参数的主流 SOTA 模型。

## 研究问题与动机
1. **Scaling 路线的边际递减**：现有蛋白语言模型（PLM）主要依赖扩大参数与数据规模提升性能，但词表粒度与领域先验注入机制被严重低估。
2. **序列-折叠的许多一映射未被分词捕获**：蛋白质在剧烈序列变异下仍能保持折叠结构，标准字符级或统计级 BPE 分词无法显式编码这种进化保守的模块化合规性。
3. **结构依赖模型的落地瓶颈**：SaProt、ProstT5 等方法需在推理时输入 3Di 结构 token，对未知结构或高通量筛查场景不友好。
4. **远程同源检测仍是硬骨头**：在序列同一性 <30% 的极端区间，即使大规模 PLM 与 HMM 基线仍难以稳定区分共享同折叠的远缘蛋白。

## 核心贡献（创新点）
1. **ZEST 演化知情词表**：从 TEDLH 的 HMM 共识序列中自动挖掘 ≥3 残基的非均聚保守区构建词汇；与 BPE 等纯统计分词的本质区别在于，其词元天然对应进化筛选出的结构功能模块。
2. **Trie-Dropout 正则化**：基于词汇前缀树在训练中随机截断长 token 为子区域；与 NLP 的 BPE-dropout 不同，其分解严格遵循生物层次嵌套，有效防止长 token 垄断导致词表稀疏。
3. **LEMON 双头紧凑架构**：共享 Transformer 编码器同时供给对比学习表示头与残基级重建头，仅
