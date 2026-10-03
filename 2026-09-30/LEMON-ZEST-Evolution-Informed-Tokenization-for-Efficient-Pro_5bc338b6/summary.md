---
title: "LEMON-ZEST-Evolution-Informed-Tokenization-for-Efficient-Pro"
source: https://arxiv.org/pdf/2609.37675v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 08:54:50"
---

# 论文速读：LEMON-ZEST-Evolution-Informed-Tokenization-for-Efficient-Pro

## 一句话总结
本文提出进化先验驱动的ZEST分词器与仅200M参数的LEMON模型，将多重序列对齐中的保守结构模块直接编码为复合Token，以单卡H100训练一周的低成本，在远缘同源检测任务上全面超越600M-3B参数的现有SOTA模型。

## 研究问题与动机
- **分词粒度被忽视**：当前PLM的发展高度依赖参数与数据的Scaling Law，却极少探讨分词策略对蛋白质序列压缩效率与表征质量的影响。
- **标准分词违背蛋白质进化规律**：蛋白质结构远比序列保守（多对一映射），字符级或纯统计BPE无法捕捉进化中保留的功能/结构域，导致模型需暴力学习本可由先验给出的折叠约束。
- **远缘同源推断仍是难点**：以<30%序列一致性为界，现有PLM在复杂结构分类层级中仍难以准确识别远缘同源关系。
- **结构依赖型方法限制泛化**：引入3Di等结构Token的方法需额外AlphaFold预测步骤，无法直接应用于无实验或无预测结构的蛋白序列。

## 核心贡献（创新点）
- **ZEST进化词表**：从TEDLH的76万+ HMM profile中提取保守区（≥3残基），聚类排序后构建约32K词表，平均压缩至~4残基/token，将生物进化先验直接固化至分词阶段。与BPE的本质区别在于词单元具有明确的折叠/结构语义，而非纯频率统计合并。
- **Trie-Dropout正则化**：基于前缀树在训练时以概率$p_{drop}$随机降级为较短匹配，打破贪心最长匹配导致的词表长尾未覆盖问题，同时为推理时的测试时增强（TTA）提供天然机制。与BPE-dropout的本质区别在于分解依据是进化层级嵌套关系而非统计合并顺序。
- **LEMON双头紧凑架构**：200M参数Y型Transformer，共享编码器同时服务于对比学习表征头（全局折叠几何）与残基重建序列头（局部序列保真），无需结构输入即在单一H100上完成全流程训练。与巨型PLM的本质区别在于用领域知识前置替代了参数/数据暴力Scaling。

## 方法详解
- **词表构建（ZEST）**：基于TEDLH库的765,248个HMM consensus序列，提取长度≥3且非同质多聚体的保守区（zones）；使用MMseqs2 linclust在70%序列一致性阈值下聚类，按跨profile频率与zone长度加权评分排序，选取Top 31,975个聚类作为复合Token，追加20个标准氨基酸残基Token与5个特殊Token，形成~32K词表。
- **Trie-Dropout机制**：将词表构建成前缀树，长Zone自然嵌套于共享前缀的更长Token中。分词时trie返回从长到短的所有合法匹配，以概率$p_{\mathrm{drop}}$覆盖贪心选择，按均匀分布
