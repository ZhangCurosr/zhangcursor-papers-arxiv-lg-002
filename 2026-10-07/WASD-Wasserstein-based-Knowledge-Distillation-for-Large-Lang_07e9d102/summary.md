---
title: "WASD-Wasserstein-based-Knowledge-Distillation-for-Large-Lang"
source: https://arxiv.org/pdf/2610.07706v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 23:25:16"
field: "大语言模型压缩与蒸馏"
keywords: ["知识蒸馏", "Wasserstein距离", "Sinkhorn散度", "大语言模型", "语义对齐", "最优传输"]
innovations: ["提出基于token嵌入语义成本矩阵的WASD蒸馏方法，首次将Wasserstein散度引入LLM蒸馏", "采用Sinkhorn散度消除熵正则偏差，保证teacher分布为唯一最优解", "推导stop-gradient对偶势损失，梯度等价于Sinkhorn目标且无需反传OT求解器"]
benchmarks: ["Dolly Eval", "Self-Instruct", "Vicuna", "Super NI", "UnNI", "AlpacaEval", "Evol-Instruct", "UltraFeedback", "HumanEval", "MBPP", "GSM8k", "DialogSum", "Flores-200"]
---

# 论文速读：WASD-Wasserstein-based-Knowledge-Distillation-for-Large-Lang

## 一句话总结
论文提出基于Wasserstein距离的知识蒸馏方法WASD，通过token embedding语义成本矩阵与Sinkhorn散度，在LLM蒸馏中显式利用词汇表内的token语义关系，替代传统仅依赖概率值的散度对齐方式，在多模型族和任务上实现稳定提升。

## 研究问题与动机
- **现有散度缺乏语义感知**：KL、RKL、GJS等f散度仅比较各词元索引处的概率值差异，无法区分语义相近词元（如sofa与couch），导致蒸馏信号丢失语义结构。
- **概率稀疏引发数值不稳定**：LLM next-token分布高度稀疏，依赖密度比的散度（如KL）在接近零概率处易产生梯度不稳定。
- **熵正则Wasserstein存在偏差**：直接引入熵正则化虽可高效计算OT，但使相同分布间距离不再为零，破坏蒸馏目标的properness。
- **语义对齐对质量-多样性权衡的价值**：语义感知的损失面鼓励学生在保持输出质量的同时保留更多语义相关的候选续写，改善ROUGE-L与Self-BLEU的折中。

## 核心贡献（创新点）
1. **提出WASD方法**：首次将基于token嵌入语义成本矩阵的Wasserstein散度引入LLM蒸馏，使分布对齐显式利用词汇表几何结构。
2. **采用Sinkhorn散度消除熵偏差**：通过自相似校正项构造无偏目标，保证teacher分布为唯一最优解，解决了熵正则OT在蒸馏中的适用性问题。
3. **推导梯度等价的可微损失**：基于对偶势与包络定理，证明stop-gradient形式的WASD损失与Sinkhorn目标梯度一致，无需反传穿过OT求解器。
4. **实证跨模型族的稳定性**：在GPT-2、OpenLLaMA2、Qwen2.5、Gemma等多族模型及指令跟随、翻译、代码生成等任务上一致超越基线。
5. **揭示语义成本的消融贡献**：通过uniform/permuted/semantic成本对比，证实提升来源于token几何而非单纯平滑效应。

## 方法详解
- **成本矩阵构建**：使用teacher模型token嵌入间的余弦距离（或L2距离）构造对称代价矩阵C，并通过nearest-k截断（k=8）构建稀疏核，保留每词元k个最近非自邻域。
- **熵正则Wasserstein距离**：引入熵正则项$\epsilon\mathcal{H}(P)$，使最优传输可通过Sinkhorn-Knopp迭代高效求解，核矩阵$K=\exp(-C/\epsilon)$。
- **Sinkhorn散度定义**：
  $$D_S^\epsilon(p,q_\theta)=\tilde{D}_W^\epsilon(p,q_\theta)-\frac{1}{2}\tilde{D}_W^\epsilon(q_\theta,q_\theta)-\frac{1}{2}\tilde{D}_W^\epsilon(p,p)$$
  自相似项校正熵正则偏差，确保$D_S^\epsilon(p,q_\theta)=0\iff p=q_\theta$。
- **对偶形式与梯度推导**：利用强对偶性，熵正则OT的对偶势$\phi^*,\psi^*$通过矩阵缩放迭代求得；由包络定理得梯度仅依赖于学生对偶势$\phi^{*(p,q_\theta)}$，无需对OT求解过程反传。
- **WASD损失**：
  $$\mathcal{L}_{WASD}=\mathbb{E}\left[\sum_l\sum_i\text{sg}(\phi_i^{*(p,q_\theta)}-\phi_i^{*(q_\theta,q_\theta)})\cdot q_\theta(y_l=i|\mathbf{x},\mathbf{y}_{<l})\right]$$
  其中学生自传输势$\phi^{*(q_\theta,q_\theta)}$通过固定点迭代求解；stop-gradient使梯度计算等价于Sinkhorn目标，但避免额外网络与反传开销。
- **超参数**：$\epsilon=0.001$，Sinkhorn与固定点迭代各10步；学习率依模型不同取$10^{-4}\sim5\times10^{-5}$；学生温度缩放T=2以控制多样性。

## 实验与结果
- **数据集与基线**：指令跟随基准（Dolly Eval、Self-Instruct、Vicuna、Super NI、UnNI）；翻译(Flores-200)、摘要(DialogSum)、算术推理(GSM8k)、代码生成(WizardCoder/HumanEval/MBPP)；基线包括GKD、TAID、DistiLLM(SKL/SRKL)、ABKD、CSD、AMiD等。
- **GPT-2 XL(1.5B)→Base(0.1B)**：WASD平均ROUGE-L达24.02，超越AMiD(23.46)与CSD(22.22)；在Super NI(29.60)与UnNI(31.86)上取得最高分。
- **GPT-2 XL→Medium(0.3B)**：WASD平均25.51，优于AMiD(24.74)与CSD(24.19)。
- **OpenLLaMA2-7B→3B**：WASD平均29.61，在Super NI上达38.83，首次超越AMiD。
- **Qwen2.5-7B→1.5B**：AlpacaEval胜率90.04%、Evol-Instruct 85.32%、UltraFeedback 73.21%，均优于AMiD。
- **任务特定蒸馏(Gemma-7B→2B)**：翻译COMET 74.53、摘要ROUGE-L 35.15、算术准确率24.87；代码生成(HumanEval 74.4、MBPP 75.4、平均74.9)。
- **GPT-4反馈评分**：WASD在Dolly Eval(37.84%)、Self Inst(20.90%)、Vicuna(25.07%)均获最高。
- **提升幅度**：相对最强基线AMiD，GPT-2-0.1B平均提升约0.56 ROUGE-L，OpenLLaMA2-3B提升约0.31；质量-多样性前沿 consistently 更优。
- **计算开销**：训练时间约为基线的2倍，GPU显存增加约20%，但推理阶段无额外开销。

## 相关工作脉络
- **f散度蒸馏系列（GKD/DistiLLM/ABKD等）**：依赖词汇索引概率比值，未利用token语义；WASD转向OT几何对齐，补充语义维度。
- **对数空间对齐方法（CSD）**：通过对数概率差对齐缓解softmax平滑，但仍为逐索引比较；WASD引入跨token语义传输成本。
- **Assistant分布方法（TAID/AMiD）**：通过辅助分布缓解优化不稳定；WASD可与AMiD结合并进一步提升，说明两者正交。
- **Wasserstein KD（SinKD/WKD/KNOT）**：SinKD作用于样本级几何，WKD用于视觉分类；WASD聚焦next-token语义对齐，目标与场景不同。
- **跨tokenizer蒸馏（ULD/MultiLevelOT/MCW-KD）**：解决词表不匹配问题；WASD假设共享词表并挖掘其嵌入几何，二者互补。
- **WPR（Na et al., 2026）**：在RLHF中使用token嵌入Wasserstein正则策略；WASD将其扩展至教师-学生分布匹配，并引入Sinkhorn散度消除熵偏差。

## 局限性与未来方向
- **计算效率**：Sinkhorn迭代带来约2倍训练时间与20%显存增加，大规模蒸馏仍较昂贵。
- **成本矩阵静态性**：当前使用teacher token嵌入的静态余弦距离，未捕获任务/域特定语义结构。
- **熵正则敏感度**：$\epsilon$过大会弱化语义成本影响，需精细调参。
- **共享词表假设**：方法依赖teacher-student共用词汇表，跨词表场景需结合投影机制。
- **未来方向**：探索快速/内存高效Sinkhorn变体；设计任务自适应成本矩阵；与跨tokenizer蒸馏框架（如DSKD）集成；研究动态更新成本矩阵的可能性。

## 研究启发与可借鉴点
- **语义OT作为蒸馏信号**：将embedding几何嵌入分布对齐可作为通用范式，适用于任何共享词表的蒸馏场景。
- **Stop-gradient + 对偶势的梯度等价技巧**：通过包络定理避免反传OT求解器，兼顾语义感知与训练效率，可迁移至其他OT损失场景。
- **Sinkhorn散度校正熵偏差**：自相似减除技巧确保properness，对任何基于熵正则OT的生成对齐任务均有参考价值。
- **质量-多样性权衡分析**：通过调节解码温度绘制ROUGE-L vs Self-BLEU前沿，可系统化评估蒸馏方法的生成行为。
- **跨模型族验证策略**：本文在GPT-2/OpenLLaMA2/Qwen2.5/Gemma上统一验证，且与DSKDv2结合实现跨架构蒸馏，展示方法通用性。

## 关键术语表
- **Wasserstein距离**：衡量两个概率分布间最小"运输成本"的距离度量，需借助代价矩阵反映底层空间几何。
- **Sinkhorn散度**：对熵正则Wasserstein距离进行自相似校正后得到的无偏散度，相同分布时值为零。
- **熵正则最优传输**：在OT目标中加入熵正则项，使传输计划更平滑且可通过Sinkhorn-Knopp迭代高效求解。
- **对偶势**：熵正则OT对偶问题中的拉格朗日乘子向量，可通过矩阵缩放迭代求得，用于计算梯度权重。
- **Stop-gradient**：在反向传播中阻断梯度的操作，使对偶势被视为常量，避免对OT求解器求导。
- **Assistant分布**：由distillation框架引入的辅助分布，用于缓解优化不稳定或改进梯度信号。
- **质量-多样性权衡**：生成模型在输出内容准确性（如ROUGE-L）与表达变化性（如Self-BLEU）之间的折中关系。
- **Cross-tokenizer蒸馏**：教师与学生模型词表不同时，通过投影或通用logit对齐实现知识迁移的方法。

## 可复现要素
- **数据集**：databricks-dolly-15k、OpenWebText、Flores-200、DialogSum、GSM8k、WizardCoder、HumanEval、MBPP、AlpacaEval、Evol-Instruct、UltraFeedback；均已公开引用。
- **代码开源**：https://github.com/aailab-kaist/WASD
- **模型权重**：使用公开预训练模型（GPT-2、OpenLLaMA2、Qwen2.5、Gemma），未发布新权重。
- **关键超参**：$\epsilon=0.001$，k=8（Qwen2.5用k=2），Sinkhorn迭代10步，学习率$10^{-4}\sim5\times10^{-5}$，温度T=2，batch size=128（任务特定32）。
- **硬件**：NVIDIA RTX PRO 6000（训练）、RTX 3090（评估）。
