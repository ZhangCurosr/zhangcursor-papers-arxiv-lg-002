---
title: "VisER-Visual-Evidence-and-Reliance-for-Object-Hallucination"
source: https://arxiv.org/pdf/2608.30480v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:12:47"
---

# 论文速读：VisER: Visual Evidence and Reliance for Object Hallucination Detection in LVLMs

## 一句话总结
本文针对大视觉语言模型（LVLMs）生成中的物体幻觉问题，提出免训练的 **VisER** 双视角检测指标。通过联合衡量**视觉证据（Visual Evidence）**与**视觉依赖（Visual Reliance）**，VisER 能够区分物体支持信号是来自图像特异性视觉证据，还是场景先验/自回归文本前缀的虚假延续，在多个主流 LVLM 与基准上显著提升了幻觉检测性能。

## 研究问题与动机
- **来源混淆（Source Confounding）**：现有免训练检测器依赖内部信号（token 概率、注意力、视觉置信度、图文相似度），但高支持度未必源于真实视觉 grounding，可能来自场景合理性、物体共现先验或已生成文本前缀的自回归延续。
- **难例失效**：缺失物体若与场景高度兼容或位于自然语言续写位置，现有信号仍会给出高置信度，导致 detector 只衡量“支持强度”却忽略“支持来源”。
- **单视角盲区**：仅依赖图像侧信号易被背景/关联 cue 误导，仅依赖文本侧对比又无法验证物体是否具备 object-specific 视觉证据。
- **部署需求**：开放场景缺乏外部标注与参考 caption，需一种无需训练、不引入额外验证生成、可直接后处理（post-hoc）的轻量化检测方案。

## 核心贡献（创新点）
- **形式化“来源混淆”分析框架**，指出内部信号高支持 ≠ 视觉 grounding，揭示现有方法在场景先验与文本前缀驱动幻觉下的共同失效机制。
- **提出 VisER 免训练双视角指标**：VE 通过 logit-lens 聚合图像 token 对物体的特异性证据并用软门控调制上下文兼容性；VR 对比图像派生支持与文本前缀支持，量化 grounding 的视觉偏向性。
- **构建贝叶斯分解理论视角**，将物体生成概率分解为前缀侧支持与图像侧兼容性增益，为双信号设计提供理论依据，且 VR 可视为平滑的图像-vs-前缀对数比值代理。
- **在 8 个主流 LVLM 与 MSCOCO / Pascal VOC 上系统验证**，VisER 在 AUROC 与 AUPR 上持续超越强基线，并达到甚至超越 POPE 提示验证方法的性能，且零额外解码开销。

## 方法详解
- **问题设定**：给定图像 $I$、指令 $x$ 与自回归响应 $y = (y_1, \dots, y_T)$，对每个生成的物体提及 $o$（对应 token 位置 $t_o$）计算 grounding 分数 $s(o, I, x, y_{<t_o})$，分数越高表示视觉 grounding 越强，可通过阈值或排序指标（AUROC/AUPR）二分类。
- **Visual Evidence (VE)**：
  - 利用 **logit lens** 将图像 token 隐藏状态 $h_{v_i}^{\ell_I}$ 投影至词表空间，计算每个图像 token 对物体 $o$ 的预测概率 $p_i(o) = \text{softmax}(h_{v_i}^{\ell_I} W_U)_o$。
  - 累计视觉证据质量 $M_{\text{vis}}(o) = \sum_{i=1}^N p_i(o)$，构造证据门控 $g(o) = \sigma\!\left(\frac{M_{\text{vis}}(o)}{\tau + \epsilon}\right)$，其中 $\tau$ 在独立校准集上通过平均 $M_{\text{vis}}$ 估计。
  - VE 分数为上下文兼容性 $\text{sim}(h_{\text{ctx}}^{\ell_T}, h_o^{\ell_T})$ 与门控的乘积：$VE(o) = \text{sim}(\cdot,\cdot) \cdot g(o)$。弱物体特异性证据会直接抑制兼容性贡献。
- **Visual Reliance (VR)**：
  - 图像派生支持 $S_{\text{img}}(o) = \sum_{i=1}^N \hat{p}_i(o) \, \text{sim}^+(h_{v_i}^{\ell_I}, h_o^{\ell_T})$，其中 $\hat{p}_i(o)$ 为归一化 logit-lens 权重，使强证据 token 贡献更大。
  - 文本前缀支持 $S_{\text{text}}(o)$：取物体 token 输入嵌入与前缀 token 最相似的 $K$ 个（$K_o = \min(K, t_o-1)$），计算平均相似度 $\frac{1}{K_o}\sum \text{sim}^+(e_j, e_o)$。
  - VR 定义为 $VR(o) = \frac{S_{\text{img}}(o)}{S_{\text{img}}(o) + S_{\text{text}}(o) + \epsilon}$，值越大表示支持更源自图像而非前缀。
- **最终 VisER 分数**：$s_{\text{VisER}}(o) = \alpha \, VE(o) + (1-\alpha) \, VR(o)$，$\alpha \in [0,1]$ 平衡两路信号（默认 $\alpha=0.4$）。两者正交互补：VE 过滤“有兼容性但无物特异性证据”的假阳性，VR 过滤“前缀驱动但图像证据不足”的假阳性。

## 实验与结果
- **数据集与协议**：MSCOCO（主实验）、Pascal VOC，均随机采样 500 张验证集图像；按 CHAIR 协议提取生成物体提及并与图像级标注匹配，划分为 grounded / hallucinated。
- **评估模型**：LLaVA-1.5-7B/13B、LLaVA-NeXT-7B、InstructBLIP、MiniGPT-4、InternVL3、Shikra-7B、Qwen2.5-VL；greedy decoding，最大输出 512 tokens。
- **基线**：Entropy、NLL、Internal Confidence、SVAR、Contextual Lens、PAS、GLSim，以及 POPE 提示验证方法。
- **主要结果**：
  - **MSCOCO**：VisER 平均 AUROC **84.89**，较最强基线 GLSim（81.26）提升 **+3.63**；平均 AUPR **95.82**。在 LLaVA-NeXT 上 AUROC 达 81.44，较基线提升 3.44。
  - **Pascal VOC**：平均 AUROC **83.89**，AUPR **93.88**，四项模型均夺冠。
  - **扩展模型**（Table 2
