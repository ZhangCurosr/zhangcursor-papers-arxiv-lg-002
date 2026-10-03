---
title: "Predictive-Geometry-of-Hidden-Trajectories-in-Transformers"
source: https://arxiv.org/pdf/2609.37717v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 18:35:16"
field: "大语言模型压缩与可解释性"
keywords: ["pullback Fisher", "hidden-state geometry", "token pruning", "low-rank compression", "knowledge distillation", "predictive saliency", "transformer interpretability"]
innovations: ["从终端next-token loss推导中间层pullback Fisher几何并证明局部Hessian分解", "提出因果支撑的tokenwise Fisher-Jacobian曲率评分", "将预测几何作为低秩分配与蒸馏正则化的损失感知信号"]
benchmarks: ["WikiText", "OpenWebText", "FineWeb"]
---

# 论文速读：Predictive-Geometry-of-Hidden-Trajectories-in-Transformers

## 一句话总结
论文从终端 next-token prediction 损失出发，证明 decoder-only transformer 中间层 hidden state 的局部二阶几何由 **pullback Fisher 算子** 支配，该算子可将输出敏感方向与 prediction-null 方向分离，并由此导出 token 级曲率评分 $\kappa_{\ell,s}$；实证表明该几何量可用于低秩分配、token 剪枝与蒸馏正则化。

## 研究问题与动机
- **核心问题**：终端 loss 仅作用于最后一层输出，但它如何通过固定下游计算约束中间每一层的 hidden state 几何？哪些 residual-stream 方向是输出敏感的，哪些近似不可见？
- **现有方法不足**：activation-space 压缩（SliceGPT、FLAT-LLM）与 token pruning（Zero-TPrune 等）主要基于经验观察（如激活范数、注意力权重），并未从终端预测目标本身推导关键方向与重要性。
- **表征几何缺乏目标驱动**：Valeriani et al. [24] 揭示隐表示存在低本征维度的经验现象，但未建立其与 next-token loss 的严格联系。
- **实用需求**：Transformer 存在显著冗余，若能识别并保留输出敏感子空间，可在压缩、蒸馏与高效推理中实现损失感知的方向选择。

## 核心贡献（创新点）
1. **Loss-induced hidden geometry**：定义 layerwise loss-to-go 函数，证明在低损失附近其主导 Hessian 项为 pullback Fisher 算子 $K_\ell = \bar{J}_\ell^\top F^* \bar{J}_\ell$，残余项受 $\sqrt{\varepsilon_*}$ 控制。*与已有工作的本质区别*：此前表征几何研究仅描述现象，本文从终端目标出发严格推导中间层的局部曲率结构。
2. **可观察与 prediction-null 方向分解**：$K_\ell$ 的谱分离出输出敏感子空间与其正交的 null 空间（$\ker \bar{J}_\ell$）。*本质区别*：提供可检验的局部子空间分解，而非经验性的低秩假设。
3. **Tokenwise predictive saliency $\kappa_{\ell,s}$**：基于 Fisher-Jacobian 恒等式导出逐 token 曲率得分，仅在 target 的 causal ancestor 集上非零，由下游 Jacobian 耦合控制。*本质区别*：提供 loss-aware 的 token 重要性度量，区别于仅依赖 attention 幅值的启发式方法。
4. **几何引导的压缩与蒸馏**：将 $K_\ell$ 谱用于非均匀 rank 分配、token 剪枝，并作为隐藏状态恢复的正则化项。*本质区别*：将几何从诊断工具转化为可操作的压缩/蒸馏信号。

## 方法详解
- **Layerwise loss-to-go 函数**：固定已训练模型，定义 $\mathcal{I}_\ell(X) = \phi(\Psi_\ell(X))$，其中 $\Psi_\ell$ 为从层 $\ell$ 到目标位置 logit 的下游映射，$\phi(z) = D_{\mathrm{KL}}(q \| \mathrm{softmax}(z))$。
- **局部 Hessian 分解（Theorem 1）**：$\nabla^2 \mathcal{I}_\ell(X_\ell^*) = \bar{J}_\ell^\top F^* \bar{J}_\ell + R_\ell$，其中 $F^* = \mathrm{diag}(p^*) - p^* p^{*\top}$ 为 softmax Fisher 矩阵，$\|R_\ell\|_{\mathrm{op}} \leq M_\ell \sqrt{2\varepsilon_*}$。
- **Tokenwise curvature score（Proposition 1）**：$\kappa_{\ell,s} = \mathrm{tr}(P_s K_\ell P_s^\top) = \|F^{*1/2} H_{\ell,s}\|_F^2$，$H_{\ell,s} = D_{X_{\ell,s}} \bar{\Psi}_\ell$ 为 token 到 logit 的 Jacobian；因果掩码下，非 ancestor token 的 $\kappa_{\ell,s} = 0$。
- **矩阵自由估计**：不显式构造 $K_\ell$（维度为 $t \times d \times t \times d$），通过 JVP-Fisher-VJP 序列计算 $K_\ell v$；利用 Hutchinson 探针估计 $\mathrm{tr}(K_\ell)$ 与 $\mathrm{tr}(K_\ell^2)$ 以得到 participation-ratio 有效秩。
- **几何引导压缩**：投影到 $K_\ell$ 的前 $k$ 大特征向量张成的子空间，丢弃分量的局部二次损失上界为 $\frac{1}{2}\lambda_{k+1}\|\delta X_{\mathrm{tail}}\|^2$。

## 实验与结果
- **数据集**：WikiText、OpenWebText、FineWeb（使用 held-out 验证序列）。
- **模型**：10 个开源 decoder-only 模型，参数范围 1B–9B（SmolLM2-1.7B、LLaMA-3.2-1B、Qwen2.5-3B、Phi-3 Mini、Mistral-7B、LLaMA-2-7B、Gemma-2B/7B/2-9B、OLMo-3-7B）。
- **Exp. 0 直接验证**：teacher-KL 二次预测在中等扰动半径下呈现强预测性，验证 $K_\ell$ 作为局部曲率算子的有效性。
- **Exp. 1 各向异性**：observable 方向扰动引发的 KL 变化比 null 方向高数个数量级；有效可观察秩随层深增加但远低于隐层维度 $d$。
- **Exp. 2 非均匀 rank 分配**：以谱尾质量阈值 $\tau$ 分配各层 rank；在 $\tau = 0.05, 0.1$ 时，Gemma 2B、Qwen2.5 3B、SmolLM2 1.7B、OLMo-3 7B 等模型获得显著 $\Delta\mathrm{PPL}_{\mathrm{adv}}$ 优势。
- **Exp. 3 Token 剪枝**：$\kappa_{\ell,s}$-based 剪枝与 attention-score 剪枝性能相当，在多处设置下略优，显著优于随机基线。
- **Exp. 4 蒸馏正则化**：几何项加于不同 KD 目标，对 SkewKL 平均提升 $\Delta_{\mathrm{Geo}} = 59.3$（10/14 配对胜出），对 RKL 平均提升 $21.3$（9/14 胜出）；对 Forward KL 平均为负（$-38.4$）。

## 相关工作脉络
- **Valeriani et al. [24]**：揭示 transformer 隐表示的本征维呈跨层收缩扩张轮廓。*本文定位*：从终端 loss 出发严格推导该结构，提供损失感知的几何解释。
- **SliceGPT [3] / FLAT-LLM [22]**：基于 SVD/激活统计的通道/行删除压缩。*本文定位*：不直接删除参数，而是从预测几何指导 rank 分配与方向保留。
- **Zero-TPrune [26] / FocusCore [29]**：基于注意力权重的 token pruning。*本文定位*：$\kappa_{\ell,s}$ 提供 loss-aware 替代，理论上仅支持于 causal ancestor。
- **LoRA/LoRA+ [11, 13]**：低秩自适应微调。*本文定位*：提供数据驱动的层间非均匀 rank 分配依据。
- **DistiLLM [16] / MiniLLM [10]**：LLM 蒸馏方法。*本文定位*：几何正则项与反向/偏斜 KL 等强 autoregressive KD 目标互补。
- **Amari [2] / Karakida & Osawa [15]**：Fisher 信息与自然梯度。*本文定位*：将 Fisher 几何从训练参数空间推广至中间 hidden state 空间。

## 局限性与未来方向
- **局部性**：理论仅在参考轨迹邻域成立，大扰动下 Taylor 近似失效；未来需建立跨大区域的几何拼合理论。
- **计算开销**：矩阵自由估计虽避免显式构造 $K_\ell$，但仍显著高于注意力/激活范数等简单统计。
- **架构范围**：仅覆盖 decoder-only pre-norm transformer；encoder-decoder、MoE、RAG、显式记忆架构尚未验证。
- **压缩信号性质**：当前作为静态诊断信号使用；与第一阶梯度启发式的系统融合未充分探索。
- **蒸馏交互**：几何正则在更长训练预算、更大 student、instruction-tuned/chat 模型上的表现待研究。

## 研究启发与可借鉴点
- **从终端目标推导中间层几何**：将最终预测目标通过固定下游映射"拉回"至中间层，为隐表示分析提供严格的损失驱动视角，可迁移至其他架构的表征诊断。
- **矩阵自由 JVP/VJP + Hutchinson 估计**：避免显式构造高维 Hessian，仅需前向/反向自动微分即可获取谱信息，适合大模型的局部几何分析。
- **Fisher-weighted 正则化设计**：将 teacher-student KL 在 logit 空间展开为路径积分 Fisher 能量，其局部 Hidden-state 形式为 $\frac{1}{2}\delta h^\top K_\ell \delta h$，可作为通用蒸馏正则器。
- **Causal support 约束**：$\kappa_{\ell,s}$ 的理论零支撑集由因果祖先决定，可指导稀疏化结构与理论一致性兼得的方法设计。
- **谱尾阈值分配策略**：以 $\sum_{i>k}\lambda_i \leq \tau \sum_i \lambda_i$ 作为数据依赖的 rank 分配规则，可推广至其他需要逐层容量规划的压缩场景。

## 关键术语表
- **Loss-to-go function $\mathcal{I}_\ell$**：从第 $\ell$ 层 hidden state 出发经固定下游计算到达终端 loss 的函数值。
- **Pullback Fisher operator $K_\ell$**：输出分布的 Fisher 度量经下游 Jacobian 拉回到中间 hidden-state 空间所诱导的二次型算子。
- **Prediction-null direction**：使中心化目标 logit 在一阶意义上不变的 hidden-state 扰动方向，构成 $\ker \bar{J}_\ell$。
- **Tokenwise curvature score $\kappa_{\ell,s}$**：第 $\ell$ 层 token $s$ 对目标位置输出曲率的 Fisher-加权贡献，等价于 $\|F^{*1/2} H_{\ell,s}\|_F^2$。
- **Participation ratio $r_{\mathrm{eff}}$**：以 $\mathrm{tr}(K)^2 / \mathrm{tr}(K^2)$ 衡量的有效可观察秩，表征预测相关方向的稀疏度。
- **Matrix-free JVP/VJP**：不显式构造 Jacobian 矩阵，仅通过向量-Jacobian 与 Jacobian-向量乘积计算 $K_\ell v$ 的技术。
- **Teacher-KL**：以模型自身无扰动预测 $p^*$ 为目标的 KL 散度，使参考点成为 stationary point，从而隔离二阶曲率项。
- **Causal ancestor set**：在因果 self-attention 下能影响目标 token logit 的所有上游 token-层位置集合。

## 可复现要素
- **数据集**：WikiText、OpenWebText、FineWeb（均为公开数据集）。
- **模型 checkpoint**：论文使用 HuggingFace 公开权重（SmolLM2、LLaMA-3.2、Qwen2.5、Phi-3 Mini、Mistral-7B、LLaMA-2-7B、Gemma-2B/7B/2-9B、OLMo-3-7B）。
- **代码**：论文声明提供复现代码存档，含矩阵自由 pullback-Fisher 估计、扰动验证、rank 分配、剪枝及基线比较。
- **关键超参**：$\tau \in \{0.01, 0.05, 0.1, 0.2\}$；剪枝率 $\rho \in \{0.05, 0.10, 0.20, 0.30\}$；蒸馏 rank $r \in \{16, 32, 64\}$；温度 $\tau = 2$；$\lambda_{\mathrm{CE}} = 0.1$，$\lambda_{\mathrm{geo}} = 0.05$；几何项每 4 步更新，评估层 $\{4, 8, 12, 16, 20\}$。
- **数值精度**：前向 pass 用 bfloat16，曲率累积用 float32。
- **Hutchinson 探针数**：论文未给出具体数值，提及使用 bootstrap 置信区间。
- **计算成本示例**：LLaMA-3.2-1B 约 0.8 GPU 小时；Gemma-2-9B 约 6.0 GPU 小时（Table 6）。
