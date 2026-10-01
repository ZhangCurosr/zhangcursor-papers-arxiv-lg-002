---
title: "The-BatchNorm-Illusion-Diagnosing-Normalization-Artifacts-in"
source: https://arxiv.org/pdf/2609.08901v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:05:12"
field: "机器学习遗忘评估"
keywords: ["machine unlearning", "batch normalization", "evaluation artifact", "linear probing", "neural collapse", "privacy verification", "model calibration"]
innovations: ["提出保权BN重校准不动点算子并证明其唯一分解性质", "识别并形式化BN腐败与BN失配两种机制相反的评估伪影", "揭示仅10张无标签图像即可恢复被掩蔽遗忘准确率的对抗性威胁"]
benchmarks: ["CIFAR-10", "CIFAR-100", "Tiny-ImageNet", "ImageNet-100"]
---

# 论文速读：The BatchNorm Illusion: Diagnosing Normalization Artifacts in Machine Unlearning Evaluation

## 一句话总结
本文揭示了机器学习遗忘评估中一个被忽视的BatchNorm伪影：仅需一次对保留数据的无梯度前向传播（不改权重），即可使九个遗忘方法中六个的表面遗忘指标逆转高达78pp；作者将BN运行统计量定义为保权不动点算子，并提出独特的线性探针分解，区分归因于统计量的测量偏差与归因于编码器几何结构的残余泄露。

## 研究问题与动机
- **评估伪影缺失**：现有机器遗忘基准（NeurIPS竞赛、Deep Unlearn、MUBox）在BN架构上报告的表面遗忘指标未考虑BN运行统计量随训练动态漂移的影响，缺乏校准控制。
- **两类失败模式混淆**：遗忘失败既可能源于编码器层面的特征残留（encoder-failure，如线性探针和重学攻击所证实），也可能源于评估过程的测量误差（measurement-failure），二者本质不同但常被混为一谈。
- **两种机制相反的伪影**：（1）BN腐败（train模式下forget数据更新EMA导致统计量漂移）；（2）BN失配（eval模式下冻结统计量与漂移后的卷积权重不匹配），表面均表现为低遗忘准确率。
- **安全风险**：仅需10张无标签图像即可恢复大部分被掩蔽的遗忘准确率，说明该伪影可被攻击者直接利用。

## 核心贡献（创新点）
1. **发现并形式化两种BN伪影**：首次系统识别BN corruption与BN misalignment，证明前者具有梯度投影不可拦截性（Theorem 2），后者源于激活分布与冻结统计量的解耦。
2. **提出BN重校准算子 $\mathcal{R}^*$**：证明其为保权、幂等的不动点算子（Theorem 1），任何预/后差异严格归因于BN状态而非权重修改（Corollary 1）。
3. **导出线性探针的唯一分解**（Proposition 1）：将LP提升拆分为BN测量偏置项与Neural Collapse残余项，揭示若干方法的表面LP低估了编码器泄露（负BN残差达20–30pp）。
4. **实证揭示大规模评估失效**：在CIFAR-10/100上对九种方法评估，六种方法的$\Delta F$ ≥ 18.9pp（最高+78pp）；重校准后遗忘准确率与重学AUC相关性升至$r=0.998$。
5. **转化为安全发现**：攻击者只需10张无标签图像即可完成恢复；GroupNorm控制使$\Delta F$精确归零，验证机制特异性。

## 方法详解
- **模型表示**：BN网络表示为$M = (\theta, \varphi)$，其中$\theta$为所有权重参数，$\varphi = \{(\mu^\ell, (\sigma^\ell)^2)\}_{\ell=1}^L$为每层BN的运行均值与方差。
- **重校准算子定义**：$\mathcal{R}^*_{\mathcal{D}_r}: (\theta, \varphi) \mapsto (\theta, \varphi^*(\theta;\mathcal{D}_r))$，即在当前权重$\theta$下用保留分布$\mathcal{D}_r$oracle层归一化重新计算运行统计量。
- **实现形式**：使用PyTorch `torch.optim.swa_utils.update_bn`，在$N_r$张保留图像上单次train-mode前向传播（禁用梯度），mini-batch size $b=128$。
- **理论性质**（Theorem 1）：
  - 幂等性：$\mathcal{R}^* \circ \mathcal{R}^* = \mathcal{R}^*$
  - 权不变性：$\theta$保持不变
  - Oracle等价性：$f(x; \mathcal{R}^*(M))$等于在固定权重下使用oracle population moments的网络输出
  - 有限样本收敛：误差为$O_p(N_r^{-1/2}) + O_L(b^{-1})$
- **线性探针分解**（Proposition 1）：
  $$\mathrm{LP}^M - \mathrm{LP}^{M_{\mathrm{retr}}} = \underbrace{(\mathrm{LP}^{M_{\mathrm{cal}}} - \mathrm{LP}^{M_{\mathrm{retr}}})}_{\text{NC-residual}} + \underbrace{(\mathrm{LP}^M - \mathrm{LP}^{M_{\mathrm{cal}}})}_{\text{BN-residual}}$$
  BN残差为零当且仅当$\varphi = \varphi^*(\theta;\mathcal{D}_r)$；NC残差在$\mathcal{R}^*$下不变。该分解满足唯一性条件。
- **BN腐败的非拦截性**（Theorem 2）：若BN在train模式， forget数据前向传播通过EMA更新运行统计量$\mu^\ell \gets (1-m)\mu^\ell + m\hat{\mu}_{\mathrm{batch}}^\ell$，该更新完全在梯度计算图之外，任何作用于权重的梯度投影算子均无法拦截。

## 实验与结果
- **数据集与架构**：CIFAR-10（单类/双类遗忘）、CIFAR-100（单类/五类遗忘），ResNet-18 backbone（CIFAR适配版：3×3 stem conv，无max-pool）。额外验证Tiny-ImageNet/ResNet-50、ImageNet-100/ResNet-50、ViT-S/16（LayerNorm控制）。
- **评估基线**：Retrain、SCRUB、GA、GA+FT、SalUn、BadTeacher、IncompTeacher、NoiseInject、SSD、PGU（共9种近似遗忘方法+Retrain参考）。
- **主要结果**：
  - 六种方法出现显著$\Delta F$：BadTeacher +78.2pp、SalUn +62.1pp、SSD +62.4pp、PGU† +60.9pp、IncompTeacher +48.0pp、NoiseInject +18.9pp。
  - GA在500步时$\Delta F=0$，但其Pre-R仅65.2%（已破坏保留性能），非真正遗忘。
  - 重校准后，遗忘准确率与重学AUC的Pearson相关系数升至$r=0.998$（$p<10^{-6}$），Spearman $\rho=0.988$。
  - 线性探针分解（Table 2）：CIFAR-100上GA、SalUn、SSD的BN残差为-20至-30pp（严重低估编码器泄露），NC残差+14至+17pp（超越Retrain参考）。
- **攻击实验**：仅10张无标签图像，BadTeacher恢复至88.7%遗忘准确率，NoiseInject恢复至91.0%；OOD场景（CIFAR-100像素+CIFAR-10归一化）仍可达ID结果的78%–95%。
- **机制证伪**：
  - GroupNorm控制：所有方法$\Delta F$精确为0.00pp（Table 6）。
  - LayerNorm控制（ViT-S/16）：$\Delta F=0.00\mathrm{pp}$。
  - ImageNet-100：BadTeacher $\Delta F=+92.0\mathrm{pp}$（Table 15）。

## 相关工作脉络
- **近似机器遗忘方法**（Cao & Yang [3]形式化问题）：与SCRUB [17]、SalUn [5]、GA [12]、BadTeacher [4]等近似方法对比，本文指出这些工作在BN架构上的评估缺乏标准化重校准控制。
- **编码器层面评估**：Gao et al. [7]使用线性探针发现Neural Collapse残余，Lee et al. [19]证明骨干冻结+分类器重训可恢复遗忘准确率；本文表明这些工作未剥离BN测量偏置，其报告的NC信号可能被低估或高估。
- **BatchNorm重校准应用**：SWA [15]使用`update_bn`校正平均模型统计量；测试时自适应 [25, 22] 类似地重对齐BN；本文贡献在于将其引入遗忘评估并证明其诊断价值。
- **大型基准**：NeurIPS遗忘竞赛 [30]、Deep Unlearn [2]、MUBox [20]均未显式报告BN重校准控制，本文诊断可直接应用于UMA [32]等验证方法。
- **成员推断攻击**：Nasr et al. [23]的梯度范数攻击等在本校检后AUC变化≤0.012，与遗忘准确率高达98pp的变化形成对比，说明该伪影对特定度量敏感。

## 局限性与未来方向
- 实验主要集中在图像分类基准（CIFAR-10/100、Tiny-ImageNet、ImageNet-100），instance-wise遗忘仅在一个条件下评估（Appendix P），未见大规模多类别实例级遗忘结果。
- 未测试语言模型；虽然LayerNorm控制（ViT实验）支持机制特异性，但文本场景中BN的类比物（LayerNorm）下诊断算子恒等，需独立验证。
- Full ImageNet-1k评估受限于计算资源未实施；ImageNet-100结果显示伪影可扩展至更大数据集，但完整规模未知。
- 防御建议依赖架构变更（GN/LN替换或冻结BN），对于必须使用BN的已有部署缺乏实用缓解策略。
- 未来方向：将诊断纳入标准评估协议（报告$\Delta F$、Pre-R、Post-R三元组）；探索retain mixing比例对train-mode腐败的缓解作用（Appendix Q显示25% retain可消除SalUn的BN gap）；结合MIA/UMA等验证器形成综合评估框架。

## 研究启发与可借鉴点
- **诊断即控制**：单行代码（`torch.optim.swa_utils.update_bn`）即可实现保权重校准，成本极低但能系统性纠正评估偏差，可作为其他基于BN架构的工作的标准诊断步骤。
- **唯一分解框架**：Proposition 1提供的代数恒等式+唯一性证明，为任何BN架构上的线性探针评估提供了可复用的归因范式，适用于公平比较不同方法。
- **攻击面量化**：将评估伪影转化为安全发现（10张图像可恢复大部分遗忘能力），展示了“诊断→威胁建模”的转化路径，提示遗忘系统的部署需考虑对抗性重校准风险。
- **多区域分类法**：(BN-residual, NC-residual)四象限分类（§4.2）为评估方法提供了清晰的表征分类工具，可迁移至其他需要区分测量误差与真实性能退化的场景。
- **消融设计**：通过GroupNorm、LayerNorm、ViT控制严格隔离机制，证明了伪影的BN特异性而非架构或数据集混淆，此类对照实验设计值得在后续工作中复现。

## 关键术语表
- **BatchNorm Illusion（BN幻觉）**：指BN运行统计量漂移导致的评估指标系统性偏差，使遗忘方法看似成功实则仍编码遗忘信息。
- **Recalibration Operator（重校准算子 $\mathcal{R}^*$）**：对模型$(\theta, \varphi)$执行单次无梯度前向传播，用保留数据重新估计BN运行统计量，保持权重不变的保权不动点算子。
- **BN Corruption（BN腐败）**：train模式下遗忘前向传播通过EMA更新运行统计量，使统计量吸收forget分布信息，与固定权重不一致。
- **BN Misalignment（BN失配）**：eval模式下冻结统计量，但卷积权重在遗忘更新后漂移，导致激活分布与统计量解耦。
- **BN-Residual（BN残差）**：线性探针分解中$\mathrm{LP}^M - \mathrm{LP}^{M_{\mathrm{cal}}}$部分，衡量由BN运行统计量引起的测量偏置（负值表示低估泄露）。
- **NC-Residual（NC残差）**：分解中$\mathrm{LP}^{M_{\mathrm{cal}}} - \mathrm{LP}^{M_{\mathrm{retr}}}$部分，归因于编码器几何结构的残余forget类线性可分性。
- **Relearning AUC（重学AUC）**：对遗忘模型在50张forget样本上fine-tune 50 epoch，积分归一化遗忘准确率曲线所得面积，衡量编码器泄露程度。
- **Non-interceptability（不可拦截性）**：Theorem 2表明BN EMA更新发生在前向传播中，任何仅作用于梯度空间的投影算子均无法阻止该更新。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100公开可用；Tiny-ImageNet、ImageNet-100公开可用；ViT-S/16使用ImageNet预训练权重。
- **代码/权重**：论文未明确开源代码仓库，但依赖PyTorch标准函数`torch.optim.swa_utils.update_bn`；各方法使用官方实现或最接近的重实现（Appendix F提供超参）。
- **关键超参**：
  - 重校准：$N_r \geq 5000$保留图像，mini-batch $b=128$，train-mode无梯度前向传播。
  - 线性探针：`sklearn.linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)`，每类50样本。
  - 重学：50 forget样本，50 epochs，lr=$10^{-4}$。
- **硬件**：单张NVIDIA A6000 GPU（≈30GB VRAM），总耗时约90 GPU-hours。
