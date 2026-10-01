---
title: "The-BatchNorm-Illusion-Diagnosing-Normalization-Artifacts-in"
source: https://arxiv.org/pdf/2609.08901v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 03:05:16"
field: "机器遗忘评估方法论"
keywords: ["machine unlearning", "BatchNorm", "normalization artifacts", "evaluation bias", "linear probing", "encoder failure"]
innovations: ["形式化BN corruption与BN misalignment两种归一化伪影并证明前者对梯度操作不可拦截", "提出权重守恒不动点重校准算子及线性探针唯一分解理论", "实证揭示六种九种方法表观遗忘可被逆转高达78pp且攻击者仅需10图像即可恢复"]
benchmarks: ["CIFAR-10", "CIFAR-100", "Tiny-ImageNet", "ImageNet-100"]
---

# 论文速读：The BatchNorm Illusion: Diagnosing Normalization Artifacts in Machine Unlearning Evaluation

## 一句话总结
论文发现并形式化了机器遗忘评估中一个此前未记录的BatchNorm诊断性伪影：仅需一次对保留数据的前向传播（不修改任何权重），即可确定性重写模型的归一化状态，使六种九种方法中"表观遗忘"被逆转多达78个百分点；该操作被证明为权重守恒的不动点算子，其引发的任何指标差异均可严格归因于BN运行统计量而非权重变化。

## 研究问题与动机
- **核心问题**：现有机器遗忘评估依赖"遗忘类准确率下降"作为核心指标，但该方法在BatchNorm架构上存在系统性测量失败——表观低遗忘准确率可能源于BN状态漂移，而非权重真正擦除了遗忘内容。
- **现有方法不足**：
  1. 主流基准（NeurIPS unlearning competition、Deep Unlearn、MUBox）均未报告BN重校准控制，导致无法区分真实遗忘与BN伪影。
  2. 编码器层面评估（线性探针、重学习攻击）虽已揭示残差信息泄漏，但BN测量偏差会掩盖或放大这些信号，使现有结论难以忠实量化编码层级严重性。
  3. 两类现象（BN corruption与BN misalignment）机制相反但表面指标不可区分，现有文献缺乏统一的诊断框架。
  4. 评估指标与隐私保证脱节： forget accuracy可被大幅操控，而membership inference等攻击AUC几乎不受影响，说明单一指标不足以支撑GDPR式擦除义务验证。

## 核心贡献（创新点）
- **C1**：首次形式化两种机械相反的归一化伪影（BN corruption与BN misalignment），并证明前者的EMA更新对任何梯度投影操作不可拦截。与已有工作本质区别：将测量失败（measurement failure）与编码器失败（encoder failure）区分为正交的两类失效模式。
- **C2**：提出BN重校准作为近零成本诊断工具，证明其为确定性权重守恒不动点算子（Theorem 1），并导出线性探针升高的唯一分解（Proposition 1），分离BN测量偏差与编码几何分量。与已有工作本质区别：此前文献无算子理论基础支撑 attribution claim，且未量化BN偏差对Neural Collapse故事的掩盖效应。
- **C3**：在CIFAR-10/100上对九种方法的实证评估显示，诊断恢复遗忘准确率与重学习可感性之间的自洽性（Pearson r=0.998），并揭示BN脆弱性与保留能力结构性纠缠。与已有工作本质区别：此前无系统性方法比较BN诊断前后的完整指标图谱。
- **C4**：将伪影转化为安全发现——攻击者持最少10张无标签图像即可恢复大部分被掩盖的遗忘类准确率；并通过GroupNorm/LayerNorm控制完成机制性证伪。与已有工作本质区别：此前未见将评估诊断直接转化为对抗恢复实验的工作。

## 方法详解
- **两个现象的机制**：
  - **BN corruption（Phenomenon 1）**：当BN处于train模式时，每次遗忘数据前向传播执行EMA更新：$\mu^\ell \leftarrow (1-m)\mu^\ell + m\hat{\mu}_{\text{batch}}^\ell$，该更新完全在梯度计算图之外，梯度投影操作无法拦截。运行统计量向遗忘类条件分布漂移，权重保持retain一致性但BN统计吸收forget信息。
  - **BN misalignment（Phenomenon 2）**：当BN冻结(eval模式)时，运行统计量不变，但卷积权重$\theta$经投影上升后漂移，导致保留输入上的激活分布$\mathcal{A}^\ell(\theta_T; \mathcal{D}_r)$与冻结的$(\mu^\ell, (\sigma^\ell)^2)$失配，BN层错误归一化，压制遗忘准确率但不涉及真实擦除。
- **重校准算子定义**：$\mathcal{R}_{\mathcal{D}_r}^*: (\theta, \varphi) \mapsto (\theta, \varphi^*(\theta; \mathcal{D}_r))$，其中$\varphi^*$为当前权重$\theta$下在oracle分层归一化时对保留分布$\mathcal{D}_r$的人口激活矩。
- **定理1（重校准为不动点算子）**：(i) 幂等性$\mathcal{R}^* \circ \mathcal{R}^* = \mathcal{R}^*$；(ii) 权重不变性：$\theta$完全不变，第一层BN前激活与原始模型相同；(iii) 逐点oracle等价：$f(x; \mathcal{R}^*(M))$等于固定权重下使用人口矩的oracle归一化前向映射；(iv) 经验集中性：$\widehat{\mathcal{R}}_{N_r, b}$偏离$\mathcal{R}^*$为$O_p(N_r^{-1/2}) + O_L(b^{-1})$。
- **推论1（BN归因）**：任何度量$g(M)$经重校准前后的差值$g(\mathcal{R}^*(M)) - g(M)$可严格归因于BN状态而非权重，即重校准不能创造遗忘也不能消除权重编码的遗忘，仅揭示模型所处归一化状态。
- **命题1（线性探针升高的唯一分解）**：$\text{LP}^M - \text{LP}^{M_{\text{retr}}} = \underbrace{(\text{LP}^{M_{\text{cal}}} - \text{LP}^{M_{\text{retr}}})}_{\text{NC-residual}} + \underbrace{(\text{LP}^M - \text{LP}^{M_{\text{cal}}})}_{\text{BN-residual}}$，其中BN-residual在$\varphi = \varphi^*(\theta; \mathcal{D}_r)$时消失，NC-residual在$\mathcal{R}^*$下不变。该分解在满足两条件的框架内唯一。
- **推论2（BN测量偏差可识别）**：非零BN-residual为纯由运行统计量引起的测量偏差，与Neural Collapse层面的特征-分类器不对齐无关。负BN-residual表示表面LP低估编码器泄漏（如GA/SalUn/SSD在CIFAR-100上BN-residual约-20至-30pp，掩盖了约+14至+17pp的NC-residual）。
- **实现细节**：经验版$\widehat{\mathcal{R}}_{N_r, b}$使用PyTorch的`torch.optim.swa_utils.update_bn`，在retain加载器上单次前向传播，train模式，禁用梯度，mini-batch size b=128，$N_r \ge 5000$样本。

## 实验与结果
- **数据集与基线**：CIFAR-10（单类/两类遗忘）、CIFAR-100（单类/五类遗忘）、Tiny-ImageNet、ImageNet-100；ResNet-18/50 backbone；九种近似机器遗忘方法：SCRUB、GA、GA+FT、SalUn、BadTeacher、IncompTeacher、NoiseInject、SSD、PGU；对照Retrain。
- **主要结果（Table 1，CIFAR-10单类）**：
  - BadTeacher：Pre-F%=18.9 → Post-F%=97.2，$\Delta F = +78.2$pp（最大逆转）
  - SalUn：Pre-F%=3.0 → Post-F%=65.1，$\Delta F = +62.1$pp
  - SSD：Pre-F%=25.7 → Post-F%=88.1，$\Delta F = +62.4$pp
  - PGU†：Pre-F%=1.1 → Post-F%=62.0，$\Delta F = +60.9$pp
  - IncompTeacher：$\Delta F = +48.0$pp；NoiseInject：$\Delta F = +18.9$pp
  - 六种方法$\Delta F \ge 18.9$pp，三种方法$\Delta F \approx 0$（Retrain、SCRUB、GA+FT）
- **自洽性恢复**：重校准后，遗忘准确率与重学习AUC的关联从Pearson r=0.63提升至r=0.998（$p<10^{-6}$），与LP-50关联在所有三种检验下显著（$p<0.005$）。
- **2D分解（Table 2）**：GA、SalUn、SSD在CIFAR-100上位于BN-deflation象限（|BN-residual|≈20-30pp > |NC-residual|≈14-17pp），表面LP低于Retrain参考，重校准后LP反超Retrain 14-17pp。
- **对抗恢复（Section 4.4）**：攻击者持10张无标签图像时，BadTeacher恢复至88.7%遗忘准确率，NoiseInject恢复至91.0%，SalUn恢复至54.4%；OOD恢复（CIFAR-100像素+CIFAR-10归一化）达到同分布恢复的78%-95%。
- **机制证伪**：GroupNorm控制下$\Delta F$ uniformly为0.00pp；ViT-S/16（LayerNorm）下$\Delta F=0.00$pp；Tiny-ImageNet/ResNet-50上BadTeacher $\Delta F=+82.2$pp；ImageNet-100/ResNet-50上BadTeacher $\Delta F=+92.0$pp且10图像恢复达82.0%。
- **最强结果**：ImageNet-100上Full-retain重校准使BadTeacher遗忘准确率从0.0%升至92.0%，保留准确率从71.6%微升至72.5%。

## 相关工作脉络
- **Cao & Yang [3]**：机器遗忘问题形式化奠基作；本文定位为其评估方法论的补充修正，而非新方法提出。
- **Gao et al. [7] / Lee et al. [19]**：编码器层面评估（线性探针、Neural Collapse归因、骨干冻结重训练）揭示残差遗忘；本文与之互补——证明BN测量偏差会掩盖或放大这些编码器级发现，重校准是前置条件而非竞争方案。
- **Ioffe & Szegedy [14]**：BatchNorm原始工作，双角色（训练稳定性+检查点参数对象）是伪影产生根源；本文非机制创新，而是诊断应用。
- **SWA [15] / Test-time adaptation [25, 22]**：已有使用update_bn重校统计量的实践；本文定位差异在于将此操作首次系统化为遗忘评估的控制变量，并提供算子理论基础。
- **NeurIPS unlearning competition [30] / Deep Unlearn [2] / MUBox [20]**：三大规模基准；本文指出三者均未报告BN重校准控制，建议后续基准纳入该诊断。
- **UMA [32] / SCRUB [17] / SISA [1]**：验证方法与精确擦除方法；本文诊断可直接应用于基于BN骨干的验证框架，提升评估可靠性。

## 局限性与未来方向
- **数据集范围**：主要评估分类基准（CIFAR-10/100、Tiny-ImageNet、ImageNet-100），以类别级遗忘集为主；实例级遗忘仅在附录P评估（SalUn的$\Delta F$从~78pp降至+3.1pp，表明类条件设置放大corruption组件但非必需条件）。未覆盖语言模型。
- **全ImageNet-1k评估受限于计算预算**：仅在ImageNet-100上验证，大规模泛化性待进一步确认。
- **阈值依赖**：部分方法的$\Delta F=0$对应退化状态（如GA在500步后Pre-R仅剩65.2%），需结合Pre-R/Post-R/LP/ReAUC联合解读，单一指标不足。
- **未来方向**：
  1. 将重校准纳入标准遗忘评估协议，要求报告$\Delta F$与Pre-R/Post-R联合值。
  2. 探索GN/LN架构作为防御BN伪影的架构级方案。
  3. 开发基于BN状态追踪的实时遗忘监控指标。
  4. 扩展至LLM场景（LayerNorm下算子为恒等映射，但需验证文本模型的等价性）。

## 研究启发与可借鉴点
- **诊断即防御**：单次无梯度前向传播可作为低成本评估控制，几乎零开销即可揭示系统性测量偏差；该方法可迁移至任何依赖BN的模型评估场景。
- **算子理论基础的价值**：通过不动点定理与唯一分解建立attribution claim，使结论不可反驳；这种形式化方法优于经验性对比，可为其他评估问题提供模板。
- **对抗利用的启示**：将评估诊断转化为安全发现（10图像恢复实验），提示模型发布场景下BN状态本身构成信息泄漏渠道；防御需架构级改变（GN/LN）而非仅评估改进。
- **指标解读的复杂性**：$\Delta F=0$可对应三种不同机制（真实遗忘、饱和权重擦除、自重校准），需多指标联合判断；这提醒团队在多指标评估中避免单一阈值决策。
- **跨数据集/跨架构验证设计**：通过GroupNorm/LayerNorm/ViT控制实验，严格隔离归一化类型这一变量，为后续研究提供因果推断的实验设计范本。

## 关键术语表
- **BN corruption（BN腐败）**：BN处于train模式时，遗忘数据前向传播通过EMA更新运行统计量，使统计量向遗忘类分布漂移的现象；梯度投影操作无法拦截该前向更新。
- **BN misalignment（BN失配）**：BN冻结于eval模式时，卷积权重漂移导致激活分布与冻结运行统计量失配，造成测量时间扭曲的现象。
- **Re-calibration operator（重校准算子）**：$\mathcal{R}^*$，对模型$(\theta, \varphi)$执行单次无梯度前向传播，将运行统计量替换为当前权重下保留分布的人口矩，权重$\theta$完全不变。
- **BN-residual（BN残差）**：$\text{LP}^M - \text{LP}^{M_{\text{cal}}}$，表示由BN运行统计量引起的线性探针测量偏差；负值表示表面LP低估编码器泄漏。
- **NC-residual（NC残差）**：$\text{LP}^{M_{\text{cal}}} - \text{LP}^{M_{\text{retr}}}$，表示重校准后仍存在的线性探针升高，归因于编码器的特征-分类器不对齐（Neural Collapse层面）。
- **Relearning AUC（重学习AUC）**：对50个遗忘类样本微调50步后，记录遗忘准确率曲线并积分得到的归一化面积，衡量编码器残差信息的可恢复性。
- **Encoder failure（编码器失败）**：模型权重仍编码遗忘内容，即使分类准确率低，通过重学习或线性探针可恢复遗忘类信息。
- **Measurement failure（测量失败）**：评估过程报告了遗忘发生，实际是由BN状态漂移引起，而非权重真正擦除遗忘信息。

## 可复现要素
- **数据集**：CIFAR-10、CIFAR-100、Tiny-ImageNet、ImageNet-100（均为公开数据集）。
- **代码开源**：论文未明确声明GitHub仓库，但提及使用PyTorch的`torch.optim.swa_utils.update_bn`及sklearn的`LogisticRegression(solver='lbfgs')`，附录G提供了PGU的重实现说明；建议查阅arxiv来源获取代码链接。
- **权重开源**：未提及公开预训练权重；使用各方法reference implementation或最接近的re-implementation。
- **关键超参**：
  - BN重校准：$N_r \ge 5000$ retain样本，train模式，禁用梯度，mini-batch size b=128。
  - 线性探针：sklearn LogisticRegression，solver='lbfgs'，max_iter=1000，每类50样本。
  - 重学习：50遗忘类样本，50 epochs，lr=1e-4。
  - 九种方法超参详见附录F（如SalUn: lr=1e-3, steps=200, threshold=0.5；GA: lr=1e-3, steps=500）。
- **计算环境**：单张NVIDIA A6000 GPU（≈30GB VRAM），总wall-clock约90 GPU-hours。
