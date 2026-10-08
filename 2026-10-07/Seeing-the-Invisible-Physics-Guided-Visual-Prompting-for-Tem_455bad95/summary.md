---
title: "Seeing-the-Invisible-Physics-Guided-Visual-Prompting-for-Tem"
source: https://arxiv.org/pdf/2610.07558v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:25:18"
field: "具身导航与多模态感知"
keywords: ["VLA", "Vision-Language-Action", "Physical-Guided Visual Prompting", "Radiation Detection", "Thermal Hazard", "Zero-Shot Navigation"]
innovations: ["提出PG-VP模块，将辐射/温度危险转化为动态虚拟障碍物引导VLA导航", "双阶段物理引导辐射风险评估，结合逆平方律物理辅助损失", "零样本即插即用，无需重新训练VLA策略"]
benchmarks: ["R2R-CE val-unseen", "RxR-CE val-unseen", "Real-robot thermal/radiation scenarios"]
---

# 论文速读：Seeing-the-Invisible-Physics-Guided-Visual-Prompting-for-Tem

## 一句话总结
本文提出物理引导视觉提示（PG-VP）模块，将不可见的辐射/温度危险转化为动态虚拟障碍物叠加在RGB帧上，从而以零样本方式引导冻结的VLA导航策略绕开危险，无需额外训练。

## 研究问题与动机
- VLA模型在安全关键设施（如核电站）导航时，仅依赖RGB视觉无法检测不可见风险（辐射、高温）。
- 传统扩展多模态输入需新增编码器、配对数据和重新训练，成本高且可能破坏原有导航性能。
- 缺乏统一机制将非几何物理量无缝注入现有VLA导航框架。

## 核心贡献（创新点）
- 提出PG-VP即插即用模块，通过物理引导的风险评估确定规避方向，并动态渲染虚拟障碍物，以零样本方式复用冻结VLA模型的避障能力。
- 无论危险类型（辐射或温度），均使用相同视觉提示模式，确保跨模态对齐训练分布。
- 设计双阶段辐射风险评估模型（风险评估+方向估计），引入物理辅助损失提升方向预测准确性。
- 在仿真基准和真实机器人平台上验证，实现无需重新训练的安全导航。

## 方法详解
- **多模态感知系统**：集成RGB相机（前向广角+左右侧视）、辐射探测器（RadEye SPRD-ER）和热像仪（FLIR Lepton 3.5），部署于Unitree Go2 EDU四足机器人。
- **温度风险评估（RA）**：将热图像划分为四个区域（左/右侧、左/右前），通过最大温度差阈值$\Delta T_{\mathrm{th}}$判断危险方向，持续0.5秒确认。
- **辐射风险评估**：两阶段MLP，风险模型（33维输入）输出风险分数，方向模型（151维输入）输出左/右/正常概率，结合物理辅助损失（逆平方律重建CPS）。
- **动态视觉提示**：虚拟墙从图像边缘生长并收缩，长度$r_w(k)$按非线性函数变化，填充混凝土纹理，方向性阴影增强深度感。

## 实验与结果
- **数据集与基准**：OmniNav平台，使用R2R-CE和RxR-CE的val-unseen splits；真实机器人场景（热/辐射源）。
- **仿真结果**：CGR在R2R-CE上达84.9%，RxR-CE上83.2%，SR下降6.8和7.9个百分点；$S_{\mathrm{guide}}=0.60$为最佳平衡点。
- **真实实验**：对比基线，最坏10%平均轨迹安全性提升63.45%（热）和32.59%（辐射）。
- **最强结果**：PG-VP($S_{\mathrm{guide}}=0.60$)在R2R-CE上SR为61.9%，SPL为58.2%，优于多数单摄方法。

## 相关工作脉络
- **VLA导航基线**：NaVILA、StreamVLN、CorrectNav、JanusVLN、NavFoM、OmniNav、ABot-N0、SPAN-Nav，本文在OmniNav基础上扩展。
- **视觉提示方法**：BYOVLA、VAP、 waypoint预测标记等，本文注入的是不存在于场景的虚拟障碍物。
- **非几何物理感知**：Son等人物理信息机器学习辐射定位工作，本文扩展至风险评估与导航结合。
- **多模态VLA扩展**：Tactile-VLA、OmnivTLA等，本文以提示方式避免重训练。
- **定位差异**：首次将非几何物理量（辐射、温度）转化为视觉提示用于VLA导航，而非直接作为模型输入。

## 局限性与未来方向
- **局限性**：仅在实验室环境验证，未涉及复杂工业场景；虚拟障碍物可能干扰正常导航；辐射源被遮挡时触发延迟。
- **未来方向**：扩展到真实工业环境导航；支持模态特异性目标（如源定位、热点探索）；兼容气体等其他模态。

## 研究启发与可借鉴点
- 即插即用模块设计可迁移至其他模态扩展问题（如气体泄漏检测）。
- 动态视觉提示机制可用于引导策略绕过特定区域，无需修改策略网络。
- 物理辅助损失结合MLP的风险评估方法值得借鉴于其他物理量感知。
- 零样本适应思路适用于资源受限场景，避免频繁重训练。

## 关键术语表
- **VLA (Vision-Language-Action)**：将视觉、语言和动作映射统一的多模态模型。
- **PG-VP (Physics-Guided Visual Prompting)**：物理引导的视觉提示模块，将非几何物理量转化为视觉提示。
- **CGR (Correct-Guidance Ratio)**：正确引导轨迹的比例，衡量提示有效性。
- **RA (Risk Assessment)**：风险评估模型，输出危险方向。
- **CPS (Counts Per Second)**：每秒计数，辐射测量单位。
- **PIML (Physics-Informed Machine Learning)**：物理信息机器学习，将物理定律融入模型训练。
- **OmniNav**：底层导航策略，采用快慢双系统架构。
- **$S_{\mathrm{guide}}$**：引导强度参数，控制虚拟墙最大长度比例。

## 可复现要素
- **数据集**：R2R-CE、RxR-CE公开；OmniNav环境未提及开源。
- **代码/权重**：使用OmniNav官方代码，PG-VP代码未提及开源。
- **关键超参**：$S_{\mathrm{guide}}=0.60$、$N_g=N_v=4$、$\Delta T_{\mathrm{th}}=30$。
