---
title: "TokenSTFormer-A-Tokenized-Spatial-temporal-Attention-Model-f"
source: https://arxiv.org/pdf/2608.16122v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:33:44"
field: "医疗AI与骨科筛查"
keywords: ["Adolescent Idiopathic Scoliosis", "Gait Analysis", "Spatial-Temporal Tokenization", "Vision Transformer", "Medical AI Screening", "Kinematic Knowledge Map"]
innovations: ["TokenSTFormer: 空间-时间tokenization增强的Vision Transformer变体", "Kinematic Knowledge Map: 238维结构化运动特征工程", "ScoliGait: 首个步态视频+X光配对的脊柱侧凸数据集"]
benchmarks: ["Accuracy", "Sensitivity", "Specificity", "PV+", "PV-"]
---

# 论文速读：TokenSTFormer: A Tokenized Spatial-Temporal Attention Model for Holistic Motion Analysis in Adolescent Idiopathic Scoliosis Screening

## 一句话总结
本文针对青少年特发性脊柱侧凸（AIS）筛查提出了一种基于步态视频的新型分析方法——TokenSTFormer模型，通过空间-时间tokenization技术提取整体运动特征，实现可规模化、低成本的AIS初筛，在自建数据集ScoliGait上达到0.79准确率，优于传统Vision Transformer基线。

## 研究问题与动机
1. **核心问题**：AIS是全球约5%青少年常见的脊柱侧弯畸形，若未及时干预会导致慢性背痛和心理困扰；现有金标准依赖X光测量Cobb Angle >10°，但反复X光照射存在累积辐射风险。
2. **传统筛查方法局限**：Adams前屈试验和Scoliometer测量依赖专业人员主观判断，受肥胖等因素干扰（参考文献7），难以大规模推广。
3. **静态图像方法不足**：单张照片分析（参考文献8）缺乏运动生物力学信息；现有方法如GaitEdge、SkeletonGait虽提取时空特征，但需要复杂的预处理管线（合成剪影图/骨架图），实用性受限。
4. **研究动机**：开发一种非侵入性、可扩展、基于自然步态视频的AIS筛查系统，避免X光辐射风险，同时利用手机摄像头实现低成本部署。

## 核心贡献（创新点）
1. **提出ScoliGait数据集**：首个同时包含步态视频与对应X光记录的脊柱侧凸数据集，1,516个视频片段（758名参与者，配对医疗标签）。
2. **设计Kinematic Knowledge Map（运动学知识图谱）**：从姿态估计数据中构建238维结构化运动特征（140个运动空间特征+32个骨骼空间特征+66个信号相关特征），代表整体步态运动的语义表示。
3. **提出TokenSTFormer模型**：在Vision Transformer基础上引入空间-时间tokenization（STT）模块，通过独立卷积层分离空间和时序token，再经LayerNorm和Dense层融合，增强特征表征能力。
4. **辅助损失设计**：使用两个CLS token的空间-时间一致性MSE作为辅助损失，强化模型对多视角特征对齐的学习。
5. **隐私保护的系统设计**：通过去标识化处理和移动端部署，实现规模化筛查可行性。

## 方法详解
### 3.1 Kinematic Knowledge Map构建
1. 使用YOLOv8进行2D姿态估计，从步态视频中提取关节坐标(x,y)。
2. 构建三类特征：
   - **运动空间特征（140维）**：整体步态模式，通过欧氏距离计算关节相关特征，三角函数计算运动角度。
   - **骨骼空间特征（32维）**：个体骨骼结构，表示身体各部位的相对位置关系。
   - **信号相关特征（66维）**：通过信号互相关（SciPy库）计算运动滞后和信号关系。
3. 所有数值经依赖归一化处理。

### 3.2 TokenSTFormer模型架构
1. **基础架构**：借鉴Vision Transformer，使用残差块 composed of Multiheaded Self-Attention (MSA) 和 Multi-Layer Perceptron (MLP)。
2. **空间-时间Tokenization (STT)**：
   - 使用2D卷积层分别处理运动学知识图谱的列方向和行方向，生成时序token $z_{temporal}^{(t,d)}$ 和空间token $z_{spatial}^{(v,d)}$。
   - 公式(3)：$z_{input}^{(t+v,d)} = concat(LN(z_{temporal}), LN(Dense(z_{spatial})))$
   - 时序token和空间token分别经过LayerNorm，空间token额外经过Dense层增强表征。
3. **辅助损失**：
   - 主损失：二元交叉熵（BCM）用于分类。
   - 辅助损失：公式(4) $Loss_{CLS} = MSE(CLS_{temp}, CLS_{spt})$，约束两个CLStoken的一致性。
   - 总损失：公式(5) $Loss = Loss_{BCM} + Loss_{CLS}$。
4. **集成模块**：LayerScale、Stochastic Depth、Dropout(0.1)。

### 3.3 实验设置
1. **数据划分**：训练集1,216，验证集150，测试集150（正负样本比2.2:1分层抽样）。
2. **超参数**：MLP维度=384，注意力头数=6，层数=5，学习率=2e-5，batch size=64，Adam优化器，余弦学习率调度。

## 实验与结果
### 数据集
- **ScoliGait数据集**：758名参与者（男320/女722），年龄13.86±2.44岁（阳性）/11.59±2.86岁（阴性），1,516个非重叠视频片段（30Hz，1080p，5秒/片段）。
- 标签：基于X光Cobb Angle >10°诊断脊柱侧凸（金标准）。

### 评估指标
准确率、敏感性、特异性、PV+、PV-。

### 主要结果
| 方法 | Accuracy | Sensitivity | Specificity | PV+ | PV- |
|------|----------|-------------|-------------|-----|-----|
| 传统方法（文献报告平均） | - | 0.447 | 0.900 | 0.683 | 0.560 |
| Transformer encoder（基线） | 0.740 | 0.796 | 0.617 | 0.820 | 0.580 |
| **TokenSTFormer（本文）** | **0.787** | **0.845** | **0.660** | **0.845** | **0.660** |

### 提升幅度
- 相比vanilla Vision Transformer encoder：准确率提升4.7个百分点（0.740→0.787），敏感性提升4.9个百分点。
- 消融实验表明：移除LayerNorm后准确率降至0.720，使用单一位置编码后降至0.687，验证STT模块的有效性。

### 最强结果
- **TokenSTFormer在测试集上达到0.787准确率，0.845敏感性**，显著优于基线模型和传统筛查方法。

## 相关工作脉络
1. **GaitEdge [10]**：基于合成剪影图的端到端步态识别，预处理复杂；本文直接使用原始视频构建运动学知识图谱，减少预处理依赖。
2. **SkeletonGait [11]**：使用骨架图进行步态识别，需要复杂的姿态估计生成功能图；本文保留原始视频信息，通过YOLOv8提取2D坐标后直接构建结构化特征。
3. **单张手机照片方法 [8]**：仅利用静态背部不对称信息，缺乏运动生物力学洞察；本文利用动态步态视频提取时序特征，提供更全面的诊断依据。
4. **传统筛查方法 [5,6]**：Adams试验和Scoliometer测量主观性强、依赖专业人员；本文模型实现自动化、客观化评估。
5. **Vision Transformer [15]**：基础架构；本文扩展至时序-空间联合建模，通过STT模块解决医疗时序数据的表征挑战。
6. **运动学研究 [13,14]**：证实脊柱侧凸步态存在可检测偏差；本文将这些先验知识结构化融入特征工程。

## 局限性与未来方向
1. **数据集规模有限**：仅758名参与者，1,516个视频片段，可能影响模型泛化能力；需更大规模多中心验证。
2. **横断面研究**：未涉及纵向随访和疾病进展预测，难以评估筛查后的临床干预效果。
3. **仅使用2D姿态估计**：未融合3D运动学信息或肌电图等多模态数据。
4. **临床部署挑战**：实际应用场景中步态视频质量可能不稳定，模型鲁棒性有待验证。
5. **未来方向**：扩展到多病种脊柱健康评估；结合可穿戴设备实现持续监测；探索自监督预训练减少标注依赖。

## 研究启发与可借鉴点
1. **结构化特征工程+深度学习融合**：将领域先验知识（运动学特征）与端到端模型结合，既保证可解释性又发挥深度学习表征能力，适用于医疗时序数据建模。
2. **空间-时间解耦tokenization**：STT模块通过独立卷积核分离空间和时序维度，再经归一化和融合，对时序分类任务具有通用迁移价值。
3. **辅助一致性损失**：使用两个不同视角token的MSE损失促进多视图特征对齐，可推广至多模态学习场景。
4. **隐私保护设计**：去标识化+移动端部署思路，为医疗AI落地提供工程参考。
5. **分层抽样策略**：针对医疗数据类别不平衡，采用2.2:1分层抽样，值得类似研究借鉴。

## 关键术语表
- **Adolescent Idiopathic Scoliosis (AIS)**：青少年特发性脊柱侧凸，青春期常见的脊柱侧弯畸形，确诊需Cobb Angle >10°。
- **Cobb Angle (CA)**：冠状面X光测量脊柱侧弯角度的金标准，>10°诊断为脊柱侧凸。
- **Kinematic Knowledge Map**：运动学知识图谱，238维结构化特征表示，包含运动空间、骨骼空间和信号相关三类特征。
- **Spatial-Temporal Tokenization (STT)**：空间-时间tokenization，通过独立卷积层分离并融合时空特征的模块设计。
- **TokenSTFormer**：本文提出的模型名称，基于Vision Transformer架构，集成STT模块的脊柱侧凸筛查模型。
- **ScoliGait Dataset**：本文构建的首个步态视频+X光配对数据集，1,516个片段，758名参与者。
- **Positive Predictive Value (PV+)**：阳性预测值，预测阳性中真正阳性的比例。
- **Negative Predictive Value (PV−)**：阴性预测值，预测阴性中真正阴性的比例。

## 可复现要素
- **数据集**：ScoliGait数据集，论文未明确公开声明（"to the best of our knowledge"暗示可能有限开放）。
- **代码**：论文未提及开源仓库，需联系作者获取。
- **权重**：未公开提供预训练权重。
- **关键超参**：MLP维度=384，注意力头数=6，层数=5，学习率=2e-5，batch size=64，Dropout=0.1，warmup ratio=0.1。
- **依赖库**：YOLOv8（姿态估计）、SciPy（信号互相关）。
- **训练环境**：论文未详细说明硬件配置。
