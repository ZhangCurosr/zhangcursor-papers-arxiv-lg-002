---
title: "woma-a-real-time-foundation-model-and-its-fine-tuned-models"
source: https://arxiv.org/pdf/2609.15130v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 22:00:44"
field: "医学影像基础模型"
keywords: ["foundation model", "self-supervised learning", "endoscopy", "real-time inference", "colonoscopy", "gastroscopy", "ConvNeXt"]
innovations: ["预注册实验设计筛选八候选基础模型，冻结探针预测微调排名", "单GPU实时多任务推理达247 fps超越TensorRT/PyTorch/ONNX Runtime", "无厂商依赖Vulkan构建实现跨平台零 toolkit 部署"]
benchmarks: ["PolypGen", "REAL-Colon", "GastroHUN", "Kvasir-SEG", "UGIAD", "GastroVision", "EDD2020"]
---

# 论文速读：woma-a-real-time-foundation-model-and-its-fine-tuned-models

## 一句话总结
woma是一个面向胃肠道内镜的实时基础模型，在无标签约100万帧内镜图像上自监督训练，随后微调出结肠镜和胃镜两个任务模型；采用预注册实验设计与严格的生产导向评估协议，在多个公开跨中心数据集上达到或超越预设通过标准，单GPU推理速度超过PyTorch、ONNX Runtime和TensorRT，且提供无厂商依赖的Vulkan便携式构建。

## 研究问题与动机
- **从研究到产品的全链路缺失**：现有内镜AI研究多聚焦单一检测任务，但缺乏从基础模型选择、头部分配、权重交付到部署优化的完整产品化路径，尤其缺少"已发表数字背后的真实评估流程"。
- **实时性要求难以满足**：内镜处理器可达60 fps（每帧16.7 ms），基础模型与特征金字塔颈部需在同一工作站GPU上占用≤8 ms，现有方案在多任务叠加后常超出预算。
- **虚假警报导致临床弃用**：系统对正常黏膜过度触发警报会促使临床医生关闭AI辅助，因此要求高特异性同时维持高灵敏度。
- **跨中心泛化与评估可信度**：训练数据与评估数据需严格隔离（如PolypGen六家医院、REAL-Colon全视频），避免帧级划分导致的数据泄露膨胀分数（论文中帧级划分使检测率从0.67虚高至0.91）。

## 核心贡献（创新点）
1. **生产导向的系统化设计框架**：在首次运行前固定需求与通过标准，八候选架构按预注册规则筛选，自监督训练遵循停止规则，微调与部署优化在同一自包含库numbat上完成，确保每个数字均读取自训练未见数据。
2. **woma基础模型及两个微调模型**：基于ConvNeXt-V2-Nano架构（15.6M参数），结肠镜模型可检出并勾画息肉、命名结肠节段、建议息肉类型、评估肠道准备质量；胃镜模型可识别22个协议位点、标记并勾画病变、命名七类发现，每项指标均独立报告达标或未达标情况。
3. **端到端实时推理引擎超越主流框架**：在单工作站GPU上，基础模型与前向推理在f16精度下达约247 fps（结肠镜）和243 fps（胃镜），较TensorRT快6-31%，较PyTorch/ONNX Runtime快1.6-3.0倍；且提供无CUDA/cuDNN依赖的Vulkan构建，在f32下甚至比CUDA构建更快。
4. **预注册实验验证基础模型选择的有效性**：冻结特征探针（GastroHUN定位分类与Kvasir-SEG息肉勾画）在Stage 1排序与两周微调后的最终排名完全一致，证明"先花测量时间再花GPU时间"的设计哲学有效。

## 方法详解
**基础模型架构**：woma采用ConvNeXt-V2-Nano结构，深茎（3×3 stride 2 + 3×3 + 3×3）将4×4 patch转为80特征，四级Stage输出stride 8/16/32特征图（P3-P5），通道数80→160→320→640，每块含7×7 depthwise卷积、通道归一化、两个1×1卷积（GELU激活）及GRN（全局响应归一化）。

**自监督预训练策略**：采用掩码自编码器（MAE）目标，先在ImageNet权重上初始化，再于约150万无标签帧（GastroNet-5M + 内部视频）上继续训练；该目标保证每位置特征学习、防止表征坍塌（effective rank ≥64警告线），且适用于卷积网络。

**多任务头设计**：一个前向传播服务所有头，共享特征金字塔颈部；检测头为anchor-free YOLOv8风格，边界框以16分箱分布表示；勾划头为轻量解码器输出640²掩码；定位头为双层分类器（10结肠节段或22+1胃位点）；息肉类型/发现头为线性分类器（4类或7类）；肠道准备头预测Boston量表四类。

**微调配方**：每步按权重随机采样一个任务执行单批处理，基础模型与头分离训练（头学习速率为基础模型的10倍）； shipped权重为运行平均，选择规则为优先满足病变门控特异性，其次达标数量最多。

**停止规则**：每20分钟探针测试，当两个探针在最近3000步增益均<0.005时停止（小于两随机种子间差异）。

## 实验与结果
**数据集**：预训练使用GastroNet-5M等约150万无标签帧；结肠镜评估用PolypGen（六医院1347息肉帧）、REAL-Colon study 4（15视频19息肉）、15个CAS-Colon视频、HyperKvasir地标、KUMC验证集、三个跨院PraNet勾画集；胃镜评估用GastroHUN验证/测试、EDD2020（39新生物帧vs1347清洁帧）、UGIAD测试集、GastroVision半集。

**主要结果**：
- 结肠镜：PolypGen灵敏度0.960（精确度≥0.85），REAL-Colon每息肉灵敏度1.000（每 procedure 1.6次误报），跨院勾画Dice均值0.807（未达0.82标准），十段宏观召回0.521（未达0.70标准）
- 胃镜：地标区域准确率0.923（达0.92标准），病变门控灵敏度0.949/特异性0.912（达标），发现宏观召回0.877（达标），UGIAD粗五准确率0.932（达标）
- 速度：f16精度下结肠镜247.5 fps、胃镜243.3 fps；TensorRTclosest但仍慢3-27%；Vulkan便携构建f32达184.8 fps（结肠镜），超越CUDA构建

**速度对比关键数字**：基础模型单前向在strict f32下numbat 4.30ms vs TensorRT 4.83ms；f16下frame-to-results Colon model 247.5 fps vs TensorRT 215.3 fps。

## 相关工作脉络
- **Repici等（Gastroenterology 2020）**：首次RCT证实实时息肉检测提升腺瘤检出率，本文继承其实时检测理念但扩展到多任务基础模型架构。
- **Hirasawa等（Gastric Cancer 2018）**：卷积网络胃癌检测灵敏度92%，本文胃镜病变门控灵敏度0.949且同时提供定位与发现分类。
- **Boers等（Medical Image Analysis 2024）**：GI基础模型架构与预训练方式对比研究，本文在其结论基础上验证ConvNeXt-V2-Nano+MAE组合在跨中心内镜任务上的最优性。
- **Ali等（PolypGen, Scientific Data 2023）**：多中心息肉检测数据集，本文以其六医院划分作为跨中心泛化核心评测基准。
- **Bifi等（REAL-Colon, Scientific Data 2024）**：真实世界结肠镜视频数据集，本文以其全视频19息肉作为过程级评估标准。
- **Fan等（PraNet, MICCAI 2020）**：息肉分割基线，本文勾画任务与其protocol对比但指出单任务模型在参数量上数倍于woma。

## 局限性与未来方向
- **部分指标未达标**：跨院勾画Dice差0.013、十段宏观召回0.521（单帧天花板限制）、息肉类型宏召回仅0.617（锯齿状息肉仅256训练帧）。
- **评估样本量较小**：REAL-Colon仅19息肉（95%下限0.83）、胃镜新生物帧仅39张，无法覆盖 blur、气泡、器械遮挡、运动胃癌等真实场景。
- **缺少完整胃镜过程数据集**：无带逐病灶标注的全程胃镜视频公开数据集，限制系统级验证。
- **GPU为工作站级别**：未在手术室实际设备（如集成嵌入式GPU）上测试，延迟与功耗表现待验证。
- **未来方向**：引入时序记忆模块提升节段定位；扩充锯齿状与微小息肉训练数据；部署于现场设备采集真实流程数据验证。

## 研究启发与可借鉴点
- **预注册实验设计范式**：在数据运行前固化候选集、通过标准与决策规则，区分"路线失败"与"候选失败"（如三个蒸馏候选因教师目标坍塌而淘汰，若非双候选设计则无法归因）。
- **冻结探针有效性验证**：Stage 1两小时冻结探针完美预测两周微调后的排名顺序，证明在资源受限时可大幅减少GPU消耗。
- **训练-推理一致性保障**：同一库numbat承载数据流、训练器、评估器与运行时，确保 shipped文件复现验收记录至三位小数，为部署可追溯性提供模板。
- **跨站硬件无关构建**：Vulkan/SPIR-V路径消除厂商库依赖，在AMD/Intel硅片及未来Apple silicon上零代码修改部署，为医疗AI的异构部署提供范式。
- **部分标签损失处理**：肠道准备标注存在分组标签（{0,1}或{2,3}），采用partial-label loss仅要求组正确而非猜测具体值，可迁移至其他标注不完整场景。

## 关键术语表
- **Foundation model（基础模型）**：在无标签大规模数据上预训练的通用特征提取器，下游任务头可在此基础上微调，woma为15.6M参数的ConvNeXt-V2-Nano。
- **Effective rank（有效秩）**：特征矩阵的谱熵度量，用于检测表征坍塌；本文警告线设为64，坍塌候选有效秩低至6.2。
- **MAE（Masked Autoencoder）**：掩码自编码器自监督目标，通过重建随机掩码patch学习纹理敏感特征，适合病变识别。
- **Pass mark（通过标准）**：实验前预注册的性能阈值，源自已发表基准或理论下限（如随机猜测 parity），如PolypGen灵敏度≥0.90。
- **Portability build（便携构建）**：不链接CUDA/cuDNN/cuBLAS，改用自定义Vulkan内核的部署版本，跨厂商GPU可用。
- **Partial-label loss（部分标签损失）**：标注仅给出类别组而非具体值时，损失函数仅惩罚组外预测而非猜测错误。
- **Anchor-free head（无锚框头）**：YOLOv8风格检测头，边界框以连续分布（16分箱）直接回归，避免手工设计锚框。
- **Blind-spot rate（盲区率）**：胃镜未检查区域的占比，本文引用WISENSE系统将其从22%降至6%作为实时辅助价值依据。

## 可复现要素
- **数据集**：PolypGen、REAL-Colon、GastroNet-5M、HyperKvasir、GastroHUN、Kvasir-SEG、CVC-ColonDB、ETIS、CVC-300、UGIAD、GastroVision、CAS-Colon、KUMC等均已公开，论文明确标注各数据集角色（预训练/微调/仅评估）。
- **代码与权重**：numbat库（0.9.14发布版）描述于Tran & Dang [6,7]；模型权重与程序可通过通讯作者获取； Records与benchmark脚本可复现所有表格图表。
- **关键超参**：输入分辨率640×640，batch size 12（四卡×每张四图像），优化器为Decoupled weight decay，学习率调度含衰减，有效rank警告线64，停止阈值0.005/3000步。
- **评估协议**：帧级划分严禁使用，按视频或患者拆分；PolypGen precision固定≥0.85操作点；REAL-Colon允许≤2次持续误报/ procedure；速度测试三重复中位数，所有引擎输出与PyTorch对齐验证。
