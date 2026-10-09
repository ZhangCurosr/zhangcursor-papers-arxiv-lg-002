---
title: "Kuration-SDK-Addressing-the-Virtual2Real-Gap-via-Data-Curati"
source: https://arxiv.org/pdf/2610.09305v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-09 16:46:14"
field: "world model evaluation and data quality"
keywords: ["world models", "data curation", "virtual2real gap", "action-conditioned", "benchmark", "Kuration SDK", "CS:GO", "diffusion models"]
innovations: ["实证揭示Virtual2Real gap：FVD/LPIPS/JEDi指标与模型可玩性不匹配", "定位action-state consistency为关键数据信号，提出测量算子", "开源Kuration SDK支持可复现的游戏数据筛选流水线"]
benchmarks: ["FVD", "LPIPS", "JEDi", "DIAMOND baseline"]
---

# 论文速读：Kuration SDK: Addressing the Virtual2Real Gap via Data Curation

## 一句话总结
本文揭示了动作条件世界模型的现有评估指标（FVD、LPIPS、JEDi）与模型实际可玩性之间的"Virtual2Real gap"，并开源了Kuration SDK工具包，通过数据筛选与诊断发现**动作-状态一致性（action-state consistency）**是 bridging gap 的关键信号。

## 研究问题与动机
1. **Benchmark失灵问题**：当前评估动作条件世界模型的指标（如FVD、LPIPS）主要衡量视觉相似度，无法捕捉物理违反、地图传送门等可玩性失败模式。
2. **缺乏通用诊断信号**：新兴的动作语义和物理 grounded benchmark（如ACT-Bench、ARR）均为任务特定，无法为domain-agnostic的世界模型训练提供通用质量信号。
3. **数据质量问题被忽视**：现有工作将DIAMOND等模型的失败归因于架构，但本文证明训练数据的内在差异（如动作响应的像素偏移幅度）才是根本原因。
4. **Sim2Real vs Virtual2Real**：与Sim2Real关注动力学保真度不同，Virtual2Real是数据分布问题——未筛选的虚拟环境样本会继承其退化边角案例。

## 核心贡献（创新点）
1. **实证揭示Virtual2Real gap**：首次在格式与分布匹配的CS:GO和CS2语料上训练相同架构模型，证明FVD/LPIPS/JEDi指标相似时，模型可玩性可能显著分化。
2. **定位根因假设**：通过控制实验发现两数据集在action-state consistency（单动作导致的像素偏移幅度）上存在1.4–2.1倍差异，这是现有指标无法捕捉的。
3. **开源Kuration SDK**：发布通用游戏数据筛选工具包，包含像素偏移测量、碰撞检测、SLAM轨迹重建等算子，支持可复现的数据配方。
4. **验证数据筛选策略**：对比四种筛选策略，证明只有基于action-state consistency的筛选能显著改善可玩性，而直觉上"物理相关"的策略（碰撞感知、稀有轨迹）无效。

## 方法详解
1. **DIAMOND复现与基线建立**：
   - 采用两阶段架构：denoiser（处理4帧+动作，输出30×56）+ upsampler（放大至150×280）
   - 20类离散动作（鼠标+键盘输入），学习率10⁻⁴，weight decay 10⁻²，600 epoch
   - 复现结果与原文FVD误差<5%，验证评测管道可靠性

2. **Controlled实验设计**：
   - 使用CS2语料（8,635 episodes，16fps）训练baseline_v1，与DIAMOND的CS:GO语料在地图（de_dust2）、分辨率、帧率、动作编码上严格匹配
   - 交叉评估：DIAMOND权重在CS2测试集上FVD恶化4.7倍，证明存在domain gap

3. **Action-State Consistency算子**：
   - `pixel_shift`：测量每次按键后的光流位移，投影到控制轴
   - `action_state_consistency`：聚合为每控制、每按键时长的响应曲线
   - 发现：CS:GO的峰值像素偏移比CS2大1.4–2.1倍；CS2释放后减速（76–78%），CS:GO反而加速（112–126%）

4. **Kuration SDK流水线**：
   ```python
   curated = (input(...).join_video(...).window(64, stride=32)
                .pixel_shift(controls=["W"])
                .score("action-state-consistency", baselines=baselines)
                .filter("w_shift_measured", ">=", 1)
                .bin("w_shift_mean", bins=8)
                .sample("equal", n=N))
   ```
   - 四种筛选策略：coverage-based（分位数均匀覆盖）、rare-trajectory（尾部分布上采样）、collision-aware（碰撞事件加权）、action-state-consistency（像素偏移幅度加权）

## 实验与结果
1. **数据集**：CS:GO语料（DIAMOND公开）、CS2语料（8,635 episodes），测试集1,090 episodes
2. **评估指标**：FVD、LPIPS、JEDi（基于V-JEPA特征的未来潜在状态预测）
3. **关键结果**：
   - baseline_v1（CS2未筛选）在epoch 600达到LPIPS 0.4785，比DIAMOND（0.6049）提升20.9%
   - 但交互 rollout 显示baseline_v1可玩性更差：更严重的几何穿透、空间不连续、控制响应差
   - FVD最佳值出现在epoch 60（2,898,653），之后540个epoch无改善，说明FVD是差的停止信号
4. **筛选策略效果**（Table 2）：
   - Coverage-based、Rare-trajectory、Collision-aware：无可玩性改善
   - Action-state-consistency：唯一显示可玩性改善的策略
5. **最强结果**：DIAMOND在CS:GO测试集上FVD最低（1,571,046 @ epoch 300），但CS2语料训练的模型指标更好却可玩性更差

## 相关工作脉络
1. **World models for physical AI**：Genie（互联网视频无动作标签）、GameNGen/DIAMOND（实时扩散游戏世界模型）——本文填补了跨语料对照实验的空白
2. **Video generation metrics**：FVD/LPIPS（I3D特征+感知距离）不针对单次交互物理违反；JEDi（V-JEPA特征）理论上更敏感但实际仍无法捕捉可玩性
3. **Action-semantics benchmarks**：ACT-Bench（需逐游戏轨迹估计器）、ARR（需辅助probe）均为任务特定，缺乏通用性
4. **Data curation**：FineWeb（文本去重）、SemDeDup（语义去重）依赖固定嵌入，无法暴露物理交互、空间轨迹等游戏数据关键特征
5. **Sim2Real vs Virtual2Real**：传统Sim2Real通过domain randomization解决动力学保真度；本文定义的Virtual2Real是数据分布问题，即使模拟器准确也会继承退化样本

## 局限性与未来方向
1. **结果局限性**：
   - 可玩性判断依赖作者交互式rollout检查，非盲审、无第二评分者
   - CS:GO与CS2运行不同引擎代（Source vs Source 2），部分差异可能源于引擎级属性而非纯数据选择
   - 仅在一个地图、一个架构、一个游戏系列上验证，"task-agnostic"主张尚未推广
   - 多种策略的定量指标未完整评估（GPU共享导致）

2. **未来方向**：
   - 构建基于几何穿透和轨迹不连续性的physics-conformance metric
   - 建立shared benchmark比较SDK全策略空间
   - 验证action-state consistency等筛选轴能否迁移到真实机器人遥操作或第一人称视频

## 研究启发与可借鉴点
1. **数据质量>指标优化**：在模型达到 competent 水平后，FVD/LPIPS等指标无法可靠指导数据筛选决策，需引入行为级诊断信号
2. **控制实验的价值**：保持架构、超参、数据格式一致，仅改变数据来源，可干净地隔离数据因素
3. **Kuration SDK的设计模式**：流水线式算子（window→measure→score→filter→sample）支持可复现数据配方，为社区提供通用基础设施
4. **可迁移假设**：action-state consistency概念可推广至其他sequential data领域（如机器人操作、第一人称视频），作为数据筛选的新维度
5. **负控制实验建议**：未来工作应运行"反向筛选"（curate for wrong direction）以排除fine-tune随机性影响

## 关键术语表
**Virtual2Real gap**：动作条件世界模型在虚拟环境数据上训练后，其评估指标与真实可玩性之间的不匹配现象
**Action-state consistency**：给定输入动作是否可靠产生相同特征的屏幕运动及其幅度，是本文识别的关键数据属性
**FVD (Fréchet Video Distance)**：基于I3D特征的Fréchet距离，衡量生成视频与真实视频的分布差异
**LPIPS (Learned Perceptual Image Patch Similarity)**：基于深度特征的逐帧感知距离指标
**JEDi**：基于V-JEPA特征的视频生成度量，预测未来潜在状态而非分类动作
**DIAMOND**：基于扩散模型的动作条件世界模型，在CS:GO游戏数据上训练
**Kuration SDK**：开源的游戏/序列数据筛选工具包，提供像素偏移测量、碰撞检测、SLAM轨迹重建等算子
**Coverage-based sampling**：数据筛选策略，按特征分位数均匀选择片段以确保多样性覆盖

## 可复现要素
- **数据集**：CS:GO语料（DIAMOND公开），CS2语料（data partner提供，未公开）
- **代码**：Kuration SDK开源（论文声明），pip installable
- **权重**：DIAMOND官方权重可用，baseline_v1未提及开源
- **关键超参**：学习率10⁻⁴，weight decay 10⁻²，600 epochs × 400 steps，batch size匹配原文
