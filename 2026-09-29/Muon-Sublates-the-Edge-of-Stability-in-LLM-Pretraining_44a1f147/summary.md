---
title: "Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining"
source: https://arxiv.org/pdf/2609.34915v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:34:04"
field: "大语言模型训练优化"
keywords: ["Muon 优化器", "稳定性边缘", "LLM 预训练", "学习率分析", "极分解", "优化动力学"]
innovations: ["推导相干校正的条件损失中性边界 2ρ/η，证明 Muon 打破经典 EoS 耦合", "提出分裂的 EoS 框架：损失中性边界存活但时序方向解耦", "系统验证 batch size 对时序对齐的影响及 LLM 层间异质性"]
benchmarks: ["CIFAR-10 (MLP/CNN)", "TinyShakespeare (Transformer)", "FineWeb (130M/1B LLM 预训练)"]
---

# 论文速读：Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining

## 一句话总结
论文研究了 Muon 优化器在 LLM 预训练中的大步骤动态行为，发现其打破了经典梯度下降"稳定性边缘（EoS）"中损失中性、更新反向与边际稳定性三者的耦合关系，提出"分裂的 EoS"框架：随机损失中性边界得以保留，但时序方向特征不再普遍伴随出现。

## 研究问题与动机
- **核心问题**：Muon 优化器广泛应用于 LLM 预训练，但其基于近似半正交极方向的更新机制改变了优化几何，经典 GD 的 EoS 图景无法直接迁移；目前尚无研究验证 EoS 是否在 Muon 预训练中存活。
- **现有方法不足**：
  1. 经典 EoS 理论（Cohen et al., 2021；Litman, 2026）主要针对确定性/全批 GD，假设损失中性、方向反向与边际稳定性在单一边界 $2/\eta$ 处耦合。
  2. Muon 的极性归一化将每次矩阵梯度替换为近似半正交方向，改变了更新的几何结构而非仅改变尺度，导致批采样与全局梯度的对齐偏差。
  3. 已有的谱感知分析（Islamov et al., 2026）主要针对确定性 Spectral GD，未考虑 LLM 预训练中的新鲜小批量、异构矩阵块及辅助 AdamW 优化器。
  4. 缺乏对随机 Muon 下 EoS 特征的理论与实验验证，尤其在大模型规模上。

## 核心贡献（创新点）
1. **推导了相干校正的条件损失中性边界**：为随机无动量精确极 Muon 推导出精确条件恒等式，给出损失中性边界 $s_{k,b}^{\mathrm{M}} = 2\rho_{k,b}/\eta_k$，其中 $\rho_{k,b}$ 为批次到总体的相干性度量；全批情况退化为经典 $2/\eta_k$，但小批量通过相干性移动边界。
2. **证明损失中性与时序方向解耦**：通过矩阵二次型严格证明，相同的有效损失曲率可产生正或负的方向对齐，T1 与 T2 事件可发生任意顺序，打破经典 EoS 中"损失平衡决定方向"的耦合假设。
3. **提出"分裂的 EoS"框架**：Muon 既非简单保留也非消除经典 EoS，而是"扬弃"——保留相干校正的损失中性边缘，但打破损失平衡与更新反向的耦合，形成两套独立信号。
4. **系统验证了 batch size 对时序对齐的影响**：在受控实验中，增大 batch size 使时序对齐趋向相干反向（$c^{\mathrm{pol}} \to -1$），而小 batch 轨迹保持接近正交；1B LLM 在较大 batch（512）下呈现更强的部分抵消但仍远离精确反向。
5. **揭示了 LLM 预训练中的层间异质性**：130M 模型的层间 directional cosine 测量显示，全局弱负对齐掩盖了中间层 key/value 矩阵与 query 矩阵的差异，补充了 Wang et al. (2026) 的曲率视角。

## 方法详解
**Muon 更新机制**：
- 对于矩阵参数块 $W_k$，计算小批量梯度 $\widehat{G}_k = \nabla L_{B_k}(W_k)$，通过 SVD 分解 $G = U_r \Sigma_r V_r^\top$ 得到极映射 $\operatorname{Pol}(G) = U_r V_r^\top$。
- 更新公式：$W_{k+1} = W_k - \eta P_k$，其中 $P_k = \operatorname{Pol}(\widehat{G}_k)$。

**两个诊断量定义**：
1. **T1：条件损失中性边界**
   - 相干性度量：$\rho_b(W_k) = \frac{\mathbb{E}_b \langle G_k, P_k \rangle_F}{\|G_k\|_*} \in [-1, 1]$，衡量保留的一阶梯度下降比例。
   - 有效曲率：$s_b^{\mathrm{M}}(W_k; \eta) = \frac{2\mathbb{E}_b[L(W_k - \eta P_k) - L(W_k) + \eta \langle G_k, P_k \rangle_F]}{\eta^2 \|G_k\|_*}$，$\eta \to 0$ 时趋近于 $\frac{\mathbb{E}_b \langle P_k, H_k[P_k] \rangle_F}{\|G_k\|_*}$。
   - 定理 2.1：条件期望损失变化为 $\mathbb{E}_b[L(W_{k+1}) - L(W_k)|W_k] = \frac{\eta^2 \|G_k\|_*}{2}(s_b^{\mathrm{M}} - \frac{2\rho_b}{\eta})$，损失非增当且仅当 $s_b^{\mathrm{M}} \leq 2\rho_b/\eta$。

2. **T2：时序正交边界**
   - 连续更新方向余弦：$c_{k,b}^{\mathrm{pol}} = \frac{\langle P_k, P_{k+1} \rangle_F}{\|P_k\|_F \|P_{k+1}\|_F} \in [-1, 1]$。
   - T2 阈值：$c_{k,b}^{\mathrm{pol}} = 0$ 区分锐角/钝角对齐，$c_{k,b}^{\mathrm{pol}} = -1$ 表示相干反向（$P_{k+1} = -P_k$，两步返回参数）。

**矩阵二次型玩具模型**：
- 损失函数 $L_{X,E}(W) = \frac{1}{2}\|WX - Y\|_F^2$，其中 $Y = W_\star X + E$。
- 推导得 $s_k^{\mathrm{M}} = \frac{\|X\|_F^2}{\operatorname{tr}(S_k)}$（依赖迹），$c_k^{\mathrm{pol}} = 1 - \frac{2\#n_-(S_k - \eta XX^\top)}{d}$（依赖负特征值计数）。
- 证明相同曲率可对应相反对齐符号。

**实验设置**：
- 受控实验：MLP（411K 参数）、CNN（545K 参数）、Transformer（112K 参数）在 CIFAR-10 / TinyShakespeare 上训练。
- LLM 实验：22M、130M、1B 参数的 Llama-like 模型，使用 FineWeb 数据，序列长度 4096，batch size 32/512，WSD（warmup-stable-decay）学习率调度。

## 实验与结果
- **受控实验（Transformer）**：在 $\eta=0.007$ 固定学习率下，有效曲率 $s^{\mathrm{M}}$ 逼近并振荡于条件边界 $2\rho/\eta$，验证了 T1 边界的存活；增大 batch size（64→512→full）使 $c^{\mathrm{pol}}$ 趋向 -1，小 batch 保持近正交。
- **130M LLM 实验**（40 tokens/parameter，batch=32，peak LR 0.02/0.04/0.08/0.12）：
  - 条件曲率反复接近边界 $2\hat{\rho}_{k,32}/\eta_k$（损失中性追踪）；
  - 全局方向余弦中位数范围 [-0.048, -0.032]，远未达到 -1（弱负对齐）；
  - 最终验证 loss：peak LR 0.02 → 3.35085，0.04 → 3.32335，0.08 → 3.31944，0.12 → 3.33932（最佳为 0.08）；
  - 层间差异：中间层 key/value 矩阵的负对齐程度高于 query 矩阵。
- **学习率扰动实验（22M & 130M）**：在固定 checkpoint 改变 LR 后，损失平衡和时序对齐响应不同——降低 LR 使条件步处于 T1 损失下降侧，加倍 LR 使其处于损失上升侧，但采样时序余弦在所有分支均保持负值。
- **1B LLM 实验**（batch=512，20B tokens）：全局 $c^{\mathrm{pol}}$ 波动更大、更负，但仍远未到达 -1；层间余弦各异，单一全局值无法描述所有层。

**最强结果**：130M LLM 在 peak LR=0.08 时达到最低验证 loss 3.31944；条件损失中性边界在全训练过程中被持续追踪，同时训练持续改进。

## 相关工作脉络
1. **EoS 起源与机制**：Wu et al. (2018)、Jastrzebski et al. (2020)、Cohen et al. (2021) 建立大学习率与 EoS 的联系；Litman (2026) 推导 EoS 耦合的起源。本文定位为将该图景扩展至 Muon。
2. **随机/自适应 EoS**：Lee & Jang (2023) 将 EoS 扩展至 SGD；Cai et al. (2026)、Kalra et al. (2026) 在 Adam 上研究 LLM 预训练的 EoS 特征。本文指出 Muon 的极性归一化使批到总体对齐更复杂，需单独研究。
3. **Muon 理论**：Jordan et al. (2024) 提出 Muon；Bernstein & Newhouse (2024) 给出无动量核心的谱范数最速下降解释；Chen et al. (2026) 关联 Muon 与解耦权重的谱范数约束。本文聚焦极归一化本身引入的动态，而非其他工程变体。
4. **非欧几里得 EoS**：Islamov et al. (2026) 将 EoS 诊断扩展至任意范数（包括 Spectral GD），但仅限全批实验。本文推导随机版本并验证于 LLM。
5. **Muon 方向曲率分析**：Wang et al. (2026) 分析 Muon 优于 Adam 的机制，分解层内/跨层曲率惩罚。本文补充时序对齐视角，揭示层间异质性。

## 局限性与未来方向
- **仅研究无动量 Muon**：实际 Muon 包含动量，动量引入时序滤波与额外优化器状态，当前 T1/T2 诊断需重新考量耦合参数-动量动态。
- **未覆盖所有 Muon 变体**：论文明确说明结果刻画的是极归一化基础动态，未包含 scaling、weight decay 等工程变体的影响。
- **单次 seed 实验**：主要 LLM 实验为单 seed，统计显著性有限。
- **自适应调度未实现**：虽然指出 T1 和 T2 可联合指导 LR/batch schedule 设计，但转化为优化自适应调度的工作留待未来。
- **固定 checkpoint 干预的有限时间窗口**：扰动实验仅观察短期响应，长期恢复行为未完全揭示。

## 研究启发与可借鉴点
1. **双信号监测框架**：T1（损失中性边界）和 T2（时序对齐）作为独立诊断量可联合监控，为 Muon 调参提供可解释信号，而非仅依赖 loss 单调性。
2. **相干校正边界 $2\rho/\eta$ 的可迁移性**：该边界形式简洁，适用于任何基于极性归一化的优化器分析，可作为新优化器 EoS 研究的基准模板。
3. **实验设计借鉴**：使用 conditional probe（冻结状态下的虚拟更新评估）分离随机性效应与训练动态效应，是区分"高维正交通假象"与"真实反馈"的有效方法。
4. **层间异质性测量**：模块级/层级的 directional cosine 追踪比全局聚合值更能揭示优化器的真实行为，建议后续工作采用类似分层诊断。
5. **LLM 规模验证策略**：从 MLP/CNN/Transformer 小模型到 22M/130M/1B LLM 的渐进验证，兼顾理论可验证性与实际应用价值，可作为大模型优化器分析的标准范式。

## 关键术语表
- **Edge of Stability (EoS)**：梯度下降训练中，曲率逼近学习率相关边界 $2/\eta$，短视损失非单调但长视仍下降的动态现象。
- **Muon 优化器**：将每个矩阵梯度替换为近似半正交极方向（SVD 的 $UV^\top$）的优化器，改变更新几何而非仅尺度。
- **T1 诊断（条件损失中性边界）**：$s_{k,b}^{\mathrm{M}} = 2\rho_{k,b}/\eta_k$，衡量期望单步损失变化为零的曲率阈值。
- **T2 诊断（时序正交边界）**：$c_{k,b}^{\mathrm{pol}} = 0$，区分连续更新方向锐角/钝角对齐，$-1$ 表示精确反向。
- **相干性 $\rho_b$**：批次极方向与总体梯度核范数的对齐比例，度量小批量采样的下降保留度。
- **分裂的 EoS**：Muon 下损失中性边界与时序方向诊断解耦，不再共享同一临界点。
- **WSD 调度**：Warmup-Stable-Decay 学习率调度，包含预热、恒定、衰减三个阶段。
- **NS-5**：Newton-Schulz 迭代 5 次，用于近似矩阵极分解的工程实现。

## 可复现要素
- **数据集**：CIFAR-10（MLP/CNN）、TinyShakespeare（小 Transformer）、FineWeb（LLM，包括 sample-10BT 和 sample-100BT 子集）；论文未提及第三方数据集下载链接，数据需从 FineWeb 官方获取。
- **代码**：开源，GitHub: https://github.com/cyzebra/Muon-Sublates-the-Edge-of-Stability-in-LLM-Pretraining
- **权重**：论文未提及模型权重开源。
- **关键超参**：
  - LLM：序列长度 4096，batch size 32（130M）/512（1B），Muon peak LR 0.02–0.12，AdamW peak LR 0.001，WSD 调度（warmup 5%，decay 20%，最终 LR 10%）。
  - 小模型：$\eta=0.007$，batch sizes 1/16/64/128/512/4096/full。
  - 精度：bfloat16 autocast，float32 参数，NS-5 使用 float32。
  - 硬件：NVIDIA A100-PCIE-40GB（LLM）、RTX 4090（小模型）。
