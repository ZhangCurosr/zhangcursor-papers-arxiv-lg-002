---
title: "On-Trajectory-Aware-Training-for-Masked-Diffusion-Language-M"
source: https://arxiv.org/pdf/2609.37974v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-03 14:12:06"
field: "扩散语言模型训练与推理对齐"
keywords: ["masked diffusion language models", "trajectory-aware training", "progressive unmasking", "backpropagation through time", "BPTT", "instruction fine-tuning", "NFE efficiency"]
innovations: ["提出 PUMBA 统一框架，联合渐进去遮罩轨迹、连续隐状态 carry 与跨步 BPTT 三个设计轴", "发现并刻画小揭示数下的局部过拟合机制，给出缓解方案（大训练 u + 置信度阈值）", "理论证明 BPTT 窗口 W 越大逼近最优采样器越近（KL 余项上界），并在 8B 级 SFT 中验证性能-NFE 折中显著提升"]
benchmarks: ["GSM8K", "IFEval", "MBPP", "TinyGSM"]
---

# 论文速读：On-Trajectory-Aware-Training-for-Masked-Diffusion-Language-M

## 一句话总结
本文提出 PUMBA（Progressive UnMasking with Backpropagation Across steps），一种统一的结构感知训练框架，将渐进去遮罩、连续隐状态传递与跨步 BPTT 联合优化整合在一起，使掩码扩散语言模型（MDM）的训练轨迹与推理轨迹对齐；在小模型控制的 TinyGSM/GSM8K 实验中逼近同规模自回归模型的推理准确率，并在 LLaDA-8B 的 SFT 中显著改善了性能–NFE（函数评估次数）折中，全画布生成上较标准 SFT 最多节省 22% NFE，块扩散上最多节省 26%。

## 研究问题与动机
1. **训练-推理轨迹不对齐**：MDM 在随机独立掩码序列上训练，而推理时由去遮罩策略（policy）沿着模型自身预测形成的轨迹逐步揭示 token，导致训练分布与推理所见 mask 模式存在分布偏移。
2. **步骤间信息断裂**：每个去噪步只能看到当前已遮罩/已揭示的离散 token，无法获知前一步的计算结果； committing 一个 token 后该位置不再被重新访问，前一步的信息丢失。
3. **既有工作各自为政**：现有方法分别解决上述问题的不同侧面（如 LOopholing 传连续隐状态、PU 做轨迹对齐、Relay 做 BPTT），但缺乏对这三个设计维度的联合控制实验与系统分析。
4. **小 u 下的局部过拟合**：在推理常用的小每步揭示数 u 下，PU 因每条序列被重复使用 J ≈ L/u 次，模型迅速"记住"该序列在当前轨迹 mask 下的答案，后续梯度枯竭，导致训练-推理对齐反而损害泛化。

## 核心贡献（创新点）
1. **提出 PUMBA 统一框架，将轨迹对齐（PU）、连续 carry 与跨步 BPTT 三个轴整合为一个可微分训练过程**：区别于先前工作只取其一（如 PUMA 只做轨迹对齐、Loopholing 只做 self-conditioning carry），PUMBA 同时联合优化三者并做消融控制实验。
2. **发现并刻画小 u 下的"局部过拟合"现象**：通过插入验证序列、追踪完成 loss 的动态，揭示 PU 在小 u 下很快拟合住序列答案、之后梯度枯竭导致隐藏 token 重新退化；这是此前 PU 论文未深入解释的训练失败机制。
3. **证明连续 carry 显著优于各类离散梯度估计器**：在 W=2 的控制实验下，Carry 达到 51.2% GSM8K，远高于 REINFORCE（41.2%）、Gumbel-softmax（42.4%）和 straight-through（44.7%），且理论证明离散桥接无法提供通过 committed token 的梯度。
4. **给出 BPTT 窗口 W 越大逼近最优采样器越近的理论界**：Proposition 1 / Theorem 1 在 carry 为压缩映射（factor γ<1）与 PL 条件下，给出 KL 余项上界 C(J/W − 1)²，W=J 时消失；小模型实验也验证 W 从 2→4→8 单调提升（44.4→48.7→52.6→55.6%）。
5. **首次在 8B 参数级做多步骤轨迹感知 SFT（4096-token、1.8B token 通用指令数据、全参数训练）**：在 LLaDA-8B-Base 上，4000 步 Post-SFT 即比"把 SFT 预算翻倍"多 3 倍的性能增益；全画布匹配分数下省 22% NFE，块扩散省 26% NFE。

## 方法详解
- **基础 MDM 训练目标**：对线性 schedule α_t = 1−t，损失为加权交叉熵
  ℓ(x₀, x_t) = (1/t) Σ_{i∈M(x_t)} −log f_θ^i(x₀^i | x_t)，每个训练步独立采样 x₀ 与 t 后单做一次 denoiser 前向。
- **渐进去遮罩（PU）轨迹构建**：每条轨迹从全遮罩序列开始，按推理 policy（top-u，即每步选最自信 u 个 mask 位置）揭示位置；揭示位置做 teacher-forcing（填入 x₀^i 真实 token），保留梯度路径的确定性；轨迹持续 J ≈ L/u 步后丢弃，换新样本。
- **连续 carry 传递**：denoiser 额外接收上一步的隐状态 h_j ∈ ℝ^{L×d}（输出投影前的 last hidden state，d≪|V|），经 zero-init LayerNorm 加到 token embedding 上，输出新的 h_{j+1}：
  (f_θ(·|x_{t_j}, h_j), h_{j+1}) = f_carry(x_{t_j}, h_j)。
  Carry 是一种"弱 student forcing"——反映前一步不完全预测，而可见 token 仍与 x₀ 一致。
- **跨步 BPTT（窗口 W）**：对同一条轨迹的连续 W 步计算总损失并取平均，沿 carry 路径反向传播；窗口边界处对 carry 执行 stop-gradient（sg[h]），实现截断 BPTT。W=1 等价于只传 carry 不跨步传梯度。
- **训练策略的细节**：
  - TinyGSM：沿用 PUMA 原始 stage-indexed curriculum（K 从 12 递增至 42），τ_train=0.9 作为附加阈值。
  - LLaDA-8B SFT：两阶段——先以标准 MDM 目标做 SFT（全画布 1.5 epoch / 块扩散 6 epoch），再从该 checkpoint 进入 Post-SFT，以 4000 步 PUMBA 微调；Prompt 全程可见仅 Response+EoS 可被 mask；u=32/64（全画布）、u=8/16（块扩散）；训练 u 明显大于推理 u（2）以规避局部过拟合，但通过 τ=0.9 的置信度阈值约束让训练 mask 仍贴近推理分布。
- **理论支撑**（Appendix E）：
  - Proposition 2（似然恒等式）：在 teacher-forced 轨迹上，commit 位置的累加 cross-entropy 精确等于 H(p*) + KL(p*||p_θ)，即训练目标与真实采样分布之差仅为常数。
  - Proposition 3 / Theorem 1（截断偏差界）：若 carry 为 γ-压缩，则在 PL 条件下，truncated gradient g_W 与全轨迹梯度 ∇ℒ_∞ 的偏差 ≤ B(J/W−1)；驻点对应的 excess KL 以 (J/W−1)² 收敛。
  - Theorem 2（信用分配）：无 carry 时，后续步骤的 loss 对 θ 的全导数不含通过 committed token 的项；有 carry 且同窗口内则包含 G_{j,r}^σ 项，证明 carry 提供了可微分的跨步信用路径。

## 实验与结果
- **受控小模型实验（TinyGSM → GSM8K，125M 双向 transformer，20 epoch，EMA）**：
  - AR: 55.3%；MDM: 34.8±2.5%；MDM×2 epoch: 42.6%；MDM×8 epoch: 43.6%。
  - PU: 40.2±0.3%；PU+Carry(W=1): 44.4%；W=4: 48.7%；W=8: 52.6%；最终达 55.6%，**匹及 AR 最优 checkpoint**，且每步可并行揭示 >1 个 token。
  - Carry 对比（W=2）：PU+REINFORCE 41.2% / Gumbel-softmax 42.4% / Straight-through 44.7% / **PU+Carry 51.2%**。
  - Mask discrepancy D_mask（定义 1）：即使训练 u≈133 远大于推理 u=2，PU 仍使 D_mask 显著低于 MDM（Figure 3），说明对齐收益不因 u 失配完全消失。
- **LLaDA-8B SFT（Dolci-Instruct-SFT，2.15M 样本，4096-token canvas，64×H100/B200）**：
  - **全画布 Post-SFT（从 1.5 epoch SFT checkpoint 起，4000 步）**：
    - 最佳配置 u=32, W=8：score 63.5（相对起点 +2.6 点），**三倍于把 SFT 预算翻倍（1.5→3 epoch，仅 +0.7 点）的收益**；同分下较 3-epoch SFT **节省最多 22% NFE**。
    - Post-SFT MDM 控制仅 +0.3 点，表明额外 4000 步的收益来自 PUMBA 本身而非单纯训练时长。
  - **块扩散 Post-SFT（从 6 epoch SFT checkpoint 起，4000 步）**：
    - 最佳配置 u=16, W=2：score 63.6（+2.2 点），**较 Post-SFT MDM 同分下节省最多 26% NFE**；该配置在 4000 步内增益也超过把 SFT 从 3 epoch 延长至 6 epoch（+0.7 点）的效果。
  - **失败模式（u 过小）**：u=1 时全画布仅访问 MDM 控制 0.6% 的不同样本，训练 collapse；块扩散因 block 并行每 block 独立轨迹，u=1 只占 10% 样本但不会完全崩溃。
  - 评估指标（mean of IFEval 0-shot / GSM8K 8-shot / MBPP 3-shot）随 Fast-dLLM τ ∈ {0.7, 0.8, 0.9, 1} 扫 NFE 曲线给出完整 trade-off。

## 相关工作脉络
1. **PUMA（Kim et al., 2026）**：首次提出渐进去遮罩轨迹训练，但仅做 teacher-forced 单步交叉熵，无 carry 也无 BPTT；本文在其轨迹构建基础上加入连续隐状态通道与多步信用分配。
2. **Loopholing（Jo et al., 2026）**：提出 hidden-state carry 并通过 self-conditioning 额外一次前向获取，带来约 30% 训练耗时；本文利用 PU 固有的前一步状态直接缓存 carry，省去额外 pass，并结合 BPTT 可进一步训练 carry 产生过程。
3. **RCD / DiffusionGemma（Hu et al., 2026; DiffusionGemma Team, 2026）**：用概率加权输入 embedding E^⊤p 作为 carry；本文采用更简洁的 last hidden state carry 以隔离"轨迹对齐"的主效应，但指出 distribution-derived carry 是可探索方向。
4. **MetaState（Xia et al., 2026）**：冻结 backbone、引入固定大小工作记忆，结合 dense + reveal 双损失项；本文全参数训练、极简 carry，更关注 instruction-following 主能力与推理/代码保持的联合评估。
5. **Relay（Rozonoyer et al., 2026）**：并发工作，同样组合 teacher-forced rollout + per-token differentiable state + truncated BPTT，但在 Fast-dLLM v2（1.5B）上仅做代码/数学两轮实验、W 固定为 2；本文做到 8B 全参数 SFT、 sweep 多个 u 与 W，并给出系统的消融与失败分析。
6. **PAPL（Peng et al., 2026）**：对 planner 诱导轨迹做 loss 重加权；本文不做任务 reward 或蒸馏，直接在通用指令 SFT 内做轨迹感知训练。

## 局限性与未来方向
- **小 u 局部过拟合仍未根治**：必须把训练 u 设得显著大于推理 u（本文 u=32/64 vs 推理 u=2），牺牲了部分对齐精度；如何在大模型上实现更小 u 的稳健对齐仍开放。
- **Teacher forcing 的不可逆错误**：commit 错 token 后上下文与 x₀ 目标矛盾，本文尝试 scheduled sampling（student forcing）未能改善（Appendix C.3）；remasking 可能是一个出路但未探索。
- **Carry 设计极简**：仅用 last hidden state 作 carry，未尝试 MetaState 式的固定大小工作记忆或 distribution-derived embedding carry，可能在复杂推理任务上受限。
- **仅在置信度-based policy 上验证**：未评估 planner-based 或 learned unmasking policy 下的表现。
- **计算开销随 W 线性增长**：W=8 训练时间约为 MDM 的 8 倍（表 7：14.4h vs 1.7h），尚未结合 gradient checkpointing / selective gradient / adaptive window 等优化。
- **数据规模有限**：SFT 仅用 1.8B token（2.15M 样本），更大规模的 post-training 行为未知。

## 研究启发与可借鉴点
1. **"轨迹对齐 + 连续 carry + BPTT" 三轴统一框架可直接迁移至其他离散扩散模型（如图像 MaskGIT 变体、multimodal diffusion）**，只需替换 policy 与 carry 维度。
2. **局部过拟合的诊断范式（插入 held-out 序列追踪 completion loss 动态）**对任何序列生成模型（AR/MDM/flow）的课程学习或 curriculum 设计都有参考价值。
3. **Post-SFT 两阶段训练流程**（先用标准 MDM 目标做粗 SFT 收敛到目标分布，再用 PUMBA 做轨迹感知精调）是数据受限场景下突破 SFT 饱和的有效策略，可复用于其他 diffusion LM 的指令微调。
4. **带 carry 的 teacher-forced 轨迹避免了 student forcing 的不可逆错误**——carry 提供弱 student forcing 而不破坏可见 token 与 target 的一致性，这一设计思想可用于任何"离散 commit + 连续辅助状态"的混合生成架构。
5. **理论界（Theorem 1 的 KL 余项上界）为 BPTT window 选择提供了定量依据**，后续可在不同架构下验证 γ（carry 压缩率）的实测值，并据此设计自适应 W schedule。

## 关键术语表
- **MDM（Masked Diffusion Model）**：通过在离散 token 序列上做逐步遮罩/去遮罩的扩散过程建模语言分布的生成模型，训练目标为加权交叉熵。
- **PU（Progressive Unmasking）**：训练时沿推理 policy 诱导的轨迹逐步揭示 token，使训练 mask 分布逼近推理分布；揭示位置 teacher-forced 填入真实 token。
- **PUMBA（Progressive UnMasking with Backpropagation Across steps）**：本文提出的统一框架，集成 PU 轨迹、连续隐状态 carry 与截断 BPTT。
- **Carry**：上一步 denoiser 的输出隐状态（last hidden state，维度 d≪|V|），经 zero-init LayerNorm 加入当前步 token embedding，作为跨步连续信息通道。
- **BPTT window W**：单次参数更新所跨的连续去噪步数；梯度沿 carry 回传 W−1 步，窗口边界处 stop-gradient。
- **NFE（Number of Function Evaluations）**：生成一个样本所需 denoiser 前向调用次数，衡量推理效率的核心指标；越小越快。
- **Mask discrepancy D_mask**：用 MMD + 指数 Hamming kernel 度量训练与推理在给定遮蔽率 t 下的 mask 分布差异，用于诊断对齐程度。
- **局部过拟合**：PU 在小 u 下因同一序列被连续 J≈L/u 步复用，模型在批次内迅速拟合住该序列当前轨迹的答案，导致验证 loss 反弹、梯度枯竭。

## 可复现要素
- **数据集**：TinyGSM（公开，Liu et al., 2023）、GSM8K（公开）、Dolci-Instruct-SFT（来自 OLMo 3 post-training pipeline，Olmo Team, 2025；论文声明原始 LLaDA SFT 数据集不公开，改用 OLMo 3 的公开 SFT 数据集）。
- **代码/权重**：论文未明确声明开源；LLaDA-8B-Base 来自 Nie et al., 2025（公开基座）；Fast-dLLM 来自 Wu et al., 2026b（公开 policy）。
- **关键超参**：
  - TinyGSM：125M 双向 transformer（14 层，hidden 512，8 heads），AdamW lr=3e-4，warmup 1000，cosine decay，batch=256，20 epoch，EMA decay=0.9999，τ_train=0.9，W∈{1,2,4,8}，u=2（推理）/  curriculum K:12→42（训练）。
  - LLaDA-8B Post-SFT：bfloat16，peak lr=2.5e-6，warmup 200，cosine，global batch=128，4000 步；全画布 u∈{32,64}、W∈{1,2,4,8}；块扩散 u∈{8,16}、W∈{1,2,4}；carry 为最后 hidden state + zero-init LayerNorm，fp32 缓存，p=1（无 dropout）。
- **硬件**：TinyGSM ~36k H100 GPU-hours；LLaDA-8B ~49k H100 + 7.3k B200 GPU-hours；64×H100 或 64×B200。
