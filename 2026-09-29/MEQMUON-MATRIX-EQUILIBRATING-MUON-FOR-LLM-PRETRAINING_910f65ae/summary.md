---
title: "MEQMUON-MATRIX-EQUILIBRATING-MUON-FOR-LLM-PRETRAINING"
source: https://arxiv.org/pdf/2609.35701v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-02 23:33:04"
field: "大规模语言模型预训练优化"
keywords: ["大语言模型", "预训练优化器", "Muon", "矩阵均衡", "自适应归一化", "低内存优化器"]
innovations: ["提出 MeqMuon，按行列 CV 自适应选择单侧/双边归一化以均衡矩阵更新幅度", "用归一化动量替代 AdamW 二阶矩，消除 second-moment 存储", "系统刻画 Muon 正交化更新的行列不均衡模式并给出机制解释"]
benchmarks: ["Llama-60M/130M/350M", "SmolLM2-135M/360M", "Qwen2-0.5B", "English C4"]
---

# 论文速读：MEQMUON-MATRIX-EQUILIBRATING-MUON-FOR-LLM-PRETRAINING

## 一句话总结
论文提出了 MeqMuon（Matrix-Equilibrating Muon），一种面向 LLM 预训练的改进 Muon 优化器，通过自动适应矩阵行/列方向的不均衡模式进行归一化，并同时消除了 AdamW 的二阶矩存储，在多项实验中实现了更优的收敛性能与更低的内存开销。

## 研究问题与动机
1. **LLM 规模持续膨胀导致预训练成本剧增**：DeepSeek V4-Pro 达 1.6T 参数、33T tokens，Kimi K3 达 2.8T 参数，高效优化器设计成为关键。
2. **Muon 正交化更新存在残留的行列不均衡**：尽管 Muon 已通过正交化将奇异值拉向 1，但矩形矩阵的正交化仅约束一侧维度，导致某些模块（如 q_proj/gate_proj）行方向不均衡更显著，而 o_proj/down_proj 则列方向不均衡更显著。
3. **已有行向归一化方案（NorMuon）覆盖面不足**：NorMuon 仅做行向归一化，无法处理列方向更严重不均衡的情形。
4. **Muon 对剩余参数仍依赖 AdamW，带来额外的二阶矩存储开销**：token embedding、LM head 等非隐藏层 2D 参数以及所有 1D 参数继续使用 AdamW，需额外维护 second-moment 状态。

## 核心贡献（创新点）
1. **系统性刻画了 Muon 更新中行列不均衡模式**：发现正交化更新在方形矩阵上两侧 CV 可先后翻转，在矩形矩阵上一侧趋零另一侧保留；未正交化动量矩阵则在两个方向均存在不均衡——这与 NorMuon 仅关注行向不均衡的观察形成本质区别。
2. **提出自适应单侧归一化（MeqMuon 对隐藏层 2D 权重）**：对正交化后更新矩阵 $U_t$ 比较行/列 CV，按 $\gamma_r \geq \gamma_c$ 时选行归一化、否则选列归一化，实现"按需选择"——区别于 NorMuon 固定行向归一化的设计。
3. **提出双边归一化替代 AdamW（针对剩余 2D 参数）**：对 embed_tokens、lm_head 等未正交化动量 $B_t$，施加 $\mathcal{N}_c(\mathcal{N}_r(B_t))$ 双边归一化，既平衡行列又无需二阶矩估计——这是消除 AdamW 状态的核心机制。
4. **全局 RMS 归一化用于 1D 参数**：对 bias、RMSNorm 权重等 1D 向量采用 $B_t / (\|B_t\|_2 / \sqrt{d})$ 的全局归一化，配合解耦权重衰减，统一纳入同一框架。
5. **内存节省与收敛提升的双赢实证**：在 Llama/SmolLM2/Qwen2 系列上全面超越 AdamW、Muon、NorMuon、SCALE；以 Qwen2-0.5B 为例，优化器状态内存降低 21.6%，且 PPL 最低。

## 方法详解
- **不均衡度量**：对矩阵 $X \in \mathbb{R}^{m \times n}$，定义行/列 RMS 向量 $r_i = \|X_{i,:}\|_2 / \sqrt{n}$、$c_j = \|X_{:,j}\|_2 / \sqrt{m}$，并以系数变异（CV）$\gamma_r$、$\gamma_c$ 量化不均衡程度（剔除 RMS $\leq 10^{-7}$ 的零行/列）。
- **动量缓冲**：每个参数维护单一动量缓冲 $B_t = \mu B_{t-1} + G_t$，$\mu = 0.95$。
- **隐藏层 2D 权重（自适应单侧归一化）**：
  - 先做 K=5 Newton–Schulz 迭代近似正交化 $U_t = Orth(B_t)$，系数 $(a,b,c)=(3.4445, -4.7750, 2.0315)$。
  - 再按 $\gamma_r(U_t)$ 与 $\gamma_c(U_t)$ 大小对比，自适应选择 $\widetilde{U}_t = \mathcal{N}_r(U_t)$ 或 $\mathcal{N}_c(U_t)$，每步每矩阵独立决策。
- **剩余 2D 参数（双边归一化，替代 AdamW）**：
  - 不对 $B_t$ 做正交化，直接施加 $\widetilde{U}_t = \mathcal{N}_c(\mathcal{N}_r(B_t))$，同时平衡行/列。
- **1D 参数（全局 RMS 归一化）**：
  - $\widetilde{U}_t = B_t / (\|B_t\|_2 / \sqrt{d})$。
- **统一更新公式**：
  - $W_{t+1} = (1 - \eta_t \lambda) W_t - \rho \eta_t \widetilde{U}_t$，其中 $\lambda = 0.1$、$\rho = 0.2$；去掉了 Muon 原版中形状补偿因子 $\sqrt{\max(m,n)}$。
- **学习率策略**：前 5% 线性 warmup，随后 cosine decay。

## 实验与结果
- **数据集与模型**：English C4，从随机初始化训练 Llama（60M/130M/350M）、SmolLM2（135M/360M）、Qwen2（0.5B）；Chinchilla-optimal 规则下 token 预算 = 20 × 参数量；seq_len=1024，global batch=512，BF16 AMP，DDP。
- **评估基线**：AdamW、SCALE、Muon、NorMuon。
- **主要结果（最终验证 PPL，越低越好）**：
  - Llama-60M：MeqMuon 29.53 vs. Muon 29.88、NorMuon 29.72、AdamW 37.40。
  - Llama-130M：21.38 vs. 21.64/21.57/24.46。
  - Llama-350M：15.94 vs. 16.04/15.98/17.02。
  - SmolLM2-135M：22.80 vs. 22.81/22.83。
  - SmolLM2-360M：17.16 vs. 17.28/17.20。
  - Qwen2-0.5B：18.63 vs. 18.79/18.76/19.65。
  - **最强结果**：MeqMuon 在所有模型规模下取得最低 PPL，相对 Muon 最大绝对提升约 0.27（Llama-130M）。
- **内存开销（优化器状态，MiB）**：MeqMuon 显著更低，Qwen2-0.5B 降至 1884.59 MiB，较 Muon/NorMuon（~2404 MiB）节省约 21.6%。
- **消融**：仅改动隐藏层 2D / 仅改动剩余 2D / 仅改动 1D 均可部分改善 PPL 并节省对应内存；三者叠加的 MeqMuon 达到最优 PPL 与最低内存。
- **归一化方向统计**：tall 矩阵（gate_proj/up_proj）几乎 100% 选择行归一化；wide 矩阵（down_proj）几乎 100% 选择列归一化，与理论预期吻合。

## 相关工作脉络
1. **AdamW（Loshchilov & Hutter, 2019）**：LLM 预训练主流基线；MeqMuon 用归一化动量替代其二阶矩估计，消除 $v_t$ 存储。
2. **Muon（Jordan et al., 2024; Liu et al., 2025）**：通过牛顿-舒尔茨正交化更新隐藏层 2D 权重；MeqMuon 在其基础上引入行列自适应归一化并移除形状补偿因子。
3. **NorMuon（Li et al., 2026）**：对 Muon 正交化更新施加 Adam-style 行向归一化；MeqMuon 发现其无法处理列向更严重的不均衡，扩展为自适应单侧/双边归一化。
4. **SCALE（Glentis et al., 2026）**：按输出维度归一化更新向量；实验仅适配 Llama（因嵌入/LM head 处理差异），MeqMuon 方法更具通用性。
5. **Adam-mini / Apollo / GaLore 等低内存优化器**：侧重点在降存储或降秩，MeqMuon 则从"矩阵不均衡结构"切入，通过归一化方式同时实现降存与提速。
6. **深层理论**：Newton–Schulz 迭代使矩形矩阵正交化仅约束一侧（$Y^\top Y = I_n$ 或 $YY^\top = I_m$），本文据此解释为何需额外归一化——这一机理分析是 NorMuon 所缺失的。

## 局限性与未来方向
- **NS 迭代近似误差**：K=5 次迭代只能逼近正交化，残留的行列 CV 需依赖后续归一化补救；更多迭代可能进一步降低开销但增加计算。
- **仅评测小/中型 LLM**：实验覆盖 60M–360M 量级，尚未在 B 级及以上模型上验证扩展性。
- **未测试更长上下文或不同数据配比**：C4 英文单一语料、固定 1024 序列长度，泛化性待检验。
- **Q/KV head 非对称结构（如 GQA）下的行为**：论文提到 SmolLM2/Qwen2 用 GQA，但未单独剖析此类架构对 CV 选择策略的影响。
- **自适应阈值敏感性**：当前以 $\gamma_r$ vs. $\gamma_c$ 作硬性判断，未讨论边界情形或引入平滑切换的潜在收益。

## 研究启发与可借鉴点
1. **用 CV 刻画矩阵方向不均衡**：将行/列 RMS 的变异系数作为诊断工具，可迁移到其它基于矩阵变换的优化器或梯度压缩方案中。
2. **自适应单侧归一化思想**：在需要正交化/归一化的矩阵更新中，先度量后再选择轴向，避免"一刀切"，设计范式可复用到其它二阶结构优化器。
3. **用归一化动量替代 AdamW 二阶矩**：在不需要逐坐标精细自适应的场景，双边归一化可同步达成方向均衡与内存压缩，为低内存优化器设计提供新路径。
4. **形状补偿因子的去除**：当更新经归一化后 RMS 已形状无关，可简化 Muon 类算法的实现并减少调参（$\sqrt{\max(m,n)}$、$\rho$ 耦合问题）。
5. **模块化消融设计**：将改进拆解为三类参数的独立组件并分别评估，清晰归因，值得在其它优化器改进工作中沿用。

## 关键术语表
**Muon**：一种针对神经网络隐藏层 2D 权重的优化器，通过对动量矩阵做近似正交化（Newton–Schulz 迭代）使更新的主奇异值趋近 1。
**NorMuon**：在 Muon 基础上对正交化更新施加 Adam-style 行向归一化，缓解行幅度不均衡的改进版本。
**RMS（Root Mean Square）归一化**：按行或列分别除以其 RMS，将非零行/列缩放到单位 RMS。
**CV（Coefficient of Variation）**：标准差与均值之比，用于无量纲度量矩阵行/列 RMS 向量的不均衡程度。
**Newton–Schulz（NS）迭代**：用于近似矩阵正交化的多项式迭代，本文使用 5 步五次迭代。
**解耦权重衰减（Decoupled Weight Decay）**：将权重衰减项与梯度更新解耦，即 AdamW 中的 $(1 - \eta_t \lambda)\theta_t$ 形式。
**RMS alignment（$\rho$）**：Muon 中与目标更新 RMS 匹配的缩放系数，MeqMuon 因归一化已形状无关而保留该缩放但不需 $\sqrt{\max(m,n)}$ 补偿。
**GQA（Grouped-Query Attention）**：多查询头的泛化，每组多个 query 头共享一个 key/value 头，影响投影矩阵形状。

## 可复现要素
- **数据集**：English C4，论文未声明公开链接（C4 本身可公开获取）。
- **代码/权重**：论文未提及代码与权重是否开源。
- **关键超参**：seq_len=1024，global batch=512，weight decay $\lambda=0.1$，momentum $\mu=0.95$，$\beta_1=0.9$、$\beta_2=0.95$（AdamW 部分），NS 迭代 K=5、$(a,b,c)=(3.4445,-4.7750,2.0315)$，$\rho=0.2$，warmup 5% 线性 + cosine decay，BF16 AMP，DDP，torch.compile，Flash attention。
- **硬件**：NVIDIA RTX A6000。
- **实现栈**：PyTorch 2.6.0、CUDA 12.4、Transformers 4.57.6。
