---
title: "Recurrent-Looped-Transformer"
source: https://arxiv.org/pdf/2610.07591v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:16:17"
field: "Transformer架构与长度泛化"
keywords: ["循环Transformer", "状态追踪", "长度外推", "反馈机制", "算法推理", "编码器-解码器分解"]
innovations: ["将Transformer层划分为并行因果编码器与循环解码器，通过每token最终状态反馈使有效计算路径随序列长度增长", "提出反馈间隔B作为并行度-精度的可调设计轴（RLT-1/RLT-2/RLT-0）", "系统揭示encoder-decoder深度分配因任务而异（奇偶校验偏好浅解码器，置换追踪偏好深解码器）"]
benchmarks: ["奇偶校验（parity）", "swap-based S5置换追踪", "标准S5状态追踪", "模5算术（无括号/有括号）", "加法"]
---

# 论文速读：Recurrent-Looped-Transformer

## 一句话总结
本文提出**循环Transformer（RLT）**，将Transformer层拆分为并行因果编码器和循环解码器，通过在每步将解码器最终状态反馈到下一步输入，使计算路径随序列长度增长而保持固定每token开销；在六个算法状态追踪任务上，RLT在远超训练长度的泛化上大幅超越固定深度Transformer。

## 研究问题与动机
- **核心问题**：算法状态追踪（如奇偶校验累积、置换群状态组合）要求在每一个输入处更新状态；但Transformer对每个token应用固定层数，与序列长度无关，理论上有界深度Transformer无法追踪任意长度 $S_5$ 置换的组合（Merrill et al., 2024，基于 $\mathsf{TC}^0 \neq \mathsf{NC}^1$ 的复杂度论断）。
- **现有方法不足**：
  - 标准Decoder-only Transformer缺乏跨token的状态传递机制，长度外推能力差；
  - 已有反馈Transformer类工作（Feedback Transformer、Recurrent Transformer等）的反馈路径或限于层内，或引入额外隐式状态，未系统探索编码器-解码器深度分配与反馈频率这两个设计轴；
  - 块级循环Transformer（Block-Recurrent Transformer）以token块为单位更新状态，无法实现每token精度的反馈。

## 核心贡献（创新点）
1. **RLT架构**：将总层数在并行因果编码器与循环解码器之间划分，每token通过门控合并将当前编码器表征与前一token解码器最终状态融合，再通过完整解码器处理——与已有反馈方法相比，反馈路径贯穿整个解码器堆栈，且与编码器全局KV记忆严格分离。
2. **长度外推突破**：仅训练于最多40 bit的奇偶校验，RLT（$5+3$ 和 $7+1$ 划分）在256 bit处达到 $100\pm0\%$ 准确率（每个seed），而8层Decoder-only Transformer仅 $50.07\pm1.63\%$（接近随机）；swap-based $S_5$ 在256次操作（训练长度8倍）处，$4+4$ 划分达 $97.30\pm2.76\%$，Transformer仅 $0.85\pm0.30\%$。
3. **深度分配的系统分析**：发现最优编码器-解码器划分取决于任务类型——奇偶校验最佳划分为1或3个解码器层，而swap-$S_5$ 准确率随解码器深度单调上升（从1层到4层从随机提升至97%）。
4. **反馈间隔（B）的设计轴**：提出RLT-2（每chunk的B个token共享一次反馈状态）以换取已知token的并行度，揭示"并行度—精度"权衡：chunk4在64-bit奇偶校验上保持 $98.99\pm1.66\%$，但swap-$S_5$ 从100%骤降至 $19.60\pm6.54\%$，证明置换追踪需要每token精确反馈。
5. **严谨的对照实验体系**：五个RLT-1划分 + 一个8层Transformer在六个算法任务、三个随机seed、最大256/512长度的测试集上进行系统比较，消融实验证实去除反馈后所有增益消失。

## 方法详解
**模型构成**：总层数 $L = L_E + L_D$，残差宽度 $d=512$，4个注意力头，FFN宽度1365。参数约26–29M（略多于Transformer的25.31M）。

**因果编码器与全局KV记忆**：对观测前缀 $x_{1:T}$ 并行计算因果编码器表征 $e_{1:T} = E_\theta(x_{1:T})$，并为每个decoder记忆组 $g$ 构造键值投影：
- $k_t^g = \mathcal{P}_K^g(e_t, t)$，$v_t^g = W_V^g\,\text{RMSNorm}_E(e_t)$
- 全局记忆 $M_{\leq t}^g = \{(k_j^g, v_j^g)\}_{j=1}^t$ 供所有decoder层跨注意力读取

**每token转移函数**：初始化 $H_0 = (s_\star, \emptyset)$，对每个token $t\geq 1$：
1. **门控合并**：$r_{t-1} = \text{RMSNorm}_s(s_{t-1})$，$g_t = \sigma(W_g[e_t; r_{t-1}] + b_g)$，$u_t = e_t + \alpha\, g_t \odot W_s\, r_{t-1}$（默认 $\alpha=0.1$）
2. **解码器堆栈**：每个decoder层依次执行因果SWA → 编码器记忆交叉注意力 → FFN（均含残差连接）
3. **输出**：$s_t = z_t^{L_D}$ 作为下一步反馈，$p_\Theta(x_{t+1}|x_{1:t}) = \text{softmax}(W_o\,\text{RMSNorm}_o(s_t))$

**完整解码器状态**：$H_t = (s_t, C_t^D)$，其中 $C_t^D$ 为每层SWA层的键值缓存（窗口大小 $W\geq 1$，默认 $W=8$）。

**三种反馈变体**（核心区别在反馈间隔 $B$）：
- **RLT-1（$B=1$）**：每token更新反馈状态 $s_t$，对长度为 $T$ 的前缀需 $T \cdot L_D$ 个串行decoder块阶段；
- **RLT-2（$B=4$，chunk4）**：每chunk的 $B$ 个token共享同一反馈边界状态 $h_{k-1}$，仅chunk边界更新 $h_k = y_{kB}$，将串行阶段降为 $\lceil T/B\rceil \cdot L_D$；
- **RLT-0（$B=\infty$）**：移除门控合并和反馈路径，直接 $z_t^0 = e_t$，退回纯编码器-解码器架构。

**训练**：teacher forcing + 交叉熵，完整BPTT（TBPTT128截断）；编码器、记忆投影、合并门和decoder联合训练。RL支持当前策略精确replay重建状态。

**计算复杂度**：prefill代价 $O((L_E+L_D)(Td^2+T^2d)+GTrm(d^2)+L_D T\min(W,T)d)$；每生成token评估 $L_E+L_D$ 个block；推理缓存 $O((L_E+G)t d_{\text{KV}}+L_D\min(t,W{-}1)d_{\text{KV}}^D+d)$。

## 实验与结果
**数据集与任务**：六个算法任务——加法（1–8位数字）、奇偶校验（3–40 bit）、无括号模5算术（3–39奇数长度）、有括号模5算术（3–40长度）、标准 $S_5$ 状态追踪（32次置换）、swap-based $S_5$（identity+10个对换，32次操作）。测试长度远超训练范围（256/512）。

**评估设置**：三个seed（42/43/44），共享训练/测试数据；global batch=512，microbatch=32；AdamW $(\beta_1,\beta_2)=(0.9,0.95)$，weight decay=0.1，LR warmup 200步至 $10^{-4}$ 后余弦衰减至 $5\times10^{-6}$；4+4至7+1划分训练2000步，mod-5训练5000步；选ID验证loss最低的checkpoint。

**主要结果（长度外推）**：
| 任务（测试长度） | RLT-1 最佳划分 | Transformer 8 | 提升幅度 |
|---|---|---|---|
| 奇偶校验（256 bit） | $100\pm0\%$（5+3, 7+1） | $50.07\pm1.63\%$ | ~50pp |
| swap-$S_5$（256 ops） | $97.30\pm2.76\%$（4+4） | $0.85\pm0.30\%$ | ~96pp |
| swap-$S_5$（512 ops） | $55.70\pm25.78\%$（4+4） | $0.85\pm0.30\%$ | ~55pp |
| 无括号mod-5（63） | $93.36\pm5.69\%$（6+2，5000步） | $33.20\pm2.33\%$ | ~60pp |
| 无括号mod-5（255） | $18–21\%$ | $19–20\%$ | 接近 |

**反馈间隔消融（4+4划分，64长度）**：
- 奇偶校验：RLT-1 $100\%$，chunk4 $98.99\pm1.66\%$，RLT-0 $50.23\pm3.80\%$
- swap-$S_5$：RLT-1 $100\%$，chunk4 $19.60\pm6.54\%$，RLT-0 接近均匀参考（$1/120$）
- 结论：置换追踪对每token反馈极度敏感

**标准 $S_5$（全部120种置换）**：所有模型均在均匀参考（0.83%）附近，说明当前设置对该难度任务仍有局限。

**加法**：训练宽度（1–8位）达到 $100\%$，但超出后准确率随位数下降（32位时约15%），各模型差异不大。

## 相关工作脉络
- **Feedback Transformer**（Fan et al., 2020）：对每token跨层表示做加权求和形成共享记忆，后续token在各层可attend——与RLT的区别在于RLT的反馈路径贯穿整个decoder堆栈并独立于编码器全局KV。
- **Recurrent Transformer**（Oncescu et al., 2026）：每层从其自身输出构造持久KV，提供精确tiling调度——RLT的循环依赖跨越完整decoder而非单层层内。
- **Full-bandwidth Transformer**（Wang et al., 2026）：通过GLU将上一token顶层隐藏状态与下一token embedding结合——RLT将output-to-input反馈置于encoder-decoder接口并顺序重放decoder历史。
- **T²MLR**（Cai et al., 2026）：将前一token中层缓存表征反馈至当前token更早层——RLT反馈贯穿全decoder并使用完整BPTT。
- **Latent Recurrent Transformer**（Huang et al., 2026）：通过KV投影和残差注入复用前一token高层状态，保持decoder-only结构——RLT分离编码器记忆与循环decoder，支持精确当前策略replay。
- **Block-Recurrent Transformer**（Hutchins et al., 2022）：每个recurrent step并行处理一个token块——RLT-2最接近此范式，但以chunk边界而非layerwise方式更新反馈，不引入专用memory token。
- **Universal Transformer / DeepLoop**：在每个position上沿深度方向重复参数访问——RLT沿token方向递归，encoder-decoder权重共享为可选项。

## 局限性与未来方向
- **标准 $S_5$ 完全失效**：所有模型（包括RLT和Transformer）final-state准确率均接近均匀参考（~0.8%），表明当前方法对全置换群追踪仍无力；
- **长距算术退化**：mod-5在长度255处所有模型回落至~20%（均匀参考），flat mod-5存在严重seed间方差；
- **加法外推瓶颈**：超过训练宽度后准确度迅速下降（32位仅~15%），未见显著优势；
- **理论支撑待补**：虽有复杂性论断（Merrill et al.）指出固定深度Transformer的理论局限，但RLT在何种计算复杂度类（如$TC^0$ vs $NC^1$）下能达到何种泛化上界，尚缺形式化分析；
- **未在真实语言任务上验证**：当前全为算法合成任务，向自然语言预训练的迁移效应未知；
- **反馈规模$\alpha$优化空间**：ablation显示降低$\alpha$并未系统改善各宽度加法准确率，但最优值可能因任务/划分而异。

## 研究启发与可借鉴点
1. **编码器-解码器深度分配作为显式设计轴**：同一总层数下，将层数分配给"并行编码"vs"循环解码"可以调控有效计算路径长度；不同任务的最优分配不同，这一原则可迁移至需要不同长度泛化的序列任务（如程序合成、长程推理）。
2. **门控合并机制（gated merge）**：$u_t = e_t + \alpha\, g_t \odot W_s r_{t-1}$ 以可学习门控控制并行输入（编码器特征）与循环记忆（前状态）的信息融合比例，结构简洁且有效，可借鉴为通用"并行-循环混合接口"。
3. **反馈间隔（chunk size）作为并行-精度调节旋钮**：RLT-2揭示 chunking 可显著提升已知token训练并行度（chunk4比RLT-1快2.27×，RLT-0快4.17–4.30×），但对需要精确顺序状态的任务有害；这提示在预训练阶段用大chunk、微调阶段用小chunk的"渐进收敛"策略是可行的工程实践。
4. **系统性的算法任务基准**：六个任务覆盖不同复杂度（加法→奇偶→模算术→置换追踪），能清晰分辨模型架构差异；这种"难度梯度+多指标"的实验设计值得在其他架构论文中借鉴。
5. **精确当前策略replay（exact policy replay）**：附录D详述RL中如何重放采样轨迹并重建当前参数下的完整状态历史，这为后续将RL应用于循环架构提供了可直接复用的方法论模板。

## 关键术语表
- **Recurrent Looped Transformer (RLT)**：将Transformer层划分为并行因果编码器与循环解码器、通过每token最终状态反馈实现随序列长度增长的有效计算深度的架构。
- **Gated Merge**：门控融合机制，以sigmoid门控控制当前token编码器表征与前一时步解码器隐藏状态之间的信息注入比例。
- **SWA（Sliding Window Attention）**：滑动窗口注意力，每decoder层仅attend到当前及最近$W-1$个位置的decoder键值，控制局部上下文窗口。
- **反馈间隔 $B$**：相邻两次decoder反馈状态更新的token间距，决定串行依赖深度与已知token并行度的权衡。
- **RLT-1 / RLT-2 / RLT-0**：分别对应 $B=1$（每token反馈）、$B=4$（每chunk反馈）、$B=\infty$（无反馈）三种变体。
- **ID / OOD**：In-Distribution（分布内，训练长度范围内）与 Out-of-Distribution（分布外，训练长度之外）泛化评估。
- **TBPTT**：Truncated Backpropagation Through Time，在BPTT中按固定步长截断梯度传播以控制显存。
- **Final-state accuracy**：状态追踪任务的评测指标，仅统计序列末尾状态的预测准确率，区分于prefix-token accuracy（中间所有状态的平均准确率）。

## 可复现要素
- **代码**：已开源，项目页面 https://github.com/yifanZhang-pro/recurrent-looped-tranformer；基于 nanogptpro-dev 分支实现（PR #164 含 RLT-0）。
- **数据**：算法任务数据seed为 20260914（训练）/ 20260915（验证）/ 20260916（测试）；加法数据seed为42；manifest.json 与 configs.json 记录运行元信息。
- **超参数**：总层数 $L=8$（$L_E+L_D$ 五划分），宽度 $d=512$，FFN宽度1365，4头注意力，SWA窗口 $W=8$，记忆组 $G=1$，反馈尺度 $\alpha=0.1$，AdamW $(0.9, 0.95)$，weight decay=0.1，LR warmup 200步至 $10^{-4}$ 后余弦衰减至 $5\times10^{-6}$，global batch 512，microbatch 32，TBPTT截断长度128。
- **硬件**：四线程CPU FP32执行；seed-42计时显示chunk4训练步骤比RLT-1快2.27×，RLT-0快4.17–4.30×。
- **Seed聚合**：所有报告结果取seed 42/43/44的均值±样本标准差（$n=3$）。
