---
title: "T1-Terminal-Agent-Reinforcement-Learning-for-Long-Horizon-Ta"
source: https://arxiv.org/pdf/2609.11042v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:38:05"
---

# 论文速读：T1-Terminal-Agent-Reinforcement-Learning-for-Long-Horizon-Ta

## 一句话总结
本文提出 T1，一个基于 Qwen3.5-122B-A10B 的 122B MoE 终端智能体，通过在真实云沙箱中执行长达 300+ tool-call 的命令行任务并进行纯强化学习训练。论文同时提供了一套解决稀疏 MoE 模型训练-推理不匹配与长 horizon 稀疏奖励的完整训练方案，在 Terminal-Bench 2.1 上将初始模型从 43.8% 提升至 64.0%，超越 GPT-5.4 与 DeepSeek-V4-Flash，并以 10B 激活参数跻身同尺寸第一梯队。

## 研究问题与动机
1. **长 horizon 终端任务的执行挑战性**：终端环境将抽象规划绑定到不可逆的系统副作用上，要求模型具备环境理解、任务分解与渐进式故障恢复能力，远超普通 reasoning benchmark。
2. **稀疏 MoE 模型的训练-推理不一致**：推理端（SGLang）与训练端（Megatron）的 kernel、归约顺序与并行策略不同，微小的数值差异会通过离散的 TopK router 导致选中的专家子网发生跳变；多轮 agent harness 的 re-tokenization 又会引入 token ID 漂移，使梯度流向从未生成该轨迹的策略。
3. **二值奖励信号匮乏**：一次 rollout batch 消耗数百 sandbox-hours，但二值通过/失败仅提供 1 bit 反馈；早期实验表明纯二值奖励下 RL 无法超越 SFT 基线。
4. **基准过拟合风险**：现有工作多在测试集分布内训练，难以区分“真实能力迁移”与“benchmark 记忆/捷径”，需完全 OOD 的训练语料验证泛化性。

## 核心贡献（创新点）
1. **T1 模型与大规模终端 RL 训练链**：首个在 122B MoE 终端智能体上实现纯 RL 后训练的工作，单任务支持 300+ 工具调用轮次；与既有工作区别在于训练语料完全独立于 Terminal-Bench 2.1，且以 10B 激活参数在同等 harness 下超越参数规模大一个数量级的闭源模型。
2. **TITO + R³ 训练-推理对齐栈**：提出 Token-In-Token-Out 保证多轮轨迹缝合处的 token ID 精确一致，以及 Rollout Routing Replay 强制训练端复用推理端的 expert mask；与已有工作（重加权、掩码、目标函数重塑）的本质区别是直接从源头消除 bookkeeping 误差而非事后修正，将 train-inference log-prob 差距从 0.021 降至 0.013，且在损失区域内实现零漂移。
3. **密集验证过程奖励设计**：将 verifier 的 per-assertion 通过数转化为固定全局尺度（S=20）的连续奖励，配合 Critic Warm-Up 与异步 actor-critic 调度；与既有工作的本质区别是放弃二值最终结果，使 partial progress 可观测并可在长 horizon 内进行时间信用分配。

## 方法详解
- **整体架构**：基于 slime v0.3.0 构建异步流水线，step t 训练与 step t+1 生成在独立加速器池上并行；推理引擎为 SGLang，训练后端为 Megatron-Core v0.16.0rc0 + Transformer Engine v2.10.0，沙箱由 Harbor v0.7.0 + Daytona v0.168.0 提供。
- **TITO (Token-In-Token-Out)**：针对 harness 在每轮重渲染历史导致的 `enc(dec(a_i)) ≠ a_i` 漂移，组装器按四级层级缝合：`STRICT`（精确前缀）→ `NORMALIZED`（在 97×17 网格内有限修剪）→ `RETOKENIZED`（文本级相等映射）→ `SPLIT`（开启新 chunk）。仅前两种允许进入 loss 区，验证显示训练区内 token 漂移率为 `0.0000%`。
- **R³ (Rollout Routing Replay)**：推理时每 token 记录 $\mathbb{I}_j^{\mathrm{r},\ell} = \mathrm{TopK}_k(s_j^\ell)$，存储代价仅 1.5 KiB/token（开销 <3%）。训练 forward pass 强制使用录制 mask 进行 softmax 归一化 $g_{j,e}^\ell = \exp(s_j^\ell)/\sum_{e'\in\mathbb{I}_j^{\mathrm{r},\ell}}\exp(s_j^{\ell,e'})$，仅对 gate 权重反传梯度，使梯度路径与生成时的子网完全对齐。
- **Critic Warm-Up 与调度**：每步先更新 critic 再更新 actor，学习率比为 30×（critic $1.5\times10^{-5}$，actor $1.0\times10^{-6}$）。先在 TMax-15k 上单独预热 critic 一个 epoch，使 EV 从冷启动的 -33.6 直接稳定至 0.71~0.86，再切换至 T1-15k 进行密集奖励训练。
- **密集验证奖励**：奖励公式 $r = P/S$（$S=20$，$P$ 为通过断言绝对数），全局固定尺度避免 per-batch 最大值漂移；若 per-assertion 记录缺失则回退到二值结果 $r=b$。奖励仅打给轨迹最后一个 response token，由 GAE ($\gamma=\lambda=1$) 向回分配信用。
- **训练超参与稳定化策略**：PPO clip $\varepsilon=0.2$，KL 项与 load-balancing 系数均设为 0（避免与 R³ 冲突）；Adam $\beta=(0.9,0.98)$，wd=0.1；batch=512 但过采样至 560 轨迹，接受最早完成的 512 个以压制重尾延迟；context length 84k，最大 turn 300+。

## 实验与结果
- **数据集**：TMax-15k（14,601 任务，仅二值验证器）、RST-38k（37,484 合成任务）、T1-15k（精选 15,000 任务，93% 含 per-assertion 记录，17 类合并分布，CLI 工程类占 67.9%）。训练集与 Terminal-Bench 2.1 完全 disjoint。
- **评估基准与 Harness**：统一使用 Terminus-2 + Daytona 沙箱（10 GiB，3600s agent 墙 / 900s verifier 墙，最多 60 turns，温度 0.1 / top-p 0.95 / top-k 20，每任务 3 次尝试）。主要基准为 Terminal-Bench 2.1（89 题）、Long-Horizon Terminal Bench（46 题，平均奖励）、Terminal-Bench Hard（100 题）。
- **核心数字**：
  - Terminal-Bench 2.1：Base 43.8% → RST-SFT 49.4% → **T1 64.0%**（RL 贡献 +14.6pp，相对提升 29.6%）。
  - Long-Horizon Terminal Bench：Base 18.9 → SFT 23.6 → **T1 27.9**（匹配 Gemini-3.1-Pro）。
  - Terminal-Bench Hard：**T1 38.0%**，超 DeepSeek-V4-Pro (36.0%)。
- **最强结果与提升幅度**：64.0% 在 122B 同尺寸模型中位列第一梯队，超 GPT-5.4 (54.8%) 与 DeepSeek-V4-Flash (56.9%)，仅差 Claude Opus 4.6 (63.8%) 两个百分点；RL 阶段贡献了从 base 到最终的 72.3% 绝对提升。
- **训练动态**：平均 turn 数从 10.4 增至 ~20.7 后 plateau；rollout 奖励均值从 0.25 升至 ~0.35 plateau；EV 在 critic warm-up 后全程保持正值，与 late-stage benchmark 跃升（step 70→110）同步。

## 相关工作脉络
1. **终端智能体数据合成**：RST [8]、TMax [6]、SETA [20]、Terminal-Universe [26] 等均探索任务自动生成；本文沿用 RST 递归合成路线，但引入 LLM 多维权重审计筛选出 T1-15k，并首次在该规模数据上验证 MoE RL 可行性。
2. **MoE RL 训练-推理不匹配修复**：IcePop [31]、KPop [4]、SAT [30]、GSPO/DPPO [17,33] 主要通过目标函数重加权或自适应掩码缓解；本文定位为“源头修复”，TITO+R³ 直接对齐 token 与路由，不与 objective 改动耦合。
3. **无 critic 组基线 (GRPO)**：本文在第 9.4 节系统论证 GRPO 在长 horizon 下的三大缺陷（组内方差退化、有效任务多样性骤降、group 闭合受重尾拖累），确立 PPO+learnable critic 为该场景的必要选择。
4. **密集/过程奖励设计**：既有工作多依赖二值通过或外部偏好模型；本文利用 verifier 自身的 per-assertion 结构构造固定尺度绝对计数奖励，使 partial progress 可微分观测，与外部 reward model 路线
