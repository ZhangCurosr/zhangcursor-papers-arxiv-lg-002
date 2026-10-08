---
title: "Learning-Explainable-Representations-of-Complex-Game-playing"
source: https://arxiv.org/pdf/2610.07638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 02:49:56"
field: "可解释强化学习与程序合成"
keywords: ["strategy synthesis", "explainable reinforcement learning", "decision transformer", "inductive logic programming", "programmatic policy", "chess tactics", "Karel domain"]
innovations: ["将Decision Transformer改造为支持离散DSL程序合成的序列建模方法，样本效率较LEAPS提升2-3个数量级", "首次用一阶逻辑Prolog规则建模国际象棋战术，并以覆盖率/分歧度作为可计算可解释性指标", "提出用XAI可解释性度量替代心智对齐量化评估的策略合成框架"]
benchmarks: ["Karel six-task benchmark (cleanHouse, fourCorners, harvester, randomMaze, stairClimber, topOff)", "Chess beginner game dataset (1297 position-move pairs)"]
---

# 论文速读：Learning-Explainable-Representations-of-Complex-Game-playing

## 一句话总结
本文提出两种可解释策略合成方法：基于一阶逻辑的棋类策略模型（ILP）与基于决策Transformer的程序化策略学习方法，前者在国际象棋上能较好逼近人类新手水平，后者在Karel环境中以更高样本效率匹配SOTA程序化策略学习算法（LEAPS）的性能。

## 研究问题与动机
- **核心问题**：如何让AI自动学习可被人类理解的游戏策略程序，从而帮助玩家改进"心智模型"（mental model），而不仅仅是生成最优但黑箱的决策。
- **现有方法不足**：已有工作多聚焦策略的计算学习效率，极少关注学习到的策略是否真正"可被玩家理解"；且策略可解释性缺乏可计算评估指标。
- **认知对齐缺口**：模型匹配理论（model matching theory）强调游戏模型与玩家心智模型的对齐，但如何量化这种对齐仍不明确。
- **可解释性衡量空白**：XAI领域已有指标（如认知块数量、模型规模）与可解释性相关，但未在策略合成中被系统应用。

## 核心贡献（创新点）
1. **棋类策略的形式化逻辑建模**：首次将国际象棋策略建模为一阶逻辑规则（Prolog），并定义覆盖率（Coverage）与分歧度（Divergence）作为可计算的可解释性指标——与以往仅用胜率评估不同。
2. **基于ILP的符号策略学习**：使用Popper系统从初学者对局数据中学习棋类战术规则，使策略具有人类棋手可理解的结构——区别于神经网络黑箱策略。
3. **决策Transformer适配程序化策略合成**：将Decision Transformer改造为支持离散动作（Action Masking + Cross-entropy Loss + GRU状态编码），直接应用于DSL程序合成——这是首次将序列建模架构用于此类任务。
4. **样本效率显著提升**：在Karel六任务基准上，DT以极少搜索程序数（如cleanHouse任务仅需59个 vs LEAPS的3627个）达到或超越LEAPS性能，证明Transformer方法的泛化能力。
5. **连接XAI与策略合成的桥梁**：提出用XAI可解释性度量替代"心智对齐"的间接评估方案，为后续策略可理解性量化提供框架。

## 方法详解

### 方法一：基于ILP的棋类策略学习
- **策略表示**：采用一阶逻辑Horn子句，Prolog语法如下：
  ```
  tactic(Position, From, To) ← feature_1(...), feature_2(...), ..., feature_n(...)
  ```
  其中`Position`描述棋盘状态，`From/To`为合法移动，规则头对应"战术动作"。
- **背景知识库B**：包含谓词词汇P（如`attacks/3`、`different_pos/2`等），编码棋盘几何与攻击关系。
- **训练数据**：$E^+$为初学者对局中的⟨position, move⟩正例；$E^-$为不在目标策略中的⟨position, move⟩反例。
- **学习目标**：寻找假设H使$H \cup B \models E^+$（完备性）且$H \cup B \not\models E^-$（一致性）。
- **工程约束**：修改Popper以仅生成合法移动，并剪枝零覆盖率或低召回率策略。
- **评估指标**：
  - **Coverage**：$|\{p \in P \mid \sigma \text{ applicable}\}| / |P|$
  - **Divergence**：策略$\sigma$与参考策略$E$的平均动作距离（用Stockfish 14/ Maia-1600作oracle计算欧氏距离$d_E$）

### 方法二：Transformer-based程序化策略合成
- **MDP形式化**：将程序合成视为MDP $\mathcal{M}_{syn} = \langle S_{syn}, A_{syn}, P_{syn}, r_{syn}, \gamma_{syn} \rangle$，其中：
  - 状态$s \in (V \cup \Sigma)^*$为当前部分展开的程序串
  - 动作$a = (r, v, i)$表示对第$i$位的非终结符$v$应用产生式$r$
  - 奖励在程序终端时取环境奖励$r_{exec}$，否则为0
- **决策Transformer改造**（原架构仅支持连续动作）：
  1. **Action Masking**：非法动作logits置$-\infty$后softmax，确保只采样合法DSL扩展
  2. **Cross-entropy Loss**：替代原L2回归损失，适配离散动作分类
  3. **GRU状态编码**：用GRU（而非直接全连接层）对MDP状态序列嵌入，增强时序表示
  4. **Return-to-go建模**：轨迹格式$(\widehat{R}_1, s_1, a_1, \ldots, \widehat{R}_T, s_T, a_T)$，每个token加时间步embedding
- **训练流程**：从Trivedi et al. (2021)的50K程序数据集预训练，沿用Chen et al. (2021)超参。

## 实验与结果

### 国际象棋实验
- **数据集**：1297对初学者在线对局的⟨position, move⟩，独立测试集100对
- **Oracle参考**：Stockfish 14与Maia-1600
- **基线**：Stockfish 14、Maia-1600、随机策略
- **结果**：学习策略在Stockfish距离度量下显著优于随机基线，逼近初学者水平（图3直方图显示分布集中）
- **结论**：ILP策略可泛化到未见位置，且输出规则具有战术语义（如fork战术图2）

### Karel程序合成实验
- **任务**：6个编程任务（cleanHouse, fourCorners, harvester, randomMaze, stairClimber, topOff）
- **评估**：平均返回（范围[0, 1.1]），10次随机初始状态平均
- **对比基线**：LEAPS（Trivedi et al., 2021，SOTA程序化策略学习）
- **核心数字**（Table 1）：

| Task | LEAPS Return | DT Return | LEAPS Unique Programs | DT Unique Programs |
|------|--------------|-----------|----------------------|-------------------|
| cleanHouse | 0.16 | **0.23** | 3627 | **59** |
| fourCorners | 0.35 | 0.35 | 9872 | **55** |
| harvester | 0.61 | **0.66** | 11708 | **28** |
| randomMaze | 0.97 | **1.00** | 295 | 63 |
| stairClimber | 0.74 | **1.10** | 298 | 49 |
| topOff | 0.80 | 0.66 | 30278 | **63** |

- **最强结果**：DT在4/6任务上超越或持平LEAPS，且搜索程序数极少（平均<60 vs LEAPS数百至数万）
- **关键结论**：Transformer方法样本效率高出2-3个数量级，同时保持竞争力性能

## 相关工作脉络
1. **Neural Program Synthesis（Gulwani et al., 2017综述）**：本文属于此脉络中基于MDP奖励的学习分支，与Input/Output对驱动的方法（如RobustFill、Neural Programmer）定位不同。
2. **Learnable Interpretable Policies（Verma et al., 2018, 2019）**：最早提出程序化可解释RL，但使用 imitation learning；本文用Transformer直接合成，不依赖展示数据。
3. **LEAPS（Trivedi et al., 2021）**：直接对比基线，使用强化学习+grammar约束搜索；本文用序列建模替代搜索，效率显著更高。
4. **Programmatic Strategies for RTS Games（Mariño et al., 2021, 2022）**：聚焦即时战略游戏，本文将其扩展到更通用的Karel域，验证方法普适性。
5. **Chess Tactics Modeling（Berliner 1975; Pitrat 1977; Bratko 1982）**：早期符号AI工作；本文继承其战术谓词思想，但引入从数据自动学习而非手工编码。
6. **Decision Transformer（Chen et al., 2021）**：原始架构针对连续控制；本文首次将其改造用于离散DSL程序合成，引入Action Masking与GRU编码等关键修改。

## 局限性与未来方向
- **局限性**：
  - 策略可解释性仅通过覆盖率/分歧度间接衡量，未进行真实人类玩家理解度实验
  - 国际象棋实验仅针对初学者策略，未扩展到中级/专家级模式
  - Karel任务空间简单，未见在复杂游戏（如围棋、星际）上的验证
  - Transformer方法依赖50K程序训练数据，数据稀缺场景未讨论
- **未来方向**（论文自述）：
  - 在更多游戏化环境中基准测试Decision Transformer
  - 开展人类用户可理解性评估实验
  - 量化学习策略对真实玩家表现的提升幅度

## 研究启发与可借鉴点
1. **XAI指标用于策略评估**：将认知块数量、覆盖度等可解释性度量正式引入策略合成评估体系，为"可理解AI"提供量化口径，可直接迁移至其他RL可解释性研究。
2. **GRU状态编码+Transformer决策**：将GRU用于MDP状态序列嵌入后再接入Transformer，比原始Decision Transformer的扁平嵌入在离散任务上表现更好；这一架构改造可复用于任何离散动作的序列决策任务。
3. **Action Masking in Decision Transformer**：针对无效动作的logits置$-\infty$技巧，解决了DT在语法/约束环境下的采样合法性问题，对编程语言合成、机器人导航等有通用价值。
4. **ILP + 启发式剪枝的工程实践**：修改Popper加入合法性约束与低召回剪枝，使符号学习可直接产出可执行代码，为其他符号-神经混合系统提供模板。
5. **从"胜率优化"到"对齐优化"**：论文将目标从纯性能转向可理解性-性能权衡，启发后续工作可在训练中加入人类偏好对齐损失（如KL散度约束策略分布）。

## 关键术语表
**Strategy Synthesis（策略合成）**：自动学习参数化游戏动作序列（可执行程序），使其可被人类理解并用于改进心智模型。
**Model Matching Theory（模型匹配理论）**：认为游戏模型与玩家心智模型的对齐程度决定策略学习效果，是本文可解释性目标的理论基础。
**Coverage（覆盖率）**：策略在给定位置上可应用的概率，衡量策略泛化范围。
**Divergence（分歧度）**：学习策略与参考策略的动作距离期望，衡量逼近精度。
**Inductive Logic Programming (ILP)**：从正负例霍恩子句中归纳逻辑规则的符号学习方法，本文用于从对局数据学习棋类战术。
**Decision Transformer**：将Transformer用于序列决策的架构，以return-to-go条件化下一个动作预测；本文对其离散化改造后应用于程序合成。
**Karel Domain**：教学用网格编程语言环境，含move/turn/pick-put等动作，本文选用其六任务基准验证程序化策略学习。
**Return-to-go ($\widehat{R}_t$)**：从时刻t到终局的累积奖励，替代原MDP中的状态价值输入Transformer。

## 可复现要素
- **数据集**：
  - 国际象棋：1297对初学者对局⟨position, move⟩（来源：在线对局采样），测试集100对
  - Karel：Trivedi et al. (2021)提供的50,000程序数据集
- **代码开源**：论文未明确声明代码开源状态，需另查
- **权重开源**：未提及
- **关键超参**：
  - ILP：Popper系统，阈值需"empirically-determined"（论文未给出具体数值）
  - Transformer：沿用Chen et al. (2021)超参（未列出具体值）
  - 训练数据量：Karel任务50K程序
- **评估协议**：Karel返回均值±标准差，10次随机初始状态；国际象棋用Stockfish 14与Maia-1600作distance oracle
