---
title: "Testing-Between-the-Test-Cases-Proving-End-to-End-Steering-i"
source: https://arxiv.org/pdf/2609.10951v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-01 10:39:30"
field: "自动驾驶形式化验证与 E2E 策略鲁棒性"
keywords: ["形式化验证", "端到端自动驾驶", "CROWN 界传播", "鲁棒性认证", "扰动族", "CARLA 仿真", "安全案例"]
innovations: ["pose-paired 线性扰动族将连续无穷多强度压缩为单参数 CROWN 可界集合", "首次将 per-frame 界传播与闭环车辆运动学耦合输出持续转向偏差容限", "在仿真通过的测试条件下发现介于清晰与恶劣之间的中间强度失效"]
benchmarks: ["CARLA Town04 高速环线", "CARLA Town06 城市干线环线", "CTE 2.19 ft 安全预算"]
---

# 论文速读：Testing-Between-the-Test-Cases-Proving-End-to-End-Steering-i

## 一句话总结
本文首次将神经网络的**形式化验证（CROWN 界传播）**应用于端到端自动驾驶转向策略的鲁棒性证明，通过在两条 CARLA 道路（Town04 高速公路、Town06 城市干线）上构建"清晰–恶劣条件"之间的**线性插值扰动族**，在不实际驾驶的情况下一次性覆盖连续无穷多干扰强度，**不仅正确预测了所有已驱动的测试用例结果，还提前发现了处于"两个测试条件之间"的中间强度失效点**。

## 研究问题与动机

1. **有限测试与无限风险的矛盾**：E2E 驾驶策略最优者恰恰最不可枚举——ISO 26262 / 21448 / 8800 等安全标准的"列举失效模式"框架对端到端网络天然不适配；任何测试活动只能访问有限输入，而视觉扰动在物理上是连续不可数的。
2. **纯仿真验证的盲区**：CARLA 等模拟器承担了大量实证测试，但改变一个参数就需重跑整个 campaign；数据驱动策略在长尾分布外泛化极差，现有"收集恶劣数据集"和"对抗样本迁移"手段仍无法穷举连续扰动空间。
3. **形式化验证在 AV 领域几乎未落地**：SMT（Reluplex、Marabou）、可达性框架（NNV、Verisig）和抽象解释（AI²）工具在学术上已有成熟进展，但对 E2E 感知–控制链路在真实物理扰动下的闭环安全保证仍几乎没有工作。
4. **关键科学问题**：能否把"连续扰动族"翻译成一个神经网络验证器可以数学界定的集合，从而用一次计算替代一场测试活动，并发现那些**仿真从未跑到却必然失败**的中间强度？

## 核心贡献（创新点）

1. **提出 pose-paired 线性扰动族** $x_p(s) = x_p^{\text{clear}} + s(x_p^{\text{cond}} - x_p^{\text{clear}}),\; s \in [0,1]$，将多像素扰动压缩为单参数族，使 CROWN 验证器能以一条界传播覆盖无穷多强度——区别于 Mohapatra et al. 只对单一强度做 $\ell_\infty$ 球或语义参数独立扰动。
2. **首次在 E2E 闭环转向场景中落地形式化验证**：把 per-frame 界传播与车辆运动学（自行车模型积分）耦合，导出"持续转向偏差容许量" $\delta_{\text{tol}}=0.0120$（对应 CTE budget 2.19 ft、$T_{\text{cl}}=1.85$ s），而非仅看单帧峰值。
3. **发现"测试条件之间"的失效**：在 Town06 干线清晰训练模型雾天/低太阳工况下，仿真实测在 $s=1$ 处偏差仅 0.69/0.37（通过），但验证器找到中间强度 $s^\star=0.41/0.60$ 使偏差达到 1.01/1.87（失稳）——这是实证测试永远抓不到的失效。
4. **给出可迁移的设计教训三则**：混合训练 DAgger 轮次需 warm start 降学习率微调而非从头训；同等神经元预算下加宽（width）比加深（depth）对降低认证界更友好（宽化使界缩小 2.3–3.7 倍）；teacher 在已通过 gate 后继续 DAgger 精炼，student 蒸馏结果再优 2.7%。

## 方法详解

### 3.1 仿真平台与 ODD
- **CARLA 0.9.16**，Tesla Model 3 蓝图，前向 RGB 640×480、FOV 90°。
- 纵向速度恒 $v_x=8.9408$ m/s（20 mph）由 PI 控制，5 Hz 控制周期（$\Delta t=0.2$ s）。
- **Town04**（分离式高速环线 1.32 mi）与 **Town06**（城市干线环线 1.42 mi）两条不同 ODD，交叉口均排除在测量外。
- 每个测试用例跑 3 次独立 lap，三者一起作为认证单元。

### 3.2 模型架构（教师–学生蒸馏）
- **教师**：PilotNet 类（5 conv + 4 FC、全 ReLU、200×66 RGB、~107k ReLU 神经元），仅用于蒸馏源。
- **学生**：3 strided conv + 2 FC 的 ReLU CNN，由 GT 分割 ROI 裁剪以提升低分辨率下车道线可读性。
- 四款学生网络规模：Town04 清晰 5,152 / 混合 15,456 ReLU；Town06 清晰 50,944 / 混合 101,888 ReLU（输入宽度按道路复杂度加倍）。
- 训练流程：PPC 专家行为克隆 → DAgger 迭代 → 教师收敛 → 蒸馏学生 → 学生 DAgger 至通过 CTE budget。

### 3.3 扰动建模与验证集构造
- 对每条路线上 $P$ 个 pose，**从相同相机位姿**各渲染一帧清晰 $x_p^{\text{clear}}$ 和一帧恶劣（雾/夜/低太阳）$x_p^{\text{cond}}$，两者经裁剪降采样送入网络。
- 扰动族：Eq.(1) 线性插值，$s \in [0,1]$ 为唯一自由参数。裁剪/降采样均为线性变换，故在原始满分辨率上插值再缩放等价于先缩放再插值，保真。
- 废弃方案对比：① Koschmieder 解析雾模型（$R^2=0.848$ 视觉相似，但策略响应差异大）；② ACDC 真实数据集（存在视角和场景漂移）；③ 雨（CARLA 渲染具时间随机性，两终点族无法表达）。

### 3.4 安全准则与运动学推导
- **CTE budget**：$w_{\text{lane}}=3.500$ m、$w_{\text{veh}}=2.164$ m → $\text{CTE}_{\text{budget}}=0.668$ m = 2.19 ft（Eq.2）。
- 由自行车模型 $\dot{\psi}=\frac{v_x}{L}\delta$ 积分得：持续偏差 $\Delta\delta$ 在 $T$ 秒内累积横向偏移 $y(T)=\frac{v_x^2 T^2}{2L}\Delta\delta$（Eq.5）。
- 令 $y(T)=\text{CTE}_{\text{budget}}$ 解出容许转向偏差（归一化到 $\delta_{\max}=70°$）：$\delta_{\text{tol}}=\frac{2L\cdot\text{CTE}_{\text{budget}}}{v_x^2 T_{\text{cl}}^2 \delta_{\max}}=0.0120$（Eq.6）。
- **闭环反应时** $T_{\text{cl}}=1.85$ s，仅在高公路标定一次，干线复用不加校准；结论在 $T_{\text{cl}}$ 因子 1.7 窗口内稳健。

### 3.5 形式化验证（CROWN 界传播）
- 算法：基础 CROWN（非 $\alpha$-CROWN、$\beta$-CROWN、SDP-CROWN），因单参数族不触发高维分支收益。
- 界为**单边放宽**，只能更松不能更紧；为提升精度，将 $[0,1]$ 切为 16 个子区间逐段界传播后取最坏，比精化算法本身更重要。
- 每 8 个控制步采样 1 个 pose，全线 $P$ 个 pose，计算相对清晰帧的持续偏差 $\Delta_p(s)=\delta_p(s)-\delta_p(0)$（Eq.7）。
- 路线平均偏差：$\bar{\Delta}(\mathbf{s})=\frac{1}{P}\sum_{p=1}^P \Delta_p(s_p)$（Eq.8），**每 pose 可取不同强度**，向量 $\mathbf{s}\in[0,1]^P$。
- 认证条件：$\max_{\mathbf{s}\in[0,1]^P}|\bar{\Delta}(\mathbf{s})|\leq\delta_{\text{tol}}$（Eq.9）。
- 结果以相对单位报告：$\bar{\Delta}_{\text{rel}}(\mathbf{s}) = \bar{\Delta}(\mathbf{s})/\delta_{\text{tol}}$，阈值 1.0 为容限边界（Eq.10）。

## 实验与结果

### 数据集与平台
- CARLA 0.9.16 Town04（高速 1.32 mi）、Town06（干线 1.42 mi）；4 款学生网络，2 种训练分布（clear-only / mixed: 清晰+雾+夜+低太阳），4 种扰动条件。

### 主要结果（Table 2 汇总）
| 道路 | 模型 | 条件 | 仿真 | 最坏 CTE (ft) | 形式化验证 |
|---|---|---|---|---|---|
| Town04 | Clear-trained | clear | PASS | 0.91 | certified† |
| Town04 | Clear-trained | fog | PASS | 1.32 | **certified** |
| Town04 | Clear-trained | night | **FAIL** | exceeds | not certified |
| Town04 | Clear-trained | low sun | **FAIL** | exceeds | not certified |
| Town04 | Mixed-trained | clear | PASS | 0.39 | certified† |
| Town04 | Mixed-trained | fog | PASS | 0.37 | certified |
| Town04 | Mixed-trained | night | PASS | 0.85 | certified |
| Town04 | Mixed-trained | low sun | PASS | 0.54 | certified |
| Town06 | Clear-trained | clear | PASS | 1.37 | certified† |
| Town06 | Clear-trained | fog | **FAIL** | exceeds | not certified |
| Town06 | Clear-trained | night | **FAIL** | exceeds | not certified |
| Town06 | Clear-trained | low sun | **FAIL** | exceeds | not certified |
| Town06 | Mixed-trained | clear | PASS | 1.30 | certified† |
| Town06 | Mixed-trained | fog | PASS | 1.95 | **not certified** |
| Town06 | Mixed-trained | night | PASS | 1.00 | **not certified** |
| Town06 | Mixed-trained | low sun | PASS | 2.19 | certified |

†vacuous：以清晰帧为参考点，偏差由构造为 0。

### 关键发现
1. **Town04 全部预测正确**：6 项测试均与仿真结果一致。
2. **Town06 发现"之间"失效**：
   - Clear-trained 雾/低太阳：$s=1$ 处仿真通过（0.69/0.37），但验证器找到 $s^\star=0.41/0.60$ 中间强度使偏差超容限 1.01/1.87。
   - Mixed-trained 雾/夜：仿真通过（1.95/1.00），但三角指标（pose-by-pose 不同强度）分别达 1.179/1.210——**非验证界松所致，而是策略确实在交替强度下失稳**。
3. **计算效率**：一条 bounds 在单 GPU 上几分钟完成，覆盖 $10^{133}$ 种强度组合（133 poses × 10 强度/pose），等价于一场测试活动的信息量但耗时相当。
4. **ODD 敏感**：两种网络尺寸、两种分辨率、16 种独立蒸馏种子均复现相同结论，说明差异来自道路几何（干线 74–79% 直路、最小半径 22–27 m vs 高速 51–56% 直路、45–63 m）而非容量或随机性。

## 相关工作脉络

1. **Koopman & Wagner (2016/2018)**：指出 ISO 26262 V-model 在 AV 场景失效，单次 fatal/6.62 亿英里推算需 ~66.2 亿英里测试——本文继承"统计安全案例不足以覆盖连续扰动"的核心论断。
2. **Mohapatra et al. (CVPR 2020)**：语义参数扰动（亮度/对比度）的 E2E 鲁棒性验证先驱；本文采纳其"参数化扰动"思路，但以 pose-paired 仿真渲染为端点、以单参数线性族替代独立语义扰动，并首次耦合闭环动力学。
3. **CROWN 系列 (Zhang et al. NeurIPS 2018; Xu et al. NeurIPS 2020)**：界传播算法基础；本文使用基础 CROWN，说明在单参数族上 $\alpha/\beta$/SDP 精化收益有限，设计启示比算法精化更关键。
4. **Bernardeschi et al. (2025 IEEE Access)**：评估视觉 E2E 鲁棒性并给出概率保证，但未覆盖全局光照宏观扰动；本文明确区分"看起来像雾"和"让策略像雾一样响应"。
5. **ACDC 数据集 (Sakaridis et al. ICCV 2021)**：真实恶劣条件配对数据集，但因视角/场景漂移被本文放弃，凸显 pose-paired 同位姿采样的必要性。
6. **NNAv/Verisig/AI² 等验证工具**：面向静态图像或开环网络；本文首次把 per-frame 界传播与闭环车辆动力学（Eq.3–6）结合，输出"持续转向偏差"而非单帧类别边界。

## 局限性与未来方向

1. **扰动族过窄**：当前仅为单参数线性插值，无法表达雨（时间随机）、雪、尘、眩光及多扰动组合；扩展至更高维参数族（如 SDP-CROWN 适用的高维 $\ell_2$ 球）是下一步。
2. **仅验证转向单一动作**：纵向速度由 PI 控制器维持，未联合验证加速度/制动决策，完整纵向–横向耦合仍未覆盖。
3. **网络规模限制**：可验证网络必须小（最大 ~102k ReLU），难以直接推广到更大规模 E2E 策略（如 DriveNet、VAD 类）；宽化优于加深虽有益，但上限仍在。
4. **CARLA 仿真–现实差距**：形式化证明只对"仿真中呈现的扰动族"成立，真实相机噪声、传感器延迟、标定误差未纳入；verification gap 仍待量化。
5. **未来验证计划**：作者计划在后续工作中对已识别的"之间失效"启动新一轮闭环测试，用以改进模型鲁棒性，并扩展至雨/雪/组合扰动。

## 研究启发与可借鉴点

1. **"两终点插值族"设计可迁移**：凡是有"好/坏"两类状态配对数据的任务（医疗影像、工业缺陷检测、语音抗噪），均可借鉴 pose-paired 线性族 + 单参数验证的方式，在有限测试基础上捕获中间态失效。
2. **teacher 过 gate 后继续 DAgger 精炼**：传统认知是 teacher 收敛即止，本文证明继续精炼能再提升 student 2.7%，提示蒸馏流程应保留 teacher 后期迭代空间。
3. **宽度>深度对认证更友好**：同等神经元预算下，宽卷积/FC 层比深网络显著收窄认证界（2.3–3.7×），指导未来"可验证高效模型"的架构搜索目标函数。
4. **per-pose 独立强度比 worst-case 单帧统计更可靠**：原文初稿用"最坏单帧偏差"排序导致与仿真结果反向；改为路线平均 $\bar{\Delta}$ 后判定稳定，提示形式化验证的聚合统计量选择需经闭环反事实检验。
5. **ODD 差异不可内推**：同一策略在高速通过而在干线失败的结论，不能简单归因于网络容量或随机性，道路几何是首要变量——跨 ODD 迁移验证结论时必须重新标定量（如 $T_{\text{cl}}$）。

## 关键术语表

- **End-to-End (E2E) steering**：相机像素直接映射到转向角的单个神经网络，无分离的感知/规划模块。
- **Bound propagation (界传播)**：以代数方式把输入区间逐层推到输出区间，ReLu 以线性上下包络近似；CROWN 为该算法家族。
- **CROWN**：Zhang et al. (NeurIPS 2018) 提出的可扩展神经网网络鲁棒性认证算法，基础形式在此工作使用。
- **ODD (Operational Design Domain)**：ISO 34503 定义的自动驾驶系统运行设计域，以道路类型、几何为一级属性；本文两条 CARLA 地图即为不同 ODD。
- **DAgger (Dataset Aggregation)**：Ross et al. (AISTATS 2011) 的行为克隆迭代训练，策略在跑偏时回查专家纠正标签并追加训练集。
- **Cross-track error (CTE)**：车辆中心到车道中心的横向偏移，本文以 2.19 ft 为安全边界。
- **Closed-loop reaction horizon ($T_{\text{cl}}$)**：假设系统性转向偏差持续而不被修正的时间窗；本文取 1.85 s，仅在高公路标定一次。
- **Vacuous certificate (空洞认证)**：以清晰帧为参考点时 $\Delta_p(0)=0$ 由构造成立，"通过"无信息量。

## 可复现要素

| 要素 | 状态 |
|---|---|
| 代码与流水线 | 开源，Zenodo DOI: 10.5281/zenodo.22101297，GitHub 组织 AD-Assurance-Lab |
| 捕获帧数据集 | 开源，Hugging Face Datasets: `AD-Assurance-Lab/steering-verification-captures` |
| 仿真平台 | CARLA 0.9.16（开源） |
| 网络权重/checkpoint | 随软件记录一并发布 |
| 关键超参 | 控制频率 5 Hz、$\Delta t=0.2$ s、$v_x=8.9408$ m/s、$\delta_{\max}=70°$、$T_{\text{cl}}=1.85$ s、输入分辨率 84×28 / 168×56、CROWN 子区间数 16、采样间隔 8 控制步 |
| 训练细节（学习率/轮次/种子数） | 论文未提及；预注册 16 种独立蒸馏种子、两种宽度用于Town06 抗随机性检验 |
