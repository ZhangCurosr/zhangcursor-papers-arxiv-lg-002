---
title: "The-Constitutional-Coverage-Trilemma-in-AI-Governance"
source: https://arxiv.org/pdf/2609.01275v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 07:28:01"
---

# 论文速读：The-Constitutional-Coverage-Trilemma-in-AI-Governance

## 一句话总结
本文在同一五维价值单纯形上同步审计 1,649 名美国用户的宪法偏好与 23 个前沿 LLM 默认配置，发现供给凸包仅覆盖需求凸包的约 2%，且自主性权重在 5/6 厂商家族中随版本单调下降；理论证明存在预算多元主义三角困境，并提出仅需 `{Honesty, Autonomy}` 两个顶点的稀疏菜单即可将平均 regret 降低 47%。

## 研究问题与动机
- **核心问题**：当前部署的前沿 AI 系统（每模型隐含一套安全/效用/诚实/自主/公平的价值排序）是否足以覆盖人类多样化的宪法需求？
- **现有 routing 思路的盲区**：主流方案聚焦“偏好采集+匹配路由”，但若底层菜单本身高度压缩，精准 elicitation 也无用武之地；本文引入 **constitutional homelessness** 概念指出该瓶颈。
- **动态漂移风险**：供应商按版本迭代时，系统价值分布是否在背离本就覆盖不足的价值维度？若是，将机械抬升最弱势群体的 welfare floor。
- **工程修复诉求**：若供给压缩是结构性的，能否在理论下界指导下用极少量顶点重构出显著优于完整原型的菜单？

## 核心贡献（创新点）
1. **同构联合审计框架**：在同一 pair-wise tradeoff 仪器上同步测量人类（n=1,649）与前沿 LLM（k=23, 6 families）的宪法分布，首创供需可比的基础设施。
2. **覆盖率理论形式化**：提出 $\beta_r(A)$ 与 regret 下界（Theorem 1），证明无严格多数主导时 $M_i^A \ge \frac{1}{2}m_i$；将静态缺口扩展为动态漂移下的 regret floor 单调上升命题（Corollary 3）。
3. **预算多元主义三角困境（Trilemma）**：证明在菜单规模受限且需保障 viability floor 时，个性化效率、组间 regret 均等化、有界菜单三者不可兼得（Theorem 2），并指出当前实证 regime 恰好处于绑定状态。
4. **稀疏顶点充分性与修复算法**：证明最优小菜单必可由单纯形顶点构成（Theorem 3-4），并将贪婪修复建模为单调子模最大化，提供 $(1-1/e)$ 近似保证；实证显示 `{e_HON, e_AUT}` 以 2 个顶点击败完整 23 原型菜单。
5. **stakes-conditioned 漂移诊断**：揭示自主性下降并非“安全训练生效”，而是集中于低风险场景（辞职信、讽刺段子），最新一代模型已丧失历史版本所具备的 stakes 区分能力。

## 方法详解
- **价值空间与福利模型**：五维值 $(SAF, HLP, HON, AUT, EQT)$ 归一化至 $(K-1)$-单纯形 $\Delta^K$；用户 $i$ 偏好 $\pi_i$，机构 $\alpha$ 的线性福利 $U_i(\alpha)=\langle \pi_i, \alpha \rangle$，理想福利 $U_i^\dagger=\max_k \pi_i^k$，menu regret $M_i^A = U_i^\dagger - \max_{\alpha\in A}\langle \pi_i,\alpha\rangle$。
- **覆盖率与 homeless 定义**：$\beta_r(A)=\max_{\alpha\in A}\alpha_r$；若 $\beta_r\le 1/2$ 为 strict-β homeless，若 $\arg\max_k \alpha_k \neq r$ 为 argmax homeless。用户首要价值对应类型为 homeless 则该用户 homeless。
- **人类侧度量**：20 题 AI Jamm 配对权衡电池（10 对×2 次反向呈现），计算 concordant winners 分布得 $\hat{\pi}_i$（平均 concordance 0.793）；discordance 经行为验证为“ deliberation ”而非噪声（Spearman $\rho=0.84, p=0.002$）。
- **LLM 审计协议**：10 场景各生成 20 个语义等价 paraphrase（Mistral Large 生成，排除于测试池防污染），每变体双向 position-swap 查询，共 $21\times10\times2=420$ trials/model；concordance floor 0.70 纳入静态菜单；temperature 0 解码。
- **漂移检验**：按家族内时间序列计算 Spearman $\rho_{f,j}$，构造 $S_j=\sum_f (k_f-1)\rho_{f,j}$，在组内置换整条版本序列（$4!\cdot6!\cdot4!\cdot4!\cdot2!\cdot2! = 39,813,120$ 种）下蒙特卡洛检验交换性，得 $p=0.013$。
- **稀疏修复**：证明总福利函数 $F(S)=\sum_i \max_{k\in S}\pi_i^k$ 单调子模（Theorem 5），采用均值贪婪与 worst-group 贪婪两路搜索，理论保证 $(1-1/e)$ 近似。

## 实验与结果
- **数据集**：人类 N=1,649（Prolific 美国，政治身份分层）；LLM 23 archetypes（Claude/GPT/Gemini/Llama/Grok/DeepSeek 六家族多版本）。
- **主要结果**：
  - 需求分布：SAF 32.6%, HON 22.6%, AUT 19.2%, HLP 18.1%, EQT 7.6%；无任何价值 > 1/3。
  - 供给覆盖系数 $\beta=(0.394, 0.257, 0.333, 0.161, 0.381)$；HLP 与 AUT 的 argmax 主导原型数均为 0。
  - 供需凸包比：保守估计约 **2%**（full precision 仅 0.10%）。
  - 漂移：AUT 权重在 5/6 家族单调下降（GPT $\rho=-0.94$, Gemini/Llama $\rho=-0.80$, DeepSeek $\rho=-1.0$）；EQT 在 5/6 上升，SAF 在 4/6 上升。
  - 三角困境实证：|A|=3 时最优均值菜单 `{SAF, HON, AUT}` 使 EQT 用户 regret 从 0.175 升至 0.215；worst-group 下界 0.093，当前 frontier $\zeta_A=0.095$ 处于绑定。
  - 稀疏修复：`{e_HON, e_AUT}` 平均 regret 0.074 vs 完整菜单 0.140（**47% 提升**, CI [43%, 52%]）；3 步均值贪婪削减 mean regret 81%、worst-group 17%；3 步最坏组贪婪削减 worst-group 64%、mean 74%。
- **稳健性**：匹配福利（matching
