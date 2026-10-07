---
title: "Optimal-and-Eficient-Online-Inverse-Optimization"
source: https://arxiv.org/pdf/2610.08735v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-07 11:12:31"
---

# 论文速读：Optimal-and-Eficient-Online-Inverse-Optimization

## 一句话总结
本文针对在线逆线性优化问题，提出确定性 proper 算法 **RVM（Revocable Variable Metric）**，在多项式时间内实现最优的 $O(\sqrt{d})$ 遗憾界，并给出有限精度比特复杂度版本，正面回答了 Sakaue 提出的开放问题。

## 研究问题与动机
- **核心问题**：学习器每轮仅观察到专家对隐藏线性目标 $w^* \in \mathbb{R}^d$ 的最优动作，目标是推断 $w^*$ 并使累积遗憾 $R_T = \sum_{t=1}^T \langle w^*, x_t - \hat{x}_t \rangle$ 随维度 $d$ 的增长尽可能小。
- **现有方法不足**：
  1. 早期 OGD/MWU 方法遗憾为 $O(\sqrt{T})$，依赖视界 $T$，无法做到 uniform in $T$。
  2. 几何切割平面法（Besbes 等、Gollapudi 等）虽实现 uniform 界，但遗憾阶为 $O(d^4 \ln T)$ 至 $\exp(O(d\ln d))$，或未达最优维度缩放。
  3. Sakaue 等 [28,29] 将遗憾降至 $O(d \ln T)$ 且每轮 $O(d^2)$，仍未突破 $d$ 的线性阶。
  4. Cai 等 [10] 的变量度量法实现 proper、确定性、多项式时间的 $O(d)$ 遗憾，但距最优阶 $\sqrt{d}$ 仍有差距；Sakaue [27] 虽达到 $O(\sqrt{d})$，但为随机化 improper 算法，每轮需 $(dT)^{O(d)}$ 次线性优化，无多项式上界。
  5. **开放问题**：能否在不牺牲最优遗憾阶的前提下，设计确定性、proper 且每轮计算时间为 $\operatorname{poly}(d,T)$ 的算法？

## 核心贡献（创新点）
1. **提出 RVM 算法**：首次以确定性、proper 方式实现 $O(\sqrt{d})$ 最优遗憾界，且每轮算术运算为 $\operatorname{poly}(d,T)$，同时保持 uniform in $T$。
2. **引入可撤销度量更新（revocable stretch）**：每条历史更新绑定一个以创建点为中心的 slab 区域，查询点一旦移出即永久撤销，避免旧信息过度固化导致搜索停滞。
3. **给出跳过变体（Proposition 3.3）**：仅当 $s_t^2 > (t+1)^{-4}$ 时创建 stretch，将活跃更新数压至 $O(d \log t)$，实现均摊 $O(d^2 \log T)$ 每轮代价而不损失遗憾界。
4. **完成有限精度比特复杂度分析（Appendix A）**：设计 rounded RVM，在 dyadic 网格上四舍五入响应与速度，证明遗憾常数放大至 21 仍为 $O(\sqrt{d})$，且每轮比特运算为 $\operatorname{poly}(d, \log t, \ell_R)$。
5. **理论最优性闭环**：证明 $O(R\sqrt{d})$ 是切割平面游戏与在线逆优化的硬下界，本文在遗憾阶、确定性、proper 性与计算效率四维度上同时达到理论极限。

## 方法详解
- **问题归约**：将在线逆线性优化归约为带**强分离预言机（strong separation oracle）**的切割平面游戏。学习器查询 $p_t$，预言机返回单位向量 $v_t$ 满足 $r_t := \langle w^*-p_t, v_t\rangle \geq 0$，遗憾为 $\sum r_t$。逆优化中取回复 $v_t = (x_t-\hat{x}_t)/\|x_t-\hat{x}_t\|$ 即可完成 proper 映射，且 $R_T \leq 2\operatorname{Reg}_T$。
- **RVM 核心更新**：从 $p_1=0, A_1=\varnothing$ 开始。第 $t$ 轮接收 $v_t$ 后，沿连续路径 $\hat{p}_t(\eta) = p_t + \
