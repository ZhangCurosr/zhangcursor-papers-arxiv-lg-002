---
title: "Steady-State-Convergence-of-Stochastic-Approximation"
source: https://arxiv.org/pdf/2609.14922v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-10-01 17:06:46"
---

# 论文速读：Steady-State-Convergence-of-Stochastic-Approximation

## 一句话总结
本文建立了常数步长收缩随机逼近（SA）的统一稳态收敛（SSC）理论，在Markovian与乘性噪声驱动下同时覆盖局部可微与局部不可微两类均值算子，获得$\mathcal{W}_2$意义下$\mathcal{O}(\sqrt{\alpha})$的最优高斯逼近界并消除对数因子，同时首次揭示非光滑情形下渐近偏差从$\alpha$阶跃变至$\sqrt{\alpha}$阶，且理论可严格覆盖异步Q-learning等重要算法。

## 研究问题与动机
- **核心问题**：常数步长SA的稳态分布形态与收敛速率在现实噪声（Markovian、乘性）与非光滑算子下缺乏统一刻画，现有理论难以指导算法的稳态偏差控制。
- **现有方法局限①**：多数前作强制要求i.i.d.或纯加性噪声假设，无法处理环境依赖的乘性扰动。
- **现有方法局限②**：普遍依赖全局可微性，对含max、谱映射、投影算子的非光滑均值算子失效。
- **现有方法局限③**：既有界中残留$\log(1/\alpha)$等对数因子，且仅在较弱的凸集距离或$W_1$度量下成立，无法反映真实稳态分布的几何精度。

## 核心贡献（创新点）
1. **统一SSC理论框架**：首次为常数步长收缩SA建立覆盖局部可微与不可微算子的统一稳态收敛理论，兼容Markovian与乘性噪声，填补了非光滑稳态分析的空白。
2. **多步通用性约简设计**：提出Step 0→3的递进约简流程，通过加性噪声约简、Jacobian漂移约简/独立高斯约简将复杂递归转化为同极限的高斯扩散过程，突破单次线性化的分析瓶颈。
3. **最优率与度量升级**：消除先前结果中的$\log(1/\alpha)$对数因子，在更强的$\mathcal{W}_2$度量下达到不可改进的$\mathcal{O}(\sqrt{\alpha})$稳态高斯逼近界，并导出退火步长下的$\mathcal{O}(\sqrt{\log t/t})$有限时间率。
4. **非光滑偏差阶跃变揭示**：证明光滑情形渐近偏差为$\alpha$阶，而单侧方向可微（max/谱/ prox-正则）情形主导偏差升至$\sqrt{\alpha}$阶，统一刻画了非光滑性带来的统计代价。
5. **重要算法的理论覆盖**：严格验证Markovian线性SA与异步Q-learning的均值算子满足本文假设，为前者提供首套SSC保障，为后者（算子非单点次微分）建立非光滑稳态收敛分析。

## 方法详解
- **目标递归与缩放变量**：研究$\theta_{t+1} = \theta_t + \alpha(\tilde{T}(x_t, \theta_t) - \theta_t)$，定义稳态缩放变量$Y_t^{(\alpha)} = (\theta_t^{(\alpha)}-\theta^*)/\sqrt{\alpha}$，目标是刻画其极限分布。
- **多步通用性约简（Multi-step Universality Reduction）**：
  - **Step 1（加性噪声约简，Prop. 1）**：将$\tilde{T}(x,\theta)$替换为$\mathcal{T}(\theta)+h(x)$，其中$h(x)=\tilde{T}(x,\theta^*)-\theta^*$，得辅助稳态$A_\infty^{(\alpha)}$，$W_2$误差$\mathcal{O}(\sqrt{\alpha})$。
  - **Step 2a（Jacobian漂移约简，Prop. 2，光滑）**：用$J=\nabla\mathcal{T}(\theta^*)$线性化$\mathcal{T}$，构造$B_\infty^{(\alpha)}$，误差$\mathcal{O}(\sqrt{\alpha})$。
  - **Step 2b（独立高斯噪声约简，Prop. 4，一般）**：将Markov噪声$h(x_t)$替换为协方差匹配长程协方差$\Sigma_h$的i.i.d.中心高斯序列，得$Z_\infty^{(\alpha)}$，误差$\mathcal{O}(\alpha^{1/4})$。
  - **Step 3**：直接调用[ZHCX24]的i.i.d.高斯SSC结果，得到$Y_\infty \sim \mathcal{N}(0,V)$，$V$为Lyapunov方程$(J-I)V+V(J-I)^\top+\Sigma_h=0$的唯一半正定解。
- **核心证明工具**：
  - **一致分块高斯耦合（Prop. 5）**：在Markov设定下将序列分块，逐块构造条件高斯近似，保持全局$W_2$可控。
  - **$W_p$-解耦论证（Lemma 1）**：通过扩展概率空间使条件律$\mathcal{L}((Y,Y^*)|\mathcal{G})=\pi_\omega$成立，且$Y^*$与$\mathcal{G}$独立、$\mathcal{L}(Y^*)=\mu$，实现耦合构造。
  - **Moreau包络+六项拆分**：定义$V_t = \mathbb{E}[M_\eta(\Delta_{nt})]$（$\Delta_k=A_k-Z_k$），将递归不等式拆为$T_1=\sum_{i=1}^6 T_{1i}$逐项估计，取$n=\lfloor\alpha^{-1/2}\rfloor$使压缩系数$<1$，迭代得$\limsup V_t \leq C\alpha^{1/2}$，三角不等式完成$W_2$界。
- **非光滑扩展**：Assump. 5要求$\mathcal{T}$在$\theta^*$处单侧方向可微（涵盖max型、谱映射、prox-正则函数），Corollary 2/3给出对应的SSC与有限时间界，附录H验证异步Q-learning在最优action数$>
