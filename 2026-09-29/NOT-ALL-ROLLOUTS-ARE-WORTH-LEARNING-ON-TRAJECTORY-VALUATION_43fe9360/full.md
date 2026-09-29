# NOT ALL ROLLOUTS ARE WORTH LEARNING: ON TRAJECTORY VALUATION FOR POST-TRAINING REINFORCEMENT LEARNING

Xuesong Jia<sup>∗</sup>, Ziao Yang<sup>∗</sup>, Zhanhe Huang, Hongfu Liu

Department of Computer Science, Brandeis University

<sup>∗</sup>Co-First Author with Equal Contribution

{jasjia,ziaoyang,erichzh,hongfuliu}@brandeis.edu

## ABSTRACT

We consider the problem of trajectory valuation in reinforcement learning: how to identify and mitigate detrimental trajectories during online training. Unlike classification, where data valuation relies on fixed training and validation sets, reinforcement learning involves dynamically generated trajectories without explicit validation signals, making conventional influence-based methods inapplicable. We propose Dynamic Trajectory Valuation (DTV), a simple and efficient framework that estimates trajectory utility at the mini-batch level and filters detrimental trajectories based solely on gradient information. By operating at the optimization level, DTV integrates seamlessly with existing reinforcement learning pipelines with minimal overhead. Extensive experiments across diverse settings, including PPO, GRPO, and DPO, demonstrate that DTV consistently improves performance, enhances data efficiency, and stabilizes optimization.

## 1 Introduction

Reinforcement learning is a paradigm in which an agent learns sequential decision-making policies through interaction with an environment, with the goal of maximizing long-term cumulative rewards [1]. Unlike classification with fixed labeled datasets, reinforcement learning data—commonly referred to as trajectories—is generated online through the agent’s rollouts. As a result, the quality of collected trajectories directly affects learning performance, and not all trajectories contribute positively to policy optimization [2, 3].

This naturally motivates a data-centric perspective in reinforcement learning, where the goal is to evaluate and curate trajectories to improve learning outcomes. Data-centric learning has recently emerged as a promising direction for improving model performance by assessing the contribution of individual samples to a target objective. A central task in this paradigm is data valuation, which aims to quantify the utility of each sample [4–6]. A straightforward approach is to measure performance differences with and without a sample, but such retraining-based methods are computationally prohibitive. Influence functions provide an efficient alternative by estimating sample influence without retraining, but introduce new challenges in approximating the inverse Hessian. Prior work has developed various approximations, including stochastic recursive methods, random projections, Kronecker-factored eigendecomposition, and low-rank approximations, as well as Hessian-free variants for scalability [7, 8]. While these methods have been extensively studied in classification, their extension to reinforcement learning remains limited.

Extending influence-based data valuation from classification to reinforcement learning is non-trivial. Classification relies on fixed training and validation sets, enabling valuation through validation performance, whereas reinforcement learning involves dynamically generated trajectories without a fixed validation set. Moreover, trajectory utility in reinforcement learning is inherently policy-dependent and shaped by temporal credit assignment and reward dynamics, making it difficult to define consistent impact measures. Finally, reinforcement learning requires dynamic, online valuation as training progresses, raising additional challenges in efficiency.

Recent work [3, 9–11] has taken initial steps toward data valuation and selection in reinforcement learning. Existing methods rely on replay buffers and surrogate impact functions, verifiable feedback, or trusted validation gradients to estimate data utility. However, these setting-specific designs raise concerns about their generalizability across different reinforcement learning paradigms.

Contributions. In this paper, we study trajectory valuation in reinforcement learning, aiming to prevent detrimental trajectories from model updates. We summarize our key contributions as follows:

• We provide a principled formulation for trajectory valuation in general reinforcement learning settings, addressing the challenges of dynamic data generation, the absence of validation signals, and policy-dependent utility within a unified framework. This formulation establishes a unified perspective for trajectory valuation across diverse reinforcement learning pipelines.

• We propose Dynamic Trajectory Valuation (DTV), a simple and efficient method that dynamically identifies and filters detrimental trajectories at the mini-batch level. Our method operates purely on gradients and integrates seamlessly with existing reinforcement learning pipelines.

• We conduct extensive experiments to demonstrate the effectiveness of DTV across diverse reinforcement learning settings, including Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO). The results show consistent performance improvements, enhanced data efficiency, and more stable optimization, validating the benefits of dynamic trajectory valuation in practice.

## 2 Related Work

Here, we review prior work on data-centric learning in reinforcement learning and influence function-based methods, and position our approach in relation to existing literature.

## 2.1 Data-Centric Learning in Reinforcement Learning

A line of work focuses on improving sample efficiency by prioritizing or filtering training data [2, 12–14], most commonly at the level of individual transitions (i.e., experiences), as in prioritized experience replay, or via structured data scheduling such as curriculum learning. These approaches implicitly acknowledge that different data points contribute unequally to learning. However, they primarily operate on local signals—such as temporal-difference error or immediate rewards—to estimate importance, without explicitly modeling the contribution of entire trajectories. As a result, they may fail to capture long-range dependencies and the global effect of a trajectory on policy improvement.

Moving beyond transition-level selection, recent approaches in reinforcement learning from human feedback operate directly at the level of complete trajectories or response sequences [15, 16]. Methods such as PPO [17], GRPO [18], and DPO [19] treat each rollout as a training unit, where the quality of entire outputs determines the learning signal. In practice, data quality is often controlled via heuristic strategies such as rejection sampling or reward thresholding [20, 21]. While effective in large-scale systems, these approaches still rely on coarse signals and lack a principled framework for evaluating trajectories based on their contribution to learning.

Recent work [3, 9–11] has begun to explore the adaptation of data valuation and selection in reinforcement learning<sup>1</sup>. IIF [9] estimates data influence using replay buffers and algorithm-specific surrogate objectives; LearnAlign [10] weights gradient alignment by learnability derived from verifiable ground-truth feedback; and GradAlign [11] evaluates candidate updates against gradients from a trusted validation set. While this line of work represents a promising direction, existing methods often rely on auxiliary signals, which may limit their general applicability across different reinforcement learning paradigms.

Taken together, existing approaches either rely on local heuristics at the transition level, employ coarse filtering strategies, or approximate reinforcement learning through supervised surrogates. A unified and principled framework for trajectory-level data valuation remains lacking.

## 2.2 Influence Function

Influence functions, a mainstream tool for data valuation, were originally developed in robust statistics [22–24] to quantify the sensitivity of model parameters to perturbations in training data. They were introduced to the machine learning community by Koh and Liang [4] and have since been widely used to estimate sample influence with respect to validation performance.

To address the high computational cost of influence functions, several works propose efficient approximations of the inverse Hessian. LiSSA [4] leverages Hessian–vector products for stochastic approximation, while EKFAC [7] exploits structured eigenvalue decompositions. DataInf [5] further simplifies influence estimation by leveraging a rank-1 structure of the empirical Hessian, and TRAK [6] employs random projections to obtain a tractable kernel-based approximation. More recently, Hessian-free influence methods have been proposed, which approximate the inverse Hessian with an identity matrix for scalability [8, 25–27].

Beyond classical influence formulations, several recent works revisit influence estimation from an optimization-aware and non-convex perspective. SGD-Influence [28] tracks the stochastic gradient descent trajectory and estimates the effect of removing a training sample by reusing intermediate iterates, thereby relaxing the local convexity assumption. MoSo [29] models data pruning by approximating a moving-one-sample-out update along the optimization path, while Z0-Inf [30] develops a zeroth-order estimator that avoids explicit gradients and Hessian computations.

Beyond static estimation, dynamic data valuation has recently attracted increasing attention [31, 32]. Early approaches approximate dynamic influence by aggregating influence estimates across training checkpoints [6, 7, 26]. However, such aggregation fails to capture the evolving nature of influence and may cancel out conflicting signals. More recent work directly estimates data value at each training step, enabling truly dynamic valuation [31, 32].

In this paper, rather than adapting influence functions from classification settings, we generalize them to reinforcement learning and develop a principled formulation for trajectory-level valuation.

## 3 Preliminaries

Below, we elaborate on the preliminaries in terms of reinforcement learning and influence functions.

## 3.1 Reinforcement Learning

A standard reinforcement learning framework is modeled as a Markov Decision Process, defined by a tuple $( S , { \mathcal { A } } , P , r , \gamma )$ , where S and $\mathcal { A }$ denote the state and action spaces, $P ( s ^ { \prime } \mid s , a )$ is the transition dynamics, $r ( s , a )$ is the reward function, and $\gamma \in [ 0 , 1 )$ is the discount factor. A policy $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { s } )$ , parameterized by θ, maps states to action distributions. The goal of reinforcement learning is to maximize the expected discounted return:

$$
J ( \theta ) = \mathbb { E } _ { \tau \sim \pi _ { \theta } } \left[ \sum _ { t = 0 } ^ { T } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \right] ,\tag{1}
$$

where $\tau = ( s _ { 0 } , a _ { 0 } , \ldots , s _ { T } )$ denotes a trajectory.

A common approach to optimizing this objective is via policy gradient methods, which estimate the gradient of $J ( \theta )$ as:

$$
\nabla _ { \theta } J ( \theta ) = \mathbb { E } _ { \tau \sim \pi _ { \theta } } \left[ \sum _ { { t = 0 } } ^ { T } \nabla _ { \theta } \log \pi _ { \theta } ( a _ { t } \mid s _ { t } ) \cdot A ^ { \pi } ( s _ { t } , a _ { t } ) \right] ,\tag{2}
$$

where $A ^ { \pi } ( s , a )$ is the advantage function, measuring the relative contribution of an action compared to a baseline policy. In practice, optimizing this objective in high-dimensional settings requires stable and scalable algorithms. Importantly, many modern approaches operate at the level of complete trajectories, where each rollout— $- \mathrm { i . e . }$ , a trajectory generated by the policy during interaction with the environment—serves as a fundamental training unit. We next introduce several representative methods in this regime, which form the basis of our study.

Proximal Policy Optimization (PPO). Proximal Policy Optimization (PPO) [17] is a widely used policy gradient method that stabilizes training via a clipped surrogate objective. Let $r _ { t } ( \theta ) { = } { \pi } _ { \theta } ( a _ { t } ~ \mid ~ s _ { t } ) / { \pi } _ { \theta _ { \mathrm { o l d } } } ( a _ { t } ~ \mid ~ s _ { t } )$ denote the importance sampling ratio. The PPO objective is:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { P P O } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } \left( \boldsymbol { r } _ { t } ( \boldsymbol { \theta } ) A _ { t } , \ \mathrm { c l i p } ( \boldsymbol { r } _ { t } ( \boldsymbol { \theta } ) , 1 - \epsilon , 1 + \epsilon ) A _ { t } \right) \right] . } \end{array}\tag{3}
$$

This clipping mechanism prevents excessively large policy updates and improves training stability.

Group Relative Policy Optimization (GRPO). Group Relative Policy Optimization (GRPO) [18] extends PPO by replacing absolute advantage estimates with relative advantages computed within a group of sampled trajectories. Given a group $\mathcal { G } { = } \{ \tau _ { i } \} _ { i = 1 } ^ { K }$ , GRPO normalizes rewards within the group by $\tilde { A } _ { i } { = } ( r _ { i } - \mu _ { \mathcal { G } } ) / \sigma _ { \mathcal { G } }$ , where $\mu _ { \mathcal { G } }$ and $\sigma _ { \mathcal { G } }$ are the mean and standard deviation of rewards in the group. The objective of GRPO becomes:

$$
\mathcal { L } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { i , t } \left[ \operatorname* { m i n } \left( r _ { i , t } ( \theta ) \tilde { A } _ { i } , \ \mathrm { c l i p } ( r _ { i , t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) \tilde { A } _ { i } \right) \right] .\tag{4}
$$

By leveraging relative comparisons, GRPO reduces reliance on value function estimation and improves robustness in scenarios such as RLHF.

Direct Preference Optimization (DPO). Direct Preference Optimization (DPO) [19] formulates policy learning directly from preference data without explicit reward modeling. Given a pair of responses $( y ^ { + } , y ^ { - } )$ for a prompt x, where $y ^ { + }$ is preferred over $y ^ { - }$ , DPO optimizes:

$$
\mathcal { L } _ { \mathrm { D P O } } ( \theta ) = - \mathbb { E } _ { ( x , y ^ { + } , y ^ { - } ) } \left[ \log \sigma \left( \beta \left( \log \frac { \pi _ { \theta } ( y ^ { + } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { + } \mid x ) } - \log \frac { \pi _ { \theta } ( y ^ { - } \mid x ) } { \pi _ { \mathrm { r e f } } ( y ^ { - } \mid x ) } \right) \right) \right] ,\tag{5}
$$

where $\pi _ { \mathrm { r e f } }$ is a reference policy and $\beta$ controls the sharpness. DPO can be interpreted as implicitly optimizing a KL-regularized reward objective while bypassing explicit reward model training.

In sum, PPO provides a stable on-policy optimization framework, GRPO enhances it with relative trajectory comparisons, and DPO extends RL to preference-based optimization without explicit rewards. These methods form the backbone of modern reinforcement learning, especially in large-scale language model post-training.

## 3.2 Influence Function

The effect of an individual training sample can be characterized by infinitesimally perturbing its contribution to the training objective and tracing the resulting change in model behavior. Given a model parameterized by $\theta ,$ , let z denote an individual training sample and the empirical risk minimization objective be defined as $\scriptstyle { \hat { \theta } } = \arg$ min<sub>θ∈Θ</sub> $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \ell \big ( z _ { i } ; \theta \big ) } \end{array}$ Following the classical influence-function formulation of Koh and Liang [4], the effect of removing a training sample $z _ { j }$ on a validation objective can be approximated by infinitesimally perturbing its contribution to the training objective. This yields the following influence score:

$$
\boldsymbol { \mathcal { T } } ( \boldsymbol { z } _ { j } ; \hat { \boldsymbol { \theta } } ) = \nabla f ( \boldsymbol { \mathcal { V } } ; \hat { \boldsymbol { \theta } } ) ^ { \top } \mathbf { H } _ { \hat { \boldsymbol { \theta } } } ^ { - 1 } \nabla \boldsymbol { \ell } ( \boldsymbol { z } _ { j } ; \hat { \boldsymbol { \theta } } ) ,\tag{6}
$$

where V denotes the validation set, $f ( \cdot )$ represents the impact function that specifies the evaluation objective on V, $\nabla \ell ( z _ { j } ; \boldsymbol { \hat { \theta } } )$ denotes the gradient contribution of sample $z _ { j }$ , and $\begin{array} { r l } {  { \mathbf { H } _ { \hat { \boldsymbol { \theta } } } = \sum _ { i = 1 } ^ { n } \nabla ^ { 2 } \ell ( z _ { i } ; \hat { \boldsymbol { \theta } } ) } } & { { } } \end{array}$ is the Hessian matrix. This formulation provides a principled measure of how individual training samples affect model behavior through their contribution to the optimization process.

## 4 Method

In this section, we first highlight three critical challenges when adapting traditional influence functions to reinforcement learning. We then present Dynamic Trajectory Valuation (DTV), a unified framework that overcomes these issues and seamlessly integrates across various reinforcement learning paradigms. Lastly, we offer an in-depth analysis of DTV by exploring its trajectory decomposition and leave-one-out extension.

## 4.1 Challenges of Influence Functions in Reinforcement Learning

Influence functions, as a crucial tool for data-centric learning, have been extensively studied in supervised learning [4]; however, their application to reinforcement learning remains largely underexplored. Extending these principles to reinforcement learning is non-trivial due to several fundamental differences:

• Input Trajectories: Supervised learning assumes fixed training and validation datasets, enabling data valuation through each sample’s influence on validation performance. In contrast, reinforcement learning generates trajectories online through environment interactions, with no fixed training set for valuation, rendering conventional valuation methods inapplicable.

• Impact Function: Classical influence functions rely on an explicit evaluation function defined on a validation set to quantify the contribution of individual samples. In reinforcement learning, trajectory usefulness is inherently policy-dependent and shaped by temporal credit assignment and reward dynamics, making such definitions elusive.

• Dynamic Mechanism: Conventional data valuation typically relies on static, post-hoc analysis. Reinforcement learning, however, requires dynamic, online valuation as trajectories evolve, raising the challenge of estimating trajectory impact efficiently without incurring prohibitive computational overhead.

Recent efforts [3, 9–11] extend data valuation and selection to reinforcement learning. Existing methods rely on settingspecific designs, including replay buffers and surrogate impact functions for PPO, verifiable-feedback-based learnability, and trusted validation gradients. However, they lack a unified and principled formulation across reinforcement learning paradigms, raising concerns about their generalizability.

## 4.2 Dynamic Trajectory Valuation

In this paper, we consider a principled formulation for general reinforcement learning by addressing the above three challenges in a unified framework. If we take a close look at Eq. (6), the first and second terms are constants across candidate samples, even given the unknown validation set and impact function. Therefore, the sample gradient is decisive for trajectory valuation, which casts the trajectory valuation problem into a sample gradient analysis problem. Suppose that learning is effective, beneficial samples should constitute the majority within a mini-batch, which leads to the following assumption.

Assumption 1 (Majority Gradient Alignment). Within a mini-batch, the majority of sample gradients collectively provide a locally informative reference direction for the current update. Accordingly, a sample whose gradient is negatively aligned with this reference direction is regarded as detrimental to the current optimization step.

Under Assumption 1, we propose Dynamic Trajectory Valuation (DTV) to dynamically identify detrimental training units during optimization. A training unit denotes the basic entity for which a per-unit gradient contribution is computed, with its form depending on the underlying learning paradigm: a trajectory segment in PPO, a completion trajectory generated for a prompt in GRPO, and a prompt with a chosen–rejected response pair in DPO. Given a mini-batch $B { = } \{ { z } _ { j } \} _ { j { = } 1 } ^ { b } , \operatorname { l e t } g _ { j } ^ { ( t ) } { = } \nabla _ { \theta } \ell ( { z } _ { j } { ; } \theta ^ { ( t ) } )$ denote the gradient contribution of training unit $z _ { j }$ , where $\ell ( z _ { j } ; \theta )$ is the training loss under the underlying algorithm. We define the DTV score as

$$
\mathcal { T } ^ { \mathrm { D T V } } ( \boldsymbol { z } _ { j } ; \boldsymbol { \theta } ^ { ( t ) } ) = \frac { 1 } { b } \sum _ { i = 1 } ^ { b } \bigl ( g _ { i } ^ { ( t ) } \bigr ) ^ { \top } g _ { j } ^ { ( t ) } .\tag{7}
$$

DTV requires no separate impact function or predefined validation set. Instead, its reference direction is derived directly from the current mini-batch gradients. Training units with negative DTV scores are identified as alignment-opposing and filtered out dynamically before executing the optimizer update.

Intuition and Dynamics. Measuring alignment in the gradient space directly reflects how each training unit impacts parameter trajectory, avoiding heuristic assumptions in the original input/feature space. While prior work [33] considers gradient-based outlier analysis for static data valuation in classification (e.g., using Isolation Forests), DTV addresses the dynamic nature of reinforcement learning. By leveraging the majority gradient as an online reference, DTV naturally adapts to evolving policy checkpoints without requiring global convergence or local convexity assumptions [31, 32].

Computational Efficiency. A key advantage of our DTV is that it enables direct and efficient valuation without additional gradient approximations. Prior implementations rely on batch-level gradients and require surrogate techniques such as ghost influence or layer-wise estimation [31, 32]. In contrast, our JAX [34] implementation leverages vectorized automatic differentiation to directly compute training-unit-level gradients, enabling efficient evaluation of Eq. (7) and seamless integration into diverse reinforcement learning pipelines.

## 4.3 DTV Decomposition and Leave-One-Out Extension

To clarify the information captured by DTV, we decompose the score in Eq. (7) into the contribution from the evaluated training unit itself and its interactions with the remaining mini-batch. Expanding the mini-batch mean gives $\begin{array} { r } { \mathcal { T } ^ { \mathrm { D T V } } ( z _ { j } ; \theta ^ { ( t ) } ) = \frac { 1 } { b } \| g _ { j } ^ { ( t ) } \| ^ { 2 } + \frac { 1 } { b } \sum _ { i \neq j } ( g _ { j } ^ { ( t ) } ) ^ { \top } g _ { i } ^ { ( t ) } = s _ { j } ^ { ( t ) } + c _ { j } ^ { ( t ) } } \end{array}$ , where $s _ { j } ^ { ( t ) } { = } \frac { 1 } { b } \| g _ { j } ^ { ( t ) } \| ^ { 2 }$ and $\begin{array} { r } { c _ { j } ^ { ( t ) } { = } \frac { 1 } { b } \sum _ { i \neq j } ( g _ { j } ^ { ( t ) } ) ^ { \top } g _ { i } ^ { ( t ) } } \end{array}$ denote the self-term and cross-term, respectively. The self-term captures self-alignment, whereas the cross-term measures aggregate gradient agreement with the remaining training units.

Remark 1 (Self-Protection Effect). Since $s _ { j } ^ { ( t ) } { \geq } 0 ,$ , DTVfilters a training unit only when $c _ { j } ^ { ( t ) } < - s _ { j } ^ { ( t ) }$ . A sufficiently large self-term can therefore offset a negative cross-term and retain a training unit even when it is strongly negatively aligned with the remaining mini-batch, an effect we refer to as self-protection. Nevertheless, the self-term should not be interpreted as universally detrimental, as its effect may depend on the underlying optimization setting.

The self-protection effect motivates a leave-one-out extension that isolates the cross-term by excluding the gradient itself from the reference direction. Specifically, we average over the remaining b−1 units and define DTV-Loo as

$$
\mathcal { T } ^ { \mathrm { { D T V - L o o } } } ( z _ { j } ; \theta ^ { ( t ) } ) = \frac { 1 } { b - 1 } \sum _ { i \neq j } \bigl ( g _ { i } ^ { ( t ) } \bigr ) ^ { \top } g _ { j } ^ { ( t ) } .\tag{8}
$$

Using the decomposition above, DTV-Loo can equivalently be written as $\scriptstyle { \mathcal { T } } ^ { \mathrm { D T V - L o o } } = \frac { b } { b - 1 } \left( \mathcal { T } ^ { \mathrm { D T V } } - s _ { j } ^ { ( t ) } \right)$ . Consequently, DTV-Loo removes the self-term while maintaining exact leave-one-out normalization. Its valuation thus relies solely on the cross-term, effectively preventing a single harmful sample from dominating the mini-batch evaluation.

More generally, DTV and DTV-Loo can be unified into a broader DTV-λ family $( \lambda \in [ 0 , 1 ] )$ by weighting the self-term. Here, λ=1 recovers DTV, whereas λ=0 completely removes the self-term to yield DTV-Loo. Despite this broader parameterization, we retain DTV and DTV-Loo as the practical instantiations, as both already achieve strong empirical performance without requiring additional tuning of λ. Empirically, DTV-Loo performs better than DTV in most evaluated settings, whereas DTV can be more effective under specific regimes, demonstrating their complementary strengths across different optimization scenarios and providing practical guidelines for method selection. Algorithm 1 summarizes the filtering procedure.

Algorithm 1 Dynamic Training-Unit Filtering with DTV and DTV-Loo   
Input: Parameters $\theta ^ { ( 0 ) }$ ; steps T; valuation method M ∈ {DTV, DTV-Loo}; per-unit objective ℓ   
Output: Updated model parameters $\theta ^ { ( T ) }$   
1: for $t = \mathsf { \bar { 0 } } , \ldots , T - 1$ do   
2: Obtain training units ${ B ^ { ( t ) } = } \{ z _ { j } \} _ { j = 1 } ^ { b }$   
3: Compute $g _ { j } ^ { ( t ) } \gets \nabla _ { \theta } \ell ( z _ { j } ; \theta ^ { ( t ) } )$ for all $z _ { j } \in B ^ { ( t ) }$   
4: Compute valuation scores $\mathcal { T } _ { j } ^ { ( t ) } \gets \mathcal { T } ^ { M } ( z _ { j } ; \theta ^ { ( t ) } )$ for all $j$   
5: Retain $B _ { + } ^ { ( t ) } \gets \{ z _ { j } \in B ^ { ( t ) } : \mathcal { T } _ { j } ^ { ( t ) } \geq 0 \}$   
6: Update $\theta ^ { ( t + 1 ) } \gets \mathrm { U p d a t e } \big ( \theta ^ { ( \bar { t } ) } , \beta _ { + } ^ { ( t ) } \big )$   
7: end for   
8: return $\theta ^ { ( T ) }$

## 5 Experiments

To evaluate the versatility and practical applicability of DTV and DTV-Loo, we apply the proposed methods across diverse reinforcement learning and LLM post-training settings. Specifically, we study online trajectory optimization with PPO, group-relative optimization with GRPO, and pairwise preference optimization with DPO, spanning different training-unit granularities, including trajectory segments, completion groups, and preference pairs. Beyond evaluating final performance, we further investigate the self-protection effect underlying the distinction between DTV and DTV-Loo, optimization dynamics, and data efficiency. All experiments follow a JAX-based [34] implementation, with the GRPO and DPO pipelines further built on Google’s open-source Tunix library [35], which we extend with vectorized training-unit-level gradient computation and dynamic filtering.

## 5.1 DTV for Online Trajectory Valuation with PPO

Experimental Setup. We compare DTV and DTV-Loo with Vanilla PPO and Iterative Influence-Based Filtering (IIF) [9] in the PPO setting. Vanilla PPO performs standard PPO updates without filtering. IIF serves as an influence-based filtering baseline by estimating the impact of individual transitions. In contrast, DTV and DTV-Loo perform adaptive trajectory-level filtering, where rollouts are partitioned into trajectory-level training units, with boundaries determined by episode termination or the rollout boundary. Following Algorithm 1, trajectory units with non-negative valuation scores are retained, while filtering a unit excludes all of its constituent transitions from the subsequent PPO update.

![](images/5b9d4bdedba1be530e20685ab39547243dea990deea871befe83934de3dad310.jpg)  
(a) Empty-8x8

We conduct PPO experiments on two partially observable MiniGrid environments [36]. Empty-8x8 follows the setting used by IIF [9], enabling a direct comparison on sparsereward navigation. DoorKey-8x8, in contrast, requires the agent to collect a key, unlock a door, and reach the goal, introducing longer-horizon dependencies and harder exploration. We report the mean over five seeds, with shaded regions indicating ±1 standard error of the mean (SEM); each checkpoint is evaluated over 1, 000 episodes. Full architectures, hyperparameters, and filtering configurations are provided in Appendix B.

Main Results. Figure 2(a)–(b) show that DTV-based trajectory filtering accelerates PPO convergence across both MiniGrid tasks. On Empty-8x8, DTV-Loo converges substantially earlier than the other methods, although all methods ultimately attain comparable returns. On the more challenging DoorKey-8x8, DTV-Loo exhibits slightly faster early-stage convergence than DTV; thereafter, both variants approach the reward ceiling and maintain a clear advantage over IIF and Vanilla PPO under the same interaction budget. Notably, the performance margin over both baselines is substantially larger on DoorKey-8x8, unders gains achieved by DTV-based filtering in the more challenging environment.

![](images/775387ccb83160ffdc87f2ee24141cc4cc4c453065dcf11c2bc8b9d8c239e4f1.jpg)  
(b) DoorKey-8x8  
Figure 1: MiniGrid tasks.

![](images/67de22d13b4ceaeeff82e2efc9f60af53da23269e3f02ea8338e34dc9f52903a.jpg)  
(a) Empty-8x8

![](images/876454ce065ebfb6cc184366362f4e4268156341a764b30da99e66cd4c5a3f37.jpg)  
(b) DoorKey-8x8

![](images/00eb03792a59b47a65e3407f69ba55b70e03ba779f778dfb5770b180b92ddd08.jpg)  
(c) Worst/Best returns  
Figure 2: PPO results on MiniGrid. (a–b) Mean test reward over PPO rounds across five seeds; shaded regions show ±1 SEM. (c) At the final checkpoint, the lowest and highest 20 returns from 1,000 evaluation episodes per seed are pooled across five seeds as the Worst and Best subsets.

At the final checkpoint, Figure 2(c) further shows that DTV-Loo yields higher returns than IIF in both the Worst and Best subsets across the two environments. These improvements are consistent with the gradient-alignment perspective in Section 4: by removing trajectory units that conflict with the dominant update direction, DTV-based filtering reduces detrimental optimization interference while preserving useful training signals. As a result, the policy achieves more stable improvement despite using fewer effective training units. Together, these results indicate that DTV-based trajectory filtering accelerates PPO convergence and improves the consistency of final-policy performance.

Overall, the PPO results establish two consistent benefits of DTV-based filtering: faster convergence and improved final-policy performance. The convergence advantage is further supported in our later DPO experiments, where DTV-Loo reaches the same performance target substantially faster. The performance benefit is also observed across the subsequent GRPO and DPO settings. Given the extensive prior study of PPO, we focus on two relatively standard benchmark settings here. We reserve a deeper analysis of the mechanisms underlying DTV and DTV-Loo for the more complex GRPO and DPO experiments that follow.

## 5.2 DTV for Group-Relative Completion Valuation with GRPO

Next, we examine DTV and DTV-Loo in the GRPO setting on two mathematical reasoning settings of different difficulty, ranging from grade-school reasoning on GSM8K [37] to more challenging competition-level reasoning on the American Invitational Mathematics Examination (AIME). GRPO generates multiple completion trajectories for each prompt and optimizes them using group-relative advantages, providing a natural setting for completion-level valuation and filtering. Across both benchmarks, DTV and DTV-Loo use policy-loss gradients to value individual completion trajectories relative to other completions generated for the same prompt. We further analyze the role of the self-term on GSM8K to better understand the performance differences between DTV and DTV-Loo under group-relative optimization.

Experimental Setup. We evaluate DTV and DTV-Loo against Vanilla GRPO and two fixed-ratio filtering baselines. Vanilla GRPO performs the standard group-relative update without filtering. All filtering methods operate on individual completions within each prompt group. Random Filtering selects completions independently of their reward signals, whereas Reward-based Filtering removes those with the lowest observed group-relative advantages. DTV and DTV-Loo instead determine which completions to retain adaptively based on within-group policy-gradient alignment, without prescribing a filtering rate. On GSM8K, we conduct the full comparison with Random and Reward-based Filtering at expected rates of 5% and 10%. On the substantially more computationally demanding AIME setting, we focus on the three core methods, Vanilla GRPO, DTV, and DTV-Loo.

For GSM8K [37], we fine-tune Gemma-3-1B-IT [38] with LoRA under Clean and Mismatch-20%, where the latter perturbs the within-group reward ranking while keeping evaluation data clean. We restrict the main corruption study to Mismatch-20%, since extending the same rank-reversal perturbation to 40% of prompt groups would substantially distort the within-group relative signal that directly drives the GRPO update. All methods use the same training budget and are evaluated over five matched seeds on the full GSM8K test set. We report exact-answer accuracy (Acc.), the improvement over the shared pre-trained model (∆ Acc.), and partial accuracy (Partial), which counts numerical predictions within 10% relative error of the ground-truth answer.

For the more challenging AIME 2024 [39], we full-parameter fine-tune DeepSeek-R1-Distill-Qwen-1.5B [40] on the DeepScaleR-Preview-Dataset [40]. Each optimization step samples eight completions per prompt, with binary mathematical correctness used as the reward for group-relative advantage estimation. Training uses a maximum response length of 8,192 tokens, whereas final evaluation uses 32,768 tokens to reduce truncation, with 16 responses sampled for each of the AIME 2024 problems. We report Avg@16, the average correctness over all sampled responses; Pass@16, the fraction of problems solved by at least one response; Maj@16, the majority-vote accuracy over the 16 responses; Extractable, the fraction of responses with a successfully parsed final answer; and Acc. given extractable, the accuracy conditioned on successful answer extraction. Full training, generation, filtering, and evaluation details for both benchmarks are provided in Appendix C.

Table 1: GRPO results on GSM8K under clean and mismatch settings. ∆ Acc. is measured relative to the shared pre-trained model. Results are reported as mean ± std. over five seeds, with the best result shown in bold. Given the consistent margins, we omit additional pairwise t-tests.
<table><tr><td rowspan="2">Method</td><td colspan="3">Clean</td><td colspan="3">Mismatch-20%</td></tr><tr><td>Acc. ↑</td><td> $\Delta \operatorname { A c c . } \uparrow$ </td><td>Partial ↑</td><td> $\operatorname { A c c . \uparrow }$ </td><td>∆ Acc. ↑</td><td>Partial ↑</td></tr><tr><td>Pre-trained</td><td> $0 . 4 7 2 3 \pm 0 . 0 0 0 0$ </td><td>0.0000</td><td> $0 . 4 9 9 6 \pm 0 . 0 0 0 0$ </td><td> $0 . 4 7 2 3 \pm 0 . 0 0 0 0$ </td><td>0.0000</td><td> $0 . 4 9 9 6 \pm 0 . 0 0 0 0$ </td></tr><tr><td>Vanilla GRPO</td><td> $0 . 4 8 2 8 \pm 0 . 0 0 7 7$ </td><td>0.0105</td><td> $0 . 5 1 1 4 \pm 0 . 0 0 8 9$ </td><td> $0 . 4 7 3 1 \pm 0 . 0 0 9 4$ </td><td>0.0008</td><td> $0 . 5 0 0 1 \pm 0 . 0 0 8 9$ </td></tr><tr><td>Random 5%</td><td> $0 . 5 2 0 7 \pm 0 . 0 3 4 1$ </td><td>0.0484</td><td> $0 . 5 4 4 2 \pm 0 . 0 3 0 7$ </td><td> $0 . 4 1 0 3 \pm 0 . 2 0 2 6$ </td><td>-0.0620</td><td> $0 . 4 2 5 5 \pm 0 . 2 0 8 6$ </td></tr><tr><td>Random 10%</td><td> $0 . 4 5 1 1 \pm 0 . 1 1 7 5$ </td><td>-0.0212</td><td> $0 . 4 7 6 0 \pm 0 . 1 1 2 7$ </td><td> $0 . 5 1 3 3 \pm 0 . 0 4 3 1$ </td><td>0.0409</td><td> $0 . 5 3 1 8 \pm 0 . 0 4 5 5$ </td></tr><tr><td>Reward 5%</td><td> $0 . 4 5 1 6 \pm 0 . 2 1 0 4$ </td><td>-0.0208</td><td> $0 . 4 7 2 9 \pm 0 . 2 1 8 1$ </td><td> $0 . 4 4 4 3 \pm 0 . 1 6 9 1$ </td><td>-0.0281</td><td> $0 . 4 6 2 8 \pm 0 . 1 7 3 3$ </td></tr><tr><td>Reward 10%</td><td> $0 . 4 9 7 3 \pm 0 . 1 1 2 8$ </td><td>0.0250</td><td> $0 . 5 1 9 3 \pm 0 . 1 0 9 4$ </td><td> $0 . 3 7 7 7 \pm 0 . 1 9 8 1$ </td><td>-0.0946</td><td> $0 . 4 0 1 2 \pm 0 . 2 0 0 0$ </td></tr><tr><td>LearnAlign (2026)</td><td> $0 . 5 0 1 4 \pm 0 . 0 1 7 0$ </td><td>0.0291</td><td> $0 . 5 2 8 1 \pm 0 . 0 1 6 0$ </td><td> $0 . 4 8 6 3 \pm 0 . 0 0 6 9$ </td><td>0.0140</td><td> $0 . 5 1 3 0 { \scriptstyle \pm 0 . 0 0 8 3 }$ </td></tr><tr><td>GradAlign (2026)</td><td> $0 . 4 8 7 6 \pm 0 . 0 0 4 0$ </td><td>0.0153</td><td> $0 . 5 1 5 4 \pm 0 . 0 0 4 6$ </td><td> $0 . 4 7 7 5 \pm 0 . 0 0 6 4$ </td><td>0.0052</td><td> $0 . 5 0 6 0 \pm 0 . 0 0 5 1$ </td></tr><tr><td>DTV (Ours)</td><td> $\mathbf { 0 . 5 4 8 3 \pm 0 . 0 0 7 5 }$ </td><td>0.0760</td><td> $\mathbf { 0 . 5 7 4 2 \pm 0 . 0 0 9 9 }$ </td><td> $\mathbf { 0 . 5 4 1 3 \pm 0 . 0 1 3 9 }$ </td><td>0.0690</td><td> $\mathbf { 0 . 5 6 6 0 \pm 0 . 0 1 2 3 }$ </td></tr><tr><td>DTV-Loo (Ours)</td><td> $0 . 5 4 1 6 \pm 0 . 0 1 4 1$ </td><td>0.0693</td><td> $0 . 5 6 4 7 \pm 0 . 0 1 4 6$ </td><td> $0 . 5 2 3 4 \pm 0 . 0 1 0 3$ </td><td>0.0511</td><td> $0 . 5 4 6 0 \pm 0 . 0 0 7 6$ </td></tr></table>

![](images/25644ff9ef13da3c1b9e3469b80bbc16804ffd7862278445af90769f83bfce81.jpg)  
(a) Score decomposition

![](images/2256526d83818549ca764de927ebb5683f011c4998adf74181a9df52ca01910f.jpg)  
(b) Filtering regions

![](images/013dd569b3a38e60156fb89fbd524371f0eef8c48440f5276c376de7cb0f076c.jpg)  
(c) Self-protected conflicts

![](images/f4385dada3e914e61beeffc6851737b10e4ceea4823033409db8fedff77f7fe7.jpg)  
(d) Cross-term contribution

![](images/0126ad5c867713d265cd268f091c26c4dea539f3e4c097d1361a5d8b8e8b6ab1.jpg)  
(e) Conflicted-unit cross-term magnitude

![](images/e6c68515f5f060227dd9bc39f11e7324a010f642d7acf5b1909fd4f77d1c4c92.jpg)  
(f) Drop ratio  
Figure 3: Empirical analysis of the self- and cross-term effect on GSM8K under Mismatch-20%. We define conflicted units as training units for which DTV and DTV-Loo make different keep/drop decisions; R denotes the fraction of units retained for visualization after trimming extreme score ranges, and IQR denotes the interquartile range (25th–75th percentiles). (a) Decomposition of the DTV score into the self- and cross-terms. (b) Filtering regions of DTV and DTV-Loo. (c) Self- and cross-term scores of conflicted units. (d) Cross-term contribution $\lvert c _ { j } \rvert / ( s _ { j } + \lvert c _ { j } \rvert )$ for all and conflicted units. (e) Cross-term magnitude of conflicted units. (f) Drop ratios of DTV and the additional filtering induced by DTV-Loo.

Table 2: GRPO results on AIME 2024 with a 32K maximum generation length. ∆ denotes the change relative to the shared pre-trained model. Results are from a single training run for each method, with the best result shown in bold.
<table><tr><td></td><td colspan="2">Pass@1</td><td colspan="2">Pass@16</td><td colspan="2">Maj@16</td><td rowspan="2">Extractable ↑</td><td rowspan="2">Acc. given extractable ↑</td></tr><tr><td>Method</td><td>Score ↑</td><td>Δ↑</td><td>Score ↑</td><td>Δ↑</td><td>Score ↑</td><td>Δ↑</td></tr><tr><td>Pre-trained</td><td>0.1063</td><td>0.0000</td><td>0.3333</td><td>0.0000</td><td>0.2333</td><td>0.0000</td><td>0.7292</td><td>0.1457</td></tr><tr><td>Vanilla GRPO</td><td>0.1063</td><td>0.0000</td><td>0.4000</td><td>0.0667</td><td>0.2333</td><td>0.0000</td><td>0.7375</td><td>0.1441</td></tr><tr><td>DTV (Ours)</td><td>0.0917</td><td>-0.0146</td><td>0.3333</td><td>0.0000</td><td>0.2667</td><td>0.0334</td><td>0.7188</td><td>0.1275</td></tr><tr><td>DTV-Loo (Ours)</td><td>0.1333</td><td>0.0270</td><td>0.4333</td><td>0.1000</td><td>0.3667</td><td>0.1334</td><td>0.7521</td><td>0.1773</td></tr></table>

Results on GSM8K. Table 1 reports the GSM8K results under Clean and Mismatch-20%. DTV achieves the best exact and partial accuracy in both settings, improving exact accuracy over the shared pre-trained model by 0.0760 under Clean and 0.0690 under Mismatch-20%. DTV-Loo also improves over Vanilla GRPO in both settings, although DTV attains slightly higher average accuracy in both cases. In contrast, Random Filtering exhibits highly variable behavior across filtering ratios and data conditions, with the relative performance of 5% and 10% reversing between Clean and Mismatch-20%, consistent with its sensitivity to stochastic sample removal. Reward-based Filtering is likewise unstable and does not provide reliable gains, with several settings falling below the shared pre-trained model. The two gradient-alignment baselines, LearnAlign and GradAlign, provide modest but more stable gains than Random and Reward-based Filtering across both settings, while remaining below DTV and DTV-Loo on all metrics. Under Mismatch-20%, their gains further diminish, with exact-accuracy improvements of only 0.0140 and 0.0052, respectively, compared with 0.0690 for DTV. Overall, these results show that gradient-based adaptive filtering using only current mini-batch gradients yields stronger and more stable GRPO performance on GSM8K. By using the batch gradient as an endogenous reference, DTV removes trajectories that conflict with the current update direction, reducing detrimental gradient interference before the model update and improving optimization performance.

Unlike the baseline methods, which rely on prescribed selection or filtering budgets, our proposed DTV and DTV-Loo make filtering decisions directly using the zero valuation threshold, allowing the effective filtering ratio to adapt dynamically over training. Notably, DTV slightly outperforms DTV-Loo across all reported metrics, suggesting that retaining the self-term can remain beneficial in this setting. To understand this difference, we revisit the self- and cross-term decomposition and examine how the self-term affects the filtering decisions of DTV and DTV-Loo on GSM8K. We return to this distinction in the latter DPO analysis to further clarify when each variant is preferable.

Figure 3 provides a closer examination of the difference between DTV and DTV-Loo, i.e., the self- and cross-term effect on GSM8K. Figure 3(a) demonstrates that the self-term constitutes the primary component of the DTV score, whereas the cross-term remains comparatively small throughout training. This suggests that DTV may be susceptible to extreme samples that dominate the batch gradient, as the batch mean used in DTV lacks robustness to outliers. Although we do not observe this issue on GSM8K, we revisit it in the next subsection to highlight the necessity of DTV-Loo, which mitigates the self-term effect while remaining effective at filtering detrimental samples. In addition, as implied by the decomposition in Section 4.3, removing the non-negative self-term causes DTV-Loo to additionally filter units with negative cross-terms that DTV still retains. Figure 3(b)–(c) visualize these conflicted units and their corresponding self and cross-term scores. The remaining question is whether these additionally filtered units are sufficiently detrimental to move. Furthermore, Figure 3(d)–(e) show that conflicted units have relatively small cross-term contributions and that many of their negative cross-terms remain close to zero, indicating weak cross-unit conflicts and similar performance between DTV and DTV-Loo. Consequently, although DTV-Loo filters substantially more units through training, as shown in Figures 3(f), its performance remains close to DTV, suggesting strong data efficiency. Meanwhile, retaining these weakly conflicted units helps explain the slightly better performance of DTV in this setting.

Scalability and Generalization Test. To further evaluate the scalability and generalization of DTV and DTV-Loo, we consider a more complicated and large-scale AIME challenge. Specifically, unlike the GSM8K experiments with Gemma-3-1B-IT and LoRA, here we full-parameter fine-tune DeepSeek-R1-Distill-Qwen-1.5B using a JAX-based implementation on a dual-worker TPU v5p-16 setup, providing an additional evaluation across model backbones, fine-tuning regimes, and implementation stacks. All other experiments use a single TPU v5p-8 node, whereas AIME requires the dual-worker setup due to HBM constraints, introducing inter-worker communication and nondeterministic task assignment at startup. A single training seed requires over 140 hours for each method; we therefore conduct one training run for each method. Given this substantial cost and the ad hoc filtering-ratio choices required by Random and Reward, we omit these two baselines and focus on Vanilla GRPO, DTV, and DTV-Loo and report the results in Table 2.

Despite this demand setting, DTV-Loo achieves the best performance across all reported metrics. Relative to the shared pre-trained model, it improves Pass@1 from 0.1063 to 0.1333 (+0.0270), Pass@16 from 0.3333 to 0.4333 (+0.1000), and Maj@16 from 0.2333 to 0.3667 (+0.1334). It also attains the highest extractable-answer rate (0.7521)

and accuracy given an extractable answer (0.1773), indicating that the gains extend beyond aggregate reasoning accuracy to more reliable final-answer generation. In comparison, Vanilla GRPO and DTV show mixed improvements across the reasoning metrics, with DTV improving Maj@16 but not Pass@1 or Pass@16.

## 5.3 DTV for Pairwise Preference Valuation with DPO

We finally turn to DPO to examine whether DTV generalizes beyond online rollout-based optimization to pairwise preference learning. In addition to the clean setting, we introduce controlled preference corruption to evaluate robustness to increasingly unreliable supervision. We further examine time-to-target efficiency and analyze the self- and cross-term decomposition underlying the self-protection effect, providing complementary views of optimization efficiency and valuation behavior.

Experimental Setup. In the DPO setting, our comparison includes Vanilla DPO, Random Pair Filtering (Random), Reward-based Filtering (Reward), DTV, and DTV-Loo. Unlike online rollout-based optimization, DPO learns from prompt–response preference pairs, where misleading preferences can directly distort the optimization direction, making it a natural testbed for training-unit valuation. Vanilla DPO performs standard DPO updates without filtering. Random Pair Filtering provides a simple data-removal baseline by discarding preference pairs uniformly at random, while Reward-based Filtering prioritizes pairs according to the model’s DPO reward margin. DTV and DTV-Loo instead perform adaptive preference-pair valuation: each preference pair is treated as a training unit, and pairs with negative valuation scores are filtered following the zero-threshold rule.

Experiments are conducted on UltraFeedback [41] with Qwen2.5-1.5B [42, 43] using a two-stage pipeline consisting of supervised fine-tuning (SFT) followed by DPO. We consider Clean, Mismatch-20%, and Mismatch-40%, where the latter two corrupt 20% and 40% of the DPO training pairs through cross-response mismatching and chosen–rejected label flipping; all held-out evaluation and test data remain clean. For both Random and Reward-based Filtering, we use fixed filtering rates of 5% and 10%. In contrast, DTV and DTV-Loo use the zero valuation threshold defined in Section 4, allowing the effective filtering ratio to adapt dynamically during training without tuning a prescribed filtering ratio. We report the normalized area under the held-out preference accuracy curve over the training trajectory (AUC) and the preference accuracy at the final checkpoint (Acc.). Full hyperparameter and implementation details, together with additional downstream instruction-following results, are provided in Appendix D.

Main Results. Table 3 summarizes in-domain preference performance on the held-out test set, covering both trainingtime behavior and final-checkpoint accuracy. DTV-Loo achieves the best AUC across all three conditions. Compared with Reward 10%, its AUC margin widens from 0.5723 vs. 0.5652 (+0.0071) under Clean to 0.5406 vs. 0.5132 (+0.0274) under Mismatch-40%, indicating an increasingly pronounced training-time advantage as preference corruption intensifies. The advantage also extends to final-checkpoint performance, where DTV-Loo achieves the highest Acc. in all three settings, with the corresponding margin over Reward 10% increasing from 0.5680 vs. 0.5647 (+0.0033) to 0.5196 vs. 0.4857 (+0.0339). The stronger robustness under increasing corruption is consistent with the leaveone-out motivation in Section 4.3. With a larger fraction of misleading training units, cross-unit conflicts become more consequential, while positive self-terms can mask negative alignment and cause DTV to retain these units. By removing the self-term and relying solely on cross-unit alignment, DTV-Loo suppresses this self-protection and better preserves the conflict signal, which is reflected in its stronger performance under heavier corruption. This robustness is further supported by paired t-tests against Reward 10%, which show significant AUC improvements under all three conditions $( p = 0 . 0 0 0 6 , p < 0 . 0 0 0 1$ , and $p < 0 . 0 0 0 1 $ ), while the Acc. improvement is significant under Mismatch-40% $( p = 0 . 0 1 7 6 )$ Together, these results underscore the robustness of DTV-Loo to corrupted preference supervision. DTV-Loo consistently achieves the best performance across training, outperforming both Random and Reward Filtering even when their fixed filtering rates are increased to 20% and 40%, which is shown in Appendix D.

Figure 4 further characterizes these gains through test-accuracy dynamics and time-to-target efficiency. The top row shows the change in test preference accuracy relative to Vanilla DPO throughout training. Under Clean, DTV-Loo establishes a clear early-stage advantage, although the gap gradually narrows as training proceeds. Under the mismatch settings, the advantage becomes both larger and more persistent, with the strongest separation observed under Mismatch 40% and substantial gains maintained through the later stages of training. Reward 10% also improves over Vanilla DPO under mismatch, but its gains remain consistently smaller, while DTV and Random 10% exhibit weaker or less stable improvements.

Computational Efficiency. Here we further demonstrate the training efficiency of our proposed DTV and DTV-Loo. As demonstrated in the bottom row of Figure 4, by dynamically filtering out harmful or conflicting trajectories to prevent counterproductive parameter updates, DTV-Loo enhances data efficiency, achieving target performance with substantially fewer optimization steps. Consequently, under corrupted preference settings, DTV-Loo reaches the target accuracy $( T _ { 9 5 } )$ in approximately 30 minutes—drastically reducing total wall-clock training time compared to

Table 3: DPO preference-accuracy results on UltraFeedback under clean and mismatch settings. Results are reported as mean $\pm \ \mathrm { s t d } .$ over five seeds, with the best result in each column shown in bold. <sup>†</sup>The final row reports two-sided paired t-test p-values comparing DTV-Loo with Reward 10% across matched seeds; significant differences $( p < 0 . 0 5 )$ are shown in bold.
<table><tr><td></td><td colspan="2">Clean</td><td colspan="2">Mismatch-20%</td><td colspan="2">Mismatch-40%</td></tr><tr><td>Method</td><td>AUC↑</td><td> ${ \mathrm { A c c . ~ } } \uparrow$ </td><td>AUC↑</td><td>Acc. ↑</td><td>AUC↑</td><td>Acc. ↑</td></tr><tr><td>Vanilla DPO</td><td> $0 . 5 6 1 1 \pm 0 . 0 0 3 5$ </td><td> $0 . 5 3 8 8 \pm 0 . 0 0 7 3$ </td><td> $0 . 5 3 7 7 \pm 0 . 0 0 2 9$ </td><td> $0 . 5 0 1 4 \pm 0 . 0 0 4 4$ </td><td> $0 . 4 9 3 2 \pm 0 . 0 1 0 5$ </td><td> $0 . 4 3 0 1 \pm 0 . 0 1 6 0$ </td></tr><tr><td>Random 5%</td><td> $0 . 5 5 8 5 \pm 0 . 0 0 2 2$ </td><td> $0 . 5 4 4 9 \pm 0 . 0 0 2 3$ </td><td> $0 . 5 3 7 1 \pm 0 . 0 0 3 2$ </td><td> $0 . 5 0 1 3 \pm 0 . 0 1 5 2$ </td><td> $0 . 4 9 0 7 \pm 0 . 0 1 0 5$ </td><td> $0 . 4 2 3 4 \pm 0 . 0 1 4 3$ </td></tr><tr><td>Random 10%</td><td> $0 . 5 5 7 8 \pm 0 . 0 0 1 7$ </td><td> $0 . 5 4 2 1 \pm 0 . 0 0 6 3$ </td><td> $0 . 5 3 6 3 \pm 0 . 0 0 4 0$ </td><td> $0 . 5 0 1 5 \pm 0 . 0 1 0 0$ </td><td> $0 . 4 9 0 0 \pm 0 . 0 1 1 3$ </td><td> $0 . 4 3 3 2 \pm 0 . 0 1 2 2$ </td></tr><tr><td>Reward 5%</td><td> $0 . 5 6 2 0 \pm 0 . 0 0 1 6$ </td><td> $0 . 5 5 5 9 \pm 0 . 0 0 6 2$ </td><td> $0 . 5 4 6 5 \pm 0 . 0 0 2 8$ </td><td> $0 . 5 3 5 0 \pm 0 . 0 0 9 2$ </td><td> $0 . 5 0 8 3 \pm 0 . 0 0 8 9$ </td><td> $0 . 4 6 8 3 \pm 0 . 0 1 9 1$ </td></tr><tr><td>Reward 10%</td><td> $0 . 5 6 5 2 \pm 0 . 0 0 1 3$ </td><td> $0 . 5 6 4 7 \pm 0 . 0 0 6 3$ </td><td> $0 . 5 4 9 0 \pm 0 . 0 0 0 6$ </td><td> $0 . 5 4 4 3 \pm 0 . 0 0 4 8$ </td><td> $0 . 5 1 3 2 \pm 0 . 0 0 9 0$ </td><td> $0 . 4 8 5 7 \pm 0 . 0 1 4 7$ </td></tr><tr><td>DTV (Ours)</td><td> $0 . 5 6 5 3 \pm 0 . 0 0 2 1$ </td><td> $0 . 5 6 1 8 \pm 0 . 0 0 2 0$ </td><td> $0 . 5 4 4 5 \pm 0 . 0 0 4 1$ </td><td> $0 . 5 2 3 6 \pm 0 . 0 0 6 8$ </td><td> $0 . 4 9 8 5 \pm 0 . 0 1 1 0$ </td><td> $0 . 4 4 1 3 \pm 0 . 0 1 0 3$ </td></tr><tr><td>DTV-Loo (Ours)</td><td> $\mathbf { 0 . 5 7 2 3 \pm 0 . 0 0 0 4 }$ </td><td> $\mathbf { 0 . 5 6 8 0 \pm 0 . 0 0 4 0 }$ </td><td> $\mathbf { 0 . 5 6 6 9 \mathop { \pm } 0 . 0 0 0 9 }$ </td><td> $\mathbf { 0 . 5 5 3 7 \pm 0 . 0 0 6 8 }$ </td><td> $\mathbf { 0 . 5 4 0 6 \pm 0 . 0 0 9 3 }$ </td><td> $\mathbf { 0 . 5 1 9 6 \pm 0 . 0 2 0 8 }$ </td></tr></table>

![](images/cedfebc4b56482f1280eae0e40bac26f60a78437850bb64f216c811a52d8842b.jpg)  
Figure 4: DPO validation dynamics and training efficiency on UltraFeedback. The top row shows the change in validation accuracy relative to Vanilla DPO throughout training under Clean, Mismatch-20%, and Mismatch-40% conditions, averaged over five seeds; shaded regions indicate variability across seeds, and the horizontal dashed line denotes Vanilla DPO. The bottom row reports, for seed 0, the wall-clock time to reach the $T _ { 9 5 }$ target defined from the five-seed best validation performance; ×NR denotes that the target is not reached within the training budget. DTV-Loo exhibits increasingly large and more persistent improvements as the mismatch rate increases, while also reaching the convergence target substantially faster under the mismatch settings.

Vanilla DPO, while Random 10% fails to reach the target under Mismatch-40% within the training budget. Similar phenomena are also observed in the PPO experiments shown in Figure 2. In short, our method achieves a double win by simultaneously boosting performance and accelerating overall training.

When to Use DTV vs. DTV-Loo. The core distinction between DTV and DTV-Loo stems from the cross-term effect. While the previous subsection illustrated a regime where a minimal cross-term effect results in marginal performance differences between DTV and DTV-Loo (e.g., on GSM8K), the cross-term plays a far more decisive role under other training conditions. Below, we examine scenarios where cross-term interactions substantially impact optimization (see the difference under the noisy Mismatch-20% setting in Table 3) and provide practical guidance on selecting between the two variants.

From a macro perspective, the self-term acts as a regularizing buffer that governs the strictness of trajectory filtering: retaining it favors conservatism, whereas removing it via DTV-Loo enforces strict cross-unit consistency. Figure 5 empirically substantiates this trade-off under Mismatch-20%. Specifically, Figure 5(a) demonstrates that the DTV score is predominantly driven by the self-term throughout training. This dominance becomes critical when a training unit is negatively aligned with the batch: a strong positive self-term can easily override a negative cross-term, thereby causing DTV to retain units that DTV-Loo would otherwise filter. In contrast to the decreasing cross-term contribution observed on GSM8K, Figure 5(b) shows that the cross-term contribution remains substantial and generally increases during DPO training, making cross-unit alignment more consequential in this setting. Figure 5(c) provides a representative case within a single mini-batch where a single sample exhibits an exceptionally large self-term, whereas the cross-terms across all samples remain negligible in magnitude. Consequently, this single outlier sample dominates the aggregate DTV valuation of the entire batch, artificially driving the score positive and masking critical cross-unit conflicts that DTV-Loo successfully identifies and removes. Moreover, the self-term violates the majority gradient assumption and disproportionately affects DTV’s valuation despite its negative cross-term. By removing this unit’s self-term, DTV-Loo is less susceptible to such a dominant unit and better preserves the cross-unit conflict signal.

![](images/5802424b88f3ff8d7fbeb0faf4c751b2affed09d687bdc76642f1e28318dc010.jpg)  
(a) Score decomposition

![](images/c026c545014a885f3faa7f62396b01df01512aaed7115b6f3167fae4c8ccd94b.jpg)  
(b) Cross-term contribution

![](images/b8787cbd4fe3234d89721cd59c3f0ffc5dc680c929321b04a885255a086fe378.jpg)  
(c) Sample rank  
Figure 5: Empirical analysis of the self-protection effect on UltraFeedback under Mismatch-20%. (a) Decomposition of the DTV Score into the self- and cross-terms. (b) Cross-term contribution $| c _ { j } | / ( s _ { j } + | c _ { j } | )$ for all and conflicted units. (c) Visualization of self- and cross-term scores across samples in a representative batch (ranked by self-term). A single dominant sample with an extreme self-term score overwhelms the cross-term interactions across the entire batch, illustrating the self-protection mechanism in standard DTV.

These insights provide clear practical guidance on selecting between the two variants, supporting DTV-Loo as the default choice in practice. By eliminating self-protection, DTV-Loo relies strictly on cross-unit alignment, offering superior robustness, faster convergence, and significant performance gains in complex or high-noise regimes such as DPO on UltraFeedback and GRPO on AIME. Meanwhile, in low-conflict scenarios like GRPO on GSM8K—where disagreements stem primarily from weak negative cross-terms—DTV-Loo performs comparably to standard DTV, demonstrating that strict filtering incurs minimal risk of discarding useful signals. Retaining the self-term in DTV is thus primarily beneficial when conservative filtering is specifically needed to safeguard weakly conflicting samples; otherwise, DTV-Loo serves as the more robust and effective default strategy.

## 6 Conclusion

In this paper, we studied trajectory valuation in reinforcement learning and proposed Dynamic Trajectory Valuation (DTV), a simple and general framework that dynamically identifies detrimental training units through mini-batch gradient alignment without requiring a predefined validation set or influence-function approximation. Building on the decomposition of DTV, we further introduced DTV-Loo to remove the self-term and analyzed the resulting selfprotection effect, revealing that the utility of the self-term can depend on the underlying optimization setting. We evaluated DTV and DTV-Loo across diverse reinforcement learning and LLM post-training paradigms, including PPO, GRPO, and DPO, spanning trajectory segments, completion trajectories, and preference pairs. Extensive experiments demonstrate that dynamic trajectory valuation can improve optimization efficiency, robustness, and final performance across these settings, highlighting the practical value of training-unit-level valuation for reinforcement learning.

## Acknowledgment

We gratefully acknowledge the support of the Google TPU Builders Program and Google Tunix library, which provided support and access to the computational framework.

## References

[1] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2018.

[2] Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.

[3] Dong Shu, Denghui Zhang, and Jessica Hullman. Learning from the right rollouts: Data attribution for ppo-based llm post-training. arXiv preprint arXiv:2604.01597, 2026.

[4] Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. In International Conference on Machine Learning, 2017.

[5] Yongchan Kwon, Eric Wu, Kevin Wu, and James Zou. Datainf: Efficiently estimating data influence in lora-tuned llms and diffusion models. arXiv preprint arXiv:2310.00902, 2023.

[6] Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. Trak: Attributing model behavior at scale. arXiv preprint arXiv:2303.14186, 2023.

[7] Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, et al. Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296, 2023.

[8] Ziao Yang, Han Yue, Jian Chen, and Hongfu Liu. Revisit, extend, and enhance hessian-free influence functions. arXiv preprint arXiv:2405.17490, 2024.

[9] Yuzheng Hu, Fan Wu, Haotian Ye, David Forsyth, James Zou, Nan Jiang, Jiaqi W Ma, and Han Zhao. A snapshot of influence: A local data attribution framework for online reinforcement learning. arXiv preprint arXiv:2505.19281, 2025.

[10] Shipeng Li, Zhiqin Yang, Shikun Li, Xiaobo Xia, Hengyu Liu, Xinghua Zhang, Gaode Chen, Dong Fang, Ying Tai, and Zhe Peng. Learnalign: Data selection for llm reinforcement learning with improved gradient alignment. arXiv preprint arXiv:2506.11480, 2026.

[11] Ningyuan Yang, Weihua Du, Weiwei Sun, Sean Welleck, and Yiming Yang. Gradalign: Gradient-aligned data selection for llm reinforcement learning. arXiv preprint arXiv:2602.21492, 2026.

[12] Ang A Li, Zongqing Lu, and Chenglin Miao. Revisiting prioritized experience replay: A value perspective. arXiv preprint arXiv:2102.03261, 2021.

[13] Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel. Prioritized level replay. In International Conference on Machine Learning. PMLR, 2021.

[14] Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In International Conference on Machine Learning, 2009.

[15] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in Neural Information Processing Systems, 2017.

[16] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 2022.

[17] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[18] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[19] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 2023

[20] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

[21] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

[22] Frank R Hampel. The influence curve and its role in robust estimation. Journal of the American Statistical Association, 1974.

[23] R Dennis Cook and Sanford Weisberg. Residuals and influence in regression. New York: Chapman and Hall, 1982.

[24] R Douglas Martin and Victor J Yohai. Influence functionals for time series. The Annals ofStatistics, 1986.

[25] Guillaume Charpiat, Nicolas Girard, Loris Felardos, and Yuliya Tarabalka. Input similarity from the neural network perspective. Advances in Neural Information Processing Systems, 2019.

[26] Garima Pruthi, Frederick Liu, Satyen Kale, and Mukund Sundararajan. Estimating training data influence by tracing gradient descent. Advances in Neural Information Processing Systems, 2020.

[27] Krishnateja Killamsetty, Sivasubramanian Durga, Ganesh Ramakrishnan, Abir De, and Rishabh Iyer. Grad-match: Gradient matching based data subset selection for efficient deep model training. In International Conference on Machine Learning, 2021.

[28] Satoshi Hara, Atsushi Nitanda, and Takanori Maehara. Data cleansing for models trained with sgd. Advances in Neural Information Processing Systems, 2019.

[29] Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, and Xiaojuan Qi. Data pruning via moving-one-sample-out. Advances in Neural Information Processing Systems, 2023.

[30] Narine Kokhlikyan, Kamalika Chaudhuri, and Saeed Mahloujifar. Z0-inf: Zeroth order approximation for data influence. arXiv preprint arXiv:2510.11832, 2025.

[31] Jiachen T Wang, Prateek Mittal, Dawn Song, and Ruoxi Jia. Data shapley in one training run. arXiv preprint arXiv:2406.11011, 2024.

[32] Ziao Yang, Longbo Huang, and Hongfu Liu. Layer-aware influence for online data valuation estimation. arXiv preprint arXiv:2510.16007, 2025.

[33] Anshuman Chhabra, Bo Li, Jian Chen, Prasant Mohapatra, and Hongfu Liu. Outlier gradient analysis: Efficiently identifying detrimental training samples for deep learning models. arXiv preprint arXiv:2405.03869, 2024.

[34] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Yash Katariya, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: composable transformations of Python+NumPy programs, 2018.

[35] Tianshu Bao, Jeff Carpenter, Lin Chai, Haoyu Gao, Yangmu Jiang, Shadi Noghabi, Abheesht Sharma, Sizhi Tan, Lance Wang, Ann Yan, Weiren Yu, et al. Tunix (tune-in-jax), 2025.

[36] Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo Perez-Vicente, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, and Jordan Terry. Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks. 2023.

[37] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[38] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, et al. Gemma 3 technical report, 2025.

[39] Hugging Face H4. AIME 2024.

[40] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025.

[41] Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Wei Zhu, Yuan Ni, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Ultrafeedback: Boosting language models with high-quality feedback, 2023.

[42] Qwen Team. Qwen2.5: A party of foundation models, 2024.

[43] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

[44] Gemma Team. Gemma 3. 2025.

[45] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Benjamin Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, et al. Livebench: A challenging, contamination-limited llm benchmark. In International Conference on Learning Representations, 2025.

[46] Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah Smith, Hanna Hajishirzi, and Nathan Lambert. Rewardbench 2: Advancing reward model evaluation. In International Conference on Learning Representations, 2026.

[47] Valentina Pyatkin, Saumya Malik, Victoria Graf, Hamish Ivison, Shengyi Huang, Pradeep Dasigi, Nathan Lambert, and Hanna Hajishirzi. Generalizing verifiable instruction following. Advances in Neural Information Processing Systems, 2025.

## Appendix

This appendix provides additional related work, implementation details for PPO, GRPO, and DPO, and extended DPO experimental results to complement the main manuscript.

## A Additional Related Work on on Data-Centric RL

Table 4: Comparison of additional requirements and applicability of gradient-based data valuation methods in reinforcement learning. The binary columns indicate whether each method uses an auxiliary valuation signal, assumes a fixed dataset, requires a validation set, uses an explicit selection budget, requires extra rollouts for valuation, or performs valuation online. The final column summarizes each method’s generalizability across reinforcement learning settings.
<table><tr><td>Method</td><td>Auxiliary signal</td><td>Fixed dataset</td><td>Validation set</td><td>Selection budget</td><td>Extra rollouts</td><td>Online valuation</td><td>Applied scenario</td></tr><tr><td>IIF (2025)</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td><td>No</td><td>Yes</td><td>PPO</td></tr><tr><td>LearnAlign (2026)</td><td>Yes</td><td>Yes</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td><td>GRPO / RLVR</td></tr><tr><td>GradAlign (2026)</td><td>Yes</td><td>No</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>GRPO</td></tr><tr><td>DTV/DTV-Loo (Ours)</td><td>No</td><td>No</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>General</td></tr></table>

Table 4 compares DTV with three representative gradient-based approaches and summarizes their requirements in reinforcement learning. IIF [9] requires an algorithm-specific surrogate target function for valuation and a filtering parameter p that determines how many negatively influential samples are removed. Its target function must also be adapted to the underlying reinforcement learning setting. IIF performs valuation online by reusing the current rollout buffer without additional rollouts. LearnAlign [10] relies on verifiable ground-truth feedback to estimate learnability and introduces N to determine the selected subset. Its formulation assumes a fixed problem dataset with ground-truth answers, as required by the Reinforcement Learning with Verifiable Rewards (RLVR) setting, rather than experience collected purely through online environment interaction. It also requires additional rollouts to estimate the learnability. GradAlign [11] requires a trusted validation set to construct the reference gradient and uses q to specify the fraction of candidate problems retained. Its formulation is developed in the GRPO setting and performs valuation online by recomputing gradients under the evolving policy. GradAlign requires additional rollouts to estimate both validation and candidate gradients for selection. In contrast, our DTV and DTV-Loo perform online valuation directly from current batch gradients, without relying on a fixed dataset, auxiliary signals, or additional rollouts. Their core filtering rule applies a fixed zero threshold without an explicit filtering-rate or selection-budget hyperparameter, and applies consistently across GRPO, DPO, and PPO.

## B Additional DTV Experimental Details for PPO

This section provides additional details for the PPO experiments, including the environments, compared methods, model and optimization configurations, and evaluation protocol.

## B.1 Environments

We evaluate PPO on MiniGrid-Empty-8x8-v0 and MiniGrid-DoorKey-8x8-v0 [36], as illustrated in Figure 1. Both are partially observable 8 × 8 sparse-reward grid worlds. Empty requires direct navigation to the goal; DoorKey requires collecting a key, unlocking a door, and then reaching the goal. MiniGrid uses its standard shaped success reward, which decreases linearly with the fraction of the episode horizon consumed; failure receives zero. The native observation is a dictionary containing an image, direction, and mission. We apply ImgObsWrapper, so the policy receives only a channel-last $7 \times 7 \times 3$ symbolic image encoding object type, color, and state. The action space contains the seven standard MiniGrid actions.

Compared Methods. We compare DTV and DTV-Loo with Vanilla PPO and Iterative Influence-Based Filtering (IIF) [9] in the PPO setting. Vanilla PPO performs standard PPO updates without filtering. Iterative Influence-Based Filtering (IIF) [9] operates at the transition level. It reuses the current on-policy rollout buffer to construct a return-based impact target, computes policy-gradient similarity scores, and removes the bottom 12.5% of records among those with negative influence. The rollout buffer is used for both PPO optimization and this local influence calculation; no separate validation rollout is sampled.

DTV and DTV-Loo instead operate on trajectory units. Each environment stream is partitioned independently at episode termination and at the rollout boundary; an unfinished episode therefore contributes one observed trajectory fragment.

Table 5: PPO training and evaluation configuration. The environment-step budget is the configured stopping target; collection ends after completing the final rollout round.
<table><tr><td>Hyperparameter</td><td>Empty-8x8</td><td>DoorKey-8x8</td></tr><tr><td>Parallel environments</td><td>16</td><td>16</td></tr><tr><td>Environment-step budget</td><td>160,000</td><td>1,000,000</td></tr><tr><td>Rollout steps per environment</td><td>128</td><td>640</td></tr><tr><td>Transitions per rollout round</td><td>2,048</td><td>10,240</td></tr><tr><td>Optimizer mini-batch size</td><td>64</td><td>256</td></tr><tr><td>PPO epochs per round</td><td>10</td><td>4</td></tr><tr><td>Optimizer</td><td>SGD</td><td>Adam</td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Discount factor γ</td><td>0.99</td><td>0.99</td></tr><tr><td>GAE trace parameter</td><td>0.95</td><td>0.95</td></tr><tr><td>PPO clipping coefficient</td><td>0.2</td><td>0.2</td></tr><tr><td>Entropy coefficient</td><td>0</td><td>0.01</td></tr><tr><td>Value-loss coefficient</td><td>0.5</td><td></td></tr><tr><td>Maximum gradient norm</td><td>0.5</td><td>0.5</td></tr><tr><td>Optimizer epsilon</td><td></td><td>0.5</td></tr><tr><td></td><td></td><td> $1 0 ^ { - 8 }$ </td></tr><tr><td>Advantage normalization</td><td>Yes</td><td>Yes</td></tr><tr><td>Evaluation episodes per checkpoint</td><td>1,000</td><td>1,000</td></tr></table>

For each unit $z _ { j } ,$ , the valuation gradient is computed from the mean PPO policy loss over its transitions. Using the gradient notation of Section 4, DTV compares $g _ { j }$ with the mean gradient of all trajectory units, whereas DTV-Loo excludes $g _ { j }$ from its reference mean. Units with negative scores are filtered. Thus, PPO uses a policy-loss score at trajectory level. The subsequent optimizer step uses a total-loss mask: all transitions in a filtered unit are removed from the complete PPO objective, including policy, value, and regularization terms. Scores are computed once per rollout round and the same mask is reused across all PPO epochs and mini-batches in that round. Filtering starts immediately on Empty-8x8 and after five unfiltered warmup rounds on DoorKey-8x8; IIF uses the same environment-specific warmup schedule.

Model and Optimization. All methods use a convolutional actor–critic with two $3 \times 3$ convolutional layers of 16 and 32 channels, ReLU activations, and a linear projection to a shared 64-dimensional representation. Separate linear heads parameterize the categorical policy and scalar value estimate. No recurrent state is used. Within an environment, all methods share the architecture, interaction budget, and base PPO hyperparameters listed in Table 5. The implementation uses JAX on TPU v5p-8 hardware.

Evaluation. At each reported checkpoint, the policy is evaluated for 1,000 episodes without parameter updates. Learning curves show the mean return across five seeds and ±1 standard error of the mean. For the Worst/Best comparison in Figure 2, the 1,000 final-checkpoint returns are sorted separately for each seed. The lowest and highest 20 returns are pooled across seeds, giving up to 100 observations in each environment–method subset. In the split violin plots, thin lines show the selected range, thick lines the interquartile range, horizontal marks the median, and white diamonds the mean.

## C Additional DTV Experimental Details for GRPO

This section provides additional details for the GRPO experiments on GSM8K and AIME, including the datasets and corruption protocol, compared methods, model and optimization configurations, and evaluation settings.

## C.1 Datasets and Benchmarks

GSM8K. We use the official GSM8K splits [37]. The data are independently shuffled for each run; the first 3,072 shuffled examples form 768 batches of four prompts. The first 691 batches (2,764 prompt occurrences) are used for training and the remaining 77 batches (308 prompts) for periodic clean evaluation. Thus, one run performs 691 updates over this selected partition. The official 1,319-example test split is used only for final evaluation. Prompts request reasoning inside <reasoning> tags and one numerical answer inside <answer> tags; targets are parsed from the substring following the GSM8K #### delimiter.

In Mismatch-20%, a stable SHA256 hash of the prompt, corruption schema, and experiment seed independently selects groups at an approximately 20% rate. The four observed rewards in each selected group are reassigned in reverse rank order. This preserves the reward multiset, mean, and standard deviation while corrupting the completion–reward association used to compute advantages. Tied rewards can make a selected group only partially changed, or unchanged when all four rewards are equal. Periodic and final evaluation data remain clean. We restrict the corruption study to Mismatch-20%, since extending the same rank-reversal perturbation to 40% of prompt groups would substantially distort the within-group relative signal that directly drives the GRPO update.

Table 6: GRPO training and evaluation configurations on GSM8K and AIME 2024
<table><tr><td>Hyperparameter</td><td>GSM8K</td><td>AIME</td></tr><tr><td>Prompts per update</td><td>4</td><td>128</td></tr><tr><td>Completions per prompt</td><td>4</td><td>8</td></tr><tr><td>Train micro-batch size</td><td></td><td>2</td></tr><tr><td>Maximum prompt / response length</td><td>256 /768</td><td>1,024 / 8,192</td></tr><tr><td>Optimizer / peak learning rate</td><td>AdamW / 1 × 10−⁶</td><td>AdamW / 1 × 10−⁶</td></tr><tr><td>Learning-rate schedule</td><td>69-step warmup, cosine decay</td><td>cosine decay, no warmup</td></tr><tr><td>Optimizer updates</td><td>691</td><td>314</td></tr><tr><td>KL coefficient</td><td>0.08</td><td>0.001</td></tr><tr><td>Clipping width</td><td>0.20 / 0.20</td><td>0.20 / 0.28</td></tr><tr><td>Adam coefficients / weight decay</td><td>(0.9, 0.99) / 0.1</td><td>(0.9, 0.99) / 0.1</td></tr><tr><td>Loss aggregation</td><td>sequence-mean-token-mean</td><td>sequence-mean-token-mean</td></tr><tr><td>Final evaluation</td><td>1,319 problems, greedy decoding</td><td>30 problems, 16 responses each</td></tr><tr><td>Evaluation temperature / top-p Maximum evaluation generation</td><td>greedy (top-k = 1) 768</td><td>0.6 / 0.95 32,768</td></tr></table>

AIME. We train on the DeepScaleR-Preview-Dataset and evaluate on all 30 AIME 2024 problems [39, 40]. After preprocessing, 40,300 training prompts are valid and 40,192 are selected after a fixed shuffle. With 128 prompts per update, this partition yields exactly 314 optimizer steps. AIME uses binary mathematical correctness as the group reward and one training seed because of the cost of 8K-token training rollouts and 32K-token evaluation generations.

## C.2 Experimental Setup

Compared Methods. We compare Vanilla GRPO, Random Filtering, Reward-based Filtering, LearnAlign [10], GradAlign [11], DTV, and DTV-Loo. Random Filtering, Reward-based Filtering, DTV, and DTV-Loo make completionlevel filtering decisions within each prompt group, whereas LearnAlign and GradAlign perform selection at the prompt level. Vanilla GRPO retains the complete group. Random and Reward-based Filtering use prescribed expected removal rates: Random selects completions independently of reward, whereas Reward-based Filtering removes completions with the lowest observed group-relative advantages. LearnAlign performs a prompt-level selection stage before the main RLVR training, after warm-up, ranking prompts using learnability-weighted gradient alignment computed from eight rollouts per prompt and retaining a predefined top-N subset; in our experiments, N is set to retain 50% of the prompts. GradAlign instead performs online prompt-level selection by ranking prompts according to their alignment with a validation gradient estimated from a small validation set using eight rollouts per validation prompt and retaining a fixed fraction (25% in this setting). We carefully reimplement both methods in JAX for TPU execution, following their original formulations as closely as possible and choosing the selection ratios with reference to the configurations used in the original studies. DTV and DTV-Loo instead perform adaptive completion-level filtering using the current mini-batch policy gradients as an endogenous reference and the zero valuation threshold, without prescribing an overall filtering rate. On GSM8K, we evaluate all methods, with Random and Reward-based Filtering using expected removal rates of 5% and 10%; on the substantially more expensive AIME setting, we compare the three core methods, Vanilla GRPO, DTV, and DTV-Loo.

Models and Optimization. GSM8K. The actor and frozen reference policy are initialized from Gemma-3-1B-IT [44]. The actor uses LoRA with rank 64 and scaling 64 on q\_einsum, kv\_einsum, attn\_vec\_einsum, gate\_proj, down\_proj, and up\_proj; base parameters remain frozen. DTV inner products therefore cover all trainable LoRA tensors.

Each update samples four completions for each of four prompts, with one generated completion trajectory treated as one valuation unit. Each completion is scored using the gradient of its policy-only GRPO loss, excluding KL. DTV uses all four completion gradients to construct the group reference, whereas DTV-Loo uses the other three gradients for the eval uated completion. Advantages are computed before filtering and are not recomputed after selection. After thresholding at zero, the selected completion mask is applied as a total-loss mask to the full policy-plus-KL GRPO update.

The shaped training reward combines: (i) a format score of +1 for the requested reasoning/answer structure and −1 otherwise; (ii) a numerical score of +4 for exact correctness and +3, +2, +1, or +0.25 for relative errors within 1%, 5%, 10%, or 25%, with −1 for larger or unparseable errors; and (iii) an additional +1.5 for an exactly correct number after the <answer> tag. Using the notation in the main-text GRPO formulation, advantages are computed from the complete group as the reward minus the group mean, divided by the group sample standard deviation plus 10<sup>−4</sup>. Filtering does not recompute these statistics. Because 5% and 10% are not integral for groups of four, Random and Reward-based Filtering remove one completion with probability 0.2 and 0.4, respectively. For GSM8K DTV-Loo only, a 25% per-group minimum retains at least one finite-score completion; an all-negative group restores its highest-scoring completion. DTV has no such minimum.

Table 7: SFT and DPO training configurations
<table><tr><td>Hyperparameter</td><td>SFT</td><td>DPO</td></tr><tr><td>Fine-tuning mode</td><td>Full parameter</td><td>LoRA (r = 64, scaling = 64)</td></tr><tr><td>Train / evaluation batch size</td><td>2/2</td><td>8/8</td></tr><tr><td>Gradient accumulation steps</td><td>4</td><td>32</td></tr><tr><td>Maximum length</td><td>Target: 768</td><td>Prompt / response: 512 / 512</td></tr><tr><td>Optimizer / peak learning rate</td><td>AdamW / 1 × 10−5</td><td>AdamW / 1 × 10−⁶</td></tr><tr><td>Warmup / decay steps</td><td>100 / 1,500</td><td>10 / 115</td></tr><tr><td>Weight decay</td><td>0.05</td><td>0.1</td></tr><tr><td>Maximum gradient norm</td><td>1.0</td><td></td></tr><tr><td>DPO coefficient β / label smoothing</td><td></td><td>0.01 / 0</td></tr><tr><td>Evaluation / save interval</td><td>100 / 100 steps</td><td>10 / 3 steps</td></tr><tr><td>Optimizer updates</td><td>1,493</td><td>115</td></tr></table>

AIME 2024. We full-parameter fine-tune DeepSeek-R1-Distill-Qwen-1.5B on the selected DeepScaleR prompts. Each update contains 128 prompts and eight completions per prompt, with one completion trajectory treated as one valuation unit. Each completion is scored from its policy-only loss gradient within the eight-completion prompt group: DTV uses all eight gradients in the reference, while DTV-Loo uses the other seven. After zero-threshold selection, the completion mask is applied as a total-loss mask to the full policy-plus-KL GRPO update. A shared mask removes groups whose eight advantages are all zero. AIME DTV-Loo has no minimum-retention rule, so an all-negative finite group may retain no completion.

Evaluations. GSM8K. We use five training seeds. Methods matched by seed share data order, rollout randomness, mismatch selection, and stochastic filtering randomness. Periodic clean evaluation every 32 steps is not used for early stopping; final metrics use step 691 and the official 1,319-example test split. Exact accuracy requires a correct parsed number, while partial accuracy accepts a prediction-to-target ratio in [0.9, 1.1]. Means and sample standard deviations are reported over independently trained seeds. Training uses JAX on one TPU v5p-8 worker with a (4, 1) mesh over fully sharded data parallel (FSDP) and tensor parallel (TP) axes.

AIME. All final checkpoints use one fixed engine-level evaluation seed and matched prompts, tokenizer, and decoding parameters. For each of the 30 problems, we sample 16 responses; these are consecutive samples from the seeded generation stream rather than independent training seeds. If a response contains </think>, only the subsequent answer portion is used before boxed-answer extraction. Grading applies the MathD, SymPy, and special-handling checks in tunix.utils.math\_eval\_metrics; a valid AIME answer must normalize to an integer in [0, 999]. We report Pass@1, Pass@16, Maj@16, extractability, and accuracy conditioned on extraction.

Training uses JAX on a dual-worker TPU v5p-16 setup. Vanilla GRPO requires approximately 120–144 hours, while DTV and DTV-Loo require approximately 168–192 hours because of per-completion valuation. One 32K checkpoint evaluation takes approximately eight generation-hours; all four checkpoints require 32.1 hours sequentially, excluding initialization and restoration. This substantial cost motivates the single-training-seed protocol.

## D Additional DTV Experimental Details and Results for DPO

This section provides additional details for the DPO experiments, including the dataset and mismatch construction, compared methods, model and optimization configurations, and evaluation protocol. We further report downstream instruction-following results and additional analyses of training dynamics and filtering rates.

## D.1 Dataset and Mismatch Construction

We use HuggingFaceH4/ultrafeedback\_binarized, derived from UltraFeedback [41]. Prompt-level partitions are constructed with a fixed seed. From train\_prefs, 25% of prompts are assigned to SFT and 75% to DPO; 10% of each stage-specific partition is held out for clean periodic evaluation. SFT prompts therefore do not reappear in DPO training, and held-out prompts never contribute to training, valuation, or optimization. test\_prefs is used only for final-checkpoint evaluation.

Table 8: Clean downstream instruction-following performance of DPO methods. Results are mean ± sample standard deviation over five training seeds. These metrics are not used for training, filtering, or checkpoint selection; the best result in each condition and column is shown in bold.
<table><tr><td>Condition</td><td>Method</td><td>LiveBench-IF ↑</td><td>RB2 Precise IF ↑</td><td>IFBench P-Strict ↑</td></tr><tr><td rowspan="7">Clean</td><td>Vanilla DPO</td><td>0.2418±0.0035</td><td>0.4938±0.0285</td><td>0.1280±0.0045</td></tr><tr><td>Random 5%</td><td>0.2413±0.0046</td><td>0.4763±0.0294</td><td>0.1227±0.0090</td></tr><tr><td>Random 10%</td><td>0.2436±0.0049</td><td>0.4813±0.0442</td><td>0.1307±0.0068</td></tr><tr><td>Reward 5%</td><td>0.2388±0.0064</td><td>0.4800±0.0203</td><td>0.1253±0.0016</td></tr><tr><td>Reward 10%</td><td>0.2370±0.0055</td><td>0.4613±0.0191</td><td>0.1247±0.0050</td></tr><tr><td>DTV (Ours)</td><td>0.2430±0.0034</td><td>0.4938±0.0198</td><td>0.1247±0.0045</td></tr><tr><td>DTV-Loo (Ours)</td><td>0.2415±0.0081</td><td>0.4238±0.0195</td><td>0.1293±0.0065</td></tr><tr><td rowspan="7">Mismatch-20%</td><td>Vanilla DPO</td><td>0.2454±0.0042</td><td>0.4950±0.0278</td><td>0.1233±0.0047</td></tr><tr><td>Random 5%</td><td>0.2404±0.0052</td><td>0.4875±0.0088</td><td>0.1273±0.0061</td></tr><tr><td>Random 10%</td><td>0.2434±0.0041</td><td>0.5025±0.0350</td><td>0.1267±0.0060</td></tr><tr><td>Reward 5%</td><td>0.2421±0.0025</td><td>0.4775±0.0196</td><td>0.1253±0.0081</td></tr><tr><td>Reward 10%</td><td>0.2429±0.0067</td><td>0.4450±0.0254</td><td>0.1233±0.0073</td></tr><tr><td>DTV (Ours)</td><td>0.2400±0.0051</td><td>0.4863±0.0327</td><td>0.1287±0.0058</td></tr><tr><td>DTV-Loo (Ours)</td><td>0.2413±0.0084</td><td>0.4613±0.0320</td><td>0.1293±0.0039</td></tr><tr><td rowspan="7">Mismatch-40%</td><td>Vanilla DPO</td><td>0.2434±0.0056</td><td>0.5638±0.0228</td><td>0.1247±0.0034</td></tr><tr><td>Random 5%</td><td>0.2492±0.0054</td><td>0.5300±0.0174</td><td>0.1260±0.0049</td></tr><tr><td>Random 10%</td><td>0.2497±0.0057</td><td>0.5600±0.0370</td><td>0.1233±0.0084</td></tr><tr><td>Reward 5%</td><td>0.2425±0.0053</td><td>0.5213±0.0122</td><td>0.1193±0.0080</td></tr><tr><td>Reward 10%</td><td>0.2448±0.0055</td><td>0.5050±0.0327</td><td>0.1193±0.0049</td></tr><tr><td>DTV (Ours)</td><td>0.2445±0.0084</td><td>0.5588±0.0196</td><td>0.1267±0.0037</td></tr><tr><td>DTV-Loo (Ours)</td><td>0.2357±0.0063</td><td>0.4563±0.0285</td><td>0.1233±0.0056</td></tr></table>

Mismatch modifies only the DPO training partition. For the Mismatch-20% and Mismatch-40% conditions, we sample without replacement exactly the floor of the requested fraction of training pairs using the experiment seed. Donors are restricted to this selected subset, and a shuffled cyclic map prevents self-mapping. The cross\_response\_flip operation keeps the target prompt, replaces its preferred response with a dispreferred response from a selected non-self donor, and replaces its dispreferred response with a preferred response from another selected non-self donor. When at least three pairs are selected, the two response fields use opposite cyclic shifts; with two pairs, they necessarily share the only nonself donor. This single operation jointly introduces cross-response mismatch and preference-label inversion. Methods matched by seed use identical corrupted pairs; different seeds corrupt independently, and all evaluation data remain clean.

## D.2 Experimental Setup

Compared Methods. We compare Vanilla DPO, Random Pair Filtering, Reward-based Filtering, DTV, and DTV-Loo. Vanilla DPO uses every preference pair. Random and Reward-based Filtering remove 5% or 10% of the pairs in a gradient-accumulation window; Random samples uniformly, whereas Reward-based Filtering removes pairs with the lowest DPO implicit reward margins. These baselines therefore require a prescribed filtering rate.

DTV and DTV-Loo instead treat each prompt–response preference pair as one training unit. Unlike PPO and GRPO, their score gradient is computed from the complete per-pair DPO loss rather than a separate policy-only surrogate. Gradients are taken with respect to every trainable LoRA tensor across the complete accumulation window. DTV compares each pair’s gradient with the average gradient over the complete window, whereas DTV-Loo forms the reference average from the remaining pairs. Non-negative finite scores are retained. The optimizer then applies a total-loss mask to the same complete DPO objective and normalizes the final loss and gradient by the retained-pair count. There is no fixed minimum retained fraction; if an entire window would otherwise be empty, the implementation restores its finite pairs to keep the update well defined.

Model and Optimization. We start from Qwen2.5-1.5B [42, 43]. The base model is first full-parameter fine-tuned on the SFT partition and exported. The iterator ends after 1,493 updates, slightly before the configured 1,500-step maximum; this is dataset exhaustion rather than early stopping. Every DPO actor and frozen reference model is initialized from this same SFT export.

![](images/a191b3885c1fa627b2ceaaae640413908928ee6b9fee673cd6dd8682feb6feb0.jpg)  
Figure 6: Absolute preference-accuracy trajectories on UltraFeedback under Clean, Mismatch-20%, and Mismatch-40% training conditions. Curves show the five-seed mean and shaded regions show ±1 sample standard deviation.

The DPO actor uses LoRA with rank 64 and scaling 64 on q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, and down\_proj. Actor and reference models use a (2, 2) mesh over fully sharded data parallel (FSDP) and tensor parallel (TP) axes. Tables 7 gives the complete optimization settings.

Evaluation. All methods use the final step-115 checkpoint; periodic evaluation is not used for checkpoint selection. We train five matched seeds, with run, data-shuffle, and curation seeds matched across methods. AUC is the normalized area under the clean held-out preference-accuracy curve over training, and Acc. is the final-checkpoint preference accuracy. Held-out pairs never enter DTV’s reference gradient. Means and sample standard deviations are reported over training seeds, and the paired two-sided tests in Table 3 compare DTV-Loo with Reward 10% on matched seeds.

## D.3 Additional Experimental Results

## D.3.1 Downstream Instruction-Following Results

We evaluate final checkpoints on three clean external instruction-following benchmarks. LiveBench-IF [45] evaluates on the benchmark’s test split and deterministic generation with a fixed seed, top-k 50, top-p 0.95, maximum prompt length 4,096, maximum response length 1,024, and batch size eight. RewardBench 2 [46] evaluates on its official test split; candidates are scored by the actor’s DPO implicit reward relative to the frozen SFT reference, and we report the Precise IF subset. IFBench [47] uses the official prepared assets and reports prompt-level strict accuracy (P-Strict), with a fixed seed, top-k 50, top-p 0.95, maximum prompt length 4,096, maximum response length 1,024, and batch size eight. Table 8 reports mean and sample standard deviation over five independently trained checkpoints. Downstream instruction-following results are mixed: DTV remains broadly competitive, while the stronger in-domain gains of DTV-Loo do not uniformly transfer across all external metrics.

## D.3.2 Training Curves and Filtering-Rate Ablation

Figure 6 complements the delta-to-Vanilla view in the main text by showing the absolute preference-accuracy trajectories. Each curve is averaged over five training seeds, and shading denotes one sample standard deviation. Periodic evaluation is always clean, including for models trained under mismatch. Under Clean, most methods converge to similar final accuracy, whereas the performance differences become increasingly pronounced as the mismatch level increases. In particular, DTV-Loo maintains faster and stronger improvement under both mismatch settings, with the largest separation observed under Mismatch-40%.

The main comparison uses fixed filtering rates of 5% and 10% for Random and Reward-based Filtering. To test whether DTV-Loo’s advantage under severe mismatch is explained only by removing more pairs, we additionally train Random and Reward variants at 20% and 40% under Mismatch-40%.

Figure 7 shows that increasing a prescribed filtering budget does not reproduce the training trajectory of DTV-Loo. Although more aggressive Reward-based Filtering improves over its lower-rate variants, DTV-Loo still exhibits faster optimization and achieves stronger overall performance, while increasing the Random Filtering rate provides no comparable benefit. These results indicate that the advantage of DTV-Loo cannot be attributed solely to a larger effective filtering ratio; rather, which preference pairs are removed is critical. More broadly, the sensitivity of Random and Reward-based Filtering to the prescribed removal rate highlights a limitation of fixed-rate filtering, as a single predefined ratio may not remain appropriate throughout training. In contrast, DTV and DTV-Loo use a fixed zero valuation threshold, allowing the effective filtering ratio to adapt dynamically to the evolving optimization state without tuning an explicit removal rate. This ablation tests sensitivity to the fixed filtering rate and it is not an evaluation of the theoretical DTV-λ family.

![](images/ab3e3d2df0b6ef2a255b05644b60c1b650fff6433727be0dbd1ed18e7219a01c.jpg)  
Figure 7: Fixed-filtering-rate ablation under Mismatch-40%. DTV and DTV-Loo are compared with Random and Reward-based Filtering at 5%, 10%, 20%, and 40%. Curves show the five-seed mean, with shading denoting ±1 sample standard deviation.