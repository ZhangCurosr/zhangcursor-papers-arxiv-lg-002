# UCB-min: Combining Bandits Under Variance Heterogeneity (Extended Version)

Lilia Izri   
LISN, CNRS   
Universite Paris-Saclay´   
Orsay, France   
izri@lisn.fr   
Benoˆıt Groz   
LISN, CNRS   
Universite Paris-Saclay´   
Orsay, France   
groz@lisn.fr LIG, CNRS   
Univ. Grenoble-Alpes, Grenoble INP Grenoble, France   
silviu.maniu@univ-grenoble-alpes.fr

## Abstract

Multi-armed bandits are a natural framework for many real-world sequential decision problems, from recommender systems and clinical trials to educational test recommendation and online advertising. The performance of these algorithms depends on how they explore different arms, and exploration strategies range from variance-independent (e.g., UCB1) to variance-aware (e.g., UCB-V). In practice the reward variance of each arm is rarely known and varies widely across arms, and the reward support is often underspecified. In these cases no single rule is uniformly best: variance-independent methods are robust but loose when variances are small, while variance-aware methods are tight only when their variance estimates and the assumed support bound are reliable. Such heterogeneous, unknown variances arise in many settings, for instance when multiple objectives are scalarized into a single reward. We propose UCBmin, a per-arm adaptive meta-algorithm that selects, for each arm, the tighter of a variance-independent and a variance-aware exploration bonus, without prior knowledge of the variance regime and without reward normalization. We prove that, with UCB-V as variance-aware base, its regret is bounded by the sum of the two base bounds at the standard exploration constant, and by their per-arm minimum, up to a constant, for a larger one. We validate it on synthetic data, an educational tutoring task, and two group-recommendation benchmarks (MovieLens, Steam), showing that UCB-min consistently tracks the best base algorithm across diverse variance regimes and datasets.

## Index Terms

Multi-armed bandits, Upper confidence bounds, Variance-aware exploration, Multi-objective bandits, Adaptive algorithms

## I. INTRODUCTION

Multi-armed bandit (MAB) algorithms are fundamental to sequential decision-making under uncertainty, with applications in clinical trials [1], [2], recommender systems [3], [4], resource allocation [5], and online advertising [6], [7]. UCB methods assign each arm an optimistic index combining its empirical mean with an exploration bonus and pull the arm with the highest index; these bonuses range from variance-independent (UCB1 [8]) to variance-aware (UCB-V [9], [10]) design. The formal definitions and regret comparisons are given in Section II.

In practice the reward variance of each arm is rarely known and varies widely across arms, so neither family is uniformly best: variance-independent methods are robust but loose when variances are small, while variance-aware methods are tight only when their variance estimates and the assumed support bound are reliable. Normalization could remove this scale dependence but is often impractical online (see Section IV-A).

This is the extended version of the short paper of the same title accepted at IEEE BigData 2026.

This occurs in many real-world applications, in particular inherently multi-objective ones: recommender systems trade off accuracy with diversity, novelty, and fairness [11], [12], clinical trials optimize efficacy and safety [2], and advertising platforms balance user satisfaction and diversity [13]. In the recommendation settings we consider, each arm is an item (e.g., a news article or product), pulling it recommends that item, and the reward vector’s dimensions capture properties such as relevance, clicks, ratings, or engagement. A common way to handle several objectives is scalarization: combining the reward dimensions into a single scalar via application-specific weights $w \in \mathbb { R } ^ { D }$ , after which standard single-objective bandits such as UCB apply. This reduction is itself a source of the variance heterogeneity above, since the scalarized variance depends on the weights and the cross-objective covariance, quantities often unknown in advance, which makes scalarized multi-objective bandits a natural motivation for our work.

To address this, we propose UCB-min, a per-arm adaptive meta-algorithm for scalarized multi-armed bandits that selects, for each arm, the tighter of a variance-independent and a variance-aware exploration bonus. This choice requires no prior knowledge of the variance regime and no reward normalization. The approach is validated on synthetic datasets and real-world multi-objective applications: an education dataset [14], where multi-objective formulations and algorithm choice matter depending on the variance regime, and group recommendation datasets [15], where per-arm adaptation provides robustness across arms with different variance profiles.

Contributions: Our main contributions are the following:

1) We analyze the effect of variance scales on regret in multi-objective bandits and empirically show that the relative performance of variance-independent (UCB1) and variance-aware (UCB-V) algorithms is data dependent.

2) We propose per-arm adaptive meta-algorithms that dynamically select between variance-independent and variance-aware exploration bonuses on a per-arm basis, without prior knowledge of the variance regime and without normalization.

3) We provide a theoretical guarantee for the per-arm minimum rule with UCB-V (UCB-minV1): the cumulative expected regret has the same asymptotic bound as the original algorithms and, for a larger exploration constant, tracks the regret of the best algorithm (Theorem 1); the variant with UCB-V2 (UCB-minV2) is validated empirically.

4) We validate our approach on synthetic, educational, and recommendation datasets, showing robust empirical improvements across diverse variance regimes.

## II. MULTI-OBJECTIVE MULTI-ARMED BANDITS

This section introduces the (multi-objective) bandit framework and the UCB algorithms used throughout the paper. We first define the general stochastic bandit setting with multiple objectives, then present linear scalarization as a means of reducing it to a single-objective problem, and discuss the challenges this reduction creates for algorithm selection.

## A. Multi-Objective Bandit Framework

We consider a stochastic multi-armed bandit with K arms $\mathcal { A } = \{ 1 , \ldots , K \}$ and D objectives. At each round $t = 1 , \dots , T$ , the bandit algorithm selects an arm $a _ { t } \in \mathcal A$ and observes a reward vector $\mathbf { r } _ { t } \in \mathbb { R } ^ { D }$ . For each arm a, the rewards are i.i.d. with unknown mean vector $\pmb { \mu } _ { a } \in \mathbb { R } ^ { D }$ and covariance

matrix $\pmb { \Sigma } _ { a } \in \mathbb { R } ^ { D \times D }$ . Let $\begin{array} { r } { N _ { a } ( t ) = \sum _ { \tau = 1 } ^ { t } \mathbf { 1 } \{ a _ { \tau } = a \} } \end{array}$ be the number of times arm a has been pulled up to time t. The empirical estimators are, with the variance normalized by $N _ { a } ( t )$ as in [16, Theorem 1]:

$$
\begin{array} { r l } & { { \hat { \pmb \mu } } _ { a } ( t ) = \frac { 1 } { N _ { a } ( t ) } \displaystyle \sum _ { \tau = 1 , a _ { \tau } = a } ^ { t } { \bf r } _ { \tau } , } \\ & { { \hat { \bf \Sigma } } _ { a } ( t ) = \frac { 1 } { N _ { a } ( t ) } \displaystyle \sum _ { \tau = 1 , a _ { \tau } = a } ^ { t } \left( { \bf r } _ { \tau } - { \hat { \pmb \mu } } _ { a } ( t ) \right) \left( { \bf r } _ { \tau } - { \hat { \pmb \mu } } _ { a } ( t ) \right) ^ { \top } . } \end{array}
$$

Pareto vs. scalarized: A natural performance criterion in multi-objective bandits is Pareto regret, which measures suboptimality with respect to the Pareto front, which is the set of arms not simultaneously dominated by any other in all objectives [17], [18]. However, identifying and tracking the Pareto front can be computationally expensive. We therefore focus on scalarized regret: given application-specific weights, the objectives are combined into a single scalar, reducing the problem to a standard singleobjective bandit that is directly compatible with existing UCB-style algorithms and well-suited when the desired trade-off is known in advance.

Linear scalarization reduces the multi-objective problem to a standard single-objective stochastic bandit, enabling the direct use of well-studied UCB algorithms and their theoretical guarantees. Given a fixed weight vector $\mathbf { w } \in \mathbb { R } ^ { D }$ with ${ \bf w } _ { d } \geq 0$ and $\textstyle \sum _ { d = 1 } ^ { D } \mathbf { w } _ { d } = 1$ , the scalarized reward observed at round t is $X _ { t } = \mathbf { w } ^ { \top } \mathbf { r } _ { t }$ . For each arm a, the scalarized mean and variance are:

$$
\boldsymbol { \mu } _ { a } ^ { ( s ) } = \mathbf { w } ^ { \top } \boldsymbol { \mu } _ { a } , \qquad \boldsymbol { \sigma } _ { a } ^ { 2 , ( s ) } = \mathbf { w } ^ { \top } \boldsymbol { \Sigma } _ { a } \mathbf { w } ,\tag{1}
$$

where the superscript (s) marks scalarized quantities.

After t rounds, the empirical estimators are:

$$
\begin{array} { r } { \hat { \mu } _ { a } ^ { ( s ) } ( t ) = \mathbf { w } ^ { \top } \hat { \pmb { \mu } } _ { a } ( t ) , \qquad \hat { \sigma } _ { a } ^ { 2 , ( s ) } ( t ) = \mathbf { w } ^ { \top } \hat { \Sigma } _ { a } ( t ) \mathbf { w } . } \end{array}\tag{2}
$$

## B. UCB Algorithms and Regret

The goal of bandit algorithms is to minimize the cumulative regret. Let $a ^ { \star } = \arg \operatorname* { m a x } _ { a } \mu _ { a } ^ { ( s ) } , \mu _ { \star } ^ { ( s ) } = \mu _ { a ^ { \star } } ^ { ( s ) }$ and $\begin{array} { r } { \dot { \Delta _ { a } } = \mu _ { \star } ^ { ( s ) } - \mu _ { a } ^ { ( s ) } } \end{array}$ . The expected cumulative regret is:

$$
R ( T ) = \mathbb { E } \Big [ \sum _ { { t = 1 } } ^ { T } \bigl ( \mu _ { \star } ^ { ( s ) } - X _ { t } \bigr ) \Big ] = \sum _ { a \in \mathcal { A } } \Delta _ { a } \mathbb { E } [ N _ { a } ( T ) ] .\tag{3}
$$

To minimize this quantity, UCB-type algorithms assign each arm an index combining the empirical scalarized mean with an exploration bonus and select the arm with the highest index. $\beta _ { a } ^ { \top } ( t )$ denotes a pure exploration bonus and $B _ { a } ^ { ( \cdot ) } ( t ) = \hat { \mu } _ { a } ^ { ( s ) } ( t ) + \beta _ { a } ^ { ( \cdot ) } ( t )$ the corresponding full index; the algorithm pulls $\arg \operatorname* { m a x } _ { a } B _ { a } ^ { ( \cdot ) } ( t )$ . This work builds upon well-known algorithms (presented below) which we apply to the scalarized multi-objective setting for rewards having support [0, b].

UCB1 [8]: UCB1 is variance-independent; its bonus and index are

$$
\beta _ { a } ^ { ( 1 ) } ( t ) = b \sqrt { \frac { 2 \log t } { N _ { a } ( t ) } } , \quad B _ { a } ^ { ( 1 ) } ( t ) = \hat { \mu } _ { a } ^ { ( s ) } ( t ) + \beta _ { a } ^ { ( 1 ) } ( t ) ,\tag{4}
$$

achieving regret

$$
\sum _ { a : \Delta _ { a } > 0 } \frac { 8 b ^ { 2 } \log T } { \Delta _ { a } } + \Bigl ( 1 + \frac { \pi ^ { 2 } } { 3 } \Bigr ) \sum _ { a } \Delta _ { a } .
$$

When $b = 1$ this reduces to the standard formulation. Two refinements from the same work are UCB1- Tuned and UCB1-Normal [8]. UCB1-Tuned replaces the exploration bonus by

$$
\sqrt { \frac { \log t } { N _ { a } ( t ) } \operatorname* { m i n } \left\{ \frac { 1 } { 4 } , \hat { \sigma } _ { a } ^ { 2 , ( s ) } ( t ) + \sqrt { \frac { 2 \log t } { N _ { a } ( t ) } } \right\} } ,
$$

using an empirical variance estimate to tighten the bonus; it often improves empirical performance but lacks formal regret guarantees. UCB1-Normal [8] targets Gaussian rewards; in our case, this assumption may not hold after scalarization of objectives.

UCB-V [9]: UCB-V explicitly uses variance in the exploration bonus:

$$
\beta _ { a } ^ { ( V ) } ( t ) = \sqrt { \frac { 2 \hat { \sigma } _ { a } ^ { 2 , ( s ) } ( t ) c \log t } { N _ { a } ( t ) } } + \frac { 3 c b \log t } { N _ { a } ( t ) } ,\tag{5}
$$

with $B _ { a } ^ { ( V ) } ( t ) = \hat { \mu } _ { a } ^ { ( s ) } ( t ) + \beta _ { a } ^ { ( V ) } ( t )$ and $c = 1 . 2 .$ Its regret bound [16, Theorem 4] is

$$
1 0 \sum _ { a : \Delta _ { a } > 0 } \left( \frac { \sigma _ { a } ^ { 2 , ( s ) } } { \Delta _ { a } } + 2 b \right) \log T ,
$$

which improves over UCB1 when per-arm variances are small relative to $b .$

$U C B \ – V 2 \ I I O J .$ : UCB-V2, our name for the version of UCB-V analyzed in [10], refines UCB-V with the bonus

$$
\beta _ { a } ^ { ( V 2 ) } ( t ) = \operatorname* { m a x } \left( \frac { \hat { \sigma } _ { a } ^ { ( s ) } ( t ) } { \sqrt { \rho \log T } } , \frac { b } { \sqrt { N _ { a } ( t ) } } \right) \cdot \frac { \rho \log T } { \sqrt { N _ { a } ( t ) } } ,\tag{6}
$$

and $B _ { a } ^ { ( V 2 ) } ( t ) = \hat { \mu } _ { a } ^ { ( s ) } ( t ) + \beta _ { a } ^ { ( V 2 ) } ( t )$ , where $\rho$ is a tuning parameter $( \rho = 2$ in our experiments). When $b = 1 , T \geq 3$ , and $\rho \geq C _ { 0 }$ for a universal constant $C _ { 0 }$ , with probability at least $1 - 3 K / T ^ { 2 }$ , its regret scales as

$$
1 6 \operatorname* { m i n } \left\{ \sqrt { \sum _ { a = 2 } ^ { K } \sigma _ { a } ^ { 2 , ( s ) } } , \sum _ { a = 2 } ^ { K } \frac { 1 6 \sigma _ { a } ^ { 2 , ( s ) } } { \sigma _ { 1 } ^ { ( s ) } } \right\} \sqrt { \rho T \log T } + ( K - 1 ) \sqrt { \rho \log T } ,
$$

where the arms are indexed so that the optimal arm $a ^ { \star } = 1$

## III. OTHER RELATED WORK

We review related work on MAB alternatives and on meta-combination strategies for bandit algorithms beyond the UCB-based methods already introduced in Section II-B.

MAB alternatives: Thompson Sampling (TS) [19], [20] and KL-UCB [21] often perform well but rely on distributional assumptions: TS needs a prior and likelihood model, and KL-UCB derives bounds from KL-divergences for specific families (e.g., Bernoulli); EUCBV [22] combines variance estimates with arm elimination. After scalarization, which mixes heterogeneous objectives (e.g., binary and continuous), the reward distribution is unknown in advance, so we focus on distribution-free UCB methods and do not compare against TS or KL-UCB, but against UCB1, UCB1-Tuned, UCB-V, UCB-V2, and EUCBV (Section V).

Multi-objective problems: Multi-objective sequential decision problems are addressed either through Pareto-based methods, which maintain the set of non-dominated solutions, or through scalarization-based methods, which reduce the objectives to a single scalar [17], [23]. Pareto-based approaches require repeated dominance checks, and the Pareto front can be prohibitively large [23, Sec. 4.2.2], which has motivated cost-reducing heuristics such as annealing [24]. Scalarization (linear, Chebyshev, randomized, etc.) is a lightweight alternative, directly compatible with UCB/Thompson-style bandit and RL algo rithms. Studies include a MOMAB framework with empirical comparisons [17], contextual Pareto regret bounds [18], the Generalized Gini Index [25], and, in multi-objective RL, a survey of MOMDP/MORL methods with conditions under which scalarization preserves optimality [23] and the learning of policies that treat several users equitably [26]. As discussed in Section I, scalarization induces heterogeneous per-arm variance, which motivates the per-arm rule validated in Section V.

Meta algorithms: Several works consider meta-combination of bandit strategies. Meta-UCB [27] treats each base learner as a single arm in a higher-level bandit, and its regret depends only on that of the best base learner. Corralling [28] adaptively selects which base algorithm to run each round. [29] proposes MIN-UCB, a per-arm min-of-indices approach combining online samples with offline data via a bias-correction term, providing instance-dependent and near-minimax guarantees but requiring offline data and a known bias bound. None of these works studies a fully online, distribution-free per-arm rule selecting between variance-independent and variance-aware indices without offline data or known bias bound, a gap this work addresses. These are therefore not directly comparable baselines: Meta-UCB and Corralling select among entire algorithms rather than per arm, MIN-UCB additionally requires offline data and a known bias bound, and adversarial selectors such as Exp3 ignore variance entirely. A related idea appears in the stopping-rule literature, where [30] combines a variance-independent and a variance-aware stopping rule by stopping at the earlier of the two; that work targets (ϵ, δ) mean estimation rather than per-arm bandit regret minimization. Another family of methods scales bandits to large recommendation settings by sharing statistics across users: CLUB [31] and its data-dependent refinement [32] cluster users online, DCCB [33] distributes the clustering over a peer-to-peer network, and DistCLUB [34] constructs clusters lazily to reduce communication. These are contextual linear bandits requiring item features and the identity of the user at each round, hence not comparable baselines in the featureless scalarized setting we consider: they exploit similarity across users, whereas UCB-min adapts to per-arm variance.

## IV. UCB-MIN: PER-ARM ADAPTIVE META-ALGORITHM

## A. Variance Changes in Multi-Objective Scalarization

Eq. (1) shows that the scalarized variance depends on the weights and on the covariance structure of the objectives, and correlation effects can reduce it substantially. Scalarized variances may therefore range from very small to comparable to the largest single-objective variance, creating a dilemma between variance-aware methods (effective when variance is small) and variance-independent ones (more robust otherwise).

Two practical difficulties make this dilemma worse. First, normalization is often impractical: reward scales are unknown online, non-stationary settings require continuous re-estimation, and for physical objectives (energy, time, cost) no predetermined upper bound exists. It does not reliably reduce regret either (see Figs. 7a-7b). Second, the algorithms above require the support bound b, which is generally unknown without normalization: Gaussian rewards are unbounded and, for real-world data, the effective range is hard to determine in advance. Using a large b inflates all exploration bonuses and leads to near-linear regret (see Fig. 8b).

## B. The UCB-min Meta-Algorithm

To control these issues, we propose UCB-MIN, an adaptive per-arm meta-algorithm that selects, for each arm, the tighter of a variance-independent and a variance-aware exploration bonus. It is analyzed

in Theorem 1, validated in Section V, and, although both bonuses still, in theory, depend on the support bound b, is empirically robust to an underestimated b, allowing the default $b = 1$ across variance regimes. Throughout, “meta” refers to combining existing base algorithms in this sense, and not to meta-learning.

Algorithm 1 UCB-min   
Require: Arms A, horizon T, weight vector w   
1: Initialize: ∀a: $N _ { a } \gets 0 , \hat { \pmb { \mu } } _ { a } \gets \mathbf { 0 } _ { D } , \hat { \pmb { \Sigma } } _ { a } \gets \mathbf { 0 } _ { D \times D }$   
2: Pull each arm a ∈ A once.   
3: for $t = 1$ to $T$ do   
4: for each $a \in { \mathcal { A } }$ do   
5: Compute $\hat { \mu } _ { a } ^ { ( s ) } ( t )$ and $\hat { \sigma } _ { a . . . } ^ { 2 , ( s ) } ( t ) \ ( \mathrm { E q . } \ ( 2 ) )$   
6: Compute the bonuses $\beta _ { a } ^ { ( 1 ) } ( t )$ (Eq. (4)) and $\beta _ { a } ^ { \mathrm { ( v a r ) } } ( t )$ (Eq. (5) or (6))   
7: $B _ { a } ( t ) \gets \hat { \mu } _ { a } ^ { ( s ) } ( t ) + \operatorname* { m i n } \{ \beta _ { a } ^ { ( 1 ) } ( t ) , \beta _ { a } ^ { ( \mathrm { v a r } ) } ( t ) \}$   
8: end for   
9: Pull $a _ { t } \gets \arg \operatorname* { m a x } _ { a } B _ { a } ( t )$ , observe $\mathbf { r } _ { t } ,$ update $N _ { a _ { t } } , \hat { \mu } _ { a _ { t } } , \hat { \Sigma } _ { a _ { t } }$   
10: end for

For each arm a, we compute both the variance-independent bonus $\beta _ { a } ^ { ( 1 ) } ( t ) \left( \mathrm { E q . } \left( 4 \right) \right)$ ) and a variance-aware bonus $\beta _ { a } ^ { \mathrm { ( v a r ) } } ( t ) ~ ( \mathrm { E q . ~ } ( 5 )$ or (6)), keep the tighter one, and add the empirical mean once:

$$
B _ { a } ( t ) = \hat { \mu } _ { a } ^ { ( s ) } ( t ) + \operatorname* { m i n } \left\{ \beta _ { a } ^ { ( 1 ) } ( t ) , \beta _ { a } ^ { ( \mathrm { v a r } ) } ( t ) \right\} = \operatorname* { m i n } \left\{ B _ { a } ^ { ( 1 ) } ( t ) , B _ { a } ^ { ( \mathrm { v a r } ) } ( t ) \right\} .\tag{7}
$$

Both forms coincide since $\hat { \mu } _ { a } ^ { ( s ) } ( t )$ is shared, and Appendix A uses the second one, which reasons directly on indices. Each arm thus benefits from variance-aware exploration when its empirical variance is low, and reverts to UCB1 when the variance-aware bonus would be too large. Both bonuses are already valid upper confidence bounds, so their minimum is valid as well, and being the tighter of the two it avoids the exploration that a larger combination (maximum, mean) would incur for no gain in guarantee. This is especially useful when b is misspecified: for a small b (e.g., the default $b = 1 ) , \beta _ { a } ^ { ( 1 ) } ( t )$ remains tight while the variance-aware bonus may still be inflated by large empirical variances. The selection also depends on the pull count $N _ { a } ( t )$ : the variance-aware bonus carries an extra $O ( 1 / N _ { a } ( t ) )$ term that dominates in early iterations, so the minimum then falls back on $\beta _ { a } ^ { ( 1 ) } ( t )$ , and the variance plays its role mainly once $N _ { a } ( t )$ is large enough.

We consider two variants depending on the variance-aware base:

• UCB-minV1: $\beta _ { a } ^ { \mathrm { ( v a r ) } } = \beta _ { a } ^ { ( V ) } ,$ , Eq. (5),

• UCB-minV2: $\beta _ { a } ^ { \mathrm { ( v a r ) } } = \beta _ { a } ^ { ( V 2 ) }$ , Eq. (6).

Both share the same structure, presented in Algorithm 1. At each round, we compute scalarized statistics $\hat { \mu } _ { a } ^ { ( s ) } ( t ) = \mathbf { w } ^ { \top } \hat { \mu } _ { a } ( t )$ and $\hat { \sigma } _ { a } ^ { 2 , ( s ) } \hat { ( } t ) = \mathbf { w } ^ { \top } \hat { \Sigma } _ { a } ( t ) \mathbf { \bar { w } }$ , evaluate both bonuses, and pull arg $\operatorname* { m a x } _ { a } B _ { a } ( t )$ with $B _ { a } ( t )$ as in Eq. (7).

Computational cost: UCB-min evaluates two bonuses per arm at each round, so its cost per round is of the same order as that of UCB1 and UCB-V, namely $O ( K )$ given the scalarized statistics. With a fixed weight vector w, these statistics are the empirical mean and variance of the scalarized rewards observed on each arm, since

$$
\mathbf { w } ^ { \top } \hat { \mathbf { \Sigma } } _ { a } ( t ) \mathbf { w } = \frac { 1 } { N _ { a } ( t ) } \sum _ { \tau \leq t , a _ { \tau } = a } \big ( X _ { \tau } - \hat { \mu } _ { a } ^ { ( s ) } ( t ) \big ) ^ { 2 } ,
$$

and can be updated online in $O ( 1 )$ time and memory per arm; maintaining the full covariance $\hat { \Sigma } _ { a } ( t )$ , at a cost of $O ( D ^ { 2 } )$ per arm, is only needed to evaluate several weight vectors on the same observations, where diagonal or low-rank approximations would reduce this cost for many objectives.

Regret upper bound: We next show that the asymptotic regret bound of UCB-minV1 is the same as the regret bound of the original algorithms:

Theorem 1. Fix per-arm $\Delta _ { a } , \ \sigma _ { a } ^ { 2 } ,$ and support bound b. The statement holds for any bounded rewards, so we omit the <sup>(s)</sup> superscript here and in the proof.

(i) For $c \geq 4$ in the UCB-V index, UCB-minV1 attains for each arm the minimum of the UCB1 and UCB-V regret bounds, up to a constant:

$$
O \left( \left( \sum _ { a } \operatorname* { m i n } \left( \frac { \sigma _ { a } ^ { 2 } } { \Delta _ { a } } + b , b ^ { 2 } / \Delta _ { a } \right) \right) \log T + \sum _ { a } \Delta _ { a } \right) ,
$$

i.e. it is never worse than the better of the two on any arm.

(ii) At the standard $c = 1 . 2 \ [ 9 ]$ , UCB-minV1 still enjoys the weaker guarantee of being bounded by the sum of the two base bounds:

$$
O \left( \left( \sum _ { a } \frac { \sigma _ { a } ^ { 2 } + b ^ { 2 } } { \Delta _ { a } } + b \right) \log T + \sum _ { a } \Delta _ { a } \right) .
$$

We use $c = 1 . 2$ in our experiments, the value fixed by [9], at which only part (ii) is proved; a larger c would keep the guarantee valid but enlarges the bonus, hence adds exploration, which is not always desirable (e.g. Fig. 8b). While our theoretical guarantees are limited to UCB-minV1 and the stronger guarantee of part (i) also requires a larger c, our experiments in Section V provide empirical evidence for similar behavior beyond the scope of the theorem, with both UCB-minV1 and UCB-minV2 appearing to track the best of their base algorithms.

Note: An alternative is global selection: at each round, estimate the regret upper bounds of both UCB1 and the variance-aware algorithm, then apply the one with the smaller bound to all arms (Meta-1 with UCB-V, Meta-2 with UCB-V2). However, it ignores heterogeneous per-arm variance, accumulates early-round estimation errors, and relies on bounds that may greatly overestimate actual regret; our experiments confirm it is unreliable, so we focus on the per-arm rule.

## V. EXPERIMENTS

We present the experimental results, with two objectives: (1) show how UCB-MIN adapts to different variance regimes, and (2) show that real-world multi-objective applications benefit from this adaptation.

## A. Datasets

We evaluate on three types of data: synthetic instances, an educational tutoring simulator, and two group recommendation benchmarks (MovieLens, Steam).

Synthetic Data. We generate instances with $K = 2 0$ arms and $D = 2$ independent Gaussian objectives, with means in [0, 1] and standard deviations in $[ \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ]$ . Two dimensions suffice, as the goal is to validate behavior under controlled variance regimes.

Educational $D a t a . ^ { 1 }$ The dataset [35] contains 4,918 students, 948 questions, and 1.38M responses. Each question is assigned a difficulty level $l \in \{ 1 , \ldots , 5 \}$ based on global success rate; each student’s proficiency $l _ { u }$ is the highest level with $\geq 5 0 \%$ correct answers. Arms correspond to the $K = 5$ difficulty levels. Following [14], pulling arm a (level $l _ { a } )$ for a student at level $l _ { u }$ yields the following reward components that all lie in [0, 1]:

• Aptitude $( \mathbf { A } ) \colon r ^ { ( A ) }$ a binary reward indicating correct/incorrect answer.

• Competence (C): $r ^ { ( C ) } = 1 - | l _ { u } - l _ { a } | / K$ (difficulty matching).

• Performance ${ \bf ( P ) } \colon r ^ { ( P ) } = \operatorname* { m a x } \{ 0 , ( l _ { a } - l _ { u } ) / K \}$ (potential progression).

• Gap (G): $r ^ { ( G ) } = 1 - | l _ { \mathrm { f a i l } } - l _ { a } | / K$ (recovery from failure with $l _ { \mathrm { f a i l } }$ the user’s failing level).

Steam<sup>2</sup> and MovieLens $2 5 M ^ { 3 } .$ : Group Recommendations. MovieLens 25M [36] contains 25M ratings (scale 0.5-5) from 162K users on 62K movies across 20 genres. Steam contains 41M recommendations from 14M users on 51K games with 337 tag categories; we use an extract of 4M user profiles. MovieLens groups of 20 users are formed via a variance-constrained process, whereas Steam groups are real community groups extracted via the Steam API. The relevance of a category for a group is the average rating over group members and items in that category. Objectives follow [15]; arms are categories and pulling category c for group G yields:

• Relevance: $r _ { c , t } ^ { ( R ) } = \mathrm { r e l } ( u , c )$ for randomly sampled user $u \in G$

• Pairwise Disagreement: $r _ { c , t } ^ { ( D ) } = | \mathrm { r e l } ( u , c ) - \mathrm { r e l } ( v , c ) |$ for sampled pair $u , v .$

Disagreement is inverted to $r _ { \operatorname* { m a x } } - r ^ { ( D ) }$ for maximization, where $r _ { \mathrm { m a x } }$ is the maximum rating of the dataset. Rewards lie in [0, 5] for MovieLens (not normalized) and [0, 1] for Steam.

## B. Evaluation Metrics

The algorithms are evaluated along three categories of metrics, formally defined in Appendix B.   
Regret: cumulative regret, as defined in Eq. (3).

Ranking quality: Hit Rate@k, assessing whether one of the last k recommended arms is among the 5 arms with the highest true scalarized mean.

Domain-specific: Click-Through Rate (CTR) for MovieLens and Steam; skill progression (NCC) for the educational task.

## C. Implementation and Experiments Details

Machine specifications: All experiments ran on a Linux server (8-core Intel Xeon E5-2609v2 2.5GHz, 64GB RAM).

Experiments: Each algorithm is run 50 times and averaged, under identical seeds, horizon and value of $b ;$ all objectives get equal weights. We aggregate over 200 users (diverse skill levels) for Education and 50 groups for Steam and MovieLens, the latter formed to ensure variance heterogeneity. We report the algorithms of Sections II-IV; on the real-world datasets, EUCBV and Meta-1 are omitted for clarity as they never reach the lowest regret on synthetic data, and UCB-minV1 as the experiments focus on UCB-minV2; their full curves are available online.<sup>4</sup> Error bars report one standard deviation over the runs; they are omitted in some figures for readability. The open-source Python implementation, datasets descriptions and supplementary materials are available online.<sup>4</sup> The raw experimental results, additional results and datasets are also available online.<sup>5</sup>

Support bound: We set b = 1 everywhere. This matches the true support for the educational and Steam data (rewards in [0, 1]); for MovieLens and high-variance synthetic data it underestimates the support, matching our setting where the reward range is unknown in advance. The theoretical guarantees of all compared methods no longer apply there; this is exactly the purpose of this evaluation, which assesses their empirical robustness to an underestimated b. Fig. 8b reports a well-specified bound for MovieLens.

## D. Results

Variance regimes - Synthetic data: Figures 1a-1b show cumulative regret under two variance regimes with the same scalarization weights. In the low-variance regime (Fig. 1a), UCB-V2 achieves roughly 10× lower final regret than UCB1, confirming the theoretical advantage of variance-aware bounds when $\sigma _ { a } ^ { 2 , ( s ) }$ is small. In the high-variance regime (Fig. 1b), this relationship reverses: UCB-V2 shows nearlinear regret growth, as large empirical variances make its bonus too large and cause over-exploration. Since Gaussian rewards exceed [0, 1] in this regime, b = 1 underestimates the true support; nonetheless, UCB1’s bonus is smaller than UCB-V2’s variance-dependent bonus, making UCB1 more robust in this regime. UCB-minV1 adapts automatically, tracking UCB-V in Fig. 1a and UCB1 in Fig. 1b without prior knowledge; UCB-minV2 behaves similarly but with lower regret thanks to the tighter UCB-V2 bound, so the remaining experiments focus on it, although Theorem 1 does not cover it. Figure 1c further motivates per-arm adaptation using a synthetic instance where half the arms have low variance $( \sigma \in [ 0 . 2 , 0 . 4 ] )$ and the other half have high variance $( \sigma \in [ 2 . 0 , 4 . 0 ] )$ ). In this heterogeneous setting, neither UCB1 nor UCB-V2 dominates: UCB1 is penalized on low-variance arms, while UCB-V2 over-explores high-variance arms. UCB-minV2 achieves the lowest regret by adapting its strategy per arm, outperforming all other tested algorithms by the end of all iterations. Figure 2 shows the resulting selection: the variance-aware bonus is chosen on 80-100% of the pulls of the low-variance arms, against 12-23% on the high-variance ones, letting UCB-minV2 reach the lowest final regret by following the variance profile of each arm.

![](images/949366e7e84f69f24734e927e376e5a141cc3edd376c2f658782a02a4ed1a15a.jpg)  
Fig. 1: Cumulative regret across variance regimes: (a) $\sigma \in [ 0 . 1 , 0 . 3 ] .$ , (b) $\sigma \in [ 2 , 4 ] .$ , (c) heterogeneous arms. (a) and (b) share the same scalarization weights, whereas (c) illustrates the impact of heterogeneous arm variances (high variance for half of the arms, low variance for the other half); its y-axis is cut at 2.2k (UCB-V and UCB1-Normal end at 3.7k and 6.5k).

![](images/aa10a64f8a0953fec25f0adeb8c6bd19428261aa6918d67b835e9a18e712d4e4.jpg)  
Fig. 2: Per-arm selection on the heterogeneous instance of Fig. 1c. Top: true scalarized variance per arm. Bottom: how the pulls of each arm split between the two bonuses.

Figure 3 shows how the choice of scalarization weights affects the variance regime and thus the relative performance of the algorithms. Here, one objective dimension has low variance $( \sigma \in [ 0 . 1 , 0 . 5 ] )$ and the other has high variance $( \sigma \in [ 2 . 0 , 4 . 0 ] )$ . With $w = ( 0 . 2 , 0 . 8 )$ (Fig. 3a), the weight vector emphasizes the high-variance dimension, resulting in large scalarized variances. In this regime, UCB1 performs better while UCB-V2 over-explores due to large variance-dependent bonuses. Conversely, with $w = ( 0 . 8 , 0 . 2 )$ (Fig. 3b), the weight vector emphasizes the low-variance dimension, reducing the scalarized variance and shifting the regime toward one where variance-aware methods perform better. In both cases, UCBminV2 tracks the better-performing base algorithm, adapting to the effective variance regime induced by the weights.

![](images/4f3946dd562d75592c1f8a780bfa47b00976723f7e9bb556476ff338aba6602f.jpg)  
(a) w = (0.2, 0.8).  
(b) w = (0.8, 0.2).  
Fig. 3: Impact of scalarization weights on cumulative regret. Dim 1 has $\sigma \in [ 0 . 1 , 0 . 5 ]$ (low variance), Dim 2 has $\sigma \in [ 2 . 0 , 4 . 0 ]$ (high variance).

Educational dataset (short horizon): For this dataset, b = 1 matches the true support since all rewards lie in [0, 1]. Figure 4 reports both cumulative regret and the progression skill for $T = 3 0 0$ (short horizon, realistic for exercise recommendation). Comparing columns (from left to right), progression skill increases, confirming the benefit of multi-objective formulations for this task. In terms of regret, UCB1 tends to perform better in the very early stages, while the variance-aware UCB-V2 becomes more effective later as more observations are collected and variance estimates stabilize. UCB-minV2 shows a similar transition: it behaves like UCB1 in the initial iterations (e.g. the first 50 in Fig. 4c) before shifting toward UCB-V2.

![](images/f9cd022831a902aca5d62c4dfdd9c87d16ef9f720582ed22f07af5f235ae4890.jpg)  
Fig. 4: Educational data: regret (top row) and progression skill (bottom row) for $T = 3 0 0 .$ . Columns correspond to different objective sets (A, A+P, A+C+G).

Figure 5 shows the reward distributions of one user across all arms in the $\mathbf { A { + } C { + } G }$ setting with equal weights, one row per objective dimension and the scalarized reward on the bottom row. The dimensions have very different shapes, Aptitude being binary and Competence and Gap continuous with arm- and user-dependent moments. After scalarization the resulting distributions differ across arms in both mean and variance, and their shape is arm-specific and hard to anticipate: some are close to Gaussian, others bi-modal. This motivates an algorithm that adapts its exploration per arm from observed data rather than from assumptions on the underlying distributions.

![](images/e7995a0da46d8d9a77e54dfd59bb9774c92f20f196cd12d552b8967c0e5308d2.jpg)  
Fig. 5: Reward distributions per arm for a given user (level 3) in the A+C+G setting with equal weights. Rows: individual objective dimensions and scalarized reward. Columns: difficulty levels (arms).

Group recommendation - MovieLens and Steam: These experiments use the two-objective formulation of [15] (relevance and pairwise disagreement) and show that the proposed meta-algorithm adaptively follows the best underlying base strategy.

![](images/a3602f8dfb948fa02e4a6f90b7ed4d612dffeb3d2c82594c89234a23e6548509.jpg)  
Fig. 6: MovieLens (top row) and Steam (bottom row): Relevance and Pairwise Disagreement. Error bars are clipped in (a) and (d).

On MovieLens (Figs. 6a-6c), rewards lie in [0, 5] (not normalized), so b = 1 underestimates the true support. UCB1’s bonus is then moderate and provides sufficient exploration, while UCB-V2 overexplores as its bonus scales with the empirical variances, which are large (> 1). UCB-minV2 identifies this regime and aligns with UCB1 throughout the horizon, also matching it on CTR (Fig. 6b) and Hit Rate@1 (Fig. 6c).

TABLE I: Final cumulative regret at the horizon $T$ of each experiment (mean over the 50 runs), computed from the plotted curves of Figs. 1, 4, and $6 ;$ lowest value per column in bold. UCB1-Tuned, Meta-1, and Meta-2 are not reported, as they have no regret guarantee. Synthetic: low-variance, high-variance, and heterogeneous regimes (Figs. 1a-1c), with $T = 2 0 { \mathrm { k } }$ , 20k, and 40k. Education: objective sets A, A+P, and $\mathrm { A { + } C { + } G . \Omega ^ { \ } } ^ { \mathrm { 6 } \mathrm { + } \mathrm { C } } ,$ : not shown in the figures of this dataset.
<table><tr><td rowspan="2">Algorithm</td><td colspan="3">Synthetic</td><td colspan="3">Education  $( T = 3 0 0 )$ </td><td rowspan="2">MovieLens</td><td rowspan="2">Steam  $T = 3 0 \mathrm { { k } }$ </td></tr><tr><td>Low</td><td>High</td><td>Hetero.</td><td>A</td><td>A+P</td><td> $\mathbf { A { + } C { + } G }$ </td></tr><tr><td>UCB1</td><td>1005.5</td><td>1064</td><td>1373</td><td>38.0</td><td>21.1</td><td>37.8</td><td>234</td><td>1381</td></tr><tr><td>UCB1-Normal</td><td>271.7</td><td>4124</td><td>6490</td><td>7.4</td><td>18.4</td><td>19.9</td><td>450</td><td>638</td></tr><tr><td>UCB-V</td><td>776.9</td><td>3248</td><td>3729</td><td>67.3</td><td>25.3</td><td>46.9</td><td>908</td><td>1086</td></tr><tr><td>UCB-V2</td><td>94.5</td><td>2323</td><td>851</td><td>38.4</td><td>20.4</td><td>37.7</td><td>406</td><td>613</td></tr><tr><td>EUCBV</td><td>172.9</td><td>1621</td><td>965</td><td></td><td>一</td><td>一</td><td></td><td>一</td></tr><tr><td>UCB-minV1</td><td>742.0</td><td>969</td><td>1267</td><td></td><td></td><td>一</td><td></td><td>一</td></tr><tr><td>UCB-minV2</td><td>92.6</td><td>958</td><td>799</td><td>38.2</td><td>20.4</td><td>37.7</td><td>233</td><td>612</td></tr></table>

On Steam (Figs. 6d-6f), rewards are in [0, 1] and per-arm variances are small, a regime where varianceaware exploration is optimal: among UCB1, UCB-V, and UCB-V2, UCB-V2 achieves the lowest regret (Fig. 6d) and UCB-minV2 closely tracks it, consistently across all metrics (Figures 6e-6f); UCB1-Tuned reaches a lower regret on this dataset. This contrast with MovieLens illustrates the dataset-dependent nature of the optimal exploration strategy and the robustness of UCB-min.

Effect of Reward Normalization on MovieLens: Figures 7a-7b compare UCB-type algorithms on MovieLens under raw rewards in [0, 5] and rewards normalized to [0, 1] by dividing by the maximum rating, with b = 1 in both cases.

![](images/a037c370355a8d39533086990f5da576244530f78748868d29f241eccdf36f68.jpg)  
Fig. 7: Effect of normalization (a: raw rewards in [0, 5], b: normalized to [0, 1]) on MovieLens cumulative regret.

The two settings induce different variance regimes. With raw rewards, empirical variances are large (> 1), placing the problem in a high-variance regime where UCB1 outperforms UCB-V2, which overexplores. After normalization, variances are reduced by a factor of 25 and UCB-V2 becomes competitive. In both cases, UCB-minV2 identifies the prevailing regime and tracks the better base algorithm throughout the horizon, confirming the robustness of the per-arm minimum rule.

Normalization does not improve the best achievable cumulative regret either. It is thus not a straight forward remedy to variance heterogeneity and, critically, requires knowing the reward upper bound in

advance, which is unavailable online.

Upper Bound Reward Effect: Figures 8a-8b compare the effect of the support bound b on MovieLens regret: $b = 1$ (used throughout) vs. $b = 5$ (the true support [0, 5]). With $b = 1$ , UCB1 and UCB-minV2 reach substantially lower regret than UCB-V2, whose variance-dependent bonus already over-explores. With $b = 5 ,$ , all bonuses grow proportionally and every method reaches near-linear regret, confirming that a large b, even well-specified, penalizes all UCB-type algorithms alike, variance-aware or not, and motivating $b = 1$ as a default. The minimum rule is especially useful here: even when b underestimates the support, it picks the tighter per-arm bonus.

![](images/d41f7db56f54e7c53cc375d88cf2817deaa1304ef24894cd264b1a6331c5a11b.jpg)  
Fig. 8: Effect of the support bound b (a: b = 1, b: $b = 5 ,$ one group) on MovieLens cumulative regret.

## VI. CONCLUSION

This paper studied exploration in bandits whose arms have different variance regimes. We showed that the scalarization step in multi-objective optimization can lead to different variance levels across arms, making the choice between variance-independent and variance-aware exploration both arm- and instancedependent. Neither UCB1 nor UCB-V2 uniformly dominates the other: their relative performance depends on the variance regime, the covariance structure of the objectives, and the reward support bound. We proposed UCB-MIN, a per-arm adaptive meta-algorithm that selects, for each arm and at each round, the tighter of the two exploration bonuses. It requires no prior knowledge of the variance regime, no reward normalization, and is empirically robust to an underestimated reward bound. Experiments on synthetic data, an educational tutoring task, and two group recommendation benchmarks confirmed that it consistently tracks the best base algorithm across diverse variance regimes and datasets.

Future work includes characterizing when minimum-of-bounds meta-algorithms achieve best-of-bothworlds regret, in particular whether the guarantee holds at the standard $c = 1 . 2$ of [9] and whether UCB-minV2’s strong empirical behavior admits a matching bound. Other directions include experiments with other weight vectors and more objectives on the real-world datasets.

## ACKNOWLEDGMENT

This work was partly supported by DataGEMS, funded by the European Union’s Horizon Europe Research and Innovation Programme, under grant agreement number 101188416.

Generative AI tools were used for language editing and for suggesting related work, coding, and initial proof ideas; all technical content and results are the authors’ own.

## REFERENCES

[1] S. S. Villar, J. Bowden, and J. Wason, “Multi-armed bandit models for the optimal design of clinical trials: benefits and challenges,” Statistical Science, vol. 30, no. 2, pp. 199–215, 2015.

[2] M. Aziz, E. Kaufmann, and M.-K. Riviere, “On multi-armed bandit designs for dose-finding clinical trials,” Journal of Machine Learning Research, vol. 22, no. 14, pp. 1–38, 2021.

[3] L. Li, W. Chu, J. Langford, and R. E. Schapire, “A contextual-bandit approach to personalized news article recommendation,” in Proc. 19th Int. Conf. World Wide Web, 2010, pp. 661–670.

[4] N. Silva, H. Werneck, T. Silva, A. C. M. Pereira, and L. Rocha, “Multi-armed bandits in recommendation systems: A survey of the state-of-the-art and future directions,” Expert Systems with Applications, vol. 197, p. 116669, 2022.

[5] J. Zuo and C. Joe-Wong, “Combinatorial multi-armed bandits for resource allocation,” in 2021 55th Annual Conference on Information Sciences and Systems (CISS). IEEE, 2021, pp. 1–4.

[6] E. M. Schwartz, E. T. Bradlow, and P. S. Fader, “Customer acquisition via display advertising using multi-armed bandit experiments,” Marketing Science, vol. 36, no. 4, pp. 500–522, 2017.

[7] D. Bouneffouf and I. Rish, “A survey on practical applications of multi-armed and contextual bandits,” arXiv preprint arXiv:1904.10040, 2019.

[8] P. Auer, N. Cesa-Bianchi, and P. Fischer, “Finite-time analysis of the multiarmed bandit problem,” Mach. Learn., vol. 47, no. 2–3, pp. 235–256, 2002.

[9] J.-Y. Audibert, R. Munos, and C. Szepesvari, “Tuning bandit algorithms in stochastic environments,” in´ Proc. Int. Conf. Algorithmic Learn. Theory. Springer, 2007, pp. 150–165.

[10] Y. Fan, Y. Han, J. Lv, X. Xu, and Z. Zhou, “Precise asymptotics and refined regret of variance-aware UCB,” in Adv. Neural Inf. Process. Syst., vol. 38, 2025, pp. 160 795–160 837.

[11] A. Lacerda, “Multi-objective ranked bandits for recommender systems,” Neurocomputing, vol. 246, pp. 12–24, 2017.

[12] Y. Zhao, Y. Wang, Y. Liu, X. Cheng, C. C. Aggarwal, and T. Derr, “Fairness and diversity in recommender systems: a survey,” ACM Trans. Intell. Syst. Technol., vol. 16, no. 1, pp. 1–28, 2025.

[13] R. Mehrotra, N. Xue, and M. Lalmas, “Bandit based optimization of multiple objectives on a music streaming platform,” in Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 2020, pp. 3224–3233.

[14] N. Bouarour, I. Benouaret, and S. Amer-Yahia, “Multi-objective test recommendation for adaptive learning,” in Trans. Large-Scale Data- Knowl.-Centered Syst. LVI. Springer, 2024, pp. 1–36.

[15] S. Amer-Yahia, S. Basu Roy, A. Chawla, G. Das, and C. Yu, “Group recommendation: Semantics and efficiency,” Proc. VLDB Endow., vol. 2, no. 1, pp. 754–765, 2009.

[16] J.-Y. Audibert, R. Munos, and C. Szepesvari, “Exploration–exploitation tradeoff using variance estimates in multi-armed´ bandits,” Theor. Comput. Sci., vol. 410, no. 19, pp. 1876–1902, 2009.

[17] M. M. Drugan and A. Nowe, “Designing multi-objective multi-armed bandits algorithms: A study,” in´ Proc. Int. Joint Conf. Neural Netw. IEEE, 2013, pp. 1–8.

[18] E. Turgay, D. Oner, and C. Tekin, “Multi-objective contextual bandit problem with similarity information,” in International Conference on Artificial Intelligence and Statistics. PMLR, 2018, pp. 1673–1681.

[19] W. R. Thompson, “On the likelihood that one unknown probability exceeds another in view of the evidence of two samples,” Biometrika, vol. 25, no. 3/4, pp. 285–294, 1933.

[20] S. Agrawal and N. Goyal, “Analysis of Thompson sampling for the multi-armed bandit problem,” in Proceedings of the 25th Annual Conference on Learning Theory (COLT), ser. JMLR Workshop and Conference Proceedings, vol. 23, 2012, pp. 39.1–39.26.

[21] A. Garivier and O. Cappe, “The KL-UCB algorithm for bounded stochastic bandits and beyond,” in´ Proc. 24th Annu. Conf. Learn. Theory. JMLR Workshop and Conference Proceedings, 2011, pp. 359–376.

[22] S. Mukherjee, K. Naveen, N. Sudarsanam, and B. Ravindran, “Efficient-UCBV: An almost optimal algorithm using variance estimates,” in Proc. AAAI Conf. Artif. Intell., vol. 32, no. 1, 2018.

[23] D. M. Roijers, P. Vamplew, S. Whiteson, and R. Dazeley, “A survey of multi-objective sequential decision-making,” Journal of Artificial Intelligence Research, vol. 48, pp. 67–113, 2013.

[24] S. Q. Yahyaa, M. M. Drugan, and B. Manderick, “Annealing-Pareto multi-objective multi-armed bandit algorithm,” in IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning. IEEE, 2014, pp. 1–8.

[25] R. Busa-Fekete, B. Szor¨ enyi, P. Weng, and S. Mannor, “Multi-objective bandits: Optimizing the generalized Gini index,”´ in International Conference on Machine Learning. PMLR, 2017, pp. 625–634.

[26] U. Siddique, P. Weng, and M. Zimmer, “Learning fair policies in multi-objective (deep) reinforcement learning with average and discounted rewards,” in International Conference on Machine Learning. PMLR, 2020, pp. 8905–8915.

[27] A. Cutkosky, A. Das, and M. Purohit, “Upper confidence bounds for combining stochastic bandits,” arXiv preprint arXiv:2012.13115, 2020.

[28] R. Arora, T. V. Marinov, and M. Mohri, “Corralling stochastic bandit algorithms,” in Proc. Int. Conf. Artif. Intell. Statist. PMLR, 2021, pp. 2116–2124.

[29] W. C. Cheung and L. Lyu, “Leveraging (Biased) information: Multi-armed bandits with offline data,” in Proc. 41st Int. Conf. Mach. Learn., vol. 235. PMLR, 2024, pp. 8286–8309.

[30] V. Mnih, “Efficient stopping rules,” Master’s thesis, University of Alberta, 2008.

[31] C. Gentile, S. Li, and G. Zappella, “Online clustering of bandits,” in Proceedings of the 31st International Conference on Machine Learning, vol. 32. PMLR, 2014, pp. 757–765.

[32] S. Li, C. Gentile, A. Karatzoglou, and G. Zappella, “Data-dependent clustering in exploration-exploitation algorithms,” arXiv preprint arXiv:1502.03473, 2015.

[33] N. Korda, B. Szor¨ enyi, and S. Li, “Distributed clustering of linear bandits in peer to peer networks,” in ´ Proceedings of the 33rd International Conference on Machine Learning, vol. 48. PMLR, 2016, pp. 1301–1309.

[34] K. Mahadik, Q. Wu, S. Li, and A. Sabne, “Fast distributed bandits for online recommendation systems,” in Proceedings of the 34th ACM International Conference on Supercomputing, 2020, pp. 4:1–4:13.

[35] Z. Wang, A. Lamb, E. Saveliev, P. Cameron, J. Zaykov et al., “Results and insights from diagnostic questions: The NeurIPS 2020 education challenge,” in NeurIPS 2020 Competition and Demonstration Track. PMLR, 2021, pp. 191–205.

[36] F. M. Harper and J. A. Konstan, “The MovieLens datasets: History and context,” ACM Transactions on Interactive Intelligent Systems, vol. 5, no. 4, pp. 19:1–19:19, 2015.

## APPENDIX A

## PROOF OF THEOREM 1

Proof. Consider a UCB bandit π with indexes $B _ { a , s , t }$ , where $B _ { a , s , t }$ is the index of arm a at step t when a has been pulled s times (here s counts pulls, unrelated to the superscript (s) of scalarized quantities), with $a ^ { \star }$ an optimal arm and $\mu ^ { \star }$ its expected reward. The approach of [8], [16] to bound the expected regret $R ( T )$ of UCB1 (resp. UCB-V) is to decompose it as $\begin{array} { r } { R ( T ) = \sum _ { a } \Delta _ { a } \mathbb { E } ( N _ { a } ( T ) ) } \end{array}$ . They bound $\mathbb { E } ( N _ { a } ( T ) )$ for suboptimal arms by noting that, under the UCB rule, arm a is pulled at step t only if its index is at least that of $a ^ { \star }$ , which requires one of two “bad” events: (i) the index of a is larger than $\mu ^ { \star }$ , or (ii) the index of $a ^ { \star }$ is smaller than $\mu ^ { \star }$ . Fixing some value u to be tuned later and bounding the probability of the two events for each step t after a has been pulled u times, any UCB bandit π must satisfy:

$$
\begin{array} { r l r } {  { \mathbb { E } ( N _ { a } ( T ) ) \le u + \sum _ { t = u + 1 } ^ { T } \mathbb { P } \big ( \exists u \le s \le t - 1 : B _ { a , s , t } > \mu ^ { \star } \big ) } } \\ & { } & { \quad \quad + \sum _ { t = u + 1 } ^ { T } \mathbb { P } \big ( \exists 1 \le s \le t - 1 : B _ { a ^ { \star } , s , t } \le \mu ^ { \star } \big ) } \end{array}\tag{8}
$$

This equation appears in slightly different forms, each upper bounding every term of Eq. (8), in [16, Theorem 2] and [8, Proof of Theorem 1]; both papers bound the expected regret of UCB1 and UCB-V by bounding each term of the sum.

We will denote by $u _ { a } ^ { ( 1 ) } , B _ { a , s , t } ^ { ( 1 ) } , U _ { a } ^ { ( 1 ) } ( T )$ the threshold value, index, and upper bound on $\mathbb { E } ( N _ { a } ( T ) )$ obtained in [8] for UCB1, and will use similarly $u _ { a } ^ { ( V ) } , B _ { a , s , t } ^ { ( V ) }$ and $U _ { a } ^ { ( V ) } ( T )$ for UCB-V, while the version without superscript will be used for UCB-minV1. So it suffices to prove that $\mathbb { E } ( N _ { a } ( T ) ) \le U _ { a } ^ { ( 1 ) } ( T ) +$ $U _ { a } ^ { ( V ) } ( T )$ to establish our regret bound for UCB-minV1 with $c = 1 . 2$

$$
\begin{array} { r l } & { u _ { a } ^ { ( 1 ) } = \left\lceil 8 b ^ { 2 } \log T / \Delta _ { a } ^ { 2 } \right\rceil , } \\ & { u _ { a } ^ { ( V ) } = \left\lceil 8 \left( \frac { \sigma _ { a } ^ { 2 } } { \Delta _ { a } ^ { 2 } } + \frac { 2 b } { \Delta _ { a } } \right) c \log T \right\rceil . } \end{array}
$$

Furthermore by monotonicity of the left expression w.r.t. $B \colon$

$$
\begin{array} { r l r } {  { \mathbb { P } \big ( \exists u \leq s \leq t - 1 : B _ { a , s , t } > \mu ^ { \star } \big ) \leq } } \\ & { } & { \operatorname* { m i n } \biggl ( \mathbb { P } \big ( \exists u \leq s \leq t - 1 : B _ { a , s , t } ^ { ( 1 ) } > \mu ^ { \star } \big ) , } \\ & { } & { \mathbb { P } \big ( \exists u \leq s \leq t - 1 : B _ { a , s , t } ^ { ( V ) } > \mu ^ { \star } \big ) \biggr ) } \end{array}\tag{9}
$$

and by union bound

$$
\begin{array} { r l r } { \mathbb { P } \big ( \exists s \leq t - 1 : B _ { a ^ { \star } , s , t } \leq \mu ^ { \star } \big ) \leq } \\ & { } & { \mathbb { P } \big ( \exists s \leq t - 1 : B _ { a ^ { \star } , s , t } ^ { ( 1 ) } \leq \mu ^ { \star } \big ) } \\ & { } & { \quad + \mathbb { P } \big ( \exists s \leq t - 1 : B _ { a ^ { \star } , s , t } ^ { ( V ) } \leq \mu ^ { \star } \big ) } \end{array}\tag{10}
$$

Since [8] and [16] obtain their bound by bounding all the probabilities appearing in the sums for their respective choice of $u ,$ combining Equations (8), (9) and (10) with $u = \operatorname* { m a x } ( u _ { a } ^ { \top } , u _ { a } ^ { ( V ) } )$ yields a sum whose terms all appear in the corresponding sums for UCB1 or UCB-V. The expected number of pulls of suboptimal arms in UCB-minV1, hence its expected regret, is therefore at most the sum of those for UCB1 and UCB-V, which gives the result for $c = 1 . 2$ in the Theorem.

For the tighter result, we prove that for $c \geq 4 ~ \mathrm { U C B - m i n V 1 }$ does not pull arm a more than a few times above the best of the two bounds above, namely $\begin{array} { r } { \mathbb { E } ( N _ { a } ( T ) ) \le \operatorname* { m i n } \bigl ( U _ { a } ^ { ( \bar { 1 } ) } ( T ) , U _ { a } ^ { ( V ) } ( T ) \bigr ) + O ( 1 ) } \end{array}$ . We first bound by a constant the last sum in Eq. (8), then bound the first two terms.

Recall from [8] that by Chernoff-Hoeffding, $\mathbb { P } ( B _ { a ^ { \star } , s , t } ^ { ( 1 ) } \le \mu ^ { \star } ) \le t ^ { - 4 }$ for all s and t. When $c \geq 4$ , the empirical Bernstein inequality $[ 1 6 , \mathrm { E q . } ~ ( 7 )$ of Theorem 1] provides a similar concentration bound for UCB-V, so we can assume $\mathbb { P } ( B _ { a ^ { \star } , s , t } ^ { ( V ) } \le \mu ^ { \star } ) \le 3 t ^ { - 4 }$ . For any choice of u, we next show that the last term of Eq. (8) is bounded by a constant. For all t and s, we have by union bound:

$$
\mathbb { P } ( B _ { a ^ { \star } , s , t } \leq \mu ^ { \star } ) \leq \mathbb { P } ( B _ { a ^ { \star } , s , t } ^ { ( 1 ) } \leq \mu ^ { \star } ) + \mathbb { P } ( B _ { a ^ { \star } , s , t } ^ { ( V ) } \leq \mu ^ { \star } ) \leq 4 t ^ { - 4 } .
$$

Consequently and decomposing by union bound:

$$
\begin{array} { r l r } {  { \sum _ { t = u + 1 } ^ { T } \mathbb { P } ( \exists 1 \leq s \leq t - 1 : B _ { a ^ { \star } , s , t } \leq \mu ^ { \star } ) } } \\ & { } & { \leq \displaystyle \sum _ { t = u + 1 } ^ { T } \sum _ { s = 1 } ^ { t - 1 } \mathbb { P } ( B _ { a ^ { \star } , s , t } \leq \mu ^ { \star } ) \leq \sum _ { t = u + 1 } ^ { T } \sum _ { s = 1 } ^ { t - 1 } 4 t ^ { - 4 } } \end{array}
$$

This proves that

$$
\sum _ { t = u + 1 } ^ { T } \mathbb { P } ( \exists 1 \leq s \leq t - 1 : B _ { a ^ { \star } , s , t } \leq \mu ^ { \star } ) \in O ( 1 )\tag{11}
$$

To fix a choice for u, we compare the two bounds: if $U _ { a } ^ { ( 1 ) } \leq U _ { a } ^ { ( V ) }$ , we set $u = u _ { a } ^ { ( 1 ) }$ . Otherwise, we set $u = u _ { a } ^ { ( V ) }$ . As we are only summing terms that appear in the smallest of the two bounds, possibly replacing some of these terms by smaller terms from the other bound (according to Eq. (9)),

$$
u + \sum _ { t = u + 1 } ^ { T } \mathbb { P } \big ( \exists u \le s \le t - 1 : B _ { a , s , t } > \mu ^ { \star } \big ) \le \operatorname* { m i n } \big ( U _ { a } ^ { ( 1 ) } , U _ { a } ^ { ( V ) } \big )
$$

Combining with Eq. (11), we deduce that

$$
\begin{array} { r } { \mathbb { E } ( N _ { a } ( T ) ) \le \operatorname* { m i n } \bigl ( U _ { a } ^ { ( 1 ) } ( T ) , U _ { a } ^ { ( V ) } ( T ) \bigr ) + O ( 1 ) } \end{array}
$$

which concludes our proof of the tighter result in the Theorem, obtained for $c \geq 4$

## APPENDIX B METRICS DESCRIPTION

a) Bandit performance:

• Cumulative Regret, as defined in Eq. (3), measures decision suboptimality (lower is better).

b) Ranking and interaction: At round t, let $\mathcal { L } _ { t } = ( a _ { 1 } , \ldots , a _ { k } )$ be the recommendation list (last k recommendations) and $s \subseteq A$ the truly relevant arms (ground truth). For $k \in \{ 1 , 3 , 5 \}$ we define Precision $\ @ \mathbf { k } = | \{ a _ { 1 } , \ldots , a _ { k } \} \cap \mathcal { S } | / k$ and Hit $\mathbf { R a t e } @ \mathbf { k } = \mathbf { 1 } \big ( | \{ a _ { 1 } , \ldots , a _ { k } \} \cap \mathcal { S } | > 0 \big ) ;$ ; the paper reports Hit Rate@1, the remaining values being available online.<sup>4</sup> These assess whether relevant items appear among the last k recommendations; we take S as the top-M arms by true scalar mean $\mu _ { a } ^ { ( s ) }$ , with $M = 5$

c) Engagement and domain: $\begin{array} { r } { \mathbf { C T R } { } = ~ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { 1 } \{ \mathrm { r e l } _ { t } ~ > ~ 0 \} } \end{array}$ , where rel<sub>t</sub> is the observed relevance of the recommended item at t (higher is better); Skill progression (NCC): number of difficulty levels mastered, a level requiring N consecutive correct responses (higher is better).