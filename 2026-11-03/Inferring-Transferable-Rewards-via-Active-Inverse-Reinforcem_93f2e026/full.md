# Inferring Transferable Rewards via Active Inverse Reinforcement Learning

Victor Villin University of Neuchˆatel

Christos Dimitrakakis<sup>∗</sup> University of Neuchˆatel

Till Freihaut<sup>†</sup> University of Zurich

Maryam Kamgarpour<sup>∗</sup> EPFL

Andreas Schlaginhaufen<sup>†</sup> EPFL

Giorgia Ramponi<sup>∗</sup> University of Zurich

## Abstract

Inverse Reinforcement Learning (IRL) aims to recover expert rewards that transfer across environments with difering dynamics. Prior work has shown that such transfer is possible when gathering expert demonstrations across suficiently diferent dynamics. In practice, however, expert demonstrations are costly, making it critical to select the right dynamics to reduce the quantity of expert data required for learning transferable rewards. We propose a two-phase approach: we first cycle through the available dynamics to obtain a reliable reward estimate, then actively select dynamics to reduce our uncertainty about the expert reward. Crucially, this yields novel sample complexity guarantees on transfer that hold for any dynamics, with a rate independent of the number of available dynamics. Empirically, our method significantly reduces the number of expert demonstrations needed to learn a transferable reward.

## 1 INTRODUCTION

Learning a transferable representation of expert behavior makes it possible to predict how the expert would act in previously unseen circumstances. Inverse Reinforcement Learning (IRL) [1] formalizes this by inferring a reward function from expert demonstrations. Unlike imitation learning methods such as Behavior Cloning (BC) [2], which directly replicate observed behavior, IRL aims to recover the underlying objective the expert is optimizing. In contrast to a cloned policy tied to its training dynamics, a reward can, in princi ple, transfer to unseen dynamics.

Despite its appeal, IRL is fundamentally ill-posed [1]. Under entropy regularization [3], the standard setting in identifiability works [4, 5], demonstrations from a fixed Markov decision process (MDP) identify the reward only up to subspace specific to that MDP. Consequently, such rewards transfer only to dynamics whose identifiable subspace suficiently overlaps with that of the original MDP [6]. Recent works show that observing expert demonstrations in as few as two suficiently diferent dynamics resolves this ambiguity [4, 5, 6]. However, these works leave open how to actively select such dynamics to reduce the ambiguity with as few expert demonstrations as possible.

In practice, expert demonstrations are costly, and only a fixed set of dynamics is available. We must therefore choose carefully which dynamics to query the expert in. This raises the following central question:

## How should we select dynamics to learn a transferable reward from a minimal number of expert demonstrations?

To address this question, we propose an active IRL setting, in which the learner chooses the dynamics under which the expert demonstrates the task. This setting is natural whenever reconfiguring the environment is cheap compared to collecting expert demonstrations. For example, a robotics lab can cheaply vary the initial placement, mass, or friction of manipulated objects [7], or a driving simulator can vary road conditions [8]; collecting demonstrations, by contrast, requires a skilled human operator for each one [9].

Closest to our work, Buening et al. [10] adaptively design dynamics to identify the expert’s reward. They empirically show that, compared to randomly chosen environments, this yields more transferable rewards from fewer demonstrations, but provide no theoretical guarantees. To the best of our knowledge, our work provides the first active IRL algorithms with sample complexity guarantees for learning transferable rewards through dynamics selection.

## Contributions. Our main contributions are:

1. We formalize active IRL as a sequential problem: over n rounds, the learner chooses dynamics from a finite set on which to request an expert demonstration, aiming to recover a reward that transfers as much as possible across the set (Equation (1)).

2. We derive novel confidence sets for the expert reward (Lemma 3.1) and show that transfer to each dynamics is controlled by how informative they are relative to the information already accumulated (Lemma 3.2).

3. This yields a two-phase algorithm: cycle through the set of dynamics until the reward estimate is reliable (Lemma 3.3), then query the expert where information gain is greatest. After n rounds of dy namics selection and expert demonstrations, this results in a transfer error of $\tilde { \mathcal { O } } ( 1 / n )$ across the set of available dynamics, independent of its cardinality (Theorem 3.6). The same rate extends to any dynamics, up to an irreducible bias (Theorem 3.7).

4. We validate our approach empirically against imitation learning and IRL baselines, as well as dynamics design for IRL [10]. Our active approach learns an equally transferable reward using up to five times fewer demonstrations than the baselines, and its learned reward generalizes best to unseen dynamics (Section 4).

## 2 PROBLEM FORMULATION

Notations. Let $\left\| \cdot \right\| _ { 2 }$ denote the Euclidean norm. For a symmetric positive semidefinite matrix $X \in \mathbb { R } ^ { d \times d } ,$ $\| X \| _ { 2 }$ denotes its spectral norm, $X ^ { \dagger }$ its Moore-Penrose inverse, and $\| v \| _ { X } : = \sqrt { \langle v , X v \rangle }$ the associated Mahalanobis norm of a vector $v \in \mathbb { R } ^ { d }$ . For a finite set $\mathcal { Z }$ $\Delta ^ { \mathcal { Z } }$ denotes the probability simplex over $\mathcal { Z }$

Regularized MDPs. A regularized MDP is a tuple $\mathcal { M } \ = \ ( \mathcal { S } , \mathcal { A } , T , \nu _ { 1 } , r , H , \eta )$ Here, and are finite state and action spaces, $T : \mathcal { S } \times \mathcal { A }  \Delta ^ { s }$ is the transition function, $\nu _ { 1 } ~ \in ~ \Delta ^ { s }$ denotes the initial state distribution, $r : S \times \mathcal { A } $ R denotes the reward, and $H \in \mathbb { N } ^ { + }$ is the horizon. We consider entropyregularized MDPs, with $\eta > 0$ denoting the inverse temperature that controls the strength of the regularization. Starting from state $s _ { 1 } \sim \nu _ { 1 }$ , an agent interacting with the MDP takes at each timestep h an action $a _ { h } \in A .$ receives the reward $r ( s _ { h } , a _ { h } )$ and the environment transitions to a new state $s _ { h + 1 } \sim T ( \cdot \mid s _ { h } , a _ { h } )$

Denoting the dynamics by $P ~ = ~ ( \nu _ { 1 } , T )$ , the optimal solution of  is a policy $\pi ~ = ~ ( \pi _ { h } ) _ { h = 1 } ^ { H }$ with $\pi _ { h } : \mathcal { S } \to \Delta ^ { \mathcal { A } }$ maximizing the regularized value $\begin{array} { r } { V _ { P } ^ { \pi } ( r ) : = \mathbb { E } _ { P } ^ { \pi } \left| \sum _ { h = 1 } ^ { H } r ( s _ { h } , a _ { h } ) - \eta ^ { - 1 } \log \left( \pi _ { h } ( a _ { h } \mid s _ { h } ) \right) \right| } \end{array}$ A state s is unreachable at timestep h if no policy visits it with positive probability at $h .$ As a convention, all policies follow an arbitrarily fixed action distribution (e.g., uniform) on unreachable states, which leaves trajectory laws and values unchanged. The MDP then admits a unique optimal policy [3]. For a reward r and dynamics $P ,$ we write $V _ { P } ^ { \star } ( r )$ and $\pi _ { P } ( r )$ for the corresponding optimal value and policy.

Setting. Given an unknown reward $r ^ { \mathrm { E } }$ , an expert follows the optimal policy $\pi _ { P } ^ { \mathrm { E } } : = \pi _ { P } ( r ^ { \mathrm { E } } )$ under any chosen dynamics P. We consider a fixed, finite family of MDPs $\{ \mathcal { M } _ { i } \}$ that share $r ^ { \mathrm { E } }$ but difer in their dynamics $\mathcal { P } : = \{ P _ { i } \}$ . A demonstration in $\mathcal { M } _ { i }$ is a trajectory $\boldsymbol { \tau } = \left( \left( \boldsymbol { s } _ { h } , \boldsymbol { a } _ { h } \right) \right) _ { h = 1 } ^ { H }$ generated by $\pi _ { P _ { i } } ^ { \mathrm { E } }$

We consider an active setting in which a learner selects dynamics sequentially, and an expert is queried after each selection. Over a budget of $n \in \mathbb { N } ^ { + }$ rounds, the learner selects at each round k dynamics $P _ { k } \in \mathcal { P }$ and observes a batch of m expert trajectories generated under $P _ { k }$ , for a total of nm trajectories, gathering them in a dataset $\textstyle { \mathcal { D } } ^ { \mathrm { E } } : = \bigcup _ { k = 1 } ^ { n } { \bigl \{ \tau _ { k , j } \bigr \} } _ { j = 1 } ^ { m }$

Objective. Given this budget, our goal is to select dynamics $\left( P _ { k } \right) _ { k = 1 } ^ { n }$ from $\mathcal { P }$ and estimate a reward r from $\mathcal { D } ^ { \mathrm { E } }$ that is transferable across $\mathcal { P }$ . We quantify transferability by the worst-case suboptimality,

$$
\operatorname * { m a x } _ { P \in \mathcal { P } } \mathrm { S u b O p t } ( r , P ) \ : = \ : \operatorname* { m a x } _ { P \in \mathcal { P } } V _ { P } ^ { \star } ( r ^ { \mathrm { E } } ) - V _ { P } ^ { \pi _ { P } ( r ) } ( r ^ { \mathrm { E } } ) ,\tag{1}
$$

and we seek a reward r minimizing (1). This objective captures the ground-truth performance of the recovered optimal policy $\pi _ { P } ( r )$ across and coincides with the notion of transferability studied by Schlaginhaufen and Kamgarpour [6] for fixed dynamics. Furthermore, it can be viewed as a regularized analogue of the imitation gap studied in imitation learning [11].

Structural assumption. Before presenting the algorithm for selecting the dynamics $P _ { k }$ and estimating the reward, we introduce our main structural assumption, which enables eficient estimation of the expert reward $r ^ { \mathrm { E } }$ from expert demonstrations $\mathcal { D } ^ { \mathrm { E } }$

Assumption 2.1 (Expert bounded linear rewards). Let $\phi : \mathcal { S } \times \mathcal { A }  \mathbb { R } ^ { d }$ be a known feature map with $\| \phi ( s , a ) \| _ { 2 } \leq B _ { \phi }$ for all $( s , a ) \in S { \times } A$ , and let $\Theta : = \{ \theta \in$ $\mathbb { R } ^ { d } : \| \dot { \theta } \| _ { 2 } \le B _ { \theta } \}$ , where $B _ { \phi } , B _ { \theta } > 0$ are known constants. We assume the expert reward is linear in unknown parameters $\theta ^ { \mathrm { E } } \in \Theta$ , i.e. $r ^ { \mathrm { E } } ( s , a ) = \langle \theta ^ { \mathrm { E } } , \phi ( s , a ) \rangle$ 1

Linear rewards are standard in RL [12] and inverse RL [1]. We index rewards $r _ { \theta } = \langle \theta , \phi \rangle$ by their parameters θ, and write $\begin{array} { r } { \phi ( \tau ) = \sum _ { ( s , a ) \in \tau } \phi ( s , a ) } \end{array}$ for trajectory features.

Algorithm 1 Active IRL   
1: input: set of dynamics ${ \mathcal { P } } _ { : }$ , warm-up length $n _ { 0 }$   
2: // Warm-up phase   
3: for round $k = 1 , \ldots , n _ { 0 }$ do   
4: Select $P _ { k }$ round-robin over $\mathcal { P }$ (6)   
5: Observe m trajectories in $P _ { k }$   
6: Estimate $\theta _ { n _ { 0 } }$ by minimizing (2) (Section 3.1)   
7: // Active phase   
8: for round $k = n _ { 0 } + 1 , \ldots , n$ do   
9: Actively select $P _ { k }$ via (7) (Section 3.2)   
10: Observe m trajectories in $P _ { k }$   
11: Estimate $\theta _ { k }$ by minimizing (2) (Section 3.1)

Finally, we assume that the expert is optimal for $\theta ^ { \mathrm { E } }$ with a known temperature. Appendix G extends our results to misspecified experts, e.g., when the assumed temperature is incorrect, in which case all our transferability results hold with an additional error proportional to the expert’s suboptimality.

## 3 ACTIVELY LEARNING TRANSFERABLE REWARDS

We now present our active method for learning transferable rewards minimizing (1), summarized in Algorithm 1. Its key idea is to query the expert under the most informative dynamics, i.e., those revealing parts of the reward that matter for transfer across $\mathcal { P }$ but remain poorly determined by previous demonstrations. The method proceeds in two phases. In a warm-up phase, the learner has too little data to tell which dynamics are informative, so it queries the expert under each available dynamics in turn. It then switches to an active phase, in which it uses its reward estimate to select the most informative dynamics.

To make this strategy precise, we first quantify the accuracy of reward estimates from adaptively collected demonstrations through ellipsoidal confidence bounds (Section 3.1). These bounds determine when the estimate is reliable enough to guide dynamics selection. In Section 3.2, we use them to construct the two-phase selection algorithm, before deriving transfer guarantees for the learned reward in Section 3.3.

## 3.1 Reward estimation

To select informative dynamics, we need to quantify how much information each dynamics provides about $\theta ^ { \mathrm { E } }$ , and how much has already been accumulated. We first introduce the loss and use its curvature to characterize the information provided under each dynamics.

We then show how the accumulated information yields an ellipsoidal confidence bound for $\theta ^ { \mathrm { E } }$ (Lemma 3.1). Proofs are deferred to Appendix B.

IRL loss. After k rounds, we estimate $\theta ^ { \mathrm { E } }$ by minimizing the empirical loss over Θ,

$$
L _ { k } ( \theta ) : = \sum _ { i = 1 } ^ { k } l _ { i } ( \theta ) , \quad l _ { i } ( \theta ) : = V _ { P _ { i } } ^ { \star } ( \Pi _ { P _ { i } } \theta ) - \langle \theta , \Pi _ { P _ { i } } \hat { \phi } _ { i } \rangle ,\tag{2}
$$

and write $\theta _ { k } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } L _ { k } ( \theta )$ . Here $P _ { i }$ are the dynamics selected at round $\begin{array} { r } { i , \hat { \phi } _ { i } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \phi ( \tau _ { i , j } ) } \end{array}$ is the empirical mean of the expert’s trajectory features, and $\Pi _ { P _ { i } }$ is the orthogonal projector onto the identifiable subspace $U _ { P _ { i } }$ defined below. Without the projections, each $l _ { i }$ is an entropy-regularized version of the minmax IRL loss [13] under $P _ { i }$

Curvature and identifiability. The optimal policy induced by a reward depends on the underlying dynamics. Hence, by selecting the dynamics under which the expert demonstrates, the learner chooses which parts of the reward can be estimated, and how eficiently. We characterize this through the Hessian of the optimal value function,

$$
\mathcal { H } _ { P } ( \theta ) : = \nabla _ { \theta } ^ { 2 } V _ { P } ^ { \star } ( \theta ) ,
$$

which is the information provided by P. It is also the curvature of the per-round loss $l _ { i }$ when $P = P _ { i } ,$ since it difers from $V _ { P _ { i } } ^ { \star }$ only by a linear term. For a direction $v \in \mathbb { R } ^ { d }$ , the curvature $\| v \| _ { \mathcal { H } _ { P } ( \theta ) } ^ { 2 }$ quantifies the sensitivity of the optimal policy $\pi _ { P } ( \theta )$ to reward perturbations along v. The larger the curvature, the fewer demonstrations under $P$ are needed to estimate the reward along v. With zero curvature, the reward along v cannot be estimated under $P ,$ , regardless of how many trajectories are collected. Such directions arise from potential shaping [14] and from reward changes at unreachable states. We therefore define the identifiable subspace $U _ { P } : = \mathrm { i m } \mathcal { H } _ { P } ( \theta )$ , the orthogonal complement of the zero-curvature directions.<sup>1</sup> Demonstrations under P only inform the components of $\theta ^ { \mathrm { E } }$ in $U _ { P }$ . The projections in (2) thus prevent sampling noise in $\hat { \phi } _ { i }$ from drifting the estimate along unidentifiable directions (Appendix B). Across all available dynamics, the identifiable components of $\theta ^ { \mathrm { E } }$ lie in the joint identifiable subspace $\begin{array} { r } { U : = \sum _ { P \in \mathcal { P } } U _ { P } } \end{array}$ , of dimension $d _ { U } \leq d _ { \cdot }$

Confidence sequence. After k rounds, the informa tion accumulated about $\theta ^ { \mathrm { E } }$ is given by the curvature of $L _ { k } \ ( 2 )$ at $\theta ^ { \mathrm { E } }$ and its smallest eigenvalue on $U$

$$
\mathcal { H } _ { k } : = \sum _ { i = 1 } ^ { k } \mathcal { H } _ { P _ { i } } ( \theta ^ { \mathrm { E } } ) , \qquad \lambda _ { k } : = \operatorname* { m i n } _ { v \in { U } \atop \| v \| _ { 2 } = 1 } \| v \| _ { \mathcal { H } _ { k } } ^ { 2 } ,
$$

where $\lambda _ { k }$ measures the information in the least informed identifiable direction. Our main estimation result is a confidence sequence for $\theta ^ { \mathrm { E } }$ , whose radius depends on the efective dimension

$$
d _ { \star } : = d _ { U } \operatorname* { m a x } _ { P \in \mathcal { P } } \left\| \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) ^ { \dagger / 2 } \Sigma _ { P } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) ^ { \dagger / 2 } \right\| _ { 2 } ,\tag{3}
$$

where $\begin{array} { r } { \Sigma _ { P } : = \mathrm { C o v } _ { P } ^ { \pi _ { P } ^ { \mathrm { E } } } \left( \Pi _ { P } \phi ( \tau ) \right) } \end{array}$ is the covariance of the projected expert trajectory features.

Lemma 3.1. Fix $n ~ \in ~ \mathbb { N } ^ { + } , ~ \delta ~ \in ~ ( 0 , 1 )$ , and $R : = { \bf \Omega }$ $\eta B _ { \phi } H ( H + 1 )$ . Let $n _ { \delta }$ be the first round $k \leq n$ such that

$$
\lambda _ { k } \ \geq \ { \frac { 3 2 R ^ { 2 } d _ { \star } } { ( 1 - e ^ { - 1 } ) ^ { 2 } m } } \log { \frac { 3 \eta d _ { \star } n m } { \delta } } ,\tag{4}
$$

with $n _ { \delta } = \infty$ if no such round exists. Then, with probability at least $1 - \delta _ { ; }$ , simultaneously for all $n _ { \delta } \le k \le n$

$$
\big \| \theta _ { k } - \theta ^ { E } \big \| _ { \mathcal { H } _ { k } } ^ { 2 } \ \leq \ \frac { 1 6 d _ { \star } } { ( 1 - e ^ { - 1 } ) ^ { 2 } m } \log \frac { 3 \eta d _ { \star } k m } { \delta } \ = : \ \beta _ { k } ^ { 2 } ( \delta ) .
$$

Proof sketch. We combine concentration of the projected empirical features under adaptive selection with the pseudo self-concordance of $L _ { k } ~ [ 1 5 ]$ After the warm-up, the localization approach of Schlaginhaufen and Kamgarpour [16] converts sampling variation into an ellipsoidal parameter-error bound. □

Lemma 3.1 quantifies the uncertainty in our reward estimate through a confidence ellipsoid shaped by the accumulated information $\mathcal { H } _ { k }$ . The ellipsoid is wide along directions where little information has been collected, and narrow where much has. Querying the expert under $P$ adds $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } )$ to the accumulated information, and is therefore informative when $P$ provides information along directions where the ellipsoid is wide.

The ellipsoid, however, is only guaranteed to contain $\theta ^ { \mathrm { E } }$ once enough information has been accumulated in every identifiable direction, as prescribed by the warmup condition (4). Since $L _ { k }$ is nonquadratic, its curvature changes with the reward parameters at a rate controlled by R [15]. The warm-up ensures that $\theta _ { k }$ lies in a region where the curvature remains comparable to that at $\theta ^ { \mathrm { E } }$ , allowing the approximately local quadratic analysis that yields the confidence bound [17, 18].

The efective dimension $d _ { \star }$ compares the variability of the expert’s features to the information they carry, and plays the role of the parameter dimension in the confidence radius. It satisfies $d _ { \star } \geq d _ { U } / \eta .$ , with equal ity under deterministic dynamics, where the projections also have no efect on the loss (2). Together with the curvature variation controlled by $R ,$ it determines the required information level (4). Through $R ^ { 2 }$ , the threshold has an explicit quartic dependence on the horizon H and quadratic dependence on the feature scale $B _ { \phi }$ . Up to logarithmic factors, both the threshold and the squared confidence radius $\beta _ { k } ^ { 2 } ( \delta )$ scale as $1 / m$ Larger batches therefore relax the warm-up requirement and shrink the confidence ellipsoid by reducing the sampling noise in the empirical features.

Lemma 3.1 thus splits dynamics selection into two goals: first accumulating enough information in every direction of $U$ to complete the warm-up, then adding information where it matters for transfer across ${ \mathcal { P } } _ { : }$ which we address next.

## 3.2 Dynamics selection

We now use the confidence sequence to select dynam ics so that the learned reward transfers across $\mathcal { P } \left( 1 \right)$ We bound the suboptimality under each $P \in \mathcal { P }$ , then derive a two-phase algorithm querying the dynamics where this bound is worst. Proofs are in Appendix C.

From information to suboptimality. Lemma 3.1 bounds the error $\theta _ { k } - \theta ^ { \mathrm { E } }$ in the geometry of the accumulated information $\mathcal { H } _ { k }$ . Transfer to $P ,$ however, is governed by the geometry of $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } )$ , which measures how sensitive the optimal policy under $P$ is to reward errors (Section 3.1). We compare the two through the information gain of $P$

$$
g _ { k } ( P ) : = \left\| \mathcal { H } _ { k } ^ { \dagger / 2 } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \mathcal { H } _ { k } ^ { \dagger / 2 } \right\| _ { 2 } ,\tag{5}
$$

which is large when P provides much information along directions where little has been accumulated. Errors along such directions remain large, and since they are important for transfer to $P ,$ they are amplified in the suboptimality under $P .$

Lemma 3.2. Fix $n ~ \in ~ \mathbb { N } ^ { + }$ and $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ With probability at least $1 - \delta$ , it holds for all $P \in \mathcal { P }$ and $n _ { \delta } \le k \le n$ that

$$
\mathrm { S u b O p t } ( \theta _ { k } , { \cal P } ) \ \leq \ g _ { k } ( { \cal P } ) \beta _ { k } ^ { 2 } ( \delta ) .
$$

Proof sketch. Let $u _ { k } : = \theta _ { k } - \theta ^ { \mathrm { E } }$ . The suboptimality is the Bregman divergence of $V _ { P } ^ { \star }$ between $\theta ^ { \mathrm { E } }$ and $\theta _ { k }$ , i.e., $\begin{array} { r } { \mathrm { S u b O p t } ( \theta _ { k } , P ) = \int _ { 0 } ^ { 1 } ( 1 - t ) \ \| u _ { k } \| _ { \mathcal { H } _ { P } ( \theta _ { k } - t u _ { k } ) } ^ { 2 } } \end{array}$ dt. On the event of Lemma 3.1, for every $P \in { \overrightarrow { P } }$ and $\mathit { \check { n } } _ { \delta } \leq k \leq n$

$$
\mathrm { S u b O p t } ( \theta _ { k } , P ) \stackrel { \mathrm { ( i ) } } { \sim } \| u _ { k } \| _ { \mathcal { H } _ { P } ( \theta ^ { \operatorname { E } } ) } ^ { 2 } \stackrel { \mathrm { ( i i ) } } { \leq } g _ { k } ( P ) \left\| u _ { k } \right\| _ { \mathcal { H } _ { k } } ^ { 2 } .
$$

(i) holds by pseudo self-concordance of $V _ { P } ^ { \star } .$ , since $\theta _ { k }$ is close to $\theta ^ { \mathrm { E } }$ after the warm-up. (ii) changes the geometry from $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } )$ to $\mathcal { H } _ { k }$ at a cost $g _ { k } ( P )$ . Lemma 3.1 bounds the last term by $\beta _ { k } ^ { 2 } ( \delta )$ □

Lemma 3.2 suggests that transfer is hardest to the dynamics with the most information left to gain, which are thus the most valuable to query. Since $\beta _ { k } ^ { 2 } ( \delta )$ does not depend on dynamics, our goal is to make $g _ { k } ( P )$

small across $\mathcal { P } .$ . However, $g _ { k } ( P )$ depends on the unknown $\theta ^ { \mathrm { E } }$ , and plugging in $\theta _ { k }$ is only reliable after the warm-up, so we first collect data passively.

Warm-up. During the warm-up, we cycle through the available dynamics using a round-robin rule that does not rely on reward estimates. Writing $k ( P )$ for the number of times $P$ was selected in the first k rounds, we choose the next dynamics according to

$$
P _ { k + 1 } \in \mathop { \mathrm { a r g } } \operatorname* { m i n } _ { P \in \mathcal { P } } k ( P ) .\tag{6}
$$

This simple rule guarantees that the warm-up condition is met within a bounded number of rounds.

Lemma 3.3. Fix $n \in \mathbb { N } ^ { + }$ and $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ . Under round-robin selection (6), the warm-up length satisfies

$$
n _ { \delta } \ \leq \ | \mathcal { P } | \left\lceil \frac { 2 R ^ { 2 } \beta _ { n } ^ { 2 } ( \delta ) } { \lambda _ { | \mathcal { P } | } } \right\rceil ,
$$

whenever the right-hand side is at most n.

Proof sketch. After k rounds, every $P \in \mathcal { P }$ was picked at least $\lfloor k / \left. \mathcal { P } \right. \rfloor$ times, so $\lambda _ { k } \geq \lfloor k / \mid \mathcal { P } \rfloor \rfloor \lambda _ { | \mathcal { P } | }$ □

The threshold (4) equals $2 R ^ { 2 } \beta _ { n } ^ { 2 } ( \delta )$ , and $\lambda _ { | \mathcal { P } | }$ is the information in the least-informed direction after one sweep over $\mathcal { P } .$ The bound thus counts the sweeps needed before Lemma 3.1 holds. It depends on the unknown $\theta ^ { \mathrm { E } }$ , although a loose $\theta ^ { \mathrm { E } } .$ -independent version can be computed (Appendix F). A similar dependence appears in active learning for maximum-likelihood estimation [17], where a passive phase provides an initial estimate of the Fisher information used to select informative labels. Here, the warm-up makes the estimated Hessians reliable enough to guide dynamics selection. Being worst-case, the bound is conservative, and Section 4 shows that the active phase performs well after a much shorter warm-up.

Active selection. After the warm-up, $\theta _ { k }$ is localized around $\theta ^ { \mathrm { E } }$ (Lemma 3.1). This is not yet enough to transfer well to dynamics that rely on weakly informed directions, but it is enough to estimate information gains, since the Hessians at $\theta _ { k }$ then match those at $\theta ^ { \mathrm { E } }$ up to a constant factor. We thus estimate $g _ { k } ( P )$ by ${ \hat { g } } _ { k } ( P )$ , defined as in (5) with Hessians at $\theta _ { k }$ instead of $\theta ^ { \mathrm { E } }$ , and query the dynamics with the largest estimated information gain,

$$
P _ { k + 1 } \in \mathop { \mathrm { a r g } } _ { P \in \mathcal { P } } \operatorname* { m a x } _ { \hat { g } _ { k } ( P ) . }\tag{7}
$$

Querying the selected dynamics adds its information to $\mathcal { H } _ { k }$ , lowering its information gain as well as that of every dynamics sharing its identifiable directions. Section 3.3 shows that this rule drives the worst-case suboptimality down across $\mathcal { P } _ { \cdot }$

Remark 3.4. Minimizing max<sub>P∈P</sub> $g _ { k } ( P )$ relates to Goptimal designs [19, 20]. If each $\mathcal { H } _ { P } ( \theta )$ were rank-one and parameter-independent, our rule (7) would reduce to greedy D-optimal design. More generally, Theorem 3.6 still holds if the spectral norm in the selection rule (7) is replaced by the trace or by log det $( I + \cdot )$ Our experiments use the latter.

Remark 3.5 (Ofline active learning). When sequential interaction is impractical, one can select all dynamics in advance by using $\theta _ { 0 } = 0$ in place of $\theta _ { k } ( \mathrm { A p p e n d i x } \mathrm { E } )$ ).

Computational complexity. The information $\begin{array} { r } { \mathcal { H } _ { P } ( \boldsymbol { \theta } ) = \eta \mathbb { E } \left[ A _ { P } ^ { \boldsymbol { \theta } } ( \boldsymbol { \tau } ) A _ { P } ^ { \boldsymbol { \theta } } ( \boldsymbol { \tau } ) ^ { \top } \right] } \end{array}$ is the second moment of the feature advantages $\bar { A } _ { P } ^ { \theta } ( \tau )$ along trajectories of $\pi _ { P } ( \theta )$ (Appendix B). It is estimated by sampling such trajectories and computing advantages through one policy evaluation. Since $U _ { P }$ does not depend on $\theta ,$ each projector $\Pi _ { P } = \mathcal { H } _ { P } ( 0 ) \mathcal { H } _ { P } ( 0 ) ^ { \dagger }$ is computable once at the start, using the uniform policy $\pi _ { P } ( 0 )$ (Appendix H). As in any IRL method, the cost of Algorithm 1 is dominated by the RL solves in lines 6 and 11. Active selection only adds line 9, which reuses the policies computed by IRL to estimate the information, whereas Buening et al. [10] require $\mathcal { O } ( k \lvert \mathcal { P } \rvert )$ RL solves at round k to choose dynamics.

## 3.3 Transfer guarantees

We now show that our two-phase approach yields a reward that transfers across $\mathcal { P }$ at a rate independent of $| \mathcal { P } |$ , and extend this guarantee to dynamics outside . Proofs of this section are deferred to Appendix D.

Transfer across . The following theorem shows that, as active rounds accumulate, less and less information remains to be gained under any $P \in \mathcal { P }$ , and the worst-case suboptimality decreases accordingly.

Theorem 3.6. Fix $n \in \mathbb { N } ^ { + } , \delta \in ( 0 , 1 )$ , and follow the active rule (7) after round $n _ { \delta }$ . With probability at least $1 - \delta$ , it holds for all $n _ { \delta } < k \leq n$ and $P \in \mathcal { P }$ that

$$
g _ { k } ( P ) \lesssim \left( 1 + \frac { m } { \eta d _ { \star } H ^ { 2 } } \right) \frac { d _ { U } \log ( k m ) } { k - n _ { \delta } } .
$$

If moreover $m \leq \eta d _ { \star } H ^ { 2 }$ and $n \geq 2 n _ { \delta }$

$$
\mathrm { S u b O p t } ( \theta _ { n } , P ) \ \lesssim \ \frac { d _ { \star } d _ { U } \log ( n m ) \log ( \eta d _ { \star } n m / \delta ) } { n m } .\tag{8}
$$

Proof sketch. For all $P \in \mathcal { P } , \mathcal { H } _ { P } ( \theta _ { k } )$ and $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } )$ are within an e factor of each other. An elliptical potential argument relating the spectral norm to the log-determinant bounds $g _ { k } ( P )$ for all $P \in \mathcal { P }$ . For $m \leq \eta d _ { \star } H ^ { 2 }$ , the front factor is at most 2, and $n \geq 2 n _ { \delta }$ gives $n - n _ { \delta } \ge n / 2$ . Lemma 3.2 then yields (8). □

Theorem 3.6 shows that, after the warm-up, the information gain of every $P \in \mathcal { P }$ decays inversely with the number of active rounds $k - n _ { \delta } .$ so that progressively no dynamics in $\mathcal { P }$ is left with much to reveal. With at most $\eta d _ { \star } H ^ { 2 }$ trajectories per round, and as long as the warm-up uses at most half the budget, the returned reward thus achieves a worst-case suboptimality of $\tilde { \mathcal { O } } \left( d _ { \star } d _ { U } / ( n m ) \right)$ over ${ \mathcal { P } } ,$ i.e., inversely proportional to the total number of trajectories.

Importantly, only afects the warm-up length (Lemma 3.3), not the suboptimality rate. Round robin instead pays for $| \mathcal { P } |$ at every round. If only one $P \in \mathcal { P }$ is informative and the others are redundant, our analysis only guarantees a suboptimality of $\tilde { \mathcal { O } } \left( d _ { \star } \left| \mathcal { P } \right| / ( n m ) \right)$ for round-robin, as it keeps querying dynamics with little information to ofer. Active selection thus replaces the number of dynamics  by the dimension $d _ { U }$ , an improvement whenever $| \mathcal { P } | > d _ { U }$

Intuitively, the condition $m \leq \eta d _ { \star } H ^ { 2 }$ states that when reconfiguring the dynamics is costly, the expert can demonstrate up to $\eta d _ { \star } H ^ { 2 }$ trajectories per round without loss. Beyond this threshold, the noise in the empirical features $\hat { \phi } _ { i }$ in (2) still shrinks, but the information each round contributes is set by its dynamics and does not grow with $m ,$ so our bound no longer improves.

Transfer beyond . We now extend our guarantee to target dynamics $P$ outside $\mathcal { P } .$ Such dynamics may require information along directions that no dynamics in $\mathcal { P }$ identifies, which we measure by $g _ { \perp } ( \dot { P ) } : = \left\| \Pi _ { U ^ { \perp } } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \Pi _ { U ^ { \perp } } \right\| _ { 2 } ,$ where $U ^ { \perp }$ is the orthogonal complement of the identifiable subspace $U .$

Theorem 3.7. Fix $\delta \in \mathsf { \Gamma } ( 0 , 1 )$ and follow the active rule (7) after round $n _ { \delta }$ . If m  ηd<sub>⋆</sub> $H ^ { 2 }$ and n $\geq 2 n _ { \delta }$ then with probability at least $1 - \delta$ , for any $P _ { \cdot }$

$$
\mathrm { S u b O p t } ( \theta _ { n } , P ) = \tilde { \mathcal { O } } \left( \frac { d _ { \star } d _ { U } } { n m } \right) + \mathcal { O } \left( g _ { \perp } ( P ) B _ { \theta } ^ { 2 } \right) .
$$

Proof sketch. Let $\theta _ { n } - \theta ^ { \mathrm { E } } = u + v$ with $u \in U$ and $v \in U ^ { \perp }$ . Theorem 3.6 bounds the error along $u ,$ up to how much information $\mathcal { P }$ provides along the directions that matter under $P .$ Since $\| v \| _ { 2 } ~ \le ~ 2 B _ { \theta }$ , we have $\| v \| _ { \mathcal H _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } \le 4 g _ { \perp } ( P ) B _ { \theta } ^ { 2 }$ . Pseudo self-concordance then yields the result, stated in full in Appendix D. □

Theorem 3.7 splits suboptimality under any $P$ into two terms. The first shrinks with the number of demon strations, and corresponds to directions shared by $U _ { P }$ and U. The second is a transfer bias, and corresponds to directions that no dynamics in $\mathcal { P }$ cover. It therefore cannot be reduced by observing the expert in $\mathcal { P } .$ Interestingly, our active rule helps reduce the first term, but only the choice of $\mathcal { P }$ can mitigate the second. When $U _ { P } \subseteq U$ , the bias term vanishes, and in particular for $P \in \mathcal { P }$ , we exactly recover (8).

## 4 EXPERIMENTS

Our experiments serve two purposes: $( i )$ to evaluate when active learning improves sample eficiency in terms of expert demonstrations, and $( i i )$ to identify which dynamics are most useful for eficiently learning transferable rewards.

We instantiate our two-phase strategy (Section 3.2) with the log-determinant criterion (Remark 3.4), which performed best among valid criteria we compare in Appendix I. Since $n _ { \delta }$ cannot be computed, we treat the warm-up length as a hyperparameter, set to one third of the budget, and study other choices in Appendix I. We compare against baselines that isolate the benefits of $( a )$ the warm-up phase, $( b )$ leveraging multiple dynamics, and $( c )$ learning transferable rewards rather than directly imitating behavior.

To evaluate (a), we run an Active only ablation that follows the active rule (7) from the start.<sup>2</sup> To evaluate $( b )$ , we compare against the environment design (ED) approach of Buening et al. [10], a round-robin baseline following (6), and methods that learn from one fixed dynamics in $\mathcal { P } \colon$ Fixed dynamics IRL and GAIL [21]. All IRL methods except GAIL optimize the min-max IRL objective (2). Finally, to evaluate $( c ) _ { ; }$ , we compare against imitation methods: Behavioral Cloning (BC), trained on a single dynamics like the methods above, and robust imitation via RIME [22].

All methods follow the same evaluation protocol. At each round $k ,$ the learner selects dynamics $P _ { k } \in \mathcal { P }$ and observes $m = 1$ expert trajectory, generated by rolling out, under $P _ { k }$ , the entropy-regularized optimal policy for the true reward parameters $\theta ^ { \mathrm { E } }$ . For fixeddynamics methods, the environment is uniformly sampled once from $\mathcal { P }$ and then fixed. Evaluation proceeds by learning a reward from collected data and deploying a policy optimized for each tested dynamics; imitation methods (BC, RIME) instead deploy a single policy across all dynamics. In all experiments, state–action features are one-hot encodings over states and actions.

We measure performance by the worst-case subopti mality (1) and by the reward error up to constants, min $c \in \mathbb { R } \left\| \dot { \theta } ^ { \mathbf { E } } - \theta _ { n } + c \mathbf { 1 } \right\| _ { 2 } .$ We report means and standard errors over 20 randomly seeded runs. These experiments focus on tractable environments that require no approximation, leaving scalability to future work.

## 4.1 Controlled sets of dynamics

We construct training sets $\mathcal { P }$ in which we control how well rewards learned in one dynamics transfer to the (c) Uniformly diverse dynamics (d) Arbitrarily diverse dynamics

![](images/1cae8f8836375682528654a56bee9aca633c5770c6276cfcff90220887e665a7.jpg)

![](images/fdcfa66de54f1c5c3213ed99ff5922067d01e5c1739894507671f7ffa2f98b28.jpg)

![](images/c62b5007e9463194d6a3d2806ff256d5758556ed73eeb741442aacf733b95b58.jpg)

![](images/94c22e603428c53551101e47369afa69af8e993301aea84d885edafdf6c99f96.jpg)  
Two-phase Active only ED Round-robin Fixed dynamics GAIL BC RIME

Figure 1: Worst-case suboptimality over the training set as a function of the number of observed expert trajectories on randomly generated MDPs. Each plot corresponds to a diferent type of training set.  
![](images/f206cdb4d4e4f233c312cdec81c4ee180237b7a0ec368dd1853bc3f4099f0672.jpg)

![](images/6a5fb03a393330af81d2ad5a84291a94d3c9a6787bf7a57f1af9ffef28cc403b.jpg)  
Figure 2: Number of times each dynamics in the training set  is selected after 100 rounds, for two types of training set. Dynamics are ordered by increasing average dissimilarity to the rest of ${ \mathcal { P } } ,$ so that $P _ { 1 0 }$ corresponds to the most isolated dynamics.

others. This is quantified by principal angles between unidentifiable subspaces [23, 6] (see also Appendix I.2).

We consider four types of sets: (1) Equivalent dynamics, where rewards transfer perfectly across ; (2) Similar dynamics, where rewards transfer consistently well; (3) Uniformly diverse dynamics, where transfer is uniformly weak across pairs; (4) Arbitrarily diverse dynamics, where transferability is heterogeneous across the set. For each random seed, we generate a set of $| \mathcal { P } | = 1 0$ dynamics. After $n = 1 0 0$ demonstrations, methods are also evaluated on unseen dynamics. The MDPs all have $| S | = 2 0$ states and $| { \mathcal { A } } | = 5$ actions.

Results. With equivalent dynamics (Figure 1(a)), all methods perform similarly. Since every dynamics reveals the same information about the reward, dynam ics selection ofers no advantage, and imitation methods remain competitive with reward learning. As dynamics become more diverse (Figures 1(b)–(d)), fixed dynamics methods (IRL, GAIL, and BC) increasingly stall. For reward learning, this is the bias of Theo rem 3.7 with a single training dynamics, which grows with the diversity of . Active selection is most benefi cial in arbitrarily diverse sets (Figure 1(d)). There, the Active only variant performs best, both on and on unseen dynamics, where it substantially outperforms ED (Table 1). This suggests that our warm-up length is conservative, a gap we discuss in Section 6.

Figure 2 shows which dynamics are selected. In uniformly diverse sets, selection closely matches roundrobin. In arbitrarily diverse sets, it concentrates on dynamics that either transfer well on average across $( P _ { 1 } , \ldots , P _ { 5 } )$ or are highly isolated from the rest $\left( { { P _ { 1 0 } } } \right)$

## 4.2 Random mazes

We next consider a maze navigation task [10], where the agent is rewarded for reaching and waiting in one of three goal states. This task makes the selected dynamics easy to visualize (Figure 3). We generate sets of $| \mathcal { P } | = 1 0 0$ mazes by randomly placing walls, ranging from nearly empty layouts to highly constrained or even unsolvable ones (e.g., blocked goals). All mazes in a set share the initial state and the reward, with goal rewards sampled uniformly in [0, 1]. Here, $| { \mathcal S } | = 4 9$ and $| { \mathcal { A } } | = 5$ (four moves and a waiting action).

Results. Active selection is clearly more sampleeficient (Figure 4). Without warm-up, it reaches with 20 trajectories the suboptimality that round-robin attains with 100. The two-phase strategy catches up after about 60 trajectories, again suggesting that the warm-up is conservative.

Figure 3 shows that the active rule often selects unsolvable mazes. When goals are all unreachable but the layout remains open, the entropy-regularized expert acts uniformly, revealing that the reward is constant across reachable non-goal states. A maze can thus be informative without being solvable.

## 5 RELATED WORK

Active IRL aims to minimize expert interactions, the main practical bottleneck, by querying the expert selectively. Prior work has mostly focused on fixed dynamics. Lopes et al. [24] and Cui and Niekum [25] select states or trajectories that maximize information about the reward, Bajgar et al. [26] provide PAC guarantees for such Bayesian state queries, and Lindner et al. [27] use confidence sets over feasible rewards to guide exploration. Our setting instead queries the ex pert under chosen dynamics.

<table><tr><td rowspan="2">Algorithm Active only</td><td colspan="2">Training set type</td></tr><tr><td>Uniform</td><td>Arbitrary</td></tr><tr><td>Two-phase ED Round-robin Fixed dynamics GAIL  $2 6 . 6 2 \pm 0 . 3 7$  BC  $1 5 . 0 5 \pm 0 . 1 8$ </td><td> ${ \bf 2 . 8 6 \pm 0 . 0 2 }$   $2 . 9 3 \pm 0 . 0 2$   $\bar { 3 } . \bar { 3 } 2 \bar { \pm } \bar { 0 . 0 2 }$   $3 . 1 5 \pm 0 . 0 2$   $1 8 . 1 8 \pm 0 . 1 6$ </td><td> ${ \bf 1 2 . 6 3 \pm 0 . 1 4 }$   $1 3 . 6 3 \pm 0 . 1 5$   $\bar { 1 6 } . 5 2 \bar { \pm } \bar { 0 } . \bar { 2 } 2 \bar { }$   $1 9 . 5 9 \pm 0 . 1 9$   $2 4 . 3 6 \pm 0 . 0 9$ </td></tr><tr><td>Suhity vmwMnMWaWAms Rewrd Eror 4 2</td><td colspan="2"> $4 . 9 1 \pm 0 . 0 3$   $2 4 . 4 7 \pm 0 . 3 4$  1.2 rnmnmin 1 0.8 0.6 0.4</td></tr></table>

Two-phase Active only ED Round-robin Fixed dynamics BC

![](images/2b6d8eeb4f2f932045cfa687b719d4e54228332a204b6be741f01e7a76422e77.jpg)  
Figure 3: Mazes most frequently selected by Active only after 100 rounds, on a representative seed. The blue triangle marks the initial state and stars mark goal states, labeled with their expert rewards. The heatmap shows the state occupancy of policies optimized for the inferred reward.  
Table 1: Worst-case suboptimality ( 100) over 1000 unseen dynamics after learning from n = 100 expert trajectories, collected on either a uniformly or an arbitrarily diverse training set of dynamics.

![](images/09935b116628eca44f89b708505581dc8bf15ef01b7e47a3488522b25adec21a.jpg)

![](images/33b0cbf654f28b320f8d3fb610401551cb004d4b6feea04c6be91cb8218de6b6.jpg)  
Figure 4: Worst-case suboptimality over the training set  of 100 random mazes, and reward error, as a function of the number of observed expert trajectories.

Learning from given dynamics. In the entropyregularized setting, suficiently diverse dynamics allow recovering rewards up to constants when the expert policy is known [4, 5]. For transfer from demonstrations, Schlaginhaufen and Kamgarpour [6] give sufficient conditions, and Hao et al. [28] derive finitesample guarantees from one source to a given target environment. These results require the dynamics to be well covered, through ergodicity [4, 5, 6] or through source data covering the target’s occupancy [28], whereas none of our results require either assumption. Our confidence sets extend the fast rates of Schlaginhaufen and Kamgarpour [16] from fixed dynamics to adaptively selected ones. Without regular ization, rewards are only partially identifiable, and recent work selects rewards that remain robust to this ambiguity [29, 30]. Beyond reward learning, Chae et al. [22] learn a single policy robust to dynamics variations by imitating experts on randomly sampled dynamics. None of these works lets the learner choose the dynamics in which to collect demonstrations.

Choosing dynamics. Amin et al. [31] show that an agent choosing the tasks can identify the expert’s reward when the expert policy is revealed in each, and bound the number of mistakes when tasks are chosen adversarially. Nitschke et al. [32] maximize behavior entropy, Bajgar et al. [33] maximize expected information gain, and Buening et al. [10] target where the current reward estimate is expected to perform worst. These methods focus on state-only rewards or deterministic experts, rely on expensive selection rules, or lack sample complexity guarantees for dynamics selection. In contrast, our work provides a practical and provably eficient framework for actively choosing dynamics to learn transferable rewards.

## 6 CONCLUSION

We introduced an active learning method for inferring transferable rewards, which adaptively selects the dy namics under which the expert demonstrates. After a round-robin warm-up, it queries the expert under the dynamics with the largest information gain. The resulting reward transfers across all available dynamics at a rate governed by the dimension of the identifiable reward subspace rather than their number, and to unseen dynamics up to a bias from the directions left uncovered. Empirically, it requires substantially fewer demonstrations than passive selection and outperforms the most closely related active approach [10].

Limitations and future work. First, our guarantees rely on a conservative warm-up, while active selection performs well without it in practice, possibly because it naturally first selects the dynamics informing directions not yet covered. Additional assumptions on the information gathered, ensuring that our local curvature argument applies early, or data-driven switching criteria could close this gap. Second, IRL across multiple dynamics requires one optimal policy per dy namics, which is costly for large , and dynamicsconditioned policies could amortize this cost. Finally, our study is restricted to finite sets of tabular environ ments and linear rewards. A promising direction is to model rewards with neural networks whose last layer is linear, using the last hidden layer as learned lowdimensional features ϕ and computing Hessians with respect to the last layer only, as in Ash et al. [34].

## References

[1] Andrew Y Ng, Stuart Russell, et al. Algorithms for inverse reinforcement learning. In Icml, volume 1, page 2, 2000.

[2] Dean A. Pomerleau. Alvinn: An autonomous land vehicle in a neural network. In D. Touretzky, editor, Advances in Neural Information Processing Systems, volume 1. Morgan-Kaufmann, 1988. URL https://proceedings. neurips.cc/paper\_files/paper/1988/file/ 812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper. pdf.

[3] Matthieu Geist, Bruno Scherrer, and Olivier Pietquin. A theory of regularized markov decision processes. In International conference on machine learning, pages 2160–2169. PMLR, 2019.

[4] Haoyang Cao, Samuel Cohen, and Lukasz Szpruch. Identifiability in inverse reinforcement learning. Advances in Neural Information Processing Systems, 34:12362–12373, 2021.

[5] Paul Rolland, Luca Viano, Norman Sch¨urhof, Boris Nikolov, and Volkan Cevher. Identifiability and generalizability from multiple experts in inverse reinforcement learning. Advances in Neural Information Processing Systems, 35:550–564, 2022.

[6] Andreas Schlaginhaufen and Maryam Kamgarpour. Towards the transferability of rewards recovered via regularized inverse reinforcement learning. Advances in Neural Information Processing Systems, 37:21461–21501, 2024.

[7] Xue Bin Peng, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. Sim-to-real transfer of robotic control with dynamics randomization. In 2018 IEEE international conference on robotics and automation (ICRA), pages 3803–3810. IEEE, 2018.

[8] Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, and Vladlen Koltun. Carla: An open urban driving simulator. In Conference on robot learning, pages 1–16. PMLR, 2017.

[9] Ajay Mandlekar, Yuke Zhu, Animesh Garg, Jonathan Booher, Max Spero, Albert Tung, Julian Gao, John Emmons, Anchit Gupta, Emre Orbay, et al. Roboturk: A crowdsourcing platform for robotic skill learning through imitation. In Conference on Robot Learning, pages 879–893. PMLR, 2018.

[10] Thomas Kleine Buening, Victor Villin, and Christos Dimitrakakis. Environment design for inverse reinforcement learning. In Proceedings of the 41st International Conference on Machine Learning, pages 24808–24828, 2024.

[11] Gokul Swamy, Sanjiban Choudhury, J Andrew Bagnell, and Steven Wu. Of moments and matching: A game-theoretic framework for closing the imitation gap. In International Conference on Machine Learning, pages 10022–10032. PMLR, 2021.

[12] Chi Jin, Zhuoran Yang, Zhaoran Wang, and Michael I Jordan. Provably eficient reinforcement learning with linear function approximation. In Jacob Abernethy and Shivani Agarwal, editors, Proceedings of Thirty Third Conference on Learning Theory, volume 125 of Proceedings of Machine Learning Research, pages 2137–2143. PMLR, 09– 12 Jul 2020. URL https://proceedings.mlr. press/v125/jin20a.html.

[13] Umar Syed and Robert E Schapire. A gametheoretic approach to apprenticeship learning. Advances in neural information processing systems, 20, 2007.

[14] Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In Icml, volume 99, pages 278–287. Citeseer, 1999.

[15] Francis Bach. Self-concordant analysis for logistic regression. Electronic Journal of Statistics, 4:384– 414, 2010.

[16] Andreas Schlaginhaufen and Maryam Kamgarpour. Fast rates for inverse reinforcement learning, 2026.

[17] Kamalika Chaudhuri, Sham M Kakade, Praneeth Netrapalli, and Sujay Sanghavi. Convergence rates of active learning for maximum likelihood estimation. Advances in Neural Information Processing Systems, 28, 2015.

[18] Joongkyu Lee and Min-hwan Oh. Improved online confidence bounds for multinomial logistic bandits. arXiv preprint arXiv:2502.10020, 2025.

[19] Jack Kiefer and Jacob Wolfowitz. Optimum designs in regression problems. The annals of mathematical statistics, 30(2):271–294, 1959.

[20] Jack Kiefer. General equivalence theory for optimum designs (approximate theory). The annals of Statistics, pages 849–879, 1974.

[21] Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning. Advances in neural information processing systems, 29, 2016.

[22] Jongseong Chae, Seungyul Han, Whiyoung Jung, Myungsik Cho, Sungho Choi, and Youngchul Sung. Robust imitation learning against variations in environment dynamics. In International Conference on Machine Learning, pages 2828– 2852. PMLR, 2022.

[23] Aur´el Gal´antai. Finite Projection Methods for Linear Systems, pages 83–115. Springer US, Boston, MA, 2004. doi: 10.1007/ 978-1-4419-9180-5 3. URL https://doi.org/ 10.1007/978-1-4419-9180-5\_3.

[24] M. Lopes, Francisco S. Melo, and L. Montesano. Active learning for reward estimation in inverse reinforcement learning. In ECML/PKDD, 2009. URL https://doi.org/ 10.1007/978-3-642-04174-7\_3.

[25] Yuchen Cui and Scott Niekum. Active reward learning from critiques. 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 6907–6914, 2018. URL https:// api.semanticscholar.org/CorpusID:4490670.

[26] Ondrej Bajgar, Dewi SW Gould, Jonathon Liu, Alessandro Abate, Konstantinos Gatsis, and Michael A Osborne. Pac apprenticeship learning with bayesian active inverse reinforcement learning. arXiv preprint arXiv:2508.03693, 2025.

[27] David Lindner, Andreas Krause, and Giorgia Ramponi. Active exploration for inverse reinforcement learning, 2023. URL https://arxiv.org/ abs/2207.08645.

[28] Guang-Yuan Hao, Lars van der Laan, Aur´elien Bibaut, and Nathan Kallus. Reward transfer from inverse reinforcement learning: A coupled minimax approach. arXiv preprint arXiv:2605.27834, 2026.

[29] Filippo Lazzati and Alberto Maria Metelli. Robustness in the face of partial identifiability in reward learning. In International Conference on Learning Representations, volume 2026, pages 107353–107390, 2026.

[30] Filippo Lazzati and Alberto Maria Metelli. Generalizing behavior via inverse reinforcement learning with closed-form reward centroids. arXiv preprint arXiv:2509.12010, 2025.

[31] Kareem Amin, Nan Jiang, and Satinder Singh. Repeated inverse reinforcement learning. Ad-

vances in neural information processing systems, 30, 2017.

[32] Paul Nitschke, Lars Lien Ankile, Eura Nofshin, Siddharth Swaroop, Finale Doshi-Velez, and Weiwei Pan. Amber: An entropy maximizing environment design algorithm for inverse reinforcement learning. In ICML 2024 Workshop on Models of Human Feedback for AI Alignment, 2024.

[33] Ondrej Bajgar, Sid William Gould, R. Mitta, Jonathon Liu, Oliver Newcombe, and Jackson Golden. Toward information theoretic active inverse reinforcement learning. ArXiv, abs/2501.00381, 2024. URL https://api. semanticscholar.org/CorpusId:275212510.

[34] Jordan Ash, Surbhi Goel, Akshay Krishnamurthy, and Sham Kakade. Gone fishing: Neural active learning with fisher embeddings. Advances in Neural Information Processing Systems, 34:8927– 8939, 2021.

[35] Titouan Renard, Andreas Schlaginhaufen, Tingting Ni, and Maryam Kamgarpour. Convergence of a model-free entropy-regularized inverse reinforcement learning algorithm. arXiv preprint arXiv:2403.16829, 2024.

## CHECKLIST

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. Yes. Section 2 for the setting, assumptions and models. Section 3.2 for algorithms.

(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. Yes. Sample complexity is analyzed in Lemma 3.3 and Theorem 3.6, and computational complexity in the ”Computational complexity” paragraph of Section 3.2 and Appendix H.

(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. Yes. Link to the anonymized source code provided in Appendix I.

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. Yes. See Section 2.

(b) Complete proofs of all theoretical results. Yes, all proofs are provided in Appendices B and C.

(c) Clear explanations of any assumptions. Yes.

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). Yes. We provide a link to the code used in experiments in Appendix I.

(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). Yes, hyperparameters are specified both in Appendix I, and through the opensourced code used for our experiments.

(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). Yes. We consistently provide the number of seeds used, and what error bars represent in both the main text (Sec tion 4) and appendix.

(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). Yes. Information is included in Appendix I.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator If your work uses existing assets. Not Applicable.

(b) The license information of the assets, if applicable. Not Applicable.

(c) New assets either in the supplemental material or as a URL, if applicable. Not applicable.

(d) Information about consent from data providers/curators. Not Applicable.

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. Not Applicable.

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to partici pants and screenshots. Not Applicable.

(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. Not Applicable.

(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. Not Applicable.

<table><tr><td colspan="3">Information and identifiability</td></tr><tr><td> $\mathcal { H } _ { P } ( \theta )$ </td><td>:=  $\nabla _ { \theta } ^ { 2 } V _ { P } ^ { \star } ( \theta )$ </td><td>, information provided by  $P$ </td></tr><tr><td> $A _ { P } ^ { \theta } ( \tau )$ </td><td>:=</td><td>feature advantage of  $\pi _ { P } ( \theta )$  along  $\tau$  (Lemma B.1)</td></tr><tr><td> $\Sigma _ { P } ^ { \mathrm { d y n } } ( \theta )$ </td><td>i=</td><td> $\mathrm { C o v } _ { P } ^ { \pi _ { P } ( \theta ) } \left( \phi ( \tau ) - A _ { P } ^ { \theta } ( \tau ) \right)$  , dynamics noise</td></tr><tr><td> $U _ { P }$ </td><td>:= :=</td><td>im  ${ \bar { \mathcal { H } } } _ { P } ( \theta )$  , identifiable subspace of  $P$  (independent of  $\theta )$ </td></tr><tr><td> $U$ </td><td></td><td> $\textstyle \sum _ { P \in { \mathcal { P } } } { \dot { U _ { P } } }$  , joint identifiable subspace, of dimension  $d _ { U }$ </td></tr><tr><td> $\Pi _ { P } , \Pi _ { U }$ </td><td>:=</td><td>orthogonal projectors onto  $U _ { P }$  and  $U$ </td></tr><tr><td> $G _ { P } ( \theta _ { 1 } , \theta _ { 2 } )$ </td><td>:=</td><td> $\begin{array} { r l } { \int _ { 0 } ^ { 1 } ( 1 - t ) \mathcal { H } _ { P } \left( t \theta _ { 2 } + ( 1 - t ) \theta _ { 1 } \right) } \end{array}$  dt</td></tr><tr><td colspan="3">Estimation and confidence</td></tr><tr><td> $\hat { \phi } _ { i }$ </td><td>=</td><td> $\begin{array} { r } { \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \phi ( \tau _ { i , j } ) } \end{array}$  , empirical features at round i</td></tr><tr><td> $L _ { k } ( \theta ) , \ \theta _ { k }$ </td><td>i=</td><td>empirical loss (2), and  $\theta _ { k } \in \arg \operatorname* { m i n } _ { \theta \in \Theta } L _ { k } ( \theta )$ </td></tr><tr><td> $\mathcal { H } _ { k }$ </td><td>:=</td><td> $\textstyle \sum _ { i = 1 } ^ { k } { \mathcal { H } } _ { P _ { i } } ( \theta ^ { \operatorname { E } } )$  , accumulated information</td></tr><tr><td> $\lambda _ { k }$ </td><td>i=</td><td> $\lambda _ { \operatorname* { m i n } } ^ { U } ( \mathcal { H } _ { k } )$  , information in the least-informed direction</td></tr><tr><td> $\Sigma _ { P }$ </td><td> $\begin{array} { r l } {  { \operatorname { C o v } _ { P } ^ { \pi _ { P } ^ { \mathrm { E } } } ( \Pi _ { P } \phi _ { . } ( \tau ) ) } } \end{array}$  三</td><td></td></tr><tr><td> $d _ { \star }$ </td><td>三</td><td> $\begin{array} { r } { d _ { U } \operatorname* { m a x } _ { P \in \mathcal { P } } \left\| \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) ^ { \dagger / 2 } \Sigma _ { P } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) ^ { \dagger / 2 } \right\| _ { \mathrm { q } } } \end{array}$ </td></tr><tr><td> $R$ </td><td>i=</td><td> $\eta H ( H + 1 ) \ddot { B _ { \phi } } ,$  pseudo self-concordance constant</td></tr><tr><td> $\beta _ { k } ^ { 2 } ( \delta )$ </td><td> $\frac { \dot { 1 } 6 d _ { \star } \operatorname { l o g } ( 3 \eta d _ { \star } k \dot { m } / \delta ) } { ( 1 - e ^ { - 1 } ) ^ { 2 } m }$  :=</td><td>, squared confidence radius (Lemma 3.1)</td></tr><tr><td> $n _ { \delta } , \ n _ { 0 }$ </td><td>:=</td><td>first round satisfying (4), and warm-up length used in practice</td></tr><tr><td colspan="3">Selection and transfer</td></tr><tr><td> $k ( P )$ </td><td>i=  $\textstyle \sum _ { i = 1 } ^ { k } 1 1 [ P _ { i } = P ] .$ </td><td>number of times 5  $P$  was selected</td></tr><tr><td> $g _ { k } ( P )$ </td><td>i=</td><td> $\left\| \mathcal { \hat { H } } _ { k } ^ { \dagger / 2 } \mathcal { \hat { H } } _ { P } ( \theta ^ { \mathrm { E } } ) \mathcal { \hat { H } } _ { k } ^ { \dagger / 2 } \right\| _ { 2 }$  , information gain of P</td></tr><tr><td> ${ \hat { g } } _ { k } ( P )$ </td><td>=</td><td> $\begin{array} { r } { \left| \left( \sum _ { i = 1 } ^ { k } \mathcal { H } _ { i } ( \boldsymbol { \theta } _ { k } ) \right) ^ { \dagger / 2 ^ { * } } \mathcal { H } _ { P } ( \boldsymbol { \theta } _ { k } ) \left( \sum _ { i = 1 } ^ { k } \mathcal { H } _ { i } ( \boldsymbol { \theta } _ { k } ) \right) ^ { \dagger / 2 } \right. _ { 2 } } \end{array}$ </td></tr><tr><td> $g _ { \bot } ( P )$ </td><td>:=</td><td> $\begin{array} { r l } {  { \big \| \Pi _ { U ^ { \perp } } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \Pi _ { U ^ { \perp } } \big \| _ { 2 } } } \end{array}$ </td></tr><tr><td> $\psi ( x )$ </td><td>i=</td><td> ${ \ddot { ( x e ^ { x } - e ^ { x } + 1 ) } } / { x ^ { 2 } }$  with  $\psi ( 0 ) : = 1 / 2$ </td></tr><tr><td colspan="3">Linear algebra</td></tr><tr><td> $\lambda _ { \operatorname* { m i n } } ^ { \mathcal { X } } ( X ) , \ \lambda _ { \operatorname* { m a x } } ^ { \mathcal { X } } ( X )$ </td><td>:=</td><td> $\begin{array} { r } { \operatorname* { m i n } _ { v \in \mathcal { X } , \| v \| _ { 2 } = 1 } \| v \| _ { X } ^ { 2 } , } \end{array}$  and  $\operatorname* { m a x } _ { v \in \mathcal { X } , \| v \| _ { 2 } = 1 } \left\| v \right\| _ { X } ^ { 2 }$ </td></tr><tr><td> $A + B$ </td><td>:=</td><td> $\{ a + b \mid \bar { a } \in A , b \in \mathbf { \bar { B } } \}$ </td></tr></table>

## Appendix

## A NOTATIONS AND PRELIMINARIES

## A.1 Notations

We enumerate below the most important notations used in our appendix.

<table><tr><td colspan="3">Problem setting</td></tr><tr><td> $\mathcal { P }$ </td><td>:=</td><td>finite set of dynamics available for expert demonstrations</td></tr><tr><td> $n , \ m$ </td><td>:=</td><td>number of rounds, and expert trajectories per round</td></tr><tr><td> $\theta ^ { \mathrm { E } } , \Theta$ </td><td>:=</td><td>expert reward parameters, and  $\begin{array} { r } { \Theta : = \big \{ \theta \in \bar { \mathbb { R } } ^ { d } : \| \theta \| _ { 2 } \leq B _ { \theta } \big \} } \end{array}$ </td></tr><tr><td> $\pi _ { P } ( \theta )$ </td><td>:二</td><td> $\arg \operatorname* { m a x } _ { \pi } V _ { P } ^ { \pi } ( \theta )$  , optimal regularized policy</td></tr><tr><td> $\nu _ { h , P } ^ { \pi } ( s )$ </td><td>:=</td><td> $\mathbb { P } _ { P } ^ { \pi } \left[ s _ { h } = s \right]$ </td></tr><tr><td> $\mu _ { h , P } ^ { \pi } ( s , a )$ </td><td>:=</td><td> $\nu _ { h , P } ^ { \pi } ( s ) \pi _ { h } ( a \mid s )$ </td></tr><tr><td> $\mathrm { E n t } _ { h } ( \pi )$ </td><td>:=</td><td> $- \sum _ { \ldots } { } _ { s , a } \mu _ { h , P } ^ { \pi } ( s , a ) \log \pi _ { h } ( a \mid s )$ </td></tr><tr><td> $\phi ( \tau )$ </td><td>:=</td><td> $\scriptstyle \sum _ { h = 1 } ^ { H } \phi ( s _ { h } , a _ { h } )$ </td></tr><tr><td> $\phi _ { P } ( \theta ) , \ \phi _ { P } ^ { E }$ </td><td>三</td><td> $\mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \phi ( \tau ) \right]$  and  $\phi _ { P } ^ { E } : = \phi _ { P } ( \theta ^ { \mathrm { E } } )$ </td></tr><tr><td> $\operatorname { S u b O p t } ( \theta , P )$ </td><td>:=</td><td> $\bar { V _ { P } ^ { \star } } ( { \theta ^ { \mathrm { E } } } ) - \bar { V _ { P } ^ { \pi _ { P } ( \theta ) } } ( { \theta ^ { \mathrm { E } } } )$ </td></tr></table>

## A.2 Preliminaries

MDPs. Throughout this appendix, we work under the same regularized MDP setup introduced in Section 2. For a policy $\pi = ( \pi _ { h } ) _ { h = 1 } ^ { H }$ , we define the state and state-action occupancies at each timestep $h \in \{ 1 , \ldots , H \}$ as

$$
\begin{array} { r } { \nu _ { h , P } ^ { \pi } ( s ) : = \mathbb { P } _ { P } ^ { \pi } \left[ s _ { h } = s \right] , \qquad \mu _ { h , P } ^ { \pi } ( s , a ) : = \nu _ { h , P } ^ { \pi } ( s ) \pi _ { h } ( a \mid s ) . } \end{array}
$$

The regularized value then reads

$$
V _ { P } ^ { \pi } ( r ) = \sum _ { h = 1 } ^ { H } \left( \langle r , \mu _ { h , P } ^ { \pi } \rangle + \frac { 1 } { \eta } \operatorname { E n t } _ { h } ( \pi ) \right) , \qquad \operatorname { E n t } _ { h } ( \pi ) : = - \sum _ { s , a } \mu _ { h , P } ^ { \pi } ( s , a ) \log \pi _ { h } ( a \mid s ) ,
$$

where $\operatorname { E n t } _ { h } ( \pi )$ is the expected Shannon entropy of $\pi { \mathrm { ~ a t } }$ timestep h under P. The optimal policy in regularized MDPs has a softmax characterization [3],

$$
\pi _ { h , P } ^ { \theta } ( a \mid s ) = \frac { \exp { \Big ( \eta Q _ { h , P } ^ { \theta } ( s , a ) \Big ) } } { \sum _ { a ^ { \prime } \in \mathcal { A } } \exp { \Big ( \eta Q _ { h , P } ^ { \theta } ( s , a ^ { \prime } ) \Big ) } } ,\tag{9}
$$

where $Q _ { h , P } ^ { \theta }$ and $V _ { h , P } ^ { \theta }$ are the optimal regularized Q-function and value, defined by backward recursion from $V _ { H + 1 , P } ^ { \theta } : = 0$

$$
Q _ { h , P } ^ { \theta } ( s , a ) : = \langle \theta , \phi ( s , a ) \rangle + \sum _ { s ^ { \prime } \in \mathcal { S } } T ( s ^ { \prime } \mid s , a ) V _ { h + 1 , P } ^ { \theta } ( s ^ { \prime } ) , \qquad V _ { h , P } ^ { \theta } ( s ) : = \frac { 1 } { \eta } \log \sum _ { a \in \mathcal { A } } \exp \left( \eta Q _ { h , P } ^ { \theta } ( s , a ) \right) .
$$

Eigenvalues. The smallest eigenvalue of X is denoted as $\lambda _ { \operatorname* { m i n } } ( X ) : = \operatorname* { m i n } _ { \| v \| _ { 2 } = 1 } \| v \| _ { X } ^ { 2 }$ and the smallest eigenvalue restricted to a subspace $\mathcal { X } \subset \mathbb { R } ^ { d }$ as $\begin{array} { r } { \lambda _ { \operatorname* { m i n } } ^ { \mathcal { X } } ( X ) : = \operatorname* { m i n } _ { \| v \| _ { 2 } = 1 , v \in \mathcal { X } } \| v \| _ { X } ^ { 2 } } \end{array}$

Features. For convenience, we use the $\begin{array} { r } { \phi ( \tau ) = \sum _ { h = 1 } ^ { H } \phi ( s _ { h } , a _ { h } ) } \end{array}$ to denote trajectory features, and $\phi _ { P } ( \theta ) =$ $\mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \phi ( \tau ) \right]$ to denote feature occupancies, with $\phi _ { P } ^ { \mathrm { E } } = \phi _ { P } ( \theta ^ { \mathrm { E } } )$

Fr´echet derivatives. For a function $f : \mathbb { R } ^ { d }  \mathbb { I }$ R that is three times diferentiable at $x \in \mathbb { R } ^ { d }$ , we denote the first three directional derivatives along $u , v , w \in \mathbb { R } ^ { d }$ by

$$
D f ( x ) [ u ] : = \langle \nabla f ( x ) , u \rangle , \quad D ^ { 2 } f ( x ) [ u , v ] : = \langle u , \nabla ^ { 2 } f ( x ) v \rangle ,
$$

$$
D ^ { 3 } f ( x ) [ u , v , w ] : = D \left( D ^ { 2 } f ( x ) [ u , v ] \right) [ w ] .
$$

where $D ^ { n } f ( x )$ denotes the n-th Fr´echet derivative of $f .$

## B PROOFS OF SECTION 3.1

## B.1 Reward estimation preliminaries

In this paper, we consider minimizing the following aggregated empirical IRL loss to estimate rewards:

$$
L _ { k } ( \theta ) : = \sum _ { i = 1 } ^ { k } l _ { i } ( \theta ) , \qquad l _ { i } ( \theta ) = V _ { P _ { i } } ^ { \star } ( \Pi _ { P _ { i } } \theta ) - \langle \Pi _ { P _ { i } } \theta , \hat { \phi } _ { i } \rangle ,
$$

where $\begin{array} { r } { \hat { \phi } _ { i } = \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \phi ( \tau _ { i , j } ) } \end{array}$ , and $\tau _ { i , j }$ is the j-th trajectory generated by the expert at round i. We denote a minimizer over Θ by $\theta _ { k } \in$ arg min $\boldsymbol { \mathrm { 1 } } _ { \theta \in \Theta } L _ { k } ( \theta )$ . We furthermore define the population loss

$$
L _ { k } ^ { \mathrm { p o p } } ( \theta ) = \sum _ { i = 1 } ^ { k } \left( V _ { P _ { i } } ^ { \star } ( \Pi _ { P _ { i } } \theta ) - \langle \Pi _ { P _ { i } } \theta , \phi _ { P _ { i } } ^ { \mathrm { E } } \rangle \right) .\tag{10}
$$

Here $\phi _ { P } ^ { \mathrm { E } } = \mathbb { E } _ { P } ^ { \pi _ { P } ^ { \mathrm { E } } } [ \phi ( \tau ) ]$ . Under adaptive dynamics selection, the population loss $L _ { k } ^ { \mathrm { p o p } }$ is a sum of per-round conditional population losses. It is minimized at the expert parameters $\theta ^ { \mathrm { E } }$

Defining $\begin{array} { r } { S _ { k } : = \sum _ { i = 1 } ^ { k } \Pi _ { P _ { i } } ( \hat { \phi } _ { i } - \phi _ { P _ { i } } ^ { \mathrm { E } } ) } \end{array}$ , the aggregated empirical loss decomposes as

$$
L _ { k } ( \theta ) = L _ { k } ^ { \mathrm { p o p } } ( \theta ) ~ - ~ \langle \theta , S _ { k } \rangle .
$$

Additionally, we introduce the following key quantities of our analysis:

$$
\begin{array} { r l r l } & { \mathcal { H } _ { P } ( \theta ) : = \nabla _ { \theta } ^ { 2 } V _ { P } ^ { \star } ( \theta ) , } & { \qquad } & { \mathcal { H } _ { k } : = \nabla ^ { 2 } L _ { k } ( \theta ^ { E } ) = \nabla ^ { 2 } L _ { k } ^ { \mathrm { p o p } } ( \theta ^ { E } ) = \displaystyle \sum _ { i = 1 } ^ { k } \mathcal { H } _ { P _ { i } } ( \theta ^ { E } ) , } \\ & { \quad \lambda _ { k } : = \lambda _ { \operatorname* { m i n } } ^ { U } ( \mathcal { H } _ { k } ) , } & { \qquad } & { R : = \eta H ( H + 1 ) B _ { \phi } , \qquad R _ { k } : = \frac { R } { \sqrt { \lambda _ { k } } } , } \\ & { \rho _ { k } : = \| \theta _ { k } - \theta ^ { E } \| _ { \mathcal { H } _ { k } } , } & { \qquad } & { \omega _ { k } : = \| S _ { k } \| _ { \mathcal { H } _ { k } ^ { \bot } } . } \end{array}
$$

Here, $\mathcal { H } _ { k } ^ { \dagger }$ denotes the Moore-Penrose inverse of $\mathcal { H } _ { k }$ . The quantity $R _ { k }$ is used only when $\lambda _ { k } > 0$ . The stochastic self-concordance constant R is justified in Lemma B.6.

Recall that $\Pi _ { P }$ is the orthogonal projector onto im $( \mathcal { H } _ { P } ( 0 ) )$ , which equals im $\left( \mathcal { H } _ { P } ( \theta ) \right)$ for every θ by Proposition B.5. The parameter-independent kernel implies

$$
\Pi _ { P } \mathcal { H } _ { P } ( \Pi _ { P } \theta ) \Pi _ { P } = \mathcal { H } _ { P } ( \theta ) .
$$

Thus the projected loss has the same Hessian as the original loss, justifying the expression for $\mathcal { H } _ { k }$ above.

Moreover, projection ensures

$$
S _ { k } \in \sum _ { i = 1 } ^ { k } \operatorname { i m } \mathopen { } \mathclose \bgroup \left( \mathcal { H } _ { P _ { i } } \mathopen { } \mathclose \bgroup \left( \theta ^ { \mathrm { E } } \aftergroup \egroup \right) \aftergroup \egroup \right) = \operatorname { i m } \mathopen { } \mathclose \bgroup \left( \mathcal { H } _ { k } \aftergroup \egroup \right) .
$$

This support property allows the noise to be controlled in the inverse Hessian norm without requiring full identifiability.

## B.2 Supporting lemmas

We present several technical lemmas laying out various properties of the optimal regularized value function and population loss 10. To make the dependence of the optimal value function on reward parameters explicit, we sometimes use the notation $V _ { \theta , P } ^ { \star } = V _ { P } ^ { \star } ( \theta )$

Lemma B.1 (Value derivatives [16]). Let Assumption 2.1 hold. Then, for any dynamics P and any $\theta , u , v \in \mathbb { R } ^ { d }$ ,

$$
\begin{array} { r l } & { \qquad D V _ { P } ^ { \star } ( \theta ) [ v ] = \langle v , \phi _ { P } ( \theta ) \rangle , } \\ & { \qquad D ^ { 2 } V _ { P } ^ { \star } ( \theta ) [ v , v ] = \| v \| _ { { \mathcal H } _ { P } ( \theta ) } ^ { 2 } = \eta \mathbb E _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \langle v , A _ { P } ^ { \theta } ( \tau ) \rangle ^ { 2 } \right] , } \\ & { \qquad D ^ { 3 } V _ { P } ^ { \star } ( \theta ) [ v , v , u ] = \eta ^ { 2 } \mathbb E _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \langle v , A _ { P } ^ { \theta } ( \tau ) \rangle ^ { 2 } \langle u , A _ { P } ^ { \theta } ( \tau ) \rangle \right] , } \end{array}
$$

where $\phi _ { P } ( \theta ) : = \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } [ \phi ( \tau ) ]$ is the expected feature vector of the optimal policy, and $\begin{array} { r } { A _ { P } ^ { \theta } ( \tau ) : = \sum _ { h = 1 } ^ { H } A _ { h , P } ^ { \theta } ( s _ { h } , a _ { h } ) } \end{array}$ is the unregularized feature advantage, with

$$
A _ { h , P } ^ { \theta } ( s , a ) : = \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \sum _ { t = h } ^ { H } \phi ( s _ { t } , a _ { t } ) ~ { \bigg | } ~ s _ { h } = s , ~ a _ { h } = a \right] - \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \sum _ { t = h } ^ { H } \phi ( s _ { t } , a _ { t } ) ~ { \bigg | } ~ s _ { h } = s \right] .
$$

In particular, $\begin{array} { r } { \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ A _ { P } ^ { \theta } ( \tau ) \right] = 0 } \end{array}$ , since $\begin{array} { r } { \mathbb { E } _ { a \sim \pi _ { h , P } ( \theta ) ( \cdot | s ) } \left[ A _ { h , P } ^ { \theta } ( s , a ) \right] = 0 } \end{array}$ for every h and s.

Proof. The three identities follow from Schlaginhaufen and Kamgarpour [16, Lemma A.13] under the linear reward structure of Assumption 2.1. □

Corollary B.2 (Deterministic transitions). Let Assumption 2.1 hold, and let P have deterministic transitions and a deterministic initial state. Then, for any $\theta , u , v \in \mathbb { R } ^ { d }$

$$
D V _ { P } ^ { \star } ( \theta ) [ v ] = \langle v , \phi _ { P } ( \theta ) \rangle ,
$$

$$
D ^ { 2 } V _ { P } ^ { \star } ( \theta ) [ v , v ] = \left\| v \right\| _ { \mathcal { H } _ { P } ( \theta ) } ^ { 2 } = \eta \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \langle v , \phi ( \tau ) - \phi _ { P } ( \theta ) \rangle ^ { 2 } \right] ,
$$

$$
D ^ { 3 } V _ { P } ^ { \star } ( \theta ) [ v , v , u ] = \eta ^ { 2 } \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \langle v , \phi ( \tau ) - \phi _ { P } ( \theta ) \rangle ^ { 2 } \langle u , \phi ( \tau ) - \phi _ { P } ( \theta ) \rangle \right] .
$$

Proof. By Lemma B.1, it sufices to show that $A _ { P } ^ { \theta } ( \tau ) = \phi ( \tau ) - \phi _ { P } ( \theta )$ . For $h \in \{ 1 , \ldots , H \}$ , let

$$
\Phi _ { h } \mathopen { } \mathclose \bgroup \left( s \aftergroup \egroup \right) : = \mathbb { E } _ { P } ^ { \pi _ { P } \mathopen { } \mathclose \bgroup \left( \theta \aftergroup \egroup \right) } \left[ \sum _ { t = h } ^ { H } \phi \mathopen { } \mathclose \bgroup \left( s _ { t } , a _ { t } \aftergroup \egroup \right) ~ \bigg | ~ s _ { h } = s \right] , \qquad \Phi _ { H + 1 } : = 0 .
$$

Since transitions are deterministic, $\left( s _ { h } , a _ { h } \right)$ determines $s _ { h + 1 }$ , so by the Markov property

$$
\mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \sum _ { t = h } ^ { H } \phi ( s _ { t } , a _ { t } ) ~ { \Big | } ~ s _ { h } , a _ { h } \right] = \phi ( s _ { h } , a _ { h } ) + \Phi _ { h + 1 } ( s _ { h + 1 } ) ,
$$

and hence $A _ { h , P } ^ { \theta } ( s _ { h } , a _ { h } ) = \phi ( s _ { h } , a _ { h } ) + \Phi _ { h + 1 } ( s _ { h + 1 } ) - \Phi _ { h } ( s _ { h } )$ . Summing over h, the terms in Φ telescope:

$$
A _ { P } ^ { \theta } ( \tau ) = \sum _ { h = 1 } ^ { H } \phi ( s _ { h } , a _ { h } ) - \Phi _ { 1 } ( s _ { 1 } ) = \phi ( \tau ) - \Phi _ { 1 } ( s _ { 1 } ) .
$$

Since the initial state is deterministic, $\Phi _ { 1 } ( s _ { 1 } ) = \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } [ \phi ( \tau ) ] = \phi _ { P } ( \theta )$ , which concludes the proof.

Lemma B.3 (Covariance decomposition). Let Assumption 2.1 hold. For any dynamics P and $\theta \in \mathbb { R } ^ { d }$

$$
\begin{array} { r } { \mathrm { C o v } _ { P } ^ { \pi _ { P } ( \theta ) } \left( \phi ( \tau ) \right) = \eta ^ { - 1 } \mathcal { H } _ { P } ( \theta ) + \Sigma _ { P } ^ { \mathrm { d y n } } ( \theta ) , } \end{array}\tag{11}
$$

where,

$$
\Sigma _ { P } ^ { \mathrm { d y n } } ( \theta ) : = \mathrm { C o v } _ { P } ^ { \pi _ { P } ( \theta ) } \left( \phi ( \tau ) - A _ { P } ^ { \theta } ( \tau ) \right) \succeq 0 .
$$

The matrix $\Sigma _ { P } ^ { \mathrm { d y n } } ( \theta )$ captures the randomness of the transitions and of the initial state.

Proof. Since $\mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ A _ { P } ^ { \theta } ( \tau ) \right] = 0$ , Lemma B.1 gives Co ${ } _ { P } ^ { \pi _ { P } ( \theta ) } \left( A _ { P } ^ { \theta } ( \tau ) \right) = \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ A _ { P } ^ { \theta } ( \tau ) A _ { P } ^ { \theta } ( \tau ) ^ { \top } \right] = \eta ^ { - 1 } \mathcal { H } _ { P } ( \theta )$ Moreover, in the decomposition

$$
\phi ( \tau ) - \phi _ { P } ( \theta ) = A _ { P } ^ { \theta } ( \tau ) + \left( \phi ( \tau ) - A _ { P } ^ { \theta } ( \tau ) - \phi _ { P } ( \theta ) \right) ,
$$

the two terms are uncorrelated by Schlaginhaufen and Kamgarpour [16, Lemma $\mathrm { A } . 7 ]$ . Taking covariances on both sides proves (11). □

Proposition B.4 (Hessian upper bound). Let Assumption 2.1 hold. For any dynamics P and $\theta ~ \in ~ \mathbb { R } ^ { d }$ $\| \mathcal { H } _ { P } ( \theta ) \| _ { 2 } \leq \eta H ^ { 2 } B _ { \phi } ^ { 2 }$

Proof. Let $v \in \mathbb { R } ^ { d }$ with $\| v \| _ { 2 } = 1$ . By Lemma B.1 and the covariance decomposition of Lemma B.3,

$$
\begin{array} { r } { \| v \| _ { \mathcal { H } ( \theta ) } ^ { 2 } = \eta \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \langle v , A _ { P } ^ { \theta } ( \tau ) \rangle ^ { 2 } \right] \leq \eta \operatorname { V a r } _ { P } ^ { \pi _ { P } ( \theta ) } \left( \langle v , \phi ( \tau ) \rangle \right) \leq \eta \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \langle v , \phi ( \tau ) \rangle ^ { 2 } \right] \leq \eta H ^ { 2 } B _ { \phi } ^ { 2 } , } \end{array}
$$

where the last step uses $\begin{array} { r } { \| \phi ( \tau ) \| _ { 2 } \leq \sum _ { h = 1 } ^ { H } \| \phi ( s _ { h } , a _ { h } ) \| _ { 2 } \leq H B _ { \phi } } \end{array}$ . Since $\mathcal { H } _ { P } ( \theta )$ is positive semidefinite, $\| \mathcal { H } _ { P } ( \theta ) \| _ { 2 } =$ ma $\mathfrak { c } _ { \| v \| _ { 2 } = 1 } \left\| v \right\| _ { \mathcal { H } _ { P } ( \theta ) } ^ { 2 }$ , which proves the claim. □

Proposition B.5 (Hessian image [16]). Let Assumption 2.1 hold and let P be any dynamics. Then ker $\left( \mathcal { H } _ { P } ( \theta ) \right)$ consists of the directions $v \in \mathbb { R } ^ { d }$ whose reward perturbation $r _ { v } ( s , a ) : = \langle v , \phi ( s , a ) \rangle$ preserves the optimal trajectory law under P, and is independent of θ. In particular, $U _ { P } = \mathrm { i m } \left( \mathcal { H } _ { P } ( \theta ) \right) = \mathrm { i m } \left( \mathcal { H } _ { P } ( 0 ) \right)$ for all $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$

Proof. By Lemma B.1, for any $v \in \mathbb { R } ^ { d }$

$$
\begin{array} { r l } & { v \in \ker \left( { \mathcal { H } } _ { P } ( \theta ) \right) \iff \operatorname { \mathbb { E } } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \langle v , A _ { P } ^ { \theta } ( \tau ) \rangle ^ { 2 } \right] = 0 } \\ & { \iff \langle v , A _ { P } ^ { \theta } ( \tau ) \rangle = 0 \quad \operatorname { \mathbb { P } } _ { P } ^ { \pi _ { P } ( \theta ) } \mathrm { - a l m o s t ~ s u r e l y } . } \end{array}
$$

By linearity, $\langle v , A _ { P } ^ { \theta } ( \tau ) \rangle$ is the cumulative unregularized advantage of $r _ { v }$ under $\pi _ { P } ( \theta )$ . By Schlaginhaufen and Kamgarpour [16, Corollary A.16], it vanishes almost surely if and only if $r _ { v }$ preserves the optimal trajectory law under $P ,$ a condition that does not depend on θ since soft-optimal policies have full support. Hence ker $( { \mathcal { H } } _ { P } ( \theta ) ) =$ ker $( \mathcal { H } _ { P } ( 0 ) )$ for all θ. Since $\mathcal { H } _ { P } ( \theta )$ is symmetric, im $\left( \mathcal { H } _ { P } ( \theta ) \right) = \ker \left( \mathcal { H } _ { P } ( \theta ) \right) ^ { \perp }$ , which concludes the proof. □

Lemma B.6 (Pseudo self-concordance). Let Assumption 2.1 hold, and define $R : = \eta H ( H + 1 ) B _ { \phi }$ . For any dynamics P and $\theta , u , v \in \mathbb { R } ^ { d }$ , the optimal regularized value function satisfies

$$
\begin{array} { r } { \big | D ^ { 3 } V _ { P } ^ { \star } ( \theta ) [ v , v , u ] \big | \leq R \ \| u \| _ { \Pi _ { P } } \ D ^ { 2 } V _ { P } ^ { \star } ( \theta ) [ v , v ] . } \end{array}
$$

Proof. By Lemma B.1 and Proposition B.5, $\langle w , A _ { P } ^ { \theta } ( \tau ) \rangle = 0$ almost surely for each w in a basis of ker $\mathscr { H } _ { P } ( \theta ) = U _ { P } ^ { \perp }$ Hence $A _ { P } ^ { \theta } ( \tau ) \in U _ { P }$ almost surely, and

$$
\langle u , A _ { P } ^ { \theta } ( \tau ) \rangle = \langle \Pi _ { P } u , A _ { P } ^ { \theta } ( \tau ) \rangle \qquad \mathrm { a l m o s t ~ s u r e l y } .
$$

At each step $h ,$ the feature advantage $A _ { h , P } ^ { \theta }$ is the diference of two conditional expectations of the remaining feature sum, each of norm at most $( H - h ^ { ' } + 1 ) B _ { \phi }$ . Therefore,

$$
\left\| A _ { P } ^ { \theta } ( \tau ) \right\| _ { 2 } \leq 2 B _ { \phi } \sum _ { h = 1 } ^ { H } ( H - h + 1 ) = H ( H + 1 ) B _ { \phi } \quad \mathrm { a l m o s t ~ s u r e l y } .
$$

Using Lemma B.1 and the Cauchy–Schwarz inequality,

$$
\begin{array} { r l } & { \left| D ^ { 3 } V _ { P } ^ { \star } ( \theta ) [ v , v , u ] \right| = \eta ^ { 2 } \left| \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \langle v , A _ { P } ^ { \theta } ( \tau ) \rangle ^ { 2 } \langle \Pi _ { P } u , A _ { P } ^ { \theta } ( \tau ) \rangle \right] \right| } \\ & { \qquad \leq \eta ^ { 2 } H ( H + 1 ) B _ { \phi } \left\| \Pi _ { P } u \right\| _ { 2 } \mathbb { E } _ { P } ^ { \pi _ { P } ( \theta ) } \left[ \langle v , A _ { P } ^ { \theta } ( \tau ) \rangle ^ { 2 } \right] } \\ & { \qquad = R \left\| u \right\| _ { \Pi _ { P } } D ^ { 2 } V _ { P } ^ { \star } ( \theta ) [ v , v ] . } \end{array}
$$

□

Lemma B.7 (Curvature stability). Let Assumption 2.1 hold and let P be any dynamics. Fix $\theta _ { 1 } , \theta _ { 2 } \in \mathbb { R } ^ { d }$ and set $a : = R \| \theta _ { 2 } - \theta _ { 1 } \| _ { \Pi _ { P } }$ . Define

$$
\psi ( x ) : = \int _ { 0 } ^ { 1 } t e ^ { x t } \mathrm { d } t = \left\{ \begin{array} { l l } { \frac { x e ^ { x } - e ^ { x } + 1 } { x ^ { 2 } } , } & { x \neq 0 , } \\ { 1 / 2 , } & { x = 0 , } \end{array} \right. \quad G _ { P } ( \theta _ { 1 } , \theta _ { 2 } ) : = \int _ { 0 } ^ { 1 } ( 1 - t ) \mathcal { H } _ { P } \left( t \theta _ { 2 } + ( 1 - t ) \theta _ { 1 } \right) \mathrm { d } t .
$$

Then:

$$
I . \ e ^ { - a } \ { \mathcal { H } } _ { P } ( \theta _ { 2 } ) \preceq { \mathcal { H } } _ { P } ( \theta _ { 1 } ) \preceq e ^ { a } { \mathcal { H } } _ { P } ( \theta _ { 2 } ) ;
$$

$$
{ \mathrm { ~ 2 . ~ } } \psi ( - a ) \mathcal { H } _ { P } ( \theta _ { 2 } ) \preceq G _ { P } ( \theta _ { 1 } , \theta _ { 2 } ) \preceq \psi ( a ) \mathcal { H } _ { P } ( \theta _ { 2 } ) ;
$$

$$
\begin{array} { r } { \mathcal { S } . \ e ^ { - a } \psi ( a ) \mathcal { H } _ { P } ( \theta _ { 1 } ) \preceq G _ { P } ( \theta _ { 1 } , \theta _ { 2 } ) \preceq e ^ { a } \psi ( - a ) \mathcal { H } _ { P } ( \theta _ { 1 } ) . } \end{array}
$$

Proof. Let $\bar { \theta } ( t ) : = t \theta _ { 2 } + ( 1 - t ) \theta _ { 1 }$ for $t \in [ 0 , 1 ]$ , and recall that $D ^ { 2 } V _ { P } ^ { \star } ( \theta ) [ v , v ] = v ^ { \top } \mathcal { H } _ { P } ( \theta ) v$

Reduction to $v \in U _ { P } \setminus \{ 0 \}$ . Let $v \in \mathbb { R } ^ { d }$ . By Proposition B.5, $( I - \Pi _ { P } ) v \in \ker { ( \mathcal { H } _ { P } ( \theta ) ) }$ for every $\theta \in \mathbb { R } ^ { d }$ , so by symmetry of $\mathcal { H } _ { P } ( \theta )$

$$
\begin{array} { r } { { \boldsymbol { v } } ^ { \top } { \mathcal { H } } _ { P } ( { \boldsymbol { \theta } } ) { \boldsymbol { v } } = ( \Pi _ { P } { \boldsymbol { v } } ) ^ { \top } { \mathcal { H } } _ { P } ( { \boldsymbol { \theta } } ) ( \Pi _ { P } { \boldsymbol { v } } ) \qquad \mathrm { f o r ~ a l l ~ } { \boldsymbol { \theta } } \in \mathbb { R } ^ { d } , } \end{array}
$$

and integrating this identity gives the same for $G _ { P } ( \theta _ { 1 } , \theta _ { 2 } )$ . If $\Pi _ { P } v \ : = \ : 0 \ :$ , every quadratic form in the claim vanishes. It therefore sufices to prove the scalar inequalities for $v \in U _ { P } \setminus \{ 0 \}$ , which we fix from now on.

Pointwise bounds. By Proposition B.5, $q ( t ) \ : = \ D ^ { 2 } V _ { P } ^ { \star } ( \bar { \theta } ( t ) ) [ v , v ] \ > \ 0$ for all $t ~ \in ~ [ 0 , 1 ]$ . Since $q ^ { \prime } ( t ) \ =$ $D ^ { 3 } V _ { P } ^ { \star } ( \bar { \theta } ( t ) ) [ v , v , \theta _ { 2 } - \theta _ { 1 } ]$ , Lemma B.6 gives

$$
\left| { \frac { \mathrm { d } } { \mathrm { d } t } } \log q ( t ) \right| \leq a \qquad { \mathrm { f o r ~ a l l ~ } } t \in [ 0 , 1 ] .
$$

Hence $| \log q ( t ) - \log q ( s ) | \leq | t - s | ~ a$ for all $s , t \in [ 0 , 1 ]$ . Taking $s = 1$ and $s = 0$ and exponentiating, we get, for all $t \in [ 0 , 1 ]$

$$
e ^ { - ( 1 - t ) a } q ( 1 ) \leq q ( t ) \leq e ^ { ( 1 - t ) a } q ( 1 ) ,\tag{12}
$$

$$
e ^ { - t a } q ( 0 ) \leq q ( t ) \leq e ^ { t a } q ( 0 ) .\tag{13}
$$

Item 1. This is (12) at $t = 0$

Items $\mathcal { Q }$ and 3. Note that $\begin{array} { r } { v ^ { \top } G _ { P } ( \theta _ { 1 } , \theta _ { 2 } ) v = \int _ { 0 } ^ { 1 } ( 1 - t ) q ( t ) \mathrm { d } t } \end{array}$ . Multiplying (12) and (13) by $( 1 - t ) \geq 0$ and integrating over [0, 1] preserves the inequalities. With the substitution $s = 1 - t$ and the definition of $\psi _ { : }$ , the resulting weights are

$$
\int _ { 0 } ^ { 1 } ( 1 - t ) e ^ { \pm ( 1 - t ) a } \mathrm { d } t = \int _ { 0 } ^ { 1 } s e ^ { \pm a s } \mathrm { d } s = \psi ( \pm a ) , \qquad \int _ { 0 } ^ { 1 } ( 1 - t ) e ^ { \pm t a } \mathrm { d } t = e ^ { \pm a } \int _ { 0 } ^ { 1 } s e ^ { \mp a s } \mathrm { d } s = e ^ { \pm a } \psi ( \mp a ) .
$$

This yields items 2 and 3, respectively.

Lemma B.8. Let Assumption 2.1 hold, and define $\textstyle \chi ( x ) : = { \frac { e ^ { x } - 1 } { x } } f o r x \neq 0$ , with $\chi ( 0 ) : = 1$ . Then, for any $k \geq 1$ with $\lambda _ { k } > 0$

$$
\begin{array} { r } { \chi \left( - R _ { k } \rho _ { k } \right) \rho _ { k } ^ { 2 } \le \langle \theta _ { k } - \theta ^ { E } , \nabla L _ { k } ^ { p o p } ( \theta _ { k } ) - \nabla L _ { k } ^ { p o p } ( \theta ^ { E } ) \rangle \le \chi \left( R _ { k } \rho _ { k } \right) \rho _ { k } ^ { 2 } . } \end{array}
$$

Proof. Let $u : = \theta _ { k } - \theta ^ { \mathrm { E } }$ and $\bar { \theta } ( t ) : = \theta ^ { \mathrm { E } } + t u$ for $t \in [ 0 , 1 ]$ . Since $\mathcal { H } _ { k }$ vanishes on $U ^ { \perp }$ and $\left\| \boldsymbol { w } \right\| _ { \mathcal { H } _ { k } } ^ { 2 } \ge \lambda _ { k } \left\| \boldsymbol { w } \right\| _ { 2 } ^ { 2 }$ for $w \in U$ , we have $\mathcal { H } _ { k } \succeq \lambda _ { k } \Pi _ { U } . \mathrm { ~ A s ~ } U _ { P _ { i } } \subseteq U$ , this gives for every $i \leq k$

$$
a _ { i } : = R \left\| u \right\| _ { \Pi _ { P _ { i } } } \leq R \left\| \Pi _ { U } u \right\| _ { 2 } \leq R \frac { \| u \| _ { \mathcal { H } _ { k } } } { \sqrt { \lambda _ { k } } } = R _ { k } \rho _ { k } .
$$

Applying (13) from the proof of Lemma B.7 to each $P _ { i }$ with $\theta _ { 1 } = \theta ^ { \mathrm { E } }$ and $\theta _ { 2 } = \theta _ { k }$ , and summing over $i ,$ we get for any $v \in \mathbb { R } ^ { d }$ and $t \in [ 0 , 1 ]$

$$
\begin{array} { r } { e ^ { - t R _ { k } \rho _ { k } } \left\| v \right\| _ { \mathcal { H } _ { k } } ^ { 2 } \le D ^ { 2 } L _ { k } ^ { \mathrm { p o p } } ( \bar { \theta } ( t ) ) [ v , v ] \le e ^ { t R _ { k } \rho _ { k } } \left\| v \right\| _ { \mathcal { H } _ { k } } ^ { 2 } , } \end{array}\tag{14}
$$

where we used $a _ { i } \le R _ { k } \rho _ { k }$ and $\begin{array} { r } { D ^ { 2 } L _ { k } ^ { \mathrm { p o p } } ( \theta ) = \sum _ { i = 1 } ^ { k } \mathcal { H } _ { P _ { i } } ( \theta ) } \end{array}$

By the fundamental theorem of calculus,

$$
\langle u , \nabla L _ { k } ^ { \mathrm { p o p } } ( \theta _ { k } ) - \nabla L _ { k } ^ { \mathrm { p o p } } ( \theta ^ { \mathrm { E } } ) \rangle = \int _ { 0 } ^ { 1 } D ^ { 2 } L _ { k } ^ { \mathrm { p o p } } ( \bar { \theta } ( t ) ) [ u , u ] \mathrm { d } t .
$$

Sandwiching the integrand using (14) with $v = u ,$ , and using $\begin{array} { r } { \int _ { 0 } ^ { 1 } e ^ { \pm x t } \mathrm { d } t = \chi ( \pm x ) } \end{array}$ and $\| u \| _ { \mathcal { H } _ { k } } = \rho _ { k }$ , gives the result. □

## B.3 Ellipsoidal confidence sets

Remark B.9. For any $u \in \mathbb { R } ^ { d }$

$$
\langle S _ { k } , u \rangle = \langle ( \mathcal { H } _ { k } ^ { \dagger } ) ^ { 1 / 2 } S _ { k } , \mathcal { H } _ { k } ^ { 1 / 2 } u \rangle .\tag{15}
$$

Indeed, let $v \in \ker ( \mathcal { H } _ { k } )$ . Since $\begin{array} { r } { \mathcal { H } _ { k } = \sum _ { i = 1 } ^ { k } \mathcal { H } _ { P _ { i } } ( \theta ^ { \mathrm { E } } ) } \end{array}$ is a sum of positive semidefinite matrices, $v ^ { \top } \mathcal { H } _ { P _ { i } } ( \theta ^ { \mathrm { E } } ) v = 0$ and hence $v \in$ ker $\left( \mathcal { H } _ { P _ { i } } ( \theta ^ { \mathrm { E } } ) \right)$ for every i. By Proposition B.5,

$$
\ker \big ( \mathcal { H } _ { P _ { i } } ( \theta ^ { \mathrm { E } } ) \big ) = U _ { P _ { i } } ^ { \perp } = \ker ( \Pi _ { P _ { i } } ) ,
$$

so Π $\operatorname { I } _ { P _ { i } } v = 0$ for every i. By definition of $S _ { k }$ and symmetry of the orthogonal projectors,

$$
\langle v , S _ { k } \rangle = \sum _ { i = 1 } ^ { k } \langle v , \Pi _ { { \cal P } _ { i } } \left( \hat { \phi } _ { i } - \phi _ { { \cal P } _ { i } } ^ { \mathrm { E } } \right) \rangle = \sum _ { i = 1 } ^ { k } \langle \Pi _ { { \cal P } _ { i } } v , \hat { \phi } _ { i } - \phi _ { { \cal P } _ { i } } ^ { \mathrm { E } } \rangle = 0 .
$$

Therefore $S _ { k } \in \ker ( \mathcal { H } _ { k } ) ^ { \perp } = \mathrm { i m } ( \mathcal { H } _ { k } )$ . Since $\mathcal { H } _ { k } ^ { \dag / 2 } \mathcal { H } _ { k } ^ { 1 / 2 }$ is the orthogonal projector onto im $( \mathcal { H } _ { k } )$ , the right-hand side of (15) equals $\langle S _ { k } , u \rangle$

To proceed, we use the first-order optimality of $\theta _ { k }$ for $L _ { k }$ over $\Theta$ and of $\theta ^ { \mathrm { E } }$ for $L _ { k } ^ { \mathrm { p o p } }$ to derive a deterministic localization bound.

Lemma B.10 (Deterministic localization bound). Let Assumption 2.1 hold and $\theta ^ { E } \in \Theta$ . For all $k \geq 1$ with $\lambda _ { k } > 0$

$$
\begin{array} { r } { \chi ( - R _ { k } \rho _ { k } ) \rho _ { k } \ \leq \ \omega _ { k } . } \end{array}\tag{16}
$$

Additionally, on the localization event

$$
\mathcal { E } _ { k } : = \Big \{ \omega _ { k } \leq \frac { 1 - e ^ { - 1 } } { R _ { k } } \Big \} ,\tag{17}
$$

we have $R _ { k } \rho _ { k } \le 1$ and

$$
\rho _ { k } \leq { \frac { 1 } { 1 - e ^ { - 1 } } } \omega _ { k } .\tag{18}
$$

Proof. Since $\theta _ { k }$ minimizes $L _ { k }$ over Θ, the first-order optimality condition gives

$$
\left. \nabla L _ { k } ( \theta _ { k } ) , \theta - \theta _ { k } \right. \geq 0 \quad \forall \theta \in \Theta .
$$

By well-specification and feature matching for the projected population loss, $\nabla L _ { k } ^ { \mathrm { p o p } } ( \theta ^ { \mathrm { E } } ) = 0$ . Using the decomposition $\nabla L _ { k } ( \theta _ { k } ) = \nabla L _ { k } ^ { \mathrm { p o p } } ( \theta _ { k } ) - S _ { k }$ and taking $\theta = \theta ^ { \mathrm { E } }$ , we have

$$
\langle \nabla L _ { k } ^ { \mathrm { p o p } } ( \theta _ { k } ) - S _ { k } , \theta ^ { \mathrm { E } } - \theta _ { k } \rangle \geq 0 .
$$

Rearranging and subtracting $\langle \nabla L _ { k } ^ { \mathrm { p o p } } ( \theta ^ { \mathrm { E } } ) , \theta _ { k } - \theta ^ { \mathrm { E } } \rangle = 0 \mathrm { . }$

$$
\begin{array} { r l } & { \langle \nabla L _ { k } ^ { \mathrm { p o p } } ( \theta _ { k } ) - \nabla L _ { k } ^ { \mathrm { p o p } } ( { \theta } ^ { \mathrm { E } } ) , \theta _ { k } - { \theta } ^ { \mathrm { E } } \rangle \leq \langle S _ { k } , \theta _ { k } - { \theta } ^ { \mathrm { E } } \rangle . } \end{array}\tag{19}
$$

By Remark B.9 and the Cauchy–Schwarz inequality, the right-hand side of (19) satisfies

$$
\begin{array} { r } { \langle S _ { k } , \theta _ { k } - \theta ^ { \operatorname { E } } \rangle = \langle ( \mathcal { H } _ { k } ^ { \dag } ) ^ { 1 / 2 } S _ { k } , \mathcal { H } _ { k } ^ { 1 / 2 } ( \theta _ { k } - \theta ^ { \operatorname { E } } ) \rangle \leq \omega _ { k } \rho _ { k } . } \end{array}\tag{20}
$$

Combining the lower bound of Lemma B.8 with (19) and (20),

$$
\begin{array} { r } { \chi ( - R _ { k } \rho _ { k } ) \rho _ { k } ^ { 2 } \le \omega _ { k } \rho _ { k } . } \end{array}
$$

Dividing by $\rho _ { k } ~ ( \mathrm { i f } ~ \rho _ { k } > 0$ ; the bound is trivial otherwise) yields (16).

Finally, on the event $\mathcal { E } _ { k } .$ , inequality (16) gives

$$
\frac { 1 - e ^ { - R _ { k } \rho _ { k } } } { R _ { k } } = \chi ( - R _ { k } \rho _ { k } ) \rho _ { k } \le \omega _ { k } \le \frac { 1 - e ^ { - 1 } } { R _ { k } } ,
$$

hence $1 - e ^ { - R _ { k } \rho _ { k } } \le 1 - e ^ { - 1 }$ , i.e. $R _ { k } \rho _ { k } \le 1$ . Since $\chi$ is increasing, $\chi ( - R _ { k } \rho _ { k } ) \geq \chi ( - 1 ) = 1 - e ^ { - 1 }$ , and therefore

$$
\left( 1 - e ^ { - 1 } \right) \rho _ { k } \leq \chi ( - R _ { k } \rho _ { k } ) \rho _ { k } \leq \omega _ { k } ,
$$

which gives (18).

In the next step, we are bounding the excess risk on the event $\mathcal { E } .$

Lemma B.11 (Excess risk bound on $\mathcal { E } _ { k } )$ . Let Assumption 2.1 hold and $\theta ^ { E } \in \Theta$ . Fix $k \geq 1$ with $\lambda _ { k } > 0$ . On the event $\mathcal { E } _ { k }$ defined in (17),

$$
L _ { k } ^ { p o p } ( \theta _ { k } ) - L _ { k } ^ { p o p } ( \theta ^ { E } ) \leq \frac { 1 } { 1 - e ^ { - 1 } } \omega _ { k } ^ { 2 } .\tag{21}
$$

Proof. Since $\theta _ { k }$ minimizes $L _ { k }$ over Θ and $\theta ^ { \mathrm { E } } \in \Theta$ , we have $L _ { k } ( \theta _ { k } ) \le L _ { k } ( \theta ^ { \mathrm { E } } )$ . Using $L _ { k } ( \theta ) = L _ { k } ^ { \mathrm { p o p } } ( \theta ) - \langle \theta , S _ { k } \rangle$

$$
\begin{array} { r l } & { L _ { k } ^ { \mathrm { p o p } } ( \theta _ { k } ) - L _ { k } ^ { \mathrm { p o p } } ( \theta ^ { \mathrm { E } } ) = \langle \theta _ { k } - \theta ^ { \mathrm { E } } , S _ { k } \rangle + \big ( L _ { k } ( \theta _ { k } ) - L _ { k } ( \theta ^ { \mathrm { E } } ) \big ) } \\ & { \qquad \leq \langle \theta _ { k } - \theta ^ { \mathrm { E } } , S _ { k } \rangle . } \end{array}
$$

By (20) and, on $\mathcal { E } _ { k }$ , Lemma B.10,

$$
\langle \theta _ { k } - \theta ^ { \operatorname { E } } , S _ { k } \rangle \leq \omega _ { k } \rho _ { k } \leq \frac { 1 } { 1 - e ^ { - 1 } } \omega _ { k } ^ { 2 } .
$$

Fix the number of trajectories per round $m \in \mathbb { N } ^ { + }$ and $\delta \in ( 0 , 1 )$ . Recall that $\begin{array} { r } { U = \sum _ { P \in \mathcal { P } } U _ { P } } \end{array}$ has dimension $d _ { U } \ \leq \ d .$ We assume $d _ { U } \ \geq \ 1 \colon$ if $U = \{ 0 \}$ , all projected scores and Hessians vanish and there is nothing to estimate. At each round $k \geq 1 , P _ { k }$ is selected based on the history, and m fresh, conditionally independent expert trajectories are drawn under $P _ { k }$ . Recall

$$
\begin{array} { r l r l } & { \mathcal { H } _ { k } : = \displaystyle \sum _ { i = 1 } ^ { k } \mathcal { H } _ { P _ { i } } ( \theta ^ { \mathrm { E } } ) , \qquad } & & { \lambda _ { k } : = \operatorname* { m i n } _ { v \in U , \ \| v \| _ { 2 } = 1 } \| v \| _ { \mathcal { H } _ { k } } ^ { 2 } , } \\ & { \Sigma _ { P } : = \mathrm { C o v } _ { P } ^ { \pi _ { P } ^ { \mathrm { E } } } \left( \Pi _ { P } \phi ( \tau ) \right) , \qquad } & & { d _ { \star } : = d _ { U } \operatorname* { m a x } _ { P \in \mathcal { P } } \left\| \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) ^ { \dag / 2 } \Sigma _ { P } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) ^ { \dag / 2 } \right\| _ { 2 } . } \end{array}
$$

Note that $\lambda _ { k } = 0$ until the selected dynamics cover all of $U .$

Since $\Sigma _ { P }$ is supported on $U _ { P } = \mathrm { i m } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } )$ , the definition of $d _ { \star } \mathrm {  ~ g i ~ }$ ves

$$
\Sigma _ { P } \ \preceq \ { \frac { d _ { \star } } { d _ { U } } } \ { \mathcal { H } } _ { P } ( \theta ^ { \mathrm { E } } ) \qquad { \mathrm { f o r ~ e v e r y ~ } } P \in { \mathcal { P } } .\tag{22}
$$

Projecting the covariance decomposition (11) onto $U _ { P }$ gives

$$
\Sigma _ { \boldsymbol { P } } = \eta ^ { - 1 } \mathcal { H } _ { \boldsymbol { P } } ( \boldsymbol { \theta } ^ { \mathrm { E } } ) + \Pi _ { \boldsymbol { P } } \Sigma _ { \boldsymbol { P } } ^ { \mathrm { d y n } } ( \boldsymbol { \theta } ^ { \mathrm { E } } ) \Pi _ { \boldsymbol { P } } \succeq \eta ^ { - 1 } \mathcal { H } _ { \boldsymbol { P } } ( \boldsymbol { \theta } ^ { \mathrm { E } } ) .
$$

Thus $d _ { \star } \geq d _ { U } / \eta .$ , with equality for deterministic transitions and initial state (Corollary B.2). The ratio $d _ { \star } / d _ { U }$ measures the largest sampling variance relative to curvature along an identifiable direction, and stochastic transitions can only increase it.

We define the analysis regularization and logarithmic factor

$$
\gamma : = \frac { 4 d _ { U } H ^ { 2 } B _ { \phi } ^ { 2 } } { d _ { \star } m } , \qquad \ell _ { k , \delta } : = \log \frac { 1 } { \delta } + d _ { U } \log 2 + \frac { 1 } { 2 } \log \operatorname * { d e t } \left( I + \frac { \mathcal { H } _ { k } } { \gamma } \right) + \frac { 1 } { 4 } .\tag{23}
$$

The matrix inside the determinant is the identity on $U ^ { \perp }$ , so only directions in U contribute. For the subsequent selection guarantees, we fix the budget n.

Lemma B.12 (Anytime self-normalized concentration). Let Assumption 2.1 hold, and define

$$
X _ { i , j } : = \frac { 1 } { m } \Pi _ { { P } _ { i } } \left( \phi ( \tau _ { i , j } ) - \phi _ { { P } _ { i } } ^ { \mathrm { E } } \right) , \quad S _ { k } : = \sum _ { i = 1 } ^ { k } \sum _ { j = 1 } ^ { m } X _ { i , j } , \quad \omega _ { k } : = \| S _ { k } \| _ { \mathcal { H } _ { k } ^ { \dagger } } .
$$

With probability at least $1 - \delta _ { i }$ simultaneously for all $k \geq 1$

$$
\| S _ { k } \| _ { ( \mathcal H _ { k } + \gamma I ) ^ { - 1 } } \leq 2 \sqrt { \frac { d _ { \star } \ell _ { k , \delta } } { d _ { U } m } } + \frac { 8 H B _ { \phi } \ell _ { k , \delta } } { m \sqrt { \lambda _ { k } + \gamma } } .\tag{24}
$$

On the same event, every round k with $\lambda _ { k } \geq \gamma$ satisfies

$$
\omega _ { k } \leq 2 \sqrt { \frac { 2 d _ { \star } \ell _ { k , \delta } } { d _ { U } m } } + \frac { 8 H B _ { \phi } \ell _ { k , \delta } } { m \sqrt { \lambda _ { k } } } .\tag{25}
$$

Proof. Since the Hessian images do not depend on the parameters, $\begin{array} { r } { U = \sum _ { P \in \mathcal { P } } } \end{array}$ im $\mathcal { H } _ { P } ( 0 )$ is fixed before sampling. In this proof, we work in a fixed orthonormal basis of $U _ { : }$ , with all vectors and matrices restricted to U (and I the identity on $U )$ unless otherwise indicated. Each increment $X _ { i , j }$ lies in $U _ { ; }$ , as it is projected onto $U _ { P _ { i } } \subseteq U$ We use the shorthands

$$
c : = \frac { d _ { \star } } { d _ { U } m } , \qquad r : = \frac { m } { 2 H B _ { \phi } } , \qquad \mathrm { s o ~ t h a t } ~ \gamma c = r ^ { - 2 } ~ \mathrm { b y ~ ( 2 3 ) } .
$$

Step 1: supermartingale. Let $\mathcal { G } _ { i , j - 1 }$ contain all information available before trajectory j of round $i ,$ including $P _ { i }$ and any learner randomization used to select it. Fresh sampling gives

$$
\mathbb { E } [ X _ { i , j } \ | \ { \mathcal G } _ { i , j - 1 } ] = 0 , \quad \| X _ { i , j } \| _ { 2 } \leq \frac { 2 H B _ { \phi } } { m } = \frac { 1 } { r } ,
$$

and, by (22),

$$
\mathbb { E } [ X _ { i , j } X _ { i , j } ^ { \top } \mid \mathcal { G } _ { i , j - 1 } ] = \frac { \Sigma _ { P _ { i } } } { m ^ { 2 } } \preceq \frac { c } { m } \mathcal { H } _ { P _ { i } } ( { \boldsymbol { \theta } } ^ { \mathrm { E } } ) .
$$

For every fixed $z \in U$ with $\| z \| _ { 2 } \leq r$ , we have $| z ^ { \top } X _ { i , j } | \leq 1$ . Using $e ^ { x } \leq 1 + x + x ^ { 2 }$ for $| x | \le 1$ , we obtain

$$
\mathbb { E } [ e ^ { z ^ { \top } X _ { i , j } } \mid \mathcal { G } _ { i , j - 1 } ] \leq 1 + \mathbb { E } [ ( z ^ { \top } X _ { i , j } ) ^ { 2 } \mid \mathcal { G } _ { i , j - 1 } ] \leq \exp \Big ( \frac { c } { m } z ^ { \top } \mathcal { H } _ { P _ { i } } ( \theta ^ { \mathrm { E } } ) z \Big ) .
$$

Iterating this bound over the m trajectories of each round shows that

$$
M _ { k } ( z ) : = \exp \bigl ( z ^ { \top } S _ { k } - c z ^ { \top } \mathcal { H } _ { k } z \bigr )
$$

is a nonnegative supermartingale with $M _ { 0 } ( z ) = 1$

Step 2: mixture and Ville’s inequality. Let $\nu _ { 0 }$ be the uniform distribution on $\{ z \in U : \| z \| _ { 2 } \leq r \}$ . The mixture $\int M _ { k } ( z ) \nu _ { 0 } ( \mathrm { d } z )$ is a nonnegative supermartingale starting at one. By Ville’s inequality, with probability at least $1 - \delta .$

$$
\int M _ { k } ( z ) \nu _ { 0 } ( \mathrm { d } z ) \leq 1 / \delta \quad { \mathrm { f o r ~ a l l ~ } } k \geq 0 .
$$

On this event, the Donsker–Varadhan variational inequality gives, for every round k and every probability distribution $\nu \ll \nu _ { 0 }$ ，

$$
\mathbb { E } _ { z \sim \nu } \big [ z ^ { \top } S _ { k } - c z ^ { \top } \mathcal { H } _ { k } z \big ] \leq \mathrm { K L } ( \nu \| \nu _ { 0 } ) + \log \frac { 1 } { \delta } .\tag{26}
$$

Since this holds for all such ν simultaneously, ν may depend on the data.

Step 3: choice of ν. Fix k and define the positive definite matrix on U

$$
W _ { k } : = c \mathcal { H } _ { k } + r ^ { - 2 } I = c \left( \mathcal { H } _ { k } + \gamma I \right) .
$$

In particular, $\lambda _ { \operatorname* { m i n } } ( W _ { k } ) = c ( \lambda _ { k } + \gamma ) \geq r ^ { - 2 }$ . If $S _ { k } = 0$ , the claimed bounds at this round hold trivially. Otherwise, set

$$
\alpha _ { k } : = \operatorname* { m i n } \left\{ \frac { 1 } { 2 } , \frac { r \sqrt { \lambda _ { \operatorname* { m i n } } ( W _ { k } ) } } { 2 \| S _ { k } \| _ { W _ { k } ^ { - 1 } } } \right\} ,
$$

and let $\nu _ { k }$ be uniform on the ellipsoid

$$
\left\{ \alpha _ { k } W _ { k } ^ { - 1 } S _ { k } + \frac { 1 } { 2 } W _ { k } ^ { - 1 / 2 } u : u \in U , \ \| u \| _ { 2 } \leq 1 \right\} .
$$

Its center and radius satisfy

$$
\left\| \alpha _ { k } W _ { k } ^ { - 1 } S _ { k } \right\| _ { 2 } \leq \frac { \alpha _ { k } \left\| S _ { k } \right\| _ { W _ { k } ^ { - 1 } } } { \sqrt { \lambda _ { \operatorname* { m i n } } ( W _ { k } ) } } \leq \frac { r } { 2 } , \quad \frac { 1 } { 2 } \left\| W _ { k } ^ { - 1 / 2 } \right\| _ { 2 } \leq \frac { r } { 2 } ,
$$

where the second bound uses $\lambda _ { \operatorname* { m i n } } ( W _ { k } ) \geq r ^ { - 2 }$ . Hence $\nu _ { k } \ll \nu _ { 0 }$ . Its mean and covariance are

$$
\mathbb { E } _ { \nu _ { k } } [ z ] = \alpha _ { k } W _ { k } ^ { - 1 } S _ { k } , \quad \mathrm { C o v } _ { \nu _ { k } } ( z ) = \frac { W _ { k } ^ { - 1 } } { 4 ( d _ { U } + 2 ) } ,
$$

and, since $r ^ { 2 } W _ { k } = I + \mathcal { H } _ { k } / \gamma$ , the ratio of support volumes gives

$$
\mathrm { K L } ( \nu _ { k } \| \nu _ { 0 } ) = d _ { U } \log 2 + \frac 1 2 \log \operatorname* { d e t } _ { U } \left( I + \frac { \mathcal { H } _ { k } } { \gamma } \right) .
$$

This restricted determinant equals the full-space determinant in (23), since $\mathcal { H } _ { k }$ vanishes on $U ^ { \perp }$

Step $\it 4 :$ solving for $S _ { k }$ . Since $0 \preceq c \mathcal { H } _ { k } \preceq W _ { k }$

$$
\begin{array} { r l } & { \mathbb E _ { \nu _ { k } } \left[ z ^ { \top } S _ { k } - c z ^ { \top } \mathcal H _ { k } z \right] = \alpha _ { k } \left\| S _ { k } \right\| _ { W _ { k } ^ { - 1 } } ^ { 2 } - c \alpha _ { k } ^ { 2 } S _ { k } ^ { \top } W _ { k } ^ { - 1 } \mathcal H _ { k } W _ { k } ^ { - 1 } S _ { k } - \frac { c \mathrm { t r } ( \mathcal H _ { k } W _ { k } ^ { - 1 } ) } { 4 ( d _ { U } + 2 ) } } \\ & { \qquad \geq \alpha _ { k } ( 1 - \alpha _ { k } ) \left\| S _ { k } \right\| _ { W _ { k } ^ { - 1 } } ^ { 2 } - \frac 1 4 . } \end{array}
$$

Substituting into (26) yields $\alpha _ { k } ( 1 - \alpha _ { k } ) \| S _ { k } \| _ { W _ { \ast } ^ { - 1 } } ^ { 2 } \leq \ell _ { k , \delta }$ . If $\alpha _ { k } = 1 / 2$ , then $\| S _ { k } \| _ { W _ { k } ^ { - 1 } } \leq 2 \sqrt { \ell _ { k , \delta } }$ . Otherwise, k $1 - \alpha _ { k } \ge 1 / 2$ and the definition of $\alpha _ { k }$ gives $\| S _ { k } \| _ { W _ { k } ^ { - 1 } } \leq 4 \ell _ { k , \delta } / ( r \sqrt { \lambda _ { \operatorname* { m i n } } ( W _ { k } ) } )$ . Combining the two cases,

$$
\| S _ { k } \| _ { W _ { k } ^ { - 1 } } \leq 2 \sqrt { \ell _ { k , \delta } } + \frac { 4 \ell _ { k , \delta } } { r \sqrt { \lambda _ { \operatorname* { m i n } } ( W _ { k } ) } } .
$$

Since $\| S _ { k } \| _ { W _ { k } ^ { - 1 } } = c ^ { - 1 / 2 } \| S _ { k } \| _ { ( \mathcal { H } _ { k } + \gamma I ) ^ { - } }$ <sub>−1</sub> and $\lambda _ { \operatorname* { m i n } } ( W _ { k } ) = c ( \lambda _ { k } + \gamma )$ , rescaling proves (24). The norm agrees with its full-space version because $S _ { k } \in U$

Step $5 { : }$ removing the regularization. If $\lambda _ { k } \geq \gamma$ , then $\mathcal { H } _ { k } \succeq \lambda _ { k } I$ on U, so $\mathcal { H } _ { k } ^ { - 1 } \preceq ( 1 + \gamma / \lambda _ { k } ) ( \mathcal { H } _ { k } + \gamma I ) ^ { - 1 }$ there, and

$$
\omega _ { k } \leq \sqrt { 1 + \frac { \gamma } { \lambda _ { k } } } \left\| S _ { k } \right\| _ { ( \mathcal { H } _ { k } + \gamma I ) ^ { - 1 } } .
$$

Applying (24) and $1 + \gamma / \lambda _ { k } \leq 2$ gives (25). The linear term simplifies exactly, because $\sqrt { 1 + \gamma / \lambda _ { k } } / \sqrt { \lambda _ { k } + \gamma } =$ $1 / \sqrt { \lambda _ { k } }$ . All bounds hold on the same event for every round k. □

Let Θ be convex with $\theta ^ { \mathrm { E } } \in \Theta$ , and let $\theta _ { k }$ minimize the projected empirical loss over Θ. Let $R \geq 2 \eta H B _ { \phi }$ be a valid pseudo-self-concordance constant from Lemma B.6. For the stochastic setting, the established choice is $R = \eta H ( H + 1 ) B _ { \phi }$ . The deterministic localization lemma remains a pathwise result and requires no additional concentration argument.

Lemma 3.1. Fix $n \in \mathbb { N } ^ { + } , ~ \delta \in ( 0 , 1 )$ , and $R : = \eta B _ { \phi } H ( H + 1 )$ . Let $n _ { \delta }$ be the first round $k \leq n$ such that

$$
\lambda _ { k } \ \geq \ { \frac { 3 2 R ^ { 2 } d _ { \star } } { ( 1 - e ^ { - 1 } ) ^ { 2 } m } } \log { \frac { 3 \eta d _ { \star } n m } { \delta } } ,\tag{4}
$$

with $n _ { \delta } = \infty \ i f$ no such round exists. Then, with probability at least $1 - \delta$ , simultaneously for all $n _ { \delta } \le k \le n$

$$
\big \| \theta _ { k } - \theta ^ { E } \big \| _ { \mathcal { H } _ { k } } ^ { 2 } \ \leq \ \frac { 1 6 d _ { \star } } { ( 1 - e ^ { - 1 } ) ^ { 2 } m } \log \frac { 3 \eta d _ { \star } k m } { \delta } \ = : \ \beta _ { k } ^ { 2 } ( \delta ) .
$$

Proof. Work on the event from Lemma B.12. Throughout, we use that $d _ { \star } \geq d _ { U } / \eta$ and $R = \eta H ( H + 1 ) B _ { \phi } \geq$ $2 \eta H B _ { \phi }$ , so that

$$
R \frac { d _ { \star } } { d _ { U } } \geq \frac { R } { \eta } \geq 2 H B _ { \phi } .\tag{27}
$$

The argument relies on the condition

$$
\lambda _ { k } \ \geq \ { \frac { 3 2 R ^ { 2 } d _ { \star } \ell _ { k , \delta } } { ( 1 - e ^ { - 1 } ) ^ { 2 } d _ { U } m } } .\tag{28}
$$

Step 1: the warm-up condition implies (28). By Proposition B.4, every eigenvalue of $\mathcal { H } _ { k }$ is at most $k \eta H ^ { 2 } B _ { \phi } ^ { 2 } .$ With $\gamma = 4 d _ { U } H ^ { 2 } B _ { \phi } ^ { 2 } / ( d _ { \star } m )$ , every eigenvalue of $\mathcal { H } _ { k } / \gamma$ is thus at most $x / 4$ , where $x : = k m \eta d _ { \star } / d _ { U } \geq 1$ . Since the determinant in $\ell _ { k , \delta }$ only involves the $d _ { U }$ dimensions of $U .$

$$
\begin{array} { r } { \ell _ { k , \delta } \leq \log \frac { 1 } { \delta } + d _ { U } \log 2 + \frac { d _ { U } } { 2 } \log \left( 1 + \frac { x } { 4 } \right) + \frac { 1 } { 4 } = \log \frac { 1 } { \delta } + \frac { d _ { U } } { 2 } \log ( 4 + x ) + \frac { 1 } { 4 } . } \end{array}
$$

Since $x \ge 1$ , we have $9 x ^ { 2 } / ( 4 + x ) \geq 9 / 5$ and $\begin{array} { l } { \frac { 1 } { 2 } \log \frac { 9 } { 5 } \geq \frac { 1 } { 4 } } \end{array}$ , hence $\begin{array} { r } { \frac { d _ { U } } { 2 } \log ( 4 + x ) + \frac { 1 } { 4 } \leq d _ { U } \log ( 3 x ) } \end{array}$ . With log $\begin{array} { r } { \frac { 1 } { \delta } \le d _ { U } } \end{array}$ log $\frac { 1 } { \delta }$ and $x \leq$ km ηd<sub>⋆</sub>, this gives

$$
\ell _ { k , \delta } \ \le \ d _ { U } \log ( 3 x / \delta ) \ \le \ d _ { U } \log ( 3 \eta d _ { \star } k m / \delta ) .\tag{29}
$$

In particular, $\ell _ { k , \delta } \ \leq \ d _ { U } \log ( 3 \eta d _ { \star } n m / \delta )$ for every $1 \leq k \leq n$ . Since $\mathcal { H } _ { k } \ \succeq \ \mathcal { H } _ { n _ { i } }$ <sub>δ</sub> for $k \geq n _ { \delta }$ , the warm-up condition (4) implies

$$
\lambda _ { k } \geq \lambda _ { n _ { \delta } } \geq \frac { 3 2 R ^ { 2 } d _ { \star } } { ( 1 - e ^ { - 1 } ) ^ { 2 } m } \log \frac { 3 \eta d _ { \star } n m } { \delta } \geq \frac { 3 2 R ^ { 2 } d _ { \star } \ell _ { k , \delta } } { ( 1 - e ^ { - 1 } ) ^ { 2 } d _ { U } m } \quad \mathrm { f o r ~ e v e r y ~ } n _ { \delta } \leq k \leq n .
$$

Thus (28) holds throughout the remaining budget, independently of the dynamics selected after the warm-up. Step 2: localization. Fix a round $n _ { \delta } ~ \le ~ k ~ \le ~ n$ . Using $\ell _ { k , \delta } \ \geq \ 1 / 4$ , the right-hand side of (28) is at least $8 R ^ { 2 } d _ { \star } / ( d _ { U } m )$ , which is at least γ since $( R d _ { \star } / d _ { U } ) ^ { 2 } \geq 4 H ^ { 2 } B _ { \phi } ^ { 2 }$ by (27). Hence $\lambda _ { k } \geq \gamma$ and (25) applies. We bound both of its terms by $( 1 - e ^ { - 1 } ) \sqrt { \lambda _ { k } } / ( 2 R )$ . For the first, this is exactly (28):

$$
2 \sqrt { \frac { 2 d _ { \star } \ell _ { k , \delta } } { d _ { U } m } } \leq \frac { ( 1 - e ^ { - 1 } ) \sqrt { \lambda _ { k } } } { 2 R } .
$$

For the second, it holds whenever $\lambda _ { k } \geq 1 6 R H B _ { \phi } \ell _ { k , \delta } / ( ( 1 - e ^ { - 1 } ) m )$ , which (28) implies by (27). Consequently, $\omega _ { k } \le ( 1 - e ^ { - 1 } ) \sqrt { \lambda _ { k } } / R = ( 1 - e ^ { - 1 } ) / R _ { k }$ , so the event $\mathcal { E } _ { k }$ holds, and Lemma B.10 yields

$$
\frac { R } { \sqrt { \lambda _ { k } } } \left\| \theta _ { k } - \theta ^ { \mathrm { E } } \right\| _ { \mathcal { H } _ { k } } \leq 1 , \qquad \left\| \theta _ { k } - \theta ^ { \mathrm { E } } \right\| _ { \mathcal { H } _ { k } } \leq \frac { \omega _ { k } } { 1 - e ^ { - 1 } } .
$$

Step 3: confidence bound. Using (28) and (27) again,

$$
\frac { 8 H B _ { \phi } \ell _ { k , \delta } } { m \sqrt { \lambda _ { k } } } \leq \frac { 1 - e ^ { - 1 } } { \sqrt { 2 } } \sqrt { \frac { d _ { \star } \ell _ { k , \delta } } { d _ { U } m } } , \quad \mathrm { h e n c e } \quad \omega _ { k } \leq \left( 2 \sqrt { 2 } + \frac { 1 - e ^ { - 1 } } { \sqrt { 2 } } \right) \sqrt { \frac { d _ { \star } \ell _ { k , \delta } } { d _ { U } m } } \leq 4 \sqrt { \frac { d _ { \star } \ell _ { k , \delta } } { d _ { U } m } } .
$$

Combining the last two bounds with (29) gives

$$
\bigl \| \theta _ { k } - \theta ^ { \mathbf { E } } \bigr \| _ { \mathcal { H } _ { k } } ^ { 2 } \leq \frac { 1 6 d _ { \star } \ell _ { k , \delta } } { ( 1 - e ^ { - 1 } ) ^ { 2 } d _ { U } m } \leq \frac { 1 6 d _ { \star } \log ( 3 \eta d _ { \star } k m / \delta ) } { ( 1 - e ^ { - 1 } ) ^ { 2 } m } .
$$

Every step holds simultaneously for all $n _ { \delta } \le k \le n$ on the event from Lemma B.12, which has probability at least $1 - \delta$ □

## C PROOFS OF SECTION 3.2

We now lay out the results employed to prove Lemma 3.2, Lemma 3.3, Theorem 3.6 and Theorem 3.7

## C.1 Supporting lemmas

We first present useful identities of the suboptimality (1).

Proposition C.1 (Remainder form). Let Assumption 2.1 hold and let P be any dynamics. For any $\theta _ { 1 } , \theta _ { 2 } \in \mathbb { R } ^ { d }$ 2

$$
\begin{array} { r } { V _ { P } ^ { \star } ( \theta _ { 2 } ) - V _ { P } ^ { \pi _ { P } ( \theta _ { 1 } ) } ( \theta _ { 2 } ) = \| \theta _ { 2 } - \theta _ { 1 } \| _ { G _ { P } ( \theta _ { 1 } , \theta _ { 2 } ) } ^ { 2 } , } \end{array}
$$

where $G _ { P }$ is defined in Lemma B.7.

Proof. Let $u : = \theta _ { 2 } - \theta _ { 1 }$ . A Taylor expansion of $V _ { P } ^ { \star }$ around $\theta _ { 1 }$ with integral remainder, together with Lemma B.1, gives

$$
\begin{array} { l } { { \displaystyle V _ { P } ^ { \star } ( \theta _ { 2 } ) = V _ { P } ^ { \star } ( \theta _ { 1 } ) + D V _ { P } ^ { \star } ( \theta _ { 1 } ) [ u ] + \int _ { 0 } ^ { 1 } ( 1 - t ) D ^ { 2 } V _ { P } ^ { \star } ( \theta _ { 1 } + t u ) [ u , u ] \mathrm { d } t } \ ~ } \\ { { \displaystyle ~ = V _ { P } ^ { \star } ( \theta _ { 1 } ) + \langle u , \phi _ { P } ( \theta _ { 1 } ) \rangle + \| u \| _ { G _ { P } ( \theta _ { 1 } , \theta _ { 2 } ) } ^ { 2 } } . } \end{array}
$$

On the other hand, the regularized value of the fixed policy $\pi _ { P } ( \theta _ { 1 } )$ is afine in the reward parameters, with slope $\phi _ { P } ( \theta _ { 1 } )$ , and equals $V _ { P } ^ { \star } ( \theta _ { 1 } )$ at $\theta _ { 1 }$ . Hence

$$
\begin{array} { r } { V _ { P } ^ { \pi _ { P } ( \theta _ { 1 } ) } ( \theta _ { 2 } ) = V _ { P } ^ { \star } ( \theta _ { 1 } ) + \langle u , \phi _ { P } ( \theta _ { 1 } ) \rangle . } \end{array}
$$

Subtracting the two expressions gives the result.

Proposition C.2 (Pseudo self-concordant suboptimality bound). Let Assumption 2.1 hold, let P be any dynamics, and let $\theta _ { 1 } , \theta _ { 2 } \in \mathbb { R } ^ { d }$ . Then, for any symmetric positive semidefinite matrix $X \in \mathbb { R } ^ { d \times d }$ with im $\mathcal { H } _ { P } ( \theta _ { 2 } ) \subseteq \mathrm { i m } X$

$$
\begin{array} { r } { V _ { P } ^ { \star } ( \theta _ { 2 } ) - V _ { P } ^ { \pi _ { P } ( \theta _ { 1 } ) } ( \theta _ { 2 } ) \ \leq \ \psi \left( R \left\| \theta _ { 2 } - \theta _ { 1 } \right\| _ { \Pi _ { P } } \right) \left\| X ^ { \dagger / 2 } \mathcal { H } _ { P } ( \theta _ { 2 } ) X ^ { \dagger / 2 } \right\| _ { 2 } \left\| \theta _ { 2 } - \theta _ { 1 } \right\| _ { X } ^ { 2 } , } \end{array}
$$

where $\psi$ is defined in Lemma B.7.

Proof. Set $u : = \theta _ { 2 } - \theta _ { 1 }$ and $\psi _ { u } : = \psi \left( R \Vert u \Vert _ { \Pi _ { P } } \right)$ . By Proposition C.1 and item 2 of Lemma $\mathrm { B } . 7 _ { \because }$

$$
\begin{array} { r } { V _ { P } ^ { \star } ( \theta _ { 2 } ) - V _ { P } ^ { \pi _ { P } ( \theta _ { 1 } ) } ( \theta _ { 2 } ) = \left\| u \right\| _ { G _ { P } ( \theta _ { 1 } , \theta _ { 2 } ) } ^ { 2 } \leq \psi _ { u } \left\| u \right\| _ { \mathcal { H } _ { P } ( \theta _ { 2 } ) } ^ { 2 } . } \end{array}
$$

Since $X ^ { 1 / 2 } X ^ { \dag / 2 } = X ^ { \dag / 2 } X ^ { 1 / 2 }$ is the orthogonal projector onto im $X ,$ and im $\mathcal { H } _ { P } ( \theta _ { 2 } ) \subseteq \mathrm { i m } X$ , we have $\mathcal { H } _ { P } ( \theta _ { 2 } ) =$ $X ^ { 1 / 2 } \left( X ^ { \dag / 2 } \mathcal { H } _ { P } ( \theta _ { 2 } ) X ^ { \dag / 2 } \right) X ^ { 1 / 2 }$ . Hence

$$
\begin{array} { r } { \left\| u \right\| _ { \mathcal { H } _ { P } ( \theta _ { 2 } ) } ^ { 2 } = \left( X ^ { 1 / 2 } u \right) ^ { \top } \left( X ^ { \dag / 2 } \mathcal { H } _ { P } ( \theta _ { 2 } ) X ^ { \dag / 2 } \right) \left( X ^ { 1 / 2 } u \right) \le \left\| X ^ { \dag / 2 } \mathcal { H } _ { P } ( \theta _ { 2 } ) X ^ { \dag / 2 } \right\| _ { 2 } \left\| u \right\| _ { X } ^ { 2 } , } \end{array}
$$

which concludes the proof.

We now show that once the warm-up phase has terminated, i.e, condition (4), pseudo self-concordance factors (Lemma B.6) are upper bounded by their localized constants.

Lemma C.3 (Localized self-concordance constants). Let Assumption 2.1 hold and $\theta ^ { E } \in \Theta$ . On the event of Lemma B.12, for every round $n _ { \delta } \le k \le n$ , where $n _ { \delta }$ is defined by the warm-up condition (4), and every $P \in \mathcal { P }$

$$
\psi \left( R \left\| \theta ^ { E } - \theta _ { k } \right\| _ { \Pi _ { P } } \right) \leq 1 , \qquad \exp \left( R \left\| \theta ^ { E } - \theta _ { k } \right\| _ { \Pi _ { P } } \right) \leq e .
$$

Proof. Fix $P \in \mathcal { P }$ and $n _ { \delta } \le k \le n$ . By Step 1 of the proof of Lemma 3.1, $\lambda _ { k } \ge \lambda _ { n _ { \delta } } > 0$ . Since $U _ { P } \subseteq U$ and $\mathcal { H } _ { k } \succeq \lambda _ { k } \Pi _ { U }$ (see the proof of Lemma B.8),

$$
R \left\| \theta ^ { \mathrm { E } } - \theta _ { k } \right\| _ { \Pi _ { P } } \leq R \left\| \theta ^ { \mathrm { E } } - \theta _ { k } \right\| _ { \Pi _ { U } } \leq R { \frac { \left\| \theta ^ { \mathrm { E } } - \theta _ { k } \right\| _ { \mathcal { H } _ { k } } } { \sqrt { \lambda _ { k } } } } = R _ { k } \rho _ { k } .
$$

By Step 2 of the proof of Lemma 3.1, the localization event $\mathcal { E } _ { k } \ ( 1 7 )$ holds, so Lemma B.10 gives $R _ { k } \rho _ { k } \le 1$ . Since ψ and exp are increasing,

$$
\psi \left( R \left\| \theta ^ { \mathrm { E } } - \theta _ { k } \right\| _ { \mathrm { I } _ { P } } \right) \leq \psi ( 1 ) = 1 \qquad { \mathrm { a n d } } \qquad \exp \left( R \left\| \theta ^ { \mathrm { E } } - \theta _ { k } \right\| _ { \mathrm { I } _ { P } } \right) \leq e .
$$

Additionally, we provide a characterization of the lower bound of optimal value Hessians evaluated at the expert parameter $\theta ^ { \mathrm { E } }$ , as well as the resulting once-cycle coverage $\lambda _ { | \mathcal { P } | }$ obtained after following round-robin for $| \mathcal { P } |$ rounds.

Lemma C.4 (Hessian lower bound). Assume finite $s$ and ${ \mathcal { A } } ,$ and $U \neq \{ 0 \}$ . Let $\nu _ { h , P } ^ { E } ( s )$ denote the probability that the expert visits s at timestep h under $P ,$ and call $( h , s )$ reachable under P when this probability is positive. For $v \in \mathbb { R } ^ { d }$ , let $x _ { h , P } ^ { v } ( s , a ) : = \langle v , A _ { h , P } ^ { \theta ^ { E } } ( s , a ) \rangle$ , and define the minimal expert occupancy and the minimal identifiable advantage gap

$$
\mu _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { \substack { P \in \mathscr { P } , a \in A } } \nu _ { h , P } ^ { E } ( s ) \pi _ { h , P } ^ { E } ( a \mid s ) , \quad \mathrm { ~ \quad } \Delta : = \operatorname* { m i n } _ { \substack { P \in \mathscr { P } : U _ { P } \neq \{ 0 \} } } \operatorname* { m a x } _ { \substack { a \in A } } \left| \alpha _ { h , P } ^ { v } ( s , a ) \right| .
$$

Then $\mu _ { \mathrm { m i n } } > 0 , \Delta > 0$ , and for every $P \in \mathcal { P }$

$$
\begin{array} { r } { \mathcal { H } _ { P } \big ( \theta ^ { E } \big ) \ \succeq \ \kappa \Pi _ { P } , \qquad \kappa : = \eta \mu _ { \mathrm { m i n } } \Delta ^ { 2 } . } \end{array}
$$

Proof. Fix $P \in \mathcal { P }$ and $v \in \mathbb { R } ^ { d }$

Step 1 (per-step decomposition). By definition of the feature advantage, $\mathbb { E } _ { a \sim \pi _ { h , P } ^ { \mathrm { E } } ( \cdot | s ) } \left[ \alpha _ { h , P } ^ { v } ( s , a ) \right] = 0$ for every $( h , s )$ . Hence $\left( \alpha _ { h , P } ^ { v } ( s _ { h } , a _ { h } ) \right) _ { h = 1 } ^ { H }$ is a martingale diference sequence along the expert trajectory, and its terms are uncorrelated. Since $\pi _ { P } ^ { \mathrm { E } } = \pi _ { P } ^ { \mathrm { - } } ( \theta ^ { \mathrm { E } } )$ , Lemma B.1 gives

$$
\begin{array} { r } { \| v \| _ { \mathcal H _ { P } ( \theta ^ { \mathrm E } ) } ^ { 2 } = \eta \mathbb E \left[ \left( \sum _ { h = 1 } ^ { H } \alpha _ { h , P } ^ { v } ( s _ { h } , a _ { h } ) \right) ^ { 2 } \right] = \eta \displaystyle \sum _ { h = 1 } ^ { H } \sum _ { s , a } \nu _ { h , P } ^ { \mathrm E } ( s ) \pi _ { h , P } ^ { \mathrm E } ( a \mid s ) \alpha _ { h , P } ^ { v } ( s , a ) ^ { 2 } . } \end{array}
$$

Step 2 (positivity of $\mu _ { \mathrm { { m i n } } }$ and $\Delta )$ . Since $\pi _ { h , P } ^ { \mathrm { E } }$ has full support (9) and $s , A$ and $\mathcal { P }$ are finite, $\mu _ { \mathrm { { m i n } } } > 0$

By Step 1, $\left\| \boldsymbol { v } \right\| _ { \mathcal { H } _ { P } ( \boldsymbol { \theta } ^ { \mathrm { E } } ) } ^ { 2 }$ is a sum of non-negative terms with positive weights on reachable pairs. Hence $v \in$ ker $\left( \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \right)$ if and only if $\alpha _ { h , P } ^ { v } ( s , a ) = 0$ for every $h ,$ every s reachable at $h ,$ and every a. Since $U _ { P } = $ ker $\big ( \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \big ) ^ { \perp }$ (Proposition B.5), every unit $v \in U _ { P }$ lies outside the kernel, so the inner maximum defining $\Delta$ is strictly positive. This maximum is a maximum of finitely many continuous functions of $v ,$ hence continuous, and the unit sphere of $U _ { P }$ is compact. The minimum over v is therefore attained and strictly positive, and finiteness of  gives $\Delta > 0$

Step 3 (conclusion). If $U _ { P } = \{ 0 \}$ , then $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) = 0 = \Pi _ { P }$ and the claim holds trivially. Otherwise, let $w \in U _ { P }$ be a unit vector, and let $( h ^ { \star } , s ^ { \star } , a ^ { \star } )$ attain the inner maximum in the definition of $\Delta$ for $w$ . Keeping only that term in Step 1,

$$
\begin{array} { r l r } { \| w \| _ { { \mathcal H } _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } } & { \ge } & { \eta \nu _ { h ^ { \star } , P } ^ { \mathrm { E } } \big ( s ^ { \star } \big ) \pi _ { h ^ { \star } , P } ^ { \mathrm { E } } \big ( a ^ { \star } \mid s ^ { \star } \big ) \alpha _ { h ^ { \star } , P } ^ { w } \big ( s ^ { \star } , a ^ { \star } \big ) ^ { 2 } \ge \eta \mu _ { \operatorname* { m i n } } \Delta ^ { 2 } = \kappa . } \end{array}
$$

For general $v \in \mathbb { R } ^ { d }$ , since $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } )$ vanishes on $U _ { P } ^ { \perp } , \| v \| _ { \mathcal H _ { P } ( \theta ^ { \mathrm E } ) } ^ { 2 } = \| \Pi _ { P } v \| _ { \mathcal H _ { P } ( \theta ^ { \mathrm E } ) } ^ { 2 } \geq \kappa \| \Pi _ { P } v \| _ { 2 } ^ { 2 } = \kappa v ^ { \top } \Pi _ { P } v$ , which is $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \succeq \kappa \Pi _ { P }$ □

Corollary C.5 (Information coverage after one cycle). Assume $U \neq \{ 0 \}$ . After selecting each dynamics $P \in \mathcal { P }$ once,

$$
\lambda _ { | \mathcal { P } | } = \lambda _ { \operatorname* { m i n } } ^ { U } \left( \sum _ { P \in \mathcal { P } } \mathcal { H } _ { P } ( \theta ^ { E } ) \right) \geq \eta \mu _ { \operatorname* { m i n } } \Delta ^ { 2 } \operatorname* { m i n } _ { v \in U } \sum _ { P \in \mathcal { P } } \| \Pi _ { P } v \| _ { 2 } ^ { 2 } > 0 ,
$$

where $\mu _ { \mathrm { { m i n } } }$ and $\Delta$ are defined in Lemma $C . 4 .$

Proof. After one complete cycle, $\begin{array} { r } { \mathcal { H } _ { | \mathcal { P } | } = \sum _ { P \in \mathcal { P } } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) } \end{array}$ . By Lemma C.4,

$$
\mathcal { H } _ { | \mathcal { P } | } \succeq \eta \mu _ { \mathrm { m i n } } \Delta ^ { 2 } \sum _ { P \in \mathcal { P } } \Pi _ { P } ,
$$

and since $v ^ { \top } \Pi _ { P } v = \left. \Pi _ { P } v \right. _ { 2 } ^ { 2 }$ , taking the minimum of the quadratic form over unit vectors in $U$ gives the lower bound. Moreover, since each $\Pi _ { P }$ is an orthogonal projector and $\begin{array} { r } { U = \sum _ { P \in \mathcal { P } } U _ { P } } \end{array}$ 2

$$
\ker \left( \sum _ { P \in \mathcal { P } } \Pi _ { P } \right) = \bigcap _ { P \in \mathcal { P } } U _ { P } ^ { \perp } = U ^ { \perp } ,
$$

so $\textstyle \sum _ { P \in { \mathcal { P } } } \Pi _ { P }$ is positive definite on $U ,$ and its minimum quadratic form over the unit sphere of $U$ is strictly positive. □

Proposition C.6. Let $a , b > 0$ with $a b \geq 1$ . Then $x \geq a \log ( b x )$ for every $x \geq 2 a \log ( 2 a b )$ .

Proof. The map $f ( x ) : = x - a \log ( b x )$ is increasing for $x \geq a .$ and $x _ { 0 } : = 2 a \log ( 2 a b ) \geq a$ since ab $\geq 1$ . Using log $y \leq y .$ , we get $f ( x _ { 0 } ) = a \left( \log ( 2 a b ) - \log \log ( 2 a b ) \right) \geq 0$ □

Lemma 3.2. Fix $n \in \mathbb { N } ^ { + }$ and $\delta \in ( 0 , 1 )$ . With probability at least $1 - \delta _ { i }$ , it holds for all $P \in \mathcal { P }$ and $n _ { \delta } \le k \le n$ that

$$
\mathrm { S u b O p t } ( \theta _ { k } , { \cal P } ) \ \leq \ g _ { k } ( { \cal P } ) \beta _ { k } ^ { 2 } ( \delta ) .
$$

Proof. We work on the event of Lemma B.12, which has probability at least $1 - \delta$ . Fix $P \in \mathcal { P }$ and a round $n _ { \delta } \le k \le n$ , and let $u _ { k } : = \theta _ { k } - \theta ^ { \mathrm { E } }$ . Recall that

$$
\mathrm { S u b O p t } ( \theta _ { k } , P ) = V _ { P } ^ { \star } ( \theta ^ { \mathrm { E } } ) - V _ { P } ^ { \pi _ { P } ( \theta _ { k } ) } ( \theta ^ { \mathrm { E } } ) , \qquad g _ { k } ( P ) = \left\| \mathcal { H } _ { k } ^ { \dagger / 2 } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \mathcal { H } _ { k } ^ { \dagger / 2 } \right\| _ { 2 } .
$$

Step 1 (image condition). By Step 1 of the proof of Lemma 3.1, $\lambda _ { k } \ge \lambda _ { n _ { \delta } } > 0$ . Since $\mathcal { H } _ { k }$ vanishes on $U ^ { \perp }$ and $\mathcal { H } _ { k } \succeq \lambda _ { k } \Pi _ { U }$ , we have im $\mathcal { H } _ { k } = U$ . As im $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) = U _ { P } \subseteq U$ by Proposition B.5, it follows that im $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \subseteq$ im $\mathcal { H } _ { k }$

Step 2 (change of geometry). By Step 1, Proposition C.2 applies with $\theta _ { 1 } = \theta _ { k } , \theta _ { 2 } = \theta ^ { \mathrm { E } }$ , and $X = \mathcal { H } _ { k }$

$$
\begin{array} { r } { \mathrm { S u b O p t } ( \theta _ { k } , P ) \le \psi \left( R \left. \theta ^ { \mathrm { E } } - \theta _ { k } \right. _ { \mathrm { I } _ { P } } \right) \left. \mathcal { H } _ { k } ^ { \dagger / 2 } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \mathcal { H } _ { k } ^ { \dagger / 2 } \right. _ { 2 } \left. u _ { k } \right. _ { \mathcal { H } _ { k } } ^ { 2 } = \psi \left( R \left. \theta ^ { \mathrm { E } } - \theta _ { k } \right. _ { \mathrm { I } _ { P } } \right) g _ { k } ( P ) \left. u _ { k } \right. _ { \mathcal { H } _ { k } } ^ { 2 } . } \end{array}
$$

Step 3 (localization). By Lemma C.3, $\psi \left( R \left\| \theta ^ { \mathrm { E } } - \theta _ { k } \right\| _ { \Pi _ { P } } \right) \leq 1$ , hence

$$
\begin{array} { r } { \mathrm { S u b O p t } ( \theta _ { k } , P ) \le g _ { k } ( P ) \| u _ { k } \| _ { \mathcal { H } _ { k } } ^ { 2 } . } \end{array}
$$

Step 4 (confidence bound). By Lemma 3.1, $\| u _ { k } \| _ { \mathcal { H } _ { k } } ^ { 2 } \le \beta _ { k } ^ { 2 } ( \delta )$ , which gives

$$
\operatorname { S u b O p t } ( \theta _ { k } , P ) \leq g _ { k } ( P ) \beta _ { k } ^ { 2 } ( \delta ) .
$$

All steps hold on the same event, simultaneously for every $P \in \mathcal { P }$ and $n _ { \delta } \le k \le n$ , which concludes the proof. Lemma 3.3. Fix $n \in \mathbb { N } ^ { + }$ and $\delta \in ( 0 , 1 )$ . Under round-robin selection (6), the warm-up length satisfies

$$
n _ { \delta } \ \leq \ | \mathcal { P } | \left\lceil \frac { 2 R ^ { 2 } \beta _ { n } ^ { 2 } ( \delta ) } { \lambda _ { | \mathcal { P } | } } \right\rceil ,
$$

whenever the right-hand side is at most n.

Proof. By definition of $\beta _ { n } ( \delta )$ in Lemma 3.1, the warm-up condition (4) reads

$$
\lambda _ { k } \ \ge \ 2 R ^ { 2 } \beta _ { n } ^ { 2 } ( \delta ) .
$$

Fix a round $k \leq n$ and let $j : = \lfloor k / \mid \mathcal { P } \mid \rfloor$ be the number of completed round-robin cycles. Each $P \in \mathcal { P }$ has been selected at least $j$ times, so $\mathcal { H } _ { k } \succeq j \mathcal { H } _ { | \mathcal { P } | }$ and hence $\lambda _ { k } \geq j \lambda _ { | \mathcal { P } | }$ , where $\lambda _ { | \mathcal { P } | } > 0$ by Corollary C.5. The warm-up condition therefore holds at round k whenever

$$
j \ge \frac { 2 R ^ { 2 } \beta _ { n } ^ { 2 } ( \delta ) } { \lambda _ { | \mathcal { P } | } } ,
$$

which is the case for $k = | \mathcal { P } | \left[ 2 R ^ { 2 } \beta _ { n } ^ { 2 } ( \delta ) / \lambda _ { | \mathcal { P } | } \right]$ . If this round is at most n, the definition of $n _ { \delta }$ as the first round satisfying (4) gives the claim. □

## D PROOFS OF SECTION 3.3

Having stated the required lemmas, we subsequently prove our main transfer results. We first present the genera transfer guarantee.

Theorem 3.6. Fix $n \in \mathbb { N } ^ { + } , ~ \delta \in ( 0 , 1 )$ , and follow the active rule (7) after round $n _ { \delta }$ . With probability at least $1 - \delta$ , it holds for all $n _ { \delta } < k \leq n$ and $P \in \mathcal { P }$ that

$$
g _ { k } ( P ) \lesssim \left( 1 + \frac { m } { \eta d _ { \star } H ^ { 2 } } \right) \frac { d _ { U } \log ( k m ) } { k - n _ { \delta } } .
$$

If moreover $m \leq \eta d _ { \star } H ^ { 2 }$ and $n \geq 2 n _ { \delta }$

$$
\mathrm { S u b O p t } ( \theta _ { n } , P ) \ \lesssim \ \frac { d _ { \star } d _ { U } \log ( n m ) \log ( \eta d _ { \star } n m / \delta ) } { n m } .\tag{8}
$$

Proof. Setup. We work on the event of Lemma 3.1, which holds with probability at least $1 - \delta .$ , simultaneously for all rounds. Let $\Lambda : = 3 2 R ^ { 2 } d _ { \star } / ( ( 1 - e ^ { - 1 } ) ^ { 2 } m )$ , which lower-bounds the right-hand side of (4) since $3 \eta d _ { \star } n m / \delta \geq e$ By definition of $n _ { \delta } .$ every round $k \geq n _ { \delta }$ satisfies $\lambda _ { k } \geq \Lambda > 0 .$ , so $\mathcal { H } _ { k }$ is invertible on $U ,$ and the confidence and localization bounds apply. The same holds for the plug-in matrix $\begin{array} { r } { \hat { \mathcal { H } } _ { k } : = \sum _ { i = 1 } ^ { k } \mathcal { H } _ { P _ { i } } ( \theta _ { k } ) } \end{array}$ , which has the same image U since Hessian images do not depend on the parameters (Proposition B.5). For $k \geq n _ { \delta }$ , we write

$$
\begin{array} { r } { \mathcal { T } _ { k } ( P ) : = \mathcal { H } _ { k } ^ { \dagger / 2 } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \mathcal { H } _ { k } ^ { \dagger / 2 } , \qquad \hat { \mathcal { T } } _ { k } ( P ) : = \hat { \mathcal { H } } _ { k } ^ { \dagger / 2 } \mathcal { H } _ { P } ( \theta _ { k } ) \hat { \mathcal { H } } _ { k } ^ { \dagger / 2 } , } \end{array}
$$

so that $g _ { k } ( P ) = \| \mathcal { T } _ { k } ( P ) \| _ { 2 }$ and ${ \hat { g } } _ { k } ( P ) = \left\| { \hat { \mathcal { T } } } _ { k } ( P ) \right\| _ { 2 }$ . All matrices below are restricted to $U .$ . Since $\left\| \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \right\| _ { 2 } \leq$ $\eta H ^ { 2 } B _ { \phi } ^ { 2 }$ (Proposition B.4) and $\mathcal { H } _ { k } \succeq \Lambda I$ on $U _ { ; }$

$$
\begin{array} { l l l } { \displaystyle { \mathcal T _ { k } ( P ) } ~ \preceq ~ c _ { m } I , } & { \quad c _ { m } : = \frac { \eta H ^ { 2 } B _ { \phi } ^ { 2 } } { \Lambda } = \frac { ( 1 - e ^ { - 1 } ) ^ { 2 } m } { 3 2 ( H + 1 ) ^ { 2 } \eta d _ { \star } } , } \end{array}\tag{30}
$$

where we used $R ^ { 2 } = \eta ^ { 2 } H ^ { 2 } ( H + 1 ) ^ { 2 } B _ { \phi } ^ { 2 }$

Step 1 (localization of $g _ { k } )$ . Fix $k \geq n _ { \delta }$ . By the proof of Lemma C.3, $R \left\| \theta _ { k } - \theta ^ { \mathrm { E } } \right\| _ { \Pi _ { P } } \leq 1$ for every $P \in { \mathcal { P } } .$ , so item 1 of Lemma B.7 gives $e ^ { - 1 } \mathcal { H } _ { P } ( { \boldsymbol { \theta } } ^ { \mathrm { E } } ) \preceq \mathcal { H } _ { P } ( { \boldsymbol { \theta } } _ { k } ) \preceq e \mathcal { H } _ { P } ( { \boldsymbol { \theta } } ^ { \mathrm { E } } )$ . Summing over the selected dynamics, $e ^ { - 1 } \mathcal { H } _ { k } \preceq$ $\hat { \mathcal { H } } _ { k } \preceq e \mathcal { H } _ { k }$

For positive definite B and positive semidefinite A on U, the Courant–Fischer theorem gives that the i-th largest eigenvalue of $B ^ { - 1 / 2 } A B ^ { - 1 / 2 }$ equals $\begin{array} { r } { \operatorname* { m a x } _ { \dim \mathcal { V } = i } \ \operatorname* { m i n } _ { v \in \mathcal { V } \setminus \{ 0 \} } \left\| v \right\| _ { A } ^ { 2 } / \left\| v \right\| _ { B } ^ { 2 } , } \end{array}$ and each such ratio increases with A and decreases with B in the Loewner order. Hence every eigenvalue of $\hat { \mathcal { T } } _ { k } ( P )$ lies within a factor $e ^ { 2 }$ of the corresponding eigenvalue of $\mathcal { T } _ { k } ( P )$ , and in particular

$$
e ^ { - 2 } g _ { k } ( P ) \ \leq \ \hat { g } _ { k } ( P ) \ \leq \ e ^ { 2 } g _ { k } ( P ) \qquad { \mathrm { f o r ~ e v e r y ~ } } P \in { \mathcal { P } } .\tag{31}
$$

Step $\textit { 2 } ( g r e e d y \ r u l e )$ For $k \geq n _ { \delta }$ , the rule selects $P _ { k + 1 } \in \arg \operatorname* { m a x } _ { P \in \mathcal { P } } \hat { g } _ { k } ( P )$ . By (31), for every $P \in \mathcal { P }$

$$
\begin{array} { r } { g _ { k } ( P ) \le e ^ { 2 } \hat { g } _ { k } ( P ) \le e ^ { 2 } \hat { g } _ { k } ( P _ { k + 1 } ) \le e ^ { 4 } g _ { k } ( P _ { k + 1 } ) . } \end{array}
$$

Since $\log ( 1 + x ) \geq x / ( 1 + x )$ , every $x \in [ 0 , c _ { m } ]$ satisfies $x \leq \bar { c } _ { m } \log ( 1 + x )$ with $\bar { c } _ { m } : = 1 + c _ { m }$ . Applying this to the largest eigenvalue of $\mathcal { T } _ { k } ( P _ { k + 1 } )$ , which lies in $[ 0 , c _ { m } ]$ by (30), and using that the other eigenvalues contribute non-negatively, gives

$$
g _ { k } ( P ) \leq e ^ { 4 } { \bar { c } } _ { m } \log \operatorname* { d e t } \left( I + { \mathcal { T } } _ { k } ( P _ { k + 1 } ) \right) \qquad { \mathrm { f o r ~ e v e r y ~ } } P \in { \mathcal { P } } .\tag{32}
$$

Step 3 (elliptical potential lemma). Fix $P \in \mathcal { P }$ and $n _ { \delta } < k \le n$ . Since $\mathcal { H } _ { k } \succeq \mathcal { H } _ { i }$ on $U$ for $i \leq k$ , we have $g _ { k } ( P ) \leq g _ { i } ( P )$ . Averaging over $i \in \left\{ n _ { \delta } , \ldots , k - 1 \right\}$ and applying (32),

$$
g _ { k } ( P ) \leq { \frac { e ^ { 4 } { \bar { c } } _ { m } } { k - n _ { \delta } } } \sum _ { i = n _ { \delta } } ^ { k - 1 } \log \operatorname * { d e t } \left( I + { \mathcal { T } } _ { i } ( P _ { i + 1 } ) \right) .
$$

Since $\mathcal { H } _ { i + 1 } = \mathcal { H } _ { i } + \mathcal { H } _ { P _ { i + 1 } } ( { \boldsymbol { \theta } } ^ { \mathrm { E } } )$ , we have det $\mathcal { H } _ { i + 1 } = \operatorname* { d e t } \mathcal { H } _ { i }$ det $\left( I + \mathcal { T } _ { i } ( P _ { i + 1 } ) \right)$ on $U _ { \textrm { \scriptsize i } }$ so the sum telescopes to log $\left( \operatorname* { d e t } \mathcal { H } _ { k } / \operatorname* { d e t } \mathcal { H } _ { n _ { \delta } } \right)$ . On $U _ { ; }$ , every eigenvalue of $\mathcal { H } _ { k }$ is at most $k \eta H ^ { 2 } B _ { \phi } ^ { 2 }$ (Proposition B.4) and every eigenvalue of $\mathcal { H } _ { n _ { \delta } }$ is at least $\Lambda$ , so every eigenvalue of $\mathcal { H } _ { n _ { \delta } } ^ { - 1 / 2 } \mathcal { H } _ { k } \mathcal { H } _ { n _ { \delta } } ^ { - 1 / 2 }$ is at most $k c _ { m }$ . Since $U$ has dimension $d _ { U }$ log $\left( \operatorname * { d e t } \mathcal { H } _ { k } / \operatorname * { d e t } \mathcal { H } _ { n _ { \delta } } \right) \leq d _ { U } \log ( 1 + k c _ { m } )$ . Taking the maximum over $P \in \mathcal { P }$

$$
\operatorname* { m a x } _ { P \in \mathcal { P } } g _ { k } ( P ) \leq e ^ { 4 } \bar { c } _ { m } \frac { d _ { U } \log ( 1 + k c _ { m } ) } { k - n _ { \delta } } .\tag{33}
$$

Since $\eta d _ { \star } \geq 1$ , we have $c _ { m } \leq$ m and $c _ { m } \lesssim m / ( \eta d _ { \star } H ^ { 2 } )$ , and $\log ( 1 + k c _ { m } ) \leq \log ( 1 + k m ) \leq 2 \log ( k m )$ as $k m \geq 2$ This proves the first claim.

Step 4 (suboptimality bound). By Lemma 3.2 and (33) at $k = n$ , and since $\beta _ { n } ^ { 2 } ( \delta ) \lesssim d _ { \star } \log ( \eta d _ { \star } n m / \delta ) / m$ and $c _ { m } d _ { \star } / m \lesssim 1 / ( \eta ( H + 1 ) ^ { 2 } )$ , it holds for every $P \in { \mathcal { P } }$ that

$$
\mathrm { S u b O p t } ( \theta _ { n } , P ) \le g _ { n } ( P ) \beta _ { n } ^ { 2 } ( \delta ) \lesssim \left( \frac { d _ { \star } } { m } + \frac { 1 } { \eta ( H + 1 ) ^ { 2 } } \right) \frac { d _ { U } \log ( n m ) \log ( \eta d _ { \star } n m / \delta ) } { n - n _ { \delta } } .
$$

If $m \leq \eta d _ { \star } H ^ { 2 }$ , then $\begin{array} { r } { \frac { 1 } { \eta ( H + 1 ) ^ { 2 } } \leq \frac { 1 } { \eta H ^ { 2 } } \leq \frac { d _ { \star } } { m } } \end{array}$ , and if $n \geq 2 n _ { \delta }$ , then $n - n _ { \delta } \ge n / 2$ . Together, these give (8), on the same event. □

Remark D.1 (Alternative selection criteria). Recall that the active rule (7) maximizes ${ \hat { g } } _ { k } ( P ) = \left\| { \hat { \mathcal { T } } } _ { k } ( P ) \right\| _ { 2 }$ , where

$$
\hat { \mathcal { T } } _ { k } ( P ) : = \hat { \mathcal { H } } _ { k } ^ { \dagger / 2 } \mathcal { H } _ { P } ( \boldsymbol { \theta } _ { k } ) \hat { \mathcal { H } } _ { k } ^ { \dagger / 2 } , \qquad \hat { \mathcal { H } } _ { k } : = \sum _ { j = 1 } ^ { k } \mathcal { H } _ { P _ { j } } ( \boldsymbol { \theta } _ { k } ) ,
$$

is the plug-in estimate of $\mathcal { T } _ { k } ( P ) = \mathcal { H } _ { k } ^ { \dagger / 2 } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \mathcal { H } _ { k } ^ { \dagger / 2 }$ . Instead of the spectral norm, the rule may maximize the trace tr $\hat { \mathcal { T } } _ { k } ( P )$ or the log-determinant log det $( I + \hat { \mathcal { T } } _ { k } ( P ) )$ . Unlike the spectral norm, these criteria account for all directions that matter under $P ,$ not only the least covered one; the log-determinant is the variant used in our experiments (Section 4). Theorem 3.6 holds for both.

Indeed, Steps 3 and 4 of the proof above only use (32), so it sufices to show that it still holds for these choices of $P _ { k + 1 }$ . By Step 1, every eigenvalue of $\hat { \mathcal { T } } _ { k } ( P )$ lies within a factor $e ^ { 2 }$ of the corresponding eigenvalue of $\mathcal { T } _ { k } ( P )$

Trace. Comparing eigenvalues and using the choice of $P _ { k + 1 }$

$$
\begin{array} { r } { g _ { k } ( P ) \le \mathrm { t r } \mathcal { T } _ { k } ( P ) \le e ^ { 2 } \mathrm { t r } \hat { \mathcal { T } } _ { k } ( P ) \le e ^ { 2 } \mathrm { t r } \hat { \mathcal { T } } _ { k } ( P _ { k + 1 } ) \le e ^ { 4 } \mathrm { t r } \mathcal { T } _ { k } ( P _ { k + 1 } ) . } \end{array}
$$

Applying the scalar bound $x \leq \bar { c } _ { m } \log ( 1 + x )$ of Step 2 to each eigenvalue of $\mathcal { T } _ { k } ( P _ { k + 1 } )$ gives tr $\mathcal { T } _ { k } ( P _ { k + 1 } ) \leq$ $\bar { c } _ { m }$ log det $( I + \mathcal { T } _ { k } ( P _ { k + 1 } ) )$ .

Log-determinant. By the same scalar bound, $g _ { k } ( P ) \leq \bar { c } _ { m }$ log det $( I + \mathcal { T } _ { k } ( P ) )$ . Since $\log ( 1 + c x ) \leq c \log ( 1 + x )$ for $c \geq 1$ , comparing eigenvalues as above gives

$$
\begin{array} { r } { \log \operatorname* { d e t } \left( I + { \mathcal { T } } _ { k } ( P ) \right) \leq e ^ { 2 } \log \operatorname* { d e t } \left( I + { \widehat { \mathcal { T } } } _ { k } ( P ) \right) \leq e ^ { 2 } \log \operatorname* { d e t } \left( I + { \widehat { \mathcal { T } } } _ { k } ( P _ { k + 1 } ) \right) \leq e ^ { 4 } \log \operatorname* { d e t } \left( I + { \mathcal { T } } _ { k } ( P _ { k + 1 } ) \right) . } \end{array}
$$

In both cases, (32) holds, and the theorem follows identically.

We now provide the proof of Theorem 3.7 which allows us to transfer the recovered reward to arbitrary unseen dynamics. We introduce a couple of quantities required for our analysis. We measure how $P$ relates to the training set $\mathcal { P }$ through the following quantities:

$$
g _ { n } ( \mathcal { P } ) : = \operatorname* { m a x } _ { P \in \mathcal { P } } g _ { n } ( P ) , \quad s _ { P } : = \left\| \Pi _ { P } \Pi _ { U ^ { \perp } } \right\| _ { 2 } , \quad \gamma ( P ) : = \operatorname* { m a x } _ { w \in U \backslash \{ 0 \} } \frac { \left\| w \right\| _ { \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } } { \operatorname* { m a x } _ { Q \in \mathcal { P } } \left\| w \right\| _ { \mathcal { H } _ { Q } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } } .
$$

Here, $s _ { P } \in [ 0 , 1 ]$ measures how far the identifiable subspace of $P$ leaves $U _ { ; }$ , and $\gamma ( P )$ compares the curvature of $P$ along U with that of the training dynamics. The ratio defining $\gamma ( P )$ is well defined. If $w \in U$ satisfies $\| w \| _ { \mathcal { H } _ { Q } ( \theta ^ { \mathrm { E } } ) } = 0$ for every $Q \in { \mathcal { P } }$ , then $\begin{array} { r } { \dot { w } \in \bigcap _ { Q \in \mathcal { P } } U _ { Q } ^ { \perp } = U ^ { \perp } } \end{array}$ by Proposition B.5, so $w = 0$ . By homogeneity, the maximum may be taken over the unit sphere of $U$ , hence it is attained and finite. Moreover, $\gamma ( P ) \leq 1$ for every $P \in \mathcal { P }$ , since $P$ itself appears in the inner maximum.

Note that $s _ { P } = 0$ and $g _ { \perp } ( P ) = 0$ are both equivalent to $U _ { P } \subseteq U$

Theorem 3.7. Fix $\delta \in ( 0 , 1 )$ and follow the active rule (7) after round $n _ { \delta }$ . If $m \le \eta d _ { \star } H ^ { 2 }$ and $n \geq 2 n _ { \delta }$ , then with probability at least 1  δ, for any $P _ { \cdot }$

$$
\mathrm { S u b O p t } ( \theta _ { n } , P ) = \tilde { \mathcal { O } } \left( \frac { d _ { \star } d _ { U } } { n m } \right) + \mathcal { O } \left( g _ { \perp } ( P ) B _ { \theta } ^ { 2 } \right) .
$$

Proof. We work on the confidence event of Lemma 3.1. Since $n \geq 2 n _ { \delta } \geq n _ { \delta }$ , its bounds and those of Lemma C.3 hold at round n. Let

$$
\Delta : = \theta _ { n } - \theta ^ { \mathrm { E } } = u + v , \qquad u : = \Pi _ { U } \Delta , \qquad v : = \Pi _ { U ^ { \perp } } \Delta .
$$

Since ${ \mathcal { H } } _ { n }$ vanishes on $U ^ { \perp }$ and $\mathcal { H } _ { n } \succeq \lambda _ { n } \Pi _ { U }$ , the confidence bound, the localization of Lemma $\mathrm { { C . 3 } } ,$ , and bounded ness of Θ give

$$
\| u \| _ { \mathcal H _ { n } } \le \beta _ { n } ( \delta ) , \qquad R \left\| u \right\| _ { 2 } \le 1 , \qquad \left\| v \right\| _ { 2 } \le 2 B _ { \theta } .
$$

Since im $\mathcal { H } _ { Q } ( \theta ^ { \mathrm { E } } ) \subseteq U = \mathrm { i m } \mathcal { H } _ { n }$ for every $Q \in { \mathcal { P } }$ , the proof of Lemma 3.2 gives $\begin{array} { r } { \| u \| _ { \mathcal { H } _ { Q } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } \le g _ { n } ( Q ) \left\| u \right\| _ { \mathcal { H } _ { n } } ^ { 2 } } \end{array}$ . Thus, by definition of $\gamma ( P )$ and since $u \in U$ , it follows that

$$
\begin{array} { r l } { \| u \| _ { \mathcal H _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } \le \gamma ( P ) \displaystyle \operatorname* { m a x } _ { Q \in \mathcal P } \| u \| _ { \mathcal H _ { Q } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } } & { } \\ { \le \gamma ( P ) g _ { n } ( \mathcal P ) \left\| u \right\| _ { \mathcal H _ { n } } ^ { 2 } } & { } \\ { \le \gamma ( P ) g _ { n } ( \mathcal P ) \beta _ { n } ^ { 2 } ( \delta ) . } \end{array}
$$

Moreover, $v = \Pi _ { U ^ { - } }$ ⊥ v implies

$$
\begin{array} { r l } & { \| v \| _ { \mathcal H _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } = v ^ { \top } \Pi _ { U ^ { \perp } } \mathcal H _ { P } ( \theta ^ { \mathrm { E } } ) \Pi _ { U ^ { \perp } } v } \\ & { \qquad \leq g _ { \perp } ( P ) \| v \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 4 g _ { \perp } ( P ) B _ { \theta } ^ { 2 } . } \end{array}
$$

Consequently,

$$
\begin{array} { r l } & { \| \Delta \| _ { \mathcal H _ { P } ( { \boldsymbol { \theta } } ^ { \mathrm { E } } ) } ^ { 2 } \leq 2 \| \boldsymbol { u } \| _ { \mathcal H _ { P } ( { \boldsymbol { \theta } } ^ { \mathrm { E } } ) } ^ { 2 } + 2 \| \boldsymbol { v } \| _ { \mathcal H _ { P } ( { \boldsymbol { \theta } } ^ { \mathrm { E } } ) } ^ { 2 } } \\ & { \qquad \leq 2 \gamma ( P ) g _ { n } ( \mathcal P ) \beta _ { n } ^ { 2 } ( \delta ) + 8 g _ { \perp } ( P ) B _ { \boldsymbol { \theta } } ^ { 2 } . } \end{array}
$$

Furthermore, $v = \Pi _ { U ^ { \perp } } v$ and the definition of $s _ { P }$ yield

$$
\begin{array} { r l } & { R \left\| \Pi _ { P } \Delta \right\| _ { 2 } \leq R \left\| u \right\| _ { 2 } + R \left\| \Pi _ { P } \Pi _ { U ^ { \perp } } v \right\| _ { 2 } } \\ & { \qquad \leq R \left\| u \right\| _ { 2 } + R s _ { P } \left\| v \right\| _ { 2 } } \\ & { \qquad \leq 1 + 2 R B _ { \theta } s _ { P } . } \end{array}
$$

Applying Proposition C.1 and item 2 of Lemma B.7 with $\theta _ { 1 } = \theta _ { n }$ and $\theta _ { 2 } = \theta ^ { \mathrm { E } }$ , the monotonicity of $\psi _ { : }$ , and $\psi ( 1 + a ) \leq e \psi ( a )$ for $a \geq 0$ , which follows from $\textstyle \psi ( x ) = \int _ { 0 } ^ { 1 } t e ^ { x t } d t$ , yields

$$
\begin{array} { r l } & { \mathrm { S u b O p t } ( \theta _ { n } , P ) \leq \psi ( R \left\| \Pi _ { P } \Delta \right\| _ { 2 } ) \left\| \Delta \right\| _ { \mathcal H _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } } \\ & { \qquad \leq \psi ( 1 + 2 R B _ { \theta } s _ { P } ) \left\| \Delta \right\| _ { \mathcal H _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } } \\ & { \qquad \leq 2 e \psi ( 2 R B _ { \theta } s _ { P } ) \Big [ \gamma ( P ) g _ { n } ( \mathcal P ) \beta _ { n } ^ { 2 } ( \delta ) + 4 g _ { \perp } ( P ) B _ { \theta } ^ { 2 } \Big ] . } \end{array}\tag{34}
$$

Since $m \leq \eta d _ { \star } H ^ { 2 }$ and $n \geq 2 n _ { \delta }$ , Theorem 3.6 and the definition of $\beta _ { n } ( \delta )$ give

$$
g _ { n } ( \mathcal { P } ) \beta _ { n } ^ { 2 } ( \delta ) \lesssim \frac { d _ { \star } d _ { U } \log ( n m ) \log ( \eta d _ { \star } n m / \delta ) } { n m } ,
$$

which, plugged into (34), yields the claim. The confidence event does not depend on $P ,$ so the bound holds simultaneously for all $P .$

Finally, $U _ { P } \subseteq U$ implies $s _ { P } = g _ { \perp } ( P ) = 0$ , so the transfer bias vanishes, and $\psi ( 0 ) = 1 / 2$ . For $P \in \mathcal { P }$ , the definition of $\gamma ( P )$ directly $\mathrm { g i v e s } \ \gamma ( P ) \leq 1$ , and we recover Theorem 3.6. □

Remark D.2 (Connection to principal angles). Suppose dim $U _ { P } \leq$ dim $U$ and denote $\alpha _ { P } : = \alpha _ { \mathrm { m a x } } ( U _ { P } , U )$ for the maximal principal angle between $U _ { P }$ and U. Then

$$
s _ { P } = \| \Pi _ { P } \Pi _ { U ^ { \perp } } \| _ { 2 } = \sin \alpha _ { P } ,
$$

so the constant in (34) reads $\psi ( 2 R B _ { \theta }$ sin $\alpha _ { P } )$ . In particular, since $\psi ( 1 ) = 1$ , it is at most one whenever $2 R B _ { \theta }$ sin $\alpha _ { P } \leq 1$ . If instead dim $U _ { P } >$ dim $U$ , then $U _ { P } \cap U ^ { \perp } \ne \{ 0 \}$ and $s _ { P } = 1$

Moreover, since $\mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) ^ { 1 / 2 } = \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) ^ { 1 / 2 } \Pi _ { P }$

$$
\begin{array} { r l } & { g _ { \perp } ( P ) = \left\| \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) ^ { 1 / 2 } \Pi _ { P } \Pi _ { U ^ { \perp } } \right\| _ { 2 } ^ { 2 } } \\ & { ~ \leq \left\| \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \right\| _ { 2 } \sin ^ { 2 } \alpha _ { P } . } \end{array}
$$

Both the constant and the transfer bias are thus controlled by $\alpha _ { P }$ . For a single training dynamics $P ^ { 0 }$ with dim $U _ { P ^ { 0 } } = \dim U _ { P } $ we have

$$
\alpha _ { P } = \alpha _ { \mathrm { m a x } } ( U _ { P ^ { 0 } } ^ { \perp } , U _ { P } ^ { \perp } ) .
$$

This is the angle between unidentifiable subspaces appearing in the transferability guarantee of Schlaginhaufen and Kamgarpour [6, Theorem 3.11].

## E OFFLINE ACTIVE LEARNING

In some cases, sequential interaction with the expert may be impractical, for instance due to limited expert availability or high environment setup costs. We therefore also consider an ofline variant, where the entire sequence of dynamics is selected in advance. Setting $\theta _ { 0 } : = 0$ and $\varepsilon : = \eta H ^ { 2 } B _ { \phi } ^ { 2 }$ , we construct the ofline information proxy and ofline transfer operator

$$
\mathcal { H } _ { k } ^ { 0 } : = \varepsilon \Pi _ { U } + \sum _ { i = 1 } ^ { k } \mathcal { H } _ { P _ { i } } ( \theta _ { 0 } ) , \qquad \mathcal { T } _ { k } ^ { 0 } ( P ) : = ( \mathcal { H } _ { k } ^ { 0 } ) ^ { \dagger / 2 } \mathcal { H } _ { P } ( \theta _ { 0 } ) ( \mathcal { H } _ { k } ^ { 0 } ) ^ { \dagger / 2 } ,
$$

and select dynamics with

$$
P _ { k + 1 } \in \underset { P \in \mathcal { P } } { \arg \operatorname* { m a x } } \ J \left( \mathcal { T } _ { k } ^ { 0 } ( P ) \right) , \quad \mathrm { w i t h } \quad J \in \left\{ \left\| \cdot \right\| _ { 2 } , \mathrm { t r } , \log \operatorname* { d e t } ( I + \cdot ) \right\} .\tag{35}
$$

Since $U$ and $\mathcal { H } _ { P } ( \theta _ { 0 } )$ do not depend on expert demonstrations, this rule constructs the full sequence $( P _ { k } ) _ { k = 1 } ^ { n }$ without any interaction with the expert.

Theorem E.1. Let Assumption 2.1 hold, assume finite and and $U \neq \{ 0 \}$ , and follow (35) at every round. Let κ be defined in Lemma $C . 4 ,$ and $\begin{array} { r } { \xi : = \operatorname* { m i n } _ { \boldsymbol { v } \in U , \| \boldsymbol { v } \| _ { 2 } = 1 } } \end{array}$ max $_ { P \in \mathcal { P } } \left\| v \right\| _ { \Pi _ { P } } ^ { 2 } > 0$ . Then, for every $n \geq n _ { \delta }$ and $P \in \mathcal { P }$

$$
g _ { n } ( P ) \leq 2 e ^ { 2 R B _ { \theta } } \left( 1 + \frac { m } { \eta d _ { \star } H ^ { 2 } } \right) \frac { d _ { U } \log ( 1 + n ) } { n } ,
$$

and $n _ { \delta } \le \lceil 2 K \log ( 4 K ) \rceil$ whenever the right-hand side is at most $n ,$ where

$$
K : = \frac { 2 d _ { U } e ^ { 2 R B _ { \theta } } \left( 2 R ^ { 2 } \beta _ { n } ^ { 2 } ( \delta ) + \eta H ^ { 2 } B _ { \phi } ^ { 2 } \right) } { \kappa \xi } .
$$

Proof. All matrices below are positive semidefinite and vanish on $U ^ { \perp }$ , so we compare them in the Loewner order on $U ,$ where $\mathcal { H } _ { k } ^ { 0 } \succeq \varepsilon \Pi _ { U }$ is invertible and $( \mathcal { H } _ { k } ^ { 0 } ) ^ { \dag }$ is its inverse. Determinants de $\scriptstyle { \mathsf { , } } _ { U }$ are taken on $U .$ , of dimension $d _ { U }$ . Let $t _ { k } : = \operatorname* { m a x } _ { P \in \mathcal { P } } \left\| \mathcal { T } _ { k } ^ { 0 } ( P ) \right\| _ { 2 }$ and $a : = R B _ { \theta }$

Step 1 (elliptical potential lemma). Since $\mathcal { H } _ { k } ^ { 0 } \succeq \varepsilon \Pi _ { U }$ and $\mathcal { H } _ { P } ( \theta _ { 0 } ) \preceq \eta H ^ { 2 } B _ { \phi } ^ { 2 } \Pi _ { U }$ (Proposition B.4), every eigenvalue of $\mathcal { T } _ { k } ^ { 0 } ( P )$ lies in [0, 1]. As $x \leq 2 \log ( 1 + x )$ on [0, 1], for every $P \in { \mathcal { P } }$ and each criterion $^ { J , }$

$$
\begin{array} { r } { \left\| \mathcal I _ { k } ^ { 0 } ( P ) \right\| _ { 2 } \le 2 \log \operatorname* { d e t } \left( I + \mathcal T _ { k } ^ { 0 } ( P _ { k + 1 } ) \right) . } \end{array}
$$

For $J = \log \operatorname* { d e t } ( I + \cdot )$ , this follows from $\left\| \mathcal { T } _ { k } ^ { 0 } ( P ) \right\| _ { 2 } \leq 2$ log det $( I + \mathcal { T } _ { k } ^ { 0 } ( P ) )$ and the choice of $P _ { k + 1 }$ . For $J = \| \cdot \| _ { 2 }$ and $J = \operatorname { t r } .$ it follows from $\begin{array} { r } { \big \| \mathcal { T } _ { k } ^ { 0 } ( P ) \big \| _ { 2 } \leq J ( \dot { \mathcal { T } } _ { k } ^ { 0 } ( P ) ) \leq \bar { J } ( \mathcal { T } _ { k } ^ { 0 } ( P _ { k + 1 } ) ) } \end{array}$ and the scalar bound applied to each eigenvalue of $\mathcal { T } _ { k } ^ { 0 } ( P _ { k + 1 } )$ . Since $\mathcal { H } _ { k } ^ { 0 }$ is nondecreasing in k, $\left\| \mathcal { T } _ { n } ^ { 0 } ( P ) \right\| _ { 2 } \leq \left\| \mathcal { T } _ { k } ^ { 0 } ( P ) \right\| _ { 2 }$ for $k \leq n$ . Averaging over $k \in \{ 0 , \ldots , n - 1 \}$ and using det<sub>U</sub> $\mathcal { H } _ { k + 1 } ^ { 0 } = \operatorname* { d e t } _ { U } \mathcal { H } _ { k } ^ { 0 }$ det $\left( I + \mathcal { T } _ { k } ^ { 0 } ( P _ { k + 1 } ) \right)$ ,

$$
t _ { n } \ \leq \ { \frac { 2 } { n } } \sum _ { k = 0 } ^ { n - 1 } \log \operatorname* { d e t } \left( I + { \mathcal T } _ { k } ^ { 0 } ( P _ { k + 1 } ) \right) \ = \ { \frac { 2 } { n } } \log \frac { \operatorname* { d e t } _ { U } \mathcal { H } _ { n } ^ { 0 } } { \varepsilon ^ { d _ { U } } } \ \leq \ \frac { 2 d _ { U } \log ( 1 + n ) } { n } ,
$$

where the last step uses that every eigenvalue of $\mathcal { H } _ { n } ^ { 0 }$ on $U$ is at most $\varepsilon + n \eta H ^ { 2 } B _ { \phi } ^ { 2 } = ( 1 + n ) \varepsilon$

Step 2 (from $\theta _ { 0 }$ to $\theta ^ { E } )$ . Since $\left. \theta ^ { \mathrm { E } } \right. _ { \Pi _ { P } } \leq \left. \theta ^ { \mathrm { E } } \right. _ { 2 } \leq B _ { \theta }$ for every $P \in \mathcal { P }$ , item 1 of Lemma B.7 with $\theta _ { 1 } = \theta ^ { \mathrm { E } }$ and $\theta _ { 2 } = \theta _ { 0 } { \mathrm { ~ g i v e s ~ } } e ^ { - a } { \mathcal { H } } _ { P } ( \theta _ { 0 } ) \preceq { \mathcal { H } } _ { P } ( \theta ^ { \mathrm { E } } ) \preceq e ^ { a } { \mathcal { H } } _ { P } ( \theta _ { 0 } )$ . Summing over the selected dynamics,

$$
\mathcal { H } _ { k } ^ { 0 } \ \preceq \varepsilon \Pi _ { U } + e ^ { a } \mathcal { H } _ { k } .\tag{36}
$$

Step 3 (transfer bound). Let $n \geq n _ { \delta }$ , so that $\lambda _ { n } \geq \Lambda$ , with Λ as in the proof of Theorem 3.6, and $\varepsilon \Pi _ { U } \preceq$ $( \varepsilon / \Lambda ) \mathcal { H } _ { n } = c _ { m } \mathcal { H } _ { n }$ . By definition of $t _ { n }$ , Step 2 and (36),

$$
{ \mathcal { H } } _ { P } ( \theta ^ { \mathrm { E } } ) \ \preceq \ e ^ { a } { \mathcal { H } } _ { P } ( \theta _ { 0 } ) \ \preceq \ e ^ { a } t _ { n } { \mathcal { H } } _ { n } ^ { 0 } \ \preceq \ e ^ { a } t _ { n } \left( c _ { m } + e ^ { a } \right) { \mathcal { H } } _ { n } ,
$$

hence $g _ { n } ( P ) = \left\| \mathcal { H } _ { n } ^ { \dagger / 2 } \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \mathcal { H } _ { n } ^ { \dagger / 2 } \right\| _ { 2 } \leq e ^ { a } ( e ^ { a } + c _ { m } ) t _ { n } \leq e ^ { 2 a } ( 1 + c _ { m } ) t _ { n }$ . By (30), $c _ { m } \leq m / ( \eta d _ { \star } H ^ { 2 } )$ , and Step 1 gives the first claim.

Step 4 (ofline warm-up). Let $v \in U$ be a unit vector and $P ^ { \star } \in \arg \operatorname* { m a x } _ { P \in \mathcal { P } } \left\| \Pi _ { P } v \right\| _ { 2 } ^ { 2 }$ , so that $\left\| \Pi _ { P ^ { \star } } v \right\| _ { 2 } ^ { 2 } \geq \xi$ . By Lemma C.4 and Step 2, $\begin{array} { r } { , \mathcal { H } _ { P ^ { \star } } ( \theta _ { 0 } ) \succeq e ^ { - a } \mathcal { H } _ { P ^ { \star } } ( \theta ^ { \mathrm { E } } ) \succeq e ^ { - a } \kappa \Pi _ { P ^ { \star } } } \end{array}$ , and by definition of $\bar { t _ { k } } , \mathcal { H } _ { P ^ { \star } } ( \theta _ { 0 } ) \preceq t _ { k } \mathcal { \bar { H } } _ { k } ^ { 0 }$ . Hence $v ^ { \top } \mathcal { H } _ { k } ^ { 0 } v \geq e ^ { - a } \kappa \xi / t _ { k }$ , and by (36),

$$
\lambda _ { k } \ \ge \ e ^ { - a } \left( \frac { e ^ { - a } \kappa \xi } { t _ { k } } - \varepsilon \right) \ \ge \ \frac { e ^ { - 2 a } \kappa \xi } { t _ { k } } - \varepsilon .
$$

By Step 1, the warm-up condition (4), which reads $\lambda _ { k } \ge 2 R ^ { 2 } \beta _ { n } ^ { 2 } ( \delta )$ , therefore holds as soon as

$$
\frac { k } { \log ( 1 + k ) } \geq \frac { 2 d _ { U } e ^ { 2 a } \left( 2 R ^ { 2 } \beta _ { n } ^ { 2 } ( \delta ) + \varepsilon \right) } { \kappa \xi } = K .
$$

Since $\log ( 1 + k ) \leq \log ( 2 k )$ , it sufices that $k \geq K \log ( 2 k )$ . As κΠ $P \preceq \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \preceq \varepsilon \Pi _ { P }$ gives $\kappa \leq \varepsilon .$ and $\xi \le 1$ , we have $K \geq 2$ , so Lemma C.6 shows that this holds for every $k \geq 2 K \log ( 4 K )$ . Since $n _ { \delta }$ is the first round satisfying the warm-up condition, $n _ { \delta } \le \lceil 2 K \log ( 4 K ) \rceil$ whenever the right-hand side is at most n. □

While looser due to the less precise ofline information $\mathcal { H } _ { k } ^ { 0 }$ , this strategy retains a transfer rate that does not depend on $| { \mathcal { P } } | ,$ , and provides a practical alternative when interaction is limited. The resulting ofline algorithm is given in Appendix H.

## F TRACTABLE WARM-UP LENGTH BOUND

The bound of Lemma 3.3 depends on $\theta ^ { \mathrm { E } }$ through $\lambda _ { | \mathcal { P } | }$ and $d _ { \star }$ . Both can be bounded using only the Hessians at $\theta = 0$ , which only involve the uniform policy and are computed with the projectors (Appendix H).

Corollary F.1. Let $\begin{array} { r } { \bar { \lambda } : = e ^ { - R B _ { \theta } } \lambda _ { \operatorname* { m i n } } ^ { U } \left( \sum _ { P \in \mathcal { P } } \mathcal { H } _ { P } ( 0 ) \right) } \end{array}$ and $\begin{array} { r } { \bar { d } _ { \star } : = d _ { U } e ^ { R B _ { \theta } } H ^ { 2 } B _ { \phi } ^ { 2 } \operatorname* { m a x } _ { P \in \mathcal { P } } \lambda _ { \operatorname* { m i n } } ^ { U _ { P } } \big ( \mathcal { H } _ { P } ( 0 ) \big ) ^ { - 1 } } \end{array}$ , and let $\bar { \beta } _ { n } ^ { 2 } ( \delta )$ be defined as $\beta _ { n } ^ { 2 } ( \delta )$ with $\bar { d } _ { \star }$ in place of $\mid d _ { \star }$ . Under round-robin selection (6),

$$
n _ { \delta } \ \leq \ | \mathcal { P } | \left\lceil \frac { 2 R ^ { 2 } \bar { \beta } _ { n } ^ { 2 } ( \delta ) } { \bar { \lambda } } \right\rceil .
$$

Proof. Step 1 (information after one sweep). Since $\left\| \theta ^ { \mathrm { E } } \right\| _ { \Pi _ { P } } \leq \left\| \theta ^ { \mathrm { E } } \right\| _ { 2 } \leq B _ { \theta }$ for every $P \in \mathcal { P }$ , item 1 of Lemma B.7 with $\theta _ { 1 } = \theta ^ { \mathrm { E } }$ and $\theta _ { 2 } = 0$ gives

$$
\begin{array} { r } { \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) \succeq e ^ { - R B _ { \theta } } \mathcal { H } _ { P } ( 0 ) . } \end{array}
$$

Summing over yields $\lambda _ { | \mathcal { P } | } \geq \bar { \lambda }$

Step 2 (efective dimension). Since $\| \phi ( \tau ) \| _ { 2 } \le H B _ { \phi }$ , we have $\begin{array} { r } { \left\| \Sigma _ { P } \right\| _ { 2 } \leq \mathbb { E } \left\| \Pi _ { P } \phi ( \tau ) \right\| _ { 2 } ^ { 2 } \leq H ^ { 2 } B _ { \phi } ^ { 2 } } \end{array}$ . Moreover, Step 1 gives

$$
\begin{array} { r } { \mathcal { H } _ { P } ( \boldsymbol { \theta } ^ { \mathrm { E } } ) ^ { \dagger } \preceq e ^ { R B _ { \boldsymbol { \theta } } } \lambda _ { \operatorname* { m i n } } ^ { U _ { P } } \left( \mathcal { H } _ { P } ( \boldsymbol { 0 } ) \right) ^ { - 1 } \Pi _ { P } , } \end{array}
$$

so that $d _ { \star } \leq \bar { d } _ { \star }$ . Since the threshold (4) is increasing in $d _ { \star } .$ , it is at most $2 R ^ { 2 } \hat { \beta } _ { n } ^ { 2 } ( \delta )$

Step 3 (warm-up length). Under round-robin selection, after k rounds every $P \in \mathcal { P }$ was selected at least $\lfloor k / \left\lceil \mathcal { P } \right\rceil \rfloor$ times, so that

$$
\lambda _ { k } \geq \lfloor k / \lvert \mathcal { P } \rvert \rfloor \lambda _ { | \mathcal { P } | } \geq \lfloor k / \lvert \mathcal { P } \rvert \rfloor \bar { \lambda } .
$$

By Step 2, the warm-up condition thus holds at round $k = | \mathcal { P } | \left. \left[ 2 R ^ { 2 } \bar { \beta } _ { n } ^ { 2 } ( \delta ) / \bar { \lambda } \right] \right.$ , which proves the claim. □

The factors $e ^ { R B _ { \theta } }$ make this bound loose, which is why we treat the warm-up length as a hyperparameter in practice (Section 4).

## G MISSPECIFIED EXPERTS

So far, we assumed that the expert is soft-optimal for $\theta ^ { \mathrm { E } }$ under the learner’s temperature η. We now only require its behavior to be explained by a nearby reward.

Assumption G.1 (Misspecified expert). There exists $\tilde { \theta ^ { \mathrm { E } } } \in \Theta$ with $R \left\| \tilde { \theta ^ { \mathrm { E } } } - \theta ^ { \mathrm { E } } \right\| _ { \Pi _ { U } } \leq 1$ such that, for every $P \in \mathcal { P }$ , the expert follows $\tilde { \pi } _ { P } ^ { \mathrm { E } } = \pi _ { P } ( \tilde { \theta ^ { \mathrm { E } } } )$ . We denote its suboptimality by

$$
\varepsilon _ { \mathrm { m i s } } : = \operatorname* { m a x } _ { P \in \mathcal { P } } \left( V _ { P } ^ { \star } ( \theta ^ { \mathrm { E } } ) - V _ { P } ^ { \tilde { \pi } _ { P } ^ { \mathrm { E } } } ( \theta ^ { \mathrm { E } } ) \right) .
$$

This covers a human demonstrator who is slightly mistaken about the task’s objective, or whose level of randomness difers from the one the learner assumes. The latter simply amounts to a rescaled reward: an expert acting with an inverse temperature $\eta ^ { \mathrm { E } } \neq \eta$ follows $\pi _ { P } ( \tilde { \theta ^ { \mathrm { E } } } )$ with $\tilde { \theta ^ { \mathrm { E } } } = \left( \eta ^ { \mathrm { E } } / \eta \right) \theta ^ { \mathrm { E } }$ , since soft-optimal policies depend on the reward only through ηθ, provided $\left( \eta ^ { \mathrm { E } } / \eta \right) \left\| \dot { \theta ^ { \mathrm { E } } } \right\| _ { 2 } \leq B _ { \theta }$ . Its suboptimality then grows quadratically with the mismatch,

$$
\varepsilon _ { \mathrm { m i s } } \leq ( \eta ^ { \mathrm { E } } / \eta - 1 ) ^ { 2 } \operatorname* { m a x } _ { P \in \mathcal { P } } \left\| \theta ^ { \mathrm { E } } \right\| _ { \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } .
$$

Since the demonstrations are well-specified for $\tilde { \theta ^ { \mathrm { E } } }$ , all our results apply to $\tilde { \theta ^ { \mathrm { E } } }$ . Transferring to $\theta ^ { \mathrm { E } }$ then only adds the expert’s own suboptimality.

Lemma G.2. Let Assumption G.1 hold, and let $d _ { \star }$ and $n _ { \delta }$ be defined with respect to the actual expert, $i . e . ,$ , at $\tilde { \theta ^ { E } }$ . With probability at least $1 - \delta$ , it holds for all $P \in \mathcal { P }$ and $n _ { \delta } \le k \le n$ that

$$
\mathrm { S u b O p t } ( \theta _ { k } , P ) \ \lesssim \ g _ { k } ( P ) \beta _ { k } ^ { 2 } ( \delta ) \ + \ \varepsilon _ { m i s } .
$$

Proof. Let $w : = \tilde { \theta ^ { \mathrm { E } } } - \theta ^ { \mathrm { E } }$ denote the misspecification and $u _ { k } : = \theta _ { k } - \tilde { \theta ^ { \mathrm { E } } }$ the estimation error, so that $\theta _ { k } - \theta ^ { \mathrm { E } } =$ $u _ { k } + w$ , and write $\tilde { \mathcal { H } } _ { P } : = \mathcal { H } _ { P } ( \boldsymbol { \theta } ^ { \mathrm { E } } )$

Step 1 (comparison of curvatures). Since $U _ { P } \subseteq U$ , we have $R \| w \| _ { \Pi _ { P } } \le R \| w \| _ { \Pi _ { U } } \le 1$ for every $P \in { \mathcal { P } } .$ , and item 1 of Lemma B.7 gives

$$
e ^ { - 1 } \mathcal { H } _ { P } ( { \boldsymbol { \theta } } ^ { \mathrm { E } } ) \preceq \tilde { \mathcal { H } } _ { P } \preceq e \mathcal { H } _ { P } ( { \boldsymbol { \theta } } ^ { \mathrm { E } } ) .
$$

Hence, the information gain evaluated at $\tilde { \theta ^ { \mathrm { E } } }$ is within a factor $e ^ { 2 }$ of $g _ { k } ( P )$ , by the argument of Step 1 in the proof of Theorem 3.6.

Step 2 (estimation error). Since the demonstrations are generated by $\pi _ { P } ( \tilde { \theta ^ { \mathrm { E } } } )$ with $\tilde { \theta ^ { \mathrm { E } } } \in \Theta$ , Lemma 3.1 applies with $\tilde { \theta ^ { \mathrm { E } } }$ in place of $\theta ^ { \mathrm { E } }$ for all $n _ { \delta } \le k \le n$ . On its event, the proof of Lemma 3.2 and Step 1 yield

$$
\| u _ { k } \| _ { \widetilde { \mathcal { H } } _ { P } } ^ { 2 } \le e ^ { 2 } g _ { k } ( P ) \beta _ { k } ^ { 2 } ( \delta ) \qquad \mathrm { ~ a n d ~ } \qquad R \| u _ { k } \| _ { \Pi _ { P } } \le 1 ,
$$

hence $R \left. u _ { k } + w \right. _ { \Pi _ { P } } \leq 2$ for all $n _ { \delta } \le k \le n$

Step 3 (decomposition). By Proposition C.1 and item 2 of Lemma B.7,

$$
\mathrm { S u b O p t } ( \theta _ { k } , P ) \le \psi ( 2 ) \left. u _ { k } + w \right. _ { \mathcal { H } _ { P } ( \theta ^ { \mathtt { E } } ) } ^ { 2 } \le 2 \psi ( 2 ) \left( \left. u _ { k } \right. _ { \mathcal { H } _ { P } ( \theta ^ { \mathtt { E } } ) } ^ { 2 } + \left. w \right. _ { \mathcal { H } _ { P } ( \theta ^ { \mathtt { E } } ) } ^ { 2 } \right) .
$$

By Steps 1 and 2, the first term satisfies $\begin{array} { r } { \| u _ { k } \| _ { \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } \leq e \| u _ { k } \| _ { \tilde { \mathcal { H } } _ { P } } ^ { 2 } \lesssim g _ { k } ( P ) \beta _ { k } ^ { 2 } ( \delta ) } \end{array}$ . For the second, Proposition C.1 and item 2 of Lemma B.7 with $\theta _ { 1 } = \widetilde { \theta ^ { \mathrm { E } } }$ and $\theta _ { 2 } = \theta ^ { \mathrm { E } } { \mathrm { ~ g i v e } }$ , since $R \| w \| _ { \Pi _ { P } } \leq 1$

$$
\varepsilon _ { \mathrm { m i s } } \geq V _ { P } ^ { \star } ( \theta ^ { \mathrm { E } } ) - V _ { P } ^ { \tilde { \pi } _ { P } ^ { \mathrm { E } } } ( \theta ^ { \mathrm { E } } ) = \left\| w \right\| _ { G _ { P } ( \theta ^ { \mathrm { E } } , \theta ^ { \mathrm { E } } ) } ^ { 2 } \geq \psi ( - 1 ) \left\| w \right\| _ { \mathcal { H } _ { P } ( \theta ^ { \mathrm { E } } ) } ^ { 2 } ,
$$

which concludes the proof.

Lemma G.2 is Lemma 3.2 plus a bias. Since $\theta _ { k }$ remains within distance $2 / R$ of $\theta ^ { \mathrm { E } }$ along every $U _ { P }$ after the warm-up (Step 2), Theorem 3.6 carries over with larger constants and the same bias. In short, the learned reward transfers as well as the expert itself, up to the usual rate. No amount of demonstrations removes this bias.

The same holds for Theorem 3.7, for any target dynamics $P ,$

$$
\mathrm { S u b O p t } ( \theta _ { n } , P ) = \tilde { \mathcal { O } } \left( \frac { d _ { \star } d _ { U } } { n m } \right) + \mathcal { O } \left( \varepsilon _ { \mathrm { m i s } } + g _ { \perp } ( P ) B _ { \theta } ^ { 2 } \right) ,
$$

since the misspecification only enters the error along $U _ { \textrm { \scriptsize i } }$ , which the proof of Theorem 3.7 bounds through the dynamics in , while the error along $U ^ { \perp }$ is bounded as before.

Remark G.3 (Unregularized experts). Deterministic experts $( \eta  \infty )$ lie outside our framework, as entropy regularization is what makes the reward identifiable along each $U _ { P } \ [ 4 , \ 6 ]$ . They can be approximated with a large $\eta ,$ at the cost of constants growing with $\eta ,$ such as $R .$

## H ALGORITHMS

Hessian estimation. Algorithm 2 estimates the information $\mathcal { H } _ { P } ( \theta )$ of a dynamics P from trajectories of its optimal policy $\pi _ { P } ( \theta )$ , using the feature advantages of Lemma B.1, which require one policy evaluation of the features. Under deterministic transitions and initial state, the advantages reduce to centered trajectory features (Corollary B.2), so no policy evaluation is needed. Under stochastic dynamics, the advantages cannot be replaced by trajectory features, whose covariance also contains the dynamics noise $\Sigma _ { P } ^ { \mathrm { d y n } }$ (Lemma B.3), which may lie outside $U _ { P }$

Projectors. Since $U _ { P }$ does not depend on θ (Proposition B.5), each projector is computed once, before any demonstration, from the information at $\theta = 0$ , whose optimal policy $\pi _ { P } ( 0 )$ is uniform. We run Algorithm 2 with $\pi _ { P } ( 0 )$ to obtain $\hat { \mathcal { H } } _ { P } ( 0 )$ , compute its eigendecomposition $\begin{array} { r } { \hat { \mathcal { H } } _ { P } ( 0 ) = \sum _ { i = 1 } ^ { d } \hat { \lambda } _ { i } \hat { u } _ { i } \hat { u } _ { i } ^ { \top } } \end{array}$ , and set

$$
\hat { \Pi } _ { P } : = \sum _ { i : \hat { \lambda } _ { i } > \varepsilon _ { \lambda } } \hat { u } _ { i } \hat { u } _ { i } ^ { \top } .
$$

With exact Hessians, this recovers $\Pi _ { P } = \mathcal { H } _ { P } ( 0 ) \mathcal { H } _ { P } ( 0 ) ^ { \dagger }$ for any ε below the smallest positive eigenvalue of $\mathcal { H } _ { P } ( 0 )$ The threshold $\varepsilon > 0$ discards the small eigenvalues that sampling noise adds along unidentifiable directions. For our experiments, we fixed $\varepsilon _ { \lambda } = 5 \cdot 1 0 ^ { - 4 }$

Two-phase active IRL. Algorithm 4 details our two-phase approach. After computing the projectors, it selects dynamics round-robin (6) during the first $n _ { 0 }$ rounds, observing m expert trajectories in each, and then computes $\theta _ { n _ { 0 } }$ with Algorithm 3. At every subsequent round $k ,$ it

1. estimates $\mathcal { H } _ { P } ( \theta _ { k - 1 } )$ for each $P \in \mathcal { P }$ with Algorithm 2, using the policies $\pi _ { P } ( \theta _ { k - 1 } )$ returned by Algorithm 3;

2. selects $P _ { k }$ by (7) and observes m expert trajectories in $P _ { k }$ ;

3. updates $\theta _ { k }$ with Algorithm 3 on all data collected so far.

Ofline active IRL. Algorithm 5 selects the whole sequence $\left( P _ { k } \right) _ { k = 1 } ^ { n }$ before observing the expert, using $\theta _ { 0 } = 0$ in place of $\theta _ { k - 1 }$ and no warm-up. The required Hessians $\mathcal { H } _ { P } ( 0 )$ are those already computed for the projectors. The expert is then observed on the selected dynamics, and the reward is estimated once with Algorithm 3.

Computational complexity. The main cost in our setting is the number of expert demonstrations, which our analysis minimizes. All other computations only require access to the dynamics in ${ \mathcal { P } } ,$ and we do not optimize them further. For completeness:

1. Projectors $( o n c e ) \colon | { \mathcal { P } } |$ runs of Algorithm 2, each with one policy evaluation of d-dimensional features and one rollout batch, followed by an eigendecomposition of a d  d matrix.

2. Information estimation (per active round): runs of Algorithm 2 with the current policies, which Algorithm 3 already returns, so no additional RL solve is needed.

3. Dynamics selection (per active round): one inverse square root of a d $\times \ d$ matrix and $| \mathcal { P } |$ evaluations of $J ,$ i.e., $\mathcal { O } ( | \mathcal { P } | \ d ^ { 3 } )$ operations.

```latex
Algorithm 2 Hessian estimation
1: Input: dynamics $P ,$ policy $\pi = \pi _ { P } ( \theta )$ , rollouts $N$
2: Set $\bar { \Phi } _ { H + 1 } : = 0$
3: for $h = H , \ldots , 1$ do
4: for $a \in \mathcal { A } , s \in \mathcal { S }$ do
5: $\begin{array} { r } { \underline { { \Phi } } _ { h } ( s , a )  \phi ( s , a ) + \sum _ { s ^ { \prime } } T ( s ^ { \prime } \mid s , a ) \bar { \Phi } _ { h + 1 } ( s ^ { \prime } ) } \end{array}$
6: $\begin{array} { r } { \bar { \Phi } _ { h } ( s ) \gets \sum _ { a } \pi _ { h } ( a \mid s ) \bar { \Phi _ { h } } ( s , a ) } \end{array}$
7: Sample $\tau _ { 1 } , . . . , \tau _ { N }$ with π in $P$
8: $\begin{array} { r } { A ( \tau _ { i } ) \gets \sum _ { h = 1 } ^ { H } \left( \Phi _ { h } ( s _ { h } , a _ { h } ) - \bar { \Phi } _ { h } ( s _ { h } ) \right) } \end{array}$ (Feature advantage)
9: return: $\begin{array} { r } { \hat { \mathcal { H } } _ { P } ( \boldsymbol { \theta } ) \gets \frac { \eta } { N } \sum _ { i = 1 } ^ { N } A ( \tau _ { i } ) A ( \tau _ { i } ) ^ { \top } } \end{array}$ (Lemma B.1)
```

Algorithm 3 Projected min-max IRL with multiple dynamics   
1: Input: expert data $\left\{ ( P _ { i } , \hat { \phi } _ { i } ) \right\} _ { i = 1 } ^ { k } ,$ , projectors $\left\{ \hat { \Pi } _ { P } \right\} _ { P \in \mathcal { P } }$ , step size $\alpha ,$ iterations $K ,$ rollouts N   
2: Initialize: $\theta ^ { ( 0 ) }  0$   
3: for $t = 0 , \ldots , K - 1$ do   
4: for each distinct $P \in \{ P _ { 1 } , \ldots , P _ { k } \}$ do   
5: $\pi  \pi _ { P } ( \theta ^ { ( t ) } )$ (RL)   
6: $\begin{array} { r } { \bar { \phi } _ { P } ^ { t }  \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \phi ( \tau _ { j } ) } \end{array}$ , with $\tau _ { j } \sim \pi$ in $P$ (Rollouts)   
7: $\begin{array} { r } { g _ { t }  \sum _ { i = 1 } ^ { k } \hat { \Pi } _ { P _ { i } } ( \bar { \phi } _ { P _ { i } } ^ { t } - \hat { \phi } _ { i } ) } \end{array}$ (Gradient of (2))   
8: $\theta ^ { ( t + 1 ) } \gets \Pi _ { \Theta } \left( \theta ^ { ( t ) } - \alpha g _ { t } \right)$   
9: $\begin{array} { r } { \theta _ { k }  \frac { 1 } { K } \sum _ { t = 0 } ^ { K - 1 } \dot { \theta } ^ { ( t ) } } \end{array}$   
10: return: $\theta _ { k } , \{ \pi _ { P } ( \theta _ { k } ) \} _ { P \in \mathcal { P } }$

Algorithm 4 Active IRL   
1: Input: , budget $n ,$ warm-up length $n _ { 0 } \geq 0 ,$ trajectories per round $m ,$ regularization $\varepsilon > 0 ,$ , criterion J   
2: Initialize: $\mathcal { D } _ { 0 } ^ { \mathrm { E } } \bar {  } \emptyset , k ( P )  \bar { 0 }$ for all $P \in \mathcal { P } , \theta _ { 0 } \gets 0$   
3: Compute $\hat { \mathcal { H } } _ { P } ( 0 )$ and ${ \hat { \Pi } } _ { P }$ for all $P \in \mathcal { P }$ from $\pi _ { P } ( 0 )$ (Algorithm 2)   
4: for $k = 1 , \dots , n$ do   
5: if $k \leq n _ { 0 }$ then (Warm-up)   
6: $P _ { k } \in \arg \operatorname* { m i n } _ { P \in \mathcal { P } } k ( P )$   
7: else (Active selection)   
8: Estimate $\hat { \mathcal { H } } _ { P } ( \theta _ { k - 1 } )$ for all $P \in \mathcal { P }$ from $\pi _ { P } ( \theta _ { k - 1 } )$ (Algorithm 2)   
9: $\begin{array} { r } { \hat { \mathcal { H } } _ { k - 1 }  \varepsilon \hat { \Pi } _ { U } + \sum _ { P \in \mathcal { P } } k ( P ) \hat { \mathcal { H } } _ { P } ( \theta _ { k - 1 } ) } \end{array}$   
10: $\begin{array} { r } { P _ { k } \in \arg \operatorname* { m a x } _ { P \in \mathcal { P } } J \left( \hat { \mathcal { H } } _ { k - 1 } ^ { \dagger / 2 } \hat { \mathcal { H } } _ { P } ( \theta _ { k - 1 } ) \hat { \mathcal { H } } _ { k - 1 } ^ { \dagger / 2 } \right) } \end{array}$   
11: $k ( P _ { k } )  k ( P _ { k } ) + 1$   
12: $\mathcal { D } _ { k } ^ { \mathrm { E } }  \mathcal { D } _ { k - 1 } ^ { \mathrm { E } } \cup \{ \tau _ { k , j } \} _ { j = 1 } ^ { m }$ , with $\tau _ { k , j }$ generated by $\pi _ { P _ { k } } ^ { \mathrm { E } }$ under $P _ { k }$ (Observe expert)   
13: if $k \geq n _ { 0 }$ then   
14: $\theta _ { k } , \{ \pi _ { P } ( \theta _ { k } ) \} _ { P \in \mathcal { P } }  \mathrm { I R L } ( \mathcal { D } _ { k } ^ { \mathrm { E } } , \{ \hat { \Pi } _ { P } \} _ { P \in \mathcal { P } } )$ (Algorithm 3)   
15: return: $\theta _ { n } , \{ \pi _ { P } ( \theta _ { n } ) \} _ { P \in \mathcal { P } }$

4. Reward estimation (per round): K gradient steps of Algorithm 3, each requiring one RL solve and one rollout batch per distinct dynamics selected so far, i.e., at most min $\mathbf { \rho } _ { 1 } ( k , | \mathcal { P } | )$ , plus $| \mathcal { P } |$ final solves for the returned policies. In our finite-horizon tabular setting, each RL solve is a single backward pass of soft value iteration [3], i.e., $\mathcal { O } ( H \left| S \right| ^ { 2 } \left| A \right| )$ operations. Alternatively, each solve can be replaced by a single soft policy iteration step per reward update, which still converges [35] and avoids nested RL loops in larger problems.

Memory is dominated by the  Hessians and projectors, $\mathrm { i . e . , } \mathcal { O } ( | \mathcal { P } | d ^ { 2 } )$ . In comparison, the environment design approach of Buening et al. [10] requires $\mathcal { O } ( k \lvert \mathcal { P } \rvert )$ RL solves at round k for selection alone.

Algorithm 5 Ofline Active IRL   
1: Input: ${ \mathcal { P } } ,$ budget $n ,$ trajectories per round $m ,$ regularization $\varepsilon > 0 ,$ , criterion J   
2: Compute $\hat { \mathcal { H } } _ { P } ( 0 )$ and ${ \hat { \Pi } } _ { P }$ for all $P \in \mathcal { P }$ from $\pi _ { P } ( 0 )$ (Algorithm 2)   
3: $\mathcal { H } _ { 0 } ^ { 0 }  \varepsilon _ { 0 } \Pi _ { U } , \mathrm { w i t h } \varepsilon : = \eta H ^ { 2 } B _ { \phi } ^ { 2 }$   
4: for $k = 1 , \dots , n$ do   
5: $\begin{array} { r } { P _ { k } \in \arg \operatorname* { m a x } _ { P \in \mathcal { P } } J \left( ( \mathcal { H } _ { k - 1 } ^ { 0 } ) ^ { \dagger / 2 } \hat { \mathcal { H } } _ { P } ( 0 ) ( \mathcal { H } _ { k - 1 } ^ { 0 } ) ^ { \dagger / 2 } \right) } \end{array}$ (Ofline active selection (35))   
6: $\mathcal { H } _ { k } ^ { 0 }  \mathcal { H } _ { k - 1 } ^ { 0 } + \hat { \mathcal { H } } _ { P _ { k } } ( 0 )$   
7: $\begin{array} { r } { \mathcal { D } ^ { \mathrm { E } }  \bigcup _ { k = 1 } ^ { n } \{ \tau _ { k , j } \} _ { j = 1 } ^ { m } } \end{array}$ , with $\tau _ { k , j }$ generated by $\pi _ { P _ { k } } ^ { \mathrm { E } }$ under $P _ { k }$ (Observe expert)   
8: $\theta _ { n } , \{ \pi _ { P } ( \theta _ { n } ) \} _ { P \in \mathcal { P } }  \mathrm { I R L } ( \mathcal { D } ^ { \mathrm { E } } , \{ \hat { \Pi } _ { P } \} _ { P \in \mathcal { P } } )$ (Algorithm 3)   
9: return: $\theta _ { n } , \{ \pi _ { P } ( \theta _ { n } ) \} _ { P \in \mathcal { P } }$

## I ADDITIONAL EXPERIMENTAL DETAILS AND RESULTS

## I.1 Common Experimental Details

All expert trajectories have length H = 100. We set $\eta = 2 0$ and use $B _ { \phi } = B _ { \theta } = 1$ throughout all experiments. Policies corresponding to every combination of dynamics and reward functions are computed using soft policy iteration.

In all experiments, the feature map $\phi$ is chosen as the indicator function over state–action pairs. Consequently, the reward function takes the form

$$
r _ { \theta } ( s , a ) = \sum _ { s ^ { \prime } , a ^ { \prime } } 1 \{ s ^ { \prime } = s , a ^ { \prime } = a \} \theta _ { s ^ { \prime } , a ^ { \prime } } .
$$

Reported suboptimality is evaluated as the worst-case value over the considered set of dynamics ${ \mathcal { P } } ,$ $\begin{array} { r } { \operatorname* { m a x } _ { P \in \mathcal { P } } \mathrm { S u b O p t } ( \theta _ { n } , P ) } \end{array}$

Implementation details. The implementation is publicly available at https://anonymous.4open.science/r/transferable-active-irl.

Compute resources. All experiments were conducted on a single AMD EPYC 7302P machine.

## I.2 Controlled sets of dynamics

Principal angles. Here we provide the details on principal angles, used for the first experimental setup. Intuitively, principal angles [23] can be seen as a natural generalization of angles between two lines or planes to higher dimensional subspaces. Formally we define them as follows.

Definition I.1 (Principal angles [23, 6]). Let $\nu , \mathcal { W } \subseteq \mathbb { R } ^ { d }$ be two subspaces. The principal angles $0 \leq \alpha _ { 1 } ( \nu , \mathcal { W } ) \leq$ $\begin{array} { r } { \dots \le \alpha _ { m } ( \mathcal { V } , \mathcal { W } ) : = \alpha _ { \operatorname* { m a x } } ( \mathcal { V } , \mathcal { W } ) \le \frac { \pi } { 2 } } \end{array}$ between and are defined recursively by

$$
\begin{array} { l } { { \displaystyle \cos \left( \alpha _ { i } ( \mathcal { V } , \mathcal { W } ) \right) = \operatorname* { m a x } _ { v \in \mathcal { V } , ~ w \in \mathcal { W } } \left. v , w \right. } } \\ { { \mathrm { s . t . } \quad \| v \| _ { 2 } = \| w \| _ { 2 } = 1 , } } \\ { { \displaystyle \left. v , v _ { j } \right. = 0 , \quad \left. w , w _ { j } \right. = 0 , \quad j = 1 , \ldots , i - 1 , } } \end{array}
$$

where $v _ { j } \in \mathcal V$ and $w _ { j } \in \mathcal W$ are maximizers corresponding to $\alpha _ { j } ( \gamma , \mathcal { W } )$ .

In the context of IRL, our work builds on the geometric perspective introduced by Schlaginhaufen and Kamgarpour [6]. In particular, their Theorems 3.10 and 3.11 characterize reward transferability through the relative geometry of the unidentifiable subspaces associated with diferent transition dynamics, quantified via principa angles. More precisely, Theorem 3.10 shows that when the second principal angle between two unidentifiable subspaces $U _ { P } ^ { \bot }$ and $U _ { P ^ { \prime } } ^ { \perp }$ is suficiently large, observing the expert on both dynamics enables the recovery of re wards that transfer to arbitrary dynamics. Conversely, Theorem 3.11 establishes a complementary regime: when the largest principal angle between $U _ { P }$ and $U _ { P ^ { \prime } }$ is suficiently small, a reward learned from demonstrations in one dynamics transfers reliably to the other. Taken together, these results reveal a geometric trade-of. Large diversity between environments, captured by a large second principal angle, improves global identifiability and enables broad transfer. In contrast, strong similarity between dynamics, captured by a small maximal principal angle, guarantees local transferability of learned rewards across environments.

Let $\alpha _ { i } ( P , P ^ { \prime } )$ denote the i-th principal angle between the subspaces $U _ { P } ^ { \bot }$ and $U _ { P ^ { \prime } } ^ { \bot }$ , and let $\bar { \alpha } _ { i } ( P ) =$ $\begin{array} { r } { \frac { 1 } { | \mathcal { P } | } \sum _ { P ^ { \prime } \in \mathcal { P } } \alpha _ { i } ( P , P ^ { \prime } ) } \end{array}$ be the average i-th principal angle of P over the set $\mathcal { P } .$

Details. In our experiments, we constructed four classes of dynamics sets, each characterized by the geometry of their unidentifiable subspaces through principal-angle statistics. For a set of dynamics $\mathcal { P } = \{ P _ { 1 } , \ldots , P _ { | \mathcal { P } | } \}$ , we define

$$
\mu _ { i } ( \mathcal { P } ) : = \frac { 1 } { | \mathcal { P } | } \sum _ { P \in \mathcal { P } } \bar { \alpha } _ { i } ( P ) , \qquad \sigma _ { i } ^ { 2 } ( \mathcal { P } ) : = \frac { 1 } { | \mathcal { P } | } \sum _ { P \in \mathcal { P } } \big ( \bar { \alpha } _ { i } ( P ) - \mu _ { i } ( \mathcal { P } ) \big ) ^ { 2 } .
$$

All training dynamics are deterministic: every state-action pair $( s , a )$ maps to a single next state. For each seed, a set $\mathcal { P }$ of $| \mathcal { P } | = 1 0$ such dynamics is obtained by an evolutionary search over candidate sets, driven by a fitness defined on principal-angle statistics. A class is specified by a target angle rank $i \in \{ 2 , \mathrm { m a x } \}$ , a target angle $\alpha ^ { \star }$ and a heterogeneity weight $\lambda \geq 0$ . For a candidate set $\mathcal { P }$ , let

$$
m _ { \operatorname* { m i n } } ( \mathcal { P } ) : = \frac { 1 } { | \mathcal { P } | } \sum _ { P \in \mathcal { P } } \operatorname* { m i n } _ { P ^ { \prime } \neq P } \alpha _ { i } ( P , P ^ { \prime } ) , \qquad m _ { \operatorname* { m a x } } ( \mathcal { P } ) : = \frac { 1 } { | \mathcal { P } | } \sum _ { P \in \mathcal { P } } \operatorname* { m a x } _ { P ^ { \prime } \neq P } \alpha _ { i } ( P , P ^ { \prime } ) ,
$$

and let $\begin{array} { r } { h ( \mathcal { P } ) : = \sqrt { \pi / 2 } - \frac { 1 } { | \mathcal { P } | - 1 } \sum _ { j = 1 } ^ { | \mathcal { P } | - 1 } \sqrt { \bar { \alpha } _ { i , ( j + 1 ) } - \bar { \alpha } _ { i , ( j ) } } } \end{array}$ , where $\bar { \alpha } _ { i , ( 1 ) } \leq \cdot \cdot \cdot \leq \bar { \alpha } _ { i , ( | \mathcal { P } | ) }$ are the sorted average angles $\bar { \alpha } _ { i } ( P ) , P \in \mathcal { P }$ . The fitness to minimize is

$$
f ( \mathcal P ) : = \operatorname* { m a x } \Big \{ \big ( m _ { \mathrm { m i n } } ( \mathcal P ) - \alpha ^ { \star } \big ) ^ { 2 } , \ \big ( m _ { \mathrm { m a x } } ( \mathcal P ) - \alpha ^ { \star } \big ) ^ { 2 } \Big \} + \lambda h ( \mathcal P ) .
$$

The first term drives both the closest and the farthest pair of every dynamics toward the target angle, so that transferability is uniform across pairs; the second term, when active, rewards large gaps between the sorted average angles, so that transferability is heterogeneous across the set. The four classes correspond to

• Equivalent: $i = \operatorname* { m a x } , \alpha ^ { \star } = 0 , \lambda = 0 ;$

• Similar: $i = 2 , \alpha ^ { \star } = 0 , \lambda = 0 ;$

• Uniformly diverse: $i = 2 , \alpha ^ { \star } = \pi / 4 , \lambda = 0 ;$

• Arbitrarily diverse: $i = 2 , \alpha ^ { \star } = \pi / 3 , \lambda = 1 0 ^ { 4 }$

The search maintains a population of 300 candidate sets, initialized as copies of a set whose $| \mathcal { P } |$ dynamics are all equal to one random deterministic dynamics (next states drawn uniformly at random for every $( s , a ) )$ . At every generation, the 30 fittest candidates survive and 270 ofspring are generated, each a copy of a uniformly drawn survivor in which every transition is independently redirected to the initial state with probability $1 / ( | S | | A | )$ when $i = 2 ,$ , each dynamics of the ofspring is additionally replaced by a fresh random deterministic dynamics with probability $0 . 1 / \left| \mathcal { P } \right|$ . The search runs for at most 500 generations, stops early once $f ( \mathcal { P } ) < 1 0 ^ { - 5 }$ , and returns the fittest candidate, with dynamics indexed by increasing $\bar { \alpha } _ { i } ( P )$ . Table 2 summarizes the statistics of the resulting dynamics sets, averaged over 20 runs.

For each seed, we evaluate the recovered reward on a set of 1000 held-out stochastic dynamics $\{ \tilde { P } _ { 1 } , \dots , \tilde { P } _ { 1 0 0 0 } \}$ drawn independently of the training set $\mathcal { P } _ { \cdot }$ . For each $\tilde { P } _ { k }$ , a concentration vector $\beta \in \mathbb { R } ^ { | \check { S } | }$ is drawn with log $\beta _ { s ^ { \prime } } \sim$ $\operatorname { U n i f } [ - 2 , 0 . 1 ]$ independently for every next state $s ^ { \prime }$ , and every row $\tilde { P } _ { k } ( \cdot \mid s , a ) , ( s , a ) \in \mathcal { S } \times \mathcal { A }$ , is then sampled independently from Dirichlet(β). Since $\beta _ { s ^ { \prime } } \leq e ^ { 0 . 1 }$ , most rows concentrate their mass on a few next states, so the held-out dynamics are genuinely stochastic yet far from uniform, and none of them coincides with a deterministic training dynamics.

Regarding rewards, for each diferent seed, $\theta ^ { \mathrm { E } }$ is uniformly sampled in $[ 0 , 1 ] ^ { d }$ , and then rescaled such that $\begin{array} { r } { \left. r ^ { \mathrm { E } } \right. _ { 2 } \leq \mathrm { i } } \end{array}$ . To estimate rewards, we run Algorithm 3 for 100 iterations, with $\alpha = 0 . 0 2$

Table 2: Principal-angle statistics (in radian) of the controlled sets of dynamics, averaged over 20 runs.
<table><tr><td>Dynamics set</td><td> $\mu _ { 2 } ( \mathcal { P } ) \pm \sigma _ { 2 } ( \mathcal { P } )$ </td><td> $\mu _ { \mathrm { m a x } } ( \mathcal { P } ) \pm \sigma _ { \mathrm { m a x } } ( \mathcal { P } )$ </td></tr><tr><td>Equivalent dynamics</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Similar dynamics</td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 9 8 \pm 0 . 0 5$ </td></tr><tr><td>Uniformly diverse dynamics</td><td> $0 . 0 3 \pm 0 . 0 3$ </td><td> $1 . 1 5 \pm 0 . 1 7$ </td></tr><tr><td>Arbitrarily diverse dynamics</td><td> $0 . 5 7 \pm 0 . 1 1$ </td><td> $1 . 4 4 \pm 0 . 0 5$ </td></tr></table>

![](images/8159ad5869197e9a39f6e3dae5d1ae6033c22566658ba1bbe5fcf879adb14d00.jpg)

![](images/a195d88bf5afb1dcf4ae46b776b4a5a4b8136392b890d29799416e3c4dc83404.jpg)

![](images/02ccfabc4820621035d5e22e308d2713002cdbbc453e2ff870899b8f3ae6e696.jpg)

![](images/51a19dfe1e831af3a02c61d970ef357c1783d3833ee93eb23fce4f077139df2e.jpg)  
log det tr   <sub>2</sub> log det (ofline) tr (ofline)   <sub>2</sub> (ofline)  
Figure 5: Worst-case suboptimality over the training set $\mathcal { P }$ as a function of the number of observed expert trajectories on randomly generated $\mathrm { M D P s } ,$ and of the criterion J used to pick dynamics online (7) or ofline (35). Each plot corresponds to a diferent structure of ${ \mathcal P } .$

Results. Figure 5 reports the suboptimality attained by our methods when using either the log det $( I + \cdot )$ , tr, or $\lVert \cdot \rVert _ { 2 }$ criterion for the active selection objective J defined in Equation (7). In these experiments, all methods operate without a warm-up phase, meaning that dynamics are selected actively from the very first round. We additionally compare against the ofline variant introduced in Section E.

We first observe that all methods achieve comparable performance when the dynamics set is uniformly distributed with respect to principal angles. Diferences only emerge in the arbitrarily distributed setting, where the spectral norm criterion and the ofline trace-based variant perform slightly worse, reaching approximately twice the suboptimality of the best-performing methods. Interestingly, the ofline variant remains competitive in this setting.

Moreover, Table 3 shows that the log-determinant and trace criteria provide the strongest transfer performance to previously unseen dynamics. In contrast, the ofline variants generally exhibit weaker transferability overall, although the trace-based ofline criterion performs best among the three ofline alternatives.

Figure 6 reports the worst-case suboptimality afte $n = 1 0 0$ rounds as a function of the warm-up share $n _ { 0 } / n ,$ on the arbitrarily diverse dynamics sets, using the log det $\left( I + \cdot \right)$ criterion for the active phase. The two endpoints recover the Active only $( n _ { 0 } = 0 )$ and round-robin $( n _ { 0 } = n )$ selection rules. Suboptimality increases monotonically with the warm-up share. Actively selecting dynamics from the very first round is the best choice, and roundrobin selection ends about three times worse. The degradation is however mild for short warm-ups: the share of $1 / 3$ used in the main paper costs roughly 15% relative to no warm-up, and shares up to $1 / 2$ remain within 30%. Most of the gap opens once the warm-up exceeds half of the budget, where the active phase no longer has enough rounds to rebalance the counts toward the most informative dynamics. The warm-up therefore acts as a cheap safeguard against poor early Hessian estimates rather than as a source of performance.

## I.3 Random mazes

Details. Mazes are $l \times l$ grids with $l = 7 , \mathrm { i . e . , } \left| S \right| = 4 9$ cells, with $| { \mathcal { A } } | = 5$ actions corresponding to four moves and a waiting action. Transitions are deterministic, a move into a wall leaves the agent in place, and the initial state is the top-left cell. For each seed, we draw three goal cells $s _ { 1 } , s _ { 2 } , s _ { 3 }$ uniformly among the non-initial cells, set $r ^ { \mathrm { E } } ( s _ { j } , a _ { \mathrm { w a i t } } ) \sim \mathcal { U } [ 0 , 1 ]$ , and $r ^ { \mathrm { E } } ( s , a ) = 0$ elsewhere. The reward vector is then rescaled to a uniformly random norm in [0, 1], so that $\begin{array} { r } { \| \dot { r } ^ { \mathrm { E } } \| _ { 2 } \leq 1 } \end{array}$

Table 3: Worst-case suboptimality on sets of held-out stochastic dynamics after learning from $n = 1 0 0$ expert trajectories collected on either uniformly or arbitrarily diverse sets of dynamics.
<table><tr><td rowspan="2">criterion J</td><td rowspan="2">Offline</td><td colspan="2">Worst-case suboptimality (×100) on held-out dynamics</td></tr><tr><td>Uniformly diverse training set</td><td>Arbitrarily diverse training set</td></tr><tr><td>log det(I + ·)</td><td>No</td><td> ${ \bf 0 2 . 8 6 \pm 0 . 0 2 }$ </td><td> $1 2 . 6 3 \pm 0 . 1 4$ </td></tr><tr><td>tr</td><td>No</td><td> $0 2 . 9 6 \pm 0 . 0 2$ </td><td> ${ \bf 1 2 . 4 0 \pm 0 . 1 4 }$ </td></tr><tr><td>_I·|2</td><td>No</td><td> $0 3 . 0 2 \pm 0 . 1 9$ </td><td> $1 5 . 2 3 \pm 0 . 1 7$ </td></tr><tr><td> $\bar { \log { \operatorname* { d e t } ( \bar { I } + \cdot ) } }$ </td><td>Yes</td><td> $\mathbf { \bar { 0 3 . 1 3 \pm 0 . 0 2 } }$ </td><td> $\bar { 1 } 3 . \bar { 5 } \bar { 4 } \bar { \pm } \bar { 0 } . \bar { 1 } 5$ </td></tr><tr><td>tr</td><td>Yes</td><td> $0 3 . 1 4 \pm 0 . 2 0$ </td><td> $1 2 . 6 4 \pm 0 . 1 5$ </td></tr><tr><td> $\left\| \cdot \right\| _ { 2 }$ </td><td>Yes</td><td> $0 3 . 1 4 \pm 0 . 0 2$ </td><td> $1 5 . 1 4 \pm 0 . 1 7$ </td></tr></table>

![](images/85fd7b7722221c7521fc91e4fdec1e926a42ea57a1bd3aca00ee01db6a717a4c.jpg)  
Figure 6: Worst-case suboptimality over the training set  after $n = 1 0 0$ rounds, as the warmup share $n _ { 0 } / n$ increases. The vertical gray dashed line corresponds to the share of $1 / 3$ used in the main paper.

A set of $| \mathcal { P } |$ mazes is generated incrementally. The i-th maze $( i = 0 , \ldots , | \mathcal { P } | - 1 )$ starts from the walls of the $( i - 1 ) – \mathrm { t h }$ and is modified by a randomized wall-adding procedure that targets a shortest path from the initial state to the nearest goal of length $\begin{array} { r } { \ell _ { i } : = \left( \frac { i } { | \mathcal { P } | - 1 } \right) ^ { 2 } l ^ { 2 } } \end{array}$ , accepted within $\pm 5 \%$ . Walls are proposed uniformly at random and kept only if they do not shorten this path, so the path length is non decreasing in i. The procedure stops after 15,000 proposals and returns the maze reached, so that $\ell _ { i }$ acts as a target rather than a constraint. In practice, for $| \mathcal { P } | = 1 0 0$ , shortest paths range from 4 to about 44 steps, well below a Hamiltonian path. For odd $i ,$ the first wall that disconnects every goal from the initial state is kept, yielding an unsolvable maze. Roughly a third of each set is unsolvable, so that rewards must be inferred partly from dynamics in which the goals are unreachable.

For reward estimation, we run Algorithm 3 with 100 inner iterations per round using SGD with step size 0.02.

Results. Beyond the results presented in the main paper, we additionally evaluate settings with dynamics sets of size $| \mathcal { P } | = 5$ and $| \mathcal { P } | = 2 0 $ in order to study the impact of the number of candidate dynamics on the efectiveness of our active learning strategy.

Figure 7 reports both the suboptimality and reward estimation error achieved by our methods (using the logdeterminant criterion), together with passive learning and ED [10], for $| \mathcal { P } | \in \{ 5 , 2 0 , 1 0 0 \}$ . Across all values of , our active selection rule (with and without warm-up) consistently outperforms the competing approaches. This suggests that actively selecting dynamics improves sample eficiency even when the set of candidate dynamics is relatively small. However, the ofline variant (see Section E) performs worse than passively selecting dynamics, hinting that in general, it is hard to entirely choose dynamics ofline for expert demonstrations to learn the expert reward.

We also observe that the passive warm-up phase used in our two-phase method is not beneficial. In all tested settings, removing this warm-up phase leads to improved sample eficiency, with the gains becoming even more

pronounced as the number of dynamics increases.

![](images/bb65d9dbceb5ce21474e65fe1691420119b5b3c04bd07b5c395d28dc139e36e4.jpg)  
Figure 7: Worst-case suboptimality over the training set of randomly generated mazes, and the corresponding reward error, as a function of the number of observed expert trajectories, and of the size of the dynamics set.