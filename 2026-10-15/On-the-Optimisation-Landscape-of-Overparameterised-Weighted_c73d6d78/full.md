# On the Optimisation Landscape of Overparameterised Weighted MMD

Tharrshann Jayan Logarajah University College London

Jeremias Knoblauch University College London

Louis Sharrock University College London

## Abstract

We study the optimisation landscape of the maximum mean discrepancy (MMD) over weighted atomic measures. For the onedimensional Gaussian kernel, we show that, for any compactly supported target distribution, suficiently many particles ensure that every local minimum has arbitrarily small MMD. Moreover, any stationary point whose MMD remains bounded away from zero is unstable under an unweighted gradient flow over the particle locations. These results hold both when the weights are constrained to the simplex, and when they are unconstrained but optimised at each step. When the target is itself finitely supported, we further strengthen this result: beyond an explicit particle threshold, there are no spurious local minima, and every spurious stationary point is unstable. To our knowledge, these are the first finite-particle overparameterisation guarantees of their kind for weighted MMD optimisation.

## 1 INTRODUCTION

A common problem in statistics, machine learning, and numerical analysis is approximating a probabil ity measure using a finite collection of representative points. This task, commonly referred to as quantisation (Graf and Luschgy, 2000), arises in various applications, including experimental design (Kiefer, 1959), numerical integration and quadrature (Mahsereci and Karvonen, 2026), and the compression of Monte Carlo output (Riabiz et al., 2022).

A natural way to formalise this problem is to choose a discrepancy between probability measures and minimise it over a suitable approximating family. We focus here on the maximum mean discrepancy (MMD), a kernel-based integral probability metric widely used for comparing probability distributions (Gretton et al., 2006, 2012). Given a reproducing kernel Hilbert space (RKHS) $\mathcal { H } _ { k }$ with kernel k, the MMD between probability measures P and $Q$ is defined by

$$
\mathrm { M M D } ( P , Q ) = \operatorname* { s u p } _ { \| f \| _ { \mathcal { H } _ { k } } \leq 1 } \left| \int f \mathrm { d } P - \int f \mathrm { d } Q \right| .\tag{1}
$$

For a characteristic kernel, the MMD defines a metric over probability measures. Moreover, for functions in the RKHS, a small MMD approximation error implies a low integration error when expectations under the target $P ^ { * }$ are replaced by expectations under $P .$ This, in addition to its generality, practicality, and favourable sample complexity, motivates the use of the MMD over alternative discrepancies in numerical integration and related problems (e.g., Chen et al., 2026).

At the level of probability measures, MMD minimisation has a particularly simple optimisation structure. For a convex class $\mathcal { Q } \subseteq \mathcal { P } ( \mathcal { X } )$ , consider

$$
\operatorname* { m i n } _ { P \in \mathcal { Q } } \mathrm { M M D } ^ { 2 } ( P , P ^ { * } ) .\tag{2}
$$

The objective is convex in P, since the kernel mean embedding is linear in the measure and the squared RKHS norm is convex. In particular, if $\mathcal { Q } = \mathcal { P } ( \mathcal { X } )$ and k is characteristic, then the unique minimiser is simply $P = P ^ { * }$

In practice, exactly representing an arbitrary probability measure is rarely feasible, and one instead seeks an approximate representation involving finitely many particles. Let $\textbf { \em x } = ~ ( x _ { 1 } , \ldots , x _ { N } )$ ∈ $\mathcal { X } ^ { N }$ denote the locations of the particles, and $\textbf { w } = ~ ( w _ { 1 } , \dots , w _ { N } ) ~ \in ~ \Delta ^ { N - 1 }$ their weights, where $\Delta ^ { N - 1 } = \lbrace \mathbf { w } \in [ 0 , 1 ] ^ { N } : \sum _ { i = 1 } ^ { N } w _ { i } = 1 \rbrace$ . In place of (2), we are then interested in solving

$$
\operatorname* { m i n } _ { { \mathbf { w } } \in \Delta ^ { N - 1 } , { \pmb x } \in { \pmb X } ^ { N } } \mathrm { M M D } ^ { 2 } \big ( \sum _ { i = 1 } ^ { N } w _ { i } \delta _ { x _ { i } } , P ^ { * } \big ) .\tag{3}
$$

This problem is commonly referred to as weighted quantisation (e.g., Belhadji et al., 2026a). To simplify the analysis, it is common also to consider a relaxed formulation in which the weights are unconstrained and optimised at each particle configuration (Belhadji et al., 2026a). The passage from (2) to (3) fundamentally changes the nature and the dificulty of the optimisation problem. Indeed, while the objective in (2) is convex as a function of the measure, the restriction to measures supported on at most N points yields a nonconvex feasible class, and the resulting objective (3) is generally nonconvex in the particle locations.

The dificulty of this problem is very well recognised. In kernel quadrature and MMD quantisation, Belhadji et al. (2019), Teymur et al. (2021), and Belhadji et al. (2026a) all emphasise the nonconvexity of optimising particle locations, even when the weights can be optimised analytically or the points are selected sequentially. Reflecting this dificulty, Chen et al. (2026) relax the goal of global minimisation and instead analyse stationary points, proving a superconvergence property for numerical integration and a finite-particle guarantee for computing them by MMD gradient flow. More generally, much of the existing literature either develops algorithms designed to miti gate the nonconvex optimisation problem, or seeks useful guarantees without requiring a global minimiser. Much less well understood is the optimisation landscape itself: namely, whether increasing the number of particles can eliminate poor local minima for finite $N .$

A complementary line of work studies MMD optimisation through measure-valued dynamics and their particle approximations, often obtaining guarantees at the population level (Arbel et al., 2019; Gladin et al., 2024; Kang et al., 2026). Such results provide insight into the behaviour of particle methods as the number of particles becomes large and, in several cases, establish convergence of finite-particle systems towards underlying measure-valued dynamics with stronger convergence guarantees. They do not, however, directly describe the optimisation landscape at any fixed and finite number of particles N. In particular, they do not rule out poor local minima of the finite-particle objective, nor provide an explicit particle threshold beyond which every local minimum approximates the target accurately.

This leads to the central question of the present work:

## Does the finite-particle optimisation landscape of MMD quantisation become benign once suficiently many particles are used?

More specifically, how large must the number of particles be to guarantee that every local minimum is accurate? And, beyond local minima, are stationary points with large MMD values necessarily unstable under the

## associated gradient flow?

Our main results (cf. Table 1) give afirmative answers to these questions for the one-dimensional Gaussian kernel. To be specific, our main contributions can be summarised as follows.

1. For the one-dimensional Gaussian kernel, we show that for every $D > 0 { \mathrm { ~ a n d ~ } } \varepsilon > 0$ there exists a finite threshold $N _ { \varepsilon } ( D )$ such that, whenever $N > N _ { \varepsilon } ( D )$ and $P ^ { * }$ has support diameter at most $D ,$ every local minimum of (3) satisfies MMD $< \varepsilon .$ We also derive computable bounds for $N _ { \varepsilon } ( D )$ . We further show that every stationary point with ${ \mathrm { M M D } } \geq \varepsilon$ is unstable for the unweighted particle gradient $f l o w ;$ see Section 3 for a precise definition. These results hold both for simplex-constrained weights and the relaxed problem. To our knowledge, this is the first finite-particle threshold for MMD quantisation guaranteeing that every local minimum lies below a prescribed error for suficiently large N.

2. In the well-specified setting, where the target ad mits an atomic representation $\begin{array} { r } { P ^ { * } = \sum _ { j = 1 } ^ { K } w _ { j } ^ { * } \delta _ { x _ { j } ^ { * } } } \end{array}$ for some finite $K ,$ , we show that there exists a finite threshold $N ( P ^ { * } )$ such that, whenever $N > N ( P ^ { * } )$ , there are no spurious local minima. That is, every local minimum exactly represents $P ^ { * }$ . We also provide computable bounds on this threshold; in the relaxed case, the thresh old is simply $N \geq K$ . Moreover, this threshold is sharp. We also show that all spurious stationary points are unstable under unweighted particle gradient flow. Once again, our results hold both for simplex-constrained weights and for the relaxed problem with unconstrained weights.

3. We establish structural properties of stationary points. In particular, we prove a maximum principle implying that every distinct particle location is active; that is, it has non-zero total weight. We also show that any collision, where $x _ { i } = x _ { j }$ for some $i \neq j ,$ is either globally optimal $\mathrm { o r } .$ if spurious, is neither a local minimum nor stable under the unweighted particle gradient flow. Sev eral of these results extend beyond the Gaussian kernel: for unconstrained weights they hold for extended totally positive (ETP) kernels, while in the simplex-constrained setting they hold additionally under translation invariance and normalisability, together with positivity of the associated Karush–Kuhn–Tucker (KKT) constant, which we verify for the Gaussian kernel.

Table 1: Summary of the one-dimensional landscape and instability guarantees. Instability refers to the unweighted particle flow $\dot { \theta } _ { i } = - \Delta ^ { \prime } ( \theta _ { i } )$ . Relaxed weights are unrestricted and optimised at each particle configuration; the simplex results concern the joint objective in the weights and positions. The rates are asymptotic as $\varepsilon  0 .$ , with D and the Gaussian bandwidth fixed.
<table><tr><td>Setting and result</td><td>Particle threshold</td><td>Guarantees</td></tr><tr><td>Relaxed weights K-atom target, ETP kernel (Theorem 4)</td><td> $\overline { { N \geq K } }$ </td><td>No spurious local minima. Every spurious stationary point is unstable.</td></tr><tr><td>Relaxed weights Target with com- pact support of diameter at most D, Gaussian kernel (Theorem 5)</td><td> $\begin{array} { r } { N > N _ { \varepsilon } ( D ) = O _ { D } \left( \frac { \log ( 1 / \varepsilon ) } { \log \log ( 1 / \varepsilon ) } \right) } \end{array}$ </td><td>Every local minimum has  $\mathrm { M M D } < \varepsilon .$  Stationary points with MMD ≥ ε are unstable.</td></tr><tr><td>Simplex weights Target with com- pact support of diameter at most D, Gaussian kernel (Theorem 6)</td><td> $N > N _ { \varepsilon } ( D ) = O _ { D } { \big ( } \varepsilon ^ { - 4 } \log ( 1 / \varepsilon ) { \big ) }$ </td><td>Every local minimum has MMD &lt; ε. Stationary points with MMD ≥ ε are</td></tr><tr><td>Simplex weights Finite atomic target, Gaussian kernel (Theorem 7)</td><td> $N > N ( P ^ { * } )$ </td><td>unstable. No spurious local minima. Every spurious stationary point is unstable.</td></tr></table>

## 2 RELATED WORK

The MMD has been widely studied since the foundational work of Gretton et al. (2006, 2012), including for quantisation of probability measures. Chen et al. (2010) introduce kernel herding for constructing point sets with fast RKHS integration rates, while Bach et al. (2012) interpret kernel herding as conditional-gradient optimisation of the corresponding convex optimisation problem over measures. Teymur et al. (2021) study sequential MMD quantisation, Xu et al. (2022) derive finite-particle MMD quantisation bounds, and Pronzato (2023) analyses rates for kernel herding and sequential Bayesian quadrature.

These works primarily address how accurately a target can be approximated using particular finite-particle constructions. A separate question is how to optimise the resulting finite-particle MMD objective. Amongst recent work, Belhadji et al. (2026a) is particularly closely related to our relaxed formulation. They consider the case where the weights are unconstrained and optimised for each particle configuration, and introduce the mean shift interacting particles algorithm. This corresponds to preconditioned gradient descent on the resulting weight-optimised objective. Related finite-particle optimisation problems are studied by Belhadji et al. (2019, 2026b).

These papers are primarily algorithmic: they do not characterise the landscape of the underlying objective itself. Closer to this question, Chen et al. (2026) analyse stationary MMD points, proving a superconvergence property for numerical integration, together with a non-asymptotic guarantee for computing such points via MMD gradient flow. Our focus is instead on the local minima of the finite-particle objective, together with the stability of its stationary points. The closest work from an optimisation landscape perspective is Alon et al. (2021), which establishes benign MMD landscapes for several structured Gaussian parametric models. Our setting instead permits, in the well-specified case, an arbitrary finite number of target atoms with arbitrary locations and unequal weights, while our approximate guarantees require only compact support.

A complementary line of work studies MMD optimisation through measure-valued dynamics and their particle approximations. Arbel et al. (2019) analyse the gradient flow of the MMD and the corresponding particle approximation. More recently, several works obtain stronger guarantees by modifying the dynamics or discrepancy (Kang et al., 2026; Gladin et al., 2024; Chen et al., 2025; Tian et al., 2026). Meanwhile, Rosenzweig et al. (2026) establish convergence to collision-free critical sets for MMDs induced by a class of nonsmooth energy kernels. In contrast, we characterise the landscape of the MMD objective directly for a fixed number of particles, rather than the convergence of a particular flow or its approximation to a population limit.

Finally, several of our proof techniques originate in the super-resolution and approximation-theory literature. Schiebinger et al. (2018) and Duval (2020) develop interpolation and Chebyshev-system techniques for recovery of sparse measures, including Gaussian measurement kernels. We adapt related tools to constrain stationary configurations of the MMD. There is also a connection to critical points of Gaussian sums: Wang (2026) recently proved finiteness of the critical set of finite isotropic Gaussian mixtures. Our witness functions are signed Gaussian combinations, so results for positive Gaussian mixtures do not directly apply.

## 3 PROBLEM SETUP

We now introduce the basic problem setup, as well as some relevant definitions. For particle locations $\pmb { x } = ( x _ { 1 } , \dots , x _ { N } )$ and weights $\mathbf { w } = ( w _ { 1 } , \dots , w _ { N } )$ , write $\begin{array} { r } { P _ { \mathbf { w } , \pmb { x } } = \sum _ { i = 1 } ^ { N } w _ { i } \delta _ { x _ { i } } } \end{array}$ for the corresponding weighted atomic measure. Define

$$
F ( \mathbf { w } , \pmb { x } ) = \mathrm { M M D } ^ { 2 } ( P _ { \mathbf { w } , \pmb { x } } , P ^ { * } ) .
$$

We can write this function in the form $F ( \mathbf { w } , \pmb { x } ) =$ $\| \Delta \| _ { \mathcal { H } _ { k } } ^ { 2 }$ , where $\Delta \in \mathcal { H } _ { k }$ is the function on X defined by

$$
\Delta ( x ) = \mu _ { P _ { \bf w } , x } ( x ) - \mu _ { P ^ { * } } ( x ) , \qquad x \in \mathcal { X } .
$$

This is known as the unnormalised witness function (e.g., Gretton et al., 2012). The terminology arises from the variational characterisation of the MMD in (1): whenever $\Delta$ is non-zero, the normalised function $\Delta / \| \Delta \| _ { \mathcal { H } _ { k } }$ attains the supremum, thereby witnessing the discrepancy between the two distributions. This function will allow us to translate stationarity of the objective into constraints on $\Delta ,$ to which we can apply interpolation and zero-counting arguments.

Whenever the relevant derivatives exist, it is straight forward to compute

$$
\frac { \partial F } { \partial w _ { i } } = 2 \Delta ( x _ { i } ) , \qquad \frac { \partial F } { \partial x _ { i } } = 2 w _ { i } \Delta ^ { \prime } ( x _ { i } ) .
$$

Below, we will frequently refer to the unweighted particle gradient flow, by which we mean

$$
\dot { x } _ { i } = - \Delta ^ { \prime } ( x _ { i } ) , \qquad i = 1 , \dots , N .
$$

We will refer to a stationary configuration as unstable under this flow if there exists a neighbourhood of the configuration such that, for some arbitrarily small perturbations of the particle locations, the resulting trajectory eventually leaves this neighbourhood.

We consider two main regimes: arbitrary compactly supported targets, and the setting in which the target admits an atomic representation $\begin{array} { r } { \breve { P } ^ { * } = \sum _ { j = 1 } ^ { K } w _ { j } ^ { * } \delta _ { x _ { i } ^ { * } } } \end{array}$ for some finite K. In the latter case, we say that the problem is well specified, since for $N \geq K$ the target can be represented exactly by the approximating atomic measure, yielding zero MMD. Note that this setting covers fitting homoscedastic Gaussian mixtures with fixed variance. Indeed, for the Gaussian kernel $k _ { \sigma } ,$ the quantity $\begin{array} { r } { \mathrm { M M D } _ { k _ { \sigma } } ^ { 2 } ( \sum _ { i = 1 } ^ { { N } } w _ { i } \mathcal { N } ( x _ { i } , \tau ^ { 2 } ) , \sum _ { j = 1 } ^ { K } w _ { j } ^ { * } \hat { \mathcal { N } } ( x _ { j } ^ { * } , \tau ^ { 2 } ) ) } \end{array}$ is a positive scalar multiple of

$$
\begin{array} { r } { \mathrm { M M D } _ { k } ^ { 2 } \underset { \sqrt { \sigma ^ { 2 } + 2 \tau ^ { 2 } } } { \underbrace { - 1 } } ( \sum _ { i = 1 } ^ { N } w _ { i } \delta _ { x _ { i } } , \sum _ { j = 1 } ^ { K } w _ { j } ^ { * } \delta _ { x _ { j } ^ { * } } ) . } \end{array}
$$

Thus, our guarantees in the well-specified case transfer directly to the Gaussian-mixture problem.

We will work on $\chi = \mathbb { R }$ and, unless otherwise speci fied, use the Gaussian kernel $k ( x , y ) = \exp ( - ( x - y ) ^ { 2 } )$ corresponding to the normalisation $2 \sigma ^ { 2 } = 1$ . This is without loss of generality: for a general bandwidth, the transformation $x \mapsto x / { \sqrt { 2 \sigma ^ { 2 } } }$ reduces the problem to this form, with the corresponding support-diameter bounds rescaled in the same way.

## 4 RELAXED VARIANT

We first consider the relaxed problem in which the weights are unconstrained and optimised for each fixed particle configuration. That is, the objective is

$$
\bar { F } ( \pmb { x } ) = \operatorname* { m i n } _ { \mathbf { w } \in \mathbb { R } ^ { N } } \mathrm { M M D } ^ { 2 } \big ( \sum _ { i = 1 } ^ { N } w _ { i } \delta _ { x _ { i } } , P ^ { * } \big ) .\tag{4}
$$

This is the weight-optimised objective underlying, for example, the mean shift interacting particles method introduced by Belhadji et al. (2026a). We begin by establishing structural consequences of stationarity, before using these to obtain a sharp result in the wellspecified setting. We then turn to arbitrary compactly supported targets.

Our analysis hinges on the observation that optimality of the weights and stationarity of the locations give

$$
\Delta ( x _ { j } ) = 0 , \qquad w _ { j } \Delta ^ { \prime } ( x _ { j } ) = 0 , \qquad j = 1 , \ldots , N .
$$

Thus, at every particle location with non-zero weight, both $\Delta$ and its derivative vanish, yielding a zero of multiplicity at least two. As we now show, suficiently many distinct such locations impose more zeros, counted with multiplicity, than a non-zero witness function can support.

## 4.1 Structural properties of stationary points

We now establish a maximum principle for the witness. Full proofs of this and subsequent results are given in App. B.

Proposition 1. Suppose x is a stationary point of F<sup>¯</sup> with $\bar { F } ( { \pmb x } ) > 0$ , and let w attain the minimum in (4). Suppose also that k is ETP. Then $\Delta \le 0 _ { i }$ , every distinct particle location is active, with positive total weight, and, for all $j = 1 , \ldots , N$

$$
\Delta ^ { \prime } ( x _ { j } ) = 0 , \qquad \Delta ^ { \prime \prime } ( x _ { j } ) < 0 .
$$

Further, all particles lie in the convex hull ofsupp(P<sup>∗</sup>).

Prop. 1 establishes that every particle location is a strict local maximum of $\Delta ,$ and that every distinct particle location has positive total weight. The latter conclusion is related to Karvonen et al. (2019), who establish positivity of optimally weighted quadrature rules under diferent assumptions. In particular, they require the target measure to be compactly supported and to admit a Lebesgue density on a bounded interval, and the moving particles to already lie within that interval. Here, positivity and containment in the convex hull both follow directly from stationarity, although our assumptions on the kernel are stronger.

The fact that every distinct particle location is active also clarifies the role of collisions; that is, configurations with $x _ { i } = x _ { j }$ for some $i \neq j$ . Whilst colliding particles may have weights of diferent signs, the maximum principle ensures that their total weight at the common location is positive, so merging them cannot remove that location from the resulting measure.

We next show that collisions are unstable under the unweighted particle gradient flow.

Proposition 2. Under the assumptions of Prop. $^ { 1 , }$ a collision is unstable under the unweighted particle gradient flow.

Proof sketch. Following ideas used for collision singularities in Prony systems (Batenkov and Yomdin, 2014) and exponential least-squares approximation (Evans et al., 1980), a pair of colliding kernel functions can be replaced by a divided-diference basis. This remains non-degenerate at the collision and converges to a kernel derivative. Smoothness of an ETP kernel gives the required continuity of $\Delta ^ { \prime \prime } ;$ Prop. 1 then implies that separating the particles is unstable.

To rule out local minima containing redundant particles, we also need to exclude zero individual weights.

Proposition 3. Suppose k is real, analytic, and universal, or that k is ETP and the problem is well specified. Then, at any nonzero local minimum of F<sup>¯</sup>, every weight vector w attaining the minimum in (4) satisfies $w _ { i } \neq 0$ for all $i = 1 , \ldots , N$ . Moreover, the particle locations are distinct.

## 4.2 Well-specified targets

We first use these structural results to obtain an exact landscape characterisation in the well-specified setting. Proposition 3 ensures that increasing N genuinely increases the number of zero constraints imposed on the witness function $\Delta .$ In particular, a putative local minimum cannot simply absorb additional particles by assigning them zero weight. We can therefore combine the stationarity conditions with the zero-counting property of ETP kernels.

Theorem 4. Suppose the kernel is ETP and the problem is well specified with $P ^ { * }$ supported on K points. If $N \geq K _ {  }$ , then F<sup>¯</sup> has no spurious local minima, and every spurious stationary point is unstable under the unweighted particle gradient flow.

Proof. First consider a nonzero local minimum. By Prop. 3, all particle locations are distinct and every optimal weight is nonzero. Hence stationarity forces a double zero of $\Delta$ at each of the N particle locations. Since ∆ is a nonzero combination of at most $N + K$ distinct kernel sections, extended total positivity implies that it has at most $N + K - 1$ zeros counting multiplicity. Thus $2 N \le N + K - 1$ , which requires $N \leq K - 1$ . Therefore, when $N \geq K$ , every local minimum must satisfy $\bar { F } = 0$

Now consider a spurious stationary point. If all par ticle locations are distinct, Prop. 1 implies that every location is active, so the same zero-counting argument again gives a contradiction when $N \geq K$ . Hence every spurious stationary point must contain a collision, which is unstable by Prop. 2. □

Theorem 4 shows that, in the relaxed well-specified problem, using enough particles to represent the target sufices to eliminate all spurious local minima. For completeness, we provide a numerical illustration of this result, as well as our other main optimisation landscape results (cf. Theorems 5–7), in App. D. The observed behaviour is consistent with our theory, recovering in particular the sharp $N \geq K$ threshold established in Theorem 4.

## 4.3 Arbitrary compactly supported targets

We now turn to arbitrary compactly supported targets. In this case, the same zero-counting principle yields a quantitative approximation guarantee without requiring the well-specified setting.

Theorem 5. Let $D > 0$ and $\varepsilon > 0$ . There exists a finite threshold $N _ { \varepsilon } ( D )$ such that, uniformly over all target distributions with compact support of diameter at most D, whenever $N > N _ { \varepsilon } ( D )$ every local minimum of $\bar { F }$ satisfies

$$
\mathrm { M M D } ( P , P ^ { * } ) < \varepsilon .
$$

Moreover, every stationary point with $\mathrm { M M D } ( P , P ^ { * } ) \geq$ ε is unstable under the unweighted particle gradient flow.

Proof sketch. This result is also proved via a zero counting argument. First, the maximum principle (cf. Prop. 1) confines the active particles to the convex hull of the target support. Meanwhile, compactness and analyticity give an upper bound on the number of distinct active locations that a stationary witness with $\mathrm { M M D } ( P , P ^ { * } ) \ge \varepsilon$ can support, depending only on D and ε. At a nonzero local minimum, all N particle locations must be distinct (cf. Prop. 3), yielding a contradiction once N exceeds this bound. Meanwhile, for a general stationary point, exceeding the bound forces a collision, which must be unstable (cf. Prop. 2).

## 5 SIMPLEX VARIANT

We now return to the original problem, where ${ \textbf { w } } \in$ $\Delta ^ { N - 1 }$ . The simplex constraint makes the analysis substantially more dificult: weight stationarity no longer forces $\Delta ( x _ { i } ) = 0$ , but instead introduces a common KKT level c.

Our proof proceeds in two stages. We first establish structural properties common to both results. We next use an interpolation argument to show that the simplex KKT level c is positive, after which a maximum principle forces every active particle location to be a strict local maximum of the witness function and provides control of collisions. We then relate the MMD, the KKT level $^ { c , }$ and the number of distinct active particle locations. For arbitrary compactly supported targets, this yields the approximate guarantee. In the well-specified setting, we additionally show that every spurious stationary point must miss some target atom by a fixed neighbourhood, which provides the positive MMD lower bound needed to obtain the exact result.

Our two main results can be stated as follows.

Theorem 6. Let $D > 0$ and $\varepsilon > 0$ . There exists a finite threshold $N _ { \varepsilon } ( D )$ such that, uniformly over all target distributions with compact support of diameter at most D, whenever $N > N _ { \varepsilon } ( D )$ , every local minimum of F over $\mathbf { w } \in \Delta ^ { N - 1 }$ and $\mathbf { \boldsymbol { x } } \in \mathbb { R } ^ { N }$ satisfies

$$
\mathrm { M M D } ( P , P ^ { * } ) < \varepsilon .
$$

Moreover, every stationary point with $\mathrm { M M D } ( P , P ^ { * } ) \geq$ $\varepsilon$ is unstable under the unweighted particle gradient flow.

Theorem 7. Suppose the problem is well specified. There exists a finite threshold $N ( P ^ { * } )$ such that, whenever $N > N ( P ^ { * } )$ , F has no spurious local minima over $\mathbf { w } \in \Delta ^ { N - 1 }$ and $\pmb { x } \in \mathbb { R } ^ { N }$ . Moreover, every spurious stationary point is unstable under the unweighted particle gradient flow.

## 5.1 Structural properties of stationary points

We first develop the structural ingredients common to both results. To begin, observe that the KKT conditions for the simplex weights give a constant c such that, for every $j = 1 , \ldots , N$ ，

$$
\Delta ( x _ { j } ) \ge c , \quad w _ { j } ( \Delta ( x _ { j } ) - c ) = 0 , \quad w _ { j } \Delta ^ { \prime } ( x _ { j } ) = 0 .
$$

As in the relaxed problem, we consider the reduced measure defined by removing zero-weight particles and merging particles lying at the same coordinate. Let $\vartheta _ { 1 } , \ldots , \vartheta _ { A }$ denote the distinct active locations and $\varsigma _ { 1 } , \ldots , \varsigma _ { A }$ their positive total weights. For these, the stationarity conditions give, for every $j = 1 , \dotsc , A$

$$
\Delta ( \vartheta _ { j } ) = c , \quad \Delta ^ { \prime } ( \vartheta _ { j } ) = 0 ,\tag{5}
$$

Unlike in the relaxed problem, we must now control the shifted witness function $\Delta - c .$ . Our first ingredient is an interpolation certificate built from Gaussian kernels and their derivatives. Given distinct centres $\bar { x } _ { 1 } , \ldots , \bar { x } _ { K }$ and distinct target locations $x _ { 1 } ^ { * } , \ldots , x _ { K } ^ { * }$ , we construct a function H centred at the $\bar { x } _ { i }$ such that $H ( x _ { i } ^ { * } ) = 1 , H ^ { \prime } ( x _ { i } ^ { * } ) = 0$ , and $H ( x ) < 1$ away from the target locations. This extends related super-resolution certificates (Schiebinger et al., 2018; Duval, 2020) by allowing the kernel centres $\bar { x } _ { i }$ to difer from the interpolation locations $\boldsymbol { x } _ { i } ^ { * }$

Proposition 8. Let $\bar { x } _ { 1 } , \ldots , \bar { x } _ { K }$ and $x _ { 1 } ^ { * } , \ldots , x _ { K } ^ { * }$ each consist of distinct real numbers. Define

$$
H ( \cdot ) = \sum _ { i = 1 } ^ { K } \left[ \alpha _ { i } k ( \cdot , \bar { x } _ { i } ) + \beta _ { i } \partial _ { 2 } k ( \cdot , \bar { x } _ { i } ) \right] .
$$

Then there exist $\alpha , \beta$ such that $H ( x ) < 1$ for all x $\notin$ $\{ x _ { 1 } ^ { * } , \ldots , x _ { K } ^ { * } \}$ , and

$$
H ( x _ { i } ^ { * } ) = 1 , \qquad H ^ { \prime } ( x _ { i } ^ { * } ) = 0 \qquad i = 1 , \ldots , K .
$$

The certificate immediately gives us control of the KKT level. Indeed, suppose we apply Prop. 8 with the interpolation centres and interpolation locations equal to the distinct active particle locations. The resulting function equals one at every active location and is strictly less than one elsewhere. At a non-optimal stationary point, this allows us to show $\begin{array} { r } { \int H \mathrm { d } ( P - P ^ { * } ) > 0 . } \end{array}$ . On the other hand, the stationarity conditions express the same quantity as $c \sum _ { i } \alpha _ { i } .$ with $\textstyle \sum _ { i } \alpha _ { i } = \| H \| _ { \mathcal { H } _ { k } } ^ { 2 } > 0$ . It follows that the KKT level must be strictly positive.

Proposition 9. Suppose we are at a stationary point of F with $F ( \mathbf { w } , \pmb { x } ) > 0$ . Then $0 < c < 1$

We can now establish the simplex analogue of Prop. 1. Unlike the relaxed case, the stationarity conditions involve the shifted witness function $\Delta - c .$ Translation invariance and normalisability allow the constant c to be represented as a kernel integral against Lebesgue measure, while the condition $c \geq 0$ ensures that the resulting signed measure admits the same variationdiminishing argument for ETP kernels.

Proposition 10. Suppose (w, x) is a stationary point of $\bar { F }$ with $F ( \mathbf { w } , { \pmb x } ) > 0$ , with weights constrained to the simplex. Suppose also that k is ETP, translation invariant, and normalisable, and that $c \geq 0$ . Then $\Delta ( x ) \leq c$ for all $x ,$ , every distinct particle location is active, and for every $i = 1 , \ldots , N$

$$
\Delta ( x _ { i } ) = c , \qquad \Delta ^ { \prime } ( x _ { i } ) = 0 , \qquad \Delta ^ { \prime \prime } ( x _ { i } ) < 0 .
$$

Thus every particle location lies at a strict local maximum of the witness function at the common level c. As in the relaxed problem, cf. Prop. 2, this immediately gives strong control of collisions.

Proposition 11. Under the assumptions of Prop. 10, every collision is unstable under the unweighted particle gradient flow.

Proof. Separate two colliding particles by a small distance $\boldsymbol { r } ~ = ~ \boldsymbol { x } _ { j } - \boldsymbol { x } _ { i }$ . Under the unweighted particle flow, $\dot { r } = - \big ( \Delta ^ { \prime } ( x _ { j } ) - \Delta ^ { \prime } ( x _ { i } ) \big ) = - \Delta ^ { \prime \prime } \hat { ( } \vartheta ) r$ for some $\vartheta \in ( x _ { i } , x _ { j } )$ by the mean value theorem. Since $\Delta ^ { \prime \prime } < 0$ in a neighbourhood of the collision, ˙r has the same sign as r. Thus the collision is unstable. □

Prop. 11 rules out collisions as stable stationary configurations under the particle flow. To obtain the corresponding conclusion for the optimisation landscape, we must also show that collisions cannot themselves be local minima.

Proposition 12. Under the assumptions of Prop. 10, a collision is not a local minimum.

Proof. Suppose $x _ { i } = x _ { j }$ , and $w _ { i } , w _ { j } > 0 .$ . Perturb the locations to $\begin{array} { r } { x _ { i } ~ - ~ \frac { w _ { j } } { w _ { i } + w _ { j } } \varepsilon } \end{array}$ and $\begin{array} { r } { x _ { j } + \frac { w _ { i } } { w _ { i } + w _ { j } } \varepsilon } \end{array}$ . If $F ( \varepsilon )$ denotes the perturbed objective, then

$$
F ( \varepsilon ) - F ( 0 ) = \frac { w _ { i } w _ { j } } { w _ { i } + w _ { j } } \Delta ^ { \prime \prime } ( x _ { j } ) \varepsilon ^ { 2 } + o ( \varepsilon ^ { 2 } ) < 0 ,
$$

for all suficiently small ε. Thus, the perturbation decreases the objective for all suficiently small ε. If one of the weights is zero, transfer an arbitrarily small amount of mass to that particle without changing the measure, then apply the same perturbation. □

Together, the maximum principle and the collision results allow us to work with active, distinct particle locations when studying nonzero local minima. This is crucial: every such location contributes the full pair of equality constraints $\Delta ( x _ { i } ) = c$ and $\Delta ^ { \prime } ( x _ { i } ) = 0$ , allowing the number of distinct active locations to be controlled by zero counting.

## 5.2 Arbitrary compactly supported targets

We first use these results to establish a guarantee for arbitrary compactly supported targets. The key step is to relate a lower bound on the MMD to a lower bound on the KKT level c, and then to an upper bound on the number of distinct active particle locations. The first relation follows by splitting the Fourier representation of the MMD into low and high frequencies.

Proposition 13. Let $P , P ^ { * }$ be probability measures on $\mathbb { R } ^ { d }$ , let $\rho = P - P ^ { * }$ , and suppose k is continuous, translation invariant, and has positive spectral density. Then, writing $\Delta = \mu _ { P } - \mu _ { P ^ { * } }$ , for every $\nu > 0$ it holds that

$$
\begin{array} { r l } & { \mathrm { M M D } ^ { 2 } ( P ^ { * } , P ) \le \displaystyle \frac { \| \Delta \| _ { 1 } ^ { 2 } } { ( 2 \pi ) ^ { d } } \int _ { | \omega | \le \nu } \frac { 1 } { \hat { k } ( \omega ) } \mathrm { d } \omega } \\ & { \qquad + \displaystyle \frac { \| \rho \| _ { \mathrm { T V } } ^ { 2 } } { ( 2 \pi ) ^ { d } } \int _ { | \omega | > \nu } \hat { k } ( \omega ) \mathrm { d } \omega . } \end{array}
$$

In particular, for the one-dimensional Gaussian kernel with $2 \sigma ^ { 2 } = 1$ 2

$$
\begin{array} { r } { \mathrm { M M D } ^ { 2 } ( P , P ^ { * } ) \leq \frac { \| \Delta \| _ { 1 } ^ { 2 } } { \pi } \operatorname { e r f i } \left( \frac { \nu } { 2 } \right) + 4 \operatorname { e r f c } \left( \frac { \nu } { 2 } \right) . } \end{array}
$$

where erfi $\begin{array} { r } { \mathfrak { i } ( z ) = \frac { 2 } { \sqrt { \pi } } \int _ { 0 } ^ { z } e ^ { t ^ { 2 } } } \end{array}$ dt, $\textstyle \operatorname { e r f c } ( z ) = { \frac { 2 } { \sqrt { \pi } } } \int _ { z } ^ { \infty } e ^ { - t ^ { 2 } }$ dt denote the imaginary error function and complementary error function, respectively.

The maximum principle then lets us control $\| \Delta \| _ { 1 }$ through c and the target support diameter, without dependence on the particular particle configuration.

Proposition 14. Suppose $P ^ { * }$ has compact support of diameter at most D, and $( \mathbf { w } , \pmb { x } )$ is a stationary point with $F ( \mathbf { w } , \pmb { x } ) > 0$ . Then

$$
\begin{array} { r l } & { \| \Delta \| _ { 1 } \leq 2 ( \sqrt { 2 } + D ) c + 1 6 c \sqrt { - \log c } } \\ & { \qquad + 6 \sqrt { \pi } \operatorname { e r f c } ( \sqrt { - \log c } ) + 4 \sqrt { \pi } c \sqrt { 1 - 2 \log c } . } \end{array}
$$

Proof sketch. Split the real line into regions near the target support, intermediate regions, and Gaussian tails. The maximum principle controls the witness directly on the central region; the tails are bounded using Gaussian decay. The main dificulty is to remove any dependence on how far the extreme particles lie from the target. We do this using the stationarity equations at the extreme particles to obtain probabilitymass bounds, after which suitable choices of the splitting points yield the stated uniform estimate. □

The bound in Prop. 14 is $O ( c { \sqrt { - \log c } } )$ , and thus tends to zero as $c \to 0 ^ { + }$ . Note, in particular, that erf $: ( \sqrt { - \log c } ) ~ = ~ O ( c / \sqrt { - \log c } )$ , so this term is of lower order. Together with Prop. 13, this implies that a stationary point with MMD bounded away from zero must have its KKT level bounded away from zero. We now convert this into a bound on the number of distinct particle positions.

Proposition 15. A positive lower bound on the KKT level c gives an upper bound on the number A of distinct active particle locations. One such bound is

$$
A \leq \frac { 2 - c } { c } \log \left( \frac { 2 ( c + 2 ) } { c } \right) + \frac { e } { 2 } \left( \frac { 8 - 5 c } { c } \right) ^ { 2 } \log \left( \frac { 2 } { c } \right) .
$$

Proof sketch. The maximum principle gives a lower bound on the probability mass in a neighbourhood of each particle. We group overlapping neighbourhoods into connected components and construct a maximal separated packing in each component. This controls the component diameters and hence, by Jensen-type zero counting for the shifted witness function $\Delta - c ,$ the number of double zeros contributed by the particles. Summing over components gives the stated bound.

For a stationary point for which $\mathrm { M M D } ( P , P ^ { * } ) \geq \varepsilon .$ Props. 13–14 give a positive lower bound on $c ,$ depend ing only on $D$ and ε. Prop. 15 then gives an upper bound on the number of distinct active particle locations. If the total number of particles N exceeds this upper bound, at least two particles must share a location. Prop. 11–12 then rule out a poor local minimum or stable stationary point. This proves Theorem 6.

## 5.3 Well-specified targets

We now turn to the well-specified setting. In this case, it remains to obtain a target-dependent positive lower bound on the MMD at every spurious stationary point. Our first result shows that such a point must remain uniformly separated from at least one target atom.

Proposition 16. Suppose that the problem is well specified, with $\begin{array} { r } { P ^ { * } = \dot { \sum } _ { j = 1 } ^ { K } w _ { j } ^ { * } \delta _ { x _ { j } ^ { * } } } \end{array}$ . Then there exists $r ( P ^ { * } ) > 0$ such that every spurious stationary point satisfies

$$
\operatorname* { m a x } _ { j } \operatorname* { m i n } _ { i } \left| x _ { i } - x _ { j } ^ { * } \right| \geq r ( P ^ { * } ) .
$$

It may not be surprising that any spurious stationary point must miss some target atoms by some amount. The important point here is that the lower bound $r ( P ^ { * } )$ is independent of the number of particles $N .$ Thus, adding more particles cannot produce a sequence of spurious stationary points which approaches every target atom arbitrarily closely.

We next convert this geometric separation into a positive lower bound on the MMD.

Proposition 17. Suppose a target atom is missed $b y$ a neighbourhood of radius r as in Prop. 16. Then

$$
\mathrm { M M D } ( P , P ^ { * } ) \geq \frac { [ w ^ { * } - ( a r ) ^ { - 2 m } ] _ { + } } { C ( a , m ) } ,
$$

where $\begin{array} { r } { [ u ] _ { + } = \operatorname* { m a x } \{ u , 0 \} , w ^ { * } = \operatorname* { m i n } _ { j } w _ { j } ^ { * } } \end{array}$ , and one may take

$$
C ( a , m ) = \frac { \sqrt { \pi } } { a } e ^ { \frac { 1 } { 2 } m ^ { 2 } a ^ { 2 } } , \qquad m \in \mathbb { N } _ { \geq 1 } , \qquad a > 0 .
$$

Thus, choosing $a , m$ such that $( a r ) ^ { - 2 m } < w ^ { * }$ gives a strictly positive lower bound on the MMD.

Prop. 17 therefore gives a positive lower bound on the MMD at every spurious stationary point, depending only on $P ^ { * }$ . The preceding quantitative argument can now be reused: Props. 13–14 convert this into a positive lower bound on $c ,$ while Prop. 15 gives a finite upper bound on the number of distinct active particle locations. Beyond this bound, a collision is unavoidable, and Prop. 11–12 rule out a spurious local minimum or stable stationary point. This proves Theorem 7.

## 6 CONCLUSION

Summary. We have shown that overparameterisation can fundamentally simplify the finite-particle MMD landscape. For the one-dimensional Gaussian kernel and any compactly supported target of diameter at most $D ,$ there is a computable threshold $N _ { \varepsilon } ( D )$ beyond which every local minimum has MMD below $\varepsilon .$ Meanwhile, every stationary point with MMD at least ε is unstable under the unweighted particle gradient flow. These conclusions hold for both simplexconstrained and relaxed, weight-optimised formula tions. In the well-specified setting, beyond $N =$ $N ( P ^ { * } )$ , all local minima are globally optimal and all spurious stationary points are unstable. For the relaxed problem, this extends to ETP kernels with the sharp threshold $N \geq K$ . Our maximum principles and collision results explain this phenomenon: suficiently many active, distinct particles impose more value and derivative constraints on the witness function than a poor stationary configuration can support. Combined with algorithm-specific convergence theory, our results provide approximation guarantees for methods converging to local minima or to stationary points stable under the corresponding particle flow.

Limitations and future work. The principal limitation of this work is the restriction of our strongest results to one dimension. Our proofs rely on onedimensional zero counting and an ordering argument for particle signs, neither of which transfers directly to higher dimensions. Developing multidimensional analogues is thus an important open problem. This notwithstanding, it is worth emphasising that our results provide the first finite-particle overparameterisation guarantees of their kind for weighted MMD optimisation. A second direction is to extend our results to broader classes of kernels. The maximum principles already hold beyond Gaussian kernels, while the simplex analysis additionally requires suitable interpolation and zero-counting arguments. In particular, Prop. 8 relies on a Chebyshev-system property of Gaussian kernels, their derivatives, and the constant function. The diferent dependence on ε in the simplex and relaxed analyses suggests that the simplex bounds are unlikely to be tight, while in the well-specified sim plex case sharper control of the interpolation matrices may improve the threshold. By contrast, the relaxed well-specified threshold $N \geq K$ is sharp. Finally, our results concern stationary configurations with finite particle locations. For asymptotically stationary trajectories, particles may escape to infinity, evading the finite-position argument. The appendix provides a preliminary analysis; controlling such escape for specific flows would turn our results into stronger global convergence guarantees.

## AI use statement

In this work, we used generative AI tools for assistance in the writing of proofs, for implementing methods and for designing or providing feedback on research methodology or experiments. We also used generative AI tools for brainstorming, sourcing & searching for information, identifying relevant literature, formatting references, suggesting experimental parameters and suggesting structure. We have not used generative AI tools for helping to develop theoretical models or conceptual frameworks, formulating mathematical claims, providing critical ingredients for proving mathematical claims, proposing or refining hypotheses, generating synthetic data sets, assisting with translation, cleaning and reformatting datasets, supporting qualitative and thematic data analysis, or interpreting results.

All LLM assisted work was checked carefully, including any routine or standard results or applications it assisted with. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

Itai Alon, Amir Globerson, and Ami Wiesel. On the optimization landscape of maximum mean discrepancy. arXiv preprint arXiv:2110.13452, 2021.

Michael Arbel, Anna Korba, Adil Salim, and Arthur Gretton. Maximum mean discrepancy gradient flow. In Advances in Neural Information Processing Systems, volume 32, pages 6484–6494. Curran Associates, Inc., 2019.

Francis Bach, Simon Lacoste-Julien, and Guillaume Obozinski. On the equivalence between herding and conditional gradient algorithms. In Proceedings of the 29th International Conference on Machine Learning, pages 1355–1362. Omnipress, 2012.

Dmitry Batenkov and Yosef Yomdin. Geometry and singularities of the Prony mapping. Journal of Singularities, 10:1–25, 2014. doi: 10.5427/jsing.2014. 10a.

B. J. C. Baxter and R. Brummelhuis. Functionals of exponential Brownian motion and divided diferences. Journal of Computational and Applied Mathematics, 236(4):424–433, 2011. doi: 10.1016/j.cam. 2011.06.010.

Ayoub Belhadji, R´emi Bardenet, and Pierre Chainais. Kernel quadrature with DPPs. In Advances in Neural Information Processing Systems, volume 32, pages 12927–12937. Curran Associates, Inc., 2019.

Ayoub Belhadji, Daniel Sharp, and Youssef Marzouk. Weighted quantization using MMD: From mean field

to mean shift via gradient flows. In Proceedings of The 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 1045–1053. PMLR, 2026a.

Ayoub Belhadji, Daniel Sharp, and Youssef M. Marzouk. To discretize continually: Mean shift interacting particle systems for Bayesian inference. arXiv preprint arXiv:2605.14142, 2026b.

Eike B¨orgens, Christian Kanzow, Patrick Mehlitz, and Gerd Wachsmuth. New constraint qualifications for optimization problems in Banach spaces based on asymptotic KKT conditions. SIAM Journal on Optimization, 30(4):2956–2982, 2020. doi: 10.1137/19M1306804.

Yutian Chen, Max Welling, and Alexander J. Smola. Super-samples from kernel herding. In Proceedings of the Twenty-Sixth Conference on Uncertainty in Artificial Intelligence, pages 109–116. AUAI Press, 2010.

Zonghao Chen, Aratrika Mustafi, Pierre Glaser, Anna Korba, Arthur Gretton, and Bharath K. Sriperumbudur. (De)-regularized maximum mean discrepancy gradient flow. Journal of Machine Learning Research, 26(235):1–77, 2025.

Zonghao Chen, Toni Karvonen, Heishiro Kanagawa, Fran¸cois-Xavier Briol, and Chris J. Oates. Stationary MMD points. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 14884–14912. PMLR, 2026.

Danilo Costarelli and Gianluca Vinti. Convergence of sampling Kantorovich operators in modular spaces with applications. Rendiconti del Circolo Matematico di Palermo Series 2, 70(2):1115–1136, 2021. ISSN 1973-4409. doi: 10.1007/s12215-020-00544-z.

Yohann de Castro, S´ebastien Gadat, Cl´ement Marteau, and Cathy Maugis-Rabusseau. SuperMix: Sparse regularization for mixtures. The Annals of Statistics, 49(3):1779–1809, 2021. doi: 10.1214/ 20-AOS2022.

DLMF. NIST Digital Library of Mathematical Functions. Release 1.2.8 of 2026-09-15, 2026. URL https://dlmf.nist.gov/. F. W. J. Olver, A. B. Olde Daalhuis, D. W. Lozier, B. I. Schneider, R. F. Boisvert, C. W. Clark, B. R. Miller, B. V. Saunders, H. S. Cohl, and M. A. McClain, eds.

Vincent Duval. A characterization of the nondegenerate source condition in super-resolution. Information and Inference: A Journal of the IMA, 9 (1):235–269, 2020. doi: 10.1093/imaiai/iaz002.

John W. Evans, William B. Gragg, and Randall J. LeVeque. On least squares exponen-

tial sum approximation with positive coeficients. Mathematics of Computation, 34(149):203–211, 1980. ISSN 00255718, 10886842. doi: 10.1090/ S0025-5718-1980-0551298-6.

Egor Gladin, Pavel Dvurechensky, Alexander Mielke, and Jia-Jie Zhu. Interaction-force transport gradient flows. In Advances in Neural Information Processing Systems, volume 37, pages 14484–14508. Curran Associates, Inc., 2024. doi: 10.52202/079017-0462.

Siegfried Graf and Harald Luschgy. Foundations of Quantization for Probability Distributions, volume 1730 of Lecture Notes in Mathematics. Springer, Berlin, Heidelberg, 2000. doi: 10.1007/BFb0103945.

Arthur Gretton. RKHS in ML: Comparing a sample and a model, 2024. URL https://www.gatsby.ucl.ac.uk/ gretton/ coursefiles/lecture5\_distribEmbed\_3.pdf. Lecture slides, Gatsby Computational Neuroscience Unit, University College London.

Arthur Gretton. RKHS in ML: Comparing two samples, 2025. URL https://www.gatsby. ucl.ac.uk/ gretton/coursefiles/lecture5\_ distribEmbed\_1.pdf. Lecture slides, Gatsby Computational Neuroscience Unit, University College London.

Arthur Gretton, Karsten Borgwardt, Malte Rasch, Bernhard Sch¨olkopf, and Alex J. Smola. A kernel method for the two-sample-problem. In Advances in Neural Information Processing Systems, volume 19, pages 513–520. MIT Press, 2006.

Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Sch¨olkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(25):723–773, 2012.

Karlheinz Gr¨ochenig, Jos´e Luis Romero, and Joachim St¨ockler. Sharp results on sampling with derivatives in shift-invariant spaces and multi-window Gabor frames. Constructive Approximation, 51(1):1– 25, 2020. doi: 10.1007/s00365-019-09456-3.

Charles R. Harris, K. Jarrod Millman, St´efan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fern´andez del R´ıo, Mark Wiebe, Pearu Peterson, Pierre G´erard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array programming with NumPy. Nature, 585(7825):357–362, 2020. doi: 10.1038/s41586-020-2649-2.

John D. Hunter. Matplotlib: A 2D graphics environ-

ment. Computing in Science & Engineering, 9(3): 90–95, 2007. doi: 10.1109/MCSE.2007.55.

Sophia Seulkee Kang, Louis Sharrock, Xiaoyuan Cheng, Fran¸cois-Xavier Briol, and Zonghao Chen. A gradient flow perspective on minimum MMD estimation. arXiv preprint arXiv:2607.03871, 2026.

Samuel Karlin and William J. Studden. Tchebychef Systems: With Applications in Analysis and Statistics, volume 15 of Pure and Applied Mathematics. Interscience Publishers, New York, 1966.

Samuel Karlin, Frank Proschan, and Richard Eugene Barlow. Moment inequalities of P´olya frequency functions. Pacific Journal of Mathematics, 11(3): 1023–1033, 1961. doi: 10.2140/pjm.1961.11.1023.

Toni Karvonen, Motonobu Kanagawa, and Simo S¨arkk¨a. On the positivity and magnitudes of Bayesian quadrature weights. Statistics and Computing, 29(6):1317–1333, 2019. doi: 10.1007/ s11222-019-09901-0.

J. Kiefer. Optimum experimental designs. Journal of the Royal Statistical Society: Series B (Methodological), 21(2):272–304, 1959. doi: 10.1111/j.2517-6161. 1959.tb00338.x.

M. G. Krein. The ideas of P. L. Chebyshev and A. A. Markov in the theory of limiting values of integrals and their further development. American Mathematical Society Translations, Series 2, 12:1– 121, 1959. doi: 10.1090/trans2/012/01.

Maren Mahsereci and Toni Karvonen. Bayesian quadrature. Foundations and Trends in Machine Learning, 19(2–3):121–240, 2026. doi: 10.1108/ FTMAL-06-2026-0132.

Luc Pronzato. Performance analysis of greedy algorithms for minimising a maximum mean discrepancy. Statistics and Computing, 33(1):14, 2023. doi: 10.1007/s11222-022-10184-1.

Python Software Foundation. Python, 2024. URL https://www.python.org/downloads/release/ python-3131/. Version 3.13.1.

Marina Riabiz, Wilson Ye Chen, Jon Cockayne, Pawel Swietach, Steven A. Niederer, Lester Mackey, and Chris J. Oates. Optimal thinning of MCMC output. Journal of the Royal Statistical Society Series B: Statistical Methodology, 84(4):1059–1081, 2022. doi: 10.1111/rssb.12503.

Matthew Rosenzweig, Dejan Slepˇcev, and Lihan Wang. Wasserstein gradient flows of maximum mean discrepancy with energy kernels. arXiv preprint arXiv:2608.01182, 2026.

Geofrey Schiebinger, Elina Robeva, and Benjamin Recht. Superresolution without separation. Infor-

mation and Inference: A Journal of the IMA, 7(1): 1–30, 2018. doi: 10.1093/imaiai/iax006.

integral operators. SIAM Journal on Scientific Computing, 20(2):699–718, 1998. doi: 10.1137/ S1064827596310779.

Antonin Schrab. A practical introduction to kernel discrepancies: MMD, HSIC & KSD. arXiv preprint arXiv:2503.04820, 2025.

Elias M. Stein and Rami Shakarchi. Complex Analysis, volume 2 of Princeton Lectures in Analysis. Princeton University Press, 2010. ISBN 9781400831159.

Onur Teymur, Jackson Gorham, Marina Riabiz, and Chris Oates. Optimal quantisation of probability measures using maximum mean discrepancy. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 1027–1035. PMLR, 2021.

The mpmath development team. mpmath: A Python library for arbitrary-precision floating-point arithmetic, 2026. URL https://mpmath.org/. Version 1.4.0.

Chenyang Tian, Bharath K. Sriperumbudur, Arthur Gretton, and Zonghao Chen. Sobolev regularized MMD gradient flow. arXiv preprint arXiv:2605.11884, 2026.

Case Van Horsen. gmpy2: Multiple-precision arithmetic for Python, 2024. URL https://pypi.org/ project/gmpy2/2.2.1/. Version 2.2.1.

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, St´efan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, C J Carey, <sup>˙</sup>Ilhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E. A. Quintero, Charles R. Harris, Anne M. Archibald, Antˆonio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17(3): 261–272, 2020. doi: 10.1038/s41592-019-0686-2.

Haiyang Wang. On finite Gaussian mixtures: Finiteness of the number of modes and an application to NPMLE. arXiv preprint arXiv:2608.16675, 2026.

N. Yarvin and V. Rokhlin. Generalized Gaussian quadratures and singular value decompositions of

Lantian Xu, Anna Korba, and Dejan Slepˇcev. Accurate quantization of measures via interacting particle-based optimization. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 24576–24595. PMLR, 2022.

# Supplementary Materials

## A BACKGROUND AND PRELIMINARIES

## A.1 Kernel basics

In this appendix we collect some basic definitions relating to kernels, as well as some other background material necessary for the subsequent proofs. For a more detailed exposition, we refer the interested reader to the included references. The definitions below are taken from Schrab (2025); Gretton (2024, 2025).

Let X be a non-empty set.

Definition 18 (Hilbert space). A vector space H is called a Hilbert space if it has an inner product $\langle \cdot , \cdot \rangle _ { \mathscr { H } }$ and is complete.

Definition 19 (Reproducing Kernel Hilbert Space). For a Hilbert space H of real-valued functions on X , a function $k : \mathcal { X } \times \overset { } { \mathcal { X } }  \mathbb { R }$ is a reproducing kernel of H if

1. $k ( x , \cdot ) \in \mathcal { H }$ for all $x \in \mathcal X$

2. $\langle f , k ( x , \cdot ) \rangle _ { \mathcal { H } } = f ( x )$ for all $f \in \mathcal H$ and all $x \in \mathcal { X }$

In this case, H is called a Reproducing Kernel Hilbert Space (RKHS).

Definition 20 (Mean Embedding). Let k be a kernel, and P be a probability distribution. Then the (kernel) mean embedding is defined $b y$

$$
\begin{array} { r } { \mu _ { P } ( t ) = \mathbb { E } _ { x \sim P } k ( x , t ) . } \end{array}
$$

The existence of such a mean embedding follows by the Riesz Representation Theorem under reasonable conditions.

Definition 21 (Maximum Mean Discrepancy). Let k be a kernel, and $P , Q$ be probability distributions. Then the maximum mean discrepancy (MMD) is defined by

$$
\mathrm { M M D } ^ { 2 } ( P , Q ) = | | \mu _ { P } - \mu _ { Q } | | _ { \mathcal { H } } ^ { 2 } .
$$

Proposition 22 (Reproducing property for the derivative). For a twice continuously diferentiable kernel $k ( x , x ^ { \prime } )$ with associated RKHS F and feature map φ(x), and $f o r \ f \in { \mathcal { F } }$ , we have

$$
\frac { d } { d x } f ( x ) = \langle f , \frac { d } { d x } \varphi ( x ) \rangle _ { \mathcal { F } } , \qquad \langle \frac { d } { d x } \varphi ( x ) , \varphi ( x ^ { \prime } ) \rangle _ { \mathcal { F } } = \frac { d } { d x } k ( x , x ^ { \prime } ) .
$$

Throughout, we will typically focus on the Gaussian kernel.

Definition 23 (Gaussian Kernel). The Gaussian kernel with bandwidth σ is, for x, y real vectors in some common space,

$$
k _ { \sigma } ( x , y ) = \exp { \left( - \frac { | | x - y | | ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } .
$$

Finally, we introduce the notion of a normalisable kernel.

Definition 24 (Normalisable kernel). A translation-invariant kernel k is called normalisable $\dot { \mathcal { f } } \mathrm { ~ 0 ~ < ~ } \mathcal { Z } : =$ $\textstyle \int _ { \mathbb { R } } k ( x , y ) d x < \infty$

## A.2 Total positivity, variation diminishing, Chebyshev systems

We now introduce further background which will be necessary for the arguments that follow. These definitions are recalled from Karlin et al. (1961).

Definition 25 (Totally Positive (TP)). A function $f ( x , y )$ of two real variables, x ranging over X and y ranging over Y, is said to be totally positive of order k $( T P _ { k } )$ if for all $x _ { 1 } < \ldots < x _ { m } , y _ { 1 } < \ldots < y _ { m } ( x _ { i } \in X , y _ { j } \in Y )$ and all $1 \leq m \leq k$

$$
f \left( { \begin{array} { c c c c } { x _ { 1 } , \ldots , x _ { m } } \\ { y _ { 1 } , \ldots , y _ { m } } \end{array} } \right) = { \left| \begin{array} { l l l } { f ( x _ { 1 } , y _ { 1 } ) } & { \ldots } & { f ( x _ { 1 } , y _ { m } ) } \\ { \vdots } & { } & { \vdots } \\ { f ( x _ { m } , y _ { 1 } ) } & { \ldots } & { f ( x _ { m } , y _ { m } ) } \end{array} \right| } \geq 0 .
$$

Additionally, if all indicated derivatives exist, and $X , Y$ represent open intervals on the line, then for $1 \leq n \leq k$

$$
{ \begin{array} { r l } { f ( x , y ) \quad \quad \cdot \cdot \cdot \quad } & { \left. { \frac { \partial ^ { n - 1 } } { \partial y ^ { n - 1 } } } f ( x , y ) \right| } \\ { \vdots \quad \quad \quad \quad \vdots } \\ { \left| { \frac { \partial ^ { n - 1 } } { \partial x ^ { n - 1 } } } f ( x , y ) \quad \cdot \cdot \cdot \quad { \frac { \partial ^ { 2 n - 2 } } { \partial x ^ { n - 1 } \partial y ^ { n - 1 } } } f ( x , y ) \right| } \end{array} } \qquad\tag{6}
$$

The Gaussian kernel is $T P _ { \infty }$ . One of the key properties of these functions we will utilise, particularly in our proofs of Props. 1 and 10, is the variation diminishing property. We here present a particularly strong form that holds with additional assumptions, lifted directly from Karlin et al. (1961).

Definition 26 (Variation diminishing property). Let $V ( g )$ denote the number of variations of sign of a function g(x) as x traverses the real line from left to right. Let h(x) be given by the absolutely convergent integral

$$
h ( x ) = \int f ( x , w ) g ( w ) d \sigma ( w ) ,\tag{7}
$$

where $f ( x , w )$ is $T P _ { k } , \sigma ( w )$ is a non-negative σ-finite measure and $V ( g ) \leq k - 1$ . Suppose also that h(x) is k times continuously diferentiable, the interchange of integral and derivative is permissible in (7), and (6) holds strictly. Suppose h ̸≡ 0. Let $Z ( h )$ denote the number of zeros of $h ( x )$ , counting multiplicities. Then $Z ( h ) \leq V ( g )$

The following definitions are recalled from Yarvin and Rokhlin (1998), which itself takes these from Karlin and Studden (1966); Krein (1959).

Definition 27 (Chebyshev system). A sequence of functions $\phi _ { 1 } , . . . , \phi _ { n }$ is referred to as a Chebyshev system on the interval [a, b] if each of them is continuous and the determinant

$$
\begin{array} { r l } { \left| \phi _ { 1 } ( x _ { 1 } ) \quad \ldots \quad \phi _ { 1 } ( x _ { n } ) \right| } & { { } } \\ { \vdots } & { { } \vdots } \\ { \phi _ { n } ( x _ { 1 } ) } & { { } \ldots } & { { } \phi _ { n } ( x _ { n } ) } \end{array}
$$

is nonzero for any sequence of points $a \leq x _ { 1 } < . . . < x _ { n } \leq b$ . Equivalently, any nontrivial linear combination of the functions has at most n − 1 distinct zeros.

We now define a special case that is particularly relevant when using the Gaussian kernel, as will be seen shortly. Definition 28 (Extended Chebyshev system). Given a set of functions $\phi _ { 1 } , . . . , \phi _ { n }$ which are continuously differentiable as needed on an interval $[ a , b ]$ , and given a sequence of ordered points $a \leq x _ { 1 } \leq \ldots \leq x _ { n } \leq b ,$ , define the sequence $m _ { 1 } , \ldots , m _ { n }$ by

$$
m _ { 1 } = 0 , \qquad m _ { j } = \left\{ \begin{array} { l l } { { 0 } } & { { x _ { j } \ne x _ { j - 1 } } } \\ { { m _ { j - 1 } + 1 } } & { { x _ { j } = x _ { j - 1 } } } \end{array} \right. .\tag{8}
$$

Let the matrix $C ( x _ { 1 } , \dots , x _ { n } ) = [ c _ { i j } ]$ be defined by the formula

$$
c _ { i j } = { \frac { d ^ { m _ { j } } \phi _ { i } } { d x ^ { m _ { j } } } } ( x _ { j } ) .
$$

Then $\phi _ { 1 } , . . . , \phi _ { n }$ is an extended Chebyshev system on [a, b] $i f \operatorname* { d e t } ( C ( x _ { 1 } , \ldots , x _ { n } ) )$ is nonzero for all such sequences $x _ { i } .$

As remarked in Yarvin and Rokhlin (1998), this is essentially just a Chebyshev system where, if a point is repeated, the repeated column is replaced with the derivative of the function, successively taking the next derivative if it is repeated again. This matrix is then mandated to be invertible.

Finally, we introduce the kernel property underlying our zero-counting arguments.

Definition 29 (Extended Totally Positive (ETP)). Given a function $K : [ a , b ] \times [ c , d ]  \mathbb { R }$ which is continuously diferentiable as needed, and given a sequence of points $x _ { 1 } , \ldots , x _ { n }$ such that $a \leq x _ { 1 } \leq \ldots \leq x _ { n } \leq b $ , let the sequence $m _ { 1 } , \ldots , m _ { n }$ be defined in (8). Let the functions $\phi _ { 1 } , . . . , \phi _ { n }$ be defined by the formula $\begin{array} { r } { \phi _ { j } ( t ) = \frac { \partial ^ { m _ { j } } K } { \partial x ^ { m _ { j } } } ( x _ { j } , t ) } \end{array}$ Then K is $E T P \ i f \phi _ { 1 } , \ldots , \phi _ { n }$ form an extended Chebyshev system on [c, d] for all such sequences of x<sub>i</sub>.

Crucially, the Gaussian kernel is ETP (e.g., Yarvin and Rokhlin, 1998). If we have a reproducing kernel on the real line, this is a stronger condition than $\mathrm { T P _ { \infty } }$ and the strictness asked for in the variation diminishing property. Indeed, positive semidefiniteness of the kernel makes the relevant determinants non-negative when $x = y ,$ , and ETP makes them nonzero. They are thus all positive when $x = y$ , and since the real line is connected and the determinants are always nonzero, must be positive for all x, y.

## A.3 Zero counting using complex analysis and specialisation to Gaussian-type functions

We will employ a bounding strategy similar to that used in Gr¨ochenig et al. (2020), Lemmas 4.1 and 4.3. In particular, we make use of the following result.

Proposition 30. Let $R > 0$ . Let $\delta > 1$ . Let Ω be an open set that contains the closure of a disc $D _ { \delta R }$ and suppose that f is holomorphic in Ω with $f ( 0 ) \neq 0$ . Let $\mathbf { n } _ { f } ( r )$ denote the number of zeros (counting multiplicity) of f in the disc of radius r. Assume that $| f ( x + i y ) | \leq \dot { A _ { f } } e ^ { B _ { f } y ^ { 2 } }$ . Then

$$
\mathbf { n } _ { f } ( R ) \leq \frac { \log \bigl ( \frac { A _ { f } } { | f ( 0 ) | } \bigr ) + \frac { 1 } { 2 } B _ { f } \delta ^ { 2 } R ^ { 2 } } { \log ( \delta ) } .
$$

Proof. By Jensen’s formula (e.g., pg. 137 Stein and Shakarchi, 2010), we have that

$$
\frac { 1 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } \log | f ( \delta R e ^ { i \theta } ) | d \theta - \log | f ( 0 ) | = \int _ { 0 } ^ { \delta R } \frac { \mathbf { n } _ { f } ( r ) } { r } d r \geq \int _ { R } ^ { \delta R } \frac { \mathbf { n } _ { f } ( r ) } { r } d r \geq \mathbf { n } _ { f } ( R ) \log ( \delta ) .
$$

By the assumed bound, we have $| f ( \delta R e ^ { i \theta } ) | \leq A _ { f } e ^ { B _ { f } \delta ^ { 2 } R ^ { 2 } \sin ^ { 2 } ( \theta ) }$ . Thus, integrating and substituting into the previous bound, it follows that

$$
\mathbf { n } _ { f } ( R ) \leq \frac { \log \bigl ( \frac { A _ { f } } { \vert f ( 0 ) \vert } \bigr ) + \frac { 1 } { 2 } B _ { f } \delta ^ { 2 } R ^ { 2 } } { \log ( \delta ) } . \boxed { }
$$

## B PROOFS

## B.1 Relaxed Variant

Proposition 1. Suppose x is a stationary point of $\bar { F }$ with $\bar { F } ( { \pmb x } ) > 0$ , and let w attain the minimum in (4). Suppose also that k is ETP. Then $\Delta \leq 0$ , every distinct particle location is active, with positive total weight, and, for all $j = 1 , \ldots , N$ ，

$$
\Delta ^ { \prime } ( x _ { j } ) = 0 , \qquad \Delta ^ { \prime \prime } ( x _ { j } ) < 0 .
$$

Further, all particles lie in the convex hull of supp(P<sup>∗</sup>).

Proof. We begin by defining the reduced measure of P by merging particles at the same location, and removing locations with zero total weight. Write $\vartheta _ { 1 } , \ldots , \vartheta _ { A }$ for its distinct locations and $\varsigma _ { 1 } , \ldots , \varsigma _ { A }$ for their nonzero total weights. By the stationarity conditions, we have

$$
\Delta ( \vartheta _ { j } ) = 0 , \qquad \Delta ^ { \prime } ( \vartheta _ { j } ) = 0 , \qquad j = 1 , \ldots , A .
$$

To apply the variation diminishing property, write

$$
\Delta ( x ) = \int k ( t , x ) g ( t ) \mathrm { d } \sigma ( t ) , \qquad g = \frac { \mathrm { d } ( P - P ^ { * } ) } { \mathrm { d } \sigma } , \qquad \sigma = | P | + P ^ { * } .
$$

Since the positive part of $g$ is supported on at most A distinct locations, we must have $V ( g ) \leq 2 A$ . Moreover, since $\bar { F } ( { \pmb x } ) > 0 .$ , the witness $\Delta$ is not identically zero. The variation diminishing property (cf. Def. 26) therefore gives $Z ( \Delta ) \leq V ( g )$ , where zeros are counted with multiplicity. On the other hand, the stationarity conditions imply that each $\vartheta _ { j }$ is a zero of multiplicity at least two. Consequently,

$$
2 A \leq Z ( \Delta ) \leq V ( g ) \leq 2 A .
$$

Thus all inequalities are equalities, and the $\vartheta _ { j }$ are the only zeros of $\Delta ,$ each with multiplicity exactly two. Since the positive part of $g$ is supported on at most A locations, equality $V ( g ) = 2 A$ requires the alternating sign pattern negative–positive–negative–· · · –positive–negative. Thus every location in the reduced measure must have strictly positive total weight. Moreover, since the negative contributions outside the positive locations must arise from $P ^ { * }$ , these locations lie within the convex hull of supp $\left( P ^ { * } \right)$ .

We next establish that $\Delta \le ~ 0$ . Since all zeros of $\Delta$ have multiplicity exactly two, $\Delta$ cannot change sign. Furthermore, the weight optimality conditions give $\begin{array} { r } { \int \Delta \mathrm { d } P = 0 , } \end{array}$ , and hence

$$
\int \Delta \mathrm { d } P ^ { * } = - \int \Delta \mathrm { d } ( P - P ^ { * } ) = - \| \Delta \| _ { \mathcal { H } _ { k } } ^ { 2 } < 0 .
$$

It follows that $\Delta \le 0$ everywhere. Finally, weight optimality gives $\Delta ( x _ { i } ) = 0$ for every particle, including those with zero individual weight. Since all zeros of $\Delta$ occur at the active locations $\vartheta _ { 1 } , \ldots , \vartheta _ { A }$ , every particle must coincide with one of these locations. Thus every distinct particle location is active, has positive total weight, and lies within the convex hull of supp(P<sup>∗</sup>). Moreover, each particle location is a strict local maximum of $\Delta .$ and its zero has multiplicity exactly two. Consequently, for every $i = 1 , \ldots , N$

$$
\Delta ^ { \prime } ( x _ { i } ) = 0 , \qquad \Delta ^ { \prime \prime } ( x _ { i } ) < 0 .
$$

Proposition 3. Suppose $k$ is real, analytic, and universal, or that k is ETP and the problem is well specified. Then, at any nonzero local minimum of ${ \bar { F } } ,$ , every weight vector w attaining the minimum in (4) satisfies $w _ { i } \neq 0$ for all $i = 1 , \ldots , N$ . Moreover, the particle locations are distinct.

Proof. Suppose $w _ { i } = 0$ . Since we are considering a local minimum with $\bar { F } ( { \pmb x } ) > 0$ , we have $\| \Delta \| _ { \mathcal { H } _ { k } } ^ { 2 } = \bar { F } ( \pmb { x } ) > 0$ Thus, $\Delta$ is not identically zero. It follows that, for every $\varepsilon > 0$ , there exists an $x ^ { \prime }$ in an ε-neighbourhood of $x _ { i }$ such that $\boldsymbol { x ^ { \prime } } \neq \boldsymbol { x } _ { j }$ for all $j = 1 , \ldots , N$ and $\Delta ( x ^ { \prime } ) \neq 0$ . This follows from analyticity when the kernel is analytic; alternatively, when the kernel is ETP and the problem is well specified, $\Delta$ is a nonzero linear combination of finitely many kernel sections and therefore has only finitely many zeros.

Fix such a point $x ^ { \prime } .$ . Since $w _ { i } = 0$ , moving the ith particle to $x ^ { \prime }$ leaves the measure unchanged if its weight remains zero. Reoptimising the weights at the new configuration can only decrease the objective relative to keeping the other weights fixed and optimising only the weight of the moved particle. Thus, using the reproducing property, we have that

$$
\begin{array} { r l } & { \bar { F } ( x _ { 1 } , \dots , x _ { i - 1 } , x ^ { \prime } , x _ { i + 1 } , \dots , x _ { N } ) \leq \underset { w _ { i } ^ { \prime } \in \mathbb { R } } { \operatorname* { m i n } } \| \Delta + w _ { i } ^ { \prime } k ( x ^ { \prime } , \cdot ) \| _ { \mathcal { H } _ { k } } ^ { 2 } } \\ & { \qquad = \bar { F } ( { \pmb x } ) + \underset { w _ { i } ^ { \prime } \in \mathbb { R } } { \operatorname* { m i n } } \left\{ 2 w _ { i } ^ { \prime } \Delta ( x ^ { \prime } ) + ( w _ { i } ^ { \prime } ) ^ { 2 } k ( x ^ { \prime } , x ^ { \prime } ) \right\} } \\ & { \qquad = \bar { F } ( { \pmb x } ) - \frac { \Delta ( x ^ { \prime } ) ^ { 2 } } { k ( x ^ { \prime } , x ^ { \prime } ) } < \bar { F } ( { \pmb x } ) . } \end{array}
$$

Since $x ^ { \prime }$ can be chosen arbitrarily close to $x _ { i }$ , this contradicts local minimality. Finally, suppose two particle locations coincide, say $x _ { i } = x _ { j }$ . We can then transfer the weight of particle i to particle $j ,$ setting $w _ { j }  w _ { j } + w _ { i }$ and $w _ { i } \gets 0 .$ , without changing the measure or the objective. This yields an optimal weight vector with a zero entry, contradicting the preceding argument. □

Proof. Define the reduced measure as in the proof of Prop. 1. Without loss of generality, suppose the collision occurs at $\vartheta _ { 1 }$ . Consider perturbing some, but not all, of the particles positioned at $\vartheta _ { 1 }$ to $\vartheta _ { 1 } + d ,$ , for suficiently small $d > 0$ . Suppose the resulting locations $( \vartheta _ { 1 } + d , \vartheta _ { 1 } , \ldots , \vartheta _ { A } )$ have weights $( \varsigma _ { 0 } ^ { d } , \ldots , \varsigma _ { A } ^ { d } )$ . Then

$$
\Delta _ { d } ( x ) = \varsigma _ { 0 } ^ { d } k ( \vartheta _ { 1 } + d , x ) + \varsigma _ { 1 } ^ { d } k ( \vartheta _ { 1 } , x ) + \sum _ { i = 2 } ^ { A } \varsigma _ { i } ^ { d } k ( \vartheta _ { i } , x ) - \mu _ { P ^ { * } } ( x ) .
$$

Following Batenkov and Yomdin (2014) and Evans et al. (1980), introduce the basis functions $\phi _ { i } ^ { d } ( \cdot ) = k ( \vartheta _ { i } , \cdot )$ for $i = 1 , \ldots , A$ , together with $\begin{array} { r } { \phi _ { 0 } ^ { d } = \frac { k ( \array+ d , \cdot ) - k ( \vartheta _ { 1 } , \cdot ) } { d } } \end{array}$ . Define $\begin{array} { r } { b _ { 0 } ^ { d } = \frac { \mu _ { P ^ { * } } ( \vartheta _ { 1 } + d ) - \mu _ { P ^ { * } } ( \vartheta _ { 1 } ) } { d } } \end{array}$ , and set $b _ { i } ^ { d } = \mu _ { P ^ { * } } ( \vartheta _ { i } )$ for $i = 1 , \ldots , A$ . We can then write

$$
\Delta _ { d } ( x ) = d \varsigma _ { 0 } ^ { d } \phi _ { 0 } ^ { d } ( x ) + ( \varsigma _ { 0 } ^ { d } + \varsigma _ { 1 } ^ { d } ) \phi _ { 1 } ^ { d } ( x ) + \sum _ { i = 2 } ^ { A } \varsigma _ { i } ^ { d } \phi _ { i } ^ { d } ( x ) - \mu _ { P ^ { * } } ( x ) .
$$

Now define

$$
\mathbf { v } ^ { d } = \left[ \begin{array} { c } { d \varsigma _ { 0 } ^ { d } } \\ { \varsigma _ { 0 } ^ { d } + \varsigma _ { 1 } ^ { d } } \\ { \varsigma _ { 2 } ^ { d } } \\ { \vdots } \\ { \varsigma _ { A } ^ { d } } \end{array} \right] .
$$

This is an equivalent representation of the optimised weights. For $d > 0$ , the map between $\mathbf { v } ^ { d }$ and $\varsigma ^ { d }$ is invertible, so optimising either $\mathrm { g i r }$ es the same result. Define the matrix $K ^ { d }$ by $K _ { i j } ^ { d } = \langle \phi _ { i } ^ { d } , \phi _ { j } ^ { d } \rangle _ { \mathcal { H } }$ . By optimality of the weights, we have $\Delta _ { d } ( \vartheta _ { 1 } + d ) = 0$ and $\Delta _ { d } ( \vartheta _ { i } ) = 0$ for $i = 1 , \ldots , A .$ . Consequently,

$$
K ^ { d } \mathbf { v } ^ { d } - \mathbf { b } ^ { d } = \left[ \begin{array} { c } { \frac { \Delta _ { d } ( \vartheta _ { 1 } + d ) - \Delta _ { d } ( \vartheta _ { 1 } ) } { d } } \\ { \Delta _ { d } ( \vartheta _ { 1 } ) } \\ { \vdots } \\ { \Delta _ { d } ( \vartheta _ { A } ) } \end{array} \right] = \mathbf { 0 } .
$$

Define the limiting basis by $\phi _ { 0 } ^ { 0 } = \partial _ { 1 } k ( \vartheta _ { 1 } , \cdot )$ , with the remaining basis functions unchanged. This also define $K ^ { 0 }$ and $ { \mathbf { b } } ^ { 0 }$ . Since k is ETP, it has the smoothness required here, and hence $K ^ { d } \to K ^ { 0 }$ and $ { \mathbf { b } } ^ { d } \to  { \mathbf { b } } ^ { 0 }$ as $d \to 0$ . In this limiting basis, $( 0 , \varsigma ) ^ { \top }$ represents the original reduced measure, and the reproducing property gives

$$
K ^ { 0 } ( 0 , \varsigma ) ^ { \top } - { \bf b } ^ { 0 } = \left[ \begin{array} { c } { \Delta ^ { \prime } ( \vartheta _ { 1 } ) } \\ { \Delta ( \vartheta _ { 1 } ) } \\ { \vdots } \\ { \Delta ( \vartheta _ { A } ) } \end{array} \right] = { \bf 0 } ,
$$

where the final equality follows from stationarity at the original configuration. Since $K ^ { 0 }$ is invertible by the ETP property, $K ^ { d }$ is also invertible for suficiently small $d ,$ and hence $\mathbf { v } ^ { d } = ( K ^ { d } ) ^ { - 1 } \mathbf { b } ^ { d }  ( 0 , \varsigma ) ^ { \top }$ as $d \to 0$

We next establish continuity of the basis functions and their derivatives as $d \to 0$ . Following Baxter and Brummelhuis (2011), observe that

$$
\phi _ { 0 } ^ { d } ( x ) = \int _ { 0 } ^ { 1 } \partial _ { 1 } k ( \vartheta _ { 1 } + t d , x ) \mathrm { d } t .
$$

By smoothness of the kernel, we can diferentiate under the integral to obtain

$$
\partial _ { x } ^ { n } \phi _ { 0 } ^ { d } ( x ) = \int _ { 0 } ^ { 1 } \partial _ { x } ^ { n } \partial _ { 1 } k ( \vartheta _ { 1 } + t d , x ) \mathrm { d } t .
$$

Since the integrand is continuous in the relevant arguments, these derivatives extend continuously to $d = 0$ . The remaining basis functions are independent of $d .$ It follows that

$$
\Delta _ { d } ^ { \prime \prime } ( x ) = \sum _ { i = 0 } ^ { A } v _ { i } ^ { d } \partial _ { x } ^ { 2 } \phi _ { i } ^ { d } ( x ) - \mu _ { P ^ { * } } ^ { \prime \prime } ( x )  \Delta ^ { \prime \prime } ( x ) ,
$$

and the extension defined by $\Delta _ { 0 } ^ { \prime \prime } ( x ) = \Delta ^ { \prime \prime } ( x )$ is jointly continuous in d and x. This provides the continuity needed for the instability argument. Finally, by the mean value theorem,

$$
\dot { d } = - \big ( \Delta _ { d } ^ { \prime } ( \vartheta _ { 1 } + d ) - \Delta _ { d } ^ { \prime } ( \vartheta _ { 1 } ) \big ) = - \Delta _ { d } ^ { \prime \prime } ( \varrho ) d ,
$$

for some $\varrho \in ( \vartheta _ { 1 } , \vartheta _ { 1 } + d )$ . By continuity, $\Delta _ { d } ^ { \prime \prime } ( \varrho ) < 0$ in a suficiently small neighbourhood of the stationary configuration along the perturbation considered above. Thus $\dot { d }$ has the same sign as $d ,$ establishing instability of the collision. Note, in particular, that any collided particles remain collided under the flow. Thus, along the perturbation we retain the negativity when the other particles change positions too, and for a suficiently small neighbourhood no other collisions occur. □

Theorem 5. Let $D > 0$ and $\varepsilon > 0$ . There exists a finite threshold $N _ { \varepsilon } ( D )$ such that, uniformly over all target distributions with compact support of diameter at most $D ,$ whenever $N > N _ { \varepsilon } ( D )$ every local minimum of $\bar { F }$ satisfies

$$
\mathrm { M M D } ( P , P ^ { * } ) < \varepsilon .
$$

Moreover, every stationary point with $\mathrm { M M D } ( P , P ^ { * } ) \ge \varepsilon$ is unstable under the unweighted particle gradient flow.

Proof. Fix a target distribution $P ^ { * }$ with compact support in $[ A , B ]$ , where $B - A \leq D$ . By Prop. 1, all particle locations lie within this interval and have positive total weight. Moreover, if $\mathrm { M M D } \ \geq \ \varepsilon .$ , then there exists $x _ { 0 } \in [ A , B ]$ such that $| \Delta ( x _ { 0 } ) | \ge \varepsilon ^ { 2 }$ . Indeed, stationarity gives $\begin{array} { r } { \int \Delta ( x ) \mathrm { d } P ( x ) = 0 } \end{array}$ , and hence $\varepsilon ^ { 2 } \le \mathrm { M M D ^ { 2 } = }$ $\begin{array} { r l } { - \int \Delta ( x ) \mathrm { d } P ^ { * } ( x ) } \end{array}$ . The claim then follows since $P ^ { * }$ is a probability measure.

We next bound the analytic continuation of $\Delta$ to the complex plane, centred at $x _ { 0 } .$ , following the approach of Gr¨ochenig et al. (2020, Lemma 4.3). By Prop. 1, $P = | P |$ , and hence $| P - P ^ { * } | \leq P + P ^ { * }$ . Moreover, by the same proposition we have $\Delta = \mu _ { P } - \mu _ { P ^ { * } } \leq 0$ , so $\mu _ { P } \leq \mu _ { P ^ { * } } \leq 1$ , since $P ^ { * }$ is a probability measure and the kernel is bounded above by one. It follows that,

$$
\begin{array} { l } { \displaystyle | \Delta ( x _ { 0 } + x + i y ) | = \left| \int e ^ { - ( x _ { 0 } + x + i y - t ) ^ { 2 } } \mathrm { d } ( P - P ^ { * } ) ( t ) \right| } \\ { \displaystyle \qquad \leq e ^ { y ^ { 2 } } \int e ^ { - ( x _ { 0 } + x - t ) ^ { 2 } } \mathrm { d } ( P + P ^ { * } ) ( t ) \leq e ^ { y ^ { 2 } } \big ( \mu _ { P } ( x + x _ { 0 } ) + \mu _ { P ^ { * } } ( x + x _ { 0 } ) \big ) \leq 2 e ^ { y ^ { 2 } } . } \end{array}
$$

We can therefore apply Prop. 30 with $f = \Delta ( \cdot + x _ { 0 } ) , A _ { f } = 2$ , and $B _ { f } = 1$ . In particular, since each distinct particle location contributes at least two zeros, counted with multiplicity, we obtain

$$
N ^ { \prime } \leq \frac { \log ( \frac { 2 } { \varepsilon ^ { 2 } } ) + \frac 1 2 \delta ^ { 2 } ( B - A ) ^ { 2 } } { 2 \log ( \delta ) } \leq \frac { \log ( \frac { 2 } { \varepsilon ^ { 2 } } ) + \frac 1 2 \delta ^ { 2 } D ^ { 2 } } { 2 \log ( \delta ) } ,
$$

where $N ^ { \prime }$ denotes the number of distinct particle locations. If N exceeds this bound, then at least two particles must share a location. By Props. 2 and $^ { 3 , }$ such configurations cannot be local minima and are unstable under the unweighted particle gradient flow. □

Optimising the bound above, we obtain an asymptotic rate as $\varepsilon \to 0$ of $O \big ( \frac { \log ( 1 / \varepsilon ) } { \log \log ( 1 / \varepsilon ) } \big )$ . We can see this by setting, for suficiently small $\varepsilon , \delta = \sqrt { - \log ( \varepsilon ) }$ , which gives

$$
N ^ { \prime } \leq \frac { \log 2 + 2 \log ( 1 / \varepsilon ) + 0 . 5 D ^ { 2 } \log ( 1 / \varepsilon ) } { \log \log ( 1 / \varepsilon ) } = O ( \frac { \log ( 1 / \varepsilon ) } { \log \log ( 1 / \varepsilon ) } ) .
$$

## B.2 Simplex Variant

Proposition 8. Let $\bar { x } _ { 1 } , \ldots , \bar { x } _ { K }$ and $x _ { 1 } ^ { * } , \ldots , x _ { K } ^ { * }$ each consist of distinct real numbers. Define

$$
H ( \cdot ) = \sum _ { i = 1 } ^ { K } \left[ \alpha _ { i } k ( \cdot , \bar { x } _ { i } ) + \beta _ { i } \partial _ { 2 } k ( \cdot , \bar { x } _ { i } ) \right] .
$$

Then there exist $\alpha , \beta$ such that $H ( x ) < 1$ for all $x \not \in \{ x _ { 1 } ^ { * } , . . . , x _ { K } ^ { * } \}$ , and

$$
H ( x _ { i } ^ { * } ) = 1 , \qquad H ^ { \prime } ( x _ { i } ^ { * } ) = 0 \qquad i = 1 , \ldots , K .
$$

Proof. First, the interpolation conditions follow by considering the matrix

$$
M ( { \pmb x } ^ { * } , \bar { \pmb x } ) = \left[ \begin{array} { c c } { { k ( { \pmb x } ^ { * } , \bar { \pmb x } ) } } & { { \partial _ { 2 } k ( { \pmb x } ^ { * } , \bar { \pmb x } ) } } \\ { { \partial _ { 1 } k ( { \pmb x } ^ { * } , \bar { \pmb x } ) } } & { { \partial _ { 1 } \partial _ { 2 } k ( { \pmb x } ^ { * } , \bar { \pmb x } ) } } \end{array} \right] .
$$

In particular, it sufices to find coeficients $\alpha , \beta$ satisfying

$$
M ( { \pmb x } ^ { * } , { \bar { \pmb x } } ) \left[ _ { \beta } ^ { \alpha } \right] = \left[ _ { \bf 0 } ^ { \bf 1 } \right] .
$$

Since the Gaussian kernel is ETP, the matrix $M ( { \pmb x } ^ { * } , { \pmb \bar { x } } )$ is invertible, and so such coeficients exist. It remains now to establish that $H ( x ) < 1$ away from the target locations. By the interpolation conditions, $H - 1$ has at least 2K zeros, counted with multiplicity. Multiplication by $- e ^ { x ^ { 2 } }$ preserves these zeros and their multiplicities. Define

$$
\tilde { \alpha } _ { i } = ( - \alpha _ { i } + 2 \beta _ { i } \bar { x } _ { i } ) e ^ { - \bar { x } _ { i } ^ { 2 } } , \qquad \tilde { \beta } _ { i } = - 2 \beta _ { i } e ^ { - \bar { x } _ { i } ^ { 2 } } , \qquad s _ { i } = 2 \bar { x } _ { i } ,
$$

so that

$$
- e ^ { x ^ { 2 } } ( H ( x ) - 1 ) = \sum _ { i = 1 } ^ { K } ( \tilde { \alpha } _ { i } + \tilde { \beta } _ { i } x ) e ^ { s _ { i } x } + e ^ { x ^ { 2 } } .
$$

By Lemma 2.6 of Schiebinger et al. (2018), this function, and hence also $H - 1$ , has at most 2K zeros, counted with multiplicity. Thus, the prescribed target locations are the only zeros of $H - 1$ , each with multiplicity exactly two. Since $H ( x ) \to 0$ as $x \to \pm \infty$ , and $H - 1$ cannot change sign at these zeros, it follows that $H ( x ) < 1$ everywhere else.

We remark that an alternative interpretation of Lemma 2.6 of Schiebinger et al. (2018) is that the Gaussian kernel, together with its derivatives and the constant function, forms a Chebyshev system. □

Proposition 9. Suppose we are at a stationary point of F with $F ( \mathbf { w } , \pmb { x } ) > 0$ . Then $0 < c < 1$

Proof. By Prop. 8, we can construct coeficients $\alpha ^ { \prime } , \beta ^ { \prime }$ such that the function $\begin{array} { r } { H ( \cdot ) ~ = ~ \sum _ { i = 1 } ^ { A } ( \alpha _ { i } ^ { \prime } k ( \cdot , \vartheta _ { i } ) + } \end{array}$ $\beta _ { i } ^ { \prime } \partial _ { 2 } k ( \cdot , \vartheta _ { i } ) )$ satisfies $H ( \vartheta _ { i } ) = 1 , H ^ { \prime } ( \vartheta _ { i } ) = 0$ for all $i = 1 , \ldots , A$ , and $H ( x ) < 1$ for all $x \not \in \{ \vartheta _ { 1 } , . . . , \vartheta _ { A } \}$

At a stationary point with $F ( \mathbf { w } , \pmb { x } ) > 0$ , we must have supp(P<sup>∗</sup>) ̸⊆ supp(P). Indeed, if this inclusion held, the target could be represented exactly at the current particle locations, and convexity of the weight optimisation problem would imply that the stationary weights achieve $F ( \mathbf { w } , \pmb { x } ) = 0$ . Since H equals one at every active particle location and is strictly less than one elsewhere, it follows that

$$
\int H ( x ) \mathrm { d } P ( x ) = \sum _ { j = 1 } ^ { A } \varsigma _ { j } H ( \vartheta _ { j } ) = 1 , \qquad \int H ( x ) \mathrm { d } P ^ { * } ( x ) < 1 .
$$

Hence, in particular,

$$
\int H ( x ) \mathrm { d } ( P - P ^ { * } ) ( x ) > 0 .\tag{9}
$$

Meanwhile, the stationarity conditions (5) give

$$
\int H ( x ) \mathrm { d } ( P - P ^ { * } ) ( x ) = \sum _ { i = 1 } ^ { A } ( \alpha _ { i } ^ { \prime } \Delta ( \vartheta _ { i } ) + \beta _ { i } ^ { \prime } \Delta ^ { \prime } ( \vartheta _ { i } ) ) = c \sum _ { i = 1 } ^ { A } \alpha _ { i } ^ { \prime } .
$$

Moreover, by the reproducing property and the interpolation conditions,

$$
\| H \| _ { \mathcal { H } } ^ { 2 } = \left[ ( { \pmb \alpha } ^ { \prime } ) ^ { \top } \quad ( \beta ^ { \prime } ) ^ { \top } \right] M ( \pmb \vartheta , \pmb \vartheta ) \left[ { \pmb \alpha } ^ { \prime } \right] = \sum _ { i = 1 } ^ { A } \alpha _ { i } ^ { \prime } > 0 ,
$$

where $M ( \cdot , \cdot )$ is as defined in the proof of Prop. 8. Combining these identities with (9) gives $c > 0$ . Finally, since the Gaussian kernel is strictly positive and bounded above by one, we have $\mu _ { P ^ { * } } ( \vartheta _ { i } ) > 0$ and $\mu _ { P } ( \vartheta _ { i } ) \leq 1$ . Thus $c = \mu _ { P } ( \vartheta _ { i } ) - \mu _ { P ^ { * } } ( \vartheta _ { i } ) < \mu _ { P } ( \vartheta _ { i } ) \leq 1$ , establishing $0 < c < 1$ □

Proposition 10. Suppose $( \mathbf { w } , \pmb { x } )$ is a stationary point of $F$ with $F ( \mathbf { w } , \pmb { x } ) > 0$ , with weights constrained to the simplex. Suppose also that k is $E T P ,$ translation invariant, and normalisable, and that $c \geq 0$ . Then $\Delta ( x ) \leq c$ for all x, every distinct particle location is active, and for every $i = 1 , \ldots , N$

$$
\Delta ( x _ { i } ) = c , \qquad \Delta ^ { \prime } ( x _ { i } ) = 0 , \qquad \Delta ^ { \prime \prime } ( x _ { i } ) < 0 .
$$

Proof. As before, we begin by defining the reduced measure of $P$ by merging particles at the same location and removing locations with zero total weight. Write $\vartheta _ { 1 } , \ldots , \vartheta _ { A }$ for its distinct locations and $\varsigma _ { 1 } , \ldots , \varsigma _ { A }$ for their nonzero total weights. By the stationarity conditions, we have that

$$
\Delta ( \vartheta _ { j } ) = c , \qquad \Delta ^ { \prime } ( \vartheta _ { j } ) = 0 , \qquad j = 1 , \ldots , A .
$$

Let $h = \Delta - c .$ . To apply the variation diminishing property to this function, we first represent the constant c as a kernel integral using translation invariance and normalisability. Let λ denote Lebesgue measure and $\begin{array} { r } { \mathcal { Z } = \int _ { \mathbb { R } } k ( t , x ) \mathrm { d } t } \end{array}$ , which is finite, positive, and independent of x. If $c = 0$ , write

$$
h ( x ) = \int k ( t , x ) g ( t ) \mathrm { d } \sigma ( t ) , \qquad g = \frac { \mathrm { d } ( P - P ^ { * } ) } { \mathrm { d } \sigma } , \qquad \sigma = | P | + P ^ { * } .
$$

If $c > 0$ , instead write

$$
h ( x ) = \int k ( t , x ) g ( t ) \mathrm { d } \sigma ( t ) , \qquad g = \frac { \mathrm { d } ( P - P ^ { * } - \frac { c } { \mathcal { Z } } \lambda ) } { \mathrm { d } \sigma } , \qquad \sigma = | P | + P ^ { * } + \lambda .
$$

Since the positive part of $g$ is supported on at most A distinct locations, we must have $V ( g ) \leq 2 A$ . Moreover, since $F ( \mathbf { w } , \pmb { x } ) > 0$ and $P , P ^ { * }$ have equal mass, we have that $\begin{array} { r } { \int h \mathrm { d } ( P - P ^ { * } ) = \| \Delta \| _ { \mathcal { H } _ { \iota } } ^ { 2 } > 0 . } \end{array}$ , so h is not identically zero. The variation diminishing property (cf. Def. 26) therefore gives $Z ( h ) \leq { \dot { V } } { \overset { \sim } { ( } } g )$ , where zeros are counted with multiplicity. On the other hand, the stationarity conditions imply that each $\vartheta _ { j }$ is a zero of multiplicity at least two. Consequently,

$$
2 A \leq Z ( h ) \leq V ( g ) \leq 2 A .
$$

Thus, all inequalities are in fact equalities, and the $\vartheta _ { j }$ are the only zeros of h, each with multiplicity exactly two. Since the positive part of $g$ is supported on at most A locations, equality $V ( g ) = 2 A$ requires the alternating sign pattern negative–positive–negative–· · · –positive–negative. Thus, every location in the reduced measure must have strictly positive total weight.

We next establish that $h \leq 0$ . Since all zeros of h have multiplicity exactly two, h cannot change sign. Furthermore, the stationarity conditions give $\textstyle \int h \mathrm { d } P = 0$ , and therefore

$$
\int h \mathrm { d } P ^ { * } = - \int h \mathrm { d } ( P - P ^ { * } ) = - \| \Delta \| _ { \mathcal { H } _ { k } } ^ { 2 } < 0 .
$$

It follows that $h \leq 0$ everywhere. Finally, the KKT conditions give $h ( x _ { i } ) = \Delta ( x _ { i } ) - c \geq 0$ for every particle, including those with zero individual weight. Since $h \leq 0$ , we must thus have $h ( x _ { i } ) = 0 .$ . Since all zeros of h occur at the active locations $\vartheta _ { 1 } , \ldots , \vartheta _ { A }$ , every particle must coincide with one of these locations. Hence every distinct particle location is active and has positive total weight. Moreover, each particle location is a strict local maximum of $h ,$ and its zero has multiplicity exactly two. Thus, since $h = \Delta - c ,$ it follows that, for every $i = 1 , \ldots , N .$

$$
\Delta ( x _ { i } ) = c , \qquad \Delta ^ { \prime } ( x _ { i } ) = 0 , \qquad \Delta ^ { \prime \prime } ( x _ { i } ) < 0 .
$$

Proposition 16. Suppose that the problem is well specified, with $\begin{array} { r } { P ^ { * } = \sum _ { j = 1 } ^ { K } w _ { j } ^ { * } \delta _ { x _ { j } ^ { * } } } \end{array}$ . Then there exists $r ( P ^ { * } ) > 0$ such that every spurious stationary point satisfies

$$
\operatorname* { m a x } _ { j } \operatorname* { m i n } _ { i } \left| x _ { i } - x _ { j } ^ { * } \right| \geq r ( P ^ { * } ) .
$$

Proof. First, if $K > 1$ , it sufices to consider configurations satisfying max<sub>j</sub> min<sub>i</sub> $\begin{array} { r } { | x _ { i } - x _ { j } ^ { * } | < \frac { 1 } { 2 } } \end{array}$ min $_ { i \neq j } \left| x _ { i } ^ { * } - x _ { j } ^ { * } \right|$ since otherwise the desired separation bound already holds. For each target location $x _ { i } ^ { * } .$ , let $\bar { x } _ { i }$ denote a nearest particle location. The separation condition ensures that the $\bar { x } _ { i }$ are distinct. By Prop. 8, we can construct

coeficients $\alpha , \beta$ such that the function $\begin{array} { r } { H ( \cdot ) = \sum _ { i = 1 } ^ { K } ( \alpha _ { i } k ( \cdot , \bar { x } _ { i } ) + \beta _ { i } \partial _ { 2 } k ( \cdot , \bar { x } _ { i } ) ) } \end{array}$ satisfies $H ( x _ { i } ^ { * } ) = 1 , H ^ { \prime } ( x _ { i } ^ { * } ) = 0$ for all $i = 1 , \ldots , K$ , and $H ( x ) < 1$ for all $x \not \in \{ x _ { 1 } ^ { * } , . . . , x _ { K } ^ { * } \}$ . Then, as in the proof of Prop. 9, we have

$$
\int H ( x ) \mathrm { d } P ( x ) = \sum _ { j = 1 } ^ { N } w _ { j } H ( x _ { j } ) \leq 1 , \qquad \int H ( x ) \mathrm { d } P ^ { * } ( x ) = \sum _ { i = 1 } ^ { K } w _ { i } ^ { * } H ( x _ { i } ^ { * } ) = 1 .
$$

Hence, at a spurious stationary point,

$$
\int H ( x ) \mathrm { d } ( P - P ^ { * } ) ( x ) \leq 0 .\tag{10}
$$

Meanwhile, Prop. 10 gives $\Delta ( { \bar { x } } _ { i } ) = c$ and $\Delta ^ { \prime } ( \bar { x } _ { i } ) = 0$ , and therefore

$$
\int H ( x ) \mathrm { d } ( P - P ^ { * } ) ( x ) = \sum _ { i = 1 } ^ { K } ( \alpha _ { i } \Delta ( \bar { x } _ { i } ) + \beta _ { i } \Delta ^ { \prime } ( \bar { x } _ { i } ) ) = c \sum _ { i = 1 } ^ { K } \alpha _ { i } .
$$

Since $c > 0$ by Prop. 9, it sufices to establish that $\textstyle \sum _ { i = 1 } ^ { K } \alpha _ { i } > 0$ whenever $\bar { \mathbf { x } }$ is suficiently close to $\pmb { x } ^ { * }$ . At $\bar { \boldsymbol { x } } = \boldsymbol { x } ^ { * }$ the reproducing property and interpolation conditions give

$$
\| H \| _ { \mathcal { H } } ^ { 2 } = \left[ \pmb { \alpha } ^ { \top } \quad \pmb { \beta } ^ { \top } \right] M ( \pmb { x } ^ { * } , \pmb { x } ^ { * } ) \left[ \pmb { \alpha } \right] = \sum _ { i = 1 } ^ { K } \alpha _ { i } > 0 ,
$$

where $M ( \cdot , \cdot )$ is as defined in the proof of Prop. 8. By continuity of the interpolation coeficients, this positivity persists in a suficiently small target-dependent neighbourhood of $\pmb { x } ^ { * }$ . Within this neighbourhood, we therefore obtain a contradiction to (10). Consequently, no spurious stationary point can lie in this neighbourhood.

We now make the neighbourhood explicit using standard matrix perturbation bounds. By the interpolation conditions,

$$
{ \bf \Lambda } \left[ \begin{array} { l } { { \alpha } } \\ { { \beta } } \end{array} \right] = M ( { \bf x } ^ { * } , \bar { \bf x } ) ^ { - 1 } \left[ \begin{array} { l } { { \bf 1 } } \\ { { \bf 0 } } \end{array} \right] ,
$$

where

$$
M ( { \pmb x } ^ { * } , \bar { \pmb x } ) = \left[ \begin{array} { c c } { { k ( { \pmb x } ^ { * } , \bar { \pmb x } ) } } & { { \partial _ { 2 } k ( { \pmb x } ^ { * } , \bar { \pmb x } ) } } \\ { { \partial _ { 1 } k ( { \pmb x } ^ { * } , \bar { \pmb x } ) } } & { { \partial _ { 1 } \partial _ { 2 } k ( { \pmb x } ^ { * } , \bar { \pmb x } ) } } \end{array} \right] .
$$

Thus, writing $e _ { 1 } = { \bigg [ } { \frac { \mathbf { 1 } } { \mathbf { 0 } } } { \bigg ] }$ , we have that

$$
T ( \bar { \pmb x } ) = \sum _ { i = 1 } ^ { K } \alpha _ { i } = e _ { 1 } ^ { \top } M ( { \pmb x } ^ { * } , \bar { \pmb x } ) ^ { - 1 } e _ { 1 } .
$$

We seek an explicit radius $r ( P ^ { * } ) > 0$ such that $T ( \bar { \pmb { x } } ) > 0$ whenever $\delta ( \bar { \pmb { x } } ) = \| \bar { \pmb { x } } - \pmb { x } ^ { * } \| _ { \infty } < r ( P ^ { * } )$ . For brevity, write $M ( \bar { \pmb x } ) = M ( { \pmb x } ^ { * } , \bar { \pmb x } ) , M ^ { * } = M ( { \pmb x } ^ { * } , { \pmb x } ^ { * } )$ , and $T ^ { * } = T ( x ^ { * } )$ . Notice that $T ^ { * } \geq 1$ . Define $E ( \bar { \pmb { x } } ) = M ( \bar { \pmb { x } } ) - M ^ { * }$ and $q ( \bar { \pmb { x } } ) = \| ( M ^ { * } ) ^ { - 1 } E ( \bar { \pmb { x } } ) \| _ { \infty } . \ \mathrm { B y }$ the Neumann series, whenever $q ( { \bar { x } } ) < 1$

$$
\| ( M ^ { \ast } ) ^ { - 1 } - M ( \bar { \pmb x } ) ^ { - 1 } \| _ { \infty } \leq \frac { q ( \bar { \pmb x } ) } { 1 - q ( \bar { \pmb x } ) } \| ( M ^ { \ast } ) ^ { - 1 } \| _ { \infty } .
$$

Let $m ^ { - 1 } = \| ( M ^ { * } ) ^ { - 1 } \| _ { \infty }$ . Then $q ( { \bar { \pmb x } } ) \leq m ^ { - 1 } \| { \boldsymbol E } ( { \bar { \pmb x } } ) \| _ { \infty }$ . By standard derivative bounds for the Gaussian kerne and the mean value theorem,

$$
\| E ( { \bar { \mathbf { x } } } ) \| _ { \infty } \leq 6 K \delta ( { \bar { \mathbf { x } } } ) .
$$

Consequently, H¨older’s inequality gives

$$
| T ^ { * } - T ( \bar { \pmb { x } } ) | \leq \| e _ { 1 } \| _ { 1 } \| ( M ^ { * } ) ^ { - 1 } - M ( \bar { \pmb { x } } ) ^ { - 1 } \| _ { \infty } \| e _ { 1 } \| _ { \infty } \leq \frac { 6 m ^ { - 2 } K ^ { 2 } \delta ( \bar { \pmb { x } } ) } { 1 - 6 m ^ { - 1 } K \delta ( \bar { \pmb { x } } ) } .
$$

Since $T ^ { * } \geq 1$ , it sufices to ensure that the final bound is strictly less than one. Rearranging the above, we have that

$$
\delta ( \bar { \pmb { x } } ) < \frac { 1 } { 6 m ^ { - 1 } K ( m ^ { - 1 } K + 1 ) } ,
$$

which also guarantees $q ( { \bar { x } } ) < 1$ . Thus, we can take any $r ( P ^ { * } ) > 0$ satisfying

$$
r ( P ^ { * } ) < \operatorname* { m i n } \left\{ \frac { 1 } { 2 } \operatorname* { m i n } _ { i \neq j } | x _ { i } ^ { * } - x _ { j } ^ { * } | , \frac { 1 } { 6 m ^ { - 1 } K ( m ^ { - 1 } K + 1 ) } \right\} ,
$$

where the first term is interpreted as $+ \infty$ whenever $K = 1$

Proposition 13. Let $P , P ^ { * }$ be probability measures on $\mathbb { R } ^ { d }$ , let $\rho ~ = ~ P - P ^ { * }$ , and suppose $k$ is continuous, translation invariant, and has positive spectral density. Then, writing $\Delta = \mu _ { P } - \mu _ { P ^ { * } } , f o r$ every $\nu > 0$ it holds that

$$
\begin{array} { r l } & { \mathrm { M M D } ^ { 2 } ( P ^ { * } , P ) \le \displaystyle \frac { \| \Delta \| _ { 1 } ^ { 2 } } { ( 2 \pi ) ^ { d } } \int _ { | \omega | \le \nu } \frac { 1 } { \hat { k } ( \omega ) } \mathrm { d } \omega } \\ & { \qquad + \displaystyle \frac { \| \rho \| _ { \mathrm { T V } } ^ { 2 } } { ( 2 \pi ) ^ { d } } \int _ { | \omega | > \nu } \hat { k } ( \omega ) \mathrm { d } \omega . } \end{array}
$$

In particular, for the one-dimensional Gaussian kernel with $2 \sigma ^ { 2 } = 1$ ，

$$
\begin{array} { r } { \mathrm { M M D } ^ { 2 } ( P , P ^ { * } ) \leq \frac { \| \Delta \| _ { 1 } ^ { 2 } } { \pi } \operatorname { e r f i } \left( \frac { \nu } { 2 } \right) + 4 \operatorname { e r f c } \left( \frac { \nu } { 2 } \right) . } \end{array}
$$

where $\begin{array} { r } { \operatorname { e r f i } ( z ) = \frac { 2 } { \sqrt { \pi } } \int _ { 0 } ^ { z } e ^ { t ^ { 2 } } \mathrm { d } t , \operatorname { e r f c } ( z ) = \frac { 2 } { \sqrt { \pi } } \int _ { z } ^ { \infty } e ^ { - t ^ { 2 } } } \end{array}$ dt denote the imaginary error function and complementary error function, respectively.

Proof. Let $\rho = P - P ^ { * }$ and $\Delta = \mu _ { P } - \mu _ { P ^ { * } }$ . By the definition of the MMD, we have that

$$
\mathrm { M M D } ^ { 2 } ( P , P ^ { * } ) = \iint k ( x , y ) \mathrm { d } \rho ( x ) \mathrm { d } \rho ( y ) = \int \Delta ( x ) \mathrm { d } \rho ( x ) .
$$

Meanwhile, by Bochner’s theorem, it follows that

$$
\mathrm { M M D } ^ { 2 } ( P , P ^ { \ast } ) = \frac { 1 } { ( 2 \pi ) ^ { d } } \int \hat { k } ( \omega ) | \hat { \rho } ( \omega ) | ^ { 2 } \mathrm { d } \omega , \qquad \frac { | \hat { \Delta } ( \omega ) | ^ { 2 } } { \hat { k } ( \omega ) } = \hat { k } ( \omega ) | \hat { \rho } ( \omega ) | ^ { 2 } .
$$

Splitting the integral at frequency ν, and using $| \hat { \Delta } ( \omega ) | \leq \| \Delta \| _ { 1 }$ and $| \hat { \rho } ( \omega ) | \leq \| \rho \| _ { \mathrm { T V } }$ , gives

$$
\begin{array} { r l } & { \mathrm { M M D } ^ { 2 } ( P , P ^ { * } ) = \displaystyle \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { | \omega | \leq \nu } \frac { | \hat { \Delta } ( \omega ) | ^ { 2 } } { \hat { k } ( \omega ) } \mathrm { d } \omega + \frac { 1 } { ( 2 \pi ) ^ { d } } \int _ { | \omega | > \nu } \hat { k } ( \omega ) | \hat { \rho } ( \omega ) | ^ { 2 } \mathrm { d } \omega } \\ & { \qquad \leq \displaystyle \frac { \| \Delta \| _ { 1 } ^ { 2 } } { ( 2 \pi ) ^ { d } } \int _ { | \omega | \leq \nu } \frac { 1 } { \hat { k } ( \omega ) } \mathrm { d } \omega + \frac { \| \rho \| _ { \mathrm { T V } } ^ { 2 } } { ( 2 \pi ) ^ { d } } \int _ { | \omega | > \nu } \hat { k } ( \omega ) \mathrm { d } \omega . } \end{array}
$$

For the one-dimensional Gaussian kernel with $2 \sigma ^ { 2 } = 1$ , we have $\hat { k } ( \omega ) = \sqrt { \pi } e ^ { - \omega ^ { 2 } / 4 }$ . Moreover, since $P$ and $P ^ { * }$ are probability measures, $\| \rho \| _ { \mathrm { T V } } ^ { 2 } \le 4$ . Substituting into the preceding bound gives

$$
\mathrm { M M D } ^ { 2 } ( P , P ^ { * } ) \leq \frac { \| \Delta \| _ { 1 } ^ { 2 } } { 2 \pi \sqrt { \pi } } \int _ { | \omega | \leq \nu } e ^ { \omega ^ { 2 } / 4 } \mathrm { d } \omega + \frac { 2 } { \sqrt { \pi } } \int _ { | \omega | > \nu } e ^ { - \omega ^ { 2 } / 4 } \mathrm { d } \omega = \frac { \| \Delta \| _ { 1 } ^ { 2 } } { \pi } \mathrm { e r f i } \left( \frac { \nu } { 2 } \right) + 4 \mathrm { e r f c } \left( \frac { \nu } { 2 } \right) . \ D ^ { 2 } ( \omega , P ^ { * } ) < 4 \omega .
$$

Proposition 14. Suppose $P ^ { * }$ has compact support of diameter at most $D ,$ and $( \mathbf { w } , \pmb { x } )$ is a stationary point with $F ( \mathbf { w } , \pmb { x } ) > 0$ . Then

$$
\begin{array} { r l } & { \| \Delta \| _ { 1 } \leq 2 ( \sqrt { 2 } + D ) c + 1 6 c \sqrt { - \log c } } \\ & { \qquad + 6 \sqrt { \pi } \operatorname { e r f c } ( \sqrt { - \log c } ) + 4 \sqrt { \pi } c \sqrt { 1 - 2 \log c } . } \end{array}
$$

Proof. Since $\begin{array} { r } { \int _ { \mathbb { R } } \Delta ( x ) \mathrm { d } x = 0 } \end{array}$ , the positive and negative parts of $\Delta$ have equal integrals, and so

$$
\| \Delta \| _ { 1 } = 2 \int _ { - \infty } ^ { \infty } [ \Delta ] _ { + } .\tag{11}
$$

Write $\begin{array} { r } { \mu _ { P } ( \cdot ) = \sum _ { i } w _ { i } e ^ { - ( x _ { i } - \cdot ) ^ { 2 } } } \end{array}$ and $\begin{array} { r } { \mu _ { P ^ { * } } ( \cdot ) = \int e ^ { - ( x ^ { * } - \cdot ) ^ { 2 } } \mathrm { d } P ^ { * } ( x ^ { * } ) } \end{array}$ . Let $L , R$ denote the leftmost and rightmost particle locations, and $A , B$ the endpoints of the target support. We split the integral above into seven regions:

$$
( - \infty , L ] , \qquad [ L , L + \delta _ { L } ] , \qquad [ L + \delta _ { L } , A - \varepsilon _ { L } ] , \qquad [ A - \varepsilon _ { L } , B + \varepsilon _ { R } ] , \qquad [ B + \varepsilon _ { R } , R - \delta _ { R } ] , \qquad [ R - \delta _ { R } , R ] , \qquad [ R , \infty ) ,
$$

treating any reversed intervals as empty. We impose $\delta _ { L } , \delta _ { R } \geq 1 / \sqrt { 2 }$ and define $D _ { L } = A - L$ and $D _ { R } = R - B$ If $D _ { L } \leq \delta _ { L } .$ , we set $\varepsilon _ { L } = 0$ and omit the contribution from $[ L + \delta _ { L } , A ]$ , and similarly on the right if $D _ { R } \leq \delta _ { R }$ We proceed as follows

$( - \infty , L ] \colon$ By the maximum principle and the Gaussian tail bound,

$$
\int _ { - \infty } ^ { L } [ \Delta ] _ { + } \leq \int _ { - \infty } ^ { L } \operatorname* { m i n } \{ c , \mu _ { P } ( x ) \} \mathrm { d } x \leq \int _ { - \infty } ^ { 0 } \operatorname* { m i n } \{ c , e ^ { - x ^ { 2 } } \} \mathrm { d } x = c { \sqrt { - \log c } } + { \frac { \sqrt { \pi } } { 2 } } \operatorname { e r f c } ( { \sqrt { - \log c } } ) .
$$

$[ L , L + \delta _ { L } ]$ : Applying the maximum principle (cf. Prop. 10) gives

$$
\int _ { L } ^ { L + \delta _ { L } } [ \Delta ] _ { + } \leq c \delta _ { L } .
$$

$[ { \cal L } + \delta _ { L } , A - \varepsilon _ { L } ] ;$ Let $X \sim P$ and define $f ( x ) = x e ^ { - x ^ { 2 } }$ . Since f has a single maximum, we have

$$
\mathbb { P } ( L + \delta _ { L } \leq X \leq A - \varepsilon _ { L } ) \leq \frac { \mathbb { E } [ f ( X - L ) ] } { \operatorname* { m i n } \{ f ( \delta _ { L } ) , f ( A - \varepsilon _ { L } - L ) \} } \qquad \\  = \frac { \sum _ { i } w _ { i } f ( x _ { i } - L ) } { \operatorname* { m i n } \{ f ( \delta _ { L } ) , f ( A - \varepsilon _ { L } - L ) \} } = \frac { \int f ( x ^ { * } - L ) \mathrm { d } P ^ { * } ( x ^ { * } ) } { \operatorname* { m i n } \{ f ( \delta _ { L } ) , f ( A - \varepsilon _ { L } - L ) \} } ,
$$

where the final equality follows from stationarity at L. Since $D _ { L } \geq \delta _ { L } \geq 1 / \sqrt { 2 } .$ the function f is decreasing on $[ D _ { L } , \infty )$ , and hence $f ( x ^ { * } - L ) \leq f ( D _ { L } )$ for every $x ^ { * } \in [ A , B ]$ . Moreover, since $D _ { L } - \varepsilon _ { L } \geq \delta _ { L }$ , we have $f ( D _ { L } - \varepsilon _ { L } ) \leq f ( \delta _ { L } )$ ). Consequently,

$$
\mathbb { P } ( L + \delta _ { L } \leq X \leq A - \varepsilon _ { L } ) \leq \frac { D _ { L } } { D _ { L } - \varepsilon _ { L } } \exp ( - 2 \varepsilon _ { L } D _ { L } + \varepsilon _ { L } ^ { 2 } ) .
$$

It follows that

$$
\int _ { L + \delta _ { L } } ^ { A - \varepsilon _ { L } } [ \Delta ] _ { + } \leq \frac { D _ { L } } { D _ { L } - \varepsilon _ { L } } \exp ( - 2 \varepsilon _ { L } D _ { L } + \varepsilon _ { L } ^ { 2 } ) \sqrt { \pi } + 2 c \sqrt { - \log c } + \sqrt { \pi } \mathrm { e r f c } ( \sqrt { - \log c } ) ,
$$

where the final two terms account for contributions from particles outside the interval. These are obtained by separating the contributions to $\mu _ { P }$ from particles inside and outside the interval, and applying the preceding Gaussian tail estimate to the latter.

$[ A - \varepsilon _ { L } , B + \varepsilon _ { R } ] ;$ : Applying the maximum principle (cf. Prop. 10) again gives

$$
\int _ { A - \varepsilon _ { L } } ^ { B + \varepsilon _ { R } } [ \Delta ] _ { + } \leq c ( \varepsilon _ { L } + \varepsilon _ { R } + B - A ) .
$$

The remaining intervals $[ B + \varepsilon _ { R } , R - \delta _ { R } ] , [ R - \delta _ { R } , R ]$ , and $[ R , \infty )$ are bounded symmetrically. Combining these estimates yields

$$
\begin{array} { r l r } {  { \int _ { - \infty } ^ { \infty } [ \Delta ] _ { + } \le c \big ( \delta _ { L } + \delta _ { R } + B - A \big ) + 6 ( c \sqrt { - \log c } + \frac { \sqrt { \pi } } { 2 } \operatorname { e r f c } ( \sqrt { - \log c } ) ) } } \\ & { } & { \quad + c \varepsilon _ { L } + \frac { D _ { L } } { D _ { L } - \varepsilon _ { L } } \exp ( - 2 \varepsilon _ { L } D _ { L } + \varepsilon _ { L } ^ { 2 } ) \sqrt { \pi } } \\ & { } & { \quad + c \varepsilon _ { R } + \frac { D _ { R } } { D _ { R } - \varepsilon _ { R } } \exp ( - 2 \varepsilon _ { R } D _ { R } + \varepsilon _ { R } ^ { 2 } ) \sqrt { \pi } . } \end{array}
$$

We now choose $\delta _ { L } = \delta _ { R } = 1 / \sqrt { 2 }$ and bound the terms involving $\varepsilon _ { L }$ and $\varepsilon _ { R } .$ Consider first

$$
c \varepsilon _ { L } + { \frac { D _ { L } } { D _ { L } - \varepsilon _ { L } } } \exp ( - 2 \varepsilon _ { L } D _ { L } + \varepsilon _ { L } ^ { 2 } ) \sqrt { \pi } .
$$

If $D _ { L } ^ { 2 } \leq { \textstyle { \frac { 1 } { 2 } } } - \log c$ , we can bound the contribution from $[ L + \delta _ { L } , A ]$ directly using the maximum principle. Since this interval has length $D _ { L } - 1 / \sqrt { 2 }$ when nonempty, its contribution is bounded by

$$
c \left( D _ { L } - { \frac { 1 } { \sqrt { 2 } } } \right) \leq c { \sqrt { - \log c } } .
$$

Otherwise, choose $\varepsilon _ { L } = D _ { L } - \sqrt { D _ { L } ^ { 2 } + \log c } .$ Then $- 2 \varepsilon _ { L } D _ { L } + \varepsilon _ { L } ^ { 2 } = \log c ,$ , and $\varepsilon _ { L } \leq { \sqrt { - \log c } } .$ . Moreover, the ratio

$$
\frac { D _ { L } } { D _ { L } - \varepsilon _ { L } } = \frac { D _ { L } } { \sqrt { D _ { L } ^ { 2 } + \log c } }
$$

is decreasing in $D _ { L }$ , and is therefore bounded by its value at $\begin{array} { r } { D _ { L } ^ { 2 } = \frac { 1 } { 2 } - \log c } \end{array}$ , namely $\textstyle { \sqrt { 1 - 2 \log c } }$ . Thus, in either case, the corresponding contribution is bounded by

$$
c { \sqrt { - \log c } } + { \sqrt { \pi } } c { \sqrt { 1 - 2 \log c } } .
$$

Applying the same argument to the $\varepsilon _ { R }$ term, and using $B - A \leq D$ , gives

$$
\int _ { - \infty } ^ { \infty } [ \Delta ] _ { + } \leq ( { \sqrt { 2 } } + D ) c + 8 c { \sqrt { - \log c } } + 3 { \sqrt { \pi } } \operatorname { e r f c } ( { \sqrt { - \log c } } ) + 2 { \sqrt { \pi } } c { \sqrt { 1 - 2 \log c } } .
$$

Combining this with (11) gives the stated bound.

Proposition 17. Suppose a target atom is missed by a neighbourhood of radius r as in Prop. 16. Then

$$
\mathrm { M M D } ( P , P ^ { * } ) \geq \frac { [ w ^ { * } - ( a r ) ^ { - 2 m } ] _ { + } } { C ( a , m ) } ,
$$

where $[ u ] _ { + } = \operatorname* { m a x } \{ u , 0 \} , w ^ { * } =$ min<sub>j</sub> $\boldsymbol { w } _ { j } ^ { * }$ , and one may take

$$
C ( a , m ) = \frac { \sqrt { \pi } } { a } e ^ { \frac { 1 } { 2 } m ^ { 2 } a ^ { 2 } } , \qquad m \in \mathbb { N } _ { \geq 1 } , \qquad a > 0 .
$$

Thus, choosing a, m such that $( a r ) ^ { - 2 m } < w ^ { * }$ gives a strictly positive lower bound on the MMD.

Proof. We begin with the variational characterisation of the MMD, which gives, for any nonzero $q \in \mathcal H$

$$
\mathrm { M M D } ( P , P ^ { * } ) \geq \frac { \left| \sum _ { i } w _ { i } q ( x _ { i } ) - \sum _ { j } w _ { j } ^ { * } q ( x _ { j } ^ { * } ) \right| } { \| q \| _ { \mathcal { H } } } .
$$

Motivated by related constructions in de Castro et al. (2021) and Costarelli and Vinti (2021), we consider a Jackson-type kernel centred at a target location $\boldsymbol { x } _ { j } ^ { * }$

$$
q _ { j } ( x ) = \left( \frac { \sin ( a ( x - x _ { j } ^ { * } ) ) } { a ( x - x _ { j } ^ { * } ) } \right) ^ { 2 m } .
$$

We first bound its RKHS norm. Writing $\hat { q } _ { j }$ and $\hat { k }$ for the Fourier transforms of $q _ { j }$ and the Gaussian kernel, respectively, we have that

$$
\Vert q _ { j } \Vert _ { \mathcal { H } } ^ { 2 } = \frac { 1 } { 2 \pi } \int _ { \mathbb { R } } \frac { \vert \hat { q } _ { j } ( \omega ) \vert ^ { 2 } } { \hat { k } ( \omega ) } \mathrm { d } \omega .
$$

Since $\hat { k } ( \omega ) = \sqrt { \pi } e ^ { - \omega ^ { 2 } / 4 } , \| \hat { q } _ { j } \| _ { \infty } \leq \pi / a$ , and $\hat { q } _ { j }$ is supported on [−2ma, 2ma] as the 2m-fold convolution of a box function, we then obtain

$$
\| q _ { j } \| _ { \mathcal { H } } ^ { 2 } \leq \frac { \pi } { a ^ { 2 } } \mathrm { e r f i } ( m a ) \leq \frac { \pi } { a ^ { 2 } } e ^ { m ^ { 2 } a ^ { 2 } } .
$$

Thus $\| q _ { j } \| _ { \mathcal { H } } \leq C ( a , m )$ , with the bound independent of $j$ by translation invariance of the Gaussian kernel. We next use the fact that $q _ { j } ( x _ { j } ^ { * } ) = 1$ and $q _ { j } \geq 0$ to obtain

$$
\sum _ { k } w _ { k } ^ { * } q _ { j } ( x _ { k } ^ { * } ) \geq w _ { j } ^ { * } .
$$

Suppose the target atom $\boldsymbol { x } _ { j } ^ { * }$ is missed by a neighbourhood of radius r. Then $q _ { j } ( x _ { i } ) \leq ( a r ) ^ { - 2 m }$ for every particle location $x _ { i }$ , and hence, since the weights sum to one,

$$
\sum _ { i } w _ { i } q _ { j } ( x _ { i } ) \leq ( a r ) ^ { - 2 m } .
$$

Consequently, letting $\begin{array} { r } { w ^ { * } = \operatorname* { m i n } _ { j } w _ { j } ^ { * } } \end{array}$ , due to the variational bound we arrive at

$$
\mathrm { M M D } ( P , P ^ { * } ) \geq \frac { [ w _ { j } ^ { * } - ( a r ) ^ { - 2 m } ] _ { + } } { \| q _ { j } \| _ { \mathcal { H } } } \geq \frac { [ w ^ { * } - ( a r ) ^ { - 2 m } ] _ { + } } { C ( a , m ) } .
$$

Proposition 15. A positive lower bound on the KKT level c gives an upper bound on the number A of distinct active particle locations. One such bound is

$$
A \leq \frac { 2 - c } { c } \log \left( \frac { 2 ( c + 2 ) } { c } \right) + \frac { e } { 2 } \left( \frac { 8 - 5 c } { c } \right) ^ { 2 } \log \left( \frac { 2 } { c } \right) .
$$

Proof. Consider an interval of radius r around a particle location $x _ { i } ,$ and let $p _ { i , r } = P ( [ x _ { i } - r , x _ { i } + r ] )$ . Since $\mu _ { P } ( x _ { i } ) = \mu _ { P ^ { * } } ( x _ { i } ) + c \geq c ;$ , we have that

$$
\mu _ { P } ( x _ { i } ) = \int e ^ { - ( x _ { i } - y ) ^ { 2 } } \mathrm { d } P ( y ) \leq p _ { i , r } + ( 1 - p _ { i , r } ) e ^ { - r ^ { 2 } } .
$$

Thus, in particular,

$$
p _ { i , r } \geq { \frac { c - e ^ { - r ^ { 2 } } } { 1 - e ^ { - r ^ { 2 } } } } = : p _ { r } .
$$

Now place an interval of radius r around each particle location, and let $C _ { 1 } , \ldots , C _ { S }$ denote the connected components of their union. Within each component, construct a maximal 2r-separated packing by selecting particle locations from left to right, choosing each successive location as soon as it lies more than $2 r$ from the preceding one. Let $B _ { s }$ denote the number of selected locations in component $C _ { s } .$ . Since the corresponding intervals are disjoint and each has probability mass at least $p _ { r }$ , we have

$$
1 \leq S \leq \sum _ { s = 1 } ^ { S } B _ { s } \leq \frac { 1 } { p _ { r } } .
$$

Moreover, by maximality of the packing, each component $C _ { s }$ has diameter at most $4 B _ { s } r$ . Let $d \leq r$ satisfy $c > e ^ { - d ^ { 2 } }$ . For each component $C _ { s } ,$ choose $z _ { 0 }$ at distance r to the left of its leftmost particle location. By construction, $z _ { 0 }$ lies at distance at least d from every particle location. Hence $\mu _ { P } ( z _ { 0 } ) \stackrel { \bar { < } } { { } \leq } e ^ { - d ^ { 2 } }$ , and therefore, recalling that $h = \Delta - c .$

$$
| h ( z _ { 0 } ) | = | c - \Delta ( z _ { 0 } ) | \geq c - e ^ { - d ^ { 2 } } .
$$

Let $\bar { h } _ { z _ { 0 } }$ denote the analytic continuation of h to the complex plane, recentred at $z _ { 0 }$ . As in the proof of Theorem $5 ,$ following the approach of Gr¨ochenig et al. (2020), we have

$$
| \bar { h } _ { z _ { 0 } } ( x + i y ) | \leq c + 2 e ^ { y ^ { 2 } } \leq ( c + 2 ) e ^ { y ^ { 2 } } .
$$

Now let $\delta > 1$ and $R _ { s } \geq ( 4 B _ { s } - 1 ) r$ , so that the disc of radius $R _ { s }$ centred at $z _ { \mathrm { 0 } }$ contains all particle locations in $C _ { s } .$ . Applying Prop. 30 with $A _ { f } = c + 2$ and $B _ { f } = 1$ gives

$$
\mathbf { n } _ { \bar { h } _ { z _ { 0 } } } ( R _ { s } ) \leq \frac { \log \left( \frac { c + 2 } { c - e ^ { - d ^ { 2 } } } \right) + \frac { 1 } { 2 } \delta ^ { 2 } R _ { s } ^ { 2 } } { \log \delta } .
$$

Since every distinct particle location contributes a zero of multiplicity two, summing over the components gives at least 2A zeros, counted with multiplicity. Consequently, using the same δ for every component, we obtain

$$
A \leq \frac { S \log \left( \frac { c + 2 } { c - e ^ { - d ^ { 2 } } } \right) + \frac { 1 } { 2 } \delta ^ { 2 } \sum _ { s = 1 } ^ { S } R _ { s } ^ { 2 } } { 2 \log \delta } .
$$

We now choose $\delta = { \sqrt { e } } , d = r , r = { \sqrt { \log ( 2 ) - \log c } }$ , and $R _ { s } = ( 4 B _ { s } - 1 ) r$ . We then have that $\textstyle \sum _ { s = 1 } ^ { S } R _ { s } ^ { 2 } =$ $\begin{array} { r } { r ^ { 2 } \sum _ { s = 1 } ^ { S } ( 4 B _ { s } - 1 ) ^ { 2 } \leq r ^ { 2 } ( 4 \sum _ { s = 1 } ^ { S } B _ { s } { - } S ) ^ { 2 } \leq r ^ { 2 } ( \frac { 4 } { n _ { * } } - 1 ) ^ { 2 } } \end{array}$ . With these choices, $p _ { r } = c / ( 2 - c ) , \mathrm { s o } S \leq 1 / p _ { r } = ( 2 - c ) / c .$ Substituting these bounds into the preceding estimate gives

$$
A \leq \frac { 2 - c } { c } \log \left( \frac { 2 ( c + 2 ) } { c } \right) + \frac { e } { 2 } \left( \frac { 8 - 5 c } { c } \right) ^ { 2 } \log \left( \frac { 2 } { c } \right) .
$$

Theorem 7. Suppose the problem is well specified. There exists $a \ f i n i t e$ threshold $N ( P ^ { * } )$ such that, whenever $N > N ( P ^ { * } )$ , F has no spurious local minima over w $\in \Delta ^ { N - 1 }$ and $\pmb { x } \in \mathbb { R } ^ { N }$ . Moreover, every spurious stationary point is unstable under the unweighted particle gradient flow.

Proof. By Props. 16 and $^ { 1 7 , }$ there exists a constant $L > 0 ;$ , depending only on $P ^ { * }$ , such that every spurious stationary point satisfies MMD $) ( P , P ^ { * } ) \geq L$ . In Prop. 13, choose $\nu > 0$ such that 4 erf $\begin{array} { r } { \mathrm { { \dot { c } } } ( \frac { \nu } { 2 } ) < \frac { L ^ { 2 } } { 2 } } \end{array}$ . Since the bound in Prop. 14 tends to zero as $c \to 0$ , we can choose $\underline { { c } } > 0$ such that, whenever $c \leq { \underline { { c } } } ,$ it holds that

$$
\| \Delta \| _ { 1 } \le L \sqrt { \frac { \pi } { 2 \operatorname { e r f i } ( \frac { \nu } { 2 } ) } } .
$$

By Prop. 13, this would imply $\mathrm { M M D } ( P , P ^ { * } ) < L$ , contradicting the lower bound. Thus, every spurious stationary point must satisfy $c > { \underline { { c } } } .$ . Applying Prop. 15 then gives a finite upper bound on the number of distinct active particle locations, depending only on $P ^ { * }$ . If N exceeds this bound, at least two particles must share a location. By Props. 11 and 12, such configurations cannot be local minima and are unstable under the unweighted particle gradient flow. □

Theorem 6. Let $D > 0$ and $\varepsilon > 0$ . There exists a finite threshold $N _ { \varepsilon } ( D )$ such that, uniformly over all target distributions with compact support of diameter at most D, whenever $N > N _ { \varepsilon } ( D )$ , every local minimum of $F$ over $\mathbf { w } \in \Delta ^ { N - 1 }$ and $\pmb { x } \in \mathbb { R } ^ { \tilde { N } }$ satisfies

$$
\mathrm { M M D } ( P , P ^ { * } ) < \varepsilon .
$$

Moreover, every stationary point with $\mathrm { M M D } ( P , P ^ { * } ) \ge \varepsilon$ is unstable under the unweighted particle gradient flow.

Proof. Fix $D > 0$ and $\varepsilon > 0 .$ , and consider any stationary point with $\mathrm { M M D } ( P , P ^ { * } ) \ge \varepsilon$ , where $P ^ { * }$ has compact support of diameter at most D. In Prop. 13, choose $\nu > 0$ such that 4 erfc $\textstyle \left( { \frac { \nu } { 2 } } \right) < { \frac { \varepsilon ^ { 2 } } { 2 } }$ . Since the bound in Prop. 14 tends to zero as $c \to 0$ , uniformly over such targets, we can choose $\underline { { c } } > 0$ , depending only on D and ε, such that, whenever $c \leq { \underline { { c } } } ,$ it holds that

$$
\| \Delta \| _ { 1 } \le \varepsilon \sqrt { \frac { \pi } { 2 \operatorname { e r f i } ( \frac { \nu } { 2 } ) } } .
$$

By Prop. 13, this would imply $\mathrm { M M D } ( P , P ^ { * } ) < \varepsilon ,$ contradicting our assumption. Thus every stationary point with $\mathrm { M M D } ( P , P ^ { * } ) \ge \varepsilon$ must satisfy $c > { \underline { { c } } } .$ . Applying Prop. 15 then gives a finite upper bound on the number of distinct active particle locations, depending only on $D$ and ε. If N exceeds this bound, at least two particles must share a location. By Props. 11 and 12, such configurations cannot be local minima and are unstable under the unweighted particle gradient flow. □

Optimising the bounds yields an asymptotic particle threshold of $N _ { \varepsilon } ( D ) = O ( \varepsilon ^ { - 4 } \log ( 1 / \varepsilon ) )$ as $\varepsilon  0$ , for fixed D. This is substantially worse than the relaxed-case rate, reflecting the larger number of particles required by our current simplex guarantees. We briefly justify this rate. As $x \to \infty$ , standard asymptotic estimates (DLMF, Eq. 7.12.1) give

$$
\operatorname { e r f c } ( x ) = O { \Big ( } { \frac { e ^ { - x ^ { 2 } } } { x } } { \Big ) } , \qquad \operatorname { e r f i } ( x ) = O { \Big ( } { \frac { e ^ { x ^ { 2 } } } { x } } { \Big ) } .
$$

By Prop. 14, we have $\| \Delta \| _ { 1 } \leq L _ { 1 } ( c )$ , where $L _ { 1 } ( c ) = O ( c { \sqrt { - \log c } } )$ as $c  0$ , for fixed D. Choosing $\nu =$ $2 \sqrt { - \log L _ { 1 } ( c ) }$ for suficiently small c in Prop. 13 gives $\mathrm { M M D ^ { 2 } } ( P , P ^ { * } ) = O ( c )$ . Thus, whenever $\mathrm { M M D } ( P , P ^ { * } ) \geq$ ε, we must have $c = \Omega ( \varepsilon ^ { 2 } )$ Finally, Prop. 15 bounds the number of distinct active particle locations by $O ( c ^ { - 2 } \log ( 1 / c ) )$ , yielding the stated threshold $N _ { \varepsilon } ( D ) = O ( \varepsilon ^ { - 4 } \log ( 1 / \varepsilon ) )$ ).

## C PRELIMINARY ANALYSIS OF ASYMPTOTIC STATIONARITY

Our main results concern stationary configurations with finite particle locations. A natural question is whether they extend to asymptotically stationary sequences, in which particle locations may escape to $\pm \infty$ . Such an extension would remove the need for compactness assumptions on the particle locations. Many of our arguments carry over essentially unchanged, but some require modification. Below, we outline these extensions and identify an obstruction to extending the finite-position analysis without studying the underlying particle dynamics. We focus on the simplex-constrained setting.

Throughout, consider a sequence $\{ ( \mathbf { w } ^ { n } , \pmb { x } ^ { n } ) \} _ { n = 1 } ^ { \infty }$ and constants $c _ { n }$ satisfying, for $j = 1 , \ldots , N$

$$
[ c _ { n } - \Delta _ { n } ( x _ { j } ^ { n } ) ] _ { + } \to 0 , \qquad w _ { j } ^ { n } ( \Delta _ { n } ( x _ { j } ^ { n } ) - c _ { n } ) \to 0 , \qquad w _ { j } ^ { n } \Delta _ { n } ^ { \prime } ( x _ { j } ^ { n } ) \to 0 ,
$$

where $\Delta _ { n }$ denotes the witness function associated with the nth configuration. This is a strong asymptotic KKT $\left( \mathrm { s } { - } \mathrm { A K K T } \right)$ sequence in the sense of B¨orgens et al. (2020); note that the distinction between strong and weak conditions is immaterial in our finite-dimensional setting. Importantly, these conditions describe asymptotic stationarity of the joint weight and location optimisation problem; the weights need not be fully optimised at each step. We may assume that at least one particle location escapes to ±∞ with nonzero limiting weight, since otherwise the finite-position analysis applies.

To study such sequences, we must track not only the particle locations, but also their pairwise distances. Let $\bar { \mathbb { R } } = \mathbb { R } \cup \{ - \infty , + \infty \}$ denote the two-point compactification of the real line. Embed the sequence $\pmb { x } ^ { n } \in \mathbb { R } ^ { N }$ into $\bar { \mathbb { R } } ^ { N } \times \bar { \mathbb { R } } ^ { N ( N - 1 ) / 2 }$ via

$$
\begin{array} { r } { { \pmb x } ^ { n } \longmapsto ( { \pmb x } ^ { n } , { \bf d } ^ { n } ) , \qquad d _ { i j } ^ { n } = | x _ { i } ^ { n } - x _ { j } ^ { n } | . } \end{array}
$$

This extends the usual compactification by retaining information about the distances between particles escaping to infinity. By Tychonof’s theorem, the resulting space is compact, so we may pass to a convergent subsequence. Since the simplex is also compact, we may assume that $( \mathbf { w } ^ { n } , \pmb { x } ^ { n } , \mathbf { d } ^ { n } )  ( \mathbf { w } ^ { \infty } , \pmb { x } ^ { \infty } , \mathbf { d } ^ { \infty } )$ . Moreover, $c _ { n }$ is bounded by the simplex constraints and the asymptotic stationarity conditions, so, passing to a further subsequence if necessary, we may assume $c _ { n } \to c .$ . The limiting stationarity conditions are then given by

$$
w _ { j } ^ { \infty } ( \Delta _ { \infty } ( x _ { j } ^ { \infty } ) - c ) = 0 , \qquad \Delta _ { \infty } ( x _ { j } ^ { \infty } ) \geq c , \qquad w _ { j } ^ { \infty } \Delta _ { \infty } ^ { \prime } ( x _ { j } ^ { \infty } ) = 0 , \qquad j = 1 , \dots , N ,\tag{12}
$$

where evaluations at infinite locations are understood as limits along the original sequence. We first show that particles with nonzero limiting weights cannot remain at a finite, positive distance from one another if at least one escapes to infinity.

Proposition 31. Suppose $w _ { i } ^ { \infty } , w _ { j } ^ { \infty }$ are nonzero. If $x _ { i } ^ { \infty } \in \{ \pm \infty \}$ , then $d _ { i j } ^ { \infty } \in \{ 0 , \infty \}$

Proof. The conclusion is immediate if $x _ { j } ^ { \infty }$ is finite or if the particles escape to opposite infinities. Otherwise, suppose that some particles escape to the same infinity while maintaining finite pairwise distances, with at least two distinct limiting locations. Consider an outermost particle with positive limiting weight within such a group. In the limit, interactions with particles outside the group vanish, while the particles at distinct locations within the group exert an outward force on the outermost particle. Since the weights are nonnegative, these contributions cannot cancel, and hence its limiting derivative cannot vanish. This contradicts the stationarity condition (12). □

We can therefore define a reduced asymptotic stationary configuration by removing zero-weight particles and merging particles for which $d _ { i j } ^ { \infty } = 0$ . Write $\vartheta _ { 1 } ^ { \infty } , \ldots , \vartheta _ { A } ^ { \infty }$ for its distinct locations and $\varsigma _ { 1 } ^ { \infty } , \ldots , \varsigma _ { A } ^ { \infty }$ for their positive total weights. By Prop. 31, distinct locations at infinity are infinitely separated, so the distance coordinates are no longer needed. For the reduced configuration, the stationarity conditions give

$$
\Delta _ { \infty } ( \vartheta _ { j } ^ { \infty } ) = c , \qquad \Delta _ { \infty } ^ { \prime } ( \vartheta _ { j } ^ { \infty } ) = 0 , \qquad j = 1 , \ldots , A .\tag{13}
$$

We first revisit Prop. 9. Its original interpolation argument does not extend directly to infinite locations, since kernel sections and their derivatives centred at escaping particles vanish at fixed finite points.

Proposition 32. Suppose we are at a spurious asymptotically stationary configuration of $F ( \mathbf { w } , \pmb { x } )$ . Then $c > 0$ Moreover, if $\vartheta _ { i } ^ { \infty } \in \{ \pm \infty \}$ , then $\varsigma _ { i } ^ { \infty } = c$

Proof. By assumption, at least one particle location escapes to infinity with positive limiting weight. Let $\vartheta _ { i } ^ { \infty } \in$ $\{ \pm \infty \}$ denote such a location. By Prop. 31, all other distinct locations with nonzero limiting weight are infinitely separated from it. Consequently, for the Gaussian kernel,

$$
\begin{array} { r } { \Delta _ { \infty } \big ( \vartheta _ { i } ^ { \infty } \big ) = \varsigma _ { i } ^ { \infty } k \big ( \vartheta _ { i } ^ { \infty } , \vartheta _ { i } ^ { \infty } \big ) = \varsigma _ { i } ^ { \infty } . } \end{array}
$$

The stationarity condition (13) therefore gives $\varsigma _ { i } ^ { \infty } = c .$ Since $\varsigma _ { i } ^ { \infty } > 0 ;$ , it follows that $c > 0 .$

Applying the argument of Prop. 10 to the finite-position particles recovers the corresponding maximum-principle conclusions. The collision result in Prop. 12 also extends essentially unchanged, while Prop. 16 holds in a stronger form.

We can also separate the contributions to the MMD from particles at finite and infinite locations. Suppose there are I escaping locations with nonzero total weight. By Props. 31 and 32, these locations are infinitely separated and each has weight c. Their contribution to the limiting squared MMD is therefore $I c ^ { 2 } \leq c .$ , since $I c \leq 1$ by the simplex constraint. Thus, these contributions are not themselves problematic.

The main obstruction arises when extending Prop. 14, whose proof relies on the weights summing to one. A corresponding argument in the asymptotic setting introduces an additional term Ic, which need not vanish as $c \to 0$ . For example, consider N particles with equal weights $1 / N$ and locations $x _ { i } ^ { n } = j n$ . As $n  \infty .$ , this sequence satisfies the asymptotic stationarity conditions with $c = 1 / N$ , while all $I = N$ particles escape, giving $I c = 1$ . Although such a configuration need not arise along the trajectories of a particular particle flow, it demonstrates that limiting stationarity alone is insuficient to control particle escape.

## D NUMERICAL ILLUSTRATION

In this appendix we provide a numerical illustration of the results established in the main text. We emphasise that these experiments are illustrative, since a finite numerical search ultimately cannot establish the absence of additional local minima. Nonetheless, the observed landscapes are qualitatively consistent with our theory. This is particularly true in the well-specified relaxed setting, where the disappearance of positive-error local minima once $N \geq K$ closely mirrors the sharp guarantee of Theorem 4. The experiments also illustrate the broader tendency for the local minima encountered numerically to improve as the number of particles increases.

Well-specified targets. For the well-specified setting, we sample 30 three-atom targets with locations $( 0 , U , 2 4 )$ where $U \sim \mathrm { U n i f } ( 0 , 2 4 )$ , and independent weights drawn from Dirichlet $( 1 , 1 , 1 )$ . We use this set of 30 throughout. This gives targets with varying atom separations and masses. For each target, we use $N = 1 , \ldots , 9$ particles in both formulations. Since these targets have $K = 3$ atoms, Theorem 4 guarantees that the relaxed problem has no spurious minima for $N \geq 3$

Compactly supported targets. For the general compactly supported setting, we additionally consider the continuous target $P ^ { * } = \mathrm { U n i f } [ 0 , 1 2 ]$ , with $N = 1 , \ldots , 1 6$ in both formulations. This provides a numerical illustration of the approximation regime of Theorems 5 and 6. All kernel integrals involving the uniform target are evaluated analytically, without discretising $P ^ { * }$

Initialisation. For each problem, we use 96 initialisations, with the same position initialisations used for the simplex and relaxed formulations. The weights are initialised from Dirichlet $( 3 , \ldots , 3 )$ to reduce the frequency of boundary starts. For the positions, let l, m, and r denote left, middle, and right reference locations. For the atomic targets we take $( l , m , r ) = ( 0 , U , 2 4 )$ , corresponding to the three target atoms, while for the uniform target we take $( l , m , r ) = ( 0 , 6 , 1 2 )$ , corresponding to the two support endpoints and their midpoint. We then use 16 starts from each of six position schemes: independent uniform draws on $[ l - 1 , r + 1 ] , [ l - 0 . 5 , m + 0 . 5 ]$ or $[ m - 0 . 5 , r + 0 . 5 ]$ ; equally spaced grids on $[ l , m ] { \mathrm { ~ o r ~ } } [ m , r ]$ with independent $\mathcal { N } ( 0 , 0 . 2 5 ^ { 2 } )$ perturbations; and independent uniform draws from $l , m , r$ with independent $\mathcal { N } ( 0 , 0 . 5 ^ { 2 } )$ perturbations. For the atomic targets, the two subinterval schemes are included specifically to search for local minima which fail to represent one of the target atoms, as in the geometric separation underlying Proposition 16.

Optimisation and certification. We optimise using Sequential Least Squares Programming (SLSQP) with analytic derivatives, tolerance $1 0 ^ { - 1 3 }$ , and at most 1400 iterations. For the atomic targets, the candidate solutions returned by SLSQP are subsequently refined using Newton’s method at 100 decimal digits. For the uniform target and each $N \geq 2 .$ , we first screen the SLSQP solutions using preliminary gradient and Hessian checks. We then refine the remaining candidate with the largest MMD. We retain positive-error points only when the gradient infinity norm is below $1 0 ^ { - 6 5 }$ , the relative Newton correction is below $1 0 ^ { - 4 5 }$ , and the joint Hessian is positive definite, restricted to mass-preserving variations in the simplex case. For the simplex problem, we additionally require all weights to be strictly positive. Every retained point is checked again at 130 decimal digits. All computations are carried out in Python 3.13.1 (Python Software Foundation, 2024), NumPy (Harris et al., 2020), SciPy (Virtanen et al., 2020), gmpy2 (Van Horsen, 2024) and mpmath (The mpmath development team, 2026) on an arm64 macOS CPU. Plots were produced with Matplotlib (Hunter, 2007). NumPy, SciPy and mpmath use BSD-3-Clause licences; gmpy2 uses an LGPL-3.0-or-later licence; Matplotlib uses its PSF-based licence. Python uses the Python Software Foundation License Version 2.

Results. Our numerical results are qualitatively consistent with our theory. This is particularly clear in the wellspecified relaxed problem, where positive-error local minima are found for $N < 3 .$ , whilst for every $N \geq K = 3$ the numerical search retains only exact, or numerically indistinguishable from exact, minima. This is consistent with Theorem 4, which gives the sharp threshold $N \geq K = 3$ in this setting. In the simplex case, positive-error minima persist for some targets throughout the range of particle numbers considered. However, as N increases, no positive-error minima are found for an increasing proportion of targets. This is consistent with Theorem $^ { 7 , }$ which gives a larger, target-dependent suficient threshold than in the relaxed case. For the continuous uniform target (Figure 1b), the MMD of the retained minima decreases steadily with N in both formulations. This is qualitatively consistent with Theorems 5 and 6, which show that suficiently many particles rule out local minima whose MMD remains bounded away from zero.

![](images/e9281541e89c4e7a05e9aedcc5a4af5ea3f5297c59617efe47c5ac1ae0ec5b79.jpg)

(b) Uniform target on [0, 12]  
![](images/68adbb06d9e4af4862b453c1ef3bd7a1408b76960480bb5acec08bfa0da5c919.jpg)  
Figure 1: A numerical illustration of the finite-particle landscapes. (a) Largest MMD among the retained loca minima, for each of the 30 well-specified three-atom targets. Dots correspond to individual targets, while boxes show medians and quartiles with 1.5-IQR whiskers. For $N \geq 3 .$ , the target can be represented exactly; hence a plotted value of zero means that the numerical search retained no positive-error local minimum. (b) MMD of the retained local minima for the continuous target $P ^ { * } = \mathrm { U n i f } [ 0 , 1 2 ]$ , shown on a logarithmic scale. The $N = 1$ solutions are obtained analytically.