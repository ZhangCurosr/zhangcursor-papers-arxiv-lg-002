# Differentiable Systematic Resampling for Variational Sequential Monte Carlo

Fredrik Cumlin   
Department of Information Science   
KTH Royal Institute of Technology Stockholm, Sweden fcumlin@kth.se Saikat Chatterjee   
Department of Information Science   
KTH Royal Institute of Technology Stockholm, Sweden sach@kth.se

## Abstract

Particle filters are a standard tool for nonlinear state estimation, but their resampling step is discrete, preventing gradient-based learning in variational sequential Monte Carlo. We introduce Differentiable Systematic Resampling (DSR), a temperaturecontrolled relaxation of systematic resampling, that preserves the CDF-ordered, banded structure of systematic resampling while enabling full gradient flow. DSR converges to exact systematic resampling as the temperature vanishes, and we prove a pointwise exponential convergence rate for the induced bias. Compared to optimal-transport-based differentiable resampling, DSR avoids iterative solvers and has substantially lower computational overhead. Experiments on stochastic dynamical systems and real-world handwriting data show that DSR achieves comparable or superior filtering and dynamics learning performance.

## 1 Introduction

Sequential Monte Carlo (SMC) methods, commonly known as particle filters (PFs) [14, 9], are standard tools for Bayesian state estimation in nonlinear and non-Gaussian dynamical systems, with applications in robotics, econometrics, computer vision, and epidemiology [30, 28, 4]. Particle filters approximate the filtering posterior by propagating and reweighting a set of particles using the system dynamics and incoming observations. In many applications, however, the state dynamics are unknown or only partially specified; that is, the transition model (system dynamics) is not available to the practitioner. One line of work addresses this by learning a state estimator directly from observations without an explicit particle representation [13, 12, 29, 7]; here we instead focus on variational sequential Monte Carlo (VSMC) [27, 24, 23], which jointly learns a parameterized transition model and proposal model by maximizing a variational lower bound on the marginal likelihood using gradient-based optimization. Importantly, VSMC requires the entire filtering recursion to be differentiable.

An essential component of SMC, and hence of VSMC, is resampling. During filtering, particle weights tend to degenerate, meaning that only one or a few particles carry most of the weight [10]. Resampling mitigates this by replacing low-weight particles with copies of high-weight ones [8, 20]. Among standard resampling schemes, systematic resampling is widely adopted due to its provably lower variance [8, 20, 10]. However, resampling relies on discrete ancestor assignments and is therefore inherently non-differentiable. While particle propagation and importance weighting can be differentiated via the reparameterization trick [19], the resampling step blocks gradient flow and produces biased or high-variance gradient estimators [27]. This makes differentiable resampling a central challenge in VSMC. A standard approach for discrete operations in gradient-based learning is to replace them with temperature-controlled continuous relaxations that recover the discrete operation in the zero-temperature limit [15, 25]. We take this approach for the resampling step of VSMC, but rather than relaxing the ancestor categorical directly, we relax the assignment of systematic resampling, based on the cumulative distribution function (CDF) of the weights, hence preserving its CDF-ordered, banded structure.

![](images/dea3daf5bb74ef57733d7cbe9041d6b709204e158a60f41f7fd12ddbcae8f4e3.jpg)  
Figure 1: Transportation maps for systematic resampling (left), DSR (middle), and EROT (right), for a sequential important resampling PF of the Lorenz-63 system. The transportation maps are calculated at step 150 using $N = 2 5$ particles, and DSR and EROT uses training relaxation parameters.

Several differentiable relaxations of resampling have been proposed. Soft multinomial resampling [17] keeps ancestor sampling discrete but draws ancestors from a smoothed distribution that mixes the importance weights with a uniform distribution, and propagates gradients through the corresponding importance-weight correction. A drawback is that the underlying scheme is multinomial resampling, which has provably higher variance than systematic resampling [8], so the method inherits an inferior starting point. Entropy-regularized optimal transport (EROT) [5] instead formulates resampling as an optimal transport problem. However, EROT relies on recursive Sinkhorn updates that scale as $\mathcal { O } ( N ^ { 2 } )$ per iteration, is sensitive to the choice of regularization parameter, and, at the regularization levels needed for stable training, produces diffuse transport plans that bear little resemblance to the sparse, CDF-ordered structure of systematic resampling (see Fig. 1). Crucially, neither approach preserves the structure of systematic resampling, which is the de facto standard at inference time due to its favorable variance properties [10].

In this paper, we introduce Differentiable Systematic Resampling (DSR), a temperature-controlled relaxation that directly softens the CDF-level ancestor assignment of systematic resampling using sigmoid functions. To further ensure that neighbours in the CDF correspond to spatially nearby particles, we equip DSR with Morton space-filling curve sorting, which reduces approximation error to systematic resampling. The relaxation introduces a bias that we prove vanishes exponentially as the temperature $\tau \to 0 ^ { + }$ . Our main contributions are as follows:

• We propose DSR, a differentiable relaxation that preserves the CDF-ordered, banded structure of systematic resampling while enabling full gradient flow.

• We establish theoretical guarantees showing that DSR converges to exact systematic resampling, with a pointwise exponential rate under a positive-margin condition, as $\tau \to 0 ^ { + }$

• We demonstrate that DSR achieves competitive or superior state estimation and dynamics learning in VSMC while training more than 2× faster than EROT.

## 2 Background

## 2.1 State space models and particle filters

We consider state space models $\mathbf x _ { t } \sim p ( \mathbf x _ { t } | \mathbf x _ { t - 1 } ) , \mathbf y _ { t } \sim p ( \mathbf y _ { t } | \mathbf x _ { t } )$ , where $\mathbf { x } _ { t } ~ \in ~ \mathbb { R } ^ { d _ { x } }$ is the latent state and $\mathbf { y } _ { t } ~ \in ~ \mathbb { R } ^ { d _ { y } }$ the observation. Particle filters approximate the filtering posterior $\begin{array} { r } { p ( \mathbf { x } _ { t } | \mathbf { y } _ { 1 : t } ) \approx \sum _ { i = 1 } ^ { N } w _ { t } ^ { ( i ) } \delta ( \mathbf { x } _ { t } - \mathbf { x } _ { t } ^ { ( i ) } ) } \end{array}$ using N weighted particles $\{ ( \mathbf { x } _ { t } ^ { ( i ) } , w _ { t } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ at time t. During inference, particles are propagated through the dynamics, reweighted by the observation likelihood, and resampled to avoid weight degeneracy. Resampling replaces the weighted particle set with an equal-weight set by sampling ancestor indices according to the weights.

Algorithm 1 Variational Sequential Monte Carlo (VSMC)   
Require: Initial particles $\{ \mathbf { x } _ { 0 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ , observations $\{ \mathbf { y } _ { t } \} _ { t = 1 } ^ { T }$   
1: for t = 1 to $\dot { \boldsymbol { { T } } }$ do   
2: Sample $\mathbf x _ { t } ^ { ( i ) } \sim q _ { \pmb \theta } ( \mathbf x _ { t } | \mathbf x _ { t - 1 } ^ { ( i ) } , \mathbf y _ { t } )$ for $i = 1 , \ldots , N$   
3: Compute weights $\begin{array} { r } { v _ { t } ^ { ( i ) } \propto \frac { p _ { \phi } ( \mathbf { x } _ { t } ^ { ( i ) } | \mathbf { x } _ { t - 1 } ^ { ( i ) } ) p ( \mathbf { y } _ { t } | \mathbf { x } _ { t } ^ { ( i ) } ) } { q _ { \theta } ( \mathbf { x } _ { t } ^ { ( i ) } | \mathbf { x } _ { t - 1 } ^ { ( i ) } , \mathbf { y } _ { t } ) } } \end{array}$ ; normalize $\begin{array} { r } { \sum _ { i } w _ { t } ^ { ( i ) } = 1 } \end{array}$   
4: $\{ \tilde { \mathbf { x } } _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N }  \mathtt { R e s a m p l e } \big ( \{ \mathbf { x } _ { t } ^ { ( i ) } , w _ { t } ^ { ( i ) } \} _ { i = 1 } ^ { N } \big )$ (discrete, non-differentiable)   
5: end for

## 2.2 Variational Sequential Monte Carlo

Variational Sequential Monte Carlo (VSMC) [27] provides a framework for jointly learning a transition model $p _ { \phi } ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } )$ and proposal distribution $q _ { \pmb { \theta } } ( \mathbf x _ { t } | \mathbf x _ { t - 1 } , \mathbf y _ { t } )$ by maximizing the so-called surrogate evidence lower-bound (ELBO):

$$
\mathcal { L } ( \pmb \theta , \pmb \phi ) = \mathbb { E } \left[ \log \hat { p } ( \mathbf { y } _ { 1 : T } ) \right] = \mathbb { E } \left[ \sum _ { t = 1 } ^ { T } \log \left( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { t } ^ { ( i ) } \right) \right] ,\tag{1}
$$

where the expectation is over all sampling and resampling operations in the particle filter. Algorithm 1 summarizes the VSMC recursion. The surrogate ELBO is a lower bound to the standard ELBO (as used for dynamical variational autoencoders, see e.g. [22]).

Optimizing L requires differentiating through the entire filtering recursion. As shown by [27], the gradient decomposes into two terms:

$$
\begin{array} { r } { \nabla \mathcal { L } = \underbrace { \mathbb { E } [ \nabla \log \hat { p } ( \mathbf { y } _ { 1 : T } ) ] } _ { g _ { \mathrm { r e p } } ( \mathrm { r e p a r a m e t e r i z a t i o n } ) } + \underbrace { \mathbb { E } \left[ \log \hat { p } ( \mathbf { y } _ { 1 : T } ) \nabla \log \tilde { \varphi } ( \mathbf { a } _ { 1 : T - 1 } ^ { 1 : N } | \varepsilon _ { 1 : T } ^ { 1 : N } ) \right] } _ { g _ { \mathrm { s c o r e } } ( \mathrm { s c o r e f u n c t i o n } ) } , } \end{array}\tag{2}
$$

where $\tilde { \varphi }$ is the marginal distribution over the discrete ancestor variables $\mathbf { a } _ { 1 : T - 1 } ^ { 1 : N }$ given the reparameterized noise $\pmb { \varepsilon } _ { 1 : T } ^ { 1 : N }$ . The reparameterization term $g _ { \mathrm { r e p } }$ differentiates through the continuous particle propagation and weighting steps via the reparameterization trick [19]. The score-function term $g _ { \mathrm { s c o r e } }$ accounts for the discrete ancestor assignments introduced by resampling, which are not reparameterizable and can have high variance [27, cf. Fig. 3].

In practice, $g _ { \mathrm { s c o r e } }$ is dropped, yielding the biased gradient $\begin{array} { r } { \nabla \mathcal { L } \approx g _ { \mathrm { r e p } } \left[ 2 7 \right] } \end{array}$ . This treats ancestor indices as fixed during backpropagation: gradients flow through the surviving particle values, but not through the ancestor selection itself, a limitation that becomes severe when weights are concentrated and resampling decisions dominate filtering performance (see Appendix F.3). Differentiable resampling methods address this by replacing the discrete ancestor selection with a continuous relaxation, so that $g _ { \mathrm { r e p } }$ flows through the full resampling mechanism. This yields a relaxed objective in place of the surrogate ELBO. At finite temperature the relaxation does not preserve expected replication counts, so the resulting likelihood estimator $\hat { p } ( \mathbf { y } _ { 1 : T } )$ is no longer unbiased and the relaxed objective is a differentiable surrogate rather than a lower bound on $\bar { \log { p ( \mathbf { y } _ { 1 : T } ) } }$ (see Section 3.1). The relaxed objectives have been shown to provide a richer learning signal for the transition model and proposal with improved state estimation and system identification across tasks [5, 32, 3, 21, 6].

## 3 Differentiable Systematic Resampling

Traditional resampling methods are discrete and non-differentiable, creating a bottleneck for gradientbased learning. We propose differentiable systematic resampling (DSR): a soft relaxation that converges to systematic resampling in the limit, while allowing gradients to flow through the resampling step. For ease of notation, we omit time subscripts in Sections 3 and $4 ;$ particle indices are instead denoted as subscripts.

## 3.1 DSR formulation

Recall that systematic resampling computes the CDF $\begin{array} { r } { F _ { i } = \sum _ { k = 1 } ^ { i } w _ { k } } \end{array}$ , draws a single offset $u _ { 0 } \sim$ $\mathcal { U } ( 0 , 1 / N )$ , and assigns $a _ { j } = \operatorname* { m i n } \{ i : { \bar { F } } _ { i } \geq u _ { j } \}$ where $u _ { j } = u _ { 0 } ^ {  } \overset {  } { + } ( j - 1 ) / N$ . Equivalently, $a _ { j }$ is the unique index i for which $F _ { i - 1 } < u _ { j } \leq F _ { i }$ , so the discrete assignment can be written as a matrix $A _ { i j } = \bar { \bf 1 } ( F _ { i - 1 } < u _ { j } \leq F _ { i } )$ . This indicator is non-differentiable with respect to the weights, blocking gradient flow.

Consider $N = 3$ particles with weights $\mathbf { w } = ( 0 . 1 , 0 . 7 , 0 . 2 )$ , giving the CDF $\mathbf { F } = ( 0 . 1 , 0 . 8 , 1 . 0 )$ and $F _ { 0 } = 0$ . Systematic resampling draws a single offset $\begin{array} { r } { \bar { u _ { 0 } } \sim \bar { \mathcal { U } } ( 0 , \frac { 1 } { 3 } ) } \end{array}$ and places the comb $u _ { j } = u _ { 0 } + ( j - 1 ) / 3$ . For $u _ { 0 } = 0 . 1$ the comb points are $u _ { 1 } = 0 . 1 0 , u _ { 2 } = 0 . 4 3 , u _ { 3 } = 0 . 7 7$ , and thresholding each against the CDF gives ancestors $a _ { 1 } = 1$ (since $F _ { 0 } < u _ { 1 } \leq F _ { 1 } ) , a _ { 2 } = 2$ (since $F _ { 1 } < u _ { 2 } \le F _ { 2 } )$ , and $a _ { 3 } = 2$ (since $F _ { 1 } < u _ { 3 } \leq F _ { 2 } ) \colon$ : the high-weight particle 2 is replicated, the low-weight particle 3 is dropped, and the assignment matrix is $A _ { i j } = \mathbf { \bar { 1 } } ( \mathbf { \dot { F } } _ { i - 1 } < u _ { j } \leq \dot { F _ { i } } )$ . The only randomness is the single draw $u _ { 0 } ;$ given $u _ { 0 } ,$ , the assignment is a deterministic thresholding of the comb against the CDF. The idea with DSR is to relax only this thresholding step and to leave the draw $u _ { 0 }$ untouched.

Our strategy is to replace the indicator with a smooth surrogate that recovers it in a limiting sense, following the temperature-relaxation approach common for discrete operations in gradient-based learning [15, 25]. Writing the indicator as a telescoping difference,

$$
\mathbb { 1 } ( F _ { i - 1 } < u \leq F _ { i } ) = \mathbb { 1 } ( u > F _ { i - 1 } ) - \mathbb { 1 } ( u > F _ { i } ) ,\tag{3}
$$

reduces the problem to relaxing each step indicator $\mathbb { 1 } ( u > F )$ separately. The sigmoid $\sigma ( x ) =$ $( 1 + e ^ { - x } ) ^ { - 1 ^ { \ast } }$ provides a natural smooth surrogate: $\sigma ( ( u - F ) / \tau ) \to \mathbb { 1 } ( u > F )$ pointwise as the temperature $\tau \to 0 ^ { + }$ . Substituting into the telescoping identity with boundary conventions $F _ { 0 } = 0$ $F _ { N + 1 } = 1$ gives the soft assignment:

$$
s _ { i } ( u ; \tau ) = \sigma \left( \frac { u - F _ { i - 1 } } { \tau } \right) - \sigma \left( \frac { u - F _ { i } } { \tau } \right) , \quad i = 1 , \dots , N\tag{4}
$$

Each $s _ { i } ( u ; \tau ) \in [ 0 , 1 ]$ is differentiable in $u , F _ { i - 1 }$ , and $F _ { i }$ , and $\begin{array} { r } { \sum _ { i } s _ { i } ( u ; \tau ) \to 1 } \end{array}$ as $\tau \to 0 ^ { + }$ . At finite $\tau ,$ the sum $\textstyle \sum _ { i } s _ { i } ( u ; \tau )$ is not necessarily one, so we normalize to obtain a proper convex combination. The transportation matrix and resampled particles are:

$$
T _ { i j } ^ { ( \tau ) } = \frac { s _ { i } ( u _ { j } ; \tau ) } { \sum _ { k = 1 } ^ { N } s _ { k } ( u _ { j } ; \tau ) } , \qquad \tilde { \mathbf { x } } _ { j } ^ { ( \tau ) } = \sum _ { i = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } \mathbf { x } _ { i } .\tag{5}
$$

Each column of $T ^ { ( \tau ) }$ sums to one, so $\tilde { \mathbf { x } } _ { j } ^ { ( \tau ) }$ is a convex combination of the ancestor particles. The row sums, however, satisfy $\begin{array} { r } { \sum _ { j = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } \ne \bar { N } w _ { i } } \end{array}$ in general, so DSR does not preserve expected replication counts at finite τ. The temperature τ controls the relaxation between gradient flow and approximation fidelity: small τ gives a sharp transport plan close to systematic resampling but produces large gradient magnitudes, while large τ gives a diffuse transport plan with smoother gradients. We show in Section 4 that the approximation error decays exponentially in $1 / \tau .$ , and demonstrate in Section 5 that $\tau = 0 . 1$ is a robust choice across all our experiments. Forming the full transport matrix $T ^ { ( \tau ) } \in \mathbb { R } ^ { N \times N }$ makes DSR $\mathcal { O } ( N ^ { 2 } )$ per resampling step, compared with $\mathcal O ( N )$ for hard systematic resampling; the resulting overhead is nonetheless small in practice due to batch processing on GPU (Table 5). An algorithmic overview of DSR is given in Appendix A.

## 3.2 Particle ordering

Since DSR operates on the empirical CDF, the particle ordering determines which particles interact through the soft assignments: particles adjacent in the CDF share mass in the rows of $T ^ { ( \tau ) }$ . It is therefore desirable to order particles so that CDF neighbours are also spatial neighbours, which keeps the soft mixture more local. Thus, we sort particles along a Morton space-filling curve [26] before constructing the CDF; an algorithmic overview of this sorting is provided in Appendix A. Locality-preserving orderings have been shown to improve convergence properties of systematic resampling [11]; in DSR, they additionally ensure the soft mixture blends spatially nearby ancestors rather than distant particles (for an analysis of this, see Appendix E). The sorting costs $\mathcal { \dot { O } } ( N _ { }$ log N) per resampling step, negligible compared to the overall resampling cost of $\mathcal { O } ( N ^ { 2 } )$ ).

Remark 3.1 (Reducing the $\mathcal { O } ( N ^ { 2 } )$ cost). The $\mathcal { O } ( N ^ { 2 } )$ cost of forming $T ^ { ( \tau ) }$ can in principle be reduced to $\mathcal { O } ( N K )$ by computing, for each column j, only the soft assignments in a band $i \in$ $[ \operatorname* { m a x } ( 1 , j - \dot { K } / 2 ) , \operatorname* { m i n } ( N , \dot { j } + \dot { K } / 2 ) ]$ and setting the remainder to zero, at the cost of an additional bias that decreases with K. This is well-motivated under Morton sorting, where the soft assignments decay to near zero away from the diagonal (Fig. 5). In our implementation, however, $T ^ { ( \tau ) }$ is formed with batched matrix operations, so the wall-clock benefit of such truncation on GPU is modest; we therefore retain the full form.

Morton sorting significantly improves the approximation to systematic resampling: at $N = 2 0 0$ , it reduces the L2 distance between DSR and hard systematic resampled particles by 35% compared to random ordering (Appendix E). For filtering and transition model quality, Morton sorting offers a modest advantage at low SNR (Appendix F.4).

## 4 Theoretical Analysis

We show that DSR is a consistent relaxation whose bias vanishes exponentially fast in $1 / \tau .$ . Let $\{ ( \mathbf { x } _ { i } , w _ { i } ) \} _ { i = 1 } ^ { N }$ be particles with normalized weights $\begin{array} { r } { w _ { i } > 0 , F _ { i } = \sum _ { k = 1 } ^ { i } w _ { k } } \end{array}$ the empirical CDF, and $T _ { i j } ^ { ( \tau ) } , \tilde { \mathbf { x } } _ { j } ^ { ( \tau ) }$ as defined in Eq. (5). The first result establishes that DSR recovers systematic resampling in the limit.

Proposition 4.1 (Asymptotic exactness). Fix $j \in [ N ]$ and let $a _ { j } ^ { * } = \operatorname* { m i n } \{ i \in [ N ] : F _ { i } \geq u _ { j } \}$ denote the hard systematic resampling index. Then lim ${ \bf \Phi } _ { \tau  0 ^ { + } } T _ { i j } ^ { ( \tau ) } = \mathbb { 1 } ( i = a _ { j } ^ { * } )$ and lim $\mathbf { \sigma } _ { \tau  0 ^ { + } } \tilde { \mathbf { x } } _ { j } ^ { ( \tau ) } = \mathbf { x } _ { a _ { j } ^ { * } }$ almost surely.

The second result quantifies the approximation error for finite τ .

Proposition 4.2 (Exponential convergence rate). Let $D = \operatorname* { s u p } _ { i , j } \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 }$ denote the particle diameter and

$$
\Delta _ { j } : = \operatorname * { m i n } \{ u _ { j } - F _ { a _ { j } ^ { * } - 1 } , \ F _ { a _ { j } ^ { * } } - u _ { j } \} > 0 \ a . s .\tag{6}
$$

the margin between u<sub>j</sub> and the nearest CDF boundary. Then there exists $\tau _ { 0 } > 0$ such thatfor all $\tau \leq \tau _ { 0 } ,$

$$
\| \widetilde { \mathbf { x } } _ { j } ^ { ( \tau ) } - \mathbf { x } _ { a _ { j } ^ { * } } \| _ { 2 } \leq 2 D N \exp \left( - \frac { \Delta _ { j } } { \tau } \right) .\tag{7}
$$

The proofs are in Appendix B.

Proposition 4.2 is primarily a qualitative guarantee: the bound depends on the particle diameter D, the count N, and the margin $\Delta _ { j }$ between $u _ { j }$ and the nearest CDF boundary. The dominant term is $\exp ( - \Delta _ { j } / \tau )$ , so concentrated weights (large $\Delta _ { j } )$ yield fast convergence while near-uniform weights, where $\Delta _ { j }$ scales as $\mathcal { O } ( 1 / N )$ , bound the worst-case rate. The bound holds pointwise in $j$ under the positive-margin event $\dot { \Delta _ { j } } > 0$ (which holds almost surely); it is not uniform over offsets, since $\Delta _ { j }$ can be made arbitrarily small by the weight and offset configuration, so no uniform exponential rate holds. Empirically, the per-particle L2 distance between DSR and hard systematic resampling decreases monotonically and accelerates as $\tau \to 0 ^ { + }$ (Appendix E), consistent with the exponential rate.

## 5 Experiments

We evaluate DSR on synthetic and real-world state estimation and dynamics learning tasks. The experiments assess four aspects: (1) training-time filtering posterior quality $\hat { p } ( \mathbf x _ { t } | \mathbf y _ { 1 : t } ) ; ( 2 )$ state estimation accuracy at inference time in standard SMC; (3) transition model quality $\hat { p } ( \mathbf x _ { t } | \mathbf x _ { t - 1 } )$ ; and (4) computational efficiency.

## 5.1 Experimental Setup

We consider nonlinear state-space models where the transition and proposal models are parameterized by fully connected neural networks and learned jointly under the VSMC objective. All methods use ${ \dot { N } } \ = \ 2 5$ particles unless stated otherwise (Appendix F.3 reports a sweep over $N \in \{ 2 5 , 5 0 , 1 0 0 , 2 0 0 \} )$ , the same Adam [18] configuration, identical network architectures, and gradient clipping at norm 5.0. We clip the estimated log-likelihood below to avoid extreme instability in training, see Appendix C. Results are averaged over 10 independent runs; we report mean ± standard deviation. Training configuration can be found in the Appendix C.

We evaluate on three settings: (1) a 4-dimensional Linear Gaussian SSM, where the Kalman filter gives the optimal posterior; (2) Lorenz-63, a 3D chaotic attractor with a nonlinear camera observation map; and (3) CharacterTrajectories, real-world handwriting dynamics with 3D position tracking. We use different signal-to-measurement noise (SMNR) ratios to experiment with different noise levels (definition of SMNR can be found in Appendix D.1).

We report NMSE (dB) for state estimation, one-step MSE and five-step MSE for transition model quality, log-likelihood (LL) and Kullback–Leibler divergence (KLD) for distribution matching, and wall-clock training time. Full definitions are in Appendix D.

We compare DSR to three resampling baselines: Hard, standard systematic resampling [16], which corresponds to estimating the ELBO gradient via $g _ { \mathrm { r e p } }$ in Eq. (2); Soft, relaxation of multinomial resampling with $\alpha ~ = ~ 0 . 5 ~ [ 1 7 ] ;$ and EROT, entropy-regularized optimal-transport resampling with $\varepsilon ~ = ~ 0 . 5$ and 100 Sinkhorn iterations [5]. All methods produce biased estimates of the surrogate ELBO gradient. Our code is available at https: $: / / \mathsf { g } \dot { \mathsf { 1 } }$ thub.com/fcumlin/ differentiable-systematic-resampling.git.

## 5.2 Linear Gaussian State Space Model

To evaluate DSR in a controlled setting where the ground truth is analytically available, we consider a 4-dimensional linear Gaussian state space model

$$
\mathbf { x } _ { t } = \mathbf { A } \mathbf { x } _ { t - 1 } + \mathbf { e } _ { t } , \quad \mathbf { y } _ { t } = \mathbf { x } _ { t } + \mathbf { w } _ { t } ,\tag{8}
$$

with $\mathbf { e } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { e } ^ { 2 } \mathbf { I } _ { 4 } ) , \mathbf { w } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { w } ^ { 2 } \mathbf { I } _ { 4 } )$ , process noise at −30 dB and observation noise set by the SMNR. The true transition matrix is a damped block-diagonal rotation $\textbf { A } = \ 0 . 9$ blockdiag $( { \bf R } ( \theta _ { 1 } ) , { \bf R } ( \theta _ { 2 } ) )$ with $\theta _ { 1 } = 0 . 3 , \theta _ { 2 } = 0 . 5$ and $\mathbf { R } ( \theta )$ the $2 \times 2$ rotation matrix. Since the model is linear Gaussian, the Kalman filter gives the optimal posterior at each t, which lets us compare the training-time particle distribution against a Bayesian oracle via KLD. We parameterize the transition model as $\hat { \textbf { A } } \in \mathbb { R } ^ { 4 \times 4 }$ with a learnable diagonal covariance, isolating the effect of resampling on parameter learning. We train with $N = 2 5$ particles on 1000 trajectories of length $T = 1 0 0 \ { \mathrm { a t } } \ S { \mathrm { M N R } } = 1 0 \ { \mathrm { d B } }$

Table 1: Performance on the linear Gaussian SSM $( d _ { x } = 4 , \mathrm { S M N R } = 1 0 \mathrm { d B } )$ . KLD KF is the training-time divergence to the Kalman posterior; NMSE is evaluated with $N = 1 0 0 0$ particles and hard systematic resampling; $\| A - { \hat { A } } \| _ { F }$ measures recovery of the true transition matrix. Results show mean ± std over 3 runs.
<table><tr><td>Method</td><td>KLD KF</td><td>NMSE (dB)</td><td> $\| A - { \hat { A } } \| _ { F }$ </td></tr><tr><td>Hard</td><td> $\mathbf { 1 7 . 4 9 { \scriptstyle \pm 0 . 0 7 } }$ </td><td> $- 1 1 . 3 3 { \pm } 0 . 0 6$ </td><td> $0 . 0 3 5 { \scriptstyle \pm 0 . 0 0 1 }$ </td></tr><tr><td>Soft</td><td> $1 7 . 5 2 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $- 1 0 . 3 3 { \pm } 0 . 0 7$ </td><td> $0 . 0 2 5 { \scriptstyle \pm 0 . 0 0 2 }$ </td></tr><tr><td>EROT</td><td> $2 2 . 0 9 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $- 1 1 . 3 1 { \pm } 0 . 0 2$ </td><td> $\underline { { 0 . 0 2 2 } } \pm 0 . 0 0 4$ </td></tr><tr><td>DSR</td><td> $2 1 . 7 7 { \scriptstyle \pm 0 . 0 6 }$ </td><td> $- \mathbf { 1 1 . 3 4 } \pm \mathbf { 0 . 0 1 }$ </td><td> $\mathbf { 0 . 0 1 9 { \scriptstyle \pm 0 . 0 0 1 } }$ </td></tr></table>

This experiment surfaces a distinction that is difficult to see in the chaotic or real-world settings: Hard achieves the best training-time posterior (KLD 17.49 vs. 21.77 for DSR), yet DSR recovers the true transition matrix most accurately $( \| A - { \hat { A } } \| _ { F }$ of $0 . 0 1 9 \mathrm { v s . 0 . 0 3 5 ) }$ . A better training-time posterior therefore does not imply a better gradient signal for parameter learning, which is consistent with the gradient-norm analysis on Lorenz-63 (Section 5.3.1), where DSR’s smoother gradients lead to better downstream performance despite comparable point-wise posterior quality. Appendix H reports two further linear Gaussian studies: an evaluation-time decomposition of the marginal likelihood, and a comparison of the DSR gradient against an unbiased VSMC gradient.

## 5.3 Lorenz-63

We evaluate all methods on the Lorenz-63 attractor with a nonlinear camera measurement map, following [32]. The Lorenz-63 system is a 3-dimensional chaotic attractor given by the discretized dynamics

$$
\begin{array} { r } { \mathbf { x } _ { t + 1 } = F ( \mathbf { x } _ { t } ) \mathbf { x } _ { t } + \mathbf { e } _ { t } , \quad \mathbf { e } _ { t } \sim \mathcal { N } ( 0 , \sigma _ { e } ^ { 2 } I ) , } \end{array}\tag{9}
$$

![](images/e11393abc3fb653edf6146d91fc1c64ab135b387120d456c05b477cb967b971a.jpg)  
Figure 2: Lorenz-63 system with nonlinear camera measurement model. (a) Sample state trajectory of Lorenz-63. Colored points indicate three observation times that are shown in the rightmost figure. (b) Camera projection geometry: the 3D state ${ \bf x } _ { t } = ( x _ { 1 } , x _ { 2 } , x _ { 3 } )$ projects onto an 8×8 pixel grid via a depth-dependent Gaussian point spread function (Eq. 10). The first two coordinates determine the 2D position, while the third coordinate (depth) controls the spread of the Gaussian. (c) Three sample camera observations at $\mathrm { S M N R } = 0 ~ \mathrm { d B }$

Table 2: Performance comparison on the Lorenz-63 system at SMNR = 10 and 0 dB. Results show mean ± std over 10 runs. Best in bold, second-best underlined.
<table><tr><td rowspan="2">Method</td><td rowspan="2">State Estimation NMSE (dB)</td><td colspan="4">Transition Model Quality</td></tr><tr><td>MSE</td><td> $\mathbf { M S E - S - s t e p }$ </td><td>LL</td><td>KLD</td></tr><tr><td></td><td colspan="5">SMNR = 10 dB</td></tr><tr><td>Hard</td><td> $- 2 8 . 8 6 { \pm } 0 . 4 1$ </td><td> $\underline { { 0 . 5 5 } } \pm \mathbf { 0 . 1 9 }$ </td><td> $5 . 5 0 { \scriptstyle \pm 3 . 6 8 }$ </td><td> $- 2 . 2 6 { \pm } 0 . 2 5$ </td><td> $5 . 4 1 { \scriptstyle \pm 1 . 4 7 }$ </td></tr><tr><td>Soft</td><td> $\overline { { - 2 7 . 8 2 } } \pm 0 . 4 5$ </td><td> $0 . 8 6 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $6 . 4 5 { \scriptstyle \pm 3 . 9 2 }$ </td><td> $- 2 . 5 5 { \pm } 0 . 4 8$ </td><td> $6 . 9 0 { \scriptstyle \pm 2 . 4 3 }$ </td></tr><tr><td>EROT</td><td> $- 2 8 . 7 6 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $3 . 3 9 { \scriptstyle \pm 2 . 5 9 }$ </td><td> $- 2 . 3 2 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $5 . 7 4 { \scriptstyle \pm 1 . 0 3 }$ </td></tr><tr><td>DSR</td><td> $- 2 9 . 9 4 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $\mathbf { 0 . 4 7 { \scriptstyle \pm 0 . 2 4 } }$ </td><td> ${ \bf 3 . 3 8 } { \scriptstyle \pm 2 . 1 1 }$ </td><td> $\mathbf { - 1 . 9 8 { \scriptstyle \pm 0 . 2 7 } }$ </td><td> $\mathbf { 3 . 6 0 } \pm \mathrm { 1 . 0 4 }$ </td></tr><tr><td></td><td colspan="5"> $\mathbf { S M N R } = 0 ~ \mathrm { d B }$ </td></tr><tr><td>Hard</td><td> $- 1 1 . 7 3 { \scriptstyle \pm 1 . 7 5 }$ </td><td> $0 . 9 6 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $9 . 6 4 \pm 1 3 . 7 1$ </td><td> $\mathbf { - 3 . 0 4 } 2 0 . 4 0$ </td><td> $\mathbf { 1 2 . 4 0 { \scriptstyle \pm 3 . 9 5 } }$ </td></tr><tr><td>Soft</td><td> $- 1 1 . 6 9 { \pm } 1 . 6 5$ </td><td> $\underline { { 0 . 9 2 } } \underline { { \pm } } \underline { { 0 . 0 8 } }$ </td><td> $\underline { { 8 . 1 9 } } \pm 6 . 4 7$ </td><td> $- 3 . 1 7 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 3 . 8 7 { \scriptstyle \pm 2 . 3 2 }$ </td></tr><tr><td>EROT</td><td> $1 2 . 5 7 { \scriptstyle \pm 1 . 6 4 }$ </td><td> $4 . 7 9 2 5 . 6 5$ </td><td> $4 8 . 4 1 \pm 6 0 . 7 2$ </td><td> $- 4 . 2 8 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $3 3 . 8 2 { \scriptstyle \pm 2 9 . 7 9 }$ </td></tr><tr><td>DSR</td><td> $\overline { { - 1 8 . 2 8 } } \pm 0 . 4 6$ </td><td> $\mathbf { 0 . 5 4 \pm 0 . 3 3 }$ </td><td> $\mathbf { 3 . 7 0 { \scriptstyle \pm 2 . 8 7 } }$ </td><td> $- 3 . 0 6 { \pm } 0 . 2 4$ </td><td> $\underline { { 1 3 . 4 1 } } \pm 1 . 9 4$ </td></tr></table>

where $F ( \mathbf { x } _ { t } ) = \exp \left( \left[ \begin{array} { c c c } { - 1 0 } & { 1 0 } & { 0 } \\ { 2 8 } & { - 1 } & { - x _ { t , 1 } } \\ { 0 } & { x _ { t , 1 } } & { - 8 / 3 } \end{array} \right] \Delta \tau \right)$ , with $\Delta \tau = 0 . 0 2$ and $\sigma _ { e } ^ { 2 } = 0 . 1$ corresponding to −10 dB process noise. The values follow [13, 12]. We use a 2D camera model with Gaussian point spread functions over an $8 \times 8$ pixel grid, following [2, 32]. The grid points $\{ g _ { i } \} _ { i = \cdot } ^ { 6 4 } .$ <sub>1</sub> are uniformly distributed in the rectangle $[ - \bar { 3 } 0 , 3 0 \bar { ] } \times [ - 4 0 , 4 0 ]$ . The state is projected into the camera plane according to its first two coordinates, with the third coordinate representing depth. The measurement function h $: \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 6 4 }$ is given by

$$
\mathbf { h } ( \mathbf { x } _ { t } ) _ { i } = 1 0 \exp \left( - \frac { 0 . 5 } { 7 + x _ { t , 3 } } \left\| \boldsymbol { g } _ { i } - \left[ \boldsymbol { x } _ { t , 1 } \right] \right\| _ { 2 } ^ { 2 } \right) ,\tag{10}
$$

where $\mathbf { h } ( \mathbf { x } _ { t } ) _ { i }$ represents the intensity at pixel i. The observations are given by $\mathbf y _ { t } = \mathbf h ( \mathbf x _ { t } ) + \mathbf w _ { t }$ with $\mathbf { w } _ { t } \sim \dot { \mathcal { N } } ( \mathbf { 0 } , \sigma _ { w } ^ { 2 } \mathbf { I } )$ , where $\sigma _ { w } ^ { 2 }$ is determined by the signal-to-measurement noise ratio (SMNR). For training, we use 1000 trajectories of observations $\mathbf { y } _ { 1 : T }$ with $T = 1 0 0 ;$ we experiment with two SMNR levels: 10 dB and 0 dB. A visualization of the Lorenz-63 system under the camera mapping is shown in Fig. 2.

We evaluate both state estimation and transition model quality (Table 2). State estimation uses 10 held-out trajectories with $T = 1 0 0 0$ and the systematic resampling at inference; the transition model $f _ { \theta }$ is evaluated directly on held-out trajectories.

![](images/aebd3fdca00f099b07b51d1d96704cabfa956e0e0b16602843416264ae3b5fa6.jpg)  
Figure 3: Gradient norm during training on Lorenz-63 (SMNR = 10 dB) for a representative run. The solid line shows the mean gradient norm per epoch across batches; the shaded region indicates the within-epoch standard deviation.

At 10 dB SMNR, DSR achieves the best NMSE (−29.94 dB) with low run-to-run variance, and the best MSE, LL, and KLD on the transition model, matching EROT on multi-step prediction. Hard and Soft struggle more on longer horizons.

The gap widens at 0 dB SMNR, where DSR improves NMSE by 5.7 dB over the second-best method (EROT at −12.57 dB). At low SNR, particle weights become highly concentrated and amplify noisy gradients; DSR’s smoother optimization landscape (Table 3) yields more stable parameter learning. The resulting transition model is not best on every one-step metric (LL, KLD), but produces particles that filter more effectively at inference time, since one-step metrics from true states do not capture error compounding during sequential filtering.

The large MSE-5-step variance of EROT at 0 dB (57.60) raises the question of whether this reflects the method or the choice of ε. We therefore evaluate EROT at ε = 1 as well (Appendix F.5). Increasing ε stabilizes training (MSE-5-step std drops to 7.19) and improves NMSE to −13.47 dB, but DSR still outperforms EROT on every metric.

## 5.3.1 Gradient norm

To assess training stability, we study the gradient provided by each resampling method on Lorenz-63 at $\mathrm { \Delta S M N R } = \mathrm { 1 0 ~ d B }$ . Figure 3 shows the mean gradient norm per epoch. DSR maintains gradient norms around $4 \times 1 0 ^ { 3 }$ throughout training, whereas Hard and EROT exhibit norms in the $1 0 ^ { 4 } – 1 \mathrm { \overline { { 0 } } ^ { 5 } }$ range with substantially larger batch-to-batch fluctuations. Table 3 confirms this across runs: DSR produces gradient norms roughly 3× lower than Hard and EROT, with within-epoch variance reduced by a similar factor. Lower gradient norms do not in themselves imply a better gradient signal, but combined with the improved performance of DSR, the pattern indicates more stable learning.

Table 3: Gradient norm statistics on Lorenz-63 (SMNR = 10 dB), averaged over runs.
<table><tr><td>Method</td><td>Mean</td><td>Within-epoch std</td></tr><tr><td>Hard</td><td>17399</td><td>17253</td></tr><tr><td>Soft</td><td>17013</td><td>8314</td></tr><tr><td>EROT</td><td>15771</td><td>6918</td></tr><tr><td>DSR</td><td>4504</td><td>2720</td></tr></table>

## 5.3.2 Scaling with particle count

To assess how each method scales with the number of particles, we train on Lorenz-63 at SMNR = 10 dB with $N \in \{ 2 5 , 5 0 , 1 0 0 , 2 0 0 \}$ , keeping all other hyperparameters fixed. DSR is the only method that trains stably at every tested N: its NMSE improves from −29.94 dB at $N = 2 5$ to −30.35 dB at $N = 2 0 0$ with run-to-run standard deviation below 0.3 dB throughout. Hard and Soft are competitive at $N \leq 1 0 0$ but collapse at $N = 2 0 0$ , with NMSE degrading $\mathrm { t o \ - 2 0 . 4 2 }$ and −21.52 dB and standard deviations above 7 and 12 dB, respectively, which is caused by training instability. EROT does not complete at $N = 2 0 0 \mathrm { : }$ the $\mathcal { O } ( N ^ { 2 } )$ transport matrix combined with 100

![](images/685ce9ba5ea734990c6772c56abe7cbba84a422e6dcf4d4138fd5e6fbe6a2892.jpg)  
Figure 4: Handwriting dynamics from the CharacterTrajectories dataset. Sample trajectory of the letter $\mathbf { \dot { a } } _ { } ^ { }$ captured at 200 Hz on a WACOM tablet. (a) state trajectory, with line thickness indicating pen pressure and colour gradient showing temporal progression from start (black dot) to finish. (b) noisy observations at $\mathrm { S M N R } = 1 0 ~ \mathrm { d B }$

Sinkhorn iterations per resampling step and the autodiff tape over $T = 1 0 0$ time steps exceeds available GPU memory. Results and a discussion are reported in Appendix F.3.

## 5.4 Character typing

To demonstrate applicability to real-world data, we evaluate on the CharacterTrajectories dataset from the UEA multivariate time series archive [1], which consists of pen-tip motion measurements captured on a WACOM tablet at 200 Hz for 26 characters from the Latin alphabet.

The raw observations $\mathbf { y } _ { t } \in \mathbb { R } ^ { 3 }$ capture 3D pen-tip position. To give the learned dynamics room to encode temporal information not directly observed, we augment the state to $\mathbf { x } _ { t } = [ \mathbf { s } _ { t } ^ { ( 1 ) } , \mathbf { s } _ { t } ^ { ( 2 ) } , \mathbf { s } _ { t } ^ { ( 3 ) } ] ^ { \top } \in$ $\mathbb { R } ^ { 9 }$ with three 3D latent components, and observe only the first:

$$
\begin{array} { r } { \mathbf { y } _ { t } = [ I _ { 3 } \quad 0 _ { 3 } \quad 0 _ { 3 } ] \mathbf { x } _ { t } + \mathbf { w } _ { t } , \qquad \mathbf { w } _ { t } \sim \mathcal { N } ( 0 , \sigma _ { w } ^ { 2 } I _ { 3 } ) , } \end{array}\tag{11}
$$

with $\sigma _ { w } ^ { 2 }$ set by the SMNR (10 dB or 0 dB). The remaining six dimensions are latent and learned end-to-end. The dataset contains 1,422 training and 1,436 test sequences of length 178; we use $N = 2 5$ particles for training and $N = 1 0 0 0$ for testing. A sample trajectory is shown in Fig. 4.

Table 4: Performance comparison of the estimated transition model on the CharacterTrajectories dataset. Results show mean ± std over 10 runs with different seeds.
<table><tr><td rowspan="2">Method</td><td colspan="2">NMSE (dB)</td></tr><tr><td> $\mathrm { S M N R } = 1 0 \ \mathrm { d B }$ </td><td> $\mathbf { S M N R } = 0 ~ \mathrm { d B }$ </td></tr><tr><td>Hard</td><td> $- 1 4 . 6 5 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $- 7 . 6 6 { \pm } 0 . 3 5$ </td></tr><tr><td>Soft</td><td> $- 1 4 . 6 9 { \pm } 0 . 2 9$ </td><td> $- 7 . 0 8 { \pm } 1 . 1 2$ </td></tr><tr><td>EROT</td><td> $- 1 4 . 8 8 { \pm } 0 . 1 7$ </td><td> $- 7 . 7 7 { \scriptstyle \pm 1 . 2 6 }$ </td></tr><tr><td>DSR</td><td> $- \mathbf { 1 5 . 2 1 } \pm 0 . 1 5$ </td><td> $- 8 . 5 9 { \scriptstyle \pm 0 . 2 2 }$ </td></tr></table>

The results can be found in Table 4. At 10 dB SMNR, all differentiable methods achieve comparable performance, with DSR obtaining marginally better NMSE (−15.21 dB) than EROT (−14.88 dB) and Soft (−14.69 dB). The non-differentiable Hard baseline (−14.65 dB) performs only marginally worse, but not significantly compared to Soft. This indicates that gradient flow through resampling aids in learning appropriate dynamics for this real-world task. However, we can observe a larger difference at 0 dB SMNR. DSR achieves −8.59 dB NMSE, which is a 0.82 dB improvement over EROT (−7.77 dB), which corresponds to a 21% reduction in MSE on average.

## 5.5 Training time comparison

Table 5 reports wall-clock training times on Lorenz-63, using NVIDIA T4 GPU. DSR adds only 9.6% overhead over Hard (2.55 h vs. 2.49 h), while EROT is more than 2× slower (5.96 h) due to the 100 Sinkhorn iterations per resampling step. Although

Table 5: Training time on Lorenz-63.
<table><tr><td>Method</td><td>Total (h)</td><td>Time/epoch (s)</td></tr><tr><td>Hard</td><td> $\mathbf { 2 . 4 9 2 0 . 1 5 }$ </td><td> $\mathbf { 1 7 . 9 4 } { \scriptstyle \pm 1 . 1 1 }$ </td></tr><tr><td>Soft</td><td> $4 . 3 3 { \scriptstyle \pm 0 . 0 5 }$ </td><td> $3 1 . 1 6 { \scriptstyle \pm 0 . 3 9 }$ </td></tr><tr><td>EROT</td><td> $5 . 9 6 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $4 2 . 9 4 { \scriptstyle \pm 0 . 7 7 }$ </td></tr><tr><td>DSR</td><td> $2 . 5 5 { \scriptstyle \pm 0 . 0 2 }$ </td><td> $\underline { { 1 8 . 3 9 } } \pm 0 . 1 6$ </td></tr></table>

DSR shares the $\mathcal { O } ( N ^ { 2 } )$ transport matrix cost with

EROT, it avoids the iterative solver: the soft assign-

ment in Eq. (5) is a single pass of sigmoid evaluations,

implemented as batched matrix operations, hence the

nested loops in Algorithm 3 do not appear in practice. This brings DSR’s per-resampling cost close to that of a discrete systematic resampling step.

The convex combination limitation. At any finite τ, DSR produces convex combinations of ancestor particles rather than exact copies, which may place mass between modes when the posterior is multimodal; the same property applies to EROT [5]. DSR’s CDF-ordered structure partially mitigates this: Morton sorting ensures that soft assignments only mix spatially nearby particles, limiting cross-mode mixing when modes are well-separated. We quantify this on a static bimodal Gaussian in Appendix G: without Morton sorting every resampled particle lands between the modes, whereas with Morton sorting the inter-modal mass drops to ≈ 0.28 and remains stable up to dimension 10, while EROT places all mass between the modes for $d \geq 3$ . Nonetheless, multimodal posteriors remain challenging for particle filters in general [33, 31], and a fuller treatment of differentiable resampling under such posteriors is an interesting direction for future work.

## 6 Conclusion

In this paper we have introduced Differentiable Systematic Resampling (DSR), a principled and computationally efficient method for enabling gradient-based learning in particle filters. DSR is a soft relaxation of systematic resampling, where a parameter is introduced that controls the tradeoff between gradient flow and exactness of systematic resampling. Unlike existing differentiable resam pling methods, DSR preserves the CDF-ordered, banded structure of systematic resampling while enabling full gradient flow through the particle filtering recursion. Our experimental results on both synthetic chaotic systems (Lorenz-63) and real-world handwriting dynamics (CharacterTrajectories) demonstrate that DSR consistently achieves higher state estimation accuracy and dynamics learning quality compared to existing methods, particularly in low signal-to-noise regimes.

## 7 Acknowledgments

The research is supported by funding and activities from Digital Futures Center, European Defence Fund REACT II project, Saab, Ericcson, VINNOVA and partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. The computing resource is provided by Chalmers e-Commons at Chalmers.

## References

[1] Anthony Bagnall, Hoang Anh Dau, Jason Lines, Michael Flynn, James Large, Aaron Bostrom, Paul Southam, and Eamonn Keogh. The uea multivariate time series classification archive, 2018, 2018.

[2] Itay Buchnik, Guy Revach, Damiano Steger, Ruud J. G. van Sloun, Tirza Routtenberg, and Nir Shlezinger. Latent-kalmannet: Learned kalman filtering for tracking from high-dimensional signals. Trans. Sig. Proc., 72:352–367, January 2024.

[3] Xiongjie Chen and Yunpeng Li. Normalizing flow-based differentiable particle filters. IEEE Transactions on Signal Processing, 73:493–507, 2025.

[4] Nicolas Chopin and Omiros Papaspiliopoulos. An Introduction to Sequential Monte Carlo. Springer Series in Statistics. Springer, 2020.

[5] Adrien Corenflos, James Thornton, George Deligiannidis, and Arnaud Doucet. Differentiable particle filtering via entropy-regularized optimal transport. In International Conference on Machine Learning, pages 2100–2111. PMLR, 2021.

[6] Domonkos Csuzdi, Olivér Töro, and Tamás Bécsi. Differentiable particle filtering using optimal placement˝ resampling. In 2024 IEEE 18th International Symposium on Applied Computational Intelligence and Informatics (SACI), pages 000183–000188, 2024.

[7] Fredrik Cumlin, Anubhab Ghosh, and Saikat Chatterjee. Dns: Data-driven nonlinear smoother for complex model-free process. In ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 356–360, 2026.

[8] Randal Douc and Olivier Cappé. Comparison of resampling schemes for particle filtering. In ISPA 2005. Proceedings of the 4th International Symposium on Image and Signal Processing and Analysis, pages 64–69. IEEE, 2005.

[9] Arnaud Doucet, Nando de Freitas, and Neil Gordon. An Introduction to Sequential Monte Carlo Methods, pages 3–14. Springer New York, New York, NY, 2001.

[10] Arnaud Doucet and Adam M. Johansen. A tutorial on particle filtering and smoothing: Fifteen years later. In Dan Crisan and Boris Rozovskii, editors, The Oxford handbook ofnonlinearfiltering, Oxford handbooks in mathematics, pages 656–705. Oxford University Press, Oxford ; N.Y., 2011.

[11] Mathieu Gerber, Nicolas Chopin, and Nick Whiteley. Negative association, ordering and convergence of resampling methods. The Annals ofStatistics, 47(4):2236 – 2260, 2019.

[12] Anubhab Ghosh, Yonina C. Eldar, and Saikat Chatterjee. Particle-based data-driven nonlinear state estimation of model-free process from nonlinear measurements. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2025.

[13] Anubhab Ghosh, Antoine Honoré, and Saikat Chatterjee. DANSE: Data-Driven Non-Linear State Estimation of Model-Free Process in Unsupervised Learning Setup. IEEE Transactions on Signal Processing, 72:1824–1838, 2024.

[14] N.J. Gordon, D.J. Salmond, and A.F.M. Smith. Novel approach to nonlinear/non-gaussian bayesian state estimation. IEE Proceedings F (Radar and Signal Processing), 140:107–113, 1993.

[15] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations, 2017.

[16] Rico Jonschkowski, Divyam Rastogi, and Oliver Brock. Differentiable particle filters: End-to-end learning with algorithmic priors. Robotics: Science and Systems XIV, 2018.

[17] Peter Karkus, David Hsu, and Wee Sun Lee. Particle filter networks with application to visual localization. In Proceedings ofThe 2nd Conference on Robot Learning, pages 169–178. PMLR, 2018.

[18] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015.

[19] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. In 2nd International Conference on Learning Representations (ICLR), 2014.

[20] Genshiro Kitagawa. Monte carlo filter and smoother for non-gaussian nonlinear state space models. Journal ofcomputational and graphical statistics, 5(1):1–25, 1996.

[21] Alina Kloss, Georg Martius, and Jeannette Bohg. How to train your differentiable filter. Autonomous Robots, 45:561–578, 2021.

[22] R. Krishnan, U. Shalit, and D. Sontag. Structured inference networks for nonlinear state space models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 31, 2017.

[23] Tuan Anh Le, Maximilian Igl, Tom Rainforth, Tom Jin, and Frank Wood. Auto-encoding sequential Monte Carlo. In International Conference on Learning Representations, 2018.

[24] Chris J. Maddison, Dieterich Lawson, George Tucker, Nicolas Heess, Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, and Yee Whye Teh. Filtering variational objectives. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 6576–6586, Red Hook, NY, USA, 2017. Curran Associates Inc.

[25] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The concrete distribution: A continuous relaxation of discrete random variables. In International Conference on Learning Representations, 2017.

[26] G. M. Morton. A computer oriented geodetic data base and a new technique in file sequencing. Technical report, International Business Machines Company, 1966.

[27] Christian Naesseth, Scott Linderman, Rajesh Ranganath, and David Blei. Variational sequential monte carlo. In International conference on artificial intelligence and statistics, pages 968–977. PMLR, 2018.

[28] Christian A. Naesseth, Fredrik Lindsten, and Thomas B. Schön. Elements of sequential monte carlo. Found. Trends Mach. Learn., 12(3):307–392, November 2019.

[29] Gustav Norén, Anubhab Ghosh, Fredrik Cumlin, and Saikat Chatterjee. Vse: Variational state estimation of complex model-free process. In ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4921–4925, 2026.

[30] Simo Särkkä. Bayesian Filtering and Smoothing. Institute of Mathematical Statistics Textbooks. Cambridge University Press, 2013.

[31] Sebastian Thrun, Dieter Fox, Wolfram Burgard, and Frank Dellaert. Robust monte carlo localization for mobile robots. Artificial Intelligence, 128(1):99–141, 2001.

[32] Wessel L. van Nierop, Nir Shlezinger, and Ruud J.G. van Sloun. Deep variational sequential monte carlo for high-dimensional observations. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2025.

[33] Vermaak, Doucet, and Perez. Maintaining multimodality through mixture tracking. In Proceedings Ninth IEEE International Conference on Computer Vision, pages 1110–1116 vol.2, 2003.

## A Differentiable Systematic Sampling Algorithm

In this section, we explain the DSR algorithm. Note that MortonSort is used by default, but is typically optional when the filtering posterior is unimodal, as particle ordering has less effect on the soft assignments in this regime (see Appendix F.4). Also note that DSR in $\mathrm { A l g } . { 3 }$ can be implemented using batched matrix operations, hence avoiding explicit loops over particles.

Algorithm 2 MortonSort: Morton Space-Filling Curve Sort.   
Require: Particles $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N }$ with $\mathbf { x } _ { i } \in \mathbb { R } ^ { d _ { x } }$ , resolution b bits   
1: for $i = 1$ to $N$ do   
2: for $d = 1$ to $d _ { x }$ do   
3: $\begin{array} { r } { c _ { i } ^ { ( d ) } = \left| \frac { x _ { i } ^ { ( d ) } - x _ { \mathrm { m i n } } ^ { ( d ) } } { x _ { \mathrm { m a x } } ^ { ( d ) } - x _ { \mathrm { m i n } } ^ { ( d ) } } \cdot \left( 2 ^ { b } - 1 \right) \right| } \end{array}$ ▷ normalize to integer grid   
4: end for   
5: Write each $c _ { i } ^ { ( d ) }$ in binary: $\begin{array} { r } { c _ { i } ^ { ( d ) } = \sum _ { k = 0 } ^ { b - 1 } c _ { i , k } ^ { ( d ) } 2 ^ { k } } \end{array}$ with $c _ { i , k } ^ { ( d ) } \in \{ 0 , 1 \}$   
6: $\begin{array} { r } { z _ { i } = \sum _ { k = 0 } ^ { b - 1 } \sum _ { d = 1 } ^ { d _ { x } } c _ { i , k } ^ { ( d ) } \cdot 2 ^ { k \cdot d _ { x } + ( d - 1 ) } } \end{array}$ ▷ interleave bits across dimensions   
7: end for   
8: $\pi =$ argsort $\mathbf { \Psi } _ { \left( \mathcal { Z } _ { 1 } , \dots , \mathcal { Z } _ { N } \right) }$   
9: return $\{ \mathbf { x } _ { \pi ( i ) } \} _ { i = 1 } ^ { N } , \pi$

```latex
Algorithm 3 Differentiable Systematic Resampling (DSR). Equation marked in green is the DSR
assignment step, and equation marked in red is the SR assignment step.
Require: Weights $\{ w _ { i } \} _ { i = 1 } ^ { N }$ , particles $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N }$ , temperature $\tau > 0 ,$ , resolution b
1: $\bar { \{ \mathbf { x } _ { i } \} } _ { i = 1 } ^ { N } , \bar { \pi } $ MortonSort $( \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N } , \bar { b } ) ;$ reorder $\{ w _ { i } \}$ accordingly
2: Compute CDF: $\begin{array} { r } { F _ { 0 } = 0 , \ F _ { i } = \sum _ { k = 1 } ^ { i } w _ { k } } \end{array}$ for $i = 1 , \ldots , N$
3: Draw $u _ { 0 } \sim$ Uniform $( 0 , \textstyle { \frac { 1 } { N } } )$
4: for $j = 1$ to $N$ do
5: $\begin{array} { r } { u _ { j } = u _ { 0 } + \frac { j - 1 } { N } } \end{array}$
6: for $i = 1$ to $\dot { N }$ do
7: $s _ { i } = \mathbb { 1 } ( F _ { i - 1 } < u _ { j } \leq F _ { i } )$ ▷ systematic resampling
8: $\begin{array} { r } { s _ { i } ( u _ { j } ; \tau ) = \sigma \Big ( \frac { u _ { j } - F _ { i - 1 } } { \tau } \Big ) - \sigma \Big ( \frac { u _ { j } - F _ { i } } { \tau } \Big ) } \end{array}$
9: end for
10: $\begin{array} { r } { T _ { i j } = s _ { i } ( u _ { j } ; \tau ) \big / \sum _ { k = 1 } ^ { N } s _ { k } ( u _ { j } ; \tau ) } \end{array}$
11: $\begin{array} { r } { \tilde { \mathbf { x } } _ { j } = \sum _ { i = 1 } ^ { N } T _ { i j } \mathbf { x } _ { i } } \end{array}$
12: end for
13: return $\{ \tilde { \mathbf { x } } _ { j } \} _ { j = 1 } ^ { N }$
```

## B Proofs of propositions

Proof of Proposition 4.1. We prove both parts of the proposition.

Part 1: Convergence of the soft transport plan.

Fix a column index $j \in [ N ]$ and define the hard systematic resampling index

$$
a _ { j } ^ { * } : = \operatorname* { m i n } \{ i \in [ N ] : F _ { i } \geq u _ { j } \} .
$$

By construction, we have

$$
F _ { a _ { j } ^ { * } - 1 } < u _ { j } \leq F _ { a _ { j } ^ { * } } .
$$

Since we can view $u _ { j }$ as a realization from a continuous distribution, the event $u _ { j } = F _ { i }$ occurs with probability zero and is ignored in the almost sure sense.

Recall that

$$
s _ { i } ( u _ { j } ; \tau ) = \sigma \bigg ( \frac { u _ { j } - F _ { i - 1 } } { \tau } \bigg ) - \sigma \bigg ( \frac { u _ { j } - F _ { i } } { \tau } \bigg ) ,
$$

where $\sigma ( x ) = ( 1 + e ^ { - x } ) ^ { - 1 }$

We analyze the limit as $\tau \to 0 ^ { + }$

Case $\mathbf { \nabla } l \colon i = a _ { j } ^ { * }$ . Since $u _ { j } > F _ { a _ { j } ^ { * } - 1 }$ and $u _ { j } < F _ { a _ { j } ^ { * } }$ almost surely, we have

$$
\frac { u _ { j } - F _ { a _ { j } ^ { * } - 1 } } { \tau }  + \infty , \qquad \frac { u _ { j } - F _ { a _ { j } ^ { * } } } { \tau }  - \infty .
$$

Therefore,

$$
\operatorname * { l i m } _ { \tau  0 ^ { + } } s _ { a _ { j } ^ { * } } ( u _ { j } ; \tau ) = 1 - 0 = 1 \quad \mathrm { a . s . }
$$

Case $2 \colon i < a _ { j } ^ { * } .$ . Then $F _ { i } < u _ { j }$ and $F _ { i - 1 } < u _ { j }$ , implying both arguments of $\sigma$ tend $\mathrm { t o } + \infty$ as $\tau \to 0 ^ { + }$ . Hence,

$$
\operatorname* { l i m } _ { \tau \to 0 ^ { + } } s _ { i } ( u _ { j } ; \tau ) = 1 - 1 = 0 .
$$

Case $3 \colon i > a _ { j } ^ { * }$ . Then $F _ { i - 1 } \geq F _ { a _ { i } ^ { * } } > u _ { j }$ , so both arguments tend $\mathrm { t o } - \infty$ , yielding

$$
\operatorname* { l i m } _ { \tau \to 0 ^ { + } } s _ { i } ( u _ { j } ; \tau ) = 0 - 0 = 0 .
$$

Combining all cases, we obtain

$$
\operatorname* { l i m } _ { \tau \to 0 ^ { + } } s _ { i } ( u _ { j } ; \tau ) = \left\{ \begin{array} { l l } { 1 , } & { i = a _ { j } ^ { \ast } , } \\ { 0 , } & { i \neq a _ { j } ^ { \ast } , } \end{array} \right. \quad \mathrm { a . s . }
$$

Since $\begin{array} { r } { \sum _ { k = 1 } ^ { N } s _ { k } ( u _ { j } ; \tau ) \to 1 } \end{array}$ almost surely, it follows that

$$
\operatorname* { l i m } _ { \tau \to 0 ^ { + } } T _ { i j } ^ { ( \tau ) } = \operatorname* { l i m } _ { \tau \to 0 ^ { + } } \frac { s _ { i } ( u _ { j } ; \tau ) } { \sum _ { k = 1 } ^ { N } s _ { k } ( u _ { j } ; \tau ) } = 1 ( i = a _ { j } ^ { * } ) \quad \mathrm { a . s . }
$$

Part 2: Convergence of the resampled particles.

Using the result above,

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { \tau  0 ^ { + } } \tilde { \mathbf { x } } _ { j } ^ { ( \tau ) } = \operatorname* { l i m } _ { \tau  0 ^ { + } } \displaystyle \sum _ { i = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } \mathbf { x } _ { i } } \\ { = \displaystyle \sum _ { i = 1 } ^ { N } ( \displaystyle \operatorname* { l i m } _ { \tau  0 ^ { + } } T _ { i j } ^ { ( \tau ) } ) \mathbf { x } _ { i } } \\ { = \mathbf { x } _ { a _ { j } ^ { * } } } \\ { = \tilde { \mathbf { x } } _ { j } ^ { * } , } \end{array}
$$

where the exchange of limit and summation is justified since the sum is finite and $T _ { i j } ^ { ( \tau ) } \in [ 0 , 1 ]$ for all $\tau > 0$ . This completes the proof. □

ProofofProposition 4.2. Recall that

$$
\tilde { \mathbf { x } } _ { j } ^ { ( \tau ) } = \sum _ { i = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } \mathbf { x } _ { i } , \qquad T _ { i j } ^ { ( \tau ) } = \frac { s _ { i } ( u _ { j } ; \tau ) } { \sum _ { k = 1 } ^ { N } s _ { k } ( u _ { j } ; \tau ) } .
$$

Let

$$
\Delta _ { j } : = \operatorname * { m i n } \{ u _ { j } - F _ { a _ { j } ^ { * } - 1 } , F _ { a _ { j } ^ { * } } - u _ { j } \} > 0 \quad \mathrm { a . s . }
$$

Since $\begin{array} { r } { \sum _ { i = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } = 1 } \end{array}$ , we have $\begin{array} { r } { \mathbf { x } _ { a _ { j } ^ { * } } = \sum _ { i = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } \mathbf { x } _ { a _ { j } ^ { * } } } \end{array}$ . Thus

$$
\tilde { \mathbf { x } } _ { j } ^ { ( \tau ) } - \mathbf { x } _ { a _ { j } ^ { * } } = \sum _ { i = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } \mathbf { x } _ { i } - \sum _ { i = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } \mathbf { x } _ { a _ { j } ^ { * } } = \sum _ { i = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } ( \mathbf { x } _ { i } - \mathbf { x } _ { a _ { j } ^ { * } } ) .
$$

Taking norms and using the triangle inequality yields

$$
\| \tilde { \mathbf { x } } _ { j } ^ { ( \tau ) } - \mathbf { x } _ { a _ { j } ^ { * } } \| _ { 2 } \leq \sum _ { i = 1 } ^ { N } T _ { i j } ^ { ( \tau ) } \| \mathbf { x } _ { i } - \mathbf { x } _ { a _ { j } ^ { * } } \| _ { 2 } \leq D \sum _ { i \neq a _ { j } ^ { * } } T _ { i j } ^ { ( \tau ) } ,
$$

where D is the diameter, $\begin{array} { r } { \operatorname* { s u p } _ { i , j } \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| _ { 2 } \leq D } \end{array}$ , and we have used the obvious equality $T _ { a _ { j } ^ { * } j } \| \mathbf { x } _ { a _ { j } ^ { * } } -$ $\mathbf { x } _ { a _ { i } ^ { * } } \Vert _ { 2 } = 0$

It therefore suffices to bound $\textstyle \sum _ { i \neq a _ { i } ^ { * } } T _ { i j } ^ { ( \tau ) }$

Lower bound on the dominant term. Using the inequalities $\sigma ( x ) \geq 1 - e ^ { - x }$ for $x \ge 0$ and $- \sigma ( x ) \geq - e ^ { x }$ for $x \leq 0$ , we obtain

$$
s _ { a _ { j } ^ { * } } ( u _ { j } ; \tau ) = \sigma \bigg ( \frac { u _ { j } - F _ { a _ { j } ^ { * } - 1 } } { \tau } \bigg ) - \sigma \bigg ( \frac { u _ { j } - F _ { a _ { j } ^ { * } } } { \tau } \bigg ) \geq 1 - e ^ { - \Delta _ { j } / \tau } - e ^ { - \Delta _ { j } / \tau } .
$$

Thus, for all $\tau \leq \Delta _ { j } / \log 4$

$$
\begin{array} { r } { s _ { a _ { j } ^ { * } } ( u _ { j } ; \tau ) \ge \frac { 1 } { 2 } . } \end{array}
$$

Upper bound on the remaining terms. Fix $i \neq a _ { j } ^ { * } . \mathrm { ~ H ~ } i < a _ { j } ^ { * }$ , then $F _ { i } \le F _ { a _ { i } ^ { * } - 1 } < u _ { j }$ , so both arguments of the sigmoids are positive and

$$
s _ { i } ( u _ { j } ; \tau ) = \sigma \bigg ( \frac { u _ { j } - F _ { i - 1 } } { \tau } \bigg ) - \sigma \bigg ( \frac { u _ { j } - F _ { i } } { \tau } \bigg ) \leq 1 - \sigma \bigg ( \frac { u _ { j } - F _ { a _ { j } ^ { * } - 1 } } { \tau } \bigg ) .
$$

Using $1 - \sigma ( x ) \le e ^ { - x }$ for $x \geq 0$ , we obtain

$$
s _ { i } ( u _ { j } ; \tau ) \leq \exp \left( - \frac { u _ { j } - F _ { a _ { j } ^ { * } - 1 } } { \tau } \right) .
$$

$\mathrm { ~ I f ~ } i > a _ { j } ^ { * }$ , then $F _ { i - 1 } \geq F _ { a _ { i } ^ { * } } > u _ { j }$ , so both arguments are negative and

$$
s _ { i } ( u _ { j } ; \tau ) \leq \sigma \bigg ( \frac { u _ { j } - F _ { a _ { j } ^ { * } } } { \tau } \bigg ) \leq \exp \bigg ( - \frac { F _ { a _ { j } ^ { * } } - u _ { j } } { \tau } \bigg ) ,
$$

where we used $\sigma ( x ) \leq e ^ { x }$ for $x \leq 0$

In both cases, we obtain the uniform bound

$$
s _ { i } ( u _ { j } ; \tau ) \le \exp \biggl ( - \frac { \Delta _ { j } } { \tau } \biggr ) .
$$

Hence,

$$
\sum _ { i \neq a _ { j } ^ { * } } s _ { i } ( u _ { j } ; \tau ) \leq N \exp \left( - \frac { \Delta _ { j } } { \tau } \right) .
$$

As $\begin{array} { r } { s _ { a _ { i } ^ { * } } ( u _ { j } ; \tau ) \ge \frac { 1 } { 2 } } \end{array}$ , it follows that $\begin{array} { r } { \sum _ { k = 1 } ^ { N } s _ { k } ( u _ { j } ; \tau ) \ge \frac { 1 } { 2 } } \end{array}$ . Combining the bounds yields

$$
\sum _ { i \neq a _ { j } ^ { \ast } } T _ { i j } ^ { ( \tau ) } = \frac { \sum _ { i \neq a _ { j } ^ { \ast } } s _ { i } ( u _ { j } ; \tau ) } { \sum _ { k = 1 } ^ { N } s _ { k } ( u _ { j } ; \tau ) } \leq \frac { N e ^ { - \Delta _ { j } / \tau } } { 1 / 2 } = 2 N \exp \left( - \frac { \Delta _ { j } } { \tau } \right) .
$$

Finally,

$$
\| \widetilde { \mathbf { x } } _ { j } ^ { ( \tau ) } - \mathbf { x } _ { a _ { j } ^ { * } } \| _ { 2 } \leq 2 D N \exp \left( - \frac { \Delta _ { j } } { \tau } \right) ,
$$

which establishes exponential convergence as $\tau \to 0 ^ { + }$

## C Architecture and training configuration

## C.1 Network Architectures

All experiments use multilayer perceptrons (MLPs) with ReLU activations to parameterize the transition and proposal models. Both models predict Gaussian distributions with diagonal covariance matrices.

Transition Model. The transition model $f _ { \theta } ( \mathbf { x } _ { t } )$ predicts $p ( \mathbf { x } _ { t + 1 } | \mathbf { x } _ { t } )$ and consists of:

• Shared trunk: Two fully-connected layers with hidden dimension h and ReLU activations

• Mean head: Linear layer mapping from hidden dimension to state dimension $d _ { x }$

• Covariance head: Linear layer mapping from hidden dimension to $d _ { x }$ , followed by softplus activation to ensure positive diagonal entries

The model outputs $( \mu _ { \theta } ( \mathbf { x } _ { t } ) , \Lambda _ { \theta } ( \mathbf { x } _ { t } ) )$ where $\Lambda _ { \theta }$ is a diagonal covariance matrix.

Proposal Model. For simplicity, the proposal model $p _ { \phi } ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } , \mathbf { y } _ { t } )$ has the same architecture as the transition model, but takes concatenated inputs $[ \mathbf { x } _ { t - 1 } , \mathbf { y } _ { t } ] .$

## C.2 Hyperparameters

Linear Gaussian SSM. The transition model is parameterized as a learnable matrix $\hat { \textbf { A } } \in \mathbb { R } ^ { 4 \times 4 }$ with a learnable diagonal covariance, and no proposal model is used (i.e., the bootstrap proposal $q ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } , \mathbf { y } _ { t } ) = p _ { \phi } ( \mathbf { x } _ { t } | \mathbf { x } _ { t - 1 } )$ is used). The true transition matrix is $\mathrm { ~ \bf ~ A ~ } = 0 . 9$ blockdia $\mathrm { g } ( \mathbf { R } ( \theta _ { 1 } ) , \mathbf { R } ( \theta _ { 2 } ) )$ with $\theta _ { 1 } ~ = ~ 0 . 3$ and $\theta _ { 2 } ~ = ~ 0 . 5$ , where $\mathbf { R } ( \theta )$ is the $2 \times 2$ rotation matrix. The process noise covariance is ${ \bf C } _ { e } = 0 . 1 \cdot { \bf I } _ { 4 }$ and observations are full-state with $\mathbf { H } = \mathbf { I } _ { 4 }$ Training uses the Adam optimizer with learning rate $1 0 ^ { - 3 }$ , batch size 64, weight decay 0, and gradient clipping at norm 5.0. We train with $N = 2 5$ particles for 500 epochs on 1000 trajectories of length $T = 1 0 0$ at SMNR = 10 dB. For DSR, we set $\tau = 0 . 1$ We clip the log-likelihood values below by −50 and −100 for 10 dB and 0 dB SMNR respectively. This avoids extreme instability during training; none of the methods converge without this.

Lorenz-63. We use $h = 3 2$ for both models. The transition model has input dimension 3 (state dimension), while the proposal model has input dimension $3 + 6 4 = 6 7$ (state plus observation). Training uses the Adam optimizer with learning rate $5 \times 1 0 ^ { - 4 }$ , batch size 64, weight decay 0, and gradient clipping at norm 5.0. We train with $N = 2 5$ particles for 500 epochs on 1000 trajectories of length $T = 1 0 0$ . For DSR, we set $\tau = 0 . 1$ across all experiments, and use 20 bits for the Morton curves.

Character Typing. We use a hidden dimension of $h = 3 2$ . The augmented state dimension is $d _ { x } = 9$ (three 3D components), with direct observations of the first 3 dimensions $( d _ { y } = 3 )$ . Training uses 25 particles, learning rate $1 \times 1 0 ^ { - 3 }$ , and batch size 64 on 1400 training sequences of length $T = 1 7 8$ All other hyperparameters match Lorenz-63.

Evaluation. We evaluate the state estimation performance using the same training setup except for the resampling method, which uses hard systematic resampling (as this resampling method is typically used in practice). We use $N = 1 0 0 0$ particles.

## D SMNR and Evaluation Measures

## D.1 SMNR

The signal-to-measurement noise ratio is given by

$$
\mathrm { S M N R } = 1 0 \log _ { 1 0 } \left( \frac { \sum _ { t = 1 } ^ { T } \mathbb { E } \left[ \left\| \mathbf { h } ( \mathbf { x } _ { t } ^ { ( j ) } ) - \mathbb { E } [ \mathbf { h } ( \mathbf { x } _ { t } ^ { ( j ) } ) ] \right\| _ { 2 } ^ { 2 } \right] } { d _ { y } \sigma _ { w } ^ { 2 } } \right) ,\tag{12}
$$

where $\mathbf { h } ( \mathbf { x } _ { t } ^ { ( j ) } )$ is the noiseless observation, $\sigma _ { w } ^ { 2 }$ is the observation noise variance per dimension, $d _ { y }$ is the observation dimension, and $N _ { \mathrm { t e s t } }$ is the number of trajectories. A low SMNR corresponds to high measurement noise.

## D.2 State Estimation

Normalized Mean Squared Error (NMSE). We evaluate state estimation accuracy using NMSE in dB:

$$
\mathrm { N M S E } = 1 0 \log _ { 1 0 } \left( \frac { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \| \mathbf { x } _ { t } - \hat { \mathbf { x } } _ { t } \| _ { 2 } ^ { 2 } } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \| \mathbf { x } _ { t } \| _ { 2 } ^ { 2 } } \right) ,\tag{13}
$$

where $\mathbf { x } _ { t }$ is the true state, $\hat { \mathbf { x } } _ { t }$ is the estimated state (the posterior mean of the particle filter), and $T$ is the trajectory length. Lower values indicate better performance.

## D.3 Transition Model Quality

One-step Mean Squared Error (MSE). Measures the quality of the learned dynamics $f _ { \theta }$ by predicting one step ahead from true states:

$$
\mathbf { M S E } = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \| \mathbf { x } _ { t + 1 } - f _ { \theta } ( \mathbf { x } _ { t } ) \| _ { 2 } ^ { 2 } .\tag{14}
$$

Lower values indicate better transition model quality.

Multi-step Mean Squared Error (MSE-5-step). Evaluates prediction over a longer horizon by rolling out the learned dynamics for 5 steps:

$$
\mathrm { M S E - 5 - s t e p } = \frac { 1 } { T - 5 } \sum _ { t = 1 } ^ { T - 5 } \| \mathbf { x } _ { t + 5 } - f _ { \theta } ^ { ( 5 ) } ( \mathbf { x } _ { t } ) \| _ { 2 } ^ { 2 } ,\tag{15}
$$

where $f _ { \theta } ^ { ( 5 ) } ( \mathbf { x } _ { t } ) = f _ { \theta } ( f _ { \theta } ( f _ { \theta } ( f _ { \theta } ( f _ { \theta } ( \mathbf { x } _ { t } ) ) ) ) )$ denotes 5 iterative applications of the learned dynamics.   
For each use of the transition model, the mean is used as input to the subsequent call of the model.

Log-Likelihood (LL). Evaluates the probabilistic quality of the learned transition model:

$$
\mathbf { L } \mathbf { L } = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } \log p _ { \theta } ( \mathbf { x } _ { t + 1 } | \mathbf { x } _ { t } ) ,\tag{16}
$$

where $p _ { \theta } ( \mathbf { x } _ { t + 1 } \vert \mathbf { x } _ { t } ) ~ = ~ \mathcal { N } ( \mathbf { x } _ { t + 1 } ; \pmb { \mu } _ { \theta } ( \mathbf { x } _ { t } ) , \pmb { \Lambda } _ { \theta } ( \mathbf { x } _ { t } ) )$ is the learned transition distribution, where $f _ { \pmb \theta } ( \mathbf x _ { t } ) = ( \pmb \mu _ { \theta } ( \mathbf x _ { t } ) , \pmb \Lambda _ { \theta } ( \mathbf x _ { t } ) )$ . Higher values indicate better probabilistic modeling.

KL Divergence (KLD). Measures the distributional mismatch between the learned dynamics and true dynamics:

$$
\mathbf { \mathrm { K L D } } = \frac { 1 } { T - 1 } \sum _ { t = 1 } ^ { T - 1 } D _ { \mathrm { K L } } \big ( p _ { \mathrm { t r u e } } ( \mathbf { x } _ { t + 1 } | \mathbf { x } _ { t } ) \| p _ { \theta } ( \mathbf { x } _ { t + 1 } | \mathbf { x } _ { t } ) \big ) .\tag{17}
$$

Lower values indicate a closer match to the true transition distribution.

## E Empirical Analysis of the DSR Relaxation

The theoretical results in Section 4 establish that DSR converges to systematic resampling as $\tau \to 0 ^ { + }$ with exponential rates. In this section, we complement the theory with an empirical study of the DSR transport plan and the resampled particles. We use a sequential importance resampling (SIR) particle filter with systematic resampling [10] on the Lorenz-63 system with oracle (true) dynamics, at 10 dB SMNR with the Gaussian point spread observation map. This isolates the resampling relaxation from the dynamics learning, so we can study DSR as an approximation to systematic resampling directly. We vary $\tau \in \{ 0 . 0 1 , \bar { 0 . } 0 5 , 0 . 1 , 0 . 5 , 1 . 0 \}$ and $N \in \{ 2 5 , 5 0$ , 100, 200}, and report results averaged over 10 sequences of length $T = 1 0 0$

## E.1 Transport Plan Visualization

Figure 5 shows the DSR transportation matrices $T ^ { ( \tau ) }$ for varying $\tau ,$ alongside hard systematic resampling and EROT $( \varepsilon = 0 . 5 )$ , at a representative time step with $N = 2 5$ particles. $\mathbf { A t } \boldsymbol { \tau } = 0 . 0 1$ the DSR transport plan closely resembles the sparse, block-diagonal structure of hard systematic resampling, with mass concentrated on a single ancestor per resampled particle. $\mathbf { A } \mathbf { s } \ \tau$ grows, the plan becomes more diffuse and spreads mass across neighbouring ancestors. $\mathbf { A t } \tau = 0 . 5$ , the plan becomes visibly banded: a high-weight ancestor contributes to many resampled particles, so DSR becomes less restrictive about spatial locality. This brings DSR qualitatively closer to EROT. A main difference is that DSR exhibits lower column-wise spread, which suggests each resampled particle is drawn from a larger set of ancestors. A lower column-wise spread implies that the resampling linear combination has more even weights, which in turn becomes less restrictive to spatial locality. On the other hand, EROT’s transport plan bears little resemblance to systematic resampling overall, whereas DSR retains the CDF-ordered banded structure even at high τ.

![](images/8e7ba3b7a6f372f3bdd1ee8869a970b78591fe38e861c3a2fc5a16c300fff230.jpg)  
Figure 5: Transportation maps for hard systematic resampling and DSR at varying temperatures $\tau \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 5 , 1 . 0 \}$ , alongside EROT $( \varepsilon = 0 . 5 )$ , for a SIR particle filter of the Lorenz-63 system with $N = 2 5$ particles.

## E.2 Effective Number of Ancestors

To quantify the sharpness of the transport plan, we compute the effective number of ancestors for each resampled particle $j \colon$

$$
N _ { \mathrm { e f f } , j } = \mathrm { e x p } \left( - \sum _ { i = 1 } ^ { N } T _ { i j } \log T _ { i j } \right) ,
$$

averaged over particles and time steps. For hard systematic resampling, $N _ { \mathrm { e f f } , j } = 1$ , as each resampled particle copies exactly one ancestor.

Figure 6a shows $N _ { \mathrm { e f f } }$ as a function of τ for different particle counts. At low temperatures, the DSR transport plan is sharply concentrated: $N _ { \mathrm { e f f } } = 1 . 7 0 \mathrm { a t } \tau = 0 . 0 1$ with $N = 2 5$ , close to the hard SR limit. As τ grows, $N _ { \mathrm { e f f } }$ grows monotonically, and for fixed τ it scales with N. This is consistent with the N-dependent factor in Proposition 4.2: as N grows, the CDF intervals $[ F _ { i - 1 } , F _ { i } ]$ become narrower, so more ancestors fall within the sigmoid bandwidth τ and contribute to each resampled particle.

For comparison, EROT $( \varepsilon ~ = ~ 0 . 5 )$ yields $N _ { \mathrm { e f f } }$ of 9.83, 18.36, 36.41, and 71.77 for $N \_ =$ 25, 50, 100, 200. At the operating temperature $\tau = 0 . 1$ used in all VSMC experiments, DSR is substantially sparser than EROT across all particle counts (e.g., 6.54 vs. 9.83 for $N = 2 5 ; 2 3 . 1 1 \mathrm { { v s } }$ 36.41 for $\dot { N } = 1 0 0 )$

## E.3 Correlation Between DSR and Systematic Resampling

The DSR resampled particles $\tilde { \mathbf { x } } _ { i } ^ { ( \tau ) }$ are continuous mixtures of ancestors rather than discrete copies. To measure whether DSR picks the same ancestors as hard systematic resampling — that is, whether the two methods assign each column j to the same region of the CDF — we compute the Pearson correlation between the DSR and hard SR particle sets, component-wise and averaged over dimensions, particles, and time steps. Note that a high correlation indicates agreement in which ancestor is selected per column; it does not imply that the resampled particles are numerically close, since a DSR particle can be a convex combination of several ancestors around the hard SR choice and still correlate strongly. The per-particle L2 distance, reported in the next subsection, measures that numerical closeness directly.

The correlation stays above 0.95 in every configuration. $\mathbf { A t } \tau = 0 . 0 1$ , it reaches 0.965 for $N = 2 5$ and 0.961 for $N = 2 0 0$ . Even at $\tau = 1 . 0 $ , the correlation stays above 0.956. So while the transport

![](images/6f283440ff49dba8b8bba80c8225fdf461aa67139fd6edbf2bcb0b269626d098.jpg)  
(a) Effective number of ancestors $N _ { \mathrm { e f f } }$ as a function of temperature τ for varying particle counts N. The dashed line indicates hard systematic resampling $( N _ { \mathrm { e f f } } = 1 )$ .

![](images/da9242173e71ebceb1089b88e1e3eb25062b5f6f31d238a4022418fa92ea6b35.jpg)  
(b) Per-particle L2 distance between DSR and hard SR resampled particles. Solid lines use Morton sorting; dashed lines use random particle ordering.

Figure 6: Empirical analysis of the DSR relaxation on the Lorenz-63 system with oracle dynamics at $\mathrm { S i n i v R } = 1 0 \mathrm { \bar { d } B }$ . Error bars show ±1 standard deviation over 10 sequences.

plan becomes more diffuse at higher τ, DSR continues to select ancestors from the same CDF regions as hard systematic resampling, because the soft assignments still respect the CDF ordering.

## E.4 L2 Distance and Effect of Morton Sorting

We measure the per-particle L2 distance between DSR and hard SR resampled particles:

$$
\mathrm { L } 2 _ { \mathrm { p a r t i c l e s } } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \| \tilde { \mathbf { x } } _ { j } ^ { ( \tau ) } - \mathbf { x } _ { a _ { j } ^ { * } } \| _ { 2 } ,
$$

where $\mathbf { x } _ { a _ { i } ^ { * } }$ is the resampled particle by systematic resampling. This directly relates to the bound in Proposition 4.2. Further, to assess the role of particle ordering, we compare Morton space-filling curve sorting (used in all VSMC experiments) against random particle ordering.

Figure 6b shows the L2 distance as a function of τ for both orderings. As suggested in Proposition 4.2, the distance grows with τ under Morton sorting: for $N = 2 5$ , it increases from $0 . 1 0 \mathrm { ~ a t } \tau = 0 . 0 1 \mathrm { ~ t ~ }$ o 0.51 at $\tau = 0 . 5$ . Morton sorting consistently reduces the L2 distance relative to random ordering, and the gap is largest at low τ and high N. $\dot { \mathrm { A t } } \tau = 0 . 0 1$ with $N = 2 0 0$ , Morton sorting reduces the distance from 0.38 to 0.25, a 35% reduction. The reason is that at low τ , each resampled particle draws mass primarily from its CDF neighbours, and Morton sorting ensures those neighbours are also spatially close, keeping the mixture near the hard ancestor. At higher temperatures $( \tau \geq 0 . 5 )$ , the plan is diffuse enough that ordering matters less.

## E.5 Bias and Variance of DSR Estimates

Here we study the bias and variance of DSR. For resampling schemes, a typical study is the study of how $\begin{array} { r } { I = \sum _ { i } \dot { w } _ { i } f ( \mathbf { x } _ { i } ) \left( f \right. } \end{array}$ is an arbitrary function) changes after resampling, in particular, studying how I behaves after resampling. To quantify the effect, we measure the bias, variance, and MSE of a test-function f estimate under each scheme. Fixing a particle set and weights, over 20000 seeds we compute $\begin{array} { r } { \hat { I } = \sum _ { i } \tilde { w } _ { j } f ( \tilde { \mathbf { x } } _ { j } ) } \end{array}$ with equal weights $\tilde { w } _ { j } = 1 / N$ after resampling, and compare against the weighted estimate $\begin{array} { r } { I = \sum _ { i } w _ { i } f ( \mathbf { x } _ { i } ) } \end{array}$ . We use $N = 1 0 0$ particles with $\mathbf { x } _ { j } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 3 } )$ and $w _ { j }$ the likelihood, and two test functions, $f ( \mathbf { x } ) = \mathbf { x }$ and $f ( \mathbf { x } ) = \mathbf { x } ^ { 2 }$ (elementwise). Hard systematic and multinomial resampling are unbiased, so their reported bias is Monte-Carlo error. Results are in Table 6.

For the linear function $f ( \mathbf { x } ) = \mathbf { x }$ , DSR trades variance for bias at every τ : the variance is two to five orders of magnitude below hard systematic resampling, since a convex combination is an average, and DSR attains lower MSE at every τ. Here the bias is non-monotone in τ, because both limits are favourable for a linear function: $\tau \to 0 ^ { + }$ recovers systematic resampling, while large τ drives every output toward the weighted mean, which estimates a linear function exactly (here $I \approx 0 ,$ , so the contraction at $\tau = 1 . 0$ is favourable and yields the lowest MSE). This should not be read as $\tau = 1 . 0$ being preferable in general, since the same contraction degrades nonlinear functions: for $f ( \mathbf { x } ) = \mathbf { x } ^ { 2 }$ the bias increases monotonically with τ. The variance reduction is structural, reflecting the CDF-ordered, banded transport of DSR (see also Section 3.1 on expected replication counts).

Table 6: Bias, variance, and MSE of resampling estimates against the weighted estimate I, over 20000 seeds with $N = 1 0 0 .$ Hard SR and multinomial are unbiased (their bias is Monte-Carlo error).
<table><tr><td></td><td colspan="3"> $f ( \mathbf { x } ) = \mathbf { x }$ </td><td colspan="4"> $f ( \mathbf { x } ) = \mathbf { x } ^ { 2 }$ </td></tr><tr><td>Method</td><td>bias</td><td>var</td><td>MSE</td><td>bias</td><td></td><td>var</td><td>MSE</td></tr><tr><td>Hard SR</td><td> $2 . 3 \times 1 0 ^ { - 4 }$ </td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td><td> $1 . 2 \times 1 0 ^ { - 3 }$ </td><td> $1 . 2 \times 1 0 ^ { - 4 }$ </td><td></td><td> $7 . 0 \times 1 0 ^ { - 4 }$ </td><td> $7 . 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Multinomial</td><td> $4 . 9 \times 1 0 ^ { - 4 }$ </td><td> $5 . 5 \times 1 0 ^ { - 3 }$ </td><td> $5 . 5 \times 1 0 ^ { - 3 }$ </td><td> $2 . 8 \times 1 0 ^ { - 4 }$ </td><td></td><td> $2 . 6 \times 1 0 ^ { - 3 }$ </td><td> $2 . 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>DSR  $( \tau = 0 . 0 1 )$ </td><td> $2 . 4 \times 1 0 ^ { - 3 }$ </td><td> $2 . 9 \times 1 0 ^ { - 5 }$ </td><td> $3 . 4 \times 1 0 ^ { - 5 }$ </td><td> $1 . 2 \times 1 0 ^ { - 1 }$ </td><td></td><td> $3 . 8 \times 1 0 ^ { - 7 }$ </td><td> $1 . 4 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>DSR  $( \tau = 0 . 0 5 )$ </td><td> $1 . 2 \times 1 0 ^ { - 2 }$ </td><td> $8 . 6 \times 1 0 ^ { - 6 }$ </td><td> $1 . 6 \times 1 0 ^ { - 4 }$ </td><td> $2 . 3 \times 1 0 ^ { - 1 }$ </td><td></td><td> $3 . 1 \times 1 0 ^ { - 7 }$ </td><td> $5 . 3 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>DSR  $( \tau = 0 . 1 )$ </td><td> $2 . 0 \times 1 0 ^ { - 2 }$ </td><td> $4 . 7 \times 1 0 ^ { - 6 }$ </td><td> $3 . 9 \times 1 0 ^ { - 4 }$ </td><td> $2 . 7 \times 1 0 ^ { - 1 }$ </td><td></td><td> $2 . 0 \times 1 0 ^ { - 7 }$ </td><td> $7 . 3 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>DSR  $( \tau = 0 . 5 )$ </td><td> $1 . 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 6 \times 1 0 ^ { - 7 }$ </td><td> $1 . 4 \times 1 0 ^ { - 4 }$ </td><td> $3 . 1 \times 1 0 ^ { - 1 }$ </td><td></td><td> $2 . 2 \times 1 0 ^ { - 9 }$ </td><td> $9 . 9 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>DSR  $( \tau = 1 . 0 )$ </td><td> $3 . 6 \times 1 0 ^ { - 3 }$ </td><td> $1 . 3 \times 1 0 ^ { - 8 }$ </td><td> $1 . 3 \times 1 0 ^ { - 5 }$ </td><td> $3 . 2 \times 1 0 ^ { - 1 }$ </td><td></td><td> $2 . 0 \times 1 0 ^ { - 1 0 }$ </td><td> $1 . 0 \times 1 0 ^ { - 1 }$ </td></tr></table>

## E.6 Discussion

The empirical analysis supports the theoretical predictions of Section 4. The DSR transport plan converges to the sparse structure of systematic resampling as $\tau  0 ^ { + }$ , with $N _ { \mathrm { e f f } } \  \ 1$ at low temperatures. At the operating temperature $\tau = 0 . 1$ , DSR is substantially sparser than EROT while staying highly correlated $\left( > 0 . 9 5 \right)$ with hard systematic resampling. Morton sorting reduces the approximation error, particularly at low τ and high N, confirming that locality-preserving orderings complement the DSR relaxation. Together, these empirical results support that DSR soft relaxation of systematic resampling and highlights differences at different values of τ and N.

## F Ablation Studies

## F.1 Dependence on τ

We study the effect of the temperature parameter τ on filtering performance in the Lorenz-63 dataset at $\mathrm { S M N R } = 1 0 ~ \mathrm { d B }$ . Table 7 reports NMSE results for different temperature values.

Table 7: Ablation study of the temperature parameter τ on Lorenz-63 (SMNR = 10 dB).
<table><tr><td>Temperature τ</td><td>NMSE (dB)</td></tr><tr><td>0.01</td><td> $- 2 8 . 3 1 \pm 0 . 1 5$ </td></tr><tr><td>0.05</td><td> $- 2 9 . 5 6 \pm 0 . 3 4$ </td></tr><tr><td>0.10</td><td> $- 2 9 . 9 4 \pm 0 . 2 8$ </td></tr><tr><td>0.50</td><td> $\mathbf { \overline { { - 3 0 . 2 1 } } \pm 0 . 2 9 }$ </td></tr><tr><td>1.00</td><td> $- 2 9 . 9 1 \pm 0 . 2 8$ </td></tr></table>

$\mathbf { A t } \boldsymbol { \tau } = 0 . 0 1$ , NMSE is notably worse (−28.31 dB) than at larger temperatures. This is consistent with the $\mathcal { O } ( 1 / \tau )$ scaling of the sigmoid derivative,

$$
{ \sigma } ^ { \prime } ( x ) = \frac { 1 } { \tau } { \sigma } ( x ) \bigl ( 1 - \sigma ( x ) \bigr ) ,
$$

which causes gradient magnitudes to explode during backpropagation at small τ, leading to unstable optimization. As τ grows, gradient magnitudes become more moderate and training stabilizes. Performance plateaus for $\tau \in [ 0 . 0 5 , 1 . 0 ]$ , with NMSE varying by less than 0.7 dB across this range and consistently low run-to-run variance. Based on these results, we fix $\tau = 0 . 1$ for all experiments;

this is a conservative choice within the stable range that keeps the soft transport plan close to hard systematic resampling (see Appendix E).

Beyond NMSE, the temperature τ governs a bias–variance tradeoff in the gradient itself. Table 8 reports, for the same setup, the gradient-norm statistics and the per-particle ${ \bar { L } } _ { 2 }$ distance of the DSR transport to hard systematic resampling (Appendix E) as a function of τ. The two quantities move in opposite directions: as τ shrinks, the approximation error $L _ { 2 }$ decreases $( 0 . 5 0 5  0 . 1 0 2 )$ while the median gradient norm and within-epoch standard deviation increase. The ratio of median withinepoch std to median gradient norm stays close to 0.45 for $\tau \geq 0 . 0 5$ , so gradients grow but remain comparably well behaved; at $\tau = 0 . 0 1$ the ratio rises to $0 . 5 5 ,$ , indicating larger variance relative to norm and less stable training, consistent with the degraded NMSE at $\tau = 0 . 0 1$ . We read $\tau = 0 . 0 1$ as a change of regime rather than of scale.

Table 8: Gradient behaviour and approximation error as a function of τ on Lorenz-63 (SMNR = 10 dB). The $L _ { 2 }$ column is the per-particle distance of the DSR transport to hard systematic resampling from the same particle set and offset (Appendix E).
<table><tr><td colspan="2"></td><td colspan="2">Gradient norm</td><td colspan="2">Within-epoch std</td><td rowspan="2"> $L _ { 2 }$  to hard SR</td></tr><tr><td>T</td><td>NMSE (dB)</td><td>Median</td><td>Mean</td><td>Median</td><td>Mean</td></tr><tr><td>0.01</td><td> $- 2 8 . 3 1 { \pm } 0 . 1 5$ </td><td>14842</td><td> $2 . 3 9 \times 1 0 ^ { 5 }$ </td><td>8163</td><td> $4 . 3 5 \times 1 0 ^ { 5 }$ </td><td>0.102±0.009</td></tr><tr><td>0.05</td><td> $- 2 9 . 5 6 { \scriptstyle \pm 0 . 3 4 }$ </td><td>6457</td><td>7228</td><td>3009</td><td>3676</td><td>0.277±0.012</td></tr><tr><td>0.10</td><td> $- 2 9 . 9 4 { \pm } 0 . 2 8 $ </td><td>4174</td><td>4504</td><td>1885</td><td>2634</td><td>0.356±0.017</td></tr><tr><td>0.50</td><td> $\mathbf { - 3 0 . 2 1 } { \scriptstyle \pm 0 . 2 9 }$ </td><td>2546</td><td>2951</td><td>1155</td><td>1985</td><td>0.505±0.026</td></tr><tr><td>1.00</td><td> $- 2 9 . 9 1 { \scriptstyle \pm 0 . 2 8 }$ </td><td>2306</td><td>3041</td><td>1027</td><td>3453</td><td> $0 . 4 7 9 { \scriptstyle \pm 0 . 0 4 6 }$ </td></tr></table>

## F.2 Annealing τ during training

The temperature is fixed at $\tau = 0 . 1$ in all main experiments. Here we test annealing τ over the course of training: starting at a larger τ for smoother gradients and decreasing it toward a sharper, more faithful assignment. We anneal τ linearly from 1.0 to 0.1 over 300 epochs (training is done for 500 epochs), and evaluate on Lorenz-63 at SMNR = 10 dB under the camera measurement map, with all other parameters at default. Table 9 compares the annealed schedule against the fixed $\tau = 0 . 1$ setting.

Table 9: Fixed versus annealed τ on Lorenz-63 (SMNR = 10 dB). Results show mean ± std over 10 runs.
<table><tr><td rowspan="3">Setting</td><td>State Est.</td><td colspan="4">Transition Model Quality</td></tr><tr><td>NMSE (dB)</td><td>MSE</td><td> $\mathrm { M S E - 5 - s t e p }$ </td><td>LL</td><td>KLD</td></tr><tr><td>Fixed  $( \tau = 0 . 1 )$ </td><td> $- 2 9 . 9 4 { \pm } 0 . 2 8 $ </td><td> $0 . 4 7 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $3 . 3 8 { \scriptstyle \pm 2 . 1 1 }$ </td><td> $- 1 . 9 8 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $3 . 6 0 { \scriptstyle \pm 1 . 0 4 }$ </td></tr><tr><td>Annealed</td><td> $- 3 0 . 3 3 { \pm } 0 . 1 0$ </td><td> $0 . 4 0 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $3 . 0 1 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $- 1 . 8 6 { \pm } 0 . 1 3$ </td><td> $3 . 1 5 { \pm } 0 . 3 2$ </td></tr></table>

Annealing gives a small but consistent improvement over the fixed setting across all metrics, and notably reduces run-to-run variance (e.g. NMSE std 0.10 versus 0.28 for fixed τ). This indicates that annealing can benefit learning; we leave a systematic study of annealing schedules for future work.

## F.3 Dependence on N

We study the effect of the number of particles N on filtering and dynamics learning performance for all resampling methods on the Lorenz-63 system at $\mathbf { S M N R } = 1 0$ dB using the Gaussian point spread observation map. All other hyperparameters are kept fixed. Results are shown in Table 10.

DSR is the only method that scales gracefully across all tested particle counts. Its NMSE improves from −29.94 dB at $N = 2 5$ to around −30.4 dB from $N = 5 0$ onwards and plateaus there, with consistently low variance (≤ 0.28 dB std) across runs. The transition model metrics show the same trend: MSE, LL, and KLD all improve with N and reach their best values at $N = 2 0 0$ (except MSE-5-step, which generally has a high variance).

By comparison, the other methods either fail to train stably or run out of memory. Hard and Soft are competitive at $N \leq 1 0 0$ , but at $N = 2 0 0$ their performance collapses, with NMSE degrading to −20.42 and −21.52 dB and standard deviations above 7 and 12 dB across runs. These large variances are the result of training instability; there is a large variance across runs; for example, one run for Soft achieved an NMSE of 0.23 dB. A likely cause is that, at large N, the variance of the biased estimator $\nabla { \mathcal { L } } \approx g _ { \mathrm { r e p } }$ (used by Hard) grows because more ancestor decisions are dropped from the gradient graph, and Soft’s importance-weight correction becomes numerically less stable as the uniform component of its sampling distribution $( 1 - \alpha ) / N$ shrinks. EROT does not complete at $N = 2 0 0$ because the $\mathcal { O } ( N ^ { 2 } )$ transport matrix, combined with 100 Sinkhorn iterations per resampling step and the autodiff tape over $T = 1 0 0$ time steps, exceeds the available GPU memory.

Table 10: Effect of particle count N on resampling methods for Lorenz-63 at SMNR = 10 dB. Results show mean ± std over 10 runs. OOM indicates that training exceeded available GPU memory.
<table><tr><td colspan="2"></td><td>State Est.</td><td colspan="4">Transition Model Quality</td></tr><tr><td>N</td><td>Method</td><td>NMSE (dB)</td><td>MSE</td><td> $\mathrm { M S E - 5 - s t e p }$ </td><td>LL</td><td>KLD</td></tr><tr><td>25</td><td>Hard</td><td> $- 2 8 . 8 6 { \pm } 0 . 4 1$ </td><td> $\underline { { 0 . 5 5 } } \pm 0 . 1 9$ </td><td> $5 . 5 0 { \scriptstyle \pm 3 . 6 8 }$ </td><td> $- 2 . 2 6 { \pm } 0 . 2 5$ </td><td> $5 . 4 1 { \scriptstyle \pm 1 . 4 7 }$ </td></tr><tr><td rowspan="4"></td><td>Soft</td><td> $- 2 7 . 8 2 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $0 . 8 6 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $6 . 4 5 { \scriptstyle \pm 3 . 9 2 }$ </td><td> $- 2 . 5 5 { \pm } 0 . 4 8$ </td><td> $6 . 9 0 { \scriptstyle \pm 2 . 4 3 }$ </td></tr><tr><td>EROT</td><td> $- 2 8 . 7 6 { \scriptstyle \pm 0 . 3 1 }$ </td><td> $0 . 5 6 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $3 . 3 9 { \scriptstyle \pm 2 . 5 9 }$ </td><td> $- 2 . 3 2 { \pm } 0 . 2 0$ </td><td> $5 . 7 4 { \pm } 1 . 0 3 $ </td></tr><tr><td>DSR</td><td> ${ \bf 2 9 . 9 4 } { \bf \pm } { 0 . 2 8 }$ </td><td> $\mathbf { 0 . 4 7 { \scriptstyle \pm 0 . 2 4 } }$ </td><td> ${ \bf 3 . 3 8 { \scriptstyle \pm 2 . 1 1 } }$ </td><td> $\mathbf { - 1 . 9 8 { \scriptstyle \pm 0 . 2 7 } }$ </td><td> $\mathbf { 3 . 6 0 } \pm 1 . 0 4$ </td></tr><tr><td>Hard</td><td> $- 2 8 . 3 0 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $\mathbf { 0 . 3 9 2 0 . 0 8 }$ </td><td> ${ \bf 1 . 7 3 { \scriptstyle \pm 0 . 6 1 } }$ </td><td> $- 1 . 9 8 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $4 . 3 0 { \scriptstyle \pm 0 . 6 6 }$ </td></tr><tr><td>50</td><td>Soft</td><td> $- 2 8 . 3 0 { \scriptstyle \pm 0 . 9 7 }$ </td><td> $0 . 5 9 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $5 . 7 8 { \scriptstyle \pm 3 . 1 9 }$ </td><td> $- 2 . 3 3 { \pm } 0 . 4 5$ </td><td> $6 . 2 7 { \scriptstyle \pm 2 . 7 1 }$ </td></tr><tr><td></td><td>EROT</td><td> $- 2 7 . 9 1 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $0 . 5 9 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $5 . 2 2 { \scriptstyle \pm 2 . 9 5 }$ </td><td> $- 2 . 4 9 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $6 . 8 8 { \scriptstyle \pm 1 . 4 4 }$ </td></tr><tr><td></td><td>DSR</td><td> $- 3 0 . 3 4 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $0 . 4 1 { \pm } 0 . 1 3$ </td><td> $1 . 9 5 { \scriptstyle \pm 1 . 0 4 }$ </td><td> $\mathbf { - 1 . 7 7 { \pm } 0 . 1 5 }$ </td><td> $\mathbf { 2 . 8 6 { \scriptstyle \pm 0 . 6 7 } }$ </td></tr><tr><td>100</td><td>Hard</td><td> $- 2 8 . 3 4 { \pm } 0 . 4 8 $ </td><td> $0 . 7 9 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $4 . 6 5 { \pm } 1 . 5 2 $ </td><td> $- 2 . 6 1 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $7 . 2 0 { \scriptstyle \pm 3 . 4 2 }$ </td></tr><tr><td rowspan="4"></td><td>Soft</td><td> $- 2 8 . 9 3 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $0 . 5 0 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $3 . 3 5 { \pm } 1 . 7 4 $ </td><td> $- 2 . 0 0 { \pm } 0 . 3 0$ </td><td> $3 . 6 1 { \scriptstyle \pm 0 . 7 5 }$ </td></tr><tr><td>EROT</td><td> $- 2 7 . 5 5 { \pm } 0 . 9 2$ </td><td> $\mathbf { 0 . 3 9 2 0 . 1 9 }$ </td><td> $\mathbf { 1 . 8 6 \pm 1 . 2 0 }$ </td><td> $- 2 . 0 3 { \pm } 0 . 2 5$ </td><td> $4 . 2 9 { \scriptstyle \pm 0 . 9 6 }$ </td></tr><tr><td>DSR</td><td> $- 3 0 . 4 2 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $\underline { { 0 . 4 4 } } \pm 0 . 1 9$ </td><td> $2 . 3 8 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $\mathbf { - 1 . 8 0 { \pm } 0 . 2 8 }$ </td><td> $\mathbf { 2 . 8 2 \pm 0 . 5 2 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">200</td><td>Hard Soft</td><td> $- 2 0 . 4 2 { \scriptstyle \pm 7 . 2 9 }$ </td><td> $0 . 7 5 { \scriptstyle \pm 0 . 3 2 }$ </td><td> $8 . 9 2 { \scriptstyle \pm 4 . 9 6 }$   $5 . 3 9 { \pm } 4 . 6 5$ </td><td> $- 2 . 5 4 { \pm } 0 . 3 0$ </td><td> $\underline { { 8 . 2 7 } } \pm 2 . 7 4$ </td></tr><tr><td>EROT</td><td> $- 2 1 . 5 2 { \scriptstyle \pm 1 2 . 2 0 }$ </td><td> $0 . 6 6 { \scriptstyle \pm 0 . 4 3 }$ </td><td>OOM</td><td> $- 2 . 5 4 { \pm } 0 . 6 6$ </td><td> $9 . 6 2 { \scriptstyle \pm 9 . 4 4 }$ </td></tr><tr><td></td><td>OOM</td><td>OOM</td><td></td><td>OOM</td><td>OOM</td></tr><tr><td>DSR</td><td> $\mathbf { - 3 0 . 3 5 { \pm } } 0 . 2 2$ </td><td> $\mathbf { 0 . 3 7 { \scriptstyle \pm 0 . 1 7 } }$ </td><td> $\mathbf { 3 . 2 7 \pm 2 . 2 3 }$ </td><td> $\mathbf { - 1 . 7 3 { \pm } 0 . 2 3 }$ </td><td> $\mathbf { 2 . 5 2 { \scriptstyle \pm 0 . 4 1 } }$ </td></tr></table>

These results illustrate two distinct scaling properties of DSR. First, it provides a stable gradient signal across particle counts, whereas Hard and Soft degrade. Second, it avoids the memory overhead of iterative solvers, whereas EROT becomes infeasible. The relaxation parameter (τ) of DSR is kept fixed at 0.1 across all N.

## F.4 Dependence on spatial sorting of CDF

We now study the effect of Morton sorting on VSMC performance. As discussed, DSR operates on the empirical CDF, and the particle ordering determines which ancestors interact through the soft transport plan. Morton space-filling curve sorting places CDF neighbors close in state space, producing a transport plan that more closely approximates hard systematic resampling (Appendix E). The alternative is the default (random) particle ordering, which allows soft assignments to mix spatially distant particles. We compare the two orderings with DSR on Lorenz-63 under the Gaussian point spread observation map (10).

The results can be seen in Table 11. At 10 dB, the two orderings are essentially indistinguishable, with overlapping error bars on every metric. At 0 dB, the picture is mixed: Morton sorting gives a better transition model (MSE-5-step of 3.70 vs. 6.55), while random ordering gives marginally better state estimation $( - 1 8 . 4 3 \ \mathrm { v s . - 1 8 . 2 7 }$ dB NMSE), though this is within one standard deviation. A plausible reading is that the two orderings trade off along the dual role of the soft transport plan. Morton sorting produces a closer approximation to hard systematic resampling, preserving the spatial coherence that helps dynamics learning when weights become concentrated. Random ordering mixes more distant particles, which softens the relaxation further; this extra diversity may help state estimation through broader particle coverage, but at the cost of a less structured gradient signal for the transition model. Overall, DSR is relatively robust to the choice of ordering, with Morton sorting offering a modest advantage for transition model learning at low SNR. Studying the interaction between particle ordering and posterior geometry, for instance, under multimodal or heavy-tailed posteriors, is an interesting direction for future work.

Table 11: Effect of particle ordering on DSR performance for Lorenz-63 at SMNR = 10 and 0 dB. Results show mean ± std over 10 runs.
<table><tr><td rowspan="2">Ordering NMSE (dB)</td><td rowspan="2">State Est.</td><td colspan="4">Transition Model Quality</td></tr><tr><td>MSE</td><td> $\mathrm { M S E - 5 - s t e p }$ </td><td>LL</td><td>KLD</td></tr><tr><td></td><td></td><td> $\mathbf { S M N R } = 1 0 \mathbf { \Gamma } \mathbf { d B }$ </td><td></td><td></td><td></td></tr><tr><td>Morton</td><td> $- 2 9 . 9 4 { \pm } 0 . 2 8 $ </td><td> $0 . 4 7 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $3 . 3 8 { \scriptstyle \pm 2 . 1 1 }$ </td><td> $- 1 . 9 8 { \scriptstyle \pm 0 . 2 7 }$ </td><td> $3 . 6 0 { \scriptstyle \pm 1 . 0 4 }$ </td></tr><tr><td>Random</td><td> $- 2 9 . 9 6 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $0 . 4 4 { \scriptstyle \pm 0 . 2 0 }$ </td><td> $2 . 3 1 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $- 1 . 9 2 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $3 . 5 9 { \scriptstyle \pm 0 . 8 0 }$ </td></tr><tr><td></td><td></td><td> $\mathrm { S M N R } = 0 ~ \mathrm { d B }$ </td><td></td><td></td><td></td></tr><tr><td>Morton</td><td> $- 1 8 . 2 7 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $0 . 5 4 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $3 . 7 0 { \scriptstyle \pm 2 . 8 7 }$ </td><td> $- 3 . 0 6 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $1 3 . 4 1 { \scriptstyle \pm 1 . 9 4 }$ </td></tr><tr><td>Random</td><td> $- 1 8 . 4 3 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $0 . 6 8 { \scriptstyle \pm 0 . 1 8 }$ </td><td> $6 . 5 5 { \scriptstyle \pm 4 . 6 4 }$ </td><td> $- 3 . 1 9 { \pm } 0 . 1 5$ </td><td> $1 4 . 4 3 { \scriptstyle \pm 2 . 2 3 }$ </td></tr></table>

## F.5 EROT regularization sensitivity at low SNR

Table 12 reports EROT performance on Lorenz-63 at 0 dB SMNR for $\varepsilon \in \{ 0 . 5 , 1 . 0 \}$ , alongside DSR for reference. The default $\varepsilon = 0 . 5 ,$ , as used in [5, 32], exhibits severe training instability at this noise level, with MSE-5-step standard deviation exceeding the mean. Increasing ε to 1.0 stabilizes training and improves mean performance across all metrics, but DSR still outperforms EROT on every measure. Training time is essentially unchanged between the two ε settings, as EROT’s cost is dominated by the fixed 100 Sinkhorn iterations; albeit a small improvement, which could be explained by faster convergence of the Sinkhorn algorithm.

Table 12: EROT regularization sensitivity on Lorenz-63 at SMNR = 0 dB. Results show mean ± std over 10 runs. DSR with $\tau = 0 . 1$ is included for reference. Best in bold.
<table><tr><td rowspan="2">Method</td><td rowspan="2">State Est.</td><td colspan="4">Transition Model Quality</td><td rowspan="2">Time/epoch (s)</td></tr><tr><td>NMSE (dB) MSE</td><td> $\mathrm { M S E - 5 - s t e p }$ </td><td>LL</td><td>KLD</td></tr><tr><td>EROT (ε = 0.5)</td><td>-12.57±1.64</td><td> $4 . 7 9 2 5 . 6 5$ </td><td>48.41±60.72</td><td> $- 4 . 2 8 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $3 3 . 8 2 { \scriptstyle \pm 2 9 . 7 9 }$ </td><td> $4 2 . 9 4 { \scriptstyle \pm 0 . 7 7 }$ </td></tr><tr><td>EROT (ε = 1.0)</td><td> $- 1 3 . 4 7 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $0 . 5 7 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $9 . 1 9 { \scriptstyle \pm 7 . 5 8 }$ </td><td> $- 3 . 4 9 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $2 0 . 2 0 { \scriptstyle \pm 2 . 3 3 }$ </td><td> $4 0 . 3 5 { \scriptstyle \pm 0 . 3 9 }$ </td></tr><tr><td>DSR (τ = 0.1)</td><td>-18.28±0.46</td><td> $\mathbf { 0 . 5 4 } \pm \mathbf { 0 . 3 3 }$ </td><td>3.70±2.87 −3.06±0.24</td><td></td><td> $\mathbf { 1 3 . 4 1 \pm 1 . 9 4 }$ </td><td> $\mathbf { 1 8 . 3 9 } { \scriptstyle \pm 0 . 1 6 }$ </td></tr></table>

## G Multimodal Posteriors and Inter-modal Mass

At finite τ, DSR forms convex combinations of ancestors, which can place mass between wellseparated modes of a multimodal posterior. To study this phenomenon, we study a static twocomponent Gaussian (i.e., a mixture of two Gaussians). We draw particles from this two-component Gaussian, perform resampling, and study the intermodal mass. The two-component Gaussian is given by

$$
{ \textstyle \frac { 1 } { 2 } } { \mathcal { N } } { \big ( } - { \textstyle \frac { \Delta } { 2 } } \mathbf { e } _ { 0 } , \sigma ^ { 2 } \mathbf { I } _ { d } { \big ) } + { \textstyle \frac { 1 } { 2 } } { \mathcal { N } } { \big ( } + { \textstyle \frac { \Delta } { 2 } } \mathbf { e } _ { 0 } , \sigma ^ { 2 } \mathbf { I } _ { d } { \big ) } ,\tag{18}
$$

where $\Delta$ is the mode separation, $\mathbf { e } _ { 0 } = ( 1 , 0 , \ldots , 0 ) ^ { \top }$ , d is the dimension, and $\sigma ^ { 2 }$ is a scalar for the covariance. The modes are separated along coordinate 0 only, while coordinates $1 , \ldots , d - 1$ carry isotropic noise and no mode information. This single-coordinate separation is deliberate: a Morton code interleaves bits across coordinates, so as d grows the informative bits of coordinate 0 are diluted among uninformative ones, which we interpret as the loss of locality expected of space-filling curves in higher dimensions. We use $N = 1 0 0 , \Delta = 1 0 , \sigma = 0 . 3$ (a separation of $\approx 3 3$ within-mode standard deviations), and measure the resampled mass in the inter-modal region $| x _ { 0 } | < 3$ . Particles are randomly permuted before resampling, and we use the manuscript defaults $( \alpha = 0 . 5 ; \tau = 0 . 1 ;$ $\varepsilon = 0 . 5$ with 100 Sinkhorn iterations). We report the inter-modal mass as mean ± std over $1 0 ^ { 4 }$ simulations.

Any scheme that copies ancestors places zero mass in the region by construction, since no particle lies there; this holds for hard systematic and soft multinomial resampling. Any inter-modal mass under DSR or EROT is therefore attributable to the relaxation. Table 13 reports the results.

Table 13: Inter-modal mass (resampled mass in $\left| x _ { 0 } \right| < 3 )$ for a static two-component Gaussian as a function of dimension d. Lower is better; 0 means no mass placed between the modes. Results show mean ± std over $1 0 ^ { 4 }$ simulations.
<table><tr><td>Method</td><td> $d = 1$ </td><td> $d = 2$ </td><td> $d = 3$ </td><td> $d = 5$ </td><td> $d = 1 0$ </td></tr><tr><td>Hard SR</td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr><tr><td>Soft</td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr><tr><td>EROT</td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 7 4 7 { \scriptstyle \pm 0 . 0 5 2 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr><tr><td>DSR (Morton)</td><td> $0 . 2 8 7 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 2 8 3 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 2 8 2 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 2 8 2 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 2 8 2 { \scriptstyle \pm 0 . 0 0 5 }$ </td></tr><tr><td>DSR (w/o Morton)</td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td></tr></table>

The results support the use of Morton sorting in this setting. Without Morton sorting, every resampled particle lands between the modes; with Morton sorting the inter-modal mass drops to $\mathrm { \Omega ) \approx 0 . 2 8 \Omega }$ , and it does not degrade over this range of d. EROT, by contrast, is sensitive to dimension: for $d \geq 3$ it places all resampled mass between the modes. Morton sorting thus substantially reduces inter-modal mass relative to no sorting for dimensions up to 10. Multimodal posteriors remain challenging for particle filters in general [33, 31], and investigating differentiable resampling under such posteriors is an interesting direction for future work.

## H Additional Linear Gaussian Experiments

The linear Gaussian SSM of Section $5 ,$ as the state transition model is known, admits optimal filtering recursion via the Kalman filter, which lets us approach two questions not accessible in the chaotic or real-world settings: how the marginal likelihood at evaluation time decomposes across methods, and how the DSR gradient compares to an unbiased VSMC gradient.

## H.1 Marginal likelihood at evaluation time

At evaluation time all methods use hard systematic resampling, so log $\hat { p } ( \mathbf { y } _ { 1 : T } )$ is directly comparable across methods, and on the linear Gaussian model the exact log-evidence is available from the Kalman filter. This yields the decomposition

$$
\underbrace { \log p _ { \mathrm { t r u e } } ( \mathbf { y } _ { 1 : T } ) - \log p _ { \hat { \phi } } ( \mathbf { y } _ { 1 : T } ) } _ { \mathrm { m o d e l ~ e r r o r } } + \underbrace { \log p _ { \hat { \phi } } ( \mathbf { y } _ { 1 : T } ) - \mathbb { E } [ \log \hat { p } _ { N } ( \mathbf { y } _ { 1 : T } ) ] } _ { \mathrm { J e n s e n } \cdot \mathrm { g a p } } ,\tag{19}
$$

where the model error measures how close the learned transition model is to the data-generating model, and the Jensen’s gap measures how well an $N -$ particle filter approximates the evidence of that learned model. Using the trained models from Section $5$ on the linear Gaussian SSM with default parameters $( T = 1 0 0 , \mathrm { { S M N R } = 1 0 ~ d B }$ , mean ± std over model seeds on 10 shared sequences), we obtain Table 14.

Table 14: Evaluation-time evidence decomposition on the linear Gaussian SSM $( T = 1 0 0$ , SMNR = 10 dB). The true log-evidence is −232.367. Results show mean ± std over model seeds.
<table><tr><td>Method</td><td> $\log p _ { \hat { \phi } }$ </td><td> $\mathbb { E } [ \log { \hat { p } } _ { N } ]$ </td><td>model error</td><td>Jensen&#x27;s gap</td></tr><tr><td>DSR</td><td> $- 2 3 2 . 9 4 2 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $\mathbf { 2 3 5 . 1 3 8 { \scriptstyle \pm 0 . 3 8 5 } }$ </td><td> $0 . 5 7 5 { \scriptstyle \pm 0 . 0 8 7 }$ </td><td> $\mathbf { 2 . 1 9 6 { \scriptstyle \pm 0 . 3 0 2 } }$ </td></tr><tr><td>EROT</td><td> $- 2 3 3 . 0 6 5 { \scriptstyle \pm 0 . 1 0 4 }$ </td><td> $- 2 3 5 . 8 5 1 \pm 0 . 4 1 4$ </td><td> $0 . 6 9 9 { \scriptstyle \pm 0 . 1 0 4 }$ </td><td> $2 . 7 8 6 { \scriptstyle \pm 0 . 4 6 7 }$ </td></tr><tr><td>Hard</td><td> $\mathbf { - 2 3 2 . 7 9 5 { \scriptstyle \pm 0 . 0 3 8 } }$ </td><td> $- 2 3 6 . 1 4 7 { \scriptstyle \pm 0 . 9 2 4 }$ </td><td> $\mathbf { 0 . 4 2 8 { \scriptstyle \pm 0 . 0 3 8 } }$ </td><td> $3 . 3 5 2 { \scriptstyle \pm 0 . 9 0 1 }$ </td></tr><tr><td>Soft</td><td> $- 2 3 2 . 9 9 6 { \scriptstyle \pm 0 . 1 3 8 }$ </td><td> $- 2 3 7 . 2 1 5 { \scriptstyle \pm 0 . 7 4 9 }$ </td><td> $0 . 6 3 0 { \scriptstyle \pm 0 . 1 3 8 }$ </td><td> $4 . 2 1 8 { \scriptstyle \pm 0 . 6 6 9 }$ </td></tr></table>

All four methods are within 0.7 of the true evidence. Hard SR attains the smallest model error (0.428 versus 0.575 for DSR, a gap outside the spread over seeds), but the methods separate on the Jensen’s gap, where DSR attains 2.196 against 3.352 for hard SR and with roughly a third of the spread (±.302 versus ±.901). The two columns measure different things: since the optimised quantity is

E[log ${ \hat { p } } _ { N } ]$ rather than log p, these results indicate that DSR learns parameters whose evidence the finite-particle filter approximates more accurately.

## H.2 Comparison against unbiased VSMC gradients

We compare the DSR gradient against an unbiased VSMC gradient. This is to study the gradient bias induced by DSR. We follow the linear Gaussian experiment of [27] adapted to our setting. We use the linear Gaussian SSM introduced in Sec. 5.2, using $\mathbf { S M N R } = 1 0 $ dB with $N = 4$ $T = 4 ,$ , and additionally a two-dimensional version (a single rotation block) at the same $N$ and $T .$ For each we generate one observation sequence $\mathbf { y } _ { 1 : T }$ that every run below targets. Gradients are taken with respect to all $d ^ { 2 }$ entries of A and the d diagonal entries of $\mathbf { Q } = \mathop { \mathrm { d i a g } } ( \operatorname { s o f t p l u s } ( \pmb { \ell } ) )$ evaluated at the data-generating parameters. The unbiased gradient $\nabla \mathcal { L }$ is estimated by averaging over 1000 bootstrap noise draws and exhaustively summing over ancestor vectors (the vectors are the systematic combs induced by a single uniform draw). Let $\hat { \bf g }$ denote an estimate obtained by averaging $n = 1 6 0 0 ( = 1 0 0 / 4 \times \mathrm { \bar { 6 } } 4 )$ independent SMC runs, matching the number of observations consumed in our training configuration (Appendix C), each run random through its bootstrap noise and resampling offsets. We report

$$
\mathrm { B i a s } = \left\| \mathbb { E } [ \hat { \mathbf { g } } ] - \nabla \mathcal { L } \right\| , \quad \mathrm { V a r i a n c e } = \mathbb { E } \left\| \hat { \mathbf { g } } - \mathbb { E } [ \hat { \mathbf { g } } ] \right\| ^ { 2 } , \quad \mathrm { M S E } = \mathbb { E } \left\| \hat { \mathbf { g } } - \nabla \mathcal { L } \right\| ^ { 2 } = \mathrm { B i a s } ^ { 2 } + \mathrm { V a r i a n c e } .
$$

We estimate $\mathbb { E } [ \hat { \bf g } ]$ from the same 1000 bootstrap draws with 500 offset sequences $( 5 \times 1 0 ^ { 5 }$ runs), and the variance separately from 200 independent realisations of $\hat { \bf g }$ at $n = 1 6 0 0$ . Results are in Table 15.

(20)

Table 15: Gradient bias, variance, and MSE against the unbiased VSMC gradient on the linear Gaussian SSM. Hard SR drops the score-function term. Results show mean ± standard errors.
<table><tr><td>Method</td><td>Bias</td><td>Variance</td><td>MSE</td></tr><tr><td colspan="3"> $( \mathrm { a } ) d = 2 , N = 4 , T = 4 ; \ \| \nabla \mathcal { L } \| = 9 . 6 4$ </td><td></td></tr><tr><td>Hard SR</td><td> $2 . 3 2 4 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 3 0 8 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $5 . 7 0 7 { \scriptstyle \pm 0 . 0 6 7 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 1 . 0 \right)$ </td><td> $1 . 2 5 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 4 1 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $1 . 8 1 8 { \scriptstyle \pm 0 . 0 1 4 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 0 . 5 \right)$ </td><td> $1 . 2 5 7 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $0 . 2 4 5 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $1 . 8 2 5 { \scriptstyle \pm 0 . 0 1 4 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 0 . 1 \right)$ </td><td> $0 . 7 6 4 { \scriptstyle \pm 0 . 0 2 0 }$ </td><td> $0 . 3 8 9 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 9 7 3 { \scriptstyle \pm 0 . 0 4 3 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 0 . 0 5 \right)$ </td><td> $0 . 3 7 0 { \scriptstyle \pm 0 . 0 5 0 }$ </td><td> $0 . 7 2 3 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td> $0 . 8 6 0 { \scriptstyle \pm 0 . 0 5 2 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 0 . 0 1 \right)$ </td><td> $0 . 3 4 3 { \scriptstyle \pm 0 . 2 2 7 }$ </td><td> $1 3 . 4 0 9 { \scriptstyle \pm 2 . 5 8 3 }$ </td><td> $1 3 . 5 2 6 { \scriptstyle \pm 2 . 5 8 7 }$ </td></tr><tr><td colspan="4">(b)  $d = 4 , N = 4 , T = 4 ; \ \| \nabla \mathcal { L } \| = 2 0 . 5 0$ </td></tr><tr><td>Hard SR</td><td> $3 . 6 2 7 { \scriptstyle \pm 0 . 0 2 8 }$ </td><td> $2 . 0 9 2 { \scriptstyle \pm 0 . 0 5 8 }$ </td><td> $1 5 . 2 4 8 { \scriptstyle \pm 0 . 2 1 2 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 1 . 0 \right)$ </td><td> $2 . 8 6 9 { \scriptstyle \pm 0 . 0 0 1 }$ </td><td> $2 . 0 7 3 { \scriptstyle \pm 0 . 0 5 7 }$ </td><td> $1 0 . 3 0 4 { \scriptstyle \pm 0 . 0 5 7 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 0 . 5 \right)$ </td><td> $2 . 9 1 4 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $2 . 1 2 7 { \scriptstyle \pm 0 . 0 5 8 }$ </td><td> $1 0 . 6 1 5 { \scriptstyle \pm 0 . 0 6 4 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 0 . 1 \right)$ </td><td> $2 . 1 3 1 { \scriptstyle \pm 0 . 0 8 8 }$ </td><td> $4 . 8 9 5 { \scriptstyle \pm 0 . 1 9 2 }$ </td><td> $9 . 4 3 7 { \scriptstyle \pm 0 . 4 2 0 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 0 . 0 5 \right)$ </td><td> $1 . 7 2 0 { \scriptstyle \pm 0 . 1 9 0 }$ </td><td> $1 6 . 3 8 2 { \scriptstyle \pm 1 . 8 6 4 }$ </td><td> $1 9 . 3 4 0 { \scriptstyle \pm 1 . 9 7 5 }$ </td></tr><tr><td> $\mathrm { D S R } \left( \tau = 0 . 0 1 \right)$ </td><td> $1 . 4 1 2 { \scriptstyle \pm 1 . 2 1 6 }$ </td><td> $3 8 4 . 8 0 5 { \scriptstyle \pm 1 0 2 . 5 6 4 }$ </td><td> $3 8 6 . 7 9 9 { \scriptstyle \pm 1 0 2 . 6 2 1 }$ </td></tr></table>

DSR has a smaller gradient bias than hard SR (which drops $g _ { \mathrm { s c o r e } } )$ at every temperature and in both settings, and the bias decreases as τ decreases: at $\tau = 1 . 0$ the bias drops from 2.324 to 1.256 $( d = 2 )$ and from 3.627 to 2.869 (d = 4) with essentially unchanged variance. Decreasing τ further keeps reducing the bias but increases the variance sharply. At the operating temperature $\tau = 0 . 1$ , DSR is competitive on MSE with hard SR and the other τ values, attaining the lowest MSE at $d = 4$ and the second lowest at $d = 2 .$ . This experiment makes the bias–variance tradeoff explicit and motivates the annealing strategy studied in Appendix F.2.