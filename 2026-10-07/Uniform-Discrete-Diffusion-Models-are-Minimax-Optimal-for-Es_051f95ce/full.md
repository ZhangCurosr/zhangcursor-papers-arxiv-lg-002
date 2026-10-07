# Uniform Discrete Diffusion Models are Minimax Optimal for Estimating Distributions with Small Effective Support Size

Dongsun Yoon\*1 and Saptarshi Chakraborty†1

1Department of Statistics, University of Michigan

## Abstract

Discrete diffusion models have emerged as a practically successful framework for generative modeling on discrete product spaces, yet their statistical generalization properties remain poorly understood. Discrete real-world data such as text or biological sequences often concentrate on a small fraction of the astronomically large ambient space because of semantic or physical constraints, but existing bounds fail to capture this distributional structure and instead scale with the size of the ambient space, giving rise to almost vacuous error bounds. We address this gap for uniform discrete diffusion, one of the two dominant discrete diffusion paradigms alongside masking diffusion, by deriving statistical guarantees governed by the effective support size ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ , a sample-size-dependent measure of distributional complexity. Given n independent and identically distributed (i.i.d.) samples from an unknown data distribution $P _ { 0 }$ on $[ K ] ^ { d } ,$ we show that, with appropriate choices of network size and hyperparameters, the expected total variation (TV) loss scales as $\mathcal { O } \left( \sqrt { \mathfrak { s } _ { n } ( P _ { 0 } ) / n } \right)$ , while the expected Kullback-Leibler (KL) divergence is bounded by $\begin{array} { r } { \mathcal { O } \left( \frac { 1 } { n } \mathfrak { s } _ { n } ( P _ { 0 } ) \log ( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) ) \log n \right) } \end{array}$ . Furthermore, we show that the TV rate is minimax optimal and that the KL rate is minimax optimal up to a factor of log n. Together, these upper and lower bounds show that uniform discrete diffusion successfully avoids the curse of dimensionality for distributions with small effective support size: the TV error rate depends on the ambient state-space size only through ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ , while the corresponding KL rate incurs only an additional logarithmic dependence on the ambient state-space size.

## 1 Introduction

Diffusion models have become a central paradigm for generative modeling (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song et al., 2021). Extensions to discrete state spaces have carried this paradigm to categorical domains such as text, graphs, molecules, and combinatorial optimization (He et al., 2023; Hoogeboom et al., 2021; Vignac et al., 2023; Sun and Yang, 2023). Austin et al. (2021) introduced structured categorical corruption kernels in discrete time, and Campbell et al. (2022) cast both the forward and reverse processes as continuous-time Markov chains (CTMCs). Score-based approaches that estimate probability ratios through score entropy have since achieved strong empirical performance in language modeling (Lou et al., 2024). Two forward processes dominate: uniform diffusion, in which each coordinate evolves toward uniformity, and masking diffusion, in which each coordinate is absorbed into a special mask state. They induce distinct reverse-time dynamics and hence different statistical and algorithmic properties (Dmitriev et al.. 2026; Zhang et al., 2026). In this work, we focus on coordinatewise uniform diffusion on the product space $[ K ] ^ { d }$

The empirical success of discrete diffusion has spurred substantial theoretical work. Most of this work concerns the convergence of reverse-time samplers (Campbell et al., 2022; Chen and Ying 2025; Ren et al., 2025; Zhang et al., 2025; Pham et al., 2025; Liang et al., 2025; Dmitriev et al., 2026), particularly fixed-grid τ-leaping and its variants. For uniform discrete diffusion, representative results (Liang et al., 2025; Dmitriev et al., 2026) take the form

$$
D _ { \mathrm { K L } } ( P _ { \delta } \| Q _ { \delta } ) \stackrel { < } { \sim } \varepsilon _ { \mathrm { s c o r e } } + e ^ { - T } d \log K + \mathcal { R } _ { \mathrm { d i s c } } .
$$

Here, $P _ { \delta }$ is the forward marginal at the early-stopping time $\delta ,$ and $Q _ { \delta }$ is the output distribution of the sampler initialized from the uniform distribution at time T. The first term $\varepsilon _ { \mathrm { s c o r e } }$ bounds the interval-weighted score-entropy error on the sampling grid, and the remaining terms account for initialization and time discretization. These guarantees take an $\varepsilon _ { \mathrm { s c o r e ^ { - } a c c u r a t e } }$ score estimate as given and do not establish that such an estimate can be learned. It therefore remains unclear how they translate into finite-sample bounds on the distribution-estimation risk when the score is learned from n training samples.

Without exploiting structure in the data distribution, finite-sample guarantees face a fundamental obstacle: the size of the ambient space. For arbitrary distributions on $[ K ] ^ { d }$ , the minimax expected risk is of order $\Theta \left( \operatorname* { m i n } \{ 1 , \sqrt { K ^ { d } / n } \} \right)$ under TV loss (Han et al., 2015; Kamath et al., 2015) and Θ $\left( \log ( 1 + K ^ { d } / n ) \right)$ under KL loss (Ye et al., 2025; Mourtada, 2026). Consistent estimation over the unrestricted class therefore requires $K ^ { d } / n  0$ , a condition far from realistic. The Score Entropy Discrete Diffusion (SEDD) language experiments (Lou et al., 2024), for instance, use a vocabulary size $K = 5 0 , 2 5 7$ and sequence length $d = 1 0 2 4$ . The resulting ambient space of size $5 0 { , } 2 5 7 ^ { 1 0 2 4 }$ is astronomically larger than any feasible training set.

The finite-sample statistical theory of discrete diffusion that has recently begun to emerge does not escape this barrier (Wakasugi and Suzuki, 2025; Cho and Wu, 2026; Srikanth et al., 2026; Zhang et al., 2026). The KL bounds of Wakasugi and Suzuki (2025) and Cho and Wu (2026) are of order ${ \widetilde { \mathcal { O } } } ( K ^ { d } / n )$ , and the TV bounds of Zhang et al. (2026) grow exponentially in $d ,$ so all become vacuous when $n \ll K ^ { d }$ . Refinements that avoid this dependence either rely on assumptions on $P _ { 0 }$ that are unverifiable in practice, such as fast spectral decay (Wakasugi and Suzuki, 2025), or depend on quantities that can themselves be as large as $K ^ { d }$ . For example, the bound of Srikanth et al. (2026) scales polynomially with $\mathrm { m a x } _ { d _ { \mathrm { H a m } } ( { \pmb x } , { \pmb y } ) = 1 } P _ { 0 } ( { \pmb y } ) / P _ { 0 } ( { \pmb x } )$ , which is enormous when a single-token substitution sharply reduces sequence probability.

Such pessimistic guarantees are at odds with the structure of real data. Text and biological sequences concentrate on a vanishing fraction of $[ K ] ^ { d }$ because of semantic or physical constraints, so the ambient cardinality greatly overstates their statistical complexity. A natural first refinement replaces $K ^ { d }$ by the support size $| \operatorname { s u p p } ( P _ { 0 } ) |$ . However, the support can itself be comparable to $K ^ { d }$ even when most of the mass lies on far fewer states. Moreover, at sample size n, states with probability well below $1 / n$ are unlikely to be observed and should not each count as a full unit of complexity.

We therefore measure complexity by the effective support size

$$
{ \mathfrak { s } } _ { n } ( P _ { 0 } ) : = \sum _ { \pmb { x } \in [ K ] ^ { d } } \operatorname* { m i n } \{ n P _ { 0 } ( \pmb { x } ) , 1 \} .
$$

Each state with probability at least $1 / n$ counts as one effective state, while rarer states contribute according to their expected counts in n samples. Up to universal constants, ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ is the expected number of distinct states in a sample of size n (Lemma A.2). It satisfies ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \leq$ min $\{ n , | \operatorname { s u p p } ( P _ { 0 } ) | \}$ and can be far smaller than both. Under Zipfian decay $p _ { ( j ) } \asymp j ^ { - \alpha }$ , for instance, ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \asymp \operatorname* { m i n } \{ | \operatorname { s u p p } ( P _ { 0 } ) | , n ^ { 1 / \alpha } \}$ (Section 2.1). The notion of effective support size has been used in the classical literature on occupancy counts, unseen species, and missing mass (Good, 1953; Karlin, 1967; Ben-Hamou et al., 2017). Related occupancy-based notions have also yielded distributiondependent guarantees for discrete distribution estimation (Falahatgar et al., 2017; Mourtada, 2026).

This leads to the central question of the paper:

Can discrete diffusion models attain distribution-estimation rates that adapt to ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$

and remain informative even when $n \ll K ^ { d } ?$

Summary of contributions. We answer this question affirmatively for uniform discrete diffusion.

• Adaptive guarantees for uniform discrete diffusion. We establish end-to-end finite-sample guarantees for uniform discrete diffusion with a learned score. They account for score estimation with ReLU networks as well as initialization and time-discretization errors, and they apply to any fixed-grid sampler satisfying a standard score-entropy guarantee, including τ-leaping. The expected TV loss is $\mathcal { O } ( \sqrt { \mathfrak { s } _ { n } ( P _ { 0 } ) / n } )$ , with number of sampling steps and network depth, width, and sparsity polynomial in $K , d ,$ and $n / { \mathfrak { s } } _ { n } ( P _ { 0 } )$ . With early stopping at $\delta = 1 / n$ , the expected KL divergence is $\begin{array} { r } { \mathcal { O } \bigl ( \frac { 1 } { n } \mathfrak { s } _ { n } ( P _ { 0 } ) \log ( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) ) \log n \bigr ) } \end{array}$ . Both bounds depend on $K ^ { d }$ at most logarithmically and remain informative when $n \ll K ^ { d }$ . The KL guarantee carries particular weight: the empirical distribution has infinite KL risk whenever some states go unobserved, so attaining it requires generalizing beyond the training samples.

• Matching minimax lower bounds. For $2 \leq s _ { n } \leq \operatorname* { m i n } \{ n , K ^ { d } \}$ , let $\mathcal { P } _ { s _ { n } } : = \{ P _ { 0 } \in \Delta _ { K ^ { d } } : \mathfrak { s } _ { n } ( P _ { 0 } ) \leq$ $s _ { n } \}$ . We establish the non-asymptotic minimax rates

$$
\operatorname* { i n f } _ { \tilde { P } } \operatorname* { s u p } _ { P _ { 0 } \in \mathcal { P } _ { s _ { n } } } \mathbb { E } _ { P _ { 0 } } d _ { \mathrm { T V } } ( P _ { 0 } , \tilde { P } ) \asymp \sqrt { \frac { s _ { n } } { n } } , \qquad \operatorname* { i n f } _ { \tilde { P } } \operatorname* { s u p } _ { P _ { 0 } \in \mathcal { P } _ { s _ { n } } } \mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } ( P _ { 0 } \| \tilde { P } )  & { \asymp \frac { s _ { n } \log \left( e K ^ { d } / s _ { n } \right) } { n } . }
$$

Hence, uniform discrete diffusion is minimax optimal under TV loss and optimal up to a log n factor under KL loss. When $s _ { n } = \operatorname* { m i n } \{ n , K ^ { d } \} , \mathcal { P } _ { s _ { n } } = \Delta _ { K ^ { d } }$ holds, and our rates reduce to the classical minimax rates for arbitrary distributions on $K ^ { d }$ states. Thus, our result extends the classical unrestricted theory, while yielding potentially much sharper rates for distributions with

small effective support size.

Allowing K and d to vary with $n ,$ these rates also characterize when consistent estimation is possible. Under TV loss, this holds if and only if $s _ { n } / n \to 0$ , and under KL loss, if and only if $( s _ { n } / n ) \log ( e K ^ { d } / s _ { n } ) \to$ 0. The gap between the two conditions yields three regimes (Figure 1): consistent estimation is impossible, possible under TV loss only, or possible under both losses.

![](images/d3e1907dc5498ea61aa1585ed8d1aabbc4fde085790d8c6957c0bce7fea6d764.jpg)  
Figure 1: Asymptotic statistical regimes for estimating discrete distributions on $[ K ] ^ { d }$ with n i.i.d. observations.

## 2 Background and Problem Setting

Notations. We use plain uppercase letters such as P and Q for probability distributions and bold uppercase letters such as P and Q for transition and generator matrices. For a positive integer K, write $[ K ] : = \{ 1 , \dots , K \}$ , and let $\Delta _ { K ^ { d } }$ denote the probability simplex on $[ K ] ^ { d }$ . For a probability distribution P on $[ K ] ^ { d }$ , let $\operatorname { s u p p } ( P ) : = \{ { \pmb x } : P ( { \pmb x } ) > 0 \}$ denote its support, and write $\mathbb { E } _ { P }$ for expectation under P. For probability distributions P and Q on $[ K ] ^ { d }$ , the Total Variation (TV) distance and Kullback-Leibler divergence are defined as, $\begin{array} { r } { d _ { \mathrm { T V } } ( P , Q ) : = \frac 1 2 \sum _ { \pmb { x } \in [ K ] ^ { d } } | P ( \pmb { x } ) - \| } \end{array}$ $\begin{array} { r } { Q ( \pmb { x } ) \vert , D _ { \mathrm { K L } } ( P \Vert Q ) : = \sum _ { \pmb { x } \in [ K ] ^ { d } } P ( \pmb { x } ) \log \left( P ( \pmb { x } ) / Q ( \pmb { x } ) \right) } \end{array}$ , where, we use the conventions $0 \log ( 0 / q ) : = 0$ and $p \log ( p / 0 ) : = \infty$ for $p > 0$ . For nonnegative quantities $u _ { n }$ and $v _ { n }$ , which are functions of n, we write, $u \lesssim v$ if $u _ { n } \leq C v _ { n }$ for a universal constant $C \ > \ 0$ , independent of n. We write $u _ { n } \ \asymp \ v _ { n }$ if both $u _ { n } \ \lesssim \ v _ { n }$ and $v _ { n } \lesssim u _ { n }$ . For $\pmb { x } = ( x ^ { 1 } , . . . , x ^ { d } ) , \pmb { y } = ( y ^ { 1 } , . . . , y ^ { d } ) \in [ K ] ^ { d }$ the Hamming distance between x and y is defined as, $d _ { \mathrm { H a m } } ( \pmb { x } , \pmb { y } ) : = \sum _ { i = 1 } ^ { d } \pmb { 1 } \{ x ^ { i } \neq y ^ { i } \}$ , where 1(·) denotes the indicator function. For $u , v > 0$ , define the Bregman divergence generated by $f ( u ) : = u \log u$ as $D _ { \mathrm { B r } } ( u , v ) : = f ( u ) - f ( v ) - f ^ { \prime } ( v ) ( u - v ) = u \log ( u / v ) - u + v$ . For $a \in [ K ]$ , let $e _ { a } \in \{ 0 , 1 \} ^ { K }$ be the a-th standard basis vector, and define the coordinatewise one-hot encoding $\boldsymbol { e } ( \boldsymbol { x } ) : = ( e _ { x ^ { 1 } } ^ { \top } , \ldots , e _ { x ^ { d } } ^ { \top } ) ^ { \top } \in \{ 0 , 1 \} ^ { d K }$ $\delta _ { x }$ denotes the Dirac distribution at x. For $\ell \leq r ,$ define $\mathrm { c l i p } _ { [ \ell , r ] } ( u ) : = \operatorname* { m i n } \{ \operatorname* { m a x } \{ u , \ell \} , r \}$ . All logarithms are with respect to (w.r.t.) the natural base.

## 2.1 Statistical Setup and Effective Support Size

We consider distribution estimation on the product space $[ K ] ^ { d }$ for $K , d \ge 2$ , where K can be interpreted as the vocabulary size and d is the sequence length. Let $P _ { 0 } \in \Delta _ { K ^ { d } }$ be an unknown data distribution, and suppose that we observe training n i.i.d. samples $X _ { 0 } ^ { ( 1 ) } , \ldots , X _ { 0 } ^ { ( n ) } \overset { \mathrm { i . i . d . } } { \sim } P _ { 0 }$ for $n \geq 2$ A distribution estimator $\widetilde { P } \in \Delta _ { K ^ { d } }$ is any (possibly randomized) data-dependent estimator of $P _ { 0 }$ . We study its expected total variation and Kullback-Leibler risks, $\mathbb { E } _ { P _ { 0 } } d _ { \mathrm { T V } } ( P _ { 0 } , \widetilde { P } )$ and $\mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } ( P _ { 0 } \| \widetilde { P } )$

Effective support size. As a measure of the complexity of $P _ { 0 }$ relative to a sample of size n, we use its effective support size (Mourtada, 2026; Falahatgar et al., 2017),

$$
\mathfrak { s } _ { n } ( P _ { 0 } ) : = \sum _ { x \in [ K ] ^ { d } } \operatorname* { m i n } \{ n P _ { 0 } ( x ) , 1 \} = \Big | \Big \{ x \in [ K ] ^ { d } : P _ { 0 } ( x ) \geq 1 / n \Big \} \Big | + n \sum _ { x : P _ { 0 } ( x ) < 1 / n } P _ { 0 } ( x ) ,
$$

where states above the sampling resolution $1 / n$ each contribute a unit, while states below it contribute only their rescaled aggregate mass. Thus ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ is nondecreasing in $n ,$ satisfies ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \leq$ min $\{ n , | \operatorname { s u p p } ( P _ { 0 } ) | \}$ , and recovers the exact support size as $n  \infty ,$ i.e., $\begin{array} { r } { \operatorname* { l i m } _ { n \to \infty } \mathfrak { s } _ { n } ( P _ { 0 } ) = \mathfrak { s } _ { \infty } ( P _ { 0 } ) : = } \end{array}$ $| \{ \pmb { x } \in [ K ] ^ { d } : P _ { 0 } ( \pmb { x } ) > 0 \} |$ . This refines the crude sparsity measure $| \operatorname { s u p p } ( P _ { 0 } ) |$ into one calibrated to the sample size, and is sensitive not just to $| \operatorname { s u p p } ( P _ { 0 } ) |$ but to the decay of the point masses. Writing $p _ { ( 1 ) } \geq \cdot \cdot \cdot \geq p _ { ( | \operatorname { s u p p } ( P _ { 0 } ) | ) } > 0$ for the sorted nonzero masses of $P _ { 0 }$ , Lemma A.1 gives, for $n \geq 2$

$$
\mathfrak { s } _ { n } ( P _ { 0 } ) \asymp \left\{ \begin{array} { l l } { \operatorname* { m i n } \{ | \operatorname { s u p p } ( P _ { 0 } ) | , n \} , } & { \mathrm { ~ i f ~ } p _ { ( j ) } \asymp | \operatorname { s u p p } ( P _ { 0 } ) | ^ { - 1 } , } \\ { \operatorname* { m i n } \{ | \operatorname { s u p p } ( P _ { 0 } ) | , n ^ { 1 / \alpha } \} , } & { \mathrm { ~ i f ~ } p _ { ( j ) } \asymp j ^ { - \alpha } , } \\ { \operatorname* { m i n } \{ | \operatorname { s u p p } ( P _ { 0 } ) | , \log n \} , } & { \mathrm { ~ i f ~ } p _ { ( j ) } \asymp e ^ { - c j } , } \end{array} \right.\tag{1}
$$

uniformly over $1 \leq j \leq | \operatorname { s u p p } ( P _ { 0 } ) |$ , with $\alpha > 1 , c > 0$ fixed and implicit constants independent of n and $| \operatorname { s u p p } ( P _ { 0 } ) |$ . These regimes span the full range of behavior: linear growth under near-uniform mass, sublinear polynomial growth n1/α under Zipfian decay, and merely logarithmic growth under $n ^ { 1 / \alpha }$ geometric decay. It is this decay-dependent behavior, rather than the ambient size $K ^ { d }$ or the raw support size $| \operatorname { s u p p } ( P _ { 0 } ) |$ , that governs the finite-sample rates established below.

For $1 \leq s _ { n } \leq \operatorname* { m i n } \{ n , K ^ { d } \}$ , we define the effective support class $\mathcal { P } _ { s _ { n } } = \mathcal { P } _ { s _ { n } } ( K ^ { d } , n ) : = \{ P _ { 0 } \in \Delta _ { K ^ { d } }$

${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \leq s _ { n } \}$ . The corresponding minimax risks in the TV and KL are

$$
\operatorname* { i n f } _ { \widetilde { P } } \operatorname* { s u p } _ { P _ { 0 } \in \mathcal { P } _ { s _ { n } } } \mathbb { E } _ { P _ { 0 } } d _ { \mathrm { T V } } ( P _ { 0 } , \widetilde { P } ) \qquad \mathrm { a n d } \qquad \operatorname* { i n f } _ { \widetilde { P } } \operatorname* { s u p } _ { P _ { 0 } \in \mathcal { P } _ { s _ { n } } } \mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } ( P _ { 0 } \| \widetilde { P } ) .
$$

The goal of this paper is to study the behavior of the above minimax risks and to understand under what model and hyperparameter choices discrete diffusion models can achieve this optimal rate.

## 2.2 Uniform Discrete Diffusion

The uniform discrete diffusion consists of a forward noising process, its score-based time reversal denoising, a fixed-grid score estimator, and a reverse sampler.

Forward process. The forward process is a continuous-time Markov chain (CTMC) on $[ K ] ^ { d }$ with independently evolving coordinates. The time-homogeneous generator of each coordinate is $Q ^ { \mathrm { c o o r d } } ( a , b ) : = 1 / K - { \bf 1 } \{ a = b \}$ for every $a , b \in [ K ]$ , which means that each coordinate jumps to another state with rate $1 / K$ . Following Campbell et al. (2022), the induced generator $Q$ on $[ K ] ^ { d }$ is

$$
Q ( x , y ) = K ^ { - 1 } { \bf 1 } \left\{ d _ { \mathrm { H a m } } ( { \pmb x } , { \pmb y } ) = 1 \right\} - d K ^ { - 1 } ( K - 1 ) { \bf 1 } \left\{ { \pmb x } = y \right\} .
$$

The one-coordinate transition kernel is given by $P _ { t } ^ { \mathrm { c o o r d } } ( a , b ) = e ^ { - t } \mathbf { 1 } \{ a = b \} + ( 1 - e ^ { - t } ) / K$ and coordinatewise independence gives $\begin{array} { r } { P _ { t } ( x , y ) = \prod _ { i = 1 } ^ { d } P _ { t } ^ { \mathrm { c o o r d } } ( x ^ { i } , y ^ { i } ) } \end{array}$ and $P _ { t } = e ^ { t Q }$ . Writing $P _ { t } : = P _ { 0 } P _ { t }$ 2 symmetry of $Q$ implies that the uniform distribution $\pi ( { \pmb x } ) : = K ^ { - d }$ is stationary. Since the chain is irreducible on the finite state space $[ K ] ^ { d }$ , π is the unique stationary distribution and $P _ { t } \to \pi$ as $t \to \infty$ (Norris, 1997, Theorem 3.6.2).

Reverse process and discrete score. Fix a terminal time $T > 0$ and let $u \in [ 0 , T ]$ denote reverse time. Initialized from $P _ { T }$ , the time reversal of the forward process is a time-inhomogeneous CTMC whose off-diagonal rates, for $u \in [ 0 , T )$ , are $\begin{array} { r } { \overleftarrow { Q } _ { u } ^ { P } ( { \boldsymbol x } , { \boldsymbol y } ) = Q ( { \boldsymbol y } , { \boldsymbol x } ) \frac { P _ { T - u } ( { \boldsymbol y } ) } { P _ { T - u } ( { \boldsymbol x } ) } , \ { \boldsymbol x } \ne { \boldsymbol y } } \end{array}$ . The diagonal entries are chosen so that each row sums to zero. With the forward generator fixed, the probability ratio $P _ { T - u } ( \pmb { y } ) / P _ { T - u } ( \pmb { x } )$ determines the reverse generator. This motivates defining the true discrete score, for $t > 0$ , as $\begin{array} { r } { \sigma _ { t } ^ { \star } ( \pmb { y } , \pmb { x } ) : = \frac { P _ { t } ( \pmb { y } ) } { P _ { t } ( \pmb { x } ) } } \end{array}$ . This ratio plays a role analogous to the continuous score ∇log $p _ { t }$ which governs the reverse drift in continuous diffusion models. For our coordinatewise uniform forward process, the off-diagonal reverse rates become

$$
\overleftarrow { Q } _ { u } ^ { P } ( \boldsymbol { x } , \boldsymbol { y } ) = \left\{ \begin{array} { l l } { K ^ { - 1 } \sigma _ { T - u } ^ { \star } ( \boldsymbol { y } , \boldsymbol { x } ) , } & { d _ { \mathrm { H a m } } ( \boldsymbol { x } , \boldsymbol { y } ) = 1 , } \\ { \quad } & { x \neq y . } \\ { 0 , } & { d _ { \mathrm { H a m } } ( \boldsymbol { x } , \boldsymbol { y } ) \geq 2 , } \end{array} \right.
$$

For sufficiently large $T ,$ generation starts from the tractable stationary distribution $\pi$ in place of $P _ { T }$ and simulates the reverse process using a learned approximation of $\sigma _ { t } ^ { \star }$

Fixed-grid score estimation. Although the reverse process evolves continuously in time, a practical sampler evaluates the score at only finitely many grid-points. In this paper, we choose a deterministic forward-time grid $\mathcal { T } : = \{ \delta = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { N } = T \}$ , where $\delta \geq 0$ is the early-stopping time and scores are estimated only at $t _ { 1 } , \ldots , t _ { N }$ . For a positive candidate score $\sigma _ { t }$ , the score-entropy loss at time $t > 0$ is given by,

$$
\mathcal { L } _ { t } ^ { \mathrm { S E } } ( \sigma ; P _ { 0 } ) : = \mathbb { E } _ { X _ { t } \sim P _ { t } } \left[ \sum _ { y : Q ( y , X _ { t } ) > 0 } Q ( y , X _ { t } ) D _ { \mathrm { B r } } \left( \sigma _ { t } ^ { \star } ( y , X _ { t } ) , \sigma _ { t } ( y , X _ { t } ) \right) \right] .
$$

This loss is nonnegative and minimized by the true score $\sigma _ { t } ^ { \star }$ on all pairs $( { \pmb y } , { \pmb x } )$ satisfying $d _ { \mathrm { H a m } } ( \pmb { y } , \pmb { x } ) =$ 1 (Lou et al., 2024). We aggregate the score error over the grid using the interval lengths as weights: $\begin{array} { r } { \mathcal { L } _ { \mathcal { T } } ^ { \mathrm { S E } } ( \sigma ; P _ { 0 } ) : = \sum _ { k = 1 } ^ { N } ( t _ { k } - t _ { k - 1 } ) \mathcal { L } _ { t _ { k } } ^ { \mathrm { S E } } ( \sigma ; P _ { 0 } ) } \end{array}$ . Since $\mathcal { L } _ { t } ^ { \mathrm { S E } }$ depends on the intractable marginal $P _ { t } .$ we instead use its denoising form. Denoting the true conditional score by $\sigma _ { t | 0 } ^ { \star } ( \pmb { y } , \pmb { x } \mid \pmb { x } _ { 0 } ) : =$ $P _ { t } ( x _ { 0 } , y ) / P _ { t } ( x _ { 0 } , x )$ , the denoising score-entropy loss is constructed as:

$$
\begin{array} { r l } { \mathcal { L } _ { t } ^ { \mathrm { D S E } } ( \sigma ; P _ { 0 } ) : = } & { \quad \mathbb { E } \left[ \displaystyle \sum _ { y : Q ( y , X _ { t } ) > 0 } Q ( y , X _ { t } ) \left( \sigma _ { t } ( y , X _ { t } ) - \sigma _ { t | 0 } ^ { \star } ( y , X _ { t } \mid X _ { 0 } ) \log \sigma _ { t } ( y , X _ { t } ) \right) \right] , } \end{array}
$$

where in the above expectation, $X _ { 0 } \sim P _ { 0 }$ , and $X _ { t } | X _ { 0 } \sim P _ { t } ( X _ { 0 } , \cdot )$ . As shown in Lemma A.14, the score-entropy and denoising score-entropy losses differ only by a term independent of $\sigma$ and therefore have the same minimizer. Accordingly, we define the denoising score-matching loss as $\begin{array} { r } { \mathcal { L } _ { \mathcal { T } } ^ { \mathrm { D S E } } ( \sigma ; P _ { 0 } ) : = \sum _ { k = 1 } ^ { N } ( t _ { k } - t _ { k - 1 } ) \mathcal { L } _ { t _ { k } } ^ { \mathrm { D S E } } ( \sigma ; P _ { 0 } ) } \end{array}$ . During training, since $P _ { 0 }$ is unknown, it is substituted by the empirical distribution $\begin{array} { r } { \widehat { P } _ { 0 } : = n ^ { - 1 } \sum _ { i = 1 } ^ { n } \delta _ { X _ { 0 } ^ { ( i ) } } } \end{array}$ , yielding the empirical training objective

$$
\begin{array} { r } { \widehat { \mathcal { L } } _ { \mathcal { T } } ^ { \mathrm { D S E } } ( \sigma ) : = \mathcal { L } _ { \mathcal { T } } ^ { \mathrm { D S E } } ( \sigma ; \widehat { P } _ { 0 } ) . } \end{array}\tag{2}
$$

Neural-network score class. For positive integers L, W, S and $B > 0$ , let $\mathcal { G } ( L , W , S , B )$ denote the class of scalar-output ReLU networks on $\mathbb { R } ^ { 1 + 2 d K }$ with depth at most L, width at most $W$ at most S nonzero parameters, and all weights and biases bounded in absolute value by B. For $G \in { \mathcal { G } } ( L , W , S , B )$ and $M \geq 1$ , define

$$
\sigma _ { t _ { k } } ^ { G } ( \pmb { y } , \pmb { x } ) : = \mathrm { c l i p } _ { [ M ^ { - 1 } , M ] } \left( G \left( k / N , e ( \pmb { y } ) , e ( \pmb { x } ) \right) \right) , \qquad k \in [ N ] , \quad \pmb { x } , \pmb { y } \in [ K ] ^ { d } .
$$

Here, $k / N$ encodes only the grid index and does not represent the physical diffusion time $t _ { k }$ Although $\sigma _ { t _ { k } } ^ { G } ( \pmb { y } , \pmb { x } )$ is defined for all $\pmb { x } , \pmb { y } \in [ K ] ^ { d }$ , only values with $d _ { \mathrm { H a m } } ( \pmb { x } , \pmb { y } ) = 1$ enter the scoreestimation objective and reverse dynamics, since $Q ( { \pmb y } , { \pmb x } ) = 0$ otherwise. Thus, the network needs only represent the score at the prescribed grid times and on pairs of states at Hamming distance one. Let $\mathcal { F } ( L , W , S , B ; M ) : = \{ \sigma ^ { G } : G \in \mathcal { G } ( L , W , S , B ) \}$ denote the resulting clipped score class. We assume that the learned score function $\hat { \sigma }$ is an $\varepsilon _ { \mathrm { o p t } }$ -approximate minimizer of the empirical objective (2), i.e.,

$$
\widehat { \mathcal { L } } _ { \mathcal { T } } ^ { \mathrm { D S E } } ( \widehat { \sigma } ) \leq \operatorname* { i n f } _ { \sigma \in \mathcal { F } ( L , W , S , B ; M ) } \widehat { \mathcal { L } } _ { \mathcal { T } } ^ { \mathrm { D S E } } ( \sigma ) + \varepsilon _ { \mathrm { o p t } } ,\tag{3}
$$

where $\varepsilon _ { \mathrm { o p t } } \geq 0$ is the optimization error.

Reverse sampler. At forward time t, the learned score $\widehat { \sigma }$ provides the estimated reverse rates $Q ( \pmb { y } , \pmb { x } ) \widehat { \sigma } _ { t } ( \pmb { y } , \pmb { x } )$ for x $\neq \boldsymbol { y }$ . Starting from the stationary distribution $\pi$ at time $T _ { i }$ a reverse sampler uses these rate estimates to approximate the reverse-time evolution down to the early-stopping time δ. Examples of fixed-grid samplers include τ-leaping (Gillespie, 2001; Campbell et al., 2022), as well as the Euler method and the Tweedie τ-leaping sampler (Lou et al., 2024). Non-fixed-grid methods include Gillespie-based samplers (Gillespie, 1976; Liu et al., 2025) and uniformization (Jensen, 1953; Chen and Ying, 2025), with score evaluations at random or adaptively chosen times. We focus on fixed-grid samplers, whose score evaluations are restricted to a prescribed deterministic time grid $\tau$

Definition 1 (Fixed-grid score-based sampler). A fixed-grid score-based sampler $\mathcal { A }$ takes as input an initial draw $X _ { T } \sim \pi$ , a grid $\tau$ , and the grid score functions $\{ \sigma _ { t _ { k } } \} _ { k = 1 } ^ { N }$ , and returns a random state $X _ { \delta } ^ { \mathcal { A } , \mathcal { T } , \sigma } \in [ K ] ^ { d }$ . We denote its output distribution by $Q _ { \delta } ^ { \mathcal { A } , \mathcal { T } , \sigma }$

For the fixed grid $\tau$ and the learned score $\widehat { \sigma }$ in (3), we abbreviate the output state and distribution as $X _ { \delta } ^ { \mathcal { A } }$ and $Q _ { \delta } ^ { \mathcal { A } }$ , respectively. For $M \geq 1$ , define the class of M-bounded grid scores by $\Sigma _ { M } ( \mathcal { T } ) : = \big \{ \sigma : M ^ { - 1 } \leq \sigma _ { t _ { k } } ( \pmb { y } , \pmb { x } ) \leq M$ for every $k \in [ N ] , \ x , y \in [ K ] ^ { d } \}$ . We formulate our sampleragnostic analysis through the following assumption, which separates the score-estimation error from an approximation remainder accounting for initialization and time discretization.

Assumption 2 (Score-based sampler guarantee). For any $P _ { 0 } \in \Delta _ { K ^ { d } }$ , grid T, $M \geq 1$ , and score $\sigma \in \Sigma _ { M } ( \mathcal { T } ) , D _ { \mathrm { K L } } \left( P _ { \delta } \| Q _ { \delta } ^ { A , \mathcal { T } , \sigma } \right) \leq C _ { \mathcal { A } } \left\{ \mathcal { L } _ { \mathcal { T } } ^ { \mathrm { S E } } ( \sigma ; P _ { 0 } ) + \mathcal { R } _ { \mathcal { A } } ( \mathcal { T } ; M ) \right\}$ , where $C _ { A } > 0$ depends only on the sampler. The remainder $\mathcal { R } _ { \mathcal { A } } ( T ; M ) \ge 0$ is deterministic and independent of $P _ { 0 }$ and of the particular score values beyond the bound M.

Guarantees based on interval-weighted grid score-entropy loss are established for fixed-grid samplers under uniform discrete diffusion in Liang et al. (2025); Dmitriev et al. (2026). In particular, Theorem 1 of Dmitriev et al. (2026), restated in our notation as Theorem A.15, provides a guarantee for τ-leaping under its stated grid conditions, with remainder $\mathscr { R } _ { \tau } ( \tau ; M ) = e ^ { - T } d \log K { + } \Delta d \log ( K / \Delta )$ where $\Delta$ denotes the largest grid interval.

## 3 Statistical Guarantees

## 3.1 Guarantee in Total Variation

We begin our study of distribution estimation under the TV loss, with proofs appearing in Appendix B. For any fixed-grid sampler satisfying Assumption 2, Theorem 3 shows that uniform discrete diffusion attains the rate $\mathcal { O } ( \sqrt { \mathfrak { s } _ { n } ( P _ { 0 } ) / n } )$ through score estimation and reverse-time sampling. We complement this upper bound with a matching minimax lower bound over $\mathcal { P } _ { s _ { n } }$ in Theorem 5. Let $D _ { n } : = $ $| \{ X _ { 0 } ^ { ( 1 ) } , . . . , X _ { 0 } ^ { ( n ) } \} |$ denote the number of distinct observations in the training sample.

Theorem 3 (TV upper bound for uniform discrete diffusion). Define the natural clipping bound $\begin{array} { r } { M : = \frac { 1 + ( K - 1 ) e ^ { - t _ { 1 } } } { 1 - e ^ { - t _ { 1 } } } } \end{array}$ , which uniformly bounds the true score between states of Hamming distance one for all $t \geq t _ { 1 }$ . Suppose that the fixed-grid sampler A satisfies Assumption 2 and that $\mathcal { R } _ { \mathcal { A } } ( T ; M ) \lesssim$ $\mathfrak { s } _ { n } ( P _ { 0 } ) / n , ~ \varepsilon _ { \mathrm { o p t } } \ \lesssim \ \mathfrak { s } _ { n } ( P _ { 0 } ) / n , ~ d \delta \ \lesssim \ \sqrt { \mathfrak { s } _ { n } ( P _ { 0 } ) / n }$ and $T \geq 1$ . Then there exists a clipped ReLU score class $\mathcal { F } ( L , W , S , B ; M )$ with $\begin{array} { r } { L \ \lesssim \ d ^ { 2 } \log ^ { 2 } \left( \frac { K } { 1 - e ^ { - t _ { 1 } } } \right) + \log ^ { 2 } \left( \frac { n T } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) , \ W \ \lesssim \ d K N + d D _ { n } + } \end{array}$ $\begin{array} { r } { d ^ { 3 } \log ^ { 3 } \left( \frac { K } { 1 - e ^ { - t } \bar { 1 } } \right) + \log ^ { 3 } \left( \frac { n T } { s _ { n } ( P _ { 0 } ) } \right) , S \lesssim d K ^ { 2 } N + d ^ { 2 } D _ { n } \log \left( \frac { K } { 1 - e ^ { - t } \bar { 1 } } \right) + d D _ { n } \log \left( \frac { n T } { s _ { n } ( P _ { 0 } ) } \right) + d ^ { 4 } \log ^ { 4 } \left( \frac { K } { 1 - e ^ { - t } \bar { 1 } } \right) + } \end{array}$ $\begin{array} { r } { \log ^ { 4 } \left( \frac { n T } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) , B \lesssim N + \left( \frac { K } { 1 - e ^ { - t _ { 1 } } } \right) ^ { \smash { 2 d } } + \frac { d n M T } { \mathfrak { s } _ { n } ( P _ { 0 } ) } } \end{array}$ , such that every εopt-optimizer ô of $\widehat { \mathcal { L } } _ { \mathcal { T } } ^ { \mathrm { D S E } }$ over this class

satisfies

$$
\mathbb { E } _ { P _ { 0 } } d _ { \mathrm { T V } } \left( P _ { 0 } , Q _ { \delta } ^ { \mathcal { A } } \right) \lesssim \sqrt { \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } } .
$$

Theorem 3 is stated for a generic fixed-grid sampler and applies once its remainder term $\mathcal { R } _ { A }$ is controlled at the stated rate. For the τ-leaping sampler, instantiating Theorem 3 with this remainder yields an explicit sampling grid and matching architecture bounds as shown in Corollary 4.

Corollary 4 (TV guarantee for τ-leaping sampler). For the τ-leaping sampler, the grid choice $\begin{array} { r } { \mathcal { T } = \Big \{ \delta = 0 , \Delta , \ldots , ( N - 1 ) \Delta , T = \log \Big ( \frac { e d n \log K } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \Big ) \Big \} } \end{array}$ , where $\begin{array} { r } { \Delta = \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { d n \log ( K d n / \mathfrak { s } _ { n } ( P _ { 0 } ) ) } } \end{array}$ , and $\begin{array} { r } { N = \left\lceil \frac { T } { \Delta } \right\rceil } \end{array}$ satisfies the sampler conditions of Theorem 3. The clipped ReLU score class can be chosen with $\begin{array} { r } { M = \frac { 1 + ( K - 1 ) e ^ { - \Delta } } { 1 - e ^ { - \Delta } } , L \lesssim d ^ { 2 } \log ^ { 2 } \left( \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) , W \lesssim \frac { d ^ { 2 } K n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \log ^ { 2 } \left( \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) + d D _ { n } + d ^ { 3 } \log ^ { 3 } \left( \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) , S \lesssim } \end{array}$ $\begin{array} { r } { \frac { d ^ { 2 } K ^ { 2 } n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \log ^ { 2 } \Big ( \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \Big ) + d ^ { 2 } D _ { n } \log \Big ( \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \Big ) + d ^ { 4 } \log ^ { 4 } \Big ( \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \Big ) } \end{array}$ , and $\begin{array} { r } { B \lesssim \left[ \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \log \left( \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) \right] ^ { 2 d } } \end{array}$

Interpretation of the sampling and network parameters. The choices of the terminal time T and grid spacing $\Delta$ in Corollary 4 calibrate the initialization and discretization errors to the statistical scale ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) / n$ , while no early stopping is needed under TV loss, so we take $\delta = 0$ . The resulting number of score evaluations and clipping level satisfy $N = \widetilde { \mathcal { O } } ( d n / { \mathfrak { s } } _ { n } ( P _ { 0 } ) )$ and $M = \widetilde { \mathcal { O } } ( K d n / \mathfrak { s } _ { n } ( P _ { 0 } ) )$ respectively. Together with the displayed bounds on the network depth L, width W, and sparsity S, these quantities depend polynomially on $K , d ,$ and $n / { \mathfrak { s } } _ { n } ( P _ { 0 } )$ , rather than on the ambient cardinality $K ^ { d }$ . Thus, the TV guarantee avoids polynomial dependence on the ambient state-space size in the number of score evaluations, the clipping level, and the network size. The number of distinct states in the sample, $D _ { n } ,$ can be upper bounded by ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ in expectation and by high probability by Lemma A.2.

Minimax lower bound. As a classical benchmark, Lemma A.6 shows that the empirical distribution $\widehat { P } _ { 0 }$ also achieves the rate $\sqrt { \mathfrak { s } _ { n } ( P _ { 0 } ) / n }$ . Theorem 5 shows that this rate is minimax optimal over $\mathcal { P } _ { s _ { n } }$

Theorem 5 (Minimax rate under TV loss). For every $2 \leq s _ { n } \leq \operatorname* { m i n } \{ n , K ^ { d } \}$ ，

$$
\operatorname* { i n f } _ { \widetilde { P } } \operatorname* { s u p } _ { P _ { 0 } \in \mathcal { P } _ { s _ { n } } } \mathbb { E } _ { P _ { 0 } } d _ { \mathrm { T V } } ( P _ { 0 } , \widetilde { P } ) \asymp \sqrt { s _ { n } / n } .
$$

Comparison with existing TV bounds. Since the score-entropy loss is naturally aligned with KL divergence, relatively few works derive TV guarantees for uniform discrete diffusion. To our knowledge, only Zhang et al. (2026) obtain a TV bound for uniform discrete diffusion, of order $\widetilde { \mathcal { O } } \big ( \sqrt { d K ^ { d + 1 } / n } \big )$ , which becomes vacuous when $n \ll K ^ { d }$ . In contrast, our rate depends on the effective support size ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ and, by Theorem 5, attains the minimax-optimal rate over the corresponding effective support classes. In the non-saturated regimes of (1), this yields rates $\mathcal { O } ( n ^ { ( \alpha ^ { - 1 } - 1 ) / 2 } )$ and $\ O ( { \sqrt { \log n / n } } )$ for Zipfian and geometric decay, respectively.

## 3.2 Guarantee in KL Divergence

We now turn to distribution estimation under KL loss, with proofs appearing in Appendix C. Unlike in the TV setting, the empirical distribution $ { \widehat { P } } _ { 0 }$ generally has infinite expected KL risk. Thus, obtaining a meaningful KL guarantee requires assigning positive probability beyond the observed training samples, making KL loss a stringent measure of distributional generalization. For any fixed-grid sampler satisfying Assumption 2, Theorem 6 shows that uniform discrete diffusion with early-stopping time $\delta = 1 / n$ attains the rate $\begin{array} { r } { \mathcal { O } \bigl ( \frac { 1 } { n } \mathfrak { s } _ { n } ( P _ { 0 } ) \log ( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) ) \log n \bigr ) } \end{array}$ . We complement this upper bound with a minimax lower bound over $\mathcal { P } _ { s _ { n } }$ in Theorem $^ { 8 , }$ showing that uniform discrete diffusion is minimax optimal up to a factor of log n. Together, these results show that uniform discrete diffusion can generalize well beyond the observed training samples rather than merely memorizing the empirical distribution.

Theorem 6 (KL guarantee for uniform discrete diffusion). Set early-stopping time $\delta : = 1 / n$ Define the natural clipping bound $\begin{array} { r } { M : = \frac { 1 + ( K - 1 ) e ^ { - t _ { 1 } } } { 1 - e ^ { - t _ { 1 } } } } \end{array}$ , which uniformly bounds the true score between states of Hamming distance one for all $t \geq t _ { 1 }$ . Suppose that the fxed-grid sampler A satisfies Assumption 2 and that $\begin{array} { r } { \mathcal { R } _ { A } ( T ; M ) \lesssim \Big ( \frac { 1 - e ^ { - 1 / n } } { K } \Big ) ^ { d } , \varepsilon _ { \mathrm { o p t } } \lesssim \Big ( \frac { 1 - e ^ { - 1 / n } } { K } \Big ) ^ { d } } \end{array}$ , and $T \geq 1$ . Then there exists a clipped ReLU score class $\mathcal { F } ( L , W , S , B ; M )$ with $L \lesssim d ^ { 2 } \log ^ { 2 } ( K n ) + \log ^ { 2 } T , W \lesssim d K N +$ $d D _ { n } + d ^ { 3 } \log ^ { 3 } ( K n ) + \log ^ { 3 } T , S \lesssim d K ^ { 2 } N + d ^ { 2 } D _ { n } \log ( K n ) + d D _ { n } \log T + d ^ { 4 } \log ^ { 4 } ( K n ) + \log ^ { 4 } T ;$ and $\begin{array} { r } { B \lesssim N + \left( \frac { K } { 1 - e ^ { - t _ { 1 } } } \right) ^ { 2 d } + d M T \left( \frac { K } { 1 - e ^ { - 1 / n } } \right) ^ { d } } \end{array}$ , such that every $\varepsilon _ { \mathrm { o p t } } { - o p t i m i z e r } \widehat { \sigma } o f \widehat { \mathcal { L } } _ { T } ^ { \mathrm { D S E } }$ over this class satisfies

$$
\mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } \left( P _ { 0 } \| Q _ { \delta } ^ { \cal A } \right) \lesssim \frac { 1 } { n } \mathfrak { s } _ { n } ( P _ { 0 } ) \log \Bigl ( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \Bigr ) \log n .
$$

Once again, for the τ-leaping sampler, we can obtain specific sampling grid and network

architecture choices as highlighted in Corollary 7.

Corollary 7 (KL guarantee for τ-leaping sampler). For the τ-leaping sampler, the grid $\mathcal { T } =$ $\{ \delta = \textstyle { \frac { 1 } { n } } , \textstyle { \frac { 1 } { n } } + \Delta , \ldots , { \frac { 1 } { n } } + ( N - 1 ) \Delta , T \}$ with $T = \log [ e d ( K / ( 1 - e ^ { - 1 / n } ) ) ^ { d } \log K ] , \Delta = ( d ^ { 2 } ( K / ( 1 -$ $e ^ { - 1 / n } ) ) ^ { d } \log ( K n ) ) ^ { - 1 }$ , and $N = \lceil ( T - 1 / n ) / \Delta \rceil$ satisfies the sampler conditions of Theorem 6. The clipped ReLU score class can be chosen with $\begin{array} { r } { M = \frac { 1 + ( K - 1 ) e ^ { - ( 1 / n + \Delta ) } } { 1 - e ^ { - ( 1 / n + \Delta ) } } , L \lesssim d ^ { 2 } \log ^ { 2 } ( K n ) , W \lesssim } \end{array}$ $d ^ { 4 } K ( K / ( 1 - e ^ { - 1 / n } ) ) ^ { d } \log ^ { 2 } ( K n ) , S \lesssim d ^ { 4 } K ^ { 2 } ( K / ( 1 - e ^ { - 1 / n } ) ) ^ { d } \log ^ { 2 } ( K n )$ , and $B \lesssim ( K / ( 1 - e ^ { - 1 / n } ) ) ^ { 2 d }$

Interpretation of the sampling and network parameters. Unlike under TV loss, early stopping is essential under KL loss. We take $\delta = 1 / n$ since the early-stopped empirical distribution $\widehat { P } _ { 1 / n }$ already achieves the rate of Theorem 6. The choices of the terminal time T and grid spacing $\Delta$ in Corollary 7 calibrate the initialization and discretization errors to the sampler-accuracy scale $\left( ( 1 - e ^ { - 1 / n } ) / K \right) ^ { d }$ , which is much smaller than the final statistical KL rate, unlike in the TV case. This difference arises from the absence of a triangle inequality for KL divergence. Instead, we use Lemma A.13, which requires the sampling error to be controlled at the uniform lower bound on the probabilities at time $1 / n$ Although the clipping level $M = \widetilde { \mathcal { O } } ( K n )$ and network depth L remain mild, the number of sampling steps $N ,$ network width $W$ , and sparsity S incur exponential dependence on $d .$ This unfavorable computational dependence is not intrinsic to the statistical KL rate: in Appendix D, we construct an exact one-step empirical reverse sampler that achieves the same statistical rate with only ${ \mathcal { O } } ( d ( D _ { n } + K ) )$ operations per generated sample.

Minimax lower bound. The following theorem characterizes the minimax rate over the effective support class $\mathcal { P } _ { s _ { n } }$ , showing that uniform discrete diffusion achieves the minimax rate up to a factor of log n.

Theorem 8 (Minimax rate under KL loss). For every $2 \leq s _ { n } \leq \operatorname* { m i n } \{ n , K ^ { d } \}$ ，

$$
\operatorname* { i n f } _ { \widetilde { P } } \operatorname* { s u p } _ { P _ { 0 } \in \mathcal { P } _ { s _ { n } } } \mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } ( P _ { 0 } \| \widetilde { P } ) \asymp \frac { s _ { n } } { n } \log ( e K ^ { d } / s _ { n } ) .
$$

Comparison with classical distribution estimators. There has been substantial interest in estimating discrete distributions (Braess et al., 2002; Paninski, 2004; Braess and Sauer, 2004; van der Hoeven et al., 2025) and the related missing mass problem (Good, 1953; Orlitsky and Suresh, 2015).

Classical asymptotic theory shows that, for a fixed state space of size $K ^ { d }$ , the minimax expected KL risk is $( K ^ { d } - 1 ) / ( 2 n ) + o ( 1 / n )$ as $n \to \infty$ . This fixed state space asymptotic does not describe the large state space regime relevant to discrete diffusion

Since the empirical distribution $\widehat { P } _ { 0 }$ generally has infinite KL risk, classical discrete distribution estimation has long relied on smoothing to assign positive probability to unseen states. Among fixed additive-smoothing estimators of the form $( n \widehat { P } _ { 0 } ( x ) + \lambda ) / ( n + \lambda K ^ { d } )$ , the Laplace estimator with $\lambda = 1$ achieves the minimax-optimal expected risk of order Θ $\left( \log ( 1 + K ^ { d } / n ) \right)$ over the unrestricted class $\Delta _ { K ^ { d } }$ (Paninski, 2004; Mourtada, 2026). However, this rate does not vanish when $n \ll K ^ { d }$ The adaptive additive-smoothing estimator with $\lambda = D _ { n } / K ^ { d }$ (Mourtada, 2026) and the absolutediscounting estimator (Falahatgar et al., 2017) both achieve the minimax-optimal rate over the effective support class of Theorem 8. Moreover, when the effective support size is maximal, i.e., $s _ { n } = \operatorname* { m i n } \{ n , K ^ { d } \}$ , which corresponds to the unrestricted class, the minimax rates in Theorems 5 and 8 recover the corresponding classical unrestricted minimax rates (Lemma A.3). Thus, minimax theory based on effective support size extends the classical unrestricted theory while remaining informative for distributions whose effective support size is small.

Comparison with existing KL bounds. Several recent works derive finite-sample KL guarantees for uniform discrete diffusion. Cho and Wu (2026) establish KL rate of order ${ \widetilde { \mathcal { O } } } ( K ^ { d } / n )$ , which they claim is nearly minimax optimal. However, their guarantees are established in the regime $n \gtrsim K ^ { d }$ which does not capture the regime $n \ll K ^ { d }$ typical of practical discrete diffusion. Wakasugi and Suzuki (2025) obtain a high-probability KL bound of the same order, together with a refined bound that removes the explicit polynomial dependence on $K ^ { d }$ . This refinement requires decay of the target distribution's coefficients in an eigenbasis of the generator, a structural condition that is difficult to verify in practice. Moreover, the refined bound is not minimax optimal. Srikanth et al. (2026) derive a high-probability KL guarantee with explicit neural-network complexity, but require unrealistic full support assumption and incur polynomial dependence on the maximum initial score ratio, which may be as large as the ambient cardinality $K ^ { d }$ . In contrast, our bound adapts to the effective support size ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ without such structural assumptions, are nearly minimax optimal under effective support class $\mathcal { P } _ { s _ { n } }$ , and yields rates $\begin{array} { r } { \mathcal { O } \Big ( n ^ { \alpha ^ { - 1 } - 1 } \log \frac { e K ^ { d } } { n ^ { 1 / \alpha } } \log n \Big ) } \end{array}$ and $\mathcal { O } \Big ( \frac { \log ^ { 2 } n } { n } \log \frac { e K ^ { d } } { \log n } \Big )$ for Zipfian and geometric decay, respectively, in the non-saturated regimes of (1).

## 4 Experiments

Controlled populations. We conduct a controlled experiment to evaluate whether uniform discrete diffusion models learn populations with smaller effective support sizes more accurately when the size of the ambient state space is held fixed. We first train a byte-level BPE tokenizer with vocabulary size $K = 2 0 4 8$ on TinyStories (Eldan and Li, 2023). Using the resulting tokenized corpus, we train an 88M-parameter, 12-layer decoder-only Transformer as an autoregressive teacher model that predicts each token conditioned on the preceding tokens. We then use the frozen teacher at temperatures {0.4, 0.7, 1.0} to define three population distributions over d = 32-token sequences, all on the same ambient state space of size $K ^ { d } = 2 0 4 8 ^ { 3 2 }$ . Lower temperatures concentrate the teacher's next-token probabilities and therefore produce populations with smaller effective support sizes.

Training and Evaluation. For each population, we consider seven training-set sizes $( n = 1 0 \mathrm { K } -$ 1M). For every population and training-set size, we train a 96M-parameter uniform SEDD model (Lou et al., 2024) for 50 epochs. Each experimental setting is repeated three times. Directly computing full-sequence TV or KL is infeasible because it would require tractable access to the model probabilities over $2 0 4 8 ^ { 3 2 }$ possible sequences. We therefore draw 100K samples from each trained SEDD model and 100K fresh samples from the corresponding teacher population and compare them using two sample-based discrepancies. Teacher-NLL KS compares the distributions of sequence negative log-likelihood under the teacher; the corresponding population discrepancy lower-bounds, but does not estimate, full-sequence TV. The second metric, 1 — MAUVE, compares the samples in a fixed GPT-2-large representation (Pillutla et al., 2021). We apply both metrics to two independent sets of 100K samples from each teacher population as a teacher-versus-teacher null control, which indicates their finite-sample evaluation floors. Lower values indicate better agreement for both metrics.

Results. Figure 2 shows that both teacher-NLL KS and 1 - MAUVE decrease as the training-set size increases. At every training-set size, uniform SEDD models trained on lower-temperature populations achieve smaller discrepancies under both metrics. Since these populations have smaller effective support sizes while sharing the same ambient state space, the results support effective support size as a meaningful measure of statistical difficulty for uniform discrete diffusion.

![](images/96cee3845696c860fda5e5e83e37ab6144a6a0a1d79fc88f339a14572bb562b6.jpg)

![](images/09da6d53b30dde6f49e15d47ad82e7f2657d2b2af0c9a1ffddce64f487d8c321.jpg)  
Figure 2: Uniform discrete diffusion on controlled populations. Teacher-NLL KS (left) and $1 - \mathrm { M A U V E }$ (right) versus training-set size for teacher temperatures {0.4, 0.7, 1.0}, with ambient state space $\mathrm { 2 0 4 8 ^ { 3 2 } }$ . Lines and markers show means over three runs after 50 training epochs, and shaded bands span the minimum and maximum across runs. Dotted lines indicate teacher-teacher null controls. Lower values indicate better agreement between population and sample distributions.

## 5 Conclusion

In this paper, we develop a statistical theory for uniform discrete diffusion models on finite product spaces whose ambient cardinality $K ^ { d }$ can be astronomically large. Our analysis is governed by the effective support size ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ , a sample-size-dependent measure of distributional complexity. We show that uniform discrete diffusion achieves the minimax-optimal TV rate $\mathcal { O } ( \sqrt { \mathfrak { s } _ { n } ( P _ { 0 } ) / n } )$ . Under KL loss, early stopping at time $1 / n$ yields the rate $\begin{array} { r } { \mathcal { O } \left( \frac { 1 } { n } \mathfrak { s } _ { n } ( P _ { 0 } ) \log ( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) ) \log n \right) } \end{array}$ , which is within a factor of log n of the minimax benchmark. This KL guarantee captures generalization beyond the observed samples, since the empirical distribution generally assigns zero probability to unseen states and therefore has infinite KL risk. These minimax rates identify three regimes in which consistent distribution estimation is impossible, possible only under TV loss, or possible under both TV and KL losses. Our controlled experiment with uniform SEDD models complements these theoretical findings and supports effective support size as a meaningful measure of statistical difficulty beyond ambient cardinality alone. Important directions for future work include closing the remaining logarithmic gap under KL loss, controlling optimization error in score estimation, and extending the analysis to masking discrete diffusion models.

## References

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems, volume 34, pages 17981–17993, 2021. URL https://proceedings.neurips.cc/paper \_files/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html.

Anna Ben-Hamou, Stéphane Boucheron, and Mesrob I. Ohannessian. Concentration inequalities in the infinite urn scheme for occupancy counts and the missing mass, with applications. Bernoulli, 23(1):249–287, 2017. doi: 10.3150/15-BEJ743.

Dietrich Braess and Thomas Sauer. Bernstein polynomials and learning theory. Journal of Approximation Theory, 128(2):187–206, 2004.

Dietrich Braess, Jürgen Forster, Tomas Sauer, and Hans U Simon. How to achieve minimax expected kullback-leibler distance from an unknown finite distribution. In International Conference on Algorithmic Learning Theory, pages 380–394. Springer, 2002.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Tom Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems, volume 35, pages 28266–28279, 2022. URL https: //proceedings.neurips.cc/paper\_files/paper/2022/hash/b5b528767aa35f5b1a60fe0aaec a0563-Abstract-Conference.html

Hongrui Chen and Lexing Ying. Convergence analysis of discrete diffusion model: Exact implementation through uniformization. Journal of Machine Learning, 4(2):108–127, 2025. doi: 10.4208/jml.240812.

Cholyeon Cho and Yuchen Wu. Minimax optimality of score-entropy discrete diffusion. arXiv preprint arXiv:2608.20635, 2026. URL https://arxiv.org/abs/2608.20635.

Daniil Dmitriev, Zhihan Huang, and Yuting Wei. Efficient sampling with discrete diffusion models: Sharp and adaptive guarantees. In Proceedings of the 39th Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 2038–2104, 2026. URL https: //proceedings.mlr.press/v336/dmitriev26a.html.

Ronen Eldan and Yuanzhi Li. TinyStories: How small can language models be and still speak coherent English?arXiv preprint arXiv:2305.07759, 2023. URL https://arxiv.org/abs/2305.07759.

Moein Falahatgar, Mesrob I. Ohannessian, Alon Orlitsky, and Venkatadheeraj Pichapati. The power of absolute discounting: All-dimensional distribution estimation. In Advances in Neural Information Processing Systems, volume 30, pages 6660–6669, 2017. URL https://proceedings.neurips.cc /paper\_files/paper/2017/hash/331316d4efb44682092a006307b9ae3a-Abstract.html.

Daniel T. Gillespie. A general method for numerically simulating the stochastic time evolution of coupled chemical reactions. Journal of Computational Physics, 22(4):403–434, 1976. doi: 10.1016/0021-9991(76)90041-3.

Daniel T. Gillespie. Approximate accelerated stochastic simulation of chemically reacting systems. The Journal of Chemical Physics, 115(4):1716–1733, 2001. doi: 10.1063/1.1378322.

I. J. Good. The population frequencies of species and the estimation of population parameters. Biometrika, 40(3/4):237–264, 1953. doi: 10.1093/biomet/40.3-4.237.

Yanjun Han, Jiantao Jiao, and Tsachy Weissman. Minimax estimation of discrete distributions under $\ell _ { 1 }$ loss. IEEE Transactions on Information Theory, 61(11):6343–6354, 2015. doi: 10.1109/ TIT.2015.2478816.

Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. DiffusionBERT: Improving generative masked language models with diffusion models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4521–4534, 2023. URL https://aclanthology.org/2023.acl-1ong.248/.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020. URL https: //proceedings.neurips.cc/paper\_files/paper/2020/hash/4c5bcfec8584af0d967f1ab1017 9ca4b-Abstract.html

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, and Max Welling. Argmax flows and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information

Processing Systems, volume 34, pages 12454–12465, 2021. URL https://proceedings .neurips.   
cc/paper\_files/paper/2021/hash/67d96d458abdef21792e6d8e590244e7-Abstract.html.

Arne Jensen. Markoff chains as an aid in the study of Markoff processes. Scandinavian Actuarial Journal, 1953(sup1):87–91, 1953. doi: 10.1080/03461238.1953.10419459.

Sudeep Kamath, Alon Orlitsky, Dheeraj Pichapati, and Ananda Theertha Suresh. On learning distributions from their samples. In Proceedings of the 28th Conference on Learning Theory, volume 40 of Proceedings of Machine Learning Research, pages 1066–1100, 2015. URL https: //proceedings.mlr.press/v40/Kamath15.html.

Samuel Karlin. Central limit theorems for certain infinite urn schemes. Journal of Mathematics and Mechanics, 17(4):373–401, 1967. URL https://www.jstor.org/stable/24902077.

Yuchen Liang, Yingbin Liang, Lifeng Lai, and Ness Shroff. Discrete diffusion models: Novel analysis and new sampler guarantees. In Advances in Neural Information Processing Systems, volume 38, pages 183461-183498, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/202 5/hash/f1e7c90552850afcc2558d78950c519d-Abstract-Conference.html.

Sulin Liu, Juno Nam, Andrew Campbell, Hannes Stärk, Yilun Xu, Tommi Jaakkola, and Rafael Gómez-Bombarelli. Think while you generate: Discrete diffusion with planned denoising. In International Conference on Learning Representations, 2025. URL https://openreview.net/f orum?id=MJNywBdSDy.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 32819–32848, 2024. URL https://proceedings.mlr.press/v235/1ou24a.html.

Jaouad Mourtada. Estimation of discrete distributions in relative entropy, and the deviations of the missing mass. Mathematical Statistics and Learning, 9(1/2):69–144, 2026. doi: 10.4171/MSL/55.

J. R. Norris. Markov Chains, volume 2 of Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 1997. doi: 10.1017/CBO9780511810633.

Kazusato Oko, Shunta Akiyama, and Taiji Suzuki. Diffusion models are minimax optimal distribution estimators. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 26517–26582, 2023. URL https://proceedi ngs.mlr.press/v202/oko23a.html.

Alon Orlitsky and Ananda Theertha Suresh. Competitive distribution estimation: Why is Good-Turing good. In Advances in Neural Information Processing Systems, volume 28, pages 2143–2151, 2015. URL https://proceedings.neurips.cc/paper\_files/paper/2015/hash/d759175de8e a5b1d9a2660e45554894f-Abstract.html.

Liam Paninski. Variational minimax estimation of discrete distributions under KL loss. In Advances in Neural Information Processing Systems, volume 17, pages 1033–1040, 2004. URL https: //proceedings.neurips.cc/paper\_files/paper/2004/hash/c57168a952f5d46724cf35dfc3d 48a7f-Abstract.html.

Le-Tuyet-Nhi Pham, Dario Shariatian, Antonio Ocello, Giovanni Conforti, and Alain Oliviero Durmus. Discrete Markov probabilistic models: An improved discrete score-based framework with sharp convergence bounds under minimal assumptions. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 49195-49258, 2025. URL https://proceedings.mlr.press/v267/pham25a.html.

Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaid Harchaoui. MAUVE: Measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems, volume 34, pages 4816-4828, 2021. URL https://proceedings.neurips.cc/paper\_files/paper/2021/hash/2 60c2432a0eecc28ce03c10dadc078a4-Abstract.html.

Yinuo Ren, Haoxuan Chen, Grant M. Rotskoff, and Lexing Ying. How discrete and continuous diffusion meet: Comprehensive analysis of discrete diffusion models via a stochastic integral framework. In International Conference on Learning Representations, 2025. URL https://open review.net/forum?id=6awxwQEI82.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pages 2256–2265, 2015. URL https://proceedings.mlr.press/v37/sohl-dickstein15.html.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=PxTI G12RRHS.

Aadithya Srikanth, Mudit Gaur, and Vaneet Aggarwal. Discrete state diffusion models: A sample complexity perspective. In Proceedings of the 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pages 3547–3555, 2026. URL https://proceedings.mlr.press/v300/srikanth26a.html.

Zhiqing Sun and Yiming Yang. DIFUSCO: Graph-based diffusion solvers for combinatorial optimization. In Advances in Neural Information Processing Systems, volume 36, pages 3706–3731, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/0ba520d93c3 df592c83a611961314c98-Abstract-Conference.html.

Dirk van der Hoeven, Julia Olkhovskaia, and Tim van Erven. Nearly minimax discrete distribution estimation in kullback-leibler divergence with high probability. arXiv preprint arXiv:2507.17316, 2025.

Clément Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. DiGress: Discrete denoising diffusion for graph generation. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=UaAD-Nu86WX.

Shintaro Wakasugi and Taiji Suzuki. State size independent statistical error bound for discrete diffusion models. In Advances in Neural Information Processing Systems, volume 38, pages 154062-154097, 2025. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash /caabce23491e5418f4c9ea348359fcd2-Abstract-Conference.html.

Jiayuan Ye, Vitaly Feldman, and Kunal Talwar. Instance-optimality for private kl distribution estimation. arXiv preprint arXiv:2505.23620, 2025.

Zikun Zhang, Zixiang Chen, and Quanquan Gu. Convergence of score-based discrete diffusion models: A discrete-time analysis. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=pq1WUegkza.

Zixuan Zhang, Hengyu Fu, Zhuoran Yang, Mengdi Wang, Tuo Zhao, and Minshuo Chen. Generalization bounds for discrete diffusion: Statistical advantage of masking. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https://openreview.net/forum?i d=0fhq7nBVHu.

## Appendices

A Preliminary Results 24   
A.1 Some Facts on Effective Support Size 24   
A.2 Properties of Uniform Forward Process 26   
A.3 Auxiliary Results for KL Bounds 28   
A.4 Score-Entropy Identities and Sampler Guarantees 33   
A.5 ReLU Network Constructions 34   
B Proofs for TV Results 37   
B.1 Upper Bound: Proofs of Theorem 3 and Corollary 4 37   
B.2 Minimax Lower Bound: Proof of Theorem 5 40   
C Proofs for KL Results 41   
C.1 Upper Bound: Proofs of Theorem 6 and Corollary 7 41   
C.2 Minimax Lower Bound: Proof of Theorem 8 43   
D Exact One-Step Empirical Reverse Sampling 47   
D.1 Exact reverse kernel and sampling algorithm 47   
D.2 Statistical guarantees . 49

Notation. We retain the notation of the main text and introduce only appendix-specific shorthand here. For any probability distribution $\mu$ on $[ K ] ^ { d } .$ write $\mu _ { t } : = \mu P _ { t }$ for its marginal after evolving under the forward process for time t. When the data distribution is $P _ { 0 }$ ,we abbreviate its point masses as $p _ { \pmb { x } } : = P _ { 0 } ( \pmb { x } )$ . For $t > 0 ,$ define

$$
B _ { 0 } ( t ) : = \left( \frac { K } { 1 - e ^ { - t } } \right) ^ { d } , \qquad B _ { 1 } ( t ) : = \frac { 1 + ( K - 1 ) e ^ { - t } } { 1 - e ^ { - t } } .
$$

For $u \in \mathbb { R }$ , write $[ u ] _ { + } : = \operatorname* { m a x } \{ u , 0 \}$ ; in the neural-network constructions, $\rho ( u ) : = \operatorname* { m a x } \{ u , 0 \}$ denotes the ReLU activation.

## A Preliminary Results

## A.1 Some Facts on Effective Support Size

Lemma A.1 (Effective support size under representative decay regimes). Let $p _ { ( 1 ) } \ \geq \ \cdots \ \geq$ $p _ { ( | \operatorname { s u p p } ( P _ { 0 } ) | ) } > 0$ denote nonzero masses of $P _ { 0 }$ in decreasing order. Suppose $n \geq 2$ . In each condition below,  means that the comparison holds uniformly over $1 \leq j \leq | \operatorname { s u p p } ( P _ { 0 } ) |$ with universal constants. Then

$$
\begin{array}{c} \mathfrak { s } _ { n } ( P _ { 0 } ) \asymp \left\{ \operatorname* { m i n } \{ | \operatorname* { s u p p } ( P _ { 0 } ) | , n \} , \quad  & { i f p _ { ( j ) } \asymp | \operatorname* { s u p p } ( P _ { 0 } ) | ^ { - 1 } , \medskip } \\ { \operatorname* { m i n } \{ | \operatorname* { s u p p } ( P _ { 0 } ) | , n ^ { 1 / \alpha } \} , \quad } & { i f p _ { ( j ) } \asymp j ^ { - \alpha } , } \\ { \operatorname* { m i n } \{ | \operatorname* { s u p p } ( P _ { 0 } ) | , \log n \} , \quad i f p _ { ( j ) } \asymp e ^ { - c j } , } \end{array} \right.
$$

where $\alpha > 1$ and $c > 0$ are fxed. The implicit constants in the conclusion depend only on the comparison constants and, in the second and third cases, on α and c, respectively.

Proof. Since $u \asymp v$ implies min $\{ n u , 1 \} \asymp \operatorname* { m i n } \{ n v , 1 \}$ , we may work directly with the representative sequences in the three cases.

If $p _ { ( j ) } \asymp | \operatorname { s u p p } ( P _ { 0 } ) | ^ { - 1 }$ , direct calculation gives ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \asymp \operatorname* { m i n } \{ | \operatorname { s u p p } ( P _ { 0 } ) | , n \}$

If $p _ { ( j ) } \asymp j ^ { - \alpha }$ , then

$$
\mathfrak { s } _ { n } ( P _ { 0 } ) \asymp \sum _ { j = 1 } ^ { | \operatorname { s u p p } ( P _ { 0 } ) | } \operatorname* { m i n } \{ n j ^ { - \alpha } , 1 \} .
$$

${ \mathrm { I f ~ } } | \operatorname { s u p p } ( P _ { 0 } ) | \leq n ^ { 1 / \alpha }$ , all summands equal 1, so $\mathfrak { s } _ { n } ( P _ { 0 } ) \asymp | \operatorname { s u p p } ( P _ { 0 } ) | . \mathrm { ~ I f ~ } | \operatorname { s u p p } ( P _ { 0 } ) | > n ^ { 1 / \alpha }$ , the first $\lfloor n ^ { 1 / \alpha } \rfloor$ summands equal 1, while

$$
n \sum _ { j > \lfloor n ^ { 1 / \alpha } \rfloor } j ^ { - \alpha } \lesssim n ^ { 1 / \alpha } .
$$

Thus ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \asymp n ^ { 1 / \alpha }$ , and hence ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \asymp \operatorname* { m i n } \{ | \operatorname { s u p p } ( P _ { 0 } ) | , n ^ { 1 / \alpha } \}$

If $p _ { ( j ) } \asymp e ^ { - c j }$ , then

$$
{ \mathfrak { s } } _ { n } ( P _ { 0 } ) \asymp \sum _ { j = 1 } ^ { | \operatorname { s u p p } ( P _ { 0 } ) | } \operatorname* { m i n } \{ n e ^ { - c j } , 1 \} .
$$

$\mathrm { I f ~ } | \operatorname { s u p p } ( P _ { 0 } ) | \leq c ^ { - 1 } \log n .$ all summands equal 1, so $\mathfrak { s } _ { n } ( P _ { 0 } ) \asymp | \operatorname { s u p p } ( P _ { 0 } ) | . \mathrm { ~ I f ~ } | \operatorname { s u p p } ( P _ { 0 } ) | > c ^ { - 1 } \log n$

the summands equal 1 for $j \leq \lfloor c ^ { - 1 } \log n \rfloor$ , while

$$
n \sum _ { j > \lfloor c ^ { - 1 } \log n \rfloor } e ^ { - c j } \lesssim 1 .
$$

Thus ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \asymp \log n ,$ and hence ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \asymp \operatorname* { m i n } \{ | \operatorname { s u p p } ( P _ { 0 } ) | , \log n \}$

Lemma A.2 (Effective support size and distinct observations). Let $X _ { 1 } , \ldots , X _ { n }$ be i.i.d. samples from $P _ { 0 }$ , and let $D _ { n } : = | \{ X _ { 1 } , \ldots , X _ { n } \} |$ denote the number of distinct observations. Then

$$
( 1 - e ^ { - 1 } ) \mathfrak { s } _ { n } ( P _ { 0 } ) \leq \mathbb { E } _ { P _ { 0 } } D _ { n } \leq \mathfrak { s } _ { n } ( P _ { 0 } ) .
$$

Moreover, for every $\eta \in ( 0 , 1 )$ , with probability at least $1 - \eta ,$

$$
D _ { n } \leq 2 \big ( \mathfrak { s } _ { n } ( P _ { 0 } ) + \log \eta ^ { - 1 } \big ) .
$$

Proof. We have $\begin{array} { r } { \mathbb { E } _ { P _ { 0 } } D _ { n } = \sum _ { x } \{ 1 - ( 1 - P _ { 0 } ( x ) ) ^ { n } \} } \end{array}$ . Since $1 - ( 1 - u ) ^ { n } \leq \operatorname* { m i n } \{ n u , 1 \}$ for $u \in [ 0 , 1 ]$ it follows that $\mathbb { E } _ { P _ { 0 } } D _ { n } \le { \mathfrak { s } } _ { n } ( P _ { 0 } )$ . Conversely, $1 - ( 1 - u ) ^ { n } \geq 1 - e ^ { - n u } \geq ( 1 - e ^ { - 1 } )$ min{nu, 1}, so $\mathbb { E } _ { P _ { 0 } } D _ { n } \geq ( 1 - e ^ { - 1 } ) { \mathfrak { s } } _ { n } ( P _ { 0 } )$ . Finally, Mourtada (2026, Lemma 10.7) gives, with probability at least $1 - \eta$

$$
D _ { n } \leq 2 \big ( \mathbb { E } _ { P _ { 0 } } D _ { n } + \log \eta ^ { - 1 } \big ) ,
$$

and the claim follows from $\mathbb { E } _ { P _ { 0 } } D _ { n } \le { \mathfrak { s } } _ { n } ( P _ { 0 } )$

Lemma A.3 (Recovery of the unrestricted KL rate). Let $s _ { \operatorname* { m a x } } : = \operatorname* { m i n } \{ K ^ { d } , n \}$ . Then,

$$
\log \left( 1 + \frac { K ^ { d } } { n } \right) \leq \frac { s _ { \operatorname* { m a x } } } { n } \log \left( \frac { e K ^ { d } } { s _ { \operatorname* { m a x } } } \right) \leq 2 \log \left( 1 + \frac { K ^ { d } } { n } \right) .
$$

Proof. If $K ^ { d } \leq n$ , then $s _ { \operatorname* { m a x } } = K ^ { d }$ and

$$
\frac { s _ { \mathrm { m a x } } } { n } \log \left( \frac { e K ^ { d } } { s _ { \mathrm { m a x } } } \right) = \frac { K ^ { d } } { n } .
$$

Since $K ^ { d } / n \le 1$ , the inequalities $x / 2 \leq \log ( 1 + x ) \leq x$ for $x \in [ 0 , 1 ]$ give the result.

If $K ^ { d } \geq n$ , then $s _ { \operatorname* { m a x } } = n$ and

$$
\frac { s _ { \mathrm { m a x } } } { n } \log \left( \frac { e K ^ { d } } { s _ { \mathrm { m a x } } } \right) = 1 + \log \left( \frac { K ^ { d } } { n } \right) .
$$

Writing $x : = K ^ { d } / n \geq 1$ , we have $1 + x \leq e x \leq ( 1 + x ) ^ { 2 }$ , and hence

$$
\log ( 1 + x ) \leq 1 + \log x \leq 2 \log ( 1 + x ) .
$$

Combining the two cases proves the claim.

## A.2 Properties of Uniform Forward Process

Lemma A.4. For every $t > 0$ and $\pmb { x } \in [ K ] ^ { d }$ 2

$$
P _ { t } ( { \pmb x } ) \geq B _ { 0 } ( t ) ^ { - 1 } .
$$

Moreover, for every $t > 0$ and x, $\pmb { y } , z \in [ K ] ^ { d } \mathrm { ~ } w i t h \mathrm { ~ } d _ { \mathrm { H a m } } ( \pmb { x } , \pmb { y } ) = 1$

$$
B _ { 1 } ( t ) ^ { - 1 } \leq \frac { P _ { t } ( y ) } { P _ { t } ( x ) } \leq B _ { 1 } ( t ) , \qquad B _ { 1 } ( t ) ^ { - 1 } \leq \frac { P _ { t } ( z , y ) } { P _ { t } ( z , x ) } \leq B _ { 1 } ( t ) .
$$

Proof. For one coordinate, $P _ { t } ^ { \mathrm { c o o r d } } ( a , b ) = e ^ { - t } \mathbf { 1 } \{ a = b \} + ( 1 - e ^ { - t } ) / K \geq ( 1 - e ^ { - t } ) / K$ . Thus, by coordinatewise independence, $P _ { t } ( z , x ) \ge ( ( 1 - e ^ { - t } ) / K ) ^ { d } = B _ { 0 } ( t ) ^ { - 1 }$ , and averaging over $z \sim P _ { 0 }$ gives $P _ { t } ( { \pmb x } ) \geq B _ { 0 } ( t ) ^ { - 1 }$

Now suppose $d _ { \mathrm { H a m } } ( \pmb { x } , \pmb { y } ) = 1$ , and let $j$ be the unique coordinate on which they differ. All other coordinate factors cancel, so

$$
\frac { P _ { t } ( z , y ) } { P _ { t } ( z , x ) } = \frac { P _ { t } ^ { \mathrm { c o o r d } } ( z ^ { j } , y ^ { j } ) } { P _ { t } ^ { \mathrm { c o o r d } } ( z ^ { j } , x ^ { j } ) } .
$$

Since the smallest and largest entries of $P _ { t } ^ { \mathrm { c o o r d } }$ are $( 1 - e ^ { - t } ) / K$ and $e ^ { - t } + ( 1 - e ^ { - t } ) / K$ , respectively, the last ratio lies in $\left[ B _ { 1 } ( t ) ^ { - 1 } , B _ { 1 } ( t ) \right]$

Finally,

$$
\frac { P _ { t } ( \pmb { y } ) } { P _ { t } ( \pmb { x } ) } = \sum _ { \pmb { z } \in [ K ] ^ { d } } \frac { P _ { 0 } ( \pmb { z } ) P _ { t } ( \pmb { z } , \pmb { x } ) } { P _ { t } ( \pmb { x } ) } \frac { P _ { t } ( \pmb { z } , \pmb { y } ) } { P _ { t } ( \pmb { z } , \pmb { x } ) } .
$$

The coefficients form a probability distribution over z, so the same bounds hold for $P _ { t } ( \pmb { y } ) / P _ { t } ( \pmb { x } )$ □

Lemma A.5 (Uniform mixing of the forward process). For every $t \geq 0$

$$
d _ { \mathrm { T V } } ( P _ { t } , \pi ) \leq 1 - ( 1 - e ^ { - t } ) ^ { d } \leq d e ^ { - t } .
$$

Proof. For one coordinate, $P _ { t } ^ { \mathrm { c o o r d } } ( a , \cdot ) = e ^ { - t } \delta _ { a } + ( 1 - e ^ { - t } ) \operatorname { U n i f } ( [ K ] )$ . Hence, with probability $( 1 - e ^ { - t } ) ^ { d }$ , all d coordinates are refreshed independently from the uniform distribution, regardless of the initial state. Therefore, for some distribution $Q _ { t }$

$$
P _ { t } = ( 1 - e ^ { - t } ) ^ { d } \pi + \left[ 1 - ( 1 - e ^ { - t } ) ^ { d } \right] Q _ { t } .
$$

It follows that $d _ { \mathrm { T V } } ( P _ { t } , \pi ) \leq 1 - ( 1 - e ^ { - t } ) ^ { d } \leq d e ^ { - t }$ , where the last inequality follows from $1 - ( 1 - u ) ^ { d } \leq$ du. □

Lemma A.6. The empirical distribution $\widehat { P } _ { 0 }$ satisfies

$$
\mathbb { E } _ { P _ { 0 } } \Big [ d _ { \mathrm { T V } } ( P _ { 0 } , \widehat { P } _ { 0 } ) \Big ] \leq \frac { 3 } { 2 } \sqrt { \frac { { \mathfrak { s } } _ { n } ( P _ { 0 } ) } { n } } .
$$

Proof. Let $\begin{array} { r } { N _ { \pmb { x } } : = \sum _ { i = 1 } ^ { n } \pmb { 1 } \{ X _ { 0 } ^ { ( i ) } = \pmb { x } \} } \end{array}$ and $H : = \{ \pmb { x } \in [ K ] ^ { d } : n P _ { 0 } ( \pmb { x } ) \geq 1 \}$ . For every $\pmb { x } \in [ K ] ^ { d }$ , the triangle inequality and Jensen's inequality give

$$
\mathbb { E } _ { P _ { 0 } } \left| \frac { N _ { \pmb { x } } } { n } - P _ { 0 } ( \pmb { x } ) \right| \leq \operatorname* { m i n } \left\{ 2 P _ { 0 } ( \pmb { x } ) , \sqrt { \frac { P _ { 0 } ( \pmb { x } ) } { n } } \right\} .
$$

Therefore,

$$
\mathbb { E } _ { P _ { 0 } } \Big [ d _ { \mathrm { T V } } ( P _ { 0 } , \widehat { P } _ { 0 } ) \Big ] \leq \sum _ { \pmb { x } \notin H } P _ { 0 } ( \pmb { x } ) + \frac { 1 } { 2 } \sum _ { \pmb { x } \in H } \sqrt { \frac { P _ { 0 } ( \pmb { x } ) } { n } } .
$$

The first term is at most $\mathfrak { s } _ { n } ( P _ { 0 } ) / n \leq \sqrt { \mathfrak { s } _ { n } ( P _ { 0 } ) / n }$ Since $| H | \leq { \mathfrak { s } } _ { n } ( P _ { 0 } )$ , the Cauchy-Schwarz inequality gives

$$
\sum _ { \pmb { x } \in H } \sqrt { \frac { P _ { 0 } ( \pmb { x } ) } { n } } \leq \sqrt { \frac { | H | } { n } \sum _ { \pmb { x } \in H } P _ { 0 } ( \pmb { x } ) } \leq \sqrt { \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } } .
$$

Combining the two bounds proves the claim.

## A.3 Auxiliary Results for KL Bounds

Lemmas A.7 to A.10 are used for proving Lemma A.11, which gives the KL divergence upper bound for the empirical distribution at time $1 / n , \widehat { P } _ { 1 / n }$

Lemma A.7 (Chernoff bounds for a binomial random variable). Let N be a binomial random variable. For every $\delta \in ( 0 , 1 )$ 2

$$
\mathbb { P } \{ N \leq ( 1 - \delta ) \mathbb { E } N \} \leq \exp \left( - \frac { \delta ^ { 2 } \mathbb { E } N } { 2 } \right) , \qquad \mathbb { P } \{ N \geq ( 1 + \delta ) \mathbb { E } N \} \leq \exp \left( - \frac { \delta ^ { 2 } \mathbb { E } N } { 2 + \delta } \right) .
$$

Proof. Write $N \sim \mathrm { B i n } ( M , p )$ , sO $\mathbb { E } N \ = \ M p$ . For every $\lambda \in \mathbb { R } , \mathbb { E } e ^ { \lambda N } = ( 1 - p + p e ^ { \lambda } ) ^ { M } \leq$ $\mathrm { e x p } \{ ( \mathbb { E } N ) ( e ^ { \lambda } - 1 ) \}$ . Applying Markov's inequality with $\lambda = \log ( 1 { + } \delta )$ and using $( 1 + \delta ) \log ( 1 + \delta ) - \delta \geq$ $\delta ^ { 2 } / ( 2 + \delta )$ gives the upper-tail bound. Applying the same argument to $e ^ { - \lambda N }$ with $\lambda = - \log ( 1 - \delta )$ and using $\delta + ( 1 - \delta ) \log ( 1 - \delta ) \geq \delta ^ { 2 } / 2$ gives the lower-tail bound. □

Lemma A.8. Let N be a binomial random variable. $I f \mathbb { E } N \geq 1$

$$
\mathbb { E } \left[ { \bf 1 } \{ N > 0 \} \log \frac { \mathbb { E } N } { N } \right] \stackrel { } { \sim } \frac { 1 } { \mathbb { E } N } .
$$

$I f 0 < \mathbb { E } N < 1$ , then $\mathbf { 1 } \{ N > 0 \} \log ( \mathbb { E } N / N ) \leq 0$

Proof. If $0 < \mathbb { E } N < 1$ , then $N > 0$ implies $N \geq 1 > \mathbb { E } N$ , proving the second claim. Suppose $\lambda : = \mathbb { E } N \geq 1$ , and let $A : = \{ \lambda / 2 \leq N \leq 3 \lambda / 2 \}$ and $Z : = ( N - \lambda ) / \lambda$ . On A, since $| Z | \le 1 / 2$ and $\cdot \log ( 1 + Z ) \leq - Z + Z ^ { 2 }$

$$
\mathbb { E } \left[ \mathbf { 1 } _ { A } \log \frac { \lambda } { N } \right] \leq - \mathbb { E } [ \mathbf { 1 } _ { A } Z ] + \mathbb { E } [ Z ^ { 2 } ] .
$$

Since $\mathbb { E } Z = 0 , \mathbb { E } Z ^ { 2 } \le 1 / \lambda$ , and Lemma A.7 gives $\mathbb { P } ( A ^ { c } ) \le 2 e ^ { - \lambda / 1 0 }$ , the Cauchy-Schwarz inequality yields

$$
- \mathbb { E } [ \mathbf { 1 } _ { A } Z ] = \mathbb { E } [ \mathbf { 1 } _ { A ^ { c } } Z ] \leq \mathbb { E } [ \mathbf { 1 } _ { A ^ { c } } \vert Z \vert ] \leq \sqrt { \frac { 2 } { \lambda } } e ^ { - \lambda / 2 0 } \lesssim \frac { 1 } { \lambda } .
$$

Hence $\mathbb { E } [ \mathbf { 1 } _ { A } \log ( \lambda / N ) ] \lesssim 1 / \lambda$

On Ac, the logarithm is positive only when $0 < N < \lambda / 2$ , where $\log ( \lambda / N ) \leq \log \lambda$ . Therefore, Lemma A.7 gives

$$
\mathbb { E } \left[ { \bf 1 } _ { A ^ { c } } { \bf 1 } \{ N > 0 \} \log \frac { \lambda } { N } \right] \leq \log \lambda e ^ { - \lambda / 8 } \lesssim \frac { 1 } { \lambda } .
$$

Combining the two bounds proves the claim.

Lemma A.9 (Weighted Caro-Wei bound). Let $G = ( V , E )$ be a finite simple graph with positive vertex weights $( w _ { v } ) _ { v \in V }$ , and let $N [ v ] : = \{ v \} \cup \{ u \in V : \{ u , v \} \in E \}$ denote the closed neighborhood $o f v$ . Let

$$
\alpha ( G ) : = \operatorname* { m a x } \{ | I | : I \subseteq V , \ \{ u , v \} \not \in E \ f o r \ a l l \ d i s t i n c t \ u , v \in I \}
$$

denote the independence number of G. Then

$$
\sum _ { v \in V } { \frac { w _ { v } } { \sum _ { u \in N [ v ] } w _ { u } } } \leq \alpha ( G ) .
$$

Proof. Let $( T _ { v } ) _ { v \in V }$ be independent exponential random variables with rates $( w _ { v } ) _ { v \in V }$ , and define $\begin{array} { r } { I : = \{ v \in V : T _ { v } = \operatorname* { m i n } _ { u \in N [ v ] } T _ { u } \} } \end{array}$ . Since the $T _ { v } \mathrm { { s } }$ are continuous, the minima are unique almost surely. If $\{ u , v \} \in E$ and $u , v \in I ,$ then $T _ { u } < T _ { v }$ and $T _ { v } < T _ { u }$ , a contradiction, so I is independent almost surely. By the exponential race identity,

$$
\mathbb { P } ( v \in I ) = \frac { w _ { v } } { \sum _ { u \in N [ v ] } w _ { u } } .
$$

Therefore,

$$
\sum _ { v \in V } { \frac { w _ { v } } { \sum _ { u \in N [ v ] } w _ { u } } } = \mathbb { E } | I | \leq \alpha ( G ) .
$$

Lemma A.10. Let $X _ { 0 } ^ { ( 1 ) } , \ldots , X _ { 0 } ^ { ( n ) } ,$ Y be i.i.d. from $P _ { 0 }$ , and deine

$$
R ( Y ) : = \operatorname* { m i n } _ { 1 \leq i \leq n } d _ { \mathrm { H a m } } \left( Y , X _ { 0 } ^ { ( i ) } \right) .
$$

Then

$$
\mathbb { E } _ { P _ { 0 } } R ( Y ) \lesssim \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } \left\{ 1 + \frac { \log \left( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \right) } { \log K } \right\} .
$$

Proof. For $r = 0 , \ldots , d ,$ let $B _ { r } ( { \pmb y } ) : = \{ { \pmb z } \in [ K ] ^ { d } : d _ { \mathrm { H a m } } ( { \pmb z } , { \pmb y } ) \leq r \}$ . For $r = 0 , \ldots , d - 1$ , independence gives

$$
\mathbb { P } _ { P _ { 0 } } \{ R ( Y ) > r \} \le \sum _ { y } P _ { 0 } ( \pmb { y } ) e ^ { - n P _ { 0 } ( B _ { r } ( \pmb { y } ) ) } .
$$

Since $P _ { 0 } ( B _ { r } ( { \pmb y } ) ) \ge P _ { 0 } ( { \pmb y } )$ and $u e ^ { - u } \leq \operatorname* { m i n } \{ u , 1 \}$

$$
\sum _ { y } P _ { 0 } ( { \pmb y } ) e ^ { - n P _ { 0 } ( B _ { r } ( { \pmb y } ) ) } \le \frac { { \mathfrak { s } } _ { n } ( P _ { 0 } ) } { n } .
$$

For a second bound, let $G _ { r }$ be the graph on $\operatorname { s u p p } ( P _ { 0 } )$ in which two distinct states are adjacent if their Hamming distance is at most r. Since the closed neighborhood of y has total weight $P _ { 0 } ( B _ { r } ( \pmb { y } ) )$ 2 Lemma A.9 and $u e ^ { - u } \leq 1$ give

$$
\sum _ { y } P _ { 0 } ( { \pmb y } ) e ^ { - n P _ { 0 } ( B _ { r } ( \pmb y ) ) } \le \frac { \alpha ( G _ { r } ) } { n } .
$$

If A is independent in $G _ { r }$ , the balls $B _ { \lfloor r / 2 \rfloor } ( { \pmb x } ) , { \pmb x } \in A$ , are pairwise disjoint, and each contains at least $K ^ { \lfloor r / 2 \rfloor }$ states. Hence $\alpha ( G _ { r } ) \leq K ^ { d - \lfloor r / 2 \rfloor }$ , and therefore

$$
\mathbb { P } _ { P _ { 0 } } \{ R ( Y ) > r \} \le \frac { 1 } { n } \operatorname* { m i n } \left\{ \mathfrak { s } _ { n } ( P _ { 0 } ) , K ^ { d - \lfloor r / 2 \rfloor } \right\} .
$$

Summing the tail probabilities yields

$$
\mathbb { E } _ { P _ { 0 } } R ( Y ) \le \frac { 2 } { n } \sum _ { \ell = 0 } ^ { d } \operatorname* { m i n } \{ \mathfrak { s } _ { n } ( P _ { 0 } ) , K ^ { d - \ell } \} .
$$

There are at most $1 + \log ( K ^ { d } / { \mathfrak { s } } _ { n } ( P _ { 0 } ) ) /$ log K indices for which $K ^ { d - \ell } \ge \mathfrak { s } _ { n } ( P _ { 0 } )$ , and their total contribution is at most

$$
\mathfrak { s } _ { n } ( P _ { 0 } ) \left\{ 1 + \frac { \log ( K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) ) } { \log K } \right\} .
$$

The remaining terms form a geometric series with largest term smaller than ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ and ratio $K ^ { - 1 } \le 1 / 2$ , so their sum is at most $2 \mathfrak { s } _ { n } ( P _ { 0 } )$ . Therefore,

$$
\sum _ { \ell = 0 } ^ { d } \operatorname* { m i n } \{ \mathfrak { s } _ { n } ( P _ { 0 } ) , K ^ { d - \ell } \} \lesssim \mathfrak { s } _ { n } ( P _ { 0 } ) \left\{ 1 + \frac { \log \left( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \right) } { \log K } \right\} ,
$$

which proves the claim.

Lemma A.11. The empirical distribution at time $1 / n , \widehat { P } _ { 1 / n }$ , satisfies

$$
\mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } \left( P _ { 0 } \| \widehat { P } _ { 1 / n } \right) \lesssim \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) \log \left( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \right) \log n } { n } .
$$

Proof. For $y \in [ K ] ^ { d } .$ let $\begin{array} { r } { N _ { \pmb { y } } : = \sum _ { i = 1 } ^ { n } \pmb { 1 } \{ X _ { 0 } ^ { ( i ) } = \pmb { y } \} } \end{array}$ and $\begin{array} { r } { R _ { \pmb { y } } : = \operatorname* { m i n } _ { 1 \leq i \leq n } d _ { \mathrm { H a m } } ( \pmb { y } , X _ { 0 } ^ { ( i ) } ) } \end{array}$ . By coordinatewise independence, choosing an observation nearest to $^ { y , }$ and using the $N _ { y }$ observations equal to y when $N _ { y } > 0$ give

$$
\widehat { P } _ { 1 / n } ( y ) \geq \frac { e ^ { - d / n } } { n } \left( \frac { 1 - e ^ { - 1 / n } } { K } \right) ^ { R _ { y } } , \qquad N _ { y } > 0 \implies \widehat { P } _ { 1 / n } ( y ) \geq \frac { N _ { y } } { n } e ^ { - d / n } .
$$

We decompose the KL divergence according to whether y appears in the sample. For the unobserved states, the first bound and $1 - e ^ { - 1 / n } \geq 1 / ( 2 n )$ give

$$
\sum _ { y : N _ { y } = 0 } P _ { 0 } ( y ) \log \frac { P _ { 0 } ( y ) } { \widehat { P } _ { 1 / n } ( y ) } \leq \sum _ { y : N _ { y } = 0 } P _ { 0 } ( y ) [ \log ( n P _ { 0 } ( y ) ) ] _ { + } + \frac { d } { n } + \log ( 2 K n ) \sum _ { y : N _ { y } = 0 } P _ { 0 } ( y ) R _ { y } .
$$

Since $\mathbb { P } _ { P _ { 0 } } \{ N _ { \pmb { y } } = 0 \} = ( 1 - P _ { 0 } ( \pmb { y } ) ) ^ { n }$

$$
\mathbb { E } _ { P _ { 0 } } \sum _ { y : N _ { y } = 0 } P _ { 0 } ( y ) [ \log ( n P _ { 0 } ( y ) ) ] _ { + } \le \frac { 1 } { n } \sum _ { y : n P _ { 0 } ( y ) \ge 1 } n P _ { 0 } ( y ) e ^ { - n P _ { 0 } ( y ) } \log ( n P _ { 0 } ( y ) ) \lesssim \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } .
$$

Moreover, Lemma A.10 gives

$$
\mathbb { E } _ { P _ { 0 } } \sum _ { y } P _ { 0 } ( y ) R _ { y } \lesssim \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } \left\{ 1 + \frac { \log \left( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \right) } { \log K } \right\} .
$$

For the observed states, the second lower bound gives

$$
\sum _ { y : N _ { y } > 0 } P _ { 0 } ( y ) \log \frac { P _ { 0 } ( y ) } { \widehat { P } _ { 1 / n } ( y ) } \leq \frac { d } { n } + \sum _ { y : N _ { y } > 0 } P _ { 0 } ( y ) \log \frac { n P _ { 0 } ( y ) } { N _ { y } } .
$$

If $n P _ { 0 } ( y ) < 1$ , the corresponding summand is nonpositive. If $n P _ { 0 } ( \pmb { y } ) \ge 1$ , Lemma A.8 gives its expectation at most $C / n$ . Since at most ${ \mathfrak { s } } _ { n } ( P _ { 0 } )$ states satisfy $n P _ { 0 } ( { \pmb y } ) \ge 1$ , the expected observed contribution is $\lesssim ( d + \mathfrak { s } _ { n } ( P _ { 0 } ) ) / n$

Combining the two parts,

$$
\mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } ( P _ { 0 } \| \hat { P } _ { 1 / n } ) \lesssim \frac { d + \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } + \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) \log ( K n ) } { n } \left\{ 1 + \frac { \log \left( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \right) } { \log K } \right\} .
$$

Since $u \mapsto u \log ( e K ^ { d } / u )$ is increasing on $[ 1 , K ^ { d } ]$ , we have $d \lesssim \mathfrak { s } _ { n } ( P _ { 0 } ) \log ( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) )$ . Also, using ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \leq n$ and $K , n \geq 2$

$$
\log ( K n ) \left\{ 1 + \frac { \log \left( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \right) } { \log K } \right\} \lesssim \log \left( \frac { e K ^ { d } } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) \log n .
$$

The claimed bound follows.

Since KL divergence does not satisfy a triangle inequality, we use Lemma A.13 instead. Its proof relies on the entropy variational inequality in Lemma A.12.

Lemma A.12 (Entropy variational inequality). Let $\mu$ and ν be probability distributions on a inite state space. Then, for every real-valued function $f$

$$
\begin{array} { r } { \mathbb { E } _ { \mu } f \leq D _ { \mathrm { K L } } ( \mu \| \nu ) + \log \mathbb { E } _ { \nu } e ^ { f } . } \end{array}
$$

Proof. The claim is immediate if $D _ { \mathrm { K L } } ( \mu \| \nu ) = \infty$ . Otherwise, define $\nu _ { f } ( \pmb { x } ) : = \nu ( \pmb { x } ) e ^ { f ( \pmb { x } ) } / \mathbb { E } _ { \nu } e ^ { f }$ . Then

$$
0 \leq D _ { \mathrm { K L } } ( \mu \| \nu _ { f } ) = D _ { \mathrm { K L } } ( \mu \| \nu ) - \mathbb { E } _ { \mu } f + \log \mathbb { E } _ { \nu } e ^ { f } ,
$$

which proves the claim.

Lemma A.13. Let $\mu , \nu , \xi$ be probability distributions on a finite state space. Suppose that, for some $B \geq 1$ and $\eta < \infty$

$$
\nu ( { \pmb x } ) \geq B ^ { - 1 } \quad f o r e v e r y { \pmb x } , \qquad D _ { \mathrm { K L } } ( \nu \| \xi ) \leq \frac { \eta } { B } .
$$

Then there exists $C _ { \eta } < \infty$ , depending only on η, such that

$$
D _ { \mathrm { K L } } ( \mu \| \boldsymbol { \xi } ) \leq 2 D _ { \mathrm { K L } } ( \mu \| \nu ) + C _ { \eta } D _ { \mathrm { K L } } ( \nu \| \boldsymbol { \xi } ) .
$$

Proof. Let $h ( u ) : = u - 1 - \log u$ and $u _ { x } : = \xi ( x ) / \nu ( x )$ . Since ν has full support and $D _ { \mathrm { K L } } ( \nu \| \xi ) < \infty$

we also have $\xi ( { \pmb x } ) > 0$ for every x. Moreover,

$$
h ( u _ { \pmb { x } } ) = \frac { D _ { \mathrm { B r } } ( \nu ( \pmb { x } ) , \pmb { \xi } ( \pmb { x } ) ) } { \nu ( \pmb { x } ) } \leq B D _ { \mathrm { K L } } ( \nu \| \pmb { \xi } ) \leq \eta .
$$

The set $\{ u > 0 : h ( u ) \leq \eta \}$ is compact, and $u \mapsto ( 1 - u ) ^ { 2 } / ( u h ( u ) )$ extends continuously to $u = 1$ with value 2. Hence this function is bounded on the set by a constant depending only on $\eta ,$ and therefore

$$
\chi ^ { 2 } ( \nu , \xi ) : = \sum _ { x } \frac { ( \nu ( x ) - \xi ( \pmb { x } ) ) ^ { 2 } } { \xi ( \pmb { x } ) } \leq C _ { \eta } D _ { \mathrm { K L } } ( \nu \| \xi ) .
$$

Applying Lemma A.12 with $f ( \pmb { x } ) : = \log ( \nu ( \pmb { x } ) / \xi ( \pmb { x } ) )$ gives

$$
D _ { \mathrm { K L } } ( \mu \| \boldsymbol { \xi } ) \le 2 D _ { \mathrm { K L } } ( \mu \| \nu ) + \log \bigl ( 1 + \chi ^ { 2 } ( \nu , \boldsymbol { \xi } ) \bigr ) \le 2 D _ { \mathrm { K L } } ( \mu \| \nu ) + C _ { \eta } D _ { \mathrm { K L } } ( \nu \| \boldsymbol { \xi } ) .
$$

## A.4 Score-Entropy Identities and Sampler Guarantees

The following lemma is the discrete diffusion score-entropy loss analogue of the standard denoising identity.

Lemma A.14. Let µ be a distribution on $[ K ] ^ { d }$ , and deine $\mu _ { t } : = \mu P _ { t }$ and $\sigma _ { t } ^ { \star , \mu } ( { \pmb y } , { \pmb x } ) : = \mu _ { t } ( { \pmb y } ) / \mu _ { t } ( { \pmb x } )$ For every $t > 0$ , there exists a finite quantity $C _ { t } ( \mu )$ , independent of the candidate score σ, such that

$$
\begin{array} { r } { \mathcal { L } _ { t } ^ { \mathrm { D S E } } ( \sigma ; \mu ) = \mathcal { L } _ { t } ^ { \mathrm { S E } } ( \sigma ; \mu ) + C _ { t } ( \mu ) . } \end{array}
$$

Consequently, for any positive score functions σ and $\widetilde { \sigma } _ { ; }$

$$
\begin{array} { r } { \mathcal { L } _ { \mathcal { T } } ^ { \mathrm { D S E } } ( \sigma ; \mu ) - \mathcal { L } _ { \mathcal { T } } ^ { \mathrm { D S E } } ( \widetilde { \sigma } ; \mu ) = \mathcal { L } _ { \mathcal { T } } ^ { \mathrm { S E } } ( \sigma ; \mu ) - \mathcal { L } _ { \mathcal { T } } ^ { \mathrm { S E } } ( \widetilde { \sigma } ; \mu ) . } \end{array}
$$

Proof. Fix $t > 0$ , let $X _ { 0 } \sim \mu ,$ and conditionally on $X _ { 0 }$ let $X _ { t } \sim P _ { t } ( X _ { 0 } , \cdot )$ . For fixed $\mathbf { \nabla } _ { \mathbf { x } , \mathbf { y } }$ the conditional score satisfies

$$
\sigma _ { t | 0 } ^ { \star } ( { \pmb y } , { \pmb x } \mid X _ { 0 } ) = \frac { { \pmb P } _ { t } ( X _ { 0 } , { \pmb y } ) } { { \pmb P } _ { t } ( X _ { 0 } , { \pmb x } ) } .
$$

Conditioning on $X _ { t } = \pmb { x }$ and using Bayes' rule,

$$
\mathbb { P } ( X _ { 0 } = z \mid X _ { t } = \pmb { x } ) = \frac { \mu ( z ) \pmb { P } _ { t } ( z , \pmb { x } ) } { \mu _ { t } ( \pmb { x } ) } .
$$

Therefore,

$$
\begin{array} { l } { \displaystyle \mathbb { E } \Big [ \sigma _ { t \mid 0 } ^ { \star } ( \pmb { y } , \pmb { x } \mid X _ { 0 } ) \mid X _ { t } = \pmb { x } \Big ] = \sum _ { z } \frac { P _ { t } ( z , \pmb { y } ) } { P _ { t } ( z , \pmb { x } ) } \frac { \mu ( z ) P _ { t } ( z , \pmb { x } ) } { \mu _ { t } ( \pmb { x } ) } } \\ { \displaystyle = \frac { \mu _ { t } ( \pmb { y } ) } { \mu _ { t } ( \pmb { x } ) } = \sigma _ { t } ^ { \star , \mu } ( \pmb { y } , \pmb { x } ) . } \end{array}
$$

Conditioning the DSE loss on $X _ { t }$ and expanding the Bregman divergence in the SE loss therefore gives

$$
\begin{array} { r } { \mathcal { L } _ { t } ^ { \mathrm { D S E } } ( \sigma ; \mu ) = \mathcal { L } _ { t } ^ { \mathrm { S E } } ( \sigma ; \mu ) + C _ { t } ( \mu ) , } \end{array}
$$

where

$$
C _ { t } ( \mu ) : = \mathbb { E } _ { X _ { t } \sim \mu _ { t } } \sum _ { y : Q ( y , X _ { t } ) > 0 } Q ( y , X _ { t } ) \sigma _ { t } ^ { \star , \mu } ( y , X _ { t } ) \big ( 1 - \log \sigma _ { t } ^ { \star , \mu } ( y , X _ { t } ) \big ) .
$$

Since $t > 0 ,$ all forward probabilities are positive, so $C _ { t } ( \mu )$ is finite. Applying the identity to both $\sigma$ and ${ \widetilde { \sigma } } .$ subtracting to cancel $C _ { t } ( \mu )$ , and summing over the grid with weights $t _ { k } - t _ { k - 1 }$ proves the second statement. □

We next record Dmitriev et al. (2026, Theorem 1) in our notation. As noted there, the same guarantee applies under early stopping.

Theorem A.15 (τ-leaping sampler guarantee). For a fixed grid $\mathcal { T } = \{ \delta = t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot < t _ { N } = T \}$ 2 let $\Delta : = \operatorname* { m a x } _ { 1 \le k \le N } ( t _ { k } - t _ { k - 1 } )$ and suppose $\Delta \lesssim 1$ . Then the τ-leaping sampler initialized from π satisfies

$$
D _ { \mathrm { K L } } \left( P _ { \delta } \| Q _ { \delta } ^ { \tau , \mathcal { T } , \sigma } \right) \lesssim \mathcal { L } _ { \mathcal { T } } ^ { \mathrm { S E } } ( \sigma ; P _ { 0 } ) + e ^ { - T } d \log K + \Delta d \log \left( \frac { K } { \Delta } \right) .
$$

## A.5 ReLU Network Constructions

We first record two ReLU network constructions following Oko et al. (2023, Lemmas F.6–F.7). For multiplication, we use the bounded-input construction in the proof of Lemma F.6, restricted to $[ 0 , 1 ] ^ { d }$ and clipped to [0, 1]. We omit the proofs.

Lemma A.16 (ReLU multiplication network). For every integer d $\geq 2$ and $\epsilon \in ( 0 , 1 )$ , there exists a ReLU network Mul $\bar { \boldsymbol { \cdot } } _ { d , \epsilon } : [ 0 , 1 ] ^ { d } \to [ 0 , 1 ]$ such that

$$
\operatorname* { s u p } _ { u \in [ 0 , 1 ] ^ { d } } \left| \mathrm { M u l t } _ { d , \epsilon } ( u ) - \prod _ { j = 1 } ^ { d } u _ { j } \right| \leq \epsilon .
$$

Moreover, the network can be chosen with

$$
\begin{array} { r } { L _ { \operatorname* { m u t } } \lesssim \log d \big ( \log \epsilon ^ { - 1 } + \log d \big ) , \qquad W _ { \operatorname* { m u t } } \lesssim d , \qquad S _ { \operatorname* { m u t } } \lesssim d \big ( \log \epsilon ^ { - 1 } + \log d \big ) , \qquad B _ { \operatorname* { m u t } } \lesssim 1 . } \end{array}
$$

Lemma A.17 (ReLU reciprocal network). For every $\epsilon \in \mathsf { \Gamma } ( 0 , 1 )$ , there exists a ReLU network $\operatorname { R e c } _ { \epsilon } : \mathbb { R }  \mathbb { R }$ such that, for every $u \in [ \epsilon , \epsilon ^ { - 1 } ]$ and $u ^ { \prime } \in \mathbb { R }$

$$
\left| \mathrm { R e c } _ { \epsilon } ( u ^ { \prime } ) - \frac { 1 } { u } \right| \le \epsilon + \frac { | u ^ { \prime } - u | } { \epsilon ^ { 2 } } .
$$

Moreover, the network can be chosen with

$$
L _ { \mathrm { r e c } } \lesssim \log ^ { 2 } \epsilon ^ { - 1 } , \qquad W _ { \mathrm { r e c } } \lesssim \log ^ { 3 } \epsilon ^ { - 1 } , \qquad S _ { \mathrm { r e c } } \lesssim \log ^ { 4 } \epsilon ^ { - 1 } , \qquad B _ { \mathrm { r e c } } \lesssim \epsilon ^ { - 2 } .
$$

Using Lemmas A.16 and A.17, we construct a ReLU network that approximates the discrete score simultaneously over a fixed time grid.

Lemma A.18 (ReLU approximation of discrete scores on a fixed grid). Let $\mathcal { T } = \{ \delta = t _ { 0 } < t _ { 1 } <$ $\cdots < t _ { N } = T \}$ be a fixed grid, and let $\begin{array} { r } { \mu = \sum _ { v = 1 } ^ { D } w _ { v } \delta _ { z _ { v } } } \end{array}$ with $w _ { v } > 0$ and $\textstyle \sum _ { v = 1 } ^ { D } w _ { v } = 1$ . Write $\mu _ { t } : = \mu P _ { t }$ . For every $\varepsilon \in ( 0 , 1 )$ , there exists a ReLU network $G _ { \varepsilon } : \mathbb { R } ^ { 1 + 2 d K }  \mathbb { R }$ with range contained in $\left[ B _ { 1 } ( t _ { 1 } ) ^ { - 1 } , B _ { 1 } ( t _ { 1 } ) \right]$ such that

$$
\operatorname* { m a x } _ { 1 \leq k \leq N } \operatorname* { m a x } _ { d _ { \mathrm { H a m } } ( { \pmb x } , { \pmb y } ) = 1 } \left| G _ { \varepsilon } \left( \frac k N , e ( { \pmb y } ) , e ( { \pmb x } ) \right) - \frac { \mu _ { t _ { k } } ( { \pmb y } ) } { \mu _ { t _ { k } } ( { \pmb x } ) } \right| \leq \varepsilon .
$$

The network may be chosen with

$$
L \lesssim \log ^ { 2 } \left( \frac { B _ { 0 } ( t _ { 1 } ) } { \varepsilon } \right) , \qquad W \lesssim d ( K N + D ) + \log ^ { 3 } \left( \frac { B _ { 0 } ( t _ { 1 } ) } { \varepsilon } \right) ,
$$

$$
S \lesssim d K ^ { 2 } N + d D \log \left( \frac { B _ { 0 } ( t _ { 1 } ) } { \varepsilon } \right) + \log ^ { 4 } \left( \frac { B _ { 0 } ( t _ { 1 } ) } { \varepsilon } \right) , \qquad B \lesssim N + B _ { 0 } ( t _ { 1 } ) ^ { 2 } + \varepsilon ^ { - 2 } .
$$

Proof. We first approximate the two forward marginals, then approximate the reciprocal of the denominator, and finally multiply. For $k \in [ N ]$ , define

$$
J _ { k } ( r ) : = \rho \left( 1 - N \rho \left( r - \frac { k } { N } \right) - N \rho \left( \frac { k } { N } - r \right) \right) ,
$$

so that $J _ { k } ( \ell / N ) = { \bf 1 } \{ k = \ell \}$ . For $i \in [ d ]$ and $a \in [ K ]$ , define $I _ { k , i , a } ^ { x } ( r , e ( \pmb { x } ) ) : = \rho ( J _ { k } ( r ) + ( e ( \pmb { x } ) ) _ { i , a } - 1 )$ and define $I _ { k , i , a } ^ { y }$ analogously. At encoded inputs, $I _ { k , i , a } ^ { x } ( \ell / N , e ( x ) ) = \mathbf { 1 } \{ k = \ell , x ^ { i } = a \}$ . For $i \in [ d ]$ and $b \in [ K ]$ , define

$$
G _ { i , b } ^ { x } : = \sum _ { k = 1 } ^ { N } \sum _ { a = 1 } ^ { K } P _ { t _ { k } } ^ { \mathrm { c o o r d } } ( b , a ) I _ { k , i , a } ^ { x } ,
$$

so that $G _ { i , b } ^ { x } ( k / N , e ( x ) ) = P _ { t _ { k } } ^ { \mathrm { c o o r d } } ( b , x ^ { i } )$ , and define $G _ { i , b } ^ { y }$ analogously.

Set η := min $\{ B _ { 0 } ( t _ { 1 } ) ^ { - 1 } , \varepsilon / 1 6 \}$ and $\varepsilon _ { 1 } : = \varepsilon \eta ^ { 2 } / 1 6$ . For each $v \in [ D ]$ , Lemma A.16 applied to $G _ { 1 , z _ { v } ^ { 1 } } ^ { \pmb { x } } , \ldots , G _ { d , z _ { v } ^ { d } } ^ { \pmb { x } }$ gives an ε1-approximation to $P _ { t _ { k } } ( z _ { v } , \pmb { x } )$ at every encoded grid input. Taking the weighted sum over v produces networks $F _ { x }$ and $F _ { y }$ satisfying

$$
| F _ { \pmb { x } } - \mu _ { t _ { k } } ( \pmb { x } ) | \le \varepsilon _ { 1 } , \qquad | F _ { \pmb { y } } - \mu _ { t _ { k } } ( \pmb { y } ) | \le \varepsilon _ { 1 }
$$

at every encoded input. Clip $F _ { x }$ to $[ B _ { 0 } ( t _ { 1 } ) ^ { - 1 } , 1 ]$ , which cannot increase its error by Lemma A.4. Since $\eta \le \mu _ { t _ { k } } ( \pmb { x } ) \le 1 \le \eta ^ { - 1 }$ , Lemma A.17, followed by clipping to $[ 1 , B _ { 0 } ( t _ { 1 } ) ]$ , gives a network $R _ { x }$ such that

$$
\big | R _ { x } - \mu _ { t _ { k } } ( \pmb { x } ) ^ { - 1 } \big | \leq \eta + \frac { \varepsilon _ { 1 } } { \eta ^ { 2 } } \leq \frac { \varepsilon } { 8 } .
$$

Next, clip $F _ { y }$ to $[ 0 , 1 ]$ , apply $\mathrm { M u l t _ { 2 , \varepsilon / ( 4 B _ { 0 } ( t _ { 1 } ) ) } }$ to $F _ { y }$ and $R _ { x } / B _ { 0 } ( t _ { 1 } )$ , and multiply the output by $B _ { 0 } ( t _ { 1 } )$ . The resulting network $\widetilde { G } _ { \varepsilon }$ satisfies

$$
\left| \widetilde G _ { \varepsilon } - \frac { \mu _ { t _ { k } } ( y ) } { \mu _ { t _ { k } } ( x ) } \right| \le \frac { \varepsilon } { 4 } + B _ { 0 } ( t _ { 1 } ) \varepsilon _ { 1 } + \frac { \varepsilon } { 8 } < \varepsilon ,
$$

where $B _ { 0 } ( t _ { 1 } ) \varepsilon _ { 1 } \leq \varepsilon / 1 6$ because $\eta \leq B _ { 0 } ( t _ { 1 } ) ^ { - 1 }$ . Finally, define

$$
G _ { \varepsilon } : = \mathrm { c l i p } _ { [ B _ { 1 } ( t _ { 1 } ) ^ { - 1 } , B _ { 1 } ( t _ { 1 } ) ] } ( \widetilde { G } _ { \varepsilon } ) .
$$

Since $B _ { 1 }$ is decreasing and $t _ { k } \geq t _ { 1 }$ , Lemma A.4 shows that the true score lies in this interval, so the final clipping cannot increase the error.

It remains to collect the architecture bounds. The selector and transition layers have constant depth, width O(dKN), sparsity $O ( d K ^ { 2 } N )$ , and weights of order N. The 2D multiplication networks contribute width $O ( d D )$ and sparsity $O ( d D ( \log \varepsilon _ { 1 } ^ { - 1 } + \log d ) )$ , while the reciprocal contributes depth $O ( \log ^ { 2 } { \eta ^ { - 1 } } )$ , width $O ( \log ^ { 3 } { \eta ^ { - 1 } } )$ , and sparsity $O ( \log ^ { 4 } { \eta ^ { - 1 } } )$ . Since $B _ { 0 } ( t _ { 1 } ) \geq K ^ { d } , \eta ^ { - 1 } \leq 1 6 B _ { 0 } ( t _ { 1 } ) / \varepsilon$ and log $\varepsilon _ { 1 } ^ { - 1 } \lesssim \log ( B _ { 0 } ( t _ { 1 } ) / \varepsilon )$ , these contributions give the stated bounds on L, W, S. Finally, the largest weights arise from the time selectors and reciprocal network, and $\eta ^ { - 2 } \lesssim B _ { 0 } ( t _ { 1 } ) ^ { 2 } + \varepsilon ^ { - 2 }$ yielding

$$
B \lesssim N + B _ { 0 } ( t _ { 1 } ) ^ { 2 } + \varepsilon ^ { - 2 } .
$$

## B Proofs for TV Results

## B.1 Upper Bound: Proofs of Theorem 3 and Corollary 4

Proof of Theorem 3. Fix a realization of the observations, so that $\widehat { P } _ { 0 }$ is supported on $D _ { n }$ points. Recall that $M = B _ { 1 } ( t _ { 1 } )$ , and set

$$
\begin{array} { r } { \varepsilon _ { \mathrm { T V } } : = \sqrt { \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { d n M T } } . } \end{array}
$$

Since $\mathfrak { s } _ { n } ( P _ { 0 } ) \leq n , d \geq 2 , M \geq 1$ , and $T \geq 1$ , we have $ { \varepsilon } _ { \mathrm { T V } } \in ( 0 , 1 )$ . Applying Lemma A.18 to $\widehat { P } _ { 0 }$ gives a score $\sigma ^ { \dagger }$ such that

$$
\operatorname* { m a x } _ { 1 \leq k \leq N } \operatorname* { m a x } _ { d _ { \mathrm { H a m } } ( { \pmb x } , { \pmb y } ) = 1 } \left| \sigma _ { t _ { k } } ^ { \dagger } ( { \pmb y } , { \pmb x } ) - \widehat \sigma _ { t _ { k } } ^ { \star } ( { \pmb y } , { \pmb x } ) \right| \leq \varepsilon _ { \mathrm { T V } } ,
$$

where $\widehat { \sigma } _ { t _ { k } } ^ { \star } ( \pmb { y } , \pmb { x } ) : = \widehat { P } _ { t _ { k } } ( \pmb { y } ) / \widehat { P } _ { t _ { k } } ( \pmb { x } )$

Both $\sigma ^ { \dagger }$ and $\widehat { \sigma } ^ { \star }$ take values in $[ M ^ { - 1 } , M ]$ . Since $f ^ { \prime \prime } ( u ) = 1 / u$ , Taylor's theorem gives $D _ { \mathrm { B r } } ( u , v ) \leq$ $M ( u - v ) ^ { 2 } / 2$ for $u , v \in [ M ^ { - 1 } , M ]$ . Moreover, $\begin{array} { r } { \sum _ { y : Q ( y , x ) > 0 } Q ( y , x ) = d ( K - 1 ) / K \le d , } \end{array}$ and hence

$$
\mathcal { L } _ { \mathcal { T } } ^ { \mathrm { S E } } ( \sigma ^ { \dagger } ; \widehat { P } _ { 0 } ) \leq \frac { d M \varepsilon _ { \mathrm { T V } } ^ { 2 } } { 2 } \sum _ { k = 1 } ^ { N } ( t _ { k } - t _ { k - 1 } ) \leq \frac { d M T \varepsilon _ { \mathrm { T V } } ^ { 2 } } { 2 } = \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { 2 n } .
$$

By Lemma A.14 and the εopt-optimality of $\widehat { \sigma }$

$$
\mathcal { L } _ { 7 } ^ { \mathrm { S E } } ( \widehat { \sigma } ; \widehat { P } _ { 0 } ) \leq \mathcal { L } _ { 7 } ^ { \mathrm { S E } } ( \sigma ^ { \dagger } ; \widehat { P } _ { 0 } ) + \varepsilon _ { \mathrm { o p t } } \lesssim \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } .
$$

Applying Assumption 2 conditionally on the observations and using the assumed sampler remainder gives

$$
D _ { \mathrm { K L } } \left( \widehat { P } _ { \delta } \| Q _ { \delta } ^ { \cal A } \right) \lesssim \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } .
$$

Thus, by Pinsker's inequality, $d _ { \mathrm { T V } } ( \widehat { P } _ { \delta } , Q _ { \delta } ^ { \cal A } ) \lesssim \sqrt { \mathfrak { s } _ { n } ( P _ { 0 } ) / n }$

By coupling the forward process initialized from $ { \widehat { P } } _ { 0 }$ with its initial state,

$$
d _ { \mathrm { T V } } ( \widehat { P } _ { 0 } , \widehat { P } _ { \delta } ) \leq 1 - \exp \left( - \frac { d ( K - 1 ) } { K } \delta \right) \leq d \delta .
$$

Therefore, the triangle inequality, Lemma A.6, and the assumption on δ yield

$$
\mathbb { E } _ { P _ { 0 } } d _ { \mathrm { T V } } \left( P _ { 0 } , Q _ { \delta } ^ { \mathcal { A } } \right) \lesssim \sqrt { \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } } + d \delta \lesssim \sqrt { \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } } .
$$

It remains to verify the architecture bounds. By Lemma A.18, it suffices to control $B _ { 0 } ( t _ { 1 } ) / \varepsilon _ { \mathrm { T V } }$ Since $B _ { 0 } ( t _ { 1 } ) = ( K / ( 1 - e ^ { - t _ { 1 } } ) ) ^ { d }$ and $M = B _ { 1 } ( t _ { 1 } ) \leq K / ( 1 - e ^ { - t _ { 1 } } )$

$$
\log \left( \frac { B _ { 0 } ( t _ { 1 } ) } { \varepsilon _ { \mathrm { T V } } } \right) \lesssim d \log \left( \frac { K } { 1 - e ^ { - t _ { 1 } } } \right) + \log \left( \frac { n T } { \varepsilon _ { n } ( P _ { 0 } ) } \right) ,
$$

where we used $K / ( 1 - e ^ { - t _ { 1 } } ) \geq 2 , { \mathfrak { s } } _ { n } ( P _ { 0 } ) \leq n ,$ and $T \geq 1$ . Substituting this bound into Lemma A.18 gives the stated bounds on $L , W , S$ Finally,

$$
B _ { 0 } ( t _ { 1 } ) ^ { 2 } = \bigg ( \frac { K } { 1 - e ^ { - t _ { 1 } } } \bigg ) ^ { 2 d } , \qquad \varepsilon _ { \mathrm { T V } } ^ { - 2 } = \frac { d n M T } { \mathfrak { s } _ { n } ( P _ { 0 } ) } ,
$$

which gives the stated bound on B.

Proof of Corollary $\it 4 .$ Let $\Lambda : = \log ( K d n / { \mathfrak { s } } _ { n } ( P _ { 0 } ) )$ . Since ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \leq n$ and $K , d \geq 2$ , we have $\Lambda > 1$

For the stated grid, $\Delta = { \mathfrak { s } } _ { n } ( P _ { 0 } ) / ( d n \Lambda )$ . By Theorem A.15,

$$
\mathcal { R } _ { \tau } ( T , \theta ; M ) = e ^ { - T } d \log K + \Delta d \log \left( \frac { K } { \Delta } \right) .
$$

The definitions of T and $\Delta$ give

$$
e ^ { - T } d \log K = \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { e n } , \qquad \Delta d \log \left( \frac { K } { \Delta } \right) = \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } \left( 1 + \frac { \log \Lambda } { \Lambda } \right) \lesssim \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } .
$$

Hence $\mathcal { R } _ { \tau } ( T ; M ) \lesssim \mathfrak { s } _ { n } ( P _ { 0 } ) / n$ . Moreover, $\delta = 0$ and $T \geq \log ( 2 e \log 2 ) > 1$ , so Theorem 3 applies.

It remains to simplify the architecture bounds. Since $t _ { 1 } = \Delta \le 1 / ( d \Lambda ) < 1$ , the inequality $1 - e ^ { - \Delta } \geq \Delta e ^ { - \Delta }$ gives

$$
\frac { K } { 1 - e ^ { - \Delta } } \leq \frac { e K d n \Lambda } { \mathfrak { s } _ { n } ( P _ { 0 } ) } , \qquad d \Delta \leq \frac { 1 } { \Lambda } .
$$

Therefore,

$$
\log \left( \frac { K } { 1 - e ^ { - \Delta } } \right) \lesssim \Lambda .
$$

Also, since log $K \le K / e$

$$
T = \Lambda + \log \left( \frac { e \log K } { K } \right) \leq \Lambda .
$$

Consequently,

$$
N = \left\lceil { \frac { T } { \Delta } } \right\rceil \lesssim { \frac { d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } } \Lambda ^ { 2 } , \qquad \log \left( { \frac { n T } { \mathfrak { s } _ { n } ( P _ { 0 } ) } } \right) \lesssim \Lambda .
$$

Substituting these bounds into Theorem 3 yields

$$
L \lesssim d ^ { 2 } \Lambda ^ { 2 } , \qquad W \lesssim \frac { d ^ { 2 } K n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \Lambda ^ { 2 } + d D _ { n } + d ^ { 3 } \Lambda ^ { 3 } , \qquad S \lesssim \frac { d ^ { 2 } K ^ { 2 } n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \Lambda ^ { 2 } + d ^ { 2 } D _ { n } \Lambda + d ^ { 4 } \Lambda ^ { 4 } .
$$

Finally, $M = B _ { 1 } ( \Delta ) \le K / ( 1 - e ^ { - \Delta } ) \lesssim K d n \Lambda / { \mathfrak { s } } _ { n } ( P _ { 0 } )$ , and

$$
\left( \frac { K } { 1 - e ^ { - \Delta } } \right) ^ { 2 d } \lesssim \left( \frac { K d n \Lambda } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) ^ { 2 d } ,
$$

where we used $d \Delta \leq 1 / \Lambda$ . Since $T \leq \Lambda$ , the remaining terms N and dn $M T / { \mathfrak { s } } _ { n } ( P _ { 0 } )$ are dominated by the same quantity. Thus

$$
B \lesssim \left[ \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \log \left( \frac { K d n } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) \right] ^ { 2 d } .
$$

## B.2 Minimax Lower Bound: Proof of Theorem 5

Proof of Theorem 5. The upper bound follows immediately from Lemma A.6 and ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \leq s _ { n }$ for $P _ { 0 } \in \mathcal { P } _ { s _ { n } }$

For the lower bound, let

$$
m : = \left\lfloor { \frac { s _ { n } } { 2 } } \right\rfloor , \qquad \gamma : = { \frac { 1 } { 2 { \sqrt { 2 } } } } { \sqrt { \frac { m } { n } } } .
$$

Since $2 \leq s _ { n } \leq n$ , we have $0 \textless \gamma \leq 1 / 4$ , and since 2m $\le \ s _ { n } \ \le \ K ^ { d }$ , choose distinct states $\pmb { x } _ { 1 } , \pmb { y } _ { 1 } , \dots , \pmb { x } _ { m } , \pmb { y } _ { m } \in [ K ] ^ { d }$ . For each $\theta \in \{ - 1 , 1 \} ^ { m }$ , define $P _ { 0 } ^ { \theta } \in \Delta _ { K ^ { d } }$ by

$$
P _ { 0 } ^ { \theta } ( { \pmb x } _ { i } ) : = \frac { 1 + \gamma \theta _ { i } } { 2 m } , \qquad P _ { 0 } ^ { \theta } ( { \pmb y } _ { i } ) : = \frac { 1 - \gamma \theta _ { i } } { 2 m } , \qquad i \in [ m ] ,
$$

and set $P _ { 0 } ^ { \theta } ( z ) = 0$ elsewhere. Since $\mathfrak { s } _ { n } ( P _ { 0 } ^ { \theta } ) \leq | \operatorname { s u p p } ( P _ { 0 } ^ { \theta } ) | = 2 m \leq s _ { n }$ , every $P _ { 0 } ^ { \theta }$ belongs to $\mathcal { P } _ { s _ { n } }$

Fix any possibly randomized estimator ${ \widetilde { P } } _ { ; }$ and define ${ \widetilde { \theta } } _ { i } = 1 { \mathrm { ~ i f ~ } } { \widetilde { P } } ( \pmb { x } _ { i } ) \geq { \widetilde { P } } ( \pmb { y } _ { i } )$ and $\widetilde { \theta } _ { i } = - 1$ otherwise. By the triangle inequality,

$$
d _ { \mathrm { T V } } ( \widetilde { P } , P _ { 0 } ^ { \theta } ) \geq \frac { \gamma } { 2 m } \sum _ { i = 1 } ^ { m } \mathbf { 1 } \{ \widetilde { \theta } _ { i } \neq \theta _ { i } \} .
$$

Let $\mathbb { P } _ { \theta }$ and $\mathbb { E } _ { \theta }$ denote probability and expectation under $( P _ { 0 } ^ { \theta } ) ^ { \otimes n }$ , including any auxiliary randomness of ${ \widetilde { P } } .$ Averaging uniformly over θ gives

$$
\operatorname* { s u p } _ { \theta } \mathbb { E } _ { \theta } d _ { \mathrm { T V } } ( \widetilde { P } , P _ { 0 } ^ { \theta } ) \geq \frac { \gamma } { 2 m } \sum _ { i = 1 } ^ { m } \frac { 1 } { 2 ^ { m } } \sum _ { \theta } \mathbb { P } _ { \theta } ( \widetilde { \theta } _ { i } \neq \theta _ { i } ) .
$$

Fix $i \in [ m ]$ and $\theta _ { - i } \in \{ - 1 , 1 \} ^ { m - 1 }$ , and let $P _ { 0 } ^ { + }$ and $P _ { 0 } ^ { - }$ denote the distributions obtained by setting $\theta _ { i } = 1$ and $\theta _ { i } = - 1$ , respectively. A direct calculation gives

$$
D _ { \mathrm { K L } } ( P _ { 0 } ^ { + } \Vert P _ { 0 } ^ { - } ) = \frac { \gamma } { m } \log \left( \frac { 1 + \gamma } { 1 - \gamma } \right) \leq \frac { 4 \gamma ^ { 2 } } { m } ,
$$

where we used $\log ( ( 1 + t ) / ( 1 - t ) ) \leq 4 t$ for $0 \leq t \leq 1 / 2$ . Therefore,

$$
D _ { \mathrm { K L } } \left( ( P _ { 0 } ^ { + } ) ^ { \otimes n } \Vert ( P _ { 0 } ^ { - } ) ^ { \otimes n } \right) \leq \frac { 1 } { 2 } ,
$$

and Pinsker's inequality gives $d _ { \mathrm { T V } } ( ( P _ { 0 } ^ { + } ) ^ { \otimes n } , ( P _ { 0 } ^ { - } ) ^ { \otimes n } ) \leq 1 / 2$ . Le Cam's two-point inequality then yields

$$
\frac { 1 } { 2 } \left\{ \mathbb { P } _ { + } ( \widetilde { \theta } _ { i } = - 1 ) + \mathbb { P } _ { - } ( \widetilde { \theta } _ { i } = 1 ) \right\} \geq \frac { 1 } { 4 } .
$$

Averaging over $\theta _ { - i }$ shows that

$$
\frac { 1 } { 2 ^ { m } } \sum _ { \theta } \mathbb { P } _ { \theta } ( \widetilde { \theta } _ { i } \neq \theta _ { i } ) \ge \frac { 1 } { 4 } .
$$

Consequently,

$$
\operatorname* { s u p } _ { \theta } \mathbb { E } _ { \theta } d _ { \mathrm { T V } } ( \widetilde { P } , P _ { 0 } ^ { \theta } ) \geq \frac { \gamma } { 8 } \geq \frac { 1 } { 3 2 \sqrt { 2 } } \sqrt { \frac { s _ { n } } { n } } ,
$$

where we used $m = \lfloor s _ { n } / 2 \rfloor \geq s _ { n } / 4$

Since $\{ P _ { 0 } ^ { \theta } : \theta \in \{ - 1 , 1 \} ^ { m } \} \subseteq { \mathcal { P } } _ { s _ { n } }$ , every estimator $\widetilde { P }$ satisfies

$$
\operatorname* { s u p } _ { P _ { 0 } \in \mathcal { P } _ { s _ { n } } } \mathbb { E } _ { P _ { 0 } } d _ { \mathrm { T V } } ( P _ { 0 } , \widetilde { P } ) \geq \frac { 1 } { 3 2 \sqrt { 2 } } \sqrt { \frac { s _ { n } } { n } } .
$$

Taking the infimum over all estimators $\widetilde { P }$ completes the proof.

## C Proofs for KL Results

## C.1 Upper Bound: Proofs of Theorem 6 and Corollary $\mathbf { 7 }$

Proof of Theorem 6. The proof follows that of Theorem 3, except that Lemma A.13 replaces the triangle inequality. Set

$$
\varepsilon _ { \mathrm { K L } } : = \frac { 1 } { \sqrt { d M T B _ { 0 } ( \delta ) } } .
$$

Since d $, M , T , B _ { 0 } ( \delta ) \ge 1$ , we have $\boldsymbol { \varepsilon } _ { \mathrm { K L } } \in ( 0 , 1 )$ . Applying Lemma A.18 to $ { \widehat { P } } _ { 0 }$ and arguing as in the proof of Theorem 3 gives a clipped score $\sigma ^ { \dagger }$ satisfying

$$
\mathcal { L } _ { T } ^ { \mathrm { S E } } ( \sigma ^ { \dagger } ; \widehat { P } _ { 0 } ) \leq \frac { d M T \varepsilon _ { \mathrm { K L } } ^ { 2 } } { 2 } = \frac { 1 } { 2 B _ { 0 } ( \delta ) } .
$$

Hence, by Lemma A.14 and the $\varepsilon _ { \mathrm { o p t } }$ -optimality of ${ \widehat { \sigma } } .$

$$
\mathcal { L } _ { \mathcal { T } } ^ { \mathrm { S E } } ( \widehat { \sigma } ; \widehat { P } _ { 0 } ) \lesssim \frac { 1 } { B _ { 0 } ( \delta ) } .
$$

Applying Assumption 2 conditionally on the observations and using the assumed sampler remainder gives

$$
D _ { \mathrm { K L } } \left( \widehat { P } _ { \delta } \Vert Q _ { \delta } ^ { A } \right) \lesssim \frac { 1 } { B _ { 0 } ( \delta ) } .
$$

By Lemma $\mathrm { A } . 4 , \widehat { P } _ { \delta } ( { \pmb x } ) \geq B _ { 0 } ( \delta ) ^ { - 1 }$ for every x. Therefore, Lemma A.13 yields

$$
D _ { \mathrm { K L } } \left( P _ { 0 } \Vert Q _ { \delta } ^ { \mathcal { A } } \right) \lesssim D _ { \mathrm { K L } } ( P _ { 0 } \Vert \widehat { P } _ { \delta } ) + \frac { 1 } { B _ { 0 } ( \delta ) } .
$$

Since $\delta = 1 / n$ , Lemma A.11 controls the first term. Moreover,

$$
B _ { 0 } ( \delta ) ^ { - 1 } = \left( \frac { 1 - e ^ { - 1 / n } } { K } \right) ^ { d } \leq ( K n ) ^ { - d } \lesssim \frac { \log n } { n } \leq \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) \log ( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) ) \log n } { n } .
$$

Taking expectations therefore gives

$$
\mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } \left( P _ { 0 } \| Q _ { \delta } ^ { \varDelta } \right) \lesssim \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) \log \left( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \right) \log n } { n } .
$$

It remains to verify the architecture bounds. Since $t _ { 1 } > \delta = 1 / n$ and $1 - e ^ { - 1 / n } \geq 1 / ( 2 n )$

$$
B _ { 0 } ( t _ { 1 } ) \leq B _ { 0 } ( \delta ) \leq ( 2 K n ) ^ { d } , \qquad M \leq \frac { K } { 1 - e ^ { - 1 / n } } \leq 2 K n .
$$

Thus

$$
\log \left( \frac { B _ { 0 } ( t _ { 1 } ) } { \varepsilon _ { \mathrm { K L } } } \right) \lesssim d \log ( K n ) + \log T .
$$

Substituting this into Lemma A.18 gives the stated bounds on L, W, S. Finally,

$$
B _ { 0 } ( t _ { 1 } ) ^ { 2 } = \bigg ( \frac { K } { 1 - e ^ { - t _ { 1 } } } \bigg ) ^ { 2 d } , \qquad \varepsilon _ { \mathrm { K L } } ^ { - 2 } = d M T \bigg ( \frac { K } { 1 - e ^ { - 1 / n } } \bigg ) ^ { d } ,
$$

which gives the stated bound on B.

Proof of Corollary 7. Define

$$
\Xi : = \frac { K } { 1 - e ^ { - 1 / n } } .
$$

Since $1 - e ^ { - 1 / n } \leq 1 / n$ we have $\Xi \geq K n$ . First note that $T = \log ( e d \Xi ^ { d } \log K ) \geq 1$

For the stated grid, $\Delta = ( d ^ { 2 } \Xi ^ { d } \log ( K n ) ) ^ { - 1 }$ . By Theorem A.15,

$$
\mathcal { R } _ { \tau } ( T , \theta ; M ) \lesssim e ^ { - T } d \log K + \Delta d \log \left( \frac { K } { \Delta } \right) .
$$

Since

$$
e ^ { - T } d \log K = { \frac { 1 } { e \Xi ^ { d } } } , \qquad \log \left( { \frac { K } { \Delta } } \right) = \log \left( d ^ { 2 } K \Xi ^ { d } \log ( K n ) \right) \lesssim d \log ( K n ) ,
$$

we obtain $\mathcal { R } _ { \tau } ( \mathcal { T } ; M ) \lesssim \Xi ^ { - d }$ , as required by Theorem 6.

It remains to simplify the architecture bounds. Since $t _ { 1 } = \delta + \Delta > \delta = 1 / n$

$$
\frac { K } { 1 - e ^ { - t _ { 1 } } } \leq \Xi , \qquad M = B _ { 1 } ( t _ { 1 } ) \leq \Xi .
$$

Moreover, $T \lesssim d \log ( K n )$ , and hence

$$
N = \left\lceil \frac { T - \delta } { \Delta } \right\rceil \lesssim d ^ { 3 } \Xi ^ { d } \log ^ { 2 } ( K n ) .
$$

Substituting these bounds into Theorem 6, and using $D _ { n } \leq n \leq \Xi ^ { d }$ , gives

$$
L \lesssim d ^ { 2 } \log ^ { 2 } ( K n ) , \qquad W \lesssim d ^ { 4 } K \Xi ^ { d } \log ^ { 2 } ( K n ) , \qquad S \lesssim d ^ { 4 } K ^ { 2 } \Xi ^ { d } \log ^ { 2 } ( K n ) .
$$

Finally, the bound on B in Theorem 6, together with $M \leq \Xi , T \lesssim d \log ( K n )$ , and the bound on N, gives $B \lesssim \Xi ^ { 2 d }$ □

## C.2 Minimax Lower Bound: Proof of Theorem 8

The empirical distribution $\widehat { P } _ { 0 }$ assigns zero probability to unobserved states and therefore has infinite KL risk. We instead use the adaptive add-λ estimator analyzed by Mourtada (2026), specialized to the state space $[ K ] ^ { d }$

Lemma C.1. Define the adaptive additive-smoothing estimator

$$
\widehat { P } ^ { \mathrm { a d } } ( { \pmb x } ) : = \frac { N _ { \pmb x } + D _ { n } / K ^ { d } } { n + D _ { n } } , \qquad { \pmb x } \in [ K ] ^ { d } .
$$

Then

$$
\mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } \left( P _ { 0 } \| \widehat { P } ^ { \mathrm { a d } } \right) \leq 4 . 4 \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) \log \left( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \right) } { n } .
$$

Proof. For $r > 0 .$ define

$$
{ \mathfrak { s } } _ { r } ^ { \circ } ( P _ { 0 } ) : = \sum _ { \pmb { x } \in [ K ] ^ { d } } \operatorname* { m i n } \left\{ e ^ { 1 - r P _ { 0 } ( { \pmb x } ) } , r P _ { 0 } ( { \pmb x } ) \right\} .
$$

By Mourtada (2026, Proposition 4.6), specialized to an alphabet of size $K ^ { d }$

$$
\mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } \left( P _ { 0 } \| \widehat { P } ^ { \mathrm { a d } } \right) \leq \frac { 2 . 4 \mathfrak { s } _ { n } ( P _ { 0 } ) + 2 \mathfrak { s } _ { n / 2 } ^ { \circ } ( P _ { 0 } ) \log \left( e K ^ { d } / \mathfrak { s } _ { n / 2 } ^ { \circ } ( P _ { 0 } ) \right) } { n + 1 } .
$$

For every $u \geq 0$

$$
\operatorname* { m i n } \left\{ e ^ { 1 - u / 2 } , { \frac { u } { 2 } } \right\} \leq \operatorname* { m i n } \{ u , 1 \} ,
$$

as follows by considering $u \leq 1 , 1 \leq u \leq 2$ , and $u \geq 2$ . Applying this pointwise with $u = n P _ { 0 } ( { \pmb x } )$ gives $\mathfrak { s } _ { n / 2 } ^ { \circ } ( P _ { 0 } ) \leq \mathfrak { s } _ { n } ( P _ { 0 } )$ . Since $t \mapsto t \log ( e K ^ { d } / t )$ is increasing on $( 0 , K ^ { d } ]$ and ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) \leq K ^ { d }$

$$
\mathfrak { s } _ { n / 2 } ^ { \circ } ( P _ { 0 } ) \log \left( \frac { e K ^ { d } } { \mathfrak { s } _ { n / 2 } ^ { \circ } ( P _ { 0 } ) } \right) \le \mathfrak { s } _ { n } ( P _ { 0 } ) \log \left( \frac { e K ^ { d } } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) , \qquad \mathfrak { s } _ { n } ( P _ { 0 } ) \le \mathfrak { s } _ { n } ( P _ { 0 } ) \log \left( \frac { e K ^ { d } } { \mathfrak { s } _ { n } ( P _ { 0 } ) } \right) .
$$

Substituting these bounds and using $n + 1 \ge n$ proves the claim.

Now we prove that rate in Lemma C.1 is actually minimax risk.

Proof of Theorem 8. Since $t \mapsto t \log ( e K ^ { d } / t )$ is increasing on $( 0 , K ^ { d } ]$ , the upper bound follows from Lemma C.1.

For the lower bound, set

$$
m : = \operatorname* { m i n } \left\{ \lfloor s _ { n } \rfloor , \left\lfloor \frac { K ^ { d } } { 2 } \right\rfloor \right\} .
$$

Since $2 \leq s _ { n } \leq$ min $\{ n , K ^ { d } \}$ , we have $2 \leq m \leq n$ and 2m $\leq K ^ { d }$ . Fix $\pmb { x } _ { \star } \in [ K ] ^ { d }$ , and let

$$
{ \mathcal { I } } : = \left\{ J \subseteq [ K ] ^ { d } \setminus \{ { \pmb x } _ { \star } \} : | J | = m - 1 \right\} .
$$

For each $J \in \mathcal { I }$ , define $P _ { 0 } ^ { J }$ by

$$
P _ { 0 } ^ { J } ( \pmb { x } _ { \star } ) : = 1 - \frac { m - 1 } { n } , \qquad P _ { 0 } ^ { J } ( \pmb { x } ) : = \frac { 1 } { n } \mathbf { 1 } \{ \pmb { x } \in J \} , \quad \pmb { x } \neq \pmb { x } _ { \star } .
$$

Since $\mathfrak { s } _ { n } ( P _ { 0 } ^ { J } ) \leq | \operatorname { s u p p } ( P _ { 0 } ^ { J } ) | = m \leq s _ { n }$ , every $P _ { 0 } ^ { J }$ belongs to $\mathcal { P } _ { s _ { n } }$

Draw J uniformly from $\mathcal { T } .$ For any possibly randomized estimator ${ \widetilde { P } } ,$

$$
\operatorname* { s u p } _ { P _ { 0 } \in \mathcal { P } _ { s _ { n } } } \mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } ( P _ { 0 } \| \widetilde { P } ) \ge \mathbb { E } _ { J } \mathbb { E } _ { P _ { 0 } ^ { J } } D _ { \mathrm { K L } } ( P _ { 0 } ^ { J } \| \widetilde { P } ) .
$$

Let $O : = \{ \pmb { x } \neq \pmb { x } _ { \star } : N _ { \pmb { x } } \geq 1 \}$ be the set of observed rare states. Under $P _ { 0 } ^ { J }$ , we have $O \subseteq J .$ and the likelihood under $P _ { 0 } ^ { J ^ { \prime } }$ is proportional to $1 \{ O \subseteq J ^ { \prime } \}$ . Hence, conditional on the sample, the posterior distribution of J is uniform over the sets $J ^ { \prime } \in \mathcal { I }$ containing $O$

Let $\overline { { P } } : = \mathbb { E } [ P _ { 0 } ^ { J } \mid X _ { 0 } ^ { ( 1 ) } , \ldots , X _ { 0 } ^ { ( n ) } ]$ . Then

$$
\begin{array} { r } { \overline { { P } } ( \pmb { x } ) = \left\{ \begin{array} { l l } { 1 - \frac { m - 1 } { n } , } & { \pmb { x } = \pmb { x } _ { \star } , } \\ { } & { } \\ { \frac { 1 } { n } , } & { \pmb { x } \in O , } \\ { } & { } \\ { \frac { 1 } { n } \frac { m - 1 - | O | } { K ^ { d } - 1 - | O | } , } & { \pmb { x } \notin O \cup \{ \pmb { x } _ { \star } \} . } \end{array} \right. } \end{array}
$$

For every $Q \in \Delta _ { K ^ { d } }$ 2

$$
\mathbb { E } \Big [ D _ { \mathrm { K L } } ( P _ { 0 } ^ { J } \| Q ) \ | \ X _ { 0 } ^ { ( 1 ) } , \dots , X _ { 0 } ^ { ( n ) } \Big ] = C ( X _ { 0 } ^ { ( 1 ) } , \dots , X _ { 0 } ^ { ( n ) } ) + D _ { \mathrm { K L } } ( \overline { { P } } \| Q ) ,
$$

where $C$ does not depend on $Q .$ Thus the conditional Bayes risk is minimized by $Q = { \overline { { P } } }$ , and the same conclusion holds for randomized estimators after averaging over their auxiliary randomness. Consequently,

$$
\begin{array} { r } { \mathbb { E } _ { J } \mathbb { E } _ { P _ { 0 } ^ { J } } D _ { \mathrm { K L } } ( P _ { 0 } ^ { J } \| \widetilde { P } ) \ge \mathbb { E } _ { J } \mathbb { E } _ { P _ { 0 } ^ { J } } D _ { \mathrm { K L } } ( P _ { 0 } ^ { J } \| \overline { { P } } ) . } \end{array}
$$

If $m - 1 - | O | > 0$ , only the unobserved states in $J \backslash O$ contribute, and

$$
D _ { \mathrm { K L } } \big ( P _ { 0 } ^ { J } \| \overline { { P } } \big ) = \frac { m - 1 - | O | } { n } \log \left( \frac { K ^ { d } - 1 - | O | } { m - 1 - | O | } \right) \geq \frac { m - 1 - | O | } { n } \log \left( \frac { K ^ { d } } { m } \right) .
$$

The same lower bound holds trivially when $m - 1 - | O | = 0$ . Moreover,

$$
\mathbb { E } _ { P _ { 0 } ^ { J } } [ m - 1 - | O | ] = ( m - 1 ) \left( 1 - \frac { 1 } { n } \right) ^ { n } .
$$

Therefore,

$$
\mathbb { E } _ { J } \mathbb { E } _ { P _ { 0 } ^ { J } } D _ { \mathrm { K L } } \big ( P _ { 0 } ^ { J } \| \overline { { P } } \big ) \geq \frac { m - 1 } { n } \left( 1 - \frac { 1 } { n } \right) ^ { n } \log \left( \frac { K ^ { d } } { m } \right) .
$$

Since $n \geq 2 , m \geq 2$ , and $K ^ { d } \ge 2 m$

$$
\left( 1 - \frac 1 n \right) ^ { n } \geq \frac 1 4 , \qquad m - 1 \geq \frac m 2 , \qquad \log \left( \frac { K ^ { d } } m \right) \geq \frac 1 3 \log \left( \frac { e K ^ { d } } m \right) .
$$

Hence

$$
\mathbb { E } _ { J } \mathbb { E } _ { P _ { 0 } ^ { J } } D _ { \mathrm { K L } } ( P _ { 0 } ^ { J } \| \widetilde { P } ) \ge \frac { 1 } { 2 4 } \frac { m \log ( e K ^ { d } / m ) } { n } .
$$

It remains to compare m with $s _ { n }$ . If $s _ { n } \leq K ^ { d } / 2$ , then $m = \lfloor s _ { n } \rfloor \ge s _ { n } / 2$ SO

$$
m \log \left( \frac { e K ^ { d } } { m } \right) \geq \frac { s _ { n } } { 2 } \log \left( \frac { e K ^ { d } } { s _ { n } } \right) .
$$

If $s _ { n } > K ^ { d } / 2$ , then $m = \lfloor K ^ { d } / 2 \rfloor \ge K ^ { d } / 3$ , while $s _ { n } \log ( e K ^ { d } / s _ { n } ) \le K ^ { d }$ , and hence

$$
m \log \left( \frac { e K ^ { d } } { m } \right) \geq \frac 1 3 s _ { n } \log \left( \frac { e K ^ { d } } { s _ { n } } \right) .
$$

Thus, in both cases,

$$
m \log \left( \frac { e K ^ { d } } { m } \right) \geq \frac 1 3 s _ { n } \log \left( \frac { e K ^ { d } } { s _ { n } } \right) .
$$

Since $\widetilde { P }$ was arbitrary,

$$
\operatorname* { i n f } _ { \widetilde { P } } \operatorname* { s u p } _ { P _ { 0 } \in \mathcal { P } _ { s _ { n } } } \mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } ( P _ { 0 } \| \widetilde { P } ) \geq \frac { 1 } { 7 2 } \frac { s _ { n } \log ( e K ^ { d } / s _ { n } ) } { n } .
$$

## D Exact One-Step Empirical Reverse Sampling

We construct an exact one-step empirical reverse sampler that attains the same TV and KL rates as the fixed-grid samplers in Theorems 3 and 6, while requiring only $O ( d ( D _ { n } + K ) )$ operations per generated sample once the distinct observations and their multiplicities have been stored. This shows that the unfavorable computational dependence in Corollary 7 is not intrinsic to the statistical KL rate. Rather than discretizing reverse time or simulating intermediate jumps, the sampler draws directly from a finite-time empirical reverse kernel.

## D.1 Exact reverse kernel and sampling algorithm

For $0 \leq \delta < T$ , define the empirical reverse transition kernel from forward time $T$ toδby

$$
\overleftarrow { P } _ { T  \delta , \widehat { P } } ( \pmb { x } , \pmb { y } ) : = \frac { \widehat { P } _ { \delta } ( \pmb { y } ) P _ { T - \delta } ( \pmb { y } , \pmb { x } ) } { \widehat { P } _ { T } ( \pmb { x } ) } , \qquad \pmb { x } , \pmb { y } \in [ K ] ^ { d } .\tag{D.1}
$$

By Lemma A.14, the unrestricted minimizer of the exact empirical denoising score-entropy loss is the empirical score $\widehat { \sigma } _ { t } ^ { \star } ( { \pmb y } , { \pmb x } ) = \widehat { P } _ { t } ( { \pmb y } ) / \widehat { P } _ { t } ( { \pmb x } )$ . If this score is available for every $t \in ( \delta , T ]$ and the reverse dynamics are simulated exactly, the resulting reverse rates are those of the time reversal of the forward process initialized from $ { \widehat { P } } _ { 0 }$ . Bayes' rule then gives (D.1) as its finite-time transition kernel. We initialize this kernel from the stationary distribution $\pi ,$ and denote the resulting output distribution by

$$
Q _ { \delta } ^ { \mathrm { o s } } : = \pi \overleftarrow { P } _ { T  \delta , \widehat { P } } .
$$

Thus, relative to the exact empirical reverse process, the only initialization discrepancy is replacing $\widehat { P } _ { T }$ by π.

Write supp $( \widehat { P } _ { 0 } ) = \{ z _ { 1 } , \dots , z _ { D _ { n } } \}$ , so that $\widehat { P } _ { 0 } ( z _ { i } ) = N _ { z _ { i } } / n$ . Given $X _ { T } = x$ , Bayes' rule gives

$$
w _ { i } ( \pmb { x } ) : = \mathbb { P } _ { \widehat { P } _ { 0 } } ( X _ { 0 } = z _ { i } \mid X _ { T } = \pmb { x } ) = \frac { N _ { z _ { i } } P _ { T } ( z _ { i } , \pmb { x } ) } { \sum _ { j = 1 } ^ { D _ { n } } N _ { z _ { j } } P _ { T } ( z _ { j } , \pmb { x } ) } , \qquad i \in [ D _ { n } ] ,
$$

where $\begin{array} { r } { P _ { T } ( z _ { i } , \pmb { x } ) = \prod _ { \ell = 1 } ^ { d } P _ { T } ^ { \mathrm { c o o r d } } ( z _ { i } ^ { \ell } , x ^ { \ell } ) } \end{array}$

Conditional on $I = i$ and $X _ { T } = x$ , the state at time δ follows the forward bridge from $z _ { i }$ to x.

Algorithm 1 Exact one-step empirical reverse sampling   
Require: Times $0 \leq \delta < T _ { \cdot }$ distinct observations $\{ z _ { i } \} _ { i = 1 } ^ { D _ { n } }$ , and counts $\{ N _ { z _ { i } } \} _ { i = 1 } ^ { D _ { n } }$   
Ensure: A sample $Y \sim Q _ { \delta } ^ { \mathrm { o s } }$   
1: Draw $X _ { T } \sim \pi$   
2: for $i = 1 , \ldots , D _ { n }$ do   
3: Compute $\widetilde { w } _ { i }  N _ { z _ { i } } \prod _ { \ell = 1 } ^ { d } P _ { T } ^ { \mathrm { c o o r d } } ( z _ { i } ^ { \ell } , X _ { T } ^ { \ell } )$   
4: end for   
5: Draw $I \in [ D _ { n } ]$ with $\mathbb { P } ( I = i \mid X _ { T } ) = \frac { \widetilde { w } _ { i } } { \sum _ { j = 1 } ^ { D _ { n } } \widetilde { w } _ { j } }$   
6: for $\ell = 1 , \ldots , d$ do   
7: Independently draw $Y ^ { \ell } \sim B _ { I , \ell } ^ { T , \delta } ( \cdot \mid X _ { T } ^ { \ell } )$   
8: end for   
9: return $\boldsymbol { Y } = ( Y ^ { 1 } , \ldots , Y ^ { d } )$

For each coordinate $\ell \in [ d ]$ , define

$$
B _ { i , \ell } ^ { T , \delta } ( a \mid x ^ { \ell } ) : = \frac { P _ { \delta } ^ { \mathrm { c o o r d } } ( z _ { i } ^ { \ell } , a ) P _ { T - \delta } ^ { \mathrm { c o o r d } } ( a , x ^ { \ell } ) } { P _ { T } ^ { \mathrm { c o o r d } } ( z _ { i } ^ { \ell } , x ^ { \ell } ) } , \qquad a \in [ K ] .
$$

Since the forward process evolves independently across coordinates, the bridge law factorizes as

$$
\mathbb { P } ( X _ { \delta } = \pmb { y } \mid X _ { 0 } = z _ { i } , X _ { T } = \pmb { x } ) = \prod _ { \ell = 1 } ^ { d } B _ { i , \ell } ^ { T , \delta } ( \pmb { y } ^ { \ell } \mid \pmb { x } ^ { \ell } ) .
$$

Thus, after sampling I, the $K ^ { d } .$ -state bridge can be sampled through d independent K-state distributions. The complete procedure is given in Algorithm 1.

Proposition D.1 (Correctness and computational cost). Conditional on the observations, Algorithm 1 returns an exact sample from $Q _ { \delta } ^ { \mathrm { o s } }$ . If the distinct observations and their multiplicities have been stored, one execution requires $O ( d ( D _ { n } + K ) )$ operations.

Proof. Conditional on $I = i$ and $X _ { T } = x$ , the bridge factorization gives

$$
\mathbb { P } ( Y = pmb { y } \mid I = i , X _ { T } = \pmb { x } ) = \frac { { P } _ { \delta } ( z _ { i } , \pmb { y } ) { P } _ { T - \delta } ( \pmb { y } , \pmb { x } ) } { { P } _ { T } ( z _ { i } , \pmb { x } ) } .
$$

Averaging over I yields

$$
\mathbb { P } ( Y = pmb { y } \mid X _ { T } = \pmb { x } ) = \frac { \widehat { P } _ { \delta } ( \pmb { y } ) P _ { T - \delta } ( \pmb { y } , \pmb { x } ) } { \widehat { P } _ { T } ( \pmb { x } ) } = \overleftarrow { P } _ { T  \delta , \widehat { P } } ( \pmb { x } , \pmb { y } ) .
$$

Averaging over $X _ { T } \sim \pi$ proves that $Y \sim Q _ { \delta } ^ { \mathrm { o s } }$

Computing the $D _ { n }$ mixture weights requires $O ( d D _ { n } )$ operations. Conditional on I, evaluating and sampling the d coordinate bridges requires $O ( d K )$ operations. Thus, the total cost is $O ( d ( D _ { n } +$ K)). □

## D.2 Statistical guarantees

We next show that the one-step sampler attains the TV and KL rates of Theorems 3 and 6 without constructing a fixed-grid score network.

Theorem D.2 (Statistical guarantees for the exact one-step sampler). For $0 \leq \delta < T$ , the following statements hold.

(i) TV loss. For every realization of the observations,

$$
d _ { \mathrm { T V } } ( P _ { 0 } , Q _ { \delta } ^ { \mathrm { o s } } ) \leq d _ { \mathrm { T V } } ( P _ { 0 } , \widehat { P } _ { 0 } ) + d \delta + d e ^ { - T } .
$$

Consequently, if $d \delta \lesssim \sqrt { \mathfrak { s } _ { n } ( P _ { 0 } ) / n }$ and $T \geq \log ( d \sqrt { n / { \mathfrak { s } } _ { n } ( P _ { 0 } ) } )$ , then

$$
\mathbb { E } _ { P _ { 0 } } d _ { \mathrm { T V } } ( P _ { 0 } , Q _ { \delta } ^ { \mathrm { o s } } ) \lesssim \sqrt { \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) } { n } } .
$$

(ii) KL loss. Suppose additionally that $\delta > 0$ . For every realization of the observations,

$$
D _ { \mathrm { K L } } ( P _ { 0 } \| Q _ { \delta } ^ { \mathrm { o s } } ) \leq D _ { \mathrm { K L } } ( P _ { 0 } \| \widehat { P } _ { \delta } ) + d \log \left( 1 + ( K - 1 ) e ^ { - T } \right) .
$$

Consequently, setting $\delta = 1 / n$ and choosing $T \geq \log ( K d n / { \mathfrak { s } } _ { n } ( P _ { 0 } ) )$ gives

$$
\mathbb { E } _ { P _ { 0 } } D _ { \mathrm { K L } } ( P _ { 0 } \| Q _ { 1 / n } ^ { \mathrm { o s } } ) \lesssim \frac { \mathfrak { s } _ { n } ( P _ { 0 } ) \log \left( e K ^ { d } / \mathfrak { s } _ { n } ( P _ { 0 } ) \right) \log n } { n } .
$$

Proof. Since $\widehat { P } _ { T } \overleftarrow { P } _ { T  \delta , \widehat { P } } = \widehat { P } _ { \delta }$ , contraction of total variation under the reverse kernel gives

$$
d _ { \mathrm { T V } } ( \widehat { P } _ { \delta } , Q _ { \delta } ^ { \mathrm { o s } } ) \leq d _ { \mathrm { T V } } ( \widehat { P } _ { T } , \pi ) \leq d e ^ { - T } ,
$$

where the last inequality follows from the coordinatewise refresh coupling. The same coupling gives

$d _ { \mathrm { T V } } ( \widehat { P } _ { 0 } , \widehat { P } _ { \delta } ) \leq d \delta$ . Therefore,

$$
d _ { \mathrm { T V } } ( P _ { 0 } , Q _ { \delta } ^ { \mathrm { o s } } ) \leq d _ { \mathrm { T V } } ( P _ { 0 } , \widehat { P } _ { 0 } ) + d \delta + d e ^ { - T } ,
$$

and the TV conclusion follows from Lemma A.6.

For the $\mathrm { K L }$ statement, the one-coordinate transition probabilities satisfy

$$
P _ { T } ^ { \mathrm { c o o r d } } ( a , b ) \leq \frac { 1 + ( K - 1 ) e ^ { - T } } { K } .
$$

Hence

$$
\widehat { P } _ { T } ( \pmb { x } ) \leq \pi ( \pmb { x } ) \left( 1 + ( K - 1 ) e ^ { - T } \right) ^ { d } .
$$

Using (D.1),

$$
\begin{array} { r } { Q _ { \delta } ^ { \mathrm { o s } } ( \pmb { y } ) \geq \left( 1 + ( K - 1 ) e ^ { - T } \right) ^ { - d } \widehat { P } _ { \delta } ( \pmb { y } ) , } \end{array}
$$

which implies

$$
D _ { \mathrm { K L } } ( P _ { 0 } \| Q _ { \delta } ^ { \mathrm { o s } } ) \leq D _ { \mathrm { K L } } ( P _ { 0 } \| \widehat { P } _ { \delta } ) + d \log \left( 1 + ( K - 1 ) e ^ { - T } \right) .
$$

For $\delta = 1 / n$ and $T \geq \log ( K d n / { \mathfrak { s } } _ { n } ( P _ { 0 } ) )$ , the second term is at most ${ \mathfrak { s } } _ { n } ( P _ { 0 } ) / n$ . The KL conclusion therefore follows from Lemma A.11. □