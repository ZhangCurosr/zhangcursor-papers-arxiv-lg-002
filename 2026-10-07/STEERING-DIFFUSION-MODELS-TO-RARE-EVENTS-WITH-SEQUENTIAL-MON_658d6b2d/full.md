# STEERING DIFFUSION MODELS TO RARE EVENTS WITH SEQUENTIAL MONTE CARLO

Aavash Subedi<sup>∗</sup> Department of Statistics University of Oxford

Tim Reichelt Department of Physics University of Oxford

Philip Stier Department of Physics University of Oxford

Yee Whye Teh Department of Statistics University of Oxford

Christopher Williams Department of Statistics University of Oxford

Saifuddin Syed Department of Statistics University of British Columbia

## ABSTRACT

Diffusion models are increasingly used as surrogates for expensive simulators in weather prediction, molecular dynamics, and materials design. In these models, computing the probability $p _ { 0 } [ E ]$ of an event E is difficult, especially when the event of interest is rare. A stable estimate using Monte Carlo becomes computationally intractable, requiring a growing sample size $\propto 1 / p _ { 0 } [ E ]$ to compensate for an increasing rarity. In this paper, we present Diffusion Importance Sampling of Rare Events or DireSMC, a sequential Monte Carlo scheme that guides a population of weighted samples towards the rare event, giving access not only to samples but also to a calibrated estimate of its probability. We set up our guidance using an analytical relaxation of the event set, allowing the method to easily extend to a wide range of user-defined rare events. We validate our method on a toy problem with analytical solutions and on a score-based climate emulator, where we obtain accurate rare-event probabilities on a range of rarities from $1 0 ^ { - 3 } \mathrm { t o } 1 0 ^ { - 5 }$ , achieving net speed-ups of 9× to 1413× over Monte Carlo.

## 1 INTRODUCTION

Score-based diffusion models (Song et al., 2021b) are a powerful tool for generative modelling, and have become standard in the quantitative sciences as cheap surrogates for expensive simulators in numerical weather prediction (Couairon et al., 2026; Price et al., 2024), molecular simulation (Abramson et al., 2024) and materials design (Zeni et al., 2025). Trained on data from a distribution p , they generate new representative samples by running a denoising process $p _ { t }$ that anneals from a Gaussian reference $p _ { 1 }$ down to the data $p _ { 0 } .$ , driven by a learned score $\dot { \nabla } _ { \boldsymbol { x } } \log \bar { p _ { t } }$

Often, scientists are interested in events E lying in regions where the data distribution assigns little mass p [E]. Such rare events carry outsized downstream impacts, and arise across a wide range of domains, from the environmental sciences (Lancelin et al., 2026; Manshausen et al., 2026; Peard et al., 2026) and bioinformatics (Xie et al., 2026) to finance (Glasserman & Li, 2005), engineering (Au & Wang, 2014) and autonomous driving (O' Kelly et al., 2018). Practitioners typically need two things: representative samples of the event, to characterise possible failure modes, and a calibrated estimate of its probability $p _ { 0 } [ E ]$ ], to inform planning and resource allocation (Kaplan & Garrick, 1981; Rubino et al., 2009). The simplest approach is to draw N independent samples and form a Monte Carlo (MC) average to estimate $p _ { 0 } [ \bar { E } ]$ ], which is difficult precisely because the event is rare. The relative error is $1 / \sqrt { N p _ { 0 } [ E ] } ,$ , hence stabilising the variance requires $N \propto 1 / p _ { 0 } [ E ]$ samples, which is prohibitively expensive (Morio et al., 2014; Rubino et al., 2009). This has motivated a large body of research designed specifically to tackle the poor scaling of standard MC (Glasserman & Li, 2005; Kahn & Harris, 1951; Morio et al., 2014).

Diffusion model steering offers a natural method for generating samples from E without the high costs of simple MC. A reward $r _ { 0 }$ defines a tilted target $\pi _ { 0 } \propto p _ { 0 } e ^ { r _ { 0 } }$ , and inference-time guidance steers a pre-trained sampler towards it. At each denoising step, the gradient of a time-dependent reward $r _ { t }$ is added, so that the sampler targets $\pi _ { t } \propto p _ { t } e ^ { r _ { t } }$ (Chung et al., 2023; Song et al., 2023; Uehara et al., 2025). However, designing a good reward is difficult as the ideal $r _ { t }$ is often intractable, and any approximations used can guide the sampler towards a different distribution to $\pi _ { 0 }$ . For rare-event sampling specifically, setting the tilting term $e ^ { r _ { 0 } } \equiv \mathbf { 1 } _ { E }$ recovers the rare-event tilted distribution $p _ { 0 } ( \cdot | E ) \propto p _ { 0 } \mathbf { 1 } _ { E }$ that we wish to sample from. However, this reward is a poor object for steering, as its gradient cannot be used to steer particles towards $E$

![](images/4e20217b0a25729932e2b9214712d4d9ff0719bab229701b5e5b2001bec32a23.jpg)  
Figure 1: We steer particles (•) from $p _ { \mathrm { r e f } }$ towards $\pi _ { 0 } .$ , which concentrates mass in the rare-event region of interest, using a guidance drift $\nabla _ { x } r _ { k }$ . Particle size reflects the importance weights accumulated along the trajectory, which are used to resample (×) and form a rare event estimate, $p _ { 0 } [ E ]$

Recent works have attempted to address these problems. The error of using an imperfect reward $r _ { t }$ can be addressed using sequential Monte Carlo (SMC) by tracking importance weights along the denoising trajectory and resampling particles to focus on high probability regions (He et al., 2026; Singhal et al., 2025; Skreta et al., 2025; Wu et al., 2023). In diffusion-based molecular simulation, Xie et al. (2026) adapt ideas from Abrams & Bussi (2013); Shirts & Chodera (2008) to estimate the occupancy of a rare metastable state using collective variables and a sequence of events $E _ { i }$ converging to E. Manshausen et al. (2026) instead steer a diffusion model using the probability-flow ODE (PF-ODE) (Song et al., 2021a), and estimate $p _ { 0 } [ E ]$ using importance sampling. These methods have high compute costs, which become especially problematic in the rare event settings of interest here.

Contributions. First, we present DireSMC where we use a sequence of tilted marginals $\pi _ { t }$ to construct an unbiased sequential Monte Carlo estimator of $p _ { 0 } [ E ]$ that runs within the denoising process, obtaining both samples and probability estimates. We show that the variance is governed by the quality of our steering rather than the rarity. Second, we show how to design a reward for rare-event simulation, and in particular for threshold-exceedance events $E = \{ x : \phi ( x ) \geq \tau \}$ giving a smooth surrogate whose look-ahead guidance is closed-form under a Tweedie posterior approximation (Chung et al., 2023; Song et al., 2023). Third, we empirically validate on: (1) a Gaussian mixture where ground truth is available and we obtain probability estimates that do not degrade across $1 0 ^ { - 3 } – 1 0 ^ { - 5 }$ orders of rarity. (2) a score-based climate emulator, where we draw samples and estimate probabilities of compound hot-and-dry extremes over the region of the 2021 Pacific Northwest heatwave, with net speed-ups of 9× to 1413× over Monte Carlo across rarities from $1 0 ^ { - 3 }$ to $1 0 ^ { - 5 }$

## 2 BACKGROUND

Diffusion models. Let $p _ { \mathrm { d a t a } }$ be the true data generating distribution. A diffusion model (Song et al., 2021b) progressively noises the data along a path $p _ { t }$ interpolating from $p _ { \mathrm { d a t a } }$ at $t = 0$ to a Gaussian reference $p _ { \mathrm { r e f } }$ at $t = 1$ . Sampling reverses this process: a $X _ { 1 } \sim p _ { \mathrm { r e f } }$ draw is evolved to $t = 0$ , under the reverse-time SDE,

$$
\begin{array} { r } { d X _ { t } = \left[ \mu _ { t } ( X _ { t } ) - \sigma _ { t } ^ { 2 } \nabla _ { x } \log p _ { t } ( X _ { t } ) \right] d t + \sigma _ { t } d \bar { W } _ { t } , \qquad t \in [ 0 , 1 ] , } \end{array}\tag{1}
$$

to generate from $p _ { \mathrm { d a t a } }$ . The drift $\mu _ { t }$ and the diffusion coefficient $\sigma _ { t }$ are fixed by the noise schedule, and the score $\nabla _ { \boldsymbol { x } } \log \boldsymbol { p _ { t } }$ is approximated by a pre-trained network. In practice, Eq. (1) cannot be simulated directly. Instead, we fix a discretisation schedule $0 = t _ { 0 } < \cdots < t _ { K } = 1$ and a numerical integrator whoese k-th step $t _ { k } \to t _ { k - 1 }$ defines a Markov transition kernel $\bar { P } _ { k - 1 | k } \big ( \mathrm { d } x _ { k - 1 } \ | \ x _ { k } \big )$ Sampling corresponds to composing the K-step chain,

$$
X _ { K } \sim p _ { \mathrm { r e f } } , \qquad X _ { k - 1 } \mid X _ { k } \sim { \bar { P } } _ { k - 1 \mid k } ( \mathrm { d } x _ { k - 1 } \mid X _ { k } ) , \qquad k = K , \dots , 1 .\tag{2}
$$

Let $p _ { k }$ denote the marginal law of $X _ { k }$ under this chain. Hence $p _ { K } = p _ { \mathrm { r e f } }$ and $p _ { k - 1 } ( x _ { k - 1 } ) =$ $\begin{array} { r } { \int p _ { k } ( x _ { k } ) \bar { P } _ { k - 1 | k } ( x _ { k - 1 } \bar { \mid } x _ { k } ) \mathrm { d } x _ { k } } \end{array}$ . Henceforth, we refer to the end-point of the discretised chain as $p _ { 0 }$ , with $X _ { 0 } \sim p _ { 0 }$ . As the discretisation is refined both $p _ { k } \to p _ { t = t _ { k } }$ and $p _ { 0 } \to p _ { \mathrm { d a t a } }$ at a rate given by the weak-order convergence of the integrator (Benton et al., 2024; Chen et al., 2023; Milstein & Tretyakov, 2004). Throughout, we assume that we can simulate from the transition kernels and have access to the score, but not to the densities $p _ { t }$ and $p _ { k }$ , which are generally intractable.

Rare Event Probabilities. Given a rare-event set $E \subset \mathbb { R } ^ { d }$ and the endpoint law of the chain $p _ { 0 }$ , we seek to estimate its probability $\begin{array} { r } { p _ { 0 } [ E ] = \int \mathbf { 1 } _ { E } ( x ) p _ { 0 } ( x ) \mathrm { d } x . } \end{array}$ . A natural estimator of $p _ { 0 } [ E ]$ is obtained by drawing N independent samples, $\bar { X } _ { 0 } ^ { n } \sim p _ { 0 }$ and forming the Monte Carlo (MC) estimate,

$$
\hat { p } _ { 0 } ^ { \mathrm { M C } } [ E ] : = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \mathbf { 1 } _ { E } \big ( X _ { 0 } ^ { n } \big ) , \qquad \mathbb { E } \big [ \hat { p } _ { 0 } ^ { \mathrm { M C } } [ E ] \big ] = p _ { 0 } [ E ] .\tag{3}
$$

However, its relative variance is $( 1 - p _ { 0 } [ E ] ) / ( N p _ { 0 } [ E ] )$ , so when the event is rare, an impractical number $N \propto 1 / p _ { 0 } [ E ]$ of runs are needed for a fixed relative error (Rubino et al., 2009).

## 3 RARE-EVENT SAMPLING AND PROBABILITY ESTIMATION

Notation. For a measure $\textstyle \mu , \mu [ f ] : = \int f ( x ) \mu ( \mathrm { d } x )$ for a function $f$ and $\mu [ A ] : = \mu [ \mathbf { 1 } _ { A } ]$ for a set $A .$ Square brackets denote action on a function or set, and parentheses a density or kernel evaluation.

We seek to construct a low variance estimator of $p _ { 0 } [ E ]$ with the same expectation as Eq. (3). We do so by reward tilting (Rubino et al., 2009; Uehara et al., 2025). A user-specified reward $\bar { r } _ { 0 } : \mathbb { R } ^ { d } $ R biases the base density to $\gamma _ { 0 } ( x ) : = p _ { 0 } ( x ) e ^ { r _ { 0 } ( x ) }$ with a normalised form given by $\pi _ { 0 } : = \gamma _ { 0 } / \gamma _ { 0 } [ 1 ]$ Since $p _ { 0 } = \gamma _ { 0 } e ^ { - r _ { 0 } }$ , both samples and estimates are functionals of the tilted law,

$$
p _ { 0 } [ E ] = \gamma _ { 0 } \left[ e ^ { - r _ { 0 } } \mathbf { 1 } _ { E } \right] , \qquad p _ { 0 } ( x \mid E ) = e ^ { - r _ { 0 } ( x ) } \mathbf { 1 } _ { E } ( x ) \pi _ { 0 } ( x ) / \pi _ { 0 } \left[ e ^ { - r _ { 0 } } \mathbf { 1 } _ { E } \right] .\tag{4}
$$

In the remainder of the section, we detail how to efficiently sample from $p _ { 0 } ( \cdot | E )$ , construct an estimator $\hat { p } _ { 0 } [ E ]$ using Sequential Monte Carlo (SMC), and design a reward that reduces its variance. While we present the following for discretised transition kernels, DireSMC holds automatically for the continuous time construction of a diffusion model.

## 3.1 STEERING THE DIFFUSION MODELS TOWARDS RARE EVENTS

The marginals of the diffusion model, $p _ { k }$ , tilted by the reward $r _ { k } .$ , define a sequence of densities,

$$
\gamma _ { k } ( x ) : = p _ { k } ( x ) e ^ { r _ { k } ( x ) } , \qquad \pi _ { k } ( x ) : = \gamma _ { k } ( x ) / \gamma _ { k } [ 1 ] , \qquad k = K , \ldots , 0 ,\tag{5}
$$

pinned at both ends. $\mathrm { A t } \ k = 0 ( \mathrm { d a t a } )$ , we set $r _ { 0 }$ to be the reward of Eq. (4), and at $k = K$ (noise) we impose $r _ { K } = 0$ , such that $\gamma _ { K } = p _ { \mathrm { r e f } }$ . We defer reward path $r _ { K - 1 } , \ldots , r _ { 1 }$ selection to Section 3.2.

Sequential Monte Carlo Sampler. We sample from the sequence of marginals of Eq. (5) using a sequential Monte Carlo sampler (Del Moral et al., 2006; Doucet et al., 2022; Vargas et al., 2023). We refer readers to App. C.2 for its general formulation; below we give a version adapted to our setting. Firstly, initialise N particles, $X _ { K } ^ { n } \sim p _ { \mathrm { r e f } }$ with weights $w _ { K } ^ { n } = 1$ . At each step $k , ( \mathrm { i } )$ propagate each particle through a backward proposal kernel, $\overline { { Q } } _ { k - 1 | k }$ , and (ii) update its weight by,

$$
X _ { k - 1 } ^ { n } \sim \bar { Q } _ { k - 1 | k } ( \cdot \mid X _ { k } ^ { n } ) , \qquad w _ { k - 1 } ^ { n } = w _ { k } ^ { n } g _ { k - 1 | k } \big ( X _ { k - 1 } ^ { n } , X _ { k } ^ { n } \big ) ,\tag{6}
$$

where $g _ { k - 1 | k }$ is the incremental weight function built from an auxiliary forward kernel $\vec { F } _ { k \vert k - 1 }$

$$
g _ { k - 1 | k } ( x _ { k - 1 } , x _ { k } ) = \left( \frac { \exp \left( r _ { k - 1 } ( x _ { k - 1 } ) \right) } { \exp \left( r _ { k } ( x _ { k } ) \right) } \right) \cdot \left( \underbrace { \frac { p _ { k - 1 } ( x _ { k - 1 } ) } { p _ { k } ( x _ { k } ) } } _ { \mathrm { i n t r a c t a b l e } } \cdot \overbrace { Q _ { k - 1 | k } ( x _ { k - 1 } | \ x _ { k } ) } ^ { \vec { F } _ { k | k - 1 } ( x _ { k } \mid x _ { k - 1 } ) } \right) .\tag{7}
$$

The first factor is the ratio of reward-tilts at consecutive steps; the second is a likelihood ratio of the base process, involving the marginals $p _ { k }$ we cannot evaluate. The choice of both kernels is flexible, provided we can sample from $\overline { { Q } } _ { k - 1 | k }$ and evaluate $g _ { k - 1 | k }$ pointwise (Del Moral et al., 2006). Optionally, when the effective sample size, $\mathrm { E S S } ( w _ { k - 1 } ) : = \| \dot { w } _ { k - 1 } \| _ { 1 } ^ { 2 } / \| w _ { k - 1 } \| _ { 2 } ^ { 2 }$ drops below a threshold $\eta N$ for $\eta \in [ 0 , 1 ]$ , we can resample, replacing each particle by an ancestor drawn in proportion to its weight and resetting the weights to the average weights of the ancestors. This is done through setting $X _ { k - 1 } ^ { n ^ { \circ } }  X _ { k - 1 } ^ { a ^ { n } }$ and $\begin{array} { r } { w _ { k - 1 } ^ { n } \stackrel {  } {  } \frac { 1 } { N } \sum _ { n } w _ { k - 1 } ^ { n } } \end{array}$ with the ancestor index $a ^ { n }$ drawn, e.g. systematically, from P $\mathsf { \Omega } \cdot [ a ^ { n } = m \mid w _ { k - 1 } ] \propto w _ { k - 1 } ^ { m }$ (Chopin et al., 2022). As resampling duplicates high-weight particles and depletes diversity, we follow it with a rejuvenation step, $X _ { k - 1 } ^ { n } \sim M _ { k - 1 } ( \cdot \ |$ $X _ { k - 1 } ^ { n } )$ , using a $\scriptstyle \cdot \ \pi _ { k - 1 }$ -invariant MCMC kernel $M _ { k - 1 }$ . We use the Metropolis-adjusted diffusion model (MADM) kernel of (Lam et al., 2026), which approximates a Metropolis-adjusted Langevin kernel using only the score and $r _ { k } ;$ both quantities that we already have access to $( \mathrm { A p p . ~ I } )$

Choice of Kernels. We construct our kernels to, (i) circumvent the intractable marginals and (ii) concentrate samples on E. For the first, we set $\vec { F } _ { k \vert k - 1 }$ as the time reversal of the base chain,

$$
\begin{array} { r } { \vec { F } _ { k | k - 1 } ( x _ { k } \mid x _ { k - 1 } ) : = p _ { k } ( x _ { k } ) \stackrel {  } { P } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) / p _ { k - 1 } ( x _ { k - 1 } ) , } \end{array}\tag{8}
$$

where $\stackrel {  } { P } _ { k - 1 | k }$ is the reverse diffusion kernel. This removes the intractable ratio and is a Markov kernel by the marginal recursion of $p _ { k - 1 }$ of Section 2. Our backward proposal, $\overline { { Q } } _ { k - 1 | k }$ is defined by adding the gradient of the reward, or the guidance, to the reverse drift of Eq. (1),

$$
\begin{array} { r } { d X _ { t } = \left[ \mu _ { t } ( X _ { t } ) - \sigma _ { t } ^ { 2 } \left( \nabla _ { x } \log p _ { t } ( X _ { t } ) + \nabla _ { x } r _ { t } ( X _ { t } ) \right) \right] d t + \sigma _ { t } d \bar { W } _ { t } , } \end{array}\tag{9}
$$

so that steering is exactly a tilt of the score, $\nabla _ { x }$ log $p _ { k } \to \nabla _ { x } \log p _ { k } + \nabla _ { x } r _ { k }$ and ${ \bar { Q } } _ { k - 1 | k }$ and $\stackrel {  } { P } _ { k - 1 | k }$ share the same integration steps but differ in the score. In our construction the guidance strength is defined within $r _ { k }$ (Uehara et al., 2025). Substituting into Eq. (7) gives the update function,

$$
g _ { k - 1 | k } ( x _ { k - 1 } , x _ { k } ) = { \frac { e ^ { r _ { k - 1 } ( x _ { k - 1 } ) } } { e ^ { r _ { k } ( x _ { k } ) } } } \cdot { \frac { { \overline { { P } } } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } { { \overline { { Q } } } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } } .\tag{10}
$$

Computing the Incremental Weights. Eq. (10) requires the ratio $\stackrel {  } { P } _ { k - 1 \mid k } / \stackrel {  } { Q } _ { k - 1 \mid k }$ pointwise. For one-step explicit integrators (Milstein & Tretyakov, 2004) with Gaussian transitions and a mean affine in the score, guidance shifts the mean but leaves the covariance invariant. So, for integrator dependent scalars $c _ { k } , v _ { k } > 0$

$$
\begin{array} { r l } & { \bar { P } _ { k - 1 | k } ( \mathrm { d } x _ { k - 1 } | x _ { k } ) = \mathcal { N } \big ( m _ { k } ( x _ { k } ) , v _ { k } I \big ) , } \\ & { \bar { Q } _ { k - 1 | k } ( \mathrm { d } x _ { k - 1 } | x _ { k } ) = \mathcal { N } \big ( m _ { k } ( x _ { k } ) + c _ { k } \nabla _ { x } r _ { k } ( x _ { k } ) , v _ { k } I \big ) . } \end{array}\tag{11}
$$

This holds for first-order stochastic integrators such as Euler-Maruyama or SEEDS-1 of (Gonzalez et al., 2023), but not for higher-order schemes that evaluate the score at a noisy intermediate point, e.g. the stochastic Heun integrator of (Karras et al., 2022).

Lemma 1. Under Eq. (11), let $\zeta _ { k } : = c _ { k } / \sqrt { v _ { k } } , \epsilon _ { k } \sim { \mathcal N } ( 0 , I )$ for sampling $x _ { k - 1 }$ from ${ \bar { Q } } _ { k - 1 | k }$ , then

$$
\begin{array} { r } { \log g _ { k - 1 | k } ( x _ { k - 1 } , x _ { k } ) = r _ { k - 1 } ( x _ { k - 1 } ) - r _ { k } ( x _ { k } ) - ( \zeta _ { k } ^ { 2 } / 2 ) \left. \nabla _ { x } r _ { k } ( x _ { k } ) \right. ^ { 2 } - \zeta _ { k } \langle \nabla _ { x } r _ { k } ( x _ { k } ) , \epsilon _ { k } \rangle . } \end{array}\tag{12}
$$

The proof is given in App. A; we use the analytical form for a ratio of Gaussians with a shared covariance alongside a discrete Cameron-Martin correction for the guidance term (Øksendal, 2003). The lemma states that the choice of numerical integrator only enters the weights via the scalar $\zeta _ { k }$ Additionally, tracking the weights requires no additional network calls as both $\nabla _ { x } r _ { k }$ and $\epsilon _ { k }$ are readily available when the particle is steered. Under this formulation, for the Euler-Maruyama integrator with a VP-SDE noise schedule: $c _ { k } = v _ { k } = \sigma _ { t _ { k } } ^ { 2 } \Delta t _ { k }$ gives $\zeta _ { k } = \sigma _ { t _ { k } } \sqrt { \Delta t _ { k } }$ , and Eq. (12) recovers the weights of (He et al., 2026). App. D.1 derives $\zeta _ { k }$ for different integrators.

Estimators. With the weights reset to the average at each resampling, the particle system at $k = 0$ is a weighted approximation of $\gamma _ { 0 } .$ , and substituting the weights into Eq. (4) gives the two outputs we are after: the probability estimate $\hat { p } _ { 0 } [ E ] = \hat { \gamma } _ { 0 } [ e ^ { - r _ { 0 } } \mathbf { 1 } _ { E } ]$ and weighted samples from the conditional,

$$
\hat { p } _ { 0 } [ E ] = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } w _ { 0 } ^ { n } e ^ { - r _ { 0 } ( X _ { 0 } ^ { n } ) } \mathbf { 1 } _ { E } ( X _ { 0 } ^ { n } ) , \hat { p } _ { 0 } [ \mathrm { d } x \mid E ] = \frac { \sum _ { n = 1 } ^ { N } w _ { 0 } ^ { n } e ^ { - r _ { 0 } ( X _ { 0 } ^ { n } ) } \mathbf { 1 } _ { E } ( X _ { 0 } ^ { n } ) \delta _ { X _ { 0 } ^ { n } } [ \mathrm { d } x ] } { \sum _ { n = 1 } ^ { N } w _ { 0 } ^ { n } e ^ { - r _ { 0 } ( X _ { 0 } ^ { n } ) } \mathbf { 1 } _ { E } ( X _ { 0 } ^ { n } ) } .\tag{13}
$$

Theorem 1. Suppose $\overline { { P } } _ { k - 1 | k } \ll \overline { { Q } } _ { k - 1 | k }$ for all $k ,$ and the rejuvenation kernels $M _ { k }$ are $\pi _ { k }$ -invariant. Then,for any $K , N ,$ , rewards $r _ { 0 } , \ldots , \underbrace { r _ { K } }$ with $r _ { K } = 0$ and $\gamma _ { 0 } [ 1 ] < \infty ,$ , and any adaptive resampling schedule, $\mathbb { E } [ \hat { p } _ { 0 } [ E ] ] = p _ { 0 } [ E ] = \mathbb { E } [ \hat { p } _ { 0 } ^ { \mathrm { M } } \bar { \mathrm { C } } [ E ] ]$ ], and ${ \hat { p } } _ { 0 } [ \cdot \mid E ]$ converges almost surely to $p _ { 0 } [ \cdot \ | \ \hat { E } ]$ as $N \to \infty$ . Moreover, ifthe score is exact and $p _ { \mathrm { d a t a } } [ \partial \dot { E } ] = \mathrm { \bar { 0 } }$ , then $\begin{array} { r } { \mathbb { E } [ \hat { p } _ { 0 } [ E ] ]  p _ { \mathrm { d a t a } } [ E ] } \end{array}$ as $K  \infty ,$

Algorithm 1 DireSMC: Steering diffusion models to rare events.   
Require: pretrained diffusion model, transition kernels, $\overline { { \boldsymbol { P } _ { k - 1 | k } } }$ and $\widehat { Q } _ { k - 1 | k }$ , rare event E, reward   
path $r _ { k } ,$ particles N, discretisation schedule $1 = t _ { K } > \cdot \cdot \cdot > t _ { 0 } = 0 ,$ ESS threshold $\eta \in [ 0 , 1 ]$   
1: $\mathbf { \dot { \boldsymbol { X } } } _ { K } ^ { n } \sim \dot { p _ { \mathrm { r e f } } } , w _ { K } ^ { n } \gets 1$ for $n = 1 , \ldots , N ,$   
2: for $k = K , \dotsc , 1$ do   
3: $X _ { k - 1 } ^ { n } \sim \bar { Q } _ { k - 1 | k } ( \mathrm { d } x _ { k - 1 } \mid X _ { k } ^ { n } )$ ▷ One integrator step with the guided score Eq. (11)   
4: $w _ { k - 1 } ^ { n }  w _ { k } ^ { n } g _ { k - 1 | k } ( X _ { k - 1 } ^ { n } , X _ { k } ^ { n } )$ ▷ Update weights using Eq. (12)   
5: if $\mathrm { E S S } ( w _ { k - 1 } ) < \eta N$ then   
6: $\begin{array} { r } { \bar { w }  \frac { 1 } { N } \sum _ { m } w _ { k - 1 } ^ { m } ; } \end{array}$   
7: $X _ { k - 1 } ^ { n }  X _ { k - 1 } ^ { a ^ { n } } , w _ { k - 1 } ^ { n }  \bar { w }$ ▷ Draw $a ^ { n }$ using systematic resampling   
8: $X _ { k - 1 } ^ { n } \sim M _ { k - 1 } ( \mathrm { d } x _ { k - 1 } \mid X _ { k - 1 } ^ { n } )$ ▷ Rejuvenate with a $\pi _ { k - 1 }$ invariant MCMC kernel   
9: end if   
10: end for   
11: return $\hat { p } _ { 0 } [ E ]$ and $\hat { p } _ { 0 } [ \mathrm { d } x \mid E ]$ from Eq. (13)

The proof is given in App. A and follows from the unbiasedness of the SMC normalisation constant estimator under adaptive resampling and $\pi _ { k - 1 }$ invariant moves (Del Moral et al., 2012; Syed et al., 2026). The estimator is well suited to the rare-event setting as it remains unbiased for any reward, however poorly it may steer. The normalisation constant, $\hat { p } _ { 0 } [ E ]$ is accumulated along the denoising process rather than estimated separately (Del Moral et al., 2006; Doucet et al., 2022; Vargas et al., 2023) (App. C.3). The rewards only affects the variance, which is the subject of Section 3.2 below.

## 3.2 REWARD SELECTION

Theorem 1 shows $\hat { p } _ { 0 } [ E ]$ is unbiased independent of reward choice, however, a poorly chosen steering inflates the variance of $\hat { p } _ { 0 } [ E ]$ to an impractical number of required samples (Chatterjee & Diaconis, 2018). We construct a reward path and associated transition kernels that reduce the variance of $\hat { p } _ { 0 } [ E ]$ . Let $\mathbb { P } _ { 0 : K }$ and $\mathbb { Q } _ { 0 : K }$ be the joint laws over the base and guided chains, $( X _ { 0 } , \ldots , X _ { K } )$ . The ideal law to simulate from is the base chain reweighted by $e ^ { r _ { 0 } \left( X _ { 0 } \right) } / \gamma _ { 0 } [ 1 ]$ , since its endpoint is $\pi _ { 0 }$ This is Doob’s h-transform, $\mathbb { P } _ { 0 : K } ^ { \star }$ of $\mathbb { P } _ { 0 : K }$ (Dai Pra, 1991; Denker et al., 2024). Writing ${ \bar { P } } _ { 0 \mid k }$ as the denoising posterior of $X _ { 0 }$ given $X _ { k }$ , the optimal reward and transitions are

$$
\begin{array} { c } { { r _ { k } ^ { \star } ( x _ { k } ) : = \log \displaystyle \int \bar { \cal P } _ { 0 \mid k } ( \mathrm { d } x _ { 0 } \mid x _ { k } ) e ^ { r _ { 0 } ( x _ { 0 } ) } , } } \\ { { { \bar { \cal P } } _ { k - 1 \mid k } ^ { \star } ( \mathrm { d } x _ { k - 1 } \mid x _ { k } ) : = { \bar { \cal P } } _ { k - 1 \mid k } ( \mathrm { d } x _ { k - 1 } \mid x _ { k } ) e ^ { r _ { k - 1 } ^ { \star } ( x _ { k - 1 } ) - r _ { k } ^ { \star } ( x _ { k } ) } , } } \end{array}\tag{14}
$$

and its marginal at step k is $\pi _ { k } ^ { \star } \propto p _ { k } e ^ { r _ { k } ^ { \star } }$ , with $\pi _ { 0 } ^ { \star } = \pi _ { 0 } ( \mathrm { A p p . ~ A } )$ . While we refer to $r _ { k } ^ { \star }$ as a reward for simplicity, it is also known as the value function in stochastic optimal control; see (Berner et al., 2024). The discrepancy between $\mathbb { P } _ { 0 : K } ^ { \star }$ and $\mathbb { Q } _ { 0 : K }$ determines the variance of our estimator.

Theorem 2. Suppose there is no resampling $( \eta = 0 ) , \bar { P } _ { k - 1 | k } \ll \bar { Q } _ { k - 1 | k }$ for all k, and $e ^ { r _ { 0 } } \geq c$ on E for some $c \in ( 0 , 1 ]$ . Then

$$
\frac { \mathrm { V a r } \big [ \widehat { p } _ { 0 } [ E ] \big ] } { p _ { 0 } [ E ] ^ { 2 } } \le \frac { 1 } { N } \left[ \Big ( \frac { \gamma _ { 0 } [ 1 ] } { c p _ { 0 } [ E ] } \Big ) ^ { 2 } \Big ( 1 + \chi ^ { 2 } \big ( { \mathbb P } _ { 0 : K } ^ { \star } \| { \mathbb Q } _ { 0 : K } \big ) \Big ) - 1 \right] ,\tag{15}
$$

with equality when $e ^ { r _ { 0 } } = c { \bf 1 } _ { E } ,$ , in which case $\mathbb { P } _ { 0 : K } ^ { \star }$ is $\mathbb { P } _ { 0 : K }$ conditioned on $X _ { 0 } \in E _ { \ O }$ . Moreover, the Doob transitions minimise Pearson’s $\chi ^ { 2 }$ divergence: for any $\begin{array} { r l } { \bar { Q } _ { k - 1 | k } , \ \chi ^ { 2 } ( \mathbb { P } _ { 0 : K } ^ { \star } \parallel \mathbb { Q } _ { 0 : K } ) } & { { } \geq } \end{array}$ $\chi ^ { 2 } ( \pi _ { K } ^ { \star } \| p _ { \mathrm { r e f } } )$ with equality when $\stackrel {  } { Q } _ { k - 1 | k } = \stackrel {  } { P } _ { k - 1 | k } ^ { \star }$ for all k. The minimum is at most $\chi ^ { 2 } ( \pi _ { 0 } \| p _ { 0 } )$ and it is zero $i f X _ { K }$ is independent of $X _ { 0 }$ under $\mathbb { P } _ { 0 : K }$

The proof is in App. A and follows from the chain rule of measures and the data-processing inequality. The bound separates two design choices: (1) the endpoint reward affects $\gamma _ { 0 } [ 1 ] / \bar { c } p _ { 0 } [ E ]$ , which equals to 1 when $e ^ { r _ { 0 } } \equiv 1 _ { E } ; ( 2 )$ the second factor is minimised when the optimal Doob’s transitions of Eq. (14) are used throughout the generation process. To elaborate, suppose $e ^ { r _ { 0 } } = c 1 _ { E } .$ , then the two distinct effects materialise. First, turning the steering off, the bound is exact with a relative variance equalling the Monte Carlo estimator keeping the first term. Second, if the Doob’s transition kernels are instead used, then the relative variance is the initialisation error between the ideal starting distribution, $\pi _ { K } ^ { \star }$ and the reference $p _ { \mathrm { r e f } }$ . The bound also gives us a practical target. With resampling turned off, we have $\mathrm { E S S } / N {  } 1 / \bar { ( } 1 + { \chi } ^ { 2 } ( { \mathbb { P } } _ { 0 : K } ^ { \star } | | { \mathbb { Q } } _ { 0 : K } ) \bar { ) }$ as $N \to \infty$ (Agapiou et al., 2017) which allows us to evaluate the quality of the simulated trajectories. Under resampling and sufficient rejuvenation, a more elaborate error analysis is needed, see Theorem 1(a) of (Syed et al., 2026). Intuitively, errors from a poor transition kernel $\overline { { Q } } _ { k - 1 | k }$ can be pruned by resampling instead of compounding.

Rewards for threshold exceedance events. We now construct $r _ { 0 }$ and $r _ { 1 } , \ldots , r _ { K - 1 }$ for events of the form $E = \{ x : \phi ( x ) \geq \tau \}$ , for an observable $\phi : \mathbb { R } ^ { d }  \mathbb { R }$ and threshold τ; App. E.1 extends $r _ { 0 }$ and $r _ { k }$ to interval and multi-constraint events. Theorem 2 identifies the ideal terminal reward $e ^ { r _ { 0 } } = c 1 _ { E }$ . However, it lacks the regularity needed to be steered with. We require a finite continuously differentiable reward so the guidance $\nabla _ { x } { r _ { t } }$ is well defined, and whoese gradient is non-zero outside $E ,$ , so particles outside the event set are actively pushed towards it. Further, we require that $r _ { 0 }$ is bounded below $E ,$ so that the factor $\gamma _ { 0 } [ 1 ] / c p _ { 0 } [ \dot { E } ]$ in Theorem 2 is controlled. Any CDF Φ with an everywhere-positive density gives such a regularity; writing $\delta > 0$ for a smoothing width,

$$
\begin{array} { r } { r _ { 0 , \delta } ( x ) : = \log \Phi \big ( ( \phi ( x ) - \tau ) / \delta \big ) , } \end{array}\tag{16}
$$

with $\begin{array} { r } { e ^ { r _ { 0 , \delta } } \ge \Phi ( 0 ) = \frac { 1 } { 2 } } \end{array}$ on $E .$ So $\begin{array} { r } { c = \frac { 1 } { 2 } } \end{array}$ in Theorem 2, and the induced $\pi _ { 0 }$ tends to $p _ { 0 } [ \cdot \mid E ]$ as $\delta \downarrow 0$ We adopt the Gaussian CDF because, for a linear observable $\phi ( x ) = a ^ { \top } x + b .$ , it is closed under convolution with a Gaussian (Rasmussen & Williams, 2006). See Fig. 5 for an illustration.

The reward sequence. Fixing ${ r } _ { 0 , \delta }$ leaves the Doob’s rewards of Eq. (14) intractable through the posterior ${ \overline { { P } } } _ { 0 \mid k }$ . Following (Chung et al., 2023; Song et al., 2023), we approximate it by a Gaussian centred at its mean $\hat { x } _ { 0 \mid k } ( x _ { k } ) = \operatorname { \mathbb { E } } [ X _ { 0 } \mid X _ { k } = x _ { k } ]$

$$
\begin{array} { r } { \bar { P } _ { 0 | k } ( \mathrm { d } x _ { 0 } \mid x _ { k } ) \approx \mathcal { N } \big ( \hat { x } _ { 0 | k } ( x _ { k } ) , \Sigma _ { k } \big ) , } \end{array}\tag{17}
$$

with ${ \hat { x } } _ { 0 \mid k }$ approximated by Tweedie’s formula from the score at $t _ { k }$ (Efron, 2011) (Eq. (42)) and $\Sigma _ { k }$ is approximated, e.g. via a diagonal ΠGDM posterior-covariance proxy (Song et al., 2023); we give the explicit forms of both in App. B. For the Gaussian-CDF reward and a linear observable ϕ, the resulting integral of $\operatorname { E q . }$ (14) is closed-form, $r _ { k } ^ { \star } ( x _ { k } )$ ≈ log Φ( $\left( \phi ( \hat { x } _ { 0 \mid k } ( x _ { k } ) ) - \tau \right) / \sqrt { \delta ^ { 2 } + a ^ { \top } \Sigma _ { k } a } )$ with the smoothing width inflated by the convolution (App. E). The approximation is accurate near the data end and poor at high noise, where ${ \hat { x } } _ { 0 \mid k }$ carries little information about $X _ { 0 }$ , and it does not vanish at $k = K$ as the boundary condition requires. We therefore temper $r _ { k } ^ { \star }$ by a guidance strength $\lambda _ { k } \geq 0$ , which switches the guidance on gradually as the look-ahead becomes reliable, and optionally approach the event in stages through a threshold $\tau _ { k }$ . This gives us our reward sequence,

$$
r _ { k } ( x _ { k } ) = \lambda _ { k } \log \Phi \Big ( \big ( \phi ( \hat { x } _ { 0 \mid k } ( x _ { k } ) ) - \tau _ { k } \big ) \Big / \sqrt { \delta ^ { 2 } + a ^ { \top } \Sigma _ { k } a } \Big ) ,\tag{18}
$$

pinned at both ends as Eq. (5) requires: $\lambda _ { 0 } = 1 , \tau _ { 0 } = \tau \mathrm { g i v e } r _ { 0 } = r _ { 0 , \delta }$ and $\lambda _ { K } = 0$ gives $r _ { K } = 0$ The schedules $\lambda _ { k }$ and $\tau _ { k }$ are the two parameters of our method; App. C relates them to classical rare-event samplers. By Theorem 1, none of these approximations affect what $\hat { p } _ { 0 } [ E ]$ estimates; by Theorem 2, they determine the variance of our estimator.

## 3.3 RELATED WORK

Several works use a pre-trained diffusion sampler to target a tilted distribution without retraining (Berner et al., 2024; Chung et al., 2023; Denker et al., 2024; Song et al., 2023; Uehara et al., 2025), but these do not compute probabilities. The weight construction combining diffusion models and SMC samplers (He et al., 2026; Singhal et al., 2025; Skreta et al., 2025; Wu et al., 2023) is common, but our contribution is to apply it to rare-event sampling and probability estimation. In diffusion-based molecular simulations, enhanced rare-event sampling ideas such as metadynamics and umbrella sampling have been adapted (Xie et al., 2026) to estimate the occupancy of a rare metastable state. Both methods track importance weights however, Xie et al. (2026) assemble $p _ { 0 } [ E ]$ by combining a sequence of events $E _ { i }$ with MBAR (Shirts $\&$ Chodera, 2008), rather than estimating $\dot { p } _ { 0 } [ E ]$ directly. Manshausen et al. (2026) use a score-based climate emulator guidance to generate Tropical Cyclone (TC) events and use importance sampling. An ODE sampler samples the target rare event, followed by two passes of a guided and unguided Probability Flow ODE to estimate its importance weights (Song et al., 2021b). We instead estimate $p _ { 0 } [ E ]$ by accumulating weights from a single denoising pass and benefit from resampling, which is unavailable to them; see App. J for a comparison.

## 4 EXPERIMENTS

We validate DireSMC in two settings: a correlated two-dimensional Gaussian mixture model (GMM) and a pretrained score-based climate emulator. For the GMM, the score and guidance are known analytically so our method may be evaluated in the absence of approximation errors in addition to a learned score, see App. F. The second is a pretrained score-based climate emulator for estimating extreme compound weather events, demonstrating the scalability and applicability of our method. Estimates are evaluated against the sampled law, $p ^ { \star }$ : closed-form $p _ { 0 } [ E ]$ for the analytical score or a large MC for the learned score; the same references define the sample-quality metrics. The ratio $\hat { p } _ { 0 } [ E ] / p ^ { * }$ measures the bias and the estimator coefficient of variation, $\mathrm { C V } = { \sqrt { \mathrm { V a r } ( { \hat { p } } _ { 0 } [ E ] ) } } / { p ^ { \star } }$ measures the relative variance computed over independent runs.

## 4.1 CORRELATED GAUSSIAN MIXTURE MODEL (GMM)

![](images/1939dd49766aac76e7501076d3a299951e4fab420dd57f3960ccc11d42b83351.jpg)  
Figure 2: Left: CV plotted against N for $\mathrm { a } p _ { 0 } [ x _ { 1 } > 1 0 ]$ problem at a fixed K. Rejuvenation tracks the lowest CV across all N. Middle: we compare our estimator against Manshausen et al. (2026) across rarities at a compute-normalised N. We consistently track a lower CV. Right: measured bias scaling with rarity using exact Doob’s steering and analytical scores, highlighting the bias of Manshausen et al. (2026), DireSMC tracks the floor, see Tables 23 and App. J.1.2.

Let p be the seven-mode GMM shown in Figure 6 and defined in Table 4. Let $\phi ( x ) = x _ { 1 }$ and define $E = \left\{ x _ { 1 } \geq \tau \right\}$ for $\tau \in \{ 7 , 8 , 9 , 1 0 \}$ . This varies the analytic mass $p _ { 0 } [ E ]$ from $6 . 9 { \times } 1 0 ^ { - 3 }$ to $1 . 4 { \times } 1 0 ^ { - 5 }$ defining a sequence of increasingly rare events. Unless stated otherwise $N = 1 0 0 0 , \delta = 0 . 0 1$ $\lambda ( t ) = \bar { 1 } - ( t / \bar { 0 . 9 } ) ^ { 2 }$ with $\lambda ( t ) = 0$ for $t > 0 . 9 , \tau ( t ) = \tau$ and $K = 1 0 0 0$ , using the SEEDS-1 integrator guided by the reward of Eq. (18); where $\Sigma _ { t }$ is the ΠGDM proxy (App. B). We run Algorithm 1 with adaptive resampling at $\mathrm { E S S } / N < 0 . 5 ,$ followed by three MADM rejuvenation sweeps (Lam et al., 2026) gated to a time $t \geq 0 . 3 0$ . Experiments are replicated over 24 random seeds and validated against a 200 million MC reference pool (App. G).

Our method recovers the reference exceedance probability for both analytical and learned scores (full per-threshold results and sample-quality metrics in Table 7, App. G.3). For a fixed $\tau = 1 0$ threshold, Fig. 2 shows that the relative variance decreases as $O ( 1 / \sqrt { N } )$ (Chatterjee & Diaconis, 2018). Resampling and rejuvenation prevent accumulation of guidance mismatch (Del Moral et al., 2006; 2012) and the CV stays flat across rarities for a fixed N.

DireSMC compares favourably against the importance sampling estimator of Manshausen et al. (2026). Fig. 2 (left) shows that DireSMC with resampling and MADM leads to lower CV for the same number of particles compared to their method, even though their method requires 3.5× as many function evaluations as us. In this setting, rejuvenation incurs an additional 1.6% network calls over the base, which we also discount. On a compute normalised particle budget, all variants of our estimator achieve a lower variance across all rarities as seen in Fig. 2 (cost detailed in App. J and Table 24). Additionally, the estimator of Manshausen et al. (2026) assumes that the law induced by an explicit integrator is reversible. However, this often does not hold in practice, leading to a discretisation bias which decreases with step count K. To isolate the effects of discretisation bias, we ran experiments with the optimal Doob’s transition kernels of $\operatorname { E q . }$ (14) such that the true $p ^ { * }$ is known and any artefacts due to approximate guidance or score estimates are removed. We also provide the PF-ODE estimator with the true Jacobian trace term, which would otherwise have to be estimated.

![](images/c459505197f9f3f5f70641797a071aa51baa48ef6ed006a6844698c04ff28685.jpg)  
Figure 3: Surface-temperature anomaly (K) for July at $\Delta T = + 1 . 5 $ K over the Pacific Northwest (PNW) target region. $L e f t .$ : average tas field over $1 0 ^ { 7 }$ samples. Remaining panels: the highestweight crossing particle produced by the guided sampler at different exceedance thresholds τ and regional average temperature $\phi ( \cdot )$ of the sample. $p ^ { \star }$ are MC estimates; $\hat { p }$ is obtained from the guided estimator; full results in Table 15. The variance ratio (Var. Ratio) at matched compute is defined as $\mathrm { V a r } ( \hat { p } _ { \mathrm { u n g u i d e d } } ) / \mathrm { V a r } ( \hat { p } _ { \mathrm { g u i d e d } } )$

The results in Fig. 2 (right) show their discretisation bias is significantly larger than DireSMC’s with the bias decreasing with increasing K (cf. Table 23). We acknowledge that our method also incurs a discretisation bias which we characterise below.

Ablations. We isolate both the estimator bias and variance introduced by SDE integrators (Euler-Maruyama, SEEDS-1 (Gonzalez et al., 2023)) and the DDIM ODE integrator (Song et al., 2021a) used in Manshausen et al. (2026). Evaluating with both analytical and learned scores, the discretisation bias decreases approximately as $O ( 1 / K )$ (see Fig. 7, 9 in App. G.2.2). For a fixed K we find the discretisation bias increases with rarity, highlighting the sensitivity of rare-event probability estimation to small errors. Both SDE integrators track a lower bias compared to DDIM which we postulate is due to the lower overall error in the generation process, see Theorem 3 of Xu et al. (2023). Moreover, we characterise two sources of errors due to the MADM MCMC kernel (Lam et al., 2026). It uses the $\nabla _ { x } \log { p _ { t = t _ { k } } }$ to estimate Metropolis Hastings (MH) acceptance ratio, hence targets the continuous-time marginals rather than the law $\gamma _ { k }$ induced by the integrator. This introduces an error decaying as $O ( 1 / K )$ to leading order (Fig. 13). Additionally, MADM assumes access to a conservative score to compute the ratio which degrades for a learned score network as $t \to 0$ (Vuong et al., 2025), see App. G.3.3 for details. Heuristically, we gate rejuvenation to $t ^ { * } > 0$ to account for this assumption (Table 9). Finally, we show that the method translates without modification to interval and multi-constraint events (Table 10).

## 4.2 GENERATING HEATWAVES WITH PRE-TRAINED CLIMATE MODEL EMULATORS

Next, we validate on the problem of estimating how the probability of extreme events evolves under a changing climate. The high cost of running physics-based climate models (Eyring et al., 2016) has motivated the practice of training cheap emulators on climate model output (Watson-Parris et al., 2022). However, as we will show, even with these cost-savings, it can be difficult to generate targeted extreme events without explicit steering.

Model. We leverage the pre-trained climate emulator of Bouabid et al. (2026) which is trained on data generated from the MPI-ESM1-2-LR model (Mauritsen et al., 2019) and released as part of the Coupled Model Intercomparison Project Phase 6 (CMIP6) (Eyring et al., 2016). The emulator generates global monthly-mean fields of near-surface temperature (tas), precipitation $( \mathtt { p r } )$ , relative humidity (hurs) and wind speed $( s \mathtt { f c w i n d } )$ , conditioned on the global mean surface temperature anomaly, ∆T, against a pre-industrial reference and the month of the year, m, to capture annual variability<sup>1</sup>; we refer to Bouabid et al. (2026) for an in-depth motivation of the emulator setup.

Setup. We consider extremes in the temperature and humidity variables averaged within the region R defined by the bounding box $4 2 ^ { \circ } - 5 2 ^ { \circ } \bar { \bf N }$ latitude and $2 3 5 ^ { \circ } { \cdot } 2 \dot { 4 } 5 ^ { \circ } \mathrm { E }$ longitude, roughly corresponding to the region affected by the 2021 Pacific Northwest (PNW) heatwave (McKinnon & Simpson, 2022). We use $N = 1 0 2 4$ particles with $K = 2 0 0$ discretisation steps using the exponential integrator; error/variance metrics are computed using estimates from 16 seeds. We generate $\sim 1 0 ^ { 7 }$ Monte Carlo samples to estimate ground-truth event probabilities. Further model details and hyperparameters are presented in App. H. We select the hyperparameters once from a pilot sweep on the marginal hot and dry $\tau _ { 1 }$ experiments (detailed below) and use them for all of our other experiments; proper per-problem tuning would lead to a larger variance reduction.

Extreme temperatures. We first verify how accurately DireSMC can estimate the probability of extreme temperatures relative to crude Monte Carlo. Assuming $\Delta T = 1 . 5 \ : \mathrm { K }$ , for three different temperature thresholds $\tau _ { 1 } , \tau _ { 2 } , \tau _ { 3 }$ (exact values in App. H) we calculate the probability that the average surface temperature in the PNW exceeds a temperature threshold in the month of July. The results in Fig. 3, (full numbers in Table 15) show that our method provides significant improvements compared to standard MC estimates with at least an order of magnitude reduction in variance. For the most extreme event $\tau _ { 3 } ,$ , only 2/16 runs provided a single sample above the threshold; hence MC provides almost no samples with which to characterise the event. Figure 3 demonstrates that the generated guided samples provide physically plausible temperature maps: the maps show more warming over land masses compared to the ocean and spatially coherent warming patterns that are not restricted solely to the target region. Additional results and ablations in App. H.

Compound hot and dry extremes. Next we probe whether our method can capture extreme events impacting multiple variables by considering events that are both extremely hot and dry because of their outsized societal impacts (Fan et al., 2023; Zscheischler et al., 2017). For these joint events it is important to not just consider the separate marginal probabilities of each variable but also to accurately capture the correlations between variables. To do so, we compute the dependence ratio of the event, defined as $d = p _ { 0 } [ x > \tau _ { \mathrm { t a s } } , x < \tau _ { \mathrm { h u r s } } ] \left/ p _ { 0 } [ x > \tau _ { \mathrm { t a s } } ] p _ { 0 } [ x < \tau _ { \mathrm { h u r s } } ] \right.$ . If the events are independent the dependence ratio would reduce to $d = 1 ,$ while $d > 1$ 1 implies that the two events co-occur. We use the non-factorised form of the reward, $\Phi ( \cdot ) = \Phi _ { 2 } ( \mathrm { h u r s } , \tan \theta )$ , a function of both variables; at the rarest level it reduces the CV fourfold relative to a product of marginal rewards (Table 18).

Table 1 shows we recover the correct joint probability and the dependence ratio against the ground-truth estimate with a positive net speed-up compared to crude MC. Notably, the dependence structure strengthens as the event becomes rarer from $d = 3 1 . 4  1 . 4 \times 1 0 ^ { 3 }$ This suggests that a hot-and-dry July over R is three orders of magnitude more likely than the product of its marginals would suggest.

Table 1: Probability estimates of three compound hot and dry extreme events over the PNW region. The dependence ratio d is defined in the main text, <sup>ˆ</sup>d denotes that estimated by our method. Net speed-up is the crude-MC cost of matching the CV of the guided run divided by the measured per-sample cost of guidance (Table 21); thresholds τ are given in App. H.1. Mean ± standard error over 16 seeds.
<table><tr><td> $p ^ { \star }$ </td><td> $\hat { p } / { p ^ { \star } }$ </td><td>CV</td><td> $d ^ { \star }$ </td><td> $\hat { d } / d ^ { \star }$ </td><td>Speed Up ↑</td></tr><tr><td> $3 . 1 4 \times 1 0 ^ { - 3 }$ </td><td> $0 . 9 7 \pm 0 . 0 2$ </td><td>0.07</td><td>31</td><td> $1 . 0 4 \pm 0 . 0 2$ </td><td>9×</td></tr><tr><td> $2 . 1 2 \times 1 0 ^ { - 4 }$ </td><td> $0 . 9 9 \pm 0 . 0 3$ </td><td>0.11</td><td>212</td><td> $1 . 0 2 \pm 0 . 0 3$ </td><td> $4 7 \times$ </td></tr><tr><td> $1 . 3 9 \times 1 0 ^ { - 5 }$ </td><td> $1 . 0 4 \pm 0 . 0 2$ </td><td>0.08</td><td>1384</td><td> $1 . 0 6 \pm 0 . 0 6$ </td><td>1413×</td></tr></table>

A warming climate. As the model can generate samples conditioned on the global temperature anomaly $\Delta T$ , it lets us ask how do extreme events change under climate change? We fix the temperature threshold at $\tau _ { \textrm t a s } \geq 5 . 8 1$ K and sweep $\Delta T \in [ 0 , 4 ]$ K, rerunning the same recipe as above. This temperature approximately corresponds to $p _ { 0 } [ E ] \stackrel { \cdot } { \approx } 1 0 ^ { - 4 }$ at 1.2 K (2021) of warming, mirroring the PNW heatwave (McKinnon & Simpson, 2022). The event moves from a threshold exceedance probability of $\sim 1 0 ^ { - 6 }$ at $\Delta T = 0$ to a near-certainty at $\Delta T = 4 \mathrm { \ K }$ Our estimator tracks it across all six orders of magnitude without any modification. Fig. 4 reports this probability estimate. See App. H.3 for details, including Table 19 and Fig. 19 for corresponding global temperature maps. This experiment shows that the method can be potentially used for climate attribution; see Van Oldenborgh et al. (2021) for an overview.

![](images/98358aec871b5d84f15e09b6bf9023ba3b8d3efa1a4f3216ddcdb7f983952f28.jpg)  
Figure 4: Estimated $\hat { p } _ { 0 } [ E ]$ versus global temperature anomaly $\Delta T$ for our method versus unguided MC at equal compute. Points are the mean over 16 runs, bars are ±1 std. Bars reaching the axis floor indicate levels where some MC runs record no event; for $\Delta T \le 1 . 0$ each MC run records at most one event. See Tables 19 and 20.

## 5 CONCLUSION

We have shown that the diffusion sampler builds the same annealing bridge that traditional rare-event samplers build by hand. Using this insight, we present DireSMC and steer the generation process towards a user-defined rare event using a smooth relaxation of the event set, accumulating importance weights and estimating the probability along the way. Empirically, on synthetic examples and a score-based climate emulator, we achieve a net speedup of 9 to 1413× over crude Monte Carlo, improving with rarity. We believe that this is of value to the scientific community where there is a growing interest in investigating how well a generative model can capture rare events (Sun et al., 2025).

Limitations & Future Work. The variance of our estimator can be improved through better approximations of Doob’s h-transform e.g. (Denker et al., 2024; Potaptchik et al., 2026; Uehara et al., 2025) as well as better reward schedule design (Syed et al., 2026); these can be readily plugged in without modification to the algorithm. Our estimates target the discretised, approximate law we sample from, and so carry a reducible discretisation bias and any errors of an approximate score. The rejuvenation kernel used (Lam et al., 2026) assumes a conservative score, which a trained network need not provide; distilling the network into an energy model (Thornton et al., 2025) could be a viable solution. Additionally the MADM kernel also does not target the law $\gamma _ { k }$ induced by the numerical integrator, with this mismatch decreasing with the weak-order convergence rate of the integrator. Algorithm 1 is agnostic to the choice of MCMC moves, and any alternatives to MADM can be used.

## ACKNOWLEDGMENTS

The authors would like to thank Abbas Mammadov, Kevin Lam, Alison Peard, Sam Howard, Charlotte Merchant and Lukas Seier for helpful discussions. AS is part of the Intelligent Earth CDT supported by funding from the UK Research and Innovation Council (UKRI) grant number EP/Y030907/1. TR acknowledges funding from the EU’s Horizon Europe program under grant agreement number 10113184, acknowledge funding from UK Research and Innovation (UKRI), and also received funding from ARIA and DSIT and Pillar VC under the Encode: AI for Science Fellowship. CW acknowledges support from the Defence Science and Technology (DST) Group Australia. CW and YWT are supported in part by the Engineering and Physical Sciences Research Council (EPSRC) through the AI Hub in Generative Models [grant number EP/Y028805/1]. PS acknowledges funding from European Union’s Horizon Europe projects CleanCloud and Embed2Scale with Grant Agreements 101137639 and 101131841 and their UKRI underwrite. We acknowledge the use of resources provided by JASMIN, the UK’s collaborative data analysis environment( https://www.jasmin.ac.uk) and the Isambard-AI National AI Research Resource (AIRR) (McIntosh-Smith et al., 2024). Isambard-AI is operated by the University of Bristol and is funded by the UK Government’s Department for Science, Innovation and Technology (DSIT) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023]

## REFERENCES

Cameron Abrams and Giovanni Bussi. Enhanced sampling in molecular dynamics using metadynamics, replica-exchange, and temperature-acceleration. Entropy, 16(1):163–199, 2013.

Josh Abramson, Jonas Adler, Jack Dunger, Richard Evans, Tim Green, Alexander Pritzel, Olaf Ronneberger, Lindsay Willmore, Andrew J Ballard, Joshua Bambrick, et al. Accurate structure prediction of biomolecular interactions with alphafold 3. Nature, 630(8016):493–500, 2024.

Sergios Agapiou, Omiros Papaspiliopoulos, Daniel Sanz-Alonso, and Andrew M Stuart. Importance sampling: Intrinsic dimension and computational cost. Statistical Science, pp. 405–431, 2017.

Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=li7qeBbCR1t.

Siu-Kui Au and Yu Wang. Engineering risk assessment with subset simulation. John Wiley & Sons, 2014.

Joe Benton, Valentin De Bortoli, Arnaud Doucet, and George Deligiannidis. Nearly d-linear convergence bounds for diffusion models via stochastic localization. In International Conference on Learning Representations, volume 2024, pp. 36916–36936, 2024.

Julius Berner, Lorenz Richter, and Karen Ullrich. An optimal control perspective on diffusion-based generative modeling. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=oYIjw37pTP.

Christopher M Bishop and Nasser M Nasrabadi. Pattern recognition and machine learning, volume 4. Springer, 2006.

Valentin De Bortoli, James Thornton, Jeremy Heng, and Arnaud Doucet. Diffusion schrödinger bridge with applications to score-based generative modeling. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=9BnCwiXB0ty.

Shahine Bouabid, Andre Nogueira Souza, and Raffaele Ferrari. Score-based generative emulation of impact-relevant earth system model outputs. Journal of Advances in Modeling Earth Systems, 18(3):e2025MS005558, 2026. doi: https://doi.org/10.1029/ 2025MS005558. URL https://agupubs.onlinelibrary.wiley.com/doi/abs/ 10.1029/2025MS005558. e2025MS005558 2025MS005558.

Benjamin Boys, Mark Girolami, Jakiw Pidstrigach, Sebastian Reich, Alan Mosca, and O Deniz Akyildiz. Tweedie moment projected diffusions for inverse problems. arXiv preprint arXiv:2310.06721, 2023.

Sourav Chatterjee and Persi Diaconis. The sample size required in importance sampling. The Annals ofApplied Probability, 28(2):1099–1135, 2018.

Sitan Chen, Sinho Chewi, Jerry Li, Yuanzhi Li, Adil Salim, and Anru Zhang. Sampling is as easy as learning the score: theory for diffusion models with minimal data assumptions. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview. net/forum?id=zyLVMgsZ0U\_.

Nicolas Chopin, Omiros Papaspiliopoulos, et al. An introduction to sequential Monte Carlo, volume 4. Springer, 2020.

Nicolas Chopin, Sumeetpal S Singh, Tomás Soto, and Matti Vihola. On resampling schemes for particle filters with weakly informative observations. The Annals of Statistics, 50(6):3197–3222, 2022.

Hyungjin Chung, Jeongsol Kim, Michael Thompson Mccann, Marc Louis Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=OnD9zGAGT0k.

Guillaume Couairon, Renu Singh, Anastase Charantonis, Christian Lessig, and Claire Monteleoni. Archesweathergen: Skillful and compute-efficient probabilistic weather forecasting with machine learning. Science Advances, 12(17):eadx2372, 2026.

Paolo Dai Pra. A stochastic control approach to reciprocal diffusion processes. Applied mathematics and Optimization, 23(1):313–329, 1991.

Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. Sequential monte carlo samplers. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 68(3):411–436, 2006.

Pierre Del Moral, Arnaud Doucet, and Ajay Jasra. On adaptive resampling strategies for sequential monte carlo methods. Bernoulli, 18(1), February 2012. ISSN 1350-7265. doi: 10.3150/10-bej335. URL http://dx.doi.org/10.3150/10-BEJ335.

Alexander Denker, Francisco Vargas, Shreyas Padhy, Kieran Didi, Simon Mathis, Vincent Dutordoir, Riccardo Barbano, Emile Mathieu, Urszula J Komorowska, and Pietro Lio. Deft: Efficient fine-tuning of diffusion models by learning the generalised h-transform. Advances in Neural Information Processing Systems, 37:19636–19682, 2024.

Teo Deveney, Jan Stanczuk, Lisa Kreusser, Chris Budd, and Carola-Bibiane Schönlieb. Closing the ode–sde gap in score-based diffusion models through the fokker–planck equation. Philosophical transactions. Series A, Mathematical, physical, and engineering sciences, 383(2298):20240503, 2025.

Arnaud Doucet, Simon Godsill, and Christophe Andrieu. On sequential monte carlo sampling methods for bayesian filtering. Statistics and computing, 10(3):197–208, 2000.

Arnaud Doucet, Will Grathwohl, Alexander G Matthews, and Heiko Strathmann. Scorebased diffusion meets annealed importance sampling. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 21482–21494. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1561. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/86b7128efa3950df7c0f6c0342e6dcc1-Paper-Conference.pdf.

Zvi Drezner. Computation of the trivariate normal integral. Mathematics of Computation, 62(205): 289–294, 1994.

Zvi Drezner and George O Wesolowsky. On the computation of the bivariate normal integral. Journal ofStatistical Computation and Simulation, 35(1-2):101–107, 1990.

Bradley Efron. Tweedie’s formula and selection bias. Journal ofthe American Statistical Association, 106(496):1602–1614, 2011. doi: 10.1198/jasa.2011.tm11181. URL https://doi.org/10. 1198/jasa.2011.tm11181. PMID: 22505788.

Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks, 107:3–11, 2018. ISSN 0893-6080. doi: https://doi.org/10.1016/j.neunet.2017.12.012. URL https://www.sciencedirect. com/science/article/pii/S0893608017302976. Special issue on deep reinforcement learning.

Veronika Eyring, Sandrine Bony, Gerald A Meehl, Catherine A Senior, Bjorn Stevens, Ronald J Stouffer, and Karl E Taylor. Overview of the coupled model intercomparison project phase 6 (cmip6) experimental design and organization. Geoscientific Model Development, 9(5):1937–1958, 2016.

Xuewei Fan, Chiyuan Miao, Jakob Zscheischler, Louise Slater, Yi Wu, Yuanfang Chai, and Amir AghaKouchak. Escalating hot-dry extremes amplify compound fire weather risk. Earth’s Future, 11(11):e2023EF003976, 2023. doi: https://doi.org/10.1029/ 2023EF003976. URL https://agupubs.onlinelibrary.wiley.com/doi/abs/ 10.1029/2023EF003976. e2023EF003976 2023EF003976.

Nial Friel and Anthony N Pettitt. Marginal likelihood estimation via power posteriors. Journal ofthe Royal Statistical Society Series B: Statistical Methodology, 70(3):589–607, 2008.

Ruiqi Gao, Emiel Hoogeboom, Jonathan Heek, Valentin De Bortoli, Kevin P. Murphy, and Tim Salimans. Diffusion meets flow matching: Two sides of the same coin. 2024. URL https: //diffusionflow.github.io/.

Alan Genz. Numerical computation of multivariate normal probabilities. Journal ofComputational and Graphical Statistics, 1(2):141–149, 1992. ISSN 10618600. URL http://www.jstor. org/stable/1390838.

Alan Genz. Numerical computation of rectangular bivariate and trivariate normal and t probabilities. Statistics and computing, 14(3):251–260, 2004.

Paul Glasserman and Jingyi Li. Importance sampling for portfolio credit risk. Management science, 51(11):1643–1656, 2005.

Paul Glasserman, Philip Heidelberger, Perwez Shahabuddin, and Tim Zajic. Multilevel splitting for estimating rare event probabilities. Operations Research, 47(4):585–600, 1999. ISSN 0030364X, 15265463. URL http://www.jstor.org/stable/223163.

Martin Gonzalez, Nelson Fernandez, Thuy Vinh Dinh Tran, Elies Gherbi, Hatem Hajri, and Nader Masmoudi. SEEDS: Exponential SDE solvers for fast high-quality sampling from diffusion models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=V6IgkYKD8P.

Pieralberto Guarniero, Adam M Johansen, and Anthony Lee. The iterated auxiliary particle filter. Journal of the American Statistical Association, 112(520):1636–1647, 2017.

Tiankai Hang, Shuyang Gu, Chen Li, Jianmin Bao, Dong Chen, Han Hu, Xin Geng, and Baining Guo. Efficient diffusion training via min-snr weighting strategy. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 7407–7417. IEEE, 2023.

Jiajun He, José Miguel Hernández-Lobato, Yuanqi Du, and Francisco Vargas. RNE: plug-and-play diffusion inference-time control and energy-based training. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=fkf7tlkN8q.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 6840–6851. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf.

Adam M Johansen and Arnaud Doucet. A note on auxiliary particle filters. Statistics & Probability Letters, 78(12):1498–1504, 2008.

Herman Kahn and Theodore E Harris. Estimation of particle transmission by random sampling. National Bureau ofStandards applied mathematics series, 12:27–30, 1951.

Stanley Kaplan and B John Garrick. On the quantitative definition of risk. Risk analysis, 1(1):11–27, 1981.

Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 26565–26577. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1926. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/a98846e9d9cc01cfb87eb694d946ce6b-Paper-Conference.pdf.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Peter E. Kloeden and Eckhard Platen. Numerical Solution of Stochastic Differential Equations, volume 23 of Stochastic Modelling and Applied Probability. Springer, 1992.

Chieh-Hsin Lai, Yuhta Takida, Naoki Murata, Toshimitsu Uesaka, Yuki Mitsufuji, and Stefano Ermon. Fp-diffusion: Improving score-based diffusion models by enforcing the underlying score fokker-planck equation. In International Conference on Machine Learning, pp. 18365–18398. PMLR, 2023.

Kevin H Lam, Tyler Farghly, Christopher Williams, Jun Yang, Yee Whye Teh, and Arnaud Doucet. Metropolis-adjusted diffusion models. arXiv preprint arXiv:2605.09654, 2026.

Amaury Lancelin, Alexander Wikner, Laurent Dubus, Clément Le Priol, Dorian S. Abbot, Freddy Bouchet, Pedram Hassanzadeh, and Jonathan Weare. Ai-boosted rare event sampling to characterize extreme weather. Phys. Rev. Lett., Jun 2026. doi: 10.1103/b1gc-9c2q. URL https://link. aps.org/doi/10.1103/b1gc-9c2q.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=XVjTT1nw5z.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=2uAaGwlP\_V.

Peter Manshausen, Noah Brenowitz, Julius Berner, Karthik Kashinath, and Mike Pritchard. Towards accurate extreme event likelihoods from diffusion model climate emulators, 2026. URL https: //arxiv.org/abs/2605.03802.

Thorsten Mauritsen, Jürgen Bader, Tobias Becker, Jörg Behrens, Matthias Bittner, Renate Brokopf, Victor Brovkin, Martin Claussen, Traute Crueger, Monika Esch, et al. Developments in the mpi-m earth system model version 1.2 (mpi-esm1. 2) and its response to increasing co2. Journal of Advances in Modeling Earth Systems, 11(4):998–1038, 2019.

Simon McIntosh-Smith, Sadaf Alam, and Christopher Woods. Isambard-ai: a leadership-class supercomputer optimised specifically for artificial intelligence. In Proceedings of the Cray User Group, pp. 44–54. 2024.

Karen A. McKinnon and Isla R. Simpson. How unexpected was the 2021 pacific northwest heatwave? Geophysical Research Letters, 49(18):e2022GL100380, 2022. doi: https://doi.org/ 10.1029/2022GL100380. URL https://agupubs.onlinelibrary.wiley.com/doi/ abs/10.1029/2022GL100380. e2022GL100380 2022GL100380.

Grigori N Milstein and Michael V Tretyakov. Stochastic numerics for mathematical physics, volume 39. Springer, 2004.

Pierre Moral. Feynman-Kac formulae: genealogical and interacting particle systems with applications. Springer, 2004.

Jérôme Morio, Mathieu Balesdent, Damien Jacquemart, and Christelle Vergé. A survey of rare event simulation methods for static input–output models. Simulation Modelling Practice and Theory, 49: 287–304, 2014.

Christian A Naesseth, Fredrik Lindsten, and Thomas B Schön. Elements of sequential monte carlo. Foundations and Trends® in Machine Learning, 12(3):187–306, 2019.

Radford M Neal. Annealed importance sampling. Statistics and computing, 11(2):125–139, 2001.

Bernt Øksendal. Stochastic differential equations. In Stochastic differential equations: an introduction with applications, pp. 38–50. Springer, 2003.

Brian C O’Neill, Claudia Tebaldi, Detlef P Van Vuuren, Veronika Eyring, Pierre Friedlingstein, George Hurtt, Reto Knutti, Elmar Kriegler, Jean-Francois Lamarque, Jason Lowe, et al. The scenario model intercomparison project (scenariomip) for cmip6. Geoscientific Model Development, 9(9):3461–3482, 2016.

Matthew O' Kelly, Aman Sinha, Hongseok Namkoong, Russ Tedrake, and John Duchi. Scalable end-to-end autonomous vehicle testing via rare-event simulation. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/paper/2018/ file/653c579e3f9ba5c03f2f2f8cf4512b39-Paper.pdf.

Art B. Owen. Monte Carlo theory, methods and examples. https://artowen.su.domains/ mc/, 2013.

Donald B Owen. Tables for computing bivariate normal probabilities. The Annals of Mathematical Statistics, 27(4):1075–1090, 1956.

A. Peard, Y. Mo, and J. W. Hall. Simulating spatial multi-hazards with generative deep learning. Natural Hazards and Earth System Sciences, 26(4):1663–1683, 2026. doi: 10.5194/ nhess-26-1663-2026. URL https://nhess.copernicus.org/articles/26/1663/ 2026/.

Emanuel Pfarr, Radu Timofte, and Frank Werner. Stochstic sampling for generative diffusion models: From euler-maruyama to higher-order schemes, 2026. URL https://arxiv.org/abs/ 2601.18425.

Michael K Pitt and Neil Shephard. Filtering via simulation: Auxiliary particle filters. Journal of the American statistical association, 94(446):590–599, 1999.

Peter Potaptchik, Adhi Saravanan, Abbas Mammadov, Alvaro Prat, Michael Samuel Albergo, and Yee Whye Teh. Meta flow maps enable scalable reward alignment. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id= K5gV8Yptne.

Ilan Price, Alvaro Sanchez-Gonzalez, Ferran Alet, Tom R. Andersson, Andrew El-Kadi, Dominic Masters, Timo Ewalds, Jacklynn Stott, Shakir Mohamed, Peter Battaglia, Remi Lam, and Matthew Willson. Gencast: Diffusion-based ensemble forecasting for medium-range weather, 2024. URL https://arxiv.org/abs/2312.15796.

Carl Edward Rasmussen and Christopher K. I. Williams. Gaussian Processesfor Machine Learning. The MIT Press, 2006. ISBN 978-0-262-25683-4. doi: 10.7551/mitpress/3206.001.0001. URL https://doi.org/10.7551/mitpress/3206.001.0001.

Gerardo Rubino, Bruno Tuffin, et al. Rare event simulation using Monte Carlo methods, volume 73. Wiley Online Library, 2009.

Yuyang Shi, Valentin De Bortoli, Andrew Campbell, and Arnaud Doucet. Diffusion schrödinger bridge matching. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 62183–62223. Curran Associates, Inc., 2023. doi: 10.52202/ 075280-2717. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/c428adf74782c2092d254329b6b02482-Paper-Conference.pdf.

Michael R Shirts and John D Chodera. Statistically optimal analysis of samples from multiple equilibrium states. The Journal ofchemical physics, 129(12), 2008.

Raghav Singhal, Zachary Horvitz, Ryan Teehan, Mengye Ren, Zhou Yu, Kathleen Mckeown, and Rajesh Ranganath. A general framework for inference-time scaling and steering of diffusion models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 55810–55827. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/ singhal25b.html.

Marta Skreta, Tara Akhound-Sadegh, Viktor Ohanesian, Roberto Bondesan, Alán Aspuru-Guzik, Arnaud Doucet, Rob Brekelmans, Alexander Tong, and Kirill Neklyudov. Feynman-kac correctors in diffusion: Annealing, guidance, and product of experts. arXiv preprint arXiv:2503.02819, 2025.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021a. URL https://openreview.net/ forum?id=St1giarCHLP.

Jiaming Song, Arash Vahdat, Morteza Mardani, and Jan Kautz. Pseudoinverse-guided diffusion models for inverse problems. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=9\_gsMA8MRKQ.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations, 2021b. URL https://openreview.net/forum? id=PxTIG12RRHS.

Y. Qiang Sun, Pedram Hassanzadeh, Mohsen Zand, Ashesh Chattopadhyay, Jonathan Weare, and Dorian S. Abbot. Can ai weather models predict out-of-distribution gray swan tropical cyclones? Proceedings of the National Academy of Sciences, 122(21):e2420914122, 2025. doi: 10.1073/pnas.2420914122. URL https://www.pnas.org/doi/abs/10.1073/pnas. 2420914122.

Saifuddin Syed, Alexandre Bouchard-Côté, Kevin Chern, and Arnaud Doucet. Optimized annealed sequential monte carlo samplers. Journal of the Royal Statistical Society Series B: Statistical Methodology, pp. qkag082, 2026.

Denis Talay and Luciano Tubaro. Expansion of the global error for numerical schemes solving stochastic differential equations. Stochastic analysis and applications, 8(4):483–509, 1990.

James Thornton, Louis Béthune, Ruixiang Zhang, Arwen Bradley, Preetum Nakkiran, and Shuangfei Zhai. Composition and control with distilled energy diffusion models and sequential monte carlo. arXiv preprint arXiv:2502.12786, 2025.

Masatoshi Uehara, Yulai Zhao, Chenyu Wang, Xiner Li, Aviv Regev, Sergey Levine, and Tommaso Biancalani. Inference-time alignment in diffusion models with reward-guided generation: Tutorial and review. arXiv preprint arXiv:2501.09685, 2025.

Geert Jan Van Oldenborgh, Karin Van Der Wiel, Sarah Kew, Sjoukje Philip, Friederike Otto, Robert Vautard, Andrew King, Fraser Lott, Julie Arrighi, Roop Singh, et al. Pathways and pitfalls in extreme event attribution. Climatic change, 166(1):13, 2021.

Francisco Vargas, Will Sussman Grathwohl, and Arnaud Doucet. Denoising diffusion samplers. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=8pvnfTAbu1f.

Aki Vehtari, Daniel Simpson, Andrew Gelman, Yuling Yao, and Jonah Gabry. Pareto smoothed importance sampling. Journal of Machine Learning Research, 25(72):1–58, 2024.

An Vuong, Michael Thompson McCann, Javier E. Santos, and Yen Ting Lin. Are we really learning the score function? reinterpreting diffusion models through wasserstein gradient flow matching. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=CzyJqXQRhJ.

D. Watson-Parris, Y. Rao, D. Olivié, Ø. Seland, P. Nowack, G. Camps-Valls, P. Stier, S. Bouabid, M. Dewey, E. Fons, J. Gonzalez, P. Harder, K. Jeggle, J. Lenhardt, P. Manshausen, M. Novitasari, L. Ricard, and C. Roesch. Climatebench v1.0: A benchmark for data-driven climate projections. Journal ofAdvances in Modeling Earth Systems, 14(10):e2021MS002954, 2022. doi: https://doi. org/10.1029/2021MS002954. URL https://agupubs.onlinelibrary.wiley.com/ doi/abs/10.1029/2021MS002954. e2021MS002954 2021MS002954.

Christopher Williams, Andrew Campbell, Arnaud Doucet, and Saifuddin Syed. Score-optimal diffusion schedules. Advances in Neural Information Processing Systems, 37:107960–107983, 2024.

Luhuan Wu, Brian Trippe, Christian Naesseth, David Blei, and John P Cunningham. Practical and asymptotically exact conditional sampling in diffusion models. Advances in Neural Information Processing Systems, 36:31372–31403, 2023.

Yuchen Wu, Minshuo Chen, Zihao Li, Mengdi Wang, and Yuting Wei. Theoretical insights for diffusion guidance: A case study for gaussian mixture models. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id= M1ADedSnlJ.

Yu Xie, Ludwig Winkler, Lixin Sun, Sarah Lewis, Adam E Foster, José Jiménez Luna, Tim Hempel, Michael Gastegger, Yaoyi Chen, Iryna Zaporozhets, et al. Enhanced diffusion sampling: Efficient rare event sampling and free energy calculation with diffusion models. arXiv preprint arXiv:2602.16634, 2026.

Yilun Xu, Mingyang Deng, Xiang Cheng, Yonglong Tian, Ziming Liu, and Tommi S. Jaakkola. Restart sampling for improving generative processes. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=wFuemocyHZ.

Claudio Zeni, Robert Pinsler, Daniel Zügner, Andrew Fowler, Matthew Horton, Xiang Fu, Zilong Wang, Aliaksandra Shysheya, Jonathan Crabbé, Shoko Ueda, et al. A generative model for inorganic materials design. Nature, 639(8055):624–632, 2025.

Jakob Zscheischler, Rene Orth, and Sonia I Seneviratne. Bivariate return periods of temperature and precipitation explain a large fraction of european crop yields. Biogeosciences, 14(13):3309–3320, 2017.

## Appendix

A Proofs 20   
A.1 Notation and two identities 20   
A.2 Structure of the tilted chain 20   
A.3 Proof of Theorem 1 21   
A.4 Proof of Theorem 2 21   
A.5 The optimal reward recursion . 22   
A.6 Proof of Lemma 1 . 22   
B Diffusion model details 23   
C Sequential methods 24   
C.1 Estimating the normalisation constant 24   
C.2 Generic SMC algorithm . 26   
C.3 Why annealing becomes necessary 26   
C.4 Splitting estimator . 27   
D Weight computation 28   
D.1 $\zeta _ { k }$ for common integrators 29   
E The guidance gradient via local linearisation 31   
E.1 Factorised guidance for multiple constraints 32   
F The correlated 2D GMM benchmark 33   
G 2D GMM experiments 36   
G.1 Experimental details 36   
G.2 Sources of Bias 37   
G.3 The threshold-exceedance experiment 39   
G.4 Box and multiple box events 45   
G.5 Ablation of Parameters 47   
H Climate experiments 48   
H.1 Setup and model details . 48   
H.2 Experiment 1: Compound hot & dry events 50   
H.3 Experiment 2: A changing climate 57   
H.4 Computational cost 58   
H.5 Hyperparameter ablation 59   
I Metropolis-adjusted Langevin corrections 60   
J Related Works Extended 61   
J.1 The PF-ODE Estimator . 62

## A PROOFS

## A.1 NOTATION AND TWO IDENTITIES

Throughout, $( X _ { 0 } , \ldots , X _ { K } )$ is the base chain of Eq. $( 2 ) , \mathbb { P } _ { 0 : K }$ its law, $p _ { k }$ its marginals, and $\mathbb { Q } _ { 0 : K }$ the law of the guided chain with kernels ${ \widehat { Q } } _ { k - 1 | k } ;$ both start from $p _ { \mathrm { r e f } } . \mathrm { A l l }$ densities are with respect to Lebesgue measure. For a reward $r _ { 0 }$ with $\gamma _ { 0 } [ 1 ] < \infty$ , write $Z _ { 0 } : = \gamma _ { 0 } [ 1 ]$ and

$$
\frac { d \mathbb { P } _ { 0 : K } ^ { \star } } { d \mathbb { P } _ { 0 : K } } ( x _ { K : 0 } ) : = \frac { e ^ { r _ { 0 } ( x _ { 0 } ) } } { Z _ { 0 } } , \qquad \frac { d \mathbb { P } _ { 0 : K } ^ { E } } { d \mathbb { P } _ { 0 : K } } ( x _ { K : 0 } ) : = \frac { \mathbf { 1 } _ { E } ( x _ { 0 } ) } { p _ { 0 } [ E ] } ,\tag{19}
$$

for the base chain tilted by $e ^ { r _ { 0 } }$ and conditioned on $X _ { 0 } \in E$ respectively; the second is the first with $e ^ { r _ { 0 } } = \mathbf { 1 } _ { E }$ . The Doob rewards are

$$
r _ { k } ^ { \star } ( x _ { k } ) : = \log \int \bar { P } _ { 0 | k } ( \mathrm { d } x _ { 0 } \mid x _ { k } ) e ^ { r _ { 0 } ( x _ { 0 } ) } , \qquad k = 0 , \ldots , K ,\tag{20}
$$

so that $r _ { 0 } ^ { \star } = r _ { 0 }$ and $\begin{array} { r } { Z _ { 0 } = \int p _ { \mathrm { r e f } } ( x _ { K } ) e ^ { r _ { K } ^ { \star } ( x _ { K } ) } \mathrm { d } x _ { K } } \end{array}$

Two identities are used repeatedly. First, the tower identity: by the Markov property,

$$
\int \bar { P } _ { k - 1 | k } ( \mathrm { d } x _ { k - 1 } \mid x _ { k } ) e ^ { r _ { k - 1 } ^ { \star } ( x _ { k - 1 } ) } = e ^ { r _ { k } ^ { \star } ( x _ { k } ) } , \qquad k = 1 , \ldots , K .\tag{21}
$$

Second, the telescoping identity: the reward ratios in the incremental weight of Eq. (10) collapse along a trajectory, since $r _ { K } = 0$

$$
\begin{array} { l } { \displaystyle \prod _ { k = 1 } ^ { K } g _ { k - 1 | k } ( x _ { k - 1 } , x _ { k } ) = e ^ { r _ { 0 } ( x _ { 0 } ) } \prod _ { k = 1 } ^ { K } \frac { \bar { P } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } { \bar { Q } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } } \\ { = e ^ { r _ { 0 } ( x _ { 0 } ) } \frac { d \mathbb { P } _ { 0 : K } } { d \mathbb { Q } _ { 0 : K } } ( x _ { K : 0 } ) , } \end{array}\tag{22}
$$

the last equality because both chains start from $p _ { \mathrm { r e f } }$

## A.2 STRUCTURE OF THE TILTED CHAIN

Lemma 2. $\mathbb { P } _ { 0 : K } ^ { \star }$ is a Markov chain. Its marginal at step k is $\pi _ { k } ^ { \star } : = p _ { k } e ^ { r _ { k } ^ { \star } } / Z _ { 0 }$ , so its initial law is $\pi _ { K } ^ { \star } = p _ { \mathrm { r e f } } e ^ { r _ { K } ^ { \star } } / Z _ { 0 }$ , its transitions are the twisted kernels

$$
\bar { P } _ { k - 1 \mid k } ^ { \star } ( \mathrm { d } x _ { k - 1 } \mid x _ { k } ) : = \bar { P } _ { k - 1 \mid k } ( \mathrm { d } x _ { k - 1 } \mid x _ { k } ) e ^ { r _ { k - 1 } ^ { \star } ( x _ { k - 1 } ) - r _ { k } ^ { \star } ( x _ { k } ) } ,\tag{23}
$$

Consequently

$$
\frac { d \mathbb { P } _ { 0 : K } ^ { \star } } { d \mathbb { Q } _ { 0 : K } } \big ( x _ { K : 0 } \big ) = \frac { e ^ { r _ { K } ^ { \star } \left( x _ { K } \right) } } { Z _ { 0 } } \prod _ { k = 1 } ^ { K } \frac { \overline { { P } } _ { k - 1 | k } ^ { \star } \big ( x _ { k - 1 } \mid x _ { k } \big ) } { \overline { { Q } } _ { k - 1 | k } \big ( x _ { k - 1 } \mid x _ { k } \big ) } .\tag{24}
$$

Proof. By Eq. (21), each $\stackrel {  } { P } _ { k - 1 | k } ^ { \star }$ integrates to one and is a Markov kernel. Multiplying the kernels and telescoping the exponents,

$$
\prod _ { k = 1 } ^ { K } \bar { P } _ { k - 1 \mid k } ^ { \star } ( x _ { k - 1 } \mid x _ { k } ) = \frac { e ^ { r _ { 0 } ( x _ { 0 } ) } } { e ^ { r _ { K } ^ { \star } ( x _ { K } ) } } \prod _ { k = 1 } ^ { K } \bar { P } _ { k - 1 \mid k } ( x _ { k - 1 } \mid x _ { k } ) ,\tag{25}
$$

using $r _ { 0 } ^ { \star } = r _ { 0 }$ . Hence

$$
\begin{array} { r l } & { \frac { p _ { \mathrm { r e f } } \left( x _ { K } \right) e ^ { r _ { K } ^ { \star } \left( x _ { K } \right) } } { Z _ { 0 } } \displaystyle \prod _ { k = 1 } ^ { K } \bar { P } _ { k - 1 | k } ^ { \star } ( x _ { k - 1 } \mid x _ { k } ) = \frac { e ^ { r _ { 0 } \left( x _ { 0 } \right) } } { Z _ { 0 } } p _ { \mathrm { r e f } } ( x _ { K } ) \displaystyle \prod _ { k = 1 } ^ { K } \bar { P } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } \\ & { \qquad = \displaystyle \frac { d \mathbb { P } _ { 0 : K } ^ { \star } } { d x _ { K : 0 } } ( x _ { K : 0 } ) , } \end{array}\tag{26}
$$

which exhibits $\mathbb { P } _ { 0 : K } ^ { \star }$ as the chain started from $\pi _ { K } ^ { \star }$ with kernels $\stackrel {  } { P } _ { k - 1 | k } ^ { \star }$ . The marginal at step k follows from the tower identity applied to the tilt: integrating out all coordinates but $x _ { k }$ gives $p _ { k } ( x _ { k } ) \int \bar { P } _ { 0 | k } ( \mathrm { d } x _ { 0 } \mid x _ { k } ) e ^ { r _ { 0 } ( x _ { 0 } ) } / \bar { Z _ { 0 } } = p _ { k } ( x _ { k } ) e ^ { r _ { k } ^ { \star } ( x _ { k } ) } / Z _ { 0 }$ . Dividing the density of $\mathbb { P } _ { 0 : K } ^ { \star }$ by that of $\begin{array} { r } { \mathbb { Q } _ { 0 : K } , p _ { \mathrm { r e f } } ( \dot { x } _ { K } ) \prod _ { k } \overleftarrow { Q } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) , \mathrm { g i v e s ~ E q . } ( 2 4 ) . } \end{array}$ □

## A.3 PROOF OF THEOREM 1

Proof. The auxiliary kernel $\vec { F } _ { k \mid k - 1 } ( x _ { k } \mid x _ { k - 1 } ) = p _ { k } ( x _ { k } ) \overleftarrow { P } _ { k - 1 \mid k } ( x _ { k - 1 } \mid x _ { k } ) / p _ { k - 1 } ( x _ { k - 1 } )$ integrates to one in $x _ { k }$ by the marginal recursion of Section 2, so it is a Markov kernel. With targets $\gamma _ { k } = p _ { k } e ^ { r _ { k } }$ and proposal $\overline { { B } } _ { k - 1 | k } = \overline { { Q } } _ { k - 1 | k }$ , the generic SMC weight of Eq. (7) is

$$
\begin{array} { r l } & { \frac { \gamma _ { k - 1 } ( x _ { k - 1 } ) \vec { F } _ { k  k - 1 } ( x _ { k }  \ x _ { k - 1 } ) } { \gamma _ { k } ( x _ { k } ) \tilde { Q } _ { k - 1  k } ( x _ { k - 1 }  \ x _ { k } ) } = \frac { p _ { k - 1 } ( x _ { k - 1 } ) e ^ { r _ { k - 1 } ( x _ { k - 1 } ) } } { p _ { k } ( x _ { k } ) e ^ { r _ { k } ( x _ { k } ) } } \cdot \frac { \vec { F } _ { k  k - 1 } ( x _ { k }  \ x _ { k - 1 } ) } { \tilde { Q } _ { k - 1  k } ( x _ { k - 1 }  x _ { k } ) } } \\ &  \phantom { \frac { \gamma _ { k - 1 } ( x _ { k - 1 } ) \tilde { Q } _ { k - 1 } ( x _ { k - 1 } ) e ^ { r _ { k - 1 } ( x _ { k - 1 } ) } } { e ^ { r _ { k } ( x _ { k } ) } } \cdot \frac { p _ { k - 1 } ( x _ { k - 1 } ) } { p _ { k } ( x _ { k } ) } \cdot \frac { p _ { k } ( x _ { k } ) \tilde { P } _ { k - 1  k } ( x _ { k - 1 }  x _ { k } ) } { p _ { k - 1 } ( x _ { k - 1 } ) \tilde { Q } _ { k - 1  k } ( x _ { k - 1 }  x _ { k } ) } } \\ &  \phantom  \frac { \gamma _ { k - 1 } ( x _ { k - 1 } ) ( x _ { k } ) e ^ { r _ { k - 1 } ( x _ { k - 1 } ) } } { e ^ { r _ { k } ( x _ { k } ) } } \cdot \frac { \tilde { P } _ { k - 1  k } ( x _ { k - 1 }  x _ { k } ) }  \tilde { Q } _ { k - 1  k } ( x _  k - \end{array}
$$

which is $\operatorname { E q . } \left( 1 0 \right)$ ; it is finite $\stackrel {  } { Q } _ { k - 1 | k } { - } \mathrm { a . s }$ . because $\bar { P } _ { k - 1 | k } \ll \bar { Q } _ { k - 1 | k }$ . The boundary conditions give $\gamma _ { K } = p _ { \mathrm { r e f } } = \pi _ { K }$ , which is normalised, and $\gamma _ { 0 } = p _ { 0 } e ^ { r _ { 0 } }$ with $Z _ { 0 } < \infty$ . Algorithm 1 is therefore the SMC sampler of Section 3.1 with adaptive resampling and $\pi _ { k }$ -invariant rejuvenation, and its output is $\hat { \gamma } _ { 0 }$ of $\operatorname { E q . } \left( 5 7 \right)$ , evaluated at $f = e ^ { - r _ { 0 } } \mathbf { 1 } _ { E }$ . By the unbiasedness of $\hat { \gamma } _ { 0 } [ f ]$ under adaptive resampling and invariant moves (Del Moral et al., 2012),

$$
\mathbb { E } [ \hat { p } _ { 0 } [ E ] ] = \gamma _ { 0 } \big [ e ^ { - r _ { 0 } } \mathbf { 1 } _ { E } \big ] = \int p _ { 0 } ( x ) e ^ { r _ { 0 } ( x ) } e ^ { - r _ { 0 } ( x ) } \mathbf { 1 } _ { E } ( x ) \mathrm { d } x = p _ { 0 } [ E ] ,\tag{28}
$$

which is also $\mathbb { E } [ \hat { p } _ { 0 } ^ { \mathrm { M C } } [ E ] ]$ since $X _ { 0 } ^ { n } \sim p _ { 0 }$ in Eq. (3). This holds for every $K , N ,$ reward sequence and resampling schedule. For the sample, ${ \hat { p } } _ { 0 } [ \cdot \mid E ]$ is πˆ reweighted by $e ^ { - r _ { 0 } } ]$ and normalised, i.e. the self-normalised estimator of Eq. (57) for the target $\gamma _ { 0 } e ^ { - \bar { r _ { 0 } } } \mathbf { 1 } _ { E } \propto \bar { p _ { 0 } } [ \cdot \mid E ]$ , so it converges almost surely to $p _ { 0 } [ \cdot \mid E ]$ as $N \to \infty$ by the same result. For the limit, $p _ { 0 } \Rightarrow p _ { \mathrm { d a t a } }$ as $K  \infty$ under the standing assumptions of Section 2, and $p _ { \mathrm { d a t a } } [ \partial E ] = 0 , \mathsf { s o } p _ { 0 } [ E ] \to p _ { \mathrm { d a t a } } [ E ]$ by the portmanteau theorem. □

## A.4 PROOF OF THEOREM 2

Proof. Step 1: the estimator is an importance-sampling average. Without resampling, $\hat { Z } = 1$ and $\begin{array} { r } { \hat { p } _ { 0 } [ E ] = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } W ^ { n } } \end{array}$ with $W ^ { n } : = W ( X _ { K : 0 } ^ { n } )$ i.i.d. under $\mathbb { Q } _ { 0 : K }$ , where by the telescoping identity Eq. (22)

$$
\begin{array} { l } { { \displaystyle W ( { \boldsymbol x } _ { K : 0 } ) : = \prod _ { k = 1 } ^ { K } g _ { k - 1 | k } ( { \boldsymbol x } _ { k - 1 } , { \boldsymbol x } _ { k } ) ~ { \frac { \mathbf { 1 } _ { E } ( { \boldsymbol x } _ { 0 } ) } { e ^ { r _ { 0 } ( { \boldsymbol x } _ { 0 } ) } } } } } \\ { ~ = \mathbf { 1 } _ { E } ( { \boldsymbol x } _ { 0 } ) { \frac { d \mathbb { P } _ { 0 : K } } { d \mathbb { Q } _ { 0 : K } } } ( { \boldsymbol x } _ { K : 0 } ) } \\ { ~ = p _ { 0 } [ E ] { \frac { d \mathbb { P } _ { 0 : K } ^ { E } } { d \mathbb { Q } _ { 0 : K } } } ( { \boldsymbol x } _ { K : 0 } ) . } \end{array}\tag{29}
$$

Hence $\mathbb { E } _ { \mathbb { Q } _ { 0 : K } } [ W ] = p _ { 0 } [ E ]$ and

$$
\frac { \mathrm { V a r } [ \hat { p } _ { 0 } [ E ] ] } { p _ { 0 } [ E ] ^ { 2 } } = \frac { 1 } { N } \Big ( \mathbb { E } _ { \mathbb { Q } _ { 0 : K } } \Big [ \Big ( \frac { d \mathbb { P } _ { 0 : K } ^ { E } } { d \mathbb { Q } _ { 0 : K } } \Big ) ^ { 2 } \Big ] - 1 \Big ) = \frac { \chi ^ { 2 } \big ( { \mathbb { P } _ { 0 : K } ^ { E } } \mid \Vert \mathbb { Q } _ { 0 : K } \big ) } { N } .\tag{30}
$$

Step 2: change of measure through the tilted chain. Both $\mathbb { P } _ { 0 : K } ^ { E }$ and $\mathbb { P } _ { 0 : K } ^ { \star }$ are tilts of $\mathbb { P } _ { 0 : K }$ by functions of $x _ { 0 }$ , so

$$
\frac { d \mathbb { P } _ { 0 : K } ^ { E } } { d \mathbb { P } _ { 0 : K } ^ { \star } } ( x _ { K : 0 } ) = \frac { Z _ { 0 } e ^ { - r _ { 0 } ( x _ { 0 } ) } \mathbf { 1 } _ { E } ( x _ { 0 } ) } { p _ { 0 } [ E ] } \ \leq \ \frac { Z _ { 0 } } { c p _ { 0 } [ E ] } ,\tag{31}
$$

using $e ^ { r _ { 0 } } \geq c$ on E. Therefore

$$
\begin{array} { r l } & { 1 + { \chi ^ { 2 } } ( { \mathbb P } _ { 0 : K } ^ { E } | | { \mathbb Q } _ { 0 : K } ) = { \mathbb E } _ { { \mathbb Q } _ { 0 : K } } \Big [ \Big ( \frac { d { \mathbb P } _ { 0 : K } ^ { E } } { d { \mathbb P } _ { 0 : K } ^ { \star } } \Big ) ^ { 2 } \Big ( \frac { d { \mathbb P } _ { 0 : K } ^ { \star } } { d { \mathbb Q } _ { 0 : K } } \Big ) ^ { 2 } \Big ] } \\ & { \qquad \leq \Big ( \frac { Z _ { 0 } } { c p _ { 0 } [ E ] } \Big ) ^ { 2 } \big ( 1 + { \chi ^ { 2 } } ( { \mathbb P } _ { 0 : K } ^ { \star } | | { \mathbb Q } _ { 0 : K } ) \big ) , } \end{array}\tag{32}
$$

and substituting into Eq. (30) gives Eq. (15).

Step 3: the equality case. If $e ^ { r _ { 0 } } = c { \bf 1 } _ { E }$ , then $Z _ { 0 } = c p _ { 0 } [ E ]$ and Eq. (31) is identically one, so $\mathbb { P } _ { 0 : K } ^ { \star } = \mathbb { P } _ { 0 : K } ^ { E }$ and Eq. (15) holds with equality.

Step 4: thefloor. Under $\mathbb { Q } _ { 0 : K } , X _ { K } \sim p _ { \mathrm { r e f } } ;$ under $\mathbb { P } _ { 0 : K } ^ { \star } , X _ { K } \sim \pi _ { K } ^ { \star }$ by Lemma 2. The data-processing inequality for the marginal at step K gives

$$
\begin{array} { r } { \chi ^ { 2 } ( \mathbb { P } _ { 0 : K } ^ { \star } \| \mathbb { Q } _ { 0 : K } ) \geq \chi ^ { 2 } ( \pi _ { K } ^ { \star } \| p _ { \mathrm { r e f } } ) . } \end{array}\tag{33}
$$

For the upper bound, $\mathbb { P } _ { 0 : K } ^ { \star }$ and $\mathbb { P } _ { 0 : K }$ share their transitions and differ only in the law of $X _ { 0 } ,$ , and $X _ { K } \sim p _ { \mathrm { r e f } }$ under $\mathbb { P } _ { 0 : K }$ , so data processing for the marginal at K and then the fact that $d \mathbb { P } _ { 0 : K } ^ { \star } / d \mathbb { P } _ { 0 : K }$ depends on $x _ { 0 }$ alone give

$$
\chi ^ { 2 } ( \pi _ { K } ^ { \star } \| p _ { \mathrm { r e f } } ) \le \chi ^ { 2 } ( \mathbb { P } _ { 0 : K } ^ { \star } \| \mathbb { P } _ { 0 : K } ) = \mathbb { E } _ { \mathbb { P } _ { 0 : K } } \Big [ \Big ( \frac { e ^ { r _ { 0 } ( X _ { 0 } ) } } { Z _ { 0 } } \Big ) ^ { 2 } \Big ] - 1 = \chi ^ { 2 } ( \pi _ { 0 } \| p _ { 0 } ) .\tag{34}
$$

$\mathrm { I f } \stackrel {  } { Q } _ { k - 1 | k } = \stackrel {  } { P } _ { k - 1 | k } ^ { \star }$ for all k, the product in Eq. (24) is one, $d \mathbb { P } _ { 0 : K } ^ { \star } / d \mathbb { Q } _ { 0 : K }$ is a function of $x _ { K }$ alone, equal to $\pi _ { K } ^ { \star } ( x _ { K } ) / \dot { p } _ { \mathrm { r e f } } ( x _ { K } )$ , and the first inequality is an equality, which is Eq. (33) with its equality case. If $X _ { K }$ is independent of $X _ { 0 }$ under $\mathbb { P } _ { 0 : K }$ then $r _ { K } ^ { \star }$ is constant, so $\pi _ { K } ^ { \star } = p _ { \mathrm { r e f } }$ and the minimum is zero.

Finally, for the CDF reward of Eq. $( 1 6 ) , \phi \geq \tau$ on E gives $\begin{array} { r } { e ^ { r _ { 0 , \delta } } \ge \Phi ( 0 ) = \frac { 1 } { 2 } } \end{array}$ there, so $\begin{array} { r } { c = { \frac { 1 } { 2 } } } \end{array}$ , and $Z _ { 0 } = { \dot { p } } _ { 0 } [ e ^ { r _ { 0 , \delta } } ] \to p _ { 0 } [ E ]$ as $\delta \downarrow 0$ by dominated convergence. □

Remark 1. Initialising from $\pi _ { K } ^ { \star }$ instead of $p _ { \mathrm { r e f } }$ , with kernels $\stackrel {  } { P } _ { k - 1 | k } ^ { \star }$ and weights $w _ { K } ^ { n } = 1$ , would make every incremental weight equal to one and the estimator deterministic, the classical zerovariance statement for Doob’s h-transform. It is circular: the normaliser of $\pi _ { K } ^ { \star }$ is $Z _ { 0 }$ , which for the indicator reward is the quantity being estimated. The boundary condition $r _ { K } = 0$ is what allows Algorithm 1 to start from $p _ { \mathrm { r e f } }$ and Eq. (33) is the price this incurs.

## A.5 THE OPTIMAL REWARD RECURSION

We recall, for completeness, why the Doob rewards are the optimal intermediate targets (Guarniero et al., 2017; Johansen & Doucet, 2008; Pitt & Shephard, 1999). Fix k and $r _ { k - 1 }$ , let $X _ { k - 1 } \mid X _ { k } \sim$ ${ \overline { { Q } } } _ { k - 1 | k } .$ , and define

$$
r _ { k } ^ { \mathrm { o p t } } ( x _ { k } ) : = \log \int \bar { P } _ { k - 1 | k } ( \mathrm { d } x _ { k - 1 } \mid x _ { k } ) e ^ { r _ { k - 1 } ( x _ { k - 1 } ) } .\tag{35}
$$

From Eq. (10), for every proposal with $\overline { { P } } _ { k - 1 | k } \ll \overline { { Q } } _ { k - 1 | k }$

$$
\mathbb { E } \big [ g _ { k - 1 | k } \bigm | X _ { k } = x _ { k } \bigm ] = e ^ { - r _ { k } ( x _ { k } ) } \int \bar { Q } _ { k - 1 | k } \big ( \mathrm { d } x _ { k - 1 } \bigm | x _ { k } \bigm ) \frac { \tilde { P } _ { k - 1 | k } \big ( x _ { k - 1 } \bigm | x _ { k } \bigm ) } { \bar { Q } _ { k - 1 | k } \big ( x _ { k - 1 } \bigm | x _ { k } \bigm ) } e ^ { r _ { k - 1 } ( x _ { k - 1 } ) }
$$

$$
= \exp \big ( r _ { k } ^ { \mathrm { o p t } } ( x _ { k } ) - r _ { k } ( x _ { k } ) \big ) .\tag{36}
$$

By the law of total variance,

$$
\operatorname { V a r } [ g _ { k - 1 \mid k } ] = \operatorname { V a r } \left[ \mathbb { E } [ g _ { k - 1 \mid k } \mid X _ { k } ] \right] + \mathbb { E } \left[ \operatorname { V a r } [ g _ { k - 1 \mid k } \mid X _ { k } ] \right] .\tag{37}
$$

The reward $r _ { k }$ enters $g _ { k - 1 | k }$ only through the factor $e ^ { - r _ { k } ( x _ { k } ) }$ , a function of $x _ { k }$ alone, so the second term does not depend on $r _ { k }$ , while the first vanishes if and only i $: r _ { k } = r _ { k } ^ { \mathrm { o p t } } +$ const. Thus $r _ { k } ^ { \mathrm { o p t } }$ minimises $\mathrm { V a r } [ g _ { k - 1 | k } ]$ over $r _ { k }$ for every proposal. The second term vanishes if and only if $x _ { k - 1 } \mapsto$ $e ^ { r _ { k - 1 } ( x _ { k - 1 } ) } { \bar { P } } _ { k - 1 \mid k } ( x _ { k - 1 } \mid x _ { k } ) / { \bar { Q } } _ { k - 1 \mid k } ( x _ { k - 1 } \mid x _ { k } )$ is $\stackrel {  } { Q } _ { k - 1 | k } { - } \mathrm { a . s }$ . constant, i.e. $\overleftarrow { Q } _ { k - 1 | k } ( \mathrm { d } x _ { k - 1 } \ |$ $x _ { k } ) \propto \hat { P } _ { k - 1 | k } ( \mathrm { d } x _ { k - 1 } \mid x _ { k } ) e ^ { r _ { k - 1 } ( x _ { k - 1 } ) }$ , which with $r _ { k } = r _ { k } ^ { \mathrm { o p t } }$ is the twisted kernel $\stackrel {  } { P } _ { k - 1 | k } ^ { \star }$ and gives $g _ { k - 1 | k } \equiv 1$ . Finally, if $r _ { k - 1 } = r _ { k - 1 } ^ { \star }$ then $r _ { k } ^ { \mathrm { o p t } } = r _ { k } ^ { \star }$ by the tower identity Eq. (21), so iterating from $r _ { 0 } = r _ { 0 } ^ { \star }$ recovers the Doob rewards at every step.

## A.6 PROOF OF LEMMA 1

Proof. Write $\mu : = m _ { k } ( x _ { k } ) , \Delta : = c _ { k } \nabla _ { x } r _ { k } ( x _ { k } )$ and $\boldsymbol { v } : = \boldsymbol { v } _ { k }$ . Under Eq. (11) both kernels are Gaussian in $x _ { k - 1 }$ with covariance vI, so

$$
\log \frac { \tilde { P } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } { \tilde { Q } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } = - \frac { \| x _ { k - 1 } - \mu \| ^ { 2 } } { 2 v } + \frac { \| x _ { k - 1 } - \mu - \Delta \| ^ { 2 } } { 2 v } = \frac { \| \Delta \| ^ { 2 } } { 2 v } - \frac { \langle \Delta , x _ { k - 1 } - \mu \rangle } { v } .\tag{38}
$$

Substituting $x _ { k - 1 } = \mu + \Delta + \sqrt { v } \epsilon _ { k }$ , the noise drawn when sampling ${ \bar { Q } } _ { k - 1 | k }$ ,

$$
\begin{array} { r l r } {  { \log \frac { \overleftarrow { P } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } { \overleftarrow { Q } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } = \frac { \| \Delta \| ^ { 2 } } { 2 v } - \frac { \| \Delta \| ^ { 2 } } { v } - \frac { \langle \Delta , \epsilon _ { k } \rangle } { \sqrt { v } } } } \\ & { } & { = - \frac { \zeta _ { k } ^ { 2 } } { 2 } \big \| \nabla _ { x } r _ { k } ( x _ { k } ) \big \| ^ { 2 } - \zeta _ { k } \big \langle \nabla _ { x } r _ { k } ( x _ { k } ) , \epsilon _ { k } \big \rangle , } \end{array}\tag{39}
$$

with $\zeta _ { k } = c _ { k } / \sqrt { v _ { k } }$ . Adding the reward tilt $r _ { k - 1 } ( x _ { k - 1 } ) - r _ { k } ( x _ { k } )$ of Eq. (10) gives Eq. (12). No approximation is made; the integrator enters only through $( c _ { k } , v _ { k } )$ . For Euler-Maruyama on a step of length $\Delta t _ { k } , m _ { k } ( x _ { k } ) = x _ { k } - \bar { [ \mu _ { t _ { k } } ( x _ { k } ) - \sigma _ { t _ { k } } ^ { 2 } \nabla _ { x } }$ <sub>x</sub> log $p _ { t _ { k } } ( x _ { k } ) ] \Delta t _ { k }$ , the guidance shifts the mean by $c _ { k } \nabla _ { x } r _ { k } ( x _ { k } )$ with $c _ { k } = \sigma _ { t _ { k } } ^ { 2 } \Delta t _ { k }$ , and $v _ { k } = \sigma _ { t _ { k } } ^ { 2 } \Delta t _ { k } , \mathrm { s o } \zeta _ { k } = \sigma _ { t _ { k } } \sqrt { \Delta t _ { k } }$ . The exponential-integrator constants are derived in App. D. □

Scope. Three cases fall outside Lemma 1. Multi stage schemes such as the stochastic Heun sampler of (Karras et al., 2022) or higher order stochastic solvers of (Gonzalez et al., 2023) evaluate the network at an additional noise-dependent interval point, so the kernel mean is not affine in a single score evaluation. Deterministic solvers have $v _ { k } ~ = ~ 0$ , giving degenerate kernels and $\zeta _ { k }  \infty$ Anisotropic or per dimension noise makes $v _ { k }$ a matrix, and the scalar $\zeta _ { k }$ is now a matrix.

## B DIFFUSION MODEL DETAILS

Training objective. The score network $s _ { \theta } ( x _ { t } , t )$ approximates the marginal score $\nabla _ { x _ { t } }$ log $p _ { t }$ and is trained by denoising score matching (Song et al., 2021b),

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathbb { E } _ { t } \mathbb { E } _ { x _ { 0 } \sim p _ { 0 } } \mathbb { E } _ { x _ { t } \sim p ( x _ { t } \mid x _ { 0 } ) } \Big [ \lambda _ { t } \big \lVert s _ { \theta } ( x _ { t } , t ) - \nabla _ { x _ { t } } \log p ( x _ { t } \mid x _ { 0 } ) \big \rVert ^ { 2 } \Big ] , } \end{array}\tag{40}
$$

with weighting $\lambda _ { t } > 0$ and $t \sim \mathcal { U } [ 0 , 1 ]$ . For a Gaussian forward kernel $p ( x _ { t } \mid x _ { 0 } ) = \mathcal { N } ( \alpha _ { t } x _ { 0 } , \bar { \sigma } _ { t } ^ { 2 } I )$ the target is closed form, $\nabla _ { x _ { t } } \log { p ( x _ { t } \mid x _ { 0 } ) } = - \epsilon / \bar { \sigma } _ { t }$ , giving the equivalent noise-prediction objective. Score-, noise-, and velocity-prediction parametrisations are related by simple bijections (Gao et al., 2024).

VP-SDE schedule. We adopt the Variance-Preserving (VP) SDE (Song et al., 2021b), with $f _ { t } ( x ) =$ $- \textstyle { \frac { 1 } { 2 } } \beta ( t ) x$ and $\sigma _ { t } = \sqrt { \beta ( t ) }$ for a prescribed $\beta ( t )$ . The forward kernel then admits the closed form,

$$
X _ { t } = \alpha _ { t } X _ { 0 } + \bar { \sigma } _ { t } \epsilon , \qquad \epsilon \sim \mathcal { N } ( 0 , I )
$$

with,

$$
\alpha _ { t } = \exp ( - \textstyle { \frac { 1 } { 2 } } \int _ { 0 } ^ { t } \beta ( s ) d s ) , \qquad \bar { \sigma } _ { t } ^ { 2 } = 1 - \alpha _ { t } ^ { 2 } .\tag{41}
$$

In experiments we use the linear schedule $\beta ( t ) = \beta _ { \operatorname* { m i n } } + t ( \beta _ { \operatorname* { m a x } } - \beta _ { \operatorname* { m i n } } ) , \beta _ { \operatorname* { m i n } } = 0 . 1 , \beta _ { \operatorname* { m a x } } = 2 0$ or a cosine schedule.

Posterior mean (Tweedie). The posterior mean of the clean state given a noisy $x _ { t }$ is available in closed form from the score (Efron, 2011),

$$
\hat { x } _ { 0 \mid t } ( x _ { t } ) = \frac { x _ { t } + \bar { \sigma } _ { t } ^ { 2 } s _ { \theta } ( x _ { t } , t ) } { \alpha _ { t } } .\tag{42}
$$

We use $\hat { x } _ { 0 \mid t }$ as the mean of the Gaussian denoising-posterior approximation in Section 3.2.

Posterior-variance proxy (ΠGDM). For the covariance of the Gaussian denoising posterior we use the ΠGDM proxy, which is exact when the data distribution is a single Gaussian and is borrowed as a tractable surrogate otherwise (Song et al., 2023). Per dimension,

$$
s _ { t } ^ { 2 } = \frac { \bar { \sigma } _ { t } ^ { 2 } V _ { \mathrm { d a t a } } } { \alpha _ { t } ^ { 2 } V _ { \mathrm { d a t a } } + \bar { \sigma } _ { t } ^ { 2 } } ,\tag{43}
$$

with $\alpha _ { t } ^ { 2 }$ and $\bar { \sigma } _ { t } ^ { 2 } = 1 - \alpha _ { t } ^ { 2 }$ the VP-SDE marginal noise variance above (note $\bar { \sigma } _ { t }$ is the marginal noise std, distinct from the diffusion coefficient $\sigma _ { t } = \sqrt { \beta ( t ) } )$ , and $V _ { \mathrm { d a t a } } = \mathrm { V a r } ( x _ { 0 } )$ the marginal data variance (for the GMM benchmark, $\begin{array} { r } { \sum _ { k } w _ { k } ( \Sigma _ { k , 1 1 } + \mu _ { k , 1 } ^ { 2 } ) - ( \sum _ { k } w _ { k } \mu _ { k , 1 } ) ^ { 2 } } \end{array}$ along the read-out dimension). At early times $( t \to T ) \alpha _ { t } \to 0 , \bar { \sigma } _ { t } \to 1$ and $s _ { t } ^ { 2 } \to V _ { \mathrm { d a t a } }$ (full prior uncertainty); at late times $( t  0 ) \bar { \sigma } _ { t } ^ { 2 }  \bar { 0 }$ and $s _ { t } ^ { 2 } \to 0$ (the posterior collapses to the Tweedie point estimate). Applied per dimension this yields the diagonal posterior covariance,

$$
\begin{array} { r } { \Sigma _ { t } = \operatorname { d i a g } \big ( s _ { t , 1 } ^ { 2 } , \ldots , s _ { t , d } ^ { 2 } \big ) , } \end{array}\tag{44}
$$

with $V _ { \mathrm { d a t a } }$ evaluated along each coordinate $( V _ { i } = 1$ for standardised data, otherwise estimated empirically).

## C SEQUENTIAL METHODS

For completeness we provide an overview of classical sequential methods that are used in rare-event simulation, such as annealed importance sampling, splitting and sequential Monte Carlo (SMC). These are methods used to bridge the gap from a tractable reference $\pi _ { K }$ to the tilted target $\pi _ { 0 }$ of Eq. (5) through a sequence of overlapping intermediate distributions (Del Moral et al., 2006; Kahn & Harris, 1951; Neal, 2001). We briefly expand on this and highlight their similarities, but refer the readers to (Moral, 2004) for a full exposition under the Feynman-Kac formulation. Additionally, see (Del Moral et al., 2006; Naesseth et al., 2019) for details on the SMC machinery. Formally the sequence of distributions defines a path-measure over the space of all trajectories particles can take, but we omit this rigour for brevity.

Annealed Importance Sampling (AIS) and Sequential Monte Carlo (SMC) build this path by tempering the reward (Del Moral et al., 2006; Neal, 2001). One such sequence of distributions tempers the reward, (Del Moral et al., 2006; Neal, 2001),

$$
\pi _ { k } ( x ) = { \frac { 1 } { Z _ { k } } } p _ { 0 } ( x ) e ^ { \beta _ { k } r ( x ) } , \qquad 1 = \beta _ { 0 } > \beta _ { 1 } > \cdot \cdot \cdot > \beta _ { K } = 0 .\tag{45}
$$

Rather than targeting the constrained target $\pi _ { 0 }$ directly, tempering gradually sharpens the reward landscape until the chain converges to the target distribution. Additionally, SMC utilises resampling along the trajectory to prune low-weight particles and multiply those with higher weights (Del Moral et al., 2006; Doucet et al., 2000).

Multilevel splitting takes another approach, where the sequence is constructed using a sequence of nested rare event sets (Glasserman et al., 1999; Kahn & Harris, 1951). Taking the canonical example of threshold exceedance, for a strictly decreasing ladder $\tau _ { 0 } = \tau > \tau _ { 1 } > \cdot \cdot \cdot > \tau _ { K }$ , the event set is defined as $E _ { k } = \{ x : \phi ( x ) \geq \tau _ { k } \}$ . The sequence is defined to be,

$$
E _ { 0 } = E \subset E _ { 1 } \subset \cdots \subset E _ { K } , \qquad \pi _ { k } ( x ) = { \frac { 1 } { Z _ { k , \tau } } } p _ { 0 } ( x ) \mathbf { 1 } \{ \phi ( x ) \geq \tau _ { k } \} ,\tag{46}
$$

where $E _ { K }$ is a region readily accessible from the state space. The aim of splitting is to estimate the rare event probability at the final τ, i.e. $\begin{array} { r } { Z _ { 0 , \tau } = \int p _ { 0 } ( x ) \hat { \mathbf { 1 } } \{ \phi ( x ) \ge \tau _ { 0 } \} d x \stackrel {  } { = } p _ { 0 } [ \boldsymbol { E } ] } \end{array}$ . We show that all of these methods compute this in the same way. All methods work by transporting particles along the sequence, SMC and splitting additionally resample particles to focus computation on high-weight trajectories (i.e. trajectories that particles undertake that are more likely under the possible sequence of trajectories they could have taken)(Del Moral et al., 2006; Glasserman et al., 1999; Kahn & Harris, 1951). Intuitively, multilevel splitting is a form of SMC/AIS with a hard indicator function replacing a smoothed distribution and the schemes only differ in their construction of their sequences (Moral, 2004).

## C.1 ESTIMATING THE NORMALISATION CONSTANT

Having constructed the sequence, the quantity of interest is the terminal normalisation constant $Z _ { 0 } .$ which for the rare-event target is $Z _ { 0 , \tau }$ . We show explicitly, that splitting estimates the probability estimate in the same way as standard SMC/AIS. Please see (Del Moral et al., 2006; Glasserman et al., 1999; Kahn & Harris, 1951; Neal, 2001) for further details.

For a sequence of distributions, $\pi _ { K } \to \cdot \cdot \cdot \to \pi _ { 0 }$ of Section 3.1, let $\begin{array} { r } { \pi _ { k } = \frac { 1 } { Z _ { k } } \gamma _ { k } } \end{array}$ be the normalised density with the normalisation constant $Z _ { k } ~ = ~ \int \gamma _ { k }$ . Each step along the trajectory incurs an incremental importance weight that is, in general, a ratio of forward and backward transition kernels between the two distributions (Del Moral et al., 2006). A particle is propagated from $\pi _ { k }$ to $\pi _ { k - 1 }$ by the proposal kernel $\overline { { B } } _ { k - 1 | k } \big ( \boldsymbol { x } _ { k - 1 } \mid \boldsymbol { x } _ { k } \big )$ , and the auxiliary kernel ${ \vec { F } } _ { k \mid k - 1 } ( x _ { k } \mid x _ { k - 1 } )$ ) running in the opposite direction defines the incremental weight of Eq. (7) (Del Moral et al., 2006),

$$
g _ { k - 1 | k } ( x _ { k - 1 } , x _ { k } ) = \frac { \gamma _ { k - 1 } ( x _ { k - 1 } ) \vec { F } _ { k | k - 1 } ( x _ { k } \mid x _ { k - 1 } ) } { \gamma _ { k } ( x _ { k } ) \vec { B } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } ,\tag{47}
$$

Choosing the proposal to be a $\pi _ { k - 1 }$ -invariant MCMC kernel, as in the rejuvenation move of Section 3.1, with the auxiliary kernel $F _ { k | k - 1 }$ taken as its time reversal, the kernels cancel and the weight reduces to the target ratio $g _ { k - 1 | k } = \dot { \gamma } _ { k - 1 } ( x _ { k } ) / \gamma _ { k } ( x _ { k } )$ (Del Moral et al., 2006; Neal, 2001), which we use throughout this subsection. The ratio of two normalisation constants $Z _ { k - 1 } / Z _ { k }$ is estimated via the Monte Carlo average of the incremental weights, i.e.,

$$
\mathbb { E } _ { \pi _ { k } } [ g _ { k - 1 | k } ] = \int \frac { \gamma _ { k - 1 } ( x ) } { \gamma _ { k } ( x ) } \frac { \gamma _ { k } ( x ) } { Z _ { k } } d x = \frac { Z _ { k - 1 } } { Z _ { k } } ,\tag{48}
$$

with the standard empirical estimator,

$$
\widehat { \frac { Z _ { k - 1 } } { Z _ { k } } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } g _ { k - 1 | k } ^ { ( i ) } , \qquad x ^ { ( i ) } \sim \pi _ { k } .\tag{49}
$$

Hence the normaliser $Z _ { 0 }$ follows from the telescoping product with the corresponding estimator given by,

$$
\hat { Z } _ { 0 } = Z _ { K } \prod _ { k = 1 } ^ { K } \Big ( \frac { 1 } { N } \sum _ { i = 1 } ^ { N } g _ { k - 1 | k } ^ { ( i ) } \Big ) ,\tag{50}
$$

where $Z _ { K }$ is known, e.g. for score based models $Z _ { K } = 1 \mathrm { a s } \gamma _ { K } = p _ { 1 }$ is the normalised Gaussian prior (Ho et al., 2020; Song et al., 2021b).

Tempering (AIS/SMC) Annealed importance sampling and SMC interpolate by tempering the reward, $\gamma _ { k } = p _ { 0 } e ^ { \beta _ { k } r }$ , with $1 = \beta _ { 0 } > \cdot \cdot \cdot > \beta _ { K } = 0$ , so the incremental weight is given by

$$
g _ { k - 1 | k } = e ^ { \left( \beta _ { k - 1 } - \beta _ { k } \right) r \left( x _ { k } \right) } , \qquad { \frac { Z _ { k - 1 } } { Z _ { k } } } = \mathbb { E } _ { \pi _ { k } } { \left[ e ^ { \left( \beta _ { k - 1 } - \beta _ { k } \right) r } \right] } .\tag{51}
$$

Each normalisation ratio is the average of this factor under the current target. When consecutive temperatures are close, i.e. $\beta _ { k - 1 } - \beta _ { k }$ small, the weights stay close to one for all particles and the ratio is estimated reliably. A single large step, by contrast can produce weights spanning many orders of magnitude - collapsing the effective sample size where the final estimate will be dominated by only a few particles that have the largest weights (Neal, 2001).

Splitting Multilevel splitting interpolates instead with hard indicators, i.e. $\gamma _ { k } = p _ { 0 } \mathbf { 1 } \{ x \in E _ { k } \}$ , for a shrinking sequence $\bar { E _ { K } } \supset \dot { \cdots } \supset \bar { E _ { 0 } }$ . The incremental weight collapses to,

$$
g _ { k - 1 | k } = { \frac { p _ { 0 } { \bf 1 } \{ x \in E _ { k - 1 } \} } { p _ { 0 } { \bf 1 } \{ x \in E _ { k } \} } } = { \bf 1 } \{ x \in E _ { k - 1 } \} ,\tag{52}
$$

for which the ratio of normalisation constant is given by,

$$
\frac { Z _ { k - 1 } } { Z _ { k } } = \mathbb { E } _ { \pi _ { k } } \bigl [ { \bf 1 } \{ x \in E _ { k - 1 } \} \bigr ] = \mathrm { P r } \bigl [ X \in E _ { k - 1 } \ | \ X \in E _ { k } \bigr ] = : \alpha _ { k } ,\tag{53}
$$

so each ratio is the conditional probability of reaching the next level, estimated by the surviving fraction, $\hat { \alpha } _ { k } = N _ { X \in E _ { k - 1 } } / N _ { X \in E _ { k } }$ . Thus the telescoping product of Eq. (50) for the threshold exceedance given in Eq. (46) becomes,

$$
Z _ { 0 } = Z _ { K } \prod _ { k = 1 } ^ { K } \alpha _ { k } \ = \ \mathrm { P r } [ X > \tau _ { K } ] \prod _ { k = 1 } ^ { K } \mathrm { P r } [ X > \tau _ { k - 1 } \mid X > \tau _ { k } ] ,\tag{54}
$$

This recovers the rare event probability estimate of the multilevel splitting algorithm (Morio et al., 2014; Rubino et al., 2009).

Both methods estimate the same object via a product of incremental ratios along the bridge where consecutive levels overlap, instead of the single rare-event ratio estimated in one step. This is precisely the failure that accumulating importance weights over the trajectory sidesteps. The diffusion process supplies the sequence of transition kernels and the incremental weights (Doucet et al., 2022; Vargas et al., 2023) carry their own normalisation avoiding a separate estimate of $Z _ { 0 } ,$ , see Appendix C.3.

## C.2 GENERIC SMC ALGORITHM

Given an unnormalised density $\gamma _ { 0 }$ with a normalised form $\pi _ { 0 } : = \gamma _ { 0 } / \gamma _ { 0 } [ 1 ]$ ], we seek to estimate integrals $\gamma _ { 0 } [ f ]$ and to sample from $\pi _ { 0 }$ . SMC samplers (Del Moral et al., 2006) transport a population of N weighted particles from a normalised reference $\gamma _ { K }$ down to $\gamma _ { 0 }$ along a sequence $\gamma _ { K } , \dots , \gamma _ { 0 }$ with $\pi _ { k } : = \gamma _ { k } / \gamma _ { k } [ 1 ]$

Algorithm. We begin by initialising $N$ samples, $X _ { K } ^ { n } \sim p _ { \mathrm { r e f } }$ with weights $w _ { K } ^ { n } = 1$ . At each step k, move each particle with a backward proposal kernel $B _ { k - 1 | k }$ and update its weight with the incremental weight $g _ { k - 1 | k }$

$$
X _ { k - 1 } ^ { n } \sim \bar { B } _ { k - 1 | k } ( \mathrm { d } x _ { k - 1 } \mid X _ { k } ^ { n } ) , \qquad w _ { k - 1 } ^ { n } = w _ { k } ^ { n } g _ { k - 1 | k } \big ( X _ { k - 1 } ^ { n } , X _ { k } ^ { n } \big ) ,\tag{55}
$$

$$
g _ { k - 1 | k } ( x _ { k - 1 } , x _ { k } ) = \frac { \gamma _ { k - 1 } ( x _ { k - 1 } ) \vec { F } _ { k | k - 1 } ( x _ { k } \mid x _ { k - 1 } ) } { \gamma _ { k } ( x _ { k } ) \overleftarrow { B } _ { k - 1 | k } ( x _ { k - 1 } \mid x _ { k } ) } ,\tag{56}
$$

where $\vec { F } _ { k \vert k - 1 }$ is a forward auxiliary kernel running from $k - 1$ to k. Any kernels can be selected, provided we can sample from $\bar { B } _ { k - 1 | k }$ and evaluate $g _ { k - 1 | k }$ pointwise. When the effective sample size $\mathrm { E S S } ( w _ { k - 1 } ) : = \| w _ { k - 1 } \| _ { 1 } ^ { 2 } / \| w _ { k - 1 } \| _ { 2 } ^ { 2 }$ drops below $\eta N$ , we resample, replacing each particle by an ancestor drawn in proportion to its weight and resetting the weights to one (Chopin et al., 2022), and rejuvenate using $\mathbf { a } \ \pi _ { k - 1 }$ -invariant MCMC kernel.

Estimators. Let $\hat { Z }$ be the product, over the resampling steps so far, of the average weight $\textstyle { \frac { 1 } { N } } \sum _ { n } w _ { k } ^ { n }$ just before the reset $( \hat { Z } = 1$ if none). The particle system at step k defines the weighted approximations

$$
\hat { \gamma } _ { 0 } [ f ] : = \hat { Z } \frac { 1 } { N } \sum _ { n = 1 } ^ { N } w _ { 0 } ^ { n } f ( X _ { 0 } ^ { n } ) , \qquad \hat { \pi } _ { 0 } ( \mathrm { d } x ) : = \frac { \sum _ { n = 1 } ^ { N } w _ { 0 } ^ { n } \delta _ { X _ { 0 } ^ { n } } ( \mathrm { d } x ) } { \sum _ { n = 1 } ^ { N } w _ { 0 } ^ { n } } .\tag{57}
$$

$\hat { \gamma } _ { 0 } [ f ]$ is unbiased, $\mathbb { E } [ \hat { \gamma } _ { k } [ f ] ] = \gamma _ { k } [ f ]$ for every $f .$ The second is biased for finite $N$ but converges almost surely to $\pi _ { k }$ as $N \to \infty$ (Del Moral et al., 2006).

## C.3 WHY ANNEALING BECOMES NECESSARY

We now motivate why annealing is required and where importance sampling alone is insufficient. Even with exact samples from the tilted target $\pi _ { 0 }$ of $\operatorname { E q . } \left( 5 \right)$ , taken with $r _ { 0 } = r _ { 0 , \delta }$ or the mollified reward. The naive endpoint (self-normalised) estimator of $p _ { 0 } [ E ]$ ] is badly behaved for estimating the normalisation constant, motivating the use of annealing, either via splitting or accumulated importance weights.

Given a smooth reward $r _ { 0 , \delta } ( x )$ , suppose we can draw exact samples $x \sim \pi _ { 0 }$ from the tilted distribution, $\pi _ { 0 } = \gamma _ { 0 } / Z _ { 0 }$ with $\gamma _ { 0 } = p _ { 0 } e ^ { r _ { 0 , \delta } }$ . The rare-event probability under the base distribution $p _ { 0 }$ can then be written as,

$$
\begin{array} { r l } & { p _ { 0 } [ E ] = \mathbb { E } _ { \pi _ { 0 } } \left[ { \mathbf 1 } \{ \phi ( x ) \geq \tau \} \frac { p _ { 0 } ( x ) } { \pi _ { 0 } ( x ) } \right] = \mathbb { E } _ { \pi _ { 0 } } \left[ { \mathbf 1 } \{ \phi ( x ) \geq \tau \} \frac { p _ { 0 } ( x ) } { p _ { 0 } ( x ) e ^ { r _ { 0 , \delta } ( x ) } / Z _ { 0 } } \right] } \\ & { \qquad = \mathbb { E } _ { \pi _ { 0 } } \left[ { \mathbf 1 } \{ \phi ( x ) \geq \tau \} Z _ { 0 } e ^ { - r _ { 0 , \delta } ( x ) } \right] , } \end{array}\tag{58}
$$

where $w ( x ) : = p _ { 0 } ( x ) / \pi _ { 0 } ( x ) = Z _ { 0 } e ^ { - r _ { 0 , \delta } ( x ) }$ is the importance weight, and the normaliser $Z _ { 0 }$ is estimated from the same samples via,

$$
\frac { 1 } { Z _ { 0 } } = \frac { 1 } { Z _ { 0 } } \int p _ { 0 } d x = \frac { 1 } { Z _ { 0 } } \int p _ { 0 } ( x ) e ^ { r _ { 0 } , \delta ( x ) } e ^ { - r _ { 0 } , \delta ( x ) } = \int \pi _ { 0 } e ^ { - r _ { 0 , \delta } ( x ) } = \mathbb { E } _ { \pi _ { 0 } } [ e ^ { - r _ { 0 , \delta } ( x ) } ] .\tag{59}
$$

Substituting Eq. (59) into Eq. (58) yields the self-normalised importance-sampling (SNIS) estimator (Owen, 2013),

$$
\hat { p } _ { 0 } [ E ] = \frac { \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ \phi ( x _ { i } ) \ge \tau \} e ^ { - r _ { 0 , \delta } ( x _ { i } ) } } { \sum _ { i = 1 } ^ { N } e ^ { - r _ { 0 , \delta } ( x _ { i } ) } } , \qquad x _ { i } \sim \pi _ { 0 } ( x )\tag{60}
$$

The numerator and denominator of (60) behave very differently. The numerator is only affected by the region that $\pi _ { 0 }$ approximates well, since the indicator $\mathbf { 1 } \{ \phi ( \bar { x } ) \geq \tau \}$ provides a “shieldi $\mathrm { \Delta } \Psi ^ { \prime }$ effect.

The denominator, by contrast is an expectation of the normalisation constant $Z _ { 0 }$ which is affected by the entire domain of $p _ { 0 }$ . As the tilt is designed to concentrate on the rare event set, we are effectively using the rare-event tilted distribution $\pi _ { 0 }$ to estimate a function of $p _ { 0 }$ . This naturally has a low ESS and yields a high-variance estimate of the normaliser $Z _ { 0 }$ . Even given perfect samples from $\pi _ { 0 }$ we essentially have another “rare-event” expectation in the denominator where no particles of $\pi _ { 0 }$ fall into the bulk. The denominator integrand $\begin{array} { r } { w ( x ) = \frac { p _ { 0 } ( x ) } { \pi _ { 0 } ( x ) } = Z _ { 0 } e ^ { - r _ { 0 , \delta } ( x ) } } \end{array}$ is largest in the bulk, where $p _ { 0 }$ holds most of its mass but the tilt $\pi _ { 0 }$ has almost none, exactly the region $\pi _ { 0 }$ is designed to avoid. The numerator $\mathbb { E } _ { q } [ \mathbf { 1 } _ { \mathrm { r a r e } } e ^ { - r _ { 0 , \delta } } ]$ stays bounded as it lives where q samples effectively, whereas the denominator $\mathbb { E } _ { q } [ e ^ { - r _ { 0 , \delta } } ]$ needs exactly the bulk samples that q avoids. Hence, the ESS for estimating the denominator becomes worse by design and this contributes a high variance estimate of $Z _ { 0 } !$ We could in practice break this down into a series of transition importance distributions, which leads u back to splitting/annealing of Section C (Del Moral et al., 2006; Morio et al., 2014; Neal, 2001).

## C.3.1 EMPIRICAL EVIDENCE ON THE 2D GMM

If we draw exact samples from $\pi _ { 0 } .$ , via Rejection Sampling or analytically, we illustrate that we still cannot estimate $Z _ { 0 }$ . For the 2D GMM case with a log-CDF tilting, we have access to the analytical normalisation constant, the convolution of a Gaussian with a Gaussian CDF simply inflates the variance. The normaliser of the tilted distribution as given in Eq. (98) repeated here again is given by,

$$
Z _ { 0 , \delta } = \int \Phi ( f ) \mathcal { N } ( f \mid \mu , \sigma ^ { 2 } ) d f = \Phi \left( \frac { \mu } { \sqrt { 1 + \sigma ^ { 2 } } } \right) ,\tag{61}
$$

with $\begin{array} { r } { f \mapsto \frac { x - \tau } { \delta } , \mu \mapsto \frac { \mu _ { i } - \tau } { \delta } } \end{array}$ and $\textstyle \sigma ^ { 2 } \mapsto { \frac { v _ { i } } { \delta ^ { 2 } } }$ (see Section F). This lets us disentangle whether the failure is in the numerator or the denominator, knowing the analytical denominator and the true probability, the true numerator is recoverable. We draw a large number of samples from $p _ { 0 }$ and use rejection sampling to generate samples from $\pi _ { 0 }$ . We find that a small number of particles drawn directly from $\pi _ { 0 }$ suffices to estimate the rare-event probability provided the denominator is known. Conversely, if the denominator is unknown we cannot estimate the probability at all - the difficulty lies entirely in the denominator as seen in Table 2, which is exactly what motivates annealing (Section 3.1 & C).

For $\tau = 7 . 0$ and $\delta = 0 . 2 5$ the true $Z _ { 0 , \delta } = 7 . 3 3 1 \times 1 0 ^ { - 3 }$ from Eq. (61).

Table 2: Self-normalised importance sampling (SNIS) estimates vs. sample size (N).
<table><tr><td>N</td><td>PSNIS</td><td>Relative Bias</td><td> $\hat { Z } _ { 0 , \mathrm { S N I S } }$ </td></tr><tr><td>100</td><td> $4 . 6 0 3 5 \times 1 0 ^ { - 1 }$ </td><td>+6,593%</td><td> $4 . 8 7 3 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>1,000</td><td> $3 . 5 8 5 4 \times 1 0 ^ { - 1 }$ </td><td>+5,113%</td><td> $3 . 7 7 0 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>10,000</td><td> $3 . 1 6 8 0 \times 1 0 ^ { - 1 }$ </td><td>+4,506%</td><td> $3 . 3 6 7 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>25,000</td><td> $1 . 9 8 0 4 \times 1 0 ^ { - 1 }$ </td><td> $+ 2 , 7 7 9 \%$ </td><td> $2 . 1 0 2 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>100</td><td> $6 . 9 2 5 0 \times 1 0 ^ { - 3 }$ </td><td> $+ 0 . 6 8 \%$ </td><td>(analytical  $Z _ { 0 , \delta } )$ </td></tr></table>

Note that pˆ<sub>SNIS</sub> differs from $\hat { Z } _ { 0 , \mathrm { S N I S } }$ because some particles do not pass through the numerator’s indicator by design of the smoothing in Section 3. The goal of annealing/splitting (Section 3.1) is precisely to make the numerator easier to estimate.

## C.4 SPLITTING ESTIMATOR

In the case where the path weights are not readily accessible, either because they are not computed or if a large number of discretisation is infeasible, e.g. for a second order numerical integrator. We can fall back to a guided diffusion proposal to build a splitting/annealed probability estimator. In this case, the annealing bridge becomes the nested sequence of event sets defined in Eq. (46). For the threshold exceedance example given in Sec C, the conditional ratio between adjacent levels is given by,

$$
\alpha _ { k } \ = \ p _ { 0 } [ X > \tau _ { h } \ | \ X > \tau _ { \ell } ] = { \frac { p _ { 0 } [ X > \tau _ { h } \cap X > \tau _ { \ell } ] } { p _ { 0 } [ X > \tau _ { \ell } ] } } = { \frac { p _ { 0 } [ X > \tau _ { h } ] } { p _ { 0 } [ X > \tau _ { \ell } ] } } .\tag{62}
$$

In the denoising-diffusion setting each conditional is sampled by transporting particles from the Gaussian prior to the corresponding target, this means that the probability, $\mathrm { P r } [ \breve { X } > \tau _ { h } ]$ is estimated

by generating samples from the tilted distribution, $p ( X \mid X > \tau _ { h } )$ , as highlighted in Section 3. Within this scheme $\alpha _ { k }$ admits two estimators, depending on whether a single tilt or one tilt per level is used.

Formulation I: single-tilt, two-indicator estimator. Use a single proposal $q _ { \ell }$ targeting the lower level $\tau _ { \ell }$ and estimate both exceedance probabilities from the same particles. With $\tau _ { \ell } < \tau _ { h }$

$$
\alpha _ { k } = \frac { \mathbb { E } _ { q _ { \ell } } \mathbf { 1 } \{ \phi ( X ) > \tau _ { h } \} e ^ { - r _ { \ell } ( X ) } ] } { \mathbb { E } _ { q _ { \ell } } \mathbf { 1 } \{ \phi ( X ) > \tau _ { \ell } \} e ^ { - r _ { \ell } ( X ) } ] } ,\tag{63}
$$

where the shared normaliser $Z _ { \ell }$ cancels between numerator and denominator, so it need not be estimated. The denominator is estimated where $q _ { \ell }$ has mass (it targets $\tau _ { \ell } )$ , so it is well-conditioned; the variance is dominated by the numerator, which lives where $q _ { \ell }$ is sparse $( X > \tau _ { h } )$

Formulation II: two-tilt transitioning. Use one proposal per level, $q \ell \propto p _ { 0 } e ^ { r \ell }$ and $q _ { h } \propto p _ { 0 } e ^ { r _ { h } }$ and estimate the numerator under $q _ { h }$ and the denominator under $q _ { \ell } .$

$$
\alpha _ { k } = \frac { Z _ { h } } { Z _ { \ell } } \frac { \mathbb { E } _ { q _ { h } } \bigl [ { \mathbf { 1 } \{ \phi ( X ) > \tau _ { h } \} } e ^ { - r _ { h } ( X ) } \bigr ] } { \mathbb { E } _ { q _ { \ell } } \bigl [ { \mathbf { 1 } \{ \phi ( X ) > \tau _ { \ell } \} } e ^ { - r _ { \ell } ( X ) } \bigr ] } .\tag{64}
$$

Each expectation is now evaluated under a tilt matched to its own threshold, so neither term suffers the ESS collapse of Formulation I; the price is that the normalisers no longer cancel and the ratio $Z _ { h } / Z _ { \ell }$ must be estimated. Annealing on the output space, i.e. between $p _ { 0 } e ^ { r _ { \ell } } \to p _ { 0 } e ^ { r _ { h } }$ , we can compute this ratio with the tempering estimator given in Eq. (51),

$$
\frac { Z _ { h } } { Z _ { \ell } } = \mathbb { E } _ { q _ { \ell } } \bigl [ e ^ { r _ { h } ( X ) - r _ { \ell } ( X ) } \bigr ] \approx \frac { 1 } { N } \sum _ { i } e ^ { r _ { h } ( x _ { i } ) - r _ { \ell } ( x _ { i } ) } , \qquad x _ { i } \sim q _ { \ell } ,\tag{65}
$$

which is reliable when adjacent tilts overlap. Both formulations require sufficient overlap of the thresholds, the only difference is Formulation I requires overlap between two hard indicators and through the smooth reward difference, $r _ { h } - r _ { \ell }$ , in Formulation II.

Notable Caveats: Evolving directly from $p ( X \mid X > \tau _ { l } ) \to p ( X \mid X > \tau _ { h } )$ is non-trivial without additional training. Unlike traditional splitting, we do not evolve/reuse particles across different thresholds (Morio et al., 2014), hence the above formulations are not a form of sequential methods. Instead they are a sequence of independent estimators sharing a compute budget. Additionally, any importance weights used assume that within the threshold set by the indicators, samples are well distributed along the tilted distribution. As such both estimators are sensitive to the quality of the guidance. Training based methods could be introduced such as Flow Matching (Albergo & Vanden-Eijnden, 2023; Lipman et al., 2023; Liu et al., 2023) or Schrödinger’s Bridges (Bortoli et al., 2021; Shi et al., 2023) but these remain beyond the scope of this work.

## D WEIGHT COMPUTATION

Let P and Q be the path-measures on $[ 0 , 1 ]$ of the base reverse-time SDE of Eq. (1) and the guided SDE of Eq. (9) respectively. Here we do not assume that Q is the ideal Doob’s path measure $\mathbb { P } ^ { * }$ and the gap between the two is varaince in Theorem 2. Both measures start from $\bar { X } _ { 1 } \sim p _ { \mathrm { r e f } }$ and differ only the drift obtained by the reward gradient term, $, - \sigma _ { t } ^ { 2 } \nabla _ { x } r _ { t }$ so Girsanov’s theorem gives us an exact form of the Radon-Nikodym Derivative (Øksendal, 2003),

$$
\frac { \mathrm { d } \mathbb { P } } { \mathrm { d } \mathbb { Q } } = \exp \Big ( - \int _ { 0 } ^ { 1 } \sigma _ { t } \nabla _ { x } r _ { t } ( X _ { t } ) \cdot \mathrm { d } \overleftarrow { W } _ { t } ~ - ~ \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \sigma _ { t } ^ { 2 } \| \nabla _ { x } r _ { t } ( X _ { t } ) \| ^ { 2 } \mathrm { d } t \Big ) ,\tag{66}
$$

where $\hat W _ { t }$ is the reverse-time Wiener process driving the guided sampler. This expression is valid provided the change of measure exists at all. A sufficient condition for existence is given by Novikov’s condition,

$$
\begin{array} { r } { \mathbb E _ { \mathbb Q } \Big [ \exp \left( \frac { 1 } { 2 } \displaystyle \int _ { 0 } ^ { 1 } \sigma _ { t } ^ { 2 } \left. \nabla _ { x } r _ { t } ( X _ { t } ) \right. ^ { 2 } \mathrm d t \right) \Big ] < \infty . } \end{array}\tag{67}
$$

The discrete estimator of Algorithm 1 does not require this condition to be fulfilled as Theorem 1 requires only $\bar { P } _ { k - 1 | k } \ll \bar { Q } _ { k - 1 | k } \mathrm { ( A p p . ~ A ) }$ .

Discrete transition kernels In practice we sample on the grid $0 = t _ { 0 } < \cdots < t _ { K } = 1$ of Section 2, with $t = 0$ the data distribution, writing $x _ { k }$ for the state at $t _ { k }$ so that step k moves $x _ { k } \to x _ { k - 1 }$ . The path measures factorise into the backward transition kernels of Section 2 and the kernel we actually simulate (Wu et al., 2023),

$$
\begin{array} { r l } & { \mathbb { P } _ { 0 : K } \big ( x _ { 0 : K } \big ) = p _ { \mathrm { r e f } } \big ( { x _ { K } } \big ) \displaystyle \prod _ { k = 1 } ^ { K } { \bar { P } } _ { k - 1 \mid k } \big ( { x _ { k - 1 } } \mid { x _ { k } } \big ) , } \\ & { \mathbb { Q } _ { 0 : K } \big ( { x _ { 0 : K } } \big ) = p _ { \mathrm { r e f } } \big ( { x _ { K } } \big ) \displaystyle \prod _ { k = 1 } ^ { K } { \bar { Q } } _ { k - 1 \mid k } \big ( { x _ { k - 1 } } \mid { x _ { k } } \big ) , } \end{array}\tag{68}
$$

where $\stackrel {  } { P } _ { k - 1 | k }$ integrates the base reverse SDE of Eq. (1) and $\stackrel {  } { Q } _ { k - 1 | k }$ the guided SDE of Eq. (9) from $t _ { k }$ down to $t _ { k - 1 }$ . Both are initialised from the same Gaussian prior $p _ { \mathrm { r e f } }$ which cancels,

$$
\frac { \mathrm { d } \mathbb { P } _ { 0 : K } } { \mathrm { d } \mathbb { Q } _ { 0 : K } } = \prod _ { k = 1 } ^ { K } \frac { \overline { { P } } _ { k - 1 | k } \big ( x _ { k - 1 } \mid x _ { k } \big ) } { \overline { { Q } } _ { k - 1 | k } \big ( x _ { k - 1 } \mid x _ { k } \big ) } .\tag{69}
$$

Incorporating a reward tilt $e ^ { r _ { 0 } ( x _ { 0 } ) }$ as a telescope product of individual reward potentials,

$$
R ( x _ { 0 : K } ) = \prod _ { k = 1 } ^ { K } G _ { k } , \qquad G _ { k } = \frac { e ^ { r _ { k - 1 } ( x _ { k - 1 } ) } } { e ^ { r _ { k } ( x _ { k } ) } } ,\tag{70}
$$

which collapses to $R = \exp \bigl ( r _ { 0 } ( x _ { 0 } ) - r _ { K } ( x _ { K } ) \bigr ) = e ^ { r _ { 0 } ( x _ { 0 } ) }$ since $r _ { K } = 0$ . Multiply Eqs. (69) and (70) yields,

$$
e ^ { r _ { 0 } ( x _ { 0 } ) } \frac { \mathrm { d } \mathbb { P } _ { 0 : K } } { \mathrm { d } \mathbb { Q } _ { 0 : K } } \big ( x _ { K : 0 } \big ) = \prod _ { k = 1 } ^ { K } G _ { k } \frac { \overleftarrow { P } _ { k - 1 | k } \big ( x _ { k - 1 } ~ \big | ~ x _ { k } \big ) } { \overleftarrow { Q } _ { k - 1 | k } \big ( x _ { k - 1 } ~ \big | ~ x _ { k } \big ) } = \prod _ { k = 1 } ^ { K } g _ { k - 1 | k } \big ( x _ { k - 1 } , x _ { k } \big ) ,\tag{71}
$$

which recovers the incremental weight of Eq. (10) accumulated by Algorithm 1.

Weight Computation Recall from Lemma 1 that, for Gaussian transition kernels of Eq. (11), the incremental log weight is given by,

$$
\begin{array} { r } { \log g _ { k - 1 | k } ( x _ { k - 1 } , x _ { k } ) = \underbrace { r _ { k - 1 } ( x _ { k - 1 } ) - r _ { k } ( x _ { k } ) } _ { \mathrm { r e w a r d } \mathrm { v e s o p e } } - \underbrace { \frac { \zeta _ { k } ^ { 2 } } { 2 } \| \nabla _ { x } r _ { k } ( x _ { k } ) \| ^ { 2 } } _ { \mathrm { c o n t r o l c o s t } } - \underbrace { \zeta _ { k } \langle \nabla _ { x } r _ { k } ( x _ { k } ) , \epsilon _ { k } \rangle } _ { \mathrm { m a r i n g a l e c o r r e c t i o n } } . } \end{array}\tag{72}
$$

As all path-measures we have introduced differ only in the drift, the change of measure between them is given by Girsanov’s theorem which yields an exact Radon-Nikodym Derivative (Øksendal, 2003) and the above computation represents its discretised form. The control-cost is the price paid by the guided SDE for perturbing the base drift towards the rare-event set and a martingale correction which accounts for any “lucky” noise $\epsilon _ { k }$ sampled along the guided step that carries the particle towards the set.

Remark. Nothing in this derivation constrains the $r _ { k }$ to be constant across time; and any family of schedules defined by $( \lambda ( t ) , \tau ( t ) , \delta ( t ) )$ do not break the validity and the weight computation remains exact. The schedules enter the reward telescope of Eq. (70) only through the endpoints themselves $R = \exp \bigl ( r _ { t _ { 0 } } ( x _ { 0 } ) - r _ { t _ { K } } ( x _ { K } ) \bigr )$ , and the kernel ratio of Eq. (69) only through the guidance $\nabla _ { \boldsymbol { x } _ { t } } \boldsymbol { r } _ { t }$

## D.1 $\zeta _ { k }$ FOR COMMON INTEGRATORS

The first-order exponential integrator (Milstein & Tretyakov, 2004) of (Gonzalez et al., 2023) integrates the linear part of the reverse drift and the noise exactly, freezing only the network output across the step. Reparameterising the time increment in half-log Signal to Noise Ratio, $\ell _ { t } : =$ $\log ( \alpha _ { t } / \bar { \sigma } _ { t } )$ ; write $\Delta \bar { \ell } _ { k } = \ell _ { t _ { k - 1 } } - \bar { \ell } _ { t _ { k } } \geq 0$ for its increment across a step.

Exponential Integrator (VP & $\hat { \epsilon } _ { \theta }$ frozen) Restating their Eq. 14 (Gonzalez et al., 2023) in our notation and abbreviating the symbols,

$$
A _ { k } : = \frac { \alpha _ { t _ { k - 1 } } } { \alpha _ { t _ { k } } } , \qquad D _ { k } : = 2 \bar { \sigma } _ { t _ { k - 1 } } \big ( e ^ { \Delta \ell _ { k } } - 1 \big ) , \qquad v _ { k } : = \bar { \sigma } _ { t _ { k - 1 } } ^ { 2 } \big ( e ^ { 2 \Delta \ell _ { k } } - 1 \big ) .\tag{73}
$$

The incremental reverse step is given by,

$$
x _ { k - 1 } = A _ { k } x _ { k } \ - \ D _ { k } \hat { \epsilon } _ { \theta } \big ( x _ { k } , t _ { k } \big ) \ + \ \sqrt { v _ { k } } \epsilon _ { k } , \qquad \epsilon _ { k } \sim \mathcal { N } ( 0 , I ) ,\tag{74}
$$

where $\hat { \epsilon } _ { \theta }$ is the noise predicted by the network. This is related by a change of variable to the score network, $\begin{array} { r } { s _ { \theta } = - \frac { \epsilon _ { \theta } } { \bar { \sigma } } } \end{array}$ (Gao et al., 2024). Hence, the guidance update in score maps to the noise prediction as,

$$
s _ { \theta }  s _ { \theta } + \nabla _ { x } r _ { t } ,\tag{75}
$$

Therefore, the change of means between the two guided and unguided process is obtained by substituting Eq.(75) into Eq.(74) to get,

$$
\begin{array} { r } { \mu _ { k } ^ { a } - \mu _ { k } ^ { \nu } = D _ { k } \bar { \sigma } _ { t _ { k } } \nabla _ { x } r _ { t _ { k } } ( x _ { k } ) = c _ { k } ^ { \mathrm { V P } } \nabla _ { x } r _ { t _ { k } } ( x _ { k } ) , } \end{array}\tag{76}
$$

where we have further equated this to the mean shift of Eq. (11). Hence, under the exponential first-order integrator and a VP noise schedule,

$$
c _ { k } ^ { \mathrm { V P } } = D _ { k } \bar { \sigma } _ { t _ { k } } = 2 \bar { \sigma } _ { t _ { k - 1 } } \bar { \sigma } _ { t _ { k } } ( e ^ { \Delta \ell _ { k } } - 1 ) .\tag{77}
$$

To obtain $\zeta _ { k }$ , we make further substitutions,

$$
\begin{array} { r l } { \left( \zeta _ { k } ^ { \operatorname { V P } } \right) ^ { 2 } = \frac { \left( c _ { k } ^ { \operatorname { V P } } \right) ^ { 2 } } { v _ { k } } = \frac { 4 \bar { \sigma } _ { t k _ { - 1 } } ^ { 2 } \bar { \sigma } _ { t k } ^ { 2 } \left( e ^ { \Delta \ell _ { k } } - 1 \right) ^ { 2 } } { \bar { \sigma } _ { t k _ { - 1 } } ^ { 2 } \left( e ^ { 2 \Delta \ell _ { k } } - 1 \right) } } \\ & { = \frac { 4 \bar { \sigma } _ { t k } ^ { 2 } \left( e ^ { \Delta \ell _ { k } } - 1 \right) ^ { 2 } } { \left( e ^ { \Delta \ell _ { k } } - 1 \right) \left( e ^ { \Delta \ell _ { k } } + 1 \right) } } \\ & { = \frac { 4 \bar { \sigma } _ { t k } ^ { 2 } \left( e ^ { \Delta \ell _ { k } } - 1 \right) } { e ^ { \Delta \ell _ { k } } + 1 } } \\ & { = 4 \bar { \sigma } _ { t k } ^ { 2 } \operatorname { t a n h } ( \Delta \ell _ { k } / 2 ) , } \end{array}\tag{78}
$$

so that $\zeta _ { k } ^ { \mathrm { V P } } = 2 \bar { \sigma } _ { t _ { k } } \sqrt { \operatorname { t a n h } ( \Delta \ell _ { k } / 2 ) }$

Exponential Integrator (VE) Under the VE schedule, $f _ { t } = 0$ and $\alpha _ { t } \equiv 1$ . Following (Gonzalez et al., 2023) and rewriting the score network as a data-prediction $\hat { x } _ { 0 \mid t } = x _ { t } + \bar { \sigma } _ { t } ^ { 2 } s _ { \theta }$ using Tweedie’s formula (Efron, 2011; Gao et al., 2024) and freezing ${ \hat { x } } _ { 0 \mid t }$ along the increment, the reverse drift is given by,

$$
x _ { k - 1 } = \kappa _ { k } x _ { k } + ( 1 - \kappa _ { k } ) \hat { x } _ { 0 \mid t } + \sqrt { v _ { k } } \epsilon _ { k } , \qquad v _ { k } = \bar { \sigma } _ { t _ { k - 1 } } ^ { 2 } ( 1 - \kappa _ { k } ) ,\tag{79}
$$

where $\begin{array} { r } { \kappa _ { k } : = \frac { \bar { \sigma } _ { t _ { k - 1 } } ^ { 2 } } { \bar { \sigma } _ { t _ { k } } ^ { 2 } } } \end{array}$ . Under a guided score the data prediction is perturbed to,

$$
\begin{array} { r l } & { \quad s _ { \theta } \to s _ { \theta } + \nabla _ { x } r _ { t } , } \\ & { \quad \hat { x } _ { 0 | t } \to \hat { x } _ { 0 | t } + \bar { \sigma } _ { t _ { k } } ^ { 2 } \nabla _ { x } r _ { t _ { k } } ( x _ { k } ) , } \end{array}\tag{80}
$$

hence, $c _ { k } = ( 1 - \kappa _ { k } ) \bar { \sigma } _ { t _ { k } } ^ { 2 }$ and it follows,

$$
\zeta _ { k } ^ { \mathrm { V E } , \hat { x } } = \bar { \sigma } _ { t _ { k } } \sqrt { e ^ { 2 \Delta \ell _ { k } } - 1 } ,\tag{81}
$$

with $\ell _ { t } = - \log \bar { \sigma } _ { t }$ when $\alpha _ { t } \equiv 1$

Freezing the noise-prediction $\hat { \epsilon } _ { k }$ would lead to an exact quadrature scheme of the noise term, v . Under a VE schedule $f _ { t } = 0 ,$ so the reverse drift carries no part linear in x and, using $\sigma _ { t } ^ { 2 } / \bar { \sigma } _ { t } = 2 \dot { \bar { \sigma } } _ { t }$ the frozen integrand may be integrated in closed form (Gonzalez et al., 2023),

$$
\begin{array} { r l r l r l } & { x _ { k - 1 } = x _ { k } - D _ { k } ^ { \mathrm { V E } } \hat { \epsilon } _ { \theta } ( x _ { k } , t _ { k } ) + \sqrt { v _ { k } } \epsilon _ { k } , } & & { D _ { k } ^ { \mathrm { V E } } : = 2 \big ( \bar { \sigma } _ { t _ { k } } - \bar { \sigma } _ { t _ { k - 1 } } \big ) , } & & { v _ { k } : = \bar { \sigma } _ { t _ { k } } ^ { 2 } - \bar { \sigma } _ { t _ { k - 1 } } ^ { 2 } , } \end{array}\tag{82}
$$

Substituting the guided noise-prediction of Eq. (75) gives $c _ { k } ^ { \mathrm { v E } } = D _ { k } ^ { \mathrm { v E } } \bar { \sigma } _ { t _ { k } } = 2 \bar { \sigma } _ { t _ { k } } ( \bar { \sigma } _ { t _ { k } } - \bar { \sigma } _ { t _ { k - 1 } } )$ , and since $\bar { \sigma } _ { t _ { k - 1 } } = \bar { \sigma } _ { t _ { k } } e ^ { - \Delta \ell _ { k } }$ when $\alpha _ { t } \equiv 1$ , both coefficients collapse onto the VP forms,

$$
c _ { k } ^ { \mathrm { V E } } = 2 \bar { \sigma } _ { t _ { k - 1 } } \bar { \sigma } _ { t _ { k } } \big ( e ^ { \Delta \ell _ { k } } - 1 \big ) , \qquad v _ { k } = \bar { \sigma } _ { t _ { k - 1 } } ^ { 2 } \big ( e ^ { 2 \Delta \ell _ { k } } - 1 \big ) ,\tag{83}
$$

so that $\zeta _ { k } ^ { \mathrm { V E } , \hat { \epsilon } } = 2 \bar { \sigma } _ { t _ { k } } \sqrt { \operatorname { t a n h } ( \Delta \ell _ { k } / 2 ) }$ , identical to $\zeta _ { k } ^ { \mathrm { V P } }$ of Eq. (78).

Euler–Maruyama Euler–Maruyama steps (Kloeden & Platen, 1992) discretise the reverse SDE directly, so the base and guided kernels share a common covariance, $v _ { k } ~ = ~ \sigma _ { t _ { k } } ^ { 2 } \Delta t _ { k }$ and $u _ { t } ~ =$ $- \sigma _ { t } ^ { 2 } \nabla _ { x } r _ { t } .$ . Hence, $c _ { k } ^ { E M } = \sigma _ { t _ { k } } ^ { 2 } \Delta t _ { k }$ , and,

$$
\zeta _ { k } ^ { E M } = \sigma _ { t _ { k } } \sqrt { \Delta t _ { k } } .\tag{84}
$$

Table 3 collects the three cases. Notably, as we increase the number of discretisations or $\Delta \ell _ { k } \to 0$ all variants given in Table 3 recover the EM $\zeta _ { k } ^ { 2 }$

Table 3: $\zeta _ { k } ^ { 2 } = c _ { k } ^ { 2 } / v _ { k }$ for different integrators and noise schedules.
<table><tr><td></td><td> $c _ { k }$ </td><td> $v _ { k }$ </td><td> $\zeta _ { k } ^ { 2 } = c _ { k } ^ { 2 } / v _ { k }$ </td></tr><tr><td>Euler–Maruyama (VP/VE)</td><td> $\sigma _ { t _ { k } } ^ { 2 } \Delta t _ { k }$ </td><td> $\sigma _ { t _ { k } } ^ { 2 } \Delta t _ { k }$ </td><td> $\sigma _ { t _ { k } } ^ { 2 } \Delta t _ { k }$ </td></tr><tr><td>exponential, VP ( frozen)</td><td> $2 \bar { \sigma } _ { t _ { k - 1 } } \bar { \sigma } _ { t _ { k } } \big ( e ^ { \Delta \ell _ { k } } - 1 \big )$ </td><td> $\bar { \sigma } _ { t _ { k - 1 } } ^ { 2 } ( \bar { e } ^ { 2 \Delta \ell _ { k } } - 1 )$ </td><td> $4 \bar { \sigma } _ { t _ { k } } ^ { 2 } \operatorname { t a n h } ( { \Delta \ell _ { k } } / { 2 } )$ </td></tr><tr><td>exponential, VE (x frozen)</td><td> $\left( 1 - \kappa _ { k } \right) \bar { \sigma } _ { t _ { k } } ^ { 2 }$ </td><td> $\bar { \sigma } _ { t _ { k - 1 } } ^ { 2 } ( 1 - \kappa _ { k } )$ </td><td> $\bar { \sigma } _ { t _ { k } } ^ { 2 } ( e ^ { 2 \Delta \ell _ { k } } - 1 )$ </td></tr><tr><td>exponential, VE (ê frozen)</td><td> $2 \bar { \sigma } _ { t _ { k - 1 } } \bar { \sigma } _ { t _ { k } } \left( e ^ { \Delta \ell _ { k } } - 1 \right)$ </td><td> $\bar { \sigma } _ { t _ { k - 1 } } ^ { 2 } \left( e ^ { 2 \Delta \ell _ { k } } - 1 \right)$ </td><td> $4 \bar { \sigma } _ { t _ { k } } ^ { 2 ^ { - } } \operatorname { t a n h } ( { \Delta \ell _ { k } } / { 2 } )$ </td></tr></table>

## E THE GUIDANCE GRADIENT VIA LOCAL LINEARISATION

![](images/ce0e6863b36b09836e6c3c1ba8fe7910208750ac290ffcf779dd0d325fbc52de.jpg)

![](images/c82f590a02184a4446de8550b9465be57f9a170b974e564c5f94eacde45f2bcb.jpg)  
Figure 5: Soft reward and its guidance for a threshold event $E _ { 7 } = \{ x : \phi ( x ) \geq 7 \}$ shown for three smoothing levels δ. Left. The soft indicator relaxes the hard indicator. Right. The guidance $\nabla _ { x } r _ { 0 , \delta } ( x )$ points toward the event set (shaded) and decays inside it, a smaller δ gives a stronger guidance.

The variance-aware guidance, illustrated in Fig. 5 used in Section 3 is,

$$
\nabla _ { x _ { t } } \log \mathbb { E } _ { p ( x _ { 0 } | x _ { t } ) } [ e ^ { r _ { 0 , \delta } ( x _ { 0 } ) } ]
$$

This appendix derives the closed-form approximation used in our implementation. We consider a general vector-valued constraint $\phi : \mathbb { R } ^ { d } \overset { \cdot } {  } \mathbb { R } ^ { m }$ using a Gaussian approximation of the denoising posterior and a local linearisation of ϕ. The scalar affine guidance in Eq. (91), used for the one-sided threshold-exceedance experiments, is recovered as a special case $m = 1$ with a linear-ϕ. We then discuss how interval and compound events are represented by multiple one-sided constraints and the approximation induced when their guidance terms are treated independently.

For a multivariate constraint, the smooth reward is a multivariate Gaussian CDF $\Phi _ { m } ( \cdot \mid \Sigma _ { c } )$ with constraint covariance $\Sigma _ { c }$ (the smoothing bandwidth; in the scalar case $\Sigma _ { c } = \delta ^ { 2 } )$ . The integral is given by,

$$
I ( x _ { t } ) = \mathbb { E } _ { p ( x _ { 0 } | x _ { t } ) } \big [ \Phi _ { m } \big ( \phi ( x _ { 0 } ) - \tau \big | \Sigma _ { c } \big ) \big ] .\tag{85}
$$

Gaussian posterior. We use the same Gaussian denoising-posterior approximation as in the main text, $p ( x _ { 0 } \mid x _ { t } ) \approx \mathcal { N } ( x _ { 0 } \mid \hat { x } _ { 0 \mid t } , \Sigma _ { t } )$ with $\hat { x } _ { 0 \mid t }$ the Tweedie estimate and $\Sigma _ { t }$ the diagonal ΠGDM posterior-covariance proxy (Appendix B), hence,

$$
I ( \boldsymbol { x } _ { t } ) \approx \int _ { \mathbb { R } ^ { d } } \boldsymbol { \mathcal { N } } ( \boldsymbol { x } _ { 0 } \mid \hat { \boldsymbol { x } } _ { 0 \mid t } , \boldsymbol { \Sigma } _ { t } ) \Phi _ { m } \big ( \phi ( \boldsymbol { x } _ { 0 } ) - \tau \big | \boldsymbol { \Sigma } _ { c } \big ) \mathrm { d } \boldsymbol { x } _ { 0 } .\tag{86}
$$

Linearisation. Linearising the constraint about the Tweedie mean,

$$
\phi ( \boldsymbol { x } _ { 0 } ) \approx \phi ( \hat { \boldsymbol { x } } _ { 0 \mid t } ) + J _ { \phi } ( \hat { \boldsymbol { x } } _ { 0 \mid t } ) ( \boldsymbol { x } _ { 0 } - \hat { \boldsymbol { x } } _ { 0 \mid t } ) , \qquad J _ { \phi } ( \hat { \boldsymbol { x } } _ { 0 \mid t } ) = \nabla _ { \boldsymbol { x } _ { 0 } } \phi \big | _ { \hat { \boldsymbol { x } } _ { 0 \mid t } } \in \mathbb { R } ^ { m \times d } ,\tag{87}
$$

makes the integrand a Gaussian convolved with a Gaussian CDF, which integrates in closed form to a single CDF with the two covariances added (Rasmussen & Williams, 2006),

$$
I ( x _ { t } ) \approx \Phi _ { m } \Big ( \phi ( \hat { x } _ { 0 \mid t } ) - \tau \Big | \Sigma _ { c } + J _ { \phi } \big ( \hat { x } _ { 0 \mid t } \big ) \Sigma _ { t } J _ { \phi } \big ( \hat { x } _ { 0 \mid t } \big ) ^ { \top } \Big ) .\tag{88}
$$

The posterior covariance $\Sigma _ { t }$ is thus pushed through the constraint Jacobian and inflates the effective observation variance.

Guidance gradient. Let $\pi _ { t } \propto p _ { t } e ^ { r _ { t } }$ be the continuous-time version of the tilted target $\pi _ { k }$ of Eq. (5). The guided score is then the unconditional score plus the gradient of log I,

$$
\begin{array} { r } { \nabla _ { x _ { t } } \log \pi _ { t } ( x _ { t } ) = \underbrace { \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } ) } _ { \mathrm { u n c o n d i t i o n a l ~ s c o r e } } + \underbrace { \nabla _ { x _ { t } } \log \Phi _ { m } \Big ( \phi ( \hat { x } _ { 0 \mid t } ) - \tau \Big | \sum _ { c } + J _ { \phi } \big ( \hat { x } _ { 0 \mid t } \big ) \sum _ { t } J _ { \phi } \big ( \hat { x } _ { 0 \mid t } \big ) ^ { \top } \Big ) } _ { \mathrm { v a r i a n c e , s u s r a r s ~ m i d i a n c e ~ a r a d i t i o n } } . } \end{array}
$$

variance-aware guidance gradient

(89)

Linear constraint. For an affine read-out $\phi ( x _ { 0 } ) = A x _ { 0 } + b$ the Jacobian is constant, $J _ { \phi } = A$ , and the guidance simplifies to

$$
\begin{array} { r } { \nabla _ { x _ { t } } \log \pi _ { t } ( x _ { t } ) = \nabla _ { x _ { t } } \log p _ { t } ( x _ { t } ) + \nabla _ { x _ { t } } \log \Phi _ { m } \Big ( A \hat { x } _ { 0 \mid t } + b - \tau \Big | \Sigma _ { c } + A \Sigma _ { t } A ^ { \top } \Big ) . } \end{array}\tag{90}
$$

Specialising further to $m = 1$ with $\phi ( x ) = a ^ { \top }$ x and $\Sigma _ { c } = \delta ^ { 2 }$ gives the scalar guidance used in the main text,

$$
\begin{array} { r l r } {  { \nabla _ { \boldsymbol { x } _ { t } } \log \mathbb { E } _ { p ( x _ { 0 } \mid \boldsymbol { x } _ { t } ) } [ \exp ( r _ { 0 , \delta } ( \boldsymbol { x } _ { 0 } ) ) ] = \nabla _ { \boldsymbol { x } _ { t } } \log \int p ( x _ { 0 } \mid \boldsymbol { x } _ { t } ) \Phi \bigg ( \frac { \phi ( \boldsymbol { x } _ { 0 } ) - \tau } { \delta } \bigg ) d \boldsymbol { x } _ { 0 } } } \\ & { } & { \approx \nabla _ { \boldsymbol { x } _ { t } } \log \Phi \bigg ( \frac { \phi ( \hat { \boldsymbol { x } } _ { 0 \mid t } ) - \tau } { \sqrt { \delta ^ { 2 } + a ^ { \top } \Sigma _ { t } a } } \bigg ) , } \end{array}\tag{91}
$$

with the inflated width $\sqrt { \delta ^ { 2 } + a ^ { \top } \Sigma _ { t } a }$

## E.1 FACTORISED GUIDANCE FOR MULTIPLE CONSTRAINTS

Consider an event defined by the intersection of m one-sided constraints,

$$
E = \bigcap _ { j = 1 } ^ { m } \left\{ \phi _ { j } ( x ) \geq \tau _ { j } \right\} ,\tag{92}
$$

such as the boxed intervals of App. G.4. Section E gives the corresponding look-ahead likelihood as a multivariate Gaussian CDF $\Phi _ { m } ( \cdot \mid \Sigma _ { \mathrm { e f f } } )$ with

$$
\Sigma _ { \mathrm { e f f } } = \Sigma _ { c } + J _ { \phi } ( \hat { x } _ { 0 | t } ) \Sigma _ { t } J _ { \phi } ( \hat { x } _ { 0 | t } ) ^ { \top } ,\tag{93}
$$

where $\Sigma _ { c }$ is the user-specified smoothing covariance and the second term is the denoising-posterior uncertainty projected through the read-out (Eq. (88)).

To obtain the guidance gradient, we require $\nabla _ { x _ { t } }$ log $\Phi _ { m }$ , which in turn requires evaluating $\Phi _ { m }$ itself. For $m = 1$ , this is available in closed form and for $m = 2 , 3$ it reduces to a low-dimensional quadrature (Drezner, 1994; Genz, 2004; Owen, 1956). For $m \geq 4$ with an unstructured or full covari ance matrix $\Sigma _ { \mathrm { e f f } }$ no such reduction exists and the function has to be approximated by (randomised) quasi-Monte Carlo techniques (Genz, 1992). The cost of that approximation is incurred once per particle per diffusion step, and unless the evaluation points are held fixed across particles and steps, the reward and its gradient become stochastic rather than deterministic functions of $x _ { t }$ . We therefore also provide the factorised surrogate below.

Optionally, $\Sigma _ { \mathrm { e f f } }$ may be approximated by its diagonal, under which $\Phi _ { m }$ factorises into a product of univariate Gaussian CDFs,

$$
I _ { \mathrm { f a c t } } ( x _ { t } ) = \prod _ { j = 1 } ^ { m } \Phi \left( \frac { \phi _ { j } ( \hat { x } _ { 0 | t } ) - \tau _ { j } } { \delta _ { \mathrm { e f f } , j } ( x _ { t } ) } \right) , \qquad \delta _ { \mathrm { e f f } , j } ^ { 2 } ( x _ { t } ) = \delta _ { j } ^ { 2 } + a _ { j } ^ { \top } \Sigma _ { t } a _ { j } , \quad a _ { j } = \nabla _ { x _ { 0 } } \phi _ { j } \big | _ { \hat { x } _ { 0 | t } } ,\tag{94}
$$

with a factorised reward $r _ { t } ^ { \mathrm { f a c t } } : = \log I _ { \mathrm { f a c t } }$ and guidance,

$$
\nabla _ { x _ { t } } r _ { t } ^ { \mathrm { f a c t } } = \sum _ { j = 1 } ^ { m } \nabla _ { x _ { t } } \log \Phi \left( \frac { \phi _ { j } ( \hat { x } _ { 0 | t } ) - \tau _ { j } } { \delta _ { \mathrm { e f f } , j } ( x _ { t } ) } \right) .\tag{95}
$$

This is simply m independent copies of the scalar guidance and while the factorisation retains the marginal posterior uncertainty of each constraint, it discards the off-diagonal entries of $\Sigma _ { \mathrm { e f f } }$ . This alters the guidance drift, and hence the implemented proposal $\mathbb { Q }$ together with the sequence of marginals $\pi _ { t }$ . Our rejuvenation kernels are made invariant with respect to the same marginals, using the factorised surrogate $I _ { \mathrm { f a c t } }$ . However, the target itself remains unchanged $p _ { 0 } e ^ { r _ { 0 , \delta } ( x ) }$ as it depends on the data-reward at $t = 0$ alone and remains unaffected by $\pi _ { t }$ or $\nabla _ { x } { r _ { t } }$ . Hence, this approximation only contributes to the variance of our estimator.

Note finally that the factorisation describes the tilted marginals and not the sampler. The guided drift is,

$$
\nabla _ { \boldsymbol { x } _ { t } } \log { p _ { t } ( \boldsymbol { x } _ { t } ) } + \nabla _ { \boldsymbol { x } _ { t } } r _ { t } ^ { \mathrm { f a c t } } ( \boldsymbol { x } _ { t } ) ,
$$

and the unconditional score retains the emulator’s full joint dependence structure. Additionally, each term is also mapped through the dense $\partial \hat { x } _ { 0 \mid t } / \partial x _ { t }$ , so a constraint supported on a single region and channel still induces a global drift across all variables. The diagonality of $\Sigma _ { \mathrm { e f f } }$ suppresses cross-constraint terms in the look-ahead variance; it does not decouple the resulting drift.

Interval Guidance A useful special case is an event defined by the interval, $E = \{ \tau _ { a } \leq \phi ( x ) \leq \tau _ { b } \}$ which is the intersection (92) with $m = 2$ , taking $\phi _ { 1 } = \phi , \tau _ { 1 } = \tau _ { a }$ and $\phi _ { 2 } = - \phi , \tau _ { 2 } = - \tau _ { b }$ . We write the reward as a difference of two individual rewards given by,

$$
e ^ { r _ { 0 , \delta } ( x ) } = \Phi \bigg ( \frac { \phi ( x ) - \tau _ { a } } { \delta } \bigg ) - \Phi \bigg ( \frac { \phi ( x ) - \tau _ { b } } { \delta } \bigg ) ,\tag{96}
$$

with the look-ahead and guidance following as before, replacing δ by $\delta _ { \mathrm { e f f } } ^ { 2 } = \delta ^ { 2 } + a ^ { \top } \Sigma _ { t } a$ which is shared across both boundaries.

## F THE CORRELATED 2D GMM BENCHMARK

This appendix specifies the synthetic benchmark used in our 2D experiments and collects the closedform quantities it admits: the base density, the normalisation constant $Z _ { 0 , \delta }$ of the soft tilt, the exact exceedance probability $p _ { 0 } [ E ] .$ , and the exact Doob guidance $\nabla _ { x } r _ { t } ^ { \star } = \nabla _ { x }$ log $G _ { t }$ (Dai Pra, 1991; Denker et al., 2024). All of these are available in closed form precisely because the GMM prior is conjugate to both the Gaussian forward kernel and the Gaussian-CDF reward, which lets us isolate the behaviour of the estimator from score- and reward-approximation error (Rasmussen & Williams, 2006).

Base distribution. The data distribution is a $M = 7$ -component Gaussian mixture in $\mathbb { R } ^ { 2 }$

$$
p _ { 0 } ( x ) \ = \ \sum _ { i = 1 } ^ { M } w _ { i } { \mathcal N } ( x ; \mu _ { i } , \Sigma _ { i } ) , \qquad \sum _ { i } w _ { i } = 1 ,\tag{97}
$$

where $w _ { i } ( i = 1 , \ldots , M )$ denote the mixture weights. The parameters are given in Table 4. The four “bulk” modes carry most of the mass near the origin; the three right-hand modes are pushed out towards large $x _ { 1 }$ and given strong off-diagonal covariance $( \rho \approx \pm 0 . 8 5 )$ of opposite sign. This is by design: it makes the tail conditional $p _ { 0 } ( x _ { 2 } \mid x _ { 1 } > \tau )$ bi-modal and markedly non-Gaussian, so that hitting the rare set is not equivalent to matching a single Gaussian tail and the proposal must reproduce a genuine multi-modal conditional.

The mixture weights w<sub>i</sub> are given in Table 4 and sum to one. The observable is the linear read-out $\phi ( x ) = a ^ { \top } ;$ x with $a = ( 1 , 0 ) ^ { \top }$ , i.e. the event of interest is the threshold exceedance $\{ x : x _ { 1 } \geq \tau \}$

![](images/d37355adca69d791a74b0427410f2e798266decfaea5bf3936222e93d6b5ddea.jpg)  
Figure 6: Base distribution $p _ { 0 } ( x )$ of the correlated 2D GMM (log-density, so all modes are visible) with the analytic $x _ { 1 }$ and $x _ { 2 }$ marginals. The two right-hand modes $( i = 5 , 6 )$ form the tilted $\mathbf { \ddot { \delta t } X } ^ { \prime \prime }$ of opposite-sign correlation that gives the rare set a bi-modal conditional.

Table 4: Parameters of the correlated 2D GMM. The right-hand modes $( i = 5 , 6 , 7 )$ carry the rareevent tail; modes 5 and 6 have correlation $\rho$ ≈ +0.85 and −0.85 respectively.
<table><tr><td>i</td><td> $w _ { i }$ </td><td> $\mu _ { i }$ </td><td> $\Sigma _ { i }$ </td></tr><tr><td>1</td><td>0.10</td><td>(−6, −2)</td><td>[0.36 0.30 ] 0.30 0.81</td></tr><tr><td>2</td><td>0.18</td><td>(−4, 4)</td><td>0.81 -0.251 -0.25 0.49</td></tr><tr><td>3</td><td>0.14</td><td>(−1, −5)</td><td>[0.49 0.20] 0.20 0.81 </td></tr><tr><td>4</td><td>0.22</td><td>(−10, 3)</td><td>0.040.04 0.04 0.09 2.401.85</td></tr><tr><td>5</td><td>0.10</td><td>(4, 3)</td><td>[1.85 2.00 2.40 -1.851</td></tr><tr><td>6 7</td><td>0.16 0.10</td><td>(4,−3) (5, 0)</td><td>-1.85 2.00 0.30 0.00</td></tr></table>

Normalisation constant $Z _ { 0 , \delta } .$ . The soft tilt $\pi _ { 0 , \delta } ( x ) \propto p _ { 0 } ( x ) e ^ { r _ { 0 , \delta } ( x ) }$ with the Gaussian-CDF reward $e ^ { r _ { 0 , \delta } ( x ) } = \Phi \big ( ( a ^ { \top } x - \tau ) / \delta \big )$ has normaliser $Z _ { 0 , \delta } = \mathbb { E } _ { p _ { 0 } } [ e ^ { r _ { 0 , \delta } ( X ) } ]$ . Under each Gaussian component the read-out $y = a ^ { \top } x \sim { \mathcal { N } } ( m _ { i } , v _ { i } )$ with $m _ { i } = { a } ^ { \top } \mu _ { i }$ and $\boldsymbol { v } _ { i } = \boldsymbol { a } ^ { \intercal } \Sigma _ { i } \boldsymbol { a } .$ , and a standard Gaussian– Gaussian-CDF convolution (see Section 3.9 of (Rasmussen & Williams, 2006)) gives the closed form,

$$
Z _ { 0 , \delta } \ = \ \sum _ { i = 1 } ^ { M } w _ { i } \ \mathbb { E } _ { \mathcal { N } ( m _ { i } , v _ { i } ) } \Bigl [ \Phi \Bigl ( \frac { y - \tau } { \delta } \Bigr ) \Bigr ] \ = \ \sum _ { i = 1 } ^ { M } w _ { i } \Phi \left( \frac { m _ { i } - \tau } { \sqrt { \delta ^ { 2 } + v _ { i } } } \right) .\tag{98}
$$

As $\delta \downarrow 0$ the reward sharpens to the hard indicator and $Z _ { 0 , \delta } \to p _ { 0 } [ E ]$

Exact exceedance probability. Because the read-out $a ^ { \top } X$ is itself a 1D GMM with component means $m _ { i }$ and variances $v _ { i }$ , the exceedance probability is available in closed form,

$$
p _ { 0 } [ E ] = p _ { 0 } [ a ^ { \top } X \geq \tau ] = \sum _ { i = 1 } ^ { M } w _ { i } \Big [ 1 - \Phi \big ( \frac { \tau - m _ { i } } { \sqrt { v _ { i } } } \big ) \Big ] = \sum _ { i = 1 } ^ { M } w _ { i } \Phi \Big ( \frac { m _ { i } - \tau } { \sqrt { v _ { i } } } \Big ) ,\tag{99}
$$

which we use as ground truth.

Tilted read-out marginal. While the full tilted density $\pi _ { 0 , \delta } ( x ) \propto p _ { 0 } ( x ) \Phi \big ( ( a ^ { \top } x - \tau ) / \delta \big )$ is not itself a Gaussian mixture, the marginal of the read-out $y = a ^ { \top }$ x is available in closed form,

$$
\pi _ { 0 , \delta } ^ { \mathrm { r e a d } } ( y ) = \frac { 1 } { Z _ { 0 , \delta } } \sum _ { i = 1 } ^ { M } w _ { i } \mathcal { N } ( y ; m _ { i } , v _ { i } ) \Phi \Big ( \frac { y - \tau } { \delta } \Big ) , \qquad y = a ^ { \top } x ,\tag{100}
$$

with $Z _ { 0 , \delta }$ from Eq. (98) and $( m _ { i } , v _ { i } )$ as above.

Forward process. We use the VP-SDE schedule of Appendix B with $\beta _ { \mathrm { m i n } } = 0 . 1 , \beta _ { \mathrm { m a x } } = 2 0$ , so that $\begin{array} { r } { B ( t ) = \int _ { 0 } ^ { t } \beta ( s ) d s = t \beta _ { \mathrm { m i n } } + \frac { 1 } { 2 } t ^ { 2 } \big ( \beta _ { \mathrm { m a x } } - \beta _ { \mathrm { m i n } } \big ) , \alpha _ { t } = e ^ { - B ( t ) / 2 } \mathrm { a n d } \bar { \sigma } _ { t } ^ { 2 } = 1 - e ^ { - B ( t ) / 2 } , } \end{array}$ . Because the forward kernel is Gaussian, the noised marginal $p _ { t }$ is again a GMM whose i-th component has mean $\alpha _ { t } \mu _ { i }$ and covariance (Rasmussen & Williams, 2006),

$$
S _ { t , i } = \alpha _ { t } ^ { 2 } \Sigma _ { i } + \bar { \sigma } _ { t } ^ { 2 } I .\tag{101}
$$

Exact score. The exact score of the noised marginal is the weighted sum of the per-component scores. For any mixture,

$$
\nabla _ { x } \log { p _ { t } ( x ) } ~ = ~ \frac { \sum w _ { i } \nabla _ { x } \mathcal { N } _ { i } } { \sum w _ { j } \mathcal { N } _ { j } } = \frac { \sum w _ { i } \mathcal { N } _ { i } \big [ - S _ { t , i } ^ { - 1 } ( x - \alpha _ { t } \mu _ { i } ) \big ] } { \sum w _ { j } \mathcal { N } _ { j } } .\tag{102}
$$

We use this exact score wherever analytical score is reported.

Exact look-ahead guidance. The exact guidance is the gradient of the Doob reward of Eq. (14) (Dai Pra, 1991; Denker et al., 2024), which for the soft reward reads $r _ { t } ^ { \star } ( x ) = \log G _ { t } ( x )$ with $G _ { t } ( \boldsymbol { x } ) = \mathbb { E } _ { p ( \boldsymbol { x } _ { 0 } | \boldsymbol { x } _ { t } ) } [ \exp ( r _ { 0 , \delta } ( \boldsymbol { x } _ { 0 } ) ) ]$ , i.e. $\nabla _ { x } r _ { t } ^ { \star } = \nabla _ { x } \log G _ { t } ( x )$ . Conjugacy makes this closed-form. We can use the fact that $p ( x _ { 0 } | x _ { t } , i ) \propto p ( x _ { t } | x _ { 0 } ) p ( x _ { 0 } | i )$ , hence the posterior under each component i is a product of two Gaussians - another Gaussian, $\mathcal { N } ( \boldsymbol { m } _ { t , i } , C _ { t , i } )$ with,

$$
\begin{array} { r } { C _ { t , i } = \Bigl ( \Sigma _ { i } ^ { - 1 } + \frac { \alpha _ { t } ^ { 2 } } { \bar { \sigma } _ { t } ^ { 2 } } I \Bigr ) ^ { - 1 } , \qquad m _ { t , i } = C _ { t , i } \Bigl ( \Sigma _ { i } ^ { - 1 } \mu _ { i } + \frac { \alpha _ { t } } { \bar { \sigma } _ { t } ^ { 2 } } x \Bigr ) . } \end{array}\tag{103}
$$

To evaluate the expected reward, we consider a linear observable $\phi ( x _ { 0 } ) = a ^ { \top } x _ { 0 }$ . Under the ith component posterior, this projection is a 1D Gaussian, $a ^ { \top } X _ { 0 } \sim { \mathcal { N } } ( a ^ { \top } m _ { t , i } , s _ { t , i } ^ { 2 } )$ with $s _ { t , i } ^ { 2 } =$ $a ^ { \top } C _ { t , i } a$ , so the per-mode look-ahead reward integrates to another Gaussian CDF with,

$$
\mathbb { E } _ { p ( x _ { 0 } | x _ { t } ) } [ \Phi ( . ) ] = \Phi ( \eta _ { t , i } ) , \qquad \eta _ { t , i } = \frac { a ^ { \top } m _ { t , i } - \tau } { \sqrt { \delta ^ { 2 } + s _ { t , i } ^ { 2 } } } .\tag{104}
$$

We can then write the global look-ahead potential $G _ { t } ( x )$ as the expected reward marginalised over all M. Using Eq.(104), we can write $\begin{array} { r } { G _ { t } ( x ) = \sum _ { i = 1 } ^ { M } w _ { t , i } \Phi ( \eta _ { t , i } ) } \end{array}$ , giving the exact guidance as follows.

$$
\begin{array} { r l } & { \nabla _ { x } r _ { t } ^ { \star } ( x ) = \nabla _ { x } \log G _ { t } ( x ) } \\ & { \qquad = \frac { 1 } { G _ { t } ( x ) } \nabla _ { x } \left( \displaystyle \sum _ { i = 1 } ^ { M } w _ { t , i } ( x ) \Phi ( \eta _ { t , i } ) \right) } \\ & { \qquad = \displaystyle \frac { 1 } { G _ { t } ( x ) } \displaystyle \sum _ { i = 1 } ^ { M } [ \Phi ( \eta _ { t , i } ) \nabla _ { x } w _ { t , i } ( x ) + w _ { t , i } ( x ) \nabla _ { x } \Phi ( \eta _ { t , i } ) ] } \end{array}\tag{105}
$$

We can rewrite the above equation as,

$$
\nabla _ { \boldsymbol { x } } \boldsymbol { r } _ { t } ^ { \star } ( \boldsymbol { x } ) = \sum _ { i = 1 } ^ { M } \widetilde { \boldsymbol { w } } _ { i } ( \boldsymbol { x } ) \left[ \frac { \nabla _ { \boldsymbol { x } } \Phi ( \eta _ { t , i } ) } { \Phi ( \eta _ { t , i } ) } + \frac { \nabla _ { \boldsymbol { x } } \boldsymbol { w } _ { t , i } ( \boldsymbol { x } ) } { \boldsymbol { w } _ { t , i } ( \boldsymbol { x } ) } \right] \qquad \widetilde { \boldsymbol { w } } _ { i } ( \boldsymbol { x } ) = \frac { \boldsymbol { w } _ { t , i } ( \boldsymbol { x } ) \Phi ( \eta _ { t , i } ) } { \sum _ { j = 1 } ^ { M } \boldsymbol { w } _ { t , j } ( \boldsymbol { x } ) \Phi ( \eta _ { t , j } ) }\tag{106}
$$

Here $\widetilde { w } _ { i }$ represents the reward weighted (posterior) responsibility of the GMM - i.e. it is the responsibility re-weighted by each mode’s chance of reaching the rare event set (Bishop & Nasrabadi, 2006). A mode i is up-weighted when $x _ { t }$ is consistent with it (i.e. $w _ { t , i }$ large) and when it is well placed to cross the threshold $( \Phi ( \eta _ { t , i } )$ large). The first term inside the bracket is the per-mode push towards the rare set; second is the routing correction that re-weights mass between modes as the reward favours different components. Evaluating the two terms in $\bar { \nabla } _ { \boldsymbol { x } } r _ { t } ^ { \star }$ , the first term is the reward push,

$$
\frac { \nabla _ { x } \Phi ( \eta _ { t , i } ) } { \Phi ( \eta _ { t , i } ) } = \frac { \varphi ( \eta _ { t , i } ) } { \Phi ( \eta _ { t , i } ) } g _ { t , i } = \psi ( \eta _ { t , i } ) g _ { t , i } ,\tag{107}
$$

where $\begin{array} { r } { g _ { t , i } = \nabla _ { x } \eta _ { t , i } = \frac { \alpha _ { t } } { \bar { \sigma } _ { t } ^ { 2 } } \frac { C _ { t , i } a } { \sqrt { \delta ^ { 2 } + s _ { t , i } ^ { 2 } } } } \end{array}$ and $\psi ( \cdot ) = \varphi ( \cdot ) / \Phi ( \cdot )$ is the Inverse-Mills ratio.

The second term of Eq. (106) represents the standard GMM responsibility at a noised time t, given by,

$$
w _ { t , i } ( x ) = \frac { w _ { i } \mathcal { N } ( x ; \alpha _ { t } \mu _ { i } , S _ { t , i } ) } { p _ { t } ( x ) } .
$$

Evaluating this yields,

$$
\begin{array} { r } { \nabla _ { x } \log { w _ { t , i } ( x ) } = \nabla _ { x } \log { \mathcal { N } ( x ; \alpha _ { t } \mu _ { i } , S _ { t , i } ) } - \nabla _ { x } \log { p _ { t } ( x ) } . } \end{array}\tag{108}
$$

Putting this together, we get the analytical guidance for the GMM,

$$
\nabla _ { x } r _ { t } ^ { \star } ( x ) = \sum _ { i = 1 } ^ { M } \widetilde { w } _ { i } ( x ) \Big [ \psi ( \eta _ { t , i } ) g _ { t , i } + \nabla _ { x } \log \mathcal { N } ( x ; \alpha _ { t } \mu _ { i } , S _ { t , i } ) - \nabla _ { x } \log p _ { t } ( x ) \Big ] .\tag{109}
$$

We refer readers to (Wu et al., 2024) for a rigorous exposition of the effect of guidance on GMMs.

## G 2D GMM EXPERIMENTS

## G.1 EXPERIMENTAL DETAILS

The learned score. The learned score for the correlated 7-component GMM of Table 4, is a velocity prediction MLP trained under the linear VP schedule (Song et al., 2021b). The network takes $( \mathbf { \bar { \boldsymbol { x } } } , \operatorname { e m b } ( t ) ) \in \mathbb { R } ^ { 2 + 1 2 8 }$ , where t is embedded using a sinusoidal time embedding through 6 layers (hidden width 512, SiLU (Elfwing et al., 2018); 1.1M parameters) to a two-dimensional output. Targets are $v = \alpha _ { t } \varepsilon - \bar { \sigma } _ { t } x _ { 0 }$ with $x _ { t } = \alpha _ { t } x _ { 0 } + \bar { \sigma } _ { t } \varepsilon$ ; the score used by the sampler is recovered as $s _ { \theta } ( x , t ) = - \hat { \varepsilon } / \bar { \sigma } _ { t }$ with $\hat { \varepsilon } = \bar { \sigma } _ { t } x + \alpha _ { t } v _ { \theta } ( x , t )$ . The loss is the per-sample v-MSE under min-SNR-γ weighting with $\gamma = 5$ and the weights given by $w ( t ) = \mathrm { m i n } ( \mathrm { S N R } ( \boldsymbol { \dot { t } } ) , \boldsymbol { \gamma } ) / ( \mathrm { S N R } ( t ) + 1 )$ ) of (Hang et al., 2023), with $t \sim \mathcal { U } [ 0 . 0 1 , 0 . 9 9 ]$ . Training draws fresh exact GMM samples every step: 40,000 steps at batch 4096 with AdamW (Kingma & Ba, 2014; Loshchilov & Hutter, 2019) (learning rate $2 \times 1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 } )$ , 1,000-step linear warm-up followed by cosine decay, gradient clipping at 1.0, and an exponential moving average of the weights (decay 0.999); the EMA weights are used for evaluation.

The tempering ladder. Our guidance ladder used through the experiment is $\lambda ( t ) = 1 - ( t / t _ { \mathrm { o n } } ) ^ { 2 }$ for $t < t _ { \mathrm { o n } } = 0 . 9$ in the case of rejuvenation and $\lambda _ { t } \equiv 1$ without it, see App. G.5. We chose this simple form of tempering, inspired by polynomial tempered likelihood (Friel & Pettitt, 2008).

The benchmark. The mixture parameters and every closed form we compare against: the base density, the exact score, the soft-tilt normaliser $Z _ { 0 , \delta }$ of Eq. (98), the exceedance probability $p _ { 0 } [ E ]$ of Eq. (99), the tilted target $\pi _ { 0 , \delta }$ of Eq. (100) and the exact Doob guidance $\nabla _ { \boldsymbol { x } } r _ { t } ^ { \bar { \star } }$ of Eq. (106) are defined in Appendix F. The right-hand modes carry strong off-diagonal covariance of opposite sign, so the tail conditional $p _ { 0 } ( x _ { 2 } \mid x _ { 1 } > \tau )$ is genuinely bi-modal: hitting the rare set requires reproducing a multi-modal conditional. The learned-score arm uses the v-prediction MLP described above.

Sampling parameters Unless stated otherwise every run below uses the deployed configuration of Section 4.1: N = 1000 particles, $K = 1 0 0 0 \thinspace \mathrm { S E E D S - 1 }$ steps on a uniform half-log SNR noise schedule with $\delta = 0 . 0 1$ , the quadratic tempering ladder λ(t) above for the rejuvenation arm and $\lambda \equiv 1$ otherwise, adaptive resampling at $\mathrm { E S } \mathrm { \bar { S } } / N \mathrm { \bar { < } } 0 . 5$ with three MADM rejuvenation sweeps per resampling (Simpson’s (1/3) rule with Eq. (119)), and 24 seeds. The MH step is $\rho = c ( t ) \bar { \sigma } _ { t } ^ { 2 }$ with $c ( t )$ fixed once from a pilot sweep targeting an acceptance ratio of ≈ 0.72 at each noise level, as prescribed in (Lam et al., 2026).

All experiments were done on a single Nvidia A100 GPU with 40 GB of VRAM.

## G.2 SOURCES OF BIAS

We begin by discussing the two main contributing factors that bias our estimate against the true $p _ { 0 } [ E ]$ , followed by additional results. Neither bias is an error of the estimator, and the path-weighted estimator remains unbiased exactly for the law we sample from. The two primary sources are the learning process where the score is approximated poorly (Appendix G.2.1) and when the discretised law we sample from differs from the continuous law (Appendix G.2.2). We separate and isolate each to study their contributions to the final estimate.

## G.2.1 TAIL BIAS OF THE LEARNED SCORE

We estimate the tail bias of the learned score by comparing the model’s output at a specific time and location $( x _ { t } , t )$ against the analytical score. Write $\dot { \Delta s } ( x , \overline { { t } } ) = s _ { \theta } ( x , t ) - \dot { s ^ { \star } } ( x , t )$ for the pointwise gap between the learned and exact scores. The induced log-density bias $b ( x _ { 0 } ) : = \log p _ { \mathrm { m o d e l } } ( x _ { 0 } ) -$ log $p ^ { \star } ( x _ { 0 } )$ accumulates along the probability-flow ODE trajectory $\operatorname { \bar { \{ x _ { t } \} } } _ { t \in [ 0 , 1 ] }$ that carries $x _ { 0 }$ back to the prior under the learned velocity field, as

$$
b ( \boldsymbol { x } _ { 0 } ) \ = \ - \int _ { 0 } ^ { 1 } \frac { \ d ^ { 1 } } { \ d ^ { 2 } } \beta ( t ) \Big [ \boldsymbol { \nabla } { \cdot } \Delta \boldsymbol { s } \ + \ \boldsymbol { s } ^ { \star } { \cdot } \Delta \boldsymbol { s } \Big ] ( \boldsymbol { x } _ { t } ) \mathrm { d } t ,\tag{110}
$$

from which the tail ratio follows by reweighting the exact conditional,

$$
\frac { P _ { \mathrm { m o d e l } } ( \phi ( X ) \geq \tau ) } { P ^ { \star } ( \phi ( X ) \geq \tau ) } = \mathbb { E } _ { p ^ { \star } ( \cdot \vert \phi ( X ) \geq \tau ) } \left[ e ^ { b ( x _ { 0 } ) } \right] .\tag{111}
$$

Table 5: Tail bias of the learned 2D-GMM score. $p _ { \mathrm { m o d e l } }$ (MC) is obtained from a large MC reference pool using unguided draws. Path integral measures the ratio obtained by integrating the pointwise score error along the probability-flow ODE (Eq. (110)) and reweighing the exact conditional. The two induce different laws but agree to within 7% across three decades of rarity. $p _ { \mathrm { d a t a } } [ E ]$ is the analytical probability.
<table><tr><td>T</td><td> $p _ { \mathrm { d a t a } } [ E ]$ </td><td> $p _ { \mathrm { m o d e l } } \left( \mathrm { M C } \right)$ </td><td> $\ p _ { \mathrm { m o d e l } } / p _ { \mathrm { d a t a } } [ E ]$ </td><td>rel. s.e.</td><td>path integral</td><td>ratio</td></tr><tr><td>7</td><td> $6 . 8 8 \times 1 0 ^ { - 3 }$ </td><td> $7 . 1 3 \times 1 0 ^ { - 3 }$ </td><td>1.037</td><td>0.08%</td><td>1.075</td><td>1.037</td></tr><tr><td>8</td><td> $1 . 2 8 \times 1 0 ^ { - 3 }$ </td><td> $1 . 3 8 \times 1 0 ^ { - 3 }$ </td><td>1.081</td><td>0.18%</td><td>1.156</td><td>1.069</td></tr><tr><td>9</td><td> $1 . 6 2 \times 1 0 ^ { - 4 }$ </td><td> $1 . 8 7 \times 1 0 ^ { - 4 }$ </td><td>1.153</td><td>0.50%</td><td>1.236</td><td>1.072</td></tr><tr><td>10</td><td> $1 . 4 0 \times 1 0 ^ { - 5 }$ </td><td> $1 . 7 2 \times 1 0 ^ { - 5 }$ </td><td>1.229</td><td>1.64%</td><td>1.267</td><td>1.031</td></tr></table>

In practice, we draw $2 . 1 6 \times 1 0 ^ { 8 }$ samples from the model unconditionally and use it as our reference pool. We find that with both methods, the tail bias widens with rarity, and the score-estimate degrades monotonically, see Table 5. All quoted comparisons for the trained model are against the model’s own rare event probability $p _ { 0 } [ E ]$ , obtained via Monte Carlo estimate.

## G.2.2 THE DISCRETISED LAW: GRID AND INTEGRATOR

Alongside the bias due to the score estimate, which in the general case cannot be estimated, our discretisation itself induces a distinct, reducible form of bias. Let P be the continuous path measure defined by the diffusion process, and $\mathbb { P } _ { k }$ bethe discretised law that we simulate. Our estimator is unbiased for the discrete law that we compute weights for under any guidance (Del Moral et al., 2006). Hence any gap that remains is a property of the base chain and the integrator alone. This section first measures this gap for two first order integrators and whether the choice of integrator reduces it at a fixed K.

Setup. Exact Doob guidance (i.e. $\nabla _ { \boldsymbol { x } } r _ { t } ^ { * } ( \boldsymbol { x } ) )$ , with resampling turned off, so neither guidance error nor adaptive resampling contributes and the residual is resolved above seed noise. We compare the Euler-Maruyama (EM) integrator against the first order exponential SDE solver of (Gonzalez et al., 2023) which we denote as SEEDS-1. Both are explicit one-step schemes of first order. The only difference is that SEEDS-1 treats the integration of the reverse SDE as a semi-linear integral, separating the linear and non-linear parts of the score, and integrating the linear and the noise term exactly, whereas EM approximates it to first order, see (Gonzalez et al., 2023; Milstein & Tretyakov, 2004; Pfarr et al., 2026) for more details. Alongside the integrator, the placement of the grid itself is a design choice: we run the sweep on a grid uniform in t and on a grid uniform in the half-log Signal-to-Noise ratio $\ell _ { t } = \log ( \alpha _ { t } / \bar { \bar { \sigma } } _ { t } )$ of App. D.1, the integrator’s own time variable. Table 6 sweeps the step count $K$ under both placements and compares the signed bias in the probability estimate with the ground truth in $\tau = 1 0$  
Table 6: Signed discretisation bias versus step count K for the two integrators, under exact Doob guidance, at different grids, uniform in t (top block) and uniform in the half-log SNR $\ell _ { t } = \log ( \alpha _ { t } / \bar { \sigma } _ { t } )$ (bottom block). Positive values are overestimates. Paired columns are EM/SEEDS-1 throughout. <sup>ˆ</sup>k is the Pareto Smoothed Importance Sampling shape estimate (Vehtari et al., 2024) on the weights, and ESS<sub>E</sub> is the ESS of the final particles, measured after thresholding to $\tau \geq 1 0 .$
<table><tr><td>K</td><td>EM bias</td><td>SEEDS-1 bias</td><td>|EM|/|SEEDS|</td><td>CV↓</td><td> $\hat { k } \downarrow$ </td><td>ESSE ↑</td></tr><tr><td colspan="7">Uniform in t</td></tr><tr><td>10</td><td> $+ 6 4 . 0 2 \% \pm 1 1 . 3 1 \%$ </td><td> $- 1 5 . 9 8 \% \pm 1 . 8 8 \%$ </td><td>4.01</td><td>0.338/0.1090.85/0.46</td><td></td><td>0.032/0.102</td></tr><tr><td>15</td><td> $+ 4 8 . 6 1 \% \pm 8 . 3 9 \%$ </td><td> $- 1 6 . 2 3 \% \pm 1 . 2 8 \%$ </td><td>3.00</td><td>0.277/0.075</td><td>0.70/0.45</td><td>0.063/0.189</td></tr><tr><td>20</td><td> $+ 5 4 . 9 0 \% \pm 1 6 . 5 0 \%$ </td><td> $- 1 5 . 1 3 \% \pm 0 . 9 7 \%$ </td><td>3.63</td><td>0.522/0.056</td><td>0.68/0.43</td><td>0.099/0.248</td></tr><tr><td>30</td><td> $+ 2 3 . 4 1 \% \pm 1 . 6 5 \%$ </td><td> $- 1 3 . 6 2 \% \pm 0 . 7 5 \%$ </td><td>1.72</td><td>0.065/0.043</td><td>0.56/0.40</td><td>0.176/0.375</td></tr><tr><td>50</td><td> $+ 1 6 . 7 5 \% \pm 1 . 3 8 \%$ </td><td> $- 8 . 3 6 \% \pm 0 . 6 3 \%$ </td><td>2.00</td><td>0.058/0.034</td><td>0.55/0.45</td><td>0.315/0.465</td></tr><tr><td>100</td><td> $+ 6 . 8 2 \% \pm 0 . 6 9 \%$ </td><td> $- 4 . 9 9 \% \pm 0 . 5 7 \%$ </td><td>1.37</td><td>0.032/0.029</td><td>0.40/0.45</td><td>0.600/0.634</td></tr><tr><td>200</td><td> $+ 3 . 7 2 \% \pm 0 . 3 0 \%$ </td><td> $- 2 . 7 1 \% \pm 0 . 2 6 \%$ </td><td>1.38</td><td>0.014/0.013</td><td>0.40/0.40</td><td>0.771/0.802</td></tr><tr><td>500</td><td> $+ 1 . 3 0 \% \pm 0 . 1 9 \%$ </td><td> $- 1 . 2 7 \% \pm 0 . 1 6 \%$ </td><td>1.02</td><td>0.009/0.008</td><td>0.39/0.40</td><td>0.901/0.906</td></tr><tr><td>1000</td><td> $+ 0 . 7 5 \% \pm 0 . 1 9 \%$ </td><td> $- 0 . 5 1 \% \pm 0 . 1 7 \%$ </td><td>1.46</td><td>0.009/0.009</td><td>0.36/0.39</td><td>0.943/0.945</td></tr><tr><td colspan="7">Uniform in  $\ell _ { t } = \log ( \alpha _ { t } / \bar { \sigma } _ { t } )$ </td></tr><tr><td>10</td><td> $- 3 4 . 5 5 \% \pm 2 . 7 7 \%$ </td><td> $+ 2 5 6 . 8 2 \% \pm 3 9 . 0 9 \%$ </td><td>0.13</td><td>0.207/0.537</td><td>0.75/0.80</td><td>0.031/0.021</td></tr><tr><td>15</td><td> $- 2 4 . 9 3 \% \pm 3 . 1 2 \%$ </td><td> $+ 2 8 . 6 8 \% \pm 2 . 4 2 \%$ </td><td>0.87</td><td></td><td></td><td>0.203/0.092 0.58/0.480.083/0.115</td></tr><tr><td>20</td><td> $- 2 0 . 6 8 \% \pm 1 . 7 6 \%$ </td><td> $+ 1 2 . 6 6 \% \pm 1 . 6 3 \%$ </td><td>1.63</td><td>0.109/0.071 0.46/0.40 0.169/0.201</td><td></td><td></td></tr><tr><td>30</td><td> $- 1 3 . 7 8 \% \pm 0 . 8 8 \%$ </td><td> $+ 2 . 1 3 \% \pm 1 . 0 1 \%$ </td><td>6.46</td><td>0.050/0.048 0.38/0.37 0.283/0.313</td><td></td><td></td></tr><tr><td>50</td><td> $- 8 . 9 8 \% \pm 0 . 5 9 \%$ </td><td> $- 2 . 1 9 \% \pm 0 . 7 0 \%$ </td><td>4.09</td><td>0.032/0.035 0.34/0.34 0.474/0.506</td><td></td><td></td></tr><tr><td>100</td><td> $- 4 . 6 3 \% \pm 0 . 4 2 \%$ </td><td> $- 2 . 5 9 \% \pm 0 . 4 2 \%$ </td><td>1.79</td><td>0.022/0.021 0.24/0.260.713/0.724</td><td></td><td></td></tr><tr><td>200</td><td> $- 2 . 5 4 \% \pm 0 . 3 6 \%$ </td><td> $- 1 . 8 9 \% \pm 0 . 3 4 \%$ </td><td>1.35</td><td>0.018/0.017 0.17/0.18 0.841/0.849</td><td></td><td></td></tr><tr><td>500</td><td> $- 1 . 2 9 \% \pm 0 . 1 7 \%$ </td><td> $- 1 . 0 2 \% \pm 0 . 1 7 \%$ </td><td>1.27</td><td>0.008/0.009 0.19/0.19 0.928/0.930</td><td></td><td></td></tr><tr><td>1000</td><td> $- 0 . 2 7 \% \pm 0 . 1 3 \%$ </td><td> $- 0 . 1 6 \% \pm 0 . 1 3 \%$ </td><td>1.66</td><td>0.007/0.0060.22/0.23 0.956/0.957</td><td></td><td></td></tr></table>

The $1 / K$ law. Both integrators are one-step schemes of weak order one (Milstein & Tretyakov, 2004), for which the weak-error admits the expansion $c _ { 1 } / K + O ( 1 / K ^ { 2 } )$ with $c _ { 1 }$ independent of the step size (Milstein & Tretyakov, 2004; Talay & Tubaro, 1990). While this holds for smooth test functions, it does not automatically hold for the indicator $\mathbf { 1 } _ { E } .$ . We therefore fit the form using ordinary least-squares instead of invoking it directly as a theorem. Fitting the coefficient $c _ { 1 }$ , weighted by the per-K standard errors $s _ { k }$

$$
\hat { c } _ { 1 } = \frac { \sum _ { k } b _ { k } K _ { k } ^ { - 1 } s _ { k } ^ { - 2 } } { \sum _ { k } K _ { k } ^ { - 2 } s _ { k } ^ { - 2 } } ,\tag{112}
$$

restricted to $K \geq 3 0 ,$ , since $K = 1 0 – 2 0$ sits outside the asymptotic regime. On the uniform-t grid the fits are $\hat { c } _ { 1 } = + 7 . 2 7$ for Euler-Maruyama against −4.41 for SEEDS-1, with largest residuals of 1.6 and 2.4 standard errors respectively, so the $\bar { 1 } / K$ law holds to within roughly two standard errors over the fitted range. The signs record that the two integrators approach the continuous limit from opposite sides on this grid: EM overestimates and SEEDS-1 underestimates at every $K .$ , and the $| \mathrm { \bar { E } M } | / | \mathrm { S E E D S } |$ column of Table 6 is the per- $K$ magnitude comparison. The fitted coefficients track the measurements closely: $\hat { c } _ { 1 } = 7 . 2 7$ predicts $\Game + 3 . 6 \%$ at $K = 2 0 0$ against a measured $+ 3 . 7 2 \% \mathrm { . }$ and +14.6% at $K = 5 0$ against +16.8%. On the uniform-ℓ grid the EM arm fits similarly $( \hat { c } _ { 1 } = - 4 . 4 5$ largest residual 2.4 standard errors), but the SEEDS-1 arm changes sign between $K = 3 0$ and $K = 5 0 \colon$ its leading coefficient is small enough that higher-order terms are not negligible over the measured range.

Grid placement. At coarse grids the two placements order the integrators differently. In uniform in t, SEEDS-1 dominates EM in every $K \colon$ : the magnitude ratio drops from 4.0 in $K = 1 0$ towards parity by $K \geq 5 0 0$ . For the uniform in the ℓ grid, the order inverts in the coarsest grids $K = 1 0$ and $K = 1 5$ , where we reproduce the result of (Gonzalez et al., 2023) where the performance of the integrator is poor at very small discretisation. We hypothesise that this is because, a uniform $\Delta \ell$ starves the $\mathrm { \cdots } \mathrm { m i d d l e ^ { \prime } { } }$ of the denoising process, which principled methods for noise-scheduling such as (Williams et al., 2024) place a larger number of discretisation on, see Fig. 4 of (Williams et al., 2024), which adversely affects the discretised law that we sample. Given that the schedule intimately interplays with Gaussian transition kernel approximation, we leave studying the effects it has on the weight computation for future work.

From $K \geq 2 0$ SEEDS-1 yields a smaller discretisation bias, and the two integrators converge to each other by $K = 2 0 0$ . We find that in our setting, SEEDS-1 reduces the discretisation bias at lower K on either grid placement away from the coarsest ℓ-grids, and both integrators are comparable at higher $K \ ' s$ . We deploy SEEDS-1 at $K = 1 0 0 0$ on the uniform-ℓ grid, where both integrators are within 0.2% of the continuous limit (Table 6, last row). The exponential integrator is also used in the larger climate experiments.

Both channels improve together. In this experiment, the integrator does not only control the bias but also affects the variance. Across Table 6 the variance diagnostics move with it: on the uniform-t EM arm the CV falls from 0.338 to 0.009, $\mathrm { E S S } _ { E }$ climbs from 0.032 to 0.943, with the PSIS <sup>ˆ</sup>k dropping below 0.5 by $K = 1 0 0$ . The grid placement moves the same diagnostics: at matched K the ℓ-grid improves $\mathrm { E S S } _ { E }$ and <sup>ˆ</sup>k for both integrators $( \mathrm { e . g . }$ . at $K = 1 0 0$ , EM’s $\mathrm { E S S } _ { E }$ is $0 . 7 1 3$ uniform in ℓ against 0.600 uniform in t, with $\hat { k }$ of 0.24 against 0.40), the coarsest ℓ-grids under SEEDS-1 excepted. These are not two separate effects but one mechanism with two consequences. A finer grid puts the discrete chain closer to the continuous dynamics, which simultaneously shrinks the gap $\mathbb { P } _ { K } - \mathbb { P }$ that the bias measures. As we simulate using the exact Doob’s guidance, this moves the discretised proposal $\mathbb { Q } ^ { K }$ close to the variance minimising Doob’s proposal which helps all other metrics. In a general setting with approximate guidance the dominant source of error is not the discretisation bias but instead the approximation error of Doob’s h-transform, e.g. using DPS.

## G.3 THE THRESHOLD-EXCEEDANCE EXPERIMENT

Everything in this subsection belongs to the deployed threshold-exceedance experiment of Section 4.1: the conditional clouds behind the sample-quality columns, the no-resampling arm, the whole-path ESS diagnostics, and how the estimator scales with the particle budget N. We summarise our results in Table. 7, where we find that our method recovers the correct probability estimates as well as the correct shape statistics. We analyse all individual aspects of these results in the section below. All experiments were done with the same guidance with $K = 1 0 0 0$ and the SEEDS-1 integrator on the uniform-ℓ grid unless stated otherwise.

## G.3.1 CONDITIONAL-SAMPLE COVERAGE: CLOUDS FOR LEARNED AND ANALYTICAL

We plot the sample clouds $( N = 2 0 4 8 )$ using the deployed configuration of Section 4.1 for both Analytical and Learned Guidance with the same guidance/resampling/rejuvenation parameters. The red samples are drawn from the exact conditional $p ( x \mid x _ { 1 } \geq \tau )$ , drawn larger at $( \bar { 2 } \times 1 0 ^ { 4 } )$ and the blue particles represent the guided samples. Together with the KS/EMD ratios of Table 7 the plots qualitatively highlight that samples are drawn from the conditional distribution itself. The two clouds remain visually indistinguishable as can be seen in Figure 10.

## G.3.2 ESS ALONG THE TRAJECTORY

Without resampling the accumulated weight is never reset and the terminal ESS asymptotically converges to $1 \bar { / } ( 1 \bar { + } \chi ^ { 2 } ( \mathbb { P } _ { 0 : K } ^ { \star } \| \mathbb { Q } _ { 0 : K } ) )$ (Agapiou et al., 2017). Hence for the sampled path, the ESS measures the gap between the implemented guidance and the exact Doob’s guidance. Figure 11 tracks

![](images/322ccffce35a5c076b367d111b060aeda64c8411d48acabf7bd2e0b5cd06ae00.jpg)

![](images/c24de2a97828536433969ddcc491ab711cbfd263c193c0e701076a4a4a8ef4a0.jpg)

Figure 7: Discretisation bias for the three integrators, Euler-Maruyama, SEEDS-1 and DDIM. We unconditionally draw 1e8 samples from the base model at different discretisation K and evaluate the bias at different rarities. The figure on the right shows the number of discretisation required to get a $< 1 0 \%$ bias for different rarities, showing that the SDE requires less steps than the ODE integrator. Top row uses the uniform log- $. S \mathrm { N R } \Delta \ell$ schedule and the bottom is uniform in time $\Delta t$  
(a)  
![](images/91c0f93e015b87391b42baeb249d89436258b116814c7d7e7f24378d7d750c5d.jpg)

(b)  
![](images/cdfd302246dadf5bf997c8caea8f1e6d824ae81cb9da850953b9adc73eeb01bb.jpg)  
Figure 8: Measured bias versus K with 1000 particles, analytical scores and Doob’s transition kernels. See App G.2.2. We show the bias of our estimate falls with a $O ( 1 / K )$ leading order term. We compare both the EM integrator and the SEEDs-1 integrator in the noise prediction form. We compare two different noise schedules: uniform in time and uniform in log-SNR ℓ. Error-bars represent standard error of the mean. Experiment is conducted the $\tau = 1 0$ threshold exceed problem. See Table 6.

it along the trajectory for both scores and five thresholds. Two features that are worth noting. The trace is flat at $\mathrm { \bar { E } S S } / \dot { N } = 1$ until $t \approx 0 . 9$ , which is the untilted stretch of the denoising time, where the guidance is turned off. After guidance is turned on, the ESS decays monotonically in both learned

![](images/d86edd79f5ab1f68db200f6069d57d9bf8eda912933ed8f3f5a6086f7144f5d6.jpg)  
Figure 9: Measured bias versus K with 10,000 particles and the deployed guidance and no rejuvenation for the learned score. We compare the discretisation bias at different rarities, we compare against a $1 0 ^ { 8 }$ unconditional MC draw of the exponential integrator at $K$ as our reference. Hence, this measures how quickly the estimates converge to the reference with K.

Table 7: Guided generation on the GMM with learned and analytical scores. Each arm is scored against the law it samples from, $p ^ { \star } ;$ ; either $p _ { \mathrm { d a t a } } [ E ]$ (analytical) or an MC estimate of $p _ { 0 } [ E ]$ (learned). Sample quality metrics are evaluated against a reference from the same law, presented as a ratio against a noise-floor. Estimates $\hat { p } / p ^ { \star }$ are biased to the MADM kernel’s floor (Fig. 13). We evaluate the two-sample Kolmogorov-Smirnov (KS) test statistic, the Earth Mover’s Distance (EMD) and the coefficient of variation $\mathsf { \bar { C } V } ^ { 2 } = \mathrm { V a r } ( \hat { p _ { E } } ) / \bar { p _ { E } ^ { 2 } }$
<table><tr><td>T</td><td>score</td><td> $p ^ { \star }$ </td><td> $\hat { p } / { p ^ { \star } }$ </td><td> $\mathrm { C V \downarrow }$ </td><td> ${ \mathrm { K S } } _ { x _ { 1 } }$ </td><td> ${ \mathrm { K S } } _ { x _ { 2 } }$ </td><td>EMD</td></tr><tr><td>7</td><td>analytical</td><td> $6 . 8 8 \times 1 0 ^ { - 3 }$ </td><td> $0 . 9 8 0 \pm 0 . 0 1 2$ </td><td>0.060</td><td>0.88</td><td>1.33</td><td>1.52</td></tr><tr><td>8</td><td>analytical</td><td> $1 . 2 8 \times 1 0 ^ { - 3 }$ </td><td> $0 . 9 6 6 \pm 0 . 0 1 2$ </td><td>0.061</td><td>1.13</td><td>1.10</td><td>1.06</td></tr><tr><td>9</td><td>analytical</td><td> $1 . 6 2 \times 1 0 ^ { - 4 }$ </td><td> $0 . 9 7 4 \pm 0 . 0 1 6$ </td><td>0.083</td><td>1.12</td><td>1.00</td><td>0.97</td></tr><tr><td>10</td><td>analytical</td><td> $1 . 4 0 \times 1 0 ^ { - 5 }$ </td><td> $0 . 9 5 4 \pm 0 . 0 1 3$ </td><td>0.069</td><td>0.94</td><td>1.05</td><td>1.46</td></tr><tr><td>7</td><td>learned</td><td> $7 . 1 3 \times 1 0 ^ { - 3 }$ </td><td> $0 . 9 7 8 \pm 0 . 0 1 2$ </td><td>0.060</td><td>1.11</td><td>1.20</td><td>1.14</td></tr><tr><td>8</td><td>learned</td><td> $1 . 3 8 \times 1 0 ^ { - 3 }$ </td><td> $0 . 9 8 5 \pm 0 . 0 1 3$ </td><td>0.064</td><td>0.97</td><td>1.00</td><td>0.73</td></tr><tr><td>9</td><td>learned</td><td> $1 . 8 7 \times 1 0 ^ { - 4 }$ </td><td> $0 . 9 8 0 \pm 0 . 0 1 8$ </td><td>0.089</td><td>0.77</td><td>0.83</td><td>0.98</td></tr><tr><td>10</td><td>learned</td><td> $1 . 7 2 \times 1 0 ^ { - 5 }$ </td><td> $0 . 9 6 5 \pm 0 . 0 1 6$ </td><td>0.079</td><td>1.23</td><td>1.07</td><td>0.98</td></tr></table>

and analytical arms. The two panels remain close to indistinguishable, highlighting that the profile is set by the guidance geometry rather than the score itself.

## G.3.3 SELECTION AND REJUVENATION

We ablate whether resampling or rejuvenation is helpful for our problem. For simplicity we provide two tables, separating the effect of each on the analytical and learned scores. Table 8 ablates the no-resampling, resampling and rejuvenation for the analytical score. Table 9 then runs the full threshold ladder on the learned score, adding the arm with no resampling at all and three noise levels where the rejuvenation move is switched off. All experiments are done using the same configuration as mentioned above. Each arm is scored against the law it samples from, $p ^ { \star }$ . Weighted sample quality statistics are evaluated against a conditional reference drawn from the same law, presented as ratios to their split-half noise floor. All reported uncertainties are computed over 24 seeds.

![](images/47ea3918ccce18948e2d614f44a9471411efb5f435fb1acfb2fcdbacfaa8b982.jpg)

(a) Analytical score  
![](images/c146513091018b69ea94e729be95b4a67bba3b2c6defacb58aa39ac314bf7414.jpg)

![](images/c09bda32302ed97f4314bb7589262540ec178904fb8759fa0c9fc5667e5fde56.jpg)

![](images/e7ec012fb1fd7f40f7920123e2e9065de2d48fe9e355637873de58f8f1adfe14.jpg)

![](images/367a975ff6016f7af6d534de5d04efebcd7f97404a2d5eed6c140a2830e1803b.jpg)

![](images/b83be60f6f75bd9fae68b2b934ad68d6265095f04d60863deb8f2f40acd18e5d.jpg)

![](images/2549f9cd495efb219c61f75f5dfd805ee89dc4fc0b57a6b60d9defb37949dc28.jpg)

![](images/8b641a0d0a3528aa5e6a770e6ad0b74287f31fc6de9715d8c2a1d919e025cbbc.jpg)

![](images/babdd214bd964c54c4ce52a88816877b9cf9c76a08d7b306914080e0dd8a53aa.jpg)

(b) Learned score  
![](images/b657739e7f52866d39ef4dba8cc2871ff534d5618bb44aa2803624f5572bbc18.jpg)

![](images/f692a388d7e7f44e521b620c5c0e81224b2f28c1c3692b02dbeb5c3b214553d9.jpg)

![](images/8f057256966a45bfa5750793d4417f6a64b0d480ed56aeb91fef581744c1ec4c.jpg)

![](images/50f8f162c6b8013eb400c1891595d8c8d3eaa7c9cae7c87d04d50ba4287365b0.jpg)

![](images/c5c55406048cf338813f66145118c1fe8b43e1e145c224f32b4bdda7686e77de.jpg)

![](images/626475f3ce6e1950d39f34c861f3d773d9698849383fbfa6bea0602b92689542.jpg)

![](images/51d61038d0ac0811f0157f797057c2e51ca53ea7780e753ffe115caaa3bf6aa3.jpg)

![](images/e1e932c3a8a691aa3636480b214502835821695841014e69b8ebd411d304366d.jpg)

![](images/3a9a74b39cee63ccbe1e2a6bcffaf5b0bd3a333b5fc87ccc6785cf566ebbc2f3.jpg)  
Figure 10: Conditional-sample coverage on the 2D GMM for the analytical (a) and learned (b) scores. Blue is the path-weighted guided cloud (2048 samples; pooled over 8 seeds for the learned score); red is a MC draw from the conditional $p ( x \mid x _ { 1 } \geq \tau )$ $L e f t .$ the joint cloud, with the threshold dashed. Centre, right: the $x _ { 1 }$ and $x _ { 2 }$ marginals on a log density scale. Corresponding quantitative KS/EMD metrics in Table 7.

Benefits of SMC Across, both Tables 8 and 9, resampling prunes low-weight trajectories and rejuvenation decorrelates duplicated samples. We find that rejuvenation keeps the relative variance constant across rarity for both analytical and learned scores, showing a noticeable decrease in CV if the base estimator is poor (Del Moral et al., 2006; 2012).

![](images/f7ec09bb16d942a1289100fde3528dbe206f021c7c3d148dd0277403e4ae23b6.jpg)  
Figure 11: Effective sample size of the path weights along diffusion time, without resampling (analytic score, left; learned score, right), for $\tau \in \{ 6 , 7 , 8 , 9 , 1 0 \}$ . Settings as Table 7. Time runs right to left, from noise (t = 1) to data (t = 0).

Table 8: Selection and rejuvenation on the 2D GMM under the analytical score, 24 seeds, all other settings as in Table 7. Resampling and rejuvenation are ablated at different threshold exceedance rarities. Weighted shape-metrics are computed against a conditional reference. Rejuvenation helps reduce the CV across all rarities; without it resampling alone can adversely prune lineages.
<table><tr><td>T</td><td>arm</td><td>CV↓</td><td> ${ \mathrm { K S } } _ { x _ { 1 } }$ </td><td> ${ \mathrm { K S } } _ { x _ { 2 } }$ </td><td>EMD</td></tr><tr><td rowspan="3">7</td><td>no resampling</td><td>0.067</td><td>0.88</td><td>1.52</td><td>1.50</td></tr><tr><td>resampling</td><td>0.062</td><td>1.05</td><td>0.99</td><td>0.83</td></tr><tr><td>resampling + rejuv. (M = 3)</td><td>0.060</td><td>0.88</td><td>1.33</td><td>1.52</td></tr><tr><td rowspan="3">8</td><td>no resampling</td><td>0.083</td><td>1.14</td><td>1.35</td><td>1.17</td></tr><tr><td>resampling</td><td>0.098</td><td>1.12</td><td>0.95</td><td>0.77</td></tr><tr><td>resampling + rejuv. (M = 3)</td><td>0.061</td><td>1.13</td><td>1.10</td><td>1.06</td></tr><tr><td rowspan="3">9</td><td>no resampling</td><td>0.113</td><td>0.91</td><td>1.11</td><td>1.03</td></tr><tr><td>resampling</td><td>0.150</td><td>1.15</td><td>1.47</td><td>1.14</td></tr><tr><td>resampling + rejuv. (M = 3)</td><td>0.083</td><td>1.12</td><td>1.00</td><td>0.97</td></tr><tr><td rowspan="3">10</td><td>no resampling</td><td>0.147</td><td>0.81</td><td>1.46</td><td>2.45</td></tr><tr><td>resampling</td><td>0.230</td><td>0.91</td><td>1.46</td><td>1.98</td></tr><tr><td>resampling + rejuv. (M = 3)</td><td>0.069</td><td>0.94</td><td>1.05</td><td>1.46</td></tr></table>

Gating the rejuvenation move by noise level For the analytical score, there is no score approximation error, hence the line-integral of the Metropolis-Hastings acceptance ratio is defined exactly up to quadrature error (Lam et al., 2026). However for a learned score it is approximated by $s _ { \theta } \approx \nabla _ { x _ { t } }$ log p<sub>t</sub>, and any errors in the learning process can lead to a non-conservative score approximation. The left of Fig. 12, evaluates this via the antisymmetric part of the Jacobian; here we reproduce the findings of (Lai et al., 2023; Thornton et al., 2025), where the score network becomes increasingly non-conservative as the denoising time, t → 0.

A MALA transition kernel requires that $M _ { k }$ fulfil the Kolmogorov criteria of reversibility. As the Metropolis-Hastings acceptance ratio of MADM (Lam et al., 2026) is estimated by the line-integral of the score, i.e. Eq. (118); this means the line integral, I(·) must fulfil the cyclic sum (or Stokes Theorem),

$$
I ( b , a ) + I ( c , b ) + I ( a , c ) = 0 ,\tag{113}
$$

Table 9: Resampling and rejuvenation ablation for the learned score network. Distributional metrics are presented as ratios to the reference pool’s split-half noise floor. Gate levels are quoted as the half-log SNR ℓ below which the rejuvenation move triggers; the adopted gate $\ell \lesssim - 0 . 2 \bar { 1 }$ is $t \geq 0 . 3 0$ The no-resampling and resampling-only arms run at $\bar { \lambda \equiv 1 }$ . Rejuvenation tracks the lowest CV.
<table><tr><td>event</td><td>arm</td><td> $\hat { p } / { p ^ { \star } }$ </td><td>CV↓</td><td> ${ \mathrm { K S } } _ { 0 }$ </td><td> ${ \mathrm { K S } } _ { 1 }$ </td><td>EMD</td></tr><tr><td rowspan="8"> $x _ { 0 } > 7$ </td><td>no resampling</td><td>0.952</td><td>0.070</td><td>0.74</td><td>1.00</td><td>1.09</td></tr><tr><td>resampling</td><td>0.988</td><td>0.093</td><td>0.94</td><td>1.25</td><td>1.47</td></tr><tr><td>resampling + rejuv. (M = 3)</td><td>0.985</td><td>0.061</td><td>1.01</td><td>1.17</td><td>1.28</td></tr><tr><td>gated  $\ell \lesssim 0 . 0 \bar { 5 }$ </td><td>0.983</td><td>0.060</td><td>0.89</td><td>1.15</td><td>1.11</td></tr><tr><td>gated  $\ell \stackrel { \triangledown } { \ \stackrel { \triangledown } { \sim } } - 0 . 2 1$ </td><td>0.978</td><td>0.060</td><td>1.11</td><td>1.20</td><td>1.14</td></tr><tr><td>gated  $\ell \lesssim - 0 . 4 6$ </td><td>0.992</td><td>0.082</td><td>0.90</td><td>1.33</td><td>1.37</td></tr><tr><td>no resampling</td><td>0.964</td><td>0.092</td><td>1.13</td><td>1.23</td><td>0.84</td></tr><tr><td>resampling</td><td>0.979</td><td>0.093</td><td>1.02</td><td>1.09</td><td>1.09</td></tr><tr><td>resampling + rejuv. (M = 3)  $x _ { 0 } > 8$ </td><td>0.995</td><td>0.056</td><td>1.01</td><td></td><td>0.93</td><td>0.78</td></tr><tr><td>gated</td><td> $\ell \lesssim 0 . 0 \dot { 5 }$ </td><td>0.990</td><td>0.059</td><td>0.93</td><td>0.83</td><td>0.92</td></tr><tr><td>gated</td><td> $\ell \stackrel { \triangledown } { \  } - 0 . 2 1$ </td><td>0.985</td><td>0.064</td><td>0.97</td><td>1.00</td><td>0.73</td></tr><tr><td></td><td>gated  $\ell \lesssim - 0 . 4 6$ </td><td>0.971</td><td>0.086</td><td>1.00</td><td>1.14</td><td>1.53</td></tr><tr><td rowspan="8"> $x _ { 0 } > 9$ </td><td>no resampling</td><td>0.983</td><td>0.127</td><td>0.94</td><td>0.92</td><td>0.73</td></tr><tr><td>resampling</td><td>0.972</td><td>0.159</td><td>0.92</td><td>1.24</td><td>1.33</td></tr><tr><td>resampling + rejuv. (M = 3)</td><td>0.984</td><td>0.068</td><td>0.96</td><td></td><td></td></tr><tr><td>gated  $\ell \lesssim 0 . 0 \dot { 5 }$ </td><td>0.985</td><td>0.073</td><td>1.10</td><td>1.12</td><td>0.92</td></tr><tr><td> $\ell \lesssim - 0 . 2 1$ </td><td>0.980</td><td>0.089</td><td>0.77</td><td>1.08 0.83</td><td>0.84</td></tr><tr><td>gated gated  $\ell \lesssim - 0 . 4 6$ </td><td>0.978</td><td>0.109</td><td>0.99</td><td>1.53</td><td>0.98 1.74</td></tr><tr><td>no resampling</td><td>1.020</td><td>0.213</td><td>1.16</td><td></td><td></td></tr><tr><td>resampling</td><td>0.986</td><td>0.166</td><td>0.75</td><td>1.13 1.74</td><td>1.85</td></tr><tr><td rowspan="5"> $x _ { 0 } > 1 0 $ </td><td>resampling + rejuv. (M = 3)</td><td></td><td></td><td></td><td></td><td>3.90</td></tr><tr><td></td><td>0.994</td><td>0.085</td><td>1.48</td><td>1.59</td><td>2.10</td></tr><tr><td>gated  $\ell \stackrel { - } { \sim } 0 . 0 \dot { 5 }$ </td><td>0.981</td><td>0.090</td><td>0.97</td><td>0.94</td><td>1.45</td></tr><tr><td>gated  $\ell \lesssim - 0 . 2 1$ </td><td>0.965</td><td>0.079</td><td>1.23</td><td>1.07</td><td>0.98</td></tr><tr><td>gated  $\ell \lesssim - 0 . 4 6$ </td><td>0.956</td><td>0.119</td><td>0.92</td><td>1.21</td><td>1.05</td></tr></table>

![](images/bfee85dcf8bfb9342130085212f69ce9489c405ea8597db530242cb79e0908a9.jpg)

![](images/a7589cc2c53b4dd8e597f7c16aa25f743cfeb51ff3d07138244c3523ff2a0e1c.jpg)  
Figure 12: (Left) Magnitude of the antisymmetric part of the Jacobian of the score network, measured as a Frobenius norm for 1024 particles for the different threshold exceedance problems of Table 9. $( \mathrm { R i g h t } )$ Same value as the left multiplied by $\rho$ for $\tau > 1 0$ threshold exceedance problem, for different calibrated step sizes $\rho ,$ denoted by their average acceptance rate. The shaded band marks the range of gates tested and the dashed line the adopted gate.

for all triples $( a , b , c )$ . This means that the acceptance ratio of the transition: $a  b  c  a$ must be equal to 1 for MADM kernel to equate to MALA. A non-conservative score does not fulfil this criterion (Vuong et al., 2025) and hence for a learned score, MADM need not be invariant to $\pi _ { k }$

To account for this, we turn off rejuvenation at a specific threshold in time or $\ell _ { t }$ when the nonconservativity of the score increases. Table 9 ablates different thresholds. $\mathbf { A t } \boldsymbol { \tau } = 1 0$ the ungated move reads KS 1.48/1.59 and EMD 2.10 against $1 . 2 3 / 1 . 0 7 / 0 . 9 8$ when gated at $t \geq 0 . 3 0$ , and the same ordering holds at $\tau = 9 ;$ we therefore adopt the gated rejuvenation for the rest of our examples, selecting the middle of the values we tested, $\ell _ { t } \stackrel { \cdot } { \le } - 0 . \bar { 2 } 1 ( t \ge 0 . 3 0 ) . \mathrm { A t } \tau = 7 , 8 ,$ where the corrector and resampling events were triggered less frequently, the gated and ungated rows coincide within noise, as the ungated version already sits close to the noise floor. Both gated and un-gated arms at $\tau = 1 0$ read 0.965 and 0.994 within 1.3 standard errors.

Step size selection As the step-size $\rho$ defines the region where a move is proposed, i.e. $\tilde { x } - x \sim$ $\mathcal { N } ( \rho g ( x ) / 2 , \rho I )$ of Eq. (114), a larger $\rho$ leads to, what we hypothesise, a higher exposure to the non-conservative part of the score network, which affects the estimated $M _ { k }$ further. For a simple visualisation, we plot the proposed step size $\rho$ multiplied by the Frobenius norm of the antisymmetric part of the Jacobian, to the right of Fig 12. This for various $\rho$ on the $\tau = 1 0$ threshold exceedance problem of Section 4.1; here the step size has been tuned to target a range of acceptance probabilities as is common (Lam et al., 2026). We elect to take smaller, more conservative steps, tuning $\rho$ to target a 0.72 acceptance.

We hypothesise that, for a fixed number of MADM steps and $\rho ,$ there is a tradeoff between the total number of MADM steps that rejuvenate duplicated particles and the net error injected into the system by an approximate MH ratio. An adaptive schedule on $\rho ( t )$ or $M ( t )$ that accounts for these trade-offs could be an interesting direction for future work.

Ablation of Errors due to MADM In this setting we ablate what the discretisation error is at different K using MADM. As MADM targets the continuous-time marginal $p _ { t }$ instead of the discretised $\gamma _ { 0 }$ law, it incurs a discretisation bias. We find the same issue while not plotted here, exists with MALA suggesting it is not a quadrature error itself. We verify this using analytical scores and the variance minimising Doob’s guidance. See Fig. 13.

![](images/de70977d5b430a9172635251c7ecdd18443108ba5b25879d9d3b7cb628712ed6.jpg)  
Figure 13: Bias introduced by MADM for the two integrators, using analytical scores and Doob’s guidance with $M = 3$ for the first two panels. Left & Middle Injected bias affects rarer samples more, with all biases decreasing as $K  \infty$ . The two integrators have a different error profile. Right Bias plotted for the $\tau = 1 0$ threshold exceedance problem. We find that the error grows with M. Error bars are due to seed variability.

## G.4 BOX AND MULTIPLE BOX EVENTS

We test the guidance on boxed geometries, each at two rarities, plus a bounded family. This is an analogue of an extreme defined as a set where variables exist in some range. Both families are boxes in the observable, i.e., $E = \{ x : a \leq \phi ( x ) \leq b \}$ with $a , b \in ( \mathbb { R } \cup \{ \pm \infty \} ) ^ { \bar { m } }$ and the inequalities read component-wise. We take the interval guidance of Eq. (96), see App. E. Any corner boxes that we use, takes $b = + \infty$ , and it represents extremes that are joint threshold exceedances; bounded boxes have finite a and b.

For simplicity we have illustrated the box boundaries we test in Fig 14 overlaid against the true unconditional density of the analytical 2D GMM. We report performance on the learned variant in Table 10, where we ablate the effect of no resampling, resampling with gated/un-gated rejuvenation. We find that similar to the other experiments, rejuvenation lowers the CV on the rare boxes $( p ^ { \star } \leq$

![](images/c337a8c9e4dd0544628bff88f9959ec0f1ba6e70cf377929e337807759fa6d72.jpg)  
Figure 14: Illustrative example of box boundaries. Plotted against a backdrop of the unconditional density of the 2D-GMM. Quantitative results of rare-event sampling are given in Table 10 below.

$2 \times 1 0 ^ { - 4 }$ , by $1 . 5 \substack { - 2 . 5 \times }$ and is neutral or slightly worse on the easy ones $( p ^ { \star } \geq 1 0 ^ { - 2 } )$ , where resampling is triggered at most twice per run.

Table 10: Box events on the learned score. Every region is drawn on the model density in Fig. 14. $p ^ { \star }$ is the $K = 1 0 0 0$ chain’s own tail (Table 5 pool). We compare the estimate $\hat { p }$ and the shape metrics of the final particles, presented as a ratio against a noise floor.
<table><tr><td>event</td><td>variant</td><td> $p ^ { \star }$ </td><td> $\hat { p } / { p ^ { \star } }$ </td><td>CV↓</td><td> ${ \mathrm { K S } } _ { 0 }$ </td><td> ${ \mathrm { K S } } _ { 1 }$ </td><td>EMD</td></tr><tr><td rowspan="4"> $[ 6 , \infty ) \times [ 1 . 5 , \infty )$ </td><td>no resampling resampling</td><td rowspan="4"> $9 . 9 7 \times 1 0 ^ { - 3 }$ </td><td>0.955 1.004</td><td>0.066 0.107</td><td>0.95 1.37</td><td>0.98 0.87</td><td>1.01 0.98</td></tr><tr><td></td><td>1.000</td><td>0.070</td><td>0.89</td><td>0.91</td><td></td></tr><tr><td rowspan="2">resampling + rejuv. (M = 3) gated  $\ell \lesssim - 0 . 2 1$ </td><td></td><td></td><td></td><td></td><td>0.96</td></tr><tr><td>1.000</td><td>0.070</td><td>0.94</td><td>0.89</td><td>1.00</td></tr><tr><td rowspan="4"> $[ 6 , \infty ) \times [ 1 . 5 , \infty )$   $\cup [ 6 , \infty ) \times ( - \infty , - 1 . 5 ]$ </td><td rowspan="2">no resampling resampling</td><td rowspan="2"> $2 . 6 0 \times 1 0 ^ { - 2 }$ </td><td>0.966 0.979</td><td>0.057 0.055</td><td>1.15 0.99</td><td>1.15</td><td>1.09</td></tr><tr><td>0.988</td><td>0.042</td><td></td><td>1.08</td><td>0.95</td></tr><tr><td rowspan="2">resampling + rejuv. (M = 3) gated  $\ell \stackrel { - } { \sim } - 0 . 2 1$ </td><td rowspan="2"></td><td>0.992</td><td>0.044</td><td>0.87 0.96</td><td>1.10 1.44</td><td>1.15 1.50</td></tr><tr><td>0.956</td><td>0.063</td><td>1.10</td><td>0.85</td><td>0.82</td></tr><tr><td rowspan="4"> $[ 6 , \infty ) \times [ 1 . 5 , \infty )$   $\cup [ 8 , \infty ) \times ( - \infty , - 3 ]$ </td><td rowspan="2">no resampling resampling resampling + rejuv. (M = 3)</td><td rowspan="4"> $1 . 0 8 \times 1 0 ^ { - 2 }$ </td><td>0.992</td><td>0.086</td><td>0.90</td><td>1.27</td><td>1.15</td></tr><tr><td>1.011</td><td>0.089</td><td>0.94</td><td>1.03</td><td>1.36</td></tr><tr><td rowspan="2"> $\mathrm { g a t e d } \ell \lesssim - 0 . 2 1$ </td><td>1.011</td><td>0.089</td><td>1.02</td><td>1.10</td><td>1.07</td></tr><tr><td>0.945</td><td>0.132</td><td>1.21</td><td>0.96</td><td>0.90</td></tr><tr><td rowspan="4"> $[ 8 . 5 , \infty ) \times [ 1 . 5 , \infty )$ </td><td rowspan="2">no resampling resampling resampling + rejuv. (M = 3)</td><td rowspan="2"> $2 . 1 6 \times 1 0 ^ { - 4 }$ </td><td>0.967</td><td>0.116</td><td>1.19</td><td>0.86</td><td>0.96</td></tr><tr><td>0.979 0.982</td><td>0.086</td><td>1.07</td><td>0.90</td><td>0.96</td></tr><tr><td rowspan="2">gated  $\ell \lesssim - 0 . 2 1$ </td><td></td><td>0.105</td><td></td><td>1.17</td><td>0.91</td><td>1.02</td></tr><tr><td>no resampling</td><td>0.986</td><td>0.123</td><td>1.15</td><td>1.32</td><td>0.85</td></tr><tr><td rowspan="4"> $[ 9 , \infty ) \times [ 1 . 5 , \infty )$   $\cup [ 9 , \infty ) \times ( - \infty , - 1 . 5 ]$ </td><td rowspan="2">resampling resampling + rejuv. (M = 3) gated  $\ell \stackrel { - } { \sim } - 0 . 2 1$ </td><td rowspan="4"> $1 . 8 7 \times 1 0 ^ { - 4 }$ </td><td>0.990</td><td>0.133 0.079</td><td>1.10</td><td>1.17</td><td>1.28</td></tr><tr><td>1.008</td><td></td><td>1.22</td><td>1.34</td><td>1.12</td></tr><tr><td rowspan="2">no resampling</td><td>1.008</td><td>0.113</td><td>1.02</td><td>1.03</td><td>0.96</td></tr><tr><td>0.972</td><td>0.164</td><td>1.11</td><td>1.21</td><td>1.26</td></tr><tr><td rowspan="4"> $\cup [ 9 . 5 , \infty ) \times ( - \infty , - 3 ]$ </td><td rowspan="2">resampling resampling + rejuv. (M = 3)</td><td rowspan="2"> $1 . 1 2 \times 1 0 ^ { - 4 }$ </td><td>0.972 0.998</td><td>0.133 0.066</td><td>1.18 1.15</td><td>1.83</td><td>1.97</td></tr><tr><td>1.010</td><td>0.088</td><td>1.06</td><td>1.00 1.00</td><td>1.29 0.82</td></tr><tr><td rowspan="2">gated  $\ell \lesssim - 0 . 2 1$  no resampling</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.968 0.987</td><td>0.032 0.050</td><td>1.04 1.18</td><td>1.16</td><td>1.13</td></tr><tr><td rowspan="4"> $[ 4 , 6 ] \times [ - 1 , 1 ]$ </td><td rowspan="2">resampling resampling + rejuv. (M = 3) gated  $\ell \lesssim - 0 . 2 1$ </td><td rowspan="4"> $8 . 6 6 \times 1 0 ^ { - 2 }$ </td><td>0.986</td><td>0.035</td><td>1.19</td><td>1.01 0.96</td><td>1.08 1.09</td></tr><tr><td>0.992</td><td>0.035</td><td>1.04</td><td>0.89</td><td>0.98</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>no resampling resampling</td><td>0.951 0.996</td><td>0.067 0.092</td><td>1.06</td><td>1.11</td><td>1.07</td></tr><tr><td rowspan="4"> $[ 8 , 1 1 ] \times [ 6 , 9 ]$ </td><td rowspan="4">resampling + rejuv. (M = 3) gated  $\ell \lesssim - 0 . 2 1$ </td><td rowspan="4"> $4 . 1 1 \times 1 0 ^ { - 4 }$ </td><td></td><td></td><td>1.02</td><td>1.06</td><td>0.96</td></tr><tr><td>0.984</td><td>0.052</td><td>0.96</td><td>1.15</td><td>1.04</td></tr><tr><td>0.983</td><td>0.051</td><td>0.97</td><td>1.10</td><td>0.98</td></tr><tr><td></td><td>0.964 0.057</td><td>0.81</td><td>1.41</td><td>1.35</td></tr><tr><td rowspan="2"> $[ 8 , 1 1 ] \times [ 6 , 9 ]$   $\cup [ 8 , 1 1 ] \times [ - 9 , - 6 ]$ </td><td>no resampling resampling</td><td rowspan="2"> $1 . 0 1 \times 1 0 ^ { - 3 }$ </td><td>0.984</td><td>0.056</td><td>0.81</td><td>0.84</td><td>1.09</td></tr><tr><td>resampling + rejuv. (M = 3)  $\ell \stackrel { - } { \sim } - 0 . 2 1$ </td><td>0.986 0.984</td><td>0.050 0.053</td><td>0.86 0.89</td><td>1.12 0.90</td><td>1.57 1.26</td></tr></table>

## G.5 ABLATION OF PARAMETERS

We ablate our parameters once: one the $\tau = 1 0$ threshold exceedance problem and we reuse it throughout our experiments. There are two parameters we ablate over the starting point of our guidance $t _ { \mathrm { o n } }$ and the exponential p in $\lambda ( t ) = 1 - ( t / t _ { \mathrm { o n } } ) ^ { p }$ . Below are the results of the ablation for our method, separated by the base, resampling and resampling with rejuvenation; we also ablate for the lowest CV of Manshausen et al. (2026) which we use for the experiments in Fig. 2. All experiments were done with 24 seeds, $N = 1 0 0 0$ $K = 1 0 0 0$ using the SEEDS-1 integrator (Gonzalez et al., 2023), with $M = 3$ steps of MADM (Lam et al., 2026) after resampling.

Table 11: Schedule ablation at $\tau = 1 0 , K = 1 0 0 0 , N = 1 0 0 0 \colon$ CV of pˆ for the tempering schedule $\lambda ( t ) = 1 - ( t / t _ { \mathrm { o n } } ) ^ { p }$ . Bold marks the minimum of each column. We find that the no-rejuvenation variants converged on the same tempering schedule $\lambda _ { t } \equiv 1$ compared to the MADM variant. All schedules with rejuvenation yield a lower CV compared to without.
<table><tr><td>schedule</td><td>base ↓</td><td>+ resampling↓</td><td>+ MADM↓</td><td>(Manshausen et al., 2026) ↓</td></tr><tr><td>analytic score</td><td></td><td></td><td></td><td></td></tr><tr><td> $\lambda \equiv 1$ </td><td>0.15</td><td>0.23</td><td>0.09</td><td>0.29</td></tr><tr><td> $t _ { \mathrm { o n } } = 1 . 0 , p = 2$ </td><td>0.28</td><td>0.32</td><td>0.08</td><td>0.68</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 9 , p = 2 ( \mathrm { d e p l o y e d } )$ </td><td>0.35</td><td>0.33</td><td>0.07</td><td>0.83</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 7 , p = 2$ </td><td>0.62</td><td>0.40</td><td>0.10</td><td>1.26</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 5 , \ p = 2$ </td><td>1.73</td><td>0.91</td><td>0.13</td><td>1.94</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 9 , \ p = 1$ </td><td>0.71</td><td>0.48</td><td>0.11</td><td>1.45</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 9 , \ p = 4$ </td><td>0.20</td><td>0.23</td><td>0.10</td><td>0.56</td></tr><tr><td>learned score</td><td></td><td></td><td></td><td></td></tr><tr><td>λ≡ 1</td><td>0.21</td><td>0.17</td><td>0.11</td><td>0.17</td></tr><tr><td> $t _ { \mathrm { o n } } = 1 . 0 , p = 2$ </td><td>0.32</td><td>0.26</td><td>0.13</td><td>0.51</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 9 , p = 2 ( \mathrm { d e p l o y e d } )$ </td><td>0.40</td><td>0.29</td><td>0.08</td><td>0.68</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 7 , p = 2$ </td><td>0.70</td><td>0.36</td><td>0.15</td><td>1.18</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 5 , \ p = 2$ </td><td>2.00</td><td>0.75</td><td>0.25</td><td>1.90</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 9 , \ p = 1$ </td><td>0.78</td><td>0.37</td><td>0.18</td><td>1.28</td></tr><tr><td> $t _ { \mathrm { o n } } = 0 . 9 , \ p = 4$ </td><td>0.23</td><td>0.18</td><td>0.16</td><td>0.45</td></tr></table>

## H CLIMATE EXPERIMENTS

## H.1 SETUP AND MODEL DETAILS

We use a pretrained score-based diffusion climate emulator that acts as a surrogate for physics-based Earth System Models (Bouabid et al., 2026) (licensed under MIT Licence), trained on large-ensemble runs from the Coupled Model Intercomparison Project (CMIP6) on different climate change (forcing) scenarios (O’Neill et al., 2016). Our results are based on the MPI-ESM1-2-LR emulator. The model outputs monthly-averaged fields conditioned on the global mean surface temperature anomaly (GMST), which captures global warming against preindustrial warming, and on the month of the year, which captures annual variability.

The emulator generates four surface fields jointly across the globe on every draw, listed in Table 12 by their CMIP6 short names. All fields are monthly means and are handled throughout as anomalies against the pre-industrial (piControl) climatology, matching the convention of Table 13.

Table 12: Emulated variables with their corresponding CMIP6 short names, descriptions and units.
<table><tr><td>Short name</td><td>Description</td><td>Units</td></tr><tr><td>tas</td><td>2 m air temperature</td><td>K</td></tr><tr><td>hurs</td><td>2 m relative humidity</td><td>%</td></tr><tr><td>pr</td><td>Precipitation rate</td><td> $\mathrm { m m d a y } ^ { - 1 }$ </td></tr><tr><td>sfcWind</td><td>10 m wind speed</td><td> $\mathrm { m } \mathrm { s } ^ { - 1 }$ </td></tr></table>

We compute a large Monte Carlo average of the rare event probability with $N = 9 , 0 1 1 , 2 0 0$ unguided draws at a GMST of $\Delta T = 1 . 5 $ K for the month of July to validate our results against. For the region R, with $\phi ( \cdot )$ as a per-variable regional average, the associated mean, standard deviation and tail values at different rarities of interest are given in Table 13.

## H.1.1 IMPLEMENTATION DETAILS

For all experiments throughout we use $K = 2 0 0$ discretisation of the exponential integrator in the noise-prediction mode, see App. D and Fig. 15. We schedule λ(t) and $\delta ( t )$ for all experiments below. For a VE-schedule, $\alpha _ { t } \equiv 1$ , so the half-log SNR is $\ell _ { t } = - \log \bar { \sigma } _ { t }$ and we specify both schedule in $\bar { \sigma } _ { t } \colon$ piecewise linear in λ and $\delta _ { t }$ is interpolated log-linearly between two endpoints. Similar to

Table 13: Tail thresholds for the four emulator variables, from $N = 9 , 0 1 1 , 2 0 0$ unguided Monte Carlo draws over R at $\Delta T = + 1 . 5$ K in July. Upper-tail entries satisfy $P ( \phi \geq \tau ) = p .$ , lowertail entries $P ( \phi \leq \tau ) = p$ . Binomial relative standard errors are 1.1%, 3.3% and 10.5% at $p =$ $1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 \dot { 0 } ^ { - 5 }$ respectively. The guided runs use the same thresholds.
<table><tr><td></td><td></td><td colspan="2">Marginal</td><td colspan="3">Upper tail</td><td colspan="3">Lower tail</td></tr><tr><td>Variable</td><td>Unit</td><td>Mean</td><td>SD</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>tas</td><td>K</td><td>+2.577</td><td>1.023</td><td>5.813</td><td>6.498</td><td>7.096</td><td>-0.539</td><td>-1.174</td><td>-1.727</td></tr><tr><td>pr</td><td>mm/day</td><td>-0.250</td><td>0.450</td><td>1.698</td><td>2.311</td><td>2.842</td><td>-1.334</td><td>-1.517</td><td>-1.650</td></tr><tr><td>hurs</td><td>%</td><td>-3.275</td><td>4.220</td><td>10.519</td><td>13.485</td><td>16.054</td><td>-15.838</td><td>-18.387</td><td>-20.554</td></tr><tr><td>sfcWind</td><td>m/s</td><td>+0.193</td><td>0.186</td><td>0.773</td><td>0.895</td><td>1.004</td><td>-0.387</td><td>-0.508</td><td>-0.613</td></tr></table>

GMM example, we turn guidance, i.e. $\lambda ( t ) \equiv 0$ off for the first 15% $( \bar { \sigma } > 4 0 )$ of the denoising process to account for any approximation errors in $\hat { x } _ { 0 \mid t } .$ , converging to $\lambda ( \bar { \sigma } )  1$ after adding a lambda overshoot at a maximum of 1.25. δ(t) is held at 0.5 until $\bar { \sigma } = 0 . 3 \mathrm { S D }$ and sharpens to 0.01 by $\bar { \sigma } = 0 . 0 3 \mathrm { S D } .$ , where SD is the standard deviation of each observable approximated by the same method as the ΠGDM mentioned in App B. We decrease $\delta$ towards the end of the denoising process to reduce the variance of our estimator due to mismatch in the generated $\pi _ { 0 } .$ For simplicity, we do not anneal $\tau ( \bar { \sigma } )$

Correctness of our method is independent of these choices, so they are selected for variance alone. We set them empirically on the $p ^ { * } \approx 1 0 ^ { - 3 }$ rarity, PNW marginal hot and dry threshold exceedance problems using a pilot sweep at the same setup as described above. We provide an ablation of the values we tested below (Table 22), but adopt the same hyper-parameters for all experiments, with $N = 1 0 2 4 , K = 2 0 0$ and 16 seeds throughout, see Table 14.

![](images/97763d8f34b4cef8925bc20dc5d9ecae7ee715802496bda85f9f9a63ed5999e0.jpg)  
Figure 15: Discretisation bias for the integrator for the $\tau _ { 1 }$ PNW heatwave event using the SEEDS-1 integrator, with 1024 samples at each point and without any rejuvenation. Dotted line represents the reference pool. We find that for SEEDS-1 and Euler-Maruyama integrators using $K = 2 0 0$ saturates the discretisation bias. We could not compare against a large K unconditional reference due to prohibitive costs. Bars represent ±1 standard error of the mean over 8 shared seeds across both integrators.

All experiments were done on a single Nvidia H100 GPU with 96 GB of VRAM. We used approximately 370 GPUh to generate the 9 million reference samples. A single batch of 1024 unconditional draws at $K = 2 0 0$ steps takes approximately 155 seconds, with the method and its variants multiplying this cost; we summarise this in Table 21.

Table 14: Deployed schedule and sampler configuration for the climate emulator experiments. $\bar { \sigma }$ is the marginal noise standard deviation. Selected once on the marginal $1 0 ^ { - 3 }$ PNW hot and dry events and applied unchanged to other experiments. <sup>‡</sup>δ is quoted in units of the observable’s unconditional standard deviation.
<table><tr><td></td><td>Symbol</td><td>Value</td></tr><tr><td>Schedules</td><td></td><td></td></tr><tr><td>Tempering onset</td><td> $\bar { \sigma } _ { \mathrm { o n } }$ </td><td>40</td></tr><tr><td>Tempering peak</td><td> $\lambda _ { \mathrm { m a x } }$ </td><td> $1 . 2 5 \mathrm { a t } \bar { \sigma } = 2$ </td></tr><tr><td>Tempering relaxation</td><td> $\bar { \sigma } _ { \mathrm { e n d } }$ </td><td>0.3</td></tr><tr><td>Smoothing width, transport</td><td> $\delta _ { \mathrm { h i } }$ </td><td>0.5 SD</td></tr><tr><td>Smoothing width, floor</td><td> $\delta _ { \mathrm { l o } }$ </td><td>0.01 SD</td></tr><tr><td>Ramp window</td><td></td><td> $\bar { \sigma } \in [ 0 . 0 3 , 0 . 3 ]$ </td></tr><tr><td>Ramp exponent</td><td></td><td> $\log _ { 1 0 } 5 0 \approx 1 . 7 0$ </td></tr><tr><td>Observable SD, frozen‡ (tas, hurs)</td><td>SD</td><td> $1 . 0 1 7 \mathrm { K } , 4 . 2 2 4 \%$ </td></tr><tr><td>Threshold</td><td>T</td><td>constant</td></tr><tr><td>Sampler</td><td></td><td></td></tr><tr><td>Particles</td><td>N</td><td>1024</td></tr><tr><td>Integration steps</td><td>K</td><td>200</td></tr><tr><td>Integrator</td><td></td><td>exponential (noise-prediction)</td></tr><tr><td>Resampling</td><td></td><td>adaptive, ESS/N &lt; 0.5</td></tr><tr><td>Rejuvenation</td><td>M</td><td>0</td></tr><tr><td>Noise range</td><td> $[ \bar { \sigma } _ { \operatorname* { m i n } } , \bar { \sigma } _ { \operatorname* { m a x } } ]$ </td><td> $[ 0 . 0 1 , 1 4 6 . 0 9 ]$ </td></tr><tr><td>Posterior proxy</td><td> $s _ { t } ^ { 2 }$ </td><td> $\bar { \sigma } ^ { 2 } / ( 1 + \bar { \sigma } ^ { 2 } )$ </td></tr><tr><td>Look-ahead bandwidth</td><td> $\delta _ { \mathrm { e f f } } ^ { 2 }$ </td><td> $\delta ( \dot { \bar { \sigma } } ) ^ { 2 } + s _ { t } ^ { 2 } \lVert \boldsymbol { a } \rVert ^ { 2 }$ </td></tr></table>

Table 15: Marginal extreme-event probabilities over the PNW target region in July at $\Delta T = + 1 . 5 \mathrm { K }$ for hot $( \phi _ { \tt t a s } \geq \tau )$ and dry $( \phi _ { \mathrm { { h u r s } } } \leq \tau )$ events. For each variable, $\tau _ { 1 } , \tau _ { 2 } , \tau _ { 3 }$ are chosen so that the event is successively rarer (exact values in App. H). Ground-truth $p ^ { \star }$ are computed with $\sim 1 0 ^ { 7 }$ MC samples. Unguided Monte Carlo and our guided method use the same computational budget: each unguided estimate uses 4813 draws (4.7× the 1024 guided particles, Table 21), over 16 disjoint replicates. Estimated probabilities, $\hat { p }$ are reported as a ratio $\hat { p } / p ^ { * }$ with mean ±1 standard deviation over 16 seeds. The variance ratio (VR) is $\mathrm { V a r } ( \hat { p } _ { \mathrm { u n g u i d e d } } ) / \mathrm { \bar { V a r } } ( \hat { p } _ { \mathrm { g u i d e d } } )$ . The number of unguided replicates with no event is 1 (hot) and 0 (dry) of 16 at $\tau _ { 1 } , 9$ and 12 at $\tau _ { 2 }$ , and 14 for both at $\tau _ { 3 } ,$ , where the other two replicates record a single event each, so the VR at $\tau _ { 3 }$ is a rough estimate. <sup>†</sup>Includes one collapsed run (App. H.2.1). The hot results are shown in Fig. 3.
<table><tr><td rowspan="3"></td><td colspan="4">Hot:  $\phi _ { \mathrm { t a s } } ( x ) \geq \tau$ </td><td colspan="4">Dry:  $\phi _ { \mathrm { h u r s } } ( x ) \leq \tau$ </td></tr><tr><td rowspan="2"> $p ^ { \star }$ </td><td colspan="2"> $\hat { p } / p ^ { \star }$ </td><td rowspan="2"></td><td rowspan="2"> $p ^ { \star }$ </td><td colspan="2"> $\hat { p } / p ^ { \star }$ </td><td rowspan="2">VR</td></tr><tr><td>Unguided</td><td>Guided</td><td>Unguided</td><td>Guided</td></tr><tr><td> $\tau _ { 1 }$ </td><td>1.00e-3</td><td> $0 . 9 0 \pm 0 . 4 5$ </td><td> ${ \bf 0 . 9 8 \pm 0 . 0 8 }$ </td><td>34×</td><td>1.00e-3</td><td> $1 . 0 3 \pm 0 . 4 4$ </td><td> ${ \bf 1 . 0 0 \pm 0 . 0 7 }$ </td><td>37×</td></tr><tr><td> $\tau _ { 2 }$ </td><td>1.00e-4</td><td> $1 . 3 0 \pm 1 . 6 7$ </td><td> ${ \bf 1 . 0 1 \pm 0 . 0 8 }$ </td><td> $4 1 1 \times$ </td><td>1.00e-4</td><td> $0 . 5 2 \pm 0 . 9 3$ </td><td> $\mathbf { 1 . 0 2 \pm 0 . 2 2 } ^ { \dagger }$ </td><td>18×</td></tr><tr><td> $\tau _ { 3 }$ </td><td>1.01e-5</td><td> $2 . 5 7 \pm 7 . 0 3$ </td><td> ${ \bf 1 . 0 5 \pm 0 . 1 3 }$ </td><td>2722×</td><td> $1 . 0 1 \mathrm { e } { - 5 }$ </td><td> $2 . 5 7 \pm 7 . 0 3$ </td><td> ${ \bf 1 . 0 6 \pm 0 . 1 2 }$ </td><td>3567×</td></tr></table>

## H.2 EXPERIMENT 1: COMPOUND HOT & DRY EVENTS

This subsection supports the first climate experiment of Section 4.2: the ablation behind the deployed recipe, and the fields the guided sampler produces at each rarity. We begin with the marginal threshold exceedance problem, with the hot (tas) and dry (hurs) marginal results reported in Table 15, followed by the joint.

## H.2.1 SELECTION AND REJUVENATION

Similar to Appendix G.3.3, we ablate the effect of resampling and rejuvenation on the hot and dry threshold exceedance events and the joint hot-and-dry event that we report the net speed-up on. This is because while resampling does not cost extra wall clock time, rejuvenation does. We evaluate whether the reduction in CV justifies the added computation cost. For this, we use a fixed $M = 5 { \bf M A D M }$ steps using 2 degrees of quadrature. Additionally, we cache the estimate of log $\Gamma _ { \pi }$ in Eq. (119) for either the current x or proposal x˜ so that it may be reused between moves. For the same reasons as highlighted in Appendix ${ \bf G }$ , we set the acceptance target at 0.72 and gate the rejuvenation move to run only at noise levels $\bar { \sigma } \geq 0 . 3 0$

Table 16 captures this ablation. We report the CV which is the spread of the estimated $\hat { p }$ across 16 seeds; <sup>ˆ</sup>k is the generalised-Pareto shape parameter of the importance weights, fitted with the top 96 (min $( 0 . 2 N , 3 \sqrt { N } ) )$ (Vehtari et al., 2024), within a run, reported as the median over seeds. We find that without resampling the median $\hat { k }$ exceeds 0.5 (no finite second moment) in seven of the nine cells. This suggests that the weights are very heavy-tailed and the convergence of the no-resampling estimator is slow to converge, with an accompanying CV that is optimistic.

Resampling lowers the CV in eight of the nine cells; the exception, dry at $1 0 ^ { - 4 }$ , comes from a single collapsed run in which one lineage takes over the crossing particles (seed 12, $\hat { p } / p ^ { \star } = 1 . 7 2$ with 5 effective families); without it the CV is 0.124. Rejuvenation does not change the CV consistently: it is lower for hot at $1 0 ^ { - 5 }$ and for dry at $1 0 ^ { - 4 }$ , where no rejuvenated run collapses, and higher in the other seven cells, by at most $2 7 \%$ . Differences of this size are within the sampling error of a CV estimated from 16 seeds. $W _ { 1 }$ is the weighted Wasserstein-1 distance of the quantity of interest ϕ(·) between the generated samples and the Monte Carlo reference, reported against a null floor. All variants report a similar $W _ { 1 }$ and there is no discernible issue with any. However, it is worth noting that rejuvenation adds ${ \sim } 1 . 1 { - } 1 . 2 5 { \times }$ wall-clock time compared to resampling alone, hence there is a trade-off between a larger N versus more rejuvenation steps at finite compute budget. In the experiments below, unless stated otherwise we run with zero rejuvenation steps.

## H.2.2 CONDITIONAL FIELDS FOR TEMPERATURE EVENTS

We plot both the local (Fig. 16) and global (Fig. 17) guided fields for the threshold exceedance problem to visualise representative draws from the conditional model obtained using guidance. In the example below, guidance is only applied to the region R for the tas variable, any structure or perturbation of other variables or away from R is due to the unguided score.

![](images/6ab6f54a4fa8977946f39fff3bf7bbc6dedfc6ddd0671356b17658b0d818a514.jpg)  
Figure 16: Top-weight crossing fields over the Pacific Northwest (PNW) target region produced by the guided sampler at different exceedance threshold τ . Guided samples based on the tas variable alone. Same setup as reference Table 15. p denotes $p _ { 0 } [ E ]$ and ϕ(·) is the regional average temperature.

![](images/49e48b40eec023723172d18a9078941e077c65515db69918e9e7bb7103e613d3.jpg)  
Figure 17: The same fields as Fig. 16 on the global domain.

Table 16: Ablation of resampling and rejuvenation on the climate emulator for the hot, dry, and joint hot-and-dry events. Same parameters as in the main text; details in App. H.2.1. Ratios are mean ± one standard error over 16 seeds. $W _ { 1 }$ is a weighted distance between the $\phi ( \cdot )$ of generated samples and draws from a reference, scaled against a null floor. The relative error of $p ^ { \star }$ is 1.1/3.3/10.5% at the three marginal rarities and $0 . 6 / 2 . 3 / 8 . 9 \%$ at the three joint rarities. CV is estimated across $\hat { p }$ estimates; <sup>ˆ</sup>k is estimated per seed and we report the median over the 16 seeds. <sup>†</sup>Includes one collapsed run; without it $\hat { p } / p ^ { \star } = 0 . 9 6 8$ and CV = 0.124. $\hat { k } > 0 . 7$ represents weights that have no finite second moment (Vehtari et al., 2024), so the CV estimates, and the speed-ups of those rows, are optimistic. Speed-up is over crude MC at matched CV, including each arm’s measured wall-clock (App. H.4); in brackets, its 95% bootstrap CI (BCa, 20,000 resamples of the 16 seeds).
<table><tr><td rowspan="2"> $p ^ { \star }$ </td><td rowspan="2">arm</td><td colspan="4">p/p* CV↓ ESS/N</td><td colspan="2">k  $W _ { 1 }$ </td><td rowspan="2">Speed-up ↑ (95% CI)</td></tr><tr><td></td><td></td><td></td><td>tas</td><td>hurs</td><td></td></tr><tr><td></td><td> $1 0 ^ { - 3 }$ </td><td>no resampling</td><td>1.024 ± 0.051</td><td>0.199</td><td>0.06</td><td>0.49 1.03</td><td></td><td></td><td>5.2× (2.8–12×)</td></tr><tr><td rowspan="5"> $1 0 ^ { - 4 }$  Ht</td><td rowspan="5"></td><td>adaptive resampling</td><td>0.982 ± 0.019</td><td>0.078</td><td></td><td>1.54</td><td></td><td></td><td>34× (20–60×)</td></tr><tr><td>+ MADM moves</td><td>0.954 ± 0.020</td><td>0.082</td><td></td><td></td><td>1.40</td><td></td><td>27× (17–44×)</td></tr><tr><td>no resampling</td><td>1.020 ± 0.049</td><td>0.191</td><td>0.03</td><td>0.57</td><td>1.07</td><td></td><td>56× (32–94×)</td></tr><tr><td>adaptive resampling</td><td>1.010 ± 0.021</td><td>0.082</td><td></td><td></td><td>0.95</td><td></td><td>311× (129–616×)</td></tr><tr><td>+ MADM moves</td><td>0.974 ± 0.020</td><td>0.083</td><td></td><td></td><td>1.17</td><td></td><td>257× (119–571×)</td></tr><tr><td rowspan="4"></td><td rowspan="4"> $1 0 ^ { - 5 }$ </td><td>no resampling</td><td>1.091 ± 0.062</td><td>0.228</td><td>0.02</td><td>0.88</td><td>0.82</td><td></td><td>395× (191–815×)</td></tr><tr><td>adaptive resampling</td><td>1.048 ± 0.034</td><td>0.129</td><td></td><td></td><td>0.87</td><td></td><td>1246× (778–2045×)</td></tr><tr><td>+ MADM moves</td><td> $1 . 0 1 9 \pm 0 . 0 2 3$ </td><td>0.089</td><td></td><td></td><td>0.64</td><td></td><td>2179× (1264–3573×)</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">Dry</td><td rowspan="5"> $1 0 ^ { - 3 }$   $1 0 ^ { - 4 }$ </td><td>no resampling adaptive resampling</td><td>0.931 ± 0.036 0.996 ± 0.018</td><td>0.153</td><td>0.06</td><td>0.53</td><td></td><td>1.15 1.42</td><td>8.9× (4.0–23×) 40× (24–60×)</td></tr><tr><td>+ MADM moves</td><td>0.962 ± 0.017</td><td>0.072 0.073</td><td></td><td></td><td></td><td>1.44</td><td>34× (17–58×)</td></tr><tr><td>no resampling</td><td>0.911 ± 0.047</td><td>0.206</td><td>0.04</td><td>0.63</td><td></td><td>1.06</td><td>49× (28–80×)</td></tr><tr><td>adaptive resampling</td><td>1.015 ± 0.055†</td><td>0.217</td><td></td><td></td><td></td><td>1.31</td><td>44× (19–179×)</td></tr><tr><td>+ MADM moves</td><td>0.964 ± 0.032</td><td>0.134</td><td></td><td></td><td></td><td>1.09</td><td>96× (37–362×)</td></tr><tr><td rowspan="4"></td><td rowspan="4"> $1 0 ^ { - 5 }$ </td><td>no resampling</td><td></td><td></td><td></td><td></td><td></td><td>0.97</td><td>185× (84–827×)</td></tr><tr><td>adaptive resampling</td><td>0.978 ± 0.082</td><td>0.333</td><td>0.02</td><td>0.75</td><td></td><td>0.68</td><td>1655× (817–3271×)</td></tr><tr><td>+ MADM moves</td><td>1.055 ± 0.029</td><td>0.112 0.142</td><td></td><td></td><td>一</td><td>0.71</td><td>815× (333–2381×)</td></tr><tr><td></td><td>1.038 ± 0.037</td><td></td><td></td><td>一</td><td>一</td><td></td><td></td></tr><tr><td rowspan="5">Joint</td><td rowspan="5"> $1 0 ^ { - 2 }$   $1 0 ^ { - 3 }$ </td><td>no resampling adaptive resampling</td><td>0.970 ± 0.020</td><td>0.082</td><td>0.09</td><td>0.46</td><td>1.25</td><td>1.17</td><td>6.0× (3.2–9.9×) 8.6× (3.5–22×)</td></tr><tr><td>+ MADM moves</td><td>0.965 ± 0.016</td><td>0.068</td><td></td><td></td><td>1.55</td><td>1.42 1.67</td><td>6.6× (3.5–11×)</td></tr><tr><td></td><td>0.950 ± 0.018</td><td>0.074</td><td></td><td></td><td>1.50</td><td></td><td></td></tr><tr><td>no resampling adaptive resampling</td><td> $0 . 9 6 7 \pm 0 . 0 4 3$ </td><td>0.179</td><td>0.03</td><td>0.67</td><td>1.22</td><td>1.09</td><td>19× (9.8–42×)</td></tr><tr><td>+ MADM moves</td><td>0.990 ± 0.028</td><td>0.113</td><td></td><td></td><td>1.34</td><td>1.39</td><td>47× (25–83×)</td></tr><tr><td rowspan="4"></td><td rowspan="4"> $1 0 ^ { - 4 }$ </td><td></td><td>1.047 ± 0.032</td><td>0.122</td><td></td><td></td><td>1.78</td><td>1.54</td><td>36× (14–90×)</td></tr><tr><td>no resampling</td><td> $1 . 0 0 4 \pm 0 . 0 6 7$ </td><td>0.268</td><td>0.02</td><td>0.85</td><td>1.28</td><td>1.02</td><td>126× (77-196×)</td></tr><tr><td>adaptive resampling</td><td> $1 . 0 4 4 \pm 0 . 0 2 1$ </td><td>0.080</td><td></td><td></td><td>1.14</td><td>0.86</td><td>1413× (854–2551×)</td></tr><tr><td>+ MADM moves</td><td> $1 . 0 4 0 \pm 0 . 0 2 5$ </td><td>0.096</td><td></td><td>一</td><td>1.00</td><td>0.95</td><td>842× (519–1372×)</td></tr></table>

## H.2.3 ESTIMATING THE MARGINALS AND THE JOINT

We expand on the multivariate hazard experiment of Section 4.2. d-values are computed using three guided runs for the guided process and read out as a fraction of joint exceedance p(hot & dry) from the Monte Carlo reference. For the joint guided runs we ablate between two variants for the CDF reward, one with a joint 2 dimensional CDF across the two variables Φ(hurs, tas) and another as a product of two individual CDFs $\Phi ( \tt h u r s ) \Phi ( \tt t a s )$ . We replace the ΠGDM’s isotropic covariance estimate, $\mathrm { C o v } ( x _ { 0 } | x _ { t } )$ with the model’s by computing the Jacobian Vector Product of $a ^ { \top } \Sigma _ { t } a$ (Boys et al., 2023), additionally we use a joint 2-dimensional bandwidth-inflated CDF $\Phi _ { 2 } ( \cdot )$ (Eq. (88)) as the reward estimated using Drezner’s algorithm with a 16-point Gauss-Legendre quadrature (Drezner & Wesolowsky, 1990).

Table 17 summarises the probability estimate obtained via the Monte Carlo reference which we compare against. In this setting we define p to be the probability a marginal exceeds a specific threshold, with the joint probability given by $\overline { { p ^ { 2 } } }$ under an independence assumption.

Table 17: Monte Carlo estimate and dependence for the compound hot-dry event at matched marginal quantiles. p represents the compound hot-dry event at matched marginal quantiles: at each level p both thresholds are the pool’s own p-quantiles, so $p _ { \mathrm { i n d e p } } = p ^ { 2 }$ exactly; quoted errors are binomial, and the $1 0 ^ { - 4 }$ row rests on 125 joint crossers.
<table><tr><td>marginal  $p$ </td><td> $\tau _ { \mathrm { t a s } } \left( \mathrm { K } \right)$ </td><td> $\tau _ { \mathrm { h u r s } } ~ ( \% )$ </td><td>Pjoint</td><td>Pindep</td><td> $d ^ { \star }$ </td></tr><tr><td> $1 0 ^ { - 1 }$ </td><td>3.8908</td><td>-8.6085</td><td> $( 5 . 0 2 8 \pm 0 . 0 0 7 ) \times 1 0 ^ { - 2 }$ </td><td> $1 . 0 0 \times 1 0 ^ { - 2 }$ </td><td> $5 . 0 3 \pm 0 . 0 1$ </td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td>4.9907</td><td>-12.7822</td><td> $( 3 . 1 3 8 \pm 0 . 0 1 9 ) \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 0 \times 1 0 ^ { - 4 }$ </td><td> $3 1 . 4 \pm 0 . 2$ </td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>5.8128</td><td> $- 1 5 . 8 3 8 2$ </td><td> $( 2 . 1 2 1 \pm 0 . 0 4 9 ) \times 1 0 ^ { - 4 }$ </td><td> $1 . 0 0 \times 1 0 ^ { - 6 }$ </td><td> $2 1 2 . 0 \pm 4 . 9$ </td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td>6.4977</td><td>-18.3871</td><td> $\dot { ( 1 . 3 8 7 \pm 0 . 1 2 4 ) } \times 1 0 ^ { - 5 }$ </td><td> $1 . 0 0 \times 1 0 ^ { - 8 }$ </td><td> $1 3 8 4 \pm 1 2 4$ </td></tr></table>

## H.2.4 FACTORISED VERSUS MULTIVARIATE GUIDANCE

We ablate the two reward variants - factorised (Eq. (94)) and joint $\left( \mathrm { E q . } \left( 8 8 \right) \right)$ of the compound weather event of Section 4.2. Across rarities, we compute the guidance based on a factorised reward (i.e. a product of two independent 1-dimensional CDFs) and its correlated $\Phi _ { 2 }$ variant. Both variants yield a different guidance drift as well as look-ahead reward $r _ { 0 , \delta }$ . Additionally, for the Gaussian surrogate of $p ( x _ { 0 } | x _ { t } ) \approx N ( \hat { x } _ { 0 | t } , \Sigma )$ , where ${ \hat { x } } _ { 0 \mid t }$ is Tweedie’s estimate, we replace the isotropic Σ approximation of (Song et al., 2023), with a JVP estimate of $\sigma ^ { 2 } A ^ { T } J A$ , where $A$ is defined by the linear function, J is the Jacobian of the network and $\sigma ^ { 2 }$ is the current noise level (Appendix E.1). The joint estimator costs roughly 1.64× wall-clock time (Appendix H.4) of the factorised but yields a lower variance of $p _ { 0 } [ E ]$ at the rarest level, which we discuss below.

The experiment was conducted with identical seeds and settings as the main text and compared against the MC reference. We use the same 16 seeds across both variants.

Table 18: Comparison of two reward estimates for compound hot & dry events. The difference between the two rewards is most prominent at the rarest level. Notably the $1 0 ^ { - 4 }$ reference itself carries an 8.9% standard error. Setup remains identical to Table 1, MC reference: Table 17.
<table><tr><td rowspan="2">marginal  $p$ </td><td colspan="3">Factorised reward</td><td colspan="3">Joint  $\Phi _ { 2 }$  reward</td></tr><tr><td> $\hat { p } / p ^ { \star }$ </td><td>CV↓</td><td> $\hat { d } / d ^ { \star }$ </td><td> $\hat { p } / p ^ { \star }$ </td><td>CV↓</td><td> $\hat { d } / d ^ { \star }$ </td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td> $\mathbf { 0 . 9 9 0 \pm 0 . 0 2 4 }$ </td><td>0.097</td><td> $1 . 0 7 \pm 0 . 0 3$ </td><td> $0 . 9 6 5 \pm 0 . 0 1 6$ </td><td>0.068</td><td> ${ \bf 1 . 0 4 \pm 0 . 0 2 }$ </td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td> $0 . 9 5 2 \pm 0 . 0 3 1$ </td><td>0.132</td><td> $0 . 9 8 \pm 0 . 0 3$ </td><td> ${ \bf 0 . 9 9 0 \pm 0 . 0 2 8 }$ </td><td>0.113</td><td> $1 . 0 2 \pm 0 . 0 3$ </td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td> $1 . 1 2 4 \pm 0 . 0 8 7$ </td><td>0.310</td><td> ${ \bf 1 . 1 2 \pm 0 . 0 8 }$ </td><td> ${ \bf 1 . 0 4 4 \pm 0 . 0 2 1 }$ </td><td>0.080</td><td> ${ \bf 1 . 0 6 \pm 0 . 0 6 }$ </td></tr></table>

The joint reward has a lower CV at every level, and the difference is most prominent at the rarest level: at $1 0 ^ { - 4 }$ the factorised reward has CV 0.310 against 0.080 (variance ratio 14.8, $p < 0 . 0 0 1 )$ and its $\hat { d } / d ^ { \star }$ is further from one (1.12 against 1.06). At $1 0 ^ { - 2 }$ and $1 0 ^ { - 3 }$ the two are statistically indistinguishable, and the cheaper factorised reward reaches a similar net speed-up (Table 21). As tas and hurs are strongly correlated $( \rho \approx 0 . 7 2 )$ , a product of marginal rewards undersamples the correlated region where $p ( x _ { i } ) / q ( x _ { i } ) > 1$ ; the few particles that reach it carry large weights, which inflates the variance at deeper rarities.

## H.2.5 CONDITIONAL FIELDS FOR COMPOUND EVENTS

The compound event as a field. Figure 18 contrasts the conditional mean field under the hot-only constraint with the one under the joint hot & dry constraint of Table 1. Guiding on the temperature alone, reduces the humidity across the region, illustrating why the dependency ratio $d > 1$

![](images/ff186bb8bbf5e926dce27e6d45fbf23755e407f2cdd75bf016a8be0766237e97.jpg)  
Figure 18: Conditional mean fields for the compound event of Table 1, over the PNW box in July at $\mathrm { G } \mathrm { \check { M } S T } + 1 . 5 \mathrm { K } .$ Top: conditioned on the hot constraint alone. Bottom: conditioned jointly on hot & dry. Left: near-surface temperature. Right: relative humidity. Both constraints are at marginal $1 0 ^ { - 3 } ;$ the box is the region of interest. Guiding on temperature alone reduces the humidity across a region.

## H.3 EXPERIMENT 2: A CHANGING CLIMATE

Table 19: Warming sweep at the frozen event $\tau = 5 . 8 1$ K over the region computed with 16 seeds per point. $p ^ { * }$ is obtained from a large Monte Carlo references, which exist only at $\Delta T \ge 1 . 5 \mathrm { K }$ (relative errors 1.1%, 3.2%, 0.7%, 0.13%) and the guided estimates match all four.
<table><tr><td>∆T (K)</td><td> $\hat { p }$ </td><td>s.e. ↓</td><td>CV↓</td><td> $p ^ { \star }$ </td><td> $\hat { p } / p ^ { \star }$ </td></tr><tr><td>1.5</td><td> $9 . 7 9 \times 1 0 ^ { - 4 }$ </td><td> $^ { \pm 0 . 1 8 \times 1 0 ^ { - 4 } }$ </td><td>0.073</td><td> $1 . 0 0 0 \times 1 0 ^ { - 3 }$ </td><td>0.978</td></tr><tr><td>2.0</td><td> $1 . 5 1 \times 1 0 ^ { - 2 }$ </td><td> $^ { \pm 0 . 0 3 \times 1 0 ^ { - 2 } }$ </td><td>0.073</td><td> $1 . 4 8 5 \times 1 0 ^ { - 2 }$ </td><td>1.016</td></tr><tr><td>3.0</td><td> $3 . 9 3 \times 1 0 ^ { - 1 }$ </td><td> $^ { \pm 0 . 0 2 \times 1 0 ^ { - 1 } }$ </td><td>0.025</td><td> $3 . 9 3 2 \times 1 0 ^ { - 1 }$ </td><td>1.000</td></tr><tr><td>4.0</td><td> $9 . 5 1 \times 1 0 ^ { - 1 }$ </td><td> $^ { \pm 0 . 0 2 \times 1 0 ^ { - 1 } }$ </td><td>0.006</td><td> $9 . 5 0 3 \times 1 0 ^ { - 1 }$ </td><td>1.001</td></tr></table>

Table 19 reports the sweep at a frozen threshold against the large Monte Carlo reference, the generated fields behind it are in Fig. 19. As the global mean surface temperature increases, the degree of warming across regions varies per region which is expected under the modelling condition of (Bouabid et al., 2026). As such, the regional mean temperature is greater than the GMST which is reflected at $\Delta T = 4 . 0 \mathsf { K }$ , where the estimated $p _ { 0 } [ E ] = 0 . 9 5$ . Table 20 reports the performance of DireSMC compared against a compute normalised MC estimate. Due to computational constraints we could not validate the estimates at $\Delta T \le 1 . 2 \ K ;$ to obtain a MC probability estimate at a 5% relative error we would require on the order of $1 0 ^ { 8 } / 1 0 ^ { 7 } / 1 0 ^ { 6 }$ draws from the unconditional model at $\Delta T = 0 . 0 / 1 . 0 / 1 . 2 \mathrm { K }$ respectively.

Table 20: Warming sweep at the frozen event $\tau = 5 . 8 1$ K over the region computed with 16 seeds per point. We compare against an unguided MC estimate at matched clock time allocating it 4.7× particles, over 16 replicates. Zero counts the number of MC replicates that record no event. See Table 19 for DireSMC validated against a large MC pool. Var. $\mathrm { R a t i o } = \mathrm { V a r } ( \hat { p } _ { \mathrm { M C } } ) / \mathrm { V a r } ( \hat { p } )$
<table><tr><td rowspan="2"> $\Delta T \left( \mathrm { K } \right)$ </td><td colspan="2">DireSMC</td><td colspan="3">Crude MC</td><td rowspan="2">Var. Ratio</td></tr><tr><td> $\hat { p }$ </td><td>CV</td><td> $\hat { p } _ { \mathrm { M C } }$ </td><td>CV</td><td>Zero</td></tr><tr><td>0.0</td><td> $1 . 0 6 \times 1 0 ^ { - 6 }$ </td><td>0.402</td><td>0</td><td>一</td><td>16/16</td><td>N/A</td></tr><tr><td>0.5</td><td> $1 . 6 5 \times 1 0 ^ { - 6 }$ </td><td>0.214</td><td>0</td><td></td><td>16/16</td><td>N/A</td></tr><tr><td>1.0</td><td> $3 . 5 1 \times 1 0 ^ { - 5 }$ </td><td>0.092</td><td> $1 . 3 0 \times 1 0 ^ { - 5 }$ </td><td>4.000</td><td>15/16</td><td>257</td></tr><tr><td>1.2</td><td> $1 . 4 6 \times 1 0 ^ { - 4 }$ </td><td>0.140</td><td> $1 . 0 4 \times 1 0 ^ { - 4 }$ </td><td>1.789</td><td>11/16</td><td>83</td></tr><tr><td>1.5</td><td> $9 . 7 9 \times 1 0 ^ { - 4 }$ </td><td>0.073</td><td> $1 . 0 3 \times 1 0 ^ { - 3 }$ </td><td>0.594</td><td>0/16</td><td>73</td></tr><tr><td>2.0</td><td> $1 . 5 1 \times 1 0 ^ { - 2 }$ </td><td>0.073</td><td> $1 . 6 6 \times 1 0 ^ { - 2 }$ </td><td>0.138</td><td>0/16</td><td>4.4</td></tr><tr><td>2.5</td><td> $1 . 1 0 \times 1 0 ^ { - 1 }$ </td><td>0.049</td><td> $1 . 1 4 \times 1 0 ^ { - 1 }$ </td><td>0.049</td><td>0/16</td><td>1.05</td></tr><tr><td>3.0</td><td> $3 . 9 3 \times 1 0 ^ { - 1 }$ </td><td>0.025</td><td> $3 . 9 6 \times 1 0 ^ { - 1 }$ </td><td>0.015</td><td>0/16</td><td>0.39</td></tr><tr><td>3.5</td><td> $7 . 5 2 \times 1 0 ^ { - 1 }$ </td><td>0.012</td><td> $7 . 5 3 \times 1 0 ^ { - 1 }$ </td><td>0.010</td><td>0/16</td><td>0.65</td></tr><tr><td>4.0</td><td> $9 . 5 1 \times 1 0 ^ { - 1 }$ </td><td>0.006</td><td> $9 . 5 0 \times 1 0 ^ { - 1 }$ </td><td>0.004</td><td>0/16</td><td>0.33</td></tr></table>

![](images/c5fc49917acecf9cbdaf17e0c2ce903e8082f96e199d6722483859a2844eadd7.jpg)  
Figure 19: Highest weight samples under the no resampling regime. We guide the sampler to the same $\tau _ { \textrm t a s } \geq 5 . 8 1 \mathrm { K }$ for a range of global average temperatures ∆T. See Table 19 for the qualitative analysis.

## H.4 COMPUTATIONAL COST

We measure the net-speed-up of the path-weighted estimator by comparing the net wall-clock time required for each example in Table 21. We compare our results against the number of $N _ { M C }$ samples required to form a Monte Carlo estimate at equal-CV, discounted by the added cost of computing the guidance. An unguided draw is one denoiser call per step (0.151 s per sample at $K = 2 0 0 )$ ; the production sampler measures $4 . 7 \times$ that per sample for a univariate or factorised reward (marginal threshold exceedances and multivariate exceedance under uncorrelated reward), and the correlated variant of the reward measures a larger $7 . 7 \times$ . In general, any m-coupled reward requires m vector Jacobian product calls per denoising step, which contributes a linearly growing computational cost to the wall-clock time.

Table 21: Net speed-up over crude Monte Carlo at matched CV for $N = 1 0 2 4$ guided draws against unguided draws. Computed using CVs obtained from Tables 15, 1 and 18. The dry $1 0 ^ { - 4 }$ entry is low because of one collapsed run (App. H.2.1); without it the CV is 0.124, close to the other $1 0 ^ { \dot { - } 4 }$ cells.
<table><tr><td rowspan="2"> $p ^ { * }$ </td><td colspan="2">Threshold Exceedances</td><td colspan="2">Joint hot &amp; dry</td></tr><tr><td>hot</td><td>dry</td><td>factorised reward</td><td>Joint reward</td></tr><tr><td>Per-sample cost</td><td>4.7×</td><td>4.7×</td><td>4.7×</td><td> $7 . 7 \times$ </td></tr><tr><td> $1 0 ^ { - 1 }$ </td><td> $0 . 7 \times$ </td><td>1.0×</td><td>1.3×</td><td>1.2×</td></tr><tr><td> $1 0 ^ { - 2 }$ </td><td> $8 . 7 \times$ </td><td>3.9×</td><td>7.0×</td><td>8.6×</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td> $3 4 \times$ </td><td>40×</td><td>56×</td><td> $4 7 \times$ </td></tr><tr><td> $1 0 ^ { - 4 }$ </td><td> $3 1 1 \times$ </td><td>44×</td><td>156×</td><td>1413×</td></tr><tr><td> $1 0 ^ { - 5 }$ </td><td>1246×</td><td>1655×</td><td>一</td><td></td></tr></table>

We find that for most of the generations, the net speedup of the path-weighted sampler out-weighs the added clock time for guided steps. Additionally, at deeper rarities where Monte Carlo has zero samples within the extreme event set of interest (e.g. for very rare events), the method becomes a point of feasibility rather than speed-up alone. These speed-ups scale with the quality of the guidance itself.

## H.5 HYPERPARAMETER ABLATION

We test a range of schedules of $\left( \lambda ( \bar { \sigma } ) , \delta ( \bar { \sigma } ) \right)$ on the threshold exceedance problem at a marginal $p ^ { * } = 1 0 ^ { - 3 }$ . We caveat that the parameters we present were chosen heuristically, and we do not independently tune the hyperparameters for any other setting presented in our paper. While we tune on the marginal hot and dry $\stackrel { \cdot } { \tau _ { 1 } } ( p ^ { * } = 1 0 ^ { - 3 } )$ problems, we believe there are further performance improvements that can be made by better schedule tuning. All ablations are done with a single change to the default parameters described in App. H.1.1. We report $\hat { p } _ { 0 } [ E ] / p ^ { * }$ against the Monte Carlo anchor, with errors displayed as the standard errors over the 16 seeds, which are shared across all arms.

Table 22: One-knob ablation of the guidance schedules at $p ^ { * } = 1 0 ^ { - 3 }$ . We vary a single parameter reference in Table 14; the same 16 seeds are used in every row. Collapsed runs are included: one hot run at ${ \lambda } _ { \operatorname* { m a x } } = 2 . 0$ , and two hot and two dry runs at $\delta _ { \mathrm { h i } } = 0 . 2 5 \mathrm { S D }$
<table><tr><td colspan="4"></td><td colspan="2">Hot (tas)</td><td colspan="2">Dry (hurs)</td></tr><tr><td>Knob</td><td>Deployed Ablated</td><td></td><td> $\hat { p } / p ^ { * }$ </td><td>CV</td><td></td><td> $\hat { p } / p ^ { * }$ </td><td>CV</td></tr><tr><td colspan="6">Deployed recipe</td><td> $0 . 9 9 6 \pm 0 . 0 1 8$ </td><td>0.072</td></tr><tr><td colspan="3">Tempering schedule  $\lambda ( \bar { \sigma } )$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Peak  $\lambda _ { \mathrm { m a x } }$ </td><td>1.25</td><td>1.0 (no overshoot)</td><td> $1 . 0 1 0 \pm 0 . 0 4 5$ </td><td></td><td>0.180</td><td> $0 . 9 8 4 \pm 0 . 0 2 3$ </td><td>0.095</td></tr><tr><td>Peak  $\lambda _ { \mathrm { m a x } }$ </td><td>1.25</td><td>1.5</td><td></td><td> $0 . 9 8 4 \pm 0 . 0 1 8$ </td><td>0.071</td><td> $0 . 9 8 4 \pm 0 . 0 2 2$ </td><td>0.090</td></tr><tr><td>Peak  $\lambda _ { \mathrm { m a x } }$ </td><td>1.25</td><td>2.0</td><td></td><td> $1 . 0 2 5 \pm 0 . 0 5 1$ </td><td>0.199</td><td> $0 . 9 4 2 \pm 0 . 0 3 7$ </td><td>0.159</td></tr><tr><td>Onset  $\bar { \sigma } _ { \mathrm { o n } }$ </td><td>40</td><td></td><td>20 (later)</td><td> $0 . 9 3 7 \pm 0 . 0 2 5$ </td><td>0.106</td><td> $0 . 9 6 9 \pm 0 . 0 3 4$ </td><td>0.139</td></tr><tr><td>Onset  $\bar { \sigma } _ { \mathrm { o n } }$ </td><td>40</td><td></td><td>80 (earlier)</td><td> $0 . 9 8 9 \pm 0 . 0 2 2$ </td><td>0.088</td><td> $0 . 9 6 7 \pm 0 . 0 2 3$ </td><td>0.094</td></tr><tr><td colspan="3">Smoothing width  $\delta ( \bar { \sigma } )$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Transport  $\delta _ { \mathrm { h i } }$ </td><td>0.5SD</td><td></td><td>0.25 SD</td><td> $1 . 0 8 8 \pm 0 . 1 1 2$ </td><td>0.411</td><td> $0 . 7 7 7 \pm 0 . 0 8 4$ </td><td>0.431</td></tr><tr><td>Floor  $\delta _ { \mathrm { l o } }$ </td><td>0.01 SD</td><td></td><td>0.05 SD</td><td> $1 . 0 0 9 \pm 0 . 0 1 9$ </td><td>0.074</td><td> $1 . 0 2 3 \pm 0 . 0 1 4$ </td><td>0.055</td></tr></table>

## I METROPOLIS-ADJUSTED LANGEVIN CORRECTIONS

We use Metropolis-Adjusted Diffusion Model (MADM, (Lam et al., 2026)) for the rejuvenation kernel $M _ { k }$ of Section 3.1. From the predictor-corrector framework of (Song et al., 2021b) the guided score can be used as a predictor and M MADM steps act as our corrector, targeting the tilted marginals $\pi _ { k }$ of Eq. (5). The unadjusted Langevin algorithm (ULA) update is,

$$
\begin{array} { r } { \tilde { x } \ = \ x + \frac { \rho } { 2 } g ( x ) + \sqrt { \rho } z , z \sim \mathcal { N } ( 0 , I ) , g ( x ) : = \nabla _ { x } \log \pi _ { k } ( x ) = s _ { \theta } ( x , t _ { k } ) + h _ { t _ { k } } ( x ) . } \end{array}\tag{114}
$$

Here g is the guided score. ULA accepts all proposals $x  \tilde { x }$ and is biased for every finite step size $\rho ,$ and only leaves $\pi _ { k }$ invariant in the $\rho \to 0$ limit.

A Metropolis-Hastings correction removes this bias while keeping the same proposal: accept x˜ with probability $A ( x , \tilde { x } ) \stackrel { - } { = } 1 \wedge \left. \Gamma _ { \pi } ( x , \tilde { x } ) H ( x , \tilde { x } ) \right.$ , otherwise stay at x, where the acceptance ratio splits into a target ratio $\Gamma _ { \pi }$ and a proposal ratio H. The proposal ratio is Gaussian and closed-form,

$$
\log H ( x , \tilde { x } ) = \frac { 1 } { 2 \rho } \big \| \tilde { x } - x - \textstyle \frac { \rho } { 2 } g ( x ) \big \| ^ { 2 } - \frac { 1 } { 2 \rho } \big \| x - \tilde { x } - \textstyle \frac { \rho } { 2 } g ( \tilde { x } ) \big \| ^ { 2 } ,\tag{115}
$$

while the target ratio is exactly the tilted-density ratio at $t _ { k }$

$$
\Gamma _ { \pi } ( x , \tilde { x } ) = { \frac { \pi _ { k } ( \tilde { x } ) } { \pi _ { k } ( x ) } } \propto { \frac { p _ { t _ { k } } ( \tilde { x } ) e ^ { r _ { t _ { k } } ( \tilde { x } ) } } { p _ { t _ { k } } ( x ) e ^ { r _ { t _ { k } } ( x ) } } } .\tag{116}
$$

Only $\Gamma _ { \pi }$ remains intractable for a diffusion model: the reward factor $e ^ { r _ { t _ { k } } }$ is available in closed form, but we do not have direct access to the base marginal $p _ { t _ { k } }$ , only its score $s _ { \theta } ( x , t _ { k } )$ . Writing the target ratio as

$$
\log \Gamma _ { \pi } ( x , \tilde { x } ) = \underbrace { \log \frac { p _ { t _ { k } } ( \tilde { x } ) } { p _ { t _ { k } } ( x ) } } _ { \mathrm { i n t r a c t a b l e } } + \underbrace { \bigl ( r _ { t _ { k } } ( \tilde { x } ) - r _ { t _ { k } } ( x ) \bigr ) } _ { \mathrm { e x a c t } }\tag{117}
$$

isolates the one hard piece: the reward term is known in closed form and needs no approximation at all. MADM (Lam et al., 2026) sidesteps the marginal-density problem by computing the remaining $p _ { t _ { k } }$ -ratio, under a well trained network where $s _ { \theta } = \nabla _ { x } \log p _ { t _ { k } }$ , using a line-integral,

$$
\log \frac { p _ { t _ { k } } ( \tilde { x } ) } { p _ { t _ { k } } ( x ) } = \int _ { 0 } ^ { 1 } \langle s _ { \theta } \big ( x + u ( \tilde { x } - x ) , t _ { k } \big ) , \tilde { x } - x \rangle \mathrm { d } u ,\tag{118}
$$

valid when $s _ { \theta }$ is a gradient field, which holds exactly for an analytic score but is not enforced by the denoising objective for a learned network. This is the conservation error of the learned score, which is small in high noise regions, where we apply rejuvenation. The line integral can be approximated by quadrature, for instance by Simpson’s rule at the midpoint $\bar { x } = ( x + \tilde { x } ) / 2$

$$
\begin{array} { r } { \widehat { \log \Gamma _ { \pi } } \ = \ \frac { 1 } { 6 } \big \langle s _ { \theta } ( x , t _ { k } ) + 4 s _ { \theta } ( \bar { x } , t _ { k } ) + s _ { \theta } ( \tilde { x } , t _ { k } ) , \tilde { x } - x \big \rangle \ + \ \big ( r _ { t _ { k } } ( \tilde { x } ) - r _ { t _ { k } } ( x ) \big ) , } \end{array}\tag{119}
$$

giving the practical acceptance probability $\widehat { \cal A } ( x , \tilde { x } ) = 1 \wedge \exp \big ( \widehat { \log \Gamma _ { \pi } } \big ) { \cal H } ( x , \tilde { x } )$ . The endpoint scores $s _ { \theta } ( x , t _ { k } ) , s _ { \theta } ( \tilde { x } , t _ { k } )$ and rewards $r _ { t _ { k } } ( x ) , r _ { t _ { k } } ( \tilde { x } )$ are shared with the proposal and its acceptance test, so the quadrature costs additional unconditional score evaluations.

The quadrature carries a truncation error of order $\mathcal { O } ( \rho ^ { 5 / 2 } )$ on the $p _ { t _ { k } } \mathrm { - r a t i o } ,$ , so the adjusted kernel is not exactly $\pi _ { k }$ -invariant but is accurate to far higher order than the $\mathcal { O } ( \rho )$ bias of the unadjusted chain (Lam et al., 2026). An exact alternative replaces the Metropolis rule with Barker acceptance, at the cost of additional score evaluations (Lam et al., 2026); but we use the Simpson-rule variant throughout.

We can summarise a single MADM move under the Simpson’s-rule variant of (Lam et al., 2026) as (i) propose a transition with Eq. (114) and (ii) accept the proposal with an acceptance probability approximated with Eqs. (115) and (119). Algorithm 2 states this as the rejuvenation kernel $M _ { k }$ called by Algorithm 1.

Algorithm 2 One MADM rejuvenation move at level $k ,$ targeting $\pi _ { k } \propto p _ { t _ { k } } e ^ { r _ { t _ { k } } }$   
Require: state $x ,$ level k, score $s _ { \theta } ( \cdot , t _ { k } )$ , reward $r _ { t _ { k } }$ , step size $\rho .$   
1: $g ( x ) \gets s _ { \theta } ( x , t _ { k } ) + \nabla _ { x } r _ { t _ { k } } ( x )$ ▷ Guided score   
2: $\begin{array} { r } { \tilde { x }  x + \frac { \rho } { 2 } g ( x ) + \sqrt { \rho } z , } \end{array}$ $z \sim \mathcal { N } ( 0 , I )$ ▷ Langevin proposal, Eq. (114)   
3: log $\widehat { A } \gets$ min 0, log $\widehat { \Gamma _ { \pi } } ( x , \tilde { x } ) + \log H ( x , \tilde { x } ) \big \}$ ▷ Eqs. (119) and (115)   
4: With probability $\widehat { A }$ set $x \gets \tilde { x } ;$ otherwise keep $x .$   
5: return x

## J RELATED WORKS EXTENDED

We compare our method for generating the rare event probabilities against the method introduced in (Manshausen et al., 2026). Both methods guide a pre-trained diffusion model towards the extremes, and reweight the resulting samples to estimate $p _ { 0 } [ E ]$ under the model density $p _ { \theta } .$ . The methods differ at the integrator. The method of (Manshausen et al., 2026) samples with an ODE and estimates the rare-event probability using the Probability Flow ODE (PF-ODE) (Song et al., 2021b) for both unguided and guided processes, which allows them to weight each sample by its exact terminal likelihood. We sample with a reverse SDE and weight each trajectory by the Girsanov density of the guided path measure, computed against the unguided process. Neither object is available to the other: a stochastic integrator has no PF-ODE likelihood, and a deterministic flow induces path measures that are mutually singular, so Girsanov’s theorem is not available.

Resampling mid-trajectory is only available to the path-weights formulation as a terminal-only weight defines no intermediate targets to resample against. During the denoising process, we measure how far the sampled trajectory Q deviates from one defined by the tilted process $\mathbb { P } _ { r _ { 0 , \delta } } ,$ and adaptively resample within a Sequential Monte Carlo scheme. Under the operating guidance of (Manshausen et al., 2026), this yields us a far larger fraction of particles from the rare-event set $E .$

Notation. The arrow on a density is the direction in which it is evaluated: backward (←) means along the sampling run itself (i.e. from a Gaussian prior → the data distribution); forward (→) is the noising direction (i.e. data → prior) obtained via integrating the ODE from the generated sample back to noise. K is the number of integrator steps; arrow-free symbols $( p _ { 0 } , q _ { 0 } )$ are the continuous-time objects, untouched by the discretisation. The key point is that, at a finite $K$ , the two directions assign different values to the same density (dependent on the integrator of choice).

p<sub>0</sub>, q<sub>0</sub> model’s true density and its guided (tilted) counterpart; target $\begin{array} { r } { p _ { 0 } [ E ] = \int _ { E } p _ { 0 } } \end{array}$   
q
q
p
p discretised density of the guided draws $x \sim \overline { { q } }$   
the same guided density evaluated by the forward ODE run - their denominator   
discretised density of the unguided process - our estimator targets this quantity   
the unguided density evaluated by a forward ODE - their numerator   
$\bar { \mathcal { E } } _ { 0 } = \log ( \vec { p } / \overleftarrow { p } )$ arrow gap of the unguided density: forward and backward evaluation of the same   
process.   
$\mathcal { E } _ { \gamma } = \log ( \overrightarrow { q } / \overleftarrow { q } )$ arrow gap of the guided density   
$\dot { D _ { \mathrm { m a p } } } = \dot { \log ( p / p _ { 0 } ) }$ the discrete chain’s gap to the model

The three discrepancies $\mathcal { E } _ { 0 } , \mathcal { E } _ { \gamma } , D _ { \mathrm { m a p } }$ are all $O ( 1 / K )$ discretisation quantities and vanish as $K  \infty$

## J.1 THE PF-ODE ESTIMATOR

Both methods target the same quantity, $p _ { 0 } [ E ]$ by importance sampling with a guided proposal: draw $x \sim \overline { { q } }$ , then reweight,

$$
p _ { 0 } [ E ] = \int p _ { 0 } \mathbf { 1 } _ { E } \mathrm { d } x = \mathbb { E } _ { x \sim \overline { { q } } } \Big [ \frac { p _ { 0 } } { \overline { { q } } } \mathbf { 1 } _ { E } \Big ] .\tag{120}
$$

The two methods differ only in how this ratio is estimated. We compute the Girsanov path ratio $\mathrm { d } \mathbb { P } / \mathrm { d } \mathbb { Q }$ using transition kernels which is accumulated during the run itself. It is a functional of the whole trajectory $X _ { [ 0 , 1 ] }$ , not of its endpoint alone. The method of (Manshausen et al., 2026) is a ratio of twoforward-evaluated endpoint densities:

$$
\begin{array} { r l r } & { \mathrm { o u r s : } \ \mathbb { E } _ { X _ { [ 0 , 1 ] } \sim \mathbb Q } \Big [ \frac { \mathrm { d } \mathbb P } { \mathrm { d } \mathbb Q } \big ( X _ { [ 0 , 1 ] } \big ) \mathbf { 1 } _ { E } \big ( X _ { 0 } \big ) \Big ] , } & { \quad \mathrm { ( M a n s h a u s e n ~ e t ~ a l . , ~ 2 0 2 6 ) : } \ \mathbb { E } _ { x \sim \tilde { q } } \big [ \frac { \vec { p } } { \sharp } \big ( x \big ) \mathbf { 1 } _ { E } ( x ) \big ] . } \end{array}\tag{121}
$$

Both estimators target the integral of Eq. (120), ours by the tower property in conditional expectations, $\mathbb { E } _ { \mathbb { Q } } [ \mathrm { d } \mathbb { P } / \mathrm { d } \mathbb { Q } \mid X _ { 0 } ] \stackrel { \smile } { = } \bar { p } ( X _ { 0 } ) / \breve { \overline { { q } } } ( X _ { 0 } )$ and theirs by the ratio $\vec { p } / \vec { q } \approx p _ { 0 } / \stackrel {  } { q }$ . Our path weight uses the backward or denoising process, simulated under the guided chain. Note that importance sampling asks us to divide by the density of the draws, i.e. the backward object $\overleftarrow { q }$ instead of $\vec { q }$ . We note that both estimators need not have the same variance (Chopin et al., 2020; Del Moral et al., 2006).

The method of (Manshausen et al., 2026) first generates a sample by integrating a deterministic ODE from $t = 1$ down to data at $t = 0$

$$
x _ { 0 } ~ = ~ x _ { 1 } ~ - ~ \int _ { 0 } ^ { 1 } \Big ( f _ { t } x _ { t } ~ - ~ \frac { 1 } { 2 } \sigma _ { t } ^ { 2 } \big ( s _ { \theta } ( x _ { t } , t ) + \gamma h _ { t } ( x _ { t } ) \mathbf { 1 } \{ t \in \Delta \} ) \Big ) \mathrm { d } t ,\tag{122}
$$

where $h _ { t }$ is their classifier guidance term and $\gamma$ is their guidance strength. After sampling, samples that belong to the rare event set of interest $x _ { E } \sim \frac {  } { q } \in E$ are selected and their likelihoods are evaluated: from each sample $x _ { E } ,$ they re-integrate the same dynamics in the noising direction $t = 0  1$ , accumulating the integral,

$$
\log \vec { p } ( x _ { E } ) = \log p _ { 1 } ( x _ { 1 } ) + \int _ { 0 } ^ { 1 } \Big ( d f _ { t } - \frac { 1 } { 2 } \sigma _ { t } ^ { 2 } \boldsymbol { \nabla } \cdot \boldsymbol { s } _ { \theta } ( x _ { t } , t ) \Big ) \mathrm { d } t ,\tag{123}
$$

where $d$ is the dimension and $x _ { 1 }$ the endpoint of the re-integration (noise), log q<sup>⇀</sup> is obtained similarly via replacing the score in Eq. 123 with its guided variant, $s _ { \theta } \to s _ { \theta } + \gamma h _ { t } ( x _ { t } ) \mathbf { 1 } \{ t \in \Delta \}$ . Their importance weight is given by,

$$
\log \frac { p _ { 1 } ( x _ { 1 } ) } { p _ { 1 } ( x _ { 1 } ^ { \gamma } ) } - \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \sigma _ { t } ^ { 2 } \left( \nabla \cdot \boldsymbol { s } _ { \theta } ( x _ { t } , t ) - \nabla \cdot \boldsymbol { s } _ { \theta } ( x _ { t } ^ { \gamma } , t ) \right) \mathrm { d } t + \frac { \gamma } { 2 } \int _ { \Delta } \sigma _ { t } ^ { 2 } \nabla \cdot \boldsymbol { h } _ { t } ( x _ { t } ^ { \gamma } ) \mathrm { d } t ,\tag{124}
$$

where $x _ { t }$ and $\boldsymbol { x } _ { t } ^ { \gamma }$ are the unguided and guided re-integration trajectories from the same sample $x _ { E }$ with endpoints $x _ { 1 }$ and $x _ { 1 } ^ { \gamma }$

The assumption of (Manshausen et al., 2026). The weight is exact iff $( \mathrm { i } ) \vec { p } = p _ { 0 }$ , or when the forward density equals the true model density, where $D _ { \mathrm { m a p } } \bar { = } 0 ,$ and $( \operatorname { i i } ) \vec { q } = \overleftarrow { q }$ , where the forward re-integration of the guided path reproduces the density of the draws, i.e, $\mathcal { E } _ { \gamma } = 0$ . Neither holds at finite $K$ for the explicit Euler integrator used, as the reverse of this integrator is the implicit Euler which is not used. Hence the forward density $\vec { q }$ does not retrace the same trajectory as the backward process ${ \overline { { q } } } .$

The bias (the expectation over guided draws in E) of this is given by,

$$
\frac { \hat { p } _ { 0 } [ E ] } { p _ { 0 } [ E ] } \ : = \ : \mathbb { E } \big [ \exp \big ( \mathcal { E } _ { 0 } - \mathcal { E } _ { \gamma } + D _ { \operatorname* { m a p } } \big ) \big ] ,\tag{125}
$$

which is unbiased iff the right-hand side is one. This happens trivially, when $\mathcal { E } _ { 0 } - \mathcal { E } _ { \gamma } = 0 \mathrm { ( i . e }$ . no guidance) and $D _ { \mathrm { m a p } } = 0 , \mathrm { i . e . } \ \overline { { p } } = p _ { 0 }$ . The non-trivial solution requires ${ \overline { { p } } } = { \vec { p } } = p _ { 0 }$ and $\vec { q } = \stackrel {  } { q }$ such that $\mathcal { E } _ { 0 } = \mathcal { E } _ { \gamma } \overset { \cdot } { = } 0$ which occurs exactly at $K  \infty$ . In a non-analytical setting, $D _ { \mathrm { m a p } }$ is unknown and cannot be evaluated, hence, both methods assume $\overline { { p } } = p _ { 0 }$ . At finite $K ,$ , an accurate estimator therefore requires the three terms to cancel. Within the finite limit, the system becomes unbiased if $\mathcal { E } _ { \gamma } = \mathcal { E } _ { 0 } + D _ { \mathrm { m a p } }$ such that individual biases cancel under addition. Our counterpart is a single condition, $\stackrel { \left. \right.} { p }  p _ { 0 } :$ the law of the K-step chain must approach the model’s, or $D _ { \mathrm { m a p } } \to 0$ We too need large K for this to be satisfied. In the absence of $p _ { 0 }$ and $D _ { \mathrm { m a p } } .$ , our condition is trivially satisfied, whereas (Manshausen et al., 2026) requires $\mathcal { E } _ { 0 } = \mathcal { E } _ { \gamma }$ or the cancellation of individual biases.

## J.1.1 EMPIRICAL RESULTS

As the laws induced by the discretised ODE and SDE integrators need not coincide generally for a learned score, see Deveney et al. (2025), we evaluate the bias using analytical scores where the true rare-event probability $p _ { 0 } [ E ]$ is known without any discretisation bias. We compare these methods using the 2D GMM threshold exceedance problem at $\tau = 1 0 .$ . As both methods use first order integrators, we use the exponential integrator for both ODE and SDE variants with analytical scores and Doob’s guidance on the tested equal ℓ discretisation (see App. G.2.2). In the ODE setting of (Manshausen et al., 2026), this is equivalent to the DDIM (Gonzalez et al., 2023; Lu et al., 2022; Song et al., 2021a) integrator, but under the EDM’s parametrisation (Karras et al., 2022) this corresponds to the Euler integrator used in (Manshausen et al., 2026). Additionally, we clip the guidance to ensure that the crossing fraction remains 10% for both methods (under no resampling) which mirrors the operational regime of (Manshausen et al., 2026). We calibrate the clipping for both methods at each K separately.

Remark To isolate the effect of discretisation bias, we provide the true $\nabla \cdot s ( x _ { t } , t )$ in Eq. (124) for the PF-ODE computation of (Manshausen et al., 2026) instead of approximating it via the Hutchinson trace estimator. However, in practice the trace estimator contributes variance to the estimate, shrinking with the number of probes in exchange for a larger computational overhead.

Table 23: Signed percentage bias for the rare-event probability estimated against the analytical ground truth (obtained exactly without discretisation) for the 2D GMM threshold exceedance problem at $\tau = 1 0$ . Our method tracks a lower bias at finite discretisation K compared to (Manshausen et al., 2026). Positive % reflects an overestimate and are presented as a percentage against the true rare-event probability.
<table><tr><td>guidance K</td><td>p/ ((Manshausen et al., 2026)) ↓</td><td></td><td>ours ↓</td></tr><tr><td>exact Doob</td><td>50</td><td> $+ 1 5 5 . 7 \% \pm 0 . 3$ </td><td> $\mathbf { 2 . 2 \% 2 0 . 7 }$ </td></tr><tr><td>exact Doob</td><td>100</td><td>+65.3%±0.2</td><td> $- 2 . 6 \% \pm 0 . 4$ </td></tr><tr><td>exact Doob</td><td>200</td><td>+31.7%±0.2</td><td> $- \mathbf { 1 . 9 \% } \mathbf { \pm 0 . 3 }$ </td></tr><tr><td>exact Doob</td><td>500</td><td>+14.7%±0.1</td><td> $- \mathbf { 1 . 0 \% } \mathbf { \mu } \mathbf { \mu } \mathbf { 0 . 2 }$ </td></tr><tr><td>clipped Doob, ～10% cross</td><td>50</td><td> $+ 4 2 . 8 \% \pm 4 . 2$ </td><td> $+ 2 . 7 \% \pm 2 . 5$ </td></tr><tr><td>clipped Doob, ~10% cross</td><td>100</td><td> $+ 2 1 . 2 \% \pm 3 . 4$ </td><td> $- 2 . 1 \% \pm 1 . 9$ </td></tr><tr><td>clipped Doob, ~10% cross</td><td>200</td><td> $+ 1 0 . 7 \% \pm 3 . 0$ </td><td> $- 1 . 6 \% \pm 1 . 8$ </td></tr><tr><td>clipped Doob, ~10% cross</td><td>500</td><td> $+ 4 . 4 \% \pm 2 . 8$ </td><td> $- \mathbf { 1 . 9 \% } \pm 2 . 2$ </td></tr></table>

Under the exact Doob’s guidance, the discretisation bias from $\overline { { p } } / \overline { { q } }$ survives, as the guidance is employed throughout the trajectory. We report the discretisation bias against K in Table 23. Exact guidance highlights that this is a discretisation bias itself which converges as ${ \vec { p } } \right. { \stackrel { \left. } { p } }$ and $D _ { \mathrm { m a p } } = 0$ For our method, any error lives in $D _ { \mathrm { m a p } } .$ The discretisation bias of (Manshausen et al., 2026) is dependent on a specific choice of γ and $h _ { t } ,$ , reflected in $\mathcal { E } _ { \gamma }$ . Hence the two proposals in Table 23 give a different estimate for $p _ { 0 } [ E ]$ . Additionally, as we adaptively resample, our crossing fraction remains high even with the clipped guidance at $8 6 . 5 \% / 8 0 . 4 \bar { \% } / 8 1 \bar { . } 3 \% / 7 \bar { 9 . } 4 \%$ (averaged across seeds) for $\bar { K ^ { ' } } = 5 0 / 1 0 0 / 2 0 0 / 5 0 \bar { 0 }$ respectively compared to theirs at $1 0 \% .$ , which results in a higher particle population concentration on the event of interest. We need not discard 90% of samples in this regime, and this is reflected in a lower seed-level standard error of our method.

Remark on computational costs. Let N be the number of particles, R the number of resampling steps, M the number of MCMC move of Algorithm 1, and P the number of Hutchinson trace estimator probes used per step in Eq. (124). Both samplers pay a different number of function evaluations per particle, and have a different compute budget. Table 24 summarises this. K is fixed and set by the discretisation bias tolerance versus compute. Which has the same weak-order convergence for both methods (first-order) (Gonzalez et al., 2023; Manshausen et al., 2026).

Table 24: Network evaluations per particle. For a K step denoising process, with R rejuvenation steps and M MCMC moves of Algorithm 1. P denotes the number of Hutchinson’s probes in Eq. (124).
<table><tr><td></td><td>score function</td><td>r</td><td> $\nabla r$ </td><td>JVP</td></tr><tr><td>ours  $( \mathrm { A l g . 1 } )$ </td><td> $K + 2 R M$ </td><td> $K { + } R M$ </td><td> $K { + } R M$ </td><td></td></tr><tr><td>odds ratio (Eq. 124)</td><td>3K</td><td>2K</td><td>2K</td><td> $2 P K$ </td></tr></table>

As an illustrative example, take the $\tau = 1 0$ example of Section $^ { 4 , }$ coupled with the guidance term of Eq. (18) and the $\Pi \bar { \mathrm { G D M } } \Sigma _ { t }$ proxy (Song et al., 2023). Averaged over 24 seeds, resampling is triggered $R = 3 . 5$ times, followed by $M = \bar { 3 } \mathrm { M A D M }$ rejuvenation steps. If we set $P = 1$ , which is a conservative estimate as typically a larger P results in a lower variance estimate of the Jacobian trace. If we were to count every single score evaluation, reward and reward-gradient evaluation as one call, our estimator uses $2 \dot { K } + \bar { 3 } R M \approx 2 0 3 2$ function calls per particle, of which rejuvenation moves account for $^ { 3 2 , }$ whereas the odds-ratio estimator with a single probe uses $5 K + 2 \bar { P K } = 7 0 0 0$ As a reward and its gradient calls are often paired together, at an equal step count Manshausen et al. (2026) receive $N / 3 . 5$ particles at the same compute budget, which is the comparison drawn in Fig 2.

## J.1.2 DETAILED COMPARISON

Below we collate the tables used to generate Fig. 2. In the first two panels, we use a learned score network, wheras the final uses an analytical score with Doob’s transition kernels. We motivate why this is selected in Section 4.1. All hyperparameters are kept the same unless they are ablated. The only exception is that each arm uses the best $\lambda ( t )$ tempering parameter as selected in App. G.5.

CV vs N. We keep the same $K = 1 0 0 0$ and ablate at $\tau = 1 0$ with learned scores. Table 25 summarises these results, where we find that the CV of DireSMC (w/ rejuvenation) outperforms Manshausen et al. (2026) at all K.

Table 25: CV of $\hat { p }$ against particle count N (Fig. 2, left), $\mathrm { a t } \ \tau = 1 0 , K = 1 0 0 0$ , with the learned score, over 24 seeds. Measured against model’s tail mass $p ^ { * }$ (Table 7).
<table><tr><td colspan="2"></td><td colspan="3">DireSMC</td></tr><tr><td>N</td><td>Manshausen et al. (2026)</td><td>base</td><td>+ resampling</td><td>+ MADM</td></tr><tr><td>100</td><td>0.631</td><td>0.999</td><td>0.611</td><td>0.321</td></tr><tr><td>250</td><td>0.385</td><td>0.506</td><td>0.394</td><td>0.184</td></tr><tr><td>500</td><td>0.259</td><td>0.349</td><td>0.303</td><td>0.134</td></tr><tr><td>750</td><td>0.206</td><td>0.266</td><td>0.168</td><td>0.082</td></tr><tr><td>1000</td><td>0.167</td><td>0.213</td><td>0.166</td><td>0.079</td></tr><tr><td>2000</td><td>0.140</td><td>0.184</td><td>0.147</td><td>0.074</td></tr><tr><td>4000</td><td>0.094</td><td>0.161</td><td>0.076</td><td>0.057</td></tr></table>

CV vs Rarity. We keep the same $K = 1 0 0 0$ and $N = 1 0 0 0$ and ablate the rarities using approxiamte guidance and learned scores. We discount the number of particles available based on number of function evaluations compared to DireSMC base. For Manshausen et al. (2026) this is $3 . 5 \times$ less particles (286) on this problem. Table 26 summarises these results, where we find that at matched compute all variants of DireSMC outperform Manshausen et al. (2026).

Bias vs K. We measure the discretisation bias using analytical scores, Doob’s transition kernels of DireSMC with no resampling, selected to isolate the errors introduced in Eq. 124. Table 27 summarises these results.

Table 26: CV of $\hat { p }$ relative to models threshold exceedance probability $p ^ { \star }$ (Table 7). $\hat { p }$ across rarities at equal compute (Fig. 2, middle), with $K = 1 0 0 0 .$ , the learned score and 24 seeds. We use $N = 1 0 0 0$ and Manshausen et al. (2026) use $n = N / 3 . 5 = 2 8 6$ (Table 24). All variants of DireSMC outperform Manshausen et al. (2026).
<table><tr><td colspan="3"></td><td colspan="3">DireSMC</td></tr><tr><td>T</td><td> $p ^ { \star }$ </td><td>Manshausen et al. (2026)</td><td>base</td><td>+ resampling</td><td>+ MADM</td></tr><tr><td>7</td><td> $7 . 1 3 \times 1 0 ^ { - 3 }$ </td><td>0.129</td><td>0.070</td><td>0.093</td><td>0.060</td></tr><tr><td>8</td><td> $1 . 3 8 \times 1 0 ^ { - 3 }$ </td><td>0.167</td><td>0.092</td><td>0.093</td><td>0.064</td></tr><tr><td>9</td><td> $1 . 8 7 \times 1 0 ^ { - 4 }$ </td><td>0.221</td><td>0.127</td><td>0.159</td><td>0.090</td></tr><tr><td>10</td><td> $1 . 7 2 \times 1 0 ^ { - 5 }$ </td><td>0.349</td><td>0.213</td><td>0.166</td><td>0.079</td></tr></table>

Table 27: Discretisation bias against the step count K (Fig. 2, right), shown as the mean $\hat { p } / p ^ { \star } \pm \mathrm { s . e } .$ over 24 seeds. Computed using analytical socres, Doob’s transition kernels and withut resampling at N = 1000. $p ^ { \star }$ is $p _ { \mathrm { d a t a } } [ E ]$ . Manshausen et al. (2026) are given the exact divergence.
<table><tr><td>T</td><td> $p ^ { \star }$ </td><td> $K$ </td><td>Manshausen et al. (2026)</td><td>DireSMC</td></tr><tr><td rowspan="4">7</td><td rowspan="4"> $6 . 8 8 \times 1 0 ^ { - 3 }$ </td><td>50</td><td> $1 . 4 6 6 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 5 0 \mathop { \pm } 0 . 0 0 6 }$ </td></tr><tr><td>100</td><td> $1 . 2 3 3 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 7 2 \pm 0 . 0 0 3 }$ </td></tr><tr><td>200</td><td> $1 . 1 2 8 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 8 1 \pm 0 . 0 0 3 }$ </td></tr><tr><td>500</td><td> $1 . 0 6 9 \pm 0 . 0 0 1$ </td><td> ${ \bf 0 . 9 9 0 \pm 0 . 0 0 2 }$ </td></tr><tr><td rowspan="4">8</td><td rowspan="4"> $1 . 2 8 \times 1 0 ^ { - 3 }$ </td><td>50</td><td> $1 . 7 0 1 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 9 5 7 \pm 0 . 0 0 5 }$ </td></tr><tr><td>100</td><td> $1 . 3 3 6 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 7 0 \pm 0 . 0 0 3 }$ </td></tr><tr><td>200</td><td> $1 . 1 7 8 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 8 0 \pm 0 . 0 0 3 }$ </td></tr><tr><td>500</td><td> $1 . 0 9 1 \pm 0 . 0 0 1$ </td><td> ${ \bf 0 . 9 9 0 \pm 0 . 0 0 2 }$ </td></tr><tr><td rowspan="4">9</td><td rowspan="4"> $1 . 6 2 \times 1 0 ^ { - 4 }$ </td><td>50</td><td> $2 . 0 4 9 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 9 6 3 \pm 0 . 0 0 6 }$ </td></tr><tr><td>100</td><td> $1 . 4 7 3 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 9 7 2 \pm 0 . 0 0 4 }$ </td></tr><tr><td>200</td><td> $1 . 2 4 0 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 8 1 \pm 0 . 0 0 3 }$ </td></tr><tr><td>500</td><td> $1 . 1 1 7 \pm 0 . 0 0 1$ </td><td> ${ \bf 0 . 9 9 0 \pm 0 . 0 0 2 }$ </td></tr><tr><td rowspan="4">10</td><td rowspan="4"> $1 . 4 0 \times 1 0 ^ { - 5 }$ </td><td></td><td></td><td></td></tr><tr><td>50 100</td><td> $2 . 5 5 7 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 9 7 8 \pm 0 . 0 0 7 }$ </td></tr><tr><td>200</td><td> $1 . 6 5 3 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 9 7 4 \pm 0 . 0 0 4 }$ </td></tr><tr><td>500</td><td> $1 . 3 1 7 \pm 0 . 0 0 2$   $1 . 1 4 7 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 9 8 1 \pm 0 . 0 0 3 }$   ${ \bf 0 . 9 9 0 \pm 0 . 0 0 2 }$ </td></tr></table>