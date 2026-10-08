# Neural Sampling with Reweighted Normalizing Flows via the Wasserstein–Fisher–Rao JKO Scheme

Chenguang Duan<sup>1</sup>, Johannes Hertrich<sup>2</sup>, and Gabriele Steidl<sup>3</sup>

<sup>1</sup>Institut für Geometrie und Praktische Mathematik, RWTH Aachen University,

duan@igpm.rwth-aachen.de

<sup>2</sup>Institute of Computer Science, University of Göttingen,

johannes.hertrich@uni-goettingen.de

<sup>3</sup>Institute of Mathematics, TU Berlin, steidl@math.tu-berlin.de

October 8, 2026

Abstract. We propose a neural algorithm for sampling from distributions specified by unnormalized Boltzmann densities. Our approach is based on the Jordan–Kinderlehrer–Otto scheme for the Kullback–Leibler divergence in the Wasserstein–Fisher–Rao geometry (WFR JKO scheme). Our contributions are twofold. First, we prove that, for any fixed step size, the exact WFR JKO iterates converge exponentially fast to the target as the number of iterations tends to infinity. Notably, this result requires no structural assumptions on the target, such as log-concavity or a logarithmic Sobolev inequality. Second, we develop a neural implementation of the WFR JKO scheme that parametrizes its transport and reaction components using reweighted normalizing flows. Numerical experiments on challenging multimodal targets demonstrate the promising performance of the proposed method.

Keywords: Wasserstein–Fisher–Rao distance, Hellinger–Kantorvich distance, Sampling from Boltzmann densities, JKO scheme, Metropolis–Hastings algorithm

## 1 Introduction

Let $\pi ( x ) = \exp ( - V ( x ) )$ be a strictly positive, integrable Boltzmann density on $\mathbb { R } ^ { d }$ . We identify π with the corresponding finite measure and denote its total mass by $Z : = \pi ( \mathbb { R } ^ { d } )$ Our aim is to generate samples from the probability distribution $\tilde { \pi } = \pi / Z$ , using the unnormalized density π without prior knowledge of Z. Such sampling problems arise in Bayesian inverse problems and data assimilation [6, 7, 17, 18, 24, 37, 43, 45], statistical mechanics [29, 40, 42], and machine learning [13, 27]. They are particularly challenging in high dimensions and when the target has several well-separated modes. In these settings, accurate sampling requires both exploration of the relevant regions of the state space and recovery of their relative probability masses.

A variational approach to this problem is to formulate sampling as the minimization of a discrepancy functional over a space of measures. A natural objective is the Kullback– Leibler (KL) divergence, whose unique minimizer is the target distribution. Equipping the space of measures with a metric determines a gradient flow of this objective and thereby provides a starting point for constructing sampling algorithms. In particular, the gradient flow of $\operatorname { K L } ( { \cdot } \parallel { \tilde { \pi } } )$ in the quadratic Wasserstein geometry is the Fokker–Planck equation associated with overdamped Langevin dynamics [28]. The law of the Langevin difusion thus decreases the KL divergence through the transport of probability mass.

This common continuous-time description leads to two discretization routes. On the particle level, an explicit Euler–Maruyama discretization of the Langevin difusion yields the unadjusted Langevin algorithm [18], with stochastic-gradient variants supporting applications involving large datasets [53]. On the level of measures, the Jordan– Kinderlehrer–Otto (JKO) scheme [28] provides an implicit variational discretization. Each step minimizes the KL divergence together with a squared Wasserstein-distance penalty relative to the preceding iterate. These variational problems can be approximated using normalizing flows, leading to neural JKO sampling methods [25]. Continuous normalizing flows are particularly suitable for this purpose, since they describe both transport and density evolution through ordinary diferential equations. Related implementations of the Wasserstein JKO scheme have also been developed for generative modeling [44, 52, 54].

Despite their diferent numerical realizations, both approaches retain the transport geometry of the underlying Wasserstein gradient flow. For multimodal targets, this geom etry can lead to slow redistribution of mass: transferring probability between separated modes requires crossing regions of low density. The resulting metastability is reflected in the convergence analysis. For nonconvex potentials, the KL divergence need not be geodesically convex in the Wasserstein metric [11], and quantitative convergence estimates commonly rely on functional inequalities, such as a logarithmic Sobolev inequality (LSI) [2, 10, 51]. The associated constants can deteriorate substantially as the modes become more strongly separated. Consequently, changing from an explicit to an implicit discretization does not, by itself, remove the dificulty of sampling multimodal targets. This motivates augmenting local transport with mechanisms for global redistribution, as in the reweighting and resampling steps of sequential Monte Carlo [16], the learned nonlocal proposals of adaptive Monte Carlo [20], and the importance corrections of neural JKO sampling [25].

The Wasserstein–Fisher–Rao (WFR) metric, also known as the Hellinger–Kantorovich metric, incorporates transport and mass adjustment into a single geometric framework [12, 31, 32]. Its Wasserstein component moves mass through the state space, while its Fisher– Rao component allows mass to be created or removed. The relative masses of separated modes can therefore be adjusted without transporting all of the redistributed mass through the intervening low-density regions. Transport and reaction play complementary roles: the former supports spatial exploration, whereas the latter corrects the allocation of mass. Beyond this geometric intuition, it has been shown that exponential convergence of the WFR gradient flow can be established without assuming log-concavity or a logarithmic Sobolev inequality for the target, provided suitable conditions on the initial density ratio hold [15, 35]. These results demonstrate that reaction can yield convergence rates independent of potential barriers and provide a theoretical motivation for using WFR-type geometries in multimodal sampling.

An explicit particle realization of this principle leads to Langevin sampling with birth–death mechanisms [34, 35, 49]. In the probability-preserving setting, the underlying dynamics correspond to a spherical counterpart of the WFR geometry. The resulting algorithms combine Langevin updates with particle duplication and removal according to the discrepancy between the current and target densities. Although these updates can accelerate redistribution among modes, their implementation introduces a further dificulty: the birth–death rates depend on the evolving density, which is not directly available from an empirical particle measure. Kernel-based approximations introduce bandwidth dependence and become dificult to resolve accurately in high dimensions [35].

Thus, exploiting the favorable properties of the continuum dynamics requires a numerical representation that also provides access to the evolving density.

These considerations motivate an implicit variational approach combined with a tractable density representation. In this paper, we develop a neural sampling method based on the WFR JKO scheme. We work with the unbalanced WFR geometry on finite positive measures and minimize the extended KL divergence KL(· ∥ π). Its minimizer is the unnormalized measure π, so the scheme approximates both the shape and the total mass of the target. The unknown normalizing constant appears only as an additive term in the objective and is therefore unnecessary for computing the variational updates. WFR gradient flows and their variational approximations have been studied in [19, 21, 41]; here, we analyze the long-time behavior of the discrete scheme at a fixed time step.

Our main theoretical result establishes geometric decay of the KL divergence along the exact WFR-JKO iterates. The contraction factor depends only on the time step and the WFR parameter, independently of the target measure. In particular, the estimate requires neither convexity of the potential nor a logarithmic Sobolev inequality for the target.

To make the variational steps computationally accessible, we parametrize both the transport velocity and the reaction rate by neural networks. The characteristic representation of the continuity equation with reaction then combines a continuous normalizing flow with a positive, learned reweighting function. We derive a computable training objective, a Monte Carlo approximation, and a stochastic trace estimator for the divergence term. This representation propagates particle positions, density values, and total mass, thereby avoiding the kernel density estimation required by standard birth–death particle implementations.

The practical algorithm combines these learned transport–reaction steps with resampling to obtain an unweighted particle ensemble. To mitigate the loss of diversity caused by particle duplication, we incorporate Metropolis–Hastings rejuvenation steps targeting the normalized current model distribution [23, 25]. The availability of model density evaluations allows proposals in both the data space and the latent space of the learned flow. Numerical experiments on multimodal targets compare the proposed method with established MCMC and neural sampling approaches and examine the efect of these rejuvenation strategies.

Contributions. The main contributions of this work are as follows:

1. We prove geometric decay of the extended KL divergence along the exact WFR JKO iterates for strictly positive, integrable targets. For each fixed time step, the contraction factor depends only on the time step and the WFR parameter, without requiring log-concavity or a logarithmic Sobolev inequality for the target.

2. We derive a computable WFR JKO objective by combining continuous normalizing flows with learned reaction rates. This representation jointly describes spatial transport and mass adjustment, provides access to density values and total mass, and avoids kernel density estimation.

3. We combine the learned transport–reaction updates with resampling and Metropolis– Hastings rejuvenation to mitigate particle degeneracy. We consider proposals in both the data and latent spaces and evaluate the resulting method on multimodal targets through comparisons with established MCMC and neural sampling approaches.

Outline of the paper. In Section 2, we collect basic results on the Wasserstein metric for probability measures as well as on the Fisher–Rao metric and the WFR metric, also known as Hellinger–Kantorivich metric, for positive measures. In particular, we rely on their dynamic formulations to formulate our loss function later in (4.1). In Section $^ { 3 , }$ we deal with the WFR–JKO scheme for minimizing the KL functional on the space of positive measures. As main result, we prove the exponential convergence of the scheme, see Theorem 3.5. In Section 4, we reformulate the WFR–JKO step minimization for an appropriate admissibility class of velocities and mobility functions. We provide a computable formulation for our objective based on ideas from the normalizing flows and present the corresponding algorithm. Since the resampling step unavoidably causes particle degeneracy we propose to rejuvenate the ensemble while preserving its target by applying a Metropolis–Hastings step after each resampling step in Section 5. Finally, we present our promising numerical results in Section 6. The appendices contain proofs of auxiliray results and the pseudo-code of the algorithms.

## 2 Preliminaries

Let $\mathcal { M } ( \mathbb { R } ^ { d } )$ denote the set of finite positive Borel measures on $\mathbb { R } ^ { d } , \mathcal { P } ( \mathbb { R } ^ { d } )$ the set of probability measure, and $\mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ its subset of measures with finite second moments. For convenience, we just write R instead of $\scriptstyle \int _ { \mathbb { R } ^ { d } }$ in the following.

The Kullback-Leibler (KL) divergence is defined, for $\mu , \nu \in \mathcal { M } ( \mathbb { R } ^ { d } )$ , as

$$
\mathrm { K L } ( \mu \parallel \nu ) = \int \log \left( \frac { \mathrm { d } \mu } { \mathrm { d } \nu } \right) ~ \mathrm { d } \mu - \mu ( \mathbb { R } ^ { d } ) + \nu ( \mathbb { R } ^ { d } )
$$

if $\mu$ is absolutely continuous with respect to $\nu ,$ so that the Radon-Nikodym derivative $\frac { \mathrm { d } \mu } { \mathrm { d } \nu }$ exits and +∞ otherwise. We have $\mathrm { K L } ( \mu , \nu ) \geq 0$ with equality if and only if $\mu = \nu$ We will need the following proposition which states the unit ball in the KL divergence is sequentially compact with respect to weak convergence.

Proposition 2.1. Let $r > 0$ and $\pi \in \mathcal { M } ( \mathbb { R } ^ { d } )$ . Then, the set $\{ \rho \in \mathcal { M } ( \mathbb { R } ^ { d } ) : \mathrm { K L } ( \rho \| \pi ) \leq r \}$ is weakly sequentially compact.

The proof is a standard consequence of Prokhorov’s theorem and the weak lower semicontinuity of KL. For convenience we give it in Appendix.

Let $C _ { b } ( \mathbb { R } ^ { d } )$ denote the space of continuous bounded functions on $\mathbb { R } ^ { d }$ . A sequence $( \mu _ { k } ) _ { k }$ in $\mathcal { M } ( \mathbb { R } ^ { d } )$ converges weakly to $\mu \in \mathcal { M } ( \mathbb { R } ^ { d } )$ , written $\mu _ { n }  \mu , { \mathrm { i f ~ } } \lint \varphi { \mathrm { ~ d } } \mu _ { n } \to \ j \varphi { \mathrm { ~ d } } \mu$ for all $\varphi \in C _ { b } ( \mathbb { R } ^ { d } )$

By $L ^ { { \boldsymbol { p } } } ( \mu ) , 1 \leq { \boldsymbol { p } } < \infty$ , we denote the Banach space of (equivalence classes) of functions which p-the power is µ-integrable with the corresponding norm. We write $L ^ { p } ( \mu , \mathbb { R } ^ { d } )$ for the corresponding space of p-integrable vector fields.

In the following, we recall the Wasserstein-2, the Fisher–Rao, and the Wasserstein– Fisher–Rao distances on these spaces, together with the gradient flows they induce.

2.1 Wasserstein Metric. The Wasserstein(-2) distance between $\mu , \nu \in \mathcal { P } _ { 2 } ( \mathbb { R } ^ { d } )$ is defined as

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) : = \operatorname* { i n f } _ { \gamma \in \Gamma ( \mu , \nu ) } \int _ { \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } } | x - y | ^ { 2 } ~ \mathrm { d } \gamma ( x , y ) ,
$$

where $\Gamma ( \mu , \nu ) : = \{ \gamma \in \mathscr { P } _ { 2 } ( \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } : ( P _ { 1 } ) _ { \# } \gamma = \mu , ( P _ { 2 } ) _ { \# } \gamma = \nu \}$ with projections $P _ { 1 } ( x , y ) =$ $x , P _ { 2 } ( x , y ) = y$ and the push-forward measures $( P _ { i } ) _ { \# } \gamma : = \gamma \circ P _ { i } , i = 1 , 2$ . equipped with $W _ { 2 }$ , the set $\mathcal { P } _ { 2 }$ becomes a metric space. Further, we have that $W _ { 2 } ( \mu _ { k } , \mu ) \to 0$ as $k \to \infty$ if and only if $\mu _ { k } \to \mu$ and $\begin{array} { r } { \int | x | ^ { 2 } ~ \mathrm { d } \mu _ { k } \to \int | x | ^ { 2 } ~ \mathrm { d } \mu } \end{array}$ as $k \to \infty$

A weakly continuous curve $\mu _ { t } : ( 0 , 1 ) \to \mathcal P _ { 2 } ( \mathbb R ^ { d } )$ is absolutely continuous if and only if there exists a Borel measurable vector field $v : ( 0 , 1 ) \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ such that $\| v _ { t } \| _ { L ^ { 2 } ( \mu _ { t } , \mathbb { R } ^ { d } ) } \in L ^ { 1 } ( 0 , 1 )$ and $\left( \mu _ { t } , v _ { t } \right)$ fulfills the continuity equation

$$
\begin{array} { r } { \partial _ { t } \mu _ { t } + \operatorname { d i v } \left( \mu _ { t } v _ { t } \right) = 0 } \end{array}\tag{2.1}
$$

in a distributional sense. The Benamou–Brenier formula [3] recasts the Wasserstein metric as the minimal kinetic energy needed to connect two probability measures as

$$
W _ { 2 } ^ { 2 } ( \mu , \nu ) = \operatorname* { i n f } _ { ( \mu _ { t } , \nu _ { t } ) } \left\{ \int _ { 0 } ^ { 1 } \int | v _ { t } | ^ { 2 } \mathrm { d } \mu _ { t } \mathrm { d } t ; \frac { \partial _ { t } \mu _ { t } + \mathrm { d i v } ( \mu _ { t } v _ { t } ) } { \mu _ { 0 } = \mu , \mu _ { 1 } = \nu } \right\} .\tag{2.2}
$$

The following proposition assiciates absolutely continuous curves with ODE solutions, see [1, Proposition 8.1.8, Lemma 8.1.4] in a one-to-one fashion.

Proposition 2.2. Let $v \colon [ 0 , 1 ] \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d } \ f u l f i l l$

$$
\int _ { I } \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } | v _ { t } ( x ) | + \mathrm { L i p } ( v _ { t } ) \ \mathrm { d } t < \infty .\tag{2.3}
$$

Then, for every $\boldsymbol { x } \in \mathbb { R } ^ { d }$ , the ODE

$$
\partial _ { t } \phi _ { t } = v _ { t } ( \phi _ { t } ) , \quad \phi ( 0 , x ) = x\tag{2.4}
$$

admits unique solution $\phi _ { t } = \phi ( t , x ) : I \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ on [0, 1] and the map $\phi _ { t } \colon \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ 2 $t \in [ 0 , 1 ]$ is a C<sup>1</sup>-difeomorphisms and its determinant of the Jacobian det $\nabla \phi _ { t } ~ f u l f i l l s$ the Liouville formula

$$
\operatorname* { d e t } \nabla \phi _ { t } ( x ) = \exp \Bigl ( \int _ { 0 } ^ { t } \operatorname { d i v } v _ { s } ( \phi _ { s } ( x ) ) \mathrm { d } s \Bigr ) .\tag{2.5}
$$

The curve

$$
\mu _ { t } : = \phi ( t , \cdot ) _ { \sharp } \mu _ { 0 }
$$

is the unique weakly continuous solution of the continuity equation (2.1) with the above vector $~ f i e l d .$

2.2 Fisher–Rao Metric. The Fisher–Rao metric for $\mu , \nu \in \mathcal { M } ( \mathbb { R } ^ { d } )$ is defined by

$$
\operatorname { F R } ^ { 2 } ( \mu , \nu ) : = 4 \int \left( \sqrt { \frac { \mathrm { d } \mu } { \mathrm { d } ( \mu + \nu ) } } - \sqrt { \frac { \mathrm { d } \nu } { \mathrm { d } ( \mu + \nu ) } } \right) ^ { 2 } \mathrm { d } ( \mu + \nu ) .\tag{2.6}
$$

The Fisher–Rao curve $\mu _ { t } : ( 0 , 1 ) \to \mathcal { M } ( \mathbb { R } ^ { d } )$ is determined by the reaction equation

$$
\partial _ { t } \mu _ { t } = g _ { t } \mu _ { t } ,
$$

where the measurable function $g _ { t } \colon  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ is the pointwise rate of mass creation. Analogously to (2.2), the metric (2.6) admits the dynamical representation

$$
\mathrm { F R } ^ { 2 } ( \mu , \nu ) = \operatorname* { i n f } _ { ( \mu _ { t } , g _ { t } ) } \left\{ \int _ { 0 } ^ { 1 } \int g _ { t } ^ { 2 } \mathrm { d } \mu _ { t } \mathrm { d } t \colon \frac { \partial _ { t } \mu _ { t } = g _ { t } \mu _ { t } , } { \mu _ { 0 } = \mu , ~ \mu _ { 1 } = \nu } \right\} .\tag{2.7}
$$

2.3 Wasserstein–Fisher–Rao Metric. Combining the transport and reaction geometries yields a metric on $\mathcal { M } ( \mathbb { R } ^ { d } )$ that allows mass to be simultaneously transported and created or destroyed. The corresponding dynamics obeys the continuity equation with reaction

$$
\begin{array} { r } { \partial _ { t } \mu _ { t } + \mathrm { { \small ~ d i v } } \left( \mu _ { t } v _ { t } \right) = g _ { t } \mu _ { t } . } \end{array}\tag{2.8}
$$

in the sense of distributions. We assume that the coeficients v and g are Hellinger-Kantorovich regular, namely, they are Borel measurable and satisfy $\| v _ { t } \| _ { L ^ { 2 } ( \mu _ { t } , \mathbb { R } ^ { d } ) } \in L ^ { 1 } ( 0 , 1 )$ and $\| g _ { t } \| _ { L ^ { 2 } ( \mu _ { t } ) } \in L ^ { 1 } ( 0 , 1 )$ .

The Wasserstein–Fisher–Rao (WFR) metric, also known as Hellinger–Kantorovich metric [12], with parameter $\alpha > 0$ is defined as the minimal total kinetic plus reactive action required to interpolate between $\mu , \nu \in \mathcal { M } ( \mathbb { R } ^ { d } )$

$$
\begin{array}{c} \mathrm { W F R } _ { \alpha } ^ { 2 } ( \mu , \nu ) = \operatorname* { i n f } _ { ( \mu _ { t } , \nu _ { t } , g _ { t } ) } \Biggl \{ \int _ { 0 } ^ { 1 } \int | v _ { t } | ^ { 2 } + \frac { 1 } { \alpha } g _ { t } ^ { 2 } \ : \mathrm { d } \mu _ { t } \ : \mathrm { d } t : \ : \frac { \partial _ { t } \mu _ { t } + \ : \mathrm { d i v } \left( \mu _ { t } v _ { t } \right) = g _ { t } \mu _ { t } , } \\ { \mu _ { 0 } = \mu , \ : \mu _ { 1 } = \nu } \end{array} \Biggr \} .
$$

Setting $g \equiv 0$ recovers the Benamou-Brenier formula (2.2), while setting $v \equiv 0$ gives the Fisher–Rao representation (2.7); in this sense $\mathrm { W F R } _ { \alpha }$ interpolates between the Wasserstein and Fisher–Rao metrics as α varies. We note that changing the parameter α in the Wasserstein–Fisher–Rao metric corresponds to rescaling, see also [33, p. 13]. More precisely, it follows directly from the its definition that

$$
\operatorname { W F R } _ { \alpha _ { 1 } } ^ { 2 } ( \mu , \nu ) = \frac { \alpha _ { 2 } } { \alpha _ { 1 } } \operatorname { W F R } _ { \alpha _ { 2 } } ^ { 2 } \big ( \big ( s _ { \alpha _ { 1 } / \alpha _ { 2 } } \big ) _ { \# } \mu , \big ( s _ { \alpha _ { 1 } / \alpha _ { 2 } } \big ) _ { \# } \nu \big ) , \quad s _ { \alpha } ( x ) = \sqrt { \alpha } x .\tag{2.9}
$$

Further, the WFR metric metrizes weak convergence:

Proposition 2.3. [32, Thm 7.15] Let $\mu _ { k } \in \mathcal { M } ( \mathbb { R } ^ { d } )$ . Then $\mathrm { W F R } _ { \alpha } ( \mu _ { k } , \mu ) \to 0$ if and only $i f \mu _ { k }  \mu \ a s \ k  \infty$

In [32], Liero, Mielke and Savare proved that the WFR metric is actually equivalent to an unbalanced optimal transport problem called logartihmic entropy transport. More precisely, we have the following theorem which was shown for $\alpha = 4$ and generalizes for arbitrary $\alpha > 0$ by the rescaling property (2.9).

Theorem 2.4. [32, Thm 7.20] Let $\mu , \nu \in \mathcal { M } ( \mathbb { R } ^ { d } )$ . Then the WFR metric coincides with the logartihmic entropy transport (LET) problem given by

$$
\mathrm { W F R } _ { \alpha } ^ { 2 } ( \mu , \nu ) = \frac { 4 } { \alpha } \operatorname* { m i n } _ { \gamma \in \mathcal { M } ( \mathbb { R } ^ { d } \times \mathbb { R } ^ { d } ) } \{ \mathrm { K L } ( \gamma _ { 1 } \| \mu ) + \mathrm { K L } ( \gamma _ { 2 } \| \nu ) + \int \ell _ { \alpha } \mathrm { d } \gamma \} ,\tag{2.10}
$$

where $\gamma _ { 1 }$ and $\gamma _ { 2 }$ are the marginals of γ and $\ell _ { \alpha }$ is the cost function

$$
\ell _ { \alpha } ( x , y ) : = \left\{ \begin{array} { l l } { - \log ( \cos ^ { 2 } ( \frac { \sqrt { \alpha } } { 2 } \| x - y \| ) , } & { i f \| x - y \| < \frac { \pi } { \sqrt { \alpha } } , } \\ { + \infty , } & { o t h e r w i s e . } \end{array} \right.
$$

The following proposition is a direct conclusion of [5, Propositions 2.3 and 2.4] and was shown for bounded $g$ instead of also in [36].

Proposition 2.5. Let $v \colon [ 0 , 1 ] \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d }$ satisfy (2.3), and let $g \colon [ 0 , 1 ] \times  { \mathbb { R } } ^ { d } \to  { \mathbb { R } }$ be measurable with

$$
\int _ { 0 } ^ { 1 } \operatorname* { s u p } _ { x \in \mathbb { R } ^ { d } } | g _ { t } ( x ) | + \mathrm { L i p } ( g _ { t } ) \mathrm { d } t < \infty\tag{2.11}
$$

as well as

(2.12)

Let $\phi \colon [ 0 , 1 ] \times  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ be the solution of the ODE (2.4). Then the curve

$$
\mu _ { t } : = \phi ( t , \cdot ) _ { \# } ( w _ { t } \mu _ { 0 } ) , \quad w _ { t } ( x ) : = \exp \left( \int _ { 0 } ^ { t } g _ { s } \left( \phi ( s , x ) \right) \mathrm { d } s \right) \quad \mu _ { 0 } - a . e .\tag{2.13}
$$

is the unique weakly continuous solution of the continuity equation with reaction (2.8).

## 3 Wasserstein–Fisher–Rao JKO Scheme

In this section, we consider the Wasserstein–Fisher–Rao Jordan-Kinderlehrer-Otto scheme for minimizing functionals on $\mathcal { M } ( \mathbb { R } ^ { d } )$ and prove exponential convergence for the KL functional $F = \mathrm { K L } ( \cdot | | \pi )$ , where $\pi > 0$

3.1 WFR JKO Scheme. Let $\mathcal { F } \colon \mathcal { M } ( \mathbb { R } ^ { d } )  [ 0 , + \infty ]$ be a proper functional which we intend to minimize by an implicit gradient descent algorithm in $\mathcal { M } ( \mathbb { R } ^ { d } )$ equipped with the WFR metric. The corresponding scheme in the Wasserstein space is known as Jordan-Kinderleher-Otto (JKO) scheme [28] and we adopt the notation for current geomtry. Given $\mu _ { k } \in \mathcal { M } ( \mathbb { R } ^ { d } )$ , one step of the WFR JKO scheme is defined, whenever a minimizer exists, by

$$
\mu _ { k + 1 } \in \underset { \mu \in \mathcal { M } ( \mathbb { R } ^ { d } ) } { \arg \operatorname* { m i n } } \biggl \{ \frac { 1 } { 2 h } \mathrm { W F R } _ { \alpha } ^ { 2 } ( \mu , \mu _ { k } ) + \mathcal { F } ( \mu ) \biggr \} .
$$

In this paper, we are only interested in

$$
\mathcal { F } = \operatorname { K L } ( \cdot \| \pi ) .
$$

Then the sequence in (3.1) is well-defined. For convenience, we add the small proof.

Lemma 3.1. Let $\pi , \mu _ { k } \in \mathcal { M } ( \mathbb { R } ^ { d } )$ with $\mathrm { K L } ( \mu _ { k } \| \pi ) < \infty$ . Then, the minimum in (3.1) is attained.

Proof. Let $\phi _ { h } ( \mu _ { k } )$ be the infimum in (3.1), and let $\nu _ { n } \in \mathcal { M } ( \mathbb { R } ^ { d } )$ constitute a minimizing sequence such that

$$
\frac { 1 } { 2 h } \mathrm { W F R } _ { \alpha } ^ { 2 } ( \nu _ { n } , \mu _ { k } ) + \mathrm { K L } ( \nu _ { n } \| \pi ) \leq \phi _ { h } ( \mu _ { k } ) + \frac { 1 } { n } .
$$

By Proposition 2.1, we know, up to subsequences, that $\nu _ { n }$ converges weakly to some $\nu \in \mathcal { M } ( \mathbb { R } ^ { d } )$ . Further, we know by Proposition 2.3 that $\mathrm { W F R } _ { \alpha } ^ { 2 } ( \cdot , \mu _ { k } )$ is continuous wrt weak convergence. Further, $\operatorname { K L } ( \cdot \| \pi )$ is lower semicontinuous wrt weak convergence. Thus

$$
\frac { 1 } { 2 h } \mathrm { W F R } _ { \alpha } ^ { 2 } ( \nu , \mu _ { k } ) + \mathrm { K L } ( \nu \| \pi ) = \phi _ { h } ( \mu _ { k } ) ,
$$

which means that $\begin{array} { r } { \nu \in \arg \operatorname* { m i n } _ { \mu } \frac { 1 } { 2 h } \mathrm { W F R } _ { \alpha } ^ { 2 } ( \mu , \mu _ { k } ) + \mathrm { K L } ( \mu \| \pi ) } \end{array}$

In the rest of this paper, let $\pi > 0$ be integrable on $\mathbb { R } ^ { d } .$ . With slight abuse of notation we identify the function π with the corresponding measure $\pi$ dx. For such $\pi _ { : }$ the WFR JKO scheme becomes

$$
\rho _ { k + 1 } \in \underset { \rho \in \mathcal { M } ( \mathbb { R } ^ { d } ) } { \arg \operatorname* { m i n } } J ( \rho ) , \quad J ( \rho ) : = \frac { 1 } { 2 h } \mathrm { W F R } _ { \alpha } ^ { 2 } ( \rho , \rho _ { k } ) + \mathrm { K L } ( \rho \| \pi ) .\tag{3.1}
$$

Note that by this definition $\mathrm { K L } ( \rho _ { k + 1 } \| \pi )$ is finite for $k \geq 0$ such that $\rho _ { k + 1 }$ is absolutely continuous wrt $\pi$ and thus also wrt the Lebesgue measure. Therefore, we rely on densities $\rho .$

Remark 3.2 (Gradient Flows). Several papers study gradient flows of the KL divergence or other functionals with respect to the WFR metric, see, e.g., [14, 21, 35, 41, 55]. Formally, such gradient flows can be defined as solutions of the reaction difusion equation

$$
\partial _ { t } \mu _ { t } - \mathrm { { \mathop { d i v } } } \left( \mu _ { t } \nabla F ^ { \prime } ( \mu _ { t } ) \right) = - \alpha \mu _ { t } F ^ { \prime } ( \mu _ { t } ) ,\tag{3.2}
$$

where $\begin{array} { r } { F ^ { \prime } ( \mu ) = \frac { \delta \mathrm { K L } ( \cdot , \pi ) } { \delta \mu } = \log \left( \frac { \mu } { \pi } \right) + 1 } \end{array}$ is the first variation of the KL divergence. The sequence $( \rho _ { k } ^ { h } ) _ { k }$ generated by the WFR JKO scheme (3.1) for some step size h can be viewed as a time-discretized version of such curves. More precisely, the authors of [19, 30] study limits of the curves $\mu _ { t } ^ { h } \colon [ 0 , \infty )  \mathcal { M } ( \mathbb { R } ^ { d } )$ defined as $\mu _ { t } ^ { h } = \rho _ { k } ^ { h }$ for $t \in [ k h , ( k + 1 ) h )$ In particular, it was shown by Fleißner [19, Example 4.12] under mild assumptions that $\mu _ { t } ^ { h }$ converges weakly to a weak solution of (3.2) as $h  0$ . In contrast, we merely study the convergence of $( \rho _ { k } ^ { h } ) _ { k }$ for fixed h as $k  \infty$

3.2 Convergence of WFR JKO for $k \to \infty$ . In this subsection, we prove exponential convergence of the sequence $\rho _ { k }$ generated by the WFR JKO scheme (3.1) towards $\pi$ for $k \to \infty$ . We use the notation

$$
I ( \rho ) = \int \rho \log ^ { 2 } \left( \frac { \rho } { \pi } \right) \mathrm { d } x .
$$

Then, the following descent lemma holds true. The main idea of the proof is to bound the WFR metric by the FR metric and to insert the optimality condition of $\rho _ { k + 1 }$ along the geometric interpolation between $\rho _ { k + 1 }$ and π.

Lemma 3.3. Let $( \rho _ { k } ) _ { k }$ be generated by (3.1). Then, we have for any $k \geq 0$ that

$$
\alpha I ( \rho _ { k + 1 } ) \leq \frac { 1 } { h ^ { 2 } } \mathrm { W F R } ^ { 2 } ( \rho _ { k + 1 } , \rho _ { k } ) .
$$

Proof. For $s \in [ 0 , 1 ]$ , let $\mu _ { s } : = \rho _ { k + 1 } ^ { 1 - s } \pi ^ { s }$ . Then $\mu _ { s t } , \ t \in [ 0 , 1 ]$ fulfills $( \mu _ { s t } ) | _ { t = 0 } = \rho _ { k + 1 }$ and $( \mu _ { s t } ) | _ { t = 1 } = \mu _ { s }$ and

$$
\partial _ { t } \mu _ { s t } = s \log \left( \frac { \pi } { \rho _ { k + 1 } } \right) \mu _ { s t } = g _ { t } \mu _ { s t } , g _ { t } : = s \log ( a ) , a : = \frac { \pi } { \rho _ { k + 1 } } .
$$

Then we conclude that the WFR distance between $\rho _ { k + 1 }$ and $\mu _ { s }$ is bounded by

$$
\begin{array} { l } { \displaystyle \operatorname { W F R } _ { \alpha } ^ { 2 } ( \rho _ { k + 1 } , \mu _ { s } ) \leq \int _ { 0 } ^ { 1 } \int \frac { \mu _ { s t } } { \alpha } g _ { t } ^ { 2 } \mathrm { d } x \mathrm { d } t = \frac { s ^ { 2 } } { \alpha } \int _ { 0 } ^ { 1 } \int \pi ^ { s t } \rho _ { k + 1 } ^ { 1 - s t } \log ^ { 2 } ( a ) \mathrm { d } x \mathrm { d } t } \\ { \displaystyle \qquad = \frac { s ^ { 2 } } { \alpha } \int \rho _ { k + 1 } \log ^ { 2 } ( a ) \int _ { 0 } ^ { 1 } a ^ { s t } \mathrm { d } t \mathrm { d } x = \frac { s } { \alpha } \int \rho _ { k + 1 } \log ( a ) \left( a ^ { s } - 1 \right) \mathrm { d } x . } \end{array}
$$

Next, we consider the function $H ( s ) : = \mathrm { K L } ( \rho _ { k + 1 } \| \pi ) - \mathrm { K L } ( \mu _ { s } \| \pi )$ . We observe that $H ( 0 ) = 0$ such that $\begin{array} { r } { H ( s ) = \int _ { 0 } ^ { s } H ^ { \prime } ( t ) } \end{array}$ dt. We compute $H ^ { \prime } ( t )$ by

$$
\begin{array} { l } { { H ^ { \prime } ( t ) = - \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \mathrm { K L } ( \mu _ { t } | | \pi ) = \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \int \rho _ { k + 1 } \left( ( 1 - t ) \log ( a ) a ^ { t } + a ^ { t } \right) \mathrm { d } x } \ ~ } \\ { { \displaystyle ~ = ( 1 - t ) \int \rho _ { k + 1 } \log ^ { 2 } ( a ) a ^ { t } \mathrm { d } x \geq 0 . } } \end{array}
$$

In particular, since the integrant is positive, we obtain

$$
\begin{array} { l } { \displaystyle H ( s ) = \int _ { 0 } ^ { s } H ^ { \prime } ( t ) \mathrm { d } t \geq ( 1 - s ) \int \rho _ { k + 1 } \log ^ { 2 } ( a ) \int _ { 0 } ^ { 1 } a ^ { t } \mathrm { d } t \mathrm { d } x } \\ { \displaystyle \quad = ( 1 - s ) \int \rho _ { k + 1 } \log ( a ) ( a ^ { s } - 1 ) \mathrm { d } x } \\ { \displaystyle \quad \geq \frac { ( 1 - s ) \alpha } { s } \mathrm { W F R } _ { \alpha } ^ { 2 } ( \rho _ { k + 1 } , \mu _ { s } ) . } \end{array}
$$

By the optimality of $\rho _ { k + 1 }$ and applying the triangle inequality in the first summand, we get

$$
\begin{array} { r l } & { 0 \leq \displaystyle \frac { 1 } { 2 h } \operatorname { W F R } _ { \alpha } ^ { 2 } ( \mu _ { s } , \rho _ { k } ) - \frac { 1 } { 2 h } \operatorname { W F R } _ { \alpha } ^ { 2 } ( \rho _ { k + 1 } , \rho _ { k } ) - H ( s ) } \\ & { \quad \leq \displaystyle \frac { 1 } { 2 h } \operatorname { W F R } _ { \alpha } ^ { 2 } ( \rho _ { k + 1 } , \mu _ { s } ) + \frac { 1 } { h } \operatorname { W F R } _ { \alpha } ( \rho _ { k + 1 } , \mu _ { s } ) \operatorname { W F R } _ { \alpha } ( \rho _ { k + 1 } , \rho _ { k } ) - H ( s ) } \\ & { \quad \leq \left( \displaystyle \frac { s } { 2 h ( 1 - s ) \alpha } - 1 \right) H ( s ) + \displaystyle \frac { 1 } { h } \sqrt { \frac { s } { ( 1 - s ) \alpha } H ( s ) } \operatorname { W F R } _ { \alpha } ( \rho _ { k + 1 } , \rho _ { k } ) . } \end{array}
$$

Division by $\sqrt { H ( s ) }$ gives

$$
\left( 1 - \frac { s } { 2 h ( 1 - s ) \alpha } \right) \sqrt { H ( s ) } \leq \frac { 1 } { h } \sqrt { \frac { s } { ( 1 - s ) \alpha } } \mathrm { W F R } _ { \alpha } ( \rho _ { k + 1 } , \rho _ { k } ) .
$$

Since for $0 \leq s < 2 h \alpha / ( 1 + 2 h \alpha )$ both sides are nonnegative, taking squares yields

$$
\left( 1 - \frac { s } { 2 h ( 1 - s ) \alpha } \right) ^ { 2 } H ( s ) \le \frac { 1 } { h ^ { 2 } } \frac { s } { ( 1 - s ) \alpha } \mathrm { W F R } _ { \alpha } ^ { 2 } ( \rho _ { k + 1 } , \rho _ { k } ) .
$$

Since for $s = 0$ both sides are zero, we have that the derivative of the left side must be smaller or equal than the derivative on the right side, i.e.,

$$
H ^ { \prime } ( 0 ) \leq \frac { 1 } { \alpha h ^ { 2 } } \mathrm { W F R } _ { \alpha } ^ { 2 } ( \rho _ { k + 1 } , \rho _ { k } )
$$

Noting that $H ^ { \prime } ( 0 ) = I ( \rho _ { k + 1 } )$ yields the assertion.

The next lemma shows that $\rho _ { k } / \pi$ remains bounded from below.

Lemma 3.4. Let $( \rho _ { k } ) _ { k }$ be generated by (3.1). Then, it holds for all $k \in \mathbb N$ that $\rho _ { k } \geq$ $\exp ( - \frac { 2 } { \alpha h } ) \pi \ a . e .$

Proof. Let $\gamma$ be the optimal plan in the LET problem (2.10) between the marginals $\rho _ { k + 1 }$ and $\rho _ { k }$ . We note that its marginals $\gamma _ { 1 }$ and $\gamma _ { 2 }$ are absoulutely continuous, since $\mathrm { K L } ( \gamma _ { 1 } \| \rho _ { k + 1 } )$ is finite. We define the energy

$$
\mathcal { E } ( \rho ) = \frac { 2 } { h \alpha } \left( \mathrm { K L } ( \gamma _ { 1 } \| \rho ) + \mathrm { K L } ( \gamma _ { 2 } \| \rho _ { k } ) + \int \ell _ { \alpha } \mathrm { d } \gamma \right) + \mathrm { K L } ( \rho \| \pi ) .
$$

By definition, we have $\rho _ { k + 1 } \in$ arg m $\mathrm { i n } _ { \rho \in \mathcal { M } ( \mathbb { R } ^ { d } ) } J ( \rho )$ . Further, by (2.10), it holds $J ( \rho _ { k + 1 } ) =$ $\mathcal { E } ( \rho _ { k + 1 } )$ and $J ( \rho ) \le { \mathcal { E } } ( \rho )$ for all $\rho \in \mathcal { M } ( \mathbb { R } ^ { d } )$

Now we define the set $A : = \{ x \in \mathbb { R } ^ { d } : \rho _ { k + 1 } ( x ) < \exp ( - \frac { 2 } { \alpha h } ) \pi ( x ) \}$ and show that $\pi ( A ) = 0$ . Assume by contradiction that $\pi ( A ) > 0$ and define $\rho ^ { * }$ by

$$
\rho ^ { * } : = \rho _ { k + 1 } + { \frac { 1 } { 2 } } w , \quad \quad w ( x ) : = { \left\{ \begin{array} { l l } { \exp ( - { \frac { 2 } { h \alpha } } ) \pi ( x ) - \rho _ { k + 1 } ( x ) , } & { { \mathrm { i f ~ } } x \in A , } \\ { 0 , } & { { \mathrm { i f ~ } } x \notin A . } \end{array} \right. }
$$

Note that by definition of $A$ , the function w is non-negative and fulfills

$$
\int w \mathrm { d } x \leq \exp ( - \frac { 2 } { h \alpha } ) \pi ( \mathbb { R } ^ { d } ) < \infty
$$

such that $\boldsymbol { \rho } ^ { * } \in \mathcal { M } ( \mathbb { R } ^ { d } )$ . Next, we will bound

$$
\mathcal { E } ( \rho ^ { * } ) - \mathcal { E } ( \rho _ { k + 1 } ) = \frac { 2 } { h \alpha } ( \mathrm { K L } ( \gamma _ { 1 } \| \rho ^ { * } ) - \mathrm { K L } ( \gamma _ { 1 } \| \rho _ { k + 1 } ) ) + \mathrm { K L } ( \rho ^ { * } \| \pi ) - \mathrm { K L } ( \rho _ { k + 1 } \| \pi ) .\tag{3.3}
$$

To bound the first diference, we compute

$$
\begin{array} { r l } & { \mathrm { K L } ( \gamma _ { 1 } \| \rho ^ { * } ) - \mathrm { K L } ( \gamma _ { 1 } \| \rho _ { k + 1 } ) = \displaystyle \int \gamma _ { 1 } \log \left( \frac { \rho _ { k + 1 } } { \rho ^ { * } } \right) + \rho ^ { * } - \rho _ { k + 1 } \mathrm { d } x } \\ & { \qquad = \displaystyle \int \gamma _ { 1 } \log \left( \frac { \rho _ { k + 1 } } { \rho _ { k + 1 } + \frac { 1 } { 2 } w } \right) + \frac { 1 } { 2 } w \mathrm { d } x \leq \frac { 1 } { 2 } \int w \mathrm { d } x , } \end{array}\tag{3.4}
$$

where the last step follows from the fact that $w \geq 0$ . To bound the second diference, we use the nonnegative, convex function $\varphi ( r ) : = r \log r - r + 1$ for $r \geq 0$ . Then we have $\varphi ( a ) - \varphi ( b ) \leq \varphi ^ { \prime } ( a ) ( a - b )$ and $\varphi ^ { \prime } ( r ) = \log ( r )$ such that

$$
\begin{array} { r l } & { \mathrm { K L } ( \rho ^ { * } \| \pi ) - \mathrm { K L } ( \rho _ { k + 1 } \| \pi ) = \displaystyle \int \pi \left( \varphi \left( \frac { \rho ^ { * } } { \pi } \right) - \varphi \left( \frac { \rho _ { k + 1 } } { \pi } \right) \right) \ \mathrm { d } x } \\ & { \qquad \leq \displaystyle \int \pi \left( \varphi ^ { \prime } \left( \frac { \rho ^ { * } } { \pi } \right) \frac { \rho ^ { * } - \rho _ { k + 1 } } { \pi } \right) \ \mathrm { d } x } \\ & { \qquad = \displaystyle \frac { 1 } { 2 } \int \log \left( \frac { \rho _ { k + 1 } + \frac { 1 } { 2 } w } { \pi } \right) w \ \mathrm { d } x . } \end{array}
$$

On $A ,$ we have $\rho _ { k + 1 } = \exp \bigl ( - \textstyle { \frac { 2 } { h \alpha } } \bigr ) \pi - w$ and consequently

$$
\log \left( \frac { \rho _ { k + 1 } ( x ) + \frac { 1 } { 2 } w ( x ) } { \pi ( x ) } \right) w ( x ) = \log \left( \exp \left( - \frac { 2 } { h \alpha } \right) - \frac { w ( x ) } { 2 \pi ( x ) } \right) w ( x ) , \quad x \in A .
$$

Outside of $A ,$ we have that $w ( x ) = 0$ such that the same equality holds. Together, we obtain

$$
\begin{array} { r l } & { \frac { 1 } { 2 } \displaystyle \int \log \left( \frac { \rho _ { k + 1 } + \frac { 1 } { 2 } w } { \pi } \right) w ~ \mathrm { d } x = \frac { 1 } { 2 } \displaystyle \int \log \left( \exp ( - \frac { 2 } { h \alpha } ) - \frac { w } { 2 \pi } \right) w ~ \mathrm { d } x } \\ & { \qquad = \frac { 1 } { 2 } \displaystyle \int - \frac { 2 } { h \alpha } w + \log \left( 1 - \frac { \exp \left( \frac { 2 } { h \alpha } \right) w } { 2 \pi } \right) w ~ \mathrm { d } x } \\ & { \qquad = - \frac { 1 } { h \alpha } \displaystyle \int w ~ \mathrm { d } x + \frac { 1 } { 2 } \displaystyle \int \log \left( 1 - \frac { \exp \left( \frac { 2 } { h \alpha } \right) w } { 2 \pi } \right) w ~ \mathrm { d } x < - \frac { 1 } { h \alpha } \displaystyle \int w ~ \mathrm { d } x , } \end{array}
$$

where the last inequality follows from our assumption $\pi ( A ) > 0$ (implying $| { \cal { A } } | > 0 )$ and the fact that $w > 0$ on A and the second integrand is strictly negative, whenever $w > 0$ In summary, we obtain

$$
\operatorname { K L } ( \rho ^ { * } \| \pi ) - \operatorname { K L } ( \rho _ { k + 1 } \| \pi ) < - { \frac { 1 } { h \alpha } } \int w \ \mathrm { d } x
$$

Inserting this together with (3.4) in (3.3), we obtain that $\mathcal { E } ( \rho ^ { \ast } ) - \mathcal { E } ( \rho _ { k + 1 } ) < 0$ . In particular, we have

$$
J ( \rho ^ { * } ) \le \mathcal { E } ( \rho ^ { * } ) < \mathcal { E } ( \rho _ { k + 1 } ) = J ( \rho _ { k + 1 } ) ,
$$

which contradicts the assumption that $\rho _ { k + 1 }$ minimizes J. Thus, we conclude $\pi ( A ) = 0 . \quad \sqcup$

Using the two lemmas, we can prove our final convergence result.

Theorem 3.5. Let $\pi \in \mathcal { M } ( \mathbb { R } ^ { d } )$ with $\pi > 0$ and $\rho _ { 0 } \in \mathcal { M } ( \mathbb { R } ^ { d } ) . \ ( \rho _ { k } ) _ { k }$ be generated by (3.1). Then, it holds

$$
\mathrm { K L } ( \rho _ { k + 1 } \| \pi ) \leq { \frac { 1 } { 1 + c } } \mathrm { K L } ( \rho _ { k } \| \pi ) ,
$$

such that

$$
\begin{array} { r } { \mathrm { K L } ( \rho _ { k + 1 } \| \pi ) \leq ( 1 + c ) ^ { - k } \mathrm { K L } ( \rho _ { 0 } \| \pi ) } \end{array}
$$

for

$$
c = \frac { \beta } { \exp ( \beta ) - \beta - 1 } , \quad w h e r e \quad \beta = \frac { 2 } { \alpha h } .
$$

Proof. The function $\begin{array} { r } { \psi ( r ) : = \frac { \exp ( r ) - 1 - r } { r ^ { 2 } } } \end{array}$ is strictly increasing on R and is related to $\varphi ( r ) : = r \log ( r ) - r + 1$ by

$$
\varphi ( r ) = \psi \left( - \log ( r ) \right) r \log ^ { 2 } ( r ) , \quad r \geq 0 .
$$

Then we have

$$
\mathrm { K L } ( \rho _ { k + 1 } | | \pi ) = \int \pi \varphi \left( \frac { \rho _ { k + 1 } } { \pi } \right) ~ \mathrm { d } x = \int \rho _ { k + 1 } \log ^ { 2 } \left( \frac { \rho _ { k + 1 } } { \pi } \right) \psi \left( - \log \left( \frac { \rho _ { k + 1 } } { \pi } \right) \right) ~ \mathrm { d } x .
$$

By Lemma 3.4 we know that $\begin{array} { r } { \frac { \rho _ { k + 1 } } { \pi } \geq \exp ( - \frac { 2 } { \alpha h } ) , \mathrm { ~ i . e . , ~ } - \log \left( \frac { \rho _ { k + 1 } } { \pi } \right) \leq \frac { 2 } { \alpha h } } \end{array}$ and since $\psi$ is increasing, this implies

$$
\mathrm { K L } ( \rho _ { k + 1 } | | \pi ) \leq \int \rho _ { k + 1 } \log ^ { 2 } \left( \frac { \rho _ { k + 1 } } { \pi } \right) \psi \left( \frac { 2 } { \alpha h } \right) \ \mathrm { d } x = \psi \left( \frac { 2 } { \alpha h } \right) I ( \rho _ { k + 1 } ) .\tag{3.5}
$$

By definition $\rho _ { k + 1 }$ minimizes J in (3.1) such that $J ( \rho _ { k + 1 } ) \le J ( \rho _ { k } )$ . Thus, using Lemma 3.3 and (3.5), we obtain the estimate

$$
\begin{array} { l } { \displaystyle \mathrm { K L } ( \rho _ { k } \| \pi ) \geq \frac { 1 } { 2 h } \mathrm { W F R } _ { \alpha } ^ { 2 } ( \rho _ { k + 1 } , \rho _ { k } ) + \mathrm { K L } ( \rho _ { k + 1 } \| \pi ) } \\ { \displaystyle \qquad \geq \frac { \alpha h } { 2 } I ( \rho _ { k + 1 } ) + \mathrm { K L } ( \rho _ { k + 1 } \| \pi ) } \\ { \displaystyle \qquad \geq \left( \frac { \alpha h } { 2 \psi ( \frac { 2 } { \alpha h } ) } + 1 \right) \mathrm { K L } ( \rho _ { k + 1 } \| \pi ) . } \end{array}
$$

This yields the claim with

$$
c = \frac { \alpha h } { 2 \psi ( \frac { 2 } { \alpha h } ) } = \frac { \beta } { \exp ( \beta ) - \beta - 1 } > 0 .
$$

This finishes the proof.

## 4 Normalizing Flow via WFR JKO Scheme

Next, we derive a computable loss function for training the velocity and reaction function in a single WFR–JKO step. Afterwards, we summarize the algorithm in order to simulate the whole scheme.

4.1 Loss Function. We consider the admissible class

$$
\begin{array} { r } { \mathcal { A } : = \left\{ ( v , g ) : \begin{array} { l l } { v : [ 0 , 1 ] \times \mathbb { R } ^ { d } \to \mathbb { R } ^ { d } , g : [ 0 , 1 ] \times \mathbb { R } ^ { d } \to \mathbb { R } \mathrm { ~ \ m e a s u r a b l e ~ } } \\ { \mathrm { ~ a n d ~ f u l f i l ~ } ( 2 . 3 ) , ( 2 . 1 1 ) \mathrm { ~ a n d ~ } ( 2 . 1 2 ) } \end{array} \right\} . } \end{array}
$$

By Proposition 2.5, we know that for $( v , g ) \in A$ , the weakly continuous curve $\mu \colon [ 0 , 1 ] \to$ $\mathcal { M } ( \mathbb { R } ^ { d } )$ satisfying the continuity equation with reaction (2.8) is uniquely determined. Therefore, restricting the JKO step (3.1) to $( v , g ) \in A$ leads to the problem

$$
\operatorname * { a r g m i n } _ { ( v _ { t } , g _ { t } ) \in \mathcal A } \mathcal L _ { k } ( v , g ) , \quad \mathcal L _ { k } ( v , g ) : = \frac { 1 } { 2 h } \int _ { 0 } ^ { 1 } \int | v _ { t } | ^ { 2 } + \frac { 1 } { \alpha } g _ { t } ^ { 2 } \ \mathrm { d } \mu _ { t } \mathrm { d } t + \mathrm { K L } ( \mu _ { 1 } \parallel \pi ) ,\tag{4.1}
$$

where $\mu _ { t } = \mu _ { t } ^ { v _ { t } , g _ { t } }$ is the unique solution of the the continuity equation with reaction (2.8) starting in $\mu _ { 0 } = \rho _ { k }$

In this section, we provide a computable formulation for the objective (4.1) of a WFR JKO step. Note that Wasserstein–Fisher–Rao schemes for neural sampling were considered, e.g. in [8]. The following proposition reformulates the action term in (4.1). The proof follows standard lines in the normalizing flow literature, see, e.g. [22, 54].

Proposition 4.1. Let $( v , g ) \in A$ . Further, let $\phi ( t , x )$ be the solution of the ODE (2.4) with vector field $v _ { t }$ and $\mu _ { t } : = \phi ( t , \cdot ) _ { \# } ( w _ { t } \rho _ { k } )$ with weight $w _ { t }$ from (2.13). Let $\pi ( x ) = \exp ( - V ( x ) )$ with $\begin{array} { r } { Z : = \int \exp { - V ( x ) } } \end{array}$ dx $< \infty$ . Then it holds

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { k } ( v , g ) = \frac { 1 } { 2 h } \int _ { 0 } ^ { 1 } \mathbb { E } \Big [ \rho _ { k } ( \mathbb { R } ^ { d } ) w _ { t } ( X _ { k } ) \Big ( | v _ { t } \circ \phi _ { t } ( X _ { k } ) | ^ { 2 } + \frac { 1 } { \alpha } g _ { t } ^ { 2 } \circ \phi _ { t } ( X _ { k } ) \Big ) \Big ] ~ \mathrm { d } t } \\ { \displaystyle \qquad + \mathbb { E } \Big [ \rho _ { k } ( \mathbb { R } ^ { d } ) w _ { 1 } ( X _ { k } ) \Big ( \log \rho _ { k } ( X _ { k } ) + \log w _ { 1 } ( X _ { k } ) + V \circ \phi _ { 1 } ( X _ { k } ) - 1 \Big ) \Big ] } \\ { \displaystyle \qquad - \int _ { 0 } ^ { 1 } \mathbb { E } \Big [ \rho _ { k } ( \mathbb { R } ^ { d } ) w _ { 1 } ( X _ { k } ) ( \mathrm { d i v } v _ { t } ) \circ \phi _ { t } ( X _ { k } ) \Big ] \mathrm { d } t + \pi ( \mathbb { R } ^ { d } ) , } \end{array}
$$

where the expectation is taken with respect to $X _ { k } \sim \rho _ { k } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k }$

Proof. First, we have, for any $t \in [ 0 , 1 ]$ , by definition of the push-forward operator that

$$
\int _ {  { { \mathbb R } } ^ { d } } | v _ { t } | ^ { 2 } + \frac { 1 } { \alpha } g _ { t } ^ { 2 } \mathrm { d } \mu _ { t } =  { { \mathbb E } } \biggl [ \rho _ { k } (  { { \mathbb R } } ^ { d } ) w _ { t } ( X _ { k } ) \left( | v _ { t } \circ \phi _ { t } ( X _ { k } ) | ^ { 2 } + \frac { 1 } { \alpha } ( g _ { t } \circ \phi _ { t } ( X _ { k } ) ) ^ { 2 } \right) \biggr ] .\tag{4.2}
$$

Second, we obtain by definition of the KL divergence that

$$
\mathrm { K L } ( \mu _ { 1 } \parallel \pi ) = \underbrace { \int _ { \mathbb { R } ^ { d } } \log \mu _ { 1 } \mathrm { d } \mu _ { 1 } } _ { A _ { 1 } } + \underbrace { \int _ { \mathbb { R } ^ { d } } V \mathrm { d } \mu _ { 1 } } _ { A _ { 2 } } - \underbrace { \mu _ { 1 } ( \mathbb { R } ^ { d } ) } _ { A _ { 3 } } + \pi ( \mathbb { R } ^ { d } ) .
$$

First, we consider the summand $A _ { 1 }$ . For the density of the push-forward measure, we have by the change of variable formula and (2.5) that

$$
\mu _ { 1 } \circ \phi _ { 1 } = \frac { w _ { 1 } \rho _ { k } } { \operatorname* { d e t } \nabla \phi _ { 1 } } = w _ { 1 } \rho _ { k } \exp \left( - \int _ { 0 } ^ { 1 } ( \operatorname { d i v } v _ { t } ) \circ \phi _ { t } \mathrm { d } t \right) ,
$$

so that

$$
\log ( \mu _ { 1 } \circ \phi _ { 1 } ) = \log \rho _ { k } + \log w _ { 1 } - \int _ { 0 } ^ { 1 } ( \operatorname { d i v } v _ { t } ) \circ \phi _ { t } \mathrm { d } t
$$

and

$$
\begin{array} { r l r } {  { \int \log \mu _ { 1 } \mathrm { d } \mu _ { 1 } = \int \log ( \mu _ { 1 } \circ \phi _ { 1 } ) w _ { 1 } \mathrm { d } \rho _ { k } } } \\ & { } & { ~ = \int \Big ( \log \rho _ { k } + \log w _ { 1 } \Big ) w _ { 1 } \mathrm { d } \rho _ { k } - \int _ { 0 } ^ { 1 } \Big ( \int ( \operatorname { d i v } v _ { t } ) \circ \phi _ { t } w _ { 1 } \mathrm { d } \rho _ { k } \Big ) \mathrm { d } t } \\ & { } & { ~ = \mathbb { E } \Big [ \Big ( \log \rho _ { k } ( X _ { k } ) + \log w _ { 1 } ( X _ { k } ) \Big ) w _ { 1 } ( X _ { k } ) \rho _ { k } ( \mathbb { R } ^ { d } ) \Big ] } \\ & { } & { ~ - \int _ { 0 } ^ { 1 } \mathbb { E } \Big [ \big ( \operatorname { d i v } v _ { t } \big ) \circ \phi _ { t } ( X _ { k } ) w _ { 1 } ( X _ { k } ) \rho _ { k } ( \mathbb { R } ^ { d } ) \Big ] \mathrm { d } t . ~ } \end{array}
$$

For the summand $A _ { 2 } .$ it follows by definition of the push-forward operator that

$$
\int V \mathrm { d } \mu _ { 1 } = \int w _ { 1 } ( V \circ \phi _ { 1 } ) \mathrm { d } \rho _ { k } = \mathbb { E } \Bigl [ \rho _ { k } ( \mathbb { R } ^ { d } ) w _ { 1 } ( X _ { k } ) V ( \phi _ { 1 } ( X _ { k } ) ) \Bigr ] ,
$$

and the summand $A _ { 3 }$ similarly

$$
\mu _ { 1 } ( \mathbb { R } ^ { d } ) = \mathbb { E } [ \rho _ { k } ( \mathbb { R } ^ { d } ) w _ { 1 } ( X _ { k } ) ] .
$$

In summary, this gives

$$
\begin{array} { r l } & { \mathrm { K L } ( \nu _ { 1 } \parallel \pi ) = \mathbb { E } \Big [ \rho _ { k } ( \mathbb { R } ^ { d } ) w _ { 1 } ( X _ { k } ) \Big ( \log \rho _ { k } ( X _ { k } ) + \log w _ { 1 } ( X _ { k } ) - \log \rho _ { k } ( \mathbb { R } ^ { d } ) + V ( \phi _ { 1 } ( X _ { k } ) ) - 1 \Big ) \Big ] } \\ & { \qquad - \int _ { 0 } ^ { 1 } \mathbb { E } \Big [ ( \normalfont { \mathrm { d i v } } v _ { t } ) \circ \phi _ { t } ( X _ { k } ) w _ { 1 } ( X _ { k } ) \rho _ { k } ( \mathbb { R } ^ { d } ) \Big ] \mathrm { d } t } \end{array}
$$

and adding (4.2) yields the assertion

Remark 4.2. Let $\xi \sim \operatorname { R a d } ^ { d }$ be Rademacher distributed, meaning that its components are iid $\xi _ { i } \sim \mathrm { U n i f } \{ - 1 , 1 \}$ , and let $\xi$ be independent of $X _ { k }$ . Then Hutchinson’s stochastic trace estimator [26] yields

$$
\begin{array} { r } { \mathbb E \Big [ \rho _ { k } ( \mathbb R ^ { d } ) w _ { 1 } ( X _ { k } ) ( \mathrm { d i v } v _ { t } ) \circ \phi _ { t } ( X _ { k } ) \Big ] = \mathbb E \big [ \rho _ { k } ( \mathbb R ^ { d } ) w _ { 1 } ( X _ { k } ) \xi ^ { \top } \nabla v _ { t } \circ \phi _ { t } ( X _ { k } ) \xi \big ] , } \end{array}\tag{4.3}
$$

where the expectation on the right-hand side is taken with respect to $( X _ { k } , \xi ) \sim ( \rho _ { k } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k } ) \mathbb { \ Q }$ Rad. Alternatively, the Rademacher distribution could be replaced by the standard Gaussian one, but

$$
\mathbb { E } _ { \mathrm { G a u s s } } [ ( \xi ^ { \top } A \xi - \mathrm { t r a c e } ( A ) ) ^ { 2 } ] \geq \mathbb { E } _ { \mathrm { R a d } } [ ( \xi ^ { \top } A \xi - \mathrm { t r a c e } ( A ) ) ^ { 2 } ] ,
$$

see, e.g., [38].

By Propositions 4.1, we suggest the following objective function to learn neural networks for $( v , g )$ :

$$
\begin{array} { l } { { \displaystyle { \mathcal E } _ { k } ^ { \mathrm { W F R } } ( \theta ) = \frac { 1 } { 2 h } \int _ { 0 } ^ { 1 } { \mathbb E } \Big [ \rho _ { k } ( { \mathbb R } ^ { d } ) w _ { t } ( X _ { k } ) \left( | v _ { t } ^ { \theta } \circ \phi _ { t } ( X _ { k } ) | ^ { 2 } + \frac { 1 } { \alpha } ( g _ { t } ^ { \theta } \circ \phi _ { t } ( X _ { k } ) ) ^ { 2 } \right) \Big ] \mathrm { d } t } \ ~ } \\ { { \displaystyle ~ + { \mathbb E } \Big [ \rho _ { k } ( { \mathbb R } ^ { d } ) w _ { 1 } ^ { \theta } ( X _ { k } ) \Big ( \log \rho _ { k } ( X _ { k } ) + \log w _ { 1 } ^ { \theta } ( X _ { k } ) + V \circ \phi _ { 1 } ( X _ { k } ) - 1 \Big ) \Big ] } } \\ { { \displaystyle ~ - \int _ { 0 } ^ { 1 } { \mathbb E } \Big [ \rho _ { k } ( { \mathbb R } ^ { d } ) w _ { 1 } ^ { \theta } ( X _ { k } ) \big ( \mathrm { d i v } v _ { t } \big ) \circ \phi _ { t } ( X _ { k } ) \Big ] \mathrm { d } t } . } \end{array}\tag{4.4}
$$

Likewise the final summand can be replaced by the Hutchinson estimator (4.3).

Given samples $X _ { k } ^ { 1 } , \ldots , X _ { k } ^ { N } \in \mathbb { R } ^ { d }$ from the distribution $\rho _ { k } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k }$ and independent samples $\xi _ { 1 } , \ldots , \xi _ { N } \sim ^ { \mathrm { i . i . d . } } ~ \mathrm { R a d } ^ { d }$ , we approximate the population risk (4.4) through an empirical risk

$$
\begin{array} { r l } & { \widehat { \mathcal { L } } _ { k } ^ { \mathrm { W F R } } ( \theta ) = \displaystyle \frac { 1 } { 2 h N } \sum _ { i = 1 } ^ { N } \int _ { 0 } ^ { 1 } w _ { t } ^ { \theta } ( X _ { k } ^ { i } ) \bigg \{ | v _ { t } ^ { \theta } \circ \phi _ { t } ( X _ { k } ^ { i } ) | ^ { 2 } + \frac { 1 } { \alpha } ( g _ { t } ^ { \theta } \circ \phi _ { t } ( X _ { k } ^ { i } ) ) ^ { 2 } \bigg \} \mathrm { d } t } \\ & { \qquad + \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { 1 } ( X _ { k } ^ { i } ) \bigg \{ \log \rho _ { k } ( X _ { k } ^ { i } ) + \log w _ { 1 } ^ { \theta } ( X _ { k } ^ { i } ) + V \circ \phi _ { 1 } ( X _ { k } ^ { i } ) - 1 \bigg \} } \\ & { \qquad - \displaystyle \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \int _ { 0 } ^ { 1 } w _ { 1 } ^ { \theta } ( X _ { k } ^ { i } ) \xi _ { i } ^ { \top } \nabla v _ { t } ^ { \theta } \circ \phi _ { t } ( X _ { k } ^ { i } ) \xi _ { i } \mathrm { d } t , } \end{array}\tag{4.5}
$$

4.2 Algorithms. In order to simulate the WFR–JKO scheme, we start with a tractable latent measure $\rho _ { 0 }$ which is usually a isometric Gaussian distribution $\mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } I )$ , where $\sigma _ { 0 }$ is the initial covariance which we consider as a hyperparameter. We draw initial samples $X _ { 0 } ^ { 1 } , . . . , X _ { 0 } ^ { N }$ from $\rho _ { 0 } = \rho _ { 0 } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { 0 }$ . Then, we iteratively generate samples $X _ { k } ^ { 1 } , . . . , X _ { k } ^ { N }$ from $\rho _ { k } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k }$ and compute the corresponding densities $\rho _ { k } ( X _ { k } ^ { i } )$ as well as the total mass $\rho _ { k } ( \mathbb { R } ^ { d } )$ by performing the following steps for $k = 0 , . . . , K$

1. We learn the velocity $v ^ { k }$ and reaction rate $g ^ { k }$ from the WRF–JKO scheme (4.1) by parameterizing $v ^ { k }$ and $g ^ { k }$ as neural networks with parameters θ and minimizing the loss function $\bar { \mathcal { L } } _ { k } ^ { \mathrm { W F R } } ( \theta )$ from (4.5). We include the detailed computation of this loss function as pseudo-code in Algorithm 1 in the appendix.

2. Once the velocity and reaction are trained, we can draw weighted samples $Y _ { k + 1 } ^ { 1 } , . . . , Y _ { k + 1 } ^ { N }$ with weights $w _ { k + 1 } ^ { 1 } , . . . , w _ { k + 1 } ^ { N }$ from $\rho _ { k + 1 } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k + 1 }$ and the total mass $\rho _ { k + 1 } ( \mathbb { R } ^ { d } )$ by solving the flow ODE $\phi$ from (2.4) and applying (2.13). We outline this computation as pseudo-code in Algorithm 2.

3. In order to obtain an ensemble of unweighted samples from $\rho _ { k + 1 } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k + 1 }$ , we perform a resampling step as proposed in [16]. More precisely, we normalize the weights $w _ { k + 1 } ^ { 1 } , . . . , w _ { k + 1 } ^ { N }$ and draw samples from the empirical distribution $\begin{array} { r l } { ~ } & { { } \sum _ { n = 1 } ^ { N } w _ { k + 1 } ^ { n } \delta _ { Y _ { k + 1 } ^ { n } } } \end{array}$ This step is outlined as pseudo-code in Algorithm 3.

Repeating this procedure for $k = 0 , . . . , K$ provides us with samples $X _ { K } ^ { 1 } , . . . , X _ { K } ^ { N }$ from $\rho _ { k } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k }$ and the total mass $\rho _ { k } ( \mathbb { R } ^ { d } )$

## 5 Metropolis–Hastings Steps

The resampling step from of the section unavoidably causes particle degeneracy: lowweight particles are discarded and high-weight particles are duplicated, so the support of the ensemble collapses. To rejuvenate the ensemble while preserving its target, we apply one step of the Metropolis–Hastings (MH) algorithm, see, e.g., [46], after each resampling step. $\mathrm { A s }$ a stationary distribution we choose the distribution $\rho _ { k } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k }$ such that the overall distribution of the samples is not afected by the MH steps. In the following, we first revisit the MH algorithm and then focus on its application in the WFR–JKO scheme.

5.1 Metropolis–Hastings Algorithm. Let $\mathcal { Q } \colon \mathbb { R } ^ { d } \times \mathcal { B } ( \mathbb { R } ^ { d } ) \to [ 0 , 1 ]$ be a Markov kernel such that $\mathcal { Q } ( x , \cdot )$ is absolutely continuous with non-zero density for all $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and let $\hat { \rho } \in \mathcal { P } ( \mathbb { R } ^ { d } )$ be an absolutely continuous probability measure, which we call the target distribution. $\mathrm { A s }$ before, we denote the densities of $\mathcal { Q } ( x , \cdot )$ and $\hat { \rho }$ again by $\mathcal { Q } ( x , y )$ and ${ \hat { \rho } } ( x )$ . Then the MH algorithm generates a sequence $( x _ { n } ) _ { n }$ starting at $\boldsymbol { x } _ { 0 } \in \mathbb { R } ^ { d }$ using the following steps

1. Draw $x ^ { \prime }$ from $\mathcal { Q } ( x _ { n } , \cdot )$ and u uniformly in [0, 1].

2. Compute the acceptance probability $\begin{array} { r } { A ( x _ { n } , x ^ { \prime } ) : = \operatorname* { m i n } \left\{ 1 , \frac { \hat { \rho } ( x ^ { \prime } ) Q ( x ^ { \prime } , x _ { n } ) } { \hat { \rho } ( x _ { n } ) Q ( x _ { n } , x ^ { \prime } ) } \right\} } \end{array}$

3. Set

$$
x _ { n + 1 } : = { \left\{ \begin{array} { l l } { x ^ { \prime } , } & { { \mathrm { i f ~ } } u < A ( x _ { n } , x ^ { \prime } ) , } \\ { x _ { n } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

The MH algorithm generates a time-homogeneous Markov chain with transition kernel $\mathcal { K } _ { \mathrm { M H } } \colon \mathbb { R } ^ { d } \times \mathcal { B } ( \mathbb { R } ^ { d } )  [ 0 , 1 ]$ given by

$$
K _ { \mathrm { M H } } ( x , B ) = \int _ { A } \mathcal { Q } ( x , y ) A ( x , y ) \mathrm { d } y + \delta _ { x } ( B ) \int _ { \mathbb { R } ^ { d } } \mathcal { Q } ( x , y ) ( 1 - A ( x , y ) ) \mathrm { d } y .
$$

It can be shown that for any choice of $\mathcal { Q } ,$ the target distribution $\hat { \rho }$ is a stationary distribution of this Markov kernel in the sense that applying the kernel onto $\hat { \rho }$ yields again $\hat { \rho }$ or more precisely

$$
\hat { \rho } ( B ) = \int { \mathcal K } _ { \mathrm { M H } } ( x , B ) \mathrm { d } \hat { \rho } ( x ) , \quad \mathrm { f o r ~ a n y ~ B o r e l ~ s e t ~ } B \in \mathcal B ( \mathbb R ^ { d } ) .
$$

5.2 Application to the WFR–JKO Scheme. After each WFR–JKO step we are given an ensemble of samples $x _ { 1 } , . . . , x _ { n }$ which follow the distribution $\rho _ { k } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k }$ . Further, we have computed the total mass $\rho _ { k } ( \mathbb { R } ^ { d } )$ . However, due to the resampling step, some of these samples are located at the exact same position. Therefore, we apply one step of the MH algorithm on these samples. More precisely, we generate a new ensemble $y _ { 1 } , . . . , y _ { n }$ by sampling $y _ { i }$ from ${ \cal K } _ { \mathrm { M H } } ( x _ { i } , \cdot )$ . As a target distribution we use $\hat { \rho } : = \rho _ { k } ( \mathbb { R } ^ { d } ) ^ { - 1 } \rho _ { k }$ . Since we know that the $x _ { i }$ already follow the distribution $\hat { \rho }$ and since this is an stationary distribution of ${ \kappa _ { \mathrm { M H } } }$ , we obtain that also the ensemble $y _ { 1 } , . . . , y _ { n }$ follows ${ \hat { \rho } } .$ It is left to choose the Markov kernel $\mathcal { Q }$ from the MH algorithm. Here we consider the following choices.

1. Gaussian proposal: One of the most standard choices in the MH algorithm is

$$
\mathcal { Q } _ { \mathrm { G a u s s } } ( x , \cdot ) = \mathcal { N } ( x , \sigma ^ { 2 } I _ { d } ) ,\tag{5.1}
$$

where $\mathcal { N } ( m , \Sigma )$ denotes the Gaussian distribution with mean m and covariance matrix Σ and $\mathcal { N } ( \cdot ; m , \Sigma )$ the corresponding density. The kernel has the denisty $\mathcal { Q } _ { \mathrm { G a u s s } } ( x , x ^ { \prime } ) = \mathcal { N } ( x ^ { \prime } ; x , \sigma ^ { 2 } I _ { d } )$ , which is symmetric such that the acceptance probability simplifies to $\begin{array} { r } { A ( x , x ^ { \prime } ) = \operatorname* { m i n } \left\{ 1 , \frac { \rho _ { k } ( \bar { x ^ { \prime } } ) } { \rho _ { k } ( x ) } \right\} } \end{array}$

2. Langevin proposal: The MH algorithm is often combined with a Langevin sampling algorithm. This corresponds to applying one gradient descent step on the energy $- \log \hat { \rho }$ before adding Gaussian noise. Since computing gradients of $- \log \hat { \rho } = - \log \rho _ { k }$ is computational expensive, we replace the gradient of $- \log \rho _ { k }$ here by − log $\pi = \nabla V$ , which can be computed directly and is motivated by the fact that $\rho _ { k }$ is already close to $\pi$ for large k. This leads to the proposal distribution

$$
\mathcal { Q } _ { \mathrm { L a n g e v i n } } ( x , \cdot ) = \mathcal { N } ( x - \tau \nabla V ( x ) , \sigma ^ { 2 } I _ { d } ) .\tag{5.2}
$$

The step size $\tau$ and the noise parameter $\sigma$ are often chosen as $\sigma ^ { 2 } = 2 \tau$ which corresponds to one step of the unadjusted Langevin algorithm. However, depending how closely $\rho _ { k }$ approximates $\pi$ already also a choice of $\sigma ^ { 2 } > \tau$ can be useful. We obtain the density

$$
\mathcal { Q } _ { \mathrm { L a n g e v i n } } ( x , x ^ { \prime } ) = \mathcal { N } ( x ^ { \prime } ; x - \tau \nabla V ( x ) , \sigma ^ { 2 } I _ { d } ) .
$$

3. Proposal in the latent space: The MH algorithm often sufers from low acceptance rates, whenever the target distribution is multimodal or somehow degenerated. Since our WFR–JKO algorithm usually starts at a very regular distribution $\rho _ { 0 }$ , we follow the idea to transform the samples from $\rho _ { k }$ back to the latent space, perform the MH algorithm in the latent space with Gaussian proposal and transform the sample back to the data space. To make this procedure precise, we define the flow ODE $\phi \colon [ 0 , k ] \times  { \mathbb { R } } ^ { d } \to  { \mathbb { R } } ^ { d }$ by $\partial _ { t } \phi _ { l + t } ( l + t , x ) = v _ { t } ^ { l } ( \phi _ { k + t } ( k + t , x ) )$ with $\phi ( 0 , x ) = x ,$ where $v _ { t } ^ { l }$ is the optimal velocity from the $\mathrm { W F R \mathrm { - } J K O }$ step connecting $\rho _ { l }$ and $\rho _ { l + 1 }$ . Then, we define the proposal distribution as

$$
\mathcal { Q } _ { \mathrm { l a t e n t } } ( x , B ) = \mathcal { Q } _ { \mathrm { b a s e } } ( \phi _ { k } ^ { - 1 } ( x ) , \phi _ { k } ^ { - 1 } ( B ) ) ,\tag{5.3}
$$

where $\mathcal { Q } _ { \mathrm { b a s e } }$ is some base Markov kernel. In our numerics we consider the cases, where it is given by the Gaussian proposal (5.1) or Langevin proposal (5.2) (with the Gaussian energy instead of $V )$ . We compute the density of $\mathcal { Q } _ { \mathrm { l a t e n t } }$ in the next lemma.

Lemma 5.1. Let $v _ { t } ^ { l } \colon [ 0 , 1 ] \times  { \mathbb { R } ^ { d } } \to  { \mathbb { R } ^ { d } }$ and $\partial _ { t } \phi _ { l + t } ( x ) = v _ { t } ^ { l } \circ \phi _ { l + t } ( x ) \ f o r \ l = 0 , . . . , k - 1$ and $t \in [ 0 , 1 ]$ . Then, the transition kernel (5.3) has the density

$$
\mathcal { Q } _ { \mathrm { l a t e n t } } ( x , x ^ { \prime } ) = \mathcal { Q } _ { \mathrm { b a s e } } ( z , z ^ { \prime } ) \exp \left( - \sum _ { l = 0 } ^ { k - 1 } \int _ { 0 } ^ { 1 } \mathrm { d i v } v _ { t } ^ { l } ( \phi _ { l + t } ( z ^ { \prime } ) ) \mathrm { d } t \right) ,
$$

where $z ^ { \prime } = \phi _ { k } ^ { - 1 } ( x ^ { \prime } )$ and $z = \phi _ { k } ^ { - 1 } ( x )$

Proof. It holds

$$
\begin{array} { r l } & { \mathcal { Q } _ { \mathrm { l a t e n t } } ( x , B ) = \mathcal { Q } _ { \mathrm { b a s e } } ( z , \phi _ { k } ^ { - 1 } ( B ) ) = \displaystyle \int _ { \phi _ { k } ^ { - 1 } ( B ) } \mathcal { Q } _ { \mathrm { b a s e } } ( z , z ^ { \prime } ) \mathrm { d } z ^ { \prime } } \\ & { \qquad = \displaystyle \int _ { B } \mathcal { Q } _ { \mathrm { b a s e } } ( z , \phi _ { k } ^ { - 1 } ( x ^ { \prime } ) ) | \operatorname* { d e t } \phi _ { k } ^ { - 1 } ( x ^ { \prime } ) | \mathrm { d } x ^ { \prime } , } \end{array}
$$

where we used the substitution $z ^ { \prime } = \phi _ { k } ^ { - 1 } ( x ^ { \prime } )$ . In particular, $\mathcal { Q } _ { \mathrm { l a t e n t } } ( x , \cdot )$ has the density

$$
\mathcal { Q } _ { \mathrm { l a t e n t } } ( x , x ^ { \prime } ) = \mathcal { Q } _ { \mathrm { b a s e } } ( z , \phi _ { k } ^ { - 1 } ( x ^ { \prime } ) ) | \operatorname* { d e t } ( \phi _ { k } ^ { - 1 } ( z ^ { \prime } ) ) | .
$$

where $z ^ { \prime } = \phi _ { k } ^ { - 1 } ( x ^ { \prime } )$ . Then the claim follows by (2.5).

5.3 Density Evaluation of $\rho _ { k }$ . For computing the acceptance probabilities in the previous subsection, we have to evaluate the densities $\rho _ { k }$ at some arbitrary point $\boldsymbol { x } \in \mathbb { R } ^ { d }$ To this end we denote by $v ^ { k }$ and $g ^ { k }$ the optimal velocities and reactions from the WFR– JKO step connecting $\rho _ { k }$ and $\rho _ { k + 1 }$ . We define the concatenations $\tilde { v } \colon [ 0 , k ] \times  { \mathbb { R } } ^ { d } \to  { \mathbb { R } } ^ { d }$ and $\tilde { g } \colon [ 0 , k ] \times  { \mathbb { R } } ^ { d } \to  { \mathbb { R } }$ for $t \in [ k , k + 1 ]$ by

$$
\tilde { v } _ { t } ( x ) = v _ { t - k } ^ { k } ( x ) , \quad \tilde { g } _ { t } ( x ) = g _ { t - k } ^ { k } ( x ) .
$$

Then, by definition, the unique solution of the continuity equation with reaction

$$
\begin{array} { r } { \partial _ { t } \tilde { \mu } _ { t } + \mathrm { d i v } \left( \tilde { v } _ { t } \tilde { \mu } _ { t } \right) = \tilde { g } _ { t } \tilde { \mu } _ { t } , \quad \tilde { \mu } _ { 0 } = \rho _ { 0 } } \end{array}
$$

fulfills $\tilde { \mu } _ { k } = \rho _ { k }$ for all k. Using the time transform $s = k - t$ we obtain that the time reversals $\mu _ { s } = \tilde { \mu } _ { k - s } , v _ { s } = \tilde { v } _ { k - s }$ and $g _ { s } = \tilde { g } _ { k - s }$ that

$$
\partial _ { s } \mu _ { s } - \mathrm { { \mathop { d i v } } } \left( v _ { s } \mu _ { s } \right) = - g _ { s } \mu _ { s } , \quad \mathrm { { w i t h } } \quad \mu _ { 0 } = \rho _ { k } \mathrm { { ~ a n d ~ } } \mu _ { k } = \rho _ { 0 } .
$$

In particular, we get by (2.13) for the reverse flow ODE

$$
\partial _ { t } \phi ^ {  } ( t , \cdot ) = - v _ { t } \circ \phi ^ {  } ( t , \cdot ) , \quad \phi ^ {  } ( 0 , \cdot ) = \mathrm { i d }
$$

that

$$
\rho _ { 0 } = \phi ^ {  } ( k , \cdot ) _ { \# } ( w \rho _ { k } ) , \quad w ( x ) = \exp ( \int _ { 0 } ^ { k } - g _ { s } \circ \phi ^ {  } ( s , x ) \mathrm { d } s ) .
$$

Using the change-of-variables formula for densities, this implies for $\phi _ { s } ^ {  } = \phi ^ {  } ( s , \cdot )$ that the corresponding densities fulfill $\begin{array} { r } { \rho _ { 0 } ( \phi _ { k } ^ {  } ( x ) ) = \frac { \rho _ { k } ( x ) } { \operatorname* { d e t } ( \nabla \phi _ { k } ( x ) ) } } \end{array}$ . Inserting (2.5) this can be reformulated as

$$
\rho _ { k } ( x ) = \rho _ { 0 } ( \phi _ { k } ^ {  } ( x ) ) \exp ( \int _ { 0 } ^ { k } g _ { s } \circ \phi _ { s } ^ {  } ( x ) - \mathrm { d i v } v _ { s } \circ \phi _ { s } ^ {  } ( x ) \mathrm { d } s ) .
$$

## 6 Numerical Results

In this section, we report our promising numerical results in comparison with existing algorithms. To this end, we first introduce the test distributions on which we evaluate our algorithm. Then, we specify the detailed implementation and comparisons. Finally, we present our results.

6.1 Test Distributions. Next, we implement our algorithms. To this end, we evaluate the WFR–JKO scheme on the same test distributions as in [25]:

\- Mustache: The two-dimensional log-density is given as log $\mathcal { N } ( 0 , \Sigma ) \circ T$ with $\Sigma = [ 1 , \sigma ; \sigma , 1 ]$ and $T ( x _ { 1 } , x _ { 2 } ) = ( x _ { 1 } , ( x _ { 2 } - ( x _ { 1 } ^ { 2 } - 1 ) ^ { 2 } ) )$ . Note that det $( \nabla T ( x ) ) = 1$ for all $x .$ In particular, we obtain directly by the transformation formula that the normalization constant is one. Depending on $\sigma \in [ 0 , 1 )$ close to 1, this probability distribution has very long and narrow tails making it hard for classical MCMC methods to sample them. In our experiments we use $\sigma = 0 . 9$

\- Shifted 8 Modes: A two-dimensional Gaussian mixture model with 8 equal weighted modes and covariance matrix $1 \times 1 0 ^ { - 2 } I$ . The modes are placed in a circle with radius 1 and center $( - 1 , 0 )$ . Due to the shifted center classical MCMC methods have dificulties to distribute the mass correctly onto the modes.

\- Shifted 8 Peaky: This is the same distribution as the shifted 8 Modes with the diference that we reduce the width of the modes to the covariance matrix $5 \times 1 0 ^ { - 3 } I$ Since the modes are disconnected, it becomes harder to sample from them.

\- GMM-d: A d-dimensional Gaussian mixture model with 10 equal weighted modes with covariance matrix $1 \times 1 0 ^ { - 2 } I$ and means drawn randomly from a uniform distribution on $[ - 1 , 1 ] ^ { d }$ . This leads to a peaky high-dimensional and multimodal probability distribution which is hard to sample from.

6.2 Implementation and Hyperparameters. In practice, we start with a very small initial step size $h _ { 0 }$ and then iteratively increase the step size in each step by choosing $h _ { k + 1 } = \tau h _ { k }$ for some growth constant $\tau > 1$ . In order to avoid overfitting and the accumulation of errors, we resample the ensemble of samples after each WFR JKO step. Moreover, we apply a Metropolis–Hastings step with Gaussian proposal after each JKO step.

For all test distributions, we start with a Gaussian distribution $\mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } I )$ for a problem dependent scaling parameter $\sigma _ { 0 }$ . To parameterize the velocity fields v and reaction rates $g ,$ we use a ResNet with 6 blocks, where each block consists out of a two-layer network with SiLU activation and 512 hidden neurons. In order to simulate and diferentiate the ODE solutions, we use the rk4 solver from torchdiffeq library [9] with a step size of 0.1. We report the problem dependent hyperparameters in Table 1.

6.3 Comparisons. We compare our methods with all comparison methods considered in [25]. Despite the neural Wasserstein JKO scheme (neural JKO) and its importancecorrected version (neural JKO IC) from [25] this includes classical Monte Carlo samplers like a Metropolis adjusted Langevin sampling (MALA) and Hamiltonian Monte Carlo (HMC), see $\mathrm { e . g . }$ , [4, 47] and two deep-learning based sampling algorithms, namely denoising difusion samplers (DDS, [50]) and continual repeated annealed flow transport Monte Carlo (CRAFT, [39]).

6.4 Results. In order to evaluate the quality of the generated samples, we compute the energy distance [48] between 50000 generated and ground truth samples, which corresponds to the maximum mean discrepancy with respect to the negative distance kernel $k ( x , y ) = - \| x - y \|$ . Given generated samples $\mathbf { X } = ( X ^ { 1 } , . . . , X ^ { N } )$ and ground truth

<table><tr><td>Distribution</td><td>d</td><td> $h _ { 0 }$ </td><td>T</td><td>α</td><td>MH step size</td><td>K</td><td> $\sigma _ { 0 }$ </td></tr><tr><td>Mustache</td><td>2</td><td>0.0025</td><td>4.0</td><td>1.0</td><td>0.03</td><td>20</td><td>1.5</td></tr><tr><td>shifted 8 modes</td><td>2</td><td>0.0025</td><td>4.0</td><td>1.0</td><td>0.03</td><td>20</td><td>1.5</td></tr><tr><td>shifted 8 peaky</td><td>2</td><td>0.0025</td><td>4.0</td><td>1.0</td><td>0.03</td><td>20</td><td>1.5</td></tr><tr><td>GMM-10</td><td>10</td><td>0.0025</td><td>4.0</td><td>1.0</td><td>0.03</td><td>20</td><td>1.5</td></tr><tr><td>GMM-20</td><td>20</td><td>0.0025</td><td>4.0</td><td>1.0</td><td>0.03</td><td>10</td><td>1.5</td></tr><tr><td>GMM-50</td><td>50</td><td>0.0025</td><td>4.0</td><td>1.0</td><td>0.03</td><td>10</td><td>1.5</td></tr><tr><td>GMM-100</td><td>100</td><td>0.0025</td><td>4.0</td><td>1.0</td><td>0.03</td><td>10</td><td>1.5</td></tr></table>

Table 1: Hyperparameters for the WFR JKO algorithm: d is the dimension of the distribution, $h _ { 0 }$ is the initial step size in the WFR–JKO algorithm, $\tau$ is the growth of this step size, α is the parameter from the WFR metric, K the number of WFR–JKO steps used, and $\sigma _ { 0 }$ is the scaling of the initial distribution. In addition, we report the step sizes for solving the ODEs and for generating the proposals in the MH step.  
samples $\mathbf { Y } = ( Y ^ { 1 } , . . . , Y ^ { N } )$ , it is given by

$$
\mathcal { E } ( \mathbf { X } , \mathbf { Y } ) = \frac { 1 } { N ^ { 2 } } \sum _ { i , j = 1 } ^ { N } \| X ^ { i } - Y ^ { j } \| - \frac { 1 } { 2 N ^ { 2 } } \sum _ { i , j = 1 } ^ { N } \| X ^ { i } - X ^ { j } \| - \frac { 1 } { 2 N ^ { 2 } } \sum _ { i , j = 1 } ^ { N } \| Y ^ { i } - Y ^ { j } \| .
$$

The energy distance is a metric on the space of probability measures. In particular, a smaller energy distance indicates a better result. The results are given in Table 2. Moreover, we plot two-dimensional marginals of generated samples in Figure 1.

We observe that the WFR–JKO sampler is always either the best or second best method among the comparisons. Just the neural JKO IC method is competitive. This is expected, as the WFR JKO scheme formalizes the heuristic approach of “transporting and reweighting” from neural JKO IC and provides it with better theoretical guarantees. In addition, we note that the runtime of neural JKO IC depends exponentially on the number of resampling steps, see [25, Remark 4.5], while the WFR–JKO does not have this exponential behavior. All other methods degenerate for high dimensional multimodal target distributions.

<table><tr><td>Distribution</td><td>MALA</td><td>HMC</td><td>DDS</td><td>CRAFT</td><td></td><td></td><td>Neural JKO Neural JKO IC WFR JKO (ours) Sampling Error</td><td></td></tr><tr><td>Mustache</td><td> $4 . 6 \times 1 0 ^ { - 2 }$ </td><td> $1 . 7 \times 1 0 ^ { - 2 }$ </td><td> $6 . 9 \times 1 0 ^ { - 2 }$ </td><td> $9 . 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 8 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 2 . 9 \times 1 0 ^ { - 3 } }$ </td><td> $\mathbf { 1 . 2 \times 1 0 ^ { - 2 } }$ </td><td> $8 . 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>shifted 8 modes</td><td> $5 . 3 \times 1 0 ^ { - 3 }$ </td><td> $4 . 1 \times 1 0 ^ { - 5 }$ </td><td>1.2 × 10−2</td><td> $5 . 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 3 \times 1 0 ^ { - 1 }$ </td><td> $\mathbf { 1 . 2 \times 1 0 ^ { - 5 } }$ </td><td> $\mathbf { 1 . 6 \times 1 0 ^ { - 5 } }$ </td><td> $2 . 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>shifted 8 peaky</td><td> $1 . 3 \times 1 0 ^ { - 1 }$ </td><td> $1 . 2 \times 1 0 ^ { - 1 }$ </td><td> $1 . 1 \times 1 0 ^ { - 2 }$ </td><td> $5 . 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 3 \times 1 0 ^ { - 1 }$ </td><td> $\mathbf { 3 . 4 \times 1 0 ^ { - 5 } }$ </td><td> $\mathbf { 3 . 1 \times 1 0 ^ { - 5 } }$ </td><td> $2 . 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>GMM-10</td><td> $1 . 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 2 \times 1 0 ^ { - 2 }$ </td><td> $3 . 7 \times 1 0 ^ { - 3 }$ </td><td> $1 . 8 \times 1 0 ^ { - 1 }$ </td><td> $1 . 1 \times 1 0 ^ { - 2 }$ </td><td> $\mathbf { 5 . 3 \times 1 0 ^ { - 5 } }$ </td><td> $\mathbf { 3 . 2 \times 1 0 ^ { - 5 } }$ </td><td> $4 . 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>GMM-20</td><td> $9 . 1 \times 1 0 ^ { - 3 }$ </td><td> $9 . 1 \times 1 0 ^ { - 3 }$ </td><td> $5 . 0 \times 1 0 ^ { - 3 }$ </td><td> $5 . 4 \times 1 0 ^ { - 1 }$ </td><td> $1 . 0 \times 1 0 ^ { - 2 }$ </td><td> ${ \bf 1 . 1 \times 1 0 ^ { - 4 } }$ </td><td> $\mathbf { 1 . 0 \times 1 0 ^ { - 4 } }$ </td><td> $6 . 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>GMM-50</td><td> $2 . 4 \times 1 0 ^ { - 2 }$ </td><td> $2 . 4 \times 1 0 ^ { - 2 }$ </td><td> $2 . 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 8 \times 1 0 ^ { 0 }$ </td><td> $2 . 7 \times 1 0 ^ { - 2 }$ </td><td> ${ \bf 1 . 0 \times 1 0 ^ { - 4 } }$ </td><td> ${ \bf 4 . 1 \times 1 0 ^ { - 4 } }$ </td><td> $1 . 1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>GMM-100</td><td> $3 . 6 \times 1 0 ^ { - 2 }$ </td><td> $3 . 7 \times 1 0 ^ { - 2 }$ </td><td> $3 . 9 \times 1 0 ^ { - 2 }$ </td><td> $2 . 8 \times 1 0 ^ { 1 }$ </td><td> $4 . 7 \times 1 0 ^ { - 2 }$ </td><td> ${ \bf 6 . 0 \times 1 0 ^ { - 4 } }$ </td><td> $\mathbf { 1 . 7 \times 1 0 ^ { - 2 } }$ </td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td></tr></table>

Table 2: Energy distance between generated samples and ground truth samples. A smaller value indicates a better result. The best result for each test distribution is marked in green, the second-best is marked in blue.

LLM Usage. We used Claude Fable for exploring the literature corresponding to continuous time gradient flows in the Wasserstein–Fisher–Rao metric. We used diferent LLMs for improving the implementations.

Acknowledgments. CD acknowledges funding by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – project number 442047500/SFB 1481 Sparsity and Singular Structures. JH acknowledges funding by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), Project 572825596. GS acknowledges funding by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation), Project STE 571/23-1.

![](images/463b1788c44c7ff7565ba84cf0ac0635f39c93fe45dcc28a80c3267df4092970.jpg)  
Figure 1: Plot of the first two coordinates of generated and ground truth samples for diferent examples. First line: shifted 8 modes; second line: shifted 8 peaky; third line: GMM-10; fourth line: GMM-20.

## References

[1] L. Ambrosio, N. Gigli, and G. Savaré. Gradient Flows: in Metric Spaces and in the Space of Probability Measures. Springer Science & Business Media, 2005. (Cited on page 5.)

[2] D. Bakry and M. Émery. Difusions hypercontractives. In J. Azéma and M. Yor, editors, Séminaire de Probabilités XIX 1983/84, pages 177–206. Springer Berlin Heidelberg, 1985. (Cited on page 2.)

[3] J.-D. Benamou and Y. Brenier. A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem. Numerische Mathematik, 84:375–393, 2000. (Cited on page 5.)

[4] M. Betancourt. A conceptual introduction to Hamiltonian Monte Carlo. arXiv preprint arXiv:1701.02434, 2017. (Cited on page 18.)

[5] K. Bredies, M. Carioni, and S. Fanzon. A superposition principle for the inhomogeneous continuity equation with Hellinger–Kantorovich-regular coeficients. Communications in Partial Diferential Equations, 47(10):2023–2069, 2022. (Cited on page 6.)

[6] G. Cardoso, Y. J. el idrissi, S. L. Corf, and E. Moulines. Monte Carlo guided denoising difusion models for Bayesian linear inverse problems. In The Twelfth International Conference on Learning Representations, 2024. (Cited on page 1.)

[7] J. Chang, C. Duan, Y. Jiao, R. Li, J. Z. Yang, and C. Yuan. Provable difusion posterior sampling for Bayesian inversion, 2025. arXiv:2512.08022. (Cited on page 1.)

[8] J. Chemseddine, C. Wald, R. Duong, and G. Steidl. Neural sampling from Boltzmann densities: Fisher-Rao curves in the Wasserstein geometry. In The Thirteenth International Conference on Learning Representations, 2025. (Cited on page 12.)

[9] R. T. Q. Chen. torchdifeq, 2018. URL https://github.com/rtqichen/ torchdiffeq. (Cited on page 18.)

[10] S. Chewi, M. A. Erdogdu, M. Li, R. Shen, and M. S. Zhang. Analysis of Langevin Monte Carlo from Poincaré to Log-Sobolev. Foundations of Computational Mathematics, 25:1345–1395, 2025. (Cited on page 2.)

[11] S. Chewi, J. Niles-Weed, and P. Rigollet. Statistical Optimal Transport, volume 2364 of Lecture Notes in Mathematics (LNM). Springer Cham, first edition, 2025. (Cited on page 2.)

[12] L. Chizat, G. Peyré, B. Schmitzer, and F.-X. Vialard. An interpolating distance between optimal transport and Fisher-Rao metrics. Foundations of Computational Mathematics, 18:1–44, 2018. (Cited on pages 2 and 6.)

[13] A. D. Cobb and B. Jalaian. Scaling Hamiltonian Monte Carlo inference for Bayesian neural networks with symmetric splitting. In Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, volume 161 of Proceedings of Machine Learning Research, pages 675–685. PMLR, 2021. (Cited on page 1.)

[14] F. R. Crucinio and S. Pathiraja. Sequential monte carlo approximations of wasserstein– fisher–rao gradient flows. arXiv preprint arXiv:2506.05905, 2025. (Cited on page 8.)

[15] F. R. Crucinio and S. Pathiraja. Sequential Monte Carlo approximations of Wasserstein–Fisher–Rao gradient flows, 2026. arXiv:2506.05905. (Cited on page 2.)

[16] P. Del Moral, A. Doucet, and A. Jasra. Sequential Monte Carlo samplers. Journal of the Royal Statistical Society Series B: Statistical Methodology, 68(3):411–436, 2006. (Cited on pages 2 and 14.)

[17] Z. Ding, C. Duan, Y. Jiao, J. Z. Yang, C. Yuan, and P. Zhang. Nonlinear assimilation via score-based sequential Langevin sampling. IEEE Transactions on Information Theory, 72(10):7754–7784, 2026. (Cited on page 1.)

[18] A. Durmus and É. Moulines. High-dimensional Bayesian inference via the unadjusted Langevin algorithm. Bernoulli, 25(4A):2854–2882, 2019. (Cited on pages 1 and 2.)

[19] F. C. Fleißner. A minimizing movement approach to a class of scalar reaction–difusion equations. ESAIM: Control, Optimisation and Calculus of Variations, 27:18, 2021. (Cited on pages 3 and 8.)

[20] M. Gabrié, G. Rotskof, and E. Vanden-Eijnden. Adaptive Monte Carlo augmented with normalizing flows. Proceedings of the National Academy of Sciences, 119(10): e2109420119, 2022. (Cited on page 2.)

[21] T. O. Gallouët and L. Monsaingeon. A JKO splitting scheme for Kantorovich-Fisher-Rao gradient flows. SIAM Journal on Mathematical Analysis, 49(2):1100–1130, 2017. (Cited on pages 3 and 8.)

[22] P. Hagemann, J. Hertrich, and G. Steidl. Stochastic normalizing flows for inverse problems: A Markov chains viewpoint. SIAM/ASA Journal on Uncertainty Quantification, 10(3):1162–1190, 2022. (Cited on page 12.)

[23] W. K. Hastings. Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57(1):97–109, 1970. (Cited on page 3.)

[24] J. Heng, A. Doucet, and Y. Pokern. Gibbs flow for approximate transport with applications to Bayesian computation. Journal of the Royal Statistical Society Series B: Statistical Methodology, 83(1):156–187, 2021. (Cited on page 1.)

[25] J. Hertrich and R. Gruhlke. Importance corrected neural JKO sampling. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 23083–23119. PMLR, 13–19 Jul 2025. (Cited on pages 2, 3, 18, and 19.)

[26] M. F. Hutchinson. A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. Communications in Statistics—Simulation and Computation, 18(3):1059–1076, 1989. (Cited on page 13.)

[27] P. Izmailov, S. Vikram, M. D. Hofman, and A. G. G. Wilson. What are Bayesian neural network posteriors really like? In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 4629–4640. PMLR, 18–24 Jul 2021. (Cited on page 1.)

[28] R. Jordan, D. Kinderlehrer, and F. Otto. The variational formulation of the Fokker-Planck equation. SIAM J. Math. Anal., 29(1):1–17, 1998. (Cited on pages 1, 2, and 7.)

[29] D. P. Landau and K. Binder. A Guide to Monte Carlo Simulations in Statistical Physics. Cambridge University Press, fourth edition, 2014. (Cited on page 1.)

[30] V. Laschos and A. Mielke. Evolutionary variational inequalities on the Hellinger-Kantorovich and spherical Hellinger-Kantorovich spaces. Communications in Partial Diferential Equations, 51(2-3):193–235, 2026. doi: 10.1080/03605302.2026.2650715. (Cited on page 8.)

[31] M. Liero, A. Mielke, and G. Savaré. Optimal transport in competition with reaction: The Hellinger–Kantorovich distance and geodesic curves. SIAM Journal on Mathematical Analysis, 48(4):2869–2911, 2016. (Cited on page 2.)

[32] M. Liero, A. Mielke, and G. Savaré. Optimal entropy-transport problems and a new Hellinger–Kantorovich distance between positive measures. Inventiones Mathematicae, 211(3):969–1117, 2018. (Cited on pages 2 and 6.)

[33] M. Liero, A. Mielke, and G. Savaré. Fine properties of geodesics and geodesic λ- convexity for the Hellinger–Kantorovich distance. Archive for Rational Mechanics and Analysis, 247(112), 2023. (Cited on page 6.)

[34] Y. Lu, J. Lu, and J. Nolen. Accelerating Langevin sampling with birth-death, 2019. arXiv:1905.09863. (Cited on page 2.)

[35] Y. Lu, D. Slepčev, and L. Wang. Birth–death dynamics for sampling: global convergence, approximations and their asymptotics. Nonlinearity, 36(11):5731–5772, 2023. (Cited on pages 2 and 8.)

[36] S. Maniglia. Probabilistic representation and uniqueness results for measurevaluedsolutions of transport equations. J. Math. Pures Appli., 87(6):601—-626, 2007. (Cited on page 6.)

[37] S. T. Martin, A. Gagneux, P. Hagemann, and G. Steidl. PnP-Flow: Plug-and-play image restoration with flow matching. In The Thirteenth International Conference on Learning Representations, 2025. (Cited on page 1.)

[38] P.-G. Martinsson and J. Tropp. Randomized numerical linear algebra: Foundations & algorithms. Acta Numerca, 29:403–572, 2020. (Cited on page 14.)

[39] A. Matthews, M. Arbel, D. J. Rezende, and A. Doucet. Continual repeated annealed flow transport Monte Carlo. In International Conference on Machine Learning, pages 15196–15219. PMLR, 2022. (Cited on page 18.)

[40] L. I. Midgley, V. Stimper, G. N. C. Simm, B. Schölkopf, and J. M. Hernández-Lobato. Flow annealed importance sampling bootstrap. In The Eleventh International Conference on Learning Representations, 2023. (Cited on page 1.)

[41] A. Mielke and J.-J. Zhu. Hellinger-Kantorovich gradient flows: Global exponential decay of entropy functionals. arXiv preprint arXiv:2501.17049, 2025. (Cited on pages 3 and 8.)

[42] F. Noè, S. Olsson, J. Köhler, and H. Wu. Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. Science, 365(6457):eaaw1147, 2019. (Cited on page 1.)

[43] A. Oliviero-Durmus, Y. Janati, E. Moulines, M. Pereyra, and S. Reich. Generative modelling meets Bayesian inference: a new paradigm for inverse problems. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 383(2299):20240334, 2025. (Cited on page 1.)

[44] D. Onken, S. W. Fung, X. Li, and L. Ruthotto. OT-flow: Fast and accurate continuous normalizing flows via optimal transport. In AAAI Conference on Artificial Intelligence, volume 35, pages 9223–9232, 2021. (Cited on page 2.)

[45] A. M. Overstall, D. C. Woods, and B. M. Parker. Bayesian optimal design for ordinary diferential equation models with application in biological science. Journal of the American Statistical Association, 115(530):583–598, 2020. (Cited on page 1.)

[46] G. O. Roberts and J. S. Rosenthal. General state space Markov chains and MCMC algorithms. Probability Surveys, 1:20 – 71, 2004. (Cited on page 15.)

[47] G. O. Roberts and R. L. Tweedie. Exponential convergence of Langevin distributions and their discrete approximations. Bernoulli, 2(4):341–363, 1996. (Cited on page 18.)

[48] G. Székely. E-statistics: The energy of statistical samples. Techical Report, Bowling Green University, 2002. (Cited on page 18.)

[49] L. Tan and J. Lu. Accelerate Langevin sampling with birth-death process and exploration component. SIAM/ASA Journal on Uncertainty Quantification, 13(3): 1265–1293, 2025. (Cited on page 2.)

[50] F. Vargas, W. Grathwohl, and A. Doucet. Denoising difusion samplers. International Conference on Learning Representations, 2023. (Cited on page 18.)

[51] S. Vempala and A. Wibisono. Rapid convergence of the unadjusted Langevin algorithm: isoperimetry sufices. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’ Alché- Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. (Cited on page 2.)

[52] A. Vidal, S. Wu Fung, L. Tenorio, S. Osher, and L. Nurbekyan. Taming hyperparameter tuning in continuous normalizing flows using the JKO scheme. Scientific Reports, 13(1):4501, 2023. (Cited on page 2.)

[53] M. Welling and Y. W. Teh. Bayesian learning via stochastic gradient Langevin dynamics. In International Conference on Machine Learning, pages 681–688, 2011. (Cited on page 2.)

[54] C. Xu, X. Cheng, and Y. Xie. Normalizing flow neural networks by JKO scheme. In Advances in Neural Information Processing Systems, volume 36, pages 47379–47405. Curran Associates, Inc., 2023. (Cited on pages 2 and 12.)

[55] J.-J. Zhu and A. Mielke. Kernel approximation of Fisher-Rao gradient flows. arXiv preprint arXiv:2410.20622, 2024. (Cited on page 8.)

## A Auxiliary Proofs

Proposition A.1. Let $r > 0$ and $\pi \in \mathcal { M } ( \mathbb { R } ^ { d } )$ . Then, the set $\mathcal { K } _ { r } : = \{ \rho \in \mathcal { M } ( \mathbb { R } ^ { d } )$ $\mathrm { K L } ( \rho \| \pi ) \leq r \}$ is weakly sequentially compact.

Proof. Recalling the convex duality x log $\displaystyle x - x + 1 = \operatorname* { s u p } _ { a \in \mathbb { R } } \{ a x - e ^ { a } + 1 \}$ , the KL divergence can be rewritten as

$$
\mathrm { K L } ( \rho | | \pi ) = \operatorname* { s u p } _ { f \in C _ { b } ( \mathbb { R } ^ { d } ) } \int f \mathrm { ~ d } \rho - \int e ^ { f } - 1 \mathrm { ~ d } \pi .
$$

Hence, we get, for $\rho \in { \cal K } _ { r }$ , that

$$
\int _ { \mathbb { R } ^ { d } } f d \rho \leq r + \int _ { \mathbb { R } ^ { d } } ( e ^ { f } - 1 ) d \pi \qquad { \mathrm { f o r ~ e v e r y ~ } } f \in C _ { b } ( \mathbb { R } ^ { d } ) .
$$

Taking $f \equiv 1$ gives

$$
\rho ( \mathbb { R } ^ { d } ) \leq r + ( e - 1 ) \pi ( \mathbb { R } ^ { d } ) ,
$$

and consequently su $\begin{array} { r } { \mathsf { \Gamma } \mathsf { l p } _ { \rho \in { \mathcal { K } _ { r } } } \rho ( \mathbb { R } ^ { d } ) \le C < \infty } \end{array}$

Next, we show that $\kappa _ { c }$ is uniformly tight. Let $\varepsilon > 0$ be arbitary small. Choose $a > 0$ suficiently large such that

$$
\frac { r } { a } < \frac { \varepsilon } { 2 } .
$$

Since $\pi$ is a finite measure on $\mathbb { R } ^ { d }$ , it is tight. Thus, there exists a compact set $K \subset \mathbb { R } ^ { d }$ such that

$$
\frac { e ^ { a } - 1 } { a } \pi ( K ^ { c } ) < \frac { \varepsilon } { 2 } .
$$

Using the measurable function $f = a \mathbf { 1 } _ { K ^ { c } }$ , we obtain, for every $\rho \in { \mathcal { K } } _ { r }$ , that

$$
r \geq a \rho ( K ^ { c } ) - ( e ^ { a } - 1 ) \pi ( K ^ { c } ) .
$$

This implies

$$
\rho ( K ^ { c } ) \leq \frac { c } { a } + \frac { e ^ { a } - 1 } { a } \pi ( K ^ { c } ) < \varepsilon
$$

and then

$$
\operatorname* { s u p } _ { \rho \in { \mathcal K } _ { c } } \rho ( K ^ { c } ) < \varepsilon ,
$$

so that $\kappa _ { c }$ is uniformly tight. Now let $( \rho _ { n } ) _ { n \geq 1 }$ be any sequence in $\kappa _ { c }$ . By Prokhorov’s theorem it follows that there exists a subsequence, still denoted by $\left( \rho _ { n } \right)$ , and some $\rho \in \mathcal { M } _ { + } ( \mathbb { R } ^ { d } )$ such that $\rho _ { n }  \rho .$

It remains to verify that $\rho \in \mathcal { K } _ { r }$ . Since KL is weakly lower semicontinuouswe obtain by $\rho _ { n } \in \mathcal { K } _ { r }$ for every n that

$$
\mathrm { K L } ( \rho \| \pi ) \leq \operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } \mathrm { K L } ( \rho _ { n } \| \pi ) \leq r .
$$

Thus $\rho \in { \mathcal { K } } _ { r }$ which finishes the proof.

## B Pseudo-code of the Algorithms

We provide pseudo-code for the computations in Section 4.2 in Algorithm 1, 2 and 3.

```tcl
Algorithm 1: Evaluate the empirical risk
Input: particles $X _ { k } ^ { i }$ with log-density log $\rho _ { k } ( X _ { k } ^ { i } )$ , velocity ${ \overline { { v ^ { k } } } } .$ , reaction $\overline { { g ^ { k } } }$ and
log-mass $\rho _ { k } ( \mathbb { R } ^ { d } )$
Output: The empirical risk $\widehat { \mathcal { L } } _ { k } ^ { \mathrm { W F R } } ( \theta )$
1 Sample i.i.d. d-dimensional Rademacher variables $\xi _ { 1 } , \dots , \xi _ { N } \sim \mathrm { R a d } ^ { d }$
2 Simulate the ODE system from $t = 0$ to $t = 1$ for $i = 1 , \ldots , N ;$
$\frac { \mathrm { d } } { \mathrm { d } t } \phi _ { t } ( X _ { k } ^ { i } ) = v _ { t } \circ \phi _ { t } ( X _ { k } ^ { i } ) , \quad \phi _ { 0 } ( X _ { k } ^ { i } ) = X _ { k } ^ { i } ,$
$\frac { \mathrm { d } } { \mathrm { d } t } \psi _ { t } ( X _ { k } ^ { i } ) = g _ { t } \circ \phi _ { t } ( X _ { k } ^ { i } ) , \quad \psi _ { 0 } ( X _ { k } ^ { i } ) = \log \rho _ { k } ( \mathbb { R } ^ { d } ) , \quad w _ { t } ( X _ { k } ^ { i } ) = \exp ( \psi _ { t } ( X _ { k } ^ { i } ) )$
$\frac { \mathrm { d } } { \mathrm { d } t } a _ { t } ( X _ { k } ^ { i } ) = w _ { t } ( X _ { k } ^ { i } ) \Big \{ \| v _ { t } \circ \phi _ { t } ( X _ { k } ^ { i } ) \| _ { 2 } ^ { 2 } + \frac { 1 } { \lambda } ( g _ { t } \circ \phi _ { t } ( X _ { k } ^ { i } ) ) ^ { 2 } \Big \} , \quad a _ { 0 } ( X _ { k } ^ { i } ) = 0 ,$
$\frac { \mathrm { d } } { \mathrm { d } t } \eta _ { t } ( X _ { k } ^ { i } ) = \xi _ { i } ^ { \top } \nabla v _ { t } \circ \phi _ { t } ( X _ { k } ^ { i } ) \xi _ { i } , \quad \eta _ { t } ( X _ { k } ^ { i } ) = 0 .$
3 Compute the per-particle loss:
$\widehat { \mathcal { L } } _ { \mathrm { W F R } } ^ { k + 1 } ( \theta ; X _ { k } ^ { i } ) \gets \frac { 1 } { 2 h } a _ { 1 } ( X _ { k } ^ { i } ) - w _ { 1 } ( X _ { k } ^ { i } ) \eta _ { 1 } ( X _ { k } ^ { i } )$
4 $+ w _ { 1 } ( X _ { k } ^ { i } ) \Big \{ \log w _ { 1 } ( X _ { k } ^ { i } ) + V \circ \phi _ { 1 } ( X _ { k } ^ { i } ) + \log \rho _ { k } ( X _ { k } ^ { i } ) - \log \rho _ { k } ( \mathbb { R } ^ { d } ) - 1 \Big \}$
5 Compute the empirical risk $\begin{array} { r } { \hat { \mathcal { L } } _ { \mathrm { W F R } } ^ { k } ( \theta ) \gets \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \hat { \mathcal { L } } _ { \mathrm { W F R } } ^ { k + 1 } ( \theta ; X _ { k } ^ { i } ) } \end{array}$
6 return $\hat { \mathcal { L } } _ { \mathrm { W F R } } ^ { k + 1 } ( \theta )$
```

Algorithm 2: Applying a JKO step   
Input: particles $X _ { k } ^ { i }$ , log-densities log $\rho _ { k } ( X _ { k } ^ { 1 } )$ , velocity $v ^ { k }$ and reaction $\overline { { g ^ { k } } }$   
Output: weighted particles $Y _ { k + 1 } ^ { i }$ with weights $w _ { k + 1 } ^ { i }$ and log-densities $\rho _ { k + 1 } \big ( Y _ { k + 1 } ^ { i } \big )$   
log-mass log $\rho _ { k + 1 } ( \mathbb { R } ^ { d } )$   
1 Simulate the ODE system from $t = 0$ to $t = 1$ for $i = 1 , \ldots , N ;$   
$\frac { \mathrm { d } } { \mathrm { d } t } \phi _ { t } ( X _ { k } ^ { i } ) = v _ { t } \circ \phi _ { t } ( X _ { k } ^ { i } ) , \quad \phi _ { 0 } ( X _ { k } ^ { i } ) = X _ { k } ^ { i } ,$   
$\frac { \mathrm { d } } { \mathrm { d } t } \psi _ { t } ( X _ { k } ^ { i } ) = g _ { t } \circ \phi _ { t } ( X _ { k } ^ { i } ) , \quad \psi _ { 0 } ( X _ { k } ^ { i } ) = \log \rho _ { k } ( \mathbb { R } ^ { d } ) .$   
$\frac { \mathrm { d } } { \mathrm { d } t } \zeta _ { t } ( X _ { k } ^ { i } ) = - \operatorname { d i v } { v _ { t } } \circ \phi _ { t } ( X _ { k } ^ { i } ) , \quad \zeta _ { 0 } ( X _ { k } ^ { i } ) = \log \rho _ { k } ( X _ { k } ^ { i } )$   
2 Update the particles $Y _ { k + 1 } ^ { i }  \phi _ { 1 } ( X _ { k } ^ { i } )$ for $i = 1 , \ldots , N .$   
3 Evaluate the weights $w _ { k + 1 } ^ { i }  \exp ( \psi _ { . } ( X _ { k } ^ { i } ) )$ for $i = 1 , \ldots , N .$   
4 Evaluate the log-density log $\rho _ { k + 1 } ( Y _ { k + 1 } ^ { i } ) \gets \psi _ { 1 } ( X _ { k } ^ { i } ) + \zeta _ { 1 } ( X _ { k } ^ { i } )$   
5 Evaluate the log-mass log $\begin{array} { r } { \rho _ { k + 1 } ( \mathbb { R } ^ { d } )  \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { k + 1 } ^ { i } . } \end{array}$   
6 return samples $Y _ { k + 1 } ^ { i }$ , weights $w _ { i }$ , log-density log $\rho _ { k + 1 } ( Y _ { k + 1 } ^ { i } ) )$ and total mass   
$\rho _ { k + 1 } ( \mathbb { R } ^ { d } )$

Algorithm 3: Resampling   
Input: weighted particles $Y _ { k + 1 } ^ { i }$ with weights $w _ { k + 1 } ^ { i }$ and log-densities $\rho _ { k + 1 } \big ( Y _ { k + 1 } ^ { i } \big )$   
Output: unweighted particles $X _ { k + 1 } ^ { i }$ with log-densities $\rho _ { k + 1 } \big ( X _ { k + 1 } ^ { i } \big )$   
1 Normalize the weights $( \widetilde w _ { k + 1 } ^ { 1 } , \dots , \dot { \widetilde w } _ { k + 1 } ^ { N } )$ ← softmax $( w _ { k + 1 } ^ { 1 } , \ldots , w _ { k + 1 } ^ { N } )$   
2 Resample $( j _ { 1 } , \dots , j _ { N } ) \sim$ Multinormial $( \widetilde w _ { k + 1 } ^ { 1 } , \dots , \widetilde w _ { k + 1 } ^ { N } )$   
3 $X _ { k + 1 } ^ { i }  X _ { k + 1 } ^ { j _ { i } }$ for $i = 1 , \ldots , N .$   
4 log $\rho _ { k + 1 } ( X _ { k + 1 } ^ { i } ) \gets$ log $\rho _ { k + 1 } ( X _ { k + 1 } ^ { j _ { i } } )$ for $i = 1 , \ldots , N .$   
5 return $( X _ { k + 1 } ^ { i } , \log \rho _ { k + 1 } ( X _ { k + 1 } ^ { i } ) )$ and $\rho _ { k + 1 } ( \mathbb { R } ^ { d } )$