# PnP-CF: A Consensus Flow-Matching Solver for General Inverse Problems

Dibyanshu Kumar TH Würzburg-Schweinfurt, CAIRO University of Würzburg dibyanshu.kumar@thws.de

Magda Gregorová TH Würzburg-Schweinfurt, CAIRO magda.gregorova@thws.de

## Abstract

Plug-and-Play (PnP) methods using continuous-time deep generative models as learned data distribution priors enable synthesizing high-quality solutions of modelbased inverse problems. The quality of the solutions depends critically on the interplay and consensus between the inverse problem solver and the generative model sampler. We revisit a recently introduced flow-based PnP framework, which we refer to as PnP-CF (Consensus Flow), and extend its application to nonlinear inverse problems, including those with learned neural-network forward operators. PnP-CF combines a flow-matching denoiser with (i) ADMM-like dual variable updates and (ii) a time-adaptive pullback onto the intermediate flow probability path that combines a stochastic and a deterministic component. We discuss how these two features stabilize the algorithm and promote consensus, and demonstrate on a set of experiments that PnP-CF not only solves standard linear inverse problems with quality at least comparable to state-of-the-art baselines, but also extends gracefully to nonlinear inverse problems, including forward operators realized by pretrained neural networks, for which existing methods produce unsatisfactory results or fail completely.

## 1 Introduction

A fundamental task in model-based inverse problems is recovering an unknown signal $\mathbf { x } \in \mathbb { R } ^ { m }$ from measurements $\mathbf { y } \in \mathbb { R } ^ { n }$ satisfying a (noisy) forward model

$$
\begin{array} { r } { \mathbf { y } = \mathcal { A } ( \mathbf { x } , \pmb { \eta } ) , } \end{array}\tag{1}
$$

where A denotes the measurement process and η represents measurement noise. The forward model may be linear or nonlinear and is typically non-injective, so recovering x from y is ill-posed and requires prior information about the unknown signal.

In a Bayesian formulation, this prior information is modeled by a density $p ( \mathbf { x } )$ over the signal space, while the forward model induces the likelihood $p ( \mathbf { y } \mid \mathbf { x } )$ . A reconstruction can then be obtained either by sampling from the posterior $p ( \mathbf { x } \mid \mathbf { y } ) , \mathbf { e . g . , b y }$ training conditional score-based diffusion models [Batzolis et al., 2021, Saharia et al., 2022], or through maximum-a-posteriori (MAP) estimation. The latter yields the minimization problem

$$
\hat { \mathbf { x } } = \underset { \mathbf { x } } { \arg \operatorname* { m i n } } \left\{ - \log p ( \mathbf { y } \mid \mathbf { x } ) - \log p ( \mathbf { x } ) \right\} = \underset { \mathbf { x } } { \arg \operatorname* { m i n } } \left\{ l ( \mathbf { y } ; \mathbf { x } ) + s ( \mathbf { x } ) \right\} ,\tag{2}
$$

where $l ( \mathbf { y } ; \mathbf { x } ) = - \log p ( \mathbf { y } \mid \mathbf { x } )$ measures consistency with the forward model and $s ( \mathbf { x } ) = - \log p ( \mathbf { x } )$ encodes the prior. This formulation allows a single unconditional generative model p(x) to serve as a reusable prior across multiple inverse problems and forward models. We refer to $l ( \mathbf { y } ; \mathbf { x } )$ and $s ( \mathbf { x } )$ as the forward-consistency and prior-consistency terms, respectively. <sup>1</sup>

In this paper we revisit the flow-based PnP framework introduced by Kumar and Gregorová [2026] for solving the MAP problem in equation 2, which we refer to as PnP-CF (Plug-and-Play Consensus Flow), and extend its application to nonlinear inverse problems, including those involving learned neural-network forward operators. The algorithmic formulation of PnP-CF, recalled in section 3, follows Kumar and Gregorová [2026]; our contributions lie in its extension to and evaluation on nonlinear and learned forward models.

PnP-CF decouples the forward-consistency and prior-consistency updates within an iterative scheme, similar to the PnP formulation of Venkatakrishnan et al. [2013], while realizing the prior-consistency update through a pretrained flow-matching denoiser, avoiding backpropagation through the generative prior. However, integrating forward-consistency optimization with a generative prior such as flow matching poses a significant challenge in maintaining consensus between the two, since the forwardconsistency step is driven by the measurements, whereas the flow prior is designed to operate along a structured family of intermediate probability paths.

PnP-CF addresses this challenge through two complementary mechanisms: (i) a decoupled ADMMlike (Alternating Direction Method of Multipliers [Boyd et al., 2010]) variable-splitting structure with dual-variable updates and (ii) a time-adaptive pullback toward the intermediate flow probability path using complementary stochastic and deterministic components (section 3). We argue that these mechanisms stabilize the interaction between the forward-consistency solver and the flow-matching denoiser, thereby promoting consensus between the measurement-driven iterate and the learned flow trajectory (section 4.1). While methods such as PnP-Flow [Martin et al., 2025] and FLOWER [Pourya et al., 2026] use flow-based denoisers as priors in similar PnP frameworks, we demonstrate through a series of experiments that PnP-CF not only achieves strong performance on standard linear inverse problems but also extends naturally to nonlinear forward operators, broadening the applicability of flow-based inverse solvers beyond the predominantly linear settings considered in existing work (section 4.2 and table 1). We further evaluate PnP-CF on segmentation-guided reconstruction, where the forward operator is a complex neural network that discards substantial image information, demonstrating that PnP-CF extends beyond analytical measurement models to highly non-injective learned operators (section 4.3 and table 3). Relative to Kumar and Gregorová [2026], we thus contribute a forward-consistency solver for nonlinear and learned operators, the first nonlinear evaluation of several flow-based baselines, and a segmentation-guided formulation with a highly non-injective neural forward operator. Our code is available at https://github.com/ dibyanshu100/PnP\_ConsensusFlow.git.

## 2 Background

## 2.1 Flow Matching and Straight Trajectories

Flow matching [Lipman et al., 2023] is a generative framework for learning a deterministic transformation that transports samples from a simple base distribution $p _ { 0 }$ , typically a standard Gaussian, to a target data distribution $p _ { 1 }$ . The central idea is to specify a continuous path of intermediate distributions $( p _ { t } ) _ { t \in [ 0 , 1 ] }$ that interpolates between $p _ { 0 }$ and $p _ { 1 }$ , and then to learn a time-dependent velocity field $\mathbf { v } _ { \pmb { \theta } }$ whose flow ψ follows this path.

A typical choice is to construct $p _ { t }$ from straight-line path Liu et al. [2023] between samples $( \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } ) \sim$ π drawn from a coupling $\pi \in \Gamma ( p _ { 0 } , p _ { 1 } )$ , i.e., a joint distribution with marginals $p _ { 0 }$ and $p _ { 1 }$ such as the independent coupling $\pi = p _ { 0 } \otimes p _ { 1 }$ , defined as

$$
\mathbf { x } _ { t } = \left( 1 - t \right) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } , \quad t \in [ 0 , 1 ] ,\tag{3}
$$

and $p _ { t }$ is the implicit distribution of $\mathbf { x } _ { t }$ . This path moves from a noise sample at $t = 0$ to a data sample at $t = 1$ with constant conditional velocity ${ \bf x } _ { 1 } - { \bf x } _ { 0 }$

The flow associated with this probability path can be characterized by a flowmap ψ<sub>t</sub> generated by the marginal velocity field $\mathbf { v } _ { t } ( \mathbf { x } ) : = \mathbb { E } _ { \boldsymbol \pi } [ \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \mid \mathbf { x } _ { t } = \mathbf { x } ]$ , solving

$$
\frac { d } { d t } \psi _ { t } ( \mathbf { x } _ { 0 } ) = \mathbf { v } _ { t } ( \psi _ { t } ( \mathbf { x } _ { 0 } ) ) , \quad \psi _ { 0 } ( \mathbf { x } _ { 0 } ) = \mathbf { x } _ { 0 } ,\tag{4}
$$

such that $p _ { t } = ( \psi _ { t } ) _ { \# } p _ { 0 }$ , and in particular $\psi _ { 1 }$ maps $p _ { 0 }$ to $p _ { 1 }$ . In practice, $\mathbf { v } _ { t }$ is unknown and we approximate it using a neural network $\mathbf { v } _ { \pmb { \theta } }$ . For straight-line trajectories, the per-pair target velocity is

$\mathbf { v } _ { t } ^ { * } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } ) = \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } ;$ regressing onto this target recovers the marginal field $\mathbf { v } _ { t }$ in expectation, leading to the conditional flow matching objective

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ( \pmb { \theta } ) = \mathbb { E } _ { ( \mathbf { x } _ { 0 } , \mathbf { x } _ { 1 } ) \sim \pi , t \sim \mathcal { U } ( 0 , 1 ) } \left[ \left\| \mathbf { v } _ { \pmb { \theta } } \big ( ( 1 - t ) \mathbf { x } _ { 0 } + t \mathbf { x } _ { 1 } , t \big ) - \big ( \mathbf { x } _ { 1 } - \mathbf { x } _ { 0 } \big ) \right\| ^ { 2 } \right] . } \end{array}\tag{5}
$$

For inference, the flow from time t to 1 can be written as $\begin{array} { r } { \psi _ { 1 } ( \mathbf { x } _ { t } ) = \mathbf { x } _ { t } + \int _ { t } ^ { 1 } \mathbf { v } _ { \pmb { \theta } } ( \mathbf { x } _ { q } , q ) d q } \end{array}$ . This integral can be approximated using an explicit Euler step, yielding the one-step estimate

$$
\psi _ { 1 } ( \mathbf x _ { t } ) \approx \tilde { \psi } _ { 1 } ( \mathbf x _ { t } ) = \mathbf x _ { t } + ( 1 - t ) \mathbf v _ { \pmb \theta } ( \mathbf x _ { t } , t ) .\tag{6}
$$

The resulting flow operator $\tilde { \psi } _ { 1 }$ can be interpreted as a one-step, time-dependent denoiser, mapping $\mathbf { x } _ { t }$ toward the data distribution at $t = 1$ . We adopt an operator-based view of PnP, using $\tilde { \psi } _ { 1 }$ as the denoising operator, i.e., $\mathcal { D } _ { t } ( \mathbf { x } _ { t } ) : = \tilde { \psi } _ { 1 } ( \mathbf { x } _ { t } )$ , and describe its role within the PnP-CF framework in section 3.

## 2.2 Plug-and-Play Algorithms

Venkatakrishnan et al. [2013] introduced the PnP algorithm, which addresses inverse problems by combining a forward-consistency term with a learned prior implemented as a denoising operator. A key feature of PnP is that the prior is trained independently of the inverse problem, typically as a standalone denoiser, and can therefore be reused across different forward operators. This decoupling enables the use of powerful learned priors without requiring problem-specific retraining. Early PnP methods relied on classical or CNN-based denoisers [Zhang et al., 2017, Ryu et al., 2019], while subsequent methods incorporated expressive generative priors including GANs [Bora et al., 2017] and diffusion models [Zhu et al., 2023, Mardani et al., 2024, Li et al., 2025, Shrestha and Fu, 2026].

Motivated by the strong empirical performance of flow matching, several recent works have adapted the PnP framework to use flow-matching models as learned priors. Martin et al. [2025] propose PnP-Flow, which integrates a pretrained flow-matching denoiser into a Forward-Backward Splitting (FBS) scheme. A related approach, FLOWER, proposed by Pourya et al. [2026], combines a flow-based denoising step with a proximal forward-consistency update and provides a Bayesian interpretation of the resulting procedure. We collectively refer to these approaches as FBS-based methods. While these methods demonstrate the effectiveness of flow-matching priors within PnP formulations for relatively simple linear forward operators, the lack of consensus between measurement-driven updates and the evolving flow trajectory limits their effectiveness on more challenging problems (see section 4.2). This motivates PnP-CF, which is specifically designed to maintain consensus between these components, enabling flow-based PnP algorithms to handle more complex forward models, including nonlinear and learned neural-network forward operators. In the next section, we describe the methodological formulation of PnP-CF.

## 3 Method

Akin to ADMM and similar in spirit to the original PnP framework of Venkatakrishnan et al. [2013], we decouple the forward-consistency $l ( \mathbf { y } ; \mathbf { x } )$ and prior-consistency $s ( \mathbf { x } )$ terms in equation 2 by introducing an auxiliary variable z and enforcing $\mathbf { x } = \mathbf { z }$

The corresponding augmented Lagrangian is

$$
\mathcal { L } _ { \rho } ( \mathbf { x } , \mathbf { z } , \mathbf { u } ) = l ( \mathbf { y } ; \mathbf { x } ) + s ( \mathbf { z } ) + \frac { \rho } { 2 } \| \mathbf { x } - \mathbf { z } + \mathbf { u } \| _ { 2 } ^ { 2 } ,\tag{7}
$$

where u is the scaled dual variable and $\rho > 0$ is a penalty parameter that controls the strength of the consensus constraint. The resulting constrained problem is solved iteratively by alternating updates of the primal variables x, z and the dual variable u. At each iteration, the primal variables are updated through two separate subproblems:

$$
\hat { \mathbf { x } }  \underbrace { \arg \operatorname* { m i n } _ { \mathbf { x } } l ( \mathbf { y } ; \mathbf { x } ) + \frac { \rho } { 2 } \| \mathbf { x } - \tilde { \mathbf { x } } \| _ { 2 } ^ { 2 } } _ { \mathcal { F } ( \mathbf { y } , \tilde { \mathbf { x } } ) \mathrm { f o r w a r d - c o n s i s t e n c y ~ o p e r a t o r } } , \qquad \hat { \mathbf { z } }  \underbrace { \arg \operatorname* { m i n } _ { \mathbf { z } } s ( \mathbf { z } ) + \frac { \rho } { 2 } \| \tilde { \mathbf { z } } - \mathbf { z } \| _ { 2 } ^ { 2 } } _ { \mathcal { D } ( \tilde { \mathbf { z } } ) \mathrm { d e n o i s i n g ~ o p e r a t o r } } ,\tag{8}
$$

followed by the scaled dual update u $ \mathbf { u } + ( \hat { \mathbf { x } } - \hat { \mathbf { z } } )$

![](images/6d9bf5ba5780a2b52f7d8bad87ec7d1b574bfc9f86bdcbbe3032f20a92d887aa.jpg)  
Figure 1: Phases of PnP-CF. Left: Starting from z˜, the denoiser $\mathcal { D } ( \tilde { \mathbf { z } } )$ produces zˆ (green). The dual correction (gray) shifts zˆ by u, and the forward-consistency operator $\mathcal { F } ( \mathbf { y } , \tilde { \mathbf { x } } )$ maps it to xˆ (blue), pulling it towards the region consistent with $\mathbf { y }$ under $l ( \mathbf { y } ; \mathbf { x } )$ $R i g h t \cdot$ The time-adaptive pullback constructs the next z˜. The deterministic component $\epsilon _ { d } ,$ computed from the previous z˜ and zˆ, combines with $\epsilon _ { s } \sim p _ { t = 0 }$ to form ϵ. The dual-corrected point xˆ + u (gray) and the ϵ together produces the new z˜, which serves as the input to the denoiser at the next iteration.

The two operators in equation $^ 8$ can be interpreted as proximal updates [Parikh and Boyd, 2014] centered at the anchor points x˜ and z˜, respectively. The forward-consistency operator $\mathcal { F }$ depends only on the forward model through $l ( \mathbf { y } ; \mathbf { x } )$ , while $\mathcal { D } _ { t } ( \mathbf { x } _ { t } ) : = \tilde { \psi } _ { 1 } ( \mathbf { x } _ { t } )$ denotes the one-step flow denoiser introduced in section 2.1. In the following, we omit the time dependence and write D for notational simplicity.

A critical design choice of the PnP algorithms is the construction of the anchor points $\tilde { \mathbf { x } }$ and z˜ connecting the alternating updates. We construct x˜ using $\mathcal { D } ,$ the one-step flow denoiser, where we, with a slight abuse of notation, align the discretized ODE time-steps $t \in [ 0 , 1 )$ with the main algorithm iterations $k = 0 , \ldots , K - 1$ . Importantly and in contrast to the FBS-based methods, we further adjust this point through the dual variable u to form the consensus point $\tilde { \mathbf { x } }  \mathcal { D } ( \tilde { \mathbf { z } } ) - \mathbf { u }$ serving as the input for the next iteration of the forward-consistency update (figure 1 left).

Conceptually, the inputs $\tilde { \mathbf { x } }$ and outputs xˆ of the forward-consistency operator $\mathcal { F }$ live in the endpoint $p _ { t = 1 }$ of the learned probability path $( p _ { t } ) _ { t \in [ 0 , 1 ] }$ of the flow-matching model. However, as the proximal update $\hat { \mathbf { x } } = \mathcal { F } ( \mathbf { y } , \tilde { \mathbf { x } } )$ pushes the intermediate sample x˜ towards the region consistent with the forward model, the output xˆ may end up in a low-probability region under the learned flow-matching prior (off-distribution) and should be further aligned before applying the denoiser D in the subsequent iteration.

Since the denoiser D operates over samples drawn from intermediate distributions $( p _ { t } ) _ { t < 1 }$ we construct z˜ by a time-adaptive pullback step along the flow-matching straight path, $\mathrm { e . g . }$ . Liu et al. [2023], $\tilde { \mathbf { z } } \gets ( 1 - t ) \boldsymbol { \epsilon } + t ( \hat { \mathbf { x } } + \mathbf { u } )$ . While similar pullback has been used in the FBS-based methods, our update differs in two critical aspects. First, the ADMM-like structure allows to further stabilize the algorithm by enforcing consensus between the primal updates through the dual variable u. Second, we construct the latent sample ϵ from the flow base distribution $p _ { t = 0 }$ as a time-adaptive combination of a stochastic $\epsilon _ { s }$ (used in FBS-based methods), and a deterministic component $\epsilon _ { d }$ (not used in FBS-based methods) computed from the pullback point of the previous iteration as shown in the right part of figure 1:

$$
\epsilon  \frac { t \epsilon _ { d } + ( 1 - t ) \epsilon _ { s } } { \sqrt { t ^ { 2 } + ( 1 - t ) ^ { 2 } } } , \quad \epsilon _ { s } \sim p _ { t = 0 } = \mathcal { N } ( \mathbf { 0 } , I ) , \quad \epsilon _ { d }  \frac { \tilde { \mathbf { z } } - t \hat { \mathbf { z } } } { 1 - t } \ ,\tag{9}
$$

where we again use the simplified notation $t \in [ 0 , 1 )$ aligned with the main algorithm iterations $k = 0 , \ldots , \bar { K } - 1$

The two components in equation 9 play complementary roles. The stochastic component $\epsilon _ { s } ,$ independently sampled from the flow base distribution $p _ { t = 0 }$ , can move the iterate across different flow trajectories and therefore promotes exploration. In contrast, the deterministic component $\epsilon _ { d }$ inferred from the previous pullback point preserves information about the trajectory already followed, keeping subsequent updates closer to the current flow trajectory. The denominator $\sqrt { t ^ { 2 } + ( 1 - t ) ^ { 2 } }$ in equation 9 normalizes the linear combination of the two components to a unit variance and maintains a consistent scale across iterations. This design follows a simple principle: at early iterations (small t), the algorithm emphasizes stochastic exploration via $\epsilon _ { s } ,$ allowing the iterate to traverse different regions of the solution space. As t → 1, the weight progressively shifts toward $\epsilon _ { d } ,$ , encouraging refinement along a consistent flow trajectory aligned with the denoiser. Overall, this method enables the algorithm to both recover from poor initializations and commit to a stable trajectory as it approaches a solution that satisfies the forward-consistency term.

Algorithm 1 PnP-CF with time-adaptive pullback (adapted from Kumar and Gregorová [2026])   
1: Input: Flow denoiser $\mathcal { D } _ { t } ,$ , observation y, forward-consistency $l ( \mathbf { y } ; \mathbf { x } )$ , penalty parameter $\rho _ { k }$   
number of iterations $K \in \mathbb { N } ,$ , time steps $\dot { t } _ { k } = k / K$   
2: Initialize: $\tilde { \mathbf { x } } ^ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , I ) , \mathbf { u } ^ { 0 } = \mathbf { 0 } , \epsilon ^ { \tilde { 0 } } = \tilde { \mathbf { x } } ^ { 0 }$   
3: for $k = 0 , \ldots , K - 1$ do   
4: $\hat { \mathbf { x } } ^ { k + 1 } =$ arg min<sub>x</sub> $\begin{array} { r } { l ( \mathbf { y } ; \mathbf { x } ) + \frac { \rho _ { k } } { 2 } \| \mathbf { x } - \tilde { \mathbf { x } } ^ { k } \| _ { 2 } ^ { 2 } } \end{array}$ ▷ forward-consistency   
5: $\tilde { \mathbf { z } } ^ { k } = ( 1 - t _ { k } ) \epsilon ^ { k } + t _ { k } ( \hat { \mathbf { x } } ^ { k + 1 } \bar { + } \mathbf { u } ^ { k } )$ ▷ time-adaptive pullback   
6: $\hat { \mathbf { z } } ^ { k + 1 } = \mathcal { D } _ { t _ { k } } ( \tilde { \mathbf { z } } ^ { k } )$ ▷ prior-consistency   
7: $\begin{array} { r } { \epsilon _ { d } ^ { k } = \frac { \tilde { \mathbf { z } } ^ { k } - t _ { k } \hat { \mathbf { z } } ^ { k + 1 } } { 1 - t _ { k } } , \qquad \epsilon _ { s } ^ { k } \sim \mathcal { N } ( \mathbf { 0 } , I ) } \end{array}$ ▷ deterministic and stochastic components   
8: $\begin{array} { r } { \epsilon ^ { k + 1 } = \frac { t _ { k } \epsilon _ { d } ^ { k } + ( 1 - t _ { k } ) \epsilon _ { s } ^ { k } } { \Gamma } } \end{array}$ ▷ adaptive latent update   
$\sqrt { t _ { k } ^ { 2 } + ( 1 - t _ { k } ) ^ { 2 } }$   
9: $\mathbf { u } ^ { k + 1 } = \mathbf { \check { u } } ^ { k } + \hat { \mathbf { x } } ^ { k + 1 } - \hat { \mathbf { z } } ^ { k + 1 }$ ▷ dual update   
10: $\tilde { \mathbf { x } } ^ { k + 1 } = \hat { \mathbf { z } } ^ { k + 1 } - \mathbf { u } ^ { k + 1 }$ ▷ consensus anchor   
11: end for   
12: return $\hat { \mathbf { z } } ^ { K }$

Finally and complementary to the time-adaptive pullback, the dual update u $ \mathbf { u } + ( \hat { \mathbf { x } } - \hat { \mathbf { z } } )$ provides an explicit mechanism for enforcing consensus between the forward and prior-consistent solutions. The residual $\hat { \mathbf { x } } - \hat { \mathbf { z } }$ measures their disagreement, and its accumulation in u carries this information across iterations, so that inconsistencies are not discarded after a single update. The dual variable then feeds this accumulated discrepancy back into both sides of the alternating scheme: it shifts the denoised solution through $\tilde { \mathbf { x } } = \hat { \mathbf { z } } - \mathbf { u }$ before the next forward-consistency update and adjusts the pullback endpoint through xˆ + u before the subsequent prior-consistency step. In this way, u acts as a persistent correction signal that counteracts systematic drift between the two updates and progressively drives them toward a common solution. Together with the time-adaptive pullback, this provides PnP-CF with a mechanism for maintaining agreement between forward-consistency and the evolving flow trajectory throughout optimization.

In summary, relative to the FBS-based formulations, PnP-CF introduces two mechanisms for maintaining consistency throughout the iterative reconstruction: (i) the dual variable u, which accumulates disagreement between the forward-consistency and denoising updates and feeds it back into subsequent iterations, and (ii) the deterministic component $\epsilon _ { d }$ in the time-adaptive pullback, which retains information from the previous pullback and promotes refinement along a consistent flow trajectory. Together, these mechanisms provide explicit cross-iteration consensus and trajectory continuity, while allowing forward and prior-consistency subproblem to be optimized more flexibly. This makes the formulation applicable to more involved forward-consistency problems, including nonlinear forward operators (section 4). We further analyze these design choices in section 4.1. The complete PnP-CF procedure is summarized in algorithm 1, with the iteration index k reintroduced to make the update sequence explicit.

## 4 Experiments and Results

In this section, we evaluate PnP-CF on inverse problems defined by a range of linear and nonlinear forward models, $\begin{array} { r } { \mathbf { y } = \mathcal { A } ( \mathbf { x } ) + \boldsymbol { \eta } . } \end{array}$ , where $\pmb { \eta } \sim \hat { \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } I )$ and A denotes either a linear or nonlinear operator. Under this additive Gaussian-noise model, the forward-consistency term in equation 2 becomes $\begin{array} { r } { l ( \mathbf { y } ; \mathbf { x } ) = \frac { 1 } { 2 \sigma ^ { 2 } } \| \mathcal { A } ( \mathbf { x } ) - \mathbf { y } \| _ { 2 } ^ { 2 } } \end{array}$ . For linear forward operators, the corresponding forwardconsistency proximal $\mathcal { F } ( { \bf y } , \tilde { { \bf x } } )$ reduces to a convex quadratic problem, which we solve efficiently using conjugate gradients [Hestenes and Stiefel, 1952]. For nonlinear forward operators, the resulting forward-consistency subproblem is generally nonquadratic and does not admit a simple solution. We therefore solve it approximately using gradient-based optimization through the differentiable forward operator $\mathcal { A } ( \mathbf { x } )$ (see appendix A for details).

For our experiments, we evaluate on randomly selected test images from CelebA [Yang et al., 2015] $( 1 2 8 \times 1 2 8 )$ and AFHQ-Cat [Choi et al., 2020] (256 × 256), using pretrained flow-matching models from Martin et al. [2025], and report PSNR and SSIM for reconstruction fidelity and LPIPS for perceptual quality. We consider three linear inverse problems: box inpainting, random inpainting, and super-resolution. For box inpainting, the forward operator A masks a centered $6 0 \times 6 0$ region for CelebA and a $1 2 0 \times 1 2 0$ region for AFHQ-Cat. For random inpainting, A masks 85% of the pixels uniformly at random. For super-resolution, A downsamples the image by factors of ×4 and ×8 for CelebA and AFHQ-Cat, respectively. We add Gaussian measurement noise with $\sigma = 0 . 0 5$ for box inpainting and super-resolution, and $\sigma = 0 . 0 1$ for random inpainting.

We additionally consider three nonlinear inverse problems: nonlinear deblurring, phase retrieval, and HDR reconstruction. For nonlinear deblurring, the forward operator is a pretrained KernelWizard network using the GOPRO-wVAE model [Tran et al., 2021], which maps a sharp image to a blurred one via a nonlinear network forward pass conditioned on a latent blur code. For phase retrieval, the forward operator is $\mathcal { A } ( \mathbf { x } ) = | T ( \bar { \mathcal { P } } ( \mathbf { x } ) )$ |, where $\mathcal { P }$ denotes symmetric zero-padding and $\tau$ is the orthonormal two-dimensional Fourier transform; we use an oversampling factor of 2. For HDR reconstruction, we use the nonlinear saturation operator $\mathcal { A } ( \mathbf { x } ) = \mathrm { c l i p } ( 2 \mathbf { x } , - 1 , 1 )$ . Gaussian measurement noise is added with $\sigma = 0 . 0 5$ for nonlinear deblurring and HDR reconstruction and $\sigma = 0 . 0 1$ for phase retrieval.

The main hyperparameter of PnP-CF is the penalty $\rho ,$ which controls the consensus strength between the forward-consistency and prior updates. In contrast to Kumar and Gregorová [2026], who use a constant penalty and a linear time grid, we introduce the schedules $\bar { \rho } _ { k } ~ = ~ \rho / ( 1 - t _ { k } ) ^ { \beta }$ and $t _ { k } = ( k / K ) ^ { \gamma }$ , where $\beta$ controls the growth of the penalty and $\gamma$ controls the progression along the flow trajectory. Since the pullback step weights $\epsilon _ { d }$ by $t _ { k } .$ choosing $\gamma < 1$ shifts the emphasis toward the deterministic component $\epsilon _ { d }$ earlier, whereas $\gamma > 1$ retains the stochastic component $\epsilon _ { s }$ for longer (see appendix B.3.1). Following Martin et al. [2025], we average the outputs of several denoiser calls per iteration. The hyperparameter values used in our experiments are reported in appendix B.3.

## 4.1 Analysis of PnP-CF

![](images/29d3e455fe266260b11cb80b96c56a05b05df28cf4db26f63ce5c4583f272d4a.jpg)

![](images/10812f639867c5d44e49a8eb4abf4ecf7a81702c694d4982335151c3ab485e8a.jpg)  
Figure 2: Ablation of PnP-CF update strategies on the box-inpainting inverse problem. Left: qualitative comparison of stochastic-only, deterministic-only, and time-adaptive pullback strategies. Right: effect of the dual variable u on the forward-consistency and denoising updates, shown through PSNR and SSIM trajectories over iterations, averaged over 100 images.

We first build intuition for the components of PnP-CF algorithm through a qualitative comparison on a linear box-inpainting problem. In the left side of figure 2 we show the effects of the stochastic and deterministic components of the pullback step in equation 9. Using only the stochastic component $\epsilon _ { s }$ (“stochastic-only”) causes the iterates to oscillate across different randomly sampled flow trajectories. Averaging these inconsistent samples yields blurry reconstructions. Using only the deterministic component $\epsilon _ { d }$ (“deterministic-only”), the algorithm becomes highly dependent on the random initial flow state, restricting exploration to a single solution path that may be difficult to align with the forward-model constraints. Finally, the proposed time-adaptive combination promotes exploration in early iterations and gradually transitions to trajectory-consistent updates, yielding reconstructions that are both sharp and semantically consistent.

Table 1: Reconstruction quality metrics for linear (left panel) and nonlinear (right panel) inverse problems on CelebA and AFHQ-Cat. Best results are shown in bold.
<table><tr><td colspan="2"></td><td colspan="3">CelebA</td><td colspan="3">AFHQ-Cat</td></tr><tr><td>Task</td><td>Method</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td rowspan="9">Inupnung Box</td><td>PnP-CF (ours)</td><td>26.36</td><td>0.05</td><td>0.89</td><td>22.65</td><td>0.13</td><td>0.82</td></tr><tr><td>PnP-Flow</td><td>24.75</td><td>0.08</td><td>0.88</td><td>21.41</td><td>0.21</td><td>0.82</td></tr><tr><td>FLOWER</td><td>25.12</td><td>0.06</td><td>0.88</td><td>21.43</td><td>0.16</td><td>0.83</td></tr><tr><td>FlowDPS</td><td>24.22</td><td>0.08</td><td>0.86</td><td>19.41</td><td>0.21</td><td>0.79</td></tr><tr><td>D-Flow</td><td>24.60</td><td>0.08</td><td>0.80</td><td>21.68</td><td>0.19</td><td>0.71</td></tr><tr><td>Flow Priors</td><td>21.98</td><td>0.11</td><td>0.81</td><td>20.24</td><td>0.20</td><td>0.76</td></tr><tr><td>OT-ODE</td><td>24.07</td><td>0.07</td><td>0.85</td><td>19.45</td><td>0.16</td><td>0.78</td></tr><tr><td>PnP-CF (ours)</td><td>26.45</td><td>0.08</td><td>0.79</td><td>23.72</td><td>0.27</td><td>0.66</td></tr><tr><td>PnP-Flow</td><td>19.97</td><td>0.22</td><td>0.61</td><td>19.74</td><td>0.38</td><td>0.55</td></tr><tr><td>Reson FLOWER Super</td><td>22.63</td><td>0.15</td><td>0.70</td><td>19.30</td><td>0.45</td><td>0.55</td></tr><tr><td>FlowDPS</td><td>24.44</td><td>0.10</td><td>0.76</td><td>20.59</td><td>0.41</td><td>0.60</td></tr><tr><td>D-Flow</td><td>24.13</td><td>0.09</td><td>0.70</td><td>20.84</td><td>0.29</td><td>0.48</td></tr><tr><td>Flow Priors</td><td>23.54</td><td>0.19</td><td>0.56</td><td>17.77</td><td>0.55</td><td>0.35</td></tr><tr><td>OT-ODE</td><td>25.33</td><td>0.06</td><td>0.78</td><td>22.47</td><td>0.20</td><td>0.58</td></tr><tr><td rowspan="9">Ranndom Inupning</td><td>PnP-CF (ours)</td><td>29.93</td><td>0.04</td><td>0.89</td><td>30.72</td><td>0.07</td><td>0.85</td></tr><tr><td>PnP-Flow</td><td>29.04</td><td>0.05</td><td>0.87</td><td>30.33</td><td>0.10</td><td>0.85</td></tr><tr><td>FLOWER</td><td>29.04</td><td>0.05</td><td>0.89</td><td>30.39</td><td>0.10</td><td>0.85</td></tr><tr><td>FlowDPS</td><td>28.51</td><td>0.05</td><td>0.87</td><td>29.72</td><td>0.12</td><td>0.83</td></tr><tr><td>D-Flow</td><td>28.84</td><td>0.04</td><td>0.87</td><td>29.33</td><td>0.09</td><td>0.81</td></tr><tr><td>Flow Priors</td><td>25.65</td><td>0.08</td><td>0.82</td><td>28.37</td><td>0.11</td><td>0.82</td></tr><tr><td>OT-ODE</td><td>24.30</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.12</td><td>0.75</td><td>26.26</td><td>0.16</td><td>0.74</td></tr></table>

<table><tr><td colspan="2"></td><td colspan="3">CelebA</td><td colspan="3">AFHQ-Cat</td></tr><tr><td colspan="2">Task Method</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td rowspan="9">Recutiton HR</td><td>PnP-CF (ours)</td><td>27.59</td><td>0.04</td><td>0.85</td><td>26.29</td><td>0.08</td><td>0.86</td></tr><tr><td>PnP-Flow</td><td>22.24</td><td>0.07</td><td>0.81</td><td>24.77</td><td>0.12</td><td>0.84</td></tr><tr><td>FLOWER</td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td>FlowDPS</td><td>22.40</td><td>0.10</td><td>0.80</td><td>25.25</td><td>0.18</td><td>0.81</td></tr><tr><td>D-Flow</td><td>24.16</td><td>0.09</td><td>0.77</td><td>25.59</td><td>0.16</td><td>0.75</td></tr><tr><td>Flow Priors</td><td>一</td><td>=</td><td>=</td><td>1</td><td>=</td><td>=</td></tr><tr><td>OT-ODE</td><td>=</td><td>=</td><td>一</td><td>一</td><td>=</td><td>–</td></tr><tr><td>PnP-CF (ours)</td><td>30.33</td><td>0.11</td><td>0.80</td><td>25.27</td><td>0.22</td><td>0.70</td></tr><tr><td>PnP-Flow</td><td>22.57</td><td>0.22</td><td>0.62</td><td>16.19</td><td>0.45</td><td>0.41</td></tr><tr><td>Phase Reival</td><td>FLOWER</td><td>16.29</td><td></td><td></td><td></td><td></td></tr><tr><td>FlowDPS</td><td></td><td>0.37</td><td>0.47</td><td>12.14</td><td>0.63</td><td>0.36</td></tr><tr><td>D-Flow</td><td>18.03</td><td>0.28</td><td>0.49</td><td>11.41</td><td>0.59</td><td>0.28</td></tr><tr><td>Flow Priors</td><td>一</td><td>=</td><td>=</td><td>=</td><td>=</td><td>=</td></tr><tr><td>OT-ODE</td><td>一</td><td>=</td><td>一</td><td>–</td><td>一</td><td>–</td></tr><tr><td rowspan="7">Noniiear Debring</td><td>PnP-CF (ours)</td><td>27.09</td><td>0.09</td><td>0.79</td><td>22.28</td><td>0.29</td><td>0.58</td></tr><tr><td>PnP-Flow</td><td>25.43</td><td>0.21</td><td>0.72</td><td>20.60</td><td>0.48</td><td>0.54</td></tr><tr><td>FLOWER</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FlowDPS</td><td>25.55</td><td>0.14</td><td>0.74</td><td>20.41</td><td>0.42</td><td>0.54</td></tr><tr><td>D-Flow</td><td>27.02</td><td>0.09</td><td>0.75</td><td>23.45</td><td>0.20</td><td>0.56</td></tr><tr><td>Flow Priors</td><td>=</td><td>=</td><td>一</td><td>=</td><td>=</td><td>=</td></tr><tr><td>OT-ODE</td><td>1</td><td>1</td><td>1</td><td>一</td><td>一</td><td>1</td></tr></table>

In the right side of figure 2, we show the effects of the ADMM-like structure of PnP-CF method on the same box-inpainting task, comparing with and without the dual variable u updates. Omitting u reduces the algorithm to FBS-style approach, with the time-adaptive pullback retained to isolate the effect of the dual update. The graphs report the PSNR and SSIM at each iteration, averaged over 100 samples. Without u, the forward-consistency and denoising updates are solved independently at each iteration, resulting in slower convergence and lower reconstruction quality in terms of both PSNR and SSIM. With u, the accumulated discrepancy progressively aligns the two subproblems, producing a faster and more stable improvement in reconstruction quality. We show a more detailed ablation in appendix D.

![](images/dd340559f7a3d9858c70aedb1f8e2656bed22e4c27bcb1d84ef1166b18aacf66.jpg)  
Figure 3: Evolution of the primal residual $\| \hat { \mathbf { x } } ^ { k } - \hat { \mathbf { z } } ^ { k } \| _ { 2 }$ across linear and nonlinear inverse problems on AFHQ-Cat (left) and CelebA (right).

We next examine the convergence behavior of PnP-CF through the primal residual $r _ { \mathrm { p r i } } ^ { k } : = \lVert \hat { \mathbf { x } } ^ { k } - \hat { \mathbf { z } } ^ { k } \rVert _ { 2 }$ As shown in figure 3, the residual decreases substantially across all linear and nonlinear tasks on both datasets, indicating progressive consensus between the forward and prior-consistency variables. Since the prior update is implemented by a learned denoiser rather than an exact proximal operator, classical ADMM convergence guarantees [Boyd et al., 2010] do not directly apply; we therefore interpret these results as empirical evidence of convergence.

![](images/54a76f9290b87158dc61ce6102d29b8578ac619a0b13e54eb067da54f12932ed.jpg)  
Figure 4: Comparison of image restoration methods on box inpainting, super-resolution, and random inpainting, with additional qualitative results provided in appendix F. Our method yields visually sharp reconstructions while preserving structural consistency across tasks.

## 4.2 Comparison with Baseline Methods

We compare against six flow-based inverse solvers: OT-ODE [Pokle et al., 2024], D-Flow [Ben-Hamu et al., 2024], Flow Priors [Zhang et al., 2024], PnP-Flow [Martin et al., 2025], FLOWER [Pourya et al., 2026], and FlowDPS [Kim et al., 2025]. Among these, only D-Flow reports results on nonlinear experiments. We extend PnP-Flow and FlowDPS to nonlinear forward operators ourselves (appendix B.4), providing, to our knowledge, their first nonlinear evaluation. OT-ODE and Flow Priors are derived specifically for linear operators and provide no mechanism for nonlinear measurements, while FLOWER only sketches a theoretical nonlinear extension via Langevin sampling without implementation or evaluation; we therefore exclude these three methods from the nonlinear comparison. For linear inverse problems, we use stronger degradations than prior baseline evaluations to assess performance under more challenging measurement conditions. Since the reported results are not directly comparable, we rerun all baselines on our test instances using their released implementations and recommended hyperparameter settings. For completeness, we also report results under the milder settings of Martin et al. [2025], Pourya et al. [2026] in appendix E.3.

Table 1 reports quantitative results averaged over 100 test images. PnP-CF achieves the best performance on most tasks and metrics, with the largest margins on nonlinear problems. FBS-based PnP methods and FlowDPS remain competitive on inpainting but degrade substantially on super-resolution and nonlinear tasks. D-Flow achieves comparable performance on nonlinear deblurring, but is the least computationally efficient method overall.

Figure 4 and right side of figure 5 show qualitative comparisons for the linear and nonlinear inverse problems, respectively. For the linear tasks, several baselines recover the overall image structure but tend to lose fine details or introduce visible artifacts, particularly in box inpainting and superresolution, whereas PnP-CF produces sharper reconstructions while better preserving structural consistency. This advantage extends to the nonlinear problems, where PnP-CF maintains consistently high reconstruction quality, while FBS-based methods show more pronounced degradation.

Table 2 reports per-batch runtime and peak GPU memory on HDR reconstruction for CelebA, using a batch size of 4 and averaging over 25 batches. PnP-CF remains computationally efficient, incurring only a small runtime overhead over PnP-Flow while maintaining a comparable peak memory footprint. This efficiency comes directly from the design of PnP-CF: the flow matching denoiser is used only for the forward prior update and is never differentiated through. Consequently, the additional nonlinear forward-consistency steps require gradients only through the forward operator, without storing or backpropagating through the flow trajectory. This leads to a substantial computational advantage over trajectory-optimization methods

Table 2: Per batch runtime and peak GPU memory (HDR recon. - CelebA)
<table><tr><td>Method</td><td colspan="2">Time (s) Mem (GB)</td></tr><tr><td>PnP-CF (ours)</td><td>17.62</td><td>0.30</td></tr><tr><td>PnP-Flow</td><td>16.40</td><td>0.30</td></tr><tr><td>D-Flow</td><td>71.42</td><td>13.34</td></tr><tr><td>FlowDPS</td><td>11.14</td><td>0.32</td></tr></table>

such as D-Flow, which repeatedly integrates the ODE and backpropagates through the resulting trajectory while optimizing the initial noise. Thus, PnP-CF extends flow-based reconstruction to complex nonlinear forward models while retaining the lightweight computational profile of PnP-style methods. A more detailed analysis of computational complexity, together with runtime and memory results across inverse problems, is provided in appendix C.

![](images/b5c2c54852dc725294be06cef2e70ca0432af2303db791c0a13e9bd3b8cb0459.jpg)  
Figure 5: Qualitative results. Left: Segmentation-guided reconstruction results for D-Flow and PnP-CF. Right: Qualitative reconstruction on nonlinear inverse problems, including HDR reconstruction, phase retrieval, and nonlinear deblurring with additional results in appendix G.

## 4.3 Segmentation-guided reconstruction

To further test the flexibility of PnP-CF beyond inverse problems with an additive Gaussian measurement model, we consider segmentation-guided reconstruction on CelebA-HQ [Karras et al., 2018]. This setting is substantially more challenging: the forward operator is a pretrained segmentation network, and is highly non-injective, since many visually distinct images can produce the same semantic segmentation mask.

Specifically, A(x) is a pretrained segmentation network that outputs, for each pixel, a distribution over C = 19 semantic classes. Since these observations are categorical rather than corrupted by additive Gaussian noise, we replace the squared-error term with the corresponding negative log-likelihood of a categorical model, i.e., the cross-entropy loss $\begin{array} { r } { l ( \mathbf { y } ; \mathbf { x } ) = - \sum _ { i } \sum _ { c = 1 } ^ { C } \mathbf { y } _ { i , c } \log \mathcal { A } ( \mathbf { x } ) _ { i , c } , } \end{array}$ , where i indexes pixels and y denotes the target segmentation mask. Similar to the other nonlinear forward operators, we approximate the corresponding proximal update using gradient-based optimization of the resulting forward-consistency objective.

The ambiguity of the segmentation mask makes pixelwise reconstruction metrics difficult to interpret. We therefore omit PSNR and instead report LPIPS and SSIM as perceptual and structural measures relative to the reference image, and additionally report mIoU, which measures consistency with the observed segmentation by comparing the segmentation mask predicted from the reconstructed image with the target segmentation mask. Table 3 shows that PnP-CF stands out by achieving both high mIoU and reconstruction quality. Among the baselines, D-Flow is the only method to attain a similarly high mIoU, indicating strong adherence to the segmentation constraint, but it performs poorly in reconstruction quality as shown in figure 5 (left), where we compare PnP-CF with D-Flow. PnP-Flow and FlowDPS achieve substantially lower mIoU and fail to reliably satisfy the target segmentation constraint. We therefore omit them from the main qualitative figure and provide their reconstructions in appendix H. Qualitatively, PnP-Flow exhibits severe mode collapse, whereas FlowDPS produces plausible but weakly conditioned samples with limited correspondence to the target segmentation.

Table 3: Segmentation-guided reconstruction results.
<table><tr><td>Method</td><td>LPIPS↓</td><td>SSIM↑</td><td>mIoU↑</td></tr><tr><td>PnP-CF</td><td>0.27</td><td>0.41</td><td>0.99</td></tr><tr><td>PnP-Flow</td><td>0.35</td><td>0.28</td><td>0.37</td></tr><tr><td>D-Flow</td><td>0.38</td><td>0.11</td><td>0.98</td></tr><tr><td>FlowDPS</td><td>0.36</td><td>0.28</td><td>0.33</td></tr></table>

Overall, segmentation-guided reconstruction demonstrates a key strength of PnP-CF: its ability to handle complex learned forward operators while preserving both measurement consistency and reconstruction quality. Among the evaluated methods, PnP-CF is the only one to achieve this balance for the segmentation operator, showing that the framework extends naturally beyond standard linear and nonlinear inverse problems to substantially more complex neural forward models.

## 5 Conclusion and Limitations

We extended the flow-based PnP framework of Kumar and Gregorová [2026], here termed PnP-CF, to nonlinear inverse problems, including forward operators realized by pretrained neural networks. Its two key mechanisms, the ADMM-like dual variable and the time-adaptive pullback onto the flow probability path, promote consensus between the forward and prior-consistency updates, and our experiments show that this consensus carries over to nonlinear forward-consistency problems. Across linear and nonlinear tasks, PnP-CF achieves strong reconstruction quality and measurement consistency while retaining the low runtime and memory footprint of PnP-style methods; on nonlinear problems in particular, none of the evaluated baselines matches its performance consistently across tasks. Segmentation-guided reconstruction further shows that PnP-CF can handle highly non-injective learned forward operators beyond standard analytical measurement models. A limitation of the current work is the lack of rigorous convergence guarantees. Although the primal residuals provide strong empirical evidence of convergence, formal analysis remains challenging because the learned flow denoiser is implicit, nonlinear, and nonconvex. The current formulation also assumes differentiable forward operators, since forward consistency relies on gradient based optimization. Extending PnP-CF to nondifferentiable forward models is an important direction for future work.

## References

Larry Armijo. Minimization of functions having lipschitz continuous first partial derivatives. Pacific Journal ofmathematics, 16(1):1–3, 1966.

Georgios Batzolis, Jan Stanczuk, Carola-Bibiane Schönlieb, and Christian Etmann. Conditional image generation with score-based diffusion models. arXiv preprint arXiv:2111.13606, 2021.

Heli Ben-Hamu, Omri Puny, Itai Gat, Brian Karrer, Uriel Singer, and Yaron Lipman. D-flow: Differentiating through flows for controlled generation. In International Conference on Learning Representations, 2024.

Ashish Bora, Ajil Jalal, Eric Price, and Alexandros G. Dimakis. Compressed sensing using generative models. In International Conference on Machine Learning, 2017.

Stephen Boyd and Lieven Vandenberghe. Convex Optimization. Cambridge University Press, 2004.

Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends in Machine Learning, 2010.

Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020.

Hyungjin Chung, Jeongsol Kim, Michael T. Mccann, Marc L. Klasky, and Jong Chul Ye. Diffusion posterior sampling for general noisy inverse problems. In International Conference on Learning Representations, 2023.

Magnus R Hestenes and Eduard Stiefel. Methods of conjugate gradients for solving linear systems. Journal of research of the National Bureau of Standards, 1952.

Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In International Conference on Learning Representations, 2018.

Jeongsol Kim, Bryan Sangwoo Kim, and Jong Chul Ye. FlowDPS: Flow-driven posterior sampling for inverse problems. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 12328–12337, 2025.

Dibyanshu Kumar and Magda Gregorová. Plug-and-play admm for inverse problems with flow matching denoiser. In Advances in Neural Information Processing Systems, 2026.

Cheng-Han Lee, Ziwei Liu, Lingyun Wu, and Ping Luo. Maskgan: Towards diverse and interactive facial image manipulation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Xiang Li, Soo Min Kwon, Shijun Liang, Ismail R. Alkhouri, Saiprasad Ravishankar, and Qing Qu. Decoupled data consistency with diffusion purification for image restoration. arXiv:2403.06054, 2025.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

Morteza Mardani, Jiaming Song, Jan Kautz, and Arash Vahdat. A variational perspective on solving inverse problems with diffusion models. In International Conference on Learning Representations, 2024.

Ségolène Martin, Anne Gagneux, Paul Hagemann, and Gabriele Steidl. Pnp-flow: Plug-and-play image restoration with flow matching. In International Conference on Learning Representations, 2025.

Neal Parikh and Stephen Boyd. Proximal algorithms. Foundations and Trends in optimization, 1(3): 127–239, 2014.

Ashwini Pokle, Matthew J. Muckley, Ricky T. Q. Chen, and Brian Karrer. Training-free linear image inverses via flows. In International Conference on Learning Representations, 2024.

Mehrsa Pourya, Bassam El Rawas, and Michael Unser. FLOWER: A flow-matching solver for inverse problems. In International Conference on Learning Representations, 2026.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computerassisted intervention. Springer, 2015.

Ernest K. Ryu, Jialin Liu, Sicheng Wang, Xiaohan Chen, Zhangyang Wang, and Wotao Yin. Plugand-play methods provably converge with properly trained denoisers. In International Conference on Machine Learning, 2019.

Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence, 45(4):4713–4726, 2022.

Rajesh Shrestha and Xiao Fu. Taming score-based denoisers in admm: A convergent plug-and-play framework. In International Conference on Learning Representations, 2026.

Phong Tran, Anh Tuan Tran, Quynh Phung, and Minh Hoai. Explore image deblurring via encoded blur kernel space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.

Singanallur V Venkatakrishnan, Charles A Bouman, and Brendt Wohlberg. Plug-and-play priors for model based reconstruction. In 2013 IEEE Global Conference on Signal and Information Processing. IEEE, 2013.

Shuo Yang, Ping Luo, Chen Change Loy, and Xiaoou Tang. From facial parts responses to face detection: A deep learning approach. In IEEE International Conference on Computer Vision, 2015.

Kai Zhang, Wangmeng Zuo, Yunjin Chen, Deyu Meng, and Lei Zhang. Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE Transactions on Image Processing, 2017.

Yasi Zhang, Peiyu Yu, Yaxuan Zhu, Yingshan Chang, Feng Gao, Ying Nian Wu, and Oscar Leong. Flow priors for linear inverse problems via iterative corrupted trajectory matching. In Advances in Neural Information Processing Systems, 2024.

Yuanzhi Zhu, Kai Zhang, Jingyun Liang, Jiezhang Cao, Bihan Wen, Radu Timofte, and Luc Van Gool. Denoising diffusion models for plug-and-play image restoration. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

## A Forward-Consistency Solver

## A.1 The x-Update under Additive Gaussian Noise

Under additive Gaussian noise with variance $\sigma ^ { 2 }$ , the forward-consistency term takes the form of a scaled squared residual,

$$
l ( \mathbf { y } ; \mathbf { x } ) = \frac { 1 } { 2 \sigma ^ { 2 } } \| \boldsymbol { A } ( \mathbf { x } ) - \mathbf { y } \| _ { 2 } ^ { 2 } .\tag{10}
$$

Inserting this expression into the x-update of equation 8 leads to

$$
\hat { \mathbf { x } } ^ { k + 1 } = \arg \operatorname* { m i n } _ { \mathbf { x } } ~ \frac { 1 } { 2 \sigma ^ { 2 } } \| \boldsymbol { A } ( \mathbf { x } ) - \mathbf { y } \| _ { 2 } ^ { 2 } + \frac { \rho _ { k } } { 2 } \| \mathbf { x } - \tilde { \mathbf { x } } ^ { k } \| _ { 2 } ^ { 2 } .\tag{11}
$$

The objective balances two terms: the data-fidelity term penalizes deviations of $\mathcal { A } ( \mathbf { x } )$ from the measurements, while the proximal term keeps x close to the current ADMM anchor $\tilde { \mathbf { x } } ^ { k }$ , with a strength set by the scheduled penalty $\rho _ { k }$ of the main algorithm. How equation 11 is solved depends on the forward operator. For linear ${ \mathcal { A } } ,$ the problem is a strongly convex quadratic, which we solve with conjugate gradients (CG; appendix $\mathbf { A } . 2 )$ . For nonlinear ${ \mathcal { A } } ,$ no closed-form minimizer is available, and we instead run a bounded number of gradient-descent iterations in which the forward-consistency gradient is obtained through a vector-Jacobian product (appendix A.3).

## A.2 Linear Forward Operators: Conjugate Gradient

## A.2.1 Normal Equations

For a linear operator ${ \mathcal { A } } ,$ the objective in equation 11 is a strongly convex quadratic, so its minimizer is unique and is characterized by a vanishing gradient,

$$
\nabla _ { \mathbf x } \left[ \frac 1 { 2 \sigma ^ { 2 } } \| \mathbfcal { A } \mathbf x - \mathbf y \| _ { 2 } ^ { 2 } + \frac { \rho _ { k } } { 2 } \| \mathbf x - \tilde { \mathbf x } ^ { k } \| _ { 2 } ^ { 2 } \right] = \mathbf 0 .\tag{12}
$$

Expanding the gradient and rearranging the terms gives the linear system

$$
\underbrace { \left( { \frac { 1 } { \sigma ^ { 2 } } } { \mathcal { A } } ^ { \top } { \mathcal { A } } + \rho _ { k } { \cal I } \right) } _ { \mathcal { M } } \mathbf { x } = \underbrace { { \frac { 1 } { \sigma ^ { 2 } } } { \mathcal { A } } ^ { \top } \mathbf { y } + \rho _ { k } { \tilde { \mathbf { x } } } ^ { k } } _ { \mathbf { b } } ,\tag{13}
$$

which we write compactly as $\mathcal { M } \mathbf { x } = \mathbf { b }$ , with

$$
\boldsymbol { \mathcal { M } } ( \mathbf { x } ) = \frac { 1 } { \sigma ^ { 2 } } \boldsymbol { \mathcal { A } } ^ { \top } ( \boldsymbol { \mathcal { A } } \mathbf { x } ) + \rho _ { k } \mathbf { x } , \qquad \mathbf { b } = \frac { 1 } { \sigma ^ { 2 } } \boldsymbol { \mathcal { A } } ^ { \top } \mathbf { y } + \rho _ { k } \tilde { \mathbf { x } } ^ { k } .\tag{14}
$$

The operator M is symmetric, since both $A ^ { \top } A$ and $\rho _ { k } { \cal I }$ are self-adjoint. It is also positive definite, since for every $\mathbf { x } \neq \mathbf { 0 }$

$$
\langle \mathbf { x } , \mathcal { M } ( \mathbf { x } ) \rangle = \frac { 1 } { \sigma ^ { 2 } } \| \pmb { \cal A } \mathbf { x } \| _ { 2 } ^ { 2 } + \rho _ { k } \| \mathbf { x } \| _ { 2 } ^ { 2 } > 0 .\tag{15}
$$

Consequently, equation 13 has a unique solution and can be solved with CG.

## A.2.2 CG Implementation

At the image resolutions considered here, the system matrix is too large to be formed or inverted explicitly. CG [Hestenes and Stiefel, 1952, Boyd and Vandenberghe, 2004] avoids this, since it accesses M only through matrix-vector products, each costing one application of $\mathcal { A }$ and one of its adjoint $\mathcal { A } ^ { \top }$ . We warm-start CG from the previous iterate $\hat { \mathbf { x } } ^ { k }$ and stop once the relative residual satisfies

$$
\frac { \| \mathbf { r } ^ { ( j ) } \| _ { 2 } } { \| \mathbf { b } \| _ { 2 } + \delta } < \mathbf { c g \_ t o l } .\tag{16}
$$

Here, $\mathbf { r } ^ { ( j ) } = \mathbf { b } - \mathcal { M } \mathbf { x } ^ { ( j ) }$ is the residual after j CG iterations, b is the right-hand side of equation $^ { 1 3 , }$ and the small constant $\delta > 0$ prevents division by zero. If the tolerance is not reached, CG stops after at most max\_cgiter iterations. Throughout our experiments, we set max\_cgiter = 10 and $\mathsf { c g } _ { - } \mathsf { t o l } = 1 0 ^ { - 5 }$ . The base penalty $\rho$ and the schedule exponent $\beta ,$ in contrast, are task-dependent (table 4), and each CG solve at iteration k uses the resulting scheduled value $\rho _ { k }$

## A.3 Nonlinear Forward Operators: Gradient Descent with Backtracking

For a nonlinear A, the subproblem equation 11 no longer has a closed-form minimizer. We therefore approximate the minimizer by a fixed number of gradient-descent iterations on

$$
f ( { \bf x } ) = \frac { 1 } { 2 \sigma ^ { 2 } } \| \boldsymbol { A } ( { \bf x } ) - { \bf y } \| _ { 2 } ^ { 2 } + \frac { \rho _ { k } } { 2 } \| { \bf x } - \tilde { \bf x } ^ { k } \| _ { 2 } ^ { 2 } ,\tag{17}
$$

whose gradient is

$$
\nabla f ( { \mathbf x } ) = \frac { 1 } { \sigma ^ { 2 } } \mathrm { v j p } _ { A } \big ( { \mathbf x } , \boldsymbol { \mathcal { A } } ( { \mathbf x } ) - { \mathbf y } \big ) + \rho _ { k } \big ( { \mathbf x } - \tilde { { \mathbf x } } ^ { k } \big ) .\tag{18}
$$

The term $\mathrm { v j p } _ { \mathcal { A } } ( \mathbf { x } , \mathbf { v } ) = \big ( \partial \mathcal { A } ( \mathbf { x } ) / \partial \mathbf { x } \big ) ^ { \top } \mathbf { v }$ denotes the vector-Jacobian product of A at x applied to the residual v, and is computed by automatic differentiation. It takes the place of the explicit adjoint $A ^ { \top }$ used in the linear case, without ever forming the full Jacobian.

A fixed step size is not reliable across all $( \rho _ { k } , \sigma )$ combinations: the curvature of the data-fidelity term in equation 17 scales as $1 / \sigma ^ { 2 }$ and, depending on the task, may exceed the curvature $\rho _ { k }$ of the proximal term. We therefore choose the step size by backtracking line search under the Armijo sufficient-decrease condition [Armijo, 1966]. Starting from $\tau = 1 / \rho _ { k }$ , each gradient step proposes

$$
\begin{array} { r } { { \mathbf { x } } _ { \mathrm { t r y } } = { \mathbf { x } } - \tau \nabla f ( { \mathbf { x } } ) , } \end{array}\tag{19}
$$

which is accepted as soon as

$$
\begin{array} { r } { f ( \mathbf { x } _ { \mathrm { t r y } } ) \leq f ( \mathbf { x } ) - \frac { 1 } { 2 } \tau \| \nabla f ( \mathbf { x } ) \| _ { 2 } ^ { 2 } ; } \end{array}\tag{20}
$$

otherwise, $\tau$ is halved, for at most 20 halvings. The next of the admm\_gd\_steps gradient steps is warm-started from the accepted step size, doubled and capped at $1 / \rho _ { k }$ , which lets later steps recover toward the initial upper bound after an overly conservative early step. We use admm\_gd\_steps = 10 for HDR reconstruction and phase retrieval and admm\_gd\_steps = 5 for nonlinear deblurring, identically on both datasets.

## B Experimental Details

The overall experimental setup is described in section 4. In this section, we provide additional details on the implementation of the forward operators used in our experiments, covering linear, nonlinear, and segmentation-based operators. We also describe how PnP-Flow and FlowDPS are adapted to the nonlinear forward models considered in our evaluation. Finally, we report the implementation detail and hyperparameter settings for PnP-CF and all baseline methods to facilitate reproducibility.

## B.1 Forward Operators with Gaussian Noise

For box and random inpainting, A is implemented as element-wise multiplication with the corresponding binary mask. For super-resolution, A is implemented as spatial downsampling, while for nonlinear deblurring, the forward operator is implemented using a frozen pretrained KernelWizard network with the GOPRO-wVAE checkpoint. Since KernelWizard operates on images in [0, 1], inputs are mapped from the normalized range [−1, 1] to [0, 1] before the forward pass and mapped back afterward. The network parameters remain fixed during reconstruction, while gradients are propagated only with respect to the input image. The blur condition is represented by a fixed latent kernel code shared across all evaluated images. For phase retrieval, the input image is symmetrically zero-padded before applying an orthonormal two-dimensional Fourier transform, after which only the magnitude of the complex Fourier coefficients is retained. Since the phase information is discarded, the resulting mapping is nonlinear and highly non-injective. For HDR reconstruction, the forward operator first scales the image intensity and subsequently clips the result to the valid normalized range [−1, 1], thereby simulating saturation and loss of information in over-exposed regions.

## B.2 Segmentation-guided Reconstruction

We evaluate segmentation-guided reconstruction on 100 test images from CelebA-HQ [Karras et al., 2018], resized to 128 × 128, using $C = 1 9$ classes. We use a frozen U-Net-based [Ronneberger et al., 2015] semantic segmentation network trained on CelebAMask-HQ [Lee et al., 2020] as the nonlinear forward operator, and a flow-matching prior originally trained on CelebA and fine-tuned on CelebA-HQ. The fine-tuning set consists of 30k CelebA-HQ samples, split into 28.5k images used for fine-tuning and 1.5k held out for testing; the 100 evaluation images are drawn from this held-out test split. Unlike the other inverse problems, the observation is categorical and there is no Gaussian measurement noise. For each reference image $\mathbf { x } ^ { \star }$ , we generate the target segmentation mask by applying the forward operator and taking the most likely class at every pixel $\mathbf { y } _ { i } = \arg \operatorname* { m a x } _ { c } \mathcal { A } ( \mathbf { x } ^ { \star } ) _ { i , c }$ where i indexes pixels.

Table 4: Hyperparameter settings per task and dataset for the time-adaptive PnP-CF variant. $K \colon$ number of PnP-CF iterations (steps\_pnp); $\rho \colon$ base ADMM penalty (rho\_admm); $\beta \colon$ penaltyschedule exponent (beta\_rho); $\gamma \colon$ time-schedule exponent (t\_schedule\_gamma); Denoiser calls: number of denoiser samples averaged per iteration (num\_denoiser\_calls); GD steps: nonlinear x-update gradient steps (admm\_gd\_steps, linear tasks use CG instead).
<table><tr><td>Dataset</td><td>Task</td><td>K</td><td> $\rho$ </td><td> $\beta$ </td><td>γ</td><td>Denoiser calls</td><td>GD steps</td></tr><tr><td>CelebA</td><td>Box inpainting</td><td>100</td><td>50</td><td>2.5</td><td>1.0</td><td>5</td><td></td></tr><tr><td>CelebA</td><td>Random inpainting</td><td>100</td><td>100</td><td>2.5</td><td>1.0</td><td>5</td><td>一</td></tr><tr><td>CelebA</td><td>Super-resolution</td><td>100</td><td>50</td><td>1.3</td><td>1.0</td><td>5</td><td></td></tr><tr><td>CelebA</td><td>Phase retrieval</td><td>100</td><td>1500</td><td>1.8</td><td>1.5</td><td>5</td><td>10</td></tr><tr><td>CelebA</td><td>HDR reconstruction</td><td>100</td><td>100</td><td>1.5</td><td>0.2</td><td>5</td><td>10</td></tr><tr><td>CelebA</td><td>Nonlinear deblurring</td><td>200</td><td>100</td><td>1.5</td><td>1.5</td><td>1</td><td>5</td></tr><tr><td>AFHQ-Cat</td><td>Box inpainting</td><td>100</td><td>50</td><td>2.5</td><td>1.5</td><td>5</td><td>一</td></tr><tr><td>AFHQ-Cat</td><td>Random inpainting</td><td>100</td><td>100</td><td>2.5</td><td>1.0</td><td>5</td><td>一</td></tr><tr><td>AFHQ-Cat</td><td>Super-resolution</td><td>500</td><td>50</td><td>1.3</td><td>1.0</td><td>5</td><td></td></tr><tr><td>AFHQ-Cat</td><td>Phase retrieval</td><td>300</td><td>1500</td><td>2.2</td><td>1.9</td><td>5</td><td>10</td></tr><tr><td>AFHQ-Cat</td><td>HDR reconstruction</td><td>100</td><td>500</td><td>1.5</td><td>1.0</td><td>5</td><td>10</td></tr><tr><td>AFHQ-Cat</td><td>Nonlinear deblurring</td><td>250</td><td>100</td><td>1.5</td><td>1.5</td><td>1</td><td>5</td></tr><tr><td>CelebA-HQ</td><td>Segmentation guided</td><td>500</td><td>500</td><td>0</td><td>1.0</td><td>1</td><td>10</td></tr></table>

Directly enforcing the one-hot segmentation mask throughout the trajectory can over-constrain early iterates, which may still be far from a plausible image. We therefore construct a softened target distribution by assigning a small amount of probability mass to a predefined set of semantically neighboring classes. Let ${ \bf y } _ { i , c }$ denote the one-hot encoding of the observed class at pixel $i ,$ and let $q ( c \mid \mathbf { y } _ { i } )$ denote the uniform distribution over the predefined neighbors of class $\mathbf { y } _ { i } . \mathrm { A t }$ iteration $k ,$ the target distribution is

$$
\tilde { \mathbf { y } } _ { i , c } ^ { k } = ( 1 - \alpha _ { k } ) \mathbf { y } _ { i , c } + \alpha _ { k } q ( c \mid \mathbf { y } _ { i } ) , \qquad \alpha _ { k } = 0 . 2 \left( 1 - \frac { k } { K - 1 } \right) ^ { 3 } .\tag{21}
$$

Thus, the target is moderately softened $( \alpha _ { 0 } = 0 . 2 )$ at the beginning of the trajectory and progressively approaches the original one-hot mask, with $\alpha _ { k }  0$ at the final iteration.

Using the same equation as in section 4.3, the segmentation forward-consistency term is given as $\begin{array} { r } { l } { _ { \mathrm { s e g } } ( \tilde { \mathbf { y } } ^ { k } ; \mathbf { x } ) = - \sum _ { i } \sum _ { c = 1 } ^ { C } \tilde { \mathbf { y } } _ { i , c } ^ { k } } \end{array}$ <sub>c</sub> log $A ( \mathbf { x } ) _ { i , c }$ . The corresponding x-update approximately minimizes $\begin{array} { r } { \lambda _ { \mathrm { C E } } l _ { \mathrm { s e g } } ( \tilde { \mathbf { y } } ^ { k } ; \mathbf { x } ) + \frac { \rho _ { k } } { 2 } \| \mathbf { x } - ( \mathbf { z } ^ { k } - \mathbf { u } ^ { k } ) \| _ { 2 } ^ { 2 } } \end{array}$ . Here, λ controls the relative contribution of the crossentropy forward-consistency term with respect to the ADMM quadratic penalty, and is set to $\lambda _ { \mathrm { C E } } = 1 0$ for all segmentation experiments. The remaining optimization and schedule hyperparameters follow the same formulation as for the other inverse problems, with the task-specific values reported in table 4.

## B.3 Hyperparameters

The main hyperparameter in our method is the penalty parameter $\rho ,$ and to further improve reconstruction performance, we consider the schedules $\rho _ { k } \stackrel { \cdot \cdot } { = } \rho / ( 1 - t _ { k } ^ { \cdot } ) ^ { \beta }$ and $t _ { k } = ( k / \bar { K } ) ^ { \gamma }$ , where $\beta$ controls the penalty growth as $t _ { k } \to 1$ , while $\gamma$ controls the progression along the flow trajectory.

For nonlinear deblurring, we observed a small improvement in reconstruction quality from an optional early-stopping at 85% of the configured iteration budget $K$ , corresponding to $\dot { k } = 0 . 8 5 K$ . This choice is not critical to the method and yields only a minor gain; all other tasks run for the full K iterations.

For phase retrieval, two random restarts are used to reduce sensitivity to initialization, following the restart strategy commonly adopted for this task in DPS-based inverse-problem settings [Chung et al., 2023]. This restart procedure is applied consistently to all evaluated methods, and for each image the reconstruction with the lowest final measurement residual $\| { \mathcal { A } } ( \mathbf { x } ) - \mathbf { y } \| _ { 2 }$ is retained.

Table 4 reports the full hyperparameter configuration used for each task and dataset, corresponding to the time-adaptive variant of PnP-CF reported in table 1. All linear tasks share max\_cgiter $= 1 0$ and cg\_tol $\mathbf { \bar { \tau } } = 1 0 ^ { - 5 } \left( \mathbf { A } . 2 \right)$ ; GD steps apply only to the nonlinear tasks $( \mathsf { A } . 3 )$

## B.3.1 Role of $\gamma$ and $\beta$

The schedule parameters $\gamma$ and $\beta$ control two distinct aspects of the algorithm and are therefore treated as separate hyperparameters. The parameter γ determines how the iteration index is mapped to the flow time, $\begin{array} { r } { t _ { k } = \left( \frac { k } { K } \right) ^ { \gamma } } \end{array}$ , and thus controls the rate at which the iterates progress along the learned flow trajectory: $\gamma = 1$ gives a linear progression, $\gamma > 1$ keeps the iterates at earlier flow times longer before accelerating near the end, and $\gamma < 1$ advances more rapidly toward later flow times — with $\gamma < 1$ correspondingly favoring the deterministic component $\epsilon _ { d }$ of the pullback step earlier in the trajectory, and $\gamma > 1$ retaining the stochastic component $\epsilon _ { s }$ for longer.

In contrast, $\beta$ acts only after $t _ { k }$ has been determined, through the ADMM penalty schedule

$$
\rho _ { k } = \frac { \rho } { ( 1 - t _ { k } ) ^ { \beta } } = \frac { \rho } { \left( 1 - ( k / K ) ^ { \gamma } \right) ^ { \beta } } ,
$$

and controls how aggressively the consensus penalty grows as the algorithm approaches the end of the trajectory.

The distinction between these roles can be seen from the terminal growth rate and the onset of penalty growth. $\mathbf { A s } k  K$ , writing $\varepsilon = 1 - k / K \to 0 ^ { + }$ , we have $t _ { k } = \bar { ( } 1 - \varepsilon ) ^ { \gamma } \approx 1 - \gamma \varepsilon$ and hence

$$
\rho _ { k } \approx \rho \gamma ^ { - \beta } \varepsilon ^ { - \beta } , k  K ,
$$

$\mathbf { s o } ,$ for the unclipped schedule, the asymptotic order of the terminal penalty growth is determined by $\beta ,$ , while $\gamma$ enters only through the multiplicative factor $\gamma ^ { - \beta }$ near $k = K$ . Meanwhile, for fixed $\beta ,$ the onset of penalty growth, defined here as the iteration fraction $s ^ { * } = k ^ { * } / K$ at which $\rho _ { k ^ { * } }$ has doubled relative to $\rho ,$ is

$$
s ^ { * } = \left( 1 - 2 ^ { - 1 / \beta } \right) ^ { 1 / \gamma } ,
$$

which is governed by $\gamma$ for any fixed $\beta .$ . Consequently, a single schedule parameter cannot independently control both properties: varying β changes the terminal growth rate but also shifts its onset, whereas varying $\gamma$ can shift the onset without changing the asymptotic growth order. Adjusting both parameters therefore provides separate degrees of freedom for controlling the timing and strength of the penalty growth. For example, a penalty that remains moderate for most iterations but grows aggressively near the end can be obtained by jointly controlling the delayed onset through $\gamma$ and the terminal growth rate through $\beta .$

Beyond this functional distinction, the two parameters are also mathematically independent. Suppose two parameter pairs $( \beta _ { 1 } , \gamma _ { 1 } )$ and $( \beta _ { 2 } , \gamma _ { 2 } )$ induced the same penalty schedule at every iteration, i.e.

$$
( 1 - s ^ { \gamma _ { 1 } } ) ^ { - \beta _ { 1 } } = ( 1 - s ^ { \gamma _ { 2 } } ) ^ { - \beta _ { 2 } } \qquad \forall s \in ( 0 , 1 ) .
$$

Proposition 1. If the above identity holds for all $s \in ( 0 , 1 )$ with $\beta _ { 1 } , \beta _ { 2 } , \gamma _ { 1 } , \gamma _ { 2 } > 0 ,$ , then $\beta _ { 1 } = \beta _ { 2 }$ and $\gamma _ { 1 } = \gamma _ { 2 }$

Proof. Taking logarithms, the identity is equivalent to

$$
{ \frac { \beta _ { 1 } } { \beta _ { 2 } } } = { \frac { \ln ( 1 - s ^ { \gamma _ { 2 } } ) } { \ln ( 1 - s ^ { \gamma _ { 1 } } ) } } = : R ( s ) , \qquad \forall s \in ( 0 , 1 ) .
$$

Since the left-hand side is a fixed constant, $R ( s )$ must be constant on $( 0 , 1 )$ . Using ln $( 1 - x ) =$ $- x + O ( x ^ { 2 } )$ as $x \to 0 ^ { + }$

$$
R ( s ) = \frac { s ^ { \gamma _ { 2 } } + O \bigl ( s ^ { 2 \gamma _ { 2 } } \bigr ) } { s ^ { \gamma _ { 1 } } + O ( s ^ { 2 \gamma _ { 1 } } ) } = s ^ { \gamma _ { 2 } - \gamma _ { 1 } } \bigl ( 1 + o ( 1 ) \bigr ) , \qquad s \to 0 ^ { + } .
$$

$\operatorname { I f } \gamma _ { 1 } \neq \gamma _ { 2 }$ , this expression tends to either 0 or ∞ as $s \to 0 ^ { + }$ , contradicting that R is constant. Hence $\gamma _ { 1 } = \gamma _ { 2 } = : \gamma$ . Substituting back, the original identity reduces to $\beta _ { 1 } \ln ( 1 - s ^ { \gamma } ) = \beta _ { 2 } \ln ( 1 - s ^ { \gamma } )$ for all $s \in ( 0 , 1 )$ ; since l $\mathrm { n } ( 1 - s ^ { \gamma } ) \neq 0$ on (0, 1), this forces $\beta _ { 1 } = \beta _ { 2 }$ □

Hence no reparametrization of $\gamma$ can absorb a change in $\beta ,$ or vice versa: the two parameters are distinct, non-redundant degrees of freedom. The parameter γ governs the temporal progression along the flow trajectory and influences the onset of penalty growth, whereas $\beta$ determines the strength and asymptotic growth rate of the ADMM consensus penalty.

## B.4 Adapting baselines to Nonlinear Operators

PnP-Flow [Martin et al., 2025] and FlowDPS [Kim et al., 2025] were originally evaluated only on linear inverse problems. For PnP-Flow, the linear forward-consistency gradient involves the adjoint $\mathcal { A } ^ { \top }$ . We extend this update to nonlinear operators by replacing the adjoint-based gradient with the corresponding vector-Jacobian product (VJP), $\begin{array} { r } { \mathrm { v j p } _ { \mathcal { A } } ( \mathbf { x } , \mathbf { v } ) = \left( \frac { \partial \mathcal { A } ( \mathbf { x } ) } { \partial \mathbf { x } } \right) ^ { \top } } \end{array}$ v, computed via automatic differentiation. For linear ${ \mathcal { A } } ,$ the VJP reduces exactly to $\mathcal { A } ^ { \top } { \mathbf v } ;$ for nonlinear ${ \mathcal { A } } ,$ it provides the required forward-consistency gradient without requiring a closed-form adjoint. This is the same mechanism used in PnP-CF’s nonlinear forward-consistency step (A.3).

For FlowDPS, we follow the practical strategy of the authors and perform a fixed number of gradientdescent updates on the forward-consistency objective. For nonlinear operators, we compute the gradient of $\| \mathbf { y } - \mathcal { A } ( \mathbf { z } ) \| _ { 2 } ^ { 2 }$ using the VJP of A. This preserves the key FlowDPS approximation that avoids differentiating through the flow-matching denoiser network.

D-Flow [Ben-Hamu et al., 2024] requires no structural adaptation to handle nonlinear operators: it optimizes the initial noise directly by backpropagating through the full generation trajectory and the forward operator via standard automatic differentiation, which places no restriction on the linearity of A. However, D-Flow’s own published experiments on image inverse problems cover only linear degradations (inpainting, super-resolution, Gaussian deblurring); its reported validation on nonlinear problems is limited to non-image domains (audio, molecule generation). We therefore run D-Flow on our nonlinear image tasks using its unmodified optimization procedure.

Tables 5 and 6 report the full per-task hyperparameter settings used for all baselines on CelebA and AFHQ-Cat, respectively. For the nonlinear tasks, we tuned PnP-Flow, FlowDPS, and D-Flow to obtain the best reconstruction quality achievable with each method, while keeping their iteration budgets and computational cost comparable to PnP-CF whenever possible.

Table 5: Hyperparameters used for all baseline methods on the CelebA dataset.
<table><tr><td></td><td>Box Inpaint.</td><td>Super-res.</td><td>Rand. Inpaint.</td><td>HDR Recon.</td><td>Phase Retr.</td><td>Nonl. Deblur.</td></tr><tr><td>OT-ODE</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>t0 (start time)</td><td>0.1</td><td>0.1</td><td>0.1</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>γ schedule</td><td>γt</td><td>constant</td><td>constant</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>K (steps)</td><td>100</td><td>100</td><td>100</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>Flow Priors</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>λ (regularization)</td><td>10,000</td><td>10,000</td><td>10,000</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>η (learning rate)</td><td>0.01</td><td>0.1</td><td>0.01</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>D-Flow</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>λ (regularization)</td><td>0.001</td><td>0.001</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>α (blending)</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>niter (max. iter.)</td><td>9</td><td>10</td><td>20</td><td>10</td><td>10</td><td>10</td></tr><tr><td>FlowDPS</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Grad. step size</td><td>2.0</td><td>2.5</td><td>3.0</td><td>0.50</td><td>2.0</td><td>0.5</td></tr><tr><td>Grad. steps</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>K (steps)</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>200</td></tr><tr><td>FLOWER</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Denoiser calls</td><td>5</td><td>5</td><td>5</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>K (steps)</td><td>100</td><td>100</td><td>100</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>PnP-Flow</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>α (learning rate factor)</td><td>0.5</td><td>0.3</td><td>0.01</td><td>0.3</td><td>0.01</td><td>0.5</td></tr><tr><td>Denoiser calls</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>K (steps)</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>200</td></tr></table>

Table 6: Hyperparameters used for all baseline methods on the AFHQ-Cat dataset.
<table><tr><td></td><td>Box Inpaint.</td><td>Super-res.</td><td>Rand. Inpaint.</td><td>HDR Recon.</td><td>Phase Retr.</td><td>Nonl. Deblur.</td></tr><tr><td colspan="7">OT-ODE</td></tr><tr><td>t0 (start time)</td><td>0.1</td><td>0.1</td><td>0.1</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>γ schedule</td><td>γt</td><td>constant</td><td>constant</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>K (steps)</td><td>100</td><td>500</td><td>100</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td colspan="7">Flow Priors</td></tr><tr><td>λ (regularization)</td><td>10,000</td><td>10,000</td><td>10,000</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>η (learning rate)</td><td>0.01</td><td>0.1</td><td>0.01</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td colspan="7">D-Flow</td></tr><tr><td>λ (regularization)</td><td>0.01</td><td>0.001</td><td>0.001</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>α (blending)</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>niter (max. iter.)</td><td>9</td><td>20</td><td>20</td><td>10</td><td>10</td><td>10</td></tr><tr><td colspan="7">FlowDPS</td></tr><tr><td>Grad. step size</td><td>2.0</td><td>2.5</td><td>3.0</td><td>0.5</td><td>0.5</td><td>0.10</td></tr><tr><td>Grad. steps</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>K (steps)</td><td>100</td><td>500</td><td>100</td><td>100</td><td>300</td><td>250</td></tr><tr><td colspan="7">FLOWER</td></tr><tr><td>Denoiser calls</td><td>5</td><td>5</td><td>5</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td>K (steps)</td><td>100</td><td>500</td><td>100</td><td>N/A</td><td>N/A</td><td>N/A</td></tr><tr><td colspan="7">PnP-Flow</td></tr><tr><td>α (learning rate factor)</td><td>0.5</td><td>0.01</td><td>0.01</td><td>0.3</td><td>0.01</td><td>0.8</td></tr><tr><td>Denoiser calls</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>K (steps)</td><td>100</td><td>500</td><td>100</td><td>100</td><td>300</td><td>250</td></tr></table>

## C Computational Complexity and Efficiency

In this section, we analyze the computational efficiency and memory footprint of all methods. Experiments are conducted on a single NVIDIA A100-SXM4-80GB GPU (CUDA 13.0), using a batch size of 4 images and averaged over 25 batches keeping the same hyperparameter settings as used in the main results (1). Table 7 reports per-batch runtime and peak GPU memory usage, averaged separately over the linear tasks (box inpainting, random inpainting, super-resolution) and the nonlinear tasks (HDR reconstruction, phase retrieval, nonlinear deblurring).

Table 7: Average per-batch runtime (in seconds) and peak GPU memory (in GB) on CelebA and AFHQ-Cat, grouped by linear tasks (box inpainting, random inpainting, super-resolution) and nonlinear tasks (HDR reconstruction, phase retrieval, nonlinear deblurring), each averaged over the three tasks in its group (batch size of 4 images, averaged over 25 batches). Best scores are bold.
<table><tr><td rowspan="2">Method</td><td colspan="2">Linear</td><td colspan="2">Nonlinear</td></tr><tr><td></td><td>Time (s) Memory (GB)</td><td></td><td>Time (s) Memory (GB)</td></tr><tr><td>PnP-CF (ours)</td><td>16.81</td><td>0.32</td><td>32.62</td><td>0.46</td></tr><tr><td>PnP-Flow</td><td>17.31</td><td>0.32</td><td>27.56</td><td>0.46</td></tr><tr><td>OT-ODE</td><td>17.49</td><td>2.34</td><td></td><td></td></tr><tr><td>D-Flow</td><td>87.77</td><td>13.36</td><td>92.19</td><td>17.84</td></tr><tr><td>Flow Priors</td><td>35.66</td><td>8.81</td><td></td><td></td></tr><tr><td>FlowDPS</td><td>10.83</td><td>0.34</td><td>20.97</td><td>0.47</td></tr><tr><td>FLOWER</td><td>19.13</td><td>0.33</td><td></td><td></td></tr><tr><td colspan="5">CelebA</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="2">Linear</td><td colspan="2">Nonlinear</td></tr><tr><td></td><td>Time (s) Memory (GB)</td><td></td><td>Time (s) Memory (GB)</td></tr><tr><td>PnP-CF (ours)</td><td>50.52</td><td>0.85</td><td>69.11</td><td>0.99</td></tr><tr><td>PnP-Flow</td><td>46.17</td><td>0.85</td><td>63.93</td><td>0.99</td></tr><tr><td>OT-ODE</td><td>44.96</td><td>8.55</td><td></td><td></td></tr><tr><td>D-Flow</td><td>141.75</td><td>50.47</td><td>113.08</td><td>50.80</td></tr><tr><td>Flow Priors</td><td>75.97</td><td>33.30</td><td></td><td></td></tr><tr><td>FlowDPS</td><td>35.01</td><td>0.87</td><td>47.61</td><td>1.01</td></tr><tr><td>FLOWER</td><td>59.71</td><td>0.86</td><td>一</td><td>一</td></tr><tr><td colspan="5"></td></tr></table>

D-Flow and Flow Priors incur significant computational and memory overhead, limiting practical efficiency. D-Flow is expensive because it repeatedly solves the full ODE and backpropagates through the entire trajectory to optimize the initial noise. This leads to high memory usage from storing intermediate activations and long runtime due to repeated full-trajectory differentiation. Flow Priors is computationally expensive, as each ODE step involves multiple gradient updates, requiring repeated forward and backward passes through the model along with costly Jacobian trace evaluations. This nested optimization leads to high overall runtime. OT-ODE avoids full trajectory backpropagation by applying a forward-consistency correction at each ODE step through a vector-Jacobian product (VJP) with respect to the denoiser network; because this network is large, computing the VJP still requires storing its intermediate activations at every step, incurring notable memory overhead despite avoiding full-trajectory differentiation. Neither OT-ODE, Flow Priors, nor FLOWER report nonlinear extensions, so their nonlinear-task averages are omitted.

In contrast, PnP-Flow, FlowDPS, FLOWER, and PnP-CF generally incur lower computational cost and substantially lower memory usage than D-Flow and Flow Priors on the evaluated tasks. In each case, the forward-consistency step avoids differentiating through the denoiser network. Among the evaluated methods, FlowDPS achieves the lowest average runtime across both linear and nonlinear tasks. PnP-CF solves the linear forward-consistency subproblem iteratively using conjugate gradients (appendix A.2), and uses a bounded number of gradient-descent steps with backtracking and VJPs for nonlinear operators (appendix A.3), without backpropagating through the denoiser in either case. These more accurate forward-consistency updates introduce additional runtime relative to FlowDPS, but keep PnP-CF’s memory usage low and comparable to PnP-Flow and FlowDPS across both linear and nonlinear tasks, and to FLOWER on the linear tasks (table 7).

## D Ablation Studies

In this section, we further study the impact of the main design choices in PnP-CF through additional ablation experiments. In particular, we analyze the role of the pullback construction and the dualvariable update, which together control how the intermediate iterates balance exploration of the flow prior with consistency to the current reconstruction trajectory. These experiments complement the analysis presented in section 4.1 and provide additional evidence for the contribution of each component.

First, we evaluate PnP-CF with only stochastic component $( \epsilon _ { s } )$ . Second, we study PnP-CF with only deterministic component $( \epsilon _ { d } )$ . Finally, we consider our full approach, PnP-CF with time-adaptive pullback strategy, which combines both stochastic and deterministic components in an adaptive manner within the consensus framework.

Table 8 reports the performance across different variants of our method. Complementing these results, figure 7 provides a qualitative comparison on CelebA and AFHQ-Cat across the six inverse problems considered in our experiments.

Figure 6 presents the PSNR, SSIM and LPIPS curves for different inverse tasks, comparing the settings with and without dual variable across optimization iterations.

![](images/04a3b8b7d8666e685752c0d16b53bf19538725a6046f2a55895202ea87b79c04.jpg)

![](images/9cc32f90fd74e42591bd1cea4a1b171ce41037b8b832e374cb6d051cd3ac4fe9.jpg)

![](images/8b010180b2566cf4a144eb9051509c6fb79deb94049ad9b55d2564dde97d3aa5.jpg)

![](images/648338ae9150e13b29568ea5e863cd0c80dbb68cd8d730fa355e833af1fca4c1.jpg)  
Figure 6: Effect of the dual-variable update on reconstruction quality across optimization iterations. PSNR (top), SSIM (middle), and LPIPS (bottom) are shown for super-resolution, random inpainting, phase retrieval, and HDR reconstruction on CelebA, comparing PnP-CF ‘with’ and ‘without’ the dual-variable update. Across tasks, the dual-variable formulation generally improves convergence speed and final reconstruction quality, with particularly large gains for super-resolution, random inpainting, and HDR reconstruction.

Table 8: Task-wise performance of the three PnP-CF pullback variants (time-adaptive, stochastic, deterministic) across all six inverse tasks. Best result per task and metric, among the three variants, is shown in bold.
<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="3">CelebA</td><td colspan="3">AFHQ-Cat</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td><td> $\mathrm { L P I P S } \downarrow$ </td><td>SSIM↑</td></tr><tr><td rowspan="3">HDR Reconstruction</td><td>PnP-CF (adaptive)</td><td>27.59</td><td>0.04</td><td>0.85</td><td>26.29</td><td>0.08</td><td>0.86</td></tr><tr><td>PnP-CF (stochastic)</td><td>22.44</td><td>0.10</td><td>0.80</td><td>22.97</td><td>0.32</td><td>0.71</td></tr><tr><td>PnP-CF (deterministic)</td><td>18.12</td><td>0.36</td><td>0.43</td><td>24.11</td><td>0.13</td><td>0.75</td></tr><tr><td rowspan="3">Box Inpainting</td><td>PnP-CF (adaptive)</td><td>26.36</td><td>0.05</td><td>0.89</td><td>22.65</td><td>0.13</td><td>0.82</td></tr><tr><td>PnP-CF (stochastic)</td><td>22.79</td><td>0.18</td><td>0.74</td><td>21.24</td><td>0.37</td><td>0.68</td></tr><tr><td>PnP-CF (deterministic)</td><td>22.50</td><td>0.10</td><td>0.72</td><td>20.65</td><td>0.21</td><td>0.66</td></tr><tr><td rowspan="3">Nonlinear Deblurring</td><td>PnP-CF (adaptive)</td><td>27.09</td><td>0.09</td><td>0.79</td><td>22.28</td><td>0.29</td><td>0.58</td></tr><tr><td>PnP-CF (stochastic)</td><td>25.71</td><td>0.15</td><td>0.77</td><td>22.27</td><td>0.32</td><td>0.55</td></tr><tr><td>PnP-CF (deterministic)</td><td>13.46</td><td>0.68</td><td>0.24</td><td>20.01</td><td>0.58</td><td>0.35</td></tr><tr><td rowspan="3">Phase Retrieval</td><td>PnP-CF (adaptive)</td><td>30.33</td><td>0.11</td><td>0.80</td><td>25.27</td><td>0.22</td><td>0.70</td></tr><tr><td>PnP-CF (stochastic)</td><td>23.72</td><td>0.21</td><td>0.70</td><td>20.22</td><td>0.47</td><td>0.57</td></tr><tr><td>PnP-CF (deterministic)</td><td>23.67</td><td>0.26</td><td>0.61</td><td>15.82</td><td>0.56</td><td>0.34</td></tr><tr><td rowspan="3">Random Inpainting</td><td>PnP-CF (adaptive)</td><td>29.93</td><td>0.04</td><td>0.89</td><td>30.72</td><td>0.07</td><td>0.85</td></tr><tr><td>PnP-CF (stochastic)</td><td>27.50</td><td>0.12</td><td>0.83</td><td>28.25</td><td>0.29</td><td>0.78</td></tr><tr><td>PnP-CF (deterministic)</td><td>21.93</td><td>0.25</td><td>0.59</td><td>22.01</td><td>0.43</td><td>0.48</td></tr><tr><td rowspan="3">Super Resolution</td><td>PnP-CF (adaptive)</td><td>26.45</td><td>0.08</td><td>0.79</td><td>23.72</td><td>0.27</td><td>0.66</td></tr><tr><td>PnP-CF (stochastic)</td><td>23.38</td><td>0.19</td><td>0.72</td><td>19.70</td><td>0.48</td><td>0.57</td></tr><tr><td>PnP-CF (deterministic)</td><td>18.17</td><td>0.34</td><td>0.42</td><td>14.05</td><td>0.68</td><td>0.27</td></tr></table>

## E Additional Experiments

## E.1 Segmentation guidance failure cases

Although PnP-CF performs well overall for segmentation-guided reconstruction, as shown in the results in sections 4.3 and appendix H, we observe occasional failure cases that reveal an important limitation of using a learned segmentation network as the forward operator.

The reconstruction is optimized to reproduce the target segmentation under a fixed segmentator. Consequently, the optimization may exploit imperfections or ambiguities in the learned forward model and produce image structures that induce the desired segmentation output without corresponding to a perceptually plausible face. In such cases, high segmentation consistency does not necessarily translate into high reconstruction quality. Figure 8 illustrates representative examples. Although the predicted segmentation boundaries closely follow the target facial regions, the reconstructed images can exhibit distorted facial geometry, unnatural local structures, or other perceptual artifacts. Interestingly, some of these artifacts appear aligned with semantic boundaries emphasized by the segmentator, suggesting that the reconstruction can satisfy the learned forward model while deviating from the natural image manifold.

These examples highlight a broader challenge when using highly non-injective learned neural networks as forward operators. Improving the robustness of the interaction between such learned forward operators and generative priors is therefore an important direction for future work.

## E.2 Experiments on default hyperparameter settings

Table 9 report results of PnP-CF run with the default (non-scheduled) hyperparameter settings: a constant penalty $\rho _ { k } = \rho ( \mathrm { i . e . , } \beta = 0 )$ and a linear time schedule $t _ { k } = k / K ( \mathrm { i . e . , } \gamma = 1 )$ , across all six inverse problems on both datasets. For CelebA, we use $K = 2 0 0$ iterations and $\rho = 1 0 0 0 \mathrm { : }$ ; for AFHQ-Cat, we use $K = 2 5 0$ iterations and $\rho = 5 0 0$ . Both datasets use $n = 5$ denoiser calls per iteration. All other settings are held fixed across the six tasks within a dataset rather than tuned per task. For the linear inverse problems, we use the more challenging degradations described in section 4, so the results are not directly comparable to those reported by Kumar and Gregorová [2026].

![](images/3c4c9104803b7deeede73e14437bb73eddf552058ba5d7de279381b48aa9f5b9.jpg)  
Figure 7: Additional qualitative ablation results comparing the pullback variants of PnP-CF on CelebA (left) and AFHQ-Cat (right). Within each dataset, columns from left to right show the clean image, degraded observation, stochastic-only (ϵ<sub>s</sub>), deterministic-only $( \epsilon _ { d } )$ , and time-adaptive $( \epsilon _ { s } , \epsilon _ { d } )$ variants. From top to bottom, the rows correspond to box inpainting, random inpainting, super-resolution, phase retrieval, nonlinear deblurring, and HDR reconstruction.

Table 9: PnP-CF results on reconstructed vs. noisy/degraded inputs with default (non-scheduled) hyperparameter settings on CelebA and AFHQ-Cat.
<table><tr><td rowspan="2">Dataset Task</td><td rowspan="2"></td><td colspan="2">PSNR↑</td><td colspan="2">SSIM↑</td><td colspan="2">LPIPS↓</td></tr><tr><td>Rec</td><td>Noisy</td><td>Rec</td><td>Noisy</td><td>Rec</td><td>Noisy</td></tr><tr><td rowspan="6">CelebA</td><td>Box Inpainting</td><td>24.71</td><td>18.91</td><td>0.76</td><td>0.66</td><td>0.08</td><td>0.31</td></tr><tr><td>Random Inpainting</td><td>30.15</td><td>11.46</td><td>0.90</td><td>0.18</td><td>0.04</td><td>1.09</td></tr><tr><td>Super Resolution</td><td>20.28</td><td>10.71</td><td>0.56</td><td>0.18</td><td>0.24</td><td>0.82</td></tr><tr><td>HDR Reconstruction</td><td>21.97</td><td>17.03</td><td>0.79</td><td>0.50</td><td>0.06</td><td>0.11</td></tr><tr><td>Nonlinear Deblurring</td><td>25.36</td><td>24.77</td><td>0.73</td><td>0.66</td><td>0.15</td><td>0.24</td></tr><tr><td>Phase Retrieval</td><td>27.35</td><td>9.12</td><td>0.68</td><td>0.21</td><td>0.20</td><td>1.24</td></tr><tr><td rowspan="6">AFHQ-Cat</td><td>Box Inpainting</td><td>21.48</td><td>18.31</td><td>0.69</td><td>0.64</td><td>0.20</td><td>0.35</td></tr><tr><td>Random Inpainting</td><td>30.61</td><td>12.41</td><td>0.85</td><td>0.20</td><td>0.07</td><td>1.13</td></tr><tr><td>Super Resolution</td><td>18.35</td><td>11.67</td><td>0.49</td><td>0.31</td><td>0.44</td><td>0.95</td></tr><tr><td>HDR Reconstruction</td><td>26.21</td><td>16.60</td><td>0.83</td><td>0.49</td><td>0.09</td><td>0.17</td></tr><tr><td>Nonlinear Deblurring</td><td>22.01</td><td>20.28</td><td>0.54</td><td>0.45</td><td>0.36</td><td>0.54</td></tr><tr><td>Phase Retrieval</td><td>15.06</td><td>9.70</td><td>0.37</td><td>0.31</td><td>0.52</td><td>1.36</td></tr></table>

## E.3 Experiment on baseline settings

For our three linear inverse problems, the settings used for the results in the main paper (section 4) are substantially more challenging than those considered in Martin et al. [2025]. We deliberately adopt these harder settings to more clearly distinguish the performance of different methods and demonstrate the robustness of our approach under more severe degradations.

We therefore also evaluate all methods under the milder settings used by Martin et al. [2025] and report the results in table 10. Specifically, we consider box inpainting with a centered $4 0 \times 4 0$ (CelebA) / 80 × 80 (AFHQ-Cat) mask, random inpainting with 70% of pixels masked, and superresolution with downsampling factors of $\times 2 \left( \mathrm { C e l e b A } \right) / \times 4 \left( \mathrm { A F H Q – C a t } \right)$ , while keeping the noise levels unchanged.

![](images/207fab117ba27721599d337fda1207463b725e03efd73ce8102132980b8a1788.jpg)  
Figure 8: Failure cases for segmentation-guided reconstruction. Left: reference images. Middle: PnP-CF reconstructions. Right: reconstructed images with the predicted segmentation boundaries overlaid in red. Despite achieving strong segmentation consistency, some reconstructions exhibit distorted facial geometry and perceptual artifacts. These examples illustrate that satisfying a learned segmentation forward model does not necessarily guarantee a visually plausible reconstruction.

The results show that our method remains competitive with the baselines on linear inverse problems under these milder degradation settings.

Table 10: Reconstruction quality on CelebA and AFHQ-Cat for linear inverse problems under the milder degradation settings used in Martin et al. [2025]. Best results are shown in bold.
<table><tr><td></td><td></td><td colspan="3">Box inpainting</td><td colspan="3">Super-resolution</td><td colspan="3">Random inpainting</td></tr><tr><td>Dataset</td><td>Method</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td rowspan="7">CelebA</td><td>PnP-CF (ours)</td><td>31.71</td><td>0.02</td><td>0.95</td><td>31.37</td><td>0.03</td><td>0.86</td><td>33.99</td><td>0.02</td><td>0.95</td></tr><tr><td>PnP-Flow</td><td>30.82</td><td>0.04</td><td>0.94</td><td>31.31</td><td>0.06</td><td>0.91</td><td>33.92</td><td>0.02</td><td>0.95</td></tr><tr><td>FLOWER</td><td>31.66</td><td>0.02</td><td>0.95</td><td>32.88</td><td>0.04</td><td>0.90</td><td>33.91</td><td>0.03</td><td>0.95</td></tr><tr><td>FlowDPS</td><td>28.48</td><td>0.06</td><td>0.89</td><td>25.94</td><td>0.12</td><td>0.79</td><td>26.42</td><td>0.11</td><td>0.80</td></tr><tr><td>D-Flow</td><td>28.28</td><td>0.06</td><td>0.85</td><td>29.40</td><td>0.04</td><td>0.83</td><td>31.96</td><td>0.03</td><td>0.91</td></tr><tr><td>Flow Priors</td><td>27.07</td><td>0.05</td><td>0.88</td><td>28.11</td><td>0.10</td><td>0.73</td><td>29.06</td><td>0.04</td><td>0.91</td></tr><tr><td>OT-ODE</td><td>29.49</td><td>0.04</td><td>0.92</td><td>31.14</td><td>0.03</td><td>0.93</td><td>28.68</td><td>0.05</td><td>0.87</td></tr><tr><td rowspan="7">AFHQ-Cat</td><td>PnP-CF (ours)</td><td>27.86</td><td>0.06</td><td>0.90</td><td>28.54</td><td>0.14</td><td>0.78</td><td>33.39</td><td>0.03</td><td>0.91</td></tr><tr><td>PnP-Flow</td><td>27.08</td><td>0.13</td><td>0.90</td><td>28.01</td><td>0.17</td><td>0.77</td><td>34.11</td><td>0.05</td><td>0.91</td></tr><tr><td>FLOWER</td><td>27.23</td><td>0.07</td><td>0.92</td><td>27.06</td><td>0.27</td><td>0.77</td><td>34.10</td><td>0.05</td><td>0.90</td></tr><tr><td>FlowDPS</td><td>24.19</td><td>0.19</td><td>0.83</td><td>26.09</td><td>0.28</td><td>0.74</td><td>27.43</td><td>0.25</td><td>0.76</td></tr><tr><td>D-Flow</td><td>26.35</td><td>0.14</td><td>0.77</td><td>25.14</td><td>0.16</td><td>0.62</td><td>31.38</td><td>0.09</td><td>0.85</td></tr><tr><td>Flow Priors</td><td>24.79</td><td>0.11</td><td>0.84</td><td>25.13</td><td>0.27</td><td>0.58</td><td>32.06</td><td>0.05</td><td>0.90</td></tr><tr><td>OT-ODE</td><td>24.47</td><td>0.10</td><td>0.87</td><td>26.09</td><td>0.12</td><td>0.71</td><td>30.05</td><td>0.09</td><td>0.85</td></tr></table>

![](images/e156fc62737cebd9699ced8c2ce1b8610541db5470bdde3fb58dda2d3bc92b74.jpg)  
Clean  
Degraded

## F Additional results on linear tasks with Gaussian noise

![](images/5e8cc155e21e6e919524c57a3515e4fb214c9c6c9409bcc0004d8597dc607eb3.jpg)

![](images/78b08cf0766083ae373e1590be360e6825388755739b08cdcdc865f8af0cfc7a.jpg)  
PSNR: 18.72

OT-ODE

![](images/8bde9852db8aa3056527ddaf27cdd752b7f25404dc82a9e829b9308d4e2c45d2.jpg)  
PSNR: 25.39

D-Flow

![](images/da13d616d17e0a8f96a4b2aac242200eea463958df768cc36f7d37f1cfc68eb4.jpg)  
Flow Prior  
PSNR: 26.01

Flow DPS

![](images/e2d774aecef45549a1e97e0137f6d3b53b1b6cb7d7629a42eb8321fcd5a2735d.jpg)

![](images/c517b9587f1c860070589dc659ff2765865144b2cb32656b35ca7218fa0c8eaf.jpg)

![](images/c58eb7234128be34558cf3e2479a4c5d7a7e40ff077285f4b9ab6047d1914f3a.jpg)

Flower

![](images/6a07da27adbc9e4b62f5fc7d808a1bc88fe2281a9e18b989227173aab32d3d0d.jpg)

![](images/8655cbf71c217c0393fea0ac3698402728f67275bdb84012bab4de495ac5ac67.jpg)

PNP-Flow

![](images/b5b78bbd83348e22001c0f0ee1568479ac723d364fe8cc535bb10653955af6ed.jpg)  
PSNR: 22.43  
PNP-CF (ours)

![](images/41256a94bac4c25b71e09dc62c818ac896d8ea4970f5582080e71631c7ba14d8.jpg)  
PSNR: 10.82  
PSNR: 23.50  
PSNR: 25.33

![](images/8050e670689e3ff74013b1c557d9c353491af50cd3b43d72cb989e5adc9b35fd.jpg)  
PSNR: 26.08  
PSNR: 22.92

![](images/ab4737868ef8abdd4a65377712c39de933163fc2de97156f86d772846f7ad6ff.jpg)  
PSNR: 11.60

![](images/5eff7044e93b4cb3565b0bc20f81424ef0216b9bb2bd3ea816f4807921db4444.jpg)  
PSNR: 25.02

![](images/048c100daf49438eeefdbb1c9869cf6e60553e356d04b5c2dec4350e0f299033.jpg)  
PSNR: 27.85

![](images/9058f0432839d94be86f1f93e373a49497dec042ccca4fc11d5cafedbc3e67e2.jpg)

![](images/f8b0012b5c435818da980b5f07c9ca294bb2bd25706f6ad2c1cf040323434b0d.jpg)  
PSNR: 24.97

![](images/504711a079886ad5f4bd7fbedeae2f67de9f4d1cc3cbab3f7afa164e896818ff.jpg)  
PSNR: 24.21

![](images/c522787c2d32bbea1b4c462cb3d5b2526ac88842de974f929e8d493b6252465e.jpg)  
PSNR: 25.91  
PSNR: 23.54

![](images/1499add5e409715e8f3d20a64517fc643a3c6d1a8dc2a5ff19621ae94e58bfc7.jpg)

![](images/af8a9d887a87cd7ab5aa53565a8b12d36ec6fe51dc2ab3d1c178c45e0d984e67.jpg)  
PSNR: 29.76

![](images/44e00bd4ffe4ac14c981c0c5ea4deb9b85bec02dda13f27ce3291eef7860f99e.jpg)  
PSNR: 20.77  
PSNR: 30.11

![](images/77dd19a7a2e0966bacff2aa93d0bb854f0b5859b08457dda075db49e6d93d9d8.jpg)  
PSNR: 26.38

Figure 9: Additional qualitative comparisons on the CelebA dataset across linear inverse tasks, including box inpainting (1st row), random inpainting (2nd row) and super-resolution (3rd row).

Clean

Degraded

OT-ODE

D-Flow

Flow Prior

Flow DPS

Flower

![](images/9f61ae413987253eab7ee530f6a1d5f65deeb0ce45f1a1659cc4b0182f1926f3.jpg)  
PNP-Flow

![](images/59b30ec2d2bbe39fc234705d34d7d56b2773b6bdfdf400c558066847728008b8.jpg)

![](images/1f7703d2380474a8e96a77cf92d30d909f1018406f1e1ce40ec8450d37143f45.jpg)

![](images/b122c552e12cac04487248c6c313a015007ee9e3f766ed0dac5ff9bbb09aa0d6.jpg)

![](images/e4e298615138042f6ebd96dfa48ee77d6115102d083acc79dbf6f3139e4e2044.jpg)  
PNP-CF (ours)

![](images/1e2975abe961149bdb025cd2ce78e2c50c56ddc115a947e1599cff6a0d5d14e6.jpg)

![](images/5cdcdf93cf39d6a7de082ff6abf94c97e1d1fcbc933a2567cbad0b8f7c925161.jpg)

![](images/00880f62a7f494f3e59810ea2844afec4108009752541e20dfff42d702e384b5.jpg)

![](images/de4bf8fc8da21a219e787f6ad1a2da2e5a075978c6c5581733d1a0e686e995dd.jpg)

![](images/b0faaf5879f962460f18decc6681a9a7d43d2ac223eb5758e45a373322e1e614.jpg)

![](images/92dc0c4d4743b0fd4194513b9e8ae671792fb90ee3e637ab0fa5f479b1a81509.jpg)

![](images/32475e89c3a6b3b094602fa8fa56c04f456e8a351d9e9abf657b41c83bf2455c.jpg)

![](images/0660f924529fe15c9c462ceee7ff1e601a7c2ef98ba4a1f5897c5d01102019c9.jpg)  
PSNR: 10.89

![](images/61b12142fcf79368d68bd6259d73323922ea51c63982995a1842cda9153a77bd.jpg)  
PSNR: 31.81

![](images/aaf6341c993971ee71da020763cf11844a89629c401d5cd0ef56aa45231e35e5.jpg)  
PSNR: 36.72

![](images/742783b8d9c4fb7149ada8e556e478412fa5ee46ae6ae6b320d0e3b06ceab5a5.jpg)  
PSNR: 36.25

![](images/39c38d59541802bc6ab1e2ba3fa1236aa7fe5724a6097dc9bb4b1c273a200fc9.jpg)  
PSNR: 35.72  
PSNR: 36.20  
PSNR: 36.13

![](images/dedc0eda7f8f26408360cecaf29bfba171d14c1d5182db01a952f46f75a72a39.jpg)

![](images/c03766151a52f0b13673eeb6f2141ddf946770760f154c10a055f481bc3b4527.jpg)

![](images/a2f6273823813ff41763cddd679314d862d37bb0cb6bde2ea4c658ba6834165c.jpg)  
PSNR: 12.12

![](images/bda2ea401b5d087b53ae1d01616d31d26becb8cc65ecbab207b89cf094012d8e.jpg)  
PSNR: 37.84  
PSNR: 25.37

![](images/01577611b12d748609b19908f958802bf0060c800dd3f3ea0c2c7cbcf8d86951.jpg)  
PSNR: 23.28

![](images/0f6c6410fc9b2d2bbe9945005e47b2c6ba629641218cb23eb5ab51161480bb85.jpg)  
PSNR: 18.60

![](images/1619ccddd58deb06426cd5c2aaf2faffdce2920de2581737fa4e111b6de1b030.jpg)  
PSNR: 24.73

![](images/2e1421d49fc076788132de1296f4ac5df6d27d39aa7720e9fe44a7e4b7ca658e.jpg)  
PSNR: 24.04

![](images/82c6407dee161719a5310be0625563b28021064f1698489bb80e2f252e9871dc.jpg)  
PSNR: 24.83

![](images/f781e6602c6b94518246d7c594630100454d102831feefb74935b1dabe48628b.jpg)  
PSNR: 26.52

Figure 10: Additional qualitative comparisons on the AFHQ-Cat dataset across linear inverse tasks, including box inpainting (1st row), random inpainting (2nd row) and super-resolution (3rd row).

## G Additional results on nonlinear tasks with Gaussian noise

Clean

Degraded

![](images/f10b14a0a237a257335e982a2b73f9ab9f6d7456200d211cb171a2df2014b966.jpg)

![](images/a4afe16920d1d992b2e686b6eda8c3217b4121bd8d16aa1b9566439bd15e5143.jpg)  
PSNR: 8.47

D-Flow

![](images/20c9b93fc041bc75dcf0f54d864e83e5b990904c0c47e786edbe836ac5a0b73e.jpg)  
PSNR: 12.16

Flow DPS

![](images/cde3544655668ef8eca3d53888ddd1c3595b97e38cd900345abd06678b931093.jpg)  
PSNR: 14.54

PNP-Flow

![](images/634a3e2d0719ea96c2806df3498eb5242849370ade12881263b8e06357d02b89.jpg)  
PSNR: 12.16

PNP-CF (ours)

![](images/f1692668da3e06ed56446f19d13e567dd796214b46c43a7c2b8ef7eb1d4456a4.jpg)  
PSNR: 38.11

![](images/70183d84888802f0ab805c4f52f10536909b410ffd916a4472d620b41b1cc8b6.jpg)

![](images/61cbf6911580b80578e4c534b2e946909cc9721387dacf35f55cdd3ef5268e40.jpg)  
PSNR: 20.10

![](images/31387a3efc250f41a9d77da4f66eceab960f264f1157b12ea29e60aff308b736.jpg)

![](images/5936aa513a6b8abbc1be5b5c54eb4c11f477ccdf39658427d7483ec9f1db3e35.jpg)

![](images/bd85c90fd570af3b8712c17e3af6ab9f715b594b9e52e06a52f12c57046057ab.jpg)

![](images/00b622205e93c2f7a7baf34f0324d037208734deb2a8737aa5238f067ce92036.jpg)

![](images/89792464a6b8e36984a00f6d1e579b8aa1f5cdfde8041118d253c9e6c24aa0f5.jpg)

![](images/d2e2424576949e9e40facaab8cee9e9600ddfb1dce7ab4da3ca4dc9939317bd0.jpg)  
PSNR: 24.84  
PSNR: 15.87

![](images/6dfa0763bf1d95c46839d845bc771c2f13a6d68ca5386e9a7ab3a843e26b5e48.jpg)  
PSNR: 23.36  
PSNR: 23.65

![](images/ea03bfacbf7df0a84341bd89ae24af96a9b19e509e75eddb618aa43211f9708a.jpg)  
PSNR: 23.56  
PSNR: 24.40

![](images/bd2c12c736811a1a99df409347da8c863ec9b397d9307cea814f80e6ae922df5.jpg)  
PSNR: 23.78  
PSNR: 26.21

![](images/4ad7de8984c8e1b8670f069d1e7b3299352f8ea40fbf28a736713bc25428298f.jpg)  
PSNR: 30.44

Figure 11: Additional qualitative comparisons on the CelebA dataset across nonlinear inverse tasks, including phase retrieval (1st row), nonlinear deblurring (2nd row) and HDR reconstruction (3rd row).

Clean

Degraded

![](images/b3a62874c6f407f518fc6a8920cb222917da738deccd391f7288c8f276f6a177.jpg)

D-Flow

![](images/7a4b8cfba933a1c6db3a384875977a1940b11dad3df3b6903f86c564e1567442.jpg)

![](images/35bd25bd6ed5726894fa9f777c2de0e39e6c39476cc5219b01288321cce3cafb.jpg)

Flow DPS

PNP-Flow

![](images/5013ed9f1abee0d7afccdcb36c55e16a80e50334bc26f2a603a4565c2e508b6c.jpg)

PNP-CF (ours)

![](images/80528babde6a972bfeae58cb08938bc292b421bdeb75d481059172bd646a6878.jpg)

![](images/0c1249366b7ac2d7e1eb44b94dd53ea3bfcf932efe5170a54308cd7a9bccaf33.jpg)  
PSNR: 10.33  
PSNR: 14.50  
PSNR: 10.88

![](images/526bec75e8705d8b68a097ac5866bf1a6946a54ff9e33c32f2251275e6f06235.jpg)

![](images/0c10ee563e386a452a30319f6459244481d8b1d7f7b343c054fc1c0a9b10bd42.jpg)

![](images/45d5ad9344f1971161d92828c8c66609e04ba985458e44fac6a5b4bcce29305f.jpg)  
PSNR: 25.94  
PSNR: 29.37

![](images/d15a5650365c17cc701dbe9f2a9fdb879ac650cf5f45df7b484f973fc3a2370e.jpg)  
PSNR: 21.67

![](images/8b30e4aa1d5438c49a87ece4fdef690470229948502a13d285082b3a9674e1be.jpg)  
PSNR: 22.49  
PSNR: 16.84

![](images/dc9bfa901bef3041e299ebca72e80342b6bf592fc959204380c079c5091b4940.jpg)  
PSNR: 29.62

![](images/ae11995e3291c976bb8c46c75e108429b897a249b50065d68b831158b9983cc0.jpg)

![](images/4eb9edcf0823f84b88e80e170e6d7e9377e5dbb1822b861f92ffa2304787d099.jpg)  
PSNR: 26.71

![](images/617725cbf05a8b1f69b6536989dbec59764be8f0e739e26469477f9c50f32c5f.jpg)  
PSNR: 27.29  
PSNR: 25.59

![](images/f0d8245ca4ec5d891e6c5d4c1f057da3117263ca73376c9ffbec032d028291ae.jpg)  
PSNR: 32.46

Figure 12: Additional qualitative comparisons on the AFHQ-Cat dataset across nonlinear inverse tasks, including phase retrieval (1st row), nonlinear deblurring (2nd row) and HDR reconstruction (3rd row).

## H Additional results on segmentation forward operator

![](images/8e3afdb471f1b437c17776c5b0d9b838148a0964891cc1d9fe916b8de58cd85a.jpg)  
Figure 13: Qualitative comparison for segmentation-guided reconstruction on CelebA-HQ. From left to right: reference image, target segmentation mask, and reconstructions from PnP-Flow, FlowDPS, D-Flow, and PnP-CF. PnP-Flow exhibits mode collapse across different masks, while FlowDPS produces plausible but weakly conditioned reconstructions. D-Flow follows the segmentation constraint more closely but introduces severe visual artifacts. In contrast, PnP-CF produces visually plausible reconstructions while maintaining strong consistency with the target segmentation.