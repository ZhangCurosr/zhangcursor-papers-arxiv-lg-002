# Planning under Noise with Neural Operators

Michael Chen<sup>1,\*</sup>, Hong Chul Nam<sup>2,\*</sup>, Xi Deng<sup>1,\*</sup>, Anima Anandkumar<sup>1</sup>

<sup>1</sup>Computing and Mathematical Sciences, California Institute of Technology

<sup>2</sup>Center for Data Science, New York University

Equal contribution

## Abstract

Planning under uncertain control execution noise requires reliable feedback policies across diverse environments. We introduce Planning Under Noise (PUN), a neural operator that learns noise-aware value fields for planning in unseen geometries. Building on an established screened-Poisson formulation of stochastic optimal control, we address two challenges: specifying boundary conditions that support navigation across varying geometries and eficiently computing fields with reliable gradients. Zero-Dirichlet obstacle boundaries terminate sampled paths at walls, weakening the goal signal along long routes and making it harder to estimate reliably. We instead impose reflecting Neumann conditions on obstacles and a Dirichlet condition at the goal, allowing sampled paths to continue exploring free space until they reach the target. To solve the resulting PDE without a configuration-space mesh, we extend a Monte Carlo (MC) solver, Walk on Stars (WoSt), to higher dimensions. However, local MC fluctuations can reverse estimated gradient directions and cause planning failures, making direct planning require costly sampling. PUN amortizes this computation by learning geometry-to-field mappings from noisy, partially converged Monte Carlo labels, avoiding expensive convergence for every training example. At inference, test-time scaling uses bounded local search guided by the predicted global field. We evaluate PUN on planning tasks across dimensions. Across four noise levels, PUN matches or outperforms all compared planners, achieving 95–100% success on the 4D replanning benchmark while maintaining the lowest runtime. At the highest tested noise level in 4D (λ = 4), PUN achieves 10.7–60.6% relative gains in success rate over other methods while running 3.29–32.15× faster.

## 1 Introduction

Planning (Latombe, 1991) is one of the most fundamental problems in robotics. Given a robot’s configuration space (LaValle, 2006), planning computes a path from the current configuration to a goal that is optimal under a given cost function. Real robots do not execute a commanded motion exactly (Thrun, 2002). These errors accumulate along a motion, so a route planned with a few centimetres of clearance can be executed into a wall. Sampling-based planners such as PRM (Kavraki et al., 1996) and RRT (LaValle & Kufner, 2001), and graph search such as A<sup>⋆</sup> (Hart et al., 1968), plan for a nominal system that executes commands exactly and leave the noise to a tracking controller. Sampling-based methods such as MPPI (Williams et al., 2017a) and PI<sup>2</sup>-CMA (Stulp & Sigaud, 2012) optimize control sequences or policy parameters using sampled trajectory costs. Optimal feedback under execution noise can be derived from a value function, which gives the minimum expected cost of reaching the goal from each configuration and satisfies the nonlinear Hamilton–Jacobi–Bellman (HJB) equation (Fleming & Soner, 2006). Solving this equation on a grid scales exponentially with the dimension of the configuration space (Bellman, 1957; Han et al., 2018).

While the problem can be converted into a linear screened Poisson equation (Theodorou et al., 2010) under control–noise matching (Assumption 1), a central dificulty in Poisson-based planning is specifying boundary conditions that produce a reliable value field across varying geometries.

Conventional mesh-based numerical methods can solve the screened Poisson equation eficiently in two and three dimensions. However, a robot with d degrees of freedom (DoF) induces a d-dimensional configuration space, making volumetric discretizations and numerical PDE solution increasingly expensive. Planning must also accommodate complex and previously unseen obstacle geometries, for which constructing a new mesh and solving the PDE from scratch can be costly. Mesh-free Monte Carlo methods (Muller, 1956; Sawhney & Crane, 2020; Sawhney et al., 2023; Jambon et al., 2026) support PDE solution on complex geometries, motivating their use for planning in higher-dimensional configuration spaces.

Among these, Walk-on-Spheres (WoS) (Muller, 1956) estimates the solution at a point by jumping repeatedly to a uniformly sampled point on the largest ball fitting in free space, until it reaches a boundary of known value, which requires every reachable boundary to be Dirichlet. Walk-on-Stars (WoSt) (Sawhney et al., 2023) accommodates reflecting obstacles by replacing that ball with a star-shaped region that may contain part of the obstacle boundary, keeping every boundary point visible from the center so that one ray continues the walk instead of branching at each contact. The screened kernels for Walk-on-Stars have been derived only in two and three dimensions (Sawhney et al., 2023). Moreover, directly planning with Monte Carlo estimates is unreliable: trajectories depend on local value gradients, and even modest estimator noise can corrupt or reverse the estimated direction, leading to planning failure.

![](images/286d5806e305f80a746074761bd7b798da86400e7157d18e32bd15241c56aa2e.jpg)

![](images/4ed31d29dbd4bda833ac10cc359ca9020b9559471029022366c2be476f6701c6.jpg)

![](images/1ae486f54e259ac26da17389b0de0a93bb9dca87484df7fefb3cec0c26715281.jpg)  
Fig. 1: (a) Fast replanning. PUN has the lowest mean runtime on unseen 4D maps: 10.8 ms/start, compared with 32.8–387.1 ms/start for the baselines. Runtimes are averaged equally over the four execution-noise levels in Table 5 and plotted on a logarithmic axis. (b) Resolution transfer. FNO maintains over 90% local-search success across all <sup>1</sup>five resolutions without retraining, reaching 92.50% at 4096 × 4096 after training at 256 × 256. Missing points indicate out-of-memory failures. (c, d) Continued goal seeking. In the illustrated geometry, the solution induced by the Dirichlet boundary cannot propagate goal information further since most trajectories are absorbed by obstacles. Consequently, it is more dificult for planners to find the target than with the mixed boundary condition.

Neural operators (Kovachki et al., 2023) can amortize expensive PDE solves by learning mappings from input geometries directly to solution fields. Moreover, prior work has shown that fully converged labels are unnecessary when training from Monte Carlo simulations: neural operators can recover accurate predictions from noisy, partially converged estimates (Viswanath et al., 2026). This enables training of neural operators with inexpensive Monte Carlo supervision while providing fast, smooth value-field predictions at inference time.

We propose Planning Under Noise (PUN), shown in Fig. 2, an operator-learning framework that maps environment geometry to value fields that account for control noise. Boundary conditions can be understood through a heat analogy. With unit desirability at the goal and zero desirability on obstacle boundaries, the goal acts like a hot source, while the obstacle walls act like cold sinks. The zero-Dirichlet condition fixes the field to zero along each obstacle surface, making the entire obstacle boundary a heat sink. In a long or narrow passage, heat spreading from the goal therefore leaks into the walls along the route. The farther the goal’s influence must travel, the more it is attenuated, leaving a weak signal in distant parts of free space that discretization, sampling, or learning errors can obscure. Narrow passages make this particularly pronounced because the absorbing walls remain close on both sides. This attenuation is visible in Fig. 1(c), where the weak goal signal provides insuficient directional guidance for starting points far from the goal, making it dificult to navigate toward the target. Moreover, the screened Poisson equation already includes decay throughout free space through its screening term; absorbing obstacle boundaries introduce an additional source of attenuation. We therefore keep a Dirichlet condition at the goal but impose homogeneous Neumann (zero-flux) conditions on obstacle boundaries. The walls now act like insulation: they prevent heat from escaping through the boundary without forcing the desirability itself to vanish there. Heat can spread along a corridor without continually draining into its walls, allowing the goal’s influence to reach farther into free space. Screening still causes decay, but the additional loss through obstacles is removed. This is precisely the efect illustrated in Fig. 1(d): compared with the absorbing Dirichlet field in Fig. 1(c), the mixed-boundary field extends the goal’s influence much farther through the environment, preserving a stronger signal for planning along long routes. The Neumann condition also makes the value gradient parallel to smooth obstacle boundaries; therefore, gradient-based planning directs motion along walls rather than into them. This permits guidance along walls in narrow passages without the strong repulsion induced by zero-Dirichlet conditions. Under our regularity assumptions, the value field and control remain bounded at obstacles, whereas zero-Dirichlet desirability makes both diverge there (Appendix C.4).

We first derive dimension-general Poisson kernels and Monte Carlo weights, extending mesh-free supervision beyond three dimensions for scalability. This enables supervision generation without a volumetric configurationspace mesh. We then train PUN using noisy, partially converged Monte Carlo estimates in the weak supervision paradigm. By amortizing the PDE solution across geometries, PUN produces spatially coherent value fields that guide gradient following and local search without requiring a fully converged Monte Carlo solve for every environment. We evaluate PUN across 2D–4D environments and compare its planning success and runtime with nominal planners and stochastic controllers under execution noise. In the experiments varying the execution noise level λ, we plan a path from the current state at each control step, follow it briefly, and repeat from the updated state; we refer to this as replanning. Our contributions can be summarized as follows:

![](images/fcaeb815fda89c80e1eaaaa0c43137742485819c05e1bfcd751b0392073f5165.jpg)  
Fig. 2: Planning Under Noise: (A) Planning: a neural operator maps geometry, goal, cost, and noise inputs to a desirability field. Grid-based local search selects feasible routes with high endpoint desirability, executes a short prefix, and replans. (B) Training: Walk-on-Stars (WoSt) estimates provide pointwise supervision; neural caching supplies learned continuation at the walk horizon.

• We formulate Poisson-based path planning as a mixed boundary-value problem, imposing a Dirichlet condition at the goal and homogeneous Neumann conditions on obstacle boundaries.

• We introduce an operator-learning framework that maps environment geometry to noise-aware value fields, enabling zero-shot planning in unseen complex geometries without per-instance retraining or numerical PDE solves.

• We generalize the mesh-free Monte Carlo solver WoSt beyond three dimensions by deriving the dimensiondependent Green’s function and step weights for the screened Poisson equation.

• We introduce a test-time scaling strategy that evaluates reachable candidate states on a local lattice using a fixed global learned field, without retraining, improving planning success over direct gradient following in the reported 2D–4D evaluations (Tables 1 and 11).

On 2D application benchmarks, zero-shot WoSt-supervised FNO reaches 93.62% local-search success, up from 83.87% with gradient following on the same fields (Table 1). With the same local-search, FNO-based PUN achieves approximately 941× and 58× speedups over direct WoSt on the 2D zero-shot application benchmarks and 4D manipulator test set, respectively (Table 12). In 4D replanning, PUN achieves 95.00–100% success at 10.51–11.66 ms/start across the tested noise levels (Table 5; Fig. 1(a)). At λ = 4, PUN is 3.29× faster than RRT and 5.09–32.15× faster than A<sup>⋆</sup>, Dijkstra, MPPI, and PI<sup>2</sup>-CMA. FNO also maintains over 90% local-search success across resolutions up to 4096 × 4096 (256× as many grid points as used in training) without retraining (Fig. 1(b)).

## 2 Preliminaries

In this section we derive the elliptic boundary value problem that our operator learns, and fix the notation used throughout.

## 2.1 Stochastic Formulation of Path Planning

Control problem. Let $\Omega \subset \mathbb { R } ^ { d }$ and let $\xi \in \overline { { \Omega } }$ be the starting point. The controlled process $X ^ { \xi , \mathbf { u } } = ( X _ { t } ^ { \xi , \mathbf { u } } ) _ { t \geq 0 }$ has interior dynamics

$$
\begin{array} { r } { \mathrm { d } X _ { t } ^ { \xi , \mathbf { u } } = \left( \mathbf { b } ( X _ { t } ^ { \xi , \mathbf { u } } ) + G ( X _ { t } ^ { \xi , \mathbf { u } } ) \mathbf { u } _ { t } \right) \mathrm { d } t + \Sigma ^ { 1 / 2 } ( X _ { t } ^ { \xi , \mathbf { u } } ) \mathrm { d } \mathbf { W } _ { t } , \qquad X _ { 0 } ^ { \xi , \mathbf { u } } = \xi , } \end{array}\tag{1}
$$

Algorithm 1 Walk(ξ, Ψ −), constant $\kappa ,$ max step Algorithm 2 Training with m queries and $N \geq 2$ walks   
K per query   
1: In: start point ξ; EMA target $\Psi _ { \theta }$ 1: In: environments $\mathcal { E } ; m , N , K .$ , loss weights, learning and EMA   
2: w ← 1 rates   
3: for $j = 1$ to K do 2: Initialize θ and $\theta ^ { - }  \theta$   
4: break if $\xi \in \Gamma _ { D }$ 3: repeat   
5: $r \gets r ( \xi )$ ▷ Eq. 6 4: Sample $a \sim \mathcal { E } ,$ , queries $x ^ { i } ,$ and goal points   
6: Sample $( \mathbf { v } , \rho )$ as in Section 4.1 5: Evaluate ${ \mathcal { T } } _ { \theta } [ a ]$ and detached $\check { T _ { \theta ^ { - } } }$ − [a]   
7: $w  w Q _ { d } ^ { \kappa } ( \rho , r ) ; \xi  \xi + \rho \mathbf { v }$ 6: $\hat { \Psi } ^ { i , n } \gets \mathrm { W a L K } ( \xi ^ { i } , \Psi _ { \theta ^ { - } } )$ , independently   
8: e $\mathbf { \xi } \xi \in \Gamma _ { D }$ then 7: $\begin{array} { r } { \widehat { \Psi } _ { i } \gets N ^ { - 1 } \sum _ { n } \hat { \Psi } ^ { i , n } } \end{array}$ ▷ Eq. 10   
8: $\theta  \theta - \alpha _ { \mathrm { l r } } \overline { { \nabla } } _ { \theta } ^ { \mathrm { ' s } } \mathfrak { L } ( \theta )$   
10: return $w g ( \xi )$   
11: end if 9: $\theta ^ { - }  \mathrm { ( 1 - \alpha \mathrm { - } \alpha \mathrm { - } \mathrm { m a } ) } \partial ^ { - } + \alpha \mathrm { { e m a } } \theta$   
12: return w sg $[ \Psi _ { \theta } .$ − (ξ)] 10: until training ends   
11: Inference: control from Eq. 11

with control $\mathbf { u } _ { t } \in \mathbb { R } ^ { k }$ and standard Brownian motion W; the process reflects normally at obstacles and workspace boundaries $\left( \Gamma _ { N } \right)$ and stops at the goal $\left( \Gamma _ { D } \right)$ . We minimize

$$
J ( x , \mathbf { u } ) = \mathbb { E } [ \int _ { 0 } ^ { \tau _ { D } ^ { \mathbf { u } } } \Big ( q ( X _ { t } ^ { \xi , \mathbf { u } } ) + \frac { 1 } { 2 } \mathbf { u } _ { t } ^ { \top } R \mathbf { u } _ { t } \Big ) \mathrm { d } t  +  \phi ( X _ { \tau _ { D } ^ { \mathbf { u } } } ^ { \xi , \mathbf { u } } ) | \xi = x ] ,\tag{2}
$$

where $\tau _ { D } ^ { \mathbf { u } } : = \operatorname* { i n f } \{ t \geq 0 : X _ { t } ^ { \xi , \mathbf { u } } \in \Gamma _ { D } \}$ is the first goal hit, $q \geq 0 , R \succ 0$ , and $\phi \geq 0$ is the terminal cost. Write $g = e ^ { - \phi / \lambda }$ for the Dirichlet desirability; operator learning fixes $\phi \equiv 0$ , hence $g \equiv 1$ , while the PDE identities retain general positive $g .$ Where the value $V ( x ) = \operatorname* { i n f } _ { \mathbf { u } } J ( x , \mathbf { u } )$ is $C ^ { 2 }$ it satisfies the stationary HJB equation, which is nonlinear: eliminating the optimal control leaves a term quadratic in ∇V (Appendix C.2, Eq. 29).

## Assumption 1 (Control–noise matching). There exists $\lambda > 0$ with $\Sigma ( x ) = \lambda { G ( x ) } R ^ { - 1 } { G ( x ) } ^ { \top }$ for all $x \in \Omega$

This is the standard condition of linearly solvable stochastic optimal control (Kappen, 2005; Todorov, 2006; 2009; Theodorou et al., 2010): a direction that is cheap to actuate is also one along which the system is uncertain, and λ sets the exchange rate between the two. This matching condition enables the exact linearization of the HJB equation used here. The Cole–Hopf change of variable $\Psi ( x ) \triangleq \exp ( - V ( x ) / \lambda )$ (Hopf, 1950; Cole, 1951; Fleming, 1977) cancels the quadratic term and turns the HJB equation into a linear elliptic equation for $\Psi ,$ as shown in Appendix C.2.

Path-planning specialization. For a fully actuated, passive system we take $\mathbf { b } = 0 , G = I { \mathrm { . } }$ , and $R = I ,$ so Assumption 1 forces $\Sigma = \lambda I$ . Writing the state cost as $\begin{array} { r } { q ( \dot { x } ) = \frac 1 2 c ( x ) ^ { 2 } } \end{array}$ with $c > 0$ , the linearized equation becomes the screened Poisson equation:

$$
\Delta \Psi - \kappa ^ { 2 } ( x ) \Psi = 0 , \qquad \kappa ( x ) \triangleq \frac { c ( x ) } { \lambda } .\tag{3}
$$

Throughout, we take c constant within each environment, so κ is constant.

Boundary conditions. Let D be the workspace, S the obstacle interiors, and $\mathcal { G }$ an open goal region. The free domain is $\Omega = D \setminus ( \overline { { S } } \cup \overline { { \mathcal { G } } } )$ , assumed connected, with boundary partitioned into $\Gamma _ { D } = \partial \mathcal { G }$ and $\Gamma _ { N } = \partial S \cup \partial D$ We impose

$$
\Psi = g \mathrm { o n } \Gamma _ { D } \qquad ( \mathrm { a b s o r b i n g ~ g o a l } ) , \qquad \partial _ { n } \Psi = 0 \mathrm { o n } \Gamma _ { N } \qquad ( \mathrm { r e f l e c t i n g } ) .\tag{4}
$$

The goal is the only Dirichlet set; obstacles and the outer domain boundary are reflecting in the PDE model. Under Assumption 2 the mixed problem has a unique solution with $\Psi \in ( 0 , 1 ]$ and $\operatorname* { i n f } _ { \overline { { \Omega } } } \Psi > 0$ , so log Ψ is bounded and the log parameterization is well defined (Proposition 3). See Appendix C.4 for a detailed analysis.

## 2.2 Neural Operator

A neural operator learns a map from problem data to a solution field (Li et al., 2021). Here we learn the desirability operator

$$
T : ( \underbrace { \mathbf { 1 } _ { \mathcal { S } } } _ { \mathrm { o b s t a c l e s } } , \underbrace { \mathbf { 1 } _ { \mathcal { G } } } _ { \mathrm { g o a l } } , \underbrace { c } _ { \mathrm { c o s t } } , \underbrace { \lambda } _ { \mathrm { n o i s e } } ) \longmapsto \Psi ,\tag{5}
$$

Both geometric inputs are masks on the workspace: $\mathbf { 1 } _ { \mathcal { S } }$ is the occupancy map of the obstacles and ${ \bf 1 } _ { \mathcal G }$ marks the goal region; goal data $g \equiv 1$ is fixed. A single trained model $\mathcal { T } _ { \theta }$ predicts the field for new environments. Its spatial gradients provide control through Eq. 11, and its values support local search (Section 4.4).

![](images/b0791683ed266cd79b7866a9aef717d64011463acb5ad944ef5710beb8aeb33d.jpg)  
Fig. 3: Replanning trajectories on the same 4D manipulator, start, and goal at λ = 0.25 (top) and λ = 2 (bottom). Columns show successive execution states; PUN’s trajectory is orange. PUN reaches the goal in both illustrated rollouts. Aggregate success and runtime are reported in Table 5.

## 3 Related Works

Path Planning. Classical planners divide into two families. Sampling-based methods such as PRM (Kavraki et al., 1996), RRT (LaValle & Kufner, 2001), and RRT<sup>⋆</sup> (Karaman & Frazzoli, 2011) build a graph of collisionfree motions, while search-based methods such as A<sup>⋆</sup> (Hart et al., 1968), Dijkstra (Dijkstra, 1959), and fast marching (Sethian, 1996) compute a cost-to-go over a discretization; both plan for a deterministic system and defer execution noise to tracking and replanning. Sampling-based methods such as MPPI (Williams et al., 2017a;b) and PI<sup>2</sup>-CMA (Stulp & Sigaud, 2012) optimize control sequences or policy parameters using sampled trajectory costs. Our implementations and their treatment of execution noise are described in Appendix D.1. We compare against these planners in Table 5. Planning Neural Operator (PNO) (Matada et al., 2025) learns the solution operator of the Eikonal equation from labels produced by a deterministic solver, but it is limited to deterministic systems. NTFields (Ni & Qureshi, 2023) and EikoNet (Smith et al., 2021) train a physics-informed network on the Eikonal residual, which fits a single geometry per model and must be retrained for each new environment. In contrast, PUN is trained over a family of geometry classes with approximate Monte Carlo supervision across diverse noise ranges to support planning with diferent noise spectra while achieving reliable stability.

Monte Carlo Methods. Monte Carlo estimators for PDEs can be categorized depending on the type of PDE, include Walk-on-Spheres (Sawhney & Crane, 2020), Walk-on-Boundaries (Sugimoto et al., 2023), and Backward Stochastic Diferential Equations (BSDE) (Zhang, 2017). Recent works combine Monte Carlo methods with neural networks to further reduce variance (Nam et al., 2024; Li et al., 2023a), which train a network by regressing on noisy Monte Carlo targets and thereby amortize repeated pointwise Monte Carlo estimation into a forward pass. Two of their ingredients we adopt directly: a deterministic cap on the number of walk steps, and evaluation of the partially trained model when a walk exceeds that cap. The difusion loss (N¨usken & Richter, 2021) truncates an SDE simulation and supplements it with a boundary physics loss.

Neural Operator. Neural operators parameterize maps between function spaces and can be evaluated on diferent discretizations, with numerical accuracy depending on the representation (Kovachki et al., 2023). The architectures we compare difer in how they propagate information: FNO (Li et al., 2021), UNO (Rahman et al., 2023), LocalNO (Liu-Schiafini et al., 2024), DISCO (Ocampo et al., 2023), GINO (Li et al., 2023b), UPT (Alkin et al., 2024) and Transolver (Wu et al., 2024). We include U-Net (Ronneberger et al., 2015) as a fixed-resolution reference. Physics-informed operators such as PINO (Li et al., 2024) add a PDE residual to the objective; we use walk labels as the primary signal with regularization on gradient terms.

## 4 Planning under Noise (PUN) with Neural Operator

PUN learns a desirability field from Walk-on-Stars supervision and uses the predicted field to guide local search. Labels are constructed directly in Ψ-space using finite-horizon walks, with learned continuation for unfinished trajectories. The neural operator is trained online on these labels. Algorithms 1 and 2 describe a single walk and the training loop, respectively; walks and terminal evaluations are batched.

## 4.1 The Walk-on-Stars Chain

Geometry and stopping. Let $\sigma = \operatorname* { i n f } \{ k : \xi _ { k } \in \Gamma _ { D } \}$ be the first exact goal-boundary hit, possibly infinite. The ideal chain stops at an exact hit, where the terminal value is $g ( \xi _ { \sigma } ) _ { }$ ; practical walks stop in a goal shell or at a finite horizon. For a spatial location x, the star radius is

$$
\begin{array} { c } { r ( \boldsymbol { x } ) = \operatorname* { m i n } \{ d _ { D } ( \boldsymbol { x } ) , d _ { \mathrm { s i l } } ( \boldsymbol { x } ) \} , } \\ { \mathrm { S t } ( \boldsymbol { x } ) = \mathrm { t h e ~ a d j a c e n t ~ c o m p o n e n t ~ o f } ~ B ( \boldsymbol { x } , \boldsymbol { r } ( \boldsymbol { x } ) ) \cap \Omega . } \end{array}\tag{6}
$$

Here $d _ { \mathrm { s i l } }$ is the distance to the closest visibility silhouette of $\Gamma _ { N }$ . An admissible star step additionally requires that this component be star-shaped about x and that its boundary consist only of the sphere and $\Gamma _ { N }$ . Appendix D.1 describes the shell stopping rule and radius floor used in place of exact absorption.

Write $\xi _ { k }$ for the WoSt position after k steps, with $\xi _ { 0 } = \xi$ the starting point and $\xi _ { k }$ random for $k \geq 1$ . Given $\xi _ { k } ,$ construct $\mathrm { S t } ( \xi _ { k } )$ with radius $r _ { k } = r ( \xi _ { k } )$ . Draw a direction uniformly on the sphere in the interior or on the inward hemisphere at a smooth reflecting boundary point, then move to the first positive hit of the sphere or an obstacle/domain boundary:

$$
\mathbf { v } _ { k } \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } ) , \qquad \rho _ { k } = \mathrm { f i r s t - h i t ~ d i s t a n c e } , \qquad \xi _ { k + 1 } = \xi _ { k } + \rho _ { k } \mathbf { v } _ { k } .\tag{7}
$$

The hemisphere restriction replaces the first distribution at $\xi _ { k } \in \Gamma _ { N }$ . We assume measurable geometry queries and positive hit distances almost surely.

On an admissible star, the direction law has surface density

$$
p ( x , \mathrm { d } \mathbf { z } ) = \frac { n _ { \mathbf { z } } \cdot ( \mathbf { z } - x ) } { \alpha ( x ) \omega _ { d } \| \mathbf { z } - x \| ^ { d } } \mathrm { d } S ( \mathbf { z } ) , \qquad \int _ { \partial \mathrm { S t } ( x ) } p ( x , \mathrm { d } \mathbf { z } ) = 1 ,\tag{8}
$$

where $\omega _ { d } = 2 \pi ^ { d / 2 } / \Gamma ( d / 2 )$ and $\alpha = 1$ in the interior or $1 / 2$ at a smooth boundary pole. Tangencies have zero density. Where a step is not admissible, Eq. 7 still defines a law on first hits, but Eq. 8 no longer describes it.

The estimator. Screening discounts each step by $W _ { k } = Q _ { d } ^ { \kappa } ( \rho _ { k } , r _ { k } ) \in ( 0 , 1 )$ , given in closed form for every d in Appendix C.5. For constant κ, Green’s identity on each admissible active star gives

$$
\begin{array} { r } { \mathbb { E } [ W _ { k } \Psi ( \xi _ { k + 1 } ) \mid \mathcal { F } _ { k } ] = \Psi ( \xi _ { k } ) , \qquad k < \sigma . } \end{array}\tag{9}
$$

Multiplying by the preceding weights makes $\begin{array} { r } { M _ { k } = ( \prod _ { i < k \wedge \sigma } W _ { j } ) \Psi ( \xi _ { k \wedge \sigma } ) } \end{array}$ a bounded martingale. This gives the finite-horizon identity below without requiring an exact goal hit in finitely many steps. The identity concerns the weighted ray sampler; its steps need not follow Brownian exit distributions.

## 4.2 Label Construction

For a finite walk cap K, write $\tau = \sigma \land K$ . Repeated conditional expectation in Eq. 9, keeping the chain fixed after a goal hit, gives

$$
\mathbb { E } _ { x } \Bigg [ \Big ( \prod _ { j < \tau } W _ { j } \Big ) \Psi ( \xi _ { \tau } ) \Bigg | \xi _ { 0 } = x \Bigg ] = \Psi ( x ) .\tag{10}
$$

The terminal factor is $g ( \xi _ { \sigma } )$ on a goal hit and the true continuation $\Psi ( \xi _ { K } )$ otherwise. With true continuation and admissible steps, averaging N independent replicas gives an unbiased label. Training substitutes learned continuation as described below. See Appendix C.6 for the log-space alternative and its bias correction.

Practical walks stop within a goal shell of width ε and use a radius floor. Under the smooth-domain and admissibility assumptions, shell stopping with true continuation changes the mean by at most $C _ { \Psi } \varepsilon$ , where $C _ { \Psi } = \operatorname* { s u p } _ { \overline { { \Omega } } } \| \nabla \Psi \|$ . The radius floor can violate admissibility and introduces an additional error of unknown sign (Sawhney et al., 2023). Appendix D.1 gives the implementation and the finite-horizon shell bound.

Bootstrapping. At the horizon Ψ is not available, so training substitutes a detached EMA copy of the model. The resulting bias is that copy’s own error at the endpoint, weighted by the product of the step weights over the K steps, so it is discounted wherever steps are long compared with the screening length.

## 4.3 Operator Parameterization and Objective

For $\lambda > 0$ and a positive diferentiable prediction, the desirability field defines the feedback control through its logarithm:<sup>1</sup>

$$
\Psi _ { \theta } = { \mathcal { T } } _ { \theta } [ a ] , \qquad V _ { \theta } = - \lambda \log \Psi _ { \theta } , \qquad \mathbf { u } _ { \theta } = \lambda R ^ { - 1 } G ^ { \top } \nabla \log \Psi _ { \theta } .\tag{11}
$$

For a coordinate-diferentiable decoder, the gradient is obtained by automatic diferentiation. A grid-output backbone requires a specified diferentiable interpolant or a spatial diferentiation rule. The input a is defined in Eq. 5, with $g \equiv 1$ fixed.

We combine value and goal supervision with gradient loss to train fields for downstream planning:

$$
\mathcal { L } _ { \mathrm { v a l } } = \mathbb { E } \big [ ( \Psi _ { \theta } - \widehat { \Psi } ) ^ { 2 } \big ] , \quad \mathcal { L } _ { \mathcal { G } } = \mathbb { E } \big [ ( \Psi _ { \theta } - g ) ^ { 2 } \big ] , \quad \mathcal { L } _ { \mathrm { g r a d } } = \frac { \mathbb { E } \big [ \| \nabla \Psi _ { \theta } - \nabla \widehat { \Psi } \| ^ { 2 } \big ] } { \mathrm { s g } \big [ \mathbb { E } \big [ \| \nabla \widehat { \Psi } \| ^ { 2 } \big ] \big ] } .\tag{12}
$$

The target gradient $\nabla \widehat \Psi$ is computed by finite diferences of the label field. The displayed gradient loss normalization assumes positive target gradient energy.

The objective is their weighted sum,

$$
{ \mathfrak { L } } ( \theta ) = { \mathcal { L } } _ { \mathrm { v a l } } + \beta _ { \mathcal { G } } { \mathcal { L } } _ { \mathcal { G } } + \beta _ { \mathrm { g r a d } } { \mathcal { L } } _ { \mathrm { g r a d } } .\tag{13}
$$

The goal loss carries the exact boundary value $g$ rather than a walk label. The gradient term is included because a small value error need not give a small gradient error, and the control in Eq. 11 is computed from the gradient: $\mathcal { L } _ { \mathrm { g r a d } }$ normalizes gradient error by target gradient energy, treating the denominator as fixed during diferentiation. Direct-Ψ supervision avoids log-transform bias and weights absolute field error, giving small absolute errors in low-desirability regions less weight than log-space supervision. Appendix C.6 reports the log-space alternative, in which errors are measured on a relative scale.

## 4.4 Grid-Based Local Search

PUN supports test-time scaling by increasing the local search budget without modifying the learned global field. More depth lets the search evaluate more reachable endpoints, while using the field for guidance. The same endpoint search can be run on other fields; in PUN, it is run on a global noise-aware field that is reused across replans.

Paths are extracted from $\Psi _ { \theta }$ on a lattice. Let $\mathcal { X } _ { h }$ be the free cells at spacing h. The feasible neighbors of x are $\mathcal { N } ( x ) = \{ x ^ { \prime } \in \mathcal { X } _ { h } : \| x ^ { \prime } - x \| _ { 2 } = h \}$ , the axis-adjacent free cells. Write $\mathcal { R } _ { k } ( x )$ for the states reachable from x in at most k such moves, including x itself. A plan is a feasible lattice path $\pi = ( x _ { 0 } , \ldots , x _ { \ell } )$ with $\ell \leq L ,$ scored by the predicted desirability $\Psi _ { \theta } ( x _ { \ell } )$ at its endpoint. For $\lambda > 0$ and $\Psi _ { \theta } > 0$ , maximizing this score is equivalent to minimizing $V _ { \theta } = - \lambda \log \Psi _ { \theta }$ at the endpoint. The score itself is well defined for any finite prediction.

A controller is specified by a depth L and commit length $e \leq L$ . With one edge committed per replan, the next move is

$$
x _ { 1 } \in \underset { y \in { \mathcal N } ( x _ { 0 } ) } { \arg \operatorname* { m a x } } ~ \underset { z \in { \mathcal R } _ { L - 1 } ( y ) } { \operatorname* { m a x } } ~ \Psi _ { \theta } ( z ) ,\tag{14}
$$

a step towards the largest desirability reachable through a candidate neighbor. For $e > 1$ , a path maximizing endpoint desirability is constructed, its first min(e, ℓ) edges are executed, and the search is repeated from the new state. Execution stops early upon reaching the goal. See Appendix D.1 for further details.

A key idea is that evaluating reachable endpoints permits an intermediate decrease in desirability when it leads to a better state, so local fluctuations need not determine the next move. This might happen when the field is not smooth and there are local valleys, so local search allows jumps over imperfect predictions. Appendix C.8 derives the graph-search rule and its large-depth limit, and Section 5 evaluates planning performance.

Optionally, local search can include a soft wall penalty,

$$
C _ { r } ( \pi ) = \sum _ { j = 1 } ^ { \ell } \left[ \frac { \left( a _ { r } ( x _ { j } ) - d _ { \mathrm { w a l l } } ( x _ { j } ) / h \right) _ { + } } { r } \right] ^ { 2 } ,
$$

Table 1: 2D application-benchmark evaluation of neural backbones trained with WoSt supervision and direct WoSt inference: planning success and relative $L _ { 2 }$ error against the FEM ground-truth field.
<table><tr><td>Model</td><td>Gradient (%)</td><td>Local search (4) (%)</td><td>Local search (10) (%)</td><td>Relative L2</td></tr><tr><td>UPT</td><td>2.48</td><td>2.78</td><td>3.57</td><td>0.5533</td></tr><tr><td>Transolver</td><td>3.26</td><td>5.34</td><td>15.30</td><td>0.5158</td></tr><tr><td>WoSt</td><td>2.71</td><td>6.30</td><td>36.45</td><td>0.1852</td></tr><tr><td>GINO</td><td>46.56</td><td>51.24</td><td>53.11</td><td>0.2628</td></tr><tr><td>U-Net</td><td>74.21</td><td>84.22</td><td>88.98</td><td>0.1605</td></tr><tr><td>DISCO</td><td>82.49</td><td>87.60</td><td>91.08</td><td>0.1041</td></tr><tr><td>LocalNO</td><td>80.34</td><td>86.63</td><td>91.35</td><td>0.0915</td></tr><tr><td>U-NO</td><td>78.08</td><td>86.11</td><td>91.68</td><td>0.0792</td></tr><tr><td>FNO</td><td>83.87</td><td>91.25</td><td>93.62</td><td>0.0615</td></tr></table>

Table 2: FNO supervision on 2D application benchmarks: success and relative error.
<table><tr><td>Source</td><td>Grad. (%)</td><td>LS (%)</td><td>Rel. err.</td></tr><tr><td>WoSt</td><td>83.87</td><td>93.62</td><td>0.0615</td></tr><tr><td>Deep Ritz</td><td>81.41</td><td>91.46</td><td>0.0608</td></tr><tr><td>PINO</td><td>32.50</td><td>59.69</td><td>0.5644</td></tr></table>

Table 3: Walk-horizon ablation on the generated 2D test set: planning success (%).  
Table 4: FNO loss ablations on 2D application benchmarks: planning success (%).
<table><tr><td>Steps</td><td>Grad. (%)</td><td>LS (%)</td></tr><tr><td>10</td><td>32.18</td><td>51.89</td></tr><tr><td>100</td><td>40.55</td><td>66.80</td></tr><tr><td>400</td><td>38.06</td><td>69.69</td></tr><tr><td>800</td><td>37.84</td><td>69.66</td></tr></table>

<table><tr><td>Model</td><td>Grad. (%)</td><td>LS (%)</td></tr><tr><td>Full model</td><td>83.87</td><td>93.62</td></tr><tr><td>value loss</td><td>67.71</td><td>83.26</td></tr><tr><td>goal loss</td><td>82.93</td><td>93.11</td></tr><tr><td>gradient loss</td><td>80.17</td><td>92.04</td></tr></table>

where $( u ) _ { + } = \operatorname* { m a x } ( u , 0 )$ , r is the preferred clearance in cells, and $d _ { \mathrm { w a l l } }$ measures distance to obstacles or the outer boundary. Normally $a _ { r } = r ;$ near a goal reachable within the local depth, it decreases with the remaining free-edge distance to that goal. Among paths retaining at least half the best positive local field gain, we prefer the smallest penalty. Narrow passages and starts or goals near walls are still possible.

## 5 Experiments

We evaluate how backbone, supervision, training objective, and controller afect planning, then test transfer across grid resolutions and replanning under execution noise. Table 1 compares controllers on the same fields, while Table 5 compares complete planners under common execution rules. Models are pretrained on generated environments and evaluated on unseen maps without fine-tuning. The main 2D application suite contains 592 maps and 37,888 starts across robotics, terrain, marine, urban, and game environments (Appendix E). Success measures goal-reaching rate; relative $L _ { 2 }$ measures field error against FEM references. Unless specified, local search (LS) denotes local search (10). Bold marks the best value in each comparison, including ties. Experiments use NVIDIA B300 and B200 GPUs; training and evaluation protocols are in Appendix D.1.

Backbone and amortization. Table 1 compares FNO (Li et al., 2021), U-NO (Rahman et al., 2023), U-Net<sup>2</sup> (Ronneberger et al., 2015), GINO (Li et al., 2023b), LocalNO (Liu-Schiafini et al., 2024), DISCO (Ocampo et al., 2023), UPT (Alkin et al., 2024), and Transolver (Wu et al., 2024), all trained with WoSt supervision. FNO has the highest success under all three controllers and the lowest relative $L _ { 2 }$ error (0.0615); its local-search (10) success is 93.62%, compared with 91.68% for the next-best U-NO. These rankings apply to the tested implementations under the training-compute policy in Appendix D.1. Direct WoSt inference reaches only 2.71% gradient and 36.45% local-search (10) success, versus 83.87% and 93.62% for FNO. Thus, at the reported solver settings, learning from approximate WoSt targets yields more useful planning fields than direct inference. We use FNO in the remaining ablations.

Supervision. With FNO fixed, Table 2 compares WoSt targets, Deep Ritz training (E & Yu, 2018), and PDE-residual training with PINO (Li et al., 2024). WoSt achieves 93.62% local-search success, versus 91.46% for Deep Ritz and 59.69% for PINO; gradient success follows the same ordering (83.87%, 81.41%, and 32.50%). Deep Ritz has slightly lower field error than WoSt (0.0608 versus 0.0615), but lower planning success. Field error is therefore not a reliable indicator of how useful these learned fields are for control.

Walk horizon and learned continuation. In Table 3, we vary the walk cap using a smaller training budget of 16 maps, eight goals per map, four trajectories per query, and 40 epochs, evaluated on generated 2D maps. Local-search success increases from 51.89% at 10 steps to 69.69% at 400 steps, but is 69.66% at 800 steps. Gradient success instead peaks at 40.55% with 100 steps. Longer walks therefore do not uniformly improve planning: the horizon changes both sampled propagation and reliance on the learned continuation (Section 4.2).

Table 5: Replanning under execution noise on 2D and 3D application benchmarks and 4D manipulators. Each entry reports success (%) / runtime (ms/start). Bold marks the highest success and lowest runtime within each dimension and noise level, including ties.
<table><tr><td>Dim.</td><td>Method</td><td>λ = 0.25</td><td>λ = 1</td><td></td><td>λ = 2</td><td>λ = 4</td></tr><tr><td rowspan="6">2D</td><td>A*</td><td>59.00 43.63</td><td>42.33</td><td>47.09</td><td>35.00 51.18</td><td>23.67</td><td>53.76</td></tr><tr><td>Dijkstra</td><td>59.00 154.94</td><td>42.33</td><td>174.95</td><td>35.00 203.41</td><td>23.67</td><td>239.97</td></tr><tr><td>RRT</td><td>82.00 /98.45</td><td>68.67</td><td>103.87</td><td>52.33 112.27</td><td>36.67</td><td>124.71</td></tr><tr><td>MPPI</td><td>84.00 1734.11</td><td>61.00</td><td>1773.31</td><td>45.67 1838.73</td><td>31.67</td><td>2126.12</td></tr><tr><td>PI2-CMA</td><td>78.33 / 1330.76</td><td>60.00</td><td>1484.91</td><td>48.00 1524.70</td><td>35.00</td><td>1606.82</td></tr><tr><td>PUN (zero-shot)</td><td>91.67 /9.55</td><td>74.00</td><td>11.14</td><td>62.33 14.07</td><td>45.67</td><td>14.51</td></tr><tr><td rowspan="6">3D</td><td>A*</td><td>36.67 73.40</td><td>23.33</td><td>60.47</td><td>12.00 73.85</td><td>6.00</td><td>65.25</td></tr><tr><td>Dijkstra</td><td>37.33 /73.46</td><td>22.67</td><td>63.04</td><td>12.00 76.99</td><td>6.00</td><td>67.73</td></tr><tr><td>RRT</td><td>53.33 238.31</td><td>35.33</td><td>169.90</td><td>25.33 122.85</td><td>12.00</td><td>126.23</td></tr><tr><td>MPPI</td><td>62.67 532.31</td><td>34.67</td><td>577.84</td><td>18.67 554.11</td><td>7.33</td><td>704.31</td></tr><tr><td>PI2-CMA</td><td>47.33 452.25</td><td>22.00</td><td>435.01</td><td>21.33 475.98</td><td>8.67</td><td>606.55</td></tr><tr><td>PUN (zero-shot)</td><td>92.00 15.64</td><td>68.67</td><td>15.34</td><td>40.00 14.30</td><td>16.00</td><td>14.43</td></tr><tr><td rowspan="6">4D</td><td>A*</td><td>100.00 16.81</td><td>96.67</td><td>17.08</td><td>72.50 78.02</td><td>59.17</td><td>62.44</td></tr><tr><td>Dijkstra</td><td>100.00 17.17</td><td>96.67</td><td>/17.34</td><td>72.50 66.75</td><td>59.17</td><td>59.39</td></tr><tr><td>RRT</td><td>98.33 24.20</td><td>95.83</td><td>25.80</td><td>90.00 43.00</td><td>85.83</td><td>38.31</td></tr><tr><td>MPPI</td><td>95.00 411.24</td><td>92.50</td><td>398.18</td><td>81.67 364.22</td><td>71.67</td><td>374.83</td></tr><tr><td>PI2-CMA</td><td>96.67 379.79</td><td>92.50</td><td>357.69</td><td>86.67 355.44</td><td>78.33</td><td>356.72</td></tr><tr><td>PUN</td><td>100.00 10.63</td><td>100.00</td><td>10.51</td><td>100.00 10.55</td><td>95.00</td><td>11.66</td></tr></table>

Further details of this study are in Appendix D.2. These success rates should not be compared directly with the full-budget application results.

Training objective. In Table 4, we remove one loss term at a time from Eq. 13. Removing value supervision causes the largest decrease: gradient success falls from 83.87% to 67.71%, and local-search success from 93.62% to 83.26%. Removing goal or gradient supervision gives smaller local-search decreases, to 93.11% and 92.04%, respectively. All three terms help in this setting, with value supervision making the largest measured contribution. These ablations use direct desirability supervision.

Path extraction. On the same learned FNO fields, Table 1 shows success rising from 83.87% with gradient following to 91.25% with local search (4) and 93.62% with local search (10), a total gain of 9.75 percentage points. This test-time refinement requires no retraining. In 2D, the two local-search variants use (L, e) = (5, 1) and (10, 3), so both depth and committed prefix length change. Local search (10) also exceeds gradient success for every other backbone, although it reaches only 36.45% on direct WoSt fields. Looking beyond immediate neighbors can therefore recover useful routes, but does not eliminate dependence on field quality. Figure 7 illustrates the two search neighborhoods on one fixed field; controller details are in Appendix D.1.

Transfer across discretizations. In Fig. 1(b), we test models trained at 256 × 256 on grids up to 4096 × 4096, without retraining. FNO maintains 90.62–96.25% local-search (10) success across the five resolutions and reaches 92.50% at 4096 × 4096, compared with 71.25% for U-NO and 57.81% for U-Net. U-NO leads at 512 × 512 (95.31% versus 94.06% for FNO); FNO leads at the other four resolutions. GINO runs out of memory above 256 × 256. The quantitative comparison supports FNO’s transfer across these discretizations; Fig. 4 separately illustrates predicted fields and paths at three resolutions.

Replanning under execution noise. Table 5 compares PUN with A<sup>⋆</sup> (Hart et al., 1968), Dijkstra (Dijkstra, 1959), RRT (LaValle & Kufner, 2001), MPPI (Williams et al., 2017a), and PI<sup>2</sup>-CMA (Stulp & Sigaud, 2012) on 2D and 3D application benchmarks and 4D manipulators. Obstacle hits count as failures (Appendix D.1). PUN has the highest reported success in all 12 dimension–noise settings, tying A<sup>⋆</sup> and Dijkstra at λ = 0.25 in 4D. At λ = 4, PUN exceeds the strongest baseline, RRT, by 9.00 percentage points in 2D (45.67% versus 36.67%), 4.00 in 3D (16.00% versus 12.00%), and 9.17 in 4D (95.00% versus 85.83%). PUN also has the lowest reported runtime throughout: 9.55–14.51 ms/start in 2D, 14.30–15.64 in 3D, and 10.51–11.66 in 4D. At λ = 4 in 4D, its 11.66 ms/start is 3.29 times faster than the fastest baseline, RRT (38.31 ms/start); Fig. 1(a) summarizes mean 4D runtimes. The advantage is relative: PUN’s 3D success still falls from 92.00% to 16.00% as noise increases. These comparisons concern the reported implementations and settings. Figure 3 shows PUN reaching the goal in two illustrative 4D rollouts at λ = 0.25 and 2.

## 6 Discussion

Extensions. Several extensions are possible. A spatially varying cost c(x) is the most useful: with constant c, the state-cost term penalizes elapsed time and the objective also penalizes control efort. Reflection carries no additional contact cost; choosing c(x) as a function of obstacle clearance would add a clearance-dependent running cost. Transient problems replace the stationary equation by its time-dependent counterpart, giving a finite-horizon value function rather than a first-exit one, which suits tasks with deadlines or moving goals. Partial information is the most direct use of our amortization: since the operator maps geometry to a field in one forward pass, replanning as a robot discovers obstacles requires only a forward pass on the updated geometry instead of a new solve. The open question is how to condition the operator on a partially observed occupancy map.

Limitations. The value field is derived for reflecting boundaries. Local search allows its use in collision-failure tasks, but nominal graph feasibility does not guarantee collision-free execution under noise. Deriving the exact value function for that execution model would require a corresponding boundary condition. Assumption 1 is a further restriction. The control–noise matching assumption enables exact HJB linearization and is standard in classical path-integral control (Kappen, 2005; Theodorou et al., 2010). Our experiments are also limited to settings where this assumption is reasonable: we test geometric planning problems in up to four dimensions and do not include full robot models with contact. Evaluating on humanoid robots with contact would test both the assumption and the capacity of the operator. Finally, adversarial geometries such as mazes remain challenging, since long corridors can lead to weak or poorly resolved learned gradients.

## 7 Conclusion

We introduced Planning Under Noise (PUN), a neural operator framework that learns noise-aware planning fields from approximate Walk-on-Stars supervision. Dimension-general screening kernels support mesh-free supervision beyond three dimensions, and local search converts predicted fields into paths on unseen geometries. Experiments across 2D–4D environments demonstrate zero-shot planning and efective learning from stochastic labels. In replanning comparisons under execution noise, PUN achieves the lowest reported runtime among the tested methods, while matching or exceeding the success of every compared planner at each tested noise level across 2D–4D.

## AI use statement

In this work, we used generative AI tools to assist with parts of method implementation. Additionally, we used generative AI tools to polish writing and proofs and identify relevant literature. We did not use generative AI tools for other applicable tasks requiring disclosure. All AI-assisted work was reviewed and verified by the authors. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## Acknowledgements

Anima Anandkumar and Xi Deng are both supported in part by the Bren endowed chair and the AI2050 senior fellow program at Schmidt Sciences. Hong Chul Nam is supported by a PhD fellowship for the Center for Data Science at NYU.

## References

Benedikt Alkin, Andreas F¨urst, Simon Schmid, Lukas Gruber, Markus Holzleitner, and Johannes Brandstetter. Universal physics transformers: A framework for eficiently scaling neural operators. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Richard Bellman. Dynamic Programming. Princeton University Press, Princeton, NJ, 1957.

Julian D. Cole. On a quasi-linear parabolic equation occurring in aerodynamics. Quarterly of Applied Mathematics, 9(3): 225–236, 1951.

Edsger W Dijkstra. A note on two problems in connexion with graphs. Numerische Mathematik, 1(1):269–271, 1959.

Weinan E and Bing Yu. The Deep Ritz method: A deep learning-based numerical algorithm for solving variational problems. Communications in Mathematics and Statistics, 6(1):1–12, 2018. doi: 10.1007/s40304-018-0127-z.

Wendell H. Fleming. Exit probabilities and optimal stochastic control. Applied Mathematics and Optimization, 4(1): 329–346, 1977.

Wendell H. Fleming and H. Mete Soner. Controlled Markov Processes and Viscosity Solutions, volume 25 of Stochastic Modelling and Applied Probability. Springer, New York, 2 edition, 2006.

David Gilbarg and Neil S. Trudinger. Elliptic Partial Diferential Equations of Second Order. Springer, 2nd edition, 2001.

Jiequn Han, Arnulf Jentzen, and Weinan E. Solving high-dimensional partial diferential equations using deep learning. Proceedings of the National Academy of Sciences, 115(34):8505–8510, 2018.

Peter E Hart, Nils J Nilsson, and Bertram Raphael. A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics, 4(2):100–107, 1968.

Eberhard Hopf. The partial diferential equation u<sub>t</sub> + uu<sub>x</sub> = µu<sub>xx</sub>. Communications on Pure and Applied Mathematics, 3 (3):201–230, 1950. doi: 10.1002/cpa.3160030302.

Cl´ement Jambon, Mohammad Sina Nabizadeh, and Mina Konakovi´c Lukovi´c. Walk on decomposed subdomains: A hybrid monte carlo–deterministic solver for elliptic pdes. ACM Transactions on Graphics (TOG), 45(4):132:1–132:22, 2026.

Hilbert J. Kappen. Path integrals and symmetry breaking for optimal control theory. Journal of Statistical Mechanics: Theory and Experiment, 2005(11):P11011, 2005.

Sertac Karaman and Emilio Frazzoli. Sampling-based algorithms for optimal motion planning. The International Journal of Robotics Research, 30(7):846–894, 2011.

Lydia E Kavraki, Petr Svestka, Jean-Claude Latombe, and Mark H Overmars. Probabilistic roadmaps for path planning in high-dimensional configuration spaces. IEEE Transactions on Robotics and Automation, 12(4):566–580, 1996.

Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to pdes. Journal of Machine Learning Research, 24(89):1–97, 2023.

Jean-Claude Latombe. Robot motion planning, volume 124 of The Kluwer International Series in Engineering and Computer Science. Kluwer Academic Publishers, 1991.

Steven M. LaValle. Planning Algorithms. Cambridge University Press, Cambridge, UK, 2006.

Steven M LaValle and James J Kufner. Randomized kinodynamic planning. The International Journal of Robotics Research, 20(5):378–400, 2001.

Zilu Li, Guandao Yang, Xi Deng, Christopher De Sa, Bharath Hariharan, and Steve Marschner. Neural caches for monte carlo partial diferential equation solvers. In SIGGRAPH Asia 2023 Conference Papers, pp. 1–10, 2023a.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial diferential equations. In International Conference on Learning Representations (ICLR), 2021.

Zongyi Li, Nikola Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, and Anima Anandkumar. Geometry-informed neural operator for large-scale 3d pdes. In Advances in Neural Information Processing Systems (NeurIPS), 2023b.

Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial diferential equations. ACM / IMS Journal of Data Science, 1(3):1–27, 2024.

Miguel Liu-Schiafini, Julius Berner, Boris Bonev, Thorsten Kurth, Kamyar Azizzadenesheli, and Anima Anandkumar. Neural operators with localized integral and diferential kernels. In International Conference on Machine Learning (ICML), 2024.

Sharath Matada, Luke Bhan, Yuanyuan Shi, and Nikolay Atanasov. Generalizable motion planning via operator learning. In International Conference on Learning Representations (ICLR), 2025.

Mervin E Muller. Some continuous monte carlo methods for the dirichlet problem. The Annals of Mathematical Statistics, 27(3):569–589, 1956.

Hong Chul Nam, Julius Berner, and Anima Anandkumar. Solving poisson equations using neural walk-on-spheres. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research. PMLR, 2024.

Ruiqi Ni and Ahmed H. Qureshi. Ntfields: Neural time fields for physics-informed robot motion planning. In International Conference on Learning Representations (ICLR), 2023.

Nikolas N¨usken and Lorenz Richter. Interpolating between BSDEs and PINNs: Deep learning for elliptic and parabolic boundary value problems. arXiv preprint arXiv:2112.03749, 2021.

Jeremy Ocampo, Matthew A. Price, and Jason D. McEwen. Scalable and equivariant spherical cnns by discrete-continuous (disco) convolutions. In International Conference on Learning Representations (ICLR), 2023.

Md Ashiqur Rahman, Zachary E. Ross, and Kamyar Azizzadenesheli. U-no: U-shaped neural operators. Transactions on Machine Learning Research, 2023.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical Image Computing and Computer-Assisted Intervention (MICCAI), pp. 234–241, 2015.

Rohan Sawhney and Keenan Crane. Monte carlo geometry processing: A grid-free approach to pde-based methods on volumetric domains. ACM Transactions on Graphics, 39(4), 2020.

Rohan Sawhney, Bailey Miller, Ioannis Gkioulekas, and Keenan Crane. Walk on stars: A grid-free monte carlo method for pdes with neumann boundary conditions. ACM Transactions on Graphics, 42(4), 2023.

James A Sethian. A fast marching level set method for monotonically advancing fronts. Proceedings of the National Academy of Sciences, 93(4):1591–1595, 1996.

Jonathan D. Smith, Kamyar Azizzadenesheli, and Zachary E. Ross. Eikonet: Solving the eikonal equation with deep neural networks. IEEE Transactions on Geoscience and Remote Sensing, 59(12):10685–10696, 2021.

Freek Stulp and Olivier Sigaud. Path integral policy improvement with covariance matrix adaptation. In International Conference on Machine Learning, 2012.

Ryusuke Sugimoto, Terry Chen, Yiti Jiang, Christopher Batty, and Toshiya Hachisuka. A practical walk-on-boundary method for boundary value problems. ACM Transactions on Graphics, 42(4), 2023.

Evangelos Theodorou, Jonas Buchli, and Stefan Schaal. A generalized path integral control approach to reinforcement learning. The Journal of Machine Learning Research, 11:3137–3181, 2010.

Sebastian Thrun. Probabilistic robotics. Communications of the ACM, 45(3):52–57, 2002.

Emanuel Todorov. Linearly-solvable Markov decision problems. In Advances in Neural Information Processing Systems (NIPS), volume 19, pp. 1369–1376, 2006.

Emanuel Todorov. Eficient computation of optimal actions. Proceedings of the National Academy of Sciences, 106(28): 11478–11483, 2009.

Hrishikesh Viswanath, Hong Chul Nam, Xi Deng, Julius Berner, Anima Anandkumar, and Aniket Bera. Operator learning using weak supervision from walk-on-spheres. arXiv preprint arXiv:2603.01193, 2026.

Grady Williams, Andrew Aldrich, and Evangelos A. Theodorou. Model predictive path integral control: From theory to parallel computation. Journal of Guidance, Control, and Dynamics, 40(2):344–357, 2017a.

Grady Williams, Nolan Wagener, Brian Goldfain, Paul Drews, James M. Rehg, Byron Boots, and Evangelos A. Theodorou. Information theoretic MPC for model-based reinforcement learning. In IEEE International Conference on Robotics and Automation (ICRA), pp. 1714–1721, 2017b.

Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, and Mingsheng Long. Transolver: A fast transformer solver for pdes on general geometries. In International Conference on Machine Learning (ICML), 2024.

Jianfeng Zhang. Backward stochastic diferential equations. In Backward Stochastic Diferential Equations: From Linear to Fully Nonlinear Theory, pp. 79–99. Springer, 2017.

## A Notation Table

Table 6 collects the shared symbols used across the paper.

Table 6: Notation, with the equation, assumption, or section where each symbol is defined.
<table><tr><td>Symbol</td><td>Definition</td><td>Meaning</td></tr><tr><td colspan="3">Control problem</td></tr><tr><td> $\xi , \ x$ </td><td>§2.1</td><td>starting point of the process or walk; conditioned as  $| \xi = x$  where the start is not otherwise fixed</td></tr><tr><td> $X _ { t } ^ { \xi , { \mathbf { u } } }$ </td><td>Eq. 1</td><td>controlled process at time t, from the prescribed start ξ under</td></tr><tr><td> $\mathbf { b }$ </td><td> $\operatorname { E q . 1 }$ </td><td>control u drift of the controlled system  $( \mathbf { b } = 0$  in the specialization)</td></tr><tr><td> $G , \ R$ </td><td>Eq. 1, Eq. 2</td><td>control matrix and control penalty  $( G = R = I$  in the special- ization)</td></tr><tr><td> $\Sigma$ </td><td>Eq. 1</td><td>diffusion tensor;  $\Sigma = \lambda I$  when  $G = R = I$  under matching</td></tr><tr><td> $q , \ c$ </td><td>Eq. 2, Eq. 3</td><td>state cost  $\textstyle q = { \frac { 1 } { 2 } } c ^ { 2 }$  , with  $c > 0$  the local slowness</td></tr><tr><td> $\phi , \ g$ </td><td>Eq. 2, §2.1</td><td>terminal cost and its transform  $g = e ^ { - \phi / \lambda } \in ( 0 , 1 ] ; g \equiv 1$  default</td></tr><tr><td> $\lambda$ </td><td>Ass. 1</td><td>temperature; sets the noise-control exchange rate</td></tr><tr><td>V, Ψ</td><td>Eq. 2, §2.1</td><td>value function and desirability  $\Psi = e ^ { - V / \lambda }$ </td></tr><tr><td colspan="3">Geometry and the elliptic problem</td></tr><tr><td> $\mathcal { G } , \Gamma _ { D }$ </td><td>Eq. 4</td><td>open goal region and its absorbing boundary  $\Gamma _ { D } = \partial \mathcal { G }$ </td></tr><tr><td> $\Gamma _ { N }$ </td><td>Eq. 4</td><td>obstacles ∂S and outer boundary, reflecting</td></tr><tr><td> $\kappa$ </td><td>Eq. 3</td><td>screening coefficient  $\kappa = c / \lambda$ </td></tr><tr><td> $\omega _ { d }$ </td><td> $\operatorname { E q . }$  15</td><td>total solid angle,  $2 \pi ^ { d / 2 } / \Gamma ( d / 2 )$ </td></tr><tr><td> $\alpha ( x )$ </td><td>Eq.8</td><td>fraction of solid angle subtended by Ω at x; 1 inside, − on ΓN</td></tr><tr><td colspan="3">The walk</td></tr><tr><td> $\xi _ { k }$ </td><td>Eq. 7</td><td>WoSt point after k steps, from  $\xi _ { 0 } = \xi ;$  random for  $k \geq 1$ </td></tr><tr><td> $r ( x ) , { \mathrm { S t } } ( x )$ </td><td>Eq. 6</td><td>radius and star centered at the current spatial location x;  $r _ { k } =$   $r ( \xi _ { k } )$ </td></tr><tr><td> $d _ { D } , \ d _ { \mathrm { s i l } }$ </td><td>Eq. 6</td><td>distance to  $\Gamma _ { D } ;$  distance to the nearest visibility silhouette</td></tr><tr><td> $\rho _ { k }$ </td><td>Eq. 7</td><td>distance travelled by step k;  $\rho _ { k } \le r _ { k }$ </td></tr><tr><td> $\sigma$ </td><td>§4.1</td><td>first exact goal-hit step inf  $\{ k : \xi _ { k } \in \Gamma _ { D } \}$  , possibly infinite; practical walks use shell stopping and a horizon</td></tr><tr><td> $W _ { k }$ </td><td> $\mathrm { E q . 2 5 }$ </td><td>step weight  $Q _ { d } ^ { \kappa } ( \rho _ { k } , r _ { k } )$ </td></tr><tr><td> $\gamma _ { d }$ </td><td>Eq. 22</td><td>full-sphere weight  $Q _ { d } ^ { \kappa } ( r , r )$ </td></tr><tr><td colspan="3">Labels and early stopping</td></tr><tr><td> $\widehat { \Psi }$ </td><td>Eq. 10</td><td>Monte Carlo estimate</td></tr><tr><td> $K , \ N , \ m$ </td><td>§4.2, Alg. 2</td><td>step budget; walks per query point; query points per gradient step</td></tr><tr><td colspan="3">Planning lattice</td></tr><tr><td> $h , \ X _ { h }$ </td><td>§4.4</td><td>lattice spacing and free grid cells</td></tr><tr><td> $L , \ e$ </td><td>§4.4</td><td>depth and number of edges committed per replan</td></tr><tr><td> $\mathcal { N } , \mathcal { R } _ { k }$ </td><td></td><td>feasible neighbors of a state; states reachable in at most k moves</td></tr><tr><td> $B _ { k }$ </td><td>§4.4 Eq. 37</td><td>largest desirability reachable within k moves</td></tr><tr><td colspan="3">Operator and training</td></tr><tr><td> $\tau$ </td><td> $\mathrm { E q . 5 }$ </td><td>exact solution operator  $a \mapsto \Psi ,$  which  $\mathcal { T } _ { \theta }$  approximates</td></tr><tr><td> $\mathcal { T } _ { { \boldsymbol { \theta } } , \mathrm { ~ } } \theta ^ { - }$ </td><td>Eq. 11, Alg. 2</td><td>the learned operator  $a \mapsto \Psi _ { \theta } ;$  its EMA target parameters</td></tr><tr><td colspan="3"> $\mathfrak { L }$ </td></tr><tr><td> $K e r n e l s$ </td><td></td><td></td></tr><tr><td>ν  $G ^ { \kappa , B }$ </td><td>Eq. 25</td><td>Bessel order  $\begin{array} { r l r } {  { \frac { d } { 2 } - 1 } } \end{array}$ </td></tr><tr><td></td><td>Eq. 18</td><td>screened Green&#x27;s function of the ball</td></tr><tr><td> $P ^ { \mathbb { R } ^ { d } }$ </td><td>Eq. 24</td><td>unscreened free-space Poisson kernel, the sampling density</td></tr></table>

## B Screened Poisson Kernels in d Dimensions

This appendix derives the kernels used by the estimator of Section 4.1 in general dimension. Throughout, $d \geq 2$ $\kappa > 0 .$ , and

$$
\nu : = \frac { d } { 2 } - 1 , \qquad r : = \| x - \mathbf { y } \| , \qquad \omega _ { d } = \frac { 2 \pi ^ { d / 2 } } { \Gamma ( d / 2 ) } .\tag{15}
$$

In Appendix B.1 the radial variable is the source separation $r = \| x - \mathbf { y } \|$ ; from Appendix B.2 onward r is the radius of the ball or star and $\rho : = \| x - \mathbf { y } \|$ denotes the source separation. We seek Green’s functions for the screened Laplacian

$$
\mathcal { L } _ { \kappa } : = - \Delta + \kappa ^ { 2 } , \qquad \mathcal { L } _ { \kappa } G ( x , \mathbf { y } ) = \delta ( x - \mathbf { y } ) .
$$

Away from the pole every case below solves the same radial ODE,

$$
u ^ { \prime \prime } ( r ) + \frac { d - 1 } { r } u ^ { \prime } ( r ) - \kappa ^ { 2 } u ( r ) = 0 , \qquad \mathrm { b a s i s } \ \{ r ^ { - \nu } I _ { \nu } ( \kappa r ) , r ^ { - \nu } K _ { \nu } ( \kappa r ) \} ,\tag{16}
$$

with $I _ { \nu }$ regular at the origin and $K _ { \nu }$ singular. Cases difer only in which combination the outer boundary condition selects; the overall constant is fixed once and for all by the flux jump at $r = 0$

## B.1 Free Space

In free space the radial solution of $\operatorname { E q } .$ . 16 that decays at infinity is the $K _ { \nu }$ branch, and the flux jump at the pole fixes its constant:

$$
G _ { \mathbb { R } ^ { d } } ( r ) = \frac { \kappa ^ { \nu } } { ( 2 \pi ) ^ { d / 2 } } \frac { K _ { \nu } ( \kappa r ) } { r ^ { \nu } }\tag{17}
$$

As a check, for $d = 2 \ ( \nu = 0 )$ this gives $G = K _ { 0 } ( \kappa r ) / 2 \pi$ , and for $d = 3 \ ( \nu = { \textstyle { \frac { 1 } { 2 } } } )$ , using $K _ { 1 / 2 } ( z ) = \sqrt { \pi / 2 z } e ^ { - z }$ gives the Yukawa potential $G = e ^ { - \kappa r } / 4 \pi r$

Radial derivative. The Neumann part of ∂St requires the normal derivative of Eq. 17. Using $\begin{array} { r l r } { \frac { d } { d z } \left[ z ^ { - \nu } K _ { \nu } ( z ) \right] = } \end{array}$ $- z ^ { - \nu } K _ { \nu + 1 } ( z )$

$$
\partial _ { r } G _ { \mathbb { R } ^ { d } } ( r ) = - \frac { \kappa ^ { \nu + 1 } } { ( 2 \pi ) ^ { d / 2 } } \frac { K _ { \nu + 1 } ( \kappa r ) } { r ^ { \nu } } < 0
$$

which is strictly negative for all $r > 0$ , so the kernel is monotonically decreasing in every dimension.

## B.2 The Ball: Dirichlet, Neumann, and Robin

Let $B _ { R } = \{ \| x \| < R \}$ with the pole at the center, as in every walk step. The solution is radial and takes the form

$$
G _ { B _ { R } } ( r ) = \frac { \kappa ^ { \nu } } { ( 2 \pi ) ^ { d / 2 } } \frac { 1 } { r ^ { \nu } } \Bigl [ K _ { \nu } ( \kappa r ) - A I _ { \nu } ( \kappa r ) \Bigr ] , \qquad 0 < r < R ,\tag{18}
$$

with the same prefactor as Eq. 17, since the flux jump at the origin is unchanged by the boundary. Only A depends on the boundary condition. Writing $z = \kappa r$ and using the scaled-derivative identities

$$
\frac { \mathrm { d } } { \mathrm { d } z } \big [ z ^ { - \nu } I _ { \nu } ( z ) \big ] = z ^ { - \nu } I _ { \nu + 1 } ( z ) , \qquad \frac { \mathrm { d } } { \mathrm { d } z } \big [ z ^ { - \nu } K _ { \nu } ( z ) \big ] = - z ^ { - \nu } K _ { \nu + 1 } ( z ) ,\tag{19}
$$

the radial derivative of Eq. 18 is proportional $\ b _ { \mathrm { ~ 0 ~ } } - K _ { \nu + 1 } ( z ) - A I _ { \nu + 1 } ( z )$ . The two cases we use follow:

$$
\mathrm { D i r i c h l e t } ~ G ( R ) = 0 :
$$

$$
A _ { D } = \frac { K _ { \nu } ( \kappa R ) } { I _ { \nu } ( \kappa R ) } ,\tag{20}
$$

$$
\mathrm { N e u m a n n } \partial _ { r } G ( R ) = 0 :
$$

$$
{ \cal A } _ { N } = - { \frac { K _ { \nu + 1 } ( \kappa R ) } { I _ { \nu + 1 } ( \kappa R ) } } .\tag{21}
$$

Note the order shift in $A _ { N } \colon$ since ${ \frac { \mathrm { d } } { \mathrm { d } z } } [ z ^ { - \nu } K _ { \nu } ] = - z ^ { - \nu } K _ { \nu + 1 }$ and likewise for $I _ { \nu } { \mathrm { . } }$ , imposing $\partial _ { r } { \cal G } ( R ) = 0$ on $r ^ { - \nu } [ K _ { \nu } - A I _ { \nu } ]$ produces the order- $( \nu + 1 )$ ratio rather than the naive $K _ { \nu } ^ { \prime } / I _ { \nu } ^ { \prime }$ , for which the derivative vanishes only when $\nu = 0$

## B.3 The Screened Poisson Kernel and the Step Weight

The walk uses the Poisson kernel, $P _ { B _ { R } } ^ { \kappa } ( \mathbf { 0 } , \mathbf { z } ) = - \partial _ { n _ { \mathbf { z } } } G _ { B _ { R } } ^ { D }$ for $\mathbf { z } \in \partial B _ { R }$ , which by radial symmetry is constant on the sphere. Its total mass has a closed form, and it is exactly the factor $\gamma _ { d }$ appearing in Eq. 22.

Proposition 1 (Total mass of the screened Poisson kernel). Let u solve $\Delta u - \kappa ^ { 2 } u = 0$ in $B _ { R }$ with $u \equiv 1$ on $\partial B _ { R }$ . Then

$$
u ( r ) = \Big ( { \frac { R } { r } } \Big ) ^ { \nu } { \frac { I _ { \nu } ( \kappa r ) } { I _ { \nu } ( \kappa R ) } } , \qquad \int _ { \partial B _ { R } } P _ { B _ { R } } ^ { \kappa } ( \mathbf { 0 } , \mathbf { z } ) \mathrm { d } S ( \mathbf { z } ) = u ( \mathbf { 0 } ) = \gamma _ { d } ( \kappa R ) ,
$$

where the full-sphere weight is

$$
\gamma _ { d } ( z ) : = \frac { ( z / 2 ) ^ { \nu } } { \Gamma ( \nu + 1 ) I _ { \nu } ( z ) } = Q _ { d } ^ { \kappa } ( r , r ) \big | _ { z = \kappa r } .\tag{22}
$$

Consequently

$$
P _ { B _ { R } } ^ { \kappa } ( \mathbf { 0 } , \mathbf { z } ) = \frac { \gamma _ { d } ( \kappa R ) } { \omega _ { d } R ^ { d - 1 } } = \frac { \Gamma ( d / 2 ) \gamma _ { d } ( \kappa R ) } { 2 \pi ^ { d / 2 } R ^ { d - 1 } } .\tag{23}
$$

Proof. Regularity at the origin selects the $I _ { \nu }$ branch of Eq. 16, and the stated normalization gives $u ( R ) = 1$ Taking $r  0$ with $I _ { \nu } ( z ) \sim ( z / 2 ) ^ { \nu } / \Gamma ( \nu + 1 )$ ,

$$
u ( \mathbf { 0 } ) = \frac { R ^ { \nu } ( \kappa / 2 ) ^ { \nu } } { \Gamma ( \nu + 1 ) I _ { \nu } ( \kappa R ) } = \frac { ( \kappa R / 2 ) ^ { \nu } } { \Gamma ( \nu + 1 ) I _ { \nu } ( \kappa R ) } = \gamma _ { d } ( \kappa R ) .
$$

The mass identity follows since $\begin{array} { r } { u ( \mathbf { 0 } ) = \int P ^ { \kappa } u \mathrm { d } S } \end{array}$ with $u \equiv 1$ on the boundary, and Eq. 23 follows from constancy on the sphere. □

Proposition 1 is the $\rho = r$ case of the star-step weight Eq. 25, and Appendix C.5 shows the general weight reduces to it. It also gives the two asymptotic regimes. Expanding $I _ { \nu }$ and using $4 ( \nu + 1 ) = 2 d$

$$
\gamma _ { d } ( z ) = [ 1 + \frac { z ^ { 2 } } { 2 d } + O ( z ^ { 4 } ) ] ^ { - 1 } = 1 - \frac { z ^ { 2 } } { 2 d } + O ( z ^ { 4 } ) , \qquad \gamma _ { d } ( z ) \sim \frac { ( z / 2 ) ^ { \nu } \sqrt { 2 \pi z } } { \Gamma ( \nu + 1 ) } e ^ { - z } ( z  \infty ) ,
$$

so a short step is nearly weight-preserving with deficit $\kappa ^ { 2 } R ^ { 2 } / 2 d .$ while a step much longer than the screening length is damped exponentially; hence steps beyond a few screening lengths contribute negligibly.

## B.4 The Screened Ball Poisson Kernel and the Star-Step Weight

Appendix B.3 treated the Poisson kernel at the center of a ball, which corresponds to a full-length sphere step. A star step may instead terminate against $\Gamma _ { N }$ at a distance $\rho < r ,$ so we need the kernel at interior points of the ball.

Following Sawhney et al. (2023), the screened ball Green’s functions are

$$
G _ { 2 } ^ { \kappa , B } ( x , { \bf y } ) = \frac { 1 } { 2 \pi } \left[ K _ { 0 } ( \kappa \rho ) - I _ { 0 } ( \kappa \rho ) \frac { K _ { 0 } ( \kappa r ) } { I _ { 0 } ( \kappa r ) } \right] , \qquad G _ { 3 } ^ { \kappa , B } ( x , { \bf y } ) = \frac { 1 } { 4 \pi } \frac { \sinh \left( ( r - \rho ) \kappa \right) } { \rho \sinh ( \kappa r ) } ,
$$

with $\rho = \| \mathbf { y } - x \|$ , which agree with the general-d expression Eq. 18 at $d = 2 , 3$ . Both vanish at $\rho = r _ { \mathrm { { ; } } }$ which is the cancellation that removes the unknown-flux term on the spherical part of the boundary.

The corresponding Poisson kernels factor as a scalar times the free-space kernel,

$$
P _ { d } ^ { \kappa , B } ( x , \mathbf { z } ) = Q _ { d } ^ { \kappa } ( \rho , r ) P ^ { \mathbb { R } ^ { d } } ( x , \mathbf { z } ) , \qquad P ^ { \mathbb { R } ^ { d } } ( x , \mathbf { z } ) = \frac { n _ { \mathbf { z } } \cdot ( \mathbf { z } - x ) } { \omega _ { d } \rho ^ { d } } ,\tag{24}
$$

with $Q _ { d } ^ { \kappa }$ given for all d by Eq. 25.

Lemma 1 (General-dimension step weight). For every d $\geq 2$ and $0 < \rho \le r$

$$
Q _ { d } ^ { \kappa } ( \rho , r ) : = \frac { \partial _ { \rho } G ^ { \kappa , B } } { \partial _ { \rho } G ^ { B } } = \frac { 2 } { \Gamma ( \nu + 1 ) } \Big ( \frac { \kappa \rho } { 2 } \Big ) ^ { \nu + 1 } \left[ K _ { \nu + 1 } ( \kappa \rho ) + \frac { K _ { \nu } ( \kappa r ) } { I _ { \nu } ( \kappa r ) } I _ { \nu + 1 } ( \kappa \rho ) \right] , \qquad \nu = \frac { d } { 2 } - 1 ,\tag{25}
$$

and $Q _ { d } ^ { \kappa } ( r , r ) = \gamma _ { d } ( \kappa r )$

Proof. Write $z = \kappa \rho$ and $\alpha = K _ { \nu } ( \kappa r ) / I _ { \nu } ( \kappa r )$ , so that by Eq. 18 and Eq. 20,

$$
G ^ { \kappa , B } = \kappa ^ { 2 \nu } ( 2 \pi ) ^ { - d / 2 } \bigl [ z ^ { - \nu } K _ { \nu } ( z ) - \alpha z ^ { - \nu } I _ { \nu } ( z ) \bigr ] .
$$

The scaled-derivative identities Eq. 19 give

$$
\partial _ { \rho } G ^ { \kappa , B } = - \frac { \kappa ^ { \nu + 1 } } { ( 2 \pi ) ^ { d / 2 } } \rho ^ { - \nu } \bigl [ K _ { \nu + 1 } ( \kappa \rho ) + \alpha I _ { \nu + 1 } ( \kappa \rho ) \bigr ] ,
$$

using $2 \nu + 1 - \nu = \nu + 1$ for the power of κ. For $d > 2$ , the denominator has $G ^ { B } = c _ { d } ( \rho ^ { 2 - d } - r ^ { 2 - d } )$ with $c _ { d } = \Gamma ( \nu ) / ( 4 \pi ^ { \nu + 1 } )$ , hence $\partial _ { \rho } G ^ { B } \stackrel {  } { = } - 2 \nu c _ { d } \rho ^ { - 2 \nu - 1 } = - \Gamma ( \nu + 1 ) ( 2 \pi ^ { \nu + 1 } ) ^ { - 1 } \rho ^ { - 2 \nu - 1 }$ , where 2νΓ $( \nu ) = 2 \Gamma ( \nu + 1 )$ Dividing the preceding display by this and collecting the powers of 2 and π gives Eq. 25. Setting $\rho = r$ makes the bracket $\left[ I _ { \nu } K _ { \nu + 1 } + I _ { \nu + 1 } K _ { \nu } \right] / I _ { \nu } ,$ , and the Bessel identity $I _ { \nu } ( z ) K _ { \nu + 1 } ( z ) + I _ { \nu + 1 } ( z ) K _ { \nu } ( z ) = 1 / z$ reduces it to $1 / ( z I _ { \nu } ( z ) )$ , leaving $( z / 2 ) ^ { \nu } / ( \Gamma ( \nu \dot { + } 1 ) I _ { \nu } ( z ) ) = \gamma _ { d } ( z )$ . For $d = 2$ , use $\begin{array} { r } { G ^ { B } ( \rho ) = ( 2 \pi ) ^ { - 1 } \log ( r / \rho ) } \end{array}$ and $\partial _ { \rho } G ^ { B } = - 1 / ( 2 \pi \rho )$ directly; the same ratio follows. The sphere reduction follows from the standard Wronskian $I _ { 0 } K _ { 1 } + I _ { 1 } K _ { 0 } = 1 / z$ We verified Eq. 25 against the tabulated $d = 2 ,$ 3 forms and the reduction against $\gamma _ { d }$ at $d \in \{ 2 , 3 , 4 , 5 , 7 , 1 0 \}$ to fourteen digits. □

Specializing Eq. 25 to the two dimensions tabulated by Sawhney et al. (2023),

$$
\begin{array} { r } { Q _ { 2 } ^ { \kappa } ( \rho , r ) = \left[ K _ { 1 } ( \kappa \rho ) + I _ { 1 } ( \kappa \rho ) \frac { K _ { 0 } ( \kappa r ) } { I _ { 0 } ( \kappa r ) } \right] \kappa \rho , } \end{array}\tag{26}
$$

$$
\begin{array} { r } { Q _ { 3 } ^ { \kappa } ( \rho , r ) = e ^ { - \kappa \rho } ( \kappa \rho + 1 ) + \left( \kappa \rho \cosh ( \kappa \rho ) - \sinh ( \kappa \rho ) \right) \frac { e ^ { - \kappa r } } { \sinh ( \kappa r ) } . } \end{array}\tag{27}
$$

Table 7 gives representative values. Because of this factorization, the estimator is inexpensive: the geometric part $P ^ { \mathbb { R } ^ { d } }$ is exactly the signed solid angle subtended by ∂St at z, so sampling directions uniformly gives unit importance weights, and screening enters only through the scalar $Q _ { d } ^ { \kappa }$

Table 7: Star-step weight $Q _ { 3 } ^ { \kappa } ( \rho , r )$ at $r = 1$ . The weight is 1 for a zero-length step and decreases with the realized hit distance $\rho ,$ reaching $\gamma _ { 3 } ( \kappa r ) = \kappa r / \sinh ( \kappa r )$ at $\rho = r$ . A step cut short by an obstacle loses less weight than a full sphere step.
<table><tr><td> $\rho / r$ </td><td>0</td><td>1/6</td><td> $\overline { { 1 / 3 } }$ </td><td> $\overline { { 1 / 2 } }$ </td><td> $\overline { { 2 / 3 } }$ </td><td> $\overline { { 5 / 6 } }$ </td><td>1</td></tr><tr><td> $\overline { { \kappa r = 1 } }$ </td><td>1.0000</td><td>0.9880</td><td>0.9593</td><td>0.9232</td><td>0.8880</td><td>0.8614</td><td>0.8509</td></tr><tr><td> $\kappa r = 4$ </td><td>1.0000</td><td>0.8558</td><td>0.6157</td><td>0.4086</td><td>0.2629</td><td>0.1766</td><td>0.1466</td></tr></table>

## B.5 Why a Ball Kernel Is Admissible on a Star Domain

The kernels of Appendices B.1–B.4 are constructed for free space and for a ball, yet Eq. 28 integrates them over ∂St, which is not a sphere. The following identity justifies this.

Proposition 2 (Admissibility). Let $A \subseteq \Omega$ be a bounded Lipschitz domain with $x \in A$ , let Ψ solve $\mathcal { L } _ { \kappa } \Psi = 0$ on $A ,$ and let G be any function with $( - \Delta + \kappa ^ { 2 } ) G ( x , \cdot ) = \delta _ { x }$ on a neighborhood of ${ \overline { { A } } } ,$ so that x is its only singularity in A. Then with $P : = - \partial _ { n } G$ and n the outward normal,

$$
\Psi ( x ) = \int _ { \partial A } \Big [ P ( x , { \bf z } ) \Psi ( { \bf z } ) + G ( x , { \bf z } ) \partial _ { n } \Psi ( { \bf z } ) \Big ] \mathrm { d } S ( { \bf z } ) .\tag{28}
$$

Proof. Green’s second identity on A for the pair $( \Psi , G )$ with the operator $\mathcal { L } _ { \kappa }$ gives

$$
\int _ { A } \left( \Psi \mathcal { L } _ { \kappa } G - G \mathcal { L } _ { \kappa } \Psi \right) = \int _ { \partial A } \left( G \partial _ { n } \Psi - \Psi \partial _ { n } G \right)
$$

the zeroth-order terms $\kappa ^ { 2 } \Psi G$ canceling identically between the two products. The second integrand on the left vanishes by hypothesis, and the first collapses to $\Psi ( x )$ by the defining property of G. Rearranging gives Eq. 28. □

The chain’s one-step identity is the case $G = G ^ { \kappa , B }$ , the screened Dirichlet Green’s function of the ball $B = B ( x , r )$ , applied on an admissible star $A = \operatorname { S t } ( x )$ . Since $G ^ { \kappa , B }$ vanishes on the spherical part of $\partial A$ and $\partial _ { n } \Psi = 0$ on the reflecting part, the flux term drops and $\begin{array} { r } { \Psi ( x ) = \int _ { \partial A } P ^ { \kappa , B } ( x , \mathbf { z } ) \Psi ( \mathbf { z } ) \mathrm { d } S ( \mathbf { z } ) } \end{array}$ with $\bar { P ^ { \kappa , B } } = - \partial _ { n } G ^ { \kappa , B }$ At a smooth boundary pole $x \in \Gamma _ { N }$ , the same steps on A minus a half-ball of radius η about $x ,$ with $\eta  0 .$ , give the identity with $\Psi ( x )$ replaced by ${ \frac { 1 } { 2 } } \Psi ( x )$ , because the small hemisphere carries half of the singular flux; this is the factor α in $\operatorname { E q . 8 }$ . The direction law is the solid-angle density $P ^ { \bar { B } } \mathrm { d } S / \alpha$ with $P ^ { B } = - \partial _ { n } G ^ { B }$ , and the weight is the flux ratio $Q _ { d } ^ { \kappa } = P ^ { \kappa , B } / P ^ { B }$ of Eq. 34, so $\begin{array} { r } { \mathbb { E } [ W _ { k } \Psi ( \xi _ { k + 1 } ) \mid \xi _ { k } = x ] = \alpha ( x ) ^ { - 1 } \int _ { \partial A } P ^ { \kappa , B } \Psi \mathrm { d } S = \Psi ( x ) } \end{array}$ , the one-step property of Section 4.1. Multiplying by the earlier weights gives the martingale property before σ; the stopped sequence is constant afterwards

The hypotheses require neither that G be the Green’s function of A nor that it satisfy any boundary condition on ∂A. All that is needed is a fundamental solution whose singularity inside A is the single pole at x. Since $\operatorname { S t } ( x , r ) \subseteq B ( x , r )$ , the ball kernel $G ^ { \kappa , B }$ has that property, and so do the free-space kernel and the ball kernel of any radius $r ^ { \prime } \geq r .$ . Each of these kernels gives a valid representation of $\Psi ( x )$ , so we must choose among them.

The choice is governed by the unknown flux $\partial _ { n } \Psi$ in Eq. 28. On ∂St that flux is zero on $\partial \mathrm { S t } _ { N } \subseteq \Gamma _ { N }$ but unknown on $\partial \mathrm { S t } _ { B } \subseteq \partial B ( x , r )$ , which lies in the interior of Ω where no boundary condition applies. A usable identity therefore needs a kernel that vanishes on $\partial \mathrm { S t } _ { B }$ while remaining finite on $\partial \mathrm { S t } _ { N }$ , and that is exactly the ball’s Dirichlet kernel: $G ^ { \kappa , B } ( x , \cdot ) \equiv 0 \mathrm { o n } \partial B ( x , r ) \supseteq \partial \mathrm { S t } _ { B }$ , and it is finite and nonzero at the interior points constituting $\partial \mathrm { S t } _ { N }$ . The free-space kernel is admissible but unsuitable here: $G ^ { \kappa , \mathbb { R } ^ { d } } > 0$ on $\partial \mathrm { S t } _ { B } ,$ so the unknown flux term does not vanish.

The walk nonetheless samples directions by the free-space solid angle. For the Laplace operator the two Poisson kernels agree on all of ∂St, because the boundary corrector $G ^ { B } - G ^ { \mathbb { R } ^ { d } }$ for a centered pole is the constant radial homogeneous solution and so has zero normal derivative; directional sampling then importance-samples the kernel exactly and the step weight is 1. Screening replaces that constant by the $I _ { \nu }$ branch of Eq. 16, and the resulting ρ-dependent corrector is precisely the second term of Eq. 25. This explains why $Q _ { d } ^ { \kappa }$ depends on the star radius r and not only on the hit distance $\rho \colon$ the ρ-dependence comes from the free-space fundamental solution, and the r-dependence comes from the ball corrector, whose normal derivative is nonzero only when $\kappa > 0$

Proposition 2 is exact, so the ball-on-star transfer contributes no error of its own. One further term is specific to the identity: for nonzero Neumann data the second integrand of Eq. 28 requires $G ^ { \kappa , B }$ at interior points of $B ,$ which is available in closed form but is estimated with a separate sample, since reusing the walk direction gives an estimator that is zero on flat boundaries and high-variance elsewhere (Sawhney et al., 2023). With $h \equiv 0$ that term drops entirely, which is the case throughout this paper.

## B.6 Dimension-Specific Forms and the Laplace Limit

Table 8: Screened Poisson kernels by dimension. All entries follow from Eq. 17, Eq. 18, and Eqs. 20–21 with $\nu = d / 2 - 1$
<table><tr><td>d</td><td> $G _ { \mathbb { R } ^ { d } } ( r )$ </td><td></td><td> $G _ { B _ { R } } ^ { D } ( r )$ </td><td> $\gamma _ { d } ( z )$ </td></tr><tr><td rowspan="2">2</td><td> $K _ { 0 } ( \kappa r )$ </td><td></td><td> $\frac { 1 } { 2 \pi } \left[ K _ { 0 } ( \kappa r ) - \frac { K _ { 0 } ( \kappa R ) } { I _ { 0 } ( \kappa R ) } I _ { 0 } ( \kappa r ) \right]$ </td><td>1</td></tr><tr><td>2π</td><td></td><td></td><td> $\overline { { I _ { 0 } ( z ) } }$ </td></tr><tr><td>3</td><td> $e ^ { - \kappa r }$  4πr</td><td></td><td> $\frac { 1 } { 4 \pi r } \left[ e ^ { - \kappa r } - e ^ { - \kappa R } \frac { \sinh ( \kappa r ) } { \sinh ( \kappa R ) } \right]$ </td><td>z sinh z</td></tr><tr><td rowspan="2">general</td><td> $\kappa ^ { \nu } \qquad K _ { \nu } ( \kappa r )$ </td><td></td><td></td><td> $( z / 2 ) ^ { \nu }$ </td></tr><tr><td> $\overline { { ( 2 \pi ) ^ { d / 2 } } } \overline { { r ^ { \nu } } }$ </td><td></td><td> $\frac { \kappa ^ { \nu } } { ( 2 \pi ) ^ { d / 2 } r ^ { \nu } } \left[ K _ { \nu } ( \kappa r ) - \frac { K _ { \nu } ( \kappa R ) } { I _ { \nu } ( \kappa R ) } I _ { \nu } ( \kappa r ) \right]$ </td><td> $\overline { { \Gamma ( \nu + 1 ) I _ { \nu } ( z ) } }$ </td></tr></table>

Table 8 collects the dimension-specific forms. For $d > 2 , K _ { \nu } ( z ) \sim 2 ^ { \nu - 1 } \Gamma ( \nu ) z ^ { - \nu }$ gives the unscreened limit

$$
G _ { \mathbb { R } ^ { d } } ( r ) \longrightarrow \frac { \Gamma ( d / 2 - 1 ) } { 4 \pi ^ { d / 2 } } r ^ { 2 - d } , \gamma _ { d } ( \kappa R ) \longrightarrow 1 ,
$$

which is $1 / ( 4 \pi r )$ in three dimensions. In two dimensions the free-space Green’s function diverges by an additive constant as $\kappa  0 \mathrm { { ; } }$ ; subtracting that constant gives $- ( 2 \pi ) ^ { - 1 }$ log r. In the Dirichlet ball kernel the constants cancel, giving $( 2 \pi ) ^ { - 1 } \log ( R / r )$ . In all dimensions $Q _ { d } ^ { \kappa } ( \rho , r ) \to 1$ and $\gamma _ { d }  1$ , recovering the unscreened weights. A purely Neumann unscreened problem needs a separate compatibility and normalization condition.

## C Proofs

## C.1 Standing Regularity Assumptions

Assumption 2 (Regularity). The free domain Ω is bounded and connected with $C ^ { 2 , \alpha }$ boundary, for $0 < \alpha < 1$ Its Dirichlet and Neumann sets are disjoint unions of boundary components, each with positive surface measure. Hence they are separated by a positive distance. This excludes Dirichlet–Neumann junctions, near which the solution can have singular gradients. The goal data satisfies $g \in C ^ { 2 , \alpha } ( \Gamma _ { D } )$ and $0 < g _ { \mathrm { m i n } } \le g \le 1$ . The cost satisfies $c \in C ^ { \alpha } ( \overline { { \Omega } } )$ and $0 < c _ { \operatorname* { m i n } } \le c \le c _ { \operatorname* { m a x } } .$ . The noise parameter $\lambda > 0$ is fixed for each problem.

Under Assumption 2, the mixed elliptic problem has a unique $\Psi \in C ^ { 2 , \alpha } ( \overline { { \Omega } } )$ (Gilbarg & Trudinger, 2001). These regularity statements concern the ideal smooth domain. Voxel edges and corners are not covered by this assumption; the smooth-boundary factor $\alpha = 1 / 2$ and global gradient bounds do not automatically extend to them. The exact PDE identities for the chain additionally require constant screening and admissible stars; regularity does not guarantee that the radius-clamped sampler is exact. Write $\mathcal { L } _ { \kappa } = - \Delta + \kappa ^ { 2 }$

## C.2 Exact Linearization

Where V is $C ^ { 2 }$ it satisfies the interior stationary HJB equation

$$
\begin{array} { r } { 0 = \displaystyle \operatorname* { m i n } _ { \mathbf { u } } \left\{ \nabla V ^ { \top } ( \mathbf { b } + G \mathbf { u } ) + q + \frac { 1 } { 2 } { \mathbf u } ^ { \top } R \mathbf { u } \right\} + \frac { 1 } { 2 } \operatorname { t r } \left( \Sigma \nabla ^ { 2 } V \right) , } \end{array}
$$

whose inner minimization is attained at $\mathbf { u } ^ { \star } = - R ^ { - 1 } G ^ { \top } \nabla V$ . Substituting that minimizer leaves a quadratic nonlinearity in $\nabla V$

$$
\begin{array} { r } { 0 = \nabla V ^ { \top } \mathbf b - \frac { 1 } { 2 } \nabla V ^ { \top } G R ^ { - 1 } G ^ { \top } \nabla V + q + \frac { 1 } { 2 } \operatorname { t r } \big ( \Sigma \nabla ^ { 2 } V \big ) , } \end{array}\tag{29}
$$

and this term cancels under the desirability transform.

Lemma 2 (Cole–Hopf linearization is exact precisely under matching). Let $V \in C ^ { 2 }$ satisfy Eq. 29 and set $\Psi = e ^ { - V / \lambda }$ . Then

$$
\begin{array} { r } { \frac { 1 } { 2 } \operatorname { t r } \big ( \Sigma \nabla ^ { 2 } \Psi \big ) + \mathbf { b } ^ { \top } \nabla \Psi - \frac { q } { \lambda } \Psi = \frac { 1 } { 2 \Psi } \nabla \Psi ^ { \top } \Big [ \Sigma - \lambda G R ^ { - 1 } G ^ { \top } \Big ] \nabla \Psi . } \end{array}\tag{30}
$$

In particular the right side vanishes for every Ψ with $\nabla \Psi \ne 0$ if and only if $\Sigma = \lambda G R ^ { - 1 } G ^ { \top }$ as quadratic forms, which is Assumption 1.

Proof. From $\Psi = e ^ { - V / \lambda }$ we get $\nabla V = - \lambda \nabla \Psi / \Psi$ and $\begin{array} { r } { \nabla ^ { 2 } V = - \lambda \frac { \nabla ^ { 2 } \Psi } { \Psi } + \lambda \frac { \nabla \Psi \otimes \nabla \Psi } { \Psi ^ { 2 } } } \end{array}$ Substitute into Eq. 29 term by term. The difusion term becomes

$$
{ \textstyle \frac { 1 } { 2 } } \operatorname { t r } ( \Sigma \nabla ^ { 2 } V ) = - { \frac { \lambda } { 2 } } { \frac { \operatorname { t r } ( \Sigma \nabla ^ { 2 } \Psi ) } { \Psi } } + { \frac { \lambda } { 2 } } { \frac { \nabla \Psi ^ { \top } \Sigma \nabla \Psi } { \Psi ^ { 2 } } }
$$

and the quadratic control term becomes

$$
- { \textstyle \frac { 1 } { 2 } } \nabla V ^ { \top } G R ^ { - 1 } G ^ { \top } \nabla V = - \frac { \lambda ^ { 2 } } { 2 } \frac { \nabla \Psi ^ { \top } G R ^ { - 1 } G ^ { \top } \nabla \Psi } { \Psi ^ { 2 } } .
$$

Adding the two quadratic contributions gives

$$
\frac { \lambda } { 2 \Psi ^ { 2 } } \nabla \Psi ^ { \top } \big [ \Sigma - \lambda G R ^ { - 1 } G ^ { \top } \big ] \nabla \Psi ,\tag{31}
$$

which is $\operatorname { E q } .$ . 30 up to the stated factor. The remaining terms are $- \lambda { \bf b } ^ { \top } \nabla \Psi / \Psi + q \mathrm { ~ } \mathrm { a n d ~ } - \frac { \lambda } { 2 } \mathrm { t r } \big ( \Sigma \nabla ^ { 2 } \Psi \big ) / \Psi$ multiplying the whole equation by $- \Psi / \lambda$ yields the stationary backward Kolmogorov equation

$$
\textstyle { \frac { 1 } { 2 } } \operatorname { t r } \left( \Sigma \nabla ^ { 2 } \Psi \right) + \mathbf { b } ^ { \top } \nabla \Psi - { \frac { q } { \lambda } } \Psi = 0\tag{32}
$$

exactly when Eq. 31 vanishes.

For universal cancellation of the transformed diferential expression, put $E = \Sigma - \lambda G R ^ { - 1 } G ^ { \top }$ . The symmetric quadratic form $v ^ { \top } E v$ vanishes for every v if and only if $E = 0$ . Smooth positive test fields can have an arbitrary gradient at a point, so matching is necessary for this operator identity. For a particular solution Ψ, the residual can vanish even when matching fails. □

For the fully actuated reflected model under Assumption 2, let Ψ solve the mixed problem and set $V = - \lambda$ log Ψ and $\mathbf { u } ^ { * } = - \nabla V$ . Assume admissible controls, including $\mathbf { u } ^ { * }$ , give well-defined dynamics with finite expected goal-hitting time and control energy. Completing the HJB square and applying Itˆo’s formula up to $\tau _ { D } \wedge T$ , then taking $T \to \infty .$ , yields

$$
J ( x , \mathbf { u } ) = V ( x ) + \frac { 1 } { 2 } \mathbb { E } _ { x } \int _ { 0 } ^ { \tau _ { D } } \| \mathbf { u } _ { t } - \mathbf { u } ^ { * } ( X _ { t } ) \| ^ { 2 } \mathrm { d } t .\tag{33}
$$

The Neumann condition removes the reflection term; bounded $V$ and $V = \phi$ at the goal justify the stopped limit.   
Thus $J ( x , \mathbf { u } ) \geq V ( x )$ , with equality for $\mathbf { u } ^ { * }$ , proving optimality.

## C.3 Positivity and Bounded Log-Desirability

Part (ii) below fixes the scale on which log Ψ varies. For constant c the screening length is $\lambda / c .$ but additional length scales arise from the domain size, goal size, and obstacle geometry, so exponential decay with distance holds only asymptotically.

Proposition 3 (Positivity and range). Under Assumption 2:

(i) $0 < \Psi \leq 1$ on ${ \overline { { \Omega } } } ,$ with $\Psi _ { \mathrm { m i n } } = \operatorname* { m i n } \Psi > 0 ;$

(ii) for the fixed annulus $a < | x | < D , 0 < a < D$ , constant κ, $\Psi ( a ) = 1$ , and $\Psi ^ { \prime } ( D ) = 0$ , the log range is $- \log \Psi ( D ) = \kappa ( D - a ) + O ( 1 ) ~ a s ~ \kappa \to \infty .$

Proof. For (i), the maximum principle excludes an interior maximum above one or a nonpositive interior minimum. At a reflecting boundary, the Hopf boundary lemma (Gilbarg & Trudinger, 2001) excludes either extremum unless the solution is constant. A constant violating the bounds contradicts the positive Dirichlet data. Continuity and compactness then give a strictly positive minimum.

For (ii), put $\nu = d / 2 - 1$ . The radial solution is

$$
\Psi ( r ) = \frac { r ^ { - \nu } [ K _ { \nu } ( \kappa r ) + A I _ { \nu } ( \kappa r ) ] } { a ^ { - \nu } [ K _ { \nu } ( \kappa a ) + A I _ { \nu } ( \kappa a ) ] } , \qquad A = \frac { K _ { \nu + 1 } ( \kappa D ) } { I _ { \nu + 1 } ( \kappa D ) } .
$$

Integrating $( r ^ { d - 1 } \Psi ^ { \prime } ) ^ { \prime } = \kappa ^ { 2 } r ^ { d - 1 } \Psi$ backward from $\Psi ^ { \prime } ( D ) = 0$ shows $\Psi ^ { \prime } < 0$ for $a < r < D$ . Thus the range is − log $\Psi ( D )$ . At fixed $a , D .$ , the large-argument Bessel asymptotics give $\Psi ( D ) = 2 ( a / D ) ^ { ( d - 1 ) / 2 } e ^ { - \kappa ( D - a ) } \bar { ( 1 + }$ $O ( \kappa ^ { - 1 } ) )$ ), which proves (ii). □

## C.4 Why We Use Reflection at Obstacles

Reflection lets the process remain in the free domain and continue toward the goal after reaching an obstacle boundary. Its corresponding condition is $\partial _ { n } \Psi = 0$ , as in Eq. 4 and the reflecting walks of Sawhney et al. (2023). This choice also keeps log Ψ, and so the control, finite up to the boundary under Assumption 2.

Zero Dirichlet data produces a logarithmic singularity. Suppose instead that obstacles are absorbing, with zero terminal desirability. Denote the resulting solution by $\Psi _ { \mathrm { a b s } }$ , retaining positive goal data and reflection at the outer workspace boundary. Then

$$
\Psi _ { \mathrm { a b s } } = 0 \ \mathrm { o n } \ \partial \mathcal { S } , \qquad \log \Psi _ { \mathrm { a b s } } ( x ) \to - \infty , \qquad V _ { \mathrm { a b s } } ( x ) = - \lambda \log \Psi _ { \mathrm { a b s } } ( x ) \to + \infty
$$

as x approaches an obstacle from the free domain. Thus zero desirability represents an infinite collision penalty. The singularity is in the logarithmic representation and value function; the linear PDE solution $\Psi _ { \mathrm { a b s } }$ can remain smooth.

The control also becomes singular at a smooth obstacle surface. With the smoothness and separation assumptions of Assumption 2, now replacing the obstacle condition by zero Dirichlet data, fix $z \in \partial S$ and write $x = z - s n ( z )$ , where $s > 0$ measures distance into the free domain. The Hopf boundary lemma gives $a _ { z } = - \partial _ { n } \Psi _ { \mathrm { a b s } } ( z ) > 0$ . Taylor expansion and $\mathbf { u ^ { \star } } = - \nabla V$ , for $G = R = I .$ , yield

$$
\begin{array} { l } { { \Psi _ { \mathrm { a b s } } ( z - s n ( z ) ) = a _ { z } s + O ( s ^ { 2 } ) , } } \\ { { V _ { \mathrm { a b s } } ( z - s n ( z ) ) = - \lambda \log ( a _ { z } s ) + O ( s ) , } } \\ { { \displaystyle \mathbf { u } _ { \mathrm { a b s } } ^ { \star } ( z - s n ( z ) ) = - \frac { \lambda } { s } n ( z ) + O ( 1 ) , \qquad s \downarrow 0 . } } \end{array}
$$

The leading control points away from the obstacle and grows as inverse distance. A bounded log-output mode cannot approximate this target uniformly all the way to the obstacle surface. Clipping the desirability to a positive floor removes the divergence numerically but introduces a boundary approximation.

Reflection preserves a bounded target. For the reflected problem, Proposition 3 gives $\Psi _ { \mathrm { m i n } } > 0$ on the closed free domain. Consequently V and log Ψ are bounded, $\partial _ { n } V = 0 ;$ , and $\| \nabla V \| _ { \infty } \leq \lambda \| \nabla \Psi \| _ { \infty } / \Psi _ { \mathrm { m i n } } < \infty$ under the stated regularity assumptions. The reflected dynamics enforce the boundary constraint; the Neumann condition does not impose a positive clearance from obstacles.

We therefore use reflection because it permits continued goal seeking with a bounded log target; other formulations need not use it. Strictly positive Dirichlet obstacle data $\Psi = \bar { e } ^ { - \phi _ { \mathrm { o b s } } / \lambda }$ with finite $\phi _ { \mathrm { o b s } }$ avoids this logarithmic singularity, but models termination at an obstacle with a finite penalty. The singularity claim applies specifically to zero Dirichlet data for Ψ, not to Dirichlet conditions in general.

## C.5 The Chain Kernel

Throughout, $G ^ { B }$ and $G ^ { \kappa , B }$ denote the unscreened and screened Dirichlet Green’s functions of $B = B ( x , r )$ with pole at the center; both are radial in $\rho = \| \mathbf { y } - x \|$ . By Lemma 1, the step weight is the flux ratio

$$
Q _ { d } ^ { \kappa } ( \rho , r ) = \frac { \partial _ { \rho } G ^ { \kappa , B } } { \partial _ { \rho } G ^ { B } } .\tag{34}
$$

Both Green’s functions are radial about the center, so the outward normal derivative at z factors as $\partial _ { n } G =$ $( \partial _ { \rho } G ) ( n _ { \mathbf { z } } \cdot \hat { e } _ { \rho } )$ and the geometric factor cancels in the ratio, which is why Eq. 34 is also the ratio of the two Poisson kernels in Eq. 24.

For $d \ge 2 , \kappa > 0$ and $0 < \rho \le r$ the step weight is bounded on both sides,

$$
\gamma _ { d } ( \kappa r ) \leq Q _ { d } ^ { \kappa } ( \rho , r ) < 1 ,\tag{35}
$$

with $\gamma _ { d }$ the full-sphere weight of Eq. 22. By the upper bound, the weighted chain of Section 4.1 remains bounded; the lower bound gives $Q _ { d } ^ { \kappa } \geq \gamma _ { d } ( \kappa r _ { \operatorname* { m a x } } )$ when the star radius is at most $r _ { \mathrm { m a x } }$ , which yields the replica bounds used in Appendix C.7.

Bounds on $Q _ { d } ^ { \kappa }$ . Positivity is immediate from Eq. 25, since $K _ { \nu + 1 } , I _ { \nu + 1 }$ and $K _ { \nu } / I _ { \nu }$ are all positive. For the upper bound, set $\begin{array} { r } { \dot { D : = } G ^ { B } - \dot { G } ^ { \kappa , B } } \end{array}$ , which is radial and vanishes at $\rho = r$ . Subtracting the two defining equations cancels the point sources and leaves $- \Delta D = \kappa ^ { 2 } G ^ { \kappa , B }$ in B, so in flux form, the constant vanishing because the right side is locally integrable at the origin,

$$
- \rho ^ { d - 1 } \partial _ { \rho } D = \kappa ^ { 2 } \int _ { 0 } ^ { \rho } t ^ { d - 1 } G ^ { \kappa , B } ( t ) \mathrm { d } t , \qquad - \rho ^ { d - 1 } \partial _ { \rho } G ^ { B } = \frac { 1 } { \omega _ { d } } .
$$

Dividing the first by the second gives the weight-deficit identity

$$
1 - Q _ { d } ^ { \kappa } ( \rho , r ) = \frac { \partial _ { \rho } D } { \partial _ { \rho } G ^ { B } } = \kappa ^ { 2 } \int _ { B ( x , \rho ) } G ^ { \kappa , B } ( x , \mathbf { y } ) \mathrm { d } \mathbf { y } ,
$$

The integrand is strictly positive on the interior of $B ,$ so the right side is strictly positive for $\rho > 0$ and $Q _ { d } ^ { \kappa } < 1$ For the lower bound, $B ( x , \rho ) \subseteq B ( x , r )$ and the integrand is nonnegative, so the deficit is at most $\kappa ^ { 2 } \int _ { B } G ^ { \kappa , B }$ integrating $( - \Delta + \kappa ^ { 2 } ) G ^ { \kappa , B } = \delta _ { x }$ over B gives $\begin{array} { r } { \kappa ^ { 2 } \int _ { B } G ^ { \kappa , B } = 1 - \bar { \int } _ { \partial B } P _ { B } ^ { \kappa } \mathrm { d } S = 1 - \gamma _ { d } ( \kappa r ) } \end{array}$ by Proposition 1. Hence $\gamma _ { d } ( \kappa r ) \leq Q _ { d } ^ { \kappa } ( \rho , r ) < 1$ , with equality on the left at $\rho = r$ . This follows directly from $B ( x , \rho ) \subseteq B ( x , r )$ monotonicity in $\rho$ also follows from the same identity but is not needed. □

## C.6 Log-Space Variant

Alternatively, log $\Psi _ { \theta }$ can be fitted to log labels. There the weights are accumulated as $\begin{array} { r } { \Lambda ^ { ( n ) } = \sum _ { j < \sigma ^ { ( n ) } \wedge K } \log W _ { j } ^ { ( n ) } } \end{array}$ plus the log of the terminal value, which avoids underflow for long products, but log $\widehat { \Psi }$ is a biased estimate of log $\bar { \Psi } _ { : }$ since the logarithm of a sample mean is not the mean of logarithms. The correction below removes that bias to $O ( N ^ { - 2 } )$ .

Bias correction. For i.i.d. replicas with fixed bounds $0 < z _ { \operatorname* { m i n } } \le e ^ { \Lambda ^ { ( n ) } } \le z _ { \operatorname* { m a x } } < \infty$ and variance $s ^ { 2 } .$

$$
\mathbb { E } [ \log \widehat { \Psi } ] = \log \bar { \Psi } - \frac { s ^ { 2 } } { 2 N \bar { \Psi } ^ { 2 } } + O ( N ^ { - 2 } ) .
$$

The constants can depend on the replica bounds. For a fixed environment, such bounds follow from a finite horizon and a bounded target log field (Appendix C.7).

For $N \geq 2 ,$ compute all ratios from shifted exponentials. Set $\begin{array} { r } { b = \operatorname* { m a x } _ { n } \Lambda ^ { ( n ) } , z _ { n } = e ^ { \Lambda ^ { ( n ) } - b } , \bar { z } = N ^ { - 1 } \sum _ { n } z _ { n } . } \end{array}$ and $s _ { z } ^ { 2 } = ( N - 1 ) ^ { - 1 } \Sigma _ { n } ( z _ { n } - \bar { z } ) ^ { 2 }$ . Then

$$
\widehat { v } = \frac { s _ { z } ^ { 2 } } { N \bar { z } ^ { 2 } } , \qquad \widehat { \Lambda } = \log \widehat { \Psi } + \frac { \widehat { v } } { 2 } = b + \log \bar { z } + \frac { \widehat { v } } { 2 } .\tag{36}
$$

Here $s _ { z } ^ { 2 }$ is the sample variance of the shifted replicas $z _ { n }$ . The random common shift makes those replicas dependent; unbiasedness is invoked only for the unshifted replicas below. Writing $\widehat { s } ^ { 2 }$ for the unbiased sample variance of the unshifted replicas $e ^ { \Lambda ^ { ( n ) } }$ , the ratio is unchanged by the common shift, $\widehat { v } = s _ { z } ^ { 2 } / ( N \bar { z } ^ { 2 } ) = \widehat { s } ^ { 2 } / ( N \widehat { \Psi } ^ { 2 } )$ so Eq. 36 adds the bias correction ${ \widehat s } ^ { 2 } / ( 2 N \widehat \Psi ^ { 2 } )$ . The correction removes the leading log bias, leaving $O ( N ^ { - 2 } )$ under the stated assumptions.

## C.7 Delta-Method Bias of the Log Label

Proposition 4 (Bias of a log sample mean). $L e t Z _ { 1 } , \dots , Z _ { N }$ be i.i.d. with $Z _ { i } \in [ z _ { \operatorname* { m i n } } , z _ { \operatorname* { m a x } } ] , 0 < z _ { \operatorname* { m i n } } \leq z _ { \operatorname* { m a x } } < \infty$ mean $\mu ,$ and variance $s ^ { 2 }$ . For $N \geq 2$ , write $\widehat \Psi = N ^ { - 1 } \sum _ { i } Z _ { i }$ and let ${ \widehat { s } } ^ { 2 }$ be the unbiased sample variance. Then

$$
\mathbb { E } [ \log \widehat \Psi ] = \log \mu - \frac { s ^ { 2 } } { 2 N \mu ^ { 2 } } + O ( N ^ { - 2 } ) , \qquad \mathbb { E } \biggl [ \log \widehat \Psi + \frac { \widehat s ^ { 2 } } { 2 N \widehat \Psi ^ { 2 } } \biggr ] = \log \mu + O ( N ^ { - 2 } ) .
$$

The constants depend on the replica bounds.

Proof. Set $U = { \widehat { \Psi } } - \mu$ . Taylor’s theorem through degree three gives log $; ( \mu + U ) = \log \mu + U / \mu - U ^ { 2 } / ( 2 \mu ^ { 2 } ) +$ $U ^ { 3 } / ( 3 \mu ^ { 3 } ) + R _ { 4 }$ with $| R _ { 4 } | \le C U ^ { 4 }$ because $\widehat \Psi \geq z _ { \mathrm { m i n } }$ . Independence gives $\mathbb { E } U = 0 , \mathbb { E } U ^ { 2 } = s ^ { 2 } / N , \mathbb { E } U ^ { 3 } = O ( N ^ { - 2 } )$ and $\mathbb { E } U ^ { 4 } = O ( N ^ { - 2 } )$ , proving the first expansion.

For the correction, expand $( \mu + U ) ^ { - 2 } = \mu ^ { - 2 } - 2 \mu ^ { - 3 } U + R _ { 2 }$ , where $| R _ { 2 } | \le C U ^ { 2 }$ . Since $\widehat { s } ^ { 2 }$ is bounded, $\mathbb { E } [ \hat { s } ^ { 2 } R _ { 2 } ] = O ( N ^ { - 1 } )$ . Also $\mathbb { E } \widehat { s } ^ { 2 } = s ^ { 2 }$ and $\mathbb { E } [ \hat { s } ^ { 2 } U ] = m _ { 3 } / N$ , with m the third central moment of $Z _ { i }$ . Hence $\mathbb { E } [ \widehat { s } ^ { 2 } / \widehat { \Psi } ^ { 2 } ] = s ^ { 2 } / \mu ^ { 2 } + O ( N ^ { - 1 } )$ , which proves the corrected expansion. Jensen’s inequality gives the nonpositive bias of the uncorrected logarithm for every $N$ □

For a fixed environment, horizon $K ,$ and target bounded by $0 < \Psi _ { \operatorname* { m i n } } ^ { - } \le \Psi _ { \theta ^ { - } } \le \Psi _ { \operatorname* { m a x } } ^ { - } < \infty$ , let $W _ { \mathrm { m i n } } > 0$ be a lower bound on the step weight, which exists by Eq. 35 when the star radius is bounded. The walk replicas then lie in $[ z _ { \mathrm { m i n } } , z _ { \mathrm { m a x } } ]$ with $z _ { \mathrm { m i n } } = W _ { \operatorname* { m i n } } ^ { K }$ min(g<sub>min</sub>, Ψ<sup>−</sup><sub>min</sub>) and $z _ { \mathrm { m a x } } = \operatorname* { m a x } ( 1 , \Psi _ { \operatorname* { m a x } } ^ { - } )$ . The target bounds hold because the log field is continuous on the compact domain. They are not assumed uniform over all trained parameters or environments. The shifted calculation in Eq. 36 preserves the variance ratio exactly in real arithmetic and avoids underflow from a common small scale.

## C.8 Bounded Local Search

Recall the lattice of Section 4.4, with vertices $\mathcal { X } _ { h }$ and edges between axis-adjacent free cells. We take this graph to be finite and $\Psi _ { \theta }$ finite at every vertex.

Reachable endpoint scores. The largest desirability reachable in k moves is computed by

$$
B _ { 0 } ( x ) = \Psi _ { \theta } ( x ) , \qquad B _ { k + 1 } ( x ) = \operatorname* { m a x } _ { y \in \{ x \} \cup \mathcal { N } ( x ) } B _ { k } ( y ) .\tag{37}
$$

Indeed, $\mathcal { R } _ { 0 } ( x ) = \{ x \}$ and

$$
\mathcal { R } _ { k + 1 } ( x ) = \mathcal { R } _ { k } ( x ) \cup \bigcup _ { y \in \mathcal { N } ( x ) } \mathcal { R } _ { k } ( y ) ,
$$

so induction gives $B _ { k } ( \boldsymbol { x } ) = \mathrm { m a x } _ { z \in \mathcal { R } _ { k } ( \boldsymbol { x } ) } \Psi _ { \boldsymbol { \theta } } ( z )$ . Equation 14 therefore compares $B _ { L - 1 }$ at candidate neighbors.   
Moving to that neighbor uses the first of the L moves.

Recovering a route. In each propagation layer, the previous score is kept unless a neighbor has a strictly higher score, in which case the corresponding edge is stored. Starting at horizon $L ,$ follow the stored edges while decreasing the remaining horizon. A layer with no edge is skipped because its maximum was already reachable at a shorter horizon. Induction on the remaining horizon gives a feasible route of at most L edges ending at a cell with score $B _ { L } ( x _ { 0 } )$ . Executing a prefix preserves graph feasibility, although the next replan may select a diferent endpoint. Continuous-motion feasibility additionally requires valid edges; collision-free sampled endpoints are insuficient. The implemented lattice uses cells marked free in the saved occupancy mask, connected through face-adjacent free cells. Continuous segments from of-center states are checked across all traversed voxels. All edges are checked for collisions. For $e = 1$ , Eq. 14 instead selects a neighbor anew from the fixed $B _ { L - 1 }$ map at each visited state.

Saturation at large depth. The reachable sets are nested, hence $B _ { k + 1 } ( x ) \geq B _ { k } ( x )$ . On a connected component C of graph diameter D, every $k \geq D$ gives $B _ { k } ( x ) = \operatorname* { m a x } _ { z \in C } \Psi _ { \theta } ( z )$ for all $x \in C$ . Once $L - 1 \geq D$ , the candidate scores in Eq. 14 coincide and direction depends on tie-breaking. For a reconstructed route, maximizing the terminal score likewise does not distinguish the lengths of paths to the same endpoint. Thus increasing depth enlarges the set of candidate endpoints, but does not guarantee a shortest path or a successful rollout.

## D Supplementary Experiments

## D.1 Evaluation and Training Protocol

Evaluation scope. The pretrained 2D application benchmark contains 592 maps (37,888 starts); the 2D generated and 4D manipulator test sets each contain 512 maps (32,768 starts). Every map uses eight goals and every goal uses eight starts, chosen at the centers of eight equal goal-distance quantile bins. Success is evaluated on the matched saved queries, including failed rollouts in the denominator. Field prediction or solve, controller preprocessing, and rollout/search all contribute to reported test-time cost. These costs are amortized over the corresponding starts; they include more than the neural forward time. FEM solutions provide the numerical ground-truth fields for reconstruction-error evaluation.

Execution model for replanning comparisons. Table 5 compares nominal planners, sampling-based controllers, and PUN at execution-noise levels $\lambda \in \{ 0 . 2 5 , 1 , 2 , 4 \}$ . Before boundary handling, the proposed transition is $\widetilde { \boldsymbol { x } } _ { k + 1 } = \boldsymbol { x } _ { k } + \boldsymbol { a } _ { k } + \sqrt { \lambda \Delta t } \boldsymbol { z } _ { k }$ , with independent $z _ { k } \sim \mathcal { N } ( 0 , I _ { d } ) , \Delta t = 1 / 6 5 , 5 3 6$ , and $\| \boldsymbol { a } _ { k } \| _ { 2 } \leq 1 / ( 8 n )$ where n is the largest grid dimension. Each control interval contains eight microsteps. Coordinates occupy a unit square in $\mathrm { 2 D } , \textrm { a 1 } \times 1 \times \frac { 1 } { 2 }$ domain in 3D, and normalized joint coordinates in 4D. The noise-increment covariance is $\lambda \Delta t I _ { d } .$ consistent with the PDE specialization $\Sigma = \lambda I _ { d } .$ In these replanning evaluations, all compared methods use the same execution and collision-checking rules. Collision checks cover the entire executed motion segment, and an obstacle hit is counted as a rollout failure. Thus, success requires reaching the goal without an obstacle hit or an outer-domain boundary violation. The reflected PDE provides the planning field, whose efectiveness is evaluated here under collision-failure execution. Joint limits are not modeled in these experiments. Moves outside the outer workspace or configuration-domain boundary are also counted as rollout failures. For each environment–goal query at a given noise level, PUN predicts the entire field in a single forward pass and reuses it throughout the rollout. The reported replanning comparisons use positive noise levels, consistent with the $\lambda > 0$ scope of the screened-Poisson derivation. Success requires entry into the saved goal-cell mask before failure. Methods replan from the current noisy state every eight microsteps. Execution also stops on a collision failure, an unavailable route, or the controller’s stopping condition.

Baseline configuration. The MPPI and PI<sup>2</sup>-CMA implementations share a horizon of $H = 3 2 { \mathrm { ~ s t e p s , ~ } } K = 1 2 8$ sampled trajectories per optimization iteration, and one iteration per planning call. Both use double precision, componentwise action bounds $[ - 0 . 0 5 , 0 . 0 5 ]$ , initial exploration standard deviation 0.025, and optimization temperature $\eta = 0 . 5$ . This temperature is distinct from the execution-noise parameter λ. A fresh call initializes every action to $( x g - x ) / H$ , where $x _ { \mathcal { G } }$ is the goal center. Thus each call evaluates $K H = 4 { , } 0 9 6$ sampled transitions, followed by a rollout of the selected nominal sequence. The wrapper supports resuming interrupted optimization from saved parameters and random-generator state; ordinary calls initialize a new optimizer rather than carrying its solution between replans. The configuration uses $( H , K , I ) = ( 3 2 , 1 2 8 , 1 )$ , where I is the iteration count. These factors describe nominal sampled work IKH. No validation-based parameter sweep was performed. We use standard values for temperature, exploration scale, action bounds, the objective weights.

$\mathrm { A } ^ { \star }$ and Dijkstra use standard C++ implementations with face-neighbor connectivity and unit edge costs. Grids are the native 2D maps, $6 4 \times 6 4 \times 3 2$ in 3D, and $1 7 ^ { 4 }$ in 4D. $\mathrm { A } ^ { \star }$ uses Manhattan distance to the goal-mask bounding box; Dijkstra uses zero heuristic. CPU RRT uses OMPL 2.0.1, goal bias 0.05, uniform domain sampling, and maximum extension 0.4 in normalized coordinates. Search stops on a goal solution; there is no fixed sample or tree-size cap.

Planning objective and geometry. Both sampling-based optimizers use the following task cost, with stage costs evaluated after each predicted transition:

$$
\begin{array} { r l } & { \ell _ { t } = \mathbf { 1 } _ { \{ x _ { t + 1 } \notin \mathcal { G } \} } \left( 1 + 1 0 \| x _ { t + 1 } - x _ { \mathcal { G } } \| _ { 2 } ^ { 2 } + \| u _ { t } \| _ { 2 } ^ { 2 } \right) + 1 0 0 I _ { t } , } \\ & { \phi _ { H } = 1 0 0 \mathbf { 1 } _ { \{ x _ { H } \notin \mathcal { G } \} } \| x _ { H } - x _ { \mathcal { G } } \| _ { 2 } ^ { 2 } . } \end{array}
$$

Here $I _ { t }$ flags a proposed segment rejected by the geometry-validity routine when its starting state is not already in the goal. Before calling the geometry step routine, the wrapper replaces rejected actions and actions starting in the goal with zero displacement. This is a penalized planning surrogate; executed collisions are failures under the evaluation protocol above. Goal absorption is delegated to the geometry step routine. After optimization, the wrapper returns the predicted state path, truncated at its first recorded goal state. Execution noise is applied outside the optimization wrapper; its action-exploration distribution should not be identified with the execution-noise distribution. Geometry routines traverse all crossed voxels, including incident cells at edge/corner crossings, to detect goal entry, obstacle contact, and domain exit. Collision takes precedence over simultaneous goal entry in first-contact scoring. Actions are displacements in the normalized coordinates specified above. Predicted trajectories use deterministic geometry; the outer rollout adds execution noise. Returned paths are subdivided into bounded microsteps.

MPPI implementation. We use the MPPI class from pytorch mppi<sup>3</sup> (Williams et al., 2017a), with $\mathtt { n o i s e \_ s i g m a } = 0 . 0 2 5 ^ { 2 } I _ { d } ,$ lambda = 0.5, the task costs above, and the stated action bounds. The one call to command refines the nominal sequence at the same state with shift nominal trajectory=False. We retain the library’s MPPI update and control-perturbation cost in addition to the supplied task cost, and generate the returned path from the final nominal action sequence. The implementation uses the original MPPI update and control-perturbation cost.

Table 9: 95% confidence intervals for replanning success rates (%) corresponding to Table 5.
<table><tr><td rowspan=1 colspan=5>Dim.         Method                         λ = 0.25</td><td rowspan=1 colspan=2>λ = 1</td><td rowspan=1 colspan=5> $\overline { { \lambda = 2 } }$                  λ = 4</td></tr><tr><td rowspan=1 colspan=4>A*</td><td rowspan=1 colspan=1>[55.92, 62.08]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>39.16, 45.50]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>31.90, 38.10]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=2>20.89, 26.45]</td></tr><tr><td rowspan=1 colspan=4>Dijkstra</td><td rowspan=1 colspan=1>[55.81, 62.19]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[39.15, 45.51]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>31.99, 38.01]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[21.08,</td><td rowspan=1 colspan=1>26.26]</td></tr><tr><td rowspan=2 colspan=4>RRT2DMPPI</td><td rowspan=1 colspan=1>[79.37, 84.63]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>65.68, 71.66]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>49.14, 55.52]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[33.46,</td><td rowspan=1 colspan=1>39.88]</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>[81.70, 86.30</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>57.77, 64.23]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>42.45, 48.89</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[28.82,</td><td rowspan=1 colspan=1>34.52]</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>PI2-CMA</td><td rowspan=1 colspan=1>[75.76, 80.90</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>56.81, 63.19]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>44.76, 51.24]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[31.95,</td><td rowspan=1 colspan=1>38.05]</td></tr><tr><td rowspan=1 colspan=4>PUN (zero-shot)</td><td rowspan=1 colspan=1>[89.90, 93.44]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[71.27, 76.73]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>59.15, 65.51]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[42.55, 48.79]</td></tr><tr><td rowspan=1 colspan=4>A*</td><td rowspan=1 colspan=1>[32.37, 40.97]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[19.38, 27.28</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[9.10, 14.90]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[4.04, 7.96]</td></tr><tr><td rowspan=1 colspan=4>Dijkstra</td><td rowspan=1 colspan=1>[33.00, 41.66]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[18.98, 26.36]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[9.05, 14.95]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[3.73, 8.27]</td></tr><tr><td rowspan=2 colspan=4>RRT3DMPPI</td><td rowspan=1 colspan=1>[48.85, 57.81</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[30.86, 39.80]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>21.28, 29.38</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[8.69, 15.31]</td></tr><tr><td rowspan=1 colspan=1>[58.44, 66.90]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[30.43, 38.91]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[14.98, 22.36]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[4.93, 9.73]</td></tr><tr><td rowspan=2 colspan=4>PI2-CMAPUN (zero-shot)</td><td rowspan=1 colspan=1>[42.92, 51.74]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>18.36, 25.64]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>17.51, 25.15]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[6.07, 11.27]</td></tr><tr><td rowspan=1 colspan=1>[89.60, 94.40]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>64.44, 72.90]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[35.62, 44.38]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[12.71, 19.29]</td></tr><tr><td rowspan=1 colspan=3>A*</td><td rowspan=1 colspan=2>[100.00, 100.00]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[95.39, 97.95]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>69.54, 75.46</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[55.94, 62.40]</td></tr><tr><td rowspan=1 colspan=3>Dijkstra</td><td rowspan=1 colspan=1>[10</td><td rowspan=1 colspan=1>0.00, 100.00]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>95.68, 97.66]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>69.66, 75.34]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[56.06, 62.28]</td></tr><tr><td rowspan=2 colspan=3>RRT4DMPPI</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[97.53, 99.13]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>94.65, 97.01]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[87.97, 92.03]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=2>83.82, 87.84]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[93.68, 96.32]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>90.59, 94.41]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>79.15, 84.19]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>[68.91, 74.43]</td></tr><tr><td rowspan=1 colspan=3>PI2-CMA</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>95.43, 97.91]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>90.70, 94.30]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[84.40, 88.94]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=2>75.77, 80.89]</td></tr><tr><td rowspan=1 colspan=3>PUN</td><td rowspan=1 colspan=2>[100.00, 100.00]</td><td rowspan=1 colspan=2>[100.00, 100.00]</td><td rowspan=1 colspan=2>[100.00, 100.00]</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=2>93.49, 96.51]</td></tr></table>

PI<sup>2</sup>-CMA implementation. Our adaptation of $\mathrm { P I ^ { 2 } { - } C M A }$ (Stulp & Sigaud, 2012) represents the action sequence by 16 uniformly spaced knots per coordinate, linearly interpolated to the 32-step horizon with aligned endpoints and clipped to the action bounds. A single parameter vector is sampled for each trajectory from $\mathcal { N } ( \boldsymbol { \theta } , \boldsymbol { \Sigma } )$ , with initial covariance $\Sigma = 0 . 0 2 5 ^ { 2 } I _ { 1 6 d } .$ For sampled parameters $\theta _ { k }$ , let $S _ { k t }$ be the remaining task cost from step t, including the terminal cost once. We compute

$$
P _ { k t } = \frac { \exp [ - ( S _ { k t } - \operatorname* { m i n } _ { j } S _ { j t } ) / \eta ] } { \sum _ { j } \exp [ - ( S _ { j t } - \operatorname* { m i n } _ { l } S _ { l t } ) / \eta ] } , \qquad w _ { k } = \sum _ { t = 0 } ^ { H - 1 } \frac { H - t } { H ( H + 1 ) / 2 } P _ { k t } ,
$$

then update $\begin{array} { r } { \theta ^ { + } = \sum _ { k } w _ { k } \theta _ { k } } \end{array}$ and $\begin{array} { r } { \Sigma ^ { + } = \sum _ { k } w _ { k } ( \theta _ { k } - \theta ) ( \theta _ { k } - \theta ) ^ { \top } } \end{array}$ , centering the covariance at the old mean. This implements temporal averaging of the per-step updates for exploration that is constant over a trajectory. The implementation symmetrizes the covariance and samples using an eigendecomposition, checking negative eigenvalues against a numerical tolerance before clipping them to zero; it imposes no positive exploration floor. We use a fixed temperature on unnormalized costs, rather than the cost-range normalization used in the original paper’s experiments. The custom wrapper implements knot-based $\mathrm { P I ^ { 2 } { - } C M A }$ , without CMA-ES evolution paths, rank-based selection, or step-size adaptation.

Hardware and timing. Both CPU and CUDA implementations were evaluated; the CPU version was faster in the implementation comparison. The optimizer’s wall-clock timer includes initialization, the one optimization iteration, logging, the final nominal rollout, and conversion of the returned path to host memory. The CUDA implementation also synchronizes after each iteration and before returning. These are per-call measurements; the tables report per-start costs and require the outer execution loop’s accounting across planning calls. Table 5 uses CPU baseline timings in $\mathrm { 2 D / 3 D / 4 D }$ since they are faster than GPU, and GPU PUN timings throughout. Hardware is an Intel Xeon Platinum 8480+ or an NVIDIA B200. CPU software is Python 3.11.11, PyTorch 2.7.1+cpu, and NumPy 1.26.4; GPU software is Python 3.9.21, PyTorch 2.7.1+cu128, and NumPy 2.0.2.

Timing excludes loading and warmup and uses three observations per selected pair, method, and noise level for the benchmark sets. Variability is minimal. Per-start measurements include initialization, all replanning calls, geometry, and execution, including failed rollouts; the number of calls depends on the realized trajectory. GPU work is synchronized at timing boundaries.

PUN timings include fresh field prediction and preparation for each start, with the field reused during replanning.

Statistical reporting for replanning. Table 5 uses 500 maps in 2D, 250 in 3D, and 480 in 4D, with three goals per map and one start per goal. Each query uses one seeded execution-noise combination at each noise level; methods share the saved queries and Gaussian noise sequence. Rollouts receive equal weight, which also gives maps equal weight because each contributes three queries. One trained model per dimension is evaluated. We compute 95% intervals using 50,000 paired map-level bootstrap resamples, retaining all queries within each sampled map. Nondegenerate intervals are the estimate ±1.96 bootstrap standard errors. Runtime intervals use separate samples of maps, respectively.

Implementation of the WoSt estimator. The experiments use the following estimator and numerical choices.

Table 10: 95% confidence intervals for replanning runtime (ms/start) corresponding to Table 5.
<table><tr><td rowspan=1 colspan=5>Dim.       Method                         λ = 0.25                    $\overline { { \lambda = 1 } }</e</td><td rowspan=1 colspan=3>q>                   <eq>\overline { { \lambda = 2 } }$                    λ = 4</td></tr><tr><td rowspan=1 colspan=1>A*</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[24.99, 62.27]</td><td rowspan=1 colspan=2>[26.25, 67.93]</td><td rowspan=1 colspan=1>[28.10, 74.26]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[29.44, 78.08]</td></tr><tr><td rowspan=1 colspan=1>Dijkstra</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[79.37, 230.51]</td><td rowspan=1 colspan=2>[88.65, 261.25]</td><td rowspan=1 colspan=1>[102.34, 304.48]</td><td rowspan=1 colspan=2>[123.40, 356.54]</td></tr><tr><td rowspan=2 colspan=1>RRT2DMPPI</td><td rowspan=1 colspan=1>[</td><td rowspan=1 colspan=1>69.41, 127.49]</td><td rowspan=1 colspan=2>[72.10, 135.64]</td><td rowspan=1 colspan=1>[78.18, 146.36]</td><td rowspan=1 colspan=2>[82.46, 166.96]</td></tr><tr><td rowspan=1 colspan=2>[1227.92, 2240.30]</td><td rowspan=1 colspan=2>[1257.88, 2288.74]</td><td rowspan=1 colspan=1>[1279.80, 2397.66]</td><td rowspan=1 colspan=2>[1400.54, 2851.70]</td></tr><tr><td rowspan=1 colspan=1>PI2-CMA</td><td rowspan=1 colspan=2>[964.06, 1697.46]</td><td rowspan=1 colspan=2>[1053.67, 1916.15]</td><td rowspan=1 colspan=1>[1067.48, 1981.92]</td><td rowspan=1 colspan=2>[1079.98, 2133.66]</td></tr><tr><td rowspan=1 colspan=1>PUN (zero-shot)</td><td rowspan=1 colspan=2>[7.79, 11.31]</td><td rowspan=1 colspan=2>[8.80, 13.48]</td><td rowspan=1 colspan=1>[10.71, 17.43]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[10.19, 18.83]</td></tr><tr><td rowspan=1 colspan=1>A*</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[50.74, 96.06]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[34.34, 86.60]</td><td rowspan=1 colspan=1>[42.19, 105.51]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[40.32, 90.18]</td></tr><tr><td rowspan=1 colspan=1>Dijkstra</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[50.08, 96.84]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[35.29, 90.79]</td><td rowspan=1 colspan=1>[43.41, 110.57]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[41.51, 93.95]</td></tr><tr><td rowspan=2 colspan=1>RRT3DMPPI</td><td rowspan=1 colspan=1>[15</td><td rowspan=1 colspan=1>5.27, 321.35]</td><td rowspan=1 colspan=2>[93.18, 246.62]</td><td rowspan=1 colspan=1>[76.54, 169.16]</td><td rowspan=1 colspan=2>[94.56, 157.90]</td></tr><tr><td rowspan=1 colspan=2>[382.10, 682.52]</td><td rowspan=1 colspan=2>[408.91, 746.77]</td><td rowspan=1 colspan=1>[433.13, 675.09]</td><td rowspan=1 colspan=2>[526.69, 881.93]</td></tr><tr><td rowspan=1 colspan=1>PI2-CMA</td><td rowspan=1 colspan=2>[332.42, 572.08]</td><td rowspan=1 colspan=2>[305.61, 564.41]</td><td rowspan=1 colspan=1>[371.44, 580.52]</td><td rowspan=2 colspan=2>[482.06, 731.04][13.69, 15.17]</td></tr><tr><td rowspan=1 colspan=1>PUN (zero-shot)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[14.95, 16.33]</td><td rowspan=1 colspan=2>[14.61, 16.07]</td><td rowspan=1 colspan=1>[13.26, 15.34]</td></tr><tr><td rowspan=1 colspan=1>A*</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[12.32, 21.30]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[12.25, 21.91]</td><td rowspan=1 colspan=1>[35.26, 120.78]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[28.18, 96.70]</td></tr><tr><td rowspan=1 colspan=1>Dijkstra</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[12.51, 21.83]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[12.37, 22.31]</td><td rowspan=1 colspan=1>[31.21, 102.29]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[27.73, 91.05]</td></tr><tr><td rowspan=2 colspan=1>RRT4DMPPI</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[16.26, 32.14]</td><td rowspan=1 colspan=2>[16.20, 35.40]</td><td rowspan=1 colspan=1>[23.09, 62.91]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>[28.09, 48.53]</td></tr><tr><td rowspan=1 colspan=2>[332.13, 490.35]</td><td rowspan=1 colspan=2>[329.20, 467.16]</td><td rowspan=1 colspan=1>[297.34, 431.10]</td><td rowspan=1 colspan=1>[30</td><td rowspan=1 colspan=1>0.44, 449.22]</td></tr><tr><td rowspan=1 colspan=1>PI2-CMA</td><td rowspan=1 colspan=2>[300.61, 458.97]</td><td rowspan=1 colspan=2>[291.21, 424.17]</td><td rowspan=1 colspan=1>[284.19, 426.69]</td><td rowspan=2 colspan=2>[288.82, 424.62][11.43, 11.89]</td></tr><tr><td rowspan=1 colspan=1>PUN</td><td rowspan=1 colspan=2>[10.50, 10.76]</td><td rowspan=1 colspan=2>[10.32, 10.70]</td><td rowspan=1 colspan=1>[10.37, 10.73]</td></tr></table>

1. Absorption and radius floor. A walk reaches $\Gamma _ { D }$ only in the limit, so absorption is triggered on the shell $\{ d _ { D } <$ $\varepsilon \}$ and the terminal value is g at the nearest point of $\Gamma _ { D } ; \varepsilon$ is chosen below the tubular-neighborhood radius of $\Gamma _ { D }$ and its distance to $\Gamma _ { N }$ . The star radius also carries a floor, $r ( x ) = \mathrm { m i n } \{ d _ { D } ( x )$ , max $( r _ { \mathrm { m i n } } , d _ { \mathrm { s i l } } ( x ) ) \}$ , so that steps do not become arbitrarily short near a silhouette. The shell costs at most $C _ { \Psi } \varepsilon$ (Eq. 38); the floor can break star-shapedness, and the error it introduces has unknown sign (Sawhney et al., 2023).

2. Weighted survival. The solid-angle sampling law in Eq. 8 leaves the scalar screening factor $Q _ { d } ^ { \kappa }$ as the per-step weight. We multiply these factors rather than sample killing events or apply Russian roulette. Conditioning out the killing randomness removes that source of variance and avoids zero replicas caused by stochastic killing.

3. Fixed-horizon continuation. Walks stop at the prescribed step cap or the goal. Capped paths contribute their accumulated weight times the detached EMA prediction at the endpoint, as in Section 4.2. Goal-reaching paths use the known boundary value. Every requested trajectory remains in the sample mean, including capped paths.

4. Log accumulation and bias correction. The trace records the sum of log step weights independently of their floating-point product, so product underflow does not erase the log evidence. The log-label variants use log-sum-exp and the sample-variance correction in Eq. 36. The direct-label variants use the weighted arithmetic mean without this correction. Direct and log targets are constructed from the same sampled paths.

5. Reuse of model-independent walks. Cached endpoints, weights, and completion states are independent of the neural parameters. The EMA continuation is reevaluated during training rather than frozen into the cache.

6. Stable screening kernels. The voxel backend for dimensions three through seven evaluates the full-sphere factor using the positive series

$$
\gamma _ { d } ( z ) = \left[ \sum _ { j = 0 } ^ { \infty } a _ { j } \right] ^ { - 1 } , \qquad a _ { 0 } = 1 , \qquad a _ { j } = a _ { j - 1 } \frac { z ^ { 2 } } { 4 j ( \nu + j ) } , \qquad \nu = \frac { d } { 2 } - 1 .
$$

The CUDA implementation sums in double precision and uses a logarithmic large-argument asymptotic branch. For interior reflecting hits, it evaluates Eq. 25 with $e ^ { - z } I _ { \nu } ( z )$ and $e ^ { z } K _ { \nu } ( z )$ , combining nonnegative terms with an exponential factor no greater than one. With $t = \kappa \rho , T = \kappa r .$ and $\delta = T - t .$ the near-sphere branch uses

$$
Q _ { d } ^ { \kappa } ( \rho , r ) \approx \gamma _ { d } ( T ) \left[ 1 + \frac { \delta ^ { 2 } } { 2 } - \frac { d - 1 } { 6 T } \delta ^ { 3 } \right] .
$$

This reproduces the sphere endpoint without two separately rounded Bessel evaluations. The CUDA branches use this expansion for $\delta \leq 2 \cdot 1 0 ^ { - 6 } ( 1 + T )$ and the limiting value one for $t \leq 1 0 ^ { - 7 }$ or $T = 0$

7. Voxel boundary queries. The same backend uses bounding-volume hierarchies for goal faces and viewdependent reflecting silhouettes, and voxel traversal for first boundary hits. This avoids triangulating the voxel boundary. Grid-plane and corner handling preserves the appropriate incident free cell and resolves boundary hits before accepting a sphere endpoint. These geometric operations implement the first-hit chain in Eq. 7.

Shell error. Let $\sigma _ { \varepsilon } = \operatorname* { i n f } \{ k : d _ { D } ( \xi _ { k } ) < \varepsilon \}$ and $\tau = \sigma _ { \varepsilon } \wedge K$ for finite K. Assume admissible active steps, the smooth-domain regularity assumptions, and an ε below the goal’s tubular-neighborhood radius and its separation from reflecting boundaries. Define $\Psi _ { \varepsilon , K } ( x )$ as the expected product of weights through τ times $g ( \pi _ { D } \xi _ { \tau } )$ if $\sigma _ { \varepsilon } \leq K$ or the true continuation $\Psi ( \xi _ { K } )$ otherwise. Then

$$
\| \Psi _ { \varepsilon , K } - \Psi \| _ { \infty } \leq C _ { \Psi } \varepsilon , \qquad C _ { \Psi } = \operatorname* { s u p } _ { \overline { { \Omega } } } \| \nabla \Psi \| .\tag{38}
$$

To prove this, bounded stopping gives $\begin{array} { r } { \Psi ( x ) = \mathbb { E } _ { x } [ ( \prod _ { i < \tau } W _ { j } ) \Psi ( \xi _ { \tau } ) ] } \end{array}$ . Only the shell event changes the terminal value. Since $g ( \pi _ { D } \xi _ { \tau } ) = \Psi ( \pi _ { D } \xi _ { \tau } )$ , the projection segment lies in the closed free domain, and its length is below $\varepsilon ,$ the terminal diference is bounded by $C _ { \Psi } \varepsilon$ . The product of weights is at most one. This proves the bound without an infinite-walk termination assumption. If $\sigma _ { \varepsilon } <$ ∞ almost surely for each start, bounded convergence also yields the same bound for the uncapped shell estimator. Replacing true continuation by a fixed bounded target adds its discounted endpoint error on $\{ \sigma _ { \varepsilon } > K \}$ , as described in Section 4.2. The radius floor is not covered: a clamped step can violate admissibility and introduce an error of unknown sign (Sawhney et al., 2023).

Antithetic directions, first-step gradient control variates, weight-floor stopping, expected-clock stopping, and randomized horizons are disabled in the reported pretrained models. In particular, the full-sphere gradient control variate is not transferred to arbitrary WoSt stars.

Backbone compute policy. FNO uses 200 epochs. Each alternative backbone has a maximum of 200 epochs and a registered training-compute allowance of 66,856.484 seconds; an epoch is not started if it is not expected to fit within the remaining allowance. Each completed epoch uses 64 training maps and eight goals per map. The 200-epoch cosine schedule is retained when a run stops at the compute cap. The allowance was derived from 1.5 times an estimated individual FNO training cost; shared work in the reference run means it is not an exact equal-GPU-time comparison. Completed training epochs were 200 for FNO, 200 for U-NO, 185 for U-Net, 13 for GINO, 200 for LocalNO, 200 for DISCO, 146 for $\mathrm { { U P T } } ,$ and 134 for Transolver. Our overall method is architecture-agnostic; we chose FNO because it gives the highest planning success among the tested implementations under this compute policy, but our overall method is compatible with other neural-operator architectures. Planning benefits from balancing sensitivity to local obstacle geometry with a global view of the environment.

Supervision parameterization. The primary WoSt rows fit direct desirability with value, goal, and gradient loss terms. The loss ablations use 200 epochs and remove one term at a time. All predictions are evaluated through the stated controller; a small field error does not guarantee path success.

Local-search implementation. Unless a depth is stated, Local search denotes Local search (10). The finite lattice controllers use legal axis-adjacent moves, with cycle detection and a step cap. Greedy uses $( L , e ) = ( 1 , 1 )$ and requires strict improvement. Local search (4) uses $( L , e ) = ( 5 , 1 )$ in 2D, 3D, and 4D. Local search (10) uses $( L , e ) = ( 1 0 , 3 )$ with route reconstruction in 2D, and $( L , e ) = ( 1 1 , 1 )$ in 3D and 4D. In the single-move variants, four or ten propagation steps are counted after the candidate first move, explaining the extra edge in L. Direct models use their raw finite desirability predictions, without clipping or enforcing a goal score during search. For the convention $\ell _ { \theta } = \log \Psi _ { \theta }$ in Appendix C.6, the maximizing search score is $\ell _ { \theta }$ . If a model instead predicts − log $\Psi _ { \theta }$ or $V _ { \theta }$ , its output must be negated to preserve this ranking (for $\lambda > 0$ in the latter case). The implemented log-label model predicts $\ell _ { \theta } = - \operatorname { s o f t p l u s } ( z _ { \theta } )$ , reconstructs desirability as $\exp ( \ell _ { \theta } )$ , and uses −z<sub>θ</sub> as its search score. Maximizing $- z _ { \theta }$ gives the same ranking as maximizing $\ell _ { \theta } .$ , consistent with this convention. Success requires entering the saved goal mask. Ties in the single-move rule favor the candidate’s own field score, then fewer previous visits, then a fixed lexicographic coordinate order (ascending in 2D, descending in $\mathrm { 3 D / 4 D ) }$ Route reconstruction retains an existing choice on equal scores, using a fixed axis traversal order for competing improvements. The 2D ten-step controller stops when the best endpoint does not improve the current score by more than $1 0 ^ { - 1 2 }$ . Missing legal moves, a fifth visit to a cell under local search, or exhaustion of the rollout budget are recorded as failures. The local diagrams below use one stored field to display these rules, with no new benchmark outcomes inferred from the illustration.

Details of the walk-horizon study and qualitative resolution-transfer examples are provided below.

## D.2 Walk Horizon and Label Budget

Label budget and planning success. The controlled horizon study in Table 3 uses 16 training maps, eight goals per map, four trajectories per query, and 40 epochs, with evaluation on the generated 2D test set. With the intermediate 100- and 400-step settings, gradient / local-search (10) success is 40.55% / 66.80% and 38.06% / 69.69%, respectively. Local-search success is highest at 400 steps (69.69%) and 0.03 percentage points lower at 800 steps, whereas gradient success is highest at 100 steps. A separate study of budget allocation without pretraining is reported in Appendix F.1.

Table 11: FNO (WoSt) planning success (%) with gradient following and local search.
<table><tr><td>Test set</td><td>Gradient</td><td>Local search (10)</td></tr><tr><td>3D application benchmarks</td><td>65.95</td><td>89.46</td></tr><tr><td>4D manipulators</td><td>97.50</td><td>99.48</td></tr></table>

Table 12: Full test-time cost (ms/start) on the 2D zero-shot application-inference subset and 4D manipulator test set. Both methods use local search in 2D and in 4D.
<table><tr><td>Method</td><td>2D zero-shot</td><td>4D manipulators</td></tr><tr><td> $\overline { { \mathrm { W o S t } + \mathrm { F N O } } }$ </td><td>1.20</td><td>2.79</td></tr><tr><td>Direct WoSt</td><td>1128.60</td><td>160.59</td></tr></table>

## D.3 Resolution Generalization

Figure 4 shows FNO predictions on held-out maps at $1 2 8 \times 1 2 8 , 2 0 0 \times 2 0 0$ , and $2 5 6 \times 2 5 6$ . Each field supports paths from multiple starts to one goal. These examples illustrate field reuse at diferent discretizations; the aggregate comparison across architectures and larger grids is Fig. 1(b).

![](images/f3592c7ee29c681319c95376c5e746795cb67084e4963599992c5d3ed1af6f13.jpg)  
Fig. 4: Zero-shot FNO field predictions and extracted paths at $1 2 8 \times 1 2 8 , 2 0 0 \times 2 0 0 ,$ and $2 5 6 \times 2 5 6$ (left to right). Color shows predicted desirability $\Psi _ { \theta ; \pmb { \mathscr { v } } }$ gray denotes obstacles, circles mark starts, and the star marks the shared goal in each map.

## E Benchmarks

2D (zero-shot; 592 maps). AI4Arctic Sea Ice (Arctic sea-ice maps from satellite observations); Fields2Benchmark (agricultural field geometries); HiRISE DTM (terrain models of the surface of Mars); HydroLAKES ASV (lake-shoreline geometries for autonomous surface vessels); LOLA LDEM (lunar elevation maps from laser altimetry); MovingAI BG512 (Baldur’s Gate maps); MovingAI Street (urban street-map grids); NOAA ENC Harbor (harbor geometries from navigational charts); Open Forest Observatory (forest structure maps from drone imagery); PNO2D (synthetic obstacle environments); TartanDrive 2 (terrain from of-road driving data).

3D (zero-shot; 278 maps). Dryad SIC Foam (porous foam geometries); Falkor MAR AUV (Mid-Atlantic Ridge seafloor for autonomous underwater vehicles); Flybo UAV (scenes for aerial-robot navigation); HVaCT Liver Tip (liver and vessel anatomy for navigation); Industrial Pipe Routing (industrial layouts for pipe routing); Lung Needle (lung anatomy for needle-path planning); MOSAiC Under-Ice AUV (Arctic under-ice geometry for autonomous underwater vehicles); PNO3D iGibson/HouseExpo (indoor scene and floor-plan geometries); ReplicaCAD Indoor UAV (indoor CAD scenes for aerial navigation); Trabecular Bone (porous internal bone structure); Underwater AT40-02 MAR (seafloor surveyed for the Mid-Atlantic Ridge); Axial North Rift (seafloor terrain at Axial Seamount’s north rift); Escanaba (seafloor terrain in Escanaba Trough); Loihi Pele’s Pit (a submarine volcanic crater); Squid5 EDR (Eastern Dry Rocks coral reef topography from underwater imagery); USGS Tahoe (Lake Tahoe lakebed topography).

geometry + goal

differentiate

4D (512 maps). eCobra600 (Omron industrial robot arm); Kinova MICO (robot arm); OpenManipulator (ROBOTIS open-source robot arm); qArm (Quanser robot arm). Each 4D state represents a manipulator configuration in joint coordinates.

## F Additional Results

## F.1 Label-Budget Allocation without Pretraining

Allocation of a fixed label budget. Shifting from 64 walks per query and 8 goals per map to 4 walks and 128 goals reduces 2D test success from 62.6% to 51.5% for DISCO and from 61.6% to 56.3% for FNO. In 3D, the same shift raises FNO success from 25.0% to 28.8%; Transolver success changes from 6.3% to 7.0%, while label-generation cost rises from 244.35 to 621.46 GPU hours. Equal nominal label budgets therefore need not imply equal computation or a common preferred allocation.

## G Qualitative Examples and Figures

The following examples illustrate the geometry-to-field mapping, stochastic labels, and path extraction. The sampling budget for each field and walk figure is indicated in its panel.

## G.1 Geometry, Fields, and Multiple Starts

Figure 5 illustrates the reuse of a goal-conditioned field across multiple initial states: the field represents a map and goal, while path extraction starts from each selected location.

single path  
![](images/b55b6c0a2aab04f67632c5256f6b3a6f405fbef92d32ce42b387513237139802.jpg)  
what is free, blocked, desired?

global navigation field  
![](images/69e645a8e7a3acf37719fc32a301781162336e14b7ee1a5a7a6f0808f4a2ec42.jpg)

![](images/3befd128c13ba3cbb4df50ce4cff59b2d9ff4cb19aac5d54952a6d6146885823.jpg)

![](images/44791f9d01792f09cacdafecbe22ea61b51a81ad26fdda70eb4ad143958cfe23.jpg)

one field solves many starts  
![](images/9e8b5160bdd5921a4f654e132d962d7531bd4717119779ec515598537cafdfae.jpg)  
Fig. 5: Goal-conditioned field prediction and reuse across initial states.

## G.2 Walk Geometry and Label Variability

Figure 6 brings together walk geometry, label variability, and spatial sampling cost.

(a) Walk geometry and boundary treatment  
![](images/b698e912f2ba95ec98103c2e8ed9ef65c8b2ef62abc542f5beedb27dab06e48d.jpg)

![](images/632988dc934157d704049a2016859cc77e36dcd557cc6a83cb07da33a087a9ec.jpg)

![](images/3c50c857043fefe86b3f060e60ef3d9a6fec39929c176a79b5028121ed8ceaf1.jpg)

![](images/15c876c3b9f1f0ee07a6a736715647dd2f61b21cfd7f91344a1d6e8540b8794f.jpg)

(b) Label and gradient variability  
![](images/89bb423615ce8d92f8880f447a314b385c5901b252a7618e42417a8733b425dc.jpg)

![](images/5f21e02fb0b7f58b849032e9290c2c1f8e20c81692bbb9996479c0a6ffee5af7.jpg)

![](images/48b7990ec83bff4d1e5555bba0a9dbc1626f8aad21cae57ae5df2eb276b795b9.jpg)

![](images/6f7efe05e77310a9c7f6e701df47d1679651c24613c7c8fbde7bc5345aed2d41.jpg)

![](images/24782d7af643a936362fcab6d4bbf49cc87bd508568e351115a8b52809150bb0.jpg)

![](images/5b804ff338ad320fcdb9fea670f330237e70fea1e86cc1e7408ea00d583443bd.jpg)

![](images/ea90d3ec8e6bc00b64c3aad2ad564905ba7fe52657a60a8fc68c01c4f66df97e.jpg)

(c) Mean walk length  
![](images/24f6c1964c0a1e8d4166730121a6a689e9d1311086aaf82162b159df05a9a445.jpg)

![](images/3d9d4c99dd3e10cb401af53c17ded9efb517563e9987126f0a7be2c0959e30cf.jpg)  
Fig. 6: WoSt walk and label statistics: (a) walk-length variability and boundary treatment; (b) label and gradient variability on a fixed map, with diferences measured against a finite Monte Carlo reference; (c) spatial variation in mean walk length.

## G.3 Local Search Example

![](images/5259d5705d56720cbf434ec5c8de7076e78476d597e6a83fb8c3b508efb4b0f5.jpg)

![](images/1e1cff7eef0d879d8066d2d5ecee83c042cbef139397ace259f8e384ca2fc6c4.jpg)

![](images/7ebf2250e24fdeaf09c6fddcd469f3917c33f3328aee874d3b44b5e99246a5ca.jpg)  
Fig. 7: Local search on one fixed urban-map field: (A) full map and routes; (B,C) enlarged neighborhoods for local search (4) and (10). Greater depth exposes endpoints farther along the passage. Gray cells are obstacles; overlays mark candidate regions and routes. Table 1 reports aggregate controller success.