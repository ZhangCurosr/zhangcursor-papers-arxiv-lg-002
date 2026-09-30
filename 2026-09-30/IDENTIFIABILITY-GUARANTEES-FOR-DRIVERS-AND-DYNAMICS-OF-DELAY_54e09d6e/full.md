# IDENTIFIABILITY GUARANTEES FOR DRIVERS AND DYNAMICS OF DELAYED PHYSICAL SYSTEMS

Julien Boussard<sup>1,2,†</sup>

Antoine Debouchage´ <sup>3</sup>

Theo Saulus´ <sup>2,4</sup>

<sup>1</sup>School of Computer Science, McGill University, Canada <sup>2</sup>Mila - Quebec AI Institute, Canada <sup>3</sup>LaMMe, Universite´ Evry Paris-Saclay, France <sup>´</sup> <sup>4</sup> DIRO, Universite de Montr ´ eal, Canada ´ <sup>†</sup>julien.boussard@mila.quebec

## ABSTRACT

Learning the dynamics of a physical system directly from observations is a key problem in natural sciences, where physical consistency and interpretability are essential. A wide range of methods have been proposed, including physicsinformed neural networks, which are powerful but do not guarantee identifiability of the dynamics, symbolic regression, which requires a set of precomputed operations, and causal discovery, which is more principled but usually relies on strong assumptions that physical systems may violate. In this work, we develop a theorygrounded method and prove that under a set of permissive assumptions, the structural drivers and drift of stochastic delayed differential equations are identifiable. Our method outperforms others on a benchmark for driver identifiability, and on a second benchmark to evaluate physical consistency of the learned dynamics.

## 1 INTRODUCTION

The evolution of many physical systems depends not only on their current state but also on their past states. For example, the climate system and ecological or neural population dynamics are driven by delayed feedbacks between variables (Chekroun & Liu, 2024; Keane et al., 2017; C¸ etin et al., 2026). Lagged dynamical systems explicitly model both instantaneous and time-delayed dependencies, which improves the predictive capacity and interpretability of the learned model. Such systems are formally described by stochastic delayed differential equations (SDDEs) (Mohammed, 1982; Scheutzow, 1984) which allow for delayed connections, as opposed to stochastic differential equations (SDEs) used to describe Markovian dynamical systems (Ito, 1944; 1951).ˆ

Lagged connections in physical systems such as the climate system typically lead to oscillations and chaotic behavior, characterized by ergodicity and chaoticity (Ghil et al., 2008; Ghil & Lucarini, 2020). Ergodic means that long-time statistics are identical across almost all initial conditions (Wal ters, 1999), while chaotic means that an infinitesimally small difference between two initial conditions will lead to two paths diverging exponentially fast (Dorfman, 1999).

Scientific machine learning (ML) has mostly focused on modeling SDE-driven systems. Neural ordinary differential equations (ODEs) (Chen et al., 2018; Dupont et al., 2019) allow the direct approximation of the gradient of a physical system from irregular sparse observations (Zhang et al., 2020). They have been extended to capture delay (Eggen & Midtfjord, 2023; Oh et al., 2025), but still do not learn the long-term statistics of the system, leading to unrealistic long-horizon trajectories. Indeed, Park et al. (2024) have shown that data-driven neural parameterizations of chaotic ordinary or partial differential equations (PDEs) fail to generalize when modeling ergodic chaotic systems, despite low test error. Physics-informed neural networks (Yin et al., 2021), or symbolic regression methods (Udrescu et al., 2020; Brunton et al., 2016) improve generalization but require a known parametric form or a set of precomputed functions and are computationally expensive.

Motivated by the above limitations, we aim to learn the true drivers and drift of an SDDE. They must therefore be identifiable, i.e. uniquely recoverable from observations. Causal discovery methods have emerged as a promising avenue to infer cause–effect relationships from time-series observations and have been extended to complex, nonlinear, lagged causal structures (Runge, 2020; Yao et al., 2022; Lippe et al., 2022; Hickman et al., 2025; Brouillard et al., 2026). To identify the causal graph, these methods typically rely on strong constraints such as acyclicity (Pamfil et al., 2020) or no instantaneous connections, which physical systems often violate. Another approach is to embed traditional regularization and feature selection methods into Neural ODEs (NODEs, Aliee et al., 2025), with identifiability guarantees derived for Markovian systems (Bellot et al., 2022).

In this work, we provide identifiability guarantees for ergodic, chaotic SDDE-driven systems, both at population level and in finite-sample settings. We use $L _ { 0 }$ regularization (Louizos et al., 2018), which has been proposed in the disentanglement literature (Lachapelle et al., 2023) and to regularize neural ODEs (Aliee et al., 2022). We show that under a set of permissive assumptions, it is sufficient for identifying the correct lagged causal connections and drift of SDDEs.

## Main contributions:

• We develop a theory-grounded method $( L _ { 0 } .$ -Neural DDE, $L _ { \mathrm { 0 } } – \mathrm { N D D E ) }$ and provide novel identifiability proofs for lagged and instantaneous drivers and drift of SDDE systems.

• We demonstrate that $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ outperforms other data-driven methods for identifying structural drivers of dynamical systems on a comprehensive benchmark.

• We show that $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ accurately captures long-term statistics on 40 simulated lowdimensional chaotic systems from the Dysts dataset (Gilpin, 2021).

## 2 RELATED WORK

Several causal discovery approaches have been proposed for time-series data. Building upon Granger causality (Granger, 1969), which fails to capture nonlinear dynamics (Sugihara et al., 2012), methods such as Neural Granger Causality (NGC) (Tank et al., 2022), CUTS+ (Cheng et al., 2024) or Temporal Causal Discovery Framework (TCDF) (Nauta et al., 2019) use various deep-learning techniques to learn nonlinear autoregressive links and identify lagged causal links directly from multivariate sequences. Constraint-based methods such as PCMCI+ (Runge, 2020) or F-PCMCI (Castri et al., 2023) infer causal structure using conditional independence tests between lagged variables, while permutation-based methods (GRaSP, Lam et al., 2022) search over permutations of variables and iteratively prune spurious edges to identify sparse causal graphs. Score-based methods such as DYNOTEARS (Pamfil et al., 2020) infer causal relationships by optimizing a score function with differentiable constraints that ensure identifiability of the causal graph (Zheng et al., 2018; Brouillard et al., 2026; Hickman et al., 2025). Noise-based methods (VARLiNGAM, Hyvarinen et al.,¨ 2010; RCD, Maeda & Shimizu, 2020) leverage the statistical independence of the noise term with respect to the inputs to identify causal connections. Finally, cross-mapping techniques discover lagged causal relationships by aligning lagged time-series (TSCI, Butler et al., 2024).

These causal methods typically discretize time and rely on strong assumptions about the causal graph (e.g., acyclicity) which often violate the nature of the continuous-time dynamics (Tagliapietra et al., 2026). To overcome this issue, a growing body of work has proposed to integrate graphical modelling approaches into Neural ODEs to identify the Jacobian of the physical system. Penalized regression has been used to estimate parameters of differential equations, with consistency guarantees (Ramsay et al., 2007; Raissi et al., 2017; Wenk et al., 2020). More recently, Bellot et al. (2022) derived identifiability results for drivers of SDEs and proposed to use adaptive group lasso (AGL, Zou, 2006) to select input features and identify drivers. This work has been applied to SDDEs without identifiability guarantees (Wu et al., 2024).

In our work, we propose to use $L _ { 0 }$ -induced sparsity (Louizos et al., 2018) on the input features to identify the instantaneous and lagged drivers of SDDEs. We derive novel theoretical guarantees for the identifiability of the dynamics of SDDEs, under a discrete, finite number of observations. $L _ { 0 } -$ induced sparsity has been used in causal representation learning (Lachapelle et al., 2023), and for regularizing neural ODEs (PathReg, Aliee et al., 2022).

## 3 METHOD

We consider the following continuous-time stochastic model, with delays $0 < \tau _ { 1 } < \cdots < \tau _ { q } ,$

$$
d X _ { t } = G ( X _ { t } , X _ { t - \tau _ { 1 } } , \ldots , X _ { t - \tau _ { q } } ) d t + \Sigma d W _ { t } ,
$$

$$
G _ { j } ( x _ { 0 } , x _ { 1 } , \ldots , x _ { q } ) \ = \ f _ { j } \bigl ( m _ { j } ^ { 0 } \odot x _ { 0 } , \ m _ { j } ^ { 1 } \odot x _ { 1 } , \ldots , \ m _ { j } ^ { q } \odot x _ { q } \bigr ) , \quad M \in \{ 0 , 1 \} ^ { D \times D \times ( q + 1 ) }\tag{1}
$$

where W is a standard D-dimensional Brownian motion and $\Sigma \in \mathbb { R } ^ { D \times D }$ is the diffusion matrix $( \Sigma \Sigma ^ { \top }$ being the noise covariance per unit time). M is a sparse binary tensor representing the dynamical drivers at timesteps $0 , \tau _ { 1 } , \dots , \tau _ { q } .$ We note $Z _ { t } : = \dot { ( } X _ { t } , X _ { t - \tau _ { 1 } } , \ldots , X _ { t - \tau _ { q } } \dot { ) }$ to simplify the notation when needed. We assume here that there is a maximum time lag $\tau _ { \mathrm { m a x } }$ for the lagged interactions, which happen within the timesteps $0 , \tau _ { 1 } , \dots , \tau _ { q } . \ f _ { j }$ is the nonlinear function giving the drift (the infinitesimal conditional mean rate of change) of $X _ { t , j }$ as a function of its drivers. We learn the functions $\widetilde { f } _ { j } \big ( \tilde { m } _ { j } ^ { 0 } \odot x _ { 0 } , \ \tilde { m } _ { j } ^ { 1 } \odot x _ { 1 } , \dots , \ \tilde { m } _ { j } ^ { q } \odot x _ { q } \big ) = : \widetilde { G } _ { j } \big ( x _ { 0 } , \dots , x _ { q } \big )$ to approximate this system.

In this section, we first show that the drivers and the dynamics, i.e. the matrix M and function $G ,$ , can in principle be recovered from a finite number of observations. We first give assumptions under which the SDDE (1) admits a unique strong solution, which is a prerequisite to identifiability. We then show that G and M are identifiable at the population level, either under a hard sparsity constraint (Theorem 1) or with an $L _ { 0 }$ penalty (Theorem 2). Under additional assumptions, we prove that the system is geometrically ergodic (Theorem 3) and provide a finite-sample identifiability theorem (Theorem 4). Proofs are given in Section A and we provide a proof map (Figure 1) to help the reader understand the construction of the proofs.

## 3.1 EXISTENCE AND UNIQUENESS OF A SOLUTION

## Because the system is not Markovia

n, the evolution of the system depends on the whole history of length $\tau _ { \mathrm { m a x } }$ rather than the current value only. This is represented by the segment space and the segment process (Mao, 2007; Hairer et al., 2011):

$$
\mathcal { C } : = C \big ( [ - \tau _ { \operatorname* { m a x } } , 0 ] ; \mathbb { R } ^ { D } \big ) , \qquad X _ { t } ^ { \mathrm { s e g } } ( s ) : = X _ { t + s } , \quad s \in [ - \tau _ { \operatorname* { m a x } } , 0 ] ,
$$

where C is the space of continuous functions from $[ - \tau _ { \operatorname* { m a x } } , 0 ]$ to $\mathbb { R } ^ { D }$ , equipped with the supremum norm $\| \varphi \| _ { \mathcal { C } } = \operatorname* { s u p } _ { s \in [ - \tau _ { \operatorname* { m a x } } , 0 ] } \| \varphi ( s ) \|$ for any given path $\varphi \in { \mathcal { C } } .$ . The segment process $( X _ { t } ^ { \mathrm { s e g } } ) _ { t \geq 0 }$ is then a Markov process on the (infinite-dimensional) space C. Accordingly, the initial condition is a C-valued random variable $X _ { 0 } ^ { \mathrm { s e g } }$ , i.e. all values observed in the previous time interval $[ - \tau _ { \operatorname* { m a x } } , 0 ]$

We impose standard assumptions on the underlying process: we require that the observed history does not anticipate future noise, that the noise acts in all dimensions (non-degenerate), and that the drift maintains finite energy along the trajectory, i.e. the system has finite energy.

Assumption 1 (Data-Generating Process). The true process X solves the SDDE Equation 1 on $\mathbb { R } ^ { D }$

(i) Initial conditions: The initial state $X _ { 0 } ^ { \mathrm { s e g } }$ is independent of W, and $\mathbb { E } \| X _ { 0 } ^ { \mathrm { s e g } } \| _ { c } ^ { 2 } < \infty$

(ii) Non-degenerate noise: The diffusion matrix $\Sigma \in \mathbb { R } ^ { D \times D }$ is invertible.

(iii) Finite energy: For every finite $\begin{array} { r } { T > 0 , \mathbb { E } \int _ { 0 } ^ { T } \| G ( X _ { t } , X _ { t - \tau _ { 1 } } , \ldots , X _ { t - \tau _ { q } } ) \| ^ { 2 } d t < \infty . } \end{array}$

To guarantee that the problem is well-posed, we need to control the drift locally and globally. Locally, it must be regular enough for trajectories to be unique, and globally it must not diverge in finite time to infinity. We thus impose the drift to be locally Lipschitz — a condition generally satisfied in most physical systems — and a radial Khasminskii condition (Khasminskii, 2012) which requires that, outside a ball whose radius can be as large as desired, the drift does not push the current state away from c faster than linearly and prevents explosiveness in finite time. It is much weaker than dissipativity, which asks the drift to actively pull the state back towards c when it is far away, like friction or damping in a physical system.

Assumption 2 (Regularity). To guarantee well-posedness and stability, we assume:

(i) Local Lipschitzness: $\{ f _ { j } \} _ { j \leq D }$ are locally Lipschitz continuous in all of their arguments jointly.

(ii) Radial Khasminskii condition: There exist constants $c \in \mathbb { R } ^ { D } , \ R > 0 ,$ , and $\beta > 0$ such that, writing $\boldsymbol { z } ~ = ~ ( x _ { 0 } , x _ { 1 } , \ldots , x _ { q } ) ~ \in ~ \mathbb { R } ^ { D ( q + 1 ) }$ and $c ^ { ( q ) } : = ( c , \dots , c ) \in \mathbb { R } ^ { D ( q + 1 ) }$ for the concatenation $o f q + 1$ copies ofc, the true drift G satisfies

$$
\langle x _ { 0 } - c , G ( z ) \rangle \leq \beta ( 1 + \| z - c ^ { ( q ) } \| ^ { 2 } ) \quad f o r a l l z w i t h \| z - c ^ { ( q ) } \| \geq R .\tag{2}
$$

These two sets of assumptions are permissive and encompass a wide range of SDDE-driven systems. We now establish the existence of a unique strong solution that does not explode in finite time, deferring the proof to Section $\mathrm { A } . 2$

Proposition 1 (Non-explosion and strong solution). Under Assumptions 1 and 2, the SDDE defined by Equation 1 has a unique strong solutionfor every initial condition with $\mathbb { E } \| X _ { 0 } ^ { \mathrm { s e g } } \| _ { c } ^ { 2 } < \infty$ without explosion i.e. it does not diverge to infinity in afinite amount oftime.

## 3.2 IDENTIFIABILITY OF THE DYNAMICAL DRIVERS

We aim to identify the drift on the regions of the state space that the process actually visits. Since the drift depends on the lagged state $Z _ { t }$ and we observe a trajectory over $[ 0 , T ]$ , the natural measure of “where the data lives” is the time-averaged law of $Z _ { t } .$ . We thus define the joint occupation measure of the states entering the drift: for a Borel set $A \subseteq \mathbb { R } ^ { D ( q + 1 ) }$ and a fixed $T > \tau _ { \mathrm { m a x } }$

$$
\bar { \mu } _ { T } ^ { q } ( A ) : = \frac { 1 } { T - { \tau _ { \operatorname* { m a x } } } } \int _ { \tau _ { \operatorname* { m a x } } } ^ { T } \mathbb { P } ( Z _ { t } \in A ) d t .\tag{3}
$$

A parameter of a model is identifiable if it is theoretically possible to uniquely determine it from observations. We thus consider an idealized learner: its model class must be able to represent the true system and minimize the population risk (Assumption 3). Although we never fully minimize the risk in practice, this is a standard assumption.

Assumption 3 (Model class and population risk minimization). Let $T > \tau _ { \mathrm { m a x } }$ be fixed.

(i) Model class: The model class H<sup>q</sup> consists of pairs $( \widetilde G , \tilde { M } )$ of candidate masks $\tilde { M } \in$ $\{ 0 , 1 \} ^ { D \times D \times ( q + 1 ) }$ and masked drifts $\widetilde { G } _ { j } ( x _ { 0 } , x _ { 1 } , \ldots , x _ { q } ) = \widetilde { f } _ { j } \big ( \tilde { m } _ { j } ^ { 0 } \odot x _ { 0 } , \tilde { m } _ { j } ^ { 1 } \odot x _ { 1 } , \ldots , \tilde { m } _ { j } ^ { q } \odot x _ { q } \big )$ with continuous ${ \widetilde { f } } _ { j } ,$ , and the same known lags $\tau _ { 1 } < \cdots < \tau _ { q }$ as the data-generating process. It contains the true pair $( G , M )$ , and every $\widetilde G \in \mathcal H ^ { q }$ has finite risk:

$$
\mathcal { R } _ { T } ( { \widetilde { G } } ) : = \mathbb { E } \int _ { \tau _ { \operatorname* { m a x } } } ^ { T } \left. { \widetilde { G } } ( Z _ { t } ) - G ( Z _ { t } ) \right. ^ { 2 } d t = ( T - \tau _ { \operatorname* { m a x } } ) \Vert { \widetilde { G } } - G \Vert _ { L ^ { 2 } ( { \frac { q } { \mu _ { T } ^ { q } } } ) } ^ { 2 } < \infty .\tag{4}
$$

(ii) Risk minimization: The learned pair $( \tilde { G } , \tilde { M } ) \in \mathcal { H } ^ { q }$ minimises $\mathcal { R } _ { T }$ over ${ \mathcal { H } } ^ { q } .$

When minimizing the risk, an input along which $G _ { j }$ is constant could be kept or dropped without changing the drift, we therefore require every true edge to actually vary (Assumption 4).

Assumption 4 (Faithfulness). For every $a \in \{ 0 , 1 , \ldots , q \}$ and every pair $( i , j )$ with $M _ { i j } ^ { a } = 1$ , there exist $z , z ^ { \prime } \in \operatorname { s u p p } ( \bar { \mu } _ { T } ^ { q } ) = \mathbb { R } ^ { D ( q + 1 ) }$ differing only in the i-th coordinate of the a-th block argument such that $G _ { j } ( z ) \neq G _ { j } ( z ^ { \prime } )$ . We show that supp $( \bar { \mu } _ { T } ^ { q } ) = \mathbb { R } ^ { D ( q + 1 ) }$ in Lemma 1.

Conversely, to avoid retaining spurious edges on which $\widetilde { G } _ { j }$ does not actually depend, we bound the number of learned edges. This entails that the sparsity level of the system is known a priori, a standard assumption in identifiability (Lachapelle et al., 2023), and we relax it in Section 3.3.

Assumption 5 (Structural Sparsity). The learned mask is no denser than the true mask:

$$
\begin{array} { r } { \| \tilde { M } \| _ { 0 } \leq \| M \| _ { 0 } . } \end{array}\tag{5}
$$

Theorem 1 (Identifiability). Under Assumptions 1–5 , G and M are identifiable:

$$
{ \widetilde { \cal G } } = { \cal G } o n { \mathbb { R } } ^ { D ( q + 1 ) } , \qquad { \tilde { \cal M } } = { \cal M } .\tag{6}
$$

The proof of this theorem is given in Section A.3.

## 3.3 L<sub>0</sub>-PENALIZED IDENTIFIABILITY

We now relax Assumption 5: instead of a hard constraint on the number of edges, it is possible to use a penalty if dropping a true edge to minimize sparsity leads to a higher increase in risk. Define, for each lag $a \in \{ 0 , \ldots , q \}$ and pair $( i , j )$ with $M _ { i j } ^ { a } = 1$ , the per-link signal strength

$$
s _ { i j } ^ { a } : = \left\| { \cal G } _ { j } - \Pi _ { - ( i , a ) } { \cal G } _ { j } \right\| _ { L _ { 2 } ( \bar { \mu } _ { T } ^ { q } ) } ,\tag{7}
$$

where $\Pi _ { - ( i , a ) }$ is the $L _ { 2 } ( \bar { \mu } _ { T } ^ { q } )$ )-orthogonal projection onto the closed subspace of functions of all coordinates except coordinate i of block $a .$ . In words, $s _ { i j } ^ { a }$ is the smallest error achievable by any model that ignores this input. Set

$$
s _ { \operatorname* { m i n } { \bf \lambda } } : = \operatorname* { m i n } { \textstyle \left\{ s _ { i j } ^ { a } : M _ { i j } ^ { a } = 1 \right\} } , \qquad d _ { \operatorname* { m a x } { \bf \lambda } } : = \operatorname* { m a x } _ { j } \sum _ { a = 0 } ^ { q } \lVert m _ { j } ^ { a } \rVert _ { 0 } .\tag{8}
$$

Define, for a constant $\lambda > 0$ , the penalized risk:

$$
J _ { \lambda } ( G , M ) = \mathcal { R } _ { T } ( G ) + \lambda \sum _ { a = 0 } ^ { q } \lVert M ^ { a } \rVert _ { 0 }\tag{9}
$$

Theorem 2 (Identifiability with the penalized objective). Let Assumptions $1 , 2$ and 3(i) hold. Suppose $M \ne 0$ and $s _ { \operatorname* { m i n } } > 0$ i.e. the system has at least one true driver, and every true driver has a measurable impact on the dynamics. This replaces Assumption 4 (faithfulness). If

$$
0 < \lambda < \frac { ( T - \tau _ { \operatorname* { m a x } } ) s _ { \operatorname* { m i n } } ^ { 2 } } { d _ { \operatorname* { m a x } } } ,\tag{10}
$$

then every minimizer of $J _ { \lambda }$ over $\mathcal { H } ^ { q }$ satisfies $\tilde { M } = M$ and ${ \widetilde { G } } = G o n \mathbb { R } ^ { D ( q + 1 ) }$ . Moreover, every feasible pair with $\tilde { M } \neq M$ satisfies

$$
J _ { \lambda } ( \tilde { G } , \tilde { M } ) - J _ { \lambda } ( G , M ) \ge \gamma ( \lambda ) : = \operatorname* { m i n } ( \lambda , ( T - \tau _ { \operatorname* { m a x } } ) s _ { \operatorname* { m i n } } ^ { 2 } - \lambda d _ { \operatorname* { m a x } } ) > 0
$$

Note that Assumption 5 (sparsity constraint) is not needed here.

## 3.4 ERGODICITY OF THE TRUE PROCESS

The results above assume that we can compute and minimize $\mathcal { R } _ { T }$ . In practice, however, we want to learn the system from discrete observations of a single long trajectory. To this aim, we need the system to be ergodic, so that trajectories converge to a unique statistical equilibrium. This is the natural notion of predictability for chaotic systems: individual trajectories cannot be forecast over long horizons, but their long-run statistics are well defined and do not depend on initial conditions.

Assumption 6 (Delay-adapted dissipativity). There exist $ { \mathcal { \textbf { c } } } \in  { \mathbb { R } } ^ { D }$ $R , \alpha , C _ { R } \geq 0$ and $\beta _ { 0 } , \dots , \beta _ { q } , \varepsilon _ { 1 } , \dots , \varepsilon _ { q } \ge 0$ such that, for all $\boldsymbol { z } = ( x _ { 0 } , \ldots , x _ { q } ) \in \mathbb { R } ^ { D ( q + 1 ) }$

$$
\begin{array} { r } { \langle x _ { 0 } - c , G ( z ) \rangle \leq \alpha - \beta _ { 0 } \| x _ { 0 } - c \| ^ { 2 } + \sum _ { a = 1 } ^ { q } \beta _ { a } \| x _ { a } - c \| ^ { 2 } \qquad i f \| x _ { 0 } - c \| \geq R , } \end{array}\tag{11}
$$

$$
\begin{array} { r } { \langle x _ { 0 } - c , G ( z ) \rangle \leq C _ { R } + \sum _ { a = 1 } ^ { q } \varepsilon _ { a } \| x _ { a } - c \| ^ { 2 } } \end{array}
$$

$$
i f \| x _ { 0 } - c \| \leq R ,\tag{12}
$$

$$
\begin{array} { r } { B : = \sum _ { a = 1 } ^ { q } ( \beta _ { a } + \varepsilon _ { a } ) < \beta _ { 0 } . } \end{array}\tag{13}
$$

Far from $c ,$ Equation 11 forces the current state to be pulled back at rate $\beta _ { 0 }$ , while past states may push outwards at rates $\beta _ { a }$ . Near $c ,$ Equation 12 only forces the drift to stay radially controlled: no restoring condition is imposed on the dynamics inside the ball, which can be chaotic. Strict domination (Equation 13) is the delay counterpart of stability: the restoring force of the present must dominate the destabilising influence of the past. This assumption holds for most damped systems with a bounded attractor (Lorenz, 1963; 1996), for any bounded delayed feedback (Mackey & Glass, 1977), and for any linear delayed feedback once the instantaneous damping is superlinear (e.g. a delayed oscillator model of ENSO (Suarez & Schopf, 1988)).

Theorem 3 (Ergodicity: existence, uniqueness, exponential convergence). Under Assumptions 1, 2 and 6, the segment process admits a unique invariant probability measure $\pi ^ { \mathrm { s e g } }$ on ${ \bar { \mathcal { C } } } .$ Write $P _ { t } ( \varphi , \cdot ) : = \mathcal { L } ( \breve { X } _ { t } ^ { \mathrm { s e g } } \mid \mathop { : } \chi _ { 0 } ^ { \mathrm { s e g } } = \varphi )$ and $V ( \varphi ) \bar { : } = \| \varphi - c \| _ { c } ^ { 2 }$ , where c denotes the constant segment with value c. The invariant measure satisfies

$$
\int _ { \mathcal { C } } \lVert \eta - c \rVert _ { \mathcal { C } } ^ { 2 p } \pi ^ { \mathrm { s e g } } ( d \eta ) < \infty , \qquad p \ge 1 a n i n t e g e r .\tag{14}
$$

There exist $C \geq 1$ and $\gamma > 0$ such that, $\forall \epsilon > 0 ,$ , the metric $d ( \varphi , \psi ) : = 1 \wedge \epsilon ^ { - 1 } \| \varphi - \psi \|$ <sub>C</sub> satisfies

$$
\mathcal { W } _ { d } \big ( P _ { t } ( \varphi , \cdot ) , \pi ^ { \mathrm { s e g } } \big ) \leq \| P _ { t } ( \varphi , \cdot ) - \pi ^ { \mathrm { s e g } } \| _ { \mathrm { T V } } \leq C e ^ { - \gamma t } \big ( 1 + V ( \varphi ) \big ) , \qquad t \geq 0 , \quad \varphi \in \mathcal { C } .\tag{15}
$$

In words, the system effectively forgets about its initial condition, and reaches exponentially quickly a regime that is uniquely determined and well behaved.

## 3.5 FINITE-SAMPLE IDENTIFIABILITY

This section characterizes identifiability when only a finite number of samples are available. We assume data is observed on a deterministic time grid $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { n } = T$ , with step sizes $\Delta _ { k } : = t _ { k + 1 } - t _ { k }$ bounded by $\Delta _ { \operatorname* { m a x } } : = \bar { \operatorname* { m a x } _ { k < n } \Delta _ { k } } \le 1$ . We also assume that the lagged connections happen within the observed time steps. Consequently, we restrict our analysis to the effective horizon $T _ { \circ } : = T - \tau _ { \mathrm { m a x } } > 0$ , denoting the sampled sequence as $Z _ { k } : = Z _ { t _ { k } }$ . All subsequent summations over $k < n$ apply strictly to the usable window $t _ { k } \ge \tau _ { \operatorname* { m a x } }$

Assumption 7 (Finite-sample model class). Let H<sup>q</sup> denote the restricted class of deterministic models derived from Assumption 3(i) satisfying the following two conditions:

(i) Local finite covering: For every radius $\rho \geq 1$ and tolerance $\epsilon > 0 ,$ we can select a finite number ofrepresentativefunctionsfrom within $\mathcal { H } _ { n } ^ { q }$ (an internal ϵ-net) such that every function in the class is within a maximum distance $o f \epsilon$ from at least one representative over the compact set $K _ { \rho } : = \{ z \in \mathbb { R } ^ { D ( q + 1 ) } : \| z - \check { c } ^ { ( q ) } \| \leq \rho \}$ . The minimal number of representatives required to cover the class this way is denoted by $N _ { n } ( \rho , \epsilon )$

(ii) Stationary envelope: The pointwise diameter of the class, defined as $\begin{array} { r l } { \mathcal { E } ( z ) } & { { } : = } \end{array}$ $\begin{array} { r } { \operatorname* { s u p } _ { \tilde { G } _ { 1 } , \tilde { G } _ { 2 } \in \mathcal { H } _ { n } ^ { q } } \Vert \tilde { G } _ { 1 } ( z ) - \tilde { G } _ { 2 } ( z ) \Vert } \end{array}$ , has a finite second moment under the stationary measure: $\begin{array} { r } { m _ { \mathcal { E } } ^ { 2 } : = \int \mathcal { E } ( z ) ^ { 2 } \mu _ { \infty } ^ { q } ( d z ) < \infty . } \end{array}$

This ensures that each bounded region can be covered by a finite number of candidate functions (i), and that candidates do not lead to wildly different equilibrium (ii).

The empirical risk is defined below, and we assume that a measurable global minimizer exists:

$$
\widehat { \mathcal { R } } _ { n } ( \tilde { G } ) : = \sum _ { k } \Delta _ { k } \left\| \frac { X _ { t _ { k + 1 } } - X _ { t _ { k } } } { \Delta _ { k } } - \tilde { G } ( Z _ { k } ) \right\| ^ { 2 }
$$

$$
( \widehat { G } , \widehat { M } ) \in \mathop { \mathrm { \arg ~ a r g ~ m i n } } _ { ( \tilde { G } , \tilde { M } ) \in \mathcal { H } _ { n } ^ { q } } \left\{ \widehat { \mathcal { R } } _ { n } ( \tilde { G } ) + \lambda \sum _ { a = 0 } ^ { q } \lVert \tilde { M } ^ { a } \rVert _ { 0 } \right\}\tag{16}
$$

Theorem 4 (Finite-sample exact support recovery $( L _ { 0 }$ estimator)). Let Assumptions $1 , 2 , 6$ and 7 hold, with $T > \tau _ { \mathrm { m a x } }$ . For a fixed $\delta \in ( 0 , 1 )$ , there exists a bound $\varepsilon _ { n } ( \delta ; \rho , \epsilon )$ dependent on two deterministic parameters $\rho$ and $\epsilon ,$ such that the uniform deviation is bounded with probability at least $1 - \delta \colon$

$$
\operatorname* { s u p } _ { \tilde { G } \in \mathcal { H } _ { n } ^ { q } } \Big | \Big [ \widehat { \mathcal { R } } _ { n } ( \tilde { G } ) - \widehat { \mathcal { R } } _ { n } ( G ) \Big ] - \mathcal { R } _ { T } ( \tilde { G } ) \Big | \leq \varepsilon _ { n } ( \delta ; \rho , \epsilon ) ,\tag{17}
$$

The explicit bound $\varepsilon _ { n } ( \delta ; \rho , \epsilon )$ and details on the selection ofρ and ϵ are given in (56)–(58).

Suppose further that the true mask $M \ne 0$ , the minimum signal strength $s _ { \operatorname* { m i n } } > 0 ,$ , and the penalty parameter $\lambda$ satisfies $0 < \lambda < T _ { \circ } s _ { \mathrm { m i n } } ^ { 2 } / d _ { \mathrm { m a x } }$ . This ensures a strictly positive margin $\gamma ( \lambda ) : =$ min $\{ \lambda , T _ { \circ } s _ { \mathrm { m i n } } ^ { 2 } - \lambda d _ { \mathrm { m a x } } \} > 0 .$

If the error bound is strictly less than the margin $( \varepsilon _ { n } ( \delta ; \rho , \epsilon ) < \gamma ( \lambda ) )$ , every measurable minimizer ofthe empirical objective achieves exact support recovery with probability at least $1 - \delta .$

$$
\mathbb { P } _ { \pi } \left( \widehat { M } ^ { a } = M ^ { a } f o r a l l a = 0 , \ldots , q , \quad \| \widehat { G } - G \| _ { L ^ { 2 } ( \mu _ { \infty } ^ { q } ) } ^ { 2 } \leq \frac { \varepsilon _ { n } ( \delta ; \rho , \epsilon ) } { T _ { \circ } } \right) \geq 1 - \delta\tag{18}
$$

Setting $\lambda = \lambda _ { 0 } T _ { \circ }$ (where $0 < \lambda _ { 0 } < s _ { \mathrm { m i n } } ^ { 2 } / d _ { \mathrm { m a x } } )$ allows us to identify the mask exactly and the drift up to a given level of error from finitely many observations. Achieving consistency inherently requires joint conditions on the observation duration, mesh size, and model class complexity; increasing the number of samples alone is not sufficient. The bound $\varepsilon _ { n }$ is explicit, but it should be read as a qualitative consistency statement: it contains factors $1 / \delta$ and a local envelope $E _ { \rho }$ that may grow with the radius $\rho ,$ itself growing with $T _ { \circ }$

## 3.6 IMPLEMENTATION AND OPTIMIZATION

The $L _ { 0 }$ -norm used in our identifiability theorems $( 1 , 4 )$ is not differentiable. To allow for differentiable optimization, we implement the relaxed $L _ { 0 }$ penalization proposed by Louizos et al. (2018)

and parametrize the matrix M as the sigmoid of differentiable logit parameters. The functions $\widetilde { f }$ are parametrized with simple MLPs, with tanh activation functions and an affine output layer. With discrete, regularly sampled observations, we train the model to predict $X _ { t + 1 } \mathrm { ~ - ~ } X _ { t }$ given $X _ { t } , X _ { t - 1 } , . . . , X _ { t - \tau }$ With irregular observations, it is possible to choose any numerical integrator, as when training Neural ODEs (Chen et al., 2018). Here, we use a simple Euler integrator, as all datasets in this paper have regular discrete temporal resolutions. We standardize each dimension by the estimated gradient mean and standard deviation. Otherwise, the method may learn to predict the mean for smooth dimensions, and mask off all of their potential drivers. We use the Adam optimizer (Kingma & Ba, 2015). We additionally propose to use gradient penalty (Gulrajani et al., 2017), to regularize our network and ensure that it does not overfit to spurious correlations. Full implementation details are provided in Section C.1. As the datasets considered in this paper are lowdimensional, and the architectures are relatively small, we run all experiments on CPUs. The code for implementing the methods can be found here https://anonymous.4open.science/ r/ODEDriverIdentifiability-C325/README.md.

Assumption 6 requires the true drift to be unbounded, and Assumption 7 requires the drift to belong to the model class. However, MLPs with tanh activation functions are bounded by construction. To satisfy both assumptions needed for finite-sample identifiability, we parametrize Ge as $\widetilde { G } _ { R } ( z ) =$ $\widetilde { f } ( z ) - K \cdot ( u - \Pi _ { R , c } ( u ) )$ , where R is an arbitrarily large radius parameter, c the center of the system, $K = \mathrm { d i a g } ( \kappa _ { 1 } , . . . , \kappa _ { D } )$ a diagonal matrix, and $\dot { \Pi } _ { R , c } \bar { ( } u )$ a projection onto the cube of center c and radius R. In practice, we can choose R large enough so that the cube encompasses all our training observations, c to be equal to zero. This correction is seen as a linear restoring force outside a cube and can be applied at inference only. More details on this modification are given in A.9.

## 4 DATASETS AND EXPERIMENTS

## 4.1 CAUSALDYNAMICS DATASET

The CausalDynamics dataset (Herdeanu et al., 2025) contains more than 14,000 simulated dynamical systems grouped in different classes: “Simple” three-dimensional chaotic dynamical systems (e.g. the Lorenz system), with or without added noise (high noise level, 2) and confounders (585 graphs); “Coupled” systems generated from simulated interactions between simple MLPs, of dimension 3–10, with added noise and with or without confounders, lagged interactions, or standardization (14096 graphs); and 2 simulated “Climate” datasets, one corresponding to a simplified coupled ocean-atmosphere model in dimension 10, and one simulating the decoupled El-Nino Southern Os-˜ cillation (ENSO), an important mode of variability of the climate system (12 graphs).

For each graph, 10 timeseries of 1000 timesteps are generated with different random seeds. The area under the receiver-operator curve (AUROC), area under the precision-recall curve (AUPRC) and structural Hamming distance (SHD) are computed for each time series, before being averaged across the system class (e.g. “Simple, Confounder, No noise”). Hyperparameter search is done for all evaluated methods by maximizing AUROC on the “Simple, No confounder, No noise, Lorenz84” graph, which is then excluded from the results. The same set of parameters is used for evaluation on all other graphs. We implement $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ and additional baselines: AGL (Bellot et al., 2022; Wu et al., 2024) and C-NODE (Aliee et al., 2025), which uses the $L _ { 1 }$ -penalty. We extend C-NODE to SDDEs similarly as Neural ODEs, by adding neural networks to capture lagged connections. All methods share the same core architecture. Hyperparameter search is conducted for each method separately, following the benchmark procedure. Details on the hyperparameter search, and the final set of parameters are reported in Section C.

## 4.2 THE DYSTS BENCHMARK

To evaluate the quality of the learned dynamics, we use the Dysts benchmark (Gilpin, 2021). It allows us to generate trajectories for low-dimensional chaotic systems (e.g. “Lorenz” system), which we use to evaluate L -NDDE, C-NODE, AGL, as well as PathReg, which uses $L _ { 0 }$ to regularize entire input-output paths, and regular NODEs. In total, we use the 40 systems for which the Jacobian is given by the dataset, which is needed for our metric computation (Section E.2). For each system, we generate one noisy training trajectory of 50 Lyapunov timesteps, with stochastic noise forcing of level 0.05, and 3 test trajectories of 250 Lyapunov timesteps, with different initial conditions. The Lyapunov timestep is the timescale over which a chaotic system loses its predictability (Lyapunov, 1992). These trajectories are long enough to densely cover the system’s space. We train each method on the training trajectory, generate 3 trajectories using the 3 test initial conditions, and compare the predicted trajectories to the true unseen test trajectories for evaluation.

We compute two metrics to evaluate short-term accuracy: the normalized root mean-square error (NRMSE) averaged over the first 5 Lyapunov timesteps, and the valid Lyapunov prediction time (VLPT), the number of Lyapunov timesteps before the predicted trajectories diverge from the true trajectory by more than 1 standardized unit. We compute two metrics to evaluate the reconstructed systems’ attractor geometry and chaoticity: the first Lyapunov exponent squared error (FLEE), indicative of the level of chaoticity of the system, and the squared error between the true and estimated Kaplan-Yorke dimension $( D _ { K Y }$ , Evans et al., 2000), a proxy for the system’s attractor dimension. We compute two metrics to evaluate the long-term statistics of the predicted trajectories: the logspectral distance (LSD), to evaluate whether the oscillation frequencies are accurately captured, and the state-space distance $( D _ { s t s p } ,$ Koppe et al., 2019), a divergence between the true and estimated equilibrium distribution. We report, for each metric and method, the median value across all datasets and the average fractional rank, to assess whether one method systematically outperforms another. More details on the metric computation are given in Section E.2.

We choose 5 systems at random for hyperparameter search and we select, for each method, the set of parameters that maximizes the VLPT before evaluating the method on the other systems. A detailed description of the hyperparameter search and final parameters is given in Section E.1.

## 5 RESULTS

## 5.1 CAUSALDYNAMICS BENCHMARK

Results on the CausalDynamics benchmark are reported in Table 1. $L _ { 0 }$ -NDDE achieves lower average rank on all three metrics. It generally has higher AUROC/AUPRC than competing methods on the “Simple”, “Coupled” and the “Climate” datasets, and consistently achieves low SHD, although the hyperparameters, only selected on the “Simple lorenz84” dataset, lead to too dense graphs on the “Coupled” datasets. Some results lead to “degenerate” metrics (e.g. AUROC of 0.50), as some methods sometimes fully sparsify or do not sparsify at all, and thus do not rank the edges. NGC and AGL achieve the lowest SHD on the MAOOAM dataset despite low AUROC, as they do not sparsify the learned graph. C-NODE performs well in general, despite not providing any identifiability guarantee. C-NODE also does not explicitly remove input features and even when a low weight is given to an input feature, the neural network can compensate for it.

## 5.2 DYSTS BENCHMARK

Results are reported in Table 2. $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ and NODE consistently rank better than other methods in all metrics except the NRMSE. The median $D _ { s t s p }$ is extremely high for AGL and PathReg, as this metric explodes when the two distributions are very different. It is an artefact of $D _ { s t s p } .$ . Additionally, we show predicted versus true trajectories for $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ , NODE, and C-NODE on 26 test systems, in Figure 2. Both NODE and $L _ { 0 } – \mathrm { \bar { N D D E } }$ are capable of modelling the systems very well.

To explain why NODE and $L _ { \mathrm { 0 } } – \mathbf { N D D E }$ outperform other methods, we report in Figure 3 the relationship between each metric and the SHD (upper half) and training MSE (lower half). We run a Kendall-τ test (Kendall, 1938), standard for non-Gaussian low-samples regimes, to measure the strength of association between the two sets of values. We find that $D _ { K Y } , D _ { s t s p }$ and VLPT have a statistically significant association with both SHD and training MSE, indicating that on this bench mark, methods that better learn the drivers, and better fit the training data perform better. C-NODE, PathReg, and AGL all apply a regularization on the weights of the neural network, preventing these methods from reaching low training error. PathReg tends to sparsify specific input-output paths without masking off input features, thus not performing input feature selection. This is also why we do not evaluate it on CausalDynamics. With no weight penalty, neural ODE achieves low training error and thus good reconstruction performance. $L _ { 0 }$ -penalty does not penalize the weights of the network but instead turns the inputs on/off, allowing $L _ { 0 } { \mathrm { - N D D E } }$ to achieve low train error while masking off irrelevant input features. With the lowest SHD, $L _ { 0 }$ -NDDE thus outperforms other methods.

<table><tr><td rowspan="2"></td><td rowspan="2">Experiment</td><td colspan="3">Simple</td><td colspan="5">Coupled</td><td colspan="2">Climate</td><td rowspan="2">Avg. rank</td></tr><tr><td>Default</td><td>Confounder</td><td>Noise</td><td>Default</td><td>Noise</td><td>Confounder</td><td>Time-lag</td><td>Standardize</td><td>MAOOAM</td><td>ENSO</td></tr><tr><td rowspan="9"></td><td>PCMCI+</td><td>41.04</td><td>23.02</td><td>47.64</td><td>224.80</td><td>183.90</td><td>324.63</td><td>327.72</td><td>228.32</td><td>80.00</td><td>529.36</td><td>7.30</td></tr><tr><td>F-PCMCI</td><td>35.30</td><td>21.07</td><td>45.09</td><td>192.90</td><td>149.60</td><td>195.74</td><td>350.61</td><td>201.79</td><td>130.00</td><td>530.27</td><td>5.80</td></tr><tr><td>NGC</td><td>28.91</td><td>19.96</td><td>28.84</td><td>840.95</td><td>842.55</td><td>670.53</td><td>793.67</td><td>840.26</td><td>31.00</td><td>337.09</td><td>7.90</td></tr><tr><td>CUTS+</td><td>48.11</td><td>24.04</td><td>61.32</td><td>152.00</td><td>150.50</td><td>272.68</td><td>247.22</td><td>310.63</td><td>130.00</td><td>608.73</td><td>7.50</td></tr><tr><td>RCD</td><td>61.85</td><td>26.74</td><td>61.38</td><td>157.05</td><td>155.65</td><td>136.53</td><td>201.11</td><td>159.84</td><td>130.00</td><td>665.36</td><td>6.80</td></tr><tr><td>GRaSP</td><td>59.04</td><td>27.09</td><td>60.18</td><td>215.94</td><td>842.55</td><td>136.00</td><td>793.67</td><td>840.26</td><td>126.00</td><td>666.27</td><td>9.70</td></tr><tr><td>AGL</td><td>26.60</td><td>16.96</td><td>21.28</td><td>319.04*</td><td>222.41*</td><td>246.37*</td><td>315.54*</td><td>312.31*</td><td>30.00*</td><td>337.09</td><td>5.55</td></tr><tr><td>C-NODE (L1)</td><td>24.07</td><td>17.10</td><td>20.50</td><td>253.91*</td><td>176.30*</td><td>179.58*</td><td>273.57*</td><td>258.09</td><td>42.00</td><td>335.36</td><td>4.50</td></tr><tr><td>L0-NDDE (Ours)</td><td>26.42</td><td>16.25</td><td>23.67</td><td>205.35</td><td>144.20</td><td>143.33</td><td>222.34</td><td>212.11</td><td>40.00</td><td>325.54</td><td>2.50</td></tr><tr><td></td><td>PCMCI+</td><td>.52 / .71</td><td>.49 / .59</td><td>.50 / .69</td><td>.67 / .25</td><td>64 / .25</td><td>.58 / .20</td><td>.58 / .24</td><td>.69 / .27</td><td>.69 / .88</td><td>.57 /.70</td><td>4.10 / 5.25</td></tr><tr><td>AURC</td><td>F-PCMCI</td><td>.51 /.70</td><td>.50/.59</td><td>.52/.70</td><td>.67 / .27</td><td>.57/.21</td><td>.55 / .19</td><td>.59 / .24</td><td>.68 / .28</td><td>.50 / .81</td><td>.57 /.70</td><td>4.80 / 6.05</td></tr><tr><td></td><td>NGC</td><td>.50 / .69</td><td>.50 / .58</td><td>.50 / .68</td><td>.50/.15</td><td>.50 / .15</td><td>.50 / .16</td><td>.50 / .20</td><td>.50/.15</td><td>.50 / .81</td><td>.50 / .67</td><td>9.85 / 10.90</td></tr><tr><td></td><td>CUTS+</td><td>.50 / .69</td><td>.50 / .58</td><td>.50 / .68</td><td>.50 /.15</td><td>.50 / .15</td><td>.49 /.16</td><td>.50 / .20</td><td>.50 / .15</td><td>.50 / .81</td><td>.50 / .67</td><td>10.10 / 10.90</td></tr><tr><td></td><td>RCD</td><td>.50 / .69</td><td>.50 / .58</td><td>.50 / .68</td><td>.50/.15</td><td>.50/.15</td><td>.51 /.18</td><td>.50 / .20</td><td>.50/.16</td><td>.50 / .81</td><td>.50/ .67</td><td>9.55 / 10.25</td></tr><tr><td></td><td>GRaSP</td><td>.52 / .71</td><td>.55 / .64</td><td>.49 / .68</td><td>.49 /.15</td><td>.50 / .50</td><td>.50 /.17</td><td>.50 / .50</td><td>.50 / .50</td><td>.48 / .81</td><td>.50 / .50</td><td>9.60 / 6.70</td></tr><tr><td></td><td>AGL</td><td>.52/.71*</td><td>56/.64</td><td>.53 / .74*</td><td>.53 /.33*</td><td>.53 /.33*</td><td>.54 / .35*</td><td>.52 /.35*</td><td>.54 / .34*</td><td>.56 / .86*</td><td>.54 / .69*</td><td>5.80 / 4.35</td></tr><tr><td></td><td>C-NODE (L1)</td><td>.64 / .79</td><td>.54 / .65</td><td>.61 / .79*</td><td>.64 / .43</td><td>.63 / .43</td><td>.67 / .47</td><td>.59 /.43</td><td>.68 / .47</td><td>.88 / .97</td><td>.49 / .66*</td><td>3.45 / 3.35</td></tr><tr><td></td><td>L0-NDDE (Ours)</td><td>.64 /.80</td><td>.58 /.66</td><td>.63 / .80</td><td>.64 / .43</td><td>.65 / .44</td><td>.67 /.47</td><td>.60 / .43</td><td>.66 /.45</td><td>.86 / .96</td><td>.58 / .75</td><td>1.75 / 2.10</td></tr></table>

Table 1: L -NDDE outperforms competing methods on the CausalDynamics benchmark. SHD (lower is better) and AUROC / AUPRC (higher is better) are reported for $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ (Ours), AGL, C-NODE and the benchmark causal discovery methods. The best values are bold, the second best values are underlined. <sup>∗</sup> indicates statistical significance between AGL or C-NODE and $L _ { 0 } – \mathbf { N D D E } .$ calculated with a Welch’s t-test (p-value 0.05). The benchmark only provides the mean values for the causal discovery methods, and we thus do not run statistical significance tests for these methods. For AUROC and AUPRC, statistical significance results always match, so we report only one <sup>∗</sup> for each pair. The last column reports the average rank for each method and metric. For readability, methods that never score best are not shown here, but in the Appendix (Table 4)

<table><tr><td rowspan="2">Method</td><td colspan="2">NRMSE (↓)</td><td colspan="2">VLPT (↑)</td><td colspan="2">FLEE (↓)</td><td colspan="2"> $D _ { K Y } \left( \downarrow \right)$ </td><td colspan="2">LSD (↓)</td><td colspan="2"> $D _ { s t s p } \left( \downarrow \right)$ </td><td colspan="2">SHD (↓)</td></tr><tr><td>Med.</td><td>Rank</td><td>Med.</td><td>Rank</td><td>Med.</td><td>Rank</td><td>Med.</td><td>Rank</td><td>Med.</td><td>Rank</td><td>Med.</td><td>Rank</td><td>Med.</td><td>Rank</td></tr><tr><td>AGL</td><td>1.15</td><td>2.69</td><td>2.85</td><td>3.14</td><td>0.25</td><td>3.47</td><td>0.88*</td><td>3.82</td><td>14.71*</td><td>2.82</td><td> $3 \mathrm { e } ^ { 6 * }$ </td><td>4.37</td><td>0.44*</td><td>4.46</td></tr><tr><td>PathReg</td><td>1.19</td><td>2.49</td><td>4.33</td><td>3.24</td><td>0.25</td><td>3.78</td><td>0.89*</td><td>3.9</td><td>16.99</td><td>3.02</td><td>3e6*</td><td>4.62</td><td>0.22</td><td>2.70</td></tr><tr><td>C-NODE</td><td>1.53</td><td>4.05</td><td>4.61</td><td>3.39</td><td>0.20</td><td>3.00</td><td>0.72*</td><td>3.17</td><td>37.79*</td><td>4.08</td><td>34.60*</td><td>2.86</td><td>0.17</td><td>2.67</td></tr><tr><td>NODE</td><td>1.20</td><td>2.94</td><td>5.12</td><td>2.63</td><td>0.16</td><td>2.51</td><td>0.076</td><td>2.17</td><td>13.99</td><td>2.57</td><td>4.29</td><td>1.62</td><td>0.22</td><td>2.70</td></tr><tr><td> $L _ { 0 } – \mathbf { N D D E }$ </td><td>1.19</td><td>2.83</td><td>6.54</td><td>2.60</td><td>0.16</td><td>2.25</td><td>0.075</td><td>1.94</td><td>14.75</td><td>2.50</td><td>2.50</td><td>1.54</td><td>0.11</td><td>2.45</td></tr></table>

Table 2: $L _ { \mathrm { 0 } } { \bf - N D E }$ outperforms other feature selection methods. $L _ { 0 } – \mathbf { N D D E }$ ranks best in five of the metrics. The difference with NODE is non significant, as NODE also performs very well, but $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ consistently ranks better than the other methods performing input feature selection, $( D _ { K Y } )$ $( D _ { s t s p } )$ ∗ especially on attractor dimension reconstruction and convergence in distribution indicates statistical significance with the $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ metric, obtained using a permutation test (p value 0.05). For each metric, best results are bolded and second best are underlined.

## 6 DISCUSSION

We hope that the theoretical framework proposed in this paper lays the foundations for the development of robust methods for modelling lagged dynamical systems. One limitation of our method is that we can only optimize for an approximate $L _ { \mathrm { 0 ^ { - } P } } \mathrm { e n a l t y }$ , as the true $L _ { 0 }$ -loss is not differentiable, although existing methods also rely on approximations. We show empirical improvements on two benchmarks that evaluate the methods mostly on SDE-driven systems as, to the best of our knowledge, no comprehensive benchmark exists for evaluating lagged systems modelling.

Despite these promising results, future work is needed to apply this method to real physical systems. Our method’s computational requirements scale quadratically with the input dimension. To apply it to high-dimensional systems, it would need to be combined with an identifiable dimensionality reduction technique (Hı zlıet al., 2025). Our method learns probabilities for the input-output mapping, but the learned drift is deterministic. Extending our method to be fully probabilistic could allow us to better represent uncertainty during long-term autoregressive rollouts. We see several exciting applications of our method. Identifying correct drivers in dynamical systems could open the door to scientific discovery across several domains. Our method could also be integrated into physical models to robustly parametrize SDDE-driven subprocesses. It could also be used to stabilize gradient estimates in differentiable models of chaotic systems (Metz et al., 2022).

## ACKNOWLEDGEMENTS

We thank Julia Kaltenborn and David Rolnick for their feedback on the manuscript. We thank Benjamin Herdeanu, Carla Roesch and Juan Nathaniel for helping us run the CausalDynamics bench mark. This research was supported in part by the Canada CIFAR AI Chairs program.

This research was enabled in part by compute resources provided by Mila (mila.quebec).

## AI USE STATEMENT

In this work, we used generative AI tools for providing critical ingredients for proving mathematical claims, assisting in the writing of proofs, and implementing methods. We have not used generative AI tools for generating synthetic data sets, helping develop theoretical models or conceptual frameworks, formulating mathematical claims, proposing or refining hypotheses, designing or providing feedback on research methodology or experiments, assisting with translation, cleaning and reformatting datasets, supporting qualitative and thematic data analysis, or interpreting results. Additionally, we used generative AI tools to identify relevant literature. We have reviewed all AI-assisted work. For the mathematical proofs, we first sketched the proofs ourselves before asking an LLM to review them and suggest corrections or improvements. We have reviewed every suggestion manually before incorporating them into the final version of the proofs. For implementing methods, we carefully verified and tested all code used for the experiments in this paper. All the identified literature and references were manually checked, and we read every paper that our work builds upon or takes inspiration from; e.g., we did not rely on an LLM-generated summary of other research papers to design this paper, or to decide whether to include or not include references. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

The code to implement the methods has been shared in Section 3.6. This code uses poetry to manage the environment. Detailed parameters, as well as hyperparameter search procedures are given in the main text as well as the appendix. As mentioned in the main text, all experiments have been run on CPU, allowing for easy reproducibility i.e. without intensive computational requirements. The metric computation is described in details in the appendix (Section E.2). The data used in this paper is taken from peer-reviewed, publicly available benchmarks.

## REFERENCES

Hananeh Aliee, Till Richter, Mikhail Solonin, Ignacio Ibarra, Fabian Theis, and Niki Kilbertus. Sparsity in continuous-depth neural networks. In Advances in Neural Information Processing Systems, volume 35, pp. 901–914, 2022. doi: 10.52202/068431-0066.

Hananeh Aliee, Fabian J. Theis, and Niki Kilbertus. Beyond predictions in neural odes: Identification and interventions, 2025. URL https://arxiv.org/abs/2106.12430.

Alexis Bellot, Kim Branson, and Mihaela van der Schaar. Neural graphical modelling in continuoustime: consistency guarantees and algorithms. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=SsHBkfeRF9L.

Richard C. Bradley. Basic Properties of Strong Mixing Conditions. A Survey and Some Open Questions. Probability Surveys, 2:107 – 144, 2005. doi: 10.1214/154957805100000104. URL https://doi.org/10.1214/154957805100000104.

Philippe Brouillard, Sebastien Lachapelle, Julia Kaltenborn, Yaniv Gurwicz, Dhanya Sridhar, Alexandre Drouin, Peer Nowack, Jakob Runge, and David Rolnick. Learning a spatial partitioning and its causal relations from temporal data. In Bijan Mazaheri and Niels Richard Hanson (eds.), Proceedings of the Fifth Conference on Causal Learning and Reasoning, volume 323 of Proceedings of Machine Learning Research, pp. 547–592. PMLR, 06–08 Apr 2026. URL https://proceedings.mlr.press/v323/brouillard26a.html.

Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15):3932–3937, 2016. doi: 10.1073/pnas.1517384113. URL https://www.pnas.org/doi/abs/10.1073/pnas.1517384113.

Kurt Butler, Daniel Waxman, and Petar M Djuric. Tangent space causal inference: Leveraging vector´ fields for causal discovery in dynamical systems. Advances in Neural Information Processing Systems, 37:120078–120102, 2024.

Luca Castri, Sariah Mghames, Marc Hanheide, and Nicola Bellotto. Enhancing causal discovery from robot sensor data in dynamic scenarios. In Conference on Causal Learning and Reasoning, pp. 243–258. PMLR, 2023.

Mickael D. Chekroun and Honghu Liu. Effective reduced models from delay differential¨ equations: Bifurcations, tipping solution paths, and enso variability. Physica D: Nonlinear Phenomena, 460:134058, 2024. ISSN 0167-2789. doi: https://doi.org/10.1016/j.physd. 2024.134058. URL https://www.sciencedirect.com/science/article/pii/ S0167278924000095.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31. Cur ran Associates, Inc., 2018. URL https://proceedings.neurips.cc/paper\_files/ paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf.

Yuxiao Cheng, Lianglong Li, Tingxiong Xiao, Zongren Li, Jinli Suo, Kunlun He, and Qionghai Dai. Cuts+: High-dimensional causal discovery from irregular time-series. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 38, pp. 11525–11533, 2024.

Yu. A. Davydov. Mixing conditions for markov chains. Theory of Probability & Its Applications, 18 (2):312–328, 1974. doi: 10.1137/1118033. URL https://doi.org/10.1137/1118033.

J. R. Dorfman. An Introduction to Chaos in Nonequilibrium Statistical Mechanics. 1999.

Emilien Dupont, Arnaud Doucet, and Yee Whye Teh. Augmented neural odes. Advances in neural information processing systems, 32, 2019.

Mari Dahl Eggen and Alise Danielle Midtfjord. Delay-sde-net: A deep learning approach for time series modelling with memory and uncertainty estimates, 2023. URL https://arxiv.org/ abs/2303.08587.

Denis J. Evans, E. G. D. Cohen, Debra J. Searles, and F. Bonetto. Note on the kaplan–yorke dimension and linear transport coefficients. Journal of Statistical Physics, 101, 2000. doi: 10.1023/A:1026449702528.

Michael Ghil and Valerio Lucarini. The physics of climate variability and climate change. Rev. Mod. Phys., 92:035002, Jul 2020. doi: 10.1103/RevModPhys.92.035002. URL https:// link.aps.org/doi/10.1103/RevModPhys.92.035002.

Michael Ghil, Mickael D. Chekroun, and Eric Simonnet. Climate dynamics and fluid mechanics:¨ Natural variability and related uncertainties. Physica D: Nonlinear Phenomena, 237(14):2111– 2126, 2008. ISSN 0167-2789. doi: https://doi.org/10.1016/j.physd.2008.03.036. URL https: //www.sciencedirect.com/science/article/pii/S0167278908001139. Euler Equations: 250 Years On.

William Gilpin. Chaos as an interpretable benchmark for forecasting and data-driven modelling. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=enYjtbjYJrf.

C. W. J. Granger. Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37(3):424–438, 1969. ISSN 00129682, 14680262. URL http://www.jstor. org/stable/1912791.

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. Improved training of wasserstein gans. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, pp. 5769–5779, Red Hook, NY, USA, 2017. Curran Associates Inc. ISBN 9781510860964.

M. Hairer, J. C. Mattingly, and M. Scheutzow. Asymptotic coupling and a general form of harris theorem with applications to stochastic delay equations. Probability Theory and Related Fields, 149(1):223–259, 2011. doi: 10.1007/s00440-009-0250-6. URL https://doi.org/10. 1007/s00440-009-0250-6.

Aristide Halanay. Differential Equations: Stability, Oscillations, Time Lags, volume 23 of Mathematics in Science and Engineering. Academic Press, 1966.

Jack K. Hale and Sjoerd M. Verduyn Lunel. Introduction to Functional Differential Equations. Applied Mathematical Sciences. Springer, New York, NY, 1 edition, 1993. ISBN 978-0-387- 94076-2. doi: 10.1007/978-1-4612-4342-7.

Benjamin Herdeanu, Juan Nathaniel, Carla Roesch, Jatan Buch, Gregor Ramien, Johannes Haux, and Pierre Gentine. Causaldynamics: A large-scale benchmark for structural discovery of dynamical causal models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference. Curran Associates, Inc., 2025. doi: 10.52202/085713-1349. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ 39a3f7dc00e4723a5ca7808b04fc4d4f-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

C¸ aglar Hı zlı, C¸ a˘ gatay Yı ldız, Matthias Bethge, ST John, and Pekka Marttinen. Identifying latent˘ state transitions in non-linear dynamical systems. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 71286–71315, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ b192f601432dff4e6777d2d06d52cfa5-Paper-Conference.pdf.

Sebastian Hickman, Ilija Trajkovic, Julia Kaltenborn, Francis Pelletier, Alexander T Archibald,´ Yaniv Gurwicz, Peer Nowack, David Rolnick, and Julien Boussard. Causal climate emulation with bayesian filtering. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=YxPI1c5e09.

Aapo Hyvarinen, Kun Zhang, Shohei Shimizu, and Patrik O. Hoyer. Estimation of a structural¨ vector autoregression model using non-gaussianity. Journal of Machine Learning Research, 11 (56):1709–1731, 2010. URL http://jmlr.org/papers/v11/hyvarinen10a.html.

Kiyosi Ito. Stochastic integral.ˆ Proceedings of the Imperial Academy, 20(8):519 – 524, 1944. doi: 10.3792/pia/1195572786. URL https://doi.org/10.3792/pia/1195572786.

Kiyosi Ito.ˆ On Stochastic Differential Equations. Number 4 in Memoirs of the American Mathematical Society. American Mathematical Society, 1951. doi: 10.1090/memo/0004. URL https://doi.org/10.1090/memo/0004.

Ioannis Karatzas and Steven E. Shreve. Brownian Motion and Stochastic Calculus. Graduate Texts in Mathematics. Springer New York, NY, 2 edition, 1991. ISBN 978-0-387-97655-6. doi: 10. 1007/978-1-4612-0949-2.

Andrew Keane, Bernd Krauskopf, and Claire M. Postlethwaite. Climate models with delay differ ential equations. Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(11):114309, 10 2017. ISSN 1054-1500. doi: 10.1063/1.5006923. URL https://doi.org/10.1063/1. 5006923.

M. G. Kendall. A new measure of rank correlation. Biometrika, 30(1-2):81–93, 06 1938. ISSN 0006-3444. doi: 10.1093/biomet/30.1-2.81. URL https://doi.org/10.1093/biomet/ 30.1-2.81.

Rafail Khasminskii. Stochastic Stability of Differential Equations. Stochastic Modelling and Applied Probability. Springer Berlin, Heidelberg, 2 edition, 2012. ISBN 978-3-642-23279-4. doi: 10.1007/978-3-642-23280-0.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015. URL https://arxiv.org/abs/ 1412.6980.

Georgia Koppe, Hazem Toutounji, Peter Kirsch, Stefanie Lis, and Daniel Durstewitz. Identifying nonlinear dynamical systems via generative recurrent neural networks with applications to fmri. PLOS Computational Biology, 15(8):1–35, 08 2019. doi: 10.1371/journal.pcbi.1007263. URL https://doi.org/10.1371/journal.pcbi.1007263.

Sebastien Lachapelle, Tristan Deleu, Divyat Mahajan, Ioannis Mitliagkas, Yoshua Bengio, Simon Lacoste-Julien, and Quentin Bertrand. Synergies between disentanglement and sparsity: Generalization and identifiability in multi-task learning. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 18171–18206. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/ v202/lachapelle23a.html.

Wai-Yin Lam, Bryan Andrews, and Joseph Ramsey. Greedy relaxations of the sparsest permutation algorithm. In James Cussens and Kun Zhang (eds.), Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence, volume 180 of Proceedings of Machine Learning Research, pp. 1052–1062. PMLR, 01–05 Aug 2022. URL https://proceedings.mlr. press/v180/lam22a.html.

Phillip Lippe, Sara Magliacane, Sindy Lowe, Yuki M Asano, Taco Cohen, and Stratis Gavves.¨ CITRIS: Causal identifiability from temporal intervened sequences. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 13557–13603. PMLR, 17–23 Jul 2022. URL https: //proceedings.mlr.press/v162/lippe22a.html.

Edward N. Lorenz. Deterministic nonperiodic flow. Journal of the Atmospheric Sciences, 20(2): 130–141, 1963.

Edward N. Lorenz. Predictability: A problem partly solved. In Proceedings of the Seminar on Predictability, volume 1, pp. 1–18, Reading, UK, 1996. ECMWF.

Christos Louizos, Max Welling, and Diederik P. Kingma. Learning sparse neural networks through l regularization. In International Conference on Learning Representations, 2018. URL https: //openreview.net/forum?id=H1Y8hhg0b.

A. M. Lyapunov. The general problem of the stability of motion. International Journal ofControl, 55(3):531–534, 1992. doi: 10.1080/00207179208934253. URL https://doi.org/10. 1080/00207179208934253.

Michael C. Mackey and Leon Glass. Oscillation and chaos in physiological control systems. Science, 197(4300):287–289, 1977. ISSN 00368075, 10959203. URL http://www.jstor.org/ stable/1744526.

Takashi Nicholas Maeda and Shohei Shimizu. Rcd: Repetitive causal discovery of linear nongaussian acyclic models with latent confounders. In Silvia Chiappa and Roberto Calandra (eds.), Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pp. 735–745. PMLR, 26–28 Aug 2020. URL https://proceedings.mlr.press/v108/maeda20a.html.

Xuerong Mao. Stochastic Differential Equations and Applications. Woodhead Publishing, 2 edition, 2007. ISBN 978-1-904275-34-3. doi: 10.1533/9780857099402. URL https://www. sciencedirect.com/science/book/9781904275343.

Hiroki Masuda. Ergodicity and exponential β-mixing bounds for multidimensional diffusions with jumps. Stochastic Processes and their Applications, 117(1):35–56, 2007. ISSN 0304-4149. doi: https://doi.org/10.1016/j.spa.2006.04.010. URL https://www.sciencedirect. com/science/article/pii/S0304414906000524.

Luke Metz, C. Daniel Freeman, Samuel S. Schoenholz, and Tal Kachman. Gradients are not all you need, 2022. URL https://arxiv.org/abs/2111.05803.

S. E. A. Mohammed. The infinitesimal generator of a stochastic functional differential equation. In W.N. Everitt and B.D. Sleeman (eds.), Ordinary and Partial Differential Equations, pp. 529–538, Berlin, Heidelberg, 1982. Springer Berlin Heidelberg. ISBN 978-3-540-39561-4.

Meike Nauta, Doina Bucur, and Christin Seifert. Causal discovery with attention-based convolutional neural networks. Machine Learning and Knowledge Extraction, 1(1):312–340, 2019. ISSN 2504-4990. doi: 10.3390/make1010019. URL https://www.mdpi.com/2504-4990/1/ 1/19.

David Nualart. The Malliavin Calculus and Related Topics. Probability and Its Applications. Springer Berlin, Heidelberg, 2 edition, 2006. ISBN 978-3-540-28328-7. doi: 10.1007/ 3-540-28329-3.

Yongkyung Oh, Seungsu Kam, Dongyoung Lim, and Sungil Kim. Modeling irregular astronomical time series with neural stochastic delay differential equations. In Proceedings ofthe 34th ACM International Conference on Information and Knowledge Management, CIKM ’25, pp. 5068–5073, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400720406. doi: 10.1145/3746252.3760805. URL https://doi.org/10.1145/3746252.3760805.

Roxana Pamfil, Nisara Sriwattanaworachai, Shaan Desai, Philip Pilgerstorfer, Konstantinos Georgatzis, Paul Beaumont, and Bryon Aragam. Dynotears: Structure learning from time-series data. In International conference on artificial intelligence and statistics, pp. 1595–1605. Pmlr, 2020.

Jeongjin Park, Nicole Tianjiao Yang, and Nisha Chandramoorthy. When are dynamical systems learned from time series data statistically accurate? In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 43975–44008. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-1396. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/4dc57702c987e1e72f0dd2921edb5ded-Paper-Conference.pdf.

Maziar Raissi, Paris Perdikaris, and George E. Karniadakis. Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations, 2017. URL https://arxiv. org/abs/1711.10561.

J. O. Ramsay, G. Hooker, D. Campbell, and J. Cao. Parameter estimation for differential equations: a generalized smoothing approach. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 69(5):741–796, 2007. doi: https://doi.org/10.1111/j.1467-9868.2007. 00610.x. URL https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j. 1467-9868.2007.00610.x.

Jakob Runge. Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets, 2020.

M. Scheutzow. Qualitative behaviour of stochastic delay equations with a bounded memory. Stochastics, 12(1):41–80, 1984. doi: 10.1080/17442508408833294. URL https://doi. org/10.1080/17442508408833294.

Ch. Skokos. The Lyapunov Characteristic Exponents and Their Computation, pp. 63–135. Springer Berlin Heidelberg, Berlin, Heidelberg, 2010. ISBN 978-3-642-04458-8. doi: 10.1007/ 978-3-642-04458-8 2. URL https://doi.org/10.1007/978-3-642-04458-8\_2.

Daniel W. Stroock and S. R. Srinivasa Varadhan. On the support of diffusion processes with applications to the strong maximum principle. In Proceedings of the Sixth Berkeley Symposium on Mathematical Statistics and Probability, volume 3, pp. 333–359. University of California Press, 1972. URL https://api.semanticscholar.org/CorpusID:35508438.

Max J. Suarez and Paul S. Schopf. A delayed action oscillator for enso. Journal of Atmospheric Sciences, 45(21):3283 – 3287, 1988. doi: 10.1175/1520-0469(1988)045⟨3283:ADAOFE⟩2. 0.CO;2. URL https://journals.ametsoc.org/view/journals/atsc/45/21/ 1520-0469\_1988\_045\_3283\_adaofe\_2\_0\_co\_2.xml.

George Sugihara, Robert May, Hao Ye, Chih hao Hsieh, Ethan Deyle, Michael Fogarty, and Stephan Munch. Detecting causality in complex ecosystems. Science, 338(6106):496–500, 2012. doi: 10.1126/science.1227079. URL https://www.science.org/doi/abs/10.1126/ science.1227079.

Nicholas Tagliapietra, Katharina Ensinger, Christoph Zimmer, and Osman Mian. Causal structure learning for dynamical systems with theoretical score analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 36740–36748, 2026.

Alex Tank, Ian Covert, Nicholas Foti, Ali Shojaie, and Emily B. Fox. Neural granger causality. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8):4267–4279, 2022. doi: 10.1109/TPAMI.2021.3065601.

Silviu-Marian Udrescu, Andrew Tan, Jiahai Feng, Orisvaldo Neto, Tailin Wu, and Max Tegmark. Ai feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 4860–4871. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/33a854e247155d590883b93bca53848a-Paper.pdf.

Peter Walters. An Introduction to Ergodic Theory. Springer New York,, 1999. URL https: //link.springer.com/book/9780387951522.

Philippe Wenk, Gabriele Abbati, Michael A Osborne, Bernhard Scholkopf, Andreas Krause, and¨ Stefan Bauer. Odin: Ode-informed regression for parameter and state inference in timecontinuous dynamical systems. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 6364–6371, 2020.

Fan Wu, Woojin Cho, David Korotky, Sanghyun Hong, Donsub Rim, Noseong Park, and Kookjin Lee. Identifying contemporaneous and lagged dependence structures by promoting sparsity in continuous-time neural networks. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, CIKM ’24, pp. 2534–2543, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704369. doi: 10.1145/3627673. 3679751. URL https://doi.org/10.1145/3627673.3679751.

Weiran Yao, Yuewen Sun, Alex Ho, Changyin Sun, and Kun Zhang. Learning temporally causal latent processes from general temporal data. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=RDlLMjLJXdq.

Yuan Yin, Vincent Le Guen, Jer´ emie Dona, Emmanuel De B´ ezenac, Ibrahim Ayed, Nicolas Thome,´ and Patrick Gallinari. Augmenting physical models with deep networks for complex dynamics forecasting. Journal of Statistical Mechanics: Theory and Experiment, 2021(12):124012, 2021. doi: 10.1088/1742-5468/ac3ae5. URL https://doi.org/10.1088/1742-5468/ ac3ae5.

Bin Yu. Rates of convergence for empirical processes of stationary mixing sequences. The Annals ofProbability, 22(1):94–116, 1994. ISSN 00911798, 2168894X. URL http://www.jstor. org/stable/2244496.

Han Zhang, Xi Gao, Jacob Unterman, and Tom Arodz. Approximation capabilities of neural ODEs and invertible residual networks, 13–18 Jul 2020. URL https://proceedings. mlr.press/v119/zhang20h.html.

Xun Zheng, Bryon Aragam, Pradeep K Ravikumar, and Eric Xing. Dags with no tears: Continuous optimization for structure learning, 2018. URL https://proceedings.neurips.cc/paper\_files/paper/2018/file/ e347c51419ffb23ca3fd5050202f9c3d-Paper.pdf.

Hui Zou. The adaptive lasso and its oracle properties. Journal of the American Statistical Association, 101(476):1418–1429, 2006. doi: 10.1198/016214506000000735. URL https: //doi.org/10.1198/016214506000000735.

Cos¸kun C¸ etin, Jose Roberto Castilho Piqueira, Burhaneddin <sup>˙</sup>Izgi, Ayse Peker-Dobie, Semra Ahmetolan, and Murat Ozkaya. Deterministic, stochastic, and mean-field pde models in neuroscience.<sup>¨</sup> Frontiers in Computational Neuroscience, Volume 20 - 2026, 2026. ISSN 1662-5188. doi: 10.3389/fncom.2026.1762692. URL https://www.frontiersin.org/journals/ computational-neuroscience/articles/10.3389/fncom.2026.1762692.

## A PROOFS

We dedicate this section to proving the different lemmas, propositions and theorems from the main content of the paper as well as some auxiliary lemmas and corollaries. We first establish that the model is well posed: Proposition 1 gives a unique, non-exploding strong solution. We then treat identifiability at the population level, first for the exact-risk formulation (Theorem 1) and then for the penalized objective (Theorem 2). The remaining three subsections provide the probabilistic machinery to go from the population result to the finite-sample one: uniform moment bounds, ergodicity of the true process, and exponential mixing of the sampled lagged state. These feed the finite-sample recovery guarantee (Theorem 4). We provide a graph illustrating the relationships between the assumptions, lemmas, propositions, and theorems in Figure 1.

## A.1 NOTATION

We denote the true data-generating objects using standard symbols: G for the drift, $G _ { j }$ for its coordinates, $f _ { j }$ for the link functions, and M for the binary mask. Candidates within the model class are indicated by a tilde $( \mathrm { e . g . , } \tilde { G } )$ , whereas empirical minimizers are denoted by a hat $( \mathrm { e } . \mathrm { g } . , \hat { G } )$ . Matrix indices follow the mask variables, where $\mathsf { \bar { M } } _ { i j } ^ { a } = 1$ indicates that the driver coordinate i at lag $\tau _ { a }$ influences the target coordinate $j .$ . The lag index $a \in \{ 0 , \ldots , q \}$ specifies the lag block, with $a = 0$ representing the instantaneous interactions. We let $m _ { j } ^ { a }$ denote the $j \mathrm { - t h }$ column of $M ^ { a }$ , and define $S _ { j } = \{ ( i , a ) : M _ { i j } ^ { a } = 1 \}$ as the active set of drivers for coordinate $j .$ The delay sequence is strictly ordered as $0 < \tau _ { 1 } < \cdots < \tau _ { q }$ , with the maximum delay defined as $\tau _ { \operatorname* { m a x } } = \tau _ { q } .$

Because the drift evaluates the trajectory at multiple past times, the process $X$ is non-Markovian, though the process of its recent histories (segments) is Markovian. We therefore operate on the segment space of continuous functions from $[ - \tau _ { \operatorname* { m a x } } , 0 ]$ to $\mathbb { R } ^ { D }$

$$
\begin{array} { r } { \mathcal { C } = C \big ( [ - \tau _ { \operatorname* { m a x } } , 0 ] ; \mathbb { R } ^ { D } \big ) \ , \qquad X _ { t } ^ { \mathrm { s e g } } ( s ) = X _ { t + s } , \quad s \in [ - \tau _ { \operatorname* { m a x } } , 0 ] , } \end{array}
$$

equipped with the supremum norm $\| \cdot \| _ { c }$ . Generic segments in this space are denoted by Greek letters such as $\varphi , \psi$ , and $\eta ,$ with $X _ { 0 } ^ { \mathrm { s e g } }$ serving as the initial condition. The drift relies strictly on $q + 1$ discrete time slices of a segment; thus, we define the lagged state as $Z _ { t } = ( X _ { t } , X _ { t - \tau _ { 1 } } , \dot { \textrm { . . . } } , X _ { t - \tau _ { q } } )$ Observations of this state are taken on a temporal grid $t _ { 0 } < \cdots < t _ { n } = T$ , where we write $Z _ { k } = Z _ { t _ { k } }$ . The effective horizon over which the drift is evaluated is $T _ { \circ } = T - \tau _ { \mathrm { m a x } }$ . Reference balls centered at the constant c (introduced in our regularity and dissipativity assumptions) are denoted by $B _ { R } \subset { \mathcal { C } }$ in the segment space and $K _ { \rho } \subset \mathbb { R } ^ { D ( q + 1 ) }$ in the lagged state space, and $c ^ { ( q ) } \in \mathbb { R } ^ { D ( q + 1 ) }$ denotes the concatenation of $q + 1$ copies of c. When $q = 0$ (no delays) we use the conventions $\tau _ { \operatorname* { m a x } } = 0$ and $\mathcal { C } = \mathbb { R } ^ { D }$ , so that the segment process is X itself.

We use subscripts on the probability operators to distinguish the initial state of the process. The probability measure and expectation for a solution initialized at a fixed segment $X _ { 0 } ^ { \mathrm { { \bar { s e g } } } } = \varphi$ are written as $\mathbb { P } _ { \varphi }$ and $\mathbb { E } _ { \varphi }$ , with the corresponding transition kernel $P _ { t } ( \varphi , \cdot )$ and semigroup $( P _ { t } ) _ { t \geq 0 }$ . This specific notation is utilized for moment and coupling estimates that require uniform bounds over $\varphi \in$ $B _ { R }$ . Conversely, when the process is initialized from equilibrium $( X _ { 0 } ^ { \mathrm { { s e g } } } \sim \pi ^ { \mathrm { { s e g } } }$ , independent of the driving noise $W )$ , we may use $\mathbb { P } _ { \pi }$ and $\mathbb { E } _ { \pi }$ . Here, $\pi ^ { \mathrm { s e g } }$ represents the invariant measure of the system, which forms the probabilistic foundation for our finite-sample analysis. Undecorated expectations E and probabilities $\mathbb { P }$ denote an arbitrary initial distribution consistent with the underlying datagenerating process.

We compare candidate drifts in the $L ^ { 2 }$ space associated with the joint occupation measure $\bar { \mu } _ { T } ^ { q }$ , which represents the time-averaged law of $Z _ { t }$ over the interval $[ \tau _ { \operatorname* { m a x } } , \bar { T } ]$ . Identifiability is first established as an identity in $L ^ { 2 } ( \bar { \mu } _ { T } ^ { q } )$ ) before being extended to all of $\mathbb { P } ^ { D ( q + 1 ) }$ . Under stationary conditions, $\bar { \mu } _ { T } ^ { q }$ coincides with the stationary law $\mu _ { \infty } ^ { q }$ of $Z _ { t }$ , rendering our bounding constants independent of the horizon $T .$

For a given candidate drift $\tilde { G }$ and mask $\tilde { M }$ , the population risk is defined as $\mathcal { R } _ { T } ( { \widetilde { G } } ) = T _ { \circ } \Vert { \widetilde { G } } -$ $G \| _ { L ^ { 2 } ( \bar { \mu } _ { T } ^ { q } ) } ^ { 2 } .$ . Its empirical counterpart is the Euler criterion $\textstyle { \widehat { \mathcal { R } } } _ { n }$ , leading to the penalized objective $J _ { \lambda } ( \widetilde { G } , \bar { \tilde { M } } ) = \mathcal { R } _ { T } ( \widetilde { G } ) + \lambda \sum _ { a } \Vert \tilde { M } ^ { a } \Vert _ { 0 }$ . Because $\textstyle { \widehat { \mathcal { R } } } _ { n }$ contains a quadratic-variation term identical across all candidates, we evaluate only the excess empirical risk ${ \widehat { \mathcal { R } } } _ { n } ( { \widetilde { G } } ) - { \widehat { \mathcal { R } } } _ { n } ( G )$ against the population risk $\mathcal { R } _ { T }$ . Although $\mathcal { R } _ { T }$ involves the unknown $G ,$ it is determined by observable quantities up to a candidate-independent constant: for every $\widetilde { G }$ with finite risk, the contrast $\begin{array} { r } { \mathcal { C } _ { T } ( { \widetilde { G } } ) : = \mathbb { E } \big [ \int _ { \tau _ { \mathrm { m a x } } } ^ { T } \| { \widetilde { G } } ( Z _ { t } ) \| ^ { 2 } d t - 2 \int _ { \tau _ { \mathrm { m a x } } } ^ { T } \langle { \widetilde { G } } ( Z _ { t } ) , d X _ { t } \rangle \big ] } \end{array}$ satisfies $\mathcal { C } _ { T } ( \widetilde { G } ) - \mathcal { C } _ { T } ( G ) = \mathcal { R } _ { T } ( \widetilde { G } )$ , as follows by substituting the SDDE and using that the square-integrable stochastic integral has zero mean.

Finally, successful recovery is governed by three primary structural quantities. The first is the signal strength $s _ { i j } ^ { a } = \Vert G _ { j } - \Pi _ { - ( i , a ) } ^ { \cdot } \bar { G } _ { j } \Vert _ { L ^ { 2 } ( \bar { \mu } _ { T } ^ { q } ) } \mathrm { ~ , ~ }$ , where $\Pi _ { - ( i , a ) }$ denotes the orthogonal projection onto the subspace of functions that do not depend on (i.e. do not use) coordinate i in lag block a. The remaining two are $s _ { \mathrm { m i n } }$ , the minimum signal strength across all true edges, and $d _ { \operatorname* { m a x } } .$ , the maximal in-degree of the system. These combine to define the margin $\gamma ( \lambda ) = \mathrm { { \bar { m i n } } } \{ \lambda , T _ { \circ } s _ { \mathrm { m i n } } ^ { 2 } - \lambda d _ { \mathrm { m a x } } \}$ which acts as the minimum objective gap incurred by any incorrectly identified mask.

## A.2 NON-EXPLOSION AND EXISTENCE OF A STRONG SOLUTION

Since the process is not Markov, we cannot apply the classical existence theorems for SDEs directly. Instead, we exploit the fact that the delays are bounded away from zero and prove existence of a unique solution by induction over intervals of length $\tau _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { a } \tau _ { a }$ (the so-called method of steps, see $\mathrm { e . g . }$ . Hale & Verduyn Lunel, 1993). On each such interval the delayed arguments are already known from the previous interval, so that the SDDE reduces to a standard (non-delayed) Ito equationˆ with a random, time-dependent drift: we solve that by truncation on a closed ball and rule out explosion with a Khasminskii-type Lyapunov test. The induction then extends the solution to all of $[ 0 , \infty )$ . The proof of the following statement only uses the initial-condition clause of Assumption 1 (independence of $X _ { 0 } ^ { \mathrm { s e g } }$ and $W$ , and its finite second moment) together with Assumption 2.

Proposition 1 (Non-explosion & Strong solution). Under Assumptions 1 and 2, the SDDE defined in Equation 1 has a unique strong solutionfor every initial condition with $\mathbb { E } \| X _ { 0 } ^ { \mathrm { s e g } } \| _ { c } ^ { 2 } < \infty$ without explosion i.e. it does not diverge to infinity in a finite amount of time.

Proof. Step 1: Initialisation on $[ - \tau _ { \mathrm { m a x } } , \tau _ { \mathrm { m i n } } ]$ The initial segment $X _ { 0 } ^ { \mathrm { s e g } }$ provides the solution on $[ - \tau _ { \operatorname* { m a x } } , 0 ]$ . For $s \in \ [ 0 , \tau _ { \mathrm { m i n } } ]$ and every lag $a \ge 1$ , the delayed argument satisfies $s - \tau _ { a } \leq$ $\tau _ { \mathrm { m i n } } - \tau _ { a } \leq 0 , \mathrm { s o } X _ { s - \tau _ { a } }$ is read from the initial segment. Define

$$
Y _ { s } = ( X _ { s - \tau _ { 1 } } , \dots , X _ { s - \tau _ { q } } ) , \quad s \in [ 0 , \tau _ { \operatorname* { m i n } } ]
$$

This is a continuous, $\mathcal { F } _ { 0 }$ -measurable path with compact range $K _ { Y } \subset \mathbb { R } ^ { D q }$ (as a continuous function on the compact $[ 0 , \tau _ { \mathrm { m i n } } ]$ composed with the continuous initial segment). On this segment, the SDDE reduces to a standard (non-delayed) Ito equation:ˆ

$$
d X _ { s } = b ( s , X _ { s } ) d s + \Sigma d W _ { s } , \quad X _ { 0 } = X _ { 0 } ^ { \mathrm { s e g } } ( 0 )
$$

with time-dependent drift $b ( s , x ) : = G ( x , Y _ { s } )$

(a) Existence and uniqueness via truncated approximation. For $n \in \mathbb { N } .$ , consider the metric projection onto the closed ball of radius n centered at $c , \pi _ { n } : \mathbb { R } ^ { D }  \bar { B } _ { n } ( c )$ which is 1-Lipschitz by convexity. Define the truncated drift:

$$
b _ { n } ( s , x ) : = b ( s , \pi _ { n } ( x ) ) = G ( \pi _ { n } ( x ) , Y _ { s } )
$$

Since G is locally Lipschitz by Assumption 2 and both $\pi _ { n } , Y _ { s }$ remain in a compact set $( \bar { B } _ { n } ( c ) \times K _ { Y } )$ the truncated drift $b _ { n }$ is globally Lipschitz in x uniformly in $s \in [ 0 , \tau _ { \operatorname* { m i n } } ]$ with Lipschitz constant $L _ { n }$ (random but $\mathcal { F } _ { 0 } .$ -measurable).

Since $L _ { n }$ and Y are $\mathcal { F } _ { \mathrm { 0 } ^ { - } } \mathrm { m e a s u r a b l e }$ while the Brownian motion $W$ is independent of $\mathcal { F } _ { 0 }$ (Assumption 1), we may apply the Picard-Lindelof theorem for SDEs with globally Lipschitz coefficients¨ (Mao, 2007, Ch. 2) conditionally on $\mathcal { F } _ { 0 }$ , i.e. for each fixed realisation of the initial segment (alternatively, one may localise on the events $\{ L _ { n } \leq m \} \in \mathcal { F } _ { 0 } , m \in \mathbb { N } )$ . Hence the truncated equation

$$
d X _ { s } ^ { ( n ) } = b _ { n } ( s , X _ { s } ^ { ( n ) } ) d s + \Sigma d W _ { s } , \quad X _ { 0 } ^ { ( n ) } = X _ { 0 } ^ { \mathrm { s e g } } ( 0 )
$$

admits a unique non-exploding strong solution on $[ 0 , \tau _ { \mathrm { m i n } } ] .$

Define the exit time from $\bar { B } _ { n } ( c ) , \zeta _ { n } : = \operatorname* { i n f } \{ s \geq 0 , \| X _ { s } ^ { ( n ) } - c \| \geq n \}$ . For any $n \leq m$ , the pathwise uniqueness on $[ 0 , \zeta _ { n } ]$ gives $X _ { s } ^ { ( n ) } = X _ { s } ^ { ( m ) }$ for all $s \leq \zeta _ { n }$ (since $b _ { n } = b _ { m } = b$ on the ball $\bar { B } _ { n } ( c ) )$ .

![](images/dd78ac1a67c5b2941f0e32c6d60fcd7840f495a0e557d527a057f0aa48e953cc.jpg)  
Figure 1: Dependency map of the theoretical results. Yellow boxes are assumptions, blue boxes are lemmas and propositions, and green boxes are theorems. Arrows denote when an assumption or a result is used in another proof. A use already implied by a chain of arrows is not drawn, unless it plays a separate role (labelled arrows). The proofs are grouped vertically into the three groups of proofs in Section A, contributing the finite-sample guarantee at the bottom.

Thus, $\zeta _ { n }$ is non-decreasing in $n ,$ and given the limit $\zeta = \operatorname* { l i m } _ { n \to \infty } \zeta _ { n }$ , define the process $X _ { s } : = X _ { s } ^ { ( n ) }$ for $s < \zeta _ { n }$ . This defines a unique strong solution on $[ 0 , \zeta )$

(b) Non-explosion via Khasminskii’s test (Khasminskii, 2012; see also Mao, 2007). Set $\Phi ( x ) : = { \bf \Phi }$ $1 + \| x - c \| ^ { 2 }$ . We work conditionally on ${ \mathcal { F } } _ { 0 } ,$ under which Y and the constants below are fixed while W remains a Brownian motion. By Dynkin’s formula applied to $\Phi ( X _ { s \wedge \zeta _ { n } } )$

$$
\mathbb { E } [ \Phi ( X _ { s \wedge \zeta _ { n } } ) \mid { \mathcal { F } } _ { 0 } ] = \Phi ( X _ { 0 } ) + \mathbb { E } { \Big [ } { \int _ { 0 } ^ { s \wedge \zeta _ { n } } } { \mathcal { L } } _ { u } \Phi ( X _ { u } ) d u { \Big | } { \mathcal { F } } _ { 0 } { \Big ] }
$$

where $\mathcal { L } _ { u } \Phi ( \boldsymbol { x } ) : = 2 \langle \boldsymbol { x } - \boldsymbol { c } , \boldsymbol { b } ( u , \boldsymbol { x } ) \rangle + \mathrm { t r } ( \Sigma \Sigma ^ { \top } )$ is the (time-dependent) generator of the equation with frozen delayed arguments.

Let $\begin{array} { r } { \rho _ { Y } ^ { 2 } : = \operatorname* { s u p } _ { s \in [ 0 , \tau _ { \operatorname* { m i n } } ] } \sum _ { a = 1 } ^ { q } \| X _ { s - \tau _ { a } } - c \| ^ { 2 } < \infty } \end{array}$ , which is $\mathcal { F } _ { 0 }$ -measurable since $K _ { Y }$ is compact, so that $\lVert ( x _ { 0 } , Y _ { s } ) - c ^ { ( q ) } \rVert ^ { 2 } \leq \lVert x _ { 0 } - c \rVert ^ { 2 } + \rho _ { Y } ^ { 2 }$ . By Assumption $2 ( \mathrm { i i } )$ , whenever $\| ( x _ { 0 } , Y _ { s } ) - c ^ { ( q ) } \| \ge R$ we have $2 \langle x _ { 0 } - c , G ( x _ { 0 } , Y _ { s } ) \rangle \leq 2 \beta ( 1 \dot { + } \| x _ { 0 } - c \| ^ { 2 } + \rho _ { Y } ^ { 2 } ) \leq 2 \beta ( 1 + \rho _ { Y } ^ { 2 } ) \Phi ( x _ { 0 } )$ . Otherwise $\left\| x _ { 0 } - c \right\| < R .$ For $\left\| x _ { 0 } - c \right\| \leq R ,$ , set

$$
M _ { R } : = \operatorname* { s u p } \{ \| G ( x _ { 0 } , y ) \| : \| x _ { 0 } - c \| \leq R , y \in K _ { Y } \} < \infty \quad \mathrm { a . s . } ,
$$

finite by continuity of G on the compact $\bar { B } _ { R } ( c ) \times K _ { Y }$ , so that $2 \langle x _ { 0 } - c , G ( x _ { 0 } , Y _ { s } ) \rangle \leq 2 R M _ { R } \leq$ $2 C _ { R } \Phi ( x _ { 0 } )$ with ${ \dot { C } } _ { R } : = R M _ { R } .$ , using $\Phi \geq 1$ . Hence, for all $\boldsymbol { x } \in \mathbb { R } ^ { D }$ and $s \in [ 0 , \tau _ { \operatorname* { m i n } } ] .$

$$
\mathcal { L } _ { s } \Phi ( x ) = 2 \langle x - c , G ( x , Y _ { s } ) \rangle + \operatorname { t r } ( \Sigma \Sigma ^ { \top } ) \leq C \Phi ( x ) ,
$$

where $C : = 2 \operatorname* { m a x } \big ( \beta ( 1 + \rho _ { Y } ^ { 2 } ) , C _ { R } \big ) + \mathrm { t r } ( \Sigma \Sigma ^ { \top } )$ is $\mathcal { F } _ { 0 }$ -measurable and a.s. finite.

As $\begin{array} { r } { \int _ { 0 } ^ { s \wedge \zeta _ { n } } \mathcal { L } _ { u } \Phi ( X _ { u } ) d u \leq C \int _ { 0 } ^ { s } \Phi ( X _ { u \wedge \zeta _ { n } } ) \cdot } \end{array}$ du, Gronwall’s inequality gives¨

$$
\mathbb { E } [ \Phi ( X _ { s \wedge \zeta _ { n } } ) \mid \mathcal { F } _ { 0 } ] \le \Phi ( X _ { 0 } ) e ^ { C s } , \quad s \in [ 0 , \tau _ { \operatorname* { m i n } } ] .
$$

Since $\Phi ( X _ { \zeta _ { n } } ) \geq 1 + n ^ { 2 }$ on $\left\{ \zeta _ { n } \leq \tau _ { \operatorname* { m i n } } \right\}$ , we have that

$$
( 1 + n ^ { 2 } ) \mathbb { P } ( \zeta _ { n } \le \tau _ { \operatorname* { m i n } } \mid \mathcal { F } _ { 0 } ) \le \mathbb { E } [ \Phi ( X _ { \zeta _ { n } } ) \mathbf { 1 } _ { \zeta _ { n } \le \tau _ { \operatorname* { m i n } } } \mid \mathcal { F } _ { 0 } ] \le \Phi ( X _ { 0 } ) e ^ { C \tau _ { \operatorname* { m i n } } } .
$$

Since $\{ \zeta \leq \tau _ { \operatorname* { m i n } } \} \subseteq \{ \zeta _ { n } \leq \tau _ { \operatorname* { m i n } } \}$ , letting $n \to \infty$ yields $\mathbb { P } ( \zeta \le \tau _ { \mathrm { m i n } } \mid \mathcal { F } _ { 0 } ) = 0 \mathrm { a . s }$ ., and integrating these conditional probabilities (which are bounded by one) gives $\mathbb { P } ( \zeta \leq \tau _ { \operatorname* { m i n } } ) = 0$ , i.e. the solution does not explode on $[ 0 , \tau _ { \mathrm { m i n } } ]$ . No integrability of $\Phi ( { \bf \bar { \cal X } } _ { 0 } ) e ^ { { \cal \tilde { C } } \tau _ { \mathrm { m i n } } }$ is needed.

Step 2: Induction on $[ k \tau _ { \mathrm { m i n } } , ( k + 1 ) \tau _ { \mathrm { m i n } } ] .$ The delayed arguments now satisfy $s - \tau _ { a } \leq k \tau _ { \operatorname* { m i n } } ,$ hence lie on the already-constructed path; parts (a)–(b) apply verbatim with $\mathcal { F } _ { 0 }$ replaced by $\mathcal { F } _ { k \tau _ { \mathrm { m i n } } } .$ Induction over k yields a unique strong solution on $[ 0 , \infty )$ with $\begin{array} { r } { \operatorname* { s u p } _ { s \leq T } \lVert X _ { s } \rVert < \infty \ \mathrm { a . s } } \end{array}$ . for every finite T.

Note: When $q = 0$ there are no delayed arguments: Step 1 applies directly on any finite interval [0, T] (with $Y$ void and $\tau _ { \mathrm { m i n } }$ replaced by T), and no induction is needed. □

## A.3 POPULATION IDENTIFIABILITY

We prove population identifiability in two stages. First, we show that the drift and the mask are pinned down on the support of the joint occupation measure: matching the risk forces $\widetilde G = G$ there by continuity, and faithfulness upgrades this to equality of the masks. Second, Lemma 1 shows that, under non-degenerate noise, the occupation measure has full support, so “on the support” is in fact “everywhere” and the faithfulness hypothesis reduces to plain faithfulness.

The proof follows the structure of the assumptions: a zero risk forces ${ \widetilde { G } } = G$ on the support of $\bar { \mu } _ { T } ^ { q } = \mathbb { R } ^ { D ( q + 1 ) }$ which fills the whole domain from the non-degenerate noise (Lemma 1), faithfulness prevents the learned mask from dropping a true edge, and sparsity prevents it from adding a spurious one. We compare these assumptions to those of Bellot et al. (2022) in Section B.

Theorem 1 (Identifiability). Under Assumptions $I { - } 5 , G$ and M are identifiable:

$$
\widetilde { \cal G } = { \cal G } o n \mathrm { s u p p } ( \bar { \mu } _ { T } ^ { q } ) , \qquad \tilde { \cal M } = { \cal M } .\tag{19}
$$

Moreover, under the non-degenerate noise ofAssumption 1 the occupation measure hasfull support, $\mathrm { s u p p } ( \bar { \mu } _ { T } ^ { q } ) = \mathbb { R } ^ { D ( q + 1 ) }$ by Lemma 1 (see below), so the drift equality extends to all $o f \mathbb { R } ^ { \breve { D } ( q + 1 ) }$

Proof. Step 1: Drift identifiability on $\operatorname { s u p p } ( \bar { \mu } _ { T } ^ { q } )$ The true pair $( G , M )$ is feasible by Assumption 3(i) and has risk zero. Nonnegativity and optimality (Assumption 3(ii)) therefore give $\mathcal { R } _ { T } ( \widetilde { G } ) = 0$ , hence by Equation $\sf t _ { \tilde { G } } = G _ { \mu _ { T } ^ { q } - \mathrm { a . e . } ; \ G }$ is continuous by Assumption 2 and $\widetilde { G }$ by Assumption 3(i), and the complement of a $\bar { \mu } _ { T } ^ { q }$ -null set meets every nonempty relatively open subset of $\operatorname { s u p p } ( { \bar { \mu } } _ { T } ^ { q } )$ , so equality on this dense subset of the (closed) support upgrades to equality on all of $\operatorname { s u p p } ( \bar { \mu } _ { T } ^ { q } )$

Step 2: Mask identifiability on the support. If $M _ { i j } ^ { a } = 1$ but ${ \tilde { M } } _ { i j } ^ { a } = 0 -$ that is, if a true edge were dropped — then $\widetilde { G } _ { j }$ is constant in coordinate i of block a on all of $\mathbb { R } ^ { D ( q + 1 ) }$ (its mask removes that input), so $G _ { j } = \widetilde { G } _ { j }$ on $\operatorname { s u p p } ( \bar { \mu } _ { T } ^ { q } )$ takes equal values at the axis-aligned witnessing pair $z , z ^ { \prime } \in$ $\operatorname { s u p p } ( \bar { \mu } _ { T } ^ { q } )$ of Assumption 4, which is a contradiction. Hence $\tilde { M } _ { i j } ^ { a } \geq M _ { i j } ^ { a }$ entry-wise for all $^ { a , }$ i.e. $\begin{array} { r } { \sum _ { a } \lVert \tilde { \boldsymbol { M } } ^ { a } \rVert _ { 0 } \geq \sum _ { a } \lVert \boldsymbol { M } ^ { a } \rVert _ { 0 } } \end{array}$ with equality iff $\tilde { M } ^ { a } = M ^ { a }$ for all a; Assumption 5 forces equality.

It remains to remove the “on the support” qualifier. The next lemma shows that non-degenerate noise makes the joint law of the lagged state charge every open set, so its support is the whole space and the identity is global.

Lemma 1 (Full support of the joint occupation measure). Under Assumptions 1 and $2 , f o r$ every $t >$ $\tau _ { \mathrm { m a x } }$ the law of $\left( X _ { t } , X _ { t - \tau _ { 1 } } , \ldots , X _ { t - \tau _ { q } } \right)$ on $\mathbb { R } ^ { D ( q + 1 ) }$ has full support; consequently $\mathrm { s u p p } ( \bar { \mu } _ { T } ^ { q } ) =$ $\mathbb { R } ^ { D ( q + 1 ) }$ for every $T > \tau _ { \mathrm { m a x } } .$

Proof. We first prove the statement for a deterministic initial history and then integrate.

Step 1: Deterministic initial segment and pilot path. Fix a deterministic initial segment $\varphi \in { \mathcal { C } } .$ , a time $t > \tau _ { \operatorname* { m a x } } ,$ a target $\boldsymbol { z } = ( z _ { 0 } , z _ { 1 } , \ldots , z _ { q } ) \in \mathbb { R } ^ { D ( q + 1 ) }$ and $\varepsilon > 0$ , and write $\mathbb { P } _ { \varphi }$ for the law of the solution started from $X _ { 0 } ^ { \mathrm { s e g } } = { \varphi }$ . We must show:

$$
\begin{array} { r } { \mathbb { P } _ { \varphi } \left( \| X _ { t } - z _ { 0 } \| < \varepsilon , \| X _ { t - \tau _ { 1 } } - z _ { 1 } \| < \varepsilon , \dots , \| X _ { t - \tau _ { q } } - z _ { q } \| < \varepsilon \right) > 0 . } \end{array}
$$

With the convention $\tau _ { 0 } = 0$ , the times $t - \tau _ { a }$ for $a = 0 , 1 , \ldots , q$ are distinct and strictly positive, since $t > \tau _ { \mathrm { m a x } } = \tau _ { q }$ . Choose a path $h : [ - \tau _ { \operatorname* { m a x } } , t ] \to \mathbb { R } ^ { D }$ which is continuously differentiable on $[ 0 , t ]$ and satisfies

$$
h | _ { [ - \tau _ { \operatorname* { m a x } } , 0 ] } = \varphi , \qquad h ( t - \tau _ { a } ) = z _ { a } \quad ( 0 \leq a \leq q ) .
$$

Such a path exists because we are prescribing finitely many values at distinct times of $( 0 , t ]$ , together with the value $h ( 0 ) = \varphi ( 0 )$ at the left endpoint (e.g. by spline interpolation). Note that h need not be differentiable at 0 and no regularity of $\varphi$ beyond continuity is required.

Step 2: The driving control. Since Σ is invertible by Assumption 1, we may define

$$
w _ { h } ( s ) \ : = \ \Sigma ^ { - 1 } \left[ h ( s ) - \varphi ( 0 ) - \int _ { 0 } ^ { s } G \big ( h ( u ) , h ( u - \tau _ { 1 } ) , \dots , h ( u - \tau _ { q } ) \big ) d u \right] , \qquad s \in [ 0 , t ] .
$$

The integrand is continuous on $[ 0 , t ] ,$ so $w _ { h }$ is continuously differentiable with $w _ { h } ( 0 ) = 0$ . In particular, $w _ { h }$ has square-integrable derivative and belongs to the Cameron–Martin space of the Wiener measure on $\dot { C } ( [ 0 , t ] ; \mathbb { R } ^ { \breve { D } } )$ (see Nualart, 2006, Ch. 1). By construction h solves the integral equation associated with Equation 1 on [0, t] driven by $w _ { h }$ in place of $W :$

$$
h ( \boldsymbol { s } ) = \varphi ( 0 ) + \int _ { 0 } ^ { s } G \big ( h ( \boldsymbol { u } ) , h ( \boldsymbol { u } - \tau _ { 1 } ) , \ldots , h ( \boldsymbol { u } - \tau _ { q } ) \big ) d \boldsymbol { u } + \Sigma w _ { h } ( \boldsymbol { s } ) .\tag{20}
$$

Step 3: Localized Gronwall estimate.¨ Let

$$
K : = { \Big \{ } ( y _ { 0 } , \dotsc , y _ { q } ) \in \mathbb { R } ^ { D ( q + 1 ) } ~ : ~ \exists u \in [ 0 , t ] { \mathrm { ~ w i t h ~ } } \operatorname* { m a x } _ { 0 \leq a \leq q } \| y _ { a } - h ( u - \tau _ { a } ) \| \leq 1 { \Big \} } ,
$$

a compact subset of $\mathbb { R } ^ { D ( q + 1 ) }$ , and let $L < \infty$ be a Lipschitz constant for $G$ on $K .$ , which exists by Assumption 2 (i). Both K and L are deterministic as functions of h alone. Choose $\eta > 0$ such that

$$
\| \Sigma \| _ { \mathrm { o p } } \eta e ^ { L \sqrt { q + 1 } t } < \mathrm { m i n } ( 1 , \varepsilon ) ,\tag{21}
$$

and define the Brownian tube event and the exit time

$$
A _ { \eta } : = \Big \{ \operatorname* { s u p } _ { s \leq t } \| W _ { s } - w _ { h } ( s ) \| < \eta \Big \} , \qquad \sigma : = t \wedge \operatorname* { i n f } \{ s \geq 0 : \| X _ { s } - h ( s ) \| \geq 1 \} .
$$

Since X and h agree with $\varphi \mathrm { ~ o n ~ } [ - \tau _ { \mathrm { m a x } } , 0 ]$ , we have $X _ { u } - h ( u ) = 0$ for every $u \in [ - \tau _ { \operatorname* { m a x } } , 0 ]$ , so that for $s \in [ 0 , \sigma ]$

$$
\| X _ { s - \tau _ { a } } - h ( s - \tau _ { a } ) \| ~ \le ~ D ( s ) : = \operatorname* { s u p } _ { 0 \le u \le s } \| X _ { u } - h ( u ) \| , \qquad 0 \le a \le q .
$$

Subtracting Equation 20 from the integral form of Equation 1 and using that both argument vectors lie in K before time σ, we obtain on $A _ { \eta } .$ , for every $s \leq \sigma$

$$
D ( s ) \ \leq \ \| \Sigma \| _ { \infty } \operatorname* { s u p } _ { u \leq t } \| W _ { u } - w _ { h } ( u ) \| + L \sqrt { q + 1 } \int _ { 0 } ^ { s } D ( u ) d u \ \leq \ \| \Sigma \| _ { \infty } \eta + L \sqrt { q + 1 } \int _ { 0 } ^ { s } D ( u ) d u ,
$$

the factor $\sqrt { q + 1 }$ coming from bounding the Euclidean norm on $\mathbb { R } ^ { D ( q + 1 ) }$ by $\sqrt { q + 1 }$ times the maximum of the blockwise norms. Gronwall’s inequality and Equation 21 then give ¨

$$
D ( \sigma ) \leq \| \Sigma \| _ { \mathrm { o p } } \eta e ^ { L \sqrt { q + 1 } t } < \operatorname* { m i n } ( 1 , \varepsilon ) .
$$

In particular $\| X _ { \sigma } - h ( \sigma ) \| < 1$ strictly, so by continuity of $s \mapsto X _ { s } - h ( s )$ the exit time cannot occur before t, i.e. $\sigma = t$ on $A _ { \eta } ,$ and the estimate $\mathrm { s u p } _ { s \leq t } \bar { \| } X _ { s } - h ( s ) \| < \varepsilon$ holds on all of [0, t].

Step 4: Positivity of the Brownian tube probability. The path $w _ { h }$ is continuously differentiable with $w _ { h } ( 0 ) = 0$ , hence lies in the Cameron–Martin space, so the law of $W - w _ { h }$ on $C ( [ 0 , t ] ; \mathbb { R } ^ { D } )$ is equivalent to the Wiener measure. Since the Brownian small-ball probability $\mathbb { P } ( \operatorname* { s u p } _ { s < t } \| W _ { s } \| < \eta )$ is strictly positive, we conclude that $\mathbb { P } ( A _ { \eta } ) > 0$ (Stroock & Varadhan, 1972, Sec. 3). Combining with Step 3 and $h ( t - \tau _ { a } ) = z _ { a }$

$$
\mathbb { P } _ { \varphi } \Big ( \operatorname* { m a x } _ { 0 \le a \le q } \| X _ { t - \tau _ { a } } - z _ { a } \| < \varepsilon \Big ) \ \ge \ \mathbb { P } ( A _ { \eta } ) \ > \ 0 .
$$

Step 5: Random initial history and conclusion. By Proposition 1, the solution is a measurable functional of $( \varphi , W ) , \operatorname { s o } \varphi \mapsto \mathbb { P } _ { \varphi } ( \cdot )$ is a probability kernel; using Assumption 1, the initial segment $X _ { 0 } ^ { \mathrm { s e g } }$ is independent of $W$ . Conditioning on $X _ { 0 } ^ { \mathrm { s e g } } = \varphi$ and integrating the strictly positive conditional probabilities of Step 4 against Law(X<sup>seg</sup><sub>0</sub> ) yields

$$
\mathbb { P } \Big ( \operatorname* { m a x } _ { 0 \leq a \leq q } \| X _ { t - \tau _ { a } } - z _ { a } \| < \varepsilon \Big ) > 0 .
$$

Since $z \in \mathbb { R } ^ { D ( q + 1 ) }$ and $\varepsilon > 0$ were arbitrary, the law of $( X _ { t } , X _ { t - \tau _ { 1 } } , \ldots , X _ { t - \tau _ { q } } )$ charges every nonempty open subset of $\mathbb { R } ^ { D ( q + 1 ) }$ , i.e. it has full support. Finally, for every nonempty open $O \subset$ $\mathbb { R } ^ { D ( q + \mathbf { \hat { 1 } } ) }$ the map $t \ \mapsto \ \mathbb { P } \big ( ( X _ { t } , X _ { t - \tau _ { 1 } } , \ldots , X _ { t - \tau _ { a } } ) \ \in \ O \big )$ is strictly positive on $( \tau _ { \operatorname* { m a x } } , T ]$ , so its integral over that interval is strictly positive. The single endpoint $t = \tau _ { \operatorname* { m a x } } ,$ , at which the lagged coordinate $X _ { t - \tau _ { q } } = X _ { 0 }$ is still determined by the initial history, has zero Lebesgue measure and does not affect the integral. Hence $\operatorname { s u p p } ( \bar { \mu } _ { T } ^ { q } ) = \mathbb { R } ^ { D ( q + 1 ) }$ □

## A.4 IDENTIFIABILITY WITH THE PENALIZED OBJECTIVE

The penalized objective decouples across output coordinates, so it is enough to argue one coordinate $j$ at a time. For each $j ,$ , we compare the true active set $S _ { j }$ against a candidate ${ \tilde { S } } _ { j }$ in two exhaustive cases; the minimum-signal condition $s _ { \operatorname* { m i n } } > 0$ makes dropping a true input strictly costly, and this yields the sufficient λ-window Equation 22.

Theorem 2 (Identifiability with the penalized objective). Let Assumptions 1, 2 and 3(i) hold. Suppose $M \ne 0$ and $s _ { \operatorname* { m i n } } > 0$ i.e. the system has at least one true driver, and every true driver has a measurable impact on the dynamics. This replaces Assumption 4 (faithfulness). If

$$
0 < \lambda < \frac { ( T - \tau _ { \operatorname* { m a x } } ) s _ { \operatorname* { m i n } } ^ { 2 } } { d _ { \operatorname* { m a x } } } ,\tag{22}
$$

then every minimizer $o f J _ { \lambda }$ over $\mathcal { H } ^ { q }$ satisfies $\tilde { M } = M$ and ${ \widetilde { G } } = G o n \mathbb { R } ^ { D ( q + 1 ) }$ . Moreover, every feasible pair with $\tilde { M } \neq M$ satisfies

$$
J _ { \lambda } ( \tilde { G } , \tilde { M } ) - J _ { \lambda } ( G , M ) \ge \gamma ( \lambda ) : = \operatorname* { m i n } ( \lambda , ( T - \tau _ { \operatorname* { m a x } } ) s _ { \operatorname* { m i n } } ^ { 2 } - \lambda d _ { \operatorname* { m a x } } ) > 0
$$

Proof. The true pair (G, M) is feasible by Assumption 3(i) and satisfies $\begin{array} { r } { \mathcal { R } _ { T } ( G ) = ~ 0 } \end{array}$ , so $\begin{array} { r } { J _ { \lambda } ( G , M ) = \lambda \sum _ { j = 1 } ^ { D } | S _ { j } | } \end{array}$ , where $S _ { j } : = \{ ( i , a ) : M _ { i j } ^ { a } = 1 \}$ is the true active set of coordinate j and $\begin{array} { r } { | S _ { j } | ~ = ~ \sum _ { a = 0 } ^ { q } \| m _ { j } ^ { a } \| _ { 0 } ~ \le ~ d _ { \operatorname* { m a x } } } \end{array}$ . Let $( \tilde { G } , \tilde { M } ) \ \in \ \mathcal { H } ^ { q }$ be any feasible pair and write $\tilde { S } _ { j } : = \{ ( i , a ) : \tilde { M } _ { i j } ^ { a } = 1 \}$ . Since the squared $L _ { 2 } ( \bar { \mu } _ { T } ^ { q } )$ risk and the $\ell _ { 0 }$ penalty of Equation 9 are both sums over the output coordinates, the objective gap decomposes as

$$
J _ { \lambda } ( \widetilde { G } , \widetilde { M } ) - J _ { \lambda } ( G , M ) = \sum _ { j = 1 } ^ { D } \Delta _ { j } , \qquad \Delta _ { j } : = ( T - \tau _ { \operatorname * { m a x } } ) \| G _ { j } - \widetilde { G } _ { j } \| _ { L _ { 2 } ( \bar { \mu } _ { T } ^ { q } ) } ^ { 2 } + \lambda \big ( | \widetilde { S } _ { j } | - | S _ { j } | \big ) .\tag{23}
$$

This decomposes the value of the gap at a fixed feasible pair. It therefore suffices to bound $\Delta _ { j }$ from below in the two exhaustive cases and to sum.

Consider first a coordinate whose mask keeps every true input, $\tilde { S } _ { j } \supseteq S _ { j }$ . Then $| \tilde { S } _ { j } | \geq | S _ { j } |$ , and since the risk term is nonnegative we obtain $\Delta _ { j } \geq 0 ;$ if the containment is strict then $| \tilde { S } _ { j } | \geq | S _ { j } | + 1$ and

$$
\Delta _ { j } \ \geq \ \lambda > \ 0 .\tag{24}
$$

Consider next a coordinate whose mask omits some true input $( i , a ) \in S _ { j } \setminus \tilde { S } _ { j }$ , that is ${ \tilde { M } } _ { i j } ^ { a } = 0$ while $M _ { i j } ^ { a } = 1$ . By the masked parametrization of Assumption 3(i), the candidate $\widetilde { G } _ { j } = \widetilde { f } _ { j } ( \tilde { m } _ { j } ^ { 0 } \mathbb { \Lambda }$ $x _ { 0 } , \ldots , \tilde { \dot { m } } _ { j } ^ { q } \odot x _ { q } )$ does not depend on coordinate i of block a, so it belongs to the closed subspace onto which $\Pi _ { - ( i , a ) }$ projects (the $L _ { 2 } ( \bar { \mu } _ { T } ^ { q } )$ space of the σ-algebra generated by the remaining coordinates; note that $G _ { j } \in L _ { 2 } ( \bar { \mu } _ { T } ^ { q } )$ by the finite-energy clause of Assumption 1, hence $\widetilde { G } _ { j } \in L _ { 2 } ( \bar { \mu } _ { T } ^ { q } )$ by finite risk).

Since $\Pi _ { - ( i , a ) } G _ { j }$ is the nearest point of that subspace to $G _ { j }$ , the signal strength Equation 7 lowerbounds the distance to any of its elements, and in particular

$$
\| G _ { j } - \widetilde { G } _ { j } \| _ { L _ { 2 } ( \bar { \mu } _ { T } ^ { q } ) } \geq \big \| { G } _ { j } - \Pi _ { - ( i , a ) } G _ { j } \big \| _ { L _ { 2 } ( \bar { \mu } _ { T } ^ { q } ) } = s _ { i j } ^ { a } \geq s _ { \operatorname* { m i n } } .
$$

Bounding the penalty term by $| \tilde { S } _ { j } | \geq 0$ and $| S _ { j } | \le d _ { \operatorname* { m a x } }$ gives

$$
\Delta _ { j } \ \ge \ ( T - \tau _ { \operatorname * { m a x } } ) ( s _ { i j } ^ { a } ) ^ { 2 } - \lambda | S _ { j } | \ \ge \ ( T - \tau _ { \operatorname * { m a x } } ) s _ { \operatorname * { m i n } } ^ { 2 } - \lambda d _ { \operatorname * { m a x } } ,\tag{25}
$$

which is strictly positive under the window Equation 22.

Now let $( \widetilde G , \tilde { M } )$ be feasible with $\tilde { M } \ne M$ and let $j$ be a coordinate in which the two masks differ. That coordinate falls into exactly one of the two cases: either it omits a true input, and Equation 25 applies, or its mask strictly contains $S _ { j }$ , and Equation 24 applies. In both cases $\Delta _ { j } \geq \gamma \bar { ( \lambda ) }$ , while every remaining coordinate satisfies $\check { \Delta _ { j } } \geq 0$ under Equation 22. Summing in Equation 23 yields $J _ { \lambda } ( \widetilde { G } , \tilde { M } ) - J _ { \lambda } ( G , M ) \ge \gamma ( \lambda ) > 0 ,$ , so no pair with an incorrect mask can be a minimizer and every minimizer satisfies $\tilde { M } = M$ . Given $\tilde { M } = M$ the true pair is feasible with objective $\lambda \sum _ { j } | S _ { j } |$ so any minimizer has $\mathcal { R } _ { T } ( \widetilde { G } ) = 0$ , that is Ge = G µ¯<sup>q</sup> -almost everywhere. G is continuous by Assumption 2 (i) and $\widetilde { G }$ by Assumption 3(i), and supp $( \bar { \mu } _ { T } ^ { q } ) = \mathbb { R } ^ { D ( q + 1 ) }$ by Lemma 1, so ${ \widetilde { G } } = G$ everywhere, exactly as in the proof of Theorem 1. □

## A.5 UNIFORM MOMENT BOUNDS

In order to prove some ergodicity results in Section A.6, we will need some estimates on the moments of the process both point-wise and in the segment space. This section provides: first a bound on the one-point moments $\mathbb { E } \Vert X _ { t } - c \Vert ^ { 2 p }$ , obtained from Ito’s formula and a Halanay comparisonˆ to handle the delayed feedback, second a lifting of these to the segment norm via the Burkholder– Davis–Gundy inequality (Karatzas & Shreve, 1991).

We first give a small lemma combining the radial bound and the residual bound from the Assumption 6 to get a bound on the whole space.

Lemma 2 (Global dissipativity estimate). Under Assumption $\delta ,$ set $b _ { a } : = \beta _ { a } + \varepsilon _ { a } f o r a = 1 , \ldots , q ,$ so that $\begin{array} { r } { B = \sum _ { a = 1 } ^ { q } b _ { a } \dot { < } \beta _ { 0 } } \end{array}$ by Equation 13. Combining the radial bound outside $\bar { B } _ { R } ( c )$ with the residual bound inside it, we obtain,for all $( x _ { 0 } , x _ { 1 } , \dots , x _ { q } ) \in \mathbb { R } ^ { D ( q + 1 ) }$

$$
\langle x _ { 0 } - c , G ( x _ { 0 } , x _ { 1 } , \ldots , x _ { q } ) \rangle \leq A - \beta _ { 0 } \| x _ { 0 } - c \| ^ { 2 } + \sum _ { a = 1 } ^ { q } b _ { a } \| x _ { a } - c \| ^ { 2 } ,\tag{26}
$$

where $A : = \operatorname* { m a x } \{ \alpha , C _ { R } + \beta _ { 0 } R ^ { 2 } \}$ and $C _ { R }$ is the constant defined in Equation $I 2 .$

Proof. Outside the ball, Equation 26 follows from the radial bound Equation 11, since $A \geq \alpha$ and $b _ { a } \geq \beta _ { a }$ . Inside it, $\beta _ { 0 } \| x _ { 0 } - c \| ^ { 2 } \leq \beta _ { 0 } R ^ { 2 }$ , so adding and subtracting this term in the residual bound Equation 12 and using $b _ { a } \geq \varepsilon _ { a }$ gives Equation 26 with the constant $\overline { { C _ { R } } } + \beta _ { 0 } R ^ { 2 } \le A$ □

Lemma 3 (One-point moment bounds). Suppose that Assumptions $I , 2$ and 6 hold. For every integer $p \geq 1$ with $\mathbb { E } \| X _ { 0 } ^ { \mathrm { s e g } } - c \| _ { \mathcal { C } } ^ { 2 p } < \infty$ there are constants $\gamma _ { p } > 0$ and $K _ { p } ^ { \star } < \infty ,$ , depending only on $\Sigma , p ,$ the delays and the constantsfrom the assumption bounds, such that

$$
\mathbb { E } \| X _ { t } - c \| ^ { 2 p } \leq \mathbb { E } \| X _ { 0 } ^ { \mathrm { s e g } } - c \| _ { \mathcal { C } } ^ { 2 p } e ^ { - \gamma _ { p } t } + K _ { p } ^ { \star }\tag{27}
$$

Proof. Write $Q : = \Sigma \Sigma ^ { \top } , \Phi _ { 1 } ( x ) : = \| x - c \| ^ { 2 }$ and $\Phi _ { p } : = \Phi _ { 1 } ^ { p }$ , so that $\Phi _ { p } ( X _ { t } ) = \| X _ { t } - c \| ^ { 2 p }$ , and recall $b _ { a } = \beta _ { a } + \varepsilon _ { a }$ and $\textstyle B = \sum _ { a = 1 } ^ { q } b _ { a }$ from Lemma 2, and set $M _ { 0 } : = \mathbb { E } \| X _ { 0 } ^ { \mathrm { s e g } } - c \| _ { \mathcal { C } } ^ { 2 p } < \infty$

Step 1: Generator boundfor $\Phi _ { p } .$ Since $\nabla \Phi _ { p } = 2 p \Phi _ { 1 } ^ { p - 1 } ( x - c )$ and $D ^ { 2 } \Phi _ { p } = 2 p \Phi _ { 1 } ^ { p - 1 } I + 4 p ( p -$ 1) $\Phi _ { 1 } ^ { p - 2 } ( x - c ) ( x - c ) ^ { \top }$ , Ito’s formula applied to Equation 1 givesˆ

$$
\mathcal { L } \Phi _ { p } = 2 p \Phi _ { 1 } ^ { p - 1 } ( { \boldsymbol { x } } _ { 0 } ) \left. { \boldsymbol { x } } _ { 0 } - { \boldsymbol { c } } , { \boldsymbol { G } } \right. + p \Phi _ { 1 } ^ { p - 1 } ( { \boldsymbol { x } } _ { 0 } ) \operatorname { t r } ( { \boldsymbol { Q } } ) + 2 p ( p - 1 ) \Phi _ { 1 } ^ { p - 2 } ( { \boldsymbol { x } } _ { 0 } ) \bigl \| { \boldsymbol { \Sigma } } ^ { \top } ( { \boldsymbol { x } } _ { 0 } - { \boldsymbol { c } } ) \bigr \| ^ { 2 } .
$$

Note that the last term vanishes when $p = 1$ . Bounding $\| \Sigma ^ { \top } ( x _ { 0 } - c ) \| ^ { 2 } \leq \| \Sigma \| _ { \mathrm { o p } } ^ { 2 } \Phi _ { 1 } ( x _ { 0 } )$ and inserting Equation 26 yields,

$$
\mathcal { L } \Phi _ { p } \leq - 2 p \beta _ { 0 } \Phi _ { p } ( x _ { 0 } ) + 2 p \sum _ { a = 1 } ^ { q } b _ { a } \Phi _ { 1 } ^ { p - 1 } ( x _ { 0 } ) \Phi _ { 1 } ( x _ { a } ) + k _ { p } \Phi _ { 1 } ^ { p - 1 } ( x _ { 0 } ) ,\tag{28}
$$

with $k _ { p } : = 2 p A + p \mathrm { t r } ( Q ) + 2 p ( p - 1 ) \| \Sigma \| _ { \mathrm { o p } } ^ { 2 }$ . The delayed terms $\Phi _ { 1 } ( x _ { a } )$ are the reason a plain Gronwall argument does not apply and a delay comparison is needed using Halanay’s inequality¨ (Halanay, 1966).

Step 2: Scaled Young inequality and the dissipativity margin. For $p > 1$ , Young’s inequality with conjugate exponents $p / ( p - 1 )$ and p gives

$$
\begin{array} { r } { \Phi _ { 1 } ^ { p - 1 } ( x _ { 0 } ) \Phi _ { 1 } ( x _ { a } ) \ \le \ \frac { p - 1 } { p } \Phi _ { p } ( x _ { 0 } ) + \frac { 1 } { p } \Phi _ { p } ( x _ { a } ) , } \end{array}
$$

so the cross terms in Equation 28 are bounded by $\begin{array} { r } { 2 ( p - 1 ) B \Phi _ { p } ( x _ { 0 } ) + 2 \sum _ { a } b _ { a } \Phi _ { p } ( x _ { a } ) } \end{array}$

The lower-order term $k _ { p } \Phi _ { 1 } ^ { p - 1 } ( x _ { 0 } )$ is treated separately using a scale $\eta _ { p }$ . Fix any

$$
\eta _ { p } \in \big ( 0 , 2 p ( \beta _ { 0 } - B ) \big ) , \qquad C _ { p } : = \operatorname* { s u p } _ { y \ge 0 } \bigl ( k _ { p } y ^ { p - 1 } - \eta _ { p } y ^ { p } \bigr ) = \frac { k _ { p } ^ { p } ( p - 1 ) ^ { p - 1 } } { p ^ { p } \eta _ { p } ^ { p - 1 } } < \infty ,
$$

so that $k _ { p } \Phi _ { 1 } ^ { p - 1 } ( x _ { 0 } ) \leq \eta _ { p } \Phi _ { p } ( x _ { 0 } ) + C _ { p } ;$ for $p = 1$ this term is already the constant $k _ { 1 }$ and we set $\eta _ { 1 } : = 0 , \overline { { C } } _ { 1 } : = k _ { 1 }$

Substituting both bounds into Equation 28 yields the delay-dissipative estimate

$$
\mathcal { L } \Phi _ { p } \le - a _ { p } \Phi _ { p } ( x _ { 0 } ) + 2 \sum _ { a = 1 } ^ { q } b _ { a } \Phi _ { p } ( x _ { a } ) + C _ { p } , \qquad a _ { p } : = 2 p \beta _ { 0 } - 2 ( p - 1 ) B - \eta _ { p } .\tag{29}
$$

The role of the free parameter is that the margin required by the comparison step is now automatic:

$$
a _ { p } - 2 B = 2 p ( \beta _ { 0 } - B ) - \eta _ { p } ~ > ~ 0 ~ \mathrm { ~ f o r ~ e v e r y ~ } p \ge 1 ,
$$

by the choice of $\eta _ { p } . \mathrm { ~ A ~ }$ pplying an unscaled Young inequality to the lower-order term would instead add $k _ { p }$ to the coefficient of $\Phi _ { p } ( x _ { 0 } )$ and can make $a _ { p }$ negative, which is what must be avoided when using Halanay inequality.

Step 3: A priori integrability. Before applying the inequality, we must show that $m _ { p } ( t ) : = \mathbb { E } \Phi _ { p } ( X _ { t } )$ is finite and locally absolutely continuous. Let $\zeta _ { n } : = \operatorname* { i n f } \{ s \geq 0 : \| X _ { s } - c \| \geq n \}$ , which increases to +∞ a.s. by Proposition 1, and set

$$
S _ { n } ( t ) : = \operatorname* { m a x } \biggr \{ \| X _ { 0 } ^ { \mathrm { s e g } } - c \| _ { \mathcal { C } } ^ { 2 p } , \operatorname* { s u p } _ { u \leq t } \Phi _ { p } ( X _ { u \wedge \zeta _ { n } } ) \biggl \} ,
$$

so that $\mathbb { E } S _ { n } ( t ) \le M _ { 0 } + n ^ { 2 p } < \infty$ and $S _ { n } ( s )$ dominates every delayed term $\Phi _ { p } ( X _ { s - \tau _ { a } } )$ for $s \leq t$ — those with $s < \tau _ { a }$ being read off the initial segment.

$\mathrm { I t } \hat { \mathrm { o } } ^ { \cdot } \mathrm { s }$ formula for $\Phi _ { p } ( X _ { u \wedge \zeta _ { n } } )$ together with Equation 29, after discarding the nonpositive term $- a _ { p } \Phi _ { p } .$ , gives the following estimate

$$
\mathbb { E } S _ { n } ( t ) \ \leq \ 2 M _ { 0 } + C _ { p } t + 2 B \int _ { 0 } ^ { t } \mathbb { E } S _ { n } ( s ) d s + \mathbb { E } \operatorname* { s u p } _ { u \leq t } \bigl | N _ { u \wedge \zeta _ { n } } \bigr | .
$$

where we wrote $\begin{array} { r } { N _ { t } : = \int _ { 0 } ^ { t } 2 p \Phi _ { 1 } ^ { p - 1 } ( X _ { s } ) ( X _ { s } - c ) ^ { \top } \Sigma d W _ { s } } \end{array}$ for the martingale part.

Since $d [ N ] _ { s } \leq 4 p ^ { 2 } \Vert \Sigma \Vert _ { \mathrm { o p } } ^ { 2 } \Phi _ { 1 } ^ { 2 p - 1 } ( X _ { s } ) d s .$ , the Burkholder–Davis–Gundy inequality, the factorisation $\Phi _ { 1 } ^ { 2 p - 1 } = \Phi _ { p } \cdot \Phi _ { 1 } ^ { p - 1 }$ and Young’s inequality ${ \sqrt { u v } } \leq { \frac { 1 } { 4 } } u + v$ bound the last term by $\begin{array} { r } { \frac { 1 } { 2 } \mathbb { E } S _ { n } ( t ) + } \end{array}$ $\begin{array} { r } { C \int _ { 0 } ^ { t } ( 1 + \mathbb { E } S _ { n } ( s ) ) d s } \end{array}$ , with C depending only on p and $\| \Sigma \| _ { \mathrm { o p } } .$ . Absorbing $\textstyle { \frac { 1 } { 2 } } \mathbb { E } S _ { n } ( t )$ on the left and applying Gronwall’s lemma gives a bound uniform in ¨ n on every finite horizon; letting $n \to \infty$ and using Fatou’s lemma,

$$
\mathbb { E } \operatorname* { s u p } _ { u \leq H } \| X _ { u } - c \| ^ { 2 p } < \infty \qquad \mathrm { f o r ~ e v e r y ~ f i n i t e ~ } H .
$$

Consequently, by the same Burkholder–Davis–Gundy bound, E su $_ { \mathrm { p } _ { u } \le H } | N _ { u } | < \infty$ , so $N$ is a true martingale on finite horizons and expectations may be taken in Ito’s formula without stopping; the ˆ positive part of $\mathcal { L } \Phi _ { p }$ is integrable by Equation 29 and Ito’s identity then controls its negative part, soˆ $m _ { p }$ is locally absolutely continuous and, for almost every $t > 0$

$$
m _ { p } ^ { \prime } ( t ) \ \leq \ - a _ { p } m _ { p } ( t ) + 2 \sum _ { a = 1 } ^ { q } b _ { a } m _ { p } ( t - \tau _ { a } ) + C _ { p } .\tag{30}
$$

Step 4: Halanay comparison. Consider the characteristic equation

$$
\gamma _ { p } + 2 \sum _ { a = 1 } ^ { q } b _ { a } e ^ { \gamma _ { p } \tau _ { a } } = a _ { p } .\tag{31}
$$

Its left-hand side is strictly increasing in $\gamma _ { p } ,$ equals $2 B \ : < \ : a _ { p }$ at $\gamma _ { p } = 0$ by Step 2, and tends to infinity, so Equation 31 has a unique positive root $\gamma _ { p } \in ( 0 , a _ { p } ) ;$ when all $b _ { a }$ vanish it reduces to $\gamma _ { p } = \bar { a } _ { p } . \mathrm { P u t } \bar { K } _ { p } ^ { \star } : = C _ { p } / ( a _ { p } - 2 \bar { B ) }$ and

$$
w ( t ) : = M _ { 0 } e ^ { - \gamma _ { p } t } + K _ { p } ^ { \star } , \qquad t \geq - \tau _ { \operatorname* { m a x } } .
$$

A direct computation using Equation 31 shows that w satisfies Equation 30 with equality, and $w ( t ) \geq$ $M _ { 0 } \geq m _ { p } ( t )$ on $[ - \tau _ { \operatorname* { m a x } } , \bar { 0 } ]$ ] since $e ^ { - \gamma _ { p } t } \geq 1$ there. Set $u : = m _ { p } - w$ , so that $u \leq 0 \mathrm { o n } \left[ - \tau _ { \mathrm { m a x } } , 0 \right]$ and $\begin{array} { r } { u ^ { \prime } ( t ) \ \dot { \ } \leq \ - \dot { a _ { p } } u ( t ) + \bar { 2 } \sum _ { a } b _ { a } u ( t - \tau _ { a } ) } \end{array}$ almost everywhere. On $[ 0 , \tau _ { \mathrm { m i n } } ]$ all delayed values $u ( t - \tau _ { a } )$ are nonpositive, hence $( e ^ { a _ { p } t } u ) ^ { \prime } \leq 0$ and $u ( t ) \leq \dot { u } ( 0 ) e ^ { - a _ { p } t } \leq \dot { 0 } :$ since $\tau _ { \mathrm { m i n } }$ is the smallest delay, induction over the intervals $[ k \tau _ { \mathrm { m i n } } , ( k + 1 ) \tau _ { \mathrm { m i n } } ]$ propagates u $, \leq 0$ to all of $[ 0 , \infty )$ (for $q = 0$ ordinary Gronwall suffices). Therefore¨ $m _ { p } ( t ) \leq w ( t )$ for every $t \geq 0 ,$ , that is

$$
\begin{array} { r } { \mathbb { E } \| X _ { t } - c \| ^ { 2 p } \leq \mathbb { E } \| X _ { 0 } ^ { \mathrm { s e g } } - c \| _ { \mathcal { C } } ^ { 2 p } e ^ { - \gamma _ { p } t } + K _ { p } ^ { \star } , \qquad t \geq 0 , } \end{array}
$$

which is the announced bound, with $\gamma _ { p }$ the positive root of Equation 31 and $K _ { p } ^ { \star } = C _ { p } / ( a _ { p } -$ 2B). □

With the one-point bound in hand, we lift to the segment norm and record the stationary and finitehorizon consequences (parts (b) and (c) of the lemma).

Lemma 4 (Segment moment bounds). Under the assumptions of Lemma 3 and for the same range of $p ,$ write $U _ { p } ( \varphi ) : = \| \varphi - c \| _ { \mathcal { C } } ^ { 2 p } f o r \varphi \in \mathcal { C }$ and let $( P _ { t } ) _ { t \geq 0 }$ denote the semigroup of the segment process on C.

(a) There are constants $A _ { p } ^ { \mathrm { s e g } } , B _ { p } ^ { \mathrm { s e g } } < \infty$ and $\gamma _ { p } > 0$ such that

$$
P _ { t } U _ { p } ( \phi ) = \mathbb { E } _ { \phi } \| X _ { t } ^ { \mathrm { s e g } } - c \| _ { C } ^ { 2 p } \ \leq \ A _ { p } ^ { \mathrm { s e g } } e ^ { - \gamma _ { p } t } U _ { p } ( \phi ) + B _ { p } ^ { \mathrm { s e g } } , \qquad t \geq 0 , \phi \in \mathcal C .\tag{32}
$$

In particular $\begin{array} { r } { M _ { 2 p } ( \phi ) : = \operatorname* { s u p } _ { t \geq 0 } \mathbb { E } _ { \phi } \Vert X _ { t } ^ { \mathrm { s e g } } - c \Vert _ { \mathcal { C } } ^ { 2 p } \leq A _ { p } ^ { \mathrm { s e g } } U _ { p } ( \phi ) + B _ { p } ^ { \mathrm { s e g } } < \infty . } \end{array}$

(b) If the segment process admits an invariant probability measure $\pi ^ { \mathrm { s e g } }$ on ${ \mathcal { C } } ,$ then

$$
\int _ { \mathcal { C } } \left\| \eta - c \right\| _ { \mathcal { C } } ^ { 2 p } \pi ^ { \mathrm { s e g } } ( d \eta ) < \infty .\tag{33}
$$

(c) For every finite horizon H there is $C _ { p , H } < \infty$ with

$$
\mathbb { E } _ { \phi } \operatorname* { s u p } _ { t \leq H } \lVert X _ { t } ^ { \mathrm { s e g } } - c \rVert _ { \mathcal { C } } ^ { 2 p } \ \leq \ C _ { p , H } \big ( 1 + U _ { p } ( \phi ) \big ) , \qquad \phi \in \mathcal { C } ,\tag{34}
$$

which is uniform over any ball $B _ { R } : = \{ \phi \in \mathcal { C } : \| \phi - c \| c \leq R \}$

Proof. Recall $\Phi _ { 1 } ( x ) \ = \ \| x - c \| ^ { 2 } , \ \Phi _ { p } \ = \ \Phi _ { 1 } ^ { p }$ and $\sigma ^ { 2 } : = \| \Sigma \| _ { \mathrm { o p } } ^ { 2 } ,$ and note that $U _ { p } ( X _ { t } ^ { \mathrm { s e g } } ) =$ $\begin{array} { r } { \operatorname* { s u p } _ { u \in [ t - \tau _ { \operatorname* { m a x } } , t ] } \Phi _ { p } ( X _ { u } ) } \end{array}$ , so that (a) is exactly an upgrade of the one-point control of Lemma 3 to a supremum over a window of length $\tau _ { \mathrm { m a x } }$ . We keep the constants $\begin{array} { r } { b _ { i } , B = \sum _ { i = 1 } ^ { q } b _ { i } , C _ { p } , \gamma _ { p } } \end{array}$ and $K _ { p } ^ { \star }$ of that lemma, together with its generator bound Equation 29. All suprema below are first taken up to the localising time $\theta _ { N } : = \operatorname* { i n f } \{ s : \| X _ { s } - c \| > N \}$ ; the resulting constants do not depend on $\bar { N }$ , and $\theta _ { N } \uparrow \infty \mathrm { a . s }$ . by Proposition 1, so monotone convergence removes the localisation at the end of each step. The finite-horizon integrability established in the proof of Lemma 3 guarantees that every expectation written below is finite before the limit is taken. When $q = 0$ we use the conventions $\tau _ { \operatorname* { m a x } } = 0$ and ${ \mathcal { C } } = \mathbb { R } ^ { D }$ : then $U _ { p } ( X _ { t } ^ { \mathrm { s e g } } ) = \Phi _ { p } ( X _ { t } )$ , part (a) is Lemma 3 itself, and in Step 5 the windows of length $\tau _ { \mathrm { m a x } }$ are replaced by windows of unit length, to which the window estimate Equation 38 of Step 2 applies verbatim.

Step 1: A one-point envelope valid on negative times. Set $m _ { p } ( s ) : = \mathbb { E } _ { \phi } \Phi _ { p } ( X _ { s } )$ for $s \geq - \tau _ { \operatorname* { m a x } } .$ For $s \in [ - \tau _ { \operatorname* { m a x } } , 0 ]$ the path is the deterministic initial segment, so $m _ { p } ( s ) \leq { \bar { U } } _ { p } ( \phi )$ . Since moreover $m _ { p } ( s ) \dot { \leq } U _ { p } ( \phi ) e ^ { - \gamma _ { p } s } + K _ { p } ^ { \star }$ for $s \geq 0$ by Lemma 3, the inequality

$$
m _ { p } ( s ) \leq U _ { p } ( \phi ) e ^ { - \gamma _ { p } s } + K _ { p } ^ { \star }\tag{35}
$$

holds for every $s \geq - \tau _ { \operatorname* { m a x } }$

Step 2: Window estimate. Fix $0 \leq v <$ w with $w - v \leq \tau _ { \operatorname* { m a x } }$ . In integrated form, for $u \in [ v , w ]$

$$
\Phi _ { p } ( X _ { u } ) = \Phi _ { p } ( X _ { v } ) + \int _ { v } ^ { u } \mathcal { L } \Phi _ { p } ( X _ { s } ) d s + ( N _ { u } - N _ { v } ) ,
$$

with $\begin{array} { r } { N _ { s } : = \int _ { 0 } ^ { s } 2 p \Phi _ { 1 } ^ { p - 1 } ( X _ { r } ) ( X _ { r } - c ) ^ { \top } \Sigma d W _ { r } } \end{array}$ as in the previous lemma.

Discarding the favourable term $- a _ { p } \Phi _ { p } \leq 0$ in Equation 29, taking the supremum over $u \in [ v , w ]$ and then expectations,

$$
\mathbb { E } \operatorname* { s u p } _ { v \leq u \leq w } \Phi _ { p } ( X _ { u } ) \ \leq \ m _ { p } ( v ) + 2 \sum _ { i = 1 } ^ { q } b _ { i } \int _ { v } ^ { w } m _ { p } ( s - \tau _ { i } ) d s + C _ { p } ( w - v ) + \mathbb { E } \operatorname* { s u p } _ { v \leq u \leq w } \left| N _ { u } - N _ { v } \right| .\tag{36}
$$

Re-anchored at $v ,$ the increment $u \mapsto N _ { u } - N _ { v }$ is a continuous local martingale started at $0 ,$ so the Burkholder–Davis–Gundy inequality at exponent 1 applies with a universal constant $C _ { \mathrm { B D G } }$ . Using $d [ N ] _ { s } \leq 4 p ^ { 2 } \sigma ^ { 2 } \Phi _ { 1 } ^ { 2 p - 1 } ( X _ { s } )$ ds and the factorisation $\Phi _ { 1 } ^ { \mathbf { \dot { 2 } } p - 1 } = \Phi _ { p } \cdot \Phi _ { 1 } ^ { p - 1 }$

$$
\mathbb { E } \operatorname* { s u p } _ { v \leq u \leq w } \left| N _ { u } - N _ { v } \right| \leq 2 p \sigma C _ { \mathrm { B D G } } \mathbb { E } \left[ \Big ( \operatorname* { s u p } _ { v \leq u \leq w } \Phi _ { p } ( X _ { u } ) \Big ) ^ { 1 / 2 } \Big ( \int _ { v } ^ { w } \Phi _ { 1 } ^ { p - 1 } ( X _ { s } ) d s \Big ) ^ { 1 / 2 } \right] .
$$

Young’s inequality $\begin{array} { r } { a b \le \frac { 1 } { 2 } a ^ { 2 } + \frac { 1 } { 2 } b ^ { 2 } } \end{array}$ , with the constant absorbed into the second factor and $C _ { N } : =$ $( 2 p \sigma C _ { \mathrm { B D G } } ) ^ { 2 }$ , gives

$$
\mathbb { E } \operatorname* { s u p } _ { v \leq u \leq w } \left| N _ { u } - N _ { v } \right| \ \leq \ \frac { 1 } { 2 } \mathbb { E } \operatorname* { s u p } _ { v \leq u \leq w } \Phi _ { p } ( X _ { u } ) + \frac { C _ { N } } { 2 } \int _ { v } ^ { w } \mathbb { E } \Phi _ { 1 } ^ { p - 1 } ( X _ { s } ) d s ,\tag{37}
$$

and $\begin{array} { r } { \Phi _ { 1 } ^ { p - 1 } \le \frac { p - 1 } { p } \Phi _ { p } + \frac { 1 } { p } \le \Phi _ { p } + 1 } \end{array}$ bounds the last integral by $\begin{array} { r } { \int _ { v } ^ { w } ( 1 + m _ { p } ( s ) ) \ d s } \end{array}$ ds. Substituting Equation 37 into Equation 36 and absorbing the half-supremum into the left-hand side,

$$
\mathbb { E } \operatorname* { s u p } _ { v \leq u \leq w } \Phi _ { p } ( X _ { u } ) \ \leq \ 2 m _ { p } ( v ) + 4 \sum _ { i = 1 } ^ { q } b _ { i } \int _ { v } ^ { w } m _ { p } ( s - \tau _ { i } ) d s + 2 C _ { p } ( w - v ) + C _ { N } \int _ { v } ^ { w } \left( 1 + m _ { p } ( s ) \right) d s .\tag{38}
$$

Step 3: Geometric drift for the segment process. Let $t \geq \tau _ { \operatorname* { m a x } }$ and apply Equation 38 with $v =$ $t - \tau _ { \operatorname* { m a x } }$ and $w = t$ . Every time argument occurring on its right-hand side lies in $\left[ t - 2 \tau _ { \operatorname* { m a x } } , t \right]$ so Equation 35 bounds each of them by $e ^ { 2 \gamma _ { p } \tau _ { \mathrm { m a x } } } U _ { p } ( \breve { \phi } ) e ^ { - \gamma _ { p } t } + \breve { K } _ { p } ^ { \star }$ . Writing $\Lambda _ { p } : = \dot { 2 } + 4 B \tau _ { \mathrm { m a x } } \dot { + }$ $C _ { N } \tau _ { \mathrm { m a x } }$ , we obtain Equation 32 for $t \geq \tau _ { \operatorname* { m a x } }$ with

$$
A _ { p } ^ { \mathrm { s e g } } : = \Lambda _ { p } e ^ { 2 \gamma _ { p } \tau _ { \mathrm { m a x } } } , \qquad B _ { p } ^ { \mathrm { s e g } } : = \Lambda _ { p } K _ { p } ^ { \star } + 2 C _ { p } \tau _ { \mathrm { m a x } } + C _ { N } \tau _ { \mathrm { m a x } } .
$$

For $0 \leq t < \tau _ { \operatorname* { m a x } }$ the window $\left[ t - \tau _ { \operatorname* { m a x } } , t \right]$ straddles the origin, and the initial history is not an Itoˆ process on its negative part. We therefore split

$$
U _ { p } ( X _ { t } ^ { \mathrm { s e g } } ) = \operatorname* { s u p } _ { u \in [ t - \tau _ { \operatorname* { m a x } } , t ] } \Phi _ { p } ( X _ { u } ) \ \leq \ U _ { p } ( \phi ) + \ \operatorname* { s u p } _ { 0 \leq u \leq t } \Phi _ { p } ( X _ { u } ) ,
$$

bound the first term directly and the second by Equation 38 with $v = 0 , w = t ,$ , whose length is at most $\tau _ { \mathrm { m a x } }$ . All time arguments in the latter lie in $[ - \tau _ { \operatorname* { m a x } } , t ] \subseteq [ t - 2 \tau _ { \operatorname* { m a x } } , t ]$ , so the computation above bounds the expectation of the second term by $\bar { A } _ { p } ^ { \mathrm { s e g } } e ^ { - \gamma \bar { p } ^ { } t } U _ { p } \bar { ( } \phi ) + B _ { p } ^ { \mathrm { s e g } }$ , while the first term is at most e<sup>γpτmax</sup> $e ^ { - \gamma _ { p } t } U _ { p } ( \phi )$ since $e ^ { - \gamma _ { p } t } \geq e ^ { - \gamma _ { p } \tau _ { \mathrm { m a x } } }$ <sup>x</sup> on this range. Replacing $A _ { n } ^ { \mathrm { s e g } }$ by $A _ { p } ^ { \mathrm { s e g } } + e ^ { \gamma _ { p } }$ τ<sub>max</sub> therefore extends Equation 32 to all $t \geq 0$ . Taking the supremum over $t \geq 0$ gives the uniform bound $M _ { 2 p } ( \phi ) \leq \dot { A _ { p } ^ { \mathrm { s e g } } } U _ { p } ( \phi ) + B _ { p } ^ { \mathrm { s e g } }$ announced in (a), established without any growth assumption on $G .$

Step 4: Stationary integrability (b). Choose $h > 0$ so that $\kappa : = A _ { n } ^ { \mathrm { s e g } } e ^ { - \gamma _ { p } h } < 1$ and write $\varrho : = B _ { p } ^ { \mathrm { s e g } }$ so that Equation 32 reads $P _ { h } U _ { p } \le \kappa U _ { p } + \varrho$ pointwise on C. One cannot integrate this against $\bar { \pi } ^ { \mathrm { s e g } }$ and cancel the two sides, since $\int U _ { p } { \dot { d } } \pi ^ { \mathrm { s e g } }$ is not yet known to be finite. Therefore, set $u _ { N } : =$ $U _ { p } \wedge N$ , a bounded measurable function. Concavity of v $\mapsto \boldsymbol { v } \wedge \boldsymbol { N }$ and Jensen’s inequality give $P _ { h } u _ { N } \le ( P _ { h } U _ { p } ) \wedge N \le ( \kappa U _ { p } + \varrho ) \wedge N$ , and invariance of $\pi ^ { \mathrm { s e g } }$ applied to the bounded function $u _ { N }$ yields

$$
\int _ { \mathcal { C } } \Bigl [ u _ { N } - \bigl ( \bigl ( \kappa U _ { p } + \varrho \bigr ) \wedge N \bigr ) + \varrho \Bigr ] d \pi ^ { \mathrm { s e g } } \ \le \ \varrho .
$$

The integrand is nonnegative, because $\kappa \leq 1$ implies $( \kappa U _ { p } + \varrho ) \wedge N \leq ( U _ { p } \wedge N ) + \varrho = u _ { N } + \varrho .$ , and it converges pointwise to $( 1 - \kappa ) U _ { p }$ as $N \to \infty$ . Fatou’s lemma therefore gives $\begin{array} { r } { \left( 1 - \kappa \right) \int _ { \mathcal { C } } U _ { p } d \pi ^ { \mathrm { s e g } } \leq } \end{array}$ $\varrho < \infty$ , which is Equation 33. The argument applies separately for each $p$ and uses no moment of $\pi ^ { \mathrm { s e g } }$ in its hypotheses.

Step 5: Finite horizon (c). Cover $[ 0 , H ]$ by the $n : = \lceil H / \tau _ { \mathrm { m a x } } \rceil$ windows $[ k \tau _ { \operatorname* { m a x } } , ( k + 1 ) \tau _ { \operatorname* { m a x } } ] \ n$ $[ 0 , H ] , k = 0 , \dots , n - 1$ , each of length at most $\tau _ { \mathrm { m a x } }$ . Since the supremum over a union is at most the sum of the suprema,

$$
\mathbb { E } _ { \phi } \operatorname* { s u p } _ { t \leq H } U _ { p } ( X _ { t } ^ { \mathrm { s e g } } ) = \mathbb { E } _ { \phi } \operatorname* { s u p } _ { \substack { u \in [ - \tau _ { \operatorname* { m a x } } , H ] } } \Phi _ { p } ( X _ { u } ) \ \leq \ U _ { p } ( \phi ) + \sum _ { k = 0 } ^ { n - 1 } \mathbb { E } _ { \phi } \operatorname* { s u p } _ { \substack { u \in [ k \tau _ { \operatorname* { m a x } } , ( k + 1 ) \tau _ { \operatorname* { m a x } } ] \wedge H } } \Phi _ { p } ( X _ { u } ) ,
$$

and each summand is bounded by Equation 38 together with the envelope Equation 35, which gives $m _ { p } ( s ) \leq U _ { p } ( \phi ) + K _ { p } ^ { \star }$ for every $s \ge - \tau _ { \operatorname* { m a x } }$ . Collecting the n contributions yields Equation 34 with a constant $C _ { p , H }$ depending only on $p , H$ , the delays and the constants of Lemma 3, and in particular uniform over $\phi \in B _ { R }$ for each fixed $R .$ □

## A.6 ERGODICITY OF THE TRUE PROCESS

Having those moment estimates both in the segment space and point-wise at hand, it is now possible to show that the true process admits a unique invariant probability and that we have geometric ergodicity in Wasserstein-d distance for some distance on the segment space. The proof is based on the application of the generalized Harris theorem for delayed SDEs by Hairer et al. (2011).

Besides the Lyapunov condition provided by Lemma 4, Harris’ theorem needs the transition laws started from any two histories in a bounded set to overlap. For SDDEs this is the tricky part: indeed, the two initial segments may differ over the whole window $[ - \tau _ { \operatorname* { m a x } } , 0 ]$ , while the noise only acts on the present. The idea of the proof is to steer one copy of the process onto the other with a control acting on [0, h]. After time $h > \tau _ { \mathrm { m a x } }$ the two segments coincide, and since Σ is invertible, Girsanov’s theorem shows that the price paid for this control is a change of measure with a bounded energy.

Lemma 5 (Uniform overlap of the segment transition kernels). Under Assumptions 1, 2 and 6, fix $h > \tau _ { \mathrm { m a x } }$ and $R _ { 0 } > 0 ,$ , and set $B _ { R _ { 0 } } : = \{ \varphi \in \mathcal { C } : \| \varphi - c \| _ { \mathcal { C } } \leq R _ { 0 } \}$ . There exists $\delta > 0$ depending on h and $R _ { 0 }$ (and on $G$ through a local Lipschitz constant, see Equation 41) such that

$$
\operatorname* { s u p } _ { \varphi , \psi \in B _ { R _ { 0 } } } \lVert P _ { h } ( \varphi , \cdot ) - P _ { h } ( \psi , \cdot ) \rVert _ { \mathrm { T V } } \leq 1 - \delta .\tag{39}
$$

Proof. Fix $\varphi , \psi \in B _ { R _ { 0 } }$ , let X start from $\varphi ,$ and put $t _ { 1 } : = h - \tau _ { \operatorname* { m a x } } > 0$ and $v : = \| \varphi - \psi \| _ { \mathcal { C } } \leq 2 R _ { 0 }$ Define the deterministic difference pilot path

$$
g ( t ) : = \left\{ \begin{array} { l l } { \varphi ( t ) - \psi ( t ) , } & { - \tau _ { \operatorname* { m a x } } \leq t \leq 0 , } \\ { ( 1 - t / t _ { 1 } ) \big ( \varphi ( 0 ) - \psi ( 0 ) \big ) , } & { 0 \leq t \leq t _ { 1 } , } \\ { 0 , } & { t _ { 1 } \leq t \leq h . } \end{array} \right.
$$

Then g is continuous, $\operatorname* { s u p } _ { t } \| g ( t ) \| \leq v$ , and its restriction to $[ 0 , h ]$ is absolutely continuous with $\Vert g ^ { \prime } ( t ) \Vert \leq ( v / t _ { 1 } ) \mathbf { 1 } _ { ( 0 , t _ { 1 } ) } ( t )$ almost everywhere.

Step 1: Localized control. The finite-horizon estimate in Lemma 4 gives

$$
\operatorname* { s u p } _ { \varphi \in B _ { R _ { 0 } } } \mathbb { E } _ { \varphi } \operatorname* { s u p } _ { 0 \leq t \leq h } \| X _ { t } - c \| ^ { 2 } < \infty .
$$

Consequently, we can choose a deterministic $M > R _ { 0 }$ such that, with $\sigma : = \operatorname* { i n f } \{ t \geq 0 : \| X _ { t } - c \| \geq$ M},

$$
\operatorname* { i n f } _ { \varphi \in B _ { R _ { 0 } } } \mathbb { P } _ { \varphi } ( \sigma > h ) \geq { \frac { 1 } { 2 } } .
$$

Write $Z _ { t } : = ( X _ { t } , X _ { t - \tau _ { 1 } } , \ldots , X _ { t - \tau _ { q } } )$ and $g _ { t } ^ { \mathrm { a r g } } : = ( g ( t ) , g ( t - \tau _ { 1 } ) , \ldots , g ( t - \tau _ { q } ) )$ . Define the progressively measurable control

$$
u _ { t } : = { { \bf 1 } } _ { \{ t < \sigma \} } \Sigma ^ { - 1 } \big [ G ( Z _ { t } ) - G ( Z _ { t } - g _ { t } ^ { \mathrm { a r g } } ) - g ^ { \prime } ( t ) \big ] , \qquad 0 \leq t \leq h .\tag{40}
$$

All arguments in this expression belong to a fixed compact subset of R $D ( q { + } 1 )$ , uniformly over $\varphi , \psi \in$ $\boldsymbol { B } _ { R _ { 0 } }$ in the finite times $t < \sigma$ . Let L be a Lipschitz constant of G on a ball containing this set. Since $\| \bar { g } _ { t } ^ { \mathrm { a r g } } \| \leq \sqrt { q + 1 } \ i$ v, the control has the deterministic energy bound

$$
\int _ { 0 } ^ { h } \| u _ { t } \| ^ { 2 } d t \leq 8 R _ { 0 } ^ { 2 } \| \Sigma ^ { - 1 } \| _ { \mathrm { o p } } ^ { 2 } \left( ( q + 1 ) L ^ { 2 } h + \frac { 1 } { t _ { 1 } } \right) = : K _ { h , R _ { 0 } } < \infty .\tag{41}
$$

Set $Y _ { t } = X _ { t } - g ( t )$ up to σ, and, when $\sigma < h ,$ , continue $Y$ after σ using the original SDDE with the same Brownian motion and its segment at σ as initial condition. Strong existence and pathwise uniqueness justify this continuation. By substitution, $Y$ satisfies

$$
\begin{array} { r } { d Y _ { t } = G ( Y _ { t } , Y _ { t - \tau _ { 1 } } , \dots , Y _ { t - \tau _ { q } } ) d t + \Sigma d W _ { t } + \Sigma u _ { t } d t , \qquad Y _ { 0 } ^ { \mathrm { s e g } } = \psi . } \end{array}
$$

On the event $E : = \{ \sigma > h \}$ , the two terminal segments agree because $g = 0 \mathrm { o n } [ h - \tau _ { \mathrm { m a x } } , h ]$ . Thus

$$
Y _ { h } ^ { \mathrm { s e g } } = X _ { h } ^ { \mathrm { s e g } } \quad \mathrm { o n } E , \qquad \mathrm { w i t h } \mathbb { P } ( E ) \geq \frac { 1 } { 2 } .
$$

Step 2: Correction ofthe controlled law. Define

$$
\Lambda : = \exp \left( - \int _ { 0 } ^ { h } u _ { t } \cdot d W _ { t } - \frac 1 2 \int _ { 0 } ^ { h } \| u _ { t } \| ^ { 2 } d t \right) , \qquad d \mathbb { Q } : = \Lambda d \mathbb { P } .
$$

The deterministic bound Equation 41 implies Novikov’s condition. Under $\mathbb { Q } ,$ the process $W _ { t } +$ $\textstyle \int _ { 0 } ^ { t } u _ { s }$ ds is Brownian, so uniqueness in law gives $\mathcal { L } _ { \mathbb { Q } } ( Y _ { h } ^ { \mathrm { s e g } } ) = P _ { h } ( \psi , \cdot )$ . The exponential martingale with integrand u also has expectation one, and therefore

$$
\mathbb { E } [ \Lambda ^ { - 1 } ] = \mathbb { E } \left[ \exp \left( \int _ { 0 } ^ { h } u _ { t } \cdot d W _ { t } - \frac { 1 } { 2 } \int _ { 0 } ^ { h } \| u _ { t } \| ^ { 2 } d t \right) \exp \left( \int _ { 0 } ^ { h } \| u _ { t } \| ^ { 2 } d t \right) \right] \leq e ^ { K _ { h , R _ { 0 } } } .
$$

Set $\ell : = e ^ { - K _ { h , R _ { 0 } } } / 4$ . Markov’s inequality gives $\mathbb { P } ( \Lambda < \ell ) \le 1 / 4$ , hence

$$
\mathbb { E } \left[ \mathbf { 1 } _ { E } \operatorname* { m i n } ( 1 , \Lambda ) \right] \geq \ell \mathbb { P } \big ( E \cap \{ \Lambda \geq \ell \} \big ) \geq \frac { e ^ { - K _ { h , R _ { 0 } } } } { 1 6 } .
$$

For a Borel set $A \subseteq { \mathcal { C } }$ , define

$$
\nu ( A ) : = \mathbb { E } \bigl [ \mathbf { 1 } _ { E } \operatorname* { m i n } ( 1 , \Lambda ) \mathbf { 1 } _ { \{ X _ { h } ^ { \mathrm { s e g } } \in A \} } \bigr ] .
$$

Since $X _ { h } ^ { \mathrm { s e g } } = Y _ { h } ^ { \mathrm { s e g } }$ on $E ,$ , this measure is dominated by both $P _ { h } ( \varphi , \cdot )$ and $P _ { h } ( \psi , \cdot )$ . Its mass is at least $e ^ { - K _ { h , R _ { 0 } } } / 1 6$ , which proves Equation 39 with $\delta : = e ^ { - K _ { h , R _ { 0 } } } / 1 6$ □

Theorem 3 (Ergodicity: existence, uniqueness, exponential convergence). Under Assumptions 1, 2 and 6, the segment process admits a unique invariant probability measure $\pi ^ { \mathrm { s e g } }$ on ${ \hat { \boldsymbol { C } } } .$ Write $P _ { t } ( \varphi , \cdot ) : = \mathcal { L } ( \breve { X } _ { t } ^ { \mathrm { s e g } } \ | \ \cdot \ X _ { 0 } ^ { \mathrm { s e g } } = \varphi )$ and $V ( \mathbf { \dot { \varphi } } ) : = \| \varphi - \mathbf { \dot { c } } \| _ { \mathcal L } ^ { 2 }$ , where c also denotes the constant segment with value c. The invariant measure satisfies

$$
\int _ { \mathcal { C } } \lVert \eta - c \rVert _ { \mathcal { C } } ^ { 2 p } \pi ^ { \mathrm { s e g } } ( d \eta ) < \infty , \qquad p \ge 1 a n i n t e g e r .
$$

There exist $C \geq 1$ and $\gamma > 0$ such that, for every $\epsilon > 0 ,$ , the metric $d ( \varphi , \psi ) : = 1 \wedge \epsilon ^ { - 1 } \| \varphi - \psi \| _ { \mathcal { C } }$ satisfies

$$
\mathcal { W } _ { d } \big ( P _ { t } ( \varphi , \cdot ) , \pi ^ { \mathrm { s e g } } \big ) \leq \| P _ { t } ( \varphi , \cdot ) - \pi ^ { \mathrm { s e g } } \| _ { \mathrm { T V } } \leq C e ^ { - \gamma t } \big ( 1 + V ( \varphi ) \big ) , \qquad t \geq 0 , \quad \varphi \in \mathcal { C } .\tag{15}
$$

Proof. By Proposition 1, time-homogeneity and pathwise uniqueness define a Markov semigroup $( P _ { t } ) _ { t \geq 0 }$ on C.

Step 1: Lyapunov estimate. For $V ( \varphi ) \ = \ \| \varphi - c \| _ { \mathcal C } ^ { 2 }$ , the decaying segment-moment bound in Lemma 4 provides constants $A _ { V } \geq 1 , B _ { V } < \infty$ and $\gamma _ { V } > 0$ such that

$$
P _ { t } V ( \varphi ) \leq A _ { V } e ^ { - \gamma _ { V } t } V ( \varphi ) + B _ { V } , \qquad t \geq 0 .\tag{42}
$$

Choose $h > \tau _ { \operatorname* { m a x } }$ sufficiently large that $A _ { V } e ^ { - \gamma _ { V } h } \ < \ 1$ . Every sublevel set $\{ V \ \leq \ S \}$ is the supremum-norm ball $B _ { \sqrt { S } }$ , and Lemma 5 gives uniform transition overlap on this set for $P _ { h }$ . Since Lemma 5 holds for every radius $R _ { 0 }$ at the same time $h ,$ , this overlap is available on every sublevel set of V, however large; in particular, the size of the level set required in Harris’ theorem (of order $B _ { V } / ( 1 - A _ { V } e ^ { - \gamma _ { V } h } ) \big )$ imposes no restriction.

Step 2: Harris’ theorem in total variation. The estimate Equation 42 is a Lyapunov condition, and Equation 39 is precisely the pairwise total-variation small-set condition in Hairer et al. (2011, Theorem 1.5). The choice of h satisfies the time requirement $h > \gamma _ { V } ^ { - 1 } \log A _ { V }$ in that theorem. It follows that $( P _ { t } ) _ { t \geq 0 }$ has a unique invariant probability measure $\pi ^ { \mathrm { s e g } }$ and that, for some $C \geq 1$ and $\gamma > 0$

$$
\| P _ { t } ( \varphi , \cdot ) - \pi ^ { \mathrm { s e g } } \| _ { \mathrm { T V } } \leq C e ^ { - \gamma t } \big ( 1 + V ( \varphi ) \big ) , \qquad t \geq 0 , \quad \varphi \in \mathcal C .
$$

Only pairwise overlap is required here; no common minorising measure for an entire sublevel set is asserted.

Step 3: Stationary moments and Wasserstein convergence. Now that an invariant measure exists, the stationary-integrability part of Lemma 4 gives all the stated polynomial moments. For any two probability measures $\mu , \nu$ on ${ \mathcal { C } } ,$ a maximal coupling has probability $\| \mu - \nu \| _ { \mathrm { T V } }$ of unequal coordinates. Since $d \leq 1$ and $d ( \varphi , \varphi ) = 0$ , this coupling gives $\mathcal { \bar { W } } _ { d } ( \mu , \nu ) \overset { \cdot } { \leq } \ddot { \lVert \mu - \nu \rVert _ { \mathrm { T V } } }$ . Combining this inequality with the preceding total-variation estimate proves Equation 15. □

Remark 1 (Stabilisation of the identifiability constants). Under Theorem 3, averaging Equation 15 over $t \in [ \tau _ { \operatorname* { m a x } } , T ]$ against the initial law and applying the evaluation map at the lags gives

$$
\| \bar { \mu } _ { T } ^ { q } - \mu _ { \infty } ^ { q } \| _ { \mathrm { T V } } \leq \frac { C \bigl ( 1 + \mathbb { E } V ( X _ { 0 } ^ { \mathrm { s e g } } ) \bigr ) } { \gamma ( T - \tau _ { \mathrm { m a x } } ) } \bigl ( e ^ { - \gamma \tau _ { \mathrm { m a x } } } - e ^ { - \gamma T } \bigr ) ,
$$

where $\mu _ { \infty } ^ { q }$ is the stationary joint law of $( X _ { 0 } , X _ { - \tau _ { 1 } } , \ldots , X _ { - \tau _ { q } } )$ under $\pi ^ { \mathrm { s e g } }$ . Total-variation convergence alone does not control squared errors of an unbounded drift (the stationary drift need not even be square-integrable). If, in addition, $\| G \| ^ { 2 }$ is uniformly integrable with respect to the family $\{ \bar { \mu } _ { T } ^ { q } \} _ { T > \tau _ { \mathrm { m a x } } } \cup \{ \mu _ { \infty } ^ { q } \}$ , then the signal strengths $s _ { i j } ^ { a }$ and hence $s _ { \mathrm { m i n } }$ stabilize as $\bar { T } \to \infty$ to their stationary counterparts, and the λ window Equation 22 scales linearly in $T$ with an asymptotically constant slope. This is what makes the finite-horizon constants ofTheorem 2 meaningful uniformly in $T ,$ rather than per-(T, initial law). The finite-sample analysis of Section A.8 avoids this issue by working directly under the stationary law and imposing a finite stationary drift energy in Equation 48 explicitly.

## A.7 MIXING OF THE SAMPLED PROCESS

The finite-sample analysis of Section A.8 requires the sampled lagged state to forget its past at an exponential rate, in the sense of $\beta .$ -mixing. Since β-mixing is a total-variation property, a Wasserstein contraction of the segment process would not be enough. Theorem 3, however, already provides geometric ergodicity in total variation: despite the infinite-dimensional state space, the noise is additive and non-degenerate, so that for $h > \tau _ { \mathrm { m a x } }$ the kernels $P _ { h } ( \varphi , \cdot )$ and $P _ { h } ( \psi , \cdot )$ overlap (Lemma 5). The mixing rate is therefore a corollary of Theorem 3. We first bound the $\beta .$ -coefficient of a stationary Markov process by the π-average distance of its kernel from equilibrium (Lemma 6), and then pass from the segment process to the lagged state and to its deterministic subsamples by monotonicity (Theorem 5). No assumption beyond those of Theorem 3 is needed, and the non-delayed case $q = 0$ is included.

Definition 1 (β-mixing coefficient). We normalize the total-variation distance of two probability measures $\mu , \nu$ on a common measurable space as

$$
\| \mu - \nu \| _ { \mathrm { T V } } : = \operatorname* { s u p } _ { A } | \mu ( A ) - \nu ( A ) | = { \textstyle { \frac { 1 } { 2 } } } \int | \mathrm { d } \mu - \mathrm { d } \nu | \ \in [ 0 , 1 ] .
$$

For sub-σ-algebras ${ \mathcal { A } } , { \mathcal { B } } \subseteq { \mathcal { F } } ,$ , the β-mixing (absolute regularity) coefficient is

$$
\beta ( \boldsymbol { A } , \boldsymbol { B } ) : = \operatorname* { s u p } \ \frac { 1 } { 2 } \sum _ { i = 1 } ^ { I } \sum _ { j = 1 } ^ { J } \left| \mathbb { P } ( \boldsymbol { A } _ { i } \cap \boldsymbol { B } _ { j } ) - \mathbb { P } ( \boldsymbol { A } _ { i } ) \mathbb { P } ( \boldsymbol { B } _ { j } ) \right| ,
$$

the supremum over all finite A-measurable partitions $\{ A _ { i } \} _ { i \le I }$ and B-measurable partitions $\{ B _ { j } \} _ { j \le J } o f \Omega$ (Bradley, 2005). Since any pair of partitions admissible for $( A ^ { \prime } , B ^ { \prime } )$ is admissiblefor $( A , { \bar { B } } )$

$$
{ \mathcal { A } } ^ { \prime } \subseteq { \mathcal { A } } , { \mathcal { B } } ^ { \prime } \subseteq { \mathcal { B } } \quad \Longrightarrow \quad \beta ( { \mathcal { A } } ^ { \prime } , { \mathcal { B } } ^ { \prime } ) \leq \beta ( { \mathcal { A } } , { \mathcal { B } } ) .\tag{43}
$$

For a process $( Y _ { s } ) _ { s \geq s _ { 0 } ; }$ , its β-mixing coefficient at gap $t > 0$ is

$$
\beta _ { Y } ( t ) : = \operatorname* { s u p } _ { s \geq s _ { 0 } } \beta { \Big ( } \sigma ( Y _ { u } : s _ { 0 } \leq u \leq s ) , \sigma ( Y _ { u } : u \geq s + t ) { \Big ) } ,\tag{44}
$$

and for a sequence $( Y _ { k } ) _ { k \geq 0 }$ we set $\begin{array} { r } { \beta _ { Y } ( m ) : = \operatorname* { s u p } _ { k > 0 } \beta \big ( \sigma ( Y _ { j } : j \le k ) , \sigma ( Y _ { j } : j \ge k + m ) \big ) } \end{array}$ $m \geq 1$ . By Equation 43, β is nonincreasing. The process is β-mixing $i f \beta _ { Y } ( t ) \to 0 a s t \to \infty ,$ , and exponentially β-mixing $i f \beta _ { Y } ( t ) \le C _ { \beta } e ^ { - \sum _ { \mathrm { e r g } } t }$ for constants $C _ { \beta } < \infty , \lambda _ { \mathrm { e r g } } > 0 $ . For a stationary process indexed by R with past $\sigma ( Y _ { u } : u \le s )$ , the supremum is redundant and one recovers the classical coefficient $\beta ( \mathcal { F } _ { \le 0 } , \mathcal { F } _ { \ge t } ) ,$ ; the one-sidedform Equation 44 avoids constructing a two-sided extension ofthe solution.

Lemma 6 (β-coefficient of a stationary Markov process). Let $( \xi _ { s } ) _ { s \geq 0 }$ be a time-homogeneous Markov process on a Polish space $E ,$ Markov with respect to its natural filtration $\mathcal { G } _ { s } : = \sigma ( \xi _ { u } :$ $0 \leq u \leq s )$ , with transition kernels $( P _ { t } ) _ { t \geq 0 }$ and invariant probability measure π, and startedfrom $\xi _ { 0 } \sim \pi$ . Then, for every $s \geq 0$ and $t > 0$

$$
\beta \Big ( \mathcal { G } _ { s } , ~ \sigma ( \xi _ { u } : u \geq s + t ) \Big ) = \int _ { E } \| P _ { t } ( x , \cdot ) - \pi \| _ { \mathrm { T V } } \pi ( \mathrm { d } x ) ,\tag{45}
$$

so that, in particular, $\beta _ { \xi } ( t )$ is given by the right-hand side.

This identity is due to Davydov (1974) for Markov chains on a general state space, and its continuous-time version is standard (see, e.g., Masuda (2007)). Only the inequality $\ " \leq \ "$ is used below. It follows by conditioning each future event on $\mathcal { G } _ { s } \mathrm { : }$ by the Markov property and stationarity, the conditional law of $( \xi _ { s + t + r } ) _ { r \geq 0 }$ is the law of the process started from $\dot { P } _ { t } ( \dot { \xi } _ { s } , \cdot )$ , while its unconditional law is the law of the process started from π, and a Markov kernel does not increase total variation. For the reverse inequality, restrict both σ-algebras to the endpoints using Equation 43: $\beta ( \sigma ( \xi _ { s } ) , \sigma ( \xi _ { s + t } ) )$ is the total-variation distance between the joint law $\pi ( \mathrm { d } x ) P _ { t } ( x , \mathrm { d } y )$ of $\left( \xi _ { s } , \xi _ { s + t } \right)$ and the product law $\pi ( \mathrm { d } x ) \pi ( \mathrm { d } y )$ , which equals the right-hand side of Equation 45.

Theorem 5 (Exponential β-mixing of the lagged state). Let Assumptions 1, 2 and 6 hold, let $C \geq 1$ and $\gamma > 0$ be the constants ofTheorem 3, and take the process in its stationary regime, $X _ { 0 } ^ { \mathrm { s e g } } \sim \overline { { \pi } } ^ { \mathrm { s e g } }$ independent ofW. Set

$$
M _ { 2 } ^ { \pi } : = \int _ { \mathcal { C } } \lVert \eta - c \rVert _ { \mathcal { C } } ^ { 2 } \pi ^ { \mathrm { s e g } } ( \mathrm { d } \eta ) < \infty , \qquad C _ { \beta } : = C \bigl ( 1 + M _ { 2 } ^ { \pi } \bigr ) ,
$$

and write $Z _ { t } : = ( X _ { t } , X _ { t - \tau _ { 1 } } , \ldots , X _ { t - \tau _ { q } } ) , t \geq 0 .$ . Then:

(i) the segment process is exponentially β-mixing: $\beta _ { X ^ { \mathrm { s e g } } } ( t ) \leq C _ { \beta } e ^ { - \gamma t } f o r a l l t > 0 ,$

(ii) the lagged state is exponentially β-mixing: $\beta _ { Z } ( t ) \leq C _ { \beta } e ^ { - \gamma t }$ for all $t > 0$ , and the same bound holdsfor $( Z _ { t } ) _ { t \geq \tau _ { \operatorname* { m a x } } } ;$

(iii) for every deterministic grid $0 \leq t _ { 0 } < t _ { 1 } < \cdot \cdot \cdot$ , the sampled sequence $Z _ { k } : = Z _ { t _ { k } }$ satisfies, for all $k \geq 0$ and $m \geq 1$

$$
\beta { \Big ( } \sigma ( Z _ { j } : j \leq k ) , \ \sigma ( Z _ { j } : j \geq k + m ) { \Big ) } \leq C _ { \beta } e ^ { - \gamma ( t _ { k + m } - t _ { k } ) } ;\tag{46}
$$

in particular $\beta _ { ( Z _ { k } ) } ( m ) \le C _ { \beta } e ^ { - \gamma \Delta _ { \mathrm { m i n } } m }$ whenever min<sub>k</sub> $( t _ { k + 1 } - t _ { k } ) \geq \Delta _ { \operatorname* { m i n } } > 0 .$

Consequently, the physical-time mixing estimate Equation 50 used in Section A.8 is implied by Assumptions 1, 2 and 6, with $\lambda _ { \mathrm { e r g } } : = \gamma$ and $C _ { \beta }$ as above.

Proof. Step 1: Stationary regime and Markov property. By Theorem 3 with $p = 1 , M _ { 2 } ^ { \pi } < \infty$ , so the initial condition $X _ { 0 } ^ { \mathrm { s e g } } \sim \pi ^ { \mathrm { s e g } }$ satisfies the initial condition of Assumption 1, and Proposition 1 provides a unique strong solution; invariance of $\pi ^ { \mathrm { s e g } }$ gives $X _ { s } ^ { \mathrm { s e g } } \sim \pi ^ { \mathrm { s e g } }$ for every $s \geq 0$ . Finite stationary drift energy is not needed for mixing; it is imposed separately in Section A.8. As in the proof of Theorem $3 , ( X _ { t } ^ { \mathrm { s e g } } ) _ { t > 0 }$ is a time-homogeneous Markov process with kernels $( P _ { t } ) _ { t > 0 }$ with respect to the filtration generated by $X _ { 0 } ^ { \mathrm { s e g } }$ and $\breve { W }$ . Since its natural filtration $\mathcal { G } _ { s } : = \sigma \mathrm { ( } X _ { u } ^ { \mathrm { s e g } } : 0 \le$ $u \leq s )$ is smaller and $\bar { X ^ { \mathrm { s e g } } }$ is adapted to it, the tower property shows that $X ^ { \mathrm { s e g } }$ is also Markov with respect to $\left( \mathcal { G } _ { s } \right)$ . The space C is a separable Banach space, hence Polish, so Lemma 6 applies with $E \stackrel { - } { = } \mathcal { C }$ and $\pi = \pi ^ { \mathrm { s e g } }$

Step 2: Segment process, item (i). Fix $s \geq 0$ and $t > 0$ . By Lemma 6 and Equation 15,

$$
\begin{array} { r l r } {  { \beta \Big ( \mathcal { G } _ { s } , \ \sigma \big ( X _ { u } ^ { \mathrm { s e g } } : u \geq s + t \big ) \Big ) = \int _ { \mathcal { C } } \lVert P _ { t } ( \eta , \cdot ) - \pi ^ { \mathrm { s e g } } \rVert _ { \mathrm { T V } } \pi ^ { \mathrm { s e g } } ( \mathrm { d } \eta ) } } \\ & { } & { \qquad \le C e ^ { - \gamma t } \int _ { \mathcal { C } } \bigl ( 1 + \lVert \eta - c \rVert _ { \mathcal { C } } ^ { 2 } \bigr ) \pi ^ { \mathrm { s e g } } ( \mathrm { d } \eta ) = C _ { \beta } e ^ { - \gamma t } . } \end{array}\tag{47}
$$

Taking the supremum over $s \geq 0$ gives (i).

Step 3: Lagged state, item (ii). The evaluation map ev $: \mathcal { C } \ \to \ \mathbb { R } ^ { D ( q + 1 ) } , \ \mathrm { e v } ( \eta ) : =$ $( \eta ( 0 ) , \eta ( - \tau _ { 1 } ) , \dots , \eta ( - \tau _ { q } ) )$ , is continuous, and $Z _ { u } = \mathrm { e v } ( X _ { u } ^ { \mathrm { s e g } } )$ for every u $, \geq 0$ . Hence

$$
\sigma ( Z _ { u } : 0 \le u \le s ) \subseteq \mathcal { G } _ { s } , \qquad \sigma ( Z _ { u } : u \ge s + t ) \subseteq \sigma ( X _ { u } ^ { \mathrm { s e g } } : u \ge s + t ) ,
$$

and Equation 43 together with Equation 47 gives $\beta _ { Z } ( t ) \leq C _ { \beta } e ^ { - \gamma t }$ . Restricting the index set to $u \ge \tau _ { \operatorname* { m a x } }$ shrinks both σ-algebras and the range of the supremum in Equation 44, so the same bound holds for $( Z _ { t } ) _ { t \geq \tau _ { \operatorname* { m a x } } }$

Step 4: Deterministic subsamples, item (iii). Fix $k \geq 0 , m \geq 1$ and set $s : = t _ { k } , t : = t _ { k + m } - t _ { k } > 0$ Then $\sigma ( Z _ { j } : j \leq k ) \subseteq \mathcal { G } _ { s }$ and σ $\lceil \left( Z _ { j } : j \geq k + m \right) \subseteq \sigma ( X _ { u } ^ { \mathrm { s e g } } : u \geq s + t )$ , so Equation 46 follows from Equation 43 and Equation 47. No regularity of the spacings is used. If all spacings are at least $\Delta _ { \mathrm { m i n } }$ , then $t _ { k + m } - t _ { k } \geq m \Delta _ { \operatorname* { m i n } }$ , and taking the supremum over k gives the last claim. □

Remark 2 (Scope of Theorem 5). For $q = 0$ the segment space reduces to $\mathbb { R } ^ { D } \left( \tau _ { \operatorname* { m a x } } = 0 \right)$ and Theorem 5 covers the Markovian SDE case. The additive form of the noise is what makes the total-variation proof valid: if the diffusion coefficient depended on the delayed state, the quadratic variation of the solution could reveal the initial segment, the kernels $P _ { h } ( \varphi , \cdot )$ and $P _ { h } ( \psi , \cdot )$ could then be mutually singular (Hairer et al., 2011), and Lemma 5 (hence the bound Equation 47) could fail.

## A.8 FINITE-SAMPLE IDENTIFIABILITY

We now move from the population results to what can be guaranteed from a single trajectory observed on a finite grid. The argument compares the empirical excess risk to its population counterpart uniformly over the model class, and concludes with the margin $\gamma ( \lambda )$ of Theorem 2. We observe the stationary solution, with $X _ { 0 } ^ { \mathrm { { \tiny { s e g } } } } \sim \pi ^ { \mathrm { { \tiny { s e g } } } }$ independent of the future Brownian motion W. Write $\mathbb { P } _ { \pi }$ and $\mathbb { E } _ { \pi }$ for probability and expectation under this law, including both the initial segment and the subsequent noise. For $\bar { Z } _ { t } : = ( \bar { X _ { t } } , X _ { t - \tau _ { 1 } } , \ldots , X _ { t - \tau _ { q } } )$ , let $\mu _ { \infty } ^ { q }$ be its stationary distribution. Thus $\bar { \mu } _ { T } ^ { q } = \mu _ { \infty } ^ { q }$ , and the signal strengths in Equation 7 do not depend on $T$ in this section. The finite-energy condition in Assumption 1 is imposed on this stationary solution:

$$
m _ { G } ^ { 2 } : = \mathbb { E } _ { \pi } \| G ( Z _ { 0 } ) \| ^ { 2 } = \int \| G ( z ) \| ^ { 2 } \mu _ { \infty } ^ { q } ( d z ) < \infty .\tag{48}
$$

Observations are available on a deterministic grid $0 = t _ { 0 } < t _ { 1 } < \cdots < t _ { n } = T$ , with $\Delta _ { k } : =$ $t _ { k + 1 } - t _ { k }$ and $\Delta _ { \operatorname* { m a x } } : = \operatorname* { m a x } _ { k < n } \Delta _ { k } \leq 1$ . We assume that $\tau _ { \mathrm { m a x } }$ is an observation time and that $X _ { t _ { k } - \tau _ { a } }$ is observed exactly for every used left endpoint $t _ { k } \ge \tau _ { \operatorname* { m a x } }$ and every lag a. An equally spaced grid has this property when each delay is an integer multiple of its spacing. All sums over k below are over $k < n$ with $t _ { k } \ge \tau _ { \operatorname* { m a x } }$ , and we set

$$
Z _ { k } : = Z _ { t _ { k } } , \qquad T _ { \circ } : = \sum _ { k } \Delta _ { k } = T - \tau _ { \mathrm { m a x } } .
$$

Assumption 7 (Finite-sample model class). Let H<sup>q</sup> denote the restricted class of deterministic models derivedfrom Assumption 3(i) satisfying thefollowing two conditions:

(i) Local finite covering: For every radius $\rho \geq 1$ and tolerance $\epsilon > 0 ,$ , we can select a finite number ofrepresentativefunctionsfrom within H<sup>q</sup> (an internal ϵ-net) such that every function in the class is within a maximum distance of ϵ from at least one representative over the compact set $K _ { \rho } : = \{ z \in \mathbb { R } ^ { D ( q + 1 ) } : \| z - c ^ { ( q ) } \| \leq \rho \}$ . The minimal number of representatives required to cover the class this way is denoted by $N _ { n } ( \rho , \epsilon )$

(ii) Stationary envelope: The pointwise diameter of the class, defined as $\begin{array} { r l } { \mathcal { E } ( z ) } & { { } : = } \end{array}$ $\begin{array} { r } { \operatorname* { s u p } _ { \tilde { G } _ { 1 } , \tilde { G } _ { 2 } \in \mathcal { H } _ { n } ^ { q } } \| \tilde { G } _ { 1 } ( z ) - \tilde { G } _ { 2 } ( z ) | } \end{array}$ , has a finite second moment under the stationary measure: $\begin{array} { r } { m _ { \mathcal { E } } ^ { 2 } : = \int \mathcal { E } ( z ) ^ { 2 } \mu _ { \infty } ^ { q } ( d z ) < \infty . } \end{array}$

The local covering condition makes E locally bounded, and it is measurable as a supremum of continuous functions. Since the true drift is feasible, $\| \widetilde { G } ( z ) - G ( z ) \| \le \mathcal { E } ( z )$ for every candidate. Thus the finite second moment of the previous assumption controls prediction errors, not the absolute size of a dissipative drift. Any deterministic locally bounded measurable upper bound for these errors with finite stationary second moment can be used in its place. All class-dependent quantities may depend on n.

Define the stationary mean-square modulus of the true drift by

$$
\omega _ { G } ( h ) : = \operatorname* { s u p } _ { 0 \leq u \leq h } \left( \mathbb { E } _ { \pi } \| G ( Z _ { u } ) - G ( Z _ { 0 } ) \| ^ { 2 } \right) ^ { 1 / 2 } , \qquad 0 \leq h \leq 1 .\tag{49}
$$

Continuity and Equation 48 imply $\omega _ { G } ( h ) \to 0$ as $h \downarrow 0$ , as shown in the proof below. By Theorem $5 ,$ the total-variation conclusion of Theorem 3 and the stationary moment bound give the physical-time mixing estimate

$$
\beta _ { Z } ( u ) \leq C _ { \beta } e ^ { - \lambda _ { \mathrm { e r g } } u } , \qquad u > 0 ,\tag{50}
$$

where $\lambda _ { \mathrm { e r g } } = \gamma$ and $\begin{array} { r } { C _ { \beta } = C \left( 1 + \int \lVert \varphi - c \rVert _ { \mathcal { C } } ^ { 2 } \pi ^ { \mathrm { s e g } } ( d \varphi ) \right) } \end{array}$ may be used, with $C , \gamma$ from Theorem 3.

Consider the empirical criterion, and assume that a measurable global minimizer exists (a finite local cover does not by itself guarantee this):

$$
\widehat { \mathcal { R } } _ { n } ( \widetilde { G } ) : = \sum _ { k } \Delta _ { k } \big \Vert \frac { X _ { t _ { k + 1 } } - X _ { t _ { k } } } { \Delta _ { k } } - \widetilde { G } ( Z _ { k } ) \big \Vert ^ { 2 } ,
$$

$$
( \widehat { G } , \widehat { M } ) \in \mathop { \mathrm { \arg ~ m i n } } _ { ( \widetilde { G } , \widetilde { M } ) \in \mathcal { H } _ { n } ^ { q } } \left\{ \widehat { \mathcal { R } } _ { n } ( \widetilde { G } ) + \lambda \sum _ { a = 0 } ^ { q } \lVert \tilde { M } ^ { a } \rVert _ { 0 } \right\} .\tag{16}
$$

The relevant comparison is between the empirical excess risk ${ \widehat { \mathcal { R } } } _ { n } ( { \widetilde { G } } ) - { \widehat { \mathcal { R } } } _ { n } ( G )$ and

$$
\begin{array} { r } { \mathcal { R } _ { T } ( \widetilde { G } ) = T _ { \circ } \| \widetilde { G } - G \| _ { L ^ { 2 } ( \mu _ { \infty } ^ { q } ) } ^ { 2 } . } \end{array}
$$

Theorem 4 (Finite-sample exact support recovery, $L _ { 0 }$ estimator). Let Assumptions $I , 2 ,$ 6 and 7 holdfor the stationary observations above. Fix $\delta \in ( 0 , 1 )$ ) and choose the deterministic quantities in Equation 56–Equation 57 below. With probability at least $1 - \delta ,$

$$
\operatorname* { s u p } _ { \tilde { G } \in \mathcal { H } _ { n } ^ { q } } \Big | \big [ \widehat { \mathcal { R } } _ { n } ( \widetilde { G } ) - \widehat { \mathcal { R } } _ { n } ( G ) \big ] - \mathcal { R } _ { T } ( \widetilde { G } ) \Big | \leq \varepsilon _ { n } ( \delta ; \rho , \epsilon ) ,\tag{51}
$$

where the explicit bound is given in Equation 58. Suppose additionally that M $\ne 0 , s _ { \mathrm { m i n } } > 0 ;$ , and

$$
0 < \lambda < \frac { T _ { \circ } s _ { \operatorname* { m i n } } ^ { 2 } } { d _ { \operatorname* { m a x } } } , \qquad \gamma ( \lambda ) : = \operatorname* { m i n } \{ \lambda , T _ { \circ } s _ { \operatorname* { m i n } } ^ { 2 } - \lambda d _ { \operatorname* { m a x } } \} > 0 .\tag{52}
$$

If

$$
\varepsilon _ { n } ( \delta ; \rho , \epsilon ) < \gamma ( \lambda ) ,\tag{53}
$$

then every measurable minimizer in Equation 16 satisfies

$$
\mathbb { P } _ { \pi } \bigg ( \widehat { M } ^ { a } = M ^ { a } f o r a l l a = 0 , \ldots , q , \quad \| \widehat { G } - G \| _ { L ^ { 2 } ( \mu _ { \infty } ^ { q } ) } ^ { 2 } \leq \frac { \varepsilon _ { n } ( \delta ; \rho , \epsilon ) } { T _ { \circ } } \bigg ) \geq 1 - \delta .\tag{54}
$$

The conclusion applies to both $q \ = \ 0$ and $q \geq 1$ . It identifies the mask exactly, not the drift exactly from finitely many observations. With $\lambda ~ = ~ \lambda _ { 0 } T _ { \mathrm { c } }$ for $0 < \lambda _ { 0 } < s _ { \operatorname* { m i n } } ^ { 2 } / \dot { d } _ { \operatorname* { m a x } } .$ , recovery follows whenever the displayed bound divided by $T _ { \circ }$ is smaller than min $\{ \lambda _ { 0 } , s _ { \mathrm { m i n } } ^ { 2 } - \lambda _ { 0 } d _ { \mathrm { m a x } } \}$ Thus consistency requires a joint condition on the observation duration, mesh and class complexity; large n alone does not suffice.

For $\rho \geq 1$ , define

$$
E _ { \rho } : = 1 \vee \operatorname* { s u p } _ { z \in K _ { \rho } } \mathcal { E } ( z ) , \qquad v _ { \rho } : = \int _ { K _ { \rho } ^ { c } } \mathcal { E } ( z ) ^ { 2 } \mu _ { \infty } ^ { q } ( d z ) , \qquad \sigma _ { \mathrm { o p } } : = \| \Sigma \| _ { \mathrm { o p } } , \quad \sigma _ { F } : = \sqrt { \operatorname { t r } ( \Sigma \Sigma ^ { \top } ) } .\tag{55}
$$

Here $E _ { \rho }$ bounds errors on the truncation region and $v _ { \rho } \to 0$ is the remaining stationary squared-error mass. For an integer $p \geq 1$ , let

$$
S _ { p } : = \mathbb { E } _ { \pi } \operatorname* { s u p } _ { 0 \leq u \leq 1 } \| X _ { u } ^ { \mathrm { s e g } } - c \| _ { \mathcal { C } } ^ { 2 p } < \infty .
$$

Finiteness follows by integrating the finite-horizon estimate of Lemma $4 ( \mathrm { c } )$ against $\pi ^ { \mathrm { s e g } }$ and using its stationary moments.

For $\delta \in ( 0 , 1 )$ , choose $p \geq 1$ and $\epsilon > 0$ , and set

$$
\eta : = \delta / 6 , \qquad J : = \operatorname* { m a x } \{ 1 , \lceil T _ { \circ } \rceil \} , \qquad \rho \geq \operatorname* { m a x } \left\{ 1 , \left( \frac { J ( q + 1 ) ^ { p } S _ { p } } { \eta } \right) ^ { 1 / ( 2 p ) } \right\} .\tag{56}
$$

We define some more quantities: the block length and logarithmic complexity,

$$
b : = \operatorname* { m a x } \left\{ 1 , \frac { 1 } { \lambda _ { \mathrm { e r g } } } \log \frac { C _ { \beta } J } { \eta } \right\} , \qquad N : = N _ { n } ( \rho , \epsilon ) , \qquad \Lambda : = \log \frac { 4 N } { \eta } .\tag{57}
$$

A valid bound in Equation 51 is then

$$
\begin{array} { r l } & { \varepsilon _ { n } ( \delta ; \rho , \epsilon ) = \cfrac { 2 m _ { \mathscr { E } } T _ { \circ } } { \eta } \omega _ { G } ( \Delta _ { \operatorname* { m a x } } ) } \\ & { \qquad + 2 \sigma _ { \mathrm { o p } } E _ { \rho } \sqrt { 2 T _ { \circ } \Lambda } + \cfrac { 2 \epsilon \sigma _ { F } } { \eta } \displaystyle \sum _ { k } \sqrt { \Delta _ { k } } } \\ & { \qquad + E _ { \rho } ^ { 2 } \sqrt { 2 T _ { \circ } ( b + \Delta _ { \operatorname* { m a x } } ) \Lambda } + 4 E _ { \rho } \epsilon T _ { \circ } + v _ { \rho } T _ { \circ } . } \end{array}\tag{58}
$$

The first line is the discretisation error. The second line controls Brownian fluctuations and the approximation by a finite net. The third line controls the centred time average, its net approximation and the stationary tail outside $K _ { \rho }$

Before giving the proof, let us describe its structure. The excess empirical risk ${ \widehat { \mathcal { R } } } _ { n } ( { \widetilde { G } } ) - { \widehat { \mathcal { R } } } _ { n } ( G )$ splits into three parts: a time average of $\| { \widetilde { G } } - G \| ^ { 2 }$ along the grid, a stochastic integral against the Brownian increments, and a discretisation error due to the grid. The discretisation error is controlled by the modulus $\omega _ { G } ~ ( \mathrm { S t e p s } ~ 1 – 2 )$ . We then truncate to the ball $K _ { \rho }$ and replace the class by a finite net (Step 3). The time average concentrates around $\mathcal { R } _ { T }$ thanks to the exponential β-mixing of Theorem 5, via the independent-block argument of Yu (1994) and Hoeffding’s inequality (Step 4), while the Brownian term has Gaussian conditional increments and is controlled by a Chernoff bound (Step 5). A union bound over the net and the margin argument of Theorem 2 conclude (Step 6).

Proof of Theorem 4. Step 1: The drift modulus needs no growth assumption. Let $G _ { A }$ be the projection of G onto the closed Euclidean ball of radius $A$ in $\mathbb { R } ^ { D }$ . It is continuous and bounded, while $\begin{array} { r } { \mathsf { \bar { \| } } G - G _ { A } \| _ { L ^ { 2 } ( \mu _ { \infty } ^ { q } ) } \to 0 } \end{array}$ by Equation 48. Stationarity and the triangle inequality in $L ^ { 2 }$ give

$$
\begin{array} { r } { \| G ( Z _ { u } ) - G ( Z _ { 0 } ) \| _ { L ^ { 2 } ( \mathbb { P } _ { \pi } ) } \leq 2 \| G - G _ { A } \| _ { L ^ { 2 } ( \mu _ { \infty } ^ { q } ) } + \| G _ { A } ( Z _ { u } ) - G _ { A } ( Z _ { 0 } ) \| _ { L ^ { 2 } ( \mathbb { P } _ { \pi } ) } . } \end{array}
$$

For fixed A, continuity of the paths and bounded convergence make the last term tend to zero as u ↓ 0. First choosing A large and then u small proves $\omega _ { G } ( h )  0$ as $h \downarrow 0 \ddagger$ ; also $\omega _ { G } ( h ) \leq 2 m _ { G }$

Fix a deterministic internal net $G _ { 1 } , \ldots , G _ { N }$ on $K _ { \rho }$ and write $h = { \widetilde { G } } - G$ and $h _ { i } = G _ { i } - G$ The suprema below are measurable: the continuous drifts form a separable space in the topology of uniform convergence on compact sets, and their population risks are continuous on the class by domination with $\breve { \varepsilon ^ { 2 } }$ . One may therefore evaluate each supremum over a countable dense subset of the class.

Step 2: Exact expansion and discretisation. Set $\Delta W _ { k } : = W _ { t _ { k + 1 } } - W _ { t _ { k } }$ . Inserting the SDDE increment into the square and cancelling the candidate-independent squared-increment term gives

$$
\begin{array} { r l r } {  { \widehat { R } _ { n } ( \widetilde { G } ) - \widehat { R } _ { n } ( G ) - \mathcal { R } _ { T } ( \widetilde { G } ) = A _ { n } ( h ) - M _ { n } ( h ) - D _ { n } ( h ) , } } \\ & { } & { A _ { n } ( h ) : = \displaystyle \sum _ { k } \Delta _ { k } ( \| h ( Z _ { k } ) \| ^ { 2 } - \int \| h ( z ) \| ^ { 2 } \mu _ { \infty } ^ { a } ( d z ) ) , } \\ & { } & { M _ { n } ( h ) : = 2 \displaystyle \sum _ { k } \langle h ( Z _ { k } ) , \Sigma \Delta W _ { k } \rangle , } \\ & { } & { D _ { n } ( h ) : = 2 \displaystyle \sum _ { k } \int _ { t _ { k } } ^ { t _ { k + 1 } } \langle h ( Z _ { k } ) , G ( Z _ { s } ) - G ( Z _ { k } ) \rangle d s . } \end{array}\tag{59}
$$

Stationarity makes the centring in $A _ { n }$ exact; there is no further Riemann-sum error for its expectation. For every candidate,

$$
| D _ { n } ( h ) | \leq \mathcal { D } _ { n } : = 2 \sum _ { k } \int _ { t _ { k } } ^ { t _ { k + 1 } } \mathcal { E } ( Z _ { k } ) \| G ( Z _ { s } ) - G ( Z _ { k } ) \| d s .
$$

Cauchy–Schwarz and stationarity imply

$$
\mathbb { E } _ { \pi } \mathcal { D } _ { n } \leq 2 m \varepsilon \sum _ { k } \int _ { 0 } ^ { \Delta _ { k } } \omega _ { G } ( u ) d u \leq 2 m \varepsilon T _ { \circ } \omega _ { G } ( \Delta _ { \operatorname* { m a x } } ) .
$$

Markov’s inequality therefore bounds $\operatorname* { s u p } _ { h } | D _ { n } ( h )$ | by the first line of Equation 58, outside an event of probability at most $\eta .$ If this expectation bound is zero, $\mathcal { D } _ { n } = 0$ almost surely and no exceptional event is needed.

Step 3: Localisation and the squared-loss net. Let

$$
\mathcal { L } _ { \rho } : = \left\{ \operatorname* { s u p } _ { \tau _ { \operatorname* { m a x } } \leq t \leq T } \| Z _ { t } - c ^ { ( q ) } \| \leq \rho \right\} .
$$

Since $\begin{array} { r } { \| Z _ { t } - c ^ { ( q ) } \| \leq \sqrt { q + 1 } \| X _ { t } ^ { \mathrm { s e g } } - c \| c _ { 1 } } \end{array}$ , a cover of $[ \tau _ { \operatorname* { m a x } } , T ]$ by J intervals of length at most one, stationarity and Markov’s inequality give

$$
\mathbb { P } _ { \pi } ( \mathcal { L } _ { \rho } ^ { c } ) \leq \frac { J ( q + 1 ) ^ { p } S _ { p } } { \rho ^ { 2 p } } \leq \eta .\tag{60}
$$

Define $f _ { h } ( z ) : = \| h ( z ) \| ^ { 2 } \mathbf { 1 } _ { K _ { \rho } } ( z ) \in [ 0 , E _ { \rho } ^ { 2 } ]$ and

$$
A _ { n } ^ { \rho } ( h ) : = \sum _ { k } \Delta _ { k } \left( f _ { h } ( Z _ { k } ) - \int f _ { h } ( z ) \mu _ { \infty } ^ { q } ( d z ) \right) .
$$

On $\mathcal { L } _ { \rho } ,$ the empirical squared losses equal their truncated versions, so $| A _ { n } ( h ) | \leq | A _ { n } ^ { \rho } ( h ) | + T _ { \circ } v _ { \rho }$ For each $h ,$ choose a centre $h _ { i }$ with sup $\smash { | \boldsymbol { K } _ { s } | \| h - h _ { i } \| \le \epsilon }$ . Both errors are bounded by $E _ { \rho }$ on this set, hence sup<sub>z</sub> $| f _ { h } ( z ) - f _ { h _ { i } } ( z ) | \le 2 E _ { \rho } \epsilon$ and

$$
\operatorname* { s u p } _ { h } | A _ { n } ( h ) | \leq \operatorname* { m a x } _ { i \leq N } | A _ { n } ^ { \rho } ( h _ { i } ) | + 4 E _ { \rho } \epsilon T _ { \circ } + T _ { \circ } v _ { \rho } \quad \mathrm { o n } \ \mathcal { L } _ { \rho } .\tag{61}
$$

Step 4: Time-average fluctuations on an irregular grid. Partition the used left endpoints into bins $I _ { j } = [ \tau _ { \operatorname* { m a x } } + j b , \tau _ { \operatorname* { m a x } } + ( j + 1 ) b )$ , for $0 \leq j < J _ { b } : = \operatorname* { m a x } \{ 1 , \lceil T _ { \circ } / b \rceil \}$ , and set $\begin{array} { r } { \boldsymbol { w } _ { j } : = \sum _ { \boldsymbol { k } : t _ { \boldsymbol { k } } \in I _ { j } } \Delta _ { \boldsymbol { k } } } \end{array}$ The intervals with left endpoints in a nonempty bin are consecutive, and only the last can extend beyond its right boundary, by at most $\Delta _ { \mathrm { m a x } }$ . Consequently,

$$
0 \leq w _ { j } \leq b + \Delta _ { \operatorname* { m a x } } , \qquad \sum _ { j } w _ { j } = T _ { \circ } , \qquad \sum _ { j } w _ { j } ^ { 2 } \leq ( b + \Delta _ { \operatorname* { m a x } } ) T _ { \circ } .\tag{62}
$$

Successive nonempty bins of the same parity have their observations separated by at least b in physical time. For either parity, Equation 50 bounds the total-variation distance between its joint block law and the product of its block marginals by $( m - 1 ) _ { + } C _ { \beta } e ^ { - \lambda _ { \mathrm { e r g } } b }$ , where m is the number of its nonempty bins. Indeed, successively separating the next block from all previous blocks costs at most the corresponding $\beta$ coefficient, and the triangle inequality adds these costs. This comparison concerns the observation blocks themselves, so its error is not multiplied by the number N of candidate functions. It is the independent-block argument of Yu (1994), here applied in physical time.

Under the product law for one parity, the variables

$$
V _ { j , i } : = \sum _ { k : t _ { k } \in I _ { j } } \Delta _ { k } \left( f _ { h _ { i } } ( Z _ { k } ) - \int f _ { h _ { i } } ( z ) \mu _ { \infty } ^ { q } ( d z ) \right)
$$

are independent and centred, with range length at most $E _ { \rho } ^ { 2 } w _ { j }$ . Hoeffding’s inequality gives, for $x > 0$

$$
\mathbb { P } _ { \mathrm { p r o d } } \left( \left| \sum _ { j \mathrm { ~ o f ~ o n e ~ p a r i t y } } V _ { j , i } \right| > x \right) \leq 2 \exp \left( - \frac { 2 x ^ { 2 } } { E _ { \rho } ^ { 4 } \sum _ { j } w _ { j } ^ { 2 } } \right) .
$$

An empty parity has sum zero and needs no bound. Using Equation 62, taking $x \quad =$ $E _ { \rho } ^ { 2 } \sqrt { T _ { \circ } ( b + \Delta _ { \mathrm { m a x } } ) \Lambda / 2 }$ , and taking the union over the N centres and two parities yield

$$
\mathbb { P } _ { \pi } \bigg ( \operatorname* { m a x } _ { i \leq N } | A _ { n } ^ { \rho } ( h _ { i } ) | > E _ { \rho } ^ { 2 } \sqrt { 2 T _ { \circ } ( b + \Delta _ { \operatorname* { m a x } } ) \Lambda } \bigg ) \leq 4 N e ^ { - \Lambda } + J _ { b } C _ { \beta } e ^ { - \lambda _ { \mathrm { e r g } } b } \leq 2 \eta .\tag{63}
$$

The last inequality uses $b \geq 1$ , hence $J _ { b } \leq J ,$ and Equation 57. Together with Equation $^ { 6 1 }$ , this supplies the third line of Equation 58.

Step 5: Brownianfluctuations and their net approximation. Use the predictable truncation

$$
M _ { n } ^ { \rho } ( h ) : = 2 \sum _ { k } \langle h ( Z _ { k } ) \mathbf { 1 } _ { K _ { \rho } } ( Z _ { k } ) , \Sigma \Delta W _ { k } \rangle .
$$

For fixed $h _ { i } ,$ , conditional on the past at $t _ { k }$ , the kth summand is centred Gaussian with variance at most $4 \sigma _ { \mathrm { o p } } ^ { 2 } \dot { E } _ { \rho } ^ { 2 } \Delta _ { k }$ . Iterating the conditional moment-generating functions gives

$$
\mathbb { E } _ { \pi } e ^ { s M _ { n } ^ { \rho } \left( h _ { i } \right) } \le \exp \left( 2 s ^ { 2 } \sigma _ { \mathrm { o p } } ^ { 2 } E _ { \rho } ^ { 2 } T _ { \circ } \right) , \qquad s \in \mathbb { R } .
$$

The Gaussian Chernoff bound and a union over the fixed net imply

$$
\mathbb { P } _ { \pi } \bigg ( \operatorname* { m a x } _ { i \leq N } | M _ { n } ^ { \rho } ( h _ { i } ) | > 2 \sigma _ { \mathrm { o p } } E _ { \rho } \sqrt { 2 T _ { \circ } \Lambda } \bigg ) \leq 2 N e ^ { - \Lambda } = \eta / 2 .\tag{64}
$$

The net remainder is controlled pathwise, not by treating a data-selected remainder as a fixed martingale:

$$
| M _ { n } ^ { \rho } ( h ) - M _ { n } ^ { \rho } ( h _ { i } ) | \leq 2 \epsilon \sum _ { k } \lVert \Sigma \Delta W _ { k } \rVert , \qquad \mathbb { E } _ { \boldsymbol { \pi } } \sum _ { k } \lVert \Sigma \Delta W _ { k } \rVert \leq \sigma _ { F } \sum _ { k } \sqrt { \Delta _ { k } } .
$$

Markov’s inequality therefore bounds the remainder simultaneously for all candidates by 2ϵ $\begin{array} { r } { \sigma _ { F } \eta ^ { - 1 } \sum _ { k } \dot { \sqrt { \Delta _ { k } } } , } \end{array}$ outside an event of probability at most η. On $\mathcal { L } _ { \rho } , \mathbf { \dot { \mathcal { M } } } _ { n } ( h ) = M _ { n } ^ { \rho } ( h )$ for every h, so this and Equation 64 give the second line of Equation 58.

Step 6: Uniform deviation and mask recovery. The exceptional probabilities in Steps 2–5 sum to at most

$$
\begin{array} { r } { \eta + \eta + 2 \eta + \eta / 2 + \eta = \frac { 1 1 } { 2 } \eta < \delta . } \end{array}
$$

Adding the three bounds in Equation 59 proves Equation 51 on the complementary event. On that event, feasibility of $( G , M )$ and empirical optimality imply

$$
\mathcal { R } _ { T } ( \widehat { G } ) + \lambda \sum _ { a = 0 } ^ { q } \lVert \widehat { M } ^ { a } \rVert _ { 0 } - \lambda \sum _ { a = 0 } ^ { q } \lVert M ^ { a } \rVert _ { 0 } \leq \varepsilon _ { n } ( \delta ; \rho , \epsilon ) .
$$

The population margin of Theorem 2, applied to $\mu _ { \infty } ^ { q } \mathrm { ~ = ~ } \bar { \mu } _ { T } ^ { q }$ , makes the left-hand side at least $\gamma ( \lambda )$ for every wrong mask. This uses its lower bound for every feasible wrong mask, not a population-minimisation assumption on the empirical estimator. Condition Equation 53 therefore excludes every wrong mask on the same event. Once the masks agree, their penalties cancel, leaving $\mathcal { R } _ { T } ( \widehat { G } ) \leq \varepsilon _ { n } ( \delta ; \rho , \epsilon )$ . Dividing by $T _ { \circ }$ proves Equation 54. □

We showed in the proof that the discretisation term is $2 m \varepsilon T _ { \circ } \omega _ { G } ( \Delta _ { \mathrm { m a x } } ) / \eta$ . When the stronger estimate $\omega _ { G } ( h ) \leq C _ { G } \sqrt { h }$ is available, Step 2 improves this term to 4m $\begin{array} { r } { \iota _ { \mathcal { E } } C _ { G } ( 3 \eta ) ^ { - 1 } \sum _ { k } \Delta _ { k } ^ { 3 / 2 } } \end{array}$ . It does not become $O ( T _ { \circ } \Delta _ { \operatorname* { m a x } } )$ ) merely from a fourth drift moment. On an equally spaced used grid, we have the equality $\begin{array} { r } { \sum _ { k } \sqrt { \Delta _ { k } } = T _ { \circ } / \sqrt { \Delta } } \end{array}$ , so the martingale net remainder divided by $T _ { \circ }$ is proportional to $\epsilon / ( \eta \sqrt { \Delta } )$ . On a general grid, the displayed sum must be retained; it is not bounded using $\Delta _ { \mathrm { m a x } }$ alone. The factors $\bar { 1 / \eta } = 6 \bar { / } \delta$ together with the local envelope $E _ { \rho } .$ , which may grow with the radius $\rho \ge ( J ( q + 1 ) ^ { p } S _ { p } / \eta ) ^ { 1 / ( 2 p ) }$ and hence with $T _ { \circ }$ , make Equation 58 a qualitative consistency statement rather than a sharp rate.

Optimization accuracy and sampling endpoints. If the returned feasible pair has a certified empirical objective gap at most $\zeta _ { n } \geq 0$ , replace Equation 53 by $\varepsilon _ { n } + \zeta _ { n } < \gamma ( \lambda )$ and the prediction bound by $( \varepsilon _ { n } + \zeta _ { n } ) \bar { / } \bar { T } _ { \circ }$ . This follows by adding $\zeta _ { n }$ to the final optimality comparison; an arbitrary output of a nonconvex optimization algorithm is not automatically certified. $\operatorname { I f } \tau _ { \operatorname* { m a x } }$ is not observed, let a be the first used left endpoint, replace $T _ { \circ }$ by $T - a$ , and use $\mathcal { R } _ { a , T } ( \widetilde { G } ) : = ( T - a ) \Vert \widetilde { G } - G \Vert _ { L ^ { 2 } ( \mu _ { \infty } ^ { q } ) } ^ { 2 }$ throughout this section and in its population margin. The proof is otherwise unchanged. Interpolated lag values require an additional error analysis and are not covered by the exact-observation statement above.

## A.9 DISSIPATIVITY AND ERGODICITY OF THE LEARNED SYSTEM

In this section, we analyze the ergodicity of both the data-generating process and the learned simulator, providing theoretical insights beyond the empirical assumptions made in our experiments. Ergodicity of the true data-generating process guarantees that a single, long trajectory visits all states in proportion to their true probabilities. This allows us to learn the invariant measure because time averages converge to ensemble averages. Conversely, ergodicity of the learned model ensures sta bility during forward simulation: rather than drifting indefinitely or exploding, the generated paths settle into their own stationary distribution (which ideally matches the true one). In what follows, we discuss sufficient conditions and architectural choices to guarantee the ergodicity of the learned system.

We parameterize the drift using an MLP with tanh hidden activations and an affine output layer. Provided there are no skip connections to the final layer, the output of this architecture is globally bounded. Consequently, when used as the drift in Equation 1, the resulting process is non-explosive. However, a globally bounded drift cannot satisfy the dissipation condition of Assumption 6 outside a compact set, nor can it uniformly approximate a drift that does over the entire domain. As a result, Theorem 6 does not apply to an unconstrained setup. We emphasize that this highlights a limitation of the sufficient condition rather than the model itself, as bounded drifts can still be ergodic. For instance, the SDE $d X _ { t } = - \operatorname { t a n h } ( X _ { t } ) d t + \sigma d W _ { t }$ yields an integrable invariant density proportional to $( \cosh x ) ^ { - 2 / \sigma ^ { 2 } }$ , and its drift can be exactly represented by our unconstrained chosen architecture.

In what follows, we describe several fixes and results to maintain dissipativity and thus ergodicity of the learned drift. We chose for our experiments to apply the post-training exterior confinement defined by Equation 67 which imposes no restriction to the architecture during the training and only applies a restoring force outside a compact set that encompasses all of our samples.

Although an unconstrained MLP architecture does not inherently satisfy these global dissipation conditions, we nevertheless provide a conditional ergodicity result in Theorem 6. This global perturbation result becomes highly relevant when the learned and true drifts exhibit compatible tail behaviors — for instance, when they share a dissipative physical baseline and differ only by bounded residuals. Thus, we retain this theorem to address scenarios equipped with such structural priors, with the caveat that its hypotheses do not automatically emerge simply from fitting an unconstrained neural network.

Theorem 6 (Conditional ergodicity transfer under bounded drift error). Suppose Assumptions 1, 2 and 6 hold for the true system, with $\begin{array} { r } { B = \sum _ { a = 1 } ^ { q } ( \beta _ { a } + \varepsilon _ { a } ) < \beta _ { 0 } } \end{array}$ . Let $\widetilde { G }$ be locally Lipschitz, of at most linear growth, and satisfy

$$
\operatorname* { s u p } _ { z \in \mathbb { R } ^ { D ( q + 1 ) } } \lVert \widetilde { G } ( z ) - G ( z ) \rVert \leq \delta _ { 0 } < \infty .
$$

For any $0 < \eta < 2 ( \beta _ { 0 } - B )$ , it satisfies Assumption 6 with the same centre and radius and with

$$
\begin{array} { l l } { { \widetilde { \beta } _ { 0 } = \beta _ { 0 } - \eta / 2 , ~ } } & { { ~ \widetilde { \alpha } = \alpha + \delta _ { 0 } ^ { 2 } / ( 2 \eta ) , ~ } } & { { ~ \widetilde { \beta } _ { a } = \beta _ { a } , } } \\ { { \widetilde { C } _ { R } = C _ { R } + \delta _ { 0 } ^ { 2 } / ( 2 \eta ) + \eta R ^ { 2 } / 2 , ~ } } & { { ~ \widetilde { \varepsilon } _ { a } = \varepsilon _ { a } ~ ( a \geq 1 ) . } } \end{array}
$$

With the same invertible additive diffusion, the learned SDDE is non-explosive and its segment process has a unique invariant probability measure, all polynomial stationary segment moments, and geometric convergence in total variation as in Theorem ${ \dot { 3 } } ,$ with model-dependent constants.

Proof. Writing $u = x _ { 0 } - c ,$ , Cauchy–Schwarz and Young’s inequality give

$$
\langle u , \widetilde { G } ( z ) - G ( z ) \rangle \leq \delta _ { 0 } \| u \| \leq \frac { \eta } { 2 } \| u \| ^ { 2 } + \frac { \delta _ { 0 } ^ { 2 } } { 2 \eta } .
$$

Add this to the exterior and interior inequalities of Assumption 6; in the latter, use $\| u \| ^ { 2 } \leq R ^ { 2 }$ . The displayed constants follow and retain strict domination. The resulting radial bound also gives the Khasminskii condition. Local Lipschitzness gives well-posedness by localisation, and the secondmoment estimate together with at-most-linear growth gives finite drift energy for square-integrable initial histories independent of the future noise. The moment and kernel-overlap arguments establishing Theorem 3 then apply to $\widetilde { G } .$ □

An important observation is that the constant $\delta _ { 0 }$ need not be small for existence of the ergodic behavior, although increasing it can have drawbacks on statistical accuracy, and worsen moments and mixing results. Note that the result above can be strengthened into an affine error bound with sufficiently small slope, $\| \widetilde G ( z ) - G ( z ) \| \le \delta _ { 0 } + \theta \| z - c ^ { ( q ) } \| , \theta \ge 0$

For the learned process to remain stable, we only need to bound the network’s error in the outward radial direction. Large errors that act tangentially, or those that provide additional inward pull, will not cause the system to diverge. We therefore turn to conditions that can be imposed directly on the learned drift rather than on its unknown error at infinity.

Dissipative architecture with bounded neural residual A simple and natural extension of the unconstrained architecture is to retain the MLP with tanh activations as nonlinear residuals and add an explicit instantaneous damping:

$$
\widetilde { G } _ { \theta } ( z ) = - K ( x _ { 0 } - c ) + N _ { \theta } ( z ) , \quad K = \mathrm { d i a g } ( \kappa _ { 1 } , . . . , \kappa _ { D } ) , \ \kappa _ { i } \geq \kappa _ { * } > 0\tag{65}
$$

Proposition 2 (Confinement by construction). Let $N _ { \theta }$ be locally Lipschitz and globally bounded, with sup $_ z \| N _ { \theta } ( z ) \| \le M _ { \theta }$ . For any fixed delays and an invertible additive diffusion as in Assumption $^ { l , }$ the drift in Equation 65 defines a non-explosive SDDE whose segment process has a unique invariant law, all polynomial stationary segment moments, and geometric convergence in total vari ation.

Proof. For every lagged state $z ,$ with $u = x _ { 0 } - c ,$

$$
\langle u , \widetilde { G } _ { \theta } ( z ) \rangle \leq - \kappa _ { * } \| u \| ^ { 2 } + M _ { \theta } \| u \| \leq - \frac { \kappa _ { * } } { 2 } \| u \| ^ { 2 } + \frac { M _ { \theta } ^ { 2 } } { 2 \kappa _ { * } } .\tag{66}
$$

This gives Assumption 6 with zero delayed coefficients; the interior radial bound follows by increasing its constant on any chosen ball. Local Lipschitzness and linear growth ensure the remaining well-posedness and energy conditions. The conclusion is a direct application of Theorem 3. □

Post-training exterior confinement The same idea can be applied after training in such a way that the fitted drift remains unchanged inside a radius vector $\bar { R } = ( R _ { 1 } , \ldots , R _ { D } ) ^ { \top }$ with strictly positive components, $R _ { i } > 0$ . Set $u = x _ { 0 } - c$ and define $\Pi _ { R } ( u ) = \langle \operatorname* { m i n } \{ R _ { j } , \operatorname* { m a x } \{ - R _ { j } , u _ { j } \} \} \rangle _ { j } ^ { }$ the projection onto the box $\textstyle \prod _ { j } [ - R _ { j } , R _ { j } ]$ . Then, define the post-hoc network:

$$
\widetilde { G } _ { \theta , R } ( z ) = N _ { \theta } ( z ) - K \cdot ( u - \Pi _ { R } ( u ) )\tag{67}
$$

Whenever $| x _ { 0 , j } - c _ { j } | \le R _ { j }$ for every $j ,$ , the network is left unchanged for all delayed inputs. This applies a linear restoring drift only to coordinates outside the given box. Furthermore, since we can rearrange Equation 67 as $\widetilde { G } _ { \theta , R } ( z ) = - K \cdot u + ( N _ { \theta } ( z ) + K \cdot \Pi _ { R } ( u ) )$ ) and $\begin{array} { r } { \mathrm { s u p } _ { z } \| N _ { \theta } ( z ) + K \Pi _ { R } ( u ) \| \le } \end{array}$ $M _ { \theta } + \Vert \bar { K } \cdot \mathbf { \bar { \phi } } _ { R } \Vert$ , Proposition 2 applies. Note, nonetheless, that this changes in general the invariant law of the learned process and thus the damping matrix and radii considered should be taken into consideration and their effect on long-run statistics assessed.

The preceding results describe some constructions to exhibit the existence of the learned equilibrium, however they say little about whether or not it agrees with the true one. Under the same invertible additive noise, we can find a stationary error bound on the invariant distributions in total variation given the true invariant measure.

Proposition 3 (Stationary error bound on invariant distributions). Suppose the true process is stationary with segment law $\pi ^ { \mathrm { s e g } }$ . Let $\widetilde { G }$ be locally Lipschitz and define a non-explosive SDDE with the same lags and diffusion. Suppose its invariant law $\widetilde { \pi } ^ { \mathrm { s e g } }$ and kernels obey

$$
\| \widetilde { P } _ { t } ( \varphi , \cdot ) - \widetilde { \pi } ^ { \mathrm { s e g } } \| _ { \mathrm { T V } } \leq \widetilde { C } e ^ { - \widetilde { \gamma } t } ( 1 + V ( \varphi ) ) , \quad V ( \varphi ) = \| \varphi - c \| _ { C } ^ { 2 } , \quad \pi ^ { \mathrm { s e g } } V < \infty .\tag{68}
$$

where the former inequality holds when the learned drift satisfies for example Assumptions 1, 2 and $6$ as in Theorem 3. $\begin{array} { r } { I f e _ { \Sigma } ^ { 2 } = \int \| \Sigma ^ { - 1 } ( \widetilde { G } - G ) ( z ) \| ^ { 2 } \mu _ { \infty } ^ { q } ( d z ) < \infty , } \end{array}$ , then for every $t > 0$ , with total variation normalized as sup $_ { A } \mathinner { | { \nu ( A ) - \mu ( A ) } }$ |,

$$
\| \pi ^ { \mathrm { s e g } } - \widetilde { \pi } ^ { \mathrm { s e g } } \| _ { \mathrm { T V } } \leq \frac { e _ { \Sigma } \sqrt { t } } { 2 } + \widetilde { C } e ^ { - \widetilde { \gamma } t } ( 1 + \pi ^ { \mathrm { s e g } } V ) .\tag{69}
$$

Proof. Step 1: Common starting equilibrium law. Fix $t > 0$ and start both equations with an initial segment of law $\pi ,$ , independent of future noise. The true terminal-segment law remains π, whereas the learned law is $\pi \widetilde { P } _ { t }$ . Hence

$$
\lVert \pi ^ { \mathrm { s e g } } - \widetilde { \pi } ^ { \mathrm { s e g } } \rVert _ { \mathrm { T V } } \leq \underbrace { \lVert \pi ^ { \mathrm { s e g } } - \pi ^ { \mathrm { s e g } } \widetilde { P } _ { t } \rVert _ { \mathrm { T V } } } _ { \mathrm { f n i t e - t i m e ~ d r i f t ~ c o m p a r i s o n } } + \underbrace { \lVert \pi ^ { \mathrm { s e g } } \widetilde { P } _ { t } - \widetilde { \pi } ^ { \mathrm { s e g } } \rVert _ { \mathrm { T V } } } _ { \mathrm { l e a r n e d ~ m i x i n g } } .\tag{70}
$$

Let $\mathbb { P }$ and $\mathbb { Q }$ be the true and learned laws of the full path on $[ - \tau _ { \operatorname* { m a x } } , t ] ,$ , including its initial history. On this path space set

$$
v ( z ) = \Sigma ^ { - 1 } ( G ( z ) - \widetilde { G } ( z ) ) , \qquad A _ { s } = \int _ { 0 } ^ { s } \lVert v ( Z _ { r } ) \rVert ^ { 2 } d r .
$$

Stationarity under P gives the essential identity

$$
\mathbb { E } _ { \mathbb { P } } A _ { t } = \int _ { 0 } ^ { t } \mathbb { E } _ { \mathbb { P } } \| v ( Z _ { s } ) \| ^ { 2 } d s = t e _ { \Sigma } ^ { 2 } .\tag{71}
$$

Step 2: Stopped change of drift. Finite mean energy does not imply a global Novikov condition. Instead, stop at $\sigma _ { m } = \operatorname* { i n f } \{ s \in [ 0 , t ] : A _ { s } \geq m \}$ , with the convention inf $\mathcal { D } = \infty$ . Under $\mathbb { Q } ,$ , let $\widetilde { B }$ be the driving Brownian motion and define

$$
\frac { d \mathbb { P } _ { m } } { d \mathbb { Q } } = \exp \left\{ \int _ { 0 } ^ { t \wedge \sigma _ { m } } v ( Z _ { s } ) \cdot d \widetilde { B } _ { s } - \frac { 1 } { 2 } A _ { t \wedge \sigma _ { m } } \right\} .\tag{72}
$$

Because $A _ { t \wedge \sigma _ { m } } \leq m$ , this exponential is a true martingale, also conditionally on the initial history. Girsanov’s theorem therefore preserves the initial law π and makes $\mathbb { P } _ { m }$ follow drift G until $\sigma _ { m }$ , then drift $\widetilde { G }$ afterwards.

Under $\mathbb { P } _ { m } .$ , the log-density in Equation 72 is a mean-zero stochastic integral plus $A _ { t \wedge \sigma _ { m } } / 2$ . Consequently, the Kullback-Leibler divergence can be written as

$$
D _ { \mathrm { K L } } ( \mathbb { P } _ { m } \| \mathbb { Q } ) = \frac 1 2 \mathbb { E } _ { \mathbb { P } _ { m } } A _ { t \wedge \sigma _ { m } } = \frac 1 2 \mathbb { E } _ { \mathbb { P } } A _ { t \wedge \sigma _ { m } } \leq \frac { t e _ { \Sigma } ^ { 2 } } { 2 } .\tag{73}
$$

The second equality holds because the auxiliary and true equations agree up to the stopping time: pathwise uniqueness from Proposition 1 gives identical stopped path laws. This is why the error remains evaluated under the true law.

If we couple these two equations with the same history and Brownian motion, their paths can differ only if $\sigma _ { m } \leq t .$ , so

$$
\| \mathbb { P } - \mathbb { P } _ { m } \| _ { \mathrm { T V } } \leq \mathbb { P } ( A _ { t } \geq m ) \leq \frac { t e _ { \Sigma } ^ { 2 } } { m } .
$$

Pinsker’s inequality and Equation 73 now imply

$$
\| \mathbb { P } - \mathbb { Q } \| _ { \mathrm { T V } } \le \frac { t e _ { \Sigma } ^ { 2 } } { m } + \sqrt { \frac { 1 } { 2 } D _ { \mathrm { K L } } ( \mathbb { P } _ { m } \| \mathbb { Q } ) } \le \frac { t e _ { \Sigma } ^ { 2 } } { m } + \frac { e _ { \Sigma } \sqrt { t } } { 2 } .
$$

Let $m  \infty$ with t fixed. Extracting a terminal segment cannot increase total variation, and its laws under $\mathbb { P } , \mathbb { Q }$ are $\pi ^ { \mathrm { s e g } } , \pi ^ { \mathrm { s e g } } \widetilde { P } _ { t }$ , respectively. Thus

$$
\| \pi ^ { \mathrm { s e g } } - \pi ^ { \mathrm { s e g } } \widetilde { P } _ { t } \| _ { \mathrm { T V } } \leq \frac { e _ { \Sigma } \sqrt { t } } { 2 } .\tag{74}
$$

Step 3: Conclusion. Integrating Equation 68 against its initial law $\pi ^ { \mathrm { s e g } }$ gives

$$
\| \pi ^ { \mathrm { s e g } } \widetilde { P } _ { t } - \widetilde { \pi } ^ { \mathrm { s e g } } \| _ { \mathrm { T V } } \leq \int \| \widetilde { P } _ { t } ( \varphi , \cdot ) - \widetilde { \pi } ^ { \mathrm { s e g } } \| _ { \mathrm { T V } } \pi ^ { \mathrm { s e g } } ( d \varphi ) \leq \widetilde { C } e ^ { - \widetilde { \gamma } _ { t } } ( 1 + \pi ^ { \mathrm { s e g } } V ) .
$$

Together with Equation 74 and Equation 70, this proves Equation 69.

## B COMPARISON WITH ADAPTIVE GROUP LASSO ASSUMPTIONS

Bellot et al. (2022) prove local consistency of the adaptive group lasso, for Markovian systems. They show similar identifiability proofs using the adaptive group lasso to select input features. AGL also requires an initial consistent estimator to build the adaptive weights. In this work, we derive identifiability guarantees for SDDE-driven systems, which extend Bellot et al. (2022) results. Additionally, we use the $L _ { 0 }$ penalty instead of AGL. The $L _ { 0 }$ route is in two ways cleaner: no initial estimator is needed, and the population theorem (2) already supplies an explicit λ window. Both results suppose an optimization oracle, which is standard in the literature, but something that is most often not available in practice.

Our theorems assume global minimization of a nonconvex, combinatorial objective. Bellot et al. (2022) assume that they can find exact solutions to their problem, which is least squares over a neural network class with an AGL penalty, and hence also nonconvex. Both results presuppose an optimization oracle — theirs for a continuous nonconvex problem, ours for a combinatorial one —. Empirically, we find that even on Markovian systems (part of the CausalDynamics benchmark, the Dysts benchmark), the relaxed $L _ { 0 }$ penalty performs better than the AGL-penalized objective. Our minimum-signal condition $s _ { \operatorname* { m i n } } > 0$ plays exactly the role of their eq. (10), and our effective sample size $\lambda _ { \mathrm { e r g } } T$ plays the role of their $n / { \bar { | | \alpha | | _ { 2 } } }$ .

## C IMPLEMENTATION DETAILS AND HYPERPARAMETERS

## C.1 IMPLEMENTATION DETAILS

The $L _ { 0 }$ -norm is not differentiable. To allow for score-based optimization, we implement the relaxed $L _ { 0 }$ penalization proposed by Louizos et al. (2018). More precisely, the matrix M is a square logits matrix parametrized as the sigmoid of differentiable parameters in R. During training, at each batch, we draw samples of M through a hard-concrete distribution, a continuous relaxation of a binary random variable. We use a low temperature parameter $\beta = 0 . 3 3$ to sharpen the hard-concrete distribution and force to turn inputs on or off. Samples are drawn on the $[ - 0 . 1 , 1 . 1 ]$ interval before being clamped to [0, 1] i.e. most samples will be exactly equal to 0 or 1. At inference, we fix M to be 0 or 1 if the probabilities (obtained from the logits) are lower or higher than 0.5. To have a smoothed convergence, we draw multiple samples of M at each batch before optimizing for its differentiable parameters. We also warm-up the model and train the neural network without penalty for 100 warm-up iterations before adding the penalty to the loss. We then train until the loss plateaus and does not decrease for 200 patience iterations. These parameters and optimization procedure are standard when using $L _ { 0 }$ -penalty (Louizos et al., 2018).

The functions $f$ are small multi-layer perceptrons (MLPs) and we use a gradient penalty to enforce that our functions are Lipschitz. To make sure that smooth dimensions $\mathrm { e . g }$ . with close to 0 gradient are not approximated by the mean and then do not have any drivers, we renormalize each dimension by the gradient mean and standard deviation.

C-NODE follows a similar implementation, where the $L _ { 0 }$ is replaced by $L _ { 1 }$ norm. In practice, we just apply a $L _ { 1 }$ penalty on the logits matrix M. AGL implementation is similar, but uses an AGL penalty instead. We find that using a gradient penalty slightly improve the results for both, so we included it in the method’s implementation. Except the penalty, everything is kept the same between methods.

## D CAUSALDYNAMICS BENCHMARK

## D.1 METRIC COMPUTATION

$L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ and “No Grad. Pen.” use a logit matrix to represent probabilities of inputting a feature or not. It is thus straightforward to calculate SHD, AUROC and AUPRC by comparing this matrix to the ground truth logits matrix. AGL and $L _ { 1 }$ penalty instead assign weights to each input feature. While these weights can still be compensated for by the neural network, we used this weight matrix to compute AUROC / AUPRC, treating the weights as feature importance. For SHD instead, we used a low threshold $( 1 0 ^ { - 8 } )$ to decide whether an input feature is turned off or not. Changing this threshold did not affect performance, as we observed that the weights were generally either 0 or above $1 0 ^ { - 3 }$ in practice.

## D.2 HYPERPARAMETER SEARCH

The causal methods evaluated in CausalDynamics have various numbers of hyperparameters. While methods such as PCMCI rely crucially on a single parameter e.g. the conditional independence test threshold, others such as TSCI or NGC depend on many parameters $\mathrm { e . g . }$ . neural network architectures. Herdeanu et al. (2025) thus decided to keep default parameters except the main parameter and perform hyperparameter search on this parameter only. We do the same and keep default parameters, highlighted in Section D.3, except for the sparsity penalty coefficient, which controls for the final level of sparsity.

For all methods, we run our model on the 10 time series of the Lorenz84 system with the following penalty coefficients and select the one maximizing AUROC: (0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100).

## D.3 FINAL HYPERPARAMETERS

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Architecture, functions  $f _ { j }$ </td><td></td></tr><tr><td>Number of layers</td><td>2</td></tr><tr><td>Number of units per layer</td><td>8</td></tr><tr><td>Optimization parameters</td><td></td></tr><tr><td>Gradient penalty coefficient</td><td>100</td></tr><tr><td>Warm-up iterations Patience iterations</td><td>100</td></tr><tr><td>Learning rate</td><td>200</td></tr><tr><td>Batch size</td><td>0.001</td></tr><tr><td></td><td>128</td></tr><tr><td> $L _ { 0 }$  implementation parameters</td><td> $( L _ { \mathrm { 0 } } – \mathrm { N D D E }$  , No grad. pen.)</td></tr><tr><td>Temperature  $\beta _ { \mathrm { t e m p } }$  Approximate distribution interval</td><td>0.33</td></tr><tr><td>Number of samples</td><td>[−0.1,1.1]</td></tr><tr><td></td><td>3</td></tr><tr><td>Sparsity penalty coefficients</td><td></td></tr><tr><td> $L _ { 0 }$ </td><td>0.025</td></tr><tr><td> $\mathrm { \bf A G L }$ </td><td>5</td></tr><tr><td>C-NODE (L1)</td><td>50</td></tr><tr><td>No grad. pen.</td><td>0.01</td></tr></table>

Table 3: Final hyperparameters used for ${ \cal L } _ { 0 } { \bf - N D B E } .$ , C-NODE, AGL and No grad. pen. The last 4 lines show the penalty coefficient for each method, as it is the only parameter tuned for each the method. Note that No grad. pen. does not use gradient penalty.

## D.4 FULL CAUSALDYNAMICS RESULTS

<table><tr><td rowspan="2"></td><td rowspan="2">Experiment</td><td colspan="3">Simple</td><td colspan="5">Coupled</td><td colspan="2">Climate</td><td rowspan="2">Avg. rank</td></tr><tr><td>Default</td><td>Confounder</td><td>Noise</td><td>Default</td><td>Noise</td><td>Confounder</td><td>Time-lag</td><td>Standardize</td><td>MAOOAM</td><td>ENSO</td></tr><tr><td rowspan="12"></td><td>PCMCI+</td><td>41.04</td><td>23.02</td><td>47.64</td><td>224.80</td><td>183.90</td><td>324.63</td><td>327.72</td><td>228.32</td><td>80.00</td><td>529.36</td><td>7.30</td></tr><tr><td>F-PCMCI</td><td>35.30</td><td>21.07</td><td>45.09</td><td>192.90</td><td>149.60</td><td>195.74</td><td>350.61</td><td>201.79</td><td>130.00</td><td>530.27</td><td>5.80</td></tr><tr><td>VARLiNGAM</td><td>35.69</td><td>22.04</td><td>42.84</td><td>311.45</td><td>248.90</td><td>159.63</td><td>449.33</td><td>349.63</td><td>130.00</td><td>453.00</td><td>8.00</td></tr><tr><td>DYNOTEARS</td><td>52.37</td><td>21.74</td><td>51.68</td><td>181.50</td><td>180.95</td><td>248.32</td><td>261.44</td><td>243.84</td><td>94.00</td><td>589.36</td><td>6.60</td></tr><tr><td>NGC</td><td>28.91</td><td>19.96</td><td>28.84</td><td>840.95</td><td>842.55</td><td>670.53</td><td>793.67</td><td>840.26</td><td>31.00</td><td>337.09</td><td>7.90</td></tr><tr><td>TSCI</td><td>52.50</td><td>22.02</td><td>60.50</td><td>174.70</td><td>173.50</td><td>265.26</td><td>244.56</td><td>244.42</td><td>108.00</td><td>666.27</td><td>7.20</td></tr><tr><td>CUTS+</td><td>48.11</td><td>24.04</td><td>61.32</td><td>152.00</td><td>150.50</td><td>272.68</td><td>247.22</td><td>310.63</td><td>130.00</td><td>608.73</td><td>7.50</td></tr><tr><td>RCD</td><td>61.85</td><td>26.74</td><td>61.38</td><td>157.05</td><td>155.65</td><td>136.53</td><td>201.11</td><td>159.84</td><td>130.00</td><td>665.36</td><td>6.80</td></tr><tr><td>GRaSP</td><td>59.04</td><td>27.09</td><td>60.18</td><td>215.94</td><td>842.55</td><td>136.00</td><td>793.67</td><td>840.26</td><td>126.00</td><td>666.27</td><td>9.70</td></tr><tr><td>TCDF</td><td>59.46</td><td>26.61</td><td>61.43</td><td>252.00</td><td>842.55</td><td>670.53</td><td>793.67</td><td>840.26</td><td>130.00</td><td>666.27</td><td>11.65</td></tr><tr><td>AGL</td><td>26.60</td><td>16.96</td><td>21.28</td><td>319.04*</td><td>222.41*</td><td>246.37*</td><td>315.54*</td><td>312.31*</td><td>30.00*</td><td>337.09</td><td>5.55</td></tr><tr><td>C-NODE (L1)</td><td>24.07 26.42</td><td>17.10 16.25</td><td>20.5 23.67</td><td>253.91* 205.35</td><td>176.30* 144.20</td><td>179.58* 143.33</td><td>273.57*</td><td>258.09</td><td>42.00</td><td>335.36</td><td>4.50</td></tr><tr><td>L0-NDDE (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>222.34</td><td>212.11</td><td>40.00</td><td>325.54</td><td>2.50</td></tr><tr><td rowspan="10">AURRC</td><td>PCMCI+</td><td>.52 / .71</td><td>.49 / .59</td><td>.50 / .69</td><td>.67 / .25</td><td>64 /.25</td><td>.58 / .20</td><td>.58 / .24</td><td>.69 / .27</td><td>.69 / .88</td><td>.57 /.70</td><td>4.10 / 5.25</td></tr><tr><td>F-PCMCI</td><td>.51 / .70</td><td>.50 / .59</td><td>.52 / .70</td><td>.67 / .27</td><td>.57 / .21</td><td>.55 / .19</td><td>59 /.24</td><td>.68 / .28</td><td>.50 /.81</td><td>57 /.70</td><td>4.80 / 6.05</td></tr><tr><td>VARLiNGAM</td><td>.50 / .69</td><td>.48 / .58</td><td>.53 / .70</td><td>.60 / .19</td><td>.57 / .18</td><td>.51 / .17</td><td>.54 / .22</td><td>.60 /.19</td><td>.50 / .81</td><td>.56 / .69</td><td>6.90 / 8.55</td></tr><tr><td>DYNOTEARS</td><td>.43 / .67</td><td>.52 / .64</td><td>.48 / .68</td><td>.59 / .21</td><td>.57 / .20</td><td>.49 /.17</td><td>.53 / .22</td><td>.65 / .23</td><td>.64 / .86</td><td>.55 / .69</td><td>7.75 / 7.95</td></tr><tr><td>NGC</td><td>.50 / .69</td><td>.50/.58</td><td>.50 / .68</td><td>.50/ .15</td><td>.50/.15</td><td>.50/.16</td><td>.50 / .20</td><td>.50/.15</td><td>.50 / .81</td><td>.50/ .67</td><td>9.85 / 10.90</td></tr><tr><td>TSCI</td><td>.46 / .68</td><td>.53 / .66</td><td>.49 / .68</td><td>.60 / .23</td><td>.53 / .17</td><td>.51 / .18</td><td>.53 /.21</td><td>.65 / .23</td><td>.58 / .84</td><td>.50 / .67</td><td>7.50 / 8.00</td></tr><tr><td>CUTS+</td><td>.50 / .69</td><td>.50 / .58</td><td>.50 / .68</td><td>.50 / .15</td><td>.50 /.15</td><td>.49 /.16</td><td>.50 / .20</td><td>.50 /.15</td><td>.50 / .81</td><td>.50 / .67</td><td>10.10 / 10.90</td></tr><tr><td>RCD</td><td>.50 /.69</td><td>.50/.58</td><td>.50/.68</td><td>.50 / .15</td><td>.50/.15</td><td>.51 /.18</td><td>.50 /.20</td><td>.50 /.16</td><td>.50 /.81</td><td>.50/.67</td><td>9.55 / 10.25</td></tr><tr><td>GRaSP</td><td>.52 / .71</td><td>.55 / .64</td><td>.49 / .68</td><td>.49 / .15</td><td>.50 / .50</td><td>.50 /.17</td><td>.50 / .50</td><td>.50 / .50</td><td>.48 / .81</td><td>.50 / .50</td><td>9.60 / 6.70</td></tr><tr><td>TCDF</td><td>.51 / .70</td><td>.50 /.58</td><td>.50/ .68</td><td>.48 / .15</td><td>.50 / .50</td><td>.50/ .50</td><td>.50 / .50</td><td>.50/.50</td><td>.50/ .81</td><td>.50/ .50</td><td>9.85 / 6.65</td></tr><tr><td></td><td>AGL</td><td>.52 /.71*</td><td>.56 /.64</td><td>.53 /.74*</td><td>.53 / .33*</td><td>.53 / .33*</td><td>.54 /.35*</td><td>.52 / .35*</td><td>.54 / .34*</td><td>.56 / .86*</td><td>.54 / .69*</td><td>5.80 / 4.35</td></tr><tr><td></td><td></td><td>.64 / .79</td><td>.54 /.65</td><td>.61 / .79*</td><td>.64 /.43</td><td>.63 /.43</td><td>.67 /.47</td><td>.59/.43</td><td>.68 /.47</td><td>.88 / .97</td><td>.49 / .66*</td><td>3.45 / 3.35</td></tr><tr><td>C-NODE (L1)</td><td>L0-NDDE (Ours)</td><td>.64 / .80</td><td>.58 / .66</td><td>.63 / .80</td><td>.64 / .43</td><td>.65 / .44</td><td>.67 /.47</td><td>.60 / .43</td><td>.66 /.45</td><td>.86 / .96</td><td>.58 / .75</td><td>1.75 / 2.10</td></tr></table>

Table 4: $L _ { \mathrm { 0 } } { \bf - N D B E }$ outperforms competing methods most of the time on the CausalDynamics benchmark. This table is similar to Table 1 with additional methods being shown.

## D.5 NO GRADIENT PENALTY

<table><tr><td rowspan="2"></td><td rowspan="2">Experiment</td><td colspan="3">Simple</td><td colspan="6">Coupled</td><td colspan="2">Climate</td></tr><tr><td>Default</td><td>Confounder</td><td>Noise</td><td>Default</td><td>Noise</td><td>Confounder</td><td>Time-lag</td><td>Standardize</td><td>MAOOAM</td><td></td><td>ENSO</td></tr><tr><td rowspan="2">SHD</td><td>No Grad. Pen.</td><td>26.95</td><td>18.21</td><td>24.47</td><td>215.08</td><td>142.67</td><td>145.08</td><td>233.58</td><td></td><td>220.68</td><td>49.00</td><td>341.27</td></tr><tr><td>L0-NDDE (Ours)</td><td>26.42</td><td>16.25</td><td>23.67</td><td>205.35</td><td></td><td>144.20</td><td>143.33</td><td>222.34</td><td>212.11</td><td>40.00</td><td>325.54</td></tr><tr><td>AUROC /</td><td>No Grad. Pen.</td><td>.64 / .80</td><td>.58 / .65</td><td>.61 /.79</td><td></td><td>.63 / .42</td><td>.64 / .44</td><td>.67 /.47</td><td>.59 / .41</td><td>.65 / .44</td><td>.85 / .96</td><td>.52 / .68</td></tr><tr><td>AUPRC</td><td>L0-NDDE (Ours)</td><td>.64 / .80</td><td>.58 / .66</td><td></td><td>.63 / .80</td><td>.64 /.43</td><td>.65 / .44</td><td>.67/.47</td><td>.60 / .43</td><td>.66 / .45</td><td>.86 / .96</td><td>.58 /.75</td></tr></table>

Table 5: $L _ { \mathrm { 0 } } – \mathrm { N D D E }$ is slightly stronger with gradient penalty. Gradient penalty helps avoid overfitting spurious correlations and leads to slightly better performance on the CausalDynamics benchmark.

## E DYSTS BENCHMARK

## E.1 TRAIN AND TEST SYSTEMS

We use 40 systems in total, split into a hyperparameter search set (5 systems), and a test set (the other 35 systems). For each system, we generate one training trajectory, and 3 test trajectories. The train trajectories last 50 Lyapunov times and the test trajectories last 250 Lyapunov times, to ensure that the trajectory visits the entire space. Trajectories are sampled at the native temporal resolution. To avoid too short or too long trajectories in training, we enforce the training trajectory to have a minimum of 1000 and a maximum of 10000 timesteps. We use a stochastic forcing of level 0.05 times the variance of each dimension of the training trajectory. Each test and training trajectory is normalized by subtracting the mean and dividing by the training trajectory standard deviation per dimension.

We studied 40 systems in total, which are the 40 Dysts systems for which the Jacobian is accessible. For these 40 systems, a method is implemented to access the Jacobian at any point along any trajectory. This is necessary to compute metrics. Among these 40 systems, we randomly chose 5 systems for doing hyperparameter search, and 35 systems for testing. On the 5 hyperparameter search systems, we ran the methods with the following sets of hyperparameters, and selected the ones that maximized the valid prediction time on the three test trajectories.

For all methods, we wait a certain number of patience iterations (200) to declare convergence i.e. when the penalized loss does not improve for 200 iterations, we stop sparsifying. After this, we fix the input mask i.e. the learned matrices $\tilde { M } ,$ , and then keep training the neural networks to learn the dynamics for a certain number of iterations n\_iter\_after\_convergence. Also, we wait a number niter\_warmup of warm-up iterations before sparsifying.

We chose the best values among the following values for the following parameters:

• Sparsity penalty coefficient: [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

• Gradient penalty coefficient: [0, 10, 100, 1000, 10<sub>0</sub>00]

• n\_iter\_after\_convergence: [0, 50, 100, 200, 500, 1000]

• niter\_warmup: [0, 25, 50, 100]

For all methods, these four parameters are key as they control the sparsity and the regularization of the methods and their final values can be found in Table 6. All other parameters are kept the same as in the CausalDynamics experiments (Table 3).
<table><tr><td>Optimization parameters</td><td> $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ </td><td>NODE</td><td>C-NODE</td><td>AGL</td><td>PathReg</td></tr><tr><td>Gradient penalty coefficient</td><td>0</td><td>100</td><td>0</td><td>1000</td><td>1000</td></tr><tr><td>niter_warmup</td><td>0</td><td>0</td><td>100</td><td>50</td><td>200</td></tr><tr><td>n_iter_after_convergence</td><td>200</td><td>0</td><td>500</td><td>1000</td><td>500</td></tr><tr><td>Sparsity penalty coefficients</td><td>0.05</td><td>0</td><td>0.05</td><td>50</td><td>1</td></tr></table>

Table 6: Final hyperparameters used for $\begin{array} { r } { L _ { 0 } { \bf - N D D E } , } \end{array}$ C-NODE, AGL and PathReg

The systems chosen randomly to perform hyperparameter search are the following: MooreSpiegel, Lorenz, Laser, RikitakeDynamo, Chua

The systems used for evaluating the methods are the following: BurkeShaw, Chen, ChenLee, Coullet, DequanLi, Duffing, Finance, GenesioTesi, Hadley, JerkCircuit, KawczynskiStrizhak, LiuChen, Lorenz84, LuChen, LuChenCheng, PanXuZhou, PehlivanWei, QiChen, RayleighBenard, Rucklidge, Sakarya, SanUmSrisuchinwong, ShimizuMorioka, SprottA, SprottB, SprottC, SprottE, SprottI, SprottTorus, Thomas, ThomasLabyrinth, Tsucs2, WangSun, YuWang, YuWang2, ZhouChen

## E.2 METRICS

As we evaluate driver identifiability on the CausalDynamics benchmark, we are now interested in evaluating the dynamics. More precisely, we want to compare methods on short-term horizon prediction performance, attractor and geometry reconstruction, and long-term statistics. At evaluation, we predict autoregressively 3 trajectories from the initial conditions of the 3 test conditions (only the initial conditions are given), and generate trajectories of the same length as the test trajectories. We then compare the predicted vs true trajectories and compute several metrics.

For short-term prediction, we compute the valid prediction time at 1, normalized by the first Lyapunov exponent (valid prediction lyapunov time, VLPT) i.e. we compute the number of Lyapunov timesteps before the normalized trajectory is further than 1 away from the true normalized trajectory. We also compute the normalized root mean square error (NRMSE), by averaging the RMSE across the first 5 Lyapunov timesteps before normalizing by the trajectory mean and standard deviation.

For geometry and attractor reconstruction, we compute the first Lyapunov exponent error (FLEE) i.e. the squared difference between the first Lyapunov exponent of the learned vs. true system. The Lyapunov exponents are computed using the Benettin algorithm (Skokos, 2010), which requires access to the Jacobian which is possible in the selected Dysts systems, and for the learned models which are by construction differentiable. This error is small when the chaoticity of the system is well approximated. We additionally compute the Kaplan-Yorke dimension error $( D _ { K Y }$ , Evans et al., 2000). Computed using the Lyapunov exponents, the $D _ { K Y }$ gives us the dimension of the attractor i.e. the fractal dimension, which would be 2 in the case of the classic Lorenz system. The reported error is the squared error between the estimated system vs. true system’s $D _ { K Y }$

We then evaluate whether the learned system correctly captures long-term statistics, in other words whether the predicted trajectories converge, and if so whether they converge to the right equilibrium distribution. We report the log-spectral distance (LSD), which is the squared difference between the true and estimated log power spectral density, to indicate whether the system captured the right oscillation and periodicity properties of the system. We additionally report the $D _ { s t s p } .$ , a state-space divergence which estimates the KL divergence between the learned and true system invariant distribution from a set of predicted and true trajectories (Koppe et al., 2019). We compute it by placing a Gaussian kernel at each trajectory point (for both true and generated trajectories) and estimating the KL divergence between the two Gaussian Mixture Models via Monte Carlo sampling.

## E.3 RESULTS

On top of the quantitative results shown in the paper, we show, in Figure 2 the predicted (red) vs true (black) trajectories for 26 systems, for the three best methods, $L _ { 0 } – \bar { \mathrm { N D D E } }$ , NODE and C-NODE.

We then analyze the performance of the different methods on this benchmark, in Figure 3.

L0-NDDE

NODE

C-NODE

L0-NDDE

NODE

C-NODE

![](images/eafe675efac1e54178258ac0e164e7b33dd8bedc637427c165fca4bea1bd8adf.jpg)

![](images/d48aeb209dfe82fe2860673ee407c5981df873b9cf559a1c65a0ec3cd602525f.jpg)

Figure 2: Example predicted trajectories (red) vs. true trajectories (black) on 26 Dysts systems, for L<sub>0</sub>-NDDE (left), NODE (middle), and C-NODE (right).

![](images/c9075573d627d3b0ca963f1583cfc727a363b08a74bee3aa4f2dd1d4a2680077.jpg)

![](images/a39b78fdab62c556b086b0e88feaae560b76c625b33908b33026c18e37b550bb.jpg)

![](images/5b08eb89b66e8dc35f200c7afae922b270f1008afadd8f9a2b6292be30106552.jpg)

![](images/2c053df1180e562570a895cf78cd72a633bc72e70c0c7d78b69911f6ce8304f3.jpg)

![](images/3df68deb0d124f5198d2fecd6a1425b518a12bd92ada807d71179e82e3a6bb70.jpg)

![](images/52372a876344044c51b647a11e4a6f82e1bfee1c026f8ece42d5e5a81fc6773e.jpg)

![](images/8cc2665ddcc8a3893d1134e4340dd75fad0f7e39f60187d0359532dbe3009349.jpg)

![](images/c165abbcb3a27dfbe73ad109b2528d55ccbf5e1f09a20a7f115856e96dda55a0.jpg)

![](images/caa91367ff4d92a34af88cca65accb92e485854d7748134db1df258ef8244f7b.jpg)

![](images/177c6dfdf9d7e427471403e3ee295d6d47cb5259631f758f214dc58f4bff95d3.jpg)

![](images/300e9ecdf06a70f798b425df6ef88bd8294302ced6df73e521605943789fc1ef.jpg)

![](images/6b305e0a96618e508d8b7ccd0000b1e9a5fe97a144facf99bd5512934a3210b8.jpg)  
Figure 3: Lower training error and lower SHD are both associated with higher performance. We perform an analysis to better understand why $L _ { \mathrm { 0 } ^ { - } } \mathrm { N D D E }$ and NODE outperform other methods on the Dysts datasets. We compute, for all methods and all datasets, the training error (Train MSE) and SHD. We then plot the 6 metrics versus the SHD (upper half) and versus the training error (lower half). We see that $D _ { K Y }$ and $D _ { s t s p }$ increase, while VLPT decreases with SHD and training error. These relationships are statistically significant according to a Kendall-τ test, a standard rank-correlation test for investigating associations between two sets of values, without assuming Gaussianity of the samples or needing large samples. The relationship is also significant between LSD and training error. This indicates that a lower training error is associated with better recon struction, indicating why NODE performs well, as the absence of input feature regularization allows it to overfit to the training data; and why $L _ { 0 } – \mathbf { N D D E }$ outperforms other methods, as it achieves lower SHD. Moreover, the $L _ { 0 }$ -penalty has the advantage that it does not perform weight decay, but explicitly turns features on or off, allowing the rest of the network to approximate the dynamics very well given the set of selected features.