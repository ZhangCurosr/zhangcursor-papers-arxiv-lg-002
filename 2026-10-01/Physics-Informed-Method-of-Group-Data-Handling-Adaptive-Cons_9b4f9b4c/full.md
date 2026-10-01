# Physics-Informed Method of Group Data Handling: Adaptive Construction of Functional Representations with an Application to the Navier-Stokes Equations

Mykhailo Minin

JSC KIEP, Kyiv, Ukraine

Corresponding author: mykhailo.minin@gmail.com

## Abstract

Physics-informed computational methods usually optimize parameters within a functional representation whose structure is fixed in advance. This work proposes a Physics-Informed Method of Group Data Handling (PI-GMDH), in which representations of coupled physical fields are progressively constructed during solution. Candidate functional directions are evaluated through the first variation of the complete physical and observational objective, introduced in packages, and followed by block-coordinate damped Gauss-Newton coefficient optimization. The framework is demonstrated with tensor-product Chebyshev functions on the incompressible Navier-Stokes equations using a two-dimensional time-dependent Taylor-Green benchmark. Under the tested configuration, adaptive PI-GMDH reached validation and held-out test losses of 5.299e-19 and 5.296e-19 with 204, 201, and 175 active functions for u, v, and p. Complete degree-by-degree and all-terms PI-GMDH variants, together with selected PINN and KAN reference configurations, are used to examine the effect of structural construction policy. The results show that, for this controlled synthetic benchmark, selective progressive construction can provide a favorable combination of accuracy, representation size, and wall-clock time. The comparison is illustrative rather than a claim of universal superiority over alternative physics-informed approaches.

## 1. Introduction

Physics-informed computational methods incorporate known physical relations directly into the construction or optimization of an approximation. Physics-informed neural networks (PINNs), for example, represent unknown fields by neural networks and include residuals of governing differential equations together with available observations, boundary conditions, and initial conditions in the optimization objective [3,4]. Related physics-informed learning methods have broadened this principle beyond a single neural architecture.

A common feature of many such approaches is that the general form of the approximation is chosen before optimization begins. Training then determines parameters within that representation. This raises a separate structural question: rather than only asking which parameter values minimize the physical objective, can the governing equations also determine which functional components should constitute the approximation?

The present work addresses this question through a Physics-Informed Method of Group Data Handling (PI-GMDH). Classical GMDH introduced self-organizing structural construction in which candidate models are generated, parametrically fitted, evaluated, and selectively retained as model complexity develops [1,2]. PI-GMDH adopts this self-organizing principle at the level of functional representations of physical fields: the governing physical equations participate directly in deciding which functional components should become active.

## 1.1 Relation to Existing Approaches

GMDH and differential models. GMDH and its descendants have previously been used for polynomial selforganization and for data-driven differential modeling. In particular, differential polynomial neural networks extend GMDH-type structures and have been used to construct or approximate partial-differential relations from observed data [12,13]. The present work therefore does not claim that differential equations have not previously appeared in

GMDH-derived methods. The distinction is that the governing PDE is assumed known and its complete residual is used to construct the functional representation of the unknown solution fields.

Residual-driven basis enrichment. Adaptive finite-element, multiscale, and reduced-order methods also use residual or error information to enrich approximation spaces. Residual-based online enrichment in generalized multiscale finite-element methods, for example, constructs additional basis functions from the current PDE residual and problem data [14]. PI-GMDH shares the objective of adapting the approximation space, but its elementary candidates are evaluated directly as functional perturbation directions through the first variation of the complete coupled physicsinformed objective, rather than being generated from local residual subproblems or solution snapshots.

## Greedy approximation in function space. Stagewise additive modeling and gradient boosting interpret

approximation as optimization in function space and progressively introduce components associated with descent directions [15]. This viewpoint is mathematically close to the present structural-selection mechanism. In PI-GMDH, however, a candidate perturbation is propagated through the governing differential operators and all coupled residual components before its relevance is evaluated. Candidate selection is therefore based on its first-order influence on the complete physical and observational objective.

Physics-informed spectral representations. Chebyshev and other orthogonal-polynomial representations are increasingly used in physics-informed neural, KAN, and operator-learning formulations [9,16-18], while adaptive spectral methods can modify active spectral content during solution [19]. Consequently, neither Chebyshev approximation nor physics-informed spectral representation is claimed as the novelty of the present work. Chebyshev functions are used here as a differentiable candidate family through which the proposed self-organizing structural mechanism can be studied.

The central PI-GMDH cycle consists of two coupled operations. First, the functional representation is structurally extended by introducing a package of new functional components. Second, the coefficients of the enlarged representation are reoptimized against the complete physical and observational objective. The package-construction rule is therefore an explicit component of the method rather than an incidental implementation detail. Different rules produce different PI-GMDH variants while retaining the same underlying physical formulation and parameteroptimization machinery.

A package may contain every function in the next level of a predefined functional hierarchy, a single function, the entire remaining candidate space, or a selectively chosen subset. The present work develops a physics-guided adaptive package rule in which candidate functions are assessed through the first variation of the complete objective. The governing equations are consequently used not only to optimize coefficients after functions have been activated, but also to decide which functional directions are useful for subsequent structural growth.

The proposed framework is not restricted to the Navier-Stokes equations. It requires a differentiable residual describing the physical problem and candidate functional directions for which the required operator responses can be evaluated. The incompressible Navier-Stokes equations are selected here because they provide a demanding coupled nonlinear test containing multiple fields, nonlinear convection, incompressibility, pressure-velocity coupling, and several derivative orders. A two-dimensional time-dependent Taylor-Green flow is used as a controlled benchmark with an analytical solution.

The principal contributions of this work are:

Formulation of PI-GMDH as a general framework in which physics-informed coefficient optimization is coupled to progressive construction of the functional representation.

 Explicit formulation of structural growth through functional packages, allowing alternative package-construction policies to be investigated within a common PI-GMDH framework.

Development of a physics-informed structural-selection mechanism in which candidate functional components are evaluated through the first variation of the complete coupled physical and observational objective, and selected components are introduced as packages before block-coordinate coefficient reoptimization.

Integration of structural construction with block-coordinate coefficient optimization, allowing coupled physical fields to develop different active functional representations while remaining linked through the same physical objective.

 A controlled Navier-Stokes benchmark that compares PI-GMDH package-construction strategies using a common Chebyshev family and optimization mechanism, with selected PINN and KAN configurations included as external physics-informed references.

The individual ingredients of this formulation have important precedents in GMDH, residual-driven basis enrichment, greedy function-space approximation, and physics-informed spectral methods. The contribution claimed here is their particular integration: self-organizing package-wise construction of coupled physical-field representations, with candidate functional directions evaluated through the first variation of the complete physics-informed objective. The literature review conducted for this study did not identify a prior formulation combining these elements in this form; this statement is therefore made as a literature-based positioning claim rather than an absolute priority claim. The Taylor-Green study reported here is intended to demonstrate this construction mechanism under controlled condition rather than to provide a comprehensive comparative evaluation across differential systems, data regimes, or selection criteria.

The numerical study therefore addresses a more specific question than whether one functional family is preferable to another: given a common candidate family and a common physics-informed objective, how does the policy used to construct packages of functional components affect the evolution, computational cost, and attainable accuracy of the representation?

## 2. Physics-Informed Method of Group Data Handling

## 2.1 Physical problem, functional representation, and packages

Consider independent variables $\bar { \xi } = ( \bar { \xi } _ { 1 } , . . . , \bar { \xi } _ { - } \mathrm { d } )$ and K unknown physical fields

$$
f ( \xi ) = \left[ f _ { 1 } ( \xi ) , \ldots , f _ { K } ( \xi ) \right] ^ { T }\tag{1}
$$

The physical problem is assumed to be expressible as

$$
H \left( \boldsymbol { f } , \boldsymbol { D } \boldsymbol { f } ; \boldsymbol { \xi } \right) = 0\tag{2}
$$

where H may contain differential equations, algebraic constraints, conservation relations, boundary conditions, initial conditions, or other known physical relations, and Df denotes the required operators acting on the fields.

Let $\hat { \boldsymbol { f } } = [ \hat { f } _ { 1 } , . . . , \hat { f } \_ { \mathrm { K J ^ { \mathrm { T } } } }$ denote the current approximation. Substitution into the governing relations defines a physical residual $\mathrm { r } ^ { \mathrm { p h y s } }$ . When observations are available, their mismatch with the corresponding approximated quantities defines a data residual rᵈᵃᵗᵃ. These contributions are assembled into the complete residual

$$
r \left( \hat { f } \right) = \left[ r ^ { d a t a } ; r ^ { p h y s } \right)\tag{3}
$$

A weighted least-squares objective is

$$
\boldsymbol { E } \left( \hat { \boldsymbol { f } } \right) = \boldsymbol { r } \left( \hat { \boldsymbol { f } } \right) ^ { T } \boldsymbol { W } \boldsymbol { r } \left( \hat { \boldsymbol { f } } \right) , \boldsymbol { W } = \boldsymbol { W } ^ { T } \succ 0\tag{4}
$$

Conventional parameter optimization reduces E within a representation selected in advance. PI-GMDH additionally permits the functional representation itself to evolve.

For each physical field, let the current approximation be represented by an active set $A _ { k }$

$$
\hat { f } _ { k } ^ { \mathrm { } } \mathopen { } \mathclose \bgroup \left( \xi \aftergroup \egroup \right) = \sum _ { i \in A _ { k } } a _ { k , i } \phi _ { i } \mathopen { } \mathclose \bgroup \left( \xi \aftergroup \egroup \right)\tag{5}
$$

where $\{ \phi _ { i } \}$ is an available or progressively explored functional family. The active sets of different physical fields need not be identical.

A structural extension of field k introduces a package

$$
P _ { \boldsymbol { k } } \subseteq \{ \phi _ { i } \colon i \notin A _ { k } \}\tag{6}
$$

after which $A _ { k }  A _ { k } \cup P _ { k }$ and the coefficients of the enlarged representation are reoptimized. PI-GMDH therefore follows the generic cycle

$$
\begin{array} { r } { \mathrm { c u r r e n t r e p r e s e n t a t i o n }  \mathrm { c a n d i d a t e } \exp { \log { \operatorname { a t i o n } } }  \mathrm { p a c k a g e } \mathrm { c o n s t r u c t i o n }  } \\ { \mathrm { s t r u c t u r a l } \mathrm { e x t e n s i o n }  \mathrm { c o e f f c i e n t o p t i m i z a t i o n } ( 7 ) } \end{array}
$$

The definition of $P _ { k }$ is deliberately not fixed by the general framework. A package may be constructed from a predefined hierarchy, from one candidate at a time, from the complete remaining candidate space, or from a physicsguided selection criterion. This separation permits the influence of structural policy to be studied independently of the functional family and coefficient optimizer.

## 2.2 Physics-guided candidate response and adaptive package construction

Candidate residual response. For physics-guided package construction, consider an elementary candidate function $\phi _ { i }$ for field k. Its local effect is examined through the infinitesimal perturbation

$$
\hat { \boldsymbol f } _ { k } \to \hat { \boldsymbol f } _ { k } + \epsilon \boldsymbol \phi _ { i } , \epsilon \to 0\tag{8}
$$

The first-order response of the complete residual is

$$
\boldsymbol { j } _ { k , i } \mathrm { = } \frac { d } { d \epsilon } r \left( \hat { f } _ { \mathrm { 1 } } , \mathrm { ~ . ~ . ~ . ~ , ~ } \hat { f } _ { k } \mathrm { + } \epsilon \phi _ { i } , \mathrm { , . ~ . . , } \hat { f } _ { K } \right) \mathrm { \Delta } _ { \epsilon \mathrm { = 0 } }\tag{9}
$$

Thus $j _ { k , i }$ is the Gâteaux derivative of the complete residual with respect to perturbation of field k in the functional direction $\phi _ { i }$ . Although the candidate is introduced into a single field, its residual-response vector contains its firstorder influence on every coupled residual component.

The relevance of the candidate follows directly from the first variation of the complete objective:

$$
\left. \frac { d E } { d \epsilon } \right| _ { \epsilon = 0 } = 2 j _ { k , i } ^ { T } W r\tag{10}
$$

The same physics-informed objective used to estimate active coefficients can therefore be used to determine whether an inactive functional direction is locally relevant to the remaining residual.

Adaptive package construction. Candidate directions are compared using the normalized residual alignment

$$
q _ { k , i } { = } \frac { \left| { j _ { k , i } ^ { T } W r } \right| } { \sqrt { \left( { j _ { k , i } ^ { T } W j _ { k , i } } \right) \left( r ^ { T } W r \right) } }\tag{11}
$$

The normalization removes dependence on the magnitudes of the current residual and candidate-induced residual response, so $0 \leq q _ { k , i } \leq 1$ measures their absolute weighted directional alignment.

A second quantity retains information about the absolute interaction magnitude,

$$
{ m _ { k , i } } \mathrm { { = } } \frac { { \left| { { j _ { k , i } ^ { T } } } { W r } \right| } } { N }\tag{12}
$$

where N is the number of sampled points used in the corresponding objective evaluation. Since $j _ { k , i } ^ { \mathrm { ~ ~ } } \mathrm { { w r } }$ accumulates weighted contributions of all residual components over those points, $m _ { k , i }$ expresses this interaction on a per-sampledpoint basis while retaining the weighted combination of residual components.

The two measures play complementary roles. q identifies directions aligned with the current residual independently of scale, while m measures the mean magnitude of their first-order interaction. Both are state dependent: the usefulness of a candidate changes as the coupled field approximations and residuals evolve.

An adaptive package can therefore be constructed by screening candidates according to $q ,$ prioritizing eligible candidates according to m, and introducing a bounded group before reoptimization. The threshold, package-size rule, and hierarchy-exploration rule are implementation parameters rather than defining properties of PI-GMDH itself.

## 2.3 Coeficient optimization

After structural extension, the coefficients of the active representation are reestimated. For field k, let

$$
J _ { k } { = } \Big [ j _ { k , 1 } j _ { k , 2 } \cdots j _ { k , n _ { k } } \Big ]\tag{13}
$$

contain residual-response columns associated with its active functions. With the remaining physical fields held fixed,

$$
r \left( a _ { k } + \delta a _ { k } \right) \approx r + J _ { k } \delta a _ { k }\tag{14}
$$

The weighted linearized least-squares correction is

$$
\delta { a } _ { k } = a r g m i n _ { \delta a } \vert r + J _ { k } \delta { a } \vert ^ { T } W \left( r + J _ { k } \delta { a } \right)\tag{15}
$$

A damped Gauss-Newton step satisfies

$$
\left( J _ { k } ^ { T } W J _ { k } + \lambda I \right) \delta a _ { k } = - J _ { k } ^ { T } W r , a _ { k } \gets a _ { k } + \delta a _ { k }\tag{16}
$$

Updating one field while holding the remaining field representations fixed follows the block-coordinate optimization principle. Block-coordinate optimization itself is not introduced as a new method here; it is the parameteroptimization mechanism coupled to structural construction. [11]

Structural and parameter optimization are linked by

$$
\left( \boldsymbol { J } _ { k } ^ { T } W \boldsymbol { r } \right) _ { i } { = } \boldsymbol { j } _ { k , i } ^ { T } W \boldsymbol { r }\tag{17}
$$

Before activation this quantity determines the first variation associated with a candidate functional direction. After activation the same residual-response vector becomes a Jacobian column used to estimate coefficient corrections.

## 2.4 Package-construction policies

Four package policies are useful for the present study. In adaptive physics-guided construction, $P _ { k }$ contains a selected subset identified from q and m. In complete degree-by-degree construction, $P _ { k }$ contains every function in the next degree shell. In one-term construction, $P _ { k }$ contains one functional component. In all-terms construction, the complete remaining candidate family through a prescribed maximum complexity is introduced in one package.

These variants are treated as controlled PI-GMDH realizations or ablations. They are not claimed to be classical GMDH algorithms. Their purpose is to isolate the consequences of the package-construction policy within a common physics-informed structural framework.

## 3. Application to the Incompressible Navier-Stokes Equations

## 3.1 Governing equations and residuals

Consider an incompressible velocity field $u ( \mathrm { x , t } ) = [ u _ { 1 } , \dotsc , \mathrm { u } _ { - }$ \_d]ᵀ and pressure $p ( \mathbf { x , t } )$ . For constant kinematic viscosity ν,

$$
\nabla \cdot \boldsymbol { u } = 0\tag{18}
$$

$$
\frac { \partial \boldsymbol { u } } { \partial t } { + } ( \boldsymbol { u \cdot \nabla } ) \boldsymbol { u } - \boldsymbol { v } \nabla ^ { 2 } \boldsymbol { u } + \nabla p { = } g\tag{19}
$$

For current approximations $\hat { u }$ and $\hat { p } _ { : }$

$$
r _ { d i v } = \sum _ { b = 1 } ^ { d } \frac { \partial \hat { u } _ { b } } { \partial x _ { b } }\tag{20}
$$

$$
r _ { m _ { a } } = \frac { \partial \hat { u } _ { a } } { \partial t } + \sum _ { b = 1 } ^ { d } \hat { u } _ { b } \frac { \partial \hat { u } _ { a } } { \partial x _ { b } } - \nu \sum _ { b = 1 } ^ { d } \frac { \partial ^ { 2 } \hat { u } _ { a } } { \partial x _ { b } ^ { 2 } } + \frac { \partial \hat { p } } { \partial x _ { a } } - g _ { a }\tag{21}
$$

When observations of velocity component $\mathbf { u _ { a } }$ are available $\mathbf { \nabla } _ { , r _ { u _ { a } } } = \hat { u } _ { a } - u _ { a } ^ { o b s }$ . Other observational, boundary, or initialcondition residuals can be included in the same manner.

## 3.2 Residual response to candidate functions

Velocity candidates. For a candidate $\phi _ { i }$ considered for velocity component $\hat { u } \lrcorner \mathsf { a }$ , the observation and incompressibility responses are

$$
\frac { d r _ { u _ { a } } } { d \epsilon } \bigg | _ { 0 } = \phi _ { i } , \frac { d r _ { d i v } } { d \epsilon } \bigg | _ { 0 } = \frac { \partial \phi _ { i } } { \partial x _ { a } }\tag{22}
$$

The momentum residual associated with the perturbed component responds as

$$
\frac { d r _ { m _ { a } } } { d \epsilon } \biggr ) _ { 0 } = \bigl ( \phi _ { i } \bigr ) _ { t } + \sum _ { b } \hat { u } _ { b } \bigl ( \phi _ { i } \bigr ) _ { x _ { b } } + \hat { u } _ { a , x _ { a } } \phi _ { i } - v \nabla ^ { 2 } \phi _ { i }\tag{23}
$$

For ${ \bf { C } } \ne { \bf { a } } ,$ , nonlinear convection also gives

$$
\left. \frac { d r _ { m _ { c } } } { d \epsilon } \right| _ { 0 } = \hat { u } _ { c , x _ { a } } \phi _ { i }\tag{24}
$$

A candidate introduced into one velocity field therefore generally affects several blocks of the coupled residual.

Pressure candidates. For $\hat { p }  \hat { p } + \epsilon \phi _ { i }$ , incompressibility is unchanged, whereas

$$
\frac { d r _ { m _ { a } } } { d \epsilon } \biggr \rvert _ { 0 } = \frac { \partial \phi _ { i } } { \partial x _ { a } }\tag{25}
$$

In the absence of direct pressure observations or another pressure constraint, the physical residual determines pressure only through its spatial derivatives. The transformation $\widehat { p } ( \mathrm { x , t } )  \widehat { p } ( \mathrm { x , t } ) + \mathrm { g ( t ) }$ leaves $\nabla \widehat { p }$ and therefore the momentum residuals unchanged.

## 3.3 Two-dimensional specialization

$$
r _ { d i v } = \hat { u } _ { x } + \hat { v } _ { y }\tag{26}
$$

$$
r _ { x } = \hat { u } _ { t } + \hat { u } \hat { u } _ { x } + \hat { v } \hat { u } _ { y } - v \left( \hat { u } _ { x x } + \hat { u } _ { y y } \right) + \hat { p } _ { x } - g _ { x }\tag{27}
$$

$$
r _ { y } = \hat { \nu } _ { t } + \hat { u } \hat { \nu } _ { x } + \hat { \nu } \hat { \nu } _ { y } - { v } \left( \hat { \nu } _ { x x } + \hat { \nu } _ { y y } \right) + \hat { p } _ { y } - g _ { y }\tag{28}
$$

If both velocity components are observed, $\boldsymbol { r } _ { u } = \hat { \boldsymbol { u } } - \boldsymbol { u } ^ { \wedge } ($ obs and $r _ { v } = \hat { v } - v {  { \wedge } } 0 {  { \mathrm { b s } } }$ , and one convenient residual ordering is $\mathbf { r } = [ r _ { u } , r _ { v } , r _ { d i v } , r _ { x } , r _ { y } ] ^ { \mathrm { T } }$

For a candidate $\phi _ { i }$ considered for $\hat { u } , \hat { v } _ { : }$ , and $\hat { p }$ respectively, the complete residual responses are

$$
\boldsymbol { j } _ { u , i } \mathrm { = } \bigl [ \phi _ { i } , 0 , \bigl ( \phi _ { i } \bigr ) _ { x } , \bigl ( \phi _ { i } \bigr ) _ { t } + \hat { u } _ { x } \phi _ { i } + \hat { u } \bigl ( \phi _ { i } \bigr ) _ { x } + \hat { v } \bigl ( \phi _ { i } \bigr ) _ { y } - v \Delta \phi _ { i } , \hat { v } _ { x } \phi _ { i } \bigr ) ^ { T }\tag{29}
$$

$$
j _ { v , i } \mathrm { = } \bigl [ 0 \mathrm { , } \phi _ { i } \mathrm { , } \bigl ( \phi _ { i } \bigr ) _ { y } \mathrm { , } \hat { u } _ { y } \phi _ { i } \mathrm { , } \bigl ( \phi _ { i } \bigr ) _ { t } \mathrm { + } \hat { u } \bigl ( \phi _ { i } \bigr ) _ { x } \mathrm { + } \hat { v } _ { y } \phi _ { i } \mathrm { + } \hat { v } \bigl ( \phi _ { i } \bigr ) _ { y } \mathrm { - } v \Delta \phi _ { i } \bigr ) ^ { T }\tag{30}
$$

$$
j _ { p , i } \mathrm { = } \big [ 0 , 0 , 0 , \big ( \phi _ { i } \big ) _ { x } , \big ( \phi _ { i } \big ) _ { y } \big ) ^ { T } , \Delta \phi _ { i } \mathrm { = } \big ( \phi _ { i } \big ) _ { x x } \mathbf { + } \big ( \phi _ { i } \big ) _ { y y }\tag{31}
$$

These expressions make explicit that candidate relevance is determined by the complete coupled system and changes as the current field approximations evolve.

## 4. Taylor-Green Benchmark and Numerical Implementation

## 4.1 Analytical benchmark

$$
y ^ { i } ( x , y , t ) { = } { \sin x } \cos y \exp \left( { - 2 \nu t } \right)\tag{32}
$$

$$
\nu ^ { \dot { \langle } } x , y , t { \doteq } - \cos x \sin y \exp { \left( - 2 \nu t \right) }\tag{33}
$$

$$
p ^ { \downarrow } \big ( x , y , t \big ) = \frac { 1 } { 4 } \big [ \cos \big ( 2 x \big ) + \cos \big ( 2 y \big ) \big ] \exp \big ( - 4 v t \big ) , v = 0 . 0 1\tag{34}
$$

$$
p _ { x } ^ { i } \mathrm { = - } \frac { 1 } { 2 } \mathrm { s i n } \left( 2 x \right) \mathrm { e x p } \left( - 4 v t \right) , p _ { y } ^ { i } \mathrm { = - } \frac { 1 } { 2 } \mathrm { s i n } \left( 2 y \right) \mathrm { e x p } \left( - 4 v t \right)\tag{35}
$$

The benchmark uses the full spatial cell $\mathbf { x } , \mathbf { y } \in [ - \pi , \pi ]$ and the temporal interval from 0 to 1. Before Chebyshev evaluation, each spatial coordinate is divided by π and time is affinely mapped from [0,1] to [−1,1].

## 4.2 Training, validation, and test protocol

The revised experimental protocol contains 250,000 training points, 50,000 validation points, and 100,000 held-out test points, sampled independently using random seeds 43, 44, and 45, respectively. Model checkpoints are selected using validation loss, and the selected checkpoint is subsequently evaluated on the held-out test set.

Exact velocity values $u ^ { * }$ and $v ^ { * }$ provide the observational information entering the physics-informed objective. Pressure and pressure gradients are not supplied as direct training targets. Pressure reconstruction is therefore assessed through its coupling to the velocity fields in the momentum equations.

$$
L { = } \frac { 1 } { N } \sum _ { n } { \left[ r _ { u } ^ { 2 } { + } r _ { v } ^ { 2 } { + } r _ { d i v } ^ { 2 } { + } r _ { x } ^ { 2 } { + } r _ { y } ^ { 2 } \right] } _ { n }\tag{36}
$$

The held-out test set provides out-of-sample evaluation over independently sampled points in the same benchmark domain. It does not constitute a continuous-domain error bound.

## 4.3 Chebyshev realization

The numerical realization uses tensor-product Chebyshev functions. After mapping each computational coordinate to the interval required by the Chebyshev representation, [10]

$$
\phi _ { a b c } { = } T _ { a } \left( \widetilde { x } \right) T _ { b } \left( \widetilde { y } \right) T _ { c } \left( \widetilde { t } \right) { , } D { = } a { + } b { + } c\tag{37}
$$

A complete three-variable expansion through total degree D contains

$$
N _ { D } { = } { \binom { D + 3 } { 3 } }\tag{38}
$$

$$
N _ { 2 } { = } 1 0 , N _ { 6 } { = } 8 4 , N _ { 1 0 } { = } 2 8 6 , N _ { 1 3 } { = } 5 6 0 , N _ { 2 0 } { = } 1 7 7 1\tag{39}
$$

Separate active sets are maintained for the two velocity components and pressure:

$$
\hat { u } = \sum _ { i \in A _ { u } } a _ { u , i } \phi _ { i } , \hat { v } = \sum _ { i \in A _ { v } } a _ { v , i } \phi _ { i } , \hat { p } = \sum _ { i \in A _ { \rho } } a _ { p , i } \phi _ { i }\tag{40}
$$

All derivatives required by the Navier-Stokes residual and candidate-response vectors are evaluated analytically from the Chebyshev representation.

## 4.4 PI-GMDH package variants

Adaptive package. Candidate Chebyshev functions are evaluated using q and m. The implementation begins from a complete low-degree representation, searches the lowest unfinished shell and can continue into higher shells when required, retains candidates satisfying the configured q threshold, prioritizes eligible candidates by m, and introduces them as a bounded package before coefficient reoptimization. The reported runs use $q \geq 0 . 0 5$ and a 256-term cap. The package-mass target is fixed after the initial degree-2 relaxation as 25% of the total eligible degree-3 projection mass, with a lower bound of $1 0 ^ { - 1 4 }$ . Eligible candidates are ordered by mean residual-interaction magnitude and accumulated until this target is reached or the 256-term cap is met; if necessary, the search continues into higher total-degree shells.

Complete degree-by-degree package. Every function in the next total-degree shell is activated before reoptimization. After completion of degree D, the representation contains all Chebyshev functions satisfying $\mathsf { a } { + } \mathsf { b } { + } \mathsf { c } \leq \mathrm { D }$ . In the main full-cell validation/test comparison, this policy was intentionally limited to Dmax = 13 to reduce computational cost because its role was to show the tendency of complete-shell growth rather than to determine its limiting accuracy at D $= 2 0 . \mathrm { A }$ separate large-sample package-granularity experiment follows complete degree-by-degree growth through D $= 2 0$

One-term package. One functional component is introduced per structural extension, followed by coefficient reoptimization. This provides a fine-grained limiting case at the cost of many optimization cycles.

All-terms package. The revised all-terms realization begins with the complete degree-2 representation containing 10 functions per field. After initial coefficient optimization, all remaining functions through total degree 20 are introduced in a single structural extension, producing 1,771 functions per field and $5 { , } 3 1 3$ coefficients across $u , v ,$ and $p ,$ followed by further coefficient optimization.

These package variants are an ablation of structural construction, not different physical models. Adaptive and nonadaptive variants use the same candidate family and physical objective; what changes is how the next group of active functional components is formed.

## 4.5 Coeficient optimization and implementation

After a structural extension, coefficients are updated by alternating block-coordinate damped Gauss-Newton optimization. In the retained implementation, one relaxation cycle uses the sequence $u  p  v  p ;$ pressure is updated twice because each velocity update changes the coupled momentum residual. The reported runs use $\lambda = 1 0 ^ { - 1 0 }$ CuPy float64 arrays, 20,000-point chunks, GPU matrix products, and dense linear solves.

Residual and Jacobian contributions are accumulated without storing the complete Jacobian simultaneously. Coefficient optimization requires $J ^ { T } \boldsymbol { W } \boldsymbol { J }$ and $J ^ { T } W r ,$ , whereas adaptive candidate evaluation requires $j _ { i } ^ { T } W r$ and $j _ { i } ^ { T } W j _ { i }$ . Because damped normal equations are used, conditioning of the active Jacobian remains a relevant numerical consideration, particularly for large nonselective representations.

## 4.6 External reference models

Selected physics-informed MLP and KAN configurations are included as external references. Several relatively simple configurations were examined and the best-performing configurations among those investigated were retained; no exhaustive architecture or hyperparameter search was performed. PI-GMDH structural-control parameters were examined more carefully during development of the present method than the neural-reference settings. The PINN and KAN results should therefore be interpreted as illustrative reference realizations rather than estimates of the best attainable accuracy or computational time of those model classes. Reported run times characterize the retained runs and do not include an unmeasured total cost of exploratory hyperparameter tuning. PI-GMDH variants likewise contain structural-control parameters, so their timings are timings of the specified runs rather than universal optimization costs. [3,8,9]

## 5. Results

## 5.1 Full-cell validation trajectories

Figure 1 shows the validation RMSE trajectories of the two observed velocity components. Adaptive PI-GMDH continues reducing both velocity errors over many orders of magnitude, whereas the reference models and the nonadaptive PI-GMDH variants level off at substantially higher values over the recorded trajectories. Figure 2 shows the corresponding pressure-gradient errors. Pressure-gradient values were not supplied as observational targets; their reconstruction is induced by the coupled Navier-Stokes momentum residuals.

Validation: measured points only.  
![](images/da5f47fad412d7fa09c486269446182c454d09bde1ec64396c957aff5d039535.jpg)  
Validation: measured points only.

Figure 1. Validation velocity-component errors versus elapsed wall-clock time for the Taylor-Green full-cell benchmark. Points are recorded validation measurements without interpolation. The complete degree-by-degree trajectory was intentionally limited to Dmax = 13 to illustrate complete-shell growth at reduced computational cost.  
![](images/f55af8b73d6f72094a2d9a887221a10aacb0c7921e1fdd4b345116fdb811d298.jpg)  
Figure 2. Validation pressure-gradient errors versus elapsed wall-clock time for the Taylor-Green full-cell benchmark. Pressure gradients were not supplied as observational targets; their reconstruction is induced through the coupled momentum equations. Unlike pressure itself, px and py are independent of the arbitrary pressure gauge. The complete degree-by-degree trajectory was intentionally limited to Dmax = 13.

The all-terms comparison is particularly informative because the adaptive and all-terms variants use the same underlying Chebyshev family. Their difference is therefore not simply representational capacity or availability of high-degree functions, but how the active representation is constructed and reoptimized during the solution process. The validation field-error trajectories show the same qualitative separation as the objective values reported below. The complete degree-by-degree trajectory provides a second control. It retains progressive structural growth but activates complete degree shells rather than selectively constructed packages. In the main full-cell benchmark it is deliberately capped at Dmax = 13, so it is used to show the tendency of complete-shell growth and is not interpreted as the best attainable accuracy of that policy at larger maximum degree. The separate package-granularity experiment in Section 5.2 complements this deliberately shortened full-cell trajectory by following complete degree-by-degree growth through D = 20.

## 5.2 Package granularity in a large-sample structural-growth experiment

A separate large-sample Taylor-Green experiment on $\mathbf { x } , \mathbf { y } \in [ - 1 , 1 ]$ and $\mathfrak { t } \in [ 0 , 1 ]$ was used to examine the computational effect of package granularity more directly. Approximately 10⁶ sampled points were used, making each coefficient-relaxation stage substantially more expensive than in the main validation/test protocol. Figure 3 compares the two limiting progressive-growth policies: activation of one functional component per structural extension and activation of a complete total-degree shell before reoptimization.

Pl-GMDH: one-term versus complete degree growth Taylor-Green flow, x, y ∈[−1, 1], t ∈[0, 1]  
![](images/0201d32972053beb5aa411aafc308bceb7e90dc61ca14a698992655d63559572.jpg)  
Recorded training losses; no smoothing or extrapolation. Each curve ends at its last logged measurement.  
Figure 3. Effect of package granularity on PI-GMDH structural growth in the large-sample Taylor-Green experiment on x,y [−1,1], t ∈ ∈ [0,1]. The one-term trajectory was interrupted after reaching an incomplete total-degree-13 representation containing 555, 554, and 554 terms for $\mathbf { u } , \mathbf { v } ,$ and p, respectively; a complete degree-13 basis contains 560 terms per field. The complete degree-by-degree trajectory continued through total degree 20. The one-term endpoint therefore represents an interrupted structural-growth trajectory and is not a completed degree-20 calculation.

The trajectories expose a trade-off between structural selectivity and optimization overhead. The one-term policy reached the approximately $1 0 ^ { - 1 9 }$ loss regime while using an almost complete degree-13 representation, but required repeated coefficient reoptimization after individual structural additions and was still running when interrupted. Complete degree-by-degree growth amortized the relaxation cost over whole shells and reached degree 20 in substantially less wall-clock time. The comparison motivates intermediate package construction: packages can preserve selective structural growth while reducing the number of expensive reoptimization stages relative to oneterm activation.

## 5.3 Validation and held-out test evaluation

The validation-selected checkpoints were evaluated on the independently sampled test set. The results are summarized in Table 1. Adaptive PI-GMDH attained a validation loss of $5 . 2 9 9 \times 1 0 ^ { - 1 9 }$ and a test loss of $5 . 2 9 6 \times 1 0 ^ { - 1 9 }$ The close agreement between these values shows that the very small sampled objective obtained during construction is reproduced on independently sampled points from the same benchmark domain rather than being restricted to the training sample.

Table 1. Validation-selected benchmark results. \*The complete degree-by-degree run was intentionally limited to Dmax = 13 to reduce computational cost and to show the tendency of complete-shell growth; it is not intended to establish the limiting D = 20 performance of that policy. †Neural reference models do not have field-wise Chebyshev active-function counts.
<table><tr><td colspan="1" rowspan="1">Method</td><td colspan="1" rowspan="1"> $\mathbf { L \_ v a l }$ </td><td colspan="1" rowspan="1">L_test</td><td colspan="1" rowspan="1">RMSE u</td><td colspan="1" rowspan="1">RMSE v</td><td colspan="1" rowspan="1"> $\mathbf { R M S E 1 p \_ x }$ </td><td colspan="1" rowspan="1">RMSE p_y</td><td colspan="1" rowspan="1">Active functionsu/v/p</td><td colspan="1" rowspan="1">Time (h)</td></tr><tr><td colspan="1" rowspan="1">PI-GMDHadaptive package</td><td colspan="1" rowspan="1">5.299e-19</td><td colspan="1" rowspan="1">5.296e-19</td><td colspan="1" rowspan="1">3.567e-10</td><td colspan="1" rowspan="1">1.948e-10</td><td colspan="1" rowspan="1">8.508e-9</td><td colspan="1" rowspan="1">5.891e-10</td><td colspan="1" rowspan="1">204 / 201 / 175</td><td colspan="1" rowspan="1">4.40</td></tr><tr><td colspan="1" rowspan="1">PI-GMDHcomplete degree-by-degree(Dmax=13)*</td><td colspan="1" rowspan="1">5.873e-8</td><td colspan="1" rowspan="1">5.776e-8</td><td colspan="1" rowspan="1">8.559e-5</td><td colspan="1" rowspan="1">1.378e-4</td><td colspan="1" rowspan="1">4.310e-3</td><td colspan="1" rowspan="1">7.413e-3</td><td colspan="1" rowspan="1">560 / 560 / 560</td><td colspan="1" rowspan="1">2.56</td></tr><tr><td colspan="1" rowspan="1">Method</td><td colspan="1" rowspan="1"> $\mathbf { L \_ v a l }$ </td><td colspan="1" rowspan="1"> $\mathbf { L _ { \alpha - } t e s t }$ </td><td colspan="1" rowspan="1">RMSE u</td><td colspan="1" rowspan="1">RMSE v</td><td colspan="1" rowspan="1"> $\mathbf { R M S E 1 p \_ x }$ </td><td colspan="1" rowspan="1"> $\mathbf { R M S E 1 p { \_ } y }$ </td><td colspan="1" rowspan="1">Active functionsu/v/p</td><td colspan="1" rowspan="1">Time (h)</td></tr><tr><td colspan="1" rowspan="1">PI-GMDH all-terms (Dmax=20)</td><td colspan="1" rowspan="1">2.776e-6</td><td colspan="1" rowspan="1">2.722e-6</td><td colspan="1" rowspan="1"> $7 . 9 9 2 \mathrm { e } { - 4 }$ </td><td colspan="1" rowspan="1">1.018e-3</td><td colspan="1" rowspan="1"> $3 . 0 7 1 \mathrm { e } ^ { - 2 }$ </td><td colspan="1" rowspan="1">4.064e-2</td><td colspan="1" rowspan="1">1771 / 1771 /1771</td><td colspan="1" rowspan="1">8.35</td></tr><tr><td colspan="1" rowspan="1">PINN reference</td><td colspan="1" rowspan="1">1.754e-6</td><td colspan="1" rowspan="1">1.739e-6</td><td colspan="1" rowspan="1">4.232e-4</td><td colspan="1" rowspan="1">3.999e-4</td><td colspan="1" rowspan="1">1.178e-3</td><td colspan="1" rowspan="1">1.062e-3</td><td colspan="1" rowspan="1">N/A†</td><td colspan="1" rowspan="1">6.34</td></tr><tr><td colspan="1" rowspan="1">KAN reference</td><td colspan="1" rowspan="1">4.127e-7</td><td colspan="1" rowspan="1">4.132e-7</td><td colspan="1" rowspan="1">1.342e-4</td><td colspan="1" rowspan="1">1.385e-4</td><td colspan="1" rowspan="1">4.697e-4</td><td colspan="1" rowspan="1">4.173e-4</td><td colspan="1" rowspan="1">N/A†</td><td colspan="1" rowspan="1">2.50</td></tr></table>

The complete degree-by-degree PI-GMDH realization, intentionally limited to $\mathrm { D m a x } = 1 3 ,$ used the complete 560- function basis in each field and produced a validation loss of $5 . 8 7 3 { \times } 1 0 ^ { - 8 }$ and a held-out test loss of $5 . 7 7 6 { \times } 1 0 ^ { - 8 }$ in 2.56 h. Its held-out RMSEs were $8 . 5 5 9 \times 1 0 ^ { - 5 }$ for u, $1 . 3 7 8 { \times } 1 0 ^ { - 4 }$ for v, $4 . 3 1 0 { \times } 1 0 ^ { - 3 }$ for px, and $7 . 4 1 3 { \times } 1 0 ^ { - 3 }$ for py. Because the purpose of this run is to display the tendency of nonselective complete-shell growth at reduced cost, it should not be interpreted as the limiting accuracy of complete degree-by-degree PI-GMDH at $\mathrm { D } = 2 0$ . The selected PINN and KAN references produced test losses of $1 . 7 3 9 \times 1 0 ^ { - 6 }$ and $4 . 1 3 2 \times 1 0 ^ { - 7 }$ , respectively. Their velocity RMSEs were in the $1 0 ^ { - 4 }$ range, while pressure-gradient RMSEs ranged from $4 . 1 7 3 \times 1 0 ^ { - 4 } \mathrm { t o } 1 . 1 7 8 \times 1 0 ^ { - 3 }$ . These results are reference realizations rather than estimates of the best attainable performance of the neural model classes because no exhaustive architecture or hyperparameter search was performed. The complete degree-by-degree realization has a lower total test objective than these selected neural references at $\mathtt { D m a x } = 1 3$ , although its individual field RMSEs are not uniformly lower; the total objective also contains divergence and momentum residual contributions. These numerical comparisons describe only the retained configurations and are not evidence that the reference approaches cannot achieve comparable or better results under more extensive tuning.

The structural ablation against the all-terms PI-GMDH variant is particularly important. The all-terms model activated 1771 functions for each field, or 5313 coefficients overall, yet produced a test loss of $2 . 7 2 2 { \times } 1 0 ^ { - 6 } .$ . Adaptive PI-GMDH used 580 active functions overall, approximately 9.2 times fewer, required 4.40 h rather than 8.35 h, and reduced the held-out objective by more than twelve orders of magnitude relative to the all-terms realization. Because both variants draw from the same Chebyshev candidate family and use the same physics-informed formulation, this comparison isolates the effect of how the active functional space is constructed and optimized. The adaptive representation remained comparatively compact, containing 204, 201, and 175 active functions for u, v, and p, respectively. The corresponding held-out test RMSEs were 3.567×10 ¹ and⁻ ⁰ $1 . 9 4 8 { \times } 1 0 ^ { - 1 0 }$ for the two velocity components. Pressure and pressure gradients were not supplied as direct training targets; nevertheless, the recovered pressure gradients reached RMSEs of $8 . 5 0 8 { \times } 1 0 ^ { - 9 }$ for px and $5 . 8 9 1 \times 1 0 ^ { - 1 0 }$ for py.

## 5.4 Representation size and structural eficiency

The all-terms degree-20 realization contains 1,771 functions for each of u, v, and p, or 5,313 coefficients overall. The complete degree-by-degree Dmax = 13 realization contains 560 functions per field, or 1,680 overall. By contrast, the validation-selected adaptive representation contains 204, 201, and 175 active functions, respectively, for a total of 580. The adaptive representation therefore uses approximately 10.9% of the coefficient count of the all-terms representation and 34.5% of the coefficient count of the complete degree-13 representation. This difference is structural rather than merely a consequence of a lower search degree: reaching a given total degree during adaptive exploration does not imply that all functions through that degree have been activated.

The combination of active-representation size, elapsed time, and held-out error provides a direct view of structural efficiency. In the present benchmark, the adaptive package policy achieves the smallest validation and test objectives while using far fewer active functions than either nonselective Chebyshev construction. The intentionally capped complete degree-by-degree result additionally shows that progressive complete-shell growth can outperform the oneshot all-terms representation even with a substantially smaller basis, emphasizing that the construction path matters in addition to final nominal capacity.

## 6. Discussion

The proposed method sits at the intersection of several established ideas rather than replacing them. From GMDH it inherits progressive structural self-organization [1,2]; from greedy function-space approximation it shares the idea of evaluating directions in a functional representation [15]; and from adaptive PDE methods it shares residual-driven enrichment of an approximation space [14]. The distinctive mechanism examined here is that candidate directions are scored after propagation through the complete coupled physics-informed residual and are activated in packages before the active coefficient blocks are reoptimized. The experiments are designed primarily to investigate structural construction of a physics-informed functional representation. The comparison among PI-GMDH variants is consequently more informative for this purpose than a direct ranking against neural architectures. Adaptive, complete-degree, one-term, and all-terms variants can share the same physical residual, functional family, and coefficient optimizer while differing in how packages are constructed. This makes package policy an explicit experimental variable.

The large-sample package-granularity experiment in Figure 3 makes the computational trade-off explicit. One-term growth had accumulated 1,663 active functions (555/554/554 for u/v/p) when it was interrupted within degree 13, whereas complete degree-by-degree growth continued through degree 20. The one-term trajectory nevertheless reached the same order of training loss shown by the completed degree-growth trajectory, but at much greater wallclock cost. Because the one-term run was interrupted and the experiment used a different, approximately millionpoint sampling protocol on $\mathbf { x } , \mathbf { y } \in [ - 1 , 1 ]$ , these timings are used only to illustrate structural-growth behavior and are not merged with the full-cell validation/test comparison in Table 1.

The validation and test results strengthen the inference suggested by the structural trajectories: making a large functional space available is not sufficient by itself. The all-terms degree-20 model has access to the complete Chebyshev space through the prescribed maximum degree and contains 5,313 coefficients, yet its held-out test loss is $2 . 7 2 2 { \times } 1 0 ^ { - 6 }$ . Adaptive PI-GMDH uses only 580 active functions and reaches $5 . 2 9 6 \times 1 0 ^ { - 1 9 }$ on the same test protocol. The deliberately limited complete degree-by-degree Dmax = 13 realization, with 1,680 coefficients, reaches $5 . 7 7 6 { \times } 1 0 ^ { - 8 } ,$ demonstrating that the path of progressive shell-wise construction can itself change the attainable result relative to one-shot activation of a larger space. Since these PI-GMDH realizations share the candidate family and physics-informed formulation, the observed separation is associated with construction and optimization path rather than with Chebyshev approximation alone. The comparison is also not determined only by the ultimate accuracy that a construction policy might eventually attain. Even if continued complete degree-by-degree growth or additional optimization of the all-terms representation were able to approach the same accuracy, the wall-clock time and number of reoptimization stages required to reach a given error level would remain practically important. Accordingly, the present timings are interpreted as part of an accuracy–cost trade-off under the tested implementation and hardware, rather than as evidence that the alternative construction policies cannot ultimately reach comparable accuracy. The present ablation isolates package-construction policy (selective versus non-selective, and fine versus coarse structural growth) but does not separately isolate the contribution of evaluating candidate directions against the complete coupled physical residual. In particular, a generic greedy selector based only on observational residuals has not been examined under the same construction protocol. The present results are therefore consistent with, but do not independently establish, the specific advantage of physics-guided candidate scoring over generic greedy structural selection. Isolating this contribution is an important subject for further investigation. This interpretation should not be overstated as a demonstrated conditioning result. In a discrete coupled residual system, mathematically independent Chebyshev functions can nevertheless produce residual-response columns with unfavorable numerical dependence, and normal equations can amplify conditioning difficulties. Condition numbers or singular-value spectra should be measured before conditioning is claimed as the mechanism responsible for the observed plateaus.

Pressure recovery provides an additional test of the coupled physical construction. Pressure and its gradients are absent from the direct observational targets, so their reconstruction is driven through the momentum residuals. The adaptive model attains pressure-gradient RMSEs of $8 . 5 0 8 { \times } 1 0 ^ { - 9 }$ for px and $5 . 8 9 1 \times 1 0 ^ { - 1 0 }$ for py on the held-out set. This does not remove the pressure-gauge ambiguity, but it demonstrates that the gradient information required by the governing equations can be recovered to high accuracy in this benchmark through coupling to the observed velocity fields.

The adaptive package policy used here is itself only one realization of PI-GMDH. Its threshold, package size, hierarchy exploration, and balance between structural extension and coefficient relaxation are control parameters. Several of these decisions are associated with quantities observable during the evolving solution, including distributions of $q$ and $m ,$ current objective reduction, active support size, and effectiveness of recent packages. This suggests a future direction in which package construction is controlled online rather than by fixed rules. Contextualbandit, reinforcement-learning, or simpler adaptive controllers could adjust candidate thresholds, package size, search depth, or the balance between structural growth and coefficient optimization.

The external PINN and KAN results should be interpreted narrowly. The present study does not perform an exhaustive architecture or hyperparameter search for these model classes, while PI-GMDH structural-control parameters were selected and examined more carefully during development of the proposed method. Substantially better PINN, KAN, or other reference results may therefore be obtainable with further tuning, and the reported comparison is not intended to establish that PI-GMDH is intrinsically more accurate or faster than those approaches. Rather, this synthetic benchmark demonstrates that the proposed construction can, under the tested settings, attain both lower error and shorter elapsed time than the selected reference runs. The practical distinction of interest is that PI-GMDH exposes structural decisions during the evolving construction process and may therefore permit some control parameters to be adapted online; this possibility remains to be demonstrated systematically. [6–9] Several limitations remain. The present numerical evidence concerns a single smooth analytical flow with dense, noise-free velocity observations. A held-out test set supports out-of-sample evaluation on independently sampled points in the same benchmark domain, but it does not establish a continuous-domain error bound or performance on different physical regimes. Pressure is inferred without direct pressure targets, yet the benchmark remains substantially more constrained than a forward Navier-Stokes problem supplied only with initial and boundary conditions. Further evaluation should progress from dense reconstruction to sparse observations and ultimately to problems driven primarily by physical constraints and initial/boundary data, and the broader PI-GMDH formulation should be tested beyond Navier-Stokes equations and beyond Chebyshev functions. Demonstrating similar structural behavior on a second differential system would provide stronger evidence that the observed behavior arises from the general construction principle rather than from a particularly favorable interaction between Taylor-Green flow and the chosen functional family.

## 7. Conclusions

This work formulates a Physics-Informed Method of Group Data Handling in which the structure of a functional approximation is constructed progressively during solution. PI-GMDH separates structural construction from parameter optimization: packages of functional components extend the active representation, after which their coefficients are optimized against the complete physical and observational objective. Within this framework, a firstvariation-based adaptive package strategy uses the governing equations to evaluate inactive functional directions before they are introduced. Normalized residual alignment measures directional relevance, while mean residualinteraction magnitude provides complementary information about the strength of the candidate interaction with the current residual. The same residual-response vectors subsequently form Jacobian columns used for coefficient optimization.

The Taylor-Green study is organized as an ablation of package construction. Adaptive physics-guided packages, complete degree-by-degree packages, one-term packages, and a one-shot all-terms package use a common Chebyshev family and common physics-informed formulation but expose the coefficient optimizer to different evolving active spaces. Adaptive PI-GMDH reached a validation loss of 5.299×10 ¹⁹ and a held-out test loss of 5.296×10 ¹⁹ with⁻ ⁻ 204/201/175 active functions for u/v/p. The intentionally limited complete degree-by-degree Dmax = 13 realization reached a test loss of $5 . 7 7 6 { \times } 1 0 ^ { - 8 }$ with 560 functions per field, while the all-terms degree-20 realization used 1771 functions per field but produced a test loss of $2 . 7 2 2 { \times } 1 0 ^ { - 6 }$ . For this benchmark, these results show that structural construction can have a large effect even when substantially larger versions of the same functional family are available. The close agreement between adaptive validation and test losses indicates that the very small sampled residual is reproduced on independently drawn points from the same domain. It should not, however, be interpreted as a continuous-domain error bound, as proof that physics-guided scoring alone causes the observed improvement, or as evidence of universal superiority over other model classes. The selected PINN and KAN configurations were not exhaustively tuned and received less parameter exploration than PI-GMDH during development; consequently, the reported benchmark establishes only that PI-GMDH can achieve a favorable accuracy–time trade-off for this synthetic problem under the tested configurations, not that competing approaches cannot achieve comparable or better performance. Future work should examine data-only versus full-physics structural selection, additional differential systems, sparse and noisy data, initial-boundary-value formulations, more robust linear solvers, and online adaptation of the package-construction policy.

## Funding

This research was partially supported by the U.S. Department of State under Grant No. NSEPEUR23002-1019.

## Declaration of Generative AI and AI-Assisted Technologies

OpenAI Codex was extensively used during the development, debugging, and refinement of the research software. OpenAI GPT was extensively used during preparation of the manuscript, including assistance with scientific writing, restructuring, language editing, and refinement of the mathematical presentation. The research methodology, selection and interpretation of numerical experiments, assessment of the results, and scientific conclusions were determined by the author. All AI-assisted software and manuscript content were reviewed by the author, who takes responsibility for the final content of the work.

## Data and Code Availability

The source code used to implement PI-GMDH and reproduce the numerical experiments presented in this work is publicly available in the PIMGDH repository on GitHub: https://github.com/mininmy/PIMGDH. The benchmark definitions and experimental settings required to reproduce the reported calculations are described in the manuscript and repository.

## References

[1] Ivakhnenko, A. G. (1971). Polynomial theory of complex systems. IEEE Transactions on Systems, Man, and Cybernetics, SMC-1(4), 364-378. doi:10.1109/TSMC.1971.4308320.

[2] Madala, H. R., & Ivakhnenko, A. G. (1994). Inductive Learning Algorithms for Complex Systems Modeling. CRC Press.

[3] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707. doi:10.1016/j.jcp.2018.10.045.

[4] Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P., Wang, S., & Yang, L. (2021). Physics-informed machine learning. Nature Reviews Physics, 3, 422-440. doi:10.1038/s42254-021-00314-5.

[5] Holl, P., Koltun, V., & Thuerey, N. (2020). Learning to control PDEs with differentiable physics. International Conference on Learning Representations (ICLR 2020).

[6] Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics informed neural networks. SIAM Journal on Scientific Computing, 43(5). doi:10.1137/20M1318043.

[7] Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449, 110768. doi:10.1016/j.jcp.2021.110768.

[8] Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljacic, M., Hou, T. Y., & Tegmark, M. (2025). KAN: Kolmogorov-Arnold Networks. International Conference on Learning Representations (ICLR 2025).

[9] Mostajeran, F., & Faroughi, S. A. (2025). Scaled-cPIKANs: Spatial variable and residual scaling in Chebyshevbased physics-informed Kolmogorov-Arnold networks. Journal of Computational Physics, 114116. doi:10.1016/j.jcp.2025.114116.

[10] Trefethen, L. N. (2000). Spectral Methods in MATLAB. SIAM. doi:10.1137/1.9780898719598.

[11] Wright, S. J. (2015). Coordinate descent algorithms. Mathematical Programming, 151, 3-34. doi:10.1007/s10107-015-0892-3.

[12] Zjavka, L. (2013). Approximation of multi-parametric functions using the differential polynomial neural network. Mathematical Sciences, 7, 33.

[13] Zjavka, L. (2016). Numerical weather prediction revisions using the locally trained differential polynomial network. Expert Systems with Applications, 44, 265-274. doi:10.1016/j.eswa.2015.08.057.

[14] Chung, E. T., & Pun, S.-M. (2019). Online adaptive basis enrichment for mixed CEM-GMsFEM. Multiscale Modeling & Simulation, 17(4), 1103-1122. doi:10.1137/18M1222995.

[15] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189-1232. doi:10.1214/aos/1013203451.

[16] Guo, C., Sun, L., Li, S., Yuan, Z., & Wang, C. (2025). Physics-informed Kolmogorov-Arnold network with Chebyshev polynomials for fluid mechanics. Physics of Fluids. doi:10.1063/5.0284999.

[17] B. Chen, J. Wang, H. Xie, Q. Wang, Y. Xia, S. Zhang, J. Zhang, "Physics-informed Chebyshev polynomial neural operator for parametric partial differential equations," Chinese Journal of Aeronautics, in press, 2026, article 104406. doi:10.1016/j.cja.2026.104406.

[18] X. Xiong, K. Lu, Z. Zhang, Z. Zeng, S. Zhou, Z. Deng, R. Hu, "J-PIKAN: A physics-informed KAN network based on Jacobi orthogonal polynomials for solving fluid dynamics," Communications in Nonlinear Science and Numerical Simulation, 152, 2026, article 109414. doi:10.1016/j.cnsns.2025.109414.

[19] H. Thabet, "Adaptive spectral learning for nonlinear variable-order fractional reaction-advection-diffusion equations," Chaos, Solitons & Fractals, 211, 2026, article 118876. doi:10.1016/j.chaos.2026.118876.