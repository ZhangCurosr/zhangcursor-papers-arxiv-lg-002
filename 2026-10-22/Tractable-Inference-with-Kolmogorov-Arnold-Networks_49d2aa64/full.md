# Tractable Inference with Kolmogorov-Arnold Networks

Juan Diego Toscano<sup>a,1</sup>, Sascha Ranftl<sup>a,1,2,∗</sup>, Zhiwei Gao<sup>a</sup>, Li-Lian Wang<sup>b</sup>, George Em Karniadakis<sup>a,∗</sup>

<sup>a</sup>Division of Applied Mathematics, Brown University, Providence, 02912, RI, USA <sup>b</sup>Division of Mathematical Sciences, Nanyang Technological University, 637371, Singapore

## Abstract

Machine-learning surrogates in computational science often require not only accurate predictions but also expectations, variances, conserved integrals, and normalization constants. Conventional neural networks, including Kolmogorov–Arnold networks (KANs), generally evaluate these quantities through Monte Carlo sampling or multidimensional quadrature, making repeated inference costly, particularly in high dimensions. We introduce KANs with Tractable Operators for Uncertainty and Constraints (TOUCKANs), whose finite-degree polynomial outer functions yield a decomposable sum–product representation. Under factorizing input measures, expectations and higher moments then reduce algebraically to deterministic recursions over one-dimensional leaf integrals. Independent and shared inner-block variants form a parameter-eficiency hierarchy, and a universal-approximation proof for its most restricted member extends to the complete class. The same deterministic operators allow conserved integrals and normalization to be imposed throughout training. Across uniform and Gaussian benchmarks through 50 dimensions, deterministic mean and variance errors remained below the median fixed-budget Monte Carlo errors in every case. TOUCK-ANs also conserved Cahn–Hilliard mass to machine precision and imposed normalization and orthogonality through deterministic moments, enabling joint computation of the lowest four Schrödinger eigenpairs in two, three, and five dimensions. TOUCKANs therefore connect KAN approximation with tractable probabilistic circuits, making statistical operators and integral constraints directly available within an expressive neural surrogate.

Keywords: Tractable inference, Kolmogorov-Arnold networks, uncertainty quantification, probabilistic circuits, sum-product networks, scientific machine learning

## 1. Introduction

Kolmogorov–Arnold Networks (KANs) [1] are neural architectures motivated by the Kolmogorov–Arnold representation theorem (KART) [2], which represents continuous multivariate functions through compositions of univariate functions and addition. Rather than assigning fixed nonlinearities to nodes, KANs place learnable univariate functions on the edges of the computational graph. Since their introduction, KANs have expanded into architectures based on alternative basis functions, deep compositions, and low-rank parameterizations [3–6], while recent studies have established broader approximation conditions and surveyed the growing family [7–9]. These developments have supported applications in physics-informed machine learning [1, 10, 11], operator learning [12, 13], and uncertainty quantification [14–16].

At the same time, many tasks in computational science and engineering require not only accurate function approximation but also eficient evaluation of statistical or integral quantities of the learned surrogate. These quantities include expectations, moments, variances, sensitivity indices, conserved integrals, and normalization constants [17–20]. Conventional neural networks, including KANs, generally evaluate them through Monte Carlo sampling or multidimensional quadrature [21–25]. The required evaluations become costly when the dimension is high or when the same functional must be recomputed throughout training or inference.

Low-dimensional factorization provides one route around this bottleneck. Separable and tensorized neural networks combine low-dimensional factors, while tractable probabilistic circuits arrange them through recursive sums and decomposable products [26–28]. Recent KAN-based separable models use related structure to reduce physics-informed training cost [11], whereas low-rank KAN variants target parameter-eficient adaptation [6]. These methods address valuable but distinct objectives and do not make statistical operators tractable through the KAN architecture itself.

To address this gap, we introduce KANs with Tractable Operators for Uncertainty and Constraints (TOUCKANs). The key idea is to restrict the KAN representation so that it acquires the decomposable multiplicative–additive structure of tractable probabilistic circuits, including Deep Polynomial Chaos Expansions [29] and Sum–Product Networks [30, 31]. Specifically, applying finite-degree polynomials to sums of univariate functions allows the multinomial theorem to rewrite the network as a decomposable sum–product representation. Under factorizing input measures, high-dimensional moments then reduce to one-dimensional leaf moments assembled through deterministic recursions. The same quantities can be incorporated into the architecture to enforce conserved integrals and normalization. Tractable inference therefore follows from the architecture of the learned surrogate rather than from a separate sampling procedure.

We realize this structure within the two-block Kůrková–Kolmogorov–Arnold network (KKAN) formulation [32]. The resulting inner-block hierarchy ranges from independent per-dimension networks to shared formulations motivated by the Lorentz and Laczkovich representation theorems [33, 34]. The shared formulations reduce the parameter footprint without changing the moment recursions, while preserving universal approximation throughout the hierarchy. Their smaller footprint also makes strong second-order optimizers such as SSBroyden computationally feasible [35, 36]. Such optimizers have enabled physics-informed networks to attain near machine-precision accuracy [37, 38], but their computational cost generally restricts them to models with relatively few parameters.

The main contributions of this work are as follows:

1. We show that finite-degree polynomial outer functions induce decomposable sum– product representations in KANs and derive deterministic recursions for expectations and variances from one-dimensional leaf moments. The corresponding theorem extends the construction to arbitrary depth and polynomial observables.

2. We construct independent, full-head, diagonal, and rank-one inner parameterizations with progressively smaller parameter footprints. By proving density of the most restricted rank-one class, we establish universal approximation for the complete hierarchy.

3. We use the deterministic integral evaluations to impose conserved linear functionals and normalization by construction. These constraints hold for every parameter value under the quadrature-discretized functional and remain diferentiable during training.

4. Across uniform and Gaussian benchmarks through $D = 5 0$ , deterministic TOUCKAN moment errors remain below the median fixed-budget Monte Carlo errors. The same framework conserves mass to machine precision for the Cahn–Hilliard equation and imposes normalization and orthogonality through deterministic moments, enabling joint computation of the lowest four Schrödinger eigenpairs in $D = 2 , 3$ , and 5.

Together, these results connect KAN approximation with tractable probabilistic circuits, making statistical operators and integral constraints directly available during inference and training.

## 2. From Kolmogorov–Arnold superposition to tractable inference

The Kolmogorov–Arnold representation theorem (KART) states that for any continuous function $f : [ 0 , 1 ] ^ { D } \to \mathbb { R }$ , there exist continuous univariate functions $\phi _ { q , d } : \mathbb { R } \to \mathbb { R }$ and $\Phi _ { q } : \mathbb { R }  \mathbb { R }$ such that [2]

$$
f ( { \bf x } ) = \sum _ { q = 1 } ^ { 2 D + 1 } \Phi _ { q } \left( \sum _ { d = 1 } ^ { D } \phi _ { q , d } ( x _ { d } ) \right) .\tag{1}
$$

Here, $\phi _ { q , d }$ are the univariate inner functions, and $\Phi _ { q }$ are the outer functions. Although KART provides an exact representation, the inner functions arising from its constructive proofs can be highly non-smooth and are therefore unsuitable as trainable components [39]. To address this problem, Kůrková reformulated the theorem as an approximation result in which the inner and outer functions are represented by finite sums of sigmoidal units, while preserving the two-block composition [40]. KKANs generalized this construction by allowing the inner and outer univariate functions to be approximated using broader learnable function classes, including neural networks and basis expansions [32]. KART also inspired modern multilayer architectures such as KANs and their variants [1, 3–6, 8, 9, 11–16]; Appendix A.1 details their relation to the two-block KKAN construction used here.

These approximation results address representation accuracy, but scientific applications often also require expectations, moments, variances, and integral constraints. These quantities can be expressed through observables of the form

$$
\mathbb { E } \big [ h ( f ( \mathbf { x } ) ) \big ] = \int h ( f ( \mathbf { x } ) ) \rho ( \mathbf { x } ) \mathrm { d } V _ { \mathbf { x } } ,\tag{2}
$$

where $\rho ( \mathbf { x } )$ is the input density and $h$ specifies the quantity of interest; for example, $h ( z ) = z$ gives the first moment and $h ( z ) = z ^ { 2 }$ gives the second moment. General neural representations typically evaluate such quantities through sampling or multidimensional quadrature. The cost becomes substantial when the dimension is high or the functional must be recomputed during training, inference, or uncertainty propagation.

One established route to reducing this cost is to construct the representation from lowdimensional factors. Separable and tensorized neural networks combine products of univariate or low-dimensional functions through summation [26, 27],

$$
f ( \mathbf { x } ) = \sum _ { r } \prod _ { d = 1 } ^ { D } g _ { r , d } ( x _ { d } ) .\tag{3}
$$

Tractable probabilistic circuits generalize this organization by arranging low-dimensional leaves through recursive sums and products [28, 30]. Their central structural condition is decomposability: the children of every product node depend on pairwise disjoint variable scopes. Under a factorizing input density, each decomposable product integrates as a product of lower-dimensional integrals, reducing the full integral to low-dimensional computations [31]. The corresponding circuit definitions and structural conditions are provided in Appendix A.2.

The same reduction can be extended to Kůrková and KKAN variants. Since their inner approximants remain univariate, decomposability can be enforced through the outer functions. Consider the quadratic choice $\Phi _ { q } ( t ) = t ^ { 2 }$ . Substitution into (1) gives

$$
\begin{array} { l } { \displaystyle f ( { \bf x } ) = \sum _ { q = 1 } ^ { 2 D + 1 } \left( \sum _ { d = 1 } ^ { D } \phi _ { q , d } ( x _ { d } ) \right) ^ { 2 } } \\ { \displaystyle = \sum _ { q = 1 } ^ { 2 D + 1 } \left[ \sum _ { d = 1 } ^ { D } \phi _ { q , d } ^ { 2 } ( x _ { d } ) + 2 \sum _ { d < d ^ { \prime } } \phi _ { q , d } ( x _ { d } ) \phi _ { q , d ^ { \prime } } ( x _ { d ^ { \prime } } ) \right] . } \end{array}\tag{4}
$$

The first sum contains only univariate squares, while each cross term is a product of functions defined on distinct coordinates. The expanded representation is therefore decomposable.

Assume now that the input density factorizes as $\begin{array} { r } { \rho ( \mathbf { x } ) = \prod _ { d = 1 } ^ { D } \rho _ { d } ( x _ { d } ) } \end{array}$ , and define the one-dimensional leaf moments

$$
\mu _ { q , d } ^ { ( k ) } = \int \phi _ { q , d } ^ { k } ( x _ { d } ) \rho _ { d } ( x _ { d } ) \mathrm { d } x _ { d } .\tag{5}
$$

Taking the expectation of (4) and applying the factorization of $\rho$ yields

$$
\mathbb { E } _ { \rho } [ f ( \mathbf { x } ) ] = \sum _ { q = 1 } ^ { 2 D + 1 } \left[ \sum _ { d = 1 } ^ { D } \mu _ { q , d } ^ { ( 2 ) } + 2 \sum _ { d < d ^ { \prime } } \mu _ { q , d } ^ { ( 1 ) } \mu _ { q , d ^ { \prime } } ^ { ( 1 ) } \right] .\tag{6}
$$

Hence, the $D .$ -dimensional expectation depends only on univariate leaf moments of orders one and two. This reduction is algebraically exact; any remaining approximation arises only from the quadrature used to evaluate those moments.

For finite-degree polynomial outer functions, the multinomial theorem produces the same decomposable structure: every expanded monomial is a product of powers of univariate inner functions, and its expectation separates under a factorizing density. Recursive expansion extends the argument to deeper polynomial compositions and polynomial observables h, so higher moments reduce to the corresponding leaf moments whenever they exist; in particular, the first and second moments determine the variance. Appendix B gives the complete higherorder and deep-layer derivations, the circuit implementation, and the complexity bounds for the explicitly expanded representation, while Theorem 1 establishes the result for arbitrary depth and polynomial observables.

## 3. Method

## 3.1. A tractable KAN architecture

The preceding section identifies two structural properties that are suficient to obtain a decomposable sum–product representation: (i) the outer nonlinearities must be finitedegree polynomials, so that the multinomial theorem applies; and (ii) the argument of the innermost outer nonlinearity must be a sum of univariate functions, so that the factors in every product generated by the expansion have pairwise disjoint scopes. We incorporate these properties into a tractable modification of the two-block KKAN architecture, which we call TOUCKAN [32].

The resulting TOUCKAN architecture comprises an inner block, a combination layer, and an outer block (Figure 1),

$$
f ( \mathbf { x } ) = \sum _ { q = 1 } ^ { Q } \Phi _ { q } ( \xi _ { q } ) , \qquad \xi _ { q } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \Psi _ { d , q } ( x _ { d } ) .\tag{7}
$$

Here, $Q$ denotes the number of branches, $\Psi _ { d , q }$ denotes a trainable univariate inner branch, and $\Phi _ { q }$ denotes a trainable outer function. We use $\Psi _ { d , q }$ to distinguish these trainable branches from the exact KART inner functions $\phi _ { q , d }$ in (1). The normalized sums $\xi _ { q }$ define the combination variables. Because every $\Psi _ { d , q }$ depends on a single coordinate, each $\xi _ { q }$ remains a sum of univariate functions, satisfying condition (ii) by construction.

The branches remain free modeling choices provided the required univariate leaf moments exist. In the most direct instantiation, each dimension carries its own independent inner network, and the Q output channels of that network are the branches $\Psi _ { d , 1 } , \ldots , \Psi _ { d , Q }$ . Denoting the independent network associated with dimension d by $\pmb { \psi } _ { d } = ( \psi _ { d , 1 } , \dots , \psi _ { d , Q } )$ , the inner branches and outer functions are

$$
\Psi _ { d , q } ( x _ { d } ) = \operatorname { t a n h } \big ( \psi _ { d , q } ( x _ { d } ) \big ) ,\tag{8}
$$

$$
\Phi _ { q } ( \xi _ { q } ) = \sum _ { p = 0 } ^ { P } C _ { q , p } T _ { p } ( \xi _ { q } ) .\tag{9}
$$

The finite Chebyshev expansion makes each $\Phi _ { q }$ a polynomial, satisfying condition (i). To retain the natural bounded domain of the Chebyshev basis, the combination variable $\xi _ { q }$ must remain in (−1, 1). Conventional Chebyshev-based KAN formulations impose this bound by applying tanh after the sum over dimensions [32]. That placement introduces a nonpolynomial function of a multivariate sum into the outer map, compromising decomposability. TOUCKAN instead applies tanh within each univariate branch. Consequently, every $\Psi _ { d , q }$ remains univariate and bounded, while the normalized average in (7) confines $\xi _ { q } ~ \mathrm { t o } ~ ( - 1 , 1 )$ The construction therefore satisfies condition (ii) while providing the appropriate input domain for the outer Chebyshev expansion. Although this direct construction satisfies both requirements, it assigns a complete inner network to every input dimension, causing the inner-block parameter count to grow proportionally with D.

![](images/d5326ae98cadc016e9330aad230750e7d7ce550d6347511c2ccd8216c39703e1.jpg)  
Figure 1: TOUCKAN architecture (7), shown in the parameter-eficient diagonal form (11). Every input dimension $x _ { d }$ is processed by the same shared network $\psi : \mathbb { R } \to \mathbb { R } ^ { Q }$ and specialized by the per-dimension diagonal scaling $\lambda _ { d } .$ , producing the bounded univariate branches $\Psi _ { d , q } = \operatorname { t a n h } \left( \lambda _ { d , q } \psi _ { q } ( x _ { d } ) \right)$ . The first combination layer forms the normalized sums $\begin{array} { r } { \xi _ { q } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } { \Psi _ { d , q } } \in ( - 1 , 1 ) } \end{array}$ , each outer block evaluates a Chebyshev expansion $\begin{array} { r } { \Phi _ { q } ( \xi _ { q } ) = \sum _ { p = 0 } ^ { P } C _ { q , p } T _ { p } ( \xi _ { q } ) } \end{array}$ , and the final combination layer sums all branches to produce $f .$

To address this problem, we turn to Lorentz’s variant of KART [33], which states that the inner functions can be chosen independently of the input dimension up to per-dimension constants,

$$
f ( { \bf x } ) = \sum _ { q = 1 } ^ { Q } g \Big ( \sum _ { d = 1 } ^ { D } \lambda _ { d } \psi _ { q } ( x _ { d } ) \Big ) .\tag{10}
$$

In this representation, the same inner functions $\psi _ { q }$ are used for every input dimension, while the constants $\lambda _ { d }$ distinguish their contributions. Laczkovich [34] extended this result by allowing the constants to depend on both the dimension and the branch through $\lambda _ { d , q }$

Following this construction, we replace the D independent inner networks with a single trainable network $\pmb { \psi } = ( \psi _ { 1 } , \dots , \psi _ { Q } ) : \mathbb { R }  \mathbb { R } ^ { Q }$ that is evaluated on every input dimension (see Figure 1). The outer block remains unchanged, with one branch-specific function $\Phi _ { q }$ for each branch as defined in (9).

In the resulting formulation, each dimension d carries one trainable vector $\boldsymbol { \lambda } _ { d } \in \mathbb { R } ^ { Q }$ , and its branches are obtained through a diagonal specialization of the shared functions,

$$
\Psi _ { d , q } ( x _ { d } ) = \operatorname { t a n h } \big ( \lambda _ { d , q } \psi _ { q } ( x _ { d } ) \big ) ,\tag{11}
$$

$$
\Psi _ { d } ( x _ { d } ) = \operatorname { t a n h } \left( \operatorname { d i a g } ( \lambda _ { d } ) \psi ( x _ { d } ) \right) .\tag{12}
$$

Here, $\mathrm { d i a g } ( \pmb { \lambda } _ { d } )$ denotes the diagonal matrix with entries $\lambda _ { d , 1 } , \ldots , \lambda _ { d , Q }$ , and tanh acts elementwise. Every dimension therefore evaluates the same $Q$ functions and rescales their outputs through $\lambda _ { d , q } .$ . These coeficients are the trainable counterparts of Laczkovich’s constants. Since each branch remains a univariate function of $x _ { d } ,$ , condition (ii) is preserved.

Sharing the inner network also reduces the parameter count. Let $N _ { \psi }$ denote the number of trainable parameters in one complete inner network, and let H denote the width of its trunk. The independent specialization assigns one complete network to every input dimension and requires $D N _ { \psi }$ parameters. The full-head specialization shares the trunk but retains one dimension-specific linear head $W _ { d } \in \mathbb { R } ^ { Q \times H }$ , requiring $N _ { \psi } + D Q H$ parameters. The diagonal specialization (11) replaces each full head with Q scalars and requires $N _ { \psi } + D Q$ parameters. Finally, the rank-one specialization imposes $\lambda _ { d , q } = \lambda _ { d }$ and requires $N _ { \psi } + D$ parameters, recovering the sharing pattern of the Lorentz–Laczkovich form in (10).

Together, these choices define a hierarchy from independent per-dimension networks to the compact rank-one formulation. The inner parameterization can therefore be selected according to the required flexibility and available parameter budget. All four parameterizations retain bounded univariate branches and the polynomial outer functions in (9), so conditions (i) and (ii) hold throughout the hierarchy. The rank-one class is contained in every richer parameterization. Its universal approximation property therefore extends to the complete hierarchy (Theorem 3 and Remark 1). Section 4.1 compares their finite-parameter accuracy and cost quantitatively.

## 3.2. Exact expectation via binomial moment recursion

We assume a factorizing input density, $\begin{array} { r } { \rho ( \mathbf { x } ) = \prod _ { d = 1 } ^ { D } \rho _ { d } ( x _ { d } ) } \end{array}$ . The expectation of every TOUCKAN parameterization in Section 3.1 then reduces exactly to one-dimensional leaf moments. The factorization makes the coordinates $x _ { 1 } , \ldots , x _ { D }$ mutually independent, enabling the dimensionwise recursion below. The complete derivation, numerical-stability analysis, and pseudocode (Algorithm 1) are provided in Appendix E.

Each Chebyshev polynomial admits the monomial form $\begin{array} { r } { T _ { p } ( \xi ) = \sum _ { i = 0 } ^ { p } a _ { p , j } \xi ^ { j } } \end{array}$ with fixed integer coeficients $a _ { p , j }$ . Substituting it into the outer expansion (9) and applying linearity of expectation to (7) gives

$$
\mathbb { E } \big [ f ( \mathbf { x } ) \big ] = \sum _ { q = 1 } ^ { Q } \sum _ { p = 0 } ^ { P } C _ { q , p } \sum _ { j = 0 } ^ { p } a _ { p , j } \mathbb { E } \big [ \xi _ { q } ^ { j } \big ] ,\tag{13}
$$

so the entire computation reduces to the moments $m _ { q , p } = \mathbb { E } [ \xi _ { q } ^ { p } ]$ of each combination variable for $p = 0 , \ldots , P$ . Since expanding the power $\xi _ { q } ^ { p }$ directly by the multinomial theorem would generate $\binom { D + p - 1 } { p }$ cross terms, the moments $m _ { q , p }$ are assembled one dimension at a time. To simplify the notation, we absorb the normalization of the combination layer into the branches by defining $\widetilde { \Psi } _ { d , q } = \Psi _ { d , q } / D$ , which gives $\begin{array} { r } { \xi _ { q } = \sum _ { d = 1 } ^ { D } \widetilde { \Psi } _ { d , q } ( x _ { d } ) } \end{array}$ . The moments of the partial sums $S _ { j , q }$ then obey the binomial recursion

$$
\mathbb { E } \big [ S _ { j , q } ^ { p } \big ] = \sum _ { k = 0 } ^ { p } \binom { p } { k } \mathbb { E } \big [ S _ { j - 1 , q } ^ { p - k } \big ] \mu _ { j , q } ^ { ( k ) } , \qquad S _ { j , q } = \sum _ { d = 1 } ^ { j } \widetilde { \Psi } _ { d , q } ( x _ { d } ) , \qquad \mu _ { d , q } ^ { ( k ) } = \mathbb { E } \big [ \widetilde { \Psi } _ { d , q } ^ { k } ( x _ { d } ) \big ] .\tag{14}
$$

Notice that we can apply this recursion since $S _ { j - 1 , q }$ and $\widetilde { \Psi } _ { j , q } ( \boldsymbol { x } _ { j } )$ are independent under a factorizing density. It is initialized by $\mathbb { E } [ S _ { 1 , q } ^ { p } ] = \mu _ { 1 , q } ^ { ( p ) }$ and terminates at $S _ { D , q } = \xi _ { q } ,$ yielding the moments $m _ { q , p } .$ The implementation uses the equivalent centered recursion derived in Appendix E.1 to keep the intermediate sums bounded.

Each leaf moment $\mu _ { d , q } ^ { ( k ) }$ is a univariate integral evaluated with an M-node Gaussian quadrature rule matched to the marginal $\rho _ { d }$ . We use Gauss–Legendre quadrature for uniform marginals and Gauss–Hermite quadrature for Gaussian marginals. These leaf integrals are the only quantities discretized in the moment calculation. The recursion (14) and reduction (13) are algebraically exact in the resulting leaf moments. The computed expectation is therefore exact for the quadrature-induced discrete product measure. Its continuum error is the corresponding one-dimensional quadrature error, which decays spectrally in M for the smooth branches used here.

The quadrature rule controls only the error in evaluating the trained surrogate. For an underlying target $F$ approximated by $f ,$ the error of the computed value ${ \widehat { \mathbb { E } } } _ { M } [ f ]$ decomposes into a surrogate approximation term and a quadrature term,

$$
\left| \mathbb { E } [ F ] - \widehat { \mathbb { E } } _ { M } [ f ] \right| \leq \mathbb { E } \left[ \left| F - f \right| \right] + \left| \mathbb { E } [ f ] - \widehat { \mathbb { E } } _ { M } [ f ] \right| .\tag{15}
$$

The first term measures the error of the fitted surrogate, whereas the second is the onedimensional quadrature error. Because the quadrature term decays spectrally with M, it becomes negligible once the leaf integrals are suficiently resolved, leaving the fitted-surrogate error as the dominant contribution. Further quadrature refinement cannot reduce this error. Improving the estimate instead requires a more accurate surrogate fit, which motivates the strong optimization procedures used in our experiments. Appendix F measures both contributions against exact and independent high-resolution references.

Substituting the moments $m _ { q , p }$ into (13) yields the expectation of the surrogate without Monte Carlo sampling or multidimensional quadrature. The calculation requires DM innerblock evaluations and $\mathcal { O } ( Q D P M + Q D P ^ { 2 } )$ arithmetic operations. For fixed P and $M ,$ the arithmetic cost is proportional to DQ. The resulting recursion realizes the general leafreplacement identity (B.10), established in Appendix B, for the TOUCKAN architecture.

## 3.3. Exact variance via cross-moment recursion

The same moment framework also gives the variance, $\operatorname { V a r } [ f ] = \mathbb { E } [ f ^ { 2 } ] - \left( \mathbb { E } [ f ] \right) ^ { 2 }$ . Computing the second moment $\mathbb { E } [ f ^ { 2 } ]$ extends the recursion of Section 3.2 from individual combination variables to pairs. Appendix E provides the complete derivation and pseudocode (Algorithm 2).

Collecting powers of the Chebyshev expansion (9) writes each outer block in monomial form, $\begin{array} { r } { \Phi _ { q } ( \xi _ { q } ) = \sum _ { j = 0 } ^ { P } \beta _ { q , j } \xi _ { q } ^ { j } } \end{array}$ . Squaring $\begin{array} { r } { f = \sum _ { q } \Phi _ { q } ( \xi _ { q } ) } \end{array}$ and taking the expectation termwise gives

$$
\mathbb { E } \big [ f ^ { 2 } ( \mathbf { x } ) \big ] = \sum _ { q , q ^ { \prime } = 1 } ^ { Q } \sum _ { j , j ^ { \prime } = 0 } ^ { P } \beta _ { q , j } \beta _ { q ^ { \prime } , j ^ { \prime } } m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) } , \qquad \beta _ { q , j } = \sum _ { p = j } ^ { P } C _ { q , p } a _ { p , j } , \qquad m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) } = \mathbb { E } \big [ \xi _ { q } ^ { j } \xi _ { q ^ { \prime } } ^ { j ^ { \prime } } \big ] .\tag{16}
$$

We call $m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) }$ the cross-moments between pairs of combination variables. Although the input coordinates are independent, $\xi _ { q }$ and $\xi _ { q ^ { \prime } }$ depend on the same coordinates and are therefore generally dependent. Their cross-moments cannot in general be assembled from the individual moments computed in Section 3.2, so the binomial recursion must be extended to pairs.

The joint moments of the partial-sum pairs obey the bivariate recursion

$$
\mathbb { E } \big [ S _ { j , q } ^ { p } S _ { j , q ^ { \prime } } ^ { p ^ { \prime } } \big ] = \sum _ { a = 0 } ^ { p } \sum _ { b = 0 } ^ { p ^ { \prime } } \binom { p } { a } \binom { p ^ { \prime } } { b } \mathbb { E } \big [ S _ { j - 1 , q } ^ { p - a } S _ { j - 1 , q ^ { \prime } } ^ { p ^ { \prime } - b } \big ] \nu _ { j , q , q ^ { \prime } } ^ { ( a , b ) } , \qquad \nu _ { d , q , q ^ { \prime } } ^ { ( k , k ^ { \prime } ) } = \mathbb { E } \big [ \widetilde { \Psi } _ { d , q } ^ { k } ( x _ { d } ) \widetilde { \Psi } _ { d , q ^ { \prime } } ^ { k ^ { \prime } } ( x _ { d } ) \big ] .\tag{17}
$$

This is the bivariate analogue of (14), which is recovered by setting $p ^ { \prime } = 0$ . The recursion is initialized by $\mathbb { E } [ S _ { 1 , q } ^ { p } S _ { 1 , q ^ { \prime } } ^ { p ^ { \prime } } ] = \nu _ { 1 , q , q ^ { \prime } } ^ { ( p , p ^ { \prime } ) }$ and delivers all cross-moments $m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) }$ at the final coordinate.

Despite the bivariate recursion, each joint leaf moment $\nu _ { d , q , q ^ { \prime } } ^ { ( k , k ^ { \prime } ) }$ remains a univariate integral because both factors depend only on the scalar coordinate $x _ { d }$ . These moments are evaluated with the same measure-matched quadrature used for the marginal leaf moments in Section 3.2.

Substituting the cross-moments into (16) and subtracting the squared expectation computed in Section 3.2 yields the variance,

$$
\mathrm { V a r } \big [ f ( \mathbf { x } ) \big ] = \sum _ { q , q ^ { \prime } = 1 } ^ { Q } \sum _ { j , j ^ { \prime } = 0 } ^ { P } \beta _ { q , j } \beta _ { q ^ { \prime } , j ^ { \prime } } m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) } - \Big ( \mathbb { E } \big [ f ( \mathbf { x } ) \big ] \Big ) ^ { 2 } .\tag{18}
$$

As in Section 3.2, the cross-moment recursion is algebraically exact in the joint leaf moments. The computed variance is therefore exact for the quadrature-induced discrete product measure, while its continuum integration error is inherited from the same one-dimensional quadrature rules. The calculation involves no Monte Carlo sampling or multidimensional integration and requires $\mathcal { O } ( Q ^ { 2 } D P ^ { 2 } M + Q ^ { 2 } D P ^ { 4 } )$ arithmetic operations. For fixed $P$ and $M$ this cost is proportional to $D Q ^ { 2 }$ . The construction extends to the r-th moment of $f$ by propagating joint moments of r-tuples of combination variables. For fixed $r ,$ the resulting cost remains polynomial in $D , Q$ , and P. The first- and second-moment recursions provide the algebraic machinery used below to enforce conserved integrals and normalization.

## 3.4. Exact enforcement of conserved integrals

The expectation recursion also makes it possible to enforce linear integral constraints. Consider a function $u ( t , \mathbf { x } )$ on a box domain $\begin{array} { r } { \bar { \Omega } = \prod _ { d = 1 } ^ { D } [ a _ { d } , b _ { d } ] } \end{array}$ whose weighted spatial integral

must remain constant in time,

$$
\mathcal { Q } _ { w } [ u ] ( t ) = \int _ { \Omega } w ( \mathbf { x } ) u ( t , \mathbf { x } ) d \mathbf { x } = \mathcal { Q } _ { 0 } \qquad \mathrm { f o r ~ a l l ~ } t ,\tag{19}
$$

where the weight factorizes as $\begin{array} { r } { w ( \mathbf { x } ) = \prod _ { d = 1 } ^ { D } w _ { d } ( x _ { d } ) } \end{array}$ . The choice $w \equiv 1$ gives the total mass, while $w ( \mathbf { x } ) = x _ { e }$ gives the first spatial moment in coordinate e. For a TOUCKAN surrogate, the same dimensionwise recursion evaluates $Q _ { w } [ f ] ( t )$ . Let $f ( t , \mathbf { x } )$ follow (7) with input vector $( t , x _ { 1 } , \ldots , x _ { D } )$ , so that the complete input dimension is $D + 1$ and time has index $d = 0$ Each combination variable then contains one time branch and one branch for every spatial coordinate,

$$
\xi _ { q } = \frac { 1 } { D + 1 } \Big ( \Psi _ { 0 , q } ( t ) + \sum _ { d = 1 } ^ { D } \Psi _ { d , q } ( x _ { d } ) \Big ) .\tag{20}
$$

As in Section 3.2, we absorb the normalization into every branch by defining $\widetilde { \Psi } _ { d , q } = \Psi _ { d , q } / ( D +$ 1) for $d = 0 , \ldots , D$ . At fixed t, the time branch is a scalar, whereas each spatial branch depends on only one coordinate. Expanding the outer Chebyshev polynomials therefore reduces $Q _ { w } [ f ] ( t )$ to weighted integrals of powers of $\xi _ { q }$ , which the binomial recursion assembles from one-dimensional leaf integrals. Relative to Section 3.2, the marginal density $\rho _ { d }$ is replaced by the weight $w _ { d }$ , while the time branch contributes the known powers $\widetilde { \Psi } _ { 0 , q } ( t ) ^ { k }$ in place of moments. The resulting weighted leaf moments and combination-variable moments are

$$
\mu _ { d , q } ^ { ( k ) } ( w ) = \int \widetilde { \Psi } _ { d , q } ( x _ { d } ) ^ { k } w _ { d } ( x _ { d } ) \mathrm { d } x _ { d } , \qquad m _ { q , j } ( t ) = \int _ { \Omega } \xi _ { q } ^ { j } w ( \mathbf { x } ) \mathrm { d } \mathbf { x } .\tag{21}
$$

Here, $m _ { q , j } ( t )$ is the weighted counterpart of the moment $m _ { q , j } = \mathbb { E } [ \xi _ { q } ^ { j } ]$ from Section 3.2. The recursion requires the weight to factorize but not to be positive, so signed factors such as $w _ { d } = x _ { d }$ are admissible. Substituting these weighted moments into (13) gives

$$
\mathcal { Q } _ { w } [ f ] ( t ) = \sum _ { q = 1 } ^ { Q } \sum _ { p = 0 } ^ { P } C _ { q , p } \sum _ { j = 0 } ^ { p } a _ { p , j } m _ { q , j } ( t ) .\tag{22}
$$

This expression is diferentiable with respect to the network parameters and time. Its weighted leaf moments are evaluated by one-dimensional quadrature, and the complete calculation has the same $\mathcal { O } ( Q D P M + Q D P ^ { 2 } )$ arithmetic cost as the expectation recursion in Section 3.2.

The deterministic evaluation of $Q _ { w } [ f ] ( t )$ allows the conservation law (19) to be imposed by a final output reprojection. At each time, the drift is $\mathcal { Q } _ { w } [ f ] ( t ) - \mathcal { Q } _ { 0 }$ . We remove it by subtracting a multiple of a fixed correction profile $g ( \mathbf { x } )$ , whose spatial form determines how the correction is distributed. The coeficient is chosen so that the correction carries exactly the drift and requires only $\mathcal { Q } _ { w } [ g ] \neq 0$ . For total mass, $g \equiv 1 \mathrm { g i v e s } \ Q _ { w } [ g ] = | \Omega |$ . The constrained surrogate is

$$
\widetilde { f } ( t , \mathbf { x } ) = f ( t , \mathbf { x } ) - \frac { \mathcal { Q } _ { w } [ f ] ( t ) - \mathcal { Q } _ { 0 } } { \mathcal { Q } _ { w } [ g ] } g ( \mathbf { x } ) , \qquad \mathrm { s o ~ t h a t } \qquad \mathcal { Q } _ { w } [ \widetilde { f } ] ( t ) = \mathcal { Q } _ { 0 } \quad \mathrm { f o r ~ a l l ~ } t .\tag{23}
$$

Equation (23) satisfies the prescribed functional for every value of the trainable parameters. Because the weighted leaf moments are evaluated by one-dimensional quadrature, the reprojection is exact for the quadrature-discretized functional. Conservation of the continuum integral (19) inherits the accuracy of those quadrature rules. The correction is diferentiable with respect to the network parameters and can therefore be treated as part of the network during training.

## 3.5. Exact density estimation: normalization as a hard constraint

The conserved functional in Section 3.4 is linear and can be enforced by an additive reprojection. A normalized density must instead remain nonnegative and integrate to one. Existing model classes handle this normalization in diferent ways. Energy-based models [41, 42] generally require sampling when estimating an intractable partition function. Normalizing flows [43] avoid direct integration by restricting the architecture to invertible maps with tractable Jacobians, while penalty formulations enforce unit integral only approximately. For a TOUCKAN surrogate, the normalizer is the second moment computed by the crossmoment recursion of Section 3.3. Normalization can therefore be enforced multiplicatively by division.

Let $\begin{array} { r } { \rho ( \mathbf { x } ) = \prod _ { d = 1 } ^ { D } \rho _ { d } ( x _ { d } ) } \end{array}$ be a factorizing base density and let $f$ be a TOUCKAN (7). Define the normalized model

$$
p ( \mathbf { x } ) = { \frac { f ^ { 2 } ( \mathbf { x } ) \rho ( \mathbf { x } ) } { \mathbb { E } _ { \rho } \left[ f ^ { 2 } \right] } } , \qquad { \mathrm { e q u i v a l e n t l y } } \qquad \psi ( \mathbf { x } ) = { \frac { f ( \mathbf { x } ) { \sqrt { \rho ( \mathbf { x } ) } } } { \sqrt { \mathbb { E } _ { \rho } \left[ f ^ { 2 } \right] } } } , \qquad p = \psi ^ { 2 } ,\tag{24}
$$

The denominator $\mathbb { E } _ { \rho } [ f ^ { 2 } ]$ is supplied by (16) and the cross-moment recursion (17). The model is well defined whenever $\mathbb { E } _ { \rho } [ f ^ { 2 } ] > 0$ , a condition that fails only when $f$ vanishes $\rho -$ almost everywhere. Under this condition, squaring guarantees $p \geq 0$ , while division by the second moment enforces $\textstyle \int p = 1$ for every parameter value. Numerically, this identity is exact for the quadrature-discretized normalizer, and continuum normalization inherits the accuracy of the one-dimensional quadrature rules. Because the denominator is recomputed from the current parameters, normalization holds throughout training and does not depend on the optimizer.

The cross-moment recursion evaluates $\mathbb { E } _ { \rho } [ f ^ { 2 } ]$ as a deterministic, diferentiable function of the trainable parameters with $\mathcal { O } ( Q ^ { 2 } D P ^ { 2 } M \dot { + } \dot { Q } ^ { 2 } D P ^ { 4 } )$ arithmetic cost. For fixed $P$ and $M$ this cost is proportional to $D Q ^ { 2 }$ . Gradients therefore pass directly through the normalizer during training.

For any nonzero scalar ${ \mathit { c } } ,$ the transformation $f \mapsto c f$ leaves $p$ unchanged. Global scale is therefore removed from the represented density rather than controlled by an auxiliary constraint. In variational problems such as quantum eigenvalue problems, where $p = \psi ^ { 2 }$ normalization requires no separate loss term. The deterministic denominator also avoids the sampling floor introduced by a Monte Carlo estimate of the normalizer. This hard-normalized representation underlies the Schrödinger eigensolver evaluated in Section 4.4.

## 4. Results

All experiments use the two-block TOUCKAN architecture introduced in Section 3.1, with the inner-block specialization stated for each case. The implementation uses JAX/Flax;

benchmark definitions and implementation details are provided in Appendix H and $\mathrm { A p \mathrm { - } }$ pendix I, respectively.

## 4.1. Parameter eficiency

Table 1: Parameter eficiency of the four TOUCKAN inner-block specializations on the Lüthen benchmark at $D = 2 0$ . The table reports trainable parameters, relative $L _ { 2 }$ test error, and relative errors of the analytically propagated mean and variance against exact references. All variants share the same outer block and moment recursion. Error columns report the mean ± standard deviation over five random seeds, and bold marks the best mean in each column.
<table><tr><td>Specialization and inner block  $\overline { { \Psi _ { d } ( x _ { d } ) } }$ </td><td>Params</td><td>rel.  $\overline { { L _ { 2 } } }$ </td><td>E[f] rel. error</td><td>Var[.f] rel. error</td></tr><tr><td>Independent  $\overline { { \operatorname { t a n h } ( \psi _ { d } ( x _ { d } ) ) } }$  [32]</td><td>41,796</td><td> $\mathbf { ( 8 . 8 \pm 5 . 5 ) \times 1 0 ^ { - 4 } }$ </td><td> $( \mathbf { 7 . 2 \pm 6 . 5 } ) \times \mathbf { 1 0 ^ { - 4 } }$ </td><td> $\mathbf { \overline { { ( 1 . 9 \pm 2 . 4 ) \times 1 0 ^ { - 4 } } } }$ </td></tr><tr><td>Full-head tanh  $( \dot { W _ { d } } h ( x _ { d } ) \dot { + } b _ { d } )$ </td><td>19,072</td><td> $( 9 . 5 \pm 2 . 9 ) \times 1 0 ^ { - 4 }$ </td><td> $( 8 . 8 \pm 3 . 0 ) \times 1 0 ^ { - 4 }$ </td><td> $( 2 . 3 \pm 1 . 9 ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Diagonal tanh  $\left( \operatorname { d i a g } ( \lambda _ { d } ) \psi ( x _ { d } ) \right)$ </td><td>3,533</td><td> $( 3 . 6 \pm 1 . 2 ) \times 1 0 ^ { - 3 }$ </td><td> $( 2 . 9 \pm 1 . 4 ) \times 1 0 ^ { - 3 }$ </td><td> $( 2 . 4 \pm 1 . 7 ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Rank-one  $\operatorname { t a n h } \left( \lambda _ { d } \psi ( x _ { d } ) \right) \left[ 3 3 \right]$ </td><td>2,733</td><td> $( 7 . 7 \pm 0 . 6 ) \times 1 0 ^ { - 2 }$ </td><td> $( 1 . 7 \pm 1 . 1 ) \times 1 0 ^ { - 2 }$ </td><td> $( 4 . 0 \pm 1 . 9 ) \times 1 0 ^ { - 3 }$ </td></tr></table>

We evaluate the parameter–accuracy trade-of of the inner-block hierarchy on the Lüthen benchmark [44] at $D = 2 0$ , whose mean and variance are available exactly. The comparison includes the independent, full-head, diagonal, and rank-one specializations defined in Section 3.1. Only the inner parameterization changes. All variants use the same outer block and SOAP training protocol [45] across five random seeds. The benchmark, data preparation, and complete training configuration are provided in Appendix H.1 and Appendix I.1.

For every trained surrogate, the analytically propagated moments agree with a $2 \times 1 0 ^ { 6 } .$ sample Monte Carlo estimate of the same network to within one Monte Carlo standard error. This independent check confirms that the errors against the exact benchmark moments are dominated by surrogate fit rather than moment propagation. The mean and variance together evaluate in 0.02 s, and the ordering of the four specializations is preserved in every seed.

The diagonal specialization uses 11.8 times fewer parameters than the independent specialization while matching its variance accuracy within seed variability. At this seed count, the independent, full-head, and diagonal variants are statistically indistinguishable in variance accuracy, with mean relative errors near $2 \times 1 0 ^ { - 4 }$ . The rank-one specialization reduces the parameter count only modestly beyond the diagonal form but increases the relative $L _ { 2 }$ error to approximately 8%.

This finite-budget loss does not contradict the universal-approximation result in $\mathrm { A p \mathrm { - } }$ pendix D. It is consistent with the rank-one structure, in which the per-dimension arguments $\lambda _ { d } \psi ( x _ { d } )$ are collinear before the tanh and can distinguish shape diferences only through saturation. On this benchmark, the diagonal coupling $\lambda _ { d , q }$ is therefore the least parameterized specialization that matches the richer variants in variance accuracy while maintaining a sub-percent relative $L _ { 2 }$ error. We consequently use the diagonal specialization for the highdimensional inference experiments in Section 4.2.

## 4.2. Tractable inference

Using the diagonal specialization selected in Section 4.1, we evaluate tractable inference across $D \in \{ 5 , 1 2 , 2 5 , 5 0 \}$ on three benchmarks. To test performance under two factorizing input densities, we use the Lüthen [44] and Bratley [46] functions with uniform inputs and the Oakley–O’Hagan-type function [47] with Gaussian inputs. All three admit exact means and variances, providing reference values for the propagated moments. Their definitions and exact reference calculations are given in Appendix H.1–Appendix H.3. The diagonal architecture remains compact at these dimensions, making second-order optimization with SSBroyden computationally feasible and thereby further reducing the optimization error of the fitted surrogate. The complete training procedure is summarized by Algorithm 3 in Appendix I.2.

The recursions directly compute the mean and raw second moment $\mathbb { E } [ f ^ { 2 } ]$ . At $D = 5 0 .$ the deterministic estimates overlap the exact references for both quantities across all three benchmarks, whereas the Monte Carlo estimates remain visibly dispersed at $N = 2 \times 1 0 ^ { 6 }$ (Figure 2).

Table 2 extends this comparison across all four dimensions and reports the variance derived from the two propagated moments.

Table 2: Relative errors of the expectation and variance for the three benchmarks across $D \in \{ 5 , 1 2 , 2 5 , 5 0 \}$ The surrogate column reports the test relative $L _ { 2 }$ error. TOUCKAN columns report one deterministic propagation through the trained surrogate. MC (median) is the median error over 200 independent runs of the true integrand, each with $N = 2 \times 1 0 ^ { 5 }$ samples (Appendix I.2). TOUCKAN is more accurate than the median Monte Carlo estimate in every case. For $D \leq 2 5$ , it outperforms all 200 Monte Carlo mean replications and all but at most one variance replication. At $D = 5 0$ , it outperforms at least 98% of the mean and 89% of the variance replications. The variance entries marked with $\ddagger$ are limited by floating-point roundof in the second-moment recursion rather than by the surrogate.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">D</td><td rowspan="2">Surrogate rel.  $L _ { 2 }$ </td><td colspan="3"> $\overline { { \mathbb { E } [ f ] } }$  rel. error</td></tr><tr><td></td><td>TOUCKAN MC (median) | TOUCKAN MC (median)</td><td></td></tr><tr><td rowspan="4">Lüthen</td><td>5</td><td> $\overline { { 1 . 0 \times 1 0 ^ { - 6 } } }$ </td><td> $5 . 6 4 \times 1 0 ^ { - 1 0 }$ </td><td> $\overline { { 1 . 3 4 \times 1 0 ^ { - 3 } } }$   $\overline { { 1 . 1 3 \times 1 0 ^ { - 8 } } }$ </td><td> $\overline { { 2 . 1 0 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>12</td><td> $2 . 7 \times 1 0 ^ { - 6 }$ </td><td> $1 . 3 0 \times 1 0 ^ { - 9 }$ </td><td> $3 . 0 7 \times 1 0 ^ { - 4 }$ </td><td> $9 . 2 7 \times 1 0 ^ { - 9 }$ </td><td> $2 . 0 9 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>25</td><td> $4 . 5 \times 1 0 ^ { - 6 }$ </td><td> $1 . 7 7 \times 1 0 ^ { - 9 }$ </td><td> $2 . 5 3 \times 1 0 ^ { - 4 }$ </td><td> $3 . 8 1 \times 1 0 ^ { - 8 }$ </td><td> $2 . 3 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>50</td><td> $8 . 9 \times 1 0 ^ { - 6 }$ </td><td> $1 . 6 3 \times 1 0 ^ { - 1 0 }$ </td><td> $1 . 1 4 \times 1 0 ^ { - 4 }$ </td><td> $1 . 6 6 \times 1 0 ^ { - 8 }$ </td><td> $1 . 9 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="4">Bratley</td><td>5</td><td> $2 . 9 \times 1 0 ^ { - 4 }$ </td><td> $7 . 2 2 \times 1 0 ^ { - 7 }$ </td><td> $\overline { { 1 . 0 3 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 8 0 \times 1 0 ^ { - 6 } } }$ </td><td> $1 . 7 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>12</td><td> $1 . 7 \times 1 0 ^ { - 3 }$ </td><td> $1 . 5 5 \times 1 0 ^ { - 7 }$ </td><td> $1 . 0 4 \times 1 0 ^ { - 3 }$ </td><td> $1 . 4 5 \times 1 0 ^ { - 5 }$ </td><td> $1 . 9 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>25</td><td> $4 . 3 \times 1 0 ^ { - 3 }$ </td><td> $3 . 5 4 \times 1 0 ^ { - 7 }$ </td><td> $9 . 8 3 \times 1 0 ^ { - 4 }$ </td><td> $9 . 1 1 \times 1 0 ^ { - 6 }$ </td><td> $1 . 7 5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>50</td><td> $1 . 1 \times 1 0 ^ { - 2 }$ </td><td> $1 . 3 1 \times 1 0 ^ { - 5 }$ </td><td> $1 . 0 8 \times 1 0 ^ { - 3 }$ </td><td> $4 . 5 9 \times 1 0 ^ { - 4 \ddagger }$ </td><td> $1 . 7 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td rowspan="4">Oakley</td><td>5</td><td> $\overline { { 4 . 7 \times 1 0 ^ { - 5 } } }$ </td><td> $2 . 6 8 \times 1 0 ^ { - 7 }$ </td><td> $\overline { { 2 . 3 3 \times 1 0 ^ { - 3 } } }$ </td><td> $1 . 9 0 \times 1 0 ^ { - 7 }$ </td><td> $\overline { { 2 . 2 5 \times 1 0 ^ { - 3 } } }$ </td></tr><tr><td>12</td><td> $1 . 4 \times 1 0 ^ { - 4 }$ </td><td> $1 . 3 5 \times 1 0 ^ { - 7 }$ </td><td> $1 . 4 5 \times 1 0 ^ { - 3 }$ </td><td> $3 . 4 5 \times 1 0 ^ { - 7 }$ </td><td> $2 . 1 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>25</td><td> $2 . 3 \times 1 0 ^ { - 4 }$ </td><td> $4 . 7 1 \times 1 0 ^ { - 7 }$ </td><td> $9 . 2 4 \times 1 0 ^ { - 4 }$ </td><td> $1 . 0 2 \times 1 0 ^ { - 6 }$ </td><td> $2 . 1 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>50</td><td> $1 . 1 \times 1 0 ^ { - 3 }$ </td><td> $8 . 5 0 \times 1 0 ^ { - 7 }$ </td><td> $7 . 9 9 \times 1 0 ^ { - 4 }$ </td><td> $2 . 6 9 \times 1 0 ^ { - 4 \ddagger }$ </td><td> $1 . 8 6 \times 1 0 ^ { - 3 }$ </td></tr></table>

Across dimensions, the propagated mean errors range from $1 0 ^ { - 1 0 }$ to $1 0 ^ { - 5 }$ and the variance errors from $1 0 ^ { - 8 }$ to $1 0 ^ { - 4 }$ , remaining below the median fixed-budget Monte Carlo error in every case. The median across replications characterizes the typical error of one fixed-budget Monte Carlo run. Even after all 200 Monte Carlo runs are averaged at a cost of $4 \times 1 0 ^ { 7 }$ evaluations (Figure 3), the deterministic mean remains more accurate in every case except Bratley at $D = 5 0$ . The deterministic variance is less accurate only for Bratley at $D = 1 2$ and $D = 5 0$ . These exceptions are confined to Bratley, which has the largest surrogate errors at the afected dimensions.

![](images/0655f7461f38638ee61151d334fe89ffb30df14ed049bdc402085183bb69bf55.jpg)  
Figure 2: Analytical TOUCKAN moments and Monte Carlo convergence at $D = 5 0 .$ . Rows correspond to the Lüthen, Bratley, and $\mathrm { O a k l e y - O ^ { \prime } }$ Hagan benchmarks, while columns show the mean and raw second moment. Blue dots show 20 independent Monte Carlo estimates of the true integrand at each sample size up to $N = 2 \times 1 0 ^ { 6 }$ , and the blue shaded region marks their ±1σ envelope. The red line is the exact value, and the black dashed line is the deterministic TOUCKAN estimate. The Monte Carlo envelope contracts at the expected $N ^ { - 1 / 2 }$ rate but remains visibly dispersed at the largest budget. The TOUCKAN estimate overlaps the exact value in all six panels without additional sampling during moment evaluation.

The comparison reflects diferent error sources. At the fixed sampling budget, the Monte Carlo median remains near $1 0 ^ { - 3 }$ across benchmarks and dimensions. Once the onedimensional quadrature is resolved, the propagated errors instead follow the relative $L _ { 2 }$ error of the surrogate, consistent with the decomposition in Section 3.2 and the sensitivity analysis in Appendix F. The variance at $D = 5 0$ is the exception. There the recursion for the second moment reaches a floating-point roundof floor before the surrogate error, so the two marked entries of Table 2 overstate the error of the surrogate itself. When roundof is negligible, improving the deterministic estimate requires a more accurate surrogate, whereas reducing the sampling error requires additional true-function evaluations.

![](images/5a8d249c42313009e2b1e31a058e44c3182cbc49f19eaf1cc4f332edcc4864f6.jpg)  
Figure 3: Cost and accuracy of TOUCKAN propagation and Monte Carlo across $D \in \{ 5 , 1 2 , 2 5 , 5 0 \}$ . Rows denote benchmarks and columns report wall time, mean error, and raw-second-moment error. MC (1 run) uses the time of one $N = 2 \times 1 0 ^ { 5 }$ run and the median error across 200 independent runs. MC (avg. of 200) reports the total time and error obtained by averaging those runs. TOUCKAN reports one warm evaluation of the compiled deterministic recursions, excluding one-time JIT compilation. The reported timings do not include surrogate training. One TOUCKAN evaluation is up to 2,600 times faster than the 200-run Monte Carlo average. For the mean and raw second moment, TOUCKAN matches or improves the averaged Monte Carlo estimator’s accuracy except for Bratley at $D = 5 0$ . Timings were measured on one NVIDIA L40S GPU.

Figure 3 reports wall time and accuracy for the mean and raw second moment, the two quantities evaluated directly by the recursions. One deterministic propagation takes approximately one millisecond at $D = 5$ and less than one second at $D = 5 0$ . TOUCKAN is up to 2,600 times faster than the 200-run Monte Carlo average. For the mean and raw second moment, TOUCKAN matches or improves the averaged Monte Carlo estimator’s accuracy except for Bratley at $D = 5 0$ , with errors up to $1 0 ^ { 5 }$ times smaller in the remaining cases.

These comparisons concern inference after the surrogate has been trained. Monte Carlo provides a basic sampling reference for the true integrand, whereas TOUCKAN evaluates moments of the fitted surrogate without additional queries to the original model. The same distinction applies to randomized quasi-Monte Carlo, whose comparison is reported in $\mathrm { A p \mathrm { - } }$ pendix G. Both direct sampling methods require access to the true function at the chosen sampling points. The RQMC comparison uses up to $2 ^ { 1 8 }$ evaluations per run at prescribed low-discrepancy points. Obtaining these evaluations can be prohibitive for expensive simulations, while fixed experimental observations or archived data generally do not coincide with the required locations. TOUCKAN instead reuses the trained surrogate for repeated

moment evaluation.

## 4.3. Exact mass conservation: Cahn–Hilliard equation

![](images/8639dd9fc12c6732987a481cf7d5b3e8427d32321f568d1a7ac9ec87514bbc35.jpg)

![](images/c6ddc9df79fd486381907e2615f4480a220cf3e94f855cea709326f4739fd769.jpg)

![](images/a5d718cf1eabc57b99298ef6d1fba907ddf2fe0c268da0a83179e2a149d45e13.jpg)

(d) Loss  
![](images/cda069c8237ee49977d4092c3852c105a5fbc78df7e733e1613117aa42c7b7bd.jpg)

(e) Relative L2 error  
![](images/7fb097c758dd582a99c95794c9fe1c9c6ede20940cebd4e67abc60f470d41131.jpg)

(f) Mass drift |Q[f](t) − Q0l  
![](images/08a7c28b28834813dbe6336b694961ad50029ff9c7c5752a7733867a30cec13f.jpg)  
Figure 4: Cahn–Hilliard equation with the exact-mass reprojection. (a)–(c): reference solution, learned surrogate, and absolute approximation error on a logarithmic color scale for the independent TOUCKAN of Table 3. (d), (e): training loss and relative $L _ { 2 }$ error for the five parameter-matched models of Table 3. The dashed line marks activation of the exact-mass reprojection for the constrained models. The MLP remains unconstrained throughout. (f): 512-node quadrature mass drift $| \mathcal { Q } [ f ] ( t ) - \mathcal { Q } _ { 0 } |$ over time. Every model with the exact-mass reprojection holds this quadrature mass at machine precision, while the unconstrained MLP baseline drifts at the level of its field error.

Beyond post-training inference, deterministic moment propagation can enforce integral constraints throughout training. We demonstrate this capability using the one-dimensional Cahn–Hilliard equation for a double-well potential,

$$
\partial _ { t } u = \partial _ { x x } \bigl ( u ^ { 3 } - u - \varepsilon \partial _ { x x } u \bigr ) , \qquad x \in [ - 1 , 1 ] , \quad t \in [ 0 , 1 ] , \quad \varepsilon = 0 . 0 2 .\tag{25}
$$

With periodic boundary conditions and the initial condition $u ( 0 , x ) = \cos ( \pi x ) - e ^ { - 4 ( \pi x ) ^ { 2 } }$ , the solution conserves the total mass $\begin{array} { r } { \mathcal { Q } [ u ] ( t ) = \int _ { - 1 } ^ { 1 } u ( t , x ) d x = \mathcal { Q } _ { 0 } } \end{array}$ . We impose this invariant using the exact-mass reprojection of Section 3.4 with $w \equiv 1$ and $g \equiv 1$ , so every projected output has total mass $\mathcal { Q } _ { 0 }$ for every parameter value. The Fourier-spectral reference conserves mass to below $1 0 ^ { - 1 6 }$ , and the complete problem and implementation are described in Appendix I.3.

We compare an unconstrained MLP baseline, a tensor neural network (TNN) [26] that we modify to enforce the same exact-mass construction, and three TOUCKAN specializations. The TOUCKANs use the independent, full-head, and diagonal inner blocks defined in Section 3.1. The TNN represents the solution as a sum of separable products in t and x, so its mass becomes a weighted sum of one-dimensional spatial factor integrals and the same reprojection applies directly. All five models have approximately 15,000 trainable parameters and receive the same total second-order optimization budget. Table 3 reports their field errors and mass drift, while the complete architectures, the TNN mass construction, and the training protocol are given in Appendix I.3.

Table 3: Field accuracy and mass conservation for the Cahn–Hilliard benchmark. Relative $L _ { 2 }$ and $L _ { \infty }$ errors are measured against the Fourier-spectral reference, while the mass column reports the worst-case drift max $ { \vert { \mathcal { Q } } \rvert } { f } ] ( t ) - { \mathcal { Q } } _ { 0 }  { \vert { \mathcal { Q } } \rvert }$ under the 512-node quadrature used by the reprojection. All models have approximately 15,000 trainable parameters and receive the same total second-order optimization budget. The constrained models conserve the quadrature mass to machine precision, and for the TOUCKANs independent rules with 1024–8192 nodes confirm the same precision for the continuum integral (Appendix F). For comparison on the same problem, the specialized backward-compatible PINN (bc-PINN) method [48] reports a relative error of $1 . 8 6 \times 1 0 ^ { - 2 }$ . Bold marks the lowest field errors; all constrained mass drifts are at machine precision.
<table><tr><td>Model</td><td>Params</td><td> $\mathrm { r e l . ~ } L _ { 2 }$ </td><td></td><td> $L _ { \infty }$ </td><td> $\overline { { \operatorname* { m a x } _ { t } | \mathscr { Q } [ f ] ( t ) - \mathscr { Q } _ { 0 } | } }$ </td></tr><tr><td>MLP (unconstrained)</td><td>14,928</td><td> $\overline { { 2 . 4 \times 1 0 ^ { - 4 } } }$ </td><td></td><td> $\overline { { { \bf 5 . 2 \times 1 0 ^ { - 4 } } } }$ </td><td> $\overline { { 2 . 3 \times 1 0 ^ { - 4 } } }$ </td></tr><tr><td>TNN (constrained) [26]</td><td>14,753</td><td> $4 . 1 \times 1 0 ^ { - 3 }$ </td><td></td><td> $6 . 7 \times 1 0 ^ { - 2 }$ </td><td> $2 . 9 \times 1 0 ^ { - 1 5 }$ </td></tr><tr><td>TOUCKAN (independent)</td><td>14,840</td><td> $\mathbf { 1 . 2 \times 1 0 ^ { - 4 } }$ </td><td></td><td> $7 . 3 \times 1 0 ^ { - 4 }$ </td><td> $5 . 2 \times 1 0 ^ { - 1 5 }$ </td></tr><tr><td>TOUCKAN (full-head)</td><td>15,328</td><td> $3 . 6 \times 1 0 ^ { - 4 }$ </td><td></td><td> $2 . 2 \times 1 0 ^ { - 3 }$ </td><td> $5 . 6 \times 1 0 ^ { - 1 4 }$ </td></tr><tr><td>TOUCKAN (diagonal)</td><td>14,912</td><td> $1 . 4 \times 1 0 ^ { - 3 }$ </td><td></td><td> $4 . 0 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 \times 1 0 ^ { - 1 4 }$ </td></tr></table>

All three TOUCKAN variants conserved mass to machine precision under the exact-mass reprojection, despite their diferent field errors. The constrained TNN matched this behavior with a drift of $2 . 9 \times 1 0 ^ { - 1 5 }$ , confirming that the reprojection is not specific to TOUCKANs and applies to any architecture whose mass integral is tractable. For the TOUCKANs, independent quadrature refinements confirmed continuum conservation beyond the 512-node rule used during training, with drift remaining near $5 \times 1 0 ^ { - 1 5 }$ (Appendix F, Figure F.8). By contrast, the unconstrained MLP baseline exhibited a mass drift of $2 . 3 \times 1 0 ^ { - 4 }$ , comparable to its field error.

![](images/066ad0eb3a7050c8356a28645eb8a62f6c12ae0c111621093dc61e0d74e17da2.jpg)  
Figure 5: Schrödinger eigenvalue problem in $D = 2 ,$ 3, and 5 (columns). Top row: maximum error in the Gram entries $\langle \psi _ { a } | \psi _ { b } \rangle$ versus number of function evaluations, against converged tensor-product references $( 2 0 0 ^ { 2 } , 1 2 0 ^ { 3 }$ , and $4 8 ^ { 5 }$ nodes): four independent Monte Carlo runs (red, up to $\mathrm { \bar { 1 0 ^ { 8 } } }$ samples), tensor-product Gauss–Hermite quadrature with $n ^ { D }$ nodes (blue), and the exact cross-moment recursion (star), which reaches $1 . 5 \times 1 0 ^ { - 1 4 }$ $1 . 2 \times 1 0 ^ { - 1 1 }$ , and $7 . 5 \times 1 0 ^ { - 8 }$ with 256, 384, and 640 univariate evaluations, respectively (the $D = 5$ value is limited by the convergence of the $4 8 ^ { 5 } .$ -node reference itself). The horizontal axis counts integrand evaluations: the TOUCKAN-based recursion additionally performs $\mathcal { O } ( Q ^ { 2 } D P ^ { 4 } )$ cross-moment arithmetic outside this count, and its full evaluation, including that arithmetic, completes in 1–3 ms per case. Middle row: convergence of the four trainable eigenvalues $E _ { a }$ to the exact spectrum (dotted lines) from a common initialization below the spectrum. Bottom row: training convergence, showing the relative $L _ { 2 }$ error of the eigenfunctions, the relative eigenvalue error averaged over the four states, and the of-diagonal Gram residual max $| { \widehat { G } } - I |$ . The dashed line marks the switch from first- to second-order optimization (S2). Together, these results show that deterministic Gram evaluation avoids both the Monte Carlo sampling floor and tensorproduct growth while all four eigenpairs converge simultaneously.

Hard conservation remained compatible with accurate field reconstruction. The independent TOUCKAN achieved the lowest relative $L _ { 2 }$ error, $1 . 2 \times 1 0 ^ { - 4 }$ , while the unconstrained MLP baseline achieved the lowest $L _ { \infty }$ error, $5 . 2 \times 1 0 ^ { - 4 }$ . Among the constrained TOUCKANs, field accuracy followed the parameter–accuracy hierarchy observed in Section 4.1. Although the TNN admits the same exact-mass construction, its relative $L _ { 2 }$ error of $4 . 1 \times 1 0 ^ { - 3 }$ lies more than an order of magnitude above that of the independent TOUCKAN at the same parameter count and optimization budget. Figure 4 summarizes the field reconstruction, training histories, and mass conservation.

## 4.4. Exact normalization: the Schrödinger eigenvalue problem

Deterministic moment propagation can also enforce normalization, extending its role from conservation laws to the admissibility of eigenstates. We consider the time-independent Schrödinger equation on $\mathbb { R } ^ { D }$ 2

$$
\begin{array} { r } { \mathcal { H } \psi = - \frac { 1 } { 2 } \Delta \psi + V ( \mathbf { x } ) \psi = E \psi , \qquad V ( \mathbf { x } ) = \frac { 1 } { 2 } \mathbf { x } ^ { \top } A \mathbf { x } , } \end{array}\tag{26}
$$

We seek the lowest $N _ { s }$ eigenpairs $( E _ { a } , \psi _ { a } )$ of a coupled harmonic oscillator with $A \ =$ $R \operatorname { d i a g } ( \omega _ { 1 } ^ { 2 } , \dots , \omega _ { D } ^ { 2 } ) R ^ { \top }$ . The dense rotation R couples all input coordinates, so neither the potential nor its eigenfunctions are separable in the original variables. Undoing this rotation nevertheless provides closed-form reference eigenpairs, allowing precise evaluation of a genuinely multivariate problem. The complete construction is given in Appendix I.4.

A neural eigensolver must learn each eigenvalue jointly with its eigenfunction while excluding zero, repeated, and higher-energy solutions. This requires normalized and mutually orthogonal states in addition to a small equation residual. The formulation of Jin et al. [49] enforces these integral constraints through sampled penalties and obtains states sequentially through an orthogonality loss, patience-based solution recognition, and symmetry switching. Its reported experiments are restricted to one-dimensional equations.

TOUCKAN represents all $N _ { s }$ states with a single network in the hard-normalized form $\psi _ { a } ~ = ~ f _ { a } \sqrt { \rho } / \sqrt { G _ { a a } }$ from (24). The Gaussian density $\rho$ captures the decay of the bound states, while the cross-moment recursion evaluates the complete Gram matrix $G _ { a b } = \mathbb { E } _ { \rho } [ f _ { a } f _ { b } ]$ deterministically at every training step. Normalization therefore holds by construction, and orthogonality is imposed through the of-diagonal Gram entries without sampling noise. The Schrödinger-equation residual $r _ { a }$ in (I.3) constrains each learned pair $( E _ { a } , \psi _ { a } )$ . The trace $\operatorname { t r } ( G ^ { - 1 } K )$ is the summed energy of the learned subspace, and by the Ky Fan principle its minimum is attained by the span of the lowest $N _ { s }$ eigenstates [50]. All $N _ { s }$ eigenpairs are optimized jointly from the first iteration, without an eigenvalue scan or sequential deflation. Sampling is used only to evaluate $r _ { a }$ and the Hamiltonian matrix K in (I.5), whereas every norm and overlap is evaluated deterministically. The derivation of the trace term and the complete loss and training procedure are given in Appendix I.4.

Table 4 reports the lowest four eigenpairs in $D \ = \ 2 , \ 3$ , and 5. Their mean relative eigenvalue errors are $1 . 7 \times 1 0 ^ { - 8 } , 1 . 4 \times 1 0 ^ { - 7 }$ , and $6 . 1 \times 1 0 ^ { - 5 }$ , respectively. For comparable one-dimensional eigenvalue problems, including the radial reduction of the hydrogen atom, Jin et al. [49] reported errors of roughly one percent. Within each dimension, the eigenvalue errors are approximately the square of the eigenfunction errors, consistent with the quadratic accuracy of a variational formulation.

The loss and eigenpair errors increase together with dimension, whereas the parameter count and training time grow only mildly. The constraint errors follow a diferent pattern. Normalization is built into the representation, so the values in Table 4 measure agreement with an independent tensor-product quadrature rather than optimization of a normalization penalty. Orthogonality also remains several orders below the eigenfunction error. The deterioration at $D = 5$ therefore reflects the approximation and optimization achieved under the tested configuration rather than a failure of deterministic constraint evaluation.

Table 4: Accuracy of the lowest four eigenpairs of the coupled oscillator in $D = 2 , 3 ,$ and 5. Eigenvalue errors are reported for each state, and the eigenfunction row reports the worst case over the four states. Normalization is evaluated using independent tensor-product Gauss–Hermite rules with $2 0 0 ^ { 2 } , 1 2 0 ^ { 3 }$ , and $4 8 ^ { 5 }$ nodes and reports the largest residual over the four states. Orthogonality is the maximum absolute ofdiagonal entry of the normalized Gram matrix at the final training iteration, evaluated by the deterministic recursion at $M = 1 2 8 .$ . The $D = 5$ normalization value is limited by the reference quadrature rather than the cross-moment recursion. Training times were measured on one NVIDIA H200 GPU.
<table><tr><td></td><td> $D = 2$ </td><td> $D = 3$ </td><td> $D = 5$ </td></tr><tr><td>Trainable parameters</td><td>10,963</td><td>13,075</td><td>17,299</td></tr><tr><td>Relative error  $E _ { 0 }$ </td><td> $2 . 2 \times 1 0 ^ { - 8 }$ </td><td> $4 . 0 \times 1 0 ^ { - 7 }$ </td><td> $5 . 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Relative error  $E _ { 1 }$ </td><td> $2 . 3 \times 1 0 ^ { - 8 }$ </td><td> $1 . 3 \times 1 0 ^ { - 8 }$ </td><td> $2 . 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Relative error  $E _ { 2 }$ </td><td> $7 . 7 \times 1 0 ^ { - 9 }$ </td><td> $5 . 6 \times 1 0 ^ { - 8 }$ </td><td> $2 . 8 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Relative error  $E _ { 3 }$ </td><td> $1 . 4 \times 1 0 ^ { - 8 }$ </td><td> $8 . 1 \times 1 0 ^ { - 8 }$ </td><td> $1 . 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Eigenfunction rel.  $L _ { 2 }$  error</td><td> $2 . 2 \times 1 0 ^ { - 4 }$ </td><td> $2 . 5 \times 1 0 ^ { - 4 }$ </td><td> $1 . 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Normalization,  $\operatorname* { m a x } _ { a } \left| \langle \psi _ { a } | \psi _ { a } \rangle - 1 \right|$ </td><td> $2 . 1 \times 1 0 ^ { - 1 4 }$ </td><td> $5 . 5 \times 1 0 ^ { - 1 2 }$ </td><td> $2 . 3 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Orthogonality,  $\operatorname* { m a x } _ { a \neq b } | \widehat { G } _ { a b } |$ </td><td> $2 . 7 \times 1 0 ^ { - 9 }$ </td><td> $4 . 9 \times 1 0 ^ { - 8 }$ </td><td> $5 . 2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Final training loss</td><td> $7 . 6 \times 1 0 ^ { - 1 2 }$ </td><td> $2 . 7 \times 1 0 ^ { - 8 }$ </td><td> $1 . 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Training time</td><td> $9 0 9 \mathrm { s }$ </td><td> $1 3 0 7 \mathrm { s }$ </td><td> $2 1 9 1 \mathrm { s }$ </td></tr></table>

Figure 5 directly compares the cost of evaluating the Gram matrix. Monte Carlo remains near 10<sup>−4</sup> error after $1 0 ^ { 8 }$ samples, while tensor-product quadrature converges spectrally but grows as $n ^ { D }$ . The cross-moment recursion instead reaches the reference accuracy using 128 univariate nodes per dimension, with complete evaluations taking 1–3 ms. Deterministic Gram evaluation therefore removes the sampling floor from normalization and orthogonality without incurring tensor-product growth.

The same figure shows all four trainable eigenvalues converging to the exact spectrum after joint optimization from the first iteration, without an eigenvalue scan or sequential deflation. The second-order stage produces the final decrease in the eigenfunction, eigenvalue, and orthogonality errors. Figure 6 confirms that the four three-dimensional eigenstates recover the corresponding modal structures. The two- and five-dimensional fields are shown in Figures I.9 and I.10 of Appendix I.4.

![](images/64377ae68614884c52d4a95d71394e469a9f39070fe6bcbfc173643a66529761.jpg)  
Figure 6: Lowest four eigenstates of the coupled three-dimensional oscillator on the $x _ { 3 } = 0$ plane: closed-form reference (left), learned surrogate (center), and absolute error on a logarithmic color scale (right). Each row is labeled by its learned eigenvalue $E _ { a }$ . On this slice, the $E _ { 2 }$ and $E _ { 3 }$ states have similar projected profiles, while the Gram residuals in Table 4 quantify their full-dimensional distinction.

## 5. Summary

In this work, we introduced TOUCKANs, a class of KAN architectures that enable tractable evaluation of statistical quantities through a decomposable polynomial structure. Finite-degree polynomial outer functions allow the multinomial theorem to rewrite the architecture as a hierarchical sum–product representation, connecting Kolmogorov–Arnold superposition with tractable probabilistic circuits. TOUCKANs instantiate this principle within the two-block KKAN architecture using either independent per-dimension networks or shared inner networks motivated by the Lorentz–Laczkovich theorems. The shared formulations specialize each dimension through full-head, diagonal, or rank-one couplings, reducing the parameter footprint as the coupling becomes more restrictive. The universality proof applies to the most restricted rank-one subclass and therefore to the complete TOUCKAN hierarchy (Theorem 3 and Remark 1).

Under factorizing input measures, the resulting sum–product representation reduces expectations and higher moments to one-dimensional integrals over the network leaves. The binomial recursions assemble these quantities without constructing the fully expanded graph. Their algebra is exact in the evaluated leaf moments, while the continuum error is limited to the corresponding one-dimensional quadrature. Quantities usually estimated through sampling can therefore be evaluated deterministically at any stage of training or inference.

Across benchmarks with uniform and Gaussian inputs up to $D = 5 0$ , the deterministic mean and variance errors remained below the median fixed-budget Monte Carlo errors in every case. Once the leaf quadrature was resolved, the remaining discrepancy followed the fitted-surrogate error rather than the propagation procedure. The one exception is the variance at $D = 5 0$ , where the second-moment recursion reaches a floating-point roundof floor before the surrogate error. The diagonal specialization provided the best tested parameter– accuracy balance for these high-dimensional experiments and kept second-order refinement computationally feasible [35]. In the lower-dimensional physics-informed problems, the richer independent and full-head couplings instead attained the lowest field errors. The inner parameterization therefore controls the finite-budget accuracy–cost trade-of, while the moment recursions remain unchanged across the hierarchy.

These benchmarks validate deterministic moment evaluation within an already-trained surrogate. Repeated moments can then be computed without new true-function evaluations, including when only fixed observations or archived simulation data are available.

Deterministic moment propagation also allows integral quantities to be enforced during training rather than approximated through sampled constraints. For the Cahn–Hilliard equation, the exact-mass reprojection conserved mass to machine precision, while the independent TOUCKAN attained a relative $L _ { 2 }$ error of $1 . 2 \times 1 0 ^ { - 4 }$ , more than two orders of magnitude below the PINN result reported on the same problem [48]. For the coupled Schrödinger problem, deterministic normalization and orthogonality, combined with the Ky Fan trace objective, enabled joint computation of the lowest four eigenpairs in $D = 2 , 3$ and 5. These results extend deterministic moment propagation from post-training inference to structural conservation and normalization during training.

The branch count grows linearly with D, and the moment recursions have the polynomial costs derived in Section 3. In practice, the tighter limitation is the coverage of the joint input space during training. Tensorized or separable architectures such as TNNs and SPINNs can form a Cartesian grid of $n ^ { D }$ joint locations from n coordinate values along each axis [26, 27]. Thus, 100 values per coordinate span $1 0 ^ { 6 }$ joint locations in three dimensions. TOUCKAN training instead evaluates complete coordinate tuples, so N collocation samples cover N joint locations and matching the same three-dimensional grid requires $1 0 ^ { 6 }$ pointwise samples. Although moment propagation factorizes the polynomial expansion, the implemented forward surrogate sums the coordinate branches before evaluating the outer polynomial and does not retain the axis-wise Cartesian factorization used during TNN $\mathrm { o r }$ SPINN training. Training coverage can therefore become the practical bottleneck even when moment propagation remains tractable. The current experiments show that sparse pointwise samples sufice for the smooth targets tested through $D = 5 0$ , while targets with richer joint structure may require substantially larger training sets.

The present formulation assumes factorizing input measures, making correlated inputs a primary extension. Variable transformations, tractable conditioning, and marginalization could broaden the admissible distributions and support variance-based sensitivity analysis. Further work should address degenerate eigenspaces, architectures that scale to higher dimensions, and broader classes of decomposable neural networks. Together, these directions extend the connection between KAN-based scientific machine learning and tractable probabilistic circuits established here.

## Data availability

To support reproducibility, the source code for our implementation and the associated data will be released publicly upon acceptance of the manuscript.

## Acknowledgements

S.R. acknowledges support from the Austrian Science Fund (FWF) through the Erwin Schrödinger Fellowship [Grant DOI: 10.55776/J4774]. The authors acknowledge the support of the NIH grant R01AT012312, MURI/AFOSR FA9550-20-1-0358 project, the DOE-MMICS SEA-CROGS DE-SC0023191 award, and the ONR Vannevar Bush Faculty Fellowship (N00014-22-1-2795). L. Wang acknowledges the support from Singapore MOE AcRF Tier 2 Grant: MOE-T2EP20224-0012.

## References

[1] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljacic, T. Hou, M. Tegmark, KAN: Kolmogorov–Arnold networks, in: International conference on learning representations, Vol. 2025, 2025, pp. 70367–70413.

[2] A. N. Kolmogorov, On the representation of continuous functions of many variables by superposition of continuous functions of one variable and addition, Translations American Mathematical Society 2 (28) (1963) 55–59.

[3] L. F. Guilhoto, P. Perdikaris, Deep learning alternatives of the Kolmogorov superposition theorem, in: The Thirteenth International Conference on Learning Representations (ICLR), 2025.

[4] Z. Chao, X. Liu, Z. Wu, X. Li, RBF-KAN: Radial basis function-Kolmogorov-Arnold networks, IEEE Internet of Things Journal (2026).

[5] Z. Bozorgasl, H. Chen, WAV-KAN: Wavelet Kolmogorov-Arnold networks, arXiv preprint arXiv:2405.12832 (2024).

[6] Y. Gao, M. K. Ng, V. Y. F. Tan, Low tensor-rank adaptation of kolmogorov–arnold networks, IEEE Transactions on Signal Processing 73 (2025) 3107–3123. doi:10.1109/ TSP.2025.3588910.

[7] V. Ismailov, Necessary and suficient conditions for universality of kolmogorov-arnold networks, arXiv preprint arXiv:2604.23765 (2026).

[8] S. A. Faroughi, F. Mostajeran, A. H. Mashhadzadeh, S. Faroughi, Kolmogorov-Arnold networks for data-driven, physics-informed, and deep-operator learning: A review, synthesis, and new analysis, Neural Networks (2026) 108791.

[9] A. Noorizadegan, S. Wang, L. Ling, J. P. Dominguez-Morales, A practitioner’s guide to Kolmogorov–Arnold networks, Computer Science Review 62 (2026) 100991.

[10] K. Shukla, J. D. Toscano, Z. Wang, Z. Zou, G. E. Karniadakis, A comprehensive and fair comparison between mlp and kan representations for diferential equations and operator networks, Computer Methods in Applied Mechanics and Engineering 431 (2024) 117290.

[11] B. Jacob, A. A. Howard, P. Stinis, SPIKANs: separable physics-informed Kolmogorov– Arnold networks, Machine Learning: Science and Technology 6 (3) (2025) 035060.

[12] D. W. Abueidda, P. Pantidis, M. E. Mobasher, DeepOKAN: Deep operator network based on Kolmogorov Arnold networks for mechanics problems, Computer Methods in Applied Mechanics and Engineering 436 (2025) 117699. doi:https://doi.org/10. 1016/j.cma.2024.117699.

URL https://www.sciencedirect.com/science/article/pii/S0045782524009538

[13] J. Lee, Z. Liu, X. Yu, Y. Wang, H. Jeong, M. Y. Niu, Z. Zhang, Kano: Kolmogorovarnold neural operator, ICLR 2026ArXiv:2509.16825 (2026).

[14] Z. Gao, G. E. Karniadakis, Scalable bayesian physics-informed kolmogorov-arnold networks, SIAM/ASA Journal on Uncertainty Quantification 13 (3) (2025) 1543–1577. arXiv:https://doi.org/10.1137/25M1729253, doi:10.1137/25M1729253.

[15] Uncertainty quantification with bayesian higher order relu-kans, Machine Learning: Science and Technology 6 (1) (2025) 015073.

[16] A. Mollaali, C. B. Moya, A. A. Howard, A. Heinlein, P. Stinis, G. Lin, Conformalizedkans: Uncertainty quantification with coverage guarantees for kolmogorov-arnold networks (kans) in scientific machine learning (2025). arXiv:2504.15240. URL https://arxiv.org/abs/2504.15240

[17] R. C. Smith, Uncertainty Quantification: Theory, Implementation, and Applications, SIAM, 2013.

[18] A. Saltelli, M. Ratto, T. Andres, F. Campolongo, J. Cariboni, D. Gatelli, M. Saisana, S. Tarantola, Global sensitivity analysis: The primer (2008).

[19] I. M. Sobol, Global sensitivity indices for nonlinear mathematical models and their monte carlo estimates, Mathematics and Computers in Simulation 55 (1-3) (2001) 271– 280.

[20] B. Sudret, Global sensitivity analysis using polynomial chaos expansions, Reliability Engineering & System Safety 93 (7) (2008) 964–979.

[21] Y. Gal, Z. Ghahramani, Dropout as a bayesian approximation: Representing model uncertainty in deep learning, in: international conference on machine learning, PMLR, 2016, pp. 1050–1059.

[22] B. Lakshminarayanan, A. Pritzel, C. Blundell, Simple and scalable predictive uncertainty estimation using deep ensembles, Advances in Neural Information Processing Systems 30 (2017).

[23] A. G. Wilson, P. Izmailov, Bayesian deep learning and a probabilistic perspective of generalization, Advances in Neural Information Processing Systems 33 (2020) 4697– 4708.

[24] M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, P. Fieguth, X. Cao, A. Khosravi, U. R. Acharya, et al., A review of uncertainty quantification in deep learning: Techniques, applications and challenges, Information fusion 76 (2021) 243–297.

[25] A. F. Psaros, X. Meng, Z. Zou, L. Guo, G. E. Karniadakis, Uncertainty Quantification in Scientific Machine Learning: Methods, Metrics, and Comparisons, arXiv:2201.07766 [cs] (2022). arXiv:2201.07766.

[26] Y. Wang, P. Jin, H. Xie, Tensor neural network and its numerical integration, arXiv preprint arXiv:2207.02754 (2022).

[27] J. Cho, S. Nam, H. Yang, S.-B. Yun, Y. Hong, E. Park, Separable physics-informed neural networks, Advances in Neural Information Processing Systems 36 (2024).

[28] Y. Choi, A. Vergari, G. Van den Broeck, Probabilistic Circuits: A Unifying Framework for Tractable Probabilistic Models, Tech. rep., University of California, Los Angeles (2020).

[29] J. Exenberger, S. Ranftl, R. Peharz, Deep polynomial chaos expansion, AISTATS 2026ArXiv preprint:2507.21273 (2026).

[30] H. Poon, P. Domingos, Sum-Product Networks: A New Deep Architecture (2012). arXiv:1202.3732.

[31] R. Peharz, S. Tschiatschek, F. Pernkopf, P. Domingos, On theoretical properties of sumproduct networks, in: Artificial Intelligence and Statistics, PMLR, 2015, pp. 744–752.

[32] J. D. Toscano, L.-L. Wang, G. E. Karniadakis, Kkans: Kurkova-kolmogorov-arnold networks and their learning dynamics, Neural Networks 191 (2025) 107831.

[33] G. G. Lorentz, Metric entropy, widths, and superpositions of functions, The American Mathematical Monthly 69 (6) (1962) 469–485.

[34] M. Laczkovich, A superposition theorem of Kolmogorov type for bounded continuous functions, Journal of Approximation Theory 269 (2021) 105609.

[35] J. F. Urbán, P. Stefanou, J. A. Pons, Unveiling the optimization process of physics informed neural networks: How accurate and competitive can PINNs be?, Journal of Computational Physics 523 (2025) 113656.

[36] A. Jnini, E. Kiyani, K. Shukla, J. F. Urban, N. A. Daryakenari, J. Muller, M. Zeinhofer, G. E. Karniadakis, Curvature-aware optimization for high-accuracy physics-informed neural networks, arXiv preprint arXiv:2604.05230 (2026).

[37] J. D. Toscano, D. T. Chen, G. E. Karniadakis, ATHENA: agentic team for hierarchical evolutionary numerical algorithms, arXiv preprint arXiv:2512.03476 (2025).

[38] J. D. Toscano, Z. Chai, G. E. Karniadakis, GRAFT-ATHENA: self-improving agentic teams for autonomous discovery and evolutionary numerical algorithms, arXiv preprint arXiv:2605.11117 (2026).

[39] F. Girosi, T. Poggio, Representation properties of networks: Kolmogorov’s theorem is irrelevant, Neural Computation 1 (4) (1989) 465–469.

[40] V. Kůrková, Kolmogorov’s theorem and multilayer neural networks, Neural Networks 5 (3) (1992) 501–506.

[41] Q. Liu, J. Xu, R. Jiang, W. H. Wong, Density estimation using deep generative neural networks, Proceedings of the National Academy of Sciences 118 (15) (2021) e2101344118.

[42] O. Rippel, R. P. Adams, High-dimensional probability estimation with deep density models, arXiv preprint arXiv:1302.5125 (2013).

[43] G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, B. Lakshminarayanan, Normalizing flows for probabilistic modeling and inference, Journal of Machine Learning Research 22 (57) (2021) 1–64.

[44] N. Lüthen, S. Marelli, B. Sudret, Sparse Polynomial Chaos Expansions: Literature Survey and Benchmark, SIAM/ASA Journal on Uncertainty Quantification 9 (2) (2021) 593–649.

[45] N. Vyas, D. Morwani, R. Zhao, I. Shapira, D. Brandfonbrener, L. Janson, S. Kakade, SOAP: Improving and stabilizing Shampoo using Adam, in: The Thirteenth International Conference on Learning Representations (ICLR), 2025.

[46] P. Bratley, B. L. Fox, H. Niederreiter, Implementation and tests of low-discrepancy sequences, ACM Trans. Model. Comput. Simul. 2 (3) (1992) 195–213.

[47] J. E. Oakley, A. O’Hagan, Probabilistic sensitivity analysis of complex models: a Bayesian approach, Journal of the Royal Statistical Society Series B: Statistical Methodology 66 (3) (2004) 751–769.

[48] R. Mattey, S. Ghosh, A novel sequential method to train physics informed neural networks for Allen-Cahn and Cahn-Hilliard equations, Computer Methods in Applied Mechanics and Engineering 390 (2022) 114474.

[49] H. Jin, M. Mattheakis, P. Protopapas, Physics-informed neural networks for quantum eigenvalue problems, in: 2022 International Joint Conference on Neural Networks (IJCNN), IEEE, 2022, pp. 1–8.

[50] K. Fan, On a theorem of Weyl concerning eigenvalues of linear transformations I, Proceedings of the National Academy of Sciences 35 (11) (1949) 652–655.

[51] L. Song, Z. Cheng, J. D. Toscano, L.-L. Wang, Explicit construction of approximate Kolmogorov superpositions with C<sup>2</sup>-smoothness, arXiv preprint arXiv:2508.04392 (2025).

[52] A. Vergari, Y. Choi, A. Liu, S. Teso, G. Van den Broeck, A Compositional Atlas of Tractable Circuit Operations for Probabilistic Inference, in: Advances in Neural Information Processing Systems, Vol. 34, Curran Associates, Inc., 2021, pp. 13189–13201.

[53] B. Wang, D. D. Mauá, G. Van den Broeck, Y. Choi, A compositional atlas for algebraic circuits, Advances in Neural Information Processing Systems 37 (2024) 141318–141355.

[54] A. de Colnet, S. Mengel, A compilation of succinctness results for arithmetic circuits, in: 18th International Conference on Principles of Knowledge Representation and Reasoning, KR 2021, 2021.

[55] O. Delalleau, Y. Bengio, Shallow vs. deep sum-product networks, Advances in neural information processing systems 24 (2011).

[56] G. Cybenko, Approximation by superpositions of a sigmoidal function, Mathematics of Control, Signals and Systems 2 (4) (1989) 303–314.

[57] K. Hornik, M. Stinchcombe, H. White, Multilayer feedforward networks are universal approximators, Neural Networks 2 (5) (1989) 359–366.

[58] R. E. Caflisch, Monte Carlo and quasi-Monte Carlo methods, Acta Numerica 7 (1998) 1–49. doi:10.1017/S0962492900002804.

[59] J. Dick, F. Y. Kuo, I. H. Sloan, High-dimensional integration: the quasi-monte carlo way, Acta Numerica 22 (2013) 133–288.

[60] I. M. Sobol’, On the distribution of points in a cube and the approximate evaluation of integrals, USSR Computational Mathematics and Mathematical Physics 7 (4) (1967) 86–112. doi:10.1016/0041-5553(67)90144-9.

[61] A. B. Owen, Scrambled net variance for integrals of smooth functions, The Annals of Statistics 25 (4) (1997) 1541–1562. doi:10.1214/aos/1031594731.

[62] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S. J. van der Walt, M. Brett, J. Wilson, K. J. Millman, N. Mayorov, A. R. J. Nelson, E. Jones, R. Kern, E. Larson, C. J. Carey, İ. Polat, Y. Feng, E. W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E. A. Quintero, C. R. Harris, A. M. Archibald, A. H. Ribeiro, F. Pedregosa, P. van Mulbregt, SciPy 1.0 Contributors, SciPy 1.0: fundamental algorithms for scientific computing in Python, Nature Methods 17 (2020) 261–272. doi:10.1038/s41592-019-0686-2.

[63] J. D. Toscano, D. T. Chen, V. Ooomen, J. Darbon, G. E. Karniadakis, A variational framework for residual-based adaptivity in neural pde solvers and operator learning, NPJ Artificial Intelligence 2 (1) (2026) 32.

[64] X. Chen, C. Liang, D. Huang, E. Real, K. Wang, Y. Liu, H. Pham, X. Dong, T. Luong, C.-J. Hsieh, Y. Lu, Q. V. Le, Symbolic discovery of optimization algorithms, Advances in Neural Information Processing Systems 36 (2023) 49205–49233.

## Appendix A. Background

## Appendix A.1. Kolmogorov–Arnold Networks

KANs are neural architectures motivated by the Kolmogorov–Arnold representation theorem [2]. The theorem states that any continuous multivariate function on a hypercube can be represented through superpositions of continuous univariate functions. In contrast to conventional multilayer perceptrons, where nonlinear activation functions are associated with nodes, KANs place learnable univariate nonlinear functions on the edges of the computational graph [1].

In practical deep KAN architectures [1], the representation is implemented layer-wise as

$$
z _ { n } ^ { ( \ell + 1 ) } = \sum _ { n ^ { \prime } = 1 } ^ { N _ { \ell } } \phi _ { n , n ^ { \prime } } ^ { ( \ell ) } \big ( z _ { n ^ { \prime } } ^ { ( \ell ) } \big ) , \qquad z ^ { ( 0 ) } = \mathbf { x } ,\tag{A.1}
$$

where the functions $\phi _ { n , n ^ { \prime } } ^ { ( \ell ) } : \mathbb { R } \to \mathbb { R }$ are learnable univariate transformations associated with edges between layers. Here, $\ell = 0 , \ldots , L$ indexes layers, and n and $n ^ { \prime } = 1 , \ldots , N _ { \ell }$ index neurons in layers ℓ + 1 and ℓ, respectively. This stacked construction departs from the theorem that motivates it: the representation (1) is a two-block structure, with a single level of inner functions and a single level of outer functions, whereas deep KANs compose many such layers into a deeply nested architecture. The departure also carries practical costs. The original KANs parameterize the edge functions with learnable B-splines, which are computationally expensive, and their performance degrades as the parameter count grows. Subsequent variants replace the splines with more eficient bases such as Chebyshev polynomials, whose recursive formulation restores stability in single precision [10], or radial basis functions; broad reviews of these variants are given in [8, 9], and necessary and suficient conditions for the universality of KAN architectures are established in [7].

Kůrková–Kolmogorov–Arnold networks (KKANs) [32] were introduced to return to the two-block structure of the theorem. Exact Kolmogorov representations are unsuited to learning, since the inner functions of the constructive proofs are highly non-smooth and carry no adjustable parameters, although explicit constructions of approximate superpositions with C<sup>2</sup>-smooth inner functions have recently been obtained [51]; following Kůrková [40], KKANs sacrifice exactness and approximate both levels of the superposition with smooth learnable components, using robust multilayer perceptrons as inner functions and linear combinations of basis functions as outer functions. The resulting architecture is a universal approximator regardless of the choice of basis. Our construction builds on KKANs; the structural constraints that make statistical inference tractable are introduced in Section 3.1. These structural constraints lead to tractable inference with KANs generally; and work well in practice with KKANs specifically.

## Appendix A.2. Sum–Product Networks and Tractable Inference

Sum–Product Networks (SPNs), more generally referred to as probabilistic circuits or tractable circuits, are deep computational architectures that represent multivariate functions through recursive compositions of sums and products over structured computational graphs. Originally introduced for tractable probabilistic inference, SPNs represent probability distributions through hierarchical decompositions into mixtures (sum nodes) and factorizations (product nodes). More generally, circuits provide a unifying framework for tractable probabilistic models [28, 52] and structured function representations [53, 54].

Let ${ \mathcal { C } } \equiv ( { \mathcal { G } } , \mathbf { W } )$ denote a circuit with computational graph $\mathcal { G }$ and parameters W, representing a function $\mathcal C (  { \mathbf { x } } )$ over input variables $\mathbf { x } = \{ x _ { 1 } , \ldots , x _ { D } \}$ . Here, $\mathcal { G }$ is an acyclic directed graph with three types of nodes: input nodes, sum nodes, and product nodes. Let in(c) denote the set of children of some node $c \in { \mathcal { G } }$ . An input node c represents a parameterized function

$$
g _ { c } ( \mathbf { x } _ { c } )\tag{A.2}
$$

over a subset of variables $\mathbf { x } _ { c } \subseteq \mathbf { x } .$ , referred to as its scope. Internal nodes recursively combine their children through weighted sums or products. A sum node computes

$$
\sum _ { c ^ { \prime } \in \mathrm { i n } ( c ) } w _ { c , c ^ { \prime } } f _ { c ^ { \prime } } ( \mathbf { x } _ { c ^ { \prime } } ) ,\tag{A.3}
$$

whereas a product node computes

$$
\prod _ { c ^ { \prime } \in \mathrm { i n } ( c ) } f _ { c ^ { \prime } } ( \mathbf { x } _ { c ^ { \prime } } ) .\tag{A.4}
$$

The scope of an internal node is recursively defined as

$$
\mathbf { x } _ { c } = \bigcup _ { c ^ { \prime } \in \mathrm { i n } ( c ) } \mathbf { x } _ { c ^ { \prime } } .\tag{A.5}
$$

Probabilistic circuits can be interpreted as neural architectures with additive and multiplicative interactions arranged hierarchically through depth, where expressive nonlinearities are localized at the input functions while multiplicative structure governs variable interactions. It follows immediately from this composition that the expressiveness of SPNs multiplies with every product node, and hence scales exponentially with depth [55].

Formally, decomposability requires

$$
\mathbf { x } _ { c ^ { \prime } } \cap \mathbf { x } _ { c ^ { \prime \prime } } = \emptyset\tag{A.6}
$$

for all distinct children $c ^ { \prime } , c ^ { \prime \prime } \in \mathrm { i n } ( c )$ of every product node c. In many applications, one additionally assumes structured-decomposability, meaning that product nodes with identical scope decompose variables in the same manner throughout the network.

Often, circuits are additionally assumed to satisfy smoothness, meaning that all children of a sum node possess identical scope [28]. While smoothness is not required for tractable integration itself, it ensures that the circuit defines a valid probabilistic model with welldefined marginalization and conditioning operations.

In the next section, we show that polynomial outer nonlinearities induce decomposable multiplicative structure in KANs by transforming sums of univariate inner functions into recursively separable product expansions.

## Appendix B. Tractable Inference with Kolmogorov–Arnold Networks

## Appendix B.1. Building intuition with shallow minimal-order networks

For the quadratic expansion (4), assuming factorizing integration measures, it follows that

$$
\int f ( \mathbf { x } ) \mathrm { d } V _ { \mathbf { x } } = \sum _ { q = 1 } ^ { 2 D + 1 } \left( \sum _ { d = 1 } ^ { D } \int \phi _ { q , d } ^ { 2 } ( x _ { d } ) \mathrm { d } x _ { d } + 2 \sum _ { d < d ^ { \prime } } \left( \int \phi _ { q , d } ( x _ { d } ) \mathrm { d } x _ { d } \right) \left( \int \phi _ { q , d ^ { \prime } } ( x _ { d ^ { \prime } } ) \mathrm { d } x _ { d ^ { \prime } } \right) \right)\tag{B.1}
$$

Consequently, the Kolmogorov–Arnold representation acquires a decomposable multiplicative structure analogous to that appearing in sum–product networks and probabilistic circuits, and the D-dimensional integral decomposes into $\scriptstyle ( 2 D + 1 ) \left( D + { \frac { D ( D - 1 ) } { 2 } } \right)$ algebraic terms, assembled from 2D unique one-dimensional leaf moments per branch (the first and second moment of each leaf, with the cross terms reusing the first moments).

## Appendix B.2. Higher orders

Consider an outer activation function as polynomial of order P, i.e.

$$
\Phi _ { q } ( t ) : = \sum _ { p = 0 } ^ { P } c _ { p } t ^ { p }\tag{B.2}
$$

where $c _ { p }$ are coeficients and the order and the summation index can depend on q. For now, consider only the leading order. By the multinomial theorem we have

$$
\left( \sum _ { d = 1 } ^ { m } z _ { q , d } \right) ^ { P } = \sum _ { \begin{array} { c } { { k _ { q , d } \geq 0 } } \\ { { \sum _ { d = 1 } ^ { m } k _ { q , d } = P } } \end{array} } \binom { P } { k _ { q , 1 } , \ldots , k _ { q , m } } \prod _ { d = 1 } ^ { m } z _ { q , d } ^ { k _ { q , d } } .\tag{B.3}
$$

Identifying $z _ { d } \equiv \phi _ { q , d } ( x _ { d } )$ , each term again factorizes into products over disjoint variables, yielding decomposable structure and tractable integration under factorizing input measures.

If the outer activation is a quadratic polynomial, then we get products over pairs of univariate inner activation functions. If the outer activation is a cubic polynomial, then we get products over triples of univariate inner activation functions. More generally, we get sums over products $\begin{array} { r } { \prod _ { d \in I } \left( \phi _ { q , d } ( x _ { d } ) \right) ^ { k _ { q , d } } } \end{array}$ , where I is a node-specific index set (analogous to the children) and the cardinality of I is the order of the polynomial defining the outer activation function. The exponent $k _ { q , d }$ accounts for duplicatives, e.g. $k _ { q , d } = 2$ for the squared terms in the expanded form (4) (right hand side) and $k _ { q , d } = 1$ for the product of pairs for which $d ^ { \prime } \neq d .$ . If the input probability density function factorizes as

$$
\rho ( \mathbf { x } ) = \prod _ { d = 1 } ^ { D } \rho ( x _ { d } ) ,\tag{B.4}
$$

then the expectations over these products decompose as

$$
\int \prod _ { d \in I } \left( \phi _ { q , d } ( x _ { d } ) \right) ^ { k _ { q , d } } \rho ( \mathbf { x } ) \mathrm { d } V _ { \mathbf { x } } = \prod _ { d \in I } \int \left( \phi _ { q , d } ( x _ { d } ) \right) ^ { k _ { q , d } } \rho ( x _ { d } ) \mathrm { d } x _ { d } .\tag{B.5}
$$

Thus far, we have only considered the leading order $P .$ The process is repeated analogously for the trailing orders $P - 1 , P - 2 , . . . ,$ and distributivity of the integral with summation over the orders $p$ in (B.2), ${ \sum _ { p } ^ { P } }$ , is trivial.

In our implementation based on Kurkova-KANs, the inner activation functions are univariate multilayer perceptrons themselves, hence the one-dimensional integrals are computed numerically. In this case, inference remains tractable but becomes near-exact, rather than exact, due to the numerical error of the one-dimensional integrals. We will see in the experiments that this is a good compromise trading inference tractability against expressivity of the architecture. We note that these low-dimensional integrals, depending on the inference task, usually need to be computed only once.

## Appendix B.3. Deep layers

This principle immediately extends also to multiple layers. Consider a nested application of a polynomial outer activation as in a deep KAN as in (A.1). Define the outer activation function as in (B.2), however now the summation index $p$ additionally depends on the layer index $\ell .$ We refrain from overloading the notation further via e.g. $p _ { q , \ell }$ and instead demonstrate the principle. For simplicity, we first discuss the leading order $P$ at layer $\ell ,$ which we equip with a layer-index, i.e. $P _ { \ell } .$ . Note that the order can depend on the layer and the edge index, but for simplicity we assume that all edges in a given order are a polynomial of the same order. Then the leading order of layer ℓ + 1 in (A.1) becomes

$$
\sum _ { n _ { \ell } = 1 } ^ { N _ { \ell } } \bigl ( z _ { n _ { \ell } } ^ { ( \ell ) } \bigr ) ^ { P _ { \ell } } ,\tag{B.6}
$$

and the leading order of layer $\ell + 2$ becomes

$$
\sum _ { n _ { \ell + 1 } = 1 } ^ { N _ { \ell + 1 } } \left( \sum _ { n _ { \ell } = 1 } ^ { N _ { \ell } } \left( z _ { n _ { \ell } } ^ { ( \ell ) } \right) ^ { P _ { \ell } } \right) ^ { P _ { \ell + 1 } } = \sum _ { n _ { \ell + 1 } = 1 } ^ { N _ { \ell + 1 } } \sum _ { n _ { \ell } = 1 } ^ { N _ { \ell } } \left( z _ { n _ { \ell } } ^ { ( \ell ) } \right) ^ { P _ { \ell } P _ { \ell + 1 } } + \mathrm { t r a i l i n g ~ o r d e r s } .\tag{B.7}
$$

If we integrate this expression, by linearity of expectation and Fubini’s theorem under mild integrability conditions, we can exchange the order of summation and integration,

$$
\mathbb { E } \left[ \sum _ { n _ { \ell + 1 } = 1 } ^ { N _ { \ell + 1 } } \left( \sum _ { n _ { \ell } = 1 } ^ { N _ { \ell } } \big ( z _ { n _ { \ell } } ^ { ( \ell ) } \big ) ^ { P _ { \ell } } \right) ^ { P _ { \ell + 1 } } \right] = \sum _ { n _ { \ell + 1 } = 1 } ^ { N _ { \ell + 1 } } \sum _ { n _ { \ell } = 1 } ^ { N _ { \ell } } \mathbb { E } \left[ \big ( z _ { n _ { \ell } } ^ { ( \ell ) } \big ) ^ { P _ { \ell } P _ { \ell + 1 } } \right] + \mathrm { t r a i l i n g ~ o r d e r s } .\tag{B.8}
$$

We can then apply again the multinomial theorem (B.3) to the leading order above and reduce the expression as discussed in Appendix B.2. In addition, we can apply the multinomial theorem to all trailing orders in (B.7) since all orders have polynomial form.

## Appendix B.4. Implementation

Repeating the procedure across all orders and layers decomposes a TOUCKAN into low-dimensional components. We defer the complete formulation for arbitrary orders and layers to Appendix C. In practice, the KAN computational graph and the repeated binomial convolutions of Section 3.2 perform this bookkeeping.

We illustrate this informally as follows. Consider a definition of $f ( \mathbf { x } )$ via a composition operator $\mathcal { F } _ { \mathcal { C } , K }$ of the set of leaves $\{ \phi _ { q , d } ( x _ { d } ) \}$ , where the operator is defined by the composition implied by the circuit $\mathcal { C }$ and K the set of powers indicating the monomial orders after expansion:

$$
f ( \mathbf { x } ) = \mathcal { F } _ { \mathcal { C } , K } \big [ \{ \phi _ { q , d } ( x _ { d } ) \} \big ]\tag{B.9}
$$

Under factorizing priors, we write

$$
\begin{array} { r } { \mathbb { E } [ f ( \mathbf { x } ) ] = \mathbb { E } \left[ \mathcal { F } _ { \mathcal { C } , K } \left[ \left\{ \phi _ { q , d } ( x _ { d } ) \right\} \right] \right] \triangleq \mathcal { F } _ { \mathcal { C } , K } \left[ \left\{ \mathbb { E } \left[ \phi _ { q , d } ^ { k _ { q , d } } ( x _ { d } ) \right] \right\} \right] . } \end{array}\tag{B.10}
$$

In other words, the expectation of the composition of leaf functions can be written as the composition of the expectation of (powers of ) the leaf functions. Therefore, the circuit implementation replaces each leaf by the expectation of its required powers and then performs a forward pass through the circuit. Leaf replacement is a general property of probabilistic circuits [28], while TOUCKANs must additionally track the powers through the binomial recursion of Section 3.2. We verify (B.10) numerically in Section 4.2.

## Appendix B.5. Computational complexity scaling

For a shallow TOUCKAN with Q outer branches and polynomial degree $P ,$ the multinomial expansion of each outer activation produces

$$
T ( D , P ) = \sum _ { p = 0 } ^ { P } { \binom { D + p - 1 } { p } } = { \binom { D + P } { P } }
$$

separable monomial terms. Hence the total algebraic cost scales as $Q \left( \begin{array} { c } { { D + P } } \\ { { P } } \end{array} \right)$ , which is $\mathcal { O } ( Q D ^ { P } )$ for fixed P. In the classical KAN width $Q = 2 D + 1$ , this gives $\mathcal { O } ( \dot { D } ^ { \grave { P } + 1 } )$ . The onedimensional moment evaluations scale as $\mathcal { O } ( Q D P M )$ when M quadrature points are used per leaf moment if distinct leaves are used per branch. For leaves with analytic expressions of the integral, we have $M = 1$ . Therefore, direct quadrature with cost $\mathcal { O } ( M ^ { D } )$ is replaced by

$$
\mathcal { O } \biggl ( Q D P M + Q \biggl ( \begin{array} { c } { { D + P } } \\ { { P } } \end{array} \biggr ) \biggr ) ,
$$

which is polynomial in D for fixed polynomial degree. We note that the scaling above applies to the explicitly expanded shallow TOUCKAN representation. In deep TOUCKANs, the relevant complexity measure becomes the size of the induced decomposable computational graph generated through recursive multinomial expansion across layers. Accordingly, tractability refers to polynomial complexity in the size of this expanded representation.

The computational complexity of the proposed moment-propagation procedure depends on the number of decomposed monomial terms induced by the polynomial outer activations. For a polynomial order P and input dimension $D _ { \mathbf { \lambda } }$ , the multinomial expansion generates combinatorial sums over separable products, while each resulting integral factorizes into low-dimensional integrals over the individual variables. Most importantly, the inference procedure scales linearly with the number of quadrature points $M .$ . In contrast to direct quadrature, whose complexity typically scales exponentially with dimension, the TOUCKAN decomposition replaces high-dimensional integration by compositions of one-dimensional moment evaluations and algebraic operations through the computational graph. Consequently, the dominant computational cost shifts from high-dimensional sampling to the evaluation of the induced low-dimensional moments and the combinatorial expansion itself. In practice, this yields substantial computational savings in high-dimensional uncertainty propagation problems compared to brute-force Monte Carlo or dense quadrature methods.

For a deep TOUCKAN, the relevant complexity is the size of the expanded decomposable computational graph. Let $T _ { \ell }$ denote the number of separable monomial terms required to represent one node at layer ℓ after recursive multinomial expansion. For input dimension $D = N _ { 0 }$ , layer widths $N _ { \ell } .$ , and polynomial degrees $P _ { \ell } .$ the first layer satisfies

$$
T _ { 1 } = \binom { D + P _ { 1 } } { P _ { 1 } } .
$$

For deeper layers, each node receives $N _ { \ell - 1 }$ inputs, each of which is itself represented by $T _ { \ell - 1 }$ separable terms. Therefore, a degree- ${ \bf \nabla } \cdot P _ { \ell }$ polynomial expansion produces the upper bound

$$
T _ { \ell } \le \binom { N _ { \ell - 1 } T _ { \ell - 1 } + P _ { \ell } } { P _ { \ell } } , ~ \ell = 2 , \ldots , L .
$$

In the simple case of uniform width $N _ { \ell } = N$ and uniform polynomial degree $P _ { \ell } = P$ , this implies

$$
T _ { L } = \mathcal { O } \big ( N ^ { P + P ^ { 2 } + \cdots + P ^ { L - 1 } } D ^ { P ^ { L } } \big ) ,
$$

for fixed P. Thus, exact moment propagation is polynomial in the input dimension for fixed depth and fixed polynomial degree, but the degree of this polynomial grows exponentially with depth. This quantifies the sense in which deep TOUCKANs are tractable with respect to the expanded decomposable representation; uncontrolled increases in both depth and polynomial order can lead to rapid combinatorial growth.

## Appendix C. General tractable moment propagation in TOUCKANs

In this appendix, we formalize the tractable inference property of TOUCKANs for arbitrary depth and polynomial order.

## Appendix C.1. Expanded decomposable representation

Consider a TOUCKAN represented through a computational graph C with leaves $\phi _ { q , d } ( x _ { d } )$ and polynomial outer activations. After recursive multinomial expansion of all polynomial activations, the resulting function admits the representation

$$
f ( \mathbf { x } ) = \sum _ { m = 1 } ^ { M } c _ { m } \prod _ { d = 1 } ^ { D } \phi _ { m , d } ( x _ { d } ) ^ { k _ { m , d } } ,\tag{C.1}
$$

where $c _ { m } \in \mathbb { R }$ are coeficients, $k _ { m , d } \in  { \mathbb { N } } _ { 0 }$ are nonnegative integer exponents, and each factor depends only on a single variable $x _ { d }$

Equation (C.1) is the fully expanded decomposable representation induced by the recursive multinomial expansion of the TOUCKAN computational graph.

Appendix C.2. Moment propagation operator

Let $\rho ( \mathbf { x } )$ be a factorizing probability density of the form

$$
\rho ( \mathbf { x } ) = \prod _ { d = 1 } ^ { D } \rho _ { d } ( x _ { d } ) .\tag{C.2}
$$

Define the leaf moment

$$
\mu _ { m , d } ^ { ( k ) } = \int \phi _ { m , d } ( x _ { d } ) ^ { k } \rho _ { d } ( x _ { d } ) \mathrm { d } x _ { d } .\tag{C.3}
$$

The expectation of the TOUCKAN can then be written as

$$
\mathbb { E } [ f ( \mathbf { x } ) ] = \sum _ { m = 1 } ^ { M } c _ { m } \prod _ { d = 1 } ^ { D } \mu _ { m , d } ^ { ( k _ { m , d } ) } .\tag{C.4}
$$

## Appendix C.3. Main theorem

Theorem 1 (Tractable moment propagation in TOUCKANs). Consider a KAN with arbitrary depth and polynomial outer activations. Assume:

1. all outer activations are finite-degree polynomials,

2. all leaf functions are integrable and their powers are integrable,

3. the input density factorizes as in (C.2).

Then every statistical moment of the form

$$
\mathbb { E } [ h ( f ( \mathbf { x } ) ) ] = \int h ( f ( \mathbf { x } ) ) \rho ( \mathbf { x } ) \mathrm { d } V _ { \mathbf { x } } ,\tag{C.5}
$$

For polynomial h, this integral reduces exactly to one-dimensional leaf moments of the form (C.3). These moments are combined through a finite algebraic composition. Consequently, the high-dimensional integral requires only low-dimensional moment evaluations followed by a forward pass through the expanded decomposable computational graph.

Appendix C.4. Proof

Proof. We proceed by induction over the computational graph.

For the base case, the leaves depend only on single variables:

$$
g _ { d } ( x _ { d } ) = \phi _ { d } ( x _ { d } ) .\tag{C.6}
$$

Hence their moments are one-dimensional integrals by definition.

Now assume that all nodes up to layer ℓ admit decomposable representations of the form

$$
z ^ { ( \ell ) } = \sum _ { m } c _ { m } \prod _ { d } \phi _ { m , d } ( x _ { d } ) ^ { k _ { m , d } } .\tag{C.7}
$$

Consider a node at layer ℓ + 1 generated through a polynomial activation

$$
\Phi ( t ) = \sum _ { p = 0 } ^ { P } a _ { p } t ^ { p } .\tag{C.8}
$$

Substituting the inductive hypothesis yields powers of sums of decomposable monomials. Applying the multinomial theorem recursively produces again a finite sum of products of univariate functions,

$$
z ^ { ( \ell + 1 ) } = \sum _ { m ^ { \prime } } c _ { m ^ { \prime } } \prod _ { d } \phi _ { m ^ { \prime } , d } ( x _ { d } ) ^ { k _ { m ^ { \prime } , d } } .\tag{C.9}
$$

Therefore decomposability is preserved through every layer.

Since the density factorizes, Fubini’s theorem implies

$$
\int \prod _ { d } \phi _ { m , d } ( x _ { d } ) ^ { k _ { m , d } } \rho ( \mathbf { x } ) \mathrm { d } V _ { \mathbf { x } } = \prod _ { d } \int \phi _ { m , d } ( x _ { d } ) ^ { k _ { m , d } } \rho _ { d } ( x _ { d } ) \mathrm { d } x _ { d } .\tag{C.10}
$$

Thus every monomial term factorizes into one-dimensional moments.

Linearity of integration then yields

$$
\mathbb { E } [ f ( \mathbf { x } ) ] = \sum _ { m } c _ { m } \prod _ { d } \mu _ { m , d } ^ { ( k _ { m , d } ) } ,\tag{C.11}
$$

which proves the claim.

For a polynomial observable h, the composition h ◦ f is again a finite polynomial composition of the same leaves, so the preceding argument yields the general claim.

## Appendix D. Universal approximation of TOUCKANs

This appendix proves universal approximation for the complete TOUCKAN hierarchy by establishing density of its most restricted rank-one subclass. In this subclass, every dimension evaluates the same Q shared profiles and specializes them through scalars $\lambda _ { d }$ that are shared across branches. The argument follows the two-block approximation strategy of Kůrková [40], but is anchored at a superposition theorem of Laczkovich [34] whose inner data have the shared-profile structure of the TOUCKAN inner block and carry the smallest suficient branch count available for it.

Theorem 2 (Laczkovich [34], Theorem 1.1). Let $D \geq 2$ , let $m > ( 2 + \sqrt { 2 } ) D$ be an integer, and let $\lambda _ { 1 } ^ { * } , \ldots , \lambda _ { D } ^ { * }$ be distinct positive numbers. Then there exist continuous functions $\varphi _ { 1 } , \ldots , \varphi _ { m } \in C ( \mathbb { R } )$ with the following property: for every bounded $f \in C ( \mathbb { R } ^ { D } )$ there is a continuous function $g \in C ( \mathbb { R } )$ such that

$$
f ( \mathbf { x } ) = \sum _ { q = 1 } ^ { m } g \Big ( \sum _ { d = 1 } ^ { D } \lambda _ { d } ^ { * } \varphi _ { q } ( x _ { d } ) \Big ) \qquad f o r \ e v e r y \ \mathbf { x } \in \mathbb { R } ^ { D } .\tag{D.1}
$$

The constants $\lambda _ { d } ^ { * }$ and the profiles $\varphi _ { q }$ are universal. Only the outer function g depends on $f .$ We anchor the argument at Theorem 2 because it carries the smallest suficient branch count among the superposition theorems of [34]. Its constants are shared across branches, so the network constructed below lies in the rank-one subclass $\lambda _ { d , q } = \lambda _ { d }$ of the diagonal coupling (11). Density of the full diagonal class then follows a fortiori. No monotonicity property is needed in the argument below. Theorem 1.2 of [34] additionally provides monotone profiles at the larger branch count discussed in Remark 2. We now define the set of functions realizable by the TOUCKAN architecture.

Definition 1 (TOUCKAN approximator set). Let $\mathcal { A } _ { W }$ denote a class of shared inner networks $\pmb { \psi } = ( \psi _ { 1 } , \dots , \psi _ { Q } ) : [ 0 , 1 ]  \mathbb { R } ^ { Q }$ (in our implementation, basis-embedded MLPs of size W) such that $\cup _ { W } A _ { W }$ is dense in $C ( [ 0 , 1 ] ; \mathbb { R } ^ { Q } )$ under the uniform norm $\| \psi - \varphi \| _ { \infty } =$ $\begin{array} { r } { \operatorname* { m a x } _ { q \leq Q } \operatorname* { s u p } _ { x \in [ 0 , 1 ] } | \psi _ { q } ( x ) - \varphi _ { q } ( x ) | } \end{array}$ . Multi-output perceptrons have this property [56, 57], and the basis-embedded networks of our implementation inherit the property whenever the embedding is injective on [0, 1]. For $Q , P , W \in \mathbb { N }$ define

$$
\mathbb { T } _ { Q , P , W } = \Big \{ F ( \mathbf { x } ) = \sum _ { q = 1 } ^ { Q } \sum _ { p = 0 } ^ { P } C _ { q , p } T _ { p } ( \xi _ { q } ) , \quad \xi _ { q } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \operatorname { t a n h } \big ( \lambda _ { d , q } \psi _ { q } ( x _ { d } ) \big ) \Big \} ,\tag{D.2}
$$

where $\psi \in \mathcal { A } _ { W } , \lambda _ { d , q } \in \mathbb { R }$ , and $C _ { q , p } \in \mathbb { R }$ , exactly as in (7)–(11).

Theorem 3 (Universal approximation of TOUCKANs). Let $D \geq 2$ and let $Q \ge m _ { L } ( D )$ where $m _ { L } ( D )$ is the smallest integer exceeding $( 2 + { \sqrt { 2 } } ) D$ . Then $\textstyle \bigcup _ { P , W } \mathbb { T } _ { Q , P , W }$ is dense in $C ( E ^ { D } )$ with $\begin{array} { r } { E ^ { D } = [ 0 , 1 ] ^ { D } } \end{array}$ : for every $f \in C ( E ^ { D } )$ and every $\varepsilon > 0$ there exist $P , W \in \mathbb { N }$ and $F \in \mathbb { T } _ { Q , P , W }$ such that

$$
\| f - F \| _ { \infty } = \operatorname* { s u p } _ { \mathbf { x } \in E ^ { D } } | f ( \mathbf { x } ) - F ( \mathbf { x } ) | < \varepsilon .\tag{D.3}
$$

Remark 1 (Universality across the TOUCKAN hierarchy). Let $\mathcal { T } _ { \mathrm { r a n k } } , \ \mathcal { T } _ { \mathrm { d i a g } } , \ \mathcal { T } _ { \mathrm { f u l l } }$ , and $\mathcal { T } _ { \mathrm { i n d } }$ denote the rank-one, diagonal, full-head, and independent TOUCKAN classes, respectively. Their parameterizations give the nested function classes

$$
\mathcal { T } _ { \mathrm { r a n k } } \subseteq \mathcal { T } _ { \mathrm { d i a g } } \subseteq \mathcal { T } _ { \mathrm { f u l l } } \subseteq \mathcal { T } _ { \mathrm { i n d } } .\tag{D.4}
$$

The first inclusion follows by imposing $\lambda _ { d , q } = \lambda _ { d }$ The diagonal construction is recovered from the full-head formulation by restricting each head to a scaled shared map, while an independent inner network can reproduce any shared-trunk construction. The proof of Theorem 3 establishes density of $\mathcal { T } _ { \mathrm { r a n k } }$ in $C ( E ^ { D } )$ . Every class containing it is therefore also dense in $C ( E ^ { D } )$ . Thus, all four TOUCKAN parameterizations are universal approximators, and their diferences concern finite-parameter accuracy and eficiency rather than asymptotic expressivity.

Proof. Let $f \in C ( E ^ { D } )$ and $\varepsilon > 0$ . By the Tietze extension theorem, $f$ extends to a bounded continuous function on $\mathbb { R } ^ { D }$ with the same supremum norm, so Theorem 2 applies with $m = m _ { L } ( D ) \leq Q$ : there are positive constants $\lambda _ { d } ^ { * } ,$ continuous profiles $\varphi _ { q } \in C ( \mathbb { R } )$ , and $g \in C ( \mathbb { R } )$ such that, writing

$$
s _ { q } ( \mathbf { x } ) : = \sum _ { d = 1 } ^ { D } \lambda _ { d } ^ { * } \varphi _ { q } ( x _ { d } ) ,\tag{D.5}
$$

we have $\begin{array} { r } { f ( \mathbf { x } ) = \sum _ { q = 1 } ^ { m } g \big ( s _ { q } ( \mathbf { x } ) \big ) } \end{array}$ on $E ^ { D }$

Step 1 (compact ranges). Each $\varphi _ { q }$ is continuous, hence bounded on $[ 0 , 1 ] \colon$ : set $A : =$ $\mathrm { m a x } _ { q } \mathrm { m a x } _ { x \in [ 0 , 1 ] } | \varphi _ { q } ( x )$ | and $\Lambda : = \operatorname* { m a x } _ { d } \lambda _ { d } ^ { * }$ , so that $s _ { q } ( \mathbf { x } ) \in [ - B , B ]$ with $B : = D \Lambda A$ . The outer function g is uniformly continuous on $[ - B - 1 , B + 1 ]$ ; let ω denote its modulus of continuity there.

Step 2 (inner approximation). Define the padded target $\varphi : = ( \varphi _ { 1 } , \ldots , \varphi _ { m } , 0 , \ldots , 0 ) \in$ $C ( [ 0 , 1 ] ; \mathbb { R } ^ { Q } )$ , whose last $Q - m$ components vanish (these channels receive zero outer coeficients in Step 5). By the density assumption on $\mathcal { A } _ { W }$ there exists $\pmb { \psi } \in \mathcal { A } _ { W }$ with $\| \psi - \varphi \| _ { \infty } < \delta$ for a $\delta \in ( 0 , 1 ]$ fixed below; in particular $\begin{array} { r } { \operatorname* { m a x } _ { q \leq m } \operatorname* { m a x } _ { x \in [ 0 , 1 ] } | \psi _ { q } ( x ) - \varphi _ { q } ( x ) | < \delta } \end{array}$ and $| \psi _ { q } | \leq$ $A + 1 \mathrm { ~ o n ~ } [ 0 , 1 ]$

Step 3 (scalars and tanh linearization). For a scale $\textstyle \tau \in ( 0 , \frac { 1 } { \Lambda ( A + 1 ) } ]$ fixed below, set the trainable scalars to $\lambda _ { d , q } : = \tau \lambda _ { d } ^ { * }$ , independent of $q ,$ so the constructed network lies in the rank-one subclass $\lambda _ { d , q } = \lambda _ { d }$ . Each branch argument $u = \lambda _ { d , q } \psi _ { q } ( x _ { d } )$ then satisfies $| u | \leq$ $\tau \Lambda ( A + 1 ) \leq 1$ , and since | tanh $( u ) - u | \leq | u | ^ { 3 } / 3$

$$
\left| \xi _ { q } - \frac { \tau } { D } \sum _ { d = 1 } ^ { D } \lambda _ { d } ^ { * } \psi _ { q } ( x _ { d } ) \right| \leq \frac { \left( \tau \Lambda ( A + 1 ) \right) ^ { 3 } } { 3 } , \qquad \left| \frac { \tau } { D } \sum _ { d = 1 } ^ { D } \lambda _ { d } ^ { * } \psi _ { q } ( x _ { d } ) - \frac { \tau } { D } s _ { q } ( \mathbf { x } ) \right| \leq \tau \Lambda \delta ,\tag{D.6}
$$

so that $\begin{array} { r } { \left| \xi _ { q } - \frac { \tau } { D } s _ { q } ( \mathbf { x } ) \right| \le \rho : = \frac { 1 } { 3 } \tau ^ { 3 } \Lambda ^ { 3 } ( A + 1 ) ^ { 3 } + \tau \Lambda \delta } \end{array}$ . Note that $\xi _ { q } \in ( - 1 , 1 )$ always (the tanh bounds each branch and the average preserves the bound), and ${ \frac { \tau } { D } } s _ { q } ( { \bf x } ) \in [ - \tau \Lambda A , \tau \Lambda A ] \ \subset$ $( - 1 , 1 )$

Step 4 (rescaled outer function). Since $g \in C ( \mathbb { R } )$ , define $\tilde { g } ( t ) : = g ( D t / \tau )$ for $t \in [ - 1 , 1 ]$ so that $\begin{array} { r } { g ( s _ { q } ( { \bf x } ) ) = \tilde { g } \big ( \frac { \tau } { D } s _ { q } ( { \bf x } ) \big ) } \end{array}$ . Provided $D \rho / \tau \leq 1$ , Step 3 gives $| D \xi _ { q } / \tau - s _ { q } ( { \bf x } ) | \le D \rho / \tau \le 1$ since $s _ { q } ( \mathbf { x } ) \in [ - B , B ]$ , both arguments lie in $[ - B - 1 , B + 1 ]$ , where the modulus $\omega$ of Step 1 applies, and therefore

$$
\left| \tilde { g } ( \xi _ { q } ) - g ( s _ { q } ( { \bf x } ) ) \right| \le \omega \Big ( \frac { D \rho } { \tau } \Big ) = \omega \Big ( \frac { D \Lambda ^ { 3 } ( A + 1 ) ^ { 3 } } { 3 } \tau ^ { 2 } + D \Lambda \delta \Big ) .\tag{D.7}
$$

The argument of $\omega$ tends to zero as $\tau  0$ and $\delta \to 0$ . Indeed, the factor τ multiplying $\delta$ in $\rho$ cancels the division by $\tau .$ . We may therefore first fix $\tau$ and then $\delta$ so that $D \rho / \tau \leq 1$ and the right-hand side is below $\varepsilon / ( 3 m )$

Step 5 (outer approximation). By the Weierstrass approximation theorem there is a polynomial of some degree $P$ , written in the Chebyshev basis as $\begin{array} { r } { \Phi _ { q } ( t ) = \sum _ { p = 0 } ^ { P } C _ { q , p } T _ { p } ( t ) } \end{array}$ with m $\mathfrak { W } _ { t \in [ - 1 , 1 ] } | \Phi _ { q } ( t ) - \tilde { g } ( t ) | < \varepsilon / ( 3 m )$ ; for the unused branches $m < q \leq Q$ set $C _ { q , p } = 0$

Step 6 (assembly). The resulting $F \in \mathbb { T } _ { Q , P , W }$ satisfies, for every $\mathbf { x } \in E ^ { D }$

$$
\left| f ( \mathbf { x } ) - F ( \mathbf { x } ) \right| \leq \sum _ { q = 1 } ^ { m } { \Big [ } | g ( s _ { q } ( \mathbf { x } ) ) - \tilde { g } ( \xi _ { q } ) { \big ] } + \left| \tilde { g } ( \xi _ { q } ) - \Phi _ { q } ( \xi _ { q } ) \right| { \Big ] } < m \Big ( \frac { \varepsilon } { 3 m } + \frac { \varepsilon } { 3 m } \Big ) < \varepsilon ,\tag{D.8}
$$

where the second term uses $\xi _ { q } \in ( - 1 , 1 )$ . This completes the proof.

Remark 2. The branch count required by Theorem 3 is $Q \ge m _ { L } ( D )$ , the smallest integer exceeding $( 2 + \sqrt { 2 } ) D$ and asymptotically about 3.41 D. This count is inherited from Theorem 2, and Laczkovich notes that it is probably not sharp [34]. Our Kůrková-variant TOUCKAN architecture replaces the single outer function $g$ in (D.1) with one Chebyshev expansion $\Phi _ { q }$ per branch. Because this architecture contains the common-outer-function class as a special case, the density statement holds a fortiori. The branch counts used in the experiments were selected for each problem and are reported in Appendix I; these finite-budget choices do not test the sharpness of the suficient bound. Because the scalars constructed in Step 3 are shared across branches, the proof establishes density of the rank-one subclass $\left( \lambda _ { d , q } = \lambda _ { d } \right)$ . Universality therefore holds throughout the complete hierarchy, and the preference for the diagonal form in Section 4.1 rests on finite-budget accuracy rather than asymptotic expressivity. Finally, the same argument applies verbatim to Theorem 1.2 of [34]. Its steps are the Tietze extension, padded vector approximation, small-τ linearization, and Chebyshev outer approximation. In that theorem, the constants $\lambda _ { d , q } ^ { * }$ carry one value per dimension–branch pair and directly mirror the diagonal coupling (11), at the larger branch count $m > ( 2 + \sqrt { 2 } ) ( 2 D - 1 )$ , asymptotically about 6.83 D.

Remark 3. The universal approximation result is inherited from the underlying Kolmogorov– Arnold representation theorems and therefore applies to continuous functions. This setting is appropriate for the scientific machine learning applications considered here, where the quantities of interest are typically continuous or smooth fields. It is more restrictive than general machine learning settings that may contain discontinuous decision boundaries. This limitation is a general aspect of Kolmogorov–Arnold representations and is not unique to our KAN variants.

## Appendix E. Moment-propagation algorithms

This appendix derives, in full detail, the exact evaluation and propagation of the first and second moments of a TOUCKAN surrogate. Exactness refers to the propagation: every recursion below is algebraically exact in the leaf moments, which are themselves onedimensional integrals evaluated by M-node Gaussian quadrature. Thus, all computed moments are exact for the quadrature-induced discrete measure and approximate the continuum moments with spectral accuracy in M (cf. the error decomposition (15)). Appendix E.1 derives the first moment and concludes with Algorithm 1. Appendix E.2 derives the second moment and the variance and concludes with Algorithm 2. Throughout, the input follows the factorizing density $\begin{array} { r } { \rho ( \mathbf { x } ) = \prod _ { d = 1 } ^ { D } \rho _ { d } ( x _ { d } ) } \end{array}$ , so the coordinates $x _ { 1 } , \ldots , x _ { D }$ are mutually independent, and the surrogate is the TOUCKAN

$$
f ( { \bf x } ) = \sum _ { q = 1 } ^ { Q } \Phi _ { q } ( \xi _ { q } ) , \qquad \Phi _ { q } ( \xi ) = \sum _ { p = 0 } ^ { P } C _ { q , p } T _ { p } ( \xi ) , \qquad \xi _ { q } = \sum _ { d = 1 } ^ { D } \widetilde { \Psi } _ { d , q } ( x _ { d } ) ,\tag{E.1}
$$

where $\widetilde { \Psi } _ { d , q } = \Psi _ { d , q } / D$ are the normalized branches and $\begin{array} { r } { T _ { p } ( \xi ) = \sum _ { j = 0 } ^ { p } a _ { p , j } \xi ^ { j } } \end{array}$ is the Chebyshev polynomial of degree p in monomial form, with fixed integer coeficients $a _ { p , j }$

## Appendix E.1. Exact first moment

Reduction to scalar moments. The outer blocks are polynomial in the combination variables, so the first moment of the network reduces to scalar moments of each $\xi _ { q }$ . Applying linearity of expectation to (E.1) and substituting the monomial form of $T _ { p } ,$

$$
\mathbb { E } \big [ f ( \mathbf { x } ) \big ] = \sum _ { q = 1 } ^ { Q } \mathbb { E } \big [ \Phi _ { q } ( \xi _ { q } ) \big ] = \sum _ { q = 1 } ^ { Q } \sum _ { p = 0 } ^ { P } C _ { q , p } \mathbb { E } \big [ T _ { p } ( \xi _ { q } ) \big ] = \sum _ { q = 1 } ^ { Q } \sum _ { p = 0 } ^ { P } C _ { q , p } \sum _ { j = 0 } ^ { p } a _ { p , j } \mathbb { E } \big [ \xi _ { q } ^ { j } \big ] ,\tag{E.2}
$$

The remaining task is therefore to compute the moments $m _ { q , p } = \mathbb { E } [ \xi _ { q } ^ { p } ]$ for $p = 0 , \ldots , P .$

Branch isolation. The computation of these moments requires the individual branches. In practice, however, all branches are computed at the same time: the inner block is a single fused map $\Psi \colon ( x _ { 1 } , \ldots , x _ { D } ) \mapsto \pmb { \xi } \in \mathbb { R } ^ { Q }$ , and no individual branch is evaluated separately. Because each branch depends only on its own coordinate, the contribution of dimension d can nevertheless be recovered by evaluating the fused map with every other slot held at a fixed reference point, chosen as the origin. Define

$$
\varphi _ { d } ( x ) = \Psi ( 0 , \ldots , 0 , \underbrace { x } _ { \mathrm { s l o t } d } , 0 , \ldots , 0 ) , \qquad \mathbf { K } = \Psi ( 0 , \ldots , 0 ) ,\tag{E.3}
$$

both vectors in $\mathbb { R } ^ { Q }$ . The evaluation $\varphi _ { d } ( x )$ contains the branch of dimension d together with the constant contribution of the $D - 1$ zeroed branches. Summing over d therefore counts this constant D times instead of once, which yields the structural identity

$$
\pmb { \xi } ( \mathbf { x } ) = \sum _ { d = 1 } ^ { D } \left[ \pmb { \varphi } _ { d } ( x _ { d } ) - \mathbf { K } \right] + \mathbf { K } .\tag{E.4}
$$

The identity follows from the additive structure of the combination layer alone and involves no approximation. In particular, it holds for any inner block whose branches are univariate, and therefore for all variants used in this paper.

Centering. The recursion is not run on the raw evaluations $\varphi _ { d }$ , because they share the common ofset K and this ofset destroys the computation in high dimension. The failure is quantified at the end of this subsection. Instead, define the centered branches

$$
\delta _ { d } ( x ) = \varphi _ { d } ( x ) - { \bf K } , \qquad \delta _ { d } ( 0 ) = { \bf 0 } .\tag{E.5}
$$

Componentwise, $\delta _ { d , q } ( x ) = \widetilde { \Psi } _ { d , q } ( x ) - \widetilde { \Psi } _ { d , q } ( 0 )$ , so each centered branch is the branch itself with its value at the origin removed. With this notation the isolation identity (E.4) becomes

$$
\pmb { \xi } ( \mathbf { x } ) = \mathbf { S } ( \mathbf { x } ) + \mathbf { K } , \qquad \mathbf { S } ( \mathbf { x } ) = \sum _ { d = 1 } ^ { D } \pmb { \delta } _ { d } ( x _ { d } ) .\tag{E.6}
$$

The remaining derivation follows this decomposition. The moments of the centered sum $S _ { q }$ are computed first, and the deterministic shift by $K _ { q }$ is reintroduced at the end.

Centered leaf moments. For each dimension and each branch, define the centered leaf moments

$$
{ \widehat { \mu } } _ { d , q } ^ { ( k ) } = \mathbb { E } \left[ \delta _ { d , q } ( x _ { d } ) ^ { k } \right] = \int \delta _ { d , q } ( x _ { d } ) ^ { k } \rho _ { d } ( x _ { d } ) d x _ { d } , \qquad k = 0 , \ldots , P .\tag{E.7}
$$

Each $\widehat { \mu } _ { d , q } ^ { ( k ) }$ is a one-dimensional integral. It is evaluated by a Gaussian quadrature rule with M nodes whose weight function matches the marginal $\rho _ { d } .$ , namely Gauss–Legendre for uniform inputs and Gauss–Hermite for Gaussian inputs. This quadrature is the only discretization in the entire procedure; everything downstream is algebraically exact in the computed leaf moments and is evaluated in double precision. The integrand follows from (E.3) and (E.5): the fused inner block is evaluated at the M nodes of each dimension, and the baseline K is subtracted. This stage costs DM evaluations of the inner block and O(QDPM) operations for the moment sums.

Binomial recursion. The moments of $S _ { q }$ are assembled one dimension at a time. Define the centered partial sums

$$
S _ { j , q } = \sum _ { d = 1 } ^ { j } \delta _ { d , q } ( x _ { d } ) , \qquad S _ { D , q } = S _ { q } .\tag{E.8}
$$

The partial sum $S _ { j - 1 , q }$ depends only on $( x _ { 1 } , \ldots , x _ { j - 1 } )$ , while the next branch $\delta _ { j , q } ( x _ { j } )$ depends only on $x _ { j }$ . Under a factorizing density, the two are therefore independent random variables. Raising the sum to the power p with the binomial theorem and using this independence,

$$
\begin{array} { l } { { \mathbb { E } \big [ S _ { j , q } ^ { p } \big ] = \mathbb { E } \Big [ \big ( S _ { j - 1 , q } + \delta _ { j , q } ( x _ { j } ) \big ) ^ { p } \Big ] = \displaystyle \sum _ { k = 0 } ^ { p } \binom { p } { k } \mathbb { E } \big [ S _ { j - 1 , q } ^ { p - k } \delta _ { j , q } ^ { k } ( x _ { j } ) \big ] } } \\ { { \displaystyle \qquad = \sum _ { k = 0 } ^ { p } \binom { p } { k } \mathbb { E } \big [ S _ { j - 1 , q } ^ { p - k } \big ] \widehat \mu _ { j , q } ^ { ( k ) } } . } \end{array}\tag{E.9}
$$

The first line uses the binomial theorem and linearity of expectation. The second line uses independence: the expectation of a product of independent variables is the product of their expectations. The result is a dynamic program over the input dimensions. It is initialized at the first dimension, $\mathbb { E } [ S _ { 1 , q } ^ { p } ] = \widehat { \mu } _ { 1 , q } ^ { ( p ) }$ , and swept forward for $j = 2 , \dots , D$ , carrying only the $P + 1$ moments $s _ { q , p } = \mathbb { E } [ S _ { j , q } ^ { p } ]$ of the current partial sum. The combinatorial growth of a direct multinomial expansion is thereby replaced by repeated binomial convolutions, at a total cost of $\mathcal { O } ( Q D P ^ { 2 } )$ .

K-shift and outer contraction. Because K is a deterministic constant, the moments of $\xi _ { q } =$ $S _ { q } + K _ { q }$ follow from the centered moments through one further binomial expansion,

$$
m _ { q , p } = \mathbb { E } \left[ \xi _ { q } ^ { p } \right] = \sum _ { j = 0 } ^ { p } { \binom { p } { j } } K _ { q } ^ { p - j } S _ { q , j } .\tag{E.10}
$$

Substituting the shifted moments into the reduction (E.2) yields the exact first moment,

$$
\mathbb { E } \big [ f ( \mathbf { x } ) \big ] = \sum _ { q = 1 } ^ { Q } \sum _ { p = 0 } ^ { P } C _ { q , p } \sum _ { j = 0 } ^ { p } a _ { p , j } m _ { q , j } .\tag{E.11}
$$

The shift and the contraction add $\mathcal { O } ( Q P ^ { 2 } )$ operations, independent of $D$

Numerical stability. We evaluate the recursion in a centered form, which keeps every intermediate quantity bounded: each centered increment entering the dynamic programs is bounded by $2 / D$ , since $| \widetilde { \Psi } _ { d , q } | \le 1 / D$ , and the partial sums remain bounded by 2. The common ofset K re-enters only through the well-conditioned shift (E.10) and its bivariate analogue (E.16). This centered evaluation reproduces the exact variance to a relative error of order $1 0 ^ { - 8 }$ at every dimension tested through $D = 5 0$ (Table 2: $1 . 1 \times 1 0 ^ { - 8 } , 9 . 3 \times 1 0 ^ { - 9 }$ $3 . 8 \times 1 0 ^ { - 8 } , 1 . 7 \times 1 0 ^ { - 8 } \mathrm { ~ a t ~ } D = 5 , 1 2 , 2 5 , 5 0 )$ . All moment evaluations are performed in double precision. The complete procedure is summarized in Algorithm 1.

Algorithm 1 Exact first moment of a TOUCKAN surrogate.   
Require: fused inner block Ψ; outer coeficients $\overline { { C _ { q , p } ; } }$ Chebyshev monomial coeficients $a _ { p , j } ;$   
quadrature rules $\{ ( x _ { d } ^ { ( i ) } , w _ { d } ^ { ( i ) } ) \} _ { i = 1 } ^ { M }$ matched to the marginals $\rho _ { d }$   
Ensure: $\mathbb { E } [ f ( \mathbf { x } ) ]$   
1: $\mathbf { K } \gets \Psi ( 0 , \dots , 0 )$ ▷ baseline (E.3)   
2: for $d = 1 , \ldots , D ; \ i = 1 , \ldots , M$ do ▷ isolated, centered branches (E.5)   
3: $\pmb { \delta } _ { d } ^ { ( i ) }  \Psi \big ( 0 , \mathrm { \ldots } , 0 , x _ { d } ^ { ( i ) } , 0 , \mathrm { \ldots } , 0 \big ) - \mathbf { K }$ ▷ node in slot d   
4: end for   
5: for $d = 1 , \ldots , D ; q = 1 , \ldots , Q ; k = 0 , \ldots , P$ do ▷ centered leaf moments (E.7)   
6: $\begin{array} { r } { \widehat { \mu } _ { d , q } ^ { ( k ) }  \sum _ { i = 1 } ^ { M } w _ { d } ^ { ( i ) } ( \delta _ { d , q } ^ { ( i ) } ) ^ { k } } \end{array}$ ▷ 1D quadrature, $\mathcal { O } ( Q D P M )$ total   
7: end for   
8: for $q = 1 , \ldots , Q$ do ▷ binomial recursion (E.9), $\mathcal { O } ( Q D P ^ { 2 } )$ total   
9: $s _ { q , p } \gets \widehat { \mu } _ { 1 , q } ^ { ( p ) }$ for $p = 0 , \ldots , P$ ▷ initialize with the first dimension   
10: for $d = 2 , \ldots , D$ do   
11: $\begin{array} { r } { s _ { q , p } \gets \sum _ { k = 0 } ^ { p } \binom { p } { k } s _ { q , p - k } \widehat { \mu } _ { d , q } ^ { ( k ) } } \end{array}$ for $p = P , \ldots , 0$ ▷ in place, descending in p   
12: end for   
13: end for   
14: $\begin{array} { r } { m _ { q , p } \xleftarrow { } \sum _ { j = 0 } ^ { p } \binom { p } { j } K _ { q } ^ { p - j } s _ { q , j } } \end{array}$ for all $q , p$ ▷ K-shift (E.10)   
15: return $\sum ^ { Q } \sum ^ { P } C _ { q , p } \sum ^ { p } a _ { p , j } m _ { q , j }$ ▷ outer contraction (E.11)   
$q = 1 \mathrm { ~ } p = 0 \qquad j = 0$

Appendix E.2. Exact second moment and variance

The variance requires the second moment, since $\operatorname { V a r } [ f ] = \mathbb { E } [ f ^ { 2 } ] - ( \mathbb { E } [ f ] ) ^ { 2 }$ with the first moment computed in Appendix E.1. The derivation repeats that of the first moment for pairs of combination variables. Branch isolation and centering carry over unchanged, so the derivation below starts from the centered branches $\delta _ { d }$ of (E.5) and from the baseline K of (E.3).

Monomial collapse. For the second moment, the outer blocks are first collapsed into a single polynomial per branch. Substituting the monomial form of $T _ { p }$ into the outer expansion of (E.1) and collecting powers gives

$$
\Phi _ { q } ( \xi _ { q } ) = \sum _ { j = 0 } ^ { P } \beta _ { q , j } \xi _ { q } ^ { j } , \qquad \beta _ { q , j } = \sum _ { p = j } ^ { P } C _ { q , p } a _ { p , j } .\tag{E.12}
$$

The full surrogate is then an ordinary polynomial in the combination variables, $f ( \mathbf { x } ) =$ $\textstyle \sum _ { q } \sum _ { j } \beta _ { q , j } \xi _ { q } ^ { j }$

Reduction to cross-moments. Squaring this polynomial and taking the expectation termwise gives

$$
\mathbb { E } \big [ f ^ { 2 } ( \mathbf { x } ) \big ] = \sum _ { q , q ^ { \prime } = 1 } ^ { Q } \sum _ { j , j ^ { \prime } = 0 } ^ { P } \beta _ { q , j } \beta _ { q ^ { \prime } , j ^ { \prime } } m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) } , \qquad m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) } = \mathbb { E } \big [ \xi _ { q } ^ { j } \xi _ { q ^ { \prime } } ^ { j ^ { \prime } } \big ] .\tag{E.13}
$$

The cross-moments $m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) }$ do not factor into the moments of Appendix E.1. Although the coordinates are independent, the combination variables $\xi _ { q }$ and $\xi _ { q ^ { \prime } }$ are built from the same coordinates and are therefore dependent random variables. The recursion must consequently track pairs.

Joint centered leaf moments. For each dimension and each pair of branches, define

$$
{ \widehat { \nu } } _ { d , q , q ^ { \prime } } ^ { ( k , k ^ { \prime } ) } = \mathbb { E } \big [ \delta _ { d , q } ( x _ { d } ) ^ { k } \delta _ { d , q ^ { \prime } } ( x _ { d } ) ^ { k ^ { \prime } } \big ] , \qquad k , k ^ { \prime } = 0 , \dots , P .\tag{E.14}
$$

Both factors depend on the same scalar coordinate, so each $\widehat { \nu } _ { d , q , q ^ { \prime } } ^ { ( k , k ^ { \prime } ) }$ remains a one-dimensional integral. It is evaluated with the same measure-matched quadrature and the same centered branch evaluations as the marginal leaf moments (E.7), so no new inner-block evaluations are needed. The first moments are recovered as the special case $\widehat { \nu } _ { d , q , q ^ { \prime } } ^ { ( k , 0 ) } = \widehat { \mu } _ { d , q } ^ { ( k ) }$ , and the symmetry under $( q , k )  ( q ^ { \prime } , k ^ { \prime } )$ restricts the computation to the pairs $q \leq q ^ { \prime }$ . The cost is $\mathcal { O } ( Q ^ { 2 } D P ^ { 2 } M )$

Bivariate recursion. Fix a pair $( q , q ^ { \prime } )$ and recall the centered partial sums (E.8). The pair $\left( S _ { j - 1 , q } , \ S _ { j - 1 , q ^ { \prime } } \right)$ depends only on $( x _ { 1 } , \ldots , x _ { j - 1 } )$ , while the branch pair $\left( \delta _ { j , q } ( x _ { j } ) , \delta _ { j , q ^ { \prime } } ( x _ { j } ) \right)$ depends only on $x _ { j }$ . Under a factorizing density, the two groups are therefore independent of each other, even though the variables within each group are dependent. This is the reason the recursion must carry joint quantities rather than only univariate ones. Expanding both powers with the binomial theorem and factorizing the expectation across the two groups,

$$
\begin{array} { r l } & { \mathbb { E } \big [ S _ { j , q } ^ { p } S _ { j , q ^ { \prime } } ^ { p ^ { \prime } } \big ] = \mathbb { E } \Big [ \big ( S _ { j - 1 , q } + \delta _ { j , q } ( x _ { j } ) \big ) ^ { p } \big ( S _ { j - 1 , q ^ { \prime } } + \delta _ { j , q ^ { \prime } } ( x _ { j } ) \big ) ^ { p ^ { \prime } } \Big ] } \\ & { \qquad = \displaystyle \sum _ { a = 0 } ^ { p } \sum _ { b = 0 } ^ { p ^ { \prime } } \binom { p } { a } \binom { p ^ { \prime } } { b } \mathbb { E } \big [ S _ { j - 1 , q } ^ { p - a } S _ { j - 1 , q ^ { \prime } } ^ { p ^ { \prime } - b } \big ] \widehat { \nu } _ { j , q , q ^ { \prime } } ^ { ( a , b ) } , } \end{array}\tag{E.15}
$$

Setting $p ^ { \prime } = 0$ recovers the marginal recursion (E.9). The recursion is initialized with $\mathbb { E } [ S _ { 1 , q } ^ { p } S _ { 1 , q ^ { \prime } } ^ { p ^ { \prime } } ] = \widehat \nu _ { 1 , q , q ^ { \prime } } ^ { ( p , p ^ { \prime } ) }$ and swept forward for $j = 2 , \dots , D$ , carrying the $( P + 1 ) ^ { 2 }$ joint moments $c _ { q , q ^ { \prime } } ^ { ( \bar { p } , p ^ { \prime } ) } = \mathbb { E } [ S _ { j , q } ^ { p } S _ { j , q ^ { \prime } } ^ { p ^ { \prime } } ]$ of the current partial-sum pair, at a total cost of $\mathcal { O } ( Q ^ { 2 } D P ^ { 4 } )$ .

Bivariate K-shift. Because $K _ { q }$ and $K _ { q ^ { \prime } }$ are deterministic constants, the cross-moments of $( \xi _ { q } , \xi _ { q ^ { \prime } } ) = ( S _ { q } + K _ { q } , S _ { q ^ { \prime } } + K _ { q ^ { \prime } } )$ follow from the centered joint moments by expanding both binomials,

$$
m _ { q , q ^ { \prime } } ^ { ( p , p ^ { \prime } ) } = \mathbb { E } \left[ \xi _ { q } ^ { p } \xi _ { q ^ { \prime } } ^ { p ^ { \prime } } \right] = \sum _ { a = 0 } ^ { p } \sum _ { b = 0 } ^ { p ^ { \prime } } { \binom { p } { a } } { \binom { p ^ { \prime } } { b } } K _ { q } ^ { p - a } K _ { q ^ { \prime } } ^ { p ^ { \prime } - b } c _ { q , q ^ { \prime } } ^ { ( a , b ) } .\tag{E.16}
$$

The shift adds $\mathcal { O } ( Q ^ { 2 } P ^ { 4 } )$ operations, independent of $D .$

Assembly. Substituting the shifted cross-moments into the reduction (E.13) gives the second moment. Subtracting the squared first moment (E.11) gives the variance,

$$
\mathrm { V a r } \big [ f ( \mathbf { x } ) \big ] = \sum _ { q , q ^ { \prime } = 1 } ^ { Q } \sum _ { j , j ^ { \prime } = 0 } ^ { P } \beta _ { q , j } \beta _ { q ^ { \prime } , j ^ { \prime } } m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) } - \Big ( \mathbb { E } \big [ f ( \mathbf { x } ) \big ] \Big ) ^ { 2 } .\tag{E.17}
$$

The numerical-stability considerations of Appendix E.1 apply unchanged. The complete procedure is summarized in Algorithm 2.

Algorithm 2 Exact variance of a TOUCKAN surrogate.   
Require: fused inner block Ψ; outer coeficients $\overline { { C _ { q , p } ; } }$ Chebyshev monomial coeficients   
$a _ { p , j } ;$ quadrature rules $\{ ( x _ { d } ^ { ( i ) } , w _ { d } ^ { ( i ) } ) \} _ { i = 1 } ^ { M }$ matched to the marginals $\rho _ { d } ;$ baseline K, centered   
branch evaluations $\delta _ { d } ^ { ( i ) }$ , and mean $\mathbb { E } [ f ]$ from Algorithm 1   
Ensure: $\mathrm { V a r } [ f ( \mathbf { x } ) ]$   
1: $\begin{array} { r } { \beta _ { q , j }  \sum _ { p = j } ^ { P } C _ { q , p } a _ { p , j } } \end{array}$ for all $q , j$ ▷ monomial collapse (E.12)   
2: for $d = 1 , \ldots , D ;$ pairs $q \le q ^ { \prime } ; k , k ^ { \prime } = 0 , . . . , P$ do ▷ joint centered leaf moments (E.14)   
3: $\begin{array} { r } { \widehat { \nu } _ { d , q , q ^ { \prime } } ^ { ( k , k ^ { \prime } ) }  \sum _ { i = 1 } ^ { M } \overline { { w } } _ { d } ^ { ( i ) } ( \delta _ { d , q } ^ { ( i ) } ) ^ { k } ( \delta _ { d , q ^ { \prime } } ^ { ( i ) } ) ^ { k ^ { \prime } } } \end{array}$ ▷ 1D quadrature, $\mathcal { O } ( Q ^ { 2 } D P ^ { 2 } M )$ total   
4: end for   
5: for pairs $q \leq q ^ { \prime }$ do ▷ bivariate recursion (E.15), $\mathcal { O } ( Q ^ { 2 } D P ^ { 4 } )$ total   
6: $c _ { \boldsymbol { q } , \boldsymbol { q } ^ { \prime } } ^ { ( p , p ^ { \prime } ) } \gets \widehat { \nu } _ { 1 , \boldsymbol { q } , \boldsymbol { q } ^ { \prime } } ^ { ( p , p ^ { \prime } ) }$ for $p , p ^ { \prime } = 0 , \ldots , P$   
7: for $d = 2 , \ldots , D$ do   
8: $\begin{array} { r } { c _ { \boldsymbol { q } , \boldsymbol { q } ^ { \prime } } ^ { ( p , p ^ { \prime } ) } \gets \sum _ { a = 0 } ^ { p } \sum _ { b = 0 } ^ { p ^ { \prime } } \binom { p } { a } \binom { p ^ { \prime } } { b } c _ { \boldsymbol { q } , \boldsymbol { q } ^ { \prime } } ^ { ( p - a , p ^ { \prime } - b ) } \widehat \nu _ { \boldsymbol { d } , \boldsymbol { q } , \boldsymbol { q } ^ { \prime } } ^ { ( a , b ) } } \end{array}$ for $p , p ^ { \prime } = P , \ldots , 0$ ▷ in place,   
descending   
9: end for   
10: $\begin{array} { r } { m _ { q , q ^ { \prime } } ^ { ( p , p ^ { \prime } ) } \xleftarrow { } \sum _ { a = 0 } ^ { p } \sum _ { b = 0 } ^ { p ^ { \prime } } \binom { p } { a } \binom { p ^ { \prime } } { b } K _ { q } ^ { p - a } K _ { q ^ { \prime } } ^ { p ^ { \prime } - b } c _ { q , q ^ { \prime } } ^ { ( a , b ) } } \end{array}$ ▷ K-shift (E.16)   
11: $m _ { q ^ { \prime } , q } ^ { ( p ^ { \prime } , p ) } \gets m _ { q , q ^ { \prime } } ^ { ( p , p ^ { \prime } ) }$ ▷ remaining pairs by symmetry   
12: end for   
13: $\begin{array} { r } { \mathbb E [ f ^ { 2 } ]  \sum _ { q , q ^ { \prime } = 1 } ^ { Q } \sum _ { j , j ^ { \prime } = 0 } ^ { P } \beta _ { q , j } \beta _ { q ^ { \prime } , j ^ { \prime } } m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) } } \end{array}$ ▷ assemble (E.13)   
14: return $\mathbb { E } \left[ f ^ { 2 } \right] - \left( \mathbb { E } \left[ f \right] \right) ^ { 2 }$ ▷ exact variance (E.17)

## Appendix F. Sensitivity analysis

Every expectation reported in this work is assembled from one-dimensional leaf moments evaluated by M-node Gaussian quadrature, so the order M is the single accuracy parameter of the inference stage (cf. the error decomposition (15)). This appendix measures the sensitivity of the computed moments to M for one representative of each problem family: the Lüthen benchmark at $D = 5 0$ (uniform inputs, Gauss–Legendre) and the Schrödinger Gram matrix at $D = 5$ (Gaussian inputs, probabilists’ Gauss–Hermite). We reuse the trained checkpoints of Section 4 without any retraining. Errors are measured against true references rather than against self-convergence: the exact closed-form mean and variance of the Lüthen function, and an independent $4 8 ^ { 5 }$ -node tensor-product Gauss–Hermite integration of the trained surrogate (roughly $2 . 5 \times 1 0 ^ { 8 }$ nodes) for the Schrödinger Gram matrix.

As shown in Fig. F.7, all quantities governed by quadrature truncation converge spectrally, as predicted. At $D = 5 0$ the Lüthen mean and variance errors decay rapidly and saturate from $M \approx 1 2$ onward at $1 . 6 3 \times 1 0 ^ { - 1 0 }$ and $1 . 6 6 \times 1 0 ^ { - 8 }$ (the values reported in Table 2). Beyond this point, the first (training) term of (15) dominates and additional nodes have no efect. The Schrödinger Gram error decays from $2 . 7 \times 1 0 ^ { - 1 }$ at $M = 4$ to a floor near $1 0 ^ { - 8 }$ for $M \geq 4 8$ . The production order $M = 1 2 8$ carries a true relative error of $1 . 7 \times 1 0 ^ { - 8 }$ in the Frobenius norm. The production orders of these representative cases therefore sit well inside the saturated regime, where the leaf-moment quadrature contributes negligibly to the reported errors.

![](images/b37d5e90f476ad2c59bf75a3716f24d7d7e15549329e9ed308d406e8d16c1300.jpg)

![](images/b9f11d05aefe98397eb92b99f2dc2051720115abf403c95c100a538a8dc637a3.jpg)  
Figure F.7: Sensitivity of the computed moments to the one-dimensional quadrature order M. (a) Lüthen function at $D = 5 0$ : relative error of the recursion mean and variance against the exact closed-form references. (b) Schrödinger Gram matrix at $D = 5 \mathrm { : }$ relative error against an independent $4 8 ^ { 5 }$ tensor-product Gauss– Hermite integration of the same surrogate. Dashed vertical lines mark the production orders $( M = 2 0 0$ and $M = 1 2 8 )$ used for the examples in the main paper.

Conservation under quadrature refinement. The same refinement analysis applies to the conserved mass of the Cahn–Hilliard experiment (Section 4.3). There, the constraint is imposed by the reprojection (23) against a 512-node midpoint rule, so the machine-precision drift in Table 3 is, by itself, the discrete identity evaluated against the enforcing functional. To probe the continuum quantity independently, we re-evaluate the trained constrained solutions on midpoint rules from $N = 8$ up to $N = 8 1 9 2$ nodes (Figure F.8a). Every rule other than the $N = 5 1 2$ production rule is independent of the enforcement, so the convergence trend across them validates the continuum value independently of the enforcing functional. Because the spatial branch enters only through the periodic Fourier embedding, the integrand is smooth and periodic and the midpoint rule converges spectrally, but non-monotonically: the error at N nodes is the sum of the integrand’s Fourier modes that alias onto the mean $( k = N , 2 N , \ldots )$ , so it oscillates with N as diferent modes fall in and out of resonance. The worst-case drift falls from $1 . 0 \times 1 0 ^ { - 2 }$ at $N = 8$ to the double-precision floor $( \sim 5 \times 1 0 ^ { - 1 5 } )$ by $N \approx 1 2 8$ , beyond which the integrand carries no resolvable spectral content and refining further moves the result only at the level of floating-point rounding. The production rule $N = 5 1 2$ thus sits far inside the converged regime. We confirm the mechanism quantitatively: at a fixed time the measured midpoint error matches the sum of the aliased Fourier modes, evaluated from an independent FFT of the field, to within $1 \%$ at every N. The moment-recursion mass and the direct nodal sum agree to ${ \sim } 1 0 ^ { - 1 6 }$ throughout. The 512- node enforcement therefore coincides with the continuum mass to machine precision: mass conservation is a property of the solution, not an artifact of the functional used to impose it.

(a) Continuum mass drift vs. N  
![](images/49012bd91a198e14bbfacd720e96a518d6f0a7d9662cf6d51006c46de66b9bd5.jpg)

(b) Mass drift over time, by rule  
![](images/8ba6c2535a2241c116cde1e36faaf726db599f04e5e7f6c356a7e624c6dc3729.jpg)  
Figure F.8: Continuum mass conservation of the Cahn–Hilliard surrogate under quadrature refinement, independent of the 512-node rule used for enforcement. (a) Worst-case mass drift max<sub>t</sub> $| \int \widetilde { f } d x - \mathcal { Q } _ { 0 } |$ of the constrained TOUCKAN as the evaluation grid is refined from $N = 8$ to $N = 8 1 9 2$ midpoint nodes. The decay is spectral but non-monotone, the signature of periodic-quadrature aliasing: the error at N nodes is the sum of the integrand’s Fourier modes with $k = N , 2 N , \ldots .$ It reaches a converged plateau near $5 \times 1 0 ^ { - 1 5 }$ by $N \approx 1 2 8$ , approximately one order above machine epsilon (dotted). The production rule $N = 5 1 2$ (dashed) lies far inside the converged regime. (b) Mass drift $| \int \widetilde { f } d x - \mathcal { Q } _ { 0 } |$ over time $t ,$ evaluated on four independent midpoint rules $( N = 6 4 , 1 2 8 , 5 1 2 , 8 1 9 2 )$ . The under-resolved $N = 6 4$ rule stays above the floor, while $N = 1 2 8$ , the production rule $N = 5 1 2 .$ , and $N = 8 1 9 2$ coincide at the double-precision floor at every time instance, so conservation holds pointwise in t and independently of the evaluation rule.

## Appendix G. Randomized quasi-Monte Carlo comparison

The analytical benchmarks also permit direct integration by randomized quasi-Monte Carlo (RQMC), which can converge much faster than Monte Carlo on smooth integrands [58, 59]. We use Sobol’ sequences [60] with Owen’s nested scrambling [61], as implemented in SciPy [62], and 200 independent scramblings for each case. Each run reports the sample mean and sample variance. Gaussian inputs are obtained through the standard normal quantile function. Table G.5 reports results for complete digital nets with $N = 2 ^ { 1 8 } = 2 6 2$ ,144 points alongside the TOUCKAN and MC results. Table G.6 compares this budget with $N = 2 { \times } 1 0 ^ { 5 }$ matching the number of evaluations in each MC run. The latter truncates the digital net and loses its balance, increasing the median errors by up to two orders of magnitude on the smooth Lüthen function.

Table G.5: Relative errors against the exact mean and variance for three benchmarks across $D \in$ {5, 12, 25, 50}. The surrogate column reports the test relative L error. TOUCKAN evaluates moments of the trained surrogate by deterministic propagation. MC and RQMC instead evaluate the true integrand and report the median error over 200 independent replications. Each MC run uses $N = 2 \times 1 0 ^ { 5 }$ pseudorandom samples, while each RQMC run uses a complete Owen-scrambled Sobol’ net with $N = 2 ^ { 1 8 } = 2 6 2$ ,144 points. Thus, the TOUCKAN errors include the discrepancy between the surrogate and the target, whereas the sampling errors arise from direct integration of the target. TOUCKAN has lower errors than the median MC estimate in every case and than the median RQMC estimate in most cases. The variance entries marked with ‡ are limited by floating-point roundof in the second-moment recursion rather than by the surrogate.
<table><tr><td colspan="3">Surrogate</td><td colspan="3">E[f] rel. error</td><td colspan="3">Var[f] rel. error</td></tr><tr><td>Benchmark D</td><td></td><td>rel. L2</td><td>TOUCKAN</td><td> $\mathrm { M C } \ ( \mathrm { m e d i a n } )$ </td><td> $\mathrm { R Q M C } \ \mathrm { ( m e d i a n ) }$ </td><td>TOUCKAN MC (median) RQMC (median)</td><td></td><td></td></tr><tr><td rowspan="5">Lüthen</td><td>5</td><td> $\overline { { 1 . 0 \times 1 0 ^ { - 6 } } }$ </td><td> $5 . 6 4 \times 1 0 ^ { - 1 0 }$ </td><td> $\overline { { 1 . 3 4 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 2 . 5 1 \times 1 0 ^ { - 9 } } }$ </td><td> $\overline { { 1 . 1 3 \times 1 0 ^ { - 8 } } }$ </td><td> $\overline { { 2 . 1 0 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 3 . 8 1 \times 1 0 ^ { - 6 } } }$ </td></tr><tr><td>12</td><td> $2 . 7 \times 1 0 ^ { - 6 }$ </td><td> $1 . 3 0 \times 1 0 ^ { - 9 }$ </td><td> $3 . 0 7 \times 1 0 ^ { - 4 }$ </td><td> $8 . 4 6 \times 1 0 ^ { - 1 0 }$ </td><td> $9 . 2 7 \times 1 0 ^ { - 9 }$ </td><td> $2 . 0 9 \times 1 0 ^ { - 3 }$ </td><td> $3 . 8 1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>25</td><td> $4 . 5 \times 1 0 ^ { - 6 }$ </td><td> $1 . 7 7 \times 1 0 ^ { - 9 }$ </td><td> $2 . 5 3 \times 1 0 ^ { - 4 }$ </td><td> $8 . 2 8 \times 1 0 ^ { - 1 0 }$ </td><td> $3 . 8 1 \times 1 0 ^ { - 8 }$ </td><td> $2 . 3 5 \times 1 0 ^ { - 3 }$ </td><td> $3 . 8 1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>50</td><td> $8 . 9 \times 1 0 ^ { - 6 }$ </td><td> $1 . 6 3 \times 1 0 ^ { - 1 0 }$ </td><td> $1 . 1 4 \times 1 0 ^ { - 4 }$ </td><td> $5 . 4 6 \times 1 0 ^ { - 1 0 }$ </td><td> $1 . 6 6 \times 1 0 ^ { - 8 }$ </td><td> $1 . 9 2 \times 1 0 ^ { - 3 }$ </td><td> $3 . 9 7 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>5</td><td> $\overline { { 2 . 9 \times 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 7 . 2 2 \times 1 0 ^ { - 7 } } }$ </td><td> $\overline { { 1 . 0 3 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 9 . 6 5 \times 1 0 ^ { - 1 0 } } }$ </td><td> $\overline { { 1 . 8 0 \times 1 0 ^ { - 6 } } }$ </td><td> $\overline { { 1 . 7 4 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 3 . 8 0 \times 1 0 ^ { - 6 } } }$ </td></tr><tr><td rowspan="4"></td><td>12</td><td> $1 . 7 \times 1 0 ^ { - 3 }$ </td><td>1.55 × 10−7</td><td> $1 . 0 4 \times 1 0 ^ { - 3 }$ </td><td> $1 . 9 1 \times 1 0 ^ { - 6 }$ </td><td> $1 . 4 5 \times 1 0 ^ { - 5 }$ </td><td> $1 . 9 4 \times 1 0 ^ { - 3 }$ </td><td> $9 . 9 6 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>25</td><td> $4 . 3 \times 1 0 ^ { - 3 }$ </td><td> $3 . 5 4 \times 1 0 ^ { - 7 }$ </td><td> $9 . 8 3 \times 1 0 ^ { - 4 }$ </td><td> $1 . 7 4 \times 1 0 ^ { - 6 }$ </td><td> $9 . 1 1 \times 1 0 ^ { - 6 }$ </td><td> $1 . 7 5 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>50</td><td> $1 . 1 \times 1 0 ^ { - 2 }$ </td><td> $1 . 3 1 \times 1 0 ^ { - 5 }$ </td><td> $1 . 0 8 \times 1 0 ^ { - 3 }$ </td><td> $2 . 0 2 \times 1 0 ^ { - 6 }$ </td><td> $4 . 5 9 \times 1 0 ^ { - 4 \frac { 1 } { 4 } }$ </td><td> $1 . 7 7 \times 1 0 ^ { - 3 }$ </td><td> $8 . 2 4 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>5</td><td> $\overline { { 4 . 7 \times 1 0 ^ { - 5 } } }$ </td><td> $\overline { { 2 . 6 8 \times 1 0 ^ { - 7 } } }$ </td><td> $\overline { { 2 . 3 3 \times 1 0 ^ { - 3 } } }$ </td><td> $2 . 5 5 \times 1 0 ^ { - 6 }$ </td><td> $\overline { { 1 . 9 0 \times 1 0 ^ { - 7 } } }$ </td><td> $\overline { { 2 . 2 5 \times 1 0 ^ { - 3 } } }$ </td><td> $\overline { { 1 . 7 9 \times 1 0 ^ { - 5 } } }$ </td></tr><tr><td rowspan="3">Oakley</td><td>12</td><td> $1 . 4 \times 1 0 ^ { - 4 }$ </td><td> $1 . 3 5 \times 1 0 ^ { - 7 }$ </td><td> $1 . 4 5 \times 1 0 ^ { - 3 }$ </td><td> $2 . 0 4 \times 1 0 ^ { - 6 }$ </td><td> $3 . 4 5 \times 1 0 ^ { - 7 }$ </td><td> $2 . 1 7 \times 1 0 ^ { - 3 }$ </td><td> $7 . 0 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>25</td><td> $2 . 3 \times 1 0 ^ { - 4 }$ </td><td> $4 . 7 1 \times 1 0 ^ { - 7 }$ </td><td> $9 . 2 4 \times 1 0 ^ { - 4 }$ </td><td> $2 . 3 1 \times 1 0 ^ { - 6 }$ </td><td> $1 . 0 2 \times 1 0 ^ { - 6 }$ </td><td> $2 . 1 3 \times 1 0 ^ { - 3 }$ </td><td> $1 . 4 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>50</td><td> $1 . 1 \times 1 0 ^ { - 3 }$ </td><td> $8 . 5 0 \times 1 0 ^ { - 7 }$ </td><td> $7 . 9 9 \times 1 0 ^ { - 4 }$ </td><td> $2 . 6 5 \times 1 0 ^ { - 6 }$ </td><td> $2 . 6 9 \times 1 0 ^ { - 4 \frac { 1 } { 4 } }$ </td><td> $1 . 8 6 \times 1 0 ^ { - 3 }$ </td><td> $2 . 6 7 \times 1 0 ^ { - 4 }$ </td></tr></table>

RQMC substantially reduces the sampling errors relative to MC on these benchmarks. At $N = 2 ^ { 1 8 }$ , its median mean errors are of order $1 0 ^ { - 9 }$ or smaller for Lüthen and $1 0 ^ { - 6 }$ or smaller for Bratley and Oakley–O’Hagan, while its median variance errors range from $4 \times 1 0 ^ { - 6 }$ to $3 \times 1 0 ^ { - 4 }$ . TOUCKAN has lower errors in most cases. The exceptions are the Lüthen means at $D = 1 2$ and 25, the Bratley means at D = 5 and 50 and variances at $D = 1 2$ and 50, and the Oakley–O’Hagan variance at $D = 5 0$ . The Lüthen means are already accurate to 10<sup>−9</sup> with both methods, while the two Oakley–O’Hagan variance errors are both approximately $2 . 7 \times 1 0 ^ { - 4 }$ . At the matched budget, TOUCKAN has lower errors in 22 of the 24 cases, with both Bratley moments at $D = 5 0$ as the exceptions.

Table G.6: Median relative errors of RQMC over 200 scramblings at the matched budget $N = 2 \times 1 0 ^ { 5 }$ and at $N = 2 ^ { 1 8 }$ , for the true integrands of Table G.5.
<table><tr><td colspan="4">E[f] rel. error</td><td colspan="2">Var[f] rel. error</td></tr><tr><td>Benchmark</td><td>D</td><td> $N = 2 \times \mathrm { 1 0 ^ { 5 } }$ </td><td> $N = 2 ^ { 1 8 }$ </td><td> $N = 2 \times \mathrm { 1 0 ^ { 5 } }$ </td><td> $N = 2 ^ { 1 8 }$ </td></tr><tr><td rowspan="4">Lüthen</td><td>5</td><td> $5 . 3 1 \times 1 0 ^ { - 7 }$ </td><td> $\overline { { 2 . 5 1 \times 1 0 ^ { - 9 } } }$ </td><td> $6 . 4 4 \times 1 0 ^ { - 6 }$ </td><td> $\overline { { 3 . 8 1 \times 1 0 ^ { - 6 } } }$ </td></tr><tr><td>12</td><td> $1 . 7 4 \times 1 0 ^ { - 7 }$ </td><td> $8 . 4 6 \times 1 0 ^ { - 1 0 }$ </td><td> $2 . 5 6 \times 1 0 ^ { - 5 }$ </td><td> $3 . 8 1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>25</td><td> $7 . 7 9 \times 1 0 ^ { - 8 }$ </td><td> $8 . 2 8 \times 1 0 ^ { - 1 0 }$ </td><td> $3 . 3 0 \times 1 0 ^ { - 5 }$ </td><td> $3 . 8 1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>50</td><td> $4 . 6 2 \times 1 0 ^ { - 8 }$ </td><td> $5 . 4 6 \times 1 0 ^ { - 1 0 }$ </td><td> $5 . 2 6 \times 1 0 ^ { - 5 }$ </td><td> $3 . 9 7 \times 1 0 ^ { - 6 }$ </td></tr><tr><td rowspan="4">Bratley</td><td>5</td><td> $\overline { { 2 . 2 2 \times 1 0 ^ { - 6 } } }$ </td><td> $9 . 6 5 \times 1 0 ^ { - 1 0 }$ </td><td> $\overline { { 1 . 3 3 \times 1 0 ^ { - 5 } } }$ </td><td> $\overline { { 3 . 8 0 \times 1 0 ^ { - 6 } } }$ </td></tr><tr><td>12</td><td> $4 . 3 9 \times 1 0 ^ { - 6 }$ </td><td> $1 . 9 1 \times 1 0 ^ { - 6 }$ </td><td> $2 . 0 9 \times 1 0 ^ { - 5 }$ </td><td> $9 . 9 6 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>25</td><td> $4 . 6 5 \times 1 0 ^ { - 6 }$ </td><td> $1 . 7 4 \times 1 0 ^ { - 6 }$ </td><td> $2 . 2 8 \times 1 0 ^ { - 5 }$ </td><td> $1 . 0 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>50</td><td> $4 . 4 4 \times 1 0 ^ { - 6 }$ </td><td> $2 . 0 2 \times 1 0 ^ { - 6 }$ </td><td> $2 . 3 3 \times 1 0 ^ { - 5 }$ </td><td> $8 . 2 4 \times 1 0 ^ { - 6 }$ </td></tr><tr><td rowspan="4">Oakley</td><td>5</td><td> $\overline { { 7 . 1 0 \times 1 0 ^ { - 6 } } }$ </td><td> $\overline { { 2 . 5 5 \times 1 0 ^ { - 6 } } }$ </td><td> $\overline { { 4 . 6 1 \times 1 0 ^ { - 5 } } }$ </td><td> $\overline { { 1 . 7 9 \times 1 0 ^ { - 5 } } }$ </td></tr><tr><td>12</td><td> $5 . 9 1 \times 1 0 ^ { - 6 }$ </td><td> $2 . 0 4 \times 1 0 ^ { - 6 }$ </td><td> $9 . 6 9 \times 1 0 ^ { - 5 }$ </td><td> $7 . 0 9 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>25</td><td> $6 . 4 5 \times 1 0 ^ { - 6 }$ </td><td> $2 . 3 1 \times 1 0 ^ { - 6 }$ </td><td> $2 . 2 1 \times 1 0 ^ { - 4 }$ </td><td> $1 . 4 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>50</td><td> $7 . 9 3 \times 1 0 ^ { - 6 }$ </td><td> $2 . 6 5 \times 1 0 ^ { - 6 }$ </td><td> $4 . 4 2 \times 1 0 ^ { - 4 }$ </td><td> $2 . 6 7 \times 1 0 ^ { - 4 }$ </td></tr></table>

These results concern diferent computational settings. MC and RQMC evaluate the true integrand, whereas TOUCKAN evaluates moments of an already-trained surrogate. Direct sampling is useful when new function evaluations are afordable. When only fixed observations or expensive simulation data are available, repeated surrogate moments and integral constraints can instead be evaluated without new calls to the original model. The analytical benchmarks have inexpensive integrands, so the measured sampling times do not represent the cost of generating data from an expensive simulation. At the same sample count, one RQMC replicate costs 0.83–0.98 times one MC replicate for the uniform benchmarks and approximately 1.55 times for the Gaussian benchmark, where the quantile transform dominates. As in the main timing comparison, surrogate training is excluded, and TOUCKAN times refer to warm evaluations with one-time compilation excluded. These timings therefore describe moment evaluation after surrogate construction rather than the total cost of solving a new integration problem.

## Appendix H. Problem descriptions

This appendix defines the three benchmark problems used in our experiments. All three share the structural properties that make exact reference moments available: the input coordinates are independent, and the function is built from sums and products of univariate terms, so every moment factorizes into one-dimensional integrals that are available in closed form or computable to machine precision by one-dimensional Gaussian quadrature. The reference means and variances below are therefore exact, not Monte Carlo estimates.

Appendix H.1. Lüthen function (uniform inputs)

The first problem is the high-dimensional polynomial-chaos benchmark of Lüthen et al. [44],

$$
\begin{array} { l } { { \displaystyle f ( { \bf x } ) = 3 - \frac { 5 } { D } \sum _ { d = 1 } ^ { D } d x _ { d } + \frac { 1 } { D } \sum _ { d = 1 } ^ { D } d x _ { d } ^ { 3 } + \frac { 1 } { 3 D } \sum _ { d = 1 } ^ { D } d \ln \left( x _ { d } ^ { 2 } + x _ { d } ^ { 4 } \right) } } \\ { { \displaystyle ~ + \frac { 1 } { D \geq 5 } \left( x _ { 1 } x _ { 2 } ^ { 2 } + x _ { 2 } x _ { 4 } - x _ { 3 } x _ { 5 } \right) + \mathbf { 1 } _ { D \geq 5 1 } x _ { 5 1 } + \mathbf { 1 } _ { D \geq 5 4 } x _ { 5 0 } x _ { 5 4 } ^ { 2 } } , } \end{array}\tag{H.1}
$$

with inputs

$$
x _ { d } \sim \mathcal { U } ( 1 , 2 ) \quad \forall d \neq 2 0 , \qquad x _ { 2 0 } \sim \mathcal { U } ( 1 , 3 ) .\tag{H.2}
$$

The interaction terms use fixed indices, so the indicators $\mathbf { 1 } _ { D \geq 5 } , \mathbf { 1 } _ { D \geq 5 1 }$ , and ${ \bf 1 } _ { D \geq 5 4 }$ define the function at every dimension: the first three interaction terms are present in all experiments reported here $\left( D \geq 5 \right)$ , while the terms $x _ { 5 1 }$ and $x _ { 5 0 } x _ { 5 4 } ^ { 2 }$ contribute only once $D \geq 5 1$ and $D \geq 5 4$ , respectively. The same indicators apply to the reference moments below. By independence and linearity, the expectation separates term by term,

$$
\begin{array} { r l } & { \mathbb { E } [ f ] = 3 - \displaystyle \frac { 5 } { D } \sum _ { d = 1 } ^ { D } d \mathbb { E } [ x _ { d } ] + \displaystyle \frac { 1 } { D } \sum _ { d = 1 } ^ { D } d \mathbb { E } [ x _ { d } ^ { 3 } ] + \displaystyle \frac { 1 } { 3 D } \sum _ { d = 1 } ^ { D } d \mathbb { E } \big [ \ln ( x _ { d } ^ { 2 } + x _ { d } ^ { 4 } ) \big ] } \\ & { \qquad + \ : { \bf 1 } _ { D \geq 5 } \left( \mathbb { E } [ x _ { 1 } ] \mathbb { E } [ x _ { 2 } ^ { 2 } ] + \mathbb { E } [ x _ { 2 } ] \mathbb { E } [ x _ { 4 } ] - \mathbb { E } [ x _ { 3 } ] \mathbb { E } [ x _ { 5 } ] \right) + { \bf 1 } _ { D \geq 5 1 } \mathbb { E } [ x _ { 5 1 } ] + { \bf 1 } _ { D \geq 5 4 } \mathbb { E } [ x _ { 5 0 } ] \mathbb { E } [ x _ { 5 4 } ^ { 2 } ] , } \end{array}\tag{H.3}
$$

where every factor is a one-dimensional integral over $\mathcal { U } ( 1 , 2 )$ (or $\mathcal { U } ( 1 , 3 )$ for $x _ { 2 0 } )$ available in closed form, including the logarithmic term. The exact variance follows in the same way: expanding $f ^ { 2 }$ and applying independence reduces $\mathbb { E } [ f ^ { 2 } ]$ to a finite combination of products of one-dimensional moments, and $\operatorname { V a r } [ f ] = \mathbb { E } [ f ^ { 2 } ] \overset { \cdot } { - } ( \mathbb { E } [ f ] ) ^ { 2 }$ In practice we evaluate all one-dimensional integrals to machine precision with 200-node Gauss–Legendre quadrature. Table H.7 lists the resulting reference moments for every dimension reported in Section 4.

Table H.7: Exact reference moments of the Lüthen function (H.1) for every reported input dimension, in physical units.
<table><tr><td>D</td><td>5</td><td>12</td><td>20</td><td>25</td><td>50</td></tr><tr><td>E[f]</td><td>-2.80818</td><td>-13.66773</td><td>-22.02031</td><td>-30.58880</td><td>-70.99621</td></tr><tr><td> $\textstyle \operatorname { V a r } [ f ]$ </td><td>6.39955</td><td>7.55664</td><td>34.18982</td><td>26.33687</td><td>20.07893</td></tr></table>

## Appendix H.2. Bratley function (uniform inputs)

The second problem is the alternating sum of nested products of Bratley et al.,

$$
f ( { \bf x } ) = \sum _ { d = 1 } ^ { D } ( - 1 ) ^ { d } \prod _ { j = 1 } ^ { d } x _ { j } , \qquad { \bf x } \sim \mathcal { U } [ 0 , 1 ] ^ { D } .\tag{H.4}
$$

Since the coordinates are independent with $\mathbb { E } [ x _ { j } ] ~ = ~ \frac { 1 } { 2 }$ , each nested product has mean $\mathbb { E } \big [ \prod _ { j = 1 } ^ { d } x _ { j } \big ] = 2 ^ { - d }$ , and the exact mean is the geometric sum

$$
\mathbb { E } [ f ] = \sum _ { d = 1 } ^ { D } \left( - { \frac { 1 } { 2 } } \right) ^ { d } = - { \frac { 1 } { 3 } } \left( 1 - \left( - { \frac { 1 } { 2 } } \right) ^ { D } \right) \ { \xrightarrow { D \to \infty } } \ - { \frac { 1 } { 3 } } .\tag{H.5}
$$

For the second moment, the product of two nested products squares the shared coordinates and leaves the remaining ones linear; with $\begin{array} { r } { \mathbb { E } [ x _ { j } ^ { 2 } ] = \frac { 1 } { 3 } } \end{array}$ 2

$$
\mathbb { E } \left[ \prod _ { j = 1 } ^ { d } x _ { j } \prod _ { j = 1 } ^ { d ^ { \prime } } x _ { j } \right] = \left( \frac { 1 } { 3 } \right) ^ { \operatorname* { m i n } ( d , d ^ { \prime } ) } \left( \frac { 1 } { 2 } \right) ^ { | d - d ^ { \prime } | } ,\tag{H.6}
$$

so the exact variance is the finite double sum

$$
\mathrm { V a r } [ f ] = \sum _ { d = 1 } ^ { D } \sum _ { d ^ { \prime } = 1 } ^ { D } ( - 1 ) ^ { d + d ^ { \prime } } \left( { \textstyle { \frac { 1 } { 3 } } } \right) ^ { \operatorname* { m i n } ( d , d ^ { \prime } ) } \left( { \textstyle { \frac { 1 } { 2 } } } \right) ^ { \left| { d - d ^ { \prime } } \right| } - \left( \mathbb { E } [ f ] \right) ^ { 2 } .\tag{H.7}
$$

Appendix H.3. Oakley–O’Hagan function (Gaussian inputs)

The third problem is an Oakley–O’Hagan-type function with standard Gaussian inputs,

$$
f ( \mathbf { x } ) = \sum _ { d = 1 } ^ { D } ( a _ { 1 , d } x _ { d } + a _ { 2 , d } \sin x _ { d } + a _ { 3 , d } \cos x _ { d } ) + \mathbf { x } ^ { \top } M \mathbf { x } , \qquad \mathbf { x } \sim \mathcal { N } ( 0 , I _ { D } ) .\tag{H.8}
$$

The coeficients are drawn once from a fixed seed (NumPy RandomState(7)) and then held fixed. Their realized values will be reproducible from the code repository released upon acceptance: $a _ { 1 , d } , a _ { 2 , d } , a _ { 3 , d } \sim \mathcal { U } ( 0 . 5 , 1 . 5 ) / \sqrt { D }$ , and $M \ \in \ \mathbb { R } ^ { D \times D }$ is symmetric with diagonal entries $M _ { d d } \sim \mathcal { U } ( - 0 . 5 , 0 . 5 ) / \sqrt { D }$ and of-diagonal entries $M _ { d d ^ { \prime } } = S _ { d d ^ { \prime } } / 2$ , where $S _ { d d ^ { \prime } } \sim$ $\mathcal { U } ( - 1 . 2 / D , 1 . 2 / D )$ . This scaling keeps the mean and the variance of $f$ of order one at every dimension D. Grouping the univariate terms as $g _ { d } ( x ) = a _ { 1 , d } x + a _ { 2 , d }$ sin $x + a _ { 3 , d }$ cos $x + M _ { d d } x ^ { 2 }$ the standard Gaussian identities E[sin $X ] = 0$ , E[cos $X ] = \mathbb { E } [ X$ sin $X ] = e ^ { - 1 / 2 }$ , E[X cos $X ] =$ $\mathbb { E } [ X ^ { 2 } \sin X ] = \mathbb { E } [ X ^ { 2 } \cos X ] = 0 , \mathbb { E } [ \bar { X } ^ { 2 } ] = \bar { 1 }$ , and $\mathbb { E } [ X ^ { 4 } ] = 3$ give the exact mean

$$
\mathbb { E } [ f ] = \sum _ { d = 1 } ^ { D } \left( a _ { 3 , d } e ^ { - 1 / 2 } + M _ { d d } \right) .\tag{H.9}
$$

The ANOVA decomposition of (H.8) truncates exactly at order two: the main efects are the centered $g _ { d } .$ , and the only interactions are the pair terms $S _ { d d ^ { \prime } } x _ { d } x _ { d ^ { \prime } }$ with $S _ { d d ^ { \prime } } = 2 M _ { d d ^ { \prime } }$ for $d < d ^ { \prime }$ . The exact variance is therefore

$$
\operatorname { V a r } [ f ] = \sum _ { d = 1 } ^ { D } \left( \mathbb { E } [ g _ { d } ^ { 2 } ] - \mathbb { E } [ g _ { d } ] ^ { 2 } \right) + \sum _ { d < d ^ { \prime } } \left( 2 M _ { d d ^ { \prime } } \right) ^ { 2 } ,\tag{H.10}
$$

with $\mathbb { E } [ g _ { d } ] = a _ { 3 , d } e ^ { - 1 / 2 } + M _ { d d }$ and

$$
\begin{array} { r } { \mathbb { E } [ g _ { d } ^ { 2 } ] = a _ { 1 , d } ^ { 2 } + \frac { 1 } { 2 } a _ { 2 , d } ^ { 2 } \left( 1 - e ^ { - 2 } \right) + \frac { 1 } { 2 } a _ { 3 , d } ^ { 2 } \left( 1 + e ^ { - 2 } \right) + 3 M _ { d d } ^ { 2 } + 2 a _ { 1 , d } a _ { 2 , d } e ^ { - 1 / 2 } . } \end{array}\tag{H.11}
$$

## Appendix I. Implementation details

This appendix records the architecture and training configuration of the experiments in Section 4.

Reproducibility conventions. All experiments run in double precision. Here, one iteration denotes one optimizer update computed on one batch of the stated size; all training totals below are iteration counts. Reported moment-evaluation timings are warm times of the compiled kernels, and the one-time JIT compilation is reported separately where relevant. The quadrature orders of the moment recursions are: 200-node Gauss–Legendre for the uniforminput benchmarks (Lüthen and Bratley, used both for the propagated moments and for the reference moments), 128-node Gauss–Hermite for the Gaussian-input benchmark (Oakley– O’Hagan) and for the Schrödinger Gram matrix, and a 512-node midpoint rule for the periodic Cahn–Hilliard integrand. All Gauss–Hermite rules are the probabilists’ rule, with weight function $e ^ { - x ^ { 2 } / 2 }$ and weights normalized by $1 / \sqrt { 2 \pi }$ , so that each rule integrates against the standard normal density. Random seeds: five seeds (111, 222, 333, 444, 555) for the parameter-eficiency study (Section 4.1) and 111 for the remaining function-approximation benchmarks (Section 4.2), 9999 for the Cahn–Hilliard and Schrödinger experiments, and NumPy RandomState(7) for the fixed problem data (the Oakley coeficients and the oscillator rotation), whose realized values will be reproducible from the code repository released upon acceptance.

Error metrics. Throughout the experiments, the accuracy of a scalar quantity s (an eigenvalue, a moment, a conserved integral) is reported as the relative error against its reference $s ^ { \mathrm { r e f } }$ 2

$$
\varepsilon ( s ) = { \frac { | s - s ^ { \mathrm { r e f } } | } { | s ^ { \mathrm { r e f } } | } } ,\tag{I.1}
$$

and the accuracy of a field u as the relative $L _ { 2 }$ error against its reference $u ^ { \mathrm { r e f } }$

$$
\varepsilon _ { L _ { 2 } } ( u ) = \frac { \| u - u ^ { \mathrm { r e f } } \| _ { 2 } } { \| u ^ { \mathrm { r e f } } \| _ { 2 } } ,\tag{I.2}
$$

with the norms evaluated on the test grid or test set specified in each experiment.

## Appendix I.1. Parameter-eficiency analysis

All four inner-block variants of Section 4.1 share the same backbone. The shared trunk is $h ( x ) = \mathrm { M L P } ( \mathrm { e m b } ( x ) )$ , where emb is a Chebyshev feature embedding of degree 15 and the MLP has three hidden layers of width $H \ : = \ : 2 0$ with tanh activations. The number of branches is $Q = 2 D + 1 = 4 1$ , and the outer block evaluates per-branch Chebyshev expansions of degree $P = 1 5$ , for $Q ( P + 1 ) = 6 5 6$ outer parameters, identical across all variants. The variants difer only in how the per-dimension branches are obtained from the trunk: independent per-dimension networks $( \Psi _ { d , q } = \operatorname { t a n h } ( \mathrm { M L P } _ { d } ( \operatorname { e m b } _ { d } ( x _ { d } ) ) ) _ { q }$ , about 2,057 parameters per dimension); a full linear head per dimension $( \Psi _ { d } = \operatorname { t a n h } ( W _ { d } h ( x _ { d } ) + b _ { d } )$ with $W _ { d } \in \mathbb { R } ^ { 4 1 \times 2 0 }$ , 861 parameters per dimension); the diagonal specialization (11) (41 scalars per dimension, applied to the shared head $\pmb { \psi } ( \pmb { x } ) = W h ( \pmb { x } ) + b \in \mathbb { R } ^ { 4 1 } )$ ; and the rank-one Lorentz form $( \lambda _ { d , q } = \lambda _ { d }$ , one scalar per dimension). In every case the inner block remains a sum of univariate branches, and the branch decomposition used by the moment recursion was verified against the forward pass to $\leq 2 \times 1 0 ^ { - 1 3 }$ on all trained checkpoints.

Training uses $N _ { C } = 2 0 0 { , } 0 0 0$ Latin hypercube samples in batches of 20,000 and 10,000 iterations of the SOAP optimizer [45] in double precision. Each variant is trained from five random seeds (111, 222, 333, 444, 555), and the error columns of Table 1 report the mean ± standard deviation across them. The inner block uses an initial learning rate of $1 0 ^ { - 3 }$ with exponential decay by a factor 0.9 every 5,000 steps; the outer block is updated by a separate SOAP instance with the same schedule scaled by 0.2. Per-sample variational residual-based attention (vRBA) weights [63] are updated as $\lambda _ { i } \gets \gamma \lambda _ { i } + \eta r _ { i } / \operatorname* { m a x } _ { j } r _ { j }$ with $\eta = 0 . 0 1$ and $\gamma = 0 . 9 9 9$ . The relative $L _ { 2 }$ test errors in Table 1 are computed on 20,000 independent Latin hypercube points, and the analytical moments of each trained surrogate are cross-checked against a $\mathrm { 2 \times 1 0 ^ { 6 } { - } s a m p l e }$ Monte Carlo estimate of the same network, agreeing to within one Monte Carlo standard error.

## Appendix I.2. High-dimensional benchmarks

The twelve surrogates of Section 4.2 span three benchmarks and $D \in \{ 5 , 1 2 , 2 5 , 5 0 \}$ Each uses the diagonal inner block (11) with a shared trunk of three hidden layers of width $H = \operatorname* { m a x } \{ 2 0 , \lfloor Q / 2 \rfloor \}$ and tanh activations, followed by a linear head producing $Q$ branch features; the outer block evaluates per-branch Chebyshev expansions of degree $P = 1 5$ . The Lüthen models use $Q = 2 D + 1$ . The Oakley–O’Hagan models use the same rule except at $D = 5$ , where $Q = 1 2$ , while the Bratley models use $Q = \lfloor 2 . 5 D \rfloor$ . At $D = 5 0$ , these choices give $Q = 1 0 1 \ ( H = 5 0 )$ for Lüthen and Oakley–O’Hagan and $Q = 1 2 5 \ : ( H = 6 2 )$ for Bratley, with 17,783, 18,482, and 25,007 trainable parameters, respectively. For the uniform problems the trunk input is a Chebyshev feature embedding of degree 15 on inputs afinely normalized to [0, 1]; for the Gaussian problem, whose inputs are unbounded, the embedding is replaced by random Fourier features whose frequencies are drawn from ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ with bandwidth $\sigma = 1$ Targets are standardized with the exact reference moments of Appendix H.1–Appendix H.3, and for the Gaussian case the training inputs are drawn by mapping a Latin hypercube sample through the standard normal quantile function.

Training proceeds in three stages in double precision (seed 111), on $N _ { C } = 2 0 0 { , } 0 0 0$ Latin hypercube samples with batch size 20,000; the full procedure is summarized in Algorithm 3. Stage one runs 5,000 iterations of Adam with initial learning rate $1 0 ^ { - 3 }$ , exponentially decayed by a factor 0.9 every 5,000 steps, with per-sample vRBA weights [63] (quadratic potential, $\eta = 0 . 0 1 , \gamma = 0 . 9 9 9 )$ . Stage two polishes all parameters with the self-scaled Broyden (SS-Broyden) quasi-Newton method with backtracking line search [35, 36] for 10,000 iterations, executed in chunks of 100 iterations with the dense inverse-Hessian approximation carried across chunks and a fresh batch drawn per chunk. Stage three refits the last linear layer (the outer Chebyshev coeficients) by ridge-regularized least squares with regularization parameter $\lambda = 1 0 ^ { - 7 }$ on the full training set, which is an exact linear solve since the outer block is linear in its coeficients. Relative $L _ { 2 }$ test errors are computed on 20,000 independent Latin hypercube points. Across all twelve checkpoints, the branch decomposition used by the moment recursion agrees with the forward pass to $\leq 4 \times 1 0 ^ { - 1 0 }$ , and the analytical moments at the production quadrature orders agree with $2 \times 1 0 ^ { 6 }$ -sample Monte Carlo estimates of the same networks to within one Monte Carlo standard error. The warm moment evaluation takes 0.3–0.6 s per model. One-time JIT compilation takes 1–38 s and is excluded from the reported warm timings.

Sampling baselines. The Monte Carlo columns of Table 2 use 200 independent runs of the true integrand with $N = 2 \times 1 0 ^ { 5 }$ pseudo-random samples each. Each run reports the sample mean and the unbiased sample variance.

Algorithm 3 Three-stage training for the high-dimensional benchmarks.   
Require: training set $\{ ( \mathbf { x } _ { i } , u _ { i } ) \} _ { i = 1 } ^ { N _ { C } }$ (Latin hypercube, standardized targets); TOUCKAN   
parameters $\theta \ : = \ : ( \theta _ { \mathrm { i n n e r } } , C )$ with outer Chebyshev coeficients $C ;$ batch size $B ;$ Adam   
iterations $N _ { 1 } ;$ SSBroyden iterations $N _ { 2 }$ in chunks of $c ;$ ridge parameter $\lambda _ { \mathrm { { r i d g e } } }$   
Ensure: trained surrogate $f _ { \theta } { } ,$ , ready for the moment evaluation of Algorithms 1 and 2   
1: Stage 1 — Adam warm-up with vRBA [63]:   
2: for $e = 1 , \ldots , N _ { 1 }$ do   
3: draw a batch of size $B ;$ compute residuals $r _ { i } = | f _ { \theta } ( \mathbf { x } _ { i } ) - u _ { i } |$   
4: $\lambda _ { i } \gets \gamma \lambda _ { i } + \eta r _ { i } /$ max<sub>j</sub> r<sub>j</sub> ▷ vRBA weights, quadratic potential   
5: update $\theta$ with Adam on $\begin{array} { r l r } { \frac { 1 } { B } \sum _ { i } \left( \left( \lambda _ { i } + 0 . 1 \right) r _ { i } \right) ^ { 2 } } & { { } } & { \triangleright \mathrm { l r ~ } 1 0 ^ { - 3 } } \end{array}$ , decay 0.9 per 5,000 steps   
6: end for   
7: Stage 2 — SSBroyden polish [36]: w ← vec(θ), H ← I   
8: for $k = 1 , \ldots , N _ { 2 } / c$ do   
draw a fresh batch of size $B$   
10: run c SSBroyden iterations with backtracking line search on the plain MSE, starting   
from $( \mathbf { w } , H )$   
11: carry the dense inverse-Hessian approximation: $H  \frac { 1 } { 2 } ( H + H ^ { \top } )$ , reset $H \gets I$ if not   
positive definite   
12: end for   
13: Stage ${ \bf 3 - }$ outer-layer refit: with $\theta _ { \mathrm { i n n e r } }$ frozen, assemble the outer Chebyshev features   
$\Phi _ { i , ( q , p ) } = T _ { p } \big ( \xi _ { q } ( \mathbf { x } _ { i } ) \big )$ on the full training set   
14: $C \gets \arg \operatorname* { m i n } _ { C } \big \| \Phi \dot { C } - \mathbf { u } \big \| _ { 2 } ^ { 2 } + \lambda _ { \mathrm { r i d g e } } \| C \| _ { 2 } ^ { 2 }$ ▷ exact linear solve

## Appendix I.3. Cahn–Hilliard equation

The reference solution of Section 4.3 is computed with a Fourier-spectral exponential time-diferencing (ETDRK4) solver on 512 spatial points with internal step $\Delta t = 1 0 ^ { - 5 }$ stored as 201 snapshots on a uniform time grid; its own mass drift is below $1 0 ^ { - 1 6 }$ . The conserved value $\mathcal { Q } _ { 0 } \approx - 0 . 2 8 2 0 9 4 7 9$ is the grid quadrature of the $t = 0$ snapshot. Training uses $N _ { C } = 2 0 0 { , } 0 0 0$ Latin hypercube collocation points in $[ 0 , 1 ] \times [ - 1 , 1 ]$ with batch size 20,000, and the 512 reference points of the $t = 0$ slice as initial-condition points. Periodicity in x is enforced exactly by a periodic Fourier-feature embedding of the spatial input, so the loss has only two terms: the mean-squared residual of (25) in conservative form, with the fourth derivative obtained by nesting forward-mode directional derivatives, and the initial-condition mean-squared error. The residual term carries per-point vRBA weights [63] (exponential potential; $\eta = 0 . 0 1 , \gamma = 0 . 9 9 9$ in the first-order stage, $\gamma = 0 . 9$ in the quasi-Newton stage), which also define the sampling distribution of the batches. The initial-condition term carries a global weight set adaptively by a gradient-norm ratio in the first-order stage and fixed at 100 in the quasi-Newton stage. All runs use double precision and seed 9999.

All five models are parameter-matched at approximately 15k. The MLP and the TOUCK-ANs use tanh activations, and the TNN uses sine activations. The independent TOUCKAN assigns one inner network per input dimension: the spatial branch wraps its input in the periodic embedding $\left( k _ { \operatorname* { m a x } } \ : = \ : 1 5 \right)$ , the time branch acts on the raw coordinate, and each branch is an MLP with four hidden layers of width 40 followed by a head producing $Q = 4 0$ bounded channels, for 14,840 parameters. The full-head and diagonal variants share one embedding and one trunk between the two input dimensions, as in Appendix I.1. The shared embedding spans the full period in $x ,$ which preserves the hard periodicity. The time interval $t \in [ 0 , 1 ]$ covers at most half of that period. On this interval the embedding is smooth and one-to-one. For the time coordinate it therefore acts as an ordinary, non-periodic feature expansion. The full-head model uses a trunk of six hidden layers of width 40 with one linear head per dimension producing $Q = 6 4$ channels (15,328 parameters). The diagonal model uses a trunk of six hidden layers of width 44, a single shared head, and one trainable scalar per dimension and channel (14,912 parameters). All three TOUCKANs use outer Chebyshev expansions of degree $P = 9$ . The MLP baseline applies the periodic embedding $\left( k _ { \operatorname* { m a x } } = 5 \right)$ to the spatial coordinate, concatenates it with t, and processes the result with five hidden layers of width 59 (14,928 parameters). The TNN [26] represents the field as the separable sum $\begin{array} { r } { f ( t , x ) = \sum _ { j = 1 } ^ { R } c _ { j } \varphi _ { j } ( t ) \psi _ { j } ( x ) } \end{array}$ of rank $R = 2 4$ with trainable coeficients $c _ { j }$ . Two factor networks with three hidden layers of width 53 produce the temporal factors $\varphi _ { j }$ and the spatial factors $\psi _ { j }$ , and the spatial input passes through the same periodic embedding $( k _ { \operatorname* { m a x } } = 5 )$ , which keeps the periodicity exact. Both networks use orthogonal weight initialization with biases at 0.5, and each factor is normalized by its root-mean-square value on its one-dimensional quadrature grid (the 512-node spatial rule and a 256-node midpoint rule in time), following [26]. The model has 14,753 parameters.

The exact-mass reprojection implements Section 3.4 with $w \equiv 1$ and $g \equiv 1$ . At each loss evaluation, the spatial branch is evaluated at 512 midpoint quadrature nodes. The branch depends on $x$ only through the periodic embedding, so the integrands $\widetilde { \Psi } ^ { k }$ are smooth and periodic and the midpoint rule converges spectrally. The weighted moments then follow from the binomial recursion, with the time branch contributing known powers, and the contraction with the outer Chebyshev coeficients through the monomial change of basis yields $\mathcal { Q } [ f ] ( t )$ in closed form (22). The output is the projection (23) with $\mathcal { Q } _ { w } [ g ] = | \Omega | = 2$ . The correction is constant in $x ,$ so it is invisible to the spatial derivatives in the residual and enters only through the time derivative. Its cost is one evaluation of the spatial branch at the 512 quadrature nodes per loss call. The constraint is thus enforced exactly for the 512-node quadrature functional. Since the integrand is smooth and periodic, the continuum mass is conserved to the spectral accuracy of the midpoint rule, which we verify directly by refining the integration grid ( Appendix F, Figure F.8).

The same reprojection carries over to the TNN because its representation is already separable. The mass functional factorizes as $\begin{array} { r } { \mathcal { Q } [ f ] ( t ) = \sum _ { j } c _ { j } \varphi _ { j } ( t ) \int _ { \Omega } \psi _ { j } ( x ) d x } \end{array}$ , where the one-dimensional factor integrals are evaluated on the same 512-node midpoint rule, and the output is projected exactly as in (23). As in the TOUCKAN case, the cost is one evaluation of the spatial factor network at the quadrature nodes per loss call.

The constrained TOUCKANs are trained by fitting first and projecting second. Each architecture is first trained without the constraint: 6,250 iterations of the Lion optimizer [64] (initial learning rate 10<sup>−3</sup>, exponential decay by 0.9 every 5,000 steps, outer coeficients at 20% of the base rate), followed by 20,000 SSBroyden iterations [36] with backtracking line search, executed in chunks of 100 with the inverse-Hessian approximation carried across chunks and a fresh vRBA-sampled batch per chunk. The exact-mass reprojection is then applied, the converged parameters are loaded unchanged (the constrained and unconstrained models share the same parameter tree), and the model is polished for a further 20,000 SSBroyden iterations under the constraint. The order matters: training the constrained model from a random initialization stalls, because the projection makes the output invariant to the spatially constant mode of the raw network, and this null space blocks the escape path that the unconstrained optimization uses early in training. The warm start instead places the model on the constraint manifold near the solution, and the second-order stage only polishes. The constrained TNN follows the same fit-then-project schedule as the TOUCKANs. The MLP baseline, which carries no constraint, is trained from scratch with the same pipeline at 6,250 first-order and 40,000 SSBroyden iterations, so every model in Table 3 receives the same total second-order budget. Errors are computed on the exact $5 1 2 \times 2 0 1$ reference grid without interpolation, and the mass drift is the maximum over the snapshots of the grid-quadrature deviation from $\mathcal { Q } _ { 0 }$

For reference, the unconstrained twins at the warm-start point (6,250 Lion and 20,000 SSBroyden iterations, before the exact-mass reprojection is applied) reach relative $L _ { 2 }$ errors of $6 . 6 \times 1 0 ^ { - 4 }$ (independent), $3 . 3 \times 1 0 ^ { - 3 }$ (full-head), and $2 . 8 \times 1 0 ^ { - 2 }$ (diagonal), with mass drifts of $6 . 4 \times 1 0 ^ { - 4 } , 3 . 6 \times 1 0 ^ { - 3 }$ , and $3 . 2 \times 1 0 ^ { - 2 }$ , respectively. The TNN twin reaches $4 . 7 \times 1 0 ^ { - 3 }$ with a mass drift of $5 . 2 \times 1 0 ^ { - 3 }$ . In each case the unconstrained drift sits at the level of the field error, consistent with the soft-conservation behavior of the MLP baseline, while the constrained continuation reduces the drift to machine precision (Table 3). Because the constrained polish also continues the optimization, these paired values do not isolate the efect of the reprojection on field accuracy. An equal-budget unconstrained continuation is left for future work.

## Appendix I.4. Schrödinger eigenvalue problem

The test problem is chosen to be genuinely multivariate and yet exactly solvable, so that errors can be measured precisely. It is the coupled harmonic oscillator: the potential in (26) is $V ( \mathbf { x } ) \ = \ \textstyle { \frac { 1 } { 2 } } \mathbf { x } ^ { \top } A \mathbf { x }$ , and specifying the problem means specifying the matrix A. We build it as $A \ : = \ : R \mathrm { d i a g } ( \omega _ { 1 } ^ { 2 } , . . . , \omega _ { D } ^ { 2 } ) R ^ { \intercal }$ from two ingredients. The frequencies $\omega _ { i } =$ $1 + 0 . 2 ( i - 1 )$ , so $\omega = ( 1 . 0 , 1 . 2 )$ in $D = 2$ and $( 1 . 0 , 1 . 2 , 1 . 4 )$ in $D = 3 ,$ set the stifness of the D normal modes. The frequencies are commensurate, so the full spectrum $E ( \mathbf { n } )$ contains degeneracies at higher energies. The four target eigenvalues, however, are non-degenerate in each dimension considered (1.1, 2.1, 2.3, 3.1 for D = 2; 1.8, 2.8, 3.0, 3.2 for $D = 3 ; 3 . 5$ 4.5, 4.7, 4.9 for $D = 5 )$ . Hence the target eigenfunctions are unique up to sign and per-state errors are well defined. The rotation $R \in \mathbb { R } ^ { D \times D }$ is a dense orthogonal matrix, the QR factor of a standard normal matrix drawn with $\mathrm { N u m P y }$ RandomState(7). This rotation mixes all coordinates, so the potential contains every cross term $x _ { d } x _ { d ^ { \prime } }$ and neither the potential nor the eigenfunctions separate in the input variables. The surrogate therefore cannot exploit any one-dimensional structure of the problem.

The exact reference is obtained by undoing the rotation. In the rotated coordinates $\mathbf { y } =$ $R ^ { \top } { \bf x }$ , the Hamiltonian splits into D independent one-dimensional oscillators with frequencies $\omega _ { i }$ , whose solutions are classical. The eigenfunctions are products of one-dimensional Hermite functions of the rotated coordinates, indexed by occupation numbers $\mathbf { n } = ( n _ { 1 } , \ldots , n _ { D } ) \in \mathbb { N } _ { 0 } ^ { D }$ and the eigenvalues are $\begin{array} { r } { E ( \mathbf { n } ) = \sum _ { i = 1 } ^ { D } \omega _ { i } \left( n _ { i } + \frac { 1 } { 2 } \right) } \end{array}$ ; sorting them gives the lowest $N _ { s }$ eigenpairs. These closed forms are used only to evaluate errors. Training never sees them.

Solving (26) with a network means finding $N _ { s }$ functions $\psi _ { 1 } , \dots , \psi _ { N _ { s } }$ and $N _ { s }$ numbers $E _ { 1 } , \dots , E _ { N _ { s } }$ that meet four requirements at once. (i) Each pair must satisfy the equation, $\mathcal { H } \psi _ { a } = E _ { a } \psi _ { a }$ . (ii) Each state must be normalized, $\textstyle { \int { \psi _ { a } ^ { 2 } d { \bf x } } = 1 }$ . Without this requirement, $\psi \equiv 0$ solves (i) for every E. (iii) The states must be mutually orthogonal, $\int { \psi _ { a } \psi _ { b } d { \bf x } } = 0$ Without this requirement, all $N _ { s }$ outputs can converge to the same state. (iv) The family must consist of the lowest $N _ { s }$ eigenpairs. Without this requirement, any $N _ { s }$ distinct eigenstates satisfy (i)–(iii). The method assigns one ingredient to each requirement: a residual loss for (i), hard normalization for (ii), a deterministically evaluated orthogonality penalty for (iii), and an energy-minimization term for (iv). The eigenvalues $E _ { a }$ are trainable scalars throughout.

The representation starts from the known decay of the solutions. Bound states of a confining potential decay rapidly at infinity. Each state is therefore written as a product $f _ { a } \chi$ where $f _ { a }$ is one output of a single TOUCKAN and $\chi ( { \bf x } ) = ( 2 \pi ) ^ { - D / 4 } e ^ { - | { \bf x } | ^ { 2 } / 4 }$ is a fixed Gaussian envelope. The envelope plays two roles. First, it carries the decay exactly, so no boundary loss and no domain truncation are needed. Second, its square is the standard normal density, $\chi ^ { 2 } = \rho ,$ so the inner product of any two states becomes a Gaussian expectation of a product of network outputs, $\begin{array} { r } { \int ( f _ { a } \chi ) ( f _ { b } \chi ) d \mathbf x = \mathbb { E } _ { \rho } [ f _ { a } f _ { b } ] } \end{array}$ . This is precisely the quantity that the crossmoment recursion of Section 3.3 computes in closed form.

This observation makes requirement (ii) free. Collect the expectations into the Gram matrix $G \in \mathbb { R } ^ { N _ { s } \times N _ { s } }$ with entries $G _ { a b } = \mathbb { E } _ { \rho } [ f _ { a } f _ { b } ]$ . The diagonal entry $G _ { a a }$ is the squared norm of the product $f _ { a \chi }$ . Each state is normalized by dividing by it, $\psi _ { a } = f _ { a } \chi / \sqrt { G _ { a a } } ;$ which is the hard-normalized form (24). The identity $\textstyle \int \psi _ { a } ^ { 2 } d \mathbf { x } = 1$ then holds for every value of the network parameters, at every stage of training; normalization never appears in the loss. The construction also removes the trivial solution: the loss below is invariant under rescaling $f _ { a } \mapsto c f _ { a } ,$ so shrinking a state toward zero changes nothing and is never rewarded.

Requirement (i) becomes a residual loss, posed for $f _ { a }$ rather than for $\psi _ { a }$ . Substituting $\psi _ { a } = f _ { a } \chi / \sqrt { G _ { a a } }$ into (26) and dividing by χ gives the residual

$$
\begin{array} { r l r l } { r _ { a } = - \frac { 1 } { 2 } \Delta f _ { a } + \frac { 1 } { 2 } \mathbf { x } \cdot \nabla f _ { a } + W ( \mathbf { x } ) ~ f _ { a } - E _ { a } f _ { a } , } & { } & { W ( \mathbf { x } ) = \frac { D } { 4 } - \frac { | \mathbf { x } | ^ { 2 } } { 8 } + \frac { 1 } { 2 } \mathbf { x } ^ { \top } A \mathbf { x } , } \end{array}\tag{I.3}
$$

so that $r _ { a } \chi = ( \mathcal { H } - E _ { a } ) ( f _ { a } \chi )$ . The derivatives of $\chi$ are absorbed analytically into the drift and potential terms of (I.3). No derivative ever acts on $\chi$ numerically. The collocation set is drawn from $\rho$ itself, and $\begin{array} { r } { \mathbb { E } _ { \rho } \big [ ( r _ { a } / \sqrt { G _ { a a } } ) ^ { 2 } \big ] = \int ( \mathcal { H } \psi _ { a } - E _ { a } \psi _ { a } ) ^ { 2 } d \mathbf { x } } \end{array}$ The residual batches, however, are resampled from this set with probabilities proportional to the vRBA weights, and the summands additionally carry those weights (Eq. (I.4)), with no inverse-probability correction. The optimized objective is therefore an adaptively reweighted $\rho -$ residual rather than an unbiased estimate of this integral, but the two have the same minimizers, since both vanish exactly when $r _ { a } \equiv 0$ . The residual is also the only term in which the trainable eigenvalues appear, so it is what pins each $E _ { a }$ to its state. Monte Carlo enters the formulation only through such Hamiltonian expectations: this residual, and the numerator of the selection term below. Every norm and overlap is deterministic and sampling-free.

Algorithm 4 Hard-normalized eigensolver training for the Schrödinger problem.   
Require: collocation set $\{ { \mathbf { x } } _ { i } \} _ { i = 1 } ^ { N _ { C } }$ drawn from $\rho ;$ TOUCKAN parameters $\theta$ with $N _ { s }$ outer   
coeficient columns; eigenvalues $E _ { 1 } , \dots , E _ { N _ { s } }$ initialized below the spectrum; weights $\lambda _ { B } ,$   
$w _ { R } ;$ batch size $B ;$ first-order iterations $N _ { 1 } ;$ SSBroyden iterations $N _ { 2 }$ in chunks of c   
Ensure: eigenpairs $( E _ { a } , \psi _ { a } )$ with $\psi _ { a } = f _ { a } \chi / \sqrt { G _ { a a } } ,$ normalized identically in $\theta$   
1: function $\operatorname { L o s s } ( \theta , E ; B , B ^ { \prime } )$ ▷ B: residual batch; $B ^ { \prime } \colon$ uniform batch for K   
2: $\big ( f _ { 1 } ( \mathbf { x } ) , \dots , f _ { N _ { s } } ( \mathbf { x } ) \big ) \gets \mathrm { T O U C K A N } _ { \theta } ( \mathbf { x } )$ , ∀ x in both batches ▷ one network $( 7 ) , N _ { s }$   
output columns   
3: $G \gets \mathbb { E } _ { \rho } [ f _ { a } f _ { b } ]$ from $\theta ,$ by the cross-moment recursion (17) ▷ deterministic Gram   
matrix, diferentiable   
4: $\begin{array} { r } { W ( \mathbf { x } ) \gets \frac { D } { 4 } - \frac { 1 } { 8 } | \mathbf { x } | ^ { 2 } + \frac { 1 } { 2 } \mathbf { x } ^ { \top } A \mathbf { x } } \end{array}$ ▷ envelope-transformed potential   
5: $\begin{array} { r } { r _ { a } \gets - \frac { 1 } { 2 } \Delta \dot { f } _ { a } \gets \frac { 1 } { 2 } \mathbf { x } \cdot \bar { \nabla } f _ { a } + W f _ { a } - E _ { a } f _ { a } } \end{array}$ , ∀ x in both batches ▷ residual Eq. (I.3)   
6: $\lambda _ { i , a } \gets \bar { \gamma } \lambda _ { i , a } + \eta \bar { | } r _ { a } ( \mathbf { x } _ { i } ) | / \operatorname* { m a x } _ { j } | r _ { a } ( \mathbf { x } _ { j } ) |$ ▷ vRBA weights   
7: $\begin{array} { r } { \mathcal { L } _ { \mathrm { r e s } }  \frac { 1 } { | \mathcal { B } | } \sum _ { \mathbf { x } _ { i } \in \mathcal { B } } \sum _ { a } ( \lambda _ { i , a } r _ { a } ( \mathbf { x } _ { i } ) / \sqrt { G _ { a a } } ) ^ { 2 } } \end{array}$ ▷ Monte Carlo   
8: $\widehat { G } _ { a b } \gets G _ { a b } / \sqrt { G _ { a a } G _ { b b } }$   
9: $\begin{array} { r } { \mathcal { L } _ { \mathrm { o r t h } }  \sum _ { a \neq b } \widehat { G } _ { a b } ^ { 2 } } \end{array}$ ▷ exact, free of Monte Carlo noise   
10: $\begin{array} { r } { K _ { a b }  \frac { 1 } { \vert { \mathcal B ^ { \prime } } \vert } \overset { \cdot } { \sum } _ { { \mathbf x } _ { i } \in { \mathcal B ^ { \prime } } } f _ { a } ( { \mathbf x } _ { i } ) \big ( r _ { b } ( { \mathbf x } _ { i } ) + E _ { b } f _ { b } ( { \mathbf x } _ { i } ) \big ) } \end{array}$ ▷ energy matrix Eq. (I.5), Monte Carlo   
11: $K \gets \frac { 1 } { 2 } \dot { ( K + K ^ { \top } ) }$   
12: return $\mathscr { L } _ { \mathrm { r e s } } + \lambda _ { B } \mathscr { L } _ { \mathrm { o r t h } } + w _ { R } \mathrm { t r } ( G ^ { - 1 } K )$ ▷ Ky Fan selection   
13: end function   
14: Stage 1 — first-order, all $N _ { s }$ states simultaneously:   
15: for $e = 1 , \ldots , N _ { 1 }$ do ▷ one gradient step per iteration   
16: $B  B$ pool points, sampled with probability ∝ their vRBA weights ▷ focuses   
high-residual regions   
17: $B ^ { \prime }  B ^ { \prime }$ pool points, sampled uniformly ▷ unbiased, for K   
18: $g \gets \nabla _ { ( \theta , E ) } \operatorname { L o s s } ( \theta , E ; B , B ^ { \prime } )$   
19: $( \theta , E ) \longleftarrow \mathrm { S O A P } \big ( ( \theta , E ) , g \big )$ [45] ▷ lr $1 0 ^ { - 3 }$ , decay 0.9 per 5,000 steps   
20: end for   
21: Stage 2 — SSBroyden polish [36]:   
22: w ← vec(θ, E), H ← I   
23: for $k = 1 , \ldots , N _ { 2 } / c$ do   
24: draw fresh B and $B ^ { \prime }$ as in Stage 1   
25: run c SSBroyden iterations with backtracking line search on Loss(w; B, B<sup>′</sup>), from   
$( \mathbf { w } , H )$   
26: $H \stackrel { \cdot } {  } \frac { 1 } { 2 } ( H + H ^ { \top } ) ;$ reset $H \gets I$ if not positive definite   
27: keep the lowest-residual w seen so far   
28: end for

The full Gram matrix is deterministic and diferentiable at every training step, exact up to the one-dimensional quadrature error of the joint leaf moments. The states share one inner block, so $f _ { a }$ and $f _ { b }$ are polynomials in the same combination variables $\xi _ { q } ,$ and the product $f _ { a } f _ { b }$ expands in exactly the cross-moments $m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) } = \mathbb { E } [ \xi _ { q } ^ { j } \xi _ { q ^ { \prime } } ^ { j ^ { \prime } } ]$ of Section 3.3. Each Gram matrix element is therefore the bilinear analogue of the second-moment reduction (16), $\begin{array} { r } { G _ { a b } = \sum _ { q , q ^ { \prime } } \sum _ { j , j ^ { \prime } } \beta _ { q , j } ^ { ( a ) } \beta _ { q ^ { \prime } , j ^ { \prime } } ^ { ( b ) } m _ { q , q ^ { \prime } } ^ { ( j , j ^ { \prime } ) } } \end{array}$ , where $\beta ^ { ( a ) }$ are the monomial-form outer coeficients of state a. The cross-moments are computed once per step: joint leaf moments on a 128- node Gauss–Hermite rule per dimension, then the bivariate binomial recursion (17) across dimensions. All $N _ { s } ^ { 2 }$ matrix elements reuse the same cross-moments and difer only in the coeficient contraction. No sampling is involved at any point, and the result is diferentiated through like any other layer of the network.

In summary, the loss combines three terms,

$$
{ \mathcal { L } } = { \frac { 1 } { B } } \sum _ { i = 1 } ^ { B } \sum _ { a = 1 } ^ { N _ { s } } \left( \lambda _ { i , a } { \frac { r _ { a } ( \mathbf { x } _ { i } ) } { \sqrt { G _ { a a } } } } \right) ^ { 2 } ~ + ~ \lambda _ { B } \sum _ { a \neq b } { \widehat { G } } _ { a b } ^ { 2 } ~ + ~ w _ { R } ~ \mathrm { t r } \left( G ^ { - 1 } K \right) , \qquad { \widehat { G } } _ { a b } = { \frac { G _ { a b } } { \sqrt { G _ { a a } G _ { b b } } } } .\tag{I.4}
$$

The three terms implement requirements (i), (iii), and (iv) in order; requirement (ii) is built into the representation and needs no term. The first term is the eigen-residual of (i), with per-point vRBA weights $\lambda _ { i , a }$ [63]. The second term enforces the orthogonality of (iii). It is computed entirely from the deterministically evaluated Gram: the matrix G delivered by the recursion is normalized to $\widehat { G }$ , and the squared of-diagonal entries are summed. No collocation points enter this term, so it carries no sampling noise, and the diagonal of $\widehat { G }$ is one by construction, so it contains no normalization component. Its weight is fixed at $\lambda _ { B } = 1 0 0$ in both training stages.

The trace term implements requirement (iv): it measures the total energy of the family, and minimizing it selects the lowest states. The matrix K holds the energy matrix elements of the unnormalized states,

$$
K _ { a b } = \int \left( f _ { a } \chi \right) \mathcal { H } \left( f _ { b } \chi \right) d \mathbf { x } = \mathbb { E } _ { \rho } \left[ f _ { a } \mathcal { H } ^ { \chi } f _ { b } \right] , \qquad \mathcal { H } ^ { \chi } f = r + E f ,\tag{I.5}
$$

where $\mathcal { H } ^ { \chi } f : = \chi ^ { - 1 } \mathcal { H } ( \chi f )$ is the envelope-transformed Hamiltonian. By (I.3), it evaluates as $\mathcal { H } ^ { \chi } f = r + E f$ . Both ingredients, the residual and the state, are already available. For a single state, $K _ { a a } / G _ { a a }$ is its energy, the Rayleigh quotient. For the family of states, $\operatorname { t r } ( G ^ { - 1 } K )$ is the natural generalization: it equals the sum of the energies of the $N _ { s }$ states after orthonormalization, and it depends only on the subspace they span. By the Ky Fan principle [50], its minimum over all $N _ { s }$ -dimensional subspaces is the sum of the $N _ { s }$ lowest eigenvalues, attained exactly when the span is the lowest eigenspace. This is requirement (iv). Three properties make the term safe. Copying an already occupied state does not lower the trace, because the trace sees only the span. Collapse of the family onto fewer than $N _ { s }$ directions drives G toward singularity and raises the trace, so collapse acts as a barrier rather than a shortcut. Indeed, no collapse or singular solve was observed in any reported run. The identity $\mathcal { H } ^ { \chi } f _ { a } = r _ { a } + E _ { a } f _ { a }$ cancels the trainable eigenvalues inside the trace, so the term steers only the network functions and leaves the $E _ { a }$ to the residual. The formulation assumes $G \succ 0$ , in particular $G _ { a a } > 0$ in the normalization; this held in every reported run. At the trained parameters, the Gram matrix is well conditioned in every dimension, with smallest eigenvalue bounded away from zero and condition number of order $1 0 ^ { 1 } – 1 0 ^ { 2 }$ The of-diagonal Gram residual that training monitors falls below $1 0 ^ { - 5 }$ in every case (Figure 5, bottom row), so the family stays near-orthogonal and far from collapse. Table I.8 reports these quantities. The term is implemented as a direct dense solve, tr $\left( G ^ { - 1 } K \right) = \operatorname { t r } \left( { \mathrm { s o l v e } } ( G , K ) \right)$ , with no added jitter or regularization. The numerator K is the second and last Monte Carlo estimate, computed on its own uniformly drawn batch and symmetrized. This estimate is unbiased for the exact energy couplings under the stated uniform sampling. The denominator G is deterministic, exact up to the one-dimensional quadrature error. Because the exact (expected) trace is stationary precisely at eigenspaces, the trace term introduces no bias at convergence for the exact objective and is never annealed. The weight for the trace term is $w _ { R } = 0 . 1$ throughout.

Table I.8: Conditioning of the Gram matrix G at the trained parameters for the Schrödinger eigenstate runs $( N _ { s } = 4$ states, evaluated by the exact recursion at the production order $M = 1 2 8 )$ . The condition number $\kappa ( G ) = \lambda _ { \mathrm { m a x } } / \lambda _ { \mathrm { m i n } }$ stays at order $1 0 ^ { 1 } – 1 0 ^ { 2 }$ with $\lambda _ { \operatorname* { m i n } } ( G ) > 0$ at the trained parameters, so the direct solve solve(G, K) is stable without regularization. Successful completion of every run indicates no observed singular solve, but these final-parameter values do not quantify the conditioning during optimization. The last column reports the of-diagonal Gram residual at the final training iteration (Figure 5).
<table><tr><td>D</td><td> $\overline { { \lambda _ { \operatorname* { m i n } } ( G ) } }$ </td><td> $\overline { { \lambda _ { \operatorname* { m a x } } ( G ) } }$ </td><td> $\overline { { \kappa ( G ) } }$ </td><td>off-diag. residual</td></tr><tr><td>2</td><td> $\overline { { 3 . 9 7 \times 1 0 ^ { - 4 } } }$ </td><td> $\overline { { 8 . 9 6 \times 1 0 ^ { - 2 } } }$ </td><td> $\overline { { 2 . 2 6 \times 1 0 ^ { 2 } } }$ </td><td> $\overline { { 2 . 7 3 \times 1 0 ^ { - 9 } } }$ </td></tr><tr><td>3</td><td> $4 . 5 8 \times 1 0 ^ { - 5 }$ </td><td> $1 . 4 2 \times 1 0 ^ { - 3 }$ </td><td> $3 . 1 1 \times 1 0 ^ { 1 }$ </td><td> $4 . 8 8 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>5</td><td> $6 . 9 9 \times 1 0 ^ { - 9 }$ </td><td> $2 . 2 3 \times 1 0 ^ { - 7 }$ </td><td> $3 . 2 0 \times 1 0 ^ { 1 }$ </td><td> $5 . 2 2 \times 1 0 ^ { - 6 }$ </td></tr></table>

One network represents all $N _ { s }$ states. The inner block is the intermediate sharing form of Section 3.1: a shared random Fourier embedding (15 frequencies, bandwidth $\sigma = 1$ , matched to the unbounded Gaussian inputs), a shared trunk of four hidden layers of width 32 with tanh activations, and one full linear head per input dimension producing $Q = 6 4$ branch channels. Each branch ends in a tanh and the channels are averaged over dimensions, so every combination variable lies in (−1, 1). The outer block evaluates per-branch Chebyshev expansions of degree $P = 9$ with $N _ { s }$ coeficient columns: the states share the same combination variables and difer only through their outer coeficients. Together with the $N _ { s }$ trainable eigenvalues, this gives the parameter counts of Table 4.

Training is simultaneous and assumes nothing about the spectrum. All $N _ { s }$ states are active from the first iteration; there is no sequential scan over eigenvalues and no scheduled sweep of trial energies. The $E _ { a }$ are initialized at small distinct values below the bottom of the spectrum, for which a lower bound on the potential sufices, and rise to their converged values by tracking the Rayleigh quotients of their states through the residual term. The optimization follows the two-stage template of the other experiments, in double precision on $N _ { C } = 2 0 0 { , } 0 0 0$ collocation points drawn by mapping a Latin hypercube sample through the standard normal quantile function, with batch size $2 0 { , } 0 0 0$ . The full procedure is summarized in Algorithm 4. Stage one runs 20,000 first-order iterations (SOAP [45]; initial learning rate $1 0 ^ { - 3 }$ , decayed by 0.9 every 5,000 steps, outer coeficients at 20% of the base rate), with vRBA weights (exponential potential, $\eta ~ = ~ 0 . 0 1 , ~ \gamma ~ = ~ 0 . 9 9 9 )$ that also define the batch sampling distribution. Stage two runs 20,000 SSBroyden iterations [36] with backtracking line search, in chunks of 100 with a fresh vRBA-sampled batch per chunk and the inverse-Hessian approximation carried across chunks. The parameters with the lowest mean residual over the second stage are kept. As in the other experiments, all reported results are single-run

observations at the stated seed.

The errors of Table 4 follow the definitions of Eq. (I.1) and Eq. (I.2), with the exact spectrum and eigenfunctions as references, the sign of each $\psi _ { a }$ aligned to the reference (eigenfunctions are defined only up to sign), and the $L _ { 2 }$ norms estimated on an independent set of 20,000 test points drawn from $\rho ,$ separate from the collocation set used for training.

The formulation inherits the problem statement of Jin et al. [49] but replaces its constraint machinery. We retain a residual loss with trainable eigenvalues, normalization and orthogonality requirements, and decay at infinity. In their MLP-based method, normalization and orthogonality are evaluated through collocation sums, and states are obtained sequentially through a sampled orthogonality penalty with patience-based solution recognition and symmetry switching. The trial-energy drive inherited from their earlier formulation is disabled in the reported experiments, and infinity is replaced by a truncated domain. These mechanisms introduce sampling or truncation error, and the reported experiments solve one-dimensional equations. TOUCKAN instead handles the integral constraints without sampling: normalization is imposed by division, and orthogonality is enforced through the deterministic Gram matrix. State selection uses a single variational trace term, while the envelope carries the decay over all of $\mathbb { R } ^ { D }$ . This formulation can therefore be trained on non-separable multivariate problems while learning all $N _ { s }$ states simultaneously.

Figures I.9 and I.10 complement Figure 6 of the main text with the learned eigenstates of the $D = 2$ and $D = 5$ runs.

![](images/be3fa6251cfa92ef1462205bd05a7f6bd093639018920816bdde18b418a4695e.jpg)  
Figure I.9: Lowest four eigenstates of the coupled two-dimensional oscillator: closed-form reference (left), learned surrogate (center), and absolute error on a logarithmic color scale (right). Each row is labeled by its learned eigenvalue $E _ { a }$

![](images/426f5fab9c1829a3b99ee14ec0840d7c577e9b28a958ce11b89fa8ef652e2840.jpg)  
Figure I.10: Lowest four eigenstates of the coupled five-dimensional oscillator on the $x _ { 3 } = x _ { 4 } = x _ { 5 } = 0$ plane: closed-form reference (left), learned surrogate (center), and absolute error on a logarithmic color scale (right). Each row is labeled by its learned eigenvalue $E _ { a }$ . On this slice, the $E _ { 1 }$ and $E _ { 2 }$ states have similar projected profiles, while the Gram residuals in Table 4 quantify their full-dimensional distinction.