# Efficient and Generalizable Archetypal Analysis for Discrete Data

A. Emilie J. Wedenborg, Jesper Løve Hinrich and Morten Mørup

Abstract—Archetypal Analysis (AA) represents observations as convex combinations of extremal data-driven profiles, yielding interpretable low-dimensional descriptions of complex datasets. Classical AA is typically formulated with a least-squares objective, which is poorly matched to binary, count, and categorical observations arising in domains such as genomics, microbiome analysis, recommender systems, and medical questionnaires. We introduce an efficient and generic likelihood-based framework for AA of discrete data that supports Bernoulli, Poisson, and multinomial observation models. The proposed optimization scheme uses local quadratic approximations of the associated AA-models negative log-likelihood, enabling efficient constrained updates through sequential minimal optimization (SMO) for observation-specific mixture weights and an active-set method for mixture weights used to construct each archetype from the data observations. To improve scalability, the active set is explicitly bounded while maintaining feasibility through a simplex-preserving proxy representation. We further propose a cross-validated predictive likelihood framework relying on quantifying how the trained AA representations generalize to held-out data as a principled criterion for selecting the number of archetypes, providing an alternative to reconstruction-error heuristics and stability-only diagnostics. Synthetic experiments establish the efficiency of our proposed procedure and show that the proposed predictive likelihood framework recovers the true model complexity. Applications to single-cell RNA sequencing, microbiome composition, and somatic mutation data further demonstrate that the learned archetypes capture interpretable domain-specific structures while providing competitive likelihood fits and stable solutions when compared to alternative inference procedures. Notably, the predictive likelihood framework here also points to suitable model complexities guiding the model selection. These results establish the proposed AA framework as an efficient approach to likelihood-based inference in AA for discrete data with the associated predictive likelihood based approach providing a complementary new tool for model order selection in AA.

Index Terms—Archetypal analysis, discrete data, likelihoodbased optimization, sequential minimal optimization, active-set methods, predictive likelihood, model selection

## I. INTRODUCTION

Understanding complex datasets often requires identifying the most extreme and informative patterns that govern variability. Rather than partitioning data into clusters or projecting observations onto orthogonal latent components, Archetypal

Analysis (AA), originally introduced by [2], provides a geometric framework in which each data point is expressed as a convex combination of extremal learned representatives. AA models observations as mixtures of archetypes that themselves lie within the convex hull of the dataset. This construction induces a low-dimensional polytope embedded in the feature space, where vertices correspond to optimally distinctive datadriven profiles.

This perspective distinguishes AA from clustering approaches, which assign observations to discrete groups with the extracted prototypes resembling average properties of the groups, and from classical matrix factorization techniques such as non-negative matrix factorization [3], which do not enforce extremality of the latent factors but produces part based representations [4], see also [5]. By constraining archetypes to be convex combinations of observed samples, AA yields interpretable representations that capture trade-offs between extreme yet realizable configurations. As a result, AA has found applications across neuroscience [6], [7], genomics [8], computer vision [9], [10], and deep generative modeling [11], see [12] for a recent survey.

## A. Optimization Challenges in Archetypal Analysis

Despite its interpretability, AA presents substantial computational challenges. The estimation problem is inherently non-convex, as both the convex mixtures used to define the archetypes and the convex reconstruction weights must be learned simultaneously. Standard approaches therefore alternate between two convex subproblems formed by solving for each set of parameters at a time, typically formulated as quadratic programs that converge to a local optimum.

Numerous algorithmic refinements have been proposed to improve scalability and stability. The Principal Convex Hull Algorithm (PCHA) introduced by [5] provides efficient projected-gradient updates and remains a widely used baseline. Other work has explored robust loss formulations [9], coreset constructions for large-scale data [13], probabilistic relaxations [14], and deep-learning-based extensions [11]. While these advances significantly improve performance for continuous data under squared-error objectives, most of these existing optimization strategies remain tightly coupled to least-squares reconstruction.

## B. Archetypal Analysis for Discrete Data

In many modern applications, including genomics, recommender systems, medical questionnaires, and single-cell sequencing, observations are binary, categorical, or counts. This makes a classical least-squares AA method suboptimal.

Probabilistic extensions of AA (PAA) have attempted to address this limitation [15]. For binary and count data, multiplicative update rules inspired by non-negative factorization methods have been proposed. However, these updates are often slow to converge and sensitive to initialization [16], [17]. Alternative strategies rely on transforming discrete data into continuous representations [18] or constraining archetypes to coincide with actual observations (archetypoids) [19]. While useful, such approaches either distort the underlying datagenerating process or restrict model flexibility.

Consequently, there remains a need for a computationally efficient and statistically principled optimization framework for likelihood-based AA applicable to discrete data.

## C. Proposed Framework

We develop a unified and efficient likelihood-based inference framework for AA, named Distribution Agnostic Archetypal Analysis (DAAA). Our approach is based on three key principles:

a) Exploiting sparsity in archetype construction: The convex combinations defining the archetypes are typically sparse at optimality. We exploit this property using an activeset strategy inspired by classical non-negativity constrained least squares methods [20], enabling efficient updates under simplex constraints. To further promote computational efficiency, we explicitly control the size of the active set by imposing a bound on the active set size using a reparameterization in which the current archetypal estimate is used as an additional observation. When the active set exceeds this limit, it is thereby reset to a single element pointing to this pseudo-observation whereas the weights are reassigned in a manner that preserves feasibility and ensures convergence. This strategy guarantees that the active set procedure remains computationally lightweight while maintaining strict convergence.

b) Low-dimensional reconstruction via Sequential Minimal Optimization: The observation-specific convex weights typically lie in a low-dimensional simplex defined by the number of archetypes used for the modeling. We formulate their update as a sequence of analytically solvable two-variable subproblems using Sequential Minimal Optimization (SMO) [21], yielding closed-form updates for these two-variable subproblems with low computational overhead.

c) Second-order likelihood expansion: Rather than deriving individual solvers for each observation model as in [15], we construct a quadratic approximation of the negative log-likelihood at each iteration. This transforms the general likelihood-based objective into a sequence of weighted quadratic subproblems compatible with our active-set and SMO updates. We focus on discrete observation models, for which the likelihood-specific gradient and second-order (curvature) terms admit simple expressions. However, this approach is generic and we note that the optimization strategy applies to any twice-differentiable separable loss for which a local quadratic approximation can be constructed.

## D. Model Selection via Predictive Likelihood

Determining the number of archetypes K remains a longstanding open problem in AA. Existing approaches typically rely on heuristic inspection of reconstruction error curves [5] or stability measures such as normalized mutual information [6]. Alternatives have been information criteria relying on training sample estimates [22], [23], see also [12]. We presently introduce an out-of-sample predictive likelihood framework for selecting K based on evaluating the heldout (predictive) log-likelihood (LL) of the trained archetypal model. We thereby transform archetype selection into a principled problem optimizing for generalization. This thereby bridges the AA model selection problem with established statistical learning methodology and provides a direct model generalization criteria for determining the complexity of the archetypal analysis induced polytope.

## E. Contributions

In summary, our main contributions are:

• A likelihood-based AA inference framework with an optimization strategy that extends to general twicedifferentiable separable losses through local quadratic approximations highlighted in the context of AA for discrete data.

• Closed-form efficient inference for the two convex subproblems, respectively exploring a constrained activeset procedure to bound the update complexity for the typically sparse mixture weights defining the archetypes, and a SMO procedure for the typical low-dimensional simplex structure used to reconstruct each observation.

• A predictive likelihood criterion for principled selection of the number of archetypes considering out-of-sample model generalization.

• Extensive empirical validation on synthetic and realworld datasets demonstrating improved convergence speed and statistical performance compared to multiplicative-update methods and projected gradient extending presently also the PCHA procedure to AA on discrete data.

## II. METHODS

## A. Likelihood-Optimized Archetypal Analysis

We introduce a general likelihood-based formulation of AA [2]. Let $\mathbf { X } \in \mathbb { R } ^ { M \times N }$ denote a data matrix with M features and N observations. The goal is to represent each observation as a convex combination of K archetypes, while constraining each archetype to lie in the convex hull of the observed data.

Let $\mathbf { C } \in \mathbb { R } ^ { N \times K }$ and $\mathbf { S } \in \mathbb { R } ^ { K \times N }$ be column-stochastic matrices. The archetype matrix is $\mathbf { Z } = \mathbf { X } \mathbf { C } $ , and the reconstructed parameter matrix is $\mathbf { R } = \mathbf { Z } \mathbf { S } = \mathbf { X } \mathbf { C } \mathbf { S }$ . The columns of C define the archetypes as convex combinations of observations, whereas the columns of S define each observation as a convex combination of archetypes.

We estimate C and S by minimizing a negative loglikelihood subject to the constraints that the columns of C and S reside on the standard simplex:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \mathbf { C } , \mathbf { S } } } & { L ( \mathbf { X } , \mathbf { R } ) } \\ { \mathrm { s . t . } } & { \mathbf { R } = \mathbf { X } \mathbf { C S } , } \\ & { c _ { j , k } \geq 0 , \quad s _ { k , j } \geq 0 , } \\ & { \displaystyle \sum _ { j = 1 } ^ { N } c _ { j , k } = 1 , \quad \sum _ { k = 1 } ^ { K } s _ { k , j } = 1 , } \end{array}\tag{1}
$$

The choice of loss function ${ \cal L } ( { \bf X } , { \bf R } )$ corresponds to choice of likelihood used to statistically model the data and allows AA to be adapted to different data types. In particular, likelihood-based formulations provide a principled way to tailor AA to specific observation models, as discussed in [15].

The classical formulation corresponds to a least-squares objective,

$$
L _ { \mathrm { L S } } ( \mathbf { X } , \mathbf { R } ) = \sum _ { i = 1 } ^ { M } \sum _ { j = 1 } ^ { N } \left( x _ { i , j } - r _ { i , j } \right) ^ { 2 } ,\tag{2}
$$

which is equivalent to maximum likelihood estimation assuming independent Gaussian noise.

More generally, by defining L as the negative log-likelihood (NLL), AA can be interpreted as a constrained likelihood maximization problem. This likelihood-based perspective forms the foundation of the optimization framework developed in the following sections. Notably, the problem in eq. (1) is jointly non-convex in (C, S) but convex when optimizing C keeping S) fixed and vice versa. Therefore, we adopt the standard alternating optimization strategy for AA in which C and S are updated iteratively while holding the other fixed until a local maximum is reached.

## B. Quadratic Approximation of the Likelihood

To obtain efficient updates for likelihood-based AA, we approximate the NLL locally by a second-order Taylor expansion. This converts each alternating subproblem into a convex quadratic program under simplex constraints. The resulting quadratic form can then be efficiently optimized using the SMO procedure for the columns of S and a constrained activeset procedure for the columns of C.

For fixed C, the reconstruction can be written as

$$
{ \bf R } = { \bf Z } { \bf S } , \qquad { \bf Z } = { \bf X } { \bf C } ,\tag{3}
$$

and assuming the loss decomposes over observations,

$$
L ( \mathbf { X } , \mathbf { R } ) = \sum _ { j = 1 } ^ { N } \ell ( \mathbf { x } _ { j } , \mathbf { r } _ { j } ) ,\tag{4}
$$

then, when C is fixed, the optimization over S separates into independent column-wise problems, $f ( \mathbf { s } _ { j } )$ . A second-order Taylor expansion around the current iterate gives

$$
f ( \mathbf { s } _ { j } ) \approx \mathrm { c o n s t . } - \mathbf { d } _ { j } ^ { \top } \mathbf { s } _ { j } + \frac { 1 } { 2 } \mathbf { s } _ { j } ^ { \top } \mathbf { H } ^ { ( j ) } \mathbf { s } _ { j } ,\tag{5}
$$

Similarly, we can formulate a similar problem by fixing S and optimizing the columns in C

$$
f ( \mathbf { c } _ { k } ) \approx { \mathrm { c o n s t . } } - \mathbf { d } _ { k } ^ { \top } \mathbf { c } _ { k } + \frac { 1 } { 2 } \mathbf { c } _ { k } ^ { \top } \mathbf { H } ^ { ( k ) } \mathbf { c } _ { k } ,\tag{6}
$$

## C. Updating S through sequential minimal optimization

Originally proposed by [21] and adapted for AA in our related work [1], SMO functions by decomposing large-scale Quadratic Programming (QP) problems into the smallest possible sub-problems. This approach is particularly effective for updating the matrix S, which is typically a low-dimensional dense matrix that represents the data as convex combinations of archetypes.

The SMO framework optimizes archetypes in pairs, updating them iteratively to ensure global convergence. For any given pair of archetypes $k ^ { \prime }$ and $k ^ { \prime \prime }$ , we define a joint weight $t _ { j }$ as $\begin{array} { r } { t _ { j } ~ = ~ \sum _ { d \in \{ k ^ { \prime } , k ^ { \prime \prime } \} } s _ { d , j } } \end{array}$ . To solve the sub-problem, we introduce a weight parameter $\alpha _ { j } ~ \in ~ [ 0 , 1 ]$ . This allows us to re-parameterize the individual weights as $s _ { k ^ { \prime } , j } ~ = ~ t _ { j } \alpha _ { j }$ and $s _ { k ^ { \prime \prime } , j } = t _ { j } ( 1 - \alpha _ { j } )$ . By substituting these into the loss function, the optimization reduces to finding the optimal $\alpha _ { j }$ for each pair, significantly reducing the computational overhead of the original QP problem. Consequently, by substituting these parameters, the loss function $F ( \mathbf { s } _ { j } )$ can be approximated as a quadratic function of $\alpha _ { j }$ :

$$
\begin{array} { r } { F ( \mathbf { s } _ { j } ) \approx c o n s t - \left[ \boldsymbol { d } _ { k ^ { \prime \prime } , j } \right] ^ { \top } \left[ t _ { j } ( 1 - \alpha _ { j } ) \right] + } \\ { \frac 1 2 \left[ t _ { j } \alpha _ { j } \right] ^ { \top } \left[ \boldsymbol { h } _ { k ^ { \prime } , k ^ { \prime } } ^ { ( j ) } \quad \boldsymbol { h } _ { k ^ { \prime } , k ^ { \prime \prime } } ^ { ( j ) } \right] \left[ \begin{array} { c } { t _ { j } \alpha _ { j } } \\ { t _ { j } ( 1 - \alpha _ { j } ) } \end{array} \right] ^ { \top } \left[ \boldsymbol { h } _ { k ^ { \prime \prime } , k ^ { \prime \prime } } ^ { ( j ) } \quad \boldsymbol { h } _ { k ^ { \prime \prime } , k ^ { \prime \prime } } ^ { ( j ) } \right] \left[ \begin{array} { c } { t _ { j } \alpha _ { j } } \\ { t _ { j } ( 1 - \alpha _ { j } ) } \end{array} \right] . } \end{array}\tag{7}
$$

Since the loss function is quadratic with respect to $\alpha _ { j }$ , it produces the following closed-form update:

$$
\alpha _ { j } ^ { \star } = - \frac { t _ { j } h _ { k ^ { \prime } , 2 } ^ { ( j ) } + \sum t _ { j } h _ { k ^ { \prime \prime } , k ^ { \prime } } ^ { ( j ) } - 2 t _ { j } h _ { k ^ { \prime \prime } , k ^ { \prime \prime } } ^ { ( j ) } - 2 d _ { k ^ { \prime } , j } + 2 d _ { k ^ { \prime \prime } , j } } { 2 t _ { j } \big ( h _ { k ^ { \prime } , k ^ { \prime } } ^ { ( j ) } - h _ { k ^ { \prime } , k ^ { \prime \prime } } ^ { ( j ) } - h _ { k ^ { \prime \prime } , k ^ { \prime } } ^ { ( j ) } + h _ { k ^ { \prime \prime } , k ^ { \prime \prime } } ^ { ( j ) } \big ) } .\tag{8}
$$

To maintain the validity of the convex combination, $\alpha _ { j } ^ { \star }$ is restricted to the interval [0, 1]. Given the convexity of the loss function, we apply a simple clipping: if $\alpha _ { j } ^ { \star } > 1$ , it is set to 1, and if $\alpha _ { j } ^ { \star } < 0$ , it is set to 0.

## D. C-update by constrained active-set optimization

For fixed S, we update the archetype construction matrix C. Since each archetype is defined as a convex combination of observations, the columns of C are constrained to lie on the simplex. Notably, only a relative few number of observations are typically here used to define each archetype. We exploit this expected sparsity of the convex combinations using an active-set strategy adapted from fast non-negative least squares [20].

For a single column $\mathbf { c } _ { k } .$ , a second-order Taylor expansion of the likelihood gives the local quadratic approximation that can be seen in Equation 6. To ensure that the Hessian remains full rank and numerically stable, a small regularization term $\frac { 1 } { 2 } \epsilon \sum _ { j } c _ { j k } ^ { 2 }$ is added.

$$
f ( \mathbf { c } _ { k } ) \approx { \mathrm { c o n s t . } } - \mathbf { d } _ { k } ^ { \top } \mathbf { c } _ { k } + \frac { 1 } { 2 } \mathbf { c } _ { k } ^ { \top } ( \mathbf { H } ^ { ( k ) } + \epsilon \mathbf { I } ) \mathbf { c } _ { k } ,\tag{9}
$$

where $\mathbf { d } _ { k }$ and $\mathbf { H } ^ { ( k ) }$ denote likelihood-specific first- and second-order terms with respect to $\mathbf { c } _ { k }$

Rather than imposing the simplex constraint through a quadratic penalty [2], we enforce it directly as an equality constraint within the active set [24]. Let A denote the current active set. The restricted update is obtained by solving the KKT system

$$
\begin{array} { r } { [ \mathbf { H } _ { A , A } ^ { ( k ) } + \epsilon \mathbf { I } \quad \mathbf { 1 } ] [ \mathbf { \overline { { z } } } ] = [ \mathbf { d } _ { A } ] , } \\ { \mathbf { 1 } ^ { \top } \quad \mathbf { 0 } ] [ \mu ] = [ \mathbf { 1 } ] , } \end{array}\tag{10}
$$

where z contains the active coefficients, $\mu$ is the Lagrange multiplier for the sum-to-one constraint, and ϵ is a small ridge term used for numerical stability. Coefficients outside the active set are set to zero, and the active set is updated according to the standard non-negativity conditions.

To control the computational cost in high-dimensional settings, we restrict the active set size to at most $N _ { \mathrm { m a x } }$ . Suppose that, during the update of column K, the current coefficient vector is $\mathbf { c } _ { k } \in \Delta _ { n }$ , where

$$
\Delta _ { n } = \{ \mathbf { c } \in \mathbb { R } ^ { n } : \mathbf { c } \geq 0 , ~ \mathbf { 1 } ^ { \top } \mathbf { c } = 1 \} .\tag{11}
$$

When the active set becomes too large, we replace the current active combination with a single proxy variable. More precisely, we define $\pmb { \theta } ~ = ~ \mathbf { c } _ { k } . \pmb { \theta } ~ \in ~ \Delta _ { n }$ can therefore be seen as a convex combination of the original observations, In data space, this corresponds to introducing the surrogate observation $\begin{array} { r } { \tilde { \mathbf { x } } _ { \pmb { \theta } } = \sum _ { i = 1 } ^ { n } \theta _ { i } \mathbf { x } _ { i } , } \end{array}$ i.e. the observation obtained from the current active convex combination. The active set is then reset and initialized with this surrogate observation with coefficient $c _ { \mathrm { p r o x y } } ~ = ~ 1$ . Subsequent updates optimize using both the original observations and the proxy. If A denotes the set of currently active original observations after the reset, the coefficient vector is represented as

$$
\mathbf { c } _ { k } = \sum _ { i \in A } c _ { i } \mathbf { e } _ { i } + c _ { \mathrm { p r o x y } } \pmb { \theta } .\tag{12}
$$

Since both $\mathbf { e } _ { i }$ and θ lie in the simplex, the simplex constraint is preserved by imposing

$$
\sum _ { i \in A } c _ { i } + c _ { \mathrm { p r o x y } } = 1 , \quad c _ { i } \geq 0 , \quad c _ { \mathrm { p r o x y } } \geq 0 .\tag{13}
$$

The KKT system is updated accordingly. Let

$$
{ \bf B } _ { A } = \left[ { \bf e } _ { i _ { 1 } } \mathrm { ~  ~ \cdot ~ } \cdot \mathrm { ~ \bf ~ e } _ { i _ { | A | } } \theta \right] ,\tag{14}
$$

where $A = \{ i _ { 1 } , \dotsc , i _ { | A | } \}$ , and let

$$
\begin{array} { r } { \pmb { \alpha } = ( c _ { i _ { 1 } } , . . . , c _ { i _ { | A | } } , c _ { \mathrm { p r o x y } } ) ^ { \top } . } \end{array}\tag{15}
$$

Then $\mathbf { c } _ { k } \ = \ \mathbf { B } _ { A } \alpha$ and the restricted quadratic program becomes

$$
\begin{array} { r l } { \underset { \alpha \geq 0 } { \operatorname* { m i n } } } & { { } - \left( \mathbf { B } _ { A } ^ { \top } \mathbf { d } _ { k } \right) ^ { \top } \pmb { \alpha } + \frac { 1 } { 2 } \pmb { \alpha } ^ { \top } \mathbf { B } _ { A } ^ { \top } \left( \mathbf { H } ^ { ( k ) } + \epsilon \mathbf { I } \right) \mathbf { B } _ { A } \pmb { \alpha } } \\ { \mathrm { s . t . } } & { { } \mathbf { 1 } ^ { \top } \pmb { \alpha } = 1 . } \end{array}\tag{16}
$$

The corresponding KKT system is

$$
\left[ \mathbf { B } _ { A } ^ { \top } \left( \mathbf { H } _ { A } ^ { ( k ) } + \epsilon \mathbf { I } \right) \mathbf { B } _ { A } \quad \mathbf { 1 } \right] \left[ \alpha \right] = \left[ \mathbf { B } _ { A } ^ { \top } \mathbf { d } _ { k } \right] .\tag{17}
$$

Equivalently, the proxy contributes one additional row and column to the restricted Hessian, with entries

$$
\mathbf { H } _ { A , \theta } ^ { ( k ) } = \mathbf { H } _ { A , : } ^ { ( k ) } \pmb { \theta } , \quad H _ { \theta , \theta } ^ { ( k ) } = \pmb { \theta } ^ { \top } \mathbf { H } ^ { ( k ) } \pmb { \theta } , \quad d _ { \theta } = \pmb { \theta } ^ { \top } \mathbf { d } _ { k } ,\tag{18}
$$

with the same augmentation applied to the ridge term. Consequently, the proxy is not an unconstrained approximation, it is an additional simplex-preserving surrogate observation whose effect on the quadratic subproblem is computed from the original likelihood expansion.

## E. Likelihood specific derivatives

For all models, updates follow a generalized structure using the partial derivatives of the loss with respect to the model parameters S and C via the chain rule (tables I and II).

When updating the matrix S for a fixed C, we define the Jacobian term $\begin{array} { r } { \alpha _ { i , k } ^ { \mathrm { ~ ~ } } = \frac { \partial r _ { i , j } } { \partial s _ { k , j } } = \sum _ { m } p _ { i , m } c _ { m , k } , } \end{array}$ , with

$$
p _ { i , j } = { \left\{ \begin{array} { l l } { x _ { i , j } , } & { { \mathrm { ( F r o b e n i u s ) } } } \\ { x _ { i , j } + \varepsilon , } & { { \mathrm { ( P o i s s o n ) } } } \\ { x _ { i , j } { \mathrm { ( 1 - 2 } \varepsilon ) } + \varepsilon , } & { { \mathrm { ( B e r n o u l l i ) } } } \end{array} \right. }\tag{19}
$$

or

$$
p _ { i , j , p } = x _ { i , j , p } \big ( 1 - \varepsilon \big ) + \frac { \varepsilon } { L - 1 } \big ( 1 - x _ { i , j , p } \big ) , \quad \mathrm { ( M u l t i n o m i a l ) } \quad\tag{20}
$$

The full update direction $d _ { k , j }$ incorporates the partial derivative and an adjustment term:

$$
d _ { k , j } = \frac { \partial L } { \partial s _ { k , j } } - \sum _ { k ^ { \prime } , k ^ { \prime \prime } } h _ { k ^ { \prime } , k ^ { \prime \prime } } ^ { ( j ) } s _ { k , j }\tag{21}
$$

where the partial derivative $\frac { \partial L } { \partial s _ { k , j } }$ and the Hessian-diagonal approximation $\begin{array} { r } { h _ { k ^ { \prime } , k ^ { \prime \prime } } ^ { ( j ) } = \frac { \partial ^ { 2 } L ^ { ' } } { \partial s _ { k ^ { \prime } , i } \partial s _ { k ^ { \prime \prime } , i } } } \end{array}$ are defined respectively for the Gaussian, Bernoulli, Poisson, and Multinomial likelihoods in table I and II.

For the update of matrix C with fixed S, we define the Jacobian term $\begin{array} { r } { \beta _ { i , j , j ^ { \prime } , k } = \frac { \partial r _ { i , j ^ { \prime } } } { \partial c _ { i , k } } = p _ { i , j } s _ { k , j ^ { \prime } } } \end{array}$

The update direction $d _ { j , k }$ is defined as:

$$
d _ { j , k } = \frac { \partial L } { \partial c _ { j , k } } + \sum _ { j ^ { \prime } , j ^ { \prime \prime } } h _ { j ^ { \prime } , j ^ { \prime \prime } } ^ { ( k ) } c _ { j , k }\tag{22}
$$

with the respective partial derivatives and Hessian approximations $\begin{array} { r } { \boldsymbol { h } _ { j ^ { \prime } , j ^ { \prime \prime } } ^ { ( k ) } = \frac { \dot { \partial ^ { 2 } } L } { \partial c _ { j ^ { \prime } , k } \partial c _ { j ^ { \prime \prime } , k } } } \end{array}$ provided in Table I and II.

TABLE I: Likelihood-specific gradient and Hessian factors.
<table><tr><td>Likelihood</td><td>Gradient factor gi, j</td><td>Hessian factor</td><td> $w _ { i , j }$ </td></tr><tr><td>Gaussian</td><td> $- 2 ( x _ { i , j } - r _ { i , j } )$ </td><td>2</td><td></td></tr><tr><td>Bernoulli</td><td> $- \left( \frac { x _ { i , j } } { r _ { i , j } } - \frac { 1 - x _ { i , j } } { 1 - r _ { i , j } } \right)$ </td><td></td><td> $\frac { x _ { i , j } } { r _ { i , j } ^ { 2 } } + \frac { 1 - x _ { i , j } } { ( 1 - r _ { i , j } ) ^ { 2 } }$ </td></tr><tr><td>Poisson</td><td> $\mathrm { ~ - ~ } \left( \frac { x _ { i , j } } { r _ { i , j } } - 1 \right)$ </td><td> $\underline { { \boldsymbol { x } } } _ { i , j }$  r.2 i,j</td><td></td></tr><tr><td>Multinomial</td><td> $- { \frac { x _ { i , j , p } } { r _ { i , j , p } } }$ </td><td>xi,j,p  $r _ { i , j , p } ^ { 2 }$ </td><td></td></tr></table>

For the Multinomial likelihood, replace the feature index j by the feature-category pair (j, p).

## F. Predictive Likelihood

Let $\mathbf { X } \in \mathbb { R } ^ { M \times N }$ denote a dataset with M features and N samples. For a fixed number of archetypes K, the model is fitted on training data $\mathbf { X } _ { \mathrm { t r a i n } } \in \mathbb { R } ^ { M \times N _ { \mathrm { t r a i n } } }$ , providing coefficient

ds

TABLE II: Generic S- and C-update components using the likelihood-specific factors from Table I.
<table><tr><td></td><td>Update Partial derivative</td><td>Hessian term</td></tr><tr><td rowspan="2">S</td><td> $\frac { \partial L } { \partial s _ { k , j } } = \sum _ { i } g _ { i , j } \alpha _ { i , k }$ </td><td> $h _ { k ^ { \prime } , k ^ { \prime \prime } } ^ { ( j ) } = \sum _ { i } w _ { i , j } \alpha _ { i , k ^ { \prime } } \alpha _ { i , k ^ { \prime \prime } }$ </td></tr><tr><td></td><td></td></tr><tr><td>C</td><td> $\frac { \partial L } { \partial c _ { j , k } } = \sum _ { i , \ell } g _ { i , \ell } \beta _ { i , j , \ell , k }$  For the Multinomial likelihood, use</td><td> $h _ { j ^ { \prime } , j ^ { \prime \prime } } ^ { ( k ) } = \sum _ { i , \ell } w _ { i , \ell } \beta _ { i , j ^ { \prime } , \ell , k } \beta _ { i , j ^ { \prime \prime } , \ell , k }$   $\begin{array} { r } { \alpha _ { i , k , p } = \sum _ { m } p _ { i , m , p } c _ { m , k } } \end{array}$  and</td></tr><tr><td colspan="3"> $\beta _ { i , j , j ^ { \prime } , k , p } = p _ { i , j , p } s _ { k , j ^ { \prime } }$  The Hessian terms are used by the quadratic-approximation updates; the likelihood-based PCHA baseline uses only the corresponding gradients.</td></tr></table>

matrices C and S. The fitted archetypes and reconstruction are respectively given by

$$
{ \bf Z } = { \bf X } ^ { \mathrm { t r a i n } } { \bf C } , \qquad { \bf R } ^ { \mathrm { t r a i n } } = { \bf Z } { \bf S } ,\tag{23}
$$

where the columns $\mathbf { r } _ { j } = ( \mathbf { R } ^ { \mathrm { t r a i n } } ) _ { \cdot j }$ define the reconstruction associated with the $j ^ { \flat }$ training sample.

To compare different values of K, we evaluate how well the fitted archetypal representation Z based on $\mathbf { X } ^ { \mathbf { t r a i n } }$ predicts held-out observations. For a test observation x, we approximate the predictive distribution by the empirical mixture over the reconstructed training parameters,

$$
\begin{array} { r } { p ( \mathbf { x } ^ { \mathrm { t e s t } } \mid \mathbf { X } ^ { \mathrm { t r a i n } } , \mathbf { Z } ) = \displaystyle \int p ( \mathbf { x } ^ { \mathrm { t e s t } } \mid \mathbf { Z } , \mathbf { s } ) p ( \mathbf { s } \mid \mathbf { X } ^ { \mathrm { t r a i n } } , \mathbf { Z } ) \ : \boldsymbol { \mathfrak { c } } } \\ { \approx \frac { 1 } { N _ { \mathrm { t r a i n } } } \displaystyle \sum _ { j = 1 } ^ { N _ { \mathrm { t r a i n } } } p ( \mathbf { x } ^ { \mathrm { t e s t } } \mid \mathbf { r } _ { j } ^ { \mathrm { t r a i n } } ) . } \end{array}\tag{24}
$$

The posterior is according to Bayes’ theorem given by

$$
p ( \mathbf { s } \mid \mathbf { X } ^ { \mathrm { t r a i n } } , \mathbf { Z } ) = \frac { p ( \mathbf { X } ^ { \mathrm { t r a i n } } | \mathbf { Z } , \mathbf { s } ) p ( \mathbf { s } ) } { \int p ( \mathbf { X } ^ { \mathrm { t r a i n } } | \mathbf { Z } , \mathbf { s } ) p ( \mathbf { s } ) d \mathbf { s } } .\tag{25}
$$

Whereas $p ( { \bf X } ^ { \mathrm { t r a i n } } | { \bf Z } , { \bf s } )$ is the likelihood, the prior $p ( \mathbf { s } )$ can be any distribution with support over the probability simplex such as the Dirichlet distribution or continuous categorical distribution [19]. However, the marginalization over s is in general intractable preventing the direct evaluation of the predictive likelihood. By imposing a uniform prior over the simplex corresponding to $p ( \mathbf { s } ) = { \mathrm { D i r i c h l e t } } ( \mathbf { s } | \mathbf { 1 } )$ the shape of the posterior distribution is only influenced by the likelihood. Consequently, $\begin{array} { r c l } { p ( \mathbf { s } } & { | } & { \mathbf { X } ^ { \mathrm { t r a i n } } , \mathbf { Z } ) } \end{array} \propto \begin{array} { r c l } { p ( \mathbf { X } ^ { \mathrm { t r a i n } } } & { | } & { \mathbf { Z } , \mathbf { s } ) } \end{array}$ and by using the training samples to approximate the integral assuming that the learned $\mathbf { S } ^ { \mathrm { t r a i n } }$ are drawn unbiased from the posterior distribution $\mathbf { s } _ { j } ^ { \mathrm { t r a i n } } \ \sim \ p ( \mathbf { s } \ \mid \ \mathbf { X } ^ { \mathrm { t r a i n } } , \mathbf { Z } )$ under this uniform prior we obtain when using ${ \bf r } _ { j } ^ { \mathrm { t r a i n } } ~ = ~ { \bf Z } { \bf s } _ { j } ^ { \mathrm { t r a i n } }$ the approximation of the predictive likelihood in eq. (24). Thus, for a test set $\mathbf { X } _ { \mathrm { t e s t } } = \{ x _ { i } \} _ { i = 1 } ^ { N _ { \mathrm { t e s t } } }$ , the predictive log-likelihood is

$$
\mathcal { L } _ { \mathrm { p r e d } } = \sum _ { i = 1 } ^ { N _ { \mathrm { t e s t } } } \log \left[ \frac { 1 } { N _ { \mathrm { t r a i n } } } \sum _ { j = 1 } ^ { N _ { \mathrm { t r a i n } } } p ( x _ { i } \mid r _ { j } ) \right] .\tag{26}
$$

For numerical stability, each term is evaluated using

$$
\log p ( \mathbf { x } _ { i } \mid \mathbf { r } _ { j } ) = \log \operatorname { s u m e x p } _ { j } \left\{ \log p ( \mathbf { x } _ { i } \mid \mathbf { r } _ { j } ) \right\} - \log N _ { \operatorname { t r a i n } } .\tag{27}
$$

TABLE III: Log-likelihood contributions used in the predictive likelihood criterion.
<table><tr><td>Model</td><td> $\underline { { \overline { { \log p ( \mathbf { x } \mid \mathbf { r } _ { j } ) } } } }$ </td></tr><tr><td>Bernoulli</td><td>M  $\sum _ { f = 1 \atop { \bf \Phi _ { \pm } } } ^ { \cdots } [ x _ { m } \log r _ { m j } + ( 1 - x _ { m } ) \log ( 1 - r _ { m j } ) ]$ </td></tr><tr><td>Poisson</td><td>M  $\sum _ { m = 1 } \left[ x _ { m } \log r _ { m j } - r _ { j m } \right]$ </td></tr><tr><td>Gaussian</td><td> $- \frac { 1 } { 2 } \left( \frac { \| \mathbf { x } - \mathbf { r } _ { j } \| _ { 2 } ^ { 2 } } { \sigma ^ { 2 } } + d \log ( 2 \pi ) + d \log \sigma ^ { 2 } \right)$ </td></tr><tr><td>Multinomial</td><td> $\sum _ { m = 1 } ^ { M } x _ { m } \log r _ { m j }$ </td></tr></table>

The likelihood contribution log $p ( \mathbf { x } \mid \mathbf { r } _ { j } )$ depends on the observation model. The models used in this work are summarized in Table III. Terms independent of $\mathbf { r } _ { j }$ are omitted for the Poisson and multinomial models, since they do not affect model comparison across values of K.

For the Gaussian model, we further need to specify the isotropic covariance $\mathbf { x } \sim \mathcal { N } ( \mathbf { r } _ { j } , \sigma ^ { 2 } \mathbf { I } _ { M } )$ , which we estimate by maximum likelihood from the training residuals as

$$
\sigma ^ { 2 } = \frac { 1 } { M N _ { \mathrm { t r a i n } } } \left. \mathbf { X } _ { \mathrm { t r a i n } } - \mathbf { R } _ { \mathrm { t r a i n } } \right. _ { F } ^ { 2 } .\tag{28}
$$

In F-fold cross-validation, the predictive score for each candidate number of archetypes is obtained by summing the held-out predictive log-likelihoods across folds,

$$
\operatorname { C V } ( K ) = \sum _ { f = 1 } ^ { F } \mathcal { L } _ { \mathrm { p r e d } } ^ { ( f ) } ( K ) , \quad K ^ { \star } = \arg \operatorname* { m a x } _ { K } \operatorname { C V } ( K ) .\tag{29}
$$

## G. Normalized Mutual Information

Normalized Mutual Information (NMI) is used together with the predictive likelihood to evaluate the solution. This was first introduced to AA in [6]. The method evaluates the columns in S as probability distributions and computes the consistency across different initializations of S. A high NMI indicates a stable solution whereas a low NMI indicates an unstable solution. The NMI between two solutions $\mathbf { S } ^ { r }$ and $\mathbf { S } ^ { r ^ { \prime } }$ is given by:

$$
\mathrm { N M I } ( \mathbf { S } ^ { r } , \mathbf { S } ^ { r ^ { \prime } } ) = \frac { 2 \mathrm { M I } ( \mathbf { S } ^ { r } , \mathbf { S } ^ { r ^ { \prime } } ) } { \mathrm { H } ( \mathbf { S } ^ { r } ) + \mathrm { H } ( \mathbf { S } ^ { r ^ { \prime } } ) } ,\tag{30}
$$

where the mutual information is defined as

$$
\operatorname { M I } ( \mathbf { S } ^ { r } , \mathbf { S } ^ { r ^ { \prime } } ) = \sum _ { k = 1 } ^ { K } \sum _ { k ^ { \prime } = 1 } ^ { K } p ( k , k ^ { \prime } ) \log \frac { p ( k , k ^ { \prime } ) } { p ( k ) p ( k ^ { \prime } ) } ,\tag{31}
$$

with

$$
\begin{array} { c } { { p ( k , k ^ { \prime } ) = \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { k , n } ^ { r } s _ { k ^ { \prime } , n } ^ { r ^ { \prime } } , } } \\ { { p ( k ) = \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { k , n } ^ { r } , \quad p ( k ^ { \prime } ) = \displaystyle \frac { 1 } { N } \sum _ { n = 1 } ^ { N } s _ { k ^ { \prime } , n } ^ { r ^ { \prime } } . } } \end{array}\tag{32}
$$

The entropy is given by

$$
\mathrm { H } ( \mathbf { S } ^ { r } ) = - \sum _ { k = 1 } ^ { K } p ( k ) \log p ( k ) .\tag{33}
$$

## H. Computational complexity and memory usage.

The alternating algorithm consists of an S-update and a C-update. In the S-update, each column ${ \bf s } _ { j }$ lies in the $K \mathfrak { - }$ dimensional simplex and is updated using pairwise SMO steps. A full sweep over all archetype pairs costs $\mathcal { O } ( N K ^ { 2 } )$ once the likelihood-specific gradients and Hessian terms have been formed. Constructing these quantities typically costs O(MNK). The main scalability challenge occurs in the C-update, since each column $\mathbf { c } _ { k }$ lies in an N-dimensional simplex. A na¨ıve active-set method may require storing and solving systems involving large subsets of the N observations, leading to memory costs as high as $\mathcal { O } ( N ^ { 2 } )$ if the full Hessian is materialized. We avoid this by imposing a maximum activeset size $N _ { \mathrm { m a x } }$ . The reduced solve then involves at most $N _ { \mathrm { m a x } }$ explicit active variables, giving a per-solve cost of $\mathcal { O } ( N _ { \mathrm { m a x } } ^ { 3 } )$ and reduced Hessian storage of $\mathcal { O } ( N _ { \mathrm { m a x } } ^ { 2 } )$ . When the active set reaches $N _ { \mathrm { m a x } }$ , its current contribution is condensed into a proxy vector. This proxy preserves the accumulated convex combination while resetting the explicit active set. Consequently, the optimization can continue to explore new active observations without allowing the reduced system to grow with N. The max-active mechanism therefore changes the practical memory requirement of the C-update from being governed by the full sample size N to being governed primarily by $N _ { \mathrm { m a x } }$ , while still representing the full coefficient vector through the combination of explicit active entries and the proxy component.

## I. Extension of PCHA to discrete data

In addition to introducing a new optimization scheme, we extended the PCHA algorithm [5] to discrete data by deriving likelihood-consistent gradient expressions for Bernoulli, Poisson and Multinomial distributed observations. This extension enables principled probabilistic modeling within the established PCHA architecture and facilitates direct empirical comparison with this procedure in the experimentation (see gradient terms in tables I and II. for specific C and S steps).

## III. RESULTS

We first present the optimization behavior analyzed on synthetic data, where the true number of archetypes is known and subsequently on three real datasets.

## A. Reproducibility

All experiments were run from the DTU hpc (The Technical University of Denmark’s high performance computer) queue as single-core CPU jobs, with identical resource requests within each experiment. The implementation used in this paper is available in the Distribution Agnostic Archetypal Analysis repository [25].

Unless stated otherwise, models were evaluated over five random seeds (0 − 4) using 5-fold predictive cross-validation.

Synthetic experiments used data seed 0, $K ^ { \star } ~ = ~ 5 ,$ candidate $K \in \{ 2 , 3 , 4 , 5 , 6 , 8 , 1 0 , 1 5 \}$ , DAAA used pairwise twocoordinate SMO the active-set cap was set to 25% unless explicitly stated. For direct method comparisons, fold splits and the initial values of C and S were controlled by using the same random seeds across methods. All methods were initialized using five seeds, (0 − 4). The initial matrices were generated from uniformly sampled random values, transformed using the absolute logarithm, and subsequently column-normalized.

For DAAA and PCHA, additional sparsity was imposed on C by setting all entries below 0.5 to zero. This sparsification step was omitted for PAA [15], as PAA was substantially more sensitive to initialization and did not perform reliably under this sparse initialization.

## B. Synthetic Data Experiments

The aim of the synthetic data experimentation was to analyze the feasibility and recoverability of the proposed framework. All synthetic data sets were generated with $K \ = \ 5$ archetypes and were fitted with candidate values $K ~ \in ~ \{ 2 , 3 , 4 , 5 , 6 , 8 , 1 0 , 1 5 \}$ . For each sample n, mixing weights were drawn from a Dirichlet distribution, $h _ { n } \sim$ Dirichlet $( \alpha { \bf 1 } _ { K ^ { * } } ) , \alpha = 0 . 3$ , such that the observations were concentrated near a small number of archetypes. for the Gaussian, Bernoulli, and Poisson experiments, archetype profiles were drawn as sparse binary vectors, $\eta _ { m k } \sim \mathrm { B e r n o u l l i } ( s ) , s =$ 0.3, with duplicate archetypes rejected. Conditional on these archetypes and the sample weights, observations were generated as, $x _ { m , n } = \eta _ { m } ^ { \top } h _ { n } + \epsilon _ { m , n } , \epsilon _ { m , n } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) , \sigma = 0 . 1$ for the Gaussian case. $x _ { m , n } \sim$ Bernoulli min $\{ \eta _ { m } ^ { \top } h _ { n } , 0 . 9 9 \} \}$  for the Bernoulli case and $x _ { m , n } \sim \mathrm { P o i s s o n } \left( r \eta _ { m } ^ { \top } h _ { n } \right) , r = 5$ for Poisson. For the multinomial experiment, each archetypefeature pair was instead assigned a categorical probability vector, $\eta _ { m k } \sim$ Dirichlet $( s { \bf 1 } _ { C } ) , C ~ = ~ 1 0 { \ : } _ { \mathrm { \scriptsize { ; } } }$ , and the samplespecific category probabilities were formed by convex mixing, $\begin{array} { r } { p _ { m , n } \ = \ \sum _ { k = 1 } ^ { K _ { 0 } } h _ { n , k } \eta _ { m , k } } \end{array}$ . The observed categorical vector was then drawn from a one-trial Multinomial distribution, $x _ { m j } \sim \mathrm { M u l t i n o m i a l } ( 1 , p _ { m j } )$ . All experiments used $N = 1 0 0 0$ samples and $M = 5 0 0$ features.

In this simple setting, all methods trivially recovered K<sup>⋆</sup>. From Figure 1, we observe the performance of PAA, PCHA, and DAAA relative to DAAA across four data-generating distributions when specifying the number of estimated components to $k \in 5 , 1 0$ , 15. All three methods reach essentially the same final loss in every setting: the loss ratio remains within ±2%, confirming that the choice of algorithm does not compromise solution quality.

The computational picture is more varied. At K = 5, PAA is substantially slower than DAAA especially for Gaussian and Bernoulli data. PCHA is dramatically faster on Gaussian data but slower for Poisson and Multinomial likelihoods. This advantage for PCHA on Gaussian data narrows as K increases; at $K = 1 0$ and $K = 1 5$ , PCHA is up to 4× slower than DAAA on Multinomial data. PAA’s relative cost also decreases with K: by $K \ = \ 1 5 \ \mathrm { i t }$ is comparable $^ { \mathrm { t o , } }$ or faster than, DAAA on Poisson and Multinomial data, while remaining slower on Gaussian and Bernoulli data. DAAA thus occupies a competitive position across all distributions and archetype counts.

![](images/f4f4430b54dfdc3c827755e97429761a1e9786f959a4072dc4ec56584e3990ff.jpg)  
Fig. 1: Cross-distribution robustness of synthetic data, for $K \in \{ 5 , 1 0 , 1 5 \}$ , with $K ^ { \star } = 5 .$ . Each point shows the average performance of PAA, PCHA, and DAAA across repeated initializations for a fixed synthetic data setting. Runtime (seconds) and final loss (NLL) are reported relative to DAAA within each distribution, so values above one indicate slower runtime or higher loss than DAAA. NMI measures stability across random initializations, with larger values indicating more consistent recovered archetypal representations.

Optimization stability, measured as the pairwise NMI between solutions from different initializations on the same dataset declines with K for all methods. At K = 5 stability is near-perfect (NMI ≈ 1) across all distributions, whereas at $K = 1 5$ it drops to approximately 0.85–0.90 for Gaussian data and as low as 0.60–0.65 for Multinomial data. All three methods exhibit comparable NMI at every setting, indicating that the decreasing stability reflects the geometry of the problem $( K ^ { \star } = 5 )$ rather than differences between optimizers.

To test the feasibility of the constrained active set we examined the loss and convergence speed relative to the size of the active set in Figure 2. Here we see that smaller active sets achieve faster early descent across all distributions. For Gaussian, Binary, and Counts data, fractions ≥ 5% converge to a final loss indistinguishable from the full active set. For Multinomial data, fractions ≥ 25% suffice, while 1% and 5% incur a persistent gap.

The proposed predictive log-likelihood framework for determining the optimal K was tested in a Gaussian setting for simplicity (noise is easy to model and control through the variance σ). In Figure 3 it can be seen that while the Variance Explained and NMI gradually increases after the optimal K the predictive log-likelihood declines, showing a usable framework for quantitative analysis of K. This signal becomes clearer as more noise is added to the data generation process.

![](images/4338af42a4fa2e77682d1346836fedb3a00164cb265973eec4c892d90c2ea1b9.jpg)  
Fig. 2: Convergence of Archetypal Analysis under varying active set sizes. Training loss as a function of wall-clock time for active-set fractions $N _ { \mathrm { m a x } } \quad \in$ {1%, 5%, 10%, 25%, 50%, 75%, 100%}. Where 100% corresponds to using all N samples. Shaded bands show mean ± std across five random seeds. The x-axis is on log scale.

## C. Real Data experiments

Having established the behavior of the method on synthetic data, we next evaluate it on three real data sets chosen for the three discrete data based likelihoods implemented in the framework. The purpose of these experiments is twofold. We first ask whether the likelihood-based model objective gives competitive fits, stable solutions, and useful predictive likelihoods on data sets whose structure is clearly non-Gaussian. We then inspect whether the resulting archetypes are interpretable within their domain. We consider the following three domains; binary data based on blood cancer mutations, count data based on gene programs and cell types for single-cell RNA sequencing, and categorical data based on microbiome profiles across body sites, these datasets are described below.

The binary Bernoulli likelihood based experiment uses the MSK-IMPACT Heme cohort [26], a targeted sequencing dataset of hematologic malignancies. We construct a binary sample-by-gene somatic mutation matrix from the single nucleotide Variant (SNV) or insertion/deletion (indel), i.e. SNV/indel calls, setting an entry to one if a sample contains at least one somatic SNV or indel in the corresponding gene and zero otherwise.

The count Poisson likelihood based experiment uses the PBMC3k experiment uses the 10x Genomics peripheral blood mononuclear cell data set available through scanpy [27], [28]. After standard filtering, we retain the $m = 2 0 0 0$ highly variable genes while fitting the model to the original integer

Number of archetypes k

![](images/68268eb3f00b9baf29f1b64165b7112d02d638bc99226f9feb40255309b70bd9.jpg)

![](images/d3510f75f97164eb100e2ade577eca088c4ae1b95455c62193c84012190a8f5a.jpg)  
Fig. 3: Model selection for Gaussian archetypal analysis on synthetic data $( K ~ = ~ 5$ true archetypes). Left: Cross-validated predictive log-likelihood (Pred. LL), Gaussian training log-likelihood (Gaussian LL), and variance explained (VE) as functions of K at a single noise level. The predictive LL peaks at the true $K ^ { * } ~ = ~ 5$ (dashed line), while Gaussian LL and VE increase monotonically, failing to identify the correct model complexity (inset). Right: Predictive LL normalized by its maximum across noise levels $\sigma \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \}$ ; shaded bands show variance across seeds. The predictive LL is maximised at $K ^ { \star } ~ = ~ 5$ across all noise levels, and the penalty for overspecifying K grows with σ, reflecting greater sensitivity to overfitting under high noise.

count matrix. This data set is therefore treated as count-valued and modeled with the Poisson likelihood. The fitted archetypes are interpreted as extreme gene-expression profiles, and the sample memberships are compared with known PBMC celltype annotations in the downstream diagnostic plot in Figure 6.

The categorical multinomial likelihood based experiment uses a Human Microbiome Project Phase 1 16S tensor aggregated at genus level [29]. The tensor has taxa, subjects, and body sites as its three axes; in the checked-in preprocessing report this corresponds to 589 taxa, 235 subjects, and 18 body sites. Since each subject-taxon slice describes a composition over body sites, this data set is fitted with the multinomial likelihood. The resulting archetypes are intended to capture recurrent microbial profiles whose mass is concentrated on different anatomical sites or combinations of sites.

For each data set, we fit DAAA, PAA, and PCHA over a grid of candidate numbers of archetypes (2 − 50) and random initialization seeds (0−4). We report the final training objective, wall-clock fitting time measured in seconds, and pairwise soft NMI between membership matrices across seeds as a measure of optimization stability. We also compute fivefold predictive log-likelihood. We use this as primary criterion to select the number of archetypes as the real data do not come with known ground-truth values of K.

From Figure 4 we observe that DAAA compared to PAA generally produced more stable solutions and better likelihood fits, suggesting that the proposed optimization strategy is better suited for discrete real-world data, while the fit time was generally more favorable for DAAA than PCHA both having similar solution quality. From the predictive log-likelihood plots we saw that model performance were similar for the DAAA and PCHA approaches however, we observe discrepancy to PAA in particular for the MSK Heme mutation data where overfitting was less of an issue for PAA. From the figure we further observe that across the three real-world datasets, the proposed DAAA framework selected different model complexities that reflected the structure of each domain: (K=10) for the MSK Heme mutation data, (K=40) for the PBMC3k single-cell RNA-seq data, and (K=2) for the HMP microbiome data. In each case, the predictive likelihood criterion identified archetypes that were biologically interpretable, including cancer-type-specific mutation profiles as illustrated in Figure 5. These profiles (archetypes) split into three distinct main categories; one driven by AML, one by MDS and one B-NHL. These correspond to myeloid and lymphoid types of blood cancer, with AML and MDS representing myeloid malignancies and B-NHL representing a lymphoid malignancy. Distinct expression patterns across gene groups with archetypes primarily described by certain PBMC cell-types as illustrated in Figure 6, and microbiome variation driven largely by body-site and sex-associated composition as illustrated in Figure 7.

![](images/11dfe9e9e516b4bfeba2b9d564dcfdd080a06682ddec0eed7a8d92efe170e0dc.jpg)  
Fig. 4: Real-data benchmarks across three datasets and distributions. Each column corresponds to one dataset (MSK Heme, Bernoulli; PBMC 3k, Poisson; HMP microbiome, Multinomial) and each row to one metric: training loss (NLL), cross-validated predictive log-likelihood (star marks the DAAA selected K), stability NMI across random initialisations, and wall-clock fit time. DAAA uses max\_active = 25% of samples. PCHA and PAA use the full dataset. Shaded bands denote ±1 standard deviation across five random seeds.

![](images/643bcbc3408c492e598dc4c32f06e37621c1dda378d68024b918f707e955bcb0.jpg)  
Fig. 5: Interpretability of learned archetypes on the MSK-IMPACT Heme somatic mutation cohort. Each row corresponds to one of $K \ = \ 1 0$ archetypes. Bar colors show the fraction of hard-assigned patients from each cancer type group. Archetypes are sorted by dominant lineage, indicated by the colored strip and vertical label on the left. The four most enriched mutations per archetype, the largest positive deviations from the dataset, wide mean mutation rate, are listed to the right. Without using cancer type labels during fitting, the method recovers clinically coherent structure: the two AML archetypes are defined by chromatin-regulator mutations (ASXL1, RUNX1) and the canonical NPM1/FLT3 driver pair; the MDS archetypes are characterised by DNA-methylation genes (DNMT3A, TET2, $\mathrm { I D H 1 } / 2 ) ;$ and five distinct B-NHL archetypes separate known molecular subtypes, including an ABC-DLBCL profile (MYD88, CD79B) and a follicular lymphoma profile (BCL2, EZH2, CREBBP). Archetype $A _ { 1 0 }$ is a diffuse, mixed-lineage profile with no single dominant mutation signal, reflecting the shared background mutation burden across all subtypes.

## IV. DISCUSSION

The synthetic study demonstrated similar performance across methods for AA inference for the different distributions (Figure 1), notably no methods were best across all the synthetic generated data in terms of fit time. DAAA worked in general more favorably for discrete data than the PCHA. However, PAA was more computationally efficient for the Poisson and Multinomial when the model was overparameterized $( K > K ^ { \star } )$ .

Notably, the constrained active set procedure enhanced convergence with minimal compromise in solution quality, supporting the utility of the proposed active set scaling. Although this scaling was evaluated here in the context of AA, we expect that similar gains may be achievable in other modeling regimes that rely on active set procedures.

For the real data, we observe poor performance from the PAA method across all regimes when compared to both the DAAA and PCHA in terms of solution quality (NLL) and stability (NMI). Whereas PCHA and DAAA here had similar performance the PCHA was notably slower than the proposed DAAA. Again, we observe that while the PAA was slower for Bernoulli, it had favorable fit time for the multinomial data. For Poisson the PAA only had favorable fit time in higher model order regimes. The poorer log-likelihood obtained by PAA should most naturally be interpreted as an optimization issue, rather than as evidence against the archetypal representation itself. Unlike PCHA and DAAA, PAA relies on multiplicative updates. Such updates preserve positivity and are monotone under suitable conditions, but they suffer from several limitations, small coefficients tend to remain small, mass is reallocated only gradually, and convergence can be slow when the likelihood is ill-conditioned or close to the boundary of the parameter space. This is particularly consequential for Bernoulli, Poisson, and multinomial likelihoods, where underestimating rare events, zero counts, or low-probability categories can have a large effect on the log-likelihood. Thus, PAA may capture the dominant archetypal structure while still producing less well-calibrated fitted probabilities or intensities, leading to substantially lower likelihood values than PCHA and DAAA.

![](images/176e7e45efbc6196dad177906100084efde790d3b181e17dfb48e89bc51c4519.jpg)  
Fig. 6: Within cell type gene groups across $K \ = \ 4 0$ archetypes (PBMC3k scRNA-seq) Dot plot of gene expression across 40 archetypes fitted by DAAA to the PBMC3k dataset (2,700 peripheral blood mononuclear cells, 2,000 highly variable genes. Archetypes (A1–A40, columns) are grouped by dominant cell type (coloured bar) and sorted by decreasing cell-type purity within each group. Genes (rows) are selected as the top drivers of within-cell-type expression variance across sub-archetypes. Dot size (square-root scaled, normalised per gene) encodes within-cell-type enrichment, the archetype mean expression minus the mean across all archetypes of the same cell type, isolating sub-state variation from the shared cell-type background. Dot colour encodes absolute expression as log(1 + mean raw count), compressing the dynamic range of integer counts. CD4 T cells (A1–A12) resolve into 12 sub-archetypes spanning a translational-activity gradient (MALAT1, RPL13), an MHC-I axis (B2M), and a mitochondrial-high state (MT-CO1); $\mathrm { C D 1 4 ^ { + } }$ monocytes (A13–A23) separate into LYZ-high classical and FTL/FTH1- enriched sub-states

Importantly, we proposed generalization as estimated based on our predictive log-likelihood framework as a way to tackle the open problem of determining the number of archetypes. For the synthetic data, we observe that the method correctly identifies the model order with the peak in generalization error enhanced as the noise level increased and where overparameterization should be more influenced by noise, see Figure 3. This points to the validity and practical use of the predictive likelihood framework for model order estimation in AA. Notably for the real data we also observed peaks in the generalization error indicating a suitable model order for the three datasets respectively modeled using the Bernoulli, Poisson, and multinomial likelihoods. When inspecting the associated identified models by the DAAA, as indicated by a star in Figure 4, we observed meaningful extracted representations in which the AA procedures extracted distinct patterns of variation in the datasets.

![](images/69045b6a5c7fdcc96737f2b2c417bda5e1eb875f41ae9f7a0adcecf17fda8d68.jpg)  
Fig. 7: Multinomial Archetype microbiome profiles across four anatomical body-site groups (235 subjects, 18 body sites, 589 genera). (Left) Each bar shows the mean relative abundance of the dominant genera, averaged across all body sites within the group (Gut: 1 site; Oral: 9 sites; Skin: 5 sites; Vaginal: 3 sites). Archetype profiles are computed as the anchor-weighted mean composition $\textstyle \sum _ { i } C _ { i k } X _ { i j p }$ , then averaged over sites within each group. A1 and A2 are nearly indistinguishable at gut and oral sites. Skin shows a moderate enrichment of Propionibacterium in A1 and Staphylococcus in A2. Vaginal sites show the main difference: A2 is 90% Lactobacillus on average, compared to 11% in A1. (Right) Histogram of A2 membership weights by sex (dashed lines: group means). Male subjects cluster near zero (mean 0.24) and female subjects near one (mean 0.86), demonstrating that the model unsupervised recovers sex as the dominant axis of variation through the vaginal Lactobacillus signal.

Whereas we presently focus on the modeling of discrete data we note that the proposed optimization- and generalization framework for estimating the AA model and model order respectively, can be used beyond the considered likelihoods.

## A. Limitations

While the proposed DAAA framework exhibits stable solutions across archetypes, distributions and datasets AA remains inherently non-convex and initialization sensitive. Similarly the method depends on data following a specific likelihood formulation and focuses only on separable likelihoods. This framework allows for the extension to heterogeneous likelihood formulations, where each data feature can have different distributions, however this is left for future work. In this paper, we introduced a constrained active set size to improve scalability. From Figure 2 this proved to a be feasible and competitive solution for AA that can favorably trade exactness for scalability, especially if the limit to the active set is very restrictive. As such, we observed favorable fit time when comparing to PCHA while achieving similar solution qualities. However the constrained active set comes with a cost in numeric precision and should not be used when such numeric solution precision is highly critical.

Although the predictive log-likelihood gave clear indications about the optimal K, this was only qualitatively assessed. The CV predictive log-likelihood is the most principled way of choosing K for AA and well established in other statistical domains but the resulting archetypes are only validated qual itatively. A more principled analysis across many domains and datasets we leave for future work. Another limitation of the predictive likelihood criterion arises in limited-data settings. The predictive score is computed by using the fitted training representation, in particular the training S-matrix, as an empirical approximation to the latent mixing distribution. When the number of training samples is small, this empirical distribution can be sparse and unstable, individual training samples can carry disproportionate weight, and the set of observed archetypal mixtures may not adequately represent the mixtures present in the held-out data. As a result the predictive likelihood can become sensitive to the particular train-test split and may penalize otherwise reasonable archetypes simply because the training S-matrix provides a poor finite-sample approximation to the underlying population distribution. This suggests that predictive likelihood is most reliable when the training set is sufficiently large to capture the relevant variation in mixture proportions, and that additional smoothing or hierarchical modeling of the latent mixing distribution may be beneficial in low-sample regimes.

## V. CONCLUSION

We introduced an efficient and generalizable likelihoodbased framework for AA for discrete data. By combining local quadratic likelihood approximations with SMO updates for the mixture weights and constrained active-set updates for the archetype construction weights, the method provides an efficient optimization scheme as highlighted for the considered Bernoulli, Poisson, and multinomial observation models.

We further derived a predictive likelihood criterion for selecting the number of archetypes. In synthetic experiments, this criterion recovered the true model complexity, whereas the NLL training reconstruction-based measures tended to favor larger models with no clear indication of a suitable number of components. The real-data examples further showed that the optimally defined number of archetypes based on generalization using the proposed predictive likelihood framework could be interpreted as meaningful domain-specific profiles characterizing cancer mutations, PBMC gene-expression programs, and microbiome body-site variation.

Overall, the proposed method offers an efficient alternative approach to likelihood based inference in AA with the proposed predictive likelihood framework providing a simple to use model order assessment procedure that enables model assessment in terms of how well the extracted AA polytope generalizable to new data.

## REFERENCES

[1] A. E. J. Wedenborg and M. Mørup, “Archetypal analysis for binary data,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025, pp. 1–5.

[2] A. Cutler and L. Breiman, “Archetypal analysis,” Technometrics, vol. 36, no. 4, pp. 338–347, 1994. [Online]. Available: https: //www.tandfonline.com/doi/abs/10.1080/00401706.1994.10485840

[3] D. Lee and H. S. Seung, “Algorithms for Non-negative Matrix Factorization,” in Advances in Neural Information Processing Systems, T. Leen, T. Dietterich, and V. Tresp, Eds., vol. 13. MIT Press, 2000.

[4] D. D. Lee and H. S. Seung, “Learning the parts of objects by nonnegative matrix factorization,” nature, vol. 401, no. 6755, pp. 788–791, 1999.

[5] M. Mørup and L. K. Hansen, “Archetypal analysis for machine learning and data mining,” Neurocomputing, vol. 80, pp. 54–63, 3 2012.

[6] J. L. Hinrich, S. E. Bardenfleth, R. E. Roge, N. W. Churchill, K. H. Madsen, and M. Morup, “Archetypal Analysis for Modeling Multisubject fMRI Data,” IEEE Journal on Selected Topics in Signal Processing, vol. 10, no. 7, pp. 1160–1171, 10 2016.

[7] A. S. Olsen, R. M. T. Høegh, J. L. Hinrich, K. H. Madsen, and M. Mørup, “Combining electro- and magnetoencephalography data using directional archetypal analysis,” Frontiers in Neuroscience, vol. Volume 16 - 2022, 2022. [Online]. Available: https://www.frontiersin. org/journals/neuroscience/articles/10.3389/fnins.2022.911034

[8] J. C. Thøgersen, M. Mørup, S. Damkiær, S. Molin, and L. Jelsbak, “Archetypal analysis of diverse Pseudomonas aeruginosa transcriptomes reveals adaptation in cystic fibrosis airways,” BMC Bioinformatics, vol. 14, no. 1, pp. 1–15, 9 2013. [Online]. Available: https://bmcbioinformatics.biomedcentral. com/articles/10.1186/1471-2105-14-279

[9] Y. Chen, J. Mairal, and Z. Harchaoui, “Fast and robust archetypal analysis for representation learning,” in Proceedings of the 2014 IEEE Conference on Computer Vision and Pattern Recognition, ser. CVPR ’14. USA: IEEE Computer Society, 2014, p. 1478–1485. [Online]. Available: https://doi.org/10.1109/CVPR.2014.192

[10] D. Wynen, C. Schmid, and J. Mairal, “Unsupervised Learning of Artistic Styles with Archetypal Style Analysis,” Advances in Neural Information Processing Systems, vol. 2018-December, pp. 6584–6593, 5 2018. [Online]. Available: https://arxiv.org/abs/1805.11155v2

[11] S. M. Keller, M. Samarin, M. Wieser, and V. Roth, “Deep Archetypal Analysis,” Lecture Notes in Computer Science (including subseries Lecture Notes in Artificial Intelligence and Lecture Notes in Bioinformatics), vol. 11824 LNCS, pp. 171–185, 9 2019.

[12] A. Alcacer, I. Epifanio, S. Mair, and M. Mørup, “A survey on archetypal analysis,” arXiv preprint arXiv:2504.12392, 2025.

[13] S. Mair and U. Brefeld, “Coresets for Archetypal Analysis,” in Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d Alche-Buc, E. Fox, and R. Garnett, Eds., vol. 32.´ Curran Associates, Inc., 2019.

[14] R. Han, B. Osting, D. Wang, and Y. Xu, “Probabilistic methods for approximate archetypal analysis,” Information and Inference: A Journal of the IMA, vol. 12, no. 1, pp. 466–493, 2 2022.

[15] S. Seth and M. J. A. Eugster, “Probabilistic archetypal analysis,” Machine Learning, 2016.

[16] N. Gillis and F. Glineur, “Nonnegative Factorization and The Maximum Edge Biclique Problem,” 2 2008.

[17] S. F. V. Nielsen and M. Mørup, “Non-negative Tensor Factorization with missing data for the modeling of gene expressions in the Human Brain,” in 2014 IEEE International Workshop on Machine Learning for Signal Processing (MLSP), 2014, pp. 1–6.

[18] J. Gimbernat-Mayol, D. Montserrat, C. Bustamante, and A. Ioannidis, “Archetypal Analysis for Population Genetics,” 1 2021.

[19] I. Cabero Fayos and I. Epifanio, “Finding archetypal patterns for binary questionnaires,” SORT (Statistics and Operations Research Transactions), vol. 44, pp. 39–66, 2 2020.

[20] R. Bro and S. Jong, “A Fast Non-negativity-constrained Least Squares Algorithm,” Journal of Chemometrics, vol. 11, pp. 393–401, 2 1997.

[21] J. Platt, “Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines,” Advances in Kernel Methods-Support Vector Learning, vol. 208, 1 1998.

[22] S. Prabhakaran, S. Raman, J. E. Vogt, and V. Roth, “Automatic model selection in archetype analysis,” in Joint DAGM (German Association for Pattern Recognition) and OAGM Symposium. Springer, 2012, pp. 458–467.

[23] A. Suleman, “Validation of archetypal analysis,” in 2017 IEEE International Conference on Fuzzy Systems (FUZZ-IEEE). IEEE, 2017, pp. 1–6.

[24] R. M. Bell, Y. Koren, and C. Volinsky, “Modeling relationships at multiple scales to improve accuracy of large recommender systems,” in Knowledge Discovery and Data Mining, 2007.

[25] A. E. J. Wedenborg, “Distribution agnostic archetypal analysis,” https:// github.com/Wedenborg/distribution-agnostic-archetypal-analysis, 2026, python package and code repository.

[26] R. N. Ptashkin, M. D. Ewalt, G. Jayakumaran, and et al., “Enhanced clinical assessment of hematologic malignancies through routine paired tumor and normal sequencing,” Nature Communications, vol. 14, p. 6895, 2023.

[27] 10x Genomics, “3k pbmcs from a healthy donor,” 10x Genomics Datasets, 2016.

[28] F. A. Wolf, P. Angerer, and F. J. Theis, “Scanpy: large-scale single-cell gene expression data analysis,” Genome Biology, vol. 19, no. 1, p. 15, 2018. [Online]. Available: https://doi.org/10.1186/s13059-017-1382-0

[29] Human Microbiome Project Consortium, “Structure, function and diversity of the healthy human microbiome,” Nature, vol. 486, no. 7402, pp. 207–214, 2012.

## VI. BIOGRAPHY SECTION

![](images/634ea0bbd1a87f1c9f2aefa023adcb3f609690b92e9770a4e13f03e93621914a.jpg)  
A. Emilie J. Wedenborg has a B.Sc. in Quantitative Biology and Disease Modeling and a M.Sc. in mathematical modeling and computing both from the Technical University of Denmark (DTU). She is currently a PhD. fellow at DTU Compute, section of Cognitive Systems. Her research interests include interpretable modeling approaches, especially related to life-sciences.

![](images/daa92d4e74ea9938677187ec74f91a764da1b3d2876db3eca8994a45d924a371.jpg)

Jesper Løve Hinrich received his M.Sc. Eng. and PhD degrees in mathematical modelling and computation from the Technical University of Denmark. He is currently a postdoc at DTU Compute, Section for Cognitive Systems. His research focuses on tensor networks and statistical machine learning, in particular their applications to quantifying and uncovering dynamics in biological, chemical, and physical systems.

![](images/fa83161ef7058ebd49eea8b53c9f2536e3dc9e1028eae04032782c4e3d24d262.jpg)

Morten Mørup received his MS and PhD degrees in applied mathematics from the Technical University of Denmark, Denmark, where he is currently professor of machine learning for the life-sciences at the Section for Cognitive Systems at DTU Compute. He has been associate editor of the IEEE Transactions on Signal Processing, area chair for multiple leading machine learning conferences, and his research focuses on machine learning, tensor decomposition, and complex network modeling for the modeling of life-science data.