# Sparsity Regularized and Robust Mean Variance Portfolio Selection Under Ellipsoidal Uncertainty

Deniz Akkaya<sup>∗</sup> Emre Can Yayla<sup>∗</sup> Buse S¸en<sup>†</sup> Mustafa C¸ . Pınar<sup>∗</sup>

## Abstract

We investigate mean–variance portfolio selection with an ℓ<sub>0</sub>-penalty to promote sparsity in asset allocations. Uncertainty in the mean return vector is incorporated through an ellipsoidal uncertainty set, yielding a robust sparse optimization framework. We characterize the structure of both local and global minimizers and exploit these properties in the risk minimization and return maximization formulations. Building on this structural insight, we develop a branchand-bound algorithm tailored to the resulting robust sparse portfolio problems, together with a new pruning rule that can discard exponentially many candidate portfolios in a single step. Extensive computational experiments on real market data, together with comparisons against a mixed-integer second-order cone programming solver, demonstrate the efectiveness and competitiveness of the proposed approach.

Keywords: Mean-variance portfolio, robust optimization, regularization, sparsity, ℓ<sub>0</sub>-norm, Branch-and-Bound

Mathematics Subject Classification (MSC 2020): 90C11, 91G10, 90C17, 90C26, 90C57

## 1 Introduction

Mean-variance portfolio selection, originally introduced by [19], forms the cornerstone of modern portfolio theory. In this framework, an investor determines portfolio weights by balancing expected return against risk measured through variance, thereby tracing the eficient frontier. Despite its conceptual elegance and widespread adoption, the classical mean-variance model is known to be highly sensitive to estimation errors in the input parameters, particularly in the expected return vector. As emphasized in the literature, small perturbations in estimated returns may lead to large and economically unintuitive changes in optimal portfolios, a phenomenon often attributed to the amplification of estimation noise [20].

To mitigate this instability, robust optimization has emerged as a systematic approach for incorporating parameter uncertainty directly into optimization models [5, 6, 14]. In robust portfolio optimization, uncertain parameters such as expected returns are assumed to lie within prescribed uncertainty sets, and portfolio decisions are made to perform well under the worst-case realization within these sets. A prominent modeling choice is the use of ellipsoidal uncertainty sets for the mean return vector, which lead to tractable reformulations and admit appealing statistical interpretations. In particular, [16] demonstrated that robust mean-variance portfolio problems with ellipsoidal uncertainty can be reformulated as convex optimization problems and that the resulting portfolios exhibit improved stability and out-of-sample performance. Subsequent contributions, including [8], explored alternative uncertainty structures and their computational implications. In addition, distributionally robust optimization has provided a broader framework for modeling uncertainty in financial decision making, where ambiguity in probability distributions is explicitly incorporated [13, 15, 25]. Among various robust formulations, we follow the analysis of robust portfolio models developed in [23] as a foundation for our study.

While robustness addresses estimation risk, practical portfolio construction typically involves additional structural considerations. In many real-world applications, investors restrict the number of assets held in a portfolio due to transaction costs, liquidity constraints, monitoring costs, or regulatory requirements. Such considerations naturally lead to sparse portfolio models, where the number of nonzero positions is explicitly controlled. Early work on cardinality-constrained portfo lio optimization highlighted the computational challenges arising from these restrictions [9, 12, 18]. More recently, sparsity-inducing formulations and scalable algorithms have attracted considerable attention in the literature. Contributions such as [7, 10, 17, 26] demonstrate the practical and computational advantages of sparse portfolios. A direct way to enforce sparsity is through cardinality constraints or $\ell _ { 0 } \mathrm { - r e g u l a r i z a t i o n }$ , where the $\ell _ { 0 } { \mathrm { - t e r m } }$ counts the number of selected assets. However, the inclusion of such a term leads to nonconvex and combinatorial optimization problems that are challenging to solve. In [22], a comprehensive analysis of sparse portfolio optimization is presented, including structural properties of optimal portfolios and an eficient enumeration-based branching algorithm. In addition, recent studies continue to explore models that combine sparsity and robustness in portfolio construction [1, 27].

Despite the extensive literature on robust portfolio optimization and the growing body of work on sparse portfolio selection, the integration of ellipsoidal mean uncertainty with exact $\ell _ { 0 ^ { - } }$ regularization remains relatively unexplored. Most robust formulations focus primarily on mitigating estimation risk but do not explicitly control portfolio cardinality. Conversely, sparse portfolio models often rely on deterministic parameter estimates and do not explicitly account for uncertainty in expected returns. A unified treatment that simultaneously addresses estimation risk and enforces exact sparsity leads to a challenging class of robust mixed-integer quadratic optimization problems. Motivating this integration, preliminary out-of-sample tests on real equity market data show that adding ellipsoidal robustness to a sparse mean-variance model can improve risk-adjusted performance over the purely sparse benchmark.

In this paper, we study sparse mean-variance portfolio selection under ellipsoidal uncertainty in the mean return vector. We consider both risk minimization and return maximization variants and incorporate an $\ell _ { 0 } { \mathrm { - p e n a l t y } }$ to promote portfolios with a prescribed level of sparsity. The ellipsoidal uncertainty set is consistent with the robust optimization framework developed in [6, 16, 23], while the $\ell _ { 0 } { \mathrm { - r e g u l a r i z a t i } }$ on follows the exact sparsity-inducing perspective studied in $[ 2 , 3 , 2 1 ]$ and recent sparse portfolio optimization research presented in [22].

Our contributions are both theoretical and computational. On the theoretical side, we characterize the structure of local and global minimizers of the robust sparsity-penalized problem. We analyze how the interaction between the quadratic risk term, the worst-case mean adjustment induced by the ellipsoidal uncertainty set, and the discontinuous $\ell _ { 0 } { \mathrm { - p e n a l t y } }$ determines the structure of optimal portfolios. These results extend structural analyses known for deterministic sparse mean-variance models to a robust setting and clarify the role of the robustness parameter in shaping portfolio composition.

On the computational side, we develop a tailored branch-and-bound framework that exploits problem-specific lower and upper bounds derived from the analytical structure of the model. These bounds enable efective pruning of candidate supports and provide a solid basis for warm-start heuristics. We conduct extensive computational experiments on real financial data and benchmark our approach against mixed-integer second-order conic formulations. The numerical results show that the proposed method is computationally efective, especially on larger instances, and often produces high-quality sparse robust portfolios with substantially reduced running times.

Overall, the paper contributes both structural analysis and algorithmic developments for robust sparse mean-variance portfolio optimization. The main contributions of this paper are as follows.

• We propose a robust sparse mean-variance portfolio model that integrates ellipsoidal uncertainty in expected returns with exact ℓ<sub>0</sub>-regularization, providing a unified treatment of robustness and sparsity.

• We develop a structural analysis of the resulting nonconvex problem, including support-wise decomposition, identification of local minimizers and results on existence and properties of global minimizers.

• We establish explicit bounds on optimal solutions and show that sparsity levels can be controlled via the regularization parameter through thresholding results.

• We design a tailored branch-and-bound algorithm that turns these structural insights into bounding and warm-start strategies, and show on real data that it is computationally efective. A key ingredient is an additional pruning step that refines the enumeration scheme of [22] for the nonrobust sparse portfolio problem: it can eliminate exponentially many subproblems in a single iteration and applies verbatim to that scheme.

We introduce the sparsity-regularized robust portfolio problem in two alternative formulations and present the necessary background and notation in Section 2. Analytical results for both variants are presented in Sections 3 and 4, where we study structural properties of locally and globally optimal portfolios, including bounds on the number of nonzero entries and existence results. A branch-and-bound algorithm is proposed in Section 5. Finally, we compare our algorithm with a state-of-the-art solver for mixed-integer second-order cone programming problems in Section 6 and conclude in Section 7.

## 2 Problem Definition and Notation

Let $\mathbb { I } _ { N } = ( \{ 1 , \dots , N \} , < )$ be the strictly ordered index set, where < denotes the standard order. Any subset $\omega \subseteq \mathbb { I } _ { N }$ inherits this property. We denote the all-ones vector by 1, the identity matrix by I, and the zero vector by 0. We denote the $i ^ { t h }$ column of a matrix D by $d _ { i }$ . For $\omega \subseteq \mathbb { I } _ { N }$ , the following notation for subvectors and submatrices will be used:

$$
\begin{array} { r } { r _ { \omega } : = ( r [ \omega [ 1 ] ] , \ldots , r [ \omega [ | \omega | ] ] ) \in \mathbb { R } ^ { | \omega | } , \quad D _ { \omega } : = \big ( ( d _ { \omega [ 1 ] } ) _ { \omega } , \ldots , ( d _ { \omega [ | \omega | ] } ) _ { \omega } \big ) \in \mathbb { R } ^ { | \omega | \times | \omega | } . } \end{array}
$$

We define the zero-padding operator $Z _ { \omega } : \mathbb { R } ^ { | \omega | }  \mathbb { R } ^ { N }$ by

$$
x = Z _ { \omega } ( x _ { \omega } ) , \quad x [ i ] = { \left\{ \begin{array} { l l } { 0 , } & { i \notin \omega , } \\ { x _ { \omega } [ k ] , } & { { \mathrm { i f } } \omega [ k ] = i . } \end{array} \right. }
$$

We introduce the indicator $\phi : \mathbb { R }  \{ 0 , 1 \}$ defined by

$$
\phi ( t ) = \left\{ 0 , \quad t = 0 , \quad \mathrm { ~ s o ~ t h a t ~ } \| x \| _ { 0 } : = \sum _ { i \in \mathbb { I } _ { N } } \phi ( x [ i ] ) = \sum _ { i \in \sigma ( x ) } \phi ( x [ i ] ) . \right.
$$

Here, $\| x \| _ { 0 } = | \sigma ( x ) |$ , where $\sigma ( x )$ is the support of x (indices of nonzero entries), and $\left. \cdot \right.$ denotes cardinality. For $x \in \mathbb { R } ^ { N }$ , the $\ell _ { p }$ norm is defined for $\begin{array} { r } { 1 \le p < \infty \mathrm { ~ a s ~ } \| x \| _ { p } : = \left( \sum _ { i \in \mathbb { I } _ { N } } | x [ i ] | ^ { p } \right) ^ { 1 / p } } \end{array}$ 2 $\| x \| _ { \infty } = \operatorname* { m a x } _ { i \in \mathbb { I } _ { N } } \{ | x [ i ] | \}$ , and the matrix induced norm by a positive definite matrix D is $\| x \| _ { D } =$ $\sqrt { x ^ { \top } D x }$ . Given $\rho > 0$ , the open $\ell _ { p }$ ball of radius $\rho$ centered at x is $B _ { p } ( x , \rho ) : = \{ y \in \mathbb { R } ^ { N } : \| x - y \| _ { p } <$ $\rho \}$ . For a matrix $A \in \mathbb { R } ^ { M \times N }$ , the spectral norm is $\| A \| _ { 2 } = s _ { 1 } ( A )$ , where $s _ { i } ( A )$ is the $i ^ { t h }$ singular value in decreasing order.

We consider a financial market consisting of N risky assets and a single risk-free asset with deterministic period return $\boldsymbol { r } _ { c } .$ . The vector of expected returns of the risky assets, denoted by $r \in \mathbb { R } ^ { N }$ , is unknown, while their return covariance matrix $D \in \mathbb { R } ^ { N \times N }$ is assumed to be known and positive definite. The initial wealth is normalized to one. Uncertainty in the mean returns is modeled through the ellipsoidal uncertainty set

$$
\begin{array} { r } { U _ { \hat { r } } : = \left\{ r \in \mathbb { R } ^ { N } : \| r - \hat { r } \| _ { D ^ { - 1 } } \leq \gamma \right\} , } \end{array}
$$

where $\hat { r }$ denotes the nominal estimate of the mean return vector and $\gamma > 0$ is a tolerance parameter controlling the size of the uncertainty set. Let ¯r denote a prescribed target return level. The investor selects portfolio weights $\boldsymbol { x } \in \mathbb { R } ^ { N }$ for the risky assets and $x _ { c } \in \mathbb { R }$ for the risk-free asset so as to minimize portfolio variance while ensuring that the target return is achieved for all admissible realizations of the mean return vector. The resulting robust mean-variance portfolio optimization problem is formulated as

$$
\begin{array} { r l } { \operatorname* { m i n } } & { x ^ { \top } D x } \\ { \mathrm { s . t . } } & { \mathbf { 1 } ^ { \top } x + x _ { c } = 1 } \\ & { r ^ { \top } x + r _ { c } x _ { c } \geq \bar { r } \quad \forall r \in U _ { \hat { r } } } \\ & { ( x , x _ { c } ) \in \mathbb { R } ^ { N + 1 } . } \end{array}
$$

Under ellipsoidal uncertainty in the mean return vector, the semi-infinite robust constraint can be reduced to a single deterministic inequality. In particular, requiring that the portfolio achieves the target return for all admissible realizations of $r ,$

$$
r ^ { \top } x + r _ { c } ( 1 - \mathbf { 1 } ^ { \top } x ) \geq \bar { r } \quad \forall r \in U _ { \hat { r } } ,
$$

is equivalent to enforcing that the nominal expected return, penalized by the worst-case deviation induced by the uncertainty set, exceeds the target level. This yields the deterministic robust counterpart

$$
\begin{array} { r } { \hat { r } ^ { \top } x + r _ { c } ( 1 - \mathbf { 1 } ^ { \top } x ) - \gamma \left\| x \right\| _ { D } \geq \bar { r } , } \end{array}
$$

where the term $\gamma \| _ { \boldsymbol { x } } \| _ { D }$ arises from minimizing the linear form $r ^ { \top } x$ over the ellipsoidal set $U _ { \hat { r } }$ Consequently, the original semi-infinite robust mean-variance problem admits the equivalent finitedimensional formulation

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { N } } ~ x ^ { \top } D x \quad \mathrm { s . t . } \quad \hat { r } ^ { \top } x + r _ { c } ( 1 - \mathbf { 1 } ^ { \top } x ) - \gamma \left\| x \right\| _ { D } \geq \bar { r } .\tag{1}
$$

This equivalence follows from standard results in robust optimization, where linear constraints subject to ellipsoidal uncertainty admit exact deterministic robust counterparts involving norm penalties [6, 16]. Let $\beta > 0$ be a sparsity-inducing penalty parameter, and define the estimated excess return vector by $\mathfrak { r } : = \hat { r } - r _ { c } \mathbf { 1 }$ together with the excess target return $\bar { \mathfrak { r } } : = \bar { r } - r _ { c }$ . We consider the following sparse robust mean-variance portfolio selection problem:

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { N } } \mathcal { F } _ { \beta } ( x ) : = x ^ { \top } D x + \beta \left\| x \right\| _ { 0 } \quad \mathrm { s . t . } \quad \mathfrak { r } ^ { \top } x - \gamma \left\| x \right\| _ { D } \geq \bar { \mathfrak { r } } .\tag{P<sup>1</sup>}
$$

An alternative robust formulation to the problem presented in (1) is obtained by reversing the roles of risk and return. Instead of minimizing variance subject to a robust return requirement, one may fix an admissible risk level and maximize the worst-case portfolio return. For a suitably chosen parameter $T > 0$ , consider

$$
\operatorname* { m a x } _ { x \in \mathbb { R } ^ { N } } \operatorname* { m i n } _ { r \in U _ { \widehat { r } } } \Bigl \{ r ^ { \top } x + ( 1 - \mathbf { 1 } ^ { \top } x ) r _ { c } \Bigr \} \quad \mathrm { s . t . } \quad x ^ { \top } D x \leq T ^ { 2 } .
$$

This formulation represents a robust counterpart of the classical mean-variance problem with a variance budget, and it provides an alternative scalarization of the risk-return trade-of. Exploiting the ellipsoidal structure of the uncertainty set $U _ { \hat { r } }$ , the inner minimization over r admits a closedform solution, yielding the deterministic equivalent

$$
\operatorname* { m a x } _ { x \in \mathbb { R } ^ { N } } \hat { r } ^ { \top } x + ( 1 - \mathbf { 1 } ^ { \top } x ) r _ { c } - \gamma \left\| x \right\| _ { D } \quad \mathrm { s . t . } \quad x ^ { \top } D x \leq T ^ { 2 } .
$$

Since $\hat { r } ^ { \top } x + ( 1 - \mathbf { 1 } ^ { \top } x ) r _ { c } = \mathfrak { r } ^ { \top } x + r _ { c } ,$ the objective can be written in terms of excess returns as

$$
\operatorname* { m a x } _ { { \boldsymbol x } \in \mathbb { R } ^ { N } } { \boldsymbol \mathfrak { r } } ^ { \top } { \boldsymbol x } + r _ { c } - \gamma \left\| { \boldsymbol x } \right\| _ { D } \quad \mathrm { s . t . } \quad \left\| { \boldsymbol x } \right\| _ { D } \leq T ,
$$

and the constant $r _ { c }$ can be dropped without afecting the set of optimal solutions.

While this model captures the same robustness considerations as problem (1), the two are not equivalent. The original formulation enforces the target return as a hard robust constraint and minimizes risk accordingly, whereas the present model fixes the risk level and optimizes the worst-case return. Consequently, the two problems generally yield diferent optimal solutions, coinciding only for specific choices of $T$ that recover the risk level induced by the optimal solution of problem (1).

Using the same sparsity-inducing penalty parameter $\beta$ as before, we introduce an alternative formulation aimed at promoting sparser solutions to the return maximization problem. Specifically, we consider the following optimization problem:

$$
\operatorname* { m i n } _ { x \in \mathbb { R } ^ { N } } \mathcal { G } _ { \beta } ( x ) : = \gamma \left\| x \right\| _ { D } - \mathfrak { r } ^ { \top } x + \beta \left\| x \right\| _ { 0 } \quad \mathrm { s . t . } \quad \left\| x \right\| _ { D } \leq T .\tag{P<sup>2</sup>}
$$

In the following sections, we develop the theoretical foundations for both formulations. Our analysis covers local optimality conditions, rigorous bounds on the nonzero components of global minimizers, asymptotic results concerning the existence of global minimizers, and principled guidelines for selecting the parameter $\beta$ to achieve prescribed sparsity levels.

## 3 Analysis of $\left( \mathcal { P } ^ { 1 } \right)$

We begin this section by introducing an assumption that rules out the degenerate solution $x = { \bf 0 }$ in which all wealth is invested in the risk-free asset.

Assumption 1. The target return ¯r strictly exceeds the period return $r _ { c }$ of the risk-free asset.

We impose the following assumption to ensure that each asset’s nominal excess return is suficiently large relative to its risk contribution and the level of mean uncertainty, thereby guaranteeing feasibility of the associated subproblems.

Assumption 2. The uncertainty radius $\gamma$ satisfies

$$
\operatorname* { m i n } _ { i \in \mathbb { I } _ { N } } \frac { | \mathfrak { r } [ i ] | } { \sqrt { d _ { i } [ i ] } } > \gamma ,
$$

where $\sqrt { d _ { i } [ i ] }$ denotes the square root of $i ^ { t h }$ diagonal entry of $D .$

Assumption 2 requires the magnitude of each asset’s nominal excess return relative to its volatility to exceed the uncertainty radius γ. Since short positions are permitted, this ensures that every singleton-support subproblem, and hence every nonempty support-restricted subproblem, is feasible. For ease of notation, define the feasible set of $\left( \mathcal { P } ^ { 1 } \right)$ as

$$
\mathcal { X } = \left\{ \boldsymbol { x } \in \mathbb { R } ^ { N } : \mathfrak { r } ^ { \top } \boldsymbol { x } - \gamma \left\| \boldsymbol { x } \right\| _ { D } \geq \bar { \mathfrak { r } } \right\} .\tag{2}
$$

We also define $H : = \sqrt { \mathfrak { r } ^ { \top } D ^ { - 1 } \mathfrak { r } }$ , which represents the maximal achievable Sharpe ratio in the market and coincides with the slope of the capital market line.

Remark 1. Note that for $\boldsymbol { x } \in \mathbb { R } ^ { N }$ and $\omega \supseteq \sigma ( x )$ , we have $x ^ { \top } D x = x _ { \omega } ^ { \top } D _ { \omega } x _ { \omega }$ . This property motivates searching for local minimizers by restricting to supports.

For $\omega \subseteq \mathbb { I } _ { N }$ , define $K _ { \omega } = \{ x \in \mathbb { R } ^ { N } : x [ i ] = 0 , \forall i \in \omega ^ { c } \}$ . We study the following subproblem to characterize local minimizers of $\left( \mathcal { P } ^ { 1 } \right)$

$$
\operatorname* { m i n } _ { x \in \mathcal { X } \cap K _ { \omega } } x ^ { \top } D x .\tag{P<sup>1</sup><sub>ω</sub>}
$$

Since D is positive definite, the objective is strictly convex, and hence the problem has a unique solution whenever feasible.

## 3.1 Minimizers of $\left( \mathcal { P } _ { \omega } ^ { 1 } \right)$

We define the restricted feasibility set for a fixed $\omega \subseteq \mathbb { I } _ { N }$

$$
\mathcal { X } _ { \omega } = \left\{ u \in \mathbb { R } ^ { | \omega | } : \mathfrak { r } _ { \omega } ^ { \top } u - \gamma \| u \| _ { D _ { \omega } } \geq \bar { \mathfrak { r } } \right\} .
$$

Using these sets and the zero-padding operator, we define the equivalent problem to $\big ( \mathcal { P } _ { \omega } ^ { 1 } \big )$ with a convex feasible region

$$
\operatorname* { m i n } _ { u \in \mathcal { X } _ { \omega } } u ^ { \top } D _ { \omega } u , \quad | \omega | \geq 1 .\tag{ZP<sup>1</sup><sub>ω</sub>}
$$

Remark 2. Assumption 2 is inherited by all subproblems: for any nonempty $\omega \subseteq \mathbb { I } _ { N } , D _ { \omega }$ is positive definite (as a principal submatrix of D) and $| \mathfrak { r } _ { \omega } [ k ] | > \gamma \sqrt { D _ { \omega } [ k , k ] }$ holds for all $k \in \mathbb { I } _ { | \omega | }$ , since the diagonal of $D _ { \omega }$ consists of diagonal entries of D.

The following lemma and remark are crucial in establishing that Assumption 2 ensures the existence of feasible solutions for every subproblem associated with a subset $\omega \subseteq \mathbb { I } _ { N }$

Lemma 1. For any $\boldsymbol { v } \in \mathbb { R } ^ { N }$ , we have $\begin{array} { r } { \boldsymbol { v } ^ { \top } D ^ { - 1 } \boldsymbol { v } = \operatorname* { m a x } _ { u \in \mathbb { R } ^ { N } } \{ 2 u ^ { \top } \boldsymbol { v } - u ^ { \top } D u \} } \end{array}$

Proof. Let $v \in \mathbb { R } ^ { N }$ be fixed, and define $Q ( u ) = 2 u ^ { \top } v - u ^ { \top } D u$ . Then

$$
Q ( u ) = - ( u - D ^ { - 1 } v ) ^ { \top } D ( u - D ^ { - 1 } v ) + v ^ { \top } D ^ { - 1 } v .
$$

Since D is positive definite, the first term is non-positive for every $u \in \mathbb { R } ^ { N }$ , with equality if and only if $u = D ^ { - 1 } v$ . Therefore the maximum is attained at $u = D ^ { - 1 } v$ , and the desired identity follows. □

Remark 3. Let $u \in \mathcal { X } _ { \omega }$ , and write $u = t v$ with $t > 0$ and $\| v \| _ { D _ { \omega } } = 1$ . Then $t ( \gamma - \mathfrak { r } _ { \omega } ^ { \top } v ) + \bar { \mathfrak { r } } \le 0$ Since $\bar { \mathfrak { r } } > 0$ , feasibility requires $\gamma - \mathfrak { r } _ { \omega } ^ { \top } v < 0$ . This holds whenever

$$
\operatorname* { m a x } _ { \| v \| _ { D _ { \omega } } = 1 } \mathfrak { r } _ { \omega } ^ { \top } v = \sqrt { \mathfrak { r } _ { \omega } ^ { \top } ( D _ { \omega } ) ^ { - 1 } \mathfrak { r } _ { \omega } } = : H _ { \omega } > \gamma ,
$$

where $H _ { \omega } ~ = ~ \sqrt { \mathfrak { r } _ { \omega } ^ { \top } ( D _ { \omega } ) ^ { - 1 } \mathfrak { r } _ { \omega } }$ is the Sharpe ratio of the subproblem defined by $\omega .$ . To guarantee feasibility across all supports, it sufices to ensure mi $1 _ { \omega \neq \emptyset } H _ { \omega } > \gamma$ . For any ω $\neq \emptyset$ and $i \in \omega$ we use Lemma 1 by taking $u = t e _ { i } \in \mathbb { R } ^ { | \omega | }$ and obtain:

$$
H _ { \omega } = \sqrt { \mathfrak { t } _ { \omega } ^ { \top } ( D _ { \omega } ) ^ { - 1 } \mathfrak { r } _ { \omega } } \geq \sqrt { 2 t \mathfrak { r } _ { \omega } [ i ] - t ^ { 2 } d _ { i } [ i ] } .
$$

Maximizing the right hand side over t yields $t = \mathfrak { r } _ { \omega } [ i ] / d _ { i } [ i ]$ and we have

$$
\sqrt { \mathfrak { r } _ { \omega } ^ { \top } ( D _ { \omega } ) ^ { - 1 } \mathfrak { r } _ { \omega } } \ge \frac { | \mathfrak { r } _ { \omega } [ i ] | } { \sqrt { d _ { i } [ i ] } } \Rightarrow \sqrt { \mathfrak { r } _ { \omega } ^ { \top } ( D _ { \omega } ) ^ { - 1 } \mathfrak { r } _ { \omega } } \ge \operatorname* { m a x } _ { i \in \omega } \frac { | \mathfrak { r } _ { \omega } [ i ] | } { \sqrt { d _ { i } [ i ] } } .
$$

In fact, the minimum is attained on singletons:

$$
\operatorname* { m i n } _ { \omega \neq \emptyset } H _ { \omega } = H _ { \operatorname* { m i n } } : = \operatorname* { m i n } _ { i \in \mathbb { I } _ { N } } \frac { | \mathfrak { r } [ i ] | } { \sqrt { d _ { i } [ i ] } } > \gamma .
$$

Assumption $\mathcal { L }$ provides this bound. Thus, strict feasibility and convexity ensure a unique solution $f o r \ \left( \mathcal { Z } \mathcal { P } _ { \omega } ^ { 1 } \right)$

Proposition 1. For nonempty $\omega \subseteq \mathbb { I } _ { N }$ , the unique solution of $\big ( \mathcal { Z } \mathcal { P } _ { \omega } ^ { 1 } \big )$ is

$$
\xi ( \omega ) : = \left( \frac { \bar { \mathfrak { r } } } { H _ { \omega } ( H _ { \omega } - \gamma ) } \right) ( D _ { \omega } ) ^ { - 1 } \mathfrak { r } _ { \omega } .
$$

Proof. For the proof, see [23, Proposition 1].

Remark 4. For $\omega \subseteq \mathbb { I } _ { N }$ with $| \omega | \ge 1$ , we write $\Xi ( \omega ) = Z _ { \omega } ( \xi ( \omega ) )$ for its zero-padding to $\mathbb { R } ^ { N }$

The next proposition gives the optimal multiplier of the subproblem constraint and shows that the constraint is active at optimality. The multiplier is used in the branching rule of the algorithm.

Proposition 2. For nonempty $\omega \subseteq \mathbb { I } _ { N }$ , the optimal Lagrange multiplier associated with the constraint of $\big ( \mathcal { Z } \mathcal { P } _ { \omega } ^ { 1 } \big )$ is $\begin{array} { r } { \mu ^ { * } = \frac { 2 \bar { \mathfrak { r } } } { ( H _ { \omega } - \gamma ) ^ { 2 } } } \end{array}$ . Moreover, the constraint is active at the optimal solution uˆ, i.e., $\mathfrak { r } _ { \omega } ^ { \top } \hat { u } - \gamma \| \hat { u } \| _ { D _ { \omega } } = \bar { \mathfrak { r } }$

Proof. The proof parallels the argument in [23, Proposition 1]. We also verify that the constraint $\mathfrak { r } _ { \omega } ^ { \top } u - \gamma \| u \| _ { D _ { \omega } } \geq \bar { \mathfrak { r } } \mathrm { o f } \left( \mathcal { Z } \mathcal { P } _ { \omega } ^ { 1 } \right)$ is active at optimality. Strict feasibility established above ensures that the KKT conditions apply. Suppose that the constraint is inactive at ˆu. Complementary slackness then gives $\mu ^ { * } = 0$ , and stationarity reduces to $2 D _ { \omega } \hat { u } = 0$ . Since $D _ { \omega } \ \succ \ 0$ , this implies $\hat { u } = 0$ which is infeasible because $\bar { \mathfrak { r } } > 0$ by Assumption 1. Thus, the optimal solution ˆu of $\big ( \mathcal { Z } \mathcal { P } _ { \omega } ^ { 1 } \big )$ satisfies ${ \mathfrak { r } } _ { \omega } ^ { \top } { \hat { u } } - \gamma \left\| { \hat { u } } \right\| _ { D _ { \omega } , \ldots } = { \bar { \mathfrak { r } } } .$

Next, we show the dual multiplier in closed-form. By Proposition 1, we have

$$
D _ { \omega } \hat { u } = \frac { \bar { \mathfrak { r } } } { H _ { \omega } ( H _ { \omega } - \gamma ) } \mathfrak { r } _ { \omega } , \quad \mathrm { a n d } \quad \frac { D _ { \omega } \hat { u } } { \| \hat { u } \| _ { D _ { \omega } } } = \frac { \mathfrak { r } _ { \omega } } { H _ { \omega } } .
$$

Substituting these identities into the stationarity condition and rearranging the terms then yields $\begin{array} { r } { \mu ^ { * } = \frac { 2 \bar { \mathfrak { r } } } { ( H _ { \omega } - \gamma ) ^ { 2 } } } \end{array}$ □

The subproblem $\big ( \mathcal { Z P } _ { \omega } ^ { 1 } \big )$ is posed in the reduced space $\mathbb { R } ^ { | \omega | }$ . The following lemma confirms that solving it is equivalent to solving $\big ( \mathcal { P } _ { \omega } ^ { 1 } \big )$ in $\mathbb { R } ^ { N }$

Lemma 2. Problems $\big ( \mathcal { Z } \mathcal { P } _ { \omega } ^ { 1 } \big )$ and $\left( \mathcal { P } _ { \omega } ^ { 1 } \right)$ are equivalent.

Proof. The map $Z _ { \omega } : \mathcal { X } _ { \omega }  \mathcal { X } \cap K _ { \omega }$ is a bijection. Moreover, for any $x _ { \omega } \in \mathcal { X } _ { \omega }$ , we have $x _ { \omega } ^ { \top } D _ { \omega } x _ { \omega } =$ $Z _ { \omega } ( x _ { \omega } ) ^ { \top } D Z _ { \omega } ( x _ { \omega } )$ . Hence the two formulations are equivalent. □

Remark 5. For $\omega \subseteq \mathbb { I } _ { N }$ with $| \omega | \ge 1$ , the point $\Xi ( \omega ) \in \mathbb { R } ^ { N }$ is the unique solution of $\left( \mathcal { P } _ { \omega } ^ { 1 } \right)$

## 3.2 (Local) Minimizers of $\left( \mathcal { P } ^ { 1 } \right)$

Since $\begin{array} { r } { \mathcal { F } _ { \beta } ( x ) = x ^ { \top } D x + \beta \sum _ { i \in \mathbb { I } _ { N } } \phi ( x [ i ] ) } \end{array}$ , activating a zero component incurs an additional penalty $\beta .$ . The following proposition identifies a neighborhood in which this penalty cannot be ofset by the change in the quadratic term of $\left( \mathcal { P } ^ { 1 } \right)$

Proposition 3. Let $\beta > 0$ and ${ \hat { x } } \in { \mathcal { X } }$ . Define $\hat { \boldsymbol { \sigma } } = \boldsymbol { \sigma } ( \hat { \boldsymbol { x } } )$ and

$$
\rho : = \operatorname* { m i n } \left\{ \operatorname* { m i n } _ { i \in \hat { \sigma } } | \hat { x } [ i ] | , \ \frac { \beta } { 2 ( \| D \hat { x } \| _ { 1 } + 1 ) } \right\} .
$$

Then $\rho > 0$ , and:

(i) $I f y \in B _ { \infty } ( 0 , \rho )$ , then P<sub>i∈I</sub> ϕ(ˆx[i] + y[i]) = P<sub>i∈σˆ</sub> ϕ(ˆx[i]) + P<sub>i∈σˆc</sub> ϕ(y[i]).

(ii) $I f y \in B _ { \infty } ( 0 , \rho ) \cap ( \mathbb { R } ^ { N } \setminus K _ { \hat { \sigma } } )$ , then $\mathcal { F } _ { \beta } ( \hat { x } + y ) \ge \mathcal { F } _ { \beta } ( \hat { x } )$ , with strict inequality whenever $\hat { \sigma } ^ { c } \neq \varnothing$

Proof. The proof follows identically from the argument in [22, Lemma 2].

Remark 6. Proposition 3 does not use feasibility of $\hat { x } + y ;$ in particular, it remains valid when y is restricted to perturbations with ${ \hat { x } } + y \in { \mathcal { X } }$

The following two results establish a correspondence between local minimizers of $\left( \mathcal { P } ^ { 1 } \right)$ and global minimizers of $\left( \mathcal { P } _ { \omega } ^ { 1 } \right)$ for $\omega \subseteq \mathbb { I } _ { N }$ , in a manner analogous to Proposition 2 and Lemma 3 in [22]. The proofs are therefore omitted.

Proposition 4. Let $\omega \subseteq \mathbb { I } _ { N } , \omega \neq \emptyset$ . For any $\beta > 0$ , the objective $\mathcal { F } _ { \beta }$ reaches a (local) minimum of $\left( \mathcal { P } ^ { 1 } \right)$ at $\Xi ( \omega )$ , with $| \sigma ( \Xi ( \omega ) ) | \geq 1$ and $\sigma ( \Xi ( \omega ) ) \subseteq \omega$

Lemma 3. Let $\beta > 0$ and let xˆ be a (local) minimizer of $\left( \mathcal { P } ^ { 1 } \right)$ . Then $\hat { x } = \Xi ( \sigma ( \hat { x } ) )$

Proposition 4 and Lemma 3 together show that the set of local minimizers of $\left( \mathcal { P } ^ { 1 } \right)$ is precisely $\{ \Xi ( \omega ) : \emptyset \neq \omega \subseteq \mathbb { I } _ { N } \}$ ; in particular, every local minimizer is completely determined by its support.

## 3.3 Global Minimizers of $\left( \mathcal { P } ^ { 1 } \right)$

We begin by deriving a bounding box that contains all global minimizers of the problem. This box is subsequently employed for Big-M calibration and for tightening the feasible region.

Proposition 5. Let $\beta > 0$ and suppose xˆ is a global minimizer of $\left( \mathcal { P } ^ { 1 } \right)$ . Then we have

$$
\| \hat { x } \| _ { \infty } \leq \sqrt { \frac { \eta } { s _ { N } ( D ) } } < \infty , w h e r e \eta = \operatorname* { m i n } _ { i \in \mathbb { I } _ { N } } \left\{ \frac { \bar { \mathfrak { r } } ^ { 2 } d _ { i } [ i ] } { \left( \lvert \mathfrak { r } [ i ] \rvert - \gamma \sqrt { d _ { i } [ i ] } \right) ^ { 2 } } \right\} .
$$

Proof. Since $\hat { x }$ is a global minimizer of $\left( \mathcal { P } ^ { 1 } \right)$ , for any $i \in \mathbb { I } _ { N }$ , we have

$$
\begin{array} { r } { \widehat { x } ^ { \top } D \widehat { x } + \beta \left\| \widehat { x } \right\| _ { 0 } \leq \Xi ( \{ i \} ) ^ { \top } D \Xi ( \{ i \} ) + \beta \| \Xi ( \{ i \} ) \| _ { 0 } \leq \Xi ( \{ i \} ) ^ { \top } D \Xi ( \{ i \} ) + \beta . } \end{array}
$$

Moreover, Assumption 1 implies $\bar { \mathfrak { r } } > 0$ , hence $0 \not \in \mathcal X$ and therefore $\| \hat { x } \| _ { 0 } \geq 1$ , which results in $\| \hat { x } \| _ { D } ^ { 2 } \leq \| \Xi ( \{ i \} ) \| _ { D } ^ { - }$ for all $i \in \mathbb { I } _ { N }$ . By the definition of $\Xi$ on singleton supports,

$$
\Xi ( \{ i \} ) = \frac { \bar { \mathrm {  ~ \ r s i g n } } ( \mathfrak { r } [ i ] ) } { | \mathfrak { r } [ i ] | - \gamma \sqrt { d _ { i } [ i ] } } e _ { i } ,
$$

and therefore

$$
\| \hat { x } \| _ { D } ^ { 2 } \leq \operatorname* { m i n } _ { i \in \mathbb { I } _ { N } } \| \Xi ( \{ i \} ) \| _ { D } ^ { 2 } = \operatorname* { m i n } _ { i \in \mathbb { I } _ { N } } \left\{ \frac { \bar { \mathfrak { r } } ^ { 2 } d _ { i } [ i ] } { ( | \mathfrak { r } [ i ] | - \gamma \sqrt { d _ { i } [ i ] } ) ^ { 2 } } \right\} = \eta .
$$

For an upper bound on the maximal entry, it is suficient to rescale $\eta$ with the smallest singular value, that is $\begin{array} { r } { \| \hat { \boldsymbol { x } } \| _ { \infty } \le \sqrt { \frac { \eta } { s _ { N } ( D ) } } } \end{array}$ □

The primary objective of Proposition 5 was to identify a bounding box that contains all globally optimal solutions. Nevertheless, deriving an upper bound on the variance is of independent interest. The following result establishes a strict separation of nonzero components from zero under the prescribed parameters. The resulting lower bound on the nonzero entries of global minimizers constitutes the central insight underlying the proposed warm-start heuristic.

Theorem 1. Let $\beta > 0$ and suppose xˆ is a global minimizer of $\left( \mathcal { P } ^ { 1 } \right)$ . Let also $\hat { \sigma } = \sigma ( \hat { x } )$ denote the support $o f { \hat { x } }$ , and we define for all $i \in \mathbb { I } _ { N }$

$$
\rho _ { i } = \operatorname* { m a x } _ { s \in \{ 1 , 1 \} } \left\{ \left\| e _ { i } + s \frac { | \mathbf { r } [ i ] | + \gamma \sqrt { d _ { i } [ i ] } } { | \mathbf { r } [ j ] | - \gamma \sqrt { d _ { j } [ j ] } } e _ { j } \right\| _ { D } \right\} \quad a n d \quad \eta = \operatorname* { m i n } _ { i \in \mathbb { I } _ { N } } \left\{ \frac { \bar { \mathbf { r } } ^ { 2 } d _ { i } [ i ] } { ( | \mathbf { r } [ i ] | - \gamma \sqrt { d _ { i } [ i ] } ) ^ { 2 } } \right\} .
$$

Then $f o r$ every $i \in { \hat { \sigma } }$ , we have

$$
| \hat { x } [ i ] | \geq \operatorname* { m i n } \left\{ \frac { \sqrt { \eta + \beta } - \sqrt { \eta } } { \rho _ { i } } , \frac { \bar { \mathfrak { r } } } { \vert \mathfrak { r } [ i ] \vert - \gamma \sqrt { d _ { i } [ i ] } } \right\} .
$$

Proof. Let $i \in { \hat { \sigma } }$ . We consider two cases based on the cardinality of ${ \hat { \sigma } } .$

Case 1: Assume $| \hat { \sigma } | \geq 2$ , then there exists $j \in \hat { \sigma }$ such that $i \neq j$ . We define $g _ { i j } : \mathbb { R } ^ { N } \to \mathbb { R } ^ { N }$ as $g _ { i j } ( x ) = x - x [ i ] e _ { i } - x [ j ] e _ { j }$ where $e _ { i }$ and $e _ { j }$ are canonical basis vectors of the specified index. We introduce the function $f ( t _ { i } , t _ { j } ) = \mathcal { F } _ { \beta } ( g _ { i j } ( \hat { x } ) + t _ { i } e _ { i } + t _ { j } e _ { j } )$ . Finally, we define a feasibility function

$$
h ( t _ { i } , t _ { j } ) = \mathfrak { r } ^ { \top } g _ { i j } ( \hat { x } ) + t _ { i } \mathfrak { r } [ i ] + t _ { j } \mathfrak { r } [ j ] - \gamma \left. { g } _ { i j } ( \hat { x } ) + t _ { i } e _ { i } + t _ { j } e _ { j } \right. _ { D } - \bar { \mathfrak { r } } .
$$

Since $\hat { x }$ is a global minimizer, Lemma 3 gives $\hat { x } = \Xi ( \hat { \sigma } )$ . By Remark $5 ,$ ˆx is therefore the unique solution of $\left( \mathcal { P } _ { \omega } ^ { 1 } \right)$ with $\omega = { \hat { \sigma } }$ . Proposition 2 then implies that the restricted constraint is active at ${ \hat { x } } .$ This, in turn, implies that the main constraint is also active, yielding $h ( \hat { x } [ i ] , \hat { x } [ j ] ) = 0$ . Precisely, we rewrite this equality as

$$
h ( { \hat { x } } [ i ] , { \hat { x } } [ j ] ) = \mathbf { r } ^ { \top } { \hat { x } } - \gamma \left\| { \hat { x } } \right\| _ { D } - { \bar { \mathbf { r } } } = 0 \Rightarrow { \bar { \mathbf { r } } } = \mathbf { r } ^ { \top } { \hat { x } } - \gamma \left\| { \hat { x } } \right\| _ { D } .
$$

Now, we define a map by fixing a choice for $t _ { j }$

$$
\begin{array} { r l } & { h ( t _ { i } , t _ { j } ) = { \bf r } ^ { \top } g _ { i j } ( \hat { x } ) + t _ { i } { \bf r } [ i ] + t _ { j } { \bf r } [ j ] - \gamma \| g _ { i j } ( \hat { x } ) + t _ { i } e _ { i } + t _ { j } e _ { j } \| _ { D } - \bar { \bf r } } \\ & { \quad \quad \quad = \gamma \left( \| \hat { x } \| _ { D } - \| g _ { i j } ( \hat { x } ) + t _ { i } e _ { i } + t _ { j } e _ { j } \| _ { D } \right) + ( t _ { i } - \hat { x } [ i ] ) { \bf r } [ i ] + ( t _ { j } - \hat { x } [ j ] ) { \bf r } [ j ] } \\ & { \quad \quad \quad \geq - \gamma \left( \| ( t _ { i } - \hat { x } [ i ] ) e _ { i } + ( t _ { j } - \hat { x } [ j ] ) e _ { j } \| _ { D } \right) + ( t _ { i } - \hat { x } [ i ] ) { \bf r } [ i ] + ( t _ { j } - \hat { x } [ j ] ) { \bf r } [ j ] } \\ & { \quad \quad \quad \geq - \gamma | t _ { i } - \hat { x } [ i ] | \sqrt { d _ { i } [ i ] } - \gamma | t _ { j } - \hat { x } [ j ] | \sqrt { d _ { j } [ j ] } + ( t _ { i } - \hat { x } [ i ] ) { \bf r } [ i ] + ( t _ { j } - \hat { x } [ j ] ) { \bf r } [ j ] , } \end{array}
$$

where the first inequality exploits $\| u \| - \| v \| \geq - \| u - v \|$ with $u = { \hat { x } }$ and $v = g _ { i j } ( \hat { x } ) + t _ { i } e _ { i } + t _ { j } e _ { j }$ and the second follows from the triangle inequality together with $\| e _ { i } \| _ { D } = \sqrt { d _ { i } [ i ] }$ . Introduce the shorthand

$$
\delta _ { i j } : = \frac { | \mathfrak { r } [ i ] | + \gamma \sqrt { { d _ { i } } [ i ] } } { | \mathfrak { r } [ j ] | - \gamma \sqrt { { d _ { j } } [ j ] } } > 0 ,
$$

where positivity follows from Assumption 2. Set $t _ { j } = \mathrm { s i g n } ( \mathbf { r } [ j ] ) \left| t _ { i } - { \hat { x } } [ i ] \right| \delta _ { i j } + { \hat { x } } [ j ]$ . Substituting this choice into the lower bound above yields

$$
\begin{array} { r l } & { h ( t _ { i } , t _ { j } ) \geq - \gamma \left| t _ { i } - \hat { x } [ i ] \right| \sqrt { d _ { i } [ i ] } + ( t _ { i } - \hat { x } [ i ] ) \mathbf { r } [ i ] + \delta _ { i j } \left| t _ { i } - \hat { x } [ i ] \right| ( \left| \mathbf { r } [ j ] \right| - \gamma \sqrt { d _ { j } [ j ] } ) } \\ & { \qquad \geq - ( \left| \mathbf { r } [ i ] \right| + \gamma \sqrt { d _ { i } [ i ] } ) \left| t _ { i } - \hat { x } [ i ] \right| + \delta _ { i j } \left| t _ { i } - \hat { x } [ i ] \right| ( \left| \mathbf { r } [ j ] \right| - \gamma \sqrt { d _ { j } [ j ] } ) = 0 . } \end{array}
$$

Thus this choice ensures $h ( t _ { i } , t _ { j } ) \geq 0$ . We may define a 1-dimensional restricted objective as

$$
\begin{array} { r l } & { f ( t ) = \| g _ { i j } ( \hat { x } ) + t e _ { i } + ( \mathrm { s i g n } ( \mathbf { r } [ j ] ) ) \vert t - \hat { x } [ i ] \vert \delta _ { i j } + \hat { x } [ j ] ) e _ { j } \| _ { D } ^ { 2 } } \\ & { \qquad + \beta \| g _ { i j } ( \hat { x } ) + t e _ { i } + ( \mathrm { s i g n } ( \mathbf { r } [ j ] ) \vert t - \hat { x } [ i ] \vert \delta _ { i j } + \hat { x } [ j ] ) e _ { j } \| _ { 0 } } \\ & { \qquad = \| \hat { x } + ( t - \hat { x } [ i ] ) e _ { i } + \mathrm { s i g n } ( \mathbf { r } [ j ] ) \vert t - \hat { x } [ i ] \vert \delta _ { i j } e _ { j } \| _ { D } ^ { 2 } } \\ & { \qquad + \beta \| \hat { x } + ( t - \hat { x } [ i ] ) e _ { i } + \mathrm { s i g n } ( \mathbf { r } [ j ] ) \vert t - \hat { x } [ i ] \vert \delta _ { i j } e _ { j } \| _ { 0 } . } \end{array}
$$

By the choice of $t _ { j }$ above, the point $\hat { x } - \hat { x } [ i ] e _ { i } + \mathrm { s i g n } ( \pmb { \mathrm { r } } [ j ] ) \left| \hat { x } [ i ] \right| \delta _ { i j } e _ { j }$ corresponding to $t = 0$ is feasible $( h \ge 0 )$ , while $f ( \hat { x } [ i ] ) = \mathcal { F } _ { \beta } ( \hat { x } )$ . Global optimality of ˆx therefore implies $f ( 0 ) \geq f ( \hat { x } [ i ] )$ , hence we have

$$
\begin{array} { r l } & { \| \hat { { \boldsymbol x } } \| _ { D } ^ { 2 } + \beta \| \hat { { \boldsymbol x } } \| _ { 0 } \leq \| \hat { { \boldsymbol x } } - \hat { { \boldsymbol x } } [ i ] e _ { i } + \mathrm { s i g n } ( \mathfrak { r } [ j ] ) \| \hat { { \boldsymbol x } } [ i ] \| \delta _ { i j } e _ { j } \| _ { D } ^ { 2 } } \\ & { \qquad + \beta \| \hat { { \boldsymbol x } } - \hat { { \boldsymbol x } } [ i ] e _ { i } + \mathrm { s i g n } ( \mathfrak { r } [ j ] ) \| \hat { { \boldsymbol x } } [ i ] \| \delta _ { i j } e _ { j } \| _ { 0 } . } \end{array}
$$

The vector $\hat { x } - \hat { x } [ i ] e _ { i } + \mathrm { s i g n } ( \mathfrak { r } [ j ] ) \left| \hat { x } [ i ] \right| \delta _ { i j } e _ { j }$ has at most $| \hat { \sigma } | - 1$ nonzero elements since the perturbation removes index i without adding a new nonzero index. Hence its $\ell _ { 0 } { \mathrm { - t e r m } }$ is at most $\Vert \hat { x } \Vert _ { 0 } - 1$ , yielding

$$
\begin{array} { r l } & { \beta \leq \| \hat { x } - \hat { x } [ i ] e _ { i } + \mathrm { s i g n } ( \mathbf { r } [ j ] ) | \hat { x } [ i ] | \delta _ { i j } e _ { j } \| _ { D } ^ { 2 } - \| \hat { x } \| _ { D } ^ { 2 } } \\ & { \quad = ( 2 \hat { x } - \hat { x } [ i ] e _ { i } + \mathrm { s i g n } ( \mathbf { r } [ j ] ) | \hat { x } [ i ] | \delta _ { i j } e _ { j } ) ^ { \top } D ( - \hat { x } [ i ] e _ { i } + \mathrm { s i g n } ( \mathbf { r } [ j ] ) | \hat { x } [ i ] | \delta _ { i j } e _ { j } ) } \\ & { \quad \leq \| 2 \hat { x } - \hat { x } [ i ] e _ { i } + \mathrm { s i g n } ( \mathbf { r } [ j ] ) | \hat { x } [ i ] | \delta _ { i j } e _ { j } \| _ { D } \| - \hat { x } [ i ] e _ { i } + \mathrm { s i g n } ( \mathbf { r } [ j ] ) | \hat { x } [ i ] | \delta _ { i j } e _ { j } \| _ { D } } \\ & { \quad \leq \left( 2 \left\| \hat { x } \right\| _ { D } + | \hat { x } [ i ] | \left\| e _ { i } - \mathrm { s i g n } ( \hat { x } [ i ] \mathbf { r } [ j ] ) \delta _ { i j } e _ { j } \right\| _ { D } \right) | \hat { x } [ i ] | \left\| e _ { i } - \mathrm { s i g n } ( \hat { x } [ i ] \mathbf { r } [ j ] ) \delta _ { i j } e _ { j } \right\| _ { D } . } \end{array}\tag{3}
$$

Here, the last two inequalities follow from the Cauchy-Schwarz and the triangle inequalities. By the definition of $\rho _ { i }$ , we have $\| e _ { i } - \mathrm { s i g n } ( \hat { x } [ i ] \mathfrak { r } [ j ] ) \delta _ { i j } e _ { j } \| _ { D } \le \rho _ { i }$ . Substituting this together with $\| \hat { x } \| _ { D } \leq \sqrt { \eta } ,$ established in the proof of Proposition 5, into (3), we obtain

$$
\beta \leq ( 2 \sqrt { \eta } + \vert \hat { x } [ i ] \vert \rho _ { i } ) \vert \hat { x } [ i ] \vert \rho _ { i } .\tag{4}
$$

Consider the polynomial $P ( v ) = \rho _ { i } ^ { 2 } v ^ { 2 } + 2 \sqrt { \eta } \rho _ { i } v - \beta$ . Since $P$ is a quadratic with a positive leading coeficient, (4) implies that $| \hat { x } [ i ] |$ must be larger than the positive root of P. Evaluating these roots gives

$$
v _ { + } , v _ { - } = \frac { - 2 \sqrt { \eta } \rho _ { i } \pm \sqrt { 4 \eta \rho _ { i } ^ { 2 } + 4 \rho _ { i } ^ { 2 } \beta } } { 2 \rho _ { i } ^ { 2 } } = \frac { - \sqrt { \eta } \pm \sqrt { \eta + \beta } } { \rho _ { i } } .
$$

Hence, we obtain the lower bound $\begin{array} { r } { | \hat { x } [ i ] | \ge \frac { \sqrt { \eta + \beta } - \sqrt { \eta } } { \rho _ { i } } . } \end{array}$

Case 2: Assume $| \hat { \sigma } | = 1$ , then we have a closed form solution for the specific entry $i \in { \hat { \sigma } }$ which takes the form

$$
\hat { x } [ i ] = \frac { \bar { \mathfrak { r } } \mathrm { s i g n } ( { \mathfrak { r } } [ i ] ) } { | { \mathfrak { r } } [ i ] | - \gamma \sqrt { d _ { i } [ i ] } } \quad \Rightarrow \quad | \hat { x } [ i ] | = \frac { \bar { \mathfrak { r } } } { | { \mathfrak { r } } [ i ] | - \gamma \sqrt { d _ { i } [ i ] } } .
$$

In both cases the lower bound takes the form

$$
\operatorname* { m i n } \left\{ \frac { \sqrt { \eta + \beta } - \sqrt { \eta } } { \rho _ { i } } , \frac { \bar { \mathfrak { r } } } { \vert \mathfrak { r } [ i ] \vert - \gamma \sqrt { d _ { i } [ i ] } } \right\} ,
$$

where the first term is active when $| \hat { \sigma } | \geq 2$ and the second when $| \hat { \sigma } | = 1$ . Since a global minimizer must fall into one of these two cases, the bound holds unconditionally for every $i \in \hat { \sigma }$ □

The first term of the bound grows like $\sqrt { \beta }$ for large $\beta ,$ , so the larger the sparsity penalty, the further the nonzero entries of a global minimizer with at least two assets must lie from zero. Having established bounds on the components of global minimizers, we next address their existence, for which we first verify that the objective is coercive.

Lemma 4. The feasible set X defined in (2) is nonempty and closed, and its objective $\mathcal { F } _ { \beta } ( x )$ is coercive on $\mathcal { X }$

Proof. Since $D \succ 0$ , we have $x ^ { \top } D x \geq s _ { N } ( D ) \| x \| _ { 2 } ^ { 2 }$ . Hence, $\mathcal { F } _ { \beta } ( \boldsymbol { x } ) = \boldsymbol { x } ^ { \top } D \boldsymbol { x } + \beta \| \boldsymbol { x } \| _ { 0 } \geq s _ { N } ( D ) \| \boldsymbol { x } \| _ { 2 } ^ { 2 }$ Therefore, $\mathcal { F } _ { \beta }$ is coercive. Moreover, the feasible set $\mathcal { X } = \{ x \in \mathbb { R } ^ { N } : { \mathfrak { r } } ^ { \top } x - \gamma \| x \| _ { D } \geq \bar { \mathfrak { r } } \}$ is closed, since it is the preimage of a closed interval under a continuous function. Assumption 2 guarantees that X is nonempty.

We are now ready to establish the existence of a global minimizer.

Theorem 2. Let $\beta > 0$ . Then the set of global minimizers of $\mathcal { F } _ { \beta }$ ,

$$
\hat { X } = \left\{ \hat { x } \in \mathcal { X } : \mathcal { F } _ { \beta } ( \hat { x } ) = \underset { x \in \mathcal { X } } { \operatorname* { m i n } } \mathcal { F } _ { \beta } ( x ) \right\}
$$

is nonempty.

Proof. By Lemma 4, the feasible set X is closed and nonempty, while $\mathcal { F } _ { \beta }$ is coercive. In addition, $\mathcal { F } _ { \beta }$ is lower semicontinuous, since $\lVert \cdot \rVert _ { 0 }$ is lower semicontinuous [21, Proof of Proposition 4.3]. Since $\mathcal { F } _ { \beta }$ is lower semicontinuous and coercive on the closed nonempty set $x ,$ , it follows from [24, Theorem 1.9] that $\mathcal { F } _ { \beta }$ attains its minimum over X . Therefore, $\hat { X }$ is nonempty. □

The following statement establishes the existence of a threshold value for the penalty parameter corresponding to each prescribed sparsity level. In particular, for every global minimizer of $\mathcal { F } _ { \beta }$ associated with a given sparsity, one can identify a penalty parameter that enforces that level.

Proposition 6. For any $1 \leq k \leq N - 1$ , there exists $\beta _ { k } > 0$ such that if $\beta > \beta _ { k }$ , then every global minimizer xˆ of $\mathcal { F } _ { \beta }$ satisfies $\| \hat { x } \| _ { 0 } \leq k$

Proof. Fix $k \in \mathbb { I } _ { N - 1 }$ and consider the set $X _ { k + 1 } = \{ x \in \mathcal { X } : \| x \| _ { 0 } \geq k + 1 \}$ . Suppose first that $X _ { k + 1 }$ is nonempty. Then, for each $\overline { { x } } \in X _ { k + 1 }$ , we have $\begin{array} { r } { \mathcal { F } _ { \beta } ( \overline { { \boldsymbol { x } } } ) = \overline { { \boldsymbol { x } } } ^ { \top } D \overline { { \boldsymbol { x } } } + \beta \left. \overline { { \boldsymbol { x } } } \right. _ { 0 } \geq \beta ( k + 1 ) } \end{array}$ . Choose any support $\omega \subset \mathbb { I } _ { N }$ such that $| \omega | \le k$ and $\Xi ( \omega ) \in \mathcal { X }$ . Such a support always exists: by Assumption 2 and Remark 3, every singleton {i} with $i \in \mathbb { I } _ { N }$ satisfies $\begin{array} { r } { H _ { \{ i \} } = \frac { | \mathfrak { r } [ i ] | } { \sqrt { d _ { i } [ i ] } } > \gamma } \end{array}$ , which guarantees $\Xi ( \{ i \} ) \in \mathcal { X }$ . Since $| \{ i \} | = 1 \leq k$ for any $k \leq N - 1$ , taking $\omega = \{ i \}$ for any $i \in \mathbb { I } _ { N }$ yields a valid choice. Choose $\beta _ { k }$ so that $\beta _ { k } \ge \Xi ( \omega ) ^ { \top } D \Xi ( \omega )$ . For such a choice,

$$
\mathcal { F } _ { \beta } ( \Xi ( \omega ) ) = \Xi ( \omega ) ^ { \top } D \Xi ( \omega ) + \beta \| \Xi ( \omega ) \| _ { 0 } \leq \beta _ { k } + \beta k < \beta ( k + 1 ) \leq \mathcal { F } _ { \beta } ( \overline { { x } } ) , \qquad \forall \overline { { x } } \in X _ { k + 1 } .
$$

whenever $\beta > \beta _ { k }$ . Let ${ \hat { x } } \in { \hat { X } }$ denote a global minimizer of $\mathcal { F } _ { \beta }$ with ${ \hat { x } } \in { \mathcal { X } }$ . Since

$$
\mathcal { F } _ { \beta } ( \hat { x } ) \leq \mathcal { F } _ { \beta } ( \Xi ( \omega ) ) < \mathcal { F } _ { \beta } ( \overline { { x } } ) , \qquad \forall \overline { { x } } \in X _ { k + 1 } ,
$$

we must have ${ \hat { x } } \notin X _ { k + 1 }$ . By the definition of $X _ { k + 1 }$ , this implies $\| \hat { x } \| _ { 0 } \leq k$ . If $X _ { k + 1 } = \varnothing$ , then the existence of a global minimizer immediately yields $\| \hat { x } \| _ { 0 } \leq k$ □

Remark 7. For any $\beta > 0$ , every admissible support induces a local minimizer $o f { \mathcal { F } } _ { \beta }$ . Nevertheless, this correspondence is not one-to-one, as a single local minimizer may be generated by multiple supports via zero-padding operators. Consequently, the number of distinct local minimizers is bounded above $b y 2 ^ { N } - 1$ , which corresponds to the total number of nontrivial supports.

In the next section, we extend the theoretical results developed here to the problem of robust return maximization under a variance budget. In particular, we adapt the main structural and sparsity-related properties to this risk-constrained setting and show that similar conclusions can be obtained in that framework.

## 4 Analysis of $\left( \mathcal { P } ^ { 2 } \right)$

In this section, we characterize the support-restricted, local, and global minimizers of $\left( \mathcal { P } ^ { 2 } \right)$ and derive bounds on the nonzero components of its globally optimal solutions. For ease of notation, define its feasible set as

$$
\mathcal { V } : = \left\{ x \in \mathbb { R } ^ { N } : \left\| x \right\| _ { D } \leq T \right\} ,\tag{5}
$$

We next impose a lower bound on T that rules out the zero vector as a global minimizer.

Assumption 3. Under Assumption 2, we further assume

$$
T > \operatorname* { m i n } _ { i \in \mathbb { I } _ { N } } \left\{ \frac { \sqrt { d _ { i } [ i ] } \beta } { | \mathfrak { r } [ i ] | - \sqrt { d _ { i } [ i ] } \gamma } \right\} .
$$

Remark 8. The purpose of this lower bound is to rule out the trivial solution $x = \mathbf { 0 }$ , in which all wealth is invested in the risk-free asset. To see this, fix $i \in \mathbb { I } _ { N }$ and consider the singleton portfolio

$$
x = { \frac { T \operatorname { s i g n } ( \mathbf { r } [ i ] ) } { \sqrt { d _ { i } [ i ] } } } e _ { i } .
$$

This portfolio satisfies $\| x \| _ { D } = T$ , and its objective value is

$$
{ \mathcal G } _ { \beta } ( x ) = - \frac { T } { \sqrt { d _ { i } [ i ] } } \left( | { \mathfrak r } [ i ] | - \gamma \sqrt { d _ { i } [ i ] } \right) + \beta < 0 ,
$$

where the strict inequality follows from Assumption 3. Thus, $\mathcal { G } _ { \beta }$ takes a negative value at a feasible point in Y, whereas $\mathcal { G } _ { \beta } ( \mathbf { 0 } ) = 0$ . Hence, the zero vector cannot be globally optimal.

Remark 9. The reasoning outlined in Remark 1 remains applicable to this objective function, as it preserves separability with respect to supports. This structural property plays a central role in developing a rigorous characterization of local minimizers.

We consider the following restricted problem in order to characterize the local minimizers of $\left( \mathcal { P } ^ { 2 } \right)$ :

$$
\operatorname* { m i n } _ { \boldsymbol x \in \mathcal { V } \cap K _ { \omega } } \gamma \left\| \boldsymbol x \right\| _ { D } - \boldsymbol \mathfrak { r } ^ { \top } \boldsymbol x .\tag{P<sup>2</sup><sub>ω</sub>}
$$

Under Assumption 2, we have $H _ { \omega } > \gamma$ for every nonempty $\omega .$ In this case, the subproblem admits a unique optimal solution, which we characterize in the next section.

## 4.1 Minimizers of $\left( \mathcal { P } _ { \omega } ^ { 2 } \right)$

For a fixed nonempty index set $\omega \subseteq \mathbb { I } _ { N }$ , we introduce the corresponding restricted feasible region $\mathcal { V } _ { \omega } = \{ u \in \mathbb { R } ^ { | \omega | } : \| u \| _ { D _ { \omega } , \mathit { \Theta } } \leq T \}$ . Based on this construction and the zero padding operator, we formulate a problem equivalent to $\big ( \mathcal { P } _ { \omega } ^ { 2 } \big )$ , now expressed over a convex feasible set:

$$
\operatorname* { m i n } _ { u \in \mathcal { V } _ { \omega } } \gamma \left\| u \right\| _ { D _ { \omega } } - \mathfrak { r } _ { \omega } ^ { \top } u , \quad \omega \neq \emptyset .\tag{ZP<sup>2</sup><sub>ω</sub>}
$$

In this case, the subproblems $\big ( \mathcal { P } _ { \omega } ^ { 2 } \big )$ and $\big ( \mathcal { Z P } _ { \omega } ^ { 2 } \big )$ admit a unique optimal solution, characterized below.

Proposition 7. For nonempty $\omega \subseteq \mathbb { I } _ { N }$ , the unique optimal solution of $\big ( \mathcal { Z P } _ { \omega } ^ { 2 } \big )$ is

$$
\pi ( \omega ) : = \frac { T } { H _ { \omega } } ( D _ { \omega } ) ^ { - 1 } \mathfrak { r } _ { \omega } .
$$

Proof. A proof of this result is given in [23, Proposition $2 ]$

Remark 10. For $\omega \subseteq { \mathbb { I } } _ { N }$ , we write $\Pi ( \omega ) = Z _ { \omega } ( \pi ( \omega ) )$ for its zero-padding to $\mathbb { R } ^ { N }$ , similarly as in Remark 4.

If $H _ { \omega } < \gamma$ , then $u = 0$ is the unique optimal solution of $\big ( \mathcal { Z P } _ { \omega } ^ { 2 } \big )$ , that is, all wealth is held in the risk-free asset. Assumption 2 excludes this degenerate case, since it guarantees mi $1 _ { \omega \neq \emptyset } H _ { \omega } > \gamma$ for every nonempty ω. As in the risk-minimization case (§3), the next proposition gives the optimal multiplier of the subproblem constraint and shows that the constraint is active at optimality. The multiplier is used in the branching rule of the algorithm.

Proposition 8. For nonempty $\omega \subseteq \mathbb { I } _ { N }$ the optimal Lagrange multiplier associated with the constraint of $\big ( \mathcal { Z P } _ { \omega } ^ { 2 } \big )$ is $\mu ^ { * } = H _ { \omega } - \gamma$ . Moreover, the constraint is active at the optimal solution uˆ, i.e., $\| \hat { u } \| _ { D _ { \omega } } = T$

Proof. The argument follows closely that of [23, Proposition 2]. In addition, we establish that the restricted constraint $\| u \| _ { D _ { \omega } } \leq T$ of $\big ( \mathcal { Z P } _ { \omega } ^ { 2 } \big )$ is active at optimality. Suppose that the constraint is inactive at ˆu. Complementary slackness then gives $\mu ^ { * } = 0$ , so that ˆu minimizes the unconstrained convex function $u \mapsto \gamma \Vert u \Vert _ { D _ { \omega } } - \mathfrak { r } _ { \omega } ^ { \top } u .$ . If ˆu $\neq 0$ , stationarity gives $\gamma D _ { \omega } \hat { u } / | | \hat { u } | | _ { D _ { \omega } } = \mathfrak { r } _ { \omega }$ and taking the $D _ { \omega } ^ { - 1 }$ -norm of both sides yields $H _ { \omega } = \| \mathfrak { r } _ { \omega } \| _ { ( D _ { \omega } ) ^ { - 1 } } = \gamma$ . If $\hat { u } = 0 ,$ , the optimality condition $\mathfrak { r } _ { \omega } \in \gamma \partial \Vert \cdot \Vert _ { D _ { \omega } } ( 0 )$ gives $H _ { \omega } \le \gamma$ . Both conclusions contradict $H _ { \omega } > \gamma$ , which is guaranteed by Assumption 2. Consequently, the optimal solution ˆu of $\big ( \mathcal { Z P } _ { \omega } ^ { 2 } \big )$ satisfies $\| \hat { u } \| _ { D _ { \omega } } = T$

Stationarity at the nonzero optimal solution now gives $\begin{array} { r } { \mathfrak { r } _ { \omega } = ( \gamma + \mu ^ { * } ) \frac { D _ { \omega } \hat { u } } { \| \hat { u } \| _ { D _ { \omega } } } } \end{array}$ . Taking the $D _ { \omega } ^ { - 1 }$ -norm and using $\| \hat { u } \| _ { D _ { \omega } } = T$ yields $\boldsymbol { H _ { \omega } } = \boldsymbol { \gamma } + \boldsymbol { \mu } ^ { * }$ . Hence, $\mu ^ { * } = H _ { \omega } - \gamma$ □

Lemma 5. Problems $\big ( \mathcal { Z P } _ { \omega } ^ { 2 } \big )$ and $\left( \mathcal { P } _ { \omega } ^ { 2 } \right)$ are equivalent.

Proof. The argument proceeds along the same lines as in Lemma 2.

## 4.2 (Local) Minimizers of $\left( \mathcal { P } ^ { 2 } \right)$

To analyze the local minimizers of $\left( \mathcal { P } ^ { 2 } \right)$ , we express its objective using the indicator function ϕ; $i . e .$ $\begin{array} { r } { \mathcal G _ { \beta } ( \boldsymbol x ) = \gamma \left\| \boldsymbol x \right\| _ { D } - \boldsymbol \tau ^ { \top } \boldsymbol x + \beta \sum _ { i \in \sigma ( \boldsymbol x ) } \phi ( \boldsymbol x [ i ] ) } \end{array}$ . The following proposition, analogous to Proposition $^ { 3 , }$ identifies a neighborhood in which activating components outside the support of a feasible point cannot decrease the objective of $\left( \mathcal { P } ^ { 2 } \right)$

Proposition 9. Let $\beta > 0$ and ${ \hat { x } } \in { \mathcal { y } } \setminus \{ 0 \}$ . Define $\hat { \boldsymbol { \sigma } } = \boldsymbol { \sigma } ( \hat { \boldsymbol { x } } )$ and

$$
\rho : = \operatorname* { m i n } \left\{ \operatorname* { m i n } _ { i \in \hat { \sigma } } | \hat { x } [ i ] | , \ \frac { \beta } { \| \pmb { \mathfrak { r } } \| _ { 1 } + \gamma \sqrt { N \left\| D \right\| _ { 2 } } + 1 } \right\} .
$$

Then $\rho > 0$ , and:

$$
\begin{array} { r } { ( i ) \ H f y \in B _ { \infty } ( 0 , \rho ) , \ t h e n \sum _ { i \in \mathbb { T } _ { N } } \phi ( \hat { x } [ i ] + y [ i ] ) = \sum _ { i \in \hat { \sigma } } \phi ( \hat { x } [ i ] ) + \sum _ { i \in \hat { \sigma } ^ { c } } \phi ( y [ i ] ) . } \end{array}
$$

(ii) If $y \in B _ { \infty } ( 0 , \rho ) \cap ( \mathbb { R } ^ { N } \setminus K _ { \hat { \sigma } } )$ , then $\mathcal G _ { \beta } ( \hat { x } + y ) \ge \mathcal G _ { \beta } ( \hat { x } )$ , with strict inequality whenever $\hat { \sigma } ^ { c } \neq \varnothing$ Proof. We first prove (i). For $y \in B _ { \infty } ( 0 , \rho )$ we have $\begin{array} { r } { \left. y \right. _ { \infty } < \operatorname* { m i n } _ { i \in \hat { \sigma } } \left| \hat { x } [ i ] \right| } \end{array}$ . This implies $\phi ( \hat { x } [ i ] +$ $y [ i ] ) = \phi ( \hat { x } [ i ] )$ for $i \in { \hat { \sigma } }$ . For $i \in \hat { \sigma } ^ { c }$ we have $\phi ( \hat { x } [ i ] + y [ i ] ) = \phi ( y [ i ] )$ , which gives the desired result. We now prove (ii). Let $y \in B _ { \infty } ( 0 , \rho ) \setminus K _ { \hat { \sigma } }$ . Then

$$
\begin{array} { r l } & { \displaystyle \mathcal { G } _ { \beta } ( \hat { x } + y ) = \gamma \| \hat { x } + y \| _ { D } - \mathfrak { r } ^ { \top } ( \hat { x } + y ) + \beta \| \hat { x } + y \| _ { 0 } } \\ & { \quad \quad \quad = \mathcal { G } _ { \beta } ( \hat { x } ) - \mathfrak { r } ^ { \top } y + \gamma ( \| \hat { x } + y \| _ { D } - \| \hat { x } \| _ { D } ) + \beta \displaystyle \sum _ { i \in \hat { \sigma } ^ { c } } \phi ( y [ i ] ) } \\ & { \quad \quad \quad \geq \mathcal { G } _ { \beta } ( \hat { x } ) - \mathfrak { r } ^ { \top } y - \gamma \| y \| _ { D } + \beta \| y _ { \hat { \sigma } ^ { c } } \| _ { 0 } } \\ & { \quad \quad \geq \mathcal { G } _ { \beta } ( \hat { x } ) - \| y \| _ { \infty } \left( \| \mathfrak { r } \| _ { 1 } + \gamma \sqrt { N \| D \| _ { 2 } } \right) + \beta \| y _ { \hat { \sigma } ^ { c } } \| _ { 0 } . } \end{array}
$$

For $\hat { \sigma } ^ { c } = \varnothing$ , inequality is trivial. If not, $\| y _ { \hat { \sigma } ^ { c } } \| _ { 0 } \geq 1$ , and the given radius provides the inequality.

The next result establishes a correspondence, analogous to Proposition 4 and Lemma 3, between local minimizers of $\left( \mathcal { P } ^ { 2 } \right)$ and global minimizers of $\big ( \mathcal { P } _ { \omega } ^ { 2 } \big )$ for a given $\omega \subseteq \mathbb { I } _ { N }$ , and the proofs are omitted accordingly.

Proposition 10. Let $\omega \subseteq \mathbb { I } _ { N } , \omega \neq \emptyset$ . For any $\beta > 0$ , the objective $\mathcal { G } _ { \beta }$ reaches a (local) minimum of $\left( \mathcal { P } ^ { 2 } \right)$ at $\Pi ( \omega )$ , with $| \sigma ( \Pi ( \omega ) ) | \geq 1$ and $\sigma ( \Pi ( \omega ) ) \subseteq \omega$ . Moreover, any nonzero (local) minimizer xˆ of (P<sup>2</sup>) satisfies $\hat { x } = \Pi ( \sigma ( \hat { x } ) )$ .

## 4.3 Global Minimizers of $\left( \mathcal { P } ^ { 2 } \right)$

We begin with an observation concerning the existence of an upper bound on the components of globally optimal portfolios.

Remark 11. Let $\beta > 0$ , and let xˆ be a global minimizer of $\left( \mathcal { P } ^ { 2 } \right)$ . In Proposition 5 we developed a non-trivial upper bound on the nonzero entries of a global minimizer of $\left( \mathcal { P } ^ { 1 } \right)$ . For $\left( \mathcal { P } ^ { 2 } \right)$ , the feasible region Y defined in (5) is compact. Thus, the corresponding bound is immediate. Indeed, every global minimizer xˆ satisfies $\| \hat { x } \| _ { D } = T$ and $\begin{array} { r } { \| \hat { x } \| _ { \infty } \le \frac { \bar { T } } { \sqrt { s _ { N } ( D ) } } < \infty } \end{array}$

We proceed by establishing lower bounds on the nonzero components of a globally optimal portfolio.

Theorem 3. Let $\beta > 0$ and suppose $\hat { x }$ is a global minimizer of $\left( \mathcal { P } ^ { 2 } \right)$ . If $\hat { \sigma } = \sigma ( \hat { x } )$ denotes the support $o f { \hat { x } }$ , then for every $i \in { \hat { \sigma } }$ , we have

$$
| \hat { x } [ i ] | \geq \operatorname* { m i n } \left\{ \frac { \beta } { | \mathfrak { r } [ i ] | + H \sqrt { d _ { i } [ i ] } } , \frac { T } { \sqrt { d _ { i } [ i ] } } \right\} .
$$

Proof. Let $| \hat { \sigma } | \geq 2$ and $i \in { \hat { \sigma } }$ . We define $g _ { i } : \mathbb { R } ^ { N } \to \mathbb { R } ^ { N }$ as $g _ { i } ( x ) = x - x [ i ] e _ { i }$ where $e _ { i }$ is the canonical basis vector of the specified index. We introduce the function $f ( t _ { i } ) = \mathcal G _ { \beta } ( g _ { i } ( \hat { x } ) + t _ { i } e _ { i } )$ . Finally, we define a feasibility function $h ( t _ { i } ) = T - \| g _ { i } ( \hat { x } ) + t _ { i } e _ { i } \| _ { D }$ . Since $\hat { x }$ is a global minimizer, it should be a minimizer of $( \mathcal { P } _ { \hat { \sigma } } ^ { 2 } )$ . As shown in Proposition 8, the restricted constraint is active at ${ \hat { x } } .$ This implies that the main constraint is also active, yielding $h ( \hat { x } [ i ] ) = 0$ . Then, we have the following three cases:

Case I: $h ( 0 ) \geq 0$ , in this case we have $\| g _ { i } ( \hat { x } ) \| _ { D } \leq T$ and due to the global optimality of $\hat { x }$ we have

$$
\begin{array} { r } { \gamma \left. g _ { i } ( \hat { x } ) \right. _ { D } - \mathfrak { r } ^ { \top } g _ { i } ( \hat { x } ) + \beta \left. g _ { i } ( \hat { x } ) \right. _ { 0 } \geq \gamma \left. \hat { x } \right. _ { D } - \mathfrak { r } ^ { \top } \hat { x } + \beta \left. \hat { x } \right. _ { 0 } . } \end{array}
$$

These together imply $\begin{array} { r } { | \hat { x } [ i ] | \geq \frac { \beta } { | \mathfrak { r } [ i ] | } } \end{array}$

Case II: $h ( 0 ) < 0$ and $f ( 0 ) \geq f ( \hat { x } [ i ] )$ . Then, similarly we have $\begin{array} { r } { | \hat { x } [ i ] | \geq \frac { \beta } { \gamma \sqrt { d _ { i } [ i ] } + | \mathfrak { r } [ i ] | } . } \end{array}$

Case III: $f ( 0 ) < f ( \hat { x } [ i ] )$ and $h ( 0 ) < 0$ . Since $g _ { i } ( \hat { x } ) \neq 0$ we can define $\begin{array} { r } { \hat { u } = \frac { T } { \Vert g _ { i } ( \hat { x } ) \Vert _ { D } } g _ { i } ( \hat { x } ) } \end{array}$ . ˆu is a feasible point so we require $\mathcal G _ { \beta } ( \hat { x } ) \le \mathcal G _ { \beta } ( \hat { u } )$ . Then we have

$$
\gamma T - \mathfrak { r } ^ { \top } \hat { x } + \beta \| \hat { x } \| _ { 0 } \leq \gamma T - \mathfrak { r } ^ { \top } \hat { u } + \beta ( \| \hat { x } \| _ { 0 } - 1 ) \quad \Rightarrow \quad \beta \leq \mathfrak { r } ^ { \top } ( \hat { x } - \hat { u } ) .
$$

Substituting the decomposition $\hat { x } = g _ { i } ( \hat { x } ) + \hat { x } [ i ] e _ { i }$ and the definition of $\hat { u } ,$ we obtain:

$$
\begin{array} { r l } & { \beta \leq \mathfrak { r } ^ { \top } \left( g _ { i } ( \hat { x } ) + \hat { x } [ i ] e _ { i } - \displaystyle \frac { T } { \| g _ { i } ( \hat { x } ) \| _ { D } } g _ { i } ( \hat { x } ) \right) = \mathfrak { r } [ i ] \hat { x } [ i ] + \left( 1 - \displaystyle \frac { T } { \| g _ { i } ( \hat { x } ) \| _ { D } } \right) \mathfrak { r } ^ { \top } g _ { i } ( \hat { x } ) } \\ & { \quad = \mathfrak { r } [ i ] \hat { x } [ i ] + ( \| g _ { i } ( \hat { x } ) \| _ { D } - T ) \displaystyle \frac { \mathfrak { r } ^ { \top } g _ { i } ( \hat { x } ) } { \| g _ { i } ( \hat { x } ) \| _ { D } } . } \end{array}
$$

Since $h ( 0 ) < 0$ , we have $\| g _ { i } ( \hat { x } ) \| _ { D } > T$ , so the coeficient $( \lVert g _ { i } ( \hat { x } ) \rVert _ { D } - T )$ is positive. By the definition of the dual norm, we have $\begin{array} { r } { \frac { \mathfrak { r } ^ { \top } g _ { i } ( \hat { x } ) } { \| g _ { i } ( \hat { x } ) \| _ { D } } \leq H } \end{array}$ . Applying this upper bound yields:

$$
\beta \leq \mathfrak { r } [ i ] \hat { x } [ i ] + ( \Vert { g } _ { i } ( \hat { x } ) \Vert _ { D } - T ) H .
$$

Next, using the triangle inequality $\| g _ { i } ( \hat { x } ) \| _ { D } \leq \| \hat { x } \| _ { D } + \| \hat { x } [ i ] e _ { i } \| _ { D }$ and noting that $\| \hat { x } \| _ { D } = T$ , we have $\Vert g _ { i } ( \hat { x } ) \Vert _ { D } - T \leq \Vert \hat { x } [ i ] e _ { i } \Vert _ { D } = | \hat { x } [ i ] | \sqrt { d _ { i } [ i ] }$ . Substituting this back into the inequality and using $\mathfrak { r } [ i ] \hat { x } [ i ] \leq | \mathfrak { r } [ i ] | | \hat { x } [ i ] | \cdot$

$$
\beta \leq \left| \mathbf { r } [ i ] \right| \left| \hat { x } [ i ] \right| + \left| \hat { x } [ i ] \right| \sqrt { d _ { i } [ i ] } H = \left| \hat { x } [ i ] \right| \left( \left| \mathbf { r } [ i ] \right| + \sqrt { d _ { i } [ i ] } H \right) \Rightarrow | \hat { x } [ i ] | \geq \frac { \beta } { H \sqrt { d _ { i } [ i ] } + | \mathbf { r } [ i ] | } .
$$

We observe that this lower bound is suitable for all three cases above. Finally, $\mathrm { i f } | \hat { \sigma } | = 1$ then we have closed form solutions for the specific entry $i \in { \hat { \sigma } }$

$$
\hat { x } [ i ] = \frac { T \mathrm { s i g n } ( \mathfrak { r } [ i ] ) } { \sqrt { d _ { i } [ i ] } } \quad \Rightarrow \quad | \hat { x } [ i ] | = \frac { T } { \sqrt { d _ { i } [ i ] } } .
$$

Combining the bounds from the $| \hat { \sigma } | \geq 2$ and $| \hat { \sigma } | = 1$ cases yields the desired result.

Remark 12. The lower bound in Theorem 3 is simpler than the one in Theorem 1: it is linear in $\beta _ { i }$ , does not involve the pairwise quantities $\rho _ { i } ^ { \phantom { } }$ , and depends on the data only through ${ \mathfrak { r } } [ i ] , d _ { i } [ i ]$ , H and T. In our experiments it also separated the nonzero components of global minimizers from zero more clearly, which is the property exploited by the warm-start heuristic of Section 5.1.

Because the feasible set is compact, the existence of a global minimizer can be verified more straightforwardly than in the earlier case. We first establish that the current objective function is lower semi-continuous for any selection of problem parameters.

Lemma 6. For $\beta > 0$ , the robust return maximization objective $\mathcal { G } _ { \beta }$ is lower semi-continuous.

Proof. The function $\lVert \cdot \rVert _ { 0 } : \mathbb { R } ^ { N } \to \mathbb { R }$ is lower semi-continuous [21, Proof of Proposition 4.3]. Consequently, the objective function $\mathcal { G } _ { \beta } ( \boldsymbol { x } ) = \gamma \left\| \boldsymbol { x } \right\| _ { D } - \boldsymbol { \mathfrak { r } } ^ { \top } \boldsymbol { x } + \beta \left\| \boldsymbol { x } \right\| _ { 0 }$ is also lower semi-continuous, as it is a sum of lower semi-continuous functions. □

Theorem 4. Let $\beta > 0$ . Then the set of global minimizers of $\mathcal { G } _ { \beta }$ ,

$$
\hat { Y } = \left\{ \hat { x } \in \mathcal { V } : \mathcal { G } _ { \beta } ( \hat { x } ) = \underset { x \in \mathcal { V } } { \operatorname* { m i n } } \mathcal { G } _ { \beta } ( x ) \right\}
$$

is nonempty.

Proof. This result follows directly from the extended form of the Weierstrass’ extreme value theorem: a lower semicontinuous function attains its minimum over a compact feasible set. □

Finally, before concluding the theoretical developments, we observe that Proposition 6 and Remark 7 extend directly to the robust return maximization problem. Consequently, for any prescribed sparsity level k, there exists a regularization parameter $\beta _ { k } > 0$ that ensures the desired sparsity level in globally optimal portfolios. Furthermore, globally optimal portfolios can be identified among $2 ^ { N } - 1$ locally optimal candidates, where this count arises from considering only nontrivial support sets.

In the next section, we provide a brief description of the proposed algorithm and the associated heuristic procedures.

## 5 Enumeration Based BnB Algorithm

In this section, we develop an enumeration-based branch-and-bound algorithm for $\left( \mathcal { P } ^ { 1 } \right)$ and $\left( \mathcal { P } ^ { 2 } \right)$ following the scheme of [22] for the nonrobust counterpart and adapting it to the robust setting through the bounds established in Sections 3 and 4. We further refine the scheme with an additional pruning step in the bounding of right nodes, which allows an entire subtree to be replaced by a single leaf evaluation.

We denote by $P \subseteq \mathbb { I } _ { N }$ the set of candidate assets that remain, whether or not the heuristic is applied, and decompose the problem into subproblems $\left( \mathcal { P } _ { \omega } ^ { 1 } \right)$ over support subsets $\omega \subseteq P$ . Each subproblem characterizes the local minimizers supported on ω and corresponds to a node in the enumeration tree. At each node we solve the lower-dimensional equivalent subproblem $\big ( \mathcal { Z P } _ { \omega } ^ { 1 } \big )$ and maintain a five-tuple $( x , l b , u b , P , S )$ , whose components are as follows.

• x represents the solution obtained by solving $\big ( \mathcal { Z } \mathcal { P } _ { \omega } ^ { 1 } \big )$ for a nonempty support subset ω. If $\omega = \emptyset$ , then the node is pruned.

• lb represents a lower score, which is less than or equal to the upper bound at each node. For ease of reference, it will be referred to as the lower bound for the remainder of the paper.

• ub represents an upper bound on the optimal value obtained from the corresponding node.

• P denotes the set of candidate assets whose inclusion is still undecided, from which the branching asset is selected.

• S denotes the set of assets already fixed in the support at this node.

Each node produced by the algorithm is inserted into a priority queue, with its priority determined by the previously defined lower bounds. At the beginning of the next iteration, the node with the highest priority $( i . e . ,$ the lowest lower bound) is removed from the priority queue, and the values of $x , l b , u b , P$ , and S are updated according to this node. If multiple nodes share the same lowest lower bound, the node that was added to the queue first is selected. The branching process then proceeds from this chosen node, allowing the algorithm to systematically explore the search space. In the following subsections, we outline each step of the algorithm and present Algorithm 1, which summarizes the entire procedure. All subroutines can be applied to $\left( \mathcal { P } ^ { 2 } \right)$ with minor modifications, and in the following sections we present computational results for both problems.

## 5.1 Warm-Start Heuristic

Computation time increases with larger N, because the algorithm may encounter many suboptimal solutions, and the number of such solutions afects eficiency. To address this, a warm-start heuristic is introduced to reduce memory usage and problem size.

The method first solves the problem with the full support set and determines a conservative elimination level using componentwise lower bounds. Then, for each asset $i \in P$ , we compute the deficit $b [ i ] - | u _ { P } [ i ] |$ between the componentwise lower bound b[i] of Theorem 1 and the weight $u _ { P } [ i ]$ that asset i receives in the solution $u _ { P } = \xi ( P )$ on the full candidate set, and remove the assets with the largest deficits from the candidate support set, thereby reducing the dimension of the problem early in the process. For instance, if the goal is a 10-sparse solution among 60 variables, about 30% of variables might be eliminated rather than removing all but 10, in order to avoid excluding potentially optimal components.

This approach speeds up computation but introduces a trade-of between solution quality and runtime, so the elimination level must be chosen carefully. Guidance for this choice can come from sparsity information provided by Proposition 6. If no warm-start is applied, that is, if no support is eliminated, the method reduces to the full branch-and-bound algorithm. The validity of the associated bounding and pruning scheme follows from the generic enumeration argument in [22, Appendix B]; consequently, when the algorithm terminates, the returned incumbent is globally optimal up to the prescribed stopping tolerance. The heuristic is most beneficial when the true solution is sparse, which aligns with portfolio optimization practice, since sparse portfolios reduce transaction costs.

## 5.2 Branching

When a node is taken from the queue for examination, the first step is to determine the most promising asset by analyzing the gradient of the Lagrangian function of $\big ( \mathcal { Z P } _ { \omega } ^ { 1 } \big )$ . This branching strategy evaluates the quality of the current solution and aims to improve it. If the selected asset set at a node is empty, the procedure is modified by choosing the index that minimizes the varianceto-return ratio.

For branching, we represent the Lagrangian function L by associating a Lagrange multiplier with the constraint defining $\mathcal { X } _ { \omega }$ and compute its gradient $( \nabla L )$ using Proposition 2. We then identify the most promising candidate among the set of possible assets. If the set of selected assets is nonempty, $i . e . , S \neq \emptyset$ , we select $j \in$ arg $\operatorname* { m a x } _ { i \in P } \left| \nabla L _ { i } \right|$ . In this case, rather than using the covariance matrix $D _ { P } ,$ we extract the submatrix $D _ { P , S }$ , whose rows correspond to indices in $P$ and columns correspond to indices in $S _ { \ i }$ , ensuring dimensional compatibility in the gradient computation.

If $S = \emptyset$ , the gradient rule is unavailable, since the submatrix ${ D _ { P , S } }$ is empty. In this case, we apply an alternative branching rule based on the variance-to-return ratio of assets in $P ,$ and select $j \in \arg \operatorname* { m i n } _ { i \in P } \mathrm { d i a g } ( D _ { P } ) [ i ] / \mathfrak { r } _ { P } [ i ]$ . This rule favors assets with relatively low variance and high return.

In both cases, an asset $j$ is selected from the candidate set $P _ { \mathrm { : } }$ , and the algorithm generates two distinct nodes: a left node in which the chosen asset is included in the support $S ^ { L }  S \cup \{ j \}$ , and a right node in which the asset is excluded from the support $P ^ { R }  P \setminus \{ j \} , S ^ { R }  S \cup \hat { P ^ { R } }$ . As a result, at every iteration the algorithm identifies the asset that appears most influential for the portfolio, which helps accelerate convergence relative to a standard BnB procedure. The primary objective is to obtain solutions with as few nonzero components as possible, that is, with small support sets.

## 5.3 Bounding

At the start of each iteration, the algorithm removes from the queue the tuple with the highest priority and updates the corresponding values $( x , l b , u b , P , S )$

For left nodes, an upper bound is obtained by solving the subproblem with fixed support $S ^ { L }$ and evaluating an upper estimate using the objective value of $\left( \mathcal { P } ^ { 1 } \right)$ at the resulting solution, $u b ^ { L } = \xi ( S ^ { L } ) ^ { \top } D _ { S ^ { L } } \xi ( S ^ { L } ) + \beta | S ^ { L } |$ . The lower bound of a left node is computed by adding the sparsity penalty $\beta$ to the parent node’s lower bound, $l b ^ { L } = l b + \beta$ . Depending on the triviality of the support, the node is then added to the queue with the updated bounds.

For right nodes, the upper bound is inherited from the parent node because the branching step excludes an asset from the candidate set without generating a new feasible solution with fixed support. The lower bound is obtained from the relaxed subproblem of type $\left( \mathcal { Z P } _ { \omega } ^ { 1 } \right)$ . At initialization, a global lower bound is obtained by solving the problem with $\omega = \mathbb { I } _ { N }$ , since this formulation omits the sparsity penalty and considers all indices. Following [22], after excluding the branching asset,

the right child is assigned the bounds

$$
u b ^ { R } = u b , \qquad l b ^ { R } = \xi ( S ^ { R } ) ^ { \top } D _ { S ^ { R } } \xi ( S ^ { R } ) + \beta \left| S \right| .\tag{6}
$$

In the scheme of [22], the right child is then enqueued. Here, before enqueuing the right child, we apply the following additional pruning step. We first check whether

$$
l b ^ { R } + \beta \geq u b ^ { * } - \epsilon ,
$$

where $u b ^ { * }$ denote the objective value of the incumbent, that is, the best feasible solution found so $\operatorname { f a r } ,$ and ϵ is the prescribed optimality tolerance. If this condition holds, then the nonterminal descendants of the right subtree are pruned. The following proposition justifies this additional pruning step.

Proposition 11. Consider $\left( \mathcal { P } ^ { 1 } \right)$ and let a node be represented by the current support set $S$ and candidate set P. Let $j \in \dot { P } , \dot { P ^ { R } } = P \setminus \{ j \}$ , and $S ^ { \hat { R } } = S \cup P ^ { \hat { R } }$ . Then $l b ^ { R }$ as defined in $( 6 )$ satisfies the following property: if $l b ^ { R } + \beta \geq u b ^ { * } - \epsilon$ , then every feasible portfolio $x \in \mathcal { X }$ such that $S \subsetneq \sigma ( x ) \subseteq S ^ { R }$ satisfies $\mathcal { F } _ { \beta } ( x ) = x ^ { \top } D x + \beta \| x \| _ { 0 } \geq u b ^ { * } - \epsilon$ . Consequently, every descendant support $\tilde { S }$ with $S \subsetneq \tilde { S } \subseteq S ^ { R }$ may be pruned.

Proof. Let $x \in \mathcal { X }$ satisfy $S \ \subsetneq \sigma ( x ) \ \subseteq \ S ^ { R }$ , and define $\tilde { S } = \sigma ( x )$ . Since $\tilde { S } \subseteq S ^ { R }$ , all nonzero components of x lie in $S ^ { R }$ . Therefore, the restriction of x to the indices in $S ^ { R }$ is feasible for $\left( \mathcal { Z P } _ { \omega } ^ { 1 } \right)$ with $\omega = S ^ { R }$ . By optimality of $\xi ( S ^ { R } )$ , we obtain

$$
x ^ { \top } D x \geq \xi ( S ^ { R } ) ^ { \top } D _ { S ^ { R } } \xi ( S ^ { R } ) .
$$

Since $S \subsetneq { \tilde { S } }$ , we also have $\| x \| _ { 0 } = | \tilde { S } | \geq | S | + 1$ . Hence, combining the two inequalities, we get

$$
\begin{array} { r } { \mathcal { F } _ { \beta } ( \boldsymbol { x } ) = \boldsymbol { x } ^ { \top } D \boldsymbol { x } + \beta \| \boldsymbol { x } \| _ { 0 } \geq \xi ( S ^ { R } ) ^ { \top } D _ { S ^ { R } } \xi ( S ^ { R } ) + \beta ( | S | + 1 ) = l b ^ { R } + \beta . } \end{array}
$$

Thus, if $l b ^ { R } + \beta \geq u b ^ { * } - \epsilon$ , then $\mathcal { F } _ { \beta } ( x ) \ge u b ^ { * } - \epsilon$ for every feasible $x \in \mathcal { X }$ with $S \subsetneq \sigma ( x ) \subsetneq S ^ { R }$ Therefore, no feasible portfolio associated with a descendant support $\tilde { S }$ satisfying $S \subsetneq \tilde { S } \subseteq S ^ { R }$ can improve the incumbent by more than ϵ. Consequently, every such descendant support $\tilde { S }$ may be pruned. □

The efect of this pruning step can also be quantified. Indeed, consider the right child defined by branching on $j ,$ , namely the node with candidate set $P ^ { R } = P \setminus \{ j \}$ and current support S. Under the standard enumeration scheme [22], this node would be enqueued and explored. Since its support remains $S$ and only indices from $P ^ { \bar { R } }$ remain available for future branching, every support generated in that subtree satisfies $S \subseteq \tilde { S } \subseteq S ^ { R }$ . When $l b ^ { R } + \beta \geq u b ^ { * } - \epsilon$ , Proposition 11 shows that all supports with $S \subsetneq \tilde { S } \subseteq S ^ { R }$ may be pruned. The number of such supports is

$$
\sum _ { i = 1 } ^ { \left| P ^ { R } \right| } { \binom { \left| P ^ { R } \right| } { i } } = 2 ^ { \left| P ^ { R } \right| } - 1 .
$$

Therefore, the additional pruning step may remove an exponentially large portion of the right subtree in a single iteration. An analogous statement holds for $\left( \mathcal { P } ^ { 2 } \right)$ after replacing $x ^ { \top }$ Dx by $\gamma \| \boldsymbol { x } \| _ { D } - \mathfrak { r } ^ { \top } \boldsymbol { x }$ and $\xi$ by π. The details are omitted for brevity.

Algorithm 1: A Branch-and-Bound Algorithm for (P<sup>1</sup>)   
Input: tolerance $\epsilon = 1 E - 8 ,$ , local and global Subroutine: Branch&Bound(q, x, lb, ub, P, S)   
upper bounds $u b = u b ^ { * } = \infty .$ candidate support Input: queue $q$ and node $( x , l b , u b , P , S )$   
set $P = \{ 1 , \ldots , N \}$ , current support set $S = \emptyset .$ if $P \neq \emptyset$ then   
fixed-support solution $u _ { P } = \xi ( P )$ , lower bound Branching:   
$l b$ from solving $\left( \mathcal { Z P } _ { \omega } ^ { 1 } \right)$ with $\omega = P ,$ the cor- if $| S | \ge 1$ then   
responding componentwise lower bound b from Select $j \in \mathrm { a r g m a x } _ { i \in P } | \nabla L _ { i } | .$   
Theorem 1, and queue $q = \varnothing .$ else   
Warm-Start: let $J _ { k } \subseteq P$ be the set of in- Select $\begin{array} { r } { j \in \arg \operatorname* { m i n } _ { i \in P } \frac { \mathrm { d i a g } ( D _ { P } ) [ i ] } { \mathfrak { r } _ { P } [ i ] } } \end{array}$   
dices corresponding to the k largest values of end   
$b [ i ] - | u _ { P } [ i ] |$ , and update $P  P \backslash J _ { k }$ . Enqueue Left node: $S ^ { L }  S \cup \{ j \}$   
$( u _ { P } , l b _  $ , ub, P, S) into q. Right node: $P ^ { R }  P \backslash \{ j \} , S ^ { R }  P ^ { R } \cup S .$   
while $q \neq \emptyset$ and $u b ^ { * } - l b > \epsilon$ do Bounding the Right Node:   
Extract the highest-priority tuple if $| S ^ { R } | \geq 1$ then   
$( x , l b , u b , P , S )$ from $q .$ $l b ^ { R } = \xi ( S ^ { R } ) ^ { \top } D _ { S ^ { R } } \xi ( S ^ { R } ) + \beta \left. S \right.$   
if $u b < u b ^ { * }$ then if $l b ^ { R } + \beta < u b ^ { * } - \epsilon$ then   
Set $u b ^ { * } $ ub and $x ^ { * }  x .$ Enqueue $( x , l b ^ { R } , u b , P ^ { R } , S )$ into $q .$   
end   
end   
q ← Branch&Bound $( q , x , l b , u b , P , S )$   
end   
end   
Bounding the Left Node:   
return: $u b ^ { * }$ and its corresponding solution $x ^ { * }$ if $\left| S ^ { L } \right| \geq 1$ then   
$u b ^ { \dot { L } } = \xi ( S ^ { L } ) ^ { \top } D _ { S ^ { L } } \xi ( S ^ { L } ) + \beta \left| S ^ { L } \right|$   
Enqueue $( \xi ( S ^ { L } ) , l b + \beta , u b ^ { L } , \dot { P ^ { R } } , \dot { S ^ { L } } )$ into $q .$   
end   
end   
return: updated queue $q .$

## 5.4 Termination

The algorithm terminates when there are no remaining nodes to explore or when the best upper bound $u b ^ { * }$ found during the search matches the lower bound lb of the current node within a userdefined tolerance. In this case, further exploration cannot yield a better solution, and the procedure stops early.

## 6 Computational Results

The algorithms were evaluated using datasets sourced from [11] and [4]. The former includes daily price data (adjusted for dividends and stock splits) for the DowJones, EuroStoxx50, FTSE100, NASDAQ100, and S&P 500 indices. The latter comprises ETF, Eurobonds, and Italian Bonds datasets, providing daily asset returns derived from prices and total returns adjusted for dividends and splits. Summary statistics of these datasets are presented in Table 1.

Experiments were run on a Linux cluster using a single CPU core and no GPU. The runs were carried out on nodes equipped with Intel Xeon Gold 6240 processors and 376 GiB RAM. The implementation used Python and Gurobi 13.0.2. The code used to generate all reported results is available at https://github.com/ecyayla/robust-mean-variance-portfolio-optimization. We benchmarked our method, enhanced with the warm-start heuristic, against Gurobi using the mixed-integer second-order cone reformulations of $\left( \mathcal { P } ^ { 1 } \right)$ and $\left( \mathcal { P } ^ { 2 } \right)$ . For both approaches the stopping tolerance was set to $1 0 ^ { - 8 }$ : the relative MIP gap tolerance for Gurobi, and the tolerance ϵ in the termination test $u b ^ { * } - l b \leq \epsilon$ for the branch-and-bound algorithm. To ensure a fair comparison, the number of threads used by Gurobi was restricted to one and its time limit was set to 12 hours. In the benchmarking experiments reported in Tables 2–3, the return of the risk-free asset $r _ { c }$ is set to 0.0002, the target return $\bar { r }$ to $5 \%$ above $r _ { c } ,$ and $T = 1$ for $\left( \mathcal { P } ^ { 2 } \right)$ . Our support-wise analysis relies on Assumption 2, which guarantees feasibility of every nonempty support-restricted subproblem of $\left( \mathcal { P } ^ { 1 } \right)$ and excludes degenerate support solutions of $\left( \mathcal { P } ^ { 2 } \right)$ . We fix $\gamma = 0 . 0 0 1$ and retain, before running either method, the assets satisfying $| \mathfrak { r } [ i ] | / \sqrt { d _ { i } [ i ] } \le \gamma$ . The same retained investment universe is used by BnB and Gurobi, and the asset counts in Table 1 report the resulting dimensions.

Tables 2–3 summarize the computational performance of the proposed BnB algorithm and Gurobi. The columns labeled “BnB CPU Time $( \mathrm { s } ) ^ { \dag }$ and “Gurobi CPU Time $( \mathrm { s } ) ^ { \dag }$ report total CPU times in seconds. Entries marked with a dash $\left( \begin{array} { c } { { \cdots } } \\ { { } } \end{array} \right)$ indicate that Gurobi did not certify optimality within 12 hours; its incumbent at termination is used to compute the reported sparsity and error. The “Solution Sparsity” column denotes the number of nonzero entries in the reported solution; when a single value is listed, both methods return solutions with the same sparsity, whereas when two values are listed, the first corresponds to BnB and the second to Gurobi. The column “Drop $\mathrm { R a t e } ^ { \mathrm { \ ' } }$ represents the proportion of assets removed during the warm-start phase, with 0 indicating that no elimination is performed. The column labeled “Error $( \% ) ^ { , }$ reports the relative objective error $1 0 0 ( v _ { \mathrm { B n B } } - v _ { \mathrm { G } } ) / \left| v _ { \mathrm { G } } \right|$ , where $v _ { \mathrm { B n B } }$ and $v _ { \mathrm { G } }$ denote the objective values of the solutions returned by BnB and by Gurobi at termination, respectively. For $\left( \mathcal { P } ^ { 2 } \right)$ , the optimal value is sometimes negative, which is why the denominator carries an absolute value. Since Gurobi is run under a fixed time limit, this value should be interpreted relative to its best available feasible solution when optimality is not certified. A negative value therefore indicates that our BnB method obtained a strictly better objective than the solution Gurobi returned at the time limit. When Gurobi terminates before the time limit, it certifies global optimality up to its gap and feasibility tolerances; in that case, no method can produce a better objective value for the same instance beyond these tolerances. Accordingly, our method is not intended to improve upon such solutions, but rather to deliver high-quality solutions within reasonable computation times, particularly for larger instances.

Finally, the column “Node Reduction $( \% ) ^ { , }$ reports the percentage reduction in the number of nodes explored by the branch-and-bound algorithm due to the additional pruning step of Proposition 11, relative to the standard enumeration scheme of [22]; a larger value indicates that a greater portion of the search tree is eliminated.

Table 2 presents the corresponding results for $\left( \mathcal { P } ^ { 1 } \right)$ across datasets of varying sizes. On the small-scale instances such as ItalianBonds, ETF, DowJones, and EuroStoxx50, both methods return identical solutions with no elimination (drop rate 0); although the runtimes are small in absolute terms, BnB is already consistently faster than Gurobi. As the problem size increases (FTSE100 and NASDAQ100), the impact of the warm-start heuristic becomes more evident: by eliminating poorly performing assets it substantially reduces the problem size, and the returned solutions remain optimal (zero error) across all drop rates, with higher drop rates yielding the largest speedups and smaller drop rates increasing computation time. On FTSE100, Gurobi requires long computation times or fails to finish within the 12-hour limit, whereas BnB returns the same optimal solutions in seconds to a few minutes at higher drop rates, its runtime growing as the drop rate decreases. A similar pattern holds on NASDAQ100, where BnB outperforms Gurobi across all values of $\beta$ and drop rates while matching its optimal objective. For the largest dataset, S&P500, Gurobi does not certify a solution within the time limit, while BnB returns solutions much faster but with a nonzero objective error (roughly 6% to 14%); here smaller drop rates reduce the error at the expense of additional computation time. Overall, the results suggest that the warm-start elimination strategy removes poorly performing assets and reduces the problem size substantially, while preserving optimality on all instances except the largest, where a controllable trade-of between computation time and solution quality remains. Finally, the additional pruning step substantially reduces the search efort on every instance, removing between 20% and 80% of the nodes explored by the enumeration scheme of [22].

Table 3 reports the results for $\left( \mathcal { P } ^ { 2 } \right)$ , which follow a pattern comparable to that of $\left( \mathcal { P } ^ { 1 } \right)$ . On the smaller datasets (ItalianBonds, ETF, DowJones, and EuroStoxx50) no assets are eliminated and the two methods return nearly identical portfolios, with BnB several times faster than Gurobi. For EuroStoxx50 with $\beta = 5 \times 1 0 ^ { - 4 }$ , the reported −0.74% error is due to numerical rounding rather than a genuine improvement over Gurobi. On the larger instances (FTSE100 and NASDAQ100), Gurobi no longer certifies optimality within the 12-hour limit, and the warm-start heuristic becomes decisive: higher drop rates yield larger speedups at a small cost in accuracy, whereas lower drop rates reduce the error at the expense of additional computation time. On FTSE100 the error even becomes negative at the lower drop rates, where BnB improves upon the solution Gurobi returns at the time limit (for $\beta = 1 0 ^ { - 3 }$ , from 1.84% at drop rate 0.8 to −1.39% at 0.6), while on NASDAQ100 it stays below 1.3%. For the largest dataset, S&P500, BnB is again substantially faster, although the accuracy gap widens to between roughly 11% and 24% and narrows as the drop rate is reduced. Overall, the warm-start elimination substantially reduces the computation time of the larger instances and usually produces good-quality solutions, at the cost of some accuracy loss when the elimination is too aggressive. The node reduction is more modest here, ranging from 0% to roughly 42%, and it is largest on the instances whose optimal solutions are sparsest. This is not surprising because the pruning condition $l b ^ { R } + \beta \geq u b ^ { * } - \epsilon$ is harder to satisfy when the portfolios have larger supports. Indeed, the reported solutions of $\left( \mathcal { P } ^ { 2 } \right)$ contain 5–271 nonzero components, compared with only 1–8 for (P<sup>1</sup>).

## 6.1 Out-of-Sample Performance

We compare the robust $( \gamma > 0 )$ and purely sparse $( \gamma = 0 )$ models out of sample. We run a rolling-window backtest on the 100 size and book-to-market-sorted assets from the Kenneth French

Table 1: Dataset characteristics
<table><tr><td>Index</td><td>Num. of assets</td><td>Days</td><td>Time interval</td></tr><tr><td>ItalianBonds</td><td>11</td><td>1564</td><td>1/2013-12/2018</td></tr><tr><td>ETF</td><td>24</td><td>1042</td><td>1/2015-12/2018</td></tr><tr><td>DowJones</td><td>28</td><td>4276</td><td>10/2006-2/2023</td></tr><tr><td>EuroStoxx50</td><td>46</td><td>4276</td><td>10/2006-2/2023</td></tr><tr><td>FTSE100</td><td>82</td><td>4276</td><td>10/2006-2/2023</td></tr><tr><td>NASDAQ100</td><td>70</td><td>4276</td><td>10/2006-2/2023</td></tr><tr><td>S&amp;P500</td><td>420</td><td>4276</td><td>10/2006-2/2023</td></tr></table>

Table 2: Computational results comparison between BnB and Gurobi for $\left( \mathcal { P } ^ { 1 } \right)$
<table><tr><td>Dataset</td><td> $\beta$ </td><td>Solution Sparsity</td><td>Drop Rate</td><td>BnB CPU Time (s)</td><td>Gurobi CPU Time (s)</td><td>Error (%)</td><td>Node Reduction  $( \% )$ </td></tr><tr><td>ItalianBonds</td><td>5E-5</td><td>1</td><td>0</td><td>0.01</td><td>0.05</td><td>0</td><td>35.35</td></tr><tr><td rowspan="3">ETF</td><td>1E-5</td><td>2</td><td>0</td><td>0.03</td><td>0.06</td><td>0</td><td>20.22</td></tr><tr><td>1E-4</td><td>2</td><td>0</td><td>0.06</td><td>0.27</td><td>0</td><td>57.43</td></tr><tr><td>5E-5</td><td>3</td><td>0</td><td>0.12</td><td>0.57</td><td>0</td><td>48.02</td></tr><tr><td rowspan="2">DowJones</td><td>5E-4</td><td>1</td><td>0</td><td>0.01</td><td>0.26</td><td>0</td><td>65.45</td></tr><tr><td>1E-4</td><td>2</td><td>0</td><td>0.02</td><td>0.42</td><td>0</td><td>71.30</td></tr><tr><td rowspan="2">EuroStoxx50</td><td>5E-4</td><td>1</td><td>0</td><td>0.02</td><td>0.67</td><td>0</td><td>79.78</td></tr><tr><td>1E-4</td><td>3</td><td>0</td><td>0.52</td><td>3.28</td><td>0</td><td>73.20</td></tr><tr><td rowspan="9">NASDAQ100</td><td>1E-4</td><td>2</td><td>0.5</td><td>1.20</td><td>56.24</td><td>0</td><td>70.30</td></tr><tr><td>1E-4</td><td>2</td><td>0.4</td><td>2.87</td><td>56.24</td><td>0</td><td>75.07</td></tr><tr><td>5E-5</td><td>3</td><td>0.5</td><td>5.44</td><td>264.82</td><td>0</td><td>61.92</td></tr><tr><td>5E-5</td><td>3</td><td>0.4</td><td>17.23</td><td>264.82</td><td>0</td><td>68.95</td></tr><tr><td>1E-5</td><td>8</td><td>0.5</td><td>132.76</td><td>35365.32</td><td>0</td><td>42.39</td></tr><tr><td>1E-5</td><td>8</td><td>0.4</td><td>1261.74</td><td>35365.32</td><td>0</td><td>44.01</td></tr><tr><td>5E-5</td><td>3</td><td>0.7</td><td>0.24</td><td>365.39</td><td>0</td><td>47.97</td></tr><tr><td>5E-5</td><td>3</td><td>0.6</td><td>1.26</td><td>365.39</td><td>0</td><td>54.00</td></tr><tr><td>5E-5</td><td>3</td><td>0.5</td><td>3.02</td><td>365.39</td><td>0</td><td>57.60</td></tr><tr><td rowspan="6">S&amp;P500</td><td>1E-5</td><td>8</td><td>0.7</td><td>4.40</td><td></td><td>0</td><td>38.91</td></tr><tr><td>1E-5</td><td>8</td><td>0.6</td><td>66.27</td><td></td><td>0</td><td>41.48</td></tr><tr><td>1E-5</td><td>8</td><td>0.5</td><td>386.43</td><td></td><td>0</td><td>48.11</td></tr><tr><td>5E-5</td><td>3</td><td>0.9</td><td>37.64</td><td></td><td>14.03</td><td>67.85</td></tr><tr><td>5E-5</td><td>3-8</td><td>0.85</td><td>377.10</td><td></td><td>6.68</td><td>74.52</td></tr><tr><td>1E-5</td><td>8-3</td><td>0.9</td><td>6412.85</td><td>1</td><td>5.82</td><td>49.17</td></tr></table>

Table 3: Computational results comparison between BnB and Gurobi for $\left( \mathcal { P } ^ { 2 } \right)$
<table><tr><td rowspan="2">Dataset</td><td rowspan="2"> $\beta$ </td><td rowspan="2">Solution Sparsity</td><td rowspan="2">Drop Rate</td><td rowspan="2">BnB CPU Time (s)</td><td rowspan="2">Gurobi CPU Time (s)</td><td rowspan="2">Error (%)</td><td rowspan="2">Node Reduction (%)</td></tr><tr><td></td></tr><tr><td>ItalianBonds</td><td>5E-3</td><td>5</td><td>0</td><td>0.04</td><td>0.11</td><td>0</td><td>0.00</td></tr><tr><td rowspan="3">ETF</td><td>1E-3</td><td>6</td><td>0</td><td>0.03</td><td>0.20</td><td>0</td><td>0.00</td></tr><tr><td>1E-3</td><td>14</td><td>0</td><td>0.62</td><td>5.98</td><td>0</td><td>15.04</td></tr><tr><td>5E-4</td><td>17</td><td>0</td><td>0.28</td><td>2.81</td><td>0</td><td>11.20</td></tr><tr><td rowspan="2">DowJones</td><td>1E-3</td><td>9</td><td>0</td><td>0.32</td><td>7.46</td><td>0</td><td>41.61</td></tr><tr><td>5E-4</td><td>12</td><td>0</td><td>0.34</td><td>6.03</td><td>0</td><td>30.53</td></tr><tr><td rowspan="2">EuroStoxx50</td><td>5E-4</td><td>14-15</td><td>0</td><td>20.50</td><td>171.00</td><td>-0.74</td><td>30.42</td></tr><tr><td>1E-4</td><td>26</td><td>0</td><td>15.16</td><td>138.85</td><td>0</td><td>19.52</td></tr><tr><td rowspan="3">NASDAQ100</td><td>1E-3</td><td>17</td><td>0.5</td><td>538.27</td><td></td><td>0.48</td><td>29.25</td></tr><tr><td>5E-4</td><td>27-31</td><td>0.5</td><td>66.36</td><td></td><td>1.22</td><td>17.98</td></tr><tr><td>5E-4</td><td>30-31</td><td>0.4</td><td>1283.38</td><td></td><td>0.59</td><td>20.25</td></tr><tr><td rowspan="6">FTSE100</td><td>1E-3</td><td>11-17</td><td>0.8</td><td>0.07</td><td></td><td>1.84</td><td>3.74</td></tr><tr><td>1E-3</td><td>14-17</td><td>0.7</td><td>2.82</td><td></td><td>-0.10</td><td>13.78</td></tr><tr><td>1E-3</td><td>16-17</td><td>0.6</td><td>154.19</td><td></td><td>-1.39</td><td>19.32</td></tr><tr><td>5E-4</td><td>13-28</td><td>0.8</td><td>0.02</td><td></td><td>6.45</td><td>0.00</td></tr><tr><td>5E-4</td><td>21-28</td><td>0.7</td><td>1.06</td><td></td><td>2.44</td><td>4.30</td></tr><tr><td>5E-4</td><td>23-28</td><td>0.6</td><td>20.56</td><td></td><td>0.61</td><td>10.46</td></tr><tr><td rowspan="6">S&amp;P500</td><td>1E-3</td><td>22-37</td><td>0.9</td><td>657.65</td><td></td><td>14.41</td><td>25.49</td></tr><tr><td>5E-4</td><td>31-73</td><td>0.9</td><td>169.11</td><td></td><td>22.62</td><td>12.67</td></tr><tr><td>1E-4</td><td>58-222</td><td>0.85</td><td>8.55</td><td></td><td>24.25</td><td>0.27</td></tr><tr><td>1E-4</td><td>78-222</td><td>0.8</td><td>564.04</td><td></td><td>16.67</td><td>0.00</td></tr><tr><td>5E-5</td><td>80-271</td><td>0.8</td><td>9.79</td><td></td><td>19.90</td><td>0.03</td></tr><tr><td>5E-5</td><td>120-271</td><td>0.7</td><td>5448.01</td><td></td><td>10.94</td><td>0.00</td></tr></table>

Data Library<sup>1</sup> (daily value-weighted returns, July 2004 to June 2026). At each rebalancing date, we estimate D and r from an in-sample window of 252 trading days and evaluate the resulting allocation out of sample over the next 21 trading days. Since the estimates are recomputed at each date, Assumption 2 need not hold for the whole universe: assets with $| \mathfrak { r } [ i ] | / \sqrt { d _ { i } [ i ] } \le \gamma$ are removed from the candidate set before solving, and rebalancing dates at which no asset survives this screening are skipped. The same screening and the same set of rebalancing dates are used for the robust and the sparse model, so that the two are compared on identical out-of-sample periods. This leaves 215 rebalances over the sample period. The risk-free return is set to $r _ { c } = 5 \times 1 0 ^ { - 5 }$ and the target return to $\bar { r } = 1 . 0 5 r _ { c } .$ . Both models are solved by the branch-and-bound algorithm using the warm-start elimination of Section 5.1 at a drop rate of 0.3.

We report the annualized out-of-sample Sharpe ratio, the annualized out-of-sample mean excess return, and the average number of selected assets |σ|. Across the tested configurations the robust model attains higher out-of-sample Sharpe ratios and mean excess returns than the sparse model in most settings. We do not claim that the robust model dominates the sparse one in general. Nonetheless, these results suggest that the robustness term can improve out-of-sample performance, which makes robust sparsity a worthwhile alternative to purely sparse portfolio selection.

## 7 Conclusion

In this paper, we considered a mean–variance portfolio selection problem that accounts for uncertainty in expected returns through an ellipsoidal uncertainty set, while at the same time promoting sparsity via an ℓ<sub>0</sub>-penalty. Bringing these two aspects together leads to a problem that is both nonconvex and discontinuous, and therefore inherently dificult to solve. To better understand this structure, we carried out a detailed analysis of both local and global minimizers for the two formulations studied, namely the robust risk minimization and robust return maximization models. In doing so, we clarified how local minimizers relate to support-restricted subproblems, established existence results for global solutions, and derived explicit lower and upper bounds on their components.

Table 4: Out-of-sample robust $( \gamma > 0 )$ versus sparse $( \gamma = 0 )$ performance on the Fama–French 100 Size×Book-to-Market portfolios. Mean excess returns are annualized and expressed in percent; $| \sigma |$ is the average number of selected assets.
<table><tr><td colspan="3">Sharpe</td><td colspan="3">Mean return</td><td colspan="2"> $| \sigma |$ </td></tr><tr><td> $\beta$ </td><td> $\gamma$ </td><td>Sparse</td><td>Robust</td><td>Sparse</td><td> $( \% )$  Robust</td><td>Sparse</td><td>Robust</td></tr><tr><td> $\overline { { 1 0 ^ { - 6 } } }$ </td><td>0.10</td><td>0.3160</td><td>0.4716</td><td>0.0102</td><td>0.0945</td><td>1.06</td><td>2.95</td></tr><tr><td> $1 0 ^ { - 6 }$ </td><td>0.15</td><td>0.4505</td><td>0.5721</td><td>0.0111</td><td>0.2905</td><td>1.00</td><td>2.94</td></tr><tr><td> $5 \times 1 0 ^ { - 7 }$ </td><td>0.10</td><td>0.4308</td><td>0.4312</td><td>0.0133</td><td>0.0862</td><td>1.32</td><td>3.75</td></tr><tr><td> $5 \times 1 0 ^ { - 7 }$ </td><td>0.15</td><td>0.4987</td><td>0.5704</td><td>0.0119</td><td>0.2896</td><td>1.06</td><td>3.38</td></tr></table>

These structural results guided the design of a tailored branch-and-bound algorithm. In particular, the bounds we obtained proved useful not only for pruning the search space, but also for constructing efective warm-start solutions. This combination plays an important role in keeping the computational efort manageable, even though the underlying problem is combinatorial due to the $\ell _ { 0 }$ term.

Our computational study on real financial data suggests that the proposed approach performs competitively against general-purpose mixed-integer second-order cone programming solvers, and in many cases achieves better performance in terms of running time without sacrificing solution quality. At the same time, the portfolios obtained reflect the intended balance: they are both robust to estimation errors and sparse enough to be practically implementable.

Overall, the paper ofers a unified perspective on combining robustness and exact sparsity in portfolio optimization under ellipsoidal uncertainty. There are several natural directions for future work, including the use of diferent uncertainty sets, the incorporation of additional practical constraints such as transaction costs or turnover limits, and the extension to alternative risk measures within the same framework.

## Acknowledgments

Buse S¸en would like to acknowledge support as a part of NCCR Automation, a National Centre of Competence in Research, funded by the Swiss National Science Foundation (grant number 51NF40 225155).

## References

[1] On the computation of the eficient frontier in advanced sparse portfolio optimization. 4OR, 24:1–33, 2026.

[2] Deniz Akkaya and Mustafa C¸ elebi Pınar. Minimizers of sparsity regularized Huber loss function. Journal of Optimization Theory and Applications, 187(1):205–233, 2020.

[3] Deniz Akkaya and Mustafa C¸ elebi Pınar. Minimizers of sparsity regularized least absolute deviations. Journal of Global Optimization, 2025.

[4] C¸ a˘gın Ararat, Francesco Cesarone, Mustafa C¸ elebi Pınar, and Jacopo Maria Ricci. MAD risk parity portfolios. Annals of Operations Research, 336:899–924, 2024.

[5] Aharon Ben-Tal and Arkadi Nemirovski. Robust convex optimization. Mathematics of Operations Research, 23(4):769–805, 1998.

[6] Aharon Ben-Tal, Laurent El Ghaoui, and Arkadi Nemirovski. Robust Optimization. Princeton University Press, 2009.

[7] Dimitris Bertsimas and Ryan Cory-Wright. A scalable algorithm for sparse portfolio selection. IN-FORMS Journal on Computing, 34(3):1489–1511, 2022.

[8] Dimitris Bertsimas and Melvyn Sim. The price of robustness. Operations Research, 52:35–53, 02 2004.

[9] P. Bonami and M. A. Lejeune. An exact solution approach for portfolio optimization problems under stochastic and integer constraints. Operations Research, 57(3):650–670, 2009.

[10] Joshua Brodie, Ingrid Daubechies, Christine De Mol, Domenico Giannone, and Ignace Loris. Sparse and stable Markowitz portfolios. Proceedings of the National Academy of Sciences, 106(30):12267–12272, 2009.

[11] Francesco Cesarone, Rosella Giacometti, Manuel L. Martino, and Fabio Tardella. A returndiversification approach to portfolio selection. Computational Management Science, 22(2), 2025.

[12] T.-J. Chang, Nigel Meade, John E. Beasley, and Yazid M. Sharaiha. Heuristics for cardinality constrained portfolio optimisation. Computers & Operations Research, 27(13):1271–1302, 2000.

[13] Erick Delage and Yinyu Ye. Distributionally robust optimization under moment uncertainty with application to data-driven problems. Operations Research, 58(3):595–612, 2010.

[14] Laurent El Ghaoui, Francois Oustry, and Herv´e Lebret. Robust solutions to uncertain semidefinite programs. SIAM Journal on Optimization, 9(1):33–52, 1998.

[15] Joel Goh and Melvyn Sim. Distributionally robust optimization and its tractable approximations. Operations Research, 58:902–917, 2010.

[16] Donald Goldfarb and Garud Iyengar. Robust portfolio selection problems. Mathematics of Operations Research, 28:1–38, 2003.

[17] Ken Kobayashi, Yuichi Takano, and Kazuhide Nakata. Cardinality-constrained distributionally robust portfolio optimization. European Journal of Operational Research, 309(3):1173–1182, 2023.

[18] Miguel Sousa Lobo, Maryam Fazel, and Stephen Boyd. Portfolio optimization with linear and fixed transaction costs. Annals of Operations Research, 152(1):341–365, 2007.

[19] Harry Markowitz. Portfolio selection. Journal of Finance, 7(1):77–91, 1952.

[20] Richard O. Michaud. The Markowitz optimization enigma: Is ‘optimized’ optimal? Financial Analysts Journal, 45(1):31–42, 1989.

[21] Mila Nikolova. Description of the minimizers of least squares regularized with ℓ -norm. Uniqueness of the global minimizer. SIAM Journal on Imaging Sciences, 6(2):904–937, 2013.

[22] Buse S¸en, Deniz Akkaya, and Mustafa C¸ elebi Pinar. Sparsity penalized mean–variance portfolio selection: Analysis and computation. Mathematical Programming, 211:281–318, 2025.

[23] Mustafa C¸ Pınar. On robust mean-variance portfolios. Optimization, 65(5):1039–1048, 2016.

[24] R. T. Rockafellar and R. J.-B. Wets. Variational Analysis. Springer, 1998.

[25] Wolfram Wiesemann, Daniel Kuhn, and Melvyn Sim. Distributionally robust convex optimization. Operations Research, 62(6):1358–1376, 2014.

[26] Zhongming Wu, Kexin Sun, Zhili Ge, Zhihua Allen-Zhao, and Tieyong Zeng. Sparse portfolio optimization via $\ell _ { 1 }$ over $\ell _ { 2 }$ regularization. European Journal of Operational Research, 319(3):820–833, 2024.

[27] Hongxin Zhao, Yilun Jiang, and Yizhou Yang. Robust and sparse portfolio: Optimization models and algorithms. Mathematics, 11(24), 2023.