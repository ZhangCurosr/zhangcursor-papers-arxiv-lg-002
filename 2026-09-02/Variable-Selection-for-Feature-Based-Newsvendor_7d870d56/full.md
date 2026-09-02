# Variable Selection for Feature-Based Newsvendor

Zhaoliang Yuan

School of Artificial Intelligence, The Chinese University of Hong Kong, Shenzhen, 225080014@link.cuhk.edu.cn

Jie Wang

School of Artificial Intelligence and School of Data Science, The Chinese University of Hong Kong, Shenzhen, jwang@cuhk.edu.cn

Feature-based newsvendor models use observable covariates to tailor inventory decisions, aiming to balance holding and shortage costs under demand uncertainty. However, high-dimensional feature sets often hinder interpretability and inflate data collection and implementation costs. This paper studies variable selection for the feature-based newsvendor problem under a hard cardinality constraint on the number of selected features. We formulate the resulting ℓ<sub>0</sub>-constrained empirical newsvendor problem with ℓ<sub>2</sub>-regularization, establish its computational hardness, and develop a mixed-integer second-order cone programming reformulation that strengthens the standard Big-M formulation. To enable scalability beyond exact optimization, we develop a randomized-rounding algorithm with a bi-criteria guarantee and a greedy heuristic. Statistically, we provide theoretical analysis of the resulting sparse policy estimator, including finite-sample estimation error, out-of-sample risk bounds, and support recovery guarantees. Extensive experiments on both synthetic and real data illustrate the computational and statistical trade-ofs among various baselines. Our results demonstrate that the proposed variable selection framework achieves competitive out-of-sample operational costs while using substantially fewer covariates.

Key Words: Contextual Stochastic Optimization, Variable Selection, Statistical Analysis, Newsvendor

## 1. Introduction

The newsvendor problem is a classic inventory model that seeks to determine the optimal order quantity by balancing the cost of overstocking and understocking in the face of random demand. It is crucial to obtain an accurate estimate of customer demand, which is influenced by a multitude of features, including weather, time, location, and economic conditions. In the big data era, leveraging available data that pairs these influential features (covariates) with corresponding demand observations enables us to make data-driven predictions of future customer demand and more reliable decisions, while ignoring the presence of covariates can lead to inconsistent decisions [3].

A natural paradigm for incorporating features into decision-making is contextual stochastic optimization (CSO) [31]. Broadly speaking, CSO can be divided into three classes: decisionrule optimization [27, 3], estimate-then-optimize [5], and integrated estimation and optimization [14, 21, 30]. In this paper, we focus on the decision-rule optimization approach, aiming to learn an end-to-end mapping from features to decisions. Linear decision rules are attractive because of their simplicity and tractability [3], but they may not capture complex feature-to-decision relationships. To improve out-of-sample performance , nonlinear decision rules based on kernel methods [6, 28] and deep learning [29, 40, 20] have also been studied.

Despite this progress, the interpretability of learned decision rules has received comparatively less attention. In many applications, only a small subset of features is truly decisionrelevant. Identifying such features can improve interpretability, reduce data collection and monitoring costs, simplify downstream implementation, and improve robustness when irrelevant features are noisy or unstable. This motivates us to consider variable selection for feature-based newsvendor models.

Variable selection has been extensively studied in statistics and machine learning through sparsity-inducing formulations, including Lasso [33], bridge-type penalties [19], the smoothly clipped absolute deviation (SCAD) penalty [37, 16], and the minimax concave penalty (MCP) [39]. These methods can be viewed as tractable surrogates for the ideal but computationally challenging $ \ell _ { 0 } .$ -constrained formulation. Convex penalties are typically easier to optimize, whereas nonconvex penalties may achieve sharper statistical properties but are harder to solve globally. In this paper, we study the following question:

## How can we eficiently solve variable selection for feature-based newsvendor and what statistical guarantees can be established?

Our contributions are summarized as follows.

(I) Exact and approximation algorithms. For the variable selection problem, we add an $\ell _ { 2 } { \mathrm { - n o r m } }$ regularization and reformulate it as a mixed-integer second-order cone program (MISOCP). Compared with the standard Big-M approach for variable selection, our MISOCP formulation yields a tighter continuous relaxation and enables a more eficient algorithm design. Based on the reformulation, we further develop two eficient approximation algorithms and show that the randomized-rounding method admits a bi-criteria guarantee.

(II) Statistical guarantees. Under the assumption that demand has sparse and linear dependence on features, we establish the concentration error bound between our estimated policy and the ground truth optimal policy, as well as the out-of-sample performance and support recovery guarantees. These performance bounds break the curse of dimensionality because they crucially depend on the sparsity level instead of the data dimension.

(III) Numerical validation. We compare the exact optimization methods, randomizedrounding, greedy, and Lasso-based approaches using synthetic and retail data, illustrating their computational and statistical trade-ofs.

## 1.1. Related Works

Data-Driven Newsvendor. The classical newsvendor model assumes that the demand distribution is known, whereas modern data-driven approaches learn ordering policies directly from historical data. Early work along these lines includes operational-statistics-based approaches [27] and feature-based linear policies for the big-data newsvendor [3]. Subsequent work has developed richer data-driven models based on quantile regression and machine learning [20], reproducing kernel Hilbert space (RKHS)-based policy classes [6, 5], and deep neural networks [29, 40]. Serrano et al. [32] provided a variable selection framework for the feature-based newsvendor problem. They select key features by adding the sparsity budget constraint to the objective and tune penalty parameters using bilevel optimization. Our work contributes to this literature by studying hard-cardinality-constrained policy learning and by providing both computational and statistical guarantees. Chang et al. [9] also studied this framework, but their computational approach can be viewed as a heuristic for solving our $\ell _ { 0 } { \mathrm { - c o n s t r a i n e d } }$ formulation, and their statistical analysis focused on the data privacy viewpoint.

Quantile Regression. A fundamental observation in the newsvendor problem is that the optimal order quantity is a conditional quantile of the demand distribution. This connection links contextual newsvendor problems closely to quantile regression, beginning with the seminal work of Koenker and Bassett [23]. Quantile-regression-based methods have been used successfully in feature-based newsvendor models [20]. In high-dimensional settings, sparse quantile regression has been studied under both $\ell _ { 1 } { \mathrm { - r e g u l a r i z e d } }$ and $\ell _ { 0 } { \ - } \mathrm { { t y p e } }$ formulations [4, 13]. The study in this paper can be interpreted as an $\ell _ { 0 } { \mathrm { - c o n s t r a i n e d } }$ and $\ell _ { 2 } { \mathrm { - r e g u l a r i z e d } }$ empirical quantile regression problem.

Variable Selection. Variable selection is a central topic in statistics and machine learning. Classical approaches promote sparsity through penalized formulations [33], or through greedy and local-search methods [24, 26, 38, 25, 10] that approximately solve $\ell _ { 0 } { \mathrm { - c o n s t r a i n e d } }$ optimization problems. Although $\ell _ { 0 }$ -constrained formulations are generally NP-hard and computationally challenging in the worst case, they often perform well in high-dimensional, small-sample, and low-signal-to-noise settings [7, 1, 38]. This motivates the development of scalable algorithms for directly solving the $\ell _ { 0 } { \mathrm { - c o n s t r a i n e d } }$ variable-selection problem for the newsvendor model, as well as theoretical analyses that quantify the optimality gap of approximation algorithms. Compared with classical sparse ridge regression, our problem is further complicated by an empirical loss that is piecewise linear and asymmetric. To address these challenges, we combine perspective reformulations and support-selection algorithms with a statistical analysis tailored to the quantile-regression structure of the newsvendor loss.

Notations. For real numbers $a , b ,$ , we let $a \vee b = \operatorname* { m a x } \{ a , b \}$ . Given a positive integer $n ,$ define $[ n ] = \{ 1 , \ldots , n \}$ . For a distribution function $F ( \cdot )$ , we denote $F ^ { - 1 } ( p ) = \operatorname* { i n f } \{ x \in \mathbb { R } : \ F ( x ) \geq$ $p \} , p \in [ 0 , 1 ]$ . Given an event $E ,$ define $\mathbf { 1 } _ { E } ( x ) = 1 { \mathrm { ~ i f ~ } } x \in E$ and otherwise $\mathbf { 1 } _ { E } = 0$ . For a vector $s \in \mathbb { R } ^ { n }$ , we use $s [ i ]$ to denote its i-th component, and define its $\ell _ { 0 } { \cdot } \mathrm { n o r m }$ $\textstyle \| s \| _ { 0 } = \sum _ { i \in [ n ] } \mathbf { 1 } _ { s [ i ] \neq 0 }$ Given an $m \times n$ matrix X and a set $S \subseteq [ n ]$ , let $\mathbf { X } _ { S }$ denote the submatrix of X with columns from the set $S .$ . Given a set ${ \mathcal { F } } _ { z }$ , let $\operatorname { C o n v } ( \mathcal { F } )$ denote its convex hull, $\mathrm { i . e . }$ , the smallest convex set containing $\mathcal { F }$

## 2. Problem Setup

The newsvendor problem seeks to solve the following stochastic optimization problem:

$$
\operatorname* { m i n } _ { z \geq 0 } \mathbb { E } [ b \operatorname* { m a x } \{ y - z , 0 \} + h \operatorname* { m a x } \{ z - y , 0 \} ] ,\tag{1}
$$

where the expectation is taken with respect to the random demand $y , z$ denotes the order quantity, and $b ,$ h represent the unit back-ordering and holding costs, respectively. A central challenge in solving (1) is that the underlying demand distribution is typically unknown.

In the context of big data, auxiliary features can be used for accurate demand estimation. Let $\boldsymbol { x } \in \mathbb { R } ^ { D }$ denote a feature vector. For a given $x ,$ the optimal order quantity can be expressed as a function $f ( x )$ with

$$
f ( x ) = \underset { z \geq 0 } { \mathrm { a r g m i n } } \left. \mathbb { E } _ { y \vert x } \big [ b \mathrm { m a x } \{ y - z , 0 \} + h \mathrm { m a x } \{ z - y , 0 \} \big ] \right. .
$$

Rather than first estimating the conditional demand distribution and then solving the resulting newsvendor problem, one can directly learn an end-to-end policy function f that maps

features to decisions by solving the following contextual stochastic optimization (CSO) problem:

$$
\operatorname* { m i n } _ { f \in { \mathcal { F } } } { \mathbb { E } } _ { ( x , y ) } \big [ b \operatorname* { m a x } \{ y - f ( x ) , 0 \} + h \operatorname* { m a x } \{ f ( x ) - y , 0 \} \big ] .\tag{2}
$$

The choice of the function class $\mathcal { F }$ critically influences both computational tractability and out-of-sample performance. In this paper, we focus on sparse linear decision rules of the form

$$
\mathcal F = \Big \{ f ( \boldsymbol x ) = \boldsymbol \omega ^ { \top } \boldsymbol x + c \colon \boldsymbol \omega \in \mathbb { R } ^ { D } , c \in \mathbb { R } , \| \boldsymbol \omega \| _ { 0 } \le d \Big \} .
$$

The $\ell _ { 0 } .$ -constraint ensures that at most d covariates actively afect the decision, thereby facilitating interpretable and explainable policies. Under the linear decision-rule model, the subproblem associated with each fixed support is a convex optimization and therefore enables us to develop mixed-integer convex program reformulations for our variable selection framework. This framework can be extended to nonlinear decision rules through basis expansion.

Remark 1 (Linear Demand Model). Consider the linear demand model $y = ( \omega ^ { * } ) ^ { \top } x + \epsilon ,$ where $\epsilon$ is a random variable independent of x with cumulative distribution function $F _ { \epsilon }$ Let $\tau = b / ( b + h )$ . The optimal feature-dependent newsvendor decision is the conditional $\tau \mathrm { - }$ quantile of $y$ given x, namely $f ^ { * } ( x ) = ( \omega ^ { * } ) ^ { \top } x + F _ { \varepsilon } ^ { - 1 } ( \tau )$ . Under this linear demand model, the slope in the linear decision rule recovers the true coeficient $\omega ^ { * }$ , while the intercept estimates the τ-quantile of the noise.

Remark 2 (Nonnegativity of order quantities). The optimal newsvendor decision has the nonnegative constraint. For the linear decision rule approach, we clip the estimated optimal order quantity to zero if it is negative. When demand is always non-negative, this procedure does not increase the newsvendor loss. In the main analysis, we focus on the unconstrained formulation in (2) for simplicity.

The objective in (2) involves the population expectation under the joint distribution of $( x , y )$ . In practice, however, only independent and identically distributed (i.i.d.) samples $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ are available. We therefore replace the population expectation with the empirical average and include an $\ell _ { 2 } \cdot$ -regularization term to improve both computational tractability and out-of-sample performance, which will be discussed in later sections. In summary, it leads to the optimization problem

$$
\operatorname* { m i n } _ { \omega , c \neq \| \omega \| _ { 0 } \leq d } \ F ( \omega , c ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big [ b \cdot \operatorname* { m a x } \big \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \big \} + h \cdot \operatorname* { m a x } \big \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \big \} \Big ] + \frac { 1 } { 2 \gamma } \| \omega \| _ { 2 } ^ { 2 } ,\tag{3}
$$

Problem (3) is nonconvex because of the $\ell _ { 0 } { \mathrm { - c o n s t r a i n t . } }$ , and is therefore generally dificult to solve. In fact, we show that it is NP-hard by a reduction from the exact cover by 3-sets problem. Its proof is provided in Appendix EC.1.

Proposition 1 (Computational Hardness). For fixed $\gamma , b , h > 0 ;$ , Problem (3) is NP-hard whenever the sparsity level d is part of the input.

Problem (3) is also related to sparse quantile linear regression. Indeed, it is equivalent to

$$
\operatorname* { m i n } _ { \omega , c \colon \| \omega \| _ { 0 } \le d } \left\{ \frac { ( b + h ) } { n } \sum _ { i = 1 } ^ { n } \rho _ { \tau } \left( y _ { i } - ( \omega ^ { \top } x _ { i } + c ) \right) + \frac { 1 } { 2 \gamma } \| \omega \| _ { 2 } ^ { 2 } \right\} ,
$$

where $\textstyle { \tau = { \frac { b } { b + h } } }$ and $\rho _ { \tau } ( u ) = u \big ( \tau - { \mathbf { 1 } } _ { \{ u < 0 \} } \big )$ . In other words, the goal of variable selection for the feature-based newsvendor is equivalent to estimating sparse coeficients that minimize the $\ell _ { 2 } { \mathrm { - r e g u l a r i z e d } }$ empirical quantile regression loss.

The remainder of the paper is organized as follows. In Section 3, we present a mixedinteger second-order cone reformulation of Problem (3), which can be solved exactly for medium-sized instances. In Section 4, we develop two approximation algorithms with performance guarantees. In Section $^ { 5 , }$ we establish statistical guarantees for the proposed approach, including consistency, sample complexity, and support recovery.

## 3. Exact Computational Algorithms

In this section, we develop exact algorithms for solving Problem (3) to global optimality. A standard approach is the $B i g \ – M$ reformulation proposed by Dai [13], which converts the sparsity-constrained problem into a mixed-integer convex program.

Big-M Method. Recall that the constraint $\| \omega \| _ { 0 } \leq d$ requires that at most d entries of $\omega$ be nonzero. To model the support of $\omega ,$ we introduce a binary vector $s \in \{ 0 , 1 \} ^ { D }$ , where $s [ i ] = 1$ indicates that $\omega [ i ]$ is allowed to be nonzero. Equivalently, we impose $\mathbf { 1 } \{ \omega [ i ] \neq 0 \} \subseteq s [ i ] , i \in$ $[ D ]$ . Assume there exists a vector $M \in \mathbb { R } _ { + } ^ { D }$ such that $| \omega [ i ] | \leq M [ i ] , i \in [ D ]$ , for some optimal solution $\omega .$ Then the logical constraint can be equivalently enforced by the linear inequality $| \omega [ i ] | \leq M [ i ] s [ i ] , i \in [ D ]$ . This observation leads to the following reformulation of Problem (3).

Proposition 2 (Big-M reformulation [13]). Assume there exists a vector $M \in \mathbb { R } _ { + } ^ { D }$ such that an optimal solution $\omega ^ { * }$ to Problem (3) satisfies $| \omega ^ { * } [ i ] | \leq M [ i ] , \forall i \in [ D ]$ . Then Problem (3) is equivalent to

$$
\begin{array} { c l } { \displaystyle \underset { \boldsymbol { \omega } , \boldsymbol { c } , \boldsymbol { s } } { \mathrm { m i n } } } & { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big [ b \operatorname* { m a x } \{ 0 , y _ { i } - \boldsymbol { \omega } ^ { \top } x _ { i } - \boldsymbol { c } \} + h \operatorname* { m a x } \{ 0 , \boldsymbol { \omega } ^ { \top } x _ { i } + \boldsymbol { c } - y _ { i } \} \Big ] + \frac { 1 } { 2 \gamma } \| \boldsymbol { \omega } \| _ { 2 } ^ { 2 } } \\ { \boldsymbol { s } . t . } & { \displaystyle \sum _ { i \in [ D ] } s [ i ] \leq d , \quad s \in \{ 0 , 1 \} ^ { D } , } \\ & { \displaystyle | \boldsymbol { \omega } [ i ] | \leq M [ i ] s [ i ] , \quad \forall i \in [ D ] . } \end{array}\tag{Big-M}
$$

Big-M method is standard but often computationally expensive for large-scale mixed-integer problems. In practice, such formulations are typically solved by branch-and-bound, which repeatedly solves convex relaxations. A major dificulty is that the Big-M formulation usually yields a weak relaxation: the optimality gap between (Big-M) and its continuous relaxation, obtained by replacing $s \in \{ 0 , 1 \} ^ { D }$ with $s \in [ 0 , 1 ] ^ { D }$ , can be substantial.

Some heuristics can help derive tight bounds of M and improve numerical performance. For example, let $( \omega _ { 0 } , c _ { 0 } )$ be a near-optimal solution to Problem (3), which can be obtained using some tractable algorithms outlined in Section 4. Any optimal solution to Problem (3) satisfies that

$$
\frac { \| \omega ^ { * } \| _ { 2 } ^ { 2 } } { 2 \gamma } \leq F ( \omega ^ { * } , c ^ { * } ) \leq F ( \omega _ { 0 } , c _ { 0 } ) .
$$

Based on this relation, the vector M with $M [ i ] = \sqrt { 2 \gamma F ( \omega _ { 0 } , c _ { 0 } ) } , i \in [ D ]$ is a valid choice because $| \omega ^ { * } [ i ] | \leq \| \omega ^ { * } \| _ { 2 } \leq M [ i ]$ . However, solving the Big-M formulation remains challenging for large-scale datasets.

Motivated by this limitation, we next introduce an alternative reformulation that is more computationally tractable. We show that it has a tighter convex relaxation than the Big-M approach and is therefore better suited for exact computation.

Perspective Reformulation. We introduce a stronger reformulation than the Big-M approach, based on the perspective function. To this end, we first rewrite the quadratic regularization term by introducing auxiliary variables $\mu \in \mathbb { R } _ { + } ^ { D }$

$$
\frac { 1 } { 2 \gamma } \| \omega \| _ { 2 } ^ { 2 } = \operatorname* { m i n } _ { \mu \in \mathbb { R } _ { + } ^ { D } } \left\{ \frac { \| \mu \| _ { 1 } } { 2 \gamma } : \omega [ i ] ^ { 2 } \leq \mu [ i ] , \forall i \in [ D ] \right\} .
$$

Therefore, Problem (3) is equivalent to the following extended Big-M formulation:

$$
\operatorname* { m i n } _ { \omega , c , s , \mu } \quad \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big [ b \cdot \operatorname* { m a x } \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \} + h \cdot \operatorname* { m a x } \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \} \Big ] + \frac { \| \mu \| _ { 1 } } { 2 \gamma } ,\tag{4a}
$$

$$
s . t . \quad \sum _ { j \in [ D ] } s [ j ] \leq d , \quad s \in \{ 0 , 1 \} ^ { D } ,\tag{4b}
$$

$$
( \omega [ j ] , s [ j ] , \mu [ j ] ) \in \mathcal S _ { j } , \quad \forall j \in [ D ] ,\tag{4c}
$$

where for each coordinate $j \in [ D ]$ , define the set

$$
\begin{array} { r } { S _ { j } : = \Big \{ ( \omega [ j ] , s [ j ] , \mu [ j ] ) \in { \mathbb { R } } \times \{ 0 , 1 \} \times { \mathbb { R } } _ { + } : \omega [ j ] ^ { 2 } \leq \mu [ j ] , \ | \omega [ j ] | \leq M [ j ] s [ j ] \Big \} . } \end{array}
$$

Günlük and Linderoth [18] have characterized the convex hull of $S _ { j }$ as

$$
\operatorname { C o n v } ( S _ { j } ) = { \Big \{ } ( \omega [ j ] , s [ j ] , \mu [ j ] ) \in \mathbb { R } \times [ 0 , 1 ] \times \mathbb { R } _ { + } : \omega [ j ] ^ { 2 } \leq \mu [ j ] s [ j ] , \ | \omega [ j ] | \leq M [ j ] s [ j ] { \Big \} } .
$$

By convexifying the constraint (4c), we can obtain the perspective reformulation of (3).

Proposition 3 (Perspective Reformulation). Under the same condition as in Proposition $^ { 2 , }$ Problem (3) is equivalent to

$$
\begin{array} { c l } { \displaystyle \underset { \omega , c , s , \mu } { \operatorname* { m i n } } } & { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big [ b \cdot \operatorname* { m a x } \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \} + h \cdot \operatorname* { m a x } \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \} \Big ] + \frac { 1 } { 2 \gamma } \sum _ { j \in [ D ] } \mu [ j ] , } \\ { \displaystyle s . t . } & { \displaystyle \sum _ { j \in [ D ] } s [ j ] \leq d , \quad s \in \{ 0 , 1 \} ^ { D } , } \\ & { \displaystyle ( \omega [ j ] , s [ j ] , \mu [ j ] ) \in \mathrm { C o n v } ( S _ { j } ) , \quad \forall j \in [ D ] . } \end{array}
$$

(Perspective)

Since the constraint $\omega [ j ] ^ { 2 } \leq \mu [ j ] s [ j ]$ is representable by a rotated second-order cone when $s [ j ] \geq 0$ and $\mu [ j ] \ge 0$ , Problem (Perspective) is a mixed-integer second-order cone program (MISOCP). The following theorem compares the Big-M and perspective formulations. Its proof follows techniques in [38] and is provided in Appendix EC.2.

Theorem 1 (Strength of Perspective Reformulation). (I) The mixed-integer formula tions (4) and (Perspective) are equivalent, and hence have the same optimal value;

(II) Let $V _ { \mathrm { r e l } } ^ { B i g - M }$ and $V _ { \mathrm { r e l } } ^ { P e r s }$ denote the optimal values of the continuous relaxations of $( \mathrm { B i g } \mathrm { - } M )$ and (Perspective), respectively. It always holds that $V _ { \mathrm { r e l } } ^ { B i g - M } \leq V _ { \mathrm { r e l } } ^ { P e r s }$

Remark 3 (Perspective Reformulation is Strictly Stronger). We provide an example to show that the perspective reformulation is strictly stronger. Take $D = 2 , d =$ $1 , b = h = 1 , n = 4 , M = 2 , \gamma = 1$ . Assume that one has access to four data samples: $\{ ( e _ { 1 } , 1 ) , ( - e _ { 1 } , - 1 ) , ( e _ { 2 } , 1 ) , ( - e _ { 2 } , - 1 ) \}$ , where $e _ { 1 } , e _ { 2 }$ denote the unit vectors in $\mathbb { R } ^ { 2 }$ . In this case, it is easy to verify $V _ { \mathrm { r e l } } ^ { \mathrm { B i g - M } } = 0 . 7 5$ and $V _ { \mathrm { r e l } } ^ { \mathrm { P e r s } } = 0 . 8 7 5$ . Hence, this problem instance ensures that $V _ { \mathrm { r e l } } ^ { \mathrm { B i g - M } } < V _ { \mathrm { r e l } } ^ { \mathrm { P e r s } }$

In summary, perspective reformulation is preferred to the Big-M method in two ways. First, Theorem 1 together with Remark 3 explains that the perspective reformulation is more computationally eficient. Since its continuous relaxation is tighter than that of the Big-M formulation, a branch-and-bound solver typically explores fewer nodes and converges faster in practice. Second, it is dificult to derive valid and tight Big-M constants for solving both formulations, and using overly conservative Big-M constants makes the formulations numerically ill-conditioned to solve. Even if we drop the constraints $| \omega [ j ] | \le M [ j ] s [ j ] , j \in [ D ]$ in Problem (Perspective), it is still a valid reformulation of Problem (3) and computationally tractable, leading to the following Big-M-free reformulation:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \omega , c , s , \mu } } & { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big [ b \operatorname* { m a x } \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \} + h \operatorname* { m a x } \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \} \Big ] + \frac { 1 } { 2 \gamma } \sum _ { j \in [ D ] } \mu [ j ] } \\ { \mathrm { s . t . ~ } } & { \displaystyle \sum _ { j \in [ D ] } s [ j ] \leq d , \quad s \in \{ 0 , 1 \} ^ { D } , } \\ & { \displaystyle \omega [ i ] ^ { 2 } \leq \mu [ i ] s [ i ] , i \in [ D ] , \quad \mu \in \mathbb { R } _ { + } ^ { D } . } \end{array}
$$

(Big-M-free-Perspective)

## 4. Scalable Approximation Algorithms

In this section, we study two scalable approximation algorithms to obtain near-optimal solutions to Problem (3) and build their performance guarantees.

## 4.1. Continuous Relaxation with Randomized Rounding

Our first approach is motivated by the MISOCP (Big-M-free-Perspective). Its continuous relaxation replacing the binary constraint $s \in \{ 0 , 1 \} ^ { D }$ with $s \in [ 0 , 1 ] ^ { D }$ becomes a convex program, and can be eficiently solved using of-the-shelf solvers. Then, we estimate the support of sparse regression coeficients using a randomized rounding scheme. See Algorithm 1 for detailed implementation.

Algorithm 1 Continuous Relaxation with Randomized Rounding for Problem (3)   
Input: Data $\overline { { \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } } }$ , sparsity budget $d ,$ regularization parameter $\gamma$   
1: Solve the continuous relaxation of (Big-M-free-Perspective) and obtain an optimal solu   
tion $( \widehat { \omega } , \widehat { c } , \widehat { s } , \widehat { \mu } )$   
2: Initialize set $S = \emptyset$   
3: for $i = 1 , \ldots , D$ do   
4: Sample an uniform number U on [0, 1].   
5: Update $S = S \cup \{ i \} \mathrm { i f } \ U \leq \widehat { s } [ i ] .$   
6: end for   
7: Obtain (˜ω, c˜) as the optimal solution to arg min $\left\{ F ( \omega , c ) : \omega [ i ] = 0 { \mathrm { i f } } i \notin S \right\}$   
ω,c   
8: return $( \tilde { \omega } , \tilde { c } )$ as the regression coeficient estimator

We provide performance guarantees for this computationally eficient algorithm below. Let $\mathbf { X } = ( x _ { 1 } , \ldots , x _ { n } ) ^ { \top } \in \mathbb { R } ^ { n \times D }$ be the data matrix. For $k \in [ D ]$ , define

$$
\theta _ { k } : = \operatorname* { m a x } _ { | S | = k } \ \sigma _ { \operatorname* { m a x } } ( \mathbf { X } _ { S } \mathbf { X } _ { S } ^ { \top } ) ,
$$

where $\sigma _ { \mathrm { m a x } } ( \cdot )$ denotes the maximum singular value. Due to the randomized rounding procedure, the support size of the estimated regression coeficient may not satisfy the sparsity budget. The following theorem provides the bi-criteria approximation guarantee of Algorithm 1. It proves that with high probability, the selected support is only moderately larger than $d ,$ while the objective value is within an additive tolerance of the optimal value.

Theorem 2 (Approximation Guarantee of Algorithm 1). Given $\delta \in ( 0 , 1 )$ and $\epsilon > 0 .$ . Let (˜ω, c˜) be the output from Algorithm 1. Suppose the regularization coeficient

$$
\frac { 1 } { 2 \gamma } \geq \frac { \log ( 2 n / \delta ) ( h \vee b ) ^ { 2 } \theta _ { 1 } } { 6 n \epsilon } + \frac { ( h \vee b ) ^ { 2 } \sqrt { 2 \theta _ { 1 } \theta _ { d } \log ( 2 n / \delta ) } } { 4 n \epsilon } .
$$

Then, with probability at least $1 - \delta ,$ it holds that

$$
\| \tilde { \omega } \| _ { 0 } \leq \left( 1 + \sqrt { \frac { 2 \log ( 2 / \delta ) } { d } } + \frac { \log ( 2 / \delta ) } { d } \right) d , \quad F ( \tilde { \omega } , \tilde { c } ) \leq o p t \nu a l ( 3 ) + \epsilon .
$$

The proof of Theorem 2 is provided in Appendix EC.3. The proof step to bound the sparsity level of $\tilde { \omega }$ follows from the usage of concentration inequality discussed in [38, Theorem 5]. The reference therein implicitly assumed that log $( 2 / \delta ) \leq d / 3$ . Our concentration result refines the concentration result and does not impose this assumption. The support-size bound in Theorem 2 depends on the target sparsity $d ,$ but not on the data dimension $D .$ . The objective performance guarantee requires that the regularization parameter $\frac { 1 } { 2 \gamma }$ is not too small: its lower bound is related to the dimension $D$ through the maximum singular values $\theta _ { 1 }$ and $\theta _ { d }$ from the design matrix. This assumption is not quite restrictive since adding $\ell _ { 2 } { \mathrm { - n o r m } }$ regularization ensures the estimated solution is more robust than unregularized models [8].

## 4.2. Greedy Algorithm

Greedy algorithm is a common heuristic for many variable selection problems. To develop a greedy algorithm for solving (3), we rewrite it as a combinatorial formulation. Indeed, by introducing a subset $S \subseteq [ D ]$ , Problem (3) is equivalent to

$$
\operatorname* { m i n } _ { S \subseteq [ D ] , | S | \leq d } \left\{ \operatorname* { m i n } _ { \omega \in \mathbb { R } ^ { D } , c \in \mathbb { R } } F ( \omega , c ) : \omega [ i ] = 0 { \mathrm { ~ i f ~ } } i \notin S \right\} .\tag{5}
$$

For a fixed support $S ,$ the inner subproblem of (5) is a convex program. Using strong duality theory on the inner minimization problem, this subproblem has an equivalent dual representation that depends on S only through submatrix $\mathbf { X } _ { S }$

Theorem 3 (Minimax Reformulation of (5)). Problem (5) has the same optimal value as the following minimax optimization problem:

$$
\operatorname* { m i n } _ { \substack { S \subseteq [ D ] , | S | \leq d } } \left\{ G ( S ) : = \operatorname* { m a x } _ { \substack { z [ \bar { u } ] \in [ - b , h ] , i \in [ n ] , \sum _ { i = 1 } ^ { n } z [ \bar { u } ] = 0 } } - z ^ { \top } \left( \frac { \gamma } { 2 n ^ { 2 } } X _ { S } X _ { S } ^ { \top } \right) z - \frac { 1 } { n } z ^ { \top } y \right\} .\tag{6}
$$

Based on Theorem 3, Greedy algorithm proceeds by iteratively finding a candidate index such that the objective G is minimized for current subset including the new index. See the detailed procedure in Algorithm 2.

Algorithm 2 Greedy Algorithm for Problem (6)   
Input: Data $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , sparsity budget $d ,$ regularization parameter $\gamma$   
1: Initialize $S _ { 0 } = \emptyset$   
2: for $k = 0 , 1 , \ldots , d - 1$ do   
3: Choose $j _ { k } \in \arg \operatorname* { m i n } _ { j \in [ D ] \backslash S _ { k } } G ( S _ { k } \cup \{ j \} )$   
4: Update $S _ { k + 1 } = S _ { k } \cup \{ j _ { k } \}$   
5: end for   
6: Obtain $( \tilde { \omega } ^ { \mathrm { g r } } , \tilde { c } ^ { \mathrm { g r } } )$ as the optimal solution to arg min $\left\{ F ( \omega , c ) : \omega [ i ] = 0 \mathrm { i f } i \notin S _ { d } \right\}$   
ω,c   
7: return $( \tilde { \omega } ^ { \mathrm { g r } } , \tilde { c } ^ { \mathrm { g r } } )$

By Theorem 3, it can be shown that the set function $G ( S )$ is non-increasing in $S ,$ and the objective of iterates in Algorithm 2 is monotonically decreasing. It is an open question on the approximation guarantees of the final iterate from the greedy algorithm, as the set function G does not satisfy the submodular assumption commonly used in literature. Inspired by [38], the greedy approach can also be integrated with Algorithm 1. Specifically, given threshold $\delta > 0$ and ωb obtained from continuous relaxation, define $\begin{array} { r } { \mathcal { C } = \{ i : ~ | \widehat { \omega } [ i ] | > \delta \} } \end{array}$ as the efective support obtained from continuous relaxation. We can apply the greedy algorithm based on C instead of [D] and can reduce the number of candidate variables. This algorithm works well especially when $\widehat { \omega }$ is sparse.

## 5. Statistical Performance Guarantees

In this section, we investigate statistical performance guarantees for our proposed variable selection framework. Assume the observed data samples $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ are i.i.d. copies of (X, Y ), where

$$
Y = ( \omega ^ { * } ) ^ { \top } X + c ^ { * } + \varepsilon .\tag{7}
$$

Here, $\boldsymbol { \omega } ^ { * } \in \mathbb { R } ^ { D }$ is a fixed d-sparse vector with $S ^ { * } = \operatorname { s u p p } ( \omega ^ { * } )$ , and $| S ^ { * } | = d ,$ and $c ^ { * } \in \mathbb { R }$ is a fixed intercept parameter. The distribution of the noise $\varepsilon$ is independent of X. We analyze the estimator from the optimization problem

$$
\left( \widehat { \omega } _ { n } , \widehat { c } _ { n } \right) = \operatorname * { a r g m i n } _ { \omega , c \colon \left\| \omega \right\| _ { 0 } \leq d } \left\{ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ b \operatorname* { m a x } \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \} + h \operatorname* { m a x } \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \} \right] + \frac { \left\| \omega \right\| _ { 2 } ^ { 2 } } { 2 \gamma _ { n } } \right\} ,
$$

where the regularization parameter $\gamma _ { n }$ is allowed to vary with the sample size $n .$ . We impose the following technical assumptions throughout our analysis.

Assumption 1 (Structure of Data). (I) The distribution of ε satisfies $F _ { \epsilon } ( 0 ) : = \mathbb { P } ( \epsilon \leq 0 ) =$ τ , where $\tau = b / ( b + h )$

(II) The random variable ε satisfies $\mathbb { E } | \varepsilon | < \infty$

(III) The density of the noise exists and is denoted as $f _ { \varepsilon } ( \cdot )$ . There exists $r _ { 0 }$ such that for all $| u | \leq r _ { 0 } ,$ , it holds that $f _ { \varepsilon } ( u ) \geq f > 0$

(IV) The augmented covariate $\widetilde { X } = ( X ^ { \top } , 1 ) ^ { \top }$ is bounded almost surely: $\| \widetilde { X } \| _ { 2 } \le K$

(V) There exists $\kappa > 0$ such that for each $u \in \mathbb { R } ^ { D } , t \in \mathbb { R } , \| u \| _ { 0 } \leq 2 d ,$ it holds that

$$
\mathbb { E } \big [ ( u ^ { \top } X + t ) ^ { 2 } \big ] \geq \kappa \big ( \| u \| _ { 2 } ^ { 2 } + t ^ { 2 } \big ) .
$$

It is noteworthy that Assumption 1(I) is not restrictive. Indeed, (7) can be reformulated as $\begin{array} { r } { Y = ( \omega ^ { * } ) ^ { \top } X + \tilde { c } ^ { * } + \tilde { \varepsilon } ^ { * } } \end{array}$ for constant $\tilde { c } ^ { * } = c ^ { * } + F _ { \varepsilon } ^ { - 1 } ( \tau )$ and random noise $\tilde { \varepsilon } ^ { * } = \varepsilon - F _ { \varepsilon } ^ { - 1 } ( \tau )$ . Such a scalar transformation always ensures Assumption 1(I) holds. Assumption 1(II) is a suficient condition that ensures the optimal newsvendor risk

$$
\mathbb { E } _ { ( X , Y ) } \big [ b \operatorname* { m a x } ( Y - ( \omega ^ { * } ) ^ { \top } X - c ^ { * } , 0 ) + h \operatorname* { m a x } ( ( \omega ^ { * } ) ^ { \top } X + c ^ { * } - Y , 0 ) \big ] = ( b + h ) \mathbb { E } _ { \varepsilon } [ \rho _ { \tau } ( \varepsilon ) ]
$$

is finite. Assumption 1(III) imposes local regularity of the noise density around the target quantile. Assumption 1(IV) imposes the boundedness of the feature vector. Assumption 1(V) is a restricted eigenvalue condition, which ensures identifiability over sparse directions. These assumptions are commonly used in sparse statistical analysis [35].

We first establish a finite-sample estimation-error bound regarding the estimator $( \widehat { \omega } _ { n } , \widehat { c } _ { n } )$ in Theorem 4. It reveals that under proper scaling of the ridge parameter, the estimator converges to the ground truth with a rate crucially dependent on the sparsity level instead of the data dimension. The estimation error vanishes quickly as the sample size n increases.

Theorem 4 (Estimation Error Bound). Let $\delta \in ( 0 , 1 )$ and suppose Assumption 1 holds. Set the ridge parameter $\textstyle { \frac { 1 } { 2 \gamma _ { n } } } \lesssim { \sqrt { \frac { d \log ( e D / d ) } { n } } }$ , then with probability at least $1 - \delta ,$ it holds that

$$
\left\| \left( \widehat { \omega } _ { n } - \omega ^ { * } \right) \right\| _ { 2 } = \mathcal { O } \left( \sqrt { \frac { d \log ( e D / d ) + \log ( 1 / \delta ) } { n } } \right) ,
$$

where $\mathcal { O } ( \cdot )$ hides the constant depending only on $b , h , r _ { 0 } , f , \kappa , K , \| \omega ^ { * } \| _ { 2 } .$

The proof of Theorem 4 is provided in Appendix EC.4. Building on Theorem 4, we further show the out-of-sample performance and support recovery guarantees of our obtained estimator. Given regression coeficients $( \omega , c )$ , define its population newsvendor risk

$$
\mathcal { R } ( \omega , c ) = \mathbb { E } _ { ( X , Y ) } \big [ b \operatorname* { m a x } ( Y - \omega ^ { \top } X - c , 0 ) + h \operatorname* { m a x } ( \omega ^ { \top } X + c - Y , 0 ) \big ] .\tag{8}
$$

Our out-of-sample performance guarantee in Corollary 1 quantifies the gap between the population newsvendor risk of $( \hat { \omega } _ { n } , \hat { c } _ { n } )$ and the optimal risk, which vanishes in the order of $d \log ( e D / d ) / n$

Corollary 1 (Out-of-Sample Performance Guarantee). Under the same setup as in Theorem 4, with probability at least $1 - \delta ,$ it holds that

$$
0 \leq \mathcal { R } ( \hat { \omega } _ { n } , \hat { c } _ { n } ) - \operatorname* { m i n } _ { \| \omega \| _ { 0 } \leq d , c } \mathcal { R } ( \omega , c ) \leq \mathcal { O } \left( \frac { d \log ( e D / d ) + \log ( 1 / \delta ) } { n } \right) ,
$$

where $\mathcal { O } ( \cdot )$ hides the constant depending only on $b , h , r _ { 0 } , \underline { { f } } , \kappa , K , \| \omega ^ { * } \| _ { 2 }$

The support recovery guarantee is provided in Corollary 2, whose proof is provided in Appendix EC.6 and uses the fact that $\widehat { \omega } _ { n }$ converges to the true sparse vector $\omega ^ { * }$ , and so is its support.

Corollary 2 (Support Recovery Guarantee). Under the same setup as in Theorem $^ { 4 , }$ and assume the sample size n is suficiently large so that with probability at least $1 - \delta ,$ , it holds that

$$
\left\| \left( \widehat { \omega } _ { n } - \omega ^ { * } \right) \right\| _ { 2 } < \operatorname* { m i n } _ { j \in S ^ { * } } | \omega ^ { * } [ j ] | .
$$

Then, with probability at least $1 - \delta ,$

$$
\widehat { \omega } _ { n } [ i ] \neq 0 , \quad \widehat { \omega } _ { n } [ j ] = 0 , \quad \forall i \in S ^ { * } , \ j \notin S ^ { * } .
$$

## 6. Numerical Study

In this section, we evaluate the performance of the proposed methods. We generate synthetic data from the linear model $Y = ( \omega ^ { * } ) ^ { \top } X + \varepsilon .$ , where $\varepsilon \sim \mathcal { N } ( 5 , \sigma ^ { 2 } )$ . The feature vector X follows a multivariate Gaussian distribution with mean zero and covariance matrix $\Sigma ,$ , where $\Sigma [ i , j ] = 0 . 5 ^ { | i - j | } , i , j \in [ D ]$ . Although our statistical analysis assumed that the feature vector has bounded support, we use Gaussian-distributed feature vector with unbounded support for numerical evaluation. We further perform scalar transformation to ensure Assumption 1(I) holds. The first d entries of $\omega ^ { * }$ are nonzero and are specified as $\begin{array} { r } { \omega ^ { * } [ i ] = ( - 1 ) ^ { i + 1 } \left( 1 - \frac { i - 1 } { d } \right) } \end{array}$ ， $i \in$ [d]. We set the unit shortage and holding costs to $b = 2$ and $h = 1$ , respectively.

For baseline comparison, our experiment includes the following methods: (I) the exact algorithms based on the Big-M and perspective formulations in Section $3 ;$ (II) the continuousrelaxation algorithm described in Algorithm 1; (III) the greedy algorithm described in Algorithm $2 ;$ and (IV) an $\ell _ { 1 }$ -based baseline obtained by replacing the $\ell _ { 0 } .$ -constraint in Problem (3) with an ℓ -regularized formulation. For each algorithm, we impose a time limit of 1800 seconds. The exact algorithm baseline is built on the perspective formulation except in Section 6.1. The mixed-integer optimization algorithms terminate once the relative optimality gap falls below 1e-4. Unless otherwise specified, the ridge and Lasso tuning parameters are selected by cross-validation. We measure the performance support recovery using false discovery proportion (FDP) and non-discovery proportion (NDP), defined in [2]:

$$
\mathrm { F D P } ( S ) = \frac { | S \setminus S ^ { * } | } { | S | } , \qquad \mathrm { N D P } ( S ) = \frac { | S ^ { * } \setminus S | } { | S ^ { * } | } ,\tag{9}
$$

where $S ^ { * }$ denotes the true support and S denotes the support estimated by a given algorithm. Smaller values of FDP and NDP indicate better support recovery performance. All computations are performed in 2.60GHz Intel(R) Xeon(R) Platinum 8358 CPU and 32GB main memory with Gurobi 13.0.1 solver.

## 6.1. Comparison of exact algorithms

We first compare the two exact algorithms in Section 3 under diferent configurations of $( D , n , d )$ . Throughout this subsection, we fix $\gamma = 5 0 .$ , M = 100, and $\sigma ^ { 2 } = 1$ . Performance is evaluated in terms of the objective value, computational time, and FDP. Table 1 reports the computational results for solving (Big-M) and (Perspective) via branch-and-bound. Overall, the perspective formulation outperforms the Big-M formulation on most instances, and the advantage becomes more pronounced as the ambient dimension D increases. This observation is consistent with Theorem 1, which shows that (Perspective) provides a stronger continuous relaxation. When $D \in \{ 1 0 0 0 , 2 0 0 0 \}$ , both approaches often reach the prescribed time limit, indicating that solving the sparse newsvendor problem to global optimality can be computationally demanding in large-scale settings. We also observe that the FDP generally decreases as D increases in these experiments. A possible explanation is that the regularization parameter $\gamma$ is not optimally tuned such that the FDP values for diferent instances are not comparable.

Table 1 Computational results of exact algorithms. Here we fix cardinality d = 50. NA indicates the algorithm fails to find any feasible solution within time limit.
<table><tr><td rowspan="2">D</td><td rowspan="2">n</td><td colspan="4">Big-M</td><td colspan="4">Perspective</td></tr><tr><td>Objective</td><td>Gap (%)</td><td>Time (s)</td><td>FDP (%)</td><td>Objective</td><td>Gap (%)</td><td>Time (s)</td><td>FDP (%)</td></tr><tr><td rowspan="3">100</td><td>500</td><td>10.71</td><td>3.77</td><td>1800.00</td><td>52.00</td><td>10.65</td><td>9.69e-3</td><td>11.33</td><td>42.00</td></tr><tr><td>1000</td><td>10.52</td><td>1.02</td><td>1800.00</td><td>48.00</td><td>10.45</td><td>9.94e-3</td><td>50.79</td><td>40.00</td></tr><tr><td>5000</td><td>11.06</td><td>0.38</td><td>1800.00</td><td>26.00</td><td>10.91</td><td>0.14</td><td>1800.00</td><td>24.00</td></tr><tr><td rowspan="3">500</td><td>500</td><td>12.38</td><td>100</td><td>1800.00</td><td>12.00</td><td>10.01</td><td>0.56</td><td>1800.00</td><td>9.70</td></tr><tr><td>1000</td><td>10.48</td><td>36.71</td><td>1800.00</td><td>11.00</td><td>10.11</td><td>1.37</td><td>1800.00</td><td>9.70</td></tr><tr><td>5000</td><td>10.77</td><td>6.85</td><td>1800.00</td><td>7.50</td><td>10.71</td><td>1.39</td><td>1800.00</td><td>5.20</td></tr><tr><td rowspan="3">1000</td><td>500</td><td>9.55</td><td>100.00</td><td>1800.00</td><td>5.15</td><td>9.17</td><td>1.44</td><td>1800.00</td><td>4.85</td></tr><tr><td>1000</td><td>11.54</td><td>100.00</td><td>1800.00</td><td>7.63</td><td>10.34</td><td>1.44</td><td>1800.00</td><td>4.15</td></tr><tr><td>5000</td><td>10.81</td><td>15.52</td><td>1800.00</td><td>4.26</td><td>10.74</td><td>3.17</td><td>1800.00</td><td>3.68</td></tr><tr><td rowspan="3">2000</td><td>500</td><td>12.01</td><td>100</td><td>1800.00</td><td>6.41</td><td>9.90</td><td>1.99</td><td>1800.00</td><td>2.46</td></tr><tr><td>1000</td><td>12.98</td><td>100</td><td>1800.00</td><td>5.51</td><td>9.56</td><td>4.34</td><td>1800.00</td><td>2.41</td></tr><tr><td>5000</td><td>NA</td><td>NA</td><td>1800.00</td><td>NA</td><td>11.37</td><td>41.04</td><td>1800.00</td><td>2.46</td></tr></table>

## 6.2. Results on Support Recovery

Next, we compare the variable-selection performance of all methods. The noise variance $\sigma ^ { 2 }$ is chosen so that $\begin{array} { r } { \frac { \mathrm { V a r } ( ( \omega ^ { * } ) ^ { \top } X ) } { \mathrm { V a r } ( \varepsilon ) } = 1 0 } \end{array}$ , that is, the signal-to-noise ratio (SNR) is 10.

![](images/99d834c04c8de9e12dccf908f4ed6130a57b7f14dcfb8e5f4d97e26ae5ef2ec0.jpg)

![](images/e37623d46c127a444b2ae0727ab32b65129bf222935ce41ebfd100d544953bfe.jpg)

![](images/8d8cb894b9ab0baddfb0d43b8c9f93c84f3a5737a0d162036636f6865ec2523e.jpg)  
Figure 1 FDP with diferent choices of sample size n and each subplot represents choices of dimension D.

![](images/1832b8006d7e6121934f295f16b6e1336b605745cf94eda67d61c034fbc54529.jpg)  
Figure 2 FDP with diferent choices of dimension D and each subplot represents choices of sample size n.

We first consider the case in which the sparsity budget matches the true support size and is fixed at 10. In this setting, FDP and NDP coincide, so we report only FDP. Figures 1 and 2 show the FDP values under diferent combinations of $( n , D )$ . The exact algorithms achieve the best support recovery performance, especially in the high-dimensional and small-sample regime. Among the scalable baselines, the greedy algorithm generally outperforms both the continuous-relaxation and Lasso baselines.

![](images/da893d196193a27fcf839fdaf040e26ba5aacf7a46a70fb12c1ab930fe681e20.jpg)  
Figure 3 Comparison in terms of FDP and NDP for n ∈ {500, 2000} and $D = 1 0 0 .$

We next fix the true support size at 10 and vary the sparsity budget to compare the methods in terms of both FDP and NDP. Figure 3 shows the results for $( D , n ) \in$ {(100, 500), (100, 2000)}. The black dashed line represents the ground-truth optimum of these metrics. As shown in the figure, exact algorithm remains close to the oracle benchmark across a wide range of sparsity budgets.

## 6.3. Results on Parameter Estimation

We examine the performance of parameter estimation for various methods on problem instances where $D = 1 0 0 , n \in \{ k \cdot 1 0 0 , k \in [ 1 0 ] \}$ , and $\mathrm { S N R } \in \{ 0 . 1 , 1 , 1 0 \}$ . The $\ell _ { 1 }$ or $\ell _ { 2 }$ regularization parameters are tuned using cross-validation. Figure 4 summarizes the numerical results, with the sample size n on the x-axis and the logarithm of relative error on the $y -$ axis. The relative error is defined as $| \hat { c } - c ^ { * } | / | c ^ { * } | \mathrm { ~ o r ~ } \| \hat { \omega } - \omega ^ { * } \| _ { 2 } / \| \omega ^ { * } \| _ { 2 }$ , where $( \hat { \omega } , \hat { c } )$ denotes the estimated parameters and $( \omega ^ { \ast } , c ^ { \ast } )$ the ground truth. All box plots are generated using 5 independent trials. The results indicate that, for all methods, estimation accuracy improves as the sample size n increases. In particular, the exact and greedy algorithms consistently outperform the continuous relaxation and Lasso-based approaches in estimating the weight vector ω<sup>∗</sup>, especially in the small-sample and low-SNR regimes.

## 6.4. Results on Out-of-Sample Performance

We examine the out-of-sample performance of various methods using the population newsvendor risk defined in (8), which is accurately estimated via sample average approximation with $1 0 ^ { 5 }$ testing samples. The experimental setup follows that of the previous subsection, except that we also examine diferent choices of the sparsity level d. Recall the ground truth weight parameter $\omega ^ { * }$ is fixed with support size $| S ^ { * } | = 1 0 .$ . For the Lasso method, the $\ell _ { 1 }$ regularization parameter is fine-tuned to achieve the desired sparsity level in the estimated parameter. Figure 5 summarizes the numerical results, with all error bars generated from 5 independent trials. The left-hand-side subplots display the sample size n on the $x \cdot$ axis. These results show that all methods improve with increasing $n ,$ while the exact and greedy algorithms consistently yield lower out-of-sample operational costs compared to the continuous relaxation and Lasso-based approaches. The right-hand-side subplots present the sparsity level d on the x-axis. When the true sparsity level |S<sup>∗</sup>| is underestimated $( d < | S ^ { * } | )$ 2 the exact and greedy algorithms dominate the others in terms of performance. When it is overestimated, the out-of-sample performance tends to stabilize. Together with the results in Sections 6.2 and 6.3, our numerical findings demonstrate that the variable selection framework delivers satisfactory parameter estimation and out-of-sample performance while using substantially fewer covariates and yielding interpretable decisions.

![](images/9594e7114ada34f160dbe87320564ca0a4fdf3d701070870ee75b96341de7812.jpg)  
Figure 4 Comparison in terms of logarithm of relative errors for diferent $( n , D , { \mathrm { S N R } } )$

![](images/104eb54689243da3f3a442b5a29a9b5fcc4030eb1c030ad746c0ee5d0b448265.jpg)  
Figure 5 Comparison in terms of out-of-sample operational cost for diferent (n, d, SNR).

## 7. Real-world case study

In this section, we evaluate the performance of various approaches using retail sales data from Kaggle[17]. The real-world dataset is provided by Corporación Favorita, an Ecuadorianbased grocery retailer that sells over 200,000 products. During preprocessing, we focus on hardware products and define the response variable Y as the sum of weekly unit sales across all four hardware product types. The covariate vector X is constructed using standard feature transformations, including one-hot encoding and normalization. The data are restricted to the time period from 2012-12-31 to 2013-12-30. The raw features in the dataset include: store number, city index, week index, month index, weekly averaged oil price, weekly promotion intensity, weekly transaction volume, holiday indicator, weekly number of hardware SKUs sold, and weekly number of active sales days. In addition, we incorporate the following two-way interactions as suggested in Chang et al. [9]: (i) holiday indicator with weekly promotion intensity; (ii) store number with weekly averaged oil price; (iii) month index with city index; (iv) month index with store number; and (v) holiday indicator with store number. After preprocessing, the final dataset consists of n = 2239 samples and D = 1157 dimensions. We split the data into 70% training and 30% testing sets across 20 independent trials. The training data are used to estimate the weight vector and intercept, while the testing data are used for evaluating the (unit) out-of-sample performance.

Table 2 Out-of-sample operational cost and its standard deviation with diferent choices of penalty coeficient γ and sparsity level d (bold-faced values represent the lowest cost under the same γ)
<table><tr><td rowspan="2">method</td><td colspan="2"> $\gamma = 0 . 5$ </td><td colspan="2"> $\gamma = 1$ </td><td colspan="2"> $\gamma = 5$ </td><td colspan="2"> $\gamma = 1 0$ </td></tr><tr><td>mean</td><td>std</td><td>mean</td><td>std</td><td>mean</td><td>std</td><td>mean</td><td>std</td></tr><tr><td>Full model</td><td>4.5359</td><td>0.1709</td><td>4.0300</td><td>0.1468</td><td>3.7305</td><td>0.1436</td><td>3.7678</td><td>0.1769</td></tr><tr><td>Exact method (d = 50)</td><td>4.9280</td><td>0.1842</td><td>4.2086</td><td>0.1672</td><td>3.2440</td><td>0.1154</td><td>3.2073</td><td>0.1246</td></tr><tr><td>Exact method (d = 100)</td><td>4.7706</td><td>0.1840</td><td>4.0875</td><td>0.1639</td><td>3.3469</td><td>0.1289</td><td>3.3499</td><td>0.1540</td></tr><tr><td>Exact method (d = 200)</td><td>4.6522</td><td>0.1748</td><td>4.0460</td><td>0.1507</td><td>3.4718</td><td>0.1363</td><td>3.5306</td><td>0.1586</td></tr><tr><td>Exact method (d = 500)</td><td>4.5707</td><td>0.1723</td><td>4.0348</td><td>0.1479</td><td>3.6941</td><td>0.1462</td><td>3.7306</td><td>0.1793</td></tr><tr><td>Exact method (d = 800)</td><td>4.5374</td><td>0.1696</td><td>4.0281</td><td>0.1475</td><td>3.7296</td><td>0.1435</td><td>3.7670</td><td>0.1757</td></tr></table>

We report the out-of-sample performance of the diferent methods in Table 2. Baseline approaches include the full model without variable selection, as well as our hard-cardinalityconstrained model (3) under various sparsity levels d. An $\ell _ { 2 }$ regularization term with different coeficients $\gamma$ is added into all models to mitigate overfitting. The numerical results indicate that as the sparsity budget d increases, the out-of-sample performance of our variable selection framework approaches that of the full model. Moreover, with an appropriate choice of the $\ell _ { 2 }$ regularization parameter and sparsity budget, our framework not only yields interpretable decisions but also attains the lowest out-of-sample operational costs among all methods considered.

Table 3 Top 20 features selected using the exact algorithm with $d = 1 0 0$ on the real-world data, where the entries with x represent the interaction between two features
<table><tr><td>Rank</td><td> $\gamma = 0 . 5$ </td><td> $\gamma = 1$ </td><td> $\gamma = 5$ </td><td> $\gamma = 1 0$ </td></tr><tr><td>1</td><td>n_days</td><td> $\mathtt { n \_ d a y s }$ </td><td>n_days</td><td> $\mathtt { n \_ d a y s }$ </td></tr><tr><td>2</td><td>n_items</td><td>n_items</td><td>n_items</td><td>n_items</td></tr><tr><td>3 4</td><td>transactions store_44</td><td> $\mathrm { c l u s t e r } \_ 5 \_ \mathrm { x } \_ \mathrm { o i l }$  store  $\mathbf { \underline { { 4 4 } } \underline { { ~ x ~ } } _ { 0 } i l }$ </td><td> $\mathrm { c l u s t e r } \_ 5 \_ \mathrm { x } \_ \mathrm { o i l }$   $\mathsf { s t o r e \_ } 4 4 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$ </td><td>cluster_5 store_44</td></tr><tr><td>5</td><td></td><td> $\mathrm { c l u s t e r } \_ 5$ </td><td> $\mathrm { c l u s t e r } \_ 5$ </td><td></td></tr><tr><td></td><td> $\mathrm { c l u s t e r } \_ 5$ </td><td></td><td></td><td> $\mathsf { s t o r e \_ } 4 4 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$ </td></tr><tr><td>6</td><td> $\mathsf { s t o r e \_ } 4 4 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$ </td><td> $\mathsf { s t o r e \_ } 4 4$ </td><td> $\mathsf { s t o r e \_ 4 4 }$ </td><td> $\mathrm { c l u s t e r } \_ 5 \_ \mathrm { x } \_ \mathrm { o i l }$ </td></tr><tr><td>7</td><td> $\mathrm { c l u s t e r } \_ 5 \_ \mathrm { x } \_ \mathrm { o i l }$ </td><td>transactions</td><td>transactions</td><td>transactions</td></tr><tr><td>8</td><td> $\mathrm { c l u s t e r } \_ 8$ </td><td> $\mathrm { c l u s t e r } \_ 8$ </td><td>month_9_x_store_9 city_Cuenca</td><td></td></tr><tr><td>9</td><td> $\mathsf { c l u s t e r \_ 8 \_ x \_ o i l }$ </td><td> $\mathsf { c l u s t e r \_ 8 \_ x \_ o i l }$ </td><td> $\mathsf { s t o r e \_ 8 \_ X \_ o i l }$ </td><td> $\mathsf { s t o r e } _ { - } 3$ </td></tr><tr><td>10</td><td> $\mathrm { t y p e \_ C \_ x \_ o i l }$ </td><td>store_3</td><td>store_8</td><td>store_8</td></tr><tr><td>11</td><td> $\mathrm { t y p e \_ C }$ </td><td> $\mathsf { s t o r e } \_ 8$ </td><td> $\mathsf { s t o r e } _ { - } 3$ </td><td>store_9_x_oil</td></tr><tr><td>12</td><td></td><td></td><td></td><td></td></tr><tr><td>13</td><td> $\mathsf { s t o r e } _ { - } 3$ </td><td> $\mathsf { s t o r e \_ 8 \_ X \_ o i l }$ </td><td>city_Cuenca</td><td> $\mathrm { m o n t h { \_ } 8 \_ x \_ s t o r e \_ } 9$ </td></tr><tr><td></td><td> $\mathsf { s t o r e \_ } 3 \_ \underline { { \mathbf { x } } } \mathsf { \_ o i l }$ </td><td> $\mathsf { s t o r e \_ } 3 \_ \underline { { \mathbf { x } } } \_ { \mathrm { o i l } }$ </td><td> $\mathsf { s t o r e } \_ 9$ </td><td> $\mathrm { m o n t h { \_ } } 9 \_ { \mathrm { - } } \mathrm { x \_ s t o r e \_ } 9$ </td></tr><tr><td>14</td><td> $\mathsf { s t o r e } \_ 9$ </td><td> $\mathsf { s t o r e } \_ 9$ </td><td> $\mathrm { { t y p e \_ C } }$ </td><td> $\mathrm { m o n t h \_ 2 \_ x \_ s t o r e \_ } 4 4$ </td></tr><tr><td>15</td><td> $s \mathrm { t o r e } \_ 8$ </td><td> $\mathrm { t y p e \_ C \_ x \_ o i l }$ </td><td> $\mathsf { s t o r e \_ } 9 \_ \underline { { \mathbf { x } } } \mathsf { \_ o i l }$ </td><td> $\mathrm { t y p e \_ C }$ </td></tr><tr><td>16</td><td> $\mathsf { s t o r e \_ } 9 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$ </td><td> $\mathsf { s t o r e \_ } 9 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$ </td><td> $\mathrm { t y p e \_ C \_ x \_ o i l }$ </td><td> $\mathsf { s t o r e \_ 8 \_ X \_ o i l }$ </td></tr><tr><td>17</td><td> $\mathsf { s t o r e \_ 8 \_ X \_ o i l }$ </td><td> $\mathrm { t y p e \_ C }$ </td><td> $\mathsf { s t o r e \_ } 3 \_ \underline { { \mathbf { x } } } \mathsf { \_ o i l }$ </td><td>city  $\mathsf { C u e n c a \_ x \_ o i l }$ </td></tr><tr><td>18</td><td> $\mathrm { c i t y \_ C u e n c a }$ </td><td> $\mathrm { \ h o l i d a y { \_ } x { \_ } s t o r e { \_ } } 4 4$ </td><td>month_8_x_store_9 month</td><td> $\mathsf { \Omega } _ { - } ^ { 8 } \underline { { \mathbf { x } } } \mathsf { \Omega } _ { - } \mathsf { s t o r e \_ } 4 4$ </td></tr><tr><td>19</td><td> $\mathrm { \ h o l i d a y { \_ } x { \_ } s t o r e { \_ } } 4 4$ </td><td> $\mathsf { s t o r e \_ } 4 1 \_ \mathrm { x \_ o i l }$ </td><td> $\mathrm { \ c i t y \_ C u e n c a \_ x \_ o i l }$ </td><td> $\mathsf { s t o r e \_ } 4 8 \_ \mathrm { x \_ o i l }$ </td></tr><tr><td>20</td><td> $\mathrm { c i t y \_ Q u i t o }$ </td><td> $\mathsf { s t o r e \_ 4 1 }$ </td><td> $\mathrm { c l u s t e r \_ 1 0 \_ x \_ o i l }$ </td><td> $\mathsf { s t o r e } \_ 9$ </td></tr></table>

We report the top 20 selected features under diferent configurations of $( \gamma , d )$ in Tables 3 and 4. The features are ranked according to their averaged absolute weight values in 20 independent trials. From these selected features, we draw the following operational insights: (I) The number of days per week on which customers purchase hardware products (n\_days) and the weekly number of active SKUs (n\_items) emerge as the most persistent features. A possible explanation is that customer demand is largely driven by weekend shopping habits and the breadth of product assortment available in stores. (II) The historical weekly transaction volume (transactions) consistently appears as a strong predictor, as it is highly correlated with future customer demand. (III) The location and type of the store (such as those represented by the store\_3, 8, 9, 44, C, city\_Cuenca) play a critical role in predicting hardware sales. (IV) The interaction terms between store 9 and the months of August and September, as well as between store 44 and February and August, suggest that the efects of store location and time on operational outcomes are heterogeneous, with certain stores exhibiting pronounced seasonal demand spikes.

Table 4 Top 20 features selected using the exact algorithm with $d = 5 0$ on the real-world data  
Rank $\gamma = 0 . 5$ $\gamma = 1$ $\gamma = 5$ $\gamma = 1 0$   
1 n\_days n\_days $\mathtt { n \_ d a y s }$ n\_days   
2 n\_items n\_items n\_items n\_items   
3 transactions $\mathrm { c l u s t e r } \_ 5 \_ \mathrm { x } \_ \mathrm { o i l }$ cluster\_5\_x\_oil store\_44\_x\_oil   
4 $\mathsf { s t o r e \_ 4 4 }$ $\mathsf { s t o r e \_ } 4 4 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$ $\mathsf { s t o r e \_ } 4 4 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$ $\mathrm { c l u s t e r } \_ 5 \_ \mathrm { x } \_ \mathrm { o i l }$   
5 $\mathrm { c l u s t e r } \_ 5$ $\mathsf { s t o r e \_ } 4 4$ $\mathrm { c l u s t e r } \_ 5$ $\mathrm { c l u s t e r } \_ 5$   
6 $\mathsf { s t o r e \_ } 4 4 \_ \mathrm { x _ { - } }$ \_oil transactions $\mathsf { s t o r e \_ 4 4 }$ store\_44   
7 $\mathrm { c l u s t e r } \_ 5 \_ \mathrm { x } \_ \mathrm { o i l }$ $\mathrm { c l u s t e r } \_ 5$ transactions transactions   
8 $\mathrm { t y p e \_ C \_ x \_ o i l }$ $\mathrm { t y p e \_ C \_ x \_ o i l }$ store\_8 city\_Cuenca   
9 $\mathrm { t y p e \_ C }$ $\mathrm { t y p e \_ C }$ store\_8\_x\_oil store\_3   
10 $\mathrm { c l u s t e r } \_ 8$ $\mathrm { c l u s t e r } \_ 8$ city\_Cuenca store\_8   
11 $\mathsf { c l u s t e r \_ 8 \_ x \_ o i l }$ $\mathsf { c l u s t e r \_ 8 \_ x \_ o i l }$ $\mathsf { s t o r e } _ { - } 3$ $\mathsf { s t o r e \_ 8 \_ X \_ o i l }$   
12 $\mathsf { s t o r e } _ { - } 3$ $\mathsf { s t o r e } \_ 9$ $\mathsf { s t o r e \_ } 9 \_ \underline { { \mathbf { x } } } \mathsf { \_ o i l }$ $\mathrm { m o n t h { \_ } } 9 \_ { \mathrm { - } } \mathrm { x \_ s t o r e \_ } 9$   
13 $\mathsf { s t o r e \_ } 3 \_ \underline { { \mathbf { x } } } \mathsf { \_ o i l }$ $\mathsf { s t o r e \_ } 9 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$ $\mathrm { \ c i t y \_ C u e n c a \_ x \_ o i l }$ $\mathrm { m o n t h { \_ } 8 \_ x \_ s t o r e \_ } 9$   
14 $\mathsf { s t o r e } \_ 9$ $\mathsf { s t o r e } \_ 8$ $\mathsf { s t o r e } \_ 9$ $\mathsf { s t o r e \_ } 9 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$   
15 $\mathsf { s t o r e \_ } 9 \_ \underline { { \mathrm { ~ x ~ } } } \mathsf { o i l }$ $\mathsf { s t o r e \_ 8 \_ X \_ o i l }$ $\mathrm { { t y p e \_ C } }$ $\mathrm { m o n t h \_ 2 \_ x \_ s t o r e \_ } 4 4$   
16 $s \mathrm { t o r e } \_ 8$ $\mathsf { s t o r e } _ { - } 3$ $\mathsf { s t o r e \_ } 3 \_ \underline { { \mathbf { x } } } \mathsf { \_ o i l }$ $\mathsf { s t o r e \_ } 3 \_ \underline { { \mathbf { x } } } \mathsf { \_ o i l }$   
17 $\mathsf { s t o r e \_ 8 \_ X \_ o i l }$ $\mathsf { s t o r e \_ } 3 \_ \underline { { \mathbf { x } } } \_ { \mathrm { o i l } }$ month\_9\_x\_store\_9 store\_9   
18 $\mathrm { c i t y \_ C u e n c a }$ $\mathrm { \ h o l i d a y { \_ } x { \_ } s t o r e { \_ } 4 4 \ t y p e { \_ } C _ { \_ } x { \_ } o i l }$ $\mathrm { m o n t h \_ 8 \_ x \_ s t o r e \_ 4 4 }$   
19 $\mathrm { \ c i t y \_ C u e n c a \_ x \_ o i l }$ city\_Cuenca ${ \mathrm { c l u s t e r } } \ 8$ $\mathrm { t y p e \_ C \_ x \_ o i l }$   
20 $\mathrm { c i t y \_ Q u i t o }$ $\mathrm { \ c i t y \_ C u e n c a \_ x \_ o i l }$ store $\_ { 4 8 \_ { \mathrm { ~ X ~ } } \mathrm { o i l } }$ $\mathsf { s t o r e \_ } 4 8 \_ \mathrm { x \_ o i l }$

## 8. Conclusion

In this paper, we studied variable selection for the feature-based newsvendor problem under a hard cardinality constraint on the number of selected covariates. We establish both computational algorithms and statistical performance guarantees of this framework. The numerical results illustrate the trade-of among computational efort, support-recovery accuracy, and out-of-sample operating cost.

Several directions remain open for future research. First, it is promising to extend this variable selection framework for nonlinear policy classes, such as the kernel-based decision rules [36]. Second, it is an interesting direction to design more eficient algorithms and hyperparameter selection procedures [12, 11]. Finally, extending the statistical analysis to misspecified, heavy-tailed, dependent, or distributionally robust settings would broaden the applicability of our framework.

## References

[1] Atamturk A, Gomez A (2025) Rank-one convexification for sparse regression. JMLR 1–50.

[2] Bajwa WU, Mixon DG (2012) Group model selection using marginal correlations: The good, the bad and the ugly. 2012 50th Annual Allerton Conference on Communication, Control, and Computing, 494–501.

[3] Ban GY, Rudin C (2019) The big data newsvendor: Practical insights from machine learning. Operations Research 67(1):90–108.

[4] Belloni A, Chernozhukov V (2011) ℓ<sub>1</sub>-penalized quantile regression in high-dimensional sparse models. The Annals of Statistics 39(1):82–130.

[5] Bertsimas D, Kallus N (2020) From predictive to prescriptive analytics. Management Science 66(3):1025–1044.

[6] Bertsimas D, Koduri N (2022) Data-driven optimization: A reproducing kernel hilbert space approach. Operations Research 70(1):454–471.

[7] Bertsimas D, Pauphilet J, Van Parys B (2021) Sparse classification: a scalable discrete optimization perspective. Machine Learning 110(11-12):3177–3209.

[8] Blanchet J, Kang Y, Murthy K (2019) Robust Wasserstein profile inference and applications to machine learning. Journal of Applied Probability 56(3):830–857.

[9] Chang J, Yang L, Zhang Y, Zhou W (2025) Feature-rich, data-private: A sparse learning framework for the highdimensional newsvendor. SSRN 1–98.

[10] Civril A, Magdon-Ismail M (2009) On selecting a maximum volume sub-matrix of a matrix and related problems. Theoretical Computer Science 410(47-49):4801–4811.

[11] Cory-Wright R, Gómez A (2023) Optimal cross-validation for sparse linear regression. arXiv preprint arXiv:2306.14851

[12] Cory-Wright R, Gómez A (2025) Stability regularized cross-validation. arXiv preprint arXiv:2505.06927 .

[13] Dai S (2023) Variable selection in convex quantile regression: ℓ<sub>1</sub>-norm or ℓ<sub>0</sub>-norm regularization? European Journal of Operational Research 305(1):338–355.

[14] Donti P, Amos B, Kolter JZ (2017) Task-based end-to-end model learning in stochastic optimization. Advances in neural information processing systems 30.

[15] Dudley RM (1999) Uniform central limit theorems, volume 23 (Cambridge university press Cambridge).

[16] Fan J, Li R (2001) Variable selection via nonconcave penalized likelihood and its oracle properties. Journal of the American Statistical Association 96(456):1348–1360.

[17] Favorita C (2017) Corporación favorita grocery sales forecasting. Kaggle, https://www.kaggle.com/competitions/favorita grocery-sales-forecasting/data.

[18] Günlük O, Linderoth J (2011) Perspective reformulation and applications. Mixed integer nonlinear programming, 61–89 (Springer).

[19] Huang J, Horowitz JL, Ma S (2008) Asymptotic properties of bridge estimators in sparse high-dimensional regression models. The Annals of Statistics 36(2):587–613.

[20] Huber J, Müller S, Fleischmann M, Stuckenschmidt H (2019) A data-driven newsvendor problem: From data to deci sion. European Journal of Operational Research 278(3):904–915.

[21] Kallus N, Mao X (2023) Stochastic optimization forests. Management Science 69(4):1975–1994.

[22] Knight K (1998) Limiting distributions for L<sub>1</sub> regression estimators under general conditions. The Annals of Statistics 26(2):755–770, URL http://dx.doi.org/10.1214/aos/1028144858.

[23] Koenker R, Bassett G (1978) Regression quantiles. Econometrica 46(1):33–50.

[24] Li Y, Dey SS, Xie W (2024) On sparse canonical correlation analysis. Advances in Neural Information Processing Systems 37:10707–10734.

[25] Li Y, Xie W (2024) Beyond symmetry: best submatrix selection for the sparse truncated svd: Y. li, w. xie. Mathematical Programming 208(1):1–50.

[26] Li Y, Xie W (2025) Exact and approximation algorithms for sparse principal component analysis. INFORMS Journal on Computing 37(3):582–602.

[27] Liyanage LH, Shanthikumar JG (2005) A practical inventory control policy using operational statistics. Operations research letters 33(4):341–348

[28] Notz PM, Pibernik R (2022) Prescriptive analytics for flexible capacity management. Management Science 68(3):1756– 1775.

[29] Oroojlooyjadid A, Snyder LV, Takáč M (2020) Applying deep learning to the newsvendor problem. IISE Transaction 52(4):444–463.

[30] Qi M, Grigas P, Shen ZJ (2025) Integrated conditional estimation-optimization. Operations Research .

[31] Sadana U, Chenreddy A, Delage E, Forel A, Frejinger E, Vidal T (2025) A survey of contextual optimization methods for decision-making under uncertainty. European Journal of Operational Research 320(2):271–289.

[32] Serrano B, Minner S, Schifer M, Vidal T (2024) Bilevel optimization for feature selection in the data-driven newsven dor problem. European Journal of Operational Research 315(2):703–714.

[33] Tibshirani R (1996) Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society Series B: Statistical Methodology 58(1):267–288.

[34] Tropp JA (2010) User-friendly tail bounds for sums of random matrices. Foundations of Computational Mathematics 12:389–434.

[35] Wainwright MJ (2019) High-dimensional statistics: A non-asymptotic viewpoint, volume 48 (Cambridge university press).

[36] Wang J, Dey SS, Xie Y (2023) Variable selection for kernel two-sample tests. arXiv preprint arXiv:2302.07415 .

[37] Xie H, Huang J (2009) SCAD-penalized regression in high-dimensional partially linear models. The Annals of Statistics 37(2):673–696.

[38] Xie W, Deng X (2020) Scalable algorithms for the sparse ridge regression. SIAM Journal on Optimization 30(4):3359– 3386.

[39] Zhang CH (2010) Nearly unbiased variable selection under minimax concave penalty. The Annals of Statistic 38(2):894–942.

[40] Zhang Y, Gao J (2017) Assessing the performance of deep learning algorithms for newsvendor problem. International conference on neural information processing, 912–921.

## Appendix for “Variable Selection for Feature-Based Newsvendor”

## EC.1. Proof of Proposition 1

We prove the NP-hardness result by making the reduction from Exact Cover by 3-Sets (X3C) to Problem (3). The formal description of (X3C) is as follows. Let $Q$ be a ground set consisting of $m : = 3 d$ elements, and $\mathcal { C } = \{ C _ { 1 } , \ldots , C _ { D } \}$ be the collection of subsets of $Q$ such that $| C _ { i } | = 3$ for $i \in [ D ]$ . Problem (X3C) aims to determine whether there exists a subset of ${ \mathcal { C } } ,$ denoted as $\mathcal { C } ^ { \prime }$ such that every element in $Q$ occurs exactly once in $\mathcal { C } ^ { \prime }$

Denote by $A \in \mathbb { R } ^ { m \times D }$ the incidence matrix of Problem (X3C) such that $A [ i , j ] = 1$ if the i-th element of $Q$ belongs to the subset $C _ { j }$ and otherwise $A [ i , j ] = 0 , \forall i , j$ . We define a scalar $M _ { 0 }$ such that $\begin{array} { r } { M _ { 0 } > \frac { m } { 3 \gamma ( b \wedge h ) } } \end{array}$ . We construct a special instance of Problem (3) by taking $n = 2 m$ and the i-th observation as

$$
\begin{array} { r } { ( x _ { i } , y _ { i } ) = \left\{ \begin{array} { c l l } { ( M _ { 0 } A [ i , : ] ^ { \top } , M _ { 0 } ) } & { \mathrm { i f ~ } 1 \leq i \leq m , } \\ { \left( - M _ { 0 } A [ i - m , : ] ^ { \top } , - M _ { 0 } \right) } & { \mathrm { i f ~ } m + 1 \leq i \leq n . } \end{array} \right. } \end{array}
$$

For weight vector $\boldsymbol \omega \in \mathbb { R } ^ { D }$ , define the error vector $r = \mathbf { 1 } - A \omega \in \mathbb { R } ^ { m }$ . For any scalar $x ,$ define $x _ { + } = \operatorname* { m a x } \{ x , 0 \}$ and $x _ { - } = \operatorname* { m a x } \{ - x , 0 \}$ . Define $b \wedge h = \operatorname* { m i n } ( b , h )$ . Thus, the fitting error part can be bounded as

$$
{ \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \left[ { b } \cdot \operatorname* { m a x } \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \} + h \cdot \operatorname* { m a x } \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \} \right]
$$

$$
= \frac { 1 } { 2 m } \sum _ { i = 1 } ^ { m } \left[ b \cdot ( M _ { 0 } r [ i ] - c ) _ { + } + + h \cdot ( M _ { 0 } r [ i ] - c ) _ { - } + b \cdot ( - M _ { 0 } r [ i ] - c ) _ { + } + h \cdot ( - M _ { 0 } r [ i ] - c ) _ { - } \right]
$$

$$
\geq \frac { b \wedge h } { 2 m } \sum _ { i = 1 } ^ { m } \Big [ | M _ { 0 } r [ i ] - c | + | M _ { 0 } r [ i ] + c | \Big ]
$$

$$
\geq { \frac { ( b \wedge h ) M _ { 0 } } { m } } \| r \| _ { 1 } ,
$$

where the last inequality is based on the relation $| u + v | + | u - v | \geq 2 | u |$

Let $( \omega , c )$ be a feasible solution to Problem (3) with $\| \omega \| _ { 0 } \leq d$ . We first show that $F ( \omega , c ) \geq$ $d / ( 2 \gamma )$

• Assume that $\| \omega \| _ { 2 } ^ { 2 } \leq d ,$ , then

$$
| \mathbf { 1 } ^ { \top } \boldsymbol { \omega } | = \left| \sum _ { i : \ \omega [ i ] \neq 0 } \omega [ i ] \right| \leq \sqrt { \sum _ { \substack { i : \ \omega [ i ] \neq 0 } } 1 } \sqrt { \sum _ { \substack { i : \ \omega [ i ] \neq 0 } } \omega ^ { 2 } [ i ] } \leq \sqrt { d } \| \boldsymbol { \omega } \| _ { 2 } ,\tag{EC.1}
$$

where the first inequality uses Cauchy-Schwarz inequality. It follows that

$$
\begin{array} { r l r } {  { d - \| \boldsymbol { \omega } \| _ { 2 } ^ { 2 } \leq d - \frac { | { \bf 1 } ^ { \top } \boldsymbol { \omega } | ^ { 2 } } { d } = ( d - { \bf 1 } ^ { \top } \boldsymbol { \omega } ) \cdot ( 1 + { \bf 1 } ^ { \top } \boldsymbol { \omega } / d ) } } \\ & { } & \\ & { } & { \leq 2 ( d - { \bf 1 } ^ { \top } \boldsymbol { \omega } ) } \\ & { } & { = \frac { 2 } { 3 } { \bf 1 } ^ { \top } r \leq \frac { 2 } { 3 } \| r \| _ { 1 } , } \end{array}
$$

where the first inequality is by (EC.1), the second inequality is because $| \mathbf { 1 } ^ { \top } \boldsymbol { \omega } | \leq$ ${ \sqrt { d } } \| \omega \| _ { 2 } \leq d ,$ and the last equality is because each column of A has exactly three ones and

$$
\mathbf { 1 } ^ { \top } r = m - \mathbf { 1 } ^ { \top } A \omega = m - ( A ^ { \top } \mathbf { 1 } ) ^ { \top } \omega = 3 d - 3 ( \mathbf { 1 } ^ { \top } \omega ) .\tag{EC.2}
$$

then, the objective

$$
F ( \omega , c ) \geq \frac { ( b \wedge h ) M _ { 0 } } { m } \| r \| _ { 1 } + \frac { 1 } { 2 \gamma } \left( d - \frac { 2 } { 3 } \| r \| _ { 1 } \right) \geq \frac { d } { 2 \gamma } ,\tag{EC.3}
$$

where the last inequality is by the choice of $M _ { 0 }$ .

• Otherwise, for $\| \omega \| _ { 2 } ^ { 2 } > d ,$ , the objective value

$$
F ( \omega , c ) \geq \frac { 1 } { 2 \gamma } \| \omega \| _ { 2 } ^ { 2 } > \frac { d } { 2 \gamma } .
$$

We now show that Problem (X3C) can be reduced to Problem (3). Assume we solve Problem (3) with optimal value upper bounded by $\frac { d } { 2 \gamma }$ and denote its optimal solution as $( \omega , c )$ This happens only if $\| \omega \| _ { 2 } ^ { 2 } \leq d ,$ and the inequalities in (EC.3) all hold with equalities. It implies that $r = 0 = 1 - A \omega$ . By (EC.2), it holds that $\mathbf { 1 } ^ { \top } \omega = d .$ By (EC.1), it holds that $\| \omega \| _ { 2 } ^ { 2 } = d .$ . Then, all inequalities in (EC.1) always hold with equalities, and therefore $\omega$ has exactly d nonzero entries. The exact cover of (X3C) is given by $\{ C _ { j } : \omega [ j ] \neq 0 \}$

Conversely, suppose Problem (X3C) has an exact cover. Let $\boldsymbol \omega \in \mathbb { R } ^ { D }$ be the associated weight vector such that $\omega [ i ] = 1$ if the i-th element of C is selected to form the cover and otherwise $\omega [ i ] = 0$ . Then the incidence matrix A satisfies that $r : = 1 - A \omega = 0$ and $\omega$ satisfies $\| \omega \| _ { 0 } = d$ and $\| \omega \| _ { 2 } ^ { 2 } = d .$ The objective value of the feasible solution $( \omega , 0 )$ to Problem (3) equals

$$
\begin{array} { c } { \displaystyle F ( \omega , 0 ) = \displaystyle \frac { 1 } { 2 m } \sum _ { i = 1 } ^ { m } \left[ b \cdot ( M _ { 0 } r [ i ] ) _ { + } + h \cdot ( M _ { 0 } r [ i ] ) _ { - } + b \cdot ( - M _ { 0 } r [ i ] ) _ { + } + h \cdot ( - M _ { 0 } r [ i ] ) _ { - } \right] + \frac { d } { 2 \gamma } } \\ { \displaystyle = \frac { d } { 2 \gamma } . } \end{array}
$$

Itjustifies that the optimal value of Problem (3) is bounded by $d / ( 2 \gamma )$ . The proof is completed.

## EC.2. Proof of Theorem 1

Both Big-M and perspective reformulation can be written as the following form of optimization:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \omega , c , s , \mu } } & { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big [ b \operatorname* { m a x } \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \} + h \operatorname* { m a x } \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \} \Big ] + \frac { 1 } { 2 \gamma } \| \mu \| _ { 1 } } \\ { \mathrm { s . t . } } & { ( \omega , c , s , \mu ) \in \mathcal { V } ^ { \mathrm { B i g . M } } \mathrm { o r } \mathcal { V } ^ { \mathrm { P e r s } } , } \\ & { \displaystyle \sum _ { j \in [ D ] } s [ j ] \leq d , } \end{array}\tag{EC.4}
$$

where

$$
\begin{array} { r l } & { \mathcal { V } ^ { \mathrm { B i g . M } } = \left\{ ( \omega , s , \mu , c ) \in \mathbb { R } ^ { D } \times \{ 0 , 1 \} ^ { D } \times \mathbb { R } _ { + } ^ { D } \times \mathbb { R } : \frac { \omega [ i ] ^ { 2 } \leq \mu [ i ] , i \in [ D ] } { | \omega [ i ] | \leq M [ i ] , s [ i ] , i \in [ D ] } \right\} , } \\ & { \mathcal { V } ^ { \mathrm { P e r s } } = \left\{ ( \omega , s , \mu , c ) \in \mathbb { R } ^ { D } \times \{ 0 , 1 \} ^ { D } \times \mathbb { R } _ { + } ^ { D } \times \mathbb { R } : \frac { \omega [ i ] ^ { 2 } \leq \mu [ i ] s [ i ] , i \in [ D ] } { | \omega [ i ] | \leq M [ i ] s [ i ] , i \in [ D ] } \right\} . } \end{array}
$$

These two problems are equivalent because $\mathcal { V } ^ { \mathrm { B i g - M } } = \mathcal { V } ^ { \mathrm { P e r s } }$ . Indeed, when the binary variable $s [ i ] = 0$ , it holds that $\omega [ i ] = 0$ and the first constraints appeared in $\mathcal { V } ^ { \mathrm { B i g - M } }$ and $\mathcal { V } ^ { \mathrm { P e r s } }$ are redundant. Otherwise, the first constraints in $\mathcal { V } ^ { \mathrm { B i g - M } }$ and $\mathcal { V } ^ { \mathrm { P e r s } }$ are equivalent.

For Part (II), if considering the continuous relaxation of these two problems, the binary constraints $s [ i ] \in \{ 0 , 1 \}$ in $\mathcal { V } ^ { \mathrm { B i g - M } }$ and $\mathcal { V } ^ { \mathrm { P e r s } }$ are replaced by $s [ i ] \in [ 0 , 1 ]$ for $i \in [ D ]$ . Denoted the continuous relaxation of these two sets as $\overline { { \gamma } } ^ { \mathrm { B i g - M } }$ and $\overline { { \mathcal { V } } } ^ { \mathrm { { P e r s } } }$ . It holds that $\overline { { \mathcal { V } } } ^ { \mathrm { { P e r s } } } \subseteq \overline { { \mathcal { V } } } ^ { \mathrm { { B i g - M } } }$ as $\omega [ i ] ^ { 2 } \leq \mu [ i ] s [ i ]$ implies $\omega [ i ] ^ { 2 } \leq \mu [ i ]$ . Therefore, the optimal value of continuous relaxation of perspective reformulation is larger than that of Big-M reformulation.

## EC.3. Proof of Theorem 2

For Part (I), we bound the support size of solution in Algorithm 1. Let $( \widehat { \omega } , \widehat { c } , \widehat { s } , \widehat { \mu } )$ be the optimal solution to the continuous relaxation of (Perspective). Let $( \tilde { \omega } , \tilde { c } , \tilde { s } )$ be the output of Algorithm 1. The support size of ω˜ is bounded by the sum of random variables $N : =$ $\textstyle \sum _ { j \in [ D ] } { \tilde { s } } [ j ]$ , where $\tilde { s } [ j ] \sim \operatorname { B e r n o u l l i } ( \widehat { s } [ j ] )$ . Its expected value

$$
\mathbb { E } \left[ N \right] = \sum _ { j \in \left[ D \right] } \widehat { s } [ j ] \leq d .
$$

For $t > 0$ , we compute the moment-generating function of N:

$$
\begin{array} { l } { { \displaystyle M _ { N } ( t ) = \mathbb E [ e ^ { t N } ] = \prod _ { j \in [ D ] } \mathbb E [ e ^ { t \widehat { s } [ j ] } ] } } \\ { { \displaystyle \qquad = \prod _ { j \in [ D ] } \big ( 1 + ( \widehat s [ j ] e ^ { t } - \widehat s [ j ] ) \big ) \leq \prod _ { j \in [ D ] } \exp \big ( \widehat s [ j ] e ^ { t } - \widehat s [ j ] \big ) } } \\ { { \displaystyle \qquad = \exp \left( ( e ^ { t } - 1 ) \sum _ { j \in [ D ] } \widehat s [ j ] \right) \leq \exp ( d ( e ^ { t } - 1 ) ) , } } \end{array}
$$

where the first inequality is based on the relation that $1 + x \leq e ^ { x } , \forall x$ , and the last inequality is because $\textstyle \sum _ { j \in [ D ] } { \widehat { s } } [ j ] \leq d .$ . For $\Delta > 0$ , applying the Markov inequality gives

$$
\begin{array} { r l r } & { } & { \mathbb { P } \left( N \geq ( 1 + \Delta ) d \right) = \mathbb { P } \left( e ^ { t N } \geq e ^ { t ( 1 + \Delta ) d } \right) \leq \frac { M _ { N } ( t ) } { e ^ { t ( 1 + \Delta ) d } } } \\ & { } & { \leq \exp ( d ( e ^ { t } - 1 ) - t ( 1 + \Delta ) d ) . } \end{array}
$$

We choose t such that the upper bound above is as sharp as possible, yielding the optimal choice $t = \log ( 1 + \Delta )$ . Then, it holds that

$$
\begin{array} { r } { \mathbb { P } \left( N \ge ( 1 + \Delta ) d \right) \le \exp ( - d ( ( 1 + \Delta ) \log ( 1 + \Delta ) - \Delta ) ) . } \end{array}
$$

For $\Delta \geq 0$ , it is easy to verify that $\begin{array} { r } { \log ( 1 + \Delta ) \ge \frac { 2 \Delta } { 1 + \Delta } } \end{array}$ , and then

$$
\mathbb { P } \left( N \ge ( 1 + \Delta ) d \right) \le \exp \left( - \frac { d \Delta ^ { 2 } } { 2 + \Delta } \right) .
$$

Taking $\begin{array} { r } { \Delta = \frac { \xi } { 2 } + \sqrt { 2 \xi + \frac { \xi ^ { 2 } } { 4 } } } \end{array}$ with $\xi : = \log ( 2 / \delta ) / d$ yields

$$
\mathbb { P } \left( \| \tilde { \omega } \| _ { 0 } \ge ( 1 + \Delta ) d \right) \le \mathbb { P } \left( N \ge ( 1 + \Delta ) d \right) \le \frac { \delta } { 2 } .\tag{EC.5}
$$

We replace this bound with a conservative one $( \Delta = \log ( 2 / \delta ) / d + \sqrt { 2 \log ( 2 / \delta ) / d } )$ to simplify the notation in the theorem statement.

For Part (II), we first reformulate the continuous relaxation of Problem (Perspective):

$$
\begin{array} { r l } { \displaystyle \underset { \omega , c , s , \mu } { \operatorname* { m i n } } } & { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ b \operatorname* { m a x } \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \} + h \operatorname* { m a x } \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \} \right] + \frac { 1 } { 2 \gamma } \| \mu \| _ { 1 } } \\ { \mathrm { s . t . } } & { ( \omega , c , s , \mu ) \in \mathbb { R } ^ { D } \times \mathbb { R } \times [ 0 , 1 ] ^ { D } \times \mathbb { R } _ { + } ^ { D } , } \\ & { \displaystyle \sum _ { j \in [ D ] } s [ j ] \leq d , } \\ & { \displaystyle \omega [ i ] ^ { 2 } \leq \mu [ i ] s [ i ] , i \in [ D ] . } \end{array}\tag{EC.6}
$$

The variable $\mu$ above can be eliminated by taking $\begin{array} { r } { \mu [ i ] = \frac { \omega [ i ] ^ { 2 } } { s [ i ] } } \end{array}$ (by default, we take $\mu [ i ] = 0$ if $\omega [ i ] = s [ i ] = 0 )$ , yielding the reformulation

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \omega , c , s } } & { \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big [ b \operatorname* { m a x } \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \} + h \operatorname* { m a x } \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \} \Big ] + \sum _ { i = 1 } ^ { D } \frac { \omega [ i ] ^ { 2 } } { 2 \gamma s [ i ] } } \\ { \mathrm { s . t . } } & { ( \omega , c , s ) \in \mathbb { R } ^ { D } \times \mathbb { R } \times [ 0 , 1 ] ^ { D } , } \\ & { \displaystyle \sum _ { j \in [ D ] } s [ j ] \leq d . } \end{array}\tag{EC.7}
$$

By fixing s and taking the duality of the remaining optimization problem, we obtain the min-max reformulation of the continuous relaxation:

$$
\operatorname* { m i n } _ { \substack { s \in [ 0 , 1 ] ^ { D } , \atop \sum _ { j \in [ D ] } s [ j ] \leq d } } \left\{ g ( s ) = \operatorname* { m a x } _ { \substack { z [ i ] \in [ - b , h ] , i \in [ n ] , \sum _ { i = 1 } ^ { n } z [ i ] = 0 } } - z ^ { \top } \left( \frac { \gamma } { 2 n ^ { 2 } } X \mathrm { D i a g } ( s ) X ^ { \top } \right) z - \frac { 1 } { n } z ^ { \top } y \right\} .\tag{EC.8}
$$

It is also clear that optval(3) = min $\{ g ( s ) : s \in \{ 0 , 1 \} ^ { D } , \sum _ { j \in [ D ] } s [ j ] \leq d \}$ . By definition,

$$
F ( { \tilde { \omega } } , { \tilde { c } } ) - \mathrm { o p t v a l } ( 3 ) \leq F ( { \tilde { \omega } } , { \tilde { c } } ) - g ( { \widehat { s } } ) = g ( { \widehat { s } } ) - g ( { \widehat { s } } ) ,\tag{EC.9}
$$

where the first inequality is because $\widehat { s }$ is the optimal solution obtained from continuous relaxation, and the second inequality is by strong duality.

Next, we provide the upper bound on $g ( { \widetilde { s } } ) - g ( { \widehat { s } } )$ . Define the inner objective of $g ( s )$ as

$$
{ \mathfrak { g } } ( s , z ) : = - z ^ { \top } \left( { \frac { \gamma } { 2 n ^ { 2 } } } X { \mathrm { D i a g } } ( s ) X ^ { \top } \right) z - { \frac { 1 } { n } } z ^ { \top } y .
$$

Let $\tilde { z }$ be the maximizer of the inner problem of $g ( \tilde { s } )$ , then

$$
\begin{array} { r l } & { g ( \widetilde s ) - g ( \widetilde s ) \le \mathfrak { g } ( \widetilde s , \widetilde z ) - \mathfrak { g } ( \widetilde s , \widetilde z ) } \\ & { \qquad = - \displaystyle \frac { \gamma } { 2 n ^ { 2 } } \widetilde { z } ^ { \top } \left( X \mathrm { D i a g } ( \widetilde s - \widehat s ) X ^ { \top } \right) \widetilde { z } } \\ & { \qquad \le \displaystyle \frac { \gamma } { 2 n ^ { 2 } } \| \widetilde { z } \| _ { 2 } ^ { 2 } \lambda _ { \operatorname* { m a x } } \Big ( X \mathrm { D i a g } ( \widehat s - \widetilde s ) X ^ { \top } \Big ) } \\ & { \qquad \le \displaystyle \frac { \gamma \left( b \vee h \right) ^ { 2 } } { 2 n } \lambda _ { \operatorname* { m a x } } \Big ( X \mathrm { D i a g } ( \widehat s - \widetilde s ) X ^ { \top } \Big ) , } \end{array}\tag{EC.10}
$$

where the first inequality is because z˜ is a feasible solution (but perhaps suboptimal) to the inner problem of $g ( \widehat { s } )$ , the second inequality is based on the Cauchy-Schwarz inequality, and the last one is because $\| \tilde { z } \| _ { 2 } ^ { 2 } \leq n ( b \vee h ) ^ { 2 }$ . It sufices to build the concentration bound on $\begin{array} { r } { \lambda _ { \operatorname* { m a x } } \Big ( X \mathrm { D i a g } ( \widehat { s } - \widetilde { s } ) X ^ { \top } \Big ) = \lambda _ { \operatorname* { m a x } } \Big ( \sum _ { i \in [ D ] } R _ { i } \Big ) } \end{array}$ with $R _ { i } = ( \widehat s [ i ] - \widetilde s [ i ] ) X _ { : , i } X _ { : , i } ^ { \top }$ . Following the similar argument as in [38, Proof of Lemma 3], it can be shown that $\mathbb { E } [ R _ { i } ] = 0 , \lambda _ { \operatorname* { m a x } } ( R _ { i } ) \leq \theta _ { 1 }$ and $\begin{array} { r } { \| \sum _ { i \in [ D ] } \mathbb { E } [ R _ { i } ^ { 2 } ] \| _ { 2 } \leq \theta _ { 1 } \theta _ { d } } \end{array}$ . Applying the matrix concentration bound in [34, Theorem 1.4] yields

$$
\mathbb { P } \bigg \{ \lambda _ { \operatorname* { m a x } } \bigg ( X \mathrm { D i a g } ( \widehat { s } - \widetilde { s } ) X ^ { \top } \bigg ) \geq t \bigg \} \leq n \exp \left( - \frac { t ^ { 2 } } { 2 \theta _ { 1 } \theta _ { d } + 2 / 3 \theta _ { 1 } t } \right) .\tag{EC.11}
$$

Taking $\begin{array} { r } { t = \frac { 2 } { 3 } \theta _ { 1 } \log ( 2 n / \delta ) + \sqrt { 2 \theta _ { 1 } \theta _ { d } \log ( 2 n / \delta ) } } \end{array}$ ensures the right-hand-side of (EC.11) being bounded by $\delta / 2$ . Combining the inequalities in (EC.9), (EC.10), and (EC.11) yields

$$
\begin{array} { r l } & { \quad \mathbb { P } \left\{ F ( \tilde { \omega } , \tilde { c } ) - \mathrm { o p t v a l } ( 3 ) \leq \frac { \gamma ( b \vee h ) ^ { 2 } } { 2 n } \cdot \left( \frac { 2 } { 3 } \theta _ { 1 } \log ( 2 n / \delta ) + \sqrt { 2 \theta _ { 1 } \theta _ { d } \log ( 2 n / \delta ) } \right) \right\} } \\ & { \geq \mathbb { P } \left\{ \lambda _ { \operatorname* { m a x } } \Big ( X \mathrm { D i a g } ( \hat { s } - \tilde { s } ) X ^ { \top } \Big ) \leq \frac { 2 } { 3 } \theta _ { 1 } \log ( 2 n / \delta ) + \sqrt { 2 \theta _ { 1 } \theta _ { d } \log ( 2 n / \delta ) } \right\} } \\ & { \geq 1 - \frac { \delta } { 2 } . } \end{array}
$$

The choice of $\gamma$ in the main statement of Theorem 2 further ensures that

$$
\mathbb { P } \left\{ F ( \tilde { \omega } , \tilde { c } ) - \mathrm { o p t v a l } ( 3 ) \leq \epsilon \right\} \geq 1 - \delta / 2 .\tag{EC.12}
$$

Applying the union bound of the concentration results in (EC.5) and (EC.12) gives the desired result.

## EC.4. Proof of Theorem 4

Let us define the empirical cost

$$
\mathcal { R } _ { n } ( \omega , c ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big [ b \cdot \operatorname* { m a x } \{ 0 , y _ { i } - \omega ^ { \top } x _ { i } - c \} + h \cdot \operatorname* { m a x } \{ 0 , \omega ^ { \top } x _ { i } + c - y _ { i } \} \Big ] .
$$

and the population cost

$$
\begin{array} { r } { \mathcal { R } ( \omega , c ) = \mathbb { E } _ { ( X , Y ) } \left[ b \cdot \operatorname* { m a x } \{ 0 , Y - \omega ^ { \top } X - c \} + h \cdot \operatorname* { m a x } \{ 0 , \omega ^ { \top } X + c - Y \} \right] . } \end{array}
$$

We rely on the following two technical lemmas that are helpful to show the proof of Theorem 4.

Lemma EC.1 (Global Risk Lower Bound). Under Assumption $^ { 1 , }$ for any $\Delta : = ( u ^ { \top } , t ) ^ { \top } \in \mathbb { R } ^ { D + 1 }$ with $\| u \| _ { 0 } \leq 2 d ,$ , it holds that

$$
\mathcal { R } ( \omega ^ { * } + u , c ^ { * } + t ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) \geq \frac { ( b + h ) \underline { { f } } \kappa r _ { 0 } } { 2 ( r _ { 0 } + K \| \Delta \| _ { 2 } ) } \cdot \| \Delta \| _ { 2 } ^ { 2 } .
$$

Proof. By definition,

$$
\begin{array} { r l } & { \quad \mathcal { R } ( \omega ^ { * } + u , c ^ { * } + t ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) } \\ & { = ( b + h ) \mathbb { E } _ { ( X , Y ) } \left[ \rho _ { \tau } ( Y - ( \omega ^ { * } + u ) ^ { \top } X - ( c ^ { * } + t ) ) - \rho _ { \tau } ( Y - ( \omega ^ { * } ) ^ { \top } X - c ^ { * } ) \right] } \\ & { = ( b + h ) \mathbb { E } _ { ( \varepsilon , X ) } \left[ \rho _ { \tau } ( \varepsilon - \Delta ^ { \top } \widetilde X ) - \rho _ { \tau } ( \varepsilon ) \right] , } \end{array}
$$

where the first equality is by the definition of quantile function $\rho _ { \tau } ( u ) = u ( \tau - 1 _ { \{ u < 0 \} } )$ , and the second one is based on the model assumption (7) and the expressions that $\widetilde { X } = ( X ^ { \top } , 1 ) ^ { \top } , \Delta =$ $( u ^ { \top } , t ) ^ { \top }$ . Based on the Knight’s identity [22], it holds that

$$
\begin{array} { r l } & { \mathbb { E } _ { ( \varepsilon , X ) } \left[ \rho _ { \tau } ( \varepsilon - \Delta ^ { \top } \widetilde X ) - \rho _ { \tau } ( \varepsilon ) \right] } \\ & { = \mathbb { E } _ { ( \varepsilon , X ) } \left[ - \Delta ^ { \top } \widetilde X ( \tau - \mathbf { 1 } _ { \left\{ \varepsilon < 0 \right\} } ) + \displaystyle \int _ { 0 } ^ { \Delta ^ { \top } \widetilde X } \left( \mathbf { 1 } _ { \left\{ \varepsilon \leq s \right\} } - \mathbf { 1 } _ { \left\{ \varepsilon < 0 \right\} } \right) \mathrm { d } s \right] } \\ & { = \mathbb { E } _ { ( \varepsilon , X ) } \left[ \displaystyle \int _ { 0 } ^ { \Delta ^ { \top } \widetilde X } \left( \mathbf { 1 } _ { \left\{ \varepsilon \leq s \right\} } - \mathbf { 1 } _ { \left\{ \varepsilon < 0 \right\} } \right) \mathrm { d } s \right] = \mathbb { E } _ { X } \left[ \displaystyle \int _ { 0 } ^ { \Delta ^ { \top } \widetilde X } \left( F _ { \varepsilon } ( s ) - F _ { \varepsilon } ( 0 ) \right) \mathrm { d } s \right] , } \end{array}\tag{EC.13}
$$

where the second and last equalities are due to the assumption that ε and X are independent together with the definition of the cumulative distribution function (cdf) $F _ { \varepsilon } ( s ) : = \mathbb { P } ( \varepsilon \leq s )$

We provide lower bound on $\begin{array} { r } { \int _ { 0 } ^ { \Delta ^ { \top } \widetilde { X } } \Big ( F _ { \varepsilon } ( s ) - F _ { \varepsilon } ( 0 ) \Big ) } \end{array}$ ds as follows. When $| \Delta ^ { \top } \widetilde { X } | \le r _ { 0 }$ , by Assumption 1(III),

$$
\int _ { 0 } ^ { \Delta ^ { \top } \widetilde { X } } \Big ( F _ { \varepsilon } ( s ) - F _ { \varepsilon } ( 0 ) \Big ) \mathrm { d } s \geq \int _ { 0 } ^ { | \Delta ^ { \top } \widetilde { X } | } \underline { { f } } s \mathrm { d } s = \frac { \underline { { f } } ( \Delta ^ { \top } \widetilde { X } ) ^ { 2 } } { 2 } ,
$$

where the inequality is because $\begin{array} { r } { F _ { \varepsilon } ( s ) - F _ { \varepsilon } ( 0 ) = \int _ { 0 } ^ { s } f _ { \varepsilon } ( t ) \mathrm { d } t \geq \int _ { 0 } ^ { s } \underline { { f } } \mathrm { d } t = \underline { { f } } s } \end{array}$ for $s \geq 0$ . When $\Delta ^ { \top } \widetilde { X } > r _ { 0 } > 0$ , it holds that

$$
\begin{array} { r l } & { \displaystyle \int _ { 0 } ^ { \Delta ^ { \top } \widetilde { X } } \Big ( F _ { \varepsilon } ( s ) - F _ { \varepsilon } ( 0 ) \Big ) \mathrm { d } s \geq \int _ { 0 } ^ { r _ { 0 } } \Big ( F _ { \varepsilon } ( s ) - F _ { \varepsilon } ( 0 ) \Big ) \mathrm { d } s + \int _ { r _ { 0 } } ^ { \Delta ^ { \top } \widetilde { X } } \Big ( F _ { \varepsilon } ( r _ { 0 } ) - F _ { \varepsilon } ( 0 ) \Big ) \mathrm { d } s } \\ & { \geq \frac { f r _ { 0 } ^ { 2 } } { 2 } + ( \Delta ^ { \top } \widetilde { X } - r _ { 0 } ) \underline { f } r _ { 0 } \geq \frac { f r _ { 0 } \Delta ^ { \top } \widetilde { X } } { 2 } , } \end{array}
$$

where the first inequality is by separating the integral on $[ 0 , \Delta ^ { \top } \widetilde X ]$ into two parts and by the monotonicity of the cdf, the second inequality is by Assumption 1(III), and the last one is by the relation $\Delta ^ { \top } \widetilde { X } > r _ { 0 }$ . It can be shown using the similar argument that $\int _ { 0 } ^ { \Delta ^ { \top } \widetilde { X } } \Big ( F _ { \varepsilon } ( s ) -$ $\begin{array} { r } { F _ { \varepsilon } ( 0 ) \Big ) \mathrm { d } s \geq - \underline { { f } } r _ { 0 } \Delta ^ { \top } \widetilde { X } / 2 } \end{array}$ for $\Delta ^ { \top } \widetilde { X } < - r _ { 0 } < 0$ . Combining these cases yields the global lower bound

$$
\int _ { 0 } ^ { \Delta ^ { \top } \widetilde { X } } \Big ( F _ { \varepsilon } ( s ) - F _ { \varepsilon } ( 0 ) \Big ) \mathrm { d } s \geq \underline { { f } } | \Delta ^ { \top } \widetilde { X } | / 2 \cdot \operatorname* { m i n } \{ r _ { 0 } , | \Delta ^ { \top } \widetilde { X } | \} \geq \frac { \underline { { f } } r _ { 0 } ( \Delta ^ { \top } \widetilde { X } ) ^ { 2 } } { 2 ( r _ { 0 } + | \Delta ^ { \top } \widetilde { X } | ) } .
$$

Substituting this lower bound to (EC.13) yields

$$
\begin{array} { r l } & { \quad \mathcal { R } ( \omega ^ { * } + u , c ^ { * } + t ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) } \\ & { \geq \mathbb { E } _ { X } \left[ \frac { ( b + h ) \underline { { f } } r _ { 0 } ( \Delta ^ { \top } \widetilde { X } ) ^ { 2 } } { 2 ( r _ { 0 } + | \Delta ^ { \top } \widetilde { X } | ) } \right] \geq \mathbb { E } _ { X } \left[ \frac { ( b + h ) \underline { { f } } r _ { 0 } ( \Delta ^ { \top } \widetilde { X } ) ^ { 2 } } { 2 ( r _ { 0 } + \| \Delta \| _ { 2 } K ) } \right] \geq \frac { ( b + h ) \underline { { f } } r _ { 0 } \kappa \| \Delta \| _ { 2 } ^ { 2 } } { 2 ( r _ { 0 } + \| \Delta \| _ { 2 } K ) } , } \end{array}
$$

where the second and last inequalities leverage Assumptions 1(IV) and 1(V), respectively. □

Lemma EC.2 (Uniform Concentration Bound on Risk). Under Assumption 1, there exists a constant $C _ { 0 }$ depending on $b , h , K$ such that, with probability at least $1 - \delta ,$ simultaneously for every $\Delta : = ( u ^ { \top } , t ) ^ { \top } \in \mathbb { R } ^ { D + 1 }$ with $\| u \| _ { 0 } \leq 2 d ,$ it holds that

$$
\begin{array} { r l } & { \left| \left( \mathcal { R } _ { n } ( \omega ^ { * } + u , c ^ { * } + t ) - \mathcal { R } _ { n } ( \omega ^ { * } , c ^ { * } ) \right) - \left( \mathcal { R } ( \omega ^ { * } + u , c ^ { * } + t ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) \right) \right| } \\ & { \qquad \leq C _ { 0 } \| \Delta \| _ { 2 } \cdot \sqrt { \frac { d \log \left( e D / d \right) + 1 + \log \left( 2 / \delta \right) } { n } } . } \end{array}
$$

Proof. The inequality for $\Delta = 0$ holds trivially. We focus on the case where $\Delta \neq 0$ instead. Define the function $\ell ( y , z ) = b \operatorname* { m a x } \{ 0 , y - z \} + h \operatorname* { m a x } \{ 0 , z - y \}$ , and the function class

$$
\mathcal { H } = \left\{ ( x , y ) \mapsto \frac { \ell ( y , x ^ { \top } ( \omega ^ { * } + u ) + c ^ { * } + t ) - \ell ( y , x ^ { \top } \omega ^ { * } + c ^ { * } ) } { \| \Delta \| _ { 2 } } : \Delta \neq 0 , \Delta : = ( u ^ { \top } , t ) ^ { \top } , \| u \| _ { 0 } \leq 2 d \right\} .
$$

It sufices to provide concentration bound on

$$
\operatorname* { s u p } _ { h \in \mathcal { H } } \Big | \mathbb { E } _ { X , Y } \big [ h ( X , Y ) \big ] - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } h ( x _ { i } , y _ { i } ) \Big | ,
$$

which can be shown based on the covering number argument.

Each element in the function class H can be indexed with two terms: $r : = \| \Delta \| _ { 2 }$ and $v : =$ $\frac { \Delta } { \| \Delta \| _ { 2 } }$ . Hence, we reformulate

$$
\mathcal { H } = \left\{ \begin{array} { l l } { ( x , y ) \mapsto h _ { r , v } ( x , y ) : = \frac { \ell ( y , x ^ { \top } ( \omega ^ { * } + r v _ { u } ) + c ^ { * } + r v _ { t } ) - \ell ( y , x ^ { \top } \omega ^ { * } + c ^ { * } ) } { r } , } \\ { \hfill } \\ { r > 0 , v = ( v _ { u } ^ { \top } , v _ { t } ) ^ { \top } \in \mathbb { R } ^ { D + 1 } , \| v \| _ { 2 } = 1 , \| v _ { u } \| _ { 0 } \leq 2 d } \end{array} \right\} .
$$

Define the empirical Rademacher complexity

$$
\widehat { \mathfrak { R } } _ { n } ( \mathcal { H } ) : = \mathbb { E } _ { \sigma _ { i } , i \in [ n ] } \left[ \operatorname* { s u p } _ { h \in \mathcal { H } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sigma _ { i } h ( x _ { i } , y _ { i } ) \right| \Big | ( x _ { i } , y _ { i } ) , i \in [ n ] \right] ,
$$

where $\sigma _ { i } , i \in [ n ]$ are i.i.d. Rademacher random variables. We provide upper bound on $\widehat { \mathfrak { R } } _ { n } ( \mathscr { H } )$ using the following procedure.

• For fixed $v \in \mathbb { R } ^ { D + 1 }$ , denote the subset of H as

$$
\mathcal { H } _ { v } = \Big \{ h _ { r , v } : r > 0 \Big \} .
$$

Define the empirical $L _ { 2 }$ pseudo-metric as

$$
\mathbf { d } _ { n } ( f , g ) = \left. \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( f ( x _ { i } , y _ { i } ) - g ( x _ { i } , y _ { i } ) \right) ^ { 2 } \right. ^ { 1 / 2 } .
$$

Let $r _ { 1 } , \ldots , r _ { M }$ with $0 < r _ { 1 } < \cdots < r _ { M }$ index an δ-packing of $\mathcal { H } _ { v }$ under the metric $\mathbf { d } _ { n } ( \cdot )$ Therefore,

$$
( M - 1 ) \delta ^ { 2 } < \sum _ { j = 1 } ^ { M - 1 } { \bf d } _ { n } ( h _ { r _ { j } , v } , h _ { r _ { j + 1 } , v } ) ^ { 2 } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { M - 1 } \Big ( h _ { r _ { j } , v } ( x _ { i } , y _ { i } ) - h _ { r _ { j + 1 } , v } ( x _ { i } , y _ { i } ) \Big ) ^ { 2 } .
$$

According to the formulation of $h _ { r , v }$ and by the convexity of $\ell ( y , \cdot )$ , it is easy to show that $h _ { r , v } ( x , y )$ is non-decreasing in $r ,$ and therefore

$$
\begin{array} { l } { \displaystyle { ( M - 1 ) \delta ^ { 2 } < \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { M - 1 } \left( h _ { r _ { j } , v } ( x _ { i } , y _ { i } ) - h _ { r _ { j + 1 } , v } ( x _ { i } , y _ { i } ) \right) ^ { 2 } } } \\ { \displaystyle \qquad \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \sum _ { j = 1 } ^ { M - 1 } \left( h _ { r _ { j + 1 } , v } ( x _ { i } , y _ { i } ) - h _ { r _ { j } , v } ( x _ { i } , y _ { i } ) \right) \right] ^ { 2 } } \\ { \displaystyle \qquad = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( h _ { r _ { M } , v } ( x _ { i } , y _ { i } ) - h _ { r _ { 1 } , v } ( x _ { i } , y _ { i } ) \right) ^ { 2 } \leq 4 ( b \vee h ) ^ { 2 } K ^ { 2 } , } \end{array}
$$

where the last inequality uses the fact that for any $( x , y )$ and $r > 0$

$$
\left| h _ { r , v } ( x , y ) \right| \leq ( b \vee h ) \cdot \frac { | x ^ { \top } ( r v _ { u } ) + r v _ { t } | } { r } \leq ( b \vee h ) \| ( x ^ { \top } , 1 ) ^ { \top } \| _ { 2 } \| v \| _ { 2 } \leq ( b \vee h ) K .
$$

The packing number ${ \cal M } \le 1 + 4 ( b \vee h ) ^ { 2 } K ^ { 2 } / \delta ^ { 2 }$ . Therefore, the covering number of $\mathcal { H } _ { \ i }$ under $\mathbf { d } _ { n } ( \cdot )$ is bounded:

$$
\begin{array} { r } { \mathcal { N } \Big ( \delta , \mathcal { H } _ { v } , \mathbf { d } _ { n } \Big ) \leq 1 + 4 ( b \vee h ) ^ { 2 } K ^ { 2 } / \delta ^ { 2 } . } \end{array}
$$

• Next, we study the covering number of the set

$$
\mathcal { V } = \Bigl \{ v = ( v _ { u } ^ { \top } , v _ { t } ) ^ { \top } \in \mathbb { R } ^ { D + 1 } : \ \| v \| _ { 2 } = 1 , \| v _ { u } \| _ { 0 } \leq 2 d \Bigr \} .
$$

For fixed support $S \subseteq [ D ]$ with $| S | \leq 2 d ,$ , the δ-covering number of $\mathcal { V } _ { J } = \Big \{ v = ( v _ { u } ^ { \top } , v _ { t } ) ^ { \top } \in$ $\mathbb { R } ^ { D + 1 } \colon \| v \| _ { 2 } = 1 , \operatorname { s u p p } ( v _ { u } ) = J \bigg \}$ under $\| \cdot \| _ { 2 }$ is at most $( 1 + 2 / \delta ) ^ { 2 d + 1 }$ . By taking the union of all sparse supports, the covering number

$$
\mathcal { N } ( \delta , \mathcal { V } , \| \cdot \| _ { 2 } ) \leq \left( \sum _ { k = 0 } ^ { \operatorname* { m i n } \{ 2 d , D \} } { \binom { D } { k } } \right) ( 1 + 2 / \delta ) ^ { 2 d + 1 } .
$$

• For fixed $r > 0$ and any two unit vectors $v = ( v _ { u } , v _ { t } ) , v ^ { \prime } = ( v _ { u } ^ { \prime } , v _ { t } ^ { \prime } )$ , it holds that

$$
\Big | h _ { r , v } ( x , y ) - h _ { r , v ^ { \prime } } ( x , y ) \Big | \leq ( b \vee h ) \cdot \frac { | x ^ { \top } ( r v _ { u } - r v _ { u } ^ { \prime } ) + r v _ { t } - r v _ { t } ^ { \prime } | } { r } \leq ( b \vee h ) K \| v - v ^ { \prime } \| _ { 2 } .
$$

For any $h _ { r , v } \in \mathcal { H }$ , we select $v ^ { \prime }$ in the δ-cover of V and $h _ { r ^ { \prime } , v ^ { \prime } }$ in the δ-cover of $\mathcal { H } _ { v ^ { \prime } }$ . It follows that

$$
\begin{array} { l } { \displaystyle \mathbf { d } _ { n } \big ( h _ { r , v } , h _ { r ^ { \prime } , v ^ { \prime } } \big ) = \Bigg [ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Big ( h _ { r , v } ( x _ { i } , y _ { i } ) - h _ { r ^ { \prime } , v ^ { \prime } } ( x _ { i } , y _ { i } ) \Big ) ^ { 2 } \Bigg ] ^ { 1 / 2 } } \\ { \displaystyle \leq \Bigg [ \frac { 2 } { n } \sum _ { i = 1 } ^ { n } \Big ( h _ { r , v } ( x _ { i } , y _ { i } ) - h _ { r , v ^ { \prime } } ( x _ { i } , y _ { i } ) \Big ) ^ { 2 } + \frac { 2 } { n } \sum _ { i = 1 } ^ { n } \Big ( h _ { r ^ { \prime } , v ^ { \prime } } ( x _ { i } , y _ { i } ) - h _ { r , v ^ { \prime } } ( x _ { i } , y _ { i } ) \Big ) ^ { 2 } \Bigg ] ^ { 1 / 2 } } \\ { \displaystyle \leq \big [ 2 ( b \vee h ) ^ { 2 } K ^ { 2 } \delta ^ { 2 } + 2 \delta ^ { 2 } \big ] ^ { 1 / 2 } = ( 2 + 2 ( b \vee h ) ^ { 2 } K ^ { 2 } ) ^ { 1 / 2 } \delta . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { \mathcal { N } \Big ( ( 2 + 2 ( b \vee h ) ^ { 2 } K ^ { 2 } ) ^ { 1 / 2 } \delta , \mathcal { H } , \mathbf { d } _ { n } \Big ) \leq \mathcal { N } \Big ( \delta , \mathcal { H } _ { v } , \mathbf { d } _ { n } \Big ) \cdot \mathcal { N } ( \delta , \mathcal { V } , \| \cdot \| _ { 2 } ) . } \end{array}
$$

It further implies that

$$
\begin{array} { l } { \log \mathcal { N } \Big ( \delta , \mathcal { H } , \mathbf { d } _ { n } \Big ) \leq \log \left[ 1 + \displaystyle \frac { 8 ( b \vee h ) ^ { 2 } K ^ { 2 } ( 1 + ( b \vee h ) ^ { 2 } K ^ { 2 } ) } { \delta ^ { 2 } } \right] + \log \left( \sum _ { k = 0 } ^ { \operatorname* { m i n } \{ 2 d , D \} } \binom { D } { k } \right) } \\ { \qquad + ( 2 d + 1 ) \log \left[ 1 + \displaystyle \frac { 2 \sqrt { 2 } \big ( 1 + ( b \vee h ) ^ { 2 } K ^ { 2 } \big ) ^ { 1 / 2 } } { \delta } \right] } \\ { \leq C _ { 1 } \cdot \left[ d \log \left( \displaystyle \frac { e D } { d } \right) + 1 + d \log \left( \displaystyle \frac { 1 } { \delta } \right) \right] , } \end{array}
$$

where $C _ { 1 }$ is a constant depending on b, h, K. As each element in $\mathcal { H }$ is uniformly bounded by $( b \lor h ) K$ , by Dudley’s entropy integral bound [15], the Rademacher complexity

$$
\begin{array} { r l r } {  { \mathfrak { R } _ { n } ( \mathcal { H } ) \le \frac { C _ { 2 } } { \sqrt { n } } \int _ { 0 } ^ { ( b \vee h ) K } \sqrt { \log \mathcal { N } \Big ( \delta , \mathcal { H } , \mathbf { d } _ { n } \Big ) } \mathrm { d } \delta } } \\ & { } & { \leq C _ { 3 } \sqrt { \frac { d \log \big ( \frac { e D } { d } \big ) + 1 } { n } } , } \end{array}
$$

where $C _ { 2 } , C _ { 3 }$ are some constants depending on $b , h , K$

Based on the symmetrization and McDiarmid’s inequalities, it holds with probability at least $1 - \delta$ that

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { h \in \mathcal { H } } \bigg | \frac { 1 } { n } \sum _ { i = 1 } ^ { n } h ( x _ { i } , y _ { i } ) - \mathbb { E } [ h ( X , Y ) ] \bigg | \leq C _ { 4 } \sqrt { \frac { d \log \big ( \frac { e D } { d } \big ) + 1 } { n } } + ( b \vee h ) K \sqrt { \frac { 2 \log ( 2 / \delta ) } { n } } } & { } \\ { \leq C _ { 0 } \sqrt { \frac { d \log ( e D / d ) + 1 + \log ( 2 / \delta ) } { n } } , } & { } \end{array}
$$

where $C _ { 0 } , C _ { 4 }$ are some constants depending on b, h, K. Re-arranging the inequality above yields the desired result. □

Finally, we provide the bound on the residual error

$$
a _ { n } : = \left\| \left( \begin{array} { c c } { { \widehat { \omega } _ { n } - \omega ^ { * } } } \\ { { \widehat { c } _ { n } - c ^ { * } } } \end{array} \right) \right\| _ { 2 } .
$$

Proof of Theorem 4. As $\begin{array} { r } { \| \widehat { \omega } _ { n } - \omega ^ { * } \| _ { 0 } \leq 2 d . } \end{array}$ , by Lemma EC.1, we obtain the lower bound

$$
\mathcal { R } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) \geq \frac { ( b + h ) \underline { { f } } \kappa r _ { 0 } } { 2 ( r _ { 0 } + K a _ { n } ) } \cdot a _ { n } ^ { 2 } .\tag{EC.14}
$$

Next, we derive its upper bound. By the definition of $( \widehat { \omega } _ { n } , \widehat { c } _ { n } )$ , it holds that

$$
\mathcal { R } _ { n } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) + \frac { 1 } { 2 \gamma _ { n } } \| \widehat { \omega } _ { n } \| _ { 2 } ^ { 2 } \leq \mathcal { R } _ { n } ( \omega ^ { * } , c ^ { * } ) + \frac { 1 } { 2 \gamma _ { n } } \| \omega ^ { * } \| _ { 2 } ^ { 2 } .
$$

Leveraging this relation and re-arranging the terms yields

$$
\begin{array} { r l } & { \quad \mathcal { R } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) } \\ & { = \Big [ \mathcal { R } _ { n } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } _ { n } ( \omega ^ { * } , c ^ { * } ) \Big ] - \Big [ \Big ( \mathcal { R } _ { n } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } _ { n } ( \omega ^ { * } , c ^ { * } ) \Big ) - \Big ( \mathcal { R } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) \Big ) \Big ] } \\ & { \leq \frac { 1 } { 2 \gamma _ { n } } \Big [ \| \omega ^ { * } \| _ { 2 } ^ { 2 } - \| \widehat { \omega } _ { n } \| _ { 2 } ^ { 2 } \Big ] - \Big [ \Big ( \mathcal { R } _ { n } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } _ { n } ( \omega ^ { * } , c ^ { * } ) \Big ) - \Big ( \mathcal { R } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) \Big ) \Big ] . } \end{array}
$$

The first term above can be further bounded as follows:

$$
\frac { 1 } { 2 \gamma _ { n } } \Big [ \| \omega ^ { * } \| _ { 2 } ^ { 2 } - \| \widehat { \omega } _ { n } \| _ { 2 } ^ { 2 } \Big ] \leq \frac { \langle \omega ^ { * } , \omega ^ { * } - \widehat { \omega } _ { n } \rangle } { \gamma _ { n } } \leq \frac { \| \omega ^ { * } \| _ { 2 } a _ { n } } { \gamma _ { n } } .
$$

By Lemma EC.2, with probability at least $1 - \delta _ { \colon }$ , the second term above can be bounded as

$$
C _ { 0 } a _ { n } \sqrt { \frac { d \log ( e D / d ) + 1 + \log ( 2 / \delta ) } { n } } .
$$

Combining these two relations implies the following relation holds with probability at least $1 - \delta \colon$

$$
\mathcal { R } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) \leq a _ { n } \cdot \left( \frac { \| \omega ^ { * } \| _ { 2 } } { \gamma _ { n } } + C _ { 0 } \sqrt { \frac { d \log ( e D / d ) + 1 + \log ( 2 / \delta ) } { n } } \right) .\tag{EC.15}
$$

Comparing this with (EC.14) yields

$$
\frac { ( b + h ) \underline { { f } } \kappa r _ { 0 } } { 2 ( r _ { 0 } + K \underline { { a } } _ { n } ) } \cdot a _ { n } ^ { 2 } \le a _ { n } \cdot \left( \frac { \| \omega ^ { * } \| _ { 2 } } { \gamma _ { n } } + C _ { 0 } \sqrt { \frac { d \log ( e D / d ) + 1 + \log ( 2 / \delta ) } { n } } \right) .
$$

Specify $\textstyle { \frac { 1 } { 2 \gamma _ { n } } } \lesssim { \sqrt { \frac { d \log ( e D / d ) } { n } } }$ , then there exists constant $C _ { 1 }$ depending on $C _ { 0 } , b , h , K , \| \omega ^ { * } \| _ { 2 }$ such that

$$
\frac { ( b + h ) \underline { { f } } \kappa r _ { 0 } } { 2 ( r _ { 0 } + K a _ { n } ) } \cdot a _ { n } \le C _ { 1 } \sqrt { \frac { d \log ( e D / d ) + 1 + \log ( 2 / \delta ) } { n } } .\tag{EC.16}
$$

Assume $n$ is suficiently large such that

$$
C _ { 1 } \sqrt { \frac { d \log ( e D / d ) + 1 + \log ( 2 / \delta ) } { n } } \leq \frac { ( b + h ) \underline { { f } } \kappa r _ { 0 } } { 4 K } ,
$$

then re-arranging (EC.16) yields

$$
\begin{array} { r l } & { a _ { n } \leq \frac { \left[ C _ { 1 } \sqrt { \frac { d \log \left( e D / d \right) + 1 + \log \left( 2 / \delta \right) } { n } } \right] r _ { 0 } } { \frac { ( b + h ) \underline { { f } } \kappa r _ { 0 } } { 2 } - \left[ C _ { 1 } \sqrt { \frac { d \log \left( e D / d \right) + 1 + \log \left( 2 / \delta \right) } { n } } \right] K } } \\ & { \leq \frac { \left[ C _ { 1 } \sqrt { \frac { d \log \left( e D / d \right) + 1 + \log \left( 2 / \delta \right) } { n } } \right] r _ { 0 } } { \frac { ( b + h ) f \kappa r _ { 0 } } { 4 } } = \frac { 4 C _ { 1 } } { \left( b + h \right) \underline { { f } } \kappa } \cdot \sqrt { \frac { d \log \left( e D / d \right) + 1 + \log \left( 2 / \delta \right) } { n } } } \end{array}
$$

holds with probability at least $1 - \delta$

## EC.5. Proof of Corollary 1

By definition,

$$
\operatorname* { m i n } _ { \| \omega \| _ { 0 } \leq d , c } \mathcal { R } ( \omega , c ) = \operatorname* { m i n } _ { \| \omega \| _ { 0 } \leq d , c } \big \{ ( b + h ) \mathbb { E } _ { ( X , Y ) } \left[ \rho _ { \tau } \big ( Y - ( \omega ^ { \top } X + c ) \big ) \right] \big \} .
$$

By the likelihood model, for each $x , ( \omega ^ { * } ) ^ { \top } x + c ^ { * }$ is the τ-quantile of $Y \mid X = x$ , and therefore the optimization problem above takes the optimal solution $( \omega ^ { \ast } , c ^ { \ast } )$ . Recall from the proof of Theorem 4 that for suficiently large $n ,$ with probability at least $1 - \delta _ { : }$ , the following two relations hold:

$$
\begin{array} { r l } & { \mathcal { R } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) \leq a _ { n } C _ { 1 } \cdot \sqrt { \frac { d \log ( e D / d ) + 1 + \log ( 2 / \delta ) } { n } } , } \\ & { \quad \quad \quad \quad a _ { n } \leq \frac { 4 C _ { 1 } } { ( b + h ) f \kappa } \cdot \sqrt { \frac { d \log ( e D / d ) + 1 + \log ( 2 / \delta ) } { n } } . } \end{array}
$$

Combining these two relations yields

$$
\mathcal { R } ( \widehat { \omega } _ { n } , \widehat { c } _ { n } ) - \mathcal { R } ( \omega ^ { * } , c ^ { * } ) \leq \frac { 4 C _ { 1 } ^ { 2 } } { ( b + h ) \underline { { f } } \kappa } \cdot \frac { d \log ( e D / d ) + 1 + \log ( 2 / \delta ) } { n } .
$$

## EC.6. Proof of Corollary 2

Under the event described in Corollary 2, for each $j \in S ^ { * }$ , it holds that

$$
\left| { \widehat { \omega } } _ { n } [ j ] \right| \geq | \omega ^ { * } [ j ] | - | { \widehat { \omega } } _ { n } [ j ] - \omega ^ { * } [ j ] | \geq | \omega _ { j } ^ { * } | - \left\| \left( { \widehat { \omega } } _ { n } - \omega ^ { * } \right) \right\| _ { 2 } > 0 .
$$

Therefore, $S ^ { * } \subseteq \operatorname { s u p p } ( { \widehat { \omega } } _ { n } )$ . On the other hand, by the feasibility of $\widehat { \omega } _ { n }$ , it holds that $| \operatorname { s u p p } ( { \widehat { \omega } } _ { n } ) | \leq d = | S ^ { * } |$ . This fact, together with $S ^ { * } \subseteq \operatorname { s u p p } ( \widehat { \omega } _ { n } )$ implies that $S ^ { * } = \operatorname { s u p p } ( \widehat { \omega } _ { n } )$