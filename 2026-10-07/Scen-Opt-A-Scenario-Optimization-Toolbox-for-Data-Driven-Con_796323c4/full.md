# Scen-Opt: A Scenario Optimization Toolbox for Data-Driven Convex Programming

Ben Wooding1 . Simone Garatti2 . Marco C. Campi³ · Abolfazl Lavaei⁴

Abstract The scenario approach is a well-established statistical framework for data-driven decision-making. In particular, in data-driven optimization, the scenario approach unveils how the problem structure governs out-of-sample generalization, and offers a principled basis for assessing and certifying the reliability of the optimal solution as per constraint satisfaction. Despite its strong theoretical development and wide applicability, no software toolbox has been available to date that enables user-friendly, data-driven convex optimization within the scenario-approach framework. In this paper, we introduce Scen-Opt, an open-source software tool that integrates convex programming with data samples while providing statistical guarantees grounded in scenario theory. Scen-Opt is implemented in Python, supporting data-driven linear, quadratic, and semidefinite programming, and offers a Python-based web application with an intuitive and reactive graphical user interface (GUI) built using modern web technologies. Scen-Opt can be used directly through its online interface or installed locally, accommodating both manual input and data-file uploads (CSV, JSON, TXT, TSV, MAT, Excel, NPY, NPZ, Parquet). Built on a Python backend with a modern JavaScript frontend, Scen-Opt offers a highly user-friendly experience and efficient usability across desktops, laptops, tablets, and mobile devices. In this paper, Scen-Opt is applied to a set of representative benchmarks, demonstrating its practical effectiveness for data-driven convex optimization with guaranteed performance.

## 1 Introduction

The increasing availability of data across scientific and engineering domains has fundamentally reshaped decision-making workflows: rather than depending solely on analytical models, which may be inaccurate, overly conservative, or impractical to obtain, there is growing interest in methods that leverage sampled observations directly to inform decision-making [27]. Data can reveal structure, expose model limitations, and support more adaptive strategies Yet, most data-driven decision procedures offer limited assurances about how the resulting decisions will perform when confronted with new, unseen situations. In many cases, solutions are derived from data but their reliability is assessed only through ad-hoc or post-hoc testing, leaving open the fundamental question of whether such solutions generalize with some robustness beyond the observed sample.

The scenario approach [8, 15, 12] provides a principled answer to this challenge. Originating in convex optimization and now encompassing a broad class of data-driven decision-making schemes, scenario theory establishes rigorous probabilistic guarantees for decisions obtained from independent and identically distributed (i.i.d.) data. By characterizing the intersection between the decision and the observed data through the notion of complexity, scenario theory offers explicit bounds on the probability that a newly observed uncertainty instance will render the decision inappropriate. This risk quantification is entirely data-driven and does not rely on assumptions about the underlying probability distribution, making it suitable for a wide spectrum of applications [9,22].

When applied to convex optimization, scenario theory attains a particularly elegant form. Each data point is treated as a constraint, the complexity is the number of constraints strictly needed to reconstruct the solution (the so called support list), and the risk corresponds to the probability that a new constraint, unseen at design time, is violated by the computed solution. The structural properties of convex optimization ensure that solutions are globally optimal and unique (when the problem is strictly convex), so the connection between complexity, sample size, and generalization becomes especially transparent. These features have driven the widespread adoption of convex optimization techniques across control theory, robotics, machine learning, data science, signal processing, and many other fields.

Original contributions. Motivated by the central challenge of making reliable decisions when only sampled data are available but robust performance guarantees are still required, this paper introduces Scen-Opt a first-of-its-kind tool that leverages scenario theory to deliver user-friendly, data-driven convex optimization with rigorous probabilistic guarantees, even when only limited sampled data are available.

The main features of the proposed tool are the following:

(i) Scen-Opt leverages a Python backend with a modern JavaScript frontend, powered by a suite of state-of-the-art solvers (27 solvers supported at the time of writing) including CVX toolbox [24] and MOSEK¹, and supports three classes of scenario optimization programs (i) linear programs (LP), (ii) quadratic programs (QP) and (iii) semidefinite programs (SDP).

(ii) For each of the mentioned scenario programs, Scen-Opt supports four problem configurations: (i) the robust problem, (ii) the relaxed problem, (iii) the regularized problem, and (iv) the general formulation combining relaxation, and regularization.

(iii) Implemented as a responsive and reactive Python Flask2 web application, Scen-Opt provides an intuitive, user-friendly interface that enables efficient interaction with all scenario-optimization functionalities. Users can access and use the application directly through the web across desktops, laptops, tablets, and mobile devices without any download or installation3.

(iv) In addition, Scen-Opt provides a Docker-based version4 that can be downloaded and executed locally, enabling users to achieve higher performance when leveraging their own computing resources.

(v) The tool supports both manual data entry and data-file uploads for sampled trajectories, providing a user-friendly interface for specifying all required inputs across the different scenario optimization programs. The full program can be uploaded in one-shot via . mat or . json, and results downloaded in the same formats. Scenario-dependent constraints can be provided either symbolically, as expressions in the uncertainty, or numerically, as pre-evaluated per-sample matrices.

Besides describing Scen-Opt, this paper demonstrates its effectiveness through a series of physical benchmarks, spanning all three supported classes of scenario optimization programs and showcasing their corresponding four sub-classes for each program type.

The web-based version of Scen-Opt is accessible at:

## https://scen-opt.woodingben.com

The Scen-Opt source code, accompanied by comprehensive installation and usage instructions for the Docker-based version, can be accessed at⁵:

$$
\mathrm { \ h t t p s { : } / / g i t h u b . c o m / K i g u l i / S c e n { - } O p t }
$$

Organization. The remainder of the paper is structured as follows. Section 2 reviews the theoretical foundations of the scenario approach in a convex optimization setting that underpin the methods and guarantees used throughout this paper. Section 3 presents the convex optimization programs supported by

Scen-Opt, outlining the three program classes and their four scenario-based variants implemented for each. Implementation details of the toolbox, including the underlying algorithms, are provided in Section 4, while the web-based graphical interface is described in Section 5. Comprehensive benchmarking results across all supported problem classes are reported in Section 6, and the paper concludes, together with directions for future development, in Section 7.

Related studies on scenario approach. Over the past two decades, the scenario approach has played a central role in advancing data-driven methods for decision-making under uncertainty. The seminal work in [8] introduced scenario-based convex programming for robust control, showing that semiinfinite constraints can be handled through randomized sampling. This line of work was further strengthened in [10, 11], which established explicit probabilistic bounds on violation and extended the framework to settings with discarded or a-posteriori-violated constraints. These developments form the foundation for a broad class of randomized and scenario-based optimization techniques that have since been generalized and applied across numerous domains.

Numerous contributions to the field have been made by many authors. Methods for convex scenario programs with multiple chance constraints are investigated in [40]. The sample complexity of random convex programs with structured uncertainty is characterized in [43], leading to less conservative bounds when exploiting problem structure. A probabilistic connection between the optimal value of scenario convex programs and that of robust and chance-constrained formulations is established in [33], with extensions to certain mixed-integer nonconvex settings; the results are further applied to infinite-dimensional linear programs in [32]. The relationship between compression learning and the original results in scenario optimization is analyzed in [31], while [30] proposes an intermediate approach to chance-constrained optimization that sits between robust optimization and fully randomized scenario methods. Distributed scenario optimization in uncertain multi-agent framework is studied in [29]. Probabilistic guarantees for convex scenario programs where validation tests are exploited to derive tighter confidence bounds on the violation probability are developed in [41]. Finally, [39,38] study exact feasibility properties of convex scenario programs with discarded constraints, providing conditions under which the tightest feasibility guarantees remain valid despite constraint removal.

A major turning point in the theory of the scenario approach is given by the works [16,22,13,21] which introduce the wait-and-judge philosophy and permit assessing generalization after a solution is obtained, using the size of a suitably defined support subsample as a measure of complexity. This enables tight, instance-specific guarantees for a quite general class of decision problems, broadening the applicability of scenario methods far beyond traditional convex optimization settings.

A further shift to enhance the broadest applicability is the new compressionbased theory in [14| establishes rigorous control over the probability that newly observed data, which alter a compressed representation, providing tight finitesample bounds under minimal assumptions. By showing that the compressedset cardinality consistently estimates this probability in a fully agnostic setting, this abstract framework offers strong support for general observation-driven decision-making and constitutes a foundational tool for modern learning tasks involving uncertainty quantification.

## 2 Foundational Concepts in Scenario (Convex) Optimization

This section reviews the foundational concepts underpinning scenario-based decision making, focusing on the theoretical principles that support datadriven convex optimization with probabilistic guarantees. We summarize the key definitions, assumptions, and central results from the scenario approach literature [21, 22| that form the basis of the methods implemented in Scen-Opt.

We begin by formalizing the class of convex optimization problems considered in this work.

Definition 1 (Convexity) Let $\mathbb { R } ^ { d }$ denote a Euclidean space with d components. A set $C \subseteq \mathbb { R } ^ { d }$ is convex if the line segment between any two points in $C$ lies entirely within $C ;$ that is, for all $x _ { 1 } , x _ { 2 } \in C$ and any $0 \leq a \leq 1$ 2

$$
\begin{array} { r } { a x _ { 1 } + ( 1 - a ) x _ { 2 } \in C . } \end{array}
$$

A function $f :  { \mathbb { R } ^ { d } } \to  { \mathbb { R } }$ is convex if its domain is a convex set and, for all $x , y$ in its domain and any $0 \leq a \leq 1$ 2

$$
f ( a x + ( 1 - a ) y ) \leq a f ( x ) + ( 1 - a ) f ( y ) .
$$

Geometrically, this means that the line segment connecting $( x , f ( x ) )$ and $( y , f ( y ) )$ lies above the graph of $f .$ □

Let $x \in \mathcal { X }$ be a vector of design variables, where $\mathcal { X } \subset \mathbb { R } ^ { d }$ . Let $\delta \in \mathbb { R } ^ { q }$ with q components, be a parameter that describes the environment to which a decision is applied. In this setting, one can formulate the following scenario program:

$$
\operatorname* { m i n } _ { x \in \mathcal { X } , \zeta _ { i } \geq 0 } \ c ( x ) \ + \ \tau \| x - \overline { { x } } \| \ + \ \rho \sum _ { i = 1 } ^ { N } \zeta _ { i }\tag{1a}
$$

$$
\mathrm { s u b j e c t ~ t o : } \quad f ( x , \delta _ { i } ) \leq \zeta _ { i } , \quad i = 1 , \ldots , N .\tag{1b}
$$

The constraints (1b) are expressed as $f ( x , \delta _ { i } ) \leq 0$ , for each data point $\delta _ { i \cdot }$ and we allow both regularization of the solution and relaxation of the constraints. The values $\zeta _ { i }$ are per-constraint scalar slack optimization variables used to relax the problem. By selecting $\zeta _ { i } > 0$ , a scenario constraint $f ( x , \delta _ { i } ) \leq 0$ may be violated, thereby enlarging the feasibility domain for x and creating the possibility of finding a solution with lower cost $c ( x )$ . To control the regret associated with constraint violations, however, the sum of the relaxation parameters, $\sum _ { i } \zeta _ { i } ,$ is also penalized in the objective function $\mathrm { ( 1 a ) }$ A fixed weight parameter $\rho \ge 0$ determines the relative importance assigned to the cost $c ( x )$ and the constraint violations. Regularization is likewise imposed in the objective to encourage the solution to remain near some x, with the fixed weight parameter $\tau \geq 0$ controlling the strength of this regularization. There are special cases of (1) that typically arise as limiting cases, but which we explicitly consider in this work.

Only Constraint Relaxation $( \tau = 0 )$

$$
\begin{array} { c } { { \displaystyle \operatorname* { m i n } _ { x \in \mathcal { X } , \zeta _ { i } \geq 0 } ~ c ( x ) ~ + ~ \rho \sum _ { i = 1 } ^ { N } \zeta _ { i } } } \\ { { \mathrm { s u b j e c t ~ t o } \colon ~ f ( x , \delta _ { i } ) \leq \zeta _ { i } , ~ i = 1 , \ldots , N . } } \end{array}
$$

Only Regularization $( \rho = 0$ and $\zeta _ { i } = 0 , \forall i )$

$$
\begin{array} { r l } & { \underset { x \in \mathcal { X } } { \operatorname* { m i n } } \quad c ( x ) + \tau \| x - \overline { { x } } \| } \\ & { \mathrm { s u b j e c t ~ t o } \mathrm { : } \quad f ( x , \delta _ { i } ) \leq 0 , \quad i = 1 , \hdots , N . } \end{array}
$$

Robust problem $( \tau = 0 , \rho = 0$ and $\zeta _ { i } = 0 , \forall i )$

$$
\begin{array} { c } { \displaystyle \operatorname* { m i n } _ { x \in \mathcal { X } } c ( x ) } \\ { \mathrm { s u b j e c t ~ t o : } \quad f ( x , \delta _ { i } ) \leq 0 , \quad i = 1 , \dots , N . } \end{array}
$$

The user can choose the formulation which is most appropriate for the problem at hand among these four alternatives. Once a formulation has been selected, let $x _ { N } ^ { * }$ denote the x-component of the corresponding optimal solution (in what follows, there is no need to explicitly consider the optimal $\zeta _ { i } ^ { * }$ components, if any). When problem (1) is solved, it yields $x _ { N } ^ { * }$ , for which the cost $c ( x _ { N } ^ { * } )$ can be directly computed.

Assumption 1 We assume that a solution to (1) exists and is unique $f o r$ every N and for every realization of the $\delta _ { i } \mathit { \Pi } _ { s . } ^ { }$ □

Existence is always guaranteed when there is constraint relaxation (for a compact $x ;$ otherwise a small $\rho$ can make the relaxed problem unbounded), while uniqueness is guaranteed whenever there is a strictly convex regularization term with $\tau > 0$ , provided existence holds. The results below apply to $x _ { N } ^ { * }$ regardless of the formulation adopted. The variables x and δ are linked through the concept of appropriateness as in the following definition.

Definition 2 (Appropriateness) The variable x is said to be appropriate for δ if it satisfies the corresponding requirement induced by $\delta ;$ that is, the satisfaction condition associated with δ holds when evaluated at x. □

In robust optimization, x is appropriate for δ if it lies within the region of feasible decisions induced by δ. The set of all values of x that are appropriate for δ is denoted by $\mathcal { X } _ { \delta }$ . We assume access to a list $\big ( \delta _ { 1 } , \dots , \delta _ { N } \big )$ of observations of $\delta ,$ where each $\delta _ { i }$ is referred to as a scenario and is used to inform the decision variable $x .$ By abuse of notation, the same expression is used even when $N = 0$ , in which case the list is interpreted as empty. Formally, $\big ( \delta _ { 1 } , \dots , \delta _ { N } \big )$ is an i.i.d. sample from a probability space $( \varDelta , \mathcal { D } , \mathbb { P } )$ , where $\varDelta$ is the set of all possible uncertainty realizations, D is the associated σ-algebra of measurable subsets of $\varDelta .$ and $\mathbb { P }$ is the underlying probability measure governing the sampling process. Although the scenario approach does not require knowledge of the underlying probability space, a distribution-free paradigm, the resulting generalization guarantees still yield insight into the reliability of the decision.

However, the appropriateness of $x _ { N } ^ { * }$ for a new, unseen scenario δ remains uncertain. This motivates the introduction of the notion of risk, which quantifies the probability that $x _ { N } ^ { * }$ fails to satisfy the requirement induced by a future scenario.

Definition 3 (Risk) The risk $V ( x )$ quantifies the probability that a newly drawn scenario δ renders x inappropriate. For any $x \in \mathcal { X } , V ( x ) = \mathbb { P } \{ \delta \ :$ $f ( x , \delta ) > 0 \}$ □

The fundamental problem is to estimate $V ( x _ { N } ^ { * } )$ , which cannot be computed directly since the probability measure P is unknown. Scenario optimization provides a powerful generalization theory that enables one to bound or estimate $V ( x _ { N } ^ { * } )$ without requiring any additional samples or any knowledge of the underlying probability distribution.

To analyze how the solution evolves as additional scenarios are incorporated, we consider (1) for some general N. This formulation allows us to track how the feasible set and its optimizer change as new scenarios are added, which is central to the generalization properties of scenario optimization. The following concept of complexity plays a central role in scenario-based theory, as it characterizes the minimal subset of scenarios that fully determine the solution.

Definition 4 (Support List and Complexity) Given a list of scenarios $\big ( \delta _ { 1 } , \dots , \delta _ { N } \big )$ , a support list is the sublist of scenarios that include all those for which $f ( x _ { N } ^ { * } , \delta _ { i } ) > 0$ and some of those for which $f ( x _ { N } ^ { * } , \delta _ { i } ) = 0$ , such that:

(i) it allows reconstruction of the solution $x _ { N } ^ { * }$ of the considered problem;

(ii) the sublist is irreducible: no element can be further removed from the sublist without the decision changing.

For a given list $\big ( \delta _ { 1 } , \dots , \delta _ { N } \big )$ , multiple support lists may exist with different cardinalities k. The minimal such cardinality among all support lists is called the complexity and is denoted by $s _ { N } ^ { * }$ □

Note that both $x _ { N } ^ { * }$ and $s _ { N } ^ { * }$ depend on the list $( \delta _ { 1 } , \dots , \delta _ { N } )$ and are therefore random elements defined on the product probability space $( \varDelta ^ { N } , \mathcal { D } ^ { N } , \mathbb { P } ^ { N } )$ 2 where the product measure arises from the independence of scenario draws. Finally, we can state the theorems where $x _ { N } ^ { * }$ is the solution of the considered problem for a known N.

Theorem 1 (Risk Certificate) Let $\epsilon ( k )$ be defined as

$$
\epsilon ( k ) : = 1 - t ( k ) , \quad k = 0 , 1 , \ldots , N - 1 , \ a n d \epsilon ( N ) = 1 ,
$$

where, for a prescribed confidence level $1 - \beta _ { i }$ the scalar $t ( k ) \in ( 0 , 1 )$ is the unique solution of

$$
\frac { \beta } { N } \sum _ { i = k } ^ { N - 1 } \binom { i } { k } t ^ { i - k } - \binom { N } { k } t ^ { N - k } = 0 , \qquad k = 0 , 1 , \dots , N - 1 .
$$

Then, for any probability measure $\mathbb { P } ,$ the following distribution-free risk bound holds:

$$
\mathbb { P } ^ { N } \bigg \{ V ( x _ { N } ^ { * } ) > \epsilon ( s _ { N } ^ { * } ) \bigg \} \le \beta .
$$

In Theorem 1, the parameter $\beta$ is referred to as the confidence level and is typically chosen to be very small, for example $1 0 ^ { - 5 } \mathrm { o r } 1 0 ^ { - 6 }$ . The result asserts that $V ( x _ { N } ^ { * } )$ , the risk that the solution $x _ { N } ^ { * }$ fails to be appropriate for a new out-of-sample scenario, is upper-bounded with confidence $1 - \beta ,$ by $\epsilon ( s _ { N } ^ { * } )$ Here, $\epsilon ( s _ { N } ^ { * } )$ is a known deterministic function evaluated at the complexity $s _ { N } ^ { * }$ of the scenario optimization problem.

Remark 1 The upper bound on the risk provided in Theorem 1 holds for any solution computed by Scen-Opt. In the sequel, we also present a result that yields a corresponding lower bound; however, this requires additional assumptions. While Scen-Opt attempts to detect violations of these assumptions, such checks cannot be performed for out-of-sample scenarios. Consequently, it is ultimately the user's responsibility to assess whether the lower bound remains applicable in cases where Scen-Opt has not flagged a violation. □

We now recall a structural property of scenario programs that will play a central role in the forthcoming theorem.

Property 1 (Non-degeneracy) For any N and with probability one with respect to the generation of the scenarios, there exists a unique support list for $\big ( \delta _ { 1 } , \dots , \delta _ { N } \big )$ □

It is often instructive to illustrate degeneracy concretely, in order to clarify the meaning of non-degeneracy. In any optimization problem, one may identify subsets of support constraints whose removal does not alter the optimal solution. Degeneracy arises when multiple distinct subsets of support constraints can be removed while still preserving the solution. A simple example occurs when a support constraint appears multiple times in the list: each repeated instance may be removed independently without affecting the optimizer. As a result, there exist multiple support lists, each omitting a different repetition, despite all having the same complexity.

Theorem 2 (Two-sided Risk Certificate) Given a confidence parameter $\beta \in ( 0 , 1 )$ , for any $k = 0 , 1 , \ldots , N - 1$ consider the polynomial equation in the variable t

$$
{ \binom { N } { k } } t ^ { N - k } - { \frac { \beta } { 2 N } } \sum _ { i = k } ^ { N - 1 } { \binom { i } { k } } t ^ { i - k } - { \frac { \beta } { 6 N } } \sum _ { i = N + 1 } ^ { 4 N } { \binom { i } { k } } t ^ { i - k } = 0 ,\tag{2a}
$$

and for $k = N$ , consider instead the polynomial equation

$$
1 - \frac { \beta } { 6 N } \sum _ { i = N + 1 } ^ { 4 N } \binom { i } { k } t ^ { i - N } = 0 .\tag{2b}
$$

For any $k = 0 , 1 , \ldots , N - 1$ , (2a) admits exactly two solutions in $[ 0 , + \infty )$ denoted $\underline { { t } } ( k )$ and $\bar { t } ( k )$ with $\underline { { t } } ( k ) \leq \bar { t } ( k )$ . For $k = N , ( \mathrm { 2 b } )$ admits exactly one solution in $\lbrack 0 , + \infty )$ , denoted $\bar { t } ( N )$ , which we deine with $\underline { { t } } ( N ) = 0$ . Let $\underline { { \epsilon } } ( k ) : =$ max $\{ 0 , 1 - \bar { t } ( k ) \}$ and $\overline { { \epsilon } } ( k ) : = 1 - \underline { { t } } ( k )$ , for all $k = 0 , 1 , \ldots , N$ . Under the assumptions of consistency and non-degeneracy, it holds for any uncertainty space ∆ and any probability measure P that:

$$
\begin{array} { r } { \mathbb { P } ^ { N } \{ \underline { { \epsilon } } ( s _ { N } ^ { * } ) \leq V ( x _ { N } ^ { * } ) \leq \overline { { \epsilon } } ( s _ { N } ^ { * } ) \} \geq 1 - \beta . } \end{array}
$$

With this theorem, one can bound the risk associated with scenario optimization for any chosen confidence level and any list of scenarios $\left( \delta _ { 1 } , \dots , \delta _ { N } \right)$ In our tool, we implement this result using the same numerical procedure described in [14, Appendix B|.

## 3 Scenario Optimization Schemes Covered by the Toolbox

In this section, we present the data-driven convex optimization schemes supported by our toolbox Scen-Opt. The problem formulations are intentionally aligned with the classical exposition in [5] to maintain familiarity, and the underlying implementation of Scen-Opt draws inspiration from the design principles established in the CVX toolbox [24].

We present below the implementation of scenario convex optimization for LP, QP, and SDP programs. As outlined in Section 2, Scen-Opt supports four subclasses for each of these optimization types, : robust, relaxation, regularization, and the combined relaxation-regularization formulation. For consistency, we introduce each class using the relaxation-regularization formulation, as it is the most general case, and then indicate how the other subclasses are obtained through appropriate adaptations.

Throughout, we denote the decision variable by $x \in \mathbb { R } ^ { d }$ . Each optimization program incorporates the constraints induced by every scenario $\delta _ { i } \in \mathbb { R } ^ { q }$ in the list $\left( \delta _ { 1 } , \dots , \delta _ { N } \right)$ , together with any additional hard constraints required by the problem structure.

## 3.1 Linear Programming (LP)

We consider the most general case of relaxed-regularized LP as

$$
\operatorname* { m i n } _ { x , \zeta _ { i } \geq 0 } \ c { e ^ { \top } } x + \tau \| x - \bar { x } \| _ { p } + \rho \sum _ { i = 1 } ^ { N } \zeta _ { i }\tag{3a}
$$

$$
\mathrm { s u b j e c t ~ t o : \quad } A ( \delta _ { i } ) x ~ + ~ b ( \delta _ { i } ) \leq \zeta _ { i } , \quad i = 1 , \ldots , N\tag{3b}
$$

$$
G x \ + \ h \leq 0 ,\tag{3c}
$$

where $\tau \geq 0$ and $\rho \geq 0 .$

LPs are so named because the objective function $c ^ { \top } x$ where $c \in \mathbb { R } ^ { d }$ , and all constraint functions in (3b) and (3c) are required to be linear. For arbitrary integers ${ \mathfrak { n } } > 0$ and m $> 0$ , the matrix $G \in \mathbb { R } ^ { \mathfrak { n } \times d }$ and vector $h \in \mathbb { R } ^ { \mathfrak { n } }$ define the hard linear constraints, while the matrix-valued function $\boldsymbol { A } ( \cdot ) \in \mathbb { R } ^ { \mathfrak { m } \times d }$ and vector-valued function $b ( \cdot ) \in \mathbb { R } ^ { \mathfrak { m } }$ specify the linear constraints associated with each scenario $\delta _ { i }$

In what follows, we discuss how the relaxed-regularized LP in (3) can be specialized to obtain each of the other LP subclasses.

Robust LP. The robust $L P$ is obtained by setting $\tau = 0 , \rho = 0 .$ and $\zeta _ { i } =$ 0 for all i, and is the strictest $\mathrm { L P }$ form available in Scen-Opt. Under these restrictions, the cost function reduces to

$$
\operatorname* { m i n } _ { x } c ^ { \top } x
$$

and constraint (3b) collapses to the hard linear constraints $A ( \delta _ { i } ) x + b ( \delta _ { i } ) \leq 0$ Relaxed LP. The relaxed $L P$ is obtained by setting $\tau = 0$ , thereby reducing the objective function to:

$$
\operatorname* { m i n } _ { \boldsymbol { x } , \boldsymbol { \zeta } _ { i } } \quad \boldsymbol { c } ^ { \top } \boldsymbol { x } + \rho \sum _ { i = 1 } ^ { N } \boldsymbol { \zeta } _ { i } ,
$$

while leaving the constraints (3b),(3c) unchanged.

Regularized LP. The regularized $L P$ corresponds to set $\rho = 0$ and $\zeta _ { i } = 0$ for all i. The cost function is given as

$$
\operatorname* { m i n } _ { \boldsymbol { x } } \quad \boldsymbol { c } ^ { \top } \boldsymbol { x } + \boldsymbol { \tau } \| \boldsymbol { x } - \boldsymbol { \bar { x } } \| _ { p } ,
$$

while scenario constraints (3b) become $A ( \delta _ { i } ) x ~ + ~ b ( \delta _ { i } ) \leq 0$

When regularization is present, different choices of norm p are supported by Scen-Opt; the field accepts any order $p \ge 1 \left( e . g . , p = 1 \mathrm { o r } p = 2 \right)$ , inf for the ∞-norm, or fro for the Frobenius norm, which for the vector $x - { \bar { x } }$ coincides with $p = 2$

To illustrate the use of scenario optimization for linear programs, we present a simple yet instructive example.

![](images/9a21c17db67c7858f3960a972dac9def82b1f8eb66c170cf00b0aac5fc8a1be3.jpg)  
Fig. 1: Robust LP: Smallest enclosing interval example.

Example 1 (Smallest Enclosing Interval.) Consider the problem of determining the smallest interval that contains a collection of sampled points $\delta _ { i }$ from a one-dimensional space. This task can be naturally formulated as a robust LP, where the decision variable consists of the center $o ^ { * }$ and half-width $w ^ { * }$ of the interval (cf. Fig. 1), such that $x _ { N } ^ { * } = \left[ o ^ { * } \ w ^ { * } \right]$ In this formulation, the robust LP of the form in (3) is specified by the matrices

$$
{ c } = { \binom { 0 } { 1 } } , \quad { A } ( \delta _ { i } ) = { \binom { - 1 } { 1 } } - 1 { \bigg ] } , \quad { b } ( \delta _ { i } ) = { \binom { \delta _ { i } } { - \delta _ { i } } } ,\tag{4}
$$

and the constraint $A ( \delta _ { i } ) x + b ( \delta _ { i } ) \leq 0$ is imposed for each sampled point (scenario) $\delta _ { i } .$ Since this is a robust LP, we set $\tau = 0 , \rho = 0$ and $\zeta _ { i } = 0$ for all i. The corresponding JSON program file is given in Listing 8 in Appendix B.

Scen-Opt supports LP scenario optimization through the following function in Listing 1, with its parameters described in detail in Appendix C.1. This function solves the LP defined in (3) and returns the optimal solution together with the information required to analyze the associated risk, including the complexity and the number of scenarios N. For diagnostic and analysis purposes, the full set of constraints is returned, along with an indicator specifying whether any degeneracy was detected during the solution process (the full solution can also be downloaded).6

```python
def solve_1p(deltas, A_d, b_d, G, h, c, tau=0.0, x_ref=np.array([0.0]),
rho=0.0, norm_type=2, solver=None, include_slack=None):
return x_out, zeta_out, cost_out, N, complexity, constraints, degeneracy
```  
Listing 1: Function to solve LP.

## 3.2 Quadratic Programming (QP)

We consider the most general case of relaxed-regularized $\mathrm { Q P }$ as

$$
\operatorname* { m i n } _ { x , \zeta _ { i } \geq 0 } \ c ^ { \top } x \ + \ \frac { 1 } { 2 } x ^ { \top } Q x \ + \ \tau \| x - \bar { x } \| _ { p } \ + \ \rho \sum _ { i = 1 } ^ { N } \zeta _ { i }\tag{5a}
$$

$$
\mathrm { s u b j e c t ~ t o : \quad } A ( \delta _ { i } ) x ~ + ~ b ( \delta _ { i } ) \leq \zeta _ { i } , \quad i = 1 , \ldots , N\tag{5b}
$$

$$
G x \ + \ h \leq 0 ,\tag{5c}
$$

where $\tau \geq 0$ and $\rho \geq 0 .$

QPs are so named as their objective includes the quadratic term ${ \frac { 1 } { 2 } } x ^ { \top } Q x$ where $Q \in \mathbb { R } ^ { d \times d }$ is a symmetric positive-semidefinite matrix, besides the linear term $c ^ { \mid } x ,$ where $c \in \mathbb { R } ^ { d }$ . A QP therefore minimizes a convex quadratic function over a polyhedral feasible set. For arbitrary numbers of hard constraints ${ \mathfrak { n } } > 0$ and scenario-dependent constraints ${ \mathfrak { m } } > 0 .$ the matrix $G \in \mathbb { R } ^ { \mathfrak { n } \times d }$ and vector $h \in \mathbb { R } ^ { \mathfrak { n } }$ encode the hard linear constraints, while the matrix-valued function $\boldsymbol { A } ( \cdot ) \in \mathbb { R } ^ { \mathrm { m } \times d }$ and vector-valued function $b ( \cdot ) \in \mathbb { R } ^ { \mathfrak { m } }$ encode the linear constraints associated with each scenario $\delta _ { i }$

In the following, we describe how the robust-relaxed-regularized QP in (5) can be specialized to recover each of the QP subclasses.

Robust QP. The robust QP in Scen-Opt is obtained by setting $\tau = 0 , \rho =$ 0, and all slack variables $\zeta _ { i } = 0$ . Under these restrictions, the cost function becomes

$$
\operatorname* { m i n } _ { x } \quad c ^ { \top } x + \frac { 1 } { 2 } x ^ { \top } Q x ,
$$

and (5b) reduces to the hard constraints $A ( \delta _ { i } ) x ~ + ~ b ( \delta _ { i } ) \leq 0 .$

Relaxed QP. The relaxed $\mathrm { Q P }$ is obtained by setting $\tau = 0$ , so resulting in the objective function

$$
\operatorname* { m i n } _ { x , \zeta _ { i } } \quad c ^ { \top } x + \frac { 1 } { 2 } x ^ { \top } Q x + \rho \sum _ { i = 1 } ^ { N } \zeta _ { i } ,
$$

while leaving the constraints (5b),(5c) unchanged.

Regularized QP. The regularized QP is obtained by setting $\rho = 0$ and $\zeta _ { i } = 0$ for all i. The objective function becomes

$$
\operatorname* { m i n } _ { x } \quad c ^ { \top } x + \frac { 1 } { 2 } x ^ { \top } Q x + \tau \| x - \bar { x } \| _ { p } ,
$$

while constraints (5b) reduce to the hard constraints $A ( \delta _ { i } ) x ~ + ~ b ( \delta _ { i } ) \leq 0 .$

As in the LP case, different norms $\| \cdot \| _ { p }$ are supported by Scen-Optwhen regularization is present.

We motivate the use of scenario optimization for quadratic programs with a common machine-learning classification problem, as illustrated in the following example.

Example 2 (Support Vector Machine) Consider the problem of binary classification, where one seeks a hyperplane that maximizes the separation margin between two labeled classes of data points. We sample points from a 2- dimensional space, where each point is assigned either a blue or a red label (cf.

![](images/5a21862bec9b4cb8fcf9169ebf7e8b4ea1b24f8cdb3c9aba5dd3e4a1cad032e0.jpg)  
Fig. 2: Robust QP: Support vector machine example.

Fig. 2), and the aim is to seek the optimal separating hyperplane that maximizes the margin between the two classes. This can be formulated as a robust QP, where the decision variable encodes the hyperplane $\mathbf { w } ^ { \top } x + h = 0$ with gradient $\mathbf { w } = \left[ \mathbf { w } _ { 1 } \ \mathbf { w } _ { 2 } \right] ^ { \top }$ and intercept h. The scenario optimization solution can therefore be written as $x _ { N } ^ { * } = \left[ \mathbf { w } _ { 1 } ^ { * } \ \mathbf { w } _ { 2 } ^ { * } \ h ^ { * } \right] ^ { \top }$   
A scenario $\delta _ { i } \in \mathbb { R } ^ { 3 }$ consists of the triple $( \delta _ { i } ( 1 ) , \mathbf { \bar { \delta } } _ { i } ( 2 ) , \delta _ { i } ( 3 ) )$ , where $( \delta _ { i } ( 1 ) , \delta _ { i } ( 2 ) )$ is the coordinate of the sample and $\delta _ { i } ( 3 )$ is its class label. Then the robust QP in (5) is described by the following matrices:

$$
\begin{array}{c} \begin{array} { r } { c = \end{array} [ \begin{array} { c c } { 0 } \\ { 0 } \\ { 0 } \end{array} ] , \quad Q = [ \begin{array} { c c } { 1 } & { 0 } & { 0 } \\ { 0 } & { 1 } & { 0 } \\ { 0 } & { 0 } & { 0 } \end{array} ] , \quad A ( \delta _ { i } ) = \begin{array} { c } { [ - \delta _ { i } ( 1 ) \delta _ { i } ( 3 ) ] ^ { \top } } \\ { - \delta _ { i } ( 2 ) \delta _ { i } ( 3 ) } \\ { - \delta _ { i } ( 3 ) } \end{array} ] , \quad b ( \delta _ { i } ) = 1 , } \end{array}\tag{6}
$$

where the constraint $A ( \delta _ { i } ) x + b ( \delta _ { i } ) \leq 0$ is enforced for each scenario $\delta _ { i }$ Relaxed QP. If the data are not linearly separable, the relaxed $\mathrm { Q P }$ allows one to find a hyperplane that best accommodates the samples by introducing slack variables to handle misclassified points. The JSON program file of the robust $\mathrm { Q P }$ is given in Listing 9 in Appendix B. □

Scen-Opt supports QP scenario optimization through the following function in Listing 2, with its parameters described in detail in Appendix C.2. The function solves the QP in (5) and returns the solution together with all information required for subsequent risk analysis, including the complexity and the number of scenarios N. For completeness, the full set of constraints is also returned, along with a flag indicating whether any degeneracy was detected.

```python
def solve_qp(deltas, A_d, b_d, G, h, c, Q, tau=0.0, x_ref=np.array([0.0]),
rho=0.0, norm_type=2, solver=None, include_slack=None):
return x_out, zeta_out, cost_out, N, complexity, constraints, degeneracy
```

## 3.3 Semidefinite Programming (SDP)

Conic optimization is a structured subclass of convex optimization, and semidefinite programming constitutes a fundamental instance of conic optimization, since the set of semidefinite matrices forms a convex cone. Scen-Opt supports SDPs in the inequality form SDP, which naturally generalizes the inequalitybased formulations used for LPs and QPs.

Several convex optimization problems can be reformulated or relaxed as an SDP, including LPs, QPs, quadratically-constrained QPs (QCQPs), secondorder cone programs (SOCPs), linear matrix inequalities (LMIs), and leastsquares problems. This makes Scen-Opt applicable to a wide range of problem classes.

We first consider the inequality form SDP, in which no equality constraints are present, and both hard and soft constraints are expressed as LMIs. Let $x = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { d } ] \in \mathbb { R } ^ { d }$ denote the vector of decision variables. The robust-relaxed-regularized SDP in inequality form can then be written as

$$
\operatorname* { m i n } _ { x , \zeta _ { i } \geq 0 } \ c ^ { \top } x \ + \ \frac { 1 } { 2 } x ^ { \top } Q x \ + \ \tau \| x - \bar { x } \| _ { p } \ + \ \rho \sum _ { i = 1 } ^ { N } \zeta _ { i }\tag{7a}
$$

subject to:

$$
F _ { 0 } ( \delta _ { i } ) \ + \ \sum _ { j = 1 } ^ { d } x _ { j } F _ { j } ( \delta _ { i } ) \ \preceq \ \zeta _ { i } \mathbb { I } _ { \mathfrak { m } \times \mathfrak { m } } , \quad i = 1 , \dots , N\tag{7b}
$$

$$
E _ { 0 } ~ + ~ \sum _ { j = 1 } ^ { d } x _ { j } E _ { j } ~ \preceq ~ 0 ,\tag{7c}
$$

where $\tau \geq 0 , \rho \geq 0$ , and $\mathbb { I } _ { \mathfrak { m } \times \mathfrak { m } }$ denotes the m × m identity matrix.

In (7), Q is a positive semidefnite symmetric matrix in $\mathbb { R } ^ { d \times d }$ and $c \in$ $\mathbb { R } ^ { d } . \ E _ { 0 } , \ . . . , \ E _ { d }$ are symmetric matrices in $\mathbb { R } ^ { { \mathsf { n } } \times { \mathsf { n } } }$ , these objects define the hard LMI constraint in (7c). Likewise, the symmetric matrix-valued functions $F _ { j } ( \cdot ) \in \mathbb { R } ^ { \mathsf { m } \times \mathsf { m } }$ , for $j = 0 , \ldots , d .$ generate the LMI constraints associated with each scenario $\delta _ { i } \in \mathbb { R } ^ { q }$

Robust SDP. The robust SDP is supported by Scen-Opt and obtained by setting $\tau = 0 , \rho = 0$ , and all $\zeta _ { i } = 0$ , which yields the objective

$$
\operatorname* { m i n } _ { x } \quad c ^ { \top } x + \frac { 1 } { 2 } x ^ { \top } Q x
$$

while (7b) reduces to the hard LMI constraints $\begin{array} { r } { F _ { 0 } ( \delta _ { i } ) + \sum _ { j = 1 } ^ { d } x _ { j } F _ { j } ( \delta _ { i } ) \preceq 0 } \end{array}$ applied for each scenario $\delta _ { i }$

Relaxed SDP. The relaxed SDP corresponds to setting $\tau = 0$ and results in the objective

$$
\operatorname* { m i n } _ { x , \zeta _ { i } } \quad c ^ { \top } x + \frac { 1 } { 2 } x ^ { \top } Q x + \rho \sum _ { i = 1 } ^ { N } \zeta _ { i } ,
$$

while constraints (7b),(7c) remain the same.

Regularized SDP. The regularized $S D P$ is supported by Scen-Opt, with its cost function taking the form

$$
\operatorname* { m i n } _ { x } \quad c ^ { \top } x + \frac { 1 } { 2 } x ^ { \top } Q x + \tau \| x - \bar { x } \| _ { p } ,
$$

and constraints (7b) becoming the hard LMI constraints $\begin{array} { r } { F _ { 0 } ( \delta _ { i } ) + \sum _ { j = 1 } ^ { d } x _ { j } F _ { j } ( \delta _ { i } ) \ \preceq } \end{array}$ 0.

Similarly to the $\mathrm { L P }$ and $\mathrm { Q P }$ cases, different norms $\| \cdot \| _ { p }$ for SDP are supported by the tool when regularization is present.

To demonstrate the use of scenario optimization for SDP inequality form, we present the following illustrative example.

Example 3 (SDP Inequality Form)

Consider a linear parameter-varying (LPV) system $\dot { \xi } = A ( \delta ) \xi$ with

$$
A ( \delta ) = \underbrace { \left[ { \begin{array} { c c } { 0 } & { 1 } \\ { - 2 } & { - 1 } \end{array} } \right] } _ { A _ { 0 } } + \delta \underbrace { \left[ { \begin{array} { c c } { 0 } & { 0 } \\ { 0 . 3 } & { 0 . 1 } \end{array} } \right] } _ { A _ { 1 } } , \qquad \delta \in [ - 0 . 2 2 , \ 1 . 0 ] ,
$$

where $A _ { 0 }$ is a stable damped oscillator and $A _ { 1 }$ captures parameter-varying uncertainty in the restoring and damping terms (see Figure 3 for the state trajectories achieved for various fixed values of δ and initializations). The goal is to find a common quadratic Lyapunov function $V ( x ) = x ^ { \top } P x$ with $P \succ 0$ certifying stability across all parameter values, i.e. $A ( \delta ) ^ { \top } P + P A ( \delta ) \preceq 0 .$

This maps to the SDP inequality form (7) with decision variable $x =$ [p11, p12, $\bar { p _ { 2 2 } } \bar { | \bar { \mathbf { \Psi } } | } \in \mathbb { R } ^ { 3 }$ , the upper-triangular entries of the $2 \times 2$ symmetric matrix P. Let $\begin{array} { r } { \varPi _ { 1 } = [  { \mathbf { \Pi } } _ { 0 } ^ { 1 }  { \mathbf { \Pi } } _ { 0 } ^ { 0 } ] , \varPi _ { 2 } = [  { \mathbf { \Pi } } _ { 1 } ^ { 0 }  { \mathbf { \Pi } } _ { 0 } ^ { 1 } ] , \varPi _ { 3 } = [  { \mathbf { \Pi } } _ { 0 } ^ { 0 }  { \mathbf { \Pi } } _ { 1 } ^ { 0 } ] } \end{array}$ be the basis matrices for the symmetric $2 \times 2$ space, so that $\textstyle P = \sum _ { j = 1 } ^ { 3 } x _ { j } \varPi _ { j }$ . The scenario-dependent constraint matrices in (7b) are

$$
\begin{array} { r } { F _ { 0 } ( \delta _ { i } ) = 0 , \qquad F _ { j } ( \delta _ { i } ) = A ( \delta _ { i } ) ^ { \top } \varPi _ { j } + \varPi _ { j } A ( \delta _ { i } ) , \quad j = 1 , 2 , 3 , } \end{array}
$$

encoding the Lyapunov inequality $A ( \delta _ { i } ) ^ { \top } P + P A ( \delta _ { i } ) \preceq \zeta _ { i } I$ . The hard constraint in (7c) enforces $P \succ 0$ via

$$
E _ { 0 } = 0 , \qquad E _ { j } = - { \cal I } I _ { j } , \quad j = 1 , 2 , 3 .
$$

The objective uses $\boldsymbol { c } = [ - 1 , 0 , - 1 ] ^ { \intercal }$ and $Q = 0 . 1 \mathbb { I } _ { 3 \times 3 }$ , while we set $\tau = 0$ and $\rho = 1$ . This amounts to maximizing trace(P) with light quadratic regularization and constraint relaxation. The corresponding JSON program file is given in Listing 10 in Appendix B. □

Scen-Opt provides SDP scenario optimization via a dedicated function in Listing 3, with parameters described in Appendix C.3. This function solves the SDP specified in (7) and returns the optimal solution together with the quantities needed for risk assessment, including the complexity and the number of scenarios N. It also outputs the complete set of constraints and a diagnostic flag indicating whether degeneracy was detected.

![](images/bfa560234951d81cbb9262449e3ae9ac2c09ba20e085e9d6b809874b789cdff3.jpg)  
Fig. 3: Robust SDP: LPV System Example.

```python
def solve_sdp(deltas, F_d, E, c, Q, tau=0.0, x_ref=np.array([0.0]), rho=0.0,
norm_type=2, solver=None, include_slack=None):
return x_out, zeta_out, cost_out, N, complexity, constraints, degeneracy
```

Listing 3: Function to solve SDP (Inequality Form)

## 4 Scen-Opt: Implementation Details

In this section, we outline the algorithms used to evaluate the risk and complexity of a scenario-based convex optimization problem, in accordance with the theory presented in Section 2. As established by Theorem 1 and Theorem 2, the risk associated with a scenario convex optimization solution depends on three quantities: the number of scenarios N, the complexity $s _ { N } ^ { * }$ , and the userselected confidence level $\beta .$ Among these, the computation of $s _ { N } ^ { * }$ is the most nontrivial task, as it requires identifying a support list, which comprises all constraints violated at the optimum together with an irreducible subset of the active constraints that, jointly, allow reconstruction of the solution (cf. Definition 4). We begin by describing the computation of the risk, followed by the procedures used to determine the complexity and other auxiliary quantities. While $s _ { N } ^ { * }$ is the true complexity, in the implementation we use k for the calculated minimal cardinality, where $k = s _ { N } ^ { * }$ in non-degenerate cases but $k \geq s _ { N } ^ { * }$ for degeneracy (cf. Remark 3).

## 4.1 Risk

def quantify\_risk(k,N,beta): return epsL, epsU

Algorithm 1 Quantifying risk bounds $\underline { { \epsilon } }$ and €   
Require: Complexity k and number of scenarios N with $0 \leq k \leq N ,$ confidence $\beta \in ( 0 , 1 )$   
Ensure: Lower bound ∈ and upper bound ¿   
1: $t _ { \mathrm { m i n } } \gets 0 . 0$ Compute lower bound $\underline { { \epsilon } }$   
2: $t _ { \mathrm { m a x } } \gets k / N$   
3: threshold $ 1 0 ^ { - 1 0 }$   
4: while $( t _ { \mathrm { m a x } } - t _ { \mathrm { m i n } } ) >$ threshold do   
5: $t \gets ( t _ { \operatorname* { m i n } } + t _ { \operatorname* { m a x } } ) / 2$   
6: $L \gets \frac { \beta } { 3 }$ betainc(k + 1, N — $k , \ t ) \ + \ { \frac { \beta } { 6 } }$ betainc(k + 1, 4N + 1 − k, t)   
7: $R  { \big ( } 1 + { \frac { \beta } { 6 N } } { \big ) }$ t N(betainc(k, $N - k + 1 , \ t )$ – betain $\mathsf { \Lambda } [ ( k + 1 , \ N - k , \ t ) ]$   
8: if $L > R$ then   
9: tmin ← t   
10: else   
11: tmax ← t   
12: end if   
13: end while   
14: $\underline { { \epsilon } } \gets t _ { \mathrm { m i n } }$ ▶ when $k = 0$ the loop is skipped and $\underline { { \epsilon } } = 0$   
15: if $k = N$ then ▶ Compute upper bound ¿   
16: ∈ ← 1.0   
17: else   
18: $t _ { \mathrm { m i n } } \gets k / N$   
19: $t _ { \mathrm { m a x } } \gets 1 . 0$   
20: while (tmax − $t _ { \mathrm { m i n } } ) >$ threshold do   
21: t← $( t _ { \operatorname* { m i n } } + t _ { \operatorname* { m a x } } ) / 2$   
22: $\begin{array} { r } { L  ( \frac { \beta } { 2 } - \frac { \beta } { 6 } ) } \end{array}$ betainc $( k + 1 , \ N - k , \ t ) \ + \ \frac { \beta } { 6 }$ betainc(k + 1, 4N + 1 − k, t)   
23: $\begin{array} { r } { R  ( 1 + \frac { \beta } { 6 N } ) } \end{array}$ t N(betainc(k, N − k + 1, t) − betainc(k + 1, N − k, t))   
24: if $L > R$ then   
25: $t _ { \mathrm { m a x } } \gets t$   
26: else   
27: $t _ { \operatorname* { m i n } } \gets t$   
28: end if   
29: end while   
30: $\overline { { \epsilon } } \gets t _ { \mathrm { m a x } }$   
31: end if   
32: return $[ \underline { { \epsilon } } , ~ \overline { { \epsilon } } ]$

The ultimate aim of Scen-Opt is to compute the risk bounds associated with a convex optimization problem solved under the scenario approach. Algorithm 1 summarizes this procedure, assuming that the number of scenarios N, the complexity $k ,$ and the user-defined confidence level $\beta$ have already been obtained. The pseudocode follows the implementation strategy of [14, Appendix B|, originally presented in MATLAB. The algorithm evaluates the bounds stated in Theorem 2, which hold under the non-degeneracy assumption and rely on the incomplete beta function betainc. The corresponding Python implementation used by Scen-Opt is provided in Listing 4.

Remark 2 When the assumption of non-degeneracy, as defined in Definition 1, is violated, the validity of the two-sided bound in Theorem 2 can no longer be guaranteed. In this case, only the upper bound on the risk remains theoretically sound, whereas the lower bound may fail to hold (see Theorem 1).

Listing 5: Function to validate candidate support lists.  
Algorithm 2 Validating candidate support list   
Require: Candidate constraints A, solved optimization problem prob, objective obj, non  
scenario constraints R   
1: function TEST SUPPORT(prob, obj, A, R)   
2: Solve problem prob' ← Problem(obj, A ∪ R)   
3: return True iff the optimal values match: prob.value ≈ prob'.value   
4: end function

Consequently, the risk certificate becomes one-sided, reflecting an additional risk reduction introduced by degeneracy. We recall that, although the toolbox issues a warning when the specified optimization problem instance is detected to be degenerate, the use of the lower bound remains at the discretion of the user, who is responsible for verifying the validity of the non-degeneracy property. □

## 4.2 Support List Validation

In Definition 4, the support list is characterized by two conditions: (i) the solution obtained when only the candidate constraints are enforced coincides with the solution obtained using the full set of constraints, and (ii) the candidate list is irreducible, meaning that removing any of its elements alters the solution. Given any candidate support list, that is any subset of the constraints, we can verify property (i) using Algorithm 2. This algorithm summarizes the procedure implemented in Listing 5, and its operation is independent of the particular solver employed.

```python
def test_support(objective, support, non_risk, ref, solver=None):
return bool
```

Algorithm 2 takes a solved problem prob and its objective obj. The set A is a subset of the scenario constraints, i.e. a candidate support list. TEST SUPPORT re-solves the problem for candidate constraints A together with the nonscenario constraints $R ^ { 7 }$ , and returns True if and only if the optimal value is reproduced. Deciding support list membership using the optimal value keeps the test well posed when the optimizer is non-unique: removing a genuinelysupporting scenario strictly changes the optimum, whereas a redundant one would leave it unchanged. A True result establishes the first requirement of Definition 4; irreducibility is checked separately.

## 4.3 Complexity Computation

```python
def get_support(constraints, non_risk_constraints, prob, objective,
solver=None, threshold=1e-8, sol_tol=1e-6):
return complexity, support, degeneracy
```

Listing 6: Function to determine complexity.

The computation of complexity is the most intricate component of Scen-Opt's implementation. Although the full procedure is realized as a single function in Listing 6, for clarity we describe its constituent steps separately.

Algorithm 3 computes the support list. Lines 3–7 screen the scenario constraints by dual value, adding each constraint with a nonzero dual to the candidate list A. Such a constraint is either active (binding) at the optimum or violated: under relaxation, stationarity of the Lagrangian in the slack $\zeta _ { i }$ forces the dual of a violated constraint to sum to $\rho > 0 ,$ so the check captures both. Line 8 tests, through TEST sUPPORT, whether A alone reproduces the optimum. If it does, the screened constraints A are the support scenarios, and line 9 greedily prunes them via PRUNE (Algorithm 4). Non-degeneracy is equivalent to the support scenarios forming a support list (Property 1), so this prune doubles as a degeneracy test: the flag is raised exactly when PRuNE removes a constraint that is active or violated at the optimum, since the support scenarios are then reducible and more than one support list exists. The removal of a spuriously screened constraint — one that is strictly slack at the optimum, yet carries a small nonzero dual — is instead ignored (see Remark 4). If only spurious screens are removed, the remaining A is an irreducible support list and the instance is certified non-degenerate. In the degenerate case the pruned list is irreducible but not guaranteed to be of minimum cardinality. If TEST\_sUPPORT is False, the dual screening was inaccurate (i.e. from numerical errors in the solver). To be conservative the degeneracy flag is raised and PRuNE restarts from the full scenario list C, which always recovers a valid support list. The complexity is the cardinality of the returned support list $k = | A |$

The greedy reduction PRUNE (Algorithm 4) turns a candidate list into an irreducible support list. It is used in both branches of Algorithm 3: on the small screened set in the normal case, and on the full scenario list when the screening fails to reproduce the optimum. It checks each constraint in turn and removes it (via TEsT\_suppoRT) whenever doing so leaves the optimum unchanged. The returned list is irreducible but not guaranteed to be of minimum cardinality. The pass is ${ \mathcal { O } } ( n )$ in the number of constraints: convexity guarantees that a constraint which is not of support cannot become of support as others are removed, so no constraint need be rechecked.

Remark 3 In cases of degeneracy, it is possible the complexity value calculated is suboptimal, $i . e .$ , for multiple support lists, the smallest support list is not necessarily identified. This sub-optimality is acceptable in practice, as a larger complexity leads to a more conservative (higher) upper bound on the risk.

Algorithm 3 Computing the support list and complexity   
Require: Scenario constraints C, non-scenario constraints R, problem prob, objective obj   
Ensure: Complexity k, support list A, flag for degeneracy d   
1: d ← False   
2: A ←[] candidate support list   
3: for all constraint c in C do   
4: if max(|c.dual value|) > threshold then default threshold 10-8   
5: append c to A c is violated or active   
6: end if   
7: end for   
8: if TEST SUPPORT(prob, obj, A, R) then  screened set reproduces the optimum   
9: As ← A the screened candidates   
10: A ← PRUNE(A, prob, obj, R)   
11: if any c ∈ As \ A is active or violated at the optimum then   
12: d ← True support scenarios reducible ⇒ degenerate   
13: end if   
14: else inaccurate duals: conservative   
15: d ← True   
16: A ← PRUNE(C, prob, obj, R) restart from the full list   
17: end if   
18: k ← |A| cardinality of the support list   
19: return (k, A, d)

Aigoritnm 4 PRuNE: greedy support-list reauction Algorithm 4 PRUNE: greedy support-list reduction

Require: Candidate constraints A, problem prob, objective obj, non-scenario constraints R   
Ensure: Irreducible support list A   
1: i = 0   
2: repeat   
3: Atemp ← A[:, i] + A[i + 1, :] Temporary list without constraint in position i   
4: if TEST\_SUPPORT(prob, obj, Atemp, R) then   
5: A.pop(i) Remove constraint in position i   
6: else   
7: i = i + 1 ▶ increment i   
8: end if   
9: until i == |A|   
10: return A

Consequently, the true complexity would yield a strictly tighter, and therefore less risky certificate than the one obtained from a suboptimal result. □

Remark 4 Some solvers (e.g. interior point) may return small nonzero duals on constraints that are neither active nor violated, so the dual screen of lines 3-7 may retain such spuriously screened constraints. Left unaddressed, their removal by PRuNE would raise the degeneracy flag even in non-degenerate instances, which is overly conservative. Scen-Opt resolves this by recording every screened constraint's activity margin at the optimum (before any re-solve overwrites the solution). A removed constraint whose margin clearly exceeds the tolerance is recognized as a spurious screen and discounted by the degeneracy test. When the margin is ambiguous, the constraint is conservatively treated as active, so the test still leans towards flagging degeneracy. □

## 4.4 Solvers

Scen-Opt builds upon CVXPY [18], a well-established convex optimization framework that interfaces with a wide range of solvers8 . By leveraging CVXPY for scenario optimization, Scen-Opt can access 27 solvers (at the time of writing), including both open-source options (e.g., CLARABEL [23], CVXOPT [2], SCS [35]) and commercial solvers (e.g., MOSEK [3], GUROBI [26]). The function get\_solvers, shown in Listing 7, automatically detects and returns all compatible solvers available on the host machine9. This list is then used to populate the dynamic solver-selection menu in the web interface, as discussed in Section 5.

```python
def get_solvers():
return solver_dict
```  
Listing 7: Function to get installed solvers.

The returned solver\_dict is a mapping from each installed solver's name to the list of program types it supports (LP', ‘QP', and ‘sDP'), which is used to filter the dropdown in the web interface based on the active program tab. Commercial solvers requiring a license are handled by a dedicated subprocess entry point (mosek\_solve.py) that sets the license environment variable before importing CVXPY, allowing concurrent users to each upload their own license without conflicts; the license is held only in RAM and never persists on disk.

## 4.5 Parallelism

With modern computational resources, one might expect significant gains from distributing computations across multiple processors or GPUs. However, for convex optimization it is well established that parallelization, particularly GPU-based acceleration, offers limited practical benefit [4,34]. The current version of Scen-Opt therefore runs sequentially. In Algorithm 3, the screening loop in lines 3–7 only reads dual values and is inexpensive, while PRUNE (Algorithm 4) admits no meaningful parallelism: although it contains a loop, each iteration depends on the outcome of the previous one, since the decision to remove a constraint affects all subsequent checks. The clearest opportunity for future versions is to solve the programs for different values of ρ and τ in parallel, since they are independent. Such an implementation must give every parallel solve its own copy of the optimization problem: if parallel solves share optimization variables, the values written by one solve can overwrite those of another, the support list may then be identified incorrectly, and the complexity k can be under-counted, which would make the risk certificate too optimistic

![](images/984b45aca973bdbff042769852ae81615dd32c4fac76bad7acd6f4be4d3aa264.jpg)  
Fig. 4: Web interface of Scen-Opt illustrating the relaxed and regularized LP.

## 5 Web Interface

Scen-Opt is accompanied by an interactive web interface designed to make the toolbox highly accessible and user-friendly. Figure 4 illustrates the interface for a relaxed and regularized LP, with annotated labels <1>-<10>. In the interface, <1> denotes the tab system that allows navigation between the different convex optimization programs supported by Scen-Opt. Each tab displays the corresponding optimization scheme, where <2> provides a dropdown menu for selecting whether regularization and/or relaxation are included or whether the basic robust formulation is used. The mathematical description of the chosen program is shown in <3>, with variable names aligned to the input fields and buttons in <4>. A row of dimension fields (d, m, n, and, for SDP, the LMI size) sets the problem sizes; changing any of them automatically reshapes the matrices already entered, zero-padding or truncating as needed.

Pressing any blue button opens a modal dialog for entering or uploading matrices or vectors. When relaxation or regularization is not applicable, the corresponding buttons and fields automatically disappear, ensuring a clean and adaptive interface. A help icon (?) beside each field opens a short tooltip explaining the corresponding input. Each matrix modal provides both a cellby-cell grid editor and a JSON-paste box, with live validation. The scenariodependent data A(δ), b(δ), and $F _ { j } ( \delta )$ can be entered either symbolically or numerically — both input modes, and their corresponding scenario file format, are detailed in the Symbolic and numeric input modes paragraph below. The confidence parameter can be specified in $< 5 > ;$ when several values of $\rho$ or τ are swept (see below), this field instead reports the overall confidence $1 - \beta n _ { \tau } n _ { \rho }$ obtained from a union bound over the $n _ { \tau } n _ { \rho }$ solved instances, with $\beta$ the user-entered per-instance value, while the scenario data $\mathrm { { f i e ^ { 1 0 } } }$ is uploaded via $< 7 >$ , whose row-wise format depends on the active input mode (see below). As Scen-Opt supports up to 27 solvers, the dynamically populated solver dropdown ${ < } 6 >$ enables selection among the solvers available on the host machine, automatically filtered to those compatible with the active program type (LP, QP, or SDP). Once all parameters are set, the optimization problem can be executed by clicking Solve at <8>. Label <9> denotes the input-mode toggle, which switches between the Symbolic and Numeric ways of entering the scenario-dependent constraints (detailed below). Label <10> shows the button used to do a full upload of a program in .json or .mat formats, with Scen-Opt automatically detecting the suitable program choice. The program can be supplied either as a file or by pasting its .json text, and the dialog provides a format reference listing the required fields for each program type; in .mat files the symbolic matrices are passed as cell arrays of strings.

Symbolic and numeric input modes. Selected via the toggle ${ \bf < 9 > }$ , Scen-Opt offers two complementary ways of providing the scenario-dependent constraints, in both cases uploaded through $< 7 >$ . Throughout, the uncertainty δ is a q-dimensional vector whose components are addressed as delta[o], ... delta[q-1]. In symbolic mode, each entry of $A ( \delta )$ and b(δ) (LP/QP), or of $F _ { j } ( \delta )$ (SDP), is entered in the matrix editors of <4> as a closed-form expression in delta $[ 0 ] , \ldots ,$ delta[q-1], $e . g .$ , 2\*delta[0]-delta[3]. The file uploaded via $< 7 >$ then contains the scenarios themselves, as an $N \times q$ table whose i-th row lists the components $\delta _ { i } [ 0 ] , \ldots , \delta _ { i } [ q - 1 ]$ of scenario $\delta _ { i } ;$ each row is substituted into the symbolic expressions to generate the i-th scenario constraint. This mode is convenient whenever the dependence on $\delta$ is available in closed form. In numeric mode, the scenario constraints are instead supplied already evaluated: the scenarios are given directly as the instances $A _ { i } , b _ { i }$ (or $F _ { j , i } )$ , so nothing is specified in the matrix editors (they are replaced by simple dimension fields), and the i-th row of the file uploaded via $< 7 >$ lists the row-wise flattened numerical blocks $\left[ A _ { i } \mid b _ { i } \right]$ for $\mathrm { L P / Q P }$ or [ $F _ { 0 , i } \mid F _ { 1 , i } \mid \cdots \mid F _ { d , i } ]$ for SDP. Numeric mode is useful when the constraint data are produced externally $( e . g . ,$ precomputed from raw measurements or another processing pipeline) and do not admit a convenient symbolic form. Both modes solve the same underlying scenario program and return identical results. In numeric mode, Scen-Opt verifies that each uploaded row has the length implied by the declared dimensions, reporting a descriptive error otherwise.

<table><tr><td>Results Table</td><td>ρ Graph</td><td>τ Graph</td><td>Program JSON</td><td colspan="3"></td></tr><tr><td></td><td></td><td>Optimization Results</td><td></td><td>Download .JSON Download .MAT</td><td></td><td></td></tr><tr><td colspan="5"></td><td></td></tr><tr><td>Parameter Optimal Cost </td><td></td><td>Value1</td><td>Value2</td><td>Value3</td><td>Value4</td></tr><tr><td></td><td></td><td>0.5528</td><td>0.7000</td><td>0.6000</td><td>0.7000</td></tr><tr><td>Optimal x</td><td></td><td>0.5528</td><td>0.7000</td><td>0.6000</td><td>0.7000</td></tr><tr><td rowspan="7">Optimal ζ</td><td></td><td>8.526e-11</td><td>1.408e-8</td><td>8.612e-9</td><td>1.088e-12</td></tr><tr><td>6.782e-11</td><td>7.471e-11</td><td>1.676e-8</td><td>7.741e-9</td><td>1.115e-12</td></tr><tr><td>7.597e-11</td><td></td><td>1.857e-8</td><td>7.328e-9</td><td>1.095e-12</td></tr><tr><td>6.704e-11</td><td></td><td>1.882e-8 1.928e-8</td><td>7.894e-9</td><td>1.014e-12</td></tr><tr><td>0.0472</td><td></td><td>2.616e-8</td><td>6.118e-9 2.025e-8</td><td>8.615e-13</td></tr><tr><td></td><td>▼</td><td>▼</td><td>▼</td><td>2.537e-13 1</td></tr><tr><td>Relaxation Parameters ρ</td><td>0.2000</td><td>0.3000</td><td>0.2000</td><td>0.3000</td></tr><tr><td>Regularization Parameters τ</td><td>0.2000</td><td>0.2000</td><td></td><td>0.3000</td><td>0.3000</td></tr><tr><td>Confidence 1 − β·nτ·nρ</td><td>0.6000</td><td></td><td>0.6000</td><td>0.6000</td><td>0.6000</td></tr><tr><td>Risk Bounds ε</td><td>[0.0873, 0.8401]</td><td></td><td>[0.0187, 0.7585]</td><td>[0.0873, 0.8401]</td><td>[0.0187, 0.7585]</td></tr><tr><td>Degeneracy Detected?</td><td>No</td><td>No</td><td></td><td>No</td><td>No</td></tr><tr><td>Complexity (support list size)</td><td>4</td><td>3</td><td>4</td><td></td><td>3</td></tr><tr><td>Number of data samples</td><td>9</td><td>9</td><td></td><td>9</td><td>9</td></tr><tr><td>Total Constraints</td><td>9</td><td></td><td>9</td><td>9</td><td>9</td></tr><tr><td>Optimization Time (s)</td><td>0.0489</td><td></td><td>0.0665</td><td>0.0566</td><td>0.0419</td></tr><tr><td>Risk Computation Time (s)</td><td>0.0295</td><td></td><td>0.0311</td><td>0.0246</td><td>0.0257</td></tr><tr><td>Error</td><td>None</td><td></td><td>None</td><td>None</td><td>None</td></tr></table>

Fig. 5: Scen-Opt web interface displaying the results table for the relaxed and regularized LP set up in Fig. 4 (d = 1, N = 9 scenarios, β = 0.1), with one column for each pair (ρ, τ), $\rho , \tau \in \{ 0 . 2 , 0 . 3 \}$

Remark 5 The Mosek [3] solver requires a license, which is not provided for the user when using the server. Instead, when trying to use Mosek, a user can upload their valid license to be checked and used. The license is only temporarily stored. For convenience, an uploaded license is cached in the browser's session storage so that subsequent solves skip the upload prompt; a cache indicator with a Clear button appears beneath the solver menu, while users with a locally installed Mosek can instead skip the upload and use their system license. Similarly, no scenario or program files uploaded persist on the disk. If privacy concerns remain, the tool can be downloaded/installed locally.

As shown in Fig. 5, pressing Solve triggers the optimization, after which the results are dynamically displayed. The interface provides up to four analysis tabs (the ρ and τ graphs appear only when relaxation or regularization is selected); the first tab presents a results table containing key information such as the optimal cost, the solution values, the associated upper and (when applicable) lower risk bounds, and an indicator flag identifying whether degeneracy was detected. The table also reports the computed complexity, the total number of constraints, the timings for acquiring the results (reported separately as the optimization solve time and the risk-bound computation time), and any numerical issues encountered during the solution process. When degeneracy is flagged, the lower risk bound is withheld and only the certified upper bound is shown, in line with Theorem 2

![](images/38fcb63b173c5198ee88a0cd04e88555fe972b62c95962c655f103c9a1327fae.jpg)  
(a) Comparing $\rho$ for different selections of τ

![](images/cebc1b297fc3c7e858555a9b98deac56501fdd5a797882a2eb1e334258434090.jpg)  
(b) Comparing τ for different selections of $\rho$  
Fig. 6: Example graphical output from Scen-Opt for cases involving relaxation or regularization, for the run of Figs. 4 and 5.

Scen-Opt allows users to specify multiple scalar values for $\rho$ and $\tau ,$ separated by commas. When multiple values are provided, each resulting optimization problem is solved independently with the entered confidence parameter $\beta ,$ and the table reports the overall confidence $1 - \beta n _ { \tau } n _ { \rho }$ , which holds simultaneously for all $n _ { \tau } n _ { \rho }$ cases. When multiple cases are present, the table automatically expands to include additional columns for each trial, and a horizontal scroll bar enables convenient navigation across all displayed results. Similarly, when d or N is large, the optimal x and optimal $\zeta _ { i }$ windows become scrollable. All the results can be subsequently downloaded in .json or .mat formats using the respective green download buttons.

To provide deeper insight, Scen-Opt also offers graphical summaries of the solutions when multiple values of $\rho$ and τ are tested. Figure 6 illustrates this functionality for the run of Figs. 4 and $5 ,$ with two choices of $\rho$ and two choices of $\tau .$ Since visualizing all $( \rho , \tau )$ combinations simultaneously would lead to cluttered plots, the interface instead presents Pareto front slices: one parameter (either $\tau \ \mathrm { o r } \ \rho )$ is fixed ${ < } 1 >$ via an interactive dropdown, while the other $< 2 >$ is varied and plotted against the resulting risk $< 3 >$ and cost <4>. The risk plot displays both the upper (blue) and lower (green) bounds, whereas the cost curve is shown in black. These visualizations allow users to conveniently explore and interpret the trade-off between risk and objective value across different regularization and relaxation settings in the spirit of [21, 22].

The final tab in the results displays the form data, which allows users to verify that the optimization problem has been specified correctly. The . json representation shown in this tab can also be saved, via a dedicated Download Program button, and later re-uploaded through the Upload Program dialog, providing a convenient workflow for experimenting with multiple related optimization problems or repeating similar analyses.

## 6 Benchmarking

Table 1 summarizes the twelve case studies used to evaluate Scen-Opt. The benchmarks span all three supported program types (LP, QP, SDP) and cover decision-variable dimensions from $d = 2$ to $d = 1 2 0$ , with scenario uncertainty dimensions up to $q = 4 8 .$ Solve times are wall-clock measurements on a desktop with an Intel Core Ultra 9 285K (24 cores) and 64 GB RAM running Windows 11 using MOSEK [3] as the underlying conic solver. For two benchmarks (Growth Bound and Radiation Therapy) Scen-Opt detects degeneracy

— the constraints active at the optimum do not form a support list — so their lower risk bounds are not certified (see Theorem 2) and are marked $\mathrm { { N / A } }$ in the table. Full problem descriptions are provided in Appendix A.

## 7 Conclusions and Future Work

This work presented Scen-Opt, the first open-source software tool designed to bring the scenario approach into practical, user-friendly data-driven convex optimization. Building on the rigorous theoretical foundations of scenario theory, which links sampled constraints, their support list, and the generalization properties of the resulting decision, Scen-Opt enabled researchers and practitioners to solve linear, quadratic, and semidefinite programs using only sampled uncertainty realizations while obtaining certified probabilistic guarantees on performance. The tool was implemented in Python and complemented with a modern JavaScript-based web interface, providing an intuitive and highly accessible user experience across devices. It supported both web-based usage and local installation, allowing flexible data entry through manual input or file uploads. Through a set of representative benchmarks, we demonstrated the practical effectiveness of Scen-Opt for reliable, data-driven decision making. Looking ahead, the framework established in this work opens the door to extending Scen-Opt to accommodate emerging advances in nonconvex scenario optimization.

Table 1: Summary of all twelve benchmark case studies solved with Scen-Opt, grouped by program type. d = decision variables, q = uncertainty dimension, $m _ { s } =$ scenario constraints (rows of A(δ) for LP/QP, LMI dimension for SDP), $\begin{array} { r l } { m _ { h } } & { { } = } \end{array}$ hard constraints (rows of G for $\mathrm { L P / Q P , }$ LMI dimension for SDP), N = number of sampled scenarios, k = complexity (number of support constraints). All benchmarks use the confidence parameter $\beta = 1 0 ^ { - 6 } .$ Solve times are wall-clock seconds on a desktop with an Intel Core Ultra 9 285K (24 cores) and 64 GB RAM running Windows 11. †Power Dispatch, Inventory and Portfolio CVaR use an augmented formulation in which the slack penalty is embedded in the objective coefficient c; the value of rho passed to the solver is therefore 0, while the displayed value reflects the user-facing penalty weight. N/A = degeneracy detected, so the lower risk bound is not certified. Scenario data come from Yahoo Finance (Portfolio CVaR) and the UCI repository (Iris SVM [20], Covariance Estimation [1], Min. Encl. Ellipsoid [42]); Radiation Therapy uses synthetic data inspired by TROTS [6], and the scenarios of all other benchmarks are generated as described in Appendix A.

<table><tr><td>Benchmark</td><td>d</td><td></td><td>q ms</td><td>mh</td><td>ρ</td><td>T</td><td>N</td><td>k</td><td>m</td><td>ē</td><td>Time (s)</td></tr><tr><td>Linear Programs (LP)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Half Width</td><td>2</td><td>1</td><td>2</td><td>0</td><td>0</td><td>0</td><td>100</td><td>2</td><td></td><td>0 0.2085</td><td>1.2</td></tr><tr><td>Growth Bound</td><td>12</td><td>6</td><td>3</td><td>6</td><td>0</td><td>0</td><td>3127</td><td>6</td><td></td><td>N/A 0.0103</td><td>1276.8</td></tr><tr><td>Inventory</td><td>11</td><td>15</td><td>6</td><td>17</td><td>100†</td><td>0</td><td>500</td><td>7</td><td></td><td>0 0.0666</td><td>4.0</td></tr><tr><td>Portfolio CVaR</td><td>13</td><td>12</td><td>4</td><td>27</td><td>0.016†</td><td>0</td><td>1255</td><td>3</td><td>0</td><td>0.0203</td><td>22.0</td></tr><tr><td>Power Dispatch</td><td>120</td><td>48</td><td>48</td><td>330</td><td>100†</td><td>0</td><td>150</td><td>34</td><td>0.0791</td><td>0.4496</td><td>2.9</td></tr><tr><td colspan="2">Quadratic Programs (QP)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Iris SVM</td><td>3</td><td>3</td><td>1</td><td>0</td><td>0</td><td>0</td><td>150</td><td>2</td><td></td><td>0 0.1441</td><td>1.2</td></tr><tr><td>Robot Navigation</td><td>118</td><td>1</td><td>3</td><td>320</td><td>0</td><td>0</td><td>500</td><td>1</td><td></td><td>0 0.0403</td><td>1.6</td></tr><tr><td>Radiation Therapy</td><td>50</td><td>3</td><td>80</td><td>100</td><td>0</td><td>0.1</td><td>200</td><td>5</td><td></td><td>N/A 0.1414</td><td>112.3</td></tr><tr><td colspan="2">Semidefinite Programs</td><td>(SDP)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LPV Stability</td><td>3</td><td>1</td><td>2</td><td>2</td><td>1</td><td>0</td><td>100</td><td>1</td><td></td><td>0 0.1858</td><td>1.7</td></tr><tr><td>Quadratic Stability</td><td>6</td><td>2</td><td>3</td><td>3</td><td>0</td><td>0</td><td>500</td><td>1</td><td></td><td>0 0.0403</td><td>2.2</td></tr><tr><td>Covariance Estimation</td><td>15</td><td>5</td><td>5</td><td>5</td><td>0</td><td>0</td><td>178</td><td>7</td><td></td><td>0 0.1780</td><td>2.1</td></tr><tr><td>Min. Encl. Ellipsoid</td><td>15</td><td>5</td><td>1</td><td>5</td><td>0</td><td>0</td><td>569</td><td>5</td><td></td><td>0 0.0518</td><td>235.6</td></tr></table>

## References

1. Aeberhard, S., Forina, M.: Wine. UCI Machine Learning Repository (1992). DOI: https://doi.org/10.24432/C5PC7J

2. Andersen, M., Dahl, J., Vandenberghe, L.: CVXOPT: Convex optimization. Astrophysics Source Code Library (2020)

3. ApS, M.: Mosek optimizer api for python. Version 9(17), 6–4 (2022)

4. Balkanski, E., Singer, Y.: Parallelization does not accelerate convex optimization: Adaptivity lower bounds for non-smooth convex minimization. arXiv:1808.03880 (2018)

5. Boyd, S.P., Vandenberghe, L.: Convex optimization. Cambridge university press (2004)

6. Breedveld, S., Heijmen, B.: Trots-the radiotherapy optimisation test set (2019)

7. Buzby, J.C., Farah-Wells, H., Hyman, J.: The estimated amount, value, and calories of postharvest food losses at the retail and consumer levels in the united states. USDA-ERS Economic Information Bulletin (121) (2014)

8. Calafiore, G.C., Campi, M.C.: The scenario approach to robust control design. IEEE Transactions on automatic control 51(5), 742–753 (2006)

9. Campi, M.C., Carè, A., Garatti, S.: The scenario approach: A tool at the service of data-driven decision making. Annual Reviews in Control 52, 1–17 (2021)

10. Campi, M.C., Garatti, S.: The exact feasibility of randomized solutions of uncertain convex programs. SIAM Journal on Optimization 19(3), 1211–1230 (2008)

11. Campi, M.C., Garatti, S.: A sampling-and-discarding approach to chance-constrained optimization: feasibility and optimality. Journal of optimization theory and applications 148(2), 257–280 (2011)

12. Campi, M.C., Garatti, S.: Introduction to the scenario approach. SIAM (2018)

13. Campi, M.C., Garatti, S.: Wait-and-judge scenario optimization. Mathematical Programming 167(1), 155–189 (2018)

14. Campi, M.C., Garatti, S.: Compression, generalization and learning. Journal of Machine Learning Research 24(339), 1–74 (2023)

15. Campi, M.C., Garatti, S., Prandini, M.: The scenario approach for systems and control design. Annual Reviews in Control 33(2), 149–157 (2009)

16. Campi, M.C., Garatti, S., Ramponi, F.A.: A general scenario theory for nonconvex optimization and decision making. IEEE Transactions on Automatic Control 63(12), 4067–4078 (2018)

17. Deasy, J.O., Blanco, A.I., Clark, V.H.: Cerr: a computational environment for radiotherapy research. Medical physics 30(5), 979–985 (2003)

18. Diamond, S., Boyd, S.: Cvxpy: A python-embedded modeling language for convex optimization. Journal of Machine Learning Research 17(83), 1–5 (2016)

19. Draxl, C., Clifton, A., Hodge, B.M., McCaa, J.: The wind integration national dataset (wind) toolkit. Applied Energy 151, 355–366 (2015)

20. Fisher, R.A.: Iris. UCI Machine Learning Repository (1936). DOI: https://doi.org/10.24432/C56C76

21. Garatti, S., Campi, M.C.: Risk and complexity in scenario optimization. Mathematical Programming 191(1), 243–279 (2022)

22. Garatti, S., Campi, M.C.: Non-convex scenario optimization. Mathematical Programming 209(1), 557–608 (2025)

23. Goulart, P.J., Chen, Y.: Clarabel: An interior-point solver for conic programs with quadratic objectives. arXiv:2405.12762 (2024)

24. Grant, M., Boyd, S., Ye, Y.: CVX users' guide (2009)

25. Grantham, A., Pudney, P., Ward, L., Belusko, M., Boland, J.: Generating synthetic five-minute solar irradiance values from hourly observations. Solar Energy 147, 209– 221 (2017)

26. Gurobi Optimization, LLC: Gurobi Optimizer Reference Manual (2024). URL https: //www.gurobi.com

27. Hou, Z.S., Wang, Z.: From model-based control to data-driven control: Survey, classification and perspective. Information Sciences 235, 3–35 (2013)

28. Kazemi, M., Majumdar, R., Salamati, M., Soudjani, S., Wooding, B.: Data-driven abstraction-based control synthesis. Nonlinear Analysis: Hybrid Systems 52 (2024)

29. Margellos, K., Falsone, A., Garatti, S., Prandini, M.: Distributed constrained optimization and consensus in uncertain networks via proximal minimization. IEEE Transactions on Automatic Control 63(5), 1372–1387 (2017)

30. Margellos, K., Goulart, P., Lygeros, J.: On the road between robust optimization and the scenario approach for chance constrained optimization problems. IEEE Transactions on Automatic Control 59(8), 2258–2263 (2014)

31. Margellos, K., Prandini, M., Lygeros, J.: On the connection between compression learning and scenario based single-stage and cascading optimization problems. IEEE Transactions on Automatic Control 60(10), 2716–2721 (2015)

32. Mohajerin Esfahani, P., Sutter, T., Kuhn, D., Lygeros, J.: From infinite to finite programs: explicit error bounds with applications to approximate dynamic programming. SIAM journal on optimization 28(3), 1968–1998 (2018)

33. Mohajerin Esfahani, P., Sutter, T., Lygeros, J.: Performance bounds for the scenario approach and an extension to a class of non-convex programs. IEEE Transactions on Automatic Control 60(1), 46–58 (2014)

34. Nemirovski, A.: On parallel complexity of nonsmooth convex optimization. Journal of Complexity 10(4), 451–463 (1994)

35. O'Donoghue, B., Chu, E., Parikh, N., Boyd, S.: Conic optimization via operator splitting and homogeneous self-dual embedding. Journal of Optimization Theory and Applications 169(3), 1042–1068 (2016). URL http://stanford.edu/\~boyd/papers/scs.html

36. Perez, A., Ferreira, G., Minor, T.: Fruit and tree nuts outlook (2017)

37. Rockafellar, R.T., Uryasev, S., et al.: Optimization of conditional value-at-risk. Journal of risk 2, 21–42 (2000)

38. Romao, L., Margellos, K., Papachristodoulou, A.: Probabilistic feasibility guarantees for convex scenario programs with an arbitrary number of discarded constraints. Automatica 149, 110601 (2023)

39. Romao, L., Papachristodoulou, A., Margellos, K.: On the exact feasibility of convex scenario programs with discarded constraints. IEEE Transactions on Automatic Control 68(4), 1986–2001 (2022)

40. Schildbach, G., Fagiano, L., Morari, M.: Randomized solutions to convex programs with multiple chance constraints. SIAM Journal on Optimization 23(4), 2479–2501 (2013)

41. Shang, C., You, F.: A posteriori probabilistic bounds of convex scenario programs with validation tests. IEEE Transactions on Automatic Control 66(9), 4015–4028 (2020)

42. Wolberg, W., Mangasarian, O., Street, N., Street, W.: Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository (1993). DOI: https://doi.org/10.24432/C5DW2B

43. Zhang, X., Grammatico, S., Schildbach, G., Goulart, P., Lygeros, J.: On the sample size of random convex programs with structured dependence on the uncertainty. Automatica 60, 182–188 (2015)

## A Case Studies

All case study benchmarks can be found in the repository at https://github.com/Kiguli/ Scen-0pt/tree/v1.0/benchmarks, and in the archived release at https://doi.org/10.5281/ zenodo. 23177690. The results presented here were run on a desktop with an Intel Core Ultra 9 285K (24 cores) and 64 GB RAM running Windows 11.

## A.1 Linear Programming Cases

## A.1.1 Inventory Management

A purchasing manager must decide how many cases of fresh produce to order daily from the Oakland Terminal Market. Fresh produce is highly perishable, e.g. strawberries might have 25% perished by the time they reach stores. Demand fluctuates based on weather, local events and seasonal patterns. The purchasing manager has a fixed budget and limited warehouse space. Too little orders means empty shelves and lost sales. Too many and produce spoils on the shelf, a wasted purchase.

Determine the optimal daily order quantities to minimize wastage for 5 perishable produce, as labeled in Table 2. The daily budget is \$12, 000 and warehouse has 800 cubic feet of refrigerator space. Products have a maximum order limit of 150 — 200 cases, item dependent.

The decision variable is $\boldsymbol { x } \in \mathbb { R } ^ { 5 }$ , representing daily order quantities for five perishable produce items (strawberries, tomatoes, lettuce, avocados, bell peppers), with per-case ordering cost c = [18.50, 14.25, 12.00, 32.00, 22.50]T USD, slack penalty $\rho = 1 0 0 ,$ and no regularization (τ = 0). Each scenario $\delta _ { i } ~ \in \mathbb { R } ^ { 1 5 }$ encodes uncertain yields $\delta _ { i } ^ { ( 1 : 5 ) }$ , demands $\delta _ { i } ^ { ( 6 : 1 0 ) }$ , and warehouse packing efficiencies $\delta _ { i } ^ { ( 1 1 : 1 5 ) }$ , drawn from correlated distributions calibrated to USDA wholesale market data. The scenario-dependent constraint matrices in (3b) are

<table><tr><td>Product</td><td>Cost ($)</td><td>Mean Demand (cases)</td><td>Yield (%)</td><td>Space (cu. ft)</td></tr><tr><td>Strawberries</td><td>18.50</td><td>85</td><td>75 (25 loss)</td><td>1.2</td></tr><tr><td>Tomatoes</td><td>14.25</td><td>120</td><td>88 (12 loss)</td><td>1.5</td></tr><tr><td>Lettuce</td><td>12.00</td><td>95</td><td>82 (18 loss)</td><td>2.0</td></tr><tr><td>Avocados</td><td>32.00</td><td>60</td><td>90 (10 loss)</td><td>0.8</td></tr><tr><td>Bell Peppers</td><td>22.50</td><td>75</td><td>85 (15 loss)</td><td>1.3</td></tr></table>

Table 2: Produce purchasing details. The data is synthetic but calibrated to real data [7, 36], along with USDA Terminal Market Prices for highest grade products.

![](images/90151514db8bf9f889c0673e93af409b2ee1253cac5d0b079d19cf8cb040954f.jpg)

![](images/81a2f0d1473bec201750b1dffb94fde4e295d2fc30383268d1e247d0a07bd8ec.jpg)

![](images/7d4d7dd7d8dc3c2a63f0c04d56c0015ea0c6e3f21f792391701801a74c34e4c8.jpg)  
Fig. 7: Left. Comparison of optimal order against mean demand. Center. Quantified risk bounds for $N = 5 0 0$ and confidence 99%. Right. Graphic of scenario samples for different produce.

<table><tr><td rowspan=1 colspan=1>Product</td><td rowspan=1 colspan=1>Order (cases)</td><td rowspan=1 colspan=1>Mean Demand</td><td rowspan=1 colspan=1>Service Level (%)</td></tr><tr><td rowspan=1 colspan=1>Strawberries</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>83.9</td><td rowspan=1 colspan=1>85.8</td></tr><tr><td rowspan=1 colspan=1>Tomatoes</td><td rowspan=1 colspan=1>155.2</td><td rowspan=1 colspan=1>118.9</td><td rowspan=2 colspan=1>72.40.2</td></tr><tr><td rowspan=1 colspan=1>Lettuce</td><td rowspan=1 colspan=1>40.4</td><td rowspan=1 colspan=1>94.3</td></tr><tr><td rowspan=2 colspan=1>AvocadosBell Peppers</td><td rowspan=1 colspan=1>118.9</td><td rowspan=1 colspan=1>59.8</td><td rowspan=2 colspan=1>99.890.0</td></tr><tr><td rowspan=1 colspan=1>121.0</td><td rowspan=1 colspan=1>74.1</td></tr></table>

Table 3: Optimal Purchasing Order

$$
A ( \delta _ { i } ) = \left[ \begin{array} { c c c c c c } { - \delta _ { i } ^ { ( 1 ) } } & & & & & \\ & { - \delta _ { i } ^ { ( 2 ) } } & & & & \\ & & { \ddots } & & & \\ & & & & { - \delta _ { i } ^ { ( 5 ) } } \\ & & & & & & { \ddots } \end{array} \right] , \quad b ( \delta _ { i } ) = \left[ \begin{array} { c } { \delta _ { i } ^ { ( 6 ) } } \\ { \delta _ { i } ^ { ( 7 ) } } \\ { \vdots } \\ { \delta _ { i } ^ { ( 1 0 ) } } \\ { - C _ { \mathrm { w h } } } \end{array} \right] ,
$$

where $s _ { j }$ is the warehouse space per case of product $j$ and $C _ { \mathrm { w h } } = 8 0 0 ~ \mathrm { f t ^ { 3 } }$ . The first five rows enforce that usable supply meets demand; the last row enforces refrigerated warehouse capacity. The hard constraints in (3c) are

$$
G = \left[ \begin{array} { l } { - \mathbb { I } _ { 5 } } \\ { \mathbb { I } _ { 5 } } \\ { c ^ { \top } } \end{array} \right] , \quad h = \left[ \begin{array} { l } { 0 _ { 5 } } \\ { - \bar { q } } \\ { - B } \end{array} \right] ,
$$

encoding non-negativity, supplier limits $\bar { q } = [ 1 5 0 , ~ 2 0 0$ , 180, 120, $1 4 0 ] ^ { \top }$ cases, and a daily budget $B = \$ 12,000.$ is the n × n identity matrix.

Scen-Opt solves the LP in 4.0 seconds considering $N = 5 0 0$ scenarios. The optimal cost is \$45, 099.67 with all \$12, 000 of budget used. The penalty $\rho = 1 0 0$ implies a cost per case of lost sales for an item, therefore the expected daily loss from stockouts is only \$66.20 or about 0.5% of the ordering budget. Lettuce is bulky (2.0 cu.ft/case), cheap (\$12/case), and has high spoilage (18%). Under a \$12, 000 budget, it is more efficient to spend budget on avocados (\$32/case but 90% yield) and accept lettuce shortfalls. With confidence $1 0 ^ { - 6 }$ , the LP problem had complexity $k = 7 ,$ no degeneracy, and risk bounds [0.0000, 0.0666]. This gives a maximum 6.7% chance of the optimal cost being nullified if another sample was acquired.

<table><tr><td>Generator</td><td>Capacity</td><td>Min Output</td><td>Cost</td><td>Ramp Rate</td></tr><tr><td>Gas 1</td><td>400 MW</td><td>100 MW</td><td>40 $/MW</td><td>200 MW/h</td></tr><tr><td>Gas 2</td><td>300 MW</td><td>75 MW</td><td>50 $/MW</td><td>150 MW/h</td></tr><tr><td>Coal</td><td>500 MW</td><td>200 MW</td><td>30 $/MW</td><td>100 MW/h</td></tr></table>

Table 4: Thermal generator parameters. Costs and capacities are synthetic but calibrated against standard values.

![](images/5b56eb390ffbf3fd4b86f8acd203f5f8b80c81ec32f1b68108be77a98ed10891.jpg)

![](images/ffb211e999948aa60ddb930ace24be0ad118802aef30e0e90c8f696336a54483.jpg)

![](images/e0545716c05676a813ac8d6f76ce8ea41f70f86a186d735063ff509dd9cd3afe.jpg)  
Fig. 8: Left. Stacked generation dispatch over 24 hours: coal provides baseload, gas ramps for the evening peak. Center. Quantified risk bounds for $N = 1 5 0$ and confidence 99%. Right. Wind and solar scenario variability (median with $2 5 - 7 5 \mathrm { t h }$ and 10 - 90th percentile bands).

## A.1.2 Power Dispatch

A grid operator must schedule three thermal generators over a 24-hour horizon to meet electricity demand while integrating uncertain wind and solar generation. Demand follows a typical residential-commercial daily profile peaking at 1000 MW. The system includes a 200 MW wind farm and 150 MW solar farm whose output cannot be known in advance.

Determine the optimal hourly generation schedule $P _ { m , t }$ for each thermal unit m at each hour t. The decision variable is $x = P \in \mathbb { R } ^ { 7 2 } \ ( 3$ generators × 24 hours), with per-MWh generation cost $c = [ \underbrace { 4 0 , \ldots , 4 0 } _ { 2 4 } , \underbrace { 5 0 , \ldots , 5 0 } _ { 2 4 } , \underbrace { 3 0 , \ldots , 3 0 } _ { 2 4 } ] ^ { \intercal }$ , slack penalty $\rho = 1 0 0$ , and no regularization $( \tau = 0 )$ . Each scenario δi $\in \mathbb { R } ^ { 4 8 }$ encodes uncertain hourly wind generation $\delta _ { i } ^ { ( 1 : 2 4 ) }$ and solar generation $\delta _ { i } ^ { ( 2 5 : 4 8 ) }$ , drawn from a Weibull distribution (shape 2.0, hourvarying scale) for wind [19] and a Beta(2, 5) cloudiness factor applied to a clear-sky envelope for solar [25]. The scenario-dependent constraint matrices in (3b) enforce power balance at each hour:

$$
A ( \delta _ { i } ) = \left[ - e _ { t } ^ { \top } \right] _ { t = 0 } ^ { 2 3 } , \quad b ( \delta _ { i } ) = \left[ \begin{array} { c } { \delta _ { i } ^ { ( t ) } + \delta _ { i } ^ { ( 2 4 + t ) } - D _ { t } } \\ { - \delta _ { i } ^ { ( t ) } - \delta _ { i } ^ { ( 2 4 + t ) } + D _ { t } } \end{array} \right] _ { t = 0 } ^ { 2 3 } ,
$$

where $e _ { t } \in \mathbb { R } ^ { 7 2 }$ selects all generators at hour t and $D _ { t }$ is the demand at hour t. Each pair of rows bounds the power imbalance from above and below. The hard constraints in (3c) are

$$
G x + h \leq 0 ,
$$

encoding generator capacity limits $P _ { \operatorname* { m i n } } \leq P _ { m , t } \leq P _ { \operatorname* { m a x } }$ and ramp-rate constraints $\vert P _ { m , t + 1 } -$ $P _ { m , t } | \leq R _ { m }$ (282 rows total).

Scen-Opt solves the LP considering $N \ = \ 1 5 0$ renewable generation scenarios in 2.9 seconds. The total cost is \$831,357, comprising \$537,670 in fuel costs and \$293,687 in power imbalance penalty $( \rho = 1 0 0$ per MW of mismatch). Coal, the cheapest unit at \$30/MWh, provides baseload at 284 — 500 MW. Gas 1 (\$40/MWh) ramps from its 100 MW minimum to 300 MW during the evening peak (hour 18). Gas 2 (\$50/MWh) remains at its 75 MW minimum throughout, dispatching it further is never cost-effective. The LP problem has confidence $1 0 ^ { - 6 }$ , complexity $k = 3 4$ , no degeneracy, and risk bounds [0.0791, 0.4496]. This gives a ≈ 55% likelihood of having the correct result for unseen data, therefore, more data is likely required to increase the confidence in this power dispatch solution.

<table><tr><td>Ticker</td><td>Asset</td><td>Class</td><td>Role</td></tr><tr><td>SPY AGG VNQ GLD EFA</td><td>S&amp;P 500 ETF US Aggregate Bond Real Estate REIT Gold Intl Developed 20+ Year Treasury</td><td>US Equity Fixed Income Real Assets Commodities Intl Equity Long Bonds</td><td>Core Growth Stability Inflation Hedge Crisis Hedge Diversification</td></tr></table>

Table 5: ETF asset universe. Historical returns are downloaded from Yahoo Finance via yfinance (5 years daily data).

## A.1.3 Portfolio Management - Conditional Value at Risk (CVaR)

A pension fund must allocate capital across 8 ETF asset classes to minimize tail risk under market uncertainty. The fund uses Conditional Value-at-Risk (CVaR) at the 95% level (the average loss in the worst 5% of scenarios) as its primary risk measure, following the linear formulation of [37]

The decision variable is $x = [ w , \alpha ] \in \mathbb { R } ^ { 9 }$ , where w $\in \mathbb { R } ^ { 8 }$ are portfolio weights and α is the VaR threshold. The objective $\dot { \boldsymbol { c } } = [ 0 , \dots , 0 , 1 ] ^ { \top }$ minimizes α, with slack penalty $\rho = 1 / ( ( 1 - 0 . 9 5 ) \cdot N )$ ≈ 0.016 so that $\alpha + \rho \sum \zeta _ { i }$ approximates CVaR at the 95% level. There is no regularization $( \tau = 0 )$

Each scenario $\delta _ { i } ~ \in \mathbb { R } ^ { 1 2 }$ encodes 8 asset returns $\delta _ { i } ^ { ( 1 : 8 ) }$ (bootstrapped from historical data), a market stress indicator $\delta _ { i } ^ { ( 9 ) } \in [ 0 , 1 ] ,$ credit spread shock $\delta _ { i } ^ { ( 1 0 ) }$ , interest rate shock $\delta _ { i } ^ { ( 1 1 ) }$ , and volatility scaling $\delta _ { i } ^ { ( 1 2 ) }$ . The scenario-dependent constraint matrix in (3b) has 4 rows:

$$
A ( \delta _ { i } ) = \left[ { \begin{array} { c c c c c c c c } { \gamma \delta _ { i } ^ { ( 1 ) } } & { \gamma \delta _ { i } ^ { ( 2 ) } } & { \gamma \delta _ { i } ^ { ( 3 ) } \gamma \delta _ { i } ^ { ( 4 ) } } & { \gamma \delta _ { i } ^ { ( 5 ) } } & { \gamma \delta _ { i } ^ { ( 6 ) } } & { \gamma \delta _ { i } ^ { ( 7 ) } } & { \gamma \delta _ { i } ^ { ( 8 ) } } & { - 1 } \\ { 0 } & { - 0 . 5 \delta _ { i } ^ { ( 1 0 ) } } & { 0 } & { 0 } & { 0 } & { - 0 . 3 \delta _ { i } ^ { ( 1 0 ) } } & { 0 } & { - \delta _ { i } ^ { ( 1 0 ) } } & { 0 } \\ { 0 } & { - 5 \delta _ { i } ^ { ( 1 1 ) } } & { 0 } & { 0 } & { 0 } & { - 1 8 \delta _ { i } ^ { ( 1 1 ) } } & { 0 } & { - 8 \delta _ { i } ^ { ( 1 1 ) } } & { 0 } \\ { - \frac { \delta _ { i } ^ { ( 9 ) } } { 2 } \delta _ { i } ^ { ( 1 ) } } & { 0 } & { 0 } & { 0 } & { - \frac { \delta _ { i } ^ { ( 9 ) } } { 2 } \delta _ { i } ^ { ( 5 ) } } & { 0 } & { - \frac { \delta _ { i } ^ { ( 9 ) } } { 2 } \delta _ { i } ^ { ( 7 ) } } & { 0 } & { 0 } \end{array} } \right] ,
$$

with $\gamma = - ( 1 { + } 0 . 3 \delta _ { i } ^ { ( 9 ) } )$ and $b ( \delta _ { i } ) = 0 _ { 4 }$ . The first row is the main CVaR loss constraint with stress amplification; rows 2 and 3 capture credit spread and interest rate duration exposure on bond assets (AGG, TLT, LQD); row 4 models equity correlation breakdown during crises (SPY, EFA, VWO). The hard constraints in (3c) encode position limits $( 5 - 3 0 \%$ per asset), full investment $( \sum w _ { j } \ = \ 1 )$ , equity allocation range $( 3 0 - 6 0 \% )$ , fixed income minimum (25%), and VaR threshold bounds (23 rows total).

Scen-Opt solves the LP considering N = 1255 market scenarios (one per historical trading day) with confidence $\beta ~ = ~ 1 0 ^ { - 6 }$ in 22.0 seconds. The optimizer tilts heavily toward alternatives: GLD (23.9%) and VNQ (21.1%) absorb the allocation not consumed by regulatory minimums. Equities sit at the 30% floor and fixed income at 25%, both at their binding lower limits. This minimizes tail risk, the CVaR(95%) is only 1.20% daily loss. The LP problem has complexity $k = 3 .$ no degeneracy, and risk bounds [0.0000, 0.0203].

## A.1.4 Smallest Enclosing Interval

We sample 100 points from the 1-dimensional space [0, 1] and seek the smallest interval that encloses all samples, following the setup in [12]. This is formulated as a robust LP, where the solution consists of the interval center $o ^ { * }$ and half-width w\*, $i . e . , x _ { N } ^ { * } = \left[ o ^ { * } w ^ { * } \right] ^ { \top } .$ The corresponding matrices defining the robust LP in (3) are given in (4), where the constraint $A ( \delta _ { i } ) { \bar { x } } + b ( \delta _ { i } ) \leq 0$ is enforced for each sampled point (scenario) $\delta _ { i } .$ Using the dataset with $N = 1 0 0 , \beta = 1 0 ^ { - 6 }$ , and complexity $k = 2 ,$ the resulting risk bounds are computed as $\epsilon = 0$ and $\bar { \epsilon } = 0 . 2 0 9$ . Scen-Opt solved the LP in 1.2 seconds. The corresponding optimal solution is computed as $x _ { 1 0 0 } ^ { * } = \left[ 0 . 5 0 1 \ 0 . 4 9 8 \right] ^ { \top }$

![](images/b4288a5a445a219c0e6b72b689235ebad040c1445f99da5e60cbe3a8c0a2ba1b.jpg)

![](images/fb5570666f2ed3d539ef78879c64b2191ce46721b645d72f4287874bd0d2644a.jpg)

![](images/6121bd92ff82275d11149aebb57f7234f3e595d6e8450a0096dd5d2c16833050.jpg)  
Fig. 9: Left. Optimal portfolio weights with position limit constraints. Center. Quantified risk bounds for $N = 1 2 5 5$ and $\beta = 1 0 ^ { - 6 }$ .Right. Portfolio return distribution with VaR(95%) and CVaR(95%) marked.

![](images/e1617cb05b764f214598802aa5e4ad6d1ddddbc05014bd589af8784a306d1a71.jpg)

![](images/4963a79a3c919171bd63626f42e01a692a339ca5e676b6c0f59054c69a8b87d3.jpg)

![](images/500d8eb2414ae005259c2db57b5664010167e0728fa6955ae5a41f5a90292ff5.jpg)  
Fig. 10: Left. Optimal half-width given sampled points. Center. Quantified risk bounds for $N = 1 0 0$ and confidence 99.9999%. Right. Density of sampled points across the region.

## A.1.5 Data-Driven Growth Bound for Finite Abstraction

In the formal methods community, mathematical guarantees on system behavior are often obtained through hybridization, whereby a continuous-space system is approximated via a finite abstraction and the abstraction is analyzed to infer guarantees for the original continuous-space dynamics. The work [28] employs scenario optimization to construct a data-driven growth bound, formulated as a linear program and derived directly from sampled system trajectories, which in turn is used in building the finite abstraction of an unknown deterministic system. As a case study, we illustrate the computation of such a growth bound using Scen-Opt, referring the reader to the original work for the full abstraction framework and controller synthesis procedure.

Finite abstractions group a set of continuous states into grid cells, each represented by a single point (typically the cell center). We consider a 3-dimensional hypercube of side length $l = 1 . 6 ,$ centered at (0, 1.2, 0), together with vehicle dynamics of the form

$$
\dot { x } ( 1 ) = u ( 1 ) \frac { c o s ( \alpha + x ( 3 ) ) } { c o s ( \alpha ) } , \quad \dot { x } ( 2 ) = u ( 1 ) \frac { s i n ( \alpha + x ( 3 ) ) } { c o s ( \alpha ) } , \quad \dot { x } ( 3 ) = u ( 1 ) t a n ( u ( 2 ) ) ,
$$

where $\begin{array} { r } { \alpha = a r c t a n ( \frac { t a n ( u ( 2 ) ) } { 2 } ) } \end{array}$

A total of 3127 trajectories $( x _ { k } , x _ { k } ^ { \prime } )$ are sampled from the hypercube under the fixed input $u = ( 0 . 3 , 0 . 3 )$ , where $( c , c ^ { \prime } )$ denotes the trajectory originating from the center. The growth bound

$$
\left[ { \begin{array} { l l l } { m _ { 1 1 } ~ m _ { 1 2 } ~ m _ { 1 3 } } \\ { m _ { 2 1 } ~ m _ { 2 2 } ~ m _ { 2 3 } } \\ { m _ { 3 1 } ~ m _ { 3 2 } ~ m _ { 3 3 } } \end{array} } \right] \left[ { \boldsymbol { 0 } } . 5 l \right] + \left[ { \gamma } _ { \gamma } \right] \qquad 
$$

with $\gamma$ being a bias term, is then computed using the following $\mathrm { L P }$

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { m _ { i j } , b _ { i } } } & { \displaystyle \sum _ { i , j = 1 , i \neq j } ^ { 3 } m _ { i j } + \sum _ { i = 1 } ^ { 3 } b _ { i } } \\ { \mathrm { s u b j e c t ~ t o : } } & { | m _ { i i } | \leq b _ { i } , } \\ & { m _ { i 1 } | x _ { k } ( 1 ) - c ( 1 ) | + m _ { i 2 } | x _ { k } ( 2 ) - c ( 2 ) | + m _ { i 3 } | x _ { k } ( 3 ) - c ( 3 ) | \leq | x _ { k } ^ { \prime } ( i ) - c ^ { \prime } ( i ) | } \\ & { \mathrm { w h e r e ~ } i , j = 1 , 2 , 3 \mathrm { ~ a n d ~ } m _ { i j } \geq 0 \mathrm { ~ w h e n ~ } i \neq j . } \end{array}
$$

We consider a single scenario $\delta _ { k }$ formed by the 6-dimensional concatenation $| x _ { k } - c | , | x _ { k } ^ { \prime } -$ $c ^ { \prime } | .$ The matrices and vectors defining the corresponding robust LP constraints are

$$
A ( \delta _ { i } ) = \left[ { \begin{array} { c c c c c } { - \delta _ { i } ( 1 ) } & { - \delta _ { i } ( 2 ) } & { - \delta _ { i } ( 3 ) } & { 0 _ { 6 } } & { 0 _ { 3 } } \\ { 0 _ { 3 } } & { - \delta _ { i } ( 1 ) } & { - \delta _ { i } ( 2 ) } & { - \delta _ { i } ( 3 ) } & { 0 _ { 6 } } \\ { 0 _ { 6 } } & { - \delta _ { i } ( 1 ) } & { - \delta _ { i } ( 2 ) } & { - \delta _ { i } ( 3 ) } & { 0 _ { 3 } } \end{array} } \right] , \quad b ( \delta _ { i } ) = \left[ { \begin{array} { c } { \delta _ { i } ( 4 ) } \\ { \delta _ { i } ( 5 ) } \\ { \delta _ { i } ( 6 ) } \end{array} } \right] ,
$$

$$
G = { \left[ \begin{array} { l l l l l } { 1 } & { 0 _ { 2 } } & { 0 _ { 6 } } & { - 1 } & { 0 _ { 2 } } \\ { - 1 } & { 0 _ { 2 } } & { 0 _ { 6 } } & { - 1 } & { 0 _ { 2 } } \\ { 0 _ { 4 } } & { 1 } & { 0 _ { 5 } } & { - 1 } & { 0 } \\ { 0 _ { 4 } } & { - 1 } & { 0 _ { 5 } } & { - 1 } & { 0 } \\ { 0 _ { 4 } } & { 0 _ { 4 } } & { 1 } & { 0 _ { 2 } } & { - 1 } \\ { 0 _ { 4 } } & { 0 _ { 4 } } & { - 1 } & { 0 _ { 2 } } & { - 1 } \end{array} \right] } , \quad h = { \left[ \begin{array} { l } { 0 } \\ { 0 } \\ { 0 } \\ { 0 } \\ { 0 } \\ { 0 } \end{array} \right] } .
$$

Solving the optimization yields the following rounded solution:

$$
M ^ { * } = { \begin{array} { l } { \Bigl [ 1 . 0 0 0 \ 0 . 0 0 0 \ 0 . 0 0 5 } \\ { 0 . 0 0 0 \ 1 . 0 0 0 \ 0 . 0 0 9 } \\ { 0 . 0 0 0 \ 0 . 0 0 0 \ 1 . 0 0 0 } \end{array} \Bigr ] } , \quad b ^ { * } = { \bigl [ } 1 . 0 0 0 \ 1 . 0 0 0 \ 1 . 0 0 0 { \bigr ] } .
$$

Using the pre-selected bias $\gamma = 0 . 0 6 7$ and confidence parameter $\beta = 1 0 ^ { - 6 }$ , the resulting growth bound is computed as [0.871 $\boldsymbol { 0 . 8 7 4 0 . 8 6 7 } \rVert ^ { \top }$ . The complexity is $k = 6 ;$ here Scen-Opt detects degeneracy: essentially all 3127 scenario constraints are (numerically) active at the optimum, while only six are needed to reproduce it, so the active constraints do not form a support list and multiple support lists exist. The lower risk bound is therefore not certified (cf. Theorem 2) and only the upper bound is reported: $\overline { { \epsilon } } = 0 . 0 1 0 3$ . Scen-Opt computes this in 1276.8 seconds.

## A.2 Quadratic Programming Cases

## A.2.1 Radiation Therapy - Prostate Brachytherapy

A medical physicist must design a brachytherapy treatment plan that delivers a prescribed dose of 145 Gy to the prostate while minimizing dose to surrounding organs at risk (OARs). Catheter placement uncertainty of ±5 mm occurs due to tissue deformation, needle deflection, and inter-fraction anatomy changes.

The treatment uses dose-influence data [17] from the Treatment and Outcome Tracking System (TROTS) Prostate BT 01 patient [6]. The original dataset has 55 catheters/dwell positions and 7 anatomical structures totaling approximately 35, 000 voxels. We subsample to $d = 5 0$ dwell positions (selected by total dose contribution) and 100 representative voxels (selected by dose variance): 20 prostate $( \mathrm { P T V } )$ , 30 rectum, 30 bladder, and 20 normal tissue. The nominal dose-influence matrix $D \in \mathbb { R } ^ { 1 0 0 \times 5 0 }$ relates dwell-position intensities to voxel doses, with entry $D _ { v , b }$ representing the dose deposited in voxel v by unit intensity of dwell position b.

![](images/60fb8f612e6873629059404f4d9671494bf39470cc0007a76adf1002d9a07718.jpg)

![](images/f421013bfc094c5988cb695958033e1303f17ba09853fd007bd73ee6d61f2fae.jpg)

![](images/7cc7c4bf7e313d5d64c8ac3a5ee4ea8c88f8b72869b8a0c588367296e4e77cd4.jpg)  
Fig. 11: Left. Dose-volume histogram showing tumor coverage above the $1 3 7 . 7 5 \mathrm { G y }$ minimum with OAR doses within limits (rectum $< 1 0 0 \mathrm { G y } ,$ bladder $< 1 2 0 \mathrm { G y } ) ;$ the x-axis is capped at 3× prescription as brachytherapy produces extreme hot spots near dwell positions. Center. Risk bounds with $k = 5$ support constraints. Right. Dwell-position intensity profile showing most channels near maximum with selective suppression near $\mathrm { O A R s }$

The TROTS dataset contains only $N \ : = \ : 2 5$ scenarios, which is too small for proper analysis. Therefore, we design synthetic data inspired by this dataset. Each scenario $\bar { \delta _ { i } } \in \mathbb { R } ^ { 3 }$ encodes a catheter placement shift in three spatial directions (anterior-posterior, lateral, superior-inferior). A 5-level grid $( 5 ^ { 3 } = 1 2 5$ scenarios) combined with 75 random samples from a truncated Gaussian $( \sigma = 2 \mathrm { { m m } ) }$ yields $N = 2 0 0$ scenarios. The perturbed doseinfluence matrix follows an affine model:

$$
\begin{array} { r } { D ( \delta ) = D _ { 0 } + \nabla _ { x } D \cdot \delta _ { 1 } + \nabla _ { y } D \cdot \delta _ { 2 } + \nabla _ { z } D \cdot \delta _ { 3 } , } \end{array}
$$

where the gradient matrices are computed via finite differences on the nominal dose-influence data.

This is a Quadratic Program in the scenario approach framework. The scenario-dependent constraint matrices enforce dose bounds under each catheter shift:

$$
A ( \delta _ { i } ) x + b ( \delta _ { i } ) \leq 0 , \quad i = 1 , \ldots , N ,
$$

where $A ( \delta _ { i } ) \in \mathbb { R } ^ { 8 0 \times 5 0 }$ encodes 20 minimum tumor dose constraints $\begin{array} { r } { \left( D _ { v } \left( \delta _ { i } \right) x \ge 1 3 7 . 7 5 ~ \mathrm { G y } , \right. } \end{array}$ i.e. 95% of the 145 Gy prescription), 30 maximum rectum dose constraints $( D _ { v } ( \delta _ { i } ) x \ \leq$ $1 0 0 \mathrm { G y } )$ and 30 maximum bladder dose constraints $\left( D _ { v } \left( \delta _ { i } \right) x \ \leq \ 1 2 0 \mathrm { G y } \right)$ . The 100 hard constraints enforce non-negativity and upper bounds on each dwell-position intensity $( 0 \leq$ $x _ { b } \leq 1 5 0 { \mathrm { ~ G y } } { \cdot } { \mathrm { s } } )$

The objective minimizes negative tumor dose (maximizing target coverage) with $\ell _ { 2 }$ regularization $( \tau = 0 . 1 )$ to encourage physically smooth dwell-time profiles:

$$
\operatorname* { m i n } _ { x } \quad c ^ { \top } x + \textstyle \frac { 1 } { 2 } x ^ { \top } Q x + \tau \| x \| _ { 2 } ,
$$

where $Q = D _ { \mathrm { P T V } } ^ { \top } D _ { \mathrm { P T V } }$ penalizes dose inhomogeneity across tumor voxels and c encodes the negative mean tumor dose contribution of each dwell position. The problem is solved robustly $( \rho = 0 ) \colon$ all dose constraints must hold for every scenario without slack relaxation.

Figure 11 presents the solution. The optimal plan achieves a tumor $D _ { 9 5 } = 1 4 5 . 1 2$ Gy (exceeding the 137.75 Gy minimum), with maximum rectum dose of 84.6 Gy (below the 100 Gy limit) and maximum bladder dose of 101.6 Gy (below the 120 Gy limit).

The left panel shows the dose-volume histogram (DVH). The tumor curve drops steeply beyond the prescription dose, confirming adequate target coverage, while the long tail reflects the hot spots inherent to brachytherapy (high doses near dwell positions are physically unavoidable and clinically acceptable). Both OAR curves fall well within their respective dose constraints.

The center panel shows the risk bounds. The complexity is $k = 5 ;$ here Scen-Opt detects degeneracy: 20 scenario constraints are active at the optimum while only five form the support list, so the lower risk bound is not certified (cf. Theorem 2) and only the upper bound is reported, $\varepsilon _ { u } = 0 . 1 4 1 4$ at 99.9999%confidence $( \beta = 1 0 ^ { - 6 } )$ . This guarantees that the dose constraints will be satisfied for a new random catheter placement with probability at least 85.9%. The low complexity relative to $d = 5 0$ decision variables indicates that only a small number of catheter-shift scenarios are critical to the solution geometry.

The right panel displays the dwell-position intensity profile. Most channels carry high intensity $( \approx 1 5 0 \mathrm { G y \cdot s } .$ the upper bound), with a few positions at low or zero intensity. This reflects the geometry of the prostate target relative to the catheter array: dwell positions near the center of the $\mathrm { P T V }$ deliver maximum intensity, while those far from the target or near OARs are suppressed.

## A.2.2 Robot Motion Planning

We consider a point robot navigating from a start position (0.5, 2.5) to a goal $( 9 . 5 , 2 . 5 )$ in a 10 × 8 m workspace containing a wide rectangular wall obstacle spanning $x \in [ 3 , 7 ] , y \in [ 0 , 3 ]$ The robot has double-integrator dynamics discretized over $T = 2 0$ timesteps with $\varDelta t = 0 . 4$ S (total horizon 8s). The state vector at each timestep is $( p _ { x } , p _ { y } , v _ { x } , v _ { y } )$ and the control input is $( a _ { x } , a _ { y } )$ , yielding $d = 1 1 \aleph$ decision variables (80 states and 38 controls).

The direct path at $y = 2 . 5$ is blocked by the wall, so the robot must arc above the wall face at $y = 3 . 0$ with a safety clearance of $d _ { \mathrm { s a f e } } = R _ { \mathrm { r o b o t } } + d _ { \mathrm { m a r g i n } } = 0 . 2 + 0 . 1 = 0 . 3 \mathrm { m } .$ The wall face position is uncertain: each scenario $\delta _ { i } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ with σ = 0.05 m perturbs the effective face location, generating 3 scenario constraints per sample (one per constrained timestep in the wall's x-range), for a total of $N \times 3 = 1$ ,500 scenario constraints. In addition, 320 hard constraints encode the dynamics $( 4 \times 1 9 = 7 6$ equality-as-inequality pairs for state transitions), initial conditions (8 constraints fixing the start state), control limits $( 2 \times 1 9 = 3 8$ upper and 38 lower bounds on acceleration at $| a | \leq 2 . 5 \mathrm { m } / \mathrm { s } ^ { 2 } )$ , workspace bounds $( 2 \times 2 0 = 4 0$ upper and 40 lower bounds on y-position), and goal proximity (4 constraints enforcing $\| p _ { T } - p _ { \mathrm { g o a l } } \| _ { \infty } \leq 0 . 1 \mathrm { m } )$ . The total problem has 1, 820 inequality constraints.

The objective minimizes a weighted combination of control effort $( R = 1 )$ and position tracking error $( Q _ { \mathrm { p o s } } = 1 0 )$ relative to a straight-line reference at $y = 2 . 5$

$$
\operatorname* { m i n } _ { x } ~ c ^ { \top } x + \frac { 1 } { 2 } x ^ { \top } Q x ~ \mathrm { s . t . } ~ A ( \delta _ { i } ) x \leq b ( \delta _ { i } ) , ~ i = 1 , \dots , N , ~ G x \leq h .
$$

With $N = 5 0 0$ scenarios, the solver returns a smooth arc trajectory with path length 9.79 m and minimum wall clearance 0.19 m from the safety boundary. The complexity is $k =$ 1 (one support constraint). Calculating the risk bounds with $\beta = 1 0 ^ { - 6 }$ yields $[ 0 . 0 0 0 0 , 0 . 0 4 0 ]$ guaranteeing that with at least 99.9999% confidence, the probability of constraint violation under a new scenario draw is at most 4%. No degeneracy is detected, and there is no regularization or relaxation $( \tau = 0$ and $\rho = 0 )$ . The QP was solved in 1.6 seconds.

## A.2.3 Minimal Iris Classification

The small iris classification dataset [20] is one of the earliest and most widely used benchmarks for evaluating classification algorithms. It includes 150 iris plants belonging to three species, Iris Setosa, Iris Versicolour, and Iris Virginica, each described by four measured features: sepal length, sepal width, petal length, and petal width.

Using a simple linear classifier, Iris Setosa can be reliably distinguished from the other two species. Moreover, only the petal-related measurements are needed for this separation. We therefore consider a minimal binary classification problem based on two features, petal length and petal width, to determine whether a given iris sample belongs to the Iris Setosa class.

![](images/1a17fb46f34ee6d07e7a322729dd647f70ce6aa313ba9b036a98c49526a5fcd6.jpg)

![](images/dd5907dfd5c5434a31e4ff34a3e08cb9c53c5f9e6b721b988ccbe760f8295cd4.jpg)

![](images/a5d692cc57ef3d1f6165c097cff1b3db858753f0425bd5c2fbec5644b7d4bee0.jpg)  
Fig. 12: Left. Optimal trajectory arcing above the wall obstacle. Center. Risk bounds as a function of complexity $k = . \ \mathbf { R i g h t }$ . Wall clearance at timesteps within the wall's x-range.

![](images/047a6b44be1d9b5451dfe18224fe79ed06f6695361070b83daf032e546c28adf.jpg)

![](images/849371560a072ec0b2ef32a3ab3c59545dff1165451d32fe0cb688f7ccf5dc1c.jpg)

![](images/dae26e4912c4ec2007a3af505027a8c801320676d23472574d937af72f2dddc9.jpg)  
Fig. 13: Left. Optimal separating hyperplane with margin boundaries and support vectors. Center. Risk bounds with complexity $k = 2 .$ Right. Petal feature distributions by class.

We sample points from the 2-dimensional feature space, labeled in blue and orange, and aim to determine a separating hyperplane that maximizes the margin between the two classes (cf. Fig 13). A natural approach for this task is the linear hard-margin support vector machine $( S V M )$ formulated for data points $x _ { i }$ with binary labels $y _ { i }$ as the following optimization problem:

$$
\begin{array} { r l } { \underset { \mathbf { w } , b } { \operatorname* { m i n } } } & { \frac { 1 } { 2 } \| \mathbf { w } \| ^ { 2 } } \\ { \mathrm { s u b j e c t ~ t o : } } & { y _ { i } ( \langle \mathbf { w } , x _ { i } \rangle + b ) \geq 1 . } \end{array}
$$

This can be formulated as a robust $\boldsymbol Q { \cal P } ,$ where the separating hyperplane is given by $\mathbf { w } ^ { \top } x + b = 0$ .The optimizer returns $\mathbf { w } ^ { * } = \left[ \mathbf { w } _ { 1 } ^ { * } \mathbf { w } _ { 2 } ^ { * } \right] ^ { \top }$ and the corresponding intercept $b ^ { * } ,$ so that, abusing notation, the scenario solution is $x _ { N } ^ { * } = \left[ \mathbf { w } _ { 1 } ^ { * } \mathbf { w } _ { 2 } ^ { * } b ^ { * } \right] ^ { \top }$ Each scenario $\delta _ { i }$ consists of the feature-label pair $( \delta _ { i } ( 1 ) , \delta _ { i } ( 2 ) , \delta _ { i } ( 3 ) )$ , where $\mathsf { \bar { ( } } \delta _ { i } ( 1 ) , \delta _ { i } ( 2 ) \mathsf { ) }$ are the feature coordinates and $\delta _ { i } ( 3 )$ is the binary class label. The associated matrices defining the robust $\mathrm { Q P }$ in (5) are given in (6). The risk is computed using the dataset with $N = 1 5 0$ , confidence level $\beta \stackrel { \cdot } { = } 1 0 ^ { - 6 }$ , and complexity $k = 2 ,$ yielding risk bounds $\underline { { \epsilon } } = 0$ and $\overline { { \epsilon } } = 0 . 1 4 4$ , solved in 1.2 seconds. The resulting optimal solution $x _ { 1 5 0 } ^ { * }$ corresponds to the linear classifier

$$
3 . 7 9 = 1 . 2 9 x _ { 1 } + 0 . 8 2 x _ { 2 } .
$$

![](images/ca44f0cb27fb8e55308204e0ee191d0589ac6fb048298f9b28411854887453f0.jpg)

![](images/67e40616082dbe4df9f4db7f994ae4210b55626cecdf18a5c82599a0b9637093.jpg)  
Fig. 14: Left. Entry-wise error $\hat { \Sigma } - { \Sigma } _ { \mathrm { f u l l } } ;$ diagonal inflation reflects the domination requirement. Center. Risk bounds with $k = 7$ support constraints. Right. Eigenvalue spectrum comparison: the estimated covariance uniformly dominates the full-sample covariance in every principal direction.

## A.3 Semidefinite Programming Cases

## A.3.1 Robust Covariance Estimation

We seek the minimum-trace positive semidefinite covariance matrix Σ that dominates all subsample covariances drawn from the UCI Wine dataset [1]. The scenario approach provides probabilistic guarantees that Σ will also dominate the covariance of a new random subsample drawn from the same population.

The Wine dataset contains 178 samples with 13 chemical measurements per wine. We select the first 5 features (alcohol, malic acid, ash, alcalinity of ash, magnesium) and standardize each to zero mean and unit variance, yielding the full-sample covariance $\boldsymbol { \Sigma } _ { \mathrm { f u l l } } \in \mathbb { R } ^ { 5 \times 5 }$

The decision variable i $\textbf { s } x = [ \sigma _ { 1 1 } , \sigma _ { 1 2 } , . . . , \sigma _ { 5 5 } ] \in \mathbb { R } ^ { 1 5 }$ , the 15 free entries of the 5 × 5 symmetric matrix $\begin{array} { r } { \Sigma = \sum _ { k = 1 } ^ { 1 5 } x _ { k } \varPi _ { k } } \end{array}$ , where $\varPi _ { k }$ are the standard basis matrices for the symmetric $5 \times 5$ space. Each scenario $\delta _ { i } \in \mathbb { R } ^ { 5 }$ is one of the 178 wine data points $( N = 1 7 8$ scenarios, one per wine).

This is a Semidefinite Program (SDP) in the scenario approach inequality form. The scenario-dependent constraint matrices encode covariance domination as a $5 \times 5$ LMI:

$$
F _ { 0 } ( \delta _ { i } ) + \sum _ { j = 1 } ^ { 1 5 } x _ { j } F _ { j } ( \delta _ { i } ) \preceq 0 ,
$$

where $F _ { 0 } ( \delta _ { i } ) = S _ { \mathrm { s u b } } ( \delta _ { i } )$ and $F _ { j } = - \pi _ { j } ,$ so that $S _ { \mathrm { s u b } } ( \delta _ { i } ) - \Sigma \preceq 0$ i.e. $\Sigma \succeq S _ { \mathrm { s u b } } ( \delta _ { i } )$ for each scenario. The hard constraint enforces $\Sigma \succ 0$ via $E _ { 0 } = 0$ and $E _ { j } = - \pi _ { j }$ . The objective minimizes trace(Σ) using $c = [ 1 , 0 , 0 , 1 , . . . , 1 ] ^ { \intercal }$ and $Q = 0$ . There is no slack penalty $( \rho = 0 )$ and no regularization $( \tau = 0 )$

Figure 14 presents the solution. The left panel shows the entry-wise error $\hat { \Sigma } - \Sigma _ { \mathrm { f u l l } }$ as a heatmap. The diagonal entries are uniformly positive (ranging from +0.54 $\mathrm { t o \ t { o } \ + 0 . 9 3 ) }$ reflecting the conservatism inherent in the domination requirement: the estimated covariance must exceed every subsample covariance, which inflates the diagonal. Off-diagonal errors are smaller and mixed in sign.

The right panel shows the risk bounds. The complexity is $k = 7$ with no degeneracy, yielding risk bounds $[ \varepsilon _ { \ell } , \varepsilon _ { u } ] = [ 0 . 0 0 0 0 , 0 . 1 7 8 ]$ at 99.9999% confidence $( \beta = 1 0 ^ { - 6 } )$ . This guarantees that Σ will dominate the covariance of a new random 50-wine subsample with probability at least 82.2%. Scen-Opt solved the SDP in 2.1 seconds.

![](images/82ed5b479786df977326a26b0a937a4568a8bb9f2b3cda66a18394552f2802e0.jpg)

![](images/cd5a922498001429aa4ef49e7fe9c34d40f2143f86d7354e02d6905e15cdb4d2.jpg)

![](images/adb143557f87015aacdf5084482ccb60eb8fa1ef94614b41980992933e4c4967.jpg)  
Fig. 15: Left. 2D projection showing 569/569 points contained (100%) inside the region. Center. Risk bounds with $k = 5$ support constraints.Right. Containment profile: sorted $( z - { \bar { x } } ) ^ { \top } P \left( z - { \bar { x } } \right)$ values for all 569 data points; the boundary at 1.0 separates interior from exterior points.

## A.3.2 Minimum Enclosing Ellipsoid - Breast Cancer Wisconsin

Given a collection of data points, we seek the tightest ellipsoid (in the trace sense) that contains all sampled observations. This is a fundamental problem in robust statistics and anomaly detection: the scenario approach provides probabilistic guarantees that unseen data points from the same distribution will also be contained.

The Breast Cancer Wisconsin dataset [42] contains 569 samples with 30 features. We select the first 5 features (mean radius, mean texture, mean perimeter, mean area, mean smoothness) and standardize each to zero mean and unit variance, with data center $\bar { x } =$ mean(data) ≈ 0.

The decision variable is $x = [ p _ { 1 1 } , p _ { 1 2 } , . . . , p _ { 5 5 } ] \in \mathbb { R } ^ { 1 5 }$ , the entries of the $5 \times 5$ symmetric shape matrix $\begin{array} { r } { P = \sum _ { k = 1 } ^ { 1 5 } x _ { k } \varPi _ { k } , } \end{array}$ defining the ellipsoid

$$
\mathcal { E } = \big \{ z \in \mathbb { R } ^ { 5 } : ( z - \bar { x } ) ^ { \top } P ( z - \bar { x } ) \leq 1 \big \} .
$$

Each scenario $\delta _ { i } \in \mathbb { R } ^ { 5 }$ is a data point drawn from the dataset $( N = 5 6 9$ scenarios).

This is a Semidefinite Program (SDP) in the scenario approach inequality form. The scenario-dependent constraint matrices encode point containment as a $1 \times 1$ (scalar) LMI:

$$
F _ { 0 } ( \delta _ { i } ) + \sum _ { j = 1 } ^ { 1 5 } x _ { j } F _ { j } ( \delta _ { i } ) \leq 0 ,
$$

where $F _ { 0 } ( \delta _ { i } ) = - 1$ and $F _ { j } ( \delta _ { i } ) = ( \delta _ { i } - { \bar { x } } ) ^ { \top } \varPi _ { j } \left( \delta _ { i } - { \bar { x } } \right)$ , so that $( \delta _ { i } - { \bar { x } } ) ^ { \top } P \left( \delta _ { i } - { \bar { x } } \right) - 1 \leq 0$ The hard constraint enforces $P \succ 0$ via $E _ { 0 } = 0$ and $E _ { j } = - \pi _ { j }$ . The objective maximizes trace(P) (i.e. minimizes − trace(P)) using $c = [ - 1 , 0 , \bar { 0 } , - 1 , . . . , - 1 ] ^ { \top }$ and $Q = 0 ;$ finding the tightest ellipsoid. There is no slack penalty $( \rho = 0 )$ and no regularization $( \tau = 0 )$

Figure 15 presents the solution. The left panel shows the 2D projection of the ellipsoid onto the first two standardized features (mean radius vs. mean texture). Since all 569 data points are used as scenarios, the ellipsoid contains all of them (100% containment).

The center panel shows the risk bounds. The complexity is $k = 5$ with no degeneracy, yielding risk bounds $[ \varepsilon _ { \ell } , \varepsilon _ { u } ] = [ 0 . 0 0 0 , 0 . 0 5 2 ]$ at 99.9999% confidence $( \beta = 1 0 ^ { - 6 } )$ , guaranteeing that a new random data point will fall inside the ellipsoid with probability at least 94.8%.

The right panel displays the containment profile: for each of the 569 data points (sorted by containment value), the quantity $( z - { \bar { x } } ) ^ { \top } P \left( z - { \bar { x } } \right)$ is plotted. Points below the dashed line at 1.0 are contained. The optimal P has trace 44.59 and eigenvalues $\{ \approx 0 , \approx 0 , \approx 0 , $ ≈ $0 , ~ 4 4 . 5 9 \}$ , revealing that the standardized data effectively lies in a 1-dimensional subspace. The first 5 breast cancer features are highly correlated (mean radius, mean perimeter, and mean area are near-linearly dependent), so P concentrates its "tightness" along the single direction of genuine data spread. The SDP was solved in 235.6 seconds

![](images/77af8319272fe1b9245de035354f0d0a48a0b8a9de65a23f6426a19e9f34bcdc.jpg)

![](images/19839c11a7b9d17503ea3321a0541ad4b41d43e6f7310986b83b4d8dfac2f07b.jpg)  
Fig. 16: Left. Stabilizing trajectories with for different scenarios δ. Right. Risk bounds for the LPV stability with complexity $k = 1$

## A.3.3 LPV Stability

Consider a linear parameter-varying (LPV) system ${ \dot { x } } = A ( \delta )$ x with

$$
A ( \delta ) = \underbrace { { \left[ { \begin{array} { l l } { 0 } & { 1 } \\ { - 2 } & { - 1 } \end{array} } \right] } } _ { A _ { 0 } } + \delta \underbrace { { \left[ { \begin{array} { l l } { 0 } & { 0 } \\ { 0 . 3 } & { 0 . 1 } \end{array} } \right] } } _ { A _ { 1 } } , \qquad \delta \in [ - 0 . 2 2 , \ 1 . 0 ] ,
$$

where $A _ { 0 }$ is a stable damped oscillator and $A _ { 1 }$ captures parameter-varying uncertainty in the restoring and damping terms. The goal is to find a common quadratic Lyapunov function $V ( x ) \ = \ x ^ { \top } P x$ with $P \ \succ \ 0$ certifying stability across all parameter values, i.e. $A ( \delta ) ^ { \top } P + \dot { P } \dot { A } ( \delta ) \preceq 0$

This maps to the SDP inequality form (7) with decision variable $x = [ p _ { 1 1 } , p _ { 1 2 } , p _ { 2 2 } ] ^ { \top } \in$ $\mathbb { R } ^ { 3 }$ , the upper-triangular entries of the $2 \times 2$ symmetric matrix $P \cdot$ Let $\begin{array} { r } { \dot { \cal I } { } _ { 1 } = \left[ \frac { 1 } { 0 } \frac { 0 } { 0 } \right] , \dot { \cal I } { } _ { 2 } = } \end{array}$ ${ \left[ \begin{array} { l l } { 0 } & { 1 } \\ { 1 } & { 0 } \end{array} \right] } , ~ { \cal { H } } _ { 3 } ~ = ~ { \left[ \begin{array} { l l } { 0 } & { 0 } \\ { 0 } & { 1 } \end{array} \right] }$ be the basis matrices for the symmetric $2 \times 2$ space, so that $P =$ $\textstyle \sum _ { j = 1 } ^ { 3 } x _ { j } \pi _ { j }$ . The scenario-dependent constraint matrices in (7b) are

$$
\begin{array} { r } { F _ { 0 } ( \delta _ { i } ) = 0 , \qquad F _ { j } ( \delta _ { i } ) = A ( \delta _ { i } ) ^ { \top } \varPi _ { j } + \varPi _ { j } A ( \delta _ { i } ) , \quad j = 1 , 2 , 3 , } \end{array}
$$

encoding the Lyapunov inequality $A ( \delta _ { i } ) ^ { \top } P + P A ( \delta _ { i } ) \preceq \zeta _ { i } I .$ The hard constraint in (7c) enforces $P \succ 0$ via

$$
E _ { 0 } = 0 , \qquad E _ { j } = - \varPi _ { j } , \quad j = 1 , 2 , 3 .
$$

The objective uses $\boldsymbol { c } = [ - 1 , 0 , - 1 ] ^ { \top }$ and $Q = 0 . 1 \mathbb { I } _ { 3 }$ , which maximizes trace(P) with light quadratic regularization $( \tau = 0 , \rho = 1 )$

Scen-Opt solves the SDP with $N = 1 0 0$ scenarios. The optimal Lyapunov matrix is

$$
P ^ { * } = \left[ { 1 0 . 6 7 \ 1 . 3 0 } \right] ,
$$

with complexity $k = 1 ,$ no degeneracy, and risk bounds $[ 0 . 0 , 0 . 1 8 5 8 ]$ at 99.9999% confidence. Only a single scenario constraint supports the solution, corresponding to the smallest sampled value $\delta \approx - 0 . 1 9$ , where the maximum eigenvalue of $A ^ { \top } { \hat { P } } + P A$ approaches zero. The SDP is solved in 1.7 seconds.

## A.3.4 Quadratic Stability - Coupled Oscillator

Consider a 3-state coupled oscillator with two uncertain parameters:

$$
\dot { x } = \left( A _ { 0 } + \delta _ { 1 } A _ { 1 } + \delta _ { 2 } A _ { 2 } \right) x , \qquad ( \delta _ { 1 } , \delta _ { 2 } ) \in [ - 1 , 1 ] ^ { 2 } ,
$$

![](images/8d6be36d66a9e40f5b0e70b0a78eae3d36a7cc8d585f94ce5daf06e491a52b64.jpg)

![](images/39e775c1966736c08752e56175b3b03384b0b7e2769292891bc5ff7feba247eb.jpg)

![](images/9258fc37daf99bb0985fedeb6422c4a63a92d39494c8118568c8106e0370d5ce.jpg)  
Fig. 17: Left. Stabilizing trajectories for example scenarios δ. Center. Risk bounds for the quadratic stability with complexity $k = 1$ . Right. Stability margins for each sampled scenario, with red less stable and blue more stable.

where

$$
A _ { 0 } = { \left[ \begin{array} { l l l } { - 0 . 3 } & { 1 . 0 } & { 0 } \\ { - 2 . 0 } & { - 1 . 0 } & { 0 . 2 } \\ { 0 . 2 } & { 0 } & { - 1 . 2 } \end{array} \right] } , \quad A _ { 1 } = { \left[ \begin{array} { l l l } { 0 } & { 0 } & { 0 } \\ { 0 . 5 } & { 0 . 1 5 } & { 0 } \\ { 0 } & { 0 } & { 0 . 3 } \end{array} \right] } , \quad A _ { 2 } = { \left[ \begin{array} { l l l } { 0 } & { 0 } & { 0 } \\ { 0 } & { 0 . 3 } & { 0 . 1 5 } \\ { 0 . 1 5 } & { 0 } & { 0 . 4 5 } \end{array} \right] } .
$$

Here A0 models a well-damped coupled oscillator, A1 perturbs the stiffness of the first mode, and $A _ { 2 }$ perturbs the inter-mode coupling and damping. This maps to the SDP inequality form (7) with decision variable

$$
x = [ p _ { 1 1 } , p _ { 1 2 } , p _ { 1 3 } , p _ { 2 2 } , p _ { 2 3 } , p _ { 3 3 } ] ^ { \top } \in \mathbb { R } ^ { 6 } ,
$$

the six independent entries of the 3 × 3 symmetric Lyapunov matrix $\begin{array} { r } { P = \sum _ { j = 1 } ^ { 6 } x _ { j } \varPi _ { j } } \end{array}$ , where $\pi _ { j }$ are the standard basis matrices for the symmetric 3 × 3 space. The scenario constraint matrices in (7b) are

$$
\begin{array} { r } { F _ { 0 } ( \delta _ { i } ) = 0 , \qquad F _ { j } ( \delta _ { i } ) = A ( \delta _ { i } ) ^ { \top } \pi _ { j } + \pi _ { j } A ( \delta _ { i } ) , \quad j = 1 , \ldots , 6 . } \end{array}
$$

The hard constraint in (7c) enforces $P \succeq 0 . 0 1 \mathbb { I } _ { 3 }$ via

$$
E _ { 0 } = 0 . 0 1 \mathbb { I } _ { 3 } , \qquad E _ { j } = - I I _ { j } , \quad j = 1 , \dots , 6 .
$$

The objective uses $c = [ 1 , 0 , 0 , 1 , 0 , 1 ] ^ { \top }$ and $Q = 0 . 0 0 1 \mathbb { I } _ { 6 }$ , which minimizes trace(P) with no regularization $( \tau = 0 , \rho = 0 )$

Scen-Opt solves the SDP with $N = 5 0 0$ scenarios sampled uniformly over $[ - 1 , 1 ] ^ { 2 }$ . The optimal solution is

$$
P ^ { * } \approx { \left[ \begin{array} { l l l } { 0 . 0 1 0 1 } & { - 0 . 0 0 0 5 } & { 0 . 0 0 0 1 } \\ { - 0 . 0 0 0 5 } & { 0 . 0 1 2 7 } & { - 0 . 0 0 0 3 } \\ { 0 . 0 0 0 1 } & { - 0 . 0 0 0 3 } & { 0 . 0 1 0 0 } \end{array} \right] }
$$

with complexity $k = 1 ,$ , no degeneracy, and risk bounds $\left[ 0 , 0 . 0 4 0 0 \right]$ at 99.9999% confidence. The Lyapunov matrix is not a scalar multiple of the identity: $P _ { 2 2 } ^ { * } = 0 . 0 1 2 7$ is noticeably larger than $P _ { 1 1 } ^ { * }$ and $P _ { 3 3 } ^ { * } \approx 0 . 0 1 0$ , reflecting the need for additional Lyapunov weight on the second state to certify stability across the parameter space. One scenario constraint supports the solution $( k = 1 )$ , corresponding to a near-worst-case parameter realization close to the $( \delta _ { 1 } , \delta _ { 2 } ) = ( 1 , 1 )$ corner where the stability margin is tightest.

## B JSON Program Files for the Illustrative Examples

Instead of entering each matrix through the dialog boxes of the web interface (Section 5), a complete problem can be loaded in one step from a JSON program file through button ${ < } 1 0 >$ in ${ \mathrm { F i g . ~ 4 , ~ } }$ either by uploading the file or by pasting its content. Listings 8–10 give the program files of Examples 1, 2 and 3, taken from the benchmarks folder of the repository. Each listing can be copied and pasted directly into Scen-Opt; after uploading the scenarios through <7>, pressing Solve reproduces the results reported in Appendix $\mathrm { A } .$

The keys follow the notation of Section 3. The key type selects the program class (LP, QP or SDP), and mode selects symbolic or numeric input. The keys c, Q, G and h hold the matrices of the same name, A\_d and b\_d hold A(δ) and b(δ), and F\_d and E hold $F _ { 0 } ( \delta ) , \dots , F _ { d } ( \delta )$ and $E _ { 0 } , \ldots , E _ { d }$ under the keys "0", "1", .... The keys rho, tau, x\_ref and p hold $\rho , \tau , .$ I2 and the norm order $p ,$ and confidence holds $\beta .$ The dimensions d, m and n are given by n\_x, rows\_A and rows\_G for LP and QP, and by n\_x, lmi\_size and lmi\_e\_size for SDP. Matrices are written row by row, and every entry is a string, so that the entries of A(δ), b(δ) and $F _ { j } ( \delta )$ can be expressions in the scenario components delta[0], delta[1], .... The scenarios themselves are not part of the program file.

```jsonl
{
"type": "LP",
"A_d": [["-1", "-1"],
["1", "-1"]],
"b_d": [["delta[0]"], ["-delta[0]"]],
"G": [],
"h": [],
"c": [["O"], ["1"]],
"rho": "0",
"tau": "O",
"x_ref": [],
"p": "",
"confidence": "1e-06",
"mode": "symbolic",
"n_x": 2,
"rows_A": 2,
"rows_G": 0
}
```  
Listing 8: JSON program file for Example 1 (smallest enclosing interval, robust LP; case study in Appendix A.1.4), taken from benchmarks/LP\_half\_width\_2d/data/ program\_symbolic.json. Here delta[0] is the sampled point $\delta _ { i } .$ The fields G and h (no hard constraints) and x\_ref and p (no regularization) are not used in this example: they can be omitted, as in the repository file, or left empty, as shown. The listing can be copied and pasted directly into Scen-Opt.

```jsonl
{
"type": "QP",
"A_d": [["-delta[0]*delta[2]", "-delta[1]*delta[2]", "-delta[2]"]],
"b_d": [["1"]],
"G": [],
"h": [],
"c": [["O"], ["O"], ["O"]],
"Q": [["1", "O", "O"],
["O", "1", "O"],
["O", "O", "O"]],
"rho": "0",
"tau": "0",
"x_ref": [],
"p": "",
"confidence": "1e-06",
"mode": "symbolic",
"n_x": 3,
"rows_A": 1,
"rows_G": 0
}
```

Listing 9: JSON program file for Example 2 (support vector machine, robust QP; Iris case study in Appendix A.2.3), taken from benchmarks/QP\_Iris\_minimal\_3d/data/ program\_symbolic.json. Here delta[0], delta[1] and delta[2] are $\delta _ { i } ( 1 ) , \delta _ { i } ( 2 )$ and $\delta _ { i } ( 3 )$ in (6). The fields G and h (no hard constraints) and x\_ref and p (no regularization) are not used in this example: they can be omitted, as in the repository file, or left empty, as shown. The listing can be copied and pasted directly into Scen-Opt.

```json
{
"type": "SDP",
"c": [["-1"], ["0"], ["-1"]],
"Q": [["O.1", "0", "0"],
["O", "0.1", "0"],
["0", "0", "0.1"]],
"F_d": {
"O": [["O", "O"],
["O", "0"]],
"1": [["O", "1"],
["1", "0"]],
"2": [["-4+0.6*delta[0]", "-1+0.1*delta[0]"],
["-1+0.1*delta[0]", "2"]],
"3": [["0", "0.3*delta[0]-2"],
["0.3*delta[0]-2","-2+0.2*delta[0]"]]
},
"E": {
"O": [["O", "O"],
["O", "0"]],
"1": [["-1", "O"],
["O", "0"]],
"2": [["0", "-1"],
["-1", "0"]],
"3": [["O", "O"],
["0", "-1"]]
},
"rho": "1.0",
"tau": "O",
"x_ref": [],
"p": "",
"confidence": "1e-06",
"mode": "symbolic",
"n_x": 3,
"lmi_size": 2,
"lmi_e_size": 2
}
```

Listing 10: JSON program file for Example 3 (LPV stability, relaxed SDP; case study in Appendix A.3.3), taken from benchmarks/SDP\_LPV\_stability\_3d/data/ program\_symbolic.json. The entries $" 0 " \ \mathrm { t o } \ \mathrm { \Omega } ^ { \mathrm { ~ * ~ } } \mathrm { \Omega }$ of $\mathtt { F \_ d }$ and E are $F _ { 0 } ( \delta ) , \dots , F _ { 3 } ( \delta )$ and $E _ { 0 } , \ldots , E _ { 3 } ,$ and delta[0] is the scalar parameter δ. The fields x\_ref and p (no regularization, $\tau = 0 )$ are not used in this example: they can be omitted, as in the repository file, or left empty, as shown. The listing can be copied and pasted directly into Scen-Opt.

## C Detailed Function Implementation

The callable parameters A\_d, b\_d (LP/QP) and F\_d (SDP) are typically generated by the parsing module (parsing·py), which provides generate\_matrix\_function() and

(e.g. "[[delta[0]+1, 0], [0, delta[1]]]") into evaluable Python functions, and their static counterparts generate\_matrix() and generate\_tensor() for constant matrices. All three solvers call the helper function get\_support () (Section C.5) post-solve to identify support constraints and compute complexity.

## C.1 solve\_1p

def solve\_1p(deltas, A\_d, b\_d, G, h, c, tau=0.0, x\_ref=np.array([0.0]), rho=0.0, norm\_type=2, solver=None, include\_slack=None):

return x\_out, zeta\_out, cost\_out, N, complexity, constraints, degeneracy

## Parameters:

\- deltas: numpy.ndarray of shape (N, K). Each row δi is substituted into A\_d and b\_d to form per-scenario constraints.

\- A\_d: Callable[[numpy.ndarray], numpy.ndarray]. Given a delta row returns an m × n constraint matrix A(δ).

\- b\_d: Callable[[numpy.ndarray], numpy.ndarray]. Given a delta row returns an m × 1 right-hand side vector b(δ).

− G: numpy.ndarray of shape (p, n) (may be an empty array to omit hard constraints).

− h: numpy . ndarray of shape (p, 1) corresponding to G.

\- c: numpy.ndarray of length n (or shape (n, 1)), objective linear coefficients.

\- tau: float. Weight for the regularization term $\| x - x _ { \mathrm { r e f } } \| _ { p } ,$ which is omitted when τ = 0.

\- x\_ref: numpy.ndarray broadcastable to same shape as x (default np.array([0.0])).

\- rho: float. Weight of the penalty $\rho \mathbf { 1 } ^ { \top } \boldsymbol { \zeta }$ on the nonnegative slack variables ζ of length N (one per scenario), which are added when include\_slack is true (or, if it is None, when $\rho \neq 0 )$

\- norm\_type: int, float or str. Order p of the vector norm in the regularization term: any $p \geq 1$ (default 2), "inf", or "fro" (the same as p = 2).

\- solver: Optional[str]. CVXPY solver11 or None (chooses default solver).

— include\_slack: Optional [bool]. Whether to add the slack variables ζ of the relaxation formulation. None (default) adds them only when $\rho \neq 0$

## Return values:

\- x\_out: numpy.ndarray of shape (n, 1). Optimizer solution for x.

\- zeta\_out: numpy . ndarray of shape (N, ). Slack variable values (zeros if no slack variables are used).

\- cost\_out: float. Optimal objective value.

— N: int. Number of rows N in deltas used to build robust constraints.

— complexity: int. Number of scenario constraints in the support list identified post-solve

\- constraints: list of CVXPY constraint objects: the N scenario constraints (the hard constraints are not included).

\- degeneracy: bool. Flag indicating whether degeneracy was detected when computing the support list.

## Exceptions and Error Checking:

\- Raises ValueError if deltas has no rows.

\- Raises ValueError if CVXPY problem status is not "optimal" or "optimal\_inaccurate".

## C.2 solve\_qp

def solve\_qp(deltas, A\_d, b\_d, G, h, c, Q, tau=0.0, x\_ref=np.array([0.0]), rho=0.0, norm\_type=2, solver=None, include\_slack=None):

return x\_out, zeta\_out, cost\_out, N, complexity, constraints, degeneracy

## Parameters:

− deltas: numpy.ndarray of shape (N, K). Each row δi is substituted into A\_d and b\_d to form per-scenario constraints.

\- A\_d: Callable[[numpy.ndarray], numpy.ndarray]. Given a delta row returns an m × n constraint matrix A(δ).

\- b\_d: Callable[[numpy.ndarray], numpy.ndarray]. Given a delta row returns an m × 1 right-hand side vector b(δ).

\- G: numpy.ndarray of shape (p, n) (may be an empty array to omit hard constraints).

− h: numpy. ndarray of shape (p, 1) corresponding to G.

\- c: numpy.ndarray of length n (or shape (n, 1)), objective linear coefficients.

– Q: numpy. ndarray of shape (n, n). Symmetric positive semidefinite quadratic cost matrix.

— tau: float. Weight for the regularization term $\| x - x _ { \mathrm { r e f } } \| _ { p }$ , which is omitted when τ = 0.

\- x\_ref: numpy.ndarray broadcastable to same shape as x (default np.array([0.0])).

\- rho: float. Weight of the penalty ρ1ζ on the nonnegative slack variables ζ of length N (one per scenario), which are added when include\_slack is true (or, if it is None, when $\rho \neq 0 )$

\- norm\_type: int, float or str. Order p of the vector norm in the regularization term: any p ≥ 1 (default 2), "inf ", or "fro" (the same as p = 2).

\- solver: Optional[str]. CVXPY solver or None (chooses default solver).

— include\_slack: Optional [bool]. Whether to add the slack variables ζ of the relaxation formulation. None (default) adds them only when $\rho \neq 0$

## Return values:

\- x\_out: numpy.ndarray of shape (n, 1). Optimizer solution for x.

\- zeta\_out: numpy . ndarray of shape (N, ). Slack variable values (zeros if no slack variables are used).

\- cost\_out: float. Optimal objective value.

\- N: int. Number of rows N in deltas used to build robust constraints.

\- complexity: int. Number of scenario constraints in the support list identified post-solve.

\- constraints: list of CVXPY constraint objects: the N scenario constraints (the hard constraints are not included).

\- degeneracy: bool. Flag indicating whether degeneracy was detected when computing the support list.

## Exceptions and Error Checking:

\- Raises ValueError if deltas has no rows

\- Raises ValueError if CVXPY problem status is not "optimal" or "optimal\_inaccurate".

\- Raises AssertionError if Q is not positive semidefinite or not symmetric.

## C.3 solve\_sdp

def solve\_sdp(deltas, F\_d, E, c, Q, tau=0.0, x\_ref=\texttt{np.array([0.0])}, rho=0.0, norm\_type=2, solver=None, include\_slack=None):

return x\_out, zeta\_out, cost\_out, N, complexity, constraints, degeneracy

## Parameters:

\- deltas: numpy.ndarray of shape (N, K). Each row $\delta _ { i }$ is substituted into F\_d form perscenario constraints.

\- F\_d: Callable[[numpy.ndarray], dict]. Given a delta row returns a dictionary of matrices $\{ k \mapsto F _ { k } ( \delta ) \}$ with keys like $^ { \circ } \circ ^ { , } \circ ^ { , } \mathbb { 1 } ^ { , } , \ldots , ^ { , } \mathbb { d } ^ { , }$ denoting the constant term.

\- E: dict of numpy.ndarray. Optional hard constraint matrices keyed as in F\_d.

– Q: numpy. ndarray of shape (n, n). Symmetric positive semidefinite quadratic cost matrix.

\- c: numpy.ndarray of length n (or shape $( n , 1 ) )$ , objective linear coefficients.

— tau: float. Weight for the regularization term $\| x - x _ { \mathrm { r e f } } \| _ { p }$ , which is omitted when $\tau = 0$

\- x\_ref: numpy.ndarray broadcastable to same shape as x (default np.array([0.0])).

\- rho: float. Weight of the penalty $\rho \mathbf { 1 } ^ { \top } \boldsymbol { \zeta }$ on the nonnegative slack variables ζ of length N (one per scenario), which are added when include\_slack is true (or, if it is None, when $\rho \neq 0 )$

\- norm\_type: int, float or str. Order p of the vector norm in the regularization term: any $p \geq$ 1 (default 2), "inf", or "fro" (the same as $p = 2 )$

\- solver: Optional[str]. CVXPY solver or None (chooses default solver).

— include\_slack: Optional [bool]. Whether to add the slack variables ζ of the relaxation formulation. None (default) adds them only when $\rho \neq 0$

## Return values:

\- x\_out: numpy.ndarray of shape (n, ). Optimizer solution for x. Note: unlike the LP and QP solvers which return a column vector of shape (n, 1), the SDP solver returns a 1-D array.

\- zeta\_out: numpy . ndarray of shape (N, ). Slack variable values (zeros if no slack variables are used).

\- cost\_out: float. Optimal objective value.

— N: int. Number of rows N in deltas used to build robust constraints.

\- complexity: int. Number of scenario constraints in the support list identified post-solve.

\- constraints: list of CVXPY constraint objects: the N scenario constraints (the hard constraints are not included)

\- degeneracy: bool. Flag indicating whether degeneracy was detected when computing the support list.

## Exceptions and Error Checking:

\- Raises ValueError if deltas has no rows

\- Raises ValueError if CVXPY problem status is not "optimal" or "optimal\_inaccurate".

\- Raises AssertionError if Q is not positive semidefinite or not symmetric

\- Raises AssertionError if any F\_d(\delta) matrices are not symmetric.

– Raises AssertionError if any E matrices are not symmetric.

## C.4 quantify\_risk

def quantify\_risk(k,N,beta):

return epsL, epsU

## Parameters:

\- k: int. Complexity, i.e. length of support list.

\- N: int. Number of scenarios.

\- beta: float. User-defined confidence parameter (0 < β < 1).

## Return values:

\- epsL: float. Lower bound of the risk ε.

\- epsU: float. Upper bound of the risk ε.

## Implementation details:

\- Uses the regularized incomplete beta function betainc from jax.scipy.special.

\- Both bounds are computed via binary search with convergence tolerance 1e-10.

\- Special case: if k = N the upper bound is set to 1.0 immediately (all scenarios are support constraints, so no generalization guarantee is possible).

## C.5 get\_support

```python
def get_support(constraints, non_risk_constraints, prob, objective,
solver=None, threshold=1e-8, sol_tol=1e-6):
return complexity, support, degeneracy
```

## Parameters:

\- constraints: list. List of CVXPY scenario constraint objects (one per scenario)

\- non\_risk\_constraints: list (may be empty). Hard constraint(s), always enforced when re-solving and never counted in the complexity.

– prob: CVXPY Problem. The solved problem instance whose solution and duals are used.   
- obiective: CVXPY obiective expression used to re-solve reduced problems.

\- solver: Optional[str]. Choice of CVXPY solver passed to Problem.solve when validating candidate supports.

\- threshold: float (default 1e-8). Dual-value threshold for the initial screening of candidate support-list members (violated or active constraints).

\- sol\_tol: float (default 1e-6). Relative tolerance for judging whether a re-solve reproduces the reference optimal value; set above the solver's value-reproducibility and below the smallest meaningful support contribution.

## Return values:

\- complexity: int. Number of items in the support list.

\- support: list. Final support list of CVXPY constraint objects, i.e., violated or active scenario constraints whose removal changes the optimal value.

\- degeneracy: bool. True if degeneracy was detected during validation/reduction procedures, i.e., when the reduction removes a screened constraint that is active or violated at the optimum, or when the dual screening fails to reproduce the optimum (conservative fallback).

## Implementation details:

Internally uses the helper function test\_support(objective, support, non\_risk, ref, solver) to validate candidate support lists: it re-solves the problem with the candidate constraints plus the always-enforced hard constraints and returns True if the optimal value matches the reference (cf. Section 4).

## Exceptions and Error Checking:

– Raises ValueError if validation fails to produce a consistent support list (suggests trying another solver).