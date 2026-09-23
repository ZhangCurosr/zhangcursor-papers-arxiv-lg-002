# Statistical Rates for Entropic Optimal Transport in the Discrete to SubGaussian Regime

Tomas Gonzalez<sup>1</sup> Gonzalo Mena<sup>2</sup>

<sup>1</sup>Department of Machine Learning, Carnegie Mellon University <sup>2</sup>Department of Statistics & Data Science, Carnegie Mellon University {tcgonzal, gmena}@andrew.cmu.edu

## Abstract

We study statistical rates in entropic optimal transport in the semi-discrete regime where one measure has finite support and the other is subGaussian. Our main result establishes parametric convergence rates for the empirical dual potentials to their population counterparts, with no dimension dependence in the leading term. Our result relies on tailored strong concavity analysis of the semi-dual objective, coupled with specialized bounds for the semi-discrete potentials. As a consequence, we obtain fast rates for downstream quantities derived from the optimal coupling. Chiefly, the empirical barycentric projection achieves a squared-error rate $n ^ { - 1 }$ , matching the fully compact case and improving over the less favorable $n ^ { - 1 / 2 }$ rate known for fully subGaussian settings. Altogether, these results may indicate a lower complexity adaptation phenomenon whereby the statistical complexity of the barycentric projection is governed by the discrete measure. As an application, we analyze Sinkhorn-EM, an EM-type algorithm in which the E-step is replaced by an entropic optimal transport problem. In a well-specified and balanced two-component Gaussian mixture model, we prove n-consistency of the empirical iterates to their population counterparts for any fixed number of iterations, matching classical EM rates <sub>up</sub> <sub>to</sub> <sub>a</sub> √<sub>log</sub> <sub>n</sub> <sub>factor.</sub> <sub>Simulations</sub> <sub>support</sub> <sub>the</sub> <sub>theory.</sub>

## 1 Introduction

Optimal transport (OT) provides a geometrically meaningful way to compare probability distributions. Given measures P and Q and a cost function, the problem is to find the coupling—a joint distribution with marginals P and Q—that minimizes the expected transport cost. The optimal coupling describes how mass is matched, while, under suitable conditions, the Brenier map provides a deterministic way to transport P onto Q. The optimal transport cost, coupling, and map have become central tools across statistics and machine learning, with applications ranging from generative modeling and domain adaptation to single-cell genomics, shape analysis, and distributiona robustness [17, 24]. In most applications, however, at least one of the underlying distributions is typically unknown and must be inferred from finite samples.

Statistical optimal transport studies the accuracy with which transport costs, couplings, and maps can be estimated from such samples. For classical OT, these estimation problems generally sufer from the curse of dimensionality: convergence rates deteriorate rapidly with the ambient dimension $d ,$ and computationally eficient estimators with sharp guarantees are scarce [11]. Entropic optimal transport (EOT) is an attractive alternative that preserves the geometric structure of OT while adding a relative-entropy penalty to the transport objective. As the strength of this regularization vanishes, EOT recovers classical OT under suitable conditions. For positive regularization, it produces a smooth optimal coupling, can be computed eficiently using Sinkhorn’s algorithm, and admits estimators with substantially more favorable sample complexity [4, 17]. Its barycentric projections provide entropic analogues of the Brenier map and, together with the optimal coupling and dual potentials, are important statistical objects in their own right.

Existing finite-sample guarantees for estimating these objects depend strongly on the support assumptions imposed on the two measures. When both measures are compactly supported, Rigollet and Stromme [19] establish parametric squared-error rates of order $n ^ { - 1 }$ for both the empirical cou pling density and its barycentric projections, where n denotes the sample size. To the best of our knowledge, an n<sup>−1</sup> expected squared-error bound for the empirical coupling density has not previously been established in the semi-discrete-to-subGaussian setting considered here. The literature on barycentric projections extends further: a parametric rate is known in the compact semi-discrete setting [18], while Masud et al. [12] obtain an n $^ { - 1 / 2 }$ squared-error rate when the source measure is subGaussian and the target measure is compactly supported. When both measures may be sub-Gaussian and unbounded, Werenski et al. [27] establish slower rates. This leaves open a natural intermediate question:

Can parametric rates for the coupling density and barycentric projections be recovered when one measure has finite support while the other is subGaussian and unbounded?

We answer this question afirmatively. Our central result establishes dimension-independent convergence rates for the empirical dual potentials in the discrete-to-subGaussian regime. This potential level result yields a parametric squared-error rate for the density of the empirical optimal coupling and, in turn, parametric convergence rates for both barycentric projections. Our analysis exploits the discrete marginal, which allows the corresponding dual potential to be represented by a finite dimensional vector. Our proof strategy is closest to that of Pooladian et al. [18], combining strong concavity of the semi-dual objective with empirical-process arguments. A key ingredient in this approach is uniform control of the dual potentials, for which existing bounds do not extend suitably to an unbounded subGaussian marginal.

Beyond its fundamental importance in the theory of statistical EOT, the discrete-to-subGaussian regime arises naturally in the analysis of the Sinkhorn expectation-maximization (Sinkhorn-EM) algorithm [14, 13] and is closely related to the setup studied in the classical work on deterministic annealing for clustering [20]. For Gaussian mixture models, Sinkhorn-EM replaces the usual likelihood-based update with an EOT problem between the discrete measure supported on the current mixture centers and the continuous data distribution. While previous work has demonstrated its practical relevance and studied its population behavior, its finite-sample properties remain poorly understood [14, 13]. Our EOT results yield the first guarantee on how accurately sample-based Sinkhorn-EM iterates approximate their population counterparts, providing a first step toward a broader theoretical understanding of the algorithm.

Contributions. We make the following contributions.

• In Theorem 1, we establish n<sup>−1</sup> expected squared-error rates for the empirical entropic dual potentials in the discrete-to-subGaussian regime, with dimension-independent leading constants and an exponentially small remainder.

• We prove an $n ^ { - 1 }$ squared-error rate for the empirical optimal coupling density (Theorem 2), as well as rates of order $n ^ { - 1 }$ and $d / n$ for the backward and forward barycentric projections, respectively (Theorem 3). A matching minimax lower bound shows that the $n ^ { - 1 }$ rate for the backward projection is optimal (Proposition 1).

• We establish a strong-concavity bound for the semi-dual objective and a dimension-independent uniform bound for the discrete-side dual potential (Propositions 2 and 3), which are key to obtain the results mentioned above.

• As an application, we give the first finite-sample guarantee for Sinkhorn-EM in a balanced, symmetric two-component Gaussian mixture, proving n<sup>−1/2</sup>-consistency of the empirical it- $^ { - 1 / 2 } .$ erates with their population counterparts for any fixed number of iterations (Theorem 4).

• We corroborate the predicted convergence rates through simulations in Section 5.

Related work. The statistical theory of optimal transport is extensive; we refer to [3] for a comprehensive treatment. In the unregularized setting, dimension-dependent rates are known for both Wasserstein distances and transport maps. In particular, H¨utter and Rigollet [11] study estimation of the unregularized Brenier map between absolutely continuous measures under regularity assumptions on the transport map and smoothness assumptions on the densities. Under suficient smoothness, their squared- $L ^ { 2 }$ risk can attain an $n ^ { - 1 / 2 }$ rate whose exponent is independent of the ambient dimension. This is a diferent statistical problem from the one considered here: we study fixed entropic regularization with a finitely supported marginal and an unbounded subGaussian marginal, and obtain guarantees for the dual potentials, coupling density, and barycentric projections without smoothness assumptions on the densities.

Statistical analyses of entropic optimal transport, popularized by Cuturi [4], initially focused primarily on transport costs, establishing sample-complexity bounds and central limit theorems under compact-support and subGaussian assumptions [8, 15]. Statistical guarantees for finer EOT objects are more limited. When both marginals are compactly supported, Rigollet and Stromme [19] establish parametric squared-error rates of order $n ^ { - 1 }$ for the empirical dual potentials, coupling density, and barycentric projections. In the compact semi-discrete setting, Pooladian et al. [18] obtain an $n ^ { - 1 }$ rate for the empirical EOT barycentric projection at fixed regularization, as an intermediate step toward estimating the unregularized Brenier map. When the source is subGaussian and the target is compactly supported, Masud et al. [12] obtain an $n ^ { - 1 / 2 }$ squared-error rate. More general noncompact settings are considered by Werenski et al. [27], who obtain slower rates under compactness or strong log-concavity assumptions on the target. Table 1 summarizes the available finite-sample rates for estimating the EOT barycentric projection at fixed regularization and situates our result within this literature. To the best of our knowledge, an $n ^ { - 1 }$ expected squared-error bound for the (canonically extended) empirical coupling density has not previously been established in this semi-discrete-to-subGaussian regime.

Lower-complexity adaptation has been studied for transport costs in both the unregularized and entropic settings. For classical OT costs, Hundrieser et al. [10] prove that empirical rates can adapt to the lower-complexity marginal. Groppe and Hundrieser [9] establish analogous adaptation results for EOT costs. These results concern transport costs, whereas our semi-discrete analysis concerns finer EOT objects, namely the dual potentials, coupling density, and barycentric projections.

Finally, our Sinkhorn-EM application connects the paper to the literature on expectation-maximization (EM) algorithms for mixture models. Finite-sample analyses of EM for Gaussian mixtures are well developed but technically delicate, relying on careful control of sample-based EM updates and their deviations from population iterates [29, 28, 1, 6, 25, 7]. Sinkhorn-EM replaces the likelihood-based EM update with an EOT-based update and has been studied primarily at the population level [16, 14, 13]. We provide its first finite-sample guarantee in a balanced, symmetric two-component Gaussian mixture. For any fixed number of iterations, the sample-based iterates approximate their population counterparts at an n<sup>−</sup> $- 1 / 2$ rate, matching the corresponding sample-based EM guarantee up to logarithmic factors and problem-dependent constants.

<table><tr><td>Work</td><td>Source Measure</td><td>Target Measure</td><td> $\mathrm { S q u a r e d } { - } L ^ { 2 }$  rate</td></tr><tr><td>Ours</td><td>SubGaussian</td><td>Finite support, with masses bounded away from zero</td><td> $n ^ { - 1 }$ </td></tr><tr><td>Ours (lower bound)</td><td>SubGaussian</td><td>Two atoms, each with mass at least  $1 / 4$ </td><td> $\Omega ( n ^ { - 1 } )$ </td></tr><tr><td>Pooladian et al. [18]</td><td>bounded above and below</td><td>Compact, with density Finite support, with masses bounded away from zero</td><td> $n ^ { - 1 }$ </td></tr><tr><td>Masud et al. [12]</td><td>SubGaussian</td><td>Compact support</td><td> $n ^ { - 1 / 2 }$ </td></tr><tr><td>Werenski et al. [27]</td><td>SubGaussian</td><td>Compact support or strongly log-concave</td><td> $n ^ { - 1 / 3 }$ </td></tr><tr><td>Rigollet and Stromme [19] Compact support</td><td></td><td>Compact support</td><td> $n ^ { - 1 }$ </td></tr></table>

Table 1: Finite-sample rates for estimating the EOT barycentric projection at fixed regularization. We report only the dependence on the sample size n, treating the regularization parameter and other problem parameters as fixed. In our notation, the map goes from $Q$ to P. The lower bound is a minimax bound; all other rows report upper bounds.

## 2 Problem setup and notation

The entropic optimal transport problem between measures P (target) and $Q$ (source) is written as [17, 3]

$$
S ( P , Q ) : = \operatorname* { i n f } _ { \pi \in \Pi ( P , Q ) } \Big \{ \int \int c ( x , y ) \mathrm { d } \pi ( x , y ) + \sigma ^ { 2 } \mathrm { K L } ( \pi \| P \otimes Q ) \Big \} ,\tag{1}
$$

where $c ( x , y ) = | | x - y | | ^ { 2 } / 2$ is the quadratic cost, and $\sigma ^ { 2 } > 0$ is a fixed regularization parameter. The infimum is attained by a unique $\pi ^ { \star } \in \Pi ( P , Q )$ in the transportation polytope $\Pi ( P , Q )$ , the set of joint distributions with prescribed marginals P and $Q )$ , and strong duality holds in the sense that [17, 3]

$$
S ( P , Q ) = \operatorname * { s u p } _ { ( f , g ) \in L ^ { \infty } ( P ) \times L ^ { \infty } ( Q ) } \Phi ( f , g ) ,\tag{2}
$$

where $\Phi ( f , g ) = \Phi ( f , g , P , Q , \sigma )$ is defined as

$$
\Phi ( f , g ) : = \int f ( x ) \mathrm { d } { \cal P } ( x ) + \int g ( y ) \mathrm { d } Q ( y ) - \sigma ^ { 2 } \int \int ( e ^ { ( f ( x ) + g ( y ) - c ( x , y ) ) / \sigma ^ { 2 } } - 1 ) \mathrm { d } ( { \cal P } \otimes Q ) ( x , y ) .\tag{3}
$$

The supremum in (2) is attained at a pair $( f ^ { * } , g ^ { * } ) \in L ^ { \infty } ( P ) \times L ^ { \infty } ( Q )$ of dual potentials, which are unique up to the translation $( f ^ { * } , g ^ { * } ) \mapsto ( f ^ { * } + a , g ^ { * } - a )$ for $a \in \mathbb { R }$ . To avoid degeneracies, we will always assume the following gauge constraint

$$
\mathbb { E } _ { P } ( f ^ { * } ( X ) ) = 0 .\tag{4}
$$

The first-order conditions for $f ^ { * } , g ^ { * }$ write as [3, 5]

$$
f ^ { * } ( x ) = - \sigma ^ { 2 } \ln \Big ( \int e ^ { ( g ^ { * } ( y ) - c ( x , y ) ) / \sigma ^ { 2 } } \mathrm { d } Q ( y ) \Big ) \quad P \mathrm { - a . s , }\tag{5a}
$$

$$
g ^ { * } ( y ) = - \sigma ^ { 2 } \ln \Big ( \int e ^ { ( f ^ { * } ( x ) - c ( x , y ) ) / \sigma ^ { 2 } } \mathrm { d } P ( x ) \Big ) \quad Q _ { \mathrm { - a . s . } }\tag{5b}
$$

By replacing $g ^ { * }$ as a function of $f ^ { * }$ , the last term in (3) cancels out, and we arrive at the semi-dual formulation:

$$
S ( P , Q ) = \operatorname* { s u p } _ { f \in L ^ { \infty } ( P ) } \Phi ( f ) ,\tag{6}
$$

where $\Phi ( f ) = \Phi ( f , P , Q , \sigma )$ is the semi-dual functional

$$
\Phi ( f ) : = \operatorname* { s u p } _ { g \in L ^ { \infty } ( Q ) } \Phi ( f , g ) = - \sigma ^ { 2 } \int \ln \Big ( \int e ^ { ( f ( x ) - c ( x , y ) ) / \sigma ^ { 2 } } \mathrm { d } P ( x ) \Big ) \mathrm { d } Q ( y ) + \int f ( x ) \mathrm { d } P ( x ) .
$$

## Joint, conditionals and barycentric projections

The optimal plan admits the following Radon-Nikodym density $p ^ { * } \left[ 1 9 \right]$

$$
p ^ { * } ( x , y ) : = \frac { \mathrm { d } \pi ^ { * } } { \mathrm { d } ( P \otimes Q ) } ( x , y ) = \exp \left( \frac { f ^ { * } ( x ) + g ^ { * } ( y ) - c ( x , y ) } { \sigma ^ { 2 } } \right) , \quad P \otimes Q \mathrm { - a . s . }\tag{7}
$$

We define the forward and backward barycentric projections, $\vec { T } ^ { * } , \overleftarrow { T } ^ { * }$ , respectively, as follows:

$$
\vec { T } ^ { * } ( x ) : = \mathbb { E } _ { \pi ^ { * } } \left( Y | X = x \right) = \int y \mathrm { d } \pi ^ { * } ( y | x ) = \int y p ^ { * } ( x , y ) \mathrm { d } Q ( y ) , \quad P \mathrm { - a . s . } ,\tag{8a}
$$

$$
\overleftarrow T ^ { * } ( y ) : = \mathbb E _ { \pi ^ { * } } \left( X | Y = y \right) = \int x \mathrm { d } \pi ^ { * } ( x | y ) = \int x p ^ { * } ( x , y ) \mathrm { d } P ( x ) , \quad Q \mathrm { - a . s . }\tag{8b}
$$

where the rightmost sides above follow from the condition that $\pi ^ { * } \in \Pi ( P , Q )$ . The distinction between forward and backward projection is relevant in our case because of the inherently asymmetric regime that we consider. In the following, unless there is ambiguity, we will drop the ∗ superscripts to denote optimal objects and will refer to them simply as $f , g , \pi , p , \overrightarrow { T } , \overleftarrow { T }$

## Semi-discrete to subGaussian setup

We make the following two assumptions about the distributions.

(A) For the discrete measure, we assume that

$$
P = \sum _ { k = 1 } ^ { K } \alpha _ { k } \delta _ { x _ { k } } , \quad \alpha _ { k } \geq 0 , \quad \sum _ { k = 1 } ^ { K } \alpha _ { k } = 1 , \quad \| x _ { k } \| \leq R , k \in [ K ] ,
$$

and $\underline { { \alpha } } : = \mathrm { m i n } _ { k } \alpha _ { k } > 0 .$

(B) We assume that $Y - \mathbb { E } Y$ is $\varepsilon ^ { 2 } .$ -subGaussian; i.e., if $Y \sim Q$ , then for all $v \in \mathbb { R } ^ { d }$

$$
\mathbb { E } e ^ { \langle v , Y - \mathbb { E } Y \rangle } \le e ^ { \varepsilon ^ { 2 } \| v \| ^ { 2 } / 2 } .
$$

That is, letting $B ( x , R )$ denote the ball centered at x of radius R, we assume that P is a nondegenerate mixture of K (fixed) atoms in the ball $B ( 0 , R )$ , and that, after centering, Q is subGaussian with proxy $\varepsilon ^ { 2 }$ . As we discuss in the appendix A we can always assume that Q is centered at the cost of expressing all bounds in terms of $\begin{array} { r } { \widetilde { R } : = \operatorname* { m a x } _ { k \in \left[ K \right] } \| x _ { k } - \mathbb { E } Y \| \leq R + \| \mathbb { E } Y \| } \end{array}$ instead of R if we assume a uniform bound on EY . In this discrete-to-subGaussian case, we can identify the potential f with a vector $f _ { k } = f ( x _ { k } )$ , and so the semidual functional Φ is simply a multivariate function $\Phi : \mathbb { R } ^ { K }  \mathbb { R }$

$$
\Phi ( f ) = \sum _ { k = 1 } ^ { K } f _ { k } \alpha _ { k } - \sigma ^ { 2 } \int \mathrm { l n } \Big ( \sum _ { k = 1 } ^ { K } \alpha _ { k } e ^ { ( f _ { k } - c ( x _ { k } , y ) ) / \sigma ^ { 2 } } \Big ) \mathrm { d } Q ( y ) .\tag{9}
$$

## Empirical setup, canonical extensions

We will investigate rates for the empirical objects arising when replacing P and Q by the empirical measures

$$
P _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { X _ { i } } , \quad Q _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { Y _ { i } } ,
$$

where $X _ { 1 } , \ldots , X _ { n } \stackrel { \mathrm { i . i . d . } } { \sim } P$ and $Y _ { 1 } , \dots , Y _ { n } \stackrel { \mathrm { i . i . d . } } { \sim } Q$ , with the two samples $( X _ { i } ) _ { i = 1 } ^ { n }$ and $( Y _ { i } ) _ { i = 1 } ^ { n }$ independent of each other. The resulting objects $f _ { n } , g _ { n } , p _ { n } , \vec { T } _ { n }$ and $\overleftarrow { T } _ { n }$ are defined in principle only $P _ { n } , Q _ { n }$ or $P _ { n } \otimes Q _ { n }$ almost surely. These can be extended to functions of the entire ambient space through the canonical extensions [15, Proposition 6]. To do so, note that the right-hand sides in (5a) and (5b) define functions on $\mathbb { R } ^ { d }$ . Consistent with our semi-discrete motivation, we will treat $g _ { n } , \overleftarrow { T } _ { n } : \mathbb { R } ^ { d }  \mathbb { R }$ as functions, $f _ { n } , \vec { T } _ { n } \in \mathbb { R } ^ { K }$ as vectors expressing evaluations at fixed support points $x _ { 1 } , \ldots x _ { k }$ , and $p _ { n } ( x _ { k } , \cdot ) : \mathbb { R } ^ { d }  \mathbb { R }$ as a function for each $k \in [ K ]$

## 3 Estimation rates for EOT-related objects

In this section, we present statistical rates for diferent objects in the setting described above. Later, in 3.1 we discuss the main ingredients used to establish these results, including new bounds that are of independent interest. We start by establishing the convergence of the potentials

Theorem 1. Let $P , Q$ satisfy conditions (A) and (B). Let $( f _ { n } , g _ { n } )$ and $( f , g )$ be the optimal entropic potentials for the problems $( P _ { n } , Q _ { n } )$ and $( P , Q )$ , respectively, where $f _ { n }$ and f satisfy the gauge condition (4), $i . e . , \mathbb { E } _ { P } f _ { n } ( X ) = \mathbb { E } _ { P } f ( X ) = 0$ . Then

$$
\mathbb { E } \| f _ { n } - f \| _ { L ^ { \infty } ( P ) } ^ { 2 } \leq \frac { C } { n } + r _ { n , d } , \qquad \mathbb { E } \| g _ { n } - g \| _ { L ^ { 2 } ( Q ) } ^ { 2 } \leq \frac { C } { n } .
$$

Here $C = C ( K , \underline { { \alpha } } , \widetilde { R } , \varepsilon , \sigma ^ { 2 } )$ is independent of d. The remainder $r _ { n , d }$ satisfies $r _ { n , d } \leq C _ { 2 } \exp ( - c n )$ ， where $C _ { 2 } = C _ { 2 } ( d , K , \underline { { \alpha } } , \widetilde { R } , \varepsilon , \sigma ^ { 2 } )$ and $c = c ( K , \underline { { \alpha } } )$ are two positive constants.

The exponentially small, dimension-dependent remainder $r _ { n , d }$ term in Theorem 1 arises only from the event that $P _ { n }$ fails to charge all atoms of P. On this event, the empirical potentials are intrinsically defined only on $\operatorname { s u p p } ( P _ { n } )$ , and the term $r _ { n , d }$ appears from (perhaps sub-optimal) control of the canonical extension on missing atoms. We view this dependence as a technical artifact of the fixed-support formulation used in the theorem.

We note in the following corollary, that this term disappears in the one-sample setting, where the discrete marginal P is kept fixed, and it can also be avoided in the two-sample setting by formulating the bounds intrinsically on the empirical support, for example, using the $L ^ { \infty } ( P _ { n } )$ norm.

Corollary 1. Let $f , f _ { n }$ be as in Theorem 1. Then

$$
\mathbb { E } \| f _ { n } - f \| _ { L ^ { \infty } ( P _ { n } ) } ^ { 2 } \leq \frac { C } { n } .
$$

In the one-sample setting, let $f _ { n }$ instead denote the optimal discrete-side potential for $( P , Q _ { n } )$ , with P kept fixed and $\mathbb { E } _ { P } f _ { n } ( X ) = 0$ . Then

$$
\mathbb { E } \Vert f _ { n } - f \Vert _ { L ^ { \infty } ( P ) } ^ { 2 } \leq \frac { C } { n } .
$$

Additionally, we have the following convergence result for the joint density

Theorem 2. Denote by p and $p _ { n }$ the densities of the optimal population and empirical couplings, $\pi , \pi _ { n }$ , with respect to $P \otimes Q$ and $P _ { n } \otimes Q _ { n }$ , respectively, as defined in (7). Then,

$$
\mathbb { E } \| p - p _ { n } \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 2 } \leq \frac { C } { n } .
$$

where $\begin{array} { r } { \| h \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 2 } : = \operatorname* { m a x } _ { k \in [ K ] } \int | h ( x _ { k } , y ) | ^ { 2 } \mathrm { d } Q ( y ) . } \end{array}$

Furthermore, we can state the following convergence bounds for the forward and backward barycentric projections. This result will be used to obtain the first convergence rate for the sample-based Sinkhorn Expectation-Maximization algorithm, presented in Section 4.

Theorem 3. Let $\vec { T } , \overleftarrow { T }$ be the population barycentric projections associated with $( P , Q )$ , and let $\vec { T } _ { n } , \overleftarrow { T } _ { n }$ be the empirical barycentric projections associated with $( P _ { n } , Q _ { n } )$ . Then,

$$
\mathbb { E } \| \overleftarrow { T } - \overleftarrow { T } _ { n } \| _ { L ^ { \infty } ( Q ) } ^ { 2 } \leq \frac { C } { n } , \quad \mathbb { E } \| \overrightarrow { T } - \overrightarrow { T } _ { n } \| _ { L ^ { \infty } ( P ) } ^ { 2 } \leq \frac { C d } { n } .\tag{10}
$$

The following lower bound is inspired by [18], who provide a lower bound for the non-regularized map.

Proposition 1 (Minimax lower bound). Fix $R > 0 , \varepsilon > 0 , \sigma ^ { 2 } > 0$ , and $M \geq 0$ . Let C be the class of pairs $( P , Q )$ satisfying Assumptions $( A ) \ – ( B )$ with these fixed parameters, $K = 2$ , minimum atom mass at least $1 / 4$ , and $\| \mathbb { E } _ { Q } Y \| \leq M$ . Then, for every integer $n \geq 1$ 2

$$
\operatorname* { i n f } _ { \widehat { T } } \ \operatorname* { s u p } _ { ( P , Q ) \in \mathcal { C } } \mathbb { E } _ { P ^ { n } \otimes Q ^ { n } } \left[ \| \widehat { T } - \overleftarrow { T } _ { P , Q } \| _ { L ^ { 2 } ( Q ) } ^ { 2 } \right] \geq \frac { R ^ { 2 } } { 6 4 n } ,
$$

where $\overleftarrow { T } _ { P , Q }$ is the backward barycentric projection and the infimum is over all estimators based on n independent samples from each of P and Q, with the two samples independent. Consequently, the $n ^ { - 1 }$ squared-error rate for the backward barycentric projection in Theorem 3 is minimax optimal over ${ \mathcal { C } } ,$ since $\| h \| _ { L ^ { 2 } ( Q ) } \leq \| h \| _ { L ^ { \infty } ( Q ) }$

The proof of the proposition above is given in Appendix A.7. We conclude this section with a sketch of the proof of Theorem 1, emphasizing the main technical ingredients and the two intermediate estimates on which the convergence rates rely.

## 3.1 High-level strategy

A similar result had previously been established in the bounded-to-bounded [19] and bounded-todiscrete case [18]. Although none of the arguments in these papers directly extend to our setup, we followed the route inspired by the latter [18], which roughly follows two steps: establishing strong concavity of the semi-dual objective Φ, and empirical process arguments that enable control of $\Phi ( f ) - \Phi ( f _ { n } )$ at the rate $n ^ { - 1 }$ . In order to extend this argument, we require control over the Hessian of Φ in our setup. Specifically, we show that

Proposition 2. Suppose that $f \in \mathbb { R } ^ { K }$ satisfies $\| f \| _ { \infty } \leq L$ (not necessarily an optimal potential). Then,

$$
\begin{array} { r } { \nabla ^ { 2 } \Phi ( f ) \preceq - \kappa \Big ( I _ { K } - \frac { 1 } { K } \mathbf { 1 1 } ^ { \top } \Big ) , } \end{array}
$$

with

$$
\kappa = \frac { 1 } { \sigma ^ { 2 } } K \underline { { \alpha } } ^ { 2 } \exp \Big ( - \left( 4 L + \widetilde { R } ^ { 2 } + 8 \widetilde { R } ^ { 2 } \varepsilon ^ { 2 } / \sigma ^ { 2 } \right) / \sigma ^ { 2 } \Big ) .
$$

In other words, the smallest non-zero eigenvalue $o f - \nabla ^ { 2 } \Phi ( f )$ is bounded below by κ.

Note that the above bound depends on the norm $\| f \| _ { \infty }$ . As we will apply this bound to empirical quantities, we require estimates on this norm. This is achieved with the following proposition

Proposition 3. Let f be the optimal entropic potential for the problem $( P , Q )$ satisfying (4). Then, with the shortcut $\| f \| _ { \infty } = \| f \| _ { L ^ { \infty } ( P ) }$ , we have

$$
\| f \| _ { \infty } \ \leq \ \widetilde { R } ^ { 2 } \ + \ \sigma ^ { 2 } \log \biggl ( \frac { 1 } { \underline { { { \alpha } } } } \biggr ) \ + \ \frac { 2 \varepsilon ^ { 2 } } { \sigma ^ { 2 } } \widetilde { R } ^ { 2 } .
$$

This result extends the bound shown in [15] in the subGaussian-to-subGaussian case. Specifically, from [15, Proposition 6] it follows that (see Proposition 6)

$$
\| f \| _ { \infty } \lesssim \widetilde { R } ^ { 2 } + d ( \varepsilon ^ { 2 } + \widetilde { R } ^ { 2 } ) + d ^ { 2 } ( \varepsilon ^ { 2 } + \widetilde { R } ^ { 2 } ) ^ { 2 } .\tag{11}
$$

However, this bound still depends on d. As shown in the proof sketch below, using such a dimensiondependent bound would not be suficient to establish the rate $n ^ { - 1 }$ in a dimension-free manner. We leave all details to the appendix.

Proof sketch of Theorem 1. The proof combines two ingredients: a strong concavity property of the semi-dual objective and localized empirical process bounds controlling fluctuations of the empirical semi-dual. Specifically, by a standard M-estimator argument [26] and Proposition 2

$$
\begin{array} { r } { \mathbb { E } \| f _ { n } - f \| _ { \infty } ^ { 2 } \lesssim \kappa ^ { - 1 } ( f _ { n } , f ) \left( \Phi ( f ) - \Phi ( f _ { n } ) \right) , } \end{array}
$$

where $\kappa ( f _ { n } , f )$ is the one in Proposition 2 that can be established uniformly over $f _ { n } , f .$ . In turn, such uniformity arises from bounds on $\| f _ { n } \| _ { \infty } , \| f \| _ { \infty }$ . By Proposition 3, $\| f _ { n } \| _ { \infty } , \| f \| _ { \infty } ,$ can be controlled in a dimension-independent way, albeit with dependence on a sample version εeof the subGaussianity parameter $\varepsilon ,$ following the argument in [15]. Using novel high-probability and moment bounds on such random $\widetilde { \varepsilon }$ (Lemma 5 in the Appendix), we can ignore the low-probability event where $\widetilde { \varepsilon }$ is large and therefore treat the quantity $\kappa ^ { - 1 } ( f _ { n } , f )$ as a deterministic and of the order

$$
\kappa ^ { - 1 } ( f _ { n } , f ) \approx \exp \Bigl ( \widetilde { R } ^ { 2 } ( 1 + \varepsilon ^ { 2 } ) \Bigr ) ,
$$

where the above notation emphasizes dependence on ${ \widetilde { R } } ^ { 2 }$ and $\varepsilon ^ { 2 }$ . Finally, the term $\Phi ( f ) - \Phi ( f _ { n } )$ is controlled at the $n ^ { - 1 }$ rate by careful localization arguments similar to the ones in [18]. □

## 4 Application: convergence of sample-based Sinkhorn-EM

We now apply the results of the previous section to obtain the first finite-sample convergence guarantee for Sinkhorn-EM.

## 4.1 Background on Sinkhorn-EM

Expectation-maximization (EM) is a standard algorithm for latent-variable models. For mixture models, its E-step computes the posterior responsibility of each mixture component for each observation, while its M-step updates the model parameters using these responsibilities. Sinkhorn expectation-maximization (Sinkhorn-EM) [16, 14, 13] replaces the usual likelihood-based responsibilities with those induced by an entropic optimal transport problem. In particular, rather than assigning responsibilities to each observation independently, Sinkhorn-EM enforces the prescribed mixture weights at the level of the aggregate coupling.

Previous work has found that Sinkhorn-EM can converge faster or more reliably than classical EM in some empirical settings [14, 13]. Its population objective can also have a more favorable optimization geometry and avoid some undesirable stationary-point configurations of the likelihood optimized by classical EM [13]. To the best of our knowledge, however, no finite-sample convergence guarantee is available for the sample-based Sinkhorn-EM algorithm. Our goal is not to provide a complete theoretical explanation for its improved empirical behavior, but rather to take a first step toward understanding it. In the same spirit as classical analyses of EM, we study Sinkhorn-EM in a canonical Gaussian-mixture test bed for which the sample-based and population iterations can be compared explicitly [29, 1, 25].

Specifically, we consider a two-component Gaussian mixture with known spherical covariance:

$$
Q ^ { * } = \alpha { \mathcal { N } } ( \theta ^ { * } , \sigma ^ { 2 } I _ { d } ) + ( 1 - \alpha ) { \mathcal { N } } ( - \theta ^ { * } , \sigma ^ { 2 } I _ { d } ) ,\tag{12}
$$

where $\theta ^ { * } \in \mathbb { R } ^ { d }$ is the unknown signal and $\alpha \in ( 0 , 1 )$ and $\sigma ^ { 2 }$ are known. For a candidate parameter $\theta ,$ define

$$
P _ { \theta } = \alpha \delta _ { \theta } + ( 1 - \alpha ) \delta _ { - \theta } .
$$

Sinkhorn-EM minimizes the EOT objective

$$
\mathcal { L } ( \theta ) : = S ( P _ { \theta } , Q ) ,\tag{13}
$$

where $Q = Q ^ { * }$ in the population problem and $Q = Q _ { n }$ in the sample-based problem. Since $Q ^ { * }$ is obtained by convolving $P _ { \theta ^ { * } }$ with a Gaussian of covariance $\sigma ^ { 2 } I _ { d } .$ , the population objective $S ( P _ { \theta } , Q ^ { * } )$ is minimized at $\theta ^ { * }$ . The EOT regularization parameter $\sigma ^ { 2 }$ therefore coincides with the Gaussian noise variance in the mixture model.

We first describe classical EM in this model. Starting from $\theta _ { \mathrm { E M } } ^ { 0 }$ , the population iterates satisfy

$$
\theta _ { \mathrm { E M } } ^ { t + 1 } = F ( \theta _ { \mathrm { E M } } ^ { t } , \alpha ) ,
$$

where

$$
F ( \theta , \alpha ) : = \mathbb { E } _ { Y \sim Q ^ { * } } \left[ Y \big ( 2 \Psi ( Y , \theta , \alpha ) - 1 \big ) \right]
$$

and

$$
\Psi ( y , \theta , \alpha ) : = \frac { \alpha \exp \left( - \frac { \| y - \theta \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } { \alpha \exp \left( - \frac { \| y - \theta \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) + ( 1 - \alpha ) \exp \left( - \frac { \| y + \theta \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } .
$$

Here $\Psi ( y , \theta , \alpha )$ is the posterior responsibility of the component centered at θ for the observation $y .$

Given i.i.d. observations $Y _ { 1 } , \dots , Y _ { n } \sim Q ^ { * }$ , let

$$
Q _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { Y _ { i } } .
$$

The corresponding empirical EM map and iterates are

$$
F _ { n } ( \theta , \alpha ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Y _ { i } \bigl ( 2 \Psi ( Y _ { i } , \theta , \alpha ) - 1 \bigr ) , \qquad \widehat { \theta } _ { \mathrm { E M } } ^ { t + 1 } = F _ { n } ( \widehat { \theta } _ { \mathrm { E M } } ^ { t } , \alpha ) .
$$

Sinkhorn-EM uses the same update map but replaces the fixed weight α appearing in the responsibilities with a transport-corrected weight. Let $f _ { n } ( \theta ) \ : = \ : ( f _ { n , 1 } ( \theta ) , f _ { n , 2 } ( \theta ) ) \in \mathbb { R } ^ { 2 }$ be an optimal semi-dual potential for $S ( P _ { \theta } , Q _ { n } )$ , and define

$$
\alpha _ { n } ( \theta ) : = \frac { \alpha \exp ( f _ { n , 1 } ( \theta ) / \sigma ^ { 2 } ) } { \alpha \exp ( f _ { n , 1 } ( \theta ) / \sigma ^ { 2 } ) + ( 1 - \alpha ) \exp ( f _ { n , 2 } ( \theta ) / \sigma ^ { 2 } ) } .\tag{14}
$$

The sample-based Sinkhorn-EM iterates are then

$$
\widehat { \theta } _ { \mathrm { S E M } } ^ { t + 1 } = F _ { n } \big ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } , \alpha _ { n } ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } ) \big ) .\tag{15}
$$

Similarly, let $f ( \theta ) = ( f _ { 1 } ( \theta ) , f _ { 2 } ( \theta ) ) \in \mathbb { R } ^ { 2 }$ be an optimal semi-dual potential for $S ( P _ { \theta } , Q ^ { * } )$ , and define

$$
\alpha ( \theta ) : = \frac { \alpha \exp ( f _ { 1 } ( \theta ) / \sigma ^ { 2 } ) } { \alpha \exp ( f _ { 1 } ( \theta ) / \sigma ^ { 2 } ) + ( 1 - \alpha ) \exp ( f _ { 2 } ( \theta ) / \sigma ^ { 2 } ) } .\tag{16}
$$

The population Sinkhorn-EM iterates satisfy

$$
\theta _ { \mathrm { S E M } } ^ { t + 1 } = F \big ( \theta _ { \mathrm { S E M } } ^ { t } , \alpha ( \theta _ { \mathrm { S E M } } ^ { t } ) \big ) .\tag{17}
$$

Thus, in this two-component model, the diference between EM and Sinkhorn-EM is the replacement of the fixed mixture weight α by the transport-corrected weights $\alpha ( \theta )$ and $\alpha _ { n } ( \theta )$

## 4.2 Finite-sample guarantee for Sinkhorn-EM

We now specialize to the balanced symmetric model, corresponding to (12) with $\alpha = 1 / 2$ . We prove that, for any fixed number of iterations, sample-based Sinkhorn-EM tracks population Sinkhorn-EM at the parametric n<sup>−</sup> $\cdot 1 / 2$ scale. Combining this statistical approximation with local contraction

of the population update gives an error bound consisting of an optimization term and a statistical term.

Let $\rho : = { \frac { \| \theta ^ { * } \| } { \sigma } }$ denote the signal-to-noise ratio. There exists a universal constant $\eta _ { 0 }$ such that, whenever $\rho \geq \eta _ { 0 }$ , the population EM operator is a contraction on $B ( \theta ^ { * } , \| \theta ^ { * } \| / 4 )$ with contraction factor $\kappa \leq e ^ { - c \rho ^ { 2 } } < 1$ for a universal constant $c > 0 \ [ 1 ]$ . Define

$$
A _ { 1 } : = \frac { \exp ( 2 \rho ^ { 2 } ) } { \operatorname* { m i n } \{ \rho , 1 \} ^ { 2 } } , \qquad A _ { 2 } : = \frac { 1 6 C _ { \theta ^ { * } , \sigma } ^ { 2 } } { ( 1 - \kappa ) ^ { 2 } \| \theta ^ { * } \| ^ { 2 } } ,
$$

where $C _ { \theta ^ { * } , \sigma }$ is the constant appearing in Proposition 5. We have the following theorem.

Theorem 4. Let $Q ^ { * }$ be as in (12) with $\alpha = 1 / 2$ and ${ { \theta } ^ { * } } \ne 0$ , and let $Y _ { 1 } , \ldots , Y _ { n } \stackrel { \mathrm { i . i . d . } } { \sim } Q ^ { * }$ . Suppose that $\rho \geq \eta _ { 0 }$ and that the initialization $\widehat { \theta } _ { \mathrm { S E M } } ^ { 0 }$ of Sinkhorn-EM satisfies $\begin{array} { r } { \left. \widehat { \theta } _ { \mathrm { S E M } } ^ { 0 } - \theta ^ { * } \right. \leq \frac { 1 } { 4 } \| \theta ^ { * } \| } \end{array}$ . There exists a universal constant $C > 0$ such that, if

$$
n \geq C \operatorname* { m a x } _ { j \in \{ 1 , 2 \} } \left\{ A _ { j } \left[ d \log ( e d A _ { j } ) + \log \frac { 1 } { \delta } \right] \right\} ,
$$

then, with probability at least $1 - \delta - n ^ { - c _ { 1 } d } - c _ { 2 } n ^ { - 2 }$ , all the iterates of Sinkhorn-EM $\widehat { \theta } _ { \mathrm { S E M } } ^ { 0 } , . . . , \widehat { \theta } _ { \mathrm { S E M } } ^ { T }$ 2 defined in (15), remain in $B ( \theta ^ { * } , \| \theta ^ { * } \| / 4 )$ and satisfy

$$
\left\| \widehat { \theta } _ { \mathrm { S E M } } ^ { \widehat { t } } - \theta ^ { * } \right\| \leq \kappa ^ { t } \left\| \widehat { \theta } _ { \mathrm { S E M } } ^ { 0 } - \theta ^ { * } \right\| + \frac { C _ { \theta ^ { * } , \sigma } } { 1 - \kappa } \sqrt { \frac { d \log n + \log ( 1 / \delta ) } { n } }
$$

for every $0 \leq t \leq T$ , where $c _ { 1 } , c _ { 2 } > 0$ are constants depending only on $\theta ^ { * }$ and $\sigma .$

The first term is the optimization error inherited from contraction of the population operator, while the second is the statistical error incurred by replacing the population distribution with its empirical measure. In particular, for any fixed number of iterations, the empirical iterates are $n ^ { - 1 / 2 }$ -consistent with their population counterparts. The dependence on n and d matches the corresponding samplebased EM guarantee up to the factor $\sqrt { \log n }$ and problem-dependent constants [1]. The additional terms $n ^ { - c _ { 1 } d }$ and $c _ { 2 } n ^ { - 2 }$ in the failure probability arise from the high-probability events used to control the empirical subGaussian parameters and vanish polynomially with the sample size.

The first step in proving Theorem 4 is to show that population Sinkhorn-EM coincides with classical EM in the balanced symmetric model.

Proposition 4 (Population Sinkhorn-EM and EM iterates coincide for balanced symmetric two– component Gaussian mixtures). Suppose that (12) holds with $\alpha = 1 / 2$ and ${ { \theta } ^ { * } } \ne 0$ . Let $\theta _ { \mathrm { S E M } } ^ { t + 1 } =$ $F \big ( \theta _ { \mathrm { S E M } } ^ { t } , \alpha ( \theta _ { \mathrm { S E M } } ^ { t } ) \big )$ and $\theta _ { \mathrm { E M } } ^ { t + 1 } = F ( \theta _ { \mathrm { E M } } ^ { t } , 1 / 2 )$ denote the population Sinkhorn-EM and EM iterates, respectively. If the two algorithms start from the same point, $\theta _ { \mathrm { S E M } } ^ { 0 } = \theta _ { \mathrm { E M } } ^ { 0 }$ , then $\theta _ { \mathrm { S E M } } ^ { t } = \theta _ { \mathrm { E M } } ^ { t }$ for every $t \geq 0$

The proof uses the fact that, in this balanced symmetric model, the population EM responsibilities already satisfy the marginal constraints imposed by EOT; see Section C. Consequently, Proposition 4 allows us to invoke existing contraction results for population EM [1]. The sample-based algorithms do not generally coincide, so the main step is to control the uniform statistical error

$$
\eta _ { n } : = \operatorname* { s u p } _ { \theta \in B ( \theta ^ { * } , \| \theta ^ { * } \| / 4 ) } \| F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha ( \theta ) ) \| .
$$

Proposition 5 (Uniform approximation of the sample-based Sinkhorn-EM update). Let $Q ^ { * }$ be as in (12) with $\alpha = 1 / 2$ and $\theta ^ { * } \neq 0$ . For each $\theta \in B ( \theta ^ { * } , \| \theta ^ { * } \| / 4 )$ , let $\alpha ( \theta )$ and $\alpha _ { n } ( \theta )$ denote the population and empirical Sinkhorn weights. There exist constants $C , c _ { 1 } , c _ { 2 } > 0 $ , depending only on σ and $\lVert \theta ^ { * } \rVert$ , such that, if

$$
n \geq C A _ { 1 } \left[ d \log ( e d A _ { 1 } ) + \log \frac { 1 } { \delta } \right] ,
$$

then, with probability at least $1 - \delta - n ^ { - c _ { 1 } d } - c _ { 2 } n ^ { - 2 }$

$$
\operatorname* { s u p } _ { \theta \in B ( \theta ^ { \ast } , \| \theta ^ { \ast } \| / 4 ) } \| F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha ( \theta ) ) \| \leq C _ { \theta ^ { \ast } , \sigma } \sqrt { \frac { d \log n + \log ( 1 / \delta ) } { n } } .
$$

The proof follows the general sample-to-population strategy used in finite-sample analyses of EM [1, 7, 25], with an additional term arising from the estimation of the transport-corrected weight. Let $\mathcal { D } _ { n } = ( Y _ { 1 } , \ldots , Y _ { n } )$ denote the observed sample. We decompose

$$
\begin{array} { r l } & { F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha ( \theta ) ) = F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha _ { n } ( \theta ) ) } \\ & { \phantom { F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) } + F ( \theta , \alpha _ { n } ( \theta ) ) - \mathbb { E } _ { \mathcal { D } _ { n } } \left[ F ( \theta , \alpha _ { n } ( \theta ) ) \right] } \\ & { \phantom { F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) } + \mathbb { E } _ { \mathcal { D } _ { n } } \left[ F ( \theta , \alpha _ { n } ( \theta ) ) \right] - F ( \theta , \alpha ( \theta ) ) . } \end{array}
$$

Write $\mathcal { B } : = B ( \theta ^ { * } , \| \theta ^ { * } \| / 4 )$ . For the first term, Corollary 4 and lemma 10 imply that, with probability at least $1 - n ^ { - c _ { 1 } d } - C _ { 1 } ^ { \prime } n ^ { - 2 }$

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \left\| F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha _ { n } ( \theta ) ) \right\| \leq C _ { 1 } \sqrt { \frac { d \log n } { n } } .\tag{18}
$$

Here the concentration bound holds uniformly in both the parameter and the weight, so it remains valid at the data-dependent weight $\alpha _ { n } ( \theta )$ . For the second term, Proposition 7 gives, with probability at least $1 - \delta - C _ { 2 } ^ { \prime } n ^ { - 2 }$ 2

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \| F ( \theta , \alpha _ { n } ( \theta ) ) - \mathbb { E } _ { \mathcal { D } _ { n } } F ( \theta , \alpha _ { n } ( \theta ) ) \| \le C _ { 2 } \sqrt { \frac { d \log ( e n ) + \log ( 1 / \delta ) } { n } } .\tag{19}
$$

The constants depend only on σ and $\lVert \theta ^ { * } \rVert$ ; in particular, one may take $C _ { 2 } = C$ max $\{ \sigma , \| \theta ^ { * } \| \}$ exp $( C \rho ^ { 2 } )$ , where C depends on the weight bound in Lemma 10. The third term is the bias of the population

update evaluated at the empirical weight. It is controlled by the one-sample backward-barycentricprojection bound from Theorem 3 and corollary 1. To see this, let $Y \sim Q ^ { * }$ be independent of $\mathcal { D } _ { n }$ Then

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathcal { D } _ { n } } \left[ F ( \theta , \alpha _ { n } ( \theta ) ) \right] - F ( \theta , \alpha ( \theta ) ) } \\ & { \quad = \mathbb { E } _ { \mathcal { D } _ { n } , Y } \left[ Y \left\{ \left( 2 \Psi ( Y , \theta , \alpha _ { n } ( \theta ) ) - 1 \right) - \left( 2 \Psi ( Y , \theta , \alpha ( \theta ) ) - 1 \right) \right\} \right] . } \end{array}
$$

Moreover, if $\overleftarrow { T } _ { n } ^ { \theta }$ and $\overleftarrow { T } ^ { \theta }$ denote the backward barycentric projections for $( P _ { \theta } , Q _ { n } )$ and $( P _ { \theta } , Q ^ { * } )$ ), respectively, then

$$
{ \overleftarrow { T } } _ { n } ^ { \theta } ( Y ) - { \overleftarrow { T } } ^ { \theta } ( Y ) = \theta \left\{ \left( 2 \Psi ( Y , \theta , \alpha _ { n } ( \theta ) ) - 1 \right) - \left( 2 \Psi ( Y , \theta , \alpha ( \theta ) ) - 1 \right) \right\} .
$$

It follows that

$$
\begin{array} { r l } & { \left\| \mathbb { E } _ { \mathcal { D } _ { n } } \left[ F ( \theta , \alpha _ { n } ( \theta ) ) \right] - F ( \theta , \alpha ( \theta ) ) \right\| } \\ & { \quad \leq \cfrac { 1 } { \| \theta \| } \left( \mathbb { E } \| Y \| ^ { 2 } \right) ^ { 1 / 2 } \bigg ( \mathbb { E } \left\| \overleftarrow { T } _ { n } ^ { \theta } ( Y ) - \overleftarrow { T } ^ { \theta } ( Y ) \right\| ^ { 2 } \bigg ) ^ { 1 / 2 } . } \end{array}
$$

For $\theta \in B ( \theta ^ { * } , \| \theta ^ { * } \| / 4 )$ 2

$$
\frac { 1 } { \| \theta \| } \leq \frac { 4 } { 3 \| \theta ^ { * } \| } , \qquad \mathbb { E } \| Y \| ^ { 2 } = \| \theta ^ { * } \| ^ { 2 } + d \sigma ^ { 2 } .
$$

Furthermore, the one-sample version of Theorem 3, with $P _ { \theta }$ fixed, gives

$$
\mathbb { E } \left\| \overleftarrow { T } _ { n } ^ { \theta } ( Y ) - \overleftarrow { T } ^ { \theta } ( Y ) \right\| ^ { 2 } \leq \frac { C _ { \mathrm { b a c k } } } { n } ,
$$

where $C _ { \mathrm { b a c k } }$ can be chosen uniformly over the local basin and is independent of d. Consequently,

$$
| | \mathbb { E } _ { \mathcal { D } _ { n } } \left[ F ( \theta , \alpha _ { n } ( \theta ) ) \right] - F ( \theta , \alpha ( \theta ) ) | | \le \frac { 4 \sqrt { C _ { \mathrm { b a c k } } } } { 3 } \sqrt { \frac { | | \theta ^ { * } | | ^ { 2 } + d \sigma ^ { 2 } } { n | | \theta ^ { * } | | ^ { 2 } } } = O \left( \sqrt { \frac { 1 + d / \rho ^ { 2 } } { n } } \right) .
$$

The last display is uniform over B and is at most $C _ { 3 } \sqrt { d / n }$ , since $d \geq 1$ and $\rho > 0$ is fixed. Combining it with (18) and (19) by the triangle inequality and a union bound gives, with probability at least $1 - \delta - n ^ { - c _ { 1 } d } - c _ { 2 } n ^ { - 2 }$

$$
\begin{array} { l } { \eta _ { n } \leq C _ { 1 } \sqrt { \displaystyle \frac { d \log n } { n } } + C _ { 2 } \sqrt { \displaystyle \frac { d \log ( e n ) + \log ( 1 / \delta ) } { n } } + C _ { 3 } \sqrt { \displaystyle \frac { d } { n } } } \\ { \qquad \leq C _ { \theta ^ { * } , \sigma } \sqrt { \displaystyle \frac { d \log n + \log ( 1 / \delta ) } { n } } , } \end{array}
$$

where the last inequality uses $n \geq 2$ , and $c _ { 2 } = C _ { 1 } ^ { \prime } + C _ { 2 } ^ { \prime }$ . This proves Proposition 5. Combining this uniform update bound with population contraction and the basin-stability argument in Section C proves Theorem 4.

## 5 Simulations

Experimental setup. We complement our convergence theorem with empirical simulations of the convergence of the empirical entropic backward barycentric projection in a setting where the population map is available in closed form. We consider population measures of the form

$$
P = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \delta _ { x _ { k } } , \qquad Q = P \ast \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { d } ) .
$$

In the fixed- $K = 2$ experiments, the centers are chosen deterministically with first coordinates R and $- R$ and the rest 0. When K is varied, the centers are sampled uniformly from the Euclidean ball $B ( 0 , R )$ , including in the $K = 2$ case. For this pair $( P , Q )$ , the population entropic optimizer has dual potential $f = 0$ , up to additive constants. Therefore, the population conditional probabilities of the discrete atoms given $Y = y$ are given by

$$
\rho _ { k } ( y ) : = \pi ( X = x _ { k } \mid Y = y ) = \frac { \exp ( - c ( x _ { k } , y ) / \sigma ^ { 2 } ) } { \sum _ { \ell = 1 } ^ { K } \exp ( - c ( x _ { \ell } , y ) / \sigma ^ { 2 } ) } ,
$$

and the population backward barycentric projection is

$$
\overleftarrow { T } ( y ) = \sum _ { k = 1 } ^ { K } x _ { k } \rho _ { k } ( y ) .
$$

For each sample size $n ,$ we draw independent samples $X _ { 1 } , \dots , X _ { n } \sim P$ and $Y _ { 1 } , \dots , Y _ { n } \sim Q$ , and form the empirical measures $P _ { n } , Q _ { n }$ . The samples from $P$ and $Q$ are not coupled. We compute the empirical entropic backward barycentric projection $\overleftarrow { T } _ { n }$ by solving the empirical semi-dual between $P _ { n }$ and $Q _ { n }$ and finding the optimal empirical potential $f _ { n }$ . Given $f _ { n } = ( f _ { n , 1 } , \ldots , f _ { n , n } )$ , we define the empirical conditional probabilities, through the canonical extension, by

$$
\rho _ { n , i } ( y ) : = \frac { \exp ( ( f _ { n , i } - c ( X _ { i } , y ) ) / \sigma ^ { 2 } ) } { \sum _ { j = 1 } ^ { n } \exp ( ( f _ { n , j } - c ( X _ { j } , y ) ) / \sigma ^ { 2 } ) } ,
$$

and

$$
\overleftarrow { T } _ { n } ( y ) = \sum _ { i = 1 } ^ { n } X _ { i } \rho _ { n , i } ( y ) .
$$

We estimate

$$
\mathbb { E } \Vert \overleftarrow { T } - \overleftarrow { T } _ { n } \Vert _ { L ^ { 2 } ( Q ) } ^ { 2 } = \mathbb { E } \left[ \int \Vert \overleftarrow { T } ( y ) - \overleftarrow { T } _ { n } ( y ) \Vert ^ { 2 } \mathrm { d } Q ( y ) \right]
$$

by Monte Carlo. For each repetition, we draw fresh training samples from $P ^ { n }$ and $Q ^ { n }$ , solve the empirical semi-dual, and approximate the conditional $L ^ { 2 } ( Q )$ error using an independent evaluation sample $\widetilde { Y } _ { 1 } , \dots , \widetilde { Y } _ { M } \sim Q$ :

$$
\widehat { \mathcal { E } } _ { n } = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \left. \overleftarrow { T } ( \widetilde { Y } _ { m } ) - \overleftarrow { T } _ { n } ( \widetilde { Y } _ { m } ) \right. ^ { 2 } .
$$

We repeat this procedure 20 times and report the average of $\widehat { \mathcal { E } } _ { n }$ . Error bars correspond to standard errors across repetitions. We use $M = 5 0 0 0$ evaluation samples.

In all plots, $d \in \{ 1 , 2 , 5 , 1 0 , 2 0 \}$ and $n \in \{ 1 0 , 2 0 , 5 0 , 1 0 0 , 2 0 0 , 5 0 0 , 1 0 0 0 \}$ . We perform three parameter sweeps: varying $R \in \{ 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ with $K = 2$ and $\sigma = 0 . 5$ , using deterministic antipodal atoms; varying $\sigma \in \{ 0 . 0 5 , 0 . 5 , 1 , 2 \}$ with $K = 2$ and $R = 1$ , using deterministic antipodal atoms; and varying $K \in \{ 2 , 2 0 , 2 0 0 , 2 0 0 0 \}$ with $R = 1$ and $\sigma = 0 . 5$ , using atoms sampled uniformly from $B ( 0 , R )$ . All results are reported on log-log plots, with n on the x-axis and the estimated squared error $\mathbb { E } \Vert \overleftarrow { T } - \overleftarrow { T } _ { n } \Vert _ { L ^ { 2 } ( Q ) } ^ { 2 }$ on the y-axis.

Results. Since the plots are in log-log scale, a decrease of one order of magnitude in error when n increases by one order of magnitude is visually consistent with an $n ^ { - 1 }$ rate, while a decrease of roughly half an order of magnitude is consistent with an $n ^ { - 1 / 2 }$ rate. Figure 1 summarizes the three parameter sweeps. Overall, the $R \mathrm { - }$ and σ-sweeps indicate that the fast regime is most visible when the problem is well conditioned: for large R or small $\sigma ,$ the observed decay is closer to $n ^ { - 1 / 2 }$ over the tested range, whereas for smaller R or larger $\sigma ,$ the curves become closer to $n ^ { - 1 }$

The top row of Figure 1 varies $R .$ . Larger R leads to larger errors, which is consistent with the geometric scaling of the two-atom problem. Indeed, since

$$
\overleftarrow { T }  ( y ) = x _ { 2 } + \rho _ { 1 } ( y ) ( x _ { 1 } - x _ { 2 } ) ,
$$

an error in estimating $\rho _ { 1 } ( y )$ produces squared barycentric error proportional to $\| x _ { 1 } - x _ { 2 } \| ^ { 2 } = ( 2 R ) ^ { 2 }$

The middle row varies $\sigma .$ Larger $\sigma$ gives cleaner convergence because the conditional probabilities depend on $\exp ( - c / \sigma ^ { 2 } )$ : small $\sigma$ makes the map closer to a hard assignment rule, while larger $\sigma$ smooths the weights and reduces sensitivity to empirical fluctuations.

The bottom row varies K. Increasing K mainly increases the constants while preserving a similar qualitative decay in n, suggesting that the lower-bound assumption on $\underline { { \alpha } } = 1 / K$ may be conservative in these benign uniform examples, although the experiments do not rule out a genuine worst-case dependence.

Main takeaway. Across the K-sweep, and in the better-conditioned regimes of the $R \mathrm { - }$ and $\sigma -$ sweeps, the empirical decay is visually consistent with the theoretically predicted $n ^ { \cdot }$ <sup>−1</sup>-like behavior.

![](images/ff7283c8d456ff0179d796b63588ceb3c14131193f3f355f5c9a839c07264fa5.jpg)  
Figure 1: Estimated squared error $\mathbb { E } \Vert \overleftarrow { T } - \overleftarrow { T } _ { n } \Vert _ { L ^ { 2 } ( Q ) } ^ { 2 }$ as a function of n. Top row: varying $R \in$ $\{ 0 . 0 1 , 0 . 1 , 1 , 1 0 \}$ with $K = 2 , \sigma = 0 . 5$ , and deterministic antipodal atoms. Middle row: varying $\sigma \in \{ 0 . 0 5 , 0 . 5 , 1 , 2 \}$ with $K = 2 , R = 1$ , and deterministic antipodal atoms. Bottom row: varying $K \in \{ 2 , 2 0 , 2 0 0 , 2 0 0 0 \}$ with $R = 1 , \sigma = 0 . 5$ , and atoms sampled uniformly from $B ( 0 , R )$ . Each curve corresponds to a diferent dimension d.

## 6 Conclusion and future work

Our results provide further evidence of lower-complexity adaptation (LCA) in entropic optima transport. We show that, when one marginal is discrete and the other is subGaussian, the empirical dual potentials and joint coupling densities achieve parametric squared-error rates of order $( n ^ { - 1 } )$ . The same rate holds for both barycentric projections, but with an intrinsic asymmetry: the projection onto the discrete marginal has a dimension-free leading constant, whereas the projection onto the subGaussian marginal carries a linear dependence on the ambient dimension through the moment of Y . Thus, noncompactness of one marginal alone does not force the slower rates known in fully subGaussian settings. Rather, in the semi-discrete regime, the statistical complexity of the coupling and of the projection onto the simpler marginal is governed primarily by the discrete measure.

Our Hessian lower bound in Proposition 2 is inspired by [22, Lemma 6], which establishes a lower bound on the Hessian of the semi-dual functional in the unregularized case under a Poincar´e assumption on the densities. However, our proof approach is substantially diferent; we don’t rely on a Cheeger inequality-based bound but instead use a perhaps cruder one. As a result, downstream bounds have an exponential dependence on $R , \varepsilon$ and $\sigma ^ { - 1 }$ . An open question is whether using more sophisticated spectral analysis machinery may yield an improved rate κ and, therefore, improved constants downstream. Additionally, it is unclear whether the n<sup>−</sup> $- 1 / 2$ rates known for fully subGaussian settings are intrinsic or reflect limitations of current analyses. Finally, our Sinkhorn EM analysis is limited to a well-specified, balanced, symmetric two-component Gaussian mixture under suitable initialization; extending it to broader mixture models and misspecified settings is an important direction.

## 7 Acknowledgements

GM is supported by NSF-DMS 2412895. We thank Tao Wang for help on a previous version of this work. We also thank Shayan Hundrieser for helpful discussions. Part of this work was done while TG was a visiting graduate student in the Federated and Collaborative Learning program at Simons Institute.

## References

[1] S. Balakrishnan, M. J. Wainwright, and B. Yu. Statistical guarantees for the EM algorithm: from population to sample-based analysis. Ann. Statist., 45(1):77–120, 2017.

[2] S. Boucheron, G. Lugosi, and O. Bousquet. Concentration inequalities. In Summer school on machine learning, pages 208–240. Springer, 2003.

[3] S. Chewi, J. Niles-Weed, and P. Rigollet. Statistical optimal transport. Springer, 2025.

[4] M. Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. In Advances in Neural Information Processing Systems, volume 26, 2013.

[5] M. Cuturi and G. Peyr´e. Semidual regularized optimal transport. SIAM Review, 60(4):941– 965, 2018.

[6] C. Daskalakis, C. Tzamos, and M. Zampetakis. Ten steps of em sufice for mixtures of two gaussians. In Conference on Learning Theory, pages 704–710. PMLR, 2017.

[7] R. Dwivedi, N. Ho, K. Khamaru, M. J. Wainwright, M. I. Jordan, and B. Yu. Singularity, misspecification and the convergence rate of em. The Annals of Statistics, 48(6):3161–3182, 2020.

[8] A. Genevay, L. Chizat, F. Bach, M. Cuturi, and G. Peyr´e. Sample complexity of sinkhorn divergences. In The 22nd international conference on artificial intelligence and statistics, pages 1574–1583. PMLR, 2019.

[9] M. Groppe and S. Hundrieser. Lower complexity adaptation for empirical entropic optimal transport. Journal of Machine Learning Research, 25(344):1–55, 2024.

[10] S. Hundrieser, T. Staudt, and A. Munk. Empirical optimal transport between diferent measures adapts to lower complexity. In Annales de l’Institut Henri Poincare (B) Probabilites et statistiques, volume 60, pages 824–846. Institut Henri Poincar´e, 2024.

[11] J.-C. H¨utter and P. Rigollet. Minimax estimation of smooth optimal transport maps. The Annals of Statistics, 49(2):1166–1194, 2021.

[12] S. B. Masud, M. Werenski, J. M. Murphy, and S. Aeron. Multivariate soft rank via entropyregularized optimal transport: Sample eficiency and generative modeling. Journal of Machine Learning Research, 24(160):1–65, 2023.

[13] G. Mena. On model-based clustering with entropic optimal transport. 2026.

[14] G. Mena, A. Nejatbakhsh, E. Varol, and J. Niles-Weed. Sinkhorn em: An expectationmaximization algorithm based on entropic optimal transport. arXiv preprint arXiv:2006.16548, 2020.

[15] G. Mena and J. Niles-Weed. Statistical bounds for entropic optimal transport: sample complexity and the central limit theorem. Advances in neural information processing systems, 32, 2019.

[16] A. Nejatbakhsh, E. Varol, E. Yemini, O. Hobert, and L. Paninski. Probabilistic joint segmentation and labeling of c. elegans neurons. In Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part V 23, pages 130–140. Springer, 2020.

[17] G. Peyr´e and M. Cuturi. Computational Optimal Transport, volume 11. Foundations and Trends in Machine Learning, 2019.

[18] A.-A. Pooladian, V. Divol, and J. Niles-Weed. Minimax estimation of discontinuous optimal transport maps: The semi-discrete case. In International Conference on Machine Learning, pages 28128–28150. PMLR, 2023.

[19] P. Rigollet and A. J. Stromme. On the sample complexity of entropic optimal transport. The Annals of Statistics, 53(1):61–90, 2025.

[20] K. Rose. Deterministic annealing for clustering, compression, classification, regression, and related optimization problems. Proceedings of the IEEE, 86(11):2210–2239, 1998.

[21] H. P. Rosenthal. On the subspaces of l p (p¿ 2) spanned by sequences of independent random variables. Israel Journal of Mathematics, 8(3):273–303, 1970.

[22] R. Sadhu, Z. Goldfeld, and K. Kato. Stability and statistical inference for semidiscrete optimal transport maps. The Annals of Applied Probability, 34(6):5694–5736, 2024.

[23] A. W. Van der Vaart. Asymptotic statistics, volume 3. Cambridge university press, 2000.

[24] C. Villani. Optimal Transport: Old and New. Springer, 2009.

[25] N. Weinberger and G. Bresler. The em algorithm is adaptively-optimal for unbalanced symmetric gaussian mixtures. Journal of Machine Learning Research, 23(103):1–79, 2022.

[26] J. Wellner et al. Weak convergence and empirical processes: with applications to statistics. Springer Science & Business Media, 2013.

[27] M. Werenski, J. M. Murphy, and S. Aeron. Estimation of entropy-regularized optimal transport maps between non-compactly supported measures. arXiv preprint arXiv:2311.11934, 2023.

[28] Y. Wu and H. H. Zhou. Randomly initialized em algorithm for two-component gaussian mixture achieves near optimality in o $( { \sqrt { n } } )$ iterations. Mathematical Statistics & Learning, 4, 2021.

[29] J. Xu, D. J. Hsu, and A. Maleki. Global analysis of expectation maximization for mixtures of two gaussians. Advances in Neural Information Processing Systems, 29, 2016.

## A Proofs of main results in Section 3

## A.1 Preliminaries

In this section we will repeatedly work with the following seminorms. For $u \in \mathbb { R } ^ { K }$ we define

$$
\operatorname { V a r } _ { \infty } ( u ) : = \operatorname* { i n f } _ { c \in \mathbb { R } } \operatorname* { m a x } _ { k \in [ K ] } | u _ { k } - c | ^ { 2 } = { \frac { 1 } { 4 } } \left( \operatorname* { m a x } _ { k \in [ K ] } u _ { k } - \operatorname* { m i n } _ { k \in [ K ] } u _ { k } \right) ^ { 2 } ,\tag{20}
$$

$$
\mathrm { V a r } _ { \alpha } ( u ) : = \operatorname* { i n f } _ { c \in \mathbb { R } } \sum _ { k = 1 } ^ { K } \alpha _ { k } | u _ { k } - c | ^ { 2 } = \sum _ { k = 1 } ^ { K } \alpha _ { k } \left( u _ { k } - \sum _ { j = 1 } ^ { K } \alpha _ { j } u _ { j } \right) ^ { 2 } ,\tag{21}
$$

and $\operatorname { V a r } ( u ) = \operatorname { V a r } _ { \widetilde { \alpha } } ( u )$ with $\widetilde { \alpha } _ { k } = 1 / K$

By working with $\mathrm { V a r } _ { \infty } ( u )$ instead of $\| u \| _ { \infty } ^ { 2 }$ we are able to handle with degeneracies arising from the fact that optimal potentials are only defined up to constant shifts. We will repeatedly use the following elementary relations among the above defined seminorms, as well as their relation with the usual $\| u \| _ { \infty } ^ { 2 }$

Lemma 1. Denote ${ \overline { { \alpha } } } = \operatorname* { m a x } _ { k \in [ K ] } \alpha _ { k }$ . We have that

$$
K \underline { { \alpha } } \mathrm { V a r } ( u ) \leq \mathrm { V a r } _ { \alpha } ( u ) \leq K \overline { { \alpha } } \mathrm { V a r } ( u )\tag{22}
$$

and

$$
\underline { { \alpha } } \mathrm { V a r } _ { \infty } ( u ) \leq \mathrm { V a r } _ { \alpha } ( u ) \leq \mathrm { V a r } _ { \infty } ( u ) .\tag{23}
$$

Further,

$$
\begin{array} { r } { \mathrm { V a r } _ { \infty } ( u ) \leq \| u \| _ { \infty } ^ { 2 } , } \end{array}\tag{24}
$$

and moreover, if

$$
\operatorname* { m i n } _ { k \in [ K ] } u _ { k } \leq 0 \leq \operatorname* { m a x } _ { k \in [ K ] } u _ { k } ,
$$

then

$$
\| u \| _ { \infty } ^ { 2 } \leq 4 \mathrm { V a r } _ { \infty } ( u ) .\tag{25}
$$

Throughout the proofs, we will work in the centered coordinates. More precisely, let $\mu : = \mathbb { E } Y$ Since the quadratic cost is invariant under common translations,

$$
c ( x , y ) = c ( x - \mu , y - \mu ) ,
$$

the entropic optimal transport problem between $( P , Q )$ is equivalent to the one between the translated measures

$$
P ^ { \mu } : = \sum _ { k = 1 } ^ { K } \alpha _ { k } \delta _ { x _ { k } - \mu } , \qquad Q ^ { \mu } : = \operatorname { L a w } ( Y - \mu ) .
$$

The measure $Q ^ { \mu }$ is centered ε<sup>2</sup>-subGaussian by assumption, while $P ^ { \mu }$ is supported in the ball of radius

$$
{ \widetilde { R } } : = \operatorname* { m a x } _ { k \in [ K ] } \| x _ { k } - \mathbb { E } ( Y ) \| \leq R + \| \mathbb { E } Y \| .
$$

Thus, without loss of generality, the proofs are carried out under the normalization $\mathbb { E } Y ~ = ~ 0$ Therefore, in what follows, we will assume that Y is centered. To account for this, we write all bounds using $\widetilde { R }$ instead of R in the main text.

## A.2 Proof of Proposition 2

Proof. Define ${ \widetilde { \Phi } } : = - \Phi$ . We will bound the derivatives of this function instead. By diferentiating (9) with respect to coordinates $f _ { k }$ we have

$$
\nabla \widetilde { \Phi } ( f ) = \int \omega ( y ) \mathrm { d } Q ( y ) - \alpha
$$

where

$$
\omega _ { k } ( y ) : = \frac { s _ { k } ( y ) } { \sum _ { k ^ { \prime } = 1 } ^ { K } s _ { k ^ { \prime } } ( y ) } , \qquad s _ { k } ( y ) : = \alpha _ { k } e ^ { ( f _ { k } - c ( x _ { k } , y ) ) / \sigma ^ { 2 } } .
$$

Diferentiating once more yields

$$
\nabla ^ { 2 } \widetilde { \Phi } ( f ) = \frac { 1 } { \sigma ^ { 2 } } \int \left( \mathrm { D i a g } ( \omega ( y ) ) - \omega ( y ) \omega ( y ) ^ { \top } \right) \mathrm { d } Q ( y ) .
$$

Note that for each y the following identity holds for $\omega = \omega ( y )$

$$
{ \mathrm { D i a g } } ( \omega ) - \omega \omega ^ { \top } = \sum _ { 1 \leq k < k ^ { \prime } \leq K } \omega _ { k } \omega _ { k ^ { \prime } } ( e _ { k } - e _ { k ^ { \prime } } ) ( e _ { k } - e _ { k ^ { \prime } } ) ^ { \top } ,
$$

where $e _ { k }$ is the k-th unit vector. Therefore,

$$
\nabla ^ { 2 } \widetilde { \Phi } ( f ) = \frac { 1 } { \sigma ^ { 2 } } \sum _ { k < k ^ { \prime } } a _ { k k ^ { \prime } } ( e _ { k } - e _ { k ^ { \prime } } ) ( e _ { k } - e _ { k ^ { \prime } } ) ^ { \top } , \quad \mathrm { w h e r e } \quad a _ { k k ^ { \prime } } : = \int \omega _ { k } ( y ) \omega _ { k ^ { \prime } } ( y ) \mathrm { d } Q ( y ) .
$$

Consequently, since for any $u \in \mathbb { R } ^ { K }$ we have

$$
\boldsymbol { u } ^ { \intercal } ( \boldsymbol { e } _ { k } - \boldsymbol { e } _ { k ^ { \prime } } ) ( \boldsymbol { e } _ { k } - \boldsymbol { e } _ { k ^ { \prime } } ) ^ { \intercal } \boldsymbol { u } = ( u _ { k } - u _ { k ^ { \prime } } ) ^ { 2 } ,
$$

we get the following lower bound

$$
u ^ { \top } \nabla ^ { 2 } \widetilde { \Phi } ( f ) u = \frac { 1 } { \sigma ^ { 2 } } \sum _ { k < k ^ { \prime } } a _ { k k ^ { \prime } } ( u _ { k } - u _ { k ^ { \prime } } ) ^ { 2 } \geq \frac { 1 } { \sigma ^ { 2 } } a _ { * } \sum _ { k < k ^ { \prime } } ( u _ { k } - u _ { k ^ { \prime } } ) ^ { 2 } ,
$$

where $\begin{array} { r } { a _ { * } : = \operatorname* { m i n } _ { k < k ^ { \prime } } a _ { k k ^ { \prime } } } \end{array}$ . Now, take $v \in { \mathbf { 1 } } ^ { \perp } \subseteq \mathbb { R } ^ { K }$ , i.e. $\textstyle \sum _ { k } v _ { k } = 0$ . Then, we have that

$$
\begin{array} { r c l } { \displaystyle \sum _ { k < k ^ { \prime } } ( v _ { k } - v _ { k ^ { \prime } } ) ^ { 2 } } & { = } & { \displaystyle \frac { 1 } { 2 } \left[ \sum _ { k ^ { \prime } , k } v _ { k } ^ { 2 } - 2 \sum _ { k ^ { \prime } } v _ { k ^ { \prime } } \sum _ { k } v _ { k } + \sum _ { k ^ { \prime } , k } v _ { k ^ { \prime } } ^ { 2 } \right] } \\ & { = } & { K \| v \| ^ { 2 } . } \end{array}
$$

Therefore, for such $v \in { { \bf 1 } ^ { \perp } }$

$$
v ^ { \top } \nabla ^ { 2 } \widetilde { \Phi } ( f ) v \geq \frac { 1 } { \sigma ^ { 2 } } K a _ { * } \| v \| ^ { 2 } .\tag{26}
$$

It remains to bound $a _ { k k ^ { \prime } }$ . Define

$$
Z _ { k k ^ { \prime } } ( y ) : = \frac { \left( \sum _ { m } s _ { m } ( y ) \right) ^ { 2 } } { s _ { k } ( y ) s _ { k ^ { \prime } } ( y ) } .
$$

Then

$$
\omega _ { k } ( y ) \omega _ { k ^ { \prime } } ( y ) = \frac { 1 } { Z _ { k k ^ { \prime } } ( y ) } .
$$

By Jensen’s inequality,

$$
a _ { k k ^ { \prime } } = \mathbb { E } \left[ \frac { 1 } { Z _ { k k ^ { \prime } } ( Y ) } \right] \geq \frac { 1 } { \mathbb { E } [ Z _ { k k ^ { \prime } } ( Y ) ] } .
$$

Now

$$
Z _ { k k ^ { \prime } } ( Y ) = \sum _ { m , \ell } { \frac { s _ { m } ( Y ) s _ { \ell } ( Y ) } { s _ { k } ( Y ) s _ { k ^ { \prime } } ( Y ) } } .
$$

For the cost $c ( x , y ) = \| x - y \| ^ { 2 } / 2$ , since the quadratic terms in Y cancel out and we can express

$$
\frac { s _ { m } ( Y ) s _ { \ell } ( Y ) } { s _ { k } ( Y ) s _ { k ^ { \prime } } ( Y ) } = \frac { \alpha _ { m } \alpha _ { \ell } } { \alpha _ { k } \alpha _ { k ^ { \prime } } } \exp \Bigl ( \left[ A _ { k k ^ { \prime } } ^ { m \ell } - { \scriptstyle \frac { 1 } { 2 } } B _ { k k ^ { \prime } } ^ { m \ell } + \langle Y , C _ { k k ^ { \prime } } ^ { m \ell } \rangle \right] / \sigma ^ { 2 } \Bigr ) ,
$$

where

$$
\begin{array} { r c l } { { A _ { k k ^ { \prime } } ^ { m \ell } } } & { { = } } & { { f _ { m } + f _ { \ell } - f _ { k } - f _ { k ^ { \prime } } , } } \end{array}
$$

$$
\begin{array} { r l r } { B _ { k k ^ { \prime } } ^ { m \ell } } & { = } & { \| x _ { m } \| ^ { 2 } + \| x _ { \ell } \| ^ { 2 } - \| x _ { k } \| ^ { 2 } - \| x _ { k ^ { \prime } } \| ^ { 2 } , } \end{array}
$$

$$
\begin{array} { r c l } { C _ { k k ^ { \prime } } ^ { m \ell } } & { = } & { x _ { m } + x _ { \ell } - x _ { k } - x _ { k ^ { \prime } } . } \end{array}
$$

Using subGaussianity, for any $w \in \mathbb { R } ^ { d }$

$$
\begin{array} { r } { \mathbb { E } e ^ { \langle Y , w \rangle / \sigma ^ { 2 } } \leq \exp \biggl ( \frac { \varepsilon ^ { 2 } } { 2 \sigma ^ { 4 } } \| v \| ^ { 2 } \biggr ) . } \end{array}
$$

Using $| f _ { k } | \le L$ and $\| x _ { k } \| \leq R$ , we bound

$$
A _ { k k ^ { \prime } } ^ { m \ell } \leq 4 L , \quad - B _ { k k ^ { \prime } } ^ { m \ell } \leq 2 R ^ { 2 } , \quad \| C _ { k k ^ { \prime } } ^ { m \ell } \| \leq 4 R .
$$

Hence,

$$
\begin{array} { r } { \mathbb { E } \bigg [ \frac { s _ { m } ( Y ) s _ { \ell } ( Y ) } { s _ { k } ( Y ) s _ { k ^ { \prime } } ( Y ) } \bigg ] \leq \frac { \alpha _ { m } \alpha _ { \ell } } { \alpha _ { k } \alpha _ { k ^ { \prime } } } \exp ( M / \sigma ^ { 2 } ) , } \end{array}
$$

with

$$
M : = 4 L + R ^ { 2 } + \frac { 8 \varepsilon ^ { 2 } } { \sigma ^ { 2 } } R ^ { 2 } .
$$

Summing over $m , \ell$ gives

$$
\mathbb { E } [ Z _ { k k ^ { \prime } } ( Y ) ] \leq \frac { e ^ { M / \sigma ^ { 2 } } } { \alpha _ { k } \alpha _ { k ^ { \prime } } } .
$$

Therefore

$$
a _ { k k ^ { \prime } } \geq \alpha _ { k } \alpha _ { k ^ { \prime } } e ^ { - M / \sigma ^ { 2 } } \geq \underline { { \alpha } } ^ { 2 } e ^ { - M / \sigma ^ { 2 } } .
$$

Combining with the bound in (26) yields

$$
v ^ { \top } \nabla ^ { 2 } \widetilde { \Phi } ( f ) v \geq \frac { K \underline { { \alpha } } ^ { 2 } e ^ { - M / \sigma ^ { 2 } } } { \sigma ^ { 2 } } \| v \| ^ { 2 } ,\tag{27}
$$

if $v \in { { \bf 1 } ^ { \perp } }$ . Additionally, note that if $v \in { \mathbf { 1 } } , v = c { \mathbf { 1 } }$ for $c \in \mathbb { R }$ and so

$$
v ^ { \top } \nabla ^ { 2 } \widetilde { \Phi } ( f ) v = \frac { 1 } { \sigma ^ { 2 } } \sum _ { k < k ^ { \prime } } a _ { k k ^ { \prime } } ( c - c ) ^ { 2 } = 0
$$

Finally, take an arbitrary $u \in \mathbb { R } ^ { K }$ . Then, $u = u - S u + S u$ where

$$
\begin{array} { r } { S : = \left( I _ { K } - \frac { 1 } { K } \mathbf { 1 } \mathbf { 1 } ^ { \top } \right) } \end{array}\tag{28}
$$

is the projection onto the orthogonal complement of the space spanned by 1, and satisfies $S = S ^ { 2 }$ Then, taking $v = S u$ and using that by the above observation $\nabla ^ { 2 }  { \widetilde { \Phi } } ( f ) ( u - S u ) = 0$ , we obtain

$$
\begin{array} { r c l } { u ^ { \top } \nabla ^ { 2 } \widetilde { \Phi } ( f ) u } & { \ge } & { \displaystyle \frac { K \underline { { \alpha } } ^ { 2 } e ^ { - M / \sigma ^ { 2 } } } { \sigma ^ { 2 } } \| S u \| ^ { 2 } } \\ & { \ge } & { u ^ { \top } \left[ \displaystyle \frac { K \underline { { \alpha } } ^ { 2 } e ^ { - M / \sigma ^ { 2 } } } { \sigma ^ { 2 } } \Big ( I _ { K } - \frac { 1 } { K } \mathbf { 1 1 } ^ { \top } \Big ) \right] u , } \end{array}
$$

which yields the desired conclusion.

The following is a direct consequence of Proposition 2, and will lead to useful upper bounds for the $f _ { n } - f$ in the proof of Theorem 1

Corollary 2. Let $f \in \mathbb { R } ^ { K }$ be arbitrary and $f ^ { * } \in \mathbb { R } ^ { K }$ be an optimal potential for $( P , Q )$ . Suppose that $\| f \| _ { \infty } \leq L$ and $\| f ^ { * } \| _ { \infty } \leq L$ . Then,

$$
\operatorname { V a r } _ { \infty } ( f - f ^ { * } ) \leq { \frac { 2 } { \kappa } } \left( \Phi ( f ^ { * } ) - \Phi ( f ) \right) ,
$$

where κ is as in Proposition 2

Proof. Define $f _ { t } = f ^ { * } + t ( f - f ^ { * } )$ . Note that $\| f _ { t } \| _ { \infty } \leq L$ . By a second-order Taylor expansion with integral remainder, and using that $f ^ { * }$ is a stationary point of $\Phi \mathrm { i }$

$$
\begin{array} { r c l } { \Phi ( f ) - \Phi ( f ^ { * } ) } & { = } & { \nabla \Phi ( f ^ { * } ) ^ { \top } ( f - f ^ { * } ) + \displaystyle \int _ { 0 } ^ { 1 } ( 1 - t ) ( f - f ^ { * } ) ^ { \top } \nabla ^ { 2 } \Phi ( f _ { t } ) ( f - f ^ { * } ) d t } \\ & { = } & { \displaystyle \int _ { 0 } ^ { 1 } ( 1 - t ) ( f - f ^ { * } ) ^ { \top } \nabla ^ { 2 } \Phi ( f _ { t } ) ( f - f ^ { * } ) d t } \\ & { \leq } & { \displaystyle - \kappa \int _ { 0 } ^ { 1 } ( 1 - t ) ( f - f ^ { * } ) ^ { \top } S ^ { \top } S ( f - f ^ { * } ) d t } \\ & { = } & { \displaystyle - \frac { \kappa } { 2 } ( f - f ^ { * } ) ^ { \top } S ( f - f ^ { * } ) , } \end{array}
$$

where $S$ is as in (28). Now, note that for $u \in \mathbb { R } ^ { K }$

$$
u ^ { \top } S u = \sum _ { k = 1 } ^ { K } u _ { k } ^ { 2 } - \frac { 1 } { K } \left( \sum _ { k = 1 } ^ { K } u _ { k } \right) ^ { 2 } = K \mathrm { V a r } ( u ) .\tag{29}
$$

Therefore, by the above, and (23),

$$
\operatorname { V a r } _ { \infty } ( f - f ^ { * } ) \leq K \operatorname { V a r } ( f - f ^ { * } ) \leq { \frac { 2 } { \kappa } } \left( \Phi ( f ^ { * } ) - \Phi ( f ) \right) .
$$

## A.3 Proof of Proposition 3

Proof. Define

$$
{ \widetilde { f } } ( x ) : = { \frac { 1 } { 2 } } \| x \| ^ { 2 } - f ( x ) .
$$

From the fact that

$$
- \frac 1 2 \| x - y \| ^ { 2 } = - \frac 1 2 \| x \| ^ { 2 } + \langle x , y \rangle - \frac 1 2 \| y \| ^ { 2 } ,
$$

we obtain

$$
\widetilde { f } ( x ) = \sigma ^ { 2 } \log \int \exp \left( \frac { \langle x , y \rangle + g ( y ) - \frac { 1 } { 2 } \| y \| ^ { 2 } } { \sigma ^ { 2 } } \right) \mathrm { d } Q ( y ) .
$$

Additionally, since

$$
g ( y ) = - \sigma ^ { 2 } \log \sum _ { k = 1 } ^ { K } \alpha _ { k } \exp \left( \frac { f ( x _ { k } ) - \frac { 1 } { 2 } \| x _ { k } - y \| ^ { 2 } } { \sigma ^ { 2 } } \right) ,
$$

we have for any fixed $k ,$

$$
g ( y ) \leq - \left( f ( x _ { k } ) - { \frac { 1 } { 2 } } \| x _ { k } - y \| ^ { 2 } \right) - \sigma ^ { 2 } \log \alpha _ { k } .
$$

Consequently, plugging this into the definition of $\widetilde { f }$ we get that for each $\boldsymbol { x } \in \mathbb { R } ^ { d }$

$$
\begin{array} { r c l } { \displaystyle \tilde { f } ( x ) } & { \leq } & { \displaystyle \sigma ^ { 2 } \log \int \exp \left( \frac { \langle x , y \rangle + \frac 1 2 \| x _ { k } - y \| ^ { 2 } - f ( x _ { k } ) - \sigma ^ { 2 } \log \alpha _ { k } - \frac 1 2 \| y \| ^ { 2 } } { \sigma ^ { 2 } } \right) \mathrm { d } Q ( y ) } \\ & { \leq } & { \displaystyle \sigma ^ { 2 } \log \int \exp \left( \frac { \langle x , y \rangle + \frac 1 2 \| x _ { k } \| ^ { 2 } + \frac 1 2 \| y \| ^ { 2 } - \langle x _ { k } , y \rangle - f ( x _ { k } ) - \sigma ^ { 2 } \log \alpha _ { k } - \frac 1 2 \| y \| ^ { 2 } } { \sigma ^ { 2 } } \right) \mathrm { d } Q ( y ) , } \\ & { \leq } & { \displaystyle \sigma ^ { 2 } \log \int \exp \left( \frac { \langle x - x _ { k } , y \rangle + \frac 1 2 \| x _ { k } \| ^ { 2 } - f ( x _ { k } ) - \sigma ^ { 2 } \log \alpha _ { k } } { \sigma ^ { 2 } } \right) \mathrm { d } Q ( y ) } \\ & { \leq } & { \displaystyle \tilde { f } ( x _ { k } ) - \sigma ^ { 2 } \log \alpha _ { k } + \sigma ^ { 2 } \log \int \exp \left( \frac { ( x - x _ { k } , y ) } { \sigma ^ { 2 } } \right) \mathrm { d } Q ( y ) } \\ & { \leq } & { \displaystyle \tilde { f } ( x _ { k } ) - \sigma ^ { 2 } \log \alpha _ { k } + \frac { \xi ^ { 2 } } { 2 \sigma ^ { 2 } } \| x - x _ { k } \| ^ { 2 } , } \end{array}
$$

where, in the last inequality, we used the subGaussianity of $Q .$ . Taking $x = x _ { k ^ { \prime } }$ and symmetrizing yields

$$
| \widetilde { f } ( x _ { k } ) - \widetilde { f } ( x _ { k ^ { \prime } } ) | \leq \sigma ^ { 2 } \log \biggl ( \frac { 1 } { \underline { { \alpha } } } \biggr ) + \frac { \varepsilon ^ { 2 } } { 2 \sigma ^ { 2 } } \| x _ { k } - x _ { k ^ { \prime } } \| ^ { 2 } .
$$

Since $\| x _ { k } - x _ { k ^ { \prime } } \| \leq 2 R$

$$
| \widetilde { f } ( x _ { k } ) - \widetilde { f } ( x _ { k ^ { \prime } } ) | \leq \underline { { \sigma ^ { 2 } \log \biggl ( \frac { 1 } { \underline { { \alpha } } } \biggr ) + 2 \frac { \varepsilon ^ { 2 } } { \sigma ^ { 2 } } R ^ { 2 } } } .
$$

Now, let

$$
m : = \sum _ { k = 1 } ^ { K } \alpha _ { k } \widetilde { f } ( x _ { k } ) .
$$

From the gauge condition $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \alpha _ { k } f ( x _ { k } ) = 0 } \end{array}$ it follows that

$$
m = \sum _ { k = 1 } ^ { K } \alpha _ { k } \widetilde { f } ( x _ { k } ) = \frac { 1 } { 2 } \sum _ { k = 1 } ^ { K } \alpha _ { k } \| x _ { k } \| ^ { 2 } \leq \frac { 1 } { 2 } R ^ { 2 } ,\tag{30}
$$

Now, define ${ \widehat { f } } ( x _ { k } ) : = { \widetilde { f } } ( x _ { k } ) - m$ . Note that

$$
\operatorname* { m a x } _ { k } \widehat { f } ( x _ { k } ) - \operatorname* { m i n } _ { k } \widehat { f } ( x _ { k } ) \leq \operatorname* { s u p } _ { k , k ^ { \prime } } | \widehat { f } ( x _ { k } ) - \widehat { f } ( x _ { k ^ { \prime } } ) | = \operatorname* { s u p } _ { k , k ^ { \prime } } | \widetilde { f } ( x _ { k } ) - \widetilde { f } ( x _ { k ^ { \prime } } ) | \leq B .\tag{31}
$$

Additionally, since $\textstyle \sum _ { k } \alpha _ { k } { \widehat { f } } ( x _ { k } ) = 0$ , we have that max $\mathfrak { x } \widehat { f } ( x _ { k } ) \geq 0$ and $\mathrm { m i n } _ { k } \widehat { f } ( x _ { k } ) \leq 0$ , so by (31)

$$
| \widehat { f } ( x _ { k } ) | \leq \sigma ^ { 2 } \log \biggl ( \frac { 1 } { \underline { { \alpha } } } \biggr ) + 2 \frac { \varepsilon ^ { 2 } } { \sigma ^ { 2 } } R ^ { 2 } .\tag{32}
$$

Now, from the fact that

$$
f ( x _ { k } ) = { \frac { 1 } { 2 } } \| x _ { k } \| ^ { 2 } - { \widetilde f } ( x _ { k } ) = { \frac { 1 } { 2 } } \| x _ { k } \| ^ { 2 } - { \widehat f } ( x _ { k } ) - m ,
$$

combined with (30) (32) we obtain

$$
| f ( x _ { k } ) | \leq \frac { 1 } { 2 } R ^ { 2 } + | \widehat f ( x _ { k } ) | + | m | \leq R ^ { 2 } + \sigma ^ { 2 } \log \biggl ( \frac { 1 } { \underline { { \alpha } } } \biggr ) + 2 \frac { \varepsilon ^ { 2 } } { \sigma ^ { 2 } } R ^ { 2 } .
$$

## A.4 Proof of Theorem 1

Proof. We will first show the bound holds for $\| f _ { n } - f \| _ { \infty }$ . We will reduce the two-sample statement to simpler ones by pivoting on $( P , Q _ { n } )$ . Specifically, denote ${ \breve { f } } _ { n }$ the optimal semidual potential for $( P , Q _ { n } )$ satisfying (4). We write

$$
\begin{array} { r c l } { \mathbb { E } \left[ \operatorname { V a r } _ { \infty } \left( f _ { n } - f \right) \right] } & { = } & { \mathbb { E } \left[ \operatorname { V a r } _ { \infty } \left( f _ { n } - \check { f } _ { n } + \check { f } _ { n } - f \right) \right] } \\ & { \leq } & { 2 \mathbb { E } \left[ \operatorname { V a r } _ { \infty } \left( f _ { n } - \check { f } _ { n } \right) \right] + 2 \mathbb { E } \left[ \operatorname { V a r } _ { \infty } \left( \check { f } _ { n } - f \right) \right] } \\ & { \lesssim } & { \frac { 1 } { n } + \frac { 1 } { n } + r _ { n , d } \lesssim \frac { 1 } { n } + r _ { n , d } . } \end{array}
$$

Where in the last line we used Lemma 2 to bound E $\left[ \operatorname { V a r } _ { \infty } \left( { \breve { f } } _ { n } - f \right) \right]$ and Lemma 3 to bound $\mathbb { E } \left[ \operatorname { V a r } _ { \infty } \left( { \breve { f } } _ { n } - f _ { n } \right) \right]$ . Since, by definition, $\mathbb { E } _ { P } ( f _ { n } - f ) = 0$ we have

$$
\operatorname* { m i n } _ { k \in [ K ] } [ f _ { n } ( x _ { k } ) - f ( x _ { k } ) ] \leq 0 \leq \operatorname* { m a x } _ { k \in [ K ] } [ f _ { n } ( x _ { k } ) - f ( x _ { k } ) ] ,
$$

and so by Lemma 1 this implies that

$$
\mathbb { E } \left( \| f _ { n } - f \| _ { \infty } ^ { 2 } \right) \leq 4 \mathbb { E } \mathrm { V a r } _ { \infty } ( f _ { n } - f ) \lesssim \frac { 1 } { n } + r _ { n , d } .
$$

Let’s now establish bounds for $g _ { n } - g$ . Recall that, using the canonical extensions to (5b) we have that for each $\boldsymbol { y } \in \mathbb { R } ^ { d }$

$$
g _ { n } ( y ) = - \sigma ^ { 2 } \log \left( \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { n } \exp \left( f _ { n } ( x _ { k } ) / \sigma ^ { 2 } - c ( x _ { k } , y ) / \sigma ^ { 2 } \right) \right)
$$

and

$$
g ( y ) = - \sigma ^ { 2 } \log \left( \sum _ { k = 1 } ^ { K } \alpha _ { k } \exp \left( f ( x _ { k } ) / \sigma ^ { 2 } - c ( x _ { k } , y ) / \sigma ^ { 2 } \right) \right) .
$$

We define the intermediate function $\breve { g } _ { n } : \mathbb { R } ^ { d } \to$ R.

$$
\check { g } _ { n } ( y ) : = - \sigma ^ { 2 } \log \left( \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { n } \exp \left( f ( x _ { k } ) / \sigma ^ { 2 } - c ( x _ { k } , y ) / \sigma ^ { 2 } \right) \right) .
$$

Then, writing $z _ { 1 } = f _ { n } ( x ) - c ( x , y ) , z _ { 0 } = f ( x ) - c ( x , y ) \in \mathbb { R } ^ { K }$ (the notation $c ( x , y )$ denotes the vector $( c ( x _ { 1 } , y ) , \dots , c ( x _ { K } , y ) )$ and $z _ { t } = t z _ { 1 } + ( 1 - t ) z _ { 0 }$ , and denoting

$$
\phi _ { 0 } ( z ) : = - \sigma ^ { 2 } \log \left( \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { n } \exp ( z _ { k } / \sigma ^ { 2 } ) \right) , \quad \phi ( t ) : = \phi _ { 0 } ( z _ { t } ) ,
$$

we have

$$
\begin{array} { r l l } { \displaystyle | g _ { n } ( y ) - \check { g } _ { n } ( y ) | } & { = } & { \displaystyle \left| \phi _ { 0 } ( z _ { 1 } ) - \phi _ { 0 } ( z _ { 0 } ) \right| = \left| \int _ { 0 } ^ { 1 } \phi ^ { \prime } ( s ) \mathrm { d } s \right| } \\ & { = } & { \displaystyle \left| \int _ { 0 } ^ { 1 } \{ \nabla \phi _ { 0 } ( z _ { s } ) , z _ { 1 } - z _ { 0 } \} \mathrm { d } s \right| } \\ & { \le } & { \displaystyle \int _ { 0 } ^ { 1 } \sum _ { k = 1 } ^ { K } z _ { 1 } ( k ) - z _ { 0 } ( k ) \| \mathrm { d } s } \\ & { \le } & { \displaystyle \sum _ { k = 1 } ^ { K } | f _ { n } ( x _ { k } ) - f ( x _ { k } ) | } \\ & { \le } & { \displaystyle K \| f _ { n } - f \| _ { L ^ { \infty } ( P _ { s } ) } } \\ & { \le } & { \displaystyle \| f _ { n } - f \| _ { L ^ { \infty } ( P _ { s } ) } . } \end{array}
$$

Above, we used that for k such that $\alpha _ { k } ^ { n } > 0$ 2

$$
\left| \frac { \partial \phi _ { 0 } } { \partial z _ { k } } ( z ) \right| = \frac { \alpha _ { k } ^ { n } \exp ( z _ { k } / \sigma ^ { 2 } ) } { \sum _ { k ^ { \prime } = 1 } ^ { K } \alpha _ { k ^ { \prime } } ^ { n } \exp ( z _ { k ^ { \prime } } / \sigma ^ { 2 } ) } \le 1 ,
$$

and otherwise, this derivative equals 0. Note that the above bound is independent of $y .$ . Since the preceding bound is uniform in $y$ and $Q$ is a probability measure, it implies

$$
\begin{array} { r } { \| g _ { n } - \check { g } _ { n } \| _ { L ^ { 2 } ( Q ) } ^ { 2 } \lesssim \| f _ { n } - f \| _ { L ^ { \infty } ( P _ { n } ) } ^ { 2 } . } \end{array}
$$

Taking expectations and using the already established bound for $f _ { n } - f .$ , we obtain

$$
\mathbb { E } \Vert g _ { n } - \check { g } _ { n } \Vert _ { L ^ { 2 } ( Q ) } ^ { 2 } \lesssim \frac { 1 } { n } .\tag{33}
$$

The analysis of $\breve { g } _ { n } ( y ) - g ( y )$ is more delicate. Let $\alpha _ { k } ^ { n }$ be the random weights associated to the empirical measure $P _ { n }$ , and define the set of active indexes

$$
A _ { n } : = \{ k \in [ K ] : \alpha _ { k } ^ { n } > 0 \}\tag{34}
$$

and $\Lambda _ { n }$ as the event

$$
\begin{array} { r } { \Lambda _ { n } : = \{ \alpha _ { k } ^ { n } \geq \alpha _ { k } / 2 , k \in [ K ] \} . } \end{array}\tag{35}
$$

Define also

$$
\Pi _ { k } ( f , \alpha , y ) : = \frac { \alpha _ { k } e ^ { ( f _ { k } - c ( x _ { k } , y ) ) / \sigma ^ { 2 } } } { \sum _ { k ^ { \prime } = 1 } ^ { K } \alpha _ { k ^ { \prime } } e ^ { ( f _ { k ^ { \prime } } - c ( x _ { k ^ { \prime } } , y ) ) / \sigma ^ { 2 } } } ,\tag{36}
$$

and

$$
R _ { n } ( y ) : = \frac { \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { n } \exp \big ( ( f ( x _ { k } ) - c ( x _ { k } , y ) ) / \sigma ^ { 2 } \big ) } { \sum _ { k = 1 } ^ { K } \alpha _ { k } \exp \big ( ( f ( x _ { k } ) - c ( x _ { k } , y ) ) / \sigma ^ { 2 } \big ) } = \sum _ { k = 1 } ^ { K } \Pi _ { k } ( f , \alpha , y ) \frac { \alpha _ { k } ^ { n } } { \alpha _ { k } } ,
$$

so that clearly

$$
\breve { g } _ { n } ( y ) - g ( y ) = - \sigma ^ { 2 } \log R _ { n } ( y ) .
$$

First, note that since $\Pi _ { k } ( f , \alpha , y )$ is a probability vector and since $\alpha _ { k } ^ { n } / \alpha _ { k } \geq 1 / 2$ for all $k ,$ , we have $R _ { n } ( y ) \geq 1 / 2$ . Then, using the fact that $x \to \log x$ is lipschitz on $[ 1 / 2 , \infty )$ with Lipschitz constant bounded by 2, we get that on $\Lambda _ { n }$

$$
| \log R _ { n } ( y ) | \leq 2 | R _ { n } ( y ) - 1 | .
$$

Moreover,

$$
| R _ { n } ( y ) - 1 | = \left| \sum _ { k = 1 } ^ { K } \Pi _ { k } ( f , \alpha , y ) \left( \frac { \alpha _ { k } ^ { n } } { \alpha _ { k } } - 1 \right) \right| \leq \operatorname* { m a x } _ { k \in [ K ] } \left| \frac { \alpha _ { k } ^ { n } - \alpha _ { k } } { \alpha _ { k } } \right| \leq \frac { 1 } { \underline { { \alpha } } } \| \alpha ^ { n } - \alpha \| _ { \infty } .
$$

Therefore, uniformly in $y .$

$$
\begin{array} { r } { \mathbf { 1 } _ { \Lambda _ { n } } | \check { g } _ { n } ( y ) - g ( y ) | ^ { 2 } \lesssim \| \alpha ^ { n } - \alpha \| _ { \infty } ^ { 2 } . } \end{array}
$$

Integrating with respect to $Q$ and taking expectations, Lemma 7 gives

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \Lambda _ { n } } \big \| \breve { g } _ { n } - g \big \| _ { L ^ { 2 } ( Q ) } ^ { 2 } \right] \lesssim \mathbb { E } \big \| \alpha ^ { n } - \alpha \big \| _ { \infty } ^ { 2 } \lesssim \frac { 1 } { n } .
$$

We now control the complement $\Lambda _ { n } ^ { c }$ . For $k \in [ K ]$ and $k ^ { \prime } \in A _ { n }$ , define

$$
a _ { k } ( y ) : = \log \alpha _ { k } + \frac { f ( x _ { k } ) - c ( x _ { k } , y ) } { \sigma ^ { 2 } } , \quad b _ { k ^ { \prime } } ( y ) : = \log \alpha _ { k ^ { \prime } } ^ { n } + \frac { f ( x _ { k ^ { \prime } } ) - c ( x _ { k ^ { \prime } } , y ) } { \sigma ^ { 2 } } .
$$

From the following log-sum-exp inequality: if $z \in \mathbb { R } ^ { M }$ ，

$$
\operatorname* { m a x } _ { m \in [ M ] } z _ { m } \leq \log \sum _ { m = 1 } ^ { M } e ^ { z _ { m } } \leq \operatorname* { m a x } _ { m \in [ M ] } z _ { m } + \log M ,
$$

We apply this twice. Let

$$
N _ { a } ( y ) : = \operatorname* { m a x } _ { k \in [ K ] } a _ { k } ( y ) , \qquad N _ { b } ( y ) : = \operatorname* { m a x } _ { k \in A _ { n } } b _ { k } ( y ) .
$$

Then, for some remainders $r _ { a } ( y )$ and $r _ { b } ( y )$ we have

$$
\log \sum _ { k = 1 } ^ { K } e ^ { a _ { k } ( y ) } = N _ { a } ( y ) + r _ { a } ( y ) , \qquad 0 \leq r _ { a } ( y ) \leq \log K ,
$$

and

$$
\log \sum _ { k \in A _ { n } } e ^ { b _ { k } ( y ) } = N _ { b } ( y ) + r _ { b } ( y ) , \qquad 0 \leq r _ { b } ( y ) \leq \log | A _ { n } | \leq \log K .
$$

Consequently,

$$
\left| \log \sum _ { k \in A _ { n } } e ^ { b _ { k } ( y ) } - \log \sum _ { k = 1 } ^ { K } e ^ { a _ { k } ( y ) } \right| \leq | N _ { b } ( y ) - N _ { a } ( y ) | + | r _ { b } ( y ) - r _ { a } ( y ) | .\tag{37}
$$

Since both $r _ { b } ( y )$ and $r _ { a } ( y )$ lie in [0, log K], we have

$$
| r _ { b } ( y ) - r _ { a } ( y ) | \leq \log K .
$$

It remains to bound $| N _ { b } ( y ) - N _ { a } ( y ) |$ . Let

$$
k ^ { \star } \in \arg \operatorname* { m a x } _ { k \in [ K ] } a _ { k } ( y ) , \quad k ^ { \prime \star } \in \arg \operatorname* { m a x } _ { k \in A _ { n } } b _ { k } ( y ) .
$$

Then

$$
| N _ { b } ( y ) - N _ { a } ( y ) | = | b _ { k ^ { \prime \star } } ( y ) - a _ { k ^ { \star } } ( y ) | \leq \operatorname* { m a x } _ { k \in [ K ] , k ^ { \prime } \in A _ { n } , } | b _ { k ^ { \prime } } ( y ) - a _ { k } ( y ) | ,
$$

and so by (37), and by the definition of $\breve { g } _ { n }$ and $^ { g , }$

$$
\left| \check { g } _ { n } ( y ) - g ( y ) \right| = \sigma ^ { 2 } \left| \log \sum _ { k ^ { \prime } \in A _ { n } } e ^ { b _ { k ^ { \prime } } ( y ) } - \log \sum _ { k = 1 } ^ { K } e ^ { a _ { k } ( y ) } \right| \leq \sigma ^ { 2 } \log K + \sigma ^ { 2 } \operatorname* { m a x } _ { k ^ { \prime } \in A _ { n } , k \in [ K ] } \left| b _ { k ^ { \prime } } ( y ) - a _ { k } ( y ) \right| .
$$

We must now bound the last term above. Using that $\alpha _ { k ^ { \prime } } ^ { n } \geq 1 / n$ for $k ^ { \prime } \in A _ { n }$ , that $\alpha _ { k } ^ { n } , \alpha _ { k } \leq 1$ , and that $\alpha _ { k } \geq \underline { { \alpha } }$ , we get

$$
\operatorname* { m a x } _ { k ^ { \prime } \in A _ { n } , \ k \in [ K ] } \left| \log \frac { \alpha _ { k ^ { \prime } } ^ { n } } { \alpha _ { k } } \right| \lesssim \log n .
$$

Moreover, by Proposition $3 , \| f \| _ { \infty }$ is bounded by a constant depending only on $R , { \sigma } ^ { 2 } , { \varepsilon } , { \underline { { \alpha } } }$ . Finally, for the quadratic cost,

$$
| c ( x _ { k } , y ) - c ( x _ { k ^ { \prime } } , y ) | = \left| { \frac { 1 } { 2 } } { \big ( } \| x _ { k } \| ^ { 2 } - \| x _ { k ^ { \prime } } \| ^ { 2 } { \big ) } - \langle x _ { k } - x _ { k ^ { \prime } } , y \rangle \right| \leq R ^ { 2 } + | \langle x _ { k } - x _ { k ^ { \prime } } , y \rangle | .
$$

Therefore,

$$
| \check { g } _ { n } ( y ) - g ( y ) | \lesssim \log n + L ( y ) , \qquad \mathrm { w h e r e ~ } L ( y ) : = \operatorname* { m a x } _ { k , k ^ { \prime } \in [ K ] } | \langle x _ { k } - x _ { k ^ { \prime } } , y \rangle | .
$$

We bound the expectation of the last term using a standard subGaussianity argument: since $Y \sim Q$ is $\varepsilon ^ { 2 } .$ -subGaussian, for every $k , k ^ { \prime } \in [ K ]$ the random variable $Z _ { k , k ^ { \prime } } : = \langle x _ { k } - x _ { k ^ { \prime } } , Y \rangle$ is subGaussian with proxy at most $\varepsilon ^ { 2 } \| x _ { k } - x _ { k ^ { \prime } } \| ^ { 2 } \leq 4 \varepsilon ^ { 2 } R ^ { 2 }$ , and so [2, Chapter 2.3]

$$
\mathbb { P } _ { Q } ( | Z _ { k , k ^ { \prime } } | > t ) \le 2 \exp \left( - \frac { t ^ { 2 } } { 8 \varepsilon ^ { 2 } R ^ { 2 } } \right) .
$$

Therefore, by the union bound,

$$
\mathbb { P } _ { Q } ( L ( Y ) > t ) \le 2 K ^ { 2 } \exp \left( - \frac { t ^ { 2 } } { 8 \varepsilon ^ { 2 } R ^ { 2 } } \right) ,
$$

and so by the layer cake representation,

$$
\mathbb { E } _ { Q } L ( Y ) ^ { 2 } = \int _ { 0 } ^ { \infty } 2 t \mathbb { P } _ { Q } ( L ( Y ) > t ) d t .
$$

Set

$$
t _ { 0 } : = 4 \varepsilon R \sqrt { \log ( 2 K ) } .
$$

Then

$$
\int _ { 0 } ^ { t _ { 0 } } 2 t \mathbb { P } _ { Q } ( L ( Y ) > t ) d t \leq t _ { 0 } ^ { 2 } \lesssim \varepsilon ^ { 2 } R ^ { 2 } \log ( 2 K ) ,
$$

and using that $\begin{array} { r } { \int _ { t _ { 0 } } ^ { \infty } x \exp ( - x ^ { 2 } / a ) d x = a / 2 \exp ( - t _ { 0 } ^ { 2 } / a ) } \end{array}$

$$
\int _ { t _ { 0 } } ^ { \infty } 2 t \mathbb { P } _ { Q } ( L ( Y ) > t ) d t \leq 4 K ^ { 2 } \int _ { t _ { 0 } } ^ { \infty } t \exp \left( - \frac { t ^ { 2 } } { 8 \varepsilon ^ { 2 } R ^ { 2 } } \right) d t = 4 K ^ { 2 } \frac { 8 \varepsilon ^ { 2 } R ^ { 2 } } { 2 } \exp \left( - \frac { t _ { 0 } ^ { 2 } } { 8 \varepsilon ^ { 2 } R ^ { 2 } } \right) \leq \frac { 1 6 \varepsilon ^ { 2 } R ^ { 2 } K ^ { 2 } } { ( 2 K ) ^ { 2 } } \lesssim \varepsilon ^ { 2 } R ^ { 2 } .
$$

Consequently,

$$
\begin{array} { r } { \mathbb { E } _ { Q } L ( Y ) ^ { 2 } \lesssim \varepsilon ^ { 2 } R ^ { 2 } \log ( 2 K ) . } \end{array}
$$

We have concluded that

$$
\begin{array} { r } { \lVert \breve { g } _ { n } - g \rVert _ { L ^ { 2 } ( Q ) } ^ { 2 } \lesssim \log ^ { 2 } n + \varepsilon ^ { 2 } R ^ { 2 } \log ( 2 K ) . } \end{array}
$$

Therefore, by Lemma 7,

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \Lambda _ { n } ^ { c } } \lVert \check { g } _ { n } - g \rVert _ { L ^ { 2 } ( Q ) } ^ { 2 } \right] \lesssim \left( \log ^ { 2 } n + \varepsilon ^ { 2 } R ^ { 2 } \log ( 2 K ) \right) \mathbb { P } ( \Lambda _ { n } ^ { c } ) \lesssim \frac { 1 } { n } .
$$

Combining the bounds on $\Lambda _ { n }$ and $\Lambda _ { n } ^ { c }$ , we conclude that

$$
\mathbb { E } \Vert \breve { g } _ { n } - g \Vert _ { L ^ { 2 } ( Q ) } ^ { 2 } \lesssim \frac { 1 } { n } .\tag{38}
$$

Finally, by the triangle inequality,

$$
\begin{array} { r } { \| g _ { n } - g \| _ { L ^ { 2 } ( Q ) } ^ { 2 } \leq 2 \| g _ { n } - \check { g } _ { n } \| _ { L ^ { 2 } ( Q ) } ^ { 2 } + 2 \| \check { g } _ { n } - g \| _ { L ^ { 2 } ( Q ) } ^ { 2 } . } \end{array}
$$

Taking expectations and using and (33) and (38), we obtain

$$
\mathbb { E } \Vert g _ { n } - g \Vert _ { L ^ { 2 } ( Q ) } ^ { 2 } \lesssim \frac { 1 } { n } .
$$

## Proof of Corollary 1

Proof. The reader can verify that the term $r _ { n , d }$ appears from the need to control $f _ { n } ( x _ { k } )$ where k is such that $\alpha _ { k } ^ { n } = 0$ . In the one-sample case, we work with the population measure, which puts mass on all $x _ { k }$ since $\alpha _ { k } \geq \underline { { \alpha } } > 0$ . Likewise, if we measure error using $L ^ { \infty } ( P _ { n } )$ then there is no need to control $f _ { n } ( x _ { k } )$ on unobserved atoms □

The following corollary strengthening to higher order moments of $\| f _ { n } - f \| _ { \infty }$

Corollary 3. Under the assumptions of Theorem 1, for every fixed integer $q \geq 1$

$$
\begin{array} { r } { \mathbb { E } \| f _ { n } - f \| _ { \infty } ^ { 2 q } \lesssim n ^ { - q } + r _ { n , d , q } . } \end{array}
$$

where the underlying constants depend on q, in addition to the parameters in Theorem 1. The term $r _ { n , d , q }$ is similar to the one defined in Theorem 1 but the leading constant is allowed to depend on $q .$ We can remove the term $r _ { n , d , q }$ either in the one-sample case or when measuring error using the norm $E \| f _ { n } - f \| _ { L ^ { \infty } ( P _ { n } ) }$

Proof. The proof is the same as the proof of Theorem 1, replacing the second-moment bounds by 2q-moment bounds. We briefly indicate the changes. To bound $\mathrm { V a r } _ { \infty } ( \breve { f } _ { n } - f )$ , we first note that the localized empirical-process estimate used in the proof of Theorem 1 is available in an arbitrary fixed moment order: indeed, by Lemma 6 for every $p \geq 2$ 2

$$
\mathbb { E } \left[ \operatorname* { s u p } _ { \mathrm { V a r } _ { \infty } ( u - f ) \leq \tau ^ { 2 } } \left| \int ( \Gamma _ { u } - \Gamma _ { f } ) d ( Q _ { n } - Q ) \right| ^ { p } \right] \lesssim \left( \tau \sqrt { \frac { K } { n } } \right) ^ { p } .
$$

Choosing $p > 2 q$ in the dyadic peeling argument gives

$$
\begin{array} { r } { \mathbb { E } \operatorname { V a r } _ { \infty } ( \breve { f } _ { n } - f ) ^ { q } \lesssim n ^ { - q } . } \end{array}
$$

Indeed, on the slice

$$
\mathrm { V a r } _ { \infty } ( \breve { f } _ { n } - f ) \in [ a _ { j } ^ { 2 } , a _ { j + 1 } ^ { 2 } ] , \qquad a _ { j } : = 2 ^ { j } n ^ { - 1 / 2 } ,
$$

strong concavity implies that the localized empirical process must be at least of order $a _ { j } ^ { 2 }$ . Markov’s inequality with moment $p$ then gives a summable bound

$$
\begin{array} { r } { \mathbb { P } \left( \operatorname { V a r } _ { \infty } ( \check { f } _ { n } - f ) \in [ a _ { j } ^ { 2 } , a _ { j + 1 } ^ { 2 } ] \right) \lesssim 2 ^ { - p j } , } \end{array}
$$

and hence

$$
\mathbb { E } \operatorname { V a r } _ { \infty } ( \check { f } _ { n } - f ) ^ { q } \lesssim \sum _ { j \geq 0 } a _ { j + 1 } ^ { 2 q } 2 ^ { - p j } \lesssim n ^ { - q } ,
$$

provided $p > 2 q$

For the two-sample term, we mimic the strategy in the proof of Lemma 3. By Lemma 4, we have that on $\Lambda _ { n } ,$

$$
\begin{array} { r } { \operatorname { V a r } _ { \infty } ( f _ { n } - \check { f } _ { n } ) \lesssim \chi ^ { 2 } ( P _ { n } \| P ) . } \end{array}
$$

Then, by Lemma $^ { 7 , }$

$$
\mathbb { E } \left[ 1 _ { \Lambda _ { n } } \mathrm { V a r } _ { \infty } ( f _ { n } - \check { f } _ { n } ) ^ { q } \right] \lesssim \mathbb { E } \left( \left[ \chi ^ { 2 } ( P _ { n } \| P ) \right] ^ { q } \right) \leq \frac { K ^ { q } } { \underline { { \alpha } } ^ { q } } \mathbb { E } \left[ \| \alpha ^ { n } - \alpha \| _ { \infty } \right] ^ { 2 q } \lesssim n ^ { - q } .
$$

In the complement $\Lambda _ { n } ^ { c } .$ , we use the crude bound $\begin{array} { r } { \mathrm { V a r } _ { \infty } ( f _ { n } - \check { f } _ { n } ) ^ { q } \lesssim \| f _ { n } \| ^ { q } + \| f \| ^ { q } } \end{array}$ and then control each term using the (dimension-dependent) bound in Proposition 6. This will lead to a polynomial bound in a random subGaussianity parameter $\widetilde { \varepsilon } ,$ whose moments are nonetheless bounded. We conclude the analysis of this event using Cauchy-Schwarz and the fact that $\mathbb { P } ( \Lambda _ { n } ^ { c } )$ decays exponentially fast. Thus,

$$
\begin{array} { r } { \mathbb { E } \operatorname { V a r } _ { \infty } ( f _ { n } - \check { f } _ { n } ) ^ { q } \lesssim n ^ { - q } + r _ { n , d } . } \end{array}
$$

Combining the one-sample and two-sample bounds yields

$$
\begin{array} { r } { \mathbb { E } \| f _ { n } - f \| _ { \infty } ^ { 2 q } \lesssim \mathbb { E } \operatorname { V a r } _ { \infty } ( f _ { n } - f ) ^ { q } + \mathbb { E } \operatorname { V a r } _ { \infty } ( f _ { n } - \check { f } _ { n } ) ^ { q } \lesssim n ^ { - q } + r _ { n , d , q } . } \end{array}
$$

Lemma 2. In the setup of Theorem $\ i , \ i f \ f _ { n } , f$ are the optimal potentials for $( P , Q _ { n } )$ and $( P , Q )$ respectively, satisfying (4). Then

$$
\mathbb { E } \left[ \operatorname { V a r } _ { \infty } ( f _ { n } - f ) \right] \lesssim \frac { 1 } { n } ,
$$

with constants that depend on $R , \sigma ^ { 2 }$ and $\varepsilon$ but not on $d .$

Proof. Let $\widetilde { \varepsilon }$ be the smallest such that $Q _ { n } , Q$ are uniformly subGaussian, which is a finite random variable by Lemma 9. Applying Proposition 3 to the pairs $( P , Q )$ and $( P , Q _ { n } )$ we have

$$
\operatorname* { m a x } \{ \| f \| _ { \infty } , \| f _ { n } \| _ { \infty } \} \ \lesssim \ R ^ { 2 } + \sigma ^ { 2 } \log \biggr ( \frac { 1 } { \underline { { { \alpha } } } } \biggr ) + \frac { { \widetilde { \varepsilon } } ^ { 2 } } { \sigma ^ { 2 } } R ^ { 2 } .
$$

Then, by Lemma 1 we have the bound

$$
\begin{array} { r c l } { Z _ { n } : = \operatorname { V a r } _ { \infty } ( f _ { n } - f ) } & { \leq } & { \| f - f _ { n } \| _ { \infty } ^ { 2 } } \\ & { \leq } & { 2 \| f \| _ { \infty } ^ { 2 } + 2 \| f _ { n } \| _ { \infty } ^ { 2 } } \\ & { \lesssim } & { R ^ { 4 } + \sigma ^ { 4 } \log ^ { 2 } \biggr ( \frac { 1 } { \underline { { \alpha } } } \biggr ) + \frac { \widetilde { \varepsilon } ^ { 4 } } { \sigma ^ { 4 } } R ^ { 4 } . } \end{array}
$$

Above, we used that $( a + b ) ^ { 2 } \leq 2 ( a ^ { 2 } + b ^ { 2 } )$ . Let’s now define the event $E _ { n } = \{ \tilde { \varepsilon } ^ { 2 } < 1 2 \varepsilon ^ { 2 } \}$ . We will rely on the following decomposition

$$
\begin{array} { r } { \mathbb { E } \left( Z _ { n } \right) = \mathbb { E } \left( Z _ { n } 1 _ { E _ { n } } \right) + \mathbb { E } \left( Z _ { n } 1 _ { E _ { n } ^ { c } } \right) \lesssim \underbrace { \mathbb { E } \left( Z _ { n } 1 _ { E _ { n } } \right) } _ { A _ { n } } + \underbrace { \mathbb { E } \left( Z _ { n } ^ { 2 } \right) ^ { 1 / 2 } } _ { B _ { n } } \underbrace { \mathbb { P } \left( E _ { n } ^ { c } \right) ^ { 1 / 2 } } _ { C _ { n } } , } \end{array}\tag{39}
$$

and bound each of $A _ { n } , B _ { n } , C _ { n }$ . First, regarding $C _ { n } .$ , it follows directly from Lemma 5 (with $m = 4 )$ that $C _ { n } = \mathbb { P } ( E _ { n } ^ { c } ) ^ { 1 / 2 } \lesssim n ^ { - 1 }$ . Additionally, by Lemma 9(d) all moments of $\widetilde { \varepsilon } ^ { 2 }$ are finite, implying that $B _ { n }$ is bounded by a polynomial of degree 4 in $R ,$ and, in particular, is finite. It only remains to bound $A _ { n }$ . Note that in $E _ { n }$ we have a deterministic bound for $f _ { n }$ :

$$
\| f _ { n } \| _ { \infty } \lesssim L = L _ { \varepsilon , R , \sigma ^ { 2 } , \underline { { \alpha } } } : = R ^ { 2 } + \sigma ^ { 2 } \log \left( \frac { 1 } { \underline { { \alpha } } } \right) + \frac { \varepsilon ^ { 2 } } { \sigma ^ { 2 } } R ^ { 2 } .
$$

Denote $\Phi _ { n }$ the semidual function for the one-sample empirical problem $( P , Q _ { n } )$ . From Corollary 2 we obtain that for $\kappa = \kappa ( \varepsilon , R , \sigma ^ { 2 } , \underline { { \alpha } } )$

$$
\begin{array} { r l } { \mathrm { V a r } _ { \infty } ( f _ { n } - f ) } & { \le \ \frac { 2 } { \kappa } \left( \Phi ( f ) - \Phi ( f _ { n } ) \right) } \\ & { \le \ \frac { 2 } { \kappa } \left( \Phi ( f ) - \Phi _ { n } ( f _ { n } ) + \Phi _ { n } ( f _ { n } ) - \Phi ( f _ { n } ) \right) } \\ & { \le \ \frac { 2 } { \kappa } \left( \Phi ( f ) - \Phi _ { n } ( f ) + \Phi _ { n } ( f _ { n } ) - \Phi ( f _ { n } ) \right) } \\ & { = \ \frac { 2 } { \kappa } \left( \int \Gamma _ { f } ( y ) \mathrm { d } ( Q _ { n } - Q ) ( y ) + \int \Gamma _ { f _ { n } } ( y ) \mathrm { d } ( Q - Q _ { n } ) ( y ) \right) } \\ & { \le \ \frac { 2 } { \kappa } \left[ \underbrace { \int \left( \Gamma _ { f } ( y ) - \Gamma _ { f _ { n } } ( y ) \right) \mathrm { d } ( Q - Q _ { n } ) ( y ) } _ { W _ { n } } \right] . } \end{array}
$$

Above, we also used that by optimality of $f _ { n } , \Phi _ { n } ( f _ { n } ) \geq \Phi _ { n } ( f )$ . Note that, from the above, in $E _ { n }$ if $Z _ { n } \geq a$ then $W _ { n } \ge \kappa a / 2$ . Consider now the dyadic partition $[ a _ { k } ^ { 2 } , a _ { k + 1 } ^ { 2 } ]$ where $k \geq 0$ and $a _ { k } = 2 ^ { k } / \sqrt { n }$ . By Markov’s inequality and the above observations, we have

$$
\begin{array} { r l } { 4 } & { = \mathbb { E } \{ \mathcal { Q } _ { 1 , 1 6 } ^ { \nu } \} } \\ & { = \quad \frac { 8 } { \nu } \{ ( \zeta _ { 1 , 1 6 } ^ { \nu } , \zeta _ { 1 , 2 5 } ^ { \nu } , \zeta _ { 2 , 4 , 1 6 } ^ { \nu } ) \} } \\ & { \leq \quad \frac { \nu } { \nu } ( \frac { \sigma _ { 1 , 1 6 } ^ { \nu } } { \nu } ) ^ { \nu } ( \zeta _ { 2 , 1 6 } ^ { \nu } , \zeta _ { 3 , 1 6 } ^ { \nu } , \zeta _ { 3 , 2 5 } ^ { \nu } ) \leq \alpha _ { 1 , 0 , 1 6 } ^ { \nu } , } \\ & { \leq \quad \frac { \nu } { \nu } ( \sigma _ { 1 , 1 6 } ^ { \nu } , \zeta _ { 1 , 1 6 } ^ { \nu } ) } \\ & { \leq \quad \frac { \nu } { \nu } \sigma _ { 1 , 1 6 } ^ { \nu } ( | \zeta _ { 1 , 1 6 } ^ { \nu } , \zeta _ { 2 , 1 6 } ^ { \nu } , \zeta _ { 3 , 1 6 } ^ { \nu } | , \zeta _ { 3 , 1 6 } ^ { \nu } ) } \\ & { \leq \quad \frac { \nu } { \nu } \sigma _ { 1 , 1 6 } ^ { \nu } ( | \zeta _ { 1 , 1 6 } ^ { \nu } , \zeta _ { 1 , 1 6 } ^ { \nu } | , \zeta _ { 1 , 1 6 } ^ { \nu } ) } \\ &  \leq \quad \frac { \nu } { \nu } \sigma _ { 1 , 1 6 } ^ { \nu } ( | \zeta _ { 1 , 1 6 } ^ { \nu } , \zeta _ { 1 , 1 6 } ^ { \nu } | , \zeta _ { 1 , 1 6 } ^ { \nu } ) ( | \zeta _ { 1 , 1 6 } ^ { \nu } , \Gamma _ { \mathrm { e } } ^ { \prime } ( \beta ) , \zeta _ { 1 , 1 6 } ^ { \nu } | , \zeta _ { 2 , 1 6 } ^ { \nu } ) | \frac { \nu }  \ \end{array}
$$

$$
\begin{array} { r l r } {  { \lesssim } } & { { } \frac { \kappa ^ { - 1 } } { n } \sum _ { k = 0 } ^ { \infty } 2 ^ { 2 k - p k } } \\ & { { } \lesssim } & { \frac { \kappa ^ { - 1 } } { n } , ~ } \end{array}
$$

if $p > 2$ . At this point, the conclusion follows from (39).

Lemma 3. In the setup of Theorem 1, let $f _ { n }$ and ${ \breve { f } } _ { n }$ be the optimal semidual potentials for $( P _ { n } , Q _ { n } )$ and $( P , Q _ { n } )$ , respectively and satisfying (4). Then,

$$
\mathbb { E } \left( \operatorname { V a r } _ { \infty } ( f _ { n } - { \check { f } } _ { n } ) \right) \lesssim { \frac { 1 } { n } }
$$

Proof. Consider the event $\Lambda _ { n }$ defined in (35). We have

$$
\begin{array} { r l } { \mathbb { E } \left[ \mathrm { V a r } _ { \infty } ( f _ { n } - \check { f } _ { n } ) _ { \infty } \right] } & { = \begin{array} { r l } { \mathbb { E } \left( 1 _ { \Lambda _ { n } } \mathrm { V a r } _ { \infty } ( f _ { n } - \check { f } _ { n } ) _ { \infty } \right) + \mathbb { E } \left( 1 _ { \Lambda _ { n } ^ { c } } \mathrm { V a r } _ { \infty } ( f _ { n } - \check { f } _ { n } ) _ { \infty } \right) } \end{array} } \\ & { \lesssim \begin{array} { r l } { \frac { \kappa } { \underline { { \alpha } } ^ { 2 } } \mathbb { E } \left( \chi ^ { 2 } ( P _ { n } \| P ) \right) + \mathbb { P } ( \Lambda _ { n } ^ { c } ) ^ { 1 / 2 } \mathbb { E } \left( \| f _ { n } \| _ { \infty } ^ { 4 } + \| \check { f } _ { n } \| _ { \infty } ^ { 4 } \right) ^ { 1 / 2 } } \end{array} } \\ & { \lesssim \begin{array} { r l } { \frac { 1 } { n } + \exp ( - c n / 2 ) \mathbb { E } \left( \big [ R ^ { 2 } + d ( \hat { \varepsilon } ^ { 2 } + R ^ { 2 } ) + d ^ { 2 } ( \hat { \varepsilon } ^ { 2 } + R ^ { 2 } ) ^ { 2 } \big ] ^ { 4 } \right) ^ { 1 / 2 } } \end{array} } \\ & { \lesssim \begin{array} { r l } { \frac { 1 } { n } + \exp ( - c n / 2 ) \left( R ^ { 8 } + d ^ { 4 } ( \varepsilon ^ { 8 } + R ^ { 8 } ) + d ^ { 8 } ( \varepsilon ^ { 1 6 } + R ^ { 1 6 } ) \right) ^ { 1 / 2 } } \end{array} } \\ & { \lesssim \begin{array} { r l } { \frac { 1 } { n } + r _ { n , d } . } \end{array} } \end{array}
$$

In the second line, we used Lemma 4 with $( P , Q _ { n } )$ and $( P _ { n } , Q _ { n } )$ (note that no condition is imposed on the second measure), the crude bound $\operatorname { V a r } _ { \infty } ( f ) \leq \| f \| _ { \infty } ^ { 2 }$ (Lemma 20), and Cauchy-Schwarz. In the third line, we used the weaker, dimension-dependent bound Proposition 6 for the potentials. These bounds depend on the random subGaussianity parameter $\widetilde { \varepsilon }$ described in the proof of 2. In the fourth line, we control the moments of $\widetilde { \varepsilon }$ using the same argument as in the proof of Lemma 2. □

## A.5 Proof of Theorem 2

Proof. We have

$$
\begin{array} { r l r } { p ( x , y ) } & { = } & { \exp \left( \frac { f ( x ) + g ( y ) - c ( x , y ) } { \sigma ^ { 2 } } \right) = \frac { \exp \Big ( \frac { f ( x ) - c ( x , y ) } { \sigma ^ { 2 } } \Big ) } { \sum _ { k = 1 } ^ { K } \alpha _ { k } \exp \Big ( \frac { f ( x _ { k } ) - c ( x _ { k } , y ) } { \sigma ^ { 2 } } \Big ) } , } \\ { p _ { n } ( x , y ) } & { = } & { \exp \left( \frac { f _ { n } ( x ) + g _ { n } ( y ) - c ( x , y ) } { \sigma ^ { 2 } } \right) = \frac { \exp \Big ( \frac { f _ { n } ( x ) - c ( x , y ) } { \sigma ^ { 2 } } \Big ) } { \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { n } \exp \Big ( \frac { f _ { n } ( x _ { k } ) - c ( x _ { k } , y ) } { \sigma ^ { 2 } } \Big ) } . } \end{array}
$$

These expressions are well defined for $x = x _ { k } , k \in [ K ] , y \in \mathbb { R } ^ { d }$ through the canonical extensions. The righ-hand sides follow by replacing (5b) in the exponents. We can bound the diference using the bound $| e ^ { a } - e ^ { b } | \leq e ^ { \operatorname* { m a x } \{ a , b \} } | a - b |$ . This leads to

$$
| p ( x , y ) - p _ { n } ( x , y ) | ~ \leq ~ \frac { \operatorname* { m a x } \{ p ( x , y ) , p _ { n } ( x , y ) \} } { \sigma ^ { 2 } } \left( | f _ { n } ( x ) - f ( x ) + g _ { n } ( y ) - g ( y ) | \right)
$$

We now bound the expectation on the event $\Lambda _ { n }$ defined in (35) and its complement. Inside $\Lambda _ { n } ,$ for each $k \in [ K ]$ we have

$$
p ( x _ { k } , y ) = { \frac { 1 } { \alpha _ { k } } } \Pi _ { k } ( f , \alpha , y ) \leq { \frac { 1 } { \underline { { \alpha } } } } , \quad p _ { n } ( x _ { k } , y ) = { \frac { 1 } { \alpha _ { k } ^ { n } } } \Pi _ { k } ( f _ { n } , \alpha ^ { n } , y ) \leq { \frac { 2 } { \underline { { \alpha } } } } .
$$

Therefore, on $\Lambda _ { n }$

$$
| p ( x , y ) - p _ { n } ( x , y ) | \lesssim | f _ { n } ( x ) - f ( x ) | + | g _ { n } ( y ) - g ( y ) |
$$

Integrating with respect to $Q ,$ and then taking the maximum over $k ,$ gives

$$
\begin{array} { r } { \mathbf { 1 } _ { \Lambda _ { n } } \| p _ { n } - p \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 2 } \lesssim \| f _ { n } - f \| _ { \infty } ^ { 2 } + \| g _ { n } - g \| _ { L ^ { 2 } ( Q ) } ^ { 2 } . } \end{array}
$$

Taking expectations and and using the estimates from Theorem 1 on $\Lambda _ { n }$

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \Lambda _ { n } } \big \| p _ { n } - p \big \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 2 } \right] \lesssim \frac { 1 } { n } .
$$

It remains to control the complement $\Lambda _ { n } ^ { c }$ . Since

$$
\| p \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 2 } = \operatorname* { m a x } _ { k \in [ K ] } \int p _ { k } ( y ) ^ { 2 } \mathrm { d } Q ( y ) \le \operatorname* { m a x } _ { k \in [ K ] } \int \left[ \frac { 1 } { \alpha _ { k } } \right] ^ { 2 } \mathrm { d } Q ( y ) \le \underline { { \alpha } } ^ { - 2 } .
$$

we get

$$
\begin{array} { r l } { \mathbb E \left[ \mathbf { 1 } _ { \Lambda _ { n } ^ { c } } \| p _ { n } - p \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 2 } \right] \leq 2 \mathbb E \left[ \mathbf { 1 } _ { \Lambda _ { n } ^ { c } } \| p _ { n } \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 2 } \right] + 2 \underline { { \alpha } } ^ { - 2 } \mathbb P ( \Lambda _ { n } ^ { c } ) } & { } \\ { \leq 2 \mathbb P ( \Lambda _ { n } ^ { c } ) ^ { 1 / 2 } \mathbb E \left[ \| p _ { n } \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 4 } \right] ^ { 1 / 2 } + 2 \underline { { \alpha } } ^ { - 2 } \mathbb P ( \Lambda _ { n } ^ { c } ) } \end{array}
$$

Since, by Lemma 7, $P ( \Lambda _ { n } ^ { c } )$ decays exponentially fast, it only sufices to show that the growth of E $\left[ \| p _ { n } \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 4 } \right]$ is bounded by a polynomial. We devote the rest of this proof to show that for every fixed integer $\mathbf { \bar { \Phi } } _ { m } \geq 1$ -，

$$
\mathbb { E } \left[ \| p _ { n } \| _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 2 m } \right] \lesssim n ^ { 2 m + 1 } .\tag{40}
$$

Indeed, let $A _ { n }$ be the set of active indexes defined in (34). Take an arbitrary $k _ { n } \in A _ { n }$ so that $\alpha _ { k _ { n } } ^ { n } \geq 1 / n$ and therefore for each $y \in \mathbb { R } ^ { d } , p _ { n } ( x _ { k _ { n } } , y ) \leq 1 / \alpha _ { k _ { n } } ^ { n } \leq n$ . Note that, by rearranging terms, we can write for an arbitrary $k \in [ K ]$

$$
p _ { n } ( x _ { k } , y ) ~ = ~ p _ { n } ( x _ { k _ { n } } , y ) \exp \left( { \frac { f _ { n } ( x _ { k } ) - f _ { n } ( x _ { k _ { n } } ) + c ( x _ { k _ { n } } , y ) - c ( x _ { k } , y ) } { \sigma ^ { 2 } } } \right)
$$

$$
\leq \begin{array} { l l } { \displaystyle \leq } & { \displaystyle \ n \exp \left( \frac { f _ { n } ( x _ { k } ) - f _ { n } ( x _ { k _ { n } } ) } { \sigma ^ { 2 } } \right) \exp \left( \frac { c ( x _ { k _ { n } } , y ) - c ( x _ { k } , y ) } { \sigma ^ { 2 } } \right) . } \end{array}
$$

And so, for each m

$$
\mathbb { E } \left( \left. p _ { n } \right. _ { L ^ { \infty } ( P ; L ^ { 2 } ( Q ) ) } ^ { 2 m } \right) \leq n ^ { 2 m } \mathbb { E } \left[ \exp \left( \frac { 2 m } { \sigma ^ { 2 } } \operatorname* { m a x } _ { k , k ^ { \prime } \in [ K ] } \left. f _ { n } ( x _ { k } ) - f _ { n } ( x _ { k ^ { \prime } } ) \right. \right) \right] \int \exp \left( \frac { 2 m } { \sigma ^ { 2 } } \tilde { L } ( y ) \right) \mathrm { d } Q ( y ) ,\tag{41}
$$

$$
\widetilde L ( y ) : = \operatorname* { m a x } _ { k , k ^ { \prime } \in [ K ] } | c ( x _ { k } , y ) - c ( x _ { k ^ { \prime } } , y ) | .
$$

we used Jensen’s inequality to the function $x \to x ^ { m }$ to write m inside integration with respect to $Q .$ We now bound the two exponential terms. We bound the second exponential as follows

$$
\begin{array} { r c l } { \displaystyle \int \exp \left( \frac { 2 m } { \sigma ^ { 2 } } \widetilde { L } ( y ) \right) \mathrm { d } Q ( y ) } & { \leq } & { \displaystyle \exp \left( \frac { 2 m } { \sigma ^ { 2 } } R ^ { 2 } \right) \sum _ { k , k ^ { \prime } = 1 } ^ { K } \int \exp \left( \frac { 2 m } { \sigma ^ { 2 } } | \langle x _ { k } - x _ { k ^ { \prime } } , y \rangle | \right) \mathrm { d } Q ( y ) } \\ & { \leq } & { \displaystyle 2 \exp \left( \frac { 2 m } { \sigma ^ { 2 } } R ^ { 2 } \right) \sum _ { k , k ^ { \prime } = 1 } ^ { K } \int \exp \left( \frac { 2 m ^ { 2 } \| x _ { k } - x _ { k ^ { \prime } } \| ^ { 2 } \varepsilon ^ { 2 } } { \sigma ^ { 4 } } \right) } \\ & { \leq } & { \displaystyle 2 K ^ { 2 } \exp \left( \frac { 2 m } { \sigma ^ { 2 } } R ^ { 2 } \right) \exp \left( \frac { 4 m ^ { 2 } R ^ { 2 } \varepsilon ^ { 2 } } { \sigma ^ { 4 } } \right) } \end{array}
$$

We first expectation is a random quantity and we will bound its expectation. First note that

$$
\operatorname* { m a x } _ { k , k ^ { \prime } \in [ K ] } | f _ { n } ( x _ { k } ) - f _ { n } ( x _ { k ^ { \prime } } ) | \leq M _ { n } : = \operatorname* { m a x } _ { i \leq n } \widetilde { L } ( Y _ { i } ) .\tag{42}
$$

Indeed, by (5a)

$$
f _ { n } ( x _ { k } ) = - \sigma ^ { 2 } \log \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \exp \left( \frac { g _ { n } ( Y _ { i } ) - c ( x _ { k } , Y _ { i } ) } { \sigma ^ { 2 } } \right) .
$$

Therefore, for every sample point $Y _ { i }$

$$
- M _ { n } \leq c ( x _ { k } , Y _ { i } ) - c ( x _ { k ^ { \prime } } , Y _ { i } ) \leq M _ { n } ,
$$

and so

$$
\begin{array} { r } { e ^ { - M _ { n } / \sigma ^ { 2 } } e ^ { ( g _ { n } ( Y _ { i } ) - c ( x _ { k ^ { \prime } } , Y _ { i } ) ) / \sigma ^ { 2 } } \le e ^ { ( g _ { n } ( Y _ { i } ) - c ( x _ { k } , Y _ { i } ) ) / \sigma ^ { 2 } } \le e ^ { M _ { n } / \sigma ^ { 2 } } e ^ { ( g _ { n } ( Y _ { i } ) - c ( x _ { k ^ { \prime } } , Y _ { i } ) ) / \sigma ^ { 2 } } . } \end{array}
$$

Averaging over samples and taking logarithms yields (42). We can also bound $M _ { n }$ as follows

$$
M _ { n } \leq R ^ { 2 } + \operatorname* { m a x } _ { 1 \leq i \leq n } \operatorname* { m a x } _ { k , k ^ { \prime } \in [ K ] } | \langle x _ { k } - x _ { k ^ { \prime } } , Y _ { i } \rangle | .
$$

Therefore, for every fixed $m \geq 1$ ，

$$
\mathbb { E } \exp \left( \frac { 2 m { \cal M } _ { n } } { \sigma ^ { 2 } } \right) ~ \le ~ e ^ { 2 m R ^ { 2 } / \sigma ^ { 2 } } \mathbb { E } \exp \left( \frac { 2 m } { \sigma ^ { 2 } } \operatorname* { m a x } _ { i , k , k ^ { \prime } } | \langle x _ { k } - x _ { k ^ { \prime } } , Y _ { i } \rangle | \right)
$$

$$
\begin{array} { r l } { \leq } & { { } e ^ { 2 m R ^ { 2 } / \sigma ^ { 2 } } \displaystyle \sum _ { i = 1 } ^ { n } \displaystyle \sum _ { k , k ^ { \prime } = 1 } ^ { K } \mathbb { E } \exp \left( \frac { 2 m } { \sigma ^ { 2 } } | \langle x _ { k } - x _ { k ^ { \prime } } , Y _ { i } \rangle | \right) } \\ { \leq } & { { } n 2 K ^ { 2 } e ^ { 2 m R ^ { 2 } / \sigma ^ { 2 } } \exp \left( \frac { 8 m ^ { 2 } R ^ { 2 } \varepsilon ^ { 2 } } { \sigma ^ { 4 } } \right) . } \end{array}\tag{43}
$$

Combining (41), (42) and (43) we get (40).

## A.6 Proof of Theorem 3

Proof. Recall that

$$
\overleftarrow { T } ( y ) = \int x p ( x , y ) \mathrm { d } P ( x ) = \sum _ { k = 1 } ^ { K } \alpha _ { k } x _ { k } p ( x _ { k } , y ) ,
$$

and

$$
\overleftarrow { T } _ { n } ( y ) = \int x p _ { n } ( x , y ) \mathrm { d } P _ { n } ( x ) = \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { n } x _ { k } p _ { n } ( x _ { k } , y ) ,
$$

Similarly,

$$
\vec { T } ^ { \prime } ( x _ { k } ) = \int y p ( x _ { k } , y ) \mathrm { d } Q ( y ) , \qquad \vec { T } _ { n } ( x _ { k } ) = \int y p _ { n } ( x _ { k } , y ) \mathrm { d } Q _ { n } ( y ) .
$$

From the facts that

$$
\int p _ { k } ( x _ { k } , y ) \mathrm { d } Q ( y ) = 1 , \int p _ { n } ( x _ { k } , y ) \mathrm { d } Q _ { n } ( y ) = 1 { \mathrm { ~ f o r ~ } } k \in [ K ] , \quad { \mathrm { ~ a n d ~ } } \sum _ { k = 1 } ^ { K } \alpha _ { k } ^ { n } p _ { n } ( x _ { k } , Y _ { i } ) = 1 , { \mathrm { ~ f o r ~ } } 1 \leq i \leq n ,
$$

It follows that for $k \in [ K ] , { \stackrel {  } { T } } _ { n } ( x _ { k } )$ is a convex combination of the sample points $Y _ { 1 } , \dots , Y _ { n }$ and that for every $y , \overleftarrow { T } _ { n } ( y )$ and $\overleftarrow { T } \left( y \right)$ are convex combinations of the atoms $x _ { k } , k \in [ K ]$ . We will use the following pointwise bound from the proof of Theorem 2:

$$
| p _ { n } ( x _ { k } , y ) - p ( x _ { k } , y ) | \leq \operatorname* { m a x } \{ p _ { n } ( x _ { k } , y ) , p ( x _ { k } , y ) \} \frac { | f _ { n } ( x _ { k } ) - f ( x _ { k } ) | + | g _ { n } ( y ) - g ( y ) | } { \sigma ^ { 2 } } .
$$

Let $\Lambda _ { n }$ be as in (35). It follows from the bounds on $g _ { n } ( y ) - g ( y )$ in the proof of Theorem 1 that on $\Lambda _ { n }$

$$
| g _ { n } ( y ) - g ( y ) | \lesssim \| f _ { n } - f \| _ { \infty } + \| \alpha ^ { n } - \alpha \| _ { \infty } .
$$

Therefore, on $\Lambda _ { n }$

$$
\operatorname* { s u p } _ { k \in [ K ] } \operatorname* { s u p } _ { y \in \mathbb { R } ^ { d } } | p _ { n } ( x _ { k } , y ) - p ( x _ { k } , y ) | \lesssim \| f _ { n } - f \| _ { \infty } + \| \alpha ^ { n } - \alpha \| _ { \infty } .\tag{44}
$$

We first prove the bound for $\overleftarrow { T } _ { n } - \overleftarrow { T } . \ \mathrm { O n } \ \Lambda _ { n } .$ , using (44) and $p ( x _ { k } , y ) \leq \underline { { \alpha } } ^ { - 1 }$

$$
\begin{array} { l } { \displaystyle \| \overleftarrow { T } _ { n } ( y ) - \overleftarrow { T } ( y ) \| \leq \displaystyle \sum _ { k = 1 } ^ { K } \| x _ { k } \| | \alpha _ { k } ^ { n } p _ { n } ( x _ { k } , y ) - \alpha _ { k } p ( x _ { k } , y ) | } \\ { \leq R \displaystyle \sum _ { k = 1 } ^ { K } [ \alpha _ { k } ^ { n } | p _ { n } ( x _ { k } , y ) - p ( x _ { k } , y ) | + | \alpha _ { k } ^ { n } - \alpha _ { k } | p ( x _ { k } , y ) ] } \\ { \lesssim \| f _ { n } - f \| _ { \infty } + \| \alpha ^ { n } - \alpha \| _ { \infty } . } \end{array}
$$

Taking the supremum over y,

$$
\begin{array} { r } { \mathbf { 1 } _ { \Lambda _ { n } } \| \overset {  } { T } _ { n } - \overset {  } { T } \| _ { L ^ { \infty } ( Q ) } ^ { 2 } \lesssim 1 _ { \Lambda _ { n } } \| f _ { n } - f \| _ { \infty } ^ { 2 } + 1 _ { \Lambda _ { n } } \| \alpha ^ { n } - \alpha \| _ { \infty } ^ { 2 } . } \end{array}
$$

Hence, by the bounds on $\Lambda _ { n }$ in the proof of Theorem 1 and the multinomial bound in Lemma $^ { 7 , }$

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \Lambda _ { n } } \big \| \overleftarrow { T } _ { n } - \overleftarrow { T } \big \| _ { L ^ { \infty } ( Q ) } ^ { 2 } \right] \lesssim \frac { 1 } { n } .
$$

On $\Lambda _ { n } ^ { c }$ , both ${ \overleftarrow { T } } _ { n } ( y )$ and $\left. _ { \left( y \right) } \right.$ are convex combinations of $x _ { k } { \mathrm { ' s . } }$ . Therefore $\| { \overleftarrow { T } } _ { n } - { \overleftarrow { T } } \| _ { L ^ { \infty } ( Q ) } \leq 2 R$ on $\Lambda _ { n } ^ { c }$ , and so, again by Lemma 7

$$
\begin{array} { r } { \mathbb { E } [ \mathbf { 1 } _ { \Lambda _ { n } ^ { c } } \big \Vert \overset {  } { T } _ { n } - \overset {  } { T } \big \Vert _ { L ^ { \infty } ( Q ) } ^ { 2 } ] \leq 4 R ^ { 2 } \mathbb { P } ( \Lambda _ { n } ^ { c } ) \lesssim e ^ { - c \underline { { \alpha } } n } . } \end{array}
$$

Combining the bounds on $\Lambda _ { n }$ and $\Lambda _ { n } ^ { c } ,$ , we conclude the bound for the backward barycentric projection. We now establish the bound for the forward diference $\vec { T } _ { n } - \vec { T }$ . For each $k \in [ K ]$ , decompose

$$
\vec { T } _ { n } ( x _ { k } ) - \vec { T } ( x _ { k } ) = \underbrace { \int ( y p ( x _ { k } , y ) ) \mathrm { d } ( Q _ { n } - Q ) ( y ) } _ { A _ { n , k } } + \underbrace { \int \vert y ( p _ { n } ( x _ { k } , y ) - p ( x _ { k } , y ) ) \vert \mathrm { d } Q _ { n } ( y ) } _ { B _ { n , k } } .
$$

To bound $A _ { n , k }$ , define the variables

$$
Z _ { i , k } : = Y _ { i } p ( x _ { k } , Y _ { i } ) , \qquad Z _ { k } : = Y p ( x _ { k } , Y ) ,
$$

where $Y \sim Q$ . Then

$$
A _ { n , k } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ Z _ { i , k } - \mathbb { E } Z _ { k } \right] ,
$$

and $\operatorname { s o } .$ , by independence

$$
\mathbb { E } \Vert A _ { n , k } \Vert ^ { 2 } = \mathbb { E } \left. \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ Z _ { i , k } - \mathbb { E } Z _ { k } \right] \right. ^ { 2 } = \frac { 1 } { n } \mathbb { E } \Vert Z _ { k } - \mathbb { E } Z _ { k } \Vert ^ { 2 } \leq \frac { 1 } { n } \mathbb { E } \Vert Z _ { k } \Vert ^ { 2 } = \frac { 1 } { n } \mathbb { E } \Vert Y p _ { k } ( Y ) \Vert ^ { 2 } .
$$

Therefore, since $p ( x _ { k } , y ) \leq \underline { { \alpha } } ^ { - 1 }$ , and by subGaussianity,

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { k \in [ K ] } \| A _ { n , k } \| ^ { 2 } \right] \leq \sum _ { k = 1 } ^ { K } \mathbb { E } \| A _ { n , k } \| ^ { 2 } \leq \frac { K } { \underline { { \alpha } } ^ { 2 } n } E \| Y \| ^ { 2 } \leq \frac { K d \varepsilon ^ { 2 } } { \underline { { \alpha } } ^ { 2 } n } .
$$

For the second term, define $\Delta _ { n , k } ( y ) : = p _ { n } ( x _ { k } , y ) - p ( x _ { k } , y )$ so that

$$
B _ { n , k } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Y _ { i } \Delta _ { n , k } ( Y _ { i } ) .
$$

Note that, by Cauchy–Schwarz:

$$
\begin{array} { l } { \displaystyle \| B _ { n , k } \| ^ { 2 } = \left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } Y _ { i } \Delta _ { n , k } ( Y _ { i } ) \right\| ^ { 2 } } \\ { \displaystyle \quad \leq \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| Y _ { i } \| | \Delta _ { n , k } ( Y _ { i } ) | \right] ^ { 2 } } \\ { \displaystyle \quad \leq \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| Y _ { i } \| ^ { 2 } \right] \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Delta _ { n , k } ( Y _ { i } ) ^ { 2 } \right] . } \end{array}
$$

Taking the maximum over $k \in [ K ]$ and $\boldsymbol { y } \in \mathbb { R } ^ { d }$ we obtain

$$
\operatorname* { m a x } _ { k \in [ K ] } \| B _ { n , k } \| ^ { 2 } \leq \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| Y _ { i } \| ^ { 2 } \operatorname* { s u p } _ { k \in [ K ] , y \in \mathbb { R } ^ { d } } | p _ { n } ( x _ { k } , y ) - p ( x _ { k } , y ) | ^ { 2 } .
$$

Additionally, on $\Lambda _ { n } ,$ , by (44) we can bound further, getting

$$
\mathbf { 1 } _ { \Lambda _ { n } } \operatorname* { m a x } _ { k \in [ K ] } \| B _ { n , k } \| ^ { 2 } \lesssim \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| Y _ { i } \| ^ { 2 } \right] \left[ \| f _ { n } - f \| _ { \infty } ^ { 2 } + \| \alpha ^ { n } - \alpha \| _ { \infty } ^ { 2 } \right] .
$$

Taking expectations and applying Cauchy–Schwarz again,

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \Lambda _ { n } } \operatorname* { m a x } _ { k \in [ K ] } \| B _ { n , k } \| ^ { 2 } \right] \leq \mathbb { E } \left[ \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| Y _ { i } \| ^ { 2 } \right] ^ { 2 } \right] ^ { 1 / 2 } \mathbb { E } \left[ 2 \mathbf { 1 } _ { \Lambda _ { n } } \| f _ { n } - f \| _ { \infty } ^ { 4 } + 2 \mathbf { 1 } _ { \Lambda _ { n } } \| \alpha ^ { n } - \alpha \| _ { \infty } ^ { 4 } \right] ^ { 1 / 2 } .
$$

Let’s bound the two expectations on the right-hand side. For the first one, note that

$$
\begin{array} { r l } { \mathbb { E } \left[ \left[ \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| Y _ { i } \| ^ { 2 } \right] ^ { 2 } \right] } & { = \mathbb { E } \left( \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| Y _ { i } \| ^ { 2 } \right) ^ { 2 } } \\ & { \quad \quad \quad = \displaystyle \frac { 1 } { n ^ { 2 } } \left[ \sum _ { i = 1 } ^ { n } \mathbb { E } \| Y _ { i } \| ^ { 4 } + \sum _ { i \neq j } \mathbb { E } ( \| Y _ { i } \| ^ { 2 } \| Y _ { j } \| ^ { 2 } ) \right] } \\ & { \quad \quad = \displaystyle \frac { 1 } { n } \mathbb { E } \| Y \| ^ { 4 } + \frac { n - 1 } { n } \left( \mathbb { E } \| Y \| ^ { 2 } \right) ^ { 2 } } \\ & { \quad \quad \quad \leq \mathbb { E } \| Y \| ^ { 4 } } \\ & { \quad \quad \quad \leq d _ { 2 } ^ { 2 } \varepsilon ^ { 4 } . } \end{array}
$$

where in the second-to-last inequality follows from Cauchy–Schwarz and in the last one an elementary moment bound for a $\varepsilon ^ { 2 } .$ -subGaussian vector [2]. Using Corollary 3 and the multinomial moment bound in Lemma 7 with $q = 2$ , we get

$$
\begin{array} { r } { \mathbb { E } \| f _ { n } - f \| _ { \infty } ^ { 4 } \lesssim n ^ { - 2 } , \quad \mathrm { a n d } \quad \mathbb { E } \| \alpha ^ { n } - \alpha \| _ { \infty } ^ { 4 } \lesssim n ^ { - 2 } . } \end{array}
$$

Putting everything together,

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \Lambda _ { n } } \operatorname* { m a x } _ { k \in \left[ K \right] } \| B _ { n , k } \| ^ { 2 } \right] \lesssim \frac { d } { n } .
$$

It remains to control $B _ { n , k }$ on $\Lambda _ { n } ^ { c }$ . Since $\vec { T } _ { n } ( x _ { k } )$ is a convex combination of the sample points, and since $p ( x _ { k } , y ) \leq \underline { { \alpha } } ^ { - 1 }$

$$
\operatorname* { m a x } _ { k \in [ K ] } \bigg \| \int y p _ { n } ( x _ { k } , y ) \mathrm { d } Q _ { n } ( y ) \bigg \| \leq \operatorname* { m a x } _ { 1 \leq i \leq n } \| Y _ { i } \| , \quad \mathrm { a n d } \quad \operatorname* { m a x } _ { k \in [ K ] } \bigg \| \int y p ( x _ { k } , y ) \mathrm { d } Q _ { n } ( y ) \bigg \| \leq \frac { 1 } { \underline { { \alpha } } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \| Y _ { i } \| \leq \underline { { \alpha } } ^ { - 1 } \operatorname* { m a x } _ { 1 \leq i \leq n } \| Y _ { i } \| .
$$

This implies that

$$
{ \bf 1 } _ { \Lambda _ { n } ^ { c } } \operatorname* { m a x } _ { k \in \left[ K \right] } \| B _ { n , k } \| ^ { 2 } \lesssim { \bf 1 } _ { \Lambda _ { n } ^ { c } } \left[ \operatorname* { m a x } _ { 1 \leq i \leq n } \| Y _ { i } \| ^ { 2 } \right] .
$$

Now, by Cauchy–Schwarz, the exponential bound in Lemma 7, and a subGaussianity moment bound

$$
\begin{array} { r l } { \mathbb E \left[ \mathbf { 1 } _ { \Lambda _ { n } ^ { c } } \underset { k \in [ K ] } { \operatorname* { m a x } } \ \lVert B _ { n , k } \rVert ^ { 2 } \right] \leq \mathbb E \left[ \mathbf { 1 } _ { \Lambda _ { n } ^ { c } } \underset { 1 \leq i \leq n } { \operatorname* { m a x } } \ \lVert Y _ { i } \rVert ^ { 2 } \right] } & { } \\ & { \leq \mathbb P ( \Lambda _ { n } ^ { c } ) ^ { 1 / 2 } \left( \mathbb E \underset { 1 \leq i \leq n } { \operatorname* { m a x } } \ \lVert Y _ { i } \rVert ^ { 4 } \right) ^ { 1 / 2 } } \\ & { \leq \mathbb P ( \Lambda _ { n } ^ { c } ) ^ { 1 / 2 } \left( \underset { i = 1 } { \overset { n } { \sum } } \mathbb E \lVert Y _ { i } \rVert ^ { 4 } \right) ^ { 1 / 2 } } \\ & { \lesssim e ^ { - c \underline { { \alpha } } n / 2 } \sqrt { n } d \varepsilon ^ { 2 } \lesssim \frac { d } { n } , } \end{array}
$$

Combining the bounds for $A _ { n , k }$ and $B _ { n , k }$ , we obtain

$$
\mathbb { E } \Vert \overrightarrow { T } _ { n } - \overrightarrow { T } \Vert _ { \infty } ^ { 2 } = \mathbb { E } \operatorname* { m a x } _ { k \in [ K ] } \Vert \overrightarrow { T } _ { n } ( x _ { k } ) - \overrightarrow { T } ( x _ { k } ) \Vert ^ { 2 } \lesssim \mathbb { E } \operatorname* { m a x } _ { k \in [ K ] } \big [ \Vert A _ { n , k } \Vert ^ { 2 } + \Vert B _ { n , k } \Vert ^ { 2 } \big ] \lesssim \frac { d } { n } .
$$

The proof is complete.

## A.7 Proof of Proposition 1

Proof. Fix $a \in \mathbb { R } ^ { d }$ with $\left\| a \right\| = R$ , and let

$$
r = { \frac { 1 } { 4 { \sqrt { n } } } } .
$$

Consider the two hypotheses

$$
P _ { 0 } = \frac { 1 } { 2 } \delta _ { - a } + \frac { 1 } { 2 } \delta _ { a } , \qquad P _ { 1 } = \left( \frac { 1 } { 2 } - r \right) \delta _ { - a } + \left( \frac { 1 } { 2 } + r \right) \delta _ { a } ,
$$

together with the common source measure

$$
Q = \delta _ { 0 } .
$$

Both pairs $( P _ { j } , Q )$ belong to ${ \mathcal { C } } ,$ since $r \leq 1 / 4$ , δ<sub>0</sub> is $\varepsilon ^ { 2 } .$ -subGaussian, and $\mathbb { E } _ { Q } Y = 0$

Because $Q$ is supported on a single point, there is only one coupling between $P _ { j }$ and $Q ,$ , namely

$$
\pi _ { j } = P _ { j } \otimes \delta _ { 0 } .
$$

It is therefore the entropic optimal coupling for every $\sigma ^ { 2 } > 0$ . Writing $T _ { j } = \overleftarrow { T } _ { P _ { j } , Q }$ , its backward barycentric projection satisfies

$$
\begin{array} { r } { T _ { j } ( 0 ) = \mathbb { E } _ { P _ { j } } [ X ] , } \end{array}
$$

and hence

$$
T _ { 0 } ( 0 ) = 0 , \qquad T _ { 1 } ( 0 ) = 2 r a .
$$

Therefore,

$$
\Vert T _ { 1 } - T _ { 0 } \Vert _ { L ^ { 2 } ( Q ) } ^ { 2 } = 4 r ^ { 2 } R ^ { 2 } .
$$

The $Q \mathrm { - }$ -samples are identical under the two hypotheses, so the total variation distance between the two experiments equals

$$
d _ { \mathrm { T V } } \left( P _ { 0 } ^ { n } \otimes Q ^ { n } , P _ { 1 } ^ { n } \otimes Q ^ { n } \right) = d _ { \mathrm { T V } } ( P _ { 0 } ^ { n } , P _ { 1 } ^ { n } ) .
$$

Moreover,

$$
{ \mathsf { K L } } ( P _ { 0 } \| P _ { 1 } ) = - { \frac { 1 } { 2 } } \log ( 1 - 4 r ^ { 2 } ) \leq { \frac { 8 } { 3 } } r ^ { 2 } ,
$$

where the inequality uses $r \leq 1 / 4$ . By tensorization and Pinsker’s inequality,

$$
d _ { \mathrm { T V } } ( P _ { 0 } ^ { n } , P _ { 1 } ^ { n } ) \leq \sqrt { \frac { n } { 2 } { \mathsf { K L } } ( P _ { 0 } \| P _ { 1 } ) } \leq 2 r \sqrt { \frac { n } { 3 } } \leq \frac { 1 } { 2 } .
$$

Let $\mathbb { E } _ { j }$ denote expectation under $P _ { j } ^ { n } \otimes Q ^ { n }$ . Le Cam’s two-point lemma now gives

$$
\begin{array} { r l } & { \underset { \widehat { T } } { \operatorname* { i n f } } \ \underset { j \in \{ 0 , 1 \} } { \operatorname* { s u p } } \mathbb { E } _ { j } \left[ \| \widehat { T } - T _ { j } \| _ { L ^ { 2 } ( Q ) } ^ { 2 } \right] \geq \frac { \| T _ { 1 } - T _ { 0 } \| _ { L ^ { 2 } ( Q ) } ^ { 2 } } { 8 } \left( 1 - d _ { \mathrm { T V } } ( P _ { 0 } ^ { n } , P _ { 1 } ^ { n } ) \right) } \\ & { \qquad \quad \geq \frac { 4 r ^ { 2 } R ^ { 2 } } { 1 6 } = \frac { R ^ { 2 } } { 6 4 n } . } \end{array}
$$

Since both hypotheses belong to ${ \mathcal { C } } ,$ this proves the claim.

## B Other technical lemmata for Section 3

The following upper bound addresses a one-sample case, and it is helpful in the proof of Lemma 3.

Lemma 4. ${ \cal I f } \left( f , g \right)$ and $( f ^ { \prime } , g ^ { \prime } )$ are the optimal potentials for $( P , Q )$ and $( P ^ { \prime } , Q )$ where Q is $\varepsilon ^ { 2 }$ subGaussian and $P , P ^ { \prime }$ are both semidiscrete supported on the same $x _ { k } \mathit { ^ { \prime } s }$ and such that $P ^ { \prime } \ll P$ Then,

$$
\operatorname { V a r } _ { \infty } \left( f ^ { \prime } - f \right) \leq { \frac { 4 } { \kappa ^ { 2 } } } \chi ^ { 2 } \left( P ^ { \prime } \| P \right) .
$$

Proof. Using Corollary 2 and similarly as in the proof of Lemma $2 ,$ if we denote by $\Phi$ and $\Phi ^ { \prime }$ the semidual functions related to $( P , Q )$ and $( P ^ { \prime } , Q )$ , we have

$$
\begin{array} { r c l } { \operatorname { V a r } _ { \infty } ( f ^ { \prime } - f ) } & { \leq } & { \displaystyle \frac { 2 } { \kappa } \left( \Phi ( f ) - \Phi ( f ^ { \prime } ) \right) } \\ & { \leq } & { \displaystyle \frac { 2 } { \kappa } \left( \Phi ( f ) - \Phi ^ { \prime } ( f ) + \Phi ^ { \prime } ( f ) - \Phi ( f ^ { \prime } ) \right) } \\ & { \leq } & { \displaystyle \frac { 2 } { \kappa } \left( \Phi ( f ) - \Phi ^ { \prime } ( f ) + \Phi ^ { \prime } ( f ^ { \prime } ) - \Phi ( f ^ { \prime } ) \right) , } \end{array}
$$

where in the last line we used the optimality of $f ^ { \prime }$ for $\Phi ^ { \prime }$ . Now, note that most of the terms will cancel out as

$$
\Phi ( f ) = \int f ( x ) \mathrm { d } { \cal P } ( x ) - \int \Gamma _ { f } ( y ) \mathrm { d } Q ( y ) , \quad \Phi ^ { \prime } ( f ) = \int f ( x ) \mathrm { d } { \cal P } ^ { \prime } ( x ) - \int \Gamma _ { f } ( y ) \mathrm { d } Q ( y ) ,
$$

and

$$
\Phi ^ { \prime } ( f ^ { \prime } ) = \int f ^ { \prime } ( x ) \mathrm { d } { \cal P } ^ { \prime } ( x ) - \int \Gamma _ { f ^ { \prime } } ( y ) \mathrm { d } Q ( y ) , \quad \Phi ( f ^ { \prime } ) = \int f ^ { \prime } ( x ) \mathrm { d } { \cal P } ( x ) - \int \Gamma _ { f ^ { \prime } } ( y ) \mathrm { d } Q ( y ) ,
$$

implying that

$$
\begin{array} { r c l } { \displaystyle \mathrm { V a r } _ { \infty } ( f ^ { \prime } - f ) } & { \le } & { \displaystyle \frac { 2 } { \kappa } \left( \int ( f - f ^ { \prime } ) ( x ) \mathrm { d } ( P - P ^ { \prime } ) ( x ) \right) } \\ & { \le } & { \displaystyle \frac { 2 } { \kappa } \left( \frac { \theta } { 2 } \mathrm { V a r } _ { \alpha } \left( f ^ { \prime } - f \right) + \frac { 1 } { 2 \theta } \chi ^ { 2 } ( P ^ { \prime } \| P ) \right) } \\ & { \le } & { \displaystyle \frac { 2 } { \kappa } \left( \frac { \theta } { 2 } \mathrm { V a r } _ { \infty } \left( f ^ { \prime } - f \right) + \frac { 1 } { 2 \theta } \chi ^ { 2 } ( P ^ { \prime } \| P ) \right) , } \end{array}
$$

where the last line follows from (23) and the second to last from Young’s inequality, Lemma H.1 in [18], that if $P ^ { \prime } \ll P$ , then for any function f and $\theta > 0$ (in the semidiscrete case we identify Var with $\mathrm { V a r } _ { \alpha } )$

$$
\int f \mathrm { d } ( P - P ^ { \prime } ) ( x ) \leq { \frac { \theta } { 2 } } \mathrm { V a r } _ { P } ( f ) + { \frac { 1 } { 2 \theta } } \chi ^ { 2 } ( P ^ { \prime } \| P ) .
$$

We conclude the proof by taking $\theta = \kappa / 2$

Proposition 6 (Dimension dependent bounds for potentials). Suppose that $P$ is supported on the ball $B ( 0 , R )$ and that $Q$ is subGaussian with proxy $\varepsilon ^ { 2 }$ . Then, under the gauge constraint (4) the entropic optimal dual potential f satisfies

$$
| f ( x ) | \lesssim R ^ { 2 } + d ( \varepsilon ^ { 2 } + R ^ { 2 } ) + d ^ { 2 } ( \varepsilon ^ { 2 } + R ^ { 2 } ) ^ { 2 } .
$$

where $\lesssim$ indicate constants independent of $R , \varepsilon , d$

Proof. We will use Proposition 6 of [15], that the optimal entropic potentials $( f , g )$ between two subGaussian measures with same proxy $\lambda ^ { 2 }$ satisfy

$$
- d \lambda ^ { 2 } \left( 1 + { \frac { 1 } { 2 } } \left( \left\| x \right\| + { \sqrt { 2 d } } \lambda \right) ^ { 2 } \right) - 1 \leq f ( x ) \leq { \frac { 1 } { 2 } } \left( \left\| x \right\| + { \sqrt { 2 d } } \lambda \right) ^ { 2 } ,
$$

and

$$
- d \lambda ^ { 2 } \left( 1 + { \frac { 1 } { 2 } } \left( \| y \| + { \sqrt { 2 d } } \lambda \right) ^ { 2 } \right) - 1 \leq g ( y ) \leq { \frac { 1 } { 2 } } \left( \| y \| + { \sqrt { 2 d } } \lambda \right) ^ { 2 } ,
$$

under the normalization constraint that $\begin{array} { r } { \mathbb { E } _ { P } ( f ( X ) ) = \mathbb { E } _ { Q } ( g ( Y ) ) = \frac { 1 } { 2 } S ( P , Q ) } \end{array}$

To apply this result we must identify a uniform subGaussianity proxy for $P$ and $Q .$ Since $P$ is bounded, it is subGaussian with proxy $R ^ { 2 }$ . Therefore, $P , Q$ are simultaneously subGaussian with proxy $\varepsilon ^ { 2 } + R ^ { 2 }$ . Using the inequality $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$ , we obtain

$$
\begin{array} { r } { \left( \| x \| + \sqrt { 2 d } \lambda \right) ^ { 2 } \leq 2 \| x \| ^ { 2 } + 4 d \lambda ^ { 2 } . } \end{array}
$$

Since $\| x \| \leq R$ , this yields

$$
\left( \| x \| + { \sqrt { 2 d } } \lambda \right) ^ { 2 } \leq 2 R ^ { 2 } + 4 d \lambda ^ { 2 } .
$$

Substituting into the upper bound gives

$$
f ( x ) \leq { \frac { 1 } { 2 } } ( 2 R ^ { 2 } + 4 d \lambda ^ { 2 } ) \lesssim R ^ { 2 } + d \lambda ^ { 2 } .
$$

For the lower bound, we similarly obtain

$$
f ( x ) \geq - d \lambda ^ { 2 } \left( 1 + \frac { 1 } { 2 } ( 2 R ^ { 2 } + 4 d \lambda ^ { 2 } ) \right) - 1 \lesssim - d \lambda ^ { 2 } ( 1 + R ^ { 2 } + d \lambda ^ { 2 } ) .
$$

Combining the two bounds yields

$$
| f ( x ) | \lesssim R ^ { 2 } + d \lambda ^ { 2 } + d ^ { 2 } \lambda ^ { 4 } .
$$

Finally, substituting $\lambda ^ { 2 } = \varepsilon ^ { 2 } + R ^ { 2 }$ gives the claim under the constraintE $\operatorname { \dot { \prime } } _ { P } ( f ( X ) ) = { \textstyle { \frac { 1 } { 2 } } } S ( P , Q )$ . To achieve the final conclusion we define $f _ { \mathrm { n e w } } ( x ) = f ( x ) - \mathbb { E } _ { P } ( f ( X ) )$ . By definition, $f _ { \mathrm { n e w } }$ satisfies (4) and

$$
\begin{array} { r c l } { \displaystyle | f _ { \mathrm { n e w } } ( x ) | } & { \le } & { \displaystyle | f ( x ) | + \sum _ { k = 1 } ^ { K } \alpha _ { k } f ( x _ { k } ) } \\ & & { \le } & { \displaystyle | f ( x ) | + \operatorname* { m a x } _ { k \in [ K ] } | f ( x _ { k } ) | \sum _ { k = 1 } ^ { K } \alpha _ { k } } \\ & { \lesssim } & { \displaystyle 2 \| f \| _ { \infty } } \\ & { \lesssim } & { \displaystyle R ^ { 2 } + d ( \varepsilon ^ { 2 } + R ^ { 2 } ) + d ^ { 2 } ( \varepsilon ^ { 2 } + R ^ { 2 } ) ^ { 2 } } \end{array}
$$

by the bound on $f ( x )$ , so the proof is concluded

Lemma 5. Let $\widetilde { \varepsilon }$ be the infimum of $\varepsilon _ { u }$ such that $Q , Q _ { n }$ are $\varepsilon _ { u } ^ { 2 }$ subGaussian uniformly over n. Then, for any integer m $\geq 2$ there is a constant $C _ { m } > 0$ such that

$$
\mathbb { P } ( \widetilde { \varepsilon } ^ { 2 } \geq 3 m \varepsilon ^ { 2 } ) \leq C _ { m } n ^ { - \frac { m } { 2 } } .\tag{45}
$$

Proof. The proof extends the one of Lemma A.3 in [9]. Defining, for each m integer

$$
\tau _ { m , n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \exp \left( \frac { \| Y _ { i } \| ^ { 2 } } { 2 m d \varepsilon ^ { 2 } } \right) , \quad \tau _ { m } = \mathbb { E } \left( \tau _ { m , n } \right) = \mathbb { E } \left( \exp \left( \frac { \| Y \| ^ { 2 } } { 2 m d \varepsilon ^ { 2 } } \right) \right) ,
$$

By Lemma $9 ( \mathrm { a } )$ , (with $\mu = Q _ { n }$ and $\alpha = 2 m d \varepsilon ^ { 2 } )$ we conclude that $Q _ { n }$ is $\tau _ { m , n } m \varepsilon ^ { 2 }$ -subGaussian and so $\widetilde { \varepsilon } ^ { 2 } \leq m \tau _ { m , n } ^ { 2 } \varepsilon ^ { 2 }$ . Therefore,

$$
\mathbb { P } ( \tilde { \varepsilon } ^ { 2 } \geq 3 m \varepsilon ^ { 2 } ) \leq \mathbb { P } \left( \tau _ { m , n } \geq 3 \right) .
$$

We now apply Markov’s inequality to powers of centered $\tau _ { m , n }$ (the other case is analogous). Since, By Jensen’s inequality, $\tau _ { m } , \leq 2 ^ { 1 / m } \leq 2 , ( 3 - \tau _ { m } ) \geq 1$ , and so we have

$$
\mathbb { P } \left( \tau _ { m , n } \geq 3 \right) \leq \mathbb { P } \left( | \tau _ { m , n } - \tau _ { m } | ^ { m } \geq ( 3 - \tau _ { m } ) ^ { m } \right) \leq \frac { \mathbb { E } \left( | \tau _ { m , n } - \tau _ { m } | ^ { m } \right) } { ( 3 - \tau _ { m } ) ^ { m } } \leq \mathbb { E } \left( | \tau _ { m , n } - \tau _ { m } | ^ { m } \right) .
$$

It only remains to bound the last term above. The case $m = 2$ corresponds to Lemma A.3 in [9]. Now, note that $\begin{array} { r } { \tau _ { m , n } - \tau _ { m } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } X _ { i } } \end{array}$ , where

$$
X _ { i } = \exp \left( \frac { \| Y _ { i } \| ^ { 2 } } { 2 m d \varepsilon ^ { 2 } } \right) - \mathbb { E } \left( \exp \left( \frac { \| Y \| ^ { 2 } } { 2 m d \varepsilon ^ { 2 } } \right) \right)
$$

is a zero mean variable. Note also that all moments up to order m are bounded. Indeed, owing to that $| a - b | ^ { k } \leq 2 ^ { k - 1 } ( | a | ^ { k } + | b | ^ { k } )$ we get that for $m ^ { \prime } \in \{ 2 , m \}$

$$
\begin{array} { r l r } { \mathbb { E } ( | X _ { i } | ^ { m ^ { \prime } } ) } & { \leq } & { 2 ^ { m ^ { \prime } - 1 } \left( \mathbb { E } \left( \exp \left( \frac { m ^ { \prime } \| Y _ { i } \| ^ { 2 } } { 2 m d \varepsilon ^ { 2 } } \right) \right) + \mathbb { E } \left( \exp \left( \frac { \| Y _ { i } \| ^ { 2 } } { 2 m d \varepsilon ^ { 2 } } \right) \right) ^ { m ^ { \prime } } \right) } \end{array}
$$

$$
\begin{array} { r l } { \leq } & { 2 ^ { m ^ { \prime } - 1 } \left( \mathbb { E } \left( \exp \left( \displaystyle \frac { \| Y _ { i } \| ^ { 2 } } { 2 d \varepsilon ^ { 2 } } \right) \right) ^ { m ^ { \prime } } + \mathbb { E } \left( \exp \left( \displaystyle \frac { \| Y _ { i } \| ^ { 2 } } { 2 d \varepsilon ^ { 2 } } \right) \right) ^ { m ^ { \prime } } \right) } \\ { \leq } & { 2 ^ { m - 1 } \left( 2 ^ { m } + 2 ^ { m } \right) } \\ { < } & { 2 ^ { 2 m } . } \end{array}
$$

by subGaussianity, the fact that $1 / m \leq 1$ , that $m \prime \geq 2$ and Jensen’s inequality. Then, by Rosenthal’s inequality (Lemma 8), for some constants $C _ { m }$ (sometimes renamed from one to the next)

$$
\begin{array} { l l l } { \mathbb { E } \left( | \tau _ { m , n } - \tau _ { m } | ^ { m } \right) } & { \leq } & { \displaystyle \frac { 1 } { n ^ { m } } C _ { m } \operatorname* { m a x } \left\{ n \mathbb { E } | X _ { i } | ^ { m } , \left( n \mathbb { E } | X _ { i } | ^ { 2 } \right) ^ { m / 2 } \right\} } \\ & { \leq } & { C _ { m } \left( n ^ { 1 - m } + n ^ { - m / 2 } \right) } \\ & { \leq } & { C _ { m } n ^ { - m / 2 } . } \end{array}
$$

where $C _ { m , j }$ are polynomial expressions involving moments of $\tau _ { m }$ up to order $m$ . These moments are all bounded by 2, by subGaussianity. □

The following the following $L ^ { 2 } \mathrm { - t y p e }$ convergence of the empirical potentials at the rate $n ^ { - 1 }$ :

Lemma 6. For vectors $x _ { k } \in B ( 0 , R ) \subseteq \mathbb { R } ^ { d } , k \in [ K ]$ , and a vector $f = f ( x _ { 1 } , \dots , x _ { k } ) \in \mathbb { R } ^ { K }$ , define the function $\Gamma _ { f } : \mathbb { R } ^ { d }  \mathbb { R }$ as $( \sigma ^ { 2 }$ , α are constants)

$$
\Gamma _ { f } ( y ) : = \sigma ^ { 2 } \log \left( \sum _ { k = 1 } ^ { K } \alpha _ { k } e ^ { ( f _ { k } - \frac 1 2 \| y - x _ { k } \| ^ { 2 } ) / \sigma ^ { 2 } } \right) .\tag{46}
$$

Then, for any two vectors $f = f ( x _ { 1 } , \dots , x _ { k } ) , f ^ { \prime } = f ^ { \prime } ( x _ { 1 } , \dots , x _ { k } ) \in \mathbb { R } ^ { K }$ we have

$$
\mathbb { E } \left( \operatorname* { s u p } _ { \mathrm { V a r } _ { \infty } ( f ^ { \prime } - f ) \leq \tau ^ { 2 } } \left| \int \left( \Gamma _ { f } ( y ) - \Gamma _ { f ^ { \prime } } ( y ) \right) \mathrm { d } ( Q - Q _ { n } ) ( y ) \right| \right) \leq C _ { \underline { { \alpha } } , R } \tau \sqrt { \frac { K } { n } } ,
$$

for some constant $C _ { \underline { { \alpha } } , R } > 0$ . Moreover, for $p \geq 2$ and a constant $C _ { \underline { { \alpha } } , R , p } > 0$

$$
\mathbb { E } \left( \left[ \operatorname* { s u p } _ { \mathrm { V a r } _ { \infty } ( f ^ { \prime } - f ) \leq \tau ^ { 2 } } \left| \int \left( \Gamma _ { f } ( y ) - \Gamma _ { f ^ { \prime } } ( y ) \right) \mathrm { d } ( Q - Q _ { n } ) ( y ) \right| \right] ^ { p } \right) \leq C _ { \underline { { \alpha } } , R , p } \left( \tau \sqrt { \frac { K } { n } } \right) ^ { p } .\tag{47}
$$

Proof. This is simply a re-statement of Lemma F.1 in [18], whose proof is based on controlling the covering numbers of the family of functions $\Gamma _ { f }$ along with a generic bound on the empirical process (Lemma H.3 in [18]), and we only comment on two minor diferences: first, our $\Gamma _ { u }$ includes dependence in $\alpha _ { k }$ . As we assume that $\underline { { \alpha } } > 0$ throughout, nothing substantially changes. Second, our definition of $\Gamma _ { u } ( y )$ contains the term $\| x _ { k } - y \| ^ { 2 }$ instead of an inner product. This is immaterial because i) the quadratic terms $\| y | ^ { 2 }$ cancel out in the diference $\Gamma _ { f } ( y ) - \Gamma _ { f ^ { \prime } } ( y )$ , and the terms $\| x _ { k } \| ^ { 2 }$ will at worst induce a dependency of the constant in R. □

Lemma 7. Let P be a discrete measure supported on K atoms with weights α, and let $P _ { n }$ be the corresponding empirical measure with weights $\alpha ^ { n }$ . Then,

$$
\mathbb { E } \left( \chi ^ { 2 } ( P _ { n } \| P ) \right) = \frac { K - 1 } { n } .
$$

Consequently,

$$
\mathbb { E } \left( \| \alpha ^ { n } - \alpha \| _ { \infty } ^ { 2 } \right) \le \mathbb { E } \left( \chi ^ { 2 } ( P _ { n } \| P ) \right) \le \frac { K - 1 } { n } .
$$

Moreover, for each $q \geq 1$ 2

$$
\begin{array} { r } { \mathbb { E } \| \alpha ^ { n } - \alpha \| _ { \infty } ^ { 2 q } \lesssim n ^ { - q } , } \end{array}
$$

where the underlying constant only depends on K and $q .$ . Additionally, let $\Lambda _ { n }$ be the event $\alpha _ { k } ^ { n } \geq$ $\alpha _ { k } / 2 , \forall k \in [ K ]$ . Then,

$$
\mathbb { P } ( \Lambda _ { n } ^ { c } ) \le K \exp ( - \underline { { \alpha } } n c ) ,
$$

for some $c > 0$

Proof. We only show the statement for arbitrary q, the other ones are essentially Lemmas E.2 and H.2 in [18]. For each $k \in [ K ]$ , write

$$
\alpha _ { k } ^ { n } - \alpha _ { k } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \xi _ { i , k } , \qquad \xi _ { i , k } : = { \bf 1 } \{ X _ { i } = x _ { k } \} - \alpha _ { k } .
$$

The $\xi _ { i , k } \mathrm { ^ { \circ } s }$ are i.i.d., centered, and satisfy $| \xi _ { i , k } | \le 1$ . By Rosenthal’s inequality (Lemma 8), and using that all moments $\mathbb { E } | \xi _ { i , k } | ^ { m } \le 1$ , we obtain that for every fixed $q \geq 1$

$$
\mathbb { E } \left| \sum _ { i = 1 } ^ { n } \xi _ { i , k } \right| ^ { 2 q } \leq C _ { 2 q } \operatorname* { m a x } \left\{ \sum _ { i = 1 } ^ { n } \mathbb { E } | \xi _ { i , k } | ^ { 2 q } , \left( \sum _ { i = 1 } ^ { n } \mathbb { E } \xi _ { i , k } ^ { 2 } \right) ^ { q } \right\} \lesssim n ^ { q } .
$$

Dividing by $n ^ { 2 q }$ , we get $\mathbb { E } | \alpha _ { k } ^ { n } - \alpha _ { k } | ^ { 2 q } \lesssim n ^ { - q }$ and consequently,

$$
\mathbb { E } \Vert \alpha ^ { n } - \alpha \Vert _ { \infty } ^ { 2 q } = \mathbb { E } \left[ \operatorname* { m a x } _ { k \in [ K ] } | \alpha _ { k } ^ { n } - \alpha _ { k } | ^ { 2 q } \right] \leq \sum _ { k = 1 } ^ { K } \mathbb { E } | \alpha _ { k } ^ { n } - \alpha _ { k } | ^ { 2 q } \lesssim n ^ { - q } .
$$

Lemma 8 (Rosenthal’s inequality). [Theorem 3 in $\it { [ 2 1 ] }$ Let $X _ { i } \in \mathbb { R } ^ { d }$ be an i.i.d sequence of zero mean random variables with finite m-th moment, $m \geq 2$ . Then,

$$
\mathbb { E } { \Big | } \sum _ { i = 1 } ^ { n } X _ { i } { \Big | } ^ { m } \leq C _ { m } \operatorname* { m a x } { \Bigg \{ } \sum _ { i = 1 } ^ { n } \mathbb { E } | X _ { i } | ^ { m } , \left( \sum _ { i = 1 } ^ { n } \mathbb { E } | X _ { i } | ^ { 2 } \right) ^ { m / 2 } { \Bigg \} } ,
$$

where $C _ { m }$ is a constant that depends only on m.

Lemma 9 (Lemmas 2 and 4 in [15]). The following statements hold:

(a) For each $\begin{array} { r } { \alpha > 0 , \ i f t = \mathbb { E } _ { \mu } \left( \exp \left( \frac { \| Y \| ^ { 2 } } { \alpha } \right) \right) } \end{array}$ is finite, then $\mu$ is $t \frac { \alpha } { 2 d } - s u b G a u s s i a n$

(b) Consequently, $i f Q$ is $\varepsilon ^ { 2 }$ subGaussian then $Q _ { n }$ is subGaussian with parameter

$$
\varepsilon _ { n } ^ { 2 } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \exp \left( \frac { \| Y _ { i } \| ^ { 2 } } { 2 d \varepsilon ^ { 2 } } \right) .
$$

(c) $Q , Q _ { n }$ are uniformly subGaussian for some a.e. finite random subGaussianity proxy $\varepsilon _ { u } ^ { 2 }$

(d) $I f \widetilde \varepsilon ^ { 2 }$ is the smallest such subGaussianity proxy, then for each integer m

$$
\mathbb { E } ( \tilde { \varepsilon } ^ { 2 m } ) \leq 2 m ^ { m } \varepsilon ^ { 2 m } .
$$

Essentially, the above variance plays the role of the usual Euclidean norm, but accounts for the fact that potentials are only defined up to additive constants. Finally, we are able to pass from the convergence of potentials to barycentric projections by relying on the following stability bound

## C Proofs and auxiliary lemmata for Sinkhorn-EM

In this appendix we present the proofs for the convergence sample-based Sinkhorn-EM, as established in Theorem 4. We will denote $\mathcal { B } : = B ( \theta ^ { * } , \| \theta ^ { * } \| / 4 )$

We start with the proof of the main result, and then we will prove the intermediate results required for it.

## C.1 Proof of Theorem 4

Proof. The proof uses the standard argument from finite-sample analyses of EM [1, 7]; we include it for completeness. The only diference is that we use the uniform deviation bound from Proposition 5.

Step 1: Conditioning on the Statistical Concentration.

By Proposition 5, there exists an event holding with probability at least $1 - \delta - n ^ { - c _ { 1 } d } - c _ { 2 } n ^ { - 2 }$ upon which the empirical updates converge uniformly to the population updates over the entire basin B. On this event, the maximum statistical error is

$$
\eta _ { n } : = \operatorname* { s u p } _ { \theta \in \mathcal { B } } \| F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha ( \theta ) ) \| \le C \sqrt { \frac { d \log ( n ) + \log ( 1 / \delta ) } { n } } .\tag{48}
$$

We condition on this high-probability event for all subsequent steps.

## Step 2: Inductive Hypothesis

We propose that for any iteration $t \in \{ 0 , \ldots , T \}$ , the following two conditions hold:

1. Basin Stability: $\widehat { \theta } _ { S E M } ^ { t } \in \boldsymbol { B } .$

2. Error Bound: $\begin{array} { r } { \| \widehat { \theta } _ { S E M } ^ { t } - \theta ^ { * } \| \leq \kappa ^ { t } \| \widehat { \theta } _ { S E M } ^ { 0 } - \theta ^ { * } \| + \eta _ { n } \sum _ { i = 0 } ^ { t - 1 } \kappa ^ { i } . } \end{array}$

Base Case $( t = 0 ) \colon \mathrm { B y }$ the initialization assumption, $\lVert \widehat { { \boldsymbol { \theta } } } _ { S E M } ^ { 0 } - { \boldsymbol { \theta } } ^ { * } \rVert \leq \lVert { \boldsymbol { \theta } } ^ { * } \rVert / 4$ , so the iterate is in the basin. The summation in the error bound is empty, so the condition holds trivially.

## Step 3: Inductive Step

Assume the hypothesis holds for step t. We analyze the error at $t + 1$ by decomposing it into statistical error and population-level contraction using the triangle inequality:

$$
\begin{array} { r l } & { \| \widehat { \theta } _ { \mathrm { S E M } } ^ { t + 1 } - \theta ^ { * } \| = \| F _ { n } ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } , \alpha _ { n } ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } ) ) - \theta ^ { * } \| } \\ & { \qquad \leq \underbrace { \| F _ { n } ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } , \alpha _ { n } ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } ) ) - F ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } , \alpha ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } ) ) \| } _ { \mathrm { S t a t i s t i c a l ~ E r r o r } } + \underbrace { \| F ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } , \alpha ( \widehat { \theta } _ { \mathrm { S E M } } ^ { t } ) ) - \theta ^ { * } \| } _ { \mathrm { P o p u l a t i o n ~ C o n t r a c t i o n } } . } \end{array}\tag{49}
$$

We bound these terms individually:

1. Statistical Error: Because ${ \widehat { \theta } } ^ { t } \in B$ by the inductive hypothesis, this term is bounded by the uniform supremum $\eta _ { n }$ from Step 1.

2. Population Contraction: By Proposition 4, the population Sinkhorn-EM and EM operators coincide under our Gaussian mixture model. Furthermore, the population EM operator F is a uniform κ-contraction for all $\theta \in B$ under our assumptions as proved in [1]. Thus, $\| F ( \widehat { \theta ^ { t } } , \alpha ( \widehat { \theta ^ { t } } ) ) - \theta ^ { * } \| \leq \kappa \| \widehat { \theta ^ { t } } - \theta ^ { * } \|$

Substituting these back into (49) and applying the inductive hypothesis for step t:

$$
\begin{array} { l } { \displaystyle \| \widehat { \theta } _ { S E M } ^ { t + 1 } - { \theta } ^ { * } \| \leq \eta _ { n } + \kappa \left( \kappa ^ { t } \| \widehat { { \theta } } ^ { \flat } - { \theta } ^ { * } \| + \eta _ { n } \displaystyle \sum _ { i = 0 } ^ { t - 1 } \kappa ^ { i } \right) } \\ { = \kappa ^ { t + 1 } \| \widehat { { \theta } } ^ { \flat } - { \theta } ^ { * } \| + \eta _ { n } \displaystyle \sum _ { i = 0 } ^ { t } \kappa ^ { i } . } \end{array}\tag{50}
$$

This verifies the recursive error bound for step $t + 1$

To complete the induction, we must show $\widehat { \theta } _ { S E M } ^ { t + 1 }$ does not leave B. The maximum distance reached by the error bound is:

$$
\lVert \widehat { { \boldsymbol { \theta } } } _ { S E M } ^ { t + 1 } - { \boldsymbol { \theta } } ^ { * } \rVert \leq \kappa ^ { t + 1 } \lVert \widehat { { \boldsymbol { \theta } } } ^ { 0 } - { \boldsymbol { \theta } } ^ { * } \rVert + \frac { \eta _ { n } } { 1 - \kappa } .\tag{51}
$$

Since $\kappa < 1$ and $\lVert \widehat { { \boldsymbol { \theta } } } ^ { 0 } - { \boldsymbol { \theta } } ^ { * } \rVert \leq \lVert { \boldsymbol { \theta } } ^ { * } \rVert / 4$ , a suficient condition for staying in the basin is $\eta _ { n } + \kappa \| \theta ^ { * } \| / 4 \leq$ $\| \theta ^ { * } \| / 4$ , which simplifies to $\eta _ { n } \leq ( 1 - \kappa ) \| \theta ^ { * } \| / 4$ . Since $\eta _ { n }$ decrease as $n ^ { - 1 / 2 }$ , for n suficiently large,

the statistical noise $\eta _ { n }$ is strictly bounded such that this inequality holds. Thus, $\widehat { \theta } _ { S E M } ^ { t + 1 } \in \boldsymbol { B }$ , closing the induction.

The final result follows by substituting the sum of the geometric series and the definition of $\eta _ { n } . \ \boxed { \begin{array} { r l } \end{array} }$

The main ingredients to prove the theorem above are Proposition 5 (statistical error control) and Proposition 4 (population contraction). We start proving Proposition 5.

## C.2 Proof of Proposition 5

For this proof, we use many intermediate results which are deferred to the next subsection.

Proof of Proposition 5. By the triangle inequality, we decompose the error as follows

$$
\begin{array} { r l } { \| F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha ( \theta ) ) \| \le \| F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha _ { n } ( \theta ) ) \| } & { } \\ { + \| F ( \theta , \alpha _ { n } ( \theta ) ) - \mathbb { E } [ F ( \theta , \alpha _ { n } ( \theta ) ) ] \| } & { } \\ { + \| \mathbb { E } [ F ( \theta , \alpha _ { n } ( \theta ) ) ] - F ( \theta , \alpha ( \theta ) ) \| . } \end{array}
$$

Now, we bound each term separately.

First term. We start bounding the first term in the decomposition with Corollary 4. Lemma 10 gives a high probability bound for $\alpha _ { n } ( \theta )$ away from 0 and 1. Therefore, since $B \subset B ( 0 , 2 \| \theta ^ { * } \| )$ Corollary 4 applies to the data-dependent map $\widehat { \alpha } _ { n } = \alpha _ { n }$ on $\tau = B$ . It gives that, with probability at least $1 - n ^ { - c d } - C n ^ { - 2 }$

$$
\operatorname* { s u p } _ { \theta \in B } \| F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha _ { n } ( \theta ) ) \| \leq C \left( \| \theta ^ { * } \| + \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } + 1 \right) \sqrt { \frac { d \log n } { n } } .
$$

Absorbing the displayed factor into the constant $C ,$ this is with probability at least $1 - n ^ { - c d } - C n ^ { - 2 }$

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \left\| F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha _ { n } ( \theta ) ) \right\| \leq C \sqrt { \frac { d \log n } { n } } .
$$

Second term. The bound for second term of the decomposition is a consequence of Proposition $^ { 7 , }$ which states that with probability at least $1 - \delta - C n ^ { - 2 }$ ，

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \| F ( \theta , \alpha _ { n } ( \theta ) ) - \mathbb { E } [ F ( \theta , \alpha _ { n } ( \theta ) ) ] \| \leq C \operatorname* { m a x } \left\{ \sigma , \| \theta ^ { * } \| \right\} \exp \left( C \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } \right) \sqrt { \frac { d \log ( e n ) + \log ( 1 / \delta ) } { n } } .
$$

Third term. This term was already handled in Section 4.2. There, we showed that

$$
\left\| \mathbb { E } [ F ( \theta , \alpha _ { n } ( \theta ) ) ] - F ( \theta , \alpha ( \theta ) ) \right\| \leq \frac { 1 } { \| \theta \| } \left( \mathbb { E } \| Y \| ^ { 2 } \right) ^ { 1 / 2 } \left( \mathbb { E } \left\| \overleftarrow { T } _ { n } ^ { \theta } ( Y ) - \overleftarrow { T } ^ { \theta } ( Y ) \right\| ^ { 2 } \right) ^ { 1 / 2 } .
$$

Moreover, under the balanced Gaussian mixture,

$$
\mathbb { E } \| Y \| ^ { 2 } = d \sigma ^ { 2 } + \| \theta ^ { * } \| ^ { 2 } ,
$$

and

$$
\frac { 1 } { \| \theta \| } \lesssim \frac { 1 } { \| \theta ^ { * } \| }
$$

for all $\theta \in B$ . Thus, by the barycentric-map bound from Theorem $3 .$

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \left\| \mathbb { E } [ F ( \theta , \alpha _ { n } ( \theta ) ) ] - F ( \theta , \alpha ( \theta ) ) \right\| \lesssim \frac { ( d \sigma ^ { 2 } + \| \theta ^ { * } \| ^ { 2 } ) ^ { 1 / 2 } } { \| \theta ^ { * } \| } \sqrt { \frac { 1 } { n } } .
$$

After absorbing the factors depending on $\sigma , \lVert \theta ^ { * } \rVert$ into $C _ { i }$

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \left\| \mathbb { E } [ F ( \theta , \alpha _ { n } ( \theta ) ) ] - F ( \theta , \alpha ( \theta ) ) \right\| \leq C { \sqrt { \frac { d } { n } } } .
$$

Combining the three bounds and absorbing constants, we obtain that with probability at least $1 - n ^ { - c _ { 1 } d } - c _ { 2 } n ^ { - 2 } - \delta$ ，

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \| F _ { n } ( \theta , \alpha _ { n } ( \theta ) ) - F ( \theta , \alpha ( \theta ) ) \| \leq C \sqrt { \frac { d \log n + \log ( 1 / \delta ) } { n } } ,
$$

as claimed.

## C.3 Proof of Proposition 4

Proof. Since $\begin{array} { r } { \alpha = \frac { 1 } { 2 } } \end{array}$ , the population data distribution is

$$
Q ^ { * } = \frac { 1 } { 2 } \mathcal { N } ( \theta ^ { * } , \sigma ^ { 2 } I _ { d } ) + \frac { 1 } { 2 } \mathcal { N } ( - \theta ^ { * } , \sigma ^ { 2 } I _ { d } ) ,
$$

which is symmetric under $y \mapsto - y$

We first show that, for every $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ 2

$$
\alpha ( \theta ) = { \frac { 1 } { 2 } } .
$$

Recall that $\alpha ( \theta )$ is defined from an optimal semidual potential

$$
f ( \theta ) = ( f _ { 1 } ( \theta ) , f _ { 2 } ( \theta ) ) \in \mathbb { R } ^ { 2 }
$$

by

$$
\alpha ( \theta ) = \frac { \alpha e ^ { f _ { 1 } ( \theta ) / \sigma ^ { 2 } } } { \alpha e ^ { f _ { 1 } ( \theta ) / \sigma ^ { 2 } } + ( 1 - \alpha ) e ^ { f _ { 2 } ( \theta ) / \sigma ^ { 2 } } } .
$$

Since here $\begin{array} { r } { \alpha = \frac { 1 } { 2 } } \end{array}$ , it is enough to prove that

$$
f _ { 1 } ( \theta ) = f _ { 2 } ( \theta ) .
$$

Let

$$
P _ { \theta } = \frac { 1 } { 2 } \delta _ { \theta } + \frac { 1 } { 2 } \delta _ { - \theta } .
$$

The semidual objective for $S ( P _ { \theta } , Q ^ { * } )$ is

$$
\Phi _ { \theta } ( f _ { 1 } , f _ { 2 } ) = { \frac { 1 } { 2 } } f _ { 1 } + { \frac { 1 } { 2 } } f _ { 2 } - \sigma ^ { 2 } \int \log \left( { \frac { 1 } { 2 } } e ^ { ( f _ { 1 } - { \frac { 1 } { 2 } } \| y - \theta \| ^ { 2 } ) / \sigma ^ { 2 } } + { \frac { 1 } { 2 } } e ^ { ( f _ { 2 } - { \frac { 1 } { 2 } } \| y + \theta \| ^ { 2 } ) / \sigma ^ { 2 } } \right) \mathrm { d } Q ^ { * } ( y ) .
$$

Using the symmetry of $Q ^ { * }$ and the identities

$$
\| - y - \theta \| ^ { 2 } = \| y + \theta \| ^ { 2 } , \qquad \| - y + \theta \| ^ { 2 } = \| y - \theta \| ^ { 2 } ,
$$

a change of variables $y \mapsto - y$ yields

$$
\Phi _ { \theta } \bigl ( f _ { 1 } , f _ { 2 } \bigr ) = \Phi _ { \theta } \bigl ( f _ { 2 } , f _ { 1 } \bigr ) \qquad \mathrm { f o r ~ a l l ~ } \bigl ( f _ { 1 } , f _ { 2 } \bigr ) \in \mathbb { R } ^ { 2 } .
$$

Now let

$$
f ( \theta ) = ( f _ { 1 } ( \theta ) , f _ { 2 } ( \theta ) )
$$

be the optimal semidual potential satisfying the gauge condition (4),

$$
\mathbb { E } _ { X \sim P _ { \theta } } [ f ( X ) ] = 0 .
$$

Since

$$
P _ { \theta } = \frac { 1 } { 2 } \delta _ { \theta } + \frac { 1 } { 2 } \delta _ { - \theta } ,
$$

this condition becomes

$$
\frac { 1 } { 2 } f _ { 1 } ( \theta ) + \frac { 1 } { 2 } f _ { 2 } ( \theta ) = 0 .
$$

Thus the gauge condition is invariant under swapping the two coordinates.

Because

$$
\Phi _ { \theta } ( f _ { 1 } , f _ { 2 } ) = \Phi _ { \theta } ( f _ { 2 } , f _ { 1 } ) ,
$$

if $( f _ { 1 } ( \theta ) , f _ { 2 } ( \theta ) )$ is an optimal potential satisfying the gauge condition, then $( f _ { 2 } ( \theta ) , f _ { 1 } ( \theta ) )$ is also an optimal potential satisfying the same gauge condition. Since optimal semidual potentials are unique up to additive constants, and the gauge condition fixes this constant uniquely, we must have

$$
( f _ { 1 } ( \theta ) , f _ { 2 } ( \theta ) ) = ( f _ { 2 } ( \theta ) , f _ { 1 } ( \theta ) ) .
$$

Hence

$$
f _ { 1 } ( \theta ) = f _ { 2 } ( \theta ) .
$$

In fact, together with the gauge condition, this also gives

$$
f _ { 1 } ( \theta ) = f _ { 2 } ( \theta ) = 0 .
$$

Therefore,

$$
\alpha ( \theta ) = \frac { 1 } { 2 } \qquad \mathrm { f o r ~ e v e r y ~ } \theta \in  { \mathbb { R } } ^ { d } .
$$

It follows that the population Sinkhorn-EM update reduces to

$$
\theta _ { \mathrm { S E M } } ^ { t + 1 } = F \left( \theta _ { \mathrm { S E M } } ^ { t } , \alpha ( \theta _ { \mathrm { S E M } } ^ { t } ) \right) = F \left( \theta _ { \mathrm { S E M } } ^ { t } , \frac { 1 } { 2 } \right) .
$$

But $F ( \theta , { \frac { 1 } { 2 } } )$ is exactly the population EM update map for the balanced symmetric two-Gaussian model [29, 1]. Since the two algorithms have the same initialization and update map, induction yields

$$
\theta _ { \mathrm { S E M } } ^ { t } = \theta _ { \mathrm { E M } } ^ { t }
$$

for every $t \geq 0$

## C.4 Other technical lemmata for Proposition 5

The bound of the first term in the decomposition in the proof of Proposition 5 relies on the a Corollary from a result in [25] along with a bound on $\alpha _ { n } ( \theta )$ , which we prove below.

Corollary 4 (Mean-update concentration with a data-dependent weight). Assume $\| \theta ^ { * } \| / \sigma \leq C _ { \theta }$ and let $\mathcal { T } \subseteq B ( 0 , 2 \| \theta ^ { * } \| )$ be deterministic. Let ${ \widehat { \alpha } } _ { n } : { \mathcal { T } }  [ 0 , 1 ]$ be any possibly data-dependent map such that, for some α<sub>0</sub> $\in \ ( 0 , 1 / 2 )$ , with probability at least $1 - \beta , \widehat { \alpha } _ { n } ( \theta ) \in [ \alpha _ { 0 } , 1 - \alpha _ { 0 } ]$ for every $\theta \in \mathcal T$ . Set $C _ { \rho } : = 1 - 2 \alpha _ { 0 }$ . There exist constants $C , c > 0$ , depending only on $C _ { \theta }$ and $C _ { \rho ; }$ , such that for $n \geq C d \log n$ and $n \geq 2 ,$ , with probability at least $1 - n ^ { - c d } - \beta ,$

$$
\operatorname* { s u p } _ { \theta \in \mathcal { T } } \| F _ { n } ( \theta , \widehat { \alpha } _ { n } ( \theta ) ) - F ( \theta , \widehat { \alpha } _ { n } ( \theta ) ) \| \leq C \left( \| \theta ^ { * } \| + \frac { \sigma } { 2 } \log \frac { 1 - \alpha _ { 0 } } { \alpha _ { 0 } } \right) \sqrt { \frac { d \log n } { n } } .
$$

Proof of Corollary 4. Apply Theorem 3 of [25] after rescaling Y and θ by $\sigma .$ . Its uniform meanupdate bound, together with $\begin{array} { r } { | 2 \alpha - 1 | \leq \frac { 1 } { 2 } | \log ( \alpha / ( 1 - \alpha ) ) | } \end{array}$ , gives, with probability at least $1 - n ^ { - c d }$ ，

$$
\left\| F _ { n } ( \theta , \alpha ) - F ( \theta , \alpha ) \right\| \leq C \left( \left\| \theta \right\| + { \frac { \sigma } { 2 } } \left| \log { \frac { \alpha } { 1 - \alpha } } \right| \right) { \sqrt { \frac { d \log n } { n } } }
$$

simultaneously for $\theta \in B ( 0 , 2 \| \theta ^ { * } \| )$ and $\alpha \in [ \alpha _ { 0 } , 1 - \alpha _ { 0 } ]$ . Intersect this event with the assumed weight-control event and evaluate the uniform bound at $\alpha = { \widehat { \alpha } } _ { n } ( \theta )$ for $\theta \in \mathcal T$ . Since $\lVert \theta \rVert \leq 2 \lVert \theta ^ { * } \rVert$

and $| \log ( \alpha / ( 1 - \alpha ) ) | \le \log ( ( 1 - \alpha _ { 0 } ) / \alpha _ { 0 } )$ on this interval, the claim follows by a union bound and an adjustment of C. □

Lemma 10. Let $Q ^ { * }$ be as in (12) with $\alpha = 1 / 2$ , and assume $\lVert \theta ^ { * } \rVert \leq C _ { \theta }$ . For each $\theta \in B$ , let $f _ { n } ( \theta ) \in \mathbb { R } ^ { 2 }$ be an optimal semidual potential for $S ( P _ { \theta } , Q _ { n } )$ , where $\begin{array} { r } { P _ { \theta } = \frac { 1 } { 2 } \delta _ { \theta } + \frac { 1 } { 2 } \delta _ { - \theta } } \end{array}$ , and define

$$
\alpha _ { n } ( \theta ) = \frac { \frac { 1 } { 2 } \exp ( f _ { n , 1 } ( \theta ) / \sigma ^ { 2 } ) } { \frac { 1 } { 2 } \exp ( f _ { n , 1 } ( \theta ) / \sigma ^ { 2 } ) + \frac { 1 } { 2 } \exp ( f _ { n , 2 } ( \theta ) / \sigma ^ { 2 } ) } .
$$

Then, with probability at least $1 - C n ^ { - 2 }$ , for every $\theta \in B ,$

$$
\alpha _ { n } ( \theta ) \in [ \alpha _ { 0 } , 1 - \alpha _ { 0 } ] ,
$$

where

$$
\alpha _ { 0 } : = \frac { 1 } { 1 + \exp \left( 2 C _ { 0 } \left[ \| \theta ^ { * } \| ^ { 2 } / \sigma ^ { 2 } + 1 \right] \right) } ,
$$

and $C _ { 0 } > 0$ is a constant depending only on the fixed problem parameters.

Proof of Lemma 10. Let $\widetilde { \varepsilon }$ be the smallest constant such that $Q _ { n }$ and $Q ^ { * }$ are uniformly subGaussian. By Lemma 9, this is finite almost surely. Applying Proposition 3 to the problem $( P _ { \theta } , Q _ { n } )$ , and using that $P _ { \theta }$ has two atoms with weights $1 / 2$ , gives

$$
\| f _ { n } ( \theta ) \| _ { \infty } \lesssim \| \theta \| ^ { 2 } + \sigma ^ { 2 } \log 2 + \frac { \widetilde { \varepsilon } ^ { 2 } } { \sigma ^ { 2 } } \| \theta \| ^ { 2 } .
$$

Define the event

$$
E _ { n } : = \{ \tilde { \varepsilon } ^ { 2 } < 1 2 \varepsilon ^ { 2 } \} .
$$

By Lemma 5 with $m = 4$

$$
\mathbb { P } ( E _ { n } ^ { c } ) \lesssim n ^ { - 2 } .
$$

On $E _ { n }$ , uniformly over $\theta \in B$

$$
\| f _ { n } ( \theta ) \| _ { \infty } \leq C _ { 0 } \big ( \| \theta ^ { * } \| ^ { 2 } + \sigma ^ { 2 } \big ) = : L ,
$$

for some constant $C _ { 0 } > 0$ depending only on the fixed problem parameters, where we used

$$
\lVert { \boldsymbol { \theta } } \rVert \leq \lVert { \boldsymbol { \theta } } ^ { * } \rVert + \lVert { \boldsymbol { \theta } } - { \boldsymbol { \theta } } ^ { * } \rVert \leq { \frac { 5 } { 4 } } \lVert { \boldsymbol { \theta } } ^ { * } \rVert .
$$

Since $\alpha = 1 / 2$ , the Sinkhorn weight satisfies

$$
\frac { \alpha _ { n } ( \theta ) } { 1 - \alpha _ { n } ( \theta ) } = \exp \left( \frac { f _ { n , 1 } ( \theta ) - f _ { n , 2 } ( \theta ) } { \sigma ^ { 2 } } \right) .
$$

Therefore, on $E _ { n }$ ，

$$
e ^ { - 2 L / \sigma ^ { 2 } } \leq { \frac { \alpha _ { n } ( \theta ) } { 1 - \alpha _ { n } ( \theta ) } } \leq e ^ { 2 L / \sigma ^ { 2 } } \qquad { \mathrm { f o r ~ e v e r y ~ } } \theta \in { \mathcal { B } } .
$$

It follows that

$$
\alpha _ { n } ( \theta ) \in [ \alpha _ { 0 } , 1 - \alpha _ { 0 } ] \qquad \mathrm { f o r ~ e v e r y ~ } \theta \in \mathcal { B } ,
$$

where

$$
\alpha _ { 0 } : = \frac { 1 } { 1 + \exp ( 2 L / \sigma ^ { 2 } ) } = \frac { 1 } { 1 + \exp \left( 2 C _ { 0 } \left[ \| \theta ^ { * } \| ^ { 2 } / \sigma ^ { 2 } + 1 \right] \right) } .
$$

The bound of the second term in the decomposition in the proof of Proposition 5 is given below.

Proposition 7 (Uniform concentration of the Sinkhorn-corrected population update). Assume (12) holds with $\alpha = 1 / 2$ and $\theta ^ { * } \neq 0$ . For each $\theta \in B$ , let $\alpha _ { n } ( \theta )$ be the empirical Sinkhorn-corrected weight, and let $\alpha _ { 0 }$ be as in Lemma 10. There exists a constant $C > 0$ , depending only on α<sub>0</sub>, such that for every $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta - C n ^ { - 2 }$ ，

$$
\operatorname* { s u p } _ { \theta \in B } \left\| F ( \theta , \alpha _ { n } ( \theta ) ) - \mathbb { E } _ { Y _ { 1 } , \ldots , Y _ { n } } [ F ( \theta , \alpha _ { n } ( \theta ) ) ] \right\| \le C \operatorname* { m a x } \left\{ \sigma , \| \theta ^ { * } \| \right\} \exp \left( C \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } \right) \sqrt { \frac { d \log ( e n ) + \log ( 1 / \delta ) } { n } } .
$$

Proof. The proof begins with a ghost-sample symmetrization. To control the deviation of $F ( \theta , \alpha _ { n } ( \theta ) )$ from its expectation, we introduce a weight $\alpha _ { n } ^ { \prime } ( \theta )$ estimated from an independent sample. Averaging $F ( \theta , \alpha _ { n } ^ { \prime } ( \theta ) )$ over that sample recovers the expectation we want to subtract, so Jensen’s inequality reduces the problem to comparing $F ( \theta , \alpha _ { n } ( \theta ) )$ and $F ( \theta , \alpha _ { n } ^ { \prime } ( \theta ) )$ . Both weights solve empirical versions of the same population calibration equation. We show that this equation is stable: small sampling errors lead to small changes in its solution, following the basic principle of Z-estimation [23, Chapter 5, Sections 5.2–5.3]. We then show that these small changes in the weight produce small changes in $F ,$ giving the desired bound after averaging over the independent sample. All bounds below are uniform over $\theta \in B$

## Step 1: The Sinkhorn calibration equation. Recall that

$$
\Psi ( y , \theta , \alpha ) : = \frac { \alpha \exp \left( - \frac { \| y - \theta \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } { \alpha \exp \left( - \frac { \| y - \theta \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) + ( 1 - \alpha ) \exp \left( - \frac { \| y + \theta \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) }
$$

Here $\alpha$ is a variable responsibility weight; the true mixture weight remains fixed at $1 / 2$ . Define

$$
G ( \theta , \alpha ) : = \mathbb { E } _ { Y \sim Q ^ { * } } \Psi ( Y , \theta , \alpha ) , \qquad G _ { n } ( \theta , \alpha ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \Psi ( Y _ { i } , \theta , \alpha ) .
$$

$$
Z _ { n } : = \operatorname* { s u p } _ { \theta \in { \mathcal { B } } } \operatorname* { s u p } _ { \alpha \in I } | G _ { n } ( \theta , \alpha ) - G ( \theta , \alpha ) | ,
$$

where $I : = [ \alpha _ { 0 } , 1 - \alpha _ { 0 } ]$ . Let $g _ { n } ^ { \theta }$ be the continuous-side potential paired with $f _ { n } ( \theta )$ . The first marginal equation (5a), applied to $( P _ { \theta } , Q _ { n } )$ at $x = \theta$ , reads

$$
1 = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \exp \left( \frac { f _ { n , 1 } ( \theta ) + g _ { n } ^ { \theta } ( Y _ { i } ) - c ( \theta , Y _ { i } ) } { \sigma ^ { 2 } } \right) .
$$

Substituting the empirical version of (5b) into this identity and using the definition of $\alpha _ { n } ( \theta )$ gives

$$
G _ { n } ( \theta , \alpha _ { n } ( \theta ) ) = \frac { 1 } { 2 } .\tag{52}
$$

Indeed, each exponential in the preceding sum equals $2 \Psi ( Y _ { i } , \theta , \alpha _ { n } ( \theta ) )$ because the discrete marginal has weights $1 / 2 , 1 / 2$ . The identical calculation applies to an independent sample. Thus $\alpha _ { n } ( \theta )$ is an exact root of $G _ { n } ( \theta , \alpha ) - 1 / 2 = 0$

Step 2: Uniform control of the calibration map. Let ${ \mathcal { G } } : = \{ y \mapsto \Psi ( y , \theta , \alpha ) : \theta \in B , \ \alpha \in I \}$ Since

$$
\Psi ( y , \theta , \alpha ) = \ell \left( \log { \frac { \alpha } { 1 - \alpha } } + { \frac { 2 \langle y , \theta \rangle } { \sigma ^ { 2 } } } \right) , \qquad \ell ( t ) : = { \frac { 1 } { 1 + e ^ { - t } } } ,
$$

the inner class is an afine class in y with parameters $( \theta , \alpha )$ of dimension $O ( d )$ . More explicitly, for $u \in ( 0 , 1 )$ , the subgraph condition $u \leq \Psi ( y , \theta , \alpha )$ is equivalent to

$$
\log { \frac { u } { 1 - u } } \leq \log { \frac { \alpha } { 1 - \alpha } } + { \frac { 2 \langle y , \theta \rangle } { \sigma ^ { 2 } } } ,
$$

which is a halfspace condition after the fixed transformation

$$
( y , u ) \longmapsto \left( y , \log { \frac { u } { 1 - u } } \right) .
$$

Hence $\mathcal { G }$ is a bounded VC-subgraph class with VC-subgraph dimension $O ( d )$ ; see, e.g., [26, Lemmata 2.6.15 and 2.6.18]. The VC entropy bound [26, Theorem 2.6.7], together with symmetrization and the entropy-integral bound for empirical processes, yields

$$
\mathbb { E } Z _ { n } \leq C \sqrt { \frac { d \log ( e n ) } { n } } .
$$

Moreover, replacing one observation changes $Z _ { n }$ by at most $1 / n ,$ since every function in $\mathcal { G }$ takes values in [0, 1]. Therefore, McDiarmid’s inequality implies that, with probability at least $1 - \delta$ ，

$$
Z _ { n } \leq \mathbb { E } Z _ { n } + \sqrt { \frac { \log ( 1 / \delta ) } { 2 n } } \leq C \sqrt { \frac { d \log ( e n ) + \log ( 1 / \delta ) } { n } } .
$$

Step 3: Population slope and update bounds. Define

$$
\lambda _ { \mathcal { B } } : = \operatorname* { i n f } _ { \theta \in \mathcal { B } } \operatorname* { i n f } _ { \alpha \in I } \partial _ { \alpha } G ( \theta , \alpha ) , \qquad L _ { \mathcal { B } } : = \operatorname* { s u p } _ { \theta \in \mathcal { B } } \operatorname* { s u p } _ { \alpha \in I } \| \partial _ { \alpha } F ( \theta , \alpha ) \| ,
$$

and

$$
M _ { B } : = \operatorname* { s u p } _ { \theta \in B } \operatorname* { s u p } _ { \alpha , \alpha ^ { \prime } \in ( 0 , 1 ) } \| F ( \theta , \alpha ) - F ( \theta , \alpha ^ { \prime } ) \| .
$$

We show that $\lambda _ { B } > 0$ and bound $L _ { B } / \lambda _ { B }$ and $M _ { B }$ independently of d. A direct calculation gives

$$
\partial _ { \alpha } \Psi ( y , \theta , \alpha ) = \frac { \Psi ( y , \theta , \alpha ) \big ( 1 - \Psi ( y , \theta , \alpha ) \big ) } { \alpha ( 1 - \alpha ) } ,
$$

and hence

$$
\partial _ { \alpha } G ( \theta , \alpha ) = \mathbb { E } \left[ \frac { \Psi ( Y , \theta , \alpha ) \big ( 1 - \Psi ( Y , \theta , \alpha ) \big ) } { \alpha ( 1 - \alpha ) } \right] .
$$

For

$$
s ( y , \theta , \alpha ) : = \log \frac { \alpha } { 1 - \alpha } + \frac { 2 \langle y , \theta \rangle } { \sigma ^ { 2 } } ,
$$

we have

$$
\Psi ( y , \theta , \alpha ) \big ( 1 - \Psi ( y , \theta , \alpha ) \big ) = \frac { 1 } { 2 + e ^ { s ( y , \theta , \alpha ) } + e ^ { - s ( y , \theta , \alpha ) } } .
$$

For $\alpha \in I ,$

$$
\left| \log { \frac { \alpha } { 1 - \alpha } } \right| \leq \log { \frac { 1 - \alpha _ { 0 } } { \alpha _ { 0 } } } .
$$

Consequently, on the event $| \langle Y , \theta \rangle | \leq 2 \| \theta ^ { * } \| ^ { 2 }$

$$
\begin{array} { r l } { \Psi ( Y , \theta , \alpha ) \big ( 1 - \Psi ( Y , \theta , \alpha ) \big ) = \displaystyle \frac { 1 } { 2 + \exp \big ( s ( Y , \theta , \alpha ) \big ) + \exp \big ( - s ( Y , \theta , \alpha ) \big ) } } & { } \\ { \geq \displaystyle \frac { 1 } { 4 } \exp \big ( - | s ( Y , \theta , \alpha ) | \big ) } & { } \\ { \geq \displaystyle \frac { 1 } { 4 } \exp \bigg ( - \bigg | \log \frac { \alpha } { 1 - \alpha } \bigg | - \frac { 2 | \langle Y , \theta \rangle | } { \sigma ^ { 2 } } \bigg ) } & { } \\ { \geq \displaystyle \frac { 1 } { 4 } \exp \bigg ( - \log \frac { 1 - \alpha _ { 0 } } { \alpha _ { 0 } } - \lambda \frac { | \theta \rangle | ^ { 2 } } { \sigma ^ { 2 } } \bigg ) } & { } \\ { = \displaystyle \frac { \alpha _ { 0 } } { 4 ( 1 - \alpha _ { 0 } ) } \exp \bigg ( - 4 \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } \bigg ) } & { } \\ { \geq \exp \bigg ( - C \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } \bigg ) , } & { } \end{array}
$$

where $c , C > 0$ depend only on $\alpha _ { 0 }$ . It follows that

$$
\partial _ { \alpha } G ( \theta , \alpha ) \geq c \exp \left( - C \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } \right) \operatorname* { P r } \left( | \langle Y , \theta \rangle | \leq 2 \| \theta ^ { * } \| ^ { 2 } \right) .
$$

Write $Y = S \theta ^ { * } + \sigma Z$ , where $S$ is Rademacher and $Z \sim N ( 0 , I _ { d } )$ is independent of $S .$ For fixed $\theta ,$

$$
\langle Y , \theta \rangle \sim \frac { 1 } { 2 } N \left( \langle \theta ^ { * } , \theta \rangle , \sigma ^ { 2 } \| \theta \| ^ { 2 } \right) + \frac { 1 } { 2 } N \left( - \langle \theta ^ { * } , \theta \rangle , \sigma ^ { 2 } \| \theta \| ^ { 2 } \right) .
$$

By symmetry, if $X _ { \theta } \sim N \left( \langle \theta ^ { * } , \theta \rangle , \sigma ^ { 2 } \| \theta \| ^ { 2 } \right)$ , then

$$
\operatorname* { P r } \left( | \langle Y , \theta \rangle | \leq 2 \| \theta ^ { * } \| ^ { 2 } \right) = \operatorname* { P r } \left( | X _ { \theta } | \leq 2 \| \theta ^ { * } \| ^ { 2 } \right) .
$$

Moreover,

$$
\operatorname* { P r } \left( | X _ { \theta } | \leq 2 \| \theta ^ { * } \| ^ { 2 } \right) \geq \operatorname* { P r } \left( 0 \leq X _ { \theta } \leq 2 \| \theta ^ { * } \| ^ { 2 } \right) .
$$

Since $\theta \in B ,$

$$
\| \theta \| \leq \frac { 5 } { 4 } \| \theta ^ { * } \| , \qquad \frac { 3 } { 4 } \| \theta ^ { * } \| ^ { 2 } \leq \langle \theta ^ { * } , \theta \rangle \leq \frac { 5 } { 4 } \| \theta ^ { * } \| ^ { 2 } .
$$

Therefore,

$$
\operatorname* { P r } \left( | \langle Y , \theta \rangle | \leq 2 \| \theta ^ { * } \| ^ { 2 } \right) \geq c \operatorname* { m i n } \left\{ \frac { \| \theta ^ { * } \| } { \sigma } , 1 \right\} ,
$$

which follows from the following inequality

$$
\Phi _ { \mathrm { s t d } } ( x ) - \Phi _ { \mathrm { s t d } } ( - x ) \geq c \operatorname* { m i n } \{ x , 1 \} , \qquad x \geq 0 ,
$$

where $\Phi _ { \mathrm { s t d } } ( \cdot )$ denotes the standard normal cumulative distribution function. Thus, we obtain

$$
\lambda _ { B } \geq c \operatorname* { m i n } \left\{ \frac { \| \theta ^ { * } \| } { \sigma } , 1 \right\} \exp \left( - C \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } \right) .
$$

Next, we upper bound $L _ { B }$ . For $\theta \in B$ and $\alpha \in I$ , let

$$
h _ { \alpha } ( t ) : = \partial _ { \alpha } \ell \left( \log \frac { \alpha } { 1 - \alpha } + \frac { 2 t } { \sigma ^ { 2 } } \right) .
$$

Because $\alpha$ is bounded away from zero and one, and the first two derivatives of the logistic function are uniformly bounded,

$$
\| h _ { \alpha } \| _ { \infty } \leq C , \qquad \| h _ { \alpha } ^ { \prime } \| _ { \infty } \leq \frac { C } { \sigma ^ { 2 } } .
$$

Consequently, dominated convergence implies

$$
\partial _ { \alpha } F ( \theta , \alpha ) = 2  { \mathbb { E } } \left[ Y h _ { \alpha } (  { \langle Y , \theta \rangle } ) \right] .
$$

Using $Y = S \theta ^ { * } + \sigma Z$ , the signal contribution satisfies

$$
\| \mathbb { E } \left[ S \theta ^ { * } h _ { \alpha } \left( S \langle \theta ^ { * } , \theta \rangle + \sigma \langle Z , \theta \rangle \right) \right] \| \leq C \| \theta ^ { * } \| .
$$

For the Gaussian contribution, conditioning on $S$ and applying Stein’s lemma gives

$$
\begin{array} { r } { \mathbb { E } _ { Z } \left[ Z h _ { \alpha } \left( S \langle \theta ^ { * } , \theta \rangle + \sigma \langle Z , \theta \rangle \right) \right] = \sigma \theta \mathbb { E } _ { Z } \left[ h _ { \alpha } ^ { \prime } \left( S \langle \theta ^ { * } , \theta \rangle + \sigma \langle Z , \theta \rangle \right) \right] . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { \| \sigma \mathbb { E } \left[ Z h _ { \alpha } \left( S \langle \theta ^ { * } , \theta \rangle + \sigma \langle Z , \theta \rangle \right) \right] \| \leq C \| \theta \| . } \end{array}
$$

Since $\theta \in B$ implies $\lVert { \boldsymbol { \theta } } \rVert \leq 5 \lVert { \boldsymbol { \theta } } ^ { * } \rVert / 4$ , we conclude that

$$
L _ { B } \leq C \| \theta ^ { * } \| .
$$

Combining this bound with

$$
\lambda _ { B } \geq c \operatorname* { m i n } \left\{ \frac { \| \theta ^ { * } \| } { \sigma } , 1 \right\} \exp \left( - C \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } \right) ,
$$

we obtain

$$
\frac { L _ { B } } { \lambda _ { B } } \leq C \operatorname* { m a x } \left. \sigma , \| \theta ^ { * } \| \right. \exp \left( C \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } \right) .
$$

Finally, write

$$
F ( \theta , \alpha ) = \mathbb { E } \left[ Y q _ { \alpha } ( \langle Y , \theta \rangle ) \right] ,
$$

where

$$
q _ { \alpha } ( t ) : = 2 \ell \left( \log { \frac { \alpha } { 1 - \alpha } } + { \frac { 2 t } { \sigma ^ { 2 } } } \right) - 1 .
$$

Uniformly over $\alpha \in ( 0 , 1 )$ ,

$$
\| q _ { \alpha } \| _ { \infty } \leq 1 , \qquad \| q _ { \alpha } ^ { \prime } \| _ { \infty } \leq { \frac { C } { \sigma ^ { 2 } } } .
$$

The signal component of $F ( \theta , \alpha )$ is therefore bounded by $\lVert \theta ^ { * } \rVert$ , while Stein’s lemma bounds the Gaussian component by $C \lVert \theta \rVert$ . Since $\theta \in B$ , this gives

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \operatorname* { s u p } _ { \alpha \in ( 0 , 1 ) } \| F ( \theta , \alpha ) \| \leq C \| \theta ^ { * } \| ,
$$

which implies that

$$
M _ { B } \leq 2 C \| \theta ^ { * } \| .
$$

Step 4: Stability of the empirical roots. Introduce a ghost sample $Y _ { 1 } ^ { \prime } , \ldots , Y _ { n } ^ { \prime } \stackrel { \mathrm { i . i . d . } } { \sim } Q ^ { * }$ , independent of $Y _ { 1 } , \dots , Y _ { n }$ . Let $G _ { n } ^ { \prime } , \alpha _ { n } ^ { \prime }$ , and $Z _ { n } ^ { \prime }$ denote the corresponding empirical calibration map, Sinkhorn-corrected weight, and empirical-process deviation. Let $\mathbb { E } ^ { \prime }$ denote expectation only with respect to the ghost sample. Define

$$
H _ { n } ( \theta ) : = F ( \theta , \alpha _ { n } ( \theta ) ) , \qquad H _ { n } ^ { \prime } ( \theta ) : = F ( \theta , \alpha _ { n } ^ { \prime } ( \theta ) ) .
$$

Let $\mathcal { E } _ { \alpha }$ and $\mathcal { E } _ { \alpha } ^ { \prime }$ be the events from Lemma 10 for the original and ghost samples, respectively. On $\mathcal { E } _ { \alpha } \cap \mathcal { E } _ { \alpha } ^ { \prime } ,$ , both $\alpha _ { n } ( \theta )$ and $\alpha _ { n } ^ { \prime } ( \theta )$ belong to I for every $\theta \in B .$ . By the mean value theorem,

$$
\lambda _ { B } | \alpha _ { n } ( \theta ) - \alpha _ { n } ^ { \prime } ( \theta ) | \leq | G ( \theta , \alpha _ { n } ( \theta ) ) - G ( \theta , \alpha _ { n } ^ { \prime } ( \theta ) ) | ,
$$

by the definition of $\lambda _ { B }$ . Using (52) for the two samples, we obtain

$$
\begin{array} { r l } & { | G ( \theta , \alpha _ { n } ( \theta ) ) - G ( \theta , \alpha _ { n } ^ { \prime } ( \theta ) ) | \leq | G ( \theta , \alpha _ { n } ( \theta ) ) - G _ { n } ( \theta , \alpha _ { n } ( \theta ) ) | + | G _ { n } ^ { \prime } ( \theta , \alpha _ { n } ^ { \prime } ( \theta ) ) - G ( \theta , \alpha _ { n } ^ { \prime } ( \theta ) ) | } \\ & { \qquad \leq Z _ { n } + Z _ { n } ^ { \prime } . } \end{array}
$$

Consequently,

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \vert \alpha _ { n } ( \theta ) - \alpha _ { n } ^ { \prime } ( \theta ) \vert \leq \frac { Z _ { n } + Z _ { n } ^ { \prime } } { \lambda _ { \mathcal { B } } } ,
$$

and therefore

$$
\operatorname* { s u p } _ { \theta \in \mathcal { B } } \Vert H _ { n } ( \theta ) - H _ { n } ^ { \prime } ( \theta ) \Vert \leq \frac { L _ { \mathcal { B } } } { \lambda _ { \mathcal { B } } } \left( Z _ { n } + Z _ { n } ^ { \prime } \right)
$$

on $\mathcal { E } _ { \alpha } \cap \mathcal { E } _ { \alpha } ^ { \prime }$

Step 5: Ghost-sample centering. Since $H _ { n } ^ { \prime }$ has the same distribution as $H _ { n } , \ \mathbb { E } ^ { \prime } H _ { n } ^ { \prime } ( \theta ) \ =$ $\mathbb { E } H _ { n } ( \theta )$ . Furthermore, for each fixed realization of the original sample, $H _ { n } ( \theta )$ is constant with respect to $\mathbb { E } ^ { \prime } .$ . Hence $H _ { n } ( \theta ) - \mathbb { E } H _ { n } ( \theta ) = \mathbb { E } ^ { \prime } [ H _ { n } ( \theta ) - H _ { n } ^ { \prime } ( \theta ) ]$ , and Jensen’s inequality gives

$$
\begin{array} { r l r } & { } & { \underset { \theta \in \mathcal { B } } { \operatorname* { s u p } } \Vert H _ { n } ( \theta ) - \mathbb { E } H _ { n } ( \theta ) \Vert = \underset { \theta \in \mathcal { B } } { \operatorname* { s u p } } \left. \mathbb { E } ^ { \prime } \left[ H _ { n } ( \theta ) - H _ { n } ^ { \prime } ( \theta ) \right] \right. } \\ & { } & { \leq \mathbb { E } ^ { \prime } \underset { \theta \in \mathcal { B } } { \operatorname* { s u p } } \Vert H _ { n } ( \theta ) - H _ { n } ^ { \prime } ( \theta ) \Vert . } \end{array}
$$

Now work on the event $\mathcal { E } _ { \alpha }$ for the original sample. Splitting according to the ghost event $\mathcal { E } _ { \alpha } ^ { \prime }$ yields

$$
\begin{array} { r l } & { \mathbb { E } _ { \theta \in \mathcal { B } } ^ { \prime } \left. H _ { n } ( \theta ) - H _ { n } ^ { \prime } ( \theta ) \right. \leq \displaystyle \frac { L _ { \mathcal { B } } } { \lambda _ { \mathcal { B } } } \mathbb { E } ^ { \prime } \left[ \left( Z _ { n } + Z _ { n } ^ { \prime } \right) \mathbf { 1 } _ { \mathcal { E } _ { \alpha } ^ { \prime } } \right] + M _ { \mathcal { B } } \operatorname* { P r } ^ { \prime } ( ( \mathcal { E } _ { \alpha } ^ { \prime } ) ^ { c } ) } \\ & { \qquad \leq \displaystyle \frac { L _ { \mathcal { B } } } { \lambda _ { \mathcal { B } } } \left( Z _ { n } + \mathbb { E } Z _ { n } \right) + M _ { \mathcal { B } } \operatorname* { P r } ( \mathcal { E } _ { \alpha } ^ { c } ) , } \end{array}
$$

By Lemma 10,

$$
\operatorname* { P r } ( { \mathcal { E } } _ { \alpha } ^ { c } ) \leq C n ^ { - 2 } .
$$

It follows that, on $\mathcal { E } _ { \alpha }$

$$
\operatorname* { s u p } _ { \theta \in B } \| H _ { n } ( \theta ) - \mathbb { E } H _ { n } ( \theta ) \| \leq \frac { L _ { \mathcal { B } } } { \lambda _ { \mathcal { B } } } \left( Z _ { n } + \mathbb { E } Z _ { n } \right) + C M _ { \mathcal { B } } n ^ { - 2 } .
$$

By Step 2, the event $\begin{array} { r } { Z _ { n } \leq C \sqrt { \frac { d \log ( e n ) + \log ( 1 / \delta ) } { n } } } \end{array}$ has probability at least $1 - \delta$ . On its intersection with $\mathcal { E } _ { \alpha }$ we have

$$
\operatorname* { s u p } _ { \theta \in B } \| H _ { n } ( \theta ) - \mathbb { E } H _ { n } ( \theta ) \| \leq \frac { L _ { B } } { \lambda _ { B } } \left( C \sqrt { \frac { d \log ( e n ) + \log ( 1 / \delta ) } { n } } + C \sqrt { \frac { d \log ( e n ) } { n } } \right) + C M _ { B } n ^ { - 2 } .
$$

The bounds from Step 3 absorb the $n ^ { - 2 }$ remainder into the displayed $n ^ { - 1 / 2 }$ scale.

$$
\operatorname* { s u p } _ { \theta \in \mathcal B } \| F ( \theta , \alpha _ { n } ( \theta ) ) - \mathbb E F ( \theta , \alpha _ { n } ( \theta ) ) \| \le C \operatorname* { m a x } \left\{ \sigma , \| \theta ^ { * } \| \right\} \exp \left( C \frac { \| \theta ^ { * } \| ^ { 2 } } { \sigma ^ { 2 } } \right) \sqrt { \frac { d \log ( e n ) + \log ( 1 / \delta ) } { n } } .
$$

We conclude that with probability at least $1 - \delta - C n ^ { - 2 }$ 2