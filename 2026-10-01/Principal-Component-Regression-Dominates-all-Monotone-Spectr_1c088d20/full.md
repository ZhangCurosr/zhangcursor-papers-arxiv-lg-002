# Principal Component Regression Dominates all Monotone Spectral Filters for Linear Regression

Juno Kim<sup>1</sup>, Hengyu Fu<sup>1</sup>, Peter Bartlett<sup>1,2</sup>, Jason D. Lee<sup>1</sup>, and Jingfeng Wu<sup>1</sup>

<sup>1</sup>University of California, Berkeley <sup>2</sup>Google DeepMind

October 1, 2026

## Abstract

We compare the instance-wise, finite-sample risks of monotone spectralfilters for linear regression, a broad class of estimators including principal component regression (PCR), gradient descent (GD), and ridge regression. We show that PCR dominates all monotone spectral filters: compared to any such filter, the risk of optimally tuned PCR is no bigger by a constant factor for all problems. Furthermore, the dominance is strong if the filter is separated from step functions (e.g., GD and ridge): there exist problem instances for which the risk of PCR is smaller by a polynomial factor in sample size dependence. Our comparison results show that PCR is optimal and thus admissible among monotone filters, significantly extending Wu et al. (2026)’s result that GD strongly dominates ridge. From a technical perspective, we establish new upper and lower bounds for general spectral filters, which are instance-wise sharp when specialized to ridge or GD, recovering or improving the best-known bounds.

## 1 Introduction

In statistical decision theory, one estimator dominates another if the risk of the former is always no larger than that of the latter, and strongly dominates another if additionally it is sometimes strictly better (Wald, 1950; Berger, 1985). A seminal example is that the James–Stein estimator strongly dominates ordinary least squares (OLS) for estimating the mean of Gaussian distributions in three or higher dimensions (James and Stein, 1961), and hence the latter is inadmissible. Another classical example is Rao–Blackwellization, which always yields an estimator that dominates the original estimator (Rao et al., 1945; Blackwell, 1947). Note that this comparison is required to hold over all problem instances of interest, providing a much stronger performance guarantee than worst-case performance.

The notion of “dominance” has recently been adapted to statistical learning contexts. In linear regression with fixed design, Dhillon et al. (2013) showed that principal component regression (PCR) strongly dominates ridge regression in terms of rate: with comparable regularization (the number of selected features for PCR and the $\ell _ { 2 }$ penalty for ridge), the excess risk of PCR is always no larger than a constant times that of ridge, while it can be arbitrarily smaller even if ridge is tuned optimally. Similarly, early-stopped gradient descent (GD) also strongly dominates ridge for linear regression in fixed design (see, e.g., Ali et al. (2019) and Wu et al. (2026, Appendix A)). For linear regression in the more challenging random design setting, a recent work by Wu et al. (2026) showed that GD strongly dominates ridge, while it is incomparable with online stochastic gradient descent (SGD).<sup>1</sup> Such instance-wise risk comparisons ofer a powerful perspective for evaluating estimators beyond their worst-case performance (Wainwright, 2019): estimators that attain the same minimax rates can now be compared according to their instance-wise rates, under the preorder defined by dominance.

Table 1: Dominance results for linear regression. For a problem class $\mathbb { P } ,$ we say algorithm A dominates B, or ${ \mathcal { A } } \preceq _ { \mathbb { P } } { \mathcal { B } } .$ , if its (excess) risk is never more than a constant multiple of $\mathcal { B } \mathrm { { s } }$ on every problem instance in P. We say A strongly dominates B, or ${ \mathcal { A } } \prec _ { \mathbb { P } } { \mathcal { B } } .$ , if additionally its risk is polynomially smaller (w.r.t. sample size) for some instances. We establish that for all well-specified linear regression problems L (4) with bounded signal-to-noise ratio, PCR strongly dominates GD; previous work showed that GD strongly dominates ridge in the same class (Wu et al., 2026). Thus, GD and ridge are both inadmissible.  
For all linear regression problems with Gaussian design $\mathbb { G } \left( 2 \right)$ , a subset of L, we prove that PCR dominates all monotone spectral filters, a wide class of regularization-based methods including GD and ridge (Definition 1). Thus, PCR is optimal and hence admissible among all such filters. We further prove that PCR strongly dominates all monotone filters that are uniformly separated from step functions $( \mathrm { e . g . }$ , Example 1).
<table><tr><td rowspan="2">well-specified linear regression problems (L)</td><td> $\mathrm { G D \prec _ { \mathbb { L } } \ r i d g e }$ </td><td>Wu et al. (2026)</td></tr><tr><td> $\mathrm { P C R \prec _ { \mathbb { L } } G D }$ </td><td>Theorem 2.3</td></tr><tr><td rowspan="2">Gaussian linear regression problems (G)</td><td>PCR ≤G all monotone filters</td><td>Theorem 2.1</td></tr><tr><td>PCR  $\prec _ { \mathbb { G } }$  all non-step monotone filters</td><td>Theorem 2.2</td></tr></table>

Contributions. In this work, we evaluate and compare the instance-wise finite-sample risks of monotone spectral filters in linear regression, a broad class of estimators including PCR, GD, and ridge regression (Definition 1). We show that for linear regression with Gaussian random design and bounded signal-to-noise ratio, PCR dominates all monotone spectral filters (Theorem 2.1) in the same sense as described above (see Definition 2). Moreover, PCR strongly dominates all monotone filters that are separated from step functions (Theorem 2.2), which includes both GD and ridge. Therefore, among monotone filters, PCR is optimal and hence admissible, while GD is inadmissible. In comparison, previous work only showed that GD strongly dominates ridge (Wu et al., 2026), i.e., ridge is inadmissible; and for linear regression with fixed design, PCR and GD both dominate ridge (Dhillon et al., 2013; Ali et al., 2019).

In addition, we make the following contributions:

1. We provide instance-wise tight upper and lower risk bounds for GD (Corollary 3.1). These bounds match, up to a constant factor, for all linear regression problems with bounded signal-to-noise ratio and all GD hyperparameters (stepsize and stopping time), and hold under weaker distributional assumptions (Assumption 1, as used by Wu et al., 2026). Thus, we fully characterize GD’s implicit regularization and the impact of hyperparameters in linear regression. Previously, instance-wise tight bounds were only known for ridge regression (Tsigler and Bartlett, 2023) and online SGD (Wu et al., 2022a; Zou et al., 2023).

2. We provide novel upper and lower risk bounds for PCR (Theorem 4.1) that, to our knowledge, improve on the best known results; see Section 6 for discussion. Our bounds further establish that PCR strongly dominates GD in the more general setting of well-specified linear regression problems (Assumption 1), matching the setting of Wu et al. (2026).

3. We introduce new techniques for analyzing general spectral filters. Our approach extends classical leaveone-out ideas in ridge analysis by controlling noncommutative matrix perturbations using tools from the theory of Schur multipliers and matrix divided diferences. While these tools were developed in other areas of mathematics (see, e.g., Aleksandrov and Peller, 2016), we demonstrate their utility in the context of statistical learning (see Section 5 and Appendix E).

The rest of the paper is structured as follows. The setting and main dominance results are presented in Section 2. The bounds for GD and PCR are presented in Sections 3 and 4; these in turn are proved using the general methods developed in Section 5. An overview of related works is given in Section 6. All missing proofs can be found in the appendix.

## 2 Main Results

## 2.1 Preliminaries

Linear regression. Let H be a separable Hilbert space of either finite or countably infinite dimension. Let $\textbf { X \in }$ H and $y \in$ R be a pair of covariates and response, with population distribution $\mu ( \mathbf { x } , y )$ . In linear regression, we seek to minimize the population risk, defined as

$$
\begin{array} { r } { \mathcal { R } ( \mathbf { w } ) : = \mathbb { E } ( \mathbf { x } ^ { \top } \mathbf { w } - y ) ^ { 2 } , \quad \mathbf { w } \in \mathbb { H } , } \end{array}
$$

where the expectation is over $\mu ( \mathbf { x } , y )$ . Denote the optimal parameter as

$$
\mathbf { w } ^ { * } \in \arg \operatorname* { m i n } \mathcal { R } ( \cdot ) .
$$

If the optimal parameter is not unique, let $\mathbf { w } ^ { * }$ be the one with minimum $\ell _ { 2 }$ -norm. The excess risk is defined as

$$
\mathcal { E } ( \mathbf { w } ) : = \mathcal { R } ( \mathbf { w } ) - \mathcal { R } ( \mathbf { w } ^ { * } ) = \| \mathbf { w } - \mathbf { w } ^ { * } \| _ { \Sigma } ^ { 2 } , \quad \mathbf { w } \in \mathbb { H } .
$$

We refer to a linear regression problem by its population probability measure $\mu ( \mathbf { x } , y )$ . When necessary, we also write $\varepsilon _ { \mu }$ to emphasize the dependence on the population distribution. Let $n \geq 1$ be the sample size and let $\left( \mathbf { x } _ { i } , y _ { i } \right) _ { i = 1 } ^ { n }$ be � independent copies of $\left( \mathbf { x } , \mathbf { y } \right)$ . We also write

$$
\mathbf { X } : = \left[ \begin{array} { l } { \mathbf { x } _ { 1 } ^ { \top } } \\ { \vdots } \\ { \mathbf { x } _ { n } ^ { \top } } \end{array} \right] \in \mathbb { H } ^ { n } , \quad \mathbf { y } : = \left[ \begin{array} { l } { y _ { 1 } } \\ { \vdots } \\ { y _ { n } } \end{array} \right] \in \mathbb { R } ^ { n } .
$$

The population and empirical covariance of the covariates are denoted as

$$
\pmb { \Sigma } : = \mathbb { E } [ \mathbf { x } \mathbf { x } ^ { \top } ] \in \mathbb { H } ^ { \otimes 2 } , \qquad \hat { \mathbf { \Sigma } } : = \frac { 1 } { n } \mathbf { X } ^ { \top } \mathbf { X } \in \mathbb { H } ^ { \otimes 2 } .
$$

We assume $\operatorname { t r } ( \Sigma ) <$ ∞ in order for the problem to be learnable. Let the eigendecomposition of $\pmb { \Sigma }$ be

$$
\pmb { \Sigma } = \sum _ { i \geq 1 } \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } , \quad \lambda _ { 1 } \geq \lambda _ { 2 } \geq . . . ,
$$

where $( \lambda _ { i } , \mathbf { u } _ { i } ) _ { i \geq 1 }$ are the eigenvalues, in non-increasing order, and their corresponding eigenvectors. For an index $k ,$ , allowed to be zero or infinity, we define

$$
\pmb { \Sigma } _ { 0 : k } : = \sum _ { i \leq k } \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } , \quad \pmb { \Sigma } _ { k : \infty } : = \sum _ { i > k } \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } ,
$$

both of which are positive semidefinite (PSD) matrices in $\mathbb { H } ^ { \otimes 2 }$ . We also denote

$$
\alpha _ { k } : = \left( \frac { \sum _ { i > k } \lambda _ { i } } { n } \right) ^ { 2 } + \frac { \sum _ { i > k } \lambda _ { i } ^ { 2 } } { n } .\tag{1}
$$

Algorithms. Principal component regression (PCR) refers to OLS applied to the features selected by principal component analysis. Formally, let $( \hat { \lambda } _ { i } , \hat { \mathbf { u } } _ { i } ) _ { i \geq 1 }$ be the eigendecomposition of $\hat { \Sigma }$ , then PCR is defined as

$$
\hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } : = \frac { 1 } { n } \Bigg ( \sum _ { i : \hat { \lambda } _ { i } > \rho } \hat { \lambda } _ { i } \hat { \mathbf { u } } _ { i } \hat { \mathbf { u } } _ { i } ^ { \top } \Bigg ) ^ { - 1 } \mathbf { X } ^ { \top } \mathbf { y } ,\tag{PCR}
$$

where the matrix inverse is understood as the Moore-Penrose pseudoinverse throughout the paper. Here, the spectral threshold $\rho \ge 0$ is a hyperparameter controlling the number of features used by PCR.

Gradient descent (GD) refers to the �-th GD iterate with fixed stepsize $\eta > 0 .$ , i.e.,

$$
\hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } : = \mathbf { w } _ { t } , \quad \mathrm { w h e r e } ~ \mathbf { w } _ { 0 } = 0 , ~ \mathbf { w } _ { s } = \mathbf { w } _ { s - 1 } - \frac { \eta } { n } \mathbf { X } ^ { \top } ( \mathbf { X } \mathbf { w } _ { s - 1 } - \mathbf { y } ) ~ ( s \geq 1 ) .\tag{GD}
$$

We assume $\eta \leq ( c ( \lambda _ { 1 } + \operatorname { t r } ( \Sigma ) / n ) ) ^ { - 1 }$ for a suficiently large constant $c > 1$ throughout the paper, which ensures GD is in the stable regime, and treat the stopping time $t \geq 0$ as the main hyperparameter.

More generally, we consider the following class of estimators.

Definition 1 (spectral filters). A spectralfilter is a measurable function $g : \mathbb { R } _ { \geq 0 } \to \mathbb { R }$ satisfying $g ( 0 ) = 0$ . The corresponding estimator is defined as

$$
\hat { \mathbf { w } } _ { g } : = \frac { 1 } { n } \psi ( \hat { \Sigma } ) \mathbf { X } ^ { \top } \mathbf { y } , \quad \mathrm { w h e r e } \ \psi ( z ) : = \left\{ \begin{array} { l l } { g ( z ) / z } & { z > 0 , } \\ { 0 } & { z = 0 . } \end{array} \right.\tag{Spec}
$$

If moreover $0 \leq g \leq 1 , g$ is a shrinkage filter; if $g$ is nondecreasing, it is a monotone filter. The class of all monotone shrinkage filters is denoted by $\mathcal { G }$

Spectral and shrinkage filters, also called spectral regularization methods, have a long history in statistics and inverse problems, see for instance (Kneip, 1994; Engl et al., 1996; De Vito et al., $2 0 0 5 \mathrm { a } , \mathrm { b } ;$ Bauer et al., 2007; Gerfo et al., 2008). Shrinkage filters can be interpreted as OLS with each feature $( \hat { \lambda } _ { i } , \hat { \mathbf { u } } _ { i } )$ weighted (shrunk) by a factor of $g ( \hat { \lambda } _ { i } )$ . Monotonicity is natural since lower variance directions should intuitively be given less weight; this property has also been studied in the context of ordered linear smoothers (Kneip, 1994; Golubev, 2010). Clearly, any useful monotone filter must be shrinkage, as otherwise it will incur constant risk. The following classical estimators are all examples of monotone shrinkage filters.

${ \mathrm { O L S } } \colon g ( z ) = 1 \{ z > 0 \}$

$\operatorname { P C R } \colon g ( z ) = 1 \{ z > \rho \}$ where $\rho \ge 0$

• Gradient descent: $g ( z ) = 1 - ( 1 - \eta z ) ^ { t }$ where $t \in \mathbb { N } , \ \eta > 0$

• Ridge regression: $g ( z ) = z / ( z + \lambda )$ where $\lambda > 0$

• Iterated Tikhonov (iterated ridge): $g ( z ) = 1 - ( \lambda / ( z + \lambda ) ) ^ { p } \ \mathrm { w h e r e } \ \lambda > 0 , \ p \geq 1 \ \mathrm { ( R i l e y , 1 9 5 5 ) }$

• Power-exponential: $g ( z ) = 1 - \exp ( - ( t z ) ^ { p } )$ where $t > 0 , \ p \ \geq \ 1$ ; includes gradient flow $( p = 1 )$ and Gaussian type filter $( p = 2 )$ (Calvetti et al., 1999)

• Ridge PCR: $g ( z ) = \operatorname* { m i n } \{ 1 , z / \rho \}$ where $\rho > 0$ (Carrasco et al., 2007)

On the other hand, some algorithms such as momentum or accelerated GD are spectral filters but generally not monotone or shrinkage; while SGD or adaptive methods such as Adam are not of the form (Spec).

Definition 2 (dominance and admissibility). Let P be a set of linear regression problems. Let $\hat { \mathbf { w } } _ { \rho }$ and $\hat { \mathbf { w } } _ { \tau } ^ { \prime }$ be two classes of estimators constructed on � samples $\left( \mathbf { x } _ { i } , y _ { i } \right) _ { i = 1 } ^ { n }$ , indexed by hyperparameters $\rho$ and �, respectively.

• We say $\hat { \mathbf { w } } _ { \rho }$ dominates $\hat { \mathbf { w } } _ { \tau } ^ { \prime }$ over $\mathbb { P } ,$ if there are constants $c _ { 0 } , n _ { 0 } \geq 1$ such that for every $\mu \in \mathbb { P }$ and $n \geq n _ { 0 }$

with probability at least 0.99,

$$
\operatorname* { i n f } _ { \rho } \mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { \rho } ) | \mathbf { X } ] \leq c _ { 0 } \operatorname* { i n f } _ { \tau } \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { \tau } ^ { \prime } ) .
$$

• We say $\hat { \mathbf { w } } _ { \rho }$ strongly dominates $\hat { \mathbf { w } } _ { \tau } ^ { \prime }$ over $\mathbb { P } ,$ if in addition to the above, there exists $d _ { 0 } > 0$ such that for every $n \geq n _ { 0 }$ , there exists $\mu _ { n } \in \mathbb { P }$ satisfying

with probability at least 0.99,

$$
\operatorname* { i n f } _ { \rho } \mathbb { E } [ \mathcal { E } _ { \mu _ { n } } ( \hat { \mathbf { w } } _ { \rho } ) | \mathbf { X } ] \leq \frac { c _ { 0 } } { n ^ { d _ { 0 } } } \operatorname* { i n f } _ { \tau } \mathbb { E } \mathcal { E } _ { \mu _ { n } } ( \hat { \mathbf { w } } _ { \tau } ^ { \prime } ) .
$$

If another class $\hat { \mathbf { w } } _ { \rho }$ strongly dominates $\hat { \mathbf { w } } _ { \tau } ^ { \prime }$ over $\mathbb { P } ,$ we say that $\hat { \mathbf { w } } _ { \tau } ^ { \prime }$ is inadmissible for $\mathbb { P } ;$ otherwise, it is admissible for P.

Note that this definition requires comparing a “high probability” upper bound for $\hat { \mathbf { w } } _ { \rho }$ against an “in expectation” lower bound for $\hat { \mathbf { w } } _ { \tau } ^ { \prime }$ . This is a mostly technical issue and can be circumvented by taking a Bayesian perspective and assuming a symmetric prior on the optimal parameter $\mathbf { w } ^ { * }$ ; see the discussion in (Tsigler and Bartlett, 2023; Wu et al., 2026).

## 2.2 PCR dominates all monotone filters

In this subsection, we consider the class $\mathbb { G } _ { b }$ of all linear regression problems with Gaussian random design and bounded signal-to-noise ratio, i.e.,

$$
\begin{array} { r } { \mathfrak { Q } _ { b } : = \left\{ \mu ( \mathbf { x } , \mathbf { y } ) : \mathbf { x } \sim N ( 0 , \Sigma ) , \ : \mathbb { E } [ \mathbf { y } | \mathbf { x } ] = \mathbf { x } ^ { \mathsf { T } } \mathbf { w } ^ { * } , \sigma ^ { 2 } / c \le \mathbb { E } [ ( \mathbf { y } - \mathbf { x } ^ { \mathsf { T } } \mathbf { w } ^ { * } ) ^ { 2 } | \mathbf { x } ] \le c \sigma ^ { 2 } \ : \mathrm {  a . s . } , \ : \| \mathbf { w } ^ { * } \| _ { \Sigma } ^ { 2 } \le b \sigma ^ { 2 } \right\} } \end{array}\tag{2}
$$

for constants $b , c > 0$ (we omit the dependence on � for brevity). Each problem instance is essentially specified by the triple $( \boldsymbol { \Sigma } , \boldsymbol { \mathbf { w } } ^ { * } , \sigma ^ { 2 } )$

Our main result shows that PCR with a well-tuned threshold dominates the class $\mathcal { G }$ of all monotone spectral filters over $\mathbb { G } _ { b } ;$ in other words, PCR is instance-wise optimal among $\mathcal { G }$ (up to a constant factor), and in particular, it is admissible.

Theorem 2.1 (PCR dominates monotone filters). Let $\hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c I } }$ <sup>r</sup> be given by (PCR) with regularization $\rho \geq 0 ,$ , and $\hat { \mathbf { w } } _ { g }$ be given by (Spec) with any monotonefilter $g \in { \mathcal { G } }$ . Then $\hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c } _ { 1 } }$ dominates $\{ \hat { \mathbf { w } } _ { g } : g \in \mathcal { G } \}$ over $\mathbb { G } _ { b } . \ S p e c i f i c a l l y$ there exist constants $c _ { 0 } , n _ { 0 } \geq 1$ only depending on � such thatfor every $g \in \mathcal { G } , \mu \in \mathbb { G } _ { b }$ and $n \geq n _ { 0 }$

with probability at least 0.99,

$$
\operatorname* { i n f } _ { \rho \geq 0 } \mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) | \mathbf { X } ] \leq c _ { 0 } \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) .
$$

Proofsketch ofTheorem 2.1. While we obtain risk upper bounds for PCR (Theorem 4.1) and lower bounds for general monotone filters � (Theorem F.1), dominance cannot be established by simply comparing these directly, as these bounds can be loose near the spectral threshold even when $g$ itself is PCR (see Theorem 4.1). Instead, the proof is divided into two branches. When the scale of the tail of the spectrum is large compared to the transition region of �, we show a stronger lower bound for the bias of $g$ based on a Cramér–Rao inequality, which can be compared against the PCR upper bound. When the tail is small, we instead invoke a ‘gluing’ lemma for linear sums of filters to reduce to the case where � vanishes on some interval containing the tail. This enables directly comparing the risk of � against that of PCR with a threshold chosen within the transition region, since the two filters only difer at scales that are much larger than the tail and so exhibit good concentration. □

Clearly, it is not possible for PCR to strongly dominate ${ \mathcal { G } } ,$ , since the latter includes PCR itself. Nonetheless, our next example shows that PCR will strongly dominate any subclass which is separated from step functions in the following sense.

Theorem 2.2 (PCR strongly dominates non-step monotone filters). Fix any $\epsilon > 0$ and $\delta \in ( 0 , 1 / 2 )$ . The class of (�, �)-non-step monotonefilters $\mathcal { G } _ { \epsilon , \delta }$ is defined as

$$
\mathcal { G } _ { \epsilon , \delta } : = \Bigg \{ g \in \mathcal { G } : \frac { \operatorname* { i n f } \{ z : g ( z ) \geq 1 - \delta \} } { \operatorname* { s u p } \{ z : g ( z ) \leq \delta \} } \geq 1 + \epsilon \Bigg \} .\tag{3}
$$

Then $\hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } }$ strongly dominates $\{ \hat { \mathbf { w } } _ { g } : g \in \mathcal { G } _ { \epsilon , \delta } \}$ over $\mathbb { G } _ { b } .$ . Specifically, there exist $c _ { 0 } , n _ { 0 } \geq 1$ , depending at most polynomially on $\epsilon ^ { - 1 } , \delta ^ { - 1 }$ , and $b ,$ for which the following holds: for every $n \geq n _ { 0 } ,$ , there exists $\mu _ { n } \in \mathbb { P }$ such that

with probability at least 0.99,

$$
\operatorname* { i n f } _ { \rho \geq 0 } \mathbb { E } [ \mathcal { E } _ { \mu _ { n } } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) | \mathbf { X } ] \leq \frac { c _ { 0 } } { \sqrt { n } } \operatorname* { i n f } _ { g \in \mathcal { G } _ { \epsilon , \delta } } \mathbb { E } \mathcal { E } _ { \mu _ { n } } ( \hat { \mathbf { w } } _ { g } ) .
$$

The condition (3) implies that � must not increase from � to $1 - \delta$ within a short interval of the form $( z , ( 1 + \epsilon ) z )$ , disallowing step filters very similar to PCR. Clearly, any monotone shrinkage filter $g \in { \mathcal { G } }$ that is not contained in $\mathcal { G } _ { \epsilon , \delta }$ for any $\epsilon , \delta > 0$ must be equal to a PCR filter a.e. Moreover, since (3) is invariant under rescaling, for any $g \in { \mathcal { G } } _ { \epsilon , \delta }$ the family of filters $\{ g ^ { ( \tau ) } ( z ) : = g ( z / \tau ) : \tau > 0 \}$ obtained by varying a scale parameter � will also be contained in $\mathcal { G } _ { \epsilon , \delta }$ . In particular, the families given in Example 1 are all non-step monotone filters, and therefore strongly dominated by PCR.

![](images/f9e34d0c39c4ea1af7892ae0ae60d0d171fb79ed79a79668129d990590afa4a4.jpg)  
An (�, �)-non-step monotone filter.

Example 1 (non-step filters). GD (for any $t \in \mathbb { N } )$ , ridge (for any $\lambda > 0 )$ , iterated Tikhonov (for any $p \geq 1 )$ and ridge PCR are all contained in $\mathcal { G } _ { 1 , 1 / 3 }$ . Power-exponentialfilters with $p \geq 1$ are contained in $\mathcal { G } _ { 1 , 1 / ( 2 ^ { p } + 1 ) }$

## 2.3 PCR strongly dominates GD

Specializing to GD, we now show that PCR also strongly dominates GD over a significantly more general set of linear regression problems given by the below assumptions. Previously, Wu et al. (2026) showed that GD strongly dominates ridge in the same problem set.

Assumption 1 (conditions for risk bounds). We assume for constants $c _ { x } > 0$ and $c _ { y } \geq 1$ that:

A. the entries of $\mathbf { \dot { \boldsymbol { \Sigma } } } ^ { - 1 / 2 } \mathbf { \boldsymbol { x } }$ are independent, centered and $c _ { x } ^ { 2 }$ -subgaussian;

B. the conditional noise is zero mean, and its variance is boundedfrom above (for upper bounds) andfrom below (for lower bounds),

$$
\begin{array} { r } { { \mathbb { E } } [ { \mathbf { y } } \mid { \mathbf { x } } ] = { \mathbf { x } } ^ { \top } { \mathbf { w } } ^ { * } , \quad \sigma ^ { 2 } / c _ { y } \leq { \mathbb { E } } [ ( { \mathbf { y } } - { \mathbf { x } } ^ { \top } { \mathbf { w } } ^ { * } ) ^ { 2 } \mid { \mathbf { x } } ] \leq c _ { y } \sigma ^ { 2 } \ a . s . ; } \end{array}
$$

C. for lower bounds only, the distribution of each component of $\mathbf { \Delta } \cdot \mathbf { { \boldsymbol { \Sigma } } } ^ { - 1 / 2 } \mathbf { x }$ is symmetric, $\begin{array} { r } { i . e . , \ \langle \mathbf { e } _ { i } , \pmb { \Sigma } ^ { - 1 / 2 } \mathbf { x } \rangle \ \stackrel { d } { = } } \end{array}$ $- \langle { { \bf { e } } _ { i } , { \bf { \Sigma } } { \bf { { Z } } } ^ { - 1 / 2 } { \bf { X } } } \rangle$ for all �.

We remark that the bound from above in Assumption 1B is only required for our risk upper bounds, while the bound from below and Assumption 1C are only required for our lower bounds.

For each signal-to-noise ratio $b > 0 ,$ , we define the set of well-specified linear regression problems as

$$
\mathbb { L } _ { b } : = \left\{ \mu ( \mathbf { x } , y ) \mathrm { ~ s a t i s f y i n g ~ A s s u m p t i o n ~ } 1 \mathrm { ~ w i t h ~ } \| \mathbf { w } ^ { * } \| _ { \Sigma } ^ { 2 } \leq b \sigma ^ { 2 } \right\} .\tag{4}
$$

These assumptions are directly borrowed from Wu et al. (2026), and we refer the reader to their paper for discussions on the coverage and limitations of the assumptions. As the main example, Gaussian linear regression problems $\boldsymbol { \mu } \in \mathbb { G } _ { b }$ satisfy Assumption 1 with $c _ { x } = 1$ and $c _ { \mathrm { y } } = c _ { \mathrm { z } }$ , hence $\mathbb { G } _ { b } \subset \mathbb { L } _ { b }$

The following result shows that PCR strongly dominates GD over $\mathbb { L } _ { b }$ for each $b > 0$ . Thus, we establish that GD is also inadmissible for linear regression over the wider class $\mathbb { L } _ { b }$

Theorem 2.3 (PCR strongly dominates GD). Let $\hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } }$ be given by (PCR) with regularization $\rho \geq 0 ,$ and $\hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } }$ be given by (GD) with anyfixed stepsize $\eta \leq ( c ( \lambda _ { 1 } + \operatorname { t r } ( \Sigma ) / n ) ) ^ { - 1 }$ and stopping time $t \geq 0 .$ . Then $\hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } }$ strongly dominates $\hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } }$ over $\mathbb { L } _ { b }$ . Specifically,

• for every $\boldsymbol { \mu } \in \mathbb { L } _ { b }$ and stopping time $t \geq 0 ,$ , there exists a PCR threshold $\rho \ge 0$ such that

with probability at least 0.99, $\mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) | \mathbf { X } ] \leq c \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } )$

where $c \geq 1$ only depends on $c _ { x } , c _ { y }$ , and �.

• For every $n \geq n _ { 0 } .$ , there exist $\mu _ { n } \in \mathbb { L } _ { b }$ and $\rho _ { n } > 0 ,$ depending only on $n , b ,$ satisfying

with probability at least 0.99,

$$
\mathbb { E } [ \mathcal { E } _ { \mu _ { n } } ( \hat { \mathbf { w } } _ { \rho _ { n } } ^ { \mathrm { p c r } } ) \vert \mathbf { X } ] \leq \frac { c } { \sqrt { n } } \operatorname* { i n f } _ { t } \mathbb { E } \mathcal { E } _ { \mu _ { n } } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } ) ,
$$

where $c \geq 1$ is a constant.

The fact that PCR can be polynomially better than even well-tuned gradient descent is quite surprising. Although a polylogarithmic improvement over GD is somewhat expected due to the bias from optimization, the polynomial advantage actually comes from the variance. While PCR has the ability to zero out smaller spectral components that are irrelevant to learning the signal, GD is forced to partially fit all directions, leading to potentially much larger variance. The same intuition is behind the polynomial separation in Theorem 2.2 for general non-step filters. We believe this is an important insight for designing admissible and computationally eficient algorithms for linear regression and broader noisy learning problems.

The proof of Theorem 2.3 requires an instance-wise risk comparison between PCR and GD for all well-specified linear regression problems, and is deferred to Section 4. To this end, we present new risk bounds for GD and PCR in the next two sections.

## 3 Tight Risk Bounds for Gradient Descent

In this section, we provide matching upper and lower bounds on the excess risk of GD for all well-specified linear regression problems with bounded signal-to-noise (SNR) ratio, i.e., over $\mathbb { L } _ { b }$ . This provides, for the first time, a complete picture of the interplay between the optimization error, implicit (tail) regularization, and variance of GD. A more general version without the SNR assumption is given in Theorem B.1 in the appendix. Both the upper and lower bounds improve upon the strongest previously known instance-wise bounds given by Wu et al. (2026).

To state the bounds, recall that $( \lambda _ { i } ) _ { i \geq 1 }$ denote the ordered eigenvalues of $\Sigma ,$ , while $\Sigma _ { 0 : k } , \Sigma _ { k : \infty }$ denote the truncated covariance matrices.

Corollary 3.1 (GD risk under bounded SNR). There exist constants $c _ { 0 } , \ldots , c _ { 4 } > 1$ that only depend on $c _ { x } , c _ { y }$ for which thefollowing hold over $\mathbb { L } _ { b }$ . Let $\hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } }$ be given by (GD) with stepsize $\eta \leq ( c _ { 4 } ( \lambda _ { 1 } + \operatorname { t r } ( \Sigma ) / n ) ) ^ { - 1 }$ and steps $t \geq 1 .$ . Let

$$
k ^ { * } : = \operatorname* { m i n } \left\{ k : \frac { 1 } { \eta t } + \frac { \sum _ { i > k } \lambda _ { i } } { n } \geq c _ { 2 } \lambda _ { k + 1 } \right\} , \quad \tilde { \lambda } : = \frac { 1 } { \eta t } + \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } , \quad D : = k ^ { * } + \frac { 1 } { \tilde { \lambda } ^ { 2 } } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } ,
$$

and additionally assume $k ^ { * } \leq n / c _ { 3 }$

• Upper bound. With probability at least $1 - \exp ( - k ^ { * } / c _ { 0 } )$ over sampling of X, it holds that

$$
\frac { 1 } { c _ { 1 } } \mathbb { E } [ \mathcal { E } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } ) | \mathbf { X } ] \leq \underbrace  \overbrace { \left. ( \mathbf { I } - \eta \mathbf { \boldsymbol { z } } ) ^ { t } \mathbf { w } ^ { * } \right. _ { \Omega _ { 0 , k ^ { * } } } ^ { \mathrm { o p t i m i z a t i o n ~ e r r o r } } } ^ { \mathrm { o p t i m i z a t i o n ~ e r r o r } } \cdot \overbrace { + \left( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) ^ { 2 } \left. \mathbf { w } ^ { * } \right. _ { \Sigma _ { 0 , k ^ { * } } ^ { - 1 } } ^ { \mathrm { i m p l i c i t a t i o n } } } _ { \mathrm { h e a d b i a s } } + \underbrace { \left. \mathbf { w } ^ { * } \right. _ { \Sigma _ { k ^ { * } , \infty } } ^ { 2 } } _ { \mathrm { h i g h - d i m e n s i o n a l ~ t a i l } } \quad \underbrace { + \left( 1 + b \right) \sigma ^ { 2 } \frac { D } { n } } _ { \mathrm { v a r i a n c e } } .
$$

• Lower bound. In expectation,

$$
c _ { 1 } \mathbb { E } \mathcal { E } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } ) \geq \left\| ( \mathbf { I } - \eta \mathbf { \Sigma } ) ^ { t } \mathbf { w } ^ { * } \right\| _ { \Sigma _ { 0 : k ^ { * } } } ^ { 2 } + \left( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) ^ { 2 } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } + \| \mathbf { w } ^ { * } \| _ { \Sigma _ { k ^ { * } \infty } } ^ { 2 } + \sigma ^ { 2 } \frac { D } { n } .
$$

The bounds can be interpreted as follows. The bias is decomposed into the essentially low-dimensional “head” and high-dimensional “tail” parts according to the critical index $k ^ { * }$ (Bartlett et al., 2020; Tsigler and Bartlett, 2023). The signal from the tail cannot be reliably estimated, so it remains in the risk. The head bias is further characterized as the sum of the GD optimization error, which is time-dependent, and the implicit regularization efect of the tail. The latter arises because the tail collectively acts as an additional ridge-like regularizer with strength $n ^ { - 1 } \textstyle \sum _ { i > k ^ { * } } \lambda _ { i }$ . This can be improved with debiasing or negative $\ell _ { 2 }$ regularization (Tsigler and Bartlett, 2023), however this may lead to an increase in the variance. Finally, the variance is obtained in terms of the efective dimension $D ,$ , which is the sum of the head rank and contribution of tail fluctuations. The proof of the variance bound is given in Wu et al. (2026) and is due to Bartlett et al. (2020).

Comparison to Wu et al. (2026). Both our upper and lower bounds improve upon the bounds for GD previously shown in Wu et al. (2026). Their Theorem 3.1 bounds the head bias with the ridge-like bias $\mathbf { \widetilde { \lambda } } ^ { 2 } \lVert \mathbf { w } ^ { * } \rVert _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 }$ , which obscures the optimization error by replacing it with the crude estimate

$$
( 1 - \eta \lambda _ { i } ) ^ { t } \leq \frac { 1 } { \eta \lambda _ { i } t } \quad \implies \quad \left\| ( \mathbf { I } - \eta \boldsymbol { \Sigma } ) ^ { t } \mathbf { w } ^ { * } \right\| _ { \Sigma _ { 0 : k ^ { * } } } ^ { 2 } \leq \frac { 1 } { ( \eta t ) ^ { 2 } } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } .
$$

While this sufices to show that GD is no worse than ridge as in their Theorem 3.2, it is loose for components above the efective regularization threshold $1 / ( \eta t )$ , since GD optimization error decays exponentially in �. Our result makes the separation of optimization and regularization explicit for every instance.

Their Theorem 4.3 gives a more detailed characterization in terms of efective bias and efective variance, the latter of which is equivalent to the head term involving $\alpha _ { k ^ { * } }$ in Theorem B.1. Here, their order-1 efective dimension is the tail regularization, while the $D / n$ term is due to the empirical fluctuation of the tail and has been absorbed into the variance in Corollary 3.1. Nonetheless, their result is restricted to moderate stopping times $t \leq c n$ and still obtains a suboptimal bound for the optimization error in the efective bias, with a decay factor of $( 1 - \eta \pmb { \Sigma } ) ^ { t / 2 }$ instead of $( 1 - \eta \pmb { \Sigma } ) ^ { t }$

Their lower bound Theorem 4.1 also uses the suboptimal OLS critical index $\begin{array} { r } { \ell ^ { * } : = \operatorname* { m i n } \lbrace k : n ^ { - 1 } \sum _ { i > k } \lambda _ { i } \geq } \end{array}$ $c _ { 2 } \lambda _ { k + 1 } \}$ , resulting in a crude lower bound which is simply the sum of OLS bias and ridge variance (Bartlett et al., 2020; Tsigler and Bartlett, 2023). On the other hand, our lower bound is tight up to constant factors.

## 4 Risk Bounds for Principal Component Regression

For PCR at spectral threshold $\rho ,$ , recalling the definition of $\alpha _ { k }$ from (1), we obtain the following upper and lower bounds. The proofs essentially follow from the general upper and lower bounds developed in Section 5 and are deferred to Appendix C.

Theorem 4.1 (PCR risk). Let $\hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } }$ be given by (PCR) with threshold $\rho \geq 0 .$ Under Assumption $^ { l , }$ there exist constants $c _ { 0 } , \ldots , c _ { 4 } > 1$ that only depend on $c _ { x } , c _ { y } \ f o r$ which thefollowing holds. Let

$$
k ^ { * } : = \operatorname* { m i n } \left\{ k : 1 . 1 \rho + \frac { 1 } { c _ { 2 } } \frac { \sum _ { i > k } \lambda _ { i } } { n } \geq \lambda _ { k + 1 } \right\} .
$$

• Upper bound. If additionally $k ^ { * } \leq n / c _ { 3 }$ , then with probability at least $1 - \delta - \exp ( - n / c _ { 0 } )$ over sampling of X, it holds that

$$
\frac { 1 } { c _ { 1 } } \mathbb { E } [ \mathcal { E } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) | \mathbf { X } ] \leq \left( \alpha _ { k ^ { * } } + \frac { \lambda _ { k ^ { * } + 1 } ^ { 2 } \log ( 1 / \delta ) } { n } \right) \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } + \| \mathbf { w } ^ { * } \| _ { \Sigma _ { k ^ { * } ; \infty } } ^ { 2 } + \sigma ^ { 2 } \frac { \operatorname* { m i n } \{ D , \hat { k } \} } { n } ,
$$

where

$$
D : = k ^ { * } + \left( \rho + \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) ^ { - 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } , \quad \hat { k } : = \# \{ i : \hat { \lambda } _ { i } > \rho \} .
$$

Also, with probability at least $1 - \exp ( - n / c _ { 0 } )$ it holds that

$$
\hat { k } \leq \# \left\{ i : \lambda _ { i } + \frac { \sum _ { j > n } \lambda _ { j } } { n } > \frac { \rho } { c _ { 2 } } \right\} .
$$

• Lower bound. Suppose additionally that $\mathbf { x } \sim N ( 0 , \pmb { \Sigma } )$ . Let

$$
r ^ { * } : = \operatorname* { m i n } \left\{ r : \lambda _ { r + 1 } + \frac { \sum _ { i > r } \lambda _ { i } } { n } \leq 0 . 9 \rho \right\} , \quad \tilde { \rho } : = \operatorname* { m a x } \left\{ 1 . 1 \rho , c _ { 3 } \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right\}
$$

and additionally assume $r ^ { * } \leq n / c _ { 3 }$ . Then in expectation,

$$
c _ { 1 } \mathbb { E } \mathscr { E } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) \geq \alpha _ { r ^ { * } } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } + \| \mathbf { w } ^ { * } \| _ { \Sigma _ { r ^ { * } ; \infty } } ^ { 2 } + \frac { \sigma ^ { 2 } } { n } \# \{ i : \lambda _ { i } > \tilde { \rho } \} .
$$

Notice that the structure of the PCR upper bound is similar to the bounds for GD (Theorem B.1) or ridge (Tsigler and Bartlett, 2023); indeed, we will soon see that all of these bounds follow from the same general principle. The gap between the bias upper and lower bounds is due to the mismatch between the critical indices $k ^ { * } , r ^ { * }$ . Ignoring the tail, the defining conditions of the two indices are essentially $\lambda _ { r ^ { * } } > 0 . 9 \rho$ and $\lambda _ { k ^ { * } } > 1 . 1 \rho$ . Both constants can be made arbitrarily close to 1 by adjusting the other constants. Nonetheless, the existence of some gap is unavoidable in our approach, as demonstrated by the following example.

Lemma 4.2. Fix $\rho > 0 , \epsilon \in ( 0 , 0 . 0 5 )$ and set $m _ { n } : = \lfloor \epsilon ^ { 2 } n / c _ { 3 } \rfloor$ . Consider the noiseless Gaussian instances

$$
\Sigma _ { n } ^ { \pm } : = \mathrm { d i a g } \left( 2 \rho , ( 1 \pm \epsilon ) \rho { \bf I } _ { m _ { n } } \right) , \quad { \bf w } _ { n } ^ { * } : = { \bf e } _ { 1 } , \quad \sigma _ { n } : = 0 .
$$

Then $k ^ { * } = 1$ and $r ^ { * } = m _ { n } + 1$ in Theorem 4.1 for both instances. Moreover the risk upper bound is $\Theta ( \epsilon ^ { 2 } \rho )$ which is attained by $\Sigma _ { n } ^ { - } .$ ; while the risk lower bound is zero, which is attained by $\Sigma _ { n } ^ { + }$

We now have all the necessary ingredients to prove Theorem 2.3.

Proof of Theorem 2.3. To prove the first claim, we directly compare the PCR upper bound and GD lower bound. Let $t \geq 1$ and let $k ^ { * } , \tilde { \lambda } , D$ be as in Corollary 3.1. If $k ^ { * } > n / c _ { 3 }$ or $D > n$ , then $D \geq k ^ { * }$ and the variance lower bound (which holds regardless of $k ^ { * }$ , see Theorem B.1) gives

$$
\mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } ) \geq c \sigma ^ { 2 } .
$$

Since the zero estimator (obtained by taking $\rho \to \infty )$ achieves risk $\| \mathbf { w } ^ { * } \| _ { \Sigma } ^ { 2 } \leq b \sigma ^ { 2 }$ , the claim follows. Hence we may suppose $k ^ { * } \leq n / c _ { 3 } , D \leq n$ . Set $\rho : = ( 1 . 1 c _ { 2 } \eta t ) ^ { - 1 }$ , so that the critical index in Theorem 4.1 is equal to $k ^ { * }$ and the corresponding efective dimension is at most $c D$ . Then with probability at least 0.99,

$$
\frac { 1 } { c _ { 1 } } \mathbb { E } [ \mathcal { E } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) | \mathbf { X } ] \leq \alpha _ { k ^ { * } } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } + \| \mathbf { w } ^ { * } \| _ { \Sigma _ { k ^ { * } \times \infty } } ^ { 2 } + \sigma ^ { 2 } \frac { D } { n } \leq c _ { 1 } \mathbb { E } \mathcal { E } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } ) .
$$

The second claim follows from Theorem 2.2 and Example 1.

Comparison to Hucker and Wahl (2023). The closest existing bounds on PCR are the upper bounds in Theorems 1 and 2 of Hucker and Wahl (2023). Our upper bound improves upon both results, as we now describe. Since their PCR algorithm is defined by the number of selected components $k ,$ we equivalently choose $\rho = \lambda _ { k + 1 }$ and set the failure probability as $\delta = e ^ { - n / c _ { 0 } }$

Let $c , C > 0$ denote constants. When $\begin{array} { r } { n ^ { - 1 } \sum _ { i > k } \lambda _ { i } \le c \lambda _ { k + 1 } } \end{array}$ , their Theorem 1 upper bounds the bias and variance by $\lambda _ { k + 1 } \| \mathbf { w } ^ { * } \| ^ { 2 }$ and $\sigma ^ { 2 } k / n$ , respectively. Noting that $k ^ { * } \leq k$ by definition, some algebra gives

$$
\frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \leq \frac { \sum _ { i > k } \lambda _ { i } + k \lambda _ { k ^ { * } + 1 } } { n } \leq c \lambda _ { k + 1 } + \frac { k } { n } \left( 1 . 1 \lambda _ { k + 1 } + \frac { 1 } { c _ { 2 } } \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) \quad \implies \quad \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } , \ \lambda _ { k ^ { * } + 1 } \lesssim \lambda _ { k + 1 } .
$$

Using that $\begin{array} { r } { n ^ { - 1 } \sum _ { i > k ^ { * } } \lambda _ { i } < c _ { 2 } \lambda _ { k ^ { * } } } \end{array}$ , each head component in our upper bound has coeficient bounded as

$$
\frac { \alpha _ { k ^ { * } } + \lambda _ { k ^ { * } + 1 } ^ { 2 } } { \lambda _ { k ^ { * } } } \leq \frac { 1 } { \lambda _ { k ^ { * } } } \bigg ( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \bigg ) ^ { 2 } + \frac { \lambda _ { k ^ { * } + 1 } } { \lambda _ { k ^ { * } } } \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } + \lambda _ { k ^ { * } + 1 } \lesssim \lambda _ { k + 1 } ,
$$

and each tail component has coeficient at most $\lambda _ { k ^ { * } + 1 } \lesssim \lambda _ { k + 1 }$ . Thus, our bounds for the bias and variance are both tighter in the setting of their Theorem 1.

When $\begin{array} { r } { n ^ { - 1 } \sum _ { i > k } \lambda _ { i } \ge C \lambda _ { k + 1 } } \end{array}$ , their Theorem 2 gives a bias upper bound of $\begin{array} { r } { ( n ^ { - 1 } \sum _ { i > j ^ { * } } \lambda _ { i } ) \| \mathbf { w } ^ { * } \| ^ { 2 } } \end{array}$ , where $\begin{array} { r } { j ^ { * } : = \operatorname* { m i n } \{ j : n ^ { - 1 } \sum _ { i > j } \lambda _ { i } \ge C \lambda _ { j + 1 } \} } \end{array}$ . Noting that $j ^ { * } \leq k$

$$
\frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \leq \frac { \sum _ { i > j ^ { * } } \lambda _ { i } } { n } + \frac { j ^ { * } } { n } \left( 1 . 1 \lambda _ { k + 1 } + \frac { 1 } { c _ { 2 } } \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) \quad \implies \quad \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } , \ \lambda _ { k ^ { * } + 1 } \lesssim \frac { \sum _ { i > j ^ { * } } \lambda _ { i } } { n } ,
$$

and the same argument shows that our bound for the bias is tighter. We remark that their variance bound is incomparable to ours in the setting of their Theorem 2.

## 5 Risk Bounds for General Spectral Filters

In this section, we present general techniques to bound the excess risk of arbitrary spectral filters. For a given filter $g : \mathbb { R } _ { \ge 0 } \to \mathbb { R }$ , the bias-variance decomposition is (omitting the dependence on $\mu , \mathbf { X } )$

$$
\mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) | \mathbf { X } ] = \underbrace { \Vert ( \mathbf { I } - g ( \hat { \mathbf { \Sigma } } ) ) \mathbf { w } ^ { * } \Vert _ { \Sigma } ^ { 2 } } _ { = : \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) } + \underbrace { \Theta \left( \frac { \sigma ^ { 2 } } { n } \right) \mathrm { t r } \left( \Sigma \hat { \Sigma } ^ { - 1 } g ( \hat { \Sigma } ) ^ { 2 } \right) } _ { = : \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g } ) } .
$$

The variance is typically straightforward to control. By monotonicity of trace, it holds that

$$
\begin{array} { r } { g _ { 1 } \geq g _ { 2 } \geq 0 \quad \implies \quad \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g _ { 1 } } ) \gtrsim \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g _ { 2 } } ) . } \end{array}
$$

Hence one can obtain variance bounds by directly comparing against ridge filters with suitable $\ell _ { 2 }$ penalty, for which tight variance bounds are known (Tsigler and Bartlett, 2023). The main dificulty lies in controlling the bias, due to the noncommutativity of �, �<sup>ˆ</sup> .

In Section 5.1, we present an upper bound for general spectral filters, building on the theory of Schur multipliers to control noncommutative matrix perturbations. This result is used to obtain our upper bounds for PCR and GD. Complementing this result, in Section 5.2, we present a useful bias lower bound for monotone filters. Finally, as an application, we derive uniformly tight bounds for iterative Tikhonov regularization.

## 5.1 General upper bound: Schur multipliers and matrix divided diference

Before stating our results, we give a brief background on Schur multipliers, norm, and factorization; more details are given in Appendix E. For simplicity, we focus on the finite-dimensional case, however, the subsequent discussions can be extended to infinite-dimensional Hilbert space in the usual manner.

Schur multiplier. Fix two symmetric matrices $\mathbf { A } \in \mathbb { R } ^ { n \times n }$ and $\mathbf { B } \in \mathbb { R } ^ { m \times m }$ with eigendecompositions

$$
\mathbf { A } : = \mathbf { U } \mathbf { A } \mathbf { U } ^ { \top } : = \sum _ { i } \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } , \quad \mathbf { B } : = \mathbf { V } \Gamma \mathbf { V } ^ { \top } : = \sum _ { j } \gamma _ { j } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } .
$$

A kernel function $k : \mathbb { R } \otimes \mathbb { R } $ R induces a matrix operator from $\mathbb { R } ^ { n \times m }$ to $\mathbb { R } ^ { n \times m }$ by

$$
k ( \mathbf { A } , \mathbf { B } ) : = \sum _ { i , j } k ( \lambda _ { i } , \gamma _ { j } ) \big ( \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } \big ) \otimes \big ( \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \big ) .
$$

This operator is referred to as the Schur multiplier, also known as a double operator integral.

Divided diference for matrix functions. For a scalar function $f : \mathbb { R } \to \mathbb { R } .$ , its (first) divided diference is a kernel function $f ^ { [ 1 ] } : \mathbb { R } \otimes \mathbb { R }  \mathbb { R }$ such that

$$
f ( x ) - f ( y ) = f ^ { [ 1 ] } ( x , y ) ( x - y ) { \mathrm { ~ f o r ~ a l l ~ } } x , y \in \mathbb { R } .
$$

Note that the existence of $f ^ { [ 1 ] }$ does not require $f$ to be diferentiable. Clearly, $f ^ { [ 1 ] }$ is symmetric in its two arguments; its of-diagonal value is given by

$$
f ^ { [ 1 ] } ( x , y ) : = { \frac { f ( x ) - f ( y ) } { x - y } } , \quad x \neq y ,
$$

while its diagonal value can be chosen as is convenient. The Schur multiplier provides a canonical way to extend divided diference for scalar functions to symmetric matrix functions, as made precise by the following proposition.

Proposition 5.1 (matrix divided diference). The Schur multiplier induced by the kernelfunction $f ^ { [ 1 ] } o f a$ scalarfunction $f : \mathbb { R } \to \mathbb { R }$ gives a (symmetric) matrix divided diference, i.e.,

$$
f ( \mathbf { A } ) - f ( \mathbf { B } ) = f ^ { [ 1 ] } ( \mathbf { A } , \mathbf { B } ) \circ ( \mathbf { A } - \mathbf { B } ) \quad f o r a l l \ s y m m e t r i c \ m a t r i c e s \ \mathbf { A } , \mathbf { B } .
$$

Schur norm and factorization. Let $I , J \subset \mathbb { R }$ be two real index sets. A scalar kernel $k : I \otimes J  \mathbb { R }$ can be viewed as an infinite-dimensional generalized matrix $k ( I , J ) : = [ k ( x , y ) ] _ { x \in I , y \in J }$ . For any $I ^ { \prime } \subset I$ and $J ^ { \prime } \subset J$ such that $| I ^ { \prime } | , | J ^ { \prime } | < \infty$

$$
k ( I ^ { \prime } , J ^ { \prime } ) : = [ k ( x , y ) ] _ { x \in I ^ { \prime } , y \in J ^ { \prime } } \in \mathbb { R } ^ { | I ^ { \prime } | \times | J ^ { \prime } | }
$$

is a finite submatrix of $k ( I , J )$ , and therefore induces a Schur multiplier via

$$
\mathbf { Z } \mapsto k ( I ^ { \prime } , J ^ { \prime } ) \odot \mathbf { Z } , \quad \mathbf { Z } \in \mathbb { R } ^ { | I ^ { \prime } | \times | J ^ { \prime } | } .
$$

Recall that matrix operator norm is denoted by $\| \cdot \| .$ . The Schur norm of a matrix, denoted by $\| \cdot \| _ { \mathfrak { S } } ,$ , is the induced operator norm of the corresponding Schur multiplier:

$$
\| k ( I ^ { \prime } , J ^ { \prime } ) \| _ { \mathfrak { S } } : = \operatorname* { s u p } _ { \| \mathbf { Z } \| \leq 1 } \| k ( I ^ { \prime } , J ^ { \prime } ) \odot \mathbf { Z } \| .
$$

The Schur norm of a kernel $k : I \otimes J  \mathbb { R }$ , equivalently denoted as $k ( I , J )$ , is the supremum of the Schur norm of its finite submatrices:

$$
\| k ( I , J ) \| _ { \mathfrak { S } } : = \operatorname* { s u p } _ { I ^ { \prime } \subset I , J ^ { \prime } \subset J , \atop | I ^ { \prime } | , | J ^ { \prime } | < \infty } \| k ( I ^ { \prime } , J ^ { \prime } ) \| _ { \mathfrak { S } } .
$$

A Schurfactorization of a kernel $k ( I , J )$ is a Hilbert space H (with inner product denoted by $\langle \cdot , \cdot \rangle$ and norm denoted by $\| \cdot \| )$ and two feature maps $\mathbf { f } : I  \mathbb { H }$ and $\mathbf { g } : J  \mathbb { H }$ for which

$$
k ( x , y ) = \langle \mathbf { f } ( x ) , \mathbf { g } ( y ) \rangle \quad { \mathrm { f o r ~ a l l ~ } } x \in I { \mathrm { ~ a n d ~ } } y \in J .
$$

Specifically, any finite submatrix admits a decomposition as

$$
k ( I ^ { \prime } , J ^ { \prime } ) = \mathbf { F } \mathbf { G } ^ { \intercal } \ : \ : \ : \mathrm { w h e r e } \ : \ : \mathbf { F } : = \left[ \begin{array} { c } { \vdots } \\ { \mathbf { f } ( x ) ^ { \intercal } } \\ { \vdots } \\ { x \in I ^ { \prime } } \end{array} \right] _ { x \in I ^ { \prime } } \in \mathbb { R } ^ { | I ^ { \prime } | } \otimes \mathbb { H } , \ : \ : \mathbf { G } : = \left[ \begin{array} { c } { \vdots } \\ { \mathbf { g } ( y ) ^ { \intercal } } \\ { \vdots } \\ { \mathbf { y } \in J ^ { \prime } } \end{array} \right] _ { y \in J ^ { \prime } } \in \mathbb { R } ^ { | J ^ { \prime } | } \otimes \mathbb { H } .
$$

Proposition 5.2 (Aleksandrov and Peller (2016), Theorems 2.2.1–2.2.3). The Schur norm of a kernel $k : I \otimes J  \mathbb { R }$ is the smallest � -factorization norm, i.e.,

$$
\| k ( I , J ) \| _ { \mathfrak E } = \operatorname* { i n f } _ { \mathbb H \mathrm { ~ } \mathbf f , \mathbf g } \bigg \{ \operatorname* { s u p } _ { x \in I } \| \mathbf f ( x ) \| \cdot \operatorname* { s u p } _ { y \in J } \| \mathbf g ( y ) \| : k ( x , y ) = \langle \mathbf f ( x ) , \mathbf g ( y ) \rangle \bigg \} ,
$$

where the infimum is taken over all possible Schurfactorizations $o f k ( I , J )$

Fix the ridge filter with regularization � as the reference filter

$$
g ^ { * } ( z ) = \frac { z } { z + \lambda } .
$$

For a shrinkage filter $g ,$ , define the relative kernel $r : \mathbb { R } _ { \geq 0 } \otimes \mathbb { R } _ { \geq 0 } \to \mathbb { R }$ by

$$
r ( x , y ) : = \frac { g ( x ) - g ( y ) } { g ^ { * } ( x ) - g ^ { * } ( y ) } , \quad x \neq y .
$$

The relative kernel can be seen as the divided diference of $g$ under the metric induced by $g ^ { * }$ . The diagonal entries of � do not afect the proof but could afect its Schur norm; they can be chosen at discretion for convenience. We use the convention $r ( 0 , 0 ) = 0$

With this background in place, we are now in a position to state our main upper bound for general spectral filters. Recall that $\begin{array} { r } { \hat { \mathbf { \Xi } } : = \frac { 1 } { n } \mathbf { X } ^ { \top } \mathbf { X } } \end{array}$ is the empirical covariance and let $\begin{array} { r } { \hat { \pmb { \Sigma } } _ { \leq k } : = \frac { 1 } { n } \mathbf { \tilde { X } } _ { \leq k } ^ { \top } \mathbf { X } _ { \leq k } } \end{array}$ denote the empirical covariance of the first � coordinates.

Theorem 5.3. Under Assumption $^ { l , }$ there exist constants $c _ { 0 } , \ldots , c _ { 4 }$ that depend only on $c _ { x } , c _ { y }$ for which the following holds. Let $g : \mathbb { R } _ { \geq 0 } \to \mathbb { R }$ be a spectral filter and $\mathit { f i x } \lambda > 0 .$ . Let � be an index satisfying

$$
k \leq { \frac { n } { c _ { 3 } } } , \quad \lambda + { \frac { \sum _ { i > k } \lambda _ { i } } { n } } \geq c _ { 2 } \lambda _ { k + 1 } .
$$

Let $r$ be the relative kernel associated with $^ { g , }$ and let $H , I \subseteq \mathbb { R } _ { \geq 0 }$ be fixed measurable sets such that the following Schur norms arefinite:

$$
\begin{array} { r } { \| r ( I , H ) \| _ { \mathfrak { S } } , \| r ( H , H ) \| _ { \mathfrak { S } } , \| r ( I , \{ 0 \} ) \| _ { \mathfrak { S } } < \infty . } \end{array}\tag{5}
$$

Then with probability at least $1 - \delta - \exp ( - n / c _ { 0 } ) ,$ if

$$
\begin{array} { r } { \sigma ( \hat { \Sigma } ) \subseteq I \quad a n d \quad \sigma ( \Sigma _ { 0 : k } ) \cup \sigma ( \hat { \Sigma } _ { \leq k } ) \subseteq H \cup \{ 0 \} , } \end{array}\tag{6}
$$

it holds that

$$
\begin{array} { r l } & { \frac { 1 } { c _ { 1 } } \mathbf { B i a s } ( \hat { \mathbf { w } } _ { g } ) \leq \left( \left( 1 + \left\| r \left( I , H \right) \right\| _ { \mathbb { G } } ^ { 2 } + \left\| r \left( I , \{ 0 \} \right) \right\| _ { \Theta } ^ { 2 } \right) \left( \alpha _ { k } + \frac { \lambda _ { k + 1 } ^ { 2 } \log \left( 1 / \delta \right) } { n } \right) + \left\| r \left( H , H \right) \right\| _ { \mathbb { G } } ^ { 2 } \frac { k + \log \left( 1 / \delta \right) } { n } \lambda ^ { 2 } \right) \left\| \mathbf { w } ^ { * } \right\| _ { \Sigma _ { \mathrm { o } , k } ^ { - 1 } } ^ { 2 } } \\ & { \qquad + \left\| \left( \mathbf { I } - g ( \Sigma ) \right) \mathbf { w } ^ { * } \right\| _ { \Sigma _ { \mathrm { o } , k } } ^ { 2 } + \left( 1 + \left\| r \left( I , \{ 0 \} \right) \right\| _ { \Theta } ^ { 2 } \right) \left\| \mathbf { w } ^ { * } \right\| _ { \Sigma _ { k , \infty } } ^ { 2 } . } \end{array}
$$

Moreover,for the ridge estimator $\hat { \mathbf { w } } _ { \lambda } ^ { \mathrm { r i d g e } }$ corresponding to $g ^ { * }$

$$
\frac { 1 } { c _ { 1 } } \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g } ) \leq \| r ( I , \{ 0 \} ) \| _ { \mathfrak E } ^ { 2 } \cdot \operatorname* { m i n } \left\{ \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { \lambda } ^ { \mathrm { r i d g e } } ) , \frac { \sigma ^ { 2 } } { n } \operatorname { r a n k } ( g ( \hat { \pmb { \Sigma } } ) ) \right\} .
$$

The Schur norms (5) are invariant under the simultaneous rescaling $g ( z ) \mapsto g ( z / \tau ) , \lambda \mapsto \tau \lambda , I \mapsto \tau I .$ , and $H \mapsto \tau H$ , where $\tau > 0$ . Hence, it sufices to evaluate one representative filter for any class obtained by tuning a scale parameter, e.g., the stepsize � for GD or regularization strength � for (iterated) ridge. Our upper bounds for GD and PCR are obtained directly from the examples below, demonstrating the utility of our approach.

Example 2. The norms (5)for ridge, GD and PCR are all bounded asfollows.

$R i d g e \ g ( z ) = z / ( z + \lambda )$ : setting $I = H = \mathbb { R } _ { \geq 0 }$ , we have $\| r ( I , I ) \| \mathfrak { s } = 1$ . In particular, this holds when specializing to OLS by taking $\lambda  0$

$G D g ( z ) = 1 - ( 1 - \eta z ) ^ { t }$ : setting $\lambda = ( \eta t ) ^ { - 1 }$ and $I = H = [ 0 , 1 / \eta ]$ , we have $\| r ( I , I ) \| _ { \mathfrak { S } } \leq 3 3$

$P C R \ g ( z ) = \mathbf { 1 } \{ z > \rho \}$ : setting $\lambda = \rho$ and $I = \mathbb { R } _ { \ge 0 } , H = \left[ 1 . 1 \rho , \infty \right)$ , we have

$$
\| r ( I , H ) \| _ { \mathfrak { S } } \le 4 2 , \quad \| r ( H , H ) \| _ { \mathfrak { S } } = 0 , \quad \| r ( I , \{ 0 \} ) \| _ { \mathfrak { S } } \le 2 .
$$

For suficiently regular filters, the computation of the norms (5) can be further reduced to certain Hölder smoothness conditions. Writing $\| \cdot \| _ { C ^ { 1 , \beta } }$ for the usual $C ^ { 1 , \beta }$ Hölder norm on $\mathbb { R } _ { \geq 0 }$ , we obtain the following corollary.

Corollary 5.4. There exist constants $c _ { 0 } , \ldots , c _ { 4 } > 0 ,$ , depending only on $c _ { x } ,$ for which the following holds. Let $g : \mathbb { R } _ { \geq 0 } \to \mathbb { R }$ be a spectral filter and $\ f i x \lambda > 0$ and $\beta \in ( 0 , 1 ]$ . Suppose the following quantities are finite:

$$
C _ { \beta } : = \operatorname* { m a x } _ { j \in \{ 0 , 1 , 2 \} } \frac { 1 } { \beta } \| z \mapsto z ^ { j } ( 1 - g ( \lambda z ) ) \| _ { C ^ { 1 , \beta } } , \quad C _ { \psi } : = \operatorname* { s u p } _ { z \geq 0 } ( z + \lambda ) | \psi ( z ) | .
$$

Let

$$
k ^ { * } : = \operatorname* { m i n } \left\{ k : \lambda + { \frac { \sum _ { i > k } \lambda _ { i } } { n } } \geq c _ { 2 } \lambda _ { k + 1 } \right\} .
$$

Then under Assumption 1, if additionally $k ^ { * } \leq n / c _ { 3 }$ , then with probability at least $1 - \delta - \exp ( - n / c _ { 0 } )$

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { c _ { 1 } } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) \leq \operatorname* { m a x } \{ C _ { \beta } , C _ { \psi } , 1 \} ^ { 2 } \left( \alpha _ { k ^ { \ast } } + \frac { k ^ { \ast } + \log ( 1 / \delta ) } { n } \lambda ^ { 2 } \right) \| \mathbf { w } ^ { \ast } \| _ { \Sigma _ { 0 ; k ^ { \ast } } ^ { - 1 } } ^ { 2 } + \operatorname* { m a x } \{ C _ { \psi } , 1 \} ^ { 2 } \| \mathbf { w } ^ { \ast } \| _ { \Sigma _ { k ^ { \ast } ; \infty } } ^ { 2 } } \\ & { \qquad + \left\| ( \mathbf { I } - g ( \Sigma ) ) \mathbf { w } ^ { \ast } \right\| _ { \Sigma _ { 0 ; k ^ { \ast } } } ^ { 2 } . } \end{array}
$$

Proof roadmap. We provide a high-level overview of how the theory of Schur multipliers is applied. Denote the full and truncated Gram matrices by $\mathbf { A } : = \mathbf { X } \mathbf { X } ^ { \top }$ and $\mathbf { A } _ { \leq k } : = \mathbf { X } _ { \leq k } \mathbf { X } _ { \leq k } ^ { \top }$ . The main challenge in the proof is to study perturbations in the form of a leave-tail-out diference,

$$
f ( \mathbf { A } ) - f ( \mathbf { A } _ { \leq k } ) ,\tag{7}
$$

which is dificult as A and $\mathbf { A } _ { \leq k }$ generally do not commute. However, the introduced machinery makes this possible:

• The matrix divided diference (Proposition 5.1) converts (7) to studying the action of the Schur multiplier $k ( \mathbf { A } , \mathbf { A } _ { \leq k } )$ on the tail matrix $\mathbf { A } _ { > k } : = \mathbf { A } - \mathbf { A } _ { \leq k }$ for $k = f ^ { [ 1 ] }$

• The Schur multiplier $k ( \mathbf { A } , \mathbf { A } _ { \leq k } )$ is equivariant to a standard Schur multiplier in the rotated space,

$$
\mathbf { Z } \mapsto k ( I , J ) \odot \mathbf { Z } ,
$$

where � and � consist of the spectra of A and $\mathbf { A } _ { \leq k }$ , respectively. So studying the action of $k ( \mathbf { A } , \mathbf { A } _ { \leq k } )$ on $\mathbf { A } _ { > k }$ reduces to controlling the scalar kernel $k ( I , J )$ , with one complication that the rotation correlates with $\mathbf { A } _ { > k }$

• This final issue is addressed by Schur factorization, which allows us to fix a factorization of the whole kernel $k ( \cdot , \cdot )$ independent of $\mathbf { A } , \mathbf { A } _ { \leq k }$ first, then apply the factorization to $k ( I , J )$ by taking appropriate submatrices indexed by eigenvalues. This operation decouples the correlation.

A similar argument is used to bound head perturbations involving $\pmb { \Sigma } _ { 0 : k }$ and $\hat { \Sigma } _ { \le k }$ . This two-scale analysis allows us to cover even discontinuous filters such as PCR, and ultimately obtain sharp instance-wise bounds.

## 5.2 General lower bound and applications

As discussed in Section 4, the PCR lower bound is not always tight due to the presence of two distinct critical indices. However, we can obtain a sharper ‘single index’ lower bound for $\hat { \mathbf { w } } _ { g }$ as long as the filter � does not saturate too quickly; the statement is presented below. Both results are a corollary of a more general lower bound for monotone filters, which we present in Theorem F.1 in the appendix.

Corollary 5.5. Under Assumption 1 with Gaussian design $\mathbf { x } \sim { \mathcal { N } } ( 0 , \pmb { \Sigma } )$ , there exist constants $c _ { 0 } , \ldots , c _ { 3 } > 0$ such that thefollowing holds. Let $g : \mathbb { R } _ { \geq 0 }  [ 0 , 1 ]$ be a monotone shrinkagefilter,fix $\lambda > 0$ and let sample size $n \geq c _ { 0 }$ . Suppose that

$$
C _ { g } : = \operatorname* { i n f } _ { z \ge 0 } \operatorname* { m a x } \left\{ 1 - g ( 1 . 1 z ) , g ( 0 . 9 z ) - g ( \lambda ) \right\}
$$

is positive. Let

$$
k ^ { * } : = \operatorname* { m i n } \left\{ k : \lambda + { \frac { \sum _ { i > k } \lambda _ { i } } { n } } \geq c _ { 2 } \lambda _ { k + 1 } \right\} .
$$

If additionally $k ^ { * } \leq n / c _ { 3 }$ , then

$$
\begin{array} { r } { c _ { 1 } \mathbb { B } \mathbf { B } \mathbf { i a s } ( \hat { \mathbf { w } } _ { g } ) \geq C _ { g } ^ { 2 } \alpha _ { k ^ { * } } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } + ( 1 - g ( \lambda ) ) ^ { 2 } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { k ^ { * } ; \infty } } ^ { 2 } + \big \| ( \mathbf { I } - g ( 1 . 1 \Sigma ) ) \mathbf { w } ^ { * } \big \| _ { \Sigma } ^ { 2 } . } \end{array}
$$

$f g$ is concave, the last term may be replaced by $\left\| ( \mathbf { I } - g ( \pmb { \Sigma } ) ) \mathbf { w } ^ { * } \right\| _ { \pmb { \Sigma } } ^ { 2 } .$

Unlike Theorem 5.3 which only requires Assumption 1, the lower bounds require both Gaussian design and monotone shrinkage �. Note that the tail coeficient $( 1 - g ( \lambda ) ) ^ { 2 }$ is always at least as large as $C _ { g } ^ { 2 } .$ . Comparing Theorem 5.3 and Corollary 5.5, we see that whenever the Schur norms (5) (or $C _ { \beta } , C _ { \psi }$ in Corollary 5.4) are $O ( 1 )$ and $C _ { g } = \Omega ( 1 )$ for the same threshold � with $g ( \lambda ) < 1$ , the upper and lower bias bounds match up to the following diferences.

• The $( k ^ { * } / n ) \lambda ^ { 2 }$ factor in the head term; this is due to head concentration, and can typically be absorbed into variance under bounded SNR (see the proof of Corollary 5.6).

• The 1.1 spectral shift in the residual term; this constant can be made arbitrarily close to 1, and disappears when $g$ is concave.

For GD and ridge $( r = 1 )$ , both conditions are satisfied and we recover the known tight bounds (Corollary 3.1 and Wu et al. (2026), Proposition 2.1), albeit under the more restrictive Gaussian design. We can also obtain analogous tight bounds for gradient flow, as well as bounds for power-exponential filters such as Gaussian type filters (Calvetti et al., 1999; Calvetti and Reichel, 2002); we omit these for brevity.

We conclude our study with an application.

Application: iterated Tikhonov. The iterated Tikhonov (iterated ridge) estimator $\hat { \mathbf { w } } _ { \lambda , p } ^ { \mathrm { r i d g e } }$ (Riley, 1955; King and Chillingworth, 1979) is given by the filter

$$
g _ { \lambda , p } ( z ) : = 1 - \left( { \frac { \lambda } { z + \lambda } } \right) ^ { p } , \quad \lambda > 0 , \quad p \ge 1 .
$$

When $p$ is an integer, it is equivalently obtained by the regularized iterative update

$$
\hat { \mathbf { w } } _ { \lambda , p } ^ { \mathrm { r i d g e } } : = \mathbf { w } _ { p } , \quad \mathrm { w h e r e } ~ \mathbf { w } _ { s } = \left\{ \begin{array} { l l } { \underset { \mathbf { w } } { \operatorname { a r g m i n } } \frac { 1 } { n } \Vert \mathbf { y } - \mathbf { X } \mathbf { w } \Vert ^ { 2 } + \lambda \Vert \mathbf { w } - \mathbf { w } _ { s - 1 } \Vert ^ { 2 } } & { s \geq 1 , } \\ { 0 } & { s = 0 . } \end{array} \right.
$$

We prove matching upper and lower risk bounds for iterated Tikhonov, generalizing the known bounds for ridge (Wu et al., 2026). In particular, the bounds are uniform with respect to both � and $p$

Corollary 5.6 (Iterated Tikhonov risk under bounded SNR). There exist constants $c _ { 0 } , \ldots , c _ { 3 } > 1$ that depend only on $c _ { x } , c _ { y }$ for which thefollowing hold over $\mathbb { L } _ { b }$ . Let $\hat { \mathbf { w } } _ { \lambda , p } ^ { \mathrm { r i d g e } }$ be given by thefilter $g _ { \lambda , p }$ with $\lambda > 0 , p \ge 1$ . Let

$$
k ^ { * } : = \operatorname* { m i n } \left\{ k : \frac { \lambda } { p } + \frac { \sum _ { i > k } \lambda _ { i } } { n } \geq c _ { 2 } \lambda _ { k + 1 } \right\} , \quad \tilde { \lambda } : = \frac { \lambda } { p } + \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } , \quad D : = k ^ { * } + \frac { 1 } { \tilde { \lambda } ^ { 2 } } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } ,
$$

and additionally assume $n \geq c _ { 0 }$ and $k ^ { * } \leq n / c _ { 3 }$

• Upper bound. With probability at least $1 - \exp ( - k ^ { * } / c _ { 0 } )$ over sampling ofX, it holds that

$$
\frac { 1 } { c _ { 1 } } \mathbb { E } [ \mathcal { E } ( \hat { \mathbf { w } } _ { \lambda , p } ^ { \mathrm { r i d e e } } ) | \mathbf { X } ] \leq \sum _ { i \leq k ^ { * } } \lambda _ { i } \left( \frac { \lambda } { \lambda _ { i } + \lambda } \right) ^ { 2 p } \mathbf { w } _ { i } ^ { * 2 } + \left( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) ^ { 2 } \left. \mathbf { w } ^ { * } \right. _ { \mathbb { Z } _ { 0 , k ^ { * } } ^ { - 1 } } ^ { 2 } + \left. \mathbf { w } ^ { * } \right. _ { \mathbb { Z } _ { k ^ { * } ; \infty } ^ { 2 } } ^ { 2 } + \left( 1 + b \right) \sigma ^ { 2 } \frac { D } { n } .
$$

• Lower bound. If additionally $\mathbf { x } \sim { \mathcal { N } } ( 0 , \pmb { \Sigma } )$ , then in expectation,

$$
c _ { 1 } \mathbb { E } \mathcal { E } ( \hat { \mathbf { w } } _ { \lambda , p } ^ { \mathrm { r i d g e } } ) \geq \sum _ { i \leq k ^ { * } } \lambda _ { i } \left( \frac { \lambda } { \lambda _ { i } + \lambda } \right) ^ { 2 p } \mathbf { w } _ { i } ^ { * 2 } + \left( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) ^ { 2 } \left. \mathbf { w } ^ { * } \right. _ { \Sigma _ { 0 ; k ^ { * } } ^ { - 1 } } ^ { 2 } + \left. \mathbf { w } ^ { * } \right. _ { \Sigma _ { k ^ { * } ; \infty } } ^ { 2 } + \sigma ^ { 2 } \frac { D } { n } .
$$

## 6 Related Works

Our analyses build upon the foundational works of Bartlett et al. (2020); Tsigler and Bartlett (2023) on benign overfitting of OLS and ridge, as well as analyses of spectral regularization (e.g., Engl et al., 1996; De Vito et al., 2005a; Bauer et al., 2007); we refer the reader to these works for more classical references. Below, we discuss works most relevant to our setting.

Dominance in linear regression. For linear regression with fixed design, Dhillon et al. (2013) showed that the risk of PCR with spectral threshold � is at most 4 times that of ridge regression with the same regularization strength �, while it can be arbitrarily smaller. Thus, PCR strongly dominates ridge in our ratewise sense. Similarly, Ali et al. (2019) showed that the risk of early-stopped gradient flow at time $t = 1 / \lambda$ is at most 1.69 times that of ridge, assuming an isotropic prior on the optimal parameter $\mathbf { w } ^ { * }$ . In the random design setting, under mild regularity assumptions (Assumption 1), Wu et al. (2026) proved that early-stopped GD with fixed stepsize at time $\eta t = 1 / \lambda$ strongly dominates ridge. On the other hand, GD is incomparable to online SGD, while GD strongly dominates the latter when restricted to problems with fast and continuously decaying spectra, such as power law decay. This was shown by proving upper bounds for the excess risk of GD and comparing with the known essentially tight bounds for ridge (Tsigler and Bartlett, 2023) and SGD (Wu et al., 2022b).

Risk bounds for GD. The previous strongest known instance-wise bounds on the risk of GD were given in Wu et al. (2026). Their upper bound sharply characterizes the tail regularization, tail and variance terms, mirroring the risk analysis of ridge (Tsigler and Bartlett, 2023). Compared to Theorem B.1, only the optimization error term is missing (Theorem 3.1) or suboptimal (Theorem 4.3); see Section 3 for a detailed comparison. A similar upper bound was previously obtained in Zou et al. (2022) by additionally assuming an isotropic prior on $\mathbf { w } ^ { * }$ Other works obtained bounds with coarser signal dependence (Raskutti et al., 2014; Kuzborskij et al., 2021; Xu et al., 2023), or analyzed GD under various source and capacity conditions (Yao et al., 2007; Lin and Rosasco, 2017; Dicker et al., 2017; Blanchard and Mücke, 2018; Lin et al., 2025); see also Section 5 of Wu et al. (2026). Risk bounds for analytic spectral filters assuming power law decay, including GD and iterated ridge, were given in Li et al. (2026b).

Risk bounds for PCR. The work most relevant to ours is Hucker and Wahl (2023). Hucker and Wahl (2023) explain the implicit regularization efect of PCR by a comparison to an “oracle” PCR, which replaces the empirical principal components with the population versions, and obtain instance-wise upper bounds for the PCR risk in the same setting as Theorem 4.1. However, their bounds are looser compared to ours, and they do not provide lower bounds. See Section 4 for a detailed comparison. Risk bounds for a debiased variant of PCR were also recently given in Zhao (2026), however they require very strong assumptions on the signal and spectrum at the selected index �, such as zero tail signal, bounded head condition number, and eigengap and small tail mass conditions; see the conditions of their Theorem 4.4. In contrast, our analysis applies to every instance $( \Sigma , \mathbf { w } ^ { * } )$ .

Bing et al. (2021) obtained coarser risk bounds in terms of ∥�∥, ∥w<sup>∗</sup>∥ and studied data-adaptive threshold selection. Huang et al. (2022) proved upper and lower bounds for PCR in terms of the largest, smallest and threshold eigenvalues using general concentration arguments, and provided tighter bounds under spectral gap assumptions. Lu and Pereverzev (2013); Dicker et al. (2017); Blanchard and Mücke (2018) studied PCR under source and capacity conditions and established minimax rates, see also Hall and Horowitz (2007); Brunel et al. (2016) for the functional perspective. Asymptotic analyses based on random matrix theory techniques were given in e.g., Gedon et al. (2024); Green and Romanov (2025) for PCR and Advani et al. (2020) for gradient flow; see the works cited in their introduction for related approaches. Nonetheless, these results do not yield the fine-grained component-wise sample complexity bounds required to study dominance.

Comparison with Li et al. (2026a). In a concurrent work, Li et al. (2026a) studied the prediction risk of a class of spectral estimators and the corresponding Gaussian sequence estimators. They established conditions under which the two are asymptotically equivalent, that is,

$$
\begin{array} { r l r } { \mathbb { E } \mathcal { E } ( \hat { { \bf w } } _ { g } ) } & { \stackrel { ? } { = } } & { ( 1 + o _ { \mathbb { P } } ( 1 ) ) \left( \left\| ( { \bf I } - g ( \Sigma ) ) { \bf w } ^ { * } \right\| _ { \Sigma } ^ { 2 } + \displaystyle \frac { \sigma ^ { 2 } } { n } \sum _ { i } g ( \lambda _ { i } ) ^ { 2 } \right) . } \end{array}
$$

In particular, their Appendix C utilizes Schur multipliers (double operator integrals) on Schatten classes to control matrix diferences, which is closely related to our Schur multiplier approach developed in Section 5.1.

However, they apply this technique to directly control diferences of $\Sigma , { \hat { \Sigma } } ,$ , which additionally requires various restrictive conditions on the signal, spectrum, sample size and filter (their Assumption 3) to conclude even a constant-order version of the above equivalence. They also require the filter to satisfy a rescaled Lipschitz property, which includes GD and ridge but excludes PCR.

In contrast, we apply this technique to $\mathbf { A } , \mathbf { A } _ { \leq k }$ to analyze the ‘leave-tail-out’ perturbation in sample space, as well as to $\pmb { \Sigma } _ { \le k } , \hat { \pmb { \Sigma } } _ { \le k }$ to separately control the head perturbation. In the process, we develop a decoupling argument via Schur factorization to handle the dependence between the perturbation and full Gram matrix. This allows us to study even discontinuous filters such as PCR by isolating the head, and to retain a precise characterization of finite-sample efects such as tail regularization for all problem instances.

We remark that contour integration can also be used to control noncommutative perturbations, which has been used for instance by Li et al. (2026b) to study spectral regularization methods under power-law decay. However, this requires the filter to be analytic.

## 7 Conclusion

In this paper, we establish that suitably tuned PCR dominates all monotone filters, and therefore is instance-wise optimal for linear regression with Gaussian random design. In particular, GD is inadmissible, and PCR can achieve polynomially smaller risk than GD. We also prove tight risk bounds for GD, characterizing the efects of optimization error, implicit bias, and variance. These results demonstrate that instance-wise comparisons reveal substantial diferences between methods that worst-case analyses do not capture. From an algorithmic perspective, this separation also highlights the variance benefit of discarding components with small eigenvalues, and suggests investigating whether similar control of weak directions can improve the statistical performance of algorithms for wider noisy estimation problems. Limitations of our work include the assumption of Gaussian design in the dominance results and general lower bounds; these may be removable with more careful analysis, as they are not required for the GD lower bound. Also, the variance bounds for PCR can likely be improved.

## Acknowledgements

JK, HF and JDL acknowledge support of NSF IIS 2107304, NSF CCF 2212262, NSF CAREER Award 2540142, NSF 2546544, NSF CCF 2019844 and ONR N00014-24-1-2639. This material is based upon work supported by the U.S. National Science Foundation under Cooperative Agreement No. 2433450.

Disclosure of AI usage. Parts of key technical ingredients were obtained from iterative conversations with GPT 5.5 and 5.6 Pro, including a proof of the PCR upper bound, an analytic version of the general upper bound, and a leave-one-out lemma used in the lower bound. AI was not used in the writing of the final draft. The authors rederived and verified all proofs and take full responsibility for the content of this paper.

## References

Madhu S. Advani, Andrew M. Saxe, and Haim Sompolinsky. High-dimensional dynamics of generalization error in neural networks. Neural Networks, 132:428–446, 2020.

A. B. Aleksandrov and V. V. Peller. Operator Lipschitz functions. Russian Mathematical Surveys, 71(4): 605–702, 2016.

Alnur Ali, J. Zico Kolter, and Ryan J. Tibshirani. A continuous-time view of early stopping for least squares regression. In Kamalika Chaudhuri and Masashi Sugiyama, editors, Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, volume 89 of Proceedings ofMachine Learning Research, pages 1370–1378. PMLR, 2019.

Peter L. Bartlett, Philip M. Long, Gábor Lugosi, and Alexander Tsigler. Benign overfitting in linear regression. Proceedings ofthe National Academy ofSciences, 117(48):30063–30070, 2020.

Frank Bauer, Sergei Pereverzev, and Lorenzo Rosasco. On regularization algorithms in learning theory. Journal ofComplexity, 23(1):52–72, 2007.

James O. Berger. Statistical decision theory and Bayesian analysis. Springer Series in Statistics. Springer-Verlag, New York, second edition, 1985.

Xin Bing, Florentina Bunea, Seth Strimas-Mackey, and Marten Wegkamp. Prediction under latent factor regression: adaptive PCR, interpolating predictors and beyond. Journal of Machine Learning Research, 22 (177):1–50, 2021.

David Blackwell. Conditional expectation and unbiased sequential estimation. The Annals ofMathematical Statistics, pages 105–110, 1947.

Gilles Blanchard and Nicole Mücke. Optimal rates for regularization of statistical inverse learning problems. Foundations of Computational Mathematics, 18(4):971–1013, 2018.

Élodie Brunel, André Mas, and Angelina Roche. Non-asymptotic adaptive prediction in functional linear models. Journal ofMultivariate Analysis, 143:208–232, 2016.

Daniela Calvetti and Lothar Reichel. Lanczos-based exponential filtering for discrete ill-posed problems. Numerical Algorithms, 29(1–3):45–65, 2002.

Daniela Calvetti, Lothar Reichel, and Qin Zhang. Iterative exponential filtering for large discrete ill-posed problems. Numerische Mathematik, 83(4):535–556, 1999.

Marine Carrasco, Jean-Pierre Florens, and Eric Renault. Linear inverse problems in structural econometrics: estimation based on spectral decomposition and regularization. In James J. Heckman and Edward E. Leamer, editors, Handbook ofEconometrics, volume 6B, chapter 77, pages 5633–5751. Elsevier, 2007.

Ernesto De Vito, Lorenzo Rosasco, Andrea Caponnetto, Umberto De Giovannini, and Francesca Odone. Learning from examples as an inverse problem. Journal of Machine Learning Research, 6(30):883–904, 2005a.

Ernesto De Vito, Lorenzo Rosasco, and Alessandro Verri. Spectral methods for regularization in learning theory. Technical Report DISI-TR-05-18, Dipartimento di Informatica e Scienze dell’Informazione (DISI), Università di Genova, Genova, Italy, 2005b.

Paramveer S. Dhillon, Dean P. Foster, Sham M. Kakade, and Lyle H. Ungar. A risk comparison of ordinary least squares vs ridge regression. Journal ofMachine Learning Research, 14(46):1505–1511, 2013.

Lee H. Dicker, Dean P. Foster, and Daniel Hsu. Kernel ridge vs. principal component regression: minimax bounds and the qualification of regularization operators. Electronic Journal ofStatistics, 11(1):1022–1047, 2017.

Heinz W. Engl, Martin Hanke, and Andreas Neubauer. Regularization of inverse problems, volume 375 of Mathematics and Its Applications. Kluwer Academic Publishers, Dordrecht, 1996.

Daniel Gedon, Antonio H. Ribeiro, and Thomas B. Schön. No double descent in principal component regression: a high-dimensional analysis. In Proceedings ofthe 41st International Conference on Machine Learning, 2024.

L. Gerfo, Lorenzo Rosasco, Francesca Odone, Ernesto De Vito, and Alessandro Verri. Spectral algorithms for supervised learning. Neural Computation, 20:1873–1897, 07 2008.

Yuri Golubev. On universal oracle inequalities related to high-dimensional linear models. The Annals of Statistics, 38(5):2751–2780, 2010.

Alden Green and Elad Romanov. The high-dimensional asymptotics of principal component regression. The Annals ofStatistics, 53(4):1697–1727, 2025.

Peter Hall and Joel L. Horowitz. Methodology and convergence rates for functional linear regression. The Annals ofStatistics, 35(1):70–91, 2007.

Ningyuan (Teresa) Huang, David W. Hogg, and Soledad Villar. Dimensionality reduction, regularization, and generalization in overparameterized regressions. SIAM Journal on Mathematics ofData Science, 4(1): 126–152, 2022.

Laura Hucker and Martin Wahl. A note on the prediction error of principal component regression in high dimensions. Theory ofProbability and Mathematical Statistics, 109:37–53, 2023.

W. James and Charles Stein. Estimation with quadratic loss. In Jerzy Neyman, editor, Proceedings ofthe Fourth Berkeley Symposium on Mathematical Statistics and Probability, volume 1, pages 361–379. University of California Press, 1961.

J. Thomas King and D. Chillingworth. Approximation of generalized inverses by iterated regularization. Numerical Functional Analysis and Optimization, 1(5):499–513, 1979.

Alois Kneip. Ordered linear smoothers. The Annals ofStatistics, 22(2):835–866, 1994.

Vladimir Koltchinskii and Karim Lounici. Concentration inequalities and moment bounds for sample covariance operators. Bernoulli, 23(1):110–133, 2017.

Ilja Kuzborskij, Csaba Szepesvári, Omar Rivasplata, Amal Rannen-Triki, and Razvan Pascanu. On the role of optimization in double descent: a least squares study. In Advances in Neural Information Processing Systems, 2021.

Yicheng Li, Yuqian Cheng, Zhuo Chen, and Qian Lin. Risk equivalence between RKHS regression and sequence models for Lipschitz spectral algorithms. arXiv preprint arXiv:2609.08817, 2026a.

Yicheng Li, Weiye Gan, Zuoqiang Shi, and Qian Lin. Generalization error curves for analytic spectral algorithms under power-law decay. arXiv preprint arXiv:2401.01599v4, 2026b.

Junhong Lin and Lorenzo Rosasco. Optimal rates for multi-pass stochastic gradient methods. Journal of Machine Learning Research, 18(97):1–47, 2017.

Licong Lin, Jingfeng Wu, and Peter L. Bartlett. Improved scaling laws in linear regression via data reuse. In Advances in Neural Information Processing Systems, 2025.

Shuai Lu and Sergei V. Pereverzev. Regularization theory for ill-posed problems: selected topics, volume 58 of Inverse and Ill-Posed Problems Series. De Gruyter, Berlin and Boston, 2013.

C Radhakrishna Rao et al. Information and the accuracy attainable in the estimation of statistical parameters. Bull. Calcutta Math. Soc, 37(3):81–91, 1945.

Garvesh Raskutti, Martin J. Wainwright, and Bin Yu. Early stopping and non-parametric regression: an optimal data-dependent stopping rule. Journal ofMachine Learning Research, 15(11):335–366, 2014.

James D. Riley. Solving systems of linear equations with a positive definite, symmetric, but possibly ill-conditioned matrix. Mathematical Tables and Other Aids to Computation, 9(51):96–101, 1955.

Hans Triebel. Theory offunction spaces, volume 78 of Monographs in Mathematics. Birkhäuser, Basel, 1983.

Alexander Tsigler and Peter L. Bartlett. Benign overfitting in ridge regression. Journal of Machine Learning Research, 24(123):1–76, 2023.

Roman Vershynin. High-dimensionalprobability: an introduction with applications in data science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, second edition, 2026.

Martin J Wainwright. High-dimensional statistics: A non-asymptotic viewpoint, volume 48. Cambridge university press, 2019.

Abraham Wald. Statistical decisionfunctions. John Wiley & Sons, New York, 1950.

Jingfeng Wu, Difan Zou, Vladimir Braverman, Quanquan Gu, and Sham M. Kakade. Last iterate risk bounds of SGD with decaying stepsize for overparameterized linear regression. In Proceedings ofthe 39th International Conference on Machine Learning, 2022a.

Jingfeng Wu, Difan Zou, Vladimir Braverman, Quanquan Gu, and Sham M. Kakade. The power and limitation of pretraining-finetuning for linear regression under covariate shift. In Advances in Neural Information Processing Systems, 2022b.

Jingfeng Wu, Peter L. Bartlett, Sham M. Kakade, Jason D. Lee, and Bin Yu. Risk comparisons in linear regression: implicit regularization dominates explicit regularization. In Proceedings of Thirty Ninth Conference on Learning Theory, 2026.

Jing Xu, Jiaye Teng, Yang Yuan, and Andrew Chi-Chih Yao. Towards data-algorithm dependent generalization: a case study on overparameterized linear regression. In Advances in Neural Information Processing Systems, 2023.

Yuan Yao, Lorenzo Rosasco, and Andrea Caponnetto. On early stopping in gradient descent learning. Constructive Approximation, 26(2):289–315, 2007.

Peng Zhao. De-floored principal component regression: when rank selection alone is insuficient for prediction. arXiv preprint arXiv:2607.16638, 2026.

Difan Zou, Jingfeng Wu, Vladimir Braverman, Quanquan Gu, and Sham M. Kakade. Risk bounds of multi-pass SGD for least squares in the interpolation regime. In Advances in Neural Information Processing Systems, 2022.

Difan Zou, Jingfeng Wu, Vladimir Braverman, Quanquan Gu, and Sham M. Kakade. Benign overfitting of constant-stepsize SGD for linear regression. Journal ofMachine Learning Research, 24(326):1–58, 2023.

## A Notation and basic concentration lemmas

Without loss of generality, assume � is diagonal throughout the appendices. For an index �, we define the following notation, following or adapted from the convention from Wu et al. (2026); Tsigler and Bartlett (2023); Zou et al. (2023):

$$
\begin{array} { r l r } & { } & { \pmb { \Sigma } = [ \pmb { \Sigma } _ { \leq k } ^ { }  \qquad \mathbf { \sigma } _ { \pmb { \Sigma } > k } ^ { \ast } ] , \quad \mathbf { w } ^ { \ast } = [ \pmb { \mathsf { w } } _ { \leq k } ^ { \ast } ] , \quad \mathbf { X } = [ \mathbf { X } _ { \leq k } \quad \mathbf { X } _ { > k } ] , } \\ & { } & { \quad \quad \mathbf { A } : = \mathbf { X } \mathbf { X } ^ { \top } , \quad \mathbf { A } _ { \leq k } : = \mathbf { X } _ { \leq k } \mathbf { X } _ { \leq k } ^ { \top } , \quad \mathbf { A } _ { > k } : = \mathbf { X } _ { > k } \mathbf { X } _ { > k } ^ { \top } . } \end{array}
$$

The empirical covariance and its head block are defined as

$$
\hat { \boldsymbol { \Sigma } } : = \frac { 1 } { n } \mathbf { X } ^ { \top } \mathbf { X } , \quad \hat { \boldsymbol { \Sigma } } _ { \leq k } : = \frac { 1 } { n } \mathbf { X } _ { \leq k } ^ { \top } \mathbf { X } _ { \leq k } .
$$

Additionally, we use the notation

$$
\tilde { \alpha } _ { k } : = \left( \frac { \sum _ { i > k } \lambda _ { i } } { n } \right) ^ { 2 } + \frac { \sum _ { i > k } \lambda _ { i } ^ { 2 } } { n } + \frac { \lambda _ { k + 1 } ^ { 2 } \log ( 1 / \delta ) } { n } .\tag{8}
$$

Lemma A.1 (typical events). Under Assumption 1A, there are constants $c _ { 0 } , c _ { 1 } , c _ { 3 } , c _ { 4 } > 1$ that depend on $c _ { x }$ for which thefollowing holds. With probability at least $1 - \exp ( - n / c _ { 0 } )$ , we have

$$
\frac 1 2 \Sigma _ { \leq k } \preceq \hat { \Sigma } _ { \leq k } \preceq 2 \Sigma _ { \leq k } \quad f o r k \leq n / c _ { 3 } ,
$$

$$
\begin{array} { r } { \left\| \mathbf { X } _ { > k } \mathbf { w } _ { > k } ^ { * } \right\| ^ { 2 } \leq c _ { 1 } n \Big \| \mathbf { w } _ { > k } ^ { * } \Big \| _ { \Sigma _ { > k } } ^ { 2 } , } \end{array}
$$

$$
\frac { 2 } { 3 } \sum _ { i > k } \lambda _ { i } - c _ { 1 } n \lambda _ { k + 1 } \leq \lambda _ { \operatorname* { m i n } } ( \mathbf A _ { > k } ) \leq \lambda _ { \operatorname* { m a x } } ( \mathbf A _ { > k } ) \leq \frac { 4 } { 3 } \sum _ { i > k } \lambda _ { i } + c _ { 1 } n \lambda _ { k + 1 } ,
$$

$$
\| \hat { \boldsymbol { \Sigma } } \| \leq \frac { c _ { 4 } } { 2 } \left( \lambda _ { 1 } + \frac { \operatorname { t r } ( \boldsymbol { \Sigma } ) } { n } \right) .
$$

In particular, thefirst claim can be tightened to: with probability at least $1 - \delta ,$

$$
\Big \| \Sigma _ { \leq k } ^ { - 1 / 2 } ( \widehat { \Sigma } _ { \leq k } - \Sigma _ { \leq k } ) \Sigma _ { \leq k } ^ { - 1 / 2 } \Big \| \leq c _ { 1 } \left( \sqrt { \frac { k + \log ( 1 / \delta ) } { n } } + \frac { k + \log ( 1 / \delta ) } { n } \right) .
$$

ProofofLemma A.1. These are applications of standard matrix concentration bounds (see, e.g., Vershynin, 2026): the first two claims appear in Wu et al. (2026, Lemma B.4) and the third claim appears in Bartlett et al. (2020, Lemma B.9). For the last two claims, Koltchinskii and Lounici (2017) showed that with probability at least $1 - \exp ( - n / c _ { 0 } )$ , we have

$$
\left\| { \hat { \boldsymbol { \Sigma } } } - { \boldsymbol { \Sigma } } \right\| \leq c \| { \boldsymbol { \Sigma } } \| \left( { \sqrt { \frac { \operatorname { t r } ( { \boldsymbol { \Sigma } } ) } { \| { \boldsymbol { \Sigma } } \| n } } } + { \frac { \operatorname { t r } ( { \boldsymbol { \Sigma } } ) } { \| { \boldsymbol { \Sigma } } \| n } } + 1 \right) \leq c ^ { \prime } \left( \| { \boldsymbol { \Sigma } } \| + { \frac { \operatorname { t r } ( { \boldsymbol { \Sigma } } ) } { n } } \right)
$$

for some constants $c _ { 0 } , c , c ^ { \prime } \geq 1 ;$ ; then the fourth claim follows, and the last claim follows by applying the above to $\pmb { \Sigma } ^ { - 1 / 2 } \mathbf { X }$ □

Lemma A.2 (tail regularization). Under Assumption 1A, there are constants $c _ { 0 } , c _ { 2 } > 1$ that only depend on $c _ { x } ^ { 2 }$ for which the following holds. For each $\lambda \geq 0$ and � such that

$$
\lambda + \frac { \sum _ { i > k } \lambda _ { i } } { n } \geq c _ { 2 } \lambda _ { k + 1 } ,
$$

with probability at least $1 - \exp ( - n / c _ { 0 } )$ it holds that

$$
\frac { 1 } { 2 } \left( \boldsymbol \lambda + \frac { \sum _ { i > k } \lambda _ { i } } { n } \right) \mathbf { I } \preceq \frac { 1 } { n } \mathbf { X } _ { > k } \mathbf { X } _ { > k } ^ { \top } + \boldsymbol \lambda \mathbf { I } \preceq 2 \left( \boldsymbol \lambda + \frac { \sum _ { i > k } \lambda _ { i } } { n } \right) \mathbf { I } .
$$

ProofofLemma A.2. This is Bartlett et al. (2020, Lemma 9) or Tsigler and Bartlett (2023, Lemma 3); from their proof, it is clear that, by choosing a large enough $c _ { 2 } .$ , the constant factors in the upper and lower bounds can be made 2 and $1 / 2 ,$ , respectively. □

Lemma A.3 (tail deviation). Under Assumption 1A, there are constants $c _ { 0 } , c _ { 1 } > 1$ that only depend on $c _ { x } f o r$ which the following holds. Let u be a fixed unit vector of an appropriate dimension.

• with probability at least $1 - \delta$ for any $\delta > 0 ,$ , it holds that

$$
\left\| \boldsymbol { \Sigma } _ { > k } ^ { 1 / 2 } \mathbf { X } _ { > k } ^ { \top } \mathbf { u } \right\| \leq c _ { 1 } \sqrt { \lambda _ { k + 1 } ^ { 2 } \log ( 1 / \delta ) + \sum _ { i > k } \lambda _ { i } ^ { 2 } } .
$$

• The $L ^ { p }$ -norm of $\mathbf { A } _ { > k } \mathbf { u }$ is bounded by

$$
\| \mathbf { A } _ { > k } \mathbf { u } \| _ { L ^ { p } } : = \left( \mathbb { B } \| \mathbf { A } _ { > k } \mathbf { u } \| ^ { p } \right) ^ { 1 / p } \leq c _ { 1 } \left( p \lambda _ { k + 1 } + \sum _ { i > k } \lambda _ { i } + \sqrt { \operatorname* { m a x } \{ n , p \} } \left( p \lambda _ { k + 1 } ^ { 2 } + \sum _ { i > k } \lambda _ { i } ^ { 2 } \right) \right) , \quad p \geq 4 .\tag{9}
$$

Proof of Lemma A.3. The first claim is a standard application of the Bernstein inequality (see, e.g., Bartlett et al., 2020, proof of Lemma B.9); the second claim is from Wu et al. (2026, Lemma E.2). □

## B GD Risk Bounds

In this section, we prove the following upper and lower bounds on GD risk under Assumption 1. The bounds match except for the additional concentration coeficient in the tail regularization term, which is bounded as ${ \cal O } ( ( \eta t ) ^ { - 2 } )$ . This term is included in the “efective variance” characterized in Wu et al. (2026), and can be absorbed in the variance when SNR is bounded (Corollary 3.1).

Theorem B.1 (GD risk). Under Assumption 1, there exist constants $c _ { 0 } , \ldots , c _ { 4 } > 1$ that only depend on $c _ { x } , c _ { y }$ for which the following holds. Let $\hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } }$ be given by (GD) with stepsize $\eta \leq ( c _ { 4 } ( \lambda _ { 1 } + \operatorname { t r } ( \Sigma ) / n ) ) ^ { - 1 }$ and steps $t \geq 1$ . Let

$$
k ^ { * } : = \operatorname* { m i n } \left\{ k : \frac { 1 } { \eta t } + \frac { \sum _ { i > k } \lambda _ { i } } { n } \geq c _ { 2 } \lambda _ { k + 1 } \right\} , \quad \tilde { \lambda } : = \frac { 1 } { \eta t } + \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } , \quad D : = k ^ { * } + \frac { 1 } { \tilde { \lambda } ^ { 2 } } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } .
$$

• Upper bound. If additionally $k ^ { * } \leq n / c _ { 3 }$ , then with probability at least $1 - \delta - \exp ( - n / c _ { 0 } )$ over sampling ofX, it holds that

$$
\frac { 1 } { c _ { 1 } } \mathbb { E } [ \mathcal { E } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } ) | \mathbf { X } ] \leq \left\| ( \mathbf { I } - \eta \boldsymbol { \Sigma } ) ^ { t } \mathbf { w } ^ { * } \right\| _ { \mathbf { Z } _ { 0 , k ^ { * } } } ^ { 2 } + \left( \alpha _ { k ^ { * } } + \frac { 1 } { ( \eta t ) ^ { 2 } } \frac { k ^ { * } + \log ( 1 / \delta ) } { n } \right) \left\| \mathbf { w } ^ { * } \right\| _ { \mathbf { Z } _ { 0 , k ^ { * } } ^ { - 1 } } ^ { 2 } + \left\| \mathbf { w } ^ { * } \right\| _ { \mathbf { Z } _ { k ^ { * } , \infty } } ^ { 2 } + \sigma ^ { 2 } \frac { D } { n } .
$$

• Lower bound. In expectation,

$$
c _ { 1 } \mathbb { E } \mathcal { E } ( \hat { { \mathbf { w } _ { t } ^ { \mathrm { g d } } } } ) \geq \left\| ( { \mathbf { I } } - \eta \Sigma ) ^ { t } { \mathbf { w } ^ { * } } \right\| _ { \Sigma _ { 0 : k ^ { * } } } ^ { 2 } + \alpha _ { k ^ { * } } \left\| { \mathbf { w } ^ { * } } \right\| _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } + \left\| { \mathbf { w } ^ { * } } \right\| _ { \Sigma _ { k ^ { * } ; \infty } } ^ { 2 } + \sigma ^ { 2 } \operatorname* { m i n } \left\{ \frac { D } { n } , 1 \right\} .
$$

Proof of Corollary 3.1. For the upper bound, we use Theorem B.1 with $\delta = \exp ( - k ^ { * } / c _ { 0 } )$ . Then

$$
\frac { \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } + \frac { 1 } { ( \eta t ) ^ { 2 } } \frac { k ^ { * } + \log ( 1 / \delta ) } { n } \le \frac { \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } + \frac { 1 } { ( \eta t ) ^ { 2 } } \frac { 2 k ^ { * } } { n } \le 3 \tilde { \lambda } ^ { 2 } \frac { D } { n } .
$$

Since $\tilde { \lambda } < c _ { 2 } \lambda _ { k ^ { * } }$ by definition and $\| \mathbf { w } ^ { * } \| _ { \Sigma } ^ { 2 } \leq b \sigma ^ { 2 }$ , we must have

$$
\left( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } + \frac { 1 } { ( \eta t ) ^ { 2 } } \frac { k ^ { * } + \log ( 1 / \delta ) } { n } \right) \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 ; k ^ { * } } ^ { - 1 } } ^ { 2 } \leq 3 c _ { 2 } ^ { 2 } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 ; k ^ { * } } } ^ { 2 } \frac { D } { n } \leq 3 c _ { 2 } ^ { 2 } b \sigma ^ { 2 } \frac { D } { n } .
$$

Bringing this into Theorem B.1 and rescaling constants give the desired bound.

For the lower bound, it sufices to note that when $k ^ { * } \leq n / c _ { 3 }$

$$
D \leq k ^ { * } + \frac { \lambda _ { k ^ { * } + 1 } } { \tilde { \lambda } } \cdot \frac { 1 } { \tilde { \lambda } } \sum _ { i > k ^ { * } } \lambda _ { i } \leq k ^ { * } + \frac { n } { c _ { 2 } } = O ( n ) ,
$$

thus the min thresholding in the variance can be removed.

Proof of Theorem B.1, upper bound. The upper bound follows from an application of Theorem 5.3 to the GD example in Example 2. By Lemma A.1, it holds that $\| \hat { \boldsymbol { \Sigma } } _ { \leq k } \| \leq \| \hat { \boldsymbol { \Sigma } } \| \leq 1 / \eta ,$ so (6) holds with probability at least $1 - \exp ( - n / c _ { 0 } )$ . Also, $c _ { 2 } \lambda _ { k ^ { * } + 1 } \leq \tilde { \lambda }$ and we may assume $\log ( 1 / \delta ) \leq n$ , so that

$$
\alpha _ { k ^ { * } } + \frac { \lambda _ { k ^ { * } + 1 } ^ { 2 } \log ( 1 / \delta ) } { n } \leq c \left( \alpha _ { k ^ { * } } + \frac { 1 } { ( \eta t ) ^ { 2 } } \frac { \log ( 1 / \delta ) } { n } \right) .
$$

Bringing this into Theorem 5.3 and combining with the ridge variance for $\lambda = ( \eta t ) ^ { - 1 }$ (Tsigler and Bartlett, 2023) proves the upper bound. □

The rest of the section is devoted to proving the lower bound.

## B.1 GD risk decomposition

Recall that the GD estimator defined in (GD) is

$$
\hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } = \mathbf { X } ^ { \top } \mathbf { \Psi } \mathbf { \Psi } ^ { ( t ) } \mathbf { y } , \quad \mathbf { \Psi } ^ { ( t ) } : = \mathbf { A } ^ { - 1 } \big ( \mathbf { I } - ( \mathbf { I } - \gamma \mathbf { A } ) ^ { t } \big ) , \quad \gamma : = \frac { \eta } { n } .
$$

Assume

$$
\eta \leq \frac { 1 } { c _ { 4 } ( \lambda _ { 1 } + \operatorname { t r } ( \pmb { \Sigma } ) / n ) } .\tag{10}
$$

Then under Assumption 1B, standard linear algebra (see, e.g. Wu et al., 2026, Appendix C) gives

$$
\mathbb { E } [ \mathcal { E } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } ) | \mathbf { X } ] \geq \langle \mathbf { B } , \mathbf { w } ^ { * \otimes 2 } \rangle + \frac { \sigma ^ { 2 } } { c _ { y } } \langle \mathbf { C } , \boldsymbol { \Sigma } \rangle ,
$$

where

$$
\mathbf { B } : = \big ( \mathbf { I } - \mathbf { X } ^ { \top } \pmb { \Psi } ^ { ( t ) } \mathbf { X } \big ) \Sigma \big ( \mathbf { I } - \mathbf { X } ^ { \top } \pmb { \Psi } ^ { ( t ) } \mathbf { X } \big ) , \quad \mathbf { C } : = \mathbf { X } ^ { \top } \big ( \pmb { \Psi } ^ { ( t ) } \big ) ^ { 2 } \mathbf { X } .
$$

Notice that Assumption 1C suggests that $\mathbb { E } \mathbf { B } _ { i j } = 0$ for $i \neq j$ (see, e.g. Wu et al., 2026, proof of Lemma C.2), which implies

$$
\mathbb { E } \langle \mathbf { B } , \mathbf { w } ^ { * \otimes 2 } \rangle = \mathbb { E } \sum _ { i } \mathbf { B } _ { i i } \mathbf { w } _ { i } ^ { * 2 } .
$$

Following the convention of Wu et al. (2026); Tsigler and Bartlett (2023), we write

$$
\pmb { \Sigma } \mathbf { e } _ { i } = \lambda _ { i } \mathbf { e } _ { i } , \quad \mathbf { X } \mathbf { e } _ { i } = \lambda _ { i } ^ { 1 / 2 } \mathbf { z } _ { i } , \quad \mathbf { z } _ { i } \in \mathbb { R } ^ { n } ,
$$

where entries of $\mathbf { z } _ { i }$ ’s are independent, mean zero, unit variance, and $c _ { x } ^ { 2 }$ -subgaussian by Assumption 1A. Then the diagonal entries of B decompose as (Wu et al., 2026; Tsigler and Bartlett, 2023),

$$
\mathbf { B } _ { i i } = \mathbf { e } _ { i } ^ { \top } \mathbf { B } \mathbf { e } _ { i }
$$

$$
\begin{array} { r l } & { \mathbf { \Theta } = \mathbf { e } _ { i } ^ { \top } \left( \mathbf { I } - \mathbf { X } ^ { \top } \Psi ^ { ( i ) } \mathbf { X } \right) \boldsymbol { \Sigma } \left( \mathbf { I } - \mathbf { X } ^ { \top } \Psi ^ { ( i ) } \mathbf { X } \right) \mathbf { e } _ { i } } \\ & { = \lambda _ { i } - 2 \lambda _ { i } ^ { 2 } \mathbf { Z } _ { i } ^ { \top } \Psi ^ { ( i ) } \boldsymbol { \Sigma } _ { i } + \lambda _ { i } \mathbf { Z } _ { i } ^ { \top } \Psi ^ { ( i ) } \mathbf { X } \boldsymbol { \Sigma } \boldsymbol { \Sigma } ^ { \top } \Psi ^ { ( i ) } \mathbf { Z } _ { i } } \\ & { = \lambda _ { i } - 2 \lambda _ { i } ^ { 2 } \mathbf { Z } _ { i } ^ { \top } \Psi ^ { ( i ) } \mathbf { z } _ { i } + \lambda _ { i } \mathbf { Z } _ { i } ^ { \top } \Psi ^ { ( i ) } \left( \displaystyle \sum _ { j } \lambda _ { j } ^ { 2 } \mathbf { z } _ { j } \mathbf { Z } _ { j } ^ { \top } \right) \Psi ^ { ( i ) } \mathbf { z } _ { i } } \\ & { = \lambda _ { i } - 2 \lambda _ { i } ^ { 2 } \mathbf { Z } _ { i } ^ { \top } \Psi ^ { ( i ) } \mathbf { z } _ { i } + \lambda _ { i } ^ { 3 } ( \mathbf { z } _ { i } ^ { \top } \Psi ^ { ( i ) } \mathbf { z } _ { i } ) ^ { 2 } + \lambda _ { i } \mathbf { Z } _ { i } ^ { \top } \Psi ^ { ( i ) } \left( \displaystyle \sum _ { j \neq i } \lambda _ { j } ^ { 2 } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \right) \Psi ^ { ( i ) } \mathbf { z } _ { i } } \\ &  = \lambda _ { i } \big ( 1 - \lambda _ { i } \mathbf { Z } _ { i } ^ { \top } \Psi ^ { ( i ) } \mathbf { z } _ { i } \big ) ^ { 2 } + \lambda _ { i } \mathbf { Z } _ { i } ^ { \top } \Psi ^ { ( i ) } \end{array}
$$

This leads to the following risk decomposition,

$$
\mathbb { E } \mathcal { E } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } ) \geq \mathbb { B } \underbrace { \sum _ { i } \lambda _ { i } { \mathbf { w } _ { i } ^ { * } } ^ { 2 } \big ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } _ { i } \big ) ^ { 2 } } _ { \mathrm { E f f e c t i v e B i a s } } + \mathbb { E } \underbrace { \sum _ { i } \lambda _ { i } { \mathbf { w } _ { i } ^ { * } } ^ { 2 } \mathbf { z } _ { i } ^ { \top } \Psi ^ { ( t ) } \bigg ( \sum _ { j \neq i } \lambda _ { j } ^ { 2 } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \bigg ) \Psi ^ { ( t ) } \mathbf { z } _ { i } } _ { \mathrm { E f f e c t i v e V a r i a n c e } } + \underbrace { \sigma ^ { 2 } \mathbb { B } \underbrace { \langle \mathbf { C } , \Sigma \rangle } _ { \mathrm { V a r i a n c e } } } _ { \mathrm { V a r i a n c e } } .
$$

It remains to bound efective bias, efective variance, and variance errors. The last is done in Wu et al. (2026, Lemma C.1), which we will restate for completeness. The main efort of this part is to provide new tight lower bounds on the efective bias and efective variance errors.

Throughout this part, let

$$
k ^ { * } : = \operatorname* { m i n } \left\{ k : \frac { 1 } { \eta t } + \frac { \sum _ { i > k } \lambda _ { i } } { n } \geq c _ { 2 } \lambda _ { k + 1 } \right\} , \quad \tilde { \lambda } : = \frac { 1 } { \eta t } + \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } .\tag{11}
$$

Lemma B.2 (reduction to ridge). The event of Lemma A.1 and stepsize condition (10) imply that

$$
\frac { 1 } { 2 } \bigg ( \mathbf { A } + \frac { n } { \eta t } \bigg ) ^ { - 1 } \preceq \Psi ^ { ( t ) } \preceq 2 \bigg ( \mathbf { A } + \frac { n } { \eta t } \bigg ) ^ { - 1 } .
$$

Proof of Lemma B.2. This is Wu et al. (2026, Lemma B.2). The lower inequality is understood on ran A. □

## B.2 Variance error

The following variance error bound is a restatement of (Wu et al., 2026, Lemma C.1), which is ultimately a reduction to the variance error of ridge (Bartlett et al., 2020; Tsigler and Bartlett, 2023).

Lemma B.3 (variance error). Under Assumption 1A, there exist $c _ { 0 } , c _ { 1 } , c _ { 2 } > 1$ that only depend on $c _ { x } \ f o r$ which thefollowing holds. For all � satisfying (10), with probability at least $1 - \exp ( - n / c _ { 0 } )$ , we have

$$
\langle \mathbf { C } , \Sigma \rangle \geq \frac { 1 } { c _ { 1 } } \operatorname* { m i n } \bigg \{ \frac { k ^ { * } + 1 / \tilde { \lambda } ^ { 2 } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } , 1 \bigg \} ,
$$

where $k ^ { * }$ and �<sup>˜</sup> are defined in (11).

ProofofLemma B.3. This is by Lemma B.2 and Tsigler and Bartlett (2023, Theorem 2).

## B.3 Efective bias error

Lemma B.4 (efective bias, tail bound). Under Assumption 1A,for some constant $c _ { 0 } > 1$ that only depends on $c _ { x } ,$ with probability at least $1 - \exp ( - n / c _ { 0 } )$ it holds that

$$
\mathrm { E f f e c t i v e B i a s } \geq \frac 1 8 \| \mathbf { w } _ { > k ^ { * } } ^ { * } \| _ { \Sigma _ { > k ^ { * } } } ^ { 2 } .
$$

Proof of Lemma B.4. Fix an index $j > k ^ { * }$ . By the definition of $k ^ { * }$ and ${ \tilde { \lambda } } ,$ , we have

$$
\tilde { \lambda } \geq c _ { 2 } \lambda _ { j } .
$$

Without loss of generality, we assume $c _ { 2 } \geq 1 6$ . By Hoefding’s inequality, Lemmas A.2 and B.2, with probability at least $1 - \exp ( - n / c _ { 0 } )$ for some constant $c _ { 0 } > 1$ , we have

$$
\| \mathbf { z } _ { j } \| ^ { 2 } \leq 2 n , \quad \| \Psi ^ { ( t ) } \| \leq 2 \biggl \| \biggl ( \mathbf { A } _ { > k ^ { * } } + \frac { n } { \eta t } \biggr ) ^ { - 1 } \biggr \| \leq \frac { 4 } { n \tilde { \lambda } } .
$$

Under this event, we have

$$
\lambda _ { j } \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } _ { j } \leq \lambda _ { j } \Vert \Psi ^ { ( t ) } \Vert \cdot \Vert \mathbf { z } _ { j } \Vert ^ { 2 } \leq \lambda _ { j } \frac { 8 } { \tilde { \lambda } } \leq \frac { 8 } { c _ { 2 } } \leq \frac { 1 } { 2 }
$$

since $c _ { 2 } \geq 1 6$ . Thus, for each $j > k ^ { * }$ , with probability at least $1 - \exp ( - n / c _ { 0 } )$ it holds that

$$
\big ( 1 - \lambda _ { j } \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } _ { j } \big ) ^ { 2 } \geq \frac { 1 } { 4 } .
$$

By Bartlett et al. (2020, Lemma 15), with probability at least $1 - 2 \exp ( - n / c _ { 0 } )$ it holds that

$$
\sum _ { j > k ^ { * } } \lambda _ { j } \mathbf { w } _ { j } ^ { * 2 } \big ( 1 - \lambda _ { j } \mathbf { z } _ { j } ^ { \mathsf { T } } \Psi ^ { ( t ) } \mathbf { z } _ { j } \big ) ^ { 2 } \geq \frac { 1 } { 2 } \sum _ { j > k ^ { * } } \lambda _ { j } \mathbf { w } _ { j } ^ { * 2 } \cdot \frac { 1 } { 4 } = \frac { 1 } { 8 } \| \mathbf { w } _ { > k ^ { * } } ^ { * } \| _ { { \boldsymbol \Sigma } _ { > k ^ { * } } } ^ { 2 } .
$$

The left-hand side is a lower bound on EfectiveBias, so we complete the proof by rescaling the constant. □

Lemma B.5 (efective bias, exponential bound). Under Assumption $I A ,$ , there exist constants $c _ { 0 } , c _ { 1 } > 1$ that only depend on $c _ { x }$ for which the following hold:

• with probability at least $1 - \exp ( - n / c _ { 0 } )$

$$
\mathrm { E f f e c t i v e B i a s } \geq \frac { 1 } { 2 } \Big \Vert ( \mathbf { I } - 1 . 1 \eta \boldsymbol { \Sigma } ) ^ { t } \mathbf { w } ^ { * } \Big \Vert _ { \boldsymbol { \Sigma } } ^ { 2 } ;
$$

• in expectation,

$$
\mathbb { E } \mathrm { E f f e c t i v e B i a s } \geq \frac { 5 } { 1 8 } \Big \Vert ( \mathbf { I } - \eta \boldsymbol { \Sigma } ) ^ { t } \mathbf { w } ^ { * } \Big \Vert _ { \Sigma } ^ { 2 } .
$$

Proof of Lemma B.5. Notice that

$$
\begin{array} { r l r } & { { \mathbf { I } } - { \mathbf { X } } ^ { \top } \Psi ^ { ( t ) } { \mathbf { X } } = { \mathbf { I } } - { \mathbf { X } } ^ { \top } { \mathbf { A } } ^ { - 1 } \big ( { \mathbf { I } } - ( { \mathbf { I } } - \gamma { \mathbf { A } } ) ^ { t } \big ) { \mathbf { X } } } & { \qquad \mathrm { d e f i n i t i o n ~ o f ~ } \Psi ^ { ( t ) } } \\ & { \qquad = { \mathbf { I } } - { \mathbf { X } } ^ { \top } { \mathbf { A } } ^ { - 1 } { \mathbf { X } } \big ( { \mathbf { I } } - ( { \mathbf { I } } - \gamma { \mathbf { X } } ^ { \top } { \mathbf { X } } ) ^ { t } \big ) } & { \qquad { \mathbf { A } } = { \mathbf { X } } { \mathbf { X } } ^ { \top } } \\ & { \qquad = \big ( { \mathbf { I } } - \gamma { \mathbf { X } } ^ { \top } { \mathbf { X } } \big ) ^ { t } , } \\ { \Longrightarrow \quad } & { { \mathbf { I } } - \lambda _ { i } { \mathbf { Z } } _ { i } ^ { \top } \Psi ^ { ( t ) } { \mathbf { Z } } _ { i } = { \mathbf { e } } _ { i } ^ { \top } \big ( { \mathbf { I } } - { \mathbf { X } } ^ { \top } \Psi ^ { ( t ) } { \mathbf { X } } \big ) { \mathbf { e } } _ { i } } & { \qquad { \mathbf { X } } { \mathbf { e } } _ { i } = \lambda _ { i } ^ { 1 / 2 } { \mathbf { Z } } _ { i } } \\ & { \qquad = { \mathbf { e } } _ { i } ^ { \top } ( { \mathbf { I } } - \gamma { \mathbf { X } } ^ { \top } { \mathbf { X } } ) ^ { t } { \mathbf { e } } _ { i } . } & { } \end{array}
$$

For each $i ,$ let the leave-one-out matrix be

$$
\mathbf { A } _ { - i } : = \sum _ { j \neq i } \lambda _ { j } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } .
$$

The stepsize condition (10) and Lemma A.1 show that

$$
\mathrm { w i t h ~ p r o b a b i l i t y ~ a t ~ l e a s t ~ 1 - e x p } ( - n / c _ { 0 } ) \colon \ \gamma \| \mathbf { A } \| \leq \frac { 1 } { 2 } ,\tag{12}
$$

$$
\begin{array} { r l } { \implies } & { { } \gamma \| \mathbf { A } _ { - i } \| , ~ \gamma \lambda _ { i } \| \mathbf { z } _ { i } \| ^ { 2 } \leq \displaystyle \frac { 1 } { 2 } . \qquad \quad \mathbf { A } = \lambda _ { i } \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } + \mathbf { A } _ { - i } . } \end{array}
$$

On this event, we have

$$
\begin{array} { r l r l } { 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \mathsf { T } } \boldsymbol { \Psi } ^ { ( t ) } \mathbf { z } _ { i } = \mathbf { e } _ { i } ^ { \mathsf { T } } ( \mathbf { I } - \gamma \mathbf { X } ^ { \mathsf { T } } \mathbf { X } ) ^ { t } \mathbf { e } _ { i } } \\ { \geq \big ( 1 - \gamma \mathbf { e } _ { i } ^ { \mathsf { T } } \mathbf { X } ^ { \mathsf { T } } \mathbf { X } \mathbf { e } _ { i } \big ) ^ { t } } & { \qquad } & & { \mathrm { J e n s e n ' s ~ i n e q u a l i t y } } \\ { = \big ( 1 - \lambda _ { i } \gamma \| \mathbf { z } _ { i } \| ^ { 2 } \big ) ^ { t } } & { \qquad } & & { \mathbf { X } \mathbf { e } _ { i } = \lambda _ { i } ^ { 1 / 2 } \mathbf { z } _ { i } } \\ { \geq 0 . } \end{array}
$$

High probability lower bound. Fix an index �. By Assumption 1A and Hoefding’s inequality, with probability at least $1 - \exp ( - n / c _ { 0 } )$ for suficiently large $c _ { 0 } ,$ , it holds that $\Vert \mathbf { z } _ { i } \Vert ^ { 2 } \leq 1 . 1 n$ . On the intersection of this event with (12), we have

$$
\begin{array} { r } { 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } _ { i } \geq \big ( 1 - \lambda _ { i } \gamma \| \mathbf { z } _ { i } \| ^ { 2 } \big ) ^ { t } \geq ( 1 - 1 . 1 \eta \lambda _ { i } ) ^ { t } \geq 0 . \qquad \gamma = \eta / n , \ \eta \leq 1 / ( 2 \lambda _ { 1 } ) } \end{array}
$$

Similarly to the proof of Lemma B.4, by Bartlett et al. (2020, Lemma 15), we have with probability at least $1 - 4 \exp ( - n / c _ { 0 } )$

$$
\mathrm { { E f f e c t i v e B i a s : } } = \sum _ { i } { \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } \big ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \mathsf { T } } \Psi ^ { ( i ) } \mathbf { z } _ { i } \big ) ^ { 2 } } \geq \frac { 1 } { 2 } \sum _ { i } { \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } ( 1 - 1 . 1 \eta \lambda _ { i } ) ^ { 2 t } } = \frac { 1 } { 2 } \big \| ( \mathbf { I } - 1 . 1 \eta \Sigma ) ^ { t } \mathbf { w } ^ { * } \big \| _ { \Sigma } ^ { 2 } ,
$$

which gives the lower bound after rescaling the constants.

Expectation lower bound. Since $\mathbf { A } _ { - i }$ and $\mathbf { z } _ { i }$ are independent, it follows that

$$
\begin{array} { r l r l } & { \mathbb { E } \big ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \mathsf { T } } \Psi ^ { ( t ) } \mathbf { z } _ { i } \big ) ^ { 2 } \geq \operatorname* { P r } \left( \gamma \| \mathbf { A } _ { - i } \| \leq \frac { 1 } { 2 } \right) \mathbb { E } \left( \mathbf { 1 } _ { \left[ \gamma \lambda _ { i } \| \mathbf { z } _ { i } \| ^ { 2 } \leq \frac { 1 } { 2 } \right] } ( 1 - \gamma \lambda _ { i } \| \mathbf { z } _ { i } \| ^ { 2 } ) ^ { 2 t } \right) } \\ & { \geq \frac { 1 } { 2 } \left( \mathbb { E } ( 1 - \gamma \lambda _ { i } \| \mathbf { z } _ { i } \| ^ { 2 } ) _ { + } ^ { 2 t } - 2 ^ { - 2 t } \right) } & & { \mathrm { ~ b y ~ } ( 1 2 ) } \\ & { \geq \frac { 1 } { 2 } \left( ( 1 - \mathbb { E } \gamma \lambda _ { i } \| \mathbf { z } _ { i } \| ^ { 2 } ) _ { + } ^ { 2 t } - 2 ^ { - 2 t } \right) } & & { \mathrm { ~ J e n s e n ' s ~ i n e q u a l i t y } } \\ & { = \frac { 1 } { 2 } \left( ( 1 - \eta \lambda _ { i } ) ^ { 2 t } - 2 ^ { - 2 t } \right) . } \end{array}
$$

By increasing �<sub>4</sub>, we may assume

$$
\begin{array} { r l } { \eta \lambda _ { i } \leq \frac { \lambda _ { i } } { c _ { 4 } ( \lambda _ { 1 } + \operatorname { t r } ( \Sigma ) / n ) } \leq \frac { 1 } { 4 } } & { \Longrightarrow \quad ( 1 - \eta \lambda _ { i } ) ^ { 2 t } \geq \bigg ( \frac { 3 } { 4 } \bigg ) ^ { 2 t } > \frac { 9 } { 4 } \cdot 2 ^ { - 2 t } } \\ & { \Longrightarrow \quad \mathbb { E } \big ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } _ { i } \big ) ^ { 2 } \geq \frac { 5 } { 1 8 } ( 1 - \eta \lambda _ { i } ) ^ { 2 t } . } \end{array}
$$

We thus have

$$
\mathbb { E } \mathrm { { f f e c t i v e B i a s } } = \mathbb { E } \sum _ { i } \lambda _ { i } { \mathbf { w } _ { i } ^ { * } } ^ { 2 } \big ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \top } { \mathbf { \Psi } } ^ { ( t ) } \mathbf { z } _ { i } \big ) ^ { 2 } \geq \frac { 5 } { 1 8 } \sum _ { i } \lambda _ { i } ( 1 - \eta \lambda _ { i } ) ^ { 2 t } { \mathbf { w } _ { i } ^ { * } } ^ { 2 } = \frac { 5 } { 1 8 } \Big \| ( \mathbf { I } - \eta \Sigma ) ^ { t } { \mathbf { w } ^ { * } } \Big \| _ { \Sigma } ^ { 2 } ,
$$

which completes our proof.

Lemma B.6 (efective bias, OLS-type lower bound). Under Assumption 1A, there exist constants $c _ { 0 } , c _ { 1 } , c _ { 5 } > 1$ that depend only on $c _ { x }$ such that the following holds. If �<sup>∗</sup> defined in (11) satisfies

$$
\frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \geq c _ { 5 } \sqrt { \frac { \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } } ,
$$

then it holds with probability $1 - \exp ( - n / c _ { 0 } )$ that

$$
\mathrm { E f f e c t i v e B i a s } \geq \frac { 1 } { c _ { 1 } } \mathopen { } \mathclose \bgroup \left( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \aftergroup \egroup \right) ^ { 2 } \| \mathbf { w } _ { \leq k ^ { * } } ^ { * } \| _ { \Sigma _ { \leq k ^ { * } } ^ { - 1 } } ^ { 2 } .
$$

ProofofLemma B.6. By the stepsize condition (10) and Lemma A.1, we have $0 \preceq \Psi ^ { ( t ) } \preceq \mathbf { A } ^ { - 1 }$ with probability at least $1 - \exp ( - n / c _ { 0 } )$ . Under this event, we have

$$
\mathrm { E f f e c t i v e B i a s } : = \sum _ { i } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } \big ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \boldsymbol { \Psi } ^ { ( t ) } \mathbf { z } _ { i } \big ) ^ { 2 } \geq \sum _ { i } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } \big ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \mathbf { A } ^ { - 1 } \mathbf { z } _ { i } \big ) ^ { 2 } ,
$$

the right-hand side of which is exactly the efective bias error of OLS (this idea has been used in, e.g., Wu et al., 2026, Lemma C.2). Furthermore, the assumption on $k ^ { * }$ and the standard small-ball concentration bound imply that for some constants $c _ { 0 } , c _ { 5 } > 1$ , with probability at least $1 - \exp ( - n / c _ { 0 } )$

$$
\begin{array} { r l } & { \displaystyle \sum _ { i > k ^ { * } } \lambda _ { i } \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } \succeq \bigg ( \displaystyle \sum _ { i > k ^ { * } } \lambda _ { i } - \frac { c _ { 5 } } { 2 } \sqrt { n \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } \bigg ) \mathbf { I } , } \\ { \implies } & { \displaystyle \sum _ { i > k ^ { * } } \lambda _ { i } \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } \succeq \frac { 1 } { 2 } \displaystyle \sum _ { i > k ^ { * } } \lambda _ { i } \mathbf { I } . } \end{array}
$$

assumption on �

This enables the lower bound analysis for the efective bias of OLS by Tsigler and Bartlett (2023) (see the proof of their Lemma 8 with $\lambda = 0 )$ , which states, with rescaling of constants, that with probability at least $1 - \exp ( - n / c _ { 0 } )$

$$
\sum _ { i } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } \big ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \mathbf { A } ^ { - 1 } \mathbf { z } _ { i } \big ) ^ { 2 } \geq \frac { 1 } { c _ { 1 } } \bigg ( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \bigg ) ^ { 2 } \| \mathbf { w } _ { \leq k ^ { * } } ^ { * } \| _ { \Sigma _ { \leq k ^ { * } } ^ { - 1 } } ^ { 2 } .
$$

We complete the proof by chaining the two inequalities with a union bound and rescaling the constant. □

## B.4 Efective variance error

Obtaining a high-probability lower bound for efective variance error is hard for general spectra. However, it is possible to obtain a lower bound on the expectation, as shown below.

Lemma B.7 (GD leave-one-out). Under Assumption 1A, there exists a constant $c _ { 0 } > 1$ that depends only on $c _ { x }$ such that the following holds. For a fixed index � with $\lambda _ { j } > 0 ,$ , let

$$
\mathbf { A } _ { - j } : = \sum _ { i \neq j } \lambda _ { i } \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } , \quad \boldsymbol { \Psi } _ { - j } ^ { ( t ) } : = \gamma \sum _ { s = 0 } ^ { t - 1 } ( \mathbf { I } - \gamma \mathbf { A } _ { - j } ) ^ { s } .
$$

If � satisfies

$$
\gamma \| \mathbf { A } _ { - j } \| \leq \frac { 1 } { 2 } , \quad 1 6 c _ { x } ^ { 4 } \lambda _ { j } \operatorname { t r } \big ( \Psi _ { - j } ^ { ( t ) } \big ) \leq \frac { 1 } { 2 } ,
$$

then for any $n \geq c _ { 0 } ,$ , in expectation over $\mathbf { z } _ { j }$ , we have

$$
\mathbb { E } _ { \mathbf { z } _ { j } } \Psi ^ { ( t ) } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \succeq \frac { 1 } { 1 6 } \big ( \Psi _ { - j } ^ { ( t ) } \big ) ^ { 2 } .
$$

Proof of Lemma B.7. In this proof, we only consider the randomness of ${ \bf z } _ { j } ;$ so E refers to expectation over $\mathbf { z } _ { j }$ It sufices to show

for any fixed vector z,

$$
\mathbb { E } \big ( \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } \big ) ^ { 2 } \geq \frac { 1 } { c _ { 1 } } \big \| \Psi _ { - j } ^ { ( t ) } \mathbf { z } \big \| ^ { 2 } .
$$

Let

$$
\mathcal { F } : = \{ \gamma \lambda _ { j } \| \mathbf { z } _ { j } \| ^ { 2 } \leq 1 / 2 \} , \quad \mathbf { m } : = \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } ,
$$

then $\operatorname* { P r } ( \mathcal { F } ) \geq 1 - \exp ( - n / c _ { 0 } )$ by Assumption 1A and the stepsize condition (10), and under ${ \mathcal F } .$ , we have

$$
\gamma \| \mathbf { A } \| \leq 1 \quad { \mathrm { s i n c e ~ } } \mathbf { A } = \mathbf { A } _ { - j } + \lambda _ { j } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } { \mathrm { a n d } } \ \gamma \| \mathbf { A } _ { - j } \| \leq { \frac { 1 } { 2 } } .
$$

Notice that

$$
\begin{array} { r l r l } & { \| \mathbf { m } \| ^ { 2 } = \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { m } ^ { \top } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } } \\ & { \qquad \leq \sqrt { \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } ( \mathbf { m } ^ { \top } \mathbf { z } _ { j } ) ^ { 2 } \mathbb { E } \left( \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } \right) ^ { 2 } } \qquad } & & { \mathrm { C a u c h y - S c h w a r z ~ i n e q u a l i t y } } \\ & { \qquad \leq \sqrt { \| \mathbf { m } \| ^ { 2 } \mathbb { E } \left( \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } \right) ^ { 2 } } , \qquad } & & { \mathbf { 1 } _ { \mathcal { F } } \leq 1 \mathrm { ~ a n d ~ } \mathbb { E } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } = \mathbf { I } } \end{array}
$$

which implies that

$$
\begin{array} { r } { \mathbb { E } \big ( \mathbf { z } _ { j } ^ { \top } \mathbf { \Psi } \mathbf { \Psi } ^ { ( t ) } \mathbf { z } \big ) ^ { 2 } \geq \| \mathbf { m } \| ^ { 2 } = \big \| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \mathbf { \Psi } \mathbf { \Psi } ^ { ( t ) } \mathbf { z } \big \| ^ { 2 } . } \end{array}
$$

It remains to bound the right-hand side. Let

$$
\mathbf { Q } _ { - j } ^ { ( t ) } : = \Psi _ { - j } ^ { ( t ) } - \Psi ^ { ( t ) } \quad \implies \quad \left\| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \mathbf { z } \right\| \geq \left\| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \Psi _ { - j } ^ { ( t ) } \mathbf { z } \right\| - \left\| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \mathbf { Q } _ { - j } ^ { ( t ) } \mathbf { z } \right\| .
$$

For the first term on the right-hand side, we have

$$
\begin{array} { r l r l } { \Big \| \mathbb { B } 1 _ { \mathcal { T } ^ { \mathbb { Z } } } \mathbb { Z } _ { j } ^ { \top } \Psi _ { - j } ^ { ( t ) } \mathbb { z } \Big \| \geq \Big \| \mathbb { B } \mathbf { z } _ { j } \mathbb { Z } _ { j } ^ { \top } \Psi _ { - j } ^ { ( t ) } \mathbb { z } \Big \| - \Big \| \mathbb { B } 1 _ { \mathcal { T } ^ { \mathbb { C } } } \mathbf { z } _ { j } \mathbb { z } _ { j } ^ { \top } \Psi _ { - j } ^ { ( t ) } \mathbb { z } \Big \| } & & { } \\ { \geq \Big ( 1 - \Big \| \mathbb { E } 1 _ { \mathcal { T } ^ { \mathbb { C } } } \mathbf { z } _ { j } \mathbb { z } _ { j } ^ { \top } \Big \| \Big ) \Big \| \Psi _ { - j } ^ { ( t ) } \mathbb { z } \Big \| } & & { \mathbb { E } \mathbb { z } _ { j } \mathbb { z } _ { j } ^ { \top } = \mathbf { I } } \\ { \geq \left( 1 - \sqrt { \mathbb { B } 1 _ { \mathcal { T } ^ { \mathbb { C } } } \mathbb { B } \| \mathbf { z } _ { j } \| ^ { 4 } } \right) \Big \| \Psi _ { - j } ^ { ( t ) } \mathbb { z } \Big \| } & & { \mathrm { C a u c h y - S c h w a r z ~ i n e q u a l i t y } } \\ { \geq \Big ( 1 - \exp ( - n / c _ { 0 } ) 4 c _ { x } ^ { 2 } \mathbb { E } \| \mathbf { z } _ { j } \| ^ { 2 } \Big ) \Big \| \Psi _ { - j } ^ { ( t ) } \mathbb { z } \Big \| } & & { \mathrm { s u b g a u s s i a n ~ h y p e r c o n t r a c t i v i t y } } \\ { = \Big ( 1 - \exp ( - n / c _ { 0 } ) 4 c _ { x } ^ { 2 } n \Big ) \Big \| \Psi _ { - j } ^ { ( t ) } \mathbb { z } \Big \| } & & { \mathbb { E } \mathbb { z } _ { j } \mathbb { z } _ { j } ^ { \top } = \mathbf { I } } \\  \geq \frac { 3 } { 4 } \Big \| \Psi _ { - j } \end{array}
$$

In sum, we have shown that

$$
\begin{array} { r l } & { \sqrt { \mathbb { E } \big ( \mathbf { z } _ { j } ^ { \top } \mathbf { \Psi } \mathbf { \Psi } \mathbf { z } \big ) ^ { 2 } } \geq \Big \| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \mathbf { \Psi } \mathbf { \Psi } ^ { ( t ) } \mathbf { z } \Big \| } \\ & { \qquad \geq \Big \| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \mathbf { \Psi } _ { - j } ^ { ( t ) } \mathbf { z } \Big \| - \Big \| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \mathbf { Q } _ { - j } ^ { ( t ) } \mathbf { z } \Big \| } \\ & { \qquad \geq \frac { 3 } { 4 } \Big \| \Psi _ { - j } ^ { ( t ) } \mathbf { z } \Big \| - \Big \| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \mathbf { Q } _ { - j } ^ { ( t ) } \mathbf { z } \Big \| , } \end{array}
$$

so to complete our proof it sufices to show that

$$
\Big \| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \mathbf { Q } _ { - j } ^ { ( t ) } \mathbf { z } \Big \| \leq \frac { 1 } { 2 } \Big \| \Psi _ { - j } ^ { ( t ) } \mathbf { z } \Big \| .
$$

Since $\lambda _ { j } > 0$ , we have ${ \bf { z } } _ { j } \in  { }$ ran A, and telescoping gives

$$
\mathbf { z } _ { j } ^ { \mathsf { T } } \Psi ^ { ( t ) } = \gamma \sum _ { s = 0 } ^ { t - 1 } \mathbf { z } _ { j } ^ { \mathsf { T } } ( \mathbf { I } - \gamma \mathbf { A } ) ^ { s } \quad \implies \quad \mathbf { z } _ { j } ^ { \mathsf { T } } \mathbf { Q } _ { - j } ^ { ( t ) } = \gamma \sum _ { r = 0 } ^ { t - 2 } \mathbf { z } _ { j } ^ { \mathsf { T } } ( \mathbf { I } - \gamma \mathbf { A } ) ^ { r } \boldsymbol { \lambda } _ { j } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \mathsf { T } } \Psi _ { - j } ^ { ( t - 1 - r ) } .
$$

Under $\mathcal { F }$ , we have $\gamma \| \mathbf { A } \| \leq 1$ , so we have

$$
\left\| \mathbb { E } { \mathbf 1 } _ { \mathcal { F } } { \mathbf Z } _ { j } { \mathbf Z } _ { j } ^ { \top } { \mathbf Q } _ { - j } ^ { ( t ) } { \mathbf Z } \right\| \leq \lambda _ { j } \gamma \sum _ { r = 0 } ^ { t - 2 } \left\| \mathbb { E } { \mathbf 1 } _ { \mathcal { F } } { \mathbf Z } _ { j } { \mathbf Z } _ { j } ^ { \top } ( { \mathbf I } - \gamma { \mathbf A } ) ^ { r } { \mathbf Z } _ { j } { \mathbf Z } _ { j } ^ { \top } { \mathbf W } _ { - j } ^ { ( t - 1 - r ) } { \mathbf Z } \right\|
$$

$$
\begin{array} { r l } { \underset { r = 0 } { \overset { t - 2 } { \sum } } \big \| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \big ( \mathbf { I } - \gamma \mathbf { A } \big ) ^ { r } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \big \| \cdot \big \| \Psi _ { - j } ^ { ( t - 1 - r ) } \mathbf { z } \big \| } & { \ \Psi _ { - j } ^ { ( t - 1 - r ) } \mathbf { z } \mathrm { ~ i s ~ i n d e p e n d e n t ~ o f ~ } \mathbf { z } _ { j } } \\ { \leq \lambda _ { j } \gamma \displaystyle \sum _ { r = 0 } ^ { t - 2 } \big \| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \big ( \mathbf { I } - \gamma \mathbf { A } \big ) ^ { r } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \big \| \cdot \big \| \Psi _ { - j } ^ { ( t ) } \mathbf { z } \big \| . } & { \ \Psi _ { - j } ^ { ( t - 1 ) 2 } \preceq \Psi _ { - j } ^ { ( t ) 2 } } \end{array}
$$

To proceed, notice that under $\mathcal { F }$

$$
\begin{array} { r l } & { ( \mathbf { I } - \gamma \mathbf { A } _ { - j } ) ^ { t } - ( \mathbf { I } - \gamma \mathbf { A } ) ^ { t } = \gamma \displaystyle \sum _ { s = 0 } ^ { t - 1 } ( \mathbf { I } - \gamma \mathbf { A } ) ^ { s } \lambda _ { j } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } ( \mathbf { I } - \gamma \mathbf { A } _ { - j } ) ^ { t - 1 - s } } \\ & { \implies \quad \mathbf { z } _ { j } ^ { \top } ( \mathbf { I } - \gamma \mathbf { A } ) ^ { r } \mathbf { z } _ { j } \leq \mathbf { z } _ { j } ^ { \top } ( \mathbf { I } - \gamma \mathbf { A } _ { - j } ) ^ { r } \mathbf { z } _ { j } . } \end{array}
$$

So we have

$$
\begin{array} { r l r l } { \mathbb { E } { \mathbf 1 } _ { \mathcal { F } ^ { \mathbf 2 } j } { \mathbf 1 } _ { j } ^ { \top } ( { \mathbf 1 } - \gamma { \mathbf A } ) ^ { r } { \mathbf 2 } _ { j } { \mathbf Z } _ { j } ^ { \top } \preceq \mathbb { E } { \mathbf 1 } _ { \mathcal { F } ^ { \mathbf 2 } j } { \mathbf 1 } _ { j } ^ { \top } ( { \mathbf 1 } - \gamma { \mathbf A } _ { - j } ) ^ { r } { \mathbf 1 } _ { j } { \mathbf Z } _ { j } ^ { \top } } & { \qquad } & & { } \\ { \preceq \mathbb { E } { \mathbf 1 } _ { j } { \mathbf Z } _ { j } ^ { \top } ( { \mathbf 1 } - \gamma { \mathbf A } _ { - j } ) ^ { r } { \mathbf 2 } _ { j } { \mathbf Z } _ { j } ^ { \top } } & { \qquad } & & { { \mathbf 1 } _ { \mathcal { F } } \preceq { \mathbf 1 } } \\ & { \preceq 1 6 c _ { x } ^ { 4 } \mathrm { t r } \left( ( { \mathbf I } - \gamma { \mathbf A } _ { - j } ) ^ { r } \right) { \mathbf 1 } . } & { \qquad } & & { \mathrm { s u b g a u s s i a n h y p e r c o n t r a c t i v i t y } } \end{array}
$$

Bringing this back, we have

$$
\begin{array} { r l r } { \left\| \mathbb { E } { \mathbf 1 } _ { \mathcal { F } ^ { \mathbf { Z } } , j } { \mathbf { z } } _ { j } ^ { \top } { \mathbf { Q } } _ { - j } ^ { ( t ) } { \mathbf { z } } \right\| \leq 1 6 c _ { x } ^ { 4 } \lambda _ { j } \gamma \displaystyle \sum _ { r = 0 } ^ { t - 2 } \mathrm { t r } \left( ( { \mathbf { I } } - \gamma { \mathbf { A } } _ { - j } ) ^ { r } \right) \cdot \left\| \Psi _ { - j } ^ { ( t ) } { \mathbf { z } } \right\| } \\ { = 1 6 c _ { x } ^ { 4 } \lambda _ { j } \mathrm { t r } \left( \Psi _ { - j } ^ { ( t - 1 ) } \right) \cdot \left\| \Psi _ { - j } ^ { ( t ) } { \mathbf { z } } \right\| } & { } & { \ \Psi _ { - j } ^ { ( t - 1 ) } : = \gamma \displaystyle \sum _ { r = 0 } ^ { t - 2 } ( { \mathbf { I } } - \gamma { \mathbf { A } } _ { - j } ) ^ { r } } \\ { \leq 1 6 c _ { x } ^ { 4 } \lambda _ { j } \mathrm { t r } \left( \Psi _ { - j } ^ { ( t ) } \right) \cdot \left\| \Psi _ { - j } ^ { ( t ) } { \mathbf { z } } \right\| } & { } & { \ \Psi _ { - j } ^ { ( t - 1 ) } \preceq \Psi _ { - j } ^ { ( t ) } . } \end{array}
$$

The choice of � guarantees that

$$
1 6 c _ { x } ^ { 4 } \lambda _ { j } \operatorname { t r } \big ( \pmb { \Psi } _ { - j } ^ { ( t ) } \big ) \leq \frac { 1 } { 2 } \quad \implies \quad \big \| \mathbb { E } \mathbf { 1 } _ { \mathcal { F } } \mathbf { 2 } _ { j } \mathbf { z } _ { j } ^ { \top } \mathbf { Q } _ { - j } ^ { ( t ) } \mathbf { z } \big \| \leq \frac { 1 } { 2 } \big \| \pmb { \Psi } _ { - j } ^ { ( t ) } \mathbf { z } \big \| ,
$$

which completes our proof.

Lemma B.8 (efective variance). Under Assumption $I A ,$ , there exist constants $c _ { 0 } , c _ { 1 } > 1$ that depend only on $c _ { x }$ such that the following holds. For any $n \geq c _ { 0 }$ , it holds in expectation that

$$
\mathbb { E } \mathrm { E f f e c t i v e B i a s } + \mathbb { E } \mathrm { E f f e c t i v e V a r i a n c e } \geq \frac { 1 } { c _ { 1 } } \frac { \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } \| \mathbf { w } _ { \leq k ^ { * } } ^ { * } \| _ { \Sigma _ { \leq k ^ { * } } ^ { - 1 } } ^ { 2 } .
$$

ProofofLemma B.8. For each $j > k ^ { * }$ , the definition of $k ^ { * }$ implies that

$$
\lambda _ { j } \leq \frac { 1 } { c _ { 2 } } \left( \frac { 1 } { \eta t } + \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) .
$$

Additionally, Lemmas $\mathrm { A . 1 } , \mathrm { A . 2 }$ and B.2 implies that with probability at least $1 - \exp ( - n / c _ { 0 } )$ ，

$$
\gamma \| \mathbf { A } _ { - j } \| \leq \frac { 1 } { 2 } , \quad \Psi _ { - j } ^ { ( t ) } \preceq 2 \bigg ( \mathbf { A } _ { - j } + \frac { n } { \eta t } \bigg ) ^ { - 1 } \preceq \frac { 4 } { n / ( \eta t ) + \sum _ { i > k ^ { * } } \lambda _ { i } } \mathbf { I } \preceq \frac { 4 } { c _ { 2 } n \lambda _ { j } } \mathbf { I } .
$$

Then for a suficiently large $c _ { 2 }$

$$
1 6 c _ { x } ^ { 4 } \lambda _ { j } \operatorname { t r } \big ( \Psi _ { - j } ^ { ( t ) } \big ) \leq 1 6 c _ { x } ^ { 4 } \lambda _ { j } \cdot \frac n { c _ { 2 } n \lambda _ { j } } \leq \frac 1 2 .
$$

Under this event of $( { \bf z } _ { i } ) _ { i \neq j }$ which we denote as $\mathscr { G } _ { - j }$ , Lemma B.7 applies, hence

$$
\mathbb { E } \Psi ^ { ( t ) } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \Psi ^ { ( t ) } \succeq \frac { 1 } { c _ { 1 } } \mathbb { E } \mathbf { 1 } _ { \mathcal { G } _ { - j } } \big ( \Psi _ { - j } ^ { ( t ) } \big ) ^ { 2 } , \quad j > k ^ { * } .
$$

Then we have

$$
\begin{array} { r l } { \mathbb { E } \mathrm { H f e c t i v e V a r i a n c e : } = \mathbb { E } \displaystyle \sum _ { i } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } \mathbf { z } _ { i } ^ { \top } \Psi ^ { ( t ) } \left( \displaystyle \sum _ { j \neq i } \lambda _ { j } ^ { 2 } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \right) \Psi ^ { ( t ) } \mathbf { z } _ { i } } & { } \\ { \geq \mathbb { E } \displaystyle \sum _ { i \leq k ^ { * } } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } \mathbf { z } _ { i } ^ { \top } \Psi ^ { ( t ) } \left( \displaystyle \sum _ { j > k ^ { * } } \lambda _ { j } ^ { 2 } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \right) \Psi ^ { ( t ) } \mathbf { z } _ { i } } & { } \\ { \geq \displaystyle \frac { 1 } { c _ { 1 } } \displaystyle \sum _ { i \leq k ^ { * } } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } \displaystyle \sum _ { j > k ^ { * } } \lambda _ { j } ^ { 2 } \mathbb { E } \mathbf { 1 } _ { g - j } \mathbf { z } _ { i } ^ { \top } \left( \Psi _ { - j } ^ { ( t ) } \right) ^ { 2 } \mathbf { z } _ { i } . } \end{array}
$$

The term $\mathbf { z } _ { i } ^ { \top } \big ( \Psi _ { - j } ^ { ( t ) } \big ) ^ { 2 } \mathbf { z } _ { i }$ appears in the lower bound of GD variance, which was analyzed by ridge reduction (Tsigler and Bartlett, 2023; Wu et al., 2026). We use a similar argument. First, we have

$$
\begin{array} { r l r } { \mathbb { E } { \mathbf 1 } _ { \mathcal { G } _ { - j } } { \mathbf z } _ { i } ^ { \top } \big ( \Psi _ { - j } ^ { ( t ) } \big ) ^ { 2 } { \mathbf z } _ { i } \geq \mathbb { B } { \mathbf 1 } _ { \mathcal { G } _ { - j } } \frac { \big ( { \mathbf z } _ { i } ^ { \top } \Psi _ { - j } ^ { ( t ) } { \mathbf z } _ { i } \big ) ^ { 2 } } { \| { \mathbf z } _ { i } \| ^ { 2 } } } & { } & { \mathrm { C a u c h y - S c h w a r z ~ i n e q u a l i t y } } \\ { \geq \frac { 1 } { 2 n } \mathbb { E } { \mathbf 1 } _ { \mathcal { G } _ { - j } \cap \{ \| { \mathbf z } _ { i } \| ^ { 2 } \leq 2 n \} } \big ( { \mathbf z } _ { i } ^ { \top } \Psi _ { - j } ^ { ( t ) } { \mathbf z } _ { i } \big ) ^ { 2 } } \\ { \geq \frac { 1 } { 8 n } \mathbb { E } { \mathbf 1 } _ { \mathcal { G } _ { - j } \cap \{ \| { \mathbf z } _ { i } \| ^ { 2 } \leq 2 n \} } \Bigg ( { \mathbf z } _ { i } ^ { \top } \bigg ( { \mathbf A } _ { - j } + \frac { n } { \eta t } \bigg ) ^ { - 1 } { \mathbf z } _ { i } \Bigg ) ^ { 2 } , } & { } & { \mathrm { L e m m a ~ B . 2 } } \end{array}
$$

For $i \leq k ^ { * }$ , write $b _ { i } : = \mathbb { E } ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \boldsymbol { \Psi } ^ { ( t ) } \mathbf { z } _ { i } ) ^ { 2 }$ . On the event $\gamma \| \mathbf { A } \| \leq 1 / 2$ , Lemma B.2 gives

$$
\mathbf { z } _ { i } ^ { \top } \bigg ( \mathbf { A } _ { - j } + \frac { n } { \eta t } \mathbf { I } \bigg ) ^ { - 1 } \mathbf { z } _ { i } \geq \mathbf { z } _ { i } ^ { \top } \bigg ( \mathbf { A } + \frac { n } { \eta t } \mathbf { I } \bigg ) ^ { - 1 } \mathbf { z } _ { i } \geq \frac { 1 } { 2 } \mathbf { z } _ { i } ^ { \top } \pmb { \Psi } ^ { ( t ) } \mathbf { z } _ { i } .
$$

Using $x ^ { 2 } \geq 1 / 2 - ( 1 - x ) ^ { 2 }$ , we obtain, for suficiently large $n ,$

$$
\mathbb { E } \mathbf { 1 } _ { \mathcal { G } _ { - j } } \mathbf { z } _ { i } ^ { \top } \big ( \Psi _ { - j } ^ { ( t ) } \big ) ^ { 2 } \mathbf { z } _ { i } \geq \frac { 1 } { 3 2 n \lambda _ { i } ^ { 2 } } \left( \frac { 1 } { 4 } - b _ { i } \right) .
$$

By the definition of $k ^ { * }$ , for $i \leq k ^ { * }$

$$
\frac { \sum _ { j > k ^ { * } } \lambda _ { j } ^ { 2 } } { n \lambda _ { i } ^ { 2 } } \leq \frac { \tilde { \lambda } ^ { 2 } } { c _ { 2 } \lambda _ { i } ^ { 2 } } \leq c _ { 2 } .
$$

Substituting these bounds into the earlier lower bound for EEfectiveVariance yields

$$
\mathbb { E } \mathrm { E f f e c t i v e V a r i a n c e } \geq \frac { 1 } { 1 2 8 c _ { 1 } } \frac { \sum _ { j > k ^ { * } } \lambda _ { j } ^ { 2 } } { n } \| \mathbf { w } _ { \leq k ^ { * } } ^ { * } \| _ { \Sigma _ { \leq k ^ { * } } ^ { - 1 } } ^ { 2 } - \frac { c _ { 2 } } { 3 2 c _ { 1 } } \mathbb { E } \mathrm { E f f e c t i v e B i a s } .
$$

Rescaling constants completes the proof.

## B.5 Proof of GD lower bound

ProofofTheorem B.1, lower bound. Following the notation introduced, the risk decomposes into

$$
\mathbb { E } \mathcal { E } ( \hat { \mathbf { w } } _ { t } ^ { \mathrm { g d } } ) \geq \mathbb { E } \mathrm { E f f e c t i v e B i a s } + \mathbb { E } \mathrm { E f f e c t i v e V a r i a n c e } + \frac { \sigma ^ { 2 } } { c _ { y } } \mathbb { E } \mathrm { V a r i a n c e } ,
$$

where all the random variables are nonnegative. Note that a high-probability lower bound of a nonnegative random variable implies an expectation lower bound with rescaled constant factors. We then apply Lemmas B.3 to B.6 and B.8 in their expectation version. For bias error, there is a constant $c _ { 1 } > 1$ such that

$$
\mathbb { E } \mathrm { E f f e c t i v e B i a s } \geq \frac { 1 } { c _ { 1 } } \operatorname* { m a x } \Bigg \{ \big \| ( { \bf I } - \eta \Sigma ) ^ { t } { \bf w } ^ { * } \big \| _ { \Sigma } ^ { 2 } , \ \| { \bf w } _ { > k ^ { * } } ^ { * } \| _ { \Sigma _ { > k ^ { * } } } ^ { 2 } \Bigg \} , \qquad \mathrm { L e m m a s ~ B . 4 ~ a n d ~ B . } \Bigg \}
$$

$$
\mathbb { E } \mathrm { H f e c t i v e B i a s } \geq \frac { 1 } { c _ { 1 } } \bigg ( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \bigg ) ^ { 2 } \| { \mathbf { w } } _ { \leq k ^ { * } } ^ { * } \| _ { \Sigma _ { \leq k ^ { * } } ^ { - 1 } } ^ { 2 } ~ \mathrm { i f } ~ \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \geq c _ { 5 } \sqrt { \frac { \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } } . ~ \mathrm { L e m m a ~ B . 6 }
$$

Combining with Lemma B.8, for some suficiently large constant $c _ { 1 } ^ { \prime } > 1$ , we have

EEfectiveBias + EEfectiveVariance

$$
\ge \frac { 1 } { c _ { 1 } ^ { \prime } } \left( \left\| ( \mathbf { I } - \eta \boldsymbol { \Sigma } ) ^ { t } \mathbf { w } ^ { * } \right\| _ { \Sigma } ^ { 2 } + \left\| \mathbf { w } _ { > k ^ { * } } ^ { * } \right\| _ { \Sigma _ { > k ^ { * } } } ^ { 2 } + \left( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) ^ { 2 } \| \mathbf { w } _ { \le k ^ { * } } ^ { * } \| _ { \Sigma _ { \le k ^ { * } } ^ { - 1 } } ^ { 2 } + \frac { \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } \| \mathbf { w } _ { \le k ^ { * } } ^ { * } \| _ { \Sigma _ { \le k ^ { * } } ^ { - 1 } } ^ { 2 } \right) .
$$

For variance, there is a constant $c _ { 1 } > 1$ such that

$$
\mathbb { E } { \mathrm { V a r i a n c e } } \geq \frac { 1 } { c _ { 1 } } \operatorname* { m i n } \left\{ \frac { k ^ { * } + ( 1 / \tilde { \lambda } ^ { 2 } ) \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } , 1 \right\} .
$$

Putting them together, we obtain the lower bound in Theorem B.1 with a rescaling of the constants. □

## C PCR Risk Bounds

## C.1 Proof of Theorem 4.1

Proof of Theorem 4.1, upper bound. We show a slightly strengthened upper bound, which introduces a slackness parameter $\epsilon \in ( 0 , 1 ]$ . Let $n \geq c _ { 0 } \epsilon ^ { - 2 }$ and

$$
k ^ { * } : = \operatorname* { m i n } \left\{ k : ( 1 + \epsilon ) \rho + { \frac { 1 } { c _ { 2 } } } { \frac { \sum _ { i > k } \lambda _ { i } } { n } } \geq \lambda _ { k + 1 } \right\} .
$$

We prove the following statement: under Assumption 1, if additionally $k ^ { * } \le \epsilon ^ { 2 } n / c _ { 3 }$ , then with probability at least $1 - \delta - \mathrm { e x p } ( - \epsilon ^ { 2 } n / c _ { 0 } )$ over the randomness of sampling X it holds that

$$
\frac { 1 } { c _ { 1 } } \mathbb { E } \big [ \mathcal { E } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) | \mathbf { X } \big ] \leq \frac { 1 } { \epsilon ^ { 2 } } \left( \alpha _ { k ^ { * } } + \frac { \lambda _ { k ^ { * } + 1 } ^ { 2 } \log ( 1 / \delta ) } { n } \right) \big \| \mathbf { w } ^ { * } \big \| _ { \Sigma _ { 0 , k ^ { * } } ^ { - 1 } } ^ { 2 } + \big \| \mathbf { w } ^ { * } \big \| _ { \Sigma _ { k ^ { * } ; \infty } } ^ { 2 } + \sigma ^ { 2 } \frac { \operatorname* { m i n } \{ D , \hat { k } \} } { n } .
$$

First suppose $\rho = 0$ , then PCR coincides with minimum-norm OLS. Let $k = k ^ { * }$ and $s : = n ^ { - 1 } \textstyle \sum _ { i > k } \lambda _ { i }$ . If $s > 0$ then $s \geq c _ { 2 } \lambda _ { k + 1 }$ , so from Wu et al. (2026, Proposition 2.1), with probability at least $1 - \exp ( - n / c _ { 0 } )$

$$
\frac { 1 } { c _ { 1 } } \mathbb { E } [ \mathcal { E } ( \hat { \mathbf { w } } _ { 0 } ^ { \mathrm { p c r } } ) | \mathbf { X } ] \leq s ^ { 2 } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 : k } ^ { - 1 } } ^ { 2 } + \big \| \mathbf { w } ^ { * } \big \| _ { \Sigma _ { k : \infty } } ^ { 2 } + \sigma ^ { 2 } \frac { D } { n } .
$$

Moreover, Lemma ${ \mathrm { A . 1 } }$ gives $\mathbf { A } _ { > k } \succeq n s \mathbf { I } / 2$ , hence ${ \hat { k } } = n$ , while $\begin{array} { r } { D = k + s ^ { - 2 } \sum _ { i > k } \lambda _ { i } ^ { 2 } \le n ( c _ { 2 } ^ { - 1 } + c _ { 3 } ^ { - 1 } ) } \end{array}$ . The desired bound follows.

If $s = 0 ,$ , minimality of � implies rank $( { \boldsymbol { \Sigma } } ) = k$ . By Lemma A.1, $\boldsymbol { D } = \boldsymbol { k } = \boldsymbol { \hat { k } }$ and the OLS bias vanishes, and the claim follows since

$$
\mathbb { E } [ \mathcal { E } ( \hat { \mathbf { w } } _ { 0 } ^ { \mathrm { p c r } } ) | \mathbf { X } ] \leq \frac { c _ { y } \sigma ^ { 2 } } { n } \operatorname { t r } ( \Sigma _ { \leq k } \hat { \Sigma } _ { \leq k } ^ { - 1 } ) \leq \frac { 2 c _ { y } \sigma ^ { 2 } k } { n } .
$$

Now assume $\rho > 0$ . Set $k = k ^ { * } , \lambda = c _ { 2 } ( 1 + \epsilon ) \rho$ and take $I = \mathbb { R } _ { \ge 0 } , H = [ ( 1 + \epsilon / 2 ) \rho , \infty )$ in Theorem $5 . 3 .$ Since $\lambda _ { k } > ( 1 + \epsilon ) \rho$ by definition of �, Lemma A.1 gives

$$
\lambda _ { \operatorname* { m i n } } ( \hat { \Sigma } _ { \le k } ) \ge ( 1 - \epsilon / 4 ) \lambda _ { k } > ( 1 + \epsilon / 2 ) \rho
$$

with probability $1 - \exp ( - \epsilon ^ { 2 } n / c _ { 0 } )$ after adjusting constants, hence (6) holds. If $k = 0$ , we take $H = \varnothing$ and omit the head terms. Then the proof of Example 2 gives

$$
\| r ( I , H ) \| _ { \mathfrak { S } } = O ( \epsilon ^ { - 1 } ) , \quad \| r ( H , H ) \| _ { \mathfrak { S } } = 0 , \quad \| r ( I , \{ 0 \} ) \| _ { \mathfrak { S } } = O ( 1 ) .
$$

Here, note that changing the reference parameter from $\rho$ to � changes these bounds by at most constant factors by Proposition E.2. Substituting into Theorem 5.3 gives the claimed bias bound.

Finally, the variance bound in Theorem 5.3 and the ridge variance upper bound (Tsigler and Bartlett, 2023) bound the variance as

$$
\mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) \leq \frac { c \sigma ^ { 2 } } { n } \operatorname* { m i n } \left\{ k + \frac { \sum _ { i > k } \lambda _ { i } ^ { 2 } } { ( \lambda + n ^ { - 1 } \sum _ { i > k } \lambda _ { i } ) ^ { 2 } } , \ : \hat { k } \right\} \leq \frac { c \sigma ^ { 2 } } { n } \operatorname* { m i n } \{ D , \hat { k } \} ,
$$

where we used $\lambda > \rho$ and rank $( g ( \hat { \Sigma } ) ) = \hat { k }$ . Combining the bounds gives the result.

ProofofTheorem 4.1, lower bound. We apply Theorem F.1 at index $r ^ { * }$ with the constant 0.9 in the definition of $\Gamma _ { i j }$ adjusted to 0.99. For $i \leq k ^ { * }$ and $j > r ^ { * }$ , it holds that $0 . 9 9 \lambda _ { i } > \rho$ and $1 . 1 \lambda _ { j } \le \rho$ , so that $\Gamma _ { i j } = 1$ and $\Lambda _ { i } \leq \lambda _ { i } + 0 . 9 \rho \leq 2 \lambda _ { i }$ . Thus

$$
\begin{array} { r l } & { c _ { 1 } \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) \geq \displaystyle \sum _ { i \leq k ^ { * } } \frac { \lambda _ { i } } { \Lambda _ { i } ^ { 2 } } \left( \frac { 1 } { n } \displaystyle \sum _ { j > r ^ { * } } \lambda _ { j } ^ { 2 } \Gamma _ { i j } ^ { 2 } \right) { \mathbf { w } } _ { i } ^ { * 2 } + \left\| ( \mathbf { I } - g ( 1 . 1 \Sigma ) ) \mathbf { w } ^ { * } \right\| _ { \Sigma } ^ { 2 } } \\ & { \qquad \geq \displaystyle \frac { \sum _ { i > r ^ { * } } \lambda _ { i } ^ { 2 } } { 4 n } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 ; k ^ { * } } ^ { - 1 } } ^ { 2 } + \| \mathbf { w } ^ { * } \| _ { \Sigma _ { r ^ { * } ; \infty } } ^ { 2 } . } \end{array}
$$

If moreover

$$
\frac { \sum _ { i > r ^ { * } } \lambda _ { i } ^ { 2 } } { n } > \frac { 1 } { c _ { 2 } } \left( \frac { \sum _ { i > r ^ { * } } \lambda _ { i } } { n } \right) ^ { 2 } ,
$$

the claim immediately follows. Otherwise, by Theorem F.1 and

$$
\frac { \sum _ { j > r ^ { * } } \lambda _ { j } } { n } \leq 0 . 9 \rho < \lambda _ { i }
$$

for all $i \leq k ^ { * }$ , we have that

$$
c _ { 1 } \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { e r } } ) \geq \sum _ { i \leq k ^ { * } } \lambda _ { i } \left( \frac { \sum _ { j > r ^ { * } } \lambda _ { j } } { n \lambda _ { i } } \right) ^ { 2 } \mathbf { w } _ { i } ^ { * 2 } \geq \left( \frac { \sum _ { i > r ^ { * } } \lambda _ { i } } { n } \right) ^ { 2 } \Vert \mathbf { w } ^ { * } \Vert _ { \Sigma _ { 0 ; k ^ { * } } ^ { - 1 } } ^ { 2 } .
$$

In both cases, we have the lower bound

$$
c _ { 1 } \mathbb { E } \mathbf { B } \mathrm { i a s } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) \geq \alpha _ { r ^ { * } } \Vert \mathbf { w } ^ { * } \Vert _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } .
$$

The final claim on variance error follows from the lemma below, which also holds more generally under Assumption 1. □

Lemma C.1. There exist constants $c _ { 0 } , c _ { 1 } , c _ { 3 } > 1 $ , depending only on $c _ { x } , c _ { y }$ , such that the following holds. $L e t \hat { k } , k ^ { * } , r ^ { * }$ be as in Theorem 4.1. Suppose $\rho > 0 , r ^ { * } \le n / c _ { 3 }$ , and define

$$
\tilde { \rho } : = \operatorname* { m a x } \left\{ 1 . 1 \rho , c _ { 3 } \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right\} .
$$

Then,

$$
\begin{array} { r l r } & { } & { w i t h \ p r o b a b i l i t y \ a t \ l e a s t \ 1 - \exp ( - n / c _ { 0 } ) , \quad c _ { 1 } \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p e r } } ) \geq \displaystyle \frac { \sigma ^ { 2 } } { n } \# \{ j : \hat { \lambda } _ { j } > \tilde { \rho } \} ; } \\ & { } & { i n \ e x p e c t a t i o n , \quad c _ { 1 } \mathbb { E } \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p e r } } ) \geq \displaystyle \frac { \sigma ^ { 2 } } { n } \# \{ j : \lambda _ { j } > \tilde { \rho } \} . } \end{array}
$$

Proof of Lemma C.1. By the definition of ${ \tilde { \rho } } ,$

$$
\operatorname { t r } \left( \Sigma ( \Sigma + \tilde { \rho } \mathbf { I } ) ^ { - 1 } \right) = \sum _ { i } \frac { \lambda _ { i } } { \lambda _ { i } + \tilde { \rho } } \leq k ^ { * } + \frac { 1 } { \tilde { \rho } } \sum _ { i > k ^ { * } } \lambda _ { i } \leq \frac { 2 n } { c _ { 3 } } .
$$

Then after increasing $c _ { 3 } .$ , with probability at least $1 - \exp ( - n / c _ { 0 } )$ , it holds that

$$
\Sigma \preceq 2 ( \hat { \Sigma } + \tilde { \rho } \mathbf { I } ) , \quad \hat { \Sigma } \preceq \frac { 3 } { 2 } \Sigma + \frac { 1 } { 2 } \tilde { \rho } \mathbf { I } .
$$

This appears in, e.g., Hucker and Wahl (2023, Lemma 5), which is a direct corollary of Koltchinskii and Lounici (2017) applied to the concentration of the empirical covariance matrix for subgaussian random vectors with covariance matrix $( \boldsymbol { \Sigma } + \tilde { \rho } \mathbf { I } ) ^ { - 1 } \boldsymbol { \Sigma }$ . Therefore, for every pair $( \hat { \lambda } _ { j } , \hat { \mathbf { u } } _ { j } )$ ,

$$
\frac { 1 } { \hat { \lambda } _ { j } } \hat { \mathbf { u } } _ { j } ^ { \top } \pmb { \Sigma } \hat { \mathbf { u } } _ { j } \geq \frac { 1 } { 3 } \left( 2 - \frac { \tilde { \rho } } { \hat { \lambda } _ { j } } \right) _ { + } .
$$

It follows that

$$
\begin{array} { r l } & { \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) \geq \displaystyle \frac { \sigma ^ { 2 } } { c _ { y } n } \sum _ { j : \hat { \lambda } _ { j } > \rho } \frac { 1 } { \hat { \lambda } _ { j } } \hat { \mathbf { u } } _ { j } ^ { \top } \boldsymbol { \Sigma } \hat { \mathbf { u } } _ { j } } \\ & { \qquad \geq \displaystyle \frac { \sigma ^ { 2 } } { 3 c _ { y } n } \sum _ { j : \hat { \lambda } _ { j } > \rho } \left( 2 - \frac { \tilde { \rho } } { \hat { \lambda } _ { j } } \right) _ { + } } \\ & { \qquad \geq \displaystyle \frac { \sigma ^ { 2 } } { 3 c _ { y } n } \# \{ j : \hat { \lambda } _ { j } > \tilde { \rho } \} . } \end{array}
$$

Moreover for each � with $\lambda _ { j } > \tilde { \rho } ,$ , we have $j \le r ^ { * } \le n / c _ { 3 }$ . Then with probability $1 - \exp ( - n / c _ { 0 } )$ it holds that $\hat { \lambda } _ { j } \ge 0 . 9 9 \lambda _ { j } > 0 . 9 9 \tilde { \rho }$ by Lemma A.1 after increasing constants if necessary, and the second claim also follows by taking expectations. □

## C.2 Proof of Lemma 4.2

ProofofLemma 4.2. To see the thresholded index, note that

$$
1 . 1 \rho + \frac { 1 } { c _ { 2 } n } \sum _ { i > 0 } \lambda _ { i } = 1 . 1 \rho + \frac { 2 \rho + m _ { n } ( 1 \pm \epsilon ) \rho } { c _ { 2 } n } < 2 \rho = \lambda _ { 1 } ,
$$

$$
1 . 1 \rho + \frac { 1 } { c _ { 2 } n } \sum _ { i > 1 } \lambda _ { i } = 1 . 1 \rho + \frac { m _ { n } ( 1 \pm \epsilon ) \rho } { c _ { 2 } n } \geq ( 1 \pm \epsilon ) \rho = \lambda _ { 2 } ,
$$

hence $k ^ { * } = 1$ . Also, for every $r \leq m _ { n }$

$$
\lambda _ { r + 1 } + \frac { 1 } { n } \sum _ { i > r } \lambda _ { i } \ge ( 1 - \epsilon ) \rho > 0 . 9 \rho ,
$$

so $r ^ { * }$ is vacuously equal to $m _ { n } + 1$ . Then for both instances, the bias lower bound from Theorem 4.1 vanishes, while the bias upper bound is

$$
\alpha _ { 1 } \| \mathbf { w } _ { n } ^ { * } \| _ { ( \Sigma _ { n } ^ { \pm } ) _ { 0 : 1 } ^ { - 1 } } ^ { 2 } = \left( \frac { m _ { n } ( 1 \pm \epsilon ) ^ { 2 } \rho ^ { 2 } } { n } + \left( \frac { m _ { n } ( 1 \pm \epsilon ) \rho } { n } \right) ^ { 2 } \right) \frac { 1 } { 2 \rho } = \Theta ( \epsilon ^ { 2 } \rho ) .
$$

We now derive the actual risk of both instances. First consider $\Sigma _ { n } ^ { - }$ . Let

$$
{ \bf X } _ { n } = { \bf Y } _ { n } ( \Sigma _ { n } ^ { - } ) ^ { 1 / 2 } , \quad { \bf Y } _ { n } = [ { \bf z } _ { 1 } { \bf Z } ] ,
$$

where $\mathbf { Y } _ { n } \in \mathbb { R } ^ { n \times ( m _ { n } + 1 ) }$ has independent standard Gaussian entries. In the population eigenbasis,

$$
\hat { \mathbf { \Sigma } } _ { n } = \left[ \begin{array} { c c } { h } & { \mathbf { G } ^ { \top } } \\ { \mathbf { G } } & { \mathbf { T } } \end{array} \right] , \quad \mathbf { G } = \frac { \sqrt { 2 ( 1 - \epsilon ) } \rho } { n } \mathbf { Z } ^ { \top } \mathbf { z } _ { 1 } .
$$

By standard concentration, with probability at least $1 - \exp ( - \epsilon ^ { 2 } n / c _ { 0 } )$ ，

$$
\begin{array} { r } { ( 1 - \epsilon ) \Sigma _ { n } ^ { - } \preceq \hat { \Sigma } _ { n } \preceq ( 1 + \epsilon ) \Sigma _ { n } ^ { - } . } \end{array}
$$

On this event, $\| \mathbf { T } \| \leq ( 1 - \epsilon ^ { 2 } ) \rho < \rho$ and $h \geq 2 ( 1 - \epsilon ) \rho \geq 1 . 9 \rho$ , hence exactly one eigenvalue is selected by PCR. Let � be this eigenvalue and let $( a , \mathbf { b } )$ be its unit eigenvector, with $a \geq 0$ . Then

$$
\mathbf { b } = \left( \lambda \mathbf { I } - \mathbf { T } \right) ^ { - 1 } \mathbf { G } a .
$$

Since $\hat { \Sigma } _ { n } \succeq 0 .$ , it holds that $\mathbf { G } \mathbf { G } ^ { \top } \preceq h \mathbf { T }$ and $\lambda \ge h \ge 1 . 9 \rho$ . Then

$$
{ \frac { \| \mathbf { b } \| } { a } } \leq { \frac { \| \mathbf { G } \| } { \lambda - \rho } } \leq { \frac { \sqrt { h \rho } } { h - \rho } } < 2 \quad \Longrightarrow \quad a ^ { 2 } \geq { \frac { 1 } { 5 } }
$$

and

$$
\| \mathbf { b } \| ^ { 2 } = a ^ { 2 } \mathbf { G } ^ { \top } ( \lambda \mathbf { I } - \mathbf { T } ) ^ { - 2 } \mathbf { G } \geq \frac { \| \mathbf { G } \| ^ { 2 } } { 5 \lambda ^ { 2 } } .
$$

Moreover with probability at least $1 - \exp ( - \epsilon ^ { 2 } n / c _ { 0 } )$

$$
\| \mathbf { G } \| ^ { 2 } = \frac { 2 ( 1 - \epsilon ) \rho ^ { 2 } } { n ^ { 2 } } \| \mathbf { Z } ^ { \top } \mathbf { z } _ { 1 } \| ^ { 2 } \geq \frac { 2 ( 1 - \epsilon ) \rho ^ { 2 } } { n ^ { 2 } } \frac { m _ { n } n } { 2 } = \Theta ( \epsilon ^ { 2 } \rho ^ { 2 } ) .
$$

Hence PCR satisfies

$$
\begin{array} { r l } & { \mathrm { B i a s } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) = \| ( \mathbf { I } - ( a , \mathbf { b } ) ( a , \mathbf { b } ) ^ { \top } ) \mathbf { e } _ { 1 } \| _ { \Sigma _ { n } ^ { - } } ^ { 2 } } \\ & { \qquad = 2 \rho ( 1 - a ^ { 2 } ) ^ { 2 } + ( 1 - \epsilon ) \rho a ^ { 2 } \| \mathbf { b } \| ^ { 2 } } \\ & { \qquad \geq c \rho \frac { \| \mathbf { G } \| ^ { 2 } } { \lambda ^ { 2 } } = \Theta ( \epsilon ^ { 2 } \rho ) , } \end{array}
$$

where we also used $\lambda \le 2 ( 1 + \epsilon ) \rho$ . Thus the upper bound is tight in this case.

On the other hand, under $\Sigma _ { n } ^ { + }$ , with probability at least $1 - \exp ( - \epsilon ^ { 2 } n / c _ { 0 } )$

$$
\hat { \Sigma } _ { n } \succeq \left( 1 - \frac { \epsilon } { 2 } \right) \Sigma _ { n } ^ { + } \succeq \left( 1 - \frac { \epsilon } { 2 } \right) ( 1 + \epsilon ) \rho \mathbf { I } > \rho \mathbf { I } .
$$

Thus PCR is equivalent to OLS. Since $m _ { n } < n , \mathbf { X } ^ { \top } \mathbf { X }$ has full rank almost surely, so $\mathrm { B i a s } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) = 0$ and the lower bound is tight in this case. □

## D Proofs for PCR Dominance

Additional notation. Let � be a monotone shrinkage filter. We define

$$
\mathbf { G } : = g ( \widehat { \Sigma } ) , \quad \mathbf { B } : = ( \mathbf { I } - \mathbf { G } ) \Sigma ( \mathbf { I } - \mathbf { G } ) , \quad p _ { i } : = \mathbb { E } \mathbf { e } _ { i } ^ { \top } \mathbf { G } \mathbf { e } _ { i } ,
$$

and set $\mathbf { G } _ { i j } : = \mathbf { e } _ { i } ^ { \top } \mathbf { G } \mathbf { e } _ { j } , \mathbf { B } _ { i j } : = \mathbf { e } _ { i } ^ { \top } \mathbf { B } \mathbf { e } _ { j } .$ . Additionally denote

$$
\Psi : = \frac 1 n \psi ( { \bf A } / n ) , \quad { \bf A } _ { - i } : = \sum _ { j \ne i } \lambda _ { j } { \bf z } _ { j } { \bf z } _ { j } ^ { \top } , \quad { \bf V } _ { - i } : = \sum _ { j \ne i } \lambda _ { j } ^ { 2 } { \bf z } _ { j } { \bf z } _ { j } ^ { \top } .
$$

We assume $c = 1$ for simplicity; otherwise, all variance upper and lower bounds are scaled by a factor of at most $c ^ { 2 }$ and $1 / c ^ { 2 }$ , respectively. The bias-variance decomposition can be written as

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) | \mathbf { X } ] = \| ( \mathbf { I } - \boldsymbol { \psi } ( \hat { \Sigma } ) \hat { \mathbf { x } } ^ { * } \| _ { \Sigma } ^ { 2 } + \frac { \sigma ^ { 2 } } { n ^ { 2 } } \operatorname { t r } ( \mathbf { X } \boldsymbol { \psi } ( \hat { \Sigma } ) \Sigma \boldsymbol { \psi } ( \hat { \Sigma } ) \mathbf { X } ^ { \top } ) } \\ { = \| ( \mathbf { I } - \mathbf { G } ) \mathbf { w } ^ { * } \| _ { \Sigma } ^ { 2 } + \frac { \sigma ^ { 2 } } { n } \operatorname { t r } ( \Sigma \hat { \Sigma } ^ { - 1 } \mathbf { G } ^ { 2 } ) . } \end{array}
$$

We thus denote, conditional on X,

$$
\begin{array} { c } { \displaystyle \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g } ) : = \frac { \sigma ^ { 2 } } { n } \mathrm { t r } ( { \boldsymbol \Sigma } \hat { \boldsymbol \Sigma } ^ { - 1 } \mathbf G ^ { 2 } ) , } \\ { \displaystyle \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) : = \| ( \mathbf I - \mathbf G ) \mathbf { w } ^ { * } \| _ { \boldsymbol \Sigma } ^ { 2 } . } \end{array}
$$

By sign symmetry, it holds in expectation that

$$
\mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) = \sum _ { i } \mathbf { w } _ { i } ^ { * 2 } \mathbb { E } \mathbf { B } _ { i i } .
$$

We use the following equality multiple times in the proofs, which is obtained by direct expansion:

$$
\mathbf { B } _ { i i } = \lambda _ { i } ( 1 - \mathbf { G } _ { i i } ) ^ { 2 } + \lambda _ { i } \sum _ { j \neq i } \lambda _ { j } ^ { 2 } ( \mathbf { z } _ { j } ^ { \top } \Psi \mathbf { z } _ { i } ) ^ { 2 } .
$$

We also define the efective bias for general filters as

$$
\mathrm { E f f e c t i v e B i a s } ( \hat { \mathbf { w } } _ { g } ) : = \sum _ { i } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } ( 1 - \mathbf { G } _ { i i } ) ^ { 2 } .
$$

From the above, it is clear that in expectation, $\mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) \geq \mathbb { E } \mathrm { E f f e c t i v e B i a s } ( \hat { \mathbf { w } } _ { g } )$

Fixed design. For completeness, we first provide a short deterministic proof of PCR dominance in the fixed design setting, following the argument of Dhillon et al. (2013).

Lemma D.1 (PCR dominates monotone filters for fixed design). Let X be fixed and suppose $\mathbb { E } [ \mathbf { y } \mid \mathbf { X } ] = \mathbf { X } \mathbf { w } ^ { * }$ $\mathbb { E } [ \| \mathbf { y } - \mathbf { X } \mathbf { w } ^ { * } \| ^ { 2 } | \mathbf { X } ] < \infty$ . Denote the excess risk by $\mathcal { E } _ { \mathbf { X } } ( \mathbf { w } ) : = \| \mathbf { w } - \mathbf { w } ^ { * } \| _ { \hat { \Sigma } } ^ { 2 } .$ For every monotone filter $g \in { \mathcal { G } }$ there exists $\rho \geq 0 ,$ , depending only on $\mathbf { X } , g ,$ , such that

$$
\mathbb { E } [ \mathcal { E } _ { \mathbf { X } } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) | \mathbf { X } ] \leq 4 \mathbb { E } [ \mathcal { E } _ { \mathbf { X } } ( \hat { \mathbf { w } } _ { g } ) | \mathbf { X } ] .
$$

ProofofLemma D.1. Write $\begin{array} { r } { \hat { \Sigma } = \Sigma = \sum _ { i } \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } } \end{array}$ and ${ \bf y } - { \bf X } { \bf w } ^ { * } = \varepsilon , { \bf v } _ { i } = { \bf X } { \bf u } _ { i } / \sqrt { n \lambda _ { i } }$ . For all $g \in { \mathcal { G } }$ , the bias-variance decomposition gives

$$
\mathbb { E } [ \mathcal { E } _ { { \mathbf { X } } } ( \hat { \mathbf { w } } _ { g } ) | { \mathbf { X } } ] = \sum _ { i : \lambda _ { i } > 0 } \lambda _ { i } ( 1 - g ( \lambda _ { i } ) ) ^ { 2 } { \mathbf { w } } _ { i } ^ { * 2 } + \frac { 1 } { n } \sum _ { i } g ( \lambda _ { i } ) ^ { 2 } \mathbb { E } [ ( { \mathbf { v } } _ { i } ^ { \top } \boldsymbol { \varepsilon } ) ^ { 2 } | { \mathbf { X } } ] .
$$

Let the PCR filter be $g _ { \rho } ( z ) : = \mathbf { 1 } \{ z > \rho \}$ and choose

$$
\rho : = \operatorname* { m a x } \left( \{ 0 \} \cup \{ \lambda _ { i } : \lambda _ { i } > 0 , \ g ( \lambda _ { i } ) \leq 1 / 2 \} \right) .
$$

By monotonicity, it holds for all � that $g _ { \rho } ( \lambda _ { i } ) = 1 \{ g ( \lambda _ { i } ) > 1 / 2 \}$ , and so

$$
g _ { \rho } ( \lambda _ { i } ) ^ { 2 } \leq 4 g ( \lambda _ { i } ) ^ { 2 } , \quad ( 1 - g _ { \rho } ( \lambda _ { i } ) ) ^ { 2 } \leq 4 ( 1 - g ( \lambda _ { i } ) ) ^ { 2 } .
$$

Substituting into the above proves the claim.

## D.1 Useful lemmas

We first show a spectral result involving the leave-one-out matrix $\mathbf { A } _ { - i }$

Lemma D.2. Let � be a monotonefilter and let � be any index with $\lambda _ { i } > 0 .$ . Choose orthonormal eigenbases $( \beta _ { j } , \mathbf { v } _ { j } ) _ { 1 \leq j \leq n } , ( \gamma _ { j } , \mathbf { w } _ { j } ) _ { 1 \leq j \leq n }$ such that

$$
\frac { \mathbf { A } } { n } = \sum _ { j = 1 } ^ { n } \beta _ { j } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } , \quad \frac { \mathbf { A } _ { - i } } { n } = \sum _ { j = 1 } ^ { n } \gamma _ { j } \mathbf { w } _ { j } \mathbf { w } _ { j } ^ { \top } ,
$$

and define

$$
h _ { g , j } ( \mathbf { z } _ { i } ) : = n \frac { \mathbf { z } _ { i } ^ { \top } \Psi \mathbf { w } _ { j } } { \mathbf { z } _ { i } ^ { \top } \mathbf { w } _ { j } } \mathbf { 1 } \{ \mathbf { z } _ { i } ^ { \top } \mathbf { w } _ { j } \neq 0 \} .
$$

Then conditional on $\{ \mathbf { z } _ { j } : j \neq i \}$ , it holds that $h _ { g , j } ( \mathbf { z } _ { i } ) \geq 0$ and $h _ { g , j }$ is invariant under sign changes in the basis $\left( \mathbf { w } _ { j } \right)$ . Moreover, for all $\lambda > 0$

$$
h _ { g , j } ( \mathbf { z } _ { i } ) \geq \frac { ( g ( \gamma _ { j } ) - g ( \lambda ) ) _ { + } } { \gamma _ { j } } \left( 1 - \frac { \lambda _ { i } \| \mathbf { z } _ { i } \| ^ { 2 } } { \lambda n } \right) ,
$$

where the right-hand side is interpreted as zero $i f \gamma _ { j } = 0 .$

ProofofLemma D.2. Note that $\mathbf { z } _ { i } ^ { \top } \mathbf { w } _ { j } \neq 0$ for every � almost surely; we work under this event. From

$$
\begin{array} { r l r } {  { ( \beta _ { k } - \gamma _ { j } ) \mathbf { v } _ { k } ^ { \top } \mathbf { w } _ { j } = \mathbf { v } _ { k } ^ { \top } \frac { \mathbf { A } } { n } \mathbf { w } _ { j } - \mathbf { v } _ { k } ^ { \top } \frac { \mathbf { A } _ { - i } } { n } \mathbf { w } _ { j } } } \\ & { } & { = \frac { \lambda _ { i } } { n } ( \mathbf { v } _ { k } ^ { \top } \mathbf { z } _ { i } ) ( \mathbf { z } _ { i } ^ { \top } \mathbf { w } _ { j } ) , } \end{array}
$$

we see that $\beta _ { k } = \gamma _ { j }$ implies $\mathbf { v } _ { k } ^ { \top } \mathbf { z } _ { i } = 0$ . Since $\mathbf { A } / n \succeq ( \lambda _ { i } / n ) \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \top } , \beta _ { k } = 0$ also implies $\mathbf { v } _ { k } ^ { \top } \mathbf { z } _ { i } = 0 .$ . Otherwise,

$$
\frac { ( \mathbf { z } _ { i } ^ { \top } \mathbf { v } _ { k } ) ( \mathbf { v } _ { k } ^ { \top } \mathbf { w } _ { j } ) } { \mathbf { z } _ { i } ^ { \top } \mathbf { w } _ { j } } = \frac { \lambda _ { i } } { n } \frac { ( \mathbf { v } _ { k } ^ { \top } \mathbf { z } _ { i } ) ^ { 2 } } { \beta _ { k } - \gamma _ { j } } , \quad \beta _ { k } \notin \{ \gamma _ { j } , 0 \} .
$$

Hence summing over � gives the identity

$$
\frac { \lambda _ { i } } { n } \sum _ { k : \beta _ { k } > 0 , \beta _ { k } \neq \gamma _ { j } } \frac { ( \mathbf { v } _ { k } ^ { \top } \mathbf { z } _ { i } ) ^ { 2 } } { \beta _ { k } - \gamma _ { j } } = 1 .
$$

From the layer-cake representation

$$
\psi ( \mathbf { A } / n ) = \int _ { 0 } ^ { 1 } ( \mathbf { A } / n ) ^ { - 1 } \mathbf { 1 } \{ g ( \mathbf { A } / n ) > s \mathbf { I } \} \mathrm { d } s ,
$$

we also have

$$
h _ { g , j } ( \mathbf { z } _ { i } ) = \int _ { 0 } ^ { 1 } h _ { g , j } ( \mathbf { z } _ { i } ; s ) \mathrm { d } s
$$

where

$$
\begin{array} { r } { h _ { g , j } ( \mathbf { z } _ { i } ; s ) : = \frac { \mathbf { z } _ { i } ^ { \top } ( \mathbf { A } / n ) ^ { - 1 } \mathbf { 1 } \{ g ( \mathbf { A } / n ) > s \mathbf { I } \} \mathbf { w } _ { j } } { \mathbf { z } _ { i } ^ { \top } \mathbf { w } _ { j } } } \\ { = \frac { \lambda _ { i } } { n } \displaystyle \sum _ { \begin{array} { c } { k : g ( \beta _ { k } ) > s } \\ { \beta _ { k } > 0 , \beta _ { k } \neq \gamma _ { j } } \end{array} } \frac { ( \mathbf { v } _ { k } ^ { \top } \mathbf { z } _ { i } ) ^ { 2 } } { \beta _ { k } ( \beta _ { k } - \gamma _ { j } ) } . } \end{array}
$$

When $\begin{array} { r } { \gamma _ { j } = 0 , h _ { g , j } ( \mathbf { z } _ { i } ; s ) \ge 0 . } \end{array}$ , so $h _ { g , j } ( \mathbf { z } _ { i } ) \geq 0$ . Suppose $\gamma _ { j } > 0$ and fix $s \in ( 0 , 1 )$ . $\begin{array} { r } { \mathrm { I f } \ g ( \gamma _ { j } ) \leq s , } \end{array}$ then every eigenvalue $\beta _ { k }$ included in the above sum satisfies $g ( \beta _ { k } ) > s \geq g ( \gamma _ { j } )$ , so $\beta _ { k } > \gamma _ { j }$ and the sum is nonnegative. If instead $g ( \gamma _ { j } ) > s ,$ every omitted $\beta _ { k }$ (except possibly $\gamma _ { j } , 0 )$ satisfies $\beta _ { k } < \gamma _ { j }$ and contributes a negative summand, hence

$$
\begin{array} { r l r } {  { h _ { g , j } ( \mathbf { z } _ { i } ; s ) \geq \frac { \lambda _ { i } } { n } \sum _ { k : \beta _ { k } > 0 , \beta _ { k } \neq \gamma _ { j } } \frac { ( \mathbf { v } _ { k } ^ { \top } \mathbf { z } _ { i } ) ^ { 2 } } { \beta _ { k } ( \beta _ { k } - \gamma _ { j } ) } } } \\ & { } & { = \frac { \lambda _ { i } } { n } \sum _ { k : \beta _ { k } > 0 , \beta _ { k } \neq \gamma _ { j } } \frac { 1 } { \gamma _ { j } } ( \frac { 1 } { \beta _ { k } - \gamma _ { j } } - \frac { 1 } { \beta _ { k } } ) ( \mathbf { v } _ { k } ^ { \top } \mathbf { z } _ { i } ) ^ { 2 } } \\ & { } & { = \frac { 1 } { \gamma _ { j } } \big ( 1 - \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \mathbf { A } ^ { - 1 } \mathbf { z } _ { i } \big ) \geq 0 . } \end{array}
$$

Therefore integrating over $s ,$ it follows that $h _ { g , j } ( \mathbf { z } _ { i } ) \geq 0$

Moreover if $g ( \lambda ) < s < g ( \gamma _ { j } )$ , every included eigenvalue also satisfies $\beta _ { k } > \lambda$ , hence

$$
\begin{array} { l } { { \displaystyle h _ { g , j } ( { \bf z } _ { i } ; s ) = \frac { \lambda _ { i } } { n } \sum _ { { \boldsymbol k } : { \boldsymbol g } ( { \boldsymbol \beta } _ { k } ) > s \atop { \boldsymbol \beta _ { k } > 0 , { \boldsymbol \beta } _ { k } \neq \boldsymbol \gamma _ { j } } } \frac { 1 } { \gamma _ { j } } \left( \frac { 1 } { \beta _ { k } - \gamma _ { j } } - \frac { 1 } { \beta _ { k } } \right) ( \mathbf { v } _ { k } ^ { \top } \mathbf { z } _ { i } ) ^ { 2 } } } \\ { { \displaystyle \ \geq \frac { 1 } { \gamma _ { j } } \left( 1 - \frac { \lambda _ { i } \| { \bf z } _ { i } \| ^ { 2 } } { \lambda n } \right) } . } \end{array}
$$

It follows that

$$
h _ { g , j } ( \mathbf { z } _ { i } ) \geq \frac { ( g ( \gamma _ { j } ) - g ( \lambda ) ) _ { + } } { \gamma _ { j } } \left( 1 - \frac { \lambda _ { i } \| \mathbf { z } _ { i } \| ^ { 2 } } { \lambda n } \right) .
$$

Finally, let S be an arbitrary diagonal sign matrix in the eigenbasis $( \mathbf { w } _ { j } )$ . The transformation $\mathbf { z } _ { i } \mapsto \mathbf { S } \mathbf { z } _ { i }$ preserves the law of $\mathbf { z } _ { i }$ and maps A to SAS, hence

$$
h _ { g , j } ( \mathbf { S } \mathbf { z } _ { i } ) = n \frac { ( \mathbf { S } \mathbf { z } _ { i } ) ^ { \top } \mathbf { S } \Psi \mathbf { S } \mathbf { w } _ { j } } { ( \mathbf { S } \mathbf { z } _ { i } ) ^ { \top } \mathbf { w } _ { j } } = h _ { g , j } ( \mathbf { z } _ { i } ) .
$$

This completes the proof.

Using this lemma, we prove the following ‘gluing’ inequality for spectral filters.

Lemma $\mathbf { D . 3 } .$ . Suppose $\begin{array} { r } { g = \sum _ { k = 1 } ^ { m } a _ { k } g _ { k } } \end{array}$ where $g _ { 1 } , \ldots , g _ { m } : \mathbb { R } _ { \geq 0 }  [ 0 , 1 ]$ are nondecreasing and $a _ { k } \geq 0 ,$ $\textstyle \sum _ { k = 1 } ^ { m } a _ { k } \leq 1$ . Then

$$
\mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) \geq \sum _ { k = 1 } ^ { m } a _ { k } ^ { 2 } \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g _ { k } } ) .
$$

Proof of Lemma D.3. Denote $\begin{array} { r } { \psi _ { k } ( z ) = g _ { k } ( z ) / z , \Psi _ { k } : = \frac { 1 } { n } \psi _ { k } ( \mathbf { A } / n ) } \end{array}$ and $\mathbf { G } _ { k } : = g _ { k } ( \hat { \Sigma } )$ , so that

$$
\mathbf { G } _ { k , i j } = \mathbf { e } _ { i } ^ { \top } \mathbf { X } ^ { \top } \Psi _ { k } \mathbf { X } \mathbf { e } _ { j } = \sqrt { \lambda _ { i } \lambda _ { j } } \mathbf { z } _ { i } ^ { \top } \Psi _ { k } \mathbf { z } _ { j } .
$$

We may additionally suppose $\textstyle \sum _ { k = 1 } ^ { m } a _ { k } = 1$ as otherwise we can add a dummy filter $g _ { m + 1 } = 0$

We claim that

$$
\begin{array} { r } { \mathbb { E } \mathbf { G } _ { k , i j } \mathbf { G } _ { \ell , i j } \ge 0 , \quad i , j \ge 1 , \quad k , \ell \le m . } \end{array}\tag{13}
$$

We may suppose $i \neq j$ and $\lambda _ { i } , \lambda _ { j } > 0$ and condition on $\{ \mathbf { z } _ { p } : p \neq i \}$ . From Lemma D.2, it holds that

$$
h _ { k , p } ( \mathbf { z } _ { i } ) : = h _ { g _ { k } , p } ( \mathbf { z } _ { i } ) : = n \frac { \mathbf { z } _ { i } ^ { \top } \Psi _ { k } \mathbf { w } _ { p } } { \mathbf { z } _ { i } ^ { \top } \mathbf { w } _ { p } } \mathbf { 1 } \{ \mathbf { z } _ { i } ^ { \top } \mathbf { w } _ { p } \neq 0 \} \geq 0 ,
$$

and similarly $h _ { \ell , p } ( { \bf z } _ { i } ) \geq 0$ . Moreover, let S be an arbitrary diagonal sign matrix in the eigenbasis $( \mathbf { w } _ { j } )$ , then both functions are invariant under the transformation $\mathbf { z } _ { i } \mapsto \mathbf { S } \mathbf { z } _ { i }$ . Averaging over all choices of S gives (noting that $| \mathbf { z } _ { i } ^ { \top } \Psi _ { k } \mathbf { z } _ { j } | \leq ( \lambda _ { i } \lambda _ { j } ) ^ { - 1 / 2 }$ is conditionally integrable)

$$
\begin{array} { r l } & { \mathbb { E } _ { { \mathbf z } _ { i } } { \mathbf { G } } _ { k , i j } { \mathbf { G } } _ { \ell , i j } = \frac { \lambda _ { i } \lambda _ { j } } { n ^ { 2 } } \frac { 1 } { 2 ^ { n } } \displaystyle \sum _ { { \mathbf { S } } } \mathbb { E } _ { { \mathbf z } _ { i } } ( { \mathbf { S } } { \mathbf { z } } _ { i } ) ^ { \top } \boldsymbol { \psi } _ { k } ( { \mathbf { S } } { \mathbf { A } } { \mathbf { S } } / n ) { \mathbf { z } } _ { j } ( { \mathbf { S } } { \mathbf { z } } _ { i } ) ^ { \top } \boldsymbol { \psi } _ { \ell } ( { \mathbf { S } } { \mathbf { A } } { \mathbf { S } } / n ) { \mathbf { z } } _ { j } } \\ & { \qquad = \frac { \lambda _ { i } \lambda _ { j } } { n ^ { 2 } } \displaystyle \sum _ { p } \mathbb { E } _ { { \mathbf z } _ { i } } ( { \mathbf { w } } _ { p } ^ { \top } { \mathbf { z } } _ { i } ) ^ { 2 } ( { \mathbf { w } } _ { p } ^ { \top } { \mathbf { z } } _ { j } ) ^ { 2 } h _ { k , p } ( { \mathbf { z } } _ { i } ) h _ { \ell , p } ( { \mathbf { z } } _ { i } ) \ge 0 . } \end{array}
$$

Taking expectations over the remaining columns proves (13).

From (13), it now follows that for all $i \geq 1$ and $k \neq \ell ,$

$$
\mathbb { E } \mathbf { e } _ { i } ^ { \top } \left( \mathbf { I } - \mathbf { G } _ { k } \right) \boldsymbol { \Sigma } ( \mathbf { I } - \mathbf { G } _ { \ell } ) \mathbf { e } _ { i } = \lambda _ { i } \mathbb { B } ( \mathbf { e } _ { i } ^ { \top } \left( \mathbf { I } - \mathbf { G } _ { k } \right) \mathbf { e } _ { i } ) ( \mathbf { e } _ { i } ^ { \top } ( \mathbf { I } - \mathbf { G } _ { \ell } ) \mathbf { e } _ { i } ) + \sum _ { j \neq i } \lambda _ { j } \mathbb { B } ( \mathbf { e } _ { i } ^ { \top } \mathbf { G } _ { k } \mathbf { e } _ { j } ) ( \mathbf { e } _ { i } ^ { \top } \mathbf { G } _ { \ell } \mathbf { e } _ { j } ) \geq 0 .
$$

Since $\begin{array} { r } { \mathbf { I } - \mathbf { G } = \sum _ { k = 1 } ^ { m } a _ { k } ( \mathbf { I } - \mathbf { G } _ { k } ) } \end{array}$ , we thus have

$$
\begin{array} { r l } { \mathbb { B } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { R } ) = } & { \mathbb { B } \lVert ( \mathbf { I } - \mathbf { G } ) \mathbf { w } ^ { * } \rVert _ { \Sigma } ^ { 2 } } \\ & { = \displaystyle \sum _ { i } \sum _ { k , \ell = 1 } ^ { m } a _ { k } a _ { \ell } \mathrm { w } _ { i } ^ { * } \mathbb { E } \mathrm { e } _ { i } ^ { \tau } ( \mathbf { I } - \mathbf { G } _ { k } ) \Sigma ( \mathbf { I } - \mathbf { G } _ { \ell } ) \mathbf { e } _ { i } } \\ & { \geq \displaystyle \sum _ { i } \sum _ { k = 1 } ^ { m } a _ { k } ^ { 2 } \mathrm { w } _ { i } ^ { * } \mathrm { Z E } \mathrm { e } _ { i } ^ { \tau } ( \mathbf { I } - \mathbf { G } _ { k } ) \Sigma ( \mathbf { I } - \mathbf { G } _ { k } ) \mathrm { e } _ { i } } \\ & { = \displaystyle \sum _ { k = 1 } ^ { m } a _ { k } ^ { 2 } \mathbb { E } \lVert ( \mathbf { I } - \mathbf { G } _ { k } ) \mathbf { w } ^ { * } \rVert _ { \Sigma } ^ { 2 } } \\ & { = \displaystyle \sum _ { k = 1 } ^ { m } a _ { k } ^ { 2 } \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { \ell k } ) . } \end{array}
$$

Moreover the expected variance satisfies

$$
\begin{array} { l } { { \displaystyle \mathbb E \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g } ) = \frac { \sigma ^ { 2 } } { n } \mathbb E \mathrm { t r } ( \boldsymbol { \Sigma } \hat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { G } ^ { 2 } ) } \ ~ } \\ { { \displaystyle ~ = \frac { \sigma ^ { 2 } } { n } \sum _ { k , \ell = 1 } ^ { m } a _ { k } a _ { \ell } \mathbb E \mathrm { t r } ( \boldsymbol { \Sigma } \hat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf G _ { k } \mathbf G _ { \ell } ) } \ ~ } \\ { { \displaystyle ~ \geq \frac { \sigma ^ { 2 } } { n } \sum _ { k = 1 } ^ { m } a _ { k } ^ { 2 } \mathbb E \mathrm { t r } ( \boldsymbol { \Sigma } \hat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf G _ { k } ^ { 2 } ) } } \end{array}
$$

$$
\mathbf { \Sigma } = \sum _ { k = 1 } ^ { m } a _ { k } ^ { 2 } \mathbb { E } \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g _ { k } } ) .
$$

Here, we have used that the factors of $\hat { \mathbf { \boldsymbol { \Sigma } } } ^ { - 1 } \mathbf { G } _ { k } \mathbf { G } _ { \ell }$ are commuting nonnegative functions of $\hat { \Sigma } .$ , hence it is positive semidefinite. The proof is complete. □

We also show the following entrywise lower bound for the of-diagonals of G, which is based on an application of the Cramér–Rao inequality as discussed at the end of the proof.

Lemma D.4. The sequence $( p _ { i } ) _ { i \geq 1 }$ is nonincreasing, with $p _ { i } = p _ { j }$ when $\lambda _ { i } = \lambda _ { j } . \ I f \lambda _ { i } \neq \lambda _ { j }$ , then

$$
\mathbb { E } \mathbf { G } _ { i j } ^ { 2 } \geq \frac { \lambda _ { i } \lambda _ { j } } { n ( \lambda _ { i } - \lambda _ { j } ) ^ { 2 } } ( p _ { i } - p _ { j } ) ^ { 2 } .\tag{14}
$$

ProofofLemma $D . 4 .$ . By monotone approximation, it sufices to prove the lemma when $g$ is smooth. Let $\Phi : \mathbb { R } _ { > 0 } \to \mathbb { R }$ be an antiderivative of � and define

$$
F ( \theta ) : = \mathbb { E } \operatorname { t r } \big ( \Phi ( \mathbf { A } _ { \theta } / n ) \big ) , \quad \mathrm { w h e r e } \quad \mathbf { A } _ { \theta } : = \sum _ { i } e ^ { \theta _ { i } } \mathbf { z } _ { i } \mathbf { z } _ { i } ^ { \mathsf { T } } .
$$

Here, the sum runs over indices � such that $\lambda _ { i } > 0$ , so that $\mathbf { A } _ { \theta } = \mathbf { A }$ when $\theta _ { i } = \log \lambda _ { i }$

We will show that $F$ is convex. Letting � vary along an afine direction parametrized by � with $\begin{array} { r } { \dot { \theta } = \frac { \mathrm { d } \theta } { \mathrm { d } t } } \end{array}$ constant,

$$
\dot { \bf A } _ { \theta } = \sum _ { i } \dot { \theta } _ { i } e ^ { \theta _ { i } } { \bf z } _ { i } { \bf z } _ { i } ^ { \top } , \quad \ddot { \bf A } _ { \theta } = \sum _ { i } \dot { \theta } _ { i } ^ { 2 } e ^ { \theta _ { i } } { \bf z } _ { i } { \bf z } _ { i } ^ { \top } .
$$

Since ran $\dot { \mathbf { A } } _ { \theta } \subseteq$ ran $\mathbf { A } _ { \theta } ,$ , it holds for every $\mathbf { v } \in \mathbb { R } ^ { n }$

$$
\begin{array} { r l } & { \mathbf { v } ^ { \top } \dot { \mathbf { A } } _ { \theta } \mathbf { A } _ { \theta } ^ { - 1 } \dot { \mathbf { A } } _ { \theta } \mathbf { v } = \underset { u \in \mathbb { R } ^ { n } } { \operatorname* { s u p } } 2 \mathbf { u } ^ { \top } \dot { \mathbf { A } } _ { \theta } \mathbf { v } - \mathbf { u } ^ { \top } \mathbf { A } _ { \theta } \mathbf { u } } \\ & { \qquad = \underset { \mathbf { u } \in \mathbb { R } ^ { n } } { \operatorname* { s u p } } \displaystyle \sum _ { i } e ^ { \theta _ { i } } \big ( 2 \dot { \theta } _ { i } ( \mathbf { z } _ { i } ^ { \top } \mathbf { v } ) ( \mathbf { z } _ { i } ^ { \top } \mathbf { u } ) - ( \mathbf { z } _ { i } ^ { \top } \mathbf { u } ) ^ { 2 } \big ) } \\ & { \qquad \leq \displaystyle \sum _ { i } \dot { \theta } _ { i } ^ { 2 } e ^ { \theta _ { i } } ( \mathbf { z } _ { i } ^ { \top } \mathbf { v } ) ^ { 2 } = \mathbf { v } ^ { \top } \ddot { \mathbf { A } } _ { \theta } \mathbf { v } , } \end{array}
$$

where the inequality follows from $2 a b - b ^ { 2 } \leq a ^ { 2 }$ . Hence

$$
\begin{array} { r } { \ddot { \mathbf { A } } _ { \theta } \succeq \dot { \mathbf { A } } _ { \theta } \mathbf { A } _ { \theta } ^ { - 1 } \dot { \mathbf { A } } _ { \theta } . } \end{array}\tag{15}
$$

At a fixed $t ,$ let $a _ { k } > 0$ be the positive eigenvalues of $\mathbf { A } _ { \theta } / n ,$ and write matrix entries in the corresponding eigenbasis. Diferentiating gives

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } t } \operatorname { t r } \left( \Phi ( \mathbf { A } _ { \theta } / n ) \right) = \displaystyle \frac { 1 } { n } \operatorname { t r } \left( \psi ( \mathbf { A } _ { \theta } / n ) \dot { \mathbf { A } } _ { \theta } \right) , } \\ { \displaystyle \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } t ^ { 2 } } \operatorname { t r } \left( \Phi ( \mathbf { A } _ { \theta } / n ) \right) = \displaystyle \frac { 1 } { n } \sum _ { k } \psi ( a _ { k } ) \ddot { \mathbf { A } } _ { \theta , k k } + \frac { 1 } { n ^ { 2 } } \sum _ { k } \psi ^ { \prime } ( a _ { k } ) \dot { \mathbf { A } } _ { \theta , k k } ^ { 2 } + \frac { 2 } { n ^ { 2 } } \sum _ { k < \ell } \frac { \psi ( a _ { k } ) - \psi ( a _ { \ell } ) } { a _ { k } - a _ { \ell } } \dot { \mathbf { A } } _ { \theta , k \ell } ^ { 2 } , } \end{array}
$$

where the divided diference is interpreted as $\psi ^ { \prime } ( a _ { k } )$ when $a _ { k } = a _ { \ell }$ . From (15), we can bound

$$
\begin{array} { r l } { \displaystyle \sum _ { k } \psi ( a _ { k } ) \ddot { \mathbf { A } } _ { \theta , k k } = \mathrm { t r } \left( \psi ( \mathbf { A } _ { \theta } / n ) \ddot { \mathbf { A } } _ { \theta } \right) } & { } \\ { \displaystyle \geq \mathrm { t r } \left( \psi ( \mathbf { A } _ { \theta } / n ) \dot { \mathbf { A } } _ { \theta } \mathbf { A } _ { \theta } ^ { - 1 } \dot { \mathbf { A } } _ { \theta } \right) } \end{array}
$$

$$
= \frac { 1 } { n } \sum _ { k } \frac { \psi ( a _ { k } ) } { a _ { k } } \dot { \mathbf { A } } _ { \theta , k k } ^ { 2 } + \frac { 1 } { n } \sum _ { k < \ell } \left( \frac { \psi ( a _ { k } ) } { a _ { \ell } } + \frac { \psi ( a _ { \ell } ) } { a _ { k } } \right) \dot { \mathbf { A } } _ { \theta , k \ell } ^ { 2 } .
$$

Combining the preceding displays and using that $\psi ( z ) = g ( z ) / z$

$$
\begin{array} { r l } & { \frac { \mathrm { d } ^ { 2 } } { \mathrm { d } t ^ { 2 } } \operatorname { t r } \left( \Phi ( \mathbf { A } _ { \theta } / n ) \right) } \\ & { \geq \frac { 1 } { n ^ { 2 } } \displaystyle \sum _ { k } \left( \psi ^ { \prime } ( a _ { k } ) + \frac { \psi ( a _ { k } ) } { a _ { k } } \right) \dot { \mathbf { A } } _ { \theta , k k } ^ { 2 } + \frac { 1 } { n ^ { 2 } } \displaystyle \sum _ { k < \ell } \left( 2 \frac { \psi ( a _ { k } ) - \psi ( a _ { \ell } ) } { a _ { k } - a _ { \ell } } + \frac { \psi ( a _ { k } ) } { a _ { \ell } } + \frac { \psi ( a _ { \ell } ) } { a _ { k } } \right) \dot { \mathbf { A } } _ { \theta , k \ell } ^ { 2 } } \\ & { = \frac { 1 } { n ^ { 2 } } \displaystyle \sum _ { k } \frac { g ^ { \prime } ( a _ { k } ) } { a _ { k } } \dot { \mathbf { A } } _ { \theta , k k } ^ { 2 } + \frac { 1 } { n ^ { 2 } } \displaystyle \sum _ { k < \ell } \frac { a _ { k } + a _ { \ell } } { a _ { k } a _ { \ell } } \frac { g ( a _ { k } ) - g ( a _ { \ell } ) } { a _ { k } - a _ { \ell } } \dot { \mathbf { A } } _ { \theta , k \ell } ^ { 2 } \geq 0 , } \end{array}
$$

due to monotonicity of $g .$ . Thus we have shown that $F$ is convex.

Moreover, $F$ is invariant under permutation since $\mathbf { z } _ { i }$ are i.i.d. Let $\theta ^ { \prime }$ be equal to $\theta$ with coordinates $i , j$ permuted. Since

$$
\left. \frac { \partial F } { \partial \theta _ { i } } \right| _ { \theta _ { i } = \log \lambda _ { i } } = \mathbb { E } \frac { \lambda _ { i } } { n } \mathbf { z } _ { i } ^ { \top } \boldsymbol { \psi } ( \mathbf { A } / n ) \mathbf { z } _ { i } = \mathbb { E } \mathbf { e } _ { i } ^ { \top } \mathbf { G } \mathbf { e } _ { i } = p _ { i } ,
$$

convexity of � implies

$$
\begin{array} { r l } { \quad } & { 0 \leq \langle \nabla F ( \theta ) - \nabla F ( \theta ^ { \prime } ) , \theta - \theta ^ { \prime } \rangle } \\ { \quad } & { = 2 ( \log \lambda _ { i } - \log \lambda _ { j } ) ( p _ { i } - p _ { j } ) . } \end{array}
$$

Hence $\lambda _ { i } > \lambda _ { j }$ implies $p _ { i } \geq p _ { j }$ , and $\lambda _ { i } = \lambda _ { j }$ implies $p _ { i } = p _ { j }$ by symmetry. Therefore $( p _ { i } ) _ { i \geq 1 }$ is nonincreasing. For the second claim, let $\mathbf { Q } _ { s }$ be the rotation of angle � on the span of $\mathbf { e } _ { i } , \mathbf { e } _ { j }$ and the identity on its orthogonal complement. Denote by $\mathbb { E } _ { s }$ the expectation under the transformation $\mathbf { \hat { X } } \mapsto \mathbf { X } \mathbf { Q } _ { s } ^ { \top }$ , equivalently $\pmb { \Sigma } \mapsto \pmb { \Sigma } _ { s } : = \mathbf { Q } _ { s } \pmb { \Sigma } \mathbf { Q } _ { s } ^ { \intercal }$ . By orthogonal equivariance,

$$
\begin{array} { r l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } s } \mathbf { e } _ { i } ^ { \top } \mathbf { Q } _ { s } ( \mathbb { B } \mathbf { G } ) \mathbf { Q } _ { s } ^ { \top } \mathbf { e } _ { j } \bigg | _ { s = 0 } = \left. \frac { \mathrm { d } } { \mathrm { d } s } \mathbb { E } _ { s } \mathbf { G } _ { i j } \right| _ { s = 0 } } & { } \\ { = \left. \frac { \mathrm { d } } { \mathrm { d } s } \left( \sin ( s ) \cos ( s ) ( \mathbb { B } \mathbf { G } _ { i i } - \mathbb { B } \mathbf { G } _ { j j } ) + ( \cos ^ { 2 } ( s ) - \sin ^ { 2 } ( s ) ) \mathbb { B } \mathbf { G } _ { i j } \right) \right| _ { s = 0 } } & { } \\ { = p _ { i } - p _ { j } . } \end{array}
$$

On the other hand, the score identity gives

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } s } \mathbb { E } _ { s } \mathbb { G } _ { i j } \Bigg \vert _ { s = 0 } = \mathbb { E } \left[ \mathbb { G } _ { i j } \left( - \frac { n } { 2 } \operatorname { t r } ( \Sigma ^ { - 1 } \dot { \Sigma } _ { 0 } ) + \frac { 1 } { 2 } \operatorname { t r } \left( \mathbf { X } \Sigma ^ { - 1 } \dot { \Sigma } _ { 0 } \Sigma ^ { - 1 } \mathbf { X } ^ { \top } \right) \right) \right] } \\ { \displaystyle = \frac { 1 } { 2 } \mathbb { E } \left[ \mathbf { G } _ { i j } \operatorname { t r } \left( \mathbf { X } \Sigma ^ { - 1 } ( \lambda _ { i } - \lambda _ { j } ) ( \mathbf { e } _ { i } \mathbf { e } _ { j } ^ { \top } + \mathbf { e } _ { j } \mathbf { e } _ { i } ^ { \top } ) \Sigma ^ { - 1 } \mathbf { X } ^ { \top } \right) \right] } \\ { \displaystyle = \frac { \lambda _ { i } - \lambda _ { j } } { \lambda _ { i } \lambda _ { j } } \mathbb { E } \left[ \mathbf { G } _ { i j } ( \mathbf { X } \mathbf { e } _ { i } ) ^ { \top } ( \mathbf { X } \mathbf { e } _ { j } ) \right] } \\ { \displaystyle = \frac { \lambda _ { i } - \lambda _ { j } } { \sqrt { \lambda _ { i } \lambda _ { j } } } \mathbb { E } \mathbf { G } _ { i j } \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { j } . } \end{array}
$$

By Cauchy–Schwarz, we thus have

$$
( p _ { i } - p _ { j } ) ^ { 2 } \leq \frac { ( \lambda _ { i } - \lambda _ { j } ) ^ { 2 } } { \lambda _ { i } \lambda _ { j } } \mathbb { E } \mathbf { G } _ { i j } ^ { 2 } \mathbb { E } ( \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { j } ) ^ { 2 } = \frac { n ( \lambda _ { i } - \lambda _ { j } ) ^ { 2 } } { \lambda _ { i } \lambda _ { j } } \mathbb { E } \mathbf { G } _ { i j } ^ { 2 } .
$$

We remark that this argument is essentially an application of the Cramér–Rao bound to the unbiased estimator $( p _ { i } - p _ { j } ) ^ { - 1 } \mathbf { G } _ { i j }$ for $( \lambda _ { i } - \lambda _ { j } ) ^ { - 1 } ( \pmb { \Sigma } _ { s } ) _ { i j }$ , when � is unknown. □

Lemma D.5. There exist constants $c _ { 0 } , c _ { 1 } , c _ { 3 } , c _ { 5 } > 1$ such that $i f n \geq c _ { 0 } , k \leq n / c _ { 3 }$ , and

$$
\frac { \sum _ { j > k } \lambda _ { j } ^ { 2 } } { n } \leq \frac { 1 } { c _ { 5 } } \left( \frac { \sum _ { j > k } \lambda _ { j } } { n } \right) ^ { 2 } ,\tag{16}
$$

then for all $i \leq k ,$

$$
c _ { 1 } \mathbb { E } { \bf B } _ { i i } \geq \lambda _ { i } \operatorname* { m i n } \left\{ 1 , \frac { \sum _ { j > k } \lambda _ { j } } { n \lambda _ { i } } \right\} ^ { 2 } .\tag{17}
$$

ProofofLemma D.5. If $\Sigma _ { j > k } \lambda _ { j } = 0$ , the statement is trivial. Otherwise, by the small-ball concentration inequality, with probability at least $1 - \exp ( - n / c _ { 0 } )$ ,

$$
\sum _ { j > k } \lambda _ { j } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \succeq \Bigg ( \sum _ { j > k } \lambda _ { j } - c _ { 1 } \sqrt { n \sum _ { j > k } \lambda _ { j } ^ { 2 } } \Bigg ) \mathbf { I } \succeq \frac { 1 } { 2 } \sum _ { j > k } \lambda _ { j } \mathbf { I } .
$$

Under this event and $\| \mathbf { z } _ { i } \| ^ { 2 } \leq 2 n$ , for all $i \leq k .$

$$
\mathbf { z } _ { i } ^ { \top } \mathbf { A } _ { - i } ^ { - 1 } \mathbf { z } _ { i } \leq \frac { 2 \| \mathbf { z } _ { i } \| ^ { 2 } } { \sum _ { j > k } \lambda _ { j } } \leq \frac { 4 n } { \sum _ { j > k } \lambda _ { j } } .
$$

Since $\mathbf { G } \preceq \mathbf { 1 } \{ \hat { \Sigma } > 0 \}$ and $\pmb { \Sigma } \succeq \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top }$

$$
\begin{array} { r l } & { \mathbf { B } _ { i i } \geq \lambda _ { i } ( \mathbf { e } _ { i } ^ { \top } ( \mathbf { I } - \mathbf { G } ) \mathbf { e } _ { i } ) ^ { 2 } } \\ & { \qquad \geq \lambda _ { i } \left( 1 - \mathbf { e } _ { i } ^ { \top } \mathbf { 1 } \{ \hat { \Sigma } > 0 \} \mathbf { e } _ { i } \right) ^ { 2 } } \\ & { \qquad = \lambda _ { i } \left( 1 + \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \mathbf { A } _ { - i } ^ { - 1 } \mathbf { z } _ { i } \right) ^ { - 2 } , } \end{array}
$$

where the last equality uses the Woodbury identity. Combining the two bounds gives, when $n \geq c _ { 0 }$

$$
c _ { 1 } \mathbb { E } \mathbf { B } _ { i i } \geq \lambda _ { i } \left( 1 + \frac { n \lambda _ { i } } { \sum _ { j > k } \lambda _ { j } } \right) ^ { - 2 } \geq \frac { \lambda _ { i } } { 4 } \operatorname* { m i n } \left\{ 1 , \frac { \sum _ { j > k } \lambda _ { j } } { n \lambda _ { i } } \right\} ^ { 2 } ,
$$

as was to be shown.

## D.2 Proof for ramp filters

We define a monotone shrinkage filter $g$ to be ‘ramp-like’ if there exist $0 < \rho _ { 0 } < \rho _ { 1 } < \infty$ such that $g ( \rho _ { 0 } ) = 0$ and $g ( \rho _ { 1 } ) = 1$ . In this subsection, we show a partial result for ramp-like filters.

Lemma D.6. There exist constants $c _ { 0 } , c _ { 1 } , c _ { 2 } , c _ { 3 }$ such that thefollowing holds. Let $g$ be a ramp-like filter with $g ( \rho _ { 0 } ) = 0 \ : a n d \ : g ( \rho _ { 1 } ) = 1 . \ : I f n \geq c _ { 0 }$ and � satisfies

$$
k \leq { \frac { n } { c _ { 3 } } } , \quad \lambda _ { k + 1 } + { \frac { 1 } { n } } \sum _ { j > k } \lambda _ { j } \leq { \frac { \rho _ { 0 } } { c _ { 2 } } } ,
$$

then there exists a threshold $\rho \in [ \rho _ { 0 } , \rho _ { 1 } ]$ such that

$$
w i t h p r o b a b i l i t y a t l e a s t 0 . 9 9 , \quad \mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p e r } } ) | \mathbf { X } ] \leq c _ { 1 } \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) .
$$

We first require the following concentration bounds.

Lemma D.7. In the setting of Lemma $D . 6 ,$ it holds with probability at least $1 - \exp ( - n / c _ { 0 } )$ that

$$
\frac { 1 } { 2 } \big ( \boldsymbol { \Sigma } + \rho _ { 0 } \mathbf { I } \big ) \preceq \hat { \boldsymbol { \Sigma } } + \rho _ { 0 } \mathbf { I } \preceq \frac { 3 } { 2 } \big ( \boldsymbol { \Sigma } + \rho _ { 0 } \mathbf { I } \big ) ,\tag{18}
$$

and simultaneously for every �,

$$
\mathbf { A } _ { - i } ^ { 2 } \preceq { c } ( n \mathbf { V } _ { - i } + n ^ { 2 } \alpha _ { k } \mathbf { I } ) .\tag{19}
$$

Proof of Lemma $D . 7 .$ Define the operators

$$
\begin{array} { l } { { \displaystyle { \bf T } : = ( { \bf \dot { \Sigma } } + \rho _ { 0 } { \bf I } ) ^ { - 1 / 2 } { \bf \dot { \Sigma } } ( { \bf \Sigma } { \bf \Sigma } + \rho _ { 0 } { \bf I } ) ^ { - 1 / 2 } } , } \\ { { { \hat { \bf T } } : = ( { \bf \dot { \Sigma } } + \rho _ { 0 } { \bf I } ) ^ { - 1 / 2 } { \hat { \bf \Sigma } } ( { \bf \Sigma } { \bf \Sigma } { \bf \Sigma } + \rho _ { 0 } { \bf I } ) ^ { - 1 / 2 } = \displaystyle \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { \bf y } _ { i } { \bf y } _ { i } ^ { \top } , \quad { \bf y } _ { i } : = ( { \bf \Sigma } { \bf \Sigma } + \rho _ { 0 } { \bf I } ) ^ { - 1 / 2 } { \bf x } _ { i } \sim N ( { \bf 0 } , { \bf T } ) . } } \end{array}
$$

It holds that $\| \mathbf { T } \| \leq 1$ and

$$
\operatorname { t r } ( \mathbf { T } ) = \sum _ { i } { \frac { \lambda _ { i } } { \lambda _ { i } + \rho _ { 0 } } } \leq k + \sum _ { j > k } { \frac { \lambda _ { j } } { \rho _ { 0 } } } \leq { \frac { n } { c _ { 3 } } } + { \frac { n } { c _ { 2 } } } .
$$

Then $\Vert \hat { \mathbf { T } } - \mathbf { T } \Vert \leq 1 / 2$ by standard concentration, and

$$
\bigl ( \Sigma + \rho _ { 0 } \mathbf { I } \bigr ) ^ { - 1 / 2 } \bigl ( \hat { \Sigma } + \rho _ { 0 } \mathbf { I } \bigr ) \bigl ( \Sigma + \rho _ { 0 } \mathbf { I } \bigr ) ^ { - 1 / 2 } = \hat { \mathbf { T } } + \rho _ { 0 } \bigl ( \Sigma + \rho _ { 0 } \mathbf { I } \bigr ) ^ { - 1 } = \mathbf { I } + \hat { \mathbf { T } } - \mathbf { T } .
$$

The first claim follows. For the second, define the sets of indices

$$
\begin{array} { l } { { H : = \{ j : j \leq k \} \cup \{ j : j > k , \lambda _ { j } > \sqrt { \alpha _ { k } } \} , } } \\ { { T : = \{ j : \lambda _ { j } > 0 , j \notin H \} . } } \end{array}
$$

We have $| H | \leq n / c _ { 3 } + n .$ , so $\| [ \mathbf { z } _ { h } ] _ { h \in H } \| ^ { 2 } \leq c n$ with probability at least $1 - \exp ( - n / c _ { 0 } )$ for constants $c , c _ { 0 }$ Then

$$
\left( \sum _ { j \in H , j \neq i } \lambda _ { j } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \right) ^ { 2 } \preceq \left\| \left[ \mathbf { z } _ { h } \right] _ { h \in H } \right\| ^ { 2 } \sum _ { j \in H , j \neq i } \lambda _ { j } ^ { 2 } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \preceq c n \sum _ { j \in H , j \neq i } \lambda _ { j } ^ { 2 } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } .
$$

For the remaining indices, the weighted covariance bound gives

$$
\left\| \sum _ { j \in T , j \neq i } \lambda _ { j } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \right\| \leq \left\| \sum _ { j \in T } \lambda _ { j } \mathbf { z } _ { j } \mathbf { z } _ { j } ^ { \top } \right\| \leq c \left( \sum _ { j > k } \lambda _ { j } + n \sqrt { \alpha _ { k } } \right) \leq 2 c n \sqrt { \alpha _ { k } } ,
$$

where the second inequality again holds with probability at least $1 - \exp ( - n / c _ { 0 } )$ . Combining the two bounds yields (19), as desired. □

ProofofLemma $D . 6 .$ . We first show that for all indices $i ,$

$$
\mathbb { E } { \bf B } _ { i i } \ge c \operatorname* { m i n } \left\{ \lambda _ { i } , \frac { \alpha _ { k } } { \lambda _ { i } } \right\} .\tag{20}
$$

Here, the right-hand side is understood as zero if $\lambda _ { i } = 0$ . Note the decomposition

$$
\mathbf { B } _ { i i } = \lambda _ { i } ( 1 - \mathbf { G } _ { i i } ) ^ { 2 } + \sum _ { j \neq i } \lambda _ { j } \mathbf { G } _ { i j } ^ { 2 } .
$$

Since � vanishes on $[ 0 , \rho _ { 0 } ]$ and $0 \leq g \leq 1$ , one has $g ( z ) \leq z / \rho _ { 0 }$ , thus

$$
p _ { i } \leq \frac { 1 } { \rho _ { 0 } } \mathbb { E } \mathbf { e } _ { i } ^ { \top } \hat { \boldsymbol { \Sigma } } \mathbf { e } _ { i } = \frac { \lambda _ { i } } { \rho _ { 0 } } .
$$

If $p _ { i } \leq 1 / 2$ (in particular if $i > k )$ , then $\mathbb { E } \mathbf { B } _ { i i } \geq \lambda _ { i } ( 1 - p _ { i } ) ^ { 2 } \geq \lambda _ { i } / 4$ is immediate. Otherwise $i \leq k$ and $\lambda _ { i } > \rho _ { 0 } / 2$ , while for $j > k$ one has $p _ { j } \leq 1 / c _ { 2 }$ and $\lambda _ { j } \le \lambda _ { k + 1 } \le \rho _ { 0 } / c _ { 2 }$ . Choosing $c _ { 2 }$ to be large, by Lemma $\mathrm { D } . 4$

$$
\mathbb { E } \mathbf { B } _ { i i } \geq \sum _ { j > k } \lambda _ { j } \mathbb { E } \mathbf { G } _ { i j } ^ { 2 } \geq \frac { c } { n \lambda _ { i } } \sum _ { j > k } \lambda _ { j } ^ { 2 } .
$$

If (16) holds, $\mathbb { E } { \mathbf { B } } _ { i i } \geq c \alpha _ { k } / \lambda _ { i }$ then follows from Lemma D.5; otherwise, this follows directly. Thus (20) is established.

Next, let E be the intersection of the event in Lemma D.7 and max $i \leq k \ \| z _ { i } \| ^ { 2 } \leq 2 n$ , so Pr $\mathcal { E } \geq 1 - \exp ( - n / c _ { 0 } )$ Under E, for $i \leq k$

$$
\begin{array} { r l } & { \displaystyle \| ( \mathbf { I } - \mathbf { G } ) \mathbf { e } _ { i } \| _ { \hat { \boldsymbol { \Sigma } } } ^ { 2 } = \frac { \lambda _ { i } } { n } \| ( \mathbf { I } - g ( \mathbf { A } / n ) ) \mathbf { z } _ { i } \| ^ { 2 } } \\ & { \displaystyle \quad = \frac { \lambda _ { i } } { n } \| ( 1 - \mathbf { G } _ { i i } ) \mathbf { z } _ { i } - \mathbf { A } _ { - i } \boldsymbol { \Psi } \mathbf { z } _ { i } \| ^ { 2 } } \\ & { \displaystyle \quad \leq 2 \lambda _ { i } ( 1 - \mathbf { G } _ { i i } ) ^ { 2 } \frac { \| \mathbf { z } _ { i } \| ^ { 2 } } { n } + \frac { 2 \lambda _ { i } } { n } \mathbf { z } _ { i } ^ { \top } \boldsymbol { \Psi } \mathbf { A } _ { - i } ^ { 2 } \boldsymbol { \Psi } \mathbf { z } _ { i } } \\ & { \displaystyle \quad \leq 2 \lambda _ { i } ( 1 - \mathbf { G } _ { i i } ) ^ { 2 } \frac { \| \mathbf { z } _ { i } \| ^ { 2 } } { n } + 2 c \left( \sum _ { j \neq i } \lambda _ { j } \mathbf { G } _ { j i } ^ { 2 } + n \alpha _ { k } \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \boldsymbol { \Psi } ^ { 2 } \mathbf { z } _ { i } \right) . } \end{array}\tag{by (19}
$$

Here, we have used

$$
\lambda _ { i } \mathbf { z } _ { i } ^ { \mathsf { T } } \Psi \mathbf { V } _ { - i } \Psi \mathbf { z } _ { i } = \sum _ { j \neq i } \lambda _ { i } \lambda _ { j } ^ { 2 } ( \mathbf { z } _ { j } ^ { \mathsf { T } } \Psi \mathbf { z } _ { i } ) ^ { 2 } = \sum _ { j \neq i } \lambda _ { j } \mathbf { G } _ { j i } ^ { 2 } .
$$

Because � vanishes on $[ 0 , \rho _ { 0 } ]$

$$
\begin{array} { r l } { \displaystyle \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \boldsymbol { \Psi } ^ { 2 } \mathbf { z } _ { i } = \frac { 1 } { n } \mathbf { e } _ { i } ^ { \top } \hat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { G } ^ { 2 } \mathbf { e } _ { i } \leq \frac { 1 } { n } \mathbf { e } _ { i } ^ { \top } \hat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { 1 } \{ \hat { \boldsymbol { \Sigma } } > \rho _ { 0 } \mathbf { I } \} \mathbf { e } _ { i } } & { } \\ { \leq \frac { 2 } { n } \mathbf { e } _ { i } ^ { \top } ( \hat { \boldsymbol { \Sigma } } + \rho _ { 0 } \mathbf { I } ) ^ { - 1 } \mathbf { e } _ { i } } & { } \\ { \leq \frac { 4 } { n } \mathbf { e } _ { i } ^ { \top } ( \boldsymbol { \Sigma } + \rho _ { 0 } \mathbf { I } ) ^ { - 1 } \mathbf { e } _ { i } \leq \frac { 4 } { n \lambda _ { i } } . } \end{array}\tag{by (18}
$$

Also, $\alpha _ { k } \leq 2 \rho _ { 0 } ^ { 2 } / c _ { 2 } ^ { 2 }$ by definition of � and $\| \Psi \| \leq 1 / ( n \rho _ { 0 } )$ , so conditional on max $i \leq k \ \| z _ { i } \| ^ { 2 } \leq 2 n$

$$
\lambda _ { i } \mathbf { z } _ { i } ^ { \top } \Psi ^ { 2 } \mathbf { z } _ { i } \leq \lambda _ { i } \| \Psi \| ^ { 2 } \| \mathbf { z } _ { i } \| ^ { 2 } \leq \frac { 4 \lambda _ { i } } { c _ { 2 } ^ { 2 } n \alpha _ { k } } .
$$

Hence we have shown that with probability at least $1 - \exp ( - n / c _ { 0 } )$ , for some constant $c ^ { \prime }$

$$
n \alpha _ { k } \lambda _ { i } \mathbf { z } _ { i } ^ { \top } \Psi ^ { 2 } \mathbf { z } _ { i } \leq c \operatorname* { m i n } \left\{ \lambda _ { i } , \frac { \alpha _ { k } } { \lambda _ { i } } \right\} \leq c ^ { \prime } \mathbb { E } \mathbf { B } _ { i i } .
$$

Moreover on the same event,

$$
2 \lambda _ { i } ( 1 - \mathbf { G } _ { i i } ) ^ { 2 } \frac { \| \mathbf { z } _ { i } \| ^ { 2 } } { n } + 2 c \sum _ { j \neq i } \lambda _ { j } \mathbf { G } _ { j i } ^ { 2 } \leq c \left( \lambda _ { i } ( 1 - \mathbf { G } _ { i i } ) ^ { 2 } + \sum _ { j \neq i } \lambda _ { j } \mathbf { G } _ { j i } ^ { 2 } \right) = c \mathbf { B } _ { i i } .
$$

Finally for $i > k$ , the assumptions give $p _ { i } \le \lambda _ { i } / \rho _ { 0 } \le 1 / c _ { 2 }$ , and we can directly bound

$$
\begin{array} { r } { \mathbb { E } \| ( { \bf I } - { \bf G } ) { \bf e } _ { i } \| _ { \hat { \Sigma } } ^ { 2 } \leq \mathbb { E } { \bf e } _ { i } ^ { \top } \hat { \Sigma } { \bf e } _ { i } = \lambda _ { i } \leq c \lambda _ { i } ( 1 - p _ { i } ) ^ { 2 } \leq c ^ { \prime } \mathbb { E } { \bf B } _ { i i } . } \end{array}
$$

Therefore, we have

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } } \sum _ { i } \mathbf { w } _ { i } ^ { * 2 } \lVert ( \mathbf { I } - \mathbf { G } ) \mathbf { e } _ { i } \rVert _ { \hat { \Sigma } } ^ { 2 } \right] \leq c \sum _ { i } \mathbf { w } _ { i } ^ { * 2 } \mathbb { E } \mathbf { B } _ { i i } = c \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) .\tag{21}
$$

Now define

$$
\rho : = \operatorname* { i n f } \{ z \geq 0 : g ( z ) > 1 / 2 \} , \quad \mathbf { P } : = \mathbf { 1 } \{ \mathbf { G } > \mathbf { I } / 2 \} .
$$

By monotonicity, $\rho \in [ \rho _ { 0 } , \rho _ { 1 } ]$ and ${ \bf 1 } \{ g ( z ) > 1 / 2 \} = { \bf 1 } \{ z > \rho \}$ for every $z \neq \rho .$ . Since $\rho > 0$ is deterministic, under Gaussian design, it is almost surely not an empirical eigenvalue. Thus P is equal to the PCR projection at threshold $\rho$ almost surely.

For every $z \in [ 0 , 1 ]$ , we have

$$
\begin{array} { r l } { ( \mathbf { 1 } \{ z > 1 / 2 \} - z ) ^ { 2 } \leq ( 1 - z ) ^ { 2 } } & { { } \implies \quad ( \mathbf { P } - \mathbf { G } ) ^ { 2 } \preceq ( \mathbf { I } - \mathbf { G } ) ^ { 2 } , } \\ { \mathbf { 1 } \{ z > 1 / 2 \} \leq 4 z ^ { 2 } } & { { } \implies \quad \mathbf { P } \preceq 4 \mathbf { G } ^ { 2 } . } \end{array}
$$

Since $\mathbf { P } , \mathbf { G } , \hat { \Sigma }$ commute and ran $( { \bf P } - { \bf G } ) \subseteq \mathrm { r a n } { \bf 1 } \{ \hat { \Sigma } > \rho _ { 0 } { \bf I } \}$ , it follows that on $\varepsilon ,$

$$
\begin{array} { r l } & { \| ( \mathbf { P } - \mathbf { G } ) \mathbf { e } _ { i } \| _ { \Sigma } ^ { 2 } \leq 2 \| ( \mathbf { P } - \mathbf { G } ) \mathbf { e } _ { i } \| _ { \hat { \Sigma } } ^ { 2 } + \rho _ { 0 } \| ( \mathbf { P } - \mathbf { G } ) \mathbf { e } _ { i } \| ^ { 2 } } \\ & { \qquad \leq 3 \| ( \mathbf { P } - \mathbf { G } ) \mathbf { e } _ { i } \| _ { \hat { \Sigma } } ^ { 2 } } \\ & { \qquad \leq 3 \| ( \mathbf { I } - \mathbf { G } ) \mathbf { e } _ { i } \| _ { \hat { \Sigma } } ^ { 2 } . } \end{array}\tag{by (18}
$$

The event $\varepsilon$ is invariant under coordinate sign changes, so by symmetry and (21),

$$
\begin{array} { r l } { \displaystyle \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } } \big \lVert ( \mathbf { P } - \mathbf { G } ) \mathbf { w } ^ { * } \big \rVert _ { \Sigma } ^ { 2 } \right] = \sum _ { i } \mathbf { w } _ { i } ^ { * 2 } \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } } \big \lVert ( \mathbf { P } - \mathbf { G } ) \mathbf { e } _ { i } \big \rVert _ { \Sigma } ^ { 2 } \right] } & { } \\ { \leq 3 \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } } \sum _ { i } \mathbf { w } _ { i } ^ { * 2 } \big \lVert ( \mathbf { I } - \mathbf { G } ) \mathbf { e } _ { i } \big \rVert _ { \hat { \Sigma } } ^ { 2 } \right] } & { } \\ { \leq c \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) . } \end{array}
$$

Therefore,

$$
\begin{array} { r } { \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } } \big \| \big ( \mathbf { I } - \mathbf { P } \big ) \mathbf { w } ^ { * } \big \| _ { \Sigma } ^ { 2 } \right] \leq 2 \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } } \big \| \big ( \mathbf { I } - \mathbf { G } \big ) \mathbf { w } ^ { * } \big \| _ { \Sigma } ^ { 2 } \right] + 2 \mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } } \big \| \big ( \mathbf { P } - \mathbf { G } \big ) \mathbf { w } ^ { * } \big \| _ { \Sigma } ^ { 2 } \right] \leq c \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) . } \end{array}
$$

Moreover, $\mathbf { P } \preceq 4 \mathbf { G } ^ { 2 }$ gives, conditional on $\mathbf { X } ,$

$$
\begin{array} { r l r } {  { \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) = \frac { \sigma ^ { 2 } } { n } \mathrm { t r } ( { \boldsymbol \Sigma } \hat { \boldsymbol { \Sigma } } ^ { - 1 } { \mathbf { P } } ) } } \\ & { } & { \leq 4 \frac { \sigma ^ { 2 } } { n } \mathrm { t r } ( { \boldsymbol \Sigma } \hat { \boldsymbol { \Sigma } } ^ { - 1 } { \mathbf { G } } ^ { 2 } ) = 4 \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g } ) . } \end{array}
$$

Combining the bias and variance bounds, we have shown that

$$
\mathbb { E } \left[ \mathbf { 1 } _ { \mathcal { E } } \mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) \mid \mathbf { X } ] \right] \leq c \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) .
$$

We may assume $\operatorname* { P r } ( \mathcal { E } ^ { c } ) \leq 0 . 0 0 5$ . If $\mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) = 0$ , the preceding inequality implies that PCR risk vanishes almost surely on $\varepsilon ,$ , proving the claim. Otherwise, taking $c _ { 1 } \geq 2 0 0 c$ , Markov’s inequality gives

$$
\operatorname* { P r } \left( \mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) \mid \mathbf { X } ] > c _ { 1 } \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) \right) \leq \operatorname* { P r } ( \mathcal { E } ^ { c } ) + \frac { c } { c _ { 1 } } \leq 0 . 0 1 .
$$

This concludes the proof.

## D.3 Proof of Theorem 2.1

Proof of Theorem 2.1. Define the �-quantile of $g$ for $\nu \in [ 0 , 1 ]$ as

$$
q _ { \nu } : = \operatorname* { i n f } \{ z \geq 0 : g ( z ) \geq \nu \} \in [ 0 , \infty ] .
$$

Suppose $q _ { 1 / 3 } = 0$ . Since $g ( z ) \geq 1 / 3$ for all $z > 0 ,$ , we may decompose $g = ( 1 / 3 ) g _ { 1 } + ( 2 / 3 ) g _ { 2 }$ where

$$
g _ { 1 } ( z ) = 1 \{ z > 0 \} , \quad g _ { 2 } ( z ) = \frac { 3 g ( z ) - 1 \{ z > 0 \} } { 2 } .
$$

Then $g _ { 1 } , g _ { 2 }$ are monotone shrinkage and in particular $g _ { 1 }$ is OLS, thus equivalent to PCR with threshold 0. Thus by Markov’s inequality and Lemma D.3, with probability at least 0.99,

$$
\mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) \geq \frac { 1 } { 9 } \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g _ { 1 } } ) \quad \mathrm { a n d } \quad \operatorname* { i n f } _ { \rho \geq 0 } \mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) | \mathbf { X } ] \leq \mathbb { E } [ \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g _ { 1 } } ) | \mathbf { X } ] \leq 1 0 0 \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g _ { 1 } } ) ,
$$

so the theorem is proved. Next, if $q _ { 2 / 3 } = \infty ,$ , it holds that $\mathbf { G } \preceq \frac { 2 } { 3 } \mathbf { I }$ so that

$$
\begin{array} { r l } & { \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) = \displaystyle \sum _ { i } \mathbf { w } _ { i } ^ { * 2 } \mathbb { E } \mathbf { e } _ { i } ^ { \top } ( \mathbf { I } - { \mathbf { G } } ) \Sigma ( \mathbf { I } - { \mathbf { G } } ) \mathbf { e } _ { i } } \\ & { \qquad \quad \geq \displaystyle \sum _ { i } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } \mathbb { E } ( \mathbf { e } _ { i } ^ { \top } ( \mathbf { I } - { \mathbf { G } } ) \mathbf { e } _ { i } ) ^ { 2 } \geq \frac { 1 } { 9 } \| \mathbf { w } ^ { * } \| _ { \Sigma } ^ { 2 } . } \end{array}
$$

Since the zero estimator (obtained by taking $\rho  \infty )$ always achieves constant risk $\| \mathbf { w } ^ { * } \| _ { \Sigma } ^ { 2 }$ , the theorem is proved.

We may thus suppose $0 < q _ { 1 / 3 } \le q _ { 2 / 3 } < \infty$ . Assume $n \geq c _ { 0 }$ and set

$$
k : = \frac { n } { c _ { 3 } } , \quad \rho : = c _ { 3 } \left( \lambda _ { k + 1 } + \frac { 1 } { n } \sum _ { i > k } \lambda _ { i } \right) .
$$

If $\rho \leq q _ { 1 / 3 }$ , decompose $g = ( g _ { 1 } + g _ { 2 } + g _ { 3 } ) / 3$ where

$$
g _ { 1 } ( z ) = \operatorname* { m i n } \{ 3 g ( z ) , 1 \} , \quad g _ { 2 } ( z ) = \operatorname* { m i n } \{ \operatorname* { m a x } \{ 3 g ( z ) - 1 , 0 \} , 1 \} , \quad g _ { 3 } ( z ) = \operatorname* { m a x } \{ 3 g ( z ) - 2 , 0 \} .
$$

It must hold that $g _ { 2 } ( 0 . 5 q _ { 1 / 3 } ) = 0$ and $g _ { 2 } ( 2 q _ { 2 / 3 } ) = 1$ , so $g _ { 2 }$ is a ramp filter and

$$
\mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) \geq \frac { 1 } { 9 } \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g _ { 2 } } )
$$

by Lemma D.3. The theorem then follows from Lemma D.6.

It remains to consider the case $\rho > q _ { 1 / 3 }$ . We directly compare with the PCR upper bound in Theorem 4.1 with threshold $\rho .$ For the variance, since $g ( z ) \geq 1 / 3$ for every $z > \rho ,$ we have conditional on $\mathbf { X } ,$

$$
\mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) = \frac { \sigma ^ { 2 } } { n } \mathrm { t r } ( \Sigma \hat { \Sigma } _ { > \rho } ^ { - 1 } ) \leq 9 \frac { \sigma ^ { 2 } } { n } \mathrm { t r } ( \Sigma \hat { \Sigma } ^ { - 1 } \mathbf { G } ^ { 2 } ) .
$$

Hence by Markov’s inequality,

$$
\mathrm { w i t h ~ p r o b a b i l i t y ~ a t ~ l e a s t ~ 0 . 9 9 5 , ~ } \quad \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) \leq c \mathbb { E } \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g } ) .
$$

To compare the bias, let $k ^ { * }$ be the critical index in Theorem $4 . 1$ , so that $k ^ { * } \le k , \lambda _ { k ^ { * } } > 1 . 1 \rho$ and

$$
\frac { 1 } { n } \sum _ { i > k ^ { * } } \lambda _ { i } \leq \frac { k - k ^ { * } } { n } \lambda _ { k ^ { * } + 1 } + \frac { 1 } { n } \sum _ { i > k } \lambda _ { i } \leq \frac { 1 } { c _ { 3 } } \left( 1 . 1 \rho + \frac { 1 } { c _ { 2 } n } \sum _ { i > k ^ { * } } \lambda _ { i } \right) + \frac { \rho } { c _ { 3 } } .
$$

Choosing $c _ { 2 } , c _ { 3 }$ suficiently large, we ensure that

$$
\frac { 1 } { n } \sum _ { i > k ^ { * } } \lambda _ { i } \leq \frac { 3 \rho } { c _ { 3 } } , \quad \lambda _ { k ^ { * } + 1 } \leq 2 \rho , \quad \frac { 1 } { n } \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } \leq \frac { 6 \rho ^ { 2 } } { c _ { 3 } } .
$$

Then with probability at least 0.995, the bias bound in Theorem 4.1 gives

$$
\begin{array} { r l } & { \displaystyle \frac { 1 } { c _ { 1 } } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { \rho } ^ { \mathrm { p c r } } ) \leq \bigg ( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } + \sqrt { \frac { \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } } \bigg ) ^ { 2 } \| { \mathbf { w } } ^ { * } \| _ { { \Sigma _ { 0 ; k ^ { * } } ^ { - 1 } } } ^ { 2 } + \| { \mathbf { w } } ^ { * } \| _ { { \Sigma _ { k ^ { * } ; \infty } } } ^ { 2 } } \\ & { \quad \quad \quad \leq c \rho ^ { 2 } \| { \mathbf { w } } ^ { * } \| _ { { \Sigma _ { 0 ; k ^ { * } } ^ { - 1 } } } ^ { 2 } + \| { \mathbf { w } } ^ { * } \| _ { { \Sigma _ { k ^ { * } ; \infty } } } ^ { 2 } . } \end{array}
$$

We proceed to show a matching bias lower bound for $\hat { \mathbf { w } } _ { g }$ in this regime. Define the set of ‘intermediate’ indices

$$
J : = \{ j : k ^ { * } < j \leq k , p _ { j } \leq 1 / 4 \} .
$$

By a similar concentration argument as in Lemma D.7, with probability at least $1 - \exp ( - n / c _ { 0 } )$

$$
\hat { \Sigma } + \rho \mathbf { I } \preceq \frac { 3 } { 2 } ( \Sigma + \rho \mathbf { I } ) , \quad \hat { \Sigma } _ { \leq k ^ { * } } \succeq 0 . 9 9 \Sigma _ { \leq k ^ { * } } \succeq \rho \mathbf { I } .
$$

It follows that $\Sigma \succeq \frac { 1 } { 3 } \hat { \Sigma }$ on the range of $\mathbf { 1 } \{ \hat { \mathbf { \Sigma } } > \rho \mathbf { I } \}$ . Since $\rho > q _ { 1 / 3 }$ , we also have $\mathbf G ^ { 2 } \succeq \frac { 1 } { 9 } \mathbf 1 \{ \hat { \boldsymbol { \Sigma } } > \rho \mathbf I \}$ , so the variance of $\hat { \mathbf { w } } _ { g }$ is lower bounded as

$$
\begin{array} { l } { \displaystyle \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g } ) \geq \frac { \sigma ^ { 2 } } { 9 n } \mathrm { t r } \left( { \boldsymbol { \Sigma } } \hat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { 1 } \{ \hat { \boldsymbol { \Sigma } } > \rho \mathbf { I } \} \right) } \\ { \displaystyle \geq \frac { \sigma ^ { 2 } } { 2 7 n } \mathrm { t r } \left( \mathbf { 1 } \{ \hat { \boldsymbol { \Sigma } } > \rho \mathbf { I } \} \right) } \\ { \displaystyle \geq c k ^ { * } \frac { \sigma ^ { 2 } } { n } . } \end{array}
$$

Moreover by Cauchy–Schwarz,

$$
p _ { i } ^ { 2 } = \left( \mathbb { E } ( \hat { \boldsymbol { \Sigma } } ^ { 1 / 2 } \mathbf { e } _ { i } , \hat { \boldsymbol { \Sigma } } ^ { - 1 / 2 } \mathbf { G } \mathbf { e } _ { i } ) \right) ^ { 2 } \leq \mathbb { E } \mathbf { e } _ { i } ^ { \top } \hat { \boldsymbol { \Sigma } } \mathbf { e } _ { i } \mathbb { E } \mathbf { e } _ { i } ^ { \top } \hat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { G } ^ { 2 } \mathbf { e } _ { i } = \lambda _ { i } \mathbb { E } \mathbf { e } _ { i } ^ { \top } \hat { \boldsymbol { \Sigma } } ^ { - 1 } \mathbf { G } ^ { 2 } \mathbf { e } _ { i } .
$$

Summing over � gives

$$
\mathbb { E } { \mathrm { V a r i a n c e } } ( \hat { \mathbf { w } } _ { g } ) \geq \frac { \sigma ^ { 2 } } { n } \sum _ { i } p _ { i } ^ { 2 } .
$$

Therefore from the two bounds,

$$
\mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) \geq c k ^ { * } \frac { \sigma ^ { 2 } } { n } \quad \mathrm { a n d } \quad \mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) \geq \frac { \sigma ^ { 2 } } { 1 6 n } \# \{ i : p _ { i } > 1 / 4 \} .
$$

On the other hand, we may assume $\mathbb { E } \mathcal { E } _ { \mu } ( \hat { \mathbf { w } } _ { g } ) \leq \sigma ^ { 2 } / c$ for any constant �, as otherwise the zero estimator achieves risk $\| \mathbf { w } ^ { * } \| _ { \Sigma } ^ { 2 } \leq b \sigma ^ { 2 }$ and the theorem is proved. Choosing � suficiently large, this and $k = n / c _ { 3 }$ imply

$$
k ^ { * } \leq k / 8 , \quad \# \{ i : p _ { i } > 1 / 4 \} \leq k / 8 \quad \implies \quad | J | \geq k / 2 .
$$

In particular, since $p _ { i }$ is nonincreasing by Lemma $\mathrm { D } . 4 .$ , it follows that $p _ { j } \leq 1 / 4$ for every $j > k$

Next, arguing as in (20), we show that for all �,

$$
\mathbb { E } \mathbf { B } _ { i i } \geq c \operatorname* { m i n } \left\{ \lambda _ { i } , \frac { \rho ^ { 2 } } { \lambda _ { i } } \right\} .\tag{22}
$$

Indeed, if $p _ { i } \leq 1 / 2$ (in particular if $i > k )$ , then $\mathbb { E } { \mathbf { B } } _ { i i } \geq c \lambda _ { i }$ . Suppose $p _ { i } > 1 / 2$ . Then for every $j \in J \ { \mathrm { o r } } \ j > k$ it holds that $i < j$ and $p _ { j } \leq 1 / 4$ , so Lemma D.4 gives

$$
\lambda _ { j } \mathbb { E } \mathbf { G } _ { j i } ^ { 2 } \geq \frac { \lambda _ { j } ^ { 2 } } { 1 6 n \lambda _ { i } } .\tag{23}
$$

If $\lambda _ { k + 1 } \geq \rho / ( 2 c _ { 3 } )$ , then summing (23) over $j \in J$ gives

$$
\mathbb { E } \mathbf { B } _ { i i } \geq \frac { | J | } { 1 6 n } \frac { \lambda _ { k + 1 } ^ { 2 } } { \lambda _ { i } } \geq c \frac { \rho ^ { 2 } } { \lambda _ { i } } .
$$

Otherwise $\begin{array} { r } { n ^ { - 1 } \sum _ { j > k } \lambda _ { j } \ge \rho / ( 2 c _ { 3 } ) } \end{array}$ , and E $\mathbf { B } _ { i i } \geq c \rho ^ { 2 } / \lambda _ { i }$ follows as before from Lemma D.5 if (16) holds, and summing (23) over $j > k$ otherwise.

Now from (22), for indices $i \le k ^ { * } , \lambda _ { i } > \rho$ implies $\mathbb { E } { \mathbf { B } } _ { i i } \geq c \rho ^ { 2 } / \lambda _ { i } ;$ for indices $i > k ^ { * } , \lambda _ { i } \le \lambda _ { k ^ { * } + 1 } \le c \rho$ implies $\mathbb { E } { \mathbf { B } } _ { i i } \geq c \lambda _ { i }$ . It follows that

$$
\begin{array} { r } { \rho ^ { 2 } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } + \| \mathbf { w } ^ { * } \| _ { \Sigma _ { k ^ { * } ; \infty } } ^ { 2 } = \displaystyle \sum _ { i \leq k ^ { * } } \frac { \rho ^ { 2 } } { \lambda _ { i } } \mathbf { w } _ { i } ^ { * 2 } + \displaystyle \sum _ { i > k ^ { * } } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } } \\ { \leq c \displaystyle \sum _ { i } \mathbf { w } _ { i } ^ { * 2 } \mathbb { E } \mathbf { B } _ { i i } = c \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) , } \end{array}
$$

Hence with probability at least 0.99, the PCR risk is at most a constant multiple of the expected risk of $g ,$ as was to be shown. □

## D.4 Proof of Theorem 2.2

ProofofTheorem 2.2. We construct a series of instances $( \boldsymbol { \Sigma } _ { n } , \mathbf w _ { n } ^ { * } )$ , dependent only on $\epsilon ,$ such that for each instance, there exists a PCR filter that is polynomially better (w.r.t. �) than any $g \in { \mathcal { G } } _ { \epsilon , \delta }$ . We may suppose $\epsilon \leq 1$ , as otherwise $G _ { \epsilon , \delta } \subset G _ { 1 , \delta }$ . Choose $\alpha > 0$ such that $( 1 + \alpha ) ^ { 4 } = 1 + \epsilon$ . For every sample size $n ,$ let

$$
\Sigma _ { n } : = \mathrm { d i a g } \left( \frac { 1 } { 2 } , ( 1 + \alpha ) ^ { 2 } s _ { n } , s _ { n } \mathbf { I } _ { m _ { n } } \right) , \quad \mathbf { w } _ { n } ^ { * } : = \sqrt { 2 \left( 1 - \frac { 1 } { m _ { n } } \right) } \mathbf { e } _ { 1 } + \frac { 1 } { 1 + \alpha } \sqrt { \frac { 1 } { s _ { n } m _ { n } } } \mathbf { e } _ { 2 } , \quad \sigma _ { n } : = 1 ,
$$

where

$$
m _ { n } : = \lfloor n ^ { 1 / 2 } \rfloor , \quad s _ { n } : = \frac { 1 } { 2 ( ( 1 + \alpha ) ^ { 2 } + m _ { n } ) } .
$$

It holds that $\operatorname { t r } ( \Sigma _ { n } ) = 1$ and $\| \mathbf { w } _ { n } ^ { * } \| _ { \Sigma _ { n } } ^ { 2 } = 1 ( \mathrm { i f } \ : b < 1 , \sigma _ { n }$ can be scaled accordingly). We claim that

• with $\rho _ { n } : = ( 1 + \alpha ) s _ { n }$ , with probability at least 0.99, it holds that $\mathbb { E } [ \mathcal { E } ( \hat { \mathbf { w } } _ { \rho _ { n } } ^ { \mathrm { p c r } } ) | \mathbf { X } ] = O ( n ^ { - 1 } )$

• for any $g \in { \mathcal { G } } _ { \epsilon , \delta }$ , it holds uniformly in $g$ that $\mathbb { E } \pmb { \mathcal { E } } ( \hat { \mathbf { w } } _ { g } ) = \Omega ( n ^ { - 1 / 2 } )$

First, we upper bound the risk of PCR at threshold $\rho _ { n }$ . We apply the strengthened version of Theorem 4.1 given in Appendix C with slack $\alpha / 2$ . Denote $\begin{array} { r } { S _ { 1 } ( k ) : = \sum _ { i > k } \lambda _ { i } } \end{array}$ . The selected index is

$$
k ^ { * } : = \operatorname* { m i n } \left\{ k : ( 1 + \alpha / 2 ) \rho _ { n } + { \frac { S _ { 1 } ( k ) } { c _ { 2 } n } } \geq \lambda _ { k + 1 } \right\} .
$$

We claim that $k ^ { * } = 2$ . Indeed, for suficiently large �,

$$
( 1 + \alpha / 2 ) \rho _ { n } + \frac { S _ { 1 } ( 0 ) } { c _ { 2 } n } < \frac { 1 } { 2 } = \lambda _ { 1 } ,
$$

$$
( 1 + \alpha / 2 ) \rho _ { n } + \frac { S _ { 1 } ( 1 ) } { c _ { 2 } n } < ( 1 + \alpha ) ^ { 2 } s _ { n } = \lambda _ { 2 } ,
$$

whereas

$$
( 1 + \alpha / 2 ) \rho _ { n } + \frac { S _ { 1 } ( 2 ) } { c _ { 2 } n } > \rho _ { n } > s _ { n } = \lambda _ { 3 } .
$$

Moreover since $m _ { n } = \lfloor n ^ { 1 / 2 } \rfloor$ , by standard matrix concentration, with probability at least $1 - \exp ( - \alpha ^ { 2 } n / c _ { 0 } )$

$$
\frac { 1 } { 1 + \alpha / 2 } \pmb { \Sigma } _ { n } \preceq \hat { \pmb { \Sigma } } _ { n } \preceq \left( 1 + \frac { \alpha } { 2 } \right) \pmb { \Sigma } _ { n } .\tag{24}
$$

In particular, for every $j \geq 3$

$$
\frac { s _ { n } } { 1 + \alpha / 2 } \leq \hat { \lambda } _ { j } \leq \left( 1 + \frac { \alpha } { 2 } \right) s _ { n } < \rho _ { n } ,
$$

while for $j \leq 2 ,$

$$
\hat { \lambda } _ { j } \ge \frac { \lambda _ { 2 } } { 1 + \alpha / 2 } = \frac { ( 1 + \alpha ) ^ { 2 } s _ { n } } { 1 + \alpha / 2 } > \rho _ { n } ,
$$

so exactly two empirical eigenvalues exceed $\rho _ { n } , \mathrm { i . e . , } \hat { k } = 2 .$ By Theorem 4.1, with probability at least 0.99,

$$
\mathbb { E } \big [ \mathcal { E } _ { \mu _ { n } } ( \hat { \mathbf { w } } _ { \rho _ { n } } ^ { \mathrm { p c r } } ) | \mathbf { X } \big ] \leq c _ { 1 } \frac { m _ { n } s _ { n } ^ { 2 } } { \alpha ^ { 2 } n } \left( 4 \left( 1 - \frac { 1 } { m _ { n } } \right) + \frac { 1 } { ( 1 + \alpha ) ^ { 4 } s _ { n } ^ { 2 } m _ { n } } \right) + \frac { 2 } { n } \leq \frac { c } { \alpha ^ { 2 } n } .
$$

Hence the PCR risk is $O ( 1 / n )$

Now fix any $g \in { \mathcal { G } } _ { \epsilon , \delta }$ and define

$$
a _ { g } : = \operatorname* { s u p } \{ s : g ( s ) \leq \delta \} , \qquad b _ { g } : = \operatorname* { i n f } \{ s : g ( s ) \geq 1 - \delta \} .
$$

By definition, $b _ { g } / a _ { g } \geq 1 + \epsilon$ . We show that the risk of $g$ is uniformly lower bounded by $\Omega ( n ^ { - 1 / 2 } )$ , where constants with polynomial dependence on �, � are hidden. On the event in (24), we have

$$
\mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g } ) = \frac { 1 } { n } \sum _ { j = 1 } ^ { m _ { n } + 2 } \frac { \hat { u } _ { j } ^ { \top } \Sigma _ { n } \hat { u } _ { j } } { \hat { \lambda } _ { j } } g ( \hat { \lambda } _ { j } ) ^ { 2 } \geq \frac { m _ { n } } { ( 1 + \alpha / 2 ) n } g \left( \frac { s _ { n } } { 1 + \alpha / 2 } \right) ^ { 2 } .
$$

For the bias, Lemma F.4 with $\tau = 1 + \alpha / 2$ yields

$$
\begin{array} { r } { \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) \geq c \alpha ^ { 2 } \displaystyle \sum _ { i } \lambda _ { i } ( 1 - g ( ( 1 + \alpha / 2 ) \lambda _ { i } ) ) ^ { 2 } \mathbf { w } _ { i } ^ { * 2 } } \\ { \geq \displaystyle \frac { c \alpha ^ { 2 } } { m _ { n } } \left( 1 - g ( ( 1 + \alpha / 2 ) ( 1 + \alpha ) ^ { 2 } s _ { n } ) \right) ^ { 2 } . } \end{array}
$$

If $a _ { g } = 0$ or $s _ { n } > ( 1 + \alpha / 2 ) a _ { g }$ , then

$$
g \left( \frac { s _ { n } } { 1 + \alpha / 2 } \right) \geq \delta ,
$$

and Variance $\big ( \hat { \mathbf { w } } _ { g } \big ) = \Omega ( n ^ { - 1 / 2 } )$ . Otherwise,

$$
( 1 + \alpha / 2 ) ( 1 + \alpha ) ^ { 2 } s _ { n } \le ( 1 + \alpha / 2 ) ^ { 2 } ( 1 + \alpha ) ^ { 2 } a _ { g } < ( 1 + \epsilon ) a _ { g } \le b _ { g } ,
$$

and

$$
\mathbb { E B i a s } ( \hat { \mathbf { w } } _ { g } ) > \frac { c \alpha ^ { 2 } \delta ^ { 2 } } { m _ { n } } = \Omega ( n ^ { - 1 / 2 } ) .
$$

The proof is complete.

ProofofExample 1. Since all considered filters are continuous, it sufices to show that

$$
\frac { \operatorname* { i n f } \{ z : g ( z ) \geq 1 - \delta \} } { \operatorname* { s u p } \{ z : g ( z ) \leq \delta \} } = \frac { g ^ { - 1 } ( 1 - \delta ) } { g ^ { - 1 } ( \delta ) } \geq 2
$$

for each family and stated value of �, which we verify below.

Gradient descent. The following ratio is increasing in �, hence taking � = 1 yields the lower bound

$$
{ \frac { g ^ { - 1 } ( 2 / 3 ) } { g ^ { - 1 } ( 1 / 3 ) } } = { \frac { 1 - ( 1 / 3 ) ^ { 1 / t } } { 1 - ( 2 / 3 ) ^ { 1 / t } } } \geq 2 .
$$

Iterated Tikhonov. The following ratio is decreasing in $p ,$ hence taking $p  \infty$ ，

$$
\frac { g ^ { - 1 } ( 2 / 3 ) } { g ^ { - 1 } ( 1 / 3 ) } = \frac { 3 ^ { 1 / p } - 1 } { ( 3 / 2 ) ^ { 1 / p } - 1 } \ge \frac { \log 3 } { \log ( 3 / 2 ) } \approx 2 . 7 1 .
$$

Ridge PCR. The ratio is always equal to 2.

Power-exponentialfilter. For $\delta = 1 / ( 2 ^ { p } + 1 )$ , we have

$$
{ \frac { g ^ { - 1 } ( 1 - \delta ) } { g ^ { - 1 } ( \delta ) } } = \left( { \frac { \log ( 2 ^ { p } + 1 ) } { \log ( 2 ^ { - p } + 1 ) } } \right) ^ { 1 / p } \geq \left( { \frac { 1 } { 2 ^ { - p } } } \right) ^ { 1 / p } = 2 .
$$

The proof is complete.

□

## E General Risk Upper Bounds

## E.1 Background on Schur multipliers

Matrix operator. A matrix operator is a linear map between matrices, i.e., a fourth-order tensor. We use ◦ to denote composition of matrix operators, or a matrix operator applied to a matrix, and ⊗ for the Kronecker product. For matrices A, B, and X of appropriate shape, we follow the convention that

$$
( \mathbf { A } \otimes \mathbf { B } ) \circ \mathbf { X } : = \mathbf { B } \mathbf { X } \mathbf { A } ^ { \top } .
$$

For matrices A, B, C, and D of appropriate shape, we have

$$
( \mathbf { A } \otimes \mathbf { B } ) \circ ( \mathbf { C } \otimes \mathbf { D } ) = ( \mathbf { A } \mathbf { C } ) \otimes ( \mathbf { B } \mathbf { D } ) .
$$

We use diag to convert a vector to a diagonal matrix, and a matrix to a vector of its diagonal entries, i.e., for a vector x and a squared matrix A,

$$
\mathrm { d i a g } ( \mathbf { x } ) : = \left[ \begin{array} { l l l l l } { \ddots } & { } & { } & { } & { } \\ { } & { \mathbf { x } _ { i } } & { } & { } \\ { } & { } & { } & { \ddots } \end{array} \right] , \quad \mathrm { d i a g } ( \mathbf { A } ) : = \left[ \begin{array} { l } { \vdots } \\ { \dot { \mathbf { A } } _ { i i } } \\ { \vdots } \\ { } \end{array} \right] .
$$

We use ⊙ for the Schur or Hadamard product, i.e., entry-wise multiplication; for matrices A, B and vector x of appropriate shape, we have

$$
( \mathbf { A } \odot \mathbf { B } ) \mathbf { x } = \mathrm { d i a g } \left( \mathbf { A } \mathrm { d i a g } ( \mathbf { x } ) \mathbf { B } ^ { \top } \right) .
$$

A matrix operator T is self-adjoint if $\langle \mathcal { T } \circ \mathbf { X } , \mathbf { Y } \rangle = \langle \mathbf { X } , \mathcal { T } \circ \mathbf { Y } \rangle$ for all X and Y of appropriate shape.

Remark 1 (Schur multiplier as a spectral operator). For symmetric matrices A, B, the left and right matrix multiplication operators, $\mathbf { I } \otimes \mathbf { A } .$ , and $\mathbf { B } ^ { \top } \otimes \mathbf { I } .$ , are both self-adjoint; additionally, they commute, that is,

$$
\mathbf { I } \otimes \mathbf { A } \circ \mathbf { B } ^ { \intercal } \otimes \mathbf { I } = \mathbf { B } ^ { \intercal } \otimes \mathbf { I } \circ \mathbf { I } \otimes \mathbf { A } = \mathbf { B } ^ { \intercal } \otimes \mathbf { A } .
$$

Hence they can be simultaneously diagonalized. Specifically, we have

$$
\mathbf { I } \otimes \mathbf { A } = \sum _ { i , j } \lambda _ { i } \big ( \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } \big ) \otimes \big ( \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \big ) , \quad \mathbf { B } ^ { \top } \otimes \mathbf { I } = \sum _ { i , j } \gamma _ { j } \big ( \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } \big ) \otimes \big ( \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \big ) .
$$

So the Schur multiplier $k ( \mathbf { A } , \mathbf { B } )$ can be understood as applying a scalar kernel function $k : \mathbb { R } \otimes \mathbb { R }  \mathbb { R }$ to left-right matrix multiplication operators $( \mathbf { I } \otimes \mathbf { A } , \mathbf { B } ^ { \top } \otimes \mathbf { I } )$ with operator eigenvalues $( \lambda _ { i } , \gamma _ { j } ) _ { i , j }$ . Therefore, the Schur multiplier $k ( \mathbf { A } , \mathbf { B } )$ is sometimes known as a double operator integral in the literature.

The Schur multiplier has the following equivalent forms:

$$
\begin{array} { r l } & { k ( \mathbf { A } , \mathbf { B } ) \circ \mathbf { X } = \displaystyle \sum _ { i , j } k ( \lambda _ { i } , \gamma _ { j } ) \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { X } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } } \\ { \quad } & { } \\ { \iff \quad } & { \mathbf { U } ^ { \top } \big ( k ( \mathbf { A } , \mathbf { B } ) \circ \mathbf { X } \big ) \mathbf { V } = \mathbf { K } \odot ( \mathbf { U } ^ { \top } \mathbf { X } \mathbf { V } ) , \mathrm { w h e r e } \mathbf { K } : = [ k ( \lambda _ { i } , \gamma _ { j } ) ] _ { i , j } \in \mathbb { R } ^ { n \times m } . } \end{array}
$$

So the canonical Schur multiplier $k ( \mathbf { A } , \mathbf { B } )$ in a matrix space is equivariant to the standard Schur multiplier associated with the spectral kernel matrix K in the rotated matrix space, i.e.,

$$
\begin{array} { r } { k ( \mathbf { A } , \mathbf { B } ) : \mathbb { R } ^ { n \times m }  \mathbb { R } ^ { n \times m } \longleftrightarrow \begin{array} { r } { S _ { \mathbf { K } } : \mathbf { U } ^ { \top } \big ( \mathbb { R } ^ { n \times m } \big ) \mathbf { V }  \mathbf { U } ^ { \top } \big ( \mathbb { R } ^ { n \times m } \big ) \mathbf { V } } \\ { \mathbf { X } \mapsto k ( \mathbf { A } , \mathbf { B } ) \circ \mathbf { X } } \end{array} \quad \Longleftrightarrow \quad S _ { \mathbf { K } } : \mathbf { U } ^ { \top } \big ( \mathbb { R } ^ { n \times m } \big ) \mathbf { V }  \mathbf { U } ^ { \top } \big ( \mathbb { R } ^ { n \times m } \big ) \mathbf { V } } \end{array}
$$

Remark 2 (Schur multiplier as a diagonal operator). Viewing the canonical Schur multiplier as a standard Schur multiplier in the rotated space and vectorizing, we have

$$
\operatorname { V e c } ( \mathbf { K } \odot \mathbf { Z } ) = \mathrm { d i a g } ( \operatorname { V e c } \mathbf { K } ) \operatorname { V e c } \mathbf { Z } .
$$

So the Schur multiplier can also be understood as a diagonal operator under the canonical vectorization.

Proof of Proposition 5.1. Following the introduced notation for the eigendecomposition of A, B, we have

$$
\mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { A } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } = \lambda _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } , \quad \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { B } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } = \gamma _ { j } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } ,
$$

which implies

$$
\begin{array} { r l } { f ^ { [ 1 ] } ( \mathbf { A } , \mathbf { B } ) \circ ( \mathbf { A } - \mathbf { B } ) = } & { \displaystyle \sum _ { i , j } f ^ { [ 1 ] } ( \lambda _ { i } , \gamma _ { j } ) \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } ( \mathbf { A } - \mathbf { B } ) \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } } \\ & { = \displaystyle \sum _ { i , j } f ^ { [ 1 ] } ( \lambda _ { i } , \gamma _ { j } ) ( \lambda _ { i } - \gamma _ { j } ) \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } } \\ & { = \displaystyle \sum _ { i , j } \big ( f ( \lambda _ { i } ) - f ( \gamma _ { j } ) \big ) \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } } \\ & { = f ( \mathbf { A } ) - f ( \mathbf { B } ) . } \end{array}
$$

Thus $f ^ { [ 1 ] } ( \mathbf { A } , \mathbf { B } )$ is a matrix divided diference.

A similar calculation leads to the following identities.

Corollary E.1 (chain rule). For all symmetric matrices A, B, and any compatible matrix C, we have

$$
f ( \mathbf { A } ) \mathbf { C } - \mathbf { C } f ( \mathbf { B } ) = f ^ { [ 1 ] } ( \mathbf { A } , \mathbf { B } ) \circ ( \mathbf { A C } - \mathbf { C B } ) .
$$

In particular, we have

$$
\big ( f ( \mathbf { A } ) - f ( \mathbf { B } ) \big ) \mathbf { B } = r ( \mathbf { A } , \mathbf { B } ) \circ ( \mathbf { A } - \mathbf { B } ) \quad f o r t h e k e r n e l \textit { r } : ( x , y ) \mapsto y f ^ { [ 1 ] } ( x , y ) ;
$$

$$
\mathbf { A } \big ( f ( \mathbf { A } ) - f ( \mathbf { B } ) \big ) \mathbf { B } = s ( \mathbf { A } , \mathbf { B } ) \circ ( \mathbf { A } - \mathbf { B } ) \quad f o r t h e k e r n e l \textit { s } : ( x , y ) \mapsto x y f ^ { [ 1 ] } ( x , y ) .
$$

The following result is extremely useful for bounding Schur norms of various kernels.

Proposition E.2 (factorization rule). If kernel $k ^ { \prime }$ admits a factorization $k ^ { \prime } ( x , y ) : = a ( x ) k ( x , y ) b ( y )$ , then

$$
\begin{array} { r } { \| k ^ { \prime } ( I , J ) \| _ { \infty } \leq \| a ( I ) \| _ { \infty } \| b ( J ) \| _ { \infty } \| k ( I , J ) \| _ { \infty } . } \end{array}
$$

Proof of Proposition E.2. Taking any finite subsets $I , J ,$ we have

$$
k ^ { \prime } ( I , J ) \odot \mathbf { Z } = \big ( \mathrm { d i a g } ( a ( I ) ) k ( I , J ) \mathrm { d i a g } ( b ( J ) ) \big ) \odot \mathbf { Z } = \mathrm { d i a g } ( a ( I ) ) \big ( k ( I , J ) \odot \mathbf { Z } \big ) \mathrm { d i a g } ( b ( J ) ) ,
$$

so the observation follows by taking the operator norm and supremum over subsets $I , J .$

## E.2 Useful lemmas

ProofofExample 2. For ridge, we have $r ( x , y ) = 1$ , so the Schur norm is 1.

GD. We only give the case when � is even for simplicity. For GD, the kernel is

$$
\begin{array} { r l r } { r ( x , y ) : = \displaystyle \frac { g ( x ) - g ( y ) } { g ^ { * } ( x ) - g ^ { * } ( y ) } } \\ & { = \displaystyle \frac { ( 1 - \eta y ) ^ { t } - ( 1 - \eta x ) ^ { t } } { x / \big ( x + 1 / ( \eta t ) \big ) - y / \big ( y + 1 / ( \eta t ) \big ) } } \\ & { = \displaystyle t \frac { ( 1 - \eta y ) ^ { t } - ( 1 - \eta x ) ^ { t } } { \eta x - \eta y } \bigg ( \eta x + \frac { 1 } { t } \bigg ) \bigg ( \eta y + \frac { 1 } { t } \bigg ) , \qquad } & & { 0 \le x , y \le \frac { 1 } { \eta } . } \end{array}
$$

Without loss of generality, assume $\eta = 1 ;$ otherwise, replace �� and �� by � and � respectively. Observe that

$$
\begin{array} { r l r } { r ( x , y ) : = t \displaystyle \frac { ( 1 - y ) ^ { t } - ( 1 - x ) ^ { t } } { x - y } \bigg ( x + \frac 1 t \bigg ) \bigg ( y + \frac 1 t \bigg ) } & { \qquad } & \\ { = t \displaystyle \frac { \sum _ { s = 0 } ^ { t - 1 } ( 1 - y ) ^ { s } ( x - y ) ( 1 - x ) ^ { t - 1 - s } } { x - y } \bigg ( x + \frac 1 t \bigg ) \bigg ( y + \frac 1 t \bigg ) } & { \qquad } & \\ { = t \displaystyle \sum _ { s = 0 } ^ { t - 1 } \bigg ( x + \frac 1 t \bigg ) ( 1 - x ) ^ { t - 1 - s } ( 1 - y ) ^ { s } \bigg ( y + \frac 1 t \bigg ) , } & { \qquad } & { 0 \le x , y \le 1 . } \end{array}
$$

By symmetry, we have

$$
r ( x , y ) = r ^ { * } ( x , y ) + r ^ { * } ( y , x ) , \quad r ^ { * } ( x , y ) : = t \sum _ { s = 0 } ^ { t / 2 - 1 } \bigg ( x + \frac { 1 } { t } \bigg ) ( 1 - x ) ^ { t - 1 - s } ( 1 - y ) ^ { s } \bigg ( y + \frac { 1 } { t } \bigg ) .
$$

So $\| r ( I , I ) \| _ { \mathfrak { S } } \leq 2 \| r ^ { * } ( I , I ) \| _ { \mathfrak { S } }$ , and it sufices to bound the right-hand side. Write

$$
\begin{array} { r l } & { r ^ { * } ( x , y ) : = t \displaystyle \sum _ { s = 0 } ^ { t / 2 - 1 } \bigg ( x + \frac { 1 } { t } \bigg ) ( 1 - x ) ^ { t - 1 - s } ( 1 - y ) ^ { s } \bigg ( y + \frac { 1 } { t } \bigg ) } \\ & { \qquad = t \displaystyle \sum _ { s = 0 } ^ { t / 2 - 1 } \bigg ( x + \frac { 1 } { t } \bigg ) ( 1 - x ) ^ { t - 1 - s } ( 1 - y ) ^ { s } y + \underbrace { \sum _ { s = 0 } ^ { t / 2 - 1 } \bigg ( x + \frac { 1 } { t } \bigg ) ( 1 - x ) ^ { t - 1 - s } ( 1 - y ) ^ { s } } _ { r _ { 1 } ^ { * } } . } \end{array}
$$

For the second component, we have

$$
\| r _ { 2 } ^ { * } ( I , I ) \| \mathfrak { g } \leq \sum _ { s = 0 } ^ { t / 2 - 1 } \operatorname* { s u p } _ { 0 \leq x \leq 1 } \left( x + \frac { 1 } { t } \right) ( 1 - x ) ^ { t - 1 - s } \operatorname* { s u p } _ { 0 \leq y \leq 1 } ( 1 - y ) ^ { s } \qquad \quad \mathrm { P r o p o s i t i o n ~ E . 2 }
$$

$$
\begin{array} { l } { \displaystyle \leq \frac { t } { 2 } \operatorname* { s u p } _ { 0 \leq x \leq 1 } \left( x + \frac { 1 } { t } \right) ( 1 - x ) ^ { t / 2 } } \\ { \leq \displaystyle \frac { t } { 2 } \left( \frac { 2 } { t } + \frac { 1 } { t } \right) = \frac { 3 } { 2 } . } \end{array}
$$

For the first component, by Abel summation, we have

$$
\begin{array} { l } { { r _ { 1 } ^ { * } ( x , y ) : = t \bigg ( x + \frac { 1 } { t } \bigg ) \displaystyle \sum _ { s = 0 } ^ { t / 2 - 1 } ( 1 - x ) ^ { t - 1 - s } ( 1 - y ) ^ { s } y } } \\ { { = t \bigg ( x + \frac { 1 } { t } \bigg ) \displaystyle \sum _ { s = 0 } ^ { t / 2 - 1 } ( 1 - x ) ^ { t - 1 - s } \big ( ( 1 - y ) ^ { s } - ( 1 - y ) ^ { s + 1 } \big ) } } \\ { { = t \bigg ( x + \frac { 1 } { t } \bigg ) \Bigg ( \displaystyle \sum _ { s = 0 } ^ { t / 2 - 1 } \big ( ( 1 - x ) ^ { t - 1 - s } - ( 1 - x ) ^ { t - s } \big ) ( 1 - y ) ^ { s } + ( 1 - x ) ^ { t } - ( 1 - x ) ^ { t / 2 } ( 1 - y ) ^ { t / 2 } \bigg ) } } \\ { { = t \displaystyle \sum _ { s = 0 } ^ { t / 2 - 1 } \bigg ( x + \frac { 1 } { t } \bigg ) x ( 1 - x ) ^ { t - 1 - s } ( 1 - y ) ^ { s } + t \bigg ( x + \frac { 1 } { t } \bigg ) ( 1 - x ) ^ { t } - t \bigg ( x + \frac { 1 } { t } \bigg ) ( 1 - x ) ^ { t / 2 } ( 1 - y ) ^ { t / 2 } . } } \end{array}
$$

Then by Proposition E.2 we have

$$
\begin{array} { r l } & { \| r _ { 1 } ^ { * } ( I , I ) \| \vert \vert \varphi } \\ & { \leq t \displaystyle \sum _ { s = 0 } ^ { t / 2 - 1 } \displaystyle \operatorname* { s u p } _ { x } \left( x + \frac { 1 } { t } \right) x ( 1 - x ) ^ { t - 1 - s } \operatorname* { s u p } _ { y } ( 1 - y ) ^ { s } } \\ & { \qquad + t \operatorname* { s u p } _ { x } \left( x + \frac { 1 } { t } \right) ( 1 - x ) ^ { t } + t \operatorname* { s u p } _ { x } \left( x + \frac { 1 } { t } \right) ( 1 - x ) ^ { t / 2 } \operatorname* { s u p } _ { y } ( 1 - y ) ^ { t / 2 } \quad \mathrm { P r o p o s i t i o n ~ E . 2 } } \\ & { \leq \frac { t ^ { 2 } } { 2 } \operatorname* { s u p } \left( x + \frac { 1 } { t } \right) x ( 1 - x ) ^ { t / 2 } + 2 t \operatorname* { s u p } _ { x } \left( x + \frac { 1 } { t } \right) ( 1 - x ) ^ { t / 2 } \qquad \quad s \leq t / 2 - 1 } \\ & { \leq \frac { t ^ { 2 } } { 2 } \left( \frac { 1 6 } { t ^ { 2 } } + \frac { 2 } { t ^ { 2 } } \right) + 2 t \left( \frac { 2 } { t } + \frac { 1 } { t } \right) = 1 5 . } \end{array}
$$

Putting things together, we have

$$
\begin{array} { r } { \| r ( I , I ) \| _ { \mathfrak { S } } \le 2 \| r ^ { * } ( I , I ) \| _ { \mathfrak { S } } \le 2 \| r _ { 1 } ^ { * } ( I , I ) \| _ { \mathfrak { S } } + 2 \| r _ { 2 } ^ { * } ( I , I ) \| _ { \mathfrak { S } } \le 3 3 . } \end{array}
$$

PCR. We prove a slightly more general result. For $\epsilon \in ( 0 , 1 ]$ , setting $I = \mathbb { R } _ { \geq 0 }$ and $H = [ ( 1 + \epsilon ) \rho , \infty )$ , we claim that

$$
\| r ( I , H ) \| _ { \mathfrak { S } } \le 2 + \frac { 4 } { \epsilon } , \quad \| r ( H , H ) \| _ { \mathfrak { S } } = 0 , \quad \| r ( I , \{ 0 \} ) \| _ { \mathfrak { S } } \le 2 .
$$

Let $I _ { 0 } : = ( \rho , \infty ) , I _ { 1 } : = [ 0 , \rho ]$ so that $I = I _ { 0 } \cup I _ { 1 }$ . Choose the diagonal entries of � to be zero. For $x \in I$ and $y \in H$ , we have

$$
r ( x , y ) : = \frac { g ( x ) - g ( y ) } { g ^ { * } ( x ) - g ^ { * } ( y ) } = \left\{ \begin{array} { l l } { \frac { 1 } { g ^ { * } ( y ) - g ^ { * } ( x ) } } & { x \in I _ { 1 } , } \\ { 0 } & { x \in I _ { 0 } . } \end{array} \right.
$$

In particular, $r ( H , H ) = 0$ . Moreover,

$$
r ( x , 0 ) = \left\{ \begin{array} { l l } { \displaystyle \frac { x + \rho } { x } } & { x > \rho , } \\ { 0 } & { 0 \leq x \leq \rho , } \end{array} \right. \quad \Longrightarrow \quad \| r ( I , \{ 0 \} ) \| _ { \mathfrak E } = \operatorname* { s u p } _ { x > \rho } \frac { x + \rho } { x } \leq 2 .
$$

It remains to bound $\lVert \boldsymbol { r } ( I , H ) \rVert _ { \mathfrak { S } }$ . Since the rows indexed by $I _ { 0 }$ vanish, $\| r ( I , H ) \| _ { \mathfrak { S } } = \| r ( I _ { 1 } , H ) \| _ { \mathfrak { S } }$ . For $x \in I _ { 1 }$ and $y \in H ,$ , we have

$$
g ^ { * } ( x ) \leq \frac { 1 } { 2 } , \quad g ^ { * } ( y ) \geq \frac { 1 + \epsilon } { 2 + \epsilon } > \frac { 1 } { 2 } ,
$$

and therefore we have the absolutely convergent expansion

$$
r ( x , y ) = \frac { 1 } { g ^ { * } ( y ) - g ^ { * } ( x ) } = \sum _ { \ell = 0 } ^ { \infty } g ^ { * } ( x ) ^ { \ell } g ^ { * } ( y ) ^ { - ( \ell + 1 ) } .
$$

Applying Proposition E.2 gives

$$
\begin{array} { r } { \| r ( I _ { 1 } , H ) \| _ { \mathfrak { S } } \le \displaystyle \sum _ { \ell = 0 } ^ { \infty } \Big \| g ^ { * } ( I _ { 1 } ) ^ { \ell } \Big \| _ { \infty } \Big \| g ^ { * } ( H ) ^ { - ( \ell + 1 ) } \Big \| _ { \infty } } \\ { \le \displaystyle \sum _ { \ell = 0 } ^ { \infty } \left( \frac { 1 } { 2 } \right) ^ { \ell } \left( \frac { 2 + \epsilon } { 1 + \epsilon } \right) ^ { \ell + 1 } = 2 + \frac { 4 } { \epsilon } , } \end{array}
$$

as claimed.

We define the auxiliary kernels

$$
h : ( x , y ) \mapsto \frac { x y } { \lambda } g ^ { [ 1 ] } ( x , y ) , \quad s : ( x , y ) \mapsto ( x + \lambda ) y \psi ^ { [ 1 ] } ( x , y ) .
$$

Lemma E.3. For any two index sets $I , J \subset \mathbb { R } _ { \geq 0 } ,$ , we have

$$
\begin{array} { r } { \| h ( I , J ) \| _ { \mathfrak { S } } \le \| r ( I , J ) \| _ { \mathfrak { S } } , \quad \| s ( I , J ) \| _ { \mathfrak { S } } \le \| r ( I , J ) \| _ { \mathfrak { S } } + \| r ( I , \{ 0 \} ) \| _ { \mathfrak { S } } . } \end{array}
$$

In particular, we have $\| s ( I , J ) \| _ { \mathfrak { S } } \leq 2 \| r ( I , J ) \| _ { \mathfrak { S } } i f 0 \in J .$

Proof of Lemma E.3. Note that

$$
r ( x , 0 ) = \frac { g ( x ) } { g ^ { * } ( x ) } = ( x + \lambda ) \frac { g ( x ) } { x } = ( x + \lambda ) \psi ( x ) .
$$

The following identities are straightforward to verify:

$$
h ( x , y ) = \frac { x } { x + \lambda } r ( x , y ) \frac { y } { y + \lambda } , s ( x , y ) = r ( x , y ) \frac { \lambda } { y + \lambda } - r ( x , 0 ) .
$$

The claims follow from Proposition E.2.

For the next lemma, we recall the definition of $\tilde { \alpha } _ { k }$ from (8).

Lemma E.4. Let $\mathbf { u } ^ { * }$ be a vector measurable w.r.t. $\mathbf { X } _ { \leq k }$ , and let $k : I \otimes J $ R be afixed kernel matrix. Then with probability at least $1 - \delta - \exp ( - n / c _ { 0 } )$ , it holds that $i f \sigma ( \bar { \mathbf { A } } ) \subseteq I$ and $\sigma ( \bar { \mathbf { A } } _ { \leq k } ) \subseteq J ,$ then

$$
\left\| \big ( k ( \bar { \mathbf { A } } , \bar { \mathbf { A } } _ { \leq k } ) \circ \bar { \mathbf { A } } _ { > k } \big ) \mathbf { u } ^ { * } \right\| \leq \| k ( I , J ) \| _ { \mathfrak { S } } \cdot c _ { 1 } \sqrt { \tilde { \alpha } _ { k } } \| \mathbf { u } ^ { * } \| .
$$

Proof of Lemma $E . 4 .$ Let the eigendecompositions of $\bar { \bf A }$ and $\bar { \mathbf { A } } _ { \leq k }$ be

$$
\bar { \mathbf { A } } = \mathbf { U } \hat { \mathbf { A } } \mathbf { U } ^ { \top } = \sum _ { i } \hat { \lambda } _ { i } \mathbf { u } _ { i } \mathbf { u } _ { i } ^ { \top } , \quad \bar { \mathbf { A } } _ { \leq k } = \mathbf { V } \hat { \mathbf { I } } \mathbf { V } ^ { \top } = \sum _ { j } \hat { \gamma } _ { j } \mathbf { v } _ { j } \mathbf { v } _ { j } ^ { \top } .
$$

By Proposition 5.2, for any $\epsilon > 0$ there exists a Hilbert space H and Schur factorization

$$
\mathbf { f } : I \to \mathbb { H } , \ \mathbf { g } : J \to \mathbb { H } , \quad { \mathrm { w i t h ~ } } \ a : = \operatorname* { s u p } _ { x \in I } \| \mathbf { f } ( x ) \| < \infty , \ b : = \operatorname* { s u p } _ { y \in J } \| \mathbf { g } ( y ) \| < \infty ,
$$

such that

$$
{ \mathrm { f o r ~ a l l ~ } } x \in I { \mathrm { ~ a n d ~ } } y \in J , k ( x , y ) = \langle \mathbf { f } ( x ) , \mathbf { g } ( y ) \rangle , { \mathrm { ~ a n d ~ } } a b \leq \| k ( I , J ) \| _ { \mathfrak { S } } + \epsilon .
$$

We may extend � by zero outside $I \times J ,$ , and $f , g$ by zero outside $I , J ,$ , respectively, so that the factorization is preserved. Define the following submatrices

$$
\mathbf { K } : = \left[ k ( \hat { \lambda } _ { i } , \hat { \gamma } _ { j } ) \right] _ { i j } , \mathbf { F } : = \left[ \begin{array} { l } { \vdots } \\ { \vdots } \\ { \mathbf { f } ^ { \top } } \\ { \vdots } \end{array} \right] , \mathbf { G } : = \left[ \begin{array} { l } { \vdots } \\ { \vdots } \\ { \mathbf { g } _ { j } ^ { \top } } \\ { \vdots } \end{array} \right] , \mathrm { w h e r e } \mathbf { f } _ { i } : = \mathbf { f } ( \hat { \lambda } _ { i } ) , \mathbf { g } _ { j } : = \mathbf { g } ( \hat { \gamma } _ { j } ) \in \mathbb { H } .
$$

Then we have

$$
\mathbf { K } = \mathbf { F } \mathbf { G } ^ { \top } , \quad \operatorname* { s u p } _ { i } \left\| \mathbf { f } _ { i } \right\| \leq a , \quad \operatorname* { s u p } _ { j } \left\| \mathbf { g } _ { j } \right\| \leq b .
$$

Furthermore, let

$$
\mathbf { v } ^ { * } : = \mathbf { V } ^ { \top } \mathbf { u } ^ { * } .
$$

Under this setup, we have

$$
\begin{array} { r l } &  \begin{array} { r l } & { \langle i \langle \mathbf { A } , \mathbf { A } _ { 2 2 } \rangle \rangle \leq \mathbf { A } _ { \mathrm { , x } + 1 } \| \mathbf { a } ^ { * } - \boldsymbol { \mathrm { \mathrm {  ~ \scriptstyle \alpha ~ } } } \boldsymbol { \mathrm { \scriptstyle \mathrm { \Lambda } } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } } \\ & { \qquad = \| \| \mathbf { K } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } } \\ & { \qquad - \| \mathbf { d i \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } } \\ & { \qquad - \| \mathbf { d i \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } ^ { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm { \Lambda } } \boldsymbol { \mathrm \Lambda } ^ { \mathrm { \Lambda } } } \\ { = } &  \frac   \end{array} \end{array}
$$

In what follows, we treat $\mathbf { X } { } _ { \leq k }$ as fixed and only consider the randomness of $\mathbf { X } _ { > k }$ . We control the $L ^ { p } ( { \mathbf { X } } _ { > k } )$ -norm of the random variable on the right-hand side. Recall that $\mathbf { V } , \mathbf { v } ^ { * }$ , and G depend only on $\mathbf { X } { } _ { \leq k }$ and are independent of $\mathbf { X } _ { > k }$ . Define

$$
\tilde { \alpha } : = \alpha _ { k } + \frac { p } { n } \lambda _ { k + 1 } ^ { 2 } , \quad p : = \operatorname* { m a x } \{ 4 , \log ( 1 / \delta ) \} \le n / c _ { 0 } .
$$

Applying (9) at order $2 p ,$ , followed by Minkowski’s inequality, we have

$$
\begin{array} { r l r } { \left\| \| \bar { \mathbf { A } } _ { > k } \nabla \mathrm { d i a g } ( \mathbf { v } ^ { * } ) \mathbf { G } \| _ { F } ^ { 2 } \right\| _ { L ^ { p } ( \mathbf { X } _ { > k } ) } \leq c _ { 1 } ^ { 2 } \widetilde { \alpha } \| \nabla \mathrm { d i a g } ( \mathbf { v } ^ { * } ) \mathbf { G } \| _ { F } ^ { 2 } } & { \qquad } & { \mathrm { L e m m a ~ A . 3 } } \\ & { } & { = c _ { 1 } ^ { 2 } \widetilde { \alpha } \| \mathrm { d i a g } ( \mathbf { v } ^ { * } ) \mathbf { G } \| _ { F } ^ { 2 } } & { \qquad } & { \nabla ^ { \top } \mathbf { V } = \mathbf { I } } \\ & { } & { = c _ { 1 } ^ { 2 } \widetilde { \alpha } \displaystyle \sum _ { j } \mathbf { v } _ { j } ^ { * 2 } \| \mathbf { g } _ { j } \| ^ { 2 } } \\ & { } & { \leq c _ { 1 } ^ { 2 } \widetilde { \alpha } b ^ { 2 } \| \mathbf { v } ^ { * } \| ^ { 2 } } & { \qquad } & { \displaystyle \operatorname* { s u p } \| \mathbf { g } _ { j } \| \leq b } \\ & { } & { = c _ { 1 } ^ { 2 } \widetilde { \alpha } b ^ { 2 } \| \mathbf { u } ^ { * } \| ^ { 2 } . } \end{array}
$$

Putting them together, we have

$$
\begin{array} { r l r l } & { \left\| \big ( k ( \bar { \mathbf { A } } , \bar { \mathbf { A } } _ { \leq k } ) \circ \bar { \mathbf { A } } _ { > k } \big ) \mathbf { u } ^ { * } \right\| _ { L ^ { p } ( \mathbf { X } _ { > k } ) } \leq a \Big \| \| \bar { \mathbf { A } } _ { > k } \mathbf { V } \mathrm { d i a g } ( \mathbf { v } ^ { * } ) \mathbf { G } \| _ { F } \| _ { L ^ { p } ( \mathbf { X } _ { > k } ) } } & & { } \\ & { \qquad \leq a b c _ { 1 } \sqrt { \bar { \alpha } } \| \mathbf { u } ^ { * } \| } & & { } \\ & { \qquad \leq \left( \| k ( I , J ) \| _ { \mathfrak E } + \epsilon \right) c _ { 1 } \sqrt { \bar { \alpha } } \| \mathbf { u } ^ { * } \| . } & & { \qquad a b \leq \| k ( I , J ) \| _ { \mathfrak E } + \epsilon } \end{array}
$$

Taking $\epsilon  0 .$ , we have

$$
\left\| \left( k ( \bar { \mathbf { A } } , \bar { \mathbf { A } } _ { \leq k } ) \circ \bar { \mathbf { A } } _ { > k } \right) \mathbf { u } ^ { * } \right\| _ { L ^ { p } ( \mathbf { X } _ { > k } ) } \leq \| k ( I , J ) \| _ { \mathfrak { S } ^ { c _ { 1 } } } \sqrt { \tilde { \alpha } } \| \mathbf { u } ^ { * } \| .
$$

Since $\tilde { \alpha } \leq c \tilde { \alpha } _ { k } .$ , applying Markov’s inequality and rescaling constants gives the claim.

Lemma E.5. Let � and � be such that

$$
k \leq n / c _ { 3 } , \quad \tilde { \lambda } : = \lambda + \frac { \sum _ { i > k } \lambda _ { i } } { n } \geq c _ { 2 } \lambda _ { k + 1 } .
$$

Under Lemmas A.1 and A.2, we have

$$
\left\| \frac { 1 } { \sqrt { n } } \pmb { \Sigma } ^ { 1 / 2 } \mathbf { X } ^ { \top } ( \bar { \mathbf { A } } + \lambda \mathbf { I } ) ^ { - 1 } \right\| \leq 9 .
$$

ProofofLemma E.5. We have

$$
\begin{array} { r l } { \bigg \| \displaystyle \frac { 1 } { \sqrt { n } } \boldsymbol { \Sigma } ^ { 1 / 2 } \boldsymbol { \mathbf { X } } ^ { \top } ( \bar { \mathbf { A } } + \lambda \boldsymbol { \mathbf { I } } ) ^ { - 1 } \bigg \| \leq \bigg \| \displaystyle \frac { 1 } { \sqrt { n } } \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \boldsymbol { \mathbf { X } } _ { \leq k } ^ { \top } ( \bar { \mathbf { A } } + \lambda \boldsymbol { \mathbf { I } } ) ^ { - 1 } \bigg \| + \bigg \| \displaystyle \frac { 1 } { \sqrt { n } } \boldsymbol { \Sigma } _ { > k } ^ { 1 / 2 } \boldsymbol { \mathbf { X } } _ { > k } ^ { \top } ( \bar { \mathbf { A } } + \lambda \boldsymbol { \mathbf { I } } ) ^ { - 1 } \bigg \| } & { } \\ { \leq \sqrt { 2 } \| \bar { \mathbf { A } } _ { \leq k } ( \bar { \mathbf { A } } + \lambda \boldsymbol { \mathbf { I } } ) ^ { - 1 } \| + \sqrt { \lambda _ { k + 1 } } \| \bar { \mathbf { A } } _ { > k } ^ { 1 / 2 } ( \bar { \mathbf { A } } + \lambda \boldsymbol { \mathbf { I } } ) ^ { - 1 } \| . } & { \quad \mathrm { L e m m a ~ A . l } } \end{array}
$$

By Lemma A.2, we have

$$
\frac 1 2 \tilde { \lambda } { \bf I I } \preceq \bar { \bf A } _ { > k } + \lambda { \bf I } \preceq 2 \tilde { \lambda } { \bf I I } \quad \implies \quad \bar { \bf A } + \lambda { \bf I } \succeq \frac 1 2 \tilde { \lambda } { \bf I I } .
$$

So the first term is bounded by

$$
\left\| \bar { \mathbf { A } } _ { \leq k } ( \bar { \mathbf { A } } + \lambda \mathbf { I } ) ^ { - 1 } \right\| \leq 1 + \left\| ( \bar { \mathbf { A } } _ { > k } + \lambda \mathbf { I } ) ( \bar { \mathbf { A } } + \lambda \mathbf { I } ) ^ { - 1 } \right\| \leq 5 ,
$$

and the second term is bounded by

$$
\sqrt { \lambda _ { k + 1 } } \big \| \bar { \mathbf { A } } _ { > k } ^ { 1 / 2 } ( \bar { \mathbf { A } } + \lambda \mathbf { I } ) ^ { - 1 } \big \| \leq \sqrt { \lambda _ { k + 1 } } \big \| ( \bar { \mathbf { A } } + \lambda \mathbf { I } ) ^ { - 1 / 2 } \big \| \leq \sqrt { 2 \lambda _ { k + 1 } / \tilde { \lambda } } \leq \sqrt { 2 / c _ { 2 } } .
$$

This completes the proof.

## E.3 Proof of Theorem 5.3

We introduce some additional notation. It is more convenient to work with the normalized Gram matrices

$$
\bar { \mathbf { A } } : = \frac { 1 } { n } \mathbf { X } \mathbf { X } ^ { \top } , \quad \bar { \mathbf { A } } _ { \leq k } : = \frac { 1 } { n } \mathbf { X } _ { \leq k } \mathbf { X } _ { \leq k } ^ { \top } , \quad \bar { \mathbf { A } } _ { > k } : = \frac { 1 } { n } \mathbf { X } _ { > k } \mathbf { X } _ { > k } ^ { \top } .
$$

Proof of Theorem 5.3. The bias iterate is

$$
\left( \mathbf { I } - \frac { 1 } { n } \mathbf { X } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } \right) \mathbf { w } ^ { * } = \left[ \left( \mathbf { I } - \frac { 1 } { n } \mathbf { X } _ { \leq k } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } _ { \leq k } \right) \mathbf { w } _ { \leq k } ^ { * } \right] + \left[ \begin{array} { c } { - \frac { 1 } { n } \mathbf { X } _ { \leq k } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } _ { > k } \mathbf { w } _ { > k } ^ { * } } \\ { - \frac { 1 } { n } \mathbf { X } _ { > k } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } _ { \leq k } \mathbf { w } _ { \leq k } ^ { * } } \end{array} \right] + \left[ \begin{array} { c } { \left( \mathbf { I } - \frac { 1 } { n } \mathbf { X } _ { > k } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } _ { > k } \right) \mathbf { w } _ { > k } ^ { * } } \\ { \left( \mathbf { I } - \frac { 1 } { n } \mathbf { X } _ { > k } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } _ { > k } \right) \mathbf { w } _ { > k } ^ { * } } \end{array} \right]
$$

Then the bias error is

$$
\begin{array} { r } { \mathrm { B i a s } = \left\| \left( \mathbf { I } - \frac { 1 } { n } \mathbf { X } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } \right) \mathbf { w } ^ { * } \right\| _ { \Sigma } \leq \left\| \left( \mathbf { I } - \frac { 1 } { n } \mathbf { X } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } \right) \left[ \mathbf { w } _ { \leq k } ^ { * } \right] \right\| _ { \Sigma } + \left\| \left( \mathbf { I } - \frac { 1 } { n } \mathbf { X } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } \right) \left[ \mathbf { w } _ { > k } ^ { * } \right] \right\| _ { \Sigma } . } \\ { \mathrm { B i a s } _ { 1 } } \end{array}
$$

Here, we use the unsquared bias. The first term is further decomposed as

$$
\begin{array} { r l } { \mathrm { P i a s } _ { 1 } : = \| \mathbf { Z } ^ { 1 / 2 } \Bigg ( \mathbf { I } - \frac { 1 } { n } \mathbf { X } ^ { \top } \boldsymbol { \psi } ( \hat { \mathbf { A } } _ { \leq k } ) \mathbf { X } \Bigg ) [ \mathbf { \Phi } _ { 0 } ^ { \mathrm { { S } } } \boldsymbol { \hat { \epsilon } } ] \| } & { } \\ & { \leq \| \mathbf { Z } ^ { 1 / 2 } \Bigg ( \mathbf { I } - \frac { 1 } { n } \mathbf { X } ^ { \top } \boldsymbol { \psi } ( \hat { \mathbf { A } } _ { \leq k } ) \mathbf { X } \Bigg ) [ \mathbf { \Phi } _ { 0 } ^ { \mathrm { { S } } } \boldsymbol { \hat { \epsilon } } ] \| + \| \mathbf { Z } ^ { 1 / 2 } \frac { 1 } { n } \mathbf { X } ^ { \top } ( \boldsymbol { \psi } ( \hat { \mathbf { A } } ) - \boldsymbol { \psi } ( \hat { \mathbf { A } } _ { \leq k } ) ) \mathbf { X } [ \mathbf { \Phi } _ { 0 } ^ { \mathrm { S } } \boldsymbol { \hat { \epsilon } } ] \| } \\ & { = \| [ \mathbf { Z } ^ { 1 / 2 } ( \mathbf { I } - \frac { 1 } { n } \mathbf { X } _ { \leq k } ^ { \top } \boldsymbol { \psi } ( \hat { \mathbf { A } } _ { \leq k } ) \mathbf { X } _ { \leq k } ) \mathbf { \Phi } _ { \leq k } ^ { \mathrm { S } } ] \| + \| \mathbf { Z } ^ { 1 / 2 } \frac { 1 } { n } \mathbf { X } ^ { \top } ( \boldsymbol { \psi } ( \hat { \mathbf { A } } ) - \boldsymbol { \psi } ( \hat { \mathbf { A } } _ { \leq k } ) ) \mathbf { X } _ { \leq k } \mathbf { \Phi } _ { \leq k } ^ { \mathrm { S } } \| } \\ &  \leq \underbrace  \| \mathbf { Z } ^ { 1 / 2 } \boldsymbol { \psi } ^ { \mathrm { { S } } } \boldsymbol { \psi } ( \hat { \mathbf { A } } _ { \leq k } ) \mathbf { X } _ { \leq k } \boldsymbol  \end{array}
$$

Leave-tail-out error. For the LTO error, let

$$
\begin{array} { r } { \mathbf { u } _ { \leq k } ^ { * } : = \sqrt { n } \mathbf { X } _ { \leq k } ( \mathbf { X } _ { \leq k } ^ { \top } \mathbf { X } _ { \leq k } ) ^ { - 1 } \mathbf { w } _ { \leq k } ^ { * } \quad \implies \quad \| \mathbf { u } _ { \leq k } ^ { * } \| \leq \sqrt { 2 } \| \mathbf { w } _ { \leq k } ^ { * } \| _ { \Sigma _ { \leq k } ^ { - 1 } } , } \end{array}
$$

which is independent of $\bar { \mathbf { A } } _ { > k }$ , then we have

$$
\begin{array} { r l r l } & { \mathrm { L T O : } = \displaystyle \frac { 1 } { \sqrt n } \big \| \Sigma ^ { 1 / 2 } { \mathbf { X } } ^ { \top } \big ( \psi ( \bar { { \mathbf { A } } } ) - \psi ( \bar { { \mathbf { A } } } _ { \leq k } ) \big ) \bar { { \mathbf { A } } } _ { \leq k } { \mathbf { u } } _ { \leq k } ^ { * } \big \| } & & { } \\ & { \leq 9 \big \| \big ( \bar { { \mathbf { A } } } + \lambda \mathbf { I } \big ) \big ( \psi ( \bar { { \mathbf { A } } } ) - \psi ( \bar { { \mathbf { A } } } _ { \leq k } ) \big ) \bar { { \mathbf { A } } } _ { \leq k } { \mathbf { u } } _ { \leq k } ^ { * } \big \| } & & { \mathrm { L e m m a ~ E . } 5 } \\ & { = 9 \big \| \big ( s ( \bar { { \mathbf { A } } } , \bar { { \mathbf { A } } } _ { \leq k } ) \circ \bar { { \mathbf { A } } } _ { \leq k } \big ) { \mathbf { u } } _ { \leq k } ^ { * } \big \| } & & { \mathrm { C o r o l l a r y ~ E . 1 ~ a n d ~ } s ( x , y ) : = ( x + \lambda ) y \psi ^ { [ 1 ] } ( x , y ) } \\ & { \leq 9 c _ { 1 } \sqrt { \widetilde { \alpha } _ { k } } \big \| s ( I , H ) \big \| _ { \mathfrak { S } } \cdot \big \| { \mathbf { u } } _ { \leq k } ^ { * } \big \| } & & { \mathrm { L e m m a ~ E . 4 ~ } } \\ & { \leq 9 \sqrt { 2 } c _ { 1 } \sqrt { \widetilde { \alpha } _ { k } } \big \| s ( I , H ) \big \| _ { \mathfrak { S } } \cdot \big \| { \mathbf { w } } _ { \leq k } ^ { * } \big \| _ { { \Sigma } _ { \leq k } ^ { - 1 } } . } & &  \big \| { \mathbf { u } } _ { \leq k } ^ { * } \big \| \leq \sqrt { 2 } \big \| { \mathbf { w } } _ { \leq k } ^ { * } \big \| _  { \Sigma } _  \leq k \end{array}
$$

Head bias error. For the bias error in the �-dimensional head space, we have

head bias $= \left\| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \big ( \mathbf { I } - \hat { \boldsymbol { \Sigma } } _ { \leq k } \psi ( \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \big ) \mathbf { w } _ { \leq k } ^ { * } \right\|$

$$
\begin{array} { r l r } & { = \| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \big ( \mathbf { I } - g ( \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \big ) \boldsymbol { \mathsf { w } } _ { \leq k } ^ { * } \| } & { g ( z ) = z \psi ( z ) } \\ & { \leq \| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \big ( \mathbf { I } - g ( \boldsymbol { \Sigma } _ { \leq k } ) \big ) \boldsymbol { \mathsf { w } } _ { \leq k } ^ { * } \| + \| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \big ( g ( \boldsymbol { \Sigma } _ { \leq k } ) - g ( \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \big ) \boldsymbol { \mathsf { w } } _ { \leq k } ^ { * } \| } \\ & { \leq \| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \big ( \mathbf { I } - g ( \boldsymbol { \Sigma } _ { \leq k } ) \big ) \boldsymbol { \mathsf { w } } _ { \leq k } ^ { * } \| + \| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \big ( g ( \boldsymbol { \Sigma } _ { \leq k } ) - g ( \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \big ) \hat { \boldsymbol { \Sigma } } _ { \leq k } ^ { 1 / 2 } \| \cdot \| \hat { \boldsymbol { \Sigma } } _ { \leq k } ^ { - 1 / 2 } \boldsymbol { \mathsf { w } } _ { \leq k } ^ { * } \| } \\ &  \leq \| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \big ( \mathbf { I } - g ( \boldsymbol { \Sigma } _ { \leq k } ) \big ) \boldsymbol { \mathsf { w } } _ { \leq k } ^ { * } \| + \| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \big ( g ( \boldsymbol { \Sigma } _ { \leq k } ) - g ( \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \big ) \hat { \boldsymbol { \Sigma } } _ { \leq k } ^ { 1 / 2 } \| \cdot \sqrt { 2 } \| \boldsymbol { \Sigma } _ { \leq k } ^ { - 1 / 2 } \ \end{array}
$$

in which the concentration error can be bounded via the Schur norm:

$$
\begin{array} { r l r l } & { \left\| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \big ( g ( \boldsymbol { \Sigma } _ { \leq k } ) - g ( \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \big ) \hat { \boldsymbol { \Sigma } } _ { \leq k } ^ { 1 / 2 } \right\| } \\ & { = \left\| \boldsymbol { \Sigma } _ { \leq k } ^ { 1 / 2 } \Big ( g ^ { [ 1 ] } ( \boldsymbol { \Sigma } _ { \leq k } , \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \circ ( \boldsymbol { \Sigma } _ { \leq k } - \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \Big ) \hat { \boldsymbol { \Sigma } } _ { \leq k } ^ { 1 / 2 } \right\| } & & { \mathrm { S c h u r m u l t i p l i e r } } \\ & { = \left\| \lambda h ( \boldsymbol { \Sigma } _ { \leq k } , \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \circ \Big ( \boldsymbol { \Sigma } _ { \leq k } ^ { - 1 / 2 } ( \boldsymbol { \Sigma } _ { \leq k } - \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \hat { \boldsymbol { \Sigma } } _ { \leq k } ^ { - 1 / 2 } \Big ) \right\| } & & { \mathrm { C o r o l l a r y ~ E . 1 ~ a n d ~ d e f i n i t i o n ~ o f } \ : h } \\ & { \leq \lambda \| h ( H \cup \{ 0 \} , H \cup \{ 0 \} ) \| \in \cdot \left\| \boldsymbol { \Sigma } _ { \leq k } ^ { - 1 / 2 } ( \boldsymbol { \Sigma } _ { \leq k } - \hat { \boldsymbol { \Sigma } } _ { \leq k } ) \hat { \boldsymbol { \Sigma } } _ { \leq k } ^ { - 1 / 2 } \right\| } & & { \mathrm { S c h u r n o r m } } \\ & { \leq \lambda \| h ( H \cup \{ 0 \} , H \cup \{ 0 \} ) \| \in \cdot c _ { 1 } \sqrt { \frac { k + \log ( 1 / \delta ) } { n } } . } & & { \mathrm { L e m m a ~ A . 1 } } \end{array}
$$

Since $h ( 0 , \cdot ) = h ( \cdot , 0 ) = 0 .$ , it holds that

$$
\begin{array} { r } { \| h ( H \cup \{ 0 \} , H \cup \{ 0 \} ) \| _ { \mathfrak E } = \| h ( H , H ) \| _ { \mathfrak E } \leq \| r ( H , H ) \| _ { \mathfrak E } , } \end{array}
$$

therefore the head bias error is bounded as

$$
\mathrm { h e a d ~ b i a s \leq \left\| \left( \mathbf { I } - g ( \Sigma _ { \leq k } ) \right) \mathbf { w } _ { \leq k } ^ { * } \right\| _ { \Sigma _ { \leq k } } + \sqrt { 2 } c _ { 1 } \| h ( H , H ) \| _ { \mathfrak E } \cdot \lambda \sqrt { \frac { k + \log ( 1 / \delta ) } { n } } \| \mathbf { w } _ { \leq k } ^ { * } \| _ { \Sigma _ { \leq k } ^ { - 1 } } . }
$$

Spillover error. The error caused by head signal spillover to the tail subspace can be controlled by simple concentration:

$$
\begin{array} { r l r l } & { \mathrm { s p i l l o v e r : } = \left\| \mathbf { \Sigma } _ { 5 \times k } ^ { 1 / 2 } \mathbf { \bar { x } } _ { 5 \times k } ^ { \top } y ( \bar { \mathbf { A } } _ { \leq k } ) \mathbf { X } _ { \leq k } \mathbf { w } _ { \leq k } ^ { * } \right\| } & \\ & { } & { = \frac { 1 } { \sqrt { n } } \left\| \mathbf { \Sigma } _ { 5 \times k } ^ { 1 / 2 } \mathbf { X } _ { > k } ^ { \top } y ( \bar { \mathbf { A } } _ { \leq k } ) \bar { \mathbf { A } } _ { \leq k } \mathbf { u } _ { \leq k } ^ { * } \right\| } & & { \mathbf { u } ^ { * } : = \sqrt { n } \mathbf { X } _ { \leq k } ( \mathbf { X } _ { \leq k } ^ { \top } \mathbf { X } _ { \leq k } ) ^ { - 1 } \mathbf { w } _ { \leq k } ^ { * } } \\ & { } & { = \frac { 1 } { \sqrt { n } } \left\| \mathbf { \Sigma } _ { 5 \times k } ^ { 1 / 2 } \mathbf { X } _ { > k } ^ { \top } y ( \bar { \mathbf { A } } _ { \leq k } ) \mathbf { u } _ { \leq k } ^ { * } \right\| } & & { g ( z ) : = z \psi ( z ) } \\ & { } & { \leq c _ { 1 } \sqrt { \bar { \alpha } _ { k } } \left\| g ( \bar { \mathbf { A } } _ { \leq k } ) \mathbf { u } _ { \leq k } ^ { * } \right\| } & & { \mathrm { L e m m a ~ } \mathbf { A } . 3 } \\ & { } & { \leq c _ { 1 } \sqrt { \bar { \alpha } _ { k } } \left\| g ( \bar { \mathbf { A } } _ { \leq k } ) \right\| \cdot \| \mathbf { u } _ { \leq k } ^ { * } \| } & & { } \\ & { } &  \leq c _ { 1 } \sqrt { 2 \bar { \alpha } _ { k } } \left( \| r ( I , H ) \| _ { \Xi } + \| r ( I , \{ 0 \} ) \| _ { \Xi } \right) \cdot \| \mathbf { w } _ { \leq k } ^ { * } \| _  \ \end{array}
$$

For the last inequality, we have also used that for any $x \in I ,$

$$
| g ( y ) | \leq | g ( x ) | + | r ( x , y ) | | g ^ { * } ( y ) - g ^ { * } ( x ) | \leq | r ( x , 0 ) | + | r ( x , y ) | .
$$

Tail error. The tail error is bounded by

$$
\begin{array} { r l } & { \mathrm { B i a s } _ { 2 } : = \left\| \boldsymbol { \Sigma } ^ { 1 / 2 } \bigg ( \mathbf { I } - \frac { 1 } { n } \mathbf { X } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } \bigg ) \left[ \mathbf { \Sigma } _ { \mathbf { w } _ { > k } ^ { * } } ^ { 0 } \right] \right\| } \\ & { \qquad \leq \left\| \boldsymbol { \Sigma } ^ { 1 / 2 } \frac { 1 } { n } \mathbf { X } ^ { \top } \boldsymbol { \psi } ( \bar { \mathbf { A } } ) \mathbf { X } \left[ \mathbf { \Sigma } _ { \mathbf { w } _ { > k } ^ { * } } ^ { 0 } \right] \right\| + \left\| \boldsymbol { \Sigma } ^ { 1 / 2 } \left[ \mathbf { \Sigma } _ { \mathbf { w } _ { > k } ^ { * } } ^ { 0 } \right] \right\| } \end{array}
$$

$$
\begin{array} { r l r l } & { = \left\| \Sigma ^ { 1 / 2 } \frac { 1 } { n } \mathbf { X } ^ { \top } \psi ( \bar { \mathbf { A } } ) \mathbf { X } _ { > k } \mathbf { w } _ { > k } ^ { * } \right\| + \left\| \Sigma _ { > k } ^ { 1 / 2 } \mathbf { w } _ { > k } ^ { * } \right\| } & & { } \\ & { \leq 9 \left\| ( \bar { \mathbf { A } } + \lambda \mathbf { I } ) \psi ( \bar { \mathbf { A } } ) \frac { 1 } { \sqrt { n } } \mathbf { X } _ { > k } \mathbf { w } _ { > k } ^ { * } \right\| + \left\| \Sigma _ { > k } ^ { 1 / 2 } \mathbf { w } _ { > k } ^ { * } \right\| } & & { \mathrm { L e m m a ~ E . 5 } } \\ & { = 9 \Bigg \| r ( \bar { \mathbf { A } } , 0 ) \frac { 1 } { \sqrt { n } } \mathbf { X } _ { > k } \mathbf { w } _ { > k } ^ { * } \Bigg \| + \left\| \Sigma _ { > k } ^ { 1 / 2 } \mathbf { w } _ { > k } ^ { * } \right\| } & & { r ( x , 0 ) = g ( x ) / g ^ { * } ( x ) = ( x + \lambda ) \psi ( x ) } \\ & { \leq 9 \| r ( I , \{ 0 \} ) \| \mathfrak { g } \cdot c _ { 1 } \| \Sigma _ { > k } ^ { 1 / 2 } \mathbf { w } _ { > k } ^ { * } \| + \left\| \Sigma _ { > k } ^ { 1 / 2 } \mathbf { w } _ { > k } ^ { * } \right\| } & & { \mathrm { L e m m a ~ A . 1 } } \\ & { = \left( 1 + 9 c _ { 1 } \| r ( I , \{ 0 \} ) \| \mathfrak { g } \right) \| \mathfrak { w } _ { > k } ^ { * } \| _ { \Sigma _ { > k } } . } \end{array}
$$

Variance error. Since $r ( x , 0 ) = g ( x ) / g ^ { * } ( x )$ , it holds that $| g | \leq \| r ( I , \{ 0 \} ) \| _ { \mathfrak { S } } \cdot | g ^ { * } |$ , so the first part follows from the monotonicity of trace. On the other hand, by Lemma E.5, we have

$$
\left\| { \frac { 1 } { n } } { ( { \bar { \mathbf { A } } } + \lambda \mathbf { I } ) ^ { - 1 } } \mathbf { X } \Sigma \mathbf { X } ^ { \top } { ( { \bar { \mathbf { A } } } + \lambda \mathbf { I } ) ^ { - 1 } } \right\| \leq 8 1 ,
$$

so that

$$
\begin{array} { l } { \displaystyle \mathrm { V a r i a n c e } ( \hat { \mathbf { w } } _ { g } ) = \frac { \sigma ^ { 2 } } { n } \operatorname { t r } \left( \frac { 1 } { n } \mathbf { X } \Sigma \mathbf { X } ^ { \top } ( \bar { \mathbf { A } } + \lambda \mathbf { I } ) ^ { - 2 } r ( \bar { \mathbf { A } } , 0 ) ^ { 2 } \right) } \\ { \displaystyle \quad \leq \frac { 8 1 \sigma ^ { 2 } } { n } \operatorname { t r } \left( r ( \bar { \mathbf { A } } , 0 ) ^ { 2 } \right) } \\ { \displaystyle \quad \leq \frac { 8 1 \sigma ^ { 2 } } { n } \left\| r ( \bar { \mathbf { A } } , 0 ) \right\| ^ { 2 } \mathrm { r a n k } ( r ( \bar { \mathbf { A } } , 0 ) ) } \\ { \displaystyle \quad \leq \frac { 8 1 \sigma ^ { 2 } } { n } \| r ( I , \{ 0 \} ) \| _ { \mathfrak { S } } ^ { 2 } \mathrm { r a n k } ( g ( \bar { \mathbf { A } } ) ) . } \end{array}
$$

This completes the proof.

## E.4 Proof of Corollary 5.4

We require the following smoothness bound for the Schur norm.

Proposition E.6. Let $I \subseteq \mathbb { R } _ { \geq 0 }$ be an interval and $\beta \in ( 0 , 1 ]$ . There exists a universal constant $c > 0$ such that for any function $f \in C ^ { 1 , \beta } ( I )$ ,

$$
\| f ^ { [ 1 ] } \| _ { \mathfrak { S } ( I , I ) } \leq \frac { c } { \beta } \| f \| _ { C ^ { 1 , \beta } ( I ) } .
$$

Proof of Proposition $E . 6 .$ In fact, the following more general statement is true: for any function $f \in$ $B _ { \infty , 1 } ^ { 1 } ( I ) \cap C ^ { 1 } ( I )$ , where $B _ { \infty , 1 } ^ { 1 } ( I )$ denotes the Besov space restricted to � (Triebel, 1983),

$$
\| f ^ { [ 1 ] } \| _ { \mathfrak { S } ( I , I ) } \leq c \| f \| _ { B _ { \infty , 1 } ^ { 1 } ( I ) } .
$$

Indeed, this is a consequence of Theorems 1.6.1, 3.1.10, and $3 . 3 . 6$ of Aleksandrov and Peller (2016). The claim then follows from the continuous embedding $C ^ { 1 , \beta } ( \mathbb { R } ) \subset B _ { \infty , 1 } ^ { 1 } ( \mathbb { R } )$ and the estimates

$$
\begin{array} { r l } & { \omega _ { 2 } ( f , h ) _ { \infty } = \displaystyle \operatorname* { s u p } _ { x , 0 \leq u \leq h } \left. \int _ { 0 } ^ { u } f ^ { \prime } ( x + u + \nu ) - f ^ { \prime } ( x + \nu ) \mathrm { d } \nu \right. \leq \| f \| _ { C ^ { 1 , \beta } } h ^ { 1 + \beta } , } \\ & { \quad \| f \| _ { B _ { \infty , 1 } ^ { 1 } } \lesssim \| f \| _ { \infty } + \displaystyle \int _ { 0 } ^ { 1 } \frac { \omega _ { 2 } ( f , h ) _ { \infty } } { h ^ { 2 } } \mathrm { d } h \leq \| f \| _ { \infty } + \frac { \| f \| _ { C ^ { 1 , \beta } } } { \beta } , } \end{array}
$$

see Triebel (1983, Theorem 2.5.12).

ProofofCorollary 5.4. We apply Theorem 5.3 with $k = k ^ { * }$ and $I = H = \mathbb { R } _ { > 0 }$ . Denote

$$
\begin{array} { r } { q _ { j } ( z ) : = z ^ { j } ( 1 - g ( \lambda z ) ) , \quad j = 0 , 1 , 2 . } \end{array}
$$

Defining diagonal values by continuous extension, the relative kernel � can be shown after some algebra to satisfy

$$
r ( \lambda x , \lambda y ) = - q _ { 0 } ^ { [ 1 ] } ( x , y ) - 2 q _ { 1 } ^ { [ 1 ] } ( x , y ) - q _ { 2 } ^ { [ 1 ] } ( x , y ) + q _ { 0 } ( x ) + q _ { 0 } ( y ) + q _ { 1 } ( x ) + q _ { 1 } ( y ) .
$$

Then by Proposition E.6 and the definitions of $C _ { \beta } , C _ { \psi }$

$$
\| r ( I , I ) \| _ { \mathfrak E } \lesssim \operatorname* { m a x } _ { j \in \{ 0 , 1 , 2 \} } \left( \| q _ { j } ^ { [ 1 ] } \| _ { \mathfrak E } + \| q _ { j } \| _ { \infty } \right) \lesssim C _ { \beta } , \quad \| r ( I , \{ 0 \} ) \| _ { \mathfrak E } \leq C _ { \psi } .
$$

Also, the condition (6) holds with probability at least $1 - \exp ( - n / c _ { 0 } )$ due to Lemma A.1. Finally, the $\lambda _ { k + 1 } ^ { 2 }$ term can be absorbed by assuming log $( 1 / \delta ) \leq n / c _ { 0 }$ without loss of generality and noting that

$$
\alpha _ { k } + \frac { \lambda _ { k + 1 } ^ { 2 } \log ( 1 / \delta ) } { n } \leq \alpha _ { k } + \frac { \log ( 1 / \delta ) } { c _ { \gamma } ^ { 2 } n } \left( \lambda + \frac { \sum _ { i > k } \lambda _ { i } } { n } \right) ^ { 2 } \leq c \left( \alpha _ { k } + \frac { \lambda ^ { 2 } \log ( 1 / \delta ) } { n } \right) .
$$

This completes the proof.

## F General Risk Lower Bounds

In this section, we prove the following bias lower bound for monotone shrinkage filters.

Theorem F.1. Under Assumption 1 with Gaussian design $\mathbf { x } \sim { \mathcal { N } } ( 0 , \pmb { \Sigma } )$ , there exist constants $c _ { 0 } , \ldots , c _ { 3 } > 1$ such that thefollowing holds. Let $g : \mathbb { R } _ { \geq 0 }  [ 0 , 1 ]$ be a monotone shrinkage filter, let sample size $n \geq c _ { 0 }$ and let � be any index satisfying $k \leq n / c _ { 3 }$ . For $i \leq k$ and $j \neq i ,$ , denote

$$
\Lambda _ { i } : = \lambda _ { i } + \lambda _ { k + 1 } + \frac { \sum _ { j > k } \lambda _ { j } } { n } , \quad \Gamma _ { i j } : = ( g ( 0 . 9 \lambda _ { i } ) - g ( 1 . 1 \lambda _ { j } ) ) _ { + } .
$$

Then

$$
c _ { 1 } \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) \geq \sum _ { i \leq k } \frac { \lambda _ { i } } { \Lambda _ { i } ^ { 2 } } \left( \frac { 1 } { n } \sum _ { j \neq i } \lambda _ { j } ^ { 2 } \Gamma _ { i j } ^ { 2 } \right) { \bf w } _ { i } ^ { * 2 } + \big \| ( { \bf I } - g ( 1 . 1 \Sigma ) ) { \bf w } ^ { * } \big \| _ { \Sigma } ^ { 2 } .
$$

$f g$ is concave, the second term may be replaced $b y \left\| ( \mathbf { I } - g ( \pmb { \Sigma } ) ) \mathbf { w } ^ { * } \right\| _ { \pmb { \Sigma } } ^ { 2 } .$

Moreover, we have the conditional lower bounds:

$$
c _ { 2 } \lambda _ { k + 1 } \leq \frac { \sum _ { i > k } \lambda _ { i } } { n } \quad \implies \quad c _ { 1 } \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) \geq \| \mathbf { w } ^ { * } \| _ { \Sigma _ { k : \infty } } ^ { 2 } ,
$$

$$
c _ { 2 } \frac { \sum _ { i > k } \lambda _ { i } ^ { 2 } } { n } \leq \left( \frac { \sum _ { i > k } \lambda _ { i } } { n } \right) ^ { 2 } \quad \Longrightarrow \quad c _ { 1 } \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) \geq \sum _ { i \leq k } \lambda _ { i } \operatorname* { m i n } \left\{ 1 , \frac { \sum _ { j > k } \lambda _ { j } } { n \lambda _ { i } } \right\} ^ { 2 } \mathbf { w } _ { i } ^ { * 2 } .
$$

## F.1 Proof of Theorem F.1

Fix $i \leq k$ and $j \neq i$ and define the leave-one-out (LOO) projection

$$
\mathbf { P } : = \mathbf { 1 } \{ \mathbf { A } _ { - j } > 0 . 9 n \lambda _ { i } \mathbf { I } \} .
$$

Also recall the notation $\Psi : = n ^ { - 1 } \psi ( \mathbf { A } / n )$ . We first state some intermediate lemmas.

Lemma F.2. For every $i \leq k$ with $\lambda _ { i } > 0$ and $j \neq i ,$ conditional on $\{ \mathbf { z } _ { \ell } : \ell \neq j \}$ , it holds for every $\mathbf { u } \in \mathbb { R } ^ { n }$ that

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { z } _ { j } } \left( \mathbf { z } _ { j } ^ { \top } \boldsymbol { \Psi } \mathbf { u } \right) ^ { 2 } \geq c \Gamma _ { i j } ^ { 2 } \big \| \mathbf { A } _ { - j } ^ { - 1 } \mathbf { P } \mathbf { u } \big \| ^ { 2 } . } \end{array}
$$

Proof of Lemma F.2. If $\lambda _ { j } = 0$ , then

$$
\mathbb { E } _ { { \mathbf { z } } _ { j } } \left( { \mathbf { z } } _ { j } ^ { \top } \Psi { \mathbf { u } } \right) ^ { 2 } = \frac { 1 } { n ^ { 2 } } \| \psi ( { \mathbf { A } } / n ) { \mathbf { u } } \| ^ { 2 } \geq g ( 0 . 9 \lambda _ { i } ) ^ { 2 } \big \| { \mathbf { A } } _ { - j } ^ { - 1 } { \mathbf { P } } { \mathbf { u } } \big \| ^ { 2 } .
$$

Assume $\lambda _ { j } > 0$ and let $\mathbf { U } : = 2 \mathbf { P } - \mathbf { I } .$ . Note that

$$
\mathbf { U A } _ { - j } \mathbf { U } = \mathbf { A } _ { - j } , \quad \mathbf { U P } = \mathbf { P } , \quad \mathbf { U ( I - P ) } = - ( \mathbf { I } - \mathbf { P } ) .
$$

Then the cross-correlation term

$$
\left( \mathbf { z } _ { j } ^ { \top } \boldsymbol { \Psi } \mathbf { P } \mathbf { u } \right) \left( \mathbf { z } _ { j } ^ { \top } \boldsymbol { \Psi } ( \mathbf { I } - \mathbf { P } ) \mathbf { u } \right)
$$

is distributionally symmetric under the transformation $\mathbf { z } _ { j } \mapsto \mathbf { U } \mathbf { z } _ { j }$ , thus has mean zero. Hence

$$
\mathbb { E } _ { \mathbf { z } _ { j } } \left( \mathbf { z } _ { j } ^ { \top } \boldsymbol { \Psi } \mathbf { u } \right) ^ { 2 } \geq \mathbb { E } _ { \mathbf { z } _ { j } } \left( \mathbf { z } _ { j } ^ { \top } \boldsymbol { \Psi } \mathbf { P } \mathbf { u } \right) ^ { 2 } ,
$$

and it sufices to prove the claim when $\mathbf { u } = \mathbf { P } \mathbf { u }$

Choose an orthonormal eigenbasis such that

$$
\frac { \mathbf { A } _ { - j } } { n } = \sum _ { \ell = 1 } ^ { n } \gamma _ { \ell } \mathbf { w } _ { \ell } \mathbf { w } _ { \ell } ^ { \top } , \quad \mathbf { u } = \sum _ { \ell : \gamma _ { \ell } > 0 . 9 \lambda _ { i } } u _ { \ell } \mathbf { w } _ { \ell } .
$$

Applying Lemma D.2 with $\lambda = 1 . 1 \lambda _ { j }$ , on the event $\| \mathbf { z } _ { j } \| ^ { 2 } \leq 1 . 0 5 n$ , for every $\gamma _ { \ell } > 0 . 9 \lambda _ { i }$ <sub>�</sub> we have

$$
h _ { g , \ell } ( \mathbf { z } _ { j } ) \geq \frac { \Gamma _ { i j } } { \gamma _ { \ell } } \left( 1 - \frac { \| \mathbf { z } _ { j } \| ^ { 2 } } { 1 . 1 n } \right) \geq \frac { c \Gamma _ { i j } } { \gamma _ { \ell } } .
$$

Since $h _ { g , \ell }$ is invariant under sign change of the coordinates of $\mathbf { z } _ { j }$ in the basis $\left( \mathbf { w } _ { \ell } \right)$ , averaging over signs gives

$$
\begin{array} { l } { { \displaystyle \mathbb { E } _ { { \bf z } _ { j } } \left( { \bf z } _ { j } ^ { \top } \Psi { \bf u } \right) ^ { 2 } \geq \frac { 1 } { n ^ { 2 } } \sum _ { \ell : \gamma _ { \ell } > 0 . 9 \lambda _ { i } } u _ { \ell } ^ { 2 } \mathbb { E } _ { { \bf z } _ { j } } \left[ ( { \bf z } _ { j } ^ { \top } { \bf w } _ { \ell } ) ^ { 2 } h _ { g , \ell } ( { \bf z } _ { j } ) ^ { 2 } { \bf 1 } \{ \| { \bf z } _ { j } \| ^ { 2 } \leq 1 . 0 5 n \} \right] } } \\ { ~ \geq \frac { c } { n ^ { 2 } } \Gamma _ { i j } ^ { 2 } \displaystyle \sum _ { \ell : \gamma _ { \ell } > 0 . 9 \lambda _ { i } } \frac { u _ { \ell } ^ { 2 } } { \gamma _ { \ell } ^ { 2 } } } \\ { ~ = c \Gamma _ { i j } ^ { 2 } \Big \| \mathbf { A } _ { - j } ^ { - 1 } { \bf P } { \bf u } \Big \| ^ { 2 } , } \end{array}
$$

where we have used that

$$
\mathbb { E } _ { \mathbf { z } _ { j } } \left[ ( \mathbf { z } _ { j } ^ { \top } \mathbf { w } _ { \ell } ) ^ { 2 } \mathbf { 1 } \{ \| \mathbf { z } _ { j } \| ^ { 2 } \leq 1 . 0 5 n \} \right]
$$

is bounded below by a constant for suficiently large �. This proves the claim.

Lemma F.3. For every $i \leq k$ with $\lambda _ { i } > 0$ and $j \neq i ,$

$$
\mathbb { E } \big \| \mathbf { A } _ { - j } ^ { - 1 } \mathbf { P z } _ { i } \big \| ^ { 2 } \geq \frac { c } { n \Lambda _ { i } ^ { 2 } } .
$$

Proof. Decompose

$$
\mathbf { A } _ { - j } = \underbrace { \sum _ { \ell < i , \ell \neq j } \lambda _ { \ell } \mathbf { z } _ { \ell } \mathbf { z } _ { \ell } ^ { \top } } _ { ( i ) } + \underbrace { \sum _ { i \leq \ell \leq k , \ell \neq j } \lambda _ { \ell } \mathbf { z } _ { \ell } \mathbf { z } _ { \ell } ^ { \top } } _ { ( i i ) } + \underbrace { \sum _ { \ell > k , \ell \neq j } \lambda _ { \ell } \mathbf { z } _ { \ell } \mathbf { z } _ { \ell } ^ { \top } } _ { ( i i i ) } .
$$

By standard concentration and Lemma A.1, with probability at least $1 - \exp ( - n / c _ { 0 } )$

$$
\begin{array} { l } { \displaystyle \| ( i i ) \| \leq \lambda _ { i } \Big \| [ \mathbf { z } _ { \ell } ] _ { i \leq \ell \leq k } \Big \| ^ { 2 } \leq c n \lambda _ { i } , } \\ { \displaystyle \| ( i i i ) \| \leq \| \mathbf { A } _ { > k } \| \leq c n \left( \lambda _ { k + 1 } + \frac { 1 } { n } \sum _ { \ell > k } \lambda _ { \ell } \right) , } \end{array}
$$

so that $\| ( i i ) \| + \| ( i i i ) \| \leq c n \Lambda _ { i }$

Denote the projection to span $\{ \mathbf { z } _ { \ell } : \ell < i , \ell \neq j \}$ by $\mathbf { I I } _ { i } .$ . Since rank $\Pi _ { i } \le k \le n / c _ { 3 }$ , it holds with probability at least $1 - \exp ( - n / c _ { 0 } )$ that $\Vert \mathbf { I I } _ { i } \mathbf { z } _ { i } \Vert ^ { 2 } \leq 0 . 0$ 1� and 0.99� $\leq \| \mathbf { z } _ { i } \| ^ { 2 } \leq 2 n$ . Also define the projection

$$
{ \mathbf { Q } } : = { \mathbf { 1 } } \{ { \mathbf { A } } _ { - j } > c _ { 6 } n \Lambda _ { i } { \mathbf { I } } \} .
$$

Taking $c _ { 6 } > 1$ , it holds that $\mathbf { Q } \preceq \mathbf { P } .$ Since the range of (�) is contained in ran $\mathbf { I I } _ { i }$

$$
\begin{array} { r l } & { ( \mathbf { I } - \mathbf { I I } _ { i } ) \mathbf { Q } = ( \mathbf { I } - \mathbf { I I } _ { i } ) \mathbf { A } _ { - j } \left( \mathbf { A } _ { - j } \left| _ { \mathrm { r a n } } \mathbf { Q } \right. \right) ^ { - 1 } \mathbf { Q } } \\ & { \qquad = ( \mathbf { I } - \mathbf { I I } _ { i } ) \left( ( i i ) + ( i i i ) \right) \left( \mathbf { A } _ { - j } \left| _ { \mathrm { r a n } } \mathbf { Q } \right. \right) ^ { - 1 } \mathbf { Q } , } \end{array}
$$

so we can ensure, for suficiently large $c _ { 6 } ,$

$$
\| \mathbf { Q } ( \mathbf { I } - \mathbf { I I } _ { i } ) \| \leq \frac { \| ( i i ) \| + \| ( i i i ) \| } { c _ { 6 } n \Lambda _ { i } } \leq \frac { 1 } { 2 0 } .
$$

It follows that

$$
\| \mathbf { Q } \mathbf { z } _ { i } \| \leq \| \mathbf { H } _ { i } \mathbf { z } _ { i } \| + \| \mathbf { Q } ( \mathbf { I } - \mathbf { I I } _ { i } ) \| \| \mathbf { z } _ { i } \| \leq \frac { \sqrt { n } } { 5 } .
$$

Also, since $\mathbf { A } _ { - j } \preceq 0 . 9 n \lambda _ { i } \mathbf { I }$ on the range of I − P, we have the series of inequalities

$$
\frac { \lambda _ { i } } { n } \| ( \mathbf { I } - \mathbf { P } ) \mathbf { z } _ { i } \| ^ { 4 } \leq \frac { 1 } { n } \mathbf { z } _ { i } ^ { \top } ( \mathbf { I } - \mathbf { P } ) \mathbf { A } _ { - j } ( \mathbf { I } - \mathbf { P } ) \mathbf { z } _ { i } \leq 0 . 9 \lambda _ { i } \| ( \mathbf { I } - \mathbf { P } ) \mathbf { z } _ { i } \| ^ { 2 } .
$$

It follows that $\| ( \mathbf { I } - \mathbf { P } ) \mathbf { z } _ { i } \| ^ { 2 } \leq 0$ .9� and hence

$$
\left\| ( \mathbf { P } - \mathbf { Q } ) \mathbf { z } _ { i } \right\| ^ { 2 } = \left\| \mathbf { z } _ { i } \right\| ^ { 2 } - \left\| \mathbf { Q } \mathbf { z } _ { i } \right\| ^ { 2 } - \left\| ( \mathbf { I } - \mathbf { P } ) \mathbf { z } _ { i } \right\| ^ { 2 } \geq 0 . 0 5 n .
$$

Therefore, using that P, Q commute with A<sub>−�</sub> , $\mathbf { A } _ { - j }$

$$
\begin{array} { l } { \left\| { \bf A } _ { - j } ^ { - 1 } { \bf P } { \bf z } _ { i } \right\| ^ { 2 } \geq \left\| { \bf A } _ { - j } ^ { - 1 } ( { \bf P } - { \bf Q } ) { \bf z } _ { i } \right\| ^ { 2 } } \\ { \geq ( c _ { 6 } n \Lambda _ { i } ) ^ { - 2 } \| ( { \bf P } - { \bf Q } ) { \bf z } _ { i } \| ^ { 2 } \geq \displaystyle \frac { c } { n \Lambda _ { i } ^ { 2 } } . } \end{array}
$$

Taking expectations proves the claim.

We also show a lower bound for the efective bias.

Lemma F.4. For any monotone shrinkagefilter $g \in { \mathcal { G } }$ , we have

$$
\mathbb { E } \mathrm { E f f e c t i v e B i a s } ( \hat { \mathbf { w } } _ { g } ) \geq \left( 1 - \frac { 1 } { \tau } \right) ^ { 2 } \left. ( \mathbf { I } - g ( \tau \Sigma ) ) \mathbf { w } ^ { * } \right. _ { \Sigma } ^ { 2 } , \quad \tau \geq 1 .
$$

Proof. For $x \ge 0 , y > 0$ , it holds that

$$
1 - g ( x ) \geq \left( 1 - { \frac { x } { y } } \right) ( 1 - g ( y ) ) .
$$

Plugging in $x = \hat { \Sigma } _ { i i }$ and $y = \tau \lambda _ { i }$ and applying Jensen’s inequality,

$$
\mathbb { E } ( 1 - \mathbf { G } _ { i i } ) ^ { 2 } \geq \left( \mathbb { E } ( 1 - \mathbf { G } _ { i i } ) \right) ^ { 2 } \geq \left( 1 - \frac { \mathbb { E } \hat { \Sigma } _ { i i } } { \tau \lambda _ { i } } \right) ^ { 2 } ( 1 - g ( \tau \lambda _ { i } ) ) ^ { 2 } = \left( 1 - \frac { 1 } { \tau } \right) ^ { 2 } ( 1 - g ( \tau \lambda _ { i } ) ) ^ { 2 } .
$$

Multiplying by $\lambda _ { i } \mathbf { w } _ { i } ^ { * 2 }$ and summing over � gives the statement.

Proof of Theorem $F . l .$ . Combining the preceding lemmas, we obtain

$$
\mathbb { E } ( \mathbf { z } _ { j } ^ { \top } \Psi \mathbf { z } _ { i } ) ^ { 2 } \geq c \Gamma _ { i j } ^ { 2 } \mathbb { E } \big \| \mathbf { A } _ { - j } ^ { - 1 } \mathbf { P } \mathbf { z } _ { i } \big \| ^ { 2 } \geq \frac { c \Gamma _ { i j } ^ { 2 } } { n \Lambda _ { i } ^ { 2 } } , \quad i \leq k , \ j \neq i .
$$

It follows that

$$
\begin{array} { r } { \mathbb { E } \mathrm { B i a s } ( \hat { \mathbf { w } } _ { g } ) \geq \displaystyle \sum _ { i \leq k } \lambda _ { i } \mathbf { w } _ { i } ^ { * 2 } \sum _ { j \neq i } \lambda _ { j } ^ { 2 } \mathbb { E } ( \mathbf { z } _ { j } ^ { \top } \Psi \mathbf { z } _ { i } ) ^ { 2 } } \\ { \geq c \displaystyle \sum _ { i \leq k } \frac { \lambda _ { i } } { \Lambda _ { i } ^ { 2 } } \left( \frac { 1 } { n } \sum _ { j \neq i } \lambda _ { j } ^ { 2 } \Gamma _ { i j } ^ { 2 } \right) \mathbf { w } _ { i } ^ { * 2 } . } \end{array}
$$

The residual lower bound follows from Lemma F.4 with $\tau = 1 . 1$ . If $g$ is concave, Jensen’s inequality instead directly gives

$$
\mathbf { e } _ { i } ^ { \top } ( \mathbf { I } - \mathbf { G } ) \mathbf { e } _ { i } \geq 1 - g ( \mathbf { e } _ { i } ^ { \top } \hat { \Sigma } \mathbf { e } _ { i } ) = 1 - g \left( \frac { \lambda _ { i } \Vert \mathbf { z } _ { i } \Vert ^ { 2 } } { n } \right)
$$

and hence

$$
\mathbb { E } \mathbf { B } _ { i i } \geq \lambda _ { i } \mathbb { E } \left( 1 - g \left( \frac { \lambda _ { i } \| \mathbf { z } _ { i } \| ^ { 2 } } { n } \right) \right) ^ { 2 } \geq \lambda _ { i } ( 1 - g ( \lambda _ { i } ) ) ^ { 2 } .
$$

For the first conditional lower bound, from Lemma $\mathrm { A . 1 }$ , with probability at least $1 - \exp ( - n / c _ { 0 } )$

$$
\lambda _ { \operatorname* { m i n } } ( \mathbf { A } _ { > k } ) \geq \frac { 2 } { 3 } \sum _ { i > k } \lambda _ { i } - c _ { 1 } n \lambda _ { k + 1 } \geq \frac { 1 } { 2 } \sum _ { i > k } \lambda _ { i }
$$

by choosing $c _ { 2 }$ suficiently large. Since $g$ is a shrinkage filter, it follows that

$$
\Psi \preceq { \bf A } ^ { - 1 } \preceq { \bf A } _ { > k } ^ { - 1 } \preceq \frac { 2 } { \sum _ { i > k } \lambda _ { i } } { \bf I } .
$$

Therefore for $i > k .$ , on $\| \mathbf { z } _ { i } \| ^ { 2 } \leq 2 n$

$$
{ \bf G } _ { i i } = \lambda _ { i } { \bf z } _ { i } ^ { \top } \Psi { \bf z } _ { i } \leq \frac { 4 n \lambda _ { i } } { \sum _ { j > k } \lambda _ { j } } \leq \frac { 4 } { c _ { 2 } } \leq \frac { 1 } { 2 } \Longrightarrow \bf { B } _ { i i } \geq \frac { \lambda _ { i } } { 4 } ,
$$

which proves the claim. The final bound follows from Lemma D.5.

## F.2 Proof of Corollary 5.5

ProofofCorollary 5.5. We derive the head and tail lower bounds separately. The residual term follows directly from Theorem F.1.

The head term. We apply Theorem F.1 with $k = k ^ { * }$ . By minimality of $k ^ { * }$ ,

$$
\frac { \sum _ { j > k ^ { * } } \lambda _ { j } } { n } \leq \frac { \sum _ { j > k ^ { * } - 1 } \lambda _ { j } } { n } \leq c _ { 2 } \lambda _ { k ^ { * } } .
$$

Hence for every $i \leq k ^ { * }$ , we have $\Lambda _ { i } \leq ( c _ { 2 } + 2 ) \lambda _ { i }$ and

$$
\alpha _ { k ^ { * } } \leq \left( \frac { \sum _ { j > k ^ { * } } \lambda _ { j } } { n } \right) ^ { 2 } + \lambda _ { k ^ { * } + 1 } \left( \frac { \sum _ { j > k ^ { * } } \lambda _ { j } } { n } \right) \leq c \lambda _ { i } ^ { 2 } .
$$

If

$$
\frac { \sum _ { j > k ^ { * } } \lambda _ { j } ^ { 2 } } { n } \leq \frac { 1 } { c _ { 5 } } \left( \frac { \sum _ { j > k ^ { * } } \lambda _ { j } } { n } \right) ^ { 2 }
$$

for suficiently large $c _ { 5 } .$ , we immediately obtain from Theorem F.1,

$$
c _ { 1 } \mathbb { E } \mathbf { B i a s } ( \hat { \mathbf { w } } _ { g } ) \geq \sum _ { i \leq k ^ { * } } \lambda _ { i } \left( \frac { \sum _ { j > k ^ { * } } \lambda _ { j } } { n \lambda _ { i } } \right) ^ { 2 } \mathbf { w } _ { i } ^ { * 2 } \geq c \alpha _ { k ^ { * } } \Vert \mathbf { w } ^ { * } \Vert _ { \Sigma _ { 0 : k ^ { * } } ^ { - 1 } } ^ { 2 } .
$$

Otherwise,

$$
\lambda _ { k ^ { * } + 1 } \left( \frac { \sum _ { j > k ^ { * } } \lambda _ { j } } { n } \right) \ge \frac { \sum _ { j > k ^ { * } } \lambda _ { j } ^ { 2 } } { n } > \frac { 1 } { c _ { 5 } } \left( \frac { \sum _ { j > k ^ { * } } \lambda _ { j } } { n } \right) ^ { 2 }
$$

implies that

$$
\frac { \sum _ { j > k ^ { * } } \lambda _ { j } } { n } < c _ { 5 } \lambda _ { k ^ { * } + 1 } ,
$$

and so choosing $c _ { 2 }$ suficiently large, we have

$$
\lambda \geq c _ { 2 } \lambda _ { k ^ { * } + 1 } - \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \geq 1 . 1 \lambda _ { k ^ { * } + 1 } .
$$

Now fix $i \leq k ^ { * }$ and consider the coeficient of $\mathbf { w } _ { i } ^ { * 2 }$ in Theorem F.1. By the definition of $C _ { g } .$ , it holds that either $1 - g ( 1 . 1 \lambda _ { i } ) \ge C _ { g } \mathrm { o r } g ( 0 . 9 \lambda _ { i } ) - g ( \lambda ) \ge C _ { g }$ . In the first case, the residual term gives

$$
\lambda _ { i } \left( 1 - g ( 1 . 1 \lambda _ { i } ) \right) ^ { 2 } \ge C _ { g } ^ { 2 } \lambda _ { i } \ge c C _ { g } ^ { 2 } \frac { \alpha _ { k ^ { * } } } { \lambda _ { i } } .
$$

In the second case, for every $j > k ^ { * }$

$$
\Gamma _ { i j } = ( g ( 0 . 9 \lambda _ { i } ) - g ( 1 . 1 \lambda _ { j } ) ) _ { + } \geq g ( 0 . 9 \lambda _ { i } ) - g ( \lambda ) \geq C _ { g } ,
$$

which further implies

$$
\frac { \lambda _ { i } } { \Lambda _ { i } ^ { 2 } } \left( \frac { 1 } { n } \sum _ { j > k ^ { * } } \lambda _ { j } ^ { 2 } \Gamma _ { i j } ^ { 2 } \right) \geq c C _ { g } ^ { 2 } \frac { \sum _ { j > k ^ { * } } \lambda _ { j } ^ { 2 } } { n \lambda _ { i } } \geq c C _ { g } ^ { 2 } \frac { \alpha _ { k ^ { * } } } { \lambda _ { i } } .
$$

In both cases, the bias is lower bounded by $C _ { g } ^ { 2 } \alpha _ { k ^ { * } } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { 0 : k } ^ { - 1 } } ^ { 2 }$

The tail term. Note that

$$
\lambda _ { k ^ { * } + 1 } \leq \frac { 1 } { c _ { 2 } } \left( \lambda + \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right) \leq \operatorname* { m a x } \left\{ \frac { 2 \lambda } { c _ { 2 } } , \frac { 2 } { c _ { 2 } } \frac { \sum _ { i > k ^ { * } } \lambda _ { i } } { n } \right\} .
$$

If $\lambda _ { k ^ { * } + 1 } \leq 2 \lambda / c _ { 2 }$ , then for all $i > k ^ { * } , 1 - g ( 1 . 1 \lambda _ { i } ) \geq 1 - g ( \lambda )$ by taking $c _ { 2 } > 2 . 2$ . Then

$$
\begin{array} { r } { \left\| ( \mathbf { I } - g ( 1 . 1 \Sigma ) ) \mathbf { w } ^ { * } \right\| _ { \Sigma } ^ { 2 } \geq ( 1 - g ( \lambda ) ) ^ { 2 } \| \mathbf { w } ^ { * } \| _ { \Sigma _ { k ^ { * } ; \infty } } ^ { 2 } . } \end{array}
$$

Otherwise, Theorem F.1 immediately gives the tail term after increasing $c _ { 2 } .$ . The proof is complete. □

## F.3 Proof of Corollary 5.6

Proof of Corollary 5.6. Set the reference filter as $g ^ { * } ( z ) = z / ( z + \lambda / p )$ . From Bernoulli’s inequality,

$$
g ^ { \ast } ( z ) \leq g _ { \lambda , p } ( z ) \leq \operatorname* { m i n } \{ 1 , p z / \lambda \} \leq 2 g ^ { \ast } ( z ) .
$$

The variance is then equal to $\sigma ^ { 2 } D / n$ up to a constant factor (Wu et al., 2026, Proposition 2.1). The min thresholding can be removed as in the proof of Corollary 3.1. We proceed to bound the bias.

The upper bound. Substituting $u = \lambda / ( x + \lambda ) , \nu = \lambda / ( y + \lambda ) \in ( 0 , 1 ]$ , we may write

$$
r ( x , y ) = r _ { p } ( u , \nu ) : = \frac { ( p - ( p - 1 ) u ) ( p - ( p - 1 ) \nu ) } { p } \frac { u ^ { p } - \nu ^ { p } } { u - \nu } , \quad u \neq \nu .
$$

For integer $p ,$ this is equal to the GD kernel in Example $^ { 2 , }$ evaluated at $( 1 - u , 1 - \nu )$ with $t = p ,$ , multiplied by the row and column factors

$$
\frac { p - ( p - 1 ) u } { p + 1 - p u } , \frac { p - ( p - 1 ) \nu } { p + 1 - p \nu } \leq 1 .
$$

Hence $\| r _ { p } \| _ { \mathfrak { S } } = O ( 1 )$ for integer $p$ (all Schur norms in $u , \nu$ are taken over $[ 0 , 1 ] ^ { 2 } )$ .

For noninteger $p ,$ , write $p = m q$ , where $m = \lfloor p \rfloor$ and $1 < q < 2$ . It is straightforward to check that

$$
r _ { p } ( u , \nu ) = \underbrace { \frac { p - ( p - 1 ) u } { q ( m - ( m - 1 ) u ^ { q } ) } } _ { \leq 1 } \underbrace { \frac { p - ( p - 1 ) \nu } { q ( m - ( m - 1 ) \nu ^ { q } ) } } _ { \leq 1 } \frac { q ( u ^ { q } - \nu ^ { q } ) } { u - \nu } r _ { m } ( u ^ { q } , \nu ^ { q } ) .
$$

Moreover, the integral representation

$$
\begin{array} { l } { \displaystyle \frac { u ^ { q } - \nu ^ { q } } { u - \nu } = \frac { \sin ( \pi ( q - 1 ) ) } { \pi ( u - \nu ) } \int _ { 0 } ^ { \infty } t ^ { q - 2 } \left( \frac { u ^ { 2 } } { u + t } - \frac { \nu ^ { 2 } } { \nu + t } \right) \mathrm { d } t } \\ { \displaystyle = \frac { \sin ( \pi ( q - 1 ) ) } { \pi } \int _ { 0 } ^ { \infty } t ^ { q - 2 } \left( \frac { u } { u + t } + \frac { t } { u + t } \frac { \nu } { \nu + t } \right) \mathrm { d } t } \end{array}
$$

and Proposition E.2 imply

$$
\| ( z \mapsto z ^ { q } ) ^ { [ 1 ] } \| _ { \infty } \leq \frac { 2 \sin ( \pi ( q - 1 ) ) } { \pi } \int _ { 0 } ^ { \infty } \frac { t ^ { q - 2 } } { 1 + t } \mathrm { d } t = 2 .
$$

Thus

$$
\| r _ { p } \| _ { \mathfrak { S } } \le q \| r _ { m } \| _ { \mathfrak { S } } \| ( z \mapsto z ^ { q } ) ^ { [ 1 ] } \| _ { \mathfrak { S } } \le 4 \| r _ { m } \| _ { \mathfrak { S } } = O ( 1 ) ,
$$

and $\| r ( \mathbb { R } _ { \ge 0 } , \mathbb { R } _ { \ge 0 } ) \| _ { \mathfrak { S } } = O ( 1 )$ uniformly in $p .$ . The bias upper bound now follows by applying Theorem 5.3 with $\lambda / p$ in place of � and $I = H = \mathbb { R } _ { \geq 0 } , \delta = \exp ( - k ^ { \ast } / c _ { 0 } )$ . The extra bias terms can be absorbed into the variance since $\lambda / p \le \tilde { \lambda } , c _ { 2 } \lambda _ { k + 1 } \le \tilde { \lambda }$ and

$$
\left( \frac { \sum _ { i > k ^ { * } } \lambda _ { i } ^ { 2 } } { n } + \frac { k ^ { * } \lambda _ { k ^ { * } + 1 } ^ { 2 } } { n } + \frac { k ^ { * } } { n } \left( \frac { \lambda } { p } \right) ^ { 2 } \right) \left. \mathbf { w } ^ { * } \right. _ { \Sigma _ { 0 ; k } ^ { - 1 } } ^ { 2 } \leq c \tilde { \lambda } ^ { 2 } \Vert \mathbf { w } ^ { * } \Vert _ { \Sigma _ { 0 ; k } ^ { - 1 } } ^ { 2 } \frac { D } { n } \leq c \Vert \mathbf { w } ^ { * } \Vert _ { \Sigma _ { 0 ; k } } ^ { 2 } \frac { D } { n } \leq c b \sigma ^ { 2 } \frac { D } { n } .
$$

The lower bound. We apply Corollary $5 . 5$ with $\lambda / p$ in place of �. Note that the filter $g _ { \lambda , p }$ is concave and $1 - g _ { \lambda , p } ( \lambda / p ) = ( 1 + 1 / p ) ^ { - p } \ge e ^ { - 1 }$ . Also, for $z \le 3 \lambda / p$

$$
1 - g _ { \lambda , p } ( 1 . 1 z ) \ge \left( 1 + \frac { 3 . 3 } { p } \right) ^ { - p } \ge e ^ { - 3 . 3 } ,
$$

whereas for $z \ge 3 \lambda / p$

$$
g _ { \lambda , p } ( 0 . 9 z ) - g _ { \lambda , p } ( \lambda / p ) \geq \left( 1 + \frac { 1 } { p } \right) ^ { - p } - \left( 1 + \frac { 2 . 7 } { p } \right) ^ { - p } \geq e ^ { - 1 } - \frac { 1 } { 3 . 7 } > 0 .
$$

Thus $C _ { g }$ is bounded below uniformly in $p .$ The claimed bound follows.