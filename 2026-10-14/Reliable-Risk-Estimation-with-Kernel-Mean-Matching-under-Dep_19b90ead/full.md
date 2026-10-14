# Reliable Risk Estimation with Kernel Mean Matching under Dependent Sampling

Alexey Zaytsev<sup>∗</sup>

Diana Koldasbayeva<sup>∗</sup> Applied AI Institute

Egor Serov<sup>∗,†</sup>

## Abstract

Kernel Mean Matching (KMM) estimates risk under covariate shift by reweighting labelled source samples so that their kernel mean embedding matches that of unlabeled target samples. Existing finite-sample guarantees are usually stated for independent samples, although many applications involve temporal, spatial, or clustered dependence. We give a finite-sample KMM analysis under dependent sampling by replacing nominal sample sizes with efective sample sizes in the concentration steps used by the classical proof. The approach uses Bernstein inequalities for geometrically strongly mixing processes and derives a model-based efective sample size. The resulting bound separates overlap, dependence, and smoothness efects with specific constants derived under a Gaussian process assumption. This analysis yields an additional tradeof: smoother losses preserve more input dependence and therefore reduce efective sample size, while also reducing function complexity and empiricalto-population risk gaps. Applying the result to prediction yields confidence intervals that achieve near-nominal coverage in synthetic experiments and substantially reduce undercoverage on real monitoring data.

## 1 INTRODUCTION

Machine learning models often face distribution shift: labelled training samples come from a source distribution, while predictions are required under a diferent target distribution. This mismatch complicates reliable evaluation, particularly in scientific domains where observations are collected along spatial transects, temporal monitoring campaigns, or clustered sampling schemes (Roberts et al., 2017). Specifically, under covariate shift, the conditional response law remains unchanged, while the marginal covariate distribution changes. In ecological and geospatial modelling, spatial transects and temporal monitoring campaigns produce structured observations, while uneven sampling may leave labelled source data systematically unrepresentative of the deployment region (Koldasbayeva and Zaytsev, 2025).

A standard approach to estimating target risk from labelled source samples and unlabelled target covariates is to reweight source losses by the ratio of target to source covariate densities (Shimodaira, 2000; Sugiyama et al., 2007). Estimating this ratio can be unstable when overlap is poor, source observations are sparse in high-ratio regions, or the ratio model is misspecified (Kouw and Loog, 2019). Kernel Mean Matching (KMM) provides an alternative to explicit densityratio estimation: it computes source weights that align the empirical source and target kernel mean embeddings in a reproducing kernel Hilbert space (RKHS) (Huang et al., 2006; Gretton et al., 2009; Serov et al., 2026).

Despite KMM’s practical success, its standard finitesample theory relies on independent observations (Yu and Szepesv´ari, 2012). This assumption is routinely violated in real-world settings characterized by temporal persistence, spatial autocorrelation, environmental gradients, or cluster sampling. For these dependent regimes, treating the data as independent leads to overconfidence, as the nominal sample size substantially overstates the true amount of statistical information.

This paper develops a dependent-sampling foundation for KMM risk-estimation guarantees. The guiding principle of our approach is that the standard i.i.d. proof structure remains valid if standard concentration inequalities are replaced by their dependent counterparts, substituting the nominal sample size n with an efective sample size $n _ { \mathrm { e f f } }$ . Although $n _ { \mathrm { e f f } }$ is a long-standing cornerstone for diagnosing correlation in Markov Chain Monte Carlo (Kass et al., 1998; Vehtari et al., 2019), bounding weight degradation in importance sampling (Martino et al., 2017), and estimating spatial degrees of freedom (Bretherton et al., 1999), our work formally integrates this concept into non-parametric RKHS generalization bounds (Gretton et al., 2012) for domain adaptation. This allows us to derive risk-estimation bounds under covari ate shift and non-i.i.d. sampling that decompose the distinct roles of distribution overlap, data dependence, and function smoothness. Furthermore, our theoretical analysis reveals an important underlying tradeof: smoother target or loss functions have lower statistical complexity, but when evaluated along a correlated covariate sequence, they preserve more dependence, which inherently reduces the efective sample size.

Specifically, our main contributions are as follows:

1. We formulate an efective-concentration condition, based on a Bernstein inequality for strongly β-mixing sequences, under which the classical finite-sample KMM guarantee naturally extends to dependent data.

2. We explicitly decompose the generalization bottleneck into three components: independent noise, density-ratio overlap, and dependent samplingsequence persistence.

3. We characterize how covariate dependence and response smoothness determine efective sample size, deriving a closed-form GP-induced $n _ { \mathrm { e f f } }$ that yields a practical dependence correction for KMM risk estimation.

4. We validate the GP-induced efective-size mechanism in controlled synthetic experiments and eval uate the resulting confidence correction on three real-world monitoring datasets, where it consistently reduces empirical undercoverage relative to the i.i.d. interval.

## 2 RELATED WORK

Kernel Mean Matching (KMM) was originally introduced as a non-parametric approach to correct for sample selection bias and covariate shift (Huang et al., 2006; Gretton et al., 2009). More broadly, learning and evaluation under covariate shift are typically addressed via importance-weighted likelihoods, crossvalidation schemes, and adapted risk estimators (Shimodaira, 2000; Sugiyama et al., 2007; Kouw and Loog, 2019). The foundational finite-sample analysis by Yu and Szepesv´ari (2012) demonstrates that KMM efectively bounds the target expectation error, provided the regression function lies within the RKHS and the source and target samples are independent. Our theoretical analysis adopts a similar proof structure but generalises it by replacing standard independent concentration steps with efective concentration inequalities for dependent data.

Dependence is ubiquitous in environmental and geospatial data, where nearby observations in space or time often carry substantially less independent information than their nominal sample size suggests. Ignoring this structure can lead to overly optimistic evaluation and confidence estimates, motivating dependenceaware validation and transfer procedures (Roberts et al., 2017; Koldasbayeva and Zaytsev, 2025). Recent work has also investigated KMM directly under spatial distribution shift, showing empirically that dependence and source–target mismatch can jointly affect transfer-risk estimation (Serov et al., 2026). Our contribution complements this empirical line of work by providing a finite-sample analysis of how dependent sampling modifies the KMM risk-estimation guarantee and by making the resulting loss of information explicit through an efective sample size.

The dependent-data component of our framework builds upon concentration inequalities for strongly mixing processes (Rio, 1993, 1999; Bosq, 2012), specif ically leveraging a Bernstein bound for geometrically β-mixing sequences (Merlev\`ede et al., 2009). We translate these bounds into an efective sample size (n<sub>ef</sub>) perspective, creating a bridge to other fields where data dependence is a central challenge. For instance, in Markov Chain Monte Carlo (MCMC), $n _ { \mathrm { e f f } }$ is the canonical diagnostic for chain convergence (Kass et al., 1998; Vehtari et al., 2019). While the MCMC literature has explored multivariate ESS for finite-dimensional outputs (Vats et al., 2019), our work extends this concept by defining efective sample sizes for infinite-dimensional RKHS embeddings. Furthermore, KMM shares conceptual roots with Importance Sampling (IS), where weight degradation due to distribution discrepancy is actively quantified via ESS (Martino et al., 2017); in our KMM bound, the domain overlap penalty B formally captures this identical discrepancy bottleneck.

Finally, Gaussian-process models provide a convenient probabilistic framework for relating dependence in the inputs to dependence in a smooth response function (Rasmussen, 2003; Srinivas et al., 2009). We use this connection to obtain an explicit, model-based efective sample size for KMM risk estimation. Rather than treating $n _ { \mathrm { e f f } }$ as an abstract concentration parameter, the $\mathrm { G P }$ construction expresses it in terms of the correlation structure of the covariates and the smoothness scale of the response. This provides a practical link between the dependent-data concentration analysis and the confidence correction evaluated in our experiments.

Table 1: Positioning relative to related theory.
<table><tr><td>Reference</td><td>Setting</td><td>Relation to Ours</td></tr><tr><td>(Huang et al., 2006;</td><td>Covariate shift KMM</td><td>Introduces KMM objective</td></tr><tr><td>Gretton et al., 2009)</td><td></td><td></td></tr><tr><td> $( \mathrm { Y u }$  and Šzepesvári,</td><td>Finite-sample bound</td><td>i.i.d. proof template</td></tr><tr><td>2012)</td><td></td><td></td></tr><tr><td>(Roberts et al., 2017; Koldas-</td><td>Spatio-temporal CV</td><td>Dependence</td></tr><tr><td>bayeva and</td><td></td><td>motivation</td></tr><tr><td>Zaytsev, 2025) (Merlevède</td><td></td><td></td></tr><tr><td>et al., 2009)</td><td>β-mixing</td><td>Scalar</td></tr><tr><td></td><td>processes</td><td>concentration tools</td></tr><tr><td>This paper</td><td>Dependent KMM</td><td>Explicit bounds via neff</td></tr></table>

Table 1 summarizes our positioning relative to this diverse theoretical landscape.

## 3 THEORY

## 3.1 Problem Setup

Consider a labelled source sequence $\{ ( \mathbf { X } _ { i } ^ { \mathrm { t r } } , Y _ { i } ^ { \mathrm { t r } } ) \} _ { i = 1 } ^ { n _ { \mathrm { t r } } }$ with joint distribution $P _ { \mathrm { t r } }$ over $\mathcal { X } \times \mathcal { V }$ , having covariate marginal $P _ { \mathrm { t r } } ^ { \mathcal { X } }$ , and an unlabeled target sequence $\{ \mathbf { X } _ { j } ^ { \mathrm { t e } } \} _ { j = 1 } ^ { n _ { \mathrm { t e } } }$ with marginal $P _ { \mathrm { t e } }$ over $\mathcal { X } \subset \mathbb { R } ^ { d }$ . The covariate space X is compact and measurable, and responses are normalized to $Y \in [ 0 , 1 ]$

Assumption 1 (Covariate Shift). The conditional response law is invariant across domains: $P _ { \mathrm { t r } } ( d y \mid \mathbf { x } ) =$ $P _ { \mathrm { t e } } ( d y \mid \mathbf { x } )$

Under Assumption 1, we define the invariant regression function $m ( \mathbf { x } ) : = \mathbb { E } [ Y \mid \mathbf { X } = \mathbf { x } ]$ . Our objective is to estimate the target expectation $\mu _ { \mathrm { t e } } : = \mathbb { E } _ { P _ { \mathrm { t e } } } [ Y ] =$ $\begin{array} { r } { \int _ { \mathcal { X } } m ( \mathbf { x } ) P _ { \mathrm { t e } } ( d \mathbf { x } ) } \end{array}$

Assumption 2 (Conditionally Independent Residuals). Conditional on all observed covariates $( { \bf X } ^ { \mathrm { t r } } , { \bf X } ^ { \mathrm { t e } } )$ , the residual errors $\epsilon _ { i } = Y _ { i } ^ { \mathrm { t r } } - m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } )$ are mutually independent, mean-zero random variables.

Remark 1 (Scope of Residual Independence). Assumption 2 isolates the structural challenge of covariate shift from idiosyncratic label noise. It allows us to prove Theorem 1 at the nominal parametric rate. While this is a standard theoretical abstraction in non-parametric regression under dependent designs (Bosq, 2012; Yu and Szepesv´ari, 2012), real-world spatial datasets may contain unobserved confounders causing the residuals to exhibit their own autocorrelation.

Assumption 3 (Bounded Density Ratio and Compact Support). The covariate space $ { \mathcal { X } } \subset \mathbb { R } ^ { d }$ is compact and measurable. The derivative $\omega ^ { * } ( \mathbf { x } ) \ = \quad$ $( d P _ { \mathrm { t e } } / d P _ { \mathrm { t r } } ^ { \mathcal { X } } ) ( { \bf x } )$ exists and satisfies $0 \leq \omega ^ { * } ( \mathbf { x } ) \leq B <$ ∞ for all $\mathbf { x } \in \mathcal { X }$

Assumption 4 (Moderate-Deviation Sample Size). For a given failure probability $\delta \in ( 0 , 1 )$ , the source sample size $n _ { \mathrm { t r } }$ is suficiently large relative to the sequence mixing rate to satisfy the moderate-deviation condition:

$$
\frac { n _ { \mathrm { t r } } } { ( \log n _ { \mathrm { t r } } ) ^ { 4 } } \geq \frac { B ^ { 2 } K _ { 2 } ^ { 2 } } { 2 K _ { 1 } \sigma _ { v , \infty } ^ { 2 } } \log \frac { 6 } { \delta } ,\tag{1}
$$

where $K _ { 1 } , K _ { 2 } > 0$ and $\sigma _ { v , \infty } ^ { 2 }$ are defined in Appendix B.

Kernel Mean Matching (KMM). Let H be an RKHS equipped with a feature map $\Phi \ :$ $\mathcal { X } ~  ~ \mathcal { H }$ , and assume the features are uniformly bounded: $\begin{array} { r } { \operatorname* { s u p } _ { \mathbf { x } \in \mathcal { X } } \| \Phi ( \mathbf { x } ) \| _ { \mathcal { H } } ~ \leq ~ C . } \end{array}$ Rather than explicitly estimating $\omega ^ { * } ( \mathbf { x } )$ , KMM (Huang et al., 2006; Yu and Szepesv´ari, 2012) computes weights $\hat { { \bf w } } ~ = ~ ( \hat { w } _ { 1 } , \dots , \hat { w } _ { n _ { \mathrm { t r } } } ) ^ { \top }$ aligning the empirical RKHS means $\begin{array} { r l r } { \hat { \mu } _ { \Phi } ^ { \mathrm { t r } } ( \mathbf { w } ) } & { : = } & { \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } w _ { i } \Phi ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) } \end{array}$ and $\begin{array} { r l } { \hat { \mu } _ { \Phi } ^ { \mathrm { t e } } } & { { } : = } \end{array}$ $\begin{array} { r } { \frac { 1 } { n _ { \mathrm { t e } } } \sum _ { j = 1 } ^ { n _ { \mathrm { t e } } } \Phi ( { \bf X } _ { j } ^ { \mathrm { t e } } ) } \end{array}$

$$
\begin{array} { r } { \hat { \mathbf { w } } = \underset { \mathbf { w } \in [ 0 , B ] ^ { n _ { \mathrm { t r } } } } { \arg \operatorname* { m i n } } \left\| \hat { \mu } _ { \Phi } ^ { \mathrm { t r } } ( \mathbf { w } ) - \hat { \mu } _ { \Phi } ^ { \mathrm { t e } } \right\| _ { \mathcal { H } } . } \end{array}\tag{2}
$$

The reweighted target mean estimator is then given by:

$$
\begin{array} { r } { \hat { \mu } _ { \mathrm { K M M } } = \frac { 1 } { n _ { \mathrm { t r } } } \displaystyle \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \hat { w } _ { i } Y _ { i } ^ { \mathrm { t r } } . } \end{array}\tag{3}
$$

For Y being a loss function we obtain a risk estimate.

## 3.2 Main Dependent KMM Bound

The classical KMM guarantee (Yu and Szepesv´ari, 2012) relies on Hoefding bounds for independent samples. To adapt this to dependent sampling, we dis till the Bernstein inequality for geometrically β-mixing processes (Merlev\`ede et al., 2009) into a deviationindependent efective sample size $n _ { \mathrm { e f f } } ^ { \mathrm { t r } }$ (detailed in Appendix B). To control the empirical RKHS featurematching error, we apply Berbee’s coupling (Berbee, 1979) and Pinelis’ inequality (Pinelis, 1994) to derive a Hilbert-space concentration bound governed by a blocked efective sample size $n _ { \mathrm { e f f } }$ <sub>,RKHS</sub> (derived in $\mathrm { A p - }$ pendix C). As a result, we decompose the KMM target estimation error into three distinct operational components.

Theorem 1 (Non-Asymptotic Dependent KMM Bound). Suppose Assumptions 1–3 hold, and Assumption 4 is satisfied for a given failure probability $\delta \ \in$ $( 0 , 1 )$ . Assume the source and target covariate $s e \mathrm { - }$ quences are strictly stationary and geometrically $\beta -$ mixing. Let the invariant regression function satisfy $m \in { \mathcal { H } }$ , and let $v ( { \bf x } ) = \omega ^ { * } ( { \bf x } ) m ( { \bf x } )$ have a long-run variance proxy $\sigma _ { v , \infty } ^ { 2 }$

Then, with probability at least $1 - \delta \colon$

$$
\begin{array} { r } { | \widehat { \mu } _ { \mathrm { K M M } } - \mu _ { \mathrm { t e } } | \leq \underbrace { B \sqrt { \frac { \log ( 6 / \delta ) } { 2 n _ { \mathrm { t r } } } } } _ { \varepsilon _ { \mathrm { n o i s e } } } + \underbrace { B \sqrt { \frac { \log ( 6 / \delta ) } { 2 n _ { \mathrm { t e f t } } ^ { \mathrm { t r } } } } } _ { \varepsilon _ { \mathrm { s h i f t } } } } \\ { + \underbrace { \mathcal { O } \left( \| m \| _ { \mathcal { H } } \left( \frac { B } { \sqrt { n _ { \mathrm { e f f , R K H S } } ^ { \mathrm { t r } } } } + \frac { 1 } { \sqrt { n _ { \mathrm { e f f , R K H S } } ^ { \mathrm { t e } } } } \right) \right) } _ { \varepsilon _ { \mathrm { a n d r o } } } , } \end{array}\tag{4}
$$

where $n _ { \mathrm { e f f } } ^ { \mathrm { t r } }$ is the scalar efective sample size evaluated $f o r \ v ( \mathbf { X } )$ derived in Appendix B. The quantities $n _ { \mathrm { e f f , R K H S } } ^ { \mathrm { t r } }$ and $n _ { \mathrm { e f f , R K H S } } ^ { \mathrm { t e } }$ scale as $\mathcal { O } ( n / \log n )$ and $d e \mathrm { - }$ note the RKHS efective sample sizes arising from the geometric blocking penalty. The exact non-asymptotic constants for $\mathcal { E } _ { \mathrm { M M D } }$ are detailed in Appendix D. If the sample size is too small to satisfy Assumption $^ { 4 , }$ the bound remains valid by substituting the deviationdependent efective sample size $n _ { \mathrm { e f f } } ^ { \mathrm { t r } } ( t )$ shown in Appendix B.

Interpretation and The Bottleneck of Dependence. Theorem 1 isolates the three fundamental challenges of risk estimation under covariate shift. This decomposition reveals how these error terms interact under strong data dependence $( n _ { \mathrm { e f f } } ^ { \mathrm { t r } } \ll n _ { \mathrm { t r } } ) ;$

1. Independent Noise $( \mathcal { E } _ { \mathrm { n o i s e } } )$ : From Assumption 2, the conditionally independent idiosyncratic label noise concentrates at nominal rate $\mathcal { O } ( 1 / \sqrt { n _ { \mathrm { t r } } } )$ Under strong spatial or temporal dependence, this noise averages out rapidly and becomes statistically negligible compared to the other terms.

2. Dependent Sampling Bottleneck $( \mathcal { E } _ { \mathrm { s h i f t } } )$ The error of evaluating the true density ratio over correlated covariates is penalized. Because the sampling process repeatedly observes similar regions of the covariate space, this term scales with the deflated efective sample size $\mathcal { O } ( 1 / \sqrt { n _ { \mathrm { e f f } } ^ { \mathrm { t r } } } )$ When $n _ { \mathrm { e f f } } ^ { \mathrm { t r } } ~ \ll ~ n _ { \mathrm { t r } }$ $\mathcal { E } _ { \mathrm { s h i f t } }$ completely dominates $\mathcal { E } _ { \mathrm { n o i s e } } .$ making sequence persistence the primary bottleneck of scalar generalization.

3. RKHS Matching $\left( \mathcal { E } _ { \mathrm { M M D } } \right)$ and Overlap Penalty (B): The feature-matching error decays at the blocked rate governed by the exact RKHS efective sample sizes. While this worstcase blocking penalty appears mathematically less severe than the scalar long-run variance deflation, both scalar terms $( \mathcal { E } _ { \mathrm { n o i s e } }$ and $\mathcal { E } _ { \mathrm { s h i f t } } )$ are multiplicatively scaled by B. Poor source-target overlap (manifested as large $B )$ quadratically inflates the required efective sample size.

Application to Risk Estimation While Theorem 1 is formulated for a general bounded scalar response $Y \in [ 0 , 1 ]$ , its primary operational application is target risk evaluation. Specifically, let $g : \mathcal { X }  \mathcal { Y } _ { \mathrm { p r e d } }$ be a fixed hypothesis, and let $\ell : \mathcal { Y } _ { \mathrm { p r e d } } \times \mathcal { Y }  [ \bar { 0 } , 1 ]$ be a bounded loss function. By setting the response to the sample loss $Y _ { i } ^ { \mathrm { t r } } = \ell ( g ( \dot { \mathbf { X } _ { i } ^ { \mathrm { t r } } } ) , Y _ { i } ^ { \mathrm { t r , l a b e l } } )$ , the true target risk is given by $R _ { \mathrm { t e } } ( g ) : = \mathbb { E } _ { P _ { \mathrm { t e } } } [ \ell ( g ( \mathbf { X } ) , Y _ { \mathrm { l a b e l } } ) ]$ and its reweighted KMM estimate is $\hat { R } _ { \mathrm { K M M } } ( g ) \ : =$ $\begin{array} { r } { \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \hat { w } _ { i } Y _ { i } ^ { \mathrm { t r } } } \end{array}$ Under Assumption 1, the invariant regression function represents the conditional risk surface $m _ { \ell } ( \mathbf { x } ) : = \mathbb { E } [ \ell ( g ( \mathbf { X } ) , Y _ { \mathrm { l a b e l } } ) \ | \ \mathbf { X } = \mathbf { x } ]$ . Assuming $m _ { \ell } ~ \in ~ \mathcal { H }$ Theorem 1 directly bounds the risk generalization error $\left| \hat { R } _ { \mathrm { K M M } } ( g ) - R _ { \mathrm { t e } } ( g ) \right|$ by substituting $\| m \| _ { \mathcal { H } }  \| m _ { \ell } \| _ { \mathcal { H } }$ . For uniform convergence guarantees over entire hypothesis classes via Block Rademacher complexity, we refer to Appendix F.

From Theory to Practice: A Generative Surrogate. Theorem 1 establishes that under spatial or temporal dependence, KMM generalisation is governed by sequence persistence through $n _ { \mathrm { e f f } } ^ { \mathrm { t r } }$ . However, as a worst-case non-asymptotic bound, it presents practical limitations: it depends on unobservable mixing constants $( K _ { 1 } , K _ { 2 } )$ , and its reliance on the RKHS norm $\| m \| _ { \mathcal { H } }$ obscures how the smoothness of the regression function governs efective sample size deflation.

To make these abstract mixing rates practically computable, we introduce a Gaussian Process (GP) covariance surrogate that yields a computable, closed-form efective sample size $n _ { \mathrm { e f f } } ^ { \mathrm { G P } } ( h )$

## 3.3 Separating Overlap, Dependence, and Smoothness via GP Modeling

Following core insights of Theorem 1, we want to analyze how the smoothness of the conditional loss surface mathematically dictates the efective sample size. By modeling the loss as a GP, we derive an closed-form efective sample size $n _ { \mathrm { e f f } } ^ { \mathrm { G P } } ( \dot { h } )$ that calibrates practical confidence intervals.

Gaussian Surrogate and Sequence Autocorrelation. In Theorem 1, sequence persistence depends on the composite function $v ( { \mathbf { X } } ) = \omega ^ { * } ( { \mathbf { X } } ) m _ { \ell } ( { \mathbf { X } } )$ . In practice, we decouple the global density ratio $\omega ^ { * }$ from the sequence autocorrelation of the loss surface. Specifically, the scale of $\omega ^ { * }$ is directly captured by the empirical squared weights $\hat { w } _ { i } ^ { 2 }$ in the numerator of the variance estimator. Meanwhile, the autocorrelation across lags is driven by the smoothness of the conditional loss $m _ { \ell } ( { \mathbf { x } } ) = \mathbb { E } [ \ell ( g ( { \mathbf { X } } ) , Y _ { \mathrm { l a b e l } } ) \mid { \mathbf { X } } = { \mathbf { x } } ]$

While $\| m \ell \| _ { \mathcal { H } }$ bounds the deterministic worst-case complexity in Theorem 1, we model the latent variation of this conditional risk surface via a zero-mean stationary GP surrogate $u \ \sim \ { \mathcal { G P } } ( 0 , k _ { h } )$ to extract a computable, data-driven length-scale h: $m _ { \ell } ( { \mathbf { x } } ) =$ $u ( { \bf x } )$ To evaluate sequence covariances analytically, we model local feature variations as stationary Gaussian $\mathbf X _ { t } \sim \mathcal N ( \pmb \mu , \pmb \sigma _ { x } ^ { 2 } \mathbf I _ { d } )$ with formal truncation to the compact domain X to preserve Assumption 3. As detailed in Appendix E) we evaluated it with spatial or temporal correlation $r _ { k } = \mathrm { C o r r } ( ( \mathbf { X } _ { s } ) _ { l } , ( \mathbf { X } _ { s + k } ) _ { l } )$

For the squared-exponential (SE) kernel $k _ { h } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) =$ $\sigma _ { u } ^ { 2 } \exp ( - \| \mathbf { \bar { x } } - \mathbf { x } ^ { \prime } \| ^ { 2 } / 2 h ^ { 2 } )$ , we define the ratio of input spread to function smoothness as $a _ { h } = 2 \sigma _ { x } ^ { 2 } / h ^ { 2 }$ Marginalizing over the covariate distribution yields the exact prior-averaged sequence autocorrelation in closed form:

$$
\bar { \rho } _ { u } ( k ) = \frac { ( 1 + a _ { h } \{ 1 - r _ { k } \} ) ^ { - d / 2 } - ( 1 + a _ { h } ) ^ { - d / 2 } } { 1 - { ( 1 + a _ { h } ) } ^ { - d / 2 } } .\tag{5}
$$

Full derivations, along with the corresponding expressions for rough Mat´ern kernels, are provided in $\mathrm { A p \mathrm { - } }$ pendix E.

The resulting GP-induced finite-sample efective sample size is thus:

$$
n _ { \mathrm { e f f } } ^ { \mathrm { G P } } ( h ) = \frac { n _ { \mathrm { t r } } } { 1 + 2 \sum _ { k = 1 } ^ { n _ { \mathrm { t r } } - 1 } \left( 1 - \frac { k } { n _ { \mathrm { t r } } } \right) \bar { \rho } _ { u } ( k ) } .\tag{6}
$$

The Smoothness-Dependence Tradeof. Equation (6) exposes a bandwidth tradeof that is absent in standard i.i.d. estimations. For smooth functions $( h \gg \sigma _ { x } ,$ , so that $a _ { h } \ll 1 )$ , Taylor expansion of (5) shows that the sequence autocorrelation inherits the persistence of the raw covariates: $\bar { \rho } _ { u } ( k ) = r _ { k } + \mathcal { O } ( a _ { h } )$ which severely deflates $n _ { \mathrm { e f f } } ^ { \mathrm { G P } }$ . Conversely, rough functions $\left( a _ { h } \gg 1 \right)$ decorrelate rapidly across lags.

Importantly, this does not imply that rough functions are statistically preferable. It highlights a tension within Theorem 1: increasing smoothness reduces the hypothesis complexity $\| m \ell \| _ { \mathcal { H } }$ , tightening the RKHS feature-matching error $\mathcal { E } _ { \mathrm { M M D } }$ . But inflates sequence persistence, penalizing the efective sample size and amplifying the dependent sampling error $( \mathcal { E } _ { \mathrm { s h i f t } } )$

Practical Confidence Calibration: The GP-Calibrated Variance Radius. By substituting the abstract theoretical scale $n _ { \mathrm { e f f } } ^ { \mathrm { t r } }$ in Theorem 1 with our computable GP-realization $n _ { \mathrm { e f f } } ^ { \mathrm { G P } } ( h _ { \ell } )$ , we obtain a theoretically grounded, practically evaluable measure of sampling variance. The loss length-scale $h _ { \ell }$ is directly estimated from the source data via empirical variogram fitting without requiring access to target labels.

Guided by this substitution, we construct an asymptotic Wald-type confidence interval. Because the target sample size $n _ { \mathrm { t e } }$ governs the feature-matching discrepancy $\mathcal { E } _ { \mathrm { M M D } ^ { - } }$ which acts as a lower-order bias under stable optimization—the dominant sampling variability is isolated entirely within the reweighted source labels. Consequently, the practical confidence radius evaluates the empirical loss variance scaled by the source persistence:

$$
\hat { r } _ { \mathrm { e f f } } = z _ { 1 - \alpha / 2 } \sqrt { \frac { \widehat { \mathrm { V a r } } \left[ \hat { w } _ { i } ( Y _ { i } ^ { \mathrm { t r } } - \hat { R } _ { \mathrm { K M M } } ( g ) ) \right] } { n _ { \mathrm { e f f } } ^ { \mathrm { G P } } ( h _ { \ell } ) } } ,\tag{7}
$$

where $z _ { 1 - \alpha / 2 } = \Phi ^ { - 1 } ( 1 - \alpha / 2 )$ denotes the standard normal quantile (z ≈ 1.96 for the nominal 95% level).

This interval treats the estimated weights wˆ as fixed post-optimization, a standard asymptotic approximation in importance weighting. The GP-based efective sample size adjusts the variance estimate for dependence propagated through the conditional loss surface. In settings with additional residual dependence beyond that represented by the covariates, this correction may partially absorb the resulting variance inflation, but it does not remove the conditional-independence requirement of Assumption 2 from the finite-sample guarantee. We evaluate the resulting practical interval empirically in Section 4.

## 4 EXPERIMENTS

We design our experiments to isolate and evaluate the three fundamental efects highlighted by our theory: covariate shift through distribution overlap, data dependence through efective sample size, and function smoothness through GP-induced loss autocorrelation. We validate our approach on controlled synthetic data and benchmark it against other corrections.

## 4.1 Artificial Data

Experimental Setup and Objectives We evaluate our dependent KMM framework on synthetic environments where ground-truth target expectations are analytically known. Covariates $\mathbf { X } _ { t }$ are generated as continuous dependent sequences with correlation length $\ell _ { x }$ under covariate shift $( \mu _ { \mathrm { t r } } \neq \mu _ { \mathrm { t e } } )$ . We evaluate both a matched Gaussian Process response surface $f _ { 0 } \sim \mathcal { G P } ( 0 , k _ { h } )$ and a deterministic misspecified harmonic surface. We benchmark our GP-calibrated confidence radius against unweighted estimation, classical i.i.d. KMM (Nominal n), raw-covariate correction (Input $n _ { \mathrm { e f f } } )$ , Block-Bootstrap, and theoretical Oracle baselines. Complete generative specifications, solver configurations, and extensive ablation grids are detailed in Appendices G.1–G.5.

Evaluation Metrics Since KMM-based variance estimators share identical weights, yielding identical point estimates, detailed in Appendix G.2, we focus exclusively on confidence interval quantification. We evaluate nominal 0.95 Coverage, considered valid if $\geq 0 . 9 2 5$ over $N = 3 0 0$ trials, interval Radius (halfwidth), and the Winkler Score (WS; (Winkler, 1972)), an additional scoring rule penalizing both excessive width and undercoverage.

Validating the Closed-Form Efective Sample Size We first verify whether the closed-form $n _ { \mathrm { e f f } } ^ { \mathrm { G P } } ( h )$ from (6) accurately predicts empirical sample-mean variance. As shown in Figure 1, the variance-based empirical $n _ { \mathrm { e f f } }$ tracks our analytical prediction along the identity line across matched GP realizations. Because the target loss typically has a diferent characteristic length-scale than the raw response Y, the formula automatically yields estimand-specific efective sample sizes without refitting KMM weights. Validation on misspecified deterministic surfaces is provided in Appendix G.3.

Table 2: Confidence Interval Calibration under Dependent Sampling. Coverage and interval radius for target-mean and target-risk estimation. Asterisks (<sup>∗</sup>) indicate significant empirical undercoverage below the nominal 0.95 level (< 0.925). Bold denotes the best valid practical estimator.
<table><tr><td rowspan="2">Method</td><td colspan="2">Target Mean (Y)</td><td colspan="2">Target Risk (L)</td></tr><tr><td>Coverage</td><td>Radius</td><td>Coverage</td><td>Radius</td></tr><tr><td>Unweighted</td><td>0.017*</td><td>0.0362</td><td>0.080*</td><td>0.0019</td></tr><tr><td>Nominal n</td><td>0.917*</td><td>0.0626</td><td>0.853*</td><td>0.0055</td></tr><tr><td>Block-Bootstrap</td><td>1.000</td><td>0.4538</td><td>0.967</td><td>0.0133</td></tr><tr><td>Input neff</td><td>1.000</td><td>0.1696</td><td>0.987</td><td>0.0150</td></tr><tr><td>GP  $n _ { \mathrm { e f f } }$  [Ours]</td><td>1.000</td><td>0.1380</td><td>0.970</td><td>0.0120</td></tr><tr><td>Strict Bound (Theorem 1)</td><td>1.000</td><td>5.2189</td><td>1.000</td><td>5.1220</td></tr><tr><td>Oracle IW</td><td>0.477*</td><td>0.0275</td><td>0.627*</td><td>0.0033</td></tr><tr><td>Oracle i.i.d.</td><td>0.903*</td><td>0.0257</td><td>0.917*</td><td>0.0024</td></tr></table>

The Smoothness-Dependence Trade-of Table 2 benchmarks confidence interval calibration. $\mathrm { A p \mathrm { - } }$ plying KMM with nominal i.i.d. variance leads to severe overconfidence, failing on both Target Y (0.917<sup>∗</sup>) and Target Risk (0.853<sup>∗</sup>). While Block-Bootstrap achieves valid coverage, it yields an excessively wide radius on Target Y (0.4538 vs. 0.1380) and requires expensive resampling.

GP  
![](images/1fd1f010158abc154a85222b4a7c386a5b42e5c44ad495780e6e2cf61ecdd472.jpg)  
Figure 1: Validation of the GP-induced efective sample size. The variance-based empirical $n _ { \mathrm { e f f } }$ aligns with our closed-form prediction (diagonal line) for both Target Y (point estimation) and Risk (loss evaluation) across matched GP realizations.

These results empirically confirm the theoretical tradeof established in Section 3.3. While naive Input $n _ { \mathrm { e f f } }$ safely restores coverage, it conservatively assumes worst-case sequence persistence $\left( \bar { \rho } _ { u } ( k ) \approx r _ { k } \right)$ , yielding unnecessarily wide intervals. In contrast, any response surface with finite length-scale decorrelates faster than raw input coordinates $\left( \bar { \rho } _ { u } ( k ) \ \leq \ r _ { k } \right)$ By analytically incorporating functional smoothness, our GP $n _ { \mathrm { e f f } }$ shrinks the confidence radius by 18.6% for Target $Y$ and 20.0% for Target Risk relative to Input $n _ { \mathrm { e f f } }$ , providing the sharpest confidence bounds that maintain valid coverage and minimal WS (Table 6).

Finally, while the non-asymptotic bound (Theorem 1) strictly guarantees coverage, its worst-case geometric penalty yields impractically wide intervals (Radius $>$ 5.1), highlighting the necessity of our GP surrogate for empirical practice.

Sample-Size Scaling of Error Bounds Theorem 1 dictates that valid KMM intervals must scale with $n _ { \mathrm { e f f } }$ rather than nominal n. Table 3 tracks risk interval calibration across $n \in \{ 4 8 , 6 4 , 1 2 8 , 2 5 6 \}$ (full sweep across $n \in \{ 3 2 , \ldots , 2 5 6 \}$ in Table 7).

Nominal n fails across all evaluated sample sizes reaching only $0 . 9 1 3 ^ { * }$ coverage even at $n \ = \ 2 5 6 .$ . It confirms that sample size alone cannot cure dependenceinduced underconfidence. Notably, in small-sample regimes such as $n = 4 8$ , Input $n _ { \mathrm { e f f } }$ also exhibits undercoverage (0.910<sup>∗</sup>), whereas GP $n _ { \mathrm { e f f } }$ reliably maintains nominal coverage (0.943). Across all sample sizes, GP $n _ { \mathrm { e f f } }$ consistently achieves $\approx 2 1 \%$ narrower intervals than Input $n _ { \mathrm { e f f } }$ , confirming the scalability of the surrogate across data regimes.

Table 3: Sample-Size Scaling of Target Risk Intervals. Nominal intervals systematically under-cover across all sample sizes. Asterisks (<sup>∗</sup>) indicate significant undercoverage below nominal 0.95 (< 0.925). Bold denotes the best valid practical estimator.
<table><tr><td>n</td><td>Method</td><td>Coverage</td><td>Radius</td><td>WS↓</td></tr><tr><td rowspan="3">48</td><td>Nominal n</td><td>0.773*</td><td>0.0048</td><td>0.0275</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.910*</td><td>0.0128</td><td>0.0290</td></tr><tr><td>GP  $\scriptstyle n _ { \mathrm { e f f } }$  [Ours]</td><td>0.943</td><td>0.0101</td><td>0.0260</td></tr><tr><td rowspan="3">64</td><td>Nominal n</td><td>0.773*</td><td>0.0046</td><td>0.0218</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.980</td><td>0.0124</td><td>0.0258</td></tr><tr><td>GP  $\scriptstyle n _ { \mathrm { e f f } }$  [Ours]</td><td>0.947</td><td>0.0098</td><td>0.0217</td></tr><tr><td rowspan="3">128</td><td>Nominal n</td><td>0.823*</td><td>0.0038</td><td>0.0155</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.983</td><td>0.0104</td><td>0.0220</td></tr><tr><td>GP  $\scriptstyle n _ { \mathrm { e f f } }$  [Ours]</td><td>0.970</td><td>0.0082</td><td>0.0181</td></tr><tr><td rowspan="3">256</td><td>Nominal n</td><td>0.913*</td><td>0.0031</td><td>0.0076</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>1.000</td><td>0.0085</td><td>0.0170</td></tr><tr><td>GP  $\scriptstyle n _ { \mathrm { e f f } }$  [Ours]</td><td>0.997</td><td>0.0067</td><td>0.0134</td></tr></table>

Robustness to Misspecification We evaluate the framework when modeling assumptions are violated. As detailed in Appendix G, the calibration ratio is robust across 2D grids varying dependence length $\ell _ { x }$ and loss smoothness h (Appendix G.4), maintaining valid coverage across sweeps over sample size $n , \ \ell _ { x } ,$ h, and clipping bound B (Appendix G.5). In addition, the method preserves valid coverage under nondiferentiable Mat´ern-1/2 paths (Appendix G.6) and non-stationary correlation structures (Appendix G.7).

## 4.2 Real Data

Controlled Overlap on Real Covariates Before evaluating real-world monitoring data, we isolate the role of source–target overlap using a semi-synthetic stress test on California Housing covariates (detailed with Figure 5 and Table 13 in Appendix H.1). By constructing source and target distributions via expo nential tilting, we demonstrate a fundamental overlap– stability trade-of: KMM successfully controls target mean error under moderate shift, but its bounded weights cannot fully reproduce the density ratio under severe overlap loss. Bounding weights limits variance amplification but leaves residual bias when adequate overlap requires more extreme weights. Appendix H.1 (Figure 6) further illustrates the bias–variance tradeof induced by the clipping bound B.

Three-Domain Real-Data Benchmark We next evaluate KMM in three real-data monitoring domains: NOAA weather stations Menne et al. (2012), $\mathrm { P M } _ { 2 . 5 }$ monitoring stations Chen (2017), and EPA ozone monitoring stations U.S. Environmental Protection Agency (2026). Unlike the matched synthetic experiments, these observational datasets do not provide known population expectations or guarantee the strict covariate-shift assumption. We therefore treat them as empirical stress tests rather than finite-sample theorem verification.

NOAA and EPA Ozone evaluate spatial transfer across disjoint monitoring stations, while $\mathrm { P M } _ { 2 . 5 }$ evaluates temporal transfer (training on 2013–2014, evaluating on 2015–2016). In all experiments, a random-forest predictor is trained separately and frozen before evaluation. Target outcomes are used only for final scoring. While KMM improves point estimation in several domains (full point estimation comparisons, including unweighted and logistic importance weighting baselines, are reported in Table 14, Appendix H.2), our primary focus remains on evaluating confidence interval calibration under data dependence.

GP-Based Dependence Correction We evaluate the proposed GP-derived efective sample size directly on these real dependent sequences. For each task, the KMM point estimate is held fixed, and only the variance calculation is modified: the classical interval uses the nominal source size (Nominal n), whereas our interval replaces n with $n _ { \mathrm { e f f } } ^ { \mathrm { G P } }$ estimated from the source data. To evaluate finite-sample coverage rigorously, we utilize a rolling-window protocol over 250 independent trials, extracting contiguous sequences of length n from the source stations. Target outcomes are revealed only after all intervals have been fixed. Full details of the experimental protocol, data splits, and grid-search hyperparameter selection are provided in Appendix H.3.

Across all three domains, replacing the nominal sample size with the GP-derived efective sample size increases empirical coverage relative to the nominal i.i.d. KMM interval (Table 4). For target-mean estimation, GP $n _ { \mathrm { e f f } }$ reaches the prespecified 0.92 reference level for $\mathrm { P M } _ { 2 . 5 }$ (0.924) and EPA Ozone (0.968), while remaining below it for NOAA (0.908). In all three cases, the GP intervals are narrower than those obtained using Input $n _ { \mathrm { e f f } }$

For target-risk estimation, the results are more mixed. GP $n _ { \mathrm { e f f } }$ reaches the reference level for $\mathrm { P M } _ { 2 . 5 }$ (0.944), but coverage remains below the reference level for EPA Ozone (0.884) and NOAA (0.712). Nevertheless, it improves substantially over the corresponding nominal intervals (0.744 and 0.504, respectively). For EPA Ozone, this increase in coverage is obtained with a smaller radius than Input $n _ { \mathrm { e f f } }$ (0.0031 versus 0.0056); the corresponding Winkler scores are 0.0081 and 0.0114.

<table><tr><td rowspan="2">Method</td><td colspan="2">Target Mean (Y)</td><td colspan="2">Target Risk (L)</td></tr><tr><td>Coverage</td><td>Radius</td><td>Coverage</td><td>Radius</td></tr><tr><td colspan="5">Dataset:  $\mathbf { P M } _ { 2 . 5 }$  (Temporal Shift)</td></tr><tr><td>Unweighted</td><td>0.604*</td><td>0.0189</td><td>0.852*</td><td>0.0016</td></tr><tr><td>Nominal n</td><td>0.848*</td><td>0.0672</td><td>0.900*</td><td>0.0056</td></tr><tr><td>Block-Bootstrap</td><td>0.884*</td><td>0.0832</td><td>0.916*</td><td>0.0068</td></tr><tr><td>Input neff</td><td>0.928</td><td>0.1359</td><td>0.952</td><td>0.0104</td></tr><tr><td>GP neff [Ours]</td><td>0.924</td><td>0.1231</td><td>0.944</td><td>0.0098</td></tr><tr><td colspan="5">Dataset: EPA Ozone (Spatial Shift)</td></tr><tr><td>Unweighted</td><td>0.144*</td><td>0.0061</td><td>0.428*</td><td>0.0004</td></tr><tr><td>Nominal n</td><td>0.580*</td><td>0.0243</td><td>0.744*</td><td>0.0014</td></tr><tr><td>Block-Bootstrap</td><td>0.640*</td><td>0.0285</td><td>0.772*</td><td>0.0016</td></tr><tr><td>Input neff</td><td>0.980</td><td>0.1026</td><td>0.976</td><td>0.0056</td></tr><tr><td>GP  $\scriptstyle n _ { \mathrm { e f f } }$  [Ours]</td><td>0.968</td><td>0.0757</td><td>0.884*</td><td>0.0031</td></tr></table>

Table 4: Confidence Interval Calibration under Dependent Sampling on Real Datasets. Empirical coverage and interval radius for target-mean and target-risk estimation. Asterisks (<sup>∗</sup>) indicate significant empirical undercoverage below the relaxed nominal 0.92 level. Bold denotes the best valid practical estimator (tightest radius among methods with valid coverage).
<table><tr><td colspan="5">Dataset: NOAA (Extreme Spatial Shift Stress-Test)</td></tr><tr><td>Unweighted</td><td>0.088*</td><td>0.0128</td><td>0.152*</td><td>0.0005</td></tr><tr><td>Nominal n</td><td>0.772*</td><td>0.0589</td><td>0.504*</td><td>0.0020</td></tr><tr><td>Block-Bootstrap</td><td>0.776*</td><td>0.0810</td><td>0.568*</td><td>0.0021</td></tr><tr><td>Input neff</td><td>0.944</td><td>0.2179</td><td>0.884*</td><td>0.0071</td></tr><tr><td>GP neff [Ours]</td><td>0.908*</td><td>0.1794</td><td>0.712*</td><td>0.0033</td></tr></table>

These results illustrate the coverage–width trade-of between the two dependence corrections. Input $n _ { \mathrm { e f f } }$ is generally more conservative, whereas GP $n _ { \mathrm { e f f } }$ produces narrower intervals but does not uniformly attain the reference coverage level. The discrepancy is most pronounced for NOAA target-risk estimation.

Because these are observational transfer tasks, departures from the covariate-shift and residualindependence assumptions may contribute to the remaining errors, together with limited source–target overlap and finite target references. We therefore interpret the experiment as an empirical stress test of the dependence correction rather than as verification of finite-sample coverage guarantees. In particular, dependence correction changes the confidence interval assigned to correlated sampling but does not correct systematic transfer bias.

## 5 CONCLUSION AND LIMITATIONS

We develop a finite-sample analysis of Kernel Mean Matching under dependent sampling. The resulting risk bounds replace nominal sample sizes by efective sample sizes and separate three sources of dificulty: dependence, source–target overlap, and residual RKHS mean-matching error. This extends the usual i.i.d. analysis of KMM to settings in which source and target observations are spatially or temporally dependent.

A key implication is that the efective sample size relevant for risk estimation is not determined by the input sequence alone. It depends on the function of the data being averaged. We make this dependence explicit through a GP surrogate that propagates input correlation through functional smoothness, yielding a closed-form, estimand-dependent approximation to $n _ { \mathrm { e f f } }$ . This distinction also clarifies the role of overlap: dependence correction changes the confidence associated with correlated sampling, but cannot compensate for insuficient source–target overlap.

Limitations and Future Work. Our main anal ysis assumes stationary dependence and focuses on risk estimation for a fixed predictor. Appendix F provides a uniform extension over hypothesis classes using Block Rademacher Complexity, while a fuller treatment of model selection and training under dependence remains open. The practical GP approximation additionally relies on estimating the input dependence structure and the functional length scale, and therefore inherits errors from these estimates and from misspecification of the GP surrogate. Extending the framework to nonstationary and more general dependence structures, and developing data-driven procedures for selecting the KMM kernel bandwidth and clipping level B, are natural directions for future work.

## AI use statement

In this work, we used generative AI tools for editing and formatting the manuscript to ensure clarity and adherence to the conference guidelines. We have not used generative AI tools for the derivation of mathematical proofs, generation of experimental results, or code writing. We have reviewed all AI-assisted text modifications line by line. We take responsibility for the final content of this work, including text, claims, and artifacts.

## References

Berbee, H. C. P. (1979). Random walks with stationary increments and renewal theory. 112.

Bosq, D. (2012). Nonparametric statistics for stochastic processes: estimation and prediction. Springer Science & Business Media.

Bretherton, C. S., Widmann, M., Dymnikov, V. P., Wallace, J. M., and Blad´e, I. (1999). The efective number of spatial degrees of freedom of a time varying field. Journal of climate, 12(7):1990–2009.

Chen, S. (2017). Beijing Multi-Site Air Quality. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5RK5G.

Gibbs, M. N. (1997). Bayesian gaussian processes for regression and classification.

Gretton, A., Borgwardt, K. M., Rasch, M. J., Sch¨olkopf, B., and Smola, A. (2012). A kernel twosample test. The journal of machine learning research, 13:723–773.

Gretton, A., Smola, A., Huang, J., Schmittfull, M., Borgwardt, K., Sch¨olkopf, B., et al. (2009). Covariate shift by kernel mean matching. Dataset shift in machine learning, 3(4):5.

Huang, J., Gretton, A., Borgwardt, K., Sch¨olkopf, B., and Smola, A. (2006). Correcting sample selection bias by unlabeled data. Advances in neural information processing systems, 19.

Kass, R. E., Carlin, B. P., Gelman, A., and Neal, R. M. (1998). Markov chain monte carlo in practice: a roundtable discussion. The American Statistician, 52(2):93–100.

Koldasbayeva, D. and Zaytsev, A. (2025). Foundation for unbiased cross-validation of spatio-temporal models for species distribution modeling. Ecological Informatics, page 103521.

Kouw, W. M. and Loog, M. (2019). A review of domain adaptation without target labels. IEEE transactions on pattern analysis and machine intelligence, 43(3):766–785.

Kuznetsov, V. and Mohri, M. (2015). Learning theory and algorithms for forecasting non-stationary time series. In Cortes, C., Lawrence, N., Lee, D., Sugiyama, M., and Garnett, R., editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc.

Martino, L., Elvira, V., and Louzada, F. (2017). Efective sample size for importance sampling based on discrepancy measures. Signal Processing, 131:386– 401.

Menne, M. J., Durre, I., Vose, R. S., Gleason, B. E., and Houston, T. G. (2012). An overview of the global historical climatology network-daily database. Journal of atmospheric and oceanic technology, 29(7):897–910.

Merlev\`ede, F., Peligrad, M., and Rio, E. (2009). Bernstein inequality and moderate deviations under strong mixing conditions. In High dimensional probability V: the Luminy volume, volume 5, pages 273–293. Institute of Mathematical Statistics.

Mohri, M. and Rostamizadeh, A. (2008). Rademacher complexity bounds for non-i.i.d. processes. In Koller, D., Schuurmans, D., Bengio, Y., and Bottou, L., editors, Advances in Neural Information Processing Systems, volume 21. Curran Associates, Inc.

Pinelis, I. (1994). Optimum bounds for the distributions of martingales in banach spaces. The Annals of Probability, pages 1679–1706.

Rasmussen, C. E. (2003). Gaussian processes in machine learning. In Summer school on machine learning, pages 63–71. Springer.

Rio, E. (1993). Covariance inequalities for strongly mixing processes. In Annales de l’IHP Probabilit´es et statistiques, volume 29, pages 587–597.

Rio, E. (1999). Th´eorie asymptotique des processus al´eatoires faiblement d´ependants, volume 31. Springer Science & Business Media.

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schr¨oder, B., Thuiller, W., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8):913–929.

Serov, E., Koldasbayeva, D., and Zaytsev, A. (2026). Kernel mean matching enhances risk estimation under spatial distribution shifts. Scientific Reports, 16(1):6921.

Shimodaira, H. (2000). Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of statistical planning and inference, 90(2):227–244.

Srinivas, N., Krause, A., Kakade, S. M., and Seeger, M. (2009). Gaussian process optimization in the bandit setting: No regret and experimental design. arXiv preprint arXiv:0912.3995.

Sugiyama, M., Krauledat, M., and M¨uller, K.-R. (2007). Covariate shift adaptation by importance weighted cross validation. Journal of Machine Learning Research, 8(5).

U.S. Environmental Protection Agency (2026). Air quality system (aqs): Airdata pre-generated data files. U.S. EPA AirData. Accessed October 2026.

Vats, D., Flegal, J. M., and Jones, G. L. (2019). Multivariate output analysis for markov chain monte carlo. Biometrika, 106(2):321–337.

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., and B¨urkner, P. (2019). Rank-normalization, folding, and localization: An improved r-hat for assessing convergence of mcmc. arxiv preprint arxiv: 190308008.

Winkler, R. L. (1972). A decision-theoretic approach to interval estimation. Journal of the American Statistical Association, 67(337):220–225.

Yu, Y. and Szepesv´ari, C. (2012). Analysis of kernel mean matching under covariate shift. arXiv preprint arXiv:1206.4650.

## CHECKLIST

1. For all models and algorithms presented, check if you include:

(a) A clear description of the mathematical setting, assumptions, algorithm, and/or model. [Yes]

(b) An analysis of the properties and complexity (time, space, sample size) of any algorithm. [Yes]

(c) (Optional) Anonymized source code, with specification of all dependencies, including external libraries. [Yes] Available at Anonymous GitHub.

2. For any theoretical claim, check if you include:

(a) Statements of the full set of assumptions of all theoretical results. [Yes]

(b) Complete proofs of all theoretical results. [Yes]

(c) Clear explanations of any assumptions. [Yes]

3. For all figures and tables that present empirical results, check if you include:

(a) The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). [Yes]

(b) All the training details (e.g., data splits, hyperparameters, how they were chosen). [Yes]

(c) A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). [Yes]

(d) A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:

(a) Citations of the creator If your work uses existing assets. [Yes]

(b) The license information of the assets, if applicable. [Not Applicable]

(c) New assets either in the supplemental material or as a URL, if applicable. [Not Applicable]

(d) Information about consent from data providers/curators. [Not Applicable]

(e) Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [Not Applicable]

5. If you used crowdsourcing or conducted research with human subjects, check if you include:

(a) The full text of instructions given to participants and screenshots. [Not Applicable]

(b) Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Not Applicable]

(c) The estimated hourly wage paid to participants and the total amount spent on participant compensation. [Not Applicable]

# Kernel Mean Matching under Dependent Sampling: Supplementary Materials

## A TAXONOMY OF EFFECTIVE SAMPLE SIZES

Throughout our theoretical analysis, we encounter diferent forms of the efective sample size $n _ { \mathrm { e f f } }$ , depending on whether we bound scalar averages, infinite-dimensional RKHS embeddings, or evaluate specific Gaussian Process realisations. Table 5 summarises these quantities to help navigate the proofs.

Table 5: Taxonomy of efective sample sizes used in the analysis.
<table><tr><td>Symbol</td><td>Controls</td><td>Source of Value</td></tr><tr><td> $n _ { \mathrm { e f f } } ( t )$ </td><td>Scalar average at deviation t</td><td>Mixing Bernstein bound</td></tr><tr><td> $n _ { \mathrm { e f f } } ^ { \mathrm { t r } }$ </td><td>Source scalar effective size (moderate regime)</td><td>Long-run variance proxy</td></tr><tr><td> $n _ { \mathrm { e f f } } ^ { \mathrm { t e } }$  ,RKHS</td><td>RKHS empirical feature-matching (MMD)</td><td>Geometric block size</td></tr><tr><td> $n _ { \mathrm { e f f } } ^ { \mathrm { G P } } ( h )$ </td><td>GP-modeled conditional loss average</td><td>Closed-form GP covariance</td></tr></table>

## B SCALAR EFFECTIVE CONCENTRATION FROM β-MIXING (PROOF OF LEMMA 1)

Lemma 1 (Scalar Concentration under β-Mixing). Let $f : \mathcal { X }  [ a , b ]$ with $A = b - a$ . Assume the strictly stationary sequence $\{ \mathbf { X } _ { i } \} _ { i = 1 } ^ { n }$ is geometrically β-mixing with coeficients $\beta ( k ) \leq c _ { 1 } \exp ( - c _ { 2 } k )$ at lag $k \geq 1 \ ( c _ { 1 } , c _ { 2 } >$ $0 ) _ { . }$ , and has long-run variance proxy $\begin{array} { r } { \sigma _ { f , \infty } ^ { 2 } = \operatorname { V a r } ( f ( \mathbf { X } _ { 1 } ) ) + 2 \sum _ { k = 1 } ^ { \infty } | \operatorname { C o v } ( f ( \mathbf { X } _ { 1 } ) , f ( \mathbf { X } _ { 1 + k } ) ) | } \end{array}$ . There exist constants $\dot { K } _ { 1 } , K _ { 2 } > 0$ , depending only on $c _ { 1 }$ and $c _ { 2 } ,$ , such that for any deviation t in the moderate regime (Definition 1 in Appendix B):

$$
\begin{array} { r l r } {  { \mathbb { P } ( | \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f ( \mathbf { X } _ { i } ) - \mathbb { E } [ f ( \mathbf { X } ) ] | \geq t ) } } \\ & { } & { \leq 2 \exp ( - \frac { 2 n _ { \mathrm { e f f } } t ^ { 2 } } { A ^ { 2 } } ) , } \end{array}\tag{8}
$$

where the efective sample size is given by:

$$
n _ { \mathrm { e f f } } = \frac { K _ { 1 } n A ^ { 2 } } { 4 \left( \sigma _ { f , \infty } ^ { 2 } + A ^ { 2 } / n \right) } .\tag{9}
$$

For general deviations outside the moderate regime, the deviation-dependent $n _ { \mathrm { e f f } } ( t )$ derived in (15) applies. In this section, we provide the full algebraic details of Lemma 1, showing how the Merlev\`ede–Peligrad–Rio Bernstein inequality transforms into a Hoefding-style efective concentration bound.

Formal Definition of β-Mixing. Let $\{ \mathbf { X } _ { i } \} _ { i \in \mathbb { Z } }$ be a strictly stationary sequence on a probability space $( \Omega , A , \mathbb { P } )$ . Let $\mathcal { F } _ { - \infty } ^ { 0 } = \sigma ( . . . , \mathbf { X } _ { 0 } )$ and $\mathcal { F } _ { k } ^ { \infty } = \sigma ( \mathbf { X } _ { k } , \mathbf { X } _ { k + 1 } , \ldots ) $ denote the past and future σ-algebras separated by a time lag $k \geq 1$ . The $\beta \mathrm { . }$ -mixing coeficient at lag k is defined by:

$$
\beta ( k ) : = \mathbb { E } \left[ \operatorname* { s u p } _ { B \in \mathcal { F } _ { k } ^ { \infty } } \left| \mathbb { P } ( B \mid \mathcal { F } _ { - \infty } ^ { 0 } ) - \mathbb { P } ( B ) \right| \right] .\tag{10}
$$

The sequence is geometrically β-mixing if there exist constants $c _ { 1 } , c _ { 2 } > 0$ such that $\beta ( k ) \leq c _ { 1 } \exp ( - c _ { 2 } k )$ for all $k \geq 1$

Theorem 2 (Merlev\`ede–Peligrad–Rio Bernstein Inequality (Merlev\`ede et al., 2009)). Let $\{ U _ { i } \} _ { i = 1 } ^ { n }$ be a centered, strictly stationary sequence with $| U _ { i } | \le A$ almost surely. Assume the sequence is geometrically β-mixing with coeficients $\beta ( k ) \leq c _ { 1 } \exp ( - c _ { 2 } k ) \ f o r \ k \geq 1$ . (Note: Merlev\`ede et al. (2009) prove this inequality under the weaker α-mixing condition, which applies directly since $\begin{array} { r } { \alpha ( k ) \leq \frac { 1 } { 2 } \beta ( k ) \ d j } \end{array}$ . Let the long-run variance proxy be defined as:

$$
\sigma _ { \infty } ^ { 2 } = \mathrm { V a r } ( U _ { 1 } ) + 2 \sum _ { k = 1 } ^ { \infty } \left| \mathrm { C o v } ( U _ { 1 } , U _ { 1 + k } ) \right| .
$$

Then there exist constants $K _ { 1 } , K _ { 2 } > 0$ , depending only on $c _ { 1 }$ and $c _ { 2 } ,$ , such that for all $t > 0 ,$

$$
\mathbb { P } \left( \left| \sum _ { i = 1 } ^ { n } U _ { i } \right| \geq t \right) \leq 2 \exp \left( - \frac { K _ { 1 } t ^ { 2 } } { n \sigma _ { \infty } ^ { 2 } + A ^ { 2 } + K _ { 2 } A t ( \log n ) ^ { 2 } } \right) .\tag{11}
$$

Definition 1 (Moderate-Deviation Regime). Let $K _ { 2 } > 0$ be the mixing constant from Theorem 2. A deviation threshold $t > 0$ is said to fall within the moderate-deviation regime if:

$$
t \leq \frac { \sigma _ { f , \infty } ^ { 2 } } { A K _ { 2 } ( \log n ) ^ { 2 } } .\tag{12}
$$

Proof of Lemma 1. Define the centered variables $U _ { i } = f ( \mathbf { X } _ { i } ) - \mathbb { E } [ f ( \mathbf { X } ) ]$ . Since f takes values in $[ a , b ]$ , the sequence $\{ U _ { i } \}$ is strictly stationary, centered, and bounded by $A = b - a$ . Applying Theorem 2 with the scaled deviation $x = n t$ , we obtain:

$$
\mathbb { P } \left( \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } U _ { i } \right| \geq t \right) = \mathbb { P } \left( \left| \sum _ { i = 1 } ^ { n } U _ { i } \right| \geq n t \right) \leq 2 \exp \left( - \frac { K _ { 1 } ( n t ) ^ { 2 } } { n \sigma _ { f , \infty } ^ { 2 } + A ^ { 2 } + K _ { 2 } A ( n t ) ( \log n ) ^ { 2 } } \right) .\tag{13}
$$

Dividing both the numerator and the denominator of the exponent by n gives:

$$
2 \exp \left( - \frac { K _ { 1 } n t ^ { 2 } } { \sigma _ { f , \infty } ^ { 2 } + A ^ { 2 } / n + K _ { 2 } A t ( \log n ) ^ { 2 } } \right) .\tag{14}
$$

Equating this exponent with the standard Hoefding form $- 2 n _ { \mathrm { e f f } } ( t ) t ^ { 2 } / A ^ { 2 }$ yields the deviation-dependent efective sample size:

$$
n _ { \mathrm { e f f } } ( t ) = \frac { K _ { 1 } n A ^ { 2 } } { 2 \left( \sigma _ { f , \infty } ^ { 2 } + A ^ { 2 } / n + K _ { 2 } A t ( \log n ) ^ { 2 } \right) } .\tag{15}
$$

Now, under the moderate-deviation regime (Definition 1), the logarithmic dependence penalty is bounded by the long-run variance:

$$
K _ { 2 } A t ( \log n ) ^ { 2 } \leq \sigma _ { f , \infty } ^ { 2 } .\tag{16}
$$

Substituting this upper bound into the denominator of $n _ { \mathrm { e f f } } ( t )$ , we obtain:

$$
\sigma _ { f , \infty } ^ { 2 } + A ^ { 2 } / n + K _ { 2 } A t ( \log n ) ^ { 2 } \leq 2 \sigma _ { f , \infty } ^ { 2 } + A ^ { 2 } / n \leq 2 \left( \sigma _ { f , \infty } ^ { 2 } + A ^ { 2 } / n \right) .\tag{17}
$$

Replacing the denominator with this upper bound yields the deviation-independent efective sample size:

$$
n _ { \mathrm { e f f } } = \frac { K _ { 1 } n A ^ { 2 } } { 4 \left( \sigma _ { f , \infty } ^ { 2 } + A ^ { 2 } / n \right) } ,\tag{18}
$$

which completes the proof.

## C HILBERT-VALUED CONCENTRATION VIA BLOCKING (PROOF OF LEMMA 2)

Lemma 2 (Hilbert Concentration via Blocking). Let $\{ \mathbf { Z } _ { i } \} _ { i = 1 } ^ { n }$ be a centered, strictly stationary sequence in a separable Hilbert space H with $\| \mathbf Z _ { i } \| _ { \mathcal { H } } \leq R$ almost surely, and β-mixing coeficients satisfying $\beta ( k ) \leq c _ { 1 } \exp ( - c _ { 2 } k )$ at lag $k \geq 1 \ ( c _ { 1 } , c _ { 2 } > 0 )$ . Then there exists a constant $K _ { H } > 0 ;$ , depending only on $c _ { 1 }$ and $c _ { 2 } .$ , such that for all $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta \colon$

$$
\left\| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbf { Z } _ { i } \right\| _ { \mathcal { H } } \leq K _ { H } R \sqrt { \frac { \left( \log n + \log ( 1 / \delta ) \right) \log ( 8 / \delta ) } { n } } .
$$

In this section, we provide the full proof of the Hilbert-space concentration bound for geometrically β-mixing sequences. The proof relies on the blocking technique and Berbee’s coupling, decoupling the dependent sequence into approximately independent blocks.

Proof of Lemma 2. Fix a block length $q \geq 1$ . We partition the sequence of length n into alternating blocks of size q:

$$
I _ { j } = \{ ( 2 j - 2 ) q + 1 , \ldots , ( 2 j - 1 ) q \} , \quad J _ { j } = \{ ( 2 j - 1 ) q + 1 , \ldots , 2 j q \} ,\tag{19}
$$

for $j = 1 , \dots , L$ , where $L = \lfloor n / ( 2 q ) \rfloor$ ⌋. The remaining observations (at most $2 q )$ contribute at most $2 q R / n$ to the norm of the empirical average, which can be safely absorbed into the universal constants of the final bound.

Define the sum over the odd blocks as $\begin{array} { r } { \mathbf { S } _ { j } = \sum _ { i \in I _ { i } } \mathbf { Z } _ { i } } \end{array}$ . By the almost sure boundedness of $\mathbf { Z } _ { i }$ , we have $\| \mathbf { S } _ { j } \| _ { \mathcal { H } } \leq$ $q R$

Because the original sequence is β-mixing, we can apply Berbee’s coupling lemma (Berbee, 1979) sequentially to the blocks. This guarantees the existence of a sequence of independent block copies $\mathbf { S } _ { 1 } ^ { \ast } , \dots , \mathbf { S } _ { L } ^ { \ast }$ , where each $\mathbf { S } _ { j } ^ { * }$ has the exact same marginal distribution as $\mathbf { S } _ { j } ,$ , such that the probability of the coupled sequence difering from the original sequence is strictly bounded by the sum of the mixing coeficients:

$$
\mathbb { P } \left( \exists j : \mathbf { S } _ { j } \neq \mathbf { S } _ { j } ^ { * } \right) \leq L \beta ( q ) .\tag{20}
$$

Therefore, we can bound the tail probability of the original odd block sums using their independent copies:

$$
\mathbb { P } \left( \left. \sum _ { j = 1 } ^ { L } \mathbf { S } _ { j } \right. _ { \mathcal { H } } \geq t \right) \leq \mathbb { P } \left( \left. \sum _ { j = 1 } ^ { L } \mathbf { S } _ { j } ^ { * } \right. _ { \mathcal { H } } \geq t \right) + L \beta ( q ) .\tag{21}
$$

Since the original variables are centered $( \mathbb { E } [ \mathbf { Z } _ { i } ] = \mathbf { 0 } )$ , the independent block variables $\mathbf { S } _ { j } ^ { * }$ are also centered. They are independent, Hilbert-valued, and strictly bounded by $q R .$ . We can therefore $\mathrm { a p p i y }$ Pinelis’ Hoefding inequality for Hilbert spaces (Pinelis, 1994):

$$
\mathbb { P } \left( \left. \sum _ { j = 1 } ^ { L } \mathbf { S } _ { j } ^ { * } \right. _ { \mathcal { H } } \geq t \right) \leq 2 \exp \left( - \frac { t ^ { 2 } } { 2 L q ^ { 2 } R ^ { 2 } } \right) .\tag{22}
$$

We apply the exact same blocking and coupling argument to the even blocks $J _ { j } .$ . Since $\scriptstyle \sum _ { i = 1 } ^ { n } \mathbf { Z } _ { i }$ is bounded by the sum of the odd and even block sequences via the triangle inequality, we apply a union bound over these two sequences evaluated at the threshold $t / 2$ . Combining these estimates yields:

$$
\mathbb { P } \left( \left. \sum _ { i = 1 } ^ { n } \mathbf { Z } _ { i } \right. _ { \mathcal { H } } \geq t \right) \leq 4 \exp \left( - \frac { t ^ { 2 } } { K L q ^ { 2 } R ^ { 2 } } \right) + 2 L \beta ( q ) ,\tag{23}
$$

for some universal constant $K > 0$

Using $L = \lfloor n / ( 2 q ) \rfloor \ge n / ( 4 q )$ , the exponent scales as $t ^ { 2 } / ( K n q R ^ { 2 } )$ . To balance the exponential term with the mixing penalty $2 L \beta ( q )$ , we utilise the assumption that the sequence is geometrically β-mixing. We select the block size $q$ to grow logarithmically with n and $1 / \delta \colon$

$$
q = \left\lceil \frac { \log n + \log ( 2 c _ { 1 } / \delta ) } { c _ { 2 } } \right\rceil .\tag{24}
$$

With this choice of $q ,$ the mixing penalty becomes negligible, $2 L \beta ( q ) \le \delta / 2$ . Setting the exponential term in (23) to be at most $\delta / 2$ and solving for t yields the desired bound:

$$
\left\| \sum _ { i = 1 } ^ { n } \mathbf { Z } _ { i } \right\| _ { \mathcal { H } } \leq K _ { H } n R \sqrt { \frac { \left( \log n + \log ( 1 / \delta ) \right) \log ( 8 / \delta ) } { n } } .\tag{25}
$$

Dividing by n finishes the proof of Lemma 2.

## D PROOF OF THE MAIN KMM BOUND (THEOREM 1)

Proof. Let $\omega _ { i } ^ { * } = \omega ^ { * } ( \mathbf { X } _ { i } ^ { \mathrm { t r } } )$ . We decompose the estimation error of $\hat { \mu } _ { \mathrm { K M M } }$ into three distinct operational terms. This decomposition provides: Term 1 captures the conditional variance of the labels, Term 2 captures the generalization of the optimal density ratio over the dependent covariate sequence, and Term 3 captures the empirical RKHS feature-matching error.

(26)

$$
\begin{array} { r l } { \hat { \mu } _ { \mathrm { K W M } } - \mu _ { \mathrm { t e } } = } & { \underbrace { \frac { 1 } { n _ { \mathrm { t r } } } \frac { n _ { \mathrm { t r } } } { \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \hat { w } _ { i } ( Y _ { i } ^ { \mathrm { t r } } - m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) ) } } _ { \mathrm { T e r m ~ 1 : ~ C o d d : i t i o n a l ~ R e s i d u a l ~ N e s i d u a l ~ N o i s e } } } \\ & { + \underbrace { \left( \frac { 1 } { n _ { \mathrm { t r } } } \frac { n _ { \mathrm { t r } } } { \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \hat { w } _ { i } ^ { * } m } \alpha _ { i } ^ { * } m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) - \mu _ { \mathrm { t e } } \right) } _ { \mathrm { T e r m ~ 2 : ~ T r u e s R a t i o ~ G e n e r a l i z a t i o n } } } \\ & { + \underbrace { \frac { 1 } { n _ { \mathrm { t r } } } \frac { n _ { \mathrm { t r } } } { \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } ( \hat { w } _ { i } - \omega _ { i } ^ { * } ) m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) } } _ { \mathrm { T e r m ~ 3 : ~ R e t e N ~ P e a t u r e ~ M a t e h i n ~ E e r o r } } . } \end{array}\tag{27}
$$

(28)

Term 1 (Conditional Residual Noise): Under Assumption $2 ,$ the residual errors $\epsilon _ { i } = Y _ { i } ^ { \mathrm { t r } } - m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } )$ are conditionally independent given all observed covariates $( { \bf X } ^ { \mathrm { t r } } , { \bf X } ^ { \mathrm { t e } } )$ , with mean zero. Since ${ \cal Y } _ { i } ^ { \mathrm { t r } } ~ \in ~ [ 0 , 1 ]$ and $m ( \mathbf { X } _ { i } ^ { \mathrm { { t r } } } ) ~ \in ~ [ 0 , 1 ]$ , each residual $\epsilon _ { i }$ is supported on the interval $[ - m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) , 1 - m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) ]$ , whose length is $( 1 ~ -$ $m ( \mathbf { X } _ { i } ^ { \mathrm { { t r } } } ) ) - ( - m ( \mathbf { X } _ { i } ^ { \mathrm { { t r } } } ) ) = 1$

Because the weights $\hat { w } _ { i }$ are measurable with respect to the covariates $( { \bf X } ^ { \mathrm { t r } } , { \bf X } ^ { \mathrm { t e } } )$ and satisfy $0 \leq \hat { w } _ { i } \leq B$ , they act as fixed constants when conditioning on the features. Applying Hoefding’s inequality to the sum of independent, bounded variables $\textstyle \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } { \hat { w } } _ { i } \epsilon _ { i }$ conditional on $\mathbf { X } ^ { \mathrm { t r } } , \mathbf { X } ^ { \mathrm { t e } }$ , the efective variance factor is bounded by:

$$
\sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \hat { w } _ { i } ^ { 2 } \cdot 1 ^ { 2 } \leq n _ { \mathrm { t r } } B ^ { 2 } .
$$

Consequently, for any $t > 0$

$$
\mathbb { P } \left( \left| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \hat { w } _ { i } \boldsymbol { \epsilon } _ { i } \right| \geq t \left| \mathbf { X } ^ { \mathrm { t r } } , \mathbf { X } ^ { \mathrm { t e } } \right. \right) \leq 2 \exp \left( - \frac { 2 ( n _ { \mathrm { t r } } t ) ^ { 2 } } { n _ { \mathrm { t r } } B ^ { 2 } } \right) = 2 \exp \left( - \frac { 2 n _ { \mathrm { t r } } t ^ { 2 } } { B ^ { 2 } } \right) .
$$

Setting this probability to $\delta / 3 .$ , we obtain that with probability at least $1 - \delta / 3 \mathrm { : }$

$$
\left| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \hat { w } _ { i } ( Y _ { i } ^ { \mathrm { t r } } - m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) ) \right| \leq B \sqrt { \frac { 1 } { 2 n _ { \mathrm { t r } } } \log \frac { 6 } { \delta } } .\tag{29}
$$

Notice that because the label noise is conditionally independent, Term 1 concentrates with the nominal sample size $n _ { \mathrm { t r } }$ , entirely unafected by covariate persistence.

Term 2 (True-Ratio Generalization): For (27), define the function $v ( { \bf x } ) = \omega ^ { * } ( { \bf x } ) m ( { \bf x } )$ . Since $0 \leq m ( \mathbf { x } ) \leq 1$ and $0 \leq \omega ^ { * } ( \mathbf { x } ) \leq B , v ( \mathbf { x } )$ takes values in [0, B] with range $A = B$ . Under covariate shift (Assumption 1 and 3), $\begin{array} { r } { \mathbb { E } _ { P _ { \mathrm { t r } } } [ v ( \mathbf { X } ) ] = \int _ { \mathcal { X } } \omega ^ { * } ( \mathbf { x } ) m ( \mathbf { x } ) P _ { \mathrm { t r } } ( d \mathbf { x } ) = \int _ { \mathcal { X } } m ( \mathbf { x } ) P _ { \mathrm { t e } } ( d \mathbf { x } ) = \mu _ { \mathrm { t e } } } \end{array}$

Applying Lemma 1 with confidence level $\delta / 3$ to the geometrically β-mixing sequence $\{ { \bf X } _ { i } ^ { \mathrm { t r } } \}$ , the deviation thresh old is $\begin{array} { r } { t \ = \ B \sqrt { \frac { \log ( 6 / \delta ) } { 2 n _ { \mathrm { e f f } } ^ { \mathrm { t r } } } } \ \leq \ \sqrt { \frac { 2 ( \sigma _ { v , \infty } ^ { 2 } + B ^ { 2 } / n _ { \mathrm { t r } } ) \log ( 6 / \delta ) } { K _ { 1 } n _ { \mathrm { t r } } } } } \end{array}$ . The burn-in condition (1) guarantees that t satisfies the moderate-deviation regime of Definition $\begin{array} { r } { 1 \ ( t \leq \frac { \sigma _ { v , \infty } ^ { 2 } } { B K _ { 2 } ( \log n _ { \mathrm { t r } } ) ^ { 2 } } ) } \end{array}$ . Consequently, with probability at least $1 - \delta / 3$

$$
\left| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \omega _ { i } ^ { * } m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) - \mu _ { \mathrm { t e } } \right| \leq B \sqrt { \frac { 1 } { 2 n _ { \mathrm { e f f } } ^ { \mathrm { t r } } } \log \frac { 6 } { \delta } } .\tag{30}
$$

Term 3 (RKHS Feature Matching Error): For (28), since $m \in { \mathcal { H } }$ , the reproducing property gives $m ( \mathbf { x } ) =$ $\langle m , \Phi ( \mathbf { x } ) \rangle _ { \mathcal { H } } .$ . Applying Cauchy–Schwarz, the triangle inequality, and the optimality of wˆ in the KMM objective (2) (observing that $\boldsymbol { \omega } ^ { * } = ( \omega ^ { * } ( \mathbf { X } _ { 1 } ^ { \mathrm { t r } } ) , \dots , \omega ^ { * } ( \mathbf { X } _ { n _ { \mathrm { t r } } } ^ { \mathrm { t r } } ) ) ^ { \top } \in [ 0 , B ] ^ { n _ { \mathrm { t r } } }$ is a feasible candidate):

$$
\begin{array} { r l } { \displaystyle \left| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } ( \hat { w } _ { i } - \omega _ { i } ^ { * } ) m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) \right| = \displaystyle \left| \left. m , \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } ( \hat { w } _ { i } - \omega _ { i } ^ { * } ) \Phi ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) \right. _ { \mathcal { H } } \right| } & { } \\ { \displaystyle } & { \leq \| m \| _ { \mathcal { H } } \left\| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } ( \hat { w } _ { i } - \omega _ { i } ^ { * } ) \Phi ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) \right\| _ { \mathcal { H } } } \\ { \displaystyle } & { \leq \| m \| _ { \mathcal { H } } \left\| \left\| \hat { \mu } _ { \Phi } ^ { \mathrm { t r } } ( \hat { w } ) - \hat { \mu } _ { \Phi } ^ { \mathrm { t e } } \right\| _ { \mathcal { H } } + \left\| \hat { \mu } _ { \Phi } ^ { \mathrm { t r } } ( \omega ^ { * } ) - \hat { \mu } _ { \Phi } ^ { \mathrm { t e } } \right\| _ { \mathcal { H } } \right) } \\ { \displaystyle } & { \leq 2 \| m \| _ { \mathcal { H } } \left\| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \omega _ { i } ^ { * } \Phi ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) - \frac { 1 } { n _ { \mathrm { t e } } } \sum _ { j = 1 } ^ { n _ { \mathrm { t e } } } \Phi ( \mathbf { X } _ { j } ^ { \mathrm { t r } } ) \right\| _ { \mathcal { H } } . } \end{array}\tag{31}
$$

Centering both empirical embeddings around the population target mean embedding $\mu _ { \Phi } \ = \ \mathbb { E } _ { P _ { \mathrm { t e } } } [ \Phi ( \mathbf { X } ) ] \ =$ $\mathbb { E } _ { P _ { \mathrm { t r } } } [ \omega ^ { * } ( \mathbf { X } ) \Phi ( \mathbf { X } ) ]$ , we decompose:

$$
\left\| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \omega _ { i } ^ { * } \Phi ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) - \frac { 1 } { n _ { \mathrm { t e } } } \sum _ { j = 1 } ^ { n _ { \mathrm { t e } } } \Phi ( \mathbf { X } _ { j } ^ { \mathrm { t e } } ) \right\| _ { \mathcal { H } } \leq \left\| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \mathbf { Z } _ { i } ^ { \mathrm { t r } } \right\| _ { \mathcal { H } } + \left\| \frac { 1 } { n _ { \mathrm { t e } } } \sum _ { j = 1 } ^ { n _ { \mathrm { t e } } } \mathbf { Z } _ { j } ^ { \mathrm { t e } } \right\| _ { \mathcal { H } } ,\tag{32}
$$

where $\mathbf { Z } _ { i } ^ { \mathrm { t r } } = \omega ^ { * } ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) \Phi ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) - \mu _ { \Phi }$ and $\mathbf { Z } _ { j } ^ { \mathrm { t e } } = \Phi ( \mathbf { X } _ { j } ^ { \mathrm { t e } } ) - \mu _ { \Phi }$

As established in the main text, $\| \Phi ( \mathbf { x } ) \| _ { \mathcal { H } } \leq C$ . Hence $\| \mu _ { \Phi } \| _ { \mathcal { H } } \leq C$ , giving the uniform bounds:

$$
\| \mathbf { Z } _ { i } ^ { \mathrm { t r } } \| _ { \mathcal { H } } \leq \| \omega ^ { * } ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) \Phi ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) \| _ { \mathcal { H } } + \| \mu _ { \Phi } \| _ { \mathcal { H } } \leq B C + C \leq 2 B C ,
$$

where the final inequality holds since $B = \operatorname* { s u p } _ { \mathbf x } \omega ^ { * } ( \mathbf x ) \geq 1$ for any valid density ratio. For the target sequence, $\| \mathbf { Z } _ { j } ^ { \mathrm { t e } } \| _ { \mathcal { H } } \leq 2 C$ . Applying Lemma 2 separately to the source sequence with $R = 2 B C$ at confidence level $\delta / 6 ,$ , and to the target sequence with $R = 2 C$ at confidence level $\delta / 6$ , each bound holds with probability at least $1 - \delta / 6$ By a union bound, with probability at least $1 - \delta / 3$

$$
\left\| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } \omega _ { i } ^ { * } \Phi ( { \mathbf { X } _ { i } ^ { \mathrm { t r } } } ) - \frac { 1 } { n _ { \mathrm { t e } } } \sum _ { j = 1 } ^ { n _ { \mathrm { t e } } } \Phi ( { \mathbf { X } _ { j } ^ { \mathrm { t e } } } ) \right\| _ { \mathcal { H } } \le C \Delta _ { \mathrm { M M D } } ( \delta ) ,\tag{33}
$$

where the empirical RKHS feature-matching discrepancy $\Delta _ { \mathrm { M M D } } ( \delta )$ is given in closed form by:

$$
\Delta _ { \mathrm { M M D } } ( \delta ) : = K _ { H } \sqrt { \log \frac { 4 8 } { \delta } } \left[ B \sqrt { \frac { \log n _ { \mathrm { t r } } + \log ( 6 / \delta ) } { n _ { \mathrm { t r } } } } + \sqrt { \frac { \log n _ { \mathrm { t e } } + \log ( 6 / \delta ) } { n _ { \mathrm { t e } } } } \right] ,\tag{34}
$$

with the blocking constant $K _ { H } > 0$ from Lemma 2.

For interpretability and physical intuition in the main text (Theorem 1), we abstract this exact non-asymptotic bound by defining the RKHS efective sample sizes as:

$$
n _ { \mathrm { e f f , R K H S } } ^ { \mathrm { t r } } : = \frac { n _ { \mathrm { t r } } } { \log n _ { \mathrm { t r } } + \log ( 6 / \delta ) } , \quad \mathrm { a n d } \quad n _ { \mathrm { e f f , R K H S } } ^ { \mathrm { t e } } : = \frac { n _ { \mathrm { t e } } } { \log n _ { \mathrm { t e } } + \log ( 6 / \delta ) } .\tag{35}
$$

This definition allows us to express the MMD error conceptually as $\begin{array} { r } { \Delta _ { \mathrm { M M D } } ( \delta ) \propto \frac { B } { \sqrt { n _ { \mathrm { e f f , R K H S } } ^ { \mathrm { t r } } } } + \frac { 1 } { \sqrt { n _ { \mathrm { e f f , R K H S } } ^ { \mathrm { t e } } } } } \end{array}$ . This formulation explicitly exposes how data dependence deflates the efective sample size for infinite-dimensional embeddings via the geometric blocking penalty, and reintroduces the target sequence sample size $\left( n _ { \mathrm { t e } } \right)$ into the final theoretical guarantee.

Substituting (34) into (31) yields:

$$
\left| \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } ( \hat { w } _ { i } - \omega _ { i } ^ { * } ) m ( \mathbf { X } _ { i } ^ { \mathrm { t r } } ) \right| \leq 2 C \| m \| _ { \mathcal { H } } \Delta _ { \mathrm { M M D } } ( \delta ) .\tag{36}
$$

Final Synthesis (Proof of Theorem 1): Finally, we assemble the complete bound. We apply a union bound across the three failure events associated with Term 1 (29), Term 2 (30), and Term 3 (36). Since each event fails with probability at most $\delta / 3$ , their intersection holds with probability at least $1 - \delta .$

By applying the triangle inequality to the initial decomposition, the absolute estimation error is strictly bounded by the sum of these three non-asymptotic components:

$$
\begin{array} { r l } { | \hat { \mu } _ { \mathrm { K M M } } - \mu _ { \mathrm { t e } } | \le \underbrace { B \sqrt { \frac { \log ( 6 / \delta ) } { 2 n _ { \mathrm { t r } } } } } _ { \mathrm { T e r m ~ 1 : ~ } \mathcal { E } _ { \mathrm { n o i s e } } } } & { + \underbrace { B \sqrt { \frac { \log ( 6 / \delta ) } { 2 n _ { \mathrm { e f f } } ^ { \mathrm { t r } } } } } _ { \mathrm { T e r m ~ 2 : ~ } \mathcal { E } _ { \mathrm { s h i t t } } } } \\ & { + \underbrace { 2 C K _ { H } \sqrt { \log { \frac { 4 8 } { \delta } } } \| m \| _ { \mathcal { H } } \left( \frac { B } { \sqrt { n _ { \mathrm { e f f , ~ R K H S } } ^ { \mathrm { t r } } } } + \frac { 1 } { \sqrt { n _ { \mathrm { e f f , R K H S } } ^ { \mathrm { t r } } } } \right) } _ { \mathrm { T e r m ~ 3 : ~ } \mathrm { E x a c t ~ N o n - A s y m p t o t i c } ~ \mathcal { E } _ { \mathrm { M M P } } } . } \end{array}\tag{37}
$$

By abstracting the absolute universal constants (2C and $K _ { H } )$ and the logarithmic confidence factor $( \sqrt { { \log ( 4 8 / \delta ) } } )$ into the asymptotic $\mathcal { O } ( \cdot )$ notation, we exactly recover the generalized statement of Theorem 1 presented in the main text. This completes the proof. □

## E GAUSSIAN PROCESS EFFECTIVE SAMPLE SIZE DERIVATIONS

This appendix provides the rigorous mathematical derivations for the Gaussian Process (GP) efective sample size introduced in Section 3.3. We first establish the general marginalised covariance for a GP evaluated over correlated inputs. To mathematically formalise the fundamental smoothness trade-of, we then derive exact closed-form expressions and asymptotic limits for two contrasting covariance families: the infinitely smooth Squared-Exponential (SE) kernel and the rough Mat´ern-1/2 (Exponential) kernel. Finally, we contrast these stochastic realisations with the asymptotic behaviour of a general fixed deterministic function.

## E.1 Marginalized Covariance for GP Realizations (Squared-Exponential Case)

Suppose u is a realization of a zero-mean GP with a squared-exponential covariance $k _ { h } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \sigma _ { u } ^ { 2 } \exp ( - \| \mathbf { x } -$ $\mathbf { x } ^ { \prime } | | \bar { 2 } / 2 h ^ { 2 } )$ . To maintain theoretical consistency with the compact domain X (Assumption 1), we formally assume the Gaussian inputs are truncated to a suficiently large compact bounding box. Following standard truncation arguments in GP learning theory (Srinivas et al., 2009), the exponentially small probability mass lost in the tails ensures our analytical moments hold up to a negligible approximation error.

If the covariates are drawn from a stationary GP $\mathbf X _ { s } \sim \mathcal N ( \pmb \mu , \pmb \sigma _ { x } ^ { 2 } \mathbf I _ { d } )$ , the diference between two inputs is strictly Gaussian:

$$
\mathbf { X } _ { s } - \mathbf { X } _ { t } \sim \mathcal { N } \left( \mathbf { 0 } , 2 \sigma _ { x } ^ { 2 } \{ 1 - r ( \| s - t \| ) \} \mathbf { I } _ { d } \right) .\tag{38}
$$

Integrating over the distribution of the d-dimensional inputs, the prior-averaged kernel covariance becomes:

$$
\mathbb { E } _ { \mathbf { X } } \big [ k _ { h } ( \mathbf { X } _ { s } , \mathbf { X } _ { t } ) \big ] = \sigma _ { u } ^ { 2 } \operatorname* { d e t } \left( \mathbf { I } _ { d } + \frac { 2 \sigma _ { x } ^ { 2 } \{ 1 - r ( \| s - t \| ) \} } { h ^ { 2 } } \mathbf { I } _ { d } \right) ^ { - 1 / 2 } .\tag{39}
$$

Subtracting the independent-input baseline gives the exact covariance of the centred process:

$$
\bar { \gamma } _ { u } ( s , t ) = \sigma _ { u } ^ { 2 } \left[ \operatorname* { d e t } \left( \mathbf { I } _ { d } + \frac { 2 \sigma _ { x } ^ { 2 } \{ 1 - r ( \Vert s - t \Vert ) \} } { h ^ { 2 } } \mathbf { I } _ { d } \right) ^ { - 1 / 2 } - \operatorname* { d e t } \left( \mathbf { I } _ { d } + \frac { 2 \sigma _ { x } ^ { 2 } } { h ^ { 2 } } \mathbf { I } _ { d } \right) ^ { - 1 / 2 } \right] .\tag{40}
$$

Letting $a _ { h } = 2 \sigma _ { x } ^ { 2 } / h ^ { 2 }$ and $r _ { k } = r ( k ) = \exp ( - k ^ { 2 } / 2 \ell _ { x } ^ { 2 } )$ , the determinant simplifies to $( 1 + a _ { h } \{ 1 - r _ { k } \} ) ^ { d }$ . Consequently, the autocorrelation function in $\mathbb { R } ^ { d }$ takes the exact closed form:

$$
\bar { \rho } _ { u } ( k ) = \frac { ( 1 + a _ { h } \{ 1 - r _ { k } \} ) ^ { - d / 2 } - ( 1 + a _ { h } ) ^ { - d / 2 } } { 1 - { ( 1 + a _ { h } ) } ^ { - d / 2 } } .\tag{41}
$$

Defining $\kappa _ { h , d } = ( 1 + a _ { h } ) ^ { - d / 2 }$ , the infinite autocorrelation sum becomes:

$$
\bar { S } ( a _ { h } , \ell _ { x } , d ) = \sum _ { k = 1 } ^ { \infty } \bar { \rho } _ { u } ( k ) = \sum _ { k = 1 } ^ { \infty } \frac { ( 1 + a _ { h } ( 1 - r _ { k } ) ) ^ { - d / 2 } - \kappa _ { h , d } } { 1 - \kappa _ { h , d } } .\tag{42}
$$

Short-Range Dependence Limit $( \ell _ { x } \ll 1 )$ : The first lag dominates the sum. Using a first-order expansion, we find:

$$
\bar { S } ( a _ { h } , \ell _ { x } ) = L ( a _ { h } ) \exp \left( - \frac { 1 } { 2 \ell _ { x } ^ { 2 } } \right) + \mathcal { O } \left( \exp \left( - \frac { 1 } { \ell _ { x } ^ { 2 } } \right) \right) , \quad \mathrm { w h e r e ~ } L ( a _ { h } ) = \frac { \sqrt { 1 + a _ { h } } + 1 } { 2 ( 1 + a _ { h } ) } .\tag{43}
$$

Long-Range Dependence Limit $( \ell _ { x } \gg 1 )$ : The sum is approximated by the integral via the Euler-Maclaurin formula:

$$
\bar { S } ( a _ { h } , \ell _ { x } ) = \ell _ { x } \mathcal { C } _ { d } ( a _ { h } ) - \frac { 1 } { 2 } + \mathcal { O } ( \ell _ { x } ^ { - 1 } ) ,\tag{44}
$$

where the constant $\mathcal { C } _ { d } ( a _ { h } )$ explicitly captures the GP smoothness efect:

$$
\mathcal { C } _ { d } ( a _ { h } ) = \int _ { 0 } ^ { \infty } \frac { \left( 1 + a _ { h } \{ 1 - \exp ( - u ^ { 2 } / 2 ) \} \right) ^ { - d / 2 } - \left( 1 + a _ { h } \right) ^ { - d / 2 } } { 1 - \left( 1 + a _ { h } \right) ^ { - d / 2 } } d u .\tag{45}
$$

Analyzing $\mathcal { C } _ { d } ( a _ { h } )$ yields two sub-regimes: 1. If $a _ { h } \ll 1$ (Smooth GP), Taylor expansion yields $\begin{array} { r } { { \mathcal C } _ { d } ( a _ { h } ) = \sqrt { \frac { \pi } { 2 } } - } \end{array}$ $\begin{array} { r } { \frac { 3 a _ { h } } { 4 } \left( \sqrt { \frac { \pi } { 2 } } - \frac { \sqrt { \pi } } { 2 } \right) + \mathcal { O } ( a _ { h } ^ { 2 } ) } \end{array}$ . 2. If $a _ { h } \gg 1$ (Rough GP), asymptotic analysis yields $\begin{array} { r } { \mathcal { C } _ { d } ( a _ { h } ) = \frac { \log a _ { h } } { \sqrt { 2 a _ { h } } } + \mathcal { O } ( a _ { h } ^ { - 1 / 2 } ) } \end{array}$

## E.2 Efective Sample Size for the Mat´ern Kernel Family (e.g., Exponential Kernel)

To demonstrate that our efective sample size framework seamlessly extends beyond the infinitely smooth Squared-Exponential (SE) kernel, we analyse the Mat´ern family of covariance functions. The Mat´ern class is governed by a smoothness parameter $\nu > 0$ . As $\nu \to \infty$ , it recovers the SE kernel. However, for finite $\nu ,$ the functions are rougher (only $\lfloor \nu - 0 . 5 \rfloor$ times mean-square diferentiable).

A widely used instantiation is the $\mathrm { M a t e r n { - } 1 / 2 }$ kernel, which corresponds to the Exponential (or Laplace) covariance function: $k _ { 1 / 2 } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) = \sigma _ { u } ^ { 2 } \exp { ( - \| \mathbf { x } - \mathbf { x } ^ { \prime } \| / h ) }$ . This models rough, non-diferentiable processes.

For scalar inputs $( d = 1 )$ , we can derive an exact analytical closed form for the marginalised covariance, similar to the SE case. As established in (40), the diference between two inputs is $\Delta X = X _ { s } - X _ { t } \sim { \mathcal { N } } ( 0 , s _ { k } ^ { 2 } )$ , where the variance is $s _ { k } ^ { 2 } = 2 \sigma _ { x } ^ { 2 } \{ 1 - r ( k ) \}$ . The prior-averaged kernel covariance requires evaluating the Gaussian integral over the absolute value:

$$
\mathbb { E } _ { X } [ k _ { 1 / 2 } ( X _ { s } , X _ { t } ) ] = \int _ { - \infty } ^ { \infty } \sigma _ { u } ^ { 2 } \exp \left( - \frac { | x | } { h } \right) \frac { 1 } { \sqrt { 2 \pi s _ { k } ^ { 2 } } } \exp \left( - \frac { x ^ { 2 } } { 2 s _ { k } ^ { 2 } } \right) d x .\tag{46}
$$

By completing the square in the exponent and exploiting the symmetry of the Gaussian distribution, this integra evaluates exactly to:

$$
\mathbb { E } _ { X } [ k _ { 1 / 2 } ( X _ { s } , X _ { t } ) ] = \sigma _ { u } ^ { 2 } \exp \left( \frac { s _ { k } ^ { 2 } } { 2 h ^ { 2 } } \right) \mathrm { e r f c } \left( \frac { s _ { k } } { \sqrt { 2 } h } \right) ,\tag{47}
$$

where er $\begin{array} { r } { \mathrm { \dot { c } } ( z ) = \frac { 2 } { \sqrt { \pi } } \int _ { z } ^ { \infty } e ^ { - t ^ { 2 } } d t } \end{array}$ is the complementary error function.

To express the induced autocorrelation $\bar { \rho } _ { u } ( k )$ , let $\begin{array} { r } { \gamma _ { k } ~ = ~ \frac { s _ { k } } { \sqrt { 2 } h } ~ = ~ \frac { \sigma _ { x } } { h } \sqrt { 1 - r _ { k } } } \end{array}$ , and let $\begin{array} { r } { \gamma _ { \infty } ~ = ~ \frac { \sigma _ { x } } { h } } \end{array}$ denote the independent-input baseline limit. The exact centered autocorrelation for the Exponential GP is:

$$
\bar { \rho } _ { u , \mathrm { E x p } } ( k ) = \frac { \exp ( \gamma _ { k } ^ { 2 } ) \operatorname { e r f c } ( \gamma _ { k } ) - \exp ( \gamma _ { \infty } ^ { 2 } ) \operatorname { e r f c } ( \gamma _ { \infty } ) } { 1 - \exp ( \gamma _ { \infty } ^ { 2 } ) \operatorname { e r f c } ( \gamma _ { \infty } ) } .\tag{48}
$$

Asymptotic Smoothness Trade-of. This closed-form derivation mathematically formalizes the smoothness trade-of. For large input deviations relative to the length-scale $( \gamma _ { k } \ \gg 1 )$ , the complementary error function admits the strict asymptotic expansion:

$$
\exp ( \gamma _ { k } ^ { 2 } ) \mathrm { e r f c } ( \gamma _ { k } ) = \frac { 1 } { \gamma _ { k } \sqrt { \pi } } \left( 1 - \mathcal { O } ( \gamma _ { k } ^ { - 2 } ) \right) .\tag{49}
$$

Although single-lag autocorrelations for both kernels asymptotically scale as $\mathcal { O } ( 1 / \gamma _ { k } )$ for large input deviations (since $a _ { h } \{ 1 - r _ { k } \} = 2 \gamma _ { k } ^ { 2 }$ , yielding $\begin{array} { r } { ( 1 + 2 \gamma _ { k } ^ { 2 } ) ^ { - 1 / 2 } = \frac { 1 } { \sqrt { 2 } \gamma _ { k } } ( 1 - \mathcal { O } ( \gamma _ { k } ^ { - 2 } ) ) } \end{array}$ for the SE model), their cumulative persistence sums behave fundamentally diferently across lags. Because the SE kernel models an infinitely diferentiable process, its spectral density decays super-exponentially; in the continuum limit, this parabolic smoothness near zero lag produces an extended cumulative dependence with an additional logarithmic enhancement $\mathcal { C } _ { d } ( a _ { h } ) =$ $\textstyle \frac { \log a _ { h } } { \sqrt { 2 a _ { h } } } + \mathcal { O } ( a _ { h } ^ { - 1 / 2 } )$ . Conversely, the non-diferentiable Mat´ern-1/2 process introduces high-frequency roughness that enforces faster sequence decorrelation across neighboring points, eliminating this logarithmic inflation and resulting in a smaller infinite sum S<sup>¯</sup> and a larger, more favorable efective sample size.

## E.3 Asymptotic Analysis of the GP Efective Sample Size for Fixed Functions

To contextualise the GP realisation results, consider a general fixed deterministic function evaluated over the same spatio-temporal data. Suppose the covariates are generated by a stationary Gaussian process $\mathbf X _ { s } \sim \mathcal N ( \pmb \mu , \pmb \sigma _ { x } ^ { 2 } \mathbf I _ { d } )$ with spatial/temporal covariance:

$$
\mathrm { C o v } ( \mathbf { X } _ { s } , \mathbf { X } _ { t } ) = \sigma _ { x } ^ { 2 } \mathbf { I } _ { d } \exp \left( - \frac { \Vert s - t \Vert ^ { 2 } } { 2 \ell _ { x } ^ { 2 } } \right) ,
$$

where $\ell _ { x }$ determines the persistence length. Assuming f is continuously diferentiable with bounded second derivatives, a first-order Taylor expansion around the mean $\pmb { \mu }$ yields the leading term:

$$
\mathrm { C o v } ( f ( \mathbf { X } _ { s } ) , f ( \mathbf { X } _ { t } ) ) = \sigma _ { x } ^ { 2 } \| \nabla f ( \pmb { \mu } ) \| ^ { 2 } r ( \| s - t \| ) + \mathcal { O } ( \sigma _ { x } ^ { 4 } ) .
$$

Consequently, the leading term of the function’s autocorrelation strictly matches the input correlation up to higher-order residuals: $\rho _ { f } ( k ) = r ( k ) + \mathcal { O } ( \sigma _ { x } ^ { 2 } )$ .

For a squared-exponential input autocorrelation, the infinite autocorrelation sum is explicitly given by the Jacob theta function $\vartheta _ { 3 } \colon$

$$
S ( \ell _ { x } ) = \sum _ { k = 1 } ^ { \infty } \exp \left( - \frac { k ^ { 2 } } { 2 \ell _ { x } ^ { 2 } } \right) = \frac { 1 } { 2 } \left[ \vartheta _ { 3 } \left( 0 , \exp \left\{ - \frac { 1 } { 2 \ell _ { x } ^ { 2 } } \right\} \right) - 1 \right] .
$$

Therefore, the exact asymptotic efective sample size is expressed as:

$$
n _ { \mathrm { e f f } } ( \ell _ { x } ) = \frac { n } { 1 + 2 S ( \ell _ { x } ) } = \frac { n } { \vartheta _ { 3 } \left( 0 , \exp \left\{ - 1 / ( 2 \ell _ { x } ^ { 2 } ) \right\} \right) } .
$$

In the short-range limit $( \ell _ { x } \to 0 )$ , the first lag dominates, yielding $n _ { \mathrm { e f f } } = n ( 1 - \mathcal { O } ( \exp ( - 1 / 2 \ell _ { x } ^ { 2 } ) ) )$ . Conversely, under long-range dependence $( \ell _ { x } \to \infty )$ , approximating the sum with an integral via the Euler-Maclaurin formula gives $1 + 2 S ( \ell _ { x } ) = \sqrt { 2 \pi } \ell _ { x } + \mathcal { O } ( 1 )$ . Thus, the efective sample size scales asymptotically as $\begin{array} { r } { n _ { \mathrm { e f f } } ( \ell _ { x } ) = \frac { n } { \sqrt { 2 \pi } \ell _ { x } + \mathcal { O } ( 1 ) } } \end{array}$

## F EXTENSION TO HYPOTHESIS CLASSES VIA EMPIRICAL PROCESSES

While Theorem 1 provides a pointwise guarantee for a fixed regression function m or a fixed loss ℓ, evaluating a data-dependent learned predictor requires a simultaneous uniform guarantee over a hypothesis class ${ \mathcal F } .$ . For independent data, this is governed by standard Rademacher complexity. For dependent β-mixing sequences, applying a naive substitution $n \  \ n _ { \mathrm { e f f } }$ is theoretically insuficient. Instead, uniform convergence must be established using Block Rademacher Complexity (Mohri and Rostamizadeh, 2008; Kuznetsov and Mohri, 2015).

Following the blocking construction from Lemma 2, we partition the sequence into alternating blocks of size $q ,$ with $L = \lfloor n / ( 2 q ) \rfloor$ . By Berbee’s coupling lemma, we decouple the dependent blocks into an independent block sequence $B _ { 1 } ^ { * } , \ldots , B _ { L } ^ { * }$ , where each block contains q observations: $B _ { j } ^ { * } = ( \mathbf { X } _ { j , 1 } ^ { * } , \ldots , \mathbf { X } _ { j , q } ^ { * } )$ . The empirical Block Rademacher Complexity of $\mathcal { F }$ is defined as:

$$
\Re _ { L } ( \mathcal { F } , q ) = \frac { 1 } { L } \mathbb { E } _ { \pmb { \sigma } } \left[ \operatorname* { s u p } _ { f \in \mathcal { F } } \sum _ { j = 1 } ^ { L } \sigma _ { j } \left( \frac { 1 } { q } \sum _ { k = 1 } ^ { q } f ( \mathbf { X } _ { j , k } ^ { * } ) \right) \right] ,\tag{50}
$$

where $\sigma _ { 1 } , \ldots , \sigma _ { L }$ are independent Rademacher variables.

Applying McDiarmid’s inequality to the coupled independent blocks and accounting for odd and even subsequences via the triangle inequality, the uniform deviation over $\mathcal { F }$ is bounded with probability at least $1 - \delta$ by:

$$
\operatorname* { s u p } _ { f \in \mathcal { F } } \left| \frac { 1 } { n } \sum _ { i = 1 } ^ { n } f ( \mathbf { X } _ { i } ) - \mathbb { E } [ f ( \mathbf { X } ) ] \right| \leq 2 \Re _ { L } ( \mathcal { F } , q ) + 3 A \sqrt { \frac { \log ( 4 / \delta ) } { 2 L } } + 2 L \beta ( q ) ,\tag{51}
$$

where $A = \operatorname* { s u p } _ { f \in { \mathcal { F } } } ( \operatorname* { s u p } _ { \mathbf { x } } f ( \mathbf { x } ) - \operatorname* { i n f } _ { \mathbf { x } } f ( \mathbf { x } ) )$ denotes the uniform bound on the function range. By choosing the block size to satisfy $\begin{array} { r } { q = \lceil \frac { \log n + \log ( 4 c _ { 1 } / \delta ) } { c _ { \infty } } \rceil } \end{array}$ , the mixing penalty becomes negligible: $2 L \beta ( q ) \le \delta / 2$ . The number of efective independent blocks $L ^ { c _ { 2 } } = \lfloor n / ( 2 q ) \rfloor = \mathcal { O } ( n / \log n )$ then governs the sample-complexity geometry of the uniform bound.

Application to RKHS and GP Function Classes. For an RKHS ball ${ \mathcal { F } } _ { R } = \{ f \in { \mathcal { H } } : \| f \| _ { \mathcal { H } } \leq R \}$ with a kernel bounded by $C ^ { 2 }$ , let $\begin{array} { r } { \bar { \Phi } _ { j } = \frac { 1 } { q } \sum _ { k = 1 } ^ { q } \Phi ( \mathbf { X } _ { j , k } ^ { * } ) } \end{array}$ be the block feature mean of the j-th block. Since $\| \Phi ( \mathbf { x } ) \| _ { \mathcal { H } } \leq C$ it follows that $\| \bar { \Phi } _ { j } \| _ { \mathcal { H } } \leq C$ . By Cauchy-Schwarz, the block Rademacher complexity evaluates cleanly to:

$$
\Re _ { L } ( \mathcal { F } _ { R } , q ) = \frac { 1 } { L } \mathbb { E } _ { \sigma } \left[ \operatorname* { s u p } _ { \| f \| _ { \mathcal { H } } \leq R } \left. f , \sum _ { j = 1 } ^ { L } \sigma _ { j } \bar { \Phi } _ { j } \right. _ { \mathcal { H } } \right] \leq \frac { R } { L } \mathbb { E } _ { \sigma } \left\| \sum _ { j = 1 } ^ { L } \sigma _ { j } \bar { \Phi } _ { j } \right\| _ { \mathcal { H } } \leq \frac { R C } { \sqrt { L } } .\tag{52}
$$

Notice that replacing the nominal sample size n with the efective number of blocks $L = \mathcal { O } ( n / \log n )$ inflates the statistical complexity, mirroring the efective sample size deflation observed in our pointwise bounds.

This formulation mathematically connects KMM risk estimation with empirical process theory under dependent sampling. While the block size q is determined entirely by the physical mixing rate of the covariate process $\{ { \mathbf { X } } _ { t } \}$ , functional smoothness directly controls the complexity radius $R = \| f \| _ { \mathcal { H } } \colon$ smoother function classes enforce smaller RKHS norms, thereby substantially reducing the complexity numerator $\Re _ { L }$ . However, as established in Section 3.3, smoother functions simultaneously retain more intra-sequence autocorrelation, deflating the pointwise variance-based $n _ { \mathrm { e f f } }$ . Across both pointwise and uniform bounds, the density-ratio overlap penalty B remains an irreducible multiplicative scale, confirming that neither hypothesis smoothness nor complex empirical-process machinery can bypass severe support mismatch.

## G ARTIFICIAL DATA EXPERIMENTAL RESULTS AND IMPLEMENTATION DETAILS

This section provides a comprehensive breakdown of the synthetic experimental setups, expanded empirical validations, and rigorous robustness stress tests that were omitted from the main text due to space constraints. We first detail the exact data-generating processes, followed by the validation of the Target $Y$ efective sample size. We then present full-page heatmaps of our theoretical calibration and conclude with extensive ablations over hyperparameters, non-diferentiable functional families, and non-stationary correlation structures.

## G.1 Detailed Synthetic Experimental Setup

To thoroughly evaluate the statistical properties of our dependent KMM framework, we construct a highly controlled synthetic environment where ground-truth expectations, density ratios, and efective sample sizes are analytically tractable.

Data-Generating Process for Covariates We generate continuous covariate sequences $X _ { t } \in \mathbb { R }$ for our 1D benchmark sweeps and hyperparameter ablations (extended to $\mathbf { X } _ { t } \in \mathbb { R } ^ { 2 }$ for the 2D spatial calibration heatmaps in Appendix G.4). To simulate temporal or spatial persistence, source covariates are sampled along a stationary Gaussian trajectory:

$$
X _ { t } ^ { \mathrm { t r } } = \mu _ { \mathrm { t r } } + \tau _ { \mathrm { t r } } \xi _ { t } , \quad \mathrm { w h e r e } \quad \xi \sim \mathcal { N } ( \mathbf { 0 } , \Sigma ) .\tag{53}
$$

The covariance matrix $\pmb { \Sigma } \in \mathbb { R } ^ { n _ { \mathrm { t r } } \times n _ { \mathrm { t } } }$ r enforces a Squared-Exponential (RBF) autocorrelation structure across sequence indices:

$$
\mathrm { C o v } ( \xi _ { s } , \xi _ { t } ) = \sigma _ { 0 } ^ { 2 } \exp \left( - \frac { | s - t | ^ { 2 } } { 2 \ell _ { x } ^ { 2 } } \right) .\tag{54}
$$

Here, $\ell _ { x }$ dictates the input correlation length and $\sigma _ { 0 } ^ { 2 } = 1 . 0$ is the baseline variance. Target covariates $X _ { t } ^ { \mathrm { t e } }$ are sampled under covariate shift with marginal mean $\mu _ { \mathrm { t e } }$ . In our baseline benchmark sweeps, we set default sample sizes $n _ { \mathrm { t r } } = n _ { \mathrm { t e } } = 9 6$ , shift parameters $\mu _ { \mathrm { t r } } = 0 . 0 , \mu _ { \mathrm { t e } } = 0 . 8 5$ , scaling factors $\tau _ { \mathrm { t r } } = \tau _ { \mathrm { t e } } = 1 . 0$ , and default sequence correlation length $\ell _ { x } = 3 . 0$

Response Surfaces and Estimands To evaluate the GP-induced smoothness correction, the true response is tested under two distinct structural environments:

1. Matched GP Realization: The raw response $f _ { \mathrm { r a w } }$ is synthesized via Random Fourier Features (using 512 Fourier basis functions) matching a stationary Gaussian Process with length-scale h $( h = 1 . 5$ by default). To enforce realistic bounded support, the function is passed through a non-linear activation:

$$
f _ { 0 } ( x ) = 0 . 5 + 0 . 4 \operatorname { t a n h } ( 0 . 9 x + f _ { \operatorname { r a w } } ( x ) ) .\tag{55}
$$

Observed targets include additive Gaussian noise $Y = \mathrm { c l i p } ( f _ { 0 } ( X ) + \varepsilon , 0 , 1 )$ with $\varepsilon \sim \mathcal { N } ( 0 , \sigma _ { \varepsilon } ^ { 2 } )$ and noise level $\sigma _ { \varepsilon } = 0 . 0 4$

2. Deterministic Misspecified Surface: To verify robustness outside GP priors, we also evaluate a fixed deterministic harmonic function $f _ { 0 } ( \mathbf { x } ) = \sin ( \omega x _ { 1 } ) + 0 . 5 \cos ( 2 \omega x _ { 2 } )$ with spatial frequency $\omega \in [ 0 . 5 , 3 . 0 ]$

For target mean estimation, we evaluate the raw response Y. For target risk estimation, we utilize a fixed, imperfect predictor $g ( x ) = 0 . 6 5 f _ { 0 } ( x ) + 0 . 1 5$ and evaluate the squared loss $\ell ( g ( x ) , y ) = ( g ( x ) - y ) ^ { 2 }$

Ground-truth target expectations $\theta _ { Y } = \mathbb { E } _ { P _ { \mathrm { t e } } } [ Y ]$ and $\theta _ { \mathrm { R i s k } } = \mathbb { E } _ { P _ { \mathrm { t e } } } [ \ell ( g ( X ) , Y ) ]$ are computed via high-precision Monte Carlo integration over 150,000 independent target samples drawn from $P _ { \mathrm { t e } }$

KMM Optimization and Variogram Calibration Kernel Mean Matching weights $\mathbf { w } \in \mathbb { R } ^ { n _ { \mathrm { t r } } }$ are estimated by solving the quadratic program:

$$
\operatorname* { m i n } _ { \mathbf { w } } \frac { 1 } { 2 } \mathbf { w } ^ { \top } \left( \mathbf { K } _ { s s } + \lambda \mathbf { I } \right) \mathbf { w } - \kappa ^ { \top } \mathbf { w } \quad \mathrm { s . t . } \quad 0 \leq w _ { i } \leq B , \quad \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } w _ { i } = n _ { \mathrm { t r } } ,\tag{56}
$$

where $\begin{array} { r } { \kappa _ { i } = \frac { n _ { \mathrm { t r } } } { n _ { \mathrm { t e } } } \sum _ { j = 1 } ^ { n _ { \mathrm { t e } } } k ( X _ { i } ^ { \mathrm { t r } } , X _ { j } ^ { \mathrm { t e } } ) } \end{array}$ . We utilize an RBF kernel $k ( x , x ^ { \prime } ) = \exp ( - ( x - x ^ { \prime } ) ^ { 2 } / ( 2 \sigma ^ { 2 } ) )$ with bandwidth $\sigma = 1 . 0$ , ridge regularization $\lambda = 1 0 ^ { - 4 }$ , and default clipping bound $B = 1 5 . 0$ . The optimization problem is solved using the interior-point solver in cvxopt.

To isolate confidence radius calibration from empirical variogram estimation noise during finite sweeps, the functional length-scales $h _ { Y }$ and $h _ { \ell }$ are pre-calibrated via empirical variogram regression over an independent calibration pool of 2,500 source observations. All Monte Carlo sweeps are conducted over $N = 3 0 0$ independent trials initialized with random seed 42.

Baseline Formulations and Implementation For robust benchmarking, we compare our proposed GPcalibrated KMM radius against multiple standard estimators:

• Unweighted: Direct sample average over source data $\begin{array} { r } { \frac { 1 } { n _ { \mathrm { t r } } } \sum _ { i = 1 } ^ { n _ { \mathrm { t r } } } Y _ { i } ^ { \mathrm { t r } } } \end{array}$ , which ignores covariate shift.

• Nominal n: The classical i.i.d. KMM confidence interval, which weights samples via KMM but uses the nominal sample size $n _ { \mathrm { t r } }$ in Wald’s variance estimator.

• Input $n _ { \mathrm { e f f } } \colon \mathrm { ~ A ~ }$ dependence correction that penalizes variance using the raw input autocorrelation $r _ { k } \colon$ $\begin{array} { r } { n _ { \mathrm { e f f } } ( \ell _ { x } ) = n _ { \mathrm { t r } } / [ 1 + 2 \sum _ { k = 1 } ^ { n _ { \mathrm { t r } } - 1 } ( 1 - k / n _ { \mathrm { t r } } ) r _ { k } ] } \end{array}$ , ignoring functional smoothness.

• Block-Bootstrap: A moving-block bootstrap with block length $q = \lceil n _ { \mathrm { t r } } ^ { 1 / 3 } \rceil ~ ( q = 6$ at default $n = 9 6 )$ using 200 bootstrap resamples.

• Strict Bound (Theorem 1): The non-asymptotic concentration bound evaluating the conservative radius $B \sqrt { \log ( 6 / \alpha ) / ( 2 n _ { \mathrm { e f f } } ^ { \mathrm { G P } } ) }$

• Oracle IW: Importance weighting using the true analytical density ratio $w ^ { * } ( x ) = p _ { \mathrm { t e } } ( x ) / p _ { \mathrm { t r } } ( x )$

• Oracle i.i.d.: Target mean and risk evaluated directly on independent target samples drawn from $P _ { \mathrm { t e } }$

## G.2 Extended Evaluation Metrics and Full Benchmark Results

In Section 4, we focused on Coverage and Radius to evaluate the confidence intervals, as point estimates are identical across all KMM-based variance estimators. Table 6 provides the complete set of evaluation metrics, including Mean Absolute Error (MAE) for point estimation and the Winkler Score (WS) for comprehensive interval evaluation.

The Winkler Score (Winkler, 1972) is a proper scoring rule for prediction intervals that rewards narrow intervals but heavily penalizes intervals that fail to cover the true value. For a nominal $1 - \alpha$ confidence interval $[ L , U ]$ (where $\alpha = 0 . 0 5 )$ and a true target value y, it is defined as:

$$
\mathrm { W S } = ( U - L ) + { \frac { 2 } { \alpha } } ( L - y ) \mathbb { I } ( y < L ) + { \frac { 2 } { \alpha } } ( y - U ) \mathbb { I } ( y > U ) .\tag{57}
$$

Lower WS indicates better interval calibration. As shown in Table 6, while the Strict Bound (Theorem 1) guarantees full coverage, its massive worst-case width results in a severely degraded WS $( > 1 0 . 0 )$ . Similarly, while Input $n _ { \mathrm { e f f } }$ is valid, it produces unnecessarily wide intervals (WS: 0.339 for Y ; 0.031 for Risk). Our proposed GP $n _ { \mathrm { e f f } }$ provides the optimal balance, achieving the lowest WS (0.276 for Y ; 0.026 for Risk) among valid estimators by analytically incorporating functional decorrelation.

## G.3 Validation under Deterministic Misspecification

In the main text (Figure 1), we demonstrated that our closed-form GP efective sample size formula accurately matches empirical variance when the underlying response surface is drawn from a matched Gaussian Process.

To verify that this theoretical result is robust and not merely an artifact of Gaussian assumptions, Figure 2 presents the validation on a deterministic harmonic surface $f _ { 0 } ( { \bf x } ) = \sin ( \omega x _ { 1 } ) + 0 . 5 \cos ( 2 \omega x _ { 2 } )$ . Despite the absence of a stochastic GP prior generating the labels, the empirical variance-based $n _ { \mathrm { e f f } }$ for both Target Y and Risk evaluation remains tightly concentrated along the diagonal identity line. This indicates that the GP surrogate can remain informative beyond the matched GP setting, although its accuracy varies with the dependence and functional regimes considered below.

## G.4 Efective Sample Size Heatmaps

To ensure that the excellent calibration observed in our scatter plots is not an artifact of a specific, ”lucky” hyperparameter combination, we evaluate the ratio of the empirical efective sample size to our predicted efective sample size: $n _ { \mathrm { e f f } } ^ { \mathrm { e m p } } / n _ { \mathrm { e f f } } ^ { \mathrm { G P } } ( h )$ . A ratio of exactly 1.0 indicates perfect theoretical calibration.

Table 6: Full Benchmark Results including Point Estimation and Winkler Score. Point estimates (MAE) are identical for Nominal $n ,$ Block-Bootstrap, Input $n _ { \mathrm { e f f } } , \mathrm { G P \ } n _ { \mathrm { e f f } }$ , and the Strict Bound, as they rely on the exact same KMM weights. An asterisk (<sup>∗</sup>) indicates empirical undercoverage significantly below the nomina 0.95 level (< 0.925). Winkler Score (WS, lower is better) penalizes both interval width and undercoverage. Bold highlights the best valid practical estimator (tightest radius and lowest WS while maintaining valid coverage). Strict Thm 1 and Oracle methods are excluded from best-method highlighting.
<table><tr><td></td><td colspan="4">Target Mean (Y)</td><td colspan="4">Target Risk (L)</td></tr><tr><td>Method</td><td>MAE</td><td>Coverage</td><td>Radius</td><td>WS↓</td><td>MAE</td><td>Coverage</td><td>Radius</td><td>WS↓</td></tr><tr><td>Unweighted</td><td>0.1138</td><td> $0 . 0 1 7 ^ { * }$ </td><td>0.0362</td><td>3.182</td><td>0.0048</td><td> $0 . 0 8 0 ^ { * }$ </td><td>0.0019</td><td>0.119</td></tr><tr><td>Nominal n</td><td>0.0196</td><td> $0 . 9 1 7 ^ { * }$ </td><td>0.0626</td><td>0.134</td><td>0.0025</td><td> $0 . 8 5 3 ^ { * }$ </td><td>0.0055</td><td>0.024</td></tr><tr><td>Block-Bootstrap</td><td>0.0196</td><td>1.000</td><td>0.4538</td><td>0.908</td><td>0.0025</td><td>0.967</td><td>0.0133</td><td>0.026</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.0196</td><td>1.000</td><td>0.1696</td><td>0.339</td><td>0.0025</td><td>0.987</td><td>0.0150</td><td>0.031</td></tr><tr><td>GP  $n _ { \mathrm { e f f } } \ ( \mathbf { O u r s } )$ </td><td>0.0196</td><td>1.000</td><td>0.1380</td><td>0.276</td><td>0.0025</td><td>0.970</td><td>0.0120</td><td>0.026</td></tr><tr><td>Strict Thm 1</td><td>0.0196</td><td>1.000</td><td>5.2189</td><td>10.438</td><td>0.0025</td><td>1.000</td><td>5.1220</td><td>10.244</td></tr><tr><td>Oracle IW</td><td>0.0334</td><td> $0 . 4 7 7 ^ { * }$ </td><td>0.0275</td><td>0.590</td><td>0.0027</td><td> $0 . 6 2 7 ^ { * }$ </td><td>0.0033</td><td>0.033</td></tr><tr><td>Oracle i.i.d.</td><td>0.0115</td><td>0.903*</td><td>0.0257</td><td>0.072</td><td>0.0010</td><td> $0 . 9 1 7 ^ { * }$ </td><td>0.0024</td><td>0.006</td></tr></table>

![](images/8909a8d83ae34ca2a9433637338db1c15eb96b36ec7a24fa7a2c53b1e296ce87.jpg)  
Figure 2: Validation under Deterministic Misspecification. Empirical variance-based $n _ { \mathrm { e f f } } \ \mathrm { v s . }$ . our predicted $n _ { \mathrm { e f f } } ^ { \mathrm { G P } }$ for both Target $Y$ and Risk when evaluated on a deterministic, non-GP harmonic surface. The surrogate formula remains highly predictive.

We compute this ratio across a dense, two-dimensional grid varying both the input dependence $\ell _ { x } \in [ 0 . 5 , 4 . 0 ]$ and the functional complexity parameters $( h \in [ 0 . 5 , 4 . 0 ]$ for GP realizations, and spatial frequency $\omega \in [ 0 . 5 , 3 . 0 ]$ for deterministic surfaces).

Figures 3 and 4 display these heatmaps across the full parameter grids. The empirical to-predicted ratio is mostly within approximately [0.7, 1.25], although larger deviations occur in some strong-dependence and smoothfunction regimes. Ratios below one indicate that the closed-form approximation overestimates the empirical efective sample size in these settings. Overall, the results show that the GP approximation tracks the empirical efective sample size across a broad range of regimes, while also identifying conditions under which its calibration deteriorates.

![](images/2ab2692273a10e9b758e553b8c238daa9232d306e9b640549befb2613c3f2c46.jpg)  
Figure 3: GP Realizations (Full Grid). Ratio of the empirical to predicted $n _ { \mathrm { e f f } }$ across diverse GP length scales (h) and input dependencies $( \ell _ { x } )$ . Ratios near 1.0 indicate close calibration. The approximation tracks the empirical efective sample size across much of the grid, with larger deviations in some strong-dependence and smooth-function regimes.

![](images/cf9ffea8b3d788c2f25a08360bd1ee97283468896b2f205d27708524498ebb36.jpg)  
Figure 4: Deterministic Misspecification (Full Grid). Ratio of the empirical to predicted $n _ { \mathrm { e f f } }$ when the true function is a deterministic harmonic surface with varying frequency ω. The GP surrogate remains informative beyond the matched stochastic-GP setting, although its calibration varies across dependence and frequency regimes.

## G.5 Hyperparameters Ablations

To further dissect the behavior of the GP-induced efective sample size, we evaluate the KMM confidence intervals across comprehensive sweeps of sample size (n, Table 7), input dependence severity $( \ell _ { x } )$ , functional smoothness (h), and maximum clipping bounds (B). Across all ablations, we report interval coverage, interval radius, and the Winkler Score (WS).

We evaluate interval calibration using the Winkler Score (WS; detailed in Appendix G.2), which penalizes intervals that are either overly wide or fail to cover the target.

To evaluate validity, we track empirical coverage over $N = 3 0 0$ independent Monte Carlo repetitions. For nominal 0.95 intervals, the binomial two-sigma lower bound is approximately $0 . 9 5 - 1 . 9 6 { \sqrt { 0 . 9 5 \times 0 . 0 5 / 3 0 0 } } \approx 0 . 9 2 5$ . We flag any coverage falling below this threshold with an asterisk (<sup>∗</sup>) as statistically significant undercoverage.

• Sample Size on Risk (n, Table 7): Nominal n fails across all evaluated sample sizes (collapsing to 0.647 at $n = 3 2$ and reaching only 0.913<sup>∗</sup> at $n = 2 5 6 )$ , demonstrating that sample size alone cannot cure dependence-induced undercoverage. At $n = 3 2 .$ , finite-sample estimation violates the moderate-deviation burn-in threshold $n / ( \log n ) ^ { 4 } \ \geq$ const (Theorem 1), exhibiting expected transient undercoverage (0.850). Across all operational regimes $( n \geq 4 8 )$ $\mathrm { G P } ~ n _ { \mathrm { e f f } }$ reliably maintains nominal coverage $( \geq 0 . 9 4 3$ , reaching 0.997 at $n = 2 5 6 )$ while consistently yielding the sharpest valid intervals and superior Winkler Scores (up to 21% narrower radius than Input $n _ { \mathrm { e f f } } )$ .

• Input Dependence on Risk $( \ell _ { x } , \mathrm { { T a b l e 8 } ) }$ : As correlation length scales from $\ell _ { x } = 1 . 5 \mathrm { t o } 6 . 0 .$ , raw covariate autocorrelation intensifies and Nominal n fails across all configurations (dropping to $0 . 7 6 7 ^ { * }$ coverage at $\ell _ { x } = 4 . 0$ and remaining undercovered throughout). In contrast, GP $n _ { \mathrm { e f f } }$ dynamically adapts to sequence persistence. Across all moderate-to-strong dependence regimes $( \ell _ { x } \ \geq 2 . 0 )$ , GP $n _ { \mathrm { e f f } }$ achieves robust valid coverage $( \geq 0 . 9 4 7 )$ while consistently yielding approximately 21% narrower intervals and superior Winkler Scores compared to the overly conservative Input $n _ { \mathrm { e f f } }$ . At weak dependence $( \ell _ { x } = 1 . 5 )$ , where decorrelation is fast and empirical estimation adds relative noise, $\mathrm { G P } ~ n _ { \mathrm { e f f } }$ exhibits a minor coverage dip (0.923<sup>∗</sup>), making Input $n _ { \mathrm { e f f } }$ the preferred valid estimator (0.953), though $\mathrm { G P } ~ n _ { \mathrm { e f f } }$ still achieves the lowest Winkler Score (0.0159).

• Functional Smoothness on Risk (h, Table 9): This sweep empirically validates the Smoothness-Dependence Tradeof (Section 3.3). Because naive Input $n _ { \mathrm { e f f } }$ accounts for raw coordinate autocorrelation, it produces an entirely static variance correction that ignores response smoothness, yielding overly conservative intervals. Meanwhile, Nominal n fails across all bandwidths (consistently under-covering below $0 . 9 1 8 ^ { * }$ dropping to 0.827<sup>∗</sup>). In stark contrast, $\mathrm { G P } ~ n _ { \mathrm { e f f } }$ dynamically adapts to the functional length-scale $h ,$ identifying that the error surface decorrelates faster than raw inputs. Consequently, GP $n _ { \mathrm { e f f } }$ dominates across all evaluated length-scales, maintaining robust coverage $( \geq 0 . 9 7 0 )$ while shrinking the confidence radius by 17% to 28% and achieving substantially lower Winkler Scores.

• Clipping Bound on Risk (B, Table 10): The clipping parameter B dictates the maximum allowable importance weight in KMM, controlling the bias-variance trade-of under domain overlap. As B scales from 5.0 to 30.0, Nominal n fails across all thresholds (dropping to $0 . 7 8 0 ^ { * }$ at $B = 5 . 0$ and $0 . 7 9 0 ^ { * }$ at $B = 3 0 . 0 )$ In contrast, GP $n _ { \mathrm { e f f } }$ demonstrates complete stability across all bounds, preserving a consistent variance penalty regardless of the clipping level. This empirically confirms the theoretical separation of density-ratio overlap (B) from sequence persistence in Theorem 1. Across the entire sweep, GP $n _ { \mathrm { e f f } }$ maintains robust valid coverage $( \geq 0 . 9 4 7 .$ reaching 0.980) while consistently shrinking interval radius by approximately 21% and yielding substantially lower Winkler Scores than Input $n _ { \mathrm { e f f } }$

## G.6 Kernel Family and Roughness Ablation (Mat´ern Processes)

The derivation of our closed-form $\mathrm { G P } n _ { \mathrm { e f f } }$ formula relies on a Squared-Exponential (RBF) covariance assumption, which models an infinitely diferentiable process. A theoretical concern is whether applying this RBF-derived correction yields overly optimistic (under-covered) intervals if the true underlying data-generating process is physically rough or non-diferentiable (e.g., admitting a Mat´ern covariance structure).

To stress-test this, we evaluate KMM over data generated from exact spectral Mat´ern realizations. The Mat´ern family is parameterized by a smoothness parameter ν. While $\nu \to \infty$ recovers the RBF kernel, smaller values $( \nu = 5 / 2 , 3 / 2 , 1 / 2 )$ yield increasingly rough, non-diferentiable processes. $\mathrm { M a t { \acute { e } r n { - } } 1 / 2 } ,$ for example, represents a continuous but nowhere-diferentiable path (equivalent to an Exponential covariance). In this experiment, our $\mathrm { G P } ~ n _ { \mathrm { e f f } }$ estimator continues to assume an RBF geometry, deliberately inducing a smoothness misspecification.

Table 11 demonstrates the robustness of the RBF-derived surrogate evaluated over Target Risk. Across all Mat´ern kernels, GP $n _ { \mathrm { e f f } }$ maintains substantially lower Winkler Scores and tighter intervals than the conservative

Table 7: Sample-size scaling of Target Risk intervals $( n ~ \in ~ \{ 3 2 , \ldots , 2 5 6 \} )$ . Bold highlights the best valid practical estimator (tightest radius and lowest Winkler Score among methods maintaining valid coverage). Asterisks (<sup>∗</sup>) indicate significant empirical undercoverage below the nominal 0.95 level $\left( < 0 . 9 2 5 \right)$ . Strict Thm 1 is excluded from best-method highlighting due to worst-case conservatism.
<table><tr><td>n</td><td>Method</td><td>Coverage</td><td>Radius</td><td>Winkler Score ↓</td></tr><tr><td rowspan="4">32</td><td>Unweighted</td><td> $0 . 7 1 7 ^ { * }$ </td><td>0.0026</td><td>0.0201</td></tr><tr><td>Nominal n</td><td> $0 . 6 4 7 ^ { * }$ </td><td>0.0047</td><td>0.0394</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td> $0 . 8 9 3 ^ { * }$ </td><td>0.0123</td><td>0.0325</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td> $0 . 8 5 0 ^ { * }$ </td><td>0.0098</td><td>0.0313</td></tr><tr><td rowspan="5">48</td><td>Unweighted</td><td> $0 . 7 2 7 ^ { * }$ </td><td>0.0022</td><td>0.0143</td></tr><tr><td>Nominal n</td><td> $0 . 7 7 3 ^ { * }$ </td><td>0.0048</td><td>0.0275</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td> $0 . 9 1 0 ^ { * }$ </td><td>0.0128</td><td>0.0290</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.943</td><td>0.0101</td><td>0.0260</td></tr><tr><td>Unweighted</td><td> $0 . 7 7 3 ^ { * }$ </td><td></td><td>0.0124</td></tr><tr><td rowspan="4">64</td><td>Nominal n</td><td> $0 . 7 7 3 ^ { * }$ </td><td>0.0020</td><td></td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.980</td><td>0.0046 0.0124</td><td>0.0218 0.0258</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.947</td><td>0.0098</td><td>0.0217</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">96</td><td>Unweighted Nominal n</td><td> $0 . 7 7 0 ^ { * }$ </td><td>0.0016</td><td>0.0101</td></tr><tr><td>Input</td><td> $0 . 8 0 3 ^ { * }$ </td><td>0.0043</td><td>0.0188</td></tr><tr><td> $n _ { \mathrm { e f f } }$  GP (Ours)</td><td>0.973</td><td>0.0117</td><td>0.0251</td></tr><tr><td> $n _ { \mathrm { e f f } }$ </td><td>0.957</td><td>0.0092</td><td>0.0213</td></tr><tr><td rowspan="4">128</td><td>Unweighted</td><td> $0 . 7 5 0 ^ { * }$ </td><td>0.0014</td><td>0.0083</td></tr><tr><td>Nominal n</td><td> $0 . 8 2 3 ^ { * }$ </td><td>0.0038</td><td>0.0155</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.983</td><td>0.0104</td><td>0.0220</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.970</td><td>0.0082</td><td>0.0181</td></tr><tr><td rowspan="4">192</td><td>Unweighted</td><td> $0 . 7 5 0 ^ { * }$ </td><td>0.0011</td><td>0.0071</td></tr><tr><td>Nominal n</td><td> $0 . 8 8 7 ^ { * }$ </td><td>0.0033</td><td>0.0109</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.987</td><td>0.0091</td><td>0.0187</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.980</td><td>0.0072</td><td>0.0152</td></tr><tr><td rowspan="5">256</td><td>Unweighted</td><td> $0 . 7 4 3 ^ { * }$ </td><td>0.0010</td><td>0.0062</td></tr><tr><td>Nominal n</td><td> $0 . 9 1 3 ^ { * }$ </td><td>0.0031</td><td>0.0076</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>1.000</td><td>0.0085</td><td>0.0170</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.997</td><td>0.0067</td><td>0.0134</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 8: Hyperparameter Ablation: Input Dependence $( \ell _ { x } \in [ 1 . 5 , 6 . 0 ] )$ on Target Risk. Asterisks (<sup>∗</sup>) indicate significant empirical undercoverage below the nominal 0.95 level (< 0.925). Bold highlights the best valid practical estimator (tightest radius and lowest Winkler Score among methods maintaining valid coverage). Strict Thm 1 is excluded from best-method highlighting due to worst-case conservatism.
<table><tr><td> $\ell _ { x }$ </td><td>Method</td><td>Coverage</td><td>Radius</td><td>Winkler Score ↓</td></tr><tr><td>1.5</td><td>Unweighted Nominal n</td><td> $0 . 8 4 3 ^ { * }$   $0 . 8 4 0 ^ { * }$ </td><td>0.0016 0.0041</td><td>0.0061 0.0161</td></tr><tr><td></td><td>Input  $n _ { \mathrm { e f f } }$  GP</td><td>0.953</td><td>0.0078</td><td>0.0176</td></tr><tr><td>2.0</td><td> $n _ { \mathrm { e f f } }$  (Ours)</td><td>0.923*</td><td>0.0062</td><td>0.0159</td></tr><tr><td></td><td>Unweighted</td><td>0.843*</td><td>0.0016</td><td>0.0068</td></tr><tr><td></td><td>Nominal n</td><td>0.870*</td><td>0.0043</td><td>0.0143</td></tr><tr><td></td><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.987</td><td>0.0096</td><td>0.0204</td></tr><tr><td>GP</td><td> $n _ { \mathrm { e f f } } \ ( \mathbf { O u r s } )$ </td><td>0.957</td><td>0.0076</td><td>0.0170</td></tr><tr><td></td><td>Strict Thm 1</td><td>1.000</td><td>4.1488</td><td>8.2975</td></tr><tr><td>3.0</td><td>Unweighted</td><td>0.767*</td><td>0.0016</td><td>0.0091</td></tr><tr><td></td><td>Nominal n</td><td>0.803*</td><td>0.0040</td><td>0.0184</td></tr><tr><td>Input</td><td> $n _ { \mathrm { e f f } }$ </td><td>0.977</td><td>0.0110</td><td>0.0226</td></tr><tr><td>GP</td><td>(Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.947</td><td>0.0086</td><td>0.0191</td></tr><tr><td>4.0</td><td>Unweighted</td><td></td><td></td><td></td></tr><tr><td></td><td>Nominal n</td><td>0.700*</td><td>0.0016</td><td>0.0154</td></tr><tr><td>Input</td><td></td><td>0.767*</td><td>0.0039</td><td>0.0216</td></tr><tr><td>GP</td><td> $n _ { \mathrm { e f f } }$ </td><td>0.980 0.950</td><td>0.0120</td><td>0.0246</td></tr><tr><td>5.0</td><td> $n _ { \mathrm { e f f } }$  (Ours)</td><td></td><td>0.0095</td><td>0.0210</td></tr><tr><td></td><td>Unweighted</td><td>0.680*</td><td>0.0016</td><td>0.0164</td></tr><tr><td>Input</td><td>Nominal n</td><td>0.773*</td><td>0.0039</td><td>0.0244</td></tr><tr><td></td><td> $n _ { \mathrm { e f f } }$ </td><td>0.980</td><td>0.0134</td><td>0.0282</td></tr><tr><td>6.0</td><td> $\mathbf { G P } \ n _ { \mathrm { e f f } } \ \left( \mathbf { O u r s } \right)$ </td><td>0.953</td><td>0.0106</td><td>0.0236</td></tr><tr><td></td><td>Unweighted</td><td>0.620*</td><td>0.0016</td><td>0.0183</td></tr><tr><td></td><td>Nominal n</td><td>0.793*</td><td>0.0042</td><td>0.0199</td></tr><tr><td></td><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.997</td><td>0.0161</td><td>0.0322</td></tr><tr><td>GP</td><td>(Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.987</td><td>0.0127</td><td>0.0259</td></tr></table>

Table 9: Hyperparameter Ablation: Functional Smoothness $( h \in [ 1 . 0 , 3 . 0 ] )$ on Target Risk. Asterisks (<sup>∗</sup>) indicate significant empirical undercoverage below the nominal 0.95 level (< 0.925). Bold highlights the best valid practical estimator (tightest radius and lowest Winkler Score among methods maintaining valid coverage).
<table><tr><td> $^ { h }$ </td><td>Method</td><td>Coverage</td><td>Radius</td><td>Winkler Score ↓</td></tr><tr><td rowspan="5">1.00</td><td>Unweighted</td><td> $0 . 1 7 3 ^ { * }$ </td><td>0.0023</td><td>0.0749</td></tr><tr><td>Nominal n</td><td> $0 . 9 1 0 ^ { * }$ </td><td>0.0062</td><td>0.0194</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.993</td><td>0.0167</td><td>0.0338</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.983</td><td>0.0129</td><td>0.0271</td></tr><tr><td>Unweighted</td><td> $0 . 0 4 7 ^ { * }$ </td><td>0.0018</td><td>0.1597</td></tr><tr><td rowspan="4">1.25</td><td>Nominal n</td><td> $0 . 9 1 7 ^ { * }$ </td><td>0.0072</td><td>0.0183</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.993</td><td>0.0195</td><td>0.0395</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.990</td><td>0.0156</td><td>0.0320</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">1.50</td><td>Unweighted</td><td> $0 . 0 3 7 ^ { * }$ </td><td>0.0015</td><td>0.1371</td></tr><tr><td>Nominal  $n$ </td><td> $0 . 8 2 7 ^ { * }$ </td><td>0.0057</td><td>0.0233</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$  GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.983</td><td>0.0154</td><td>0.0316</td></tr><tr><td></td><td>0.980</td><td>0.0128</td><td>0.0272</td></tr><tr><td rowspan="4">1.75</td><td>Unweighted</td><td>0.137*</td><td>0.0028</td><td>0.1304</td></tr><tr><td>Nominal n</td><td> $0 . 9 0 7 ^ { * }$ </td><td>0.0070</td><td>0.0188</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.997</td><td>0.0189</td><td>0.0378</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.990</td><td>0.0136</td><td>0.0281</td></tr><tr><td rowspan="4">2.00</td><td>Unweighted</td><td> $0 . 2 7 7 ^ { * }$ </td><td>0.0017</td><td>0.0536</td></tr><tr><td>Nominal n</td><td>0.860*</td><td>0.0057</td><td>0.0220</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.987</td><td>0.0154</td><td>0.0323</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.970</td><td>0.0116</td><td>0.0257</td></tr><tr><td rowspan="5">3.00</td><td>Unweighted</td><td>0.490*</td><td>0.0018</td><td>0.0292</td></tr><tr><td>Nominal n</td><td> $0 . 8 7 7 ^ { * }$ </td><td>0.0057</td><td>0.0187</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.990</td><td>0.0153</td><td>0.0310</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.983</td><td>0.0113</td><td>0.0238</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 10: Hyperparameter Ablation: Clipping Bound $( B \in [ 5 . 0 , 3 0 . 0 ] )$ on Target Risk. Asterisks (<sup>∗</sup>) indicate significant empirical undercoverage below the nominal 0.95 level $\left( < \ 0 . 9 2 5 \right)$ . Bold highlights the best valid practical estimator (tightest radius and lowest Winkler Score among methods maintaining valid coverage).
<table><tr><td>B</td><td>Method</td><td>Coverage</td><td>Radius</td><td>Winkler Score ↓</td></tr><tr><td rowspan="5">5.0</td><td>Unweighted</td><td> $0 . 7 6 7 ^ { * }$ </td><td>0.0016</td><td>0.0100</td></tr><tr><td>Nominal n</td><td> $0 . 7 8 0 ^ { * }$ </td><td>0.0030</td><td>0.0154</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.987</td><td>0.0082</td><td>0.0168</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.963</td><td>0.0065</td><td>0.0141</td></tr><tr><td>Unweighted</td><td>0.767*</td><td>0.0016</td><td>0.0100</td></tr><tr><td rowspan="4">7.5</td><td>Nominal n</td><td> $0 . 8 0 3 ^ { * }$ </td><td>0.0035</td><td>0.0147</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.993</td><td>0.0095</td><td>0.0192</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.980</td><td>0.0075</td><td>0.0156</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">10.0</td><td>Unweighted</td><td> $0 . 7 6 7 ^ { * }$ </td><td>0.0016</td><td>0.0100</td></tr><tr><td>Nominal  $n$ </td><td> $0 . 8 3 7 ^ { * }$ </td><td>0.0038</td><td>0.0150</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$  GP (Ours)</td><td>0.990</td><td>0.0103</td><td>0.0209</td></tr><tr><td> $n _ { \mathrm { e f f } }$ </td><td>0.970</td><td>0.0082</td><td>0.0172</td></tr><tr><td rowspan="4">15.0</td><td>Unweighted</td><td>0.767*</td><td>0.0016</td><td>0.0100</td></tr><tr><td>Nominal n</td><td> $0 . 8 0 3 ^ { * }$ </td><td>0.0042</td><td>0.0171</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.987</td><td>0.0112</td><td>0.0231</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.967</td><td>0.0089</td><td>0.0191</td></tr><tr><td rowspan="4">20.0</td><td>Unweighted</td><td> $0 . 7 6 7 ^ { * }$ </td><td>0.0016</td><td>0.0100</td></tr><tr><td>Nominal n</td><td> $0 . 7 9 7 ^ { * }$ </td><td>0.0043</td><td>0.0184</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.987</td><td>0.0118</td><td>0.0245</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.960</td><td>0.0093</td><td>0.0203</td></tr><tr><td rowspan="5">30.0</td><td>Unweighted</td><td>0.767*</td><td>0.0016</td><td>0.0100</td></tr><tr><td>Nominal n</td><td> $0 . 7 9 0 ^ { * }$ </td><td>0.0045</td><td>0.0207</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.980</td><td>0.0122</td><td>0.0254</td></tr><tr><td>GP (Ours)  $n _ { \mathrm { e f f } }$ </td><td>0.947</td><td>0.0096</td><td>0.0214</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Input $n _ { \mathrm { e f f } }$ . For Mat´ern-5/2, the empirical coverage of our method dips slightly to 0.913. However, the theoretical benchmark Oracle i.i.d. (which evaluates risk on genuinely independent target samples) drops to an identica 0.917. This confirms that this mild degradation reflects the inherent sample complexity of evaluating small samples on this specific manifold, rather than a failure of the efective sample size approximation.

Even on the extremely rough, nowhere-diferentiable Mat´ern- $1 / 2$ process, our misspecified RBF formula maintains robust valid coverage (0.993) while shrinking the confidence radius by approximately 27% relative to the raw input correction.

Table 11: Kernel Type Ablation (Roughness Stress Test) on Target Risk. Asterisks (<sup>∗</sup>) denote significant empirical undercoverage below the nominal 0.95 level $\left( < 0 . 9 2 5 \right)$ . GP $n _ { \mathrm { e f f } }$ maintains valid interval coverage even when the true generating process is non-diferentiable, consistently outperforming Input $n _ { \mathrm { e f f } }$ in Winkler Score (WS). Note that the slight undercoverage on Mat´ern-5/2 is matched by the Oracle i.i.d. benchmark, indicating inherent finite-sample variance rather than a failure of the GP correction.
<table><tr><td>Data Kernel</td><td>Method</td><td>Coverage</td><td>Radius</td><td>WS↓</td></tr><tr><td rowspan="4">RBF (Smooth)</td><td>Unweighted</td><td>0.110*</td><td>0.0022</td><td>0.113</td></tr><tr><td>Nominal n</td><td>0.903*</td><td>0.0076</td><td>0.023</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.993</td><td>0.0206</td><td>0.042</td></tr><tr><td>GP (Ours)  $\mathbf { \delta } _ { n _ { \mathrm { e f f } } }$ </td><td>0.987</td><td>0.0147</td><td>0.031</td></tr><tr><td rowspan="4">Matérn-5/2</td><td>Unweighted</td><td>0.623*</td><td>0.0019</td><td>0.019</td></tr><tr><td>Nominal n</td><td>0.850*</td><td>0.0059</td><td>0.029</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.960</td><td>0.0160</td><td>0.036</td></tr><tr><td>GP  $\mathbf { \nabla } _ { n _ { \mathrm { e f f } } }$  (Ours) Oracle i.i.d.</td><td>0.913* 0.917*</td><td>0.0095 0.0062</td><td>0.028 0.018</td></tr><tr><td rowspan="4">Matérn-3/2</td><td>Unweighted</td><td>0.080*</td><td>0.0028</td><td>0.164</td></tr><tr><td>Nominal n</td><td>0.900*</td><td>0.0065</td><td>0.020</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>1.000</td><td>0.0175</td><td>0.035</td></tr><tr><td>GP (Ours)  $\mathbf { \delta } _ { n _ { \mathrm { e f f } } }$ </td><td>0.990</td><td>0.0128</td><td>0.026</td></tr><tr><td rowspan="4">Matérn-1/2 (Rough)</td><td>Unweighted</td><td>0.057*</td><td>0.0023</td><td>0.132</td></tr><tr><td>Nominal n</td><td>0.923*</td><td>0.0068</td><td>0.019</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>1.000</td><td>0.0184</td><td>0.037</td></tr><tr><td>GP (Ours)  $\mathbf { \delta } _ { n _ { \mathrm { e f f } } }$ </td><td>0.993</td><td>0.0134</td><td>0.027</td></tr></table>

## G.7 Robustness to Non-Stationary Covariates

A core structural assumption in standard mixing theory is the strict stationarity of the covariate sampling process $\mathbf { X } _ { t } .$ In realistic ecological or spatio-temporal monitoring, however, the efective correlation length $\ell _ { x } ( t )$ frequently fluctuates due to non-uniform sampling velocity, terrain transitions, or seasonal variations.

To stress-test our framework against structural non-stationarity, we evaluate KMM over covariate sequences generated via a non-stationary Gibbs covariance kernel (Gibbs, 1997). The Gibbs kernel generalizes the stationary RBF covariance by allowing the local correlation length $\ell _ { x } ( t )$ to vary continuously across sequence indices:

$$
\operatorname { C o v } ( X _ { s } , X _ { t } ) = \sigma _ { 0 } ^ { 2 } { \sqrt { \frac { 2 \ell _ { x } ( s ) \ell _ { x } ( t ) } { \ell _ { x } ( s ) ^ { 2 } + \ell _ { x } ( t ) ^ { 2 } } } } \exp \left( - { \frac { | s - t | ^ { 2 } } { \ell _ { x } ( s ) ^ { 2 } + \ell _ { x } ( t ) ^ { 2 } } } \right) .\tag{58}
$$

We evaluate three sampling regimes with sample size $n _ { \mathrm { t r } } = 1 2 8 \colon$

1. Stationary Baseline: Fixed persistence length $\ell _ { x } ( t ) \equiv 2 . 5$

2. Abrupt Switch: A sharp structural regime break where the sequence abruptly shifts from dense oversampling $( \ell _ { x } = 0 . 5$ for $t \leq n _ { \mathrm { t r } } / 2 )$ to long-range persistent dependence $( \ell _ { x } = 4 . 5$ for $t > n _ { \mathrm { t r } } / 2 )$ , with sequence average $\bar { \ell } _ { x } = 2 . 5$

3. Smooth Wave: A continuous harmonic oscillation simulating periodic variations in sampling density: $\ell _ { x } ( t ) = 2 . 5 + 2 . 0 \sin ( 2 \pi t / n _ { \mathrm { t r } } )$ , with average $\bar { \ell } _ { x } = 2 . 5$

Throughout all non-stationary trials, our estimator deliberately remains blind to the time-varying profile $\ell _ { x } ( t )$ simply plugging the sequence average $\bar { \ell } _ { x } = 2 . 5$ into the stationary $\mathrm { G P } ~ n _ { \mathrm { e f f } }$ formula.

As reported in Table 12, because KMM evaluates risk as a trajectory-wide integral, local fluctuations in dependence length are safely bounded by the sequence-averaged length scale. While Nominal n systematically under-covers on Target Risk across all configurations (collapsing to $0 . 8 8 0 ^ { * }$ under abrupt switches), our GP $n _ { \mathrm { e f f } }$ maintains robust valid coverage $\left( \geq 0 . 9 8 3 \right)$ across both stationary and non-stationary profiles. Furthermore, GP $n _ { \mathrm { e f f } }$ achieves a 17.5% tighter confidence radius and superior Winkler Scores compared to the overly conservative Input $n _ { \mathrm { e f f } }$ , confirming that the proposed correction remains safe even when strict stationarity is violated.

Table 12: Robustness to Non-Stationary Covariates on Target Risk. Asterisks (<sup>∗</sup>) indicate significant empirical undercoverage below the nominal 0.95 level $\left( < 0 . 9 2 5 \right)$ ). Covariates are generated via a non-stationary Gibbs kernel with mean dependence $\bar { \ell } _ { x } \ = \ 2 . 5$ . Plugging the trajectory-average correlation length into our stationary formula maintains valid coverage (≥ 0.983) and yields the lowest Winkler Score (WS) among valid estimators across all non-stationary regimes.
<table><tr><td>Dependence Profile</td><td>Method</td><td>Coverage</td><td>Radius</td><td>WS↓</td></tr><tr><td rowspan="4">Stationary</td><td>Unweighted</td><td>0.700*</td><td>0.0020</td><td>0.013</td></tr><tr><td>Nominal n</td><td>0.900*</td><td>0.0048</td><td>0.014</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.993</td><td>0.0120</td><td>0.024</td></tr><tr><td>GP (Ours)  $\mathbf { \delta } _ { n _ { \mathrm { e f f } } }$ </td><td>0.990</td><td>0.0099</td><td>0.020</td></tr><tr><td rowspan="4">Abrupt Switch</td><td>Unweighted</td><td>0.717*</td><td>0.0020</td><td>0.014</td></tr><tr><td>Nominal n</td><td>0.880*</td><td>0.0046</td><td>0.014</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.997</td><td>0.0114</td><td>0.023</td></tr><tr><td>GP (Ours)  $\mathbf { \delta } _ { n _ { \mathrm { e f f } } }$ </td><td>0.990</td><td>0.0094</td><td>0.019</td></tr><tr><td rowspan="4">Smooth Wave</td><td>Unweighted</td><td>0.750*</td><td>0.0020</td><td>0.011</td></tr><tr><td>Nominal n</td><td>0.893*</td><td>0.0046</td><td>0.016</td></tr><tr><td>Input  $n _ { \mathrm { e f f } }$ </td><td>0.997</td><td>0.0115</td><td>0.023</td></tr><tr><td>GP (Ours)  $\mathbf { \nabla } \mathbf { n } _ { \mathbf { e f f } }$ </td><td>0.983</td><td>0.0095</td><td>0.020</td></tr></table>

## H REAL DATA EXPERIMENTS DETAILS

## H.1 Semi-Synthetic Overlap Stress Test on California Housing

The controlled synthetic experiments isolate dependence and evaluate the dependence-induced efective sample size developed in the main text. We use a separate semi-synthetic experiment to isolate a diferent component of the problem: deterioration of source–target overlap. Real covariates and geographic coordinates are retained from the California Housing dataset, while outcomes are generated from a known conditional mechanism shared by source and target. This allows overlap to be varied without simultaneously introducing conditional shift.

Let $q _ { 0 } ( x )$ denote the empirical distribution over the California Housing covariates. We define a standardized overlap score:

$$
g ( x ) = \mathrm { s t d } ( 0 . 7 5 X _ { \mathrm { M e d I n c } } + 0 . 4 5 X _ { \mathrm { L a t i t u d e } } + 0 . 2 5 X _ { \mathrm { A v e R o o m s } } ) ,
$$

which is truncated to [−3, 3]. Source and target distributions are constructed by symmetric exponential tilting:

$$
q _ { S } ^ { ( \gamma ) } ( x ) \propto q _ { 0 } ( x ) \exp \{ - \gamma g ( x ) / 2 \} , \qquad q _ { T } ^ { ( \gamma ) } ( x ) \propto q _ { 0 } ( x ) \exp \{ + \gamma g ( x ) / 2 \} .\tag{59}
$$

Consequently,

$$
\frac { q _ { T } ^ { ( \gamma ) } ( x ) } { q _ { S } ^ { ( \gamma ) } ( x ) } \propto \exp \{ \gamma g ( x ) \} ,\tag{60}
$$

so the population density ratio is analytically known.

We consider shift severities $\gamma \in \{ 0 , 0 . 5 , 1 , 1 . 5 , 2 , 3 \}$ $\mathrm { A t } ~ \gamma = 0$ , source and target coincide; increasing γ progressively reduces overlap. For each value of γ, we draw $n _ { S } = n _ { T } = 2 5 6$ observations independently with replacement from the corresponding tilted empirical distributions. Results are averaged over 100 repetitions.

Let the standardized California Housing predictors be denoted by X. We construct:

$$
\mu ( X ) = 0 . 2 + 0 . 6 \sigma \{ \eta ( X ) \} ,
$$

where $\sigma$ is the logistic function and

$$
\begin{array} { r l } & { \eta ( X ) = 0 . 7 5 X _ { \mathrm { M e d I n c } } + 0 . 2 5 \sin ( X _ { \mathrm { H o u s e A g e } } ) + 0 . 2 0 \operatorname { t a n h } ( X _ { \mathrm { A v e R o o m s } } ) } \\ & { \qquad - 0 . 1 5 \operatorname { t a n h } ( X _ { \mathrm { A v e O c u p } } ) + 0 . 2 0 \sin ( X _ { \mathrm { L a t i t u d e } } ) + 0 . 2 0 \cos ( X _ { \mathrm { L o n g i t u d e } } ) } \\ & { \qquad + 0 . 1 2 X _ { \mathrm { M e d I n c } } X _ { \mathrm { A v e R o o m s } } . } \end{array}\tag{61}
$$

The observed response is

$$
\begin{array} { r } { Y ( s ) = \mu ( X ( s ) ) + \varepsilon ( s ) , } \end{array}
$$

where $\varepsilon ( s )$ is a bounded spatial random field generated using random Fourier features:

$$
\varepsilon ( s ) = 0 . 1 2 \operatorname { t a n h } \{ Z ( s ) \} .
$$

The Fourier coeficients are independently sampled from symmetric zero-mean Gaussian distributions in each repetition. Hence the residual process has zero expectation across realizations and is spatially correlated within a realization. Since $\mu ( X ) \in [ 0 . 2 , 0 . 8 ]$ and $\varepsilon ( s ) \in [ - 0 . 1 2 , 0 . 1 2 ]$ , the response remains bounded in [0.08, 0.92] by construction. Across the final experiment, the realized response range was [0.091, 0.917].

The same conditional mechanism is used for source and target for all γ. Thus, the experiment changes covariate overlap while keeping the conditional response mechanism invariant. The target estimand is the population target mean:

$$
\mu _ { T } ^ { ( \gamma ) } = \sum _ { x } q _ { T } ^ { ( \gamma ) } ( x ) \mu ( x ) ,\tag{62}
$$

which is known from the semi-synthetic construction.

Estimators and KMM Configuration We compare three estimators: the unweighted source mean, self normalized oracle importance weighting using the known ratio in Eq. (60), and KMM using source and unlabeled target covariates. The KMM weights satisfy:

$$
0 \leq w _ { i } \leq B , \qquad \sum _ { i = 1 } ^ { n _ { S } } w _ { i } = n _ { S } ,
$$

with clipping bound $B = 1 0$ . We use an RBF kernel with bandwidth $h = 8$ and ridge regularization $\lambda = 1 0 ^ { - 2 }$

These KMM hyperparameters were selected in a separate no-shift sanity check at $\gamma = 0$ , using only covariate balance and weight stability and without using outcomes. We evaluated $h \in \{ 1 , 2 , 4 , 8 \}$ and $\lambda \in \{ 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \}$ The selected configuration gave mean weight ESS 215.9 out of 256, mean maximum weight $3 . 1 6 ,$ no weights at the upper bound, and reduced empirical MMD<sup>2</sup> from $5 . 8 1 \times 1 0 ^ { - 4 }$ to $2 . 1 \times 1 0 ^ { - 5 }$ in the sanity experiment.

Overlap and Weight Diagnostics We quantify empirical source–target discrepancy using $\mathrm { M M D ^ { 2 } }$ and weight concentration using:

$$
\mathrm { E S S } _ { w } = \frac { \left( \sum _ { i } w _ { i } \right) ^ { 2 } } { \sum _ { i } w _ { i } ^ { 2 } } .\tag{63}
$$

This quantity is a diagnostic of importance-weight concentration and is distinct from the dependence-induced efective sample size $n _ { \mathrm { e f f } }$ central to our theoretical results.

![](images/4b511ca816e3ea98c7ffb6c4f1cc5b0dc016cea9689bfb75049e607cc2a2642d.jpg)

(b) Estimation error  
![](images/8d23a9565c5a10606e2b89a379fb64ff7c7b829277cfaa9816bad78bd83eea5f.jpg)  
Figure 5: Controlled Overlap Stress Test on California Housing Covariates. Increasing γ progressively separates source and target distributions. Left: Oracle and KMM weight ESS decrease as overlap deteriorates. Right: KMM controls target-mean error under moderate shift, but its bounded weights cannot fully reproduce the density ratio under severe overlap loss. Weight ESS measures weight concentration and is distinct from the dependence-induced $n _ { \mathrm { e f f } }$ . Error bars show one standard error across 100 repetitions.

Table 13: Semi-Synthetic Overlap Stress Test on California Housing Covariates. Increasing γ separates source and target distributions. ESS denotes weight efective sample size $( \mathrm { E S S } _ { w } )$ , not the dependence-induced $n _ { \mathrm { e f f } }$ . MAE is computed over 100 repetitions.
<table><tr><td>γ</td><td> $\bf { M M D ^ { 2 } }$ </td><td>Oracle ESS</td><td>KMM ESS</td><td>Oracle Max w</td><td>Unweighted MAE</td><td>Oracle MAE</td><td>KMM MAE</td></tr><tr><td>0.0</td><td>0.0007</td><td>256.0</td><td>218.3</td><td>1.0</td><td>0.0380</td><td>0.0380</td><td>0.0381</td></tr><tr><td>0.5</td><td>0.0048</td><td>198.3</td><td>154.8</td><td>4.3</td><td>0.0461</td><td>0.0359</td><td>0.0355</td></tr><tr><td>1.0</td><td>0.0176</td><td>87.0</td><td>71.5</td><td>15.5</td><td>0.0789</td><td>0.0371</td><td>0.0354</td></tr><tr><td>1.5</td><td>0.0393</td><td>38.7</td><td>31.5</td><td>36.6</td><td>0.1203</td><td>0.0465</td><td>0.0365</td></tr><tr><td>2.0</td><td>0.0704</td><td>22.1</td><td>26.9</td><td>58.9</td><td>0.1634</td><td>0.0653</td><td>0.0460</td></tr><tr><td>3.0</td><td>0.1549</td><td>9.8</td><td>26.3</td><td>92.8</td><td>0.2403</td><td>0.1013</td><td>0.1260</td></tr></table>

Results and Discussion The construction produces the intended progressive deterioration in overlap (Figure 5 and Table 13). Mean empirical $\mathrm { M M D ^ { 2 } }$ increases from 0.0007 at $\gamma = 0$ to 0.1549 at $\gamma = 3$ , while oracle weight ESS falls from 256 to 9.8. KMM weight ESS similarly decreases from 218.3 to 26.3. Across all replicates, γ is strongly positively associated with $\mathrm { M M D ^ { 2 } }$ (Spearman $\rho = 0 . 9 8 6 )$ and negatively associated with both oracle and KMM weight ESS $( \rho = - 0 . 9 3 7$ and −0.975, respectively).

At $\gamma = 0$ , all three estimators have essentially identical error: MAE is 0.0380 for the unweighted and oracle estimators and 0.0381 for KMM. This confirms that weighting does not introduce unnecessary distortion when source and target coincide.

As overlap deteriorates, the unweighted estimator rapidly degrades, whereas reweighting substantially reduces target-mean error. $\mathrm { A t } \ \gamma = 2$ , for example, MAE is 0.1634 without weighting, 0.0653 with oracle weighting, and 0.0460 with KMM, yielding an MAE reduction of approximately 72% relative to the unweighted baseline.

The severe-overlap regime illustrates the limitation of interpreting weight ESS as a measure of successful transfer. $\mathrm { A t } \ \gamma = 3 .$ , the mean maximum oracle weight is 92.8 and oracle weight ESS falls to 9.8. KMM is constrained by $B = 1 0$ and consequently retains a larger ESS of 26.3. Nevertheless, its MAE is 0.1260, compared with 0.1013 for oracle weighting. The larger KMM ESS reflects more stable weight concentration, not better approximation of the target density ratio. Bounding the weights limits variance amplification but can leave residual reweighting bias when adequate overlap would require more extreme weights. Consistent with this pattern, KMM weight ESS is negatively associated with KMM absolute error across replicates (Spearman $\rho = - 0 . 3 9 1 )$ , but this association does not imply that larger ESS alone guarantees smaller target error.

The Role of Clipping Bound B in Theorem 1 The sweep exposes a direct balance–stability trade-of. When $\gamma = 0 , B ^ { * } = 1$ and increasing the admissible weight bound has essentially no efect. $\mathrm { A t } ~ \gamma = 1$ , the empirical results similarly stabilize once B approaches $B ^ { * } \approx 1 9 . 8 5$

Under stronger overlap loss, however, the required population ratio becomes much larger than the weight bounds considered. $\mathrm { A t } \gamma = 2$ , increasing B from 5 to 20 reduces postweighting $\mathrm { M M D ^ { 2 } }$ from 0.0162 to 0.0047 and reduces bias magnitude from 0.0611 to 0.0013. This improvement in balance is accompanied by a decrease in weight ESS from 52.1 to 15.1. Increasing B further provides little additional RMSE improvement, illustrating the finite-sample bias–variance trade-of induced by the weight constraint (Figure 6).

The same phenomenon is more pronounced at $\gamma = 3$ , where $B ^ { * } = 4 7 3 9 . 9$ and even $B = 1 0 0$ represents only approximately 2.1% of the population ratio bound. Increasing B from 5 to 100 reduces post-weighting $\mathrm { M M D ^ { 2 } }$ from 0.0768 to 0.0288 and bias magnitude from 0.1533 to 0.0465, while weight ESS falls from 51.8 to 4.0. Thus, a larger weight ESS does not by itself indicate successful target correction: restrictive bounds can retain a comparatively large ESS precisely because they prevent the weights required to remove source–target imbalance.

These results clarify that the bounded-density-ratio assumption in Theorem 1 is substantive rather than merely technical. When $B < B ^ { * } ( \gamma )$ , the admissible KMM weights cannot represent the population density ratio, leaving residual RKHS imbalance. Relaxing the bound reduces this imbalance and its associated bias, but simultaneously increases weight concentration and sampling variability.

![](images/4a8ee2d4c503cc1615a56fe3477da6315080b14d47e9eee37f780a10a78266e7.jpg)

![](images/96db473b24bf3ce9d4b6d53b151765a015fd7433b1b4c3faba97b57482e3850f.jpg)

![](images/a59b39dc23643dc43b5933500fac9123c6b09504060c24cbb2ebd9f02a49b21d.jpg)  
Figure 6: Efect of the KMM clipping bound B under deteriorating overlap. Increasing B allows KMM to achieve better source–target balance and reduces residual estimation bias, but at the cost of increasingly concentrated weights and lower weight ESS. When the population density-ratio bound $B ^ { * }$ is moderate $( \gamma =$ 1), the results stabilize once B reaches B<sup>∗</sup>.Under severe overlap loss $( \gamma = 2 , 3 ) , B ^ { * }$ is much larger than the considered clipping levels, exposing a pronounced balance–stability trade-of. Weight ESS here measures weight concentration and is distinct from the dependence-induced $n _ { \mathrm { e f f } }$ used elsewhere in the paper.

## H.2 Three-Domain Real-Data Benchmark: Point Estimation

In Section 4.2, our primary focus is on confidence interval calibration under data dependence. For completeness, Table 14 reports point estimation metrics across the three real-data monitoring domains: NOAA weather stations, $\mathrm { P M } _ { 2 . 5 }$ monitoring stations, and EPA ozone monitoring stations.

We compare the unweighted source mean, logistic importance weighting (Log-IW), and KMM for two distinct estimands:

1. Target Response Mean: $\mu _ { T } = \operatorname { \mathbb { E } } _ { T } [ Y ] .$

2. Prediction Squared-Error Risk: $R _ { T } ( f ) = \mathbb { E } _ { T } [ ( Y - f ( X ) ) ^ { 2 } ]$ for the frozen random-forest predictor $f .$

Datasets and response scaling. All responses are clipped and normalized to $[ 0 , 1 ] ;$ ; consequently, the reported errors are in normalized units, not physical units. NOAA uses daily station observations from 2025, with response $Y = \mathrm { c l i p } ( \mathrm { ( T M A X + 3 0 ) / 8 0 , 0 , 1 } )$ , where TMAX is measured in degrees Celsius. Predictors are minimum temperature, precipitation, and sine/cosine encodings of day of year. The Beijing $\mathrm { P M } _ { 2 . 5 }$ dataset uses $Y =$ clip $\mathrm { \ P M _ { 2 . 5 } / 5 0 0 , 0 , 1 ) }$ , with concentration measured in $\mu \mathrm { { g } / m ^ { 3 } }$ . Predictors are temperature, pressure, dew point, rainfall, wind speed, $\mathrm { S O _ { 2 } , \ N O _ { 2 } , \ C O , \ O _ { 3 } . }$ , and sine/cosine encodings of day of year. EPA Ozone uses 2022 daily measurements of parameter 44201: the daily maximum eight-hour ozone concentration, normalized as

$Y = \mathrm { { c l i p } ( O _ { 3 } / 0 . 2 , 0 , 1 ) }$ , with ozone in ppm. Temperature and wind measurements are joined by monitoring-site identifier and calendar date; predictors are temperature, wind speed, and sine/cosine encodings of day of year.

Predictor and weighting methods. The random forest is trained separately from the evaluation tasks and frozen thereafter. It uses 100 trees, a minimum leaf size of 10, all predictors as candidate features, and random seed 20261005. Missing predictor values are handled by median imputation in the prediction pipeline, and predictions are clipped to [0, 1]. KMM uses an RBF kernel with a pooled source–target, covariate-only standardization and median-distance bandwidth. Weights satisfy $0 ~ \leq ~ w _ { i } ~ \leq ~ 1 0$ and $\begin{array} { r } { \sum _ { i } w _ { i } = n . } \end{array}$ . Logistic importance weighting estimates density ratios by classifying source versus target covariates; ratios are uppercapped at 10 and self-normalized for point estimation. Target outcomes are not used to train the predictor or fit the importance weights.

Point-estimation versus confidence interval tasks. The point-estimation results in Table 14 utilize a fundamentally diferent task construction compared to the rolling-window interval evaluation detailed later. Specifically, point estimation is computed over full-series or full-quarter transfers: 141 NOAA transfer tasks, 48 $\mathrm { P M } _ { 2 . 5 }$ station–quarter tasks, and 50 EPA Ozone full-series transfer tasks. For $\mathrm { P M } _ { 2 . 5 }$ , the predictor is trained on 2013–2014 data, with source observations drawn from 2015 and target observations from the corresponding quarters of 2016. In contrast, the confidence interval evaluation (described in Appendix H.3) relies on a strict rolling-window protocol over 250 independent trials of fixed length n. Therefore, the point-estimation errors reported here reflect the overall performance on broad transfer tasks and do not directly correspond to the specific 250 finite-sample segments utilized in the interval calibration experiments.

Evaluation reference. Population target means and risks are unknown. For evaluation, they are replaced by the empirical response mean and empirical squared-error risk on the held-out target sample. MAE is the unweighted average of task-level absolute estimation errors. Target outcomes are accessed only for final scoring. Tasks can share stations and evaluation periods, so task counts do not represent independent Monte Carlo repetitions.

Table 14: Point Estimation on the Three-Domain Real-Data Benchmark. Mean Absolute Error (MAE) is reported for target response mean $\mu _ { T } = \mathbb { E } _ { T } [ Y ]$ and squared-error risk $R _ { T } ( f ) = \mathbb { E } _ { T } [ ( Y - f ( X ) ) ^ { 2 } ]$ . Bold indicates the lowest MAE within each dataset–estimand pair.
<table><tr><td>Data</td><td>Estimand</td><td>Unweighted</td><td>Log-IW</td><td>KMM</td></tr><tr><td>NOAA</td><td>Target Mean (µT)</td><td>0.0503</td><td>0.0389</td><td>0.0379</td></tr><tr><td> $\mathrm { P M } _ { 2 . 5 }$ </td><td>Target Mean (µT)</td><td>0.0208</td><td>0.0270</td><td>0.0155</td></tr><tr><td>EPA Ozone</td><td>Target Mean (µμT)</td><td>0.0290</td><td>0.0245</td><td>0.0239</td></tr><tr><td>NOAA</td><td>Target Risk  $( R _ { T } ( f ) )$ </td><td>0.00209</td><td>0.00205</td><td>0.00213</td></tr><tr><td>PM2.5</td><td>Target Risk  $( R _ { T } ( f ) )$ </td><td>0.00214</td><td>0.00205</td><td>0.00244</td></tr><tr><td>EPA Ozone</td><td>Target Risk  $( R _ { T } ( f ) )$ </td><td>0.00137</td><td>0.00109</td><td>0.00120</td></tr></table>

Discussion of Point Estimation Results. As shown in Table 14, KMM achieves the lowest target-mean MAE across all three domains, reducing the unweighted error from 0.0503 to 0.0379 for NOAA, from 0.0208 to 0.0155 for $\mathrm { P M } _ { 2 . 5 }$ , and from 0.0290 to 0.0239 for EPA Ozone. However, the pattern difers for prediction risk: Logistic-IW achieves the lowest risk MAE across all three domains, while KMM does not consistently outperform the unweighted baseline. These results highlight that achieving improved covariate balance does not guarantee uniform improvements in point estimation across all observational transfer tasks.

## H.3 Three-Domain Real-Data Benchmark: Confidence Interval Protocol

The interval calibration experiment addresses a distinct question: how accounting for dependent data generation alters interval coverage and width around a fixed KMM estimate. GP-based intervals improve empirical coverage relative to nominal intervals, although coverage remains below the nominal 95% level for NOAA and EPA Ozone. We therefore interpret these results as a real-data stress test of the dependence correction rather than as verification of the theorem’s assumptions or finite-sample coverage guarantees.

Rolling-Window Evaluation Protocol. For the confidence interval calibration experiment in Section 4.2, we adopted a rigorous Monte Carlo rolling-window protocol to ensure statistically robust coverage estimates. For NOAA and EPA Ozone, the data is partitioned into disjoint sets of predictor-training stations, source stations, and held-out target stations (fixed using data-split seed 20261005).

Over 250 independent Monte Carlo repetitions (initialized with random seed 42), the evaluation procedure randomly selects a source station and extracts a contiguous time-series window of specific size n. The unlabelled target sample is constructed by uniformly sampling up to 1000 observations from the respective target station pool. The KMM weights are optimized on this pair, and the empirical autocorrelations $r _ { k }$ alongside the functional length-scale h are estimated directly from the source window. All confidence intervals are completely fixed before the target outcomes are revealed for final scoring. These are conditional plug-in intervals that treat fitted weights as fixed; they do not separately propagate weight estimation variance or sampling variability in the finite-target reference.

Real-Data Hyperparameter Selection and Configurations. Because the target mean response Y and the target loss surface L exhibit diferent functional smoothness profiles, their optimal balance-variance trade-ofs occur at diferent KMM clipping bounds B. To rigorously benchmark the estimators and report the tightest valid intervals, we evaluated a comprehensive grid over the source sample size n ∈ {96, 128, 192, 256} and clipping bound $B \in \{ 1 0 . 0 , 1 5 . 0 , 2 0 . 0 \}$

The representative configurations reported in the main text (Table 4) were selected to maximize valid coverage while minimizing the Winkler Score. Given the presence of unobserved confounders and structural nonstationarity in real-world geospatial data, we relaxed the strict validity threshold slightly to 0.920. The optimal configurations are as follows:

$\mathbf { P M _ { 2 . 5 } }$ : Target Mean was evaluated at $n = 1 2 8 , B = 2 0 . 0 ;$ Target Risk at $n = 1 2 8 , B = 1 5 . 0$

• EPA Ozone: Target Mean was evaluated at $n = 2 5 6 , B = 1 5 . 0 ;$ Target Risk at $n = 2 5 6 , B = 2 0 . 0$

• NOAA: Both estimands required the maximum sample capacity due to extreme spatial shift, evaluated at $n = 2 5 6 , B = 2 0 . 0$