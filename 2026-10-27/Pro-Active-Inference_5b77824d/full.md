## Pro-Active Inference

Skyler Wu skylerw@stanford.edu   
Department of Statistics   
Stanford University   
Yash Nair yashnair@stanford.edu   
Department of Statistics   
Stanford University   
Antoine Maechler maechler@stanford.edu   
Department of Management Science and Engineering   
Stanford University   
Emmanuel J. Candès candes@stanford.edu   
Department of Statistics   
Stanford University

## Abstract

Estimating a population mean is fundamental across scientific applications, but observing every outcome can be prohibitively expensive. Often, covariates and machine learning (ML) predictions are available for the full population, even when only a small subset of outcomes can be observed. How can we leverage this information to proactively select observations to label, update ML predictions as labels arrive, and construct statistically valid confidence intervals? We develop a general, without-replacement formulation of Pro-Active Inference (PAI), extending our earlier with-replacement approach and building on Active Statistical Inference. Rather than making fixed-order label-skip decisions, PAI proactively selects which observation to label next from the remaining covariate pool while accommodating online ML updates. We establish PAI’s finite-sample unbiasedness and asymptotic validity, develop an improved finite-sample variance estimator, and extend PAI to superpopulation inference without Bonferroni corrections or α-splitting. We characterize PAI’s potential efficiency gains over Batch Active Inference in a quasi-oracle setting, highlighting the role of prediction-residual imbalance. Across eight diverse datasets, PAI achieves greater sampling efficiency than Sequential Active Inference in the vast majority of tested settings with near-nominal statistical coverage, using the same ML predictor and uncertainty score. However, comparisons with simpler baselines such as Prediction-Powered Inference and uniform sampling, as well as the closely related Prediction-Powered Active Testing, reveal when additional adaptivity helps or hurts statistical efficiency. Through interpretable, theory-motivated diagnostics and frozen-ML ablations, we investigate when and why PAI gains or loses efficiency, highlighting the initial residual-estimation problem and subsequent online learning dynamics.

## 1 Introduction

Estimating a population mean—such as disease prevalence, the proportion of households below the poverty line, or the fraction of files containing malware—is fundamental across scientific and policy applications. However, collecting ground-truth outcomes can be prohibitively expensive, often requiring human expert annotation, specialized testing, or additional surveys. In many modern settings, however, covariates and machine learning (ML) predictions are readily available for the entire population, even when ground-truth outcomes can only be obtained for a small subset. How can we fully leverage this information to decide which outcomes to collect and efficiently estimate a population mean, with valid confidence intervals?

As a concrete example, consider a policymaker seeking to estimate the proportion of households living below the poverty line following a major economic disruption. Suppose demographic information is available for many households, but determining their current economic circumstances requires costly surveys. An ML model trained on historical data could predict each household’s probability of poverty, although its predictions may be imperfect, particularly under changing economic conditions. Given a limited survey budget, the policymaker could survey households uniformly at random, obtaining a statistically valid sample mean but potentially leaving efficiency gains from ML predictions untapped. Alternatively, they could preferentially survey households whose poverty status the model is most uncertain about, update the model as new responses arrive, and adapt subsequent sampling accordingly. However, simply averaging the adaptively collected responses can introduce selection bias, while relying on ML predictions alone risks bias from prediction errors.

The challenge, therefore, is to leverage the full covariate pool and progressively acquired responses to construct efficient estimators and confidence intervals under adaptive, nonuniform sampling.

Problem setup Formally, suppose we observe covariates $x _ { 1 } , \ldots , x _ { N }$ with corresponding unobserved responses $y _ { 1 } , \ldots , y _ { N } ,$ each of which may be collected at a (potentially substantial) cost. Given a labeling budget n $\ll N _ { \ast }$ , we seek to sequentially label data points, so as to construct an estimator and confidence interval, by leveraging, at each timestep, the full covariate pool, previously observed responses, and a black-box ML model that can be updated as new responses arrive. Specifically, we aim to use ML predictions both to adaptively select which responses to collect and to improve the resulting estimator through regression adjustment. Our primary goal is to construct an efficient estimator and statistically valid confidence interval for the finite-population mean

$$
\theta : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } y _ { i } ,\tag{1}
$$

treating the population as fixed and the sampling process alone as random. We also consider inference for the superpopulation mean $\mathbb { E } [ Y ]$ when the $( x _ { i } , y _ { i } )$ are assumed to have been drawn i.i.d. from an underlying distribution.

Positioning A natural starting point for the above problem is Active Statistical Inference (ASI) (Zrnic & Candès, 2024), which leverages ML predictions to guide label acquisition and construct statistically valid estimators and confidence intervals. In its sequential formulation (SAI), observations are processed in a fixed, predetermined order, with the ML model potentially updated as labels are acquired. Crucially, at round $t , { \mathrm { S A I } }$ can only make a Bernoulli decision to either acquire or skip the response $y _ { t }$ associated with the current observation $x _ { t }$ . Once an observation is skipped, its response cannot be acquired later. When the full covariate pool is available in advance, however, why restrict each round to a label-skip decision about the current observation, rather than proactively selecting which observation to label next from the entire remaining pool?

Motivated by this question, our earlier work (Wu et al., 2026) introduced Pro-Active Inference (PAI) in the context of LLM evaluation, combining proactive acquisition with online-updated ML predictions and regression adjustment. However, this initial formulation sampled with replacement, potentially revisiting previously labeled observations and wasting sampling opportunities. Extending PAI to sampling without replacement, however, requires more than modifying the acquisition policy: the estimator and its underlying martingale-based analysis must also be reformulated, as naive modifications can violate conditional unbiasedness and lead to severe confidence-interval miscoverage.

Our contributions In the present work, we therefore develop a general, without-replacement formulation of PAI, together with a reformulated estimator and corresponding martingale-based inferential guarantees.

We establish its finite-sample unbiasedness and asymptotic validity, develop an improved finite-sample variance estimator, and extend PAI to superpopulation inference without requiring Bonferroni corrections or α-splitting. Theoretically, we characterize PAI’s potential efficiency gains over Batch Active Inference (BAI) in a quasi-oracle setting, highlighting the role of prediction-residual imbalance. Empirically, through experiments across eight diverse testbeds, we investigate PAI’s statistical efficiency and confidence-interval coverage relative to existing methods. Beyond comparing performance, we develop interpretable diagnostics of residual imbalance, the relative difficulty of direct versus prediction-adjusted estimation, and the quality of online ML updates, together with targeted ablations, to understand when and why PAI gains or loses efficiency, and, more broadly, when ML-assisted statistical inference is beneficial.<sup>1</sup>

Towards the completion of this manuscript, we also became aware of the closely related works of Ashouritaklimi et al. (2026); Hamilton et al. (2026), which develop alternative approaches to adaptive sampling without replacement. While these approaches share important methodological features with PAI, they differ in their estimator constructions and uses of ML predictions. We discuss these and other related approaches in Section 2 and empirically compare PAI with the former in Section 6.

## 2 Related work

Prediction-powered inference A natural starting point for estimating a population mean is uniform sampling, which yields statistically valid inference without involving an ML model, but may leave substantial efficiency gains from available ML predictions untapped. Prediction-Powered Inference (PPI; Angelopoulos et al. (2023)) leverages ML predictions to impute unobserved outcomes and corrects the resulting estimation error using a smaller sample of observed labels. Subsequent extensions include PPI++ (Angelopoulos et al., 2026), which adapts the strength of the prediction-based adjustment to improve statistical efficiency, and cross-PPI (Zrnic & Candès, 2024), which allows the available labeled data to be used for both training ML predictors and constructing inference through cross-fitting. While these approaches primarily consider nonadaptively acquired labels, PAI additionally uses ML predictions to guide which outcomes to acquire, potentially updating both its regression adjustment and acquisition policy as new labels are observed.

Active statistical inference Our work builds on the Active Statistical Inference (ASI) framework of Zrnic & Candès (2024), which leverages ML predictions to guide label acquisition while constructing statistically valid estimators and confidence intervals for statistical estimands. In its sequential formulation (Sequential Active Inference; SAI), observations arrive in a stream, with covariates modeled as i.i.d. draws from a superpopulation, and the method makes Bernoulli label-skip decisions according to a potentially updated ML-based acquisition policy, subject to a prespecified labeling budget. Once an observation is skipped, however, its label cannot be acquired later. When the full covariate pool is available in advance, PAI instead proactively selects which observation to label next from the remaining unlabeled pool, using the full covariate information and previously acquired labels to update its ML-based regression adjustment and acquisition policy. Related developments include optimized and robust acquisition policies for ASI (Ma & Candès, 2026; Li et al., 2026), as well as applications to human annotation and evaluation of LLM outputs (Gligori´c et al., 2025; Angelopoulos et al., 2025).

In our earlier work (Wu et al., 2026), we introduced a with-replacement formulation of PAI for LLM eval uation, using sampling policies and regression adjustments based on an online-updated working Bayesian factor model. However, this formulation allowed previously labeled observations to be sampled again, potentially wasting part of the labeling budget on repeated queries. The present work instead samples exclusively from the remaining unlabeled observations, ensuring that each query acquires a new label and that the full budget is spent on distinct observations, while developing the corresponding estimator and inferential guarantees. Finally, Tolochinsky et al. (2026) also consider sampling without replacement in the context of best-model identification for LLMs, but use uniform sampling rather than an adaptive, MLguided acquisition policy.

Active testing and measurement PAI is also closely connected to a growing literature on efficient statistical inference in the presence of missing outcome data, particularly when labels are acquired adaptively. Most closely related is a line of work studying importance-weighted estimation under adaptive label acquisition (Farquhar et al., 2021; Kossen et al., 2021; Hamilton et al., 2026; Ashouritaklimi et al., 2026). With the exception of Hamilton et al. (2026), these works focus primarily on estimating the risk of a fixed ML model, but their core methods readily apply to our population mean estimation task outlined in Section 1. In particular, the active-testing literature (Farquhar et al., 2021; Kossen et al., 2021) develops unbiased estimators under adaptive sampling without replacement.

Building on this literature, one of the methods most closely related to PAI is Active Measurement (AM; Hamilton et al., 2026), which studies the estimation of finite-population quantities, such as bird counts, rather than the prediction risk of a fixed ML model. AM uses online-updated ML predictions to guide acquisition and develops alternative weighting schemes and variance estimators for combining importanceweighted estimates. In particular, AM generalizes the estimation framework underlying Farquhar et al. (2021)’s LURE estimator by considering alternative schemes for aggregating conditionally unbiased perround estimates. Indeed, at the level of the per-round estimator, AM coincides with PAI when our prediction-based regression adjustment is omitted. PAI instead incorporates online-updated ML predictions into both acquisition and regression adjustment, potentially improving sampling efficiency while preserving finite-sample unbiasedness.

Another closely related method is Prediction-Powered Active Testing (PPAT; Ashouritaklimi et al., 2026), which similarly builds on the active-testing literature, but additionally incorporates ML predictions directly into the estimator through a prediction-powered control variate. In all of their experiments, PPAT uses a fixed ML proxy for regression adjustment and a potentially updated surrogate model to guide acquisition, with a tunable control-variate coefficient. The authors further note that the control-variate proxy could also be updated as new labels are acquired, although they do not investigate this extension experimentally. In contrast to their experimental implementation, PAI directly incorporates online-updated ML predictions into its imputation and residual correction.

Like PAI, AM and PPAT construct their estimators by aggregating per-round importance-weighted estimates, which are conditionally unbiased under fixed tuning parameters. Under suitable regularity conditions, both approaches establish asymptotic Gaussian limiting distributions through martingale arguments, enabling the construction of asymptotically valid confidence intervals using corresponding variance estimators. However, PPAT aggregates its per-round estimates using nonuniform LURE-based weights (Farquhar et al., 2021), whereas PAI weights them equally.

Among these closely related approaches, we focus our empirical comparisons on PPAT, since its use of ML predictions in both acquisition and regression-adjusted estimation makes it a particularly direct methodological comparator to PAI.

## 3 Pro-Active Inference methodology

We first describe PAI’s adaptive sampling procedure, then develop estimators and confidence intervals for finite-population means, before extending our methodology to superpopulation inference.

## 3.1 Sampling setup and notation

Let $\mathcal { I } = ( 1 , \ldots , N )$ index the finite population, and let $O _ { t - 1 } \subseteq \tau$ denote the indices with observed responses before round $t ,$ with $O _ { 0 } = \emptyset$ . Given a labeling budget $n \ll N$ , at each round $t = 1 , \ldots , n ,$ , PAI constructs an acquisition distribution (or sampling policy) $q _ { t }$ over the remaining unlabeled indices $R _ { t - 1 } : = \mathbb { Z } \backslash O _ { t - 1 } ,$ , using the full covariate pool and all previously acquired responses. It then samples an index $I _ { t } \sim q _ { t } ,$ , observes its response $y _ { I _ { t } } ,$ , and updates $O _ { t } : = O _ { t - 1 } \cup \{ I _ { t } \}$ . We denote by ${ \hat { f } } ^ { ( t - 1 ) }$ the ML predictor available before round $t ,$ initialized at $\hat { f } ^ { ( 0 ) }$ and potentially updated after each newly acquired response. Formally, let $\mathcal { F } _ { t - 1 } : =$ $\sigma ( \{ I _ { s } , y _ { I _ { s } } \} _ { s = 1 } ^ { t - 1 } )$ . We require $\hat { f } ^ { ( t - 1 ) }$ and $q _ { t }$ to be $\mathcal { F } _ { t - 1 } .$ -measurable: depending only on past queries/labels.

The sampling policy $q _ { t }$ may depend on both the full covariate pool and previously acquired labels, but not on the unobserved responses. A natural strategy is to assign higher sampling probabilities to observations where the ML predictor is most uncertain. For binary outcomes, writing $p _ { t } ( x ) : = { \hat { f } } ^ { ( t - 1 ) } ( x )$ , common uncertainty scores include $u _ { t } ( x ) = 2 \operatorname* { m i n } ( p _ { t } ( x ) , 1 - p _ { t } ( x ) )$ (Zrnic & Candès, 2024) and $u _ { t } ( x ) = \sqrt { p _ { t } ( x ) ( 1 - p _ { t } ( x ) ) }$ the latter motivated by Neyman allocation. Our experiments use the former, with uniform mixing to ensure positive sampling probabilities (Section 5), although our theoretical framework supports general adaptive sampling policies satisfying the regularity conditions below.

## 3.2 Pro-Active Inference estimator for finite-population estimands

After adaptively collecting n labels without replacement, we propose the following Pro-Active Inference (PAI) estimator of the finite-population mean:

$$
\hat { \theta } _ { n } : = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \phi _ { t } , \quad \phi _ { t } : = \frac { 1 } { N } \left( \sum _ { i \in O _ { t - 1 } } y _ { i } + \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) + \frac { 1 } { N } \frac { y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) } { q _ { t } ( I _ { t } ) } .\tag{2}
$$

Intuitively, the first term in $\phi _ { t }$ represents our current plug-in estimate of θ—using observed labels where available and imputing ML predictions for the remaining unobserved labels—while the second term provides an importance-weighted bias correction for prediction errors. In Theorem 1, we show that $\hat { \theta } _ { n }$ is unbiased for θ at every finite n and derive its asymptotic distribution using a martingale central limit theorem under mild regularity conditions.

Theorem 1. Consider a triangular array of problems (indexed by m) such that $n _ { m } \uparrow \infty , N _ { m } \uparrow \infty ,$ , and $n _ { m } / N _ { m } $ $\rho \in [ 0 , 1 - \epsilon ]$ for some $\epsilon > 0 . ^ { 2 }$ For each t, suppose that (a) $\hat { f } ^ { ( { t - 1 } ) } ( \cdot )$ and $q _ { t } ( \cdot )$ are $\mathcal { F } _ { t - 1 }$ -measurable, and that (b) conditional on $\mathcal { F } _ { t - 1 } , q _ { t } ( \cdot )$ is a probability distribution on $R _ { t - 1 } : = \tau \backslash O _ { t - 1 }$ with $q _ { t } ( i ) \geq \tau / | R _ { t - 1 } | = \tau / ( N - t + 1 )$ for all $i \in \mathbb { R } _ { t - 1 } , f o r$ some fixed $\tau \in ( 0 , 1 \dot { ] }$

Then, $\hat { \theta } _ { n }$ is unbiased for θ at every finite n. Furthermore, under mild regularity conditions (variance stabilization, Lindeberg, and uniformfourth-moment controlfor weighted residuals; Assumptions 4-6 in Appendix $A . 1 ) , \hat { \theta } _ { n }$ obeys thefollowing limiting distribution:

$$
{ \sqrt { n } } \left( { \hat { \theta } } _ { n } - \theta \right) / { \hat { \sigma } } _ { n } \ { \xrightarrow { d } } { \mathcal { N } } \left( 0 , 1 \right) .\tag{3}
$$

And, the variance estimator is given by the following, where $\begin{array} { r } { \hat { \theta } _ { t - 1 } : = \frac { 1 } { t - 1 } \sum _ { s = 1 } ^ { t - 1 } \phi _ { s } f o r t \ge 2 . } \end{array}$

$$
\begin{array} { r l r } {  { \hat { \sigma } _ { n } ^ { 2 } : = \hat { A } _ { n } - \hat { B } _ { n } } } \\ & { } & { \quad : = \frac { 1 } { n N ^ { 2 } } \sum _ { t = 1 } ^ { n } \frac { ( y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) ) ^ { 2 } } { q _ { t } ( I _ { t } ) ^ { 2 } } - \frac { 1 } { n N ^ { 2 } } \sum _ { t = 2 } ^ { n } ( N \hat { \theta } _ { t - 1 } - \sum _ { i \in O _ { t - 1 } } y _ { i } - \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) ) ^ { 2 } . } \end{array}\tag{4}
$$

(5)

Proof. Please see Appendix A.2.

By Theorem 1, it follows that an asymptotic (1 − α)-level confidence interval for θ is given by

$$
\left[ \hat { \theta } _ { n } \pm z _ { 1 - \alpha / 2 } \left( \hat { \sigma } _ { n } / \sqrt { n } \right) \right] .\tag{6}
$$

## 3.3 An improved finite-sample variance estimator

Although $\hat { \sigma } _ { n } ^ { 2 } : = \hat { A } _ { n } - \hat { B } _ { r }$ is consistent for the limiting variance $\sigma ^ { 2 }$ governing the martingale CLT in Theorem 1 (see proof in Appendix A.2), empirically, it can occasionally underestimate variance at finite labeling budgets, yielding confidence intervals that are too narrow and coverage below the nominal 1 − α level.

Inspired by the higher-order bias-correction philosophy of the ABC and BCa bootstrap methods (c.f. Efron (1987); Diciccio $\&$ Efron (1992)), we derive an additive correction to $\hat { \sigma } _ { n } ^ { 2 }$ using a second-order Taylor expansion. Specifically, we examine the finite-sample error introduced by substituting $\hat { \theta } _ { t - 1 }$ for the unknown population mean θ in our variance estimator (Equation 4). This motivates the following bias-corrected variance estimator ${ \hat { \sigma } } _ { n , \mathrm { c o r } } ^ { 2 } ,$ where $\hat { A } _ { n }$ and ${ \hat { B } } _ { n }$ are defined above in Theorem 1:

$$
\hat { v } _ { n , \mathrm { c o r } } ^ { 2 } : = \hat { A } _ { n } - \hat { B } _ { n } + \hat { C } _ { n } , \quad \mathrm { w i t h } ~ \hat { C } _ { n } : = \frac { 1 } { n } \sum _ { s = 1 } ^ { n - 1 } \left( \frac { 1 } { N } \frac { y _ { I _ { s } } - \hat { f } ^ { ( s - 1 ) } ( x _ { I _ { s } } ) } { q _ { s } ( I _ { s } ) } \right) ^ { 2 } w _ { s } \quad \mathrm { a n d } \quad w _ { s } : = \sum _ { k = s } ^ { n - 1 } \frac { 1 } { k ^ { 2 } } .
$$

Importantly, the correction $\hat { C } _ { n }$ is nonnegative and vanishes asymptotically, preserving the asymptotic normality guarantee of Theorem 1. We provide the full derivation, proof of preserved asymptotic validity, and a characterization of when the correction reduces mean squared error in Appendix A.3. We use the biascorrected variance estimator $\hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 }$ in all our experiments and recommend it as the default for practitioners.

## 3.4 Pro-Active Inference estimator for superpopulation estimands

In the preceding subsections, we focused on the finite-population mean $\begin{array} { r } { \theta : = N ^ { - 1 } \sum _ { i = 1 } ^ { N } y _ { i } } \end{array}$ as our estimand, treating $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ as a fixed finite population.

However, oftentimes scientists may instead wish to adaptively sample from a finite population without replacement while making inference about the larger superpopulation from which the finite population was drawn. For example, following a natural disaster, policymakers may survey a subset of households from a representative sample of residents to assess their economic circumstances, while seeking to draw statistically principled conclusions about the broader population of affected households. In this subsection, we extend PAI to accommodate such superpopulation estimands.

Formally, we assume a superpopulation model with $( x _ { i } , y _ { i } ) \stackrel { \mathrm { i . i . d . } } { \sim } F ,$ with mean $\theta : = \mathbb { E } [ y _ { i } ]$ and variance $\tau ^ { 2 } : = \dot { \mathrm { V a r } } ( y _ { i } )$ . Our goal is to construct a confidence interval for the superpopulation mean θ. For this subsection only, define $\begin{array} { r } { \theta _ { N } : = N ^ { - 1 } \sum _ { i = 1 } ^ { N } y _ { i } } \end{array}$ to be the size-N finite-population mean of $y _ { 1 } , \ldots , y _ { N } .$ , where the $y _ { i }$ are now treated as random variables.

Let $\hat { \theta } _ { n }$ be our PAI estimator for $\theta _ { N }$ from Equation 2. The critical observation is that

$$
\sqrt { n } \left( \widehat { \theta } _ { n } - \theta \right) = \underbrace { \sqrt { n } ( \widehat { \theta } _ { n } - \theta _ { N } ) } _ { \mathrm { f i n i t e - p o p u l a t i o n ~ s a m p l i n g ~ e r r o r } } + \underbrace { \frac { \sqrt { n } } { \sqrt { N } } \sqrt { N } ( \theta _ { N } - \theta ) } _ { \mathrm { s u p e r p o p u l a t i o n ~ e r r o r } } .\tag{7}
$$

Intuitively, this decomposition separates the uncertainty from observing only a subset of the finite population from the additional uncertainty arising because the finite population itself is a sample from a larger superpopulation. Using this decomposition, we obtain the following limiting distribution.

Proposition 2. Retain the setup and assumptions of Theorem 1, where $\sigma ^ { 2 }$ denotes the limiting variance governing PAI’s finite-population martingale CLT. Further suppose that $\left( x _ { i } , y _ { i } \right) \stackrel { i . i . d . } { \sim } F$ with mean $\theta : = \mathbb { E } [ y _ { i } ]$ and variance $\tau ^ { 2 } : = \dot { V a r ( y _ { i } ) }$ . Then,

$$
{ \sqrt { n } } \left( { \hat { \theta } } _ { n } - \theta \right) { \xrightarrow { d } } { \mathcal { N } } \left( 0 , \sigma ^ { 2 } + \rho \tau ^ { 2 } \right) ,\tag{8}
$$

where $\rho : =$ lim $n / N$ from Theorem 1.

Proof. Please see Appendix A.4.

By Proposition 2 and Slutsky’s Theorem, an asymptotically valid $( 1 - \alpha )$ -confidence interval for the superpopulation mean θ is

$$
\left[ \hat { \theta } _ { n } \pm z _ { 1 - \alpha / 2 } \sqrt { \frac { \hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 } + ( n / N ) \hat { \tau } _ { n } ^ { 2 } } { n } } \right] ,\tag{9}
$$

where $\hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 }$ is our consistent bias-corrected variance estimator from Section 3.3, and $\hat { \tau } _ { n } ^ { 2 }$ is any consistent estimator of $\tau ^ { 2 }$ . To estimate the superpopulation variance $\tau ^ { 2 } = \mathbb { E } [ y _ { i } ^ { 2 } ] - \theta ^ { 2 }$ , one simple approach is to apply PAI again to estimate the finite-population second moment $\begin{array} { r } { \theta _ { N , 2 } : = N ^ { - 1 } \sum _ { i = 1 } ^ { N } y _ { i } ^ { 2 } . } \end{array}$ , using the exact same queried indices $I _ { 1 } , \ldots , I _ { n }$ used to construct $\hat { \theta } _ { n }$ . This procedure simply treats $y _ { i } ^ { 2 }$ as the outcome, with corresponding ML predictions $( \mathbf { e . g . } , \hat { g } ^ { ( t ) } ( x _ { i } ) : = ( \hat { f } ^ { ( t ) } ( x _ { i } ) ) ^ { 2 } )$ . Denoting the resulting PAI estimator by ${ \hat { \theta } } _ { n , 2 } ,$ we construct

$$
\hat { \tau } _ { n } ^ { 2 } : = \hat { \theta } _ { n , 2 } - \hat { \theta } _ { n } ^ { 2 } ,\tag{10}
$$

which is consistent for $\tau ^ { 2 }$ under the corresponding regularity conditions of Theorem 1. Importantly, this procedure requires no additional labels, since observing $y _ { I _ { t } }$ also reveals $y _ { I _ { t } } ^ { 2 }$ . For binary outcomes $y _ { i } \in \{ 0 , 1 \}$ , we can alternatively use the even simpler $\hat { \tau } _ { n } ^ { 2 } = \hat { \theta } _ { n } ( 1 - \hat { \theta } _ { n } )$ , since $y _ { i } ^ { 2 } = y _ { i }$

Furthermore, we note that our superpopulation confidence interval avoids both Bonferroni corrections and α-splitting by directly accounting for both finite-population sampling uncertainty and superpopulation variability in a single asymptotic distribution.

## 4 A theoretical characterization of PAI’s efficiency

Having established the PAI estimator and its inferential guarantees in Section $^ { 3 , }$ we now theoretically investigate when and why PAI can yield efficiency gains over active inference in the finite-population setting.

For our theoretical analysis, we consider a quasi-oracle setting where both PAI and Batch Active Inference (BAI), the batch variant of SAI, share the same fixed ML model $f ,$ with oracle-known prediction residuals $e _ { i } : = y _ { i } - f ( x _ { i } )$ and absolute residuals $r _ { i } : = | e _ { i } | . ^ { 3 }$ Recall that BAI estimates the finite-population mean using

$$
\hat { \theta } _ { \mathrm { a c t i v e } } : = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \Gamma _ { t } , \quad \Gamma _ { t } = f ( x _ { t } ) + \left( y _ { t } - f ( x _ { t } ) \right) \frac { \xi _ { t } } { \pi _ { t } ( x _ { t } ) } ,\tag{11}
$$

where $\pi _ { t } ( x _ { t } )$ is the probability of labeling observation t at time t and $\xi _ { t } \sim$ Bern $\left( \pi _ { t } ( x _ { t } ) \right)$ ) indicates whether its label was acquired. Following Zrnic & Candès (2024, Section 4 and Equation 10)’s recommended oracle sampling rule, we let both BAI and PAI sample proportionally to the true absolute prediction residuals:

$$
\mathrm { B A I : } \quad \pi _ { t } ( x _ { t } ) \propto r _ { t } \implies \pi _ { t } ( x _ { t } ) = \frac { n r _ { t } } { \sum _ { i = 1 } ^ { N } r _ { i } } , \quad \mathrm { P A I : } \quad q _ { t } ( i ) \propto r _ { i } \implies q _ { t } ( i ) = \frac { r _ { i } } { \sum _ { k \notin { O _ { t - 1 } } } r _ { k } } ,\tag{12}
$$

with the n present in BAI’s label-skip probability because we want the expected number of BAI queries to be ${ n . ^ { 4 } }$ By construction, PAI will achieve an exact budget of n labels.

We adopt this quasi-oracle comparison setting to isolate the efficiency differences between PAI and BAI, giving both methods access to the same fixed ML model and analogous oracle sampling policies based on the true absolute prediction residuals. Fixing the ML model and omitting uniform mixing and budget stabilization also makes our theoretical comparison analytically tractable.

Because both PAI and BAI are unbiased for $\theta ,$ to characterize their performance differences, it suffices to compare their variances. Let $R _ { t - 1 } = \tau \backslash O _ { t - 1 }$ represent the set of unlabeled indices after PAI sampling round $t - 1 .$ , where $\mathcal { T } = [ 1 : N ]$ is our full set of indices. Furthermore, for any set of indices $R \subseteq \mathbb { Z } ,$ define the sums of unsigned and signed residuals $U ( R )$ and $S ( R )$ , respectively:

$$
U ( R ) : = \sum _ { i \in R } r _ { i } , \quad S ( R ) : = \sum _ { i \in R } e _ { i } .\tag{13}
$$

Finally, define $b _ { t }$ to be the net signed direction of the residual mass before PAI sampling round $t ,$ and define ${ \bar { b } } ^ { 2 }$ to be the weighted-average squared net signed direction of the residual mass over the n PAI sampling rounds:

$$
b _ { t } : = \frac { S ( R _ { t - 1 } ) } { U ( R _ { t - 1 } ) } = \frac { \sum _ { i \in R _ { t - 1 } } \left( y _ { i } - f ( x _ { i } ) \right) } { \sum _ { i \in R _ { t - 1 } } | y _ { i } - f ( x _ { i } ) | } , \quad \bar { b } ^ { 2 } : = \frac { \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } b _ { t } ^ { 2 } \right] } { \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } \right] } \in [ 0 , 1 ] .\tag{14}
$$

Under the above setup, we obtain a theoretical characterization of PAI’s sampling efficiency over BAI.

Theorem 3. Under the quasi-oracle setting above, assuming the below relative variance ratio is well-defined, we have

$$
\frac { V a r ( \hat { \theta } _ { a c t i v e } ) } { V a r ( \hat { \theta } _ { P A I } ) } = \frac { 1 } { 1 - \bar { b } ^ { 2 } } \cdot \frac { 1 } { 1 + \eta _ { n } } ,\tag{15}
$$

where $\eta _ { n } \geq 0$ is a remainder term capturing additional finite-budget and residual-magnitude effects (see Appendix A.5 for the full derivation). Furthermore, if $\ ` { n } r _ { m a x } / U ( R _ { 0 } ) \to 0 , ^ { 5 }$ with $r _ { m a x } = \operatorname * { m a x } _ { i \in [ 1 : N ] } r _ { i } ,$ , then $\eta _ { n }  0 ,$ , yielding

$$
\frac { V a r ( \hat { \theta } _ { a c t i v e } ) } { V a r ( \hat { \theta } _ { P A I } ) } = \frac { 1 + o ( 1 ) } { 1 - \bar { b } ^ { 2 } } .
$$

Proof. Please see Appendix A.5.

□

In context, Theorem 3 suggests that PAI can substantially outperform BAI when (i) the ML model’s prediction residuals systematically point in the same direction, rather than canceling each other out (i.e., when ${ \bar { b } } ^ { 2 }$ is large); and (ii) the sampling policy effectively prioritizes observations with large absolute prediction residuals (as in our quasi-oracle setting). We do emphasize that such prediction residual imbalance is not a measure of predictive accuracy: a highly accurate model can still have a large ${ \bar { b } } ^ { 2 }$ if its small residuals consistently point in one direction. Alternatively, a large ${ \bar { b } } ^ { 2 }$ could arise when the ML model predicts most observations accurately, but exhibits a systematic bias on a small subgroup of the population.

In Section 6, we will investigate whether this theoretical insight empirically carries over to more realistic settings with learned acquisition policies and online ML updates.

## 5 Experimental setup

Our empirical experiments seek to characterize PAI’s sampling efficiency and statistical coverage relative to its intellectual predecessor Sequential Active Inference (SAI) (Zrnic & Candès, 2024), the closely related contemporary method Prediction-Powered Active Testing (PPAT) (Ashouritaklimi et al., 2026), and the simpler baselines Prediction-Powered Inference (PPI) (Angelopoulos et al., 2023) and uniform sampling. We evaluate these methods across eight real-world datasets spanning computational biology, LLM evaluations, economics, fraud detection, and cybersecurity.

We focus on finite-population mean estimation with binary outcomes $y _ { i } \in \{ 0 , 1 \}$ . The finite-population setting allows us to directly evaluate statistical coverage against the known population mean, whereas the superpopulation mean is generally unknown for our testbeds. Focusing on binary outcomes also provides simple, well-established uncertainty scores for guiding acquisition $( \mathrm { e . g . } , u _ { t } ( x ) = 2 \operatorname* { m i n } ( f ( x ) , 1 - f ( x ) )$ where f is a probabilistic ML predictor; see Zrnic & Candès (2024, Section 4)), avoiding additional modeling and acquisition-policy choices that arise in regression settings and could complicate the interpretation of performance differences. We emphasize that none of these methods are restricted to binary outcomes.

Importantly, to isolate differences between the inferential methods, we use the same underlying ML model and initialization across PAI, SAI, PPI, and PPAT, as well as the same online learning procedure wherever applicable. We also use the same underlying uncertainty scores and uniform-mixing parameter for PAI and SAI, following the recommendations of Zrnic & Candès (2024), while respecting PPAT’s own estimator and recommended acquisition mechanisms. Rather than engaging in extensive dataset-specific hyperparameter tuning, our goal is to understand how effectively these methods leverage the same underlying predictive information for statistical inference.

## 5.1 Datasets and ML models

All dataset processing routines can be found in the data\_processing notebook in our Github repository.

ACS We use the ACSIncome task (Ding et al., 2021), where the binary outcome indicates whether an individual’s annual income exceeds \$50, 000, using demographic covariates. Adapting their distribution-shift setting, we train the initial ML model on California and evaluate on Puerto Rico, motivated by the importance of label-efficient data collection in resource-constrained settings, such as following natural disasters. Our test population consists of 9, 071 individuals with 11 covariates.

AlphaFold We use the protein-disorder dataset from Angelopoulos et al. (2023), originally derived from Bludau et al. (2022). Each observation corresponds to an amino-acid residue, with a binary outcome indicating whether it belongs to an intrinsically disordered region. We estimate the proportion of disordered residues among 10, 802 observations, using probabilistic ML predictions obtained from a logistic regression model fitted to AlphaFold-derived structural features. Since this dataset provides only the predictions and labels, without the underlying covariates, all ML-assisted methods use fixed predictions without online updates. For SAI, this corresponds to the Batch Active Inference (BAI) setting.

LLM evaluations We study estimation of LLM accuracy on two benchmark suites: MMLU-Pro (Wang et al., 2024), containing 12, 032 questions, and a composite of BBH, GPQA, IFEval, MATH, and MuSR (Suzgun et al., 2023; Rein et al., 2024; Zhou et al., 2023; Hendrycks et al., 2021; Sprague et al., 2024), containing 9, 574 questions. Following Wu et al. (2026), we use their train/test splits and 64-dimensional question and model factors fitted to historical evaluation data from Fourrier et al. (2024); Myrzakhan et al. (2024), with approximately 2.2K held-out LLMs per suite. For each held-out LLM, we fix the historical question factors and initialize its model factor at the empirical mean of historical model factors, using their empirical marginal variances to initialize the diagonal curvature for online logistic regression updates. These testbeds also allow us to directly compare the sampling efficiency of our without-replacement PAI estimator against its with-replacement predecessor from Wu et al. (2026).

BODMAS We use the BODMAS malware detection dataset (Yang et al., 2021), where each observation represents a software sample with a binary label indicating whether it is malicious or benign. We employ a chronological 50/50 train-test split, fitting the initial ML model on the earlier observations and performing inference on the later 67, 218 observations. After removing features that are identically zero in the training split, each observation has 2, 330 covariates.

HIV We use the HIV molecular activity dataset from MoleculeNet (Wu et al., 2018), accessed through DeepChem. Each observation represents a chemical compound screened for its ability to inhibit HIV replication, with confirmed or moderately active compounds labeled as positive and inactive compounds labeled as negative. We represent compounds using ECFP molecular fingerprints and employ a 50/50 traintest split, yielding a test population of 20, 560 compounds with 1, 025 covariates.

Pew We use Wave 168 of the Pew Research Center’s American Trends Panel, which surveys U.S. adults about online scams and related topics. Our binary outcome indicates whether a respondent reports having ever lost money because of an online scam or attack. We use demographic covariates and responses to other scam-related survey questions and employ a 50/50 train-test split, yielding a test population of 4, 273 respondents with 53 covariates.

ULB We use the ULB Credit Card Fraud Detection dataset (Dal Pozzolo et al., 2015), containing anonymized credit card transactions from European cardholders, with a binary outcome indicating whether a transaction is fraudulent. The covariates comprise 28 PCA-transformed features, transaction time, and amount. We employ a 50/50 train-test split, yielding a test population of 142, 404 transactions with 30 covariates. This highly imbalanced dataset, with approximately 0.17% fraudulent transactions, allows us to investigate statistical inference for rare events.

ML models and online updates For all ML-assisted methods (i.e., PAI, SAI, PPI, and PPAT), we use the same underlying probabilistic ML model and online updating procedure whenever applicable. For all datasets except the LLM evaluation suites and AlphaFold, we standardize covariates using training-split statistics and fit an initial logistic regression on the training data, initializing its diagonal curvature approximation to the identity. For the two LLM evaluation suites, following Wu et al. (2026), we instead initialize the model coefficients using the empirical mean of historical model factors and the diagonal curvature using their inverse empirical marginal variances, retaining the original question-factor representations. For all datasets except AlphaFold, we update the logistic regression model after every acquired label using computationally-efficient diagonal stochastic Newton updates adapted from Bercu et al. (2020), as detailed in Appendix A.6. AlphaFold is the sole exception, since only fixed ML predictions are available.

## 5.2 Baselines and competitor methods

PAI and SAI For PAI and SAI, we use the same underlying ML predictor and online updating procedure. Following Zrnic & Candès (2024, Sections 4 and 7.2), both methods also use the uncertainty score $\boldsymbol u _ { t } ( \boldsymbol x ) =$ $2 \operatorname* { m i n } ( \hat { f } ^ { ( t - 1 ) } ( x ) , 1 - \hat { f } ^ { ( t - 1 ) } ( x ) )$ and uniform-mixing parameter $\tau = 0 . 5$ . At each round $t ,$ PAI samples without replacement from the remaining unlabeled pool $R _ { t - 1 }$ according to the acquisition policy

$$
q _ { t } ( i ) = \frac { \tau } { | R _ { t - 1 } | } + ( 1 - \tau ) \frac { u _ { t } ( x _ { i } ) } { \sum _ { j \in R _ { t - 1 } } u _ { t } ( x _ { j } ) } .\tag{16}
$$

In contrast, SAI processes the test observations sequentially, at round t choosing whether to acquire label $y _ { t }$ with probability $\pi _ { t } ( x _ { t } )$ . Its mean estimator is

$$
\hat { \theta } _ { \mathrm { a c t i v e } } : = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \left( \hat { f } ^ { ( t - 1 ) } ( x _ { t } ) + \left( y _ { t } - \hat { f } ^ { ( t - 1 ) } ( x _ { t } ) \right) \frac { \xi _ { t } } { \pi _ { t } ( x _ { t } ) } \right) ,\tag{17}
$$

where $\xi _ { t } \sim \operatorname { B e r n } ( \pi _ { t } ( x _ { t } ) )$ indicates whether the label is acquired. To target a labeling budget of n observations from a test population of size $N _ { \cdot }$ , we use the following nominal acquisition probability:

$$
\pi _ { t } ^ { \mathrm { n o m } } ( x _ { t } ) = \tau \left( \frac { n } { N } \right) + ( 1 - \tau ) \frac { n u _ { t } ( x _ { t } ) } { \sum _ { j = 1 } ^ { N } u _ { t } ( x _ { j } ) } .\tag{18}
$$

To prevent budget underutilization and ensure good-faith comparisons, we additionally implement the SAI budget-stabilization rule of Zrnic & Candès (2024, Equation 8) and their periodic budget catch-up heuristic from Section 8.4. These adjustments are applied to the uncertainty-based probabilities before mixing with the uniform component, with the resulting acquisition probabilities $\pi _ { t } ( x _ { t } )$ used in the SAI estimator.

PPI and uniform sampling We implement PPI as SAI (Equation 17) with uniform Bernoulli acquisition probabilities $\pi _ { t } ( x _ { t } ) = n / N$ , retaining the shared ML predictions and online updates in the estimator. Our implementation of uniform sampling, while preserving the SAI estimator structure, additionally removes the ML predictions, recovering the corresponding inverse-probability-weighted estimator. Thus, PPI isolates the benefits of ML-assisted inference without adaptive acquisition, while uniform sampling serves as a baseline without any ML assistance. Because both methods use fixed acquisition probabilities, neither requires SAI’s budget-stabilization rule or periodic budget catch-up heuristic.

PPAT We implement the online-tuned $\hat { \lambda } _ { M }$ variant of PPAT from Ashouritaklimi et al. (2026), using its prediction-powered control-variate estimator and proposed acquisition policy specialized to binary out comes (Section 3.2). Following Section 5.1, we initialize $\hat { \lambda } _ { M } = 0 . 5$ and update it after every 100 acquired labels. We also use uniform mixing with $\tau = 0 . 5 ,$ matching PAI and SAI for consistency. We initialize PPAT’s control-variate proxy and acquisition surrogate using the same initial ML predictor as the other ML-assisted methods. During sampling, we update the acquisition surrogate after every acquired label using our shared online learning procedure, while keeping the control-variate proxy fixed. Importantly, this is precisely the fixed-proxy configuration adopted by the PPAT authors themselves when benchmarking against Active Statistical Inference in Appendix C.4.3, combined with their online surrogate updates in Section 5.1. Finally, we select the online-tuned $\hat { \lambda } _ { M }$ variant over the fixed choices $\lambda \in \{ 0 . 5 , 1 \}$ because it had the strongest performance in our preliminary comparisons.

## 5.3 Budgets, metrics, and coverage

Budgets and coverage For each dataset, we consider ten evenly-spaced budgets $n \in [ 2 . 5 \% , 2 5 \% ]$ of the test population and target 95% nominal coverage for all experiments. All methods construct nominal 95% Wald-type confidence intervals of the form $[ \hat { \theta } _ { \mathrm { m e t h o d } } ( n ) \pm 1 . 9 6 \hat { \sigma } _ { \mathrm { m e t h o d } } ( n ) ]$ , using their respective plug-in variance estimators. For each method and budget, we run 1000 random seeds on the non-LLM datasets, and 100 random seeds for each of approximately 2.2K held-out LLMs on the two LLM evaluation suites. We compute empirical coverage as the proportion of confidence intervals containing the true finite-population mean θ, averaging over random seeds and, for the LLM evaluation suites, held-out LLMs.

Metrics To quantify sampling efficiency, we use effective sample size multipliers (ESS multipliers) based on estimated variances. For a method M and reference method $B ,$ both evaluated at labeling budget $n ,$ we define the ESS multiplier ofM with respect to B as

$$
\gamma _ { \mathrm { e f f } } ^ { M / B } ( n ) : = \hat { \sigma } _ { B } ^ { 2 } ( n ) / \hat { \sigma } _ { M } ^ { 2 } ( n ) .\tag{19}
$$

In particular, taking uniform sampling U as the reference, the effective sample size (ESS) of method M is

$$
n _ { \mathrm { e f f } } : = \gamma _ { \mathrm { e f f } } ^ { M / U } ( n ) \cdot n ,\tag{20}
$$

with uniform sampling having ESS n by definition. An ESS multiplier greater than 1 indicates higher sampling efficiency than the reference method. For example, at budget n, a multiplier of 1.5 with respect to uniform sampling means achieving approximately the same confidence interval width as uniform sampling with 1.5n labels, corresponding to roughly 33% fewer labels for comparable precision. We report ESS multipliers relative to uniform sampling, SAI, PPI, and PPAT to facilitate comparisons across methods.

## 6 Results

Our empirical results seek to investigate four primary questions. First, how does PAI compare in sampling efficiency and statistical coverage against its predecessors: uniform sampling, PPI, and SAI? Second, to what extent does our theoretical characterization of PAI’s efficiency over active inference (Theorem 3) carry over to real-data settings with learned acquisition policies and online ML updates? Third, how does PAI compare against the closely related contemporary method PPAT? Fourth, and perhaps most importantly, beyond simply reporting wins and losses, can we identify interpretable explanations for when and why PAI outperforms or underperforms these competing methods, yielding insights applicable to ML-assisted adaptive statistical inference more broadly?

We investigate each of these questions below. Before proceeding, however, we first discuss an important coverage failure on the ULB Credit Card Fraud dataset.

Coverages on the ULB dataset We observe that all three methods using ML predictions to guide acquisition—PAI, SAI, and PPAT—severely undercover the nominal 95% level at the smallest labeling budgets. At the 2.5% budget, empirical coverages are 64.6%, 82.0%, and 63.4% for PAI, SAI, and PPAT, respectively, compared to 92.1% for both PPI and Uniform.

![](images/7dce5c198ab3745ecb0f82fe21965f3e376e023b6d9baf348c8ffa4cfd58440b.jpg)  
Figure 1: Comparing sampling efficiency of Pro-Active Inference (PAI), Sequential Active Inference (SAI), and Prediction-Powered Inference (PPI) with respect to uniform sampling across seven datasets. Each subplot shows the effective sample size (ESS) multiplier with respect to uniform sampling as a function of labeling budget for PAI, SAI, and PPI. For example, on ACS at a 10% budget, an ESS multiplier of 1.6 means PAI matches the confidence interval width of uniform sampling at a 16% budget. Standard errors are shown, but negligible on the plot scale. Higher ESS multipliers indicate better sampling efficiency, with values above 1 indicating an improvement over uniform sampling.

We hypothesize that this undercoverage is driven by the rarity of positive labels, potentially exacerbated by confidently mispredicted positives under ML-guided acquisition. Of the N = 142, 404 observations in the ULB test set, only $N _ { 1 } = 2 2 \bar { 3 }$ (approximately 0.15%) are fraudulent transactions $( y _ { i } = 1 )$ . At the 2.5% labeling budget, even uniform sampling would acquire only approximately 5.6 positive labels in expectation, potentially compromising the accuracy of normal-approximation confidence intervals. Of these 223 positive observations, the shared initial ML model assigns 39 of them a fraud probability below 1%. These confidently mispredicted positives have large prediction residuals, yet can receive low acquisition probabilities under the ML-guided sampling policies.

Crucially, observing one of these confidently mispredicted positives can provide both (i) a substantial upward correction to the estimated population mean ${ \widehat { \theta } } ;$ and (ii) evidence of substantial sampling variability, particularly through inverse-probability weighting contributions to the variance estimators. Sampling trajectories that fail to acquire these observations can therefore underestimate both $\hat { \theta }$ and its standard error ${ \hat { \sigma } } ,$ yielding overly narrow confidence intervals that fail to cover the true population mean.

Although the rarity of positive outcomes does not itself invalidate asymptotic inference, we suspect that the normal approximations underlying these methods are unreliable at the labeling budgets considered, with ML-guided acquisition potentially compounding this problem by undersampling rare but consequential positive observations.

Because apparent gains in sampling efficiency are not meaningful without adequate statistical coverage, we exclude ULB from the subsequent ESS comparisons and focus on the remaining seven datasets.

![](images/a52a5718ea5338cc990cb4b4185277ce22680a8e1f9498f426e0084918dab940.jpg)  
Figure 2: ESS multipliers of PAI over SAI as a function of initial residual imbalance $b _ { 0 } ^ { 2 } .$ . Each color represents one dataset, with ten points corresponding to the ten labeling budgets tested. Unlike Figure 1, ESS multipliers here are measured with respect to SAI rather than uniform sampling. For example, an ESS multiplier of 2.5 means that PAI matches the confidence interval width of SAI while using only 40% as many labels. Higher ESS multipliers indicate better sampling efficiency, with values above 1 indicating that PAI outperforms SAI. Spearman rank correlations were computed across the seven datasets at each fixed budget, and we report the median, interquartile range (IQR), and minimum of the resulting ten correlations.

## 6.1 How does PAI compare against uniform sampling, PPI, and SAI?

On the remaining seven datasets, all methods achieve near-nominal coverage (Figure 6), enabling meaningful comparisons of sampling efficiency. First, PAI achieves higher ESS than SAI on all 10 of 10 tested budgets on ACS, BBH+GPQA+IFEval+MATH+MuSR, BODMAS, MMLU-Pro, and Pew; on 9 of 10 tested budgets on AlphaFold; and on 3 of 10 tested budgets on HIV (Figure 1). One of the motivations for developing without-replacement PAI was to address limitations of our earlier with-replacement PAI formulation encountered in LLM evaluations (Wu et al., 2026). Reassuringly, on both LLM evaluation suites, without replacement PAI consistently outperforms its with-replacement predecessor in sampling efficiency (Appendix Figure 7). In short, PAI represents a compelling improvement over SAI in pool-based settings, supporting its role as a natural successor to SAI when the full covariate bank is available for proactive sampling.

However, this does not mean that more sophisticated adaptive inference necessarily outperforms simpler methods such as Uniform and PPI. Somewhat surprisingly, PAI only outperforms uniform sampling on 5 of 7 datasets (albeit across all tested budgets), while considerably underperforming on HIV and Pew at all tested budgets (Figure 1). Against PPI, PAI outperforms on 4 of 7 datasets (again, across all tested budgets), while substantially underperforming on BODMAS, HIV, and Pew. We defer a detailed investigation of these performance trends to Section 6.4.

## 6.2 Does our theoretical efficiency characterization hold empirically?

Motivated by Theorem 3, we define the quasi-oracle initial residual imbalance (IRB) diagnostic $b _ { 0 } ^ { 2 }$ as a one-step analogue of the ${ \bar { b } } ^ { 2 }$ quantity in our theoretical characterization, which can be computed without running full sample trajectories:

$$
b _ { 0 } ^ { 2 } : = \left( \frac { \sum _ { i = 1 } ^ { N } \Big ( y _ { i } - \hat { f } ^ { ( 0 ) } ( x _ { i } ) \Big ) } { \sum _ { i = 1 } ^ { N } \Big | y _ { i } - \hat { f } ^ { ( 0 ) } ( x _ { i } ) \Big | } \right) ^ { 2 } .\tag{21}
$$

Intuitively, IRB measures the extent to which the initial ML model’s prediction residuals systematically point in the same direction, rather than canceling each other out. Values near 0 indicate that positive and negative residuals largely cancel, while values near 1 indicate that residuals of one sign dominate in aggregate magnitude. As discussed in Section 4, IRB is not necessarily indicative of predictive accuracy.

![](images/af3293f8dbeb9046a0400df56bc0ff24c57b62b1085e765924a1e79435a7f53e.jpg)  
Figure 3: ESS multipliers of PAI over PPAT across seven datasets. Each subplot shows PAI’s ESS multiplier with respect to PPAT as a function of labeling budget. For example, on AlphaFold at a 2.5% budget, an ESS multiplier of 1.3 means PAI matches the confidence interval width of PPAT at a 3.25% budget. Standard errors are shown but are generally negligible at the plot scale. Higher ESS multipliers indicate better sampling efficiency, with values above 1 indicating that PAI outperforms PPAT.

Figure 2 provides strong empirical corroboration of the qualitative prediction from Theorem 3. Even with online-updated ML models and acquisition policies, IRB exhibits a strong positive rank correlation with PAI’s ESS multiplier gains over SAI across labeling budgets. While we cannot expect the exact 1/(1 − <sup>¯</sup>b<sup>2</sup>) functional form to carry over to these more realistic settings, the empirical relationship is striking: larger initial residual imbalance is consistently associated with greater efficiency gains for PAI over SAI.

On quasi-oracle diagnostics We emphasize that the diagnostics in this section (i.e., IRB, ECB, and IVR in Equations 21-23, respectively) are quasi-oracle, computed using the full test population of covariates and labels. These labels are used only for retrospective diagnostic analysis, not for guiding the estimators’ acquisition policies or constructing their estimates. Our goal is to identify interpretable properties associated with sampling efficiency, without relying on post-hoc analyses of the realized sampling trajectories.

## 6.3 How does PAI compare against PPAT?

PAI outperforms PPAT on 10 of 10 tested budgets on AlphaFold; on 9 of 10 budgets each on ACS and MMLU-Pro; and on 8 of 10 budgets on BBH+GPQA+IFEval+MATH+MuSR (Figure 3). However, PAI considerably underperforms PPAT on all tested budgets on BODMAS, HIV, and Pew. Curiously, these are precisely the same three datasets where PAI also underperforms PPI (Section 6.1). This common pattern motivates our investigation into when and why PAI gains or loses efficiency relative to competing methods, which we pursue in the following subsection.

![](images/af34357c147d9a2ee9d6b45dcd75c5292020b1fc4c5d2c9ad1e033d0d0ff1134.jpg)  
Expected 1-Step Relative Change in Brier Score (ECB)

![](images/46c87c4faad23cb89272be8cc6687c025c7b8e2d2b0234fffdd6c3f800f7e995.jpg)  
Expected 1-Step Relative Change in Brier Score (ECB)

Figure 4: ESS multipliers of PAI over PPI and PPAT as a function of the expected 1-step relative change in Brier score (ECB). Each color represents one dataset, with ten points corresponding to the ten labeling budgets tested. The left and right subplots show ESS multipliers with respect to PPI and PPAT, respectively. For example, on ACS in the left subplot, an ESS multiplier of 1.25 means that PAI matches the confidence interval width of PPI while using 20% fewer labels. Higher ESS multipliers indicate better sampling efficiency, with values above 1 indicating that PAI outperforms the corresponding competitor. Spearman rank correlations were computed across the six datasets at each fixed budget, and we report the minimum and maximum of the resulting ten correlations in each subplot.

## 6.4 What can we learn from PAI’s efficiency gains and losses?

## 6.4.1 When and why does PAI outperform or underperform PPI and PPAT?

One potential explanation for PAI’s relative efficiency gains and losses against PPI and PPAT lies in its increased sensitivity to the quality of online ML updates. Specifically, PAI incorporates updated ML predictions into both (i) its plug-in estimate and residual correction; and (ii) its acquisition policy. In contrast, PPI, implemented in our experiments as SAI with a uniform acquisition policy, is only affected by online ML updates through its prediction-powered regression adjustment. Similarly, PPAT, as implemented in our experiments, only uses the online-updated ML model to guide acquisition (which Ashouritaklimi et al. (2026) refer to as the “surrogate"), while the prediction-based control variate retains the initial ML predic tions (which the authors refer to as the “proxy"). Importantly, this fixed-proxy formulation is also used by the PPAT authors themselves in their empirical comparisons against Active Statistical Inference (Ashouritaklimi et al., 2026, Section 5.1 and Appendix C.4.3).

Thus, while PAI has two channels of sensitivity to online prediction quality, PPI and PPAT, as implemented in our experiments, each have only one. We hypothesize that deteriorating online ML predictions can harm PAI’s sampling efficiency through both channels: less accurate imputations can increase residual-estimation variability, while poorly-guided acquisition can further inflate variance by assigning unfavorable sampling probabilities to observations with large prediction residuals. Conversely, when online ML updates improve predictions, PAI can potentially benefit from both improved imputation and better-guided acquisition.

To investigate this hypothesis, we quantify how the initial online ML update affects probabilistic prediction quality. For each dataset, we define the expected 1-step relative change in Brier score (ECB) as follows:

$$
\mathrm { E C B } : = \frac { \sum _ { i = 1 } ^ { N } q _ { 1 } ( i ) \mathrm { B r i e r } _ { - i } \left( \hat { f } ^ { ( 0 , + i ) } \right) } { \sum _ { i = 1 } ^ { N } q _ { 1 } ( i ) \mathrm { B r i e r } _ { - i } \left( \hat { f } ^ { ( 0 ) } \right) } - 1 , \quad \mathrm { w h e r e ~ B r i e r } _ { - i } ( f ) : = \frac { 1 } { N - 1 } \sum _ { \substack { j = 1 , j \neq i } } ^ { N } \left( y _ { j } - f ( x _ { j } ) \right) ^ { 2 } .\tag{22}
$$

Here, $q _ { 1 } ( i )$ is PAI’s initial sampling policy (Equation 16), Brier<sub>−</sub> $_ { - i } ( f )$ is the Brier score of model $f$ evaluated over the test population excluding index $i ,$ and ${ \hat { f } } ^ { ( 0 , + i ) }$ is the online-updated ML model obtained after observing $( x _ { i } , y _ { i } )$ (see Equation 28 in Appendix $_ { \mathrm { A } . 6 ) }$ . Intuitively, ECB measures the relative change in our ML model’s expected probabilistic prediction error after one online update, averaging over the initial PAI sampling policy. Negative ECB values indicate model improvement, while positive values indicate deterioration.

![](images/61d4996f475486f1e36bfd2967425204018071871b75b0b5d1969330ca4cc997.jpg)  
Initial Variance Ratio With Respect To Uniform Sampling (IVR)

Figure 5: ESS multipliers of PAI over uniform sampling as a function of initial variance ratio (IVR). Each color represents one dataset, with ten points corresponding to the ten labeling budgets tested. An IVR(n) greater than 1 indicates that the initial variance approximation favors PAI over uniform sampling, suggesting a potential efficiency advantage for PAI. Higher ESS multipliers indicate better sampling efficiency, with values above 1 indicating that PAI outperforms uniform sampling. Spearman rank correlations were computed across the seven datasets at each fixed budget, and we report the minimum and maximum of the resulting ten correlations.

Figure 4 provides strong empirical support for our hypothesis. Among the six datasets with online ML updates,<sup>6</sup> all three datasets where PAI outperforms both PPI and PPAT on the majority of tested budgets (ACS, MMLU-Pro, and BBH+GPQA+IFEval+MATH+MuSR) correspond to negative ECB values, while all three datasets where PAI underperforms both methods (BODMAS, HIV, and Pew) correspond to positive ECB values. Beyond this striking separation, ECB is also strongly negatively rank-correlated with PAI’s ESS multipliers over both competing methods across the tested labeling budgets.

To further investigate this hypothesis, we conduct a frozen-ML ablation across all six datasets, holding PAI’s initial ML model fixed throughout sampling and thereby disabling online updates in both imputation and acquisition (Appendix Figure 8). Strikingly, on the three datasets with negative ECB values and where PAI outperformed both PPI and PPAT (ACS and the two LLM evaluation suites), freezing substantially reduces PAI’s relative sampling efficiency, reversing its advantages over both PPI and PPAT. Conversely, on the three datasets with positive ECB values (BODMAS, HIV, and Pew), freezing substantially improves PAI’s relative efficiency, reversing its disadvantages against PPI on all three datasets and against PPAT on Pew and 6 of 10 budgets on HIV. On BODMAS, freezing substantially narrows PAI’s disadvantage against PPAT, although it does not eliminate it.

We emphasize that the competing methods’ implementations remain unchanged throughout these ablations, as our goal is to understand the effect of online updating on PAI’s relative efficiency, using the competing methods as fixed reference points, rather than attempting to optimize PAI’s performance against them. Furthermore, because freezing simultaneously affects PAI’s imputation and acquisition, these ablations identify the combined effect of online updating without isolating the contribution of either channel.

## 6.4.2 When and why does PAI outperform or underperform uniform sampling?

Our empirical results raise an important question: when is it advantageous to use ML predictions and adaptive acquisition to estimate a population mean, rather than simply estimating the mean directly through uniform sampling? Intuitively, PAI uses ML predictions to reduce the estimation problem to estimating a residual correction. However, this residual-estimation problem is not necessarily less variable than directly estimating the population mean, particularly if the acquisition policy assigns unfavorable sampling probabilities to observations with large prediction residuals.

Interestingly, the two datasets where PAI underperforms uniform sampling (HIV and Pew) constitute a strict subset of the three datasets where PAI underperforms both PPI and PPAT (BODMAS, HIV, and Pew).

To investigate when residual estimation under PAI is more favorable than direct estimation under uniform sampling, we define the initial variance ratio (IVR) at labeling budget n as

$$
\operatorname { I V R } ( n ) : = \frac { \operatorname { V a r } ( \hat { \theta } _ { \operatorname { W i n i f } , n } ) } { n ^ { - 1 } \operatorname { V a r } ( \phi _ { 1 } ) } = \frac { ( 1 - n / N ) \theta } { \sum _ { i = 1 } ^ { N } q _ { 1 } ( i ) \left( \bar { f } ^ { ( 0 ) } + \frac { 1 } { N } \frac { y _ { i } - \bar { f } ^ { ( 0 ) } ( x _ { i } ) } { q _ { 1 } ( i ) } - \theta \right) ^ { 2 } } , \quad \mathrm { w h e r e ~ } \bar { f } ^ { ( 0 ) } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \hat { f } ^ { ( 0 ) } ( x _ { i } ) .\tag{23}
$$

Here, uniform sampling is implemented using the no-ML Bernoulli label-skip estimator of Zrnic & Candès (2024), with uniform inclusion probability $n / N$ . The second equality in Equation 23 holds for binary outcomes $y _ { i } \in \{ 0 , 1 \}$ under this sampling design.

Intuitively, IVR compares the variance of directly estimating the population mean under uniform sampling against the initial variance of estimating the ML model’s mean residual under PAI’s acquisition policy. Larger IVR values suggest that the residual-estimation problem is less noisy than directly estimating the outcome mean, and thus greater potential efficiency gains for PAI over uniform sampling. To obtain a simple, interpretable diagnostic without relying on realized sampling trajectories, we approximate PAI’s variance at budget n using its initial one-step variance, scaled by $1 / n$ . Thus, IVR captures the joint effect of the initial ML predictions and acquisition policy, without accounting for subsequent online model updates, corresponding changes in the acquisition policy, or without-replacement effects.

Figure 5 shows a strong positive rank correlation between IVR and PAI’s realized ESS multipliers over uniform sampling across labeling budgets. This suggests that even the initial ML predictions and acquisition policy already contain substantial information about PAI’s eventual relative sampling efficiency, despite the subsequent complexities of online ML updates and adaptive sampling.

What did we learn? Tying together our insights from both Sections 6.4.1 and 6.4.2, we note that IVR and ECB offer complementary perspectives on PAI’s sampling efficiency. While IVR characterizes the relative difficulty of the initial residual-estimation problem under PAI’s acquisition policy, ECB characterizes how the quality of ML predictions changes after an initial online update. Our frozen-ML PAI ablations (described in Section 6.4.1; results in Appendix Figure 8) further demonstrate that subsequent online learning can substantially improve or deteriorate PAI’s realized efficiency relative to its frozen-model counterpart. Thus, understanding when ML-assisted adaptive inference is advantageous requires considering both the initial difficulty of the residual-estimation problem and the subsequent behavior of the online learner.

## 7 Discussion and conclusion

Contributions In this work, we develop a more general, without-replacement formulation of Pro-Active Inference (PAI), building on our earlier formulation for efficient LLM evaluation (Wu et al., 2026), and provide a comprehensive statistical treatment of its properties and broader applicability to ML-assisted statistical inference. Our original goal was to build upon the Active Statistical Inference framework of Zrnic & Candès (2024) while pursuing two key desiderata: (i) making fuller use of the available covariate information by proactively selecting observations for labeling from the entire remaining pool, rather than restricting each round to a label-skip decision about the current observation in a predetermined sequence; and (ii) sampling without replacement.

Methodologically, we establish PAI’s finite-sample unbiasedness and asymptotic validity, develop an improved finite-sample variance estimator, and extend PAI to superpopulation inference without the need for Bonferroni corrections or α-splitting. We further characterize PAI’s potential efficiency gains over Active Statistical Inference through a theoretical analysis highlighting the role of prediction-residual imbalance. Empirically, through extensive experiments across eight datasets, interpretable theory-motivated diagnostics, and frozen-ML ablations, we investigated when and why PAI outperforms or underperforms competing inference methods, with the broader goal of understanding when ML-assisted adaptive statistical inference itself is advantageous.

On our original goal of building upon Active Statistical Inference Turning to our original goal of building upon Active Statistical Inference, we believe PAI largely achieves both of our stated desiderata. Beyond satisfying them by construction, PAI achieves greater sampling efficiency than SAI across the vast majority of our tested settings, while using the same underlying ML predictor and uncertainty score. Unlike SAI, which processes observations in a predetermined order and cannot revisit skipped observations, PAI can reconsider all remaining unlabeled observations after each acquisition. This flexibility can be particularly valuable when certain observations have large prediction residuals, as PAI retains opportunities to acquire their labels even if they were not selected initially. Our theoretical characterization further identifies prediction-residual imbalance as a key condition under which PAI can achieve substantial efficiency gains over Batch Active Inference. Reassuringly, our empirical initial residual imbalance (IRB) diagnostic strongly corroborates this theoretical intuition, even in more realistic settings with online ML updates and learned acquisition policies.

Understanding the efficiency of ML-assisted adaptive inference Beyond PAI’s improvements over SAI, our experiments highlight that more sophisticated ML-assisted adaptive inference does not necessarily outperform simpler methods such as PPI and uniform sampling. Our proposed diagnostics offer two complementary perspectives on when and why these efficiency gains or losses occur: (i) IVR characterizes how much easier it is to estimate the ML model’s prediction residuals rather than the population mean directly; while (ii) ECB characterizes whether online ML updates improve or deteriorate prediction quality. Our frozen-ML ablations further corroborate the importance of online learning dynamics, while our comparisons with PPAT highlight how different uses of updated ML predictions in acquisition and estimation can affect efficiency.

From retrospective diagnostics to prospective guidance Although our proposed diagnostics IRB, ECB, and IVR were computed retrospectively using the fully labeled test population, an appealing feature is that they summarize important characteristics of the inference problem into a small number of interpretable statistics, without requiring full sampling trajectories. This raises the possibility of approximating these diagnostics using representative validation data or small pilot samples, or reasoning about their values using domain-specific knowledge. For example, IVR depends partly on the unknown population mean θ, for which domain scientists may be able to provide plausible ranges, allowing sensitivity analyses when combined with validation- or pilot-based estimates of the relevant residual moments. Similarly, prelim inary assessments of online ML updates could help inform whether to continually update or freeze the predictor, or perhaps consider a different update rule. Such assessments could eventually guide the choice of inference method, acquisition policy, or online updating strategy before committing to expensive label collection. Developing and validating these prospective diagnostics constitutes promising future work, particularly given their dependence on the quality and representativeness of the available auxiliary data.

Other limitations and future directions Our experiments deliberately focused on binary outcomes, standardized the underlying ML predictors, and, for PAI and SAI, the acquisition heuristics, to facilitate interpretable comparisons. However, a growing literature investigates optimized and robust acquisition policies for Active Statistical Inference (Ma & Candès, 2026; Li et al., 2026), and alternative ML models and online updating procedures could yield different method-specific relative efficiencies. Rather than extensively tuning each method’s acquisition policy and ML model, our goal was to understand how effectively the competing inferential procedures leverage comparable predictive information. Exploring alternative, potentially domain-specific acquisition policies and online learning strategies for PAI therefore constitutes an important direction for future work.

Furthermore, the severe undercoverage observed on the ULB dataset highlights the challenges of MLassisted adaptive inference for rare events at finite labeling budgets. Other promising directions include extending our empirical investigations to continuous outcomes and superpopulation settings, developing PAI for more general estimands such as M-estimation, and exploring further domain-specific applications.

More broadly, we hope this work provides a foundation for developing statistically reliable, label-efficient inference methods that make effective use of the increasingly abundant ML predictions and covariate information available in modern applications.

## Acknowledgments

We thank Kevin Luo, Alexandru Lopotenco, Etaash Katiyar, Valérie Ho, Noah Cowan, and Ginnie Ma for their many insightful comments, objective outside advice, and helpful discussions. S.W. and Y.N. were both supported by the National Science Foundation Graduate Research Fellowship Program. E.J.C. was supported by the Office of Naval Research grant N000014-20-1-2157. We also gratefully acknowledge support from Arena Intelligence’s Academic Partnerships Program. This work used Marlowe (Kapfer et al., 2025), Stanford University’s GPU-based Computational Instrument, supported by Stanford HAI and Stanford Research Computing. We thank Stanford University and the Stanford Research Computing Center for providing the computational resources and support that contributed to these research results.

## References

Anastasios N Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I Jordan, and Tijana Zrnic. Prediction-Powered Inference. Science, 382(6671):669–674, 2023.

Anastasios N Angelopoulos, Jacob Eisenstein, Jonathan Berant, Alekh Agarwal, and Adam Fisch. Costoptimal Active AI Model Evaluation. arXiv preprint arXiv:2506.07949, 2025.

Anastasios N Angelopoulos, John C Duchi, and Tijana Zrnic. PPI++: Efficient Prediction-Powered Inference. The Annals of Applied Statistics, 20(3):2235–2252, 2026.

Kianoosh Ashouritaklimi, Valentin Kilian, Daolang Huang, Tom Rainforth, and François Caron. Prediction-Powered Active Testing. arXiv preprint arXiv:2607.08347, 2026.

Bernard Bercu, Antoine Godichon, and Bruno Portier. An Efficient Stochastic Newton Algorithm for Parameter Estimation in Logistic Regressions. SIAM Journal on Control and Optimization, 58(1):348–367, 2020.

Isabell Bludau, Sander Willems, Wen-Feng Zeng, Maximilian T Strauss, Fynn M Hansen, Maria C Tanzer, Ozge Karayel, Brenda A Schulman, and Matthias Mann. The Structural Context of Posttranslational Modifications at a Proteome-Wide Scale. PLoS biology, 20(5):e3001636, 2022.

Andrea Dal Pozzolo, Olivier Caelen, Reid A Johnson, and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In SSCI, pp. 159–166, 2015.

Thomas Diciccio and Bradley Efron. More Accurate Confidence Intervals in Exponential Families. Biometrika, 79(2):231–245, 1992.

Frances Ding, Moritz Hardt, John Miller, and Ludwig Schmidt. Retiring Adult: New Datasets for Fair Machine Learning. Advances in Neural Information Processing Systems, 34, 2021.

Bradley Efron. Better Bootstrap Confidence Intervals. Journal of the American Statistical Association, 82(397): 171–185, 1987.

Sebastian Farquhar, Yarin Gal, and Tom Rainforth. On Statistical Bias in Active Learning: How and When to Fix It. arXiv preprint arXiv:2101.11665, 2021.

Clémentine Fourrier, Nathan Habib, Alina Lozovskaya, Konrad Szafer, and Thomas Wolf. Open LLM Leaderboard v2. https://huggingface.co/spaces/open-llm-leaderboard/open\_llm\_ leaderboard, 2024.

Kristina Gligori´c, Tijana Zrnic, Cinoo Lee, Emmanuel Candes, and Dan Jurafsky. Can Unconfident LLM Annotations be Used for Confident Conclusions? In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3514–3533, 2025.

Peter Hall and Christopher C Heyde. Martingale Limit Theory and its Application. Academic Press, 2014.

Max Hamilton, Jinlin Lai, Wenlong Zhao, Subhransu Maji, and Daniel Sheldon. Active Measurement: Efficient Estimation at Scale. Advances in Neural Information Processing Systems, 38:145769–145807, 2026.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring Mathematical Problem Solving With the MATH Dataset. Advances in Neural Information Processing Systems, 2021.

Craig Kapfer, Kurt Stine, Balasubramanian Narasimhan, Christopher Mentzel, and Emmanuel Candès. Marlowe: Stanford’s GPU-based Computational Instrument, 2025. URL https://doi.org/10.5281/ zenodo.14751899.

Jannik Kossen, Sebastian Farquhar, Yarin Gal, and Tom Rainforth. Active Testing: Sample-Efficient Model Evaluation. In International Conference on Machine Learning, pp. 5753–5763. PMLR, 2021.

Puheng Li, Tijana Zrnic, and Emmanuel Candes. Robust Sampling for Active Statistical Inference. Advances in Neural Information Processing Systems, 38:68727–68756, 2026.

Virginia L Ma and Emmanuel J Candès. Optimized Labeling Resource Allocation for Prediction-Assisted Inference via OPAL. arXiv preprint arXiv:2606.03211, 2026.

Aidar Myrzakhan, Sondos Mahmoud Bsharat, and Zhiqiang Shen. Open-LLM-Leaderboard: From Multi-Choice to Open-Style Questions for LLMs Evaluation, Benchmark, and Arena. arXiv preprint arXiv:2406.07545, 2024.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In First Conference on Language Modeling, 2024.

Zayne Rea Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning. In The Twelfth International Conference on Learning Representations, 2024.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, et al. Challenging Big-Bench Tasks and Whether Chain-of-Thought Can Solve Them. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 13003–13051, 2023.

Elad Tolochinsky, Yaniv Tenzer, and Yaniv Romano. Valid Best-Model Identification for LLM Evaluation via Low-Rank Factorization. arXiv preprint arXiv:2605.10405, 2026.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

Skyler Wu, Yash Nair, and Emmanuel J Candés. Efficient Evaluation of LLM Performance with Statistical Guarantees. arXiv preprint arXiv:2601.20251, 2026.

Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. MoleculeNet: A Benchmark for Molecular Machine Learning. Chemical Science, 9(2):513–530, 2018.

Limin Yang, Arridhana Ciptadi, Ihar Laziuk, Ali Ahmadzadeh, and Gang Wang. BODMAS: An Open Dataset for Learning based Temporal Analysis of PE Malware. In 4th Deep Learning and Security Workshop, 2021.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-Following Evaluation for Large Language Models. CoRR, 2023.

Tijana Zrnic and Emmanuel J Candès. Cross-Prediction-Powered Inference. Proceedings of the National Academy ofSciences, 121(15):e2322083121, 2024.

Tijana Zrnic and Emmanuel Candès. Active Statistical Inference. In International Conference on Machine Learning, pp. 62993–63010. PMLR, 2024.

## A Proofs and other formulas

## A.1 Regularity assumptions for Theorem 1

We enumerate some mild regularity conditions for Theorem 1: Assumptions 4-5 are standard conditions for martingale central limit theorems (c.f. Hall & Heyde (2014, Corollary 3.2)) while Assumption 6 is mild and used to facilitate consistency of our variance estimator. Define $\Delta _ { t } : = \phi _ { t } - \theta .$

Assumption 4 (Variance stabilization). The average conditional variance converges in probability to a positive deterministic limit $\sigma ^ { 2 }$ , which does not depend on m:

$$
\frac { 1 } { n } \sum _ { t = 1 } ^ { n } V a r \left( \Delta _ { t } \mid \mathcal { F } _ { t - 1 } \right) \stackrel { p } { \to } \sigma ^ { 2 } .
$$

Assumption 5 (Lindeberg condition). The triangular array of problem instances in conjunction with the sampling policies $q _ { t } ( \cdot )$ obey the Lindeberg condition. Specifically, for every $\epsilon > 0$

$$
\frac { 1 } { n } \sum _ { t = 1 } ^ { n } \mathbb { E } [ \Delta _ { t } ^ { 2 } \mathbf { 1 } ( | \Delta _ { t } | > \epsilon \sqrt { n } ) | \mathcal { F } _ { t - 1 } ] \stackrel { p } {  } 0 .
$$

Assumption 6 (Uniform fourth-moment control for weighted residuals). There exists a deterministic constant $C < \infty$ such that

$$
\operatorname* { s u p } _ { m \geq 1 } \operatorname* { s u p } _ { 1 \leq t \leq n _ { m } } \mathbb { E } \left[ \left( { \frac { 1 } { N } } { \frac { y _ { I _ { t } } - { \hat { f } } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) } { q _ { t } ( I _ { t } ) } } \right) ^ { 4 } \right] \leq C .
$$

## A.2 Proof of Theorem 1

Proof. We begin by showing that the $\Delta _ { t }$ are martingale differences—that their cumulative sums form a martingale. By definition of $\mathcal { F } _ { t - 1 }$ and $q _ { t } ( \cdot ) : = P ( I _ { t } = \cdot \mid \mathcal { F } _ { t - 1 } )$ , with division by $q _ { t } ( \cdot )$ well-defined by assumption,

$$
\begin{array} { r } { \mathbb { E } \left[ \Delta _ { t } \mid \mathcal { F } _ { t - 1 } \right] = \mathbb { E } \left[ \frac { 1 } { N } \left( \displaystyle \sum _ { i \in \mathcal { O } _ { t - 1 } } y _ { i } + \displaystyle \sum _ { i \notin \mathcal { O } _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) + \displaystyle \frac { 1 } { N } \frac { y _ { t _ { i } } - \hat { f } ^ { ( t - 1 ) } \big ( x _ { t _ { i } } \big ) } { \mathcal { G } _ { t } ( t _ { i } ) } - \theta \mid \mathcal { F } _ { t - 1 } \right] } \\ { = \displaystyle \frac { 1 } { N } \displaystyle \sum _ { i \in \mathcal { O } _ { t - 1 } } y _ { i } + \frac { 1 } { N } \left( \displaystyle \sum _ { i \notin \mathcal { O } _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) + \mathbb { E } \left[ \frac { 1 } { N } \frac { y _ { t _ { i } } - \hat { f } ^ { ( t - 1 ) } \big ( x _ { t _ { i } } \big ) } { \mathcal { G } _ { t } \left( T _ { t } \right) } \mid \mathcal { F } _ { t - 1 } \right] - \theta } \\ { = \displaystyle \frac { 1 } { N } \displaystyle \sum _ { i \in \mathcal { O } _ { t - 1 } } y _ { i } + \frac { 1 } { N } \left( \displaystyle \sum _ { i \notin \mathcal { O } _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) + \displaystyle \frac { 1 } { N } \sum _ { i \notin \mathcal { O } _ { s - 1 } } \left( \frac { y _ { i } - \hat { f } ^ { ( t - 1 ) } \big ( x _ { i } \big ) } { \mathcal { G } _ { t } \big ( \hat { x } \big ) } \right) q _ { t } ( i ) - \theta } \\ { = \displaystyle \frac { 1 } { N } \displaystyle \sum _ { i \in \mathcal { O } _ { t - 1 } } \theta = 0 , } \end{array}
$$

which implies that $( \Delta _ { t } , \mathcal { F } _ { t } )$ is a martingale difference array. Now, it immediately follows that

$$
\mathbb { E } \left[ \hat { \theta } _ { n } - \theta \right] = \mathbb { E } \left[ \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \phi _ { t } - \theta \right] = \mathbb { E } \left[ \frac { 1 } { n } \sum _ { t = 1 } ^ { n } ( \phi _ { t } - \theta ) \right] = \mathbb { E } \left[ \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \Delta _ { t } \right] = 0 ,
$$

implying that $\hat { \theta } _ { n }$ is unbiased for θ at every finite n. Now, by Assumptions 4-5, the martingale central limit theorem as presented in Corollary 3.2 of Hall & Heyde (2014) tells us that

$$
{ \sqrt { n } } \left( { \hat { \theta } } _ { n } - \theta \right) { \xrightarrow { d } } { \mathcal { N } } \left( 0 , \sigma ^ { 2 } \right) .
$$

To compute a consistent estimator for $\sigma ^ { 2 } .$ , we proceed by computing the martingale increment conditional variances. By definition of $\mathcal { F } _ { t - 1 }$ , we have

$$
\begin{array} { r } { \mathrm { V a r } \left( \Delta _ { t } \mid \mathcal { F } _ { t - 1 } \right) } \\ { = \mathrm { V a r } \left( \displaystyle \frac { 1 } { N } \left( \displaystyle \sum _ { y \in \mathcal { O } _ { t - 1 } } \mathcal { H } + \displaystyle \sum _ { y \in \mathcal { O } _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) + \displaystyle \frac { 1 } { N } \frac { y _ { t } - \hat { f } ^ { ( t - 1 ) } \big ( x _ { t } \big ) } { q ( I _ { t } ) } - \theta \mid \mathcal { F } _ { t - 1 } \right) } \\ { = \mathrm { V a r } \left( \displaystyle \frac { 1 } { N } \frac { y _ { t } - \hat { f } ^ { ( t - 1 ) } \big ( x _ { t } \big ) } { q ( I _ { t } ) } \mid \mathcal { F } _ { t - 1 } \right) } \\ { = \mathbb { E } \left[ \left( \displaystyle \frac { 1 } { N } \frac { y _ { t } - \hat { f } ^ { ( t - 1 ) } \big ( x _ { t } \big ) } { q ( I _ { t } ) } \right) ^ { 2 } \mid \mathcal { F } _ { t - 1 } \right] - \left( \mathbb { E } \left[ \displaystyle \frac { 1 } { N } \frac { y _ { t } - \hat { f } ^ { ( t - 1 ) } \big ( x _ { t } \big ) } { q ( I _ { t } ) } \mid \mathcal { F } _ { t - 1 } \right] \right) ^ { 2 } } \\ { = \displaystyle \frac { 1 } { N ^ { 2 } } \displaystyle \sum _ { i \neq j \in \mathcal { O } _ { t - 1 } } \left( \frac { y _ { i } - \hat { f } ^ { ( t - 1 ) } \big ( x _ { i } \big ) } { q ( i ) } \right) ^ { 2 } - \frac { 1 } { N ^ { 2 } } \left( \displaystyle \sum _ { i \neq j \in \mathcal { O } _ { t - 1 } } \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) \right) ^ { 2 } . } \end{array}
$$

By Assumption 4 and substitution, we know that

$$
\begin{array} { r } { \frac { 1 } { n } \displaystyle \sum _ { t = 1 } ^ { n } \mathrm { V a r } ( \Delta _ { t } \mid \mathcal { F } _ { t - 1 } ) = \frac { 1 } { n } \displaystyle \sum _ { t = 1 } ^ { n } \left( \frac { 1 } { N ^ { 2 } } \displaystyle \sum _ { i \notin O _ { t - 1 } } \frac { \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) ^ { 2 } } { q _ { t } ( i ) } - \frac { 1 } { N ^ { 2 } } \left( \displaystyle \sum _ { i \notin O _ { t - 1 } } \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) \right) ^ { 2 } \right) } \\ { = \displaystyle \frac { 1 } { n N ^ { 2 } } \displaystyle \sum _ { t = 1 } ^ { n } \left( \displaystyle \sum _ { i \notin O _ { t - 1 } } \frac { \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) ^ { 2 } } { q _ { t } ( i ) } \right) - \frac { 1 } { n N ^ { 2 } } \displaystyle \sum _ { t = 1 } ^ { n } \left( \displaystyle \sum _ { i \notin O _ { t - 1 } } \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) \right) ^ { 2 } } \\ { : = A _ { n } - B _ { n } \stackrel { p } { \to } \sigma ^ { 2 } . } \end{array}
$$

To show that $\hat { \sigma } _ { n } ^ { 2 } \overset { p } { \to } \sigma ^ { 2 } .$ , it suffices to show that (a) $\hat { A } _ { n } - A _ { n } = o _ { p } ( 1 )$ and (b) $\hat { B } _ { n } - B _ { n } = o _ { p } ( 1 )$ . Beginning with the first statement,

$$
\begin{array} { r } { \hat { A } _ { n } - A _ { n } = \displaystyle \frac { 1 } { n N ^ { 2 } } \sum _ { t = 1 } ^ { n } \frac { \left( y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } \left( x _ { I _ { t } } \right) \right) ^ { 2 } } { q _ { t } \left( I _ { t } \right) ^ { 2 } } - \frac { 1 } { n N ^ { 2 } } \sum _ { t = 1 } ^ { n } \left( \sum _ { i \notin O _ { t - 1 } } \frac { \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } \left( x _ { i } \right) \right) ^ { 2 } } { q _ { t } \left( i \right) } \right) } \\ { = \displaystyle \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \frac { 1 } { N ^ { 2 } } \left( \frac { \left( y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } \left( x _ { I _ { t } } \right) \right) ^ { 2 } } { q _ { t } \left( I _ { t } \right) ^ { 2 } } - \sum _ { i \notin O _ { t - 1 } } \frac { \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } \left( x _ { i } \right) \right) ^ { 2 } } { q _ { t } \left( i \right) } \right) : = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } C _ { t } . } \end{array}
$$

Because we have

$$
\begin{array} { r } { \mathbb { E } [ C _ { t } \mid \mathcal { F } _ { t - 1 } ] = \mathbb { E } \left[ \frac { 1 } { N ^ { 2 } } \left( \frac { \left( y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } \left( x _ { I _ { t } } \right) \right) ^ { 2 } } { q _ { t } \left( I _ { t } \right) ^ { 2 } } - \underset { i \neq O _ { t - 1 } } { \sum } \frac { \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } \left( x _ { i } \right) \right) ^ { 2 } } { q _ { t } \left( i \right) } \right) \mid \mathcal { F } _ { t - 1 } \right] } \\ { = \mathbb { E } \left[ \frac { 1 } { N ^ { 2 } } \left( \frac { \left( y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } \left( x _ { I _ { t } } \right) \right) ^ { 2 } } { q _ { t } \left( I _ { t } \right) ^ { 2 } } \right) \mid \mathcal { F } _ { t - 1 } \right] - \frac { 1 } { N ^ { 2 } } \left( \underset { i \neq O _ { t - 1 } } { \sum } \frac { \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } \left( x _ { i } \right) \right) ^ { 2 } } { q _ { t } \left( i \right) } \right) } \\ { = \underset { i \neq O _ { t - 1 } } { \sum } \frac { 1 } { N ^ { 2 } } \left( \frac { \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } \left( x _ { i } \right) \right) ^ { 2 } } { q _ { t } \left( i \right) ^ { 2 } } \right) q _ { t } ( i ) - \frac { 1 } { N ^ { 2 } } \left( \underset { i \neq O _ { t - 1 } } { \sum } \frac { \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } \left( x _ { i } \right) \right) ^ { 2 } } { q _ { t } \left( i \right) } \right) = 0 , } \end{array}
$$

we can deduce that $\left( C _ { t } , \mathcal { F } _ { t } \right)$ is a martingale difference array. By orthogonality of martingale increments $( \mathrm { i . e . }$ tower property), we further find that

$$
\mathbb { E } \left[ \left( \hat { A } _ { n } - A _ { n } \right) ^ { 2 } \right] = \mathbb { E } \left[ \left( \frac { 1 } { n } \sum _ { t = 1 } ^ { n } C _ { t } \right) ^ { 2 } \right] = \frac { 1 } { n ^ { 2 } } \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ C _ { t } ^ { 2 } \right] .
$$

Now, because $( a - b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$

$$
\begin{array} { r l } & { \mathbb { E } \left[ C _ { t } ^ { \alpha } \right] \leq \mathbb { E } \left[ 2 \left( \frac { 1 } { N ^ { 2 } } \frac { \left( \frac { \partial _ { t } } { \partial t } - \widehat f ^ { ( \varepsilon - 1 ) } ( x _ { t } ) \right) ^ { 2 } } { \mathcal { H } ( t ) } \right) ^ { 2 } \right] + 2 \left( \frac { 1 } { N ^ { 2 } } \displaystyle \sum _ { i \notin Q _ { t - 1 } } \frac { \left( \frac { \partial _ { t } } { \partial t } - \widehat f ^ { ( \varepsilon - 1 ) } ( x _ { t } ) \right) ^ { 2 } } { \mathcal { H } ( t ) } \right) ^ { 2 } \right] } \\ & { = 2 \mathbb { E } \left[ \left( \frac { 1 } { N } \frac { \left( \widehat y _ { t } - \widehat f ^ { ( \varepsilon - 1 ) } ( x _ { t } ) \right) } { \mathcal { H } ( t ) } \right) ^ { 4 } \right] + 2 \mathbb { E } \left[ \left( \mathbb { E } \left[ \frac { 1 } { N ^ { 2 } } \left( \frac { \left( y _ { t } - \widehat { f } ^ { ( \varepsilon - 1 ) } ( x _ { t } ) \right) } { \mathcal { H } ( t ) ^ { 2 } } \right) ^ { 2 } | \mathcal { F } _ { t - 1 } \right] \right) ^ { 2 } \right] } \\ & { = 2 \mathbb { E } \left[ \left( \frac { 1 } { N } \frac { \left( \partial _ { t } - \widehat f ^ { ( \varepsilon - 1 ) } ( x _ { t } ) \right) } { \mathcal { H } ( t ) } \right) ^ { 4 } \right] + 2 \mathbb { E } \left[ \mathbb { E } \left[ \left( \frac { 1 } { N } \frac { \left( \partial _ { t } - \widehat { f } ^ { ( \varepsilon - 1 ) } ( x _ { t } ) \right) } { \mathcal { H } ( t ) } \right) ^ { 4 } | \mathcal { F } _ { t - 1 } \right] \right] } \\ &  \qquad = 4 \mathbb { E } \left[ \left( \frac { 1 } { N } \frac { \left( \partial _ { t } - \widehat { f } ^ { ( \varepsilon - 1 ) } ( x _ { t } ) \right) } { \mathcal { H } ( t ) } \right) ^  \end{array}
$$

with the last step by Assumption 6. As such, we deduce that E $: [ ( { \hat { A } } _ { n } - A _ { n } ) ^ { 2 } ]  0 .$ , which implies that $\hat { A } _ { n } - A _ { n } = o _ { p } ( 1 )$ . Working on the second term, we first observe that

$$
\begin{array} { r } { B _ { n } = \displaystyle \frac { 1 } { n N ^ { 2 } } \sum _ { t = 1 } ^ { n } \left( \sum _ { i \notin \theta _ { t - 1 } } \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) \right) ^ { 2 } = \displaystyle \frac { 1 } { n N ^ { 2 } } \sum _ { t = 1 } ^ { n } \left( N \theta - \sum _ { i \in O _ { t - 1 } } y _ { i } - \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) ^ { 2 } } \\ { = \displaystyle \frac { 1 } { n N ^ { 2 } } \sum _ { t = 2 } ^ { n } \left( N \theta - \sum _ { i \in O _ { t - 1 } } y _ { i } - \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) ^ { 2 } + \frac { 1 } { n N ^ { 2 } } \left( N \theta - \sum _ { i = 1 } ^ { N } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) ^ { 2 } . } \end{array}
$$

With this alternative form, and using the equality that $a ^ { 2 } - b ^ { 2 } = ( a - b ) ^ { 2 } + 2 b ( a - b )$ to combine similar terms, we have the following:

$$
\begin{array} { r } { \tilde { \partial } _ { \theta } - \mathcal { B } _ { \infty } - \frac { 1 } { \varepsilon ^ { 0 / 2 } } \frac { 1 } { \varepsilon ^ { 0 / 2 } } ( \mathcal { K } \tilde { \partial } _ { \theta } - \sum _ { i \in \mathcal { K } _ { i } } - \sum _ { \ell \in \mathcal { K } _ { i } } \frac { 1 } { \varepsilon ^ { 0 / 2 } } ( \varepsilon ^ { 0 } - 1 ) ) ^ { 2 } } \\ { - \frac { 1 } { \varepsilon ^ { 0 / 2 } } \frac { 1 } { \varepsilon ^ { 0 / 2 } } ( \mathbf { x } - \sum _ { \ell \in \mathcal { K } _ { i } } \sum _ { \ell = \mathcal { K } _ { i } } \frac { 1 } { \varepsilon ^ { 0 / 2 } } ( \varepsilon ^ { 0 } - 1 ) ) ^ { 2 } - \frac { 1 } { \varepsilon ^ { 0 / 2 } } ( \mathbf { x } - \sum _ { \ell } \tilde { \partial } _ { \theta } ( \varepsilon _ { 0 } ) ) ^ { 2 } ) } \\ { - \frac { 1 } { \varepsilon ^ { 0 / 2 } } \frac { 1 } { \varepsilon ^ { 0 / 2 } } ( \mathcal { K } \tilde { \partial } _ { \theta } - \sum _ { \ell \in \mathcal { K } _ { i } } \sum _ { \ell \in \mathcal { K } _ { i } } \frac { 1 } { \varepsilon ^ { 0 / 2 } } \varepsilon ^ { 0 / 2 } \varepsilon ^ { 0 / 2 } ) } \\ { - \frac { 1 } { \varepsilon ^ { 0 / 2 } } \frac { 1 } { \varepsilon ^ { 0 / 2 } } [ ( \mathcal { K } \tilde { \partial } _ { \varepsilon - \mathcal { K } _ { i } } - \sum _ { \ell \in \mathcal { K } _ { i } } \frac { 1 } { \varepsilon ^ { 0 / 2 } } \varepsilon ^ { 0 / 2 } ) \varepsilon ^ { 0 / 2 } ] ^ { 2 } - ( \mathcal { K } ^ { 0 } - \sum _ { \ell \in \mathcal { K } _ { i } } \frac { 1 } { \varepsilon ^ { 0 / 2 } } \varepsilon ^ { 0 / 2 } \varepsilon ) ^ { 2 } ] } \\  - \frac { 1 }  \varepsilon ^ \end{array}
$$

Taking absolute values, we have

$$
\begin{array} { r } { | \hat { B } _ { n } - B _ { n } | \le \underbrace { \displaystyle \frac { 1 } { n } \sum _ { t = 2 } ^ { n } \left( \hat { \theta } _ { t - 1 } - \theta \right) ^ { 2 } } _ { T _ { 1 } } + \underbrace { \frac { 2 } { n } \sum _ { t = 2 } ^ { n } \left| \theta - \frac { \sum _ { i \in O _ { t - 1 } } y _ { i } + \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) } { N } \right| \left| \hat { \theta } _ { t - 1 } - \theta \right| } _ { T _ { 2 } } } \\ { + \underbrace { \frac { 1 } { n } \left( \theta - \frac { \sum _ { i = 1 } ^ { N } \hat { f } ^ { ( 0 ) } ( x _ { i } ) } { N } \right) ^ { 2 } } _ { T _ { 3 } } . } \end{array}
$$

Working on $T _ { 1 }$ , we have the following because the $\Delta _ { s }$ are martingale differences and by orthogonality of martingale increments:

$$
{ \widehat { \theta } } _ { t - 1 } - \theta = { \frac { 1 } { t - 1 } } \sum _ { s = 1 } ^ { t - 1 } \Delta _ { s } \implies \mathbb { E } \left[ \left( { \widehat { \theta } } _ { t - 1 } - \theta \right) ^ { 2 } \right] = { \frac { 1 } { ( t - 1 ) ^ { 2 } } } \sum _ { s = 1 } ^ { t - 1 } \mathbb { E } \left[ \Delta _ { s } ^ { 2 } \right] .
$$

Now, by Jensen’s Inequality, the fact that $( a - b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 } .$ , and the following fact:

$$
\mathbb { E } \left[ \frac { 1 } { N } \frac { y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) } { q _ { t } ( I _ { t } ) } \mid \mathcal { F } _ { t - 1 } \right] = \frac { 1 } { N } \sum _ { i \notin O _ { t - 1 } } \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) ,
$$

we have the following, noting that $\begin{array} { r } { \theta - \frac { 1 } { N } \sum _ { i \in O _ { s - 1 } } y _ { i } = \frac { 1 } { N } \sum _ { i \notin O _ { s - 1 } } y _ { i } \colon } \end{array}$

$$
\begin{array} { r l } & { \quad \geq \Xi \{ \frac { 1 } { \tau } ( \displaystyle \sum _ { i = 1 } ^ { j } - \sum _ { j = 1 } ^ { j - 1 } ( \frac { 1 } { \tau } \sum _ { i = 2 } ^ { j - 1 } - \sum _ { j = 1 } ^ { j - 1 } \rho ( x _ { i , j } ) ) + \frac { 1 } { \tau } ( x _ { i , j } ) - x _ { i , j } ^ { \tau } x _ { i , j } ) \} ^ { \tau } } \\ & { \leq \mathbb { E } [ ( \frac { 1 } { \tau } x _ { i , j } - x _ { i , j + 1 } ^ { \tau } \nu _ { i , j - 2 } ) ] ^ { \tau } \{ 1 \geq ( \displaystyle \sum _ { i = 1 } ^ { j } ( \sum _ { j = 1 } ^ { j } \nu _ { i , j } ) \sum _ { i = 1 } ^ { j - 1 } ( \sum _ { j = 1 } ^ { j - 1 } \nu _ { i , j } x _ { i , j } ) ) ^ { \tau } \} } \\ & { \qquad \quad \leq 2 \mathbb { E } [ ( \frac { 1 } { \tau } x _ { i , j } - x _ { i , j + 1 } ^ { \tau - 1 } \nu _ { i , j - 2 } ) ^ { \tau } ] ^ { \tau } \{ 1 \geq ( \displaystyle \sum _ { i = 1 } ^ { j } \nu _ { i , j } x _ { i , j } ) \sum _ { i = 1 } ^ { j } \nu _ { i , j } \sum _ { i = 1 } ^ { j - 1 } \rho ( x _ { i , j } ) \} ^ { \tau } } \\ & { \qquad \quad - \{ \frac { 1 } { \tau } ( x _ { i , j } ) - x _ { i , j } ^ { \tau - 1 } x _ { i , j } \} ^ { \tau } [ ( x _ { i , j } ) - x _ { i , j } ] ^ { \tau } \{ 1 , \sum _ { i = 1 } ^ { j - 1 } \nu _ { i , j } x _ { i , j } \} ^ { \tau } } \\ &  \qquad \quad - \nu _ { i } \{ ( \frac { 1 }  \tau  \end{array}
$$

Thus, we conclude that

$$
\mathbb { E } [ T _ { 1 } ] = \frac { 1 } { n } \sum _ { t = 2 } ^ { n } \mathbb { E } \left[ \left( \hat { \theta } _ { t - 1 } - \theta \right) ^ { 2 } \right] = \frac { 1 } { n } \sum _ { t = 2 } ^ { n } \frac { 1 } { \left( t - 1 \right) ^ { 2 } } \sum _ { s = 1 } ^ { t - 1 } \mathbb { E } \left[ \Delta _ { s } ^ { 2 } \right] \leq \frac { 1 } { n } \sum _ { t = 2 } ^ { n } \frac { 4 C ^ { 1 / 2 } } { t - 1 } = O \left( \frac { \log n } { n } \right) .
$$

Moving on to $T _ { 3 } ,$ , we note that for any $t ,$ by definition of $\theta ,$

$$
\begin{array} { r } { \theta - \displaystyle \frac { 1 } { N } \sum _ { i \in O _ { t - 1 } } y _ { i } - \displaystyle \frac { 1 } { N } \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) = \displaystyle \frac { 1 } { N } \sum _ { i \notin O _ { t - 1 } } \left( y _ { i } - \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) } \\ { = \mathbb { E } \left[ \frac { 1 } { N } \frac { y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) } { q _ { t } ( I _ { t } ) } \mid \mathcal { F } _ { t - 1 } \right] . } \end{array}
$$

By Jensen’s inequality, we deduce that for any $t ,$

$$
\begin{array} { r } { \mathbb { E } \left[ \left( \theta - \frac { 1 } { N } \displaystyle \sum _ { i \in \mathcal { O } _ { t - 1 } } y _ { i } - \frac { 1 } { N } \displaystyle \sum _ { i \notin \mathcal { O } _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) ^ { 2 } \right] = \mathbb { E } \left[ \left( \mathbb { E } \left[ \frac { 1 } { N } \frac { y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) } { q _ { t } ( I _ { t } ) } \mid \mathcal { F } _ { t - 1 } \right] \right) ^ { 2 } \right] } \\ { \leq \mathbb { E } \left[ \mathbb { E } \left[ \left( \frac { 1 } { N } \frac { y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) } { q _ { t } ( I _ { t } ) } \right) ^ { 2 } \mid \mathcal { F } _ { t - 1 } \right] \right] = \mathbb { E } \left[ \left( \frac { 1 } { N } \frac { y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) } { q _ { t } ( I _ { t } ) } \right) ^ { 2 } \right] } \\ { \leq \left( \mathbb { E } \left[ \left( \frac { 1 } { N } \frac { y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) } { q _ { t } ( I _ { t } ) } \right) ^ { 4 } \right] \right) ^ { 1 / 2 } \leq C ^ { 1 / 2 } . } \end{array}
$$

Thus, it also follows that, corresponding to $t = 1$

$$
\mathbb { E } [ T _ { 3 } ] = \mathbb { E } \left[ \frac { 1 } { n } \left( \theta - \frac { \sum _ { i = 1 } ^ { N } \hat { f } ^ { ( 0 ) } ( x _ { i } ) } { N } \right) ^ { 2 } \right] \leq \frac { C ^ { 1 / 2 } } { n } = O \left( \frac { 1 } { n } \right) .
$$

Now, working on $T _ { 2 } ,$ by Cauchy-Schwarz, and using our above facts,

$$
\begin{array} { r l r } & { } & { \mathbb { E } [ T _ { 2 } ] = \displaystyle \frac { 2 } { n } \sum _ { t = 2 } ^ { n } \mathbb { E } \left[ \left| \theta - \frac { \sum _ { i \notin O _ { t - 1 } } y _ { i } + \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } \left( x _ { i } \right) } { N } \right| \left| \hat { \theta } _ { t - 1 } - \theta \right| \right] } \\ & { } & { \le \displaystyle \frac { 2 } { n } \sum _ { t = 2 } ^ { n } \mathbb { E } \left[ \left( \theta - \frac { \sum _ { i \in O _ { t - 1 } } y _ { i } + \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } \left( x _ { i } \right) } { N } \right) ^ { 2 } \right] ^ { 1 / 2 } \mathbb { E } \left[ \left( \hat { \theta } _ { t - 1 } - \theta \right) ^ { 2 } \right] ^ { 1 / 2 } } \\ & { } & { \le \displaystyle \frac { 2 } { n } \sum _ { t = 2 } ^ { n } \left( C ^ { 1 / 2 } \right) ^ { 1 / 2 } \left( \frac { 4 C ^ { 1 / 2 } } { t - 1 } \right) ^ { 1 / 2 } = \frac { 4 C ^ { 1 / 2 } } { n } \sum _ { t = 2 } ^ { n } \frac { 1 } { \sqrt { t - 1 } } = O \left( \frac { 1 } { \sqrt { n } } \right) . } \end{array}
$$

Putting everything together, we deduce that

$$
\mathbb { E } [ \left. \hat { B } _ { n } - B _ { n } \right. ] \leq \mathbb { E } [ T _ { 1 } ] + \mathbb { E } [ T _ { 2 } ] + \mathbb { E } [ T _ { 3 } ] = O \left( \frac { \log n } { n } \right) + O \left( \frac { 1 } { \sqrt { n } } \right) + O \left( \frac { 1 } { n } \right) \to 0 ,
$$

which implies that $\hat { B } _ { n } \stackrel { L ^ { 1 } } {  } B _ { n } ,$ which implies $\hat { B } _ { n } - B _ { n } = o _ { p } ( 1 )$ . Thus, we conclude that $\hat { \sigma } _ { n } ^ { 2 } \overset { p } { \to } \sigma ^ { 2 } .$ , which by Slutsky’s gives us our desired final result:

$$
{ \sqrt { n } } \left( { \hat { \theta } } _ { n } - \theta \right) / { \hat { \sigma } } _ { n } \ { \xrightarrow { d } } { \mathcal { N } } \left( 0 , 1 \right) .
$$

## A.3 Derivation and properties of the improved finite-sample variance estimator

In this section, we motivate, derive, and provide commentary on the improved finite-sample variance estimator $\hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 }$ introduced in Section 3.3.

## A.3.1 Inspirations from second-order Taylor expansions and the ABC and BCa bootstraps

Before starting our derivation, recall from the proof of Theorem 1 in Appendix A.2 that we assumed the following variance stabilization to facilitate our martingale CLT:

$$
{ \frac { 1 } { n } } \sum _ { t = 1 } ^ { n } \operatorname { V a r } ( \Delta _ { t } \mid { \mathcal { F } } _ { t - 1 } ) : = A _ { n } - B _ { n } { \xrightarrow { p } } \sigma ^ { 2 } ,
$$

where $\hat { A } _ { n }$ and ${ \hat { B } } _ { n }$ are the plug-in estimates for $A _ { n }$ and $B _ { n . }$ , respectively. From the aforementioned proof, we know that $\hat { A } _ { n }$ is unbiased for $A _ { n }$ because $A _ { n } - { \hat { A } } _ { n }$ is a martingale sequence. Thus, we elect to biascorrect by working on ${ \hat { B } } _ { n }$ , which is biased for $B _ { n }$ because we plugged in $\hat { \theta } _ { t - 1 }$ in place of θ. Specifically, we proceed to derive a correction term $\hat { C } _ { n }$ via a second-order Taylor expansion $\hat { B } _ { n } - B _ { n }$ around θ, inspired philosophically by the ABC and BCa bootstrap methods (c.f. Efron (1987); Diciccio & Efron (1992)). From

the aforementioned proof, we know that

$$
\begin{array} { l } { \displaystyle \hat { B } _ { n } - B _ { n } = \frac { 1 } { n N ^ { 2 } } \sum _ { t = 2 } ^ { n } \left( N \hat { \theta } _ { t - 1 } - \sum _ { i \in O _ { t - 1 } } y _ { i } - \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) ^ { 2 } } \\ { \displaystyle - \frac { 1 } { n N ^ { 2 } } \sum _ { t = 2 } ^ { n } \left( N \theta - \sum _ { i \in O _ { t - 1 } } y _ { i } - \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) ^ { 2 } } \\ { \displaystyle - \frac { 1 } { n N ^ { 2 } } \left( N \theta - \sum _ { i = 1 } ^ { N } \hat { f } ^ { ( 0 ) } ( x _ { i } ) \right) ^ { 2 } . } \end{array}
$$

Now, for $t = 1 , \ldots , n ,$ define ψ<sub>t</sub> $\begin{array} { r } { \langle \sum _ { i } \sum _ { 1 } \sum _ { i \in O _ { t - 1 } } y _ { i } + \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \rangle } \end{array}$ and $g _ { t } ( \theta ) = ( \theta - \psi _ { t } ) ^ { 2 }$ . Then, by substitution, we have the following, where $g _ { 1 } ( \theta )$ can be interpreted as a boundary term:

$$
{ \hat { B } } _ { n } - B _ { n } = { \frac { 1 } { n } } \sum _ { t = 2 } ^ { n } \left( g _ { t } ( { \hat { \theta } } _ { t - 1 } ) - g _ { t } ( \theta ) \right) - { \frac { 1 } { n } } g _ { 1 } ( \theta ) .
$$

Taylor-expanding each $g _ { t }$ around $\theta ,$ we have, after noting that $g _ { t } ^ { \prime } ( \theta ) = 2 ( \theta - \psi _ { t } )$ and that $g _ { t } ^ { \prime \prime } ( \theta ) = 2 .$ , the following exact expression as the $g _ { t }$ were originally quadratic:

$$
\begin{array} { c } { { g _ { t } ( \hat { \theta } _ { t - 1 } ) = g _ { t } ( \theta ) + g _ { t } ^ { \prime } ( \theta ) ( \hat { \theta } _ { t - 1 } - \theta ) + { \displaystyle \frac { 1 } { 2 } g _ { t } ^ { \prime \prime } } ( \theta ) ( \hat { \theta } _ { t - 1 } - \theta ) ^ { 2 } } } \\ { { = g _ { t } ( \theta ) + 2 ( \theta - \psi _ { t } ) ( \hat { \theta } _ { t - 1 } - \theta ) + ( \hat { \theta } _ { t - 1 } - \theta ) ^ { 2 } . } } \end{array}
$$

Substituting, we have the following, with our boundary term having no $\hat { \theta } _ { t - 1 }$ analog:

$$
\hat { B } _ { n } - B _ { n } = \underbrace { \frac { 1 } { n } \sum _ { t = 2 } ^ { n } ( \hat { \theta } _ { t - 1 } - \theta ) ^ { 2 } } _ { \mathrm { s e c o n d - o r d e r T a y l o r c u r v a t u r e \ t e r m } } + \underbrace { \frac { 1 } { n } \sum _ { t = 2 } ^ { n } 2 ( \theta - \psi _ { t } ) ( \hat { \theta } _ { t - 1 } - \theta ) } _ { \mathrm { f i r s t - o r d e r T a y l o r t e r m } } - \underbrace { \frac { 1 } { n } g _ { 1 } ( \theta ) } _ { \mathrm { b o u n d a r y t e r m } } .
$$

Because the purpose of a second-order Taylor correction is to correct the second-order curvature term, we thus explore a correction ${ \hat { C } } _ { n }$ that well-approximates $\begin{array} { r } { \frac { 1 } { n } \sum _ { t = 2 } ^ { n } ( \widehat { \theta } _ { t - 1 } - \theta ) ^ { 2 } } \end{array}$ . In particular, we do not attempt to bias-correct the first-order Taylor term because it is (a) sign-indefinite; (b) generally not safely estimable without plugging in $\hat { \theta }$ for θ (and risk introducing another layer of finite-sample bias); and (c) better treated as a component of the stochastic error.

## A.3.2 Constructing the correction

Recall our notation in Theorem 1 of

$$
\Delta _ { t } : = \phi _ { t } - \theta , \quad \mathrm { w i t h } \quad \phi _ { t } : = \frac { 1 } { N } \left( \sum _ { i \in O _ { t - 1 } } y _ { i } + \sum _ { i \notin O _ { t - 1 } } \hat { f } ^ { ( t - 1 ) } ( x _ { i } ) \right) + \frac { 1 } { N } \frac { y _ { I _ { t } } - \hat { f } ^ { ( t - 1 ) } ( x _ { I _ { t } } ) } { q _ { t } ( I _ { t } ) } ,
$$

recalling that $\begin{array} { r } { \hat { \theta } _ { t - 1 } : = \frac { 1 } { t - 1 } \sum _ { s = 1 } ^ { t - 1 } \phi _ { s } } \end{array}$ . Since $\begin{array} { r } { \hat { \theta } _ { t - 1 } = \theta + \frac { 1 } { t - 1 } \sum _ { s = 1 } ^ { t - 1 } \Delta } \end{array}$ <sub>s</sub> and the $\Delta _ { s }$ are martingale differences, by orthogonality we have the following expectation of the target second-order Taylor curvature:

$$
\mathbb { E } [ \frac { 1 } { n } \sum _ { t = 2 } ^ { n } ( \hat { \theta } _ { t - 1 } - \theta ) ^ { 2 } ] = \frac { 1 } { n } \sum _ { t = 2 } ^ { n } \frac { 1 } { ( t - 1 ) ^ { 2 } } \sum _ { s = 1 } ^ { t - 1 } \mathbb { E } [ v _ { s } ] = \frac { 1 } { n } \sum _ { s = 1 } ^ { n - 1 } \mathbb { E } [ v _ { s } ] w _ { s } , \quad \mathrm { w i t h } \quad w _ { s } : = \sum _ { k = s } ^ { n - 1 } \frac { 1 } { k ^ { 2 } } ,\tag{24}
$$

where the second equality exchanges the order of summation, and reusing computations from our proof of Theorem 1 in Appendix $\mathsf { A } . 2 ,$ we have per-step conditional variances $v _ { s }$ as follows:

$$
\begin{array} { r l } & { v _ { 5 } : = \operatorname { V a r } ( \Delta _ { s } \mid \mathcal { F } _ { \varepsilon - 1 } ) } \\ & { \quad = \frac { 1 } { N ^ { 2 } } \displaystyle \sum _ { i \not = \theta _ { 0 - 1 } } ^ { \infty } \frac { \left( \underline { { \partial } } _ { u } - \hat { f } ^ { ( s - 1 ) } ( x _ { i } ) \right) ^ { 2 } } { \theta _ { 0 } ( s ) } - \frac { 1 } { N ^ { 2 } } \left( \displaystyle \sum _ { i \not = \theta _ { 0 - 1 } } \left( \mathbb { E } _ { u } - \hat { f } ^ { ( s - 1 ) } ( x _ { i } ) \right) \right) ^ { 2 } } \\ & { \quad = \frac { 1 } { N ^ { 2 } } \displaystyle \sum _ { i \not = \theta _ { 0 - 1 } } ^ { \infty } \frac { \left( \underline { { \partial } } _ { u } - \hat { f } ^ { ( s - 1 ) } ( x _ { i } ) \right) ^ { 2 } } { \theta _ { 0 } ( s ) } - \left( \theta - \frac { 1 } { N } \displaystyle \sum _ { i \in \theta _ { 0 - 1 } } y _ { i } - \frac { 1 } { N } \displaystyle \sum _ { i \not = \theta _ { 0 - 1 } } \hat { f } ^ { ( s - 1 ) } ( x _ { i } ) \right) ^ { 2 } } \\ & { \quad = \frac { 1 } { N ^ { 2 } } \displaystyle \sum _ { i \not = \theta _ { 0 - 1 } } ^ { \infty } \frac { \left( y _ { i } - \hat { f } ^ { ( s - 1 ) } ( x _ { i } ) \right) ^ { 2 } } { \theta _ { 0 } ( s ) } - ( \theta - \psi _ { s } ) ^ { 2 } . } \\ & { \quad = \frac { 1 } { N ^ { 2 } } \displaystyle \sum _ { i \not = \theta _ { 0 - 1 } } ^ { \infty } \frac { \left( \mathbb { J } _ { u } - \hat { f } ^ { ( s - 1 ) } ( x _ { i } ) \right) ^ { 2 } } { \theta _ { 0 } ( s ) } - ( \theta - \psi _ { s } ) ^ { 2 } . } \end{array}
$$

Ideally, our oracle correction term would just be $\begin{array} { r } { \hat { C } _ { n } ^ { * } = \frac { 1 } { n } \sum _ { s = 1 } ^ { n - 1 } v _ { s } w _ { s } } \end{array}$ , but it is clear from above that $v _ { s }$ is unknown as it involves $\theta .$ To find a data-driven proxy for the above $v _ { s } ,$ , we note that letting

$$
\zeta _ { s } = \frac { 1 } { N } \frac { y _ { I _ { s } } - \hat { f } ^ { ( s - 1 ) } ( x _ { I _ { s } } ) } { q _ { s } ( I _ { s } ) } ,
$$

we have

$$
\begin{array} { r l r } & { } & { \mathbb { E } [ \zeta _ { s } ^ { 2 } \mid \mathcal { F } _ { s - 1 } ] = \displaystyle \sum _ { i \notin O _ { s - 1 } } \left( \frac { 1 } { N } \frac { y _ { i } - \hat { f } ^ { ( s - 1 ) } ( x _ { i } ) } { q _ { s } ( i ) } \right) ^ { 2 } P ( I _ { s } = i \mid \mathcal { F } _ { s - 1 } ) } \\ & { } & { \quad = \displaystyle \sum _ { i \notin O _ { s - 1 } } \left( \frac { 1 } { N } \frac { y _ { i } - \hat { f } ^ { ( s - 1 ) } ( x _ { i } ) } { q _ { s } ( i ) } \right) ^ { 2 } q _ { s } ( i ) = \frac { 1 } { N ^ { 2 } } \displaystyle \sum _ { i \notin O _ { s - 1 } } \frac { \left( y _ { i } - \hat { f } ^ { ( s - 1 ) } ( x _ { i } ) \right) ^ { 2 } } { q _ { s } ( i ) } . } \end{array}
$$

Thus, our derived additive correction is

$$
\hat { C } _ { n } : = \frac { 1 } { n } \sum _ { s = 1 } ^ { n - 1 } \left( \frac { 1 } { N } \frac { y _ { I _ { s } } - \hat { f } ^ { ( s - 1 ) } ( x _ { I _ { s } } ) } { q _ { s } ( I _ { s } ) } \right) ^ { 2 } w _ { s } , \quad \mathrm { w i t h } \quad w _ { s } : = \sum _ { k = s } ^ { n - 1 } \frac { 1 } { k ^ { 2 } } .
$$

Our bias-corrected variance estimator is thus $\hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 } : = \hat { A } _ { n } - \hat { B } _ { n } + \hat { C } _ { n }$ . As a follow-up, we do not directly target $g _ { s } ( \theta )$ for correction. Although $g _ { s } ( \theta )$ appears in the analytical second-order curvature expression, estimating it would require an additional plug-in approximation for $\theta ,$ which risks introducing a second layer of finite-sample bias. We therefore use $\hat { C } _ { n }$ as a feasible positive surrogate: it targets the directly estimable component of the curvature correction and preserves consistency (see below).

## A.3.3 Preserving consistency and asymptotic normality

To prove that ${ \hat { \sigma } } _ { n , \mathrm { c o r } } ^ { 2 } \overset { p } { \to } \sigma ^ { 2 }$ , where $\sigma ^ { 2 }$ is the limiting average increment variance that governs the martingale CLT in Theorem 1, by Slutsky’s it suffices to prove that $\hat { C } _ { n } \overset { p } {  } 0 ,$ as we already know that $\hat { A } _ { n } - \hat { B } _ { n } \overset { p } {  } \sigma ^ { 2 }$ from the corresponding proof in Appendix A.2. By Jensen’s Inequality and Assumption $6 ,$ we know that

$$
\mathbb { E } [ \hat { C } _ { n } ] = \frac { 1 } { n } \sum _ { s = 1 } ^ { n - 1 } \mathbb { E } \left[ \left( \frac { 1 } { N } \frac { y _ { I _ { s } } - \hat { f } ^ { ( s - 1 ) } ( x _ { I _ { s } } ) } { q _ { s } ( I _ { s } ) } \right) ^ { 2 } \right] w _ { s } \leq \frac { C ^ { 1 / 2 } } { n } \sum _ { s = 1 } ^ { n - 1 } w _ { s } = \frac { C ^ { 1 / 2 } } { n } \sum _ { s = 1 } ^ { n - 1 } \sum _ { k = s } ^ { n } \frac { 1 } { k ^ { 2 } } \to 0 .
$$

Because $\hat { C } _ { n } \geq 0 ,$ , it follows that $\mathbb { E } [ | \hat { C } _ { n } | ]  0 ,$ which implies that $\hat { C } _ { n } \overset { L _ { 1 } } {  } 0 ,$ which implies that $\hat { C } _ { n } = o _ { p } ( 1 )$ Immediately by Slutsky’s Theorem, under the assumptions of Theorem 1, we have

$$
\begin{array} { r } { \sqrt { n } (  { \hat { \theta } } _ { n } - \theta ) /  { \hat { \sigma } } _ { n , \mathrm { c o r } } \overset { d } { \to } \mathcal { N } ( 0 , 1 ) . } \end{array}
$$

## A.3.4 An MSE reduction argument for $\hat { \sigma } _ { n , { \tt c o r } } ^ { 2 }$

We proceed to characterize when $\hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 }$ will have a lower MSE than $\hat { \sigma } _ { n } ^ { 2 }$ . Expanding and substituting above, we have the following decomposition after adding and subtracting $A _ { n } - B _ { n } ;$

$$
\begin{array} { r l r } & { } & { \hat { \sigma } _ { n } ^ { 2 } - \sigma ^ { 2 } = \hat { A } _ { n } - \hat { B } _ { n } - \sigma ^ { 2 } = ( \hat { A } _ { n } - A _ { n } ) - ( \hat { B } _ { n } - B _ { n } ) + ( A _ { n } - B _ { n } - \sigma ^ { 2 } ) } \\ & { } & { = ( \hat { A } _ { n } - A _ { n } ) - \left( \cfrac { 1 } { n } \displaystyle \sum _ { t = 2 } ^ { n } ( \hat { \theta } _ { t - 1 } - \theta ) ^ { 2 } + \cfrac { 1 } { n } \displaystyle \sum _ { t = 2 } ^ { n } 2 ( \theta - \psi _ { t } ) ( \hat { \theta } _ { t - 1 } - \theta ) - \frac { 1 } { n } g _ { 1 } ( \theta ) \right) + ( A _ { n } - B _ { n } - \sigma ^ { 2 } ) } \\ & { } & { \overset { } { = } \underbrace { ( \hat { A } _ { n } - A _ { n } ) + ( A _ { n } - B _ { n } - \sigma ^ { 2 } ) - \cfrac { 1 } { n } \displaystyle \sum _ { t = 2 } ^ { n } 2 ( \theta - \psi _ { t } ) ( \hat { \theta } _ { t - 1 } - \theta ) + \frac { 1 } { n } ( \theta - \psi _ { 1 } ) ^ { 2 } } _ { \mathrm { c o n m o n e r r o r } E _ { n } } - \underbrace { \frac { 1 } { n } \displaystyle \sum _ { t = 2 } ^ { n } ( \hat { \theta } _ { t - 1 } - \theta ) ^ { 2 } } _ { \mathrm { t a r g e t e d e r r o r } T _ { n } } . } \end{array}
$$

Reusing the same scaffolding, for $\hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 } = \hat { A } _ { n } - \hat { B } _ { n } + \hat { C } _ { n , }$ , we have

$$
\hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 } - \sigma ^ { 2 } = E _ { n } - T _ { n } + \hat { C } _ { n } .
$$

Squaring and simplifying, we have

$$
\begin{array} { r } { \mathrm { M S E } ( \hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 } ) - \mathrm { M S E } ( \hat { \sigma } _ { n } ^ { 2 } ) = \mathbb { E } \left[ \left( E _ { n } - T _ { n } + \hat { C } _ { n } \right) ^ { 2 } \right] - \mathbb { E } \left[ \left( E _ { n } - T _ { n } \right) ^ { 2 } \right] } \\ { = \mathbb { E } \left[ \left( \hat { C } _ { n } - T _ { n } \right) ^ { 2 } \right] + 2 \mathbb { E } \left[ \hat { C } _ { n } E _ { n } \right] - \mathbb { E } \left[ T _ { n } ^ { 2 } \right] . } \end{array}
$$

Finally, we have the following necessary and sufficient criteria for when $\mathrm { M S E } ( \hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 } ) \leq \mathrm { M S E } ( \hat { \sigma } _ { n } ^ { 2 } )$

$$
\mathbf { M S E } ( \hat { \sigma } _ { n , \mathrm { c o r } } ^ { 2 } ) \leq \mathbf { M S E } ( \hat { \sigma } _ { n } ^ { 2 } ) \iff \underbrace { \frac { \mathbb { E } \left[ \left( \hat { C } _ { n } - T _ { n } \right) ^ { 2 } \right] } { \mathbb { E } \left[ T _ { n } ^ { 2 } \right] } } _ { \alpha _ { n } } + \underbrace { \frac { 2 \mathbb { E } \left[ \hat { C } _ { n } E _ { n } \right] } { \mathbb { E } \left[ T _ { n } ^ { 2 } \right] } } _ { \beta _ { n } } \leq 1 .
$$

In context, $\hat { C } _ { n }$ is our data-driven plug-in estimate for $T _ { n }$ . In words, the result above tells us that so long as (a) $\hat { C } _ { n }$ predicts $T _ { n }$ better than had we used $\mathrm { ~ 0 ~ } ( \mathrm { i . e . , ~ } \alpha _ { n }$ small); and (b) our correction term $\hat { C } _ { n }$ is not excessively<sup>7</sup> adversarially positively correlated with the common remaining error term $E _ { n } \ ( \mathrm { i . e . , } \beta _ { n }$ small), then we should expect $\hat { \sigma } _ { n , \mathrm { c o r } } ^ { \mathrm { ~ \tiny ~ { ~ \cdot ~ } ~ } }$ to have a lower MSE for $\sigma ^ { 2 }$ than $\hat { \sigma } _ { n } ^ { 2 }$

## A.4 Proof of Proposition 2

Let $\hat { \theta } _ { n }$ be our PAI estimator for $\theta _ { N }$ from Equation 2. We first observe that

$$
\sqrt { n } \left( \hat { \theta } _ { n } - \theta \right) = \sqrt { n } ( \hat { \theta } _ { n } - \theta _ { N } ) + \frac { \sqrt { n } } { \sqrt { N } } \sqrt { N } ( \theta _ { N } - \theta ) .
$$

For any $s , t ,$ we have the following, noting that $\theta _ { N }$ is a deterministic function of $y _ { 1 } , \ldots , y _ { N }$ . Formally, the limits here are taken with respect to $( n , N ) \uparrow \infty$ in Theorem ${ 1 \prime } \boldsymbol { \mathrm s }$ triangular array setup, letting $\mathcal { F } _ { N } =$

σ $\cdot \left( \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } \right)$

$$
\begin{array} { r l r } & { } & { \operatorname* { l i m } P \left( \sqrt { n } ( \hat { \theta } _ { n } - \theta _ { N } ) \leq s , \sqrt { N } ( \theta _ { N } - \theta ) \leq t \right) = \operatorname* { l i m } \mathbb { E } \left[ { \bf 1 } \left( \sqrt { n } ( \hat { \theta } _ { n } - \theta _ { N } ) \leq s , \sqrt { N } ( \theta _ { N } - \theta ) \leq t \right) \right] } \\ & { } & { = \operatorname* { l i m } \mathbb { E } \left[ { \bf 1 } \left( \sqrt { n } ( \hat { \theta } _ { n } - \theta _ { N } ) \leq s \right) { \bf 1 } \left( \sqrt { N } ( \theta _ { N } - \theta ) \leq t \right) \right] } \\ & { } & { = \operatorname* { l i m } \mathbb { E } \left[ { \bf E } \left[ { \bf 1 } \left( \sqrt { n } ( \hat { \theta } _ { n } - \theta _ { N } ) \leq s \right) { \bf 1 } \left( \sqrt { N } ( \theta _ { N } - \theta ) \leq t \right) | { \bf \Phi } _ { N } \right] \right] } \\ & { } & { = \operatorname* { l i m } \mathbb { E } \left[ { \bf 1 } \left( \sqrt { N } ( \theta _ { N } - \theta ) \leq t \right) \mathbb { E } \left[ { \bf 1 } \left( \sqrt { n } ( \hat { \theta } _ { n } - \theta _ { N } ) \leq s \right) | { \bf \Phi } _ { \cal F } \right] \right] } \\ & { } & { = \Phi \left( \frac { s } { \sigma } \right) \operatorname* { l i m } \mathbb { E } \left[ { \bf 1 } \left( \sqrt { N } ( \theta _ { N } - \theta ) \leq t \right) \right] = \Phi \left( \frac { s } { \sigma } \right) \Phi \left( \frac { t } { \tau } \right) , } \end{array}
$$

where $\sigma ^ { 2 }$ is the limit of the average conditional variances of the $\Delta _ { t }$ (c.f. Theorem 1 and Assumption 4). The last two equalities come from the asymptotic normality of $\hat { \theta } _ { n }$ proven in Theorem 1, the Dominated Convergence Theorem, and the standard Central Limit Theorem for i.i.d. random variables. Because the joint CDF factors into two independent terms, we deduce that

$$
( \sqrt { n } ( \widehat { \theta } _ { n } - \theta _ { N } ) , \sqrt { N } ( \theta _ { N } - \theta ) ) \stackrel { d } {  } \mathcal { N } ( \mathbf { 0 } , \mathrm { d i a g } ( \sigma ^ { 2 } , \tau ^ { 2 } ) ) .
$$

By the Continuous Mapping Theorem, it follows that

$$
{ \sqrt { n } } \left( { \hat { \theta } } _ { n } - \theta \right) { \xrightarrow { d } } { \mathcal { N } } \left( 0 , \sigma ^ { 2 } + \rho \tau ^ { 2 } \right) ,\tag{25}
$$

where $\rho : = \operatorname* { l i m } { n } / N$ from Theorem 1.

## A.5 Proof of Theorem 3

From the proof of Theorem 1 in Appendix A.2, we have the following after substituting in PAI’s oracle sampling rule, with the expectation necessary because we have to average over all random sampling paths, and permissible because of orthogonality of martingale increments:

$$
\begin{array} { r } { \mathrm { V a r } ( \hat { \theta } _ { \mathrm { P a A } } ) = \displaystyle \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ \left( \displaystyle \sum _ { i \neq \theta _ { i - 1 } } \frac { \left( y _ { i } - f ( x _ { i } ) \right) ^ { 2 } } { q _ { t } ( i ) } \right) - \left( \displaystyle \sum _ { i \neq \theta _ { i - 1 } } \left( y _ { i } - f ( x _ { i } ) \right) \right) ^ { 2 } \right] } \\ { = \displaystyle \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ \left( \displaystyle \sum _ { i \in R _ { i - 1 } } \frac { \tau _ { i } ^ { 2 } } { q _ { t } ( i ) } \right) - \left( \displaystyle \sum _ { i \in R _ { t - 1 } } e _ { i } \right) ^ { 2 } \right] } \\ { = \displaystyle \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } - S ( R _ { t - 1 } ) ^ { 2 } \right] : = \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } \left( 1 - b _ { t } ^ { 2 } \right) \right] } \\ { = \displaystyle \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } \right] , \displaystyle \sum _ { i = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } \right] - \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } b _ { t } ^ { 2 } \right] , } \end{array}
$$

if we define $b _ { t }$ as capturing the net signed direction of the residual mass:

$$
b _ { t } = \frac { S ( R _ { t - 1 } ) } { U ( R _ { t - 1 } ) } = \frac { \sum _ { i \in R _ { t - 1 } } y _ { i } - f ( x _ { i } ) } { \sum _ { i \in R _ { t - 1 } } | y _ { i } - f ( x _ { i } ) | } .
$$

To formalize, define ${ \bar { b } } ^ { 2 }$ to be the weighted-average (squared) net signed direction of the residual mass over the n sampling rounds (weighted by $U ( { \check { R } } _ { t - 1 } ) ^ { 2 } )$ :

$$
\bar { b } ^ { 2 } : = \frac { \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } b _ { t } ^ { 2 } \right] } { \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } \right] } .\tag{26}
$$

Then, we have

$$
\mathrm { V a r } ( \hat { \theta } _ { \mathrm { P A I } } ) = \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } \right] - \bar { b } ^ { 2 } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } \right] = \frac { 1 - \bar { b } ^ { 2 } } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 1 } ^ { n } \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } \right] : = ( 1 - \bar { b } ^ { 2 } ) A _ { \mathrm { P A I } } .
$$

Working on the expectation term, letting $I _ { t }$ represent the index sampled at time $t ,$ letting $\begin{array} { r } { Q ( R ) = \sum _ { i \in R } r _ { i } ^ { 2 } } \end{array}$ (“quadratic"), and letting $\begin{array} { r } { C ( R ) = \sum _ { i \in R } r _ { i } ^ { 3 } } \end{array}$ (“cubic"), we have by LOTE the following:

$$
\begin{array} { r } { \mathbb { E } \left[ U ( R _ { t } ) ^ { 2 } \ | \ R _ { t - 1 } \right] = \mathbb { E } [ ( U ( R _ { t - 1 } ) - r _ { I _ { t } } ) ^ { 2 } \ | \ R _ { t - 1 } ] = \mathbb { E } \left[ U ( R _ { t - 1 } ) ^ { 2 } - 2 r _ { I _ { t } } U ( R _ { t - 1 } ) + r _ { I _ { t } } ^ { 2 } \ | \ R _ { t - 1 } \right] } \\ { = U ( R _ { t - 1 } ) ^ { 2 } - 2 U ( R _ { t - 1 } ) \underset { i \in R _ { t - 1 } } { \sum } \ \underset { U ( R _ { t - 1 } ) } { \sum } + \underset { i \in R _ { t - 1 } } { \sum } \ \frac { r _ { i } ^ { 3 } } { U ( R _ { t - 1 } ) } = U ( R _ { t - 1 } ) ^ { 2 } - 2 Q ( R _ { t - 1 } ) + \frac { C ( R _ { t - 1 } ) } { U ( R _ { t - 1 } ) } . } \end{array}
$$

By Tower, we have the following recursion:

$$
\mathbb { E } \left[ U ( R _ { t } ) ^ { 2 } \right] = \mathbb { E } [ U ( R _ { t - 1 } ) ^ { 2 } ] - \mathbb { E } \left[ 2 Q ( R _ { t - 1 } ) - \frac { C ( R _ { t - 1 } ) } { U ( R _ { t - 1 } ) } \right]
$$

Applying the recursion t times, and letting $R _ { 0 } = \tau ,$ we have

$$
\mathbb { E } \left[ U ( R _ { t } ) ^ { 2 } \right] = U ( R _ { 0 } ) ^ { 2 } - \sum _ { s = 0 } ^ { t - 1 } \mathbb { E } \left[ 2 Q ( R _ { s } ) - \frac { C ( R _ { s } ) } { U ( R _ { s } ) } \right] ,
$$

which tells us the following, after switching the second order of summation and reindexing:

$$
\begin{array} { r l r } & { } & { A _ { \mathrm { P A I } } = \displaystyle \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 1 } ^ { n } \mathbb { E } [ U ( R _ { t - 1 } ) ^ { 2 } ] = \displaystyle \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 0 } ^ { n - 1 } \mathbb { E } [ U ( R _ { t } ) ^ { 2 } ] } \\ & { } & { \quad = \displaystyle \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { t = 0 } ^ { n - 1 } \left( U ( R _ { 0 } ) ^ { 2 } - \sum _ { s = 0 } ^ { t - 1 } \mathbb { E } \left[ 2 Q ( R _ { s } ) - \frac { C ( R _ { s } ) } { U ( R _ { s } ) } \right] \right) } \\ & { } & { \quad = \displaystyle \frac { 1 } { n N ^ { 2 } } U ( R _ { 0 } ) ^ { 2 } - \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { s = 0 } ^ { n - 2 } ( n - s - 1 ) \mathbb { E } \left[ 2 Q ( R _ { s } ) - \frac { C ( R _ { s } ) } { U ( R _ { s } ) } \right] . } \end{array}
$$

Now, because $Q ( R _ { s } ) \leq Q ( R _ { 0 } )$ and because $C ( R _ { s } ) / U ( R _ { s } ) \geq 0$ for all $s ,$ we know by property of arithmetic sums that

$$
\begin{array} { r l r } & { } & { A _ { \mathrm { P A I } } \geq \displaystyle \frac { 1 } { n N ^ { 2 } } U ( R _ { 0 } ) ^ { 2 } - \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { s = 0 } ^ { n - 2 } ( n - s - 1 ) 2 Q ( R _ { 0 } ) } \\ & { } & { \quad \quad \quad = \displaystyle \frac { 1 } { n N ^ { 2 } } U ( R _ { 0 } ) ^ { 2 } - \frac { Q ( R _ { 0 } ) ( n - 1 ) } { n N ^ { 2 } } = \frac { 1 } { N ^ { 2 } } \left( \frac { U ( R _ { 0 } ) ^ { 2 } } { n } - \frac { n - 1 } { n } Q ( R _ { 0 } ) \right) . } \end{array}
$$

Computing BAI/SAI’s variance (c.f. Equation 11), we have

$$
\operatorname { V a r } ( { \hat { \theta } } _ { \mathrm { a c t i v e } } ) = { \frac { 1 } { N ^ { 2 } } } \sum _ { t = 1 } ^ { N } { \frac { ( y _ { t } - f ( x _ { t } ) ) ^ { 2 } ( 1 - \pi _ { t } ( x _ { t } ) ) } { \pi _ { t } ( x _ { t } ) } } = { \frac { 1 } { N ^ { 2 } } } \sum _ { t = 1 } ^ { N } r _ { t } ^ { 2 } \left( { \frac { 1 } { \pi _ { t } ( x _ { t } ) } } - 1 \right) .
$$

After substituting in SAI/BAI’s oracle policy, we have:

$$
\mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) = \frac { 1 } { N ^ { 2 } } \sum _ { t = 1 } ^ { N } r _ { t } ^ { 2 } \left( \frac { 1 } { \pi _ { t } ( x _ { t } ) } - 1 \right) = \frac { 1 } { N ^ { 2 } } \left( \frac { U ( R _ { 0 } ) ^ { 2 } } { n } - Q ( R _ { 0 } ) \right) .
$$

This is very convenient, because it tells us that

$$
A _ { \mathrm { { P A I } } } \geq \mathrm { { V a r } } ( \hat { \theta } _ { \mathrm { { a c t i v e } } } ) + \frac { Q ( R _ { 0 } ) } { n N ^ { 2 } } .
$$

Working on an upper bound, recall that

$$
A _ { \mathrm { P A I } } = \frac { 1 } { n N ^ { 2 } } U ( R _ { 0 } ) ^ { 2 } - \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { s = 0 } ^ { n - 2 } ( n - s - 1 ) \mathbb { E } \left[ 2 Q ( R _ { s } ) - \frac { C ( R _ { s } ) } { U ( R _ { s } ) } \right] .
$$

Because only one unit is removed at each timestep, letting $r _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { i \in R _ { 0 } } r _ { i } ,$ we have

$$
Q ( R _ { s } ) \geq Q ( R _ { 0 } ) - s r _ { \operatorname* { m a x } } ^ { 2 } .
$$

Moreover,

$$
\frac { C ( R _ { s } ) } { U ( R _ { s } ) } = \frac { \sum _ { i \in R _ { s } } r _ { i } ^ { 3 } } { \sum _ { i \in R _ { s } } r _ { i } } \leq \frac { r _ { \operatorname* { m a x } } ^ { 2 } \sum _ { i \in R _ { s } } r _ { i } } { \sum _ { i \in R _ { s } } r _ { i } } = r _ { \operatorname* { m a x } } ^ { 2 } .
$$

Thus, we have

$$
\begin{array} { r l r } & { } & { A _ { \mathrm { P A I } } \leq \displaystyle \frac { 1 } { n N ^ { 2 } } U ( R _ { 0 } ) ^ { 2 } - \frac { 1 } { n ^ { 2 } N ^ { 2 } } \sum _ { s = 0 } ^ { n - 2 } ( n - s - 1 ) \left( 2 Q ( R _ { 0 } ) - ( 2 s + 1 ) r _ { \mathrm { m a x } } ^ { 2 } \right) } \\ & { } & { = \displaystyle \frac { 1 } { N ^ { 2 } } \left( \frac { U ( R _ { 0 } ) ^ { 2 } } { n } - \frac { n - 1 } { n } Q ( R _ { 0 } ) \right) + \frac { r _ { \mathrm { m a x } } ^ { 2 } } { n ^ { 2 } N ^ { 2 } } \sum _ { s = 0 } ^ { n - 2 } ( n - s - 1 ) ( 2 s + 1 ) . } \end{array}
$$

Working on the last summation, we have

$$
\begin{array} { r l r } & { } & { \displaystyle \sum _ { s = 0 } ^ { n - 2 } ( n - s - 1 ) ( 2 s + 1 ) = \sum _ { s = 0 } ^ { n - 2 } ( n - 1 ) ( 2 s + 1 ) - s ( 2 s + 1 ) } \\ & { } & { = ( n - 1 ) \sum _ { s = 0 } ^ { n - 2 } ( 2 s + 1 ) - 2 \sum _ { s = 0 } ^ { n - 2 } s ^ { 2 } - \sum _ { s = 0 } ^ { n - 2 } s = ( n - 1 ) ^ { 3 } - \left( 2 \sum _ { s = 0 } ^ { n - 2 } s ^ { 2 } + \sum _ { s = 0 } ^ { n - 2 } s \right) = \frac { n ( n - 1 ) ( 2 n - 1 ) } { 6 } . } \end{array}
$$

Thus, we conclude that

$$
\begin{array} { r } { A _ { \mathrm { P A I } } \leq \displaystyle \frac { 1 } { N ^ { 2 } } \left( \frac { U ( R _ { 0 } ) ^ { 2 } } { n } - \frac { n - 1 } { n } Q ( R _ { 0 } ) \right) + \frac { ( n - 1 ) ( 2 n - 1 ) r _ { \mathrm { m a x } } ^ { 2 } } { 6 n N ^ { 2 } } } \\ { = \mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) + \frac { Q ( R _ { 0 } ) } { n N ^ { 2 } } + \frac { ( n - 1 ) ( 2 n - 1 ) r _ { \mathrm { m a x } } ^ { 2 } } { 6 n N ^ { 2 } } . } \end{array}
$$

Putting everything together, we deduce that

$$
\mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) + \frac { Q ( R _ { 0 } ) } { n N ^ { 2 } } \le A _ { \mathrm { P A I } } \le \mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) + \frac { Q ( R _ { 0 } ) } { n N ^ { 2 } } + \frac { ( n - 1 ) ( 2 n - 1 ) r _ { \mathrm { m a x } } ^ { 2 } } { 6 n N ^ { 2 } } ,
$$

which implies that

$$
\begin{array} { r } { ( 1 - \bar { b } ^ { 2 } ) \left( \mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) + \frac { Q ( R _ { 0 } ) } { n N ^ { 2 } } \right) \leq \mathrm { V a r } ( \hat { \theta } _ { \mathrm { P A I } } ) } \\ { \leq ( 1 - \bar { b } ^ { 2 } ) \left( \mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) + \frac { Q ( R _ { 0 } ) } { n N ^ { 2 } } + \frac { ( n - 1 ) ( 2 n - 1 ) r _ { \mathrm { m a x } } ^ { 2 } } { 6 n N ^ { 2 } } \right) , } \end{array}
$$

or, more suggestively, recalling that $\begin{array} { r } { \mathrm { ( i ) } Q ( R _ { 0 } ) = \sum _ { i = 1 } ^ { N } ( y _ { i } - f ( x _ { i } ) ) ^ { 2 } ; \mathrm { ( i i ) } r _ { \mathrm { m a x } } = \operatorname* { m a x } _ { i \in [ 1 : N ] } | y _ { i } - f ( x _ { i } ) | ; } \end{array}$ ; and (iii) ${ \bar { b } } ^ { 2 }$ is the weighted-average (squared) net signed direction of the residual mass over the n sampling rounds (c.f. Equation 26):

$$
\frac { 1 } { 1 - \bar { b } ^ { 2 } } \cdot \frac { 1 } { 1 + \frac { Q ( R _ { 0 } ) } { n N ^ { 2 } \bar { \mathrm { V a r } } ( \bar { \theta } _ { \mathrm { a c t i v e } } ) } + \frac { ( n - 1 ) ( 2 n - 1 ) r _ { \mathrm { m a x } } ^ { 2 } } { 6 n N ^ { 2 } \bar { \mathrm { V a r } } ( \bar { \theta } _ { \mathrm { a c t i v e } } ) } } \leq \frac { \mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) } { \mathrm { V a r } ( \hat { \theta } _ { \mathrm { P A I } } ) } \leq \frac { 1 } { 1 - \bar { b } ^ { 2 } } \frac { 1 } { 1 + \frac { Q ( R _ { 0 } ) } { n N ^ { 2 } \bar { \mathrm { V a r } } ( \bar { \theta } _ { \mathrm { a c t i v e } } ) } } ,
$$

or, with one final substitution of $\begin{array} { r } { \mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) = \frac { 1 } { N ^ { 2 } } \left( \frac { U ( R _ { 0 } ) ^ { 2 } } { n } - Q ( R _ { 0 } ) \right) } \end{array}$

$$
\frac { 1 } { 1 -  { { \overline { { b } } } } ^ { 2 } } \cdot \frac { 1 } { 1 + \frac { Q ( R _ { 0 } ) } { U ( R _ { 0 } ) ^ { 2 } - n Q ( R _ { 0 } ) } + \frac { ( n - 1 ) ( 2 n - 1 ) r _ { \mathrm { m a x } } ^ { 2 } } { 6 ( U ( R _ { 0 } ) ^ { 2 } - n Q ( R _ { 0 } ) ) } } \le \frac { \mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) } { \mathrm { V a r } ( \hat { \theta } _ { \mathrm { P A I } } ) } \le \frac { 1 } { 1 -  { { \overline { { b } } } } ^ { 2 } } \frac { 1 } { 1 + \frac { Q ( R _ { 0 } ) } { U ( R _ { 0 } ) ^ { 2 } - n Q ( R _ { 0 } ) } } .
$$

Formally, when $n r _ { \mathrm { m a x } } / U ( R _ { 0 } ) ~ \to ~ 0 , ^ { 8 }$ both remainder terms $\frac { Q ( R _ { 0 } ) } { U ( R _ { 0 } ) ^ { 2 } - n Q ( R _ { 0 } ) }$ and $\frac { ( n - 1 ) ( 2 n - 1 ) r _ { \operatorname* { m a x } } ^ { 2 } } { 6 ( U ( R _ { 0 } ) ^ { 2 } - n Q ( R _ { 0 } ) ) }$ vanish asymptotically<sup>9</sup>, yielding the following asymptotic relative-efficiency relationship:

$$
\frac { \mathrm { V a r } ( \hat { \theta } _ { \mathrm { a c t i v e } } ) } { \mathrm { V a r } ( \hat { \theta } _ { \mathrm { P A I } } ) } = \frac { 1 + o ( 1 ) } { 1 - \bar { b } ^ { 2 } } .
$$

The sufficient condition above admits an intuitive decomposition:

$$
\frac { n r _ { \mathrm { m a x } } } { U ( R _ { 0 } ) } = \frac { n } { N } \cdot \frac { r _ { \mathrm { m a x } } } { \bar { r } } , \quad \mathrm { w h e r e } \quad \bar { r } : = \frac { 1 } { N } U ( R _ { 0 } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | y _ { i } - f ( x _ { i } ) | .
$$

Since $r _ { \operatorname* { m a x } { } } \geq { \bar { r } } ,$ , this condition necessarily requires $n / N \to 0 .$ , which is permitted by Theorem 1. More precisely, the labeling fraction must vanish sufficiently quickly relative to $r _ { \operatorname* { m a x } } / \bar { r }$ . In particular, $n / N \to 0$ suffices if $r _ { \operatorname* { m a x } } / \bar { r } = O ( 1 )$ .

## A.6 Diagonal stochastic Newton updates for online logistic regression

The following is adapted from Equations 3.1-3.3 of Bercu et al. (2020), who maintain a dense $D \times D$ curvature matrix, whereas we use a diagonal approximation.

Let $\beta _ { t } ~ \in ~ \mathbb { R } ^ { D }$ denote our logistic regression coefficients at round $t ,$ and let $h _ { t } \ \in \ \mathbb { R } ^ { D }$ denote our accumulated diagonal curvature vector. For all datasets except for AlphaFold (which provides fixed probabilistic ML predictions without model access) and the two LLM evaluation suites MMLU-Pro and BBH+GPQA+IFEval+MATH+MusR, we initialize $\beta _ { 0 } ~ { = } ~ \beta _ { \mathrm { i n i t } }$ by fitting on a training split of the data and initialize $h _ { 0 } = \mathbf { 1 } _ { D }$ . For the two LLM evaluation suites, we use the historical-data regimen described in Wu et al. (2026): fitted question factors are held fixed, while $\beta _ { 0 }$ and $h _ { 0 }$ are initialized to the empirical mean and inverse marginal variances, respectively, of the historical LLMs’ fitted model factors. These initializations are identical across all new LLMs.

After observing $( x _ { I _ { t } } , y _ { I _ { t } } )$ , we compute

$$
h _ { t } = h _ { t - 1 } + \sigma \left( \beta _ { t - 1 } ^ { \top } x _ { I _ { t } } \right) \left( 1 - \sigma \left( \beta _ { t - 1 } ^ { \top } x _ { I _ { t } } \right) \right) x _ { I _ { t } } ^ { \odot 2 } ,\tag{27}
$$

$$
\beta _ { t } = \beta _ { t - 1 } + ( y _ { I _ { t } } - \sigma \left( \beta _ { t - 1 } ^ { \top } x _ { I _ { t } } \right) ) x _ { I _ { t } } \oslash h _ { t } ,\tag{28}
$$

where $\odot 2$ represents elementwise squaring, and $\oslash$ represents elementwise division.

## B Additional results

![](images/8339bde4eba1b0cbcd2db48701cd07e0e424fbb8676ede5331a2f9037df23576.jpg)  
Figure 6: Comparing empirical coverages of Pro-Active Inference (PAI), Sequential Active Inference (SAI), Prediction-Powered Inference (PPI), Uniform Sampling (Uniform), and Prediction-Powered Active Testing (PPAT) across seven datasets. Each subplot shows the empirical coverage of nominal 95% normal-approximation confidence intervals as a function of labeling budget. For all datasets except the two LLM evaluation suites (MMLU-Pro and BBH+GPQA+IFEval+MATH+MusR), empirical coverages are computed over 1000 random seeds. For the two LLM evaluation suites, empirical coverages are averaged over 2.2K test models per evaluation suite and 100 seeds for computational efficiency. Standard errors are shown, but are negligible at the plotted scale.

![](images/13cac78cac22e5c98c92b41ea744ecfe62d6098904a33ff6019c55ace6198534.jpg)  
Figure 7: ESS multipliers of without-replacement PAI over with-replacement PAI across the two LLM evaluation suites. Each subplot shows the ESS multiplier of without-replacement PAI with respect to withreplacement PAI as a function of labeling budget. For example, on MMLU-Pro at a 22.5% budget, an ESS multiplier of 1.3 means that without-replacement PAI matches the confidence interval width of withreplacement PAI at a 29.25% budget. Standard errors are shown, but are negligible at the plotted scale. Higher ESS multipliers indicate better sampling efficiency, with values above 1 indicating that without replacement PAI outperforms with-replacement PAI.

<sup>ponding</sup> <sup>PAI</sup> <sup>variant</sup> <sup>outperfor</sup>m<sub>s</sub> <sub>the</sub> <sub>c</sub> <sup>ecifications</sup> <sup>in</sup> <sup>Section</sup> <sup>5.Higher</sup> <sup>ESS</sup> <sup>multipliers</sup> <sup>ind</sup>i<sub>cate</sub> <sub>better</sub> <sub>sampling</sub> <sub>efficiency,</sub> <sub>with</sub> <sub>value</sub> <sup>ML</sup> <sup>model</sup> <sup>is</sup> <sup>held</sup> <sup>fixe</sup><sub>d</sub> <sub>throughout</sub> <sub>sampling,</sub> <sub>affecting</sub> <sub>both</sub> <sub>imputation</sub> <sub>and</sub> <sub>acquisition.</sub> <sub>The</sub> <sub>co</sub> ) <sup>en-ML</sup> <sup>PAI</sup> <sup>(red</sup> <sup>circles</sup> <sup>with</sup> <sup>respect</sup> <sup>to</sup> <sup>the</sup> <sup>corresponding</sup> <sup>co</sup>m<sub>petitor</sub> <sub>method</sub> <sub>as</sub> <sub>a</sub> <sub>function</sub> <sub>of</sub> <sub>labeli</sub> <sup>PPAT,</sup> <sup>or</sup> <sup>Uniform)</sup> <sup>and</sup> <sup>each</sup> <sup>column</sup> <sup>corresponds</sup> <sup>to</sup> <sup>one</sup> <sup>d</sup>a<sub>taset.</sub> <sub>Each</sub> <sub>subplot</sub> <sub>shows</sub> <sub>the</sub> <sub>ESS</sub> <sub>mu</sub> <sup>rs</sup> <sup>of</sup> <sup>PAI</sup> <sup>and</sup> <sup>its</sup> <sup>frozen-</sup>M<sub>L</sub> <sub>ablation</sub> <sub>over</sub> <sub>PPI,</sub> <sub>PPAT,</sub> <sub>and</sub> <sub>uniform</sub> <sub>sampling</sub> <sub>across</sub> <sub>six</sub> <sub>datasets</sub>  
![](images/1938cae3e961fc758e8fffcf9748b240b2d4f4abc397baf1285220ff9f3e9b2f.jpg)