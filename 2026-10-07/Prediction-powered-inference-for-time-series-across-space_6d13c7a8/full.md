# Prediction-powered inference for time series across space

Shahzar Rizvi<sup>1</sup> David Burt<sup>2</sup> Vishwak Srinivasan<sup>1</sup>

Renato Berlinghieri<sup>3</sup> Stefano Del Col<sup>4</sup> Tamara Broderick<sup>1</sup>

<sup>1</sup>MIT LIDS <sup>2</sup>UMass Amherst <sup>3</sup>Stanford University <sup>4</sup>Assicurazioni Generali

shahzar@mit.edu, dburt@umass.edu, vishwaks@mit.edu,

renb@stanford.edu, stefano.delcol@generali.com, tbroderick@mit.edu

## Abstract

The following motif is common in spatiotemporal settings: we have a sequence of covariate and label pairs observed for a relatively short, recent time period. We have access to unlabeled covariates over a longer time period. Data is observed over many spatial locations. For instance, crop yield might be observed over a large geographical area for recent years, but weather data (which is informative about crop yield) is available for a much longer period. The goal is to estimate, at each spatial location, the expected label (e.g., crop yield) in the future and provide a valid confidence interval for this value. The observed time period alone is too short for reliable estimates. Imputing missing labels with machine learning can cause substantial bias. Prediction-powered inference (PPI) can correct for this bias, but it relies on an i.i.d. assumption that breaks under our expected temporal dependencies. Heteroskedasticity and autocorrelation consistent (HAC) procedures account for temporal correlation, but have not been adapted to cases where some labels are imputed. We provide reliable point estimates and confidence intervals given: short labeled time series (across spatial locations), a longer unlabeled time series, and an imperfect predictor of labels given covariates. We show our method outperforms natural alternatives.

## 1 Introduction

We start with an illustrative motivating example and then describe the general case in more detail. We have been collaborating with Generali Group, one of the world’s largest insurance and asset management groups. Generali faces the following practical risk-modeling setting related to insuring assets. In a range of locations across a wide geographic span, Generali has a historical record of insurance claims that are largely caused by adverse weather. Generali also has access to a longer record of daily weather data, but without corresponding claims information. Generali would like to use this additional information to support risk estimation in locations exposed to weather-related hazards. Moreover, confidence intervals with reliable coverage for these risk estimates are important to support risk assessment and decision-making.

Generali’s problem is an example of a more general motif. In the motif, we have access to data at a set of locations spanning a large geographic area. At each location, we have data across time steps in the recent past consisting of both responses and a collection of covariates that we expect to be informative about those responses. Prior to the labeled data, we have access to a much longer historical sequence of just the covariates (i.e., no responses). We would like to use all the available data to estimate the expected response value in future time periods, and we would like to provide reliable confidence intervals for this estimate.

We expect this motif to arise commonly in practice. For instance, Lupone et al. [1] have access to limited recent data labeled with mammal population sizes and longer-term historical data on environmental variables; they build predictors from the shorter-term labeled data and “hindcast” mammal populations across the past, unlabeled data. Similarly, practitioners are interested in (1) estimating future crop yield using meteorological covariates [2], and researchers might have started collecting crop yield more recently than the extent of the meteorological record; and (2) estimating future $\mathrm { P M } _ { 2 . 5 }$ levels using satellite, meteorological, and land-use covariates [3, 4].

This motif presents challenges to standard inferential frameworks due to the limited labeled data and known temporal dependencies. For instance, a classical and widely used approach to similar problems (without an unlabeled dataset) is to estimate the future response with the empirical average of the observed responses and report a standard confidence interval. We could just ignore the unlabeled data and apply this approach, but we expect poor point estimations due to the limited availability of labeled data. For instance, consider our running insurance example, and suppose a particular location actually experiences a storm every 2 years in the long run. Suppose we have 5 years of labeled data; that, by chance, we only observe one storm in those 5 years; and that an insurance company pays out 10% of insured asset value (payout = 0.1) when a storm hits. The true mean annual payout then would be $0 . 1 \times 1 / 2 = 0 . 0 5$ at this location, but the classical estimate would yield $0 . 1 \times 1 / 5 = 0 . 0 2$

A natural and widespread correction [e.g. 5, 6, 7, 8], which we refer to as “inference with imputation” (IWI),<sup>1</sup> is to fit a machine learning predictor using some of the available labeled data. IWI uses this predictor to predict the responses from the observed covariates, so all unlabeled points are now assigned (predicted) labels. IWI treats the predicted labels as though they were true labels and applies the classical approach. In the example above, we might hope that incorporating the historical record of weather data would ensure we have a better estimate of storm frequency relative to the classical approach. However, IWI is known to introduce its own bias [10]. For instance, we might expect bias in the machine learning predictions due to unobserved covariates relevant to the problem.

This bias is essentially the problem that prediction-powered inference (PPI) [11, 12] is designed to solve. PPI uses some of the labeled data to estimate the bias between the predicted and true responses and then uses this estimate to debias the final estimated mean response. While PPI should substantially improve point estimation, we expect its confidence interval to be miscalibrated in our setting since it relies on an assumption that data is i.i.d. In particular, a straightforward way to apply PPI would be to treat the data across time, observed at any single spatial point, as i.i.d. But we know that meaningful dependencies exist across time in our setting.

Another way to frame our problem is the following: PPI relies on an i.i.d. assumption that is meaningfully wrong in the spatiotemporal setting of interest to us. In the present work, we propose to work under a more spatiotemporally realistic (and strictly weaker) assumption than i.i.d. (namely, time stationarity), and to develop an analogous framework for valid confidence intervals for the long-run mean response of interest. Concretely, our contributions form two main directions. First, we decide how to efficiently use limited labeled data in the face of temporal correlation. E.g., while the order of data does not matter for standard PPI, we decide between interleaving or chunking data reserved for prediction vs. for debiasing. Second, we adapt heteroskedasticity and autocorrelation consistent (HAC) procedures [13, 14, 15, 16] within a PPI framework. PPI confidence intervals rely on estimated uncertainties. HAC corrects uncertainties using estimates of the covariance between response data on nearby time steps. By combining the bias correction of PPI with the uncertainty adaptation of HAC, we aim to construct inferentially valid confidence intervals that leverage machine learning predictions while accounting for the temporal dependence of the observed data. Along the way, we sketch out an asymptotic justification of our approach, relying on our choice of the Newey–West [14, 17] asymptotic regime as opposed to that of Kiefer–Vogelsang [18]. Though we have not filled in a full proof, we describe what makes the problem non-trivial and hope to complete the theory in a future version of the manuscript.

In what follows, we start by establishing notation for our problem setup in Section 2. We briefly review past methods in Section 3 along the way to establishing our method, which we call PPI for time series across space (PPITSAS, pronounced “pizzas”), in Section 4. We describe our experimental setup in Section 5.1, where we address subtle challenges that arise in validating method performance.

Finally, in Section 5.2, we empirically demonstrate that our proposed method, PPITSAS, outperforms the evaluated alternative approaches in our setting.

## 1.1 Related work

Relation to missingness variations. To adapt PPI to handle temporal dependencies, we choose in the present work to build from the PPI and HAC literatures. Alternatively, we might have built from a literature aiming to extend the missingness conditions under which PPI (and similar adjustments) work. One can conceptualize standard PPI as working under an assumption that responses are “missing completely at random” (MCAR) within the full data set. Recent work has focused on extending to “missing at random” (MAR), where the missingness indicator can be predicted from the covariates, but not the missing responses themselves. If we include observation times among the covariates, then our setting can be considered a special case of MAR. See Song et al. [19, Section 2] for a discussion of connections between PPI and missingness.

For any method to work, we need a regularity assumption that relates the unlabeled responses to the labeled responses. In MCAR, this is the i.i.d. assumption. Kluger et al. [20] consider two distinct ways to address this need for regularity in MAR. (1) They make a standard assumption (familiar from the distribution shift literature) that covariates in the missing and not-missing data overlap substantially. If we consider time as a covariate (to fit their framework), our unlabeled data are, by design, strictly prior in time to our labeled data; thus, overlap fails. (2) They remark that more generally, one might extend their approach to cases where asymptotic normality and bootstrap consistency results hold. Even if a formal result to this effect were available, the work still needs to be done to choose an appropriate regularity assumption (analogous to the i.i.d. assumption, distribution shift with overlap, or time stationarity) and work out the asymptotic normality result. We see this work as equivalent in scope to what we accomplish in this paper.

Salerno et al. [21] likewise consider a case with observations missing at random and filled in via prediction, though they more explicitly assume spatial dependence. The distinction between time and space need not be meaningful; we could treat our time coordinate as a spatial coordinate in their framework. However, they again rely on an overlap assumption (to fit propensity models), which again fails when we treat time as a covariate (just as Salerno et al. [21] explicitly treat their spatial coordinates as covariates).

Mean estimation vs. anytime valid testing. Csillag et al. [22] demonstrate that any e-value procedure has a PPI counterpart, and give an application in change-point detection for time-series data. Csillag et al. [23] extend PPI and address a risk monitoring problem with temporal dependence. But both papers address anytime valid sequential testing rather than mean estimation.

## 2 Problem setup

We next describe the available data and our analysis goals in more detail.

Data. We observe data at a fixed set of locations $ { \boldsymbol { S } } : = \{ s _ { 1 } , \ldots , s _ { N _ { S } } \}$ . Across all locations, we observe data across the same set of times. For convenience, we assume all times are in the positive integers $( \mathbb { N } = \{ 1 , 2 , . . . \} )$ . E.g., in our running example, each integer corresponds to a day, and we have access to daily weather at many sites across a country for multiple decades. We expect it would be conceptually straightforward, but notationally cumbersome, to extend our development to non-integer times. We assume the times at which we observe unlabeled data, $\mathcal { T } _ { \mathrm { U } } = \{ 1 , . . . , T _ { \mathrm { U } } \}$ occur directly before those with labeled data, $\mathcal { T } _ { \mathrm { L } } = \{ T _ { \mathrm { U } } + 1 , \ldots , T _ { \mathrm { U } } + T _ { \mathrm { L } } \}$ . In particular, we have access to (1) covariates $X _ { t } ( s ) \in \mathbb { R } ^ { D }$ across $s \in { \mathcal { S } }$ and all times $t \in \mathcal { T } _ { \mathrm { U } } \cup \mathcal { T } _ { \mathrm { L } }$ , and (2) responses $Y _ { t } ( s ) \in \mathbb { R }$ across $s \in S$ and $t \in \mathcal { T } _ { \mathrm { L } }$

Estimand. Roughly, our goal is to estimate the long-run mean of the responses at each spatial location. To make this estimand precise, we need some assumptions; the assumptions leading to our estimand follow Section 7.2 of [24]. We start by assuming that, for each spatial location $s ,$ , there exist two stochastic processes indexed by $t \in \mathbb { N }$ : the covariate process $\{ X _ { t } ( s ) \} _ { t }$ and the response process $\{ Y _ { t } ( s ) \} _ { t }$ . We assume that, for all $\dot { t } \in \mathbb { N } , \mathbb { E } [ Y _ { t } ( s ) ]$ exists, and for all ${ \dot { t _ { 1 } } } , { \dot { t _ { 2 } } } \in \mathbb { N } , \mathbf { C o v } [ { \dot { Y } } _ { t _ { 1 } } ( s ) , \mathbf { \dot { Y } } _ { t _ { 2 } } ( s ) ]$ exists. Finally, we assume that $( 1 ) \{ Y _ { t } ( s ) \} _ { : }$ <sub>t</sub> is covariance-stationary $( \mathrm { i . e . }$ , for each location $s \in S .$ there exist real-valued $\mu ( s )$ and $\{ \gamma _ { j } ( s ) \} _ { j = 0 } ^ { \infty }$ such that, for all times $ t , t _ { 1 } , t _ { 2 } \in \mathbb { N } .$ the left and center of Eq. (1) hold) and (2) that the covariances are absolutely summable (right, Eq. (1)).

$$
\mathbb { E } [ Y _ { t } ( s ) ] = \mu ( s ) ; \quad \mathrm { C o v } [ Y _ { t _ { 1 } } ( s ) , Y _ { t _ { 2 } } ( s ) ] = \gamma _ { | t _ { 1 } - t _ { 2 } | } ( s ) ; \quad \sum _ { j = 0 } ^ { \infty } | \gamma _ { j } ( s ) | < \infty .\tag{1}
$$

With these assumptions in hand, Proposition 7.5 of [24] ensures that, at each spatial location $s ,$ the long-run mean response $( \theta ( s )$ in Eq. (2)) exists and is well defined:

$$
\theta ( s ) : = \operatorname* { l i m } _ { T  \infty } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } Y _ { t } ( s ) .\tag{2}
$$

Our goal is the following: for each spatial location $s \in S ,$ , estimate the long-run mean response and provide a valid confidence interval for it.

## 3 Background: Prediction-powered inference

Since PPITSAS builds on PPI, we first establish the standard PPI method and notation.

To apply PPI, one first partitions the indices $\mathcal { T } _ { \mathrm { L } }$ of labeled data into indices $\mathcal { T } _ { \mathrm { f i t } }$ used for fitting the predictor $\hat { f }$ and indices $\mathcal { T } _ { \mathrm { r e c } }$ used for debiasing, or “rectifying,” the final estimate. Since PPI assumes data and label-missingness are i.i.d., only the sizes $T _ { \mathrm { f i t } } : = | \overline { { \mathcal { T } } } _ { \mathrm { f i t } } | , T _ { \mathrm { r e c } } : = | \mathcal { T } _ { \mathrm { r e c } }$ | matter, not the order of the indices. We let ${ \hat { Y } } _ { t } ( s ) : = { \hat { f } } ( X _ { t } ( s ) )$ ) be the predicted response at location s and time t.

For location s, PPI constructs its point estimate $\widehat { \theta } ^ { \mathrm { P P I } } ( s )$ by starting from the empirical mean predicted response on the unlabeled data, ${ \hat { \mu } } ( s )$ . It computes an estimate $\hat { \Delta } ( s )$ of the expected bias $\Delta ( s )$ of ${ \hat { \mu } } ( s )$ and subtracts that from ${ \hat { \mu } } ( s ) \colon$

$$
\hat { \mu } ( s ) : = \frac { 1 } { T _ { \mathrm { U } } } \sum _ { t \in \mathcal { T } _ { \mathrm { U } } } \hat { Y } _ { t } ( s ) ,
$$

$$
\hat { \Delta } ( s ) : = \frac { 1 } { T _ { \mathrm { r e c } } } \sum _ { t \in \mathcal { T } _ { \mathrm { r e c } } } \Big ( \hat { Y } _ { t } ( s ) - Y _ { t } ( s ) \Big ) ,\tag{3}
$$

$$
\Delta ( s ) : = \frac { 1 } { T _ { \mathrm { U } } } \sum _ { t \in \mathcal { T } _ { \mathrm { U } } } \mathbb { E } \Big [ \Big ( \hat { Y } _ { t } ( s ) - Y _ { t } ( s ) \Big ) \Big ] , \qquad \hat { \theta } ^ { \mathrm { P P I } } ( s ) : = \hat { \mu } ( s ) - \hat { \Delta } ( s ) .\tag{4}
$$

Let $\hat { \sigma } _ { \hat { Y } } ^ { \mathrm { P P I } } ( s ) ^ { 2 }$ be the unbiased sample variance of the predicted responses and $\hat { \sigma } _ { \hat { Y } - Y } ^ { \mathrm { P P I } } ( s ) ^ { 2 }$ be the unbiased sample variance of the bias terms (see Eq. (32) and Eq. (33) in Appendix D). If the data is i.i.d., the point estimate $\widehat { \theta } ^ { \mathrm { P P I } } ( s )$ is the sum of two independent sample means; so, as discussed in [11], an asymptotically valid $( 1 - \alpha ) \%$ confidence interval is

$$
\hat { \theta } ^ { \mathrm { P P I } } ( s ) \pm z _ { 1 - \alpha / 2 } \sqrt { \hat { \sigma } _ { \hat { Y } } ^ { \mathrm { P P I } } ( s ) ^ { 2 } / T _ { \mathrm { U } } + \hat { \sigma } _ { \hat { Y } - Y } ^ { \mathrm { P P I } } ( s ) ^ { 2 } / T _ { \mathrm { r e c } } } .\tag{5}
$$

## 4 Our method: PPI for time series across space (PPITSAS)

Here, we discuss our proposed method PPITSAS for this problem, summarized in Algorithm 1 in Appendix A. We use this section to elucidate design choices that we make in our algorithm.

## 4.1 Splitting the data and training the predictor

Sharing information across space for a better predictor. As in standard PPI, we partition $\mathcal { T } _ { \mathrm { L } }$ into disjoint subsets $\mathcal { T } _ { \mathrm { f i t } }$ and $\mathcal { T } _ { \mathrm { r e c } }$ . One might naïvely apply PPI directly and separately to each spatial coordinate, including fitting a separate predictor to data from each spatial coordinate. However, given our sparsity of labeled data, we instead use data from $\mathcal { T } _ { \mathrm { f i t } }$ across all locations $s \in { \mathcal { S } }$ to fit a single predictor $\tilde { f } .$ With $N _ { S } \cdot T _ { \mathrm { f i t } }$ data points to fit $\hat { f } ,$ rather than the naïve $T _ { \mathrm { { f i t } } }$ , we can choose a much more modest $T _ { \mathrm { { f i t } } }$ —leaving more data for accurate debiasing.

Splitting the labeled data. Unlike in standard PPI, the precise set of indices assigned to $\mathcal { T } _ { \mathrm { f i t } }$ and $\mathcal { T } _ { \mathrm { r e c } }$ (rather than just their sizes) matters in our setting due to temporal dependencies in the data. We consider two options: chunking and interleaving. In a chunked split, we preserve the temporal ordering of the labeled data and divide $\mathcal { T } _ { \mathrm { L } }$ at a single cutoff, using the earliest $T _ { \mathrm { r e c } }$ time points to form $\mathcal { T } _ { \mathrm { r e c } }$ , and the last $T _ { \mathrm { { f i t } } }$ time points to form $\mathcal { T } _ { \mathrm { f i t } }$ . W use the later block of $\mathcal { T } _ { \mathrm { L } }$ for $\mathcal { T } _ { \mathrm { f i t } }$ so that the predictor is less correlated with $\mathcal { T } _ { \mathrm { U } }$ . In an interleaved split, we set aside observations at evenly spaced times for fitting.

A chunked split exhibits greater correlation within each chunk, compared to interleaving; since the predictor is trained on more contiguous, and hence more correlated data, we expect the chunked split to exhibit more prediction bias $( \Delta ( s )$ in Eq. (4)). However, for the chunked split (vs. the interleaved split), there is less correlation between the fitted chunk and the rectifying chunk. So we expect a better estimate $( \hat { \Delta } ( s )$ in Eq. (3)) of the prediction bias $\Delta ( s )$ than using the interleaved split. Finally, then, we expect that $\widehat { \theta } ^ { \mathrm { P P I } } ( s )$ in Eq. (4) applied with chunking should outperform the same estimator applied with interleaving, because its performance depends on the quality of its bias estimation.

To test these intuitions, we set up a simple autoregressive model (with a temporal component but no spatial component) in Appendix C. We let $T _ { \mathrm { U } } = 4 { , } 0 0 0 , T _ { \mathrm { L } } = 1 { , } 0 0 0 , T _ { \mathrm { f t } } = 1 0 0$ For the interleaved split, we put data points indexed $\{ 4 0 0 1 , 4 0 1 1 , \bar { 4 } 0 2 1 , . . . , 4 9 9 1 \}$ in $\mathcal { T } _ { \mathrm { f i t } }$ and the remaining points in $\mathcal { T } _ { \mathrm { r e c } }$ . We compute expectations and standard errors via Monte Carlo averages across $1 0 ^ { 5 }$ independent replicates. We see in Table 1 that the chunking approach has a higher prediction bias, but also a better estimate of that bias—and thus results in a final estimator $\hat { \theta } ^ { \mathrm { P P I } }$ with less overall bias.

<table><tr><td></td><td> $\Delta$ </td><td> $\mathbb { E } [ \hat { \Delta } ]$ </td><td> $\mathbb { E } [ \hat { \theta } ^ { \mathrm { P P I } } - \theta ]$ </td></tr><tr><td>Chunking</td><td>-0.606</td><td>-0.602</td><td>0.003</td></tr><tr><td>Interleaving</td><td>-0.080</td><td>-0.001</td><td>0.078</td></tr></table>

Table 1: For each row (using a chunked or interleaved split), we report true prediction bias (left column), estimated prediction bias (middle), and bias of the PPI estimate (right). In each column, we report the expected result by averaging over $1 0 ^ { \dot { 5 } }$ replicates. In every case, the standard error has size strictly less than $3 \times 1 0 ^ { - 3 }$ ; see Table 3 in Appendix C for the full standard errors.

Buffering. Even with the chunked split, there are temporally near data at the boundary between the fitting and rectifying sets. We could

choose to buffer this data by throwing out some of the observations at the boundary of the two sets, but buffering loses already-scarce labeled data. Alternative options include: (1) do not buffer, and assume that the correlations between the two datasets are negligible and (2) do not buffer but account for, and adjust for, correlations between the two datasets. Presently we choose option (1) based on rough preliminary experiments that suggest negligible changes from buffering; we reserve accounting for induced correlations for future work.

## 4.2 Applying HAC

Recall that $\widehat { \theta } ^ { \mathrm { P P I } } ( s )$ is the sum of two sample means. So the confidence interval in Eq. (5) is asymptotically valid roughly because, under an i.i.d. data assumption, (1) the two sample means are independent and $( 2 ) \hat { \sigma } _ { \hat { \nu } } ^ { \mathrm { P P I } } ( \mathbf { \bar { s } } ) ^ { 2 } / T _ { \mathrm { U } }$ and $\hat { \sigma } _ { \hat { Y } - Y } ^ { \mathrm { P P I } } ( s ) ^ { 2 } / T _ { \mathrm { r e c } }$ are “good” estimates of the variance of the respective sample means, ${ \hat { \mu } } ( s )$ and $\hat { \Delta } ( s ) ^ { \ j }$ . It remains to make a similar argument, and develop an analogous confidence interval, when we no longer have the i.i.d. assumption, but instead make a temporal stationarity assumption. We sketch out such an argument here and provide more details, including a detailed discussion of the challenges of a formal proof, in Appendix E.2.

Separate treatment of ${ \hat { \mu } } ( s )$ and $\hat { \Delta } ( s )$ under temporal dependence. Under temporal dependence, it will no longer generally hold that ${ \hat { \mu } } ( s )$ and $\hat { \Delta } ( s )$ are independent. Instead, we observe that, roughly, under temporal stationarity and standard regularity assumptions discussed in Appendix $\mathrm { E , }$ we can expect ${ \hat { \mu } } ( s )$ and $\hat { \Delta } ( s )$ to be asymptotically uncorrelated. If the two are asymptotically normal, this observation will be enough to treat them separately, as in the standard PPI case.

Toward a valid confidence interval in the Newey–West regime. HAC methods compute estimated autocorrelations and use them to adjust variance estimates. They have traditionally been applied to the classical estimator we describe in Section 1. But in the present work we propose to apply HAC to adjust variance estimates for ${ \hat { \mu } } ( s )$ and $\hat { \Delta } ( s )$ . A parameter called the bandwidth describes how far (across time) we assume autocorrelations can be non-zero. There are two major asymptotic regimes to consider. The first is the Newey–West asymptotic regime [14, Theorem 2], in which the bandwidth is a vanishing proportion of the sample size. The second is the Kiefer–Vogelsang asymptotic regime [18] (often called “fixed-b” asymptotics), in which the bandwidth approaches a fixed proportion of the sample size.

We first focus on the Newey–West asymptotic limit, where the bandwidth grows more slowly than the length of the labeled and unlabeled time series. In this setting, we conjecture that under reasonable stationarity, mixing, and moment conditions on the response and covariates, as well as a moment condition on the predictor, the following is an asymptotically valid $( 1 - \alpha ) \%$ confidence interval for $\theta ( s )$ in our setting:

$$
\hat { \theta } ^ { \mathrm { P P I } } ( s ) \pm z _ { 1 - \alpha / 2 } \sqrt { \hat { \sigma } _ { \hat { Y } } ^ { \mathrm { H A C } } ( s ) ^ { 2 } / T _ { \mathrm { U } } + \hat { \sigma } _ { Y - \hat { Y } } ^ { \mathrm { H A C } } ( s ) ^ { 2 } / T _ { \mathrm { r e c } } } .\tag{6}
$$

We provide a sketch for why this conjecture should be true in Appendix $\mathrm { E } . 2 .$ . Very roughly, we can apply the Newey–West procedure to find “good” (consistent) variance estimates for the individual terms in the sums in ${ \hat { \mu } } ( s )$ and $\hat { \Delta } ( s )$ ; call them $\hat { \sigma } _ { \hat { Y } } ^ { \mathrm { H A C } } ( s ) ^ { 2 }$ and $\hat { \sigma } _ { \hat { Y } - Y } ^ { \mathrm { H A C } } ( s ) ^ { 2 }$ , respectively (see Eq. (9) and $\operatorname { E q . }$ (10) in Algorithm 1). It would then follow from the asymptotic normality of ${ \hat { \mu } } ( s )$ and $\hat { \Delta } ( s )$ , and their asymptotic uncorrelatedness, that the confidence interval in Eq. (6) has the claimed coverage.

Newey–West vs. Kiefer–Vogelsang. In the Kiefer–Vogelsang asymptotics, the limit of the variance estimate is not a constant but a non-degenerate random variable, and so we need not have normality of ${ \hat { \mu } } ( s )$ and $\hat { \Delta } ( s )$ . Given the resulting added complications of working with Kiefer–Vogelsang and also the standard use of Newey–West in the literature, we henceforth use Newey–West. Additionally, in our motivating example of weather risk and insurance, we have several years of data and expect that the data decorrelates on a scale of several weeks. In this setting where bandwidth is much shorter than the amount of available data, the Newey–West asymptotics should be reasonable.

Newey–West practical choices. We use the Newey–West procedure with the Bartlett kernel [14] and the data-driven plug-in estimate of the bandwidth [17].

## 5 Experiments

After describing our setup, we show that our method (PPITSAS) outperforms alternative methods.

## 5.1 Experimental setup

Necessity of simulated data. Unlike other common predictive machine learning settings, standard data sets (including Generali’s insurance data) do not have ground truth for our setting. To understand why, consider again the thought experiment from Section 1, where the true (but unknown) mean annual payout is 0.05, but the empirical mean payout is 0.020. If we use the observed data to determine “ground truth” (by treating the empirical mean as “ground truth”), a method that estimates the true mean annual payout perfectly will be said to perform “worse” (i.e., be farther from the putative “ground truth”) than methods that underestimate the payout. Simulated data, by contrast, will allow us to access real ground truth.

Simulation details. We aim to create a setup with many similar aspects to the real insurance problem (and other real data problems). See Appendix F.1 for full details of our setup. We take a lattice of $N s = 1 2 1$ locations. We imagine the scalar covariate $X _ { t } ( s )$ as a daily weather measurement and $Y _ { t } ( s )$ as a daily payout. We take $T _ { \mathrm { L } } = 4 { , } 0 0 0$ labeled times (roughly, 10 years) and $T _ { \mathrm { U } } = 1 6 { , } 0 0 0$ unlabeled times (roughly, 40 years). We generate $X _ { t } ( s )$ as a transformation of a Gaussian process across the spatial coordinate. In real life, we generally expect that the predictor cannot fully capture the response’s relationship to the covariates (e.g. due to unobserved but important covariates). As a particularly simple and interpretable version of this issue, we take $\dot { Y _ { t } ( s ) }$ to have a quadratic dependence on $X _ { t } ( s )$ and use a linear predictor. We generate versions with more-extreme weather events (heavy-tailed $X _ { t } ( s )$ values, which are more realistic for the insurance application) and lessextreme weather events.

Performance metrics. For each method, we report point estimate quality, confidence interval coverage, and confidence interval width. See Appendix F.2 for full details. For the point estimate, we compute a scaled root mean square error (RMSE); more specifically, for each s and each draw from our data-generating process, we compute the squared error normalized by the (square of) the ground truth $\theta \bar { ( } s )$ . We average over these cases and finally take the square root. The scaling allows greater interpretability of the final number; e.g., an RMSE of 1 would suggest errors are on the order of $\theta ( s )$ itself in size. Likewise, for confidence interval width, we report the scaled half-width for interpretability, where we again scale by $\theta ( s )$ . We imagine a practitioner interested in a one-sided hypothesis test—e.g., to test whether the mean response is above a certain threshold. A scaled half-width of 1 suggests the test cannot distinguish $\theta ( s )$ from 0.

<table><tr><td rowspan="2"></td><td colspan="3">Less Extreme Weather</td><td colspan="3">More Extreme Weather</td></tr><tr><td>Scaled Method RMSE</td><td>Coverage</td><td>Scaled Half-Width</td><td>Scaled RMSE</td><td>Coverage</td><td>Scaled Half-Width</td></tr><tr><td>Classical</td><td>0.0503</td><td>30.02%</td><td>0.0194</td><td>0.0372</td><td>31.56%</td><td>0.0147</td></tr><tr><td>IWI</td><td>0.0399</td><td>21.19%</td><td>0.0088</td><td>0.0302</td><td>20.79%</td><td>0.0064</td></tr><tr><td>chunkPPI</td><td>0.0299</td><td>30.91%</td><td>0.0115</td><td>0.0218</td><td>32.88%</td><td>0.0087</td></tr><tr><td>HAC</td><td>0.0503</td><td>91.72%</td><td>0.0871</td><td>0.0372</td><td>88.00%</td><td>0.0593</td></tr><tr><td>PPITSAS (ours)</td><td>0.0299</td><td>93.45%</td><td>0.0527</td><td>0.0218</td><td>92.64%</td><td>0.0362</td></tr><tr><td>Oracle</td><td>0.0221</td><td>93.72%</td><td>0.0410</td><td>0.0166</td><td>92.12%</td><td>0.0294</td></tr></table>

Table 2: For each method, we report the performance metrics in the less extreme weather setting (left) and the more extreme weather setting (right). The best performance for each metric, according to the goalposts defined in Section 5.1, are shaded in green; in the event of a tie, all winners are shaded. We ignore scaled half-widths for coverages below 85%, and shade them in gray.

Competing methods. We compare to the classical method; inference with imputation (IWI); “chunkPPI,” a modification of PPI using our chunking approach from Section 4.1; and HAC (applied to the classical method). For all methods using a predictor (IWI, chunkPPI, ours), we train the predictor across spatial coordinates as described in Section 4.1.

Goalposts. We first look for methods with better point estimation (i.e., lower RMSE). Second, we look for methods that meet near-nominal coverage. Since our nominal coverage is 95%, we will consider methods to be near-nominal with 85% or higher coverage. Finally, only for methods with near-nominal coverage, we look for methods to have a lower half-width. We emphasize that half-width is meaningless without coverage; it is trivial to make a confidence interval with arbitrarily small width if we do not require coverage.

To provide a bound on what performance is possible, we also include an oracle comparison; in particular, since we have access to ground truth, we imagine a world where we have responses at all time points (both $\mathcal { T } _ { \mathrm { L } }$ and $\mathcal { T } _ { \mathrm { U } } )$ and compute the classical estimate and HAC confidence intervals from this (dramatically extended) data—which we do not actually have access to in the problem.

## 5.2 Results

Table 2 displays the performance metrics in the less extreme weather and more extreme weather settings. In both settings, we observe the same patterns. The classical method performs poorly, with the worst scaled RMSEs of all the methods considered. The IWI method improves on the classical method when it comes to point estimation, but this comes at the cost of even worse coverage. The chunkPPI method further improves on point estimation over IWI, with the lowest scaled RMSEs of all methods considered. Our method (PPITSAS) achieves the best point estimation of all methods considered; it ties with chunkPPI since they use the same point estimate. In the less extreme weather setting, the scaled RMSE of chunkPPI and PPITSAS is about 35% higher than the scaled RMSE of the oracle. In the more extreme weather setting, the scaled RMSE of chunkPPI and PPITSAS is about 31% higher than the scaled RMSE of the oracle and at least 27% lower than the scaled RMSE of any of the other methods.

The classical method, IWI, and chunkPPI all fail to achieve coverage close to nominal. As such, their scaled half-widths are not relevant. While HAC performs the same as the classical method in point estimation (since they use the same point estimate), it is able to substantially improve coverage. PPITSAS performs even better than $\mathrm { H A C } .$ , getting the closest to nominal coverage in both settings and essentially tying with the oracle. Between the two methods attaining near-nominal coverages (HAC and PPITSAS), PPITSAS has the smaller scaled half-widths. Thus PPITSAS is uniformly the best in both settings: it has the lowest scaled RMSEs of all of the methods; it has the best coverages; and, among the methods with near-nominal coverages, it has the smallest scaled half-widths.

## Acknowledgments and disclosure of funding

The authors thank Generali for supporting this work and for their data and insight. VS was supported in part by a MathWorks Fellowship. RB was supported in part by a Panasonic AI and Wellness Fellowship. The authors thank Tyler McCormick and Stephen Bates for helpful discussions during the early stages of this work. Research described in this article was conducted under contract to the Health Effects Institute (HEI), an organization jointly funded by the United States Environmental Protection Agency (EPA) (Assistance Award No. CR-84114801) and certain motor vehicle and engine manufacturers. The contents of this article do not necessarily reflect the views of HEI, or its sponsors, nor do they necessarily reflect the views and policies of the EPA or motor vehicle and engine manufacturers.

## References

[1] Luke Lupone, Raylene Cooke, Anthony R. Rendall, Angelina Siegrist, Cara Penton, Matt Carlyon, Tim Ouchtomsky, and John G. White. Hindcasting long-term data unveils the influence of a changing climate on small mammal communities. Diversity and Distributions, 30(10): e13901, 2024.

[2] Joshua Fan, Junwen Bai, Zhiyun Li, Ariel Ortiz-Bobea, and Carla P. Gomes. A GNN-RNN approach for harnessing geospatial and temporal information: Application to crop yield prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 36(11):11873–11881, 2022.

[3] Allan C. Just, Robert O. Wright, Joel Schwartz, Brent A. Coull, Andrea A. Baccarelli, Martha María Tellez-Rojo, Emily Moody, Yujie Wang, Alexei Lyapustin, and Itai Kloog. Using high-resolution satellite aerosol optical depth to estimate daily PM geographical distribution in Mexico City. Environmental Science & Technology, 49(14):8576–8584, 2015.

[4] Qingyang Xiao, Guannan Geng, Shigan Liu, Jiajun Liu, Xia Meng, and Qiang Zhang. Spatiotemporal continuous estimates of daily 1 km PM from 2000 to present under the tracking air pollution in China (TAP) framework. Atmospheric Chemistry and Physics, 22(19):13229–13242, 2022.

[5] Eric R. Gamazon, Heather E. Wheeler, Kaanan P. Shah, Sahar V. Mozaffari, Keston Aquino-Michaels, Robert J. Carroll, Anne E. Eyler, Joshua C. Denny, GTEx Consortium, Dan L. Nicolae, et al. A gene-based association method for mapping traits using reference transcriptome data. Nature genetics, 47(9):1091–1098, 2015.

[6] Alexander Gusev, Kate Lawrenson, Xianzhi Lin, Paulo C. Lyra Jr, Siddhartha Kar, Kevin C. Vavra, Felipe Segato, Marcos A.S. Fonseca, Janet M. Lee, Tanya Pejovic, et al. A transcriptomewide association study of high-grade serous epithelial ovarian cancer identifies new susceptibility genes and splice variants. Nature genetics, 51(5):815–823, 2019.

[7] Ramon Viñas, Chaitanya K. Joshi, Dobrik Georgiev, Phillip Lin, Bianca Dumitrascu, Eric R. Gamazon, and Pietro Lio. Hypergraph factorization for multi-tissue gene expression imputation. Nature machine intelligence, 5(7):739–753, 2023.

[8] Jue Hou, Zijian Guo, and Tianxi Cai. Surrogate assisted semi-supervised inference for high dimensional risk prediction. Journal ofMachine Learning Research, 24(265):1–58, 2023.

[9] Siruo Wang, Tyler McCormick, and Jeffrey T. Leek. Methods for correcting inference based on outcomes predicted by machine learning. Proceedings ofthe National Academy ofSciences, 117(48):30266–30275, 2020.

[10] Keshav Motwani and Daniela Witten. Revisiting inference after prediction. Journal of Machine Learning Research, 24(394):1–18, 2023.

[11] Anastasios N. Angelopoulos, Stephen Bates, Clara Fannjiang, Michael I. Jordan, and Tijana Zrnic. Prediction-powered inference. Science, 382(6671):669–674, 2023.

[12] Anastasios N. Angelopoulos, John C. Duchi, and Tijana Zrnic. PPI++: Efficient predictionpowered inference. arXiv preprint arXiv:2311.01453, 2024.

[13] Halbert White and Ian Domowitz. Nonlinear regression with dependent observations. Econometrica, 52(1):143–161, 1984. ISSN 00129682, 14680262.

[14] Whitney K. Newey and Kenneth D. West. A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3):703–708, 1987. ISSN 00129682, 14680262.

[15] Wouter J. den Haan and Andrew T. Levin. A practitioner’s guide to robust covariance matrix estimation. In Robust Inference, volume 15 of Handbook ofStatistics, pages 299–342. Elsevier, 1997.

[16] Eben Lazarus, Daniel J. Lewis, James H. Stock, and Mark W. Watson. HAR inference: Recommendations for practice. Journal of Business & Economic Statistics, 36(4):541–559, 2018.

[17] Whitney K. Newey and Kenneth D. West. Automatic lag selection in covariance matrix estimation. The Review ofEconomic Studies, 61(4):631–653, 10 1994. ISSN 0034-6527.

[18] Nicholas M. Kiefer and Timothy J. Vogelsang. A new asymptotic theory for heteroskedasticityautocorrelation robust tests. Econometric Theory, 21(6):1130–1164, 2005. ISSN 02664666, 14694360.

[19] Yilin Song, Dan M. Kluger, Harsh Parikh, and Tian Gu. Demystifying prediction powered inference. arXiv preprint arXiv:2601.20819, 2026.

[20] Dan M. Kluger, Kerri Lu, Tijana Zrnic, Sherrie Wang, and Stephen Bates. Prediction-powered inference with imputed covariates and nonuniform sampling. arXiv preprint arXiv:2501.18577, 2025.

[21] Stephen Salerno, Zhenke Wu, and Tyler McCormick. Spatially robust inference with predicted and missing at random labels. arXiv preprint arXiv:2603.11368, 2026.

[22] Daniel Csillag, Claudio José Struchiner, and Guilherme Tegoni Goedert. Prediction-powered e-values. In Forty-second International Conference on Machine Learning, 2025.

[23] Daniel Csillag, Pedro Dall’Antonia, Claudio José Struchiner, and Guilherme Tegoni Goedert. Extending prediction-powered inference through conformal prediction. arXiv preprint arXiv:2510.16166, 2025.

[24] James D. Hamilton. Time Series Analysis. Princeton University Press, 1994.

[25] Richard C. Bradley. Basic properties of strong mixing conditions. a survey and some open questions. Probability Surveys, 2:107 – 144, 2005.

[26] Donald W. K. Andrews. Heteroskedasticity and autocorrelation consistent covariance matrix estimation. Econometrica, 59(3):817–858, 1991. ISSN 00129682, 14680262.

[27] Youri A. Davydov. Convergence of distributions generated by stationary stochastic processes. Theory ofProbability & Its Applications, 13(4):691–696, 1968.

[28] Emmanuel Rio. Covariance inequalities for strongly mixing processes. Annales de l’I.H.P. Probabilités et statistiques, 29(4):587–597, 1993.

[29] Ildar. A. Ibragimov. Some limit theorems for stationary processes. Theory of Probability & Its Applications, 7(4):349–382, 1962.

## A PPITSAS algorithm

For the HAC procedure, see Appendix B.

Algorithm 1 PPITSAS. PPI for time-series across space   
Input: Target location s, significance level $\alpha ,$ fitting proportion $p _ { \mathrm { { f i t } } }$ , and fitting algorithm $\mathsf { A l g } _ { \mathrm { f i t } }$   
Input: Labeled data with size $T _ { \mathrm { L } } : = | \mathcal { T } _ { \mathrm { L } } |$   
D<sub>L</sub> = (X<sub>t</sub>(s<sup>′</sup>), Y<sub>t</sub>(s<sup>′</sup>))	<sub>t∈T , s′∈S</sub>   
and unlabeled data with size $T _ { \mathrm { U } } : = | \mathcal { T } _ { \mathrm { U } } |$   
D<sub>U</sub> = X<sub>t</sub>(s<sup>′</sup>)	<sub>∈T , s′∈S</sub> .   
1: Partition $\mathcal { T } _ { \mathrm { L } }$ into disjoint, consecutive rectifying and fitting sets $\mathcal { T } _ { \mathrm { r e c } }$ and $\mathcal { T } _ { \mathrm { f i t } }$ with sizes $T _ { \mathrm { r e c } } : =$   
$| \mathcal { T } _ { \mathrm { r e c } } |$ and $T _ { \mathrm { f i t } } : = | \dot { T _ { \mathrm { f i t } } } |$ such that $T _ { \mathrm { f i t } } = p _ { \mathrm { f i t } } \cdot \dot { T } _ { \mathrm { L } }$   
2: Fit the predictor   
$\begin{array} { r } { \hat { f }  \mathsf { A l g } _ { \mathrm { f i t } } \Big ( \big \{ ( X _ { t } ( s ^ { \prime } ) , Y _ { t } ( s ^ { \prime } ) ) : t \in \mathcal { T } _ { \mathrm { f i t } } , s ^ { \prime } \in \mathcal { S } \big \} \Big ) . } \end{array}$   
Denote ${ \hat { Y } } _ { t } ( s ) : = { \hat { f } } ( X _ { t } ( s ) )$ for each $t .$   
3: Compute the average prediction and bias at s:   
$\hat { \mu } ( s ) \longleftarrow \frac { 1 } { T _ { \mathrm { U } } } \sum _ { t \in \mathcal { T } _ { \mathrm { U } } } \hat { Y } _ { t } ( s ) ,$ (7)   
∆( <sup>ˆ</sup> s) ← 1 X Yˆt(s) − Yt(s). (8)   
T<sub>rec</sub> t∈T<sub>rec</sub>   
4: Estimate the corresponding long-run variances:   
σˆ<sup>2</sup><sub>Yˆ</sub> (s) ← HAC Yˆt(s)	<sub>t∈T</sub>  , (9)   
σˆ<sup>2</sup><sub>Yˆ</sub> <sub>−Y</sub>(s) ← HAC Yˆt(s) − Yt(s)	<sub>t∈Trec</sub> . (10)   
5: Form the PPITSAS point estimate   
<sup>ˆ</sup>θ<sup>PPITSAS</sup>(s) ← µˆ(s) − ∆( <sup>ˆ</sup> s). (11)   
Output: Point estimate $\hat { \theta } ^ { \mathrm { P P I T S A S } } ( s )$ and $( 1 - \alpha ) ^ { \mathfrak { ( } }$ % confidence interval   
σˆ<sup>2</sup><sub>Yˆ</sub> (s)   
<sup>ˆ</sup>θ<sup>PPITSAS</sup>(s) ± z<sub>1−α/2</sub> T<sub>U</sub> + T<sub>rec</sub> (12)

## B Newey–West procedure

Algorithm 2 HAC. Heteroskedasticity and autocorrelation consistent variance estimation [14]   
Input: Time-series observations $\{ W _ { t } \} _ { t = 1 } ^ { T }$   
Output: Newey–West long-run variance estimate $\hat { \sigma } ^ { 2 }$   
1: Compute the sample mean   
$\bar { W }  \frac { 1 } { T } \sum _ { t = 1 } ^ { T } W _ { t } .$ (13)   
2: Center the observations:   
$\xi _ { t } : = W _ { t } - \bar { W } , \qquad t \in \{ 1 , \ldots , T \}$ (14)   
3: Compute the sample autocovariances   
T−j   
1   
γˆ<sub>j</sub> ← <sub>T</sub> X ξ<sub>t</sub>ξ<sub>t+j</sub> , j ∈ {0, . . . , T − 1}. (15)   
t=1   
4: Set the pilot bandwidth   
T 2/9   
m ← 4 (16)   
100   
5: Compute   
m m   
sˆ<sup>(0)</sup> ← γˆ<sub>0</sub> + 2 X γˆ<sub>j</sub>, sˆ<sup>(1)</sup> ← 2 X jγˆ<sub>j</sub> . (17)   
j=1 j=1   
6: Compute the Newey–West bandwidth   
L<sup>ˆ</sup> ← <sup>3</sup><sub>2</sub> 1/3 T<sup>1/3</sup> <sup></sup> sˆ<sup>(1)</sup> 2/3 (18)   
<sup></sup> sˆ<sup>(0)</sup>   
7: Compute the HAC long-run variance estimate   
Lˆ   
σˆ <sup>2</sup> ← γˆ<sub>0</sub> + 2 X 1 − <sup>h</sup><sub>Lˆ</sub> <sub>+ 1</sub> γˆ<sub>j</sub> . (19)   
8: return ${ \hat { \sigma } } ^ { 2 } .$

## C Data splitting demonstration

Consider the following simple autoregressive model, which, for the sake of demonstration, has a temporal dimension only. For $X _ { 1 } \sim \tilde { \mathcal { N } } ( 0 , 1 ) , \{ Z _ { t } \} _ { t \in \mathbb { N } } { \overset { \mathrm { i i d } } { \sim } } \mathcal { N } ( 0 , 1 )$ , and $| \rho | < 1$ , the process is defined for each $t \in \mathbb { N }$ by

$$
X _ { t } = \rho X _ { t - 1 } + \sqrt { 1 - \rho ^ { 2 } } Z _ { t }
$$

$$
Y _ { t } = X _ { t } ^ { 2 } .\tag{20}
$$

(21)

This process is covariance-stationary with absolutely summable covariances. We aim to estimate

$$
\theta : = \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } Y _ { t } .\tag{22}
$$

Note that $X _ { t } \sim \mathcal { N } ( 0 , 1 )$ for each $t \in \mathbb { N } .$ By stationarity, $\theta = \mathbb { E } [ Y _ { 1 } ] = \mathbb { E } [ X _ { 1 } ^ { 2 } ] = 1$

We use this example to demonstrate that while interleaving yields a less biased predictor than chunking, it also yields a poorer estimate of the prediction bias. We take $\rho = 0 . 9 5 , T _ { \mathrm { L } } = 1 0 0 0$ $T _ { \mathrm { U } } = 4 0 0 0 , T _ { \mathrm { f i t } } \stackrel { \cdot } { = } 1 0 0$ . We have $\mathcal { T } _ { \mathrm { U } } = \{ 1 , \dots , 4 0 \mathrm { 0 0 } \}$ and $\mathcal { T } _ { \mathrm { L } } = \{ 4 0 0 1 , . . . , 5 0 0 0 \}$

For the chunking approach, we take $\mathcal { T } _ { \mathrm { r e c } } ^ { \mathrm { c h u n k } } = \{ 4 0 0 1 , \dots , 4 9 0 0 \}$ and $\mathcal { T } _ { \mathrm { f i t } } ^ { \mathrm { c h u n k } } = \{ 4 9 0 1 , \dots , 5 0 0 0 \}$ For the interleaving approach, we take $\mathcal { T } _ { \mathrm { f i t } } ^ { \mathrm { i n t e r } } = \{ 4 0 0 1 , 4 0 1 1 , 4 0 2 1 , \dots , 4 9 9 1 \}$ and $\mathcal { T } _ { \mathrm { r e c } } ^ { \mathrm { i n t e r } } = \mathcal { T } _ { \mathrm { L } } \backslash \mathcal { T } _ { \mathrm { f i t } }$

<table><tr><td></td><td>E[∆]</td><td> $\mathrm { E } [ \hat { \Delta } ]$ </td><td> $\mathrm { E } [ \hat { \theta } ^ { \mathrm { P P I } } - \theta ]$ </td></tr><tr><td>Chunking</td><td>-0.6058 (0.0020)</td><td>-0.6024 (0.0022)</td><td>-0.0029 (0.0011)</td></tr><tr><td>Interleaving</td><td>-0.0796 (0.0008)</td><td>−0.0012 (0.0002)</td><td>–0.0778 (0.0007)</td></tr></table>

Table 3: For each row (using a chunked or interleaved split of the labeled data), we report the true prediction bias (left column), estimated prediction bias (middle column), and bias of the PPI estimate (right column). In each column, we report the result averaged over $\mathrm { 1 0 ^ { 5 } }$ replicates. Monte Carlo standard errors are reported in parentheses.

We compute expectations and standard errors via Monte Carlo averages across $1 0 ^ { 5 }$ independent replicates and report them in Table 3. As expected, we see that the chunking approach has a higher prediction bias, but also a better estimate of that bias. The PPI estimator $\hat { \theta } ^ { \mathrm { P P I } }$ is less biased when using the chunking approach rather than the interleaving approach.

## C.1 Code and reproducibility

The code to generate the simulated data, to run the chunking and interleaving methods, and to compute the reported metrics is available at https://github.com/shahzarrizvi/ppitsas. The experiment was run locally on a MacBook Pro with an Apple M3 Max chip, consisting of a 14-core CPU (10 performance and 4 efficiency cores), and 36 GB of unified memory. The experiments were CPU-based and did not use GPU acceleration. Generating the simulated data and running the chunking and interleaving methods took less than 5 minutes of computation.

## D Previous methods

In what follows below, let $s \in S$ be the location at which the mean response is being estimated.

## D.1 Classical

The classical approach estimates the mean response simply using the sample mean of the observed responses.

$$
\hat { \theta } ^ { \mathrm { C L A } } ( s ) : = \frac { 1 } { T _ { \mathrm { L } } } \sum _ { t \in \mathcal { T } _ { \mathrm { L } } } Y _ { t } ( s ) .\tag{23}
$$

The confidence interval is constructed by leveraging asymptotic normality. Estimate the variance as

$$
\hat { \sigma } ^ { \mathrm { C L A } } ( s ) ^ { 2 } : = \frac { 1 } { T _ { \mathrm { L } } - 1 } \sum _ { t \in \mathcal { T } _ { \mathrm { L } } } \left( Y _ { t } ( s ) - \hat { \theta } ^ { \mathrm { C L A } } ( s ) \right) ^ { 2 } .\tag{24}
$$

If $\{ Y _ { t } ( s ) \} _ { t \in T _ { \mathrm { L } } }$ are independent and identically distributed, then, by the central limit theorem,

$$
\sqrt { T _ { \mathrm { L } } } \cdot \frac { \hat { \theta } ^ { \mathrm { C L A } } ( s ) - \theta ( s ) } { \hat { \sigma } ^ { \mathrm { C L A } } ( s ) } \implies \mathcal { N } ( 0 , 1 ) \quad \mathrm { a s } \quad T _ { \mathrm { L } }  \infty .\tag{25}
$$

Under these assumptions, an asymptotically valid confidence interval is

$$
\hat { \theta } ^ { \mathrm { C L A } } ( s ) \pm z _ { 1 - \alpha / 2 } \frac { \hat { \sigma } ^ { \mathrm { C L A } } ( s ) } { \sqrt { T _ { \mathrm { L } } } } .\tag{26}
$$

## D.2 Inference with imputation (IWI)

Recall that $\hat { Y } _ { t } ( s )$ is the predicted response at location s and time t. The IWI estimate is constructed by averaging over the observed responses which were not used for fitting the predictor along with the long sequence of predicted responses.

$$
\hat { \theta } ^ { \mathrm { I W I } } ( s ) : = \frac { 1 } { T _ { \mathrm { r e c } } + T _ { \mathrm { U } } } \left( \sum _ { t \in \mathcal { T } _ { \mathrm { r e c } } } Y _ { t } ( s ) + \sum _ { t \in \mathcal { T } _ { \mathrm { U } } } \hat { Y } _ { t } ( s ) \right) .\tag{27}
$$

When constructing the confidence interval, the predicted responses are treated as true responses. The variance estimate is

$$
\hat { \sigma } ^ { \mathrm { I W I } } ( s ) ^ { 2 } : = \frac { 1 } { T _ { \mathrm { r e c } } + T _ { \mathrm { U } } - 1 } \left( \sum _ { t \in \mathcal { T } _ { \mathrm { r e c } } } \left( Y _ { t } ( s ) - \hat { \theta } ^ { \mathrm { I W I } } ( s ) \right) ^ { 2 } + \sum _ { t \in \mathcal { T } _ { \mathrm { U } } } \left( \hat { Y } _ { t } ( s ) - \hat { \theta } ^ { \mathrm { I W I } } ( s ) \right) ^ { 2 } \right) .\tag{28}
$$

This approach is not generally inferentially valid, even under i.i.d. sampling [9, 10]. But ignoring this and applying a normal $( 1 - \alpha ) \%$ confidence interval yields

$$
\hat { \theta } ^ { \mathrm { I W I } } ( s ) \pm z _ { 1 - \alpha / 2 } \frac { \hat { \sigma } ^ { \mathrm { I W I } } ( s ) } { \sqrt { T _ { \mathrm { r e c } } + T _ { \mathrm { U } } } } .\tag{29}
$$

## D.3 Prediction-powered inference (PPI)

The PPI estimate is constructed by estimating the mean predicted response on the unlabeled data and the mean prediction bias on the labeled data which was not already used for fitting the predictor.

$$
\hat { \mu } ( s ) : = \frac { 1 } { T _ { \mathrm { U } } } \sum _ { t \in { \cal T } _ { \mathrm { U } } } \hat { Y } _ { t } ( s ) \qquad \hat { \Delta } ( s ) : = \frac { 1 } { T _ { \mathrm { r e c } } } \sum _ { t \in { \cal T } _ { \mathrm { r e c } } } \Big ( \hat { Y } _ { t } ( s ) - Y _ { t } ( s ) \Big ) .\tag{30}
$$

The point estimate $\widehat { \theta } ^ { \mathrm { P P I } } ( s )$ debiases the average predicted response ${ \hat { \mu } } ( s )$ by the average prediction bias $\hat { \Delta } ( s )$

$$
\hat { \theta } ^ { \mathrm { P P I } } ( s ) : = \hat { \mu } ( s ) - \hat { \Delta } ( s ) .\tag{31}
$$

The variances of the predicted responses and the biases are estimated separately.

$$
\hat { \sigma } _ { \hat { Y } } ^ { \mathrm { P P I } } ( s ) ^ { 2 } : = \frac { 1 } { T _ { \mathrm { U } } - 1 } \sum _ { t \in \mathcal { T } _ { \mathrm { U } } } \left( \hat { Y } _ { t } { ( s ) } - \hat { \mu } { ( s ) } \right) ^ { 2 }\tag{32}
$$

$$
\hat { \sigma } _ { Y - \hat { Y } } ^ { \mathrm { P P I } } ( s ) ^ { 2 } : = \frac { 1 } { T _ { \mathrm { r e c } } - 1 } \sum _ { t \in \mathcal { T } _ { \mathrm { r e c } } } \left( \left( \hat { Y } _ { t } ( s ) - Y _ { t } ( s ) \right) - \hat { \Delta } ( s ) \right) ^ { 2 }\tag{33}
$$

If the data is independent and identically distributed, the point estimate $\widehat { \theta } ^ { \mathrm { P P I } } ( s )$ is the sum of two independent sample means; as discussed in [11], an asymptotically valid $( 1 - \alpha ) ^ { \prime }$ % confidence interval is

$$
\hat { \theta } ^ { \mathrm { P P I } } ( s ) \pm z _ { 1 - \alpha / 2 } \sqrt { \frac { \hat { \sigma } _ { \hat { Y } } ^ { \mathrm { P P I } } ( s ) ^ { 2 } } { T _ { \mathrm { U } } } + \frac { \hat { \sigma } _ { Y - \hat { Y } } ^ { \mathrm { P P I } } ( s ) ^ { 2 } } { T _ { \mathrm { r e c } } } } .\tag{34}
$$

## D.4 Heteroskedasticity and autocorrelation consistent procedures (HAC)

This method modifies the classical approach in Appendix D.1 with a more suitable variance estimate. The point estimate is the same as $ { \hat { \theta } } ^ { \mathrm { C L A } } ( s )$

$$
\hat { \theta } ^ { \mathrm { H A C } } ( s ) : = \frac { 1 } { T _ { \mathrm { L } } } \sum _ { t \in \mathcal { T } _ { \mathrm { L } } } Y _ { t } ( s ) = \hat { \theta } ^ { \mathrm { C L A } } ( s ) .\tag{35}
$$

In lieu of the sample variance, we instead estimate the variance using the HAC variance estimator— summarized in Algorithm 2—as

$$
\hat { \sigma } ^ { \mathrm { H A C } } ( s ) ^ { 2 } = \mathsf { H A C } \Big ( \big \{ Y _ { t } ( s ) \big \} _ { t \in \mathcal { T } _ { \mathrm { L } } } \Big ) .\tag{36}
$$

Under certain assumptions on the process $\{ Y _ { t } ( s ) \} _ { t \in \mathbb { N } }$ (which we elaborate on in Appendix E.1), we obtain asymptotic normality of the form

$$
\sqrt { T _ { \mathrm { L } } } \cdot \frac { \hat { \theta } ^ { \mathrm { H A C } } ( s ) - \theta ( s ) } { \hat { \sigma } ^ { \mathrm { H A C } } ( s ) } \Rightarrow { \cal N } ( 0 , 1 ) .\tag{37}
$$

Consequently, an asymptotically valid $( 1 - \alpha )$ % confidence interval is

$$
\hat { \theta } ^ { \mathrm { H A C } } ( s ) \pm z _ { 1 - \alpha / 2 } \frac { \hat { \sigma } ^ { \mathrm { H A C } } ( s ) } { \sqrt { T _ { \mathrm { L } } } } .\tag{38}
$$

## D.5 Oracle

The oracle method is named as such since it has access to $\{ Y _ { t } ( s ) \} _ { t \in \pi }$ . This additional data (which is otherwise unavailable in realistic scenarios) changes the point estimate and the variance estimates.

We have a point estimate:

$$
\hat { \theta } ^ { \mathrm { O R A } } ( s ) : = \frac { 1 } { T _ { \mathrm { L } } + T _ { \mathrm { U } } } \sum _ { t \in \mathcal { T } _ { \mathrm { L } } \cup \mathcal { T } _ { \mathrm { U } } } Y _ { t } ( s ) .\tag{39}
$$

The variance estimate $\hat { \sigma } ^ { \mathrm { O R A } } ( s ) ^ { 2 }$ is obtained via the HAC variance estimator (Algorithm 2) as

$$
\hat { \sigma } ^ { \mathrm { O R A } } ( s ) ^ { 2 } = \mathsf { H A C } \bigg ( \big \{ Y _ { t } ( s ) \big \} _ { t \in \mathcal { T } _ { \mathrm { L } } \cup \mathcal { T } _ { \mathrm { U } } } \bigg ) .\tag{40}
$$

Under the same assumptions on $\{ Y _ { t } ( s ) \} _ { t \in { \mathcal { T } } _ { \mathrm { L } } }$ as in Appendix $_ { \mathrm { D . 4 , } }$ we have asymptotic normality of the form

$$
\sqrt { T _ { \mathrm { L } } + T _ { \mathrm { U } } } \cdot \frac { \hat { \theta } ^ { \mathrm { O R A } } ( s ) - \theta ( s ) } { \hat { \sigma } ^ { \mathrm { O R A } } ( s ) } \implies \mathcal { N } ( 0 , 1 ) .\tag{41}
$$

See Appendix E.1 for more details. This yields the asymptotically valid $( 1 - \alpha ) ^ { \prime }$ % confidence interval

$$
\hat { \theta } ^ { \mathrm { O R A } } ( s ) \pm z _ { 1 - \alpha / 2 } \frac { \hat { \sigma } ^ { \mathrm { O R A } } ( s ) } { \sqrt { T _ { \mathrm { L } } + T _ { \mathrm { U } } } } .\tag{42}
$$

## E Asymptotics

We do not currently have any complete theoretical results. We conjecture that the HAC confidence intervals in Appendices D.4 and D.5 and the PPITSAS confidence intervals in Appendix A are asymptotically valid. We discuss the assumptions we believe are sufficient for the asymptotic validity of these confidence intervals, and sketch out proofs of our conjectured asymptotic validity.

## E.1 Asymptotic validity of HAC confidence intervals

Let $\{ W _ { t } \} _ { t \in \mathbb { N } }$ be a stochastic process on a probability space $( \Omega , { \mathcal { F } } , \operatorname { P } )$ , and suppose that for some $\nu > 1$

(1) ${ \mathbb E } [ | W _ { 1 } | ^ { 4 \nu } ] < \infty ,$

(2) $\{ W _ { t } \} _ { t \in \mathbb { N } }$ is α-mixing with

$$
\sum _ { j = 1 } ^ { \infty } j ^ { 2 } \alpha ( j ) ^ { ( \nu - 1 ) / \nu } < \infty ,
$$

(3) $\{ W _ { t } \} _ { t \in \mathbb { N } }$ is stationary.

We discuss the limitations of these three assumptions in Appendix G.

Condition (1) is a basic moment assumption. Condition (2) is a mixing assumption; in particular, for $\mathcal { F } _ { t } ^ { t + j } : = \sigma ( W _ { t } , \dots , W _ { t + j } )$ , we say that $\{ W _ { t } \} _ { t \in \mathbb { N } }$ is α-mixing if the mixing coefficients

$$
\alpha ( j ) : = \operatorname* { s u p } _ { t \in \mathbb { N } } \ \operatorname* { s u p } _ { A \in { \mathcal { F } } _ { 1 } ^ { t } } \left| \mathbf { P } ( A \cap B ) - \mathbf { P } ( A ) \mathbf { P } ( B ) \right|\tag{43}
$$

satisfy $\alpha ( j ) \to 0$ as $j \to \infty$ [25]. Condition (3) is a stationarity assumption; a process is stationary if for every m $\geq 1$ , every $t _ { 1 } , \ldots , t _ { m } \in \mathbb { N }$ , and every $j \in \mathbb N$

$$
( W _ { t _ { 1 } } , \dots , W _ { t _ { m } } ) \stackrel { \mathrm { ~ d ~ } } { = } ( W _ { t _ { 1 } + j } , \dots , W _ { t _ { m } + j } ) .
$$

That is, finite-dimensional distributions are invariant under time shifts.

Note that stationarity with the fourth moment condition implies covariance-stationarity, so we have that there exist real-valued $\mu$ and $\{ \gamma _ { j } \} _ { j = 0 } ^ { \infty }$ such that for all times $t , t _ { 1 } , t _ { 2 } \in \mathbb { N }$

$$
\begin{array} { r } { \mathbb { E } [ W _ { t } ] = \mu \qquad \operatorname { C o v } [ W _ { t _ { 1 } } , W _ { t _ { 2 } } ] = \gamma _ { | t _ { 1 } - t _ { 2 } | } . } \end{array}\tag{44}
$$

By Lemma 1 of [26], the assumptions also imply that the covariances are absolutely summable:

$$
\sum _ { j = 0 } ^ { \infty } | \gamma _ { j } | < \infty .\tag{45}
$$

Hence, by Proposition 7.5 of [24], we have that

$$
\theta : = \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } W _ { t }\tag{46}
$$

exists and is well-defined.

Let $T \in \mathbb { N }$ . We show that for

$$
\bar { W } : = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } W _ { t } \qquad \mathrm { a n d } \qquad \hat { \sigma } ^ { 2 } : = \mathsf { H A C } \Big ( \big \{ W _ { t } \big \} _ { t = 1 } ^ { T } \Big ) ,\tag{47}
$$

the sample mean is asymptotically normal:

$$
{ \sqrt { T } } \cdot { \frac { { \bar { W } } - \theta } { \hat { \sigma } } } \implies { \mathcal { N } } ( 0 , 1 ) .\tag{48}
$$

The stationarity and the fourth moment condition imply that the process is fourth-order stationary. By Lemma 1 of [26], our assumptions imply that $\{ W _ { t } \} _ { t \in \mathbb { N } }$ has absolutely summable fourth-order cumulants. Furthermore, by Davydov’s covariance inequality [27, 28] with $p = q = 2 \nu$ , there exists some constant $C > 0$ such that

$$
\begin{array} { r l } & { | \gamma _ { j } | \leq C \| W _ { t } - \mu \| _ { 2 \nu } \| W _ { t + j } - \mu \| _ { 2 \nu } \alpha ( j ) ^ { ( \nu - 1 ) / \nu } } \\ & { \qquad = C \| W _ { t } - \mu \| _ { 2 \nu } ^ { 2 } \alpha ( j ) ^ { ( \nu - 1 ) / \nu } . } \end{array}
$$

$$
\big ( \mathrm { D a v y d o v } \big )\tag{49}
$$

$$
\left( { \mathrm { s t a t i o n a r i t y } } \right)\tag{50}
$$

Hence for $C ^ { \prime } = C \| W _ { t } - \mu \| _ { 2 \nu } ^ { 2 } < \infty .$

$$
\sum _ { j = 1 } ^ { \infty } j ^ { 2 } | \gamma _ { j } | \leq \sum _ { j = 1 } ^ { \infty } j ^ { 2 } C \| W _ { t } - \mu \| _ { 2 \nu } ^ { 2 } \alpha ( j ) ^ { ( \nu - 1 ) / \nu }\tag{51}
$$

$$
= C ^ { \prime } \sum _ { j = 1 } ^ { \infty } j ^ { 2 } \alpha ( j ) ^ { ( \nu - 1 ) / \nu }\tag{52}
$$

$$
< \infty .\tag{53}
$$

So our process has p-summable autocovariances for $p = 2$ . This satisfies the conditions for the consistency of the Newey–West variance estimate $\hat { \sigma } ^ { 2 } [ \bar { 1 } 7 ] ;$ if

$$
s ^ { ( 1 ) } : = 2 \sum _ { j = 1 } ^ { \infty } j \gamma _ { j }\tag{54}
$$

is nonzero, then by Theorem 2 of [17],

$$
\hat { \sigma } ^ { 2 } \implies \Omega : = \gamma _ { 0 } + 2 \sum _ { j = 1 } ^ { \infty } \gamma _ { j } .\tag{55}
$$

Finally, by Theorem 1.7 of [29], we have that

$$
{ \sqrt { T } } ( { \bar { W } } - \theta ) \implies { \mathcal { N } } ( 0 , \Omega ) .\tag{56}
$$

Hence by Slutsky’s lemma, we have that

$$
\sqrt { T } \cdot \frac { \bar { W } - \theta } { { \hat { \sigma } } } \Longrightarrow { \cal N } ( 0 , 1 ) ,\tag{57}
$$

and hence the confidence interval

$$
\bar { W } \pm z _ { 1 - \alpha / 2 } \frac { \hat { \sigma } } { \sqrt { T } }\tag{58}
$$

is an asymptotically valid $( 1 - \alpha ) \%$ confidence interval.

## E.2 Argument for asymptotic validity of PPITSAS confidence intervals

We conjecture that with the following qualitative conditions, the confidence intervals we propose in Eq. (6) are asymptotically valid as $\bar { T _ { \mathrm { U } } } , \bar { T } _ { \mathrm { L } }  \infty \colon$

• Strict-sense stationarity of the covariate and response processes.

• Existence of $2 + \delta$ moments for the response and predictor.

• Sufficiently strong mixing of the covariate and response process.

We sketch an incomplete argument for asymptotic validity. First, for a sufficiently strong mixing condition, we should be able to treat a predictor trained on $\mathcal { T } _ { \mathrm { f i t } }$ as fixed without affecting the rest of the inference process. It suffices that the data predictor is effectively independent from the portions of the processes used to calculate $\hat { \Delta } ( s )$ and ${ \hat { \mu } } ( s )$ . This effective independence of the predictor should be achievable with a suitably strong form of mixing, together with a demonstration that the proportion of data that is correlated with the training data in the rectifying term is asymptotically negligible, or possibly a buffer between the fitting data and the inference data.

Second, we argue that stationarity and a sufficiently strong mixing condition on the covariate and response process $\{ ( X _ { t } ( s ) , Y _ { t } ( s ) \} _ { t }$ imply similar conditions on the residual process $\{ \hat { Y } _ { t } ( s ) - Y _ { t } ( s ) \} _ { } { } _ { }$ t and the prediction process $\{ \hat { Y } _ { t } ( s ) \} _ { t }$

Third, the mixing, stationarity, and moment conditions imply a joint central limit theorem for ${ \hat { \mu } } ( s )$ and $\hat { \Delta } ( s )$ by [29, Theorem 1.7]. This joint central limit theorem implies asymptotic normality of the PPI estimator, as a linear combination of jointly normal variables is normal.

Fourth, the mixing condition should imply that as $T _ { \mathrm { L } } , T _ { \mathrm { U } }  \infty$ , the sample means $\hat { \Delta } ( s )$ and ${ \hat { \mu } } ( s )$ become uncorrelated [27, 28]. Therefore, by asymptotic normality, the two terms are asymptotically independent and hence the variance of their sum is the sum of their variances.

Fifth, [26] demonstrates that stationarity, moment, and mixing conditions yield the Newey–West assumptions for consistent variance estimates when applying the Newey–West procedure [14] to $\{ \hat { Y } _ { t } ( s ) \bar  \} _ { t \in \mathcal { T } _ { \mathrm { r e c } } }$ and $\{ \hat { Y } _ { t } ( s ) - Y _ { t } ( s ) \} _ { t \in \mathcal { T } _ { \mathrm { U } } }$ , respectively.

Finally, combining the asymptotically consistent estimates of the variances with the joint central limit theorem and Slutsky’s lemma yields the asymptotic validity of the confidence interval.

The most subtle part of this argument is the first step, wherein we need to argue that under a sufficient mixing condition, the predictor trained using the same time series data used for inference can be treated as fixed for the remainder of the analysis. Intuitively, this should be true in the asymptotic setting of very long time series, since data at distant times should be effectively independent; therefore the model training cannot affect the estimate of the prediction bias or the estimate of the mean prediction. However, we leave this as a conjecture, and plan to formalize sufficient conditions for this in future work.

## F Experiments

## F.1 Simulation setup

In both settings, we take $\mathcal { S } = \{ 0 , 0 . 1 , . . . , 0 . 9 , 1 \} ^ { 2 }$ with $N s = 1 2 1$ . We take $T _ { \mathrm { L } } = 4 { , } 0 0 0$ and $T _ { \mathrm { U } } = 1 6 { , } 0 0 0$ , corresponding to roughly 10 and 40 years, respectively. From $\mathcal { T } _ { \mathrm { L } }$ , we take $T _ { \mathrm { { f i t } } } = 5 0$ points for fitting a linear predictor and the remaining $T _ { \mathrm { r e c } } = 3 9 5 0$ points for rectifying it.

Less extreme weather setting. We model the weather as a linear function added to the Gaussian process and the payout as a noisy measurement of the square of the weather.

Let $G _ { t } ( s ) \sim \mathcal { G P } ( 0 , k _ { \mathrm { S } } \times k _ { \mathrm { T } } )$ be a zero-mean Gaussian process with a covariance kernel which separates into spatial and temporal kernels. We take $k _ { \mathrm { { S } } }$ and $k _ { \mathrm { T } }$ to be Gaussian kernels with length scales $\ell _ { \mathrm { S } } = 0 . 1$ and $\ell _ { \mathrm { T } } = 1 0$ . The weather and payout are then given by

$$
X _ { t } ( s ) = s ^ { \mathrm { T } } \mathbf { 1 } + 2 + G _ { t } ( s )\tag{59}
$$

$$
Y _ { t } ( s ) = \mathrm { c l i p } _ { [ 0 , 1 ] } \Big ( ( X _ { t } ( s ) ^ { 2 } + \varepsilon ) / 5 0 0 \Big ) ,\tag{60}
$$

where $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ is independent and identically distributed Gaussian noise. We take $\sigma = 0 . 5$ In this case, the weather process $X _ { t } ( s )$ has temporal correlations on the order of 1-2 weeks, but is not heavy-tailed. The long-run mean of the payout process $Y _ { t } ( s )$ varies across space, but hovers around 0.01.

More extreme weather setting. We model the weather as a linear function added to the exponential of Gaussian process and the payout as a noisy measurement of the square of the weather.

As in the less extreme weather setting, let $G _ { t } ( s ) \sim \mathcal { G P } ( 0 , k _ { \mathrm { S } } \times k _ { \mathrm { T } } )$ be a zero-mean Gaussian process with a separable covariance kernel governed by length scales $\ell _ { \mathrm { S } } = 0 . .$ 1 and $\ell _ { \mathrm { T } } = 1 0$ . The weather and payout are given by

$$
X _ { t } ( s ) = s ^ { \mathrm { T } } \mathbf { 1 } + 1 0 + \exp ( G _ { t } ( s ) )\tag{61}
$$

$$
\begin{array} { r } { Y _ { t } ( s ) = \mathrm { c l i p } _ { [ 0 , 1 ] } \Big ( ( X _ { t } ( s ) ^ { 2 } + \varepsilon ) / 1 2 0 0 \Big ) , } \end{array}\tag{62}
$$

where $\varepsilon \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I )$ is independent and identically distributed Gaussian noise. We take $\sigma = 1$

In this case, the weather process $X _ { t } ( s )$ has temporal correlations on the order of 1-2 weeks and is heavy-tailed. Just like in the less extreme weather setting, the payout process $Y _ { t } ( s )$ has a mean which varies across space, but is generally around 0.01.

## F.2 Reporting

In each of the less extreme and more extreme weather settings, we run all of the alternatives (classic, IWI, chunkPPI, HAC, and oracle) and our method PPITSAS at a significance level of $\alpha = 0 . 0 5$

Performance metrics. In this section, we discuss how the performance metrics are computed. Let $\widehat { \theta } ( s ; \mathcal { D } )$ and $\hat { I } ( s ; \mathcal { D } )$ be the point estimate and confidence interval, respectively, corresponding to one of the methods we consider, which takes as input a location $s \in S$ and a dataset D.

We sample 100 datasets $\mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { 1 0 0 }$ . For the scaled RMSE, we compute the normalized squared errors at each of the $N s = 1 2 1$ locations. We report

$$
\sqrt { \frac { 1 } { 1 2 1 \cdot 1 0 0 } \sum _ { s \in \mathcal { S } } \sum _ { i = 1 } ^ { 1 0 0 } \left( \frac { \hat { \theta } ( s ; \mathcal { D } _ { i } ) - \theta ( s ) } { \theta ( s ) } \right) ^ { 2 } } .\tag{63}
$$

We compute the estimated coverage of each method. We estimate the coverage by averaging over the 100 sampled datasets and the 121 locations. We report

$$
\frac { 1 } { 1 2 1 \cdot 1 0 0 } \sum _ { s \in \cal { S } } \sum _ { i = 1 } ^ { 1 0 0 } { \bf 1 } \big \{ \theta ( s ) \in \hat { I } \big ( s ; \mathcal { D } _ { i } \big ) \big \} .\tag{64}
$$

We compute the scaled half-width as well. We report

$$
\frac { 1 } { 1 2 1 \cdot 1 0 0 } \sum _ { s \in \cal { S } } \sum _ { i = 1 } ^ { 1 0 0 } \frac { 1 } { 2 } | \hat { I } ( s ; { \cal { D } } _ { i } ) | .\tag{65}
$$

The resulting statistics are reported in Table 2.

## F.3 Code and reproducibility

The code to generate the simulated data for each of the two weather settings, to run each of the methods on the simulated data, and to compute the performance metrics is available at https: //github.com/shahzarrizvi/ppitsas.

These experiments were run locally on a MacBook Pro with an Apple M3 Max chip, consisting of a 14-core CPU (10 performance and 4 efficiency cores), and 36 GB of unified memory. The experiments were CPU-based and did not use GPU acceleration. Generating the simulated data and running the methods on the simulated data for both of the two weather settings took a total of approximately 1 hour of computation.

## G Limitations

As discussed in Section 2 and Appendix E, our method assumes the relevant processes are stationary in time. This assumption is strictly weaker and more realistic than the i.i.d. assumption made by standard PPI. In our framework, the stationarity assumption allows for the estimand of the long-run mean response to exist. Stationarity is also what makes the long history of unlabeled data relevant; if the distributions of the covariate and response processes are shifting dramatically in time, we have less reason to believe that leveraging long histories of data will improve our estimation.

However, the assumption can be unrealistic in applications where the data-generating process is changing over time. In our insurance example, changes in the climate or claims practices can result in a mean response which changes over time. We consider developing simulation-based procedures which maintain validity under weaker forms of temporal stability (like local stationarity or smoothly varying distributions) an important direction for future work.

We also make other regularity assumptions about the covariate and response processes in Appendix E. By assuming the processes are strongly mixing and satisfy a moment condition, we are able to get a central limit theorem on the sample mean. Without mixing, it is possible for the temporal dependence to persist over arbitrarily long periods of time, making it difficult to get the independence needed for a central limit theorem. The moment condition ensures that sample means, sample variances, and sample autocovariances are sufficiently well-behaved. While such moment conditions are standard, they exclude heavy-tailed processes. In our insurance example, the response process is bounded, and so the moment condition is not a concern.

Another limitation of our present work is in our experiments. We have conducted two experiments comparing the performance of PPITSAS against alternatives. In both experiments, we use a linear model to simulate responses from covariates when in actuality they have a quadratic relationship. We intend to consider more realistic simulators (such as a random forest) and more complicated relationships between the covariates and the responses in future work.

## H Licenses

In this appendix, we list the libraries and packages used to run our experiments (including versions), and the corresponding licenses.

• numpy==2.0.2. BSD 3-Clause license.

• scipy==1.13.1. BSD 3-Clause license.

• scikit-learn==1.6.1. BSD 3-Clause license.