# TabCUQ: Context-Uncertainty Quantification for Tabular Foundation Models<sup>∗</sup>

Yuebei Shi<sup>1,2</sup>, Molei Liu<sup>3,4</sup> and Zijian Guo<sup>2</sup>

<sup>1</sup>School of Mathematics, Zhejiang University

<sup>2</sup>Center for Data Science, Zhejiang University

<sup>3</sup>Department of Biostatistics, Peking University Health Science Center

<sup>4</sup>Beijing International Center for Mathematical Research, Peking University

## Abstract

Tabular foundation models use in-context learning to estimate conditional expected outcomes and average treatment efects for specific groups of individuals. Using these estimates to guide scientific decisions requires reliable uncertainty quantification. The challenge is to account simultaneously for two sources of uncertainty: limited-context bias and context variation arising from the small labeled sample supplied at inference. Predictive spread does not directly quantify both sources of uncertainty, and we observe that the ordinary bootstrap may substantially overestimate context variation, whereas subsampling may underestimate it. We introduce TabCUQ, a perturbation-based framework that injects noise into observed outcomes and applies the foundation model to the perturbed contexts. TabCUQ estimates context variation using wild-bootstrap dispersion and accounts for limited-context bias using the maximum deviation between perturbed and original estimates. For a kernel-regression analogue whose estimation error admits an exact decomposition into these two components, we show that our framework provides asymptotically valid pointwise coverage. In simulations, TabCUQ substantially improves coverage, with average interval widths typically about twice those of an oracle bias-aware benchmark. Across seven real-world tasks, it also improves coverage of reference functions learned from much larger samples.

## 1 Introduction

Scientific decisions often depend on estimates drawn from small tabular datasets. Tabular foundation models ofer a promising approach to prediction in these settings. Conventional methods, such as generalised additive models, random forests and neural networks trained from scratch, are typically fitted separately to each dataset [Hastie et al., 2009]. The foundation models studied here are neura networks trained in advance on many synthetic and real datasets. This training allows the same model to be reused for prediction tasks on new datasets. For a new task, the user supplies a small labeled dataset containing examples of features and their observed outcomes. The model takes these examples as part of its input and uses them to predict the outcome of a new observation from its features, without updating its pretrained parameters. This process is known as in-context learning. Built on this idea, tabular foundation models, including TabPFN [Hollmann et al., 2025], TabICL [Qu et al., 2025] and TabFM [Kong and Das, 2026], have demonstrated strong predictive performance on small-data benchmarks. Their predictive success raises a question: how reliably can estimates obtained from a small labeled dataset support a scientific conclusion?

We focus on two quantities that inform scientific decisions: the conditional mean and the conditional average treatment efect. A chemist may need to determine whether the mean reaction yield under specified experimental conditions exceeds a threshold for further development. Time and material costs limit the number of experiments available to estimate this quantity [Shields et al., 2021, Hickman et al., 2022]. The average outcome across repeated experiments under those conditions is a conditional mean. A clinical researcher may instead ask whether a treatment shortens average recovery time for patients with a particular clinical profile. Their average recovery time under treatment minus their average recovery time without treatment is the conditional average treatment efect (CATE); a negative value indicates faster recovery.

We use conditional-mean estimation to explain how a foundation model produces an estimate. As illustrated in Figure 1, the foundation model takes two inputs: context and query. The context is the labeled dataset containing observed features and outcomes, and the query specifies the feature values at which the mean is to be estimated. In the chemistry example, the context contains completed experiments and their measured yields, while the query specifies the reaction conditions of interest. The model processes the context and query without updating its pretrained weights, a procedure called in-context learning. We use its point prediction as an estimator of the conditional mean. Although the prediction rule is fixed, the estimate depends on the labeled observations supplied as context.

![](images/b44f0ff0c958b1e7afe050e8dd97a90234cd44ea773b0dbdb90d40c3cd69c26d.jpg)  
Figure 1: In-context prediction with a tabular foundation model (TFM) pretrained on many synthetic datasets. The context contains observed feature–outcome pairs, while the query supplies feature values with its outcome Y left unknown. With its pretrained weights fixed, the model processes both inputs to produce a predictive distribution for Y .

We distinguish two components of context uncertainty: context variation and limited-context bias. A fresh context of the same size sampled from the target population can produce a diferent estimate. Each context consists of newly sampled observations, so both features and outcomes vary. We refer to this sampling variation as context variation. To understand the second component, imagine repeatedly drawing contexts of the same size and averaging the resulting estimates at a fixed query. Even after the sampling fluctuations are averaged out, the average estimate may still lie above or below the true conditional mean at that query. We refer to this systematic gap as limited-context bias. In our experiments, TabPFN frequently exhibits such bias with small contexts, analogous to the regularization bias commonly encountered in classic machine learning methods. In statistical terminology, these are the sampling variability and bias of the estimator induced by the pretrained prediction rule. The same distinction applies to a CATE estimator at a fixed clinical profile. Panel A of figure 2 illustrates both components. Both components matter when scientific decisions depend on an estimated mean or treatment efect. Sampling fluctuations or bias can make an estimate appear favorable even when the true mean yield is inadequate or the treatment benefit is negligible. A point estimate alone therefore does not establish whether the data support the scientific conclusion. We seek confidence intervals for the conditional mean or CATE. A valid confidence-interval procedure covers the fixed target with probability at least the stated confidence level over repeated contexts of the same size. Such intervals help assess whether the data support a mean yield above the development threshold or a treatment efect below zero, taking estimation uncertainty into account.

Constructing such confidence intervals requires estimating context variation and accounting for limited-context bias. Bootstrap and subsampling provide standard tools for approximating sampling distributions under appropriate conditions and are natural candidates for estimating context variation. Applying classical resampling methods to pretrained tabular foundation models raises the question of whether they accurately capture variation across the labeled contexts supplied at inference, while the pretrained network parameters remain fixed. Nevertheless, accurately estimating context variation is only part of the inferential task. In nonparametric regression, estimation bias that is non-negligible relative to the estimator’s standard deviation can impair confidence-interval coverage, even when that standard deviation is accurately estimated [Wasserman, 2006, Section 5.7]. Pretraining does not itself guarantee negligible bias at the available context size. Through extensive simulation and real-world studies, we observe that the Tabular foundation models tend to sufer from an estimation bias with the limited aomunt of context. We therefore develop TabCUQ, a perturbation-based confidence-interval construction that accounts for both context variation and limited-context bias.

## 1.1 Main results and contributions

The paper makes two main contributions. First, we find empirically that wild-bootstrap perturbations efectively estimate the scale of context variation for the pretrained tabular prediction rules evaluated here. Interestingly, we observe that the ordinary bootstrap produces standard-error estimates substantially larger than the true standard deviations, whereas the evaluated subsampling procedure produces smaller estimates. These findings motivate our use of wild-bootstrap perturbations in the uncertainty quantification of the tabular foundation model. Second, we introduce TabCUQ, a perturbation-based confidence-interval construction for conditional means and CATEs that combines an estimate of context variation with a characterization of limited-context bias. In a kernel-regression setting, we prove that the proposed bias cushion bounds estimation bias and establish coverage for the resulting interval under the stated assumptions.

TabCUQ generates perturbed contexts by applying wild-bootstrap perturbations to the observed outcomes, while holding the features and context size fixed. We then apply the tabular foundation model to each perturbed context to obtain the corresponding perturbed estimates. We use the empirical standard deviation of these estimates to estimate the scale of context variation. To characterize limited-context bias, we define a bias cushion as the largest absolute displacement of the perturbed estimates from the original estimate. We construct a normal-approximation confidence interval centered on the original point estimate and widen it by this cushion on each side. The construction thus uses the same perturbed estimates both to quantify sampling variability and to provide an additional margin for bias.

A Variation and bias due to limited context  
![](images/6c3c38503254a1f3d4ab487dc0a04e6896577f3f3c46eeea56c13cb5d1e296ee.jpg)

B TabCUQ quantifies context uncertainty  
![](images/9711c9899bd09d23df0b014657cb458a4d6bad35ae78fbb75a947985a4b4ab6d.jpg)  
Figure 2: TabCUQ quantifies context uncertainty. (A) With the pretrained TFM and query fixed, independently sampled contexts yield diferent estimates. Their spread represents context variation, while the gap between their across-context mean and truth represents limited-context bias. (B) Wild-bootstrap outcome perturbations provide an estimate of context variation and an empirical bias cushion. The blue central segment represents the contribution of estimated variation, while the orange extensions represent the empirical bias cushion.

We evaluate TabCUQ in extensive simulated tasks of conditional-mean and CATE inference with three tabular foundation models. Even when wild-bootstrap standard errors are reasonably well calibrated, confidence intervals that ignore limited-context bias can substantially undercover. TabCUQ improves coverage across the main simulation settings, generally reaching the nominal 95% level, with average interval widths typically around twice those of an oracle benchmark that uses the true bias and variance. We further evaluate TabCUQ on seven real-world tasks spanning socioeconomic, environmental, behavioural, protein and single-cell data. Since the true conditional means are unknown, we assess coverage relative to reference functions fitted on much larger samples. With contexts of small sizes, the average reference coverage across tasks increases from 0.614–0.842 for wild bootstrap intervals to 0.966–1.000 for TabCUQ, while average interval widths of TabCUQ remain around two times those of the oracle benchmark.

## 1.2 Relation to existing work

TabCUQ builds on classical resampling methods for inference about fixed population quantities. The nonparametric bootstrap and subsampling approximate sampling distributions by recomputing estimators on resampled observations [Efron, 1979, Politis et al., 1999]. Classical wild-bootstrap methods perturb regression residuals with covariates held fixed to accommodate heteroskedastic errors [Wu, 1986]. TabCUQ applies outcome perturbations to pretrained in-context prediction rules, using the resulting estimates to quantify context variation and construct a margin for limited-context bias. Its use of a bias cushion also connects the construction to bias-aware methods that calibrate confidence intervals to accommodate estimation bias [Armstrong and Kolesár, 2016].

For pretrained models, posterior methods ofer another approach to uncertainty quantification. PFNs were introduced to approximate Bayesian posterior predictive distributions under a specified prior over tasks [Müller et al., 2022]. Predictive intervals from these distributions concern future responses and include outcome noise. Posterior methods can also target uncertainty about conditional means and other functionals. The approximate martingale posterior method of Nagler and Rügamer [2025] uses PFN-initialised predictive distributions and copula updates to quantify uncertainty about conditional means, quantiles and other summaries at a fixed query. The authors motivate learning-rate choices that widen credible intervals to accommodate nonparametric bias. TabMGP [Ng et al., 2026] constructs martingale posteriors for parameters defined as minimisers of an expected loss. Fortini et al. [2026] use a predictive central limit theorem to obtain Gaussian approximations to posterior uncertainty about conditional event probabilities under a limiting predictive distribution. These approaches provide posterior uncertainty for a range of targets using pretrained models. In contrast, our proposal assesses the uncertainty from the frequentist perspective, which has distinct properties and diferent application scenarios from posterior credibility. Our method constructs intervals by perturbing the observed outcomes and repeatedly evaluating the original point estimator. We assess their coverage at a fixed query over repeated draws of the labeled context.

Conformal prediction constructs prediction sets for future responses [Shafer and Vovk, 2008], whereas TabCUQ constructs confidence intervals for conditional means and CATEs. Under exchangeability, standard split conformal prediction guarantees marginal coverage [Lei et al., 2018]; coverage at a particular query can still fall below the nominal level [Barber et al., 2021]. For tabular foundation models, C-USIM [Park et al., 2026] uses held-out observations to calibrate highest predictive density regions and studies the gap between conditional and marginal coverage. In causal inference, Lei and Candès [2021] develop conformal methods for counterfactual outcomes and individual treatment efects. An individual treatment efect is the diference between a person’s two potential outcomes; CATE is the conditional expectation of this diference given the features. These are diferent inferential targets: even valid coverage for outcomes or individual efects at a fixed query does not by itself establish confidence-interval coverage for their conditional means.

## 2 Uncertainty Quantification for Tabular Foundation Model

Uncertainty quantification is essential when predictions from tabular foundation models inform decisions based on small- to medium-sized datasets. A positive estimate of a conditional mean or treatment efect does not, by itself, establish that the underlying quantity is positive. The estimated sign may reflect limited-context bias or context variation. We first review tabular foundation models in Section 2.1 and then discuss the need for uncertainty quantification and the challenges involved in the remainder of this section.

## 2.1 Basics of Tabular Foundation Models

We review prior-data fitted networks (PFNs), which provide the methodological foundation for TabPFN [Müller et al., 2022]. PFNs are pretrained on synthetic datasets and subsequently applied to new datasets at prediction time. Let $\phi$ index a data-generating rule, with prior density $\pi ( \phi )$ To construct each synthetic dataset, the simulator first samples $\phi \sim \pi$ and then generates labeled observations independently from the sampled rule, conditional on $\phi .$ . The observations are randomly divided into a labeled context and held-out queries. For each query, the network receives the context together with the query features and predicts the corresponding outcome label. The query labels are withheld from the network inputs and used only to compute the training loss.

Let D denote the labeled context and $( X _ { \mathrm { t e s t } } , Y _ { \mathrm { t e s t } } )$ a held-out query, where $X _ { \mathrm { t e s t } }$ and $Y _ { \mathrm { t e s t } }$ are its features and outcome label, respectively. A PFN with parameters $\theta$ outputs a predictive density $q _ { \theta } ( y \mid x , D )$ for the outcome label of a query with features $x ,$ given context $D .$ . The population pretraining objective is

$$
\mathscr { L } ( \theta ) = \mathbb { E } _ { \phi \sim \pi } \mathbb { E } _ { D , ( X _ { \mathrm { t e s t } } , Y _ { \mathrm { t e s t } } ) \mid \phi } \left[ - \log q _ { \theta } ( Y _ { \mathrm { t e s t } } \mid X _ { \mathrm { t e s t } } , D ) \right] .\tag{1}
$$

In practice, we sample synthetic datasets using the procedure described above and approximate the expectation in (1) by an empirical average of the losses on the sampled synthetic datasets. This empirical loss is used to update the network parameters $\theta ,$ with new synthetic datasets sampled throughout training.

We follow Müller et al. [2022] and connect the pretraining objective in (1) to the Bayesian posterior predictive density. Let $p _ { \pi } ( y \mid x , D )$ denote the Bayesian posterior predictive density of the query outcome with the prior $\pi .$ , admitting the following expression

$$
p _ { \pi } ( y \mid x , D ) = \int p ( y \mid x , \phi ) p _ { \pi } ( \phi \mid x , D ) d \phi .\tag{2}
$$

Here, $p ( \boldsymbol { y } \mid \boldsymbol { x } , \phi )$ is the conditional response density under rule $\phi ,$ and $p _ { \pi } ( \phi \mid x , D )$ is the posterior density of the rule given the context and query features. The loss in (1) decomposes as

$$
\mathcal { L } ( \boldsymbol { \theta } ) = C + \mathbb { E } _ { D , X _ { \mathrm { t e s t } } } \left[ \mathrm { K L } ( p _ { \pi } ( \cdot \mid X _ { \mathrm { t e s t } } , D ) \parallel q _ { \theta } ( \cdot \mid X _ { \mathrm { t e s t } } , D ) ) \right] ,\tag{3}
$$

where $C = \mathbb { E } [ - \log p ( Y _ { \mathrm { t e s t } } \mid X _ { \mathrm { t e s t } } , D ) ]$ is independent of $\theta .$ . Minimizing the population loss is therefore equivalent to minimizing the expected KL divergence from $p _ { \pi }$ . Over a rich class of $q _ { \theta }$ functions, this divergence is minimized with $q _ { \theta } ( \cdot \mid x , D ) = p _ { \pi } ( \cdot \mid x , D )$ for almost every $( D , x )$ . Thus, $p _ { \pi }$ is the population predictive target, although a fitted PFN may not achieve it due to restrictions on the network class, finite pretraining data, and imperfect optimization.

Conventional Bayesian methods typically obtain the predictive density by approximating this posterior distribution $p _ { \pi } ( \phi \mid x , D )$ for each context and then applying (2) to learn the posterior outcome distribution. In contrast, PFNs are based on directly using $q _ { \theta } ( y \mid x , D )$ to approximate the posterior predictive density $p _ { \pi } ( y \mid x , D )$ through pretraining on simulated datasets, without explicitly constructing the posterior over $\phi$ or evaluating the integral in (2) at prediction time.

## 2.2 Inferential Targets and Real-data Examples

We consider two inferential targets that inform the scientific decisions. The first target is the conditional mean in regression, which describes the average outcome for given covariates. Let $D _ { n } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ be a context consisting of i.i.d. observations, and fix a query feature vector x. Our target is the population conditional mean $\mu ( x ) = \mathbb { E } [ Y \mid X = x ]$ . The tabular foundation model provides a predictive densit $q _ { \theta } ( y \mid x , D _ { n } )$ and the corresponding conditional mean estimator

$$
{ \widehat { \mu } } ( x ) = \int y q _ { \theta } ( y \mid x , D _ { n } ) d y .\tag{4}
$$

The second target is the conditional average treatment efect (CATE), defined as the diference between the conditional means of the potential outcomes under treatment and control [Athey and Imbens, 2016]. Let $A \in \{ 0 , 1 \}$ indicate treatment, X denote pretreatment covariates, and $Y ( a )$ denote the potential outcome under treatment assignment a. At a given covariate value x, we define the corresponding CATE as $\tau ( x ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x ]$ . To identify $\tau ( x )$ from observed data, we assume the standard causal identification condition: (i) consistency, $Y = Y ( A )$ ; (ii) overlap, $0 < P ( A = 1 \mid X = x ) < 1$ for all possible $x ;$ (iii)unconfoundedness, $\{ Y ( 0 ) , Y ( 1 ) \} \perp A \mid X$ [Hernán and Robins, 2020]. Writing $\mu _ { a } ( x ) = \mathbb { E } [ Y \mid A = a , X = x ]$ for $a \in \{ 0 , 1 \}$ , these assumptions give $\tau ( x ) = \mu _ { 1 } ( x ) - \mu _ { 0 } ( x )$ . This representation reduces CATE estimation to two regression problems. Given a context $D _ { n } ^ { \tau } = \{ ( X _ { i } , A _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ consisting of i.i.d. observations, we apply the estimator in (4) separately to the treated and control groups, obtaining ${ \widehat { \mu } } _ { 1 } ( x )$ and ${ \widehat { \mu } } _ { 0 } ( x )$ , respectively. The T-learner [Künzel et al., 2019] then takes their diference:

$$
\widehat { \tau } ( x ) = \widehat { \mu } _ { 1 } ( x ) - \widehat { \mu } _ { 0 } ( x ) .\tag{5}
$$

Although tabular foundation model achieves strong prediction performance, the point estimates produced by the tabular foundation model alone are insuficient to guide scientific decisions. We illustrate this in Figure 3 with two real-data examples. Our first example concerns mutation-induced changes in protein stability. The feature vector X is the diference between the embeddings of the wild-type and mutant protein sequences and the response is the diference in Gibbs free energy of unfolding between the mutant and wild-type proteins, measured in kcal/mol. A positive value indicates that the mutant protein is more stable than the wild type, whereas a negative value indicates that it is less stable. Here, $\mu ( x )$ represents the expected stability change conditional on the mutation’s feature representation. Using the protein stability dataset accompanying GeoStab [Xu et al., 2024], we consider the N139V mutation in protein 1G3P.

We randomly draw 20 contexts, each containing 100 labeled observations sampled without replacement from a total of 8,754 mutations. For each context, we obtain a TabPFN 2.6 estimate at the same query. A model fitted to the much larger full dataset provides an empirical reference estimate of approximately −0.091 kcal/mol. As shown in Figure $\mathrm { 3 ( a ) }$ , six of the 20 TabPFN estimates are positive, while the remaining 14 are negative. Thus, a decision based solely on the estimated direction of the stability change may depend on the particular observations included in the context. This sensitivity to context sampling also matters when estimates are used to assess the efects of an intervention.

Our second example concerns a job-training programme and uses a pretreatment outcome to provide a benchmark with a known null efect. We use the National Supported Work experiment studied by LaLonde [1986], with the sample compiled by Dehejia and Wahba [1999]. Treatment is assignment to the training programme, and we take earnings in 1974, before treatment, as the outcome. Since later assignment cannot change income already earned, the true CATE is zero. Yet, for a fixed baseline covariate profile, the CATE estimates range from approximately −676 to 952

![](images/5a4e965fa229bc0290f3002de6a3ef23d1f7a1a430e95a4335b579da9cbcaf68.jpg)  
Figure 3: Context uncertainty in two real-world examples. (a) Estimated conditional-mean stability change for the N139V mutation in PDB 1G3P. (b) Estimated CATE of job-training assignment on pretreatment earnings in the National Supported Work experiment, where the true efect is zero. Pink and teal indicate positive and negative estimates the solid orange line in (a) indicates the pseudo-truth from a LightGBM reference model fitted to the full protein data observations, approximately −0.091 kcal/mol. The gray dashed line in (b) marks the known zero CATE.

US dollars across the 20 contexts shown in Figure 3(b). Interpreting their signs alone could suggest either a benefit or a harm where neither is possible.

Together, these examples show that point estimates can lead to diferent scientific conclusions as the context changes. Variation across contexts demonstrates context uncertainty, while the apparent displacement of estimates from their reference values raises the possibility of limited-context bias. These observations motivate uncertainty quantification that accounts for both sampling variability and potential bias before the estimated magnitude or sign is used to guide a scientific decision.

## 2.3 Challenges of Uncertainty Quantification for Tabular Foundation Models

To support scientific conclusions about conditional means and CATEs, we construct confidence intervals by bounding the estimation error. We focus on the conditional mean to illustrate the main idea and the same construction applies to CATE. A confidence interval provides a range of plausible values for the average outcome at x in regression. For a fixed query x and significance level $\alpha \in ( 0 , 1 )$ , we seek a data-dependent bound $R _ { \alpha } ( D _ { n } ; x )$ satisfying

$$
\begin{array} { r } { \mathbb { P } \{ | \widehat { \mu } ( x ) - \mu ( x ) | \le R _ { \alpha } ( D _ { n } ; x ) \} \ge 1 - \alpha . } \end{array}
$$

Using this bound as the radius gives the 1 − α confidence interval

$$
C _ { \alpha } ( D _ { n } ; x ) = [ \widehat { \mu } ( x ) - R _ { \alpha } ( D _ { n } ; x ) , ~ \widehat { \mu } ( x ) + R _ { \alpha } ( D _ { n } ; x ) ] .
$$

Its width indicates estimation precision, and an interval entirely above or below a scientifically relevant benchmark supports the corresponding conclusion about $\mu ( x )$ . One of the most commonly used confidence interval is the Wald interval [Wasserman, 2004], which uses an estimated standard error multiplied by a normal critical value as its radius:

$$
C _ { \mathrm { W a l d } } ( D _ { n } ; x ) = \left[ \widehat { \mu } ( x ) - z _ { 1 - \alpha / 2 } \widehat { s } ( x ) , \widehat { \mu } ( x ) + z _ { 1 - \alpha / 2 } \widehat { s } ( x ) \right] ,
$$

where $z _ { 1 - \alpha / 2 }$ is the $( 1 - \alpha / 2 )$ quantile of the standard normal distribution. Here, ${ \widehat { s } } ( x )$ estimates the standard deviation of ${ \widehat { \mu } } ( x )$ across repeated contexts and thus quantifies context variation. Approximate nominal coverage requires an accurate standard error estimate, an appropriate normal approximation, and limited-context bias that is negligible relative to the standard error.

A  
![](images/9e174d688feb103e2d82e6b391b087a6bf6a438f47801fded303cbb91062585b.jpg)

B  
![](images/a1607f1e1f53cd17b8c54337ad48d388e933a238cf79c2968248b2b7597767a1.jpg)  
Figure 4: Context variation and limited-context bias of TabPFN 2.6 in the simulation setting with $Y = \sin ( \pi X ) + \varepsilon . $ , where independent $X , \varepsilon \sim N ( 0 , 1 )$ and the context size $n = 3 0 0 . \mathrm { ~ ( A ) }$ At $x = - 1 . 7 3 7$ , TabPFN estimates from 100 independent contexts (gray) and 100 bootstrap estimates conditional on one context (blue). (B) Ratios of mean resampling variance to the variance of TabPFN estimates across independent contexts, averaged over 20 queries in $[ - 3 , 3 ]$ . We use the same 100 contexts to estimate both variance, with 100 resamples per method and context. Subsampling draws $m = 3 0$ without replacement and scales variance by $m / ( n - m )$

Bootstrap and subsampling are often used to estimate the standard error. Ordinary bootstrap draws n observations with replacement from the observed context, whereas subsampling draws $m < n$ observations without replacement [Efron, 1979, Politis et al., 1999]. We supply each resampled dataset to the foundation model as a new context and use the variation across the resulting point estimates to estimate the standard error ${ \widehat { s } } ( x )$ in section 2.3. As an example, we consider TabPFN in the sine simulation setting (figure 4). Panel A illustrates both context variation and limited-context bias at the fixed query $x = - 1 . 7 3 7$ . The gray histogram shows TabPFN estimates across 100 independent contexts. Their empirical mean is 0.566, compared with the true conditional mean of 0.736, indicating a limit-context bias of 0.17. The blue histogram shows 100 ordinary-bootstrap estimates from a single context and has a wider spread than the gray histogram. Panel B shows that ordinary bootstrap overestimates variance on average across the query grid, whereas subsampling underestimates it. At each of 20 equally spaced queries in [−3, 3], we average the variance estimates from each method over the same 100 contexts and divide by the empirical variance of TabPFN estimates across those contexts. The two bars show these ratios averaged over queries: 1.64 for ordinary bootstrap and 0.45 for subsampling.

We next discuss possible reasons why ordinary bootstrap and subsampling misestimate context variation in our experiments. Ordinary bootstrap reuses the observed data, with some observations repeated and others omitted. TabPFN may respond diferently to these repetitions than to new observations, so the bootstrap variation may not match the variation we want to estimate. Subsampling uses smaller contexts. We rescale its standard-error estimates to account for the smaller size, but the adjusted estimates still underestimate context variation in our experiments. For

Wald intervals, overestimating the standard error produces wider intervals, whereas underestimating it can lead to insuficient coverage even when the point estimator’s bias is small.

Accurately estimating context variation does not, however, address limited-context bias. When this bias is large relative to the standard error, a Wald interval may fail to attain nominal coverage even if the standard error is accurately estimated. Since the population target is unknown, the bias cannot be determined directly from the observed labeled context. In the next section, we therefore seek an interval construction that accounts for both context variation and limited-context bias.

## 3 Method

TabCUQ constructs confidence intervals by combining an estimate of context variation with an empirical bias cushion for limited-context bias. We design the particular perturbation or resampling method in the following Section 3.1. We develop the confidence interval construction for a conditional mean in Section 3.2, then extend it to a conditional average treatment efect in Section 3.3.

## 3.1 Perturbation design

We first consider the conditional mean at a fixed query x, given the observed context $D _ { n } = ( X _ { i } , Y _ { i } ) _ { i = 1 } ^ { n }$ We use the wild bootstrap to perturb the outcomes while keeping all observed feature rows unchanged, where the outcome perturbation is formed by adding a randomly weighted residual to an estimated conditional mean. To avoid overfitting, we use five-fold cross-fitting to obtain the out-of-fold predictions used to construct both the residuals and the perturbed responses.

We partition the observation indices into five disjoint folds $I _ { 1 } , \ldots , I _ { 5 }$ of approximately equal size. For each $k = 1 , \ldots , 5$ , the other four folds form the context $D _ { - k } = \{ ( X _ { j } , Y _ { j } ) \} _ { j \notin I _ { k } }$ . Let ${ \widehat { \mu } } _ { - k } ( x )$ denote the estimator in (4) evaluated using $D _ { - k }$ as the context. For each observation $i \in I _ { k }$ , we use $X _ { i }$ as the query to obtain the out-of-fold prediction. Repeating this calculation for $k = 1 , \ldots , 5$ provides exactly one out-of-fold prediction for each observation, that is, we define

$$
{ \widehat { \mu } } ( X _ { i } ) = { \widehat { \mu } } _ { - k } ( X _ { i } ) , \quad { \mathrm { f o r } } \quad i \in I _ { k } \quad { \mathrm { w i t h } } \quad k = 1 , \ldots , 5 .\tag{6}
$$

We then compute the out-of-fold residuals and subtract their sample mean from each residual before using them to generate the response perturbations. For $i \in I _ { k } .$ , with $k = 1 , \ldots , 5$ , define the out-of-fold residual and the centered residual used below by

$$
r _ { i } = Y _ { i } - { \widehat { \mu } } ( X _ { i } ) \quad { \mathrm { a n d } } \quad e _ { i } = r _ { i } - { \overline { { r } } } , \quad { \mathrm { w i t h } } \quad { \overline { { r } } } = { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } r _ { j } .
$$

Then, we form each perturbed response by multiplying the centered residual $e _ { i }$ by a random sign and adding the result to the corresponding out-of-fold prediction $\widehat { \mu } ( X _ { i } )$ , for $i \in I _ { k }$ . The out-of-fold predictions and residuals are computed once and reused in all resamples. For each resample $b = 1 , \dots , B$ , draw an independent Rademacher multiplier $v _ { i , b }$ for every observation:

$$
\mathbb { P } ( v _ { i , b } = 1 ) = \mathbb { P } ( v _ { i , b } = - 1 ) = \frac { 1 } { 2 } , \qquad 1 \le i \le n , \quad 1 \le b \le B .
$$

Throughout the paper, we mainly use Rademacher multipliers. We may also adopt other multiplier distributions used in the wild-bootstrap literature, including the standard normal distribution and Mammen’s two-point distribution, both of which have mean zero and unit variance [Mammen, 1993, Kline and Santos, 2012]. For each resample $b = 1 , \dots , B$ , define the perturbed outcome as

$$
Y _ { i , b } = { \widehat { \mu } } ( X _ { i } ) + v _ { i , b } e _ { i } , \qquad i \in I _ { k } , \quad k = 1 , \ldots , 5 .\tag{7}
$$

The resulting context $D _ { b } = \{ ( X _ { i } , Y _ { i , b } ) \} _ { i = 1 } ^ { n }$ contains the original n feature vectors in the same row order, with only the outcome variables perturbed.

## 3.2 Standard error, empirical bias cushion, and confidence interval

In this section, we leverage the perturbed data set $D _ { b } = \{ ( X _ { i } , Y _ { i , b } ) \} _ { i = 1 } ^ { n }$ with $1 \leq b \leq B$ to quantify the context variation and limited-context bias of the tabular foundation model. Let ${ \widehat { \mu } } _ { b } ( x )$ denote the estimator in (4) evaluated at the fixed query x using $D _ { b } = \{ ( X _ { i } , Y _ { i , b } ) \} _ { i = 1 } ^ { n }$ as the context, where the variation across $\{ \widehat { \mu } _ { b } ( x ) \} _ { 1 \leq b \leq B }$ comes from the multiplier vectors.

In classical regression settings, wild-bootstrap methods can consistently approximate sampling distributions under suitable regularity conditions [Wu, 1986, Mammen, 1993]. Motivated by these results, we use the sample standard deviation of the PFN mean estimates obtained from the perturbed contexts as a bootstrap standard-error estimate, that is,

$$
\widehat { s } ( x ) = \left\{ \frac { 1 } { B - 1 } \sum _ { b = 1 } ^ { B } \bigl ( \widehat { \mu } _ { b } ( x ) - \overline { { \mu } } _ { B } ( x ) \bigr ) ^ { 2 } \right\} ^ { 1 / 2 } \quad \mathrm { w i t h } \quad \overline { { \mu } } _ { B } ( x ) = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \widehat { \mu } _ { b } ( x ) .\tag{8}
$$

The estimated scale ${ \widehat { s } } ( x )$ describes dispersion across perturbed contexts, but does not account for possible limited-context bias relative to $\mu ( x )$

To adjust for this limited-context bias, we introduce an empirical bias cushion constructed from the same perturbed estimates:

$$
\widehat { r } ( x ) = \operatorname* { m a x } _ { 1 \leq b \leq B } \left| \widehat { \mu } _ { b } ( x ) - \widehat { \mu } ( x ) \right| ,\tag{9}
$$

where ${ \widehat { r } } ( x )$ is the largest absolute diference between a perturbed estimate and the original estimate. We use this quantity as an empirical proxy of an upper bound for the limited-context bias. We provide theoretical support for this interpretation in a kernel-regression setting in section 4. Specifically, under the stated assumptions, ${ \widehat { r } } ( x )$ is a high probability upper bound for the bias of the kernel regression. Computed from the observed context and the generated outcome perturbations, this cushion provides an additional margin for possible systematic error. Adding it to the margin for context variation yields a confidence interval designed to accommodate both context variation and limited-context bias.

With the standard-error estimate and the empirical bias cushion defined, we now combine them to construct an interval for the conditional mean $\mu ( x )$ . For a significance leve $\alpha \in ( 0 , 1 )$ , define the TabCUQ interval by

$$
C _ { \alpha } ( D _ { n } ; x ) = \left[ \widehat { \mu } ( x ) - R _ { \alpha } ( D _ { n } ; x ) , \widehat { \mu } ( x ) + R _ { \alpha } ( D _ { n } ; x ) \right] ,\tag{10}
$$

with $R _ { \alpha } ( D _ { n } ; x ) = z _ { 1 - \alpha / 2 } \widehat { s } ( x ) + \widehat { r } ( x )$ . The term $z _ { 1 - \alpha / 2 } { \widehat s } ( x )$ addresses context uncertainty through a normal approximation based on the wild-bootstrap standard-error estimate ${ \widehat { s } } ( x )$ . The empirical bias cushion ${ \widehat { r } } ( x )$ further widens the interval on both sides to account for possible limited-context bias. Section 4 develops the corresponding coverage analysis under a tractable kernel regression regime. Algorithm 1 summarizes the procedure for constructing a conditional-mean interval.

## 3.3 Inference for conditional average treatment efects

In the following section, we generalize our proposal to uncertainty quantification for CATE. Recall the CATE estimation using the T-learner in Section 2 [Künzel et al., 2019], which estimates the conditional mean in each treatment arm and takes the diference between the two estimates. Let $D _ { n } ^ { \tau } = \{ ( X _ { i } , A _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ be a context consisting of i.i.d. observations, where $X _ { i }$ denotes pretreatment

Algorithm 1 TabCUQ for a conditional mean   
Input: Context $D _ { n } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } ~ ( n \geq 5 )$ , fixed query x, pretrained TFM with fixed weights $\theta ,$   
$B \geq 2$ resamples, and level $\alpha \in ( 0 , 1 )$   
Output: Original estimate ${ \widehat { \mu } } ( x )$ and interval $C _ { \alpha } ( D _ { n } ; x )$   
Step 1: Compute the original estimate ${ \widehat { \mu } } ( x )$ from $D _ { n }$ using $( 4 )$   
Step 2: Construct the out-of-fold predictions and centered residuals once:   
(2a) Partition $\{ 1 , \ldots , n \}$ into five nearly equally sized disjoint folds $I _ { 1 } , \ldots , I _ { 5 } .$   
(2b) For each $k = 1 , \ldots , 5 ,$ , use D $\mathbf { \Omega } _ { - k } = \{ ( X _ { j } , Y _ { j } ) : j \notin I _ { k } \}$ as the context to compute   
$\widehat { \mu } _ { - k } ( X _ { i } )$ and set ${ \widehat { \mu } } ( X _ { i } ) \longleftarrow { \widehat { \mu } } _ { - k } ( X _ { i } )$ for $i \in I _ { k }$   
(2c) Compute and center the residuals for $i = 1 , \ldots , n \colon$   
$r _ { i } \gets Y _ { i } - \widehat { \mu } ( X _ { i } ) , \qquad \overline { { r } } \gets \frac { 1 } { n } \sum _ { j = 1 } ^ { n } r _ { j } , \qquad e _ { i } \gets r _ { i } - \overline { { r } } .$   
Step 3: For each resample $b = 1 , \ldots , B \colon$   
(3a) Draw $v _ { i , b } \in \{ - 1 , + 1 \}$ with equal probabilities, independently across $i , b$ and of the   
data and fold partition.   
(3b) Set $Y _ { i , b } \gets \widehat \mu ( X _ { i } ) + v _ { i , b } e _ { i }$ for $i = 1 , \ldots , n .$   
(3c) Form $D _ { b } \gets \{ ( X _ { i } , Y _ { i , b } ) \} _ { i = 1 } ^ { n }$ in the original row order and compute ${ \widehat { \mu } } _ { b } ( x )$ from $D _ { b }$   
Step 4: Compute the standard-error estimate:   
$\overline { { \mu } } _ { B } ( x ) \gets \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \widehat { \mu } _ { b } ( x ) , \qquad \widehat { s } ( x ) \gets \left\{ \frac { 1 } { B - 1 } \sum _ { b = 1 } ^ { B } \bigl ( \widehat { \mu } _ { b } ( x ) - \overline { { \mu } } _ { B } ( x ) \bigr ) ^ { 2 } \right\} ^ { 1 / 2 } .$   
Step 5: Compute the empirical bias cushion $\begin{array} { r } { \widehat { r } ( x ) \gets \operatorname* { m a x } _ { 1 \leq b \leq B } | \widehat { \mu } _ { b } ( x ) - \widehat { \mu } ( x ) | , } \end{array}$   
Step 6: Form the interval and return ${ \widehat { \mu } } ( x )$ and $C _ { \alpha } ( D _ { n } ; x )$   
$R _ { \alpha } ( D _ { n } ; x ) \gets z _ { 1 - \alpha / 2 } \widehat { s } ( x ) + \widehat { r } ( x )$   
$C _ { \alpha } ( D _ { n } ; x ) \gets [ \widehat { \mu } ( x ) - R _ { \alpha } ( D _ { n } ; x ) , \widehat { \mu } ( x ) + R _ { \alpha } ( D _ { n } ; x ) ] .$   
covariates, $A _ { i } \in \{ 0 , 1 \}$ indicates treatment, and $Y _ { i }$ is the observed outcome. Under consistency,   
unconfoundedness, and overlap as stated in Section 2, the CATE is $\tau ( x ) = \mathbb { E } [ Y ( 1 ) - Y ( 0 ) \mid X = x ] =$   
$\mu _ { 1 } ( x ) - \mu _ { 0 } ( x )$ , where $\mu _ { a } ( x ) = \mathbb { E } [ Y \mid X = x , A = a ]$ denotes the conditional mean for the treatment   
assignment a. For each arm $a \in \{ 0 , 1 \}$ , we apply the pretrained model with $D _ { a } = \{ ( X _ { i } , Y _ { i } ) : A _ { i } = a \}$   
as the context to obtain the conditional-mean estimate ${ \widehat \mu } _ { a } ( x )$ in (4). The original T-learner estimate   
is $\widehat { \tau } ( x ) = \widehat { \mu } _ { 1 } ( x ) - \widehat { \mu } _ { 0 } ( x )$   
We apply Steps 2–3 of Algorithm 1 separately to $D _ { 0 }$ and $D _ { 1 }$ . Cross-fitting and residual centering   
are performed within each arm. Let ${ \widehat { \mu } } _ { a , b } ( x )$ denote the resulting prediction from arm a in resample   
b. The perturbed CATE estimate is   
$\widehat { \tau } _ { b } ( x ) = \widehat { \mu } _ { 1 , b } ( x ) - \widehat { \mu } _ { 0 , b } ( x ) , \qquad b = 1 , \ldots , B .$ (11)   
Then following (8) and (9), we compute $\widehat { s } _ { \tau } ( x )$ as the sample standard deviation of the perturbed

CATE estimates and $\widehat { r } _ { \tau } ( x )$ as their maximum absolute deviation from ${ \widehat { \tau } } ( x )$ . The CATE interval is

$$
C _ { \alpha , \tau } ( D _ { n } ^ { \tau } ; x ) = \left[ \widehat { \tau } ( x ) - \left\{ z _ { 1 - \alpha / 2 } \widehat { s } _ { \tau } ( x ) + \widehat { r } _ { \tau } ( x ) \right\} , \widehat { \tau } ( x ) + \left\{ z _ { 1 - \alpha / 2 } \widehat { s } _ { \tau } ( x ) + \widehat { r } _ { \tau } ( x ) \right\} \right] .\tag{12}
$$

Algorithm 2 summarizes the procedure for constructing a CATE interval.

Algorithm 2 TabCUQ for a CATE using a T-learner   
Input: Context $\overline { { D _ { n } ^ { \tau } = \{ ( X _ { i } , A _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } } }$ with at least five observations in each arm, fixed query x,   
pretrained TFM with fixed weights $\theta , B \geq 2$ resamples, and level $\alpha \in ( 0 , 1 )$   
Output: Original CATE estimate ${ \widehat { \tau } } ( x )$ and interval $C _ { \alpha , \tau } ( D _ { n } ^ { \tau } ; x )$   
Step 1: For each $a \in \{ 0 , 1 \}$ , form $D _ { a } = \{ ( X _ { i } , Y _ { i } ) : A _ { i } = a \}$ and apply Steps 1–3 of Algorithm 1 to   
obtain ${ \widehat { \mu } } _ { a } ( x )$ and $\{ \widehat { \mu } _ { a , b } ( x ) \} _ { b = 1 } ^ { B }$ , using independent multipliers across arms.   
Step 2: Compute the original and perturbed CATE estimates:   
$\widehat { \tau } ( x ) \longleftarrow \widehat { \mu } _ { 1 } ( x ) - \widehat { \mu } _ { 0 } ( x ) ,$   
$\widehat { \tau } _ { b } ( x ) \gets \widehat { \mu } _ { 1 , b } ( x ) - \widehat { \mu } _ { 0 , b } ( x ) , \qquad b = 1 , \ldots , B .$   
Step 3: Compute the standard-error estimate $\widehat { s } _ { \tau } ( x )$ and empirical bias cushion $\widehat { r } _ { \tau } ( x )$ from   
$\{ \widehat { \tau } _ { b } ( x ) \} _ { b = 1 } ^ { B }$ using (8) and (9), respectively, with ${ \widehat { \tau } } ( x )$ as the original estimate. Form   
the interval in (12) and return ${ \widehat { \tau } } ( x )$ and $C _ { \alpha , \tau } ( D _ { n } ^ { \tau } ; x )$

For average treatment efects, semiparametric methods based on influence functions allow valid inference with flexible estimates of nuisance functions under suitable conditions [Hines et al., 2022]. With continuously distributed covariates, however, a conditional mean or CATE at a fixed query is generally not pathwise diferentiable in an unrestricted nonparametric model. The usual regular root-n theory for average treatment efects therefore does not apply directly. Although orthogonality can reduce sensitivity to nuisance-estimation errors, it does not by itself eliminate the bias introduced by estimating the conditional efect. For the tabular foundation models considered here, reliable inference must account for both context variation, arising from the randomness of the finite labeled sample supplied to the model, and limited-context bias, the systematic diference between the estimator’s expectation across sampled contexts and the target.

In our experiments, we use a T-learner: the pretrained model estimates the conditional mean outcome separately under treatment and control, and their diference estimates CATE. Other learners could also be considered. For example, the DR-learner combines estimated outcome regressions and treatment probabilities to construct pseudo-outcomes, which are then regressed on the covariates to estimate CATE [Kennedy, 2023]. Such an approach could reduce the contribution of nuisance-estimation errors, although the final regression step may still introduce approximation or regularization bias. Here, we focus on uncertainty quantification for the T-learner, using TabCUQ to estimate context variation and provide an empirical bias cushion. Extending this construction to other learners would require accounting for their additional estimation steps and assessing the resulting interval coverage.

## 4 Theory

To clarify the statistical reasoning behind TabCUQ, we study outcome perturbations in the setting of kernel regression. This tractable setting allows us to examine how the perturbed estimates relate to sampling variation and how the bias cushion provides an upper bound for the nonparametric bias. The purpose of our theoretical analysis is to explain the rationale for TabCUQ’s interval construction, especially the role of the empirical bias cushion, rather than to establish theoretical guarantees for pretrained tabular foundation models.

We first introduce the basic setup of kernel regression and then adapt the perturbation procedure described in Sections 3.1 and 3.2 to this setting. Let $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ be i.i.d. observations of $( X , Y )$ where $X \in \mathbb { R } ^ { d } , Y \in \mathbb { R }$ , and $\mathbb { E } | Y | < \infty$ . Throughout this section, we fix our query feature at $\boldsymbol { x } \in \mathbb { R } ^ { d }$ and our target is the regression function $\mu ( x ) = \mathbb { E } [ Y \mid X = x ]$ . Defining $\varepsilon _ { i } = Y _ { i } - \mu ( X _ { i } )$ gives the representation

$$
Y _ { i } = \mu ( X _ { i } ) + \varepsilon _ { i } , \qquad \mathbb { E } [ \varepsilon _ { i } \mid X _ { i } ] = 0 .
$$

We write $\sigma ^ { 2 } ( x ) = \mathbb { E } [ \varepsilon _ { i } ^ { 2 } \mid X _ { i } = x ]$ and assume $\sigma ^ { 2 } ( x )$ is continuous at x.

For a nonnegative kernel K and bandwidth $h _ { n } > 0$ , define the kernel weights

$$
w _ { i } ( x ) = \frac { K ( ( X _ { i } - x ) / h _ { n } ) } { \sum _ { j = 1 } ^ { n } K ( ( X _ { j } - x ) / h _ { n } ) } , \qquad 1 \leq i \leq n ,\tag{13}
$$

with all weights set to zero when the denominator is zero. When the denominator is positive, the Nadaraya-Watson estimator is the solution to the local-constant least-squares problem

$$
{ \widehat { \mu } } ( x ) = \underset { a \in \mathbb { R } } { \arg \operatorname* { m i n } } \sum _ { i = 1 } ^ { n } w _ { i } ( x ) ( Y _ { i } - a ) ^ { 2 } = \sum _ { i = 1 } ^ { n } w _ { i } ( x ) Y _ { i } .\tag{14}
$$

Intuitively, equation (14) estimates $\mu ( x )$ by a weighted average of the responses, with weights determined by the kernel, the bandwidth, and the feature locations relative to x.

In this kernel-regression setting, we construct the outcome perturbations without cross-fitting. We form the residuals $r _ { i } = Y _ { i } - \widehat \mu ( X _ { i } )$ , where $\widehat { \mu } ( X _ { i } )$ is the estimator in (14) evaluated at $X _ { i } ,$ and center them as $e _ { i } = r _ { i } - { \bar { r } }$ , where $\begin{array} { r } { \bar { r } = n ^ { - 1 } \sum _ { j = 1 } ^ { n } r _ { j } } \end{array}$ . These fitted values and residuals are computed once and reused in all perturbations. For each perturbation $b = 1 , \dots , B$ , we generate $Y _ { i , b } = \hat { \mu } ( X _ { i } ) + v _ { i , b } e _ { i }$ using the Rademacher multipliers described in Section 3.1. Keeping the features and kernel weights fixed, we replace the responses in (14) with $Y _ { i , b }$ to obtain $\begin{array} { r } { \widehat { \mu } _ { b } ( x ) = \sum _ { i = 1 } ^ { n } w _ { i } ( x ) Y _ { i , b } } \end{array}$ As in Section 3.2, we compute the standard-error estimate ${ \widehat { s } } ( x )$ from the perturbed estimates $\{ \widehat { \mu } _ { b } ( x ) \} _ { 1 \leq b \leq B }$ The empirical bias cushion ${ \widehat { r } } ( x )$ is the largest absolute diference between all perturbed estimates $\{ \widehat { \mu } _ { b } ( x ) \} _ { 1 \leq b \leq B }$ and the original estimate ${ \widehat { \mu } } ( x )$ . We combine these quantities to form the interval $C _ { \alpha } ( D _ { n } ; x )$ in (10). We analyze this construction under the following assumptions.

Assumption 1 (Kernel regression assumptions). 1. The observations $( X _ { i } , Y _ { i } )$ are i.i.d., with $X _ { i } \in \mathbb { R } ^ { d }$ and satisfy $Y _ { i } = \mu ( X _ { i } ) + \varepsilon _ { i }$ with $\mathbb { E } | \mu ( X ) | < \infty$ . The design distribution has a density $f _ { X }$ that is continuous at the fixed query x, with $0 < f _ { X } ( x ) < \infty$ . There exists a finite constant $C > 0$ such that

$$
\mathbb { E } [ \varepsilon _ { i } \mid X _ { i } ] = 0 , \qquad \mathbb { E } [ \varepsilon _ { i } ^ { 2 } \mid X _ { i } ] = \sigma ^ { 2 } ( X _ { i } ) , \qquad \mathbb { E } [ \varepsilon _ { i } ^ { 4 } \mid X _ { i } ] \leq C \quad \mathrm { a l m o s t ~ s u r e l y } ,
$$

where $\sigma ^ { 2 } ( x ) > 0$ is continuous at x.

2. The regression function $\mu$ is Hölder smooth in the following sense:: For some $\beta > 0$ and finite $L > 0$ , the regression function $\mu : \mathbb { R } ^ { d } $ R is ℓ times continuously diferentiable with $\ell = \lceil \beta \rceil - 1$ , and satisfies

$$
| D ^ { \gamma } \mu ( u ) - D ^ { \gamma } \mu ( v ) | \leq L \| u - v \| ^ { \beta - \ell } , \qquad u , v \in \mathbb { R } ^ { d } , \quad | \gamma | = \ell .
$$

where $\gamma = ( \gamma _ { 1 } , \ldots , \gamma _ { d } ) \in \mathbb { N } _ { 0 } ^ { d }$ with $\mathbb { N } _ { 0 } ^ { d }$ denoting the set of d-tuples of nonnegative integers, $\begin{array} { r } { | \gamma | = \sum _ { j = 1 } ^ { d } \gamma _ { j } , D ^ { \gamma } } \end{array}$ denotes the corresponding partial derivative with $D ^ { \mathbf { 0 } } \mu = \mu$ and $\| \cdot \|$ is the Euclidean norm.

3. There exist constants $0 < r _ { 0 } \le 1$ and $0 < K _ { \operatorname* { m i n } } \le K _ { \operatorname* { m a x } } <$ ∞ such that

$$
K _ { \operatorname* { m i n } } \mathbf { 1 } \{ \left\| u \right\| \leq r _ { 0 } \} \leq K ( u ) \leq K _ { \operatorname* { m a x } } \mathbf { 1 } \{ \left\| u \right\| \leq 1 \} , \qquad u \in \mathbb { R } ^ { d } .
$$

The kernel K is of order $\ell = \lceil \beta \rceil - 1$ , that is, for every multi-index $\gamma = ( \gamma _ { 1 } , \ldots , \gamma _ { d } ) \in \mathbb { N } _ { 0 } ^ { d }$ with $| \gamma | \leq \ell ,$

$$
\int _ { \mathbb { R } ^ { d } } K ( u ) d u = 1 , \qquad \int _ { \mathbb { R } ^ { d } } \prod _ { j = 1 } ^ { d } u _ { j } ^ { \gamma _ { j } } K ( u ) d u = 0 , \quad 1 \leq | \gamma | \leq \ell .
$$

In addition, the kernel satisfies $\begin{array} { r } { \int _ { \mathbb { R } ^ { d } } \| u \| ^ { \beta } | K ( u ) | d u < \infty } \end{array}$

4. The deterministic bandwidth satisfies

$$
h _ { n } \to 0 , \qquad \frac { n h _ { n } ^ { d } } { \log n } \to \infty .
$$

In the above assumption, the Hölder smoothness of the function $\mu ( \cdot )$ and kernel assumptions are standard in nonparametric estimation; see Tsybakov [2009].

In the following, we shall use the Nadaraya–Watson estimator as an analytically tractable example to provide reasoning for our TabCUQ proposal. We use the terms context variation and limited context bias to connect the interpretation of its estimation error with our discussion of tabular foundation models. For the Nadaraya–Watson estimator, these two components admit an explicit characterization through the exact decomposition

$$
{ \widehat { \mu } } _ { \mathrm { N W } } ( x ) - \mu ( x ) = \underbrace { { \widehat { \mu } } ^ { \mathrm { o r } } ( x ) - \mu ( x ) } _ { \mathrm { c o n t e x t ~ v a r i a t i o n } } + \underbrace { b _ { n } ( x ) } _ { \mathrm { l i m i t e d ~ c o n t e x t ~ b i a s } }\tag{15}
$$

where ${ \widehat { \mu } } ^ { \mathrm { o r } } ( x )$ denotes an oracle estimator of $\mu$ and $b _ { n } ( x )$ denotes the bias component, defined as

$$
{ \widehat { \mu } } ^ { \mathrm { o r } } ( x ) : = \mu ( x ) + \sum _ { i = 1 } ^ { n } w _ { i } ( x ) \varepsilon _ { i } \quad { \mathrm { a n d } } \quad b _ { n } ( x ) = \sum _ { i = 1 } ^ { n } w _ { i } ( x ) \{ \mu ( X _ { i } ) - \mu ( x ) \} .\tag{16}
$$

Conditional on the observed covariates, context variation corresponds to the mean-zero fluctuation induced by the outcome noise. Limited context bias corresponds to the discrepancy between the weighted regression values within the context and the target $\mu ( x )$ . The oracle estimator removes this bias while preserving context variation. The two quantities used by TabCUQ correspond to the two terms in (15): the standard-error estimate ${ \widehat { s } } ( x )$ estimates the scale $\rho _ { n }$ of the oracle fluctuation, while the empirical bias cushion ${ \widehat { r } } ( x )$ provides a high-probability upper bound for the bias $b _ { n } ( x )$

Under Assumption 1, the oracle estimator satisfies

$$
{ \frac { { \widehat { \mu } } ^ { \mathrm { o r } } ( x ) - \mu ( x ) } { \rho _ { n } } } \overset { d } { \to } N ( 0 , 1 ) \qquad \mathrm { w i t h } \qquad \rho _ { n } = \sigma ( x ) \sqrt { \frac { \nu _ { K } } { n h _ { n } ^ { d } f _ { X } ( x ) } } ,\tag{17}
$$

where $\begin{array} { r } { \nu _ { K } = \int _ { \mathbb { R } ^ { d } } K ( u ) ^ { 2 } } \end{array}$ du and $\rho _ { n }$ describes the asymptotic sampling scale of ${ \widehat { \mu } } ^ { \mathrm { o r } } ( x ) - \mu ( x )$

Theorem 1 (An upper bound on bias). Under Assumption 1, suppose the number B of perturbations satisfy

$$
\frac { n h _ { n } ^ { 2 \beta + d } } { \log B } \longrightarrow 0 .\tag{18}
$$

Then, as $n \to \infty$ , the empirical bias cushion rb(x) defined in (9) satisfies

$$
\mathbb { P } \{ | b _ { n } ( x ) | \leq { \widehat { r } } ( x ) \} \longrightarrow 1 .\tag{19}
$$

Theorem 1 shows that the empirical bias cushion is a high probability upper bound for the bias component. On the event $| b _ { n } ( x ) | \leq { \widehat { r } } ( x )$ , the decomposition in (15) gives $| { \widehat { \mu } } ( x ) - \mu ( x ) | \leq$ $| \widehat { \mu } ^ { \mathrm { o r } } ( x ) - \mu ( x ) | + \widehat { r } ( x )$ . Together with the limiting distribution in (17), the bias bound therefore yields a pointwise coverage guarantee for $C _ { \alpha } ( D _ { n } ; x )$ . The following theorem states this guarantee and bounds the interval width.

Theorem 2 (Pointwise coverage and interval width). Under Assumption 1 and suppose that B satisfies (18). For fixed $\alpha \in ( 0 , 1 )$ , the confidence interval $C _ { \alpha } ( D _ { n } ; x )$ satisfies

$$
\operatorname* { l i m } _ { n \to \infty } \operatorname* { i n f } \mathbb { P } \{ \mu ( x ) \in C _ { \alpha } ( D _ { n } ; x ) \} \geq 1 - \alpha .\tag{20}
$$

Moreover,

$$
\mathrm { l e n g t h } \{ C _ { \alpha } ( D _ { n } ; x ) \} = O _ { p } \Big ( \rho _ { n } \sqrt { \log B } \Big ) .\tag{21}
$$

Theorem 2 establishes asymptotic pointwise coverage of at least $1 - \alpha$ and provides an upper bound on the interval width. The usual kernel-regression error scale combines sampling variability and bias, $( n h _ { n } ^ { d } ) ^ { - 1 / 2 } + h _ { n } ^ { \beta } .$ . The width bound depends on this budget through the factor $\sqrt { \log B _ { n } }$ highlighting a tradeof among bias accommodation, interval width, and computational cost. The bias component depends on the unknown regression function and is generally harder to estimate than the standard error. Balancing these terms gives $h _ { n } \asymp n ^ { - 1 / ( 2 \beta + d ) }$ and the classical optimal pointwise length benchmark $n ^ { - \beta / ( 2 \bar { \beta } + d ) }$ . Under the theorem’s conditions, choosing this bandwidth and $B = \lceil n ^ { c } \rceil$ for any fixed $c > 0$ yields length $\{ C _ { \alpha } ( D _ { n } ; x ) \} = O _ { p } ( n ^ { - \beta / ( 2 \beta + d ) } \sqrt { \log n } )$ . Our length bound is therefore near-optimal up to a logarithmic factor, which arises from taking the maximum over perturbations to accommodate bias without directly estimating it.

## 5 Simulation Studies

## 5.1 Experimental design and benchmarks

We evaluate the coverage and width of nominal 95% confidence intervals for conditional means $\mu ( x )$ and conditional average treatment efects $\tau ( x )$ in simulations where the target values are known. The primary experiments use TabPFN 2.6 with eight ensemble members. Each Monte Carlo repetition generates an independent labeled context, including both features and outcomes, while the queries, pretrained model, and model seed remain fixed. Table 1 summarizes the main data-generating processes. For conditional means, we consider univariate and multivariate regression models with diferent mean functions, outcome noise distributions, and covariate structures. For CATEs, we consider nine designs that vary the baseline response, treatment-efect function, outcome noise, and treatment assignment.

We compare TabCUQ with confidence intervals based solely on the wild-bootstrap standard error (referred to as Wild bootstrap). Both methods use the same point estimate and estimated standard error, so this comparison isolates the contribution of the empirical bias cushion. For CATE estimation, we use a T-learner with separate contexts for the two treatment arms. We also consider an oracle bias-aware (OBA) benchmark that uses the known target value and independent datasets generated from the true data-generating distribution to estimate the bias and sampling standard deviation of the original estimator. These quantities are used to construct benchmark intervals that provide a reference for interval width. The OBA method relies on information available in our simulations but unavailable in practice.

At each query point, coverage is estimated by the proportion of Monte Carlo confidence intervals containing the true target, while interval width is averaged across repetitions. We also evaluate the

Table 1: Synthetic data-generating processes for conditional means and CATE. U and M stand for paired univariate and multivariate designs, respectively; C stands for the CATE designs.
<table><tr><td colspan="4">A. Conditional means:  $Y = \mu ( X ) + \varepsilon ( X )$ </td></tr><tr><td colspan="4">Univariate:  $X \sim N ( 0 , 1 )$  . Multivariate:  $X \sim N ( 0 , \Sigma _ { \rho } )$  , with  $p \in \{ 5 , 1 0 \}$  and  $\rho = 0$  for  $_ \mathrm { M 1 - M 8 . }$  M1 also uses  $\rho = 0 . 5 ;$  and</td></tr><tr><td>M9 uses  $p = 2$   $\rho \in \{ 0 , 0 . 5 \}$  Design</td><td>Univariate mean</td><td>Multivariate mean</td><td>Outcome noise</td></tr><tr><td>Linear (U1 / M1)</td><td>x</td><td> $\textstyle \sum _ { j = 1 } ^ { p } x _ { j }$ </td><td>Z</td></tr><tr><td>Quadratic  $\left( \mathrm { U 2 / \ M 2 } \right)$ </td><td> $x ^ { 2 }$ </td><td> $\textstyle \sum _ { j = 1 } ^ { \bar { p } } x _ { j } ^ { 2 }$ </td><td>Z</td></tr><tr><td>Bimodal (U3 / M3)</td><td>x</td><td> $\textstyle \sum _ { j = 1 } ^ { p } x _ { j }$ </td><td> $1 . 5 U + Z$ </td></tr><tr><td>Heteroskedastic linear</td><td>x</td><td> $\textstyle \sum _ { j = 1 } ^ { p } x _ { j }$ </td><td> $\mathrm { 1 D } \colon ( 1 + | x | ) Z$ </td></tr><tr><td>(U4 / M4)</td><td></td><td></td><td> $\big ( 1 + { \frac { | \sum _ { j } x _ { j } | } { p } } \big ) Z$  Multi:</td></tr><tr><td>Absolute value (U5 / M5)</td><td> $| x |$ </td><td> $\scriptstyle \sum _ { j = 1 } ^ { p } | x _ { j } |$ </td><td>Z</td></tr><tr><td>Sine (U6 / M6)</td><td> $\sin ( \pi x )$ </td><td> $\textstyle \sum _ { j = 1 } ^ { p } \sin ( \pi x _ { j } )$ </td><td>Z</td></tr><tr><td>Step (U7 / M7)</td><td> $\operatorname { s g n } _ { + } ( x )$ </td><td> $\textstyle \sum _ { j = 1 } ^ { p } \operatorname { s g n } _ { + } ( x _ { j } )$ </td><td>Z</td></tr><tr><td>Heteroskedastic quadratic (U8 / M8)</td><td> $x ^ { 2 }$ </td><td> $\textstyle \sum _ { j = 1 } ^ { p } x _ { j } ^ { 2 }$ </td><td> $\mathrm { 1 D } \colon ( 1 + | x | ) Z$ </td></tr><tr><td></td><td></td><td></td><td> $\big ( 1 + \frac { | \sum _ { j } x _ { j } | } { p } \big ) Z$  Multi:</td></tr><tr><td>Interaction (M9)</td><td></td><td> $x _ { 1 } + x _ { 2 } + x _ { 1 } x _ { 2 }$ </td><td>Z</td></tr></table>

$$
. \ C A T E \colon Y = \mu _ { 0 } ( X ) + A \tau ( X ) + \varepsilon ( X )
$$

$X \sim N ( 0 , 1 )$ and $A \mid X = x \sim$ Bernoulli{e(x)}. Common settings are $\varepsilon ( x ) = Z$ and $e ( x ) = 0 . 5 ;$ a dash in the final column denotes these settings.
<table><tr><td>Design</td><td>Baseline  $\mu _ { 0 } ( x )$ </td><td>Effect  $\tau ( x )$ </td><td>Noise / assignment change</td><td></td></tr><tr><td>C1: Linear effect</td><td>x</td><td>x</td><td></td><td></td></tr><tr><td>C2: Null effect</td><td>x</td><td>0</td><td></td><td></td></tr><tr><td>C3: Opposite slopes</td><td> $- x$ </td><td>2x</td><td></td><td></td></tr><tr><td>C4: Quadratic baseline</td><td> $x ^ { 2 }$ </td><td>x</td><td></td><td></td></tr><tr><td>C5: Bimodal noise</td><td>x</td><td>x</td><td> $\varepsilon ( x ) = 1 . 5 U + Z$ </td><td></td></tr><tr><td>C6: Heteroskedastic noise</td><td>x</td><td>x</td><td> $\varepsilon ( x ) = ( 1 + | x | ) Z$ </td><td></td></tr><tr><td>C7: Nonlinear effect</td><td>x</td><td> $x ^ { 2 }$ </td><td></td><td></td></tr><tr><td>C8: Treatment imbalance</td><td>x</td><td>x</td><td> $e ( x ) = 0 . 2$ </td><td></td></tr><tr><td>C9: Weak overlap</td><td>x</td><td>x</td><td></td><td> $e ( x ) = \{ 1 + \exp ( x ) \} ^ { - 1 }$ </td></tr></table>

Here $Z \sim N ( 0 , 1 )$ and $U \in \{ - 1 , 1 \}$ with equal probabilities are independent of the covariates and of each other. We use $\mathrm { s g n } _ { + } ( z ) = 1$ for $z \geq 0$ and −1 otherwise. The covariance $\Sigma _ { \rho }$ has unit diagonal and of-diagonal entries $\rho .$ In Panel $\mathrm { A } ,$ a noise formula without a dimension label applies to both univariate and multivariate versions; the interaction design is multivariate only.

standard-error estimates. We report both pointwise results and averages over prespecified query regions. For the univariate and CATE settings, we use 20 equally spaced points in $[ - 3 , 3 ]$ , including eight edge points satisfying $| x | \geq 2$ . For multivariate regression, we consider 20 query points in each of three Mahalanobis probability strata (0–50%, 50–90%, and 90–99%). The main univariate experiments use $S = 3 0 0$ independent contexts of size $n = 3 0 0$ , with $B = 3 0 0$ wild-bootstrap perturbations per context. For multivariate regression, we use $S = 1 0 0$ independent contexts of size $n = 3 0 0$ , with $B = 1 0 0$ wild-bootstrap perturbations per context. The CATE experiments use the same numbers of contexts and perturbations, with a total sample size of $n = 6 0 0$ across the two treatment arms. Complete description of the simulation settings and evaluation procedures are given in Appendix B.

## 5.2 Main results

TabCUQ substantially improves conditional-mean coverage across the univariate and multivariate designs. For the eight univariate designs, coverage averaged over the edge queries ranges from 0.677 to 0.945 for Wild bootstrap and from 0.970 to 1.000 for TabCUQ; see Figure 5. The empirical bias cushion thus raises the regional average coverage above the nominal 0.95 level in all eight designs. The greatest improvement occurs in the sinusoidal design, where the coverage increases from 0.677 to 0.970. TabCUQ also improves coverage in all 20 multivariate settings evaluated in the 90–99% Mahalanobis stratum; see Figure 6. In these settings, the average coverage ranges from 0.523 to 0.851 for the Wild bootstrap and from 0.928 to 0.998 for TabCUQ. The details are given in Table 7 in Appendix D.

![](images/f496d2ab214682b4ffa0e2b8043b36ca5403919efba59e85ee97b19424663695.jpg)  
Figure 5: Univariate conditional-mean interval performance for TabPFN 2.6 across eight designs, with $S = 3 0 0$ $n = 3 0 0$ , and $B = 3 0 0 .$ . (a) Coverage averaged over the eight edge queries satisfying $| x | \geq 2$ . (b) Mean pointwise interval-width/OBA-width ratio over the same queries. Dashed reference lines mark 0.95 coverage and a width ratio of one.

Detailed results under the quadratic design are presented in Figure 7 to illustrate the role of the empirical bias cushion in improving coverage in queries with substantial limited-context bias. Figure 7(a) shows that the absolute Monte Carlo bias is small near the center of the query grid but increases sharply toward its endpoints. The mean empirical bias cushion also increases toward these endpoints, providing a larger addition to the interval half-width where the bias is most pronounced. At $x = 3$ , the absolute Monte Carlo bias is 1.247, and the mean empirical bias cushion is 2.171. The Wild bootstrap interval achieves coverage of only 0.683 at this query. Adding the empirical bias cushion increases coverage to 0.937, close to the nominal 0.95 level; see Figure 7(b). Figure 7(c) shows the corresponding increase in mean interval width, with the largest absolute increases near the endpoints of the grid. TabCUQ retains the original point estimate as the interval center and improves coverage by adding the empirical bias cushion for limited-context bias to the interval half-width.

The CATE experiments reveal an additional source of variation in coverage arising from the combination of arm-specific estimation biases. Across the nine CATE designs, edge-average coverage ranges from 0.729 to 0.968 for Wild bootstrap intervals and from 0.968 to 1.000 for TabCUQ; see Figure 8. Let $b _ { a } ( x )$ denote the bias of the estimated conditional mean in treatment arm a. The bias of the estimated CATE then satisfies $b _ { \tau } ( x ) = b _ { 1 } ( x ) - b _ { 0 } ( x )$ , so the arm-specific biases may cancel or reinforce each other. In the null-efect design, the edge-average absolute CATE bias is 0.062, and the Wild bootstrap interval achieves coverage of 0.950. In contrast, the opposite-slopes design produces an edge-average absolute CATE bias of 0.537, and Wild bootstrap coverage falls to 0.730. The corresponding standard-error calibration ratios are similar, at 1.103 and 1.107, respectively. Adding the empirical bias cushion increases coverage in the opposite-slopes design to 0.989. Our CATE experiments also examine nonlinear treatment efects and changes in treatment assignment. Wild bootstrap coverage falls to 0.854 under treatment imbalance, 0.839 for a nonlinear efect, and 0.729 under weak overlap. TabCUQ increases coverage to 0.990, 0.983, and 0.968, respectively. Weak overlap remains particularly challenging at individual queries. At $x = 3$ , where treated observations are sparse, coverage increases from 0.540 to only 0.890 after adding the empirical bias cushion.

![](images/b14413ff5a0bccf0c04f972d1fd5a950d1c4ba1a60638b14dd233fbd91d75237.jpg)  
Figure 6: Multivariate conditional-mean interval performance for TabPFN 2.6 across 20 settings, with $S = 1 0 0$ independent contexts of size $n = 3 0 0$ and $B = 1 0 0 .$ . (a) Coverage averaged over 20 fixed queries in the 90–99% Mahalanobis stratum. (b) Mean pointwise interval-width/OBA-width ratio over the same queries. Dashed reference lines mark 0.95 coverage and a width ratio of one.

We also compare interval widths with an oracle benchmark (OBA) constructed using the Monte Carlo bias and standard deviation. This benchmark provides a reference for interval eficiency using information unavailable from a single observed context. Across the univariate conditional-mean designs, the mean pointwise TabCUQ/OBA width ratios range from 1.64 to 2.44; the corresponding ratios range from 1.443 to 2.072 in the multivariate tail experiments. For CATE estimation, the ratios of regional mean TabCUQ width to regional mean OBA width range from 1.59 to 2.71 across the nine designs.

![](images/cdd7cfae108d1e806914d8f6a8711183737d05e8a7eb9e229f9303fcefe085f8.jpg)  
Absolute limited-context bias Empirical bias cushion

![](images/ac5ac33ba6b04063769e2ffd097228fa32b564996451131698017b93971a577a.jpg)

(c) Interval width  
![](images/699cda672031986d71ad21e853494edaffefcb851d90c8531c8f92cf7595fff1.jpg)

Figure 7: Pointwise conditional-mean results in the quadratic design for TabPFN 2.6, with $S = 3 0 0 .$ $n = 3 0 0$ , and $B = 3 0 0 .$ . (a) Absolute Monte Carlo limited-context bias (orange dashed) and mean empirical bias cushion (blue solid). (b) Coverage of wild-bootstrap and TabCUQ intervals; the horizontal dashed line marks 0.95. (c) Mean interval widths for wild bootstrap, TabCUQ, and OBA. The cushion in (a) is the added component of the TabCUQ half-width.  
(a) CATE Coverage  
![](images/3a557130403d3a341b3fc6c265634d1d65931b8d1abdb9d3037f21723a3c8e79.jpg)

(b) Relative width  
![](images/90c234ef5c8a9032cd57d380810583c9e1232fa1a3ce5ca195f5a74451df53d4.jpg)  
Figure 8: CATE interval performance for TabPFN 2.6 across nine designs, with $S = 1 0 0 { \mathrm { ; } }$ , total $n = 6 0 0$ , and $B = 1 0 0 .$ . (a) Coverage averaged over the eight edge queries satisfying $| x | \geq 2 .$ . (b) Mean pointwise interval-width/OBA-width ratio over the same queries. Dashed reference lines in $^ { ( \mathrm { a } , \mathrm { b } ) }$ mark 0.95 coverage and a width ratio of one.

## 5.3 Robustness and coverage–width trade-ofs

The coverage improvement extends to diferent pretrained models while interval widths vary across models. We evaluate TabCUQ with TabPFN 3 and TabFM using the same experimental design and perturbation budget as a matched TabPFN 2.6 comparison; see Table 2. Across the four univariate designs, TabCUQ achieves edge-average coverage of 0.986–1.000 with TabPFN 3 and 0.971–0.998 with TabFM, compared with 0.854–0.936 and 0.784–0.900 for the corresponding Wild bootstrap intervals. Similar improvements are observed in the multivariate experiments, where TabCUQ achieves tail-average coverage of 0.996–1.000 for TabPFN 3 and 0.998–1.000 for TabFM. The relative widths remain substantial and depend on the pretrained model, ranging from 1.7 to 2.4 times the corresponding OBA benchmark across these comparisons.

Table 2: Sensitivity to the choice of foundation models. Entries give ranges across all the simulation settings. Panel A concerns the eight univariate edge queries in U1–U4; Panel B concerns the 20 queries in the 90–99% Mahalanobis stratum. Relative width is the mean pointwise TabCUQ/OBA width ratio, using each model’s own Monte Carlo bias and standard deviation to construct its OBA reference. All three models are evaluated on the same nine multivariate settings: M1 at $p \in \{ 5 , 1 0 \}$ and $\rho \in \{ 0 , 0 . 5 \}$ , M2–M4 at $p = 5$ and $\rho = 0$ , and M9 at $p = 2$ and $\rho \in \{ 0 , 0 . 5 \}$
<table><tr><td>Model</td><td>Designs</td><td>Wild bootstrap coverage</td><td>TabCUQ coverage</td><td>TabCUQ width /OBA</td></tr><tr><td colspan="5">A. Univariate edge queries</td></tr><tr><td>TabPFN 2.6</td><td>4</td><td>0.738-0.910</td><td>0.966-1.000</td><td>1.773-2.082</td></tr><tr><td>TabPFN 3</td><td>4</td><td>0.854-0.936</td><td>0.986-1.000</td><td>1.967-2.419</td></tr><tr><td>TabFM</td><td>4</td><td>0.784–0.900</td><td>0.971-0.998</td><td>1.722-2.164</td></tr><tr><td colspan="5">B. Multivariate tail queries</td></tr><tr><td>TabPFN 2.6</td><td>9</td><td>0.633-0.851</td><td>0.981-0.998</td><td>1.651–2.072</td></tr><tr><td>TabPFN 3</td><td>9</td><td>0.852-0.915</td><td>0.996-1.000</td><td>2.044-2.291</td></tr><tr><td>TabFM</td><td>9</td><td>0.861-0.940</td><td>0.998-1.000</td><td>2.131–2.442</td></tr></table>

Increasing the context size reduces interval width but does not necessarily resolve Wild bootstrap undercoverage. We examine context sizes ranging from n = 100 to $n = 3 0 0 0$ in the four original univariate designs. In the linear design, increasing n from 100 to 3000 reduces the average TabCUQ interval width from 3.86 to 1.31, while coverage increases from 0.979 to 0.999. However, Wild bootstrap coverage remains at 0.845 even when $n = 3 0 0 0$ . The OBA interval also becomes narrower, with its mean width decreasing from 2.45 to 0.66. Thus, although larger contexts improve absolute precision, they do not necessarily eliminate the need for bias protection or reduce the relative width cost of TabCUQ. Additional sample-size results are reported in Appendix E.

The perturbation budget moderately afects the empirical bias cushion. Because the cushion is defined through the maximum absolute displacement among perturbed estimates, adding more perturbations cannot decrease this maximum when the draws are nested. In the four original univariate designs, increasing the number of perturbations from 100 to 500 increases the average wild-bootstrap displacement cushion by 8.5%–14.4%. Thus, the perturbation budget moderately afects the interval being constructed rather than merely the numerical precision of its computation. Additional results are reported in Appendix E.

## 6 Real-World Studies

## 6.1 Data sets and study design

We evaluate TabCUQ on seven real-world regression tasks spanning socioeconomic, environmental, behavioural, protein, and single-cell data. These tasks represent diferent prediction settings. Census income has a weak predictive signal and a heavy-tailed outcome, whereas tree-cover prediction has a stronger but highly nonlinear relationship between the covariates and outcome. Job Corps and Jester Jokes provide moderate- and high-dimensional regression settings. The two protein-stability tasks evaluate generalization to mutations on proteins largely absent from the training pool, with close sequence homologues removed. CITE-seq examines uncertainty quantification using 2,000 gene-expression features. We consider the task of CATE inference on the Jester Jokes dataset while implementing conditional-mean inference on all the other datasets. Detailed dataset descriptions and preprocessing procedures are provided in Appendix I.

Table 3: Basic information of the real-world datasets included in our studies.
<table><tr><td>Dataset</td><td>Pool</td><td>Evaluation</td><td>Input p</td><td>Outcome</td></tr><tr><td>Census</td><td>11,800</td><td>200</td><td>2</td><td>Income</td></tr><tr><td>Tree cover</td><td>11,800</td><td>200</td><td>4</td><td>Canopy cover</td></tr><tr><td>Job Corps</td><td>10,014</td><td>200</td><td>23</td><td>Earnings</td></tr><tr><td>Jester Jokes</td><td>48,245</td><td>200</td><td>110</td><td>Rating</td></tr><tr><td>Protein engineering  $\Delta \Delta G$ </td><td>8,754</td><td>461</td><td>145</td><td>Gibbs free energy changes  $( \Delta \Delta G )$ </td></tr><tr><td>Protein engineering  $\Delta T _ { m }$ </td><td>4,346</td><td>571</td><td>155</td><td>Melting temperature changes  $( \Delta T _ { m } )$ </td></tr><tr><td>CITE-seq CD71</td><td>30,469</td><td>200</td><td>2,000</td><td>CD71 (CLR)</td></tr></table>

Unlike in simulations, the population conditional mean or CATE is unknown in real data. We therefore evaluate confidence intervals relative to a fixed, data-derived reference function. For each dataset, we reserve an evaluation set and fit the reference function using the remaining observations. The reference model is selected by five-fold cross-validation from a collection of regression methods, including random forests, XGBoost, LightGBM, multilayer perceptrons, ridge regression, and a constant predictor. A simple Cell-mean predictor is additionally included for the Census dataset with discrete covariates only. The reference function is fitted using a substantially larger sample than the contexts supplied to the foundation model and is treated as fixed during evaluation. Consequently, the reported coverage concerns the estimated reference function rather than the unknown population conditional-mean or CATE functions.

We repeatedly sample contexts of size $n _ { \mathrm { c t x } } \in \{ 2 0 0 , 3 0 0 , 5 0 0 , 1 0 0 0 \}$ without replacement from the reference pool. For each sampled context, we obtain a point estimate and confidence interval at the same fixed evaluation queries. The reference function and evaluation queries remain unchanged across repetitions. We measure reference coverage by the proportion of intervals containing the large-sample reference function and summarize the interval widths across the repeated contexts. The main analyzes use 200 held-out evaluation points for all tasks except protein-stability prediction, while the two protein tasks use the evaluation sets of 461 and 571 mutations prespecified by the data providers. For each sampled context, we generate 100 wild-bootstrap perturbations of the observed outcomes for TabCUQ. Additional details on the setting and evaluation strategy are reported in Appendix J.

## 6.2 Results

TabCUQ substantially improves reference coverage with interval widths comparable in scale to the oracle benchmark. At $n _ { \mathrm { c t x } } = 1 0 0 0$ , Wild bootstrap intervals achieve average reference coverage between 0.614 and 0.842 across the seven tasks, whereas intervals with the wild-bootstrap displacement cushion achieve coverage between 0.966 and 1.000. The mean pointwise TabCUQ/OBA width ratios range from 1.62 to 2.27, indicating that the coverage improvements are achieved with interval widths of the same order as those of the oracle bias-aware benchmark. The improvement is observed in various inferential tasks. For example, coverage increases from 0.649 to 0.997 for Census income and from 0.682 to 0.973 for tree-cover prediction. Similar improvements occur in the biological application examples with higher dimensional predictors. In specific, the coverage increases from 0.779 to 0.997 for protein $\Delta \Delta G .$ , from 0.658 to 0.967 for protein $\Delta T _ { m } ,$ and from 0.614 to 0.966 for CITE-seq. These results show that the improvement is not confined to weak-signal problems, as substantial gains are also observed in nonlinear regression and molecular prediction tasks. Figure 9 presents pointwise coverage and relative-width comparisons for six of the tasks, with complete results for all seven reported in Appendix J.

Increasing the context size does not consistently resolve Wild bootstrap undercoverage. In the context-size sensitivity experiments, Wild bootstrap reference coverage decreases from 0.828 to 0.587 for Census and from 0.855 to 0.761 for Jester as $n _ { \mathrm { c t x } }$ increases from 100 to 2000. Tree cover shows a diferent pattern, with coverage increasing from 0.503 to 0.689, although it remains below the nominal level. Undercoverage also persists in the protein-stability tasks, where Wild bootstrap coverage ranges from 0.779 to 0.812 for $\Delta \Delta G$ and from 0.658 to 0.689 for $\Delta T _ { m }$ across context sizes of 100–1000. In a separate CITE-seq context-size analysis using 500 principal components, Wild bootstrap coverage increases from 0.703 to 0.760 but remains below nominal. This analysis uses a diferent feature representation and reference function from the main CITE-seq experiment. In contrast, TabCUQ intervals maintain reference coverage close to or above 0.95 across most context sizes and datasets. The contrasting trends across datasets indicate that increasing the number of context observations does not necessarily lead to better calibrated intervals. The complete results are provided in Appendix J.

Favorable average reference coverage can still conceal substantial diferences across individual queries. This is particularly evident in the main CITE-seq experiment using 1,000 gene-expression features. Although the TabCUQ intervals achieve average reference coverage of 0.966, approximately 10% of evaluation points have pointwise coverage below 0.90. The Job Corps and Jester analyses also show considerable variation in interval width across queries. These results suggest that the improvement in reference coverage is not uniform across evaluation points, and that favorable dataset-level averages should not be interpreted as evidence of adequate coverage at every query.

![](images/bbece372e05a7d8cd71570d3409194b7ef72071844e88337b38b0ffb8cb41671.jpg)  
Figure 9: Pointwise reference coverage and relative interval width across six real-world tasks, with context size n = 1,000. For each task, the left panel shows the distribution across evaluation queries of reference coverage. The right panel shows the distribution of mean interval width over those contexts divided by the OBA width at the same query. Wild bootstrap uses the estimated standard error without the empirical bias cushion; TabCUQ adds the empirical bias cushion from the same wild-bootstrap perturbations. Linear model denotes ordinary least-squares intervals, included only for the Census and Tree cover examples. OBA stands for the oracle benchmark. Inner boxes indicate medians and interquartile ranges; horizontal bars indicate identical values across queries. Dashed lines mark 0.95 coverage and a width ratio of one.

## AI use statement

We used generative AI tools to improve the clarity and wording of the manuscript and to assist in searching for relevant literature. The authors take full responsibility for the final content of the

paper, including its scientific claims and references.

## References

Timothy B. Armstrong and Michal Kolesár. Simple and honest confidence intervals in nonparametric regression. arXiv preprint arXiv:1606.01200, 2016. URL https://arxiv.org/abs/1606.01200.

Susan Athey and Guido Imbens. Recursive partitioning for heterogeneous causal efects. Proceedings of the National Academy of Sciences, 113(27):7353–7360, 2016. doi: 10.1073/pnas.1510489113.

Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. The limits of distribution-free conditional predictive inference. Information and Inference: A Journal of the IMA, 10(2):455–482, 2021. doi: 10.1093/imaiai/iaaa017.

Rajeev H. Dehejia and Sadek Wahba. Causal efects in nonexperimental studies: Reevaluating the evaluation of training programs. Journal of the American Statistical Association, 94(448): 1053–1062, 1999. doi: 10.1080/01621459.1999.10473858.

Bradley Efron. Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1): 1–26, 1979. doi: 10.1214/aos/1176344552.

Sandra Fortini, Kenyon Ng, Sonia Petrone, Judith Rousseau, and Susan Wei. Uncertainty decomposition for Bayes-filtered transformers via Bayesian predictive inference. arXiv preprint arXiv:2602.04596, 2026. URL https://arxiv.org/abs/2602.04596.

Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer, 2 edition, 2009. doi: 10.1007/978-0-387-84858-7.

Miguel A. Hernán and James M. Robins. Causal Inference: What If. Chapman & Hall/CRC, Boca Raton, 2020. URL https://miguelhernan.org/whatifbook.

Riley J. Hickman, Matteo Aldeghi, Florian Häse, and Alán Aspuru-Guzik. Bayesian optimization with known experimental and design constraints for chemistry applications. Digital Discovery, 1: 732–744, 2022. doi: 10.1039/D2DD00028H.

Oliver Hines, Oliver Dukes, Karla Diaz-Ordaz, and Stijn Vansteelandt. Demystifying statistical learning based on eficient influence functions. The American Statistician, 76(3):292–304, 2022. doi: 10.1080/00031305.2021.2021984.

Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, Frank Hutter, et al. Accurate predictions on small data with a tabular foundation model. Nature, 637:319–326, 2025. doi: 10.1038/s41586-024-08328-6.

Edward H. Kennedy. Towards optimal doubly robust estimation of heterogeneous causal efects. Electronic Journal of Statistics, 17(2):3008–3049, 2023. doi: 10.1214/23-EJS2157.

Patrick Kline and Andres Santos. A score based approach to wild bootstrap inference. Journal of Econometric Methods, 1(1):23–41, 2012. doi: 10.1515/2156-6674.1006.

Weihao Kong and Abhimanyu Das. Introducing TabFM: A zero-shot foundation model for tabular data. Google Research Blog, 2026. URL https://research.google/blog/ introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/. Accessed 2026- 08-27.

Sören R. Künzel, Jasjeet S. Sekhon, Peter J. Bickel, and Bin Yu. Metalearners for estimating heterogeneous treatment efects using machine learning. Proceedings of the National Academy of Sciences, 116(10):4156–4165, 2019. doi: 10.1073/pnas.1804597116.

Robert J. LaLonde. Evaluating the econometric evaluations of training programs with experimental data. The American Economic Review, 76(4):604–620, 1986. URL https://www.jstor.org/ stable/1806062.

Jing Lei, Max G’Sell, Alessandro Rinaldo, Ryan J. Tibshirani, and Larry Wasserman. Distributionfree predictive inference for regression. Journal of the American Statistical Association, 113(523): 1094–1111, 2018. doi: 10.1080/01621459.2017.1307116.

Lihua Lei and Emmanuel J. Candès. Conformal inference of counterfactuals and individual treatment efects. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 83(5):911–938, 2021. doi: 10.1111/rssb.12445. URL https://doi.org/10.1111/rssb.12445.

Enno Mammen. Bootstrap and wild bootstrap for high dimensional linear models. The Annals of Statistics, 21(1):255–285, 1993. doi: 10.1214/aos/1176349025.

Samuel Müller, Noah Hollmann, Sebastian Pineda Arango, Josif Grabocka, and Frank Hutter. Transformers can do Bayesian inference. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=KSugKcbNf9.

Thomas Nagler and David Rügamer. Uncertainty quantification for prior-data fitted networks using martingale posteriors. arXiv preprint arXiv:2505.11325, 2025. doi: 10.48550/arXiv.2505.11325. URL https://arxiv.org/abs/2505.11325.

Kenyon Ng, Edwin Fong, David T. Frazier, Jeremias Knoblauch, and Susan Wei. TabMGP: Martingale posterior with TabPFN. In International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2510.25154.

Sungwoo Park, Sunghee Park, and Won Chang. Conformal prediction and conditional coverage for tabular foundation models. arXiv preprint arXiv:2609.34887, 2026. URL https://arxiv.org/ abs/2609.34887.

Dimitris N. Politis, Joseph P. Romano, and Michael Wolf. Subsampling. Springer, New York, 1999. doi: 10.1007/978-1-4612-1554-7.

Jingang Qu, David Holzmüller, Gaël Varoquaux, and Marine Le Morvan. TabICL: A tabular foundation model for in-context learning on large data. arXiv preprint arXiv:2502.05564, 2025. URL https://arxiv.org/abs/2502.05564.

Glenn Shafer and Vladimir Vovk. A tutorial on conformal prediction. Journal of Machine Learning Research, 9(12):371–421, 2008. URL https://jmlr.org/papers/v9/shafer08a.html.

Benjamin J. Shields, Jason Stevens, Jun Li, Marvin Parasram, Farhan Damani, Jesus I. Martinez Alvarado, Jacob M. Janey, Ryan P. Adams, and Abigail G. Doyle. Bayesian reaction optimization as a tool for chemical synthesis. Nature, 590:89–96, 2021. doi: 10.1038/s41586-021-03213-y.

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer Series in Statistics. Springer, New York, 2009. doi: 10.1007/b13794. URL https://doi.org/10.1007/b13794.

Larry Wasserman. All of Statistics: A Concise Course in Statistical Inference. Springer, 2004. doi: 10.1007/978-0-387-21736-9.

Larry Wasserman. All of Nonparametric Statistics. Springer Texts in Statistics. Springer, 2006. doi: 10.1007/0-387-30623-4.

C. F. Jef Wu. Jackknife, bootstrap and other resampling methods in regression analysis. The Annals of Statistics, 14(4):1261–1295, 1986. doi: 10.1214/aos/1176350142.

Yunxin Xu, Di Liu, and Haipeng Gong. Improving the prediction of protein stability changes upon mutations by geometric learning and a pre-training strategy. Nature Computational Science, 4: 840–850, 2024. doi: 10.1038/s43588-024-00716-2.