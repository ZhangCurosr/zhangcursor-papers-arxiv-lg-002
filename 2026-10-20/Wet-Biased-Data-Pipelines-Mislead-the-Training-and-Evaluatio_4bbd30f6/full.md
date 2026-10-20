# Wet-Biased Data Pipelines Mislead the Training and Evaluation of Generative Precipitation Nowcasting

Christoph Metzl<sup>1,\*</sup>, Stefan Kapsreiter<sup>1,2</sup>, Kianusch Vahid Yousefnia<sup>1</sup>, Virginia Poli<sup>3</sup>, Richard M¨uller<sup>4</sup>, Frank Kr¨uger<sup>2</sup>, and Tobias B¨olle<sup>1</sup>

<sup>1</sup>Deutsches Zentrum f¨ur Luft- und Raumfahrt, Institut f¨ur Physik der Atmosph¨are, Oberpfafenhofen, Germany

<sup>2</sup>Wismar University of Applied Sciences, Wismar, Germany

<sup>3</sup>Arpae Emilia-Romagna, Hydro-Meteo-Climate Service (SIMC), Bologna, Italy

<sup>4</sup>German Weather Service, Ofenbach, Germany Correspondence and requests for materials should be addressed to Christoph Metzl (christoph.metzl@dlr.de).

## 1 Abstract

Generative models for precipitation nowcasting are trained and evaluated predominantly on wet-biased datasets, which over-represent precipitation events to emphasize rare extremes. Selecting test cases based on target observations is known to favor forecasts that match the selected distribution rather than the intended deployment distribution. We show analytically how this targetdependent sampling also shifts the conditional distribution learned by generative models. When training and test sampling match, even an ideal model learns precisely the distribution favored by evaluation, allowing the evaluation bias to conceal the training bias. Using a fixed satellite-to-radar difusion model, we demonstrate that target-dependent test sampling can reverse the calibration assessment. The model appears under-predictive on a wet-biased test set but over-predictive on a deployment-representative test set. This result is replicated outside the model’s training domain after a seven-year gap. Utilizing classifierfree guidance as an example, we demonstrate how the contradictory calibration assessment can misdirect downstream model development.

## 2 Introduction

Extreme rainfall can endanger large populations through flooding [1] and cause substantial infrastructure damage [2]. It is expected to become more frequent and intense in the face of climate change [3], although the magnitude of these responses varies regionally [4]. This underscores the need for timely and skillful forecasts of heavy precipitation [5].

Nowcasting refers to weather forecasting at very short lead times, typically on the order of a few hours, with very high spatial and temporal resolution [6]. In this regime, numerical weather prediction (NWP) models face several limitations, including latency associated with observation processing and data assimilation, model spin-up, and insuficient spatiotemporal resolution [7]. Although emerging Machine Learning (ML) based weather models achieve competitive forecast skill compared to NWP [8], most still rely on NWP analyses for initialization and consequently inherit limitations associated with latency and resolution [9].

Traditional nowcasting methods, while data driven, are physically motivated and rely on motion field estimation and advection but struggle with forecasting the growth and decay of rain cells [10–12]. Early deterministic ML nowcasting models [13, 14] achieve superior aggregate skill relative to conventional extrapolation baselines, particularly for low precipitation, but their pixelwise regression objectives tend to blur aggressively, leading to overly smooth precipitation fields [14, 15]. Alternatively, nowcasting can be formulated as a probabilistic classification task over discrete intensity bins [16]. These probability maps can achieve strong verification scores, but representative deterministic fields may lack sharp, realistically organized precipitation structures [17].

Therefore, generative modeling is an increasingly prominent direction in precipitation nowcasting [18–29]. Generative models like Generative Adversarial Networks (GANs)[30], difusion [31] and flow matching [32] models aim to produce ensembles of spatially coherent precipitation fields.

Special emphasis is placed on forecast skill for extreme precipitation [18, 19, 33], which is addressed through innovations in model architectures [34, 35] and loss functions [36, 37]. The rare event focus additionally motivates the evaluation on wet-biased test sets, which over-represent precipitation events relative to their climatological frequency. Wet-biased training sets also can help address the challenges of learning from rare events [38]. Thus, a substantial number of studies and in particular seminal works [18–20, 22, 23, 39] use wetbiased datasets for both training and testing. When cases are selected using the target observations, this practice shifts both the training and test distributions away from the intended deployment distribution. We refer to this mismatch as distribution shift.

The crucial risk is that evaluation can reward the very bias introduced during training. A generative model is trained to reproduce the rainfall probabilities represented in the sampled data. Evaluation on similarly sampled cases can therefore conceal their departure from the probabilities needed under deployment conditions. In practice, the assessment can even misidentify the direction of forecast bias: a model that over-predicts rain under deployment conditions may appear under-predictive on a wet-biased test set, suggesting an adjustment in the wrong direction. We examine this interaction theoretically and investigate its evaluation consequences through a controlled comparison of the same model on diferently sampled test sets.

On the evaluation side, verification on samples that have been selected based on the forecast targets or future observations is a known failure mode, referred to as the forecaster’s dilemma [40]. This approach does not estimate performance over the intended deployment distribution anymore. Proper skill scores become improper [41] and calibration diagnostics become misleading [42]. It also breaks the covariate shift assumption that underpins a substantial part of the ML literature on counteracting distribution shift [43, 44]. While the forecaster’s dilemma is well established, it has generally remained peripheral in nowcasting discussions up to now.

On the training side, the efect of target-dependent sampling on the training of generative nowcasting models is to our knowledge unexplored. Generative models learn to sample from the conditional distribution represented by their training set. Therefore, the distribution shift is directly built into the model itself.

As a concrete case we study these efects with a difusion model that nowcasts radar reflectivity purely from geostationary satellite observations. While radar input is the gold standard for nowcasting, contiguous coverage is mostly limited to industrialized nations over land [45]. The geostationary satellite, on the other hand, has almost global coverage and provides early detection of convective initiation [46]. Thus, there is considerable recent interest in satellite to radar translation [47–50] and satellite based severe weather nowcasting [17, 51–54]. Our model is trained over Germany on a wet-biased dataset and then evaluated on one wet-biased and one uniformly sampled test set, representative of the deployment distribution. Crucially, by fixing the model the efect of the evaluation distribution gets isolated. Facilitated through the satellite input choice, we repeat this experiment over Italy, which is fully outside of the model’s training domain, seven years after the training period as an independent robustness test.

Finally, these findings are not an outright indictment of wet-biased datasets. They are useful for learning rare events, but distributional consequences need to be acknowledged. Specifically, evaluation has to be performed on the intended deployment distribution. We provide recommendations, and note the relevance to neighboring fields.

## 3 Results

## 3.1 Sampling Practices in Generative Precipitation Nowcasting

Wet-biased training and test sets are common across the surveyed influential and recent generative precipitation nowcasting studies (Table 1). Additional work is reviewed in Supplementary Table 1.

We find that wet-biased training and test datasets are the norm in the literature reviewed on generative nowcasting. This result is consistent across all generative model types [18, 23, 26, 36] and applies both to meteorological and ML domains [20, 22, 33, 34].

Some works do not use the same sampling scheme for training and test sets [18, 19, 36] motivating the three-distribution setup of Section 3.3.

The utilized sampling taxonomy diferentiates between the following sampling categories: ’Importance sampling’ refers to Kahn-style Monte Carlo importance sampling [55] which was first applied to precipitation nowcasting by [18]. ’Deterministic filter’ subsumes fixed inclusion rules such as minimum rainy-pixel fractions [35], mean-rain thresholds [23] and rainy-day subsets [22]. ’Intensitybalanced’ stratifies samples into bins based on precipitation intensity and then samples the bins equally [21]. ’Event-curated’ datasets select samples on a case by case basis often focusing on heavy-rain [34] or extreme [36] events. Specifically, we refer to the SEVIR dataset [39] as event-curated even though its storm events selected from the NOAA/NCEI Storm Events Database are complemented by a stochastic sampling component that is biased towards larger precipitation intensities. SEVIR is a highly influential benchmark dataset, which is used in a large number of nowcasting studies, including [20, 22–27, 29, 37]. This shows how a wet bias can be inherited silently by every downstream paper making use of a benchmark. ’Uniform’ refers to unbiased sampling leading to a deployment consistent dataset. While comparatively rare, this strategy is employed by some works for a subset of their datasets either for both training and testing [26, 29] or only for restricted test scores [19, 54].

Importantly, all surveyed sampling schemes except for [33] and the ’uniform type make use of future observations to select samples and are thus targetdependent protocols.

Table 1: Dataset sampling and calibration evaluation practices in a sample of influential and recent generative precipitation nowcasting studies. Additional work is surveyed in Supplementary Table 1. Sampling implications are discussed in Section 3.1, calibration practices are surveyed in Section 3.2 and their implications discussed in Section 3.3. Deployment recovery indicates if the estimation of the verification scores on the deployment distribution by inverse probability weighting (IPW) is possible (cf. Section 3.2).
<table><tr><td rowspan=1 colspan=7>Study       Model    Training     Test          Deployment Probabilisticsampling     sampling     recovery      calibration</td></tr><tr><td></td><td></td><td></td><td rowspan=1 colspan=1>Deterministic Equivalent</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>DGMR [33]</td><td rowspan=1 colspan=2>filter</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>DiffCast     Diffusion</td><td rowspan=1 colspan=2>Event-</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>[23]</td><td rowspan=1 colspan=2>curated/</td><td rowspan=1 colspan=3>to training</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>determinis-</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>tic filter</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>CasCast     Diffusion</td><td rowspan=1 colspan=2>Event-</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>[22]</td><td rowspan=1 colspan=2>curated/</td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>determinis-</td><td rowspan=1 colspan=2></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>tic filter</td><td rowspan=1 colspan=2></td><td></td></tr><tr><td rowspan=1 colspan=2>LDCast [21] Diffusion</td><td rowspan=1 colspan=2>Intensity-</td><td rowspan=1 colspan=2>Equivalent</td><td rowspan=1 colspan=1>Possible</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>balanced</td><td rowspan=1 colspan=2>to training</td><td rowspan=14 colspan=1>(shifted)Reliability(uniform)MixedMixed        Rank histogram(shifted)Event-       No            Rank histogram</td></tr><tr><td rowspan=1 colspan=2>PDDN [34] Diffusion</td><td rowspan=1 colspan=2>Event-</td><td rowspan=1 colspan=2>Equivalent</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>curated</td><td rowspan=1 colspan=2>to training</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Stormscope</td><td rowspan=1 colspan=1>Diffusion1</td><td rowspan=1 colspan=2>Unclear</td><td rowspan=1 colspan=2>Uniform</td><td rowspan=1 colspan=1>Uniform set</td></tr><tr><td rowspan=1 colspan=1>[54]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>calibration</td><td rowspan=1 colspan=1>available</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>set; strong-</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=2>FlowCast    Flow</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>rain FSS se</td></tr><tr><td rowspan=1 colspan=2>Event-</td><td rowspan=1 colspan=2>Equivalent</td></tr><tr><td rowspan=1 colspan=2>[26]          matching</td><td rowspan=1 colspan=2>curated/</td><td rowspan=1 colspan=2>to training</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>uniform</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=2>FREUD     Flow</td><td rowspan=1 colspan=2>Event-</td><td rowspan=1 colspan=2>Equivalent</td></tr><tr><td rowspan=1 colspan=2>[29]          matching</td><td rowspan=1 colspan=2>curated /</td><td rowspan=1 colspan=2>to training</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>uniform</td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>GPTCast    Autore-</td><td rowspan=1 colspan=2>Deterministic</td><td></td><td></td></tr><tr><td rowspan=1 colspan=7>[36]          gressive  filter          curated                       (shifted)</td></tr></table>

## 3.2 The Forecaster’s Dilemma and Distribution Shift

Selecting test cases using future observations changes which predictive distribution is rewarded by evaluation.

As established in the previous section, evaluating generative nowcasting models on wet-biased test datasets is a common practice. The wet bias is conventionally a product of target-dependent sampling $\mathrm { i . e . }$ , the evaluation is conditioned on ground truth observations, $y ,$ because samples are selected retrospectively based on precipitation observed during the forecast period. However, forecasts should be evaluated conditionally only on information x that is available at forecast time, not outcomes, y [41]. This failure mode is often termed the forecaster’s dilemma [40]: as forecasters are overwhelmingly judged on their performance on extreme events, it is tempting to restrict the evaluation solely to cases where extremes were indeed observed. Under these conditions it becomes beneficial to strongly over-predict extremes while skillful and calibrated forecasts are unjustly discredited whenever predictability is imperfect.

More precisely, a strictly proper scoring rule is defined as a verification score that is maximized by a forecast probability density $p ( y \mid x )$ if and only ${ \mathrm { i f ~ } } p ( y \mid x )$ reflects the forecaster’s true beliefs [56]. Strictly proper scoring rules cannot be hedged or gamed, meaning by reporting a forecast probability density $p ^ { * } ( y \mid x )$ deviating from ones true beliefs $p ( y \mid x )$ about future events one cannot achieve a better score [57]. Gneiting and Ranjan [41] showed that restricting the test set by outcome is equivalent (up to scaling) to weighting a (strictly) proper score by some weight $w ( y )$ , which depends on the sampling strategy. The expected value of the score is then optimized by the shifted density

$$
p ^ { * } ( y \vert x ) \propto w ( y ) p ( y \vert x )\tag{1}
$$

and not by the true predictive density $p ( y \mid x )$ . This constitutes a viable hedging strategy and thus renders strictly proper scoring rules improper. Crucially, $p ^ { * } ( y \mid x )$ has exactly the mathematical form of the wet-biased training distribution that is simultaneously used to train the model (cf. Section 3.3). This problem can easily stay hidden as target-dependent sampling breaks calibration diagnostics like rank histograms as well [42].

In ML terminology, evaluating generative nowcasting models on wet-biased test datasets, which introduces a mismatch between the test and deployment distribution, is a special case of distribution shift. However, the literature on standard ML distribution shift deals with covariate shift: the distribution of inputs, $p ( x )$ , difers between training and testing or deployment, while the conditional distribution of the target given the inputs, $p ( y \mid x )$ , remains unchanged [43, 44]. This enables the study of post-hoc calibration methods, though they can fail in practice [58, 59]. Since most nowcasting dataset sampling schemes rely on target-dependent sampling, the conditional itself changes and the covariate shift assumption is violated. Post-processing on a wet-biased validation set therefore cannot restore deployment calibration even in principle.

Target-dependent sampling is also directly tied to the evaluation practices in the field. Most works only report raw pixel-based scores on wet-biased test sets. It is in theory possible to recover estimates of the verification scores on the actual deployment distribution by inverse probability weighting (IPW) with Horvitz-Thompson totals and H´ajek-type ratios [60]. To the best of our knowledge this has only been utilized by DGMR [18] so far. Furthermore, IPW requires every candidate sample to have a nonzero chance of inclusion in the test set, which deterministic filter and event-curation approaches do not fulfill and thus excludes this possibility for a large set of works. In particular, it is not possible to apply IPW to the widespread SEVIR dataset [39] without major adjustments. A small subset of works report some pixel-based verification scores for a uniform sampling protocol [19, 26, 29], which is faithful to a deployment scenario. Nevertheless, they often are presented as secondary. For example [19, 29] do not present their uniform protocol scores in the main text and do not directly contrast them with the headline wet-biased scores. Therefore, potential misrepresentations of scores with respect to a deployment consistent evaluation that is predicted by the theory above, is not discussed.

Moreover, calibration diagnostics are reported only rarely: Table 1 contains all works that we are aware of that present any calibration measures, like rank histograms and reliability diagrams, at all. Among these, every published rank histogram is computed on a wet-biased test set. Deployment consistent calibration evidence exists only on the reliability side: DGMR [18] reports IPW reliability diagrams, and Stormscope [54] evaluates reliability on a uniformly sampled set. Notably, DGMR explicitly states that its rank histogram implementation cannot incorporate the importance weights [18]. Consequently, even in the only work that corrects its verification scores, the rank histogram remains tied to the shifted test distribution. Critically, to the best of our knowledge there exists no controlled calibration comparison of a single generative nowcasting model evaluated on both a wet-biased and a deployment consistent test set. Closing this gap is a central contribution of this work.

## 3.3 Target-Dependent Sampling Shifts the Learning Objective

Target-dependent training selection changes the conditional distribution that an ideal generative model learns.

Let $\overset { \sim } { x } \in \mathbb { R } ^ { C \times T _ { x } \times H \times W }$ denote a sequence of past observations serving as model input, where C is the number of input channels, $T _ { x }$ is the number of input time steps, and H and $W$ are the height and width dimensions of the spatial grid dimensions respectively. The C channels may contain any combination of fixedgrid data sources, such as radar composites, geostationary satellite imagery, or lightning observations. Let $\boldsymbol { y } \in \mathbb { R } ^ { T _ { y } \times H \times W }$ denote the corresponding future radar target sequence, consisting of $T _ { y }$ forecast time steps on the same spatial grid. We treat both as realizations of the multivariate random variables X and Y with joint climatological density $p _ { \mathrm { c l i m } } ( x , y )$ . This is our choice for the deployment distribution since it corresponds to a protocol assuming continuous operation. Though, this is not the only viable choice. The crucial point is that the deployment distribution can only be conditioned on information available at forecast time and cannot be conditioned on future observations. Hence, the commonly used target-dependent test datasets are ruled out. Therefore, an ideal generative nowcasting model (unlimited data, model capacity, and compute, with an optimizer that attains the true minimum) samples from the conditional density $p _ { \mathrm { c l i m } } ( y \mid x )$

(a) Three-distribution setup  
![](images/2885bf4e81d563e241b699bb2ba27a5ade6db7ebd540bcfae21b4d0f35906689.jpg)

![](images/c376293b80ae43ac19fb358dbb4be632388979c35fea00b70b1e8f91e6b38a92.jpg)

(c) Acceptance after selection  
![](images/0aa3fa49e52efd23fb3c4f38d83a42bd7fc7ff83ee9afb542346e432a6ec3988.jpg)

![](images/1ed09a99038d1a7ce364d5cdca4e3d75e8b6f015acbdbad2d6898673f45d3e2c.jpg)  
Figure 1: Importance sampling creates distinct deployment, training, and test distributions and shifts the empirical training data. a, The climatological deployment distribution $p _ { \mathrm { c l i m } } ( x , y )$ is shifted into training and test distributions by their respective target-dependent acceptance probabilities, $a _ { \mathrm { t r a i n } } ( y )$ and $a _ { \mathrm { t e s t } } ( y )$ b, Distribution of the acceptance probability $a ( y )$ over all candidate crops from the 2015–2018 training period. c, Distribution of the same acceptance probability after sampling, evaluated over the crops retained for model training. Histograms in b and c show the percentage of crops per bin on a logarithmic scale. d, Normalized pooled radar-reflectivity distributions for all pixels in the complete training period and for the importance sampled crops actually used to train the model.

Dataset sampling is formalized by a binary selection variable $S \in \{ 0 , 1 \}$ with acceptance probability $a ( y ) \in [ 0 , 1 ] ;$

$$
S \mid X = x , Y = y \sim \operatorname { B e r n o u l l i } ( a ( y ) ) ,\tag{2}
$$

where a candidate sample is included in the dataset if and only if $S = 1$ . Although we borrow the term acceptance probability from Kahn-style importance sampling [18, 55], Eq. (2) in principle applies to every sampling category of Section 3.1 with a quantifiable inclusion rule. For example, deterministic filters correspond to the special case $a ( y ) \in \{ 0 , 1 \}$ . Critically, a depends on the target y, reflecting that all these schemes are target-dependent selection protocols. For some sampling schemes a additionally depends on $x .$ . We omit this here as the central point is the dependence on $y .$ . Since training and test sets are sometimes constructed with diferent sampling schemes (cf. Section 3.1), the pipeline consists of three distinct distributions: the training distribution $p _ { \mathrm { t r a i n } } ( x , y ) = p _ { \mathrm { c l i m } } ( x , y \vert S _ { \mathrm { t r a i n } } = 1 )$ with acceptance probability $a _ { \mathrm { t r a i n } } ( y )$ , the test distribution $p _ { \mathrm { t e s t } } ( x , y ) = p _ { \mathrm { c l i m } } ( x , y | S _ { \mathrm { t e s t } } = 1 )$ with acceptance probability $a _ { \mathrm { t e s t } } ( y )$ , and the deployment distribution $p _ { \mathrm { c l i m } } ( x , y )$ itself.

Applying Bayes’ theorem to the conditional density under acceptance yields, for a generic acceptance probability $a _ { \mathrm { t r a i n } } ( y )$ 2

$$
p _ { \mathrm { t r a i n } } ( y | x ) = \frac { \mathbb { P } \left( S _ { \mathrm { t r a i n } } = 1 | X = x , Y = y \right) p _ { \mathrm { c l i m } } ( y | x ) } { \mathbb { P } \left( S _ { \mathrm { t r a i n } } = 1 | X = x \right) } = \frac { a _ { \mathrm { t r a i n } } ( y ) p _ { \mathrm { c l i m } } ( y | x ) } { Z _ { \mathrm { t r a i n } } ( x ) } ,\tag{3}
$$

where $\mathbb { P }$ denotes probability under the climatological distribution and the specified sampling mechanism. The normalizing factor

$$
Z _ { \mathrm { t r a i n } } ( x ) = \mathbb { E } _ { Y \sim p _ { \mathrm { c l i m } } ( \cdot \mid x ) } \big [ a _ { \mathrm { t r a i n } } ( Y ) \big ]\tag{4}
$$

is the expected training acceptance probability for a given input $x ,$ with E denoting expectation over target sequences $Y$ drawn from $p _ { \mathrm { c l i m } } ( y \mid x )$ The same derivation applies to the test distribution using its corresponding acceptance probability. Because $a ( y )$ depends on $y ,$ , the conditional density itself changes. This distinguishes target-dependent sampling from covariate shift, for which $a ( y )$ would depend on $x$ only and $\operatorname { E q }$ . (3) would reduce to $p _ { \mathrm { t r a i n } } ( y \vert x ) =$ $p _ { \mathrm { c l i m } } ( y \mid x )$ . The sampling additionally shifts the input marginal, $p _ { \mathrm { t r a i n } } ( x )$ ∝ $Z _ { \mathrm { t r a i n } } ( x ) p _ { \mathrm { c l i m } } ( x )$ , leading to an over-representation of inputs, $x ,$ that precede rain. However, it is the shifted conditional that is learned by the model and that no covariate-shift correction can address.

Generative models learn to sample from the conditional distribution represented by their training set. An ideal generative model trained on a wet-biased dataset therefore reproduces $p _ { \mathrm { t r a i n } } ( y | x )$ of $\operatorname { E q } .$ . (3) rather than $p _ { \mathrm { c l i m } } ( y \mid x ) ;$ : the distribution shift is built directly into the model. Comparing with Equation (1) from Section 3.2, Eq. (3) is exactly the hedging density $p ^ { * } ( y \vert x ) \propto w ( y ) p ( y \vert x )$ of Gneiting and Ranjan [41] with weight $w ( y ) \ : = \ : a _ { \mathrm { t r a i n } } ( y )$ . An ideal model trained on the wet-biased training set thus represents the optimal hedge against evaluation on a test set constructed with the same acceptance probability. The test bias perfectly conceals the training bias when $a _ { \mathrm { t e s t } } ( y ) = a _ { \mathrm { t r a i n } } ( y )$ . Consequently, a substantial number of works train the optimal cheater for their own improper benchmark and would deploy them on a distribution where the hedge is detrimental.

For score-based generative models the shift takes a particularly transparent form. Taking the logarithm and the gradient with respect to y of $\operatorname { E q }$ . (3), the x-only normalizer $Z _ { \mathrm { t r a i n } } ( x )$ drops out and the learned score decomposes as

$$
\nabla _ { y } \log p _ { \mathrm { t r a i n } } ( y | x ) = \nabla _ { y } \log p _ { \mathrm { c l i m } } ( y | x ) + \nabla _ { y } \log a _ { \mathrm { t r a i n } } ( y ) .\tag{5}
$$

An ideal score-based model hence learns the deployment score augmented by a deterministic drift up the gradient of the log acceptance probability. The wet bias is directly part of the training objective itself. Equation (5) is exact at the data level. At positive difusion noise levels, the statement holds in mollified form i.e. the shift is convolved with the noise kernel.

The structure of the drift predicts where the wet bias should primarily materialize. For the Kahn-style importance sampling [55] used in this work and in [18, 19] the acceptance probability for a sample y is defined as

$$
a ( y ) = \operatorname* { m i n } \left\{ 1 , q _ { \operatorname* { m i n } } + \frac { m } { D } \sum _ { j = 1 } ^ { D } \left[ 1 - \exp \left( - \frac { y _ { j } } { s } \right) \right] \right\} .\tag{6}
$$

Here, $D = T _ { y } H W$ is the total number of pixel values across the target sequence, and $y _ { j }$ denotes the rain rate at its $j \mathrm { t h }$ entry. The parameter $q _ { \mathrm { m i n } }$ sets the acceptance probability for an entirely dry sequence, m controls the increase in acceptance probability with precipitation, and s is a rain-rate scale controlling how quickly each pixel’s contribution saturates. The specific parameter values chosen in this work are listed in Supplementary Table 2. The per-pixel derivative of $a ( y )$ decays exponentially with intensity and vanishes entirely once the acceptance probability is clipped at one,

$$
\partial a / \partial y _ { j } \propto e ^ { - y _ { j } / s } .\tag{7}
$$

The drift therefore pushes hardest in the transition from a dry to light rain regime and barely at all at the extremes. This implies a stronger over-prediction of light rain, while high precipitation intensities are reinforced only marginally in comparison. We examine whether the observed forecast biases are consistent with this interpretation in Section 3.4.

Figure 1 illustrates the three distributions and the empirical sampling shift. Acceptance probabilities accumulate at the clipping limit, while rainy targets are enriched more strongly at light to moderate reflectivities than in the extreme right tail.

In addition, Eq. (3) is invertible. Thus, the deployment distribution can be recovered by reweighting with $1 / a ( y )$ and renormalizing. This corresponds to the IPW verification presented in Section 3.2 but one can also utilize IPW alternatively at the training or post-processing stages of the pipeline (Supplementary Section 7). As before, this requires $a ( y ) > 0$ for all candidate samples.

Furthermore, the discussion above holds in the idealized limit of unlimited data, model capacity, and compute, with an optimizer that attains the true minimum. In practice, rare extremes provide relatively few training examples, and a budget-constrained model may favor more frequent, lower-intensity precipitation patterns, leading to under-prediction of the strongest events even under wet-biased sampling and thus deviate from the predicted shift. The theory therefore establishes that the learning objective is misspecified. This does not mean that any trained model necessarily exhibits the full predicted bias, and it has to be demonstrated experimentally.

The limited and scattered published evidence compiled in Table 1 is consistent with these predictions. All published rank histograms are computed on wet-biased test sets and appear skewed toward apparent under-prediction [18, 29, 36], while the only two deployment consistent calibration diagnostics (the IPW reliability diagrams of DGMR [18] and the uniform set reliability diagrams of Stormscope [54]) both indicate over-prediction. Though, the sampling scheme for the training set of Stormscope is not reported. DGMR presents both diagnostics side by side, an IPW diagram indicating over-prediction next to an unweighted rank histogram on the wet-biased test set suggesting mild under-prediction, but they do not comment on the apparent contradiction. It is exactly the sign flip in calibration that is predicted here. The nearly flat rank histogram of LDCast [21] is no exception. Its training and test sets share the same intensity-balanced shift, and by Eq. (3) a well-trained shifted model evaluated on its own shift should appear approximately calibrated. However, these pieces of evidence are uncontrolled. Models, sampling schemes, lead times, and diagnostics all difer across studies, and directions are judged visually from published figures, as none of these works report summary statistics of their histograms. In the following section we perform a comprehensive controlled comparison.

## 3.4 Wet Bias Leads to Opposite Conclusions About the Same Model

We test experimentally whether target-dependent test sampling conceals forecast bias. Changing only the test-set sampling reverses the calibration diagnosis of the same model: the wet-biased set suggests under-prediction, while the uniform set establishes over-prediction.

We train an EDM-style difusion model [61] to nowcast radar reflectivity solely from geostationary satellite imagery over Germany with a wet-biased training dataset. The model takes three satellite frames at 15 min intervals (t − 30, t − 15, and t minutes) as input and predicts radar reflectivity at t, t + 15, and t + 30 minutes. It is evaluated with 20-member ensembles on two test sets from the same held-out period. One test set is uniformly sampled and thus representative of the deployment distribution, while the other set is constructed with the same target-dependent Kahn-style importance sampling as the training set and is therefore wet-biased. Crucially, by fixing the model the efect of the evaluation distribution gets isolated as it is the only varied parameter. The identical frozen model is also evaluated with the same set up over Italy, which lies fully outside of the training domain, with a seven-year time gap as an independent robustness test.

Table 2 presents pixel-based and categorical verification scores of the model for both datasets and both domains together with the IPW (Horvitz-Thompson / H´ajek) estimates from the importance sampled sets. Complementary scores are listed in Supplementary Table 7, and Supplementary Fig. 5 compares the radially averaged power spectra of forecasts and observations on both test sets. Square brackets indicate the 95 % confidence interval (CI). Some scores differ statistically significantly between the two German sets: CSI at 10 dBZ nearly doubles on the importance sampled set (0.198 [0.182, 0.213] vs. 0.363 [0.350, 0.376]), while the pooled CRPS is ∼ 11 % better (12.26 [12.00, 12.53] vs. 10.95 [10.70, 11.20]). On top of the score inflation is the fact that the diagnosis changes qualitatively. On the importance sampled set the frequency bias is below 1 at every threshold (0.69/0.25/0.23 at 10/30/38 dBZ). This implies a coherently under-predictive model. On the uniform set the bias is 1.54 at 10 dBZ but 0.75/0.69 at 30/38 dBZ. Here over-prediction of light rain coexisting with under-prediction of moderate and strong rain. Therefore, the wet-biased set does not only change the score values, its directional conclusions are selfconsistent but wrong.

The rank histograms exhibited in Fig. 2 also demonstrate a sign flip in the calibration assessment of the model between the two test sets. The rank histogram skews to the right on the importance sampled set, indicating that the observations are above the ensemble too often, which implies an under-predictive model. However, on the uniform set the rank histogram skews to the left, signifying the observations are below the ensemble too often, which signals an overpredictive model. This result holds not just visually, but the mean-rank bias does indeed flip its sign statistically significantly between the uniform (−0.065 [−0.079, −0.051]) and the importance sampled (+0.260 [+0.224, +0.292]) test sets. Evaluating the same model one arrives at opposite calibration conclusions between the importance sampled and the uniform test set, which require opposite interventions (add rain vs. remove rain).

The sign flip demonstrates that target-dependent evaluation can conceal an existing forecast bias. This behavior is consistent with the two theoretical mechanisms discussed previously: (i) Rank histograms conditioned on observations can, misleadingly, skew towards under-prediction even for calibrated models [42] (cf. Section 3.2). (ii) The shifted training dataset predicts a real wet bias (cf. Section 3.3). On the importance sampled test set (i) potentially masks the true bias of (ii), which is only observed on the uniform test set. As the model is kept fixed, the experiment isolates the efect of test-set selection, but it does not establish how much of the model’s light-rain over-prediction originates from training selection.

Table 2: Verification of the same nowcasting model on uniform and importance sampled (wet-biased) test sets over Germany and Italy. The IPW column reestimates performance under the uniform sampling distribution from the importance sampled set using Horvitz-Thompson totals and H´ajek-type ratios. Values in brackets are 95% confidence intervals from a 10,000-replicate block bootstrap with seven-day blocks. Categorical scores and fractions skill score (FSS) are member averaged and aggregated over all lead times. Italian thresholds are the Marshall-Palmer equivalents of the German reflectivity thresholds (0.10, 2.61, and 9.55 mm $\mathrm { h } ^ { - 1 }$ , respectively). Pooled CRPS is reported in dBZ for Germany and mm $\mathrm { h } ^ { - 1 }$ for Italy and is therefore not comparable across domains.
<table><tr><td colspan="2">Metric Threshold</td><td colspan="2">Uniform test set</td><td colspan="2">test set</td><td colspan="2">Importance sampled IPW estimate from importance set</td></tr><tr><td colspan="2">Germany</td><td colspan="6"></td></tr><tr><td rowspan="3">Bias</td><td>10 dBZ</td><td></td><td>1.541 [1.411, 1.705]</td><td></td><td>0.688 [0.666, 0.715]</td><td></td><td>1.540 [1.416, 1.697]</td></tr><tr><td>30 dBZ</td><td></td><td>0.747 [0.643, 0.894]</td><td></td><td>0.245 [0.209, 0.290]</td><td></td><td>0.735 [0.635, 0.877]</td></tr><tr><td>38 dBZ</td><td></td><td>0.687 [0.580, 0.856]</td><td></td><td>0.233 [0.192, 0.287]</td><td></td><td>0.667 [0.568, 0.820]</td></tr><tr><td rowspan="3">CSI</td><td>10 dBZ</td><td></td><td>0.198 [0.182, 0.213]</td><td></td><td>0.363 [0.350, 0.376]</td><td></td><td>0.198 [0.182, 0.213]</td></tr><tr><td>30 dBZ</td><td></td><td>0.036 [0.028, 0.043]</td><td></td><td>0.039 [0.032, 0.047]</td><td></td><td>0.035 [0.028, 0.043]</td></tr><tr><td>38 dBZ</td><td></td><td>0.017 [0.013, 0.022]</td><td></td><td>0.015 [0.011, 0.019]</td><td></td><td>0.017 [0.012, 0.021]</td></tr><tr><td rowspan="3">FSS128</td><td>10 dBZ</td><td></td><td>0.656 [0.628, 0.681]</td><td></td><td>0.789 [0.779, 0.799]</td><td></td><td>0.656 [0.630, 0.679]</td></tr><tr><td>30 dBZ</td><td></td><td>0.273 [0.231, 0.321]</td><td></td><td>0.237 [0.196, 0.290]</td><td></td><td>0.267 [0.220, 0.322]</td></tr><tr><td>38 dBZ</td><td></td><td>0.247 [0.188, 0.300]</td><td></td><td>0.193 [0.143, 0.251]</td><td></td><td>0.240 [0.178, 0.301]</td></tr><tr><td>Pooled CRPS</td><td></td><td>12.26 [12.00, 12.53]</td><td></td><td></td><td>10.95 [10.70, 11.20]</td><td></td><td>12.26 [11.98, 12.55]</td></tr><tr><td colspan="2">Italy</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">Bias</td><td>10 dBZ</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>30 dBZ</td><td></td><td>1.453 [1.364, 1.574]</td><td></td><td>0.675 [0.657, 0.692] 0.292 [0.262, 0.326]</td><td></td><td>1.447 [1.355, 1.561]</td></tr><tr><td>38 dBZ</td><td></td><td>0.695 [0.635, 0.765] 0.881 [0.766, 1.016]</td><td></td><td>0.360 [0.305, 0.421]</td><td></td><td>0.682 [0.625, 0.749] 0.848 [0.735, 0.977]</td></tr><tr><td rowspan="3">CSI</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>10 dBZ 30 dBZ</td><td></td><td>0.190 [0.176, 0.201]</td><td></td><td>0.356 [0.350, 0.363] 0.065 [0.059, 0.072]</td><td></td><td>0.191 [0.177, 0.203] 0.054 [0.048, 0.059]</td></tr><tr><td>38 dBZ</td><td></td><td>0.052 [0.047, 0.058] 0.020 [0.017, 0.024]</td><td></td><td>0.022 [0.018, 0.025]</td><td></td><td>0.020 [0.017, 0.023]</td></tr><tr><td rowspan="3">FSS128</td><td>10 dBZ</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>30 dBZ</td><td></td><td>0.670 [0.646, 0.691] 0.332 [0.296, 0.369]</td><td></td><td>0.807 [0.798, 0.816] 0.313 [0.278, 0.348]</td><td></td><td>0.671 [0.646, 0.694] 0.337 [0.301, 0.372]</td></tr><tr><td>38 dBZ</td><td></td><td>0.269 [0.221, 0.318]</td><td></td><td>0.245 [0.198, 0.295]</td><td></td><td>0.271 [0.226, 0.315]</td></tr><tr><td colspan="2">Pooled CRPS</td><td></td><td>0.074 [0.060, 0.088]</td><td></td><td>0.788 [0.710, 0.864]</td><td></td><td>0.075 [0.061, 0.089]</td></tr></table>

(d) Italy  
![](images/97de1741023610941b2556857fbd65e482ec21e26c4cf684dda31d9fb9cae869.jpg)

![](images/c6749b605897f25ea56210396c8af0088ddb4d07045dee2a5d3eeb57fe80fc4d.jpg)

![](images/67d3437de95da6fb408219179490c4d931e6c0148d6542fe3fbe9d843f3cb862.jpg)

![](images/d33fd7a772d0a1053facc58d85527ccab237f4df855d3153f686b06005f5df5f.jpg)  
Figure 2: Rank histograms for the same 20-member forecast ensemble evaluated on uniform and importance sampled (wet-biased) test sets over Germany and Italy. Error bars and the reported mean-rank-bias intervals are 95 % confidence intervals (CI) from a 10,000-replicate block bootstrap with seven-day blocks. The signed normalized mean-rank bias ranges from −1 (all observations below the ensemble) to +1 (all observations above the ensemble), with zero indicating a centered histogram. The dashed line denotes the relative frequency expected under perfect calibration. Histograms aggregate all lead times.

The reliability diagrams displayed in Fig. 3 are fully consistent with this analysis. Changing the test set sampling protocol flips the calibration result at the 10 dBZ threshold, whereas the higher thresholds of 30 dBZ and 38 dBZ remain comparatively close to perfect reliability over the supported bins.

The model’s out of domain generalization works remarkably well. The CIs of most scores of the German and Italian domains presented in Table 2 intersect. Notably, the model even performs statistically significantly better on the Italian domain for the 30 dBZ threshold. The calibration diagnostics of Figs. 2 and 3 are in agreement as well. Crucially, the calibration sign flip observed over Germany is fully replicated over Italy.

Furthermore, the IPW scores presented in Table 2, which re-estimate performance under the uniform sampling distribution from the importance sampled set, fully agree with the uniform scores within error bars for both domains. The same is true for the IPW estimates of the reliability curves shown in Fig. 3. This confirms that the mismatch between test sets is purely a sampling artifact, which can indeed be corrected through IPW under the condition of known, non zero acceptance probabilities.

Lastly, the observed over-prediction of light rain coexisting with underprediction of moderate and strong rain indicates an imperfect model that behaves in a consistent manner with the interpretation from the previous section that the shift induced by the Kahn-style importance sampling is stronger at the transition from a dry to light rain regime than at the right tail of the distribution.

## 3.5 How Wet-Biased Evaluation Steers Development Astray

Increasing classifier-free guidance improves several forecast scores while amplifying over-prediction on the deployment consistent test set.

In the previous section we showed that on the importance sampled test set the model is diagnosed as under-predictive at all thresholds. This erroneous calibration assessment is a plausible motivation to indiscriminately increase the predicted rainfall amounts. We examine classifier-free guidance (CFG) [62] as an intervention to achieve this in our experiments. Originally introduced as a sample-sharpening mechanism for image generation, CFG is increasingly used in generative precipitation nowcasting [22, 25, 29] as it strengthens the influence of the conditioning information during sampling.

We evaluate guidance scales $s \in \{ 1 , 2 , 4 \}$ on the deployment consistent uniform German test set. Table 3 summarizes pixel-based scores; complementary scores are listed in Supplementary Table 8. CSI at 38 dBZ doubles from $s = 1$ to $s = 2 ~ ( 0 . 0 1 7 ~ [ 0 . 0 1 3 , 0 . 0 2 2 ] ~ \mathrm { v s . } ~ 0 . 0 3 5 ~ [ 0 . 0 2 6 , 0 . 0 4 3 ] )$ , but levels of for $s = 4$

![](images/09b8a0c0fde06179d8b500f382ffe4ae4a1d9ead7251b679b30320cc0dc2eb34.jpg)  
(d) 0.10 mm h−<sup>1</sup>

![](images/1f3aa0b751102e96c5679dd4fb7c312a8857320c945676218ef19ddbd4d82a97.jpg)

![](images/7323d403dee601cddc4924d042cc3e5f3a376d945ab661bd021f5e4779db802b.jpg)

(e) 2.61 mm h−<sup>1</sup>  
![](images/e4e44adc1586216358329a259d89e1434d553742357b12cdfde031da9ca669f0.jpg)

(f) 9.55 mm h−<sup>1</sup>  
![](images/2ae9061bc641bab71ddc44e2765f04b1d9c0532247e426b9900de7af3a5fc6f2.jpg)  
Figure 3: Reliability diagrams over Germany (top row) and Italy (bottom row) for threshold-matched exceedance levels. Columns show 10, 30, and 38 dBZ over Germany and their Marshall-Palmer equivalents of 0.10, 2.61, and $9 . 5 5 \mathrm { m m h ^ { - 1 } }$ over Italy. The same model is evaluated on the uniform test set, on the raw importance sampled (wet-biased) test set, and using an IPW (Horvitz-Thompson/H´ajek) estimate from the importance sampled set. Shading denotes 95% confidence intervals from a 10,000-replicate block bootstrap with sevenday blocks. Most intervals are narrower than the markers and are consequently obscured. An interval is omitted where fewer than ten seven-day blocks contribute forecasts to the corresponding probability bin. Marker area follows the logarithm of the corresponding refinement-bin count using one common scale across all panels and estimates. The diagonal denotes perfect reliability, and points aggregate all lead times.

![](images/056a9607dd0b5b6de636b7ab923e2b3fe22c7877c07d83dbb6657f33c0d21271.jpg)

(a) Guidance scale 1  
![](images/12fba7d4acfcec84061736b42489f2769c8494270f3b877dde66fd6b503cd1a7.jpg)

![](images/2997168de36d299bec700caab87566867cd9b08739421919ddab63b6ffdd97ab.jpg)  
(d) Guidance scale 2

(c) Guidance scale 2  
![](images/5f976f235cd97fa062135ce299d2788bfab0123b782b583b9de24b3a1a50d045.jpg)  
(e) Guidance scale 4

![](images/78ce2c1d7bd48cfdb7ac3c88659e4529e97148e855d944fb056152da6f575a93.jpg)

(f) Guidance scale 4  
![](images/ef84e5bf5115ea9376f1766f288555044d62c605733f1c74dd8525aab38ffb15.jpg)  
Figure 4: Calibration of classifier-free guidance scales 1, 2, and 4 on the German uniform test set. Each row shows one guidance scale. The left column contains rank histograms aggregated over all lead times. Error bars and confidence intervals are estimated the same way as in Fig. 2. The right column contains reliability diagrams for exceedance of 10, 30, and 38 dBZ. Confidence intervals and refinement-bin counts are depicted analogously to Fig. 3. The dashed lines denote perfect calibration.

Table 3: Verification of classifier-free guidance scales 1, 2, and 4 on the German uniform test set. Values in brackets are 95% confidence intervals from a 10,000- replicate block bootstrap with seven-day blocks. Categorical scores and fractions skill score (FSS) are member averaged and aggregated over all lead times; CRPS is reported in dBZ. Guidance scale 1 is repeated from Table 2 to make the guidance comparison self-contained.
<table><tr><td>Metric</td><td>Threshold</td><td colspan="2">Guidance scale 1</td><td colspan="2">Guidance scale 2</td><td colspan="2">Guidance scale 4</td></tr><tr><td rowspan="3">Bias</td><td>10 dBZ</td><td></td><td>1.541 [1.411, 1.705]</td><td></td><td>1.694 [1.563, 1.855]</td><td></td><td>2.056 [1.887, 2.268]</td></tr><tr><td>30 dBZ</td><td></td><td>0.747 [0.643, 0.894]</td><td></td><td>1.685 [1.451, 2.020]</td><td></td><td>3.761 [3.233, 4.527]</td></tr><tr><td>38 dBZ</td><td></td><td>0.687 [0.580, 0.856]</td><td></td><td>2.137 [1.804, 2.695]</td><td></td><td>6.467 [5.354, 8.409]</td></tr><tr><td rowspan="3">CSI</td><td>10 dBZ</td><td></td><td>0.198 [0.182, 0.213]</td><td></td><td>0.241 [0.222, 0.259]</td><td></td><td>0.249 [0.227, 0.269]</td></tr><tr><td>30 dBZ</td><td></td><td>0.036 [0.028, 0.043]</td><td></td><td>0.064 [0.051, 0.076]</td><td></td><td>0.067 [0.054, 0.080]</td></tr><tr><td>38 dBZ</td><td></td><td>0.017 [0.013, 0.022]</td><td></td><td>0.035 [0.026, 0.043]</td><td></td><td>0.031 [0.023, 0.039]</td></tr><tr><td rowspan="3">FSS128</td><td>10 dBZ</td><td></td><td>0.656 [0.628, 0.681]</td><td></td><td>0.714 [0.687, 0.738]</td><td></td><td>0.709 [0.679, 0.736]</td></tr><tr><td>30 dBZ</td><td></td><td>0.273 [0.231, 0.321]</td><td></td><td>0.409 [0.345, 0.470]</td><td></td><td>0.393 [0.336, 0.445]</td></tr><tr><td>38 dBZ</td><td></td><td>0.247 [0.188, 0.300]</td><td></td><td>0.400 [0.311, 0.464]</td><td></td><td>0.265 [0.193, 0.326]</td></tr><tr><td>Pooled</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CRPS</td><td></td><td></td><td>12.26 [12.00, 12.53]</td><td></td><td>8.34 [7.99, 8.69]</td><td></td><td>6.80 [6.17, 7.44]</td></tr></table>

Pooled CRPS on the other hand keeps improving with increasing guidance scale (12.26 [12.00, 12.53] vs. 8.34 [7.99, 8.69] vs. 6.80 [6.17, 7.44]). Evaluating solely aggregate scores commonly optimized in nowcasting would indicate that increasing s appears highly beneficial. However, the calibration diagnostics presented in Fig. 4 reveal a substantial cost accompanying these score improvements. Increasing guidance progressively shifts observations toward low ensemble ranks and moves the reliability curves below the diagonal, demonstrating increasingly strong over-prediction. The frequency bias grows as well, especially at 38 dBZ, turning the previous under-prediction for s = 1 (0.687 [0.580, 0.856]) into strong over-prediction at s = 4 (6.467 [5.354, 8.409]). The improvement in pooled CRPS therefore coexists with worsening calibration and cannot, by itself, justify increasing guidance when reliable forecast probabilities are required.

The case study exhibited in Fig. 5 illustrates how the same intensity shift materializes in a light rain case. The case study exhibited in Fig. 6 illustrates this shift in a convective rain case. A German convective and an Italian light rain case together with the corresponding satellite inputs are shown in Supplementary Figs. 8–11.

These findings largely agree with those of [29], who report that increasing CFG shifts forecasts toward stronger precipitation, initially improves most pixel-based scores, reduces ensemble spread, and eventually produces unrealistic intensities. The studies difer, however, in their CRPS response and in the conclusion drawn from this behavior. One possible contributor to the conflicting CRPS response is the estimator. While we evaluate the empirical forecast ensemble, the released evaluation pipeline of [29] fits a Gaussian distribution from the ensemble mean and standard deviation. Because precipitation ensembles are generally non-Gaussian and CFG changes their shape and spread, these two CRPS formulations may result in diferent rankings of the guidance scales. Moreover, contrary to [29] we consider CFG not intrinsically flawed. We argue that this depends on which distribution its scale is selected and for which forecast deficiency it is intended to compensate. The importance sampled test set incorrectly diagnoses our model as uniformly under-predictive and thereby supplies a plausible rationale for increasing guidance. Evaluation on the deployment consistent test set reveals that the same intervention amplifies an existing light-rain over-prediction and progressively destroys calibration. These results support selecting guidance on a deployment consistent validation dataset using both pixel-based scores and calibration diagnostics. They demonstrate a trade-of whose acceptability depends on the intended forecast use.

## 4 Discussion

In this work we investigated the efects of wet-biased training and test sets caused by target-dependent sampling on generative precipitation nowcasting. We showed that target-dependent sampling applied to the training set shifts the learning objective of generative models. For an ideal model, this shift is equivalent to the optimal hedge against evaluation on a test set constructed with the same sampling protocol. These two shifts interact, causing the test bias to conceal the training bias. Agreement with the selected test distribution therefore does not establish calibration under deployment conditions.

This is not an outright indictment of importance sampling or wet-biased datasets. They are useful to reduce variance in budget constrained models as extremes need to be seen often enough during training to be learned. Our satellite to radar difusion model underscores this fact. While it is over-predictive for light rain it still under-predicts strong rain when evaluated on a uniformly sampled dataset, even though its training set over samples it. The failure is leaving the induced distribution shift completely unacknowledged and evaluating the model on the same shifted distribution. As long as the sampling scheme’s acceptance probabilities are known and strictly positive for every candidate sample this can be achieved through IPW, which can be applied to score reweighting (Section 3.2) or in principle also at the training or post-processing stages (Supplementary Section 7).

To avoid inadequate use of wet-biased datasets we recommend the following:

1. The intended deployment distribution should be stated explicitly and any diferences between it and the training and test distribution should be flagged.

2. Verification should be done on deployment consistent test sets. If importance sampling is utilized to save compute, acceptance probabilities should be recorded, sampling schemes that permit acceptance probabilities equal to zero for any candidate sample need to be avoided and reported scores

Germany: light case, 2020-07-24 04:00 UTC; exceedance threshold 10 dBZ s = 1, member 2 s = 2, member 2 Ground truth (+30 min)  
![](images/0e181709705273367e84b6751d9225dddfd0a6a61be92ca2e6b74e762e58bbaf.jpg)  
s = 1 exceedance probability

![](images/292307ed540e966d8b7d5cf4407cffbd903291c73f9eacb769134c344bb96333.jpg)

![](images/51db8328bad804f8126a5cc6a9da29126748f0e80e557bcb18761a2e80e8f824.jpg)  
Exceedance probability

s = 2 exceedance probability  
![](images/fb74c3840f5f99935a387f2bddb77334866fd203720c33d0cae267c63a642e0c.jpg)

![](images/c9f3e3c68857aeb5e901c3a0de23487825e793d650c85208dab59f1c9be7c36c.jpg)

![](images/d2a81aacad8073ca5c39b740d5fc129087f74865f629e906da8ec8f0e9ee81ff.jpg)  
Figure 5: German light-rain case on 24 July 2020, 04:00 UTC and a lead time of 30 minutes. Each panel shows the same $1 2 8 \times 1 2 8 – p i x e l$ crop at 1 km spatial resolution. $\mathbf { a } , \mathbf { b } ,$ The same paired ensemble member at guidance scales $s = 1$ and $s = 2 ; { \mathbf c } ,$ , ground truth. $\mathbf { d } , \mathbf { e } ,$ Exceedance probabilities $p _ { s = 1 }$ and $\scriptstyle p _ { s = 2 } .$ , defined at each pixel as the fraction of the 20 ensemble members reaching or exceeding 10 dBZ at the displayed lead time. f, The pixelwise probability diference $\Delta p =$ $p _ { s = 2 } - p _ { s = 1 }$ . Aggregated over all members, lead times, and pixels in this case, the observed exceedance fraction is 3.67%, compared with forecast fractions of 23.58% at $s = 1$ and $2 6 . 8 7 \%$ at $s = 2$ . The corresponding CSI changes only from 0.044 to 0.047, so the additional 3.29 percentage points of forecast event area yield little categorical benefit while strengthening the already excessive light-rain coverage.

Ground truth (+30 min)  
s = 1, member 5  
![](images/3d39a0e623a4d257b8a56602f7eab1604ac6101275bffee8e66ebbb20bf86106.jpg)

s = 2, member 5  
![](images/0589a9773ee9ce8d1ec86b9d2867c9bb501eed177fa40235c8731ba30432cbbc.jpg)

![](images/38de1cdf56ae4a6bd0926375a51e72a4a5998ad41cdebd45d7de3585e04cb7b9.jpg)

$$
\mathrm { h } ^ { - 1 }
$$

s = 1 exceedance probability  
Exceedance probability  
![](images/6490c4f117faef6a14876b8a47fec74088cb2b851c5b41510c04e0c45f25b54f.jpg)

s = 2 exceedance probability  
![](images/225fbf64649ce9893d6929c6b5253da846cc280bbc4f99874bbf2dcf0e89ecfd.jpg)

![](images/d74b1199d1193b06b4fbed17bed8d62d3a1b36e59f84452cc575ca749ba6e503.jpg)  
Figure 6: Italian convective case on 13 July 2025, 14:00 UTC and a lead time of 30 minutes. Each panel shows the same $1 2 8 \times 1 2 8 – p i x e l$ crop at 1 km spatial resolution. $\mathbf { a } , \mathbf { b }$ , The same paired ensemble member at guidance scales $s = 1$ and $s = 2 ; { \bf c } ,$ ground truth. $\mathbf { d } , \mathbf { e } ,$ Exceedance probabilities $p _ { s = 1 }$ and $p _ { s = 2 } .$ defined at each pixel as the fraction of the 20 ensemble members reaching or exceeding $9 . 5 5 \mathrm { m m h ^ { - 1 } }$ at the displayed lead time. f, The pixelwise probability diference $\Delta p = p _ { s = 2 } - p _ { s = 1 }$ . Aggregated over all members, lead times, and pixels in this case, increasing the guidance scale from $s = 1$ to $s = 2$ raises CSI from 0.057 to 0.168 and moves the forecast exceedance fraction from 0.55% to 2.03%, compared with 1.59% in the observations. At the same time, the median member maximum increases from 94.67 to 322.04 mm $\mathrm { h } ^ { - 1 }$ , illustrating that the improved localization and categorical overlap are accompanied by excessive peak intensities and additional forecast event area.

should be IPW estimates. Alternatively forecast-based stratification can be used to emphasize extremes [40, 41].

3. Calibration measures like rank histograms and reliability diagrams ought to be reported to complement pixel-based scores.

4. Post-processing should happen exclusively on deployment consistent validation datasets.

5. In the longer term: The construction of a community benchmark for nowcasting with deployment consistent test sets analogously to WeatherBench for ML-based medium-range weather forecasting [63, 64].

Several limitations remain: the theory presented only holds in the idealized limit and real models will always face some budget constraints. Extremes can remain under-predicted despite a wet-biased training set. Therefore, experimental confirmation is paramount. We investigated one model family (EDM difusion models) represented by one architecture, on one task (satellite to radar nowcasting), for one sampling scheme (Kahn-style importance sampling). Magnitudes will difer for radar to radar nowcasting, GANs, flow matching and diferent architecture and training algorithms. The mechanism itself, however, applies to generative objectives in general. A uniform test set is only a proxy for deployment. Further operational complications like data latency or quality efects are out of scope of this work.

The mechanism we describe is not specific to nowcasting. It requires only a generative model fitted to a training set whose sample selection depends on the target, and a verification protocol that conditions the same way. This combination is likely to appear when rare events dominate the value of a model. For example, generative precipitation downscaling enriches its training data with rainy scenes in a similar way. Where the efect of this choice has been examined, it is consistent with our derivation [65]. ML climate emulation faces the same tension from the opposite side, as extremes are under sampled by the very simulations that supply the training data [66], while the distribution of interest is a future climate. Our recommendations are mostly transferable: state the intended deployment distribution, retain the acceptance probabilities that separate it from the training and test distributions, and verify on the deployment distribution. Irrespective of the field, a skill claim is only as trustworthy as the distribution it was certified on.

## 5 Methods

## 5.1 Data

This section describes the data sources together with their processing and sampling routines for our satellite to radar nowcasting model for both the German and Italian domains, which are displayed in Fig. 7 together with the cropping scheme. Additional details including software specifications can be found in Supplementary Section 3.

German radar. The target over Germany is the DWD RADOLAN WX composite [67], a quality controlled national radar reflectivity mosaic in dBZ quantized in steps of 0.5 dBZ. It is provided every 5 min on a $9 0 0 \times 1 1 0 0$ pixel grid with 1 km spacing in the DWD polar-stereographic projection (WGS84 ellipsoid, central latitude ${ 9 0 } ^ { \circ } \mathrm { N } ,$ central longitude $1 0 ^ { \circ } \mathrm { E } ,$ true-scale latitude $6 0 ^ { \circ } \mathrm { N } )$ We only use timestamps on the 15 min grid to align the radar data with the lower temporal resolution of the satellite data. Pixels outside the composite coverage are flagged as missing by the product. We use the months May to October of the years 2015 to 2020 and split by calendar year: 2015–2018 for training, 2019 for validation, and 2020 for testing.

Italian radar. The target over Italy is the IT-DPC-SRI surface rainfall intensity product [68] in $\mathrm { m m h ^ { - 1 } }$ . It is quality controlled and derived by the Italian Civil Protection Department from the national radar mosaic. The archive provides the product on a 1200 × 1400 pixel grid with 1 km spacing in a transverse Mercator projection (WGS84 ellipsoid, latitude of origin $4 2 ^ { \circ } \mathrm { N } ,$ central meridian 12.5◦E, scale factor 1, no false easting or northing). Since July 2020 the product is updated every 5 min. We retrieve only the timestamps on the 15 min grid from the continuously updated ArcoDataHub version of the archive. Pixels outside the composite coverage are flagged as missing by the product. We utilize the months May to October of 2025 as the Italian test period and process it analogously to the German data. The model was neither trained nor fine-tuned on Italian data.

Satellite imagery. The input in both domains is Meteosat Second Generation (MSG) SEVIRI Level 1.5 imagery [46] of the $0 ^ { \circ }$ full-disc service with a 15 min repeat cycle from the platforms Meteosat-10 in 2015–2017 and 2025 and Meteosat-11 in 2018–2020. Four of the 12 SEVIRI channels are used as predictors: the broadband 0.4–1.1 µm high-resolution visible (HRV) and the 0.8 µm visible (VIS008) channel as reflectance in percent, and the $1 0 . 8 \mu \mathrm { m }$ infrared (IR 108) and 6.2 µm water-vapor (WV 062) channel as brightness temperature in K. No solar-angle correction is applied. The visible channels remain part of the input at night, where they are close to zero. Each scene is resampled with a nearest-neighbor approach onto the radar grid of the respective domain, so that satellite and radar fields share pixel coordinates.

Cropping. Samples are 128 × 128 pixel (128 km) crops cut from a fixed lattice with a stride of 64 pixels, so that neighboring crops overlap by 50%. Over Germany the lattice covers a window of $5 1 2 \times 8 9 6$ pixels of the full grid with vertices (NW, NE, SE, SW) at $( 5 4 . 6 0 ^ { \circ } \mathrm { N } , 5 . 8 5 ^ { \circ } \mathrm { E } )$ $( 5 4 . 6 3 ^ { \circ } \mathrm { N } , 1 3 . 5 6 ^ { \circ } \mathrm { E } )$ 2 $( 4 6 . 9 8 ^ { \circ } \mathrm { N } , 1 2 . 8 8 ^ { \circ } \mathrm { E } )$ , and $( 4 6 . 9 6 ^ { \circ } \mathrm { N } , 6 . 6 4 ^ { \circ } \mathrm { E } )$ , producing $7 \times 1 3 = 9 1$ crops per timestep. Over Italy it covers 1024×1344 pixels with vertices at $( 4 7 . 6 3 ^ { \circ } \mathrm { N } , 5 . 4 9 ^ { \circ } \mathrm { E } )$

![](images/4bc2afcfb49c3d12d71296f6e2bfdf457d6d64e4fe10d09fa40686ee0713299f.jpg)  
(c) Italy Radar rain rate

![](images/a723ca722124ef7d73fc479523b63099401af8df195aa227c8e9718b4428462c.jpg)

(d) Italy Meteosat HRV  
![](images/cf823bb84fbd81692d31862f73249318be7f9e561e73d9c45ebbff1c4a6d142a.jpg)  
Figure 7: Geolocated input domains and crop construction. $\mathbf { a } , \mathbf { b } ,$ German RADOLAN WX radar reflectivity in dBZ and the corresponding Meteosat-10 SEVIRI high-resolution visible (HRV) channel at 1 June 2016, 11:45 UTC, near the peak of the Simbach flooding; $\mathbf { c } , \mathbf { d } ,$ , Italian SRI radar rain rate in mm $\mathrm { h } ^ { - 1 }$ and the corresponding HRV channel at 16 May 2023, 06:00 UTC, during the Emilia–Romagna flooding. Black outlines in the upper-left corner of each radar panel show one solid 128 × 128-pixel reference crop and its dashed right and lower neighbours.

(47.65◦N, 19.12◦E), $( 3 5 . 6 1 ^ { \circ } \mathrm { N } , 1 7 . 9 8 ^ { \circ } \mathrm { E } )$ , and $( 3 5 . 6 0 ^ { \circ } \mathrm { N } , 6 . 6 9 ^ { \circ } \mathrm { E } )$ , producing 15 × 20 = 300 crops per timestep. The windows were chosen to minimize the number of pixels outside radar coverage.

Sample construction and quality control. Every sample uses three satellite frames at t − 30 min, t − 15 min, and t as input and three radar frames at t, t + 15 min, and t + 30 min as target. A timestamp is globally invalid if the radar file is missing, the satellite data is missing, or any of the twelve SEVIRI channels contains a missing value. Furthermore, a crop at a given timestamp is invalid if more than 360 of its 16,384 radar pixels are missing. This threshold was set to balance data quality against the number of retained crops over Germany. In total, these two rules mask 2.25% of the German crop-timestamps. The same rules are applied unchanged over Italy to keep the two pipelines identical where possible. There, 61.8% of all crop-timestamps are masked. This is due to many crops permanently lying outside the radar coverage. Finally, a six-frame sequence is considered only if all frames are valid for its crop and fall within May to October. The influence on sample count of each quality control mask is listed in Supplementary Table 4. Missing radar values are filled before standardization. Over Germany, missing radar pixels are set to −33 dBZ, just below the lowest product class of −32.5 dBZ, and are thereby treated as no echo. Over Italy, missing rain rates are set to $0 \mathrm { m m h ^ { - 1 } }$ , i.e. they are treated as no rain.

Importance sampling. The German training set and the wet-biased test sets of both domains are constructed with the Kahn-style importance sampling of Section $3 . 3 ,$ which is adopted from DGMR [18]. The acceptance probability $a ( y )$ for a three-frame radar target sample y in mm $\mathrm { h } ^ { - 1 }$ is defined in Eq. (6). The specific parameter values chosen in this work are listed in Supplementary Table 2. Their sensitivity is analyzed in Supplementary ${ \mathrm { F i g . } }$ 1 and Supplementary Table 3. Its efect on the training distribution is shown in Fig. 1. For Germany, the reflectivity target is converted to rain rate using the Marshall-Palmer relation $Z = 2 5 \dot { 6 } R ^ { 1 . 4 \bar { 2 } }$ before evaluating $a ( y )$ . For Italy, the rain rate is used directly. The same relation defines the Italian equivalents of the German verification thresholds (Section 5.3). A candidate is accepted if a uniform random number does not exceed $a ( y )$ , using a fixed seed per split, and its acceptance probability is stored with the sample for IPW. In the German training period 425,713 of 6,221,943 candidates (6.84%) are accepted. The training set is complemented by a uniformly sampled validation set of 2019 that is used for checkpoint selection (Section 5.2).

Uniform test sets and dataset sizes. The deployment consistent test sets are uniform random samples without replacement from all candidate sequences of the test months. They are limited to the same sample count as the corresponding importance sampled set. Both German test sets contain 122,644 sequences from 2020 and both Italian test sets 95,111 sequences from 2025.

Therefore, diferences between the two protocols are not confounded by sample size. The importance sampled test sets use the identical acceptance function as the training set, i.e. $a _ { \mathrm { t e s t } } = a _ { \mathrm { t r a i n } }$ in the notation of Section 3.3. Supplementary Table 5 summarizes all set sizes.

## 5.2 Model

This section specifies the normalization, difusion formulation, the model architecture, the training algorithm, inference algorithm and the classifier-free guidance protocol.

Normalization. We utilize Z-score normalization, i.e all channels are standardized to zero mean and unit variance. Radar reflectivity uses the mean of −3.9248 dBZ and standard deviation of 25.7472 dBZ of all target pixels of the importance sampled German training set. Each satellite channel uses the mean and standard deviation of the training period over the full German grid (Supplementary Table 6). Over Italy, the satellite input is standardized with the German constants, so that the network always receives inputs on the scale it was trained on. Italian rain rates are stored as standardized log(1 + R) for the ground truth only. They are transformed back to mm $\mathrm { h } ^ { - 1 }$ before verification.

Difusion formulation. We follow the EDM formulation of Karras et al. [61]. Let $y$ denote the standardized three-frame radar target and c the satellite conditioning. We wrap the network $F _ { \theta }$ in the preconditioned denoiser

$$
D _ { \theta } ( \tilde { y } ; \sigma , c ) = c _ { \mathrm { s k i p } } ( \sigma ) \tilde { y } + c _ { \mathrm { o u t } } ( \sigma ) F _ { \theta } \left( c _ { \mathrm { i n } } ( \sigma ) \tilde { y } ; c _ { \mathrm { n o i s e } } ( \sigma ) , c \right) .\tag{8}
$$

The preconditioning coeficients are

$$
c _ { \mathrm { s k i p } } ( \sigma ) = \frac { \sigma _ { \mathrm { d a t a } } ^ { 2 } } { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } } ,\tag{9}
$$

$$
c _ { \mathrm { o u t } } ( \sigma ) = \frac { \sigma \sigma _ { \mathrm { d a t a } } } { \sqrt { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } } } ,\tag{10}
$$

$$
c _ { \mathrm { i n } } ( \sigma ) = \frac { 1 } { \sqrt { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } } } ,\tag{11}
$$

$$
c _ { \mathrm { n o i s e } } ( \sigma ) = \frac { 1 } { 4 } \ln \sigma .\tag{12}
$$

Only the noisy target is scaled by ${ \mathit { c } } _ { \mathrm { { i n } } } ,$ , while the conditioning c is passed to the network without this scaling. Training minimizes

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { ( y , c ) , \sigma , n \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I ) } \Big [ \lambda ( \sigma ) \left| \left| D _ { \theta } ( y + n ; \sigma , c ) - y \right| \right| _ { 2 } ^ { 2 } \Big ] ,\tag{13}
$$

with

$$
\lambda ( \sigma ) = \frac { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } } { ( \sigma \sigma _ { \mathrm { d a t a } } ) ^ { 2 } } .\tag{14}
$$

The squared error is averaged over all three target frames and all pixels. One noise level per sample is drawn from ln $\sigma \sim \mathcal N ( P _ { \mathrm { m e a n } } , P _ { \mathrm { s t d } } ^ { 2 } )$ . Supplementary Figure 2 summarizes one training step from the corruption of the target through the conditional denoiser to the loss. All parameters take the default values of [61] except for two changes induced by the unit variance of the standardized target. We set $\sigma _ { \mathrm { { d a t a } } } = 1$ instead of $0 . 5 ^ { } .$ , and $P _ { \mathrm { m e a n } } = - 0 . 5$ instead of −1.2, which is the original noise-level prior shifted by ln 2 so that the ratio of noise level to data scale covered during training is unchanged $( P _ { \mathrm { s t d } } = 1 . 2$ is retained). The score of the learned distribution follows from the denoiser as

$$
\begin{array} { r } { \nabla _ { \tilde { y } } \log p ( \tilde { y } ; \sigma , c ) = \big ( D _ { \theta } ( \tilde { y } ; \sigma , c ) - \tilde { y } \big ) / \sigma ^ { 2 } , } \end{array}\tag{15}
$$

which links the trained model to Eq. (5). The IPW-trained model of Supplementary Section 7 multiplies the per-sample loss by $1 / a ( y )$ , normalized by the mean of this weight over the training set, and is otherwise identical.

Architecture. $F _ { \theta }$ is a conditional two-dimensional U-Net with $8 2 . 3 \times 1 0 ^ { 6 }$ parameters that treats the three target frames as channels (Supplementary Figs. 3 and 4). It operates on four resolution levels from 128 × 128 to $1 6 \times 1 6$ pixels with 128 to 1024 channels, two residual blocks per level, and one multi-head selfattention block in the bottleneck. The noise level enters every residual block through adaptive group normalization driven by a Fourier embedding of $c _ { \mathrm { n o i s e } } .$ The satellite conditioning is processed by a separate encoder that applies two dimensional convolutions to each of the three input frames, adds a sinusoidal encoding of the time of day of each frame, and fuses the frames with a three dimensional convolution into 64 feature maps at full resolution. These features are concatenated with the noisy target at the network input and are additionally injected additively, after average pooling and a linear projection, at every coarser level of the encoder and decoder. Further architectural details are given in Supplementary Section 4.

Training. The model is trained with AdamW with a batch size of 128 for at most 100 epochs with a five epoch linear warm-up followed by cosine decay of the learning rate. We apply gradient clipping, random rotations and flips of the crops, and an exponential moving average (EMA) of the weights. Checkpoint selection is based on the EDM loss of Eq. (13) evaluated with the EMA weights on the uniformly sampled 2019 validation set. Early stopping terminates the training after ten epochs without improvement. The selected checkpoint is epoch 81. The training took about five days on a single NVIDIA HGX A100 80GB 500W GPU. Hyperparameters are listed in Supplementary Table 6.

Inference and ensemble generation. Forecasts are generated with the second-order multistep solver DPM-Solver++(2M) [69] in its data-prediction form with $N = 5 0$ steps on the noise schedule of [61]

$$
\begin{array} { r } { \sigma _ { i } = \big ( \sigma _ { \operatorname* { m a x } } ^ { 1 / \rho } + \frac { i } { N - 1 } ( \sigma _ { \operatorname* { m i n } } ^ { 1 / \rho } - \sigma _ { \operatorname* { m a x } } ^ { 1 / \rho } ) \big ) ^ { \rho } , } \end{array}\tag{16}
$$

for $i = 0 , \ldots , N - 1$ with $\sigma _ { \mathrm { m a x } } = 8 0 , \sigma _ { \mathrm { m i n } } = 0 . 0 0 2$ , and $\rho = 7$ , followed by a final step to $\sigma = 0$ . Each of the 20 ensemble members starts from independent Gaussian noise $\mathcal { N } ( 0 , \sigma _ { \mathrm { m a x } } ^ { 2 } I )$ whose seed is a deterministic function of the sample and member index. Hence, members with the same index start from identical noise across guidance scales and across the baseline and the IPW-trained model. This enables the paired comparisons shown in the case studies. The denoised output is de-standardized to dBZ without any clipping. Over Italy it is subsequently converted to mm $\mathrm { h } ^ { - 1 }$ with the Marshall-Palmer relation.

Classifier-free guidance. During training the satellite conditioning is dropped with probability 0.1, in which case the network denoises the target without any conditioning apart from the noise level (Supplementary Section 4). At inference, classifier-free guidance [62] with scale s combines the conditional and the unconditional denoiser output as

$$
D _ { s } ( \tilde { y } ; \sigma , c ) = D _ { \theta } ( \tilde { y } ; \sigma , \mathcal { O } ) + s \left[ D _ { \theta } ( \tilde { y } ; \sigma , c ) - D _ { \theta } ( \tilde { y } ; \sigma , \mathcal { O } ) \right]\tag{17}
$$

at every solver step, so that $s = 1$ recovers the unguided model with one network evaluation per step and $s \neq 1$ requires two. We evaluate $s \in \{ 1 , 2 , 4 \}$ . All results except the guidance study use $s = 1$

## 5.3 Verification

This section presents our verification protocol, defines all verification metrics and our uncertainty estimation approach.

Protocol. The model is evaluated with 20-member ensembles at lead times of 0, 15, and 30 min on the uniform and the importance sampled test set of each domain. All reported scores aggregate the three lead times by pooling the underlying statistics (contingency counts, sums of squares, or sums of persample scores) before ratios are formed. Lead-time resolved scores are shown in Supplementary Fig. 6. Following DGMR [18], all scores except the fractions skill score (FSS) and the power spectra are computed on the central $6 4 \times 6 4$ pixels of each $1 2 8 \times 1 2 8$ crop to exclude boundary efects, whereas FSS and power spectra use the full crop because the largest neighborhood scale equals the crop size. Thresholds are 10, 30, and 38 dBZ over Germany. Over Italy they are the equivalents 0.10, 2.61, and 9.55 mm $\mathrm { h } ^ { - 1 }$ obtained from the Marshall-Palmer relation $Z = 2 5 6 R ^ { 1 . 4 2 }$ and rounded to the product precision, so that categorical scores are comparable across domains up to the uncertainty of the Z–R relation. CRPS values are reported in the native unit of each domain and are not comparable across domains. We refer to [70] for the definitions of all scores and to Supplementary Section 5 for implementation details. Scores omitted from the main tables, namely POD, FAR, FSS at 8 and 32 pixels, and the pixelwise CRPS, are listed in Supplementary Tables 7 and 8.

Categorical scores. For each threshold, the $2 \times 2$ confusion matrix consists of the counts of true positives (TP), false positives (FP), false negatives (FN), and true negatives (TN), which are pooled over pixels, lead times, and samples. We report the probability of detection (POD), the false alarm ratio (FAR), the frequency bias (Bias), and the critical success index (CSI):

$$
{ \begin{array} { r l } { { \mathrm { P O D } } = { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F N } } } } , } & { { \mathrm { F A R } } = { \frac { \mathrm { F P } } { \mathrm { F P } + { \mathrm { T P } } } } , } \\ { { \mathrm { B i a s } } = { \frac { \mathrm { T P } + { \mathrm { F P } } } { \mathrm { T P } + { \mathrm { F N } } } } , } & { { \mathrm { C S I } } = { \frac { \mathrm { T P } } { \mathrm { T P } + { \mathrm { F P } } + { \mathrm { F N } } } } . } \end{array} }\tag{18}
$$

Furthermore we report the FSS at a neighborhood size of $n \in \{ 8 , 3 2 , 1 2 8 \}$ pixels. It compares the exceedance fractions $P ^ { f }$ and $P ^ { o }$ of forecast and observation obtained with an $n \times n$ moving average with zero padding,

$$
\mathrm { F S S } _ { n } = 1 - \frac { \sum _ { i } \left( P _ { i } ^ { f } - { P _ { i } ^ { o } } \right) ^ { 2 } } { \sum _ { i } { \left( P _ { i } ^ { f } \right) ^ { 2 } } + \sum _ { i } { \left( P _ { i } ^ { o } \right) ^ { 2 } } } ,\tag{19}
$$

where numerator and denominator are accumulated over all pixels, lead times, and samples before the ratio is taken. We compute these scores for every ensemble member separately and then average over members. This measures the skill of a typical member. We refrain from scoring the mean of a precipitation ensemble as it represents an overly smooth field and because averaging in the logarithmic dBZ unit is not equivalent to averaging rain rates.

Probabilistic scores. The continuous ranked probability score (CRPS) is a proper scoring rule for the full predictive distribution. For an ensemble of $N = 2 0$ members $x _ { 1 } , \ldots , x _ { N }$ and the observation $x _ { a }$ at one pixel we use its kernel representation [57]

$$
\mathrm { C R P S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| x _ { i } - x _ { a } \right| - \frac { 1 } { 2 N ^ { 2 } } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } | x _ { i } - x _ { j } | ,\tag{20}
$$

which is the CRPS of the empirical cumulative distribution function of the ensemble. The pixel values are averaged over the pixels of a field, over all samples, and lead times. Because the pixelwise CRPS sufers from the double penalty problem, we additionally report a pooled CRPS similar to DGMR [18]. The observation and every member are first smoothed with a $1 6 \times 1 6$ pixel averagepooling window with a stride of 4 pixels, and the CRPS is then computed on the pooled cells that cover the central domain.

Power spectra. To assess whether the forecasts reproduce the spatial variability of the observations across scales, we compute radially averaged power spectral densities (RAPSD). Each 128 × 128 field is transformed with a twodimensional discrete Fourier transform, the squared modulus is averaged over annuli of constant radial wavenumber, and the resulting one-dimensional spectra are averaged over ensemble members, samples, and lead times separately for forecasts and observations. The spectra of both domains and test sets are compared in Supplementary Fig. 5. Since they are computed in the native unit of each domain, absolute levels are not comparable across domains.

Rank histograms. The rank of the observation among the 20 sorted member values is computed for every pixel of the central domain, every lead time, and every sample leading to 21 bins. Pixels at which the observation and all members are below 10 dBZ or $0 . 1 0 \mathrm { m m h ^ { - 1 } }$ for the German and Italian domain respectively are excluded since their 21 values are all tied at no rain and thus their rank is arbitrary. Remaining values below that threshold are treated as tied and ties are broken uniformly at random. The histograms are summarized by the signed normalized mean-rank bias $b = 2 \bar { r } / M - 1 \in [ - 1 , 1 ]$ , where ¯r is the mean rank on the scale $0 , \ldots , M$ and $M = 2 0$ is the ensemble size, so that $b < 0$ indicates observations predominantly below the ensemble (over-prediction) and $b > 0$ the opposite.

Reliability diagrams. The forecast probability of threshold exceedance is the fraction of members exceeding the threshold at a pixel. Probabilities are binned into the 21 attainable values $k / 2 0$ $k = 0 , \ldots , 2 0$ , and the observed relative frequency of exceedance is computed per bin over the central domain, all lead times, and all samples. The refinement count per bin is indicated by the marker area.

Inverse probability weighting. Scores on the deployment distribution are re-estimated from the importance sampled test set by IPW with the recorded acceptance probabilities. Every sample i receives the weight $w _ { i } = 1 / a ( y _ { i } )$ . Each statistic $S = \textstyle \sum _ { i } S _ { i }$ accumulated over samples is replaced by its Horvitz-Thompson total [60] $\begin{array} { r } { \hat { S } = \sum _ { i } w _ { i } S _ { i } } \end{array}$ , and scores that are ratios of two such statistics, i.e. all categorical scores, the FSS, and the reliability frequencies, become H´ajek-type ratio estimators $\hat { S } _ { 1 } / \hat { S } _ { 2 }$ . Per-sample scores are estimated by the self-normalized weighted mean $\sum _ { i } w _ { i } \mathrm { C R P S } _ { i } / \sum _ { i } w _ { i }$ . Rank histograms are not reweighted.

Uncertainty quantification. Confidence intervals are obtained by a block bootstrap that respects the temporal correlation structure. Samples are assigned to fixed seven-day calendar blocks anchored at a common epoch, which yields 27 blocks per domain and test year. For each of 10,000 replicates, blocks are drawn with replacement, the per-block suficient statistics are summed, and all scores, histograms, and reliability curves are recomputed from the summed statistics. Reported intervals are the 2.5th and 97.5th percentiles of the replicates. Comparisons between models or guidance scales use the same block draws for both runs (paired resampling).

## 5.4 Use of AI-Assisted Tools

The authors used OpenAI Codex and Anthropic Claude Code to assist with software development and with drafting, revising and editing the manuscript. The research questions and scientific ideas originated with the authors. All AI-assisted outputs were reviewed and checked by the authors, who take full responsibility for the code, analyses and manuscript.

## Data availability

The RADOLAN WX radar composites were obtained from the Deutscher Wetterdienst (DWD) upon request. Enquiries regarding access to these historical composites should be directed to DWD’s data service (datenservice@dwd.de). Availability and access conditions are determined by DWD. The MSG SEVIRI Level 1.5 satellite observations are available from the EUMETSAT Data Store (https://data.eumetsat.int/), subject to EUMETSAT’s access and licensing conditions. EUMETSAT has confirmed that these observations may not be redistributed in the processed form used in our machine-learning datasets. Consequently, the prepared paired satellite–radar datasets cannot be redistributed by the authors. The Italian IT-DPC-SRI radar data are publicly available through ArcoDataHub under a CC BY-SA 4.0 license (https://arcodatahub. com/datasets/notebooks/italian-radar-dpc-sri/).

## Code availability

The source code for data processing, dataset sampling, model training, forecast generation and verification for both study domains is provided confidentially to editors and reviewers in the ZIP archive accompanying this submission. Public release of the code on GitHub is planned for publication, subject to institutional clearance; the present archive is provided for editorial assessment and peer review.

## References

[1] Tellman, B. et al. Satellite imaging reveals increased proportion of population exposed to floods. Nature 596, 80–86 (2021).

[2] Koks, E. E. et al. A global multi-hazard risk analysis of road and railway infrastructure assets. Nature Communications 10, 2677 (2019).

[3] Myhre, G. et al. Frequency of extreme precipitation increases extensively with event rareness under global warming. Scientific Reports 9, 16063 (2019).

[4] Pfahl, S., O’Gorman, P. A. & Fischer, E. M. Understanding the regional pattern of projected future changes in extreme precipitation. Nature Climate Change 7, 423–427 (2017).

[5] Majumdar, S. J. et al. Multiscale Forecasting of High-Impact Weather: Current Status and Future Challenges. Bulletin of the American Meteorological Society 102, E635–E659 (2021).

[6] Wang, Y. et al. Guidelines for Nowcasting Techniques (2017).

[7] Sun, J. et al. Use of NWP for Nowcasting Convective Precipitation: Recent Progress and Challenges. Bulletin of the American Meteorological Society 95, 409–426 (2014).

[8] Price, I. et al. Probabilistic weather forecasting with machine learning. Nature 637, 84–90 (2025).

[9] Bouall\`egue, Z. B. et al. The Rise of Data-Driven Weather Forecasting: A First Statistical Assessment of Machine Learning–Based Weather Forecasts in an Operational-Like Context. Bulletin of the American Meteorological Society 105, E864–E883 (2024).

[10] Wilson, J. W., Crook, N. A., Mueller, C. K., Sun, J. & Dixon, M. Nowcasting Thunderstorms: A Status Report. Bulletin of the American Meteorological Society 79, 2079–2100 (1998).

[11] Pulkkinen, S. et al. Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0). Geoscientific Model Development 12, 4185–4219 (2019).

[12] B¨olle, T., Metzl, C. & Yousefnia, K. V. A Minimal Model of the Deep-Convection Life Cycle and Its Verification in Remote Sensing Observations. Journal of the Atmospheric Sciences 83, 699–713 (2026).

[13] Shi, X. et al. Convolutional LSTM Network: a machine learning approach for precipitation nowcasting. In Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 1, NIPS’15, 802–810 (MIT Press, Cambridge, MA, USA, 2015).

[14] Ayzel, G., Schefer, T. & Heistermann, M. RainNet v1.0: a convolutional neural network for radar-based precipitation nowcasting. Geoscientific Model Development 13, 2631–2644 (2020).

[15] Franch, G. et al. Precipitation Nowcasting with Orographic Enhanced Stacked Generalization: Improving Deep Learning Predictions on Extreme Events. Atmosphere 11, 267 (2020). Number: 3.

[16] Espeholt, L. et al. Deep learning for twelve hour precipitation forecasts. Nature Communications 13, 5145 (2022). Number: 1.

[17] Agrawal, S. et al. An Operational Deep Learning System for Satellite-Based High-Resolution Global Nowcasting (2025). ArXiv:2510.13050 [cs].

[18] Ravuri, S. et al. Skilful precipitation nowcasting using deep generative models of radar. Nature 597, 672–677 (2021). Number: 7878.

[19] Zhang, Y. et al. Skilful nowcasting of extreme precipitation with Nowcast-Net. Nature 619, 526–532 (2023). Number: 7970.

[20] Gao, Z. et al. PreDif: Precipitation Nowcasting with Latent Difusion Models. In Advances in Neural Information Processing Systems, vol. 36, 78621–78656 (Curran Associates, Inc., 2023).

[21] Leinonen, J., Hamann, U., Nerini, D., Germann, U. & Franch, G. Latent difusion models for generative precipitation nowcasting with accurate uncertainty quantification (2023). ArXiv:2304.12891 [physics].

[22] Gong, J. et al. CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling. In Proceedings of the 41st International Conference on Machine Learning, 15809–15822 (PMLR, 2024).

[23] Yu, D. et al. DifCast: A Unified Framework via Residual Difusion for Precipitation Nowcasting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 27758–27767 (2024).

[24] Zhao, Z., Dong, X., Wang, Y. & Hu, C. Advancing Realistic Precipitation Nowcasting With a Spatiotemporal Transformer-Based Denoising Difusion Model. IEEE Transactions on Geoscience and Remote Sensing 62, 1–15 (2024).

[25] Foo, S. Q. et al. STLDM: Spatio-temporal latent difusion model for precipitation nowcasting. Transactions on Machine Learning Research (2025).

[26] Ribeiro, B. P. & Pucer, J. F. Flowcast: Advancing precipitation nowcasting with conditional flow matching. In The Fourteenth International Conference on Learning Representations (2026).

[27] Wen, P. et al. DuoCast: Duo-Probabilistic Difusion for Precipitation Nowcasting. Proceedings of the AAAI Conference on Artificial Intelligence 40, 39442–39450 (2026).

[28] Ling, X., Li, C., Huang, T., Dong, Q. & Duan, G. Langprecip: Languageaware multimodal precipitation nowcasting. In Forty-third International Conference on Machine Learning (2026).

[29] Schusterbauer, J., Wiese, J., Stracke, N., Phan, T. & Ommer, B. Probabilistic Precipitation Nowcasting with Rectified Flow Transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 25742–25756 (2026).

[30] Goodfellow, I. et al. Generative adversarial networks. Commun. ACM 63, 139–144 (2020).

[31] Song, Y. et al. Score-based generative modeling through stochastic diferential equations. In International Conference on Learning Representations (2021).

[32] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M. & Le, M. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations (2023).

[33] Nooten, C. C. v. et al. Improving precipitation nowcasting for high-intensity events using deep generative models with balanced loss and temperature data: a case study in the Netherlands. Artificial Intelligence for the Earth Systems -1 (2023).

[34] Wang, R., Fung, J. C. H. & Lau, A. K. H. Skillful Precipitation Nowcasting Using Physical-Driven Difusion Networks. Geophysical Research Letters 51, e2024GL110832 (2024).

[35] Zhang, W. et al. MS-EdgeCast: A Dual-Stage Framework With a Multiscale Convolutional Recurrent Network and Edge-Guided Difusion for Convective Storm Nowcasting. IEEE Transactions on Geoscience and Remote Sensing 64, 1–13 (2026).

[36] Franch, G. et al. GPTCast: a weather language model for precipitation nowcasting. Geoscientific Model Development 18, 5351–5371 (2025).

[37] Gao, C., Luo, Y., Leng, L., Liu, J. & Zhang, T. Physics-Guided Latent Diffusion Model for Intensity- and Extremity-Aware VIL Nowcasting. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 19, 17715–17728 (2026).

[38] Krawczyk, B. Learning from imbalanced data: open challenges and future directions. Progress in Artificial Intelligence 5, 221–232 (2016).

[39] Veillette, M., Samsi, S. & Mattioli, C. SEVIR : A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology. In Advances in Neural Information Processing Systems, vol. 33, 22009– 22019 (Curran Associates, Inc., 2020).

[40] Lerch, S., Thorarinsdottir, T. L., Ravazzolo, F. & Gneiting, T. Forecaster’s Dilemma: Extreme Events and Forecast Evaluation. Statistical Science 32, 106–127 (2017).

[41] Gneiting, T. & Ranjan, R. Comparing Density Forecasts Using Thresholdand Quantile-Weighted Scoring Rules. Journal of Business & Economic Statistics 29, 411–422 (2011).

[42] Bellier, J., Zin, I. & Bontron, G. Sample Stratification in Verification of Ensemble Forecasts of Continuous Scalar Variables: Potential Benefits and Pitfalls. Monthly Weather Review 145, 3529–3544 (2017).

[43] Shimodaira, H. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference 90, 227–244 (2000).

[44] Sugiyama, M., Krauledat, M. & M¨uller, K.-R. Covariate Shift Adaptation by Importance Weighted Cross Validation. Journal of Machine Learning Research 8, 985–1005 (2007).

[45] Saltikof, E. et al. An Overview of Using Weather Radar for Climatological Studies: Successes, Challenges, and Potential. Bulletin of the American Meteorological Society 100, 1739–1752 (2019).

[46] Schmetz, J. et al. AN INTRODUCTION TO METEOSAT SECOND GEN-ERATION (MSG). Bulletin of the American Meteorological Society 83, 977–992 (2002).

[47] Veillette, M. S., Hassey, E. P., Mattioli, C. J., Iskenderian, H. & Lamey, P. M. Creating Synthetic Radar Imagery Using Convolutional Neural Networks. Journal of Atmospheric and Oceanic Technology 35, 2323–2338 (2018).

[48] Lee, Y. & Hilburn, K. Validating GOES Radar Estimation via Machine Learning to Inform NWP (GREMLIN) Product over CONUS. Journal of Applied Meteorology and Climatology 63, 471–486 (2024).

[49] Qin, Y. et al. SRDif: A Cross-Modal Difusion Model for Satellite-to-Radar Translation in Precipitation Nowcasting. IEEE Transactions on Geoscience and Remote Sensing 64, 4106416–4106416 (2026).

[50] Zeng, Q. et al. MCDA-UNet: A satellite data-based model for radar composite reflectivity retrieval. Atmospheric Research 330, 108619 (2026).

[51] K¨u¸c¨uk, C., Giannakos, A., Schneider, S. & Jann, A. Transformer-Based Nowcasting of Radar Composites from Satellite Images for Severe Weather. Artificial Intelligence for the Earth Systems 3 (2024).

[52] Dai, K. et al. Four-hour thunderstorm nowcasting using a deep difusion model for satellite data. Proceedings of the National Academy of Sciences 122, e2517520122 (2025).

[53] Metzl, C. et al. Physical Scales Matter: The Role of Receptive Fields and Advection in Satellite-Based Thunderstorm Nowcasting with Convolutional Neural Networks. Artificial Intelligence for the Earth Systems 4 (2025).

[54] Pathak, J. et al. Learning Accurate Storm-Scale Evolution from Observations (2026). ArXiv:2601.17268 [physics].

[55] Kahn, H. Use of diferent Monte Carlo sampling techniques (The RAND Corporation, Santa Monica, Calif, 1955). Open Library ID: OL20777605M.

[56] Winkler, R. L. & Murphy, A. H. “Good” Probability Assessors. Journal of Applied Meteorology and Climatology 7, 751–758 (1968).

[57] Gneiting, T. & Raftery, A. E. Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association 102, 359– 378 (2007).

[58] Ovadia, Y. et al. Can you trust your model’ s uncertainty? Evaluating predictive uncertainty under dataset shift. In Advances in Neural Information Processing Systems, vol. 32 (Curran Associates, Inc., 2019).

[59] Park, S., Bastani, O., Weimer, J. & Lee, I. Calibrated Prediction with Covariate Shift via Unsupervised Domain Adaptation. In Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics, 3219–3229 (PMLR, 2020).

[60] Horvitz, D. G. & Thompson, D. J. A Generalization of Sampling Without Replacement from a Finite Universe. Journal of the American Statistical Association 47, 663–685 (1952).

[61] Karras, T., Aittala, M., Aila, T. & Laine, S. Elucidating the design space of difusion-based generative models. In Oh, A. H., Agarwal, A., Belgrave, D. & Cho, K. (eds.) Advances in Neural Information Processing Systems (2022).

[62] Ho, J. & Salimans, T. Classifier-free difusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications (2021).

[63] Rasp, S. et al. WeatherBench: A Benchmark Data Set for Data-Driven Weather Forecasting. Journal of Advances in Modeling Earth Systems 12, e2020MS002203 (2020).

[64] Rasp, S. et al. WeatherBench 2: A Benchmark for the Next Generation of Data-Driven Global Weather Models. Journal of Advances in Modeling Earth Systems 16, e2023MS004019 (2024).

[65] Harris, L., McRae, A. T. T., Chantry, M., Dueben, P. D. & Palmer, T. N. A Generative Deep Learning Approach to Stochastic Downscaling of Precipitation Forecasts. Journal of Advances in Modeling Earth Systems 14, e2022MS003120 (2022).

[66] Addison, H., Kendon, E., Ravuri, S., Aitchison, L. & Watson, P. A. Machine learning emulation of a local-scale UK climate model (2022). ArXiv:2211.16116 [physics.ao-ph].

[67] (DWD), D. W. Wetter und Klima - Deutscher Wetterdienst - Leistungen - RADOLAN-RADVOR-Kompositformatbeschreibung (2025).

[68] Franch, G. et al. IT-DPC-SRI: A Cloud-Optimized Archive of Italian Radar Precipitation (2010-2025) (2026). ArXiv:2602.15088 [physics] version: 1.

[69] Lu, C. et al. DPM-Solver++: Fast Solver for Guided Sampling of Difusion Probabilistic Models. Machine Intelligence Research 22, 730–751 (2025). ArXiv:2211.01095 [cs].

[70] Wilks, D. S. (ed.) Statistical Methods in the Atmospheric Sciences (Fourth Edition) (Elsevier, 2019).

## Acknowledgements

C. M. carried out his contributions within the Italia–Deutschland Science-4–Services Network in Weather and Climate (IDEAS4S; SESTO, 4823IDEAP4). This Italian–German research network of universities, research institutes, and DWD was funded by the Federal Ministry of Digital and Transport (BMDV). We gratefully acknowledge the computational and data resources provided through the joint high performance data analytics (HPDA) project “terrabyte” of the DLR and the Leibniz Supercomputing Center (LRZ).

## Author contributions

C.M. conceived the study, developed the methodology and theoretical analysis, and supervised S.K.’s research. K.V.Y. contributed to the theoretical analysis. S.K. implemented the initial data processing pipeline and DDPM-based model. C.M. developed the final data processing and verification pipelines and EDMbased model, conducted the experiments, analysed the results, and wrote the original manuscript draft. V.P. contributed to radar data processing. T.B., K.V.Y., V.P. and R.M. supervised C.M.’s research. F.K. supervised S.K.’s research. T.B. acquired funding. All authors contributed to refining the research concept and reviewed and edited the manuscript.

## Competing interests

The authors declare no competing interests.

# Wet-Biased Data Pipelines Mislead the Training and Evaluation of Generative Precipitation Nowcasting

# Supplementary Information

## 1 Additional Sampling Practices Survey

This section surveys the dataset sampling and calibration evaluation practices of additional generative precipitation nowcasting studies in Supplementary Table 1, complementing Table 1.

Supplementary Table 1: Dataset sampling and calibration evaluation practices in additional generative precipitation nowcasting studies. Studies included in Table 1 are omitted.

<table><tr><td>Study</td><td>Model</td><td>Training sampling</td><td>Test sampling</td><td>Deployment recovery</td><td>Probabilistic calibration</td></tr><tr><td>LangPrecip [1]</td><td>Flow matching</td><td>Event- curated; filtering unclear</td><td>Equivalent to training</td><td>Unclear</td><td></td></tr><tr><td>PostCast [2]</td><td>Diffusion</td><td>Mixed; dataset- inherited</td><td>Random 500 Mixed per dataset from test splits</td><td></td><td></td></tr><tr><td>PreDiff [3]</td><td>Diffusion Event-</td><td>curated</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>DYffCast [4]</td><td>Diffusion Unclear</td><td></td><td>Unclear</td><td>Unclear</td><td>Spread-skill only</td></tr><tr><td>DuoCast [5]</td><td>Diffusion Event-</td><td>curated / determinis- tic filtering</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>MoCast+ [6]</td><td>Diffusion</td><td>1 Event- curated / determinis- tic filtering</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>Nowcast3D [7] Diffusion</td><td></td><td>Event- curated; filtering unclear</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>PixelFlowCast Flow [8]</td><td>matching curated /</td><td>Event- unclear</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>RainDiff [9]</td><td>Diffusion</td><td>Event- curated/ uniform / unclear</td><td>Equivalent to training</td><td>Mixed</td><td></td></tr><tr><td>HARECast [10]</td><td>Diffusion</td><td>Event- curated/ determinis- tic filter</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>EchoCast-3D [11]</td><td>Diffusion</td><td>Uniform (warm season)</td><td>Equivalent to training; extreme- event subset</td><td>Mixed</td><td>Coverage / spread-skill only</td></tr><tr><td>CRFT [12]</td><td>Flow matching</td><td>Event- curated / determinis- tic filtering</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>NowcastDiff [13]</td><td>Diffusion</td><td>Event- curated; details unclear</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>cGAN-att [14] GAN</td><td></td><td>Event- curated; mechanism unclear</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>RadarDiT [15] Diffusion</td><td></td><td>Event- curated; details unclear</td><td>Equivalent to training</td><td>No</td><td>Spread-skill only</td></tr><tr><td>CastDiffuser [16]</td><td>Diffusion</td><td>Event- curated</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>RNDiff [17]</td><td>Diffusion</td><td>filter</td><td>Deterministic Equivalent to training</td><td>No</td><td></td></tr><tr><td>Physics- guided LDM [18]</td><td>Diffusion</td><td>Event- curated</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>HyDiff [19]</td><td></td><td>Diffusion Deterministic Equivalent filter</td><td>to training</td><td>No</td><td></td></tr><tr><td>Spatiotemporal Diffusion Transformer DDPM [20]</td><td></td><td>Event- curated</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>MTLDM [21]</td><td>Diffusion</td><td>unclear</td><td>Chronological;Equivalent construction to training</td><td>Unclear</td><td></td></tr><tr><td>Forecastformer Diffusion [22]</td><td></td><td>Uniform source; additional selection unclear</td><td>Equivalent to training</td><td>Unclear</td><td></td></tr><tr><td>HybridFlow [23]</td><td>Flow matching</td><td>Event- curated; details unclear</td><td>Equivalent to training</td><td>No</td><td></td></tr><tr><td>MS-EdgeCast [24]</td><td>Diffusion</td><td>filter</td><td>Deterministic Equivalent to training</td><td>No</td><td></td></tr><tr><td>DTCA [25]</td><td>Diffusion</td><td>Unclear</td><td>Unclear</td><td>Unclear</td><td></td></tr><tr><td>SSLDM-ISI [26]</td><td>Diffusion</td><td>unclear</td><td>Chronological;Equivalent construction to training</td><td>Unclear</td><td></td></tr><tr><td>TRDM [27]</td><td>Diffusion</td><td>construction to training unclear</td><td>Chronological;Equivalent</td><td>Unclear</td><td></td></tr><tr><td>STLDM [28]</td><td>Diffusion</td><td>Event- curated/ determinis-</td><td>Equivalent to training</td><td>No</td><td></td></tr></table>

## 2 Importance Sampling Configuration and Sensitivity

This section lists the parameter configuration of the importance sampling in Supplementary Table 2 and studies their sensitivity in Supplementary Fig. 1 and Supplementary Table 3.

Supplementary Table 2: Parameters of the target-dependent importance sampling scheme used to construct the training set. The normalization size D is the number of radar values in one target crop.
<table><tr><td>Parameter</td><td>Symbol</td><td>Value</td></tr><tr><td>Saturation scale</td><td>S</td><td> $1 \mathrm { m m h ^ { - 1 } }$ </td></tr><tr><td>Rain contribution coefficient</td><td>m</td><td>2</td></tr><tr><td>Minimum acceptance probability</td><td>qmin</td><td>0.004</td></tr><tr><td>Time steps</td><td> $T _ { y }$ </td><td>3</td></tr><tr><td>Crop height</td><td>H</td><td>128 pixels</td></tr><tr><td>Crop width</td><td>W</td><td>128 pixels</td></tr><tr><td>Normalization size</td><td> $D = T _ { y } H W$ </td><td>49,152</td></tr></table>

Supplementary Table 3: Radar-reflectivity class fractions in the retained training crops under the training configuration and two sensitivity settings. Values are percentages of all pooled target pixels and sum to 100% within each row. The lower- and higher-acceptance settings multiply both m and $q _ { \mathrm { m i n } }$ by 0.05 and 5, respectively, relative to Supplementary Table 2, while holding s, $T _ { y } , H _ { ; }$ and W fixed. Class boundaries are left-inclusive and right-exclusive except for the open outer bounds.
<table><tr><td>Setting</td><td>No rain (&lt; 10 dBZ)</td><td>Light (10–30 dBZ)</td><td>Moderate (30–38 dBZ)</td><td>Strong (38–48 dBZ)</td><td>Extreme (≥ 48 dBZ)</td></tr><tr><td>Lower-acceptance sensitivity (0.05×)</td><td>56.251</td><td>35.519</td><td>6.852</td><td>1.281</td><td>0.098</td></tr><tr><td>Training configuration</td><td>57.788</td><td>34.767</td><td>6.188</td><td>1.160</td><td>0.097</td></tr><tr><td>Higher-acceptance sensitivity (5×)</td><td>72.465</td><td>23.538</td><td>3.270</td><td>0.655</td><td>0.071</td></tr></table>

## 3 Data Processing Details

Supplementary Table 4 lists the counts removed by each quality-control rule and Supplementary Table 5 the sizes of all datasets. The German radar fields are stored standardized with preliminary statistics and are re-standardized on loading with the values in Supplementary Table 6. Standardized fields are stored in 16-bit floating point and read as 32-bit numbers. The Italian ground truth is stored as $\log ( 1 + R )$ standardized with mean 0.3147 and standard deviation 0.5722 and is transformed back to $\mathrm { m m h ^ { - 1 } }$ and clipped at zero before verification.

The mean inverse acceptance weight $1 / a ( y )$ is 14.60 on the German training set, which normalizes the IPW loss, 12.54 on the German importance sampled test set, and 20.70 on the Italian one. The corresponding weight sums

![](images/0eb08e7bfd19b94c3ffbcfd12d91c3799c8da2816100afbc386d1ccf8ce00a5d.jpg)  
Supplementary Fig. 1: Sensitivity of the pooled training-target distribution to the efective acceptance coeficients. Curves show normalized radar-reflectivity frequencies across all target pixels retained under the training configuration in Supplementary Table 2 and two sensitivity settings. The lower-acceptance setting scales m and $q _ { \mathrm { m i n } }$ by 0.05 relative to the training configuration, whereas the higher-acceptance setting scales both by $5 ;$ all other parameters are unchanged. Higher acceptance causes a larger fraction of candidate crops to reach unit acceptance probability, reduces the relative enrichment of rainy crops, and shifts the retained distribution toward no-rain values.

Supplementary Table 4: Efect of quality control on sample count for both domains. Timestamps refer to the 15-min grid of the May–October seasons. A timestamp is globally invalid if the radar file is missing, if at least one of the SEVIRI channel files is missing so that the scene cannot be gridded, or if a gridded channel contains missing values. Crop-timestamps are timestamps multiplied by the number of lattice crops (91 over Germany, 300 over Italy). A crop-timestamp is masked if its timestamp is globally invalid or if more than 360 of its radar pixels are missing. Candidate sequences are six-frame samples for which all frames are valid and fall within May–October.
<table><tr><td>Quantity</td><td>Germany (2015–2020)</td><td>Italy (2025)</td></tr><tr><td>Timestamps on the 15-min grid</td><td>105,984</td><td>17,664</td></tr><tr><td>Globally invalid timestamps</td><td>807</td><td>30</td></tr><tr><td>missing satellite source files</td><td>690</td><td>16</td></tr><tr><td>missing satellite values</td><td>35</td><td>14</td></tr><tr><td>missing radar files</td><td>82</td><td>0</td></tr><tr><td>Crop-timestamps</td><td>9,644,544</td><td>5,299,200</td></tr><tr><td>Masked crop-timestamps</td><td>217,121 (2.25%)</td><td>3,275,540 (61.8%)</td></tr><tr><td>Candidate sequences, training</td><td>6,221,943</td><td></td></tr><tr><td>Candidate sequences, validation</td><td>1,568,930</td><td></td></tr><tr><td>Candidate sequences, test</td><td>1,543,127</td><td>1,967,513</td></tr></table>

Supplementary Table 5: Sizes of all datasets used in this study. Importance sampled sets use Eq. (6), while uniform sets are simple random samples without replacement of the same size as the corresponding importance sampled set. The acceptance rate is the size of the importance sampled set relative to the candidate pool. Only the uniform validation set is used for checkpoint selection.
<table><tr><td>Split (season)</td><td>Candidates</td><td>Importance sampled</td><td></td><td>Uniform Acceptance rate</td></tr><tr><td>Training (2015–2018)</td><td>6,221,943</td><td>425,713</td><td></td><td>6.84%</td></tr><tr><td>Validation (2019)</td><td>1,568,930</td><td>125,765</td><td>125,765</td><td>8.02%</td></tr><tr><td>Test Germany (2020)</td><td>1,543,127</td><td>122,644</td><td>122,644</td><td>7.95%</td></tr><tr><td>Test Italy (2025)</td><td>1,967,513</td><td>95,111</td><td>95,111</td><td>4.83%</td></tr></table>

of 1,538,438 and 1,969,183 estimate the candidate-pool sizes of 1,543,127 and 1,967,513 in Supplementary Table 4.

Data processing used wradlib 2.4.0 [29], satpy 0.55.0 [30], pyresample 1.33.0, and zarr 2.16.1.

## 4 Hyperparameters and Training Details

This section lists the hyperparameters in Supplementary Table 6, presents the value chain for one training step of the model in Supplementary Fig. 2, while additional details for architecture are displayed in Supplementary Figs. 3 and 4.

Supplementary Table 6: Hyperparameters used for model training and ensemble inference. The baseline and IPW models difer only in the per-sample loss multiplier. The model was implemented in PyTorch 2.5.1 [31] with PyTorch Lightning. Values labelled as PyTorch defaults are stated explicitly. The epoch is zero indexed. Training and inference are seeded, but bitwise reproducibility is not guaranteed because deterministic kernels were not enforced and nondeterministic GPU algorithms were not disabled. All computation is in 32-bit floating point.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Data and preprocessing</td><td></td></tr><tr><td>Satellite predictors</td><td>HRV, IR_108, VIS008, WV_062</td></tr><tr><td>Input/target sequence</td><td>3 input frames (−30, -15, 0 min); 3 target frames (0, 15, 30</td></tr><tr><td>Temporal resolution</td><td>min) 15 min</td></tr><tr><td>Crop size and stride</td><td>128 × 128 pixels; 64-pixel stride (50% overlap)</td></tr><tr><td>Satellite standardization (µ, σ)</td><td>HRV (11.3462, 14.2666); IR_108 (270.7699, 16.8125); VIS008 (12.9961, 15.3883); WV_062 (232.0065, 4.3216)</td></tr><tr><td>Radar standardization (µ, σ) (−3.9248, 25.7472) dBZ Sampling used for</td><td></td></tr><tr><td>training/validation</td><td>Importance sampling / uniform validation</td></tr><tr><td>Training/validation samples</td><td>425,713 / 125,765 crops (3,326 optimization steps per epoch)</td></tr><tr><td>Network</td><td></td></tr><tr><td>Architecture</td><td>Conditional U-Net; 82,297,571 trainable parameters</td></tr><tr><td>Input/output channels</td><td>4 satellite and 1 noisy-radar / 1 radar</td></tr><tr><td>Satellite feature width</td><td>64 (2-D widths 32, 16; subsequent 3-D width 64)</td></tr><tr><td>U-Net channel widths</td><td>128, 256, 512, 1024</td></tr><tr><td>Residual blocks</td><td>2 per encoder and decoder level; 2 in bottleneck</td></tr><tr><td>Bottleneck attention</td><td>8-head spatial self-attention at 16 × 16 pixels</td></tr><tr><td>Normalization / activation</td><td>Group normalization (32 groups; 8 in satellite encoder) /</td></tr><tr><td>Noise embedding</td><td>SiLU 128-dimensional random Fourier features; frequency scale 16</td></tr><tr><td>Time-of-day encoding</td><td>16-dimensional sinusoidal encoding of the fraction of day of each input frame; linear layer; added to the per-frame</td></tr><tr><td>Multi-scale conditioning</td><td>Satellite features average pooled by factors 2, 4, 8 and projected to 256, 512, 1024 channels; added after each downsampling and before each upsampling</td></tr><tr><td>Unconditional branch</td><td>Satellite encoder and all injections skipped; dedicated first-level stem (513,920 parameters); all deeper levels</td></tr><tr><td>Upsampling</td><td>shared Nearest-neighbour resize followed by  $3 \times 3$  convolution</td></tr><tr><td>EDM objective</td><td></td></tr><tr><td>Data-noise scale</td><td> $\sigma _ { \mathrm { { d a t a } } } = 1 . 0$ </td></tr><tr><td>Training-noise distribution</td><td>ln  $\sigma \sim \mathcal N ( P _ { \mathrm { m e a n } } , P _ { \mathrm { s t d } } ^ { 2 } )$ </td></tr><tr><td>Noise-distribution parameters</td><td> $P _ { \mathrm { m e a n } } = - 0 . 5 , P _ { \mathrm { s t d } } = 1 . 2$ </td></tr><tr><td>Loss</td><td>EDM-preconditioned, noise-level-weighted mean squared</td></tr><tr><td>Baseline / IPW multiplier</td><td>error  $1 \ / \ [ 1 / a ( y ) ] / 1 4 . 6 0 2 8$ </td></tr><tr><td>Optimization</td><td></td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>AdamW  $\beta _ { 1 } , \beta _ { 2 } ; \epsilon ;$  weight decay</td><td>(0.9, 0.999); 10−8; 0.01 (PyTorch defaults)</td></tr><tr><td>Batch size / gradient</td><td>128 / 1</td></tr><tr><td>accumulation Initial learning rate</td><td> $1 . 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Linear warm-up for 5 epochs, then cosine annealing for 95</td></tr><tr><td>Warm-up start / cosine</td><td>epochs  $1 0 ^ { - 6 } \mathrm { ~ / ~ } 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>minimum Epoch cap / selected</td><td>100 / epoch 81 (baseline and IPW)</td></tr><tr><td>checkpoint Early stopping</td><td>Uniform-validation EDM loss; patience 10; minimum change</td></tr><tr><td>Gradient clipping</td><td>0 Global norm 1.0</td></tr><tr><td>Exponential moving average</td><td>Decay 0.999 from step  $0 ;$  EMA weights used for validation</td></tr><tr><td>Classifier-free conditioning</td><td>and inference 0.10</td></tr><tr><td>dropout Spatial augmentation</td><td>Random 90°rotation and independent horizontal/vertical</td></tr><tr><td>Training seed / numerical</td><td>flips (p = 0.5 each) 31 / 32-bit / single NVIDIA HGX A100 80GB 500W GPU</td></tr><tr><td>precision / device Ensemble inference</td><td></td></tr><tr><td>Sampler</td><td>DPM-Solver++(2M) with the EDM (Karras) noise schedule</td></tr><tr><td>Sampling steps</td><td>50</td></tr><tr><td>Sampling-noise range and</td><td> $\sigma _ { \mathrm { m a x } } = 8 0 , \sigma _ { \mathrm { m i n } } = 0 . 0 0 2 , \rho = 7$ </td></tr><tr><td>schedule exponent Initial latent</td><td>Independent  $\mathcal { N } ( 0 , \sigma _ { \mathrm { m a x } } ^ { 2 } I )$  noise for each member</td></tr><tr><td>Ensemble size</td><td>20</td></tr><tr><td>Classifier-free guidance scales</td><td>1, 2, and 4 (scale 1 for the baseline/IPW comparison)</td></tr><tr><td>Network evaluations per member</td><td>50 at guidance scale 1; 100 at scales 2 and 4</td></tr><tr><td>Inference seed / batch size</td><td>8932  /  128</td></tr></table>

## 5 Verification Details

The rank histogram, CRPS, and power spectra are computed with pysteps 1.12.0. [32]; the FSS re-implements the pysteps smoothing so that numerator and denominator of Eq. (19) can be accumulated across samples and blocks. The random tie-breaking of the rank histogram uses an unseeded random number generator, so repeated evaluations difer by the tie-breaking noise. This afects only how the tied no-rain mass is distributed across neighboring ranks and does not alter any conclusion, but the histograms are not bitwise reproducible. The block bootstrap uses a fixed seed; the 27 seven-day blocks span 26 April to 25 October 2020 over Germany and 27 April to 26 October 2025 over Italy, i.e. block boundaries are fixed by the calendar rather than by the data. No confidence intervals are computed for the power spectra.

## 6 Additional Evaluation

This section presents additional evaluation results. Supplementary Figure 5 displays the radially averaged power spectral density (RAPSD) on both domains and test set sample protocols, while Supplementary Fig. 6 shows the lead time dependence of selected skill scores. Complementary verification scores for Table 2 are in Supplementary Table 7, while complementary verification scores for the classifier-free guidance comparison are in Supplementary Table 8.

## 7 IPW Training and Post-Processing

Section 3.2 and Table 2 apply IPW at the verification stage, where every test sample is weighted by the inverse of its acceptance probability. Because Eq. (3) is invertible, the same correction can in principle be applied at two other stages of the pipeline: to the loss function, so that the model learns the deployment conditional in the first place, or to the samples of the trained model in a postprocessing step. Both variants are the identical Horvitz-Thompson reweighting [33] applied to a diferent object, namely the loss instead of the score, or the predictive density instead of the test set. Both require that $a ( y )$ is known and strictly positive for every candidate sample. This section reports an experiment for the training stage and outlines the post-processing variant.

Weighting the per-sample loss by $w ( y ) = 1 / a ( y )$ turns the empirical training objective over the importance sampled set into an unbiased estimate of the objective over the deployment distribution, in the same way as importance weights correct a Monte Carlo estimate [34] or a covariate-shifted training set [35]. For

![](images/52823a81f72813c5d35242f5c111ce5e07322731354e8f74520748670434bcaf.jpg)  
Supplementary Fig. 2: Schematic of one EDM training step in the notation of Section 5.2.

![](images/8b13f42f46e0947de4443bb0faa81f72b0055f26b45ac62b9a70d0af21cc2bc7.jpg)  
Supplementary Fig. 3: High-level architecture of the conditional 2D U-Net $F _ { \theta }$ The three noisy radar frames are concatenated with the full-resolution satellite features and the temporal dimension is folded into the channel dimension. Encoder and decoder operate on four resolution levels from $1 2 8 \times 1 2 8$ to $1 6 \times 1 6$ pixels with 128 to 1024 channels and two residual blocks per level; the bottleneck additionally contains an 8-head spatial self-attention block. Downsampling uses strided $3 \times 3$ convolutions and upsampling nearest-neighbour resizing followed by a $3 \times 3$ convolution. Skip connections are concatenated and reduced by a $1 \times 1$ projection. Multi-scale satellite conditioning (green arrows) is added after each downsampling and before each upsampling. A final $1 \times 1$ convolution yields the three output frames.

![](images/fb09a0663d31054e0ee4479f4458bf159c9d2748b2e6eb473e4df1eb6fce3ab6.jpg)  
(a) Satellite conditioning encoder.

![](images/974dc9c870839204be261408fdb7838a15d348908fcad1110812a184495a9c67.jpg)  
(b) Difusion embedding and conditioned residual block.

Supplementary Fig. 4: Conditioning components of the conditional difusion model. (a) Satellite conditioning encoder. Each of the three input frames with four channels passes through two convolution blocks (convolution, group normalization, SiLU) with 32 and 16 channels. A 16-dimensional sine–cosine encoding of the time of day of each frame, followed by a linear layer, is added to the per-frame features. After restoring the temporal axis, a $3 \times 3 \times 3$ convolution block fuses the frames into 64 feature channels per time step. This representation is concatenated with the noisy radar at the U-Net input (full-resolution conditioning) and, after spatial pooling, folding time into channels, and a $1 \times 1$ projection, added at the $6 4 ^ { 2 }$ , 32<sup>2</sup>, and $1 6 ^ { 2 }$ levels (multi-scale conditioning). (b) Conditioned residual block. The noise level $c _ { \mathrm { n o i s e } }$ is mapped by a Fourier embedding, a linear layer, and SiLU to channel-wise scale and shift parameters that modulate the group-normalized activations (adaptive GroupNorm, AdaGN). The block consists of two $3 \times 3$ convolutions with group normalization and SiLU and a residual connection that is the identity or a $1 \times 1$ convolution when the channel count changes.

![](images/27dd83d0f6d74e724e8ffad79a7a4eb003f171f091ca74e74e69b621521795af.jpg)  
Supplementary Fig. 5: Radially averaged power spectral density (RAPSD) of the model forecasts and corresponding ground-truth fields over Germany (top row) and Italy (bottom row), evaluated on the uniform (left column) and importance sampled (right column) test sets. Curves aggregate all forecast lead times. Forecast spectra use the color of the respective sampling protocol, while ground truth is shown by the charcoal dashed curves. Both axes are logarithmic, and spatial frequency is given in pixel−<sup>1</sup>. The absolute spectral densities should not be compared across domains because the German fields contain radar reflectivity in dBZ, whereas the Italian fields contain precipitation rates in mm $\mathrm { h } ^ { - 1 }$ Notably, the model reproduces the RAPSD almost perfectly when evaluated on the importance sampled German set. However, the model lies consistently above the ground truth on the uniform set. Furthermore, the RAPSD does not generalize well to the Italian domain on either set.

![](images/90b6a48610cec82e92fd36055c40e7e54ae7020493be333eaa88771247fbfaa7.jpg)  
Supplementary Fig. 6: Critical success index (CSI; top row) and fractions skill score at a neighborhood scale of 128 pixels $\mathrm { ( F S S _ { 1 2 8 } ; }$ bottom row) as functions of lead time on the uniform German and Italian test sets. Columns correspond to threshold-matched intensity levels of 10, 30, and 38 dBZ over Germany and 0.10, 2.61, and $9 . 5 5 \mathrm { m m h ^ { - 1 } }$ over Italy, respectively. Scores are member averaged; solid blue curves with circles denote Germany and dashed magenta curves with squares denote Italy. Across the available 0–30-minute range, both scores are nearly invariant with lead time, apart from a modest decrease in CSI at the highest threshold. This weak temporal degradation is consistent with the satellite-to-radar conversion being the dominant dificulty over these short lead times, rather than forecast extrapolation alone.

Supplementary Table 7: Complementary verification scores for Table 2. Definitions, aggregation, threshold matching, evaluation protocols, and confidence intervals are as in Table 2. FSS is shown at the two neighborhood scales omitted from the main table. CRPS is reported in dBZ for Germany and $\mathrm { m m h ^ { - 1 } }$ for Italy and is not comparable across domains.
<table><tr><td colspan="2">Metric Threshold</td><td>Uniform test set</td><td></td><td></td><td>test set</td><td colspan="2">Importance sampled IPW estimate from importance set</td></tr><tr><td colspan="2">Germany</td></tr><tr><td>10 dBZ</td><td></td><td>0.420 [0.411, 0.431]</td><td></td><td>0.450 [0.439, 0.460]</td><td></td><td>0.421 [0.411, 0.431]</td><td></td></tr><tr><td>POD</td><td>30 dBZ</td><td></td><td>0.060 [0.050, 0.072]</td><td></td><td>0.047 [0.038, 0.058]</td><td></td><td>0.059 [0.048, 0.072]</td></tr><tr><td></td><td>38 dBZ</td><td></td><td>0.029 [0.022, 0.036]</td><td></td><td>0.019 [0.013, 0.024]</td><td></td><td>0.028 [0.021, 0.035]</td></tr><tr><td>FAR</td><td>10 dBZ</td><td></td><td>0.727 [0.700, 0.756]</td><td></td><td>0.347 [0.322, 0.375]</td><td></td><td>0.727 [0.701, 0.755]</td></tr><tr><td></td><td>30 dBZ</td><td></td><td>0.919 [0.900, 0.939]</td><td></td><td>0.809 [0.785, 0.840]</td><td></td><td>0.919 [0.901, 0.938]</td></tr><tr><td></td><td>38 dBZ</td><td></td><td>0.958 [0.946, 0.972]</td><td></td><td>0.919 [0.906, 0.938]</td><td></td><td>0.958 [0.947, 0.971]</td></tr><tr><td>FSS8</td><td>10 dBZ</td><td></td><td>0.422 [0.398, 0.444]</td><td></td><td>0.600 [0.589, 0.611]</td><td></td><td>0.423 [0.399, 0.443]</td></tr><tr><td></td><td>30 dBZ</td><td></td><td>0.121 [0.100, 0.143]</td><td></td><td>0.119 [0.099, 0.142]</td><td></td><td>0.121 [0.099, 0.143]</td></tr><tr><td></td><td>38 dBZ</td><td></td><td>0.077 [0.058, 0.093]</td><td></td><td>0.063 [0.046, 0.078]</td><td></td><td>0.076 [0.057, 0.091]</td></tr><tr><td>FSS32</td><td>10 dBZ</td><td></td><td>0.526 [0.500, 0.550]</td><td></td><td>0.686 [0.676, 0.696]</td><td></td><td>0.527 [0.502, 0.549]</td></tr><tr><td></td><td>30 dBZ</td><td></td><td>0.188 [0.155, 0.223]</td><td></td><td>0.169 [0.138, 0.206]</td><td></td><td>0.186 [0.151, 0.223]</td></tr><tr><td>CRPS</td><td>38 dBZ</td><td></td><td>0.151 [0.112, 0.184]</td><td></td><td>0.114 [0.082, 0.148]</td><td></td><td>0.146 [0.107, 0.181]</td></tr><tr><td></td><td></td><td></td><td>12.21 [12.04, 12.37]</td><td></td><td>11.97 [11.78, 12.14]</td><td></td><td>12.21 [12.03, 12.40]</td></tr><tr><td colspan="2">Italy</td></tr><tr><td></td><td>10 dBZ</td><td>0.392 [0.382, 0.400]</td><td></td><td>0.439 [0.432, 0.446]</td><td></td><td>0.392 [0.383, 0.400]</td><td></td></tr><tr><td>POD</td><td>30 dBZ</td><td></td><td>0.084 [0.075, 0.094]</td><td></td><td>0.079 [0.071, 0.088]</td><td></td><td>0.086 [0.077, 0.094]</td></tr><tr><td></td><td>38 dBZ</td><td></td><td>0.037 [0.031, 0.045]</td><td></td><td>0.029 [0.024, 0.034]</td><td></td><td>0.037 [0.031, 0.042]</td></tr><tr><td>FAR</td><td>10 dBZ</td><td></td><td>0.730 [0.710, 0.755]</td><td></td><td>0.349 [0.334, 0.363]</td><td></td><td>0.729 [0.707, 0.753]</td></tr><tr><td></td><td>30 dBZ</td><td></td><td>0.879 [0.866, 0.891]</td><td></td><td>0.729 [0.711, 0.747]</td><td></td><td>0.875 [0.861, 0.888]</td></tr><tr><td></td><td>38 dBZ</td><td></td><td>0.958 [0.949, 0.966]</td><td></td><td>0.920 [0.910, 0.930]</td><td></td><td>0.957 [0.949, 0.964]</td></tr><tr><td>FSS8</td><td>10 dBZ</td><td></td><td>0.415 [0.394, 0.432]</td><td></td><td>0.592 [0.585, 0.599]</td><td></td><td>0.416 [0.396, 0.434]</td></tr><tr><td></td><td>30 dBZ</td><td></td><td>0.157 [0.142, 0.171]</td><td></td><td>0.172 [0.156, 0.187]</td><td></td><td>0.160 [0.145, 0.173]</td></tr><tr><td></td><td>38 dBZ</td><td></td><td>0.076 [0.065, 0.087]</td><td></td><td>0.074 [0.064, 0.084]</td><td></td><td>0.077 [0.066, 0.087]</td></tr><tr><td>FSS32</td><td>10 dBZ</td><td></td><td>0.536 [0.512, 0.556]</td><td></td><td>0.695 [0.687, 0.703]</td><td></td><td>0.537 [0.512, 0.558]</td></tr><tr><td></td><td>30 dBZ</td><td></td><td>0.242 [0.217, 0.266]</td><td></td><td>0.237 [0.213, 0.260]</td><td></td><td>0.245 [0.222, 0.268]</td></tr><tr><td></td><td>38 dBZ</td><td></td><td>0.159 [0.133, 0.186]</td><td></td><td>0.144 [0.120, 0.169]</td><td></td><td>0.162 [0.137, 0.185]</td></tr><tr><td>CRPS</td><td></td><td></td><td>0.079 [0.064, 0.094]</td><td></td><td>0.872 [0.791, 0.953]</td><td></td><td>0.080 [0.065, 0.095]</td></tr></table>

Supplementary Table 8: Complementary verification scores for the classifier-free guidance comparison in Table 3. The evaluation set, aggregation, and confidence intervals are as in Table 3. FSS is shown at the two neighborhood scales omitted from the main table; CRPS is reported in dBZ.
<table><tr><td>Metric</td><td>Threshold</td><td colspan="2">Guidance scale 1</td><td></td><td>Guidance scale 2</td><td>Guidance scale 4</td><td></td></tr><tr><td rowspan="3">POD</td><td>10 dBZ</td><td></td><td>0.420 [0.411, 0.431]</td><td>0.523</td><td>[0.505, 0.541]</td><td>0.609</td><td>[0.589, 0.631]</td></tr><tr><td>30 dBZ</td><td></td><td>0.060 [0.050, 0.072]</td><td></td><td>0.161 [0.135, 0.190]</td><td></td><td>0.301 [0.264, 0.338]</td></tr><tr><td>38 dBZ</td><td></td><td>0.029 [0.022, 0.036]</td><td></td><td>0.107 [0.087, 0.125]</td><td></td><td>0.226 [0.196, 0.252]</td></tr><tr><td rowspan="3">FAR</td><td>10 dBZ</td><td></td><td>0.727 [0.700, 0.756]</td><td></td><td>0.692 [0.664, 0.720]</td><td></td><td>0.704 [0.677, 0.733]</td></tr><tr><td>30 dBZ</td><td></td><td>0.919 [0.900, 0.939]</td><td></td><td>0.904 [0.885, 0.925]</td><td></td><td>0.920 [0.904, 0.937]</td></tr><tr><td>38 dBZ</td><td></td><td>0.958 [0.946, 0.972]</td><td></td><td>0.950 [0.937, 0.964]</td><td></td><td>0.965 [0.955, 0.975]</td></tr><tr><td rowspan="3">FSS8</td><td>10 dBZ</td><td></td><td>0.422 [0.398, 0.444]</td><td></td><td>0.487 [0.461, 0.510]</td><td>0.498</td><td>[0.470, 0.524]</td></tr><tr><td>30 dBZ</td><td></td><td>0.121 [0.100, 0.143]</td><td></td><td>0.204 [0.168, 0.236]</td><td></td><td>0.202 [0.168, 0.234]</td></tr><tr><td>38 dBZ</td><td></td><td>0.077 [0.058, 0.093]</td><td></td><td>0.146 [0.113, 0.171]</td><td></td><td>0.115 [0.085, 0.141]</td></tr><tr><td rowspan="3">FSS32</td><td>10 dBZ</td><td></td><td>0.526 [0.500, 0.550]</td><td></td><td>0.595 [0.568, 0.621]</td><td></td><td>0.605 [0.575, 0.633]</td></tr><tr><td>30 dBZ</td><td></td><td>0.188 [0.155, 0.223]</td><td></td><td>0.304 [0.252, 0.353]</td><td></td><td>0.299 [0.248, 0.344]</td></tr><tr><td>38 dBZ</td><td></td><td>0.151 [0.112, 0.184]</td><td></td><td>0.268 [0.207, 0.313]</td><td></td><td>0.194 [0.141, 0.237]</td></tr><tr><td>CRPS</td><td></td><td></td><td>12.21 [12.04, 12.37]</td><td></td><td>8.46 [8.15, 8.78]</td><td></td><td>7.09 [6.46, 7.73]</td></tr></table>

the EDM loss of Eq. (13) this means that the minimizer approximates the deployment score $\nabla _ { y } \log p _ { \mathrm { c l i m } } ( y | x )$ instead of the shifted score of $\operatorname { E q . }$ (5). We divide the weights by their mean over the training set (14.60) so that the average loss magnitude, and thereby the efective learning rate, is unchanged, and otherwise train exactly as the baseline (Supplementary Table 6). Both models select epoch 81 by the uniform validation loss. To our knowledge, the only previous account of training-time importance weights in generative nowcasting is a single sentence in the supplement of DGMR, which reports no significant advantage in a GAN setting without further analysis [36].

The weights range from 1 for crops with $a ( y ) = 1$ , which make up 10.9% of the training set, to 238 for the driest crops, and the crops with $a ( y ) < 0 . 0 1$ carry 57% of the total weight. The efective sample size $( \textstyle \sum _ { i } w _ { i } ) ^ { 2 } / \textstyle \sum _ { i } w _ { i } ^ { 2 }$ of the weighted training set is 50,523, i.e. 11.9% of its 425,713 crops. IPW training therefore trades the wet bias of the objective for a much noisier gradient estimate in which rainy crops contribute little.

Supplementary Table 9 and Supplementary Fig. 7 compare both models on the German uniform test set at guidance scale 1. Both CRPS measures improve under IPW training. CRPS decreases from 12.21 [12.04, 12.37] to 11.24 [11.05, 11.43] dBZ and pooled CRPS from 12.26 [12.00, 12.53] to 11.23 [11.01, 11.44] dBZ. The rank histogram flattens and the signed mean-rank bias moves from $- 0 . 0 6 5 \ \left[ - 0 . 0 7 9 , \ - 0 . 0 5 1 \right] \mathrm { ~ t o ~ } - 0 . 0 4 6 \ \left[ - 0 . 0 6 0 , \ - 0 . 0 3 2 \right]$ , a paired diference of +0.019 [+0.016, +0.022], while the frequency bias at 10 dBZ drops from 1.54 to 1.42. The light rain over-prediction of the baseline is thus reduced but not removed. At the same time, categorical skill deteriorates at every threshold and most strongly for strong rain: CSI at 38 dBZ halves from 0.017 to 0.007, $\mathrm { F S S _ { 1 2 8 } }$ falls from 0.247 to 0.145, POD at 30 and 38 dBZ halves, and FAR rises. Reweighting shrinks the efective sample size of exactly those rare events that the sampling scheme was designed to enrich, so that the model efectively sees them about as rarely as under climatological sampling.

IPW training is therefore a choice rather than a free correction. If the goal is a calibrated probabilistic product on the deployment distribution, weighting the loss, or training on a climatologically sampled set, is the principled option. If the goal is maximal skill for rare events under a fixed compute budget, the shift may be retained deliberately. Then, however, it must be stated, and the verification must nevertheless be performed on deployment consistent data or with the weighted scores of Section 5.3.

The third option leaves the training untouched and corrects the predictive distribution of the trained model. Inspired by [37], inverting Eq. (3) and using that the deployment conditional integrates to one gives

$$
p _ { \mathrm { c l i m } } ( y | x ) = \frac { p _ { \mathrm { t r a i n } } ( y | x ) / a ( y ) } { \int p _ { \mathrm { t r a i n } } ( y ^ { \prime } | x ) / a ( y ^ { \prime } ) \mathrm { d } y ^ { \prime } } ,\tag{1}
$$

so that the unknown normalizer $Z ( x )$ is eliminated. For an ensemble of $N _ { e }$ members $y ^ { ( k ) }$ drawn from the trained model, any expectation under the deployment conditional can be estimated with the self-normalized member weights

$$
\mathbb { E } _ { \mathrm { c l i m } } \big [ \varphi ( Y ) | x \big ] \approx \sum _ { k = 1 } ^ { N _ { e } } \omega ^ { ( k ) } \varphi \big ( y ^ { ( k ) } \big ) , \qquad \omega ^ { ( k ) } = \frac { 1 / a \big ( y ^ { ( k ) } \big ) } { \sum _ { j = 1 } ^ { N _ { e } } 1 / a \big ( y ^ { ( j ) } \big ) } ,\tag{2}
$$

where $\varphi$ may be a threshold exceedance, a pixel value, or any other functional of the forecast. Because a is a known function of the target alone, the weights can be evaluated directly on the generated members without reference to observations. Supplementary Equation (2) is the ensemble analogue of the H´ajek estimators used for verification, applied within one forecast rather than across the test set.

We have not applied this correction to our model for two reasons, which we consider limitations of the approach rather than of this study. First, Supplementary Equation (1) is exact only for an ideal model that has learned $p _ { \mathrm { t r a i n } } ( y \mid x )$ perfectly. A budget-constrained model deviates from the predicted tilt in a way the theory does not specify (Section 3.3), so the corrected ensemble must still be verified and, if necessary, tuned. This requires a deployment consistent validation set and returns the problem to the verification stage. Second, the member weights are self-normalized within one forecast of only $N _ { e } = 2 0$ members, whereas $1 / a ( y )$ spans more than two orders of magnitude. Whenever an ensemble contains both dry and rainy members, the driest members receive nearly all of the weight and the efective number of members collapses, so that exceedance probabilities become coarsely quantized and noisy. A reliable estimate of Supplementary Equation (1) therefore requires far larger ensembles than are common in nowcasting, and probably larger than are afordable for difusion models at operational latency. Post-processing IPW is thus most attractive for low-dimensional targets or for models whose sampling is cheap. A systematic evaluation is left for future work.

Supplementary Table 9: Verification of the baseline and inverse-probabilityweighted (IPW) trained models on the German uniform test set at guidance scale 1. Values in brackets are 95% confidence intervals from a 10,000-replicate block bootstrap with seven-day blocks. Categorical scores and fractions skill score (FSS) are member averaged and aggregated over all lead times; Pooled CRPS and CRPS are reported in dBZ. The baseline is repeated from Table 2 to make the model comparison self-contained.
<table><tr><td rowspan=1 colspan=6>Metric        Threshold      Baseline          IPW-trained</td></tr><tr><td rowspan=1 colspan=6>10 dBZ    1.541[1.411, 1.705] 1.416 [1.295, 1.570]</td></tr><tr><td rowspan=1 colspan=4>Bias           30 dBZ    0.747 [0.643, 0.894]</td><td rowspan=1 colspan=2>0.670 [0.578, 0.804]</td></tr><tr><td rowspan=1 colspan=4>38 dBZ    0.687 [0.580, 0.856]</td><td rowspan=1 colspan=2>0.673 [0.565, 0.856]</td></tr><tr><td rowspan=1 colspan=4>10 dBZ   0.420 [0.411, 0.431]</td><td rowspan=1 colspan=2>0.329 [0.323, 0.337]</td></tr><tr><td rowspan=1 colspan=4>POD          30 dBZ    0.060 [0.050, 0.072]</td><td rowspan=1 colspan=2>0.031 [0.026, 0.036]</td></tr><tr><td rowspan=1 colspan=4>38 dBZ    0.029 [0.022, 0.036]</td><td rowspan=1 colspan=2>0.012 [0.009, 0.015]</td></tr><tr><td rowspan=1 colspan=4>10 dBZ   0.727 [0.700, 0.756]</td><td rowspan=1 colspan=2>0.767 [0.744, 0.792]</td></tr><tr><td rowspan=1 colspan=4>FAR          30 dBZ    0.919 [0.900, 0.939]</td><td rowspan=1 colspan=2>0.954 [0.944, 0.965]</td></tr><tr><td rowspan=1 colspan=4>38 dBZ   0.958 [0.946, 0.972]</td><td rowspan=1 colspan=2>0.982 [0.977, 0.988]</td></tr><tr><td rowspan=1 colspan=2>10 dBZ</td><td rowspan=1 colspan=2>0.198 [0.182, 0.213]</td><td rowspan=1 colspan=2>0.158 [0.146, 0.169]</td></tr><tr><td rowspan=1 colspan=1>CSI</td><td rowspan=1 colspan=1>30 dBZ</td><td rowspan=1 colspan=1>0.036 [0.028,</td><td rowspan=1 colspan=1>0.043]</td><td rowspan=1 colspan=1>0.019 [0.015,</td><td rowspan=1 colspan=1>0.022]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>38 dBZ</td><td rowspan=1 colspan=1>0.017 [0.013,</td><td rowspan=1 colspan=1>0.022]</td><td rowspan=1 colspan=1>0.007 [0.005,</td><td rowspan=1 colspan=1>0.009]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10 dBZ</td><td rowspan=1 colspan=1>0.656 [0.628,</td><td rowspan=1 colspan=1>0.681]</td><td rowspan=1 colspan=1>0.570 [0.546,</td><td rowspan=1 colspan=1>0.591]</td></tr><tr><td rowspan=1 colspan=1>FSS128</td><td rowspan=1 colspan=1>30 dBZ</td><td rowspan=1 colspan=1>0.273 [0.231,</td><td rowspan=1 colspan=1>0.321]</td><td rowspan=1 colspan=1>0.176 [0.148,</td><td rowspan=1 colspan=1>0.208]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>38 dBZ</td><td rowspan=1 colspan=1>0.247 [0.188,</td><td rowspan=1 colspan=1>0.300]</td><td rowspan=1 colspan=2>0.145 [0.110, 0.184]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10 dBZ</td><td rowspan=1 colspan=1>0.422 [0.398,</td><td rowspan=1 colspan=1>0.444]</td><td rowspan=1 colspan=2>0.350 [0.330, 0.368]</td></tr><tr><td rowspan=1 colspan=1>FSS8</td><td rowspan=1 colspan=1>30 dBZ</td><td rowspan=1 colspan=1>0.121 [0.100,</td><td rowspan=1 colspan=1>0.143]</td><td rowspan=1 colspan=2>0.069 [0.058, 0.080]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>38 dBZ</td><td rowspan=1 colspan=1>0.077 [0.058,</td><td rowspan=1 colspan=1>0.093]</td><td rowspan=1 colspan=1>0.036 [0.027,</td><td rowspan=1 colspan=1>0.043]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>10 dBZ</td><td rowspan=1 colspan=1>0.526 [0.500,</td><td rowspan=1 colspan=1>0.550]</td><td rowspan=1 colspan=1>0.446 [0.424,</td><td rowspan=1 colspan=1>0.465]</td></tr><tr><td rowspan=1 colspan=1>FSS32</td><td rowspan=1 colspan=1>30 dBZ</td><td rowspan=1 colspan=1>0.188 [0.155,</td><td rowspan=1 colspan=1>0.223]</td><td rowspan=1 colspan=1>0.114 [0.095,</td><td rowspan=1 colspan=1>0.135]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>0.080 [0.059, 0.101]Pooled CRPS              12.26 [12.00, 12.53] 11.23 [11.01, 11.44]CRPS                      12.21 [12.04, 12.37]11.24 [11.05, 11.43]</td></tr></table>

## 8 Additional Case Studies

This section presents additional case studies on the German and Italian domain in Supplementary Figs. 9 and 11 together with their latest satellite inputs in Supplementary Figs. 8 and 10 respectively.

![](images/7f2abcbfa143b39e3e6efc6bfd10b2fa65ce09c1bd92d0a61473e637bd32a24d.jpg)  
(c) IPW-trained

![](images/8ad7dc5c743e0244d911663ceeafa6e4a1673845141ca7ed86d7345707664c08.jpg)

![](images/34a2424f3387aaad690dfb3330a6fd2a0974881f46e8f49ad6cdfaaa03b17920.jpg)

![](images/2dd7179639d6e95f31ddb74c0608e1668c597928e2dcb9ae9179f26a0a003fb6.jpg)  
Supplementary Fig. 7: Calibration of the baseline and inverse-probabilityweighted (IPW) trained models on the German uniform test set at guidance scale 1. Both models are trained on the same importance sampled training set; the IPW-trained model weights its loss by the inverse sampling acceptance probability. Each row shows one model. The left column contains rank histograms aggregated over all lead times; error bars and the reported signed normalized mean-rank-bias intervals are 95% confidence intervals from a 10,000-replicate block bootstrap with seven-day blocks. The dashed horizontal line denotes the relative frequency expected under perfect calibration. The right column contains reliability diagrams for exceedance of 10, 30, and 38 dBZ. Shading denotes the corresponding 95% block-bootstrap confidence intervals; intervals are omitted where fewer than ten seven-day blocks contribute to a probability bin. Marker area follows the logarithm of the refinement-bin count using one common scale across panels and models, and the diagonal denotes perfect reliability. IPW training moves the rank histogram and mean-rank bias toward perfect calibration, although residual overprediction remains.

Germany: latest satellite inputs (t = 0), 2020-05-10 15:15 UTC HRV 120 IR 10.8 µm  
VIS 0.8 µm  
![](images/371f0226c3ccc3aa080def0b5a4bf331c9b32bfe3a3a1d0c1ef8b905b1ae1de7.jpg)

![](images/c03cbc0b20ca4713aa1cea98d18ad6ab0150c560ab2edd20f38f6b311e815348.jpg)

![](images/3cad6ad2b29ea6bf555737a244d8c5e7d9779070da9355a645f1a72750271b4d.jpg)

![](images/455e60f2cc20a5cef949025df890e2ec8f6b465ae2b9ce8f68c24418f3bf9449.jpg)  
Supplementary Fig. 8: Latest satellite inputs $( t = 0 )$ for the German convective case in Supplementary Fig. 9: a, high-resolution visible (HRV); b, 10.8-µm infrared; $\mathbf { c } , 0 . 8 – \mu \mathrm { m }$ visible; and $\mathbf { d } , 6 . 2 – \mu \mathrm { m }$ water vapour. Visible channels show reflectance and thermal channels brightness temperature.

Germany: convective case, 2020-05-10 15:15 UTC; exceedance threshold 38 dBZ s = 1, member 14 s = 2, member 14 Ground truth (+30 min)  
![](images/8a9ffdfe091e4528dd5ce9ff2b6432465b564443f179f475533b3f2cdab30ba8.jpg)  
s = 1 exceedance probability

![](images/5c62fc671d1c7a439b84c13cd05d4c88f17a9eb15afe14fd21b579246277f7a3.jpg)

![](images/3c30043c4e0a4267e00a2eac7c41ac17a76315365ce27ce1f2e02b27002a5cb8.jpg)

![](images/6ac604e08e61fdf490b88badc332d5bc9d6371a7a19b737d9b6e20222489c78a.jpg)

s = 2 exceedance probability  
![](images/e731e50d6bcb30ce64f4d853d2fec2aa7b006234156d395d0a9bee80cab4a47c.jpg)

Exceedance probability <sup>p</sup>s=2 − <sup>p</sup>s=1  
![](images/f16bc9d3b89d8901eef361f6769f873e4e9c7062f11e67357b0b9f2d0a11fe98.jpg)  
Supplementary Fig. 9: German convective case on 10 May 2020, 15:15 UTC and a lead time of 30 minutes. Each panel shows the same $1 2 8 \times 1 2 8 .$ -pixel crop at 1 km spatial resolution. $\mathbf { a } , \mathbf { b } ,$ , The same paired ensemble member at guidance scales $s = 1$ and $s = 2 ; \mathbf { c } ,$ ground truth. $\mathbf { d } , \mathbf { e } ,$ Exceedance probabilities $p _ { s = 1 }$ and $p _ { s = 2 }$ , defined at each pixel as the fraction of the 20 ensemble members reaching or exceeding 38 dBZ at the displayed lead time. f, The pixelwise probability diference $\Delta p = p _ { s = 2 } - p _ { s = 1 }$ . Aggregated over all members, lead times, and pixels in this case, increasing the guidance scale from $s = 1$ to $s = 2$ raises CSI from 0.060 to 0.132 and the forecast exceedance fraction from 0.66% to 2.28%, compared with 1.36% in the observations. The median member maximum increases from 54.08 to 59.71 dBZ.

Italy: latest satellite inputs (t = 0), 2025-06-20 01:15 UTC  
![](images/980e7daac4ff912c8bdde60c3fb309df05ad28dfc97a3453bce7784358b69273.jpg)

![](images/f3dcc02bed6b1ad428f6e014dd10c9176723486f4e201370fd0d5ebcef011ff0.jpg)

![](images/d5d941dbb60a4c8c709a3d1ea9223d2f53170fa546cededdcda58c4eb43d23de.jpg)

![](images/8a376da90edf32fae5313b402aeee742d8de2f86928da516d28fe9844568f1b8.jpg)  
Supplementary Fig. 10: Latest satellite inputs (t = 0) for the Italian light-rain case in Supplementary Fig. 11. Panels and units are defined as in Supplementary Fig. 8; the near-zero visible-channel reflectances are expected at 01:15 UTC.

Italy: light case, 2025-06-20 01:15 UTC; exceedance threshold 0.1 mm $\mathrm { h } ^ { - 1 }$ s = 1, member 10 s = 2, member 10 Ground truth (+30 min)

![](images/7ed87089400c5062e8f55dd6dd81ba08925e32e92ecd00a03a56f7ef02bc622a.jpg)

![](images/eb150d3a3c481803be9c9f4a65bd8381475e95aec278d8fb78cac7a384d6ff22.jpg)

![](images/2ebc627138ca67f1d120dbc75c30d7c2023a2a07b89eac210fb5ad882e8119d8.jpg)

s = 1 exceedance probability  
![](images/73a563649173c30b2ce2deadfeae05fe6a84b538b137f126ec0d08f58df5b55b.jpg)

s = 2 exceedance probability  
![](images/949b7bf32194ead960846fcd9232bdbeea35b7cf9f57200f426b8236fbbb6794.jpg)

![](images/5f5fd469a40800493279c547fca3ca3724dbfa6e73ebf10bb4db3835413ba5f5.jpg)  
Supplementary Fig. 11: Italian light-rain case on 20 June 2025, 01:15 UTC and a lead time of 30 minutes. Crop dimensions, spatial resolution, panel definitions, and guidance scales are as in Supplementary Fig. 9, using an exceedance threshold of 0.10 mm h−<sup>1</sup>. Aggregated over all members, lead times, and pixels in this case, the observed exceedance fraction is 2.86%, whereas the forecast fraction increases from 12.87% at $s = 1$ to 17.88% at $s = 2 .$ . CSI rises only from 0.054 to 0.064, so the additional 5.01 percentage points of forecast event area primarily amplify false light-rain coverage.

## Supplementary References

[1] Ling, X., Li, C., Huang, T., Dong, Q. & Duan, G. Langprecip: Languageaware multimodal precipitation nowcasting. In Forty-third International Conference on Machine Learning (2026).

[2] Gong, J. et al. Postcast: Generalizable postprocessing for precipitation nowcasting via unsupervised blurriness modeling. In The Thirteenth International Conference on Learning Representations (2025).

[3] Gao, Z. et al. PreDif: Precipitation Nowcasting with Latent Difusion Models. In Advances in Neural Information Processing Systems, vol. 36, 78621–78656 (Curran Associates, Inc., 2023).

[4] Seal, D., Arcucci, R., R¨uhling-Cachay, S. & Quilodr´an-Casas, C. DYfCast: Regional Precipitation Nowcasting Using IMERG Satellite Data. A case study over South America (2024). ArXiv:2412.02723 [cs.LG].

[5] Wen, P. et al. DuoCast: Duo-Probabilistic Difusion for Precipitation Nowcasting. Proceedings of the AAAI Conference on Artificial Intelligence 40, 39442–39450 (2026).

[6] Wu, B. et al. MoCast: Learning Turbulent Motions Under Physical Guidance for Precipitation Nowcasting. Proceedings of the AAAI Conference on Artificial Intelligence 40, 15950–15958 (2026).

[7] Chen, H. et al. Nowcast3D: Reliable precipitation nowcasting via gray-box learning (2026). ArXiv:2511.04659 [cs.LG].

[8] Zhu, Y., Shi, C., Feng, Y. & Niu, D. PixelFlowCast: Latent-Free Precipitation Nowcasting via Pixel Mean Flows (2026). ArXiv:2605.10046 [cs.CV].

[9] Nguyen, T., Ma, J., Khan, F. S., Taieb, S. B. & Khan, S. RainDif: End-toend Precipitation Nowcasting Via Token-wise Attention Difusion (2025). ArXiv:2510.14962 [cs.CV].

[10] Wen, P. et al. Stable Attention Response for Reliable Precipitation Nowcasting (2026). ArXiv:2605.13181 [cs.LG].

[11] Wan, J. et al. Generative machine learning for skilful 3D radar nowcasting. npj Climate and Atmospheric Science (2026).

[12] Zhou, Y. et al. More realistic and accurate precipitation nowcasting with Conditional Rectified Flow Transformers. Engineering Applications of Artificial Intelligence 165, 113402 (2026).

[13] Liu, J. et al. Uncertainty-aware precipitation nowcasting with difusion model simulating precipitation evolution processes. Engineering Applications of Artificial Intelligence 178, 115140 (2026).

[14] Piran, M. J., Wang, X., Kim, H. J. & Kwon, H. H. Precipitation nowcasting using transformer-based generative models and transfer learning for improved disaster preparedness. International Journal of Applied Earth Observation and Geoinformation 132, 103962 (2024).

[15] Wan, J. et al. RadarDiT: An advanced radar echo extrapolation model for three gorges reservoir area via difusion transformer. Journal of Hydrology: Regional Studies 61, 102703 (2025).

[16] Niu, D. et al. CastDifuser: Cascaded latent difusion framework for highresolution precipitation nowcasting via multi source fusion. Neurocomputing 670, 132435 (2026).

[17] Ling, X., Li, C., Qin, F., Yang, P. & Huang, Y. RNDif: Rainfall nowcasting with Condition Difusion Model. Pattern Recognition 160, 111193 (2025).

[18] Gao, C., Luo, Y., Leng, L., Liu, J. & Zhang, T. Physics-Guided Latent Diffusion Model for Intensity- and Extremity-Aware VIL Nowcasting. IEEE Journal ofSelected Topics in Applied Earth Observations and Remote Sensing 19, 17715–17728 (2026).

[19] Sun, X. et al. Advancing Radar Echo Extrapolation With Hypergraph-Enhanced Latent Difusion Model. IEEE Transactions on Geoscience and Remote Sensing 63, 1–14 (2025).

[20] Zhao, Z., Dong, X., Wang, Y. & Hu, C. Advancing Realistic Precipitation Nowcasting With a Spatiotemporal Transformer-Based Denoising Difusion Model. IEEE Transactions on Geoscience and Remote Sensing 62, 1–15 (2024).

[21] Li, C. et al. Extreme Precipitation Nowcasting Using Multitask Latent Difusion Models. IEEE Transactions on Geoscience and Remote Sensing 63, 1–15 (2025).

[22] Geng, L., Min, J., Geng, H. & Zhuang, X. Forecastformer: Spatiotemporal Decoupled Transformer and Difusion Models for Severe Convective Storm Nowcasting. IEEE Transactions on Geoscience and Remote Sensing 63, 1–14 (2025).

[23] Ling, X., Duan, G., Li, C., Huang, T. & He, T. HybridFlow: A Hybrid Velocity Generation Framework for Precipitation Nowcasting. IEEE Transactions on Geoscience and Remote Sensing 64, 1–16 (2026).

[24] Zhang, W. et al. MS-EdgeCast: A Dual-Stage Framework With a Multiscale Convolutional Recurrent Network and Edge-Guided Difusion for Convective Storm Nowcasting. IEEE Transactions on Geoscience and Remote Sensing 64, 1–13 (2026).

[25] Li, C. et al. Precipitation Nowcasting Using Difusion Transformer With Causal Attention. IEEE Transactions on Geoscience and Remote Sensing 62, 1–16 (2024).

[26] Ling, X. et al. Spacetime Separable Latent Difusion Model With Intensity Structure Information for Precipitation Nowcasting. IEEE Transactions on Geoscience and Remote Sensing 62, 1–15 (2024). Conference Name: IEEE Transactions on Geoscience and Remote Sensing.

[27] Ling, X., Li, C., Qin, F., Zhu, L. & Huang, Y. Two-Stage Rainfall-Forecasting Difusion Model. IEEE Geoscience and Remote Sensing Letters 21, 1–5 (2024).

[28] Foo, S. Q. et al. STLDM: Spatio-temporal latent difusion model for precipitation nowcasting. Transactions on Machine Learning Research (2025).

[29] Heistermann, M., Jacobi, S. & Pfaf, T. Technical Note: An open source library for processing weather radar data (wradlib). Hydrology and Earth System Sciences 17, 863–871 (2013).

[30] Raspaud, M. et al. PyTroll: An Open-Source, Community-Driven Python Framework to Process Earth Observation Satellite Data. Bulletin of the American Meteorological Society 99, 1329–1336 (2018).

[31] Paszke, A. et al. PyTorch: an imperative style, high-performance deep learning library. In Proceedings of the 33rd International Conference on Neural Information Processing Systems, 721, 8026–8037 (Curran Associates Inc., Red Hook, NY, USA, 2019).

[32] Pulkkinen, S. et al. Pysteps: an open-source Python library for probabilistic precipitation nowcasting (v1.0). Geoscientific Model Development 12, 4185–4219 (2019).

[33] Horvitz, D. G. & Thompson, D. J. A Generalization of Sampling Without Replacement from a Finite Universe. Journal of the American Statistical Association 47, 663–685 (1952).

[34] Kahn, H. Use of diferent Monte Carlo sampling techniques (The RAND Corporation, Santa Monica, Calif, 1955). Open Library ID: OL20777605M.

[35] Shimodaira, H. Improving predictive inference under covariate shift by weighting the log-likelihood function. Journal of Statistical Planning and Inference 90, 227–244 (2000).

[36] Ravuri, S. et al. Skilful precipitation nowcasting using deep generative models of radar. Nature 597, 672–677 (2021). Number: 7878.

[37] Vahid Yousefnia, K., B¨olle, T., Z¨obisch, I. & Gerz, T. A machine-learning approach to thunderstorm forecasting through post-processing of simulation data. Quarterly Journal of the Royal Meteorological Society 150, 3495– 3510 (2024).