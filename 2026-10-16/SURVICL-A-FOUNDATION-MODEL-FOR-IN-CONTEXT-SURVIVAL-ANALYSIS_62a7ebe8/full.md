# SURVICL: A FOUNDATION MODEL FOR IN-CONTEXT SURVIVAL ANALYSIS

Julie Alberge ∗ Inria de Saclay France

Linus Bleistein EPFL Switzerland

Judith Abecassis´ Inria de Saclay France

Marine Le Morvan Inria de Saclay France

Gael Varoquaux ¨ Inria de Saclay, :probabl. France

David Holzmuller¨ Inria de Saclay France

## ABSTRACT

Tabular foundation models (TFMs) now outperform gradient-boosted trees in classification and regression. Yet in survival analysis, which predicts when an event occurs from censored data, often on small clinical cohorts, we find that no existing TFM beats the half-century-old Cox model. We address this issue by introducing SurvICL, a TFM that predicts survival curves in-context. SurvICL is pretrained on purely synthetic datasets generated from a novel flexible proportional-odds prior that outperforms other prior types in our ablation study. Unlike most other survival models, SurvICL is trained with a plain quantile regression loss, which is possible because its synthetic pre-training data generator can provide uncensored event times. Additionally, we introduce the largest survival benchmark to date spanning 103 datasets. On the 63 datasets held out from all design decisions, SurvICL outperforms 20 tree-based, deep, and foundation-model baselines without any tuning: it wins 82.9% of pairwise comparisons in integrated Brier score, against 71.3% for the runner-up, the Cox model, and is the first model to rank above Cox on small datasets.

## 1 INTRODUCTION

Many decisions hinge on when an event will happen: how long will a patient survive after a cancer diagnosis? When will a customer churn? When will a machine fail? Survival analysis answers such questions by predicting, for an individual with features $\pmb { x } \in \mathbb { R } ^ { d }$ , the survival function $S ( t | x ) =$ $\tilde { \mathbb { P } } ( T ^ { \star } > t | { \bf \dot { x } } )$ , that is, the probability that the event has not occurred by time t (Kleinbaum and Klein, 2012; Machin et al., 2006; Wiegrebe et al., 2024; Jeanselme et al., 2024). What sets survival analysis apart from regression, aside from strictly positive predictions, is censoring. A patient who leaves a clinical study alive after three years does not reveal their time of death, only that it exceeds three years. Discarding such partial observations biases predictions, and treating them as observed events is plainly wrong: a survival model must learn from both observed and censored events. In addition, survival predictions feed decisions (such as choosing a treatment given a five-year survival probability) and must therefore be calibrated probabilities rather than mere rankings of individuals (Goldstein et al., 2020; Lillelund et al., 2025; Alberge et al., 2026).

Survival analysis is often a small-data problem. Many time-to-event datasets come from clinical studies, where observing events requires following patients for years, and cohorts of a few hundred individuals are the norm (Indrayan and Mishra, 2021; O’Donnell et al., 2025): in all the data we collect for this work, more than 70% of the datasets have fewer than 1,000 samples. In this regime, the flexibility of modern machine learning is a liability. Half a century after its introduction, the Cox proportional hazards model (Cox, 1972) remains the workhorse of practitioners, and for good reason: on the small datasets of our benchmark, none of the gradient-boosted trees, deep survival networks, or existing survival foundation models outperforms it.

![](images/949933f70e89a3db6b96ee3c399dd5062cfa6a8fd40525e2c86f7b481b7fb575.jpg)  
Figure 1: SurvICL is the best survival model on 63 held-out datasets. Pairwise win rate in integrated Brier score (IBS): the fraction of (dataset, competitor) pairs on which a method achieves a lower IBS. SurvICL wins 82.9% of its comparisons, against 71.3% for the runner-up, the classical Cox model. All existing survival TFMs trail behind the Cox model.

Tabular foundation models (TFMs) seem very well suited to this regime. Pre-trained on many synthetic datasets, models such as TabPFN and TabICL predict in-context and now outperform gradientboosted trees for classification and regression, in particular on small datasets (Hollmann et al., 2023; 2025; Grinsztajn et al., 2025; Qu et al., 2025; 2026; Zhang et al., 2025). This recipe has been extended beyond standard supervised learning to causal effect estimation (Robertson et al., 2025; Balazadeh et al., 2025; Ma et al., 2025b). Survival analysis has not yet benefited from this progress. Recent survival TFMs either recast survival as classification over a time grid for off-the-shelf TFMs (Kim et al., 2026), or are pre-trained on synthetic survival data (Seletkov et al., 2026; Bohm et al.¨ , 2026; Qi et al., 2026) but do not beat the Cox model or random survival forests (Figure 1).

Building a High-Performing TFM for Survival Analysis. We introduce SurvICL, a tabular foundation model for in-context survival analysis, built on the TabICLv2 backbone (Qu et al., 2026). Similar to SurvivalPFN (Qi et al., 2026), it handles censoring in the data generation rather than the loss. During pre-training, SurvICL sees censored datasets in its context, but is supervised with the latent, uncensored event times of the simulator: it learns to correct for censoring in-context, so that a plain quantile-regression loss suffices. SurvICL predicts for each individual a continuous quantile function of the event time, from which we can directly infer a monotone survival function.

Scaling Evaluation of Survival Models. We evaluate SurvICL in the largest benchmark to date for survival analysis: 103 real-world datasets from healthcare, genomics, economics and engineering, with up to 52,000 samples and 15,000 features, against 20 baselines spanning classical, tree-based, deep and foundation models. For comparison, survival papers at machine-learning venues typically report results on 5 to 20 datasets, and the largest evaluations of survival foundation models span 48 to 81 datasets (Kim et al., 2026; Lyu et al., 2026; Pham et al., 2026a; Qi et al., 2026). All design choices were made on 40 development datasets, and all results are reported on 63 held-out datasets never used during development. On these held-out datasets, SurvICL is the best method both in prediction accuracy and in discrimination (Figure 1): it wins 82.9% of pairwise comparisons for the integrated Brier score (IBS), against 71.3% for the runner-up, the Cox model, and in 77.7% for the C-index, against 71.6% for random survival forests (see Figure 12). On small datasets, where no previous method surpasses the Cox model, SurvICL is the first to rank above it. On the largest datasets, gradient boosting still maintains a slight edge. Importantly, these gains do not come at the expense of computational efficiency. Across predictive metrics and dataset-size regimes, SurvICL lies on or close to the performance-runtime Pareto frontier (Appendix D). Thus, SurvICL combines strong predictive performance with efficient out-of-the-box inference. We commit to releasing code and model in an open-source license.

## Contributions. Our contributions are threefold:

1. We design a foundation model for survival analysis that, using in-context learning, outperforms survival-analysis models (classical, tree-based, deep, all three existing survival foundation models as well as an adaptation of a classification foundation model), on integrated Brier score (IBS) and C-index.

2. We identify key ingredients of in-context survival analysis through extensive ablations: (i) a proportional-odds (PO) prior with a flexible, multi-modal baseline hazard, which outperforms

Cox, accelerated failure time, and existing TFM priors; and (ii) event-only targets, rather than also learning the censoring distribution.

3. An extensive benchmark of 103 survival datasets with a strict separation between development and held-out datasets, covering high-dimensional genomics cohorts $( p > n )$ and large registries, and comparing 21 methods on four metrics.

## 2 RELATED WORK

Deep Learning for Survival Analysis. The toolbox of survival analysis has been considerably extended in the last years to include deep architectures (Katzman et al., 2018; Lee et al., 2018; Kvamme et al., 2019) that can accommodate a wide variety of data and event types, including time series, competing events, images and text (Zhu et al., 2016; Thorsen-Meyer et al., 2022; Bleistein et al., 2024; Wiegrebe et al., 2024; Alberge et al., 2024; Jeanselme et al., 2024). Our work builds on this research by tailoring deep transformers and training procedures for tabular data, which is the most common form of survival data.

Deep Learning for Tabular Data. Gradient-based decision tree models have long dominated benchmarks on tabular data (Gorishniy et al., 2021; Grinsztajn et al., 2022; Ye et al., 2024). Tabular foundation models have recently challenged their superiority and sparked a new line of research that focused on designing tailored architectures and training procedures (Hollmann et al., 2023; 2025; Qu et al., 2025; 2026; Kim et al., 2025; Ding et al., 2025; Zhang et al., 2025). We build on these architectural advances and adapt them to calibrated survival prediction. Once pre-trained, our method performs in-context inference without fine-tuning or weight updates.

Tabular Foundation Models for Survival Analysis is a burgeoning field of research. Efforts follow three distinct strategies. (i) Reducing survival to a task that generic TFMs already solve. Kim et al. (2026) frame survival prediction as a series of binary classification problems over a time grid, so that any TFM that handles classification can be deployed for survival analysis. Vieyra (2026) instead plugs an off-the-shelf TFM regressor into a log-normal accelerated failure time model and impute censored times iteratively, while SurvFM (Lyu et al., 2026) converts censored follow-up into pseudoobservations of the restricted mean survival time (RMST), which are then regressed in-context by unmodified TFMs. These approaches inherit the strengths of generic TFMs, but their backbone has never seen censored data: they either depend on a time discretization, impose a parametric form on the event-time distribution, or only output a summary statistic (the RMST) rather than a full survival function. (ii) Adapting generic TFMs with a survival head. Pham et al. (2026b;a) train classical survival heads (Cox, DeepHit, multi-task logistic regression) on top of frozen or fine-tuned representations of TabPFN, TabDPT (Ma et al., 2025a), and TabICL, and systematically compare these adaptation interfaces on 74 datasets. They find that zero-shot reformulations work best on small datasets, whereas trained heads take over as the sample size grows. This line of work matches the common practice of fitting shallow survival models on top of deep foundation models, and gives up the pre-training + in-context learning recipe: the head must be fitted, and possibly tuned, on each new dataset. (iii) Pre-training on synthetic survival data. Closest to our work, Seletkov et al. (2026), Bohm et al.¨ (2026) and Qi et al. (2026) pre-train a tabular foundation model on synthetic survival data, following previous work in the field of tabular foundation models (Hollmann et al., 2023; Qu et al., 2025; 2026). All three models predict on a discretized time grid (see Table 1), while SurvICL outputs a continuous quantile function of the event time. Finally, foundation models for time-toevent prediction have also been pre-trained on large real-world electronic health records (Steinberg et al., 2024); such models are tied to a specific data modality and coding system, whereas we target generic tabular data through a synthetic prior.

## 3 SURVIVAL ANALYSIS SETTINGS: CENSORED TARGETS, SMALL SAMPLES

## 3.1 CENSORED DATA AND SURVIVAL MODELS

Survival Data. In survival analysis, data comes in the form of i.i.d. samples $\mathcal { D } _ { n } = \{ ( \boldsymbol { x } _ { i } , t _ { i } , \delta _ { i } ) \} _ { i = 1 } ^ { n }$ For each instance $i , \pmb { x } _ { i } \in \mathbb { R } ^ { d }$ denotes the feature vector and $t _ { i } > 0$ is a positive valued time. Unlike standard supervised learning, the target is subject to right-censoring, meaning that their true event time is not necessarily observed but might be censored, in which case we only observe the censoring time $c _ { i } > 0$ . This proxy time yields weak information on the true event time, since it will necessarily happen after censoring. We denote by $t _ { i } ^ { \star }$ the possibly unobserved true time of event, such that

$$
t _ { i } : = \operatorname* { m i n } \{ t _ { i } ^ { \star } , c _ { i } \} \quad \mathrm { a n d } \quad \delta _ { i } : = \mathbb { 1 } _ { t _ { i } ^ { \star } \leq c _ { i } } .
$$

Predicting with Survival Models. Our goal is to approximate the survival function

$$
S ( t | x _ { i } ) : = \mathbb { P } ( T ^ { \star } > t | X = x _ { i } )
$$

at any time $t > 0 .$ This stands in contrast to classification or regression, where the model’s goal is to output a value $\widehat { y } _ { i }$ which approximates the binary or continuous outcome $y _ { i } .$ . The survival function can be used to predict the time of event $\hat { t } _ { i }$ for individual i as the first time it crosses the threshold 0.5.

Hazard-based Models. Following the seminal work of Cox (1972), an extensive family of models parametrized the hazard of the survival process, the time-wise derivative of the event probability:

Hazard:

$$
h ( t | x _ { i } ) : = \operatorname* { l i m } _ { \varepsilon \to 0 } { \frac { \mathbb { P } ( T ^ { \star } \in [ t , t + \varepsilon ) | X = x _ { i } , T ^ { \star } \geq t ) } { \varepsilon } } \quad = - \partial _ { t } \log S ( t | x _ { i } )
$$

and trains it via (partial) likelihood maximization (Friedman, 1982; Fornili et al., 2013; Katzman et al., 2018; Kvamme et al., 2019; Kvamme and Borgan, 2021; Bleistein et al., 2024). Once the hazard is learned, it can be used to predict the survival function using numerical integration.

Discrete-Time Models. On the other side, some models bin time to simplify the computation of the model’s likelihood and frame survival analysis as a classification task (Yu et al., 2011; Fotso, 2018; Lee et al., 2018; Kim et al., 2026). More precisely, in their approach, time is discretized into $B + 1$ bins $( [ s _ { k } , s _ { k + 1 } [ ) _ { k = 0 } ^ { B }$ for $0 = s _ { 0 } < \cdot \cdot \cdot < s _ { B } = + \infty$ . The tuple $( { \pmb x } _ { i } , t _ { i } , \delta _ { i } )$ is then expanded into a series of points by creating a series of binary outcomes $y _ { i , k } : = \mathbb { 1 } _ { t _ { i } \leq s _ { k } }$ and considering the expanded dataset $\{ ( \boldsymbol { { \mathbf { x } } } _ { i } , s _ { k } ) , y _ { i , k } \} _ { i = 1 } ^ { n }$ . For individuals who experience the event, $B$ points are created, while for censored individuals only the bins prior to the censoring time are added to the expanded dataset. This approach is the dominant approach in survival TFMs (see Table 1). While flexible and easy to train, unconstrained formulations of these models can output non-monotonic survival functions, requiring post-hoc workarounds such as clipping probabilities (Kim et al., 2026).

## 3.2 DESIGN GOALS FOR A SURVIVAL TFM

Performance on Small Datasets. Small datasets are ubiquitous in survival analysis (O’Donnell et al., 2025), notably because time-to-event data is expensive and time-consuming to collect (to analyze event times that span over N years, one first needs to wait for N years) and because datasets often originate from healthcare (Indrayan and Mishra, 2021). State-of-the-art models perform well on large samples $( n \geq 2 0 0 0 )$ , but struggle on small datasets (Zhao et al., 2026).

Out-of-the-Box Calibration. In survival analysis, accurate probability estimates are crucial for decision-making (Goldstein et al., 2020; Alberge et al., 2026). Yet, models are often trained and evaluated chasing the C-index, which has been criticized as it does not assess calibration and is not a proper metric (not minimized for true probabilities, Lillelund et al., 2025). We train SurvICL to provide calibrated predictions without hyperparameter tuning.

## 4 SURVICL ARCHITECTURE: QUANTILES SUPERVISED BY LATENT TIMES

The TabICLv2 Backbone. Our architecture inherits the major components of TabICLv2 (Qu et al., 2026). In particular, we use repeated feature grouping, which bags groups of features before tokenizing them and target-aware embedding. We adapt the original architecture of Qu et al. (2026) to handle triplets as inputs, which correspond to features, censoring indicators and observed times. More details on TabICLv2 and our modifications can be found in Appendix A.1.

Continuous Quantile Head. To obtain calibrated and accurate predictions, we use a multiquantile pinball loss following TabICLv2. Given an input $\textbf { \em x } \in \mathbb { R } ^ { \bar { d } }$ , our model outputs ${ \hat { \pmb q } } ( { \pmb x } ) =$ $\hat { ( } \hat { q } _ { \tau } ( \pmb { x } ) ) _ { \tau \in \mathcal { T } } \in [ 0 , + \infty ) ^ { M - 1 }$ , whose entries are the predicted quantiles of the event time at the levels $\dot { \mathcal { T } } \dot { = } \{ 1 / \tilde { M } , 2 / \dot { M } , \dots , ( M - 1 ) / M \}$ . We then compute

$$
\mathcal { L } ( \hat { q } ( x ) , t ^ { \star } ) : = \frac { 1 } { M - 1 } \sum _ { \tau \in T } \ell _ { \tau } \big ( \hat { q } _ { \tau } ( x ) , t ^ { \star } \big ) \mathrm { ~ w h e r e ~ } \ell _ { \tau } ( \hat { q } , t ^ { \star } ) : = \left\{ \begin{array} { l l } { \tau \left( t ^ { \star } - \hat { q } \right) } & { \mathrm { i f ~ } t ^ { \star } \geq \hat { q } } \\ { ( 1 - \tau ) \left( \hat { q } - t ^ { \star } \right) } & { \mathrm { i f ~ } t ^ { \star } < \hat { q } } \end{array} \right. ,
$$

![](images/981ceef067525335e673510de9d49e161636068ffec38161138b2c98bfa3a777.jpg)  
Figure 2: (a) Overview over our synthetic data prior: building on Qu et al. (2026), we construct a pipeline to simulate survival datasets by generating an event time and censoring time distribution. (b) Three example datasets drawn from our prior, which illustrate the high diversity in observed event time and censoring time distributions.

where $t ^ { \star }$ is the latent event time drawn from the prior (see below). We use $M = 1 0 0 0 .$ , i.e., 999 quantile levels as in TabICLv2. To ensure monotonicity, we sort the predicted quantiles except during pre-training. Experiments with other numbers of percentiles are reported in Figure 4. This loss is one of the major design choices that drive the capabilities of our model, and distinguishes it from previous models, which use classification losses or the hybrid loss of Lee et al. (2018).

Latent Time Supervision. The target $t ^ { \star }$ of the pinball loss is never observed in real data: it is the latent event time generated by our prior (Section 5). During pre-training, the context only contains censored observations $( { \pmb x } _ { i } , t _ { i } , \delta _ { i } )$ , while the loss is computed on the uncensored $t _ { i } ^ { \star }$ of the query points. The model thus learns to infer the event-time distribution from censored data in-context. The survival-specific knowledge lives in the prior, so that the above regression loss is valid: we need neither a censored likelihood, nor inverse-probability-of-censoring weights, nor a discretization of time. SurvivalPFN (Qi et al., 2026) also supervises with latent times, but over a discretized time grid and with censoring times as an additional target, which we find to be detrimental (Section 6).

Mean-Duration Scaling. We rescale time, to cater for the wide variety of time range across datasets.   
Our strategy is to divide survival times by the mean observed duration.

Event-Only Targets. In survival analysis, censoring times are often considered to be hurdles to efficient estimation. Interestingly, Qi et al. (2026) use the censored times as an alternative target: their model learns to approximate two distributions (one for censored times and one for event times) by predicting the event and censoring times during pre-training. We ablate this mixed prediction task and show that it degrades the performance of our model.

## 5 PRETRAINING: SYNTHETIC DATA PRIORS WITH PROPORTIONAL-ODDS

## A key novelty of our work is our synthetic data prior, illustrated in Figure 2.

Proportional-Odds (PO) Prior. We start by sampling numerical and categorical features $( { \pmb x } _ { i } ) _ { i = 1 } ^ { n }$ using the structural causal prior of Qu et al. (2026). The number of nodes (from 2 to 32) and features (from 1 to 100) span a wide range to allow for diverse datasets. We then sample survival times from a novel proportional-odds prior. Like the Cox model, it combines a baseline with an individual effect $\eta ( \pmb { x } _ { i } )$ , but on the odds rather than the hazard scale: hazard ratios then vary over time and may vanish in the long run, a common pattern in clinical data that Cox priors cannot generate. For the baseline hazard, we set

$$
h _ { 0 } ( t ) = a \mathrm { ~ s o f t p l u s } \big [ f ( t / t _ { \mathrm { m a x } } ) \big ] + \sum _ { k = 1 } ^ { K } A _ { k } \exp \biggl ( - \frac { ( t - \mu _ { k } ) ^ { 2 } } { 2 \sigma _ { k } ^ { 2 } } \biggr ) ,
$$

with $a , ( A _ { k } ) _ { k = 1 } ^ { K } , ( \mu _ { k } ) _ { k = 1 } ^ { K }$ and $( \sigma _ { k } ) _ { k = 1 } ^ { K }$ positive values, and $t _ { \mathrm { m a x } } > 0$ . The parameter $K$ is taken randomly between 1 and 5. This parametrization adds a random number of positive Gaussian bumps at random times, each with its own width and height, directly on top of the smooth trend. We then integrate this baseline hazard to obtain $\begin{array} { r } { S _ { 0 } ( t ) = \exp \big ( - \int _ { 0 } ^ { t } h _ { 0 } ( u ) d u \big ) } \end{array}$ . To generate the individual specific part of the hazard function, we apply a random function $\eta ( \pmb { x } _ { i } )$ to the features as in Qu et al. (2026). We then define the individual-specific survival function as

$$
S ( t | x _ { i } ) : = \frac { 1 } { \frac { 1 - S _ { 0 } ( t ) } { S _ { 0 } ( t ) } \exp ( \eta ( { x } _ { i } ) ) + 1 } .
$$

Setting $\eta ( { \pmb x } _ { i } ) = 0$ recovers the baseline hazard, while positive values increase the odds of experiencing the event by any time. This prior draws on a long line of work on mixture models in survival analysis (McLachlan and McGiffin, 1994; De Angelis et al., 1999; Li et al., 2024). We sample event times by inverting the cumulative distribution function, with $U \sim \mathcal { U } ( [ 0 , 1 ] )$

$$
t _ { i } ^ { \star } = H _ { 0 } ^ { - 1 } \Big ( \log \big [ 1 + \exp ( - \eta ( \pmb { x } _ { i } ) ) ( U ^ { - 1 } - 1 ) \big ] \Big )
$$

with $H _ { 0 } ( t ) : = - \log S _ { 0 } ( t )$ . We draw 5 times per individual to speed up convergence.

Generating Censoring Times. We add two sources of censoring to our model. First, in 50% of datasets, we apply a shared end-of-study cutoff to a randomly sampled set of individuals. Secondly, we generate a concurrent censoring time distribution by sampling another pair of baseline and individual specific hazards and combining them as for the event time distribution. We then sample from this second distribution, and set the observed time to be the minimum between the two. Lastly, to introduce tied times, we move some individual times to a common time grid with low probability.

Filtering. We remove datasets where more than 95% of samples are censored, as well as nonpredictable datasets, adapting the filter of Qu et al. (2026): a dataset is non-predictable if predicting the mean survival time achieves a lower IBS (Section 6) than an ExtraTrees-based survival model (Geurts et al., 2006). Every pretraining dataset thus carries exploitable signal.

Training. During training, we split every generated dataset into a context (train) set $\mathcal { D } _ { c }$ of size $n _ { c }$ and a query (test) set $\mathcal { D } _ { q }$ of size $n _ { \ell }$ with $n = n _ { c } + n _ { \ell }$ . We then compute and take the gradients of

$$
\frac { 1 } { n _ { \ell } } \sum _ { i \in \mathcal { D } _ { q } } \mathcal { L } \big ( f _ { \theta } ( \pmb { x } _ { i } , \mathcal { D } _ { c } ) , t _ { i } ^ { \star } \big ) ,
$$

where $f _ { \theta } ( \pmb { x } _ { i } , \pmb { D } _ { c } )$ denotes the quantiles predicted from the context $\mathcal { D } _ { c } \mathbf { ; }$ the loss is computed on the query set only. The context size varies during pre-training. We adopt the pre-training setup from Qu et al. (2026) and only use stage 1 for 200K steps, which takes around 72 hours on two H100 GPUs.

## 6 LARGE SCALE EVALUATION ON 63 DATASETS

## 6.1 SURVIVAL METRICS

We evaluate our models on four different metrics to overcome the pitfalls of single-metric evaluation (Lillelund et al., 2025). In the following, $N \geq 1$ denotes the size of the evaluation dataset. All metrics are computed using the SurvEval package (Qi et al., 2023)

Integrated Brier Score (IBS). The IBS is computed by integrating the Brier score at time $t > 0$ which is equal for an individual to the difference between the predicted survival curve and the observed binary survival indicator function, over time. It is hence equal to

$$
\frac { 1 } { N _ { t e s t } t _ { \mathrm { m a x } } } \sum _ { i = 1 } ^ { N } \int _ { 0 } ^ { t _ { \mathrm { m a x } } } \mathrm { B S } _ { i } ( t ) d t \quad \mathrm { ~ w h e r e ~ } \mathrm { B S } _ { i } ( t ) : = \frac { \widehat { S } ( t \mid x _ { i } ) ^ { 2 } \mathbb { I } _ { t _ { i } \leq t , \delta _ { i } = 1 } } { \widehat { G } ( t _ { i } ) } + \frac { ( 1 - \widehat { S } ( t \mid x _ { i } ) ) ^ { 2 } \mathbb { I } _ { t _ { i } > t } } { \widehat { G } ( t ) } .
$$

Unlike the C-index, the IBS measures the accuracy of predicted probabilities, not just rankings.

C-index at the 50th Quantile. We evaluate discrimination using an IPCW concordance index truncated at the median observed training follow-up time $t _ { 5 0 } = \mathrm { Q } \mathrm { { \bar { u a n t i l e } _ { 0 . 5 } \{ \mathit { t } _ { i } ^ { t r a i n } \} } }$ , where $t _ { i } ^ { \mathrm { t } }$ rain includes both event and censoring times. For each test individual, we use the predicted risk $r _ { i } =$ $1 - \widehat { S } ( t _ { 5 0 } | \mathbf { \mathscr { x } } _ { i } )$ . Concordance is computed over comparable pairs with an observed event before t<sub>50</sub> band weighted by $\widehat { G } ( t _ { i } ) ^ { - 2 }$ , where $\widehat { G }$ is the Kaplan–Meier estimate of the censoring survival function b bfitted on the training data. This corresponds to a truncated IPCW concordance estimator.

![](images/ab12990fbacdc5865a4fcd742dba68b6d1c6b8648a8a2af5721509a2464adf7b.jpg)

![](images/accf08421df9d5690f5682b7ae8951158c49afbe9aa5eb7bee8486b0977dce31.jpg)  
Figure 3: Overview of (a) the held-out datasets and (b) a sample of 500 pre-training datasets. In (b), the sample size is replaced by the context (train) set size (see Section 5).

MAE-Hinge. We use absolute error for event times and a one-sided hinge error for censored times:

$$
\mathrm { M A E } _ { \mathrm { H i n g e } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left[ \delta _ { i } | t _ { i } - \hat { t } _ { i } | + \left( 1 - \delta _ { i } \right) \operatorname* { m a x } ( 0 , t _ { i } - \hat { t } _ { i } ) \right] ,
$$

where $\delta _ { i }$ indicates an observed event and $\hat { t } _ { i }$ is the predicted median survival time. For censored individuals, only predictions earlier than the censoring time are penalized.

KM-Calibration. We assess marginal calibration with the normalized integrated squared error, in [0, 1], between the average predicted survival curve $\begin{array} { r } { \frac { 1 } { N _ { t e s t } } \sum _ { i } \widehat { S } ( t | \mathbf { x } _ { i } ) } \end{array}$ and the Kaplan–Meier curve of the test set (Chapfuwa et al., 2020).

## 6.2 BASELINES: FROM COX TO FOUNDATION MODELS

All models are described in more detail in Appendix B.1, we refer to Table 3 for a synthetic overview.

Shallow models. We include Cox model (Cox, 1972) and KM estimator (Kaplan and Meier, 1958) as baselines. The latter does not use covariates, and hence any model is expected to outperform it.

Deep Networks. We include five deep networks tailored for survival analysis (Lee et al., 2018; Katzman et al., 2018; Nagpal et al., 2021; Rindt et al., 2022; Kirpichenko et al., 2026). These models are fitted on each dataset and use various modeling techniques (from Bayesian prior to a MLP that predicts the survival function at different times).

Tree-Based Models. Following the seminal work of Ishwaran et al. (2008) on random survival forests (RSF) and Grinsztajn et al. (2022), a body of work has adapted tree-based methods to survival analysis. On top of RSF, we evaluate SurvivalBoost (Alberge et al., 2024) and alternative gradientboosted tree-based methods implemented in sksurv (Polsterl¨ , 2020).

Tabular Foundation Models (TFMs). We include four open-weight tabular foundation models in our benchmark: SurvPFN (Bohm et al.¨ , 2026), SurvivalPFN (Qi et al., 2026), Survival-IC (SIC) (Seletkov et al., 2026) and TFM4Survival (Kim et al., 2026).

## 6.3 DATASETS AND HOLD-OUT PROTOCOL

In total, we consider 103 datasets with varying properties, see Figure 3. The datasets are partitioned into 40 development datasets used during model development and 63 held-out datasets at random. For development, we source 2 PyCox (Kvamme et al., 2019), 25 SurvSet (Drysdale, 2022), one from a challenge and 12 cBioPortal (Cerami et al., 2012; Gao et al., 2013; De Bruijn et al., 2023) datasets. For held-out datasets, we use 6 from Pycox, 39 from SurvSet and 18 datasets from cBioPortal.

Protocol. All design choices of SurvICL (prior, target, and checkpoint) were made using the 40 development datasets; the 63 held-out datasets were kept for the final evaluation. We evaluated each method over ten event-stratified 80/20 train–test splits (test sets capped at 4,096 observations).

![](images/a7aa1a9ab4fb02722bf5e4f6732bfa9237e6acddd3080cd3589c58f2856638ca.jpg)

![](images/ef2e7a9ff313e005f723a0848cbfe53b6f6b4bb5be29ed02465ba49654853a67.jpg)

(c) Ablations  
![](images/f3e230f2666ca2149035062897bc5df3ada39f01bd331107e4f6ad566c2b8e80.jpg)  
Figure 4: (a, b) Critical difference diagrams of average ranks on all held-out datasets (lower is better), for IBS and C-index at the median horizon $t _ { 5 0 } ;$ methods joined by a bar are not significantly different. (c) Ablations of prior and training choices, by min–max normalized IBS (higher is better).

Baseline hyperparameters were selected using a budget of up to 50 random search steps per dataset, minimizing the integrated Brier score on an event-stratified 80/20 split within the seed-0 training partition. The selected configuration was reused across all ten evaluation splits, with each estimator refitted on the corresponding training split. Methods without tunable hyperparameters were evaluated directly. Search spaces and method-specific budgets are detailed in Appendix B.1. Cox regression uses an $\ell _ { 2 }$ penalty to regularize estimation, including when $p > n .$

SurvICL uses each dataset’s training split as context, with at most 4,096 labeled observations selected by event-stratified subsampling, without fine-tuning or dataset-specific hyperparameter tuning. Benchmark results for all methods and metrics are in Appendix D. Baseline runtimes measure fitting and survival-curve prediction, excluding hyperparameter search and metric computation.

## 6.4 RESULTS: SURVICL IS BEST OVERALL

SurvICL is the best survival model overall. On the 63 held-out datasets, SurvICL wins 82.9% of pairwise comparisons in IBS, against 71.3% for the runner-up, the Cox model (Figure 1), and the difference is statistically significant (Figure 4a, ranks and winrates are related by an affine transform). It also achieves the highest win rate in C-index, 77.7% against 71.6% for random survival forests (Figures 4b and 12). All existing survival TFMs trail behind classical methods. That the runners-up are the Cox model and random survival forests, with few parameters and strong inductive biases, underlines how hard survival analysis is for flexible models, which overfit small censored datasets.

SurvICL is the first model to rank above Cox on small datasets. 69.8% of our held-out datasets have fewer than 1,000 samples, making this regime central and not a niche. SurvICL is the first to rank above Cox in IBS (win rate 84.9% against 77.2%, Figure 23) and it matches Cox in C-index, although these differences are not statistically significant (Figures 22 and 26).

On large datasets, boosting remains strongest. On held-out datasets with more than 1,000 samples, SurvivalBoost is clearly the best method in IBS (Figure 37, 86.9% winrate). SurvICL ranks second in average IBS and first in C-index 50th quantile, ahead of TabSurv-WSA, RSF, and all other TFMs, although it wins fewer pairwise comparisons than on small datasets (Figure 38).

<table><tr><td>Model</td><td>Loss</td><td>Continuous Time</td><td>Prior</td><td>Prediction Target</td></tr><tr><td>SurvPFN</td><td>NLL + ranking</td><td>√</td><td>Weibull</td><td>Observed times</td></tr><tr><td>SurvivalPFN</td><td>NLL</td><td>x</td><td>Random MLP</td><td>Latent event + censoring times</td></tr><tr><td>Survival-IC</td><td>DeepHit Loss</td><td>x</td><td>EH Model</td><td>Observed times</td></tr><tr><td>TFM4Survival</td><td>BCE</td><td>x</td><td>X</td><td>Binary survival labels</td></tr><tr><td>SurvICL (ours)</td><td>Pinball Loss</td><td>√</td><td>PO prior</td><td>Latent event times</td></tr></table>

Table 1: Major differences between existing survival TFMs and SurvICL. “Latent” targets are un censored times generated by the prior, unobserved in real data.

Performance-runtime trade-off. SurvICL lies on the Pareto frontier across C-index and IBS for all dataset-size regimes (Appendix D), and on the frontier for MAE in the small regime and all held-out datasets. SurvICL combines strong predictive performance with competitive runtime.

Calibration and time accuracy. SurvICL is among the best-calibrated methods: its average KMcalibration rank (6.0) trails only the KM estimator (4.3), the Cox model (5.0), DeepSurv (5.6) and sksurv GBS-Cox (5.9) (Figure 15). Both KM and the Breslow baseline hazard of the Cox model enforce calibration by construction. In MAE, SurvICL is close to Survival-IC and slightly behind SurvivalPFN, without statistically significant difference (Figure 18).

What makes SurvICL work. Figure 4c ablates design choices on development datasets with a shallower model trained for 10K steps, cf. Appendix C for more details. The prior has a substantial impact on performance: by keeping the exact same architecture and only changing the generation process, our proportional-odds prior outperforms Cox (77.6) and mixed priors (84.0). Compared to the prior of the other TFMs, our prior yields better results than SurvPFN (79.0), SurvivalPFN (82.7), and SIC (83.7). This demonstrates that our prior is highly effective, while adding a censoring-time prediction head actually reduces performance (81.0). Administrative censoring and time discretization have limited effects at this scale; we retain them to represent study-end censoring and finite temporal resolution. The final model uses 999 quantiles for finer resolution, despite a small advan tage for 99 in the ablation (88.5 versus 87.2).

## 7 DISCUSSION AND CONCLUSION

Designing Efficient Survival TFMs. SurvICL differs from existing survival TFMs along three axes (Table 1). Output and supervision. Unlike previous survival TFMs that rely on discrete time bins such as TFM4Survival (Kim et al., 2026) and SurvivalPFN (Qi et al., 2026), which use classification or likelihood losses over time bins, SurvPFN (Bohm et al.¨ , 2026) uses a censored likelihood with a ranking term over 1,000 bins, and Survival-IC (Seletkov et al., 2026) uses the DeepHit loss (Lee et al., 2018) over 10 bins. SurvICL is the first to output a continuous quantile function. Quantile outputs are not new per se: TabICLv2 (Qu et al., 2026) uses a pinball loss for regression, and censored quantile regression (Portnoy, 2003; Pearce et al., 2022) modifies this loss to account for censoring. With latent-time supervision, no such modification is needed, as censoring is handled by the prior. SurvivalPFN also supervises with latent times, but over a time grid and with censoring times as an additional target, which our ablations show to be detrimental. Prior. SurvivalPFN draws times from random noised neural networks with several censoring mechanisms, SurvPFN draws Weibull times from a risk score and a dataset-specific shape, and Survival-IC mixes an extended hazard and a discrete-time model, with covariate-independent and administrative censoring. Our PO prior instead combines a flexible, multi-modal baseline hazard with a random risk function, for both event and censoring times; in our ablations, it outperforms the SurvPFN prior as well as Cox and AFT priors. Backbone and evaluation. Like Survival-IC, built on TabICL (Qu et al., 2025), we reuse an existing architecture, TabICLv2, which is not our contribution. Finally, we provide, to our knowledge, the first comparison of all four survival TFMs, on 63 datasets.

Implications. On the small cohorts that dominate clinical research, SurvICL is the first model to rank above the Cox model, the default of practitioners for half a century, without any training or tuning, while remaining among the best-calibrated methods. This makes foundation models a credible default for survival analysis.

Limitations and outlook. On the largest datasets, gradient boosting remains slightly stronger, calling for longer contexts or light fine-tuning. Baseline hyperparameters are tuned on the first split and reused across splits; as tuning data then overlaps other splits’ test sets, this protocol, if anything, favors the baselines. SurvICL addresses right-censored, single-event data; competing risks and time-varying covariates are natural extensions of the prior. We release our code and models to foster research on foundation models for time-to-event data.

## ACKNOWLEDGEMENTS

LB and JAl thank Julie Josse and Charlotte Voinot for helpful discussions on survival analysis. We thank Felix Lefebvre and Farouk Yartaoui for inspiring our figure design.´

Part of this research was conducted while LB was at EPFL (Schools of Computer and Communication Sciences and School of Life Sciences). His research was funded by an EPFL AI Center Postdoctoral Fellowship. DH and JAl acknowledge HPC resources from GENCI–IDRIS (Grant 2024-AD011014864R2 and AD011017641 respectively). DH thanks Probabl for providing Claude API access. JAl, JAb, and GV acknowledge funding from the European Union’s Horizon Europe research and innovation program under grant agreement No 1010954433 (Intercept-T2D). GV acknowledges support from ANR via grant TaFoMo (ANR-25-CE23-1822). MLM acknowledges support from ANR via the grant StatQA (ANR-25-CE23-5646). This work is partly supported by Hi! PARIS and ANR/France 2030 program (ANR-23-IACL-0005).

## AI USE STATEMENT

In this work, we used generative AI tools to generate synthetic data sets, design or provide feedback on research methodology or experiments, implement methods, draft preliminary plots and provide feedback on research methodology or experiments. We have not used generative AI tools to help develop theoretical models or conceptual frameworks, formulate mathematical claims, assist with translation, clean and reformat datasets, support qualitative and thematic data analysis, and interpret results. The categories provide critical ingredients for proving mathematical claims, assist in the writing of proofs, propose or refine hypotheses are not applicable to this work.

Additionally, we used generative AI tools to create or modify scientific figures or images, suggest experimental parameters, create or edit software code, and to brainstorm and source and search for information.

We have reviewed all AI-assisted work. The paper has been proofread by all authors. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

The codebase, dataset preprocessing files, and model checkpoints will be released in a few weeks.

## REFERENCES

Julie Alberge, Vincent Maladiere, Olivier Grisel, Judith Ab\` ecassis, and Ga´ el Varoquaux. Survival¨ models: Proper scoring rule and stochastic optimization with competing risks. arXiv preprint arXiv:2410.16765, 2024.

Julie Alberge, Tristan Haugomat, Gael Varoquaux, and Judith Ab¨ ecassis. On the calibration of´ survival models with competing risks. arXiv preprint arXiv:2602.00194, 2026.

Vahid Balazadeh, Hamidreza Kamkari, Valentin Thomas, Benson Li, Junwei Ma, Jesse C. Cresswell, and Rahul G. Krishnan. CausalPFN: Amortized causal effect estimation via in-context learning. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Linus Bleistein, Van-Tuan Nguyen, Adeline Fermanian, and Agathe Guilloux. Dynamical survival analysis with controlled latent states. arXiv preprint arXiv:2401.17077, 2024.

Samuel Bohm, Lennart Purucker, Frank Hutter, and Pascal Schlosser. Survpfn: Towards foundation¨ models for survival predictions. arXiv preprint arXiv:2606.04564, 2026.

Ethan Cerami, Jianjiong Gao, Ugur Dogrusoz, Benjamin E Gross, Selcuk Onur Sumer, Bulent Ar-¨ man Aksoy, Anders Jacobsen, Caitlin J Byrne, Michael L Heuer, Erik Larsson, et al. The cbio cancer genomics portal: an open platform for exploring multidimensional cancer genomics data. Cancer discovery, 2(5):401–404, 2012.

Paidamoyo Chapfuwa, Chenyang Tao, Chunyuan Li, Irfan Khan, Karen J. Chandross, Michael J. Pencina, Lawrence Carin, and Ricardo Henao. Calibration and uncertainty in neural time-to-event modeling. IEEE Transactions on Neural Networks and Learning Systems, 2020.

David R Cox. Regression models and life-tables. Journal of the royal statistical society: Series B (methodological), 34(2):187–202, 1972.

R De Angelis, R Capocaccia, T Hakulinen, B Soderman, and A Verdecchia. Mixture models for cancer survival analysis: application to population-based data with covariates. Statistics in medicine, 18(4):441–454, 1999.

Ino De Bruijn, Ritika Kundra, Brooke Mastrogiacomo, Thinh Ngoc Tran, Luke Sikina, Tali Mazor, Xiang Li, Angelica Ochoa, Gaofei Zhao, Bryan Lai, et al. Analysis and visualization of longitudinal genomic and clinical data from the aacr project genie biopharma collaborative in cbioportal. Cancer research, 83(23):3861–3867, 2023.

Jiayuan Ding, Jianhui Lin, Shiyu Jiang, Yixin Wang, Ziyang Miao, Zhaoyu Fang, Jiliang Tang, Min Li, and Xiaojie Qiu. Tabula: A tabular self-supervised foundation model for single-cell transcriptomics. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

Erik Drysdale. SurvSet: An open-source time-to-event dataset repository. arXiv preprint arXiv:2203.03094, 2022.

Nick Erickson, Lennart Purucker, Andrej Tschalzev, David Holzmuller, Prateek Desai, David Sali-¨ nas, and Frank Hutter. TabArena: A living benchmark for machine learning on tabular data. Advances in Neural Information Processing Systems, 38, 2025.

Marco Fornili, Federico Ambrogi, Patrizia Boracchi, and Elia Biganzoli. Piecewise exponential artificial neural networks (peann) for modeling hazard function with right censored data. In International Meeting on Computational Intelligence Methodsfor Bioinformatics and Biostatistics, pages 125–136. Springer, 2013.

Stephane Fotso. Deep neural networks for survival analysis based on a multi-task framework. arXiv preprint arXiv:1801.05512, 2018.

Michael Friedman. Piecewise exponential models for survival data with covariates. The Annals of Statistics, 10(1):101–113, 1982.

Jianjiong Gao, Bulent Arman Aksoy, Ugur Dogrusoz, Gideon Dresdner, Benjamin Gross, S Onur¨ Sumer, Yichao Sun, Anders Jacobsen, Rileen Sinha, Erik Larsson, et al. Integrative analysis of complex cancer genomics and clinical profiles using the cbioportal. Science signaling, 6(269): pl1–pl1, 2013.

Pierre Geurts, Damien Ernst, and Louis Wehenkel. Extremely randomized trees. Machine learning, 63(1):3–42, 2006.

Mark Goldstein, Xintian Han, Aahlad Puli, Adler Perotte, and Rajesh Ranganath. X-cal: Explicit calibration for survival analysis. Advances in neural information processing systems, 33:18296– 18307, 2020.

Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning models for tabular data. Advances in neural information processing systems, 34:18932–18943, 2021.

Leo Grinsztajn, Edouard Oyallon, and Ga ´ el Varoquaux. Why do tree-based models still outperform ¨ deep learning on typical tabular data? Advances in neural information processing systems, 35: 507–520, 2022.

Leo Grinsztajn, Klemens Fl´ oge, Oscar Key, Felix Birkel, Philipp Jund, Brendan Roof, Benjamin¨ Jager, Dominik Safaric, Simone Alessi, Adrian Hayler, Mihir Manium, Rosen Yu, Felix Jablonski,¨ Shi Bin Hoo, Anurag Garg, Jake Robertson, Magnus Buhler, Vladyslav Moroshan, Lennart Pu-¨ rucker, Clara Cornu, Lilly Charlotte Wehrhahn, Alessandro Bonetto, Bernhard Scholkopf, Sauraj¨ Gambhir, Noah Hollmann, and Frank Hutter. Tabpfn-2.5: Advancing the state of the art in tabular foundation models, 2025. URL https://arxiv.org/abs/2511.08667.

Noah Hollmann, Samuel Muller, Katharina Eggensperger, and Frank Hutter. TabPFN: A transformer¨ that solves small tabular classification problems in a second. In International Conference on Learning Representations, 2023.

Noah Hollmann, Samuel Muller, Lennart Purucker, Arjun Krishnakumar, Max K¨ orfer, Shi Bin Hoo,¨ Robin Tibor Schirrmeister, and Frank Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 01 2025. doi: 10.1038/s41586-024-08328-6. URL https://www. nature.com/articles/s41586-024-08328-6.

A Indrayan and A Mishra. The importance of small samples in medical research. Journal of Postgraduate Medicine, 67(4):219–223, 2021.

Hemant Ishwaran, Udaya B. Kogalur, Eugene H. Blackstone, and Michael S. Lauer. Random survival forests. The Annals ofApplied Statistics, 2(3):841 – 860, 2008. doi: 10.1214/08-AOAS169. URL https://doi.org/10.1214/08-AOAS169.

Vincent Jeanselme, Nikita Agarwal, and Chen Wang. Review of language models for survival analysis. In AAAI 2024 Spring Symposium on Clinical Foundation Models, 2024.

Edward L Kaplan and Paul Meier. Nonparametric estimation from incomplete observations. Journal ofthe American statistical association, 53(282):457–481, 1958.

Jared L Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger. Deepsurv: personalized treatment recommender system using a cox proportional hazards deep neural network. BMC medical research methodology, 18(1):24, 2018.

Da In Kim, Wei Siang Lai, and Kelly W Zhang. Tabular foundation models can do survival analysis. arXiv preprint arXiv:2601.22259, 2026.

Myung Jun Kim, Felix Lefebvre, Ga ´ etan Brison, Alexandre Perez-Lebel, and Ga ¨ el Varoquaux. ¨ Table foundation models: on knowledge pre-training for tabular learning. arXiv preprint arXiv:2505.14415, 2025.

Stanislav Kirpichenko, Andrei Konstantinov, and Lev Utkin. Tabsurv: Adapting modern tabular neural networks to survival analysis. arXiv preprint arXiv:2605.03944, 2026.

David G. Kleinbaum and Mitchel Klein. Survival Analysis: A Self-Learning Text. Statistics for Biology and Health. Springer, New York, NY, 2012.

Havard Kvamme and Ørnulf Borgan. Continuous and discrete-time survival prediction with neural˚ networks. Lifetime data analysis, 27(4):710–736, 2021.

Havard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks˚ and cox regression. Journal ofmachine learning research, 20(129):1–30, 2019.

Changhee Lee, William Zame, Jinsung Yoon, and Mihaela Van Der Schaar. Deephit: A deep learning approach to survival analysis with competing risks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

Yunzhe Li, Juhee Lee, and Athanasios Kottas. Bayesian nonparametric erlang mixture modeling for survival analysis. Computational Statistics & Data Analysis, 191:107874, 2024.

Christian Marius Lillelund, Shi-ang Qi, Russell Greiner, and Christian Fischer Pedersen. Stop chasing the c-index: This is how we should evaluate our survival models. arXiv preprint arXiv:2506.02075, 2025.

Yue Lyu, Steven H. Lin, Xuelin Huang, and Ziyi Li. SurvFM enables tabular foundation models for right-censored survival prediction. arXiv preprint arXiv:2607.09577, 2026.

Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Alex Labach, Jesse Cresswell, Keyvan Golestan, Guangwei Yu, Anthony L Caterini, and Maks Volkovs. TabDPT: Scaling tabular foundation models on real data. In Advances in Neural Information Processing Systems, 2025a.

Yuchen Ma, Dennis Frauen, Emil Javurek, and Stefan Feuerriegel. Foundation models for causal inference via prior-data fitted networks. arXiv preprint arXiv:2506.10914, 2025b.

David Machin, Yin Bun Cheung, and Mahesh Parmar. Survival analysis: a practical approach. John Wiley & Sons, 2006.

Geoff J McLachlan and David C McGiffin. On the role of finite mixture models in survival analysis. Statistical methods in medical research, 3(3):211–226, 1994.

Chirag Nagpal, Xinyu Li, and Artur Dubrawski. Deep survival machines: Fully parametric survival regression and representation learning for censored data with competing risks. IEEE Journal of Biomedical and Health Informatics, 25(8):3163–3175, 2021.

Autumn O’Donnell, Michael Cronin, Shirin Moghaddam, and Eric Wolsztynski. A systematic review on machine learning techniques for survival analysis in cancer. Cancer Medicine, 14(22): e71375, 2025.

Tim Pearce, Jong-Hyeon Jeong, Yichen Jia, and Jun Zhu. Censored quantile regression neural networks for distribution-free survival analysis. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Minh-Khoi Pham, Luca Cotugno, D. Cernei, Alina Sˆırbu, S. Masi, G. Prencipe, A. Pingitore, P. Landi, Tai Tan Mai, Martin Crane, and Marija Bezbradica. Adaptation interfaces for in-context tabular foundation models in time-to-event prediction. arXiv preprint arXiv:2609.04901, 2026a.

Minh-Khoi Pham, Luca Cotugno, Alina Sˆırbu, Tai Tan Mai, Martin Crane, and Marija Bezbradica. Tabular foundation models for clinical survival analysis via survival-aware adaptation. In International Conference on AI in Healthcare (AIiH), 2026b. arXiv:2606.12006.

Sebastian Polsterl. scikit-survival: A library for time-to-event analysis built on top of scikit-learn.¨ Journal of Machine Learning Research, 21(212):1–6, 2020.

Stephen Portnoy. Censored regression quantiles. Journal of the American Statistical Association, 98(464):1001–1012, 2003.

Shi-ang Qi, Weijie Sun, and Russell Greiner. Survivaleval: A comprehensive open-source python package for evaluating individual survival distributions. In Proceedings of the AAAI Symposium Series, volume 2, pages 453–457, 2023.

Shi-ang Qi, Vahid Balazadeh, Michael Cooper, Russell Greiner, and Rahul G Krishnan. Survivalpfn: Amortizing survival prediction via in-context bayesian inference. arXiv preprint arXiv:2605.15488, 2026.

Jingang Qu, David Holzmuller, Ga¨ el Varoquaux, and Marine Le Morvan. Tabicl: A tabular founda-¨ tion model for in-context learning on large data. arXiv preprint arXiv:2502.05564, 2025.

Jingang Qu, David Holzmuller, Ga ¨ el Varoquaux, and Marine Le Morvan. Tabiclv2: A better, faster, ¨ scalable, and open tabular foundation model. arXiv preprint arXiv:2602.11139, 2026.

David Rindt, Robert Hu, David Steinsaltz, and Dino Sejdinovic. Survival regression with proper scoring rules and monotonic neural networks. In International Conference on Artificial Intelligence and Statistics, 2022.

Jake Robertson, Arik Reuter, Siyuan Guo, Noah Hollmann, Frank Hutter, and Bernhard Scholkopf.¨ Do-pfn: In-context learning for causal effect estimation. arXiv preprint arXiv:2506.06039, 2025.

Dmitrii Seletkov, Paul Hager, Georgios Kaissis, Rickmer Braren, Daniel Rueckert, and Raphael Rehms. Survival in-context: Amortized bayesian survival analysis via prior-fitted networks. arXiv preprint arXiv:2603.29475, 2026.

Ethan Steinberg, Jason Fries, Yizhe Xu, and Nigam Shah. MOTOR: A time-to-event foundation model for structured medical records. In International Conference on Learning Representations (ICLR), 2024.

Hans-Christian Thorsen-Meyer, Davide Placido, Benjamin Skov Kaas-Hansen, Anna P Nielsen, Theis Lange, Annelaura B Nielsen, Palle Toft, Jens Schierbeck, Thomas Strøm, Piotr J Chmura, et al. Discrete-time survival analysis in the critically ill: a deep learning approach using heterogeneous data. NPJ digital medicine, 5(1):142, 2022.

Mariana Vargas Vieyra. Staying alive: Uncensored survival analysis with tabular foundation models. arXiv preprint arXiv:2606.03689, 2026.

Simon Wiegrebe, Philipp Kopper, Raphael Sonabend, Bernd Bischl, and Andreas Bender. Deep learning for survival analysis: a review. Artificial Intelligence Review, 57(3):65, 2024.

Han-Jia Ye, Si-Yang Liu, Hao-Run Cai, Qi-Le Zhou, and De-Chuan Zhan. A closer look at deep learning methods on tabular datasets. arXiv preprint arXiv:2407.00956, 2024.

Chun-Nam Yu, Russell Greiner, Hsiu-Chin Lin, and Vickie Baracos. Learning patient-specific cancer survival distributions as a sequence of dependent regressors. Advances in neural information processing systems, 24, 2011.

Xiyuan Zhang, Danielle C Maddix, Junming Yin, Nick Erickson, Abdul Fatir Ansari, Boran Han, Shuai Zhang, Leman Akoglu, Christos Faloutsos, Michael W Mahoney, et al. Mitra: Mixed synthetic priors for enhancing tabular foundation models. arXiv preprint arXiv:2510.21204, 2025.

Yonghao Zhao, Changtao Li, Chi Shu, Qingbin Wu, Hong Li, Chuan Xu, Tianrui Li, Ziqiang Wang, Zhipeng Luo, and Yazhou He. Tackling small sample survival analysis via transfer learning: A study of colorectal cancer prognosis. Artificial Intelligence in Medicine, page 103426, 2026.

Xinliang Zhu, Jiawen Yao, and Junzhou Huang. Deep convolutional neural network for survival analysis with pathological images. In 2016 IEEE international conference on bioinformatics and biomedicine (BIBM), pages 544–547. IEEE, 2016.

## Appendices

## Contents

A SurvICL model details 17   
A.1 Architecture . 17   
A.2 Pretraining 18   
A.3 Example of generated datasets . . 18   
B Experimental Details 19   
B.1 Details on Baselines 19   
B.2 Hyperparameter tuning 21   
B.3 Issues for some methods 23   
B.4 Details on Datasets . 23   
C Ablations 26   
D Other Results 27   
D.1 All held-out datasets . . 27   
D.2 Held-out datasets with less than 1000 samples . 35   
D.3 Held-out datasets with more than 1000 samples 43

## A SURVICL MODEL DETAILS

## A.1 ARCHITECTURE

We build SurvICL on top of TabICLv2 (Qu et al., 2026), which is at the time of writing the leading fully open-source model on the TabArena benchmark (Erickson et al., 2025). While refer to the original paper for the details, the key components of the TabICLv2 architecture can be summarized as follows:

• Preprocessing: First, some light preprocessing is applied, including ordinal encoding of categorical columns. After the preprocessing, the model receives a fully numerical table.

• Target-aware embedding: The original covariates $( x _ { i j } ) \in \mathbb { R } ^ { n \times m }$ are transformed into embeddings $( z _ { i j } ) \in \mathbb { R } ^ { n \times m \times 1 2 8 }$ containing one embedding vector $z _ { i j } ~ \in ~ \mathbb { R } ^ { 1 2 8 }$ per cell $( i , j )$ . Each embedding vector is obtained from an affine transformation applied to three cells in the same row (“repeated feature grouping”). An affine transformation of the target $y _ { i }$ is added for the context set rows.

• Column-wise embedding: Each column $j$ of the resulting embeddings is processed independently using three induced self-attention blocks that allow interactions between different rows in the same column, resulting in a complexity of $O ( n m )$ , which is linear in the number of rows, unlike a full Transformer. Crucially, the attention is designed to only retrieve information from the training set, meaning that predictions on test samples are independent of the other test samples.

• Row-wise interaction: Next, three full transformer blocks allow different columns within a row to interact. Before these blocks, four extra columns with fixed values are appended, and after the blocks, the values of these four columns are concatenated and serve as row embeddings, while the other columns are discarded.

• ICL transformer: After adding another embedding of the targets, the embeddings of each row interact via 12 regular transformer layers, again making sure that test samples cannot access other test samples.

• Head: Finally, a multilayer perceptron (MLP) head transforms the output embeddings into 999 quantile predictions (for regression).

Overall, the TabICLv2 architecture has a runtime complexity of $O ( n ^ { 2 } + n m ^ { 2 } )$ , and SurvICL preserves this complexity since it only modifies the $O ( n m )$ preprocessing and embedding parts. In addition, like other TFMs, we use ensembling, where the predictions of eight forward passes with different preprocessings and column permutations are averaged.

Building on TabICLv2, we introduce the following adaptations to handle survival data:

• Time normalization: We normalize observed times by their mean over the context set, yielding $\tilde { t } _ { i } = t _ { i } / \bar { t } ,$ , where $\bar { t } _ { \mathrm { c o n t e x t } }$ is the mean duration of the observed times. This makes the time representation invariant to the choice of measurement units. During pre-training, both context and query sets are normalized and the mean t¯is taken across context and query set. During inference (after pre-training), the mean for t¯is taken across the context set, and the predicted quantiles are transformed back to the original time scale.

• Censoring-aware target embedding: We replace the scalar target embedding with a joint representation of the observed time and event indicator. Specifically, we sum the embedding of the continuous time $\tilde { t } _ { i }$ and a learned embedding of the binary indicator $\delta _ { i } .$ . This representation is added to every cell embedding in context row i, allowing the column-wise encoder to distinguish observed events from censored observations.

• Survival information at the ICL stage: We also encode $( \tilde { t } _ { i } , \delta _ { i } )$ using a separate pair of time and event-indicator embeddings whose sum matches the dimension of the row representation. This embedding is added to context row representations before the ICL transformer. At both embedding stages, survival outcomes are supplied only for context rows.

• Survival prediction: The MLP head predicts quantiles of the conditional event-time distribution, using a softplus transformation to ensure that the quantiles are positive. After restoring the original time scale and forcing the predicted quantiles to be monotonous by sorting them, we obtain an estimated cumulative distribution function by linear interpolation between quantile levels. The survival function is then given by

$$
{ \widehat { S } } ( t \mid x ) = 1 - { \widehat { F } } ( t \mid x ) .
$$

Table 2: Backbone architecture and prior for SurvICL.
<table><tr><td colspan="2">Backbone</td></tr><tr><td>Column-wise transformer Row-wise transformer</td><td>3 blocks, 8 heads, 128 inducing points, QSSMax attention 3 blocks, 8 heads, 4 CLS tokens</td></tr><tr><td></td><td></td></tr><tr><td>In-context (ICL) transformer</td><td>12 blocks, 8 heads, QSSMax attention</td></tr><tr><td>Max. sequence length (rows)</td><td>1024</td></tr><tr><td>Max. features per dataset</td><td>100 (min. 1)</td></tr><tr><td>Regression head</td><td>Pinball / quantile, 999 quantile levels</td></tr><tr><td>Target embedding</td><td>Sum</td></tr><tr><td>Duration rescaling</td><td>Mean-duration</td></tr><tr><td>Label draws per row</td><td>5</td></tr><tr><td>Training steps</td><td>200,000</td></tr><tr><td>Learning rate</td><td>0.0008</td></tr><tr><td colspan="2">Synthetic prior</td></tr><tr><td>Prior generator</td><td>GraphSCM(graph_scm_survival)</td></tr><tr><td>Structural family</td><td>Proportional odds (generation_type po)</td></tr><tr><td>Survival parameter range</td><td>Wide</td></tr><tr><td>Loss</td><td>Pinball loss on death-time quantiles</td></tr></table>

## A.2 PRETRAINING

We follow many of the pretraining settings from TabICLv2 but make the following modifications:

• We omit the second and third stages of pretraining, which scale to larger datasets, since the second stage did not show improvements on our benchmark.

• For computational reasons, we only train for 200K steps instead of 500K. Moreover, our pretraining consisted of two pretrainings for 100K steps each, meaning that the learningrate schedule consisted of two 100K-steps schedules.

• We generate pretraining datasets with 1024 samples each, of which a random fraction in [0.03, 0.9] is used as context (train) and the rest as query (test) samples. We lowered the lower bound from 0.3 to 0.03 to cover the smaller datasets found in our benchmark.

• We pretrain fully in 32-bit floats to avoid potential instabilities.

## A.3 EXAMPLE OF GENERATED DATASETS

Figure 5 presents a diverse set of examples sampled from our final Proportional Odds (PO) prior. These datasets incorporate both administrative censoring and time discretization. In each subfigure: left and middle panels are histograms showing the observed events and censoring events over time while the right panel is the resulting Kaplan-Meier survival probability curve.

![](images/9115a0b421054585035194c4554664fa0bc5a3e08750c76d801a20f52521754e.jpg)  
(a) Sample 1

![](images/4e760e8de9cb468039a8be7a248b20ff7f1d60e6c737f3e91a95ed66922e56cd.jpg)  
(b) Sample 2

![](images/d98c6cad80d73714b814c8719c1c3d501b5b46ae74fd0b52ee7d51d48a7c9446.jpg)  
(c) Sample 3

![](images/0adf803c603d7a772c32d5f13cb1857f83577ef22f3c40c3d6ea0ed642f6e2be.jpg)  
(d) Sample 4

![](images/c81070d044a8d585ad1aa6dc9aa894a90c9876575dac33f214fa8dfa7305da11.jpg)  
(e) Sample 5

![](images/17ec12990ffd3e8b8927df95d003d6df2f8816f5e8c9da6a36d40208f617de54.jpg)  
(f) Sample 6  
Figure 5: Six generated datasets using our final PO prior, incorporating administrative censoring and time discretization. For each sample, we show the histogram of observed event (left), the censoring events (center), and the Kaplan-Meier curve (right).

## B EXPERIMENTAL DETAILS

## B.1 DETAILS ON BASELINES

## B.1.1 USUAL METHODS

Those two methods are the most used in the medical setting.

Kaplan-Meier Estimator (Kaplan and Meier, 1958): A non-parametric statistic used to estimate the marginal survival probability from observed lifetime data. Because it does not take individual covariates into account, it serves as the most basic populationlevel baseline for survival functions and is consistent.

Cox Model (Cox Proportional Hazards) (Cox, 1972): The standard semiparametric workhorse of survival analysis. It models the hazard rate as a product of a time-varying baseline hazard and an exponential linear combination of the covariates. It relies on the strong assumption of proportional hazards (the hazard ratio between any two individuals is constant over time).

<table><tr><td>Type</td><td>Model Name</td><td>Reference</td></tr><tr><td>TFM</td><td>SurvICL SurvPFN SurvivalPFN Survival-IC</td><td>This work Böhm et al. 2026 Qi et al. 2026 Seletkov et al. 2026</td></tr><tr><td>Tree Based</td><td>TFM4Survival SurvivalBoost GBC RSF</td><td>Kim et al. 2026 Alberge et al. 2024 Pölsterl 2020 Ishwaran et al. 2008</td></tr><tr><td>Deep Neural Net</td><td>TabSurv SumoNet DSM DeepSurv DeepHit</td><td>Kirpichenko et al. 2026 Rindt et al. 2022 Nagpal et al. 2021 Katzman et al. 2018 Lee et al. 2018</td></tr><tr><td>Shallow Model</td><td>Cox Kaplan-Meier</td><td>Cox 1972 Kaplan and Meier 1958</td></tr></table>

Table 3: Overview of methods we consider in our experimental section.

## B.1.2 TREE-BASED MODELS

These models adapt decision trees and boosting trees to handle right-censored data, remaining highly competitive on tabular datasets.

RSF (Random Survival Forests) (Ishwaran et al., 2008): An extension of the Random Forest algorithm for time-to-event data. It constructs an ensemble of survival trees using specialized splitting criteria (like the log-rank test) to handle censored observations.

GBS (Gradient Boosting for Survival) (Polsterl¨ , 2020): A boosting approach implemented in scikit-survival that sequentially builds regression trees. In our evaluation, we include two variants corresponding to different optimization targets: sksurv GBS (Cox), which minimizes the negative log partial likelihood of the Cox proportional hazards model, and sksurv GBS (C-index), which directly optimizes a smoothed concordance index.

SurvivalBoost (Alberge et al., 2024): A modern, highly optimized gradient boosting framework specifically tailored for survival analysis, optimizing a proper scoring rule.

## B.1.3 DEEP NEURAL NETWORKS

These methods leverage multi-layer architectures to capture complex, non-linear relationships between covariates and survival times, often dropping the proportional hazards assumption.

DeepHit (Lee et al., 2018): A discrete-time survival network that optimizes a loss that combines a likelihood and a ranking metric over an MLP, usable in survival analysis and competing events.

DeepSurv (Katzman et al., 2018): It uses a Multi-Layer Perceptron (MLP) to replace the linear combination of covariates in the standard Cox model, allowing it to learn complex, non-linear interactions while still optimizing the Cox partial likelihood.

DSM (Deep Survival Machines) (Nagpal et al., 2021): DSM uses a neural network to model the survival function as a mixture of parametric distributions (such as Weibull or Log-Normal), providing well-calibrated survival estimates.

SumoNet (Rindt et al., 2022): SurvivalMonotonic-net utilizes a partially monotonic neural network to directly optimize the right-censored log-likelihood.

TabSurv (Kirpichenko et al., 2026): A specialized deep learning framework that adapts modern tabular neural networks to time-to-event prediction. Because this framework supports multiple representation backbones, we evaluate several architectural variants that appear throughout our results: the default TabSurv, TabSurv-MLP (using a standard Multi-Layer Perceptron), TabSurv-RealMLP (utilizing the more robust RealMLP tabular backbone), as well as TabSurv-WAS and TabSurv-WSA (which incorporate specialized attention and sparse embedding mechanisms designed specifically for tabular features).

## B.1.4 TABULAR FOUNDATION MODELS

This category represents the most modern class of baselines: tabular foundation models trained on synthetic tabular data. They generalize to new datasets in a zero-shot or in-context manner.

SurvPFN: (Bohm et al.¨ , 2026) SurvPFN is a survival-specific Prior-Data Fitted Network (PFN) based on the architecture of TabPFN v2, pretrained on synthetic censored survival tasks (sampling from different distributions such as Weibull).

SurvivalPFN (Qi et al., 2026): SurvivalPFN performs amortized Bayesian survival inference by using the observed covariates, event times, and censoring indicators as in-context examples to directly predict posterior event-time distributions for new individuals based on the CausalPFN architecture.

Survival-IC (Seletkov et al., 2026): Survival In-Context (SIC) is a prior-fitted foundation model pretrained on synthetic survival datasets to approximate Bayesian posterior predictive inference and based on the architecture of TabICL.

TFM4Survival (Kim et al., 2026): TFM4Survival adapts pretrained tabular foundation models such as TabICL to censored time-to-event data through a sampling-based formulation, allowing existing TFMs to perform survival prediction without survival-specific pretraining.

## B.2 HYPERPARAMETER TUNING

Table 4 shows a description of the hyperparameter search spaces for each method. These spaces are used with a budget of 50 iterations.

Table 4: Hyper-parameter search space for every baseline in our survival benchmark. Each cell is searched with RandomizedSearchCV for 50 iterations on a single stratified 80/20 split of the seed-0 training fold, scored by the integrated Brier score (IBS). randint/loguniform denote the scipy.stats distributions; a list denotes a categorical choice; a lone value is held fixed (not searched).

<table><tr><td>model</td><td>parameter</td><td>range</td></tr><tr><td colspan="3">Others</td></tr><tr><td>Kaplan-Meier</td><td colspan="2">no hyperparameters (population-level estimator)</td></tr><tr><td>Cox (lifelines, penalized)</td><td>penalizer</td><td>1oguniform(1e-04, 10)</td></tr><tr><td colspan="3">Tree-based methods</td></tr><tr><td>Random Survival Forest</td><td>max features min samples leaf</td><td>sqrt, log2, 0.3, 0.5, 1.0 randint(1,30) randint(2, 20)</td></tr><tr><td>sksurv GBS (Cox)</td><td>dropout rate learning rate loss n estimators subsample</td><td>0.0, 0.1, 0.2 loguniform(1e-03,0.3) coxph randint(50, 500) 0.7, 0.85, 1.0</td></tr><tr><td>sksurv GBS (C-index)</td><td>dropout rate learning rate loss max depth min samples leaf n estimators</td><td>0.0, 0.1, 0.2 loguniform(1e-03,0.3) coxph randint(2,8) randint(1,20) randint(50,400)</td></tr><tr><td></td><td>max depth n horizons per observation n iter n iter before feedback Neural network methods</td><td>loguniform(0.01,0.1) randint(2, 10) randint(1,5) randint(5,50) randint(5,50)</td></tr><tr><td></td><td>0.2, 0.5, 0.9 batch size 128, 256, 512</td><td>randint(256, 1024) 1oguniform(1e-05,0.001) 40, 60, 80, 100 [128, 128], [256, 256], [64, 64]</td></tr><tr><td></td><td>weight decay activation batch norm batch size</td><td>1oguniform(1e-05,0.001) ReLU True, False 128, 256, 512</td></tr><tr><td></td><td>dropout early stopping hidden dim lr</td><td>0.0, 0.1, 0.2, 0.4 True (patience 20) 16, 32, 64, 128 loguniform(1e-05,0.001) 64, 128, 256, 512</td></tr><tr><td>Deep Survival Machines</td><td>weight decay batch size</td><td>1oguniform(1e-05,0.1) 128, 256, 512</td></tr><tr><td></td><td>distribution iters k layers learning rate temperature</td><td>Weibull, LogNormal randint(50, 200) randint(2,7) [100], [100, 100], [50, 50], [100, 100, 100] loguniform(1e-04, 0.01) 0.5, 1.0, 2.0</td></tr><tr><td>SuMo-Net</td><td>batch size dropout layers layers surv lr n iter weight decay</td><td>128, 256, 512 0.0, 0.1, 0.25 [64, 64], [100, 100], [32, 32, 32] [64], [100], [100, 100] 1oguniform(1e-04, 0.01) randint(200, 1500)</td></tr><tr><td>TabSurv-LAS</td><td>activation batch size d block dropout emb dim rtdl k llh radius lr n blocks rtdl activation</td><td>ReLU, SiLU, SELU 32, 64, 96 128, 256, 512 1oguniform(0.01, 0.1) 8, 12, 16 8,16,32 1,2, 3,4, 5 1oguniform(1e-04, 0.005)</td></tr><tr><td></td><td>rtdl n bins activation batch size emb dim rtdl hidden dim layer n 1lh radius lr normalization rtdl activation rtdl n bins</td><td>32, 48, 64 ReLU, SiLU, SELU 32, 64, 96 8, 12, 16 128, 256, 512 randint(1,5) 1,2, 3, 4,5 1oguniform(1e-04, 0.005) False, True True, False 32,48, 64</td></tr><tr><td></td><td>activation batch size emb dim rtdl frequency scale hidden dim 1lh radius lr n frequencies n hidden layers activation</td><td>Mish, SELU 32, 64, 96 8, 12, 16 1oguniform(0.01, 1.0) 128, 256, 512 1,2, 3,4,5 1oguniform(1e-04, 0.005) 32, 48, 64 randint(1,5)</td></tr><tr><td></td><td>batch size d block emb dim rtdl k</td><td>ReLU, SELU 32, 64, 96 64, 128, 256 8, 12, 16</td></tr><tr><td>TabSurv-WSA</td><td>llh radius lr n blocks rtdl activation rtdl n bins activation batch size</td><td>8,16,32 1,3,5 1oguniform(1e-04, 0.005) randint(1,4) True, False 32, 48, 64 ReLU, SELU</td></tr><tr><td rowspan="5"></td><td>k</td><td>8,16,32</td></tr><tr><td>llh radius</td><td>1,3,5</td></tr><tr><td>lr</td><td>1oguniform(1e-04, 0.005)</td></tr><tr><td>n blocks</td><td>randint(1,4)</td></tr><tr><td>rtdl activation rtdl n bins</td><td>True, False 32, 48, 64</td></tr><tr><td></td><td colspan="2">SurvICL and tabular foundation models</td></tr><tr><td>SIC</td><td colspan="2">no hyperparameters (frozen DeepHit10 checkpoint)</td></tr><tr><td>SurvivalPFN</td><td>calibrate max context size</td><td>False, True 1024, 2048, 4096, 8192</td></tr><tr><td>SurvPFN</td><td>max context size max features</td><td>1024, 2048, 4096 5,10</td></tr><tr><td>TFM-Survival</td><td>max train rows n intervals</td><td>4096,8192 randint(5,20)</td></tr><tr><td>Surv. Classif.: TabICL</td><td>n intervals</td><td>randint(5,40)</td></tr></table>

## B.3 ISSUES FOR SOME METHODS

Some baselines failed to complete their hyperparameter search or final fit on a small subset of datasets, for reasons intrinsic to the method rather than to the benchmark protocol (e.g., a training run that diverged numerically, or a step of the method’s own pipeline that is only defined for datasets larger than the fold at hand). We assign the method a trivially achievable score on each dataset on which it did not complete: an IBS of 1 (the maximum attainable value in expectation if G is estimated perfectly) or a concordance index of 0.5 (no better than chance), matching the metric being ranked. All critical-difference diagrams and pairwise win-rate figures are therefore always computed over the full 63-dataset held-out cohort. Methods with at least one imputed dataset are marked with an asterisk in the critical-difference diagrams and with a hatched bar in the win-rate figures; Table 5 lists the affected (method, dataset) pairs and their cause.

Table 5: Baseline (method, dataset) pairs that created issues
<table><tr><td>Method</td><td>Dataset(s)</td><td>Cause</td></tr><tr><td>DeepSurv</td><td>divorce, actg, FRTCS</td><td>Training loss diverged (became non-finite) during optimization on these three small datasets.</td></tr><tr><td>TFM4Surival: TabICL sac3, DLBCL, gse1992</td><td></td><td>Out-of-memory issue on these larger/high- dimensional datasets.</td></tr><tr><td>SurvivalPFN</td><td>AML_Bull, gse1992, vdv</td><td>The method&#x27;s internal dimensionality- reduction step (a truncated SVD) requires more components than the smaller of the fold&#x27;s row or feature count provides, and errors when that bound is violated.</td></tr><tr><td>TabSurv</td><td>DBCD, cesc tcga, kirc tcga, lgg tcga</td><td>The method&#x27;s obtained a MAE of value in- fty.</td></tr></table>

## B.4 DETAILS ON DATASETS

An overview of the development and held-out datasets is shown in Tables 6 and 7.

<table><tr><td>name</td><td>source</td><td>n samples</td><td>n features</td><td>censoring frac</td><td>km survival at tmax</td></tr><tr><td>Framingham</td><td>survset</td><td>4699</td><td>18</td><td>68.7%</td><td>60.5 %</td></tr><tr><td>MCLcleaned</td><td>survset</td><td>92</td><td>574</td><td>30.4 %</td><td>0.0 %</td></tr><tr><td>Melanoma</td><td>survset</td><td>205</td><td>5</td><td>72.2 %</td><td>64.5 %</td></tr><tr><td>NSBCD</td><td>survset</td><td>115</td><td>549</td><td>67.0 %</td><td>57.4 %</td></tr><tr><td>Rossi</td><td>survset</td><td>432</td><td>29</td><td>73.6 %</td><td>73.6 %</td></tr><tr><td>Unemployment</td><td>survset</td><td>452</td><td>8</td><td>43.4 %</td><td>5.8 %</td></tr><tr><td>acath</td><td>survset</td><td>3504</td><td>3</td><td>33.4 %</td><td>0.0 %</td></tr><tr><td>aids</td><td>survset</td><td>467</td><td>5</td><td>59.7 %</td><td>49.2 %</td></tr><tr><td>brca tcga pan can atlas 2018</td><td>cbioportal</td><td>980</td><td>137</td><td>95.2 %</td><td>60.1 %</td></tr><tr><td>cgd</td><td>survset</td><td>128</td><td>25</td><td>65.6 %</td><td>47.6 %</td></tr><tr><td>challenge ens</td><td>custom</td><td>3173</td><td>42</td><td>49.6 %</td><td>5.9 %</td></tr><tr><td>chop</td><td>survset</td><td>414</td><td>3833</td><td>60.1 %</td><td>39.5 %</td></tr><tr><td>cost</td><td>survset</td><td>518</td><td>13</td><td>22.0 %</td><td>22.0 %</td></tr><tr><td>crc orion 2024</td><td>cbioportal</td><td>74</td><td>131</td><td>66.2 %</td><td>30.4 %</td></tr><tr><td>csl</td><td>survset</td><td>446</td><td>6</td><td>39.5 %</td><td>17.3 %</td></tr><tr><td>d.oropha.rec</td><td>survset</td><td>192</td><td>30</td><td>27.6 %</td><td>16.5 %</td></tr><tr><td>dataDIVAT3</td><td>survset</td><td>4267</td><td>17</td><td>94.4 %</td><td>84.7 %</td></tr><tr><td>dataOvarian1</td><td>survset</td><td>912</td><td>163</td><td>40.4 %</td><td>11.9 %</td></tr><tr><td>epileptic</td><td>survset</td><td>605</td><td>5</td><td>63.5 %</td><td>46.3 %</td></tr><tr><td>filchain</td><td>pycox</td><td>6524</td><td>8</td><td>69.9 %</td><td>65.6 %</td></tr><tr><td>gse3143</td><td>survset</td><td>158</td><td>8664</td><td>68.4 %</td><td>41.9 %</td></tr><tr><td>kirc tcga pan can atlas 2018</td><td>cbioportal</td><td>369</td><td>53</td><td>92.7 %</td><td>86.5 %</td></tr><tr><td>kirp tcga pan can atlas 2018</td><td>cbioportal</td><td>251</td><td>56</td><td>96.0 %</td><td>92.9 %</td></tr><tr><td>luad tcga pan can atlas 2018</td><td>cbioportal</td><td>392</td><td>98</td><td>82.4 %</td><td>68.4 %</td></tr><tr><td>lusc tcga pan can atlas 2018</td><td>cbioportal</td><td>481</td><td>100</td><td>57.2 %</td><td>0.0 %</td></tr><tr><td>micro.censure</td><td>survset</td><td>117</td><td>118</td><td>77.8 %</td><td>42.4 %</td></tr><tr><td>msk 2026</td><td>cbioportal</td><td>40755</td><td>56</td><td>57.0 %</td><td>31.7 %</td></tr><tr><td>oldmort</td><td>survset</td><td>4603</td><td>17</td><td>57.2 %</td><td>0.2 %</td></tr><tr><td>ov tcga pan can atlas 2018</td><td>cbioportal</td><td>427</td><td>34</td><td>48.2 %</td><td>6.5 %</td></tr><tr><td>ova</td><td>survset</td><td>358</td><td>13</td><td>25.7 %</td><td>22.2 %</td></tr><tr><td>ovarian</td><td>survset</td><td>26</td><td>4</td><td>53.8 %</td><td>49.7 %</td></tr><tr><td>pancan pcawg 2020</td><td>cbioportal</td><td>361</td><td>255</td><td>62.9 %</td><td>0.0 %</td></tr><tr><td>rdata</td><td>survset</td><td>1040</td><td>7</td><td>47.4 %</td><td>38.1 %</td></tr><tr><td>sarc tcga pan can atlas 2018</td><td>cbioportal</td><td>181</td><td>147</td><td>87.3 %</td><td>74.3 %</td></tr><tr><td>smarto</td><td>pycox</td><td>3873</td><td>39</td><td>88.1 %</td><td>72.5 %</td></tr><tr><td>ucs tcga pan can atlas 2018</td><td>cbioportal</td><td>57</td><td>11</td><td>38.6 %</td><td>9.2 %</td></tr><tr><td>uvm tcga pan can atlas 2018</td><td>cbioportal</td><td>80</td><td>6</td><td>71.2 %</td><td>38.5 %</td></tr><tr><td>vlbw</td><td>survset</td><td>617</td><td>55</td><td>82.7 %</td><td>0.0 %</td></tr><tr><td>whas500</td><td>survset</td><td>461</td><td>18</td><td>61.8 %</td><td>0.0 %</td></tr><tr><td>zinc</td><td>survset</td><td>431</td><td>23</td><td>81.2 %</td><td>79.0 %</td></tr><tr><td>AML Bull</td><td>survset</td><td>116</td><td>6283</td><td>42.2 %</td><td>28.3 %</td></tr><tr><td>Aids2</td><td>survset</td><td>2839</td><td>14</td><td>38.0 %</td><td>5.8 %</td></tr><tr><td>Bergamaschi</td><td>survset</td><td>82</td><td>10</td><td>65.9 %</td><td>58.3 %</td></tr><tr><td>DBČD</td><td>survset</td><td>295</td><td>4919</td><td>73.2 %</td><td>61.0 %</td></tr><tr><td>DLBCL</td><td>survset</td><td>240</td><td>7399</td><td>42.5 %</td><td>26.8 %</td></tr><tr><td>Dialysis</td><td>survset</td><td>6805</td><td>37</td><td>76.4 %</td><td>58.7 %</td></tr><tr><td>FRTCS</td><td>survset</td><td>697</td><td>5</td><td>89.7 %</td><td>89.1 %</td></tr><tr><td>LeukSurv</td><td>survset</td><td>1043</td><td>30</td><td>15.7 %</td><td>4.4 %</td></tr><tr><td>Pbc3</td><td>survset</td><td>349</td><td>21</td><td>82.5 %</td><td>63.4 %</td></tr><tr><td>TRACE</td><td>survset</td><td>1878</td><td>6</td><td>49.0 %</td><td>43.9 %</td></tr><tr><td>UnempDur</td><td>survset</td><td>3241</td><td>6</td><td>38.7 %</td><td>10.5 %</td></tr><tr><td>acc tcga pan can atlas 2018</td><td>cbioportal</td><td>91</td><td>39</td><td>63.7 %</td><td>49.9 %</td></tr><tr><td>actg</td><td>survset</td><td>1151</td><td>20</td><td>91.7 %</td><td>90.1 %</td></tr><tr><td>blca tcga pan can atlas 2018</td><td>cbioportal</td><td>301</td><td>84</td><td>76.1 %</td><td>62.0 %</td></tr><tr><td>breast</td><td>survset</td><td>100</td><td>4</td><td>74.0 %</td><td>71.4 %</td></tr><tr><td>burn</td><td>survset</td><td>154</td><td>14</td><td>68.8 %</td><td>48.0 %</td></tr><tr><td>cancer</td><td>survset</td><td>228</td><td>30</td><td>27.6 %</td><td>5.0 %</td></tr><tr><td>cesc tcga pan can atlas 2018</td><td>cbioportal</td><td>241</td><td>97</td><td>95.0 %</td><td>91.3 %</td></tr><tr><td>chol tcga pan can atlas 2018</td><td>cbioportal</td><td>36</td><td>53</td><td>50.0 %</td><td>19.5 %</td></tr><tr><td>coadread tcga pan can atlas 2018</td><td>cbioportal</td><td>543</td><td>133</td><td>86.2 %</td><td>62.1 %</td></tr><tr><td>colon</td><td>survset</td><td>929</td><td>14</td><td>51.3 %</td><td>45.5 %</td></tr><tr><td>crc hta8 htan 2024</td><td>cbioportal</td><td>83</td><td>91</td><td>69.9 %</td><td>24.4 %</td></tr><tr><td>dataDIVAT2</td><td>survset</td><td>1837</td><td>4</td><td>68.3 %</td><td>31.1 %</td></tr><tr><td>diabetes</td><td>survset</td><td>394</td><td>4</td><td>60.7 %</td><td>53.0 %</td></tr><tr><td>difg glass</td><td>cbioportal</td><td>661</td><td>179</td><td>24.8 %</td><td>0.0 %</td></tr><tr><td>divorce</td><td>survset</td><td>3371</td><td>5</td><td>69.4 %</td><td>55.7 %</td></tr><tr><td>e1684</td><td>survset</td><td>284</td><td>3</td><td>31.0 %</td><td>28.3 %</td></tr><tr><td>esca tcga pan can atlas 2018</td><td>cbioportal</td><td>182</td><td>90</td><td>58.2 %</td><td>10.7 %</td></tr><tr><td>gbm tcga pan can atlas 2018</td><td>cbioportal</td><td>277</td><td>30</td><td>31.8 %</td><td>0.0 %</td></tr><tr><td>gbsg</td><td>pycox</td><td>2232</td><td>7</td><td>43.2 %</td><td>36.9 %</td></tr><tr><td>grace</td><td>survset</td><td>1000</td><td>5</td><td>67.6 %</td><td>63.5 %</td></tr><tr><td>gse1992</td><td>survset</td><td>124</td><td>15545</td><td>71.8 %</td><td>45.4 %</td></tr><tr><td>hdfail</td><td>pycox</td><td>52422</td><td>38</td><td>94.5 %</td><td>56.1 %</td></tr><tr><td>heart</td><td>survset</td><td>103</td><td>4</td><td>27.2 %</td><td>15.2 %</td></tr><tr><td>heartvalve</td><td>survset</td><td>256</td><td>27</td><td>78.9 %</td><td>53.0 %</td></tr><tr><td>hepatoCellular</td><td>survset</td><td>227</td><td>43</td><td>57.3 %</td><td>51.2 %</td></tr><tr><td>hnsc tcga pan can atlas 2018</td><td>cbioportal</td><td>364</td><td>119</td><td>83.0 %</td><td>44.6 %</td></tr><tr><td>1gg tcga pān can atlas 2018 lihc tcga pan can atlas 2018</td><td>cbioportal</td><td>422</td><td>63</td><td>91.9 %</td><td>43.3 %</td></tr><tr><td>meso tcgà pan can atlas 2018</td><td>cbioportal</td><td>371</td><td>70</td><td>64.7 %</td><td>18.2 %</td></tr><tr><td>metabric</td><td>cbioportal</td><td>86</td><td>58</td><td>15.1 %</td><td>0.0 %</td></tr><tr><td></td><td>pycóx</td><td>1904</td><td>9</td><td>42.1 %</td><td>0.0 %</td></tr><tr><td>mgus</td><td>survset</td><td>241</td><td>13</td><td>6.6 %</td><td>1.8 %</td></tr><tr><td>msk 2021</td><td>cbioportal</td><td>25659</td><td>86</td><td>60.4 %</td><td>34.3 %</td></tr><tr><td>nki70</td><td>survset</td><td>144</td><td>77</td><td>66.7 %</td><td>48.0 %</td></tr><tr><td>nwtco</td><td>pycox</td><td>4028</td><td>6</td><td>85.8 %</td><td>84.9 %</td></tr><tr><td>paad tcga pan can atlas 2018</td><td>cbioportal</td><td>184</td><td>75</td><td>46.2 %</td><td>13.7 %</td></tr><tr><td>pbc</td><td>survset</td><td>312</td><td>6</td><td>59.9 %</td><td>34.1 %</td></tr><tr><td>pharmacoSmoking</td><td>survset</td><td>125</td><td>19</td><td>28.8 %</td><td>28.8 %</td></tr><tr><td>phpl04K8a</td><td>survset</td><td>442</td><td>21</td><td>46.6 %</td><td>0.0 %</td></tr><tr><td>prostate prostateSurvival</td><td>survset</td><td>502</td><td>28</td><td>29.5 %</td><td>23.8 %</td></tr><tr><td></td><td>survset</td><td>14294</td><td>8</td><td>94.4 %</td><td>81.4 %</td></tr><tr><td>rhc sac3</td><td>survset</td><td>5735</td><td>79</td><td>35.1 %</td><td>0.0 %</td></tr><tr><td></td><td>pycox</td><td>50000</td><td>45</td><td>37.3 %</td><td>20.2 %</td></tr><tr><td>scania</td><td>survset</td><td>1931</td><td>9</td><td>43.8 %</td><td>15.9 %</td></tr><tr><td>skcm tcga pan can atlas 2018</td><td>cbioportal</td><td>433</td><td>172</td><td>50.3 %</td><td>5.7 %</td></tr><tr><td>stad tcgā pân can atlas 2018</td><td>cbioportal</td><td>355</td><td>112</td><td>75.5 %</td><td>64.4 %</td></tr><tr><td>stagec</td><td>survset</td><td>146</td><td>22</td><td>63.0 %</td><td>55.8 %</td></tr><tr><td>support</td><td>pycox</td><td>8873</td><td>14</td><td>32.0 %</td><td>24.1 %</td></tr><tr><td>ucec tcga pan can atlas 2018</td><td>cbioportal</td><td>486</td><td>58</td><td>90.7 %</td><td>83.6 %</td></tr><tr><td>uis</td><td>survset</td><td>628</td><td>17</td><td>19.1 %</td><td>16.6 %</td></tr><tr><td>vdv</td><td>survset</td><td>78</td><td>4705</td><td>56.4 % 6.6 %</td><td>56.4 % 0.0 %</td></tr><tr><td>veteran wpbc</td><td>survset survset</td><td>137 198</td><td>9 32</td></tr></table>

Table 6: Overview of development datasets.

Table 7: Overview of held-out datasets.

## C ABLATIONS

To isolate the impact of specific design choices, we conduct ablation studies using a shared nanotabiclexsurvival backbone trained for 10,000 steps. Each ablation alters exactly one variable against a control arm trained on the Proportional Odds (po) prior as detailed in the following table. One arm can be trained in 2h on 2 A100.

Table 8: Backbone architecture shared by every arm.
<table><tr><td colspan="2">Backbone (nanotabiclexsurvival)</td></tr><tr><td>Column-wise transformer</td><td>2 blocks, 4 heads, 128 inducing points, SSMax attention</td></tr><tr><td>Row-wise transformer</td><td>2 blocks, 4 heads, 4 CLS tokens</td></tr><tr><td>In-context (ICL) transformer</td><td>4 blocks, 4 heads, SSMax attention</td></tr><tr><td>Regression head</td><td>Pinball / quantile, 99 quantile levels</td></tr><tr><td>Target embedding</td><td>Sum</td></tr><tr><td>Duration rescaling (control)</td><td>Mean-duration</td></tr><tr><td>Label draws per row (control)</td><td></td></tr><tr><td>Learning rate</td><td>0.0015</td></tr><tr><td colspan="2">Synthetic prior</td></tr><tr><td>Prior generator</td><td>GraphSCM(graph_scm_survival)</td></tr><tr><td>Loss</td><td>Pinball loss on death-time quantiles</td></tr></table>

## C.0.1 ABLATION DESIGN

Survival-family axis. Table 9 lists the generative equation of every family tested.

Table 9: Structural survival families compared on the family axis. H, O, A and h denote the cumulative hazard, cumulative odds, Aranda-Ordaz/Box-Cox transform, and hazard of $T \mid x ; { \textrm { a } } 0$ subscript denotes the shared random baseline.
<table><tr><td>Family</td><td>Arm(s)</td><td>Generative structure</td></tr><tr><td>Proportional hazards</td><td>COX</td><td> $H ( t \mid x ) = e ^ { \eta } H _ { 0 } ( t )$ </td></tr><tr><td>Proportional odds</td><td>po</td><td> $O ( t \mid x ) = e ^ { \eta } O _ { 0 } ( { t } ) , O = ( 1 - S ) / S$ </td></tr><tr><td>Transformation family</td><td>transform</td><td> $A _ { \rho } ( \dot { t } \mid x ) = e ^ { \eta } \dot { A } _ { \rho } ( S _ { 0 } ( t ) ) , \rho \sim [ 0 . 0 5 , 3 ]$  per dataset (interpolates PH at  $\dot { \rho } \to 0$  and PO at  $\rho = 1 )$ </td></tr><tr><td>Additive hazards</td><td>ah</td><td> $h ( t \mid \bar { x } ) = h _ { 0 } ( t ) + c \cdot \operatorname { t a n h } ( \eta )$  , clamped at 0</td></tr><tr><td>Accelerated hazards</td><td>ach</td><td> $h ( t \mid x ) = h _ { 0 } ( t e ^ { \eta } ) ( \mathrm { c u r v e s } \mathrm { l e a v e } S { = } 1 $  together)</td></tr><tr><td>AFT, homoscedastic</td><td>10  $\mathtt { i f t - h o m o }$ </td><td> $\log { \dot { T } } = \mu ( x ) + ( \log T _ { 0 } - \mathbb { E } \log T _ { 0 } )$ </td></tr><tr><td>AFT, heteroscedastic</td><td>aft-hetero</td><td>as above with a per-subject log-time spread, producing crossing survival curves</td></tr><tr><td>Quantile model</td><td>quantile</td><td>log  $Q _ { p } \bar { ( } T \mid x ) = \log Q _ { 0 , p } + \eta _ { p } ( x ) ,$  3 anchor levels</td></tr><tr><td>Classical parametric laws</td><td>law</td><td>Weibull / gamma / exponential / log-logistic mixture (“all&quot;), proportional hazards</td></tr><tr><td>Simplified (reduced)</td><td>simple</td><td>single-law, non-mixture, single label draw</td></tr><tr><td>Cox + AFT mixture</td><td>cox-aft</td><td>per-dataset random choice of cox or aft</td></tr><tr><td>Grand mixture</td><td>grand-mix</td><td>per-dataset random choice among cox, aft, po, ah, quantile</td></tr><tr><td>SurvPFN prior</td><td>survpfn</td><td>closed-form Weibull PH/non-PH times, ported from SurvPFN&#x27;s own pretraining prior</td></tr><tr><td>SurvivalPFN prior</td><td>survivalpfn</td><td>nonparametric hazard shape via a monotone Bernstein- polynomial map, ported from SurvivalPFN&#x27;s prior</td></tr><tr><td>SIC prior</td><td>sic</td><td>extended-hazards law with a per-dataset hazard-family draw, ported from SIC&#x27;s pretraining recipe</td></tr></table>

Recipe axes. Seven arms each flip exactly one flag off the control, holding generation type=po fixed (Table 10). Two of the axes (discretization, administrative censoring) are already off in the control, so we ablate by turning them on rather than removing them – the same one-knob-at-a-time principle, applied in the direction the control recipe permits.

Table 10: Recipe axes ablated against the po control, one flag at a time.
<table><tr><td>Axis</td><td>Control</td><td>Variant(s)</td></tr><tr><td>Discretization</td><td>Off</td><td>On (10% per-dataset chance of coarsening times to a grid)</td></tr><tr><td>Administrative censoring</td><td>Off</td><td>On (50% per-dataset chance of a study-end cutoff)</td></tr><tr><td>Label draws per row</td><td>1</td><td>5,20</td></tr><tr><td>Quantile head width</td><td>99 levels</td><td>999 levels</td></tr><tr><td>Auxiliary censoring head</td><td>Event-only targets</td><td>+ censoring-distribution head (train-time only, dropped at inference)</td></tr></table>

## C.0.2 EVALUATION PROTOCOL

Arms are evaluated at their final checkpoint on frozen, held-out validation datasets. To account for varying dataset scales and censoring rates, raw metrics are min–max normalized per dataset (100 = best arm, 0 = worst arm) before averaging. The primary ranking metric is the mean Integrated Brier Score (IBS).

## D OTHER RESULTS

## D.1 ALL HELD-OUT DATASETS

## D.1.1 IBS

Results for the IBS on held-out datasets are shown in Figures 6, 7, 8, 9.

![](images/5f7ead6e6a7295e5f10ef680967879d6ec5d37882e7685311208dc40c7bc342b.jpg)  
Figure 6: IBS on all held-out datasets, compared to the IBS score obtained with a Kaplan-Meier estimator (higher is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/e33698f28fc4f287b02e19462267bd3357af576be3aada228ae2ba58da44576c.jpg)  
Figure 7: Critical difference diagram of average ranks on all held-out datasets (lower is better) for IBS. Methods joined by a bar are not significantly different

![](images/2cd71f63fed66f1a2d5294ce434f647d47baf6fd3c52eee5976a9a92e7e1bf59.jpg)  
Figure 8: Pairwise win rate IBS all held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a lower IBS. The scores are the same as in Figure 1, but shown with a different grouping.

![](images/574b9c8af0279215e9e124f209b15dc8d4dbf19c0c75edc3ea9aacd0ae90e35a.jpg)  
Mean fit + predict time per fold (s)  
Figure 9: Pareto front of mean normalized IBS (higher is better) and fit + predict time. We report average performances over all held-out datasets. IBS was normalized to provide comparability across all datasets.

## D.1.2 C-INDEX50

Results for the C-index on held-out datasets are shown in Figures 10, 11, 12, 13.

![](images/b73c15a4ed57436963641d97d84869ce559da717e9334fdeb9e518df18d0317e.jpg)  
Figure 10: C-index on the 50th quantile on all held-out datasets, compared to the score obtained with a Kaplan-Meier estimator (higher is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/bdc61d54051fabdb9ff41ec720f51d7724e5ea45b1ea7bd008f31c3b65942d4a.jpg)  
Figure 11: Critical difference diagram of average ranks on all held-out datasets (lower is better) for the C-index on the 50th quantile. Methods joined by a bar are not significantly different.

![](images/3dc9149fb478096b384f67449241a6c56fcd2a37057294c0b46ccf1c0947f8bf.jpg)  
Figure 12: Pairwise win rate in the C-index on the 50th quantile on all held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a higher C-index.

![](images/1792e43b53610e4c388968c65333f3e595b46129814cee242c1209dc4530c2b9.jpg)  
Figure 13: Pareto front of mean C-index on the 50th quantile (higher is better) and fit + predict time. We report average performances over all held-out datasets.

## D.1.3 KM-CALIBRATION

Results for the KM-calibration on held-out datasets are shown in Figures 14, 15, 16.

![](images/e9c2d04086825ef2aa8460b377c8fec0190b5c59270c0f0a35568f9ccc6356fd.jpg)  
Figure 14: KM calibration on all held-out datasets (lower is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/43b82f7e8cdf50c9b7133f49b91228ddf45b34dc137d10582c167cbcadfcb5fc.jpg)  
Figure 15: Critical difference diagram of average ranks on all held-out datasets (lower is better) for the KM calibration. Methods joined by a bar are not significantly different.

![](images/78db5cc956d8eefe708af8e972962af8c12aacadecc39f8778be86a469ecdd11.jpg)  
Figure 16: Pairwise win rate in KM calibration on all held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a better KM calibration.

## D.1.4 MAE

Results for the MAE on held-out datasets are shown in Figures 17, 18, 19, 20.

![](images/338e70abce1d1edc88f921ff0bc2c429b2d98d788265d849d8320b8b8360e110.jpg)  
Figure 17: MAE on all held-out datasets, compared to the MAE obtained with a Kaplan-Meier estimator (higher is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/7edd7aff1d3daf01500b4813755fcfd7bd53c81b5dc0852cd52f3119cd2e76fc.jpg)  
Figure 18: Critical difference diagram of average ranks on all held-out datasets (lower is better) for the MAE. Methods joined by a bar are not significantly different

![](images/d592fc202ff868c45a377396ad1fffa6ace996971c325bf6963ecf8a7dc401b1.jpg)  
Figure 19: Pairwise win rate in MAE on all held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a lower MAE.

![](images/fb919d5b6aa86c0924db10d1652c378b9432590ba50ca37a186c148bc51e3701.jpg)  
Figure 20: Pareto front of mean MAE (lower is better) and fit + predict time. We report average performances over all held-out datasets.

## D.2 HELD-OUT DATASETS WITH LESS THAN 1000 SAMPLES

## D.2.1 IBS

Results for the IBS on small held-out datasets are shown in Figures 21, 22, 23, 24.

![](images/12685cd7d2c3c60a042c07477d50f3fc477ab248f456aab1bf1e19ebae68a347.jpg)  
Figure 21: IBS on the small held-out datasets, compared to the IBS score obtained with a Kaplan-Meier estimator (higher is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/1fa9a422fb955414f80a4b311b2fe7c3ef042217666c8900cf6b7c859e917fb7.jpg)  
Figure 22: Critical difference diagram of average ranks on the small held-out datasets (lower is better) for the IBS. Methods joined by a bar are not significantly different.

![](images/bbd4896e7b22a8da81b10867dc6d42cae195712c99b47e5082e2ccd436d44f9f.jpg)  
Figure 23: Pairwise win rate in IBS on the small held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a lower IBS.

![](images/533a649c5ba8fb03fdeee9d39bd880fe24ee2eb9696847c4036b7acb68d9b9f5.jpg)  
Figure 24: Pareto front of mean normalized IBS (higher is better) and fit + predict time. We report average performances over the small held-out datasets. IBS was normalized to provide comparability across all datasets.

## D.2.2 C-INDEX50

Results for the C-index at the 50th quantile on small held-out datasets are shown in Figures 25, 26, 27, 28.

![](images/a8f75a0f12d91a64ff05c2dc7623e503f2143298413b1cfba6e386a864607ece.jpg)  
Figure 25: C-index at the 50th quantile on the small held-out datasets, compared to the C-index score obtained with a Kaplan-Meier estimator (higher is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/028d60d2ac459041ec34e987b6a429429bdead3a1f78afeca6956d479c1a969c.jpg)  
Figure 26: Critical difference diagram of average ranks on the small held-out datasets (lower is better) for the C-index at the 50th quantile. Methods joined by a bar are not significantly different.

![](images/d0e577861d81d15a0246b233dc1433d5360f4d445331f663ba2a324939c8dbbe.jpg)  
Figure 27: Pairwise win rate in the C-index at the 50th quantile on the small held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a higher C-index.

![](images/4a061fa28f0874308da25d769f4d3f90a81222279924bbe7999813d60e55e21f.jpg)  
Figure 28: Pareto front of mean C-index on the 50th quantile (higher is better) and fit + predict time. We report average performances over the small held-out datasets.

## D.2.3 KM-CALIBRATION

Results for the KM-calibration on small held-out datasets are shown in Figures 29, 30, 31.

![](images/683e06685de66126a7f8d1e6b3f1fc962acaada1973fee7f61324119e3222e87.jpg)  
Figure 29: KM calibration on the small held-out datasets (lower is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/91c17d6dd0d004c5b5cfa59890ee228c040169814ee6d6f79f5557ada34392a6.jpg)  
Figure 30: Critical difference diagram of average ranks on the small held-out datasets (lower is better) for the KM calibration. Methods joined by a bar are not significantly different.

![](images/48be7a2240bbd56ce19b2cabe09345aa38341bc4b958d8b836598bf815ed9b98.jpg)  
Figure 31: Pairwise win rate in the KM-calibration on the small held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a better KM-calibration.

## D.2.4 MAE

Results for the MAE on small held-out datasets are shown in Figures 32, 33, 34, 35.

![](images/aef9d2306ed08fa0251c8f47f96b344fd2811c7c3e017c7f3f0f2399d623acae.jpg)  
Figure 32: MAE on the small held-out datasets, compared to the MAE obtained with a Kaplan-Meier estimator (higher is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/5afbf430b6cf05370f7085affafc015d998b1bdac4c02bd6a36927eaeb770457.jpg)  
Figure 33: Critical difference diagram of average ranks on the small held-out datasets (lower is better) for the MAE. Methods joined by a bar are not significantly different.

![](images/9b3c78335f35ab7708bd9d8fa294947849a846f42c737784212192fdcb47150e.jpg)  
Figure 34: Pairwise win rate in the MAE on the small held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a lower MAE.

![](images/049c6a0cc4087e74809c68945cffdd856080d05ce1d79ccda9927595d5b15e60.jpg)  
Figure 35: Pareto front of mean MAE (lower is better) and fit + predict time. We report average performances over the small held-out datasets.

## D.3 HELD-OUT DATASETS WITH MORE THAN 1000 SAMPLES

## D.3.1 IBS

Results for the IBS on large held-out datasets are shown in Figures 36, 37, 38, 39.

![](images/ff35bcbf69f6fa7abaf577ba674a771b5382b4ce9688f231dafc15f7a3403570.jpg)  
Figure 36: IBS on the large held-out datasets, compared to the IBS obtained with a Kaplan-Meier estimator (higher is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/40a81c59ace892a716c2b22adb0162905f8f462d8b2e268f1cd720fe95fe88ad.jpg)  
Figure 37: Critical difference diagram of average ranks on the large held-out datasets (lower is better) for the IBS. Methods joined by a bar are not significantly different.

![](images/a6c418beca020ee470a1c51da7b89971d97b47ce4a52ce5a9f3138b37892f131.jpg)  
Figure 38: Pairwise win rate in the IBS on the large held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a lower IBS.

![](images/a0e28153fc762652f429107d7f70a24addf4940bda3d6adf9595250805570ffd.jpg)  
Figure 39: Pareto front of mean normalized IBS (higher is better) and fit + predict time. We report average performances over the large held-out datasets. IBS was normalized to provide comparability across all datasets.

## D.3.2 C-INDEX

Results for the C-index on large held-out datasets are shown in Figures 40, 41, 42, 43.

![](images/d51dfadd4e9a21b8a2538367e972c4a0287d828ced82283828a253d6a309d031.jpg)  
Figure 40: C-index at the 50th quantile on the large held-out datasets, compared to the Cindex obtained with a Kaplan-Meier estimator (higher is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/27a31233e59883982fa039c4b5ea96f41552629b5997ccb1aeee1cead76962da.jpg)  
Figure 41: Critical difference diagram of average ranks on the large held-out datasets (lower is better) for the C-index at the 50th quantile. Methods joined by a bar are not significantly different.

![](images/ef27f31da2b42d85c98c66857c902b45368921f16da1d1570e2e81da3c61603a.jpg)  
Figure 42: Pairwise win rate in the C-index at the 50th quantile on the large held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a higher C-index.

![](images/f5eeb53cf2d6d56511300be290673bc7dd25a76ebafec85561103062b4c41e80.jpg)  
Figure 43: Pareto front of mean C-index on the 50th quantile (higher is better) and fit + predict time. We report average performances over the large held-out datasets.

## D.3.3 KM-CALIBRATION

Results for the KM-calibration on large held-out datasets are shown in Figures 44, 45, 46.

![](images/e551383bffd6af771cc8d301b8d8d5b122cddb3d62a6bebdb6084e4a3b2569de.jpg)  
Figure 44: KM calibration on the large held-out datasets (lower is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/437a83bcfe7e80da93a82aee541e260db61249c4d455680973a5e2307f606072.jpg)  
Figure 45: Critical difference diagram of average ranks on the large held-out datasets (lower is better) for the KM calibration. Methods joined by a bar are not significantly different.

![](images/09386b9b62a5c970942a60baeff53501136f9450964646928534b64accb7a0a8.jpg)  
Figure 46: Pairwise win rate in the KM calibration on the large held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a better KM calibration.

## D.3.4 MAE

Results for the MAE on large held-out datasets are shown in Figures 47, 48, 49.

![](images/b556d21c14e0a0b98acd03815acd3fc1ef89740e0336474e9c7a373e8c2f4f37.jpg)  
Figure 47: MAE on the large held-out datasets, compared to the MAE obtained with a Kaplan-Meier estimator (higher is better). The points show individual dataset performances, and a boxplot summarizes the overall distribution. The dot color represents the dataset size.

![](images/a2b4b8fb6dfe68a0338ec4b53a52985b1c2b225ba917a141d62999587d1d8a4e.jpg)  
Figure 48: Critical difference diagram of average ranks on the large held-out datasets (lower is better) for the MAE. Methods joined by a bar are not significantly different.

![](images/4c380c2855a02c00fa1899d266bf9ed9a93fc8654b2c9b1846b6e0114d69d144.jpg)  
Figure 49: Pairwise win rate in the MAE on the large held-out datasets: the fraction of (dataset, competitor) pairs on which a method achieves a lower MAE.

![](images/65a06fa286bdaf3cfac11f7c399db8b051d95250ad975320b74a1001b6ecf343.jpg)  
Figure 50: Pareto front of mean MAE (lower is better) and fit + predict time. We report average performances over the large held-out datasets.