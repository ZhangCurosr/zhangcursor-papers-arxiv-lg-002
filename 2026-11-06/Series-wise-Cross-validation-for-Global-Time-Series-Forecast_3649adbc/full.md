# Series-wise Cross-validation for Global Time Series Forecasting

Vitor Cerqueira   
University of Coimbra, CISUC/LASI, DEI   
Coimbra, Portugal   
Mariana Oliveira   
Faculty of Engineering, University of Porto   
Porto, Portugal   
Luis Roque   
Faculty of Engineering, University of Porto   
Porto, Portugal

vitorc@dei.uc.pt

mariana.oliveira@fe.up.pt

luis\_roque<sup>@</sup>live.com

## Abstract

We consider the problem of cross-validation (CV) in time series forecasting, with a focus on global forecasting models based on deep learning. The prevailing CV strategy in forecasting is to partition data across time with the last part of each series used for testing. With global models trained on collections of time series, however, a fundamentally diferent approach becomes possible: partitioning across series and time. We formalize series-wise CV for global forecasting models and benchmark several instantiations, including K-fold CV, bootstrapping, and Monte Carlo CV, on 11 datasets with 10 neural network architectures. We evaluate CV procedures along five dimensions: performance estimation, overoptimism rate, model selection accuracy, regret, and downstream forecasting performance. We withhold a random fraction of series from training so that they appear only in the test set. This protocol simulates the emergence of new series not seen during training, a common scenario in global forecasting. Under this protocol, our results reveal a nuanced trade-of: series-wise CV yields better downstream forecasting accuracy by guiding model selection toward configurations that generalize across both time and series, while time-wise CV retains a slight advantage for estimating out-of-sample error.

Keywords: Time series forecasting, Cross-validation, Model selection, Series-wise Crossvalidation, Global forecasting models, Distribution shift

## 1 Introduction

## 1.1 Context and Motivation

Cross-validation (CV) is the predominant approach for estimating performance and selecting models in machine learning [3, 5]. Evaluating models in held-out data yields more reliable estimates than in-sample or goodness-of-fit measures [30], and enables the selection of algorithms that generalize well in unseen data.

With the rise of deep learning in forecasting [1], state-of-the-art models are increasingly global: a single model is trained on a collection of time series, rather than fitting one model per series as in classical methods such as ARIMA. In both global and local approaches, the standard evaluation protocol is to hold out the last observations of a given series for testing. We call this time-wise CV (Figure 1a).

![](images/c388a748a047674090729c0638e9bc263401b7a8d6ad09ff99086345b58d171d.jpg)  
(a) Time-wise CV

![](images/9c45fc9ec352b6ab19bfcd6c27bfd67d26f51ef925edeabe8e1c24b2b9ef7458.jpg)  
(b) Series-wise CV  
Figure 1: Illustration of time-wise CV (a) and series-wise CV (b). In time-wise CV, all series contribute to both training and testing, with only the temporal dimension used for splitting. In series-wise CV, the split is across series and time: one subset is used for training, a disjoint subset for testing, and both are further split in time to preserve temporal order.

Time-wise CV preserves temporal order and emulates a realistic deployment scenario, but training and validation data come from the same series and difer only in time. Thus, raising questions about generalization to entirely new series, as opposed to temporal extrapolation on series whose history was used for training. Alternative splitting strategies, such as blocking [6] or embargoing [28], have been explored for single time series; however, time-wise splitting yields better performance estimates in that setting [9]. Moreover, most previous work about time series forecasting has focused on local models fit to individual series, leaving a gap in our understanding of CV for global forecasters trained on many series.

Datasets comprising multiple time series open a new possibility: partitioning across series as well as time. In series-wise CV (Figure 1b), models are trained on the history of one subset of series and evaluated on the future of a disjoint subset. This dual partitioning, across both the series and temporal dimensions, is the focus of this paper.

## 1.2 Objectives and Contributions

We hypothesize that series-wise CV can improve downstream forecasting accuracy by encouraging the selection of models that generalize across both time and the space of time series. In time-wise CV, validation data come from the same series used for training, difering only in time; the selected model is therefore optimized for temporal extrapolation on known dynamics. Series-wise CV, by contrast, evaluates candidates on entirely unseen series, rewarding configurations that not only generalize well across time but also transfer well to new patterns. This objective is especially relevant as time series forecasting has moved from purely local models to global models and, more recently, to time series foundation models [2]. These models are applied to populations of series that evolve over time, in which new entities arrive and the dynamics of existing series shift [17].

This paper makes the following contributions.

1. We formalize series-wise CV for global forecasting models and present several instantiations, such as K-fold CV or bootstrapping, that partition datasets across both series and time.

2. We design a benchmark that evaluates CV procedures along five dimensions: (i) performance estimation (how close CV estimates are to the true out-of-sample error); (ii) overoptimism rate (how often the actual loss is greater than the estimated one); (iii) selection accuracy (how often CV picks the best algorithm); (iv) regret (gap between the selected model and the best attainable); and (v) downstream forecasting performance, treating each CV procedure as part of the full forecasting pipeline.

3. We experiment on 11 benchmark time series datasets using 10 well-established neural network architectures. Our evaluation protocol simulates the emergence of new series in the test set that were not seen during training, a common scenario in global forecasting. The results reveal a nuanced trade-of: series-wise CV yields better downstream forecasting performance while time-wise CV retains a slight edge for performance estimation. In other words, series-wise CV guides model selection toward configurations that generalize better, even though time-wise splits produce CV estimates closer to out-of-sample error.

The code to reproduce the experiments is available at https://github.com/vcerqueira/ experiments-tscv.

## 2 Background

This section reviews global time series forecasting and the standard practice of CV for temporally ordered data. In forecasting, CV holds out the future of a series so that temporal order is preserved. Studies of this approach have focused on individual series, so how to evaluate models trained jointly on many series remains an open question, especially in scenarios where new series emerge in the dataset.

## 2.1 Time series forecasting

Let $\mathcal { Y } = \{ Y ^ { i } \} _ { i = 1 } ^ { n }$ denote a collection of n time series. Each series ${ \cal Y } ^ { i } = \{ y _ { j } ^ { i } \} _ { j = 1 } ^ { n _ { i } }$ is a sequence of numeric observations indexed by time $j ,$ , with $n _ { i }$ the length of the i-th series. The forecasting task is to predict the next h values of a series given its past; we write the target as $Y _ { t + 1 : t + h } ^ { i } = \{ y _ { t + 1 } ^ { i } , \dots , y _ { t + h } ^ { i } \}$

Global forecasting models train a single model on Y, exploiting shared structure across series and often improving accuracy and data eficiency relative to local (per-series) models $[ 1 6 ,$ 13]. Using a context window of length $p ,$ each series is turned into input-output pairs via time-delay embedding [8]: at each time $t ,$ the input is the window $X _ { t } ^ { i } \in \mathbb { R } ^ { p }$ and the output is the subsequent horizon $Y _ { t + 1 : t + h } ^ { i } \in \mathbb { R } ^ { h }$

$$
X _ { t } ^ { i } = \{ y _ { t - p + 1 } ^ { i } , \ldots , y _ { t } ^ { i } \} , \qquad Y _ { t + 1 : t + h } ^ { i } = \{ y _ { t + 1 } ^ { i } , \ldots , y _ { t + h } ^ { i } \} .\tag{1}
$$

Training amounts to fitting a global function $f _ { \theta } : \mathbb { R } ^ { p }  \mathbb { R } ^ { h }$ so that $\hat { Y } _ { t + 1 : t + h } ^ { i } = f _ { \theta } ( X _ { t } ^ { i } )$ approximates the true future values in all series and time steps.

Deep learning ofers a wide variety of approaches to time series forecasting, including recurrent, convolutional, attention-based, and feed-forward designs. In large-scale benchmark competitions and standardized accuracy studies, global models based on deep learning have repeatedly achieved state-of-the-art results relative to classical statistical methods [1, 24], which has driven their adoption for forecasting.

## 2.2 Time series cross-validation

The key design choice in CV is how to partition the data into training and validation sets. For i.i.d. data, CV is straightforward: observations can be randomly assigned to training and validation folds [3]. Time series violate the i.i.d. assumption, so the partitioning of observations must take temporal order into account. Many variants of randomized CV have been proposed to recover the data eficiency of k-fold CV (e.g., blocking [6] or embargoing [28]), but in most forecasting settings it is widely held that credible evaluation requires preserving temporal order, with test observations corresponding to more recent observations than those in the training set [9, 15].

Several studies have benchmarked CV strategies under temporal constraints for forecasting and related dependent-data settings [6, 7, 9, 26]. Much of it focuses on whether CV recovers the true generalization error; a smaller body also studies how the choice of CV afects model selection [10]. All of this work on univariate time series forecasting considers single time series (local models fit per series) [9] or assumes spatial dependence between series [26]. To our knowledge, there are no existing studies of global forecasters trained jointly in many series in the context addressed in this paper, which motivates the series-wise perspective in the next section.

## 3 Series-wise Time Series CV

This section formally introduces series-wise time series CV, describing its key principles and nested application. We also present how state-of-the-art estimators, such as bootstrapping or Monte Carlo simulation, can be applied for series-wise CV.

## 3.1 Definition and framework

The fundamental principle behind series-wise CV is that, at any given fold, the model is trained on one subset of time series and evaluated on another, disjoint subset, while also respecting temporal order. This separation provides a stricter evaluation than time-wise CV, where the model has already observed each series during training. Thus series-wise CV evaluates generalization not only across time but also across the space of time series, a desirable property when the goal is to select models that perform well on unseen series or under distribution shift.

Formally, series-wise CV is grounded on partitioning the collection Y into disjoint subsets. Let the dataset $\mathcal { Y } = \{ Y ^ { i } \} _ { i = 1 } ^ { n }$ be partitioned into non-overlapping subsets of series indices:

$$
{ \mathcal { T } } _ { \mathrm { t r a i n } } \subset \{ 1 , \ldots , n \} , \quad { \mathcal { T } } _ { \mathrm { t e s t } } = \{ 1 , \ldots , n \} \setminus { \mathcal { T } } _ { \mathrm { t r a i n } } ,
$$

such that the training and test sets are defined as:

$$
\mathcal { V } _ { \mathrm { t r a i n } } = \{ Y ^ { i } : i \in \mathcal { T } _ { \mathrm { t r a i n } } \} , \quad \mathcal { V } _ { \mathrm { t e s t } } = \{ Y ^ { i } : i \in \mathcal { T } _ { \mathrm { t e s t } } \} .
$$

Crucially, $\mathcal { V } _ { \mathrm { t r a i n } } \cap \mathcal { V } _ { \mathrm { t e s t } } = \emptyset$ , ensuring no time series appears in both subsets.

In addition to partitioning across series, series-wise CV preserves temporal constraints by further splitting each time series across time. For each series $Y ^ { i }$ with length $n _ { i }$ , we define a temporal split point $\tau _ { i } < n _ { i }$ such that:

$$
Y _ { \mathrm { h i s t o r y } } ^ { i } = \{ y _ { 1 } ^ { i } , \dots , y _ { \tau _ { i } } ^ { i } \} , \quad Y _ { \mathrm { f u t u r e } } ^ { i } = \{ y _ { \tau _ { i } + 1 } ^ { i } , \dots , y _ { n _ { i } } ^ { i } \} .
$$

During training, the model $f _ { \theta }$ is fitted using input-output pairs constructed from $\{ Y _ { \mathrm { h i s t o r y } } ^ { i }$ $i \in \mathcal { T } _ { \mathrm { t r a i n } } \}$ . During evaluation, the model is tested on forecasting $Y _ { \mathrm { f u t u r e } } ^ { i }$ for series $i \in \mathcal { T } _ { \mathrm { t e s t } }$ using only the corresponding $Y _ { \mathrm { h i s t o r y } } ^ { i }$ as context. This dual partitioning, across both series and time, distinguishes series-wise CV from conventional time-wise CV, where all series contribute to both training and testing, with only the temporal dimension used for splitting. The series-wise CV framework can be expressed as:

$$
\hat { Y } _ { \mathrm { f u t u r e } } ^ { i } = f _ { \theta } ( Y _ { \mathrm { h i s t o r y } } ^ { i } ) , \quad \forall i \in \mathcal { I } _ { \mathrm { t e s t } } ,\tag{2}
$$

where $f _ { \theta }$ is trained exclusively on series from $\mathcal { T } _ { \mathrm { t r a i n } }$

The performance estimate for a given fold is computed by aggregating forecasting errors over the test series:

$$
\hat { \mathcal { L } } = \frac { 1 } { | \mathcal { T } _ { \mathrm { t e s t } } | } \sum _ { i \in \mathcal { T } _ { \mathrm { t e s t } } } \ell \left( Y _ { \mathrm { f u t u r e } } ^ { i } , \hat { Y } _ { \mathrm { f u t u r e } } ^ { i } \right) ,\tag{3}
$$

where $\ell ( \cdot , \cdot )$ denotes a loss function. When a strategy produces multiple folds, the overall performance estimate is typically the mean of the per-fold losses $\hat { \mathcal { L } }$ over all folds.

## 3.2 Strategies

The specific manner in which $\mathcal { T } _ { \mathrm { t r a i n } }$ and $\mathcal { T } _ { \mathrm { t e s t } }$ are constructed defines diferent series-wise CV strategies, such as the following:

• Holdout: The classical and simplest approach is to partition the n series into a training set and a test set by assigning a random subset of series (e.g., a fixed percentage, often 60–80%) to $\mathcal { T } _ { \mathrm { t r a i n } }$ and the remainder to $\mathcal { T } _ { \mathrm { t e s t } }$ . Holdout is most reliable when the number of series is large enough that the held-out test set is representative of the population [9].

• Repeated Holdout (K-Holdout): A single holdout split can yield unrepresentative train or test sets. Repeating the holdout K times with diferent random partitions and averaging the performance estimates yields more stable and reliable estimates.

• Monte Carlo CV (K-MC-CV): For each of K repetitions, a random subset of series is drawn for training, a disjoint random subset for testing, and the remaining series are left unused in that repetition. This difers from K-Holdout, where the training and test sets together cover all n series in every repetition. Diferent variants of Monte Carlo CV have shown competitive performance estimates in other data-dependent scenarios [3].

• Bootstrap: Whereas holdout uses sampling without replacement, bootstrap constructs $\mathcal { T } _ { \mathrm { t r a i n } }$ by sampling n series with replacement from $\{ 1 , \ldots , n \} ; \mathcal { T } _ { \mathrm { t e s t } }$ is the set of indices that were not selected (out-of-bag) (on average, 63.2% of series appear at least once in the training set [32]).

![](images/d385310046baf677a806af9ed3178ed2a0a273079cfd50a6c07dc437c9af7659.jpg)

![](images/7bf330aa468e57cfac85ed5f3a8e21cc41996c2285f671065b095ac7174fc297.jpg)  
(a) K-fold CV with k=5

![](images/47f30377dee11af1f570a0bcf0fc4b1374a1ec536380fd8160888fa23745b4fe.jpg)

![](images/bf4f5825e569a5cd77760353d5acc71fdc0b5194461602fb87cea997c3ac89bc.jpg)  
(b) Holdout

![](images/839342f00f38300f2bb0eb46c8e593f0db933bee3148e6a4a18ec033fdf83507.jpg)  
(c) Monte Carlo CV (example fold)

![](images/be7ee4b7a72c0019b710a8901a2a03816eb1989ff7a3c0ded273bce4d756db36.jpg)  
(d) Bootstrap  
Figure 2: Illustration of series-wise CV partitioning schemes. Each horizontal bar represents a time series; blue indicates training data and red indicates test data. (a) K-fold CV with $K { = } 5 3$ the series are split into 5 disjoint folds, and each fold serves as the test set exactly once. (b) Holdout: a single random split assigns most series to training and the remainder to testing. (c) Monte Carlo CV: a random subset of series is used for training and a disjoint subset for testing; gray indicates unused series in that iteration. (d) Bootstrap: training series are sampled with replacement (darker shades indicate that a series may appear multiple times in the training set) and out-of-bag series form the test set.

• K-Bootstrap: As with Holdout, repeating the bootstrap procedure K times and averaging reduces variance in the performance estimate.

• K-fold Cross-validation (Kf-CV): Holdout and bootstrap do not guarantee that every series is used for testing. In K-fold CV, the series indices $\{ 1 , \ldots , n \}$ are partitioned into K disjoint blocks. Each fold uses one block as $\mathcal { T } _ { \mathrm { t e s t } }$ and the union of the remaining blocks as $\mathcal { T } _ { \mathrm { t r a i n } }$ , so every series is tested exactly once and all data contribute to the estimate.

Figure 2 illustrates these partitioning schemes; blue regions denote training data and red regions denote test data.

![](images/c7151fa8aabadfab150735e785f46064359b06f74ca2d2be47f1e4dc800e751b.jpg)  
Figure 3: Nested time-wise CV. The outer loop partitions data across time for performance estimation; the inner loop further splits the outer training portion across time to select hyperparameters. All series contribute to both training and testing at every level.

## 3.3 Nested series-wise CV

In practice, CV serves both performance estimation and model selection: we use CV to estimate how well a model generalizes and to choose among algorithms or hyperparameter configurations [3]. Nested CV separates these roles by using an outer loop for performance estimation and an inner loop for model or hyperparameter selection on each outer training set. Figure 3 shows the time-wise approach for nested CV.

In nested series-wise CV, the outer loop applies one of the strategies from Section 3.2 to partition the collection Y into series-wise folds. For each outer fold, $\mathcal { T } _ { \mathrm { t r a i n } }$ and $\mathcal { T } _ { \mathrm { t e s t } }$ are defined as before; the model is fitted on the training series and evaluated on the test series. Within each outer fold, however, we must also select a model or tune hyperparameters using only the data from the outer training series.

The inner loop can partition the outer training series across time only, or across both series and time. In this work we adopt the simpler option: time-wise holdout within the inner loop. Concretely, for each outer fold the most recent observations of $\mathcal { T } _ { \mathrm { t r a i n } }$ serve as the inner validation set, and the remaining history serves as the inner training set. Once the best hyperparameters are selected, the model is retrained on all of $\mathcal { T } _ { \mathrm { t r a i n } }$ and evaluated on $\mathcal { T } _ { \mathrm { t e s t } }$ to produce one term of the outer performance estimate. This design mirrors standard nested CV practice, where the inner procedure is often a single holdout even when the outer loop is K-fold CV, efectively balancing statistical rigor with computational tractability. For the comparison of CV approaches in Section 4, this choice also isolates the efect of the outer CV strategy and enables a fair comparison across outer-loop partitioning methods. Figure 4 illustrates the complete workflow. The outer series-wise folds guarantee that performance is estimated on series unseen during both training and tuning.

![](images/da0b19251de970305313e85c44c0d8417766f731a43f97cd7d38e2a03575ad40.jpg)  
Figure 4: Nested series-wise CV. The outer loop partitions data across series and time for performance estimation: training series and test series are disjoint. The inner loop applies a time-wise holdout on the outer training series to select hyperparameters. This ensures that outer-test series are never seen during training or tuning.

## 4 Experiments

This section describes the experimental setup designed to benchmark the performance of diferent series-wise time series CV approaches for performance estimation and model selection of global forecasting models. The evaluation pipeline is described in Section 4.1. The datasets and learning algorithms are described in Sections 4.2 and 4.3, respectively.

## 4.1 Performance estimation evaluation methodology

Test set for CV methods To benchmark the performance of diferent CV approaches, we design an evaluation procedure that emulates a realistic scenario by holding out a test set of observations of each time series. The dataset Y is partitioned across time into two disjoint subsets, $\mathcal { V } _ { i n - s e t }$ and $\mathcal { \partial } _ { o u t - s e t }$ . The out-set $\mathcal { \partial } _ { o u t - s e t }$ is constructed by withholding the last $2 \times h$ observations from each time series, while the in-set $\mathcal { V } _ { i n - s e t }$ consists of all earlier observations:

$$
\begin{array} { r l r } { \mathcal { V } _ { i n - s e t } = \left\{ y _ { 1 : n _ { i } - 2 h } ^ { ( i ) } \right\} _ { i = 1 } ^ { n } , } & { { } } & { \mathcal { V } _ { o u t - s e t } = \left\{ y _ { n _ { i } - 2 h + 1 : n _ { i } } ^ { ( i ) } \right\} _ { i = 1 } ^ { n } . } \end{array}\tag{4}
$$

For all CV procedures applied to $\mathcal { D } _ { i n - s e t } .$ , the temporal test portion in each fold spans $\tau = h$ observations, matching the forecasting horizon. The in-set is used for training, hyperparameter tuning, and performance estimation by CV. The out-set is held out entirely and only used for the final model evaluation to compute the true forecasting loss. The $2 \times h$ window is consumed in a rolling origin manner [15], predicting the next h observations at each time step. In other words, the out-set is the test set for CV procedures, enabling the assessment of the performance estimates, selection accuracy, and forecasting performance.

Equation (4) splits the data across time only, so every series contributes to both the in-set and the out-set. To emulate a population in which new series appear only at test time, we additionally restrict the in-set to a random subset of series. Let ${ \mathcal { S } } \subseteq \{ 1 , \ldots , n \}$ be a random subset of series indices with $| S | = \lfloor \rho n \rfloor$ and $\rho = 0 . 7$ . The in-set used by the CV procedures retains the earlier observations of these series only, while the out-set is unchanged:

$$
\begin{array} { r } { \mathcal { V } _ { i n - s e t } = \left\{ y _ { 1 : n _ { i } - 2 h } ^ { ( i ) } \right\} _ { i \in \mathcal { S } } , \qquad \mathcal { V } _ { o u t - s e t } = \left\{ y _ { n _ { i } - 2 h + 1 : n _ { i } } ^ { ( i ) } \right\} _ { i = 1 } ^ { n } . } \end{array}\tag{5}
$$

The remaining series, $\{ 1 , \ldots , n \} \setminus S$ , appear only in the out-set and are therefore unseen during training, tuning, and model selection, emerging as new series at test time. This emulates a setting where the population of series evolves over time, precisely where series-wise CV might shine and increasingly common in the age of global forecasting models and time series foundation models [2]. Setting $\rho = 1$ recovers the split in Equation (4), where all series appear in the in-set; we also report results for this case for completeness.

Nested CV All procedures use nested CV on the $\mathscr { P } _ { i n - s e t }$ only. For each candidate forecaster, the inner loop applies a time-wise split (cf. Section 3.3) to choose hyperparameters. The outer loop follows the chosen series-wise (or time-wise) CV scheme: for each outer fold, we retrain the model with the inner-selected hyperparameters on all outer-training data, evaluate on the outer-test portion of the in-set, and record the loss. Averaging these outer-fold losses yields the performance estimate for that forecaster under the given CV procedure. To score the CV procedure itself, we retrain each forecaster using its best hyperparameters on the entire in-set, according to each estimator, and evaluate on $\mathcal { D } _ { o u t - s e t }$

Forecasting performance metric We use MASE as the forecasting performance metric [23], with a global scaling variant. Standard MASE scales a model’s MAE by a seasonal naive $\mathrm { M A E } ;$ here, we define the scaling factor (Q) as the MAE of the seasonal naive baseline computed once over all series available in the in-set. For each CV fold, the numerator is the model MAE on that fold’s evaluated series/horizons. Using a fixed global (Q) keeps scores comparable across folds and CV procedures, even when evaluated series difer.

CV evaluation metrics We formalize the evaluation of CV procedures using the framework of the Combined Algorithm Selection and Hyperparameter optimization (CASH) problem [31]. Given a set of learning algorithms A and a dataset Y, the goal is to select the algorithm $A ^ { \star } \in { \mathcal { A } }$ that minimizes the expected loss on unseen data. For each algorithm $A \in A ,$ a CV procedure produces an estimated MASE loss $\hat { \mathcal { L } } ( A )$ computed on the in-set ${ \ y } _ { i n - s e t }$ . The true loss $\mathcal { L } ( A )$ is the MASE error incurred by algorithm A when evaluated on the held-out out-set $\mathcal { V } _ { o u t - s e t }$ . The CV procedure selects the algorithm with the lowest estimated loss, $A _ { s e l } = \arg \operatorname* { m i n } _ { A \in \mathcal { A } } \hat { \mathcal { L } } ( A )$ , while the oracle selects the algorithm with the lowest true loss, $A ^ { \star } = \arg \operatorname* { m i n } _ { A \in { \mathcal { A } } } { \mathcal { L } } ( A )$ . Based on these quantities, we define the following evaluation metrics across a collection of D datasets:

• MAPEE (Mean Absolute Performance Estimation Error) (e.g. [9]): The average absolute diference between the estimated MASE loss and the true MASE loss, averaged across all algorithms and datasets:

$$
\mathrm { M A P E E } = \frac { 1 } { D } \frac { 1 } { | \cal A | } \sum _ { d = 1 } ^ { D } \sum _ { A \in \cal A } \left| \hat { \mathcal { L } } _ { d } ( A ) - \mathcal { L } _ { d } ( A ) \right| .
$$

• Overoptimism Rate: The proportion of dataset and model pairs in which the true MASE loss was under-estimated by the CV procedure, considering that in certain applications under-estimating the error may be more costly than over-estimating it:

$$
\mathrm { O v e r o p t i m i s m } = \frac { 1 } { D } \frac { 1 } { | A | } \sum _ { d = 1 } ^ { D } \sum _ { A \in \mathcal { A } } \mathbb { 1 } \left[ \hat { \mathcal { L } } _ { d } ( A ) < \mathcal { L } _ { d } ( A ) \right] .
$$

• Selection Accuracy: The proportion of datasets where the CV-selected model is also the oracle-selected model:

$$
\mathrm { A c c u r a c y } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \mathbb { 1 } \left[ A _ { s e l } ^ { ( d ) } = A ^ { \star ( d ) } \right] .
$$

• Regret: The average diference between the true loss of the selected model and the true loss of the oracle model. Regret captures robustness: a procedure may occasionally miss the optimum, yet still select an accurate alternative [10].<sup>1</sup>

$$
\mathrm { R e g r e t } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \left( \mathcal { L } _ { d } ( A _ { s e l } ^ { ( d ) } ) - \mathcal { L } _ { d } ( A ^ { \star ( d ) } ) \right) .
$$

• Downstream Forecasting Performance: Ultimately, a CV procedure is part of a forecasting workflow. For a given CV procedure g, we apply it to the in-set to select an algorithm $A _ { s e l } ^ { g } ,$ retrain on the entire in-set, and evaluate on the out-set according to MASE. This metric directly captures the end-to-end value of a CV procedure: a method that consistently selects well-generalizing models will achieve better downstream forecasting accuracy.

We repeat the complete evaluation procedure for 3 diferent seeds (123, 456, 789) to assess the robustness of the results for distinct initializations. We report the average score in the main body of the paper, while the appendix contains the per-seed results.

The overall protocol for scoring a CV procedure is illustrated in Figure 5, detailing the flow of information required for computing each metric. In the outer fold, the plot shows an example partitioning. Diferent strategies can be applied based on the list in Section 3.2.

## 4.2 Datasets

We conduct experiments on 11 benchmark datasets spanning diverse domains, sampling frequencies, and forecasting horizons: Tourism [4] (monthly and quarterly), M1 [21] (monthly and quarterly), M3 [22] (monthly and quarterly), ECL [34], Exchange [18], Hospital [14], Trafic [14], and Weather [34]. Trafic and ECL were originally collected at an hourly frequency, and Weather at a 10-minute frequency; we resample these datasets to daily frequency by taking the mean to improve computational eficiency and make the experiments more tractable. The datasets are described in Table 1.

![](images/e06960b6f770f778fa9445ec3b9b9b7ee5458df56aa25e31106b3ef51e95a370.jpg)  
Figure 5: Evaluation protocol and flow of information for scoring a CV procedure.

## 4.3 Methods

For forecasting models we consider 10 deep learning architectures: MLP, DLinear [33], NLinear [33], TiDE [12], NHITS [11], NBEATS [27], KAN [20], DeepNPTS [29], PatchTST [25], and TFT [19]. For each model, hyperparameters are tuned with 20 trials of random search within the nested CV procedure described above. Complete model-specific hyperparameter search spaces and fixed random seeds are provided in the supplementary code repository (https://github.com/vcerqueira/experiments-tscv).

We remark that our goal is to compare CV techniques, not to benchmark neural architectures against one another. We therefore prioritize well-established, computationally eficient architectures that allow large-scale experimentation. The architecture pool should be viewed as a representative testbed for evaluating CV methodologies rather than a ranking of the most accurate forecasters.

We benchmark all series-wise CV methods listed in Section 3.2, along with the standard time-wise CV approach (Time-CV) which is typically used in time series forecasting (e.g. [27, 19, 25, 33, 12]). The number of folds for K-Holdout, K-MC-CV, K-Bootstrap, and Kf-CV is set to 5. In Holdout and K-Holdout, the percentage of the training set is 70% of the time series. For each fold in K-MC-CV, the training and testing sets are composed of 50% and 20% of the time series, while the remaining 30% are unused.

Table 1: Summary of the datasets used in the experimental setup, including the number of time series, number of observations, forecast horizon, and frequency.
<table><tr><td>Dataset</td><td>Frequency  $\#$ </td><td>time series</td><td> $\#$  observations</td><td>Horizon (h)</td></tr><tr><td rowspan="2">Tourism</td><td>Monthly</td><td>366</td><td>109,280</td><td>12</td></tr><tr><td>Quarterly</td><td>427</td><td>42,544</td><td>4</td></tr><tr><td rowspan="2">M1</td><td>Monthly</td><td>617</td><td>55,998</td><td>12</td></tr><tr><td>Quarterly</td><td>203</td><td>9,944</td><td>4</td></tr><tr><td rowspan="2">M3</td><td>Monthly</td><td>1,428</td><td>167,562</td><td>12</td></tr><tr><td>Quarterly</td><td>756</td><td>37,004</td><td>4</td></tr><tr><td rowspan="3">Exchange Weather Traffic</td><td>Daily</td><td>8</td><td>60,704</td><td>14</td></tr><tr><td>Daily</td><td>21</td><td>7,707</td><td>14</td></tr><tr><td>Daily</td><td>862</td><td>630,984</td><td>14</td></tr><tr><td>ECL</td><td>Daily</td><td>321</td><td>352,137</td><td>14</td></tr><tr><td>Hospital</td><td>Monthly</td><td>767</td><td>64,428</td><td>12</td></tr><tr><td>Total</td><td></td><td>5,776</td><td>1,538,292</td><td>一</td></tr></table>

## 5 Results

This section presents the results of the experiments, starting with the scores for performance estimation and model selection (Section 5.1) and then the scores for forecasting accuracy (Section 5.2).

## 5.1 Performance Estimation and Model Selection

Table 2 shows the results on performance estimation and model selection. Time-CV shows the best performance estimation error (MAPEE), followed by Bootstrap and Kf-CV. Holdout and K-MC-CV show the largest MAPEE. Overoptimism ranks the methods diferently. K-Bootstrap and Kf-CV are the least overoptimistic approaches (∼0.26), while K-Holdout and K-MC-CV also stay near that score. Time-CV (0.324) under-estimates the loss more often than each of these series-wise procedures.

Regarding model selection, the accuracy of the estimators ranges between 15.2% and 36.4%, which is better than random selection (1 over 10 alternatives, which equals to 10% accuracy on average). Holdout shows the highest accuracy, followed by K-MC-CV. Concerning the regret of each CV procedure, K-MC-CV also shows the best score, followed by Holdout and K-Bootstrap. These results suggest that time-wise partitioning gives the smallest absolute error in the performance estimate, while series-wise procedures under-estimate that error less often. Series-wise CV also provides competitive alternatives for model selection based on the selection accuracy and regret scores.

Table 2: Results of performance estimation (MAPEE), overoptimism rate, selection accuracy, and regret for each CV approach. Numbers in bold (underlined) denote the best (second-best) approach.
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Method</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>MAPEE</td><td>0.120</td><td>0.141</td><td>0.124</td><td>0.140</td><td>0.126</td><td>0.132</td><td>0.109</td></tr><tr><td>Overoptimism</td><td>0.315</td><td>0.373</td><td>0.258</td><td>0.267</td><td>0.255</td><td>0.261</td><td>0.324</td></tr><tr><td>Accuracy</td><td>0.242</td><td>0.364</td><td>0.242</td><td>0.303</td><td>0.152</td><td>0.212</td><td>0.273</td></tr><tr><td>Regret</td><td>0.023</td><td>0.019</td><td>0.044</td><td>0.016</td><td>0.019</td><td>0.020</td><td>0.034</td></tr></table>

Table 3: MASE score of each CV approach on each dataset, and respective average rank, average MASE, and top 2 counts (averaged across seeds).
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Methods</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>ECL</td><td>0.596</td><td>0.592</td><td>0.591</td><td>0.595</td><td>0.600</td><td>0.604</td><td>0.577</td></tr><tr><td>Exchange</td><td>0.151</td><td>0.153</td><td>0.147</td><td>0.145</td><td>0.145</td><td>0.147</td><td>0.155</td></tr><tr><td>Traffic</td><td>0.702</td><td>0.671</td><td>0.688</td><td>0.691</td><td>0.691</td><td>0.677</td><td>0.692</td></tr><tr><td>Weather</td><td>0.976</td><td>0.920</td><td>0.939</td><td>0.928</td><td>0.940</td><td>0.934</td><td>1.115</td></tr><tr><td>Hospital</td><td>0.811</td><td>0.814</td><td>0.797</td><td>0.796</td><td>0.804</td><td>0.804</td><td>0.804</td></tr><tr><td>M1-M</td><td>0.904</td><td>0.901</td><td>0.909</td><td>0.912</td><td>0.903</td><td>0.903</td><td>0.920</td></tr><tr><td>M1-Q</td><td>1.124</td><td>1.138</td><td>1.120</td><td>1.131</td><td>1.121</td><td>1.114</td><td>1.124</td></tr><tr><td>M3-M</td><td>0.813</td><td>0.797</td><td>0.755</td><td>0.799</td><td>0.797</td><td>0.786</td><td>0.737</td></tr><tr><td>M3-Q</td><td>0.832</td><td>0.827</td><td>0.891</td><td>0.819</td><td>0.848</td><td>0.865</td><td>0.874</td></tr><tr><td>Tourism-M</td><td>1.320</td><td>1.320</td><td>1.312</td><td>1.333</td><td>1.333</td><td>1.321</td><td>1.323</td></tr><tr><td>Tourism-Q</td><td>1.275</td><td>1.304</td><td>1.494</td><td>1.278</td><td>1.307</td><td>1.290</td><td>1.387</td></tr><tr><td>Avg. Rank</td><td>4.303</td><td>3.545</td><td>3.561</td><td>3.712</td><td>4.030</td><td>3.970</td><td>4.879</td></tr><tr><td>Avg</td><td>0.864</td><td>0.858</td><td>0.877</td><td>0.857</td><td>0.863</td><td>0.859</td><td>0.883</td></tr><tr><td>Top 2 Count</td><td>3.667</td><td>5.000</td><td>5.667</td><td>4.333</td><td>3.333</td><td>5.000</td><td>2.000</td></tr></table>

## 5.2 Downstream Forecasting Accuracy

Table 3 shows the downstream forecasting performance with each CV procedure as part of the forecasting pipeline (cf. Section 4.1). In this scenario, time-wise CV shows the worst average rank, being outperformed by all series-wise CV variants. Holdout shows the best average rank of 3.545 and the second best average MASE (0.858), and attains the top 2 score for 5 out of the 11 datasets. These results support the hypothesis that series-wise CV can improve downstream forecasting relative to time-wise CV. It is worth noting that Holdout is an approach that conducts a single holdout split of the in-set to select the model, rather than a K-fold procedure. As mentioned before, the scores presented above are the average of the 3 seeds. For further details, the per-seed results are available in the appendix (Tables 6–11). In those, Holdout shows the best forecasting performance for 2 out of the 3 seeds, with K-Holdout showing the best performance for the other seed. Across all seeds, time-wise

Table 4: Results of performance estimation (MAPEE), overoptimism rate, selection accuracy, and regret for each CV approach with a complete in-set.
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Method</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>MAPEE</td><td>0.119</td><td>0.109</td><td>0.108</td><td>0.127</td><td>14.321</td><td>0.105</td><td>0.092</td></tr><tr><td>Overoptimism</td><td>0.306</td><td>0.424</td><td>0.255</td><td>0.252</td><td>0.242</td><td>0.261</td><td>0.306</td></tr><tr><td>Accuracy</td><td>0.455</td><td>0.273</td><td>0.273</td><td>0.273</td><td>0.273</td><td>0.364</td><td>0.333</td></tr><tr><td>Regret</td><td>0.025</td><td>0.024</td><td>0.022</td><td>0.020</td><td>0.019</td><td>0.020</td><td>0.012</td></tr></table>

CV systematically underperforms the series-wise CV variants in downstream forecasting accuracy.

## 5.3 Results with a Complete In-set

The results so far employ an evaluation protocol that holds out a random fraction of series from the in-set so that they appear only in the out-set. This protocol simulates the emergence of new series not seen during training, a common scenario in global forecasting. In this section, we also present the results where the in-set contains all time series. The results are shown in Table 4 and Table 5.

When all series appear in the in-set, the forecasting comparison changes. Time-CV attains the best MAPEE and regret and the best forecasting accuracy. On overoptimism, time-wise CV is outperformed by several series-wise approaches, as in Table 2. Selection accuracy is highest for Bootstrap, with time-wise CV being third best.

Overall, the forecasting advantage of series-wise CV disappears once no new series emerge at test time, while its tendency for overoptimism remains. This suggests that the choice of CV procedure should match the expected population dynamics. Series-wise CV is preferable when the population of series is expected to evolve and new series appear at test time. Time-wise CV remains the better choice for downstream accuracy when the population is static. When under-estimating the error is costly, series-wise procedures are efective approaches in both regimes.

## 6 Discussion

## 6.1 Key findings

This paper formalizes series-wise CV for global time series forecasting, partitioning data across both series and time, and empirically compares several instantiations against the prevailing time-wise CV baseline. Our experiments span 11 datasets and 10 neural architectures, evaluating CV procedures along diferent dimensions: performance estimation accuracy, model selection quality, and downstream forecasting performance. The key findings are:

• Performance estimation. Time-wise CV yields the lowest estimation error (Table 2), indicating that temporal partitioning remains the most reliable strategy for approximating out-of-sample loss. However, time-wise CV also tends to produce overoptimistic estimates more often than several series-wise CV procedures.

Table 5: MASE score of each CV approach on each dataset, and respective average rank, average MASE, and top 2 counts with a complete in-set.
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Methods</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>ECL</td><td>0.592</td><td>0.583</td><td>0.587</td><td>0.598</td><td>0.594</td><td>0.596</td><td>0.594</td></tr><tr><td>Exchange</td><td>0.147</td><td>0.142</td><td>0.148</td><td>0.148</td><td>0.143</td><td>0.144</td><td>0.156</td></tr><tr><td>Traffic</td><td>0.686</td><td>0.701</td><td>0.686</td><td>0.684</td><td>0.687</td><td>0.683</td><td>0.686</td></tr><tr><td>Weather</td><td>0.922</td><td>0.936</td><td>0.930</td><td>0.926</td><td>0.926</td><td>0.933</td><td>0.941</td></tr><tr><td>Hospital</td><td>0.813</td><td>0.797</td><td>0.801</td><td>0.797</td><td>0.808</td><td>0.797</td><td>0.800</td></tr><tr><td>M1-M</td><td>0.907</td><td>0.906</td><td>0.916</td><td>0.910</td><td>0.919</td><td>0.921</td><td>0.904</td></tr><tr><td>M1-Q</td><td>1.276</td><td>1.209</td><td>1.124</td><td>1.196</td><td>1.186</td><td>1.175</td><td>1.127</td></tr><tr><td>M3-M</td><td>0.779</td><td>0.763</td><td>0.762</td><td>0.778</td><td>0.780</td><td>0.750</td><td>0.761</td></tr><tr><td>M3-Q</td><td>0.860</td><td>0.842</td><td>0.845</td><td>0.838</td><td>0.832</td><td>0.821</td><td>0.821</td></tr><tr><td>Tourism-M</td><td>1.316</td><td>1.334</td><td>1.335</td><td>1.316</td><td>1.327</td><td>1.321</td><td>1.303</td></tr><tr><td>Tourism-Q</td><td>1.288</td><td>1.290</td><td>1.336</td><td>1.256</td><td>1.255</td><td>1.340</td><td>1.244</td></tr><tr><td>Avg. Rank</td><td>4.076</td><td>4.091</td><td>4.364</td><td>3.909</td><td>4.167</td><td>3.939</td><td>3.455</td></tr><tr><td>Avg</td><td>0.871</td><td>0.864</td><td>0.861</td><td>0.859</td><td>0.860</td><td>0.862</td><td>0.849</td></tr><tr><td>Top 2 Count</td><td>4.667</td><td>3.667</td><td>2.333</td><td>3.667</td><td>3.667</td><td>3.667</td><td>4.333</td></tr></table>

• Model selection and regret. Series-wise methods, notably Holdout and K-MC-CV, achieve higher selection accuracy and lower regret than time-wise CV, showing that optimizing for generalization across series can improve the choice of model.

• Downstream forecasting. When CV is embedded in the full forecasting pipeline, almost all series-wise CV variants outperform time-wise CV on average rank and average MASE (Table 3). This outcome supports our central hypothesis: validating on unseen series steers selection toward configurations that generalize across both time and the space of series.

• Dynamics of the population of time series. Series-wise CV delivers better downstream accuracy when new series appear at test time. When only previously seen series remain in the population, time-wise CV is the stronger choice for downstream forecasting accuracy.

One important result from our experiments is that accurate performance estimation does not imply accurate model selection, corroborating previous works in this topic [10]. Time-wise CV produces estimates closer to the true out-of-sample error in absolute value, and series-wise procedures under-estimate that error less often. Series-wise CV also more often identifies the best-performing model. This outcome can be understood by the fact that time-wise CV rewards models that extrapolate well on familiar dynamics, while series-wise CV rewards models that transfer to unseen patterns. When the goal is to deploy a model that performs robustly across an evolving population of series, which is the typical use case for global forecasters, including time series foundation models, the latter objective appears more aligned with downstream success.

For practitioners building global forecasters, our results suggest the adoption of series-wise CV, particularly Holdout, K-Holdout or K-MC-CV, when the primary goal is downstream accuracy, and the use of time-wise CV when the magnitude of the error estimate matters most. When under-estimating the error is an important concern, series-wise procedures are also an efective choice.

## 6.2 Limitations and future work

Series-wise CV methods involving K folds incur roughly K-fold additional training time. However, this overhead is confined to ofline model development and does not afect inference latency. Moreover, folds can be executed in parallel, and our results show that even singlesplit series-wise variants (K=1) like Holdout outperform time-wise CV on downstream accuracy, ofering a low-cost entry point for practitioners with limited compute budgets.

It is also worth noting that we use a simplified time-wise CV in the inner loop of nested series-wise CV, mirroring common practice where the inner procedure is simpler than the outer loop [3]. On the other hand, this choice also isolates the efect of the outer CV strategy, which supports a fair comparison.

## 7 Conclusions

CV for time series forecasting has traditionally partitioned data across time alone. The shift toward global models, trained jointly on many series, opens a complementary axis: partitioning across series. In this paper, we formalized series-wise CV for global forecasting models, presented several instantiations, and benchmarked them against the prevailing time-wise baseline on 11 datasets and 10 neural architectures, along five dimensions spanning performance estimation, model selection, and downstream forecasting.

Series-wise CV is most valuable in the setting that increasingly defines global forecasting. When new series emerge at test time, it selects models that forecast more accurately downstream as it rewards transfer to unseen series rather than only temporal extrapolation on familiar ones. Global forecasters are often typically deployed in exactly this regime, on populations in which new series continue to appear. Time-wise CV remains the more reliable way to approximate the magnitude of out-of-sample error, and it is also preferable for downstream accuracy when the population of series is expected to remain static. Serieswise procedures under-estimate that error less often, in both the evolving and the static population.

The experimental results also point to several directions for future work. Our evaluation emulates an evolving population through a series-wise out-set in which a fraction of series are unseen during training; studying richer forms of distribution shift, such as gradual drift within series or varying fractions of emerging series, would further delineate when series-wise CV helps. We release our benchmark code to support these extensions and to facilitate the adoption of series-wise CV.

## Acknowledgements

This work is funded by national funds through FCT – Foundation for Science and Technology, I.P., within the scope of the research unit UID/00326 - Centre for Informatics and Systems of the University of Coimbra, https://doi.org/10.54499/UID/00326/2025. This work was partially funded by project AISym4Med (101095387) supported by Horizon Europe Cluster 1: Health, Connected-Health (n.o 46858), supported by Competitiveness and Internationalisation Operational Programme (POCI) and Lisbon Regional Operational Programme (LISBOA 2020), under the PORTUGAL 2020 Partnership Agreement, through the European Regional Development Fund (ERDF). AISym4Med is funded by the European Union under the Horizon Europe Framework Programme Grant Agreement Nº: 101095387. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Commission. Neither the European Union nor the European Health and Digital Executive Agency can be held responsible for them.

## References

[1] Taha Aksu, Gerald Woo, Juncheng Liu, Xu Liu, Chenghao Liu, Silvio Savarese, Caiming Xiong, and Doyen Sahoo. Gift-eval: A benchmark for general time series forecasting model evaluation. arXiv preprint arXiv:2410.10393, 2024.

[2] Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. arXiv preprint arXiv:2403.07815, 2024.

[3] Sylvain Arlot and Alain Celisse. A survey of cross-validation procedures for model selection. 2010.

[4] George Athanasopoulos, Rob J Hyndman, Haiyan Song, and Doris C Wu. The tourism forecasting competition. International Journal of Forecasting, 27(3):822–844, 2011.

[5] Stephen Bates, Trevor Hastie, and Robert Tibshirani. Cross-validation: what does it estimate and how well does it do it? Journal of the American Statistical Association, 119(546):1434–1445, 2024.

[6] Christoph Bergmeir and José M Benítez. On the use of cross-validation for time series predictor evaluation. Information Sciences, 191:192–213, 2012.

[7] Christoph Bergmeir, Rob J Hyndman, and Bonsoo Koo. A note on the validity of cross-validation for evaluating autoregressive time series prediction. Computational Statistics & Data Analysis, 120:70–83, 2018.

[8] Gianluca Bontempi, Souhaib Ben Taieb, and Yann-Aël Le Borgne. Machine learning strategies for time series forecasting. Business Intelligence: Second European Summer School, eBISS 2012, Brussels, Belgium, July 15-21, 2012, Tutorial Lectures 2, pages 62–77, 2013.

[9] Vitor Cerqueira, Luis Torgo, and Igor Mozetič. Evaluating time series forecasting models: An empirical study on performance estimation methods. Machine Learning, 109(11):1997–2028, 2020.

[10] Vitor Cerqueira, Luis Torgo, and Carlos Soares. Model selection for time series forecasting an empirical analysis of multiple estimators. Neural processing letters, 55(7):10073–10091, 2023.

[11] Cristian Challu, Kin G Olivares, Boris N Oreshkin, Federico Garza Ramirez, Max Mergenthaler Canseco, and Artur Dubrawski. Nhits: Neural hierarchical interpolation for time series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 6989–6997, 2023.

[12] Abhimanyu Das, Weihao Kong, Andrew Leach, Shaan Mathur, Rajat Sen, and Rose Yu. Long-term forecasting with tide: Time-series dense encoder. arXiv preprint arXiv:2304.08424, 2023.

[13] Rakshitha Godahewa, Kasun Bandara, Geofrey I Webb, Slawek Smyl, and Christoph Bergmeir. Ensembles of localised models for time series forecasting. Knowledge-Based Systems, 233:107518, 2021.

[14] Rakshitha Godahewa, Christoph Bergmeir, Geofrey I. Webb, Rob J. Hyndman, and Pablo Montero-Manso. Monash time series forecasting archive. In Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

[15] Hansika Hewamalage, Klaus Ackermann, and Christoph Bergmeir. Forecast evaluation for data scientists: common pitfalls and best practices. Data Mining and Knowledge Discovery, 37(2):788–832, 2023.

[16] Tim Januschowski, Jan Gasthaus, Yuyang Wang, David Salinas, Valentin Flunkert, Michael Bohlke-Schneider, and Laurent Callot. Criteria for classifying forecasting methods. International Journal of Forecasting, 36(1):167–177, 2020.

[17] Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In International conference on learning representations, 2021.

[18] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu. Modeling long-and short-term temporal patterns with deep neural networks. In The 41st international ACM SIGIR conference on research & development in information retrieval, pages 95–104, 2018.

[19] Bryan Lim, Sercan Ö Arık, Nicolas Loef, and Tomas Pfister. Temporal fusion transformers for interpretable multi-horizon time series forecasting. International Journal of Forecasting, 37(4):1748–1764, 2021.

[20] Ziming Liu, Yixuan Wang, Sachin Vaidya, Fabian Ruehle, James Halverson, Marin Soljačić, Thomas Y Hou, and Max Tegmark. Kan: Kolmogorov-arnold networks. arXiv preprint arXiv:2404.19756, 2024.

[21] Spyros Makridakis, Allan Andersen, Robert Carbone, Robert Fildes, Michele Hibon, Rudolf Lewandowski, Joseph Newton, Emanuel Parzen, and Robert Winkler. The accuracy of extrapolation (time series) methods: Results of a forecasting competition. Journal of forecasting, 1(2):111–153, 1982.

[22] Spyros Makridakis and Michele Hibon. The m3-competition: results, conclusions and implications. International journal of forecasting, 16(4):451–476, 2000.

[23] Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. The m4 competition: Results, findings, conclusion and way forward. International Journal offorecasting, 34(4):802–808, 2018.

[24] Spyros Makridakis, Evangelos Spiliotis, and Vassilios Assimakopoulos. M5 accuracy competition: Results, findings, and conclusions. International Journal of Forecasting, 2022.

[25] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. arXiv preprint arXiv:2211.14730, 2022.

[26] Mariana Oliveira, Luís Torgo, and Vítor Santos Costa. Evaluation procedures for forecasting with spatiotemporal data. Mathematics, 9(6):691, 2021.

[27] Boris N Oreshkin, Dmitri Carpov, Nicolas Chapados, and Yoshua Bengio. N-beats: Neural basis expansion analysis for interpretable time series forecasting. arXiv preprint arXiv:1905.10437, 2019.

[28] Jef Racine. Consistent cross-validatory model-selection for dependent data: hv-block crossvalidation. Journal of econometrics, 99(1):39–61, 2000.

[29] Syama Sundar Rangapuram, Jan Gasthaus, Lorenzo Stella, Valentin Flunkert, David Salinas, Yuyang Wang, and Tim Januschowski. Deep non-parametric time series forecaster. arXiv preprint arXiv:2312.14657, 2023.

[30] Leonard J Tashman. Out-of-sample tests of forecasting accuracy: an analysis and review. International journal of forecasting, 16(4):437–450, 2000.

[31] Chris Thornton, Frank Hutter, Holger H Hoos, and Kevin Leyton-Brown. Auto-weka: Combined selection and hyperparameter optimization of classification algorithms. In Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 847–855, 2013.

[32] Robert J Tibshirani. An introduction to the bootstrap.

[33] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers efective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 11121–11128, 2023.

[34] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond eficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 11106–11115, 2021.

Table 6: MASE score of each CV approach on each dataset, and respective average rank, average MASE, and top 2 counts (seed 123).
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Methods</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>ECL</td><td>0.586</td><td>0.595</td><td>0.585</td><td>0.586</td><td>0.586</td><td>0.606</td><td>0.574</td></tr><tr><td>Exchange</td><td>0.152</td><td>0.168</td><td>0.141</td><td>0.146</td><td>0.143</td><td>0.145</td><td>0.159</td></tr><tr><td>Traffic</td><td>0.700</td><td>0.675</td><td>0.690</td><td>0.698</td><td>0.690</td><td>0.690</td><td>0.695</td></tr><tr><td>Weather</td><td>1.063</td><td>0.921</td><td>0.939</td><td>0.939</td><td>0.939</td><td>0.939</td><td>1.148</td></tr><tr><td>Hospital</td><td>0.806</td><td>0.807</td><td>0.800</td><td>0.796</td><td>0.806</td><td>0.821</td><td>0.808</td></tr><tr><td>M1-M</td><td>0.910</td><td>0.901</td><td>0.901</td><td>0.910</td><td>0.901</td><td>0.901</td><td>0.932</td></tr><tr><td>M1-Q</td><td>1.125</td><td>1.115</td><td>1.123</td><td>1.154</td><td>1.125</td><td>1.115</td><td>1.120</td></tr><tr><td>M3-M</td><td>0.843</td><td>0.779</td><td>0.751</td><td>0.779</td><td>0.773</td><td>0.782</td><td>0.737</td></tr><tr><td>M3-Q</td><td>0.858</td><td>0.830</td><td>0.906</td><td>0.817</td><td>0.858</td><td>0.823</td><td>0.896</td></tr><tr><td>Tourism-M</td><td>1.296</td><td>1.296</td><td>1.296</td><td>1.321</td><td>1.359</td><td>1.320</td><td>1.323</td></tr><tr><td>Tourism-Q</td><td>1.309</td><td>1.239</td><td>1.914</td><td>1.275</td><td>1.290</td><td>1.309</td><td>1.434</td></tr><tr><td>Avg. Rank</td><td>5.000</td><td>3.136</td><td>3.273</td><td>3.909</td><td>3.773</td><td>4.000</td><td>4.909</td></tr><tr><td>Avg</td><td>0.877</td><td>0.848</td><td>0.913</td><td>0.856</td><td>0.861</td><td>0.859</td><td>0.893</td></tr><tr><td>Top 2 Count</td><td>1</td><td>6</td><td>8</td><td>4</td><td>4</td><td>5</td><td>2</td></tr></table>

## Appendix A. Results by Seed

The main body of the paper shows the results averaged across three seeds: 123, 456, 789. For transparency, Tables 6– 11 show the per-seed scores across all five dimensions. In general, the key conclusions remain. Time-wise CV is outperformed by almost all series-wise CV variants on downstream forecasting performance. On MAPEE it is best or second-best in each seed. The best approach for forecasting accuracy is not the same in every run: Holdout leads in two of the three seeds, and K-Holdout leads in the remaining one.

Table 7: MASE score of each CV approach on each dataset, and respective average rank, average MASE, and top 2 counts (seed 456).
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Methods</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>ECL</td><td>0.598</td><td>0.595</td><td>0.602</td><td>0.583</td><td>0.598</td><td>0.598</td><td>0.583</td></tr><tr><td>Exchange</td><td>0.151</td><td>0.142</td><td>0.145</td><td>0.149</td><td>0.141</td><td>0.141</td><td>0.147</td></tr><tr><td>Traffic</td><td>0.686</td><td>0.684</td><td>0.684</td><td>0.684</td><td>0.684</td><td>0.686</td><td>0.687</td></tr><tr><td>Weather</td><td>0.945</td><td>0.898</td><td>0.940</td><td>0.920</td><td>0.942</td><td>0.920</td><td>1.050</td></tr><tr><td>Hospital</td><td>0.840</td><td>0.813</td><td>0.795</td><td>0.797</td><td>0.801</td><td>0.796</td><td>0.803</td></tr><tr><td>M1-M</td><td>0.902</td><td>0.902</td><td>0.925</td><td>0.925</td><td>0.907</td><td>0.907</td><td>0.895</td></tr><tr><td>M1-Q</td><td>1.123</td><td>1.184</td><td>1.123</td><td>1.113</td><td>1.123</td><td>1.113</td><td>1.131</td></tr><tr><td>M3-M</td><td>0.846</td><td>0.839</td><td>0.736</td><td>0.839</td><td>0.839</td><td>0.736</td><td>0.737</td></tr><tr><td>M3-Q</td><td>0.816</td><td>0.827</td><td>0.908</td><td>0.823</td><td>0.827</td><td>0.866</td><td>0.831</td></tr><tr><td>Tourism-M</td><td>1.301</td><td>1.326</td><td>1.321</td><td>1.357</td><td>1.321</td><td>1.321</td><td>1.323</td></tr><tr><td>Tourism-Q</td><td>1.248</td><td>1.404</td><td>1.292</td><td>1.284</td><td>1.340</td><td>1.248</td><td>1.293</td></tr><tr><td>Avg. Rank</td><td>4.364</td><td>4.227</td><td>3.955</td><td>3.636</td><td>4.045</td><td>3.136</td><td>4.636</td></tr><tr><td>Avg</td><td>0.860</td><td>0.874</td><td>0.861</td><td>0.861</td><td>0.866</td><td>0.848</td><td>0.862</td></tr><tr><td>Top 2 Count</td><td>4</td><td>3</td><td>4</td><td>5</td><td>3</td><td>7</td><td>2</td></tr></table>

Table 8: MASE score of each CV approach on each dataset, and respective average rank, average MASE, and top 2 counts (seed 789).
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Methods</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>ECL</td><td>0.604</td><td>0.586</td><td>0.585</td><td>0.615</td><td>0.615</td><td>0.606</td><td>0.574</td></tr><tr><td>Exchange</td><td>0.152</td><td>0.148</td><td>0.154</td><td>0.142</td><td>0.152</td><td>0.154</td><td>0.159</td></tr><tr><td>Traffic</td><td>0.721</td><td>0.654</td><td>0.690</td><td>0.690</td><td>0.700</td><td>0.654</td><td>0.695</td></tr><tr><td>Weather</td><td>0.920</td><td>0.939</td><td>0.939</td><td>0.926</td><td>0.939</td><td>0.943</td><td>1.148</td></tr><tr><td>Hospital</td><td>0.787</td><td>0.821</td><td>0.796</td><td>0.796</td><td>0.803</td><td>0.796</td><td>0.802</td></tr><tr><td>M1-M</td><td>0.901</td><td>0.901</td><td>0.901</td><td>0.901</td><td>0.901</td><td>0.902</td><td>0.932</td></tr><tr><td>M1-Q</td><td>1.125</td><td>1.115</td><td>1.115</td><td>1.125</td><td>1.115</td><td>1.115</td><td>1.120</td></tr><tr><td>M3-M</td><td>0.751</td><td>0.773</td><td>0.779</td><td>0.779</td><td>0.779</td><td>0.841</td><td>0.737</td></tr><tr><td>M3-Q</td><td>0.823</td><td>0.823</td><td>0.858</td><td>0.817</td><td>0.858</td><td>0.906</td><td>0.896</td></tr><tr><td>Tourism-M</td><td>1.365</td><td>1.339</td><td>1.320</td><td>1.321</td><td>1.320</td><td>1.321</td><td>1.323</td></tr><tr><td>Tourism-Q</td><td>1.268</td><td>1.268</td><td>1.275</td><td>1.275</td><td>1.290</td><td>1.313</td><td>1.434</td></tr><tr><td>Avg. Rank</td><td>3.545</td><td>3.273</td><td>3.455</td><td>3.591</td><td>4.273</td><td>4.773</td><td>5.091</td></tr><tr><td>Avg</td><td>0.856</td><td>0.851</td><td>0.856</td><td>0.853</td><td>0.861</td><td>0.868</td><td>0.893</td></tr><tr><td>Top 2 Count</td><td>6</td><td>6</td><td>5</td><td>4</td><td>3</td><td>3</td><td>2</td></tr></table>

Table 9: Results of performance estimation (MAPEE), overoptimism rate, selection accuracy, and regret for each CV approach (seed 123).
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Method</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>MAPEE</td><td>0.105</td><td>0.166</td><td>0.133</td><td>0.135</td><td>0.136</td><td>0.129</td><td>0.112</td></tr><tr><td>Overoptimism</td><td>0.282</td><td>0.336</td><td>0.245</td><td>0.336</td><td>0.227</td><td>0.255</td><td>0.327</td></tr><tr><td>Accuracy</td><td>0.091</td><td>0.455</td><td>0.364</td><td>0.091</td><td>0.182</td><td>0.273</td><td>0.182</td></tr><tr><td>Regret</td><td>0.031</td><td>0.008</td><td>0.073</td><td>0.021</td><td>0.022</td><td>0.018</td><td>0.043</td></tr></table>

Table 10: Results of performance estimation (MAPEE), overoptimism rate, selection accuracy, and regret for each CV approach (seed 456).
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Method</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>MAPEE</td><td>0.134</td><td>0.118</td><td>0.120</td><td>0.157</td><td>0.121</td><td>0.133</td><td>0.103</td></tr><tr><td>Overoptimism</td><td>0.336</td><td>0.4</td><td>0.291</td><td>0.236</td><td>0.264</td><td>0.264</td><td>0.309</td></tr><tr><td>Accuracy</td><td>0.364</td><td>0.273</td><td>0.182</td><td>0.364</td><td>0.182</td><td>0.273</td><td>0.455</td></tr><tr><td>Regret</td><td>0.019</td><td>0.035</td><td>0.028</td><td>0.014</td><td>0.015</td><td>0.017</td><td>0.014</td></tr></table>

Table 11: Results of performance estimation (MAPEE), overoptimism rate, selection accuracy, and regret for each CV approach (seed 789).
<table><tr><td colspan="7">Series-wise</td><td>Time-wise</td></tr><tr><td>Method</td><td>Bootstrap</td><td>Holdout</td><td>Kf-CV</td><td>K-MC-CV</td><td>K-Bootstrap</td><td>K-Holdout</td><td>Time-CV</td></tr><tr><td>MAPEE</td><td>0.122</td><td>0.140</td><td>0.119</td><td>0.127</td><td>0.122</td><td>0.132</td><td>0.114</td></tr><tr><td>Overoptimism</td><td>0.327</td><td>0.382</td><td>0.236</td><td>0.227</td><td>0.273</td><td>0.264</td><td>0.336</td></tr><tr><td>Accuracy</td><td>0.273</td><td>0.364</td><td>0.182</td><td>0.455</td><td>0.091</td><td>0.091</td><td>0.182</td></tr><tr><td>Regret</td><td>0.019</td><td>0.016</td><td>0.032</td><td>0.012</td><td>0.020</td><td>0.026</td><td>0.043</td></tr></table>