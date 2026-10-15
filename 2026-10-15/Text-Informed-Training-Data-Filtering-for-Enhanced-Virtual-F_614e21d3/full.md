# Text-Informed Training Data Filtering for Enhanced Virtual Flow Meter Performance

Kristofer Hjelmen Nesland<sup>1,2</sup>, Bjarne Grimstad<sup>2</sup>, Antorweep Chakravorty<sup>1</sup>, and Dan Sui<sup>1</sup>

<sup>1</sup> University of Stavanger, Stavanger, Norway <sup>2</sup> Solution Seeker AS, Oslo, Norway krnes5410@uis.no

Abstract. In sectors such as energy, aquaculture and chemical processing, the efectiveness of both mechanistic and data-driven models is frequently limited by poor-quality training data, which obscures underlying physical laws and degrades predictive accuracy. Such quality issues commonly manifest as anomalies in the dataset, ranging from sensor faults to process upsets and incorrect operator inputs. While traditional rulebased filtering can remove a subset of the anomalies, it is unable to leverage contextual information stored as unstructured text, such as engineer comments and operational logs that explain under what conditions the data was collected. We propose a text-informed data filtering method that leverages large language models (LLMs) to integrate unstructured operational data with tabular sensor data. The method consists of two steps: (1) contextual anomaly detection by reasoning over both data modalities, and (2) generation of a binary mask that is applied to filter the training dataset for model development. To investigate the potential benefit, a steady-state soft-sensor regression model, namely a virtual flow meter that estimates well flow rates, is used as the downstream evaluation target. Experiments on a synthetic oil and gas dataset show improved model performance compared to traditional rule-based filtering. Ablation studies show that textual operator comments can provide filtering-relevant signals beyond sensor data alone, with the multimodal approach reducing downstream MAPE from 7.1 % to 5.4 % relative to sensor-only filtering. This work demonstrates that integrating unstructured operational knowledge with sensor data through LLMs can improve data quality for industrial machine learning applications, ofering a scalable alternative to manual data curation.

Keywords: Multimodal data fusion · Industrial machine learning · Datacentric AI · Text-informed data filtering · Virtual flow meter.

## 1 Introduction

The attention of researchers and practitioners has gradually shifted from advancing model design to enhancing the quality and quantity of the data [19]. This data-centric paradigm is particularly relevant in industrial settings, where an increasing amount of sensor data is captured but data quality issues and anomalies are significant due to harsh operational environments [6]. If not addressed, the data quality issues will continue to afect downstream modelling tasks that depend on the data. However, manual quality control of the data is infeasible in many cases due to the large volume of data and the challenge of synthesising information across disparate data streams and modalities [11].

![](images/c5accd39ebcbce4b712882dc7d0c83bec2c41ce04c43895ce7c922e28bbfadb4.jpg)  
Fig. 1. Traditional flow of information in industrial data curation. The operator synthesises background domain knowledge, operational context, and sensor readings into unstructured comments. These comments, together with the raw sensor data, then inform a manual curation step that determines which data points enter model training.

Classical anomaly detection methods, as surveyed by Han et al. [8] and consolidated in the PyOD toolbox [20], are unimodal: they operate on sensor data alone. Most existing benchmarks follow the same pattern, with little coverage of settings where relevant quality signals are distributed across multiple modalities [19]. Yet, to successfully identify anomalous data, sensor readings need to be seen relative to contextual knowledge of the asset. This task involves sifting through unstructured textual data like maintenance logs, shift handovers and technician notes. Consequently, subject matter experts have taken on a role as the bridge between the two data modalities and spent considerable time integrating information across dashboards, PDFs and more (Fig. 1).

Advances in artificial intelligence (AI) open up new possibilities for bridging the gap between sensor data and unstructured textual data. Today, LLMs can process natural language alongside structured numerical tables within a single context window, making it feasible to reason jointly over operator comments and sensor measurements without human involvement [9].

## 1.1 Problem Definition

Following Chandola et al. [4], an anomaly is a pattern in data that does not conform to a well-defined notion of normal behaviour. While “anomaly” and “outlier” are often used interchangeably in the literature, this paper uses “anomaly” throughout. The term is preferred because the deviations considered here stem from identifiable physical events or instrument failures, rather than purely statistical deviation from a distribution. Chandola et al. [4] further distinguish between point anomalies, which are data instances that deviate from the rest of the data in an absolute statistical sense, and contextual anomalies, which are only anomalous given additional context. The anomalies encountered in industrial production data are often contextual in nature and a sensor reading that falls within historical statistical ranges may still represent unreliable data if the well was undergoing an instrument fault at the time of measurement. Detecting such anomalies purely from statistical properties of the sensor readings is therefore insuficient.

The problem addressed in this paper is: given a dataset of paired sensor measurements and operator comments for a set of assets, determine which data points should be classified as anomalies and thereby excluded before model training. We formulate the problem as assigning a binary weight per data point, where one means “include” and zero means “exclude”. A continuous weighting scheme, where anomalous points are down-weighted rather than excluded, was considered but would require determining a suitable degree of down-weighting, which is non-trivial and can depend on the choice of downstream model.

Open industrial datasets such as the Petrobras 3W dataset for well failures [17], the Volve field repository [5], and the ManyWells simulator [7] exist, but none pair sensor measurements with unstructured operator comments and ground-truth anomaly labels. This motivates a synthetic benchmark approach.

## 1.2 Contributions

## Our contributions are:

1. A text-informed data filtering method that uses an LLM to jointly reason over tabular sensor measurements and unstructured operator comments, producing a binary data mask.

2. Experimental results on a synthetic oil and gas dataset showing that the multimodal method outperforms classical sensor-based filters on downstream soft-sensor performance, and that unstructured operator comments carry filtering-relevant signals invisible to sensor-based methods alone.

## 2 Related Work

## 2.1 Language Models for Tabular Data and Multimodal Data Fusion

Brown et al. [3] demonstrate that large language models can perform few-shot learning across a wide range of tasks, and subsequent work has extended this to tabular data classification [9] and tabular anomaly detection [15]. Wu et al. [18] further demonstrate multimodal in-context anomaly detection, combining structured and unstructured inputs within a single LLM context. However, none of these methods are applied to the problem of data quality filtering for industrial model training, where the goal is to improve downstream predictive performance rather than to classify anomalies as an end in itself.

## 2.2 Soft-Sensors and Virtual Flow Metering

Soft-sensor models use available process measurements to estimate variables that are dificult to measure directly, with regression-based methods surveyed in Souza et al. [13]. In oil and gas, neural network-based virtual flow meters have demonstrated strong performance across diverse well configurations [12]. The predictive quality of such models is sensitive to the training data they are fitted on, yet the efect of systematic data filtering on downstream soft-sensor performance has received little attention. This paper uses a simple soft-sensor model as the downstream evaluation target, making the impact of each filtering strategy directly measurable.

## 3 Benchmark Dataset

## 3.1 Dataset Implementation

We construct a synthetic benchmark, built on ManyWells [7], a steady-state multiphase flow simulator for oil and gas wells with varying reservoir properties, wellbore geometries and fluid compositions. The simulator uses a one-dimensional drift-flux model and has been validated against real production data from more than 300 wells. A well test is a separator flow test in which the well is temporarily routed through a dedicated test separator to measure the volumetric flow rates of oil, gas and water, with mean values over a stable test interval reported as a single observation. Each ManyWells simulation thus represents one such well-test data point, described by measurements of pressures, temperatures, choke position, phase flow rates and more. Reservoir pressure and flow composition are assumed constant per well, so the benchmark does not model temporal efects such as reservoir depletion.

Data Generation. Let J denote the set of all wells in the benchmark. For each well $j \in J ,$ , the simulator produces $N _ { j }$ data points by sampling operating conditions (choke opening, gas lift rate) from the well’s feasible domain and solving the steady-state flow equations. Each data point is described by a sensor measurement vector $\mathbf { x } _ { i j } \in \mathbb { R } ^ { d }$ and an optional textual comment $t _ { i j } \in { \mathcal { T } } \cup \{ \emptyset \}$ 2 where $i = 1 , \ldots , N _ { j }$ indexes the data points within well $j , \tau$ is the set of all nonempty strings and ∅ denotes the absence of a comment. A subset of data points is then designated as anomalous through controlled perturbations of either the simulation parameters or the resulting sensor readings. The ground-truth weight is $w _ { i j } = 0$ whenever any perturbation is applied and $w _ { i j } = 1$ otherwise.

The wells are partitioned into three roles:

– Reference wells $( J _ { \mathrm { r e f } } \subset J )$ with disclosed ground-truth weights, used as incontext examples for the LLM.

Table 1. Anomaly groups.
<table><tr><td>Category</td><td>Perturbation type</td><td>Physical mechanism</td></tr><tr><td>Scaling</td><td>Well parameter  $( f _ { D } , D )$ </td><td>Mineral deposits restricting flow</td></tr><tr><td>Hydrate formation Well parameter</td><td> $( f _ { D } , D )$ </td><td>Ice-like plugs in the wellbore</td></tr><tr><td>Liquid loading</td><td>Well parameter (α)</td><td>Inability to lift liquids</td></tr><tr><td>Emulsion</td><td>Well parameter  $\left( f _ { D } \right)$ </td><td>Oil-water emulsions increasing friction</td></tr><tr><td>Wax deposition</td><td>Well parameter  $\left( f _ { D } \right)$ </td><td>Wax build-up increasing friction</td></tr><tr><td>Data quality</td><td>Simulated parameter</td><td>Sensor faults and communication failures</td></tr></table>

– Training wells $( J _ { \mathrm { d e v } } \subset J )$ with undisclosed weights, on which all filtering methods are applied.

Evaluation splits: each training well also has a held-out evaluation set generated under nominal conditions (no anomalies), used solely for measuring downstream model performance.

Textual Comments. Each injected anomaly carries a textual comment $t _ { i j }$ inspired by real-world operator logs from Norwegian ofshore operations. Comments are written in a mixture of Norwegian and English, reflecting actual communication practices, and vary in specificity from precise diagnoses (e.g. “Injecting MEG to dissolve hydrate plug after unplanned shutdown”) to vague observations (e.g. “Potential for scaling”). Some anomalous data points carry an empty comment $( t _ { i j } = \emptyset )$ , mimicking the common case where anomalies go unreported. Conversely, some nominal data points carry positive or neutral comments (e.g. “Test went according to plan”), as occurs in practice when operators note routine events.

## 3.2 Dataset Details

Sensor Features. Each sensor measurement vector $x _ { i j }$ contains d = 4 features: Bottomhole pressure $( P _ { B H } , \ \mathrm { b a r } )$ , Wellhead pressure $( P _ { W H }$ , bar), Volumetric liquid flow rate $( Q , { \mathrm { S m } } ^ { 3 } / { \mathrm { h } } )$ and Wellbore length $( L , { \textrm { m } } )$ . This is only a small subset of the measurements provided by ManyWells. For a thorough analysis of the ranges of the features, we refer to the ManyWells paper [7].

Anomaly Categories and Injection Mechanisms. Anomalies are grouped into six categories that reflect common operational issues in oil and gas production (Table 1). The categories are inspired by Theyab [14].

The well parameters perturbed are the wellbore Darcy friction factor $f _ { D }$ , the wellbore internal diameter D, and the phase composition vector α giving the gas, oil and water fractions. Two perturbation mechanisms are used. Well parameter perturbations modify physical parameters before simulation, producing data points that are internally consistent but reflect abnormal well conditions such as flow restrictions from scaling, wax or emulsions, physical obstructions from hydrate plugs, or liquid loading. Simulated parameter perturbations modify sensor readings after simulation to model instrument faults such as transmitter drift or flatlined meters.

Dataset Sizes. We utilise 13 well configurations spanning a range of fluid compositions, splitting them into three reference wells $( J _ { \mathrm { r e f } } = \{ 5 1 , 7 7 6 , 1 2 3 4 \} )$ used for in-context examples, and ten training wells

$( J _ { \mathrm { d e v } } = \{ 2 8 9 , 4 0 6 , 6 4 1 , 7 1 5 , 7 4 5 , 1 1 0 0 , 1 5 8 0 , 1 7 7 3 , 1 8 2 0 , 1 8 4 6 \} )$ . The indexes refer to existing configurations in the ManyWells dataset. Training set sizes vary from 10 to 60 data points per well, with 50 evaluation points per well. In total, the benchmark contains 80 reference, 311 training and 500 evaluation data points. Each training well has, on average, 10 injected anomalies placed at random positions, alongside non-anomalous data points. The anomaly rate across wells varies from roughly 30 % to 60 % of the training data.

The distribution of the textual data is designed to be broadly representative of the Norwegian ofshore asset that motivated the benchmark design, though comment coverage and specificity vary across assets depending on operational culture and documentation practices. Of anomalous points, 55.4 % carry a comment describing the fault condition and the remaining 44.6 % go unreported, reflecting the case where anomalies are not logged. Of nominal points, 92.9 % carry no comment, with the remaining 7.1 % receiving a routine note confirming normal operation. For simplicity, anomalous points never receive routine comments, and nominal points never receive anomaly-indicating ones.

## 4 Method

## 4.1 Problem Formulation

For each well $j \in J .$ , the dataset $\mathcal { D } _ { j } = \{ ( \mathbf { x } _ { i j } , t _ { i j } ) \} _ { i = 1 } ^ { N _ { j } }$ collects $N _ { j }$ data points as defined in the benchmark description above.

The goal is to determine a binary weight vector $\mathbf { w } _ { j } \ = \ ( w _ { 1 j } , \ldots , w _ { N _ { j j } } ) \ \in$ $\{ 0 , 1 \} ^ { N _ { j } }$ for each well, where $w _ { i j } ~ = ~ 1$ indicates that data point i should be retained for model training and $w _ { i j } = 0$ indicates that it should be excluded as anomalous. In total, a number of $\begin{array} { r } { N = \sum _ { j \in J _ { \mathrm { d e v } } } N _ { j } } \end{array}$ weights need to be determined. A weighting function is a function $\mathcal { F }$ that maps the dataset to a weight vector:

$$
\mathcal { F } : \mathcal { D } \mapsto \mathbf { w }\tag{1}
$$

The weight vector is subsequently used in a weighted regression objective when fitting a downstream model. A good filtering method should exclude data points that are anomalous (due to physical faults or sensor errors) while retaining data that reflects normal well behaviour, thereby improving the predictive accuracy of the fitted model on the nominal evaluation data.

## 4.2 LLM-Based Multimodal Anomaly Detection

The proposed anomaly detection method uses an LLM to integrate both sensor measurements $\mathbf { x } _ { i j }$ and textual comments $t _ { i j }$ into a single anomaly-classification decision per data point. This multimodal approach is contrasted with traditional statistical baselines that operate on sensor measurements alone.

The LLM is given a structured prompt containing: (1) a task description specifying that the goal is to assign binary weights to each sample, (2) a feature dictionary explaining the physical meaning and units of each variable, (3) $| J _ { \mathrm { r e f } } |$ reference datasets with disclosed weights as in-context examples and (4) the target training dataset with unknown weights. Each dataset is presented as a JSON array of records, with each record containing an input object (sensor values and comment) and an output object (the weight to be predicted). The LLM returns a structured JSON response containing one decision per data point, each consisting of a binary weight and a brief natural-language reasoning.

The system instructions include a feature dictionary (component 2) because the JSON field names are domain-specific abbreviations whose physical meaning and units are not self-evident. Providing explicit column semantics is standard practice when serialising tabular data for LLMs, enabling the model to draw on pre-trained domain knowledge [9,15]. The reference datasets in component (3) follow the few-shot in-context learning paradigm of Brown et al. [3].

Formally, let S denote the system instructions (task description and feature dictionary), and let ${ \mathcal { R } } = \{ { \mathcal { D } } _ { j } \} _ { j \in J _ { \mathrm { r e f } } }$ denote the set of reference datasets with known weights. The LLM filtering function is applied independently to each training well $j \in J _ { \mathrm { d e v } }$ :

$$
\mathcal { F } _ { \mathrm { L L M } } : ( { \cal S } , \mathcal { R } , \mathcal { D } _ { j } ) \mapsto { \bf w } _ { j } , \quad j \in J _ { \mathrm { d e v } }\tag{2}
$$

where $\mathcal { D } _ { j }$ is the target training dataset for well j (with weights unknown), as defined above, and the LLM performs in-context learning from R without any gradient-based training. A single sample $( { \bf x } _ { i j } , t _ { i j } , w _ { i j } )$ from one of the reference wells is shown in Fig. 2.

```json
{
"INPUT": {
"WELL_IDX": 51, "PBH": 261.8, "PWH": 71.9, "QLIQ": 89.5, "L": 3390.6,
"COMMENT": "WHP transmitter drifting. Reading static despite flow
variations."
},
"OUTPUT": { "WEIGHT": 0 }
}
```  
Fig. 2. A sample in the utilised JSON format.

## 4.3 Baseline Methods

The baselines operate on liquid flow rate (Q) and pressure diferential $( P _ { \mathrm { B H } , i } -$ $P _ { \mathrm { W H } , i } )$ . Wellbore length (L) is not included as it is a constant per well and contributes no within-well discriminating signal. This is in contrast with the LLM method where the model can potentially gain some insights from comparing the wellbore length with the reference wells.

Two classical anomaly-detection methods are applied:

1. Tukey’s fences [16]: compute the interquartile range $( I Q R = Q _ { 3 } - Q _ { 1 } )$ for each axis and flag points outside $[ Q _ { 1 } - 1 . 5 \cdot \mathrm { I Q R } , ~ Q _ { 3 } + 1 . 5 \cdot \mathrm { I Q R } ]$ on either axis as anomalous $( w _ { i j } = 0 )$ . Here, $Q _ { 1 }$ is the first quartile and $Q _ { 3 }$ is the third quartile.

2. Robust Mahalanobis [10]: fit a Minimum Covariance Determinant (MCD) estimator, with support fraction of 0.85, to the data and compute the squared Mahalanobis distance $d _ { i j } ^ { 2 }$ for each point. Points exceeding the $\chi _ { p } ^ { 2 } ( 0 . 9 5 )$ quantile are flagged $( w _ { i j } = 0 )$ , where $p$ is the number of features.

Two additional reference methods are included:

1. Uniform: $w _ { i j } = 1$ for all $i \in \{ 1 , \dots , N _ { j } \} , j \in J _ { \mathrm { d e v } }$ (no filtering), establishing a lower bound on performance.

2. Oracle: $w _ { i j } \ = \ w _ { i j } ^ { * } .$ , where $\boldsymbol { w _ { i j } ^ { * } }$ denotes the ground-truth weight from the simulator, establishing an upper bound.

## 4.4 Ablation and Prompt Augmentation

To study the contribution of each modality, two more input views are defined. The text-only view strips all sensor measurements, providing only the engineer comment for each row. The sensor-only view removes the comment field.

To investigate whether explicitly presenting statistical context improves LLM performance, we study two augmented variants that provide pre-computed anomaly statistics in the prompt. The first one augments the prompt with a binary flag for each sample, calculated with the Tukey algorithm. The second one injects the Mahalanobis distance for each data point into the prompt.

## 5 Experimental Setup

## 5.1 Research Questions and Metrics

Our research questions are:

1. Multimodal LLM vs. classical: Does multimodal LLM filtering, combining sensor data and operator comments, outperform classical rule-based methods (Tukey, Mahalanobis) on downstream soft-sensor performance?

2. Value of textual information in LLM filtering: Do operator comments provide filtering-relevant signals beyond what sensor data alone captures when used as LLM input?

Table 2. Role of each method per research question. B = baseline, • = tested, abl. = ablation, — = reference bound.
<table><tr><td>Method RQ1 RQ2 RQ3</td></tr><tr><td>Uniform (no filter)</td></tr><tr><td>Tukey B</td></tr><tr><td>Mahalanobis B</td></tr><tr><td>LLM (multimodal) B LLM (text only) abl.</td></tr><tr><td>LLM (sensor only) B</td></tr><tr><td> $\mathrm { L L M } + \mathrm { T u k e y }$ </td></tr><tr><td> $\mathrm { L L M + M a h a l a n o b i s }$  Oracle (ground truth)</td></tr></table>

3. Classical anomaly scores in the prompt: Does adding pre-computed classical anomaly scores to the multimodal LLM prompt outperform multimodalonly LLM filtering?

All research questions are evaluated on downstream soft-sensor quality: mean absolute percentage error (MAPE), mean squared error (MSE), and fraction of predictions within 10 % of the true value (W10). “Outperforms” means strictly better on two of these three reported metrics. Table 2 summarises the role of each method for each research question.

MAPE, MSE, and W10 are calculated as:

$$
\begin{array} { r l } { ~ } & { { } - ~ \mathbf { M A P E } \colon \frac { 1 0 0 } { n } \sum _ { i = 1 } ^ { n } \left| \frac { Q _ { i } - \hat { Q } _ { i } } { Q _ { i } } \right| . } \end{array}
$$

$$
\begin{array} { r } { - \ \mathbf { M } \mathbf { S } \mathbf { E } \colon \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( Q _ { i } - \hat { Q } _ { i } ) ^ { 2 } . } \end{array}
$$

– W10: $\begin{array} { r } { \frac { 1 0 0 } { n } \sum _ { i = 1 } ^ { n } \mathbf { 1 } \bigg [ \Big | \frac { Q _ { i } - \hat { Q } _ { i } } { Q _ { i } } \Big | \leq 0 . 1 0 \Big ] } \end{array}$ , where 1[·] equals 1 if the condition holds and 0 otherwise.

Here, n is the number of evaluation points.

## 5.2 LLM Inference

The LLM filtering function $\mathcal { F } _ { \mathrm { L L M } }$ is instantiated with Anthropic’s Claude Sonnet 4.6 [1], using structured output via JSON schema enforcement. This ensures that the model returns exactly $N _ { j }$ decisions per well in a parseable format. Each decision includes a binary weight and a one-sentence reasoning string. Because LLM inference is stochastic, each LLM method is run R = 3 independent times per well. Downstream metrics are computed by pooling the prediction errors from all runs. The repeated runs allow for a direct analysis of filtering consistency across runs, presented in the stability analysis in Section 6.2.

## 5.3 Soft-Sensor Model

To evaluate the downstream impact of each filtering method, a simple mechanistic soft-sensor model is fitted per well, $j .$ The model predicts the liquid (oil and water) flow rate Q from the pressure diferential $\varDelta P _ { i } = P _ { \mathrm { B H } , i } - P _ { \mathrm { W H } , i }$ and the vertical wellbore length $L _ { j }$ . Net driving pressure is the diferential minus hydrostatic head, truncated at zero:

$$
\varDelta P _ { \mathrm { n e t } , i } = \mathrm { m a x } \biggl ( \varDelta P _ { i } - \frac { \rho g L _ { j } } { C _ { F } } , 0 \biggr )\tag{3}
$$

$$
\hat { Q } _ { i } = c \sqrt { \varDelta P _ { \mathrm { n e t } , i } }\tag{4}
$$

Here $\varDelta P _ { i }$ is expressed in bar, $c$ is a flow coeficient that lumps geometry and friction into a single constant, and $\rho$ is an efective fluid density $\mathrm { ( k g / m ^ { 3 } ) }$ that absorbs multiphase efects in a single scalar. $g$ is standard gravitational acceleration $\left( \mathrm { m } / \mathrm { s } ^ { 2 } \right)$ , and $C _ { F } = 1 0 ^ { 5 }$ converts hydrostatic head from pascals to bar (Pa/bar). With $\varDelta P _ { \mathrm { n e t } , i }$ as net driving pressure, a steady one-dimensional balance in which frictional losses scale with the square of a representative velocity implies a quadratic relation between flow rate and net pressure drop, $\varDelta P _ { \mathrm { n e t } } \propto Q ^ { 2 }$ , and hence $Q \propto \sqrt { \varDelta P _ { \mathrm { n e t } } }$ . This is the same structure as a Darcy–Weisbach-type loss in pipe flow [2]. The parameters c and $\rho$ are fitted per well by minimising the weighted mean squared error, where we drop the $j$ subscript since we optimise per well:

$$
c ^ { * } , \rho ^ { * } = \arg \operatorname* { m i n } _ { c , \rho } \ \frac { 1 } { \sum _ { i = 1 } ^ { N _ { j } } w _ { i } } \sum _ { i = 1 } ^ { N _ { j } } w _ { i } \left( Q _ { i } - \hat { Q } _ { i } \right) ^ { 2 }\tag{5}
$$

The efective density $\rho$ is found via grid search, while $c$ is solved in closed form for each candidate $\rho .$ The model is intentionally simple: the focus of this study is not the model architecture but the efect of diferent filtering strategies on downstream predictive performance. A more expressive model such as the one proposed by Sandnes et al. [12] would draw on a higher-dimensional feature space, making it dificult to visualise the relationship between training data and model fit in two dimensions. By restricting the model to a single varying predictor per well $( \varDelta P )$ , the relationship between the retained training data and the fitted model can be visualised in two dimensions, keeping the results directly human-interpretable, as illustrated in Fig. 3.

![](images/a05a409d35d039e14fb928a32fe426fb8911096be6f1913fc411a164453407f9.jpg)  
Fig. 3. Example of a dataset for a single well and the model fitted to the nominal data points.

## 6 Results and Discussion

## 6.1 Downstream Soft-Sensor Performance

All methods share the same model architecture and evaluation set, the only diference is which samples are classified as anomaly and excluded from training. “Uniform” retains every sample, including injected anomalies. “Oracle” correctly classifies all anomalies. The evaluation set contains no anomalies, ensuring that diferences in performance reflect the filtering strategy alone.

Table 3 reports downstream metrics on the held-out evaluation sets: MSE, MAPE and W10. Predictions are pooled across wells and, for LLM variants, across the R = 3 independent classification runs. The 95 % confidence intervals are computed by a percentile bootstrap with 10,000 resamples over the pooled evaluation points, taking the 2.5th and 97.5th percentiles as bounds.

The role of each method per research question is summarised in Table 2. Fig. 4 presents the results as a cumulative performance plot. These are informationdense plots that, given a performance threshold (x-axis), show the percentage of the total evaluation samples (y-axis) where the model has a better or equal performance than the threshold. We address each research question in turn.

Without filtering, Uniform (no filter) performs worst on all reported metrics (MSE 148, MAPE 12.5 %, W10 51 %). Among the classical filters, the best method depends on the metric: Mahalanobis achieves the lowest MAPE (9.4 %), while Tukey achieves the lowest MSE (112 $\mathrm { ( S m ^ { 3 } / h ) ^ { 2 } } )$ ), both reach W10 of about

Table 3. MSE, MAPE and W10, pooled across wells and classification runs (95 % confidence intervals).
<table><tr><td>Method</td><td>MSE  $\left( ( \mathrm { S m } ^ { 3 } / \mathrm { h } ) ^ { 2 } \right)$ </td><td>MAPE (%) W10 (%)</td></tr><tr><td>Uniform (no filter)</td><td>148 [123, 176] 12.5 [11.5, 13.6] 51 [46, 55]</td><td></td></tr><tr><td>Tukey</td><td>112 [86, 144]</td><td>10.0 [8.9, 11.2] 66 [62, 70]</td></tr><tr><td>Mahalanobis</td><td>123 [98, 151]</td><td>9.4 [8.3, 10.6] 66 [62, 71]</td></tr><tr><td>LLM (multimodal)</td><td>53 [44, 64]</td><td>5.4 [4.9, 5.9] 87 [85, 89]</td></tr><tr><td>LLM (text only)</td><td>86 [71, 103]</td><td>8.0 [7.4, 8.7] 78 [76, 81]</td></tr><tr><td>LLM (sensor only)</td><td>88 [74, 103]</td><td>7.1 [6.5, 7.8] 84 [82, 86]</td></tr><tr><td>LLM + Tukey</td><td>64 [49, 82]</td><td>5.7 [5.1, 6.4] 88 [87, 90]</td></tr><tr><td>LLM + Mahalanobis</td><td>47 [36, 59]</td><td>5.1 [4.6, 5.6] 89 [87, 91]</td></tr><tr><td>Oracle (ground truth)</td><td>36 [26, 48]</td><td>4.1 [3.5, 4.8] 90 [87, 92]</td></tr></table>

66 %. LLM (multimodal) improves on the best classical result in each case (MSE 53, MAPE 5.4 %, W10 87 %). The confidence intervals for multimodal LLM filtering do not overlap those of Mahalanobis or Tukey on any of the metrics (Table 3), and Fig. 4 shows the same ordering across the full error distribution. RQ1: multimodal LLM filtering outperforms both Tukey and Mahalanobis on MSE, MAPE, and W10 under the criterion defined in Section 5.

The key comparison for RQ2 is LLM (multimodal) against LLM (sensor only), which isolates the contribution of the textual modality while holding sensor input constant. Adding operator comments to the sensor-based prompt improves performance on all three metrics: MAPE drops from 7.1 % to 5.4 % (1.7 percentage points), MSE drops from 88 to 53, and W10 rises from 84 % to 87 %. The confidence intervals on MAPE do not overlap (sensor only: [6.5, 7.8] vs. multimodal: [4.9, 5.9]), indicating a statistically meaningful diference. The same goes for MSE and W10. RQ2: textual operator comments carry filteringrelevant information that is not captured by sensor data alone, and the LLM can leverage this when both modalities are presented jointly.

Augmenting the multimodal LLM prompt with robust Mahalanobis distance improves downstream performance relative to LLM (multimodal) alone. LLM + Mahalanobis performs best among all non-Oracle methods (MSE 47, MAPE 5.1 %, W10 89 %), improving on multimodal-only by 6 MSE units, 0.3 percentage points of MAPE, and 2 percentage points of W10. The augmentation of the prompt with pre-computed flags from Tukey’s fences does not provide the same performance jump and has a negative efect on MSE and MAPE, but a positive efect on W10. RQ3: adding classical anomaly scores to the prompt can improve downstream metrics relative to multimodal LLM filtering alone, but the efect is not universally positive and depends on the selected method.

![](images/711df38973e21ae83571797e41827549ca4073ce9640d6253e719b8fb880d527.jpg)  
Fig. 4. Results presented as a cumulative performance plot.

## 6.2 Cross-Method Agreement and Consistency of LLM Filtering

Beyond downstream soft-sensor metrics, we examine whether the LLM-based filtering methods agree on which data points to classify as anomaly, and how consistently each method reproduces its own decisions across independent runs. To quantify this, we compare the binary weight vectors produced by each LLM method across all training wells $\left( J _ { \mathrm { d e v } } \right)$ . Because LLM-based methods are stochastic, each LLM method ${ \mathcal { F } } _ { a }$ is executed $R = 3$ independent times, yielding weight vectors $\mathbf { w } _ { i } ^ { ( a , r ) } = \mathcal { F } _ { a } ^ { ( r ) } ( \mathcal { D } _ { j } )$ for $r \in \{ 1 , 2 , 3 \}$

We define the pairwise percentage of matching decisions between run $r _ { a }$ of method a and run $r _ { b }$ of method b as

$$
\mathrm { M a t c h } ( a , r _ { a } , b , r _ { b } ) = \frac { 1 0 0 } { \sum _ { j \in J _ { \mathrm { d e v } } } N _ { j } } \sum _ { j \in J _ { \mathrm { d e v } } } \left| \left\{ i \in \{ 1 , \dots , N _ { j } \} : w _ { i j } ^ { ( a , r _ { a } ) } = w _ { i j } ^ { ( b , r _ { b } ) } \right\} \right| .\tag{6}
$$

For each pair of methods $( a , b )$ , we report Match $( a , b )$ , the average over all valid run combinations:

$$
\mathrm { M a t c h } ( a , b ) = \frac { 1 } { | P _ { a b } | } \sum _ { ( r _ { a } , r _ { b } ) \in P _ { a b } } \mathrm { M a t c h } ( a , r _ { a } , b , r _ { b } ) ,\tag{7}
$$

where $P _ { a b }$ is the set of valid run pairs: $\{ ( r _ { a } , r _ { b } ) : r _ { a } \neq r _ { b } \}$ when $a = b ,$ and $\{ 1 , \ldots , R _ { a } \} \times \{ 1 , \ldots , R _ { b } \}$ when $a \neq b$

Table 4. Run-to-run consistency: Match(a, a) across all training wells (percent matching decisions).
<table><tr><td>Method Match (%)</td></tr><tr><td>LLM (multimodal)</td></tr><tr><td>96.6 LLM (text only) 100.0</td></tr><tr><td>LLM (sensor only) 92.7</td></tr><tr><td> $\mathrm { L L M } + \mathrm { T u k e y }$  96.4</td></tr><tr><td> $\mathrm { L L M + M a h a l a n o b i s }$  97.6</td></tr></table>

We first examine run-to-run consistency $( a = b )$ presented in Table 4. LLM (text only) is perfectly stable across all three runs (100 % match), whereas LLM (sensor only) agrees with itself on 92.7 % of decisions. This gap suggests that sensor-based prompts leave more room for stochastic variation, likely because numeric patterns near the exclusion threshold are harder to resolve consistently than comment-based cues. Adding pre-computed Mahalanobis distances to the prompt increases consistency to 97.6 %, up from 96.6 % for LLM (multimodal), indicating that classical anomaly scores can stabilise LLM filtering decisions. However, LLM + Tukey is marginally lower at 96.4 %.

For the cross-method agreement (a ̸= b), the largest disagreement appears between the two unimodal baselines: LLM (sensor only) and LLM (text only) match on only 74.1 % of decisions. This confirms that the two modalities identify largely diferent subsets of anomalous points, supporting the view that operator comments encode filtering-relevant context not visible in sensor readings alone (RQ2). LLM (multimodal) agrees with neither unimodal method fully (84.9 % vs. sensor only, 86.5 % vs. text only), consistent with joint reasoning over both inputs rather than deferring to a single modality.

## 6.3 Explainability of LLM Filtering

Unlike Tukey’s fences or Mahalanobis distance, which apply transparent arithmetic rules, the LLM methods are fundamentally black-box classifiers. To obtain some insight into the decisions, each LLM call is prompted to produce a one-sentence natural-language justification alongside every binary weight assignment. We illustrate this in Fig. 5 with data points from well 745 processed by LLM (multimodal). The examples show that the model weighs text against sensor readings rather than just pattern-matching on either.

The model’s reasoning for the numbered data points is:

– 11: “Comment indicates ‘Masse emulsjoner’ (lots of emulsions), which as seen in reference datasets leads to unreliable measurements.”

– 22: “Comment indicates ‘Emulsjoner i flowback’ (emulsions in flowback), which compromises measurement reliability per reference datasets.”

– 23: “Comment mentions only minor easily-soluble emulsions (‘Kun litt lettløselige emulsjoner’), not severe enough to exclude.”

![](images/c2b45b8e585475763005d97ef212584ef5675050924b0de1385d90ba18a85a95.jpg)  
Fig. 5. Data points of the well with index 745 processed by the LLM (multimodal) method. Data points of particular interest are numbered.

– 27: “PBH (327.9) is slightly lower than typical but QLIQ is consistent; no anomalous comment.”

– 34: “Normal data point with values consistent with the main cluster.”

– 44: “PWH of 7.8 is slightly low but PBH and QLIQ are consistent with the cluster; acceptable data point.”

Points 11 and 22 together show how operator comments can drive classification independent of proximity to the cluster. Point 11 lies far from the cluster and contains a comment indicating severe emulsion, so it is excluded. As seen in Fig. 5, point 22 is considerably closer to the cluster but is excluded nonetheless, because its comment flags emulsions in flowback, a condition the model treats as compromising based on the reference data. The contrasting cases are points 27 and 44. Both points sit at a significant distance from the cluster (Fig. 5), and the model acknowledges that the pressure values are somewhat of, but with no anomalous comment and flow rates within the expected range, neither is classified as an anomaly. Point 34 has values that are consistent with the main cluster and no comment. Like point 34, point 23 is also close to the main cluster and is not classified as an anomaly despite its comment hinting at emulsion issues. In sum, the six points of Fig. 5 show that the model integrates the text and sensor modalities non-trivially. The outcome is not determined by text alone. For instance, point 23 is accepted despite an emulsion hint. Similarly, sensor distance is not the sole deciding factor, as point 11 is classified as an anomaly while point 44 is not, despite both having similar distances from the cluster.

## 7 Conclusion

This paper addressed data quality issues in industrial datasets by proposing a multimodal LLM method that jointly reasons over sensor data and unstructured operator comments to filter anomalous training samples. On the downstream regression task, the multimodal method outperforms the sensor-only method by 1.7 percentage points and it is shown that the textual data adds signal beyond the LLM only assessing the sensor data. More broadly, this points to operator comments as an underused data-centric resource: a modality already recorded in industrial workflows that can be folded into model development without the manual curation it currently demands.

The results reported in this paper are based on a synthetic benchmark of 13 wells, and performance on real-world data may deviate. The informativeness of comments is a design choice calibrated to a real asset, and the magnitude of the text benefit depends on it. Further, the study of the model’s “reasoning” traces provides only a limited insight into the behaviour of inherently opaque LLM methods. Beyond interpretability, LLM-based approaches introduce practical overhead relative to classical rule-based algorithms: inference latency is higher, behaviour is non-deterministic, and reliance on proprietary models raises concerns about data privacy and service availability. These factors may limit applicability in latency-sensitive or security-constrained operational settings. All LLM-based experiments use a single model (Claude Sonnet 4.6). Filtering decisions and downstream metrics may difer for other models.

## 7.1 Future Work

Real-world data would help assess how well the results generalise and how sensitive they are to reference dataset size and composition. For industrial applicability, treating time-dependent phenomena remains an open challenge, particularly in determining how long textual information stays valid for anomaly assessment. Regarding explainability, more work is needed to understand the relative importance of sensor data and text.

Reproducibility The code for generating the benchmark dataset and reproducing the experiments is publicly available at https://github.com/krNesland/nesland2026-text-informed-filtering.

Generative AI Use Generative AI tools were utilised in the work leading up to this paper. Google Gemini was partly used to suggest relevant literature. Anthropic Sonnet 4.6 was used to develop some of the mathematical notation and to increase the readability of the paper. The authors have reviewed and edited all AI-generated content and take full responsibility for the final content.

Disclosure of Interests. K. H. Nesland and B. Grimstad are employed by Solution Seeker AS and A. Chakravorty and D. Sui are at the University of Stavanger. The underlying PhD project is industrially funded (Research Council of Norway, Solution

Seeker AS and industry partner). Solution Seeker develops production-optimisation software related to the methods discussed and ManyWells, used for the synthetic benchmark, originates from that research environment. No other competing financial interests are declared.

## References

1. Anthropic: Introducing Sonnet 4.6. https://www.anthropic.com/news/ claude-sonnet-4-6 (2026)

2. Brown, G.O.: The history of the Darcy-Weisbach equation for pipe flow resistance. In: Environmental and Water Resources History. pp. 34–43 (2002). https://doi. org/10.1061/40650(2003)4

3. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert-Voss, A., Krueger, G., Henighan, T., Child, R., Ramesh, A., Ziegler, D., Wu, J., Winter, C., et al.: Language models are few-shot learners. In: Advances in Neural Information Processing Systems. vol. 33, pp. 1877–1901 (2020)

4. Chandola, V., Banerjee, A., Kumar, V.: Anomaly detection: A survey. ACM Computing Surveys 41(3), 15:1–15:58 (2009). https://doi.org/10.1145/1541880. 1541882

5. Equinor: Disclosing all Volve data. https://www.equinor.com/news/archive/ 14jun2018-disclosing-volve-data (2018)

6. Gaddam, A., Wilkin, T., Angelova, M., Gaddam, J.: Detecting sensor faults, anomalies and outliers in the Internet of Things: A survey on the challenges and solutions. Electronics 9(3), 511 (2020). https://doi.org/10.3390/electronics9030511

7. Grimstad, B., Lundby, E., Andersson, H.: ManyWells: Simulation of multiphase flow in thousands of wells. Geoenergy Science and Engineering 257, 214226 (2026). https://doi.org/10.1016/j.geoen.2025.214226

8. Han, S., Hu, X., Huang, H., Jiang, M., Zhao, Y.: ADBench: Anomaly detection benchmark. In: Advances in Neural Information Processing Systems. vol. 35, pp. 32142–32159 (2022)

9. Hegselmann, S., Buendia, A., Lang, H., Agrawal, M., Jiang, X., Sontag, D.: TabLLM: Few-shot classification of tabular data with large language models. In: Proceedings of The 26th International Conference on Artificial Intelligence and Statistics. pp. 5549–5581 (2023), https://proceedings.mlr.press/v206/ hegselmann23a.html

10. Hubert, M., Debruyne, M., Rousseeuw, P.J.: Minimum covariance determinant and extensions. WIREs Computational Statistics 10(3), e1421 (2018). https://doi.org/ 10.1002/wics.1421

11. Patel, D., Lin, S., Rayfield, J., Zhou, N., Vaculin, R., Martinez, N., O’Donncha, F., Kalagnanam, J.: AssetOpsBench: Benchmarking AI agents for task automation in industrial asset operations and maintenance. arXiv preprint arXiv:2506.03828 (2025). https://doi.org/10.48550/arXiv.2506.03828

12. Sandnes, A.T., Grimstad, B., Kolbjørnsen, O.: Multi-task learning for virtual flow metering. Knowledge-Based Systems 232, 107458 (2021). https://doi.org/10.1016/ j.knosys.2021.107458

13. Souza, F.A.A., Araújo, R., Mendes, J.: Review of soft sensor methods for regression applications. Chemometrics and Intelligent Laboratory Systems 152, 69–79 (2016). https://doi.org/10.1016/j.chemolab.2015.12.011

14. Theyab, M.A.: Fluid flow assurance issues: Literature review. SciFed Journal of Petroleum 2(1), 1–11 (2018)

15. Tsai, C.P., Teng, G., Wallis, P., Ding, W.: AnoLLM: Large language models for tabular anomaly detection. In: The Thirteenth International Conference on Learning Representations (2024), https://openreview.net/forum?id=7VkHfT5X2

16. Tukey, J.W.: Exploratory Data Analysis. Addison-Wesley (1977)

17. Vargas, R.E.V., Munaro, C.J., Ciarelli, P.M., Medeiros, A.G., Amaral, B.G.D., Barrionuevo, D.C., Araújo, J.C.D.D., Ribeiro, J.L., Magalhães, L.P.: A realistic and public dataset with rare undesirable real events in oil wells. Journal of Petroleum Science and Engineering 181, 106223 (2019). https://doi.org/10.1016/ j.petrol.2019.106223

18. Wu, Z., Wang, J., Cheng, Z., Zhou, Y., Wang, W., Pu, J., Li, C., Ma, C.: ICAD-LLM: One-for-all anomaly detection via in-context learning with large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 15986–15994 (2026). https://doi.org/10.1609/aaai.v40i19.38632

19. Zha, D., Bhat, Z.P., Lai, K.H., Yang, F., Jiang, Z., Zhong, S., Hu, X.: Datacentric artificial intelligence: A survey. ACM Computing Surveys 57(5), 129:1– 129:42 (2025). https://doi.org/10.1145/3711118

20. Zhao, Y., Nasrullah, Z., Li, Z.: PyOD: A Python toolbox for scalable outlier detection. Journal of Machine Learning Research 20(96), 1–7 (2019)