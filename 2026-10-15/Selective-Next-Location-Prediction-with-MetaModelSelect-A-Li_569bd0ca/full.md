# Selective Next-Location Prediction with MetaModelSelect: A Lightweight Post-Hoc Confidence Metamodel

Calvin Jary

Systems and Computer Engineering

Carleton University

Ottawa, Ontario, Canada

calvinjary@sce.carleton.ca

Nafiseh Kahani

Systems and Computer Engineering

Carleton University

Ottawa, Ontario, Canada

kahani@sce.carleton.ca

Abstract—Next-location prediction can support proactive mobile-network decisions, but acting on unreliable predictions can waste resources. We apply MetaModelSelect, a lightweight post-hoc confidence metamodel, to selective cellular next-location prediction. The mobility-specific model combines a learned embedding of the predicted location with the four largest softmax probabilities and Softmax Concentration. It learns whether a frozen recurrent neural network’s top-1 prediction is correct, using only its outputs and without retraining the backbone. On path-balanced Changchun cellular mobility trajectories, we evaluate backbones trained with cross-entropy (CE) and selfadaptive training plus entropy minimization (SAT+EM), comparing against eight scalar confidence scores, two location-aware lookup methods, and a raw-logit multilayer perceptron. Relative to maximum softmax probability, MetaModelSelect achieves mean per-run reductions in area under the risk–coverage curve (AURC), excess AURC (E-AURC), and area under the generalized risk–coverage curve (AUGRC) of 2.6%, 4.2%, and 3.0% for CE, and 2.0%, 3.3%, and 1.9% for SAT+EM. It also achieves the lowest reported mean risk at every evaluated partial coverage. The results show that MetaModelSelect improves confidence ranking in this mobility setting while leaving the underlying location predictor unchanged.

Index Terms—selective classification, next-location prediction, confidence estimation, recurrent neural networks, abstention

## I. INTRODUCTION

Next-location prediction uses recent mobility history to anticipate a user’s next observed location. Such predictions can inform proactive caching, resource allocation, and handoff preparation in mobile networks. However, a predictor that always returns a location also commits to decisions when its evidence is weak. An incorrect prediction can direct resources toward the wrong region. Average accuracy does not reveal which individual predictions are reliable enough to use.

Selective prediction adds an abstention option: a confidence score ranks the predictions, and the system accepts only those above a threshold. Its central trade-off is between coverage, the fraction accepted, and selective risk, the error rate among accepted predictions [1], [2]. This formulation separates the quality of a mobility predictor from the quality of its confidence ranking. It also allows different network decisions to use different acceptance thresholds without changing the predictor.

We study the practical post-hoc setting in which a nextlocation model is already trained and remains frozen. The available information consists of its output logits and predicted location; intermediate neural representations are unnecessary. Our testbed uses the public Changchun cellular traces [3] and a compact recurrent neural network (RNN) [4]. Each prediction concerns the next hourly observation in a clustered cellular region. The task therefore differs from predicting a precise coordinate or the next distinct place visited.

Confidence estimation for mobility is not new. Earlier methods used confidence counters [5], and LOTUS estimated trustworthiness from place information and membership probabilities [6]. We build on this motivation and on the outputonly correctness-modeling approach of MetaModelSelect [7]. Our adaptation combines a learned embedding of the predicted cellular location with the four largest softmax probabilities and Softmax Concentration. A small multilayer perceptron (MLP) learns whether the RNN’s top-1 prediction is correct. Its score controls acceptance; it does not replace the predicted location. The feature representation combines location-specific information with evidence about leading alternatives and the full output distribution.

We evaluate RNNs trained with cross-entropy (CE) and self-adaptive training plus entropy minimization (SAT+EM) [8]. Four questions structure the study: RQ1, which scalar confidence scores rank errors most effectively; RQ2, whether the metamodel improves on scalar, location-aware, and generic learned selectors; RQ3, which tested features contribute to that improvement; and RQ4, what parameter and latency costs it adds.

The contributions are a risk–coverage evaluation of selective prediction on path-balanced cellular trajectories, an output-only adaptation of learned correctness estimation to this setting, and comparisons with eight scalar scores, two location-aware lookup methods, and a raw-logit MLP. Across both backbone objectives, the metamodel achieves the lowest reported mean area under the risk–coverage curve (AURC), excess AURC (E-AURC), area under the generalized risk– coverage curve (AUGRC), and partial-coverage risks. Relative to maximum softmax probability (MSP), the reported mean per-run integrated-metric reductions are 2.6%, 4.2%, and 3.0% for CE and 2.0%, 3.3%, and 1.9% for SAT+EM. These are modest improvements in confidence ranking on the evaluated dataset; their deployment value requires validation under temporal change and downstream network objectives.

## II. BACKGROUND AND RELATED WORK

## A. Mobility prediction and confidence

Next-location prediction methods include mobility Markov chains [9], recurrent networks with spatial and temporal context [10], LSTMs [11], attention with recurrence [12], and transformers [13]. Surveys identify accuracy, scalability, and computational cost as recurring challenges [14]. The observation modality matters: cellular records associate users with serving regions, whereas GPS and point-of-interest data describe different spatial units. Our backbone follows the compact two-layer vanilla RNN previously studied for Changchun traces [4]; the present contribution concerns its acceptance decisions.

Earlier mobility work explicitly considered prediction confidence. Petzold et al. compared confidence counters for Markov and state predictors [5]. LOTUS estimated next-place trustworthiness using the current place, predicted place, and membership probability [6]. Thus, location-conditioned confidence is established motivation. Our study evaluates compact learned confidence for a frozen deep cellular predictor through modern risk–coverage metrics and includes location-aware lookup controls.

## B. Selective classification and scalar scores

The reject option trades classification errors against abstention [1]. For a fixed classifier under 0/1 loss, ranking by conditional correctness yields the ideal selective ordering [2]. AURC integrates selective risk over coverage; E-AURC subtracts the oracle contribution associated with the backbone’s error rate. AUGRC instead integrates the probability of an accepted error [15]. We report all three alongside risk at specified coverages.

Post-hoc scores include MSP [16], negative entropy, energy confidence [17], logit margin, concentration statistics, and centered p-norm-normalized logits [18]. They are inexpensive but impose a predetermined mapping from model outputs to confidence. A learned selector can combine several output characteristics and condition their interpretation on the predicted class.

## C. Learned correctness estimation

ConfidNet learns confidence from internal network features [19]. MetaModelSelect instead learns top-1 correctness using output-derived confidence statistics and a predicted-class embedding [7]. We adapt the latter principle from image classification to cellular mobility. The mobility feature set combines predicted-location identity with the top four softmax probabilities and Softmax Concentration. The feature choices and their measured effects are evaluated in the present domain.

Training-time methods offer a complementary intervention: SAT+EM modifies the backbone objective [8], while confidence-aware contrastive learning modifies representation learning [20]. We use CE and SAT+EM backbones to test whether output-only correctness modeling remains useful under different training objectives. The comparison is restricted to single-pass post-hoc selectors; it does not establish superiority over internal-feature heads, ensembles, or other access regimes.

## III. PROBLEM FORMULATION AND METAMODEL

## A. Prediction and abstention

Let $\mathbf { x } _ { t } ~ = ~ ( \ell _ { t - 4 } , \ldots , \ell _ { t } )$ contain five hourly location observations and let $y _ { t } = \ell _ { t + 1 }$ denote the next observed region. There are $K = 2 .$ ,134 location classes. A frozen RNN produces class logits $\mathbf { z } _ { t } ~ \in ~ \mathbb { R } ^ { K }$ , probabilities $\mathbf { p } _ { t } = \operatorname { s o f t m a x } ( \mathbf { z } _ { t } )$ , and prediction $\hat { c } _ { t } = \mathrm { a r g }$ max<sub>k</sub> $_ { p _ { t , k } }$ . The selector scores this existing decision; it never changes the predicted location.

For confidence score $s _ { t }$ and threshold $\tau ,$ define the accepted indices $A _ { \tau } = \{ t : s _ { t } \geq \tau \}$ on a test set of size N. Coverage and selective risk are

$$
C ( \tau ) = \frac { | A _ { \tau } | } { N } , \qquad R ( \tau ) = \frac { \sum _ { t \in A _ { \tau } } | \mathcal { k } [ \hat { c } _ { t } \neq y _ { t } ] } { | A _ { \tau } | } ,\tag{1}
$$

where risk is defined for nonempty accepted sets. At full coverage, it equals the backbone’s top-1 error. A useful selector places correct predictions before errors, reducing risk as coverage decreases.

## B. Output-derived representation

Write the sorted probabilities as $p _ { t , ( 1 ) } \geq \cdot \cdot \cdot \geq p _ { t , ( K ) }$ . The metamodel combines a learned predicted-location embedding $\mathbf { e } ( \hat { c } _ { t } ) \in \mathbb { R } ^ { d _ { e } }$ with five numerical features:

$$
\boldsymbol { \psi } _ { t } = \big [ \mathbf { e } ( \hat { c } _ { t } ) ; p _ { t , ( 1 ) } ; . . . ; p _ { t , ( 4 ) } ; \mathrm { S C } ( \mathbf { p } _ { t } ) \big ] ,\tag{2}
$$

where Softmax Concentration (SC) is

$$
\mathrm { S C } ( \mathbf { p } ) = \sum _ { k = 1 } ^ { K } p _ { k } ^ { 2 } .\tag{3}
$$

Table I summarizes the representation. The embedding makes the score conditional on the predicted region. The top-four probabilities describe the leading decision and its strongest alternatives. SC measures concentration across the full output and is monotone-equivalent to negative Gini impurity, since Gini impurity equals $1 - \textstyle \sum _ { k } p _ { k } ^ { 2 }$ . SC is distinct from the Lorenz-Gini coefficient evaluated as a baseline.

The numerical block is standardized using training-derived means and population standard deviations, with those statistics fixed for evaluation. The selected representation is held fixed before test evaluation. The test comparisons and ablations assess this representation rather than serving as a test-set feature-selection procedure.

TABLE I  
METAMODELSELECT INPUTS. THE TOTAL DIMENSION IS $d _ { e } + 5 \colon$ 48 FOR CE AND 29 FOR SAT+EM.
<table><tr><td>Input (dimension)</td><td>Information represented</td></tr><tr><td> $( d _ { e } )$ </td><td>Predicted-location embedding Location-dependent reliability.</td></tr><tr><td>Top-4 probabilities (4)</td><td>Confidence and ambiguity among leading alternatives.</td></tr><tr><td>Softmax Concentration (1)</td><td>Concentration of the full output distribution.</td></tr></table>

## C. Correctness training and inference

MetaModelSelect has two fully connected hidden layers of width 200, each followed by batch normalization and ReLU, and a scalar output logit $a _ { \phi } ( \psi _ { t } )$ . Its target is $r _ { t } = \mathbb { k } [ \hat { c } _ { t } = y _ { t } ]$ Ordinary unweighted binary cross-entropy gives

$$
\mathcal { L } ( \phi ) = \frac { 1 } { M } \sum _ { t = 1 } ^ { M } \left[ \log ( 1 + e ^ { a _ { \phi } ( \psi _ { t } ) } ) - r _ { t } a _ { \phi } ( \psi _ { t } ) \right] ,\tag{4}
$$

for M selector-training examples. Only the metamodel and its embedding are updated; the RNN remains frozen. At inference, $s _ { t } = \sigma ( a _ { \phi } ( \psi _ { t } ) )$ supplies a correctness score. Training toward correctness motivates the ranking, but a sigmoid output does not establish probability calibration or a guaranteed risk bound.

Figure 1 summarizes training. Its correctness branch denotes the supervision target: the true next location determines whether $\hat { c } _ { t }$ is correct and enters the binary loss. This branch is not part of the deployable feature vector in (2). Figure 2 shows the inference path, which uses only the predicted class and output-derived features. One frozen-RNN pass and one metamodel pass produce a score that is thresholded to accept $\hat { c } _ { t }$ or abstain. No true label, hidden RNN state, ensemble, or repeated stochastic prediction is required. The fallback after abstention is application-specific and is not evaluated here.

## IV. EXPERIMENTAL SETUP

## A. Data and evaluation protocol

The Changchun dataset contains one week of hourly observations from 2,066,000 anonymized mobile-phone users [3]. Its 7,251 base stations are represented by 2,134 clustered location classes. Filtering incomplete 24-hour trajectories and grouping identical daily paths yields 197,415 unique complete paths. The outer split assigns 157,932 paths (80%) to training and 39,483 (20%) to testing. Five-observation windows produce 19 window–target pairs per path: 3,000,708 training examples and 750,177 test examples.

Each distinct complete path contributes once, regardless of how many users followed it. This path-balanced evaluation prevents repeated trajectories from dominating the results; it does not remove location-class imbalance or estimate uservolume-weighted performance. Windows within a path overlap, and the path split is not a chronological holdout.

TABLE II  
BASE-RNN ARCHITECTURE AND TRAINING SETTINGS. SHARED SETTINGS ARE IDENTICAL FOR CE AND SAT+EM.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Window length / embedding 5 / 10</td><td></td></tr><tr><td>Recurrent architecture</td><td>2 layers, 533 hidden units</td></tr><tr><td>Normalization</td><td>Layer normalization</td></tr><tr><td>Epochs / batch size</td><td>150 / 2500</td></tr><tr><td>Optimizer / learning rate</td><td>Adam  $/ \ 2 . 4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Gradient clipping norm</td><td>1.0</td></tr><tr><td>SAT+EM-specific settings</td><td></td></tr><tr><td>SAT momentum</td><td>0.99</td></tr><tr><td>Entropy coefficient</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>SAT pretraining</td><td>0 epochs</td></tr></table>

For each objective, we train three backbones with seeds 1, 2, and 3. Selector fitting and five-fold validation use outputs from the same 80% partition used to train the RNNs. These are selector-validation folds, not out-of-fold backbone predictions. The test partition is excluded from model selection. One metamodel is fitted to each frozen backbone with selector seed 42 held constant. This protocol tests performance on heldout paths, while retaining the limitation that selector training observes in-sample backbone behavior.

## B. Backbones and selector settings

The compact two-layer vanilla RNN follows the architecture studied in [4]; Table II gives this study’s configuration. The objectives are CE and SAT+EM [8]. SAT+EM adds an abstention output during backbone training; post-hoc scoring is defined over the K location-class outputs, with softmax normalized over those classes.

Table III lists the metamodel settings. Conditional on the selected feature representation, embedding dimension $d _ { e }$ and batch size are chosen separately for each objective by trainingonly five-fold validation, minimizing AURC. The head architecture, optimizer, and training duration are fixed. The embedding has K+1 rows: index zero is reserved for a missing-entry token, and the remaining rows represent location classes. The complete input dimensions are 48 for CE $( d _ { e } = 4 3 )$ and 29 for SAT+EM $( d _ { e } = 2 4 )$

## C. Comparison methods

Table IV defines eight scalar scores, all oriented so that larger means more confident. MSP is the standard failuredetection baseline [16]. Energy temperature T and centered pNorm order $p$ are selected within the training partition by five-fold validation AURC [17], [18]. Both objectives select $p = 9 .$

Location-Reliability uses training counts of predictions and correct predictions for each predicted location $c { \mathrm { : } }$

$$
s _ { \mathrm { l o c } } ( c ) = \frac { N _ { \mathrm { c o r r e c t } } ( c ) + 1 } { N _ { \mathrm { p r e d } } ( c ) + 2 } .\tag{5}
$$

This Laplace-smoothed lookup tests location identity without distributional features.

![](images/c81415be4962ffc389ac78ad2f98136abeb65bfad356b17d7be2452b49b918a2.jpg)

Fig. 1. MetaModelSelect training. The frozen base model supplies correctness as the ground truth, to the metamodel, as a supervision target for training. The base model also outputs logits, which supply the predicted-location embedding and the five numerical features.  
![](images/8d24469c4f68257ebb8df8722e5493c56001815d037f8536fe99d53ad8211de3.jpg)  
Fig. 2. MetaModelSelect inference. The frozen base model supplies the predicted class and output logits. The metamodel combines the class embedding, top-four probabilities, and softmax concentration, then thresholds its correctness score to accept or abstain.

TABLE III  
SELECTED AND FIXED METAMODELSELECT SETTINGS. SELECTION USES FIVE-FOLD CROSS-VALIDATION WITHIN THE TRAINING SPLIT.
<table><tr><td>Selected settings:</td><td>CE / SAT+EM</td></tr><tr><td>Embedding dimension  $d _ { e }$ </td><td>43  /  24</td></tr><tr><td>MetaModelSelect batch size 63 / 59</td><td></td></tr><tr><td>Fixed settings:</td><td></td></tr><tr><td>Numerical inputs</td><td>Top-4 probabilities and SC (5)</td></tr><tr><td>Hidden architecture</td><td>2 layers, 200 units each</td></tr><tr><td>Normalization / activation</td><td>Batch normalization / ReLU</td></tr><tr><td>Output</td><td>1 logit; inference sigmoid</td></tr><tr><td>Loss</td><td>Binary cross-entropy with logits</td></tr><tr><td>Optimizer / learning rate</td><td>Adam  $/ 1 0 ^ { - 5 }$ </td></tr><tr><td>Epochs Dropout / weight decay</td><td>50 None / 0</td></tr></table>

Location-Calibrated MSP divides the training MSP range into 20 equal-width bins and combines the predicted location c with its MSP bin b:

$$
s _ { \mathrm { L C M } } ( c , b ) = \frac { N _ { \mathrm { c o r r e c t } } ( c , b ) + 5 s _ { \mathrm { l o c } } ( c ) } { N _ { \mathrm { p r e d } } ( c , b ) + 5 } .\tag{6}
$$

Sparse cells shrink toward class-level reliability; an empty

TABLE IV  
SCALAR CONFIDENCE SCORES; LARGER IS MORE CONFIDENT. ONLYENERGY TEMPERATURE T AND PNORM ORDER p ARE TUNED.
<table><tr><td>Score</td><td>Definition</td></tr><tr><td>MSP (Softmax)</td><td> $\operatorname* { m a x } _ { k } p _ { k }$ </td></tr><tr><td>Entropy</td><td> $\sum _ { k } p _ { k } \log p _ { k }$ </td></tr><tr><td>Energy</td><td> $\begin{array} { r } { T \log \sum _ { k } \exp ( z _ { k } / T ) } \end{array}$ </td></tr><tr><td>Maximum logit</td><td> $\operatorname* { m a x } _ { k } \ z _ { k }$ </td></tr><tr><td>Logit margin</td><td> $z _ { ( 1 ) } - z _ { ( 2 ) }$ </td></tr><tr><td>Softmax Concentration</td><td> $\textstyle \sum _ { k } p _ { k } ^ { 2 }$ </td></tr><tr><td>Lorenz-Gini</td><td> $2 \textstyle \sum _ { i = 1 } ^ { K } i p _ { [ i ] } \dotsc { K + 1 }$   $K \sum _ { i = 1 } ^ { K } p _ { [ i ] }$  K</td></tr><tr><td>pNorm</td><td> $\mathrm { m a x } _ { k } \big ( z _ { k } - \bar { z } \big )$   $\overline { { | | \mathbf { z } - \bar { z } \mathbf { 1 } | | _ { p } } }$ </td></tr><tr><td colspan="2"> $z _ { ( 1 ) }$  and  $z _ { ( 2 ) }$  are the two largest logits;  $\begin{array} { r } { \bar { z } = K ^ { - 1 } \sum _ { k } z _ { k } ; p _ { [ i ] } } \end{array}$  are sorted in ascending order. The energy sign is reversed to give a confidence score.</td></tr></table>

cell falls back to that value. The binning is fixed before test evaluation.

RawLogit-MLP learns correctness directly from the classlogit vector using binary cross-entropy and a fixed two-hidden-

layer head. It tests a generic learned representation against the metamodel’s compact inputs. The comparison is to this configuration; no layer-count sweep is performed.

## D. Metrics and implementation

Scores are sorted in descending order. At target coverage q, the highest-scored $\lceil q N \rceil$ examples are accepted. This evaluates a ranking at specified coverages; a deployable threshold must instead be chosen on validation data and may attain different coverage after distribution shift.

We report risk at 100%, 95%, 90%, 80%, 70%, and 50% coverage. In the tables, q denotes percentage error at $q \%$ coverage; integrated metrics are dimensionless. AURC trapezoidally integrates selective risk over coverage. E-AURC subtracts the oracle AURC for the same classifier. Generalized risk is $G ( \tau ) = C ( \tau ) R ( \tau )$ , the accepted-error fraction of all examples; AUGRC integrates this quantity [15]. Lower is better throughout. Within a fixed backbone, AURC and E-AURC differ by a selector-independent term, so their absolute improvements coincide.

Tables report means and sample standard deviations over the three backbone runs with their associated selectors. These SDs describe run-to-run variation, not confidence intervals. All methods use PyTorch. Latency is measured on an AMD Ryzen 7950X3D CPU and an NVIDIA RTX 4090 GPU. Selector times include feature computation and are incremental to the RNN forward pass. The timed metamodel uses the SAT+EMselected embedding dimension, $d _ { e } = 2 4$

## V. RESULTS AND ANALYSIS

## A. Scalar confidence scores

Table V answers RQ1. Since each selector reorders a fixed set of predictions, full-coverage risk is unchanged: 17.46% for CE and 16.61% for SAT+EM. MSP and SC give the strongest overall scalar rankings. Under CE, both have AURC 0.0421 and AUGRC 0.0337 at the table’s precision. Under SAT+EM, both again have AURC approximately 0.0387 and AUGRC approximately 0.0311.

The remaining scores are less effective overall. For example, pNorm has AURC 0.0753 under CE and 0.0481 under SAT+EM, despite training-side tuning. Energy confidence and maximum logit are particularly weak under CE. Strong performance of a confidence score in other domains therefore need not transfer to these cellular predictors. MSP and SC are the principal scalar controls for the learned selector.

## B. Main selector comparison

Table VI answers RQ2. MetaModelSelect achieves the lowest mean AURC, E-AURC, AUGRC, and risk at every listed partial coverage for both objectives. For CE, AURC decreases from 0.04207 with MSP to 0.04099; for SAT+EM, it decreases from 0.03870 to 0.03792.

Table VII summarizes three complementary comparisons. CE→MM and SATEM→MM replace MSP with the metamodel while keeping the corresponding backbone fixed. CE→SATEM instead changes the backbone objective from

![](images/72538c862b0a59ef1ff8f0ea4f8e0c4909145aec3b8ccc9c048bd62c72bd3197.jpg)  
Fig. 3. Classwise mean softmax confidence among correct CE predictions, sorted by decreasing mean. This large variation from class-to-class motivates our class-specific metamodel embedding.

CE to SAT+EM while using MSP for both. Each entry reports the mean and standard deviation of the per-run relative reduction, $1 0 0 ( M _ { \mathrm { b a s e } } - M _ { \mathrm { n e w } } ) / M _ { \mathrm { b a s e } } ,$ for metric $M ;$ these summaries can differ slightly from ratios of the rounded means in Table VI.

Table VII separates improvements from training the backbone from those obtained by changing the selector. With MSP used for both backbones, switching from CE to SAT+EM reduces AURC by 8.00%. Replacing MSP with the metamodel then reduces AURC by 2.03% on the frozen SAT+EM backbone; the corresponding reduction on the CE backbone is 2.55%. The metamodel therefore improves the reported mean ranking performance even after the backbone has been trained for selective prediction. These percentages use different reference configurations and should not be added.

At 90% coverage, CE risk falls from 12.72% with MSP to 12.42% with MetaModelSelect; SAT+EM risk falls from 11.89% to 11.73%. At 50% coverage, the corresponding reductions are 1.71% to 1.54% and 1.51% to 1.40%. These comparisons hold the accepted fraction fixed, so the improvement reflects a better ordering of the same predictions.

Location-Reliability performs poorly: predicted-location identity alone, as represented by this lookup, is insufficient. Location-Calibrated MSP improves several fixed-coverage risks but worsens AURC and E-AURC relative to MSP under both objectives. Its competitive high-coverage behavior therefore does not extend uniformly across the full ranking. RawLogit-MLP also underperforms the strongest scalar scores on the integrated metrics. The metamodel’s advantage is consistent with a useful compact representation, without establishing that generic learned selectors cannot be improved.

## C. Feature behavior and ablations

The diagnostic figures illustrate the feature design. Figure 3 shows that mean softmax confidence among correct predictions differs across locations. This motivates conditioning on location identity; because the plot conditions on correctness, it does not directly estimate each location’s error rate or demonstrate what the learned embedding encodes.

Figure 4 plots MSP against SC. Their strong association does not make them identical: predictions with similar MSP can differ in how the remaining mass is distributed. Figure 5 shows the leading two probabilities and the overlap between correct and incorrect predictions. The second probability describes a competing alternative, but neither scatter plot alone establishes a conditional error probability or the incremental value of a feature.

TABLE V  
SCALAR POST-HOC BASELINES. VALUES ARE MEAN (STANDARD DEVIATION) OVER THREE BACKBONE SEEDS; LOWER IS BETTER. CROSS-VALIDATION SELECTED p = 9 FOR PNORM UNDER BOTH OBJECTIVES.
<table><tr><td>Metric</td><td>MSP</td><td>Entropy Conf.</td><td>Energy Conf.</td><td>Max Logit</td><td>Logit Margin</td><td>Softmax Conc.</td><td>Lorenz-Gini</td><td>pNorm (p = 9)</td></tr><tr><td colspan="9">Backbone: Cross-Entropy (CE)</td></tr><tr><td>AUGRC</td><td>0.0337(1)</td><td>0.0346(1)</td><td>0.0729(4)</td><td>0.0701(4)</td><td>0.0343(2)</td><td>0.0337(1)</td><td>0.0357(1)</td><td>0.0532(7)</td></tr><tr><td>E-AURC</td><td>0.0259(1)</td><td>0.0271(1)</td><td>0.1134(12)</td><td>0.1064(10)</td><td>0.0265(1)</td><td>0.0259(1)</td><td>0.0288(1)</td><td>0.0591(11)</td></tr><tr><td>AURC</td><td>0.0421(2)</td><td>0.0433(1)</td><td>0.1296(13)</td><td>0.1226(11)</td><td>0.0427(2)</td><td>0.0421(2)</td><td>0.0450(1)</td><td>0.0753(11)</td></tr><tr><td>100</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td></tr><tr><td>95</td><td>14.86(6)</td><td>14.98(6)</td><td>16.83(7)</td><td>16.63(8)</td><td>15.24(5)</td><td>14.87(6)</td><td>15.15(6)</td><td>16.61(12)</td></tr><tr><td>90</td><td>12.72(3)</td><td>12.93(1)</td><td>16.38(7)</td><td>16.07(6)</td><td>13.08(3)</td><td>12.71(4)</td><td>13.13(7)</td><td>15.68(19)</td></tr><tr><td>80</td><td>8.76(4)</td><td>9.21(1)</td><td>15.65(6)</td><td>15.16(6)</td><td>8.96(5)</td><td>8.78(3)</td><td>9.32(4)</td><td>13.41(19)</td></tr><tr><td>70</td><td>5.43(3)</td><td>5.68(5)</td><td>14.93(6)</td><td>14.31(6)</td><td>5.45(3)</td><td>5.44(2)</td><td>6.04(4)</td><td>11.10(16)</td></tr><tr><td>50</td><td>1.71(1)</td><td>1.81(1)</td><td>13.41(13)</td><td>12.61(12)</td><td>1.68(8)</td><td>1.71(1)</td><td>2.09(2)</td><td>6.79(17)</td></tr><tr><td colspan="9">Backbone: Self-Adaptive Training + Entropy Minimization (SAT+EM)</td></tr><tr><td>AUGRC</td><td></td><td></td><td></td><td>0.0370(8)</td><td></td><td></td><td></td><td></td></tr><tr><td>E-AURC</td><td>0.0311(2)</td><td>0.0313(2)</td><td>0.0391(9)</td><td></td><td>0.0319(2)</td><td>0.0311(2)</td><td>0.0318(2)</td><td>0.0369(3) 0.0335(3)</td></tr><tr><td></td><td>0.0241 (2)</td><td>0.0243(2)</td><td>0.0378(16)</td><td>0.0346(14) 0.0493(14)</td><td>0.0251(2) 0.0398(3)</td><td>0.0240(2) 0.0387(3)</td><td>0.0253(3)</td><td></td></tr><tr><td>AURC</td><td>0.0387(3)</td><td>0.0390(3)</td><td>0.0524(16)</td><td></td><td></td><td></td><td>0.0399(3)</td><td>0.0481(4)</td></tr><tr><td>100 95</td><td>16.61(6) 14.03(4)</td><td>16.61(6) 14.06(5)</td><td>16.61(6) 14.68(7)</td><td>16.61(6) 14.43(7)</td><td>16.61(6) 14.37(5)</td><td>16.61(6) 14.01(5)</td><td>16.61(6) 14.11(5)</td><td>16.61(6)</td></tr><tr><td>90</td><td>11.89(4)</td><td>11.92(2)</td><td>13.01(9)</td><td>12.58(8)</td><td>12.17(3)</td><td>11.85(2)</td><td>12.01(2)</td><td>14.63(7) 12.83(6)</td></tr><tr><td>80</td><td>7.95(5)</td><td>8.04(4)</td><td>10.00(16)</td><td>9.35(17)</td><td>8.16(5)</td><td>7.93(4)</td><td></td><td></td></tr><tr><td>70</td><td>4.83(6)</td><td>4.87(5)</td><td>7.33(28)</td><td>6.64(25)</td><td>4.97(6)</td><td>4.84(5)</td><td>8.19(5)</td><td>9.54(8)</td></tr><tr><td>50</td><td>1.51(5)</td><td>1.55(4)</td><td>3.50(29)</td><td>3.10(23)</td><td>1.55(3)</td><td>1.52(5)</td><td>5.05(6) 1.68(3)</td><td>6.66(7) 2.82(7)</td></tr></table>

TABLE VI

MAIN SELECTIVE NEXT-LOCATION PREDICTION RESULTS. VALUES ARE MEAN (STANDARD DEVIATION) OVER THREE BACKBONE SEEDS; LOWER IS BETTER. 100 IS THE BACKBONE ERROR AND IS IDENTICAL ACROSS SELECTORS.
<table><tr><td>Metric</td><td>MSP</td><td>Softmax Conc.</td><td>Location- Reliability</td><td>Location- Calibrated MSP</td><td>RawLogit-MLP</td><td>MetaModelSelect</td></tr><tr><td colspan="7">Backbone: Cross-Entropy (CE)</td></tr><tr><td>AUGRC</td><td>0.033 69(11)</td><td>0.033 72(11)</td><td>0.05473(38)</td><td>0.033 56(10)</td><td>0.04545(29)</td><td>0.03267(16)</td></tr><tr><td>E-AURC</td><td>0.025 86(4)</td><td>0.025 89(4)</td><td>0.07429(42)</td><td>0.027 72(23)</td><td>0.056 51(34)</td><td>0.02478(4)</td></tr><tr><td>AURC</td><td>0.042 07(15)</td><td>0.042 10(15)</td><td>0.090 50(52)</td><td>0.043 93(15)</td><td>0.072 72(44)</td><td>0.040 99(15)</td></tr><tr><td>100</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td><td>17.46(6)</td></tr><tr><td>95</td><td>14.86(6)</td><td>14.87(6)</td><td>15.42(9)</td><td>14.70(7)</td><td>14.83(7)</td><td>14.68(7)</td></tr><tr><td>90</td><td>12.72(3)</td><td>12.71(4)</td><td>14.16(9)</td><td>12.47(4)</td><td>12.88(10)</td><td>12.42(6)</td></tr><tr><td>80</td><td>8.76(4)</td><td>8.78(3)</td><td>12.21(7)</td><td>8.52(5)</td><td>9.98(7)</td><td>8.42(4)</td></tr><tr><td>70</td><td>5.43(3)</td><td>5.44(2)</td><td>10.71(13)</td><td>5.25(2)</td><td>8.15(5)</td><td>5.09(3)</td></tr><tr><td>50</td><td>1.71(1)</td><td>1.71(1)</td><td>8.36(12)</td><td>1.68(2)</td><td>6.05(5)</td><td>1.54(2)</td></tr><tr><td colspan="7">Backbone: Self-Adaptive Training + Entropy Minimization (SAT+EM)</td></tr><tr><td>AUGRC</td><td>0.031 10(18)</td><td>0.031 06(17)</td><td>0.05251(41)</td><td>0.031 54(21)</td><td>0.041 25(35)</td><td>0.03052(18)</td></tr><tr><td>E-AURC</td><td>0.02406(24)</td><td>0.02402 23</td><td>0.07214(74)</td><td>0.026 82(35)</td><td>0.050 02(56)</td><td>0.023 28(25)</td></tr><tr><td>AURC</td><td>0.038 70(29)</td><td>0.038 65(28)</td><td>0.086 78(82)</td><td>0.041 45(44)</td><td>0.06466(65)</td><td>0.037 92(33)</td></tr><tr><td>100</td><td>16.61(6)</td><td>16.61(6)</td><td>16.61(6)</td><td>16.61(6)</td><td>16.61(6)</td><td>16.61(6)</td></tr><tr><td>95</td><td>14.03(4)</td><td>14.01(5)</td><td>14.80(6)</td><td>13.97(5)</td><td>14.10(6)</td><td>13.93(5)</td></tr><tr><td>90</td><td>11.89(4)</td><td>11.85 (2)</td><td>13.66(6)</td><td>11.78(6)</td><td>12.17(7)</td><td>11.73(2)</td></tr><tr><td>80</td><td>7.95(5)</td><td>7.93(4)</td><td>11.78(6)</td><td>7.87(4)</td><td>9.17(9)</td><td>7.82(4)</td></tr><tr><td>70</td><td>4.83(6)</td><td>4.84(5)</td><td>10.24(7)</td><td>4.72(5)</td><td>7.21(6)</td><td>4.67(6)</td></tr><tr><td>50</td><td>1.51(5)</td><td>1.52(5)</td><td>7.99(15)</td><td>1.71(6)</td><td>5.12(9)</td><td>1.40(4)</td></tr></table>

TABLE VII

HEADLINE RELATIVE IMPROVEMENTS (% REDUCTION; HIGHER IS BETTER). ENTRIES ARE MEAN(STD) ACROSS THREE RUNS IN COMPACT UNCERTAINTY NOTATION. CE = CROSS-ENTROPY, SAT = SAT+EM. MMS = METAMODELSELECT.
<table><tr><td>Metric</td><td>CE→MMS</td><td>CE→SATEM</td><td>SATEM→MMS</td></tr><tr><td>AUGRC</td><td>3.03(14)</td><td>7.70(71)</td><td>1.87(16)</td></tr><tr><td>E-AURC</td><td>4.16(4)</td><td>6.93(99)</td><td>3.26(59)</td></tr><tr><td>AURC</td><td>2.55(2)</td><td>8.00(82)</td><td>2.03(37)</td></tr></table>

Table VIII provides the direct feature test for RQ3 on the CE backbone. Removing the location embedding causes the largest deterioration: AURC rises from 0.04099 to 0.04197, E-AURC from 0.02478 to 0.02576, and AUGRC from 0.03267 to 0.03362. Removing the largest probability causes the nextlargest degradation. Removing each of the second, third, and fourth probabilities, or SC, also increases every mean integrated metric. These ablations support the combined representation and identify location identity and the leading softmax probability as its largest measured contributors.

## D. Computational overhead

Table IX addresses RQ4. With K + 1 embedding rows, five numerical inputs, two 200-unit hidden layers with affine batch

![](images/f95e421034d7e991d20f1a4d81c6ee420a7d478e22e553ced72b69cc314b1bd1.jpg)  
Fig. 4. Top-1 Softmax versus Softmax Concentration. There is a correlation between the two features, however the two features offer complementary information to the metamodel.

![](images/fbde952224b67e78fe315e452cd4a8191856d9402685d2a6fddb56fccd3ae6cc.jpg)  
Fig. 5. Largest versus second-largest softmax probability. A greater Top-2 Softmax increases the likelihood of an incorrect prediction.

normalization, and one output logit, the metamodel parameter count is

$$
P ( K , d _ { e } ) = ( K + 2 0 1 ) d _ { e } + 4 2 , 4 0 1 .\tag{7}
$$

For $\begin{array} { r c l } { K } & { = } & { 2 , 1 3 4 } \end{array}$ , this gives 142,806 parameters for CE $( d _ { e } = 4 3 )$ and 98,441 for SAT+EM $( d _ { e } = 2 4 )$ . The smaller configuration has approximately 4.9% as many parameters as the RNN and 79% fewer than RawLogit-MLP. At fixed embedding dimension, each additional location adds $d _ { e }$ parameters; the numerical feature block and hidden-layer widths stay constant.

For the $d _ { e } = 2 4$ metamodel, measured incremental latency is 0.131 ms per example on CPU and 0.465 ms on GPU, including feature construction. Both are below one millisecond, although they amount to approximately 29% and 80% of the respective backbone times. The combined RNN-plusselector GPU time is approximately 1.044 ms. RawLogit-MLP is faster in this setup despite having more parameters, illustrating that parameter count alone does not determine runtime. The metamodel combines compact storage and low absolute selector latency with a measurable relative cost on this small backbone.

TABLE VIII  
CE-BACKBONE FEATURE ABLATIONS. MEAN (SAMPLE STANDARD DEVIATION) OVER THREE SEEDS, WITH PARENTHESES APPLYING TO THE LAST DISPLAYED DIGITS. LOWER IS BETTER.
<table><tr><td>Variant</td><td>AURC</td><td>E-AURC</td><td>AUGRC</td></tr><tr><td>Full MetaModelSelect</td><td>0.04099(15)</td><td>0.02478(4)</td><td>0.03267(16)</td></tr><tr><td>No embedding</td><td>0.04197(24)</td><td>0.02576(16)</td><td>0.03362(15)</td></tr><tr><td>No Top  $p _ { ( 1 ) }$  Softmax</td><td>0.04137(15)</td><td>0.02516(9)</td><td>0.03305(15)</td></tr><tr><td>No Top  $p _ { ( 2 ) }$  Softmax</td><td>0.04131(17)</td><td>0.02510(11)</td><td>0.03299(14)</td></tr><tr><td>No Top p(3) Softmax</td><td>0.04122(15)</td><td>0.02501(13)</td><td>0.03288(15)</td></tr><tr><td>No Top  $p _ { ( 4 ) }$  Softmax</td><td>0.04111(18)</td><td>0.02490(12)</td><td>0.03278(16)</td></tr><tr><td>No Softmax Concentration</td><td>0.04115(20)</td><td>0.02494(11)</td><td>0.03286(16)</td></tr></table>

TABLE IX

MODEL COMPLEXITY AND PER-EXAMPLE LATENCY. METAMODEL PARAMETER COUNTS ARE CALCULATED FROM THE STATED ARCHITECTURE. PARENTHESES GIVE TIMING SDS IN THE LAST DISPLAYED DIGITS.
<table><tr><td>Method</td><td>Parameters</td><td>CPU (ms)</td><td>GPU (ms)</td></tr><tr><td>Base RNN</td><td>2,020,635</td><td>0.451(6)</td><td>0.579(6)</td></tr><tr><td>Maximum Softmax Probability</td><td>0</td><td>0.015(1)</td><td>0.029(1)</td></tr><tr><td>Softmax Concentration</td><td>0</td><td>0.014(1)</td><td>0.034(1)</td></tr><tr><td>Loc.-Rel.</td><td>2,135</td><td>0.019(1)</td><td>0.027(1)</td></tr><tr><td>Loc.-MSP</td><td>42,719</td><td>0.022(1)</td><td>0.082(1)</td></tr><tr><td>RawLogit-MLP</td><td>468,401</td><td>0.075(2)</td><td>0.276(1)</td></tr><tr><td>MetaModelSelect  $( d _ { e } = 2 4 )$ </td><td>98,441</td><td>0.131(1)</td><td>0.465(4)</td></tr></table>

Selector times include feature construction and exclude the RNN pass. Timing is available for $d _ { e } \ = 2 4 ;$ dashes indicate no timing measurement for $d _ { e } = 4 3 .$ . Lookup counts denote stored values.

## VI. DISCUSSION AND CONCLUSION

## A. What the results establish

The experiments show that MetaModelSelect can improve confidence ranking for an existing cellular predictor. MSP and Softmax Concentration are the strongest overall scalar controls, but the metamodel achieves the lowest reported mean integrated metrics and risk at every evaluated partial coverage under both CE and SAT+EM. The metamodel has 98,441 parameters with the SAT+EM-selected embedding and 142,806 with the CE-selected embedding; the smaller configuration adds less than one millisecond in the measured setup.

The distinction between predicting a location and predicting correctness informs our design. The RNN learns a multiclass mapping from mobility history; the metamodel learns which of that RNN’s decisions to accept. Its location embedding can condition the interpretation of confidence on the predicted region, while the numerical inputs describe the leading alternatives and output-distribution shape. Location-Reliability performs poorly alone, and Location-Calibrated MSP does not improve AURC or E-AURC over MSP. These controls suggest that the benefit extends beyond a location prior or a binned location/MSP lookup. The raw-logit MLP also performs worse despite its larger parameter count. This comparison supports the chosen representation within the tested configurations.

The improvements are modest in absolute size. For example, at 80% coverage, CE risk decreases from 8.76% with MSP to 8.42% with the metamodel, a reduction of 0.34 percentage points. SAT+EM lowers the backbone’s full-coverage error from 17.46% to 16.61%; applying the metamodel further improves the confidence ordering without changing that fullcoverage error. The relative reductions in Table VII summarize these complementary contributions: backbone training changes the predictions, while post-hoc selection improves which predictions are accepted. The reported means and standard deviations describe three backbone runs and should not be read as a formal significance test.

The full-coverage rows anchor this comparison. With no abstention, the CE and SAT+EM backbones have accuracies of 82.54% and 83.39%, respectively, and every selector inherits the same error rate for its backbone. A selector improves the reliability of the accepted subset rather than correcting rejected predictions. At 80% coverage, for example, the CE metamodel accepts four fifths of the examples with 91.58% accuracy, compared with 91.24% for MSP at the same coverage. The corresponding SAT+EM accuracies are 92.18% and 92.05%. Comparing selectors at a common coverage therefore separates ranking quality from the benefit obtained simply by making fewer predictions. Operationally, the appropriate coverage depends on the cost of an incorrect accepted prediction and the cost of invoking the fallback.

## B. Scope and limitations

Output-only inference permits deployment without access to hidden states or further backbone training. Nevertheless, low absolute latency does not imply negligible overhead: the metamodel adds approximately 29% of the base CPU time and 80% of its GPU time. Timing depends on the execution setup, and no downstream network-control experiment establishes whether the observed risk reductions offset this cost.

The dataset covers one city, one week, and hourly clustered region. Each unique complete daily path receives equal weight, so the results describe path-balanced prediction rather than user-volume-weighted network impact. A path split also does not establish performance on future periods or new cities. Such evaluations are needed before making robustness claims.

Selector training and hyperparameter selection use outputs from the same 80% path split used to train the backbones. The test split remains held out, but correctness labels from a separate calibration split or out-of-fold backbone predictions would provide a stricter training protocol. Fixed-coverage test rankings assess discrimination; they do not guarantee that a chosen threshold will attain a target risk or coverage after deployment. Independent selector-seed replication and validation of operational thresholds remain useful extensions.

## C. Conclusion

This study adapts compact output-only correctness modeling to selective cellular next-location prediction. A predictedlocation embedding and five softmax-derived inputs improve the confidence rankings of frozen CE and SAT+EM RNNs relative to the evaluated scalar, lookup, and raw-logit selectors.

The method provides an economical mechanism for abstention, with further validation needed under temporal shifts, different cities, and application-specific costs.

## REFERENCES

[1] C. K. Chow, “On optimum recognition error and reject tradeoff,” IEEE Transactions on Information Theory, vol. 16, no. 1, pp. 41–46, 1970.

[2] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[3] Z. Du, Y. Yang, C. Gao, L. Huang, Q. Huang, and Y. Bai, “The temporal network of mobile phone users in Changchun municipality, northeast China,” Scientific Data, vol. 5, p. 180228, 2018.

[4] C. Jary and N. Kahani, “An accurate, low-parameter and deployable ML architecture for next location prediction,” in 2023 IEEE Future Networks World Forum (FNWF). IEEE, 2023, pp. 1–6.

[5] J. Petzold, F. Bagci, W. Trumler, and T. Ungerer, “Comparison of different methods for next location prediction,” in Euro-Par 2006 Parallel Processing. Springer, 2006, pp. 909–918.

[6] P. Baumann, M. Langheinrich, A. K. Dey, and S. Santini, “Quantifying the uncertainty of next-place predictions,” in Proceedings of the 8th EAI International Conference on Mobile Computing, Applications and Services, ser. MobiCASE. ACM, 2016, pp. 74–85.

[7] C. Jary and N. Kahani, “MetaModelSelect: A lightweight post-hoc metamodel for selective classification,” 2026, unpublished manuscript.

[8] L. Feng, M. O. Ahmed, H. Hajimirsadeghi, and A. H. Abdi, “Towards better selective classification,” in International Conference on Learning Representations, 2023.

[9] S. Gambs, M.-O. Killijian, and M. Nunez del Prado Cortez, “Next˜ place prediction using mobility markov chains,” in Proceedings of the First Workshop on Measurement, Privacy, and Mobility, ser. MPM ’12. ACM, 2012, pp. 1–6.

[10] Q. Liu, S. Wu, L. Wang, and T. Tan, “Predicting the next location: A recurrent model with spatial and temporal contexts,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 30, no. 1, 2016, pp. 194–200.

[11] A. Crivellari and E. Beinat, “LSTM-based deep learning model for predicting individual mobility traces of short-term foreign tourists,” Sustainability, vol. 12, no. 1, p. 349, 2020.

[12] J. Zeng, X. He, H. Tang, and J. Wen, “Predicting the next location: A self-attention and recurrent neural network model with temporal context,” Transactions on Emerging Telecommunications Technologies, vol. 32, no. 6, p. e3898, 2021.

[13] S. E. Henouda, F. Z. Laallam, O. Kazar, and A. Sassi, “Next location prediction using transformers,” International Journal of Business Intelligence and Data Mining, vol. 21, no. 2, pp. 247–263, 2022.

[14] A. G. Chekol and M. S. Fufa, “A survey on next location prediction techniques, applications, and challenges,” EURASIP Journal on Wireless Communications and Networking, vol. 2022, no. 1, p. 29, 2022.

[15] J. Traub, T. J. Bungert, C. T. Luth, M. Baumgartner, K. H. Maier-¨ Hein, L. Maier-Hein, and P. F. Jager, “Overcoming common flaws in¨ the evaluation of selective classification systems,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[16] D. Hendrycks and K. Gimpel, “A baseline for detecting misclassified and out-of-distribution examples in neural networks,” in International Conference on Learning Representations, 2017. [Online]. Available: https://arxiv.org/abs/1610.02136

[17] W. Liu, X. Wang, J. D. Owens, and Y. Li, “Energy-based out-ofdistribution detection,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 21 464–21 475.

[18] L. F. Cattelan and D. Silva, “How to fix a broken confidence estimator: Evaluating post-hoc methods for selective classification with deep neural networks,” in Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, ser. Proceedings of Machine Learning Research, vol. 244. PMLR, 2024, pp. 547–584.

[19] C. Corbiere, N. Thome, A. Bar-Hen, M. Cord, and P. P\` erez, “Addressing´ failure prediction by learning model confidence,” in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 2902–2913.

[20] Y.-C. Wu, S.-H. Lyu, H. Shang, X. Wang, and C. Qian, “Confidenceaware contrastive learning for selective classification,” in Proceedings of the 41st International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 235. PMLR, 2024, pp. 53 706–53 729.