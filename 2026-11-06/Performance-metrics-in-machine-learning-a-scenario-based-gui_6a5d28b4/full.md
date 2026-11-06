# Performance metrics in machine learning: a scenario-based guide from the confusion matrix to calibration, regression and uncertainty

Mehmet Baygin<sup>1,\*</sup>, Sengul Dogan<sup>2</sup>, Turker Tuncer<sup>2</sup>

<sup>1</sup> Erzurum Technical University, Erzurum, Turkey

mehmet.baygin@erzurum.edu.tr

<sup>2</sup> Firat University, Elazig, Turkey

sdogan@firat.edu.tr; turkertuncer@firat.edu.tr

\* Corresponding author: Mehmet Baygin, mehmet.baygin@erzurum.edu.tr

## Abstract

Background and purpose: Machine-learning and deep-learning studies are compared through performance metrics. The same predictions can yield very different numbers depending on the metric, the class mix, the decision threshold, the averaging rule, the evaluation unit and the software convention. This methodological tutorial explains these mechanisms with controlled scenarios whose inputs every reader can recompute.

Materials and methods: Twenty-two confusion-matrix metrics were organized in a prevalencecoded metric map. Fifteen controlled scenarios (S1–S15) covered binary and multi-class classification, ranking and probability quality, regression, interval estimation, aggregation, reverse calculation and software conventions. Every scenario fixed its inputs in released files and changed one factor at a time. All values were recomputed with independent library functions (scikit-learn, statsmodels, SciPy and mlscorecheck) or independent reimplementations.

Results: All 205 verification checks passed, 202 of them independent and 3 internal. An always-negative rule reached an accuracy of 0.990 at a prevalence of 0.010, with a balanced accuracy of 0.500. At fixed sensitivity and specificity, precision fell from 0.889 to 0.008 as prevalence fell from 0.5 to 0.001. Three probability variants shared a ROC-AUC of 0.825, but their calibration slopes were 2.08, 1.04 and 0.52. With 10 participants, a participant-majority accuracy of 0.700 had a 95% interval of 0.348–0.933. Fold-averaged scores were flagged as inconsistent only when they were tested as one pooled test set.

Conclusions: The scenarios isolate mechanisms; they do not estimate how often each mechanism occurs in published studies. A literature audit and a real-data demonstration are specified as protocols in an appendix and were not executed. The released inputs, scripts, goalbased metric table and reporting checklist support transparent metric choice and reporting.

Keywords: performance metrics; confusion matrix; class imbalance; Matthews correlation coefficient; calibration; precision–recall curve; regression error; confidence intervals; reporting guidelines

## 1. Introduction

Machine-learning and deep-learning studies are ranked by numbers. A reported accuracy, F1 score or area under the curve is read as a property of the model. It is, however, also a property of the test set, the decision threshold, the averaging rule, the evaluation unit and the software that computed it. The same predictions can therefore support very different summaries. On a test set with 1% positive cases, a rule that never predicts the positive class reaches an accuracy of 0.990 and a balanced accuracy of only 0.500.

The data behind a performance estimate are determined by the evaluation design. Leakage between training and test sets can inflate performance. The same risk arises if segment-level and participant-level estimates are combined [1, 2]. These design issues are outside this guide’s scope. The design is held fixed here. Metric interpretation and reporting are examined for a fixed set of predictions.

Complete metric reports are required by guidance for prediction models and image analysis [3, 4]. Yet the reasons behind these recommendations are not always clear. Accuracy is widely accepted as misleading under imbalance. The size of this effect depends on numerical inputs. These are rarely provided, so readers often cannot repeat the calculation.

This study is a methodological tutorial on performance metrics with released and independently verified inputs. Constructed scenarios are presented; an empirical literature survey was not performed. A prevalence-coded map is used to organize 22 confusion-matrix metrics. One mechanism is examined in each of 15 controlled scenarios. Classification, probability quality, regression and reporting are covered. All values are recomputed with independent software. Internal re-derivations are counted separately. Reported values can be screened with a consistency framework that separates inconsistent reports from reports that cannot be assessed. All proportions are reported on the 0–1 scale.

## 1.1. Literature review

Surveys of classification measures established the vocabulary used in this study. Sokolova and Lapalme organized measures by their invariance to changes in the confusion matrix [5]. Ferri and colleagues grouped measures through experimental comparisons across datasets [6]. Binary and multi-class assessment methods were reviewed by Tharwat [7]. Powers described the links between precision, recall, informedness, markedness and correlation [8].

Class imbalance is often cited as a source of misleading metric values. Its effects were quantified by Luque and colleagues [9]. Brodersen and colleagues recommended balanced accuracy for unequal classes [10]. All four binary confusion-matrix cells are used in MCC [11]. Chicco and Jurman preferred MCC to accuracy and F1 for binary evaluation [12]. Gorodkin extended it to K categories [13]. Sensitivity and specificity are combined in Youden’s index [14].

Paradoxes also occur in agreement statistics. Agreement expected by chance is accounted for in Cohen’s kappa [15]. Feinstein and Cicchetti described high agreement with low kappa under skewed marginal totals [16].

Threshold-free summaries measure different aspects of performance. The true-positive and false-positive rate trade-off is described by ROC analysis [17]. ROC-AUC represents the probability that a random positive case scores above a random negative case [18]. Hand identified inconsistent misclassification costs across classifiers in this measure [19]. Precision– recall plots were preferred to ROC plots for imbalanced data by Saito and Rehmsmeier [20]. Davis and Goadrich showed that linear interpolation between precision–recall points is not achievable [21].

Probability quality is a separate property from discrimination. The Brier score and the logarithmic score are proper scoring rules [22, 23]. Guo and colleagues reported that modern neural networks are often overconfident [24]. The expected calibration error became the common summary of this effect [25]. Van Calster and colleagues described calibration as the weak point of predictive analytics [26]. Steyerberg and colleagues placed calibration and discrimination in one framework [27], and decision curve analysis adds a clinical-utility view [28].

Regression metric choice has also been debated. Willmott and Matsuura preferred mean absolute error to root mean square error [29]. Root mean square error was defended for Gaussian errors by Chai and Draxler [30]. Instability of percentage errors near zero was demonstrated by Hyndman and Koehler [31]. Chicco and colleagues compared the coefficient of determination with percentage and absolute errors [32].

Uncertainty and model comparisons require inferential methods. Wilson and Clopper–Pearson intervals are used for binomial proportions [33, 34]. Their coverage was compared by Brown and colleagues [35]. Paired classifiers on one test set can be compared with McNemar’s test [36]. DeLong’s method is used for correlated ROC curves [37]. Dietterich examined test error rates [38]. Bootstrap intervals cover other metrics [39]. Cluster-level resamples are required for clustered data [40].

Domain guidance and tools address several of these issues. Hicks and colleagues explained metric interpretation in medical artificial intelligence [41]. They also released MediMetrics. Suitable metrics were linked to image-analysis problems in Metrics Reloaded [4]. Reinke and colleagues catalogued metric-related pitfalls [42]. Required prediction-model report items are specified in TRIPOD+AI [3].

Reported values can be checked for mathematical consistency. In the GRIM test, a reported mean is checked against the stated sample size [43]. Consistency tests for binary classification scores were developed by Fazekas and Kovács [44]. These tests were released in mlscorecheck by Kovács and Fazekas [45]. Cross-validation, multi-class classification and regression are also supported by the package.

## 1.2. Literature gaps

Four gaps concern access to inputs, coverage across metric families, software conventions and interpretation of consistency checks.

• Metric behaviour is usually explained in words or through small examples. Inputs are often unavailable. Readers can follow the argument but cannot repeat its calculation or vary one factor.

• Threshold-based, probabilistic and regression metrics are usually discussed separately. A single fixed set of scenarios across these families is rarely provided with common notation.

• Undefined cases and software conventions are described in documentation. Their numerical effects are seldom quantified. The effect on a reported value is therefore difficult to assess.

• Consistency tests are available. Yet the effects of numerical precision, known class counts and the aggregation model are rarely explained. A failed check may be read as evidence of an error despite insufficient information for assessment.

## 1.3. Motivation and proposed guide

Each metric mechanism is illustrated with values traceable to fixed inputs. A metric map separates prevalence-dependent and prevalence-independent metrics at fixed sensitivity and specificity. One factor is varied in each of 15 scenarios. A common template is used: purpose, locked inputs, manipulated factor, results, figure, interpretation and reporting recommendation. Values are computed from formulas and checked with independent libraries. Consistency checks assign reported values to four outcome classes. Application goals are linked to primary and complementary metrics. Figure 1 summarizes the study structure.

Fig. 1. Structure of the scenario-based guide to performance metrics. (a) Metric map: the binary confusion matrix yields 11 prevalence-independent and 11 prevalence-dependent metrics (Figure 2), beside multi-class averaging rules, ranking and probability metrics and regression metrics; (b) controlled scenarios S1–S15 in four blocks (A confusion matrix, B scores and probabilities, C regression, D inference and reporting), built from 11 locked input files with base seed 20261102; (c) verification and consistency: the formula implementation is re-computed with scikit-learn 1.9.1, statsmodels, SciPy and mlscorecheck at the versions shown (202 independent and 3 internal checks, 0 failures), and each consistency check is assigned to one of four outcome classes; (d) reporting guide: a checklist of 13 reporting items (Table 18) linked to scenarios S1–S15, and two pre-specified protocols in Appendix D (literature audit and real-data demonstration). Neither protocol was executed. Stages within each band are grouped by card colour. The colours from Figure 2 are used for the two prevalence groups in band a. Workflow order is shown by arrows. The scope of interpretation for the synthetic scenarios is stated in the grey footer.  
![](images/4c4669a2d06a3a0df0769c33164a4d8917b76a448cb5b48cf4bd6bec31f58ee7.jpg)  
Synthetic, locked scenarios; each isolates one mechanism and does not estimate how often it occurs in published studies

The metric map, scenarios, verification layer and reporting guide are linked in Figure 1. The organization of the guide is shown. No numerical estimate is presented.

## 1.4. Novelties and contributions

The main contributions are listed below.

• Fifteen controlled scenarios are provided with released inputs. One mechanism is isolated in each scenario. All values can be recomputed from the files.

• Common notation is used for confusion-matrix, ranking, probability, regression and inferential metrics. The metric families are linked by a prevalence-coded map. Their use in applications is described in a goal-based table.

• Undefined cases and software conventions are quantified with identical inputs. The effect of each option on the reported value is demonstrated.

• A consistency framework with four outcome classes is presented and demonstrated with mlscorecheck. Inconsistent reports are distinguished from reports with insufficient information for assessment. Numerical precision is stated for every case.

The contributions concern metric interpretation and reporting. No new metric is proposed. The frequency of these mechanisms in published studies is not estimated.

## 1.5. Novelty boundary relative to established evaluation methods

The new elements are the fixed scenario set, verification code and prevalence-coded metric map. A four-class scheme is also provided for reports of consistency checks. All metrics, intervals and tests are established methods [5, 35]. The consistency tests of Fazekas and Kovács are applied through mlscorecheck [44, 45]. Existing metric calculators include MediMetrics [41]. The scenarios are synthetic. Each was designed to isolate a mechanism. Effect sizes in real studies cannot be estimated from them. A literature audit is specified in Appendix D for this purpose. It was not executed.

## 2. Materials and methods

The notation, metric families and scenario design are defined in this section. Uncertainty estimation, consistency checks and verification procedures are then described. The scope of the study is also stated.

## 2.1. Notation and the confusion matrix

A binary test set holds � actual positive and � actual negative cases, with $T = P + N$ . At a fixed decision threshold, the classifier produces $T P$ true positives, � false negatives, � false positives and � true negatives. The predicted totals are $P P = T P + F P$ and $P N = F N + T N$ The prevalence is $\pi = { P \mathord { \left/ { \vphantom { P T } } \right. \kern - delimiterspace } T }$ . The positive class is specified in every scenario. Several metric values are altered by a swap of class labels. The 22 metrics derived from these counts are mapped in Figure 2.

<table><tr><td colspan="2" rowspan="2">Total  $T = P + N$  1,000</td><td rowspan="2">Predicted positive  $P P = T P + F P$ </td><td>Predicted negative</td><td>Prevalence</td><td>Accuracy</td><td>Balanced accuracy</td></tr><tr><td> $P N = F N + T N$  830</td><td> ${ \pi } = { P } / { T }$  0.100</td><td> $\begin{array} { r } { A c c = \frac { T P + T N } { T } } \end{array}$  0.890</td><td> $\begin{array} { r } { B A = \frac { T P R + T N R } { 2 } } \end{array}$  0.850</td></tr><tr><td colspan="2">Actual positive  $P = T P + F N$ </td><td>170 True positive</td><td>False negative</td><td>Sensitivity, recall</td><td>Miss rate</td><td>Likelihood ratio (+)</td></tr><tr><td colspan="2">100</td><td>TP 80</td><td>FN 20</td><td> $\begin{array} { r } { T P R = { \frac { T P } { P } } } \end{array}$  0.800</td><td> $\begin{array} { r } { F N R = { \frac { F N } { P } } } \end{array}$  0.200</td><td> $\begin{array} { r } { L R ^ { + } = \frac { T P R } { F P R } } \end{array}$  8.00</td></tr><tr><td colspan="2">Actual negative</td><td>False positive</td><td>True negative</td><td>False alarm rate</td><td>Specificity</td><td>Likelihood ratio (−)</td></tr><tr><td colspan="2"> $N = F P + T N$  900</td><td>FP</td><td>TN</td><td> $\begin{array} { r } { F P R = \frac { F P } { N } } \end{array}$ </td><td> $\begin{array} { r } { T N R = { \frac { T N } { N } } } \end{array}$ </td><td> $\begin{array} { r } { L R ^ { - } = \frac { F N R } { T N R } } \end{array}$ </td></tr><tr><td colspan="2">Counts</td><td>90 Precision</td><td>810 False omission rate</td><td>0.100 Informedness</td><td>0.900 Markedness</td><td>0.222 Diagnostic odds ratio</td></tr><tr><td colspan="2" rowspan="3">Unchanged at fixed TPR and TNR Changes with</td><td> $\begin{array} { r } { P P V = \frac { T P } { P P } } \end{array}$ </td><td> $\begin{array} { r } { F O R = { \frac { F N } { P N } } } \end{array}$ </td><td> $B M = T P R + T N R - 1$ </td><td> $M K = P P V + N P V - 1$ </td><td> $\begin{array} { r } { D O R = \frac { L R ^ { + } } { L R ^ { - } } } \end{array}$ </td></tr><tr><td>0.471</td><td>0.024</td><td>0.700</td><td>0.446</td><td>36.0</td></tr><tr><td>False discovery rate</td><td>Neg. predictive value</td><td> $\mathrm { F _ { 1 } \ s c o r e }$ </td><td>Matthews correlation coefficient</td><td></td></tr><tr><td colspan="2">prevalence π</td><td> $\begin{array} { r } { F D R = { \frac { F P } { P P } } } \end{array}$ </td><td> $\begin{array} { r } { N P V = \frac { T N } { P N } } \end{array}$ </td><td> $\begin{array} { r } { F _ { 1 } = \frac { 2 T P } { 2 T P + F P + F N } } \end{array}$ </td><td> $\begin{array} { r } { M C C = \frac { T P \cdot T N - F P \cdot F N } { \sqrt { P P \cdot P N \cdot P \cdot N } } } \end{array}$ </td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td>0.529 Jaccard index (CSI)</td><td></td><td>0.593</td><td>0.559</td><td></td></tr><tr><td colspan="2"></td><td></td><td>0.976 Fowlkes-Mallows</td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td> $\mathrm { C o h e n " s \ k a p p a }$ </td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Worked example:</td><td></td><td></td><td></td><td>Geometric mean</td><td>Prevalence threshold</td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">TP = 80, FN = 20</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td> $\sqrt { P P V \cdot T P R }$ </td><td> $\begin{array} { r } { \kappa = \frac { A c c - p _ { e } } { 1 - p _ { e } } } \end{array}$ </td><td></td><td> $\frac { \sqrt { T P R \cdot F P R } - F P R } { T P R - F P R }$ </td></tr><tr><td colspan="2"></td><td> $\frac { T P } { T P + F P + F N }$ </td><td></td><td></td><td> $\sqrt { T P R \cdot T N R }$ </td><td></td></tr><tr><td colspan="2"> $F P = 9 0 , T N = 8 1 0$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td>0.614</td><td>0.534</td><td>0.849</td><td>0.261</td></tr><tr><td colspan="2"></td><td>0.421</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Fig. 2. Map of confusion-matrix metrics with a numerical example. Each cell contains a metric name, formula and value. The example uses TP = 80, FN = 20, FP = 90 and $\mathrm { T N } = 8 1 0 ,$ at a prevalence of 0.100. Blue cells mark metrics unchanged at fixed TPR and TNR. Gold cells mark metrics affected by prevalence π. Counts and margins are shown in grey cells. Proportions are reported on the 0–1 scale to three decimals. LR+ is reported to two decimals and DOR to one. Counts in the confusion-matrix cells are coloured blue (true) and red (false).

In the kappa cell, $\mathrm { p \_ e } = ( \mathrm { P \cdot \mathrm { P P + N \cdot \mathrm { P N } } ) / \mathrm { T } ^ { 2 } }$ is the chance agreement.

Figure 2 shows each metric with its formula and its value for one worked confusion matrix. Its colour coding states which metrics stay fixed when only prevalence changes and does not rank the metrics.

The colour coding was verified numerically. A second matrix with the same sensitivity and specificity but a prevalence of 0.010 left all 11 blue-coded metrics unchanged and changed all 11 gold-coded metrics. Section 3.1 examines this mechanism over six prevalences.

## 2.2. Metric families

## 2.2.1. Threshold-based binary metrics

Accuracy is the proportion of correct decisions. It is defined in Eq. (1).

$$
\mathrm { A c c } = { \frac { T P + T N } { T } } = \pi \mathrm { T P R } + ( 1 - \pi ) \mathrm { T N R } ,\tag{1}
$$

where TPR is the true-positive rate (sensitivity) and TNR is the true-negative rate (specificity). In Eq. (1), the second form shows that accuracy is a prevalence-weighted mean of the two class-wise rates. This identity is used again in Section 2.5.

The class-wise rates condition on the actual class, whereas the predictive values condition on the predicted class. Eq. (2) defines the two pairs.

$$
\mathrm { T P R } = \frac { T P } { P } , \quad \mathrm { T N R } = \frac { T N } { N } , \quad \mathrm { P P V } = \frac { T P } { P P } , \quad \mathrm { N P V } = \frac { T N } { P N } ,\tag{2}
$$

where PPV is the positive predictive value (precision) and NPV is the negative predictive value. In Eq. (2), �� and � are the predicted positive and negative totals. The rates in the first pair do not depend on $\pi ;$ the predictive values do.

The $F _ { \beta }$ score combines precision and recall with a weight on recall. Eq. (3) gives its count form.

$$
F _ { \beta } = \frac { ( 1 + \beta ^ { 2 } ) T P } { ( 1 + \beta ^ { 2 } ) T P + \beta ^ { 2 } F N + F P } ,\tag{3}
$$

where $\beta > 0$ sets the relative weight of recall. In Eq. (3), $\beta = 1$ gives the F1 score, $2 T P / ( 2 T P + F P + F N )$ . The score does not contain � , which Section 3.1 examines.

Balanced accuracy and informedness summarize the two class-wise rates. Eq. (4) defines them.

$$
\mathrm { B A } = \frac { \mathrm { T P R } + \mathrm { T N R } } { 2 } , \quad \mathrm { B M } = \mathrm { T P R } + \mathrm { T N R } - 1 ,\tag{4}
$$

where BA is balanced accuracy and BM is bookmaker informedness, which equals Youden’s index [14]. In Eq. (4), both quantities use only the class-wise rates and are therefore unchanged by prevalence.

The Matthews correlation coefficient is the Pearson correlation between actual and predicted binary labels [11]. Eq. (5) gives it in marginal form.

$$
\mathsf { M C C } = \frac { T P \cdot T N - F P \cdot F N } { \sqrt { P P \cdot P N \cdot P \cdot N } } ,\tag{5}
$$

where the denominator is the product of the four marginal totals. In Eq. (5), MCC is undefined when any marginal total is zero. Swapping the class labels leaves it unchanged.

Cohen’s kappa compares observed agreement with agreement expected from the marginal totals [15]. Eq. (6) defines it.

$$
{ \kappa } = \frac { \mathsf { A c c } - p _ { e } } { 1 - p _ { e } } , \quad p _ { e } = \frac { P \cdot P P + N \cdot P N } { T ^ { 2 } } ,\tag{6}
$$

where $p _ { e }$ is the chance agreement implied by the marginal totals. In Eq. (6), the observed agreement is accuracy. The remaining metrics of Figure 2 follow from the same counts; Table 1 lists their ranges, undefined cases and infinite cases.

Table 1. Properties of the metrics used in this guide. Prevalence: whether the value changes when only the class mix changes at fixed class-wise rates (for score metrics, at fixed classconditional score distributions); not applicable for regression. Undefined: the formula is 0/0 or has no value. Infinite: a non-zero numerator meets a zero denominator, or a logarithm of zero occurs. LR+ and LR− = positive and negative likelihood ratios; DOR = diagnostic odds ratio; MK = markedness; FM = Fowlkes–Mallows index; PT = prevalence threshold.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Range</td><td rowspan=1 colspan=1>Best</td><td rowspan=1 colspan=1>Prevalence</td><td rowspan=1 colspan=1>Undefined (0/0)</td><td rowspan=1 colspan=1>Infinite</td></tr><tr><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>T = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Sensitivity (TPR)</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>P = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Specificity (TNR)</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>N= 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Precision (PPV)</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>PP = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>NPV</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>PN = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>TP + FP + FN = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Balanced accuracy</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>P = 0 or N = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Informedness (BM)</td><td rowspan=1 colspan=1>-1 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>P = 0 or N = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>MK</td><td rowspan=1 colspan=1>-1 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>PP = 0 or PN = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>MCC</td><td rowspan=1 colspan=1>-1 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>any marginal total =0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Cohen&#x27;s kappa</td><td rowspan=1 colspan=1>at most 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>chance agreement =1</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>LR+</td><td rowspan=1 colspan=1>0 to ∞</td><td rowspan=1 colspan=1>higher</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>TPR = FPR = 0</td><td rowspan=1 colspan=1>FPR = 0 andTPR &gt; 0</td></tr><tr><td rowspan=1 colspan=1>LR-</td><td rowspan=1 colspan=1>0 to ∞</td><td rowspan=1 colspan=1>lower</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>FNR = TNR = 0</td><td rowspan=1 colspan=1>TNR = 0 andFNR&gt; 0</td></tr><tr><td rowspan=1 colspan=1>DOR</td><td rowspan=1 colspan=1>0 to ∞</td><td rowspan=1 colspan=1>higher</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>TP·TN = FP·FN= 0</td><td rowspan=1 colspan=1>FP · FN = 0 andTP · TN &gt; 0</td></tr><tr><td rowspan=1 colspan=1>Jaccard index</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>TP + FP + FN = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>FM</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>PP = 0 or P = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Geometric mean</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>P = 0 or N = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>PT</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>context-dependent</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>TPR = FPR</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>ROC-AUC</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>fixed</td><td rowspan=1 colspan=1>P = 0 or N = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Average precision</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>P = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Brier score</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>T = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>0 to ∞</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>T = 0</td><td rowspan=1 colspan=1>p = 0 or 1 forthe wrong label</td></tr><tr><td rowspan=1 colspan=1>ECE</td><td rowspan=1 colspan=1>0 to 1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>changes</td><td rowspan=1 colspan=1>T = 0</td><td rowspan=1 colspan=1>never; valuedepends on thebins</td></tr><tr><td rowspan=1 colspan=1>MAE, RMSE,MedAE</td><td rowspan=1 colspan=1>0 to ∞</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>not applicable</td><td rowspan=1 colspan=1>T = 0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>MAPE</td><td rowspan=1 colspan=1>0 to ∞</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>not applicable</td><td rowspan=1 colspan=1>target = error = 0</td><td rowspan=1 colspan=1>target = 0 anderror ≠ 0</td></tr><tr><td rowspan=1 colspan=1>sMAPE</td><td rowspan=1 colspan=1>0 to 2</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>not applicable</td><td rowspan=1 colspan=1>target = prediction=0</td><td rowspan=1 colspan=1>never</td></tr><tr><td rowspan=1 colspan=1>R²</td><td rowspan=1 colspan=1>-∞ to 1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>not applicable</td><td rowspan=1 colspan=1>constant targetswith zero error</td><td rowspan=1 colspan=1>constant targetswith non-zeroerror</td></tr></table>

Table 1 fixes the properties that the scenarios test one by one. Its prevalence column refers to a change in class mix at fixed class-wise rates and does not describe other kinds of distribution shift.

## 2.2.2. Multi-class averaging

A K-class confusion matrix gives one-versus-rest counts $T P _ { k } , F P _ { k }$ and $F N _ { k }$ for each class � with support $n _ { k }$ . Eq. (7) defines the three common F1 summaries.

$$
\begin{array} { l } { { \displaystyle F _ { 1 } ^ { \mathrm { m a c r o } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } F _ { 1 , k } , \quad { \displaystyle F } _ { 1 } ^ { \mathrm { w e i g h t e d } } = \sum _ { k = 1 } ^ { K } \frac { n _ { k } } { T } F _ { 1 , k } , \quad { \displaystyle F } _ { 1 } ^ { \mathrm { m i c r o } } } } \\ { { \displaystyle = \frac { 2 \sum _ { k } T P _ { k } } { 2 \sum _ { k } T P _ { k } + \sum _ { k } F P _ { k } + \sum _ { k } F N _ { k } } , } } \end{array}\tag{7}
$$

where $F _ { 1 , k }$ is the F1 score of class � against all other classes. In Eq. (7), � is the number of cases. In single-label problems the micro average equals accuracy, and the macro average of recall equals balanced accuracy. The multi-class MCC is Gorodkin’s $R _ { K }$ [13].

## 2.2.3. Ranking and probability metrics

The ROC-AUC is computed from scores without a threshold. Eq. (8) gives its Mann–Whitney form [18].

$$
\mathrm { A U C } = \frac { 1 } { P N _ { \mathrm { p a i r s } } } \sum _ { i \in \mathcal { P } } \sum _ { j \in \mathcal { N } } \big [ \mathbb { 1 } ( s _ { i } > s _ { j } ) + 1 / 2 \mathbb { 1 } ( s _ { i } = s _ { j } ) \big ] ,\tag{8}
$$

where $s _ { i }$ is the score of case �, � and � are the sets of positive and negative cases, and $P N _ { \mathrm { p a i r s } } = P \cdot N$ . In Eq. (8), �(⋅) is the indicator function and ties count one half. Average precision summarizes the precision–recall curve as a step function. Eq. (9) defines it.

$$
{ \mathrm { A P } } = \sum _ { m } \left( R _ { m } - R _ { m - 1 } \right) P _ { m } ,\tag{9}
$$

where $P _ { m }$ and $R _ { m }$ are precision and recall at the �-th distinct score threshold, ordered by decreasing score, and $R _ { 0 } = 0$ . In Eq. (9), a non-informative ranking has a population precision equal to the prevalence � at every recall level, so � is the population reference line [20]. In a finite sample the expected AP of a random ranking exceeds �. With one positive and one negative case, the two possible rankings give AP values of 1 and 0.5, so the expected AP is 0.75 at a prevalence of 0.5.

Probability quality is summarized by proper scoring rules [23]. Eq. (10) defines the Brier score and the log loss.

$$
{ \mathrm { B r i e r } } = { \frac { 1 } { T } } { \sum _ { i = 1 } ^ { T } } ( p _ { i } - y _ { i } ) ^ { 2 } , \quad { \mathrm { L o g L o s s } } = - { \frac { 1 } { T } } { \sum _ { i = 1 } ^ { T } } [ y _ { i } \mathrm { l n } p _ { i } + ( 1 - y _ { i } ) \mathrm { l n } ( 1 - p _ { i } ) ] ,\tag{10}
$$

where $p _ { i }$ is the predicted probability of the positive class and $y _ { i } \in \{ 0 , 1 \}$ is the label. In Eq. (10), lower values are better for both rules [22]. The expected calibration error bins the probabilities. Eq. (11) defines it.

$$
\mathrm { E C E } _ { B } = \sum _ { b = 1 } ^ { B } \frac { n _ { b } } { T } \big | \bar { y } _ { b } - \bar { p } _ { b } \big | ,\tag{11}
$$

where � is the number of equal-width bins, $n _ { b }$ is the number of cases in bin �, $\hat { y } _ { b }$ is their observed positive rate and $\hat { p } _ { b }$ is their mean predicted probability [25]. In Eq. (11), empty bins contribute zero. The calibration slope and intercept come from logistic models on the logit of the predicted probability [26]. Eq. (12) states the two models.

$$
\begin{array} { r } { \mathrm { l o g i t P r } ( y _ { i } = 1 ) = a + b \mathrm { l o g i t } ( p _ { i } ) , \qquad \mathrm { l o g i t P r } ( y _ { i } = 1 ) = a _ { 0 } + \mathrm { l o g i t } ( p _ { i } ) , } \end{array}\tag{12}
$$

where � is the calibration slope and $a _ { 0 }$ is the calibration intercept with the slope fixed at one. In Eq. (12), a well-calibrated model has $b = 1$ and $a _ { 0 } = 0$ . A slope below one indicates overconfident probabilities, and a slope above one indicates underconfident probabilities.

## 2.2.4. Regression metrics

Regression errors are $e _ { i } = \hat { y } _ { i } - y _ { i }$ for � test targets. Eq. (13) defines the two absolute-scale summaries.

$$
\mathrm { M A E } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } | \boldsymbol { e } _ { i } | , \quad \mathrm { R M S E } = \sqrt { \frac { 1 } { T } \sum _ { i = 1 } ^ { T } { e _ { i } ^ { 2 } } } ,\tag{13}
$$

where $\hat { y } _ { i }$ is the prediction and $y _ { i }$ is the target. In Eq. (13), $\mathrm { R M S E } \geq \mathrm { M A E }$ always holds for one error vector with equal weights. The median absolute error (MedAE) replaces the mean by the median. Eq. (14) defines the relative errors.

$$
\mathsf { M A P E } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \frac { \lvert e _ { i } \rvert } { \lvert y _ { i } \rvert } , \quad \mathsf { s M A P E } = \frac { 1 } { T } \sum _ { i = 1 } ^ { T } \frac { 2 \lvert e _ { i } \rvert } { \lvert y _ { i } \rvert + \lvert \hat { y } _ { i } \rvert } ,\tag{14}
$$

where a target $y _ { i } = 0$ gives an infinite term when $e _ { i } \neq 0$ and an undefined term when $e _ { i } = 0$ [31]. In Eq. (14), sMAPE is the symmetric variant with a per-case value of at most 2. The coefficient of determination compares the model with a constant prediction. Eq. (15) defines it.

$$
R ^ { 2 } = 1 - \frac { \sum _ { i } { ( y _ { i } - \hat { y } _ { i } ) } ^ { 2 } } { \sum _ { i } { ( y _ { i } - \bar { y } ) } ^ { 2 } } ,\tag{15}
$$

where $\bar { y }$ is the mean of the test targets. In Eq. (15), $R ^ { 2 }$ is negative when the model is worse than predicting �‾ for every case [32].

## 2.3. Controlled scenario design

Each scenario isolates one mechanism. It fixes all inputs except one manipulated factor. Confusion-matrix scenarios lock the four counts. Score-based scenarios lock the label and score vectors, because one confusion matrix cannot determine an AUC or a calibration curve. Random inputs were drawn once with seed 20261102 (offsets +6, +7 and +9 for S6, S7 and S9) and saved as CSV files. These saved inputs are reused without regeneration. The model for each random input is specified in Appendix Table C1. Fold counts for S14 are provided in Appendix Table C2. A common template is used for all scenarios: purpose, locked inputs, manipulated factor, results, figure, interpretation and reporting recommendation. The 15 scenarios are arranged in four blocks in Table 2.

Table 2. Controlled scenarios S1–S15. The four blocks cover confusion matrices (A), scores and probabilities (B), regression (C), and inference and reporting (D). Fixed inputs are listed in Appendix Table A1.
<table><tr><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>Block</td><td rowspan=1 colspan=1>Mechanism</td><td rowspan=1 colspan=1>Locked input</td><td rowspan=1 colspan=1>Manipulated factor</td></tr><tr><td rowspan=1 colspan=1>S1</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>Accuracy paradox</td><td rowspan=1 colspan=1>two confusion matrices,prevalence 0.010</td><td rowspan=1 colspan=1>classifier</td></tr><tr><td rowspan=1 colspan=1>S2</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>Prevalence shift</td><td rowspan=1 colspan=1>TPR 0.800, TNR 0.900, T =10,000</td><td rowspan=1 colspan=1>prevalence</td></tr><tr><td rowspan=1 colspan=1>S3</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>F1, MCC and positive class</td><td rowspan=1 colspan=1>TP 90, FN 10, FP 10; one 2 × 2table</td><td rowspan=1 colspan=1>TN count; positive label</td></tr><tr><td rowspan=1 colspan=1>S4</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>Multi-class averaging</td><td rowspan=1 colspan=1>one 3 × 3 confusion matrix</td><td rowspan=1 colspan=1>averaging rule</td></tr><tr><td rowspan=1 colspan=1>S5</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>ROC and PR under lowprevalence</td><td rowspan=1 colspan=1>100 positive and 100 negativebinormal scores</td><td rowspan=1 colspan=1>prevalence (0.5, 0.010)</td></tr><tr><td rowspan=1 colspan=1>S6</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>Threshold and error cost</td><td rowspan=1 colspan=1>validation and test sets, 2,000cases each</td><td rowspan=1 colspan=1>threshold rule</td></tr><tr><td rowspan=1 colspan=1>S7</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>Discrimination andcalibration</td><td rowspan=1 colspan=1>5,000 logits and labels</td><td rowspan=1 colspan=1>logit scale</td></tr><tr><td rowspan=1 colspan=1>S8</td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>Kappa and marginal totals</td><td rowspan=1 colspan=1>two 2 × 2 tables, n = 100</td><td rowspan=1 colspan=1>marginal distribution</td></tr><tr><td rowspan=1 colspan=1>S9</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>Outliers in regression error</td><td rowspan=1 colspan=1>100 targets, two error vectors</td><td rowspan=1 colspan=1>error distribution</td></tr><tr><td rowspan=1 colspan=1>S10</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>Near-zero targets</td><td rowspan=1 colspan=1>10 targets, error 1</td><td rowspan=1 colspan=1>target magnitude</td></tr><tr><td rowspan=1 colspan=1>S11</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>Negative R²</td><td rowspan=1 colspan=1>21 targets, three predictors</td><td rowspan=1 colspan=1>reference prediction</td></tr><tr><td rowspan=1 colspan=1>S12</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>Small test sets</td><td rowspan=1 colspan=1>accuracy 0.950 at three sizes</td><td rowspan=1 colspan=1>test size</td></tr><tr><td rowspan=1 colspan=1>S13</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>Aggregation level</td><td rowspan=1 colspan=1>10 participants, 700 segments</td><td rowspan=1 colspan=1>aggregation rule</td></tr><tr><td rowspan=1 colspan=1>S14</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>Reverse calculation</td><td rowspan=1 colspan=1>eight reported triplets, five folds</td><td rowspan=1 colspan=1>reported values,rounding, aggregation</td></tr><tr><td rowspan=1 colspan=1>S15</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>Software conventions</td><td rowspan=1 colspan=1>S1, S5 and S10 inputs, smalllabel vectors</td><td rowspan=1 colspan=1>library option</td></tr></table>

The mechanism and varied factor for each scenario are specified in Table 2. Possible effects are demonstrated with constructed inputs. Their frequency in practice is not estimated.

Thresholds for S6 were selected on a validation set. They were then applied without change to a separate test set of equal size. Test data were excluded from threshold selection. For the lowprevalence set in S5, the same 100 positive scores were retained. Each negative score was repeated 99 times. Class-conditional score distributions were thus identical to those in the balanced set. Labels in S7 were drawn as $y _ { i } \sim \mathrm { B e r n o u l l i } ( \mathrm { s i g m o i d } ( z _ { i } ) )$ . Thus, sigmoid(��) is the true event probability. The variant with � = 1 is calibrated by construction.

## 2.4. Uncertainty and model comparison

A proportion such as accuracy, sensitivity or specificity is a binomial estimate when cases are independent. The Wilson score interval is defined in Eq. (16) [33].

$$
\frac { \hat { p } + \displaystyle \frac { z ^ { 2 } } { 2 n } } { 1 + \displaystyle \frac { z ^ { 2 } } { n } } \pm \frac { z } { 1 + \displaystyle \frac { z ^ { 2 } } { n } } \sqrt { \frac { \hat { p } ( 1 - \hat { p } ) } { n } + \frac { z ^ { 2 } } { 4 n ^ { 2 } } } ,\tag{16}
$$

where $\hat { p } = k / n$ is the observed proportion of � successes in � cases and $z = 1 . 9 6$ for a 95% interval. In Eq. (16), the interval stays between 0 and 1. The exact Clopper–Pearson interval uses beta quantiles [34]. Both intervals have better coverage than the normal approximation [35]. Repeated segments of one participant are not independent [40]. For S13, participants were therefore resampled with replacement (10,000 resamples, seed 20261115) and percentile intervals were computed; the participant-majority proportion received an exact Clopper– Pearson interval. Table 3 lists the methods for other quantities, their assumptions and whether each was applied here.

Table 3. Interval and test methods by quantity. Independent cases are assumed unless a cluster method is named. Status: whether the method was applied to a scenario in this study.
<table><tr><td rowspan=1 colspan=1>Quantity</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Assumption</td><td rowspan=1 colspan=1>Status</td></tr><tr><td rowspan=1 colspan=1>Proportion (accuracy,sensitivity, specificityprecision)</td><td rowspan=1 colspan=1>Wilson score interval;exact Clopper-Pearsoninterval</td><td rowspan=1 colspan=1>independent cases; fixeddenominator</td><td rowspan=1 colspan=1>applied (S12, S13)</td></tr><tr><td rowspan=1 colspan=1>Paired decisions of twoclassifiers on one test set</td><td rowspan=1 colspan=1>McNemar&#x27;s test</td><td rowspan=1 colspan=1>independent cases; sametest set</td><td rowspan=1 colspan=1>recommended; notapplied</td></tr><tr><td rowspan=1 colspan=1>Two correlated ROC-AUCs</td><td rowspan=1 colspan=1>DeLong&#x27;s method</td><td rowspan=1 colspan=1>independent cases; sametest set</td><td rowspan=1 colspan=1>recommended; notapplied</td></tr><tr><td rowspan=1 colspan=1>MCC, F1, AP, Brier, ECEand other functions of thedata</td><td rowspan=1 colspan=1>percentile bootstrap</td><td rowspan=1 colspan=1>resampling unit is theindependent unit</td><td rowspan=1 colspan=1>recommended; notapplied</td></tr><tr><td rowspan=1 colspan=1>Repeated measurements ofone participant</td><td rowspan=1 colspan=1>cluster bootstrap overparticipants</td><td rowspan=1 colspan=1>participants areindependent</td><td rowspan=1 colspan=1>applied (S13)</td></tr><tr><td rowspan=1 colspan=1>Scores averaged over cross-validation folds</td><td rowspan=1 colspan=1>no exact interval; foldresults reported</td><td rowspan=1 colspan=1>folds share training data</td><td rowspan=1 colspan=1>illustrated (S14)</td></tr></table>

Table 3 links each quantity to an interval or test and to the assumption behind it. Methods marked as recommended were not applied to any scenario in this study.

## 2.5. Consistency assessment framework

Reported values can be checked against each other. Eq. (1) implies that accuracy lies between sensitivity and specificity. When TPR ≠ TNR, it also implies the prevalence in Eq. (17).

$$
\pi = \frac { \mathrm { A c c } - \mathrm { T N R } } { \mathrm { T P R } - \mathrm { T N R } } ,\tag{17}
$$

where all three values come from one test set, one threshold and one aggregation level. In Eq. (17), rounding enters through the denominator. Each reported value was therefore treated as an interval of half a unit of its last reported decimal, and the reporting precision of every case is kept in Table 14. The implied prevalence was evaluated over that interval box rather than at one point.

When the class counts � and � are known, a stronger check enumerates all integer pairs (� , � ) and keeps those that reproduce every reported value within its rounding interval. An empty set means that the values cannot belong to one confusion matrix. The same tests were run with mlscorecheck 1.0.3 [45]. Its fold-wise model treats scores as means over folds (mean of scores), and its single-test-set model treats them as computed on one pooled confusion matrix [44]. Table 4 defines the four outcome classes used to report a check.

Table 4. Outcome classes for a consistency check. A check is made under a stated model of the experiment: one test set or a mean over folds. Examples refer to the cases of S14 (Table 14).
<table><tr><td rowspan=1 colspan=1>Outcome</td><td rowspan=1 colspan=1>Condition</td><td rowspan=1 colspan=1>Example in S14</td></tr><tr><td rowspan=1 colspan=1>Consistent</td><td rowspan=1 colspan=1>at least one confusion matrix reproducesthe values exactly</td><td rowspan=1 colspan=1>case A: 1 exact and 9 roundingcompatible (TP, TN) pairs</td></tr><tr><td rowspan=1 colspan=1>Consistent withinrounding</td><td rowspan=1 colspan=1>a confusion matrix reproduces the valuesonly within half a unit of the last decimal</td><td rowspan=1 colspan=1>case E: 0 exact and 1 rounding-compatible pair</td></tr><tr><td rowspan=1 colspan=1>Inconsistent</td><td rowspan=1 colspan=1>no confusion matrix reproduces the valuesunder the stated model</td><td rowspan=1 colspan=1>case B (accuracy outside the intervalbetween TPR and TNR); case D</td></tr><tr><td rowspan=1 colspan=1>Not assessable</td><td rowspan=1 colspan=1>the information or the model does notallow a check</td><td rowspan=1 colspan=1>case C2 (TPR close to TNR); foldmeans tested as one pooled test set</td></tr></table>

Table 4 separates a failed check from a check that could not be made. A report classed as inconsistent is not thereby shown to contain an error in the underlying experiment.

## 2.6. Implementation and verification

All metrics were implemented from their formulas in Python. The implementation returns nota-number whenever a denominator is zero instead of applying a convention. Table 1 and Table 15 classify each such case as undefined (0/0) or infinite. The scenario builder wrote the locked inputs and all results to one results file. A separate script recomputed the reported values with scikit-learn 1.9.1 [46], statsmodels 0.15.0 [47], SciPy 1.17.1 [48] and mlscorecheck 1.0.3 [45]. Each check was assigned to one of three kinds. An independent library check recomputed a value with a function of another library, either exactly or within a Monte Carlo tolerance when random resampling was involved. An independent re-implementation coded the same formula separately, for example the ECE weighting on library bin statistics. An internal check rederived a value from the input file with the same arithmetic and is not counted as independent. Monte Carlo checks used 20,000 and 2,000 random permutations for the expected AP of the two S5 sets (seed 7) and 10,000 SciPy bootstrap resamples for the S13 intervals (seed 11). Appendix Table B1 lists every check group with its reference implementation, resampling settings and tolerance. The scenario specification and all tables in this paper were generated from the same results file, so no value was typed by hand.

## 2.7. Scope and article type

This study is a methodological tutorial. Its evidence consists of constructed scenarios with released inputs and verified arithmetic. Two empirical extensions were planned but not carried out. These were a literature audit of reporting practice and a demonstration with public realdata tasks. Both are specified in Appendix D so that they can be run and reported separately.

## 3. Results

The scenarios were run as specified in Section 2.3, and the values were recomputed as described in Section 2.6. All 205 checks passed (Appendix Table B1). Of these, 178 were exact independent library checks, 6 were Monte Carlo library checks, 18 were independent reimplementations and 3 were internal re-derivations. Values are proportions on the 0–1 scale unless a count or a cost is stated.

## 3.1. Confusion-matrix scenarios

The accuracy paradox appeared as constructed in S1 (Table 5).

Table 5. Scenario S1, accuracy paradox. T = 10,000 cases with 100 positive cases (prevalence 0.010); the positive class is the minority class. Counts are TP / FN / FP / TN.
<table><tr><td rowspan=1 colspan=1>Rule</td><td rowspan=1 colspan=1>Counts</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>BA</td><td rowspan=1 colspan=1>TPR</td><td rowspan=1 colspan=1>TNR</td><td rowspan=1 colspan=1>PPV</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>MCC</td></tr><tr><td rowspan=1 colspan=1>Alwaysnegative</td><td rowspan=1 colspan=1>0 / 100 / 0 /9,900</td><td rowspan=1 colspan=1>0.990</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>undefined</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>undefined</td></tr><tr><td rowspan=1 colspan=1>Screeningmodel</td><td rowspan=1 colspan=1>80 /20 /495 / 9,405</td><td rowspan=1 colspan=1>0.949</td><td rowspan=1 colspan=1>0.875</td><td rowspan=1 colspan=1>0.800</td><td rowspan=1 colspan=1>0.950</td><td rowspan=1 colspan=1>0.139</td><td rowspan=1 colspan=1>0.237</td><td rowspan=1 colspan=1>0.321</td></tr></table>

Table 5 shows that accuracy ranked the always-negative rule above the screening model (0.990 versus 0.949). Balanced accuracy reversed that order (0.500 versus 0.875), and the alwaysnegative rule had undefined precision and MCC.

The screening model detected 80 of 100 positive cases. Its precision was only 0.139, because 495 false positives arose from 9,900 negative cases. Accuracy therefore rewarded the rule that ignored the positive class, whereas the class-wise rates exposed it.

Prevalence changed every gold-coded metric of Figure 2 in S2, while sensitivity and specificity stayed at 0.800 and 0.900 (Figure 3).

![](images/e207fb9705fdbe220d11e81d8e709240bda56316abff26d0c0c6011a4ada7cf9.jpg)

![](images/ce5cbd8c6141978244555a129b2d2eb3b081e824c9ffc78f97ca10e7fed543db.jpg)  
-- PPV -- NPV -- F1 -- MCC -- κ -- Acc -- BA ---- Zero, chance-level MCC (b)  
Fig. 3. Dependence of confusion-matrix metrics on prevalence and on the true-negative count (scenarios S2 and S3a). (a) PPV, NPV, F1, MCC, κ, accuracy (Acc, dashed) and balanced accuracy (BA) against test-set prevalence (log axis) at fixed sensitivity 0.800 and specificity 0.900; lines are analytic curves and markers are the six locked prevalences 0.001, 0.01, 0.05, 0.1, 0.2 and 0.5 (T = 10,000 each); (b) F1 and MCC against the number of true negatives (TN = 0, 10, 100, 1,000 and 10,000; symmetric-log axis, linear between 0 and 10) with TP 90, FN 10 and FP 10; lines are analytic curves and markers are the locked tables, and the black dashed line marks zero (chance-level MCC). In S3a κ equals MCC to three decimals and is therefore not drawn in (b). Each metric keeps one colour in all figures (Acc grey dashed, BA blue, PPV red, NPV green, F1 gold, MCC purple, κ brown).

Figure 3 shows that precision fell from 0.889 at a prevalence of 0.5 to 0.008 at 0.001, whereas balanced accuracy stayed at 0.850. These curves follow from the fixed class-wise rates and do not describe a classifier whose behaviour changes with the class mix.

Accuracy rose from 0.850 to 0.900 as prevalence fell, because the specificity of the larger negative class exceeded the sensitivity. The F1 score fell from 0.842 to 0.016, MCC from 0.704 to 0.074 and kappa from 0.700 to 0.014. Such values can be compared across test sets only when the target population and the evaluation purpose are stated. Differences in these values cannot alone be attributed to differences between models.

True negatives were excluded from F1 in S3 but were included in MCC and kappa (Table 6). Table 6. Scenario S3: F1, MCC and the choice of positive class. (a) TN is varied at fixed TP 90, FN 10 and FP 10. (b) The same 2 × 2 table is evaluated with each class designated as positive. Counts are listed as TP / FN / FP / TN.

<table><tr><td colspan="1" rowspan="1">Setting</td><td colspan="1" rowspan="1">Counts</td><td colspan="1" rowspan="1">Acc</td><td colspan="1" rowspan="1">F1</td><td colspan="1" rowspan="1">MCC</td><td colspan="1" rowspan="1">K</td></tr><tr><td colspan="1" rowspan="1">(a) TN = 0</td><td colspan="1" rowspan="1">90 / 10 / 10 / 0</td><td colspan="1" rowspan="1">0.818</td><td colspan="1" rowspan="1">0.900</td><td colspan="1" rowspan="1">-0.100</td><td colspan="1" rowspan="1">-0.100</td></tr><tr><td colspan="1" rowspan="1">(a) TN = 10</td><td colspan="1" rowspan="1">90 / 10 / 10 / 10</td><td colspan="1" rowspan="1">0.833</td><td colspan="1" rowspan="1">0.900</td><td colspan="1" rowspan="1">0.400</td><td colspan="1" rowspan="1">0.400</td></tr><tr><td colspan="1" rowspan="1">(a) TN = 100</td><td colspan="1" rowspan="1">90 / 10 / 10 / 100</td><td colspan="1" rowspan="1">0.905</td><td colspan="1" rowspan="1">0.900</td><td colspan="1" rowspan="1">0.809</td><td colspan="1" rowspan="1">0.809</td></tr><tr><td colspan="1" rowspan="1">(a) TN = 1,000</td><td colspan="1" rowspan="1">90 / 10 / 10 / 1,000</td><td colspan="1" rowspan="1">0.982</td><td colspan="1" rowspan="1">0.900</td><td colspan="1" rowspan="1">0.890</td><td colspan="1" rowspan="1">0.890</td></tr><tr><td colspan="1" rowspan="1">(a) TN = 10,000</td><td colspan="1" rowspan="1">90 / 10 / 10 / 10,000</td><td colspan="1" rowspan="1">0.998</td><td colspan="1" rowspan="1">0.900</td><td colspan="1" rowspan="1">0.899</td><td colspan="1" rowspan="1">0.899</td></tr><tr><td colspan="1" rowspan="1">(b) positive = class 1</td><td colspan="1" rowspan="1">40 / 10 / 20 / 930</td><td colspan="1" rowspan="1">0.970</td><td colspan="1" rowspan="1">0.727</td><td colspan="1" rowspan="1">0.715</td><td colspan="1" rowspan="1">0.712</td></tr><tr><td colspan="1" rowspan="1">(b) positive = class 0</td><td colspan="1" rowspan="1">930 / 20 / 10 / 40</td><td colspan="1" rowspan="1">0.970</td><td colspan="1" rowspan="1">0.984</td><td colspan="1" rowspan="1">0.715</td><td colspan="1" rowspan="1">0.712</td></tr></table>

In Table 6, F1 remained at 0.900 as TN increased from 0 to 10,000. MCC increased from −0.100 to 0.899. After the class labels were reversed, F1 changed from 0.727 to 0.984. MCC remained at 0.715. The positive class must therefore be stated with every F1 value.

Several F1 summaries were calculated from the same three-class confusion matrix in S4 (Table 7).

Table 7. Scenario S4: alternative averages for the same three-class confusion matrix. Class supports are A 800, B 150 and C 50. Gorodkin’s R\_K is used for MCC. BA is the macro average of recall.
<table><tr><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>MicroF1</td><td rowspan=1 colspan=1>MacroF1</td><td rowspan=1 colspan=1>Weighted F1</td><td rowspan=1 colspan=1>BA</td><td rowspan=1 colspan=1>MCC</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>F1 A</td><td rowspan=1 colspan=1>F1 B</td><td rowspan=1 colspan=1>F1 C</td></tr><tr><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.707</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.694</td><td rowspan=1 colspan=1>0.665</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.944</td><td rowspan=1 colspan=1>0.733</td><td rowspan=1 colspan=1>0.444</td></tr></table>

In Table 7, micro F1 and accuracy were both 0.890. Macro F1 was 0.707. This gap was due to minority class C, with an F1 of 0.444. All summaries were obtained from the same classifier. The weighted average (0.888) stayed close to the micro average, because class A held 800 of the 1,000 cases. Balanced accuracy, the macro average of recall, was 0.694. An F1 value without its averaging rule is therefore not interpretable in multi-class work. Identical observed agreement gave different kappa values in S8 (Table 8).

Table 8. Scenario S8, kappa and marginal totals (n = 100 in each table). Counts are TP / FN / FP / TN; p\_e is the chance agreement of Eq. (6).
<table><tr><td rowspan=1 colspan=1>Table</td><td rowspan=1 colspan=1>Counts</td><td rowspan=1 colspan=1>Observed agreement</td><td rowspan=1 colspan=1>p_e</td><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>MCC</td></tr><tr><td rowspan=1 colspan=1>Balanced marginals</td><td rowspan=1 colspan=1>40 / 6 / 9 / 45</td><td rowspan=1 colspan=1>0.850</td><td rowspan=1 colspan=1>0.501</td><td rowspan=1 colspan=1>0.700</td><td rowspan=1 colspan=1>0.701</td></tr><tr><td rowspan=1 colspan=1>Skewed marginals</td><td rowspan=1 colspan=1>80 / 5 / 10 / 5</td><td rowspan=1 colspan=1>0.850</td><td rowspan=1 colspan=1>0.780</td><td rowspan=1 colspan=1>0.318</td><td rowspan=1 colspan=1>0.327</td></tr></table>

0.318. The lower value reflects the higher chance agreement implied by skewed marginal totals and does not indicate worse raters.

## 3.2. Score and probability scenarios

ROC-AUC did not change with prevalence in S5, whereas average precision did (Figure 4).

![](images/b6630b468f7b58efbdc927d8101d9323e3fbc3a757a54163da59036654d16451.jpg)

![](images/0e963d6e04485b01bbeb80bd4f41fbd292c0a6cdc49724c663731ccbdbcbdd7a.jpg)  
Fig. 4. ROC and precision–recall curves at two prevalences with identical class-conditional score distributions (scenario S5). (a) ROC curves of the balanced set $( \pi = 0 . 5 0 0 , \mathrm { n } = 2 0 0 ;$ solid blue) and the low-prevalence set $( \pi = 0 . 0 1 0 , \mathrm { n } = 1 0 , 0 0 0 ;$ dashed red), which coincide; the black dashed diagonal is the chance line; (b) step precision–recall curves of the same two sets, each with its population reference line (precision equal to the prevalence π) dashed in the matching colour; the expected AP of a random ranking in these finite sets is slightly higher (Section 3.2). Legends give ROC-AUC and average precision (AP, step-wise scikitlearn estimator). Positive scores follow N(1.5, 1) and negative scores N(0, 1), 100 of each; the low-prevalence set repeats each negative 99 times.

Figure 4 shows a ROC-AUC of 0.866 in both sets and an AP that fell from 0.884 to 0.354. The AP of the low-prevalence set lay far above its population reference of 0.010, so the ranking remained informative even though precision was low. The expected AP of a random ranking was 0.512 in the balanced set and 0.011 in the low-prevalence set, slightly above the respective prevalence.

Five operating points were evaluated from the same score vector in S6 (Table 9).

Table 9. Scenario S6: comparison of threshold rules. Thresholds were chosen on the validation set and applied to the test set (2,000 cases each, prevalence 0.10). All metrics are test-set values. Cost per 1,000 test cases uses a false-negative cost 20 times the false-positive cost.
<table><tr><td rowspan=1 colspan=1>Rule (chosen onvalidation data)</td><td rowspan=1 colspan=1>Threshold</td><td rowspan=1 colspan=1>TPR</td><td rowspan=1 colspan=1>TNR</td><td rowspan=1 colspan=1>PPV</td><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>Cost per1,000</td></tr><tr><td rowspan=1 colspan=1>Default 0.5</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=1>0.820</td><td rowspan=1 colspan=1>0.940</td><td rowspan=1 colspan=1>0.603</td><td rowspan=1 colspan=1>0.695</td><td rowspan=1 colspan=1>414.0</td></tr><tr><td rowspan=1 colspan=1>Youden</td><td rowspan=1 colspan=1>0.415</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.902</td><td rowspan=1 colspan=1>0.503</td><td rowspan=1 colspan=1>0.643</td><td rowspan=1 colspan=1>308.0</td></tr><tr><td rowspan=1 colspan=1>Maximum F1</td><td rowspan=1 colspan=1>0.568</td><td rowspan=1 colspan=1>0.780</td><td rowspan=1 colspan=1>0.963</td><td rowspan=1 colspan=1>0.700</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>473.5</td></tr><tr><td rowspan=1 colspan=1>TPR ≥ 0.95</td><td rowspan=1 colspan=1>0.280</td><td rowspan=1 colspan=1>0.950</td><td rowspan=1 colspan=1>0.807</td><td rowspan=1 colspan=1>0.353</td><td rowspan=1 colspan=1>0.515</td><td rowspan=1 colspan=1>274.0</td></tr><tr><td rowspan=1 colspan=1>Minimum cost</td><td rowspan=1 colspan=1>0.353</td><td rowspan=1 colspan=1>0.920</td><td rowspan=1 colspan=1>0.867</td><td rowspan=1 colspan=1>0.435</td><td rowspan=1 colspan=1>0.591</td><td rowspan=1 colspan=1>279.5</td></tr></table>

Table 9 shows test-set sensitivities from 0.780 to 0.950 for the same scores. The threshold that minimized cost on validation data gave a test cost of 279.5 per 1,000 cases, slightly above the 274.0 of the sensitivity-constrained rule; validation optimality is therefore not test optimality. The validation constraint of a sensitivity of at least 0.95 was met on the test set (0.950) at a precision of 0.353. The F1-optimal threshold of 0.568 gave the highest F1 (0.738) but the highest cost (473.5). A threshold is therefore a design choice that must be reported with its rule, its data split and, where used, its cost ratio.

Ranking and decisions were identical across the three probability variants of S7, whereas probability quality differed (Table 10).

Table 10. Scenario S7: discrimination and calibration. Predicted probabilities are defined as sigmoid(a · z). The same fixed logit vector z is used for all 5,000 cases. The variant a = 1 is calibrated by construction. ECE uses 10, 15 or 20 equal-width bins.

<table><tr><td rowspan=1 colspan=1>a</td><td rowspan=1 colspan=1>ROC-AUC</td><td rowspan=1 colspan=1>Acc at0.5</td><td rowspan=1 colspan=1>Brier</td><td rowspan=1 colspan=1>Log loss</td><td rowspan=1 colspan=1>ECE 10</td><td rowspan=1 colspan=1>ECE 15</td><td rowspan=1 colspan=1>ECE 20</td><td rowspan=1 colspan=1>Slope</td><td rowspan=1 colspan=1>Intercept</td></tr><tr><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.769</td><td rowspan=1 colspan=1>0.170</td><td rowspan=1 colspan=1>0.519</td><td rowspan=1 colspan=1>0.106</td><td rowspan=1 colspan=1>0.106</td><td rowspan=1 colspan=1>0.106</td><td rowspan=1 colspan=1>2.08</td><td rowspan=1 colspan=1>-0.272</td></tr><tr><td rowspan=1 colspan=1>1.0</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.769</td><td rowspan=1 colspan=1>0.157</td><td rowspan=1 colspan=1>0.477</td><td rowspan=1 colspan=1>0.013</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>1.04</td><td rowspan=1 colspan=1>0.020</td></tr><tr><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>0.825</td><td rowspan=1 colspan=1>0.769</td><td rowspan=1 colspan=1>0.168</td><td rowspan=1 colspan=1>0.547</td><td rowspan=1 colspan=1>0.094</td><td rowspan=1 colspan=1>0.094</td><td rowspan=1 colspan=1>0.095</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.479</td></tr></table>

All variants in Table 10 had a ROC-AUC of 0.825. Accuracy was also identical at 0.769 for a threshold of 0.5. Differences were captured by the Brier score, log loss, ECE and calibration slope. Probability quality therefore cannot be assessed from a discrimination metric alone. The calibrated variant had a calibration slope of 1.04 and an ECE of 0.013 with 10 bins. Its ECE rose to 0.019 with 20 bins. This change was caused solely by the number of bins. The reliability diagrams are presented in Figure 5.

(a) Underconfident, a = 0.5  
![](images/91cafa4b965edb1348406bb5cf3d0bd4da0c85bf5c61e0f1b2c9a0ccd2d4efd0.jpg)

(b) Calibrated, a = 1  
![](images/75dda4ab75ae73d7a82d4127ed4d3a5e4d688aae74aec05e53ff2f6e5c50dfdc.jpg)

(c) Overconfident, a = 2  
![](images/02720404dbcaf1fc1fcecd42bc06f4ec389ef1d8ce0d2b340eac44a82b548b63.jpg)

![](images/705fbda3ee9db6f5561969bce356bbdbfc4c7698483b31d1b963075d5a86ac9f.jpg)

![](images/29174f99aefa74eb5684f769001c105dd8edcbf7f6ff95c94f1ac0af2179c793.jpg)

![](images/96ed3d5ada82a84c0b459ae58d9b8731b0945ec46434dfe8bcfdb639a9bdc75d.jpg)  
Fig. 5. Reliability diagrams for three probability variants with the same score order (scenario S7, n = 5,000). Probabilities are p = sigmoid(a·z) for logit scale (a) a = 0.5 (underconfident, blue), (b) a = 1 (calibrated, red) and (c) a = 2 (overconfident, grey). Upper panels show the observed frequency of y = 1 against the mean predicted probability in 10 equal-width bins, with the identity line dashed black; lower strips show the number of cases per bin. Each panel

prints the Brier score, the ECE (10 equal-width bins) and the calibration slope (logistic regression of y on logit(p)). The highest bin in (a) contains 1 case.

Figure 5 shows the underconfident variant below the identity line at low probabilities and above it at high probabilities, with the reverse pattern for the overconfident variant. Reliability diagrams describe calibration on this test set and do not guarantee calibration under a different class mix.

## 3.3. Regression scenarios

MAE and RMSE ranked the two S9 models in opposite order (Table 11).

Table 11. Regression scenarios S9–S11. Model A has an absolute error of 2 for all 100 cases; model B has 0.5 for 95 cases and 30 for 5 cases. (a) MAPE over the nine non-zero targets; the percentage error of the zero target is infinite. r = Pearson correlation, undefined for a constant prediction.
<table><tr><td rowspan=1 colspan=1>Scenario</td><td rowspan=1 colspan=1>Predictor</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>MedAE</td><td rowspan=1 colspan=1>MAPE</td><td rowspan=1 colspan=1>R2</td><td rowspan=1 colspan=1>r</td></tr><tr><td rowspan=1 colspan=1>S9</td><td rowspan=1 colspan=1>Model A</td><td rowspan=1 colspan=1>2.000</td><td rowspan=1 colspan=1>2.000</td><td rowspan=1 colspan=1>2.000</td><td rowspan=1 colspan=1>0.045</td><td rowspan=1 colspan=1>0.987</td><td rowspan=1 colspan=1>0.99</td></tr><tr><td rowspan=1 colspan=1>S9</td><td rowspan=1 colspan=1>Model B</td><td rowspan=1 colspan=1>1.975</td><td rowspan=1 colspan=1>6.726</td><td rowspan=1 colspan=1>0.500</td><td rowspan=1 colspan=1>0.049</td><td rowspan=1 colspan=1>0.849</td><td rowspan=1 colspan=1>0.94</td></tr><tr><td rowspan=1 colspan=1>S10</td><td rowspan=1 colspan=1>Target + 1</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>0.432(a)</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>S11</td><td rowspan=1 colspan=1>Target + 15</td><td rowspan=1 colspan=1>15.000</td><td rowspan=1 colspan=1>15.000</td><td rowspan=1 colspan=1>15.000</td><td rowspan=1 colspan=1>0.305</td><td rowspan=1 colspan=1>-5.136</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>S11</td><td rowspan=1 colspan=1>Constant 45(training mean)</td><td rowspan=1 colspan=1>6.429</td><td rowspan=1 colspan=1>7.853</td><td rowspan=1 colspan=1>5.000</td><td rowspan=1 colspan=1>0.121</td><td rowspan=1 colspan=1>-0.682</td><td rowspan=1 colspan=1>undefined</td></tr><tr><td rowspan=1 colspan=1>S11</td><td rowspan=1 colspan=1>Constant 50 (testmean)</td><td rowspan=1 colspan=1>5.238</td><td rowspan=1 colspan=1>6.055</td><td rowspan=1 colspan=1>5.000</td><td rowspan=1 colspan=1>0.107</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>undefined</td></tr></table>

Table 11 shows a slightly lower MAE for model B (1.975 versus 2.000) and a much higher RMSE (6.726 versus 2.000). Model B’s MedAE of 0.500 hid its five large errors, so a single absolute-error summary did not describe the error distribution.

A constant absolute error of 1 produced very different percentage errors in S10. The absolute percentage error was 2.000 at a target of 0.5 and 0.005 at a target of 200. At a target of 0 it was infinite, so the reported MAPE of 0.432 required the zero target to be excluded.

A perfectly correlated but biased predictor in S11 had a Pearson correlation of 1.00 and an $R ^ { 2 }$ of −5.136. Predicting the training mean of 45 gave −0.682, and predicting the test mean gave exactly 0.000. Figure 6 shows the three regression scenarios together.

(a) Absolute-error distributions at nearly equal MAE

![](images/f5380ab722d06700aeebd665f7cc59a41655db4034083324435dea84f8c9fa6b.jpg)  
(b) APE at constant absolute error 1

![](images/f72b21009c915b06601416c0b17d2604ece7d5f679a36c5ce39ee26837ce8947.jpg)

(c) Test targets 40–60, three predictors  
![](images/b5778bd5d21eab81fe55c7fa782fc0275288bab6f499b2eee0b2b215bec59af0.jpg)  
Fig. 6. Regression error metrics under outliers, near-zero targets and biased prediction (scenarios S9–S11). (a) Absolute errors of model A (blue; 100 cases with error 2) and model B (red; 95 cases with error 0.5 and 5 cases with error 30) on a log axis; marker area scales with the number of cases, open downward triangles above each row mark MAE and filled upward triangles below each row mark RMSE, and the row labels give both values; (b) absolute percentage error, $\mathrm { A P E } = \left| \hat { \mathbf { y } } - \mathbf { y } \right| / \left| \mathbf { y } \right|$ on the 0–1 scale, against target y on log–log axes for a constant absolute error of 1 (S10); APE is infinite at y = 0 (error 1), so that target is omitted; (c) predictions against test targets 40–60 for a predictor with bias +15 (blue circles), the constant training mean 45 (red squares) and the constant test mean 50 (grey diamonds), with the identity line dashed black and R² in the legend.

Figure 6 shows that the error distribution, the target scale and the reference prediction each changed the reading of a regression metric. Its panels use constructed vectors and do not rank regression metrics in general.

## 3.4. Inference and reporting scenarios

The same accuracy carried very different uncertainty in S12 (Table 12).

Table 12. Scenario S12, 95% intervals for an accuracy of 0.950 at three test sizes. $\mathrm { C P } = \tt { e x a c t }$ Clopper–Pearson interval.
<table><tr><td colspan="1" rowspan="1">Correct / n</td><td colspan="1" rowspan="1">Acc</td><td colspan="1" rowspan="1">Wilson 95% CI</td><td colspan="1" rowspan="1">CP 95% CI</td><td colspan="1" rowspan="1">Wilson half-width</td></tr><tr><td colspan="1" rowspan="1">38 / 40</td><td colspan="1" rowspan="1">0.950</td><td colspan="1" rowspan="1">0.835–0.986</td><td colspan="1" rowspan="1">0.831–0.994</td><td colspan="1" rowspan="1">0.076</td></tr><tr><td colspan="1" rowspan="1">380 / 400</td><td colspan="1" rowspan="1">0.950</td><td colspan="1" rowspan="1">0.924–0.967</td><td colspan="1" rowspan="1">0.924–0.969</td><td colspan="1" rowspan="1">0.022</td></tr><tr><td colspan="1" rowspan="1">3,800 / 4,000</td><td colspan="1" rowspan="1">0.950</td><td colspan="1" rowspan="1">0.943–0.956</td><td colspan="1" rowspan="1">0.943–0.957</td><td colspan="1" rowspan="1">0.007</td></tr></table>

Table 12 shows a 95% Wilson interval of 0.835–0.986 for 38 of 40 correct cases and 0.943– 0.956 for 3,800 of 4,000. These intervals assume independent cases and are too narrow for repeated measurements of the same participant.

A Wilson half-width of at most 0.02 at an accuracy of 0.95 required 480 test cases, and a halfwidth of 0.01 required 1,840. Two models whose accuracies differ by a few hundredths cannot be separated on a test set of 40 cases.

The aggregation rule changed accuracy in S13, and the uncertainty depended on the resampling unit (Table 13).

Table 13. Scenario S13, accuracy at three aggregation levels (10 participants, 700 segments).

Cluster bootstrap: participants resampled with replacement, 10,000 resamples, seed 20261115, percentile intervals. The naive Wilson interval treats segments as independent and is shown for contrast only.

<table><tr><td rowspan=1 colspan=1>Summary</td><td rowspan=1 colspan=1>Estimate</td><td rowspan=1 colspan=1>95% interval</td><td rowspan=1 colspan=1>Interval method</td></tr><tr><td rowspan=1 colspan=1>Segment-level pooled accuracy</td><td rowspan=1 colspan=1>0.857</td><td rowspan=1 colspan=1>0.677-0.916</td><td rowspan=1 colspan=1>cluster bootstrap</td></tr><tr><td rowspan=1 colspan=1>Segment-level pooled accuracy</td><td rowspan=1 colspan=1>0.857</td><td rowspan=1 colspan=1>0.829–0.881</td><td rowspan=1 colspan=1>naive Wilson (segments asindependent)</td></tr><tr><td rowspan=1 colspan=1>Mean per-participant accuracy</td><td rowspan=1 colspan=1>0.691</td><td rowspan=1 colspan=1>0.556–0.825</td><td rowspan=1 colspan=1>cluster bootstrap</td></tr><tr><td rowspan=1 colspan=1>Participant-majority accuracy</td><td rowspan=1 colspan=1>0.700 (7 / 10)</td><td rowspan=1 colspan=1>0.348-0.933</td><td rowspan=1 colspan=1>exact Clopper-Pearson</td></tr></table>

Table 13 shows a pooled segment-level accuracy of 0.857, a mean per-participant accuracy of 0.691 and a participant-majority accuracy of 0.700. The cluster-bootstrap interval of the pooled accuracy (0.677–0.916) was much wider than the naive Wilson interval (0.829–0.881), which wrongly treats segments as independent.

The participant-majority accuracy of 7 of 10 had an exact interval of 0.348–0.933. Ten participants therefore cannot distinguish this result from chance-level majority decisions. Figure 7 shows both inference scenarios.

![](images/df1804918de19b80f13a5b623fbc1d9abc071d4c0f239eabd5aaed6735290051.jpg)

![](images/469f77f6982f4676a38f826dfe5d26f4079b780e66e2ea5a7971db60b3318cef.jpg)  
Fig. 7. Uncertainty of accuracy and dependence on the aggregation level (scenarios S12 and S13). (a) Half-width of the 95% Wilson interval against test-set size n (log axis, 20–10,000) at accuracy 0.950 (blue line), with the three locked cases 38/40, 380/400 and 3,800/4,000 (red

markers, half-widths printed); (b) segment accuracy within each participant against the   
participant’s number of segments (log axis; 10 participants, 700 segments); filled red markers   
are participants whose segment majority is correct and open markers those whose majority is wrong. Horizontal lines give the segment-pooled accuracy (blue), the mean per-participant accuracy (red, solid) and the participant-majority accuracy (fraction of participants with a correct majority; red, dotted); the black dashed line marks 0.5. Error bars to the right of the grey line give 95% intervals: cluster bootstrap over participants for the pooled and mean accuracies and exact Clopper–Pearson for the participant-majority accuracy (Table 13).

Figure 7 shows that interval width shrinks with the square root of the test size and that three participants with many segments dominated the pooled estimate. Its intervals assume that the 10 constructed participants are independent and say nothing about other cohorts. Reverse calculation separated possible from impossible reports in S14 (Table 14).

Table 14. Scenario S14, reverse calculation of reported triplets. Values are shown with their reported precision; the tolerance is half a unit of the last reported decimal and applies to all three values. A4 repeats A at four decimals. Pairs: (TP, TN) pairs compatible with all three values within the stated tolerance. Exact pairs are given in parentheses. Implied π: the range obtained from Eq. (17) within the tolerance box.
<table><tr><td rowspan=1 colspan=1>Case</td><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>Acc</td><td rowspan=1 colspan=1>TPR</td><td rowspan=1 colspan=1>TNR</td><td rowspan=1 colspan=1>Tolerance</td><td rowspan=1 colspan=1>Pairs(exact)</td><td rowspan=1 colspan=1>Impliedπ</td><td rowspan=1 colspan=1>Outcome</td></tr><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>900</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>±0.005</td><td rowspan=1 colspan=1>9 (1)</td><td rowspan=1 colspan=1>0.0000.200</td><td rowspan=1 colspan=1>consistent</td></tr><tr><td rowspan=1 colspan=1>A4</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>900</td><td rowspan=1 colspan=1>0.8900</td><td rowspan=1 colspan=1>0.8000</td><td rowspan=1 colspan=1>0.9000</td><td rowspan=1 colspan=1>±0.00005</td><td rowspan=1 colspan=1>1 (1)</td><td rowspan=1 colspan=1>0.099-0.101</td><td rowspan=1 colspan=1>consistent</td></tr><tr><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>900</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.92</td><td rowspan=1 colspan=1>±0.005</td><td rowspan=1 colspan=1>0 (0)</td><td rowspan=1 colspan=1>none</td><td rowspan=1 colspan=1>inconsistent</td></tr><tr><td rowspan=1 colspan=1>C1</td><td rowspan=1 colspan=1>unknown</td><td rowspan=1 colspan=1>unknown</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>±0.005</td><td rowspan=1 colspan=1>notcomputed</td><td rowspan=1 colspan=1>0.133-0.267</td><td rowspan=1 colspan=1>counts notassessable;prevalencebounded</td></tr><tr><td rowspan=1 colspan=1>C2</td><td rowspan=1 colspan=1>unknown</td><td rowspan=1 colspan=1>unknown</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>±0.005</td><td rowspan=1 colspan=1>notcomputed</td><td rowspan=1 colspan=1>0.000-1.000</td><td rowspan=1 colspan=1>counts notassessable;prevalencenot assessable</td></tr><tr><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>0.9133</td><td rowspan=1 colspan=1>0.8400</td><td rowspan=1 colspan=1>0.9400</td><td rowspan=1 colspan=1>±0.00005</td><td rowspan=1 colspan=1>0 (0)</td><td rowspan=1 colspan=1>0.266-0.268</td><td rowspan=1 colspan=1>inconsistent</td></tr><tr><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>150</td><td rowspan=1 colspan=1>0.9150</td><td rowspan=1 colspan=1>0.8400</td><td rowspan=1 colspan=1>0.9400</td><td rowspan=1 colspan=1>±0.00005</td><td rowspan=1 colspan=1>1(1)</td><td rowspan=1 colspan=1>0.249-0.251</td><td rowspan=1 colspan=1>consistent</td></tr><tr><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>170</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.900</td><td rowspan=1 colspan=1>±0.0005</td><td rowspan=1 colspan=1>1(0)</td><td rowspan=1 colspan=1>0.1340.164</td><td rowspan=1 colspan=1>consistentwithinrounding</td></tr></table>

Case B in Table 14 was impossible. Its accuracy was outside the interval bounded by sensitivity and specificity. For case D, the reported accuracy was 0.9133. This was 0.0017 below the feasible value of 0.9150. Such a discrepancy can be detected by integer enumeration. It may be missed in a visual review of three plausible values.

The consistency result depended on the number of reported decimals. Nine (TP, TN) pairs were compatible with case A at two decimals. Only one pair remained for the same values at four decimals in case A4. Case E was consistent only after allowance for numerical precision. Its sensitivity of 0.833 was compatible with 25 of 30 cases (0.8333) at three decimals. The same value written as 0.8330 would have been inconsistent.

The implied prevalence was informative only when sensitivity and specificity were far apart. It lay between 0.133 and 0.267 for case C1 and covered the whole interval from 0 to 1 for case C2. Even the consistent case A, reported to two decimals, fixed the prevalence only to the range from 0 to 0.200.

Fold averages broke the pooled identity of Eq. (1). The mean fold scores implied a prevalence of 0.2126, whereas the pooled prevalence was 0.2000. mlscorecheck found the mean fold scores consistent under its fold-wise model. It flagged them as inconsistent when they were tested as one pooled test set. A check under the wrong aggregation model therefore produced a false alarm, which Table 4 classes as not assessable.

Software options changed reported values for identical predictions in S15 (Table 15).

Table 15. Scenario S15, software conventions in scikit-learn 1.9.1 for identical predictions. Status: the mathematical value of the quantity (finite, infinite or undefined). Library values are listed in the order of the options; a value that replaces an infinite or undefined quantity is assigned by the library.

<table><tr><td rowspan=1 colspan=1>Quantity</td><td rowspan=1 colspan=1>Status</td><td rowspan=1 colspan=1>Options</td><td rowspan=1 colspan=1>Library values</td></tr><tr><td rowspan=1 colspan=1>Precision with no positivepredictions</td><td rowspan=1 colspan=1>undefined (0/0)</td><td rowspan=1 colspan=1>zero division = warn / 0 / 1 /nan</td><td rowspan=1 colspan=1>0.000 / 0.000 /1.000 / undefined</td></tr><tr><td rowspan=1 colspan=1>Macro F1 with class 2 absent</td><td rowspan=1 colspan=1>per-class F1 of class2 undefined (0/0)</td><td rowspan=1 colspan=1>labels inferred / labels fixedto {0, 1, 2}</td><td rowspan=1 colspan=1>0.667 / 0.444</td></tr><tr><td rowspan=1 colspan=1>MCC of the always-negativerule (S1)</td><td rowspan=1 colspan=1>undefined (0/0)</td><td rowspan=1 colspan=1>formula / library</td><td rowspan=1 colspan=1>undefined / 0.000</td></tr><tr><td rowspan=1 colspan=1>PR summary, S5 low-prevalence set, continuousscores</td><td rowspan=1 colspan=1>finite; estimatorchoice</td><td rowspan=1 colspan=1>AP (step) / trapezoidal area</td><td rowspan=1 colspan=1>0.354 / 0.353</td></tr><tr><td rowspan=1 colspan=1>PR summary, same set, 7distinct scores</td><td rowspan=1 colspan=1>finite; estimatorchoice</td><td rowspan=1 colspan=1>AP (step) / trapezoidal area</td><td rowspan=1 colspan=1>0.230 / 0.412</td></tr><tr><td rowspan=1 colspan=1>MAPE with one zero target(S10)</td><td rowspan=1 colspan=1>infinite (error 1,target 0)</td><td rowspan=1 colspan=1>eps convention / zero targetexcluded</td><td rowspan=1 colspan=1> $4 . 5 \times 1 0 ^ { 1 4 } / 0 . 4 3 2$ </td></tr></table>

Table 15 shows that an undefined precision was reported as 0, 1 or not a number depending on one option. An infinite percentage error at a zero target became $4 . 5 \times 1 0 ^ { 1 4 }$ under the eps convention of scikit-learn. These values are assigned by library conventions; they are neither errors of the library nor mathematical values of the metric.

The precision–recall summary depended on the estimator once scores were coarse. With continuous scores, AP and the trapezoidal area were 0.354 and 0.353. After rounding to 7 distinct values, they were 0.230 and 0.412. Linear interpolation therefore overstated the precision–recall summary in this example.

## 4. Discussion

## 4.1. Main findings

The scenarios showed that one set of predictions supports several defensible numbers. The principal findings are summarized below.

• Accuracy, precision, NPV, F1, MCC and kappa changed with prevalence at fixed sensitivity and specificity, whereas balanced accuracy, informedness and the likelihood ratios did not.

• F1 did not use true negatives and changed when the class labels were swapped, whereas MCC used all four counts and was symmetric.

• ROC-AUC and AP were unchanged by any strictly increasing transformation of the scores, because both depend only on the ranking. AP changed with prevalence, and the Brier score, log loss and ECE changed with the probability transformation.

• Thresholds chosen on validation data did not carry exact guarantees to the test set, and the F1-optimal threshold differed from the cost-optimal threshold.

• MAE, RMSE and MedAE ranked models differently when errors had heavy tails, and the percentage error was infinite at a zero target.

• The resampling unit, the reporting precision, the known class counts and the aggregation model decided how wide an interval was and whether a consistency check could be made at all.

All values were recomputed, but they remain properties of constructed inputs. The findings describe mechanisms and do not measure how often these mechanisms affect published results.

## 4.2. Interpretation and scope of the synthetic evidence

Each scenario was built to isolate one mechanism. The effect sizes were determined by the selected inputs. They should not be treated as typical values for published studies. A prevalence of 0.010 was selected for S1 and a bias of 15 units for S11. These choices made the mechanisms easy to observe.

The score-based scenarios depended on specified distributions. Binormal scores were used in S5. Negative scores were repeated in the low-prevalence set to preserve the class-conditional distributions. In real data, these distributions rarely remain fixed after a prevalence change. Case mix and measurement conditions may also vary [27]. A metric unchanged by class mix is therefore not guaranteed to transfer between datasets.

A monotone transformation of a well-calibrated logit was used in the probability scenarios. Real miscalibration may be non-monotone or class-dependent. It may also arise from dataset shift [24]. ECE in S7 depended on the equal-width bin rule. This dependence was demonstrated directly in the scenario.

Random inputs were drawn once with fixed seeds. Different values would be obtained in S5– S7 and S9 with other seeds. The direction of each effect follows from the formulas in Section 2.2. Variation across seeds was not quantified. Ten constructed participants were used in S13. The arithmetic of aggregation and bootstrap resamples was demonstrated. This scenario does not represent a validation design.

## 4.3. Metric choice follows the application goal

The sensitivity of each metric to specific changes is demonstrated by the scenarios. Metric choice must be guided by the application. For screening, missed positive cases may be the main concern. Sensitivity should then be reported at a stated threshold. For confirmation, precision should be assessed at the prevalence of the target population. Risk probabilities require calibration assessment and a proper scoring rule. These cannot be supplied by a ranking metric. Common application goals are linked to primary and complementary metrics in Table 16.

Table 16. Metric choice by application goal. The main application question is addressed by the primary metric. Additional information is provided by complementary metrics. The threshold, target-population prevalence and evaluation unit must be stated for every entry.
<table><tr><td rowspan=1 colspan=1>Application goal</td><td rowspan=1 colspan=1>Primary metric</td><td rowspan=1 colspan=1>Complementary metrics</td><td rowspan=1 colspan=1>Scenario</td></tr><tr><td rowspan=1 colspan=1>Screening, missed positivescostly</td><td rowspan=1 colspan=1>sensitivity at a pre-specified threshold</td><td rowspan=1 colspan=1>specificity, PPV at the targetprevalence, cost at the threshold</td><td rowspan=1 colspan=1>S2, S6</td></tr><tr><td rowspan=1 colspan=1>Confirmation, false alarmscostly</td><td rowspan=1 colspan=1>PPV at the targetprevalence</td><td rowspan=1 colspan=1>sensitivity, FDR, the prevalenceused</td><td rowspan=1 colspan=1>S2</td></tr><tr><td rowspan=1 colspan=1>Balanced overall decisionsunder imbalance</td><td rowspan=1 colspan=1>MCC or balancedaccuracy</td><td rowspan=1 colspan=1>sensitivity, specificity, confusionmatrix</td><td rowspan=1 colspan=1>S1, S3</td></tr><tr><td rowspan=1 colspan=1>Retrieval of rare positives</td><td rowspan=1 colspan=1>average precision withits baseline</td><td rowspan=1 colspan=1>precision at a fixed recall or at k,ROC-AUC</td><td rowspan=1 colspan=1>S5, S15</td></tr><tr><td rowspan=1 colspan=1>Ranking without a fixedthreshold</td><td rowspan=1 colspan=1>ROC-AUC</td><td rowspan=1 colspan=1>ROC curve, average precisionwhen positives are rare</td><td rowspan=1 colspan=1>S5</td></tr><tr><td rowspan=1 colspan=1>Probabilities used for decisionsor risk communication</td><td rowspan=1 colspan=1>log loss or Brier score</td><td rowspan=1 colspan=1>calibration slope and intercept,reliability diagram, ECE with itsbins</td><td rowspan=1 colspan=1>S7</td></tr><tr><td rowspan=1 colspan=1>Multi-class with importantminority classes</td><td rowspan=1 colspan=1>macro F1 or balancedaccuracy</td><td rowspan=1 colspan=1>per-class metrics, confusionmatrix</td><td rowspan=1 colspan=1>S4</td></tr><tr><td rowspan=1 colspan=1>Regression, large errors costly</td><td rowspan=1 colspan=1>RMSE</td><td rowspan=1 colspan=1>MAE, error distribution</td><td rowspan=1 colspan=1>S9</td></tr><tr><td rowspan=1 colspan=1>Regression, typical error intarget units</td><td rowspan=1 colspan=1>MAE</td><td rowspan=1 colspan=1>RMSE, MedAE</td><td rowspan=1 colspan=1>S9</td></tr><tr><td rowspan=1 colspan=1>Relative error, targets far fromzero</td><td rowspan=1 colspan=1>MAPE or sMAPE</td><td rowspan=1 colspan=1>MAE, the zero-target rule</td><td rowspan=1 colspan=1>S10</td></tr><tr><td rowspan=1 colspan=1>Agreement between raters</td><td rowspan=1 colspan=1>Cohen&#x27;s kappa</td><td rowspan=1 colspan=1>observed agreement, marginaldistribution</td><td rowspan=1 colspan=1>S8</td></tr></table>

The purpose of each primary metric is specified in Table 16. The extra information from complementary metrics is also listed. This table provides an initial basis for metric choice. An application-specific cost or utility analysis is still required [28].

## 4.4. Comparison with existing guidance

Existing guidance varies in scope and in access to inputs for repeat calculations. The present guide is compared with closely related works in Table 17.

Table 17. Scope of closely related guidance and the present guide. A descriptive comparison is provided. Superiority cannot be inferred.
<table><tr><td rowspan=1 colspan=1>Work</td><td rowspan=1 colspan=1>Scope</td><td rowspan=1 colspan=1>Examples</td><td rowspan=1 colspan=1>Tool</td><td rowspan=1 colspan=1>Consistencytesting</td></tr><tr><td rowspan=1 colspan=1>Sokolova andLapalme [5]</td><td rowspan=1 colspan=1>classification measuresand their invariances</td><td rowspan=1 colspan=1>analytical</td><td rowspan=1 colspan=1>none</td><td rowspan=1 colspan=1>no</td></tr><tr><td rowspan=1 colspan=1>Chicco andJurman [12]</td><td rowspan=1 colspan=1>binary classification;MCC, F1 and accuracy</td><td rowspan=1 colspan=1>confusion-matrix usecases</td><td rowspan=1 colspan=1>none</td><td rowspan=1 colspan=1>no</td></tr><tr><td rowspan=1 colspan=1>Hicks andcolleagues [41]</td><td rowspan=1 colspan=1>binary metrics inmedical artificialintelligence</td><td rowspan=1 colspan=1>publishedgastroenterologystudies</td><td rowspan=1 colspan=1>MediMetrics webcalculator</td><td rowspan=1 colspan=1>metrics derivedfrom reportedsubsets</td></tr><tr><td rowspan=1 colspan=1>Maier-Hein andcolleagues;Reinke andcolleagues [4,42]</td><td rowspan=1 colspan=1>image-analysisvalidation</td><td rowspan=1 colspan=1>illustrated pitfalls</td><td rowspan=1 colspan=1>recommendationframework</td><td rowspan=1 colspan=1>no</td></tr><tr><td rowspan=1 colspan=1>Fazekas andKovács; Kovácsand Fazekas [44,45]</td><td rowspan=1 colspan=1>reported performancescores</td><td rowspan=1 colspan=1>published studies</td><td rowspan=1 colspan=1>mlscorecheck</td><td rowspan=1 colspan=1>yes</td></tr><tr><td rowspan=1 colspan=1>This guide</td><td rowspan=1 colspan=1>classification,probability, regressionand inference</td><td rowspan=1 colspan=1>15 locked scenarioswith released inputs</td><td rowspan=1 colspan=1>scripts andchecklist</td><td rowspan=1 colspan=1>four outcomeclasses usingmlscorecheck</td></tr></table>

Metric explanations, calculators and consistency tests are available in separate works, as shown in Table 17. These works have different aims and cover different domains. Superiority cannot therefore be inferred from this comparison.

Fixed inputs across metric families are provided in this guide. A goal-based metric table and a rule for reports of consistency checks are also included. The tests of Fazekas and Kovács are retained as the basis for these checks [44]. Metric choice is guided by the application problem, in line with Metrics Reloaded [4].

## 4.5. Reporting guide

A short set of report items was derived from the scenarios. Table 18 links each item to the scenario that motivates it.

Table 18. Checklist of report items derived from the scenarios.
<table><tr><td colspan="1" rowspan="1">Item</td><td colspan="1" rowspan="1">What to report</td><td colspan="1" rowspan="1">Scenario</td></tr><tr><td colspan="1" rowspan="1">Test set</td><td colspan="1" rowspan="1">number of cases per class and the prevalence</td><td colspan="1" rowspan="1">S1, S2</td></tr><tr><td colspan="1" rowspan="1">Positive class</td><td colspan="1" rowspan="1">the class named positive for every binary metric</td><td colspan="1" rowspan="1">S3</td></tr><tr><td colspan="1" rowspan="1">Imbalance</td><td colspan="1" rowspan="1">balanced accuracy, sensitivity, specificity and MCC besideaccuracy</td><td colspan="1" rowspan="1">S1</td></tr><tr><td colspan="1" rowspan="1">Multi-class</td><td colspan="1" rowspan="1">the averaging rule and per-class values</td><td colspan="1" rowspan="1">S4</td></tr><tr><td colspan="1" rowspan="1">Agreement</td><td colspan="1" rowspan="1">kappa with the marginal distribution</td><td colspan="1" rowspan="1">S8</td></tr><tr><td colspan="1" rowspan="1">Threshold</td><td colspan="1" rowspan="1">the threshold, the rule that chose it, its data split and any costratio</td><td colspan="1" rowspan="1">S6</td></tr><tr><td colspan="1" rowspan="1">Ranking</td><td colspan="1" rowspan="1">ROC-AUC with average precision and its baseline</td><td colspan="1" rowspan="1">S5</td></tr><tr><td colspan="1" rowspan="1">Probabilities</td><td colspan="1" rowspan="1">a proper scoring rule, calibration slope and intercept, and theECE binning</td><td colspan="1" rowspan="1">S7</td></tr><tr><td colspan="1" rowspan="1">Regression</td><td colspan="1" rowspan="1">MAE and RMSE together, an error plot, the zero-target rule andthe R² reference</td><td colspan="1" rowspan="1">S9-S11</td></tr><tr><td colspan="1" rowspan="1">Uncertainty</td><td colspan="1" rowspan="1">n, a 95% interval with its method and the resampling unit</td><td colspan="1" rowspan="1">S12</td></tr><tr><td colspan="1" rowspan="1">Evaluation unit</td><td colspan="1" rowspan="1">segment, participant or fold level, and pooled or averagedaggregation</td><td colspan="1" rowspan="1">S13,S14</td></tr><tr><td colspan="1" rowspan="1">Recomputability</td><td colspan="1" rowspan="1">the confusion matrix or a prediction file, and the reportingprecision of each value</td><td colspan="1" rowspan="1">S14</td></tr><tr><td colspan="1" rowspan="1">Software</td><td colspan="1" rowspan="1">library, version and options such as zero_division and labels</td><td colspan="1" rowspan="1">S15</td></tr></table>

Table 18 lists the minimum information needed to interpret and recompute a reported metric.

The checklist is intended for use alongside domain standards such as TRIPOD+AI [3].

## 4.6. Advantages of the proposed guide

The main practical advantages are listed below.

• Every input is released as a file, and every value can be recomputed with independent software.

• One notation covers confusion-matrix, ranking, probability, regression and inferential metrics.

• The prevalence coding of Figure 2 shows which metrics are invariant to a change in class mix at fixed class-wise rates.

• The four outcome classes of Table 4, together with the stated reporting precision, prevent a non-assessable check from being reported as an error.

## 4.7. Limitations and future work

The main limitations are listed below.

• The scenarios are synthetic and do not estimate how often each mechanism affects published results.

• The literature audit and the real-data demonstration were not executed and are specified only in Appendix D.

• Random inputs were drawn from one seed per scenario, and variation across seeds was not quantified.

• The cluster bootstrap in S13 rests on 10 constructed participants, so its intervals are themselves imprecise.

• Multi-label classification, segmentation, object detection and generative-model evaluation were outside the scope.

• Decision-analytic measures such as net benefit were cited but not examined in a scenario [28].

• The comparison with existing guidance in Table 17 is descriptive and was not based on a systematic search.

The next evaluation should execute the audit of Appendix D.1 on 200 articles with coding form version 1.0. Its consistency rate should be reported only over assessable articles, and inter-rater agreement should be measured on a subsample coded by a second person. The real-data demonstration of Appendix D.2 should then test whether the mechanisms of S2, S5, S7 and S9 change model rankings on four public tasks.

## 4.8. Potential applications

The guide can serve three groups. Authors can use Tables 16 and 18 when they choose and report metrics. Reported values can be checked by reviewers with mlscorecheck and classified with Table 4. The fixed inputs can also be used for teaching. Students can alter one factor and recalculate all metrics.

## 5. Conclusions

A methodological tutorial on performance metrics was developed. Its inputs were released and verified. A prevalence-coded metric map, 15 controlled scenarios and a goal-based metric table are provided. A four-class framework is used for consistency checks. Established metrics and tests are retained. The new elements are the fixed inputs, verification code and rules for metric reports.

An accuracy of 0.990 was obtained for a rule with no positive predictions. Precision fell from 0.889 to 0.008 solely due to prevalence. The same ROC-AUC of 0.825 was obtained for three probability variants. Their calibration slopes ranged from 0.52 to 2.08. All 205 checks passed. Of these, 202 were independent.

Individual mechanisms were isolated by design. Their frequency in published studies was not measured. The literature audit and real-data demonstration in Appendix D are proposed as the next steps. These protocols must be completed before conclusions about published practice can be drawn. The present guide can be used for metric choice and reporting within this scope.

## Declarations

Ethical approval and informed consent: Not applicable. Only constructed data were used.   
No human participants or animals were involved.

Data availability: The locked scenario inputs, results files, verification log (verify\_checks.csv) and scenario specification are included in the archive ancillary\_files\_v2.zip. The archive is provided as ancillary files with the arXiv submission. It is listed in the anc/ directory of the arXiv record.

Code availability: The same archive contains the metric implementation, scenario builder, verification script, figure scripts, manuscript build scripts, an environment specification (environment.txt) and the run commands (README.md).

Funding: This research received no external funding.

Competing interests: The authors declare no competing interests.

CRediT authorship contribution statement: Mehmet Baygin: Conceptualization, Methodology, Software, Formal analysis, Writing – original draft. Sengul Dogan: Validation, Writing – review and editing. Turker Tuncer: Methodology, Supervision, Writing – review and editing.

## References

[1] S. Kapoor, A. Narayanan, Leakage and the reproducibility crisis in machine-learning-based science, Patterns 4 (2023) 100804. https://doi.org/10.1016/j.patter.2023.100804

[2] S. Varma, R. Simon, Bias in error estimation when using cross-validation for model selection, BMC Bioinformatics 7 (2006) 91. https://doi.org/10.1186/1471-2105-7-91

[3] G.S. Collins, K.G.M. Moons, P. Dhiman, R.D. Riley, A.L. Beam, B. Van Calster, M. Ghassemi, X. Liu, J.B. Reitsma, M. van Smeden, A.-L. Boulesteix, J.C. Camaradou, L.A. Celi, S. Denaxas, A.K. Denniston, B. Glocker, R.M. Golub, H. Harvey, G. Heinze, M.M. Hoffman, A.P. Kengne, E. Lam, N. Lee, E.W. Loder, L. Maier-Hein, B.A. Mateen, M.D. McCradden, L. Oakden-Rayner, J. Ordish, R. Parnell, S. Rose, K. Singh, L. Wynants, P. Logullo, TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods, BMJ 385 (2024) e078378. https://doi.org/10.1136/bmj-2023-078378

[4] L. Maier-Hein, A. Reinke, P. Godau, M.D. Tizabi, F. Buettner, E. Christodoulou, B. Glocker, F. Isensee, J. Kleesiek, M. Kozubek, M. Reyes, M.A. Riegler, M. Wiesenfarth, A.E. Kavur, C.H. Sudre, M. Baumgartner, M. Eisenmann, D. Heckmann-Nötzel, T. Rädsch, L. Acion, M. Antonelli, T. Arbel, S. Bakas, A. Benis, M.B. Blaschko, M.J. Cardoso, V. Cheplygina, B.A. Cimini, G.S. Collins, K. Farahani, L. Ferrer, A. Galdran, B. van Ginneken, R. Haase, D.A. Hashimoto, M.M. Hoffman, M. Huisman, P. Jannin, C.E. Kahn, D. Kainmueller, B. Kainz, A. Karargyris, A. Karthikesalingam, F. Kofler, A. Kopp-Schneider, A. Kreshuk, T. Kurc, B.A. Landman, G. Litjens, A. Madani, K. Maier-Hein, A.L. Martel, P. Mattson, E. Meijering, B. Menze, K.G.M. Moons, H. Müller, B. Nichyporuk, F. Nickel, J. Petersen, N. Rajpoot, N. Rieke, J. Saez-Rodriguez, C.I. Sánchez, S. Shetty, M. van Smeden, R.M. Summers, A.A. Taha, A. Tiulpin, S.A. Tsaftaris, B. Van Calster, G. Varoquaux, P.F. Jäger, Metrics reloaded: recommendations for image analysis validation, Nature Methods 21 (2024) 195–212. https://doi.org/10.1038/s41592-023- 02151-z

[5] M. Sokolova, G. Lapalme, A systematic analysis of performance measures for classification tasks, Information Processing & Management 45 (2009) 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

[6] C. Ferri, J. Hernández-Orallo, R. Modroiu, An experimental comparison of performance measures for classification, Pattern Recognition Letters 30 (2009) 27–38. https://doi.org/10.1016/j.patrec.2008.08.010

[7] A. Tharwat, Classification assessment methods, Applied Computing and Informatics 17 (2021) 168–192. https://doi.org/10.1016/j.aci.2018.08.003

[8] D.M.W. Powers, Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation, arXiv preprint (2020) arXiv:2010.16061. https://doi.org/10.48550/arXiv.2010.16061

[9] A. Luque, A. Carrasco, A. Martín, A. de las Heras, The impact of class imbalance in classification performance metrics based on the binary confusion matrix, Pattern Recognition 91 (2019) 216–231. https://doi.org/10.1016/j.patcog.2019.02.023

[10] K.H. Brodersen, C.S. Ong, K.E. Stephan, J.M. Buhmann, The Balanced Accuracy and Its Posterior Distribution, in: 2010 20th International Conference on Pattern Recognition, 2010, pp. 3121–3124. https://doi.org/10.1109/icpr.2010.764

[11] B.W. Matthews, Comparison of the predicted and observed secondary structure of T4 phage lysozyme, Biochimica et Biophysica Acta (BBA) - Protein Structure 405 (1975) 442–451. https://doi.org/10.1016/0005-2795(75)90109-9

[12] D. Chicco, G. Jurman, The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation, BMC Genomics 21 (2020) 6. https://doi.org/10.1186/s12864-019-6413-7

[13] J. Gorodkin, Comparing two K-category assignments by a K-category correlation coefficient, Computational Biology and Chemistry 28 (2004) 367–374. https://doi.org/10.1016/j.compbiolchem.2004.09.006

[14] W.J. Youden, Index for rating diagnostic tests, Cancer 3 (1950) 32–35. https://doi.org/10.1002/1097-0142(1950)3:1<32::aid-cncr2820030106>3.0.co;2-3

[15] J. Cohen, A Coefficient of Agreement for Nominal Scales, Educational and Psychological Measurement 20 (1960) 37–46. https://doi.org/10.1177/001316446002000104

[16] A.R. Feinstein, D.V. Cicchetti, High agreement but low Kappa: I. the problems of two paradoxes, Journal of Clinical Epidemiology 43 (1990) 543–549. https://doi.org/10.1016/0895-4356(90)90158-l

[17] T. Fawcett, An introduction to ROC analysis, Pattern Recognition Letters 27 (2006) 861– 874. https://doi.org/10.1016/j.patrec.2005.10.010

[18] J.A. Hanley, B.J. McNeil, The meaning and use of the area under a receiver operating characteristic (ROC) curve., Radiology 143 (1982) 29–36. https://doi.org/10.1148/radiology.143.1.7063747

[19] D.J. Hand, Measuring classifier performance: a coherent alternative to the area under the ROC curve, Machine Learning 77 (2009) 103–123. https://doi.org/10.1007/s10994-009- 5119-5

[20] T. Saito, M. Rehmsmeier, The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets, PLOS ONE 10 (2015) e0118432. https://doi.org/10.1371/journal.pone.0118432

[21] J. Davis, M. Goadrich, The relationship between Precision-Recall and ROC curves, in: Proceedings of the 23rd international conference on Machine learning - ICML '06, 2006, pp. 233–240. https://doi.org/10.1145/1143844.1143874

[22] G.W. Brier, Verification of forecasts expressed in terms of probability, Monthly Weather Review 78 (1950) 1–3. https://doi.org/10.1175/1520- 0493(1950)078<0001:vofeit>2.0.co;2

[23] T. Gneiting, A.E. Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation, Journal of the American Statistical Association 102 (2007) 359–378. https://doi.org/10.1198/016214506000001437

[24] C. Guo, G. Pleiss, Y. Sun, K.Q. Weinberger, On calibration of modern neural networks, in: Proceedings of the 34th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 70, PMLR, 2017, pp. 1321–1330. https://proceedings.mlr.press/v70/guo17a.html

[25] M. Pakdaman Naeini, G. Cooper, M. Hauskrecht, Obtaining Well Calibrated Probabilities Using Bayesian Binning, Proceedings of the AAAI Conference on Artificial Intelligence 29 (2015). https://doi.org/10.1609/aaai.v29i1.9602

[26] B. Van Calster, D.J. McLernon, M. van Smeden, L. Wynants, E.W. Steyerberg, Calibration: the Achilles heel of predictive analytics, BMC Medicine 17 (2019) 230. https://doi.org/10.1186/s12916-019-1466-7

[27] E.W. Steyerberg, A.J. Vickers, N.R. Cook, T. Gerds, M. Gonen, N. Obuchowski, M.J. Pencina, M.W. Kattan, Assessing the Performance of Prediction Models, Epidemiology 21 (2010) 128–138. https://doi.org/10.1097/ede.0b013e3181c30fb2

[28] A.J. Vickers, E.B. Elkin, Decision Curve Analysis: A Novel Method for Evaluating Prediction Models, Medical Decision Making 26 (2006) 565–574. https://doi.org/10.1177/0272989x06295361

[29] C. Willmott, K. Matsuura, Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance, Climate Research 30 (2005) 79–82. https://doi.org/10.3354/cr030079

[30] T. Chai, R.R. Draxler, Root mean square error (RMSE) or mean absolute error (MAE)? – Arguments against avoiding RMSE in the literature, Geoscientific Model Development 7 (2014) 1247–1250. https://doi.org/10.5194/gmd-7-1247-2014

[31] R.J. Hyndman, A.B. Koehler, Another look at measures of forecast accuracy, International Journal of Forecasting 22 (2006) 679–688. https://doi.org/10.1016/j.ijforecast.2006.03.001

[32] D. Chicco, M.J. Warrens, G. Jurman, The coefficient of determination R-squared is more informative than SMAPE, MAE, MAPE, MSE and RMSE in regression analysis evaluation, PeerJ Computer Science 7 (2021) e623. https://doi.org/10.7717/peerj-cs.623

[33] E.B. Wilson, Probable Inference, the Law of Succession, and Statistical Inference, Journal of the American Statistical Association 22 (1927) 209–212. https://doi.org/10.1080/01621459.1927.10502953

[34] C.J. Clopper, E.S. Pearson, The use of confidence or fiducial limits illustrated in the case of the binomial, Biometrika 26 (1934) 404–413. https://doi.org/10.1093/biomet/26.4.404

[35] L.D. Brown, T.T. Cai, A. DasGupta, Interval Estimation for a Binomial Proportion, Statistical Science 16 (2001) 101–133. https://doi.org/10.1214/ss/1009213286

[36] Q. McNemar, Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages, Psychometrika 12 (1947) 153–157. https://doi.org/10.1007/bf02295996

[37] E.R. DeLong, D.M. DeLong, D.L. Clarke-Pearson, Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach, Biometrics 44 (1988) 837. https://doi.org/10.2307/2531595

[38] T.G. Dietterich, Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms, Neural Computation 10 (1998) 1895–1923. https://doi.org/10.1162/089976698300017197

[39] B. Efron, R.J. Tibshirani, An Introduction to the Bootstrap, Chapman and Hall/CRC, 1994. https://doi.org/10.1201/9780429246593

[40] C.A. Field, A.H. Welsh, Bootstrapping Clustered Data, Journal of the Royal Statistical Society Series B: Statistical Methodology 69 (2007) 369–390. https://doi.org/10.1111/j.1467-9868.2007.00593.x

[41] S.A. Hicks, I. Strümke, V. Thambawita, M. Hammou, M.A. Riegler, P. Halvorsen, S. Parasa, On evaluation metrics for medical applications of artificial intelligence, Scientific Reports 12 (2022) 5979. https://doi.org/10.1038/s41598-022-09954-8

[42] A. Reinke, M.D. Tizabi, M. Baumgartner, M. Eisenmann, D. Heckmann-Nötzel, A.E. Kavur, T. Rädsch, C.H. Sudre, L. Acion, M. Antonelli, T. Arbel, S. Bakas, A. Benis, F. Buettner, M.J. Cardoso, V. Cheplygina, J. Chen, E. Christodoulou, B.A. Cimini, K. Farahani, L. Ferrer, A. Galdran, B. van Ginneken, B. Glocker, P. Godau, D.A. Hashimoto, M.M. Hoffman, M. Huisman, F. Isensee, P. Jannin, C.E. Kahn, D. Kainmueller, B. Kainz, A. Karargyris, J. Kleesiek, F. Kofler, T. Kooi, A. Kopp-Schneider, M. Kozubek, A. Kreshuk, T. Kurc, B.A. Landman, G. Litjens, A. Madani, K. Maier-Hein, A.L. Martel, E. Meijering, B. Menze, K.G.M. Moons, H. Müller, B. Nichyporuk, F. Nickel, J. Petersen, S.M. Rafelski, N. Rajpoot, M. Reyes, M.A. Riegler, N. Rieke, J. Saez-Rodriguez, C.I. Sánchez, S. Shetty, R.M. Summers, A.A. Taha, A. Tiulpin, S.A. Tsaftaris, B. Van Calster, G. Varoquaux, Z.R. Yaniv, P.F. Jäger, L. Maier-Hein, Understanding metric-related pitfalls in image analysis validation, Nature Methods 21 (2024) 182–194. https://doi.org/10.1038/s41592-023-02150-0

[43] N.J.L. Brown, J.A.J. Heathers, The GRIM Test, Social Psychological and Personality Science 8 (2017) 363–369. https://doi.org/10.1177/1948550616673876

[44] A. Fazekas, G. Kovács, Testing the consistency of performance scores reported for binary classification problems, Applied Soft Computing 164 (2024) 111993. https://doi.org/10.1016/j.asoc.2024.111993

[45] G. Kovács, A. Fazekas, mlscorecheck: Testing the consistency of reported performance scores and experiments in machine learning, Neurocomputing 583 (2024) 127556. https://doi.org/10.1016/j.neucom.2024.127556

[46] L. Buitinck, G. Louppe, M. Blondel, F. Pedregosa, A. Mueller, O. Grisel, V. Niculae, P. Prettenhofer, A. Gramfort, J. Grobler, R. Layton, J. Vanderplas, A. Joly, B. Holt, G. Varoquaux, API design for machine learning software: experiences from the scikit-learn project, arXiv preprint (2013) arXiv:1309.0238. https://doi.org/10.48550/arXiv.1309.0238

[47] S. Seabold, J. Perktold, Statsmodels: Econometric and Statistical Modeling with Python, in: Proceedings of the Python in Science Conference, 2010, pp. 92–96. https://doi.org/10.25080/majora-92bf1922-011

[48] P. Virtanen, R. Gommers, T.E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, E. Burovski, P. Peterson, W. Weckesser, J. Bright, S.J. van der Walt, M. Brett, J. Wilson, K.J. Millman, N. Mayorov, A.R.J. Nelson, E. Jones, R. Kern, E. Larson, C.J. Carey, İ. Polat, Y. Feng, E.W. Moore, J. VanderPlas, D. Laxalde, J. Perktold, R. Cimrman, I. Henriksen, E.A. Quintero, C.R. Harris, A.M. Archibald, A.H. Ribeiro, F. Pedregosa, P. van Mulbregt, A. Vijaykumar, A.P. Bardelli, A. Rothberg, A. Hilboll, A. Kloeckner, A. Scopatz, A. Lee, A. Rokem, C.N. Woods, C. Fulton, C. Masson, C. Häggström, C. Fitzgerald, D.A. Nicholson, D.R. Hagen, D.V. Pasechnik, E. Olivetti, E. Martin, E. Wieser, F. Silva, F. Lenders, F. Wilhelm, G. Young, G.A. Price, G.-L. Ingold, G.E. Allen, G.R. Lee, H. Audren, I. Probst, J.P. Dietrich, J. Silterra, J.T. Webber, J. Slavič, J. Nothman, J. Buchner, J. Kulick, J.L. Schönberger, J.V. de Miranda Cardoso, J. Reimer, J. Harrington, J.L.C. Rodríguez, J. Nunez-Iglesias, J. Kuczynski, K. Tritz, M. Thoma, M. Newville, M. Kümmerer, M. Bolingbroke, M. Tartre, M. Pak, N.J. Smith, N. Nowaczyk, N. Shebanov, O. Pavlyk, P.A. Brodtkorb, P. Lee, R.T. McGibbon, R. Feldbauer, S. Lewis, S. Tygier, S. Sievert, S. Vigna, S. Peterson, S. More, T. Pudlik, T. Oshima, T.J. Pingel, T.P. Robitaille, T. Spura, T.R. Jones, T. Cera, T. Leslie, T. Zito, T. Krauss, U. Upadhyay, Y.O. Halchenko, Y. Vázquez-Baeza, SciPy 1.0: fundamental algorithms for scientific computing in Python, Nature Methods 17 (2020) 261–272. https://doi.org/10.1038/s41592-019-0686-2

[49] M.J. Page, J.E. McKenzie, P.M. Bossuyt, I. Boutron, T.C. Hoffmann, C.D. Mulrow, L. Shamseer, J.M. Tetzlaff, E.A. Akl, S.E. Brennan, R. Chou, J. Glanville, J.M. Grimshaw, A. Hróbjartsson, M.M. Lalu, T. Li, E.W. Loder, E. Mayo-Wilson, S. McDonald, L.A. McGuinness, L.A. Stewart, J. Thomas, A.C. Tricco, V.A. Welch, P. Whiting, D. Moher, The PRISMA 2020 statement: an updated guideline for reporting systematic reviews, BMJ (2021) n71. https://doi.org/10.1136/bmj.n71

## Appendix A. Locked scenario inputs

This appendix lists the locked input files of the scenarios. Appendix Table A1 gives each file with its number of rows and its MD5 checksum, so that any copy can be checked against the version used here.

Table A1. Locked scenario inputs. The header line is excluded from row counts. Seed: random seed of generated inputs; fixed values are typed into the builder script.
<table><tr><td rowspan=1 colspan=1>File</td><td rowspan=1 colspan=1>Rows</td><td rowspan=1 colspan=1>MD5 checksum</td><td rowspan=1 colspan=1>Seed</td></tr><tr><td rowspan=1 colspan=1>S4 confusion matrix.csv</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>271c0bdba80ff6127d8cb044aaf57b2c</td><td rowspan=1 colspan=1>fixed values</td></tr><tr><td rowspan=1 colspan=1>S5 scores balanced.csv</td><td rowspan=1 colspan=1>200</td><td rowspan=1 colspan=1>6ec7bf6e50b36178e315d65bdbcd793a</td><td rowspan=1 colspan=1>20261102</td></tr><tr><td rowspan=1 colspan=1>S5_scores_prev001.csv</td><td rowspan=1 colspan=1>10,000</td><td rowspan=1 colspan=1>1f983dbe4449d9f635a99091e24f477f</td><td rowspan=1 colspan=1>20261102</td></tr><tr><td rowspan=1 colspan=1>S6_test.csv</td><td rowspan=1 colspan=1>2,000</td><td rowspan=1 colspan=1>fb1828efe6975430db94f796a7ca53ff</td><td rowspan=1 colspan=1>20261108</td></tr><tr><td rowspan=1 colspan=1>S6 validation.csv</td><td rowspan=1 colspan=1>2,000</td><td rowspan=1 colspan=1>b21f4b0d3662a0e46e44a308920a874d</td><td rowspan=1 colspan=1>20261108</td></tr><tr><td rowspan=1 colspan=1>S7_logits.csv</td><td rowspan=1 colspan=1>5,000</td><td rowspan=1 colspan=1>c52c4aca050cb901b81eaf25ca56f0b9</td><td rowspan=1 colspan=1>20261109</td></tr><tr><td rowspan=1 colspan=1>S9 regression.csv</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>f492c4e8559764446cea77cebb2a2ebe</td><td rowspan=1 colspan=1>20261111</td></tr><tr><td rowspan=1 colspan=1>S10 near zero.csv</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>dec629f50d9cb178d007dcdb78c02a0c</td><td rowspan=1 colspan=1>fixed values</td></tr><tr><td rowspan=1 colspan=1>S11_negative_r2.csv</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>87929f6a37db4b9a525ddf5d2588b745</td><td rowspan=1 colspan=1>fixed values</td></tr><tr><td rowspan=1 colspan=1>S13 participants.csv</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>8bef723eab1d8a55ef021f85548d2e00</td><td rowspan=1 colspan=1>fixed values</td></tr><tr><td rowspan=1 colspan=1>S15 coarse scores.csv</td><td rowspan=1 colspan=1>10,000</td><td rowspan=1 colspan=1>3b40ef335adb5d1b7709b73183e6bfdb</td><td rowspan=1 colspan=1>fixed values</td></tr></table>

Appendix Table A1 fixes the inputs behind every number in Sections 3.1–3.4. A file whose checksum differs from the listed value was not used in this study.

## Appendix B. Verification summary

Independent verification is summarized in this appendix. The number of checks and the independent implementation for each scenario group are listed in Appendix Table B1.

Table B1. Verification checks by scenario. Kind defines the verification method. Independent library: a value is recomputed with a function from another library. Monte Carlo: a library function is applied to random resamples or permutations. Results are compared within a Monte Carlo tolerance. Independent re-implementation: the formula is coded separately. Internal: the same calculation is repeated from the input file. All checks passed.

<table><tr><td colspan="1" rowspan="1">Scenario</td><td colspan="1" rowspan="1">Checks</td><td colspan="1" rowspan="1">Kind</td><td colspan="1" rowspan="1">Reference implementation</td><td colspan="1" rowspan="1">Tolerance</td></tr><tr><td colspan="1" rowspan="1">S1</td><td colspan="1" rowspan="1">12</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S2</td><td colspan="1" rowspan="1">42</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S3</td><td colspan="1" rowspan="1">49</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S4</td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S5</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S5</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">independent library(Monte Carlo)</td><td colspan="1" rowspan="1">scikit-learn average_precision_scoreover random permutations (20,000balanced, 2,000 low-prevalence; seed 7)</td><td colspan="1" rowspan="1">0.002</td></tr><tr><td colspan="1" rowspan="1">S5</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">independent library(Monte Carlo)</td><td colspan="1" rowspan="1">scikit-learn average_precision_scoreover random permutations (20,000balanced, 2,000 low-prevalence; seed 7)</td><td colspan="1" rowspan="1">0.0005</td></tr><tr><td colspan="1" rowspan="1">S5</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">independent re-implementation</td><td colspan="1" rowspan="1">enumeration of both rankings</td><td colspan="1" rowspan="1"> $1 0 ^ { - 1 2 }$ </td></tr><tr><td colspan="1" rowspan="1">S6</td><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S6</td><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">independent re-implementation</td><td colspan="1" rowspan="1">scikit-learn confusion_matrix counts +cost formula</td><td colspan="1" rowspan="1"> $1 0 ^ { - 9 }$ </td></tr><tr><td colspan="1" rowspan="1">S7</td><td colspan="1" rowspan="1">9</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S7</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn LogisticRegression</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 4 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S7</td><td colspan="1" rowspan="1">9</td><td colspan="1" rowspan="1">independent re-implementation</td><td colspan="1" rowspan="1">scikit-learn calibration curve binstatistics + bin weighting</td><td colspan="1" rowspan="1"> $1 0 ^ { - 6 }$ </td></tr><tr><td colspan="1" rowspan="1">S7</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">independent re-implementation</td><td colspan="1" rowspan="1">SciPy optimizer on separately codedlikelihood</td><td colspan="1" rowspan="1"> $1 0 ^ { - 5 }$ </td></tr><tr><td colspan="1" rowspan="1">S8</td><td colspan="1" rowspan="1">14</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S9</td><td colspan="1" rowspan="1">8</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S10</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S11</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S12</td><td colspan="1" rowspan="1">6</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">statsmodels proportion confint</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S13</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">statsmodels proportion confint</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S13</td><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">independent library(Monte Carlo)</td><td colspan="1" rowspan="1">SciPy bootstrap over participants(percentile; 10,000 resamples; seed 11)</td><td colspan="1" rowspan="1">0.02</td></tr><tr><td colspan="1" rowspan="1">S13</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">internal</td><td colspan="1" rowspan="1">pandas re-derivation from the input file</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S14</td><td colspan="1" rowspan="1">7</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">mlscorecheck</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 9 } } }$ </td></tr><tr><td colspan="1" rowspan="1">S15</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">independent library</td><td colspan="1" rowspan="1">scikit-learn average precision score</td><td colspan="1" rowspan="1"> $\overline { { 1 0 ^ { - 1 2 } } }$ </td></tr></table>

All checks passed, as summarized in Appendix Table B1. The scenario calculations were confirmed by agreement with independent libraries. The scenario design itself was not validated by this agreement.

## Appendix C. Generating models and fold counts

The random inputs and fold example can be regenerated from the details in this appendix without the code archive. The model, seed and size for each random input are listed in Appendix Table C1.

Table C1. Models used to generate random scenario inputs. $\mathrm { N } ( \mu , \sigma ^ { 2 } )$ = normal distribution; U(a, b) = uniform distribution; sigmoid $\mathrm { ( z ) = 1 / \left( 1 + e ^ { - z } \right) }$ . Inputs were drawn once with the NumPy default\_rng generator and saved. The saved inputs are reused without regeneration.
<table><tr><td rowspan=1 colspan=1>Scenario</td><td rowspan=1 colspan=1>Seed</td><td rowspan=1 colspan=1>Generating model</td><td rowspan=1 colspan=1>Size</td></tr><tr><td rowspan=1 colspan=1>S5</td><td rowspan=1 colspan=1>20261102</td><td rowspan=1 colspan=1>positive scores N(1.5, 1); negative scores N(0, 1); theprevalence-0.010 set repeats each negative score 99 times</td><td rowspan=1 colspan=1>100 + 100; 100+ 9,900</td></tr><tr><td rowspan=1 colspan=1>S6</td><td rowspan=1 colspan=1>20261108</td><td rowspan=1 colspan=1>logit z: positives N(1.0, 1.22), negatives N(-2.0, 1.22); score =sigmoid(z); validation set drawn first, then test set from thesame generator</td><td rowspan=1 colspan=1>2 × 2,000 (200positive each)</td></tr><tr><td rowspan=1 colspan=1>S7</td><td rowspan=1 colspan=1>20261109</td><td rowspan=1 colspan=1>logit z ~ N(−1, 1.52); label y ~ Bernoulli(sigmoid(z)), sosigmoid(z) is the true event probability; variants sigmoid(a ·z)</td><td rowspan=1 colspan=1>5,000</td></tr><tr><td rowspan=1 colspan=1>S9</td><td rowspan=1 colspan=1>20261111</td><td rowspan=1 colspan=1>targets U(20, 80) rounded to 0.1; error signs alternate; modelB has errors of 30 at indices 7, 23, 46, 68 and 91 (0-based) and0.5 elsewhere</td><td rowspan=1 colspan=1>100</td></tr><tr><td rowspan=1 colspan=1>S13</td><td rowspan=1 colspan=1>20261115</td><td rowspan=1 colspan=1>cluster bootstrap: 10 participants resampled with replacement</td><td rowspan=1 colspan=1>10,000resamples</td></tr></table>

The same generator, seed and order of random draws must be used to reproduce the released inputs.

Counts and fold-wise scores for all five folds of S14 are listed in Appendix Table C2.

TN. Each fold contains 100 cases.
<table><tr><td colspan="1" rowspan="1">Fold</td><td colspan="1" rowspan="1">Counts</td><td colspan="1" rowspan="1">P</td><td colspan="1" rowspan="1">N</td><td colspan="1" rowspan="1"> $\operatorname { A c c }$ </td><td colspan="1" rowspan="1">TPR</td><td colspan="1" rowspan="1">TNR</td></tr><tr><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">18 / 2 / 10 / 70</td><td colspan="1" rowspan="1">20</td><td colspan="1" rowspan="1">80</td><td colspan="1" rowspan="1">0.880</td><td colspan="1" rowspan="1">0.900</td><td colspan="1" rowspan="1">0.875</td></tr><tr><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">9 / 1 / 6 / 84</td><td colspan="1" rowspan="1">10</td><td colspan="1" rowspan="1">90</td><td colspan="1" rowspan="1">0.930</td><td colspan="1" rowspan="1">0.900</td><td colspan="1" rowspan="1">0.933</td></tr><tr><td colspan="1" rowspan="1">Fold</td><td colspan="1" rowspan="1">Counts</td><td colspan="1" rowspan="1">P</td><td colspan="1" rowspan="1">N</td><td colspan="1" rowspan="1">Acc</td><td colspan="1" rowspan="1">TPR</td><td colspan="1" rowspan="1">TNR</td></tr><tr><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">25 / 5 / 7 / 63</td><td colspan="1" rowspan="1">30</td><td colspan="1" rowspan="1">70</td><td colspan="1" rowspan="1">0.880</td><td colspan="1" rowspan="1">0.833</td><td colspan="1" rowspan="1">0.900</td></tr><tr><td colspan="1" rowspan="1">4</td><td colspan="1" rowspan="1">12 / 3 / 9 / 76</td><td colspan="1" rowspan="1">15</td><td colspan="1" rowspan="1">85</td><td colspan="1" rowspan="1">0.880</td><td colspan="1" rowspan="1">0.800</td><td colspan="1" rowspan="1">0.894</td></tr><tr><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">20 / 5 / 8 / 67</td><td colspan="1" rowspan="1">25</td><td colspan="1" rowspan="1">75</td><td colspan="1" rowspan="1">0.870</td><td colspan="1" rowspan="1">0.800</td><td colspan="1" rowspan="1">0.893</td></tr><tr><td colspan="1" rowspan="1">Pooled</td><td colspan="1" rowspan="1">84 / 16 / 40 / 360</td><td colspan="1" rowspan="1">100</td><td colspan="1" rowspan="1">400</td><td colspan="1" rowspan="1">0.888</td><td colspan="1" rowspan="1">0.840</td><td colspan="1" rowspan="1">0.900</td></tr><tr><td colspan="1" rowspan="1">Mean offolds</td><td colspan="1" rowspan="1">not applicable</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">0.888</td><td colspan="1" rowspan="1">0.847</td><td colspan="1" rowspan="1">0.899</td></tr></table>

Equal-sized folds with different prevalence are compared in Appendix Table C2. Fold means differ from pooled sensitivity and specificity. Mean and pooled accuracy agree because the same number of cases is included in every fold.

## Appendix D. Pre-specified protocols for empirical extension

Two empirical extensions are specified in this appendix. Neither was carried out in the present study. Version 1.0 of both protocols is frozen with this paper; any later change is to be reported as a deviation.

## D.1. Literature audit

The planned audit will estimate the frequency of reports with the information required in Sections 3.1–3.4.

• Sources: OpenAlex works of type article published from 1 January 2022 to 31 December 2025 with open full text, checked against Crossref.

• Classification query: title and abstract search for (“machine learning” OR “deep learning”) AND (classification OR classifier) AND (accuracy OR sensitivity OR F1 OR AUC).

• Regression query: title and abstract search for (“machine learning” OR “deep learning”) AND regression AND (RMSE OR “mean absolute error” OR “R2”).

• Strata and sample: 150 classification articles (37 or 38 per publication year) and 50 regression articles (12 or 13 per year), drawn with the OpenAlex sample parameter and seed 20261102 within each stratum.

• Exclusions: reviews, editorials and protocols; segmentation, detection and generativemodel studies; studies without a numerical test-set result; and duplicates. Excluded records are replaced by the next sampled record of the same stratum.

• Result selection: the headline result of the model proposed by the authors on the first heldout test set; when no held-out set exists, the first cross-validation result reported in the results section.

• Pilot: the first 20 articles test coding form version 1.0 (Appendix Table D1); any change creates version 1.1 before the main sample is coded.

• Reliability: a second person codes a random 20% of articles independently, and inter-rater agreement is reported with Cohen’s kappa. The first coder recodes another random 20%

after at least two weeks, and this intra-rater agreement is reported separately and labelled as such.

• Consistency checks: every check is run with mlscorecheck under the model stated by the article and classified with Table 4. Each flagged inconsistency is reviewed again.

• Reporting: proportions are reported with Wilson intervals. Only assessable articles are included in the denominator of each consistency rate. Relevant items from PRISMA 2020 are applied [49].

Table D1. Coding form version 1.0 for the pre-specified literature audit. Each field is assigned one of the listed values. Separate codes are used for not reported, not applicable, not assessable and inconsistent.

<table><tr><td rowspan=1 colspan=1>Field</td><td rowspan=1 colspan=1>Values</td></tr><tr><td rowspan=1 colspan=1>Task type</td><td rowspan=1 colspan=1>binary; multi-class; regression</td></tr><tr><td rowspan=1 colspan=1>Test-set size and class distribution</td><td rowspan=1 colspan=1>counts; not reported</td></tr><tr><td rowspan=1 colspan=1>Metrics reported</td><td rowspan=1 colspan=1>list of metric names</td></tr><tr><td rowspan=1 colspan=1>Averaging rule</td><td rowspan=1 colspan=1>micro; macro; weighted; not reported; not applicable</td></tr><tr><td rowspan=1 colspan=1>Positive class named</td><td rowspan=1 colspan=1>yes; no; not applicable</td></tr><tr><td rowspan=1 colspan=1>Threshold and its selection rule</td><td rowspan=1 colspan=1>value and rule; not reported; not applicable</td></tr><tr><td rowspan=1 colspan=1>Evaluation unit</td><td rowspan=1 colspan=1>instance; participant; fold mean; not reported</td></tr><tr><td rowspan=1 colspan=1>Confusion matrix or prediction file</td><td rowspan=1 colspan=1>available; not available</td></tr><tr><td rowspan=1 colspan=1>Uncertainty</td><td rowspan=1 colspan=1>interval method and resampling unit; not reported</td></tr><tr><td rowspan=1 colspan=1>Reporting precision</td><td rowspan=1 colspan=1>number of decimals per metric</td></tr><tr><td rowspan=1 colspan=1>Consistency check</td><td rowspan=1 colspan=1>consistent; consistent within rounding; inconsistent; notassessable</td></tr></table>

## D.2. Real-data demonstration

The planned demonstration will test whether model order on real data is altered by the scenario mechanisms.

• Tasks: four public-data tasks are selected. These comprise moderately balanced and strongly imbalanced binary tasks, a multi-class image task, and a regression task with targets near zero.

• Models: at least six models are evaluated per task. A convolutional network is included for the image task.

• Design: thresholds and calibration are tuned only on validation data. All metrics are computed from the same test predictions.

• Analysis: model orders from each metric are compared with Kendall’s $\tau _ { b }$ . A bootstrap distribution is obtained over test cases. Low agreement is interpreted in relation to a predefined application goal from Table 16. It is not treated as evidence of a failed metric.