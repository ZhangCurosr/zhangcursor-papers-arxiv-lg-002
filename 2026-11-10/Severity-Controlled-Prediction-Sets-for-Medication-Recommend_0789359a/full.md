# Severity-Controlled Prediction Sets for Medication Recommendation

Yu Gu<sup>1∗</sup> Zijun Yu<sup>1∗</sup> Chi-Kuang Yeh<sup>2</sup> Xinyu Wang<sup>1</sup> Ziyang Song<sup>3†</sup> <sup>1</sup>McGill University, <sup>2</sup>Georgia State University, <sup>3</sup>Ohio University

## Abstract

Medication recommendation is a multi-label prediction task. However, existing prediction-set evaluation and calibration methods rely mainly on exact-overlap metrics and therefore treat all missed medications as equally severe, overlooking how closely the prediction set matches them within the medication hierarchy. To address this, we introduce Conformal Severity Control (CSC), a post-hoc framework for calibrating medication prediction sets under hierarchy-aware mismatch severity. CSC uses the Anatomical Therapeutic Chemical (ATC) hierarchy to quantify mismatch severity and controls patient-level severity without retraining the predictor, supporting both average severity and upper-quantile severity for the highest-severity mismatches. Experiments on MIMIC-IV show that CSC tracks user-specified severity tolerances and shifts prediction sets away from distant mismatches toward exact or close matches.

## 1 Introduction

![](images/e68a253eb1385f508c0f346dbb15ebc0b325e69e1335730a33d63a3697880cb9.jpg)  
Figure 1: Motivation for Conformal Severity Control (CSC). Prediction sets can achieve the same recall while differing substantially in ATC mismatch severity. CSC selects a score threshold on calibration data so that the expected severity stays below a user-specified tolerance.

Modern medication recommendation systems increasingly use deep learning models and LLM-based predictors to recommend drugs from electronic health records (EHRs) [Ali et al., 2023, Lv et al., 2026, Wang et al., 2026]. These models assign scores over large drug vocabularies, which can be used to form candidate medication sets because multiple drugs may be plausible for the same patient [Kompa et al., 2021, Begoli et al., 2019]. However, the evaluation and calibration of these sets largely rely on exact-overlap criteria including recall, F1, Jaccard, and false-negative rate [Shang et al., 2019, Yang et al., 2021, Ali et al., 2023]. These criteria treat every missed medication as the same error, even though a prediction set containing a therapeutically closer medication should incur lower severity than one containing only distant medications. Consequently, two prediction sets can receive the same overlap score while differing substantially in mismatch severity (Figure 1). This motivates calibrating prediction sets to control hierarchy-aware mismatch severity.

![](images/8078fcd5cdde1687e11c3d2d04aaf0ae5e94834fe5230b055adb47c8efaa7bf7.jpg)  
Figure 2: Closest ATC match for disease-associated medications. For four disease cohorts, each selected ground-truth medication is assigned its closest predicted medication and categorized as exact, close, moderate, or distant. LLM sets threshold the predicted probabilities at 0.25; CSC sets use CSC-Avg with $\alpha = 0 . 1 2 5$ , whose cutoff is lower. The shift toward exact and close matches therefore reflects set expansion. Details are in Appendix D.1.

To characterize mismatch severity in medication prediction, we analyze LLM-generated prediction sets across cardiovascular, diabetes, respiratory, and neurology cohorts. For each cohort, we focus on selected treatment medications and identify the closest predicted match under the medication hierarchy, categorizing each match as exact, close, moderate, or distant. As shown in Figure 2, missed medications in LLM-generated sets are more often matched only to distant medication predictions. In contrast, the CSC-calibrated sets reduce these distant mismatches and increase exact or close matches. This motivates post-hoc calibration under hierarchy-aware mismatch severity.

Risk-control frameworks calibrate a threshold on top of black-box prediction scores so that the expected loss of future prediction sets is bounded [Angelopoulos and Bates, 2023]. Existing frameworks such as Risk-Controlling Prediction Sets (RCPS) and Conformal Risk Control (CRC) provide post-hoc, distribution-free calibration for black-box predictors [Bates et al., 2021, Angelopoulos et al., 2024]. However, their guarantees are only as informative as the loss being controlled. When instantiated with overlap-based losses, these frameworks still treat all missed medications equally and therefore cannot distinguish therapeutically closer predictions from distant mismatches. Even after replacing binary overlap with hierarchy-aware mismatch severity, controlling only the average severity can be insufficient. For patients with multiple medications, average severity can obscure a small number of high-severity mismatches when most medications are matched exactly or closely [Zhao et al., 2024]. These limitations call for prediction set calibration that controls mismatch severity and emphasizes the most severe mismatches within each patient.

To address this gap, we introduce Conformal Severity Control (CSC), a post-hoc, model-agnostic framework for calibrating medication prediction sets under hierarchy-aware mismatch severity. CSC uses the Anatomical Therapeutic Chemical (ATC) hierarchy [WHOCC, 2026] to measure how severe each medication mismatch is and calibrates prediction sets to control a user-specified patient-level severity tolerance, without retraining or modifying the underlying predictor. We further extend CSC from average severity control to upper-quantile severity control [Ben-Tal and Teboulle, 2007, Huang et al., 2026], which emphasizes the highest-severity mismatches within each patient. On MIMIC-IV, CSC tracks target severity tolerances, induces patient-specific prediction-set sizes under one global threshold, and reduces distant ATC-branch mismatches.

## 2 Related Work

## 2.1 Clinical Prediction

Prior work in clinical prediction has mainly focused on improving predictive models for specific downstream tasks [Song et al., 2024, 2026a]. In medication recommendation, models have improved candidate drug sets through stronger EHR modeling [Ali et al., 2023], safety-aware objectives [Shang et al., 2019], and richer sources of clinical information [Tan et al., 2022]. A complementary line of work incorporates medical ontologies and structured clinical relationships into the predictive model to improve predictions [Yao et al., 2023, Sun et al., 2023, Tan et al., 2024]. While these approaches improve predictive performance, they are typically specific to individual model architectures. In contrast, our method is post-hoc and model-agnostic, and can be applied to any predictive model without redesigning the predictor itself.

![](images/882ff22253561c58d35be8d3714ae84d357864fea073d2cf0ecf2cb4c15e762c.jpg)  
Figure 3: Overview of CSC. Given patient input, a predictive model assigns scores to candidate medications, inducing a nested family of prediction sets as the threshold λ varies. CSC evaluates these candidate sets using an ATC-based hierarchy-aware mismatch severity and calibrates λ to control the expected upper-tail patient-level severity.

## 2.2 Clinical Calibration via Risk Control

Conformal prediction and risk-control frameworks provide distribution-free procedures for selecting prediction sets whose expected risk is controlled under exchangeability [Bates et al., 2021, Angelopoulos et al., 2024]. These frameworks are well suited to clinical prediction because they calibrate prediction sets post hoc from the scores of an already trained predictor, without requiring changes to the predictive model itself [Vazquez and Facelli, 2022]. Existing conformal methods have mainly controlled coverage-based losses, such as miscoverage, false negatives, or false positives [Fisch et al., 2022]. However, such losses do not distinguish therapeutically closer medication predictions from distant errors. We bridge this gap by quantifying the severity of each medication mismatch and calibrating prediction sets under both average and upper-quantile severity.

## 3 Methodology

## 3.1 Problem Setup

We study multi-label drug recommendation from EHR data, which provide longitudinal patient information for clinical decision support. For each patient, let $X \in { \mathcal { X } }$ denote the patient input and let $Y \subseteq \kappa$ denote the ground-truth medication set, where K is the medication label space. Each medication is represented by an ATC code, which organizes medications hierarchically from broad anatomical and therapeutic groups to finer pharmacological and chemical categories [WHOCC, 2026]. Details of the ATC taxonomy are provided in Appendix C.1.

As illustrated in Figure 3, a trained predictive model assigns a score $f _ { k } ( X ) \in [ 0 , 1 ]$ to each medication label $k \in \mathcal { K }$ . For a threshold $\lambda \in [ \bar { 0 } , 1 ]$ ], we define the induced prediction set as

$$
C _ { \lambda } ( X ) : = \{ k \in { \mathcal K } : f _ { k } ( X ) \geq 1 - \lambda \} ,\tag{1}
$$

where λ controls the inclusiveness of the prediction set, with smaller values yielding more selective sets and larger values yielding more inclusive sets.

## 3.2 Hierarchy-Aware Mismatch Severity

Given the prediction set $C _ { \lambda } ( X )$ , we quantify how each ground-truth medication is matched by the prediction set. We view the ATC hierarchy as a rooted tree of maximum depth D, where medication labels are leaves and internal nodes represent broader medication classes. For a groundtruth medication $y \in Y$ and a predicted medication $k \in \mathcal { K }$ , let $\operatorname { L C A } ( y , k )$ denote their lowest common ancestor (LCA), i.e., the most specific ATC class containing both medications. If the shared ancestor is deeper in the tree, then k is closer to y in the medication taxonomy and should receive lower mismatch severity; if the shared ancestor is closer to the root, then the mismatch is more distant and should receive higher severity. We define the pairwise mismatch severity between y and k as

$$
s ( y , k ) : = 1 - \frac { \operatorname { d e p t h } ( \mathrm { L C A } ( y , k ) ) } { D } .\tag{2}
$$

where $\mathrm { { d e p t h } ( \cdot ) }$ denotes the depth of a node in the ATC tree. Thus, $s ( y , k ) = 0$ when y and k share the same medication label, while larger values indicate more distant medication classes. Prior work has used ATC distance to quantify therapeutic drug similarity [Wang et al., 2013] and ATC proximity to characterize therapeutic relatedness [Bohn et al., 2025].

For a ground-truth medication $y \in Y$ , we define its label-level mismatch severity with respect to the prediction set $C _ { \lambda } ( X )$ by the closest predicted medication:

$$
s { \bigl ( } y , C _ { \lambda } ( X ) { \bigr ) } : = { \left\{ \right.} \begin{array} { l l } { \operatorname* { m i n } _ { k \in C _ { \lambda } ( X ) } s ( y , k ) , } & { C _ { \lambda } ( X ) \neq \emptyset , } \\ { 1 , } & { C _ { \lambda } ( X ) = \emptyset . } \end{array}\tag{3}
$$

It assigns maximal severity when the prediction set is empty, preserving boundedness and monotonicity. The quantity $s ( y , C _ { \lambda } \overset { \cdot } { ( } X ) )$ is zero when $y \in C _ { \lambda } ( X )$ , small when the prediction set contains a medication close to y in the ATC hierarchy, and large when all predicted medications are distant from y. Since $C _ { \lambda } ( X )$ expands as λ increases, $s ( y , C _ { \lambda } ( X ) )$ is non-increasing in λ for every $y \in Y$

## 3.3 Average and Upper-Quantile Severity Losses

The label-level mismatch severity $s ( y , C _ { \lambda } ( X ) )$ scores a single ground-truth medication. Since a patient may have multiple ground-truth medications, we aggregate label-level severities into a patient-level loss. For $Y \neq \emptyset$ , the average severity loss is

$$
\mathcal { S } _ { \mathrm { a v g } } ( Y , C _ { \lambda } ( X ) ) : = \frac { 1 } { | Y | } \sum _ { y \in Y } s \big ( y , C _ { \lambda } ( X ) \big ) ,\tag{4}
$$

and for $Y = \emptyset$ we set this loss, and the upper-quantile loss below, to zero.

Average severity can mask a few high-severity mismatches when most ground-truth medications are matched exactly or closely. To emphasize such mismatches, we control the upper tail of the within-patient severities. For $\dot { Y } \neq \varnothing$ , let $m = | \boldsymbol { Y } |$ , write $z _ { y } = s ( y , C _ { \lambda } ( X ) )$ , and fix a quantile level $\beta \in [ 0 , 1 )$ . The upper-quantile severity loss is the conditional value-at-risk (CVaR) at level β of the uniform distribution over $\{ z _ { y } \} _ { y \in Y }$ [Rockafellar and Uryasev, 2000, Ben-Tal and Teboulle, 2007]:

$$
\mathcal { S } _ { \beta } ( Y , C _ { \lambda } ( X ) ) : = \operatorname* { m i n } _ { t \in [ 0 , 1 ] } \Big [ t + \frac { 1 } { ( 1 - \beta ) m } \sum _ { y \in Y } ( z _ { y } - t ) _ { + } \Big ] ,\tag{5}
$$

where $( u ) _ { + } = \operatorname* { m a x } \{ u , 0 \}$ . Equivalently, sort $z _ { ( 1 ) } \geq \cdot \cdot \cdot \geq z _ { ( m ) }$ , and let $q = ( 1 - \beta ) m$ and $r = \lceil q \rceil$ Then

$$
\mathscr { S } _ { \beta } ( Y , C _ { \lambda } ( X ) ) = \frac { \sum _ { j = 1 } ^ { r - 1 } z _ { ( j ) } + \left( q - r + 1 \right) z _ { ( r ) } } { q } .\tag{6}
$$

The level $\beta$ sets how much of the tail is averaged. At $\beta = 0$ the loss is the average severity $ { S _ { \mathrm { a v g } } }$ in $( 4 ) ,$ and larger $\beta$ concentrates it on fewer, more severe mismatches, up to the worst label-level severity $\operatorname* { m a x } _ { y \in Y } z _ { y }$ once $\beta \ge 1 - 1 / m$ . The same $\beta$ thus acts as worst-case severity for patients with few medications and as a tail average for patients with many.

Proposition 3.1 (Properties of the severity loss). For every $\beta \in [ 0 , 1 )$ , the loss $S _ { \beta } ( Y , C _ { \lambda } ( X ) )$

(i) lies in [0, 1];

(ii) is non-increasing in $\lambda ;$

(iii) equals 0 at $\lambda = 1$

The same holdsfor $S _ { \mathrm { a v g } } = S _ { \mathrm { 0 } }$

These are the conditions of conformal risk control [Angelopoulos et al., 2024]: a bounded loss that is non-increasing in λ and vanishes at the largest threshold.

Remark 3.2 (Connection to FNR control). For $Y \neq \emptyset .$

$$
\mathcal { S } _ { \mathrm { a v g } } ( Y , C _ { \lambda } ( X ) ) = \frac { 1 } { | Y | } \sum _ { y \in Y \backslash C _ { \lambda } ( X ) } s \big ( y , C _ { \lambda } ( X ) \big ) \ \leq \ \frac { | Y \setminus C _ { \lambda } ( X ) | } { | Y | } = \mathrm { F N R } ( Y , C _ { \lambda } ( X ) ) .
$$

CRC-FNR calibrates the loss under the hierarchy-blind cost $s ( y , k ) = \mathbb { 1 } \{ k \neq y \}$ . At a common α on the same calibration data, CSC-Avg therefore selects $\begin{array} { r } { \hat { \lambda } _ { \mathrm { C S C } } \ \leq \ \hat { \lambda } _ { \mathrm { F N R } } . } \end{array}$ giving nested sets $C _ { \hat { \lambda } _ { \mathrm { C S C } } } ( X ) \subseteq C _ { \hat { \lambda } _ { \mathrm { F N R } } } ( X )$ . The $\begin{array} { r } { \mathrm { g a p F N R } - \overline { { S } } _ { \mathrm { a v g } } = \frac { 1 } { | Y | } \sum _ { y \in Y \setminus C _ { \lambda } ( X ) } \bigl ( \overline { { 1 } } - s \bigl ( y , \overline { { C _ { \lambda } ( X ) } } \bigr ) \bigr ) } \end{array}$ is the credit for near misses. CSC thus meets the same tolerance with smaller sets, and the saving comes from missed medications that have an ATC relative in the set.

## 3.4 Conformal Severity Control

We now introduce CSC, a post-hoc calibration procedure that selects the threshold λ to control patient-level severity. Given a new patient, our goal is to construct prediction sets whose expected patient-level severity is controlled below a user-specified tolerance $\alpha .$ . To achieve this without retraining the predictor or assuming a parametric distribution, we work in the CRC framework under exchangeability [Bates et al., 2021, Angelopoulos et al., 2024]. Using the held-out calibration split, we select a threshold $\hat { \lambda }$ with finite-sample severity control. For a tolerance $\alpha \in ( 0 , 1 )$ , a level $\beta \in [ 0 , 1 )$ , and the loss $S _ { \beta }$ from Section 3.3, the threshold is

$$
\hat { \lambda } : = \operatorname* { m i n } \left\{ \lambda \in \Lambda : \frac { 1 } { n + 1 } \left( \sum _ { i = 1 } ^ { n } S _ { \beta } ( Y _ { i } , C _ { \lambda } ( X _ { i } ) ) + 1 \right) \leq \alpha \right\} , \quad \Lambda = \{ j / 4 0 0 : j = 0 , \ldots , 4 0 0 \} ,\tag{7}
$$

with min $\varnothing : = 1$ . The +1 term accounts for the unseen test patient. Every loss is zero at $\lambda = 1 \mathrm { b y }$ Proposition 3.1(iii), so the feasible set is empty only when $\mathbf { \bar { \alpha } } \propto 1 / ( n + \mathbf { \bar { 1 } } )$ , and the fallback then returns the full label set. The prediction set for a new patient is $C _ { \hat { \lambda } } ( X )$

Theorem 3.3 (Expected severity control). Let f be a fixed score function, and let $( X _ { 1 } , Y _ { 1 } ) , \ldots , ( X _ { n } , Y _ { n } ) , ( X , Y )$ be exchangeable. For any $\beta \in [ 0 , 1 )$ and $\alpha \in ( 0 , 1 )$ , the threshold $\hat { \lambda }$ in (7) satisfies

$$
\mathbb { E } \left[ S _ { \beta } ( Y , C _ { \hat { \lambda } } ( X ) ) \right] \leq \alpha ,\tag{8}
$$

where the expectation is over the calibration data and the new patient.

Theorem 3.3 is the CRC guarantee [Angelopoulos et al., 2024, Theorem 1] applied to ${ \mathcal { S } } _ { \beta } .$ , with Proposition 3.1 supplying its conditions. The guarantee is marginal: it bounds severity averaged over calibration draws and future patients. For $\beta > 0$ , it controls the expected within-patient tail, rather than an upper quantile of severity across the patient population. Exchangeability is supported by the random patient-level split in Section 4 and does not extend to temporal or cross-site shift.

## 3.5 Algorithmic Summary

Algorithm 1 summarizes CSC. Phase 1 runs once on the calibration split. It scans Λ in ascending order, computes $S _ { \beta }$ for every calibration patient, and stops at the first λ that meets the tolerance, returning $\hat { \lambda } = 1$ if none does. Phase 2 applies $\hat { \lambda }$ to new patients. Theorem 3.3 holds for any finite grid containing $\lambda = 1$ , and refining the grid can only lower $\hat { \lambda } ;$ we use a step of $1 / 4 0 0$

```latex
Algorithm 1 Conformal Severity Control: Offline Calibration and Online Prediction
Require: Calibration set $\overline { { \mathcal { D } _ { \mathrm { c a l } } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n } } } ,$ , scoring model f trained without $\mathcal { D } _ { \mathrm { c a l } } ,$ , tolerance
${ \boldsymbol { \alpha } } \in ( 0 , 1 )$ , level $\beta \in [ 0 , 1 ) , \mathrm { A T } \dot { \mathrm { C } }$ tree of depth D, grid $\Lambda = \{ \bar { j } / 4 0 0 : j = 0 , \dots , 4 0 0 \}$
Ensure: Calibrated threshold $\hat { \lambda }$ and prediction set $C _ { \hat { \lambda } } ( X )$ for a new patient $X .$
1: % Phase 1: Offline Calibration
2: $\hat { \lambda } \gets 1$ (fallback: full label set)
3: for $\lambda \in \Lambda$ in ascending order do
4: for each $( X _ { i } , Y _ { i } ) \in \mathsf { \bar { D } } _ { \mathrm { c a l } }$ do
5: $C _ { \lambda } ( X _ { i } ) \longleftarrow \{ \overset { \cdot } { k } \in \mathcal { K } : f _ { k } ( X _ { i } ) \geq 1 - \lambda \}$
6: $\mathbf { i f } Y _ { i } = \varnothing$ then
7: $S _ { i } ( \lambda ) \gets 0$
8: else
9: $z _ { i , y } \gets s ( y , C _ { \lambda } ( X _ { i } ) )$ for each $y \in Y _ { i } ,$ using $s ( y , k ) = 1 - \operatorname { d e p t h } ( \mathrm { L C A } ( y , k ) ) / D$
10: Sort $z _ { i , ( 1 ) } \geq \cdot \cdot \cdot \geq z _ { i , ( m _ { i } ) }$ , where $m _ { i } = | { \tilde { Y _ { i } } } | ;$ set $q _ { i } = ( 1 - \beta ) m _ { i } , r _ { i } = \lceil q _ { i } \rceil$
11: $\begin{array} { r } { S _ { i } ( \lambda ) \gets \left( \sum _ { j = 1 } ^ { r _ { i } - 1 } z _ { i , ( j ) } + \left( q _ { i } - r _ { i } + 1 \right) z _ { i , ( r _ { i } ) } \right) / q _ { i } } \end{array}$
12: end if
13: end for
14: $\begin{array} { r } { \mathbf { i f } \ \frac { 1 } { n + 1 } \left( \sum _ { i = 1 } ^ { n } S _ { i } ( \boldsymbol { \lambda } ) + 1 \right) \leq \alpha } \end{array}$ then
15: $\hat { \lambda }  \lambda ;$ break
16: end if
17: end for
18: % Phase 2: Online Inference for a New Patient
19: Compute scores $f _ { k } ( X )$ for all $k \in \mathcal { K }$
20: Return $C _ { \hat { \lambda } } ( X ) = \{ k \in \mathcal { K } : f _ { k } ( X ) \geq 1 - \hat { \lambda } \}$
```

## 4 Experiment Design

Dataset and Pre-processing. We evaluate on MIMIC-IV, a publicly available de-identified EHR dataset of patients admitted to the emergency department or intensive care unit (ICU) at Beth Israel Deaconess Medical Center in Boston, MA [Johnson et al., 2023]. It contains inpatient EHR records, including diagnoses, procedures, medication prescriptions, and admission/discharge timestamps. Diagnoses and procedures are represented using ICD codes, medications are mapped from NDC to ATC level-4 codes. Dataset statistics and preprocessing are detailed in Appendix C.

Medication Recommendation Problem. We construct patient sequences by treating each inpatient admission as one visit. For patient i, let $X _ { i } = [ X _ { i } ^ { ( 1 ) } , \dots , X _ { i } ^ { ( N _ { i } ) } ]$ denote the temporally ordered visit sequence, where each visit contains diagnoses, procedures, and medications. We select the last visit $t = N _ { i }$ of each patient as the target visit. The model input includes patient demographics, diagnoses and procedures from visits 1 through t, and medication history from visits 1 through t − 1. The prediction target is the medication set prescribed at visit $t ,$ denoted by $Y _ { i } ^ { ( t ) } \subseteq \kappa$

LLM Drug Recommender. We use an LLM-based drug recommendation predictor built on the Llama-based Meerkat-8B model [Kim et al., 2025]. The Meerkat-8B LLM takes patient demographics and longitudinal visit history as input using the prompt template provided in Appendix C.4. We obtain sequence embeddings using LLM2Vec [BehnamGhader et al., 2024] and place a multi-label prediction head on top of these embeddings. We fine-tune the model with LoRA adapters on the attention projection layers while keeping the remaining parameters frozen [Hu et al., 2022]. The model is trained with binary cross-entropy (BCE) over medication labels and predicts a score for each medication label. For evaluation, we report precision, recall, and F1.

![](images/2964ee2f5db8c005afce5b77f6f401141744882b5db4584d1cc5c0c6783b05da.jpg)  
Figure 4: Empirical ATC taxonomy-based mismatch severity across target tolerances. For each $\alpha ,$ CSC selects $\hat { \lambda }$ on the full calibration split, and the resulting prediction sets are evaluated once on the original test split. The left panel shows average severity (CSC-Avg); the right panel shows within-patient empirical CVaR at $\hat { \beta } = 0 . 7 5 \ : ( \mathrm { C S C - U p p e r } )$ . The dashed line marks $y = \alpha$ . Each point is a single-split empirical estimate.

Data Splits and Calibration. We split patients into training, calibration, and test sets in a 60:20:20 ratio. We use patient-level splits, with all visits from the same patient kept in one split, so exchangeability is assumed between calibration and test patient-level examples. The training split is used to fine-tune the LLM-based predictor for drug recommendation. The calibration split is used to select the CSC threshold $\hat { \lambda } ,$ and the test split is held out for final evaluation.

Calibration Setup. We evaluate CSC under both average severity and upper-quantile severity control at tolerance levels $\alpha \in \{ 0 . 1 2 5 , 0 . 2 5 , 0 . 3 5 \}$ . Since medications are evaluated at ATC level $^ { 4 , }$ label-level severity takes values in {0, 0.25, 0.5, 0.75, 1}. Thus, α specifies the target average patient-level severity. For example, $\alpha = 0 . 2 5$ means that the calibrated prediction sets allow an average severity comparable to a one-level ATC mismatch. For upper-quantile severity control, we evaluate $\beta \in \{ 0 . 5 0 , 0 . 7 5 , 0 . 9 0 \}$ , where larger $\beta$ focuses on fewer high-severity mismatches per patient. The calibrated threshold $\hat { \lambda }$ is selected on the calibration split and then fixed for evaluation on the test split.

Baselines. We evaluate two CSC variants, CSC-Avg and CSC-Upper, against both conformal and non-conformal baselines. For the conformal baseline, CRC-FNR calibrates the multi-label falsenegative rate (FNR) rather than the hierarchy-aware mismatch severity. For non-conformal baselines, Score-threshold selects all labels whose predicted probability exceeds a global cutoff $\tau ,$ while Score-top-k selects the k highest-scoring labels for each patient. We tune τ or k on the calibration split to match the recall or prediction set size of CSC. Prevalence-top-k ignores patient-specific scores and recommends the k most frequent medication labels in the calibration split.

Evaluation Metrics. We report empirical average severity, upper-quantile severity, prediction set size, precision, recall, and F1 on the held-out test split. Severity metrics evaluate the calibrated prediction sets under the ATC hierarchy, while precision, recall, and F1 measure exact label overlap.

## 5 Results

## 5.1 Severity Calibration Across User-Specified Tolerances

We evaluate whether CSC achieves empirical severity close to the user-specified tolerance $\alpha .$ . For each α, we select λ<sup>ˆ</sup> on the full calibration split and evaluate the resulting prediction sets once on the original test split. Figure 4 shows that CSC-Avg attains average severity 0.1218, 0.2439, and 0.3465 at $\alpha = 0 . 1 2 5 , 0 . 2 5$ , and 0.35, and CSC-Upper with $\beta = 0 . 7 5$ attains within-patient empirical CVaR $0 . 1 1 0 5 , 0 . 2 3 4 1$ , and 0.3449. All six point estimates lie below their targets. For a fixed prediction set, larger $\beta$ defines a stricter loss because it averages fewer and higher-severity mismatches; Table 7 reports other values of $\beta ,$ , each calibrated separately.

![](images/c538a181292184fe8bdcaafa1f83e163100c01332a5d63c6da11dc9ea89e1b5e.jpg)

![](images/8741437d77c457775aa84415c564722f960c0d0b7b51e005998165cf8a7bd125.jpg)  
Figure 5: CSC produces patient-specific prediction sets under a single calibrated threshold $\hat { \lambda } .$ (a) CSC matches fixed Top-k baseline in average set size but produces a broad distribution of $k _ { i }$ across patients. (b) Larger ground-truth medication sets correspond to larger calibrated sets, while fixed Top-k returns the same set size for all patients without considering patient-specific mismatch severity.

Table 1: Calibration results across target tolerances. For each $\alpha , \hat { \lambda }$ is selected on the full calibration split, and all other entries are point estimates on the original test split. The controlled loss is average severity for CSC-Avg and within-patient empirical CVaR at $\beta = 0 . 7 5$ for CSC-Upper. Precision, recall, and F1 are averaged across patients.
<table><tr><td>Method</td><td>α</td><td>λ</td><td>Set Size</td><td>Controlled Loss</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td rowspan="3">CSC-Avg</td><td>0.125</td><td>0.9325</td><td>28.53</td><td>0.1218</td><td>0.315</td><td>0.813</td><td>0.438</td></tr><tr><td>0.25</td><td>0.8225</td><td>16.38</td><td>0.2439</td><td>0.443</td><td>0.661</td><td>0.510</td></tr><tr><td>0.35</td><td>0.7175</td><td>11.66</td><td>0.3465</td><td>0.522</td><td>0.551</td><td>0.512</td></tr><tr><td rowspan="3">CSC-Upper  $( \beta = 0 . 7 5 )$ </td><td>0.125</td><td>0.9875</td><td>62.87</td><td>0.1105</td><td>0.166</td><td>0.945</td><td>0.274</td></tr><tr><td>0.25</td><td>0.9675</td><td>40.83</td><td>0.2341</td><td>0.240</td><td>0.888</td><td>0.366</td></tr><tr><td>0.35</td><td>0.9425</td><td>30.94</td><td>0.3449</td><td>0.298</td><td>0.833</td><td>0.423</td></tr></table>

Table 1 shows the resulting trade-off: for CSC-Avg, increasing α from 0.125 to 0.35 reduces the mean set size from 28.53 to 11.66, raising precision from 0.315 to 0.522 and lowering recall from 0.813 to 0.551. At the same $\alpha ,$ CSC-Upper returns larger sets because it controls the worst-covered medications within each patient.

Finite-sample correction. To isolate the +1 correction in Eq. (7), we compare CSC-Avg with a plug-in rule that uses the same scores, loss, and grid but selects the smallest λ whose uncorrected calibration loss is at most α. At $\alpha = 0 . 2 5$ , we draw 100 calibration subsamples without replacement for each $n \in \{ 2 5 , 5 0 , 1 0 0 \}$ and evaluate both rules on the same fixed test split. Table 2 shows that the plug-in test loss exceeds α in 38–41% of subsamples, whereas CSC exceeds it in 13–28%. CSC pays for this with larger sets, 2.07 additional labels on average at $n = 2 5$ . With the full calibration split, the two rules nearly coincide. These exceedance fractions are empirical diagnostics on a fixed test split, not failure probabilities of Theorem 3.3, which bounds the expected loss. Appendix D.3 extends the comparison to $n \leq 2 { , } 0 0 0$

## 5.2 Patient-Adaptive Set Sizes under a Global Severity Threshold

We next examine the set sizes that a single global threshold induces. For patient $i ,$ the threshold $\hat { \lambda }$ yields an effective set size $k _ { i } = | C _ { \hat { \lambda } } ( X _ { i } ) |$ that depends on the patient’s score distribution, whereas fixed Top-k assigns the same k to every patient. Figure 5a shows CSC-Upper at $\alpha = 0 . 2 5$ and $\beta = 0 . 7 5 \mathrm { { : } }$ its mean set size is close to that of size-matched Top-k $( \bar { k } _ { i } = 4 0 . 8 \dot { 3 }$ versus $k = 4 0 )$ , but $k _ { i }$ varies widely across patients (median $^ { 3 8 , }$ interquartile range [26, 52]). Figure 5b shows that $k _ { i }$ increases with the ground-truth medication count $| { \bar { Y } } _ { i } |$ . This variation describes how the calibrated cutoff distributes set sizes across patients; it does not by itself imply lower ATC mismatch at a matched budget, which Table 3 examines.

Table 2: Effect of the finite-sample correction at α = 0.25 (CSC-Avg). Plug-in omits the +1 correction in Eq. (7). For $n \leq 1 0 0 .$ , entries are means over 100 calibration subsamples evaluated on the same fixed test split; brackets give empirical 2.5–97.5% quantiles of test loss across subsamples, and Exceed is the fraction of subsamples with test loss above α. The full-calibration row is a single point estimate.
<table><tr><td rowspan="2">n</td><td colspan="3">CSC</td><td colspan="3">Plug-in</td></tr><tr><td>Loss [quantiles]</td><td>Size</td><td>Exceed</td><td>Loss [quantiles]</td><td>Size</td><td>Exceed</td></tr><tr><td>25</td><td>0.214 [0.154, 0.287]</td><td>18.67</td><td>13%</td><td>0.245 [0.171, 0.319]</td><td>16.60</td><td>40%</td></tr><tr><td>50</td><td>0.230 [0.183, 0.284]</td><td>17.44</td><td>19%</td><td>0.245 [0.198, 0.303]</td><td>16.45</td><td>41%</td></tr><tr><td>100</td><td>0.238 [0.203, 0.275]</td><td>16.80</td><td>28%</td><td>0.246 [0.210, 0.287]</td><td>16.35</td><td>38%</td></tr><tr><td>3,826</td><td>0.244</td><td>16.38</td><td></td><td>0.246</td><td>16.23</td><td></td></tr></table>

Table 3: Matched baseline diagnostics at $\alpha = 0 . 2 5$ . All rules use the same predictor scores. Matching parameters are selected on the calibration split, and all entries are point estimates on the original test split. CRC-FNR matched to CSC’s calibration recall or mean set size selects the same cutoff as CSC and returns identical sets. Score-top-k is matched to CSC’s mean calibration set size. Bold marks the loss that each CSC variant controls.
<table><tr><td>Method</td><td>Set size</td><td> $\widehat { S } _ { \mathrm { a v g } }$ </td><td> $\widehat { S } _ { 0 . 7 5 }$ </td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>CSC-Avg (= matched CRC-FNR)</td><td>16.38</td><td>0.2439</td><td>0.6247</td><td>0.443</td><td>0.661</td><td>0.510</td></tr><tr><td>Score-top-k (k = 16)</td><td>16.00</td><td>0.2206</td><td>0.6032</td><td>0.414</td><td>0.677</td><td>0.462</td></tr><tr><td>CSC-Upper (= matched CRC-FNR)</td><td>40.83</td><td>0.0680</td><td>0.2341</td><td>0.240</td><td>0.888</td><td>0.366</td></tr><tr><td>Score-top-k (k = 40)</td><td>40.00</td><td>0.0649</td><td>0.2361</td><td>0.235</td><td>0.889</td><td>0.346</td></tr></table>

## 5.3 Calibration and Set-Size Trade-offs under Matched Baselines

Table 3 compares CSC with baselines that use the same predictor scores at α = 0.25. All matching parameters are selected on the calibration split before test evaluation. CRC-FNR uses the same nested threshold family as CSC; when its target is matched to CSC’s calibration recall or mean set size, it selects the same cutoff and returns identical sets. Score-top-k uses a different set-construction rule. At matched mean size, it attains lower average loss than CSC-Avg (0.2206 at k = 16 versus 0.2439) and similar within-patient CVaR to CSC-Upper (0.2361 at k = 40 versus 0.2341), while CSC attains higher precision and F1 in both comparisons. CSC therefore does not yield lower ATC mismatch at a fixed budget. Its role is to choose the cutoff directly from a prespecified ATC-risk tolerance with the finite-sample correction in Eq. (7); Table 2 isolates the effect of that correction. Appendix D.5 reports all tolerances and the prevalence baseline.

## 6 Discussion and Limitations

CSC controls an ATC taxonomy-based mismatch cost. The consistency between the mismatch cost and the practical clinical harms need to be further studied. Our evidence covers one single-center dataset, MIMIC-IV, its 300 most frequent ATC-4 labels, and two predictors, an LLM-based one and a non-LLM one (Appendix D.6), and the marginal guarantee assumes exchangeability. Future work includes clinician-grounded evaluation and extending CSC to temporal and cross-hospital distribution shift.

## 7 Conclusion

We introduced CSC, a post-hoc framework for calibrating medication prediction sets under ATC-based mismatch severity. CSC replaces binary missed-label errors with a hierarchy-aware severity loss and calibrates a single threshold to control expected patient-level severity for any trained scoring model under exchangeability. On MIMIC-IV, CSC converts user-specified severity tolerances into calibrated prediction sets: α sets the allowed severity level, while $\beta$ controls emphasis on the highest-severity mismatches. The calibrated sets track the target severity levels, adapt their size across patients under one global threshold.

## Acknowledgments and Disclosure of Funding

Computing resources were provided via an NVIDIA Academic Grant Program award to Ziyang Song at Ohio University. Zijun Yu acknowledges partial support from an ISM Graduate Scholarship. Chi-Kuang Yeh acknowledges partial support from start-up funds provided by Georgia State University.

## References

Zafar Ali, Yi Huang, Irfan Ullah, Junlan Feng, Chao Deng, Nimbeshaho Thierry, Asad Khan, Asim Ullah Jan, Xiaoli Shen, Wu Rui, and Guilin Qi. Deep learning for medication recommendation: A systematic survey. Data Intell., 5(2):303–354, May 2023.

Hang Lv, Zixuan Guo, Yanchao Tan, Wanzi Shao, Hengyu Zhang, and Carl Yang. Awaken the giant: Activating llms via deep model guidance for boundary-aware medication recommendation. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, KDD ’26, page 1030–1041, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400722585. doi: 10.1145/3770854.3780297. URL https://doi.org/10.1145/ 3770854.3780297.

Xinyu Wang, Hanwei Wu, Zhenghan Tai, Sicheng Lyu, Qincheng Lu, Ziyu Zhao, Jijun Chi, Jingrui Tian, Xiao-Wen Chang, and Ziyang Song. Saferx-agent: A knowledge-grounded multi-agent framework for safe and explainable medication recommendation, 2026. URL https://arxiv. org/abs/2605.29146.

Benjamin Kompa, Jasper Snoek, and Andrew L Beam. Second opinion needed: communicating uncertainty in medical machine learning. NPJ Digit. Med., 4(1):4, January 2021.

Edmon Begoli, Tanmoy Bhattacharya, and Dimitri Kusnezov. The need for uncertainty quantification in machine-assisted medical decision making. Nature, 01 2019. doi: 10.1038/s42256-018-0004-1.

Junyuan Shang, Cao Xiao, Tengfei Ma, Hongyan Li, and J. Sun. Gamenet: Graph augmented memory networks for recommending medication combination. volume 33, pages 1126–1133, 07 2019. doi: 10.1609/aaai.v33i01.33011126.

Chaoqi Yang, Cao Xiao, Fenglong Ma, Lucas Glass, and J. Sun. Safedrug: Dual molecular graph encoders for recommending effective and safe drug combinations. In Proceedings ofthe Thirtieth International Joint Conference on Artificial Intelligence, IJCAI 2021, pages 3735–3741, 08 2021. doi: 10.24963/ijcai.2021/514.

Anastasios N. Angelopoulos and Stephen Bates. Conformal prediction: A gentle introduction. Found. Trends Mach. Learn., 16(4):494–591, March 2023. ISSN 1935-8237. doi: 10.1561/2200000101.

Stephen Bates, Anastasios Angelopoulos, Lihua Lei, Jitendra Malik, and Michael Jordan. Distributionfree, risk-controlling prediction sets. Journal ofthe ACM, 68:1–34, 12 2021. doi: 10.1145/3478535.

Anastasios Nikolas Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal risk control. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=33XGfHLtZg.

Zihao Zhao, Yi Jing, Fuli Feng, Jiancan Wu, Chongming Gao, and Xiangnan He. Leave no patient behind: Enhancing medication recommendation for rare disease patients. In Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, pages 533–542, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704314. doi: 10.1145/3626772.3657785.

WHO Collaborating Centre for Drug Statistics Methodology WHOCC. Guidelines for ATC classification and DDD assignment. Oslo, 2026.

Aharon Ben-Tal and Marc Teboulle. An old-new concept of convex risk measures: The optimized certainty equivalent. Mathematical Finance, 17:449–476, 02 2007. doi: 10.1111/j.1467-9965. 2007.00311.x.

Jiayi Huang, Amirmohammad Farzaneh, and Osvaldo Simeone. Optimized certainty equivalent risk-controlling prediction sets, 2026. URL https://arxiv.org/abs/2602.13660.

Ziyang Song, Qincheng Lu, Hao Xu, He Zhu, David Buckeridge, and Yue Li. Timelygpt: Extrapolatable transformer pre-training for long-term time-series forecasting in healthcare. In Proceedings of the 15th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics, BCB ’24, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400713026. doi: 10.1145/3698587.3701364. URL https: //doi.org/10.1145/3698587.3701364.

Ziyang Song, Qincheng Lu, He Zhu, David Buckeridge, and Yue Li. Trajgpt: Irregular time-series representation learning of health trajectory. IEEE Journal ofBiomedical and Health Informatics, 30(5):3888–3899, 2026a. doi: 10.1109/JBHI.2025.3620205.

Yanchao Tan, Chengjun Kong, Leisheng Yu, Pan Li, Chaochao Chen, Xiaolin Zheng, Vicki S. Hertzberg, and Carl Yang. 4sdrug: Symptom-based set-to-set small and safe drug recommendation. In Proceedings ofthe 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’22, page 3970–3980, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450393850. doi: 10.1145/3534678.3539089. URL https://doi.org/10.1145/ 3534678.3539089.

Zijun Yao, Bin Liu, Fei Wang, Daby Sow, and Ying Li. Ontology-aware prescription recommendation in treatment pathways using multi-evidence healthcare data. ACM Trans. Inf. Syst., 41(4), April 2023.

Mengxuan Sun, Jinghao Niu, Xuebing Yang, Yifan Gu, and Wensheng Zhang. Cehmr: Curriculum learning enhanced hierarchical multi-label classification for medication recommendation. Artif. Intell. Med., 143(C), September 2023. ISSN 0933-3657. doi: 10.1016/j.artmed.2023.102613. URL https://doi.org/10.1016/j.artmed.2023.102613.

Weicong Tan, Weiqing Wang, Xin Zhou, Wray Buntine, Gordon Bingham, and Hongzhi Yin. OntoMedRec: Logically-pretrained model-agnostic ontology encoders for medication recommendation. World Wide Web, 27(3), May 2024.

Janette Vazquez and Julio C Facelli. Conformal prediction in clinical medical sciences. J. Healthc. Inform. Res., 6(3):241–252, September 2022.

Adam Fisch, Tal Schuster, Tommi Jaakkola, and Dr.Regina Barzilay. Conformal prediction sets with limited false positives. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato, editors, Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings ofMachine Learning Research, pages 6514–6532. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/fisch22a.html.

Yong-Cui Wang, Shi-Long Chen, Nai-Yang Deng, and Yong Wang. Network predicting drug’s anatomical therapeutic chemical code. Bioinformatics, 29(10):1317–1324, 2013. doi: 10.1093/ bioinformatics/btt158.

Justin Bohn, James P. Gilbert, Christopher Knoll, David M. Kern, and Patrick B. Ryan. Large-scale empirical identification of candidate comparators for pharmacoepidemiological studies. Drug Safety, 48(11):1229–1241, 2025. doi: 10.1007/s40264-025-01569-y.

R Rockafellar and Stan Uryasev. Optimization of conditional value-at-risk. Journal ofrisk, 2:21–42, 01 2000.

Alistair E. W. Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J. Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, Li-wei H. Lehman, Leo A. Celi, and Roger G. Mark. Mimic-iv, a freely accessible electronic health record dataset. Scientific Data, 10 (1):1, 2023. doi: 10.1038/s41597-022-01899-x.

Hyunjae Kim, Hyeon Hwang, Jiwoo Lee, Sihyeon Park, Dain Kim, Taewhoo Lee, Chanwoong Yoon, Jiwoong Sohn, Jungwoo Park, Olga Reykhart, Thomas Fetherston, Donghee Choi, Soo Heon Kwak, Qingyu Chen, and Jaewoo Kang. Small language models learn enhanced reasoning skills from medical textbooks. NPJ Digit. Med., 8(1):240, May 2025.

Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. LLM2Vec: Large language models are secretly powerful text encoders. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id= IW1PR7vEBf.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id= nZeVKeeFYf9.

Ziyang Song, Lia Shen, Yixuan Li, Qincheng Lu, Weihao Li, Haoran Xu, He Zhu, Xinyu Wang, and Yue Li. Smi: Ontology-grounded semantic ids for hierarchy-aware medical concept representation, 2026b. URL https://arxiv.org/abs/2610.18672.

Yixuan Li, Weihao Li, and Ziyang Song. Hcoe: Hyperbolic clinical ontology embeddings from biomedical language models, 2026. URL https://arxiv.org/abs/2609.30763.

Yabin Kuang and Minzhu Xie. DrugDoctor: enhancing drug recommendation in cold-start scenario via visit-level representation learning and training. Briefings in Bioinformatics, 25(6):bbae464, 2024. doi: 10.1093/bib/bbae464.

## A Notations of Variables

Table 4: Summary of notation used throughout the paper.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td colspan="2">Spaces and label sets</td></tr><tr><td> $\mathcal { X }$ </td><td>Input space.</td></tr><tr><td> $\kappa$ </td><td>Medication label space.</td></tr><tr><td> $\mathcal { V }$ </td><td>Output space of label sets, e.g., subsets of  $\kappa .$ </td></tr><tr><td> $k \in \mathcal { K }$ </td><td>Candidate or predicted medication label.</td></tr><tr><td> $Y \subseteq \kappa$ </td><td>Ground-truth medication label set for a patient.</td></tr><tr><td> $y \in Y$ </td><td>Single ground-truth medication label.</td></tr><tr><td colspan="2">Random variables, observations, and calibration data</td></tr><tr><td> $X \in { \mathcal { X } }$ </td><td>Patient/model input, viewed as a random variable.</td></tr><tr><td> $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ </td><td>Exchangeable calibration dataset, viewed as random input-label pairs</td></tr><tr><td> $x _ { i }$ </td><td>Observed features of visit i in a patient&#x27;s longitudinal record</td></tr><tr><td> $_ n$ </td><td>Number of calibration examples.</td></tr><tr><td> $_ { i }$ </td><td>Calibration-example index.</td></tr><tr><td> $\mathbb { E } [ \cdot ]$ </td><td>Expectation with respect to future exchangeable data.</td></tr><tr><td colspan="2">Prediction sets and thresholding</td></tr><tr><td> $f _ { k } ( X )$ </td><td>Model score assigned to label k for input  $X .$ </td></tr><tr><td> $\lambda \in [ 0 , 1 ]$ </td><td>Threshold parameter controlling the prediction set size.</td></tr><tr><td> $C _ { \lambda } ( { \dot { X } } )$ </td><td>Prediction set at threshold λ.</td></tr><tr><td> $\lambda _ { 1 } , \lambda _ { 2 }$ </td><td>Thresholds used to express the nestedness of prediction sets.</td></tr><tr><td> $\hat { \lambda }$ </td><td>Calibrated threshold for the generic or upper-quantile severity loss.</td></tr><tr><td> $\hat { \lambda } _ { \mathrm { a v g } }$ </td><td>Calibrated threshold for average-severity control.</td></tr><tr><td colspan="2">Hierarchy-aware severity</td></tr><tr><td> $s : \mathcal { K } \times \mathcal { K }  [ 0 , 1 ]$ </td><td>Hierarchy-aware pairwise severity function.</td></tr><tr><td> $s ( y , k )$ </td><td>Severity of predicting label k when the ground-truth label is</td></tr><tr><td> $\operatorname { L C A } ( y , k )$ </td><td>Lowest common ancestor of  $_ y$  and k in the ATC hierarchy.</td></tr><tr><td> $\mathrm { { d e p t h } } ( \cdot )$ </td><td>Depth of a node in the ATC hierarchy.</td></tr><tr><td> $s ( y , C _ { \lambda } ( X ) )$ </td><td>Severity of the closest label in  $C _ { \lambda } ( X )$  to the true label</td></tr><tr><td colspan="2">Patient-level losses and conformal calibration</td></tr><tr><td> $\ell ( C _ { \lambda } ( X ) , Y )$ </td><td>Generic bounded monotone loss used in the conformal calibration</td></tr><tr><td> $\mathcal { L }$ </td><td>framework. Generic patient-level severity loss used by CSC.</td></tr><tr><td> $\mathcal { L } _ { i } ( \lambda )$ </td><td>Patient-level loss of calibration example i at threshold</td></tr><tr><td> $\beta \in [ 0 , 1 )$ </td><td>Severity quantile level</td></tr><tr><td> $\tilde { { S _ { \beta } } ( Y , C _ { \lambda } ( X ) ) }$ </td><td>Patient-level upper-tail severity loss at severity quantile level</td></tr><tr><td> $\dot { S _ { \mathrm { a v g } } } ( Y , \dot { C _ { \lambda } } ( \dot { X } ) )$ </td><td>Patient-level average severity loss. This is a special case of</td></tr><tr><td></td><td> $S _ { \beta } ( Y , C _ { \lambda } ( X ) )$  where  $\beta = 0 .$ </td></tr><tr><td> $\alpha \in ( 0 , 1 )$ </td><td>User-specified target tolerance for the population loss or severity.</td></tr><tr><td colspan="2">Drug recommendation task notation</td></tr><tr><td> $[ X _ { i } ^ { ( 1 ) } , \dots , X _ { i } ^ { ( N _ { i } ) } ]$ </td><td>Temporally ordered patient sequence.</td></tr><tr><td> $\bar { Y } _ { i } ^ { ( \bar { t } ) }$ </td><td>Target medications set at visit t.</td></tr></table>

## B Proofs of the Propositions and Main Theorem

## B.1 Proof of Proposition 3.1

Proof of Proposition 3.1. Part (i). For nonempty ${ \cal Y } ,$ , define

$$
F _ { \beta } ( t , \lambda ) : = t + \frac { 1 } { ( 1 - \beta ) | Y | } \sum _ { y \in Y } \bigl ( s ( y , C _ { \lambda } ( X ) ) - t \bigr ) _ { + } .
$$

Since $F _ { \beta } ( t , \lambda ) \ge 0$ for $t \in [ 0 , 1 ]$ and $s ( y , C _ { \lambda } ( X ) ) \leq 1$

$$
0 \leq S _ { \beta } ( Y , C _ { \lambda } ( X ) ) = \operatorname* { m i n } _ { t \in [ 0 , 1 ] } F _ { \beta } ( t , \lambda ) \leq F _ { \beta } ( 1 , \lambda ) = 1 .
$$

For empty Y , the loss is zero by convention.

Part (ii). For fixed $Y \neq \emptyset$ and $\beta ,$ each label-level severity is non-increasing in λ. Thus, for $\lambda _ { 1 } \leq \lambda _ { 2 }$

$$
F _ { \beta } ( t , \lambda _ { 2 } ) \leq F _ { \beta } ( t , \lambda _ { 1 } ) \quad \mathrm { f o r e v e r y } t \in [ 0 , 1 ] .
$$

Taking minima over the same interval gives

$$
\begin{array} { r l } {  { S _ { \beta } ( Y , C _ { \lambda _ { 2 } } ( X ) ) = \operatorname* { m i n } _ { t \in [ 0 , 1 ] } F _ { \beta } ( t , \lambda _ { 2 } ) } } \\ & { \leq \operatorname* { m i n } _ { t \in [ 0 , 1 ] } F _ { \beta } ( t , \lambda _ { 1 } ) = S _ { \beta } ( Y , C _ { \lambda _ { 1 } } ( X ) ) . } \end{array}
$$

For empty Y, the loss remains zero.

Part (iii) holds because $C _ { 1 } ( X ) = \{ k : f _ { k } ( X ) \geq 0 \} = K \supseteq Y , \operatorname { s o } s ( y , C _ { 1 } ( X ) ) = s ( y , y ) = 0$ for every $y \in Y$ , and the loss of an all-zero collection is 0. □

## B.2 Proof of Theorem 3.3

Proof of Theorem 3.3. We establish the result by verifying that $S _ { \beta } ( Y , C _ { \lambda } ( X ) )$ ) satisfies the conditions of conformal risk control [Angelopoulos et al., 2024], then applying that framework carefully to handle the selection of $\hat { \lambda } .$

Step 1: Loss properties. By Proposition 3.1, the severity satisfies two properties required by the conformal risk control framework.

1. Boundedness. $S _ { \beta } ( y , C _ { \lambda } ( x ) ) \in [ 0 , 1 ]$ for all $( x , y ) \in \mathcal { X } \times \mathcal { Y }$ and all $\lambda \in \Lambda$

2. Monotonicity. $S _ { \beta } ( y , C _ { \lambda _ { 1 } } ( x ) ) \ge S _ { \beta } ( y , C _ { \lambda _ { 2 } } ( x ) )$ whenever $\lambda _ { 1 } \leq \lambda _ { 2 } .$ , since larger $\lambda$ induces larger prediction sets, $C _ { \lambda _ { 1 } } ( x ) \subseteq C _ { \lambda _ { 2 } } ( x )$ , and severity is non-increasing in set size.

Step 2: Calibration on a finite grid. The threshold $\hat { \lambda }$ in (7) is selected from the finite grid $\Lambda = \{ j / 4 0 0 : j = 0 , \ldots , 4 0 0 \}$ , which contains $\lambda = 1$ , with the fallback $\hat { \lambda } = 1$ when no $\lambda \in \Lambda$ satisfies the calibration condition. The argument below uses only that $\Lambda$ is finite and contains 1. The threshold $\hat { \lambda }$ is a function of the calibration data $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ only.

Step 3: Expected loss on the test point. Define the augmented empirical risk

$$
\widehat { R } ( \lambda ; \mathcal { D } _ { n + 1 } ) : = \frac { 1 } { n + 1 } \sum _ { i = 1 } ^ { n + 1 } S _ { \beta } ( Y _ { i } , C _ { \lambda } ( X _ { i } ) ) ,
$$

where

$$
{ \mathcal D } _ { n + 1 } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n + 1 } , \qquad ( X _ { n + 1 } , Y _ { n + 1 } ) = ( X , Y ) .
$$

Consider the auxiliary threshold

$$
\lambda ^ { * } = \operatorname* { m i n } \Bigl \{ \lambda \in \Lambda : \widehat { R } ( \lambda ; \mathcal { D } _ { n + 1 } ) \leq \alpha \Bigr \} .
$$

By Proposition 3.1(iii), $\widehat { R } ( 1 ; { \mathcal { D } } _ { n + 1 } ) = 0 \leq \alpha$ , so this set is nonempty and $\lambda ^ { * }$ is well defined.

Since $\lambda ^ { * }$ is a symmetric function of the augmented dataset $\mathcal { D } _ { n + 1 }$ , and the samples $\{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n + 1 }$ are jointly exchangeable, the random variables

$$
S _ { \beta } ( Y _ { i } , C _ { \lambda ^ { * } } ( X _ { i } ) ) , \qquad i = 1 , \dots , n + 1 ,
$$

are exchangeable. Therefore,

$$
\begin{array} { r l } {  { \mathbb { E } [ \pmb { S } _ { \beta } ( Y , C _ { \lambda ^ { * } } ( X ) ) ] = \frac { 1 } { n + 1 } \sum _ { i = 1 } ^ { n + 1 } \mathbb { E } [ \pmb { S } _ { \beta } ( Y _ { i } , C _ { \lambda ^ { * } } ( X _ { i } ) ) ] } \qquad } & { } \\ & { = \mathbb { E } \Big [ \widehat { R } ( \lambda ^ { * } ; \mathscr { D } _ { n + 1 } ) \Big ] \leq \alpha . } \end{array}\tag{9}
$$

It remains to compare $\lambda ^ { * }$ with the calibrated threshold $\hat { \lambda }$ defined in (7). By Step 1,

$$
0 \leq S _ { \beta } ( Y , C _ { \lambda } ( X ) ) \leq 1 \qquad { \mathrm { f o r ~ a l l ~ } } \lambda .
$$

Hence, for every $\lambda \in \Lambda$

$$
\widehat { R } ( \lambda ; \mathcal { D } _ { n + 1 } ) = \frac { 1 } { n + 1 } \sum _ { i = 1 } ^ { n + 1 } S _ { \beta } ( Y _ { i } , C _ { \lambda } ( X _ { i } ) ) \leq \frac { 1 } { n + 1 } \left( \sum _ { i = 1 } ^ { n } S _ { \beta } ( Y _ { i } , C _ { \lambda } ( X _ { i } ) ) + 1 \right) .
$$

Therefore, every λ satisfying the calibration condition in (7) is also feasible for the definition of $\lambda ^ { * }$ which implies

$$
\lambda ^ { * } \leq \hat { \lambda } .
$$

If no $\lambda \in \Lambda$ satisfies the calibration condition, then $\hat { \lambda } = 1 \ge \lambda ^ { * }$ , so the inequality holds in both cases. Finally, by monotonicity of $S _ { \beta } ( Y , C _ { \lambda } ( X ) )$ with respect to $\lambda ,$

$$
\begin{array} { r } { \mathcal { S } _ { \beta } ( Y , C _ { \hat { \lambda } } ( X ) ) \leq { \mathcal { S } _ { \beta } ( Y , C _ { \lambda ^ { * } } ( X ) ) } \qquad \mathrm { a l m o s t ~ s u r e l y } . } \end{array}
$$

Taking expectations and applying (9) gives

$$
\begin{array} { r } { \mathbb { E } \big [ S _ { \beta } ( Y , C _ { \hat { \lambda } } ( X ) ) \big ] \leq \mathbb { E } [ S _ { \beta } ( Y , C _ { \lambda ^ { * } } ( X ) ) ] \leq \alpha , } \end{array}
$$

which proves (8).

Step 4: The $\beta = 0$ case. When $\beta = 0$ , the severity functional reduces to the average severity, $\begin{array} { r } { S _ { 0 } = S _ { \mathrm { a v g } } . } \end{array}$ . The boundedness and monotonicity properties in Step 1 hold for $ { S _ { \mathrm { a v g } } }$ by Proposition 3.1, so Steps 2–3 apply verbatim, yielding (8) for all $\left| \beta \in \left[ 0 , 1 \right) \right.$ . □

## C Dataset Details

## C.1 ATC Hierarchy and Severity Interpretation

The World Health Organization’s Anatomical Therapeutic Chemical (ATC) system organizes medications into a five-level hierarchy, from broad anatomical categories to specific chemical substances [WHOCC, 2026]. In our setting, this hierarchy contains 14 anatomical main groups, 90 therapeutic subgroups, 248 pharmacological subgroups, 840 chemical subgroups, and 5,495 chemical substances. Each ATC code can therefore be viewed as a path in a rooted tree, where longer shared prefixes indicate greater therapeutic similarity [Song et al., 2026b, Li et al., 2026]. For example, A10BA02 denotes metformin, and its successive prefixes A, A10, A10B, and A10BA encode its ancestor categories in the ATC hierarchy. In our experiments, medications are represented at ATC level-4, as described in Appendix C.3.

This ATC hierarchy gives a direct interpretation of the severity function in Eq. (2). For ATC-4 labels, a lowest common ancestor at depth 4 (same ATC-4 code) yields severity 0; depth 3 (same pharmacological subgroup) yields severity $^ 1 / 4 ;$ depth 2 (same therapeutic class) yields severity $^ 1 / 2 ;$ depth 1 (same anatomical group) yields severity $^ 3 / 4 ;$ and depth 0 (different top-level groups) yields severity 1. Thus, prediction sets are penalized less for including medications closer to the ground-truth in the ATC hierarchy and more for including only distant medications.

Table 5: Structure of the ATC medication hierarchy used in this study. # Concepts denotes the number of unique medication concepts at each level.
<table><tr><td>Level</td><td>Meaning</td><td># Concepts</td><td>Example Code</td><td>Example Code Description</td></tr><tr><td>1</td><td>Anatomical main group</td><td>14</td><td>A</td><td>Alimentary tract and metabolism</td></tr><tr><td>2</td><td>Therapeutic main group</td><td>90</td><td>A10</td><td>Drugs used in diabetes</td></tr><tr><td>3</td><td>Pharmacological subgroup</td><td>248</td><td>A10B</td><td>Blood glucose lowering drugs, excluding insulins</td></tr><tr><td>4</td><td>Chemical subgroup</td><td>840</td><td>A10BA</td><td>Biguanides</td></tr><tr><td>5</td><td>Chemical substance</td><td>5,495</td><td>A10BA02</td><td>Metformin</td></tr></table>

## C.2 Dataset and Cohort Statistics

We used a large-scale, de-identified MIMIC-IV EHR dataset from Beth Israel Deaconess Medical Center (BIDMC), released via PhysioNet under a credentialed data use agreement [Johnson et al., 2023]. This dataset contains longitudinal inpatient records, including diagnoses, procedures, medication prescriptions, and admission/discharge timestamps. Patient identifiers are removed and all timestamps are shifted by a patient-specific offset. This preserves relative temporal ordering within each patient sequence while preventing alignment of absolute calendar time across patients. We construct visit-based patient sequences by treating each inpatient admission as a visit and ordering visits chronologically by admission time. Table 6 reports basic cohort statistics after preprocessing.

Table 6: Basic statistics of the pre-processed MIMIC-IV EHR dataset. “#” denotes counts and “/patient” denotes the average number of records per-patient.
<table><tr><td>Dataset</td><td>#patients</td><td>#visits</td><td>#visits/patient</td><td>#conditions/patient</td><td>#procedures/patient</td><td>#drugs/patient</td></tr><tr><td>MIMIC-IV</td><td>19,135</td><td>35,974</td><td>1.88</td><td>21.74</td><td>4.70</td><td>43.89</td></tr></table>

## C.3 Preprocessing and Concept Normalization

We applied the same preprocessing pipeline to the MIMIC-IV dataset, including selection criteria, code normalization, and data splitting, yielding consistent visit-based patient EHR sequences. We applied the same normalization procedure across cohorts to ensure a fair comparison.

Cohort construction and splitting. We treat each inpatient admission as a visit and represent each patient as a chronologically ordered sequence of visits. We perform patient-level training/calibration/test splits so that all visits from the same patient appear in only one split, avoiding information leakage across splits.

Diagnosis and procedure codes. For each visit, we extract diagnoses and procedures from the corresponding MIMIC-IV tables and represent them as unordered sets of codes. We retain the original ICD coding systems (ICD-9-CM and ICD-10-CM for diagnoses; ICD-9 and ICD-10-PCS for procedures) and preserve the recorded ICD version to avoid conflating codes across systems. We map ICD-9/10 diagnosis and procedure codes to CCS categories, and de-duplicate repeated codes within each visit. Specifically, 11,736 ICD-9-CM and 72,446 ICD-10-CM diagnosis codes are mapped to 285 CCS condition categories, while 4,670 ICD-9-PROC and 79,758 ICD-10-PCS procedure codes are mapped to 231 CCS procedure categories, yielding 285 condition and 231 procedure concepts in total.

Medication mapping and normalization. We follow the ATC-L4 medication normalization setting used in SafeRx-Agent [Wang et al., 2026]. Medications are extracted from the prescription tables, where drugs are recorded as National Drug Codes (NDCs). We standardize them using a two-stage mapping pipeline: NDC → RxNorm (RxCUI) → ATC. We retain ATC level-4 codes to preserve meaningful therapeutic and chemical structure. Unmappable NDC entries (e.g., nondrug supplies, institution-specific packaging codes, or missing references) are excluded from the medication representation. Finally, we retain the 300 most frequent ATC level-4 medication codes as the prediction vocabulary. The severity loss is evaluated over this retained medication vocabulary.

Visit-level feature construction. Each visit $x _ { i }$ is represented by three unordered sets: diagnosis codes, procedure codes, and mapped ATC-4 medication codes. These per-visit sets are then composed into a time-ordered patient sequence $\{ x _ { 1 } , \ldots , x _ { t } \}$ by admission time. Basic cohort statistics after preprocessing are reported in Table 6.

## C.4 LLM Prompt Template

Prompt Template for Drug Recommendation   
Demographics: Insurance: <INSURANCE>; Language: <LANGUAGE>;   
Religion: <RELIGION>; Marital status: <MARITAL\_STATUS>; Ethnicity: <ETHNICITY>.   
The patient has <VISIT\_NUM> ICU visits.   
In visit 1, diagnoses: <DIAG\_NAME>, . . . ; procedures: <PROC\_NAME>, . . . ; medications:   
<MED\_NAME>,   
Current visit: diagnoses: <DIAG\_NAME>, . . . ; procedures: <PROC\_NAME>,   
Task: Predict the medications to prescribe for the current visit. Output a multi-label set of   
4-level ATC codes. Choose any number of codes from the medication vocabulary. Note that   
a patient may have multiple medications.   
Output format: A multi-label set of ATC level-4 codes.

## C.5 Computing Resources

Experiments for training the LLM predictor are conducted on an NVIDIA DGX Spark system with 128 GB of memory. Training each LLM takes 2.45 hours per epoch. For MIMIC-IV, we further split the training portion into training and validation subsets. The model is fine-tuned for up to 10 epochs, with early stopping based on validation performance. Mixed-precision training (FP16) is used to improve training efficiency.

## D Additional Results

## D.1 LLM and CSC Prediction Set Matching Analysis

We provide additional implementation details and result discussion for the prediction set matching analysis shown in Figure 2. The goal of this analysis is to characterize whether treatment-associated medications are exactly matched, or instead matched by therapeutically close or distant medications in the prediction set. Beyond exact-overlap metrics, this analysis measures the hierarchical distance between each ground-truth medication and its closest predicted medication under the ATC taxonomy. To achieve this, we construct four disease cohorts representing distinct clinical categories and identify a set of representative treatment medications for each disease category. We compare the prediction sets generated by the baseline LLM drug recommender with the CSC-calibrated prediction sets constructed from the same LLM predictor. For each ground-truth treatment medication, we identify the closest predicted medication in the ATC hierarchy and categorize the match as exact, close, moderate, or distant according to the shared hierarchy depth.

Cohort Construction. We analyze four disease cohorts: cardiovascular disease, diabetes, respiratory disease, and neurology. The cardiovascular cohort contains 346 patients and corresponds to hypertension or chronic cardiovascular disease. The diabetes cohort contains 11 patients and corresponds to type 2 diabetes or metabolic syndrome. The respiratory cohort contains 29 patients and corresponds to COPD or asthma. The neurology cohort contains 25 patients and corresponds to epilepsy or chronic neurologic disease. These cohorts are analyzed independently, so the same patient may appear in more than one cohort because comorbidities are common.

For each cohort, we evaluate only ground-truth medications that belong to ATC branches associated with the target disease category, not all medications in the patient EHR data. For cardiovascular disease, we use C02, C03, C07, C08, and C09. For diabetes, we use A10. For respiratory disease, we use R03. For neurology, we use N03A. Across the four cohorts, each patient has an average of 2.91 cardiovascular-associated, 1.55 diabetes-associated, 1.07 respiratory-associated, and 1.36 neurology-associated ground-truth medication labels, respectively.

Prediction Sets from the LLM Drug Recommender and CSC. We use the LLM-based drug predictor described in Section 4 as the base prediction model and apply the CSC framework to its medication prediction scores. This yields two prediction sets for comparison. The first is the LLM prediction set obtained by thresholding the LLM-predicted medication probabilities at 0.25. The second is the CSC-calibrated prediction set obtained from the same LLM prediction scores using CSC-Avg with target severity tolerance α = 0.125. For each selected target medication, we compare it with all medications in the prediction set and select the one with the closest ATC hierarchy match. We then categorize the best match as exact, close, moderate, or distant according to its ATC hierarchy-aware mismatch severity: exact hits have severity 0, close non-exact matches have severity in (0, 0.25], moderate matches have severity 0.5, and distant mismatches have severity at least 0.75.

Overall Results Across Disease Cohorts. Across all evaluated disease-associated medication labels from the four cohorts, CSC achieves better best-match outcomes than the LLM prediction sets. The exact-hit rate increases from 51.56% for the LLM prediction sets to 94.13% for the CSC-calibrated prediction sets. The exact-or-close match rate increases from 57.16% to 95.96%. The distant mismatch rate decreases from 34.86% to 0.83%, and the mean best-match severity decreases from 0.328 to 0.0268. These results show that CSC shifts prediction sets away from distant mismatches and toward exact or close medication matches.

Macro-Averaged Results. Because the cardiovascular cohort contains many more labels than the other cohorts, we also compute macro-averaged results across the four disease groups. The macro-average exact-hit rate increases from 29.69% for the LLM prediction sets to 90.26% for the CSC-calibrated prediction sets. The macro-average distant mismatch rate decreases from 51.64% to 1.79%, and the mean best-match severity decreases from 0.490 to 0.0394. These results show that the improvement is not driven only by the large cardiovascular cohort; CSC consistently shifts disease-associated medication matches away from distant mismatches and toward exact hits across disease areas.

Cardiovascular Cohort Results. In the cardiovascular cohort, the exact-hit rate increases from 53.77% for the LLM prediction sets to 94.54% for the CSC-calibrated prediction sets. The distant mismatch rate decreases from 33.33% to 0.69%. In absolute counts, the LLM prediction sets produce 542 exact hits, 43 close non-exact matches, 87 moderate matches, and 336 distant mismatches. The CSC-calibrated prediction sets produce 953 exact hits, 15 close non-exact matches, 33 moderate matches, and only 7 distant mismatches. These results show that CSC substantially increases exact hits and nearly removes distant ATC-branch mismatches in the largest cohort.

Diabetes Cohort Results. In the diabetes cohort, the exact-hit rate increases from 11.76% for the LLM prediction sets to 88.24% for the CSC-calibrated prediction sets. The distant mismatch rate decreases from 70.59% to 0%. In absolute counts, the LLM prediction sets produce 2 exact hits, 3 close non-exact matches, and 12 distant mismatches. The CSC-calibrated prediction sets produce 15 exact hits, 2 close non-exact matches, and no distant mismatches. Although this cohort is small, these results show that CSC substantially increases exact hits and removes distant ATC-branch mismatches.

Respiratory Cohort Results. In the respiratory cohort, the exact-hit rate increases from 3.23% for the LLM prediction sets to 87.10% for the CSC-calibrated prediction sets. The distant mismatch rate decreases from 96.77% to 6.45%. In absolute counts, the LLM prediction sets produce 1 exact hit and 30 distant mismatches. The CSC-calibrated prediction sets produce 27 exact hits, 2 moderate matches, and 2 distant mismatches. These results show that CSC produces the largest reduction in distant ATC-branch mismatches in the respiratory cohort.

Neurology Cohort Results. In the neurology cohort, the exact-hit rate increases from 50.00% for the LLM prediction sets to 91.18% for the CSC-calibrated prediction sets. The distant mismatch rate decreases from 5.88% to 0%. In absolute counts, the LLM prediction sets produce 17 exact hits, 15 close non-exact matches, and 2 distant mismatches. The CSC-calibrated prediction sets produce

Table 7: Effect of $\beta$ at $\alpha = 0 . 2 5 \colon$ fixed threshold versus separate calibration. The fixed-threshold column keeps the CSC-Upper threshold calibrated at $\beta = 0 . 7 5 ( \hat { \lambda } = 0 . 9 6 7 5$ , mean set size 40.83) and changes only $\beta$ in the loss. The separate-calibration columns select a new threshold for each $\beta .$ Losses are patient-averaged within-patient empirical CVaR; all entries are point estimates on the original test split.
<table><tr><td rowspan="2"> $\beta$ </td><td>Fixed threshold</td><td colspan="3">Separate calibration</td></tr><tr><td>Test loss</td><td> $\hat { \lambda }$ </td><td>Test loss</td><td>Set size</td></tr><tr><td>0.50</td><td>0.1335</td><td>0.9275</td><td>0.2463</td><td>27.48</td></tr><tr><td>0.75</td><td>0.2341</td><td>0.9675</td><td>0.2341</td><td>40.83</td></tr><tr><td>0.90</td><td>0.3521</td><td>0.9825</td><td>0.2364</td><td>54.08</td></tr></table>

31 exact hits, 3 close non-exact matches, and no distant mismatches. These results show that CSC increases exact hits and removes the remaining distant ATC-branch mismatches in this cohort.

Transitions from Distant Mismatches to Exact Hits or Close Matches. We further measure how often the same ground-truth medication changes from a distant mismatch under the LLM prediction sets to an exact hit or close match under the CSC-calibrated prediction sets. This transition occurs for 30.65% of cardiovascular labels, 70.59% of diabetes labels, 83.87% of respiratory labels, and 5.88% of neurology labels. The mean severity reduction is largest in respiratory disease, where it reaches 0.831. The corresponding reductions are 0.544 for diabetes, 0.286 for cardiovascular disease, and 0.140 for neurology. These results show that CSC not only increases exact hits, but also moves many high-severity distant mismatches into exact or low-severity categories.

Key Findings. This prediction-set matching analysis shows why exact-overlap evaluation is not sufficient for medication recommendation. Exact-overlap metrics measure whether a ground-truth medication is exactly included in the prediction set; however, they do not distinguish a therapeutically close medication match in the ATC hierarchy from a distant match. The proposed hierarchy-aware mismatch severity captures this difference by measuring how far the closest predicted medication is from each ground-truth medication. The disease-cohort results show that the LLM prediction sets often leave disease-associated medications matched only to distant ATC branches, especially in the diabetes and respiratory cohorts. In contrast, the CSC-calibrated prediction sets substantially reduce distant mismatches and shift the best-match distribution toward exact hits and close matches. These findings show that CSC improves the structure of prediction errors, not only the number of exact hits.

## D.2 Effect of Upper-Quantile Level $\beta$

Increasing $\beta$ has two effects. For a fixed prediction set, it moves the loss toward the most severe mismatches within each patient; under separate calibration, it also changes the selected threshold. Table 7 separates the two at $\alpha = 0 . 2 5$ . Calibration with $\beta = 0 . 7 5$ selects $\hat { \lambda } = 0 . 9 6 7 5$ . Holding this threshold fixed gives mean set size 40.83 for every $\beta ,$ and the patient-averaged empirical CVaR rises from 0.1335 at $\bar { \beta } = 0 . 5 0$ to 0.2341 at $\beta = 0 . 7 5$ and 0.3521 at $\beta = 0 . 9 0$ . The last value exceeds 0.25 because this threshold was calibrated for $\beta = 0 . 7 5$ . Calibrating each $\beta$ separately instead selects larger thresholds as $\beta$ grows, with mean set sizes 27.48, 40.83, and 54.08 and losses 0.2463, 0.2341, and 0.2364, all below the target. Larger $\beta$ thus gives stronger protection against the worst-covered medications at the cost of larger sets.

## D.3 Finite-Sample Correction across Calibration Sizes

This appendix extends Table 2 to larger calibration sizes. Both rules use the same predictor scores, loss, and grid Λ; CSC applies the +1 correction in $\operatorname { E q . }$ (7), and the plug-in rule omits it. For each n, we draw 100 calibration subsamples without replacement from the 3,826 calibration patients. Within a repetition, the subsamples for different n are nested prefixes of one random permutation, and both rules use the same subsample. All rules are evaluated on the same fixed test split. As n grows, the correction shrinks and the two rules converge. For $n \geq 1 , 0 0 0$ , subsamples drawn from the same 3,826 patients overlap substantially, so their spread understates the variability of independent calibration sets.

Table 8: Finite-sample correction across calibration sizes at $\alpha = 0 . 2 5$ (CSC-Avg). For $n \leq 2 { , } 0 0 0$ loss and size are means over 100 calibration subsamples on a fixed test split; brackets give empirical 2.5–97.5% quantiles of test loss, and Exceed is the fraction of subsamples with test loss above α. Exceed is a fixed-test empirical diagnostic, not a failure probability of Theorem 3.3. The row $n = 3 , 8 2 6$ uses the full calibration split once.
<table><tr><td rowspan="2">n</td><td colspan="3">CSC</td><td colspan="3">Plug-in</td></tr><tr><td>Loss [quantiles]</td><td>Size</td><td>Exceed</td><td>Loss [quantiles]</td><td>Size</td><td>Exceed</td></tr><tr><td>25</td><td>0.214 [0.154, 0.287]</td><td>18.67</td><td>13%</td><td>0.245 [0.171, 0.319]</td><td>16.60</td><td>40%</td></tr><tr><td>50</td><td>0.230 [0.183, 0.284]</td><td>17.44</td><td>19%</td><td>0.245 [0.198, 0.303]</td><td>16.45</td><td>41%</td></tr><tr><td>100</td><td>0.238 [0.203, 0.275]</td><td>16.80</td><td>28%</td><td>0.246 [0.210, 0.287]</td><td>16.35</td><td>38%</td></tr><tr><td>250</td><td>0.242 [0.220, 0.262]</td><td>16.53</td><td>23%</td><td>0.245 [0.224, 0.266]</td><td>16.36</td><td>30%</td></tr><tr><td>500</td><td>0.244 [0.229, 0.259]</td><td>16.36</td><td>20%</td><td>0.246 [0.229, 0.261]</td><td>16.28</td><td>26%</td></tr><tr><td>1,000</td><td>0.245 [0.236, 0.254]</td><td>16.33</td><td>13%</td><td>0.246 [0.236, 0.254]</td><td>16.29</td><td>18%</td></tr><tr><td>2,000</td><td>0.245 [0.239, 0.249]</td><td>16.31</td><td>2%</td><td>0.245 [0.240, 0.249]</td><td>16.29</td><td>2%</td></tr><tr><td>3,826</td><td>0.244</td><td>16.38</td><td>一</td><td>0.246</td><td>16.23</td><td></td></tr></table>

![](images/249a69255cba080acd7b74ba9db46202799b4385ad24e91358d893af67c22392.jpg)  
Figure 6: ATC level-2 branch composition for average-severity CSC at $\alpha = 0 . 2 5$ . The left panel shows true positives and false positives among the most frequently selected ATC branches, while the right panel reports branch-level recall and precision.

## D.4 ATC Hierarchy Branch Statistics

We further inspect how CSC allocates prediction mass across the ATC hierarchy. For each ATC level-2 branch, we aggregate medication-level true positives and false positives among predicted medications. We also report branch-level recall and purity, where purity is the branch-level precision TP/(TP + FP). These plots are diagnostic only.

Figure 6 shows the branch-level composition for CSC-Avg at $\alpha = 0 . 2 5$ . Most prediction mass is concentrated in common ATC level-2 branches, including A12 mineral supplements, A06 drugs for constipation, N02 analgesics, B01 antithrombotic agents, and J07 vaccines. CSC-Avg keeps prediction sets relatively compact, so branch-level purity remains moderate in several common branches, but recall varies across medication families.

Figure 7 shows the corresponding branch-level composition for CSC-Upper with $\alpha = 0 . 2 5$ and $\beta = 0 . 7 5$ . Because CSC-Upper controls the worst-covered medications within each patient, it selects larger prediction sets and expands coverage across more ATC branches. This increases branch-level recall in many branches, but also introduces more false positives and lowers purity. The pattern is consistent with the intended trade-off of upper-tail severity control: reducing high-severity misses requires more conservative prediction sets.

## D.5 Matched Baseline Comparisons

Table 9 extends Table 3 to all three tolerances. All matching parameters are selected on the calibration split and applied once to the original test split, so the test set sizes of matched baselines can differ slightly from those of CSC. Size matching selects the k whose mean calibration set size is closest to that of CSC, and recall matching does the same for patient-averaged recall; ties go to the more inclusive candidate. Score-top-k ranks medications by predictor score for each patient, with ties broken by label index. Prevalence-top-k returns the k medications most frequent in the calibration split to every patient.

![](images/7ab2181c1ea6f0d3a0f7e6a14d79bb47f021e788987cc74e00ea114e8aca8b18.jpg)  
Figure 7: ATC level-2 branch composition for upper-quantile severity CSC at $\alpha = 0 . 2 5$ and $\beta = 0 . 7 5$ Compared with average-severity calibration, upper-tail calibration selects larger sets and increases branch-level recall in many medication families, at the cost of lower branch-level purity.

CRC-FNR is not listed separately: in all six settings, matching its target to CSC’s calibration recall or mean set size selects CSC’s cutoff and returns identical sets. Score-threshold uses the same threshold family, and at CSC’s cutoff it reproduces CSC’s sets. Score-top-k attains lower average loss than CSC-Avg at every tolerance under both matching criteria. Its within-patient CVaR differs from that of CSC-Upper by at most 0.012, lower at one setting and higher at the others. CSC attains higher patient-averaged precision and F1 than Score-top-k in every comparison. These diagnostics therefore show no fixed-budget advantage for CSC in ATC mismatch; they report the set-size and exact-overlap cost of the operating point that CSC selects for a prespecified tolerance. Prevalence-top-k, which ignores patient-specific scores, has much higher loss at matched size and needs much larger sets to match recall.

## D.6 Application to DrugDoctor-Based Scores

We apply CSC unchanged to scores from a non-LLM recommender based on DrugDoctor [Kuang and Xie, 2024]. The scores were supplied as precomputed matrices for the same calibration and test patients. We verified that patient IDs and label matrices match those of the LLM predictor, and the authors confirmed the ordering of the 300-label vocabulary; only the scores differ. The loss matrix, grid, and calibration rule are unchanged. Table 10 shows that all six test losses lie below their targets, and CSC-Upper again selects larger sets with higher recall and lower precision. The supplied files do not document whether DrugDoctor’s training and model selection excluded the calibration and test patients, so we report these results as an empirical application and do not invoke Theorem 3.3 for them.

## D.7 Sensitivity to the Taxonomy Cost

We recalibrate CSC-Avg at $\alpha = 0 . 2 5$ under three monotone transformations of the normalized LCA distance $d = s ( \boldsymbol { y } , \boldsymbol { k } )$ in Eq. (2), namely ${ \sqrt { d } } , d ,$ and $d ^ { 2 }$ , which change the penalty for nonzero mismatches but preserve its ordering. We also use a branch-dependent cost built from the ATC tree. For a node v, let $\mathbf { \bar { \rho } } _ { N ( v ) }$ be its number of ATC-5 descendants and $I ( v ) = - \log \{ N ( v ) / N ( \mathrm { r o o t } ) \}$ }; the cost between labels u and v is $I ( u ) + I ( v ) - 2 I ( \mathrm { L C A } ( u , v ) )$ , normalized by its maximum over the 300-label vocabulary. Among the 302 label pairs whose LCA has depth $^ { 3 , }$ the equal-edge cost is 0.25 for every pair, whereas this cost ranges from 0.081 to 0.449. Every cost is bounded and zero for exact matches, so the loss stays monotone and the calibration procedure is unchanged. For label L01XC, which is absent from the hierarchy file, we keep its code-prefix path and count one pseudo-leaf.

Table 9: Matched baseline diagnostics at all tolerances. Matching parameters are selected on the calibration split, and all entries are point estimates on the original test split. CRC-FNR matched to CSC’s calibration recall or mean set size returns the same sets as CSC in every block. “Size” and “recall” mark the matching criterion. Median [IQR] summarizes set sizes across test patients; fixed-k rules give every patient the same size. Bold marks the loss that CSC controls in each block.
<table><tr><td>Rule</td><td>Mean size</td><td>Median [IQR]</td><td> $\widehat { S } _ { \mathrm { a v g } }$ </td><td> $\widehat { S } _ { 0 . 7 5 }$ </td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td colspan="6">CSC-Avg, α = 0.125 (λ = 0.9325)</td><td></td><td></td></tr><tr><td>CSC-Avg (= matched CRC-FNR)</td><td>28.53</td><td>27 [17, 37]</td><td>0.1218</td><td>0.3833</td><td>0.315</td><td>0.813</td><td>0.438</td></tr><tr><td>Score-top-k (size, k = 28)</td><td>28.00</td><td></td><td>0.1108</td><td>0.3754</td><td>0.303</td><td>0.821</td><td>0.405</td></tr><tr><td>Score-top-k (recall, k = 27)</td><td>27.00</td><td></td><td>0.1171</td><td>0.3917</td><td>0.310</td><td>0.813</td><td>0.410</td></tr><tr><td>Prevalence-top-k (size, k = 28)</td><td>28.00</td><td></td><td>0.2146</td><td>0.6128</td><td>0.246</td><td>0.655</td><td>0.327</td></tr><tr><td>Prevalence-top-k (recall, k = 50)</td><td>50.00</td><td></td><td>0.1043</td><td>0.3529</td><td>0.175</td><td>0.817</td><td>0.270</td></tr><tr><td colspan="6">CSC-Avg, α = 0.25 (λ = 0.8225)</td><td></td><td></td></tr><tr><td>CSC-Avg (= matched CRC-FNR)</td><td>16.38</td><td>14 [8, 23]</td><td>0.2439</td><td>0.6247</td><td>0.443</td><td>0.661</td><td>0.510</td></tr><tr><td>Score-top-k (size, k = 16)</td><td>16.00</td><td></td><td>0.2206</td><td>0.6032</td><td>0.414</td><td>0.677</td><td>0.462</td></tr><tr><td>Score-top-k (recall, k = 15)</td><td>15.00</td><td></td><td>0.2360</td><td>0.6262</td><td>0.426</td><td>0.658</td><td>0.464</td></tr><tr><td>Prevalence-top-k (size, k = 16)</td><td>16.00</td><td></td><td>0.3700</td><td>0.7778</td><td>0.314</td><td>0.504</td><td>0.347</td></tr><tr><td>Prevalence-top-k (recall, k = 28)</td><td>28.00</td><td></td><td>0.2146</td><td>0.6128</td><td>0.246</td><td>0.655</td><td>0.327</td></tr><tr><td colspan="6">CSC-Avg, α = 0.35 (λ = 0.7175)</td><td></td><td></td></tr><tr><td>CSC-Avg (= matched CRC-FNR)</td><td>11.66</td><td>9 [5, 17]</td><td>0.3465</td><td>0.7573</td><td>0.522</td><td>0.551</td><td>0.512</td></tr><tr><td>Score-top-k (size, k = 12)</td><td>12.00</td><td></td><td>0.2913</td><td>0.6965</td><td>0.467</td><td>0.593</td><td>0.466</td></tr><tr><td>Score-top-k (recall, k = 10)</td><td>10.00</td><td></td><td>0.3388</td><td>0.7445</td><td>0.501</td><td>0.541</td><td>0.463</td></tr><tr><td>Prevalence-top-k (size, k = 12)</td><td>12.00</td><td></td><td>0.4455</td><td>0.8341</td><td>0.341</td><td>0.426</td><td>0.339</td></tr><tr><td>Prevalence-top-k (recall, k = 19)</td><td>19.00</td><td></td><td>0.3098</td><td>0.7164</td><td>0.296</td><td>0.555</td><td>0.348</td></tr><tr><td colspan="6">CSC-Upper (β = 0.75), α = 0.125 (λ = 0.9875)</td><td></td><td></td><td></td></tr><tr><td>CSC-Upper (= matched CRC-FNR)</td><td>62.87</td><td>58 [41, 80]</td><td>0.0307</td><td>0.1105</td><td>0.166</td><td>0.945</td><td>0.274</td></tr><tr><td>Score-top-k (size, recall; k = 62)</td><td>62.00</td><td></td><td>0.0289</td><td>0.1121</td><td>0.164</td><td>0.945</td><td>0.264</td></tr><tr><td>Prevalence-top-k (size, k = 62)</td><td>62.00</td><td></td><td>0.0744</td><td>0.2642</td><td>0.149</td><td>0.865</td><td>0.241</td></tr><tr><td>Prevalence-top-k (recall, k = 105)</td><td>105.00</td><td></td><td>0.0246</td><td>0.0912</td><td>0.098</td><td>0.944</td><td>0.171</td></tr><tr><td colspan="6">CSC-Upper (β = 0.75), α = 0.25 (λ</td><td></td><td></td></tr><tr><td>CSC-Upper (= matched CRC-FNR)</td><td>= 0.9675) 40.83</td><td>38 [26, 52]</td><td>0.0680</td><td>0.2341</td><td>0.240</td><td>0.888</td><td>0.366</td></tr><tr><td>Score-top-k (size, recall; k = 40)</td><td>40.00</td><td></td><td>0.0649</td><td>0.2361</td><td>0.235</td><td>0.889</td><td>0.346</td></tr><tr><td>Prevalence-top-k (size, k = 40)</td><td>40.00</td><td></td><td>0.1488</td><td>0.4776</td><td>0.202</td><td>0.758</td><td>0.296</td></tr><tr><td>Prevalence-top-k (recall, k = 70)</td><td>70.00</td><td></td><td>0.0622</td><td>0.2243</td><td>0.136</td><td>0.887</td><td>0.224</td></tr><tr><td colspan="6">CSC-Upper (β = 0.75), α = 0.35 (λ = 0.9425)</td><td></td><td></td></tr><tr><td>CSC-Upper (= matched CRC-FNR)</td><td>30.94</td><td>29 [19, 40]</td><td>0.1073</td><td>0.3449</td><td>0.298</td><td>0.833</td><td>0.423</td></tr><tr><td>Score-top-k (size, k = 31)</td><td>31.00</td><td></td><td>0.0960</td><td>0.3330</td><td>0.283</td><td>0.843</td><td>0.390</td></tr><tr><td>Score-top-k (recall, k = 30)</td><td>30.00</td><td></td><td>0.1010</td><td>0.3475</td><td>0.289</td><td>0.836</td><td>0.395</td></tr><tr><td>Prevalence-top-k (size, k = 31)</td><td>31.00</td><td></td><td>0.1946</td><td>0.5794</td><td>0.233</td><td>0.690</td><td>0.320</td></tr><tr><td>Prevalence-top-k (recall, k = 55)</td><td>55.00</td><td></td><td></td><td>0.0966 0.3302</td><td>0.163</td><td>0.834</td><td>0.257</td></tr></table>

Table 11 shows that every cost stays below its own target, while the cutoff, set size, and recall change with the cost. Losses on different scales are compared only with the target in their own row. These costs remain taxonomy-based and do not measure clinical harm.

Table 10: CSC with DrugDoctor-based scores. Each row calibrates on all 3,826 calibration patients and reports point estimates on the 3,827 test patients. Bold marks the loss that each CSC variant controls.
<table><tr><td>Method</td><td>α</td><td>λ</td><td> $\widehat { S } _ { \mathrm { a v g } }$ </td><td> $\widehat { S } _ { 0 . 7 5 }$ </td><td>Set size</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>CSC-Avg</td><td>0.125</td><td>0.9375</td><td>0.1197</td><td>0.3708</td><td>43.86</td><td>0.204</td><td>0.816</td><td>0.316</td></tr><tr><td>CSC-Avg</td><td>0.25</td><td>0.8575</td><td>0.2464</td><td>0.6091</td><td>24.73</td><td>0.309</td><td>0.659</td><td>0.399</td></tr><tr><td>CSC-Avg</td><td>0.35</td><td>0.7700</td><td>0.3462</td><td>0.7316</td><td>16.73</td><td>0.385</td><td>0.541</td><td>0.419</td></tr><tr><td>CSC-Upper</td><td>0.125</td><td>0.9825</td><td>0.0312</td><td>0.1103</td><td>93.81</td><td>0.105</td><td>0.940</td><td>0.185</td></tr><tr><td>CSC-Upper</td><td>0.25</td><td>0.9600</td><td>0.0740</td><td>0.2436</td><td>58.86</td><td>0.159</td><td>0.876</td><td>0.262</td></tr><tr><td>CSC-Upper</td><td>0.35</td><td>0.9450</td><td>0.1054</td><td>0.3337</td><td>47.63</td><td>0.191</td><td>0.834</td><td>0.301</td></tr></table>

Table 11: Taxonomy-cost sensitivity for CSC-Avg at $\alpha = 0 . 2 5 .$ . Each cost is calibrated separately on the full calibration split, and entries are point estimates on the original test split. Each test loss is on its own scale and is compared only with the 0.25 target.
<table><tr><td>Cost</td><td>λ</td><td>Test loss</td><td>Set size</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td> $\sqrt { d }$ </td><td>0.8550</td><td>0.2464</td><td>18.64</td><td>0.412</td><td>0.700</td><td>0.499</td></tr><tr><td>d (equal-edge)</td><td>0.8225</td><td>0.2439</td><td>16.38</td><td>0.443</td><td>0.661</td><td>0.510</td></tr><tr><td> $d ^ { 2 }$ </td><td>0.7700</td><td>0.2440</td><td>13.68</td><td>0.485</td><td>0.603</td><td>0.515</td></tr><tr><td>Branch-dependent IC</td><td>0.6950</td><td>0.2477</td><td>10.94</td><td>0.537</td><td>0.529</td><td>0.508</td></tr></table>