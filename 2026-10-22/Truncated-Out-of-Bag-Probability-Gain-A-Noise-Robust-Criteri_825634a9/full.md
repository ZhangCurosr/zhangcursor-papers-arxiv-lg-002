# Truncated Out-of-Bag Probability Gain: A Noise-Robust Criterion for Random-Forest Wrapper Feature Selection

Sejong Oh<sup>1</sup>

<sup>1</sup> Department of Software Science, Dankook University, Youngin, Republic of Korea

Corresponding author: Sejong Oh, sejongoh@dankook.ac.kr · ORCID: 0000-0002-9365-274X

Acknowledgments. NA.

## Abstract

Random-Forest wrapper feature selection can use out-of-bag (OOB) predictions as a built-in validation signal, but the criteria that decide each forward step accumulate OOB estimation noise. We propose TOP-Gain (Truncated Out-of-bag Probability Gain), which scores a candidate feature by the mean sample-level increase in the OOB probability of the true class, with decreases truncated at zero. Under a simple noise model, truncation removes about two-thirds of the noise variance contributed by unaffected samples while preserving the signal, and it makes the ranking of candidate features markedly more reproducible. In a pre-registered benchmark on 26 OpenML-CC18 datasets, TOP-Gain achieved the best mean rank in area under the ROC curve (ROC-AUC), accuracy and Macro-F1 among 15 methods; it was significantly better than 13 of 14 baselines and matched the accuracy of VSURF with 21 times less single-core computation. Within an identical wrapper, it selected significantly more stable feature sets than net-gain, Brier, log-loss and accuracy criteria without loss of accuracy. A confirmatory study on 24 further datasets replicated the advantage over all baselines. The accuracy gains were broad rather than tied to specific dataset characteristics, and, as pre-registered, the stability gain grew as relevant features became sparser.

Keywords: Feature selection · Wrapper methods · Random forest · Out-of-bag estimation · Selection stability · Pre-registered benchmark

## 1 Introduction

Selecting a small subset of informative variables is a routine step in the analysis of tabular data, from biomedical profiles to industrial and web data [1], [2]. A compact feature panel reduces measurement cost, limits overfitting and gives analysts a short list of variables to examine further [3]. Two properties matter. The panel must be predictive, so that a classifier trained on it generalises, and it should be stable, so that the same features are chosen when the data are perturbed slightly [4], [5]. We measure stability with Nogueira’s estimator Φ [5], which is corrected for chance.

Feature selection methods are commonly grouped into filters, embedded methods and wrappers [1]. Filters score each feature with a statistic that does not depend on the model that will later use the features. The simplest filters are univariate tests, which rank features by how well each one separates the classes on its own. Information-theoretic filters go a step further and account for the relations between features. mRMR [6] trades the relevance of a feature to the class against its redundancy with the features already chosen, and JMI [7] rewards features that are informative jointly with them. ReliefF [8] takes a different route and scores a feature by how well it distinguishes each instance from its nearest neighbours of other classes. Embedded methods select features as a by-product of fitting a model. Elastic Net [9] drives the coefficients of uninformative features of a linear model to zero, HSIC-Lasso [10] applies the same sparsity principle to a kernel measure of dependence, and LassoNet [11] extends sparsity penalties to neural networks. A third group aggregates the results of a base selector to make them more reliable. Stability Selection [12] fits sparse models on many random subsamples and keeps the features that are selected most often, and ensemble feature selection [13] combines several selectors or several data perturbations in a similar way. Filters are fast and their rankings are typically stable, but they ignore how the features will be used and can miss features that are useful only in combination with others. Wrappers, discussed next, are usually more predictive but less stable [4].

Wrappers [14] score candidate feature subsets directly by the performance of a learning algorithm. Their scores therefore reflect predictive utility, including the non-linear effects and interactions that the classifier is able to exploit. Random Forests [15] are a natural engine for such methods: every tree is grown on a bootstrap sample, so the trees that did not see an instance provide an out-of-bag (OOB) prediction for it, and the OOB predictions of all instances estimate generalisation performance without an inner cross-validation loop. Random-Forest-based selection has accordingly become well established. The impurity and permutation importances of a single forest [15], [16] rank all features at once, although both measures are biased when features are correlated or measured on different scales [17]. RF-RFE [18] refits the forest repeatedly and removes the least important features at each round, and varSelRF [19] follows the same backward strategy but uses the OOB error to decide how many features to keep. VSURF [20] first discards features whose importance does not exceed a noise threshold and then builds nested models by forward selection. Boruta [21] compares the importance of every feature with that of randomly permuted “shadow” copies and keeps only the features that beat them. The work closest to ours is that of Velthoen et al. [22], who select variables by forward selection that minimises an OOB estimate of a proper scoring rule, the continuous ranked probability score for regression, and stop the search with a binomial test on the signs of the per-sample risk differences. We build on this RF-OOB forward-selection framework; the framework itself is not our contribution.

Our contribution concerns the criterion that each forward step maximises. The most common choice, validation accuracy, is a step function of the subset: many candidates reach exactly the same accuracy, and the tie is then broken by noise. Continuous criteria avoid such ties. Examples are the mean predicted probability of the true class, which we call the net probability gain, and the strictly proper scoring rules Brier score [23] and log-loss [24]; the latter two are the classification analogues of the criterion of [22]. All of these criteria, however, aggregate the per-sample increases and decreases of the predicted probability with equal weight. When the probabilities are OOB estimates from a modest forest, most of the per-sample changes caused by adding a single feature are estimation noise of either sign, and this noise enters the criteria in full.

Several other methods also work with per-sample quantities, but none of them truncates per-sample probability gains to select features. Conditional predictive impact [25] compares the loss of each instance under the original feature and under a knockoff-perturbed copy, averages these differences and tests them; it is designed for statistical inference about a given model rather than for subset selection. Model-X knockoffs [26] use synthetic copies of the features to select a set with a controlled false discovery rate. SHAP values [27] attribute each individual prediction to the features, and their averages are often thresholded to select features. Danyluk and Arnosti [28] score features by their effect on estimated class probabilities rather than on class labels, which makes the score sensitive to features that change the predicted probabilities without changing the predicted class.

In this paper, we propose TOP-Gain (Truncated Out-of-bag Probability Gain), a criterion that counts only improvements. For a candidate � and the current subset �, TOP-Gain averages the sample-level increase max(0, $p _ { i } ^ { S \cup \{ f \} } - p _ { i } ^ { S } )$ of the OOB true-class probability and ignores the decreases. The idea of crediting a feature only for the individual cases it helps was motivated by case-based feature attribution [29], and the improvement-only form also resembles expected improvement in Bayesian optimisation [30]. The change to the RF-OOB wrapper is a single line, but it alters how the wrapper responds to estimation noise, and the rest of the paper examines this effect analytically and empirically. The main contributions are as follows.

1. A noise-robust selection criterion with a simple explanation. We show that truncation keeps the signal of truly improved instances while removing about two-thirds of the noise variance contributed by unaffected instances (Section 2.3). The effect can be observed directly: TOP-Gain ranks candidate features more reproducibly than every alternative criterion, and the part it discards is the noisiest component of the score (Section 4.6).

2. Stability without an accuracy cost. In a controlled comparison inside an identical wrapper, TOP-Gain selects more stable feature sets than the net gain, the proper scoring rules Brier and log-loss, and OOB accuracy and balanced accuracy, on 22–26 of 26 datasets each, while its accuracy is equal to or higher than theirs (Section 4.2).

3. The accuracy of the best RF wrapper at a fraction of its cost. Among 15 methods on 26 OpenML-CC18 datasets, TOP-Gain has the best rank in ROC-AUC, accuracy and Macro-F1. It is significantly better than 13 of 14 baselines and on par with VSURF, which needs 21 times more single-core computation (Sections 4.1, 4.4), and the advantage replicates against all 12 baselines tested on 24 independent datasets (Section 4.3).

4. Where the stability benefit is largest, and how to tune it. A hypothesis registered before the confirmatory data were analysed was supported: the stability gain from truncation grows as relevant features become sparser, whereas the accuracy gain is broad rather than tied to an identifiable type of dataset (Section 4.3). The size of the surrogate forest further acts as a direct handle on stability that leaves accuracy unchanged (Section 4.8).

5. An open and reproducible implementation. All code, raw results and the pre-registration protocol are publicly available.

The paper is organised as follows. Section 2 defines the TOP-Gain criterion, analyses why truncation reduces estimation noise and what it costs, and describes the forward-selection algorithm, its candidate pool and its computational cost. Section 3 presents the experimental design: the pre-registered dataset rules and hypotheses for the OpenML-CC18 benchmark and for the confirmatory study on the AutoML Benchmark, the compared methods, the evaluation protocol and statistics, and the runtime and sensitivity studies. Section 4 reports the results, beginning with the comparison with the baselines and the controlled comparison of criteria, followed by the confirmatory study, the computational cost, the accuracy–stability–cost trade-off, the analysis of the mechanism, ground-truth experiments and the sensitivity to the surrogate configuration. Section 5 discusses the findings, the practical use of TOP-Gain and its limitations, and Section 6 concludes. Online Resource 1 contains the dataset lists and the supplementary studies.

## 2 The TOP-Gain Criterion

## 2.1 Notation and OOB true-class probability

Let $\pmb { X } \in \mathbb { R } ^ { N \times P }$ be the training data, with � training instances (rows) and � features (columns), and let $y _ { i } \in \{ 1 , \ldots , C \}$ be the class label of instance �.

OOB class probabilities. For a feature subset �, we fit a Random Forest with � trees on the columns �. Each tree is grown on a bootstrap sample of the � instances. On average, a bootstrap sample leaves out about 37% of the instances, so each instance � is out of bag (OOB) for a subset $O _ { i }$ of about 0.37 � trees (Fig. 1a). These trees did not see instance � during training. Their predictions for � therefore behave like predictions on new data.

For each tree $t \in O _ { i : }$ , let $\pi _ { t } ( c \mid x _ { i } ^ { S } )$ be the class frequency of class � in the leaf that instance � falls into. The OOB class-probability vector of � averages these frequencies over the OOB trees (Fig. 1b). We use its component for the true class:

$$
\begin{array} { r } { p _ { i } ^ { S } \ = \ \frac { 1 } { | O _ { i } | } { \sum } _ { t \in O _ { i } } { \pi } _ { t } \big ( y _ { i } \mid \pmb { x } _ { i } ^ { S } \big ) . } \end{array}\tag{1}
$$

A value near 1 means that the forest on � predicts instance � confidently and correctly, without having seen it. An instance that is in bag for every tree receives the uniform value $1 / C ,$ , and $p _ { i } ^ { \emptyset } = 1 / C$

The quantity is computed for every instance from a single forest fit, with no inner cross-validation. It is, however, an estimate from only about 0.37 � trees, and its noise is the subject of Section 2.3.

![](images/227c40bc87e34c9b66cee880c2d39b8a0fa9f59e7995dbf8c389c5031115d9ab.jpg)

![](images/e57dbde5472142ad00eb72d84dc20334d250cf5b05124acc77303d5dcbb9238b.jpg)

![](images/7448a2602528ffc13eb6bf96bf8257fcd4a305669632a62266bcdbfdcbad88a7.jpg)  
Fig. 1 OOB true-class probability and the truncated gain. (a) Each tree is grown on a bootstrap sample; instance i is out of bag (white) for some of the trees. (b) Averaging the leaf class frequencies of these OOB trees gives the OOB class-probability vector of i; its true-class component is $p _ { i } ^ { S } .$ . (c) Adding a candidate f changes �<sub>�</sub> by $\Delta p _ { i }$ . The net gain averages all changes; TOP-Gain averages only the increases (illustrative values)

## 2.2 Truncated OOB probability gain

Given the current subset � and a candidate $f \notin S ,$ , TOP-Gain scores the candidate by

$$
\begin{array} { r } { \mathrm { T O P - G a i n } ( f \mid S ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m a x } \Big ( 0 , ~ p _ { i } ^ { S \cup \{ f \} } - p _ { i } ^ { S } \Big ) . } \end{array}\tag{2}
$$

The net probability gain is the positive part, which is TOP-Gain, minus the negative part:

$$
\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( p _ { i } ^ { S \cup \{ f \} } - p _ { i } ^ { S } \right) \ = \ \mathrm { T O P - G a i n } ( f \mid S ) \ - \ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \operatorname* { m a x } \Big ( 0 , \ p _ { i } ^ { S } - p _ { i } ^ { S \cup \{ f \} } \Big ) . } \end{array}\tag{3}
$$

TOP-Gain discards the negative part (Fig. 1c). Because $p _ { i } ^ { S }$ is common to all candidates at a step, ranking by net gain is equivalent to ranking by the mean OOB true-class probability of � ∪ {�}. TOP-Gain is not equivalent to any set-level score of $S \cup \{ f \}$ . It measures how many instances, and by how much, the candidate improves relative to the current subset.

## 2.3 Why truncation reduces noise

To see why truncation helps, write the observed change for instance � as $\Delta _ { i } = \delta _ { i } + \varepsilon _ { i } ,$ where $\delta _ { i }$ is the true effect of the candidate and $\varepsilon _ { i } \sim N ( 0 , \sigma ^ { 2 } )$ is OOB estimation noise whose size depends on the number of trees for which � is out of bag. At a typical forward step, a candidate leaves most instances unaffected, so that $\delta _ { i } = 0$ and the observed change is pure noise. For such an instance the net gain receives the full noise variance $\sigma ^ { 2 }$ , whereas TOP-Gain receives only

$$
\begin{array} { l } { \operatorname { V a r } ( \operatorname* { m a x } ( 0 , \varepsilon _ { i } ) ) ~ = ~ \sigma ^ { 2 } \left( \frac { 1 } { 2 } - \frac { 1 } { 2 \pi } \right) ~ \approx ~ 0 . 3 4 \sigma ^ { 2 } , } \end{array}\tag{4}
$$

so truncation removes about two-thirds of the noise variance of the unaffected instances. Each of them adds a positive floor to the score,

$$
\begin{array} { r } { E [ \mathrm { m a x } ( 0 , \varepsilon _ { i } ) ] ~ = ~ \frac { \sigma } { \sqrt { 2 \pi } } ~ \approx ~ 0 . 4 0 \sigma , } \end{array}\tag{5}
$$

but because all candidates at a step are evaluated with the same forests and data, their noise levels are similar, the floor shifts all scores alike, and it largely cancels in the ranking. For the instances that a candidate truly improves, with $\delta _ { i } \gg \sigma ,$ , the truncation changes almost nothing, so the signal is kept.

A toy simulation makes the consequence concrete. With � = 300 instances, candidate A raises 20% of the instances by 0.10 and candidate B raises another 20% by 0.07, and every instance carries independent noise $\varepsilon _ { i } \sim N ( 0 , \sigma ^ { 2 } )$ . Over 10,000 replicates per noise level, the net gain chooses the better candidate A in 99.3%, 88.4% and 77.0% of the replicates at $\sigma = 0 . 0 3$ , 0.06 and 0.10, whereas TOP-Gain does so in 100.0%, 95.6% and 82.3% (Fig. 2). The gain is largest at moderate noise. This is the regime of a 40-tree surrogate, with about 15 OOB trees per instance, in the middle of the search; Section 4.6 examines whether this prediction holds on real data.

![](images/7e7d48e8bff9e9b0db5d0fb0a0294fc56ffe538c42d42597edb9bfb2665ed202.jpg)  
Fig. 2 Toy simulation of Section 2.3: probability of choosing the truly better candidate under the net gain and under TOP-Gain, as a function of the noise level σ (10,000 replicates per level)

The argument assumes that negative changes are mostly noise, and TOP-Gain indeed does not penalise real harm. The argument rests on the assumption that negative changes are mostly noise rather than real harm. We discuss when this assumption may fail in Section 5.

## 2.4 Forward selection algorithm

Algorithm 1 embeds the criterion in a standard forward search.

Algorithm 1. TOP-Gain forward selection

Input : training data (X, y), target size K, surrogate size T, seed s   
Output: ordered feature list S   
1 A <- CandidatePool(X, y, K) # all features if P <= 150, else Section 2.5   
2 S <- [ ]; p\_base <- uniform 1/C   
3 for step = 1 .. K:   
4 for each f in A \ S (in parallel):   
5 p\_f <- OOB true-class probabilities of an RF(T trees, seed s) on S ∪ {f}   
6 score(f) <- mean\_i max(0, p\_f[i] - p\_base[i])   
7 f\* <- argmax score(f) (ties broken by feature name)   
8 S <- S + [f\*]   
9 p\_base <- OOB true-class probabilities of an RF(T trees, seed s + step) on S   
10 return S

Because the output is an ordered list, one run with $K _ { \mathrm { m a x } }$ steps yields nested panels for every smaller �. All candidates at a step are evaluated with the same seed, so their score differences are not caused by different bootstrap draws.

Every experiment in Sections 3 and 4 uses surrogate forests with $T = 4 0$ trees, a maximum depth of 5 and a minimum leaf size of 2. These values were fixed a priori and not tuned on the benchmark data. The number of trees governs a trade-off between cost and noise: each candidate at each step needs its own forest, so � multiplies the total cost, while with $T = 4 0$ every instance is out of bag for about 15 trees, which gives a usable probability estimate at a cost of seconds per fold. Shallow trees keep each fit cheap and the OOB probabilities smooth. Section 4.8 shows that accuracy is insensitive to � between 20 and 160 and to the depth, while stability increases with �. After each step the reference probabilities are refitted with seed � + step (line 9); a variant that reuses seed � gives equivalent results (Online Resource 1, Section S4).

## 2.5 Candidate pool for higher-dimensional data

When $P \leq 1 5 0$ , every feature is a candidate at every step. When $P > 1 5 0$ , evaluating all features at every step would be wasteful, and the search is restricted to a candidate pool � that is chosen once, on the training fold, before the first step. Let $m = \operatorname* { m i n } ( P$ , max(200,5�)), and let three screening lists $L _ { \mathrm { R F } } , \ L _ { F }$ and $L _ { \mathrm { v a r } }$ contain the $\lfloor m / 2 \rfloor$ top-ranked features by the impurity importance of a 40-tree Random Forest on all features, by the ANOVA � statistic and by the variance, respectively. The pool is the union of the three lists,

$$
A ~ = ~ L _ { \mathrm { R F } } ~ \cup ~ L _ { F } ~ \cup ~ L _ { \mathrm { v a r } } , \qquad \lfloor m / 2 \rfloor ~ \le ~ \lfloor A \rfloor ~ \le ~ 3 \lfloor m / 2 \rfloor ,\tag{6}
$$

so a feature enters the pool if any one of the three criteria ranks it highly. The lists are complementary: the forest importance captures non-linear and interaction effects, the � statistic captures marginal class separation, and the variance retains high-variance features that the other two criteria may miss in small samples. The size of the union depends on how much the lists overlap; with $K \leq 2 0$ , the pool contains between 100 and 300 features.

## 2.6 Computational cost

A run fits $O ( K \cdot | A | )$ forests of � shallow trees on at most � features. For $P > 1 5 0$ the pool size $| A |$ is bounded independently of �, so the cost grows only weakly with �. The candidate evaluations within a step are independent and parallelise across cores. Measured selection times are reported in Section 4.4.

## 3 Experimental Design

## 3.1 Benchmark suites and pre-registration

To support claims that hold beyond a hand-picked set of datasets, the dataset rules, the dataset characteristics (moderators) and the hypotheses were written down in a protocol before any selector was run on the new data. Every later change was appended as a dated amendment together with its reason, and the protocol and its amendments are public (Online Resource 1, Section S6). Table 1 summarises the two benchmark suites that resulted.

Table 1 The two benchmark suites. � and � are the numbers of instances and features in use, after subsampling to at most 1,500 instances and one-hot encoding

<table><tr><td></td><td>OpenML-CC18 (main benchmark)</td><td>AutoML Benchmark (confirmatory study)</td></tr><tr><td>Source</td><td>OpenML-CC18, 72 curated tasks [31]</td><td>OpenML suite 271 [32], minus every CC18 dataset</td></tr><tr><td>Eligibility</td><td>R1: P ≥ 20; R2: N ≤ 50,000; R3: 0.60 ≤ AUC of a full-feature RF &lt; 0.99</td><td>R1 and R3 unchanged; R2 relaxed to  $N \leq 1 0 ^ { 6 }$ </td></tr><tr><td>Eligible datasets</td><td>26</td><td>24</td></tr><tr><td>N in use</td><td>522-1,500</td><td>100-1,500</td></tr><tr><td>P in use</td><td>18-1,776</td><td>20–71,640</td></tr><tr><td>Classes</td><td>2-10</td><td>2-50</td></tr><tr><td>Majority/minority ratio</td><td>1.0-15.3</td><td>1.0-104.4</td></tr><tr><td>Relevance density M-rel</td><td>0.03-1.00</td><td>0.01–0.95</td></tr><tr><td>Methods</td><td>TOP-Gain, 14 baselines, 5 alternative criteria</td><td>TOP-Gain, net gain, 12 Python baselines</td></tr><tr><td>Repeats of 5-fold CV</td><td>3</td><td>3</td></tr><tr><td>Hypotheses</td><td>H1–H3 (pre-registered)</td><td>H4a, H4b (registered after the CC18 analysis)</td></tr></table>

The main benchmark draws on OpenML-CC18, a suite of 72 curated classification tasks [31]. A dataset is eligible if it satisfies three structural rules:  
$\mathbf { R 1 } \colon P \geq 2 0 ;$  
$\begin{array} { r } { \mathbf { R } 2 \colon N \le 5 0 , 0 0 0 ; } \end{array}$  
• R3: 0.60 ≤ ROC-AUC of a full-feature reference Random Forest < 0.99.

The first rule excludes problems where selection is nearly trivial, the second keeps the computation feasible, and the third excludes datasets at the ceiling or near chance level, on which all selectors perform alike. Twenty-six datasets are eligible (Table 1; Online Resource 1, Table S1).

Before any selector was run, we also computed moderators that describe each dataset. Non-linearity M-nl is the ROC-AUC of the full-feature Random Forest minus that of an L2-regularised logistic regression. Redundancy M-red is the mean, over features, of the maximum absolute Spearman correlation with any other feature. Headroom M-head is one minus the ROC-AUC of the full-feature Random Forest. Relevance density M-rel is the fraction of features whose ANOVA � statistic is significant at a false discovery rate of 0.05. Dimensionality, class imbalance and the share of categorical features complete the set.

The pre-registered hypotheses concern two per-dataset outcomes: ΔAUC, the ROC-AUC of TOP-Gain minus the median ROC-AUC of the baselines, and ΔΦ, the stability of TOP-Gain minus that of the net-gain criterion. They were derived from the noise mechanism, which suggested that truncation helps most when many candidates offer similar gains:

• H1: ΔAUC increases with non-linearity M-nl;

H2: ΔΦ increases with redundancy M-red;

H3: ΔAUC decreases as headroom M-head shrinks.

Each hypothesis is tested by a Spearman permutation test, with Holm’s correction across the three.

After the CC18 analysis had been completed, H1–H3 were recorded as not supported (Section 4.3). A new hypothesis was then registered from the exploratory CC18 results, together with a new and independent pool of datasets, before any result on that pool was computed:

• H4a (primary): ΔAUC decreases with relevance density M-rel;

H4b (secondary): ΔΦ decreases with M-rel.

The confirmatory pool is OpenML suite 271, the AutoML Benchmark (AMLB) collection of classification tasks [32], from which every CC18 dataset was removed. Rules R1 and R3 were kept unchanged, and R2 was relaxed to $N \leq 1 0 ^ { 6 }$ , because all data are subsampled anyway; 24 datasets are eligible (Table 1). Both hypotheses are tested by one-sided Spearman permutation tests with 10,000 permutations and the alternative $\rho < 0$ at $\mathfrak { a } = 0 . 0 5$ , and H4a counts as confirmed only if its test rejects.

## 3.2 Compared methods

TOP-Gain is compared with 14 established selectors. The filters are the ANOVA � test, mRMR [6], JMI [7] and ReliefF [8]. Elastic Net [9], HSIC-Lasso [10] and Stability Selection [12] represent embedded and aggregation methods. The model-based methods are Boruta [21], SVM-RFE [33], the impurity and permutation importances of a Random Forest [16] and RF-RFE [18], and we include the original R implementations of varSelRF [19] and VSURF [20] with their package defaults. Every selector returns a ranked list, of which the first � features are evaluated; for varSelRF and VSURF, which return sets, the ranking follows their elimination order and importances. As registered, varSelRF and VSURF were not run on the AMLB pool for computational reasons.

To isolate the effect of truncation, we also replace the TOP-Gain criterion by five alternatives inside exactly the same forward wrapper, with the same candidate pool, forests, seeds and tie-breaking. The alternatives are the net probability gain, the OOB Brier score, the OOB log-loss with probabilities clipped at $1 0 ^ { - 3 }$ , OOB accuracy and OOB balanced accuracy.

## 3.3 Protocol

Each dataset is evaluated by stratified 5-fold cross-validation, repeated three times, where repeat � ∈ {1,2,3} uses seed � both for the folds and for every selector. Datasets with more than 1,500 instances are first reduced to 1,500 by stratified subsampling; categorical variables are one-hot encoded and numeric missing values are imputed with training-fold medians. The panel sizes are $K \in \{ 5 , 1 0 , 1 5 , 2 0 \}$ keeping only the values with $\bar { K } \le P / 2$ . Each panel is evaluated with four downstream classifiers: a Random Forest (100 trees, balanced class weights), XGBoost (100 rounds, depth 4), an RBF-kernel support vector machine and an L2-regularised logistic regression. A selector call that failed for an infrastructure reason, such as lack of memory or a failure of parallel transport, was re-run once in full, and no result was ever replaced by that of another method.

The primary outcomes were fixed before the analysis: ROC-AUC (one-vs-rest macro average) for predictive performance and Nogueira’s Φ over the five folds for stability. Accuracy, Macro-F1 and selection time are secondary outcomes.

## 3.4 Statistical analysis

Following Demšar [34], the unit of analysis is the dataset. For each dataset and method, each metric is averaged over the � values, the four classifiers and the three repeats; the panels for different � are nested prefixes of one ordered list and therefore not independent units. For each metric we apply a Friedman test with the Nemenyi critical difference (CD) at $\mathfrak { a } = 0 . 0 5$ . For TOP-Gain versus each other method we report the mean paired difference with a bootstrap 95% confidence interval (CI; 10,000 resamples), the wins and losses over datasets, and the two-sided Wilcoxon signed-rank �-value, Holmcorrected within each family of comparisons (the 14 baselines or the five criteria).

## 3.5 Runtime study

The benchmark ran each selector with its own default parallelism. TOP-Gain and the Python baselines used ten cores, VSURF used a cluster of nine workers, and varSelRF is single-threaded, so the benchmark timings are not directly comparable. We therefore re-timed all 15 methods under two stated settings. In the first, every method runs on one core. In the second, every method that supports parallelism uses ten cores; varSelRF has no parallel option.

The measurements use the same fold (repeat 1, fold 1) of all 26 CC18 datasets and run one call at a time on an otherwise idle machine. A discarded warm-up call per method absorbs the start-up of the worker processes and of Rscript. We record the wall-clock time of each selector call and, for varSelRF and VSURF, also the R-internal fitting time, and we compare the selected lists of the two settings. The machine has an Intel Core Ultra 5 225 processor with ten cores and 15.4 GB of RAM and runs Windows 11. The software comprises Python 3.14.3 with scikit-learn 1.8.0, NumPy 2.4.3, pandas 3.0.1 and joblib 1.5.3, and R 4.6.1 with randomForest 4.7-1.2, varSelRF 0.7-9 and VSURF 1.2.1.

## 3.6 Sensitivity study and further studies

A further study, also registered before it was run, examines the sensitivity of TOP-Gain to its surrogate configuration by changing one setting at a time from the default. On all 26 CC18 datasets with three repeats, the number of trees is set to $T \in \{ 2 0 , 8 0 , 1 6 0 \}$ and the maximum depth to 3 or 8. On the 16 AMLB datasets with $P > 1 5 0$ , where the candidate pool is active, the pool size is scaled by 0.5 or 2 with one repeat. Each variant is compared with default TOP-Gain by the dataset-level statistics of Section 3.4, Holm-corrected over the variants.

Online Resource 1 reports three further studies. The first is the original ten-dataset biomedical benchmark, which includes microarray and proteomics data with � ≫ � (Section S2). The second comprises ground-truth and controlled-imbalance studies on synthetic data and on OVA\_Lung transcriptomic data (Section S3). The third is a common-seed robustness check and an ablation of multi-objective extensions (Section S4).

## 3.7 Use of generative AI tools

An AI assistant, Claude (Anthropic), was used to implement the experiment and analysis code, to run the experiments and statistical analyses, and to draft parts of the manuscript text. The authors conceived the method, designed the study, wrote the pre-registration protocol and decided every amendment, interpreted the results, and reviewed and verified all code, reported numbers, references and text. Every number in the paper is produced by the public analysis scripts from the stored result files. No generative AI was used to create images: the figures were drawn with plotting code from the experimental data, and the schematic Fig. 1 uses illustrative values. The authors take full responsibility for the content of the article.

## 4 Results

## 4.1 Main benchmark: TOP-Gain versus 14 baselines (CC18)

Table 2 and Fig. 3 summarise the comparison with the 14 baselines on the 26 CC18 datasets.

baseline, with a bootstrap 95% CI; W/L counts datasets; $p _ { \mathrm { H o l m } }$ is the Holm-adjusted Wilcoxon � for ROC-AUC (14 comparisons). Time is the median over datasets of the selection time per fold under the benchmark’s default parallelism (Section 4.4 gives controlled timings)
<table><tr><td>Method</td><td>Rank (AUC) ↓</td><td>ROC- AUC</td><td>Accuracy</td><td>Macro- F1</td><td>Φ</td><td>∆AUC [95% CI]</td><td>W/L</td><td> $p _ { \mathrm { H o l m } }$ </td><td>Time/fold (s)</td></tr><tr><td>TOP-Gain</td><td>3.92</td><td>0.8692</td><td>0.8200</td><td>0.7285</td><td>0.514</td><td>一</td><td>一</td><td>1</td><td>3.8</td></tr><tr><td>VSURF</td><td>4.15</td><td>0.8663</td><td>0.8178</td><td>0.7248</td><td>0.722</td><td>0.003 [0.000, 0.006]</td><td>13/13</td><td>0.27</td><td>18.3</td></tr><tr><td>RF-RFE</td><td>5.88</td><td>0.8603</td><td>0.8124</td><td>0.7170</td><td>0.782</td><td>0.009 [0.002, 0.020]</td><td>18/8</td><td>0.028</td><td>1.4</td></tr><tr><td>HSIC-Lasso</td><td>6.50</td><td>0.8595</td><td>0.8090</td><td>0.7141</td><td>0.694</td><td>0.010 [0.002, 0.022]</td><td>19/7</td><td>0.015</td><td>0.15</td></tr><tr><td>varSelRF</td><td>6.62</td><td>0.8603</td><td>0.8139</td><td>0.7177</td><td>0.847</td><td>0.009 [0.004, 0.015]</td><td>20/6</td><td>0.015</td><td>31.7</td></tr><tr><td>RF impurity</td><td>6.69</td><td>0.8620</td><td>0.8119</td><td>0.7172</td><td>0.853</td><td>0.007 [0.003, 0.013]</td><td>18/8</td><td>0.028</td><td>0.5</td></tr><tr><td>RF permutation</td><td>6.96</td><td>0.8619</td><td>0.8134</td><td>0.7181</td><td>0.518</td><td>0.007 [0.004, 0.012]</td><td>23/3</td><td>0.001</td><td>4.5</td></tr><tr><td>Elastic Net</td><td>7.77</td><td>0.8560</td><td>0.8061</td><td>0.7108</td><td>0.663</td><td>0.013 [0.004, 0.028]</td><td>20/6</td><td>0.004</td><td>0.05</td></tr><tr><td>Stability Selection</td><td>8.69</td><td>0.8470</td><td>0.7936</td><td>0.6998</td><td>0.682</td><td>0.022 [0.006, 0.051]</td><td>20/6</td><td>0.005</td><td>0.05</td></tr><tr><td>ANOVA</td><td>8.77</td><td>0.8534</td><td>0.8032</td><td>0.7070</td><td>0.844</td><td>0.016 [0.008, 0.025]</td><td>23/3</td><td>0.001</td><td>0.001</td></tr><tr><td>JMI</td><td>9.00</td><td>0.8457</td><td>0.7999</td><td>0.6994</td><td>0.549</td><td>0.024 [0.011, 0.044]</td><td>22/4</td><td>&lt;0.001</td><td>0.29</td></tr><tr><td>Boruta</td><td>9.96</td><td>0.8445</td><td>0.7916</td><td>0.6926</td><td>0.833</td><td>0.025 [0.014, 0.037]</td><td>22/4</td><td>&lt;0.001</td><td>6.8</td></tr><tr><td>mRMR</td><td>10.04</td><td>0.8409</td><td>0.7976</td><td>0.6952</td><td>0.551</td><td>0.028 [0.014, 0.050]</td><td>22/4</td><td>&lt;0.001</td><td>0.28</td></tr><tr><td>SVM-RFE</td><td>11.65</td><td>0.8404</td><td>0.8019</td><td>0.6993</td><td>0.549</td><td>0.029 [0.016, 0.045]</td><td>24/2</td><td>&lt;0.001</td><td>0.15</td></tr><tr><td>ReliefF</td><td>13.38</td><td>0.8016</td><td>0.7787</td><td>0.6609</td><td>0.455</td><td>0.068 [0.036, 0.107]</td><td>24/2</td><td>&lt;0.001</td><td>0.05</td></tr></table>

![](images/1cd350e8d6d5b591354994f654d576e6face9515b091d01baaa4d32aca3a0fd7.jpg)  
Fig. 3 Critical-difference diagram of the Friedman mean ranks of ROC-AUC on the 26 CC18 datasets (15 methods, three repeats; lower rank is better). The blue bar below the axis spans the methods whose mean rank lies within one critical difference (CD) of TOP-Gain, that is, the methods not significantly different from it (Nemenyi test, $\mathfrak { a } = 0 . 0 5 )$

The Friedman test rejects equal performance for all three predictive metrics $( p < 1 0 ^ { - 1 3 } )$ . TOP-Gain has the best mean rank for ROC-AUC (3.92), accuracy (3.54) and Macro-F1 (3.50), and VSURF is second in all three (4.15, 4.44 and 4.54). In ROC-AUC, TOP-Gain is significantly better than 13 of the

14 baselines $( p _ { \mathrm { H o l m } } \le 0 . 0 2 8 )$ , with mean differences between 0.007 and 0.068, and the same holds for accuracy $( p _ { \mathrm { H o l m } } \le 0 . 0 1 6 )$ and Macro-F1 $( p _ { \mathrm { H o l m } } \le 0 . 0 0 6 )$ . The exception is VSURF. Its ROC-AUC differs from that of TOP-Gain by only +0.003 (CI [0.000, 0.006]; 13 wins and 13 losses; $p _ { \mathrm { H o l m } } =$ 0.27), and although the accuracy $( + 0 . 0 0 2 , p _ { \mathrm { H o l m } } = 0 . 0 2 9 )$ and Macro-F1 $( + 0 . 0 0 4 , p _ { \mathrm { H o l m } } = 0 . 0 4 4 )$ differences are significant, they are very small. We therefore describe TOP-Gain as on par with VSURF and ahead of all other baselines.

The ranking also holds for each downstream classifier separately (Online Resource 1, Table S5). TOP-Gain has the best ROC-AUC rank with the Random Forest and XGBoost classifiers and is second, behind VSURF by less than 0.4 rank, with the support vector machine and logistic regression; in accuracy and Macro-F1 it ranks first for all four classifiers.

Stability shows the opposite picture. With $\Phi = 0 . 5 1 4 .$ , TOP-Gain has the lowest mean stability rank of the 15 methods. RF impurity (0.853), varSelRF (0.847), ANOVA (0.844) and Boruta (0.833) are the most stable, and TOP-Gain is comparable to RF permutation importance, mRMR, JMI, SVM-RFE and ReliefF $( p _ { \mathrm { H o l m } } = 1 . 0 )$ . This is the familiar trade-off between the stability of filters and importance rankings and the accuracy of wrappers [4]. Among the wrapper criteria, however, TOP-Gain is the most stable (Section 4.2), and on the AMLB pool its absolute stability is in the middle of the ranking (Section 4.3).

## 4.2 The effect of truncation: criterion comparison (CC18)

Because only the criterion changes across the rows of Table 3, this comparison isolates the effect of truncation.

Table 3 Six criteria inside the same RF-OOB forward wrapper (CC18; Friedman mean ranks among the six criteria, CD = 1.48). Δ is TOP-Gain minus the alternative; $p _ { \mathrm { H o l m } }$ is over five comparisons.
<table><tr><td>Criterion</td><td>Rank AUC↓</td><td>ROC- AUC</td><td>Macro- F1</td><td>Rank φ ↓</td><td>Φ</td><td>ΔAUC (W/L,  $\underline { { p _ { \mathrm { H o l m } } ) } }$ </td><td>∆Φ [95% CI] (W/L,  $\underline { { p _ { \mathrm { H o l m } } ) } }$ </td></tr><tr><td>TOP-Gain</td><td>2.23</td><td>0.8692</td><td>0.7285</td><td>1.42</td><td>0.514</td><td>一</td><td>1</td></tr><tr><td>OOB log-loss</td><td>2.35</td><td>0.8682</td><td>0.7277</td><td>3.15</td><td>0.417</td><td>+0.001 (13/13, 0.41)</td><td>+0.097 [0.060, 0.134] (22/4, &lt;0.001)</td></tr><tr><td>OOB Brier score</td><td>3.06</td><td>0.8664</td><td>0.7279</td><td>3.10</td><td>0.420</td><td>+0.003 (18/8, 0.016)</td><td>+0.094 [0.062, 0.127] (24/2, &lt;0.001)</td></tr><tr><td>Net probability gain</td><td>3.73</td><td>0.8655</td><td>0.7256</td><td>2.40</td><td>0.449</td><td>+0.004 (21/5, &lt;0.001)</td><td>+0.065 [0.037, 0.093] (22/4, &lt;0.001)</td></tr><tr><td>OOB balanced accuracy</td><td>4.62</td><td>0.8589</td><td>0.7174</td><td>5.38</td><td>0.276</td><td>+0.010 (22/4, &lt;0.001)</td><td>+0.238 [0.198, 0.274] (25/1, &lt;0.001)</td></tr><tr><td>OOB accuracy</td><td>5.02</td><td>0.8504</td><td>0.7127</td><td>5.54</td><td>0.265</td><td>+0.019 (24/2, &lt;0.001)</td><td>+0.249 [0.210, 0.287] (26/0, &lt;0.001)</td></tr></table>

Stability separates the criteria clearly. Against every alternative, TOP-Gain is more stable on 22–26 of the 26 datasets $( p _ { \mathrm { H o l m } } < 0 . 0 0 1 )$ ; its Φ exceeds that of the proper scoring rules by about 0.095, a relative gain of 23%, and that of the discrete criteria by about 0.24, a relative gain of 90%. Predictive performance does not suffer: TOP-Gain’s ROC-AUC equals that of the log-loss criterion (13 wins and 13 losses) and is significantly higher than that of the other four criteria. The comparison with the net gain is the most direct one, because the two criteria differ only in whether sample-level decreases are counted. Discarding them raises Φ by 0.065 (22 wins, 4 losses) and ROC-AUC by 0.004 (21 wins, 5 losses). Truncation therefore buys stability without any loss of accuracy.

## 4.3 Confirmatory study (AMLB) and the pre-registered hypotheses

On the 24 AMLB datasets, none of which was used to develop TOP-Gain or its hypotheses, the advantage over the baselines replicates and is, if anything, stronger (Table 4). TOP-Gain has the best Friedman rank among the 13 methods for ROC-AUC (2.67), accuracy (3.13) and Macro-F1 (3.21), with RF-RFE second in all three (4.21, 4.79 and 5.08). It is significantly better than all 12 baselines on all three metrics $( p _ { \mathrm { H o l m } } \leq 0 . 0 2 9 ) ;$ even against the closest one, RF-RFE, the ROC-AUC difference is +0.011 (CI [0.004, 0.019]; 17 wins, 7 losses; $p _ { \mathrm { H o l m } } = 0 . 0 1 1 )$ . The comparison with the net gain also replicates: TOP-Gain again selects more stable sets (ΔΦ = +0.080, CI [0.046, 0.115]; 22 wins, 2 losses; $p _ { \mathrm { H o l m } } < 0 . 0 0 1 )$ and has a slightly higher ROC-AUC (+0.006; 18 wins, 6 losses; $p _ { \mathrm { H o l m } } = 0 . 0 1 1 )$ , accuracy and Macro-F1 (+0.002 each, $p _ { \mathrm { H o l m } } = 0 . 0 3 6 )$ . Its absolute stability ranks seventh of 13 on this pool; it is significantly below RF impurity, ANOVA, RF-RFE and Boruta and not significantly different from the other eight baselines.

Table 4 Confirmatory AMLB study (24 datasets × 3 repeats; 12 Python baselines). Rank = Friedman mean rank (13 methods, CD = 3.72).
<table><tr><td>Method</td><td>Rank AUC ↓</td><td>ROC-AUC</td><td>Rank Φ↓</td><td>Φ</td><td>ΔAUC [95% CI]</td><td>W/L</td><td> ${ \underline { { p _ { \mathrm { H o l m } } } } }$ </td></tr><tr><td>TOP-Gain</td><td>2.67</td><td>0.8069</td><td>8.21</td><td>0.444</td><td></td><td>一</td><td></td></tr><tr><td>RF-RFE</td><td>4.21</td><td>0.7958</td><td>4.50</td><td>0.659</td><td>0.011 [0.004, 0.019]</td><td>17/7</td><td>0.011</td></tr><tr><td>RF impurity</td><td>5.12</td><td>0.7925</td><td>2.50</td><td>0.739</td><td>0.014 [0.007, 0.023]</td><td>19/5</td><td>0.005</td></tr><tr><td>RF permutation</td><td>5.54</td><td>0.7882</td><td>9.17</td><td>0.386</td><td>0.019 [0.009, 0.030]</td><td>21/3</td><td>&lt;0.001</td></tr><tr><td>HSIC-Lasso</td><td>6.58</td><td>0.7833</td><td>6.33</td><td>0.524</td><td>0.024 [0.014, 0.035]</td><td>22/2</td><td>&lt;0.001</td></tr><tr><td>Boruta</td><td>6.75</td><td>0.7751</td><td>4.21</td><td>0.655</td><td>0.032 [0.015, 0.053]</td><td>19/5</td><td>&lt;0.001</td></tr><tr><td>JMI</td><td>7.21</td><td>0.7781</td><td>9.33</td><td>0.336</td><td>0.029 [0.013, 0.049]</td><td>19/5</td><td>&lt;0.001</td></tr><tr><td>mRMR</td><td>7.96</td><td>0.7711</td><td>10.29</td><td>0.313</td><td>0.036 [0.018, 0.058]</td><td>21/3</td><td>&lt;0.001</td></tr><tr><td>ANOVA</td><td>8.04</td><td>0.7593</td><td>2.58</td><td>0.735</td><td>0.048 [0.026, 0.077]</td><td>21/3</td><td>&lt;0.001</td></tr><tr><td>Elastic Net</td><td>8.67</td><td>0.7597</td><td>6.71</td><td>0.478</td><td>0.047 [0.031, 0.065]</td><td>21/3</td><td>&lt;0.001</td></tr><tr><td>SVM-RFE</td><td>8.92</td><td>0.7656</td><td>9.29</td><td>0.356</td><td>0.041 [0.029, 0.054]</td><td>23/1</td><td>&lt;0.001</td></tr><tr><td>Stability Selection</td><td>9.12</td><td>0.7509</td><td>8.42</td><td>0.412</td><td>0.056 [0.033, 0.085]</td><td>22/2</td><td>&lt;0.001</td></tr><tr><td>ReliefF</td><td>10.21</td><td>0.7349</td><td>9.46</td><td>0.360</td><td>0.072 [0.052, 0.093]</td><td>23/1</td><td>&lt;0.001</td></tr><tr><td>Net probability gain</td><td></td><td>0.8006</td><td></td><td>0.364</td><td>0.006 [0.002, 0.011]</td><td>18/6</td><td>0.011</td></tr></table>

Table 5 reports the tests of all five pre-registered hypotheses.

Table 5 Tests of the pre-registered hypotheses. Spearman correlation between the moderator and the per-dataset outcome, with a permutation �-value.
<table><tr><td>Hypothesis</td><td>Data</td><td>Moderator → outcome</td><td>Predicted</td><td>ρ</td><td>p</td><td>Result</td></tr><tr><td>H1</td><td>CC18 (26)</td><td>M-nl → ∆AUC</td><td> $\rho > 0$ </td><td>+0.37</td><td>0.065 (Holm 0.20)</td><td>not supported</td></tr><tr><td>H2</td><td>CC18 (26)</td><td>M-red → ∆Φ</td><td> $\rho > 0$ </td><td>-0.29</td><td>0.15 (Holm 0.29)</td><td>not supported (opposite sign)</td></tr><tr><td>H3</td><td>CC18 (26)</td><td>M-head → ∆AUC</td><td> $\rho > 0$ </td><td>-0.12</td><td>0.55 (Holm 0.55)</td><td>not supported</td></tr><tr><td>H4a (primary)</td><td>AMLB (24)</td><td>M-rel → ∆AUC</td><td> $\rho < 0$ </td><td>-0.28</td><td>0.089 (one- sided)</td><td>not confirmed</td></tr><tr><td>H4b (secondary)</td><td>AMLB (24)</td><td>M-rel → ∆Φ</td><td> $\rho < 0$ </td><td>-0.57</td><td>0.004 (one- sided)</td><td>supported</td></tr></table>

No moderator of the accuracy gain was confirmed. Hypotheses H1–H3 were not supported on CC18, and H4a, the accuracy hypothesis suggested by the CC18 data, was not confirmed on the new data. TOP-Gain’s accuracy advantage thus appears to be broad: ΔAUC is positive on 24 of the 26 CC18 datasets and on all 24 AMLB datasets, and it is not concentrated in a structurally identifiable subgroup that we could have predicted.

The stability moderator, in contrast, was supported. The stability gain from truncation grows as the relevant features become sparser (Fig. 4), which replicates the exploratory CC18 correlation $( \rho ^ { \mathsf { \Delta } } ) ^ { \mathsf { \Delta } }$ −0.42) on independent data. For example, ΔΦ is +0.27 on porto-seguro, +0.28 on kick and +0.19 on KDDCup09\_appetency, all with $M _ { \mathrm { r e l } } \leq 0 . 0 2 5$ , but at most 0.01 on covertype, dilbert, robert and volkert, all with $M _ { \mathrm { r e l } } \geq 0 . 7 1$ . This is what the noise argument of Section 2.3 predicts: when few features are relevant, most candidates at each step are uninformative and offer similar, noisedominated gains, and this is the regime in which removing noise from the score matters most.

An exploratory analysis, which is post hoc and unadjusted, further showed that on AMLB the accuracy gain increased with non-linearity $\mathrm { M - n l ~ ( p = + 0 . 5 0 , } p = 0 . 0 1 1 ) \mathrm { ~ a n d ~ w i t h ~ l o g } _ { 1 0 } P \left( \mathrm { p = + } 0 . 5 2 \right)$ The direction for M-nl agrees with the unconfirmed CC18 hypothesis H1 $( \rho = + 0 . 3 7 )$ . We report these as observations to be tested in future work, not as findings.

![](images/3c208d8a4eeba53971d20b35b9277819086d3d9470f02ee47dea994d69626926.jpg)

![](images/133cc4c61bcc8f676f0c17c13702897ee192346465972c2b8935bd01e7c078f9.jpg)  
Fig. 4 Relevance density and the stability gain from truncation: ΔΦ = Φ(TOP-Gain) − Φ(net gain) per dataset against the relevance density M-rel, on CC18 (exploratory) and on the AMLB pool (confirmatory test of H4b)

## 4.4 Computational cost under identical settings

Table 6 reports the selection time of every method on one fold of each CC18 dataset under the two settings of Section 3.5.

Table 6 Selection time under identical settings (26 CC18 datasets, repeat 1, fold 1; one call per method and dataset). Total = sum over datasets; ratio = total time relative to TOP-Gain; speed-up = total time on one core divided by that on ten cores. varSelRF has no parallel option.
<table><tr><td>Method</td><td>Total, 1 core (s)</td><td>Median, 1 core (s)</td><td>Ratio to TOP- Gain, 1 core</td><td>Total, 10 cores (s)</td><td>Ratio to TOP-Gain, 10 cores</td><td>Speed-up 10 vs 1</td></tr><tr><td>TOP-Gain</td><td>1,075</td><td>17</td><td>1.00</td><td>165</td><td>1.00</td><td>6.5</td></tr><tr><td>VSURF</td><td>22,859</td><td>61</td><td>21.27</td><td>4,736</td><td>28.72</td><td>4.8</td></tr><tr><td>varSelRF</td><td>2,556</td><td>24</td><td>2.38</td><td>2,609</td><td>15.82</td><td>1.0</td></tr><tr><td>RF permutation</td><td>701</td><td>5.2</td><td>0.65</td><td>382</td><td>2.31</td><td>1.8</td></tr><tr><td>Boruta</td><td>206</td><td>4.7</td><td>0.19</td><td>125</td><td>0.76</td><td>1.6</td></tr><tr><td>RF-RFE</td><td>37</td><td>1.1</td><td>0.03</td><td>25</td><td>0.15</td><td>1.5</td></tr><tr><td>SVM-RFE</td><td>37</td><td>0.1</td><td>0.03</td><td>37</td><td>0.22</td><td>1.0</td></tr><tr><td>RF impurity</td><td>23</td><td>0.7</td><td>0.02</td><td>10</td><td>0.06</td><td>2.2</td></tr><tr><td>mRMR</td><td>22</td><td>0.3</td><td>0.02</td><td>21</td><td>0.13</td><td>1.1</td></tr><tr><td>JMI</td><td>18</td><td>0.3</td><td>0.02</td><td>17</td><td>0.10</td><td>1.1</td></tr><tr><td>Stability Selection</td><td>16</td><td>0.04</td><td>0.01</td><td>15</td><td>0.09</td><td>1.0</td></tr><tr><td>HSIC-Lasso</td><td>15</td><td>0.2</td><td>0.01</td><td>7.4</td><td>0.04</td><td>2.1</td></tr><tr><td>Elastic Net</td><td>3.7</td><td>0.05</td><td>0.00</td><td>3.7</td><td>0.02</td><td>1.0</td></tr><tr><td>ReliefF</td><td>0.5</td><td>0.01</td><td>0.00</td><td>0.6</td><td>0.00</td><td>0.8</td></tr><tr><td>ANOVA</td><td>0.07</td><td>0.00</td><td>0.00</td><td>0.07</td><td>0.00</td><td>0.9</td></tr></table>

VSURF, the only baseline as accurate as TOP-Gain, is the most expensive method in both settings. On one core it needs 21 times the total time of TOP-Gain (22,859 s versus 1,075 s), and on ten cores 29 times. TOP-Gain is faster on 25 of the 26 datasets on one core and on all 26 on ten cores, with median per-dataset ratios of 3.0 and 4.1. The gap grows with dimensionality: on Bioresponse $( P = 1 , 7 7 6 )$ VSURF needs 17,798 s on one core against 162 s for TOP-Gain, a factor of 110, and on Internet-Advertisements (� = 1,558) 3,139 s against 119 s.

The comparison with varSelRF depends on the setting. On one core the two methods cost about the same: varSelRF needs 2.4 times the total time of TOP-Gain, with a median per-dataset ratio of 1.3, and TOP-Gain is faster on 16 of the 26 datasets. With ten cores the ratio rises to 16, because TOP-Gain parallelises well whereas varSelRF does not. In fact, TOP-Gain has the largest parallel speed-up of all methods, 6.5 on ten cores against 4.8 for VSURF, since its candidate evaluations within a step are independent. The filters, the embedded methods and the single-fit importance rankings are much cheaper than every wrapper; Boruta and RF permutation importance cost about a fifth and two-thirds of TOP-Gain on one core.

Three further observations support these comparisons. For varSelRF and VSURF, the R-internal fitting time accounts for more than 99.7% of the measured wall-clock time on one core (2,549 of 2,556 s and 22,852 of 22,859 s), so the transfer of the data to R does not affect the results. The worker count did not change any TOP-Gain selection, which was identical under both settings on all 26 datasets, whereas VSURF’s selections differed on 24 of the 26 datasets because its parallel and sequential modes draw different random-number streams; the accuracy results of Section 4.1 refer to its parallel mode. Finally, on the AMLB pool, TOP-Gain’s selection time grew only slowly with $P ,$ from 2 s per fold at � = 20 to 65 s at $P = 7 1 , 6 4 0$ , because the candidate pool is bounded (Section 2.5); RF permutation importance, by contrast, needed about 10,000 s per fold at � = 71,640.

## 4.5 Accuracy, stability and cost

Fig. 5 combines the three aspects by placing every method according to its accuracy rank and stability rank on CC18, with a marker size that grows with its single-core selection time. Three groups emerge. The filters and importance rankings, such as ANOVA, RF impurity and Boruta, are stable and fast but less accurate. VSURF is accurate and stable but by far the most expensive method, while varSelRF is stable and moderately accurate at a single-core cost similar to that of TOP-Gain. TOP-Gain is the most accurate method at a moderate cost, with lower absolute stability. No method dominates TOP-Gain in accuracy: all methods that are more stable are less accurate, and the only method that is equally accurate, VSURF, needs 21 times more single-core computation.

![](images/457caae5d3d928657c2c4ae1610b4b193510553a2a04e0f170fba7bade3d5105.jpg)  
Fig. 5 Accuracy–stability–cost trade-off on CC18. Horizontal axis: Friedman mean rank of ROC-AUC; vertical axis: mean rank of the stability Φ (lower is better on both axes). Marker area grows with the single-core selection time (Table 6)

## 4.6 Why truncation helps: ranking reliability

To test the mechanism directly, we measured how reproducibly each criterion ranks the candidates at a fixed step of the search. We used six datasets (synthetic, qsar-biodeg, MiceProtein, splice, Bioresponse and OVA\_Breast) and fixed � to the first $s \in \{ 1 , 3 , 6 , 1 0 \}$ features chosen by TOP-Gain on the training part of the first fold. All remaining candidates were then scored 20 times under two kinds of perturbation: changing only the forest seeds, or drawing stratified 80% subsamples of the training data. Over the 190 pairs of replicates, we computed the mean pairwise Spearman correlation of the candidate scores, the agreement of the best candidate and of the five best candidates, and the intraclass correlation (ICC). As a diagnostic, we also scored the negative part that TOP-Gain discards.

Table 7 Reproducibility of the candidate ranking (mean over 6 datasets × 4 subset sizes; higher is better). † TOP-Gain significantly higher than every other criterion (Wilcoxon over 24 units, α = 0.05).
<table><tr><td>Criterion</td><td>Spearman (seed)</td><td>Spearman (data)</td><td>Top-5 (seed)</td><td>Top-5 (data)</td><td>Top-1 (seed)</td><td>Top-1 (data)</td><td>ICC (data)</td></tr><tr><td>TOP-Gain</td><td>0.617†</td><td>0.606†</td><td>0.509</td><td>0.423†</td><td>0.454</td><td>0.358</td><td>0.547†</td></tr><tr><td>Net probability gain</td><td>0.511</td><td>0.538</td><td>0.446</td><td>0.383</td><td>0.443</td><td>0.411</td><td>0.391</td></tr><tr><td>OOB Brier score</td><td>0.539</td><td>0.514</td><td>0.471</td><td>0.382</td><td>0.497</td><td>0.408</td><td>0.391</td></tr><tr><td>OOB log-loss</td><td>0.520</td><td>0.492</td><td>0.438</td><td>0.373</td><td>0.459</td><td>0.387</td><td>0.377</td></tr><tr><td>OOB accuracy</td><td>0.428</td><td>0.356</td><td>0.356</td><td>0.279</td><td>0.378</td><td>0.301</td><td>0.336</td></tr><tr><td>OOB balanced accuracy</td><td>0.417</td><td>0.344</td><td>0.348</td><td>0.281</td><td>0.388</td><td>0.306</td><td>0.330</td></tr><tr><td>Negative part (discarded)</td><td>0.434</td><td>0.486</td><td>0.257</td><td>0.250</td><td>0.227</td><td>0.220</td><td>0.424</td></tr></table>

TOP-Gain ranks the candidates most reproducibly under both perturbations (Table 7), with Spearman correlations of 0.62 and 0.61 against 0.34–0.54 for the alternatives $( p \leq 0 . 0 0 0 2 )$ . Under data perturbation, which corresponds to the fold-to-fold variation measured by Φ, it also has the most reproducible set of five leading candidates and the largest share of between-candidate variance (ICC 0.55 versus 0.33–0.39). Its agreement on the single best candidate, however, is not higher. The stability gain therefore does not come from a more reproducible winner at each step, but from a more reproducible group of leading candidates: even when the winner changes, the features added over successive steps overlap more, and this is what the set-level measure Φ rewards.

The discarded negative part is the least reproducible component of all, with top-1 and top-5 agreement of only 0.22–0.26. Subtracting it, as the net gain does, adds a mostly noisy term to an informative one. The advantage of TOP-Gain is largest in the middle of the search, where many candidates offer similar gains: under data perturbation its Spearman correlation is 0.79 versus 0.68 for the best alternative at $\vert S \vert = 3$ , and 0.50 versus 0.40 at $\left| S \right| = 6$ . This matches the moderate-noise regime of the toy simulation (Fig. 2).

## 4.7 Ground truth

Synthetic data with a known ground truth confirm that the more stable selections are also more correct. With ten informative features among 50, TOP-Gain recovered 10.0, 10.0, 9.8 and 9.4 informative features at class-imbalance ratios $\rho = 1$ , 5, 10 and 20, whereas the net gain recovered 9.6, 9.0, 9.0 and 7.8; TOP-Gain also had the highest stability at every ratio $( \Phi \ge 0 . 8 5 ;$ ; Online Resource 1, Table S3a). At the most extreme ratio, $\rho = 2 0$ , the proper scoring rules nevertheless gave higher minority recall and Macro-F1 than TOP-Gain (Macro-F1 0.742 for Brier versus 0.713), although they recovered fewer informative features. In an XOR benchmark, TOP-Gain recovered the most informative features for $K \leq 4$ , but, like every greedy forward criterion, it did not reliably recover the pure interaction pair.

## 4.8 Sensitivity to the surrogate configuration

Table 8 lists the variants of the surrogate configuration and their effects relative to the default.

Table 8 Sensitivity of TOP-Gain to its surrogate configuration. Each variant changes one setting of the default $( T = 4 0$ , depth 5, pool factor 1). Δ is the variant minus default TOP-Gain over datasets (W/L = wins and losses of the variant). $p _ { \mathrm { H o l m } }$ is the Wilcoxon � corrected over the variants. ${ ^ { 6 } } \mathrm { A U C }$ rank” is the variant’s Friedman rank of ROC-AUC when it replaces default TOP-Gain among the 15 CC18 methods. Time is the median selection time per fold (ten cores).
<table><tr><td>Variant</td><td>ΔAUC (W/L, pHolm)</td><td>∆Macro-F1  $\left( p _ { \mathrm { H o l m } } \right)$ </td><td> $\Delta \Phi [ 9 5 \% \mathrm { C I } ] \left( \mathrm { W } / \mathrm { L } , p _ { \mathrm { H o l m } } \right)$ </td><td>AUC rank (position)</td><td>Time/fold (s)</td></tr><tr><td>Default  $( T = 4 0 ,$  depth 5)</td><td></td><td></td><td></td><td>3.92 (1st)</td><td>4.9</td></tr><tr><td> $T = 2 0$ </td><td>-0.001 (8/18, 0.15)</td><td>−0.002 (0.20)</td><td>-0.061 [-0.078, -0.044] (2/24, &lt;0.001)</td><td>4.38 (2nd)</td><td>2.2</td></tr><tr><td> $T = 8 0$ </td><td>+0.001 (17/9, 0.15)</td><td>+0.000 (0.97)</td><td>+0.052 [0.040, 0.067] (26/0, &lt;0.001)</td><td>3.58 (1st)</td><td>7.1</td></tr><tr><td> $T = 1 6 0$ </td><td>+0.001 (18/8, 0.12)</td><td>+0.001 (0.43)</td><td>+0.105 [0.086, 0.124] (25/1, &lt;0.001)</td><td>3.62 (1st)</td><td>13.5</td></tr><tr><td>depth 3</td><td>−0.002 (6/20, 0.051)</td><td>−0.004 (0.001)</td><td>+0.038 [0.013, 0.064] (18/8, 0.009)</td><td>4.62 (2nd)</td><td>3.6</td></tr><tr><td>depth 8</td><td>+0.001 (16/10, 0.18)</td><td>+0.001 (0.97)</td><td>-0.036 [-0.059, -0.015] (7/19, 0.009)</td><td>3.42 (1st)</td><td>4.2</td></tr><tr><td>pool × 0.5 (AMLB,  $P > 1 5 0 )$ </td><td>−0.001 (6/10, 0.87)</td><td>−0.003 (0.70)</td><td>+0.005 [-0.018, 0.027] (9/7, 0.96)</td><td></td><td>13.9 (default 27.2)</td></tr><tr><td>Variant</td><td>ΔAUC (W/L,  $\underline { { p _ { \mathrm { H o l m } } ) } }$ </td><td>∆Macro-F1</td><td> $\Delta \Phi [ 9 5 \% \mathrm { C I } ] \left( \mathrm { W } / \mathrm { L } , p _ { \mathrm { H o l m } } \right)$ </td><td>AUC rank</td><td>Time/fold (s)</td></tr><tr><td> ${ \mathrm { p o o l } } \times 2 ( { \mathrm { A M L B } } , $ </td><td>-0.002 (4/9, 0.87)</td><td> $\underline { { ( p _ { \mathrm { H o l m } } ) } }$  −0.001 (0.86)</td><td>-0.002 [-0.016, 0.015] (4/8, 0.96)</td><td>(position)</td><td>39.7 (default 27.2)</td></tr></table>

Predictive performance is insensitive to the surrogate configuration. Every change of � between 20 and 160 or of the depth between 3 and 8 moves the mean ROC-AUC by at most 0.0023, and no ROC-AUC difference is significant after Holm correction. With every variant TOP-Gain remains first or second among the 15 CC18 methods, and the only method ever ahead of it is VSURF; only the shallowest trees (depth 3) cost slightly more accuracy and Macro-F1 than the other settings.

Stability, in contrast, depends on �, as the noise mechanism of Section 2.3 predicts, since more trees give each instance more OOB trees and thus less estimation noise. Relative to $T = 4 0 ,$ , Φ changes by −0.061 at � = 20, +0.052 at $T = 8 0$ and +0.105 at $T = 1 6 0$ . At $T = 1 6 0$ , TOP-Gain moves from the 15th to the 10th stability rank among the 15 methods, at 2.8 times the selection time of the default. Deeper trees (depth 8) reduce stability and shallower trees (depth 3) increase it, while the candidatepool size has no detectable effect on accuracy or stability, so halving the pool halves the cost at no loss.

These results show that $T = 4 0$ is a reasonable compromise between cost and accuracy that does not favour TOP-Gain’s accuracy results, and that the number of trees is a direct handle on stability that users can turn up when the reproducibility of the selected set matters more than selection time. Table 8 compares the variants only with default TOP-Gain, because the net-gain criterion was run only at $T =$ 40 and a comparison with it would not isolate the effect of truncation at other values of �.

## 5 Discussion

The results support one consistent account of what truncation does. At a typical forward step, most candidates change the OOB probabilities of most instances only by noise. Criteria that average these changes, whether the net gain, the Brier score or the log-loss, let this noise reorder near-equivalent candidates from one fold to the next. Truncation removes most of that noise variance while keeping the instances that a candidate truly improves (Section 2.3). The effect is visible at three levels: in the reproducibility of the candidate ranking (Section 4.6), in the stability of the selected sets on 50 datasets from two independent suites (Sections 4.2 and 4.3), and in a confirmed moderator, as the gain is largest when relevant features are sparse and most candidates are therefore noise (H4b). Truncation does not cost accuracy: TOP-Gain was never less accurate than the net gain and was equal to or better than the proper scoring rules.

The noise argument of Section 2.3 rests on the assumption that negative changes in the OOB probabilities are mostly estimation noise, and TOP-Gain indeed does not penalise real harm. Suppose candidate A improves every instance by 0.02, whereas candidate B improves 30% of the instances by 0.10 and worsens the other 70% by 0.10. The net gain is 0.020 for A and −0.040 for B, but TOP-Gain is 0.020 for A and 0.030 for B, so TOP-Gain prefers the harmful B. Our results suggest that such cases are rare in practice. The negative part that TOP-Gain discards was the least reproducible component of the score (Section 4.6), which indicates that it consists mostly of noise rather than harm; a Random Forest makes little use of a feature in the regions where it does not help, so large systematic harm from a single added feature is uncommon; and TOP-Gain was never less accurate than the net gain on any of the two benchmark suites (Sections 4.2, 4.3). The assumption can nevertheless fail under severe class imbalance, where real harm to the few minority instances is outweighed by small gains on the majority class. At an imbalance ratio of 20, the proper scoring rules indeed gave higher minority recall and Macro-F1 than TOP-Gain, although TOP-Gain recovered more of the informative features (Section 4.7). A class-weighted variant of TOP-Gain, which would give the minority instances more weight in the truncated sum, is a possible remedy that we have not tested.

The RF-OOB forward-selection framework itself is not new [22]; our contribution is the criterion and the evidence for it. In practical terms, TOP-Gain reaches the accuracy of the best RF wrapper we tested, VSURF, at a small fraction of its cost, and it is significantly more accurate than the 13 other baselines on CC18 and all 12 baselines on AMLB. Its accuracy advantage is broad rather than tied to an identifiable type of dataset, since neither the pre-registered H1–H3 nor H4a was confirmed.

TOP-Gain is therefore a good choice when the predictive performance of a small panel is the main goal and selection times of seconds to minutes per fold are acceptable. When the exact identity of the selected features matters more than accuracy, more stable but less accurate rankings, such as RF impurity, ANOVA or Boruta, may be preferable. TOP-Gain’s absolute stability is low on CC18 and intermediate on AMLB, but it can be raised without loss of accuracy by a larger surrogate forest; with � = 160 , Φ increases by 0.10 at 2.8 times the cost (Section 4.8). Combining TOP-Gain with subsample aggregation, as in Stability Selection [12] or randomised forward selection [35], is a natural further step and is left for future work.

The study has several limitations. We examined the criterion only within greedy forward selection; floating or beam search might change the absolute numbers, and although the criterion does not depend on the search, we did not test other search strategies. As discussed above, TOP-Gain ignores real harm, which can matter under severe class imbalance. TOP-Gain is the most stable wrapper criterion but not the most stable selector (Sections 4.1 and 4.3), and, like all greedy forward methods, it cannot detect features that are informative only jointly until one of them has been selected. The scope of the comparisons is also limited in three respects: varSelRF and VSURF were run on CC18 only, so the comparison with VSURF rests on 26 datasets; the absolute stability values hold for the default � = 40, because stability depends on � (Section 4.8); and all datasets were subsampled to at most 1,500 instances. Finally, CC18 and AMLB contain few small-sample, very high-dimensional datasets. In our original benchmark, which includes microarray and proteomics data, TOP-Gain also ranked first, but with smaller and less significant margins (Online Resource 1, Section S2).

## 6 Conclusion

TOP-Gain scores a candidate feature by the truncated sample-level gain in the out-of-bag Random Forest probability of the true class, and in doing so it removes most of the estimation noise that other continuous criteria accumulate. In a pre-registered benchmark on 26 CC18 datasets and a confirmatory study on 24 AutoML Benchmark datasets, TOP-Gain achieved the best predictive rank and was significantly better than 13 of 14 and 12 of 12 baselines, respectively, while matching the accuracy of VSURF at a fraction of its cost. Within an identical wrapper, it gave the most stable selections of six criteria at equal or better accuracy, and, as hypothesised in advance, its stability gain was largest when relevant features were sparse. Because it requires only a one-line change, we recommend TOP-Gain as the default criterion for Random-Forest forward wrapper selection.

## Declarations

Funding. This research was supported by a research fund from the Dankook University in 2025

Competing interests. The authors declare no competing interests.

Data availability. All datasets are publicly available on OpenML; their dataset IDs are listed in Online Resource 1 (Table S1). The per-fold results, the screening data and the protocol with all amendments are available in the code repository.

Code availability. The complete code is available at https://github.com/dkumango/TOP\_Gain. It covers the selectors, benchmark runners, analysis scripts and the pre-registration protocol. The main scripts are:

• topg\_dataset\_screening.py: eligibility rules and moderators;

topg\_run\_cc18.py: CC18 and AMLB benchmarks;

• topg\_analyze\_cc18.py and topg\_analyze\_amlb.py: pre-registered analyses;

• topg\_runtime\_benchmark.py: runtime study;

• topg\_run\_sensitivity.py: sensitivity study.

Ethics approval and consent. Not applicable. The study uses only public, de-identified benchmark datasets.

Author contributions. Oh: Conceptualization, Methodology, Funding acquisition, Investigation, Project administration, Validation, Writing – review and editing.

Use of generative AI. The use of an AI assistant is described in Section 3.7.

## References

1. I. Guyon and A. Elisseeff, “An introduction to variable and feature selection,” J. Mach. Learn. Res., vol. 3, pp. 1157–1182, 2003.

2. Y. Saeys, I. Inza, and P. Larrañaga, “A review of feature selection techniques in bioinformatics,” Bioinformatics, vol. 23, no. 19, pp. 2507–2517, 2007, https://doi.org/10.1093/bioinformatics/btm344

3. L. M. McShane et al., “Criteria for the use of omics-based predictors in clinical trials,” Nature, vol. 502, no. 7471, pp. 317–320, 2013, https://doi.org/10.1038/nature12564

4. A. Kalousis, J. Prados, and M. Hilario, “Stability of feature selection algorithms: a study on high-dimensional spaces,” Knowl. Inf. Syst., vol. 12, no. 1, pp. 95–116, 2007, https://doi.org/10.1007/s10115-006-0040-8

5. S. Nogueira, K. Sechidis, and G. Brown, “On the stability of feature selection algorithms,” J. Mach. Learn. Res., vol. 18, no. 174, pp. 1–54, 2018.

6. H. Peng, F. Long, and C. Ding, “Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 27, no. 8, pp. 1226–1238, 2005, https://doi.org/10.1109/TPAMI.2005.159

7. H. H. Yang and J. Moody, “Data visualization and feature selection: new algorithms for nongaussian data,” in Adv. Neural Inf. Process. Syst. 12, 2000, pp. 687–693.

8. M. Robnik-Šikonja and I. Kononenko, “Theoretical and empirical analysis of ReliefF and RReliefF,” Mach. Learn., vol. 53, no. 1–2, pp. 23–69, 2003, https://doi.org/10.1023/A:1025667309714

9. H. Zou and T. Hastie, “Regularization and variable selection via the elastic net,” J. R. Stat. Soc. B, vol. 67, no. 2, pp. 301–320, 2005, https://doi.org/10.1111/j.1467-9868.2005.00503.x

10. M. Yamada, W. Jitkrittum, L. Sigal, E. P. Xing, and M. Sugiyama, “High-dimensional feature selection by feature-wise kernelized Lasso,” Neural Comput., vol. 26, no. 1, pp. 185–207, 2014, https://doi.org/10.1162/NECO\_a\_00537

11. I. Lemhadri, F. Ruan, L. Abraham, and R. Tibshirani, “LassoNet: a neural network with feature sparsity,” J. Mach. Learn. Res., vol. 22, no. 127, pp. 1–29, 2021.

12. N. Meinshausen and P. Bühlmann, “Stability selection,” J. R. Stat. Soc. B, vol. 72, no. 4, pp. 417–473, 2010, https://doi.org/10.1111/j.1467-9868.2010.00740.x

13. V. Bolón-Canedo and A. Alonso-Betanzos, “Ensembles for feature selection: a review and future trends,” Inf. Fusion, vol. 52, pp. 1–12, 2019, https://doi.org/10.1016/j.inffus.2018.11.008

14. R. Kohavi and G. H. John, “Wrappers for feature subset selection,” Artif. Intell., vol. 97, no. 1–2, pp. 273–324, 1997, https://doi.org/10.1016/S0004-3702(97)00043-X

15. L. Breiman, “Random forests,” Mach. Learn., vol. 45, no. 1, pp. 5–32, 2001, https://doi.org/10.1023/A:1010933404324

16. A. Altmann, L. Toloşi, O. Sander, and T. Lengauer, “Permutation importance: a corrected feature importance measure,” Bioinformatics, vol. 26, no. 10, pp. 1340–1347, 2010, https://doi.org/10.1093/bioinformatics/btq134

17. C. Strobl, A.-L. Boulesteix, A. Zeileis, and T. Hothorn, “Bias in random forest variable importance measures: illustrations, sources and a solution,” BMC Bioinformatics, vol. 8, Art. no. 25, 2007, https://doi.org/10.1186/1471-2105-8-25

18. B. Gregorutti, B. Michel, and P. Saint-Pierre, “Correlation and variable importance in random forests,” Stat. Comput., vol. 27, no. 3, pp. 659–678, 2017, https://doi.org/10.1007/s11222-016- 9646-1

19. R. Díaz-Uriarte and S. Alvarez de Andrés, “Gene selection and classification of microarray data using random forest,” BMC Bioinformatics, vol. 7, Art. no. 3, 2006, https://doi.org/10.1186/1471-2105-7-3

20. R. Genuer, J.-M. Poggi, and C. Tuleau-Malot, “Variable selection using random forests,” Pattern Recognit. Lett., vol. 31, no. 14, pp. 2225–2236, 2010, https://doi.org/10.1016/j.patrec.2010.03.014

21. M. B. Kursa and W. R. Rudnicki, “Feature selection with the Boruta package,” J. Stat. Softw., vol. 36, no. 11, pp. 1–13, 2010, https://doi.org/10.18637/jss.v036.i11

22. J. Velthoen, J.-J. Cai, and G. Jongbloed, “Forward variable selection for random forest models,” J. Appl. Stat., vol. 50, no. 13, pp. 2836–2856, 2023, https://doi.org/10.1080/02664763.2022.2095362

23. G. W. Brier, “Verification of forecasts expressed in terms of probability,” Mon. Weather Rev., vol. 78, no. 1, pp. 1–3, 1950, https://doi.org/10.1175/1520- 0493(1950)078<0001:VOFEIT>2.0.CO;2

24. T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” J. Am. Stat. Assoc., vol. 102, no. 477, pp. 359–378, 2007, https://doi.org/10.1198/016214506000001437

25. D. S. Watson and M. N. Wright, “Testing conditional independence in supervised learning algorithms,” Mach. Learn., vol. 110, pp. 2107–2129, 2021, https://doi.org/10.1007/s10994- 021-06030-6

26. E. Candès, Y. Fan, L. Janson, and J. Lv, “Panning for gold: ‘model-X’ knockoffs for high dimensional controlled variable selection,” J. R. Stat. Soc. B, vol. 80, no. 3, pp. 551–577, 2018, https://doi.org/10.1111/rssb.12265

27. S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Adv. Neural Inf. Process. Syst. 30, 2017, pp. 4765–4774.

28. A. Danyluk and N. Arnosti, “Feature selection via probabilistic outputs,” in Proc. 29th Int. Conf. Mach. Learn. (ICML), 2012, pp. 971 - 978.

29. S. Oh, “Predictive case-based feature importance and interaction,” Inf. Sci., vol. 593, pp. 155– 176, 2022, https://doi.org/10.1016/j.ins.2022.02.003

30. D. R. Jones, M. Schonlau, and W. J. Welch, “Efficient global optimization of expensive blackbox functions,” J. Glob. Optim., vol. 13, no. 4, pp. 455–492, 1998, https://doi.org/10.1023/A:1008306431147

31. B. Bischl et al., “OpenML benchmarking suites,” in Proc. NeurIPS Datasets and Benchmarks Track, 2021.

32. P. Gijsbers et al., “AMLB: an AutoML benchmark,” J. Mach. Learn. Res., vol. 25, no. 101, pp. 1–65, 2024.

33. I. Guyon, J. Weston, S. Barnhill, and V. Vapnik, “Gene selection for cancer classification using support vector machines,” Mach. Learn., vol. 46, no. 1–3, pp. 389–422, 2002, https://doi.org/10.1023/A:1012487302797

34. J. Demšar, “Statistical comparisons of classifiers over multiple data sets,” J. Mach. Learn. Res., vol. 7, pp. 1–30, 2006.

35. L. Mentch and S. Zhou, “Randomization as regularization: a degrees of freedom explanation for random forest success,” J. Mach. Learn. Res., vol. 21, no. 171, pp. 1–36, 2020.

## Online Resource 1

## Truncated Out-of-Bag Probability Gain: A Noise-Robust Criterion for Random-Forest Wrapper Feature Selection

Knowledge and Information Systems

[Author names] · Corresponding author: [Name], [Affiliation, City, Country], [e-mail]

This Online Resource contains:

• S1 the dataset lists;

• S2 the original ten-dataset benchmark;

• S3 the ground-truth and controlled-imbalance studies;

• S4 the robustness and ablation checks;

• S5 the results per downstream classifier;

• S6 the protocol amendments.

## S1 Datasets

Tables S1a and S1b list the eligible datasets of the two suites with their moderators, sorted by the number of features.

Table S1a CC18 datasets (26 eligible; sorted by �). ID = OpenML dataset ID; � = instances in the original data; $N _ { \mathrm { u s e d } } =$ instances used after subsampling to at most 1,500; � = classes; Imb. = majority/minority ratio; moderators as defined in Section 3.1 of the main text.

<table><tr><td>Dataset</td><td>ID</td><td>N</td><td> $N _ { \mathrm { u s e d } }$ </td><td>P</td><td>C</td><td>Imb.</td><td>M-rel</td><td>M-nl</td><td>M-red</td><td>M-head</td></tr><tr><td>climate-model-simulation-crashes</td><td>40994</td><td>540</td><td>540</td><td>18</td><td>2</td><td>10.7</td><td>0.22</td><td>-0.05</td><td>0.02</td><td>0.10</td></tr><tr><td>pcl</td><td>1068</td><td>1,109</td><td>1,109</td><td>21</td><td>2</td><td>13.4</td><td>0.95</td><td>0.02</td><td>0.90</td><td>0.15</td></tr><tr><td>kc2</td><td>1063</td><td>522</td><td>522</td><td>21</td><td>2</td><td>3.9</td><td>1.00</td><td>-0.02</td><td>0.91</td><td>0.19</td></tr><tr><td>kcl</td><td>1067</td><td>2,109</td><td>1,500</td><td>21</td><td>2</td><td>5.5</td><td>0.95</td><td>0.02</td><td>0.90</td><td>0.17</td></tr><tr><td>jm1</td><td>1053</td><td>10,885</td><td>1,500</td><td>21</td><td>2</td><td>4.2</td><td>1.00</td><td>-0.02</td><td>0.89</td><td>0.33</td></tr><tr><td>steel-plates-fault</td><td>40982</td><td>1,941</td><td>1,500</td><td>27</td><td>7</td><td>12.4</td><td>1.00</td><td>0.03</td><td>0.83</td><td>0.04</td></tr><tr><td>churn</td><td>40701</td><td>5,000</td><td>1,500</td><td>29</td><td>2</td><td>6.1</td><td>0.46</td><td>0.06</td><td>0.47</td><td>0.08</td></tr><tr><td>sick</td><td>38</td><td>3,772</td><td>1,500</td><td>30</td><td>2</td><td>15.3</td><td>0.38</td><td>0.03</td><td>0.41</td><td>0.01</td></tr><tr><td>wdbc</td><td>1510</td><td>569</td><td>569</td><td>30</td><td>2</td><td>1.7</td><td>0.83</td><td>-0.01</td><td>0.85</td><td>0.01</td></tr><tr><td>GesturePhaseSegmentationProcessed</td><td>4538</td><td>9,873</td><td>1,500</td><td>32</td><td>5</td><td>2.9</td><td>0.56</td><td>0.07</td><td>0.75</td><td>0.18</td></tr><tr><td>satimage</td><td>182</td><td>6,430</td><td>1,500</td><td>36</td><td>6</td><td>2.4</td><td>1.00</td><td>0.01</td><td>0.94</td><td>0.02</td></tr><tr><td>pc3</td><td>1050</td><td>1,563</td><td>1,500</td><td>37</td><td>2</td><td>8.7</td><td>0.86</td><td>0.03</td><td>0.90</td><td>0.16</td></tr><tr><td>pc4</td><td>1049</td><td>1,458</td><td>1,458</td><td>37</td><td>2</td><td>7.2</td><td>0.84</td><td>0.04</td><td>0.86</td><td>0.06</td></tr><tr><td>PhishingWebsites</td><td>4534</td><td>11,055</td><td>1,500</td><td>38</td><td>2</td><td>1.3</td><td>0.71</td><td>-0.00</td><td>0.55</td><td>0.02</td></tr><tr><td>qsar-biodeg</td><td>1494</td><td>1,055</td><td>1,055</td><td>41</td><td>2</td><td>2.0</td><td>0.83</td><td>0.01</td><td>0.70</td><td>0.06</td></tr><tr><td>mfeat-zernike</td><td>22</td><td>2,000</td><td>1,500</td><td>47</td><td>10</td><td>1.0</td><td>1.00</td><td>-0.01</td><td>0.87</td><td>0.03</td></tr><tr><td>credit-g</td><td>31</td><td>1,000</td><td>1,000</td><td>48</td><td>2</td><td>2.3</td><td>0.50</td><td>0.00</td><td>0.40</td><td>0.21</td></tr><tr><td>first-order-theorem-proving</td><td>1475</td><td>6,118</td><td>1,500</td><td>51</td><td>6</td><td>5.3</td><td>0.78</td><td>0.07</td><td>0.78</td><td>0.22</td></tr><tr><td>spambase</td><td>44</td><td>4,601</td><td>1,500</td><td>57</td><td>2</td><td>1.5</td><td>0.91</td><td>0.01</td><td>0.46</td><td>0.02</td></tr><tr><td>ozone-level-8hr</td><td>1487</td><td>2,534</td><td>1,500</td><td>72</td><td>2</td><td>14.8</td><td>0.89</td><td>0.02</td><td>0.88</td><td>0.10</td></tr><tr><td>mfeat-fourier</td><td>14</td><td>2,000</td><td>1,500</td><td>76</td><td>10</td><td>1.0</td><td>1.00</td><td>0.01</td><td>0.43</td><td>0.02</td></tr><tr><td>cylinder-bands</td><td>6332</td><td>540</td><td>540</td><td>143</td><td>2</td><td>1.4</td><td>0.15</td><td>0.06</td><td>0.35</td><td>0.10</td></tr><tr><td>nomao</td><td>1486</td><td>34,465</td><td>1,500</td><td>145</td><td>2</td><td>2.5</td><td>0.87</td><td>0.00</td><td>0.95</td><td>0.02</td></tr><tr><td>madelon</td><td>1485</td><td>2,600</td><td>1,500</td><td>500</td><td>2</td><td>1.0</td><td>0.03</td><td>0.19</td><td>0.12</td><td>0.26</td></tr><tr><td>Internet-Advertisements</td><td>40978</td><td>3,279</td><td>1,500</td><td>1,558</td><td>2</td><td>6.1</td><td>0.25</td><td>0.00</td><td>0.92</td><td>0.02</td></tr><tr><td>Bioresponse</td><td>4134</td><td>3,751</td><td>1,500</td><td>1,776</td><td>2</td><td>1.2</td><td>0.24</td><td>0.10</td><td>0.68</td><td>0.16</td></tr></table>

Table S1b AMLB datasets (24 eligible; OpenML suite 271 minus CC18; sorted by �).
<table><tr><td>Dataset</td><td>ID</td><td>N</td><td> $N _ { \mathrm { u s e d } }$ </td><td>P</td><td>C</td><td>Imb.</td><td>M-rel</td><td>M-nl</td><td>M-red</td><td>M-head</td></tr><tr><td>sylvine</td><td>41146</td><td>5,124</td><td>1,500</td><td>20</td><td>2</td><td>1.0</td><td>0.30</td><td>0.01</td><td>0.33</td><td>0.03</td></tr><tr><td>Higgs</td><td>42769</td><td>1,000,000</td><td>1,500</td><td>28</td><td>2</td><td>1.1</td><td>0.21</td><td>0.03</td><td>0.33</td><td>0.29</td></tr><tr><td>Satellite</td><td>40900</td><td>5,100</td><td>1,500</td><td>36</td><td>2</td><td>67.2</td><td>0.75</td><td>-0.02</td><td>0.94</td><td>0.04</td></tr><tr><td>ada</td><td>41156</td><td>4,147</td><td>1,500</td><td>48</td><td>2</td><td>3.0</td><td>0.67</td><td>0.00</td><td>0.34</td><td>0.10</td></tr><tr><td>MiniBooNE</td><td>41150</td><td>130,064</td><td>1,500</td><td>50</td><td>2</td><td>2.6</td><td>0.10</td><td>0.05</td><td>0.68</td><td>0.04</td></tr><tr><td>covertype</td><td>1596</td><td>581,012</td><td>1,500</td><td>54</td><td>7</td><td>104.4</td><td>0.71</td><td>0.01</td><td>0.30</td><td>0.09</td></tr><tr><td>jannis</td><td>41168</td><td>83,733</td><td>1,500</td><td>54</td><td>4</td><td>23.0</td><td>0.46</td><td>0.03</td><td>0.42</td><td>0.21</td></tr><tr><td>jasmine</td><td>41143</td><td>2,984</td><td>1,500</td><td>144</td><td>2</td><td>1.0</td><td>0.40</td><td>0.04</td><td>0.38</td><td>0.14</td></tr><tr><td>APSFailure</td><td>41138</td><td>76,000</td><td>1,500</td><td>170</td><td>2</td><td>54.6</td><td>0.70</td><td>0.07</td><td>0.82</td><td>0.02</td></tr><tr><td>volkert</td><td>41166</td><td>58,310</td><td>1,500</td><td>180</td><td>10</td><td>9.4</td><td>0.95</td><td>0.04</td><td>0.84</td><td>0.13</td></tr><tr><td>porto-seguro</td><td>42742</td><td>595,212</td><td>1,500</td><td>204</td><td>2</td><td>26.3</td><td>0.02</td><td>0.13</td><td>0.29</td><td>0.35</td></tr><tr><td>madeline</td><td>41144</td><td>3,140</td><td>1,500</td><td>259</td><td>2</td><td>1.0</td><td>0.04</td><td>0.23</td><td>0.14</td><td>0.22</td></tr><tr><td>philippine</td><td>41145</td><td>5,832</td><td>1,500</td><td>308</td><td>2</td><td>1.0</td><td>0.39</td><td>0.04</td><td>0.51</td><td>0.20</td></tr><tr><td>fabert</td><td>41164</td><td>8,237</td><td>1,500</td><td>800</td><td>7</td><td>3.9</td><td>0.22</td><td>0.03</td><td>0.37</td><td>0.15</td></tr><tr><td>gina</td><td>41158</td><td>3,153</td><td>1,500</td><td>970</td><td>2</td><td>1.0</td><td>0.38</td><td>0.07</td><td>0.81</td><td>0.03</td></tr><tr><td>christine</td><td>41142</td><td>5,418</td><td>1,500</td><td>1,611</td><td>2</td><td>1.0</td><td>0.40</td><td>0.10</td><td>0.49</td><td>0.22</td></tr><tr><td>dilbert</td><td>41163</td><td>10,000</td><td>1,500</td><td>2,000</td><td>5</td><td>1.1</td><td>0.84</td><td>-0.00</td><td>0.82</td><td>0.02</td></tr><tr><td>kick</td><td>41162</td><td>72,983</td><td>1,500</td><td>2,408</td><td>2</td><td>7.2</td><td>0.02</td><td>0.07</td><td>0.61</td><td>0.36</td></tr><tr><td>riccardo</td><td>41161</td><td>20,000</td><td>1,500</td><td>4,296</td><td>2</td><td>3.0</td><td>0.02</td><td>0.07</td><td>0.45</td><td>0.07</td></tr><tr><td>guillermo</td><td>41159</td><td>20,000</td><td>1,500</td><td>4,296</td><td>2</td><td>1.5</td><td>0.01</td><td>-0.01</td><td>0.47</td><td>0.36</td></tr><tr><td>robert</td><td>41165</td><td>10,000</td><td>1,500</td><td>7,200</td><td>10</td><td>1.1</td><td>0.90</td><td>0.04</td><td>0.86</td><td>0.21</td></tr><tr><td>amazon-commerce-reviews</td><td>1457</td><td>1,500</td><td>1,500</td><td>10,000</td><td>50</td><td>1.0</td><td>0.70</td><td>-0.04</td><td>0.56</td><td>0.05</td></tr><tr><td>arcene</td><td>41157</td><td>100</td><td>100</td><td>10,000</td><td>2</td><td>1.3</td><td>0.08</td><td>-0.04</td><td>0.81</td><td>0.16</td></tr><tr><td>KDDCup09_appetency</td><td>1111</td><td>50,000</td><td>1,500</td><td>71,640</td><td>2</td><td>54.6</td><td>0.02</td><td>0.09</td><td>0.93</td><td>0.34</td></tr></table>

## S2 Original ten-dataset benchmark, including $\mathrm { P } \gg \mathrm { N }$ data

Datasets. The original benchmark used ten biomedical and life-science datasets from OpenML:

a synthetic benchmark;

• mouse protein expression;

• three tumour-expression microarray tasks;

• ovarian-cancer proteomics;

• three molecular or sequence datasets.

These datasets have up to 15,154 features and include microarray and proteomics data with $N \ll P .$

Design.

• Selectors. TOP-Gain was compared with nine baselines (ANOVA, mRMR, JMI, ReliefF, Elastic Net, HSIC-Lasso, Stability Selection, Boruta, SVM-RFE) and with the five alternative criteria.

Panel sizes. For $P \leq 5 0 0 , K = \operatorname* { m a x } ( 2 , \lfloor r P \rfloor )$ with the fraction $r \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 \}$ of the features. For $P > 5 0 0 , K \in \{ 1 0 , 2 0 , 3 0 , 5 0 \}$

Protocol. The four downstream classifiers and the statistics are those of the main text. Five repeats of 5-fold cross-validation were run, with seeds 42, 1, 2, 3 and 4.

## Results (Table S2a and Table S2b).

TOP-Gain had the best Friedman mean rank for ROC-AUC (2.90), accuracy (2.80) and Macro-F1 (2.90) among ten methods (Friedman $p \leq 0 . 0 1 8 )$

Its mean ROC-AUC exceeded that of every baseline by 0.006–0.028, with bootstrap CIs excluding zero. With ten datasets, however, the difference was significant after Holm correction only against SVM-RFE and ReliefF.

• All six criteria had equal accuracy (Friedman $p \geq 0 . 3 9 )$ . TOP-Gain was the most stable criterion (mean rank 1.40) and significantly more stable than Brier, log-loss, accuracy and balanced accuracy $( p _ { \mathrm { H o l m } } \le 0 . 0 2 7 )$ . It was more stable than the net gain on 7 of 10 datasets $( \Delta \Phi + 0 . 0 2 6 , p _ { \mathrm { H o l m } } = 0 . 1 1 )$

Table S2a Original benchmark, TOP-Gain vs nine baselines (10 datasets × 5 repeats; Friedman ranks among 10 methods, CD = 4.28).
<table><tr><td>Method</td><td>Rank AUC ↓</td><td>Rank Φ↓</td><td>∆AUC [95% CI]  $( \mathrm { W } / \mathrm { L } , p _ { \mathrm { H o l m } } )$ </td><td>∆Φ [95% CI]  $\underline { { ( \mathrm { W / L } , p _ { \mathrm { H o l m } } ) } }$ </td></tr><tr><td>TOP-Gain</td><td>2.90</td><td>7.90</td><td></td><td></td></tr><tr><td>ANOVA</td><td>6.50</td><td>1.50</td><td>+0.010 [0.003, 0.019] (7/3, 0.16)</td><td>-0.426 [-0.511, -0.341] (0/10, 0.02)</td></tr><tr><td>mRMR</td><td>5.40</td><td>7.40</td><td>+0.018 [0.001, 0.048] (7/3, 0.17)</td><td>-0.040 [-0.216, 0.145] (4/6, 1.00)</td></tr><tr><td>JMI</td><td>5.20</td><td>6.00</td><td>+0.013 [0.001, 0.032] (8/2, 0.16)</td><td>-0.113 [-0.296, 0.076] (5/5, 1.00)</td></tr><tr><td>ReliefF</td><td>7.40</td><td>6.70</td><td>+0.020 [0.004, 0.045] (9/1, 0.03)</td><td>-0.077 [-0.236, 0.078] (4/6, 1.00)</td></tr><tr><td>ElasticNet</td><td>5.60</td><td>4.60</td><td>+0.007 [0.002, 0.014] (8/2, 0.16)</td><td>-0.201 [-0.301, -0.109] (0/10, 0.02)</td></tr><tr><td>HSIC-Lasso</td><td>3.60</td><td>4.90</td><td>+0.006 [0.001, 0.014] (7/3, 0.17)</td><td>-0.175 [-0.304, -0.041] (2/8, 0.24)</td></tr><tr><td>Stability Selection</td><td>4.90</td><td>5.30</td><td>+0.009 [0.002, 0.017] (7/3, 0.16)</td><td>-0.165 [-0.237, -0.101] (0/10, 0.02)</td></tr><tr><td>Boruta</td><td>5.90</td><td>2.70</td><td>+0.022 [0.006, 0.043] (8/2, 0.10)</td><td>-0.336 [-0.389, -0.284] (0/10, 0.02)</td></tr><tr><td>SVM-RFE</td><td>7.60</td><td>8.00</td><td>+0.028 [0.009, 0.050] (10/0, 0.02)</td><td>+0.028 [-0.063, 0.121] (6/4, 1.00)</td></tr></table>

Table S2b Original benchmark, criterion comparison (Friedman ranks among 6 criteria, CD = 2.38).
<table><tr><td>Method</td><td>Rank AUC ↓</td><td>Rank Φ ↓</td><td> $\Delta \mathrm { A U C } \ [ 9 5 \% \mathrm { C I } ] \left( \mathrm { W } / \mathrm { L } , p _ { \mathrm { H o l m } } \right)$ </td><td> $\Delta \Phi [ 9 5 \% \mathrm { C I } ] \left( \mathrm { W } / \mathrm { L } , p _ { \mathrm { H o l m } } \right)$ </td></tr><tr><td>TOP-Gain</td><td>3.90</td><td>1.40</td><td></td><td></td></tr><tr><td>Net probability gain</td><td>2.70</td><td>2.30</td><td>-0.001 [-0.002, 0.000] (4/6, 1.00)</td><td>+0.025 [-0.002, 0.054] (7/3, 0.11)</td></tr><tr><td>OOB Brier score</td><td>3.00</td><td>3.60</td><td>-0.000 [-0.002, 0.002] (3/7, 1.00)</td><td>+0.074 [0.044, 0.105] (10/0, 0.01)</td></tr><tr><td>OOB log-loss</td><td>3.30</td><td>3.40</td><td>-0.000 [-0.002, 0.001] (4/6, 1.00)</td><td>+0.074 [0.029, 0.120] (9/1, 0.03)</td></tr><tr><td>OOB accuracy</td><td>3.80</td><td>5.00</td><td>+0.001 [-0.001, 0.002] (5/5, 1.00)</td><td>+0.146 [0.108, 0.184] (10/0, 0.01)</td></tr><tr><td>OOB balanced accuracy</td><td>4.30</td><td>5.30</td><td>+0.001 [-0.001, 0.002] (5/5, 1.00)</td><td>+0.161 [0.126, 0.203] (10/0, 0.01)</td></tr></table>

## S3 Ground-truth and controlled-imbalance studies

## Synthetic imbalance sweep.

Data. $N = 2 { , } 0 0 0 , P = 5 0$ . Ten features are informative (a mean shift of the minority class) and 40 are noise.

• Design. Imbalance ratios $\rho \in \{ 1 , 5 , 1 0 , 2 0 \}$ and $K = 1 0$ . The downstream classifier is a classbalanced RF.

• Measures. Minority recall, balanced accuracy, Macro-F1, ROC-AUC, Φ, and the number of informative features selected (TP).

Table S3a Synthetic imbalance sweep $( K = 1 0 ;$ ; 10 informative features among 50). TP = number of informative features selected (mean over five folds).
<table><tr><td>ρ</td><td>Criterion</td><td>Minority recall</td><td>Bal. acc.</td><td>Macro-F1</td><td>ROC-AUC</td><td>TP (/10)</td><td>Φ</td></tr><tr><td>1</td><td>TOP-Gain</td><td>0.914</td><td>0.922</td><td>0.922</td><td>0.980</td><td>10.0</td><td>1.000</td></tr><tr><td>1</td><td>Net gain</td><td>0.906</td><td>0.918</td><td>0.917</td><td>0.977</td><td>9.6</td><td>0.900</td></tr><tr><td>1</td><td>OOB Brier</td><td>0.917</td><td>0.920</td><td>0.920</td><td>0.977</td><td>9.8</td><td>0.950</td></tr><tr><td>1</td><td>OOB accuracy</td><td>0.911</td><td>0.921</td><td>0.921</td><td>0.976</td><td>9.2</td><td>0.838</td></tr><tr><td>5</td><td>TOP-Gain</td><td>0.691</td><td>0.841</td><td>0.880</td><td>0.984</td><td>10.0</td><td>1.000</td></tr><tr><td>5</td><td>Net gain</td><td>0.691</td><td>0.839</td><td>0.876</td><td>0.976</td><td>9.0</td><td>0.788</td></tr><tr><td>5</td><td>OOB Brier</td><td>0.691</td><td>0.840</td><td>0.878</td><td>0.975</td><td>9.0</td><td>0.763</td></tr><tr><td>5</td><td>OOB accuracy</td><td>0.670</td><td>0.830</td><td>0.872</td><td>0.975</td><td>8.6</td><td>0.663</td></tr><tr><td>10</td><td>TOP-Gain</td><td>0.525</td><td>0.761</td><td>0.826</td><td>0.976</td><td>9.8</td><td>0.950</td></tr><tr><td>10</td><td>Net gain</td><td>0.531</td><td>0.764</td><td>0.828</td><td>0.980</td><td>9.0</td><td>0.838</td></tr><tr><td>10</td><td>OOB Brier</td><td>0.520</td><td>0.759</td><td>0.820</td><td>0.977</td><td>9.0</td><td>0.838</td></tr><tr><td>10</td><td>OOB accuracy</td><td>0.448</td><td>0.720</td><td>0.775</td><td>0.956</td><td>7.0</td><td>0.388</td></tr><tr><td>20</td><td>TOP-Gain</td><td>0.305</td><td>0.652</td><td>0.713</td><td>0.962</td><td>9.4</td><td>0.850</td></tr><tr><td>20</td><td>Net gain</td><td>0.274</td><td>0.637</td><td>0.701</td><td>0.946</td><td>7.8</td><td>0.550</td></tr><tr><td>20</td><td>OOB Brier</td><td>0.347</td><td>0.673</td><td>0.742</td><td>0.955</td><td>8.4</td><td>0.688</td></tr><tr><td>20</td><td>OOB log-loss</td><td>0.326</td><td>0.662</td><td>0.729</td><td>0.965</td><td>8.4</td><td>0.625</td></tr><tr><td>20</td><td>OOB accuracy</td><td>0.253</td><td>0.625</td><td>0.680</td><td>0.938</td><td>6.6</td><td>0.400</td></tr></table>

Results of the imbalance sweep (Table S3a).

• TOP-Gain selected the most informative features at every imbalance ratio. It recovered all 10 at $\rho \leq 5 .$ , and 9.8 and 9.4 at $\rho = 1 0$ and 20.

• It also had the highest stability throughout $( \Phi \ge 0 . 8 5 )$

At $\rho = 2 0$ , the proper scoring rules gave higher minority recall and Macro-F1 than TOP-Gain, although they recovered fewer informative features. Selecting the right features does not by itself guarantee the best threshold-dependent metrics under severe imbalance.

## Epistasis benchmark.

• Data. $N = 1 { , } 5 0 0 , P = 3 0$ . Two features $( F _ { 1 } , F _ { 2 } )$ act only through an XOR interaction, four $( F _ { 3 }  – F _ { 6 } )$ have additive effects, four are noisy copies of the additive drivers, and 20 are pure noise.

• Design. $K \in \{ 2 , 4 , 6 \}$

Table S3b Epistasis benchmark (XOR pair $F _ { 1 } , F _ { 2 }$ with no marginal effect; additive drivers $F _ { 3 } ^ { } - F _ { 6 } ^ { } )$ Values: informative features selected (/6) | RF ROC-AUC | Φ.
<table><tr><td>Criterion</td><td> $K = 2$ </td><td> $K = 4$ </td></tr><tr><td>TOP-Gain</td><td>1.8 | 0.616 | 0.786</td><td>3.6 | 0.705 | 0.769 3.8 | 0.698 | 0.646</td></tr><tr><td>Net gain</td><td>1.4 | 0.618 | 0.679</td><td>2.4 | 0.685 | 0.481 3.2 | 0.699 | 0.604</td></tr><tr><td>OOB Brier</td><td>1.2 | 0.613 | 0.464</td><td>2.4 | 0.692 | 0.423 3.2 | 0.711 | 0.500</td></tr><tr><td>OOB log-loss</td><td>1.2 | 0.612 | 0.357</td><td>2.8 | 0.691 | 0.423 3.2 | 0.707 | 0.354</td></tr><tr><td>OOB accuracy</td><td>1.0 | 0.606 | 0.571</td><td>2.0 | 0.673 | 0.510 2.8 | 0.729 | 0.438</td></tr><tr><td>TOP-Gain + all extensions</td><td>1.6 | 0.623 | 0.679</td><td>3.2 | 0.696 | 0.596 4.0 | 0.736 | 0.562</td></tr></table>

## Results of the epistasis benchmark (Table S3b).

• For small panels $( K \leq 4 )$ , TOP-Gain recovered the most informative features with the highest stability.

• No criterion reliably captured the pure XOR pair: it was recovered in at most 1 of 5 folds. Greedy forward selection cannot detect a feature whose marginal contribution is zero until its partner has been selected.

## OVA\_Lung controlled imbalance.

• Data. � = 10,935, with lung tumours as the minority class. We used the natural data $( \rho =$ 11.3) and subsamples with 70 minority instances and $\rho \in \{ 1 , 5 , 1 0 , 2 0 \}$ , each drawn with three seeds.

• Design. � = 10, with a class-balanced RF as the downstream classifier.

Table S3c OVA\_Lung imbalance sweep (� = 10; balanced accuracy / ROC-AUC; mean over folds × seeds). “Natural” uses all 1,545 samples (� = 11.3).

<table><tr><td>Method</td><td>Natural</td><td>ρ= 1</td><td>ρ= 5</td><td>ρ = 10</td><td>ρ= 20</td></tr><tr><td>TOP-Gain</td><td>0.901 / 0.963</td><td>0.926 / 0.948</td><td>0.923 / 0.945</td><td>0.911 / 0.940</td><td>0.897 / 0.944</td></tr><tr><td>Net gain</td><td>0.921 / 0.955</td><td>0.933 / 0.952</td><td>0.925 / 0.941</td><td>0.916 / 0.939</td><td>0.904 / 0.945</td></tr><tr><td>OOB Brier</td><td>0.913 / 0.944</td><td>0.926 / 0.955</td><td>0.924 / 0.938</td><td>0.906 / 0.947</td><td>0.887 / 0.938</td></tr><tr><td>OOB accuracy</td><td>0.905 / 0.958</td><td>0.919 / 0.945</td><td>0.922 / 0.941</td><td>0.888 / 0.954</td><td>0.895 / 0.947</td></tr><tr><td>TOP-Gain + all extensions</td><td>0.929 / 0.949</td><td>0.929 / 0.953</td><td>0.925 / 0.951</td><td>0.916 / 0.944</td><td>0.909 / 0.943</td></tr><tr><td>ANOVA</td><td>0.929 / 0.934</td><td>0.931 / 0.935</td><td>0.929 / 0.941</td><td>0.921 / 0.940</td><td>0.916 / 0.935</td></tr><tr><td>Elastic Net</td><td>0.929 / 0.952</td><td>0.836 / 0.886</td><td>0.914 / 0.946</td><td>0.916 / 0.939</td><td>0.899 / 0.931</td></tr><tr><td>mRMR</td><td>0.905 / 0.959</td><td>0.933 / 0.958</td><td>0.916 / 0.962</td><td>0.896 / 0.955</td><td>0.861 / 0.946</td></tr></table>

## Results of the OVA\_Lung study (Table S3c).

• With ten-gene panels in a 10,935-gene space, the differences between methods were small and inconsistent across imbalance tiers. The ROC-AUC ranges of the wrapper criteria overlapped within about ±0.01.

TOP-Gain had the highest ROC-AUC on the natural data, while ANOVA had the highest balanced accuracy in four of the five tiers.

• In this setting TOP-Gain’s stability advantage over the net gain was not uniform (Table S3d).

Table S3d Stability Φ on OVA\_Lung, TOP-Gain versus the net gain
<table><tr><td></td><td>Natural</td><td>ρ= 1</td><td>ρ=5</td><td>ρ = 10</td><td>ρ= 20</td></tr><tr><td>TOP-Gain Φ</td><td>0.25</td><td>0.26</td><td>0.28</td><td>0.26</td><td>0.29</td></tr><tr><td>Net gain Φ</td><td>0.39</td><td>0.21</td><td>0.23</td><td>0.28</td><td>0.30</td></tr></table>

## S4 Robustness and ablation checks

Common-seed reference forest (Table S4a). Algorithm 1 refits the reference probabilities with seed � + step. Refitting them with the candidates’ seed � gave equivalent results.

• ROC-AUC: $p = 0 . 3 6 ;$ Φ: $p = 0 . 7 0 $ , original versus common seed at the dataset level.

The Grand Rank in Table S4a is the mean rank over all evaluation slices: datasets × � × classifiers × metrics.

Table S4a TOP-Gain with a common-seed reference forest
<table><tr><td>Comparison</td><td>TOP-Gain (original)</td><td>TOP-Gain (common seed)</td></tr><tr><td>Mean ROC-AUC / Macro-F1</td><td>0.9519 / 0.9046</td><td>0.9514 / 0.9041</td></tr><tr><td>Mean stability Φ</td><td>0.480</td><td>0.485</td></tr><tr><td>Original vs common seed (dataset level)</td><td></td><td>ROC-AUC p = 0.36; Φ p = 0.70</td></tr><tr><td>Grand Rank among TOP-Gain + nine baselines</td><td>1st (3.49)</td><td>1st (3.61)</td></tr><tr><td>Friedman mean rank of ROC-AUC (ten methods)</td><td>1st (1.85)</td><td>1st (2.55)</td></tr><tr><td>∆ROC-AUC vs baselines: 95% CI excludes zero</td><td>9 of 9</td><td>9 of 9</td></tr><tr><td>ROC-AUC vs baselines: significant after Holm</td><td>5 of 9</td><td>2 of 9 (ReliefF, SVM-RFE)</td></tr><tr><td>Φ vs alternative criteria: significant after Holm</td><td>5 of 5  $\dot { ( p _ { \mathrm { H o l m } } } \le 0 . 0 2 0 )$ </td><td> $5 \mathrm { o f } 5 ( p _ { \mathrm { H o l m } } \le 0 . 0 1 0 )$ </td></tr></table>

## Multi-objective extensions (Table S4b).

• What was added. A loss penalty (the weighted negative part), inverse-prevalence class weights, Laplacian smoothing of the OOB probabilities, and a pairwise synergy search among the ten best candidates.

• Effect. None of these improved predictive performance (all $p \ge 0 . 1 6 )$ . They reduced stability on all ten datasets $( p = 0 . 0 0 2 )$ and increased runtime by 20–33%.

• Synthetic imbalance sweep. The fully extended variant was worse than plain TOP-Gain at every ratio (Table S4c).

Table S4b TOP-Gain versus TOP-Gain with extensions (ten-dataset benchmark; Grand Rank among these three methods).
<table><tr><td>Variant</td><td>Grand Rank↓</td><td>ROC- AUC</td><td>Accuracy</td><td>Macro- F1</td><td>Stability φ</td><td>Time/fold (s)</td></tr><tr><td>TOP-Gain</td><td>1.92</td><td>0.9519</td><td>0.9158</td><td>0.9046</td><td>0.480</td><td>66.1</td></tr><tr><td>+ loss penalty, class weights, smoothing</td><td>1.98</td><td>0.9520</td><td>0.9150</td><td>0.9036</td><td>0.415</td><td>79.6</td></tr><tr><td>+ loss penalty, class weights, smoothing, pair synergy search</td><td>2.10</td><td>0.9520</td><td>0.9145</td><td>0.9029</td><td>0.425</td><td>88.1</td></tr></table>

Table S4c Synthetic imbalance sweep, plain TOP-Gain versus the fully extended variant (Macro-F1 / informative features selected)
<table><tr><td>ρ</td><td>TOP-Gain Macro-F1 / TP</td><td>Fully extended Macro-F1 / TP</td></tr><tr><td>5</td><td>0.880 / 10.0</td><td>0.870 / 9.0</td></tr><tr><td>20</td><td>0.713 / 9.4</td><td>0.704 / 7.6</td></tr></table>

## S5 Results per downstream classifier

Table S5 repeats the Friedman comparison of the main text separately for each downstream classifier.

Table S5 Friedman mean rank of TOP-Gain among the baselines, per downstream classifier (position in parentheses; best other method and its rank).
<table><tr><td>Classifier</td><td>CC18 ROC- AUC</td><td>CC18 Accuracy</td><td>CC18 Macro- F1</td><td>AMLB ROC- AUC</td><td>AMLB Accuracy</td><td>AMLB Macro-F1</td></tr><tr><td>Random Forest</td><td>3.35 (1st; VSURF 4.46)</td><td>3.60 (1st; VSURF 4.81)</td><td>3.35 (1st; VSURF 4.31)</td><td>3.00 (1st; RF- RFE 4.54)</td><td>3.12 (1st; RF- RFE 4.08)</td><td>3.54 (1st; RF- RFE 5.06)</td></tr><tr><td>XGBoost</td><td>3.62 (1st; VSURF 4.62)</td><td>4.29 (1st; VSURF 5.40)</td><td>3.35 (1st; VSURF 4.88)</td><td>2.38 (1st; RF perm. 4.75)</td><td>3.27 (1st; RF- RFE 4.88)</td><td>2.42 (1st; RF- RFE 4.54)</td></tr><tr><td>RBF-SVM</td><td>4.85 (2nd; VSURF 4.81)</td><td>4.04 (1st; VSURF 6.08)</td><td>3.92 (1st; VSURF 5.35)</td><td>3.00 (1st; RF- RFE 4.17)</td><td>3.25 (1st; RF- RFE 4.29)</td><td>3.79 (1st; RF- RFE 4.94)</td></tr><tr><td>Logistic regression</td><td>5.04 (2nd; VSURF 4.69)</td><td>4.58 (1st; VSURF 5.79)</td><td>5.08 (1st; VSURF 5.69)</td><td>3.29 (1st; RF- RFE 4.75)</td><td>4.21 (1st; HSIC- Lasso 5.46)</td><td>3.92 (1st; RF- RFE 5.54)</td></tr></table>

## S6 Protocol amendments

The protocol was written before any selector was run on the benchmark data. Each change was appended as a dated amendment with its reason, and the full protocol is part of the code repository.
<table><tr><td>Amendment</td><td>Date</td><td>Type</td><td>Content</td></tr><tr><td>A1</td><td>2026-10- 11</td><td>Implementation fix</td><td>Drop all-missing columns before imputation (dataset sick); re-screen.</td></tr><tr><td>A2</td><td>2026-10- 11</td><td>Design decision, before any selector run</td><td>Fixed panel sizes  $K \in \{ 5 , 1 0 , 1 5 , 2 0 \} , K \leq P / 2 .$ </td></tr><tr><td>A3</td><td>2026-10- 12</td><td>Registration, before any AMLB result</td><td>H1–H3 not supported; H4 and the AMLB pool registered.</td></tr><tr><td>A4</td><td>2026-10- 13</td><td>Implementation fix, output unchanged</td><td>Memory-efficient loader (verified bit-identical); re-run policy for infrastructure errors.</td></tr><tr><td>A5</td><td>2026-10- 14</td><td>Efficiency fix, selections verified identical</td><td>RF permutation passes NumPy arrays.</td></tr><tr><td>A6</td><td>2026-10- 15</td><td>Crash fix, non-crashing runs unchanged</td><td>JMI choice when all scores are NaN.</td></tr><tr><td>A7</td><td>2026-10- 20</td><td>Re-runs under A4</td><td>Four AMLB cells re-run once; all succeeded.</td></tr><tr><td>A8</td><td>2026-10- 20</td><td>Registration, before the study was run</td><td>Hyperparameter sensitivity study (T, depth, pool size); completed without failures.</td></tr></table>