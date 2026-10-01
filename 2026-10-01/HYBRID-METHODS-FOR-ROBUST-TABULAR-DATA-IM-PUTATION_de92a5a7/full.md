# HYBRID METHODS FOR ROBUST TABULAR DATA IM-PUTATION

Jinwei Li   
Department of Data Science   
Friedrich-Alexander-Universität Erlangen-Nürnberg   
Erlangen, Germany   
jinwei.li@fau.de   
Daniel Tenbrinck   
Department of Data Science   
Friedrich-Alexander-Universität Erlangen-Nürnberg   
Erlangen, Germany   
daniel.tenbrinck@fau.de   
Michelle Bruch   
Department of Data Science   
Friedrich-Alexander-Universität Erlangen-Nürnberg   
Erlangen, Germany   
michelle.bruch@fau.de

## ABSTRACT

Missing data are a fundamental challenge in statistical analysis and machine learning, as the choice of imputation method substantially impacts downstream inference. In this work, we propose two hybrid imputation methods called NuclearForest and SoftForest, which combine nuclear-norm-based low-rank initialization using Singular Value Thresholding (SVT) and SoftImpute, respectively, with a non-iterative Random Forest refinement. For the SVT-based component, we further introduce an adaptive step-size rule, prove adaptive step-size bounds, and establish convergence for the corresponding zero-initialized iteration. The low-rank initialization provides a structured warm start that captures the global covariance patterns in the data, while the subsequent Random Forest step recovers residual nonlinear signals encoding local dependencies. We conduct an extensive benchmark on diverse datasets from different application domains, comparing the proposed methods with seven established imputation methods under the Missing Completely at Random (MCAR), Missing at Random (MAR), and Missing Not at Random (MNAR) mechanisms across varying missingness rates. Our results demonstrate that NuclearForest and SoftForest match or exceed the imputation fidelity of state-of-the-art iterative methods such as MissForest, while significantly reducing computational cost. In particular, they achieve speedups of approximately 5.81× and 9.52× over MissForest by replacing iterative cycles with a single refinement step. Our approach effectively exploits the low-rank structure of real-world tabular data and accommodates mixed-type variables, providing an efficient and robust solution for data imputation in bioinformatics, economics, and beyond.

## 1 INTRODUCTION

Motivation Missing values are ubiquitous in real-world biomedical datasets, and data imputation is a critical prerequisite in the preprocessing pipeline. Consequently, the reliability of downstream analyses, including statistical testing and classification using machine learning methods, depends directly on the quality of the imputed data. However, state-of-the-art Random-Forest-based imputation methods, such as MissForest (Stekhoven & Bühlmann, 2012), require repeated fitting of predictive models until convergence, resulting in substantial computational cost. This computational complexity limits the practicality of iterative Random-Forest-based imputation methods when imputation must be performed repeatedly across experimental conditions. We introduce NuclearForest and SoftForest, two hybrid imputation methods that combine low-rank matrix completion with non-iterative Random Forest refinement, substantially reducing runtime while preserving high-quality imputations on tabular datasets.

Background Missing-data imputation can be approached from several complementary paradigms. Classical theory distinguishes between missing completely at random (MCAR), missing at random (MAR), and missing not at random (MNAR), since the validity of an imputation strategy depends on the mechanism that generated the missing entries (Rubin, 1976). Practical imputation methods span simple univariate replacement, model-based multiple imputation such as MICE (Van Buuren & Groothuis-Oudshoorn, 2011), and non-parametric machine-learning approaches such as MissForest (Stekhoven & Bühlmann, 2012). A complementary line of work exploits global correlation structures through low-rank matrix completion and nuclear-norm regularization, including Singular Value Thresholding (SVT) (Cai et al., 2010) and SoftImpute (Mazumder et al., 2010).

MissForest, proposed by Stekhoven & Bühlmann (2012), is a random-forest-based imputation method that has been shown to perform well on biological datasets, outperforming imputation methods such as kNN by modeling complex interactions and nonlinear relationships. Subsequent studies have shown that random-forest-based imputation can achieve strong imputation accuracy in datasets with nonlinear relationships and mixed variable types, with particularly favorable performance reported under MCAR/MAR missingness in metabolomics benchmarks (Tang & Ishwaran, 2017; Wei et al., 2018). We therefore choose random forests for the refinement step because our target setting is tabular imputation rather than representation learning from large unstructured data. For medium-sized tabular datasets, Random-Forest-based models remain state-of-the-art and outperform neural-network baselines even under extensive hyperparameter-search budgets (Grinsztajn et al., 2022). Moreover, random forests are a standard non-parametric regression tool and often perform reasonably well under default hyperparameter settings (Breiman, 2001; Probst et al., 2019). For missing-data imputation, this property is particularly valuable: random-forest-based models can capture nonlinear effects and feature interactions while accommodating both numerical and categorical variables, which motivates their use in MissForest (Stekhoven & Bühlmann, 2012). However, MissForest often is computationally expensive, as it repeatedly fits variable-wise random forests until convergence. The simple univariate initialization used before the iterative updates may provide a limited starting point, since it does not explicitly impose low-rank or other global structural assumptions on the completed matrix.

The SVT algorithm (Cai et al., 2010) is a classical first-order method for nuclear-norm-based matrix completion. Its appeal lies in a simple iterative structure, where each iteration alternates between singular-value shrinkage and a residual update. However, because standard SVT relies on fixed algorithmic parameters, including the singular-value threshold and the step size, its practical convergence speed can be sensitive to these choices and may require many iterations when they are not well matched to the data. One response in the literature is to adapt the threshold during the iteration (Zarmehi & Marvasti, 2017).

Approximate low-rank structure in real-world datasets. Many real-world datasets exhibit an approximately low-rank structure, as their variation is often governed by a relatively small number of latent factors. This is particularly relevant for omics data, here referring to tabular measurements of molecular abundances, where correlated biological pathways and shared regulatory mechanisms induce dependencies among measured features (Hilafu et al., 2020). Therefore, methods based on low-rank matrix completion provide a natural framework for omics data imputation by exploiting latent structure, but they may not fully capture nonlinear dependencies in the data.

Missingness mechanisms in MS-omics data. Mass-spectrometry (MS) omics data commonly contain missing values, and the missingness mechanism can affect imputation performance and downstream analysis (Sethi & Brietzke, 2016; Wei et al., 2018). Using the metabolomics dataset from Wei et al. (2018), we benchmark imputation methods under two artificial missingness settings: a joint MCAR/MAR condition and a separate MNAR condition. Because MCAR and MAR are often difficult to distinguish in MS-based metabolomics, the joint MCAR/MAR setting follows the benchmark design of Wei et al. (2018) and is generated by uniformly masking observed entries. In an additional experiment using a housing dataset (Harish Kumar Data Lab, 2024), MCAR and MAR are evaluated separately to further assess the behavior of the imputation methods under distinct missingness mechanisms. MNAR is modeled as left-censored missingness, where low-abundance features fall below the limit of detection or quantification. Accordingly, we generate MNAR missingness by selecting features and removing observations below variable-specific quantile cutoffs (Wei et al., 2018).

Benchmark methods and evaluation metrics. In this paper, real-world data from different applications will be imputed and evaluated using nine different imputation methods: MissForest, kNN, Mean, Median, Half-min, SVT, SoftImpute, and our proposed methods NuclearForest and SoftForest. We evaluate imputation quality from multiple complementary perspectives, covering normalized root mean squared error (NRMSE), global sample structure, univariate group-level signals, distributiona similarity, and downstream predictive utility. Specifically, we use NRMSE and an NRMSE-based sum of ranks (SOR) score for masked-entry reconstruction; PCA/PLS Procrustes analysis for structural preservation in reduced-dimensional spaces; Student’s t-test followed by Pearson correlation analysis for preservation of univariate group differences; and Gower’s distance together with downstream predictive $R ^ { 2 }$ degradation to assess distributional and task-level effects.

## Contributions

1. Adapted SVT. We introduce an efficient adaptive SVT variant (Algorithm 2) for imputation that replaces zero initialization with column-mean warm starting and modulates the step size according to the relative observed-entry reconstruction error, rather than relying on a fixed step size δ. Specifically, $\delta _ { k }$ is contracted by a factor of 0.9 when the error increases and expanded by a factor of 1.05, up to $2 p$ , when the error decreases, where $p = | \Omega | / ( n _ { 1 } n _ { 2 } )$ is the observed-entry ratio. Under a finite-contraction condition, the adaptive step sizes remain bounded away from zero and satisfy $\delta _ { k } < 2$ . Proposition 1 proves these bounds, while Corollary 1 gives convergence and an $O ( T ^ { - 1 } )$ rate for the corresponding zero-initialized iteration. In addition, an ablation study separately evaluates column-mean warm starting and adaptive step-size modulation, indicating that both components contribute to improved imputation quality. See Appendix D.3.

2. Hybrid imputation algorithms. We propose NuclearForest and SoftForest, two hybrid imputation methods that combine low-rank initialization with a single subsequent random-forest refinement step. NuclearForest uses the proposed adaptive SVT variant as its initialization module, whereas SoftForest uses SoftImpute as the corresponding low-rank initializer. Across the evaluated realdata settings, both methods achieve competitive imputation quality relative to iterative randomforest baselines while reducing runtime by approximately 5.81× for NuclearForest and 9.52× for SoftForest in one representative experiment.

3. Comprehensive real-data benchmark. We provide a broad empirical evaluation of nine imputation methods on metabolomics and housing datasets under MCAR, MAR, and MNAR missingness mechanisms. The benchmark compares methods across reconstruction, structural, statistical, and downstream predictive metrics, allowing us to assess not only masked-entry reconstruction error but also the preservation of low-dimensional sample structure, group-level signals, and predictive utility.

## 2 METHODOLOGY

Matrix completion via nuclear norm minimization. Let $M \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ be a partially observed matrix, and an entry of M is indexed by a pair $( i , j )$ , where $i \in \{ 1 , \ldots , n _ { 1 } \}$ is the row index and $j \in$ $\{ 1 , \dots , n _ { 2 } \}$ is the column index. Let $\Omega \subseteq \{ 1 , \dots , n _ { 1 } \} \times \{ 1 , \dots , \bar { n _ { 2 } } \}$ denote the index set of observed entries. The sampling operator $\mathcal { P } _ { \Omega }$ is defined entrywise by $\left( \mathcal { P } _ { \Omega } ( X ) \right) _ { i j } = \left\{ \begin{array} { l l } { X _ { i j } , } & { ( i , j ) \in \Omega , } \\ { 0 , } & { ( i , j ) \notin \Omega . } \end{array} \right.$ . The matrix completion problem aims to recover a complete matrix X that agrees with the partially observed matrix $M$ on the observed index set Ω. Since infinitely many matrices can satisfy these observed constraints, recovery is ill posed without additional a priori assumptions about the expected solution. A standard assumption is that the underlying complete matrix has a low-rank structure, meaning that much of its variation can be represented in a low-dimensional linear subspace. This assumption leads to the following rank-minimization formulation:

$$
\operatorname* { m i n } _ { X \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } } } \operatorname { r a n k } ( X ) \qquad \mathrm { s . ~ t . ~ } \qquad \mathcal { P } _ { \Omega } ( X ) = \mathcal { P } _ { \Omega } ( M ) .\tag{1}
$$

Since rank minimization is nonconvex and NP-hard (Recht et al., 2010), the optimization problem equation 1 is commonly replaced by a convex relaxation (Candes & Recht, 2012)

$$
\operatorname* { m i n } _ { X \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } } } \| X \| _ { * } \qquad \mathrm { s . ~ t . } \qquad \mathcal { P } _ { \Omega } ( X ) = \mathcal { P } _ { \Omega } ( M ) ,\tag{2}
$$

where $\begin{array} { r } { \| X \| _ { * } = \sum _ { i } \sigma _ { i } ( X ) } \end{array}$ denotes the nuclear norm, i.e., the sum of the singular values of X. This leads to a convex formulation of the matrix completion problem based on nuclear norm minimization. Solutions to this relaxed problem can be efficiently computed via singular value thresholding methods.

Singular Value Thresholding (SVT). The SVT algorithm was introduced by Cai et al. (2010) as a first-order method for solving nuclear-norm minimization problems. In the matrix completion setting, instead of treating the constrained nuclear-norm problem equation 2 directly, the method is derived from the closely related regularized problem

$$
\operatorname* { m i n } _ { X } \tau \| X \| _ { * } + \frac { 1 } { 2 } \| X \| _ { F } ^ { 2 } \qquad \mathrm { s . ~ t . } \qquad \mathcal { P } _ { \Omega } ( X ) = \mathcal { P } _ { \Omega } ( M ) ,
$$

where $\tau > 0$ is a threshold parameter and $\| \cdot \| _ { F }$ denotes the Frobenius norm. For large values of ${ \mathit { \Pi } } _ { \tau , { \mathit { \Pi } } }$ the minimizer of this problem approaches the minimum-Frobenius norm solution of the corresponding nuclear-norm minimization problem. To derive the SVT algorithm, one introduces the following Lagrangian $\begin{array} { r } { \mathcal { L } ( X , Y ) = \tau \| X \| _ { * } + \frac { 1 } { 2 } \| X \| _ { F } ^ { 2 } + \langle Y , \mathcal { P } _ { \Omega } ( M - X ) \rangle } \end{array}$ , where $Y$ is the dual variable associated with the equality constraint. Following the Uzawa interpretation of Cai et al. (2010), this yields an iterative scheme that alternates between minimizing the Lagrangian with respect to the primal variable $X$ and performing a dual gradient step in the dual variable $\mathrm { \Delta }$ . Starting from $Y ^ { 0 } = 0$ , the SVT iteration is given by

$$
X ^ { k } = { \mathcal { S } } _ { \tau } ( Y ^ { k - 1 } ) , \qquad Y ^ { k } = Y ^ { k - 1 } + \delta _ { k } { \mathcal { P } } _ { \Omega } ( M - X ^ { k } ) ,
$$

where $\delta _ { k } \ > \ 0$ denotes the step-size parameter. In the classical SVT implementation of Cai et al. (2010), this step size is typically chosen to be constant, i. $ \therefore \delta _ { k } \ = \ \delta$ The operator $S _ { \tau }$ denotes the proximal operator of the nuclear norm given by $S _ { \tau } ( Y ) : = \mathrm { p r o x } _ { \tau \parallel \cdot \parallel _ { * } } \bar { ( } Y ) =$ arg min<sub>X</sub> $\begin{array} { r } { \left\{ \tau \| X \| _ { * } + \frac { 1 } { 2 } \| X - Y \| _ { F } ^ { 2 } \right\} } \end{array}$ . If $\scriptstyle { Y \ = \ U \Sigma V ^ { T } }$ is a singular value decomposition of $Y$ with $\Sigma = \operatorname { d i a g } ( \sigma _ { 1 } , . . . , \sigma _ { r } )$ , where $\sigma _ { 1 } , \ldots , \sigma _ { r }$ are the singular values of $Y ,$ , then $S _ { \tau }$ acts by softthresholding the singular values through:

$$
\begin{array} { r } { S _ { \tau } ( Y ) = U \mathrm { ~ d i a g } \left( ( \sigma _ { i } - \tau ) ^ { + } \right) V ^ { T } , \qquad ( a ) ^ { + } : = \mathrm { m a x } ( a , 0 ) . } \end{array}\tag{3}
$$

Hence, each SVT iteration alternates between two operations: singular value thresholding, which promotes a low-rank structure, and a dual update on the observed entries, which drives the reconstruction toward consistency with the available data.

Improved SVT variant. In this work, we propose an improved variant of the classical SVT method with two key modifications. First, instead of initializing the missing entries as zero, we use a column-mean warm start, which provides an inexpensive data-dependent initialization that preserves the observed entries and places the initial completion on the empirical feature scale, thereby reducing the burden on subsequent low-rank recovery and random-forest refinement. Second, we replace the empirical fixed step size $\delta = 1 . 2 / p$ (where $\begin{array} { r } { p \ = \ \frac { | \Omega | } { n _ { 1 } n _ { 2 } } } \end{array}$ is the observed-entry rate of the matrix) used in the original SVT implementation with an adaptive rule: $\delta _ { 0 } = 1 . 2 p$ , followed by multiplicative expansion or contraction according to the observed-entry residual, with a cap at $\bar { \delta _ { \operatorname* { m a x } } } = \bar { 2 }$ . Under the finite-contraction assumption, the adaptive step sizes remain bounded away from zero. In our experiments, this condition is empirically satisfied, as the residual-based rule does not trigger contraction steps on the evaluated real datasets. Thus, the adaptive step sizes remain within the standard SVT admissible range (Cai et al., 2010). Proposition 1 gives the step-size bounds, and Corollary 1 gives the zero-initialized convergence result. Instead of using an adaptive threshold as proposed by Zarmehi and Marvasti (Zarmehi & Marvasti, 2017), we keep τ fixed, matching the original SVT parameter settings. In all numerical experiments, we compute the SVT updates using a full singular value decomposition.

SoftImpute. In contrast to the SVT algorithm, SoftImpute, introduced by Mazumder et al. (2010), approaches matrix completion through the following penalized nuclear norm problem

$$
\operatorname* { m i n } _ { X } \ \frac { 1 } { 2 } \| \mathcal P _ { \Omega } ( M - X ) \| _ { F } ^ { 2 } + \lambda \| X \| _ { * } ,\tag{4}
$$

where $\lambda > 0$ is a fixed regularization parameter. This objective is convex: the first term penalizes reconstruction error on the observed entries, while the nuclear-norm term promotes low-rank structure

through the same convex surrogate of the rank function used in nuclear-norm matrix completion. Therefore, SoftImpute is closely related to the SVT formulation, but replaces the hard constraint $\mathcal { P } _ { \Omega } ( X ) = \mathcal { P } _ { \Omega } ( M )$ with a penalized reconstruction term on the observed entries.

To solve this minimization problem, SoftImpute proceeds as follows. Given a current iterate $X ^ { k }$ , the method defines

$$
\begin{array} { r } { W ^ { k } = \mathcal { P } _ { \Omega } ( M ) + \mathcal { P } _ { \Omega } ^ { \perp } ( X ^ { k } ) , } \end{array}
$$

where $\mathcal { P } _ { \Omega } ^ { \perp }$ denotes the complementary projection onto the unobserved entries. This leads to the following equivalent reformulation:

$$
W ^ { k } = X ^ { k } + { \mathcal { P } } \Omega ( M - X ^ { k } ) .
$$

Thus, $W ^ { k }$ coincides with the observed data on Ω and with the current iterate on the complementary index set. The next iterate is then defined by $X ^ { k + 1 } = S _ { \lambda } ( W ^ { k } )$ , where $S _ { \lambda }$ denotes the singular value soft-thresholding operator introduced in equation 3. Hence, each iteration of SoftImpute consists of an imputation step in which the missing entries are filled-in using the current iterate, followed by a singular value thresholding step that promotes a low-rank structure. The parameter λ determines the amount of shrinkage applied to the singular values. In contrast to SVT, SoftImpute is formulated entirely in terms of the primal variable. Let $X ^ { \star }$ denote a solution of the penalized nuclear norm minimization problem. Then the objective values generated by SoftImpute decrease monotonically and converge to the optimal value. Moreover, the convergence rate is of order $\mathcal { O } ( 1 / k )$

Random forest regression. As a random-forest-based iterative imputation baseline, we use a MissForest-style procedure in which each incomplete column is treated as a supervised regression problem. Let $\bar { \boldsymbol X } ^ { ( t ) } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ denote the completed matrix at iteration t. At each iteration, incomplete columns are visited sequentially. For an incomplete column $j ,$ a random forest regressor $f _ { j } ^ { \left( t \right) }$ is trained on the observed rows $\{ i : ( i , j ) \in \Omega \}$ , using the current imputed predictor vectors $X _ { i , - j } ^ { ( t ) }$ and the observed responses $M _ { i j }$ . Here, $X _ { i , - j } ^ { ( t ) }$ denotes the row-i predictor vector formed by all columns except the target column j. The trained model is then used to update the missing entries in column $j \colon { \cal X } _ { i j } ^ { ( t + 1 ) } = f _ { j } ^ { ( t ) } \left( { \cal X } _ { i , - j } ^ { ( t ) } \right) , ( i , j ) \notin \Omega$ , while observed entries are kept fixed. This round-robin procedure is repeated until either a maximum number of iterations is reached or the change between consecutive completed matrices falls below a predefined tolerance.

NuclearForest and SoftForest. We propose two hybrid imputation methods that combine lowrank completion with nonlinear random-forest refinement. NuclearForest first obtains a low-rank initialization using SVT, whereas SoftForest uses SoftImpute. This design changes the role of the random forest compared with fully iterative random-forest imputation. Rather than starting from a simple univariate initialization and repeatedly updating incomplete features until convergence, as in MissForest-style methods, our methods first construct a structured low-rank completion and then apply a single random-forest refinement pass. For each incomplete feature, a random-forest regressor is trained on the rows where that feature is observed, using the corresponding matrix completed by SVT or SoftImpute as the predictor space. The fitted model is then applied once to the missing entries of that feature, while the originally observed entries remain fixed. The single-pass design is deliberate: the low-rank stage already captures global correlation structure, so the random forest is used primarily as a nonlinear residual corrector that refines the low-rank estimate by modeling interaction effects and nonlinear dependencies. Additional refinement passes are possible and may slightly improve accuracy in some settings, but they increase computational cost and provide only marginal further gains in our experiments. We therefore use the single-pass random-forest refinement to emphasize the accuracy–efficiency trade-off of the proposed hybrid framework.

## 3 EXPERIMENTS AND RESULTS

In our numerical experiments, we benchmark nine imputation methods on two datasets under three missingness mechanisms. For each setting, missing entries are regenerated independently over 10 repeated experimental runs, ensuring different missingness masks across runs while preserving reproducibility. Neural-network-based imputation methods are not included as primary baselines because the focus of this study is on efficient classical and hybrid tabular imputers. This choice i consistent with recent tabular-learning benchmarks showing that tree-based models remain highly competitive and can outperform neural-network baselines on many medium-sized tabular datasets (Grinsztajn et al., 2022). On the publicly available metabolomics dataset (Wei et al., 2018), we use the four evaluation metrics adopted by Wei et al. (2018). Since MAR is commonly approximated by MCAR in metabolomics benchmarks, we evaluate MCAR/MAR via uniform random masking and MNAR via left-censored missingness. On the housing dataset, we use the same four metrics together with three additional task-specific metrics. For the MAR setting, we generated missing values using a conditional logistic masking mechanism, where the missingness probability of each variable was modeled as a function of other variables rather than the variable’s own value (Schouten et al., 2018). Runtime is reported as wall-clock time in seconds and summarized as mean ± standard deviation (std) over repeated runs under the same local computational environment. All experiments were run on macOS 14.1 using an Apple M3 Pro processor with 11 CPU cores and 18 GB of RAM. The code of this benchmark experiment as well as of the two proposed imputation algorithms will be publicly released upon acceptance to ensure reproducibility.

Ablations and SVD implementation in the SVT. We evaluate the contribution of each design choice by isolating (i) zero initialization versus column-mean warm start and (ii) fixed versus adaptive step size. Across the four metrics (NRMSE, PCA Procrustes distance, p-value correlation, and PLS Procrustes distance), both components improve imputation quality, with the full method performing best overall under MCAR/MAR missingness. Fig. 1 summarizes the NRMSE ablation, while the complete ablation results are reported in Appendix D.3. We also compare dense full SVD with partial SVD. The original

![](images/fca8f4f82435ff45996a77ff84fa3a90dfbe1cf8e1ed2011bd583fe79ddbfa27.jpg)  
Figure 1: Ablation study.

SVT implementation of Cai et al. (2010) exploits the sparsity of the dual iterate $Y ^ { k }$ and computes only the dominant singular values needed by the shrinkage operator, using PROPACK/Lanczos bidiagonalization. This is appropriate for large sparse matrix completion problems. In contrast, our experimental matrices are dense after initialization. On the metabolomics dataset (198 × 131), the full-SVD implementation achieved a 2.89× speedup over the partial-SVD SVT variant, with indistinguishable imputation error (Appendix D.2 ). For this reason, we use the full SVD in all experiments. For large sparse datasets, one may instead use partial SVD approximations, which are commonly employed to reduce the cost of singular-value shrinkage when only the leading singular components are needed.

Metabolomics dataset under MCAR/MAR missingness. NRMSE is computed on masked entries after column-wise z-score standardization of both the ground-truth and imputed matrices, using means and standard deviations estimated from the complete ground-truth matrix. This transformation enables an unbiased comparison when using NRMSE by standardizing variable scales, preventing variables with larger values from dominating the evaluation (Wei et al., 2018). As shown in Fig. 2a, the imputation quality of MissForest, NuclearForest, and SoftForest is very close to each other with respect to the NRMSE metric. As can be observed, MissForest performs slightly better for missing rates between 20% and 60%. Next, PCA is performed using the first two principal components, as they capture the greatest variance in the data (Wei et al., 2018). Moreover, the symmetric Procrustes sum of squared errors is computed to compare the distribution. In Fig. 2b, NuclearForest yields the lowest PCA Procrustes distance at 10%, 20%, and 40% missingness. Following Wei et al. (2018), we evaluate whether imputation preserves univariate group-difference signals by conducting Welch’s two-sample t-tests for each variable between groups in the complete and imputed data, and then computing the Pearson correlation between the resulting log-transformed p-values. As shown in Fig. 2c, NuclearForest achieves the highest Pearson log-p correlation values at 20% and 90% missingness and remains competitive across the remaining missing rates, where MissForest obtains the best results. PLS Procrustes analysis is used to quantify the structural distortion. NuclearForest achieves the lowest PLS Procrustes distance for missing rates between 10% and 30% in Fig. 2d. As shown in Table 1, the hybrid methods (SoftForest and NuclearForest) remain significantly faster, providing \~9.52× and \~5.81× speedups, respectively. Further runtime analysis is provided in Appendix D.4.

![](images/39bfb2b80415fec9ff39ddef509c57ed3f53edce47a34c825c971801c500b5e3.jpg)  
(a) NRMSE

![](images/bf72abb96a15cf7f8d059910126ff4a52f754fef870a8627e3b9d0f95360edb2.jpg)

![](images/30bbc6e89a246004ae4bc33cb44d9aabe5a4ccc50161647d1cab32b98adb01f0.jpg)  
(c) Pearson log-p correlation

(b) PCA Procrustes distance  
![](images/84f30b96980c33f1870a693de6cbcee05ce67329f7491f6d2e584b2f2e309c37.jpg)  
(d) PLS Procrustes distance  
Figure 2: Comparison of imputation methods on the metabolomics dataset under MCAR/MAR.

Table 1: Runtime comparison for metabolomics dataset, reported in seconds as mean ± std.
<table><tr><td>Mechanism</td><td>MissForest</td><td>SoftForest</td><td>NuclearForest</td></tr><tr><td>MCAR/MAR</td><td> $7 7 . 7 8 7 \pm 2 1 . 4 2 6$ </td><td> $8 . 1 7 1 \pm 1 . 9 3 1$ </td><td> $1 3 . 3 8 9 \pm 2 . 3 6 5$ </td></tr><tr><td>MNAR</td><td> $8 9 . 4 3 8 \pm 2 2 . 8 4 5$ </td><td> $4 . 3 4 4 \pm 2 . 1 3 7$ </td><td> $9 . 7 9 7 \pm 2 . 3 8 1$ </td></tr></table>

![](images/0e060026bb461a0e6fa675a1d0cfa4c4cefc57e35be25165ebac802c78da53b3.jpg)

![](images/32fced57290a838a3da72770a012610048c633f6a85709c8e54a9ea06f50f7ce.jpg)

(a) SOR  
![](images/acfe9cf8e0e2e9866aa570e8d5b2a4bc8f142277bea6a7f1dd35ffd99ec8b08a.jpg)  
(c) Pearson Log-p correlation

(b) PCA Procrustes distance  
![](images/2437ba3c1a279fbb73d2675b455a2f7ed2f3d920b0d4c613b4508aa1bd1d8e1f.jpg)  
(d) PLS Procrustes distance  
Figure 3: Comparison of the imputation methods using metabolomics dataset under MNAR

![](images/e36bf9c87c87833daef63f6bc662a7a20d50242768bc3d7dfd7b9a5648e38455.jpg)  
(a) NRMSE

![](images/e4e1d00c658217c8d736cd4a2d975984582155eb0fb1ac796cc712bf8bde3c27.jpg)  
(b) PFC

![](images/106f17e7063f40203581c4cb81e308951c20a77984774a8f15d94cac80e8052e.jpg)  
(c) Gower’s distance

![](images/94bcbbde4d2bedcd72a8717ad9774e447fdb517c48e2d1c947c963c49edd90ca.jpg)  
(d) Predictive $\mathsf { R } ^ { 2 }$ degradation

![](images/3c9b017e45f9465e525465ac17b96d84ad9ed0519597d15a048b70b447611d6f.jpg)

![](images/9cf39480a171587149de691f07c6f330b75b24d39149cce6556e33d4a8b66b8b.jpg)  
(f) PLS Procrustes distance

(e) PCA Procrustes distance  
![](images/f030984d3ee4b32d6abe8ae7f3724b2273d217a89367dfcab8c475f175748b0b.jpg)  
(g) NRMSE

![](images/542657c3c76b013ab4cd761988c6eb310ea046d31267c633aabd343887506c86.jpg)  
(h) PFC

![](images/bd233eb08065a647f1df2373be1ae8cbbb5258d1b306de91bb21939729df11ad.jpg)  
(i) Gower’s distance

![](images/8dea88ec683be2d52fb8bf938d8e6300661bb30cbca840bd38f4a8ffb809297a.jpg)  
(j) Predictive $\mathrm { R } ^ { 2 }$ degradation

![](images/f5b9f1f3fcf82fdf4fa7528424731701ff974ce2ccc19327721fbe1b1eb73828.jpg)  
(k) PCA Procrustes distance

![](images/5a332662739970369de9ff89218c1c4d1913978dc783b69a2a03e72775dc62c1.jpg)  
(l) PLS Procrustes distance  
Figure 4: Comparison of imputation methods under MCAR (a–f) and MAR (g–l).

Table 2: Runtime comparison on the housing dataset, reported in seconds as mean ± std.
<table><tr><td>Mechanism</td><td>MissForest</td><td>SoftForest</td><td>NuclearForest</td></tr><tr><td>MCAR</td><td> $5 . 9 3 4 \pm 0 . 1 8 7$ </td><td> $0 . 6 6 4 \pm 0 . 0 4 5$ </td><td> $0 . 6 5 9 \pm 0 . 0 4 7$ </td></tr><tr><td>MAR</td><td> $5 . 9 7 3 \pm 0 . 1 9 8$ </td><td> $0 . 6 6 1 \pm 0 . 0 3 9$ </td><td> $0 . 6 6 1 \pm 0 . 0 3 4$ </td></tr></table>

Metabolomics dataset under MNAR missingness. A non-parametric method called NRMSEbased sum of ranks (SOR) is used to evaluate the imputation error for the skewness of the MNAR distribution (Wei et al., 2018). Fig. 3a indicates that SoftImpute demonstrates a clear advantage in this metric, achieving the lowest values across all missingness levels except at 20% missingness, where SoftForest performs best. As shown in Fig. 3 b–d, the left-censored Half-min imputation performs best for three other metrics, yielding results similar to those reported by Wei et al. (2018). This behavior is expected under the MNAR design, where missing values are generated from lowabundance entries below feature-specific thresholds. Since Half-min replaces missing entries with a small value, its inductive bias matches the left-censored missingness mechanism.

Housing dataset under MCAR and MAR missingness. We further benchmark data imputation on the publicly available Kaggle housing dataset (Harish Kumar Data Lab, 2024) under MCAR and MAR mechanisms, having six numerical and seven categorical features. In terms of estimation error, NRMSE after z-score transformation is used and for the binary/one hot columns, we calculate the proportion of falsely classified entries (PFC) (Stekhoven & Bühlmann, 2012). Fig. 4 a–b show that NuclearForest performs best for missing rates between 10% and 60% in terms of NRMSE, and at 10%, 30%, and 50%–60% in terms of PFC under the MCAR mechanism, while SoftForest achieves the lowest PFC values at 20% and 40% missingness. Under the MAR mechanism, the results in Fig. 4 g–h indicate that NuclearForest obtains the lowest NRMSE at 10%–60% missingness, while SoftForest performs best at 70% missingness. In terms of PFC, NuclearForest achieves the best results at 10% and 50%–60% missingness, whereas SoftForest obtains the lowest values at 20%–40% missingness. For distributional evaluation, Gower’s distance measures the dissimilarity for the mixed data types (El Badisy et al., 2024; Gower, 1971). Fig. 4c shows that, under the MCAR mechanism, NuclearForest obtains the lowest values at 10% and 30%–60% missingness, while SoftForest achieves the lowest value at 20% missingness. Under the MAR mechanism, Fig. 4i shows that NuclearForest performs best at 10%, 30%, and 50%–60% missingness, whereas SoftForest obtains the lowest values at 20%, 40% and 70%–90% missingness. Downstream price prediction evaluates the utility of the imputed data for regression tasks. This follows a prediction-oriented evaluation perspective in supervised learning with missing values (Josse et al., 2024; Le Morvan et al., 2021; Morvan & Varoquaux, 2024), where missing-value handling is assessed by its impact on downstream predictive performance rather than only by reconstruction error. We therefore compute Downstream Predictive $R ^ { 2 }$ Degradation by training a ridge regression model to predict housing prices from all remaining features. Overall, these results suggest that the most effective method for preserving downstream predictive performance depends on the severity of missingness. Hybrid methods are particularly competitive at low to moderate missingness levels, whereas MissForest and SoftImpute become more favorable under higher missingness. To maintain consistency across datasets, we additionally evaluate imputation performance using global and group-level structure preservation metrics, including PCA and PLS Procrustes distance, and Pearson correlation of log-transformed p-values (Wei et al., 2018). These metrics are not commonly used for evaluating regression tasks, where predictive performance (e.g., regression error) is typically the primary criterion, originating instead from omics-based studies where preserving multivariate structure and statistical inference is essential. Specifically, we divide the dataset into three groups of price tercile labels to assess the imputation methods. Fig. 4 e–f and Fig. 4 k–l show the results of PCA/PLS Procrustes distance and the experimental results with respect to the log-transformed p-values is given in Appendix D Fig. 7. Under the MCAR mechanism, NuclearForest obtains the lowest PCA and PLS Procrustes distance at 10%–70% missingness. At 80%–90% missingness, SVT achieves the best result for both PCA and PLS Procrustes distances. When missingness follows the MAR mechanism, SoftForest obtains the best results across most missingness levels, while NuclearForest performs best at 30% missingness. The Pearson log-p correlation results in Fig. 7a–b show that, under MCAR, median imputation achieves the best performance across most missingness rates, with the exception of the 10%–20% setting. For evaluation under MAR, SoftForest performs best at 50%–70% missing rates.

## 4 CONCLUSION

We proposed NuclearForest and SoftForest, two hybrid imputation methods that combine low-rank initialization with a single random-forest refinement pass. NuclearForest is built on an improved SVT variant with warm starting and adaptive step sizes. For this adaptive SVT component, we prove stepsize bounds and a convergence result for the corresponding zero-initialized iteration, and our ablation studies show that both the warm start and adaptive update contribute to improved imputation quality under the MCAR/MAR missingness setting considered in our experiments. Across the metabolomics and housing benchmarks, the proposed hybrid methods offer a favorable accuracy–efficiency trade-off. On the former, NuclearForest remains competitive with MissForest across missingness levels and is particularly strong at low missing rates, while MissForest achieves the best performance in several medium- and high-missingness settings. Importantly, the proposed hybrids are substantially faster: in the MCAR/MAR metabolomics experiment, SoftForest and NuclearForest achieve approximately 9.52× and 5.81× speedups over MissForest, respectively. A limitation is that these gains do not fully extend to the left-censored MNAR setting, where Half-min is well aligned with this mechanism. This suggests that the proposed hybrids are most effective when the missingness pattern preserves enough global correlation structure for informative low-rank initialization.

The advantage of the hybrid strategy is more pronounced on the smaller housing dataset, where NuclearForest and SoftForest perform strongly across many low- to moderate-missingness settings, with especially strong performance on PCA and PLS Procrustes metrics, indicating better preservation of low-dimensional and group-level structure. These findings suggest that low-rank completion provides an effective structural warm start, while a single random forest refinement step can capture nonlinear feature relationships with lower computational cost than fully iterative random forest imputation. Overall, this supports the proposed hybrid strategy as a practical and computationally efficient approach for preserving structural information during tabular data imputation. Future work will further characterize the full hybrid procedure theoretically and explore low-rank warm starts for other nonlinear refiners, including neural-network-based imputers.

## ACKNOWLEDGEMENTS

This work was supported by the Bayerisches Verbundforschungsprogramm (BayVFP) of the Free State of Bavaria under the funding line “Digitalisierung”. We also thank Gina Pommerenke, Judith Mehler, Johanna Schuerlein and Dr. Josef Scheiber at BioVariance GmbH for valuable biological expertise and discussions that helped inform the interpretation of the biomedical data.

## AI USE STATEMENT

In this work, we used generative AI tools to improve the grammar and readability of the manuscript, identify potentially relevant literature, and assist with coding. All AI-assisted content was manually reviewed and verified. We take responsibility for the final content of this work.

## REFERENCES

Leo Breiman. Random forests. Machine learning, 45(1):5–32, 2001.

Jian-Feng Cai, Emmanuel J Candès, and Zuowei Shen. A singular value thresholding algorithm for matrix completion. SIAM Journal on optimization, 20(4):1956–1982, 2010.

Emmanuel Candes and Benjamin Recht. Exact matrix completion via convex optimization. Communications ofthe ACM, 55(6):111–119, 2012.

Damek Davis, Dmitriy Drusvyatskiy, and Liwei Jiang. Gradient descent with adaptive stepsize converges (nearly) linearly under fourth-order growth. Mathematical Programming, pp. 1–66, 2025.

Imad El Badisy, Nathalie Graffeo, Mohamed Khalis, and Roch Giorgi. Multi-metric comparison of machine learning imputation methods with application to breast cancer survival. BMC Medical Research Methodology, 24(1):191, 2024.

John C Gower. A general coefficient of similarity and some of its properties. Biometrics, pp. 857–871, 1971.

Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? Advances in neural information processing systems, 35: 507–520, 2022.

Harish Kumar Data Lab. Housing Price Prediction Dataset. Kaggle, 2024. URL https://www. kaggle.com/datasets/harishkumardatalab/housing-price-prediction.

Haileab Hilafu, Sandra E Safo, and Lillian Haine. Sparse reduced-rank regression for integrating omics data. BMC bioinformatics, 21(1):283, 2020.

Julie Josse, Jacob M Chen, Nicolas Prost, Gaël Varoquaux, and Erwan Scornet. On the consistency of supervised learning with missing values. Statistical Papers, 65(9):5447–5479, 2024.

Marine Le Morvan, Julie Josse, Erwan Scornet, and Gaël Varoquaux. What’s a good imputation to predict with missing values? Advances in Neural Information Processing Systems, 34:11530– 11540, 2021.

Yunwen Lei and Ding-Xuan Zhou. Analysis of singular value thresholding algorithm for matrix completion. Journal ofFourier Analysis and Applications, 25(6):2957–2972, 2019.

Rahul Mazumder, Trevor Hastie, and Robert Tibshirani. Spectral regularization algorithms for learning large incomplete matrices. The Journal of Machine Learning Research, 11:2287–2322, 2010.

Marine Le Morvan and Gaël Varoquaux. Imputation for prediction: beware of diminishing returns. arXiv preprint arXiv:2407.19804, 2024.

Philipp Probst, Marvin N Wright, and Anne-Laure Boulesteix. Hyperparameters and tuning strategies for random forest. Wiley Interdisciplinary Reviews: data mining and knowledge discovery, 9(3): e1301, 2019.

Benjamin Recht, Maryam Fazel, and Pablo A Parrilo. Guaranteed minimum-rank solutions of linear matrix equations via nuclear norm minimization. SIAM review, 52(3):471–501, 2010.

Donald B Rubin. Inference and missing data. Biometrika, 63(3):581–592, 1976.

Rianne Margaretha Schouten, Peter Lugtig, and Gerko Vink. Generating missing values for simulation purposes: a multivariate amputation procedure. Journal of Statistical Computation and Simulation, 88(15):2909–2930, 2018.

Sumit Sethi and Elisa Brietzke. Omics-based biomarkers: application of metabolomics in neuropsychiatric disorders. International Journal of Neuropsychopharmacology, 19(3):pyv096, 2016.

Daniel J Stekhoven and Peter Bühlmann. Missforest—non-parametric missing value imputation for mixed-type data. Bioinformatics, 28(1):112–118, 2012.

Fei Tang and Hemant Ishwaran. Random forest missing data algorithms. Statistical Analysis and Data Mining: The ASA Data Science Journal, 10(6):363–377, 2017.

Stef Van Buuren and Karin Groothuis-Oudshoorn. mice: Multivariate imputation by chained equations in r. Journal ofstatistical software, 45:1–67, 2011.

Runmin Wei, Jingye Wang, Mingming Su, Erik Jia, Shaoqiu Chen, Tianlu Chen, and Yan Ni. Missing value imputation approach for mass spectrometry-based metabolomics data. Scientific reports, 8 (1):663, 2018.

Nematollah Zarmehi and Farokh Marvasti. Adaptive singular value thresholding. In 2017 International Conference on Sampling Theory and Applications (SampTA), pp. 442–445. IEEE, 2017.

## A ADAPTIVE SVT VARIANT AND STEP-SIZE ANALYSIS

This section compares the proposed adaptive SVT variant with the standard fixed-step SVT and provides the proof of the step-size boundedness result.

Adaptive Step Size vs. Fixed Step Size The original paper by Cai et al. (2010) uses a fixed step size $\bar { \delta } = 1 . 2 / \bar { p } .$ , where p is the sampling rate. Moreover, they prove convergence for a fixed step size satisfying $0 < \delta < 2$ . In this paper, our SVT uses an adaptive step size that increases when the error is decreasing (step 1.05) and shrinks when the error increases (step 0.9).

In a convergence analysis paper by Lei & Zhou (2019), they prove that a necessary and sufficient condition for convergence with respect to the Bregman distance, specifically the step size sequence $\delta _ { k } ,$ , must satisfy $\sum _ { k = 1 } ^ { \infty ^ { - } } \delta _ { k }$ = ∞ and the corresponding convergence proof is provided by (Lei & Zhou, 2019, Proposition 5). The necessity is derived using a lower bound on the decay of the Bregman distance. This provides theoretical justification for employing an adaptive step size, as long as the step sizes do not decay too rapidly (bounded away from zero), ensuring divergence of their cumulative sum.

Recent paper in optimization theory suggests that adaptive step size strategies can accelerate firstorder methods beyond what is achievable with constant step sizes. In particular, Davis et al. (2025) demonstrate that gradient descent with adaptive step sizes achieves nearly linear convergence under weaker growth conditions where constant step size methods exhibit only sublinear rates.

Although these results are established for smooth unconstrained optimization (nuclear norm is not smooth), they provide important insight into the role of step size selection in gradient-based algorithms. Since the SVT algorithm can be interpreted as a gradient descent method applied to the dual problem, this perspective motivates the use of adaptive step size strategies within SVT. Moreover, it can improve convergence behavior by applying larger effective updates in the case of slow progress.

Warm-start initialization (column-mean imputation) vs. cold-start (zero matrix) Standard SVT updates the matrix starting from zero. Motivated by the warm-start strategy used in SoftImpute (Mazumder et al., 2010), we initialize the missing entries with column means before applying the low-rank completion step. This deterministic initialization provides a simple and stable starting point for the iterative updates.

## Adaptive step size convergence under finitely many contraction steps

Linear equality constraints (Cai et al., 2010). Set the objective function

$$
f _ { \tau } ( \mathbf { X } ) = \tau \| \mathbf { X } \| _ { * } + \frac { 1 } { 2 } \| \mathbf { X } \| _ { F } ^ { 2 }
$$

for some fixed $\tau > 0$ , and consider the following optimization problem:

$$
\begin{array} { r l } { \operatorname* { m i n i m i z e } } & { f _ { \tau } ( \mathbf { X } ) } \\ { \mathrm { s u b j e c t ~ t o } } & { \mathcal { A } ( \mathbf { X } ) = \mathbf { b } , } \end{array}\tag{3.1}
$$

where $\mathcal { A }$ is a linear transformation mapping $n _ { 1 } \times n _ { 2 }$ matrices into $\mathbb { R } ^ { m }$ and $\ b { A } ^ { * }$ denotes its adjoint. Then the Lagrangian for this problem is of the form

$$
\begin{array} { r } { \mathcal { L } ( \mathbf { X } , \mathbf { y } ) = f _ { \tau } ( \mathbf { X } ) + \langle \mathbf { y } , \mathbf { b } - \mathcal { A } ( \mathbf { X } ) \rangle , } \end{array}\tag{3.2}
$$

where $\mathbf { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ and $\mathbf { y } \in \mathbb { R } ^ { m }$ , and starting with $ { \mathbf { y } } ^ { 0 } = 0$ , Uzawa’s iteration is given by

$$
\left\{ \begin{array} { l l } { \mathbf { X } ^ { k } = \mathcal { S } _ { \tau } \big ( \mathcal { A } ^ { * } ( \mathbf { y } ^ { k - 1 } ) \big ) , } \\ { \mathbf { y } ^ { k } = \mathbf { y } ^ { k - 1 } + \delta _ { k } \big ( \mathbf { b } - \mathcal { A } ( \mathbf { X } ^ { k } ) \big ) . } \end{array} \right.\tag{3.3}
$$

Shrinkage iterations (Cai et al., 2010). For the matrix completion problem, let $\mathcal { A } = \mathcal { P } _ { \Omega } , b =$ $\mathcal { P } _ { \Omega } ( M )$ , and note that $\| \mathcal { P } \Omega \| = 1$ . Then (3.1) reduces to

$$
\begin{array} { r l } { \mathrm { m i n i m i z e } } & { \tau \| \mathbf { X } \| _ { * } + \displaystyle \frac { 1 } { 2 } \| \mathbf { X } \| _ { F } ^ { 2 } } \\ { \mathrm { s u b j e c t ~ t o } } & { \mathcal { P } _ { \Omega } ( \mathbf { X } ) = \mathcal { P } _ { \Omega } ( \mathbf { M } ) . } \end{array}\tag{2.8}
$$

and the corresponding iteration (3.3) becomes

$$
\left\{ \begin{array} { l l } { \mathbf { X } ^ { k } = \mathcal { S } _ { \tau } \big ( \mathbf { Y } ^ { k - 1 } \big ) , } \\ { \mathbf { Y } ^ { k } = \mathbf { Y } ^ { k - 1 } + \delta _ { k } \mathcal { P } _ { \Omega } \big ( \mathbf { M } - \mathbf { X } ^ { k } \big ) , } \end{array} \right.\tag{2.7}
$$

where we write $\mathbf { Y } ^ { k }$ in place of $\mathbf { y } ^ { k }$ .

Theorem 1 (SVT convergence (Cai et al., 2010)). Suppose the step sizes obey

$$
0 < \operatorname* { i n f } _ { k } \delta _ { k } \leq \operatorname* { s u p } _ { k } \delta _ { k } < 2 / \| A \| ^ { 2 }
$$

Then the sequence $\{ \mathbf { X } ^ { k } \}$ obtained via (3.3) converges to the unique solution to (3.1). In particular, the sequence $\{ \mathbf { X } ^ { k } \}$ obtained via (2.7) converges to the unique solution of (2.8) provided that

$$
0 < \operatorname* { i n f } _ { k } \delta _ { k } \leq \operatorname* { s u p } _ { k } \delta _ { k } < 2
$$

Theorem 2. (Lei & Zhou, 2019) Let $\{ ( X ^ { k } , Y ^ { k } ) \} _ { k \in \mathbb { N } }$ be produced by (3.3) and $\mathbf { b } _ { 0 } \neq 0$ . Here, b<sub>0</sub> denotes the orthogonal projection ofb onto the range ofA. The term $D _ { \Psi } ^ { Y ^ { T } } ( X ^ { \star } , X ^ { T } )$ denotes the Bregman distance associated with Ψ, evaluated at $X ^ { T }$ with subgradient $Y ^ { T }$ . In general, for ${ \widetilde { Y } } \in \partial { \Psi } ( { \widetilde { X } } ) , D _ { \Psi } ^ { \widetilde { Y } } ( X , \widetilde { X } ) = \Psi ( X ) - \Psi ( \widetilde { X } ) - \langle X - \widetilde { X } , \widetilde { Y } \rangle$

Then thefollowing statements hold.

1. If sup<sub>k</sub> $\begin{array} { r } { \delta _ { k } < \frac { 1 } { 2 \| \mathcal { A } \| ^ { 2 } } } \end{array}$ , then

$$
\operatorname* { l i m } _ { T \to \infty } D _ { \Psi } ^ { Y ^ { T } } ( X ^ { \star } , X ^ { T } ) = 0 \quad i f a n d o n l y \ : i f \quad \sum _ { k = 1 } ^ { \infty } \delta _ { k } = \infty .
$$

2. $\begin{array} { r } { I f \mathrm { s u p } _ { k } \delta _ { k } < \frac { 2 } { \| \mathcal { A } \| ^ { 2 } } } \end{array}$ , then

$$
\Vert X ^ { T + 1 } - X ^ { \star } \Vert _ { F } ^ { 2 } \leq \tilde { C } \left[ \sum _ { k = 1 } ^ { T } \delta _ { k } \right] ^ { - 1 } , \quad \forall T \in \mathbb { N } ,
$$

where $\tilde { C }$ is a constant independent $o f T .$

Proposition 1 (Adaptive step sizes bounds under condition). Let $p : = | \Omega | / ( n _ { 1 } n _ { 2 } )$ denote the observed-entry ratio. Assume $0 < p < 1$ , let $\delta _ { 0 } = 1 . 2 p$ and set $\delta _ { 1 } = \delta _ { 0 }$ . Let

$$
e _ { k } : = \frac { \| \mathcal { P } _ { \Omega } ( X ^ { k + 1 } - M ) \| _ { F } } { \| \mathcal { P } _ { \Omega } ( M ) \| _ { F } }
$$

denote the relative observed-entry reconstruction error, as in Algorithm 2. For $k \geq 1$ , let the step sizes be generated by

$$
\delta _ { k + 1 } = { \left\{ \begin{array} { l l } { 0 . 9 \delta _ { k } , } & { i f e _ { k } > e _ { k - 1 } , } \\ { \operatorname* { m i n } { \big ( } 1 . 0 5 \delta _ { k } , 2 p { \big ) } , } & { i f e _ { k } \leq e _ { k - 1 } . } \end{array} \right. }
$$

Assumefurther that the total number ofcontraction steps isfinite, namely, there exists $B \in  { \mathbb { N } } _ { 0 }$ such that at most B indices k satisfy $e _ { k } > e _ { k - 1 }$ . Then the following statements hold:

(a) For all $k \geq 0 ,$

$$
0 . 9 ^ { B } \delta _ { 0 } \leq \delta _ { k } \leq 2 p < 2 .\tag{5}
$$

Hence

$$
0 < \operatorname* { i n f } _ { k } \delta _ { k } \leq \operatorname* { s u p } _ { k } \delta _ { k } < 2 .
$$

(b) The step sizes have divergent sum:

$$
\sum _ { k = 0 } ^ { \infty } \delta _ { k } = \infty .\tag{6}
$$

Proof. The proof is in three parts.

Part 1: positivity and upper bound. We first show by induction that

$$
0 < \delta _ { k } \leq 2 p \qquad \forall k \geq 0 .
$$

For $k = 0$ , this is immediate from $\delta _ { 0 } = 1 . 2 p$ and $0 < p < 1$ , since

$$
0 < \delta _ { 0 } = 1 . 2 p \leq 2 p .
$$

For $k = 1$ , this also holds because $\delta _ { 1 } = \delta _ { 0 }$

Now let $k \geq 1$ and assume $0 < \delta _ { k } \le 2 p$ . There are two cases. If $e _ { k } > e _ { k - 1 }$ , then

$$
\delta _ { k + 1 } = 0 . 9 \delta _ { k } ,
$$

so $0 < \delta _ { k + 1 } < 2 p$

If $e _ { k } \leq e _ { k - 1 }$ , then

$$
\delta _ { k + 1 } = \operatorname* { m i n } { \left( 1 . 0 5 \delta _ { k } , 2 p \right) } ,
$$

which again implies $0 < \delta _ { k + 1 } \leq 2 p$

Thus, by induction, $0 < \delta _ { k } \le 2 p$ for all $k \geq 0$

Part 2: uniform positive lower bound under finitely many contraction steps. For $k \geq 1$ , define

$$
N _ { k } : = \# \{ j \in \{ 1 , \dots , k - 1 \} : e _ { j } > e _ { j - 1 } \} ,
$$

with $N _ { 0 } = N _ { 1 } = 0$ . Thus $N _ { k }$ counts the number of contraction steps used in generating $\delta _ { k }$ from the initial step size. By assumption,

$$
N _ { k } \leq B \qquad \forall k \geq 0 .
$$

An expansion step never decreases the step size, while a contraction step multiplies the step size by exactly 0.9. Therefore, after generating $\delta _ { k }$ , the smallest possible value of $\delta _ { k }$ is obtained by applying the factor 0.9 at each of the $N _ { k }$ contraction steps and no decrease at the remaining steps. Hence

$$
\delta _ { k } \geq 0 . 9 ^ { N _ { k } } \delta _ { 0 } \geq 0 . 9 ^ { B } \delta _ { 0 } .
$$

Combining this with Part 1 yields

$$
0 . 9 ^ { B } \delta _ { 0 } \leq \delta _ { k } \leq 2 p < 2 ,
$$

which proves equation 5.

Part 3: divergence of the sum. From the lower bound just proved,

$$
\delta _ { k } \geq 0 . 9 ^ { B } \delta _ { 0 } > 0 \qquad \forall k \geq 0 .
$$

Therefore,

$$
\sum _ { k = 0 } ^ { \infty } \delta _ { k } \ \ge \ \sum _ { k = 0 } ^ { \infty } 0 . 9 ^ { B } \delta _ { 0 } = \infty ,
$$

which proves equation 6.

Corollary 1 (Convergence of the corresponding zero-initialized adaptive SVT iteration). Consider the standard SVT iteration (2.7) with zero initialization ${ \cal Y } ^ { 0 } = 0 ,$ , and let the step sizes be generated by the same adaptive rule as in Proposition 1. Define $e _ { k }$ analogouslyfor this zero-initialized iteration, and assume that its total number ofcontraction steps is finite, which means there exists $B \in  { \mathbb { N } } _ { 0 }$ such that at most B indices k satisfy $e _ { k } > e _ { k - 1 }$

Then, by the same argument as in Proposition $^ { l , }$

$$
0 . 9 ^ { B } \delta _ { 0 } \leq \delta _ { k } \leq 2 p < 2 ,
$$

and hence

$$
0 < \operatorname* { i n f } _ { k } \delta _ { k } \leq \operatorname* { s u p } _ { k } \delta _ { k } < 2 .
$$

Therefore, by the SVT convergence theorem ofCai et al. (2010), the corresponding zero-initialized adaptive SVT iterates converge to a unique solution. Moreover, Proposition 9 of Lei & Zhou (2019) provides the following convergence rate bound for the zero-initialized iteration:

$$
D _ { \Psi } ^ { Y ^ { T + 1 } } ( X ^ { * } , X ^ { T + 1 } ) \leq \widetilde { C } \left[ \sum _ { k = 1 } ^ { T } \delta _ { k } \right] ^ { - 1 } ,
$$

where $\widetilde { C }$ is independent of T. Since

$$
\delta _ { k } \geq 0 . 9 ^ { B } \delta _ { 0 } ,
$$

we have

$$
\sum _ { k = 1 } ^ { T } \delta _ { k } \geq T 0 . 9 ^ { B } \delta _ { 0 } ,
$$

and therefore

$$
D _ { \Psi } ^ { Y ^ { T + 1 } } ( X ^ { * } , X ^ { T + 1 } ) \leq \frac { \widetilde { C } } { T 0 . 9 ^ { B } \delta _ { 0 } } = O ( T ^ { - 1 } ) .
$$

## B ALGORITHMS

Algorithm 1 Singular Value Thresholding (SVT) (Cai et al., 2010)   
Require: Partially observed matrix $M \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ , observed index set $\Omega ,$ threshold $\tau ,$ step size $\delta ,$   
tolerance ε   
1: Initialize $Y ^ { 0 } \gets 0$   
2: repeat   
3: $X ^ { k }  S _ { \tau } ( Y ^ { k - 1 } )$   
4: $Y ^ { k } \gets Y ^ { \dot { k } - 1 } + \delta ^ { ' } { \mathcal { P } } \Omega ( M - X ^ { k } )$   
5: until $\| \mathcal { P } _ { \Omega } ( X ^ { k } - M ) \| _ { F } / \| \mathcal { P } _ { \Omega } ( M ) \| _ { F } < \varepsilon$   
6: return $\widehat { M }$ with $\widehat { M } _ { i j } = M _ { i j }$ for $( i , j ) \in \Omega ,$ and $\widehat { M } _ { i j } = X _ { i j } ^ { k }$ otherwise

Algorithm 2 SVT with warm start and adaptive step size   
Require: Partially observed matrix $M \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ , observed index set $\Omega ,$ threshold $\tau ,$ tolerance $\varepsilon$   
1: Initialize $X ^ { 0 }$ by column-mean imputation; $Y ^ { 0 }  X ^ { 0 }$   
2: $\delta _ { 0 } \gets 1 . 2 p , \delta _ { \operatorname* { m a x } } \gets \operatorname* { m i n } ( 2 p , 2 )$ , where $p = | \Omega | / ( n _ { 1 } n _ { 2 } ) , e _ { - 1 } \gets \infty$   
3: repeat   
4: $Y ^ { k + 1 } \gets Y ^ { k } + \delta _ { k } \mathcal { P } _ { \Omega } ( M - X ^ { k } )$   
5: $X ^ { k + 1 }  S _ { \tau } ( Y ^ { k + 1 } )$   
6: $e _ { k } \gets \| \mathcal { P } _ { \Omega } ( \dot { X } ^ { k + 1 } - M ) \| _ { F } / \| \mathcal { P } _ { \Omega } ( M ) \| _ { F }$   
7: $\delta _ { k + 1 } \gets 0 . 9 \delta _ { k }$ if $e _ { k } > e _ { k - 1 } ,$ , else min(1.05 $\delta _ { k } , \delta _ { \mathrm { m a x } } )$   
8: until $e _ { k } < \varepsilon$   
9: return $\widehat { M }$ with $\widehat { M } _ { i j } = M _ { i j }$ on Ω, $\widehat { M } _ { i j } = X _ { i j } ^ { k + 1 }$ otherwise

Algorithm 3 SoftImpute (Mazumder et al., 2010)   
Require: Partially observed matrix $M \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } }$ , observed index set $\Omega ,$ maximum rank $r _ { \mathrm { m a x } } .$   
regularization $\lambda ,$ tolerance ε   
1: Initialize $X ^ { 0 }  0$   
2: repeat   
3: $W ^ { k }  { \mathcal { P } } _ { \Omega } ( M ) + { \mathcal { P } } _ { \Omega } ^ { \perp } ( X ^ { k } )$   
4: $U \Sigma V ^ { \top }  \operatorname { S V D } ( W ^ { \bar { k } } )$ , truncate to top $r _ { \mathrm { m a x } }$ components   
5: $X ^ { k + 1 }  S _ { \lambda } ( W ^ { \lambda } )$   
6: until $\| X ^ { k + 1 } - \mathring X ^ { k } \| _ { F } ^ { 2 ^ { \prime } } / \| X ^ { k } \| _ { F } ^ { 2 } < \varepsilon$   
7: return $\widehat { M }$ with $\widehat { M } _ { i j } = M _ { i j }$ on Ω, $\widehat { M } _ { i j } = X _ { i j } ^ { k + 1 }$ otherwise

## C EXPERIMENTAL DETAILS

This section summarizes the hyperparameters and implementation details used in the experiments.

Table 3: Hyperparameter settings used in the additional experiments. SVT\_OGpaper follows the fixed step size configuration of Cai et al. (2010). SoftImpute follows the soft-thresholded SVD formulation of Mazumder et al. (2010).
<table><tr><td>Method</td><td>Configuration / hyperparameters</td></tr><tr><td>Mean</td><td>Column-wise mean imputation.</td></tr><tr><td>Median</td><td>Column-wise median imputation.</td></tr><tr><td>Half-min</td><td>Missing entries are replaced by one half of the observed column minimum  $1 0 ^ { - 6 }$ </td></tr><tr><td>kNN</td><td>when the minimum is positive; otherwise a floor value of is used.  $k = 1 0$  nearest neighbors with distance-based weighting.</td></tr><tr><td>MissForest</td><td>Iterative Random Forest imputation using 100 trees, maximum iterations = 10, median initialization, and  $\mathrm { n \_ j o b s = - 1 }$ </td></tr><tr><td>SVT_OGpaper</td><td>The original  $\operatorname { s v T }$  method with a fixed step size and  $\delta = \operatorname* { m i n } ( 1 . 2 / p , 2 )$  zero initialization, skip-ahead dual initialization, maximum iterations</td></tr><tr><td>SVT</td><td>= 1000, and tolerance  $\stackrel { - } { = } 1 0 ^ { - 5 }$  Proposed adaptive  $\operatorname { s v T }$  variant with  $\tau = 5 \operatorname* { m a x } ( n _ { 1 } , n _ { 2 } )$  , column-mean warm start, initial step size  $\delta _ { 0 } = \operatorname* { m i n } ( 1 . 2 p , 2 p )$  , step-size cap  $\operatorname* { m i n } ( 2 p , 2 )$  contraction factor 0.9, expansion factor 1.05, maximum iterations = 1000,</td></tr><tr><td>NF_OGpaper</td><td>and tolerance  $= 1 0 ^ { - 5 }$  Two-stage method using SVT_OGpaper for initialization, followed by one column-wise Random Forest refinement with 100 trees, and  $\mathrm { n \_ j o b s = - 1 }$ </td></tr><tr><td>NuclearForest</td><td>Two-stage method using the proposed adaptive SVT variant for initializa- tion, followed by one column-wise Random Forest refinement with 100</td></tr><tr><td>SoftImpute</td><td>trees, and  $\mathrm { n \_ j o b s = - 1 }$  SoftImpute with data-scaled regularization  $\lambda \ : = \ : 0 . 0 1 \lambda _ { 0 }$  , where  $\lambda _ { 0 }$  is the largest singular value of the zero-filled matrix; rank cap min  $( n , p )$ </td></tr><tr><td>SoftForest</td><td>maximum  ${ \mathrm { i t e r a t i o n s } } = 1 0 0 0 ,$  and convergence threshold  $= \mathrm { 1 0 ^ { - 5 } }$  Two-stage method using SoftImpute initialization with  $\lambda = 0 . 0 1 \lambda _ { 0 } ,$  rank cap mir  $\scriptstyle 1 ( n , p )$  , maximum iterations = 1000, and threshold  $= 1 0 ^ { - 5 }$  , fol- lowed by one column-wise Random Forest refinement with 100 trees, and  $\mathrm { n \_ j o b s = - 1 }$ </td></tr></table>

Table 4: Global experimental settings
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Number of repeated runs 10</td><td></td></tr><tr><td>Random seeds</td><td> $^ { 4 2 , 4 3 , \dots , 5 1 }$  , assigned sequentially to the 10 repeated runs</td></tr><tr><td>Missingness rates</td><td>10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%</td></tr></table>

Table 5: Total wall-clock runtime.
<table><tr><td>Experiment</td><td>Total runtime (min)</td><td>Total runtime (h)</td></tr><tr><td>Metabolomics dataset under MCAR/MAR missingness</td><td>192.5</td><td>3.21</td></tr><tr><td>Metabolomics dataset under MNAR missingness</td><td>192.0</td><td>3.20</td></tr><tr><td>Housing dataset under MCAR missingness</td><td>12.9</td><td>0.22</td></tr><tr><td>Housing dataset under MAR missingness</td><td>13.0</td><td>0.22</td></tr></table>

## D ADDITIONAL EXPERIMENTAL RESULTS

## D.1 RESULTS INCLUDING THE ORIGINAL SVT METHOD

![](images/472f00af5e09573c2f7ed9302a16b7ca27bb8e2ab09fe0aa51354d37c5049e97.jpg)  
(a) NRMSE

![](images/e8df2bfc46d50df5a2e2b21ebd2f002b9514951b92a8d7402b5b9f3be468a9c5.jpg)  
(b) PCA Procrustes distance

![](images/b9f22252d9dd37b0f00733e145f9613f97365f16ff51ca58dd9fd54540a54fec.jpg)  
(c) Pearson log-p correlation

![](images/e060e36285fbd48db798272985aeadc00ed611f1701509c5ebe1cd4ce20d0d30.jpg)  
(d) PLS Procrustes distance

![](images/b17e617a1a414947fd8cfcda0ae177165323ee0b12f7f5513c5271cddd2ddd14.jpg)  
(e) SOR

![](images/9512c9a43a4a6b56c97196764c9cd13233d5f94e26cc2c7573f12ef98a35710c.jpg)

![](images/7bb74c5f99aaadbde7e1c11052c3856b359344c0a92126eccab92833d638f073.jpg)  
(g) Pearson log-p correlation

(f) PCA Procrustes distance  
![](images/bccb3568633185f0974ff565f2316a314750464fd1706a303594c29de758b387.jpg)  
(h) PLS Procrustes distance  
Figure 5: Comparison of SVT\_OGpaper and NF\_OGpaper with the adapted SVT and NuclearForest methods on the metabolomics dataset under MCAR/MAR (subfigures a–d) and MNAR (subfigures e–h).

![](images/47c385b943fc977fd6d8c12e6ca53b005f5a7d27d425a339f46fef6a1fec2ec7.jpg)  
(a) NRMSE

![](images/f168fe0f47ae2fe1ed6d42e487060aaf565c048d77a8b8808037f9dd3ec3aa3c.jpg)  
(b) PFC

![](images/a10c6829b4995fd10bd10cf1547102f9f1244521d33c27ab773d13560d40e7e9.jpg)  
(c) Gower’s distance

![](images/1aa5f0187a883d460f5cbff052ae5f3d7a652069da13543b7325d6c08c7288f5.jpg)  
(d) Predictive R<sup>2</sup> degradation

![](images/ac94312c263495398bc9f17d10392f34a31d13795279cafa6d313e186a48b0d4.jpg)  
(e) PCA Procrustes distance

![](images/bc3bf8e69846411fb134f4c43195a15da31aae997742c2ea7cba3fc2be5ba133.jpg)  
(f) PLS Procrustes distance

![](images/10e73f2c7675d01e5b81b8602c8be33a701d69a907675a2dd00dce7aea0413b6.jpg)  
(g) NRMSE

![](images/ba0872f6d3b4370b832036d82cbe125bb2300a4d37c4b344886ec3200c780af0.jpg)  
(h) PFC

![](images/3f9bab380c1771d82b1aeb228fe1f9bafb6d0185912da0b880637cea6dfcafbb.jpg)  
(i) Gower’s distance

![](images/580b8af6f2aa28cc0b2cf85347a963434b4057a558473ea9790663f5053566b6.jpg)  
(j) Predictive R<sup>2</sup> degradation

![](images/d88d30f45a4b7f53d48177833817cda93f724150cbd61d3e22fb8712b8eceb9d.jpg)  
(k) PCA Procrustes distance

![](images/125b2b4508d11c398edd77d604eda16ea0ae99701fa42a07d460bf931085e6f4.jpg)  
(l) PLS Procrustes distance

Figure 6: Comparison of SVT\_OGpaper and NF\_OGpaper with the adapted SVT and NuclearForest methods on the housing dataset under MCAR (subfigures a–f) and MAR (subfigures g–l).  
![](images/098b1e8e34b6699f96df5d977a6e181d473107e159c5fe25f1246b8047a550c8.jpg)  
(a) Pearson log-p correlation

![](images/76dce261c6d7ef01c42366902f3eae143af82c40ed445e3577e910cda2807747.jpg)  
(b) Pearson log-p correlation  
Figure 7: Comparison of imputation methods on the housing dataset under MCAR (subfigure a) and MAR (subfigure b).

## D.2 PARTIAL SVD VERSUS FULL SVD IN SVT

D.2 compares partial and full SVD implementations inside SVT. These results assess whether the choice of SVD solver materially affects imputation quality in the small-to-moderate dense

matrices considered in this work. SVT\_partial uses a partial SVD implementation, whereas SVT and SVT\_OGpaper use full SVD.

Table 6: NRMSE
<table><tr><td>Rate</td><td> $\mathrm { S V T \_ O G p a p e r }$ </td><td> $\mathrm { S V T \_ p a r t i a l }$ </td><td>SVT</td></tr><tr><td>10%</td><td> $0 . 8 0 6 \pm 0 . 0 1 1$ </td><td> $0 . 8 0 4 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 7 4 0 \pm 0 . 0 5 0 }$ </td></tr><tr><td>20%</td><td> $0 . 9 2 0 \pm 0 . 0 2 9$ </td><td> $0 . 9 1 7 \pm 0 . 0 2 8$ </td><td> $\mathbf { 0 . 8 1 4 \pm 0 . 0 3 2 }$ </td></tr><tr><td>30%</td><td> $0 . 9 9 9 \pm 0 . 0 1 2$ </td><td> $0 . 9 9 5 \pm 0 . 0 1 4$ </td><td> $\mathbf { 0 . 8 7 4 \pm 0 . 0 1 5 }$ </td></tr><tr><td>40%</td><td> $1 . 1 0 4 \pm 0 . 0 2 7$ </td><td> $1 . 0 9 6 \pm 0 . 0 2 8$ </td><td> $\mathbf { 0 . 9 3 8 \pm 0 . 0 2 8 }$ </td></tr><tr><td>50%</td><td> $1 . 6 4 6 \pm 0 . 0 6 7$ </td><td> $1 . 6 4 6 \pm 0 . 0 6 7$ </td><td> $\mathbf { 1 . 0 2 3 \pm 0 . 0 2 3 }$ </td></tr><tr><td>60%</td><td> $1 . 5 7 9 \pm 0 . 0 1 4$ </td><td> $1 . 5 7 9 \pm 0 . 0 1 4$ </td><td> $\mathbf { 1 . 1 4 0 \pm 0 . 0 1 3 }$ </td></tr><tr><td>70%</td><td> $1 . 7 1 7 \pm 0 . 0 3 5$ </td><td> $1 . 7 1 7 \pm 0 . 0 3 5$ </td><td> $\mathbf { 1 . 2 5 2 \pm 0 . 0 3 5 }$ </td></tr><tr><td>80%</td><td> $2 . 0 2 8 \pm 0 . 0 6 6$ </td><td> $2 . 0 2 8 \pm 0 . 0 6 6$ </td><td> $\mathbf { 1 . 3 6 0 \pm 0 . 0 3 3 }$ </td></tr><tr><td>90%</td><td> $2 . 7 5 3 \pm 0 . 0 6 9$ </td><td> $2 . 7 5 3 \pm 0 . 0 6 9$ </td><td> $\mathbf { 1 . 3 3 2 \pm 0 . 0 2 8 }$ </td></tr></table>

Table 7: PCA Procrustes distance

$$
0 . 0 0 2 8 \pm 0 . 0 0 0 6
$$

$$
0 . 0 0 2 8 \pm 0 . 0 0 0 7
$$

$$
0 . 0 0 9 5 \pm 0 . 0 0 2 6
$$

$$
0 . 0 0 9 6 \pm 0 . 0 0 2 7
$$

$$
\mathbf { 0 . 0 0 1 9 \pm 0 . 0 0 0 7 }
$$

$$
0 . 0 2 1 6 \pm 0 . 0 0 2 9
$$

$$
\mathbf { 0 . 0 0 5 9 \pm 0 . 0 0 1 1 }
$$

$$
0 . 0 2 1 5 \pm 0 . 0 0 3 0
$$

$$
\mathbf { 0 . 0 1 1 9 \pm 0 . 0 0 1 4 }
$$

$$
0 . 0 3 9 3 \pm 0 . 0 0 4 5
$$

$$
0 . 0 3 8 5 \pm 0 . 0 0 4 2
$$

$$
\mathbf { 0 . 0 2 2 6 \pm 0 . 0 0 4 8 }
$$

$$
0 . 1 0 2 0 \pm 0 . 0 1 7 8
$$

$$
0 . 1 0 2 0 \pm 0 . 0 1 7 8
$$

$$
\mathbf { 0 . 0 4 8 8 \pm 0 . 0 1 } 2 3
$$

$$
0 . 1 6 0 0 \pm 0 . 0 3 0 9
$$

$$
0 . 1 6 0 0 \pm 0 . 0 3 0 9
$$

$$
\mathbf { 0 . 0 9 5 2 \pm 0 . 0 3 1 2 }
$$

$$
0 . 3 2 3 1 \pm 0 . 0 2 3 8
$$

$$
0 . 3 2 3 1 \pm 0 . 0 2 3 8
$$

$$
\mathbf { 0 . 1 7 1 5 \pm 0 . 0 3 4 3 }
$$

$$
0 . 7 5 9 4 \pm 0 . 0 7 2 3
$$

$$
0 . 9 6 2 6 \pm 0 . 0 1 9 9
$$

$$
0 . 7 5 9 4 \pm 0 . 0 7 2 3
$$

$$
0 . 9 6 2 6 \pm 0 . 0 1 9 9
$$

$$
\mathbf { 0 . 8 0 8 0 \pm 0 . 0 6 0 4 }
$$

Table 8: Pearson correlation of $- \log _ { 1 0 } p$
<table><tr><td>Rate</td><td> $\mathrm { S V T \_ O G p a p e r }$ </td><td> $\mathrm { S V T \_ p a r t i a l }$ </td><td>SVT</td></tr><tr><td>10%</td><td> $0 . 9 8 2 \pm 0 . 0 0 4$ </td><td> $0 . 9 8 2 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 9 9 5 \pm 0 . 0 0 3 }$ </td></tr><tr><td>20%</td><td> $0 . 9 5 6 \pm 0 . 0 1 2$ </td><td> $0 . 9 5 6 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 9 9 1 } \pm \mathbf { 0 . 0 0 2 }$ </td></tr><tr><td>30%</td><td> $0 . 9 1 6 \pm 0 . 0 2 0$ </td><td> $0 . 9 1 5 \pm 0 . 0 2 0$ </td><td> $\mathbf { 0 . 9 8 0 \pm 0 . 0 0 3 }$ </td></tr><tr><td>40%</td><td> $0 . 8 5 8 \pm 0 . 0 1 5$ </td><td> $0 . 8 5 8 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 9 6 0 \pm 0 . 0 0 6 }$ </td></tr><tr><td>50%</td><td> $0 . 7 7 0 \pm 0 . 0 2 5$ </td><td> $0 . 7 7 0 \pm 0 . 0 2 5$ </td><td> $\mathbf { 0 . 9 3 8 \pm 0 . 0 0 9 }$ </td></tr><tr><td>60%</td><td> $0 . 6 9 5 \pm 0 . 0 3 9$ </td><td> $0 . 6 9 5 \pm 0 . 0 3 9$ </td><td> $\mathbf { 0 . 8 8 9 \pm 0 . 0 1 8 }$ </td></tr><tr><td>70%</td><td> $0 . 6 0 6 \pm 0 . 0 4 2$ </td><td> $0 . 6 0 6 \pm 0 . 0 4 2$ </td><td> $\mathbf { 0 . 8 3 8 \pm 0 . 0 3 0 }$ </td></tr><tr><td>80%</td><td> $0 . 4 7 8 \pm 0 . 1 1 1$ </td><td> $0 . 4 7 8 \pm 0 . 1 1 1$ </td><td> $\mathbf { 0 . 7 5 7 \pm 0 . 0 3 6 }$ </td></tr><tr><td>90%</td><td> $0 . 3 0 0 \pm 0 . 0 4 4$ </td><td> $0 . 3 0 0 \pm 0 . 0 4 4$ </td><td> $\mathbf { 0 . 6 3 1 \pm 0 . 0 4 4 }$ </td></tr></table>

Table 9: PLS Procrustes distance
<table><tr><td>Rate</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td><td> $\mathrm { S V T \_ p a r t i a l }$ </td><td>SVT</td></tr><tr><td>10%</td><td> $0 . 0 0 4 5 \pm 0 . 0 0 0 4$ </td><td> $0 . 0 0 4 5 \pm 0 . 0 0 0 5$ </td><td> $\mathbf { 0 . 0 0 4 1 \pm 0 . 0 0 0 5 }$ </td></tr><tr><td>20%</td><td> $0 . 0 1 2 4 \pm 0 . 0 0 0 4$ </td><td> $0 . 0 1 2 5 \pm 0 . 0 0 0 5$ </td><td> $\mathbf { 0 . 0 0 9 4 } \pm \mathbf { 0 . 0 0 1 } 2$ </td></tr><tr><td>30%</td><td> $0 . 0 2 7 2 \pm 0 . 0 0 1 7$ </td><td> $0 . 0 2 7 1 \pm 0 . 0 0 1 5$ </td><td> $\mathbf { 0 . 0 2 0 3 \pm 0 . 0 0 1 8 }$ </td></tr><tr><td>40%</td><td> $0 . 0 5 6 2 \pm 0 . 0 0 5 1$ </td><td> $0 . 0 5 6 0 \pm 0 . 0 0 4 9$ </td><td> $\mathbf { 0 . 0 4 0 3 \pm 0 . 0 0 2 2 }$ </td></tr><tr><td>50%</td><td> $0 . 1 6 3 0 \pm 0 . 0 2 4 3$ </td><td> $0 . 1 6 3 0 \pm 0 . 0 2 4 3$ </td><td> $\mathbf { 0 . 0 7 4 0 \pm 0 . 0 0 8 9 }$ </td></tr><tr><td>60%</td><td> $0 . 2 0 2 0 \pm 0 . 0 0 4 8$ </td><td> $0 . 2 0 2 0 \pm 0 . 0 0 4 8$ </td><td> $\mathbf { 0 . 1 2 7 2 \pm 0 . 0 1 1 6 }$ </td></tr><tr><td>70%</td><td> $0 . 3 7 4 2 \pm 0 . 0 2 9 8$ </td><td> $0 . 3 7 4 2 \pm 0 . 0 2 9 8$ </td><td> $\mathbf { 0 . 2 2 8 1 \pm 0 . 0 2 8 7 }$ </td></tr><tr><td>80%</td><td> $0 . 6 6 2 4 \pm 0 . 0 2 0 1$ </td><td> $0 . 6 6 2 4 \pm 0 . 0 2 0 1$ </td><td> $\mathbf { 0 . 4 3 0 7 \pm 0 . 0 6 3 7 }$ </td></tr><tr><td>90%</td><td> $0 . 8 9 0 5 \pm 0 . 0 1 4 9$ </td><td> $0 . 8 9 0 5 \pm 0 . 0 1 4 9$ </td><td> $\mathbf { 0 . 6 9 5 6 \pm 0 . 0 3 9 9 }$ </td></tr></table>

Table 10: Runtime comparison between partial SVD and full SVD in SVT.
<table><tr><td>Method</td><td>Mean runtime (s)</td><td>Std runtime (s)</td><td>Min runtime (s)</td><td>Max runtime (s)</td></tr><tr><td>SVT_OGpaper</td><td> $5 . 4 6 5 \pm 1 . 8 1 7$ </td><td>1.817</td><td>3.933</td><td>14.748</td></tr><tr><td>SVT_partial</td><td> $1 5 . 8 1 0 \pm 5 . 0 6 9$ </td><td>5.069</td><td>8.206</td><td>29.109</td></tr><tr><td>SVT</td><td> ${ \bf 5 . 2 9 2 \pm 1 . 2 6 7 }$ </td><td>1.267</td><td>3.896</td><td>9.304</td></tr></table>

## D.3 ABLATION STUDY

The ablation isolates the effects of the two modifications to the SVT: column-mean warm starting and the adaptive step-size rule. The plots compare these variants across missingness levels.

![](images/6fac223d9cd0af948d740fc4d8ac401c90661c9268c818c794fead5e1dac146c.jpg)  
Figure 8: Ablation study on the metabolomics dataset under MCAR/MAR missingness.

Table 11: NRMSE across missing rates
<table><tr><td>Rate</td><td> $\mathrm { Z e r o \cdot f i x e d }$ </td><td> $\mathrm { Z e r o \cdot a d a p t i v e }$ </td><td> $\mathrm { W a r m \cdot f i x e d }$ </td><td> $\mathbf { W a r m \cdot a d a p t i v e }$ </td></tr><tr><td>10%</td><td> $0 . 8 1 1 \pm 0 . 0 2 7$ </td><td> $0 . 8 1 1 \pm 0 . 0 2 7$ </td><td> $\mathbf { 0 . 7 3 4 \pm 0 . 0 4 2 }$ </td><td> $0 . 7 3 6 \pm 0 . 0 4 3$ </td></tr><tr><td>20%</td><td> $0 . 9 2 2 \pm 0 . 0 2 4$ </td><td> $0 . 9 2 3 \pm 0 . 0 2 4$ </td><td> $\mathbf { 0 . 8 0 9 \pm 0 . 0 2 6 }$ </td><td> $0 . 8 1 0 \pm 0 . 0 2 5$ </td></tr><tr><td>30%</td><td> $1 . 0 0 8 \pm 0 . 0 2 0$ </td><td> $1 . 0 0 5 \pm 0 . 0 2 0$ </td><td> $0 . 8 7 7 \pm 0 . 0 1 7$ </td><td> $\mathbf { 0 . 8 7 4 \pm 0 . 0 1 6 }$ </td></tr><tr><td>40%</td><td> $1 . 1 1 3 \pm 0 . 0 2 5$ </td><td> $1 . 1 0 4 \pm 0 . 0 2 3$ </td><td> $0 . 9 4 9 \pm 0 . 0 2 7$ </td><td> $\mathbf { 0 . 9 3 8 \pm 0 . 0 2 6 }$ </td></tr><tr><td>50%</td><td> $1 . 6 5 6 \pm 0 . 0 5 8$ </td><td> $1 . 2 3 7 \pm 0 . 0 1 1$ </td><td> $1 . 0 5 2 \pm 0 . 0 2 0$ </td><td> $\mathbf { 1 . 0 2 7 \pm 0 . 0 1 7 }$ </td></tr><tr><td>60%</td><td> $1 . 6 2 0 \pm 0 . 0 7 7$ </td><td> $1 . 3 9 2 \pm 0 . 0 2 4$ </td><td> $1 . 1 6 8 \pm 0 . 0 2 5$ </td><td> $\mathbf { 1 . 1 2 7 \pm 0 . 0 2 2 }$ </td></tr><tr><td>70%</td><td> $1 . 7 0 9 \pm 0 . 0 3 2$ </td><td> $1 . 6 0 4 \pm 0 . 0 2 3$ </td><td> $1 . 3 4 3 \pm 0 . 0 3 3$ </td><td> $\mathbf { 1 . 2 5 1 \pm 0 . 0 2 9 }$ </td></tr><tr><td>80%</td><td> $2 . 0 6 3 \pm 0 . 0 7 8$ </td><td> $1 . 9 5 2 \pm 0 . 0 5 9$ </td><td> $1 . 5 6 6 \pm 0 . 0 6 1$ </td><td> $\mathbf { 1 . 3 7 8 \pm 0 . 0 3 6 }$ </td></tr><tr><td>90%</td><td> $2 . 7 6 0 \pm 0 . 0 6 3$ </td><td> $2 . 5 4 2 \pm 0 . 0 7 3$ </td><td> $1 . 6 5 6 \pm 0 . 0 6 9$ </td><td> $\mathbf { 1 . 3 2 7 \pm 0 . 0 3 5 }$ </td></tr></table>

Table 12: PCA Procrustes distance across missing rates
<table><tr><td>Rate</td><td> $\mathrm { Z e r o \cdot f i x e d }$ </td><td> $\mathrm { Z e r o \cdot a d a p t i v e }$ </td><td> $\mathrm { W a r m \cdot f i x e d }$ </td><td> $\mathbf { W a r m \cdot a d a p t i v e }$ </td></tr><tr><td>10%</td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 0 0 2 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td></tr><tr><td>20%</td><td> $0 . 0 1 0 \pm 0 . 0 0 2$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 0 0 6 \pm 0 . 0 0 1 }$ </td></tr><tr><td>30%</td><td> $0 . 0 2 1 \pm 0 . 0 0 4$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 4$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 0 1 } 2 \pm \mathbf { 0 . 0 0 3 }$ </td></tr><tr><td>40%</td><td> $0 . 0 3 7 \pm 0 . 0 0 5$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 0 2 2 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 4$ </td></tr><tr><td>50%</td><td> $0 . 1 0 5 \pm 0 . 0 1 7$ </td><td> $0 . 0 8 1 \pm 0 . 0 1 8$ </td><td> $0 . 0 4 9 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 0 4 8 \pm 0 . 0 1 0 }$ </td></tr><tr><td>60%</td><td> $0 . 1 7 5 \pm 0 . 0 2 7$ </td><td> $0 . 1 6 7 \pm 0 . 0 2 1$ </td><td> $\mathbf { 0 . 0 9 5 \pm 0 . 0 2 3 }$ </td><td> $0 . 0 9 5 \pm 0 . 0 2 5$ </td></tr><tr><td>70%</td><td> $0 . 3 3 4 \pm 0 . 0 3 5$ </td><td> $0 . 3 4 6 \pm 0 . 0 4 1$ </td><td> $\mathbf { 0 . 1 8 4 \pm 0 . 0 3 4 }$ </td><td> $0 . 1 9 4 \pm 0 . 0 4 9$ </td></tr><tr><td>80%</td><td> $0 . 7 2 1 \pm 0 . 0 7 5$ </td><td> $0 . 7 2 6 \pm 0 . 0 8 0$ </td><td> $\mathbf { 0 . 4 2 5 \pm 0 . 1 1 8 }$ </td><td> $0 . 4 3 5 \pm 0 . 1 1 2$ </td></tr><tr><td>90%</td><td> $0 . 9 6 1 \pm 0 . 0 2 0$ </td><td> $0 . 9 6 8 \pm 0 . 0 1 1$ </td><td> $0 . 8 2 6 \pm 0 . 0 5 6$ </td><td> $\mathbf { 0 . 8 2 2 \pm 0 . 0 5 7 }$ </td></tr></table>

Table 13: PLS Procrustes distance across missing rates

$$
\mathrm { Z e r o \cdot f i x e d }
$$

$$
\mathrm { Z e r o \cdot a d a p t i v e }
$$

$$
\mathrm { W a r m \cdot f i x e d }
$$

$$
\mathbf { W a r m \cdot a d a p t i v e }
$$

$$
0 . 0 0 5 \pm 0 . 0 0 0
$$

$$
0 . 0 0 5 \pm 0 . 0 0 0
$$

$$
\mathbf { 0 . 0 0 4 \pm 0 . 0 0 0 }
$$

$$
0 . 0 0 4 \pm 0 . 0 0 0
$$

$$
0 . 0 1 2 \pm 0 . 0 0 1
$$

$$
0 . 0 1 2 \pm 0 . 0 0 1
$$

$$
0 . 0 0 9 \pm 0 . 0 0 1
$$

$$
\mathbf { 0 . 0 0 9 \pm 0 . 0 0 1 }
$$

$$
0 . 0 2 7 \pm 0 . 0 0 3
$$

$$
0 . 0 2 7 \pm 0 . 0 0 3
$$

$$
0 . 0 2 1 \pm 0 . 0 0 3
$$

$$
\mathbf { 0 . 0 2 1 \pm 0 . 0 0 3 }
$$

$$
0 . 0 5 2 \pm 0 . 0 0 6
$$

$$
0 . 0 5 2 \pm 0 . 0 0 6
$$

$$
0 . 0 4 0 \pm 0 . 0 0 2
$$

$$
\mathbf { 0 . 0 4 0 \pm 0 . 0 0 2 }
$$

$$
0 . 1 6 3 \pm 0 . 0 2 2
$$

$$
0 . 1 0 5 \pm 0 . 0 0 9
$$

$$
0 . 0 7 2 \pm 0 . 0 0 9
$$

$$
\mathbf { 0 . 0 7 1 \pm 0 . 0 0 8 }
$$

$$
0 . 2 2 0 \pm 0 . 0 2 3
$$

$$
0 . 1 9 4 \pm 0 . 0 1 8
$$

$$
0 . 1 2 6 \pm 0 . 0 1 7
$$

$$
\mathbf { 0 . 1 2 5 \pm 0 . 0 1 5 }
$$

$$
0 . 3 7 8 \pm 0 . 0 4 0
$$

$$
0 . 3 7 8 \pm 0 . 0 3 2
$$

$$
\mathbf { 0 . 2 2 6 \pm 0 . 0 2 6 }
$$

$$
0 . 2 2 9 \pm 0 . 0 2 4
$$

$$
0 . 6 7 0 \pm 0 . 0 3 4
$$

$$
0 . 6 8 1 \pm 0 . 0 4 5
$$

$$
\mathbf { 0 . 4 2 8 \pm 0 . 0 4 5 }
$$

$$
0 . 4 3 2 \pm 0 . 0 4 9
$$

$$
0 . 8 8 3 \pm 0 . 0 2 2
$$

$$
0 . 8 9 0 \pm 0 . 0 2 2
$$

$$
\mathbf { 0 . 7 0 6 \pm 0 . 0 4 8 }
$$

$$
0 . 7 0 7 \pm 0 . 0 4 4
$$

Table 14: Pearson log-p correlation across missing rates
<table><tr><td>Rate</td><td> $\mathrm { Z e r o \cdot f i x e d }$ </td><td> $\mathrm { Z e r o \cdot a d a p t i v e }$ </td><td> $\mathrm { W a r m \cdot f i x e d }$ </td><td> $\mathbf { W a r m \cdot a d a p t i v e }$ </td></tr><tr><td>10%</td><td> $0 . 9 8 2 \pm 0 . 0 0 6$ </td><td> $0 . 9 8 2 \pm 0 . 0 0 6$ </td><td> $0 . 9 9 6 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 9 9 6 \pm 0 . 0 0 2 }$ </td></tr><tr><td>20%</td><td> $0 . 9 5 1 \pm 0 . 0 1 6$ </td><td> $0 . 9 5 1 \pm 0 . 0 1 6$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 9 9 0 \pm 0 . 0 0 2 }$ </td></tr><tr><td>30%</td><td> $0 . 9 0 6 \pm 0 . 0 2 0$ </td><td> $0 . 9 0 6 \pm 0 . 0 2 0$ </td><td> $\mathbf { 0 . 9 7 9 \pm 0 . 0 0 6 }$ </td><td> $0 . 9 7 8 \pm 0 . 0 0 5$ </td></tr><tr><td>40%</td><td> $0 . 8 5 6 \pm 0 . 0 2 0$ </td><td> $0 . 8 5 7 \pm 0 . 0 2 1$ </td><td> $0 . 9 6 2 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 9 6 2 \pm 0 . 0 0 6 }$ </td></tr><tr><td>50%</td><td> $0 . 7 6 5 \pm 0 . 0 2 4$ </td><td> $0 . 7 8 1 \pm 0 . 0 2 5$ </td><td> $\mathbf { 0 . 9 3 6 \pm 0 . 0 1 0 }$ </td><td> $0 . 9 3 5 \pm 0 . 0 0 9$ </td></tr><tr><td>60%</td><td> $0 . 6 8 2 \pm 0 . 0 3 4$ </td><td> $0 . 6 9 3 \pm 0 . 0 4 7$ </td><td> $\mathbf { 0 . 8 9 4 \pm 0 . 0 1 2 }$ </td><td> $0 . 8 9 3 \pm 0 . 0 1 4$ </td></tr><tr><td>70%</td><td> $0 . 6 0 8 \pm 0 . 0 3 6$ </td><td> $0 . 6 0 9 \pm 0 . 0 4 8$ </td><td> $0 . 8 3 0 \pm 0 . 0 3 6$ </td><td> $\mathbf { 0 . 8 3 1 \pm 0 . 0 3 4 }$ </td></tr><tr><td>80%</td><td> $0 . 4 7 2 \pm 0 . 0 8 5$ </td><td> $0 . 4 6 3 \pm 0 . 0 8 4$ </td><td> $0 . 7 4 5 \pm 0 . 0 3 2$ </td><td> $\mathbf { 0 . 7 6 0 \pm 0 . 0 3 2 }$ </td></tr><tr><td>90%</td><td> $0 . 2 9 7 \pm 0 . 0 5 3$ </td><td> $0 . 3 2 0 \pm 0 . 0 4 3$ </td><td> $0 . 6 1 5 \pm 0 . 0 4 3$ </td><td> $\mathbf { 0 . 6 3 2 \pm 0 . 0 5 4 }$ </td></tr></table>

Table 15: Runtime comparison of SVT ablation variants. Runtime is reported in seconds as mean ± standard deviation
<table><tr><td>Method</td><td> $\mathrm { M e a n } \pm { \mathrm { S t d } } .$ </td><td>Min.</td><td>Max.</td></tr><tr><td>Zero init + fixed step size</td><td> $6 . 4 4 0 \pm 1 . 6 1 6$ </td><td>4.385</td><td>12.716</td></tr><tr><td>Zero init + adaptive step size</td><td> ${ \bf 6 . 1 4 7 \pm 1 . 1 5 2 }$ </td><td>4.377</td><td>11.303</td></tr><tr><td>Warm start + fixed step size</td><td> $6 . 3 7 8 \pm 1 . 3 5 5$ </td><td>4.421</td><td>11.093</td></tr><tr><td>Warm start + adaptive step size</td><td> $6 . 4 5 9 \pm 1 . 6 6 9$ </td><td>4.229</td><td>12.460</td></tr></table>

## D.4 RUNTIME DECOMPOSITION AND SCALING

Table 16: Stage-decomposed runtime on the metabolomics dataset $( 1 9 8 \times 1 3 0 .$ , MCAR). Times presented are means over three measured runs. Pre denotes Preprocess, SVD denotes one SVD. NF denotes NuclearForest, MF denotes MissForest, and SI denotes SoftImpute.
<table><tr><td>Missing</td><td>Pre. (ms)</td><td>SVD (ms)</td><td>SVT (s)</td><td>RF (s)</td><td>NF (s)</td><td>MF (s)</td><td>SI (s)</td></tr><tr><td>20%</td><td>0.18 ms</td><td>4.09 ms</td><td>8.97 s</td><td>10.31 s</td><td>19.28 s</td><td>105.50 s</td><td>0.48 s</td></tr><tr><td>50%</td><td>0.22 ms</td><td>10.18 ms</td><td>10.28 s</td><td>7.75 s</td><td>18.03 s</td><td>76.64 s</td><td>0.80 s</td></tr><tr><td>80%</td><td>0.19 ms</td><td>12.81 ms</td><td>8.22 s</td><td>5.08 s</td><td>13.30 s</td><td>50.84 s</td><td>0.46 s</td></tr></table>

Mask generation is part of the evaluation protocol rather than the imputation procedure and is therefore excluded from the reported imputation runtime.

We conduct a scaling study using rank-10 Gaussian matrices, with 5% noise and 30% MCAR masking. Matrix size is reported as number of rows by number of columns. With 130 columns, increasing the number of rows from 1000 to 2000 approximately doubled runtime from 66.0 s to 130.0 s. However, for a matrix with 2000 rows and 1000 columns, runtime reached 6885.7 s, with RF refinement accounting for 87.0% of total runtime.

Table 17: Runtime scaling study on rank-10 Gaussian matrices with 5% noise and 30% MCAR masking.
<table><tr><td>Matrix size</td><td>Entries</td><td>SVT stage</td><td>RF stage</td><td>Total</td><td>RF share</td></tr><tr><td> $( 1 0 0 0 \times 1 3 0 )$ </td><td>130,000</td><td>19.58 s</td><td>46.44 s</td><td>66.02 s</td><td>70.3%</td></tr><tr><td> $( 2 0 0 0 \times 1 3 0 )$ </td><td>260,000</td><td>25.84 s</td><td>104.19 s</td><td>130.02 s</td><td>80.1%</td></tr><tr><td> $( 1 0 0 0 \times 5 0 0 )$ </td><td>500,000</td><td>169.73 s</td><td>653.89 s</td><td>823.62 s</td><td>79.4%</td></tr><tr><td> $( 5 0 0 0 \times 2 0 0 )$ </td><td>1,000,000</td><td>106.67 s</td><td>727.89 s</td><td>834.56 s</td><td>87.2%</td></tr><tr><td> $( 2 0 0 0 \times 1 0 0 0 )$ </td><td>2,000,000</td><td>893.83 s</td><td>5991.90 s</td><td>6885.72 s</td><td>87.0%</td></tr></table>

## E FULL NUMERICAL RESULTS

## E.1 MCAR/MAR RESULTS ON THE METABOLOMICS DATASET

Table 18: NRMSE across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 6 1 8 \pm 0 . 0 2 0$ </td><td> $0 . 8 8 8 \pm 0 . 0 1 2$ </td><td> $1 . 0 0 6 \pm 0 . 0 0 1$ </td><td> $1 . 0 2 9 \pm 0 . 0 0 2$ </td><td> $2 . 3 8 9 \pm 0 . 0 6 3$ </td><td> $0 . 8 1 1 \pm 0 . 0 2 7$ </td></tr><tr><td>20%</td><td> $\mathbf { 0 . 6 4 6 \pm 0 . 0 2 1 }$ </td><td> $0 . 9 0 1 \pm 0 . 0 0 9$ </td><td> $1 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $1 . 0 2 9 \pm 0 . 0 0 2$ </td><td> $2 . 3 5 3 \pm 0 . 0 4 3$ </td><td> $0 . 9 2 2 \pm 0 . 0 2 4$ </td></tr><tr><td>30%</td><td> $\mathbf { 0 . 6 8 0 \pm 0 . 0 0 7 }$ </td><td> $0 . 9 2 2 \pm 0 . 0 0 5$ </td><td> $1 . 0 0 6 \pm 0 . 0 0 1$ </td><td> $1 . 0 3 1 \pm 0 . 0 0 3$ </td><td> $2 . 3 4 0 \pm 0 . 0 4 9$ </td><td> $1 . 0 0 8 \pm 0 . 0 2 0$ </td></tr><tr><td>40%</td><td> $\mathbf { 0 . 7 1 9 \pm 0 . 0 0 9 }$ </td><td> $0 . 9 4 9 \pm 0 . 0 0 5$ </td><td> $1 . 0 0 7 \pm 0 . 0 0 1$ </td><td> $1 . 0 3 0 \pm 0 . 0 0 2$ </td><td> $2 . 3 2 3 \pm 0 . 0 2 7$ </td><td> $1 . 1 1 3 \pm 0 . 0 2 5$ </td></tr><tr><td>50%</td><td> $\mathbf { 0 . 7 6 0 \pm 0 . 0 0 8 }$ </td><td> $0 . 9 7 0 \pm 0 . 0 0 8$ </td><td> $1 . 0 0 7 \pm 0 . 0 0 1$ </td><td> $1 . 0 3 1 \pm 0 . 0 0 3$ </td><td> $2 . 3 1 1 \pm 0 . 0 3 0$ </td><td> $1 . 6 5 6 \pm 0 . 0 5 8$ </td></tr><tr><td>60%</td><td> $\mathbf { 0 . 8 1 2 \pm 0 . 0 0 9 }$ </td><td> $1 . 0 0 1 \pm 0 . 0 1 0$ </td><td> $1 . 0 0 8 \pm 0 . 0 0 1$ </td><td> $1 . 0 3 1 \pm 0 . 0 0 2$ </td><td> $2 . 3 1 4 \pm 0 . 0 1 6$ </td><td> $1 . 6 2 0 \pm 0 . 0 7 7$ </td></tr><tr><td>70%</td><td> $0 . 8 7 5 \pm 0 . 0 0 6$ </td><td> $1 . 0 6 1 \pm 0 . 0 0 8$ </td><td> $1 . 0 1 0 \pm 0 . 0 0 1$ </td><td> $1 . 0 3 3 \pm 0 . 0 0 2$ </td><td> $2 . 2 9 1 \pm 0 . 0 2 1$ </td><td> $1 . 7 0 9 \pm 0 . 0 3 2$ </td></tr><tr><td>80%</td><td> $0 . 9 4 7 \pm 0 . 0 0 7$ </td><td> $1 . 1 4 8 \pm 0 . 0 1 6$ </td><td> $1 . 0 1 3 \pm 0 . 0 0 2$ </td><td> $1 . 0 3 5 \pm 0 . 0 0 4$ </td><td> $2 . 2 6 5 \pm 0 . 0 3 3$ </td><td> $2 . 0 6 3 \pm 0 . 0 7 8$ </td></tr><tr><td>90%</td><td> $1 . 0 4 1 \pm 0 . 0 1 5$ </td><td> $1 . 1 9 7 \pm 0 . 0 1 5$ </td><td> $1 . 0 2 0 \pm 0 . 0 0 3$ </td><td> $1 . 0 4 0 \pm 0 . 0 0 6$ </td><td> $2 . 2 1 8 \pm 0 . 0 1 7$ </td><td> $2 . 7 6 0 \pm 0 . 0 6 3$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 6 1 7 \pm 0 . 0 1 9$ </td><td> $0 . 8 6 6 \pm 0 . 0 1 9$ </td><td> $0 . 6 2 8 \pm 0 . 0 2 0$ </td><td> $0 . 7 3 6 \pm 0 . 0 4 3$ </td><td> $\mathbf { 0 . 6 1 4 \pm 0 . 0 2 0 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 6 5 7 \pm 0 . 0 1 9$ </td><td> $0 . 8 7 4 \pm 0 . 0 2 1$ </td><td> $0 . 6 5 9 \pm 0 . 0 2 1$ </td><td> $0 . 8 1 0 \pm 0 . 0 2 5$ </td><td> $0 . 6 4 7 \pm 0 . 0 1 9$ </td><td></td></tr><tr><td>30%</td><td> $0 . 6 9 7 \pm 0 . 0 1 2$ </td><td> $0 . 9 0 1 \pm 0 . 0 1 5$ </td><td> $0 . 7 0 1 \pm 0 . 0 1 0$ </td><td> $0 . 8 7 4 \pm 0 . 0 1 6$ </td><td> $0 . 6 8 5 \pm 0 . 0 1 0$ </td><td></td></tr><tr><td>40%</td><td> $0 . 7 4 5 \pm 0 . 0 0 6$ </td><td> $0 . 9 3 2 \pm 0 . 0 1 1$ </td><td> $0 . 7 4 4 \pm 0 . 0 0 7$ </td><td> $0 . 9 3 8 \pm 0 . 0 2 6$ </td><td> $0 . 7 2 4 \pm 0 . 0 0 7$ </td><td></td></tr><tr><td>50%</td><td> $0 . 8 0 9 \pm 0 . 0 0 6$ </td><td> $0 . 9 8 6 \pm 0 . 0 1 5$ </td><td> $0 . 7 8 7 \pm 0 . 0 0 8$ </td><td> $1 . 0 2 7 \pm 0 . 0 1 7$ </td><td> $0 . 7 6 8 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>60%</td><td> $0 . 8 5 8 \pm 0 . 0 0 5$ </td><td> $1 . 0 6 0 \pm 0 . 0 1 8$ </td><td> $0 . 8 3 9 \pm 0 . 0 0 7$ </td><td> $1 . 1 2 7 \pm 0 . 0 2 2$ </td><td>0.819 ± 0.007</td><td></td></tr><tr><td>70%</td><td> $0 . 9 2 2 \pm 0 . 0 0 5$ </td><td> $1 . 1 8 3 \pm 0 . 0 2 5$ </td><td> $0 . 9 0 5 \pm 0 . 0 0 7$ </td><td> $1 . 2 5 1 \pm 0 . 0 2 9$ </td><td> $\mathbf { 0 . 8 7 3 \pm 0 . 0 0 7 }$ </td><td></td></tr><tr><td>80%</td><td> $1 . 0 0 2 \pm 0 . 0 0 7$ </td><td> $1 . 3 7 4 \pm 0 . 0 2 0$ </td><td> $0 . 9 8 6 \pm 0 . 0 1 0$ </td><td> $1 . 3 7 8 \pm 0 . 0 3 6$ </td><td> $\mathbf { 0 . 9 4 5 \pm 0 . 0 0 6 }$ </td><td></td></tr><tr><td>90%</td><td> $1 . 0 6 1 \pm 0 . 0 0 8$ </td><td> $1 . 9 4 7 \pm 0 . 0 5 9$ </td><td> $1 . 0 6 7 \pm 0 . 0 1 0$ </td><td> $1 . 3 2 7 \pm 0 . 0 3 5$ </td><td> $\mathbf { 1 . 0 1 7 \pm 0 . 0 0 6 }$ </td><td></td></tr></table>

Table 19: PCA Procrustes distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td> $_ \mathrm { H a l f - m i n }$ </td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 7 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 7 \pm 0 . 0 0 1$ </td><td> $0 . 0 3 8 \pm 0 . 0 0 7$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td></tr><tr><td>20%</td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 6 \pm 0 . 0 0 3$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 3$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 4$ </td><td> $0 . 0 8 1 \pm 0 . 0 1 3$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 2$ </td></tr><tr><td>30%</td><td> $\mathbf { 0 . 0 1 0 \pm 0 . 0 0 2 }$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 7$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 6$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 7$ </td><td> $0 . 1 5 3 \pm 0 . 0 4 3$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 4$ </td></tr><tr><td>40%</td><td> $0 . 0 2 0 \pm 0 . 0 0 3$ </td><td> $0 . 0 6 2 \pm 0 . 0 2 0$ </td><td> $0 . 0 6 2 \pm 0 . 0 1 6$ </td><td> $0 . 0 6 4 \pm 0 . 0 1 5$ </td><td> $0 . 2 4 3 \pm 0 . 0 4 1$ </td><td> $0 . 0 3 7 \pm 0 . 0 0 5$ </td></tr><tr><td>50%</td><td> $\mathbf { 0 . 0 3 0 \pm 0 . 0 0 4 }$ </td><td> $0 . 1 0 1 \pm 0 . 0 2 8$ </td><td> $0 . 0 9 9 \pm 0 . 0 3 5$ </td><td> $0 . 1 0 1 \pm 0 . 0 3 4$ </td><td> $0 . 4 0 0 \pm 0 . 0 7 7$ </td><td> $0 . 1 0 5 \pm 0 . 0 1 7$ </td></tr><tr><td>60%</td><td> $\mathbf { 0 . 0 5 3 \pm 0 . 0 1 0 }$ </td><td> $0 . 1 6 4 \pm 0 . 0 4 9$ </td><td> $0 . 2 1 5 \pm 0 . 0 6 6$ </td><td> $0 . 2 2 1 \pm 0 . 0 7 3$ </td><td> $0 . 6 2 5 \pm 0 . 0 7 2$ </td><td> $0 . 1 7 5 \pm 0 . 0 2 7$ </td></tr><tr><td>70%</td><td> $\mathbf { 0 . 0 9 2 \pm 0 . 0 1 3 }$ </td><td> $0 . 2 9 9 \pm 0 . 0 3 0$ </td><td> $0 . 3 5 9 \pm 0 . 1 1 5$ </td><td> $0 . 3 6 5 \pm 0 . 1 1 8$ </td><td> $0 . 8 0 2 \pm 0 . 0 3 6$ </td><td> $0 . 3 3 4 \pm 0 . 0 3 5$ </td></tr><tr><td>80%</td><td> $\mathbf { 0 . 2 1 4 \pm 0 . 0 4 1 }$ </td><td> $0 . 6 0 8 \pm 0 . 1 0 4$ </td><td> $0 . 5 9 6 \pm 0 . 0 9 9$ </td><td> $0 . 6 0 0 \pm 0 . 0 9 6$ </td><td> $0 . 9 1 0 \pm 0 . 0 2 9$ </td><td> $0 . 7 2 1 \pm 0 . 0 7 5$ </td></tr><tr><td>90%</td><td> $\mathbf { 0 . 5 6 3 \pm 0 . 0 7 2 }$ </td><td> $0 . 8 3 3 \pm 0 . 0 5 9$ </td><td> $0 . 8 4 5 \pm 0 . 0 4 3$ </td><td> $0 . 8 4 8 \pm 0 . 0 3 7$ </td><td> $0 . 9 7 2 \pm 0 . 0 0 8$ </td><td> $0 . 9 6 1 \pm 0 . 0 2 0$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 0 0 2 \pm 0 . 0 0 1 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 2 0 \pm 0 . 0 0 7$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 0 0 5 \pm 0 . 0 0 1 }$ </td><td></td></tr><tr><td>30%</td><td> $0 . 0 1 2 \pm 0 . 0 0 3$ </td><td> $0 . 0 4 8 \pm 0 . 0 1 4$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 2$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 3$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 3$ </td><td></td></tr><tr><td>40%</td><td> $0 . 0 2 3 \pm 0 . 0 0 4$ </td><td> $0 . 0 8 6 \pm 0 . 0 1 5$ </td><td> $0 . 0 2 5 \pm 0 . 0 0 3$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 0 1 9 \pm 0 . 0 0 3 }$ </td><td></td></tr><tr><td>50%</td><td> $0 . 0 4 0 \pm 0 . 0 0 7$ </td><td> $0 . 1 7 4 \pm 0 . 0 2 5$ </td><td> $0 . 0 4 9 \pm 0 . 0 0 8$ </td><td> $0 . 0 4 8 \pm 0 . 0 1 0$ </td><td> $0 . 0 3 2 \pm 0 . 0 0 5$ </td><td></td></tr><tr><td>60%</td><td> $0 . 0 7 9 \pm 0 . 0 0 8$ </td><td> $0 . 2 8 4 \pm 0 . 0 3 8$ </td><td> $0 . 0 9 5 \pm 0 . 0 1 1$ </td><td> $0 . 0 9 5 \pm 0 . 0 2 5$ </td><td> $0 . 0 6 1 \pm 0 . 0 0 7$ </td><td></td></tr><tr><td>70%</td><td> $0 . 1 6 1 \pm 0 . 0 2 6$ </td><td> $0 . 4 5 8 \pm 0 . 0 4 5$ </td><td> $0 . 1 9 9 \pm 0 . 0 3 4$ </td><td> $0 . 1 9 4 \pm 0 . 0 4 9$ </td><td> $0 . 1 2 2 \pm 0 . 0 2 5$ </td><td></td></tr><tr><td>80%</td><td> $0 . 3 8 8 \pm 0 . 0 4 7$ </td><td> $0 . 6 8 0 \pm 0 . 0 6 1$ </td><td> $0 . 4 2 5 \pm 0 . 0 5 7$ </td><td> $0 . 4 3 5 \pm 0 . 1 1 2$ </td><td> $0 . 2 7 3 \pm 0 . 0 3 9$ </td><td></td></tr><tr><td>90%</td><td> $0 . 7 7 0 \pm 0 . 0 4 6$ </td><td> $0 . 9 3 9 \pm 0 . 0 3 1$ </td><td> $0 . 8 3 7 \pm 0 . 0 4 0$ </td><td> $0 . 8 2 2 \pm 0 . 0 5 7$ </td><td> $0 . 6 4 1 \pm 0 . 0 3 9$ </td><td></td></tr></table>

Table 20: PLS Procrustes distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 1$ </td><td> $0 . 0 5 7 \pm 0 . 0 0 8$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 0$ </td></tr><tr><td>20%</td><td> $0 . 0 0 8 \pm 0 . 0 0 1$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 1$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 1$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 1$ </td><td> $0 . 1 1 6 \pm 0 . 0 1 5$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 1$ </td></tr><tr><td>30%</td><td> $0 . 0 1 6 \pm 0 . 0 0 2$ </td><td> $0 . 0 4 3 \pm 0 . 0 0 5$ </td><td> $0 . 0 4 0 \pm 0 . 0 0 6$ </td><td> $0 . 0 4 2 \pm 0 . 0 0 6$ </td><td> $0 . 1 9 3 \pm 0 . 0 3 4$ </td><td> $0 . 0 2 7 \pm 0 . 0 0 3$ </td></tr><tr><td>40%</td><td> $\mathbf { 0 . 0 2 7 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 6 9 \pm 0 . 0 0 9$ </td><td> $0 . 0 6 5 \pm 0 . 0 0 9$ </td><td> $0 . 0 6 7 \pm 0 . 0 0 9$ </td><td> $0 . 2 7 4 \pm 0 . 0 2 3$ </td><td> $0 . 0 5 2 \pm 0 . 0 0 6$ </td></tr><tr><td>50%</td><td> $\mathbf { 0 . 0 4 3 \pm 0 . 0 0 2 }$ </td><td> $0 . 1 0 1 \pm 0 . 0 0 9$ </td><td> $0 . 0 9 7 \pm 0 . 0 1 1$ </td><td> $0 . 0 9 8 \pm 0 . 0 1 2$ </td><td> $0 . 3 7 1 \pm 0 . 0 3 7$ </td><td> $0 . 1 6 3 \pm 0 . 0 2 2$ </td></tr><tr><td>60%</td><td> $\mathbf { 0 . 0 7 3 \pm 0 . 0 1 0 }$ </td><td> $0 . 1 6 6 \pm 0 . 0 2 3$ </td><td> $0 . 1 7 7 \pm 0 . 0 1 4$ </td><td> $0 . 1 8 1 \pm 0 . 0 1 7$ </td><td> $0 . 5 3 7 \pm 0 . 0 3 6$ </td><td> $0 . 2 2 0 \pm 0 . 0 2 3$ </td></tr><tr><td>70%</td><td> $\mathbf { 0 . 1 } 2 2 \pm \mathbf { 0 . 0 1 } 2$ </td><td> $0 . 3 1 3 \pm 0 . 0 3 1$ </td><td> $0 . 2 9 1 \pm 0 . 0 5 8$ </td><td> $0 . 2 9 5 \pm 0 . 0 6 0$ </td><td> $0 . 6 7 5 \pm 0 . 0 3 9$ </td><td> $0 . 3 7 8 \pm 0 . 0 4 0$ </td></tr><tr><td>80%</td><td> $\mathbf { 0 . 2 3 5 \pm 0 . 0 2 3 }$ </td><td> $0 . 5 2 5 \pm 0 . 0 5 3$ </td><td> $0 . 4 7 6 \pm 0 . 0 6 1$ </td><td> $0 . 4 7 4 \pm 0 . 0 6 4$ </td><td> $0 . 7 9 7 \pm 0 . 0 3 0$ </td><td> $0 . 6 7 0 \pm 0 . 0 3 4$ </td></tr><tr><td>90%</td><td> $\mathbf { 0 . 5 1 6 \pm 0 . 0 5 7 }$ </td><td> $0 . 7 0 9 \pm 0 . 0 4 2$ </td><td> $0 . 7 1 5 \pm 0 . 0 3 9$ </td><td> $0 . 7 1 8 \pm 0 . 0 3 6$ </td><td> $0 . 8 6 7 \pm 0 . 0 4 5$ </td><td> $0 . 8 8 3 \pm 0 . 0 2 2$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td> $\mathrm { S o f t I m p u t e }$ </td><td> $\mathrm { S o f t F o r e s t }$ </td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 0 0 3 \pm 0 . 0 0 1 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 0 0 8 \pm 0 . 0 0 1$ </td><td> $0 . 0 2 0 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 8 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 0 0 7 \pm 0 . 0 0 1 }$ </td><td></td></tr><tr><td>30%</td><td> $0 . 0 1 6 \pm 0 . 0 0 3$ </td><td> $0 . 0 5 1 \pm 0 . 0 0 7$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 3$ </td><td> $0 . 0 2 1 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 0 1 5 \pm 0 . 0 0 2 }$ </td><td></td></tr><tr><td>40%</td><td> $0 . 0 2 9 \pm 0 . 0 0 2$ </td><td> $0 . 1 0 9 \pm 0 . 0 1 6$ </td><td> $0 . 0 3 8 \pm 0 . 0 0 3$ </td><td> $0 . 0 4 0 \pm 0 . 0 0 2$ </td><td> $0 . 0 2 8 \pm 0 . 0 0 2$ </td><td></td></tr><tr><td>50%</td><td> $0 . 0 5 4 \pm 0 . 0 0 3$ </td><td> $0 . 2 3 0 \pm 0 . 0 3 1$ </td><td> $0 . 0 7 1 \pm 0 . 0 0 6$ </td><td> $0 . 0 7 1 \pm 0 . 0 0 8$ </td><td> $0 . 0 4 8 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>60%</td><td> $0 . 1 0 0 \pm 0 . 0 1 0$ </td><td> $0 . 3 7 6 \pm 0 . 0 4 8$ </td><td> $0 . 1 3 2 \pm 0 . 0 2 7$ </td><td> $0 . 1 2 5 \pm 0 . 0 1 5$ </td><td> $0 . 0 8 4 \pm 0 . 0 1 0$ </td><td></td></tr><tr><td>70%</td><td> $0 . 1 9 2 \pm 0 . 0 2 2$ </td><td> $0 . 5 5 5 \pm 0 . 0 2 9$ </td><td> $0 . 2 7 8 \pm 0 . 0 3 2$ </td><td> $0 . 2 2 9 \pm 0 . 0 2 4$ </td><td> $0 . 1 6 2 \pm 0 . 0 1 7$ </td><td></td></tr><tr><td>80%</td><td> $0 . 4 1 6 \pm 0 . 0 4 3$ </td><td> $0 . 6 9 6 \pm 0 . 0 4 8$ </td><td> $0 . 4 6 7 \pm 0 . 0 5 4$ </td><td> $0 . 4 3 2 \pm 0 . 0 4 9$ </td><td> $0 . 3 3 6 \pm 0 . 0 4 8$ </td><td></td></tr><tr><td>90%</td><td> $0 . 7 1 3 \pm 0 . 0 3 1$ </td><td> $0 . 8 8 9 \pm 0 . 0 2 2$ </td><td> $0 . 7 3 2 \pm 0 . 0 3 4$ </td><td> $0 . 7 0 7 \pm 0 . 0 4 4$ </td><td> $0 . 6 2 7 \pm 0 . 0 3 1$ </td><td></td></tr></table>

Table 21: Pearson log-p correlation across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 9 9 5 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 3 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 1 \pm 0 . 0 0 3$ </td><td> $0 . 8 9 2 \pm 0 . 0 1 8$ </td><td> $0 . 9 8 2 \pm 0 . 0 0 6$ </td></tr><tr><td>20%</td><td> $0 . 9 9 1 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 2$ </td><td> $0 . 9 8 3 \pm 0 . 0 0 3$ </td><td> $0 . 9 8 1 \pm 0 . 0 0 4$ </td><td> $0 . 8 1 4 \pm 0 . 0 3 7$ </td><td> $0 . 9 5 1 \pm 0 . 0 1 6$ </td></tr><tr><td>30%</td><td> $\mathbf { 0 . 9 8 4 \pm 0 . 0 0 4 }$ </td><td> $0 . 9 7 5 \pm 0 . 0 0 4$ </td><td> $0 . 9 7 0 \pm 0 . 0 0 6$ </td><td> $0 . 9 6 6 \pm 0 . 0 0 7$ </td><td> $0 . 7 1 9 \pm 0 . 0 4 7$ </td><td> $0 . 9 0 6 \pm 0 . 0 2 0$ </td></tr><tr><td>40%</td><td> $\mathbf { 0 . 9 7 2 \pm 0 . 0 0 7 }$ </td><td> $0 . 9 5 8 \pm 0 . 0 1 0$ </td><td> $0 . 9 5 7 \pm 0 . 0 0 8$ </td><td> $0 . 9 5 2 \pm 0 . 0 0 8$ </td><td> $0 . 6 6 6 \pm 0 . 0 8 0$ </td><td> $0 . 8 5 6 \pm 0 . 0 2 0$ </td></tr><tr><td>50%</td><td> $\mathbf { 0 . 9 5 6 \pm 0 . 0 0 8 }$ </td><td> $0 . 9 4 2 \pm 0 . 0 0 8$ </td><td> $0 . 9 4 0 \pm 0 . 0 1 0$ </td><td> $0 . 9 3 1 \pm 0 . 0 1 2$ </td><td> $0 . 5 8 2 \pm 0 . 0 7 2$ </td><td> $0 . 7 6 5 \pm 0 . 0 2 4$ </td></tr><tr><td>60%</td><td> $\mathbf { 0 . 9 3 8 \pm 0 . 0 1 0 }$ </td><td> $0 . 9 1 6 \pm 0 . 0 1 3$ </td><td> $0 . 9 0 9 \pm 0 . 0 1 7$ </td><td> $0 . 9 0 0 \pm 0 . 0 2 2$ </td><td> $0 . 5 1 1 \pm 0 . 0 8 8$ </td><td> $0 . 6 8 2 \pm 0 . 0 3 4$ </td></tr><tr><td>70%</td><td> $\mathbf { 0 . 8 9 7 \pm 0 . 0 1 6 }$ </td><td> $0 . 8 6 8 \pm 0 . 0 3 9$ </td><td> $0 . 8 8 6 \pm 0 . 0 3 0$ </td><td> $0 . 8 6 7 \pm 0 . 0 3 4$ </td><td> $0 . 4 6 4 \pm 0 . 1 1 5$ </td><td> $0 . 6 0 8 \pm 0 . 0 3 6$ </td></tr><tr><td>80%</td><td> $\mathbf { 0 . 8 3 8 \pm 0 . 0 2 4 }$ </td><td> $0 . 7 2 0 \pm 0 . 0 3 1$ </td><td> $0 . 8 1 0 \pm 0 . 0 2 9$ </td><td> $0 . 7 8 4 \pm 0 . 0 3 9$ </td><td> $0 . 3 0 7 \pm 0 . 1 6 4$ </td><td> $0 . 4 7 2 \pm 0 . 0 8 5$ </td></tr><tr><td>90%</td><td> $0 . 7 1 9 \pm 0 . 0 4 7$ </td><td> $0 . 4 7 9 \pm 0 . 0 8 0$ </td><td> $0 . 7 2 3 \pm 0 . 0 4 6$ </td><td> $0 . 7 0 4 \pm 0 . 0 4 6$ </td><td> $0 . 3 1 1 \pm 0 . 1 1 4$ </td><td> $0 . 2 9 7 \pm 0 . 0 5 3$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 9 9 5 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 1 \pm 0 . 0 0 4$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 9 9 6 \pm 0 . 0 0 2 }$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>20%</td><td> $0 . 9 9 1 \pm 0 . 0 0 2$ </td><td> $0 . 9 7 6 \pm 0 . 0 0 6$ </td><td> $0 . 9 9 1 \pm 0 . 0 0 2$ </td><td> $0 . 9 9 0 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 9 9 1 \pm 0 . 0 0 2 }$ </td><td></td></tr><tr><td>30%</td><td> $0 . 9 8 3 \pm 0 . 0 0 5$ </td><td> $0 . 9 5 4 \pm 0 . 0 0 5$ </td><td> $0 . 9 8 2 \pm 0 . 0 0 5$ </td><td> $0 . 9 7 8 \pm 0 . 0 0 5$ </td><td> $0 . 9 8 3 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>40%</td><td> $0 . 9 6 8 \pm 0 . 0 0 6$ </td><td> $0 . 9 2 2 \pm 0 . 0 0 8$ </td><td> $0 . 9 6 7 \pm 0 . 0 0 7$ </td><td> $0 . 9 6 2 \pm 0 . 0 0 6$ </td><td> $0 . 9 6 8 \pm 0 . 0 0 6$ </td><td></td></tr><tr><td>50%</td><td> $0 . 9 5 5 \pm 0 . 0 0 9$ </td><td> $0 . 8 7 5 \pm 0 . 0 0 7$ </td><td> $0 . 9 4 8 \pm 0 . 0 0 7$ </td><td> $0 . 9 3 5 \pm 0 . 0 0 9$ </td><td> $0 . 9 5 3 \pm 0 . 0 0 6$ </td><td></td></tr><tr><td>60%</td><td> $0 . 9 3 2 \pm 0 . 0 0 7$ </td><td> $0 . 8 0 7 \pm 0 . 0 2 4$ </td><td> $0 . 9 2 0 \pm 0 . 0 1 4$ </td><td> $0 . 8 9 3 \pm 0 . 0 1 4$ </td><td> $0 . 9 3 1 \pm 0 . 0 0 8$ </td><td></td></tr><tr><td>70%</td><td> $0 . 8 8 8 \pm 0 . 0 1 8$ </td><td> $0 . 7 3 9 \pm 0 . 0 3 3$ </td><td> $0 . 8 7 6 \pm 0 . 0 1 7$ </td><td> $0 . 8 3 1 \pm 0 . 0 3 4$ </td><td> $0 . 8 9 0 \pm 0 . 0 1 4$ </td><td></td></tr><tr><td>80%</td><td> $0 . 8 2 4 \pm 0 . 0 3 2$ </td><td> $0 . 6 3 0 \pm 0 . 0 4 8$ </td><td> $0 . 8 0 2 \pm 0 . 0 3 3$ </td><td> $0 . 7 6 0 \pm 0 . 0 3 2$ </td><td> $0 . 8 2 7 \pm 0 . 0 3 2$ </td><td></td></tr><tr><td>90%</td><td> $0 . 7 1 4 \pm 0 . 0 5 0$ </td><td> $0 . 4 5 8 \pm 0 . 0 7 0$ </td><td> $0 . 6 7 1 \pm 0 . 0 7 7$ </td><td> $0 . 6 3 2 \pm 0 . 0 5 4$ </td><td> $\mathbf { 0 . 7 2 4 \pm 0 . 0 5 6 }$ </td><td></td></tr></table>

## E.2 MNAR RESULTS ON THE METABOLOMICS DATASET

Table 22: SOR across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $8 1 . 2 0 0 \pm 7 . 8 1 5$ </td><td> $1 1 1 . 4 0 0 \pm 7 . 5 7 5$ </td><td> $1 2 6 . 9 0 0 \pm 5 . 9 9 0$ </td><td> $9 7 . 6 0 0 \pm 7 . 6 1 9$ </td><td> $8 3 . 9 0 0 \pm 1 2 . 9 9 1$ </td><td> $\mathbf { 4 9 . 9 0 0 \pm 9 . 2 3 1 }$ </td></tr><tr><td>20%</td><td> $1 5 8 . 3 0 0 \pm 9 . 0 3 1$ </td><td> $2 1 6 . 8 0 0 \pm 7 . 4 9 5$ </td><td> $2 5 3 . 0 0 0 \pm 1 3 . 2 4 1$ </td><td> $1 8 2 . 7 0 0 \pm 1 4 . 2 4 4$ </td><td> $1 5 3 . 3 0 0 \pm 1 8 . 2 5 2$ </td><td> $1 1 9 . 9 0 0 \pm 1 4 . 8 7 3$ </td></tr><tr><td>30%</td><td> $2 4 5 . 0 0 0 \pm 1 9 . 0 6 7$ </td><td> $3 2 8 . 3 0 0 \pm 1 2 . 6 5 0$ </td><td> $3 8 0 . 2 0 0 \pm 1 1 . 0 1 3$ </td><td> $2 7 1 . 7 0 0 \pm 1 6 . 2 0 7$ </td><td> $2 2 2 . 6 0 0 \pm 2 7 . 2 4 0$ </td><td> $1 6 2 . 3 0 0 \pm 1 5 . 8 0 5$ </td></tr><tr><td>40%</td><td> $3 2 7 . 6 0 0 \pm 1 7 . 4 9 4$ </td><td> $4 3 4 . 6 0 0 \pm 1 2 . 0 1 1$ </td><td> $5 0 9 . 4 0 0 \pm 1 3 . 0 3 2$ </td><td> $3 6 9 . 0 0 0 \pm 2 2 . 7 8 4$ </td><td> $2 7 9 . 4 0 0 \pm 1 9 . 8 7 3$ </td><td> $2 1 5 . 3 0 0 \pm 1 6 . 6 8 7$ </td></tr><tr><td>50%</td><td> $3 9 5 . 6 0 0 \pm 2 7 . 2 9 3$ </td><td> $5 3 8 . 5 0 0 \pm 1 3 . 9 3 8$ </td><td> $6 2 4 . 7 0 0 \pm 1 0 . 6 0 5$ </td><td> $4 4 1 . 9 0 0 \pm 1 9 . 4 5 1$ </td><td> $3 5 6 . 3 0 0 \pm 2 9 . 2 0 8$ </td><td> $3 0 6 . 2 0 0 \pm 1 7 . 4 1 5$ </td></tr><tr><td>60%</td><td> $4 8 0 . 6 0 0 \pm 2 3 . 1 1 4$ </td><td> $6 5 4 . 1 0 0 \pm 2 3 . 4 7 3$ </td><td> $7 4 0 . 5 0 0 \pm 1 6 . 9 3 9$ </td><td> $5 1 8 . 1 0 0 \pm 1 6 . 4 6 2$ </td><td> $3 9 8 . 6 0 0 \pm 2 3 . 1 7 7$ </td><td> $3 4 5 . 9 0 0 \pm 2 8 . 1 3 6$ </td></tr><tr><td>70%</td><td> $5 6 1 . 4 0 0 \pm 3 0 . 8 3 7$ </td><td> $7 6 1 . 7 0 0 \pm 2 0 . 2 0 5$ </td><td> $8 6 5 . 6 0 0 \pm 1 2 . 7 0 3$ </td><td> $5 9 1 . 1 0 0 \pm 1 9 . 1 4 0$ </td><td> $4 7 5 . 9 0 0 \pm 4 1 . 7 5 7$ </td><td> $4 1 4 . 2 0 0 \pm 2 9 . 5 0 6$ </td></tr><tr><td>80%</td><td> $5 9 1 . 7 0 0 \pm 3 3 . 4 5 0$ </td><td> $8 8 1 . 3 0 0 \pm 1 9 . 7 9 9$ </td><td> $9 6 1 . 3 0 0 \pm 1 3 . 7 4 4$ </td><td> $6 3 4 . 9 0 0 \pm 1 9 . 5 6 4$ </td><td> $5 2 1 . 0 0 0 \pm 3 8 . 0 4 1$ </td><td> $5 1 6 . 9 0 0 \pm 1 8 . 1 5 0$ </td></tr><tr><td>90%</td><td> $6 4 1 . 1 0 0 \pm 3 0 . 1 1 3$ </td><td> $9 7 3 . 9 0 0 \pm 4 4 . 1 7 5$ </td><td> $1 0 6 2 . 7 0 0 \pm 2 4 . 5 0 4$ </td><td> $6 8 6 . 4 0 0 \pm 3 2 . 8 3 4$ </td><td> $5 6 6 . 4 0 0 \pm 2 9 . 7 3 7$ </td><td> $6 2 0 . 5 0 0 \pm 7 2 . 9 6 8$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $5 5 . 9 0 0 \pm 1 1 . 3 9 6$ </td><td> $5 5 . 4 0 0 \pm 1 0 . 0 9 1$ </td><td> $6 8 . 4 0 0 \pm 1 0 . 1 8 9$ </td><td> $6 5 . 8 0 0 \pm 1 2 . 7 1 7$ </td><td> $6 1 . 6 0 0 \pm 6 . 0 7 7$ </td><td></td></tr><tr><td>20%</td><td> $\mathbf { 1 0 0 . 5 0 0 \pm 1 5 . 6 3 6 }$ </td><td> $1 3 1 . 8 0 0 \pm 1 5 . 0 8 3$ </td><td> $1 2 3 . 7 0 0 \pm 1 1 . 0 3 6$ </td><td> $1 4 5 . 8 0 0 \pm 1 2 . 4 1 7$ </td><td> $1 3 0 . 2 0 0 \pm 1 4 . 5 8 2$ </td><td></td></tr><tr><td>30%</td><td> $\mathbf { 1 5 3 . 5 0 0 \pm 1 4 . 5 3 9 }$ </td><td> $1 8 9 . 8 0 0 \pm 2 4 . 7 6 9$ </td><td> $1 9 5 . 4 0 0 \pm 1 1 . 8 0 6$ </td><td> $2 1 5 . 1 0 0 \pm 1 6 . 2 1 0$ </td><td> $2 1 0 . 1 0 0 \pm 1 6 . 3 4 7$ </td><td></td></tr><tr><td>40%</td><td> $\mathbf { 2 1 0 . 5 0 0 \pm 1 7 . 0 9 6 }$ </td><td> $2 3 6 . 7 0 0 \pm 1 3 . 0 9 8$ </td><td> $2 6 0 . 4 0 0 \pm 1 9 . 6 1 4$ </td><td> $3 0 0 . 0 0 0 \pm 1 9 . 2 1 8$ </td><td> $2 8 9 . 1 0 0 \pm 1 4 . 8 3 6$ </td><td></td></tr><tr><td>50%</td><td> ${ \pm 5 4 . 9 0 0 \pm 2 2 . 3 5 8 }$ </td><td> $2 8 8 . 7 0 0 \pm 3 2 . 6 1 9$ </td><td> $3 1 1 . 1 0 0 \pm 2 8 . 0 8 9$ </td><td> $4 1 0 . 7 0 0 \pm 2 3 . 5 6 6$ </td><td> $3 6 1 . 4 0 0 \pm 1 4 . 1 9 1$ </td><td></td></tr><tr><td>60%</td><td> $\mathbf { 3 2 2 . 3 0 0 \pm 1 9 . 5 7 9 }$ </td><td> $3 3 2 . 8 0 0 \pm 2 6 . 4 6 9$ </td><td> $4 0 5 . 8 0 0 \pm 2 5 . 2 0 9$ </td><td> $4 9 3 . 7 0 0 \pm 1 8 . 7 7 4$ </td><td> $4 5 5 . 6 0 0 \pm 2 3 . 4 0 6$ </td><td></td></tr><tr><td>70%</td><td> $\mathbf { 3 5 1 . 7 0 0 \pm 3 3 . 6 5 5 }$ </td><td> $3 8 0 . 1 0 0 \pm 2 0 . 3 5 5$ </td><td> $4 5 0 . 9 0 0 \pm 3 0 . 2 8 2$ </td><td> $6 0 5 . 8 0 0 \pm 4 7 . 7 5 6$ </td><td> $5 4 7 . 6 0 0 \pm 2 5 . 7 4 7$ </td><td></td></tr><tr><td>80%</td><td> $\mathbf { 4 1 4 . 6 0 0 \pm 3 4 . 8 7 8 }$ </td><td> $4 1 6 . 4 0 0 \pm 2 0 . 5 7 6$ </td><td> $4 9 3 . 5 0 0 \pm 1 5 . 2 7 0$ </td><td> $7 6 2 . 5 0 0 \pm 3 0 . 3 7 6$ </td><td> $6 6 9 . 9 0 0 \pm 1 4 . 6 8 5$ </td><td></td></tr><tr><td>90%</td><td> $4 5 4 . 6 0 0 \pm 1 8 . 4 7 6$ </td><td> $\mathbf { 4 5 0 . 2 0 0 \pm 3 2 . 3 6 2 }$ </td><td> $5 2 9 . 0 0 0 \pm 2 9 . 4 6 2$ </td><td> $9 3 8 . 5 0 0 \pm 3 3 . 0 1 6$ </td><td> $7 9 8 . 7 0 0 \pm 3 1 . 9 9 7$ </td><td></td></tr></table>

Table 23: PCA Procrustes distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td> $_ \mathrm { H a l f - m i n }$ </td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 2$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 0 1 \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 1$ </td></tr><tr><td>20%</td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 5$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 5$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 0 0 2 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 2$ </td></tr><tr><td>30%</td><td> $0 . 0 1 1 \pm 0 . 0 0 3$ </td><td> $0 . 0 3 8 \pm 0 . 0 1 8$ </td><td> $0 . 0 3 7 \pm 0 . 0 1 7$ </td><td> $0 . 0 2 9 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 0 0 4 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 2$ </td></tr><tr><td>40%</td><td> $0 . 0 2 1 \pm 0 . 0 1 1$ </td><td> $0 . 0 5 6 \pm 0 . 0 2 1$ </td><td> $0 . 0 6 6 \pm 0 . 0 3 5$ </td><td> $0 . 0 4 7 \pm 0 . 0 2 0$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 0 0 6 \pm 0 . 0 0 2 }$ </td></tr><tr><td>50%</td><td> $0 . 0 3 2 \pm 0 . 0 0 8$ </td><td> $0 . 1 0 8 \pm 0 . 0 3 6$ </td><td> $0 . 1 2 2 \pm 0 . 0 3 3$ </td><td> $0 . 0 8 8 \pm 0 . 0 2 3$ </td><td> $\mathbf { 0 . 0 0 9 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 3$ </td></tr><tr><td>60%</td><td> $0 . 0 5 0 \pm 0 . 0 1 3$ </td><td> $0 . 1 7 0 \pm 0 . 0 6 1$ </td><td> $0 . 1 9 1 \pm 0 . 0 4 0$ </td><td> $0 . 1 2 9 \pm 0 . 0 2 7$ </td><td> $\mathbf { 0 . 0 1 0 \pm 0 . 0 0 2 }$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 3$ </td></tr><tr><td>70%</td><td> $0 . 0 7 8 \pm 0 . 0 1 5$ </td><td> $0 . 2 7 8 \pm 0 . 0 9 2$ </td><td> $0 . 3 0 0 \pm 0 . 0 6 2$ </td><td> $0 . 1 9 6 \pm 0 . 0 4 2$ </td><td> $\mathbf { 0 . 0 1 7 \pm 0 . 0 0 4 }$ </td><td> $0 . 0 2 3 \pm 0 . 0 1 0$ </td></tr><tr><td>80%</td><td> $0 . 1 0 1 \pm 0 . 0 2 5$ </td><td> $0 . 3 1 9 \pm 0 . 0 8 5$ </td><td> $0 . 3 8 3 \pm 0 . 0 4 8$ </td><td> $0 . 2 5 7 \pm 0 . 0 3 8$ </td><td> $\mathbf { 0 . 0 1 8 \pm 0 . 0 0 4 }$ </td><td> $0 . 0 3 0 \pm 0 . 0 0 5$ </td></tr><tr><td>90%</td><td> $0 . 1 6 0 \pm 0 . 0 2 9$ </td><td> $0 . 4 2 6 \pm 0 . 1 1 6$ </td><td> $0 . 4 8 9 \pm 0 . 0 4 3$ </td><td> $0 . 3 3 9 \pm 0 . 0 3 8$ </td><td> $\mathbf { 0 . 0 2 5 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 4 9 \pm 0 . 0 1 7$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 0 0 1 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 1 \pm 0 . 0 0 0$ </td><td></td></tr><tr><td>20%</td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 4 \pm 0 . 0 1 0$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 3$ </td><td></td><td></td></tr><tr><td>30%</td><td> $0 . 0 0 7 \pm 0 . 0 0 3$ </td><td> $0 . 0 3 8 \pm 0 . 0 2 5$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 2$ </td><td></td></tr><tr><td>40%</td><td></td><td> $0 . 0 4 9 \pm 0 . 0 1 9$ </td><td> $0 . 0 1 5 \pm 0 . 0 0 4$ </td><td> $0 . 0 1 9 \pm 0 . 0 1 4$ </td><td> $0 . 0 1 0 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>50%</td><td> $0 . 0 1 1 \pm 0 . 0 0 4$ </td><td></td><td> $0 . 0 2 7 \pm 0 . 0 0 6$ </td><td> $0 . 0 3 5 \pm 0 . 0 0 7$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 9$ </td><td></td></tr><tr><td>60%</td><td> $0 . 0 2 0 \pm 0 . 0 0 6$ </td><td> $0 . 0 9 3 \pm 0 . 0 3 5$ </td><td> $0 . 0 4 6 \pm 0 . 0 1 7$ </td><td></td><td> $0 . 0 3 0 \pm 0 . 0 1 0$ </td><td></td></tr><tr><td></td><td> $0 . 0 3 0 \pm 0 . 0 1 1$ </td><td> $0 . 1 1 5 \pm 0 . 0 3 4$ </td><td></td><td> $0 . 0 4 9 \pm 0 . 0 2 5$ </td><td> $0 . 0 5 1 \pm 0 . 0 2 1$ </td><td></td></tr><tr><td>70%</td><td> $0 . 0 4 1 \pm 0 . 0 0 8$ </td><td> $0 . 1 3 7 \pm 0 . 0 3 3$ </td><td> $0 . 0 6 9 \pm 0 . 0 2 1$   $0 . 1 0 0 \pm 0 . 0 2 4$ </td><td> $0 . 0 8 7 \pm 0 . 0 4 7$ </td><td> $0 . 0 8 9 \pm 0 . 0 3 3$ </td><td></td></tr><tr><td>80%</td><td> $0 . 0 6 3 \pm 0 . 0 1 8$ </td><td> $0 . 1 9 5 \pm 0 . 0 3 8$ </td><td></td><td> $0 . 2 1 1 \pm 0 . 0 5 0$ </td><td> $0 . 1 8 6 \pm 0 . 0 5 4$ </td><td></td></tr><tr><td>90%</td><td> $0 . 0 8 6 \pm 0 . 0 2 3$ </td><td> $0 . 2 3 1 \pm 0 . 0 5 7$ </td><td> $0 . 1 5 0 \pm 0 . 0 4 3$ </td><td> $0 . 3 2 2 \pm 0 . 0 8 9$ </td><td> $0 . 2 9 7 \pm 0 . 0 8 3$ </td><td></td></tr></table>

Table 24: PLS Procrustes distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td> $_ \mathrm { H a l f - m i n }$ </td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 6 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 0 0 1 \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td></tr><tr><td>20%</td><td> $0 . 0 0 6 \pm 0 . 0 0 2$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 4$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 4$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 0 0 2 \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 1$ </td></tr><tr><td>30%</td><td> $0 . 0 1 5 \pm 0 . 0 0 5$ </td><td> $0 . 0 4 1 \pm 0 . 0 2 1$ </td><td> $0 . 0 4 0 \pm 0 . 0 1 6$ </td><td> $0 . 0 3 1 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 0 0 5 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 4$ </td></tr><tr><td>40%</td><td> $0 . 0 2 4 \pm 0 . 0 0 8$ </td><td> $0 . 0 6 1 \pm 0 . 0 2 9$ </td><td> $0 . 0 6 7 \pm 0 . 0 2 5$ </td><td> $0 . 0 5 1 \pm 0 . 0 1 9$ </td><td> $\mathbf { 0 . 0 0 7 \pm 0 . 0 0 2 }$ </td><td> $0 . 0 0 9 \pm 0 . 0 0 2$ </td></tr><tr><td>50%</td><td> $0 . 0 3 9 \pm 0 . 0 0 6$ </td><td> $0 . 0 9 0 \pm 0 . 0 2 8$ </td><td> $0 . 1 0 7 \pm 0 . 0 2 0$ </td><td> $0 . 0 8 0 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 0 1 0 \pm 0 . 0 0 1 }$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 3$ </td></tr><tr><td>60%</td><td> $0 . 0 5 7 \pm 0 . 0 1 5$ </td><td> $0 . 1 2 8 \pm 0 . 0 5 2$ </td><td> $0 . 1 5 5 \pm 0 . 0 4 0$ </td><td> $0 . 1 1 4 \pm 0 . 0 2 9$ </td><td> $\mathbf { 0 . 0 1 } 2 \pm \mathbf { 0 . 0 0 } 3$ </td><td> $0 . 0 2 4 \pm 0 . 0 0 5$ </td></tr><tr><td>70%</td><td> $0 . 0 8 7 \pm 0 . 0 1 5$ </td><td> $0 . 2 0 9 \pm 0 . 0 7 0$ </td><td> $0 . 2 4 8 \pm 0 . 0 4 1$ </td><td> $0 . 1 7 7 \pm 0 . 0 3 3$ </td><td> $\mathbf { 0 . 0 2 0 \pm 0 . 0 0 4 }$ </td><td> $0 . 0 3 6 \pm 0 . 0 1 0$ </td></tr><tr><td>80%</td><td> $0 . 1 0 7 \pm 0 . 0 2 0$ </td><td> $0 . 2 5 5 \pm 0 . 0 5 9$ </td><td> $0 . 2 9 6 \pm 0 . 0 3 7$ </td><td> $0 . 2 1 3 \pm 0 . 0 2 9$ </td><td> $\mathbf { 0 . 0 2 1 \pm 0 . 0 0 4 }$ </td><td> $0 . 0 5 0 \pm 0 . 0 1 8$ </td></tr><tr><td>90%</td><td> $0 . 1 5 6 \pm 0 . 0 2 0$ </td><td> $0 . 3 7 0 \pm 0 . 1 0 4$ </td><td> $0 . 3 8 4 \pm 0 . 0 3 3$ </td><td> $0 . 2 7 8 \pm 0 . 0 2 7$ </td><td> $\mathbf { 0 . 0 2 8 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 8 6 \pm 0 . 0 3 7$ </td></tr><tr><td>Rate</td><td>NF_OGpaper</td><td> $\mathrm { S o f t I m p u t e }$ </td><td> $\mathrm { S o f t F o r e s t }$ </td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 1$ </td><td></td></tr><tr><td>20%</td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 1 1 \pm 0 . 0 0 3$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td> $0 . 0 0 5 \pm 0 . 0 0 1$ </td><td></td></tr><tr><td>30%</td><td> $0 . 0 1 1 \pm 0 . 0 0 4$ </td><td> $0 . 0 3 8 \pm 0 . 0 1 4$ </td><td> $0 . 0 1 3 \pm 0 . 0 0 5$ </td><td> $0 . 0 1 2 \pm 0 . 0 0 6$ </td><td> $0 . 0 1 4 \pm 0 . 0 0 6$ </td><td></td></tr><tr><td>40%</td><td> $0 . 0 1 7 \pm 0 . 0 0 6$ </td><td> $0 . 0 5 9 \pm 0 . 0 1 8$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 6$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 8$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 8$ </td><td></td></tr><tr><td>50%</td><td> $0 . 0 2 4 \pm 0 . 0 0 4$ </td><td> $0 . 0 9 3 \pm 0 . 0 2 7$ </td><td> $0 . 0 3 0 \pm 0 . 0 0 4$ </td><td> $0 . 0 2 8 \pm 0 . 0 0 8$ </td><td> $0 . 0 3 3 \pm 0 . 0 0 7$ </td><td></td></tr><tr><td>60%</td><td> $0 . 0 4 0 \pm 0 . 0 1 2$ </td><td> $0 . 1 3 2 \pm 0 . 0 5 6$ </td><td> $0 . 0 5 4 \pm 0 . 0 2 3$ </td><td> $0 . 0 4 8 \pm 0 . 0 2 5$ </td><td> $0 . 0 6 0 \pm 0 . 0 2 7$ </td><td></td></tr><tr><td>70%</td><td> $0 . 0 4 9 \pm 0 . 0 1 0$ </td><td> $0 . 1 7 6 \pm 0 . 0 4 7$ </td><td> $0 . 0 7 7 \pm 0 . 0 1 7$ </td><td> $0 . 0 8 3 \pm 0 . 0 4 6$ </td><td> $0 . 0 9 2 \pm 0 . 0 3 4$ </td><td></td></tr><tr><td>80%</td><td> $0 . 0 7 3 \pm 0 . 0 1 6$ </td><td> $0 . 2 5 7 \pm 0 . 0 4 6$ </td><td> $0 . 1 0 8 \pm 0 . 0 2 5$ </td><td> $0 . 1 3 3 \pm 0 . 0 4 4$ </td><td> $0 . 1 4 3 \pm 0 . 0 4 2$ </td><td></td></tr><tr><td>90%</td><td> $0 . 0 9 3 \pm 0 . 0 1 5$ </td><td> $0 . 3 0 9 \pm 0 . 0 5 7$ </td><td> $0 . 1 5 9 \pm 0 . 0 4 6$ </td><td> $0 . 2 7 0 \pm 0 . 0 7 4$ </td><td> $0 . 2 5 8 \pm 0 . 0 6 5$ </td><td></td></tr></table>

Table 25: Pearson log-p correlation across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td> $_ \mathrm { H a l f - m i n }$ </td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 9 9 1 \pm 0 . 0 0 7$ </td><td> $0 . 9 6 2 \pm 0 . 0 2 4$ </td><td> $0 . 9 4 2 \pm 0 . 0 3 6$ </td><td> $0 . 9 5 6 \pm 0 . 0 2 7$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 9 9 7 \pm 0 . 0 0 1 }$ </td></tr><tr><td>20%</td><td> $0 . 9 8 3 \pm 0 . 0 0 6$ </td><td> $0 . 9 4 8 \pm 0 . 0 2 2$ </td><td> $0 . 9 2 4 \pm 0 . 0 3 0$ </td><td> $0 . 9 4 6 \pm 0 . 0 2 4$ </td><td> $\mathbf { 0 . 9 9 4 \pm 0 . 0 0 2 }$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 4$ </td></tr><tr><td>30%</td><td> $0 . 9 6 4 \pm 0 . 0 1 5$ </td><td> $0 . 8 9 0 \pm 0 . 0 4 8$ </td><td> $0 . 8 4 9 \pm 0 . 0 4 2$ </td><td> $0 . 8 8 8 \pm 0 . 0 3 2$ </td><td> $\mathbf { 0 . 9 8 9 \pm 0 . 0 0 4 }$ </td><td> $0 . 9 8 8 \pm 0 . 0 0 6$ </td></tr><tr><td>40%</td><td> $0 . 9 5 0 \pm 0 . 0 1 2$ </td><td> $0 . 8 4 2 \pm 0 . 0 4 6$ </td><td> $0 . 7 8 7 \pm 0 . 0 3 7$ </td><td> $0 . 8 3 8 \pm 0 . 0 3 1$ </td><td> $\mathbf { 0 . 9 8 7 \pm 0 . 0 0 3 }$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 3$ </td></tr><tr><td>50%</td><td> $0 . 9 4 8 \pm 0 . 0 1 3$ </td><td> $0 . 8 4 7 \pm 0 . 0 3 5$ </td><td> $0 . 7 8 8 \pm 0 . 0 2 8$ </td><td> $0 . 8 4 2 \pm 0 . 0 2 3$ </td><td> $\mathbf { 0 . 9 8 6 \pm 0 . 0 0 4 }$ </td><td> $0 . 9 7 5 \pm 0 . 0 0 6$ </td></tr><tr><td>60%</td><td> $0 . 9 2 1 \pm 0 . 0 1 6$ </td><td> $0 . 7 9 8 \pm 0 . 0 7 9$ </td><td> $0 . 7 3 4 \pm 0 . 0 6 1$ </td><td> $0 . 8 0 3 \pm 0 . 0 4 2$ </td><td> $\mathbf { 0 . 9 8 2 \pm 0 . 0 0 5 }$ </td><td> $0 . 9 6 7 \pm 0 . 0 0 9$ </td></tr><tr><td>70%</td><td> $0 . 9 2 1 \pm 0 . 0 1 3$ </td><td> $0 . 7 7 4 \pm 0 . 0 6 5$ </td><td> $0 . 7 0 4 \pm 0 . 0 4 6$ </td><td> $0 . 7 9 3 \pm 0 . 0 3 1$ </td><td> $\mathbf { 0 . 9 7 6 \pm 0 . 0 0 4 }$ </td><td> $0 . 9 5 9 \pm 0 . 0 0 7$ </td></tr><tr><td>80%</td><td> $0 . 9 0 8 \pm 0 . 0 1 9$ </td><td> $0 . 7 0 8 \pm 0 . 0 7 0$ </td><td> $0 . 6 5 3 \pm 0 . 0 4 2$ </td><td> $0 . 7 5 2 \pm 0 . 0 3 5$ </td><td> $\mathbf { 0 . 9 7 4 \pm 0 . 0 0 4 }$ </td><td> $0 . 9 4 2 \pm 0 . 0 0 9$ </td></tr><tr><td>90%</td><td> $0 . 9 0 5 \pm 0 . 0 1 7$ </td><td> $0 . 6 9 8 \pm 0 . 1 2 0$ </td><td> $0 . 6 8 3 \pm 0 . 0 3 8$ </td><td> $0 . 7 7 7 \pm 0 . 0 2 7$ </td><td> $\mathbf { 0 . 9 7 2 \pm 0 . 0 0 5 }$ </td><td> $0 . 9 2 4 \pm 0 . 0 2 9$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 9 9 2 \pm 0 . 0 0 6$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 8$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 6$ </td><td> $0 . 9 9 6 \pm 0 . 0 0 2$ </td><td> $0 . 9 9 2 \pm 0 . 0 0 6$ </td><td></td></tr><tr><td>20%</td><td> $0 . 9 8 8 \pm 0 . 0 0 4$ </td><td> $0 . 9 8 1 \pm 0 . 0 0 7$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 5$ </td><td> $0 . 9 8 5 \pm 0 . 0 1 2$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 5$ </td><td></td></tr><tr><td>30%</td><td> $0 . 9 7 6 \pm 0 . 0 1 4$ </td><td> $0 . 9 6 5 \pm 0 . 0 1 9$ </td><td> $0 . 9 7 3 \pm 0 . 0 1 4$ </td><td> $0 . 9 6 7 \pm 0 . 0 2 8$ </td><td> $0 . 9 6 6 \pm 0 . 0 2 3$ </td><td></td></tr><tr><td>40%</td><td> $0 . 9 6 8 \pm 0 . 0 0 9$ </td><td> $0 . 9 4 9 \pm 0 . 0 1 9$ </td><td> $0 . 9 6 2 \pm 0 . 0 0 8$ </td><td> $0 . 9 3 7 \pm 0 . 0 3 8$ </td><td> $0 . 9 4 7 \pm 0 . 0 1 8$ </td><td></td></tr><tr><td>50%</td><td> $0 . 9 6 8 \pm 0 . 0 0 7$ </td><td> $0 . 9 4 2 \pm 0 . 0 1 7$ </td><td> $0 . 9 6 4 \pm 0 . 0 0 6$ </td><td> $0 . 9 3 3 \pm 0 . 0 2 5$ </td><td> $0 . 9 5 1 \pm 0 . 0 1 5$ </td><td></td></tr><tr><td>60%</td><td> $0 . 9 4 3 \pm 0 . 0 1 4$ </td><td> $0 . 9 3 1 \pm 0 . 0 1 7$ </td><td> $0 . 9 3 7 \pm 0 . 0 2 0$ </td><td> $0 . 9 0 2 \pm 0 . 0 5 7$ </td><td></td><td></td></tr><tr><td>70%</td><td> $0 . 9 5 0 \pm 0 . 0 1 1$ </td><td> $0 . 9 1 1 \pm 0 . 0 1 9$ </td><td> $0 . 9 4 1 \pm 0 . 0 1 2$ </td><td> $0 . 8 8 2 \pm 0 . 0 5 4$ </td><td> $0 . 9 1 1 \pm 0 . 0 4 2$ </td><td></td></tr><tr><td>80%</td><td> $0 . 9 3 5 \pm 0 . 0 1 5$ </td><td> $0 . 9 0 7 \pm 0 . 0 2 2$ </td><td> $0 . 9 3 2 \pm 0 . 0 1 4$ </td><td> $0 . 8 2 3 \pm 0 . 0 5 0$ </td><td> $0 . 9 0 3 \pm 0 . 0 3 1$   $0 . 8 6 1 \pm 0 . 0 3 5$ </td><td></td></tr><tr><td>90%</td><td> $0 . 9 4 0 \pm 0 . 0 0 9$ </td><td> $0 . 8 7 4 \pm 0 . 0 1 9$ </td><td> $0 . 9 3 6 \pm 0 . 0 0 9$ </td><td> $0 . 7 8 6 \pm 0 . 0 7 0$ </td><td> $0 . 8 3 7 \pm 0 . 0 4 9$ </td><td></td></tr></table>

## E.3 MCAR RESULTS ON THE HOUSING DATASET

Table 26: NRMSE across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td>SVT_OGpaper</td></tr><tr><td>10%</td><td> $0 . 8 4 2 \pm 0 . 0 3 2$ </td><td> $1 . 0 4 9 \pm 0 . 0 2 9$ </td><td> $1 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $1 . 1 4 7 \pm 0 . 0 1 2$ </td><td> $1 . 8 2 9 \pm 0 . 0 5 7$ </td><td> $1 . 4 5 1 \pm 0 . 0 3 7$ </td></tr><tr><td>20%</td><td> $0 . 8 9 2 \pm 0 . 0 2 0$ </td><td> $1 . 0 7 2 \pm 0 . 0 2 6$ </td><td> $1 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $1 . 1 4 1 \pm 0 . 0 1 1$ </td><td> $1 . 8 1 4 \pm 0 . 0 3 2$ </td><td> $1 . 4 6 0 \pm 0 . 0 3 8$ </td></tr><tr><td>30%</td><td> $0 . 9 5 0 \pm 0 . 0 1 5$ </td><td> $1 . 0 6 0 \pm 0 . 0 2 4$ </td><td> $1 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $1 . 1 3 4 \pm 0 . 0 0 5$ </td><td> $1 . 8 1 8 \pm 0 . 0 2 0$ </td><td> $1 . 5 2 0 \pm 0 . 0 2 1$ </td></tr><tr><td>40%</td><td> $1 . 0 2 0 \pm 0 . 0 2 3$ </td><td> $1 . 0 1 1 \pm 0 . 0 1 0$ </td><td> $1 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $1 . 1 3 8 \pm 0 . 0 0 5$ </td><td> $1 . 8 0 7 \pm 0 . 0 5 6$ </td><td> $1 . 4 0 1 \pm 0 . 0 1 6$ </td></tr><tr><td>50%</td><td> $1 . 0 7 2 \pm 0 . 0 2 1$ </td><td> $1 . 0 0 6 \pm 0 . 0 0 6$ </td><td> $1 . 0 0 2 \pm 0 . 0 0 1$ </td><td> $1 . 1 3 7 \pm 0 . 0 0 6$ </td><td> $1 . 7 9 9 \pm 0 . 0 3 9$ </td><td> $1 . 6 8 9 \pm 0 . 0 3 6$ </td></tr><tr><td>60%</td><td> $1 . 1 2 4 \pm 0 . 0 3 1$ </td><td> $1 . 0 1 8 \pm 0 . 0 0 5$ </td><td> $1 . 0 0 3 \pm 0 . 0 0 1$ </td><td> $1 . 1 4 0 \pm 0 . 0 0 4$ </td><td> $1 . 7 5 9 \pm 0 . 0 4 7$ </td><td> $1 . 6 1 0 \pm 0 . 0 3 1$ </td></tr><tr><td>70%</td><td> $1 . 1 4 7 \pm 0 . 0 2 4$ </td><td> $1 . 0 2 7 \pm 0 . 0 0 6$ </td><td> $\mathbf { 1 . 0 0 4 \pm 0 . 0 0 1 }$ </td><td> $1 . 1 4 0 \pm 0 . 0 0 4$ </td><td> $1 . 7 8 9 \pm 0 . 0 4 5$ </td><td> $1 . 6 5 0 \pm 0 . 0 2 5$ </td></tr><tr><td>80%</td><td> $1 . 1 7 6 \pm 0 . 0 2 6$ </td><td> $1 . 0 2 9 \pm 0 . 0 0 5$ </td><td> $1 . 0 0 5 \pm 0 . 0 0 2$ </td><td> $1 . 1 3 6 \pm 0 . 0 0 6$ </td><td> $1 . 7 6 6 \pm 0 . 0 4 0$ </td><td> $1 . 7 3 8 \pm 0 . 0 2 0$ </td></tr><tr><td>90%</td><td> $1 . 1 5 9 \pm 0 . 0 2 2$ </td><td> $1 . 0 4 5 \pm 0 . 0 1 0$ </td><td> $\mathbf { 1 . 0 0 7 \pm 0 . 0 0 3 }$ </td><td> $1 . 1 3 5 \pm 0 . 0 0 8$ </td><td> $1 . 7 3 0 \pm 0 . 0 2 8$ </td><td> $1 . 9 0 9 \pm 0 . 0 3 0$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 8 3 2 \pm 0 . 0 2 0$ </td><td> $1 . 3 8 6 \pm 0 . 0 3 3$ </td><td> $0 . 8 3 5 \pm 0 . 0 2 0$ </td><td> $1 . 0 1 9 \pm 0 . 0 1 7$ </td><td> $\mathbf { 0 . 8 2 0 \pm 0 . 0 1 7 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 8 6 3 \pm 0 . 0 1 4$ </td><td> $1 . 4 4 5 \pm 0 . 0 4 7$ </td><td> $0 . 8 6 6 \pm 0 . 0 1 5$ </td><td> $1 . 0 1 3 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 8 5 2 \pm 0 . 0 1 2 }$ </td><td></td></tr><tr><td>30%</td><td> $0 . 9 1 0 \pm 0 . 0 1 2$ </td><td> $1 . 5 3 3 \pm 0 . 0 3 0$ </td><td> $0 . 9 1 4 \pm 0 . 0 1 8$ </td><td> $1 . 0 1 3 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 8 8 5 \pm 0 . 0 1 1 }$ </td><td></td></tr><tr><td>40%</td><td> $0 . 9 7 6 \pm 0 . 0 2 3$ </td><td> $1 . 6 4 4 \pm 0 . 0 3 6$ </td><td> $0 . 9 5 4 \pm 0 . 0 1 4$ </td><td> $1 . 0 1 1 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 9 0 9 \pm 0 . 0 0 7 }$ </td><td></td></tr><tr><td>50%</td><td> $1 . 0 5 6 \pm 0 . 0 1 6$ </td><td> $1 . 7 2 0 \pm 0 . 0 3 3$ </td><td> $1 . 0 1 5 \pm 0 . 0 1 5$ </td><td> $1 . 0 0 7 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 9 4 1 \pm 0 . 0 0 7 }$ </td><td></td></tr><tr><td>60%</td><td> $1 . 1 1 5 \pm 0 . 0 1 9$ </td><td> $1 . 8 1 3 \pm 0 . 0 3 2$ </td><td> $1 . 0 7 1 \pm 0 . 0 1 4$ </td><td> $1 . 0 0 7 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 9 8 0 \pm 0 . 0 1 0 }$ </td><td></td></tr><tr><td>70%</td><td> $1 . 1 8 8 \pm 0 . 0 2 4$ </td><td> $1 . 8 9 4 \pm 0 . 0 2 7$ </td><td> $1 . 1 5 2 \pm 0 . 0 2 6$ </td><td> $1 . 0 0 5 \pm 0 . 0 0 5$ </td><td> $1 . 0 1 3 \pm 0 . 0 1 3$ </td><td></td></tr><tr><td>80%</td><td> $1 . 2 3 5 \pm 0 . 0 2 5$ </td><td> $1 . 9 5 5 \pm 0 . 0 1 7$ </td><td> $1 . 2 3 2 \pm 0 . 0 3 4$ </td><td> $\mathbf { 1 . 0 0 4 \pm 0 . 0 0 5 }$ </td><td> $1 . 0 6 7 \pm 0 . 0 1 3$ </td><td></td></tr><tr><td>90%</td><td> $1 . 2 8 3 \pm 0 . 0 1 5$ </td><td> $2 . 0 0 7 \pm 0 . 0 1 0$ </td><td> $1 . 3 1 9 \pm 0 . 0 1 8$ </td><td> $1 . 0 0 8 \pm 0 . 0 0 2$ </td><td> $1 . 1 1 6 \pm 0 . 0 2 5$ </td><td></td></tr></table>

Table 27: PFC across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td> $_ \mathrm { H a l f - m i n }$ </td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 1 4 5 \pm 0 . 0 1 7$ </td><td> $0 . 2 7 1 \pm 0 . 0 2 7$ </td><td> $0 . 2 5 3 \pm 0 . 0 1 4$ </td><td> $0 . 2 5 3 \pm 0 . 0 1 4$ </td><td> $0 . 3 3 8 \pm 0 . 0 2 1$ </td><td> $0 . 2 7 1 \pm 0 . 0 1 6$ </td></tr><tr><td>20%</td><td> $0 . 1 7 3 \pm 0 . 0 0 7$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 4$ </td><td> $0 . 2 5 3 \pm 0 . 0 1 2$ </td><td> $0 . 2 5 3 \pm 0 . 0 1 2$ </td><td> $0 . 3 3 3 \pm 0 . 0 1 0$ </td><td> $0 . 2 7 4 \pm 0 . 0 1 0$ </td></tr><tr><td>30%</td><td> $0 . 1 9 4 \pm 0 . 0 1 0$ </td><td> $0 . 2 4 3 \pm 0 . 0 1 8$ </td><td> $0 . 2 4 7 \pm 0 . 0 0 7$ </td><td> $0 . 2 4 7 \pm 0 . 0 0 7$ </td><td> $0 . 3 3 1 \pm 0 . 0 1 0$ </td><td> $0 . 2 8 1 \pm 0 . 0 0 6$ </td></tr><tr><td>40%</td><td> $0 . 2 2 7 \pm 0 . 0 1 0$ </td><td> $0 . 2 3 6 \pm 0 . 0 0 8$ </td><td> $0 . 2 4 7 \pm 0 . 0 0 6$ </td><td> $0 . 2 4 7 \pm 0 . 0 0 6$ </td><td> $0 . 3 3 2 \pm 0 . 0 0 9$ </td><td> $0 . 2 4 5 \pm 0 . 0 1 8$ </td></tr><tr><td>50%</td><td> $0 . 2 5 2 \pm 0 . 0 1 4$ </td><td> $0 . 2 4 3 \pm 0 . 0 0 5$ </td><td> $0 . 2 5 0 \pm 0 . 0 0 6$ </td><td> $0 . 2 5 0 \pm 0 . 0 0 6$ </td><td> $0 . 3 3 3 \pm 0 . 0 0 9$ </td><td> $0 . 2 9 5 \pm 0 . 0 1 3$ </td></tr><tr><td>60%</td><td> $0 . 2 7 0 \pm 0 . 0 1 7$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 5$ </td><td> $0 . 2 5 3 \pm 0 . 0 0 4$ </td><td> $0 . 2 5 3 \pm 0 . 0 0 4$ </td><td> $0 . 3 3 3 \pm 0 . 0 0 6$ </td><td> $0 . 2 8 9 \pm 0 . 0 0 7$ </td></tr><tr><td>70%</td><td> $0 . 2 8 4 \pm 0 . 0 1 7$ </td><td> $0 . 2 5 9 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 2 5 1 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 2 5 1 \pm 0 . 0 0 4 }$ </td><td> $0 . 3 3 1 \pm 0 . 0 0 4$ </td><td> $0 . 2 9 6 \pm 0 . 0 0 6$ </td></tr><tr><td>80%</td><td> $0 . 2 9 3 \pm 0 . 0 1 2$ </td><td> $0 . 2 6 3 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 2 5 2 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 2 5 2 \pm 0 . 0 0 2 }$ </td><td> $0 . 3 3 2 \pm 0 . 0 0 3$ </td><td> $0 . 3 0 8 \pm 0 . 0 0 4$ </td></tr><tr><td>90%</td><td> $0 . 2 8 0 \pm 0 . 0 0 8$ </td><td> $0 . 2 6 6 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 2 5 2 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 2 5 2 \pm 0 . 0 0 1 }$ </td><td> $0 . 3 3 1 \pm 0 . 0 0 2$ </td><td> $0 . 3 2 2 \pm 0 . 0 0 7$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 1 4 6 \pm 0 . 0 1 8$ </td><td> $0 . 2 7 4 \pm 0 . 0 1 4$ </td><td> $0 . 1 4 3 \pm 0 . 0 1 7$ </td><td> $0 . 2 6 2 \pm 0 . 0 1 3$ </td><td> $\mathbf { 0 . 1 4 1 \pm 0 . 0 1 7 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 1 6 4 \pm 0 . 0 1 2$ </td><td> $0 . 2 7 6 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 1 6 0 \pm 0 . 0 0 8 }$ </td><td> $0 . 2 6 0 \pm 0 . 0 1 1$ </td><td> $0 . 1 6 1 \pm 0 . 0 1 1$ </td><td></td></tr><tr><td>30%</td><td> $0 . 1 8 3 \pm 0 . 0 0 7$ </td><td> $0 . 2 8 1 \pm 0 . 0 0 7$ </td><td> $0 . 1 8 0 \pm 0 . 0 0 9$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 1 7 6 \pm 0 . 0 0 9 }$ </td><td></td></tr><tr><td>40%</td><td> $0 . 2 0 5 \pm 0 . 0 0 8$ </td><td> $0 . 2 9 2 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 1 9 2 \pm 0 . 0 0 8 }$ </td><td> $0 . 2 5 2 \pm 0 . 0 0 4$ </td><td> $0 . 1 9 4 \pm 0 . 0 0 7$ </td><td></td></tr><tr><td>50%</td><td> $0 . 2 4 4 \pm 0 . 0 0 6$ </td><td> $0 . 3 0 2 \pm 0 . 0 0 7$ </td><td> $0 . 2 2 2 \pm 0 . 0 0 5$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 2 1 7 \pm 0 . 0 0 6 }$ </td><td></td></tr><tr><td>60%</td><td> $0 . 2 6 1 \pm 0 . 0 0 9$ </td><td> $0 . 3 1 3 \pm 0 . 0 0 6$ </td><td> $0 . 2 4 2 \pm 0 . 0 0 8$ </td><td> $0 . 2 5 7 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 2 4 1 \pm 0 . 0 0 6 }$ </td><td></td></tr><tr><td>70%</td><td> $0 . 2 8 4 \pm 0 . 0 0 9$ </td><td> $0 . 3 1 7 \pm 0 . 0 0 4$ </td><td> $0 . 2 7 3 \pm 0 . 0 0 6$ </td><td> $0 . 2 5 5 \pm 0 . 0 0 5$ </td><td> $0 . 2 6 3 \pm 0 . 0 0 6$ </td><td></td></tr><tr><td>80%</td><td> $0 . 2 9 8 \pm 0 . 0 0 8$ </td><td> $0 . 3 2 6 \pm 0 . 0 0 3$ </td><td> $0 . 2 9 7 \pm 0 . 0 1 2$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 3$ </td><td> $0 . 2 9 6 \pm 0 . 0 1 0$ </td><td></td></tr><tr><td>90%</td><td> $0 . 3 1 1 \pm 0 . 0 0 8$ </td><td> $0 . 3 2 6 \pm 0 . 0 0 2$ </td><td> $0 . 3 3 2 \pm 0 . 0 0 6$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 1$ </td><td> $0 . 3 1 8 \pm 0 . 0 2 8$ </td><td></td></tr></table>

Table 28: Gower’s distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td> $_ \mathrm { H a l f - m i n }$ </td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 1 3 0 \pm 0 . 0 0 8$ </td><td> $0 . 2 2 8 \pm 0 . 0 1 9$ </td><td> $0 . 2 1 5 \pm 0 . 0 0 9$ </td><td> $0 . 2 1 2 \pm 0 . 0 0 9$ </td><td> $0 . 3 4 2 \pm 0 . 0 1 8$ </td><td> $0 . 2 6 5 \pm 0 . 0 0 9$ </td></tr><tr><td>20%</td><td> $0 . 1 4 7 \pm 0 . 0 0 5$ </td><td> $0 . 2 1 6 \pm 0 . 0 0 8$ </td><td> $0 . 2 1 5 \pm 0 . 0 0 8$ </td><td> $0 . 2 0 9 \pm 0 . 0 0 9$ </td><td> $0 . 3 3 7 \pm 0 . 0 0 8$ </td><td> $0 . 2 6 9 \pm 0 . 0 1 2$ </td></tr><tr><td>30%</td><td> $0 . 1 6 1 \pm 0 . 0 0 6$ </td><td> $0 . 2 0 8 \pm 0 . 0 1 2$ </td><td> $0 . 2 1 2 \pm 0 . 0 0 6$ </td><td> $0 . 2 0 5 \pm 0 . 0 0 6$ </td><td> $0 . 3 3 3 \pm 0 . 0 0 8$ </td><td> $0 . 2 8 0 \pm 0 . 0 0 7$ </td></tr><tr><td>40%</td><td> $0 . 1 8 5 \pm 0 . 0 0 7$ </td><td> $0 . 2 0 2 \pm 0 . 0 0 5$ </td><td> $0 . 2 1 1 \pm 0 . 0 0 4$ </td><td> $0 . 2 0 5 \pm 0 . 0 0 5$ </td><td> $0 . 3 3 5 \pm 0 . 0 0 8$ </td><td> $0 . 2 4 6 \pm 0 . 0 1 2$ </td></tr><tr><td>50%</td><td> $0 . 2 0 4 \pm 0 . 0 0 8$ </td><td> $0 . 2 0 7 \pm 0 . 0 0 3$ </td><td> $0 . 2 1 4 \pm 0 . 0 0 5$ </td><td> $0 . 2 0 8 \pm 0 . 0 0 5$ </td><td> $0 . 3 3 5 \pm 0 . 0 0 8$ </td><td> $0 . 3 0 4 \pm 0 . 0 1 2$ </td></tr><tr><td>60%</td><td> $0 . 2 2 1 \pm 0 . 0 1 0$ </td><td> $0 . 2 1 6 \pm 0 . 0 0 4$ </td><td> $0 . 2 1 7 \pm 0 . 0 0 3$ </td><td> $0 . 2 1 1 \pm 0 . 0 0 3$ </td><td> $0 . 3 3 3 \pm 0 . 0 0 6$ </td><td> $0 . 2 9 7 \pm 0 . 0 0 6$ </td></tr><tr><td>70%</td><td> $0 . 2 3 2 \pm 0 . 0 0 9$ </td><td> $0 . 2 2 0 \pm 0 . 0 0 4$ </td><td> $0 . 2 1 5 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 2 0 9 \pm 0 . 0 0 2 }$ </td><td> $0 . 3 3 4 \pm 0 . 0 0 4$ </td><td> $0 . 3 0 5 \pm 0 . 0 0 5$ </td></tr><tr><td>80%</td><td> $0 . 2 4 1 \pm 0 . 0 0 8$ </td><td> $0 . 2 2 1 \pm 0 . 0 0 3$ </td><td> $0 . 2 1 5 \pm 0 . 0 0 2$ </td><td> $\mathbf { 0 . 2 0 9 \pm 0 . 0 0 2 }$ </td><td> $0 . 3 3 2 \pm 0 . 0 0 5$ </td><td> $0 . 3 2 1 \pm 0 . 0 0 3$ </td></tr><tr><td>90%</td><td> $0 . 2 3 2 \pm 0 . 0 0 7$ </td><td> $0 . 2 2 5 \pm 0 . 0 0 3$ </td><td> $0 . 2 1 5 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 2 1 0 \pm 0 . 0 0 2 }$ </td><td> $0 . 3 2 9 \pm 0 . 0 0 3$ </td><td> $0 . 3 4 8 \pm 0 . 0 0 5$ </td></tr><tr><td>Rate</td><td>NF_OGpaper</td><td> $\mathrm { S o f t I m p u t e }$ </td><td> $\mathrm { S o f t F o r e s t }$ </td><td> $\operatorname { s v r }$ </td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 1 3 2 \pm 0 . 0 1 0$ </td><td> $0 . 2 5 3 \pm 0 . 0 0 8$ </td><td> $0 . 1 3 0 \pm 0 . 0 0 9$ </td><td> $0 . 2 2 2 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 1 2 9 \pm 0 . 0 0 9 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 1 4 4 \pm 0 . 0 0 8$ </td><td> $0 . 2 6 0 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 1 4 } 2 \pm \mathbf { 0 . 0 0 } 7$ </td><td> $0 . 2 2 0 \pm 0 . 0 1 0$ </td><td> $0 . 1 4 2 \pm 0 . 0 0 7$ </td><td></td></tr><tr><td>30%</td><td> $0 . 1 5 8 \pm 0 . 0 0 5$ </td><td> $0 . 2 7 3 \pm 0 . 0 0 6$ </td><td> $0 . 1 5 7 \pm 0 . 0 0 6$ </td><td> $0 . 2 1 7 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 1 5 4 \pm 0 . 0 0 5 }$ </td><td></td></tr><tr><td>40%</td><td> $0 . 1 7 4 \pm 0 . 0 0 5$ </td><td> $0 . 2 9 1 \pm 0 . 0 0 7$ </td><td> $0 . 1 6 9 \pm 0 . 0 0 5$ </td><td> $0 . 2 1 5 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 1 6 6 \pm 0 . 0 0 3 }$ </td><td></td></tr><tr><td>50%</td><td> $0 . 2 0 5 \pm 0 . 0 0 4$ </td><td> $0 . 3 0 8 \pm 0 . 0 0 6$ </td><td> $0 . 1 9 3 \pm 0 . 0 0 4$ </td><td> $0 . 2 1 7 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 1 8 3 \pm 0 . 0 0 5 }$ </td><td></td></tr><tr><td>60%</td><td> $0 . 2 2 2 \pm 0 . 0 0 6$ </td><td> $0 . 3 2 8 \pm 0 . 0 0 5$ </td><td> $0 . 2 1 1 \pm 0 . 0 0 6$ </td><td> $0 . 2 2 0 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 2 0 2 \pm 0 . 0 0 5 }$ </td><td></td></tr><tr><td>70%</td><td> $0 . 2 4 1 \pm 0 . 0 0 6$ </td><td> $0 . 3 4 2 \pm 0 . 0 0 4$ </td><td> $0 . 2 3 6 \pm 0 . 0 0 4$ </td><td> $0 . 2 1 8 \pm 0 . 0 0 3$ </td><td> $0 . 2 1 8 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>80%</td><td> $0 . 2 5 0 \pm 0 . 0 0 4$ </td><td> $0 . 3 5 5 \pm 0 . 0 0 3$ </td><td> $0 . 2 5 3 \pm 0 . 0 0 8$ </td><td> $0 . 2 1 6 \pm 0 . 0 0 3$ </td><td> $0 . 2 3 9 \pm 0 . 0 0 6$ </td><td></td></tr><tr><td>90%</td><td> $0 . 2 5 9 \pm 0 . 0 0 4$ </td><td> $0 . 3 6 2 \pm 0 . 0 0 2$ </td><td> $0 . 2 7 7 \pm 0 . 0 0 4$ </td><td> $0 . 2 1 7 \pm 0 . 0 0 1$ </td><td> $0 . 2 5 6 \pm 0 . 0 1 7$ </td><td></td></tr></table>

Table 29: Downstream predictive $R ^ { 2 }$ degradation across missing rates

$$
\operatorname { S V T \_ O G p a p e r }
$$

$$
0 . 0 2 3 \pm 0 . 0 0 8
$$

$$
0 . 0 3 8 \pm 0 . 0 1 7
$$

$$
0 . 1 0 0 \pm 0 . 0 2 1
$$

$$
0 . 1 1 3 \pm 0 . 0 2 0
$$

$$
0 . 3 3 0 \pm 0 . 0 4 4
$$

$$
0 . 3 3 9 \pm 0 . 0 4 5
$$

$$
0 . 0 3 4 \pm 0 . 0 1 2
$$

$$
0 . 1 1 4 \pm 0 . 0 2 9
$$

$$
0 . 1 9 2 \pm 0 . 0 2 6
$$

$$
0 . 2 1 7 \pm 0 . 0 3 0
$$

$$
0 . 4 7 9 \pm 0 . 0 2 6
$$

$$
0 . 4 4 9 \pm 0 . 0 3 4
$$

$$
0 . 0 5 0 \pm 0 . 0 2 3
$$

$$
0 . 1 7 3 \pm 0 . 0 3 3
$$

$$
0 . 2 7 6 \pm 0 . 0 3 3
$$

$$
0 . 3 2 4 \pm 0 . 0 4 1
$$

$$
0 . 5 7 1 \pm 0 . 0 5 6
$$

$$
0 . 4 9 9 \pm 0 . 0 5 6
$$

$$
0 . 0 6 5 \pm 0 . 0 1 7
$$

$$
0 . 2 1 7 \pm 0 . 0 2 0
$$

$$
0 . 3 5 6 \pm 0 . 0 4 0
$$

$$
0 . 4 1 4 \pm 0 . 0 5 2
$$

$$
0 . 6 3 0 \pm 0 . 0 4 1
$$

$$
0 . 7 6 0 \pm 0 . 0 9 6
$$

$$
\mathbf { 0 . 0 5 9 \pm 0 . 0 3 2 }
$$

$$
0 . 3 2 5 \pm 0 . 0 4 8
$$

$$
0 . 4 4 8 \pm 0 . 0 3 5
$$

$$
0 . 5 5 9 \pm 0 . 0 3 8
$$

$$
0 . 6 7 2 \pm 0 . 0 3 4
$$

$$
1 . 1 8 0 \pm 0 . 1 1 8
$$

$$
\mathbf { 0 . 0 6 0 \pm 0 . 0 3 4 }
$$

$$
0 . 4 8 0 \pm 0 . 0 5 4
$$

$$
0 . 5 6 2 \pm 0 . 0 6 6
$$

$$
0 . 6 9 7 \pm 0 . 1 0 9
$$

$$
0 . 6 8 9 \pm 0 . 0 5 5
$$

$$
1 . 1 5 2 \pm 0 . 0 9 6
$$

$$
\mathbf { 0 . 0 6 9 \pm 0 . 0 5 8 }
$$

$$
0 . 5 2 2 \pm 0 . 0 8 4
$$

$$
0 . 6 5 0 \pm 0 . 1 0 2
$$

$$
0 . 8 3 1 \pm 0 . 1 4 8
$$

$$
0 . 7 0 5 \pm 0 . 0 6 1
$$

$$
1 . 2 2 3 \pm 0 . 1 4 7
$$

$$
\mathbf { 0 . 0 6 8 \pm 0 . 0 5 5 }
$$

$$
0 . 5 2 7 \pm 0 . 1 4 5
$$

$$
0 . 6 7 4 \pm 0 . 1 0 4
$$

$$
0 . 9 5 4 \pm 0 . 1 6 4
$$

$$
0 . 6 5 3 \pm 0 . 0 7 0
$$

$$
1 . 0 7 6 \pm 0 . 2 2 8
$$

$$
0 . 1 7 9 \pm 0 . 1 3 3
$$

$$
0 . 2 8 4 \pm 0 . 1 4 6
$$

$$
0 . 7 4 6 \pm 0 . 2 9 3
$$

$$
1 . 5 5 2 \pm 0 . 7 4 5
$$

$$
0 . 4 8 9 \pm 0 . 0 7 2
$$

$$
0 . 8 1 3 \pm 0 . 1 8 9
$$

$$
\mathrm { N F \_ O G p a p e r }
$$

$$
0 . 0 1 0 \pm 0 . 0 0 7
$$

$$
0 . 2 6 9 \pm 0 . 0 4 3
$$

$$
\mathbf { 0 . 0 1 0 \pm 0 . 0 0 4 }
$$

$$
0 . 0 5 9 \pm 0 . 0 2 3
$$

$$
0 . 0 1 1 \pm 0 . 0 0 6
$$

$$
0 . 0 1 6 \pm 0 . 0 1 5
$$

$$
0 . 3 1 5 \pm 0 . 0 2 9
$$

$$
0 . 0 1 5 \pm 0 . 0 1 8
$$

$$
0 . 0 9 7 \pm 0 . 0 3 1
$$

$$
\mathbf { 0 . 0 1 5 \pm 0 . 0 1 2 }
$$

$$
0 . 0 4 0 \pm 0 . 0 1 9
$$

$$
0 . 2 8 2 \pm 0 . 0 4 1
$$

$$
0 . 0 5 7 \pm 0 . 0 2 8
$$

$$
0 . 1 3 1 \pm 0 . 0 3 8
$$

$$
\mathbf { 0 . 0 3 0 \pm 0 . 0 2 0 }
$$

$$
\mathbf { 0 . 0 2 5 \pm 0 . 0 1 8 }
$$

$$
0 . 2 6 3 \pm 0 . 0 4 2
$$

$$
0 . 1 2 6 \pm 0 . 0 4 3
$$

$$
0 . 1 6 6 \pm 0 . 0 4 5
$$

$$
0 . 0 6 5 \pm 0 . 0 3 8
$$

$$
0 . 2 1 6 \pm 0 . 0 3 1
$$

$$
0 . 2 3 9 \pm 0 . 0 5 7
$$

$$
0 . 2 0 1 \pm 0 . 0 4 2
$$

$$
0 . 1 2 1 \pm 0 . 0 4 8
$$

$$
0 . 2 7 7 \pm 0 . 0 9 0
$$

$$
0 . 1 9 7 \pm 0 . 0 4 0
$$

$$
0 . 3 9 4 \pm 0 . 1 0 8
$$

$$
0 . 2 8 7 \pm 0 . 0 8 3
$$

$$
0 . 2 2 6 \pm 0 . 0 9 8
$$

$$
0 . 3 5 7 \pm 0 . 1 1 6
$$

$$
0 . 1 7 5 \pm 0 . 0 5 3
$$

$$
0 . 6 7 5 \pm 0 . 1 6 6
$$

$$
0 . 3 1 8 \pm 0 . 1 1 6
$$

$$
0 . 3 6 8 \pm 0 . 1 2 1
$$

$$
0 . 3 8 8 \pm 0 . 1 3 8
$$

$$
0 . 1 0 4 \pm 0 . 0 5 0
$$

$$
0 . 7 2 0 \pm 0 . 2 0 4
$$

$$
0 . 2 9 8 \pm 0 . 1 0 6
$$

$$
0 . 5 5 2 \pm 0 . 2 2 3
$$

$$
0 . 2 7 2 \pm 0 . 1 2 2
$$

$$
\mathbf { 0 . 0 4 } 2 \pm \mathbf { 0 . 0 2 4 }
$$

$$
0 . 9 0 1 \pm 0 . 2 8 1
$$

$$
0 . 4 1 4 \pm 0 . 2 3 2
$$

$$
1 . 2 3 2 \pm 0 . 5 2 7
$$

Table 30: PCA Procrustes distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 0 2 4 \pm 0 . 0 0 4$ </td><td> $0 . 0 4 5 \pm 0 . 0 0 6$ </td><td> $0 . 0 4 3 \pm 0 . 0 0 3$ </td><td> $0 . 0 6 0 \pm 0 . 0 0 5$ </td><td> $0 . 1 2 2 \pm 0 . 0 0 7$ </td><td> $0 . 1 1 2 \pm 0 . 0 1 1$ </td></tr><tr><td>20%</td><td> $0 . 0 6 7 \pm 0 . 0 0 8$ </td><td> $0 . 0 9 6 \pm 0 . 0 0 7$ </td><td> $0 . 0 9 1 \pm 0 . 0 0 6$ </td><td> $0 . 1 2 1 \pm 0 . 0 0 8$ </td><td> $0 . 2 2 7 \pm 0 . 0 2 0$ </td><td> $0 . 2 2 5 \pm 0 . 0 2 0$ </td></tr><tr><td>30%</td><td> $0 . 1 2 4 \pm 0 . 0 1 0$ </td><td> $0 . 1 5 4 \pm 0 . 0 2 0$ </td><td> $0 . 1 4 2 \pm 0 . 0 0 5$ </td><td> $0 . 1 8 5 \pm 0 . 0 0 9$ </td><td> $0 . 3 3 5 \pm 0 . 0 1 5$ </td><td> $0 . 3 6 1 \pm 0 . 0 2 2$ </td></tr><tr><td>40%</td><td> $0 . 2 0 8 \pm 0 . 0 1 1$ </td><td> $0 . 1 9 8 \pm 0 . 0 1 1$ </td><td> $0 . 1 9 8 \pm 0 . 0 0 7$ </td><td> $0 . 2 5 7 \pm 0 . 0 0 8$ </td><td> $0 . 4 3 6 \pm 0 . 0 1 7$ </td><td> $0 . 3 6 7 \pm 0 . 0 1 9$ </td></tr><tr><td>50%</td><td> $0 . 3 1 9 \pm 0 . 0 2 1$ </td><td> $0 . 2 8 4 \pm 0 . 0 1 1$ </td><td> $0 . 2 8 2 \pm 0 . 0 1 4$ </td><td> $0 . 3 5 4 \pm 0 . 0 1 6$ </td><td> $0 . 5 5 9 \pm 0 . 0 1 8$ </td><td> $0 . 4 9 2 \pm 0 . 0 2 5$ </td></tr><tr><td>60%</td><td> $0 . 4 4 1 \pm 0 . 0 1 5$ </td><td> $0 . 3 9 3 \pm 0 . 0 1 8$ </td><td> $0 . 3 7 8 \pm 0 . 0 1 1$ </td><td> $0 . 4 5 5 \pm 0 . 0 1 3$ </td><td> $0 . 6 5 7 \pm 0 . 0 2 4$ </td><td>0.584 ± 0.020</td></tr><tr><td>70%</td><td> $0 . 5 6 2 \pm 0 . 0 2 4$ </td><td> $0 . 5 0 3 \pm 0 . 0 1 3$ </td><td> $0 . 4 7 6 \pm 0 . 0 1 1$ </td><td> $0 . 5 5 2 \pm 0 . 0 1 4$ </td><td> $0 . 7 6 1 \pm 0 . 0 2 2$ </td><td> $0 . 7 1 2 \pm 0 . 0 2 5$ </td></tr><tr><td>80%</td><td> $0 . 7 0 0 \pm 0 . 0 3 0$ </td><td> $0 . 6 2 7 \pm 0 . 0 2 2$ </td><td> $0 . 6 1 0 \pm 0 . 0 1 4$ </td><td> $0 . 6 7 3 \pm 0 . 0 1 1$ </td><td> $0 . 8 4 9 \pm 0 . 0 1 3$ </td><td> $0 . 8 3 4 \pm 0 . 0 1 3$ </td></tr><tr><td>90%</td><td> $0 . 8 3 6 \pm 0 . 0 2 7$ </td><td> $0 . 7 9 3 \pm 0 . 0 2 4$ </td><td> $0 . 7 8 0 \pm 0 . 0 1 6$ </td><td> $0 . 8 1 3 \pm 0 . 0 1 4$ </td><td> $0 . 9 2 8 \pm 0 . 0 1 4$ </td><td> $0 . 9 2 9 \pm 0 . 0 1 2$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td> $\mathrm { S o f t I m p u t e }$ </td><td> $\mathrm { S o f t F o r e s t }$ </td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 0 2 1 \pm 0 . 0 0 3$ </td><td> $0 . 1 1 2 \pm 0 . 0 1 0$ </td><td> $0 . 0 2 2 \pm 0 . 0 0 4$ </td><td> $0 . 0 4 5 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 0 2 0 \pm 0 . 0 0 3 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 0 5 5 \pm 0 . 0 0 6$ </td><td> $0 . 2 4 0 \pm 0 . 0 2 7$ </td><td> $0 . 0 5 7 \pm 0 . 0 0 6$ </td><td> $0 . 0 9 5 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 0 5 3 \pm 0 . 0 0 4 }$ </td><td></td></tr><tr><td>30%</td><td> $0 . 1 0 1 \pm 0 . 0 0 6$ </td><td> $0 . 3 8 2 \pm 0 . 0 2 8$ </td><td> $0 . 1 0 5 \pm 0 . 0 0 7$ </td><td> $0 . 1 5 1 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 0 9 4 \pm 0 . 0 0 5 }$ </td><td></td></tr><tr><td>40%</td><td> $0 . 1 8 5 \pm 0 . 0 1 3$ </td><td> $0 . 5 2 3 \pm 0 . 0 2 3$ </td><td> $0 . 1 7 6 \pm 0 . 0 1 3$ </td><td> $0 . 2 0 9 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 1 4 9 \pm 0 . 0 0 9 }$ </td><td></td></tr><tr><td>50%</td><td> $0 . 3 1 0 \pm 0 . 0 1 9$ </td><td> $0 . 6 4 8 \pm 0 . 0 1 9$ </td><td> $0 . 2 8 6 \pm 0 . 0 1 6$ </td><td> $0 . 2 9 3 \pm 0 . 0 1 3$ </td><td> $\mathbf { 0 . 2 3 5 \pm 0 . 0 1 3 }$ </td><td></td></tr><tr><td>60%</td><td> $0 . 4 5 1 \pm 0 . 0 1 6$ </td><td> $0 . 7 5 3 \pm 0 . 0 2 9$ </td><td> $0 . 4 2 4 \pm 0 . 0 1 5$ </td><td> $0 . 3 8 7 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 3 4 0 \pm 0 . 0 1 4 }$ </td><td></td></tr><tr><td>70%</td><td> $0 . 6 0 0 \pm 0 . 0 3 1$ </td><td> $0 . 8 3 6 \pm 0 . 0 2 1$ </td><td> $0 . 5 9 5 \pm 0 . 0 3 0$ </td><td> $0 . 4 8 3 \pm 0 . 0 1 8$ </td><td> $\mathbf { 0 . 4 5 1 \pm 0 . 0 2 6 }$ </td><td></td></tr><tr><td>80%</td><td> $0 . 7 4 3 \pm 0 . 0 2 2$ </td><td> $0 . 8 9 6 \pm 0 . 0 1 5$ </td><td> $0 . 7 6 5 \pm 0 . 0 2 8$ </td><td> $\mathbf { 0 . 6 0 7 \pm 0 . 0 1 8 }$ </td><td> $0 . 6 2 0 \pm 0 . 0 2 9$ </td><td></td></tr><tr><td>90%</td><td> $0 . 8 7 7 \pm 0 . 0 1 8$ </td><td> $0 . 9 5 2 \pm 0 . 0 1 2$ </td><td> $0 . 9 0 4 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 1 8 }$ </td><td> $0 . 8 0 9 \pm 0 . 0 2 5$ </td><td></td></tr></table>

Table 31: PLS Procrustes distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 0 2 7 \pm 0 . 0 0 4$ </td><td> $0 . 0 4 7 \pm 0 . 0 0 7$ </td><td> $0 . 0 4 6 \pm 0 . 0 0 4$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 6$ </td><td> $0 . 1 4 0 \pm 0 . 0 1 1$ </td><td> $0 . 1 3 4 \pm 0 . 0 1 4$ </td></tr><tr><td>20%</td><td> $0 . 0 6 9 \pm 0 . 0 0 7$ </td><td> $0 . 1 0 0 \pm 0 . 0 0 6$ </td><td> $0 . 0 9 6 \pm 0 . 0 0 6$ </td><td> $0 . 1 2 5 \pm 0 . 0 0 9$ </td><td> $0 . 2 5 4 \pm 0 . 0 2 2$ </td><td> $0 . 2 5 7 \pm 0 . 0 2 4$ </td></tr><tr><td>30%</td><td> $0 . 1 2 4 \pm 0 . 0 0 7$ </td><td> $0 . 1 5 7 \pm 0 . 0 1 7$ </td><td> $0 . 1 4 8 \pm 0 . 0 0 7$ </td><td> $0 . 1 8 7 \pm 0 . 0 1 0$ </td><td> $0 . 3 6 4 \pm 0 . 0 2 0$ </td><td> $0 . 3 9 3 \pm 0 . 0 2 6$ </td></tr><tr><td>40%</td><td> $0 . 2 0 7 \pm 0 . 0 1 2$ </td><td> $0 . 2 0 3 \pm 0 . 0 1 3$ </td><td> $0 . 2 0 6 \pm 0 . 0 0 9$ </td><td> $0 . 2 6 0 \pm 0 . 0 1 0$ </td><td> $0 . 4 6 6 \pm 0 . 0 2 0$ </td><td> $0 . 4 0 3 \pm 0 . 0 2 0$ </td></tr><tr><td>50%</td><td> $0 . 3 1 4 \pm 0 . 0 2 0$ </td><td> $0 . 2 9 0 \pm 0 . 0 1 2$ </td><td> $0 . 2 9 2 \pm 0 . 0 1 4$ </td><td> $0 . 3 5 8 \pm 0 . 0 1 9$ </td><td> $0 . 5 8 9 \pm 0 . 0 2 0$ </td><td> $0 . 5 2 6 \pm 0 . 0 2 4$ </td></tr><tr><td>60%</td><td> $0 . 4 3 9 \pm 0 . 0 2 1$ </td><td> $0 . 3 9 6 \pm 0 . 0 2 0$ </td><td> $0 . 3 8 5 \pm 0 . 0 1 6$ </td><td> $0 . 4 5 5 \pm 0 . 0 1 6$ </td><td> $0 . 6 8 0 \pm 0 . 0 2 5$ </td><td> $0 . 6 1 0 \pm 0 . 0 2 3$ </td></tr><tr><td>70%</td><td> $0 . 5 6 2 \pm 0 . 0 2 9$ </td><td> $0 . 5 1 0 \pm 0 . 0 1 1$ </td><td> $0 . 4 8 7 \pm 0 . 0 0 9$ </td><td> $0 . 5 5 6 \pm 0 . 0 1 3$ </td><td> $0 . 7 7 8 \pm 0 . 0 2 3$ </td><td> $0 . 7 3 1 \pm 0 . 0 2 5$ </td></tr><tr><td>80%</td><td> $0 . 6 9 9 \pm 0 . 0 3 1$ </td><td> $0 . 6 2 6 \pm 0 . 0 2 0$ </td><td> $0 . 6 1 4 \pm 0 . 0 1 3$ </td><td> $0 . 6 7 6 \pm 0 . 0 1 3$ </td><td> $0 . 8 6 1 \pm 0 . 0 1 6$ </td><td> $0 . 8 4 5 \pm 0 . 0 1 4$ </td></tr><tr><td>90%</td><td> $0 . 8 3 8 \pm 0 . 0 2 7$ </td><td> $0 . 7 9 8 \pm 0 . 0 2 4$ </td><td> $0 . 7 8 7 \pm 0 . 0 1 4$ </td><td> $0 . 8 1 7 \pm 0 . 0 1 4$ </td><td> $0 . 9 3 4 \pm 0 . 0 1 4$ </td><td> $0 . 9 3 4 \pm 0 . 0 1 3$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 0 2 4 \pm 0 . 0 0 4$ </td><td> $0 . 1 3 4 \pm 0 . 0 1 4$ </td><td> $0 . 0 2 5 \pm 0 . 0 0 5$ </td><td> $0 . 0 4 9 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 0 2 3 \pm 0 . 0 0 3 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 0 6 0 \pm 0 . 0 0 7$ </td><td> $0 . 2 7 2 \pm 0 . 0 3 1$ </td><td> $0 . 0 6 2 \pm 0 . 0 0 8$ </td><td> $0 . 1 0 1 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 0 5 7 \pm 0 . 0 0 5 }$ </td><td></td></tr><tr><td>30%</td><td> $0 . 1 0 6 \pm 0 . 0 0 8$ </td><td> $0 . 4 1 4 \pm 0 . 0 3 2$ </td><td> $0 . 1 1 2 \pm 0 . 0 0 9$ </td><td> $0 . 1 5 6 \pm 0 . 0 0 7$ </td><td></td><td></td></tr><tr><td>40%</td><td> $0 . 1 9 3 \pm 0 . 0 1 5$ </td><td> $0 . 5 5 3 \pm 0 . 0 2 6$ </td><td> $0 . 1 8 9 \pm 0 . 0 1 3$ </td><td> $0 . 2 1 5 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 0 9 7 \pm 0 . 0 0 5 }$ </td><td></td></tr><tr><td>50%</td><td> $0 . 3 2 5 \pm 0 . 0 2 4$ </td><td> $0 . 6 7 5 \pm 0 . 0 2 1$ </td><td> $0 . 3 0 0 \pm 0 . 0 2 0$ </td><td> $0 . 3 0 2 \pm 0 . 0 1 4$ </td><td> $\mathbf { 0 . 1 5 5 \pm 0 . 0 1 0 }$ </td><td></td></tr><tr><td>60%</td><td> $0 . 4 6 3 \pm 0 . 0 2 1$ </td><td></td><td> $0 . 4 3 9 \pm 0 . 0 1 7$ </td><td> $0 . 3 9 3 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 2 4 1 \pm 0 . 0 1 1 }$ </td><td></td></tr><tr><td></td><td></td><td> $0 . 7 7 1 \pm 0 . 0 3 1$ </td><td></td><td></td><td> $\mathbf { 0 . 3 4 1 \pm 0 . 0 1 9 }$ </td><td></td></tr><tr><td>70% 80%</td><td> $0 . 6 1 0 \pm 0 . 0 3 1$ </td><td> $0 . 8 4 9 \pm 0 . 0 2 2$ </td><td> $0 . 6 0 5 \pm 0 . 0 2 6$   $0 . 7 7 1 \pm 0 . 0 3 0$ </td><td> $0 . 4 9 2 \pm 0 . 0 1 6$   $\mathbf { 0 . 6 1 1 \pm 0 . 0 1 9 }$ </td><td> $\mathbf { 0 . 4 5 4 \pm 0 . 0 2 3 }$ </td><td></td></tr><tr><td>90%</td><td> $0 . 7 4 5 \pm 0 . 0 2 6$   $0 . 8 8 0 \pm 0 . 0 2 0$ </td><td> $0 . 9 0 4 \pm 0 . 0 1 7$   $0 . 9 5 5 \pm 0 . 0 1 3$ </td><td> $0 . 9 0 7 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 7 7 9 \pm 0 . 0 1 6 }$ </td><td> $0 . 6 1 7 \pm 0 . 0 2 9$   $0 . 8 0 8 \pm 0 . 0 2 6$ </td><td></td></tr></table>

Table 32: Pearson log-p correlation across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 9 9 3 \pm 0 . 0 0 4$ </td><td> $0 . 9 9 3 \pm 0 . 0 0 4$ </td><td> $0 . 9 8 3 \pm 0 . 0 0 8$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 2$ </td><td> $0 . 8 9 2 \pm 0 . 0 3 1$ </td><td> $0 . 7 6 1 \pm 0 . 0 3 3$ </td></tr><tr><td>20%</td><td> $0 . 9 7 0 \pm 0 . 0 0 8$ </td><td> $0 . 9 7 1 \pm 0 . 0 1 5$ </td><td> $0 . 9 5 2 \pm 0 . 0 1 6$ </td><td> $0 . 9 8 4 \pm 0 . 0 0 8$ </td><td> $0 . 8 5 4 \pm 0 . 0 3 5$ </td><td> $0 . 6 3 1 \pm 0 . 0 4 5$ </td></tr><tr><td>30%</td><td> $0 . 9 2 6 \pm 0 . 0 1 5$ </td><td> $0 . 9 6 6 \pm 0 . 0 1 5$ </td><td> $0 . 9 2 7 \pm 0 . 0 2 4$ </td><td> $\mathbf { 0 . 9 7 3 \pm 0 . 0 1 4 }$ </td><td> $0 . 7 6 3 \pm 0 . 0 4 0$ </td><td> $0 . 4 8 6 \pm 0 . 0 4 7$ </td></tr><tr><td>40%</td><td> $0 . 8 8 5 \pm 0 . 0 3 2$ </td><td> $0 . 9 5 9 \pm 0 . 0 1 8$ </td><td> $0 . 9 2 0 \pm 0 . 0 2 3$ </td><td> $\mathbf { 0 . 9 7 0 \pm 0 . 0 1 5 }$ </td><td> $0 . 6 6 1 \pm 0 . 0 5 6$ </td><td> $0 . 2 9 4 \pm 0 . 0 5 3$ </td></tr><tr><td>50%</td><td> $0 . 8 4 6 \pm 0 . 0 2 3$ </td><td> $0 . 9 2 7 \pm 0 . 0 2 1$ </td><td> $0 . 8 9 0 \pm 0 . 0 2 9$ </td><td> $\mathbf { 0 . 9 5 1 \pm 0 . 0 1 5 }$ </td><td> $0 . 6 7 5 \pm 0 . 0 8 2$ </td><td> $0 . 5 2 2 \pm 0 . 1 1 3$ </td></tr><tr><td>60%</td><td> $0 . 7 9 4 \pm 0 . 0 3 2$ </td><td> $0 . 9 1 9 \pm 0 . 0 2 9$ </td><td> $0 . 8 7 5 \pm 0 . 0 3 6$ </td><td> $\mathbf { 0 . 9 4 1 \pm 0 . 0 2 9 }$ </td><td> $0 . 6 0 0 \pm 0 . 0 6 4$ </td><td> $0 . 3 2 5 \pm 0 . 1 1 9$ </td></tr><tr><td>70%</td><td> $0 . 7 4 3 \pm 0 . 0 4 0$ </td><td> $0 . 8 9 5 \pm 0 . 0 2 3$ </td><td> $0 . 8 7 0 \pm 0 . 0 3 8$ </td><td> $\mathbf { 0 . 9 3 3 \pm 0 . 0 2 4 }$ </td><td> $0 . 4 9 6 \pm 0 . 1 3 4$ </td><td> $0 . 1 8 3 \pm 0 . 0 9 4$ </td></tr><tr><td>80%</td><td> $0 . 6 6 5 \pm 0 . 0 6 7$ </td><td> $0 . 8 3 6 \pm 0 . 0 5 0$ </td><td> $0 . 8 2 9 \pm 0 . 0 4 9$ </td><td> $\mathbf { 0 . 9 0 2 \pm 0 . 0 3 6 }$ </td><td> $0 . 4 0 2 \pm 0 . 1 1 5$ </td><td> $0 . 0 9 2 \pm 0 . 0 6 9$ </td></tr><tr><td>90%</td><td> $0 . 5 7 2 \pm 0 . 0 4 8$ </td><td> $0 . 7 2 0 \pm 0 . 0 5 3$ </td><td> $0 . 7 5 4 \pm 0 . 0 6 9$ </td><td> $\mathbf { 0 . 8 2 3 \pm 0 . 0 6 1 }$ </td><td> $0 . 3 8 6 \pm 0 . 1 3 7$ </td><td> $0 . 1 3 6 \pm 0 . 0 9 4$ </td></tr><tr><td>Rate</td><td>NF_OGpaper</td><td> $\mathrm { S o f t I m p u t e }$ </td><td> $\mathrm { S o f t F o r e s t }$ </td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $\mathbf { 0 . 9 9 5 \pm 0 . 0 0 3 }$ </td><td> $0 . 7 5 2 \pm 0 . 0 3 3$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 4$ </td><td> $0 . 9 7 5 \pm 0 . 0 0 8$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 3$ </td><td></td></tr><tr><td>20%</td><td> $\mathbf { 0 . 9 8 7 \pm 0 . 0 0 6 }$ </td><td> $0 . 6 3 7 \pm 0 . 0 3 8$ </td><td> $0 . 9 8 6 \pm 0 . 0 0 7$ </td><td> $0 . 9 2 0 \pm 0 . 0 1 9$ </td><td> $0 . 9 7 0 \pm 0 . 0 0 8$ </td><td></td></tr><tr><td>30%</td><td> $0 . 9 6 6 \pm 0 . 0 0 7$ </td><td> $0 . 5 4 1 \pm 0 . 0 4 3$ </td><td> $0 . 9 5 4 \pm 0 . 0 1 2$ </td><td> $0 . 8 5 6 \pm 0 . 0 2 5$ </td><td> $0 . 9 4 0 \pm 0 . 0 1 5$ </td><td></td></tr><tr><td>40%</td><td> $0 . 9 2 4 \pm 0 . 0 2 9$ </td><td> $0 . 4 4 2 \pm 0 . 0 5 2$ </td><td> $0 . 8 7 9 \pm 0 . 0 2 5$ </td><td> $0 . 7 9 6 \pm 0 . 0 3 1$ </td><td> $0 . 9 0 2 \pm 0 . 0 3 8$ </td><td></td></tr><tr><td>50%</td><td> $0 . 8 1 9 \pm 0 . 0 4 2$ </td><td> $0 . 4 5 2 \pm 0 . 0 7 3$ </td><td> $0 . 8 0 8 \pm 0 . 0 5 3$ </td><td> $0 . 7 1 4 \pm 0 . 0 2 5$ </td><td> $0 . 8 5 7 \pm 0 . 0 2 2$ </td><td></td></tr><tr><td>60%</td><td> $0 . 6 5 5 \pm 0 . 0 9 0$ </td><td> $0 . 3 7 7 \pm 0 . 0 7 1$ </td><td> $0 . 6 9 8 \pm 0 . 0 5 1$ </td><td> $0 . 6 5 6 \pm 0 . 0 3 8$ </td><td> $0 . 7 9 3 \pm 0 . 0 3 0$ </td><td></td></tr><tr><td>70%</td><td> $0 . 5 3 2 \pm 0 . 0 9 1$ </td><td> $0 . 3 0 6 \pm 0 . 1 0 3$ </td><td> $0 . 5 6 3 \pm 0 . 0 7 4$ </td><td> $0 . 6 1 9 \pm 0 . 0 5 5$ </td><td> $0 . 7 4 9 \pm 0 . 0 5 6$ </td><td></td></tr><tr><td>80%</td><td> $0 . 5 1 1 \pm 0 . 0 7 7$ </td><td> $0 . 2 4 1 \pm 0 . 0 7 6$ </td><td> $0 . 5 3 9 \pm 0 . 0 6 6$ </td><td> $0 . 5 4 2 \pm 0 . 0 3 7$ </td><td> $0 . 6 9 0 \pm 0 . 0 7 0$ </td><td></td></tr><tr><td>90%</td><td> $0 . 5 4 9 \pm 0 . 0 6 6$ </td><td> $0 . 2 2 9 \pm 0 . 1 0 1$ </td><td> $0 . 5 6 9 \pm 0 . 0 7 4$ </td><td> $0 . 5 2 3 \pm 0 . 0 5 5$ </td><td> $0 . 6 0 4 \pm 0 . 0 8 8$ </td><td></td></tr></table>

## E.4 MAR RESULTS ON THE HOUSING DATASET

Table 33: NRMSE across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 8 5 2 \pm 0 . 0 3 3$ </td><td> $1 . 0 6 3 \pm 0 . 0 6 1$ </td><td> $1 . 0 0 9 \pm 0 . 0 0 5$ </td><td> $1 . 1 1 9 \pm 0 . 0 1 2$ </td><td> $1 . 8 0 8 \pm 0 . 0 6 6$ </td><td> $1 . 4 1 5 \pm 0 . 0 4 6$ </td></tr><tr><td>20%</td><td> $0 . 9 2 0 \pm 0 . 0 2 9$ </td><td> $1 . 0 6 0 \pm 0 . 0 2 1$ </td><td> $1 . 0 1 2 \pm 0 . 0 0 3$ </td><td> $1 . 1 3 2 \pm 0 . 0 0 5$ </td><td> $1 . 7 7 9 \pm 0 . 0 5 0$ </td><td> $1 . 4 5 8 \pm 0 . 0 4 4$ </td></tr><tr><td>30%</td><td> $0 . 9 7 9 \pm 0 . 0 2 0$ </td><td> $1 . 0 6 8 \pm 0 . 0 1 9$ </td><td> $1 . 0 1 8 \pm 0 . 0 0 3$ </td><td> $1 . 1 2 4 \pm 0 . 0 0 8$ </td><td> $1 . 7 7 4 \pm 0 . 0 3 9$ </td><td> $1 . 5 3 4 \pm 0 . 0 2 9$ </td></tr><tr><td>40%</td><td> $1 . 0 3 2 \pm 0 . 0 2 7$ </td><td> $1 . 0 5 5 \pm 0 . 0 1 0$ </td><td> $1 . 0 2 3 \pm 0 . 0 0 3$ </td><td> $1 . 1 2 4 \pm 0 . 0 0 7$ </td><td> $1 . 7 3 3 \pm 0 . 0 4 0$ </td><td> $1 . 3 7 1 \pm 0 . 0 3 0$ </td></tr><tr><td>50%</td><td> $1 . 0 8 4 \pm 0 . 0 1 9$ </td><td> $1 . 0 6 8 \pm 0 . 0 0 4$ </td><td> $1 . 0 3 3 \pm 0 . 0 0 4$ </td><td> $1 . 1 2 7 \pm 0 . 0 0 2$ </td><td> $1 . 7 0 2 \pm 0 . 0 3 1$ </td><td> $2 . 0 0 7 \pm 0 . 0 6 5$ </td></tr><tr><td>60%</td><td> $1 . 1 0 7 \pm 0 . 0 2 6$ </td><td> $1 . 0 9 9 \pm 0 . 0 0 8$ </td><td> $1 . 0 4 6 \pm 0 . 0 0 8$ </td><td> $1 . 1 3 2 \pm 0 . 0 0 6$ </td><td>1.687 ± 0.020</td><td> $1 . 9 6 5 \pm 0 . 0 5 6$ </td></tr><tr><td>70%</td><td> $1 . 1 6 3 \pm 0 . 0 6 9$ </td><td> $1 . 1 2 1 \pm 0 . 0 0 6$ </td><td> $1 . 0 7 1 \pm 0 . 0 0 6$ </td><td> $1 . 1 5 6 \pm 0 . 0 3 2$ </td><td> $1 . 6 8 9 \pm 0 . 0 4 5$ </td><td> $1 . 8 9 4 \pm 0 . 0 3 9$ </td></tr><tr><td>80%</td><td> $1 . 1 7 6 \pm 0 . 0 3 9$ </td><td> $1 . 1 2 9 \pm 0 . 0 1 1$ </td><td> $1 . 0 8 8 \pm 0 . 0 0 9$ </td><td> $1 . 2 0 4 \pm 0 . 0 3 1$ </td><td> $1 . 7 0 4 \pm 0 . 0 3 2$ </td><td> $1 . 9 0 3 \pm 0 . 0 4 1$ </td></tr><tr><td>90%</td><td> $1 . 1 9 5 \pm 0 . 0 4 0$ </td><td> $1 . 1 2 4 \pm 0 . 0 1 4$ </td><td> $1 . 0 9 1 \pm 0 . 0 1 3$ </td><td> $1 . 2 0 8 \pm 0 . 0 2 2$ </td><td> $1 . 6 9 9 \pm 0 . 0 2 8$ </td><td> $1 . 9 3 4 \pm 0 . 0 4 0$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 8 3 5 \pm 0 . 0 2 4$ </td><td> $1 . 3 2 6 \pm 0 . 0 6 3$ </td><td> $0 . 8 3 5 \pm 0 . 0 2 5$ </td><td> $1 . 0 3 6 \pm 0 . 0 2 7$ </td><td> $\mathbf { 0 . 8 3 4 \pm 0 . 0 2 4 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 8 5 8 \pm 0 . 0 2 2$ </td><td> $1 . 4 2 5 \pm 0 . 0 5 1$ </td><td> $0 . 8 5 8 \pm 0 . 0 1 7$ </td><td> $1 . 0 2 4 \pm 0 . 0 1 6$ </td><td> $\mathbf { 0 . 8 5 3 \pm 0 . 0 2 0 }$ </td><td></td></tr><tr><td>30%</td><td> $0 . 9 0 2 \pm 0 . 0 1 2$ </td><td> $1 . 5 2 0 \pm 0 . 0 3 3$ </td><td> $0 . 9 0 3 \pm 0 . 0 1 4$ </td><td> $1 . 0 2 1 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 8 9 6 \pm 0 . 0 1 3 }$ </td><td></td></tr><tr><td>40%</td><td> $0 . 9 6 3 \pm 0 . 0 1 8$ </td><td> $1 . 5 9 1 \pm 0 . 0 4 1$ </td><td> $0 . 9 3 0 \pm 0 . 0 1 4$ </td><td> $1 . 0 2 4 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 9 2 3 \pm 0 . 0 0 9 }$ </td><td></td></tr><tr><td>50%</td><td> $1 . 0 1 4 \pm 0 . 0 1 3$ </td><td> $1 . 6 8 5 \pm 0 . 0 2 4$ </td><td> $0 . 9 7 5 \pm 0 . 0 0 9$ </td><td> $1 . 0 2 7 \pm 0 . 0 0 4$ </td><td> $\mathbf { 0 . 9 5 8 \pm 0 . 0 0 7 }$ </td><td></td></tr><tr><td>60%</td><td> $1 . 0 4 7 \pm 0 . 0 1 1$ </td><td> $1 . 7 4 1 \pm 0 . 0 2 2$ </td><td> $1 . 0 1 6 \pm 0 . 0 2 4$ </td><td> $1 . 0 3 5 \pm 0 . 0 0 8$ </td><td> $\mathbf { 0 . 9 9 9 \pm 0 . 0 1 3 }$ </td><td></td></tr><tr><td>70%</td><td> $1 . 0 7 3 \pm 0 . 0 1 7$ </td><td> $1 . 8 1 2 \pm 0 . 0 1 5$ </td><td> $\mathbf { 1 . 0 5 2 \pm 0 . 0 2 1 }$ </td><td> $1 . 0 5 7 \pm 0 . 0 1 1$ </td><td> $1 . 0 5 5 \pm 0 . 0 1 9$ </td><td></td></tr><tr><td>80%</td><td> $1 . 0 9 3 \pm 0 . 0 2 2$ </td><td> $1 . 8 6 6 \pm 0 . 0 1 3$ </td><td> $1 . 0 8 0 \pm 0 . 0 2 4$ </td><td> $\mathbf { 1 . 0 8 0 \pm 0 . 0 0 7 }$ </td><td> $1 . 1 1 5 \pm 0 . 0 3 2$ </td><td></td></tr><tr><td>90%</td><td> $1 . 0 9 1 \pm 0 . 0 3 0$ </td><td> $1 . 9 0 5 \pm 0 . 0 1 3$ </td><td> $1 . 1 0 1 \pm 0 . 0 3 5$ </td><td> $\mathbf { 1 . 0 8 5 \pm 0 . 0 1 2 }$ </td><td> $1 . 1 2 8 \pm 0 . 0 3 7$ </td><td></td></tr></table>

Table 34: PFC across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td> $_ \mathrm { H a l f - m i n }$ </td><td> $\operatorname { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 1 3 4 \pm 0 . 0 1 9$ </td><td> $0 . 2 5 6 \pm 0 . 0 3 0$ </td><td> $0 . 2 2 9 \pm 0 . 0 2 2$ </td><td> $0 . 2 2 9 \pm 0 . 0 2 2$ </td><td> $0 . 3 0 1 \pm 0 . 0 1 8$ </td><td> $0 . 2 4 3 \pm 0 . 0 1 7$ </td></tr><tr><td>20%</td><td> $0 . 1 7 8 \pm 0 . 0 1 2$ </td><td> $0 . 2 4 4 \pm 0 . 0 2 4$ </td><td> $0 . 2 4 7 \pm 0 . 0 1 3$ </td><td> $0 . 2 4 7 \pm 0 . 0 1 3$ </td><td> $0 . 3 1 8 \pm 0 . 0 1 0$ </td><td> $0 . 2 7 6 \pm 0 . 0 1 4$ </td></tr><tr><td>30%</td><td> $0 . 2 0 4 \pm 0 . 0 1 6$ </td><td> $0 . 2 3 6 \pm 0 . 0 1 1$ </td><td> $0 . 2 3 5 \pm 0 . 0 1 3$ </td><td> $0 . 2 3 5 \pm 0 . 0 1 3$ </td><td> $0 . 3 1 0 \pm 0 . 0 1 6$ </td><td> $0 . 2 7 0 \pm 0 . 0 1 5$ </td></tr><tr><td>40%</td><td> $0 . 2 3 7 \pm 0 . 0 1 6$ </td><td> $0 . 2 4 2 \pm 0 . 0 0 6$ </td><td> $0 . 2 4 0 \pm 0 . 0 0 6$ </td><td> $0 . 2 4 0 \pm 0 . 0 0 6$ </td><td> $0 . 3 1 0 \pm 0 . 0 0 7$ </td><td> $0 . 2 4 1 \pm 0 . 0 1 1$ </td></tr><tr><td>50%</td><td> $0 . 2 5 8 \pm 0 . 0 1 2$ </td><td> $0 . 2 4 7 \pm 0 . 0 0 5$ </td><td> $0 . 2 4 4 \pm 0 . 0 0 2$ </td><td> $0 . 2 4 4 \pm 0 . 0 0 2$ </td><td> $0 . 3 1 5 \pm 0 . 0 0 3$ </td><td> $0 . 2 9 1 \pm 0 . 0 0 9$ </td></tr><tr><td>60%</td><td> $0 . 2 6 6 \pm 0 . 0 1 0$ </td><td> $0 . 2 5 9 \pm 0 . 0 0 5$ </td><td> $0 . 2 4 2 \pm 0 . 0 0 5$ </td><td> $0 . 2 4 2 \pm 0 . 0 0 5$ </td><td> $0 . 3 1 3 \pm 0 . 0 0 6$ </td><td> $0 . 2 9 3 \pm 0 . 0 0 8$ </td></tr><tr><td>70%</td><td> $0 . 2 8 4 \pm 0 . 0 2 2$ </td><td> $0 . 2 6 7 \pm 0 . 0 0 8$ </td><td> $0 . 2 5 6 \pm 0 . 0 2 4$ </td><td> $0 . 2 5 6 \pm 0 . 0 2 4$ </td><td>0.313 ± 0.005</td><td> $0 . 2 9 5 \pm 0 . 0 0 9$ </td></tr><tr><td>80%</td><td> $0 . 2 9 1 \pm 0 . 0 2 7$ </td><td> $0 . 2 8 7 \pm 0 . 0 1 2$ </td><td> $0 . 2 9 1 \pm 0 . 0 2 3$ </td><td> $0 . 2 9 1 \pm 0 . 0 2 3$ </td><td> $0 . 3 1 5 \pm 0 . 0 0 5$ </td><td> $0 . 2 9 9 \pm 0 . 0 0 5$ </td></tr><tr><td>90%</td><td> $0 . 3 0 2 \pm 0 . 0 2 0$ </td><td> $0 . 2 9 1 \pm 0 . 0 1 1$ </td><td> $0 . 2 9 5 \pm 0 . 0 1 8$ </td><td> $0 . 2 9 5 \pm 0 . 0 1 8$ </td><td> $0 . 3 1 8 \pm 0 . 0 0 3$ </td><td> $0 . 3 0 2 \pm 0 . 0 0 5$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td> $\mathrm { S o f t I m p u t e }$ </td><td> $\mathrm { S o f t F o r e s t }$ </td><td> $\operatorname { s v r }$ </td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $\mathbf { 0 . 1 2 8 \pm 0 . 0 1 9 }$ </td><td> $0 . 2 4 5 \pm 0 . 0 1 5$ </td><td> $0 . 1 3 2 \pm 0 . 0 1 7$ </td><td> $0 . 2 3 2 \pm 0 . 0 1 8$ </td><td> $0 . 1 2 9 \pm 0 . 0 1 7$ </td><td></td></tr><tr><td>20%</td><td> $0 . 1 6 2 \pm 0 . 0 1 7$ </td><td> $0 . 2 7 0 \pm 0 . 0 1 3$ </td><td> $\mathbf { 0 . 1 5 4 \pm 0 . 0 1 7 }$ </td><td> $0 . 2 4 8 \pm 0 . 0 1 5$ </td><td> $0 . 1 5 6 \pm 0 . 0 1 8$ </td><td></td></tr><tr><td>30%</td><td> $0 . 1 8 1 \pm 0 . 0 0 9$ </td><td> $0 . 2 6 4 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 1 7 7 \pm 0 . 0 1 1 }$ </td><td> $0 . 2 3 7 \pm 0 . 0 1 2$ </td><td> $0 . 1 7 8 \pm 0 . 0 1 1$ </td><td></td></tr><tr><td>40%</td><td> $0 . 2 0 7 \pm 0 . 0 1 0$ </td><td> $0 . 2 7 3 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 1 9 4 \pm 0 . 0 0 5 }$ </td><td> $0 . 2 4 1 \pm 0 . 0 0 6$ </td><td> $0 . 1 9 7 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>50%</td><td> $0 . 2 3 7 \pm 0 . 0 0 9$ </td><td> $0 . 2 8 9 \pm 0 . 0 0 4$ </td><td> $0 . 2 2 0 \pm 0 . 0 0 5$ </td><td> $0 . 2 4 3 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 2 1 9 \pm 0 . 0 0 7 }$ </td><td></td></tr><tr><td>60%</td><td> $0 . 2 5 5 \pm 0 . 0 0 9$ </td><td> $0 . 2 9 1 \pm 0 . 0 0 6$ </td><td> $0 . 2 4 7 \pm 0 . 0 1 3$ </td><td> $0 . 2 4 2 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 2 3 9 \pm 0 . 0 1 0 }$ </td><td></td></tr><tr><td>70%</td><td> $0 . 2 6 8 \pm 0 . 0 0 5$ </td><td> $0 . 2 9 6 \pm 0 . 0 0 5$ </td><td> $0 . 2 6 5 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 2 4 4 \pm 0 . 0 0 4 }$ </td><td> $0 . 2 6 4 \pm 0 . 0 1 8$ </td><td></td></tr><tr><td>80%</td><td> $0 . 2 7 9 \pm 0 . 0 0 9$ </td><td> $0 . 3 0 1 \pm 0 . 0 0 5$ </td><td> $0 . 2 8 2 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 2 7 0 \pm 0 . 0 2 7 }$ </td><td> $0 . 2 9 3 \pm 0 . 0 2 7$ </td><td></td></tr><tr><td>90%</td><td> $0 . 2 8 2 \pm 0 . 0 2 5$ </td><td> $0 . 3 0 7 \pm 0 . 0 0 3$ </td><td> $0 . 2 9 0 \pm 0 . 0 2 6$ </td><td> $\mathbf { 0 . 2 8 0 \pm 0 . 0 2 5 }$ </td><td> $0 . 2 8 8 \pm 0 . 0 2 5$ </td><td></td></tr></table>

Table 35: Gower’s distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td> $_ \mathrm { H a l f - m i n }$ </td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 1 2 8 \pm 0 . 0 1 2$ </td><td> $0 . 2 1 6 \pm 0 . 0 1 9$ </td><td> $0 . 2 0 3 \pm 0 . 0 1 7$ </td><td> $0 . 1 9 4 \pm 0 . 0 1 8$ </td><td> $0 . 3 2 3 \pm 0 . 0 1 7$ </td><td> $0 . 2 5 0 \pm 0 . 0 1 2$ </td></tr><tr><td>20%</td><td> $0 . 1 5 1 \pm 0 . 0 0 8$ </td><td> $0 . 2 1 4 \pm 0 . 0 1 2$ </td><td> $0 . 2 2 0 \pm 0 . 0 1 1$ </td><td> $0 . 2 1 2 \pm 0 . 0 1 0$ </td><td> $0 . 3 4 1 \pm 0 . 0 0 9$ </td><td> $0 . 2 7 8 \pm 0 . 0 1 2$ </td></tr><tr><td>30%</td><td> $0 . 1 6 8 \pm 0 . 0 1 2$ </td><td> $0 . 2 1 4 \pm 0 . 0 1 0$ </td><td> $0 . 2 1 1 \pm 0 . 0 1 0$ </td><td> $0 . 2 0 2 \pm 0 . 0 1 0$ </td><td> $0 . 3 3 4 \pm 0 . 0 1 0$ </td><td> $0 . 2 8 0 \pm 0 . 0 1 2$ </td></tr><tr><td>40%</td><td> $0 . 1 8 8 \pm 0 . 0 0 8$ </td><td> $0 . 2 2 3 \pm 0 . 0 0 3$ </td><td> $0 . 2 1 9 \pm 0 . 0 0 4$ </td><td> $0 . 2 1 3 \pm 0 . 0 0 4$ </td><td> $0 . 3 3 7 \pm 0 . 0 0 5$ </td><td> $0 . 2 4 8 \pm 0 . 0 0 9$ </td></tr><tr><td>50%</td><td> $0 . 2 0 5 \pm 0 . 0 0 7$ </td><td> $0 . 2 2 6 \pm 0 . 0 0 5$ </td><td> $0 . 2 2 2 \pm 0 . 0 0 3$ </td><td> $0 . 2 1 7 \pm 0 . 0 0 4$ </td><td> $0 . 3 3 8 \pm 0 . 0 0 5$ </td><td> $0 . 3 5 6 \pm 0 . 0 0 5$ </td></tr><tr><td>60%</td><td> $0 . 2 1 5 \pm 0 . 0 0 9$ </td><td> $0 . 2 3 8 \pm 0 . 0 0 3$ </td><td> $0 . 2 2 5 \pm 0 . 0 0 4$ </td><td> $0 . 2 1 9 \pm 0 . 0 0 4$ </td><td>0.338 ± 0.005</td><td> $0 . 3 4 7 \pm 0 . 0 0 9$ </td></tr><tr><td>70%</td><td> $0 . 2 3 1 \pm 0 . 0 1 8$ </td><td> $0 . 2 4 4 \pm 0 . 0 0 3$ </td><td> $0 . 2 3 3 \pm 0 . 0 1 2$ </td><td> $0 . 2 2 6 \pm 0 . 0 1 2$ </td><td> $0 . 3 3 7 \pm 0 . 0 0 4$ </td><td> $0 . 3 4 2 \pm 0 . 0 0 9$ </td></tr><tr><td>80%</td><td> $0 . 2 3 6 \pm 0 . 0 1 5$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 6$ </td><td> $0 . 2 5 1 \pm 0 . 0 1 0$ </td><td> $0 . 2 4 3 \pm 0 . 0 1 0$ </td><td> $0 . 3 3 8 \pm 0 . 0 0 6$ </td><td> $0 . 3 4 7 \pm 0 . 0 0 6$ </td></tr><tr><td>90%</td><td> $0 . 2 4 7 \pm 0 . 0 1 2$ </td><td> $0 . 2 5 4 \pm 0 . 0 0 6$ </td><td> $0 . 2 5 2 \pm 0 . 0 0 9$ </td><td> $0 . 2 4 4 \pm 0 . 0 0 9$ </td><td> $0 . 3 3 5 \pm 0 . 0 0 5$ </td><td> $0 . 3 4 7 \pm 0 . 0 0 7$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $\mathbf { 0 . 1 2 3 \pm 0 . 0 1 1 }$ </td><td> $0 . 2 3 7 \pm 0 . 0 1 1$ </td><td> $0 . 1 2 5 \pm 0 . 0 1 2$ </td><td> $0 . 2 0 7 \pm 0 . 0 1 2$ </td><td> $0 . 1 2 5 \pm 0 . 0 1 1$ </td><td></td></tr><tr><td>20%</td><td> $0 . 1 4 4 \pm 0 . 0 1 1$ </td><td> $0 . 2 6 5 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 1 4 0 \pm 0 . 0 0 9 }$ </td><td> $0 . 2 2 2 \pm 0 . 0 1 2$ </td><td> $0 . 1 4 2 \pm 0 . 0 0 8$ </td><td></td></tr><tr><td>30%</td><td> $0 . 1 5 9 \pm 0 . 0 0 7$ </td><td> $0 . 2 6 7 \pm 0 . 0 1 0$ </td><td> $0 . 1 5 7 \pm 0 . 0 0 8$ </td><td> $0 . 2 1 2 \pm 0 . 0 1 0$ </td><td> $\mathbf { 0 . 1 5 6 \pm 0 . 0 0 8 }$ </td><td></td></tr><tr><td>40%</td><td> $0 . 1 7 6 \pm 0 . 0 0 8$ </td><td> $0 . 2 8 7 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 1 7 1 \pm 0 . 0 0 3 }$ </td><td> $0 . 2 1 9 \pm 0 . 0 0 5$ </td><td> $0 . 1 7 2 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>50%</td><td> $0 . 1 9 9 \pm 0 . 0 0 6$ </td><td> $0 . 3 0 8 \pm 0 . 0 0 5$ </td><td> $0 . 1 8 9 \pm 0 . 0 0 4$ </td><td> $0 . 2 2 1 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 1 8 7 \pm 0 . 0 0 5 }$ </td><td></td></tr><tr><td>60%</td><td> $0 . 2 1 2 \pm 0 . 0 0 6$ </td><td> $0 . 3 2 0 \pm 0 . 0 0 6$ </td><td> $0 . 2 0 6 \pm 0 . 0 0 8$ </td><td> $0 . 2 2 2 \pm 0 . 0 0 5$ </td><td> $\mathbf { 0 . 2 0 2 \pm 0 . 0 0 6 }$ </td><td></td></tr><tr><td>70%</td><td> $\mathbf { 0 . 2 1 8 \pm 0 . 0 0 4 }$ </td><td> $0 . 3 3 3 \pm 0 . 0 0 5$ </td><td> $0 . 2 1 8 \pm 0 . 0 0 7$ </td><td> $0 . 2 2 6 \pm 0 . 0 0 4$ </td><td> $0 . 2 2 1 \pm 0 . 0 1 2$ </td><td></td></tr><tr><td>80%</td><td> $\mathbf { 0 . 2 2 5 \pm 0 . 0 0 7 }$ </td><td> $0 . 3 4 2 \pm 0 . 0 0 5$ </td><td> $0 . 2 3 0 \pm 0 . 0 0 7$ </td><td> $0 . 2 4 0 \pm 0 . 0 1 3$ </td><td> $0 . 2 4 0 \pm 0 . 0 1 4$ </td><td></td></tr><tr><td>90%</td><td> $\mathbf { 0 . 2 2 8 \pm 0 . 0 1 3 }$ </td><td> $0 . 3 4 8 \pm 0 . 0 0 4$ </td><td> $0 . 2 3 6 \pm 0 . 0 1 4$ </td><td> $0 . 2 4 5 \pm 0 . 0 1 4$ </td><td> $0 . 2 4 2 \pm 0 . 0 1 1$ </td><td></td></tr></table>

Table 36: Downstream predictive $R ^ { 2 }$ degradation across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\operatorname { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 0 2 1 \pm 0 . 0 0 8$ </td><td> $0 . 0 2 6 \pm 0 . 0 2 4$ </td><td> $0 . 0 6 8 \pm 0 . 0 2 2$ </td><td> $0 . 0 5 7 \pm 0 . 0 2 3$ </td><td> $0 . 2 0 6 \pm 0 . 0 4 0$ </td><td> $0 . 2 1 5 \pm 0 . 0 4 2$ </td></tr><tr><td>20%</td><td> $0 . 0 3 1 \pm 0 . 0 1 5$ </td><td> $0 . 0 6 6 \pm 0 . 0 1 9$ </td><td> $0 . 1 4 3 \pm 0 . 0 1 9$ </td><td> $0 . 1 2 9 \pm 0 . 0 2 2$ </td><td> $0 . 3 1 8 \pm 0 . 0 5 0$ </td><td> $0 . 3 2 0 \pm 0 . 0 5 8$ </td></tr><tr><td>30%</td><td> $0 . 0 2 8 \pm 0 . 0 1 9$ </td><td> $0 . 1 4 4 \pm 0 . 0 3 4$ </td><td> $0 . 2 2 5 \pm 0 . 0 3 0$ </td><td> $0 . 2 1 0 \pm 0 . 0 3 9$ </td><td> $0 . 3 7 4 \pm 0 . 0 5 5$ </td><td> $0 . 3 5 3 \pm 0 . 0 6 2$ </td></tr><tr><td>40%</td><td> $0 . 0 6 0 \pm 0 . 0 1 0$ </td><td> $0 . 2 2 8 \pm 0 . 0 6 8$ </td><td> $0 . 2 7 5 \pm 0 . 0 5 1$ </td><td> $0 . 2 6 4 \pm 0 . 0 5 2$ </td><td> $0 . 3 8 5 \pm 0 . 0 4 7$ </td><td> $0 . 5 1 3 \pm 0 . 0 8 1$ </td></tr><tr><td>50%</td><td> $0 . 0 7 0 \pm 0 . 0 3 1$ </td><td> $0 . 3 2 3 \pm 0 . 0 5 0$ </td><td> $0 . 3 4 1 \pm 0 . 0 3 6$ </td><td> $0 . 3 5 4 \pm 0 . 0 5 6$ </td><td> $0 . 3 9 4 \pm 0 . 0 4 4$ </td><td> $1 . 2 6 4 \pm 0 . 1 3 8$ </td></tr><tr><td>60%</td><td> $\mathbf { 0 . 0 5 7 \pm 0 . 0 2 7 }$ </td><td> $0 . 4 5 3 \pm 0 . 0 7 5$ </td><td> $0 . 4 2 1 \pm 0 . 0 4 5$ </td><td> $0 . 4 6 9 \pm 0 . 0 4 9$ </td><td> $0 . 4 1 8 \pm 0 . 0 5 9$ </td><td> $1 . 3 3 9 \pm 0 . 1 5 6$ </td></tr><tr><td>70%</td><td> $\mathbf { 0 . 0 4 9 \pm 0 . 0 1 8 }$ </td><td> $0 . 4 3 1 \pm 0 . 0 5 5$ </td><td> $0 . 4 2 8 \pm 0 . 0 4 4$ </td><td> $0 . 6 5 1 \pm 0 . 1 6 9$ </td><td> $0 . 3 9 8 \pm 0 . 0 3 1$ </td><td> $1 . 2 2 0 \pm 0 . 1 1 6$ </td></tr><tr><td>80%</td><td> $0 . 0 7 1 \pm 0 . 0 5 7$ </td><td> $0 . 3 9 4 \pm 0 . 0 3 6$ </td><td> $0 . 4 2 8 \pm 0 . 0 4 0$ </td><td> $0 . 5 9 6 \pm 0 . 2 3 6$ </td><td> $0 . 3 7 1 \pm 0 . 0 5 8$ </td><td> $1 . 1 6 5 \pm 0 . 1 5 0$ </td></tr><tr><td>90%</td><td> $0 . 1 3 8 \pm 0 . 1 9 1$ </td><td> $0 . 3 8 3 \pm 0 . 0 7 3$ </td><td> $0 . 4 1 4 \pm 0 . 0 4 5$ </td><td> $0 . 6 1 6 \pm 0 . 3 2 4$ </td><td> $0 . 3 7 6 \pm 0 . 0 5 0$ </td><td> $1 . 2 2 5 \pm 0 . 1 3 7$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td> $\mathrm { S o f t I m p u t e }$ </td><td> $\mathrm { S o f t F o r e s t }$ </td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $0 . 0 1 9 \pm 0 . 0 0 9$ </td><td> $0 . 1 7 6 \pm 0 . 0 3 6$ </td><td> $0 . 0 1 7 \pm 0 . 0 0 8$ </td><td> $0 . 0 3 5 \pm 0 . 0 1 9$ </td><td> $\mathbf { 0 . 0 1 3 \pm 0 . 0 0 7 }$ </td><td></td></tr><tr><td>20%</td><td> $0 . 0 2 3 \pm 0 . 0 1 4$ </td><td> $0 . 2 3 6 \pm 0 . 0 5 0$ </td><td> $0 . 0 2 0 \pm 0 . 0 1 2$ </td><td> $0 . 0 7 7 \pm 0 . 0 2 6$ </td><td> $\mathbf { 0 . 0 1 3 \pm 0 . 0 0 6 }$ </td><td></td></tr><tr><td>30%</td><td> $0 . 0 2 5 \pm 0 . 0 1 2$ </td><td> $0 . 2 1 0 \pm 0 . 0 4 7$ </td><td> $\mathbf { 0 . 0 2 1 \pm 0 . 0 1 6 }$ </td><td> $0 . 1 2 1 \pm 0 . 0 2 9$ </td><td> $0 . 0 4 0 \pm 0 . 0 2 4$ </td><td></td></tr><tr><td>40%</td><td> $0 . 0 2 0 \pm 0 . 0 2 4$ </td><td> $0 . 1 7 7 \pm 0 . 0 4 3$ </td><td> $\mathbf { 0 . 0 1 6 \pm 0 . 0 1 1 }$ </td><td> $0 . 1 4 2 \pm 0 . 0 4 3$ </td><td> $0 . 0 5 9 \pm 0 . 0 3 0$ </td><td></td></tr><tr><td>50%</td><td> $\mathbf { 0 . 0 3 9 \pm 0 . 0 2 7 }$ </td><td> $0 . 1 3 5 \pm 0 . 0 4 6$ </td><td> $0 . 0 4 4 \pm 0 . 0 3 9$ </td><td> $0 . 1 5 8 \pm 0 . 0 5 3$ </td><td> $0 . 0 8 4 \pm 0 . 0 4 0$ </td><td></td></tr><tr><td>60%</td><td> $0 . 0 7 2 \pm 0 . 0 3 3$ </td><td> $0 . 1 1 5 \pm 0 . 0 5 2$ </td><td> $0 . 1 1 6 \pm 0 . 0 4 7$ </td><td> $0 . 2 0 3 \pm 0 . 0 3 1$ </td><td> $0 . 1 6 0 \pm 0 . 0 5 4$ </td><td></td></tr><tr><td>70%</td><td> $0 . 0 5 6 \pm 0 . 0 4 0$ </td><td> $0 . 0 8 1 \pm 0 . 0 3 7$ </td><td> $0 . 1 3 8 \pm 0 . 0 6 5$ </td><td> $0 . 1 9 7 \pm 0 . 0 7 0$ </td><td> $0 . 2 7 3 \pm 0 . 0 9 7$ </td><td></td></tr><tr><td>80%</td><td> $0 . 0 7 0 \pm 0 . 0 6 1$ </td><td> $\mathbf { 0 . 0 5 0 \pm 0 . 0 4 5 }$ </td><td> $0 . 1 9 9 \pm 0 . 0 9 5$ </td><td> $0 . 2 1 1 \pm 0 . 0 5 8$ </td><td> $0 . 3 7 9 \pm 0 . 2 0 5$ </td><td></td></tr><tr><td>90%</td><td> $0 . 0 8 9 \pm 0 . 0 5 8$ </td><td> $\mathbf { 0 . 0 4 6 \pm 0 . 0 4 4 }$ </td><td> $0 . 2 9 3 \pm 0 . 1 1 7$ </td><td> $0 . 1 7 9 \pm 0 . 0 5 8$ </td><td> $0 . 3 2 6 \pm 0 . 1 4 2$ </td><td></td></tr></table>

Table 37: PCA Procrustes distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 0 2 2 \pm 0 . 0 0 4$ </td><td> $0 . 0 3 8 \pm 0 . 0 0 4$ </td><td> $0 . 0 3 8 \pm 0 . 0 0 3$ </td><td> $0 . 0 4 9 \pm 0 . 0 0 5$ </td><td> $0 . 0 9 1 \pm 0 . 0 0 6$ </td><td> $0 . 0 8 1 \pm 0 . 0 0 5$ </td></tr><tr><td>20%</td><td> $0 . 0 7 0 \pm 0 . 0 0 9$ </td><td> $0 . 0 8 5 \pm 0 . 0 0 6$ </td><td> $0 . 0 8 6 \pm 0 . 0 0 4$ </td><td> $0 . 1 0 9 \pm 0 . 0 0 4$ </td><td> $0 . 1 7 9 \pm 0 . 0 1 0$ </td><td> $0 . 1 8 6 \pm 0 . 0 1 0$ </td></tr><tr><td>30%</td><td> $0 . 1 3 1 \pm 0 . 0 0 8$ </td><td> $0 . 1 3 9 \pm 0 . 0 0 5$ </td><td> $0 . 1 4 1 \pm 0 . 0 0 5$ </td><td> $0 . 1 6 2 \pm 0 . 0 0 6$ </td><td> $0 . 2 4 3 \pm 0 . 0 1 1$ </td><td> $0 . 2 8 4 \pm 0 . 0 1 8$ </td></tr><tr><td>40%</td><td> $0 . 2 1 3 \pm 0 . 0 2 9$ </td><td> $0 . 2 2 4 \pm 0 . 0 1 5$ </td><td> $0 . 2 1 4 \pm 0 . 0 1 0$ </td><td> $0 . 2 3 3 \pm 0 . 0 1 1$ </td><td> $0 . 3 1 0 \pm 0 . 0 0 9$ </td><td> $0 . 2 8 9 \pm 0 . 0 1 2$ </td></tr><tr><td>50%</td><td> $0 . 3 1 2 \pm 0 . 0 2 1$ </td><td> $0 . 3 4 5 \pm 0 . 0 1 2$ </td><td> $0 . 3 0 7 \pm 0 . 0 1 1$ </td><td> $0 . 3 2 0 \pm 0 . 0 1 0$ </td><td> $0 . 3 6 5 \pm 0 . 0 1 5$ </td><td> $0 . 4 5 8 \pm 0 . 0 2 1$ </td></tr><tr><td>60%</td><td> $0 . 4 0 4 \pm 0 . 0 2 7$ </td><td> $0 . 5 1 6 \pm 0 . 0 2 1$ </td><td> $0 . 4 2 1 \pm 0 . 0 1 8$ </td><td> $0 . 4 0 0 \pm 0 . 0 1 0$ </td><td> $0 . 4 1 0 \pm 0 . 0 0 9$ </td><td> $0 . 4 9 4 \pm 0 . 0 2 4$ </td></tr><tr><td>70%</td><td> $0 . 5 5 2 \pm 0 . 1 0 9$ </td><td> $0 . 6 7 2 \pm 0 . 0 2 2$ </td><td> $0 . 5 6 8 \pm 0 . 0 1 2$ </td><td> $0 . 5 1 6 \pm 0 . 0 2 7$ </td><td> $0 . 4 6 8 \pm 0 . 0 1 2$ </td><td> $0 . 5 2 1 \pm 0 . 0 2 3$ </td></tr><tr><td>80%</td><td> $0 . 5 9 5 \pm 0 . 0 5 8$ </td><td> $0 . 7 2 8 \pm 0 . 0 1 8$ </td><td> $0 . 6 4 6 \pm 0 . 0 1 0$ </td><td> $0 . 6 1 9 \pm 0 . 0 3 9$ </td><td> $0 . 5 0 9 \pm 0 . 0 1 2$ </td><td> $0 . 5 5 3 \pm 0 . 0 2 0$ </td></tr><tr><td>90%</td><td> $0 . 6 2 0 \pm 0 . 0 5 5$ </td><td> $0 . 7 5 0 \pm 0 . 0 1 4$ </td><td> $0 . 6 9 7 \pm 0 . 0 0 6$ </td><td> $0 . 6 6 6 \pm 0 . 0 2 6$ </td><td> $0 . 5 5 6 \pm 0 . 0 1 6$ </td><td> $0 . 5 9 5 \pm 0 . 0 2 4$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $\mathbf { 0 . 0 1 8 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 7 8 \pm 0 . 0 0 7$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 2$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 2$ </td><td> $0 . 0 1 8 \pm 0 . 0 0 2$ </td><td></td></tr><tr><td>20%</td><td> $0 . 0 4 7 \pm 0 . 0 0 3$ </td><td> $0 . 1 9 1 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 0 4 7 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 8 9 \pm 0 . 0 0 4$ </td><td> $0 . 0 4 8 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>30%</td><td> $0 . 0 9 3 \pm 0 . 0 0 4$ </td><td> $0 . 2 9 8 \pm 0 . 0 1 9$ </td><td> $0 . 0 9 5 \pm 0 . 0 0 5$ </td><td> $0 . 1 4 1 \pm 0 . 0 0 5$ </td><td></td><td></td></tr><tr><td>40%</td><td> $0 . 1 5 9 \pm 0 . 0 1 3$ </td><td> $0 . 3 8 8 \pm 0 . 0 1 6$ </td><td> $\mathbf { 0 . 1 4 3 \pm 0 . 0 1 2 }$ </td><td> $0 . 2 1 3 \pm 0 . 0 0 8$ </td><td>0.092 ± 0.003  $0 . 1 5 0 \pm 0 . 0 0 9$ </td><td></td></tr><tr><td>50%</td><td> $0 . 2 4 7 \pm 0 . 0 1 3$ </td><td> $0 . 4 5 9 \pm 0 . 0 2 0$ </td><td> $\mathbf { 0 . 2 2 5 \pm 0 . 0 1 4 }$ </td><td> $0 . 3 0 6 \pm 0 . 0 1 1$ </td><td> $0 . 2 3 3 \pm 0 . 0 1 2$ </td><td></td></tr><tr><td>60%</td><td> $0 . 3 1 4 \pm 0 . 0 1 1$ </td><td> $0 . 5 0 9 \pm 0 . 0 1 5$ </td><td> $\mathbf { 0 . 2 9 2 \pm 0 . 0 1 4 }$ </td><td> $0 . 4 1 2 \pm 0 . 0 2 1$ </td><td> $0 . 3 3 5 \pm 0 . 0 2 3$ </td><td></td></tr><tr><td>70%</td><td></td><td></td><td> $\mathbf { 0 . 3 7 2 \pm 0 . 0 1 9 }$ </td><td> $0 . 5 5 7 \pm 0 . 0 1 8$ </td><td></td><td></td></tr><tr><td>80%</td><td> $0 . 3 7 2 \pm 0 . 0 1 4$   $0 . 4 1 6 \pm 0 . 0 1 6$ </td><td> $0 . 5 4 1 \pm 0 . 0 1 8$   $0 . 5 6 5 \pm 0 . 0 1 2$ </td><td> $\mathbf { 0 . 4 1 3 \pm 0 . 0 1 1 }$ </td><td> $0 . 6 4 2 \pm 0 . 0 1 3$ </td><td> $0 . 4 6 7 \pm 0 . 0 4 2$   $0 . 5 7 9 \pm 0 . 0 6 4$ </td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>90%</td><td> $\mathbf { 0 . 4 5 3 \pm 0 . 0 2 3 }$ </td><td> $0 . 6 0 6 \pm 0 . 0 1 8$ </td><td> $0 . 4 7 2 \pm 0 . 0 3 7$ </td><td> $0 . 6 9 4 \pm 0 . 0 0 7$ </td><td> $0 . 6 1 9 \pm 0 . 0 7 9$ </td><td></td></tr></table>

Table 38: PLS Procrustes distance across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td> $_ \mathrm { H a l f - m i n }$ </td><td> $\mathbf { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 0 2 3 \pm 0 . 0 0 4$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 3$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 3$ </td><td> $0 . 0 4 9 \pm 0 . 0 0 5$ </td><td> $0 . 1 0 1 \pm 0 . 0 0 6$ </td><td> $0 . 0 9 5 \pm 0 . 0 0 6$ </td></tr><tr><td>20%</td><td> $0 . 0 7 1 \pm 0 . 0 0 9$ </td><td> $0 . 0 8 9 \pm 0 . 0 0 7$ </td><td> $0 . 0 9 1 \pm 0 . 0 0 4$ </td><td> $0 . 1 1 3 \pm 0 . 0 0 5$ </td><td> $0 . 1 9 5 \pm 0 . 0 1 1$ </td><td> $0 . 2 0 8 \pm 0 . 0 1 2$ </td></tr><tr><td>30%</td><td> $0 . 1 3 0 \pm 0 . 0 0 8$ </td><td> $0 . 1 4 5 \pm 0 . 0 0 9$ </td><td> $0 . 1 4 7 \pm 0 . 0 0 7$ </td><td> $0 . 1 6 6 \pm 0 . 0 0 9$ </td><td> $0 . 2 6 3 \pm 0 . 0 1 8$ </td><td> $0 . 3 0 7 \pm 0 . 0 2 3$ </td></tr><tr><td>40%</td><td> $0 . 2 0 6 \pm 0 . 0 2 5$ </td><td> $0 . 2 3 5 \pm 0 . 0 1 8$ </td><td> $0 . 2 2 4 \pm 0 . 0 1 3$ </td><td> $0 . 2 3 9 \pm 0 . 0 1 6$ </td><td> $0 . 3 3 2 \pm 0 . 0 1 2$ </td><td> $0 . 3 1 8 \pm 0 . 0 1 2$ </td></tr><tr><td>50%</td><td> $0 . 3 1 3 \pm 0 . 0 2 1$ </td><td> $0 . 3 5 8 \pm 0 . 0 1 6$ </td><td> $0 . 3 1 9 \pm 0 . 0 1 3$ </td><td> $0 . 3 2 7 \pm 0 . 0 1 1$ </td><td> $0 . 3 8 5 \pm 0 . 0 1 7$ </td><td> $0 . 4 6 5 \pm 0 . 0 2 1$ </td></tr><tr><td>60%</td><td> $0 . 3 9 8 \pm 0 . 0 3 3$ </td><td> $0 . 5 3 7 \pm 0 . 0 2 4$ </td><td> $0 . 4 3 5 \pm 0 . 0 2 1$ </td><td> $0 . 4 0 9 \pm 0 . 0 1 5$ </td><td> $0 . 4 3 0 \pm 0 . 0 1 0$ </td><td> $0 . 4 9 9 \pm 0 . 0 2 3$ </td></tr><tr><td>70%</td><td> $0 . 5 5 5 \pm 0 . 1 1 1$ </td><td> $0 . 6 9 5 \pm 0 . 0 2 1$ </td><td> $0 . 5 8 2 \pm 0 . 0 1 2$ </td><td> $0 . 5 2 1 \pm 0 . 0 3 0$ </td><td>0.481 ± 0.012</td><td> $0 . 5 2 1 \pm 0 . 0 1 8$ </td></tr><tr><td>80%</td><td> $0 . 5 8 5 \pm 0 . 0 7 1$ </td><td> $0 . 7 4 9 \pm 0 . 0 2 2$ </td><td> $0 . 6 5 8 \pm 0 . 0 1 1$ </td><td> $0 . 6 2 3 \pm 0 . 0 4 1$ </td><td> $0 . 5 1 9 \pm 0 . 0 1 2$ </td><td> $0 . 5 5 1 \pm 0 . 0 2 5$ </td></tr><tr><td>90%</td><td> $0 . 6 1 2 \pm 0 . 0 6 3$ </td><td> $0 . 7 6 5 \pm 0 . 0 1 3$ </td><td> $0 . 7 0 8 \pm 0 . 0 0 6$ </td><td> $0 . 6 6 8 \pm 0 . 0 2 8$ </td><td> $0 . 5 6 5 \pm 0 . 0 1 4$ </td><td> $0 . 5 8 8 \pm 0 . 0 2 7$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td> $\mathrm { S o f t I m p u t e }$ </td><td> $\mathrm { S o f t F o r e s t }$ </td><td> $\operatorname { s v r }$ </td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $\mathbf { 0 . 0 1 9 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 9 2 \pm 0 . 0 0 8$ </td><td> $0 . 0 1 9 \pm 0 . 0 0 2$ </td><td> $0 . 0 4 1 \pm 0 . 0 0 3$ </td><td> $0 . 0 2 0 \pm 0 . 0 0 3$ </td><td></td></tr><tr><td>20%</td><td> $0 . 0 5 1 \pm 0 . 0 0 3$ </td><td> $0 . 2 1 5 \pm 0 . 0 1 3$ </td><td> $\mathbf { 0 . 0 5 1 \pm 0 . 0 0 3 }$ </td><td> $0 . 0 9 4 \pm 0 . 0 0 4$ </td><td> $0 . 0 5 1 \pm 0 . 0 0 4$ </td><td></td></tr><tr><td>30%</td><td> $\mathbf { 0 . 0 9 7 \pm 0 . 0 0 6 }$ </td><td> $0 . 3 2 4 \pm 0 . 0 2 6$ </td><td> $0 . 0 9 9 \pm 0 . 0 0 7$ </td><td> $0 . 1 4 8 \pm 0 . 0 0 7$ </td><td> $0 . 0 9 7 \pm 0 . 0 0 6$ </td><td></td></tr><tr><td>40%</td><td> $0 . 1 6 5 \pm 0 . 0 1 2$ </td><td> $0 . 4 1 2 \pm 0 . 0 2 1$ </td><td> $\mathbf { 0 . 1 4 9 \pm 0 . 0 1 1 }$ </td><td> $0 . 2 2 4 \pm 0 . 0 1 0$ </td><td> $0 . 1 5 6 \pm 0 . 0 1 0$ </td><td></td></tr><tr><td>50%</td><td> $0 . 2 5 4 \pm 0 . 0 1 4$ </td><td> $0 . 4 7 7 \pm 0 . 0 2 1$ </td><td> $\mathbf { 0 . 2 3 0 \pm 0 . 0 1 8 }$ </td><td> $0 . 3 1 7 \pm 0 . 0 1 2$ </td><td> $0 . 2 3 9 \pm 0 . 0 1 5$ </td><td></td></tr><tr><td>60%</td><td> $0 . 3 1 9 \pm 0 . 0 1 3$ </td><td> $0 . 5 2 6 \pm 0 . 0 1 7$ </td><td> $\mathbf { 0 . 2 9 8 \pm 0 . 0 1 6 }$ </td><td> $0 . 4 2 6 \pm 0 . 0 2 4$ </td><td> $0 . 3 4 1 \pm 0 . 0 2 5$ </td><td></td></tr><tr><td>70%</td><td> $\mathbf { 0 . 3 7 4 \pm 0 . 0 1 6 }$ </td><td> $0 . 5 5 1 \pm 0 . 0 1 9$ </td><td> $0 . 3 7 5 \pm 0 . 0 1 8$ </td><td> $0 . 5 7 1 \pm 0 . 0 1 6$ </td><td> $0 . 4 6 7 \pm 0 . 0 4 3$ </td><td></td></tr><tr><td>80%</td><td> $\mathbf { 0 . 4 1 3 \pm 0 . 0 1 6 }$ </td><td> $0 . 5 7 2 \pm 0 . 0 1 3$ </td><td> $0 . 4 1 3 \pm 0 . 0 1 6$ </td><td> $0 . 6 5 3 \pm 0 . 0 1 4$ </td><td> $0 . 5 7 7 \pm 0 . 0 6 3$ </td><td></td></tr><tr><td>90%</td><td> $\mathbf { 0 . 4 4 1 \pm 0 . 0 1 7 }$ </td><td> $0 . 6 1 4 \pm 0 . 0 1 5$ </td><td> $0 . 4 5 9 \pm 0 . 0 2 5$ </td><td> $0 . 7 0 4 \pm 0 . 0 0 8$ </td><td> $0 . 6 1 6 \pm 0 . 0 7 4$ </td><td></td></tr></table>

Table 39: Pearson log-p correlation across missing rates
<table><tr><td>Rate</td><td>MissForest</td><td>kNN</td><td>Mean</td><td>Median</td><td>Half-min</td><td> $\operatorname { S V T \_ O G p a p e r }$ </td></tr><tr><td>10%</td><td> $0 . 9 9 4 \pm 0 . 0 0 3$ </td><td> $0 . 9 9 5 \pm 0 . 0 0 4$ </td><td> $0 . 9 6 8 \pm 0 . 0 1 0$ </td><td> $0 . 9 8 7 \pm 0 . 0 0 6$ </td><td> $0 . 8 8 6 \pm 0 . 0 2 1$ </td><td> $0 . 8 0 8 \pm 0 . 0 2 6$ </td></tr><tr><td>20%</td><td> $0 . 9 7 2 \pm 0 . 0 1 1$ </td><td> $\mathbf { 0 . 9 8 8 \pm 0 . 0 0 5 }$ </td><td> $0 . 9 1 8 \pm 0 . 0 2 5$ </td><td> $0 . 9 5 2 \pm 0 . 0 1 7$ </td><td> $0 . 8 0 8 \pm 0 . 0 3 4$ </td><td> $0 . 6 8 2 \pm 0 . 0 4 7$ </td></tr><tr><td>30%</td><td> $0 . 9 4 0 \pm 0 . 0 2 3$ </td><td> $\mathbf { 0 . 9 7 0 \pm 0 . 0 1 5 }$ </td><td> $0 . 8 8 9 \pm 0 . 0 2 3$ </td><td> $0 . 9 2 1 \pm 0 . 0 2 2$ </td><td> $0 . 6 9 8 \pm 0 . 0 2 8$ </td><td> $0 . 5 5 1 \pm 0 . 0 4 2$ </td></tr><tr><td>40%</td><td> $0 . 8 7 9 \pm 0 . 0 2 1$ </td><td> $\mathbf { 0 . 9 3 1 \pm 0 . 0 2 0 }$ </td><td> $0 . 8 3 2 \pm 0 . 0 2 7$ </td><td> $0 . 8 5 4 \pm 0 . 0 2 9$ </td><td> $0 . 6 3 8 \pm 0 . 0 4 2$ </td><td> $0 . 4 8 8 \pm 0 . 0 4 5$ </td></tr><tr><td>50%</td><td> $0 . 8 1 0 \pm 0 . 0 2 1$ </td><td> $0 . 8 2 7 \pm 0 . 0 3 7$ </td><td> $0 . 7 8 3 \pm 0 . 0 4 2$ </td><td> $0 . 7 6 7 \pm 0 . 0 4 1$ </td><td> $0 . 5 3 3 \pm 0 . 0 4 1$ </td><td> $0 . 6 2 9 \pm 0 . 0 4 5$ </td></tr><tr><td>60%</td><td> $0 . 7 5 1 \pm 0 . 0 4 0$ </td><td> $0 . 7 0 7 \pm 0 . 0 6 3$ </td><td> $0 . 7 3 9 \pm 0 . 0 3 6$ </td><td> $0 . 6 7 4 \pm 0 . 0 4 6$ </td><td> $0 . 4 7 3 \pm 0 . 0 5 6$ </td><td> $0 . 5 4 1 \pm 0 . 0 6 4$ </td></tr><tr><td>70%</td><td> $0 . 5 9 3 \pm 0 . 1 4 4$ </td><td> $0 . 5 8 9 \pm 0 . 0 8 3$ </td><td> $0 . 6 4 8 \pm 0 . 0 2 3$ </td><td> $0 . 5 1 8 \pm 0 . 0 4 4$ </td><td> $0 . 4 0 3 \pm 0 . 0 5 7$ </td><td> $0 . 4 4 0 \pm 0 . 0 5 8$ </td></tr><tr><td>80%</td><td> $0 . 5 1 3 \pm 0 . 1 4 3$ </td><td> $0 . 5 6 3 \pm 0 . 0 8 0$ </td><td> $\mathbf { 0 . 6 0 6 \pm 0 . 0 4 2 }$ </td><td> $0 . 4 4 7 \pm 0 . 0 4 3$ </td><td> $0 . 3 5 6 \pm 0 . 0 2 6$ </td><td> $0 . 3 8 6 \pm 0 . 0 3 5$ </td></tr><tr><td>90%</td><td> $0 . 4 2 9 \pm 0 . 1 1 8$ </td><td> $0 . 5 7 9 \pm 0 . 0 7 8$ </td><td> $\mathbf { 0 . 6 1 3 \pm 0 . 0 2 9 }$ </td><td> $0 . 3 8 6 \pm 0 . 0 3 2$ </td><td> $0 . 3 0 2 \pm 0 . 0 2 4$ </td><td> $0 . 2 9 7 \pm 0 . 0 4 8$ </td></tr><tr><td>Rate</td><td> $\mathrm { N F \_ O G p a p e r }$ </td><td>SoftImpute</td><td>SoftForest</td><td>SVT</td><td>NuclearForest</td><td></td></tr><tr><td>10%</td><td> $\mathbf { 0 . 9 9 6 \pm 0 . 0 0 3 }$ </td><td> $0 . 7 9 7 \pm 0 . 0 3 0$ </td><td> $0 . 9 9 6 \pm 0 . 0 0 2$ </td><td> $0 . 9 6 0 \pm 0 . 0 1 0$ </td><td> $0 . 9 9 4 \pm 0 . 0 0 3$ </td><td></td></tr><tr><td>20%</td><td> $0 . 9 8 5 \pm 0 . 0 0 5$ </td><td> $0 . 6 7 4 \pm 0 . 0 4 2$ </td><td> $0 . 9 8 5 \pm 0 . 0 0 4$ </td><td> $0 . 8 8 7 \pm 0 . 0 2 7$ </td><td> $0 . 9 6 1 \pm 0 . 0 0 8$ </td><td></td></tr><tr><td>30%</td><td> $0 . 9 5 9 \pm 0 . 0 1 5$ </td><td> $0 . 5 5 4 \pm 0 . 0 3 6$ </td><td> $0 . 9 4 8 \pm 0 . 0 2 0$ </td><td> $0 . 8 2 4 \pm 0 . 0 2 2$ </td><td> $0 . 9 1 0 \pm 0 . 0 2 8$ </td><td></td></tr><tr><td>40%</td><td> $0 . 8 8 7 \pm 0 . 0 2 7$ </td><td> $0 . 4 8 7 \pm 0 . 0 3 8$ </td><td> $0 . 9 0 4 \pm 0 . 0 2 4$ </td><td> $0 . 7 2 3 \pm 0 . 0 3 2$ </td><td> $0 . 8 2 9 \pm 0 . 0 2 5$ </td><td></td></tr><tr><td>50%</td><td> $0 . 7 6 8 \pm 0 . 0 2 8$ </td><td> $0 . 4 0 3 \pm 0 . 0 2 9$ </td><td> $\mathbf { 0 . 8 4 8 \pm 0 . 0 3 0 }$ </td><td> $0 . 6 3 2 \pm 0 . 0 4 3$ </td><td> $0 . 7 4 7 \pm 0 . 0 3 3$ </td><td></td></tr><tr><td>60%</td><td> $0 . 6 8 9 \pm 0 . 0 3 1$ </td><td> $0 . 3 4 1 \pm 0 . 0 3 8$ </td><td> $\mathbf { 0 . 7 7 0 \pm 0 . 0 2 8 }$ </td><td> $0 . 5 6 9 \pm 0 . 0 3 0$ </td><td> $0 . 6 2 9 \pm 0 . 0 4 3$ </td><td></td></tr><tr><td>70%</td><td> $0 . 6 6 0 \pm 0 . 0 4 8$ </td><td> $0 . 2 8 9 \pm 0 . 0 2 7$ </td><td> $\mathbf { 0 . 6 9 9 \pm 0 . 0 2 7 }$ </td><td> $0 . 4 6 6 \pm 0 . 0 2 5$ </td><td> $0 . 4 3 6 \pm 0 . 0 5 6$ </td><td></td></tr><tr><td>80%</td><td> $0 . 5 8 3 \pm 0 . 0 5 1$ </td><td> $0 . 2 6 0 \pm 0 . 0 1 2$ </td><td> $0 . 5 9 4 \pm 0 . 0 2 2$ </td><td> $0 . 4 2 9 \pm 0 . 0 3 9$ </td><td> $0 . 2 6 7 \pm 0 . 0 5 6$ </td><td></td></tr><tr><td>90%</td><td> $0 . 5 2 4 \pm 0 . 0 4 5$ </td><td> $0 . 2 1 9 \pm 0 . 0 1 8$ </td><td> $0 . 5 1 4 \pm 0 . 0 3 0$ </td><td> $0 . 4 5 0 \pm 0 . 0 3 8$ </td><td> $0 . 2 1 4 \pm 0 . 0 8 9$ </td><td></td></tr></table>

## E.5 RUNTIME COMPARISON

Table 40: Runtime comparison of imputation methods on the metabolomics dataset under MCAR/MAR missingness. Runtime is reported in seconds.
<table><tr><td>Method</td><td>Mean (s)</td><td>Std (s)</td><td>Min (s)</td><td>Max (s)</td></tr><tr><td>MissForest</td><td>77.787</td><td>21.426</td><td>49.290</td><td>116.269</td></tr><tr><td>kNN</td><td>0.040</td><td>0.009</td><td>0.021</td><td>0.063</td></tr><tr><td>Mean</td><td>0.001</td><td>0.001</td><td>0.000</td><td>0.004</td></tr><tr><td>Median</td><td>0.002</td><td>0.001</td><td>0.001</td><td>0.006</td></tr><tr><td>Half-min</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.002</td></tr><tr><td>SVT_OGpaper</td><td>5.889</td><td>1.261</td><td>4.180</td><td>12.519</td></tr><tr><td>NF_OGpaper</td><td>13.581</td><td>2.343</td><td>9.826</td><td>20.209</td></tr><tr><td>SoftImpute</td><td>0.384</td><td>0.144</td><td>0.111</td><td>0.895</td></tr><tr><td>SoftForest</td><td>8.171</td><td>1.931</td><td>5.264</td><td>11.753</td></tr><tr><td>SVT</td><td>5.660</td><td>0.956</td><td>4.408</td><td>9.867</td></tr><tr><td>NuclearForest</td><td>13.389</td><td>2.365</td><td>9.772</td><td>21.601</td></tr></table>

Table 41: Runtime comparison of imputation methods on the metabolomics dataset under MNAR missingness. Runtime is reported in seconds.
<table><tr><td>Method</td><td>Mean (s)</td><td>Std (s)</td><td>Min (s)</td><td>Max (s)</td></tr><tr><td>MissForest</td><td>89.438</td><td>22.845</td><td>11.206</td><td>111.187</td></tr><tr><td>kNN</td><td>0.025</td><td>0.012</td><td>0.006</td><td>0.060</td></tr><tr><td>Mean</td><td>0.001</td><td>0.001</td><td>0.000</td><td>0.005</td></tr><tr><td>Median</td><td>0.002</td><td>0.001</td><td>0.001</td><td>0.005</td></tr><tr><td>Half-min</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.001</td></tr><tr><td>SVT_OGpaper</td><td>5.611</td><td>0.769</td><td>4.911</td><td>9.745</td></tr><tr><td>NF_OGpaper</td><td>9.751</td><td>2.228</td><td>5.806</td><td>13.438</td></tr><tr><td>SoftImpute</td><td>0.125</td><td>0.085</td><td>0.013</td><td>0.562</td></tr><tr><td>SoftForest</td><td>4.344</td><td>2.137</td><td>0.946</td><td>7.712</td></tr><tr><td>SVT</td><td>5.446</td><td>0.448</td><td>4.851</td><td>7.636</td></tr><tr><td>NuclearForest</td><td>9.797</td><td>2.381</td><td>5.936</td><td>16.075</td></tr></table>

Table 42: Runtime comparison of imputation methods on the housing dataset under MCAR missingness. Runtime is reported in seconds.
<table><tr><td>Method</td><td>Mean (s)</td><td>Std (s)</td><td>Min (s)</td><td>Max (s)</td></tr><tr><td>MissForest</td><td>5.934</td><td>0.187</td><td>5.516</td><td>6.454</td></tr><tr><td>kNN</td><td>0.034</td><td>0.024</td><td>0.010</td><td>0.144</td></tr><tr><td>Mean</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.004</td></tr><tr><td>Median</td><td>0.001</td><td>0.000</td><td>0.001</td><td>0.003</td></tr><tr><td>Half-min</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.003</td></tr><tr><td>SVT_OGpaper</td><td>0.119</td><td>0.096</td><td>0.002</td><td>0.285</td></tr><tr><td>NF_OGpaper</td><td>0.750</td><td>0.062</td><td>0.650</td><td>0.926</td></tr><tr><td>SoftImpute</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.001</td></tr><tr><td>SoftForest</td><td>0.664</td><td>0.045</td><td>0.573</td><td>0.845</td></tr><tr><td>SVT</td><td>0.003</td><td>0.001</td><td>0.002</td><td>0.007</td></tr><tr><td>NuclearForest</td><td>0.659</td><td>0.047</td><td>0.576</td><td>0.882</td></tr></table>

Table 43: Runtime comparison of imputation methods on the housing dataset under MAR missingness. Runtime is reported in seconds.
<table><tr><td>Method</td><td>Mean (s)</td><td>Std (s)</td><td>Min (s)</td><td>Max (s)</td></tr><tr><td>MissForest</td><td>5.973</td><td>0.198</td><td>5.719</td><td>6.754</td></tr><tr><td>kNN</td><td>0.030</td><td>0.018</td><td>0.008</td><td>0.140</td></tr><tr><td>Mean</td><td>0.001</td><td>0.001</td><td>0.000</td><td>0.007</td></tr><tr><td>Median</td><td>0.001</td><td>0.000</td><td>0.001</td><td>0.003</td></tr><tr><td>Half-min</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.001</td></tr><tr><td>SVT_OGpaper</td><td>0.120</td><td>0.096</td><td>0.002</td><td>0.305</td></tr><tr><td>NF_OGpaper</td><td>0.761</td><td>0.075</td><td>0.647</td><td>1.121</td></tr><tr><td>SoftImpute</td><td>0.001</td><td>0.000</td><td>0.000</td><td>0.002</td></tr><tr><td>SoftForest</td><td>0.661</td><td>0.039</td><td>0.601</td><td>0.820</td></tr><tr><td>SVT</td><td>0.003</td><td>0.001</td><td>0.002</td><td>0.007</td></tr><tr><td>NuclearForest</td><td>0.661</td><td>0.034</td><td>0.604</td><td>0.766</td></tr></table>