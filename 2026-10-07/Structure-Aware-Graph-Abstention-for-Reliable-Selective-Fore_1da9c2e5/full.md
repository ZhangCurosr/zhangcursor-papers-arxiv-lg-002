# Structure-Aware Graph Abstention for Reliable Selective Forecasting

Jianxiang Xie University of New South Wales

## Abstract

Selective forecasting abstains on high-risk test windows under a retained-coverage budget. Existing gates such as TEM (Brusokas et al., 2025) score each forecast as a whole; for multivariate outputs, trajectories can look plausible while violating dependencies among variables. We treat instance-level plausibility and relational consistency as distinct reliability axes and operationalize the latter via a learned sparse graph and a Dirichletstyle structural energy E<sub>struct</sub>(Y , A <sup>ˆ</sup> ), trained with error-weighted graph regularization and score–error alignment (Section 4). On seven long-horizon benchmarks and four backbones, structural gating often reduces selective MSE versus TEM at matched coverage, with the largest gains where cross-variable structure appears more informative in our benchmarks; gains are not universal, indicating a complementary abstention signal. Table 1 is a Protocol A ranking diagnostic (seed 2024); three-seed deployable Protocol B on an aligned subset is in Table 3 (full validation→test grids: Appendix A).

## 1 INTRODUCTION

Selective forecasting adds an abstention layer on top of point prediction: under a retained-coverage budget, the system withholds forecasts scored as high risk, trading how often predictions are released for lower error on those that are released. We study coverageconstrained score gating (lower abstention score ⇒ more likely accepted), using scalar scores as selectiverisk proxies rather than calibrated probabilities.

Most selective rules score each forecast as a whole. Time–energy models (TEM) (Brusokas et al., 2025)— our primary comparator, not our contribution—train an energy-based model on decoder trajectories after a deep forecaster and abstain on high instance energy. This is insuficient for multivariate outputs: each chan

Belal Alsinglawi Zayed University

nel can look plausible while the joint future violates dependencies seen in training (Wu et al., 2020; Chen et al., 2022; Benidis et al., 2022). Selective forecasting for structured outputs therefore faces two questions: is the forecast plausible, and is it relationally consistent across variables? Instancelevel plausibility and relational consistency are distinct notions of reliability; TEM targets the former, while we explicitly score the latter.

We call violation of learned cross-variable structure structural deviation (high structural energy; Section 4). Coupled load and weather panels illustrate the failure mode: co-evolving channels can drift out of sync on the horizon while marginal traces stay benign. Relational gating is most informative when cross-channel coupling is stable (Electricity, Weather, Trafic) and less so when instance error dominates or coupling is weak (Section 5).

We present SASF (structure-aware selective forecasting): a lightweight adaptive graph head yields adjacency A and structural energy $E _ { \mathrm { s t r u c t } } ( \hat { Y } , A )$ ; auxiliary losses make $E _ { \mathrm { s t r u c t } }$ rank selective risk (Section 4). Prior graph work mostly improves Y<sup>ˆ</sup> itself; we use dependencies as a decision signal for abstention (Section 2). We compare graph-only $E _ { \mathrm { s t r u c t } }$ to instancelevel $E _ { \mathrm { T E M } }$ under matched coverage (Section 5; deployable validation→test thresholds in Appendix A).

## Contributions.

• We identify a failure mode of instance-level selective forecasting: forecasts can be individually plausible yet structurally deviant across variables.

• We operationalize relational consistency through learned A and $E _ { \mathrm { s t r u c t } }$ , with error-weighted graph learning and score–error alignment so structural deviation predicts selective risk.

• Empirically, relational gating helps most where cross-variable dependence is more informative for abstention and on selected backbones, while instance-level scores remain competitive elsewhere—relational abstention complements rather than replaces TEM (Tables 1–2; multiseed deployable Protocol B in Table 3 and Appendix A).

Scope. We evaluate on public forecasting benchmarks, not deployed cyber-physical testbeds, and do not claim formal safety verification, conformal guarantees, or anomaly-detection certification.

## 2 RELATED WORK

Strong multivariate forecasters span decomposition Transformers and modern attention designs (Wu et al., 2021; Zhou et al., 2022; Nie et al., 2022; Wu et al., 2022; Zhou et al., 2021; Zeng et al., 2023), typically optimized for average accuracy (Benidis et al., 2022). Selective prediction studies coverage–risk tradeofs via abstention (Geifman and El-Yaniv, 2017; Cortes et al., 2016); conformal and ensemble methods target uncertainty for predictive sets or calibrated sets (Vovk et al., 2005; Romano et al., 2019; Angelopoulos Anastasios and Stephen, 2021; Gal and Ghahramani, 2016; Laksh minarayanan et al., 2017; Gneiting and Raftery, 2007). We instead compare learned scalar abstention scores under explicit selective protocols (Section 5) and adopt the instance-energy TEM pipeline (Brusokas et al., 2025; LeCun et al., 2006; Hyvärinen and Dayan, 2005) as baseline.

Prior work thus falls into two families: (i) instancelevel selective scores over whole forecasts, and (ii) graph structure used to improve point forecasts (Wu et al., 2020; Liu et al., 2022; Chen et al., 2022, 2023; Cai et al., 2024; Kim et al., 2023; Sriramulu et al., 2023; Zhang et al., 2024; Yu et al., 2017; Kipf and Welling, 2017). We use learned dependencies to decide whether a forecast should be released— the graph is a relational abstention signal, not solely a predictive inductive bias. Section 4 instantiates this under retained-coverage constraints.

## 3 PROBLEM FORMULATION

Selective forecasting is a risk-aware decision layer on top of point forecasting: after producing $\hat { Y }$ , the system decides whether to release a forecast or abstain, trading retained coverage (how often predictions are issued) against error on released predictions. This objective is complementary to fitting $f _ { \boldsymbol { \theta } } { : }$ we choose an acceptance rule $g$ that minimizes post-abstention risk— average loss on retained samples—subject to a nominal retained-coverage (equivalently, abstention-rate) constraint.

Let $\boldsymbol { X } \in \mathbb { R } ^ { B \times L \times D }$ denote a batch of historical multivariate sequences, where $B$ is the batch size, L is the look-back window, and D is the number of variables. The forecasting task is to predict future trajectories

$$
Y \in \mathbb { R } ^ { B \times H \times D } ,
$$

with prediction horizon H. A forecasting model $f _ { \theta }$ produces

$$
\hat { Y } = f _ { \theta } ( X ) \in \mathbb { R } ^ { B \times H \times D } .
$$

Given ground-truth $Y _ { i \textrm { \scriptsize { F } } i }$ , we define the per-sample squared error that underlies selective risk:

$$
e _ { b } = \frac { 1 } { H D } \sum _ { h = 1 } ^ { H } \sum _ { d = 1 } ^ { D } ( \hat { Y } _ { b , h , d } - Y _ { b , h , d } ) ^ { 2 } .\tag{1}
$$

An acceptance function (selective decision rule)

$$
g ( X _ { b } , { \hat { Y } } _ { b } ) \in \{ 0 , 1 \}
$$

determines whether a prediction is retained $( g = 1 )$ or rejected $( g \ : = \ : 0 )$ . Let $\mathcal { S } = \{ b : g ( X _ { b } , \hat { Y } _ { b } ) = 1 \}$ denote the set of accepted samples. Empirical retained coverage and selective risk are

$$
\mathrm { c o v } = \frac { | \cal S | } { B } , \qquad \mathrm { r i s k } = \frac { 1 } { | { \cal S } | } \sum _ { b \in { \cal S } } e _ { b } \quad ( \mathrm { w h e n } | { \cal S } | > 0 ) .\tag{2}
$$

Given a target coverage level $c \in ( 0 , 1 ]$ , we seek a rule $g$ that minimizes selective risk under an approximate retained-coverage constraint:

$$
\operatorname* { m i n } _ { g } { \mathrm { ~ } } \operatorname { r i s k } _ { \mathrm { ~ } } \mathrm { ~ s . t . ~ } \ \operatorname { c o v } \approx c .\tag{3}
$$

We instantiate cov ≈ c via the ranking and transfer protocols in Section 5 (target retained fraction $\rho ,$ with $c = \rho )$ . The central question is not how to improve $f _ { \theta } ,$ but how to score multivariate reliability when instancelevel plausibility can miss structural deviation. We compare (i) TEM energy $E _ { \mathrm { T E M } }$ (instance-level, prior pipeline) and (ii) structural energy $E _ { \mathrm { s t r u c t } }$ (relational, Section 4).

## 4 METHOD

## 4.1 Overview

SASF augments backbone $f _ { \theta }$ with (1) our graph head $( A , E _ { \mathrm { s t r u c t } } )$ and (2) the prior TEM branch $( E _ { \mathrm { T E M } } .$ for comparison). Selective decisions are made from the abstention scores according to the evaluation protocols in Section 5. The graph head is trained jointly with $f _ { \theta }$

## 4.2 Why relational energy?

Instance-level gates can accept trajectories that look plausible in isolation yet break relational consistency: $E _ { \mathrm { T E M } }$ can be small while predicted co-evolution violates nominal dependencies in $A . \quad E _ { \mathrm { s t r u c t } }$ scores such structural deviation for abstention, complementing $E _ { \mathrm { T E M } }$ Section 5.4 visualizes relational misses (E<sub>TEM</sub> accepts, $E _ { \mathrm { s t r u c t } }$ rejects).

![](images/3975ced3d98051de23500967a5dd6b199e593629d59ece758eb1e881f60750ed.jpg)  
Figure 1: Overview: backbone ${ \hat { Y } } ;$ baseline TEM energy $E _ { \mathrm { T E M } }$ (prior pipeline); our graph head yields A and $E _ { \mathrm { s t r u c t } } ( \hat { Y } , A )$ . Under matched test-split quantile gating, we obtain comparable selective evaluations for graphonly vs. TEM-only gates.

## 4.3 Adaptive sparse graph head

Learned A encodes which variables co-evolve under the training distribution. The parameterization is intentionally lightweight: we use a simple global graph not to maximize graph expressivity, but to isolate whether relational structure itself can serve as an abstention signal (rather than to advance graph-based forecasting). We use learnable node embeddings $U \in \mathbb { R } ^ { D \times d _ { g } }$ afinities $M = \mathrm { R e L U } ( U U ^ { \top } )$ , and row-wise top-k sparsification when $D { > } 1 0 \ ( k { = } 1 0 )$ , then row softmax

$$
A = \mathrm { s o f t m a x } _ { \mathrm { r o w } } ( M _ { \mathrm { s p a r s e } } ) .\tag{4}
$$

For small $D ,$ we omit sparsification to avoid over pruning. If no adaptive head is trained, one may instead build A from training-set channel correlations (absolute Pearson), with the same top-k and row normalization—used as a fallback in analysis scripts.

## 4.4 Dirichlet-style structural energy

If variables i and $j$ are strongly connected in A, their predicted trajectories should co-evolve; we penalize large horizon disagreement on weighted edges (Dirichlet-style graph energy; Appendix E.2). Given forecast batch $\tilde { { Y } } \in \tilde { \mathbb { R } } ^ { B \times H \times D }$ , fix $b \in \{ 1 , \ldots , B \}$ and form pairwise squared gaps

$$
\mathrm { d i s t } _ { b , i , j } = \sum _ { h = 1 } ^ { H } \left( \hat { Y } _ { b , h , i } - \hat { Y } _ { b , h , j } \right) ^ { 2 } ,\tag{5}
$$

and define the per-example structural energy (a scalar abstention score for sample b) as this graph smooth ness functional,

$$
E _ { \mathrm { s t r u c t } } ( \hat { Y } _ { b } , A ) = \frac { 1 } { D ^ { 2 } } \sum _ { i = 1 } ^ { D } \sum _ { j = 1 } ^ { D } A _ { i j } \mathrm { d i s t } _ { b , i , j } .\tag{6}
$$

Large $E _ { \mathrm { s t r u c t } } ( \hat { Y } _ { b } , A )$ indicates structural deviation from relational consistency under A. At test time we sort $\{ E _ { \mathrm { s t r u c t } } ( \hat { Y } _ { b } , A ) \} _ { b = 1 } ^ { B }$ (written $E _ { \mathrm { s t r u c t } } ( \hat { Y } , A )$ when batchindex dependence is clear). A is a single global unsigned graph, so the score captures smoothness-based deviation rather than lags or sample-specific structure.

Theoretical interpretation. Although A may be asymmetric after row-wise sparsification and normalization, the pairwise energy in Eq. (6) depends only on the symmetrized weights $W = ( A { + } A ^ { \top } ) / 2$ (Lemma 1, Appendix E.1). If $E _ { \mathrm { s t r u c t } } ( Y _ { b } , A ) \leq \epsilon$ and $\lambda _ { \operatorname* { m a x } } ( L _ { W } ) >$ 0, Theorem 2 implies a one-sided certificate on the same per-sample selective error $e _ { b } \ ( \mathrm { E q . \ ( 1 ) } )$

$$
e _ { b } \geq \frac { D } { 2 H \lambda _ { \operatorname* { m a x } } ( L _ { W } ) } \Big [ \sqrt { E _ { \mathrm { s t r u c t } } ( \hat { Y } _ { b } , A ) } - \sqrt { \epsilon } \Big ] _ { + } ^ { 2 } ,\tag{7}
$$

Thus, under graph-smooth ground truth, large $E _ { \mathrm { s t r u c t } } ( \hat { Y } _ { b } , A )$ certifies non-negligible $e _ { b }$ (not an exact MSE predictor). Appendix E links score ordering to selective risk for a fixed acceptance set.

## 4.5 Training objectives

The auxiliary losses play complementary roles: $\mathcal { L } _ { \mathrm { g r a p h } }$ learns what relational consistency looks like on ground truth futures, whereas $\mathcal { L } _ { \mathrm { a l i g n } }$ teaches $E _ { \mathrm { s t r u c t } } ( \hat { Y } , A )$ when structural deviation predicts forecast error.

Forecasting loss.

$$
\mathcal { L } _ { \mathrm { f o r e c a s t } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } e _ { b } .\tag{8}
$$

Error-weighted graph regularization. Let $Y$ be the ground-truth future and $\hat { Y }$ the model prediction. For each batch sample $b ,$ let $e _ { b }$ be the per-sample MSE from Eq. (1) and $\boldsymbol { \bar { e } } = \frac { 1 } { B } \sum _ { b } \boldsymbol { e } _ { b }$ . We use normalized weights $w _ { b } = \mathrm { c l i p } ( e _ { b } / \bar { e } , 0 . 1 , 1 0 )$ . With learned $A ,$ we compute structural energy on $Y$ (same formula as Eq. (6) with $Y$ in place of $\hat { Y } )$ and define

$$
\mathcal { L } _ { \mathrm { g r a p h } } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } w _ { b } E _ { \mathrm { s t r u c t } } ( Y _ { b } , A ) .\tag{9}
$$

Weighting by $w _ { b }$ emphasizes structural patterns in difficult forecasting regions, so A is shaped more strongly where selective decisions matter. $\mathcal { L } _ { \mathrm { g r a p h } }$ is computed on $Y \ ( \mathrm { n o t } \ { \hat { Y } } )$ to learn nominal dependencies.

Alignment loss. To encourage $E _ { \mathrm { s t r u c t } } ( \hat { Y } , A )$ to preserve the risk ordering used in selective prediction, we optimize a diferentiable score–risk association surrogate: batch Pearson correlation between $s _ { b } =$ $E _ { \mathrm { s t r u c t } } ( \hat { Y } _ { b } , A )$ and targets $~ t _ { b } ~ = ~ \log ( 1 { + } e _ { b } )$ (or $t _ { b } \ =$ $\boldsymbol { e } _ { b } )$ . Concretely, $\mathcal { L } _ { \mathrm { a l i g n } } = - \operatorname { C o r r } ( s , t ) \ ( \mathrm { E q . ~ } ( 1 1 )$ $\mathrm { A p - }$ pendix E.4); this does not directly optimize permutation agreement, but is consistent with oracle ranking under approximate score–risk alignment (Proposition 6, Appendix E.4). The joint objective is

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { \mathrm { f o r e c a s t } } + \lambda _ { g } { \mathcal { L } } _ { \mathrm { g r a p h } } + \lambda _ { a } { \mathcal { L } } _ { \mathrm { a l i g n } } ,\tag{10}
$$

with hyperparameters $\lambda _ { g } , \lambda _ { a }$ (defaults in our implementation: $\lambda _ { g } = 0 . 1 , \ \lambda _ { a } = 0 . 0 5 )$ . Appendix E explains when $E _ { \mathrm { s t r u c t } }$ can track risk under graph smoothness; it does not imply universally optimal selective ranking.

## 4.6 TEM baseline (prior work)

We instantiate TEM following Brusokas et al. (2025): an EBM on decoder trajectories trained jointly with $f _ { \theta }$ and the graph head (same splits and early stop ping). At inference, $E _ { \mathrm { T E M } }$ scores each forecast; graph and TEM gates use raw energies under identical rules (Section 5).

## 5 EXPERIMENTS

We compare graph-only $E _ { \mathrm { s t r u c t } }$ and TEM-only E<sub>TEM</sub> gates (raw per-example scores, no z-score before sorting). Protocol A fits test-split quantile thresholds (ranking diagnostic; Table 1). Protocol B calibrates on validation and applies thresholds to test (deployable; Table $5 ,$ Appendix A). Simple instance-level baselines (MC dropout, an error predictor) are in $\mathrm { A p \mathrm { - } }$ pendix D; graph vs. TEM remains the primary axis comparison because both gates are trained in the same SASF pipeline under identical selective rules.

Why graph-only vs. TEM-only (not fusion). We report separate gates as the primary comparison because our goal is to test whether relational consistency adds an independent abstention signal beyond instance-level energy. A fused score may improve practical performance but would confound this question by blending two axes whose contributions we aim to isolate. Score fusion (e.g., validation z-scoring before blending) is supported in the repository as an optional downstream extension, not as the core empirical claim.

## 5.1 Datasets and backbones

Seven multivariate long-horizon benchmarks (ETTh1/2, ETTm1 $^ { . / 2 , }$ Electricity, Trafic, Weather) with standard splits. Backbones: Autoformer, FEDformer, PatchTST, TimesNet (Wu et al., 2021; Zhou et al., 2022; Nie et al., 2022; Wu et al., 2022). The graph head and TEM branch train jointly with $\lambda _ { g } = 0 . 1 , \ \lambda _ { a } = 0 . 0 5$ , top-k=10 when $D { > } 1 0 .$ The graph head is lightweight in parameter count (embedding matrix plus sparsified $A ) { \mathrm { ; } }$ top-k row sparsity limits relational scoring to a small neighborhood per variable, so runtime remains dominated by the forecaster backbone (Appendix F).

## 5.2 Protocols and metrics

Protocol B (deployable). Thresholds are validation ρ-quantiles of each score, applied to test without re-calibration; selective MSE is mean $e _ { b } \ ( \mathrm { E q . \ ( 1 ) } )$ over accepted points. $N / A$ marks zero test acceptances under split shift.

Protocol A (diagnostic; Table 1). Test-split quantiles hold nominal coverage by construction, isolating ranking under matched retention. The Full column is full-coverage test MSE. Tables 1–2 use raw ebm\_only/graph\_only energies (fusion variants in the repository).

Ranking view. Under Protocol A, at nominal coverage $\rho ,$ the $\lfloor \rho B \rfloor$ test samples with lowest abstention scores are retained by construction (Eq. (3)). Under Protocol B, the threshold is the validation ρ-quantile of each score, applied to test without re-calibration; realized test coverage can difer from $\rho$ under split shift (including zero acceptances, N/A). In both cases, selective risk on the accepted set depends on how well scores separate lower- from higher-error samples $e _ { b }$ (Eq. (1)). For a fixed realized acceptance set of size $k ,$ sorting by the true error oracle minimizes empirical selective risk (Theorem 3, Appendix E.3). Training encourages $E _ { \mathrm { s t r u c t } }$ to align with error proxies via $\mathcal { L } _ { \mathrm { a l i g n } }$ (Section 4); mis-ranked acceptances bound excess risk at size k (Theorem 4).

I%↑ compares graph vs. TEM at the same nominal target $\rho \mathrm { : \ ( T E M - G r a p h ) / T E M \times 1 0 0 }$ . S%↑ compares graph vs. full coverage: $( \mathrm { F u l l - G r a p h } _ { \rho } ) / \mathrm { F u l l } \times 1 0 0$ . Table 2 uses the same relative-improvement convention $( 1 { - } \mathrm { M S E _ { s e l } } / \mathrm { M S E _ { o r i g } } ) { \times } 1 0 0$ with tau\_mode=quantile; Full (A0) aligns with S%↑ in Table 1. Appendix A (Table 4) reports a complementary validation→test threshold protocol that need not match every cell numerically.

## 5.3 TEM vs. graph energy (main results)

Table 1 summarizes Protocol A selective MSE for TEM-only vs. graph-only gates; deployable Protocol B results are in Table 5.

Findings. (F1) Coupling. Graph-only gates improve most on Weather, Electricity, and Trafic— benchmarks where cross-variable structure appears more informative for abstention in our setting—often on Autoformer, FEDformer, and PatchTST (positive $\mathbf { I \% } / \mathbf { S \% }$ in Table 1). (F2) Complementarity. TimesNet and several ETT splits favor TEM; these failures are informative: relational abstention targets structural risk and should not dominate when errors are primarily marginal. (F3) Transfer. Protocol B trends align with Protocol A on overlapping cells but include split-shift N/A cells (Appendix A). Risk–coverage curves: Appendix C.

## 5.4 Case study: when TEM misses structural failures

To make the qualitative diference between instancelevel and relational abstention explicit, Figure 2 visualizes acceptance disagreements at $\rho = 0 . 8 0$ on the test split (chosen only for visual separation in the rank– rank scatter, not because it is a reported operating point; tabulated coverages remain $\rho \in \{ 0 . 5 , 0 . 7 , 0 . 9 \}$ ). We use percentile ranks of the two scores (lower score ⇒ more likely accepted) and dashed lines at 80% nominal retention under the same Protocol A quantile logic as Table 1.

We define a relational miss as a sample $E _ { \mathrm { T E M } }$ would accept but $E _ { \mathrm { s t r u c t } }$ rejects. Figure 2 and Figure 3 show relational misses carry higher test MSE than mutual acceptance (qualitative support for Section 4).

## 5.5 Ablations

On Weather, Electricity, and ETTh2 (PatchTST, Autoformer), A0 is full SASF; A1 drops alignment $\scriptstyle ( \lambda _ { a } = 0 )$ ; A2 uses uniform $w _ { b } ;$ A3 fixes A to a correlation graph. A1 tests whether $E _ { \mathrm { s t r u c t } }$ ranks risk without explicit alignment; A2 whether error weighting shapes A; A3 whether adaptation beats a static graph. Table 2 (Protocol A): $\mathrm { A 1 / A 2 }$ usually hurt versus A0; A3 is often worse than A0. Validation→test ablations: Appendix A.

## 5.6 When does relational abstention help?

Three regimes. We interpret backbone–dataset variation through three regimes: (i) relational failure dominates—high error aligns with violations of stable cross-variable structure (relational misses; Section 5.4); (ii) marginal failure dominates—error is mainly per-channel or trajectory-level, so instancelevel $E _ { \mathrm { T E M } }$ ranks risk well; (iii) global graph mismatch—a single training-time A poorly matches the test regime (weak or shifting coupling), so $E _ { \mathrm { s t r u c t } }$ is less discriminative. (F1)–(F3) map to these regimes rather than to uniform superiority of either gate.

SASF is most useful in regime (i): $E _ { \mathrm { s t r u c t } }$ targets a risk axis that $E _ { \mathrm { T E M } }$ does not explicitly test, consistent with gains on Weather, Electricity, and Trafic. In regimes (ii)–(iii), $E _ { \mathrm { T E M } }$ often wins $( \mathrm { e . g . }$ , TimesNet and several ETT splits in F2); we interpret these primarily as informative scope limits rather than evidence for universal superiority of either gate.

Link to analysis. Theorem 2 formalizes regime (i): when $E _ { \mathrm { s t r u c t } } ( Y , A )$ is small, large $E _ { \mathrm { s t r u c t } } ( \hat { Y } , A )$ certifies non-negligible squared error; the certificate weakens when smoothness fails (regimes (ii)–(iii), Appendix E.6). Heterogeneous backbone–dataset results therefore reflect how much selective risk lies on the relational axis captured by $E _ { \mathrm { s t r u c t } }$ , not an unexpected breakdown of the method.

Table 1: (Protocol $\mathbf { A } ;$ ranking diagnostic.) Test selective prediction MSE across datasets and backbones (lower is better). Thresholds are test-split empirical ρ-quantiles of each raw score (TEM vs. structural), not validation-calibrated deployment thresholds (Protocol B: Table 5). Graph is bolded when lower than TEM at the same ρ. I%↑: (TEM − Graph)/TEM × 100. S%↑: (Full − Graph $_ { \rho } ) / \mathrm { F u l l } \times 1 0 0$ . Positive is better for both.
<table><tr><td rowspan=2 colspan=6>Dataset   Backbone              ρ= 0.5TEMGraph 1%↑S%↑</td><td rowspan=1 colspan=4>ρ= 0.7</td><td rowspan=1 colspan=4>ρ= 0.9</td><td rowspan=1 colspan=1>Full</td></tr><tr><td rowspan=1 colspan=4>TEMGraph1%↑S%↑</td><td rowspan=1 colspan=4>TEMGraph 1%↑S%↑</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>ETTh1    Autoformer 0.4453</td><td rowspan=1 colspan=1>0.4209</td><td rowspan=1 colspan=2>+5.5+4.1</td><td rowspan=1 colspan=1>0.4403</td><td rowspan=1 colspan=1>0.4091</td><td rowspan=1 colspan=2>+7.1+6.8</td><td rowspan=1 colspan=1>0.4388</td><td rowspan=1 colspan=1>0.4250</td><td rowspan=1 colspan=2>+3.1+3.1</td><td rowspan=1 colspan=1>0.4388</td></tr><tr><td rowspan=1 colspan=3>FEDformer 0.4030</td><td rowspan=1 colspan=1>0.3893</td><td rowspan=1 colspan=1>+3.4</td><td rowspan=1 colspan=1>+3.9</td><td rowspan=1 colspan=1>0.4063</td><td rowspan=1 colspan=1>30.3828</td><td rowspan=1 colspan=1>+5.8</td><td rowspan=1 colspan=1>+5.5</td><td rowspan=1 colspan=1>0.4051</td><td rowspan=1 colspan=1>0.3956</td><td rowspan=1 colspan=1>+2.3</td><td rowspan=1 colspan=1>+2.3</td><td rowspan=1 colspan=1>0.4049</td></tr><tr><td rowspan=1 colspan=2>PatchTST</td><td rowspan=1 colspan=1>0.3837</td><td rowspan=1 colspan=1>0.3547</td><td rowspan=1 colspan=1>+7.6</td><td rowspan=1 colspan=1>+7.6</td><td rowspan=1 colspan=1>0.3824</td><td rowspan=1 colspan=1>0.3650</td><td rowspan=1 colspan=1>+4.6</td><td rowspan=1 colspan=1>+4.9</td><td rowspan=1 colspan=1>0.3839</td><td rowspan=1 colspan=1>0.3780</td><td rowspan=1 colspan=1>+1.5</td><td rowspan=1 colspan=1>+1.5</td><td rowspan=1 colspan=1>0.3838</td></tr><tr><td rowspan=1 colspan=2>TimesNet</td><td rowspan=1 colspan=1>0.3716</td><td rowspan=1 colspan=1>0.3484</td><td rowspan=1 colspan=1>+6.2</td><td rowspan=1 colspan=1>+10.2</td><td rowspan=1 colspan=1>0.3740</td><td rowspan=1 colspan=1>0.3685</td><td rowspan=1 colspan=1>+1.5</td><td rowspan=1 colspan=1>+5.0</td><td rowspan=1 colspan=1>0.3862</td><td rowspan=1 colspan=1>0.3916</td><td rowspan=1 colspan=1>-1.4</td><td rowspan=1 colspan=1>-1.0</td><td rowspan=1 colspan=1>0.3879</td></tr><tr><td rowspan=1 colspan=1>ETTh2</td><td rowspan=1 colspan=1>Autoformer</td><td rowspan=1 colspan=1>0.2587</td><td rowspan=1 colspan=1>0.2448</td><td rowspan=1 colspan=1>+5.4</td><td rowspan=1 colspan=1>+18.7</td><td rowspan=1 colspan=1>0.2665</td><td rowspan=1 colspan=1>0.2685</td><td rowspan=1 colspan=1>-0.8</td><td rowspan=1 colspan=1>+10.8</td><td rowspan=1 colspan=1>0.2882</td><td rowspan=1 colspan=1>0.2907</td><td rowspan=1 colspan=1>-0.9</td><td rowspan=1 colspan=1>+3.5</td><td rowspan=1 colspan=1>0.3011</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FEDformer</td><td rowspan=1 colspan=1>0.2703</td><td rowspan=1 colspan=1>0.2452</td><td rowspan=1 colspan=1>+9.3</td><td rowspan=1 colspan=1>3+22.4</td><td rowspan=1 colspan=1>0.3025</td><td rowspan=1 colspan=1>0.2603</td><td rowspan=1 colspan=1>+14.0</td><td rowspan=1 colspan=1>+17.6</td><td rowspan=1 colspan=1>0.3143</td><td rowspan=1 colspan=1>0.2924</td><td rowspan=1 colspan=1>+7.0</td><td rowspan=1 colspan=1>+7.4</td><td rowspan=1 colspan=1>0.3159</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PatchTST</td><td rowspan=1 colspan=1>0.2597</td><td rowspan=1 colspan=1>0.1951+</td><td rowspan=1 colspan=1>24.9</td><td rowspan=1 colspan=1>+24.1</td><td rowspan=1 colspan=1>0.2595</td><td rowspan=1 colspan=1>0.2021</td><td rowspan=1 colspan=1>+22.1</td><td rowspan=1 colspan=1>+21.3</td><td rowspan=1 colspan=1>0.2615</td><td rowspan=1 colspan=1>0.2358</td><td rowspan=1 colspan=1>+9.8</td><td rowspan=1 colspan=1>+8.2</td><td rowspan=1 colspan=1>0.2569</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TimesNet</td><td rowspan=1 colspan=1>0.2493</td><td rowspan=1 colspan=1>0.3313-</td><td rowspan=1 colspan=1>32.9</td><td rowspan=1 colspan=1>-26.0</td><td rowspan=1 colspan=1>0.2534</td><td rowspan=1 colspan=1>0.3173</td><td rowspan=1 colspan=1>-25.2</td><td rowspan=1 colspan=1>-20.7</td><td rowspan=1 colspan=1>0.2552</td><td rowspan=1 colspan=1>0.2828</td><td rowspan=1 colspan=1>-10.8</td><td rowspan=1 colspan=1>-7.6</td><td rowspan=1 colspan=1>0.2629</td></tr><tr><td rowspan=1 colspan=1>ETTm1</td><td rowspan=1 colspan=1>Autoformer</td><td rowspan=1 colspan=1>0.4509</td><td rowspan=1 colspan=1>0.3928</td><td rowspan=1 colspan=1>+12.9</td><td rowspan=1 colspan=1>+11.0</td><td rowspan=1 colspan=1>0.4435</td><td rowspan=1 colspan=1>0.4342</td><td rowspan=1 colspan=1>+2.1</td><td rowspan=1 colspan=1>+1.6</td><td rowspan=1 colspan=1>0.4390</td><td rowspan=1 colspan=1>0.4374</td><td rowspan=1 colspan=1>+0.4</td><td rowspan=1 colspan=1>+0.8</td><td rowspan=1 colspan=1>0.4411</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FEDformer</td><td rowspan=1 colspan=1>0.4724</td><td rowspan=1 colspan=1>0.3980</td><td rowspan=1 colspan=1>+15.7</td><td rowspan=1 colspan=1>+11.7</td><td rowspan=1 colspan=1>0.4678</td><td rowspan=1 colspan=1>0.4326</td><td rowspan=1 colspan=1>+7.5</td><td rowspan=1 colspan=1>+4.1</td><td rowspan=1 colspan=1>0.4615</td><td rowspan=1 colspan=1>0.4448</td><td rowspan=1 colspan=1>+3.6</td><td rowspan=1 colspan=1>+1.4</td><td rowspan=1 colspan=1>0.4509</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PatchTST</td><td rowspan=1 colspan=1>0.4003</td><td rowspan=1 colspan=1>0.3262</td><td rowspan=1 colspan=1>+18.5</td><td rowspan=1 colspan=1>+9.4</td><td rowspan=1 colspan=1>0.3816</td><td rowspan=1 colspan=1>0.3450</td><td rowspan=1 colspan=1>+9.6</td><td rowspan=1 colspan=1>+4.2</td><td rowspan=1 colspan=1>0.3657</td><td rowspan=1 colspan=1>0.3548</td><td rowspan=1 colspan=1>+3.0</td><td rowspan=1 colspan=1>+1.5</td><td rowspan=1 colspan=1>0.3601</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TimesNet</td><td rowspan=1 colspan=1>0.2824</td><td rowspan=1 colspan=1>0.3171</td><td rowspan=1 colspan=1>-12.3</td><td rowspan=1 colspan=1>-1.4</td><td rowspan=1 colspan=1>0.2877</td><td rowspan=1 colspan=1>0.3219</td><td rowspan=1 colspan=1>-11.9</td><td rowspan=1 colspan=1>-2.9</td><td rowspan=1 colspan=1>0.2969</td><td rowspan=1 colspan=1>0.3168</td><td rowspan=1 colspan=1>-6.7</td><td rowspan=1 colspan=1>-1.3</td><td rowspan=1 colspan=1>0.3127</td></tr><tr><td rowspan=1 colspan=1>ETTm2</td><td rowspan=1 colspan=1>Autoformer</td><td rowspan=1 colspan=1>0.1844</td><td rowspan=1 colspan=1>0.1736</td><td rowspan=1 colspan=1>+5.9</td><td rowspan=1 colspan=1>-1.0</td><td rowspan=1 colspan=1>0.1762</td><td rowspan=1 colspan=1>0.1582</td><td rowspan=1 colspan=1>+10.2</td><td rowspan=1 colspan=1>+8.0</td><td rowspan=1 colspan=1>0.1735</td><td rowspan=1 colspan=1>0.1522</td><td rowspan=1 colspan=1>+12.3</td><td rowspan=1 colspan=1>+11.4</td><td rowspan=1 colspan=1>0.1719</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FEDformer</td><td rowspan=1 colspan=1>0.1363</td><td rowspan=1 colspan=1>0.1593</td><td rowspan=1 colspan=1>-16.9</td><td rowspan=1 colspan=1>+8.7</td><td rowspan=1 colspan=1>0.1342</td><td rowspan=1 colspan=1>0.1495</td><td rowspan=1 colspan=1>-11.4</td><td rowspan=1 colspan=1>+14.3</td><td rowspan=1 colspan=1>0.1688</td><td rowspan=1 colspan=1>0.1575</td><td rowspan=1 colspan=1>+6.7</td><td rowspan=1 colspan=1>+9.7</td><td rowspan=1 colspan=1>0.1744</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PatchTST</td><td rowspan=1 colspan=1>0.1516</td><td rowspan=1 colspan=1>0.1547</td><td rowspan=1 colspan=1>-2.0</td><td rowspan=1 colspan=1>+2.5</td><td rowspan=1 colspan=1>0.1469</td><td rowspan=1 colspan=1>0.1471</td><td rowspan=1 colspan=1>-0.1</td><td rowspan=1 colspan=1>+7.3</td><td rowspan=1 colspan=1>0.1530</td><td rowspan=1 colspan=1>0.1419</td><td rowspan=1 colspan=1>+7.3</td><td rowspan=1 colspan=1>+10.5</td><td rowspan=1 colspan=1>0.1586</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TimesNet</td><td rowspan=1 colspan=1>0.1299</td><td rowspan=1 colspan=1>0.1474</td><td rowspan=1 colspan=1>-13.5</td><td rowspan=1 colspan=1>-3.4</td><td rowspan=1 colspan=1>0.1341</td><td rowspan=1 colspan=1>0.1301</td><td rowspan=1 colspan=1>+3.0</td><td rowspan=1 colspan=1>+8.8</td><td rowspan=1 colspan=1>0.1370</td><td rowspan=1 colspan=1>0.1316</td><td rowspan=1 colspan=1>+3.9</td><td rowspan=1 colspan=1>+7.7</td><td rowspan=1 colspan=1>0.1426</td></tr><tr><td rowspan=1 colspan=1>Electricity</td><td rowspan=1 colspan=1>Autoformer</td><td rowspan=1 colspan=1>0.2925</td><td rowspan=1 colspan=1>0.2633</td><td rowspan=1 colspan=1>+10.0</td><td rowspan=1 colspan=1>+9.8</td><td rowspan=1 colspan=1>0.2933</td><td rowspan=1 colspan=1>0.2762</td><td rowspan=1 colspan=1>+5.8</td><td rowspan=1 colspan=1>+5.4</td><td rowspan=1 colspan=1>0.2924</td><td rowspan=1 colspan=1>0.2863</td><td rowspan=1 colspan=1>+2.1</td><td rowspan=1 colspan=1>+2.0</td><td rowspan=1 colspan=1>0.2920</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FEDformer</td><td rowspan=1 colspan=1>0.1890</td><td rowspan=1 colspan=1>0.1601</td><td rowspan=1 colspan=1>+15.3</td><td rowspan=1 colspan=1>+15.8</td><td rowspan=1 colspan=1>0.1892</td><td rowspan=1 colspan=1>0.1683</td><td rowspan=1 colspan=1>+11.0+</td><td rowspan=1 colspan=1>11.5</td><td rowspan=1 colspan=1>0.1897</td><td rowspan=1 colspan=1>0.1859</td><td rowspan=1 colspan=1>+2.0</td><td rowspan=1 colspan=1>+2.2</td><td rowspan=1 colspan=1>0.1901</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PatchTST</td><td rowspan=1 colspan=1>0.1978</td><td rowspan=1 colspan=1>0.1724</td><td rowspan=1 colspan=1>+12.8+</td><td rowspan=1 colspan=1>15.1</td><td rowspan=1 colspan=1>0.2015</td><td rowspan=1 colspan=1>0.1804</td><td rowspan=1 colspan=1>+10.5</td><td rowspan=1 colspan=1>+11.1</td><td rowspan=1 colspan=1>0.2026</td><td rowspan=1 colspan=1>0.1979</td><td rowspan=1 colspan=1>+2.3</td><td rowspan=1 colspan=1>+2.5</td><td rowspan=1 colspan=1>0.2030</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TimesNet</td><td rowspan=1 colspan=1>0.1753</td><td rowspan=1 colspan=1>0.1311</td><td rowspan=1 colspan=1>+25.2</td><td rowspan=1 colspan=1>+20.3</td><td rowspan=1 colspan=1>0.1694</td><td rowspan=1 colspan=1>0.1398</td><td rowspan=1 colspan=1>+17.5</td><td rowspan=1 colspan=1>+15.0</td><td rowspan=1 colspan=1>0.1654</td><td rowspan=1 colspan=1>0.1494</td><td rowspan=1 colspan=1>+9.7</td><td rowspan=1 colspan=1>+9.2</td><td rowspan=1 colspan=1>0.1645</td></tr><tr><td rowspan=1 colspan=1>Traffic</td><td rowspan=1 colspan=1>Autoformer</td><td rowspan=1 colspan=1>1.0117</td><td rowspan=1 colspan=1>0.9479</td><td rowspan=1 colspan=1>+6.3</td><td rowspan=1 colspan=1>+3.2</td><td rowspan=1 colspan=1>0.9971</td><td rowspan=1 colspan=1>0.9532</td><td rowspan=1 colspan=1>+4.4</td><td rowspan=1 colspan=1>+2.6</td><td rowspan=1 colspan=1>0.9853</td><td rowspan=1 colspan=1>0.9608</td><td rowspan=1 colspan=1>+2.5</td><td rowspan=1 colspan=1>+1.9</td><td rowspan=1 colspan=1>0.9791</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FEDformer</td><td rowspan=1 colspan=1>0.6945</td><td rowspan=1 colspan=1>0.7003</td><td rowspan=1 colspan=1>-0.8</td><td rowspan=1 colspan=1>+0.8</td><td rowspan=1 colspan=1>0.6982</td><td rowspan=1 colspan=1>0.6948</td><td rowspan=1 colspan=1>+0.5</td><td rowspan=1 colspan=1>+1.6</td><td rowspan=1 colspan=1>0.7046</td><td rowspan=1 colspan=1>0.6998</td><td rowspan=1 colspan=1>+0.7</td><td rowspan=1 colspan=1>+0.9</td><td rowspan=1 colspan=1>0.7061</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PatchTST</td><td rowspan=1 colspan=1>0.6260</td><td rowspan=1 colspan=1>0.6087</td><td rowspan=1 colspan=1>+2.8</td><td rowspan=1 colspan=1>+3.1</td><td rowspan=1 colspan=1>0.6132</td><td rowspan=1 colspan=1>0.5918</td><td rowspan=1 colspan=1>+3.5</td><td rowspan=1 colspan=1>+0.3</td><td rowspan=1 colspan=1>0.5994</td><td rowspan=1 colspan=1>0.5886</td><td rowspan=1 colspan=1>+1.8</td><td rowspan=1 colspan=1>+0.3</td><td rowspan=1 colspan=1>0.5903</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>TimesNet</td><td rowspan=1 colspan=1>0.5922</td><td rowspan=1 colspan=1>0.5464</td><td rowspan=1 colspan=1>+7.7</td><td rowspan=1 colspan=1>+6.4</td><td rowspan=1 colspan=1>0.5866</td><td rowspan=1 colspan=1>0.5513</td><td rowspan=1 colspan=1>+6.0</td><td rowspan=1 colspan=1>+5.6</td><td rowspan=1 colspan=1>0.5866</td><td rowspan=1 colspan=1>0.5695</td><td rowspan=1 colspan=1>+2.9</td><td rowspan=1 colspan=1>+2.5</td><td rowspan=1 colspan=1>0.5839</td></tr><tr><td rowspan=1 colspan=1>Weather</td><td rowspan=1 colspan=1>Autoformer</td><td rowspan=1 colspan=1>0.2675</td><td rowspan=1 colspan=1>0.1896</td><td rowspan=1 colspan=1>+29.1</td><td rowspan=1 colspan=1>+20.5</td><td rowspan=1 colspan=1>0.2521</td><td rowspan=1 colspan=1>0.2012</td><td rowspan=1 colspan=1>+20.2</td><td rowspan=1 colspan=1>+15.7</td><td rowspan=1 colspan=1>0.2414</td><td rowspan=1 colspan=1>0.2236</td><td rowspan=1 colspan=1>+7.4</td><td rowspan=1 colspan=1>+6.3</td><td rowspan=1 colspan=1>0.2386</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>FEDformer</td><td rowspan=1 colspan=1>0.3210</td><td rowspan=1 colspan=1>0.2348+</td><td rowspan=1 colspan=1>26.9</td><td rowspan=1 colspan=1>+22.8</td><td rowspan=1 colspan=1>0.3130</td><td rowspan=1 colspan=1>0.2561</td><td rowspan=1 colspan=1>+18.2</td><td rowspan=1 colspan=1>+15.8</td><td rowspan=1 colspan=1>0.3090</td><td rowspan=1 colspan=1>0.2829</td><td rowspan=1 colspan=1>+8.4</td><td rowspan=1 colspan=1>+6.9</td><td rowspan=1 colspan=1>0.3040</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PatchTST0</td><td rowspan=1 colspan=1>.1783</td><td rowspan=1 colspan=1>0.1338</td><td rowspan=1 colspan=1>+24.9</td><td rowspan=1 colspan=1>+20.7</td><td rowspan=1 colspan=1>0.1727</td><td rowspan=1 colspan=1>0.1381</td><td rowspan=1 colspan=1>+20.0</td><td rowspan=1 colspan=1>+18.1</td><td rowspan=1 colspan=1>0.1701</td><td rowspan=1 colspan=1>0.1548</td><td rowspan=1 colspan=1>+9.0</td><td rowspan=1 colspan=1>+8.2</td><td rowspan=1 colspan=1>0.1687</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>TimesNet0.1278</td><td rowspan=1 colspan=1>0.1133</td><td rowspan=1 colspan=1>+11.3</td><td rowspan=1 colspan=1>+17.7</td><td rowspan=1 colspan=1>0.1316</td><td rowspan=1 colspan=1>0.1216</td><td rowspan=1 colspan=1>+7.6</td><td rowspan=1 colspan=1>+11.7</td><td rowspan=1 colspan=1>0.1343</td><td rowspan=1 colspan=1>0.1256</td><td rowspan=1 colspan=2>+6.5 +8.8</td><td rowspan=1 colspan=1>0.1377</td></tr></table>

![](images/33c0cce28540e2b6d74774c09cec1028386b055eb90688f94018413a7d28f30c.jpg)

![](images/772a51f960573054bb8341586185be8473b0161efe6df1c5bcb739fa0dc4b1ba.jpg)  
Figure 2: Case study (coverage 0.80). Percentile ranks on the test split (lower ⇒ more likely accepted). Dashed lines mark 80% nominal coverage via test-split score quantiles under the same protocol as Table 1. Left: ranks of E<sub>TEM</sub> vs. $E _ { \mathrm { s t r u c t } } ;$ upper-left marks relational misses (TEM accepts, graph rejects). Right: test MSE by acceptance region; relational misses skew higher-error than mutual acceptance.

Table 2: Graph-only ablation (PatchTST, Autoformer; tau\_mode=quantile, Protocol A). Test relative improvement $( 1 - \mathrm { M S E _ { s e l } / M S E _ { o r i g } } ) \times 1 0 0 $ at $c \in \{ 0 . 5 , 0 . 7 , 0 . 9 \}$ on Weather, Electricity, ETTh2. A0 full; A1 w/o alignment; A2 w/o error weighting; A3 correlation-graph gate. Full (A0) matches Table 1 S%↑; bold: best per column within each backbone block.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Ablation</td><td colspan="3">Weather</td><td colspan="3">Electricity</td><td colspan="3">ETTh2</td></tr><tr><td>0.5</td><td>0.7</td><td>0.9</td><td>0.5</td><td>0.7</td><td>0.9</td><td>0.5</td><td>0.7</td><td>0.9</td></tr><tr><td rowspan="4">PatchTST</td><td>Full (A0)</td><td>20.70</td><td>18.10</td><td>8.20</td><td>15.10</td><td>11.10</td><td>2.50</td><td>24.10</td><td>21.30</td><td>8.20</td></tr><tr><td>w/o align (A1)</td><td>-8.13</td><td>-2.04</td><td>0.78</td><td>13.30</td><td>9.74</td><td>2.12</td><td>-4.76</td><td>1.70</td><td>4.45</td></tr><tr><td>w/o err.-wt (A2)</td><td>10.16</td><td>12.51</td><td>7.88</td><td>12.87</td><td>5.54</td><td>0.80</td><td>9.17</td><td>9.15</td><td>5.38</td></tr><tr><td>Corr. gate (À3)</td><td>-7.30</td><td>-3.97</td><td>-0.81</td><td>13.61</td><td>10.19</td><td>1.75</td><td>4.29</td><td>6.86</td><td>1.88</td></tr><tr><td rowspan="4">Autoformer</td><td>Full (A0)</td><td>20.50</td><td>15.70</td><td>6.30</td><td>9.80</td><td>5.40</td><td>2.00</td><td>18.70</td><td>10.80</td><td>3.50</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  align (A1)</td><td>1.17</td><td>5.21</td><td>3.10</td><td>8.33</td><td>4.62</td><td>1.78</td><td>-0.13</td><td>-0.40</td><td>0.15</td></tr><tr><td>w/o err.-wt (Á2)</td><td>18.88</td><td>15.22</td><td>6.14</td><td>7.49</td><td>4.92</td><td>0.31</td><td>16.11</td><td>7.19</td><td>0.36</td></tr><tr><td>Corr. gate (À3)</td><td>0.93</td><td>2.59</td><td>1.44</td><td>8.97</td><td>3.40</td><td>1.78</td><td>-5.73</td><td>0.72</td><td>0.80</td></tr></table>

![](images/24b3833d86fc77a74e08315128f361d649f163e6d633c2ded94c5d4ca375bdef.jpg)  
Figure 3: A representative relational miss (test index 9451): TEM accepts at 0.80 target coverage while the structural gate rejects. Heatmap: channel-wise squared error over the horizon (log-scaled).

Practical implications. $E _ { \mathrm { T E M } }$ and $E _ { \mathrm { s t r u c t } }$ encode diferent reliability axes; deployments may prefer one gate, switch by regime, or apply fusion (repository) when the dominant failure mode is known—relational abstention complements instance-level scoring rather than replacing it.

Table 3: Multiseed deployable Protocol B (aligned subset). Same three complete (dataset, $\rho )$ cells for both backbones: ETTh1 $( \rho { = } 0 . 7 , 0 . 9 )$ and ETTh2 $\left( \rho { = } 0 . 9 \right)$ , with seeds {0, 1, 2024} (partial cells excluded).
<table><tr><td>Backbone</td><td>Cells</td><td>Seeds</td><td>Wins</td><td>Mean I% ↑</td></tr><tr><td>PatchTST</td><td>3</td><td>9</td><td>9/9</td><td> $+ 1 0 . 6 \pm 1 2 . 8$ </td></tr><tr><td>Autoformer</td><td>3</td><td>9</td><td>7/9</td><td> $+ 7 . 1 \pm 1 7 . 3$ </td></tr></table>

## 5.7 Threats to validity

Statistical robustness. Table 1 (Protocol A) uses one checkpoint per cell (seed 2024) and isolates rank ing under matched test coverage; it is not a deployable threshold-transfer readout. To probe seed sensitivity under Protocol B, we rerun validation→test selection with seeds {0, 1, 2024} on ETTh1/ETTh2/Electricity for PatchTST and Autoformer, excluding partial cells rather than imputing them (Appendix B). Table 3 aligns both backbones on the same three complete (dataset, ρ) settings; mean I% ↑ averages cell-level means (each cell over seeds), and ± is the std across those cell means.

On this aligned subset, graph selective MSE wins all nine PatchTST seed runs $( 9 / 9 )$ and seven of nine Autoformer runs $( 7 / 9 ;$ one-sided binomial sign test p≈0.09). Extended PatchTST complete cells (n=21 seed runs, all graph wins) yield binomial and Wilcoxon tests with $p \ll 1 0 ^ { - 5 }$ (subset; not the full main-table grid; details in Appendix B). Small $\mathbf { I \% } / \mathbf { S \% }$ gaps elsewhere in Table 1 should still be read cautiously; we emphasize directional patterns (F1–F3) rather than uniform gains on every backbone–dataset pair.

Structural assumptions. Relational gating assumes that a global, unsigned smoothness prior over variables is a useful abstention proxy. This may hold unevenly across datasets and backbones (Section 5.6); it does not model lags, signed coupling, or samplespecific graphs.

Deployment transfer. Protocol B applies validation thresholds to test without recalibration. Validation–test score shift can yield zero test acceptances (N/A in Appendix A), so deployable selective MSE is undefined in those cells. This limits threshold transfer under Protocol B; Protocol A ranking diagnostics are unafected.

## 6 CONCLUSION

We studied structure-aware selective forecasting: relational consistency, operationalized by E<sub>struct</sub>, complements instance-level TEM under explicit coverage constraints. Gains concentrate where relational structure is informative (Section 5.6); negative cells mark scope limits rather than a universal replacement for instance scores. Multivariate selective forecasting should ask both whether trajectories are plausible and whether they are relationally consistent. Limitations (protocols, seeds, deployment transfer) are in Section 5.7.

## AI Use Statement

Generative AI tools were used only for English language editing, readability, and LaTeX/formatting assistance on the manuscript. They were not used for model design, training, or the experimental pipeline; for theoretical results, proofs, or mathematical claims; or for producing or altering primary experimental numbers reported in the paper.

All AI-assisted wording and formatting suggestions were reviewed, verified, and edited by the authors. The authors take full responsibility for the final content of this work, including text, claims, and artifacts.

## References

Angelopoulos Anastasios, N. and Stephen, B. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv preprint arXiv:2107.07511.

Benidis, K., Rangapuram, S. S., Flunkert, V., Wang, B., Maddix, D. C., Turkmen, C., Gasthaus, J., Bohlke-Schneider, M., Salinas, D., Stella, L., Aubet, F.-X., Callot, L., and Januschowski, T. (2022). Deep learning for time series forecasting: Tutorial and literature survey. ACM Computing Surveys, 55(6).

Brusokas, J., Tirupathi, S., Zhang, D., and Pedersen, T. B. (2025). The time-energy model: Selective time-series forecasting using energy-based models. Transactions on Machine Learning Research.

Cai, W., Wang, K., Wu, H., Chen, X., and Wu, Y. (2024). Forecastgrapher: Redefining multivariate time series forecasting with graph neural networks. arXiv preprint arXiv:2405.18036.

Chen, L., Chen, D., Shang, Z., Wu, B., Zheng, C., Wen, B., and Zhang, W. (2023). Multi-scale adaptive graph neural network for multivariate time series forecasting. IEEE Transactions on Knowledge and Data Engineering, 35(10):10748–10761.

Chen, W., Wang, Y., Du, C., Jia, Z., Liu, F., and Chen, R. (2022). Balanced graph structure learning for multivariate time series forecasting. arXiv preprint arXiv:2201.09686.

Cortes, C., DeSalvo, G., and Mohri, M. (2016). Learning with rejection. In International conference on algorithmic learning theory, pages 67–82. Springer.

Gal, Y. and Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. In International Conference on Machine Learning (ICML).

Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep neural networks. Advances in neural information processing systems, 30.

Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477):359– 378.

Hyvärinen, A. and Dayan, P. (2005). Estimation of non-normalized statistical models by score match ing. Journal of Machine Learning Research, 6(4).

Kim, J., Lee, H., Yu, S., Hwang, U., Jung, W., and Yoon, K. (2023). Hierarchical joint graph learning and multivariate time series forecasting. IEEE Access, 11:118386–118394.

Kipf, T. N. and Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR).

Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems (NeurIPS), volume 30.

LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., Huang, F., et al. (2006). A tutorial on energy-based learning. Predicting structured data, 1(0).

Liu, Y., Liu, Q., Zhang, J.-W., Feng, H., Wang, Z., Zhou, Z., and Chen, W. (2022). Multivariate timeseries forecasting with temporal polynomial graph neural networks. Advances in neural information processing systems, 35:19414–19426.

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2022). A time series is worth 64 words: Long-term forecasting with transformers. arxiv 2022. arXiv preprint arXiv:2211.14730.

Romano, Y., Patterson, E., and Candès, E. J. (2019). Conformalized quantile regression. In Advances in Neural Information Processing Systems (NeurIPS), volume 32.

Sriramulu, A., Fourrier, N., and Bergmeir, C. (2023). Adaptive dependency learning graph neural networks. Information Sciences, 625:700–714.

Vovk, V., Gammerman, A., and Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.

Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., and Long, M. (2022). Timesnet: Temporal 2d-variation modeling for general time series analysis. arXiv preprint arXiv:2210.02186.

Wu, H., Xu, J., Wang, J., and Long, M. (2021). Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting. Advances in neural information processing systems, 34:22419–22430.

Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., and Zhang, C. (2020). Connecting the dots: Multivariate time series forecasting with graph neural networks. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 753–763.

Yu, B., Yin, H., and Zhu, Z. (2017). Spatio-temporal graph convolutional networks: A deep learning framework for trafic forecasting. arXiv preprint arXiv:1709.04875.

Zeng, A., Chen, M., Zhang, L., and Xu, Q. (2023). Are transformers efective for time series forecasting? In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 11121–11128.

Zhang, W., Yin, C., Liu, H., Zhou, X., and Xiong, H. (2024). Irregular multivariate time series forecasting: A transformable patching graph neural networks approach. In Forty-first International Conference on Machine Learning.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W. (2021). Informer: Beyond eficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 11106– 11115.

Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., and Jin, R. (2022). Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In International conference on machine learning, pages 27268–27286. PMLR.

## A SUPPLEMENTARY ABLATION

Deployable evaluation (Protocol B). This appendix is the primary deployment-style readout: thresholds are fit on validation quantiles and applied to test selection (Section 5). Table 5 is the full TEM vs. graph comparison; Table 4 gives a complementary ablation under the same validation→test pipeline. Main-text Table 1 remains Protocol A (test quantiles) for ranking diagnostics only.

Table 4: (Supplementary.) Ablation study for graph-only selective inference (PatchTST and Autoformer). For each target coverage, we choose energy thresholds on the validation split and apply them to select samples on the test split (complementary to the selective reporting used for Table 1 in the main text). We report test relative improvement $( 1 - \mathrm { M S E _ { s e l } / M S E _ { o r i g } } ) \times 1 0 0 $ at $c \in \{ 0 . 5 , 0 . 7 , 0 . 9 \}$ . Datasets: Weather, Electricity, ETTh2 (multivariate M). A0: full model (learned adaptive graph, alignment + error-weighted graph loss); A1: $\lambda _ { a } { = } 0$ (no alignment); A2: uniform graph regularization weights (no error weighting); A3: fixed correlation-graph gate (no learned A). Bold values: best relative improvement in each column within each backbone block (ties bolded together).
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Ablation</td><td colspan="3">Weather</td><td colspan="3">Electricity</td><td colspan="3">ETTh2</td></tr><tr><td>0.5</td><td>0.7</td><td>0.9</td><td>0.5</td><td>0.7</td><td>0.9</td><td>0.5</td><td>0.7</td><td>0.9</td></tr><tr><td rowspan="4">PatchTST</td><td>Full (A0)</td><td>12.78</td><td>9.73</td><td>3.55</td><td>28.66</td><td>20.50</td><td>17.36</td><td>38.17</td><td>11.27</td><td>8.80</td></tr><tr><td>w/o align (A1)</td><td>-9.06</td><td>-2.42</td><td>0.29</td><td>8.62</td><td>9.31</td><td>13.52</td><td>-7.09</td><td>-2.58</td><td>2.03</td></tr><tr><td>w/o err.-wt (Á2)</td><td>12.31</td><td>9.42</td><td>3.39</td><td>24.01</td><td>20.39</td><td>14.70</td><td>37.91</td><td>10.72</td><td>8.54</td></tr><tr><td>Corr. gate (À3)</td><td>-7.51</td><td>-7.30</td><td>-3.28</td><td>14.60</td><td>14.16</td><td>14.27</td><td>10.87</td><td>-5.78</td><td>-7.65</td></tr><tr><td rowspan="4">Autoformer</td><td>Full (A0)</td><td>20.50</td><td>15.66</td><td>6.87</td><td>17.83</td><td>14.10</td><td>13.94</td><td>76.77</td><td>66.98</td><td>52.25</td></tr><tr><td>w/o align (A1)</td><td>3.12</td><td>1.75</td><td>5.29</td><td>12.33</td><td>13.53</td><td>8.97</td><td>58.73</td><td>53.66</td><td>31.62</td></tr><tr><td>w/o err.-wt (Á2)</td><td>19.77</td><td>14.48</td><td>2.66</td><td>11.55</td><td>11.59</td><td>6.45</td><td>75.56</td><td>64.38</td><td>51.98</td></tr><tr><td>Corr. gate (À3)</td><td>3.73</td><td>3.47</td><td>2.70</td><td>16.03</td><td>13.76</td><td>12.56</td><td>26.01</td><td>-17.65</td><td>-8.10</td></tr></table>

Validation→test thresholding (TEM vs. graph). Under the same validation-calibrated thresholding protocol, Table 5 reports TEM vs. graph selective MSE across all backbones and datasets. In a few backbone–dataset cases, a validation-derived threshold can yield zero accepted test samples (coverage = 0) because abstention-score distributions shift across splits; we mark these as $\mathrm { N } / \mathrm { A }$ since selective MSE and derived I/S metrics are undefined.

Table 5: Validation→test selective prediction MSE across datasets and backbones (lower is better). For each $\rho \in \{ 0 . 5 , 0 . 7 , 0 . 9 \}$ , acceptance thresholds are fit as empirical ρ-quantiles of each raw abstention score (TEM vs. structural) on the validation split and then applied to the test split without re-calibration (Protocol B). Graph is bolded when lower than TEM at the same ρ. I%↑ is relative MSE reduction vs. TEM: (TEM−Graph)/TEM×100. S%↑ is relative MSE reduction vs. full-split MSE: $( \mathrm { F u l l - G r a p h } _ { \rho } ) / \mathrm { F u l l } \times 1 0 0 .$ . Positive is better for both. $N / A$ indicates that the validation-derived threshold yields zero accepted test samples (coverage = 0), so selective MSE and the derived I/S metrics are undefined.
<table><tr><td>Dataset</td><td>Backbone</td><td colspan="4">ρ= 0.5</td><td colspan="4">ρ= 0.7</td><td colspan="4">ρ= 0.9</td><td rowspan="2">Full</td></tr><tr><td></td><td></td><td>TEM</td><td>Graph</td><td>1%↑</td><td>S%↑</td><td>TEM</td><td>Graph</td><td>1%↑</td><td>S%↑</td><td>TEM</td><td>Graph</td><td>1%↑</td><td>S%↑</td></tr><tr><td>ETTh1</td><td>Autoformer 0.4562</td><td></td><td>0.3983</td><td>+12.7</td><td>+9.2</td><td>0.4418</td><td>0.4186</td><td>+5.3</td><td>+4.6</td><td>0.4392</td><td>0.4145</td><td>+5.6</td><td>+5.5</td><td>0.4388</td></tr><tr><td></td><td>FEDformer 0.4019</td><td></td><td>0.3100</td><td>+22.9</td><td>+23.4</td><td>0.4071</td><td>0.3849</td><td>+5.5</td><td>+4.9</td><td>0.4044</td><td>0.3989</td><td>+1.4</td><td>+1.5</td><td>0.4049</td></tr><tr><td></td><td>PatchTST</td><td>0.3810</td><td>0.3394</td><td>+10.9</td><td>+11.6</td><td>0.3826</td><td>0.3649</td><td>+4.6</td><td>+4.9</td><td>0.3836</td><td>0.3827</td><td>+0.2</td><td>+0.3</td><td>0.3838</td></tr><tr><td></td><td>TimesNet</td><td>0.3713</td><td>0.2729</td><td>+26.5</td><td>+29.6</td><td>0.3715</td><td>0.3146</td><td>+15.3</td><td>+18.9</td><td>0.3870</td><td>0.3332</td><td>+13.9</td><td>+14.1</td><td>0.3879</td></tr><tr><td>ETTh2</td><td>Autoformer 0.2549</td><td></td><td>0.0955</td><td>+62.5</td><td>+68.3</td><td>0.2598</td><td>0.1210</td><td></td><td>+53.4 +59.8</td><td></td><td>0.2715 0.1591</td><td>+41.4</td><td>+47.2</td><td>0.3011</td></tr><tr><td></td><td>FEDformer</td><td>N/A</td><td>0.2701</td><td>N/A</td><td>+14.5</td><td>0.3384</td><td>0.2682</td><td>+20.7</td><td>+15.1</td><td>0.3103</td><td>0.2917</td><td>+6.0</td><td>+7.7</td><td>0.3159</td></tr><tr><td></td><td>PatchTST</td><td>N/A</td><td>0.1566</td><td>N/A</td><td>+39.0</td><td>N/A</td><td>0.2269</td><td>N/A</td><td>+11.7</td><td>0.2587</td><td>0.2120</td><td>+18.1</td><td>+17.5</td><td>0.2569</td></tr><tr><td></td><td>TimesNet</td><td>0.2360</td><td>0.2629</td><td>-11.4</td><td>+0.0</td><td>0.2519</td><td>0.2629</td><td>-4.4</td><td>+0.0</td><td>0.2537</td><td>0.2629</td><td>-3.6</td><td>+0.0</td><td>0.2629</td></tr><tr><td>ETTm1</td><td>Autoformer 0.4678</td><td></td><td>0.4170</td><td>+10.9</td><td>+5.5</td><td>0.4477</td><td>0.4400</td><td>+1.7</td><td>+0.2</td><td>0.4392</td><td>0.4411</td><td>-0.4</td><td>+0.0</td><td>0.4411</td></tr><tr><td></td><td>FEDformer 0.4648</td><td></td><td>0.4053</td><td>+12.8</td><td>+10.1</td><td>0.4655</td><td>0.4422</td><td>+5.0</td><td>+1.9</td><td>0.4538</td><td>0.4509</td><td>+0.6</td><td>+0.0</td><td>0.4509</td></tr><tr><td></td><td>PatchTST</td><td>0.4115</td><td>0.3296</td><td>+19.9</td><td>+8.5</td><td>0.3924</td><td>0.3528</td><td>+10.1</td><td>+2.0</td><td>0.3733</td><td>0.3600</td><td>+3.6</td><td>+0.0</td><td>0.3601</td></tr><tr><td></td><td>TimesNet</td><td>0.2835</td><td>0.3187</td><td>-12.4</td><td>-1.9</td><td>0.2859</td><td>0.3177</td><td>-11.1</td><td>-1.6</td><td>0.2967</td><td>0.3137</td><td>-5.7</td><td>-0.3</td><td>0.3127</td></tr><tr><td>ETTm2</td><td>Autoformer 0.1732</td><td></td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.1746</td><td>0.1744</td><td>+0.1</td><td>-1.5</td><td>0.1719</td><td>0.1662</td><td>+3.3</td><td>+3.3</td><td>0.1719</td></tr><tr><td></td><td>FEDformer</td><td>0.2440</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.1886</td><td>0.1883</td><td>+0.2</td><td>-8.0</td><td>0.1325</td><td>0.1596</td><td>-20.5</td><td>+8.5</td><td>0.1744</td></tr><tr><td></td><td>PatchTST</td><td>0.1519</td><td>N/A</td><td>N/A</td><td>N/A</td><td>0.1478</td><td>0.1653</td><td>-11.8</td><td>-4.2</td><td>0.1483</td><td>0.1502</td><td>-1.3</td><td>+5.3</td><td>0.1586</td></tr><tr><td>Electricity</td><td>TimesNet</td><td>0.1270</td><td>0.1049</td><td>+17.4</td><td>+26.4</td><td>0.1328</td><td>0.1725</td><td>-29.9</td><td>-21.0</td><td>0.1367</td><td>0.1374</td><td>-0.5</td><td>+3.6</td><td>0.1426</td></tr><tr><td></td><td>Autoformer 0.2924</td><td></td><td>0.2567</td><td>+12.2</td><td>+12.1</td><td>0.2933</td><td>0.2565</td><td></td><td>+12.5 +12.2</td><td>0.2926</td><td>0.2618</td><td>+10.5</td><td>+10.3</td><td>0.2920</td></tr><tr><td></td><td>FEDformer 0.1888</td><td></td><td>0.1593</td><td>+15.6</td><td>+16.2</td><td>0.1894</td><td>0.1589</td><td>+16.1</td><td>+16.4</td><td>0.1897</td><td>0.1628</td><td>+14.2</td><td>+14.4</td><td>0.1901</td></tr><tr><td></td><td>PatchTST</td><td>0.1983</td><td>0.1508</td><td>+24.0</td><td>+25.7</td><td>0.2009</td><td>0.1632</td><td></td><td>+18.8 +19.6</td><td></td><td>0.2023 0.1671</td><td>+17.4</td><td>+17.7</td><td>0.2030</td></tr><tr><td></td><td>TimesNet</td><td>0.1769</td><td>0.1306</td><td>+26.2</td><td>+20.6</td><td>0.1708</td><td>0.1316</td><td>+23.0</td><td>+20.0</td><td>0.1661</td><td>0.1306</td><td>+21.4</td><td>+20.6</td><td>0.1645</td></tr><tr><td>Traffic</td><td>Autoformer 1.0131</td><td></td><td>0.9573</td><td>+5.5</td><td>+2.2</td><td>0.9958</td><td>0.9478</td><td>+4.8</td><td>+3.2</td><td>0.9874</td><td>0.9538</td><td>+3.4</td><td>+2.6</td><td>0.9791</td></tr><tr><td></td><td>FEDformer 0.6946</td><td></td><td>0.7084</td><td>-2.0</td><td>-0.3</td><td>0.6979</td><td>0.7012</td><td>-0.5</td><td>+0.7</td><td>0.7046</td><td>0.6972</td><td>+1.0</td><td>+1.3</td><td>0.7061</td></tr><tr><td></td><td>PatchTST</td><td>0.6270</td><td>0.6436</td><td>-2.6</td><td>-9.0</td><td>0.6130</td><td>0.6122</td><td>+0.1</td><td>-3.7</td><td>0.5983</td><td>0.5922</td><td>+1.0</td><td>-0.3</td><td>0.5903</td></tr><tr><td></td><td>TimesNet</td><td>0.5900</td><td>0.5286</td><td>+10.4</td><td>+9.5</td><td>0.5858</td><td>0.5434</td><td>+7.2</td><td>+6.9</td><td>0.5867</td><td>0.5683</td><td>+3.1</td><td>+2.7</td><td>0.5839</td></tr><tr><td>Weather</td><td>Autoformer 0.2648</td><td></td><td>0.2004</td><td>+24.3</td><td>+16.0</td><td>0.2514</td><td>0.2178</td><td>+13.4</td><td>+8.7</td><td>0.2400</td><td>0.2322</td><td>+3.3</td><td>+2.7</td><td>0.2386</td></tr><tr><td></td><td>FEDformer 0.3134</td><td></td><td>0.2300</td><td>+26.6</td><td>+24.3</td><td>0.3084</td><td>0.2523</td><td>+18.2</td><td>+17.0</td><td>0.3051</td><td>0.2933</td><td>+3.9</td><td>+3.5</td><td>0.3040</td></tr><tr><td></td><td>PatchTST</td><td>0.1792 0.1232</td><td>0.1443 0.1215</td><td>+19.5 +1.4</td><td>+14.5 +11.8</td><td>0.1722 0.1311</td><td>0.1490 0.1251</td><td>+13.5 +4.6</td><td>+11.7 +9.2</td><td>0.1701 0.1333</td><td>0.1614 0.1305</td><td>+5.1 +2.1</td><td>+4.3 +5.2</td><td>0.1687 0.1377</td></tr><tr><td></td><td>TimesNet</td></table>

## B MULTISEED ROBUSTNESS (PROTOCOL B)

Main Tables 1–2 use seed 2024 only. The aligned multiseed summary (Table 3) is reported in Section 5.7; this appendix records extended PatchTST cells, statistical tests, and excluded partial Protocol B settings.

PatchTST (extended complete cells). Adding ETTh1 ρ=0.5 and Electricity $( \rho { = } 0 . 5 , 0 . 7 , 0 . 9 )$ yields 7 cells / 21 seed runs, all graph wins (21/21). Binomial sign test (n=21): p≈1.9×10<sup>−6</sup>; Wilcoxon (n=21): $p { \approx } 5 . 1 { \times } 1 0 ^ { - 7 }$ (subset test; not the full main-table grid). Excluded (incomplete Protocol B cells). Autoformer ETTh1 $\rho { = } 0 . 5 ;$ Autoformer ETTh2 $\rho { = } 0 . 5 , 0 . 7 ;$ PatchTST ETTh2 $\rho { = } 0 . 5 , 0 . 7$ (not all three seeds yield comparable Graph vs. TEM selective MSE under validation-calibrated thresholds). Per-seed logs are included with the experiment code release.

## C BACKBONE–DATASET SELECTIVE CURVES

We provide supplementary risk–coverage visualizations for selected backbone–dataset pairs to complement the main tables. Each plot reports test selective MSE (y-axis) against target coverage (x-axis). For each target coverage level, thresholds follow the same test-split empirical quantile logic as Section 5: for each raw abstention score we take the empirical quantile on the test split corresponding to that coverage level.

How to read the curves. Lower is better. The dashed line (orig\_MSE) is the full-coverage test MSE without

PatchTST.

abstention. Solid blue (ebm\_only) is instance-level $E _ { \mathrm { T E M } } ;$ orange dashed (graph\_only) is $E _ { \mathrm { s t r u c t } }$ . A curve below orig\_MSE means abstention helps; graph\_only below ebm\_only means the structural gate ranks high-error samples better under test-split quantile gating (Section 5).

Autoformer.

![](images/5da38c78c87337d6b37de7f60cf9813fb4c1f1c5bb1cf4c12d666e0bf55913e2.jpg)  
(a) Weather

![](images/07537de00e9a280f8e76dbdb7ded06e9c306cd959917fcecc4d6e784b3dc2803.jpg)  
(b) Electricity

![](images/63e929aea979aeeaaff858f3cfb59fc918dbb548ff2402f9a208022b4cec4cc2.jpg)  
(c) ETTh2  
Figure 4: Autoformer: test selective MSE vs. target coverage on three datasets, using the same test-split quantile protocol as Section 5. We compare TEM-only (ebm\_only), graph-only (graph\_only), and the full-coverage baseline (orig\_MSE). Across these datasets, the structural gate typically yields substantially lower selective error at matched nominal coverage.

![](images/b5f411632d075560e4a3e540a08222d24b69e677daa350f529c14ecf16f3f6b8.jpg)  
(a) Weather

![](images/c3a7e7c8f81ef2846fb1365525a4a94733e3fc8210a138eec6de9615dd2ea40c.jpg)  
(b) Electricity

![](images/503430ba52ccb7f4f77efafa806190fafdc265dc5b397f30bff8b0be489a3556.jpg)  
(c) ETTh2  
Figure 5: PatchTST: test selective MSE vs. target coverage on three datasets, using the same test-split quantile protocol as Section 5. We compare TEM-only (ebm\_only), graph-only (graph\_only), and the full-coverage baseline (orig\_MSE).

## D ADDITIONAL BASELINE COMPARISONS

We provide a supplementary comparison to two simple instance-level baselines (MC dropout and an error predictor) on two representative backbones across three datasets. The TEM comparison is already reported in the main results.

Table 6: (Supplementary comparison.) Test selective MSE on retained samples at nominal coverages on the test split (thresholds from test-split quantiles of raw scores; aligned with Table 1; lower is better). We compare our graph-only structural gate to MC dropout and an error predictor (instance-level baselines) on two representative backbones across three datasets (the TEM comparison is reported in the main table).  
PatchTST
<table><tr><td>Dataset</td><td>Method</td><td>50%</td><td>70%</td><td>90%</td></tr><tr><td rowspan="3">ETTh2</td><td>Graph (ours)</td><td>0.195</td><td>0.202</td><td>0.236</td></tr><tr><td>Error predictor</td><td>0.226</td><td>0.245</td><td>0.251</td></tr><tr><td>MC dropout</td><td>0.199</td><td>0.214</td><td>0.239</td></tr><tr><td rowspan="3">Electricity</td><td>Graph (ours)</td><td>0.172</td><td>0.180</td><td>0.198</td></tr><tr><td>Error predictor</td><td>0.183</td><td>0.185</td><td>0.208</td></tr><tr><td>MC dropout</td><td>0.174</td><td>0.187</td><td>0.202</td></tr><tr><td rowspan="3">Weather</td><td>Graph (ours)</td><td>0.134</td><td>0.138</td><td>0.155</td></tr><tr><td>Error predictor</td><td>0.140</td><td>0.149</td><td>0.156</td></tr><tr><td>MC dropout</td><td>0.158</td><td>0.160</td><td>0.160</td></tr></table>

Autoformer
<table><tr><td>Dataset</td><td>Method</td><td>50%</td><td>70%</td><td>90%</td></tr><tr><td rowspan="3">ETTh2</td><td>Graph (ours)</td><td>0.245</td><td>0.269</td><td>0.291</td></tr><tr><td>Error predictor</td><td>0.297</td><td>0.290</td><td>0.304</td></tr><tr><td>MC dropout</td><td>0.293</td><td>0.296</td><td>0.296</td></tr><tr><td rowspan="3">Electricity</td><td>Graph (ours)</td><td>0.263</td><td>0.276</td><td>0.286</td></tr><tr><td>Error predictor</td><td>0.281</td><td>0.284</td><td>0.287</td></tr><tr><td>MC dropout</td><td>0.279</td><td>0.284</td><td>0.288</td></tr><tr><td rowspan="3">Weather</td><td>Graph (ours)</td><td>0.190</td><td>0.201</td><td>0.224</td></tr><tr><td>Error predictor</td><td>0.214</td><td>0.220</td><td>0.227</td></tr><tr><td>MC dropout</td><td>0.196</td><td>0.205</td><td>0.226</td></tr></table>

## E THEORETICAL ANALYSIS

This appendix connects the implemented structural score to selective forecasting: learned relations → structura energy → forecast-error certificates → score ranking → selective risk. All statements analyze the energy in Eq. (6) without modifying the algorithm.

## E.1 Symmetrized structural energy

For $Z \in \mathbb { R } ^ { H \times D }$ (horizon × variables), write $E _ { \mathrm { s t r u c t } } ( Z , A )$ as in $\mathrm { E q . ~ } ( 6 )$ with dist $_ { i , j } ( Z ) = \| Z _ { : , i } - Z _ { : , j } \| _ { 2 } ^ { 2 }$ . Define symmetrized weights $\begin{array} { r } { W = \frac { 1 } { 2 } ( A + A ^ { \top } ) } \end{array}$

Lemma 1 (Symmetrization invariance). For any nonnegative A and trajectory $Z , E _ { \mathrm { s t r u c t } } ( Z , A ) = E _ { \mathrm { s t r u c t } } ( Z , W )$

Proof. Since dis $\begin{array} { r } { \mathrm { t } _ { i j } \ = \ \mathrm { d i s t } _ { j i } , \ \sum _ { i , j } A _ { i j } \mathrm { d i s t } _ { i j } \ = \ \frac { 1 } { 2 } \sum _ { i , j } ( A _ { i j } + A _ { j i } ) \mathrm { d i s t } _ { i j } \ = \ \sum _ { i , j } W _ { i j } \mathrm { d i s t } _ { i j } } \end{array}$ . Scaling by $1 / D ^ { 2 }$ is unchanged. □

For analysis, let $D _ { W } = \mathrm { d i a g } ( W { \bf 1 } )$ and $L _ { W } = D _ { W } - W \succeq 0$ . Then $\Vert Z \Vert _ { W } : = \sqrt { E _ { \mathrm { s t r u c t } } ( Z , A ) }$ is a seminorm on $\mathbb { R } ^ { H \times D }$ . If W is connected on variables, $\| Z \| _ { W } = 0$ exactly when all columns of $Z$ coincide (trajectories constant across variables); restricted to $\{ Z \in \mathbb { R } ^ { H \times D } : Z \mathbf { 1 } _ { D } = \mathbf { 0 } \} , \parallel \cdot \parallel _ { W }$ is a norm. Theorem 2 uses the seminorm triangle inequality on $\mathbb { R } ^ { \mathbf { \hat { H } } \times D } ;$ ; test-time scores are unchanged.

## E.2 Structural energy as an error certificate

Assumption 1 (graph-smooth ground truth). For ground-truth future Y , $E _ { \mathrm { s t r u c t } } ( Y , A ) ~ \leq ~ \epsilon$ for the (learned) adjacency A used at scoring time.

Theorem 2 (Structural-energy error certificate). Assume $\lambda _ { \operatorname* { m a x } } ( L _ { W } ) > 0$ . For any forecast $\hat { Y }$

$$
\| \hat { Y } - Y \| _ { F } ^ { 2 } \geq \frac { D ^ { 2 } } { 2 \lambda _ { \operatorname* { m a x } } ( L _ { W } ) } \left[ \sqrt { E _ { \mathrm { s t r u c t } } ( \hat { Y } , A ) } - \sqrt { \epsilon } \right] _ { + } ^ { 2 } , \qquad [ x ] _ { + } = \operatorname* { m a x } ( x , 0 ) .
$$

Proof. By the triangle inequality for $\lVert \cdot \rVert _ { W } , \lVert \hat { Y } - Y \rVert _ { W } \geq \left[ \lVert \hat { Y } \rVert _ { W } - \lVert Y \rVert _ { W } \right] _ { + }$ . Squaring and using $E _ { \mathrm { s t r u c t } } ( Y , A ) \leq \epsilon$ gives $E _ { \mathrm { s t r u c t } } ( \hat { Y } - Y , A ) \ge \left[ \sqrt { E _ { \mathrm { s t r u c t } } ( \hat { Y } , A ) - \sqrt { \epsilon } } \right] _ { + } ^ { 2 }$ . For residual $R = \hat { Y } - Y$ , pairwise expansion and $L _ { W } \succeq 0$ yield $\begin{array} { r } { E _ { \mathrm { s t r u c t } } ( R , A ) = \frac { 1 } { D ^ { 2 } } \sum _ { i j } W _ { i j } \| R _ { : , i } - R _ { : , j } \| _ { 2 } ^ { 2 } \leq \frac { 2 \lambda _ { \operatorname* { m a x } } ( L _ { W } ) } { D ^ { 2 } } \| R \| _ { F } ^ { 2 } } \end{array}$ . Combine the two displays. □

Corollary 1 (per-sample selective error). With $e _ { b } = \Vert \hat { Y } _ { b } - Y _ { b } \Vert _ { F } ^ { 2 } / ( H D ) \ \mathrm { ( E q . \ ( 1 ) ) }$ , Theorem 2 is equivalent to Eq. (7) in Section 4 (divide the Frobenius bound by HD). If $E _ { \mathrm { s t r u c t } } ( Y , A ) ~ = ~ 0 .$ , then $e _ { b } \geq$ $\begin{array} { r } { \frac { D } { 2 H \lambda _ { \operatorname* { m a x } } ( L _ { W } ) } E _ { \mathrm { s t r u c t } } ( \hat { Y } , A ) } \end{array}$ : large structural energy is a one-sided certificate on the same error used in selective risk, not an exact MSE predictor.

## E.3 Selective ranking and excess risk

On a test batch of size B, let per-sample errors $e _ { 1 } , \ldots , e _ { B } \ ( \mathrm { E q . \ ( 1 ) } )$ ) and target coverage $c \in ( 0 , 1 ] , k = \lfloor c B \rfloor$ . For abstention score $S = ( S _ { 1 } , \dots , S _ { B } )$ , let $\mathcal { S } _ { S } ( k )$ be the k samples with smallest scores and $\begin{array} { r } { R _ { S } ( k ) = \frac { 1 } { k } \sum _ { b \in S _ { S } ( k ) } e _ { b } } \end{array}$ Let $\begin{array} { r } { S ^ { * } ( k ) \in \arg \operatorname* { m i n } _ { | T | = k } \sum _ { b \in \mathcal { T } } e _ { b } } \end{array}$ and $\begin{array} { r } { R ^ { * } ( k ) = \frac { 1 } { k } \sum _ { b \in S ^ { * } ( k ) } e _ { b } } \end{array}$

Theorem 3 (Oracle ranking optimality). For every score S and every k, $R _ { S } ( k ) \geq R ^ { * } ( k )$ $I f e _ { i } < e _ { j } \Rightarrow S _ { i } < S _ { j }$ for all $i , j$ , then $R _ { S } ( k ) = R ^ { * } ( k )$ for all k.

Proof. Among k-subsets, $S ^ { * } ( k )$ minimizes $\textstyle \sum _ { b \in T } e _ { b }$ by definition. If S is strictly order-consistent with $e ,$ the lowest-k scores coincide with the lowest-k errors, so $S _ { S } ( k ) = S ^ { * } ( k )$ □

Theorem 4 (Excess selective risk under disagreement). Assume $0 \leq e _ { b } \leq E _ { \operatorname* { m a x } }$ for all b. Let $m _ { k } = | S _ { S } ( k ) \ \rangle$ $S ^ { * } ( k ) |$ and $\delta _ { k } = m _ { k } / k$ . Then $0 \leq R _ { S } ( k ) - R ^ { * } ( k ) \leq E _ { \operatorname* { m a x } } \delta _ { k }$

Proof. Let $F = { \cal S } _ { \cal S } ( k ) \backslash { \cal S } ^ { * } ( k )$ and $M = S ^ { * } ( k ) \setminus S _ { S } ( k ) ; | F | = | M | = m _ { k }$ . Canceling common samples, $R _ { S } ( k ) -$ $\begin{array} { r } { R ^ { * } ( \dot { k } ) = \frac { 1 } { k } ( \sum _ { i \in F } e _ { i } - \sum _ { j \in M } e _ { j } ) \le \frac { m _ { k } E _ { \operatorname* { m a x } } } { k } } \end{array}$ □

Theorem 3 matches coverage-constrained selective forecasting (Section 5): under fixed $\rho ,$ we sort raw scores and retain the lowest fraction. Theorem 4 bounds suboptimality by the fraction of mis-ranked acceptances at size k (finite-batch, empirical statement).

## E.4 Alignment objective and implementation

Training uses $\mathcal { L } _ { \mathrm { a l i g n } } = - r ( s , t )$ with $s _ { b } = E _ { \mathrm { s t r u c t } } ( \hat { Y } _ { b } , A )$ and default $t _ { b } = \log ( 1 + e _ { b } )$ . The Pearson objective does not directly optimize permutation agreement; it is a smooth surrogate encouraging positive score–risk association.

Lemma 5 (Monotone target). For $t ( e ) = \log ( 1 + e ) , e _ { i } < e _ { j } \iff t ( e _ { i } ) < t ( e _ { j } ) ,$ oracle error ranking is unchanged by the log target.

Proposition 6 (Pairwise ranking stability under approximate alignment). Suppose at evaluation time $t _ { b } ~ =$ $\alpha s _ { b } + \beta + \xi _ { b }$ with $\alpha > 0$ and $| \xi _ { b } | \le \eta . ~ I f s _ { j } - s _ { i } > 2 \eta / \alpha$ , then $t _ { j } > t _ { i }$ (hence $e _ { j } > e _ { i }$ when $t = \log ( 1 + e )$ and the map is invertible on the observed range).

$$
P r o o f . \ t _ { j } - t _ { i } = \alpha ( s _ { j } - s _ { i } ) + ( \xi _ { j } - \xi _ { i } ) \geq \alpha ( s _ { j } - s _ { i } ) - 2 \eta > 0 .
$$

Thus correlation alignment is consistent with the ranking objective under an approximately positive afine score– risk relation, but does not guarantee perfect ranking on finite batches. Diferentiable Pearson correlation: with $\tilde { s } _ { b } = s _ { b } - \bar { s } , \tilde { t } _ { b } = t _ { b } - \bar { t } ,$

$$
r ( s , t ) = \frac { \sum _ { b = 1 } ^ { B } \tilde { s } _ { b } \tilde { t } _ { b } } { \sqrt { \sum _ { b = 1 } ^ { B } \tilde { s } _ { b } ^ { 2 } + \varepsilon } \sqrt { \sum _ { b = 1 } ^ { B } \tilde { t } _ { b } ^ { 2 } + \varepsilon } } ,\tag{11}
$$

with fixed $\varepsilon > 0 ;$ gradients flow through $s _ { b }$ into the graph head and forecaster.

## E.5 Structure-blind scoring limits

A structure-blind score $\begin{array} { r } { S _ { \mathrm { m a r g } } ( Z ) = \sum _ { d = 1 } ^ { D } \phi _ { d } ( Z _ { : , d } ) } \end{array}$ depends only on per-channel summaries, not on cross-channel gaps.

Proposition 7 (Marginal scores need not detect relational deviation). There exist $Z ^ { ( 1 ) } , Z ^ { ( 2 ) }$ and nonnegative A such that $S _ { \mathrm { m a r g } } ( Z ^ { ( 1 ) } ) = S _ { \mathrm { m a r g } } ( Z ^ { ( 2 ) } )$ while $E _ { \mathrm { s t r u c t } } ( Z ^ { ( 1 ) } , A ) \neq E _ { \mathrm { s t r u c t } } ( Z ^ { ( 2 ) } , A )$

Proof. Take $D = 2$ with $A _ { 1 2 } = A _ { 2 1 } > 0$ . Let $x \in \mathbb { R } ^ { H }$ and a permutation P with $\phi _ { 1 } ( x ) = \phi _ { 2 } ( x ) = \phi _ { 1 } ( P x ) =$ $\phi _ { 2 } ( P x )$ . Set $Z ^ { ( 1 ) } = [ x \ : x ]$ and $Z ^ { ( 2 ) } = [ x \ : P x ]$ (columns). Then $S _ { \mathrm { m a r g } }$ matches, but $E _ { \mathrm { s t r u c t } } ( Z ^ { ( 2 ) } , A ) \propto \| x - P x \| _ { 2 } ^ { 2 }$ can exceed $E _ { \mathrm { s t r u c t } } ( Z ^ { ( 1 ) } , A ) = 0$ □

This concerns separable structure-blind rules; TEM is our learned instance-level comparator in experiments—we do not assume it equals $S _ { \mathrm { m a r g } } ,$ but the example motivates a distinct relational abstention axis.

## E.6 Scope and assumptions

A1. Adjacency entries are nonnegative; analysis uses $W = ( A + A ^ { \top } ) / 2$ (Lemma 1). A2. Theorem 2 assumes $\lambda _ { \operatorname* { m a x } } ( L _ { W } ) > 0$ and Assumption 1 for the same A deployed at test time (learned A may mis-specify coupling, regime (iii) in Section 5.6). A3. Theorems 3–4 are finite-batch statements for empirical selective risk at nominal k. A4. Proposition 6 is conditional on post-training approximate alignment, not implied in full generality by minimizing Pearson correlation on training batches. These results do not imply universal superiority of $E _ { \mathrm { s t r u c t } } ;$ when ground-truth smoothness is weak or error is predominantly marginal, certificates and relational scores are expected to be less informative, consistent with Section 5.6.

## F COMPUTE RESOURCES

All experiments were run with $\mathrm { P y }$ Torch and CUDA on a single NVIDIA L4 GPU (24GB per device). Software versions are pinned in environment.yml; default training schedules follow the project README.