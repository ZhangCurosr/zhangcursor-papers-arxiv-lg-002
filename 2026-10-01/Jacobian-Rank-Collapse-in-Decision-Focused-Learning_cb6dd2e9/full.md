# Jacobian Rank Collapse in Decision-Focused Learning

Aojie Yuan<sup>1</sup> Haiyue Zhang<sup>1</sup> Zijian Su<sup>2</sup>

<sup>1</sup>University of Southern California <sup>2</sup>University of Michigan

aojieyua@usc.edu haiyuez@usc.edu simoon@umich.edu

## Abstract

Decision-focused learning (DFL) trains predictors through downstream objectives, but a different loss need not provide an independent parameter-update direction. We characterize this restriction through the predictor Jacobian, using sparse index tracking to distinguish the covariance entries read by the optimizer from the parameter directions available to learning. Rank-one Jacobians make nonzero perexample gradients collinear; a conditional spectral bound describes near-collinearity. A batch-subspace characterization and counterexamples show why these local statements imply neither common minimizers nor collinear batch updates.

Experiments examine when geometry translates into decision quality. Across 38 one-parameter equity configurations, DFL gains over MSE remain below 1.8%; a 385-parameter conditional predictor also has pointwise rank one. In validation-tuned shortest-path and knapsack experiments, full-capacity SPO+ reduces mean regret by 11.6% and 10.6%, respectively; only knapsack survives correction across eight comparisons. The capacity contrast persists on fresh datasets across batch orders and training budgets. Holding expressivity fixed, invertible coordinate scaling lowers spectral effective rank and ordinary SGD gains; compensating for the scaling restores the original trajectories. Financial forward-target controls separate forecast accuracy from decision quality; a matched neural comparison finds no aggregate DFL advantage in the tested architecture. These findings distinguish local rank restrictions, coordinatedependent optimization and predictive accuracy. Predictor geometry helps explain available learning directions, while held-out decision quality remains the test of practical benefit.

(a) One local direction, two losses  
![](images/b1da74678fbe61082e55dcb55d98d98bc35195513831a6e6dd477421db6acd26.jpg)

(b) Full-capacity empirical gains  
![](images/a6971c77b5f59447c520839f327f3a0db086b029b7c71aa44c9f9409e10aa77c.jpg)  
Figure 1: Local geometry restricts directions; task gains require evidence. Left: rank-one parameter Jacobians map nonzero loss gradients onto one line, but batch directions can differ across examples. Right: full-capacity regret reductions in the controlled extension, with pointwise 95% paired-dataset bootstrap intervals over ten datasets. Only knapsack passes Holm correction across eight comparisons. Filled markers denote Holm-adjusted significance.

## 1 Introduction

When should a practitioner pay for decision-focused learning (DFL) rather than fit a predictor and optimize its outputs? Differentiable optimization and decision-aware surrogates offer several ways to train end to end (Elmachtoub and Grigas, 2022; Donti et al., 2017; Wilder et al., 2019; Mandi et al., 2024). Their benefit depends on both the downstream objective and the predictor: a decisionrelevant error direction is useful only if the model can express an update along it. We study this second restriction.

At a fixed input, the central object is the predictor Jacobian $\mathbf { J } = \partial \mathrm { v e c } ( \Sigma ) / \partial \theta$ . Both MSE and task gradients pass through $\mathbf { J } ^ { \top }$ before updating the parameters. At rank one, their nonzero images lie on one line (Proposition 4); a spectral bound controls angular separation when both the singularvalue gap and the gradients’ leading components permit it (Theorem 1). Rank collapse therefore excludes a new independent direction, but does not exclude different signs, stationary points or final decisions. We distinguish this exact local statement from the empirical question of whether DFL improves test performance.

Why finance, and what should transfer? Sparse index tracking separates three restrictions: selecting assets determines which covariance entries the optimizer reads; the predictor maps output gradients into parameter directions; aggregation combines directions across examples. These operations need not impose the same geometry.

Across 38 one-parameter equity configurations, DFL gains over MSE remain below 1.8%. Yet a 385-parameter conditional predictor also has pointwise rank one: parameter count alone misses the bottleneck. The financial baseline reconstructs trailing covariance rather than forecasting future covariance (Section 4). Figure 1 connects the mechanism to the controlled evidence; Figure 2 shows the training paths.

We test transfer in shortest-path and knapsack tasks using shared initializers, separate validation and exact decision oracles. A capacity sweep first measures the performance contrast; an invertible reparameterization then holds expressivity fixed to examine coordinate effects. Finally, forward covariance baselines test whether the financial findings depend on a reconstruction target. This sequence separates three questions: which directions are available, how optimization uses them, and whether the resulting predictions improve decisions.

## Contributions.

1. A geometric account from output support to batch updates. A support-energy inequality, conditional spectral bound and batch-subspace characterization separate what a decision loss observes from the directions a predictor can follow. Counterexamples rule out inferring common minimizers or collinear batch updates from pointwise rank one.

2. Controlled tests beyond the financial setting. Two synthetic tasks use matched initial predictors, independently selected hyperparameters and saved models for direct verification. A freshdata follow-up varies minibatch order and training budget while allowing either loss to retain the unchanged baseline. An invertible reparameterization then holds expressivity fixed: spectral rank and ordinary SGD gains fall together, while compensation restores the original updates. This distinguishes a coordinate-dependent optimization effect from lost expressivity.

3. Financial evidence with explicit scope. Equity comparisons separate parameter count, local geometry and observed decision quality. A chronological forward-target control compares futurecovariance MSE, task validation, Ledoit–Wolf and EWMA; better covariance forecasts need not yield better tracking. A matched neural forward-target comparison reports a small aggregate difference and frequent validation selection of the unchanged initializer. Dynamic-selection experiments and negative results delimit these claims. Regularization coefficients and benefit thresholds remain validation choices rather than consequences of rank collapse.

## 2 Related Work

Sparse Index Tracking. Sparse portfolio construction has a rich history (Kolm et al., 2014): evolutionary heuristics (Beasley et al., 2003), mixed-integer programming (Canakgoz and Beasley, 2009), $\ell _ { 1 }$ -penalized formulations (Benidis et al., 2018; Brodie et al., 2009), and cardinalityconstrained optimization (Xu et al., 2016). These approaches address sparse portfolio construction; our focus is on training covariance predictors through the resulting decisions.

Decision-Focused Learning. DFL optimizes downstream objectives using differentiable QPs (Donti et al., 2017; Amos and Kolter, 2017; Agrawal et al., 2019), SPO+ (Elmachtoub and Grigas, 2022), combinatorial surrogates (Wilder et al., 2019; Vlastelica et al., 2020), or implicit differentiation (Blondel et al., 2022; Paulus et al., 2024). PG losses have asymptotic decision-quality guarantees under misspecification (Gupta and Huang, 2024); PEAR characterizes regret gradients through active-constraint tangent spaces and local curvature (Lee et al., 2026). Other work studies regret decomposition (Aldridge, 2026), online DFL (Capitaine et al., 2026), and prediction inflation in portfolios (Wang and Hasuike, 2026). Our analysis concerns a different stage: the predictor Jacobian that maps these output-space signals into trainable parameter directions.

Portfolio Optimization and Covariance Estimation. End-to-end portfolio construction (Butler and Kwon, 2023; Zhang et al., 2020; Kim et al., 2025) and high-dimensional covariance estimation (Ledoit and Wolf, 2004; Fan et al., 2013; Friedman et al., 2008; Engle, 2002) are both active areas. Concurrent work by Jeon et al. (2026) applies DFL to sparse tangent portfolios and reports gains in larger universes—complementary to our study of local gradient geometry and low-capacity predictors.

## 3 Problem Formulation

Let $\mathbf { r } _ { t } ~ \in ~ \mathbb { R } ^ { N }$ denote asset returns, $\mathbf { w } _ { \mathrm { i d x } }$ the index weights, and $\Sigma \ : = \ : \mathrm { C o v } ( \mathbf { r } _ { t } )$ . Tracking error minimization reduces to min $\mathsf { i } _ { \mathbf { W } } \bigl ( \mathbf { W } - \mathbf { W } _ { \mathrm { i d x } } \bigr ) ^ { \top } \Sigma \bigl ( \mathbf { W } - \mathbf { W } _ { \mathrm { i d x } } \bigr )$ . The joint selection-and-weighting problem is NP-hard (Beasley et al., 2003), so we decompose into two stages.

Stage 1: Subset selection. Choose ${ \mathcal { S } } \subseteq \{ 1 , \ldots , N \}$ with $| S | = K$ via market-cap ranking (default) or tracking-score: $s _ { i } = w _ { i } ^ { \mathrm { i d x } } \cdot ( \hat { \Sigma } _ { i , : } \mathbf { w } _ { \mathrm { i d x } } ) ^ { 2 } / \hat { \Sigma } _ { i i }$ . Under tracking-score selection, $s$ depends on ${ \hat { \Sigma } } ,$ , so off-block estimation errors propagate to stock selection.

Stage 2: Weight optimization (QP). Given $s ,$ solve:

$$
\operatorname* { m i n } _ { \mathbf { w } _ { \mathcal { S } } } ~ \mathbf { w } _ { \mathcal { S } } ^ { \top } \hat { \boldsymbol { \Sigma } } _ { \mathcal { S } \mathcal { S } } \mathbf { w } _ { \mathcal { S } } - 2 \mathbf { w } _ { \mathcal { S } } ^ { \top } \hat { \boldsymbol { \Sigma } } _ { \mathcal { S } ; \mathbf { w } _ { \mathrm { i d x } } } ~ \mathrm { s . t . } ~ \mathbf { 1 } ^ { \top } \mathbf { w } _ { \mathcal { S } } = 1 , \mathbf { w } _ { \mathcal { S } } \geq 0 .\tag{1}
$$

Selection $\boldsymbol { \mathcal { S } }$ is detached from the graph (Appendix C.10). The $\mathrm { Q P }$ reads at most $2 N K - K ^ { 2 }$ of $N ^ { 2 }$ entries; this asymmetry drives our gradient alignment analysis.

![](images/52a59c2d0ee509bd441ed3f1be7b5282fd1ac2a71bb5c76386926f8b9a2fbbb3.jpg)  
Figure 2: Two objectives, one predictor. MSE directly supervises covariance prediction; task loss differentiates through the QP. Both reach θ through $\mathbf { J } ^ { \top }$ , with selection detached. The toy example illustrates sparse weighting for an equal-weight index with identity covariance.

## 4 Methodology

## 4.1 Two-Stage Baseline and DFL

The standard two-stage approach trains under Frobenius loss $\mathcal { L } _ { \mathrm { M S E } } = N ^ { - 2 } \lVert \hat { \boldsymbol { \Sigma } } ( \mathbf { x } _ { t } ; \boldsymbol { \theta } ) - { \boldsymbol { \Sigma } } _ { \mathrm { r e a l i z e d } , t } \rVert _ { F } ^ { 2 }$ where $\Sigma _ { \mathrm { r e a l i z e d } , t }$ is the sample covariance of the preceding 63 trading days. In the inspected training implementation this is a trailing reconstruction target, not a future covariance label. For shrinkage families it is also the input sample covariance, so this baseline should not be interpreted as an optimally specified supervised forecasting model. In DFL, we embed the QP as a differentiable layer:

$$
\mathcal { L } _ { \mathrm { t a s k } } = \frac { 2 5 2 } { H } \sum _ { s = 0 } ^ { H - 1 } \bigl ( \mathbf { w } ^ { * } ( \hat { \Sigma } _ { t } ) ^ { \top } \mathbf { r } _ { t + s } - r _ { t + s } ^ { \mathrm { i d x } } \bigr ) ^ { 2 } + \frac { \gamma } { 2 } \| \mathbf { w } _ { t } ^ { * } - \mathbf { w } _ { t - 1 } ^ { * } \| _ { 1 } ,\tag{2}
$$

where t denotes the first return in the forward training window, $\mathbf { w } ^ { * } ( \hat { \Sigma } _ { t } )$ is the QP solution, $H = 2 1$ trading days, and $\gamma$ controls turnover. Gradients flow through the KKT conditions (Amos and Kolter, 2017) into θ. Selection S is detached from the graph.

## 4.2 Block-Diagonal DFL (BD-DFL)

Under detached selection, the task gradient is restricted to the current support $\tau$ . Errors outside that support remain unpenalized by the task objective and may matter after selection changes. BD-DFL adds relative MSE regularization:

$$
{ \mathcal { L } } _ { \mathrm { B D - D F L } } = { \mathcal { L } } _ { \mathrm { t a s k } } + \beta \cdot \| { \hat { \boldsymbol { \Sigma } } } ( \theta ) - { \boldsymbol { \Sigma } } _ { \mathrm { r e a l i z e d } } \| _ { F } ^ { 2 } / ( \| { \boldsymbol { \Sigma } } _ { \mathrm { r e a l i z e d } } \| _ { F } ^ { 2 } + 1 0 ^ { - 1 2 } ) ,\tag{3}
$$

where $\beta = 0$ recovers pure DFL and large $\beta$ emphasizes covariance fit. A selection-robustness perspective motivates this penalty: entries outside today’s support may enter tomorrow’s QP. Classical DRO provides related regularization principles (Mohajerin Esfahani and Kuhn, 2018; Blanchet and Murthy, 2019), but does not establish the swap-dependent coefficient claimed for this particular model. We therefore choose $\beta$ empirically and state the limitation in Appendix A.7.

![](images/5b389af0347682dce3b5005381fb623fe7a3f910f7ed0c37ef2cbbd6653155d7.jpg)  
Figure 3: Where gradient directions are lost. (a) The task-relevant support has $2 N K - K ^ { 2 }$ entries (36% here). (b) A rank-one predictor Jacobian maps nonzero gradients onto one line, with either sign. (c) With multiple active singular directions, angular separation can survive; improved task performance is possible, not guaranteed.

Proposition 1 (Uniform control of selected inputs). For every error matrix E and subset ${ \mathcal { S } } ,$ $\| E _ { S S } \| _ { \mathrm { o p } } \leq \| E \| _ { F }$ and $\| E _ { S , : } w _ { \mathrm { i d x } } \| _ { 2 } \ \leq \ \| E \| _ { F } \| w _ { \mathrm { i d x } } \| _ { 2 }$ . Thus a Frobenius penalty controls the two perturbed QP inputs uniformly over subsets. Converting this into a decision or regret bound requires solution-stability assumptions.

## 4.3 Covariance Models

Five architectures of increasing capacity, each trained under MSE and DFL: Shrinkage $( 1 \mathsf { p } ) \colon \hat { \Sigma } =$ $( 1 - \alpha ) S + \alpha \mu I$ (Ledoit and Wolf, 2004); Factor $( O ( N K _ { f } ) \mathbf { p } ) \colon \hat { \Sigma } = B F B ^ { \top } + D$ ; Neural $( O ( N r ) { \mathfrak { p } } ) ;$ $\begin{array} { r } { \mathbf { M } \mathbf { L } \mathbf { P }  L L ^ { \top } + D ; } \end{array}$ Structured (12p): learnable per-sector targets; Conditional (385p): regimeadaptive $\alpha _ { t } = \sigma ( \mathrm { M L P } ( \mathbf { z } _ { t } ) )$ . Training: Adam $( \ln 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , clip 1.0), early stopping on validation TE, cvxpylayers (Agrawal et al., 2019) for QP differentiation. Details in Appendix B.2.

## 5 Local Geometry and Batch Limits

Throughout this section, g denotes an output-space loss gradient, J a pointwise predictor Jacobian, and $J _ { \mathrm { s t a c k } }$ the Jacobian stacked over examples. We reserve $r _ { \mathrm { e f f } }$ for spectral entropy rank and $d _ { \mathrm { p r o x y } } = d h$ for the archived heterogeneity heuristic. Input sensitivity uses a different Jacobian, $J _ { x } = \partial f _ { \theta } ( x ) / \partial x$ , whose spectral rank is denoted $r _ { \mathrm { i n } }$ . Neither $r _ { \mathrm { i n } }$ nor $d _ { \mathrm { p r o x y } }$ is interchangeable with the spectral rank of J or $J _ { \mathrm { s t a c k } }$

## 5.1 Gradient Alignment Theory

The task gradient decomposes as:

$$
\nabla _ { \boldsymbol { \theta } } \ell = \underbrace { \frac { \partial \ell } { \partial \mathbf { w } ^ { * } } } _ { \mathrm { t a s k } } \cdot \underbrace { \frac { \partial \mathbf { w } ^ { * } } { \partial \boldsymbol { \hat { \Sigma } } } } _ { \mathrm { Q P J a c o b i a n } } \cdot \underbrace { \frac { \partial \boldsymbol { \hat { \Sigma } } } { \partial \boldsymbol { \theta } } } _ { \mathrm { m o d e l } } .\tag{4}
$$

The QP Jacobian $\partial \mathbf { w } ^ { * } / \partial \hat { \Sigma }$ , derived from KKT differentiation (Amos and Kolter, 2017), acts as a support-restricted linear map: it zeroes out entries $\hat { \Sigma } _ { i j }$ where both $i \not \in { \mathcal { S } }$ and $j \not \in { \mathcal { S } } ,$ , since the QP reads only $\hat { \Sigma } _ { S S }$ and $\hat { \Sigma } _ { S , : } \mathbf { w } _ { \mathrm { i d x } }$ . MSE can place gradient energy on all entries. A small support limits alignment only when little MSE energy concentrates on that support.

Proposition 2 (Gradient Alignment Bound). Let ${ \mathcal { S } } \subset \{ 1 , \ldots , N \} , \ | S | = K$ . Define the taskrelevant support $\mathcal { T } = \{ ( i , j ) : i \in \mathcal { S } o r j \in \mathcal { S } \}$ , with $| \mathcal { T } | = 2 N K - K ^ { 2 }$ . For nonzero gradients, suppose the normalized MSE energy obeys $\mathbb { E } [ \| \mathbf { g } _ { \mathrm { M S E } } | _ { T } \| ^ { 2 } / \| \mathbf { g } _ { \mathrm { M S E } } \| ^ { 2 } ] \le | T | / N ^ { 2 }$ . Then:

$$
\begin{array} { r } { \mathbb { E } | \cos ( { \bf g } _ { \mathrm { M S E } } , { \bf g } _ { \mathrm { t a s k } } ) | \leq \sqrt { \frac { 2 K } { N } - \frac { K ^ { 2 } } { N ^ { 2 } } } \approx \sqrt { \frac { 2 K } { N } } \ ( K \ll N ) . } \end{array}\tag{5}
$$

This support-only inequality is sharp for unrestricted ambient vectors (Appendix $\scriptstyle \mathbf { A . 4 } ) ;$ attaining it with gradients of the tracking QP is a separate question.

Proposition 3 (Decision Invariance). The $Q P ( 1 )$ depends on $\hat { \Sigma }$ only through $\hat { \Sigma } _ { S S } a n d \hat { \Sigma } _ { S , : } \mathbf { w } _ { \mathrm { i d x } } ; o f f \mathrm { - }$ block errors are irrelevant to both the solution and regret (Proposition 7; proofs in Appendix A.5).

This argument concerns the support of the output gradient. Its concentration within the selected block and its image under the predictor Jacobian are separate quantities; neither follows from cardinality alone.

## 5.2 Predictor Jacobian and Gradient Alignment

The support bound concerns covariance-space geometry. We now characterize its image in parameter space; experimental capacity proxies are considered separately in Section 7.5.

Proposition 4 (Jacobian Rank Collapse). Let $ { \mathbf { J } } \in  { \mathbb { R } } ^ { N ^ { 2 } \times d }$ be the Jacobian ofa differentiable covariance predictor. For two nonzero parameter gradients,

$$
\cos ( \nabla _ { \boldsymbol { \theta } } \ell _ { \mathrm { M S E } } , \nabla _ { \boldsymbol { \theta } } \ell _ { \mathrm { t a s k } } ) = \frac { \mathbf { g } _ { \mathrm { M S E } } ^ { \top } \mathbf { J } \mathbf { J } ^ { \top } \mathbf { g } _ { \mathrm { t a s k } } } { \left\| \mathbf { J } ^ { \top } \mathbf { g } _ { \mathrm { M S E } } \right\| \left\| \mathbf { J } ^ { \top } \mathbf { g } _ { \mathrm { t a s k } } \right\| } .\tag{6}
$$

If rank $\mathbf { \partial } ( \mathbf { J } ) = 1 ,$ , this cosine belongs to $\{ - 1 , + 1 \}$ , regardless of the covariance-space angle.

Proof sketch. Write $\mathbf { J } = \sigma \mathbf { u } \mathbf { v } ^ { \top }$ . Then $\mathbf { J } ^ { \top } \mathbf { g } = \sigma ( \mathbf { u } ^ { \top } \mathbf { g } ) \mathbf { v }$ , so both nonzero images are scalar multiples of v. Their signs and zeros can differ. Equality of minimizers requires further assumptions (Appendix A.1). □

The following theorem gives the exact form of parameter-space alignment and bounds how quickly it collapses to ±1 as the Jacobian approaches rank one.

Theorem 1 (Spectral Gap Controls Gradient Alignment). Let $\mathbf { J } = U S V ^ { \top } \in \mathbb { R } ^ { N ^ { 2 } \times d }$ be the Jacobian of $\mathrm { v e c } ( \Sigma ( \pmb { \theta } ) )$ , with $r = { \mathrm { r a n k } } ( \mathbf { J } ) \geq 1$ , thin factors $U \in \mathbb { R } ^ { N ^ { 2 } \times r }$ and $V \in \mathbb { R } ^ { d \times r }$ having orthonormal columns, and ${ \cal S } = \mathrm { d i a g } ( \sigma _ { 1 } , . . . , \sigma _ { r } )$ . Assume both parameter gradients and both leading components ${ \bf u } _ { 1 } ^ { \top } { \bf g }$ are nonzero, and set $\sigma _ { 2 } = 0$ when $r = 1$ . Then:

1. (Exact form.) Parameter-space alignment is the alignment of the σ-weighted coordinates of the two gradients along the singular directions:

$$
\cos ( \nabla _ { \pmb { \theta } } \ell _ { \mathrm { M S E } } , \nabla _ { \pmb { \theta } } \ell _ { \mathrm { t a s k } } ) = \cos ( S U ^ { \top } \mathbf { g } _ { \mathrm { M S E } } , S U ^ { \top } \mathbf { g } _ { \mathrm { t a s k } } ) .\tag{7}
$$

2. (Near-rank-one collapse.) For each gradient let $\rho =$ min $\{ 1 , ( \sigma _ { 2 } / \sigma _ { 1 } ) / | \cos ( \mathbf { g } , \mathbf { u } _ { 1 } ) | \}$ , and let Θ = arcsin ρ<sub>MSE</sub> + arcsin $\rho _ { \mathrm { t a s k } } . \mathrm { \ } I f \Theta < \pi / 2 ,$ , then

$$
| \cos ( \nabla _ { \theta } \ell _ { \mathrm { M S E } } , \nabla _ { \theta } \ell _ { \mathrm { t a s k } } ) | \ge \cos \Theta , \qquad \mathrm { s i g n } \cos ( \cdot , \cdot ) = \mathrm { s i g n } \big [ ( \mathbf { u } _ { 1 } ^ { \top } \mathbf { g } _ { \mathrm { M S E } } ) ( \mathbf { u } _ { 1 } ^ { \top } \mathbf { g } _ { \mathrm { t a s k } } ) \big ] .\tag{8}
$$

3. (Endpoints.) $\sigma _ { 2 } { = } 0$ gives | cos |=1 (Proposition $4 ) ; \mathbf { J } \mathbf { J } ^ { \top } = c I$ with $c > 0$ preserves outputspace alignment. The expected support bound then transfers only under Proposition 2’s energy assumption.

Proof sketch. $\nabla _ { \pmb { \theta } } \ell = V ( S U ^ { \top } \mathbf { g } )$ with $V ^ { \top } V ~ = ~ I$ gives $( 7 ) ;$ each $S U ^ { \top } \mathbf { g }$ lies within angle arcsin $\rho \mathrm { o f \pm e _ { 1 } }$ , and the angle is a metric on the sphere (Appendix A.1). □

Rank, angles, and a proxy are different quantities. The spectral effective rank is $r _ { \mathrm { e f f } } \ =$ $\begin{array} { r } { \exp ( - \sum _ { i } p _ { i } \log p _ { i } ) } \end{array}$ , where $p _ { i } = \sigma _ { i } ^ { 2 } / \sum _ { j } \sigma _ { j } ^ { 2 }$ . For a zero Jacobian we use the convention $r _ { \mathrm { e f f } } = 0$ Concentration of this distribution at one forces $\sigma _ { 2 } / \sigma _ { 1 }  0$ , but does not ensure that a particular task gradient has a nonzero leading component. The empirical quantity $d _ { \mathrm { p r o x y } } = d h$ is a separate heterogeneity-based proxy; no equality between it and $r _ { \mathrm { e f f } }$ is assumed. Corollary 2 requires the two projected gradients to be non-collinear, not merely rank greater than one.

## 5.3 From a Single Example to a Batch

Let $f _ { i } ( \theta )$ be the prediction for example i and $J _ { i }$ its Jacobian. A batch loss has gradient $G =$ $\sum _ { i } J _ { i } ^ { \top } g _ { i }$ . The relevant parameter subspace is therefore shared across examples, rather than determined by the rank of any one $J _ { i }$

Proposition 5 (Batch gradient subspace). Let $J _ { \mathrm { s t a c k } } = [ J _ { 1 } ^ { \top } , \ldots , J _ { n } ^ { \top } ] ^ { \top } \neq 0$ . Every batch gradient lies in $\mathcal { V } = \mathrm { r a n g e } ( J _ { \mathrm { s t a c k } } ^ { \top } )$ , whose dimension is $\mathrm { r a n k } ( J _ { \mathrm { s t a c k } } )$ . All possible nonzero batch gradients are collinear if and only if this rank is one. In particular, rank one for each $J _ { i }$ is insufficient unless their nonzero row spaces share a common line.

Proof. Stacking the output gradients gives $G = J _ { \mathrm { s t a c k } } ^ { \top } [ g _ { 1 } ^ { \top } , \ldots , g _ { n } ^ { \top } ] ^ { \top }$ . Its possible values form exactly V. A nonzero vector space contains only collinear pairs precisely when its dimension is one. Here “possible” ranges over differentiable output losses at the fixed parameter value; it does not assert that a particular pair of losses realizes every direction. □

Two counterexamples. A scalar predictor $f ( \theta ) = \theta$ has rank one, yet losses $( f - 1 ) ^ { 2 }$ and $( f - 2 ) ^ { 2 }$ have different minimizers. For a batch, take $f _ { 1 } ( \theta ) = \theta _ { 1 }$ and $f _ { 2 } ( \theta ) = \theta _ { 2 } . { \mathrm { ~ A t ~ } } \theta = 0$ , the two losses $\frac { 1 } { 2 } [ ( f _ { 1 } + 1 ) ^ { 2 } + ( f _ { 2 } + 1 ) ^ { 2 } ]$ and $\frac { 1 } { 2 } [ ( f _ { 1 } + 1 ) ^ { 2 } + ( f _ { 2 } - 1 ) ^ { 2 } ]$ have orthogonal gradients $( 1 , 1 )$ and $( 1 , - 1 )$ although each example Jacobian has rank one. Appendix A.2 gives the full construction and the covariance-model interpretation.

This distinction matters for conditional shrinkage: $\Sigma ( x ; \theta )$ depends on a scalar $\alpha ( x ; \theta )$ for each $x ,$ but $\nabla _ { { \boldsymbol { \theta } } } \alpha ( { \boldsymbol { x } } ; { \boldsymbol { \theta } } )$ can change direction across examples. A pointwise rank-one measurement does not establish a one-dimensional training problem. The results concern raw Euclidean gradients at a fixed parameterization; adaptive optimizer histories and changes in θ require additional analysis.

A measured architectural bottleneck. Figure 4 tests this distinction directly. The conditional predictor’s 385 parameters produce a rank-one covariance Jacobian at each measured input. The structured model has several active directions, but the near-rank-one bound is vacuous for its sampled task gradients. Appendix D.2 reports the numerical scope; neither observation is a test-loss guarantee.

## 6 Experimental Setup

## 6.1 Data

Top 100 S&P 500 stocks by market cap (N=97 after ≥90% coverage filter), daily returns 2006– 2025. Rolling-window protocol: 3-year train, 6-month validation, 1-year test, 9 folds covering 2016–2025 in the primary neural evaluation. Features: 21-day volatility, 63-day momentum, sector, market-cap quintile. Scaling: S&P 500 full (N=478, 5 folds), S&P 1500 (N=1258, 8 folds), 20- year backtest (N=100, 17 folds). Cross-market: FTSE 100, Nikkei 225, Euro Stoxx 50, ASX 200, Hang Seng. Cross-domain: NOAA weather stations (N∈{100, 189, 415}), EPA PM2.5 monitors $( N { \in } \{ 1 2 4 , 1 3 2 \} )$ ). Unless stated otherwise, financial paired comparisons use Wilcoxon signed-rank tests across folds: one-sided for directional hypotheses and two-sided for difference tests. Equivalence uses TOST with the stated margin. Correlation tests, synthetic dataset-level inference and descriptive forward-target controls are identified separately.

![](images/971d75248be2dda47b56230dfb946518194e56b0d2801270fe94cce1edabe415.jpg)

![](images/4107d506c91b8be0b92f07cb25135d86456c63a0eb90150df1c110d9085739b7.jpg)  
Figure 4: Pointwise Jacobian measurements on 54 inputs per model class. A 385-parameter conditional predictor is numerically rank one because its output passes through a scalar shrinkage intensity. Its gradient direction can still vary across inputs. These measurements distinguish parameter count from local rank; they do not measure the stacked batch rank.

## 6.2 Comparison Units and Evidence Provenance

The unit of comparison is a matched fold or seed within one experiment, not an individual daily return. Chronological folds share training history and may be dependent. We report their variability descriptively; unadjusted signed-rank p-values do not by themselves establish generalization across markets or tasks. A non-significant difference is not evidence of equivalence; only comparisons with an explicit TOST margin receive an equivalence interpretation. Relative changes are computed from unrounded source values, so they need not equal ratios of rounded table entries.

The release distinguishes three records: corrected equity evaluations with bounded test windows; an archived diagnostic scoring record predating those corrections; and separate spatial and combinatorial experiments. The 33 archived diagnostic cases are not an independent validation of the corrected harness. Configurations can share folds and data, so the reported count of paired folds is not a count of independent datasets. Appendix B.1 maps evidence types and reproducibility scope.

## 6.3 Baselines

We compare: Naive (top-K by index weight), K-only LW (K×K covariance only), Ledoit-Wolf (analytical shrinkage (Ledoit and Wolf, 2004)), POET (factor + adaptive thresholding (Fan et al., 2013)), GraphicalLasso $( \ell _ { 1 } { \cdot } \mathrm { p e n a l i z e d }$ precision (Friedman et al., 2008)), ValTuned (grid search on validation TE), SPO+ (Elmachtoub and Grigas, 2022), LODL (locally optimized decision losses (Shah et al., 2022)), and MSE-trained factor/neural models. Primary metric: annualized TE = std $( \mathbf { w } ^ { \top } \mathbf { r } _ { t } - r _ { t } ^ { \mathrm { { i d x } } } ) \times \sqrt { 2 5 2 }$

## 7 Results

## 7.1 Small Gains in One-Parameter Equity Models

Main finding: DFL adds a ∼3,300× training-cost overhead with little measured benefit in the tested low-capacity settings.

At N=100, fitted one-parameter models achieve similar test performance across different training objectives. This is an empirical result, complementary to the local collinearity statement: the theorem restricts available directions, whereas test-loss equivalence must be measured.

Table 1: One-parameter comparison at $N { = } 1 0 0 .$ . ValTuned, SPO+ and DFL meet the TOST criterion $( \delta = 2 0 \ : \mathrm { b p s } )$ in all 12 pairwise comparisons (three pairs, four sparsity levels; max $| d | = 0 . 6 6 )$ . POET and GLasso are different model families, shown for reference. At $K = 5 0 ( K / N = 0 . 5 2 )$ , their ordering changes and the $\sqrt { 2 K / N }$ bound is vacuous; no equivalence with these baselines is claimed.
<table><tr><td>K</td><td>LW</td><td>POET</td><td>GLasso</td><td>ValTuned</td><td>SPO+</td><td>DFL</td></tr><tr><td>5</td><td>0.0860</td><td>0.0765</td><td>0.0768</td><td>0.0773</td><td>0.0766</td><td>0.0766</td></tr><tr><td>10</td><td>0.0511</td><td>0.0457</td><td>0.0461</td><td>0.0458</td><td>0.0458</td><td>0.0458</td></tr><tr><td>20</td><td>0.0331</td><td>0.0267</td><td>0.0269</td><td>0.0269</td><td>0.0269</td><td>0.0268</td></tr><tr><td>50</td><td>0.0096</td><td>0.0104</td><td>0.0124</td><td>0.0088</td><td>0.0088</td><td>0.0089</td></tr></table>

Table 2: Higher-capacity comparison. $N { = } 4 7 8 ,$ 5 folds, one evaluation window. d: Cohen’s d on per-fold relative change; p: one-sided Wilcoxon, for which 0.031 is the floor at $n { = } 5$ (i.e. 5/5 folds).
<table><tr><td>Model</td><td>K MSE TE</td><td>DFL TE</td><td>d</td><td>p</td></tr><tr><td>GLasso</td><td>20</td><td>0.0575</td><td></td><td></td></tr><tr><td>POET</td><td>20</td><td>0.0503</td><td></td><td></td></tr><tr><td>Shrink. (1p)</td><td>20</td><td>0.0517 0.0514</td><td>-1.07</td><td>0.031*</td></tr><tr><td>Cond. (385p)</td><td>20</td><td>0.0454 0.0448</td><td>-0.83</td><td>0.062</td></tr><tr><td>Struct. (12p)</td><td>20</td><td>0.0513 0.0508</td><td>-1.60</td><td>0.031*</td></tr><tr><td>Struct. (12p)</td><td>50</td><td>0.0309</td><td>0.0305 -1.03</td><td>0.062</td></tr></table>

Table 1 reports equivalence under TOST with a 20-bps margin for all 12 pairwise comparisons among ValTuned, SPO+, and DFL. Each improves on Ledoit–Wolf by 7–19%, so fitting the parameter matters even when these objectives yield similar test errors. POET and GLasso are separate model families; their agreement at smaller K is empirical. LODL also matches DFL, indicating that explicit KKT differentiation is not necessary to obtain this performance. Higher-capacity models show different outcomes (Table 2); the neural gain is significant at $K \leq 3 0$ but absent at $K { = } 5 0$ (Appendix C.1).

## 7.2 Architecture and Loss Choice at Larger Scale

## Mainfinding: Architecture changes can matter more than the choice oftraining loss.

$\mathrm { A t } \ N { = } 4 7 8$ , we compare changes in predictor architecture with changes in training loss (Table 2).

Table 2 separates architectural improvements from within-model loss changes. GLasso has 14.3% higher TE than POET, while the MSE-trained conditional predictor has 9.8% lower TE than POET. Switching from MSE to DFL yields smaller relative improvements: 0.68%, 1.07% and 1.34% for the 1-, 12- and 385-parameter models, respectively. These parameter counts do not order pointwise Jacobian rank: the conditional predictor remains rank one. The one-parameter model improves in all five folds, but its 0.68% gain remains within the range observed at smaller scale. With five folds, $p = 0 . 0 3 1$ is the smallest attainable one-sided Wilcoxon value; consistency of sign should therefore be read alongside effect size.

## 7.3 Cross-Market Evidence for One-Parameter Models

Across six markets and the reported scale variants, one-parameter gains remain modest (Table 3). Across all 38 configurations, gains lie in $[ - 0 . 5 7 \% , + 1 . 7 6 \% ]$ over $K / N \in [ 0 . 0 0 8 , 0 . 4 3 ]$ , with no detected monotone association with sparsity $( \rho = - 0 . 1 6 , p = 0 . 3 5 )$ . This observation is empirical; rank-one collinearity does not imply it. The structured model gives larger relative TE reductions on ASX (2.95%), Hang Seng (2.77%) and S&P 1500 (2.41%, all $p ~ \leq ~ 0 . 0 0 8 )$ , while gains on Euro Stoxx, Nikkei and FTSE are below 1%. Architecture alone does not predict the size of the

Table 3: Cross-market tracking gains. Positive $\Delta \%$ denotes lower TE than matched MSE. The listed comparisons total 81 paired folds; the broader 38-configuration analysis contains 316 paired folds, with shared data and histories. Nikkei, Euro Stoxx, ASX and Hang Seng use one common four-market configuration with eight folds per market. $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1$ ; one-sided Wilcoxon for the structured-model comparison.
<table><tr><td>Market</td><td>N</td><td>K</td><td>Shrink ∆%</td><td>Struct ∆%</td><td>Pstruct</td></tr><tr><td>S&amp;P 500 (20y)</td><td>100</td><td>20</td><td>+0.29</td><td>+0.40</td><td>0.020*</td></tr><tr><td>FTSE 100</td><td>91</td><td>10</td><td>-0.09</td><td>+0.37</td><td>0.039*</td></tr><tr><td>Nikkei 225</td><td>218</td><td>10</td><td>+0.17</td><td>+0.87</td><td>0.027*</td></tr><tr><td>Euro Stoxx 50</td><td>47</td><td>10</td><td>+0.29</td><td>+0.95</td><td>0.008**</td></tr><tr><td>ASX 200</td><td>159</td><td>10</td><td>+0.98</td><td>+2.95</td><td>0.004* **</td></tr><tr><td>Hang Seng</td><td>62</td><td>10</td><td>+1.76</td><td>+2.77</td><td>0.008**</td></tr><tr><td>S&amp;P 1500</td><td>1258</td><td>10</td><td>+0.32</td><td>+2.41</td><td>0.004**</td></tr><tr><td>S&amp;P 1500</td><td>1258</td><td>20</td><td>+0.44</td><td>+2.21</td><td>0.004**</td></tr><tr><td>S&amp;P 1500</td><td>1258</td><td>50</td><td>-0.13</td><td>+0.98</td><td>0.039*</td></tr></table>

improvement (Appendix C.5).

## 7.4 BD-DFL under Dynamic Selection

The preceding comparisons hold the selection protocol fixed. We next examine changing asset support, where errors outside the current support can affect subsequent decisions. The effect of regularization depends on both the model and selection rule. With dynamic tracking-score selection at $N { = } 1 0 0 .$ , neural DFL increases mean TE by 8.6% at K=10; BD-DFL with $\beta = 0 . 0 0 1$ reduces it by 8.2% relative to MSE. At N=478, the corresponding reduction is 7.4% $( p \ : = \ : 0 . 0 6 3$ , five folds). In the corrected N=451 run, pure DFL already improves by 13.0% at K=10 and BD-DFL improves by 15.6% $( p = 0 . 0 0 2 4$ , 9/11 wins). At K=20, pure DFL increases TE by 4.5%, while BD-DFL reduces it by 5.5% $( p = 0 . 1 0 3 )$ . These results motivate validation of regularization, not a universal benefit or coefficient (Appendix C.7).

## 7.5 Limits of Transfer

Historical experiments test the limits of transfer. A heterogeneity-based capacity proxy changes with the chosen partition and is not a Jacobian rank (Appendix D.1). NOAA and EPA show opposite performance changes with only one or two evaluation windows per configuration. The archived prediction record also predates evaluation-harness corrections; its scoring is not prospective validation of the corrected results.

The older shortest-path sweeps measure the input Jacobian, whereas the propositions concern the parameter Jacobian. Separate PyEPO probes use a sampled stacked parameter Jacobian, but their empirical benefit threshold does not transfer from equities (Appendices E.1–E.2). These distinct measurements motivate the matched, all-output parameter probes and independent validation used next.

## 7.6 Validation-Tuned Transfer Beyond Finance

To avoid relying on hyperparameters transferred from finance, we tune shortest-path and knapsack models separately (Appendix E.3). Each task uses ten independently generated datasets, a shared train-only ridge initializer, and identical validation budgets for MSE and SPO+ at every capacity. The probes measure all output coordinates at 32 held-out examples; they distinguish pointwise from stacked spectral rank.

Full-capacity SPO+ reduces mean test regret by 11.59% on shortest path and 10.59% on knapsack. Across the eight task–capacity comparisons, Holm-adjusted two-sided Wilcoxon p-values

![](images/1921ac5faedfac8526f49e25f130022b8ddbfefaca4d9407695a8fb59f2059d6.jpg)

![](images/13a897822f02caa4fc51c70abe934a48767d362372fedb69151d614aae1a92e1.jpg)  
Figure 5: Validation-tuned cross-domain extension. (a) Mean regret reduction with pointwise 95% paireddataset bootstrap intervals (ten datasets; 10,000 resamples). (b) Mean absolute MSE/SPO+ batch-gradient cosine at the selected MSE model, using 32 held-out examples. Each capacity is tuned separately; full capacity has 144/72 update directions for path/knapsack. Only full-capacity knapsack passes Holm correction across eight tests.

$$
\begin{array} { r l } { - 0 - \mathrm { ~ S c a l a r ~ } } & { { } \mathrm { - } \square - \mathrm { F u l l ~ c a p a c i t y } } \end{array}
$$

![](images/844a3b2093fde0102b441d82fcbcde473bd4f1911588405adf8f2ed10dd4a136.jpg)

![](images/dd0f5f8ae544b83da10fde484aed8e1f2a9304385285c1e6a290bccdbd82cfa8.jpg)  
Figure 6: Capacity contrast on fresh data and longer training. Mean test-regret reduction versus validation-selected MSE; whiskers are pointwise 95% paired-dataset bootstrap intervals. Each task uses ten new datasets and three minibatch orders, averaged within dataset before inference. Both losses may select the unchanged initializer. Filled markers denote comparisons passing Holm correction across eight tests; only full-capacity knapsack passes. Scalar gains below 0.6% are not evidence of equivalence.

are 0.0684 and 0.0156, respectively. At one, two and eight update directions, mean changes range from −0.61% to +1.89%, with pointwise bootstrap intervals crossing zero. Figure 5 shows the accompanying loss of batch-gradient alignment as capacity increases. These measurements support a capacity-dependent contrast in this protocol, not a universal gain threshold or a causal identification of rank alone.

A fresh-data sensitivity study crosses three minibatch orders with 20/80-epoch budgets and allows both losses to select the unchanged ridge predictor. Full-capacity mean gains remain 12.12/12.54% for path and 13.67/13.76% for knapsack; scalar gains stay below 0.6%. Inference averages training seeds within each independent dataset, rather than counting them as additional datasets. Figure 6 shows the follow-up: only full-capacity knapsack passes Holm correction (Appendix E.4).

## 7.7 Holding Expressivity Fixed

The capacity sweep changes both geometry and the function class. We therefore add an invertible coordinate transformation to a full affine predictor, $P = P _ { 0 } + D _ { \varepsilon } \Theta$ , where $D _ { \varepsilon } = \mathrm { d i a g } ( 1 , \varepsilon , \dots , \varepsilon )$ Every positive ε represents the same predictors and preserves exact Jacobian rank. Shrinking ε nevertheless lowers spectral effective rank and attenuates ordinary SGD updates in most output directions.

![](images/7d121d2e74e38c0aac2e2f57824a3f7135f4d851938072ba52f59aaa3dca065a.jpg)

![](images/24feda8dae054dde9bb7a3c13330f11752f484176e56008c8764eb97c01ad59a.jpg)  
Figure 7: Same function class, different ordinary SGD behavior. Invertible coordinate scaling reduces spectral effective rank while preserving exact rank and expressivity. Compensated SGD recovers the original predictor updates. Whiskers are pointwise 95% paired-dataset bootstrap intervals over ten fresh datasets per task, not multiplicity-adjusted tests.

On ten fresh datasets per task, reducing ε from 1 to 0.01 lowers mean SPO+ regret reduction from 11.79% to 1.19% for path and from 10.17% to 0.52% for knapsack. Compensating the parameter update by $D _ { \varepsilon } ^ { - 2 }$ restores the identity-coordinate trajectories and their gains (Figure 7). Thus expressivity alone cannot explain this contrast. Spectral rank and conditioning still change together: this control identifies a removable coordinate effect, not a causal effect of rank independent of optimization. Appendix F.1 gives the derivation and full protocol.

## 7.8 Forecast Accuracy Is a Separate Test

The fixed-class control addresses optimization coordinates; it leaves the prediction target unchanged. We next revisit the historical financial MSE target, which reconstructs trailing covariance. To test that choice, a separate chronological control fits the same scalar shrinkage family to future covariance, with forward labels purged at the training boundary. Across ten annual test folds (2016–2025), validation-selected future-target MSE reduces relative covariance error by 37.25% but increases mean tracking error by 4.13%. Task-validation tuning reduces mean tracking error by 1.46%; Ledoit–Wolf reduces it by 0.69%. These descriptive results expose a forecast-to-decision gap rather than establish DFL superiority. The fixed, survivorship-biased universe and scalar predictor limit this comparison (Appendix F.2).

We then compare future-target MSE with DFL in the same 197-parameter residual covariance network, using matched initializations, forward windows and validation budgets. Across the same ten test years, with three initialization seeds averaged within year, mean annualized TE is 5.711% for MSE and 5.715% for DFL: DFL is 0.07% worse in relative terms, despite improving in six years. Validation retains the unchanged initializer in 17/30 MSE and 12/30 DFL fits. Thus this neural control does not reproduce the larger historical DFL gains. It is neither an equivalence test nor evidence about all neural predictors; Appendix F.3 reports the architecture, yearly results and training limits.

## 8 Conclusion

Decision losses and predictors impose distinct restrictions: output support determines which errors a decision observes, the predictor Jacobian maps gradients into parameter directions, and stacking determines the batch subspace. Rank one restricts local directions without fixing training outcomes.

The experiments identify three distinct sources of variation. Capacity-dependent gains persist across tested batch orders and budgets, with stronger statistical evidence for knapsack. Invertible coordinate scaling changes ordinary SGD outcomes without changing expressivity, while better covariance forecasts can worsen tracking. A useful assessment of DFL must therefore examine geometry, optimization and held-out decision quality together. The matched neural forward-target control shows no aggregate DFL advantage under its finite budget. Broader architectures, longer training and point-in-time financial universes are needed to establish robustness; linking trainingtrajectory geometry to generalization remains open.

## Ethics statement

This work studies structural limitations of decision-focused learning in predict-then-optimize pipelines. Our experiments use publicly available financial, weather and air-quality data, together with synthetic optimization tasks. No human subjects are involved. Runtime comparisons describe computational cost in the tested settings; they do not measure energy savings. Financial backtests can be misinterpreted as prospective investment evidence. The reported tracking metrics do not establish profitability, deployment safety, or robustness to future market conditions.

## AI use statement

We used Claude (Anthropic) and Codex (OpenAI) to assist with analysis and plotting code, result auditing, literature checks, drafting and revising prose, and figure design. AI-assisted auditing identified evaluation-harness defects and unsupported theoretical steps, which prompted corrections and narrower claims in this revision. The accompanying audit records distinguish checked data, historical results, and remaining limitations. The authors remain responsible for the scientific content and the final submitted version.

## Reproducibility statement

The released bundle contains saved results, editable figures, numerical checks and a training-source snapshot. Three new combinatorial studies include 880 selected models, data arrays and rerunnable scripts. A scalar forecast-target study saves 70 annual portfolio evaluations; a matched neural study adds 120 candidate fits and 60 selected models. Historical real-data training has not been independently repeated in this release. Appendix B.1 distinguishes current corrected equity results, archived diagnostics and new synthetic runs; Appendix B.2 gives the equity training protocol.

The current equity comparisons use a repaired evaluation harness that confines holdings to each test window and uses actual trading dates. Earlier defects could extend validation holdings into test periods and changed some reported results. The correction history and remaining limits are documented in Appendix B.1. Archived diagnostic scores are not presented as independently certified preregistration or as a prospective test of the corrected harness.

## References

Akshay Agrawal, Brandon Amos, Shane Barratt, Stephen Boyd, Steven Diamond, and J. Zico Kolter. Differentiable convex optimization layers. In Advances in Neural Information Processing Systems (NeurIPS), 2019.

Irene Aldridge. Regret equals covariance: A closed-form characterization for stochastic optimization. arXiv preprint arXiv:2605.14019, 2026.

Brandon Amos and J. Zico Kolter. OptNet: Differentiable optimization as a layer in neural networks. In International Conference on Machine Learning (ICML), 2017.

John E. Beasley, Nigel Meade, and T.-J. Chang. An evolutionary heuristic for the index tracking problem. European Journal of Operational Research, 148(3):621–643, 2003.

Konstantinos Benidis, Yiyong Feng, and Daniel P. Palomar. Sparse portfolios for high-dimensional financial index tracking. IEEE Transactions on Signal Processing, 66(1):155–170, 2018.

Jose Blanchet and Karthyek Murthy. Quantifying distributional model risk via optimal transport. Mathematics of Operations Research, 44(2):565–600, 2019.

Mathieu Blondel, Quentin Berthet, Marco Cuturi, Roy Frostig, Stephan Hoyer, Felipe Llinares-López, Fabian Pedregosa, and Jean-Philippe Vert. Efficient and modular implicit differentiation. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Joshua Brodie, Ingrid Daubechies, Christine De Mol, Domenico Giannone, and Ignace Loris. Sparse and stable Markowitz portfolios. Proceedings of the National Academy of Sciences, 106 (30):12267–12272, 2009.

Andrew Butler and Roy H. Kwon. Integrating prediction in mean-variance portfolio optimization. Quantitative Finance, 23(3):429–452, 2023.

Nalan A. Canakgoz and John E. Beasley. Mixed-integer programming approaches for index tracking and enhanced indexation. European Journal of Operational Research, 196(1):384–399, 2009.

Aymeric Capitaine, Maxime Haddouche, Eric Moulines, Michael I. Jordan, Etienne Boursier, and Alain Durmus. Online decision-focused learning. In International Conference on Learning Representations (ICLR), 2026.

Priya L. Donti, Brandon Amos, and J. Zico Kolter. Task-based end-to-end model learning in stochastic optimization. In Advances in Neural Information Processing Systems (NeurIPS), 2017.

Adam N. Elmachtoub and Paul Grigas. Smart “predict, then optimize”. Management Science, 68 (1):9–26, 2022.

Robert Engle. Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. Journal of Business & Economic Statistics, 20 (3):339–350, 2002.

Jianqing Fan, Yuan Liao, and Martina Mincheva. Large covariance estimation by thresholding principal orthogonal complements. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 75(4):603–680, 2013.

Jerome Friedman, Trevor Hastie, and Robert Tibshirani. Sparse inverse covariance estimation with the graphical lasso. Biostatistics, 9(3):432–441, 2008.

Vishal Gupta and Michael Huang. Decision-focused learning with directional gradients. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Haeun Jeon, Seunghoon Choi, Hyunglip Bae, Yongjae Lee, and Woo Chang Kim. Decision-focused sparse tangent portfolio optimization. arXiv preprint arXiv:2607.00581, 2026.

Siddharth Joshi and Stephen Boyd. Sensor selection via convex optimization. IEEE Transactions on Signal Processing, 57(2):451–462, 2009.

Juchan Kim, Inwoo Tae, and Yongjae Lee. Estimating covariance for global minimum variance portfolio: A decision-focused learning approach. arXiv preprint arXiv:2508.10776, 2025.

Petter N. Kolm, Reha Tütüncü, and Frank J. Fabozzi. 60 years of portfolio optimization: Practical challenges and current trends. European Journal of Operational Research, 234(2):356–371, 2014.

Andreas Krause, Ajit Singh, and Carlos Guestrin. Near-optimal sensor placements in Gaussian processes: Theory, efficient algorithms and empirical studies. Journal of Machine Learning Research, 9:235–284, 2008.

Olivier Ledoit and Michael Wolf. A well-conditioned estimator for large-dimensional covariance matrices. Journal ofMultivariate Analysis, 88(2):365–411, 2004.

Junhyeong Lee, Sangjin Jin, and Yongjae Lee. Decision-focused learning via tangent-space projection of prediction error. arXiv preprint arXiv:2605.01361, 2026.

Jayanta Mandi, James Kotary, Senne Berden, Maxime Mulamba, Victor Bucarey, Tias Guns, and Ferdinando Fioretto. Decision-focused learning: Foundations, state of the art, benchmark and future opportunities. Journal of Artificial Intelligence Research, 80:1623–1701, 2024.

Peyman Mohajerin Esfahani and Daniel Kuhn. Data-driven distributionally robust optimization using the Wasserstein metric: Performance guarantees and tractable reformulations. Mathematical Programming, 171(1–2):115–166, 2018.

Anselm Paulus, Georg Martius, and Vít Musil. LPGD: A general framework for backpropagation through embedded optimization layers. In International Conference on Machine Learning (ICML), 2024.

Sanket Shah, Kai Wang, Bryan Wilder, Andrew Perrault, and Milind Tambe. Decision-focused learning without differentiable optimization: Learning locally optimized decision losses. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Bo Tang and Elias B. Khalil. PyEPO: a PyTorch-based end-to-end predict-then-optimize library for linear and integer programming. Mathematical Programming Computation, 16:297–335, 2024. doi: 10.1007/s12532-024-00255-x.

Marin Vlastelica, Anselm Paulus, Vít Musil, Georg Martius, and Michal Rolínek. Differentiation of blackbox combinatorial solvers. In International Conference on Learning Representations (ICLR), 2020.

Yi Wang and Takashi Hasuike. Decision-induced ranking explains prediction inflation and excessive turnover in SPO-based portfolio optimization. arXiv preprint arXiv:2605.01176, 2026.

Bryan Wilder, Bistra Dilkina, and Milind Tambe. Melding the data-decisions pipeline: Decisionfocused learning for combinatorial optimization. In AAAI Conference on Artificial Intelligence, 2019.

Fengmin Xu, Zhaosong Lu, and Zongben Xu. An efficient optimization approach for a cardinalityconstrained index tracking problem. Optimization Methods and Software, 31(2):258–271, 2016.

Zihao Zhang, Stefan Zohren, and Stephen Roberts. Deep learning for portfolio optimization. The Journal ofFinancial Data Science, 2(4):8–20, 2020.

## Appendix Guide

The appendices separate mathematical arguments from experimental evidence. Appendix A gives proofs and counterexamples; Appendix B documents provenance, training and cost; Appendix C collects equity comparisons; Appendix D examines the diagnostic and spatial transfer; Appendix E reports combinatorial experiments; Appendix F separates expressivity, optimization coordinates and predictive targets; Appendix G summarizes open questions.

## Reading by claim.

## Local directions and batch limits.

Appendix A: proofs, conditional bounds and counterexamples. These are fixed-parameter statements, not guarantees about training trajectories or test performance.

## Financial utility and robustness.

Appendix C: matched-fold tracking errors, risk, transaction costs and regularization. Rolling folds can share history; configurations are not independent datasets. Appendix B.1 identifies corrected evaluations and remaining provenance limits.

## Controlled evidence outside finance.

Appendices E.3–E.4: exact decision oracles, independently generated datasets, saved selected models and budget sensitivity. The statistical unit is a dataset; changing capacity also changes expressivity.

## Demonstration controls.

Appendix F: fixed-function-class reparameterization and chronological scalar and neural futuretarget baselines. The neural comparison has no aggregate DFL gain in its tested architecture; this is a descriptive result, not an equivalence test.

## Exploration and failure boundaries.

Appendix D and the earlier combinatorial studies retain proxy failures, sparse spatial evaluations and input-Jacobian measurements. They are distinct from the new parameter-geometry tests; $\mathsf { A p - }$ pendix G collects unresolved findings.

## A Geometry and Proofs

## A.1 Proof of Proposition 4 (Jacobian Rank Collapse)

Proof. At a fixed input and parameter value, write the nonzero rank-one Jacobian as ${ J } = \sigma u { v } ^ { \top }$ with unit vectors u, v and $\sigma > 0$ . By the chain rule, $\nabla _ { \boldsymbol { \theta } } \ell = J ^ { \top } g = \sigma ( \boldsymbol { u } ^ { \top } g ) \boldsymbol { v }$ . Every nonzero gradient is thus a scalar multiple of v, with cosine equal to the sign of the product of its two coefficients. This proves the statement for any parameter dimension $d ,$ including conditional predictors with many parameters. If either coefficient vanishes, its parameter gradient is zero and cosine is undefined. Differentiability is local; for a $\mathrm { Q P }$ it requires an appropriate regular solution map.

$$
\nabla _ { \boldsymbol { \theta } } \ell = J ^ { \top } \boldsymbol { g } .\tag{9}
$$

Corollary 1 (Vacuousness of the $\sqrt { K / N }$ bound for d=1). Proposition 2 bounds covariance-space alignment under its support-energy assumption. For a scalar predictor parameter, any two nonzero parameter gradients instead have cosine ±1. Their signs and magnitudes can still depend on K and $N ;$ the covariance-space angle does not survive as a separate parameter-space angle.

Proof. The optimizer updates $\theta$ using the parameter-space gradient $\nabla _ { \boldsymbol { \theta } } \ell ,$ which lies in R for a scalar parameter. The $N ^ { 2 }$ -space misalignment between g<sub>MSE</sub> and $\mathbf { g } _ { \mathrm { t a s k } }$ is projected onto this subspace via (9). For any two vectors $\mathbf { u } , \mathbf { v }$ with angle $\phi$ in $\mathbf { \mathbb { R } } ^ { N ^ { 2 } }$ , their projections onto a one-dimensional subspace span(J) yield scalars $\mathbf { J } ^ { \top }$ u and $\mathbf { J } ^ { \top } \mathbf { v }$ with cosine sign $( \mathbf { J } ^ { \top } \mathbf { u \cdot J } ^ { \top } \mathbf { v } ) \in \{ - 1 , + 1 \}$ , collapsing $\phi$ to either 0 or π. In particular, the covariance-space angle does not survive as an independent parameter-space angle for $d { = } 1$ □

Concordance vs. discordance. Proposition 4 establishes that d=1 gradients are collinear but does not determine the sign: $\cos = + 1$ (concordant, MSE descent also reduces task loss) vs. $\cos = - 1$ (discordant, MSE descent increases task loss). We formalize conditions for concordance and verify them experimentally.

Lemma 1 (Concordance for unimodal losses). Let $\Sigma ( \alpha ) = ( 1 - \alpha ) S + \alpha \mu I$ be the linear shrinkage family with $\alpha \in [ 0 , 1 ]$ . Suppose:

(i) $\ell _ { \mathrm { M S E } } ( \alpha )$ is strictly convex with minimizer $\alpha _ { \mathrm { M S E } } ^ { * } ,$

(ii) $\ell _ { \mathrm { t a s k } } ( \alpha )$ is unimodal (single minimizer $\alpha _ { \mathrm { t a s k } } ^ { * } )$ with strictly negative/positive derivative to the left/right of its interior minimizer, and an L-Lipschitz derivative on the interval between the two minimizers;

(iii) $| \alpha _ { \mathrm { M S E } } ^ { * } - \alpha _ { \mathrm { t a s k } } ^ { * } | \le \delta$ for some $\delta > 0 .$

Then cos $( \nabla _ { \alpha } \ell _ { \mathrm { M S E } } , \nabla _ { \alpha } \ell _ { \mathrm { t a s k } } ) = + 1$ for all α $\not \in \mathrm { ~ [ m i n ( } \alpha _ { \mathrm { M S E } } ^ { * } , \alpha _ { \mathrm { t a s k } } ^ { * } )$ , max $\left[ \alpha _ { \mathrm { M S E } } ^ { * } , \alpha _ { \mathrm { t a s k } } ^ { * } \right) ]$ ], and the task-loss gap satisfies $\ell _ { \mathrm { t a s k } } ( \alpha _ { \mathrm { M S E } } ^ { * } ) - \ell _ { \mathrm { t a s k } } ( \alpha _ { \mathrm { t a s k } } ^ { * } ) \leq L \delta ^ { 2 } / 2 .$

Proof. Condition (i) ensures $\nabla _ { \alpha } \ell _ { \mathrm { M S E } } ~ < ~ 0$ for $\alpha < \alpha _ { \mathrm { M S E } } ^ { * }$ and $> 0$ for $\alpha > \alpha _ { \mathrm { M S E } } ^ { * } .$ . Condition (ii) ensures $\nabla _ { \alpha } \ell _ { \mathrm { t a s k } }$ has the same sign pattern around $\alpha _ { \mathrm { t a s k } } ^ { * }$ . For $\alpha < \mathrm { m i n } ( \alpha _ { \mathrm { M S E } } ^ { * } , \alpha _ { \mathrm { t a s k } } ^ { * } )$ or $\alpha >$ max $( \alpha _ { \mathrm { M S E } } ^ { * } , \alpha _ { \mathrm { t a s k } } ^ { * } )$ , both gradients share the same sign, hence $\cos { \mathbf { \theta } } = { \mathbf { \theta } } + 1 { \mathbf { \theta } }$ . Let $a \ = \ \alpha _ { \mathrm { t a s k } } ^ { * }$ and $b = \alpha _ { \mathrm { M S E } } ^ { * } .$ . Since $\ell _ { \mathrm { t a s k } } ^ { \prime } ( a ) = 0$ and its derivative is L-Lipschitz, integrating $| \ell _ { \mathrm { t a s k } } ^ { \prime } ( t ) | \leq L | t - a |$ between a and b bounds the loss difference by $L | b - a | ^ { 2 } / 2 \le L \delta ^ { 2 } / 2$ □

Condition verification. We verify conditions (i)–(iii) across 20 train/test folds (4 markets: S&P 500, Nikkei 225, Euro Stoxx 50, Hang Seng; $K { = } 2 0 ; \alpha \ \in \ [ 0 . 0 1 , 0 . 9 9 ] )$ . Condition (i): $\ell _ { \mathrm { M S E } } ( \alpha )$ is convex quadratic by construction (Ledoit and Wolf, 2004). Condition (ii): $\ell _ { \mathrm { t a s k } } ( \alpha )$ is unimodal in 20/20 folds (checked via monotonicity before and after the minimum, tolerating $\leq 2$ noise violations). Condition (iii): $| \alpha _ { \mathrm { M S E } } ^ { * } - \alpha _ { \mathrm { t a s k } } ^ { * } |$ averages 0.154 (median 0.040), with 14/20 folds having $\mathrm { g a p < 0 . 1 5 }$

Scope. The Jacobian is $\mathrm { v e c } ( \mu I - S )$ and has one column. This restricts directions, not minimizers: even two scalar convex losses $( \dot { \alpha } - a ) ^ { 2 }$ and $( \alpha - b ) ^ { 2 }$ can prefer different parameters. The quadratic loss-gap bound above uses smoothness and nearby minimizers; the empirical unimodality check does not certify those assumptions globally.

Broader verification. Across 38 d=1 configurations (6 equity markets, $K \in \{ 5 , 1 0 , 2 0 , 5 0 , 1 0 0 \}$ N from 47 to 1258, $K / N$ from 0.008 to 0.43), the DFL gain over MSE stays below 1.8% with median 0.25%, consistent with the $O ( \delta ^ { 2 } )$ gap from Lemma 1. It does not grow with $K / N \left( \rho { = } { - } 0 . 1 6 \right.$ $\scriptstyle { p = 0 . 3 5 } )$ , an empirical observation compatible with, but not implied by, Corollary 1: the $\Theta ( \sqrt { K / N } )$ mechanism is invisible through a rank-1 Jacobian, although signs, stationary points and test losses can still vary with sparsity.

Extension to $d > 1$ . For a d-parameter model $\Sigma ( \pmb \theta )$ with $\pmb \theta \in \mathbb R ^ { d } .$ , the Jacobian $ { \mathbf { J } } \in  { \mathbb { R } } ^ { N ^ { 2 } \times d }$ maps $N ^ { 2 }$ -space gradients to d-dimensional parameter-space gradients via $\nabla _ { \theta } \ell = \mathbf { J } ^ { \top } \mathbf { g }$ . The parameterspace alignment becomes:

$$
\cos ( \nabla _ { \pmb \theta } \ell _ { \mathrm { M S E } } , \nabla _ { \pmb \theta } \ell _ { \mathrm { t a s k } } ) = \frac { \mathbf { g } _ { \mathrm { M S E } } ^ { \top } \mathbf { J } \mathbf { J } ^ { \top } \mathbf { g } _ { \mathrm { t a s k } } } { \left\| \mathbf { J } ^ { \top } \mathbf { g } _ { \mathrm { M S E } } \right\| \cdot \left\| \mathbf { J } ^ { \top } \mathbf { g } _ { \mathrm { t a s k } } \right\| } .\tag{10}
$$

The matrix $\mathbf { J } \mathbf { J } ^ { \top } \in \mathbb { R } ^ { N ^ { 2 } \times N ^ { 2 } }$ is a positive semidefinite form of rank at most d. When $d = 1$ , it is rank-1 and collapses all angles to $\{ 0 , \pi \}$ (Proposition 4). Because $\mathbf { J } \mathbf { J } ^ { \top } = U S ^ { 2 } U ^ { \top }$ , the relevant inner product weights each singular direction by $\sigma _ { i } ^ { 2 } \mathrm { { ; } }$ ; it is not the plain projection $P _ { J } = U U ^ { \top }$ onto col(J), and the two agree only when all nonzero $\sigma _ { i }$ are equal. In the extreme case of $N ^ { 2 }$ orthogonal columns of equal norm, $\mathbf { J } \mathbf { J } ^ { \top } = { \sigma } ^ { 2 } I$ and the parameter-space alignment equals the $N ^ { 2 }$ -space alignment $\cos ( \mathbf { g } _ { \mathrm { M S E } } , \mathbf { g } _ { \mathrm { t a s k } } )$ ; the expected numerical bound requires the support-energy assumption, and full rank alone is not enough.

Proof of Theorem 1. Exact form. With $\mathbf { J } = U S V ^ { \top }$ (thin SVD, $V \in \mathbb { R } ^ { d \times r }$ with orthonormal columns), $\nabla _ { \pmb { \theta } } \ell = \mathbf { J } ^ { \top } \mathbf { g } = V$ a with $\mathbf { a } = S U ^ { \top } \mathbf { g }$ . Multiplication by V preserves inner products and norms, so cos $( \nabla _ { \pmb \theta } \ell _ { \mathrm { M S E } } , \nabla _ { \pmb \theta } \ell _ { \mathrm { t a s k } } ) = \cos ( \mathbf { a } , \mathbf { b } )$ with $\mathbf { b } = S U ^ { \top } \mathbf { g } _ { \mathrm { t a s k } }$

Near-rank-one collapse. Write $\mathbf { a } = ( a _ { 1 } , \mathbf { a } _ { \perp } )$ with $a _ { 1 } = \sigma _ { 1 } \mathbf { u } _ { 1 } ^ { \prime } \mathbf { g }$ and $( \mathbf { a } _ { \perp } ) _ { i } = \sigma _ { i } \mathbf { u } _ { i } ^ { \top } \mathbf { g }$ for $i \geq 2$ Then $\begin{array} { r } { \| \mathbf { a } _ { \perp } \| ^ { 2 } = \sum _ { i > 2 } \sigma _ { i } ^ { 2 } ( \mathbf { u } _ { i } ^ { \top } \mathbf { g } ) ^ { 2 } \leq \sigma _ { 2 } ^ { 2 } \| \mathbf { g } \| ^ { 2 } , } \end{array}$ , so

$$
\frac { \| \mathbf { a } _ { \perp } \| } { | a _ { 1 } | } \leq \frac { \sigma _ { 2 } } { \sigma _ { 1 } } \cdot \frac { \| \mathbf { g } \| } { | \mathbf { u } _ { 1 } ^ { \top } \mathbf { g } | } = \frac { \sigma _ { 2 } / \sigma _ { 1 } } { | \cos ( \mathbf { g } , \mathbf { u } _ { 1 } ) | } .
$$

Let $\tilde { \mathbf { a } } = \mathrm { s i g n } ( a _ { 1 } ) \mathbf { a }$ , so $\tilde { a } _ { 1 } > 0$ and the angle $\theta _ { a }$ between a˜ and $\mathbf { e } _ { 1 }$ lies in $[ 0 , \pi / 2 )$ with sin $\theta _ { a } = $ $\| \mathbf { a } _ { \perp } \| / \| \mathbf { a } \| \leq \| \mathbf { a } _ { \perp } \| / | a _ { 1 } | ;$ hence $\theta _ { a } \ \leq$ arcsin ρ , and likewise $\theta _ { b } \le$ arcsin $\rho _ { \mathrm { t a s k } }$ . The angle between unit vectors is a metric on the sphere, so the angle between a˜ and b<sup>˜</sup> is at most $\theta _ { a } + \theta _ { b } \le \Theta$ When $\Theta < \pi / 2$ this gives $\cos ( \tilde { \mathbf { a } } , \tilde { \mathbf { b } } ) \geq \cos \Theta > 0$ , and since $\cos ( { \mathbf { a } } , { \mathbf { b } } ) = \mathrm { s i g n } ( a _ { 1 } b _ { 1 } ) \cos ( { \tilde { \mathbf { a } } } , \tilde { \mathbf { b } } )$ both the bound on | cos | and the sign rule follow.

Endpoints. If $\sigma _ { 2 } = 0$ then $\rho = 0 , \Theta = 0 \mathrm { a n d } | \cos | = 1$ . If J has $N ^ { 2 }$ orthogonal columns of norm σ, then $S = \sigma I$ and U is square orthogonal, so $\cos ( \mathbf { a } , \mathbf { b } ) = \cos ( \mathbf { g } _ { \mathrm { M S E } } , \mathbf { g } _ { \mathrm { t a s k } } ) .$ □

Numerical stress check. The released script verify\_geometry.py samples 5,000 Jacobians with singular-value ratios from $1 0 ^ { - 6 }$ to one. The exact alignment identity agrees to within $4 . 2 \times$ $1 0 ^ { - 1 4 }$ ; the bound and sign rule have no violations in the 3,926 non-vacuous cases. The script also checks the batch, leading-component and support-sharpness constructions. These computations are checks of the formulas, not replacements for their proofs.

Connecting $d _ { \mathrm { p r o x y } }$ to the Jacobian spectrum. We define the entropy-based spectral effective rank by $\begin{array} { r } { r _ { \mathrm { e f f } } = \exp ( - \sum _ { i } p _ { i } \log p _ { i } ) } \end{array}$ , with $p _ { i } = \sigma _ { i } ^ { 2 } / \sum _ { j } \sigma _ { j } ^ { 2 }$ . The empirical proxy $d _ { \mathrm { p r o x y } } = d \cdot h$ is distinct from this measured quantity. Neither a parameter count nor a heterogeneity coefficient alone determines the singular values or the gradients’ leading components. The conditional-estimator and spectral-bound experiments (Appendices D.2 and D.1) should therefore be interpreted as diagnostic checks rather than a universal identity between $d \cdot h$ and rank.

Corollary 2 (Non-collinear parameter gradients). Suppose the projections of g<sub>MSE</sub> and $\mathbf { g } _ { \mathrm { t a s k } }$ onto col(J) are non-collinear. Their parameter-space images are then non-collinear. Writing c for their cosine, the task-gradient component orthogonal to the MSE gradient has norm

$$
\lVert \nabla _ { \theta } \ell _ { \mathrm { t a s k } } \rVert \sqrt { 1 - c ^ { 2 } } > 0 .
$$

Proof. On $\operatorname { c o l } ( \mathbf { J } )$ , the map $\mathbf { J } ^ { \top }$ is injective. It therefore preserves non-collinearity of the two projected vectors assumed in the statement. The stated norm follows by orthogonal decomposition.

## A.2 Counterexamples and Batch Geometry

Proposition 5 is an exact statement about the attainable first-order subspace at a fixed parameter value. It does not require a QP. Positive averaging weights can be absorbed into the rows of $J _ { \mathrm { s t a c k } }$ without changing the span; zero-weight examples can be omitted.

Different minimizers with one parameter. On $( 0 , 3 )$ , set $f ( \theta ) = \theta , L _ { 1 } = ( f - 1 ) ^ { 2 }$ and $L _ { 2 } =$ $( f - 2 ) ^ { 2 }$ . Then $J = 1$ , while arg min $L _ { 1 } = 1$ and arg min $L _ { 2 } = 2 .$ . Their nonzero derivatives have opposite signs for $1 < \theta < 2 . { \mathrm { ~ A t ~ } } \theta = 1$ , only the first derivative vanishes. Thus collinearity does not imply concordance, common stationary points, or common minimizers. The example can use the positive scalar covariance $\Sigma ( \theta ) = \theta$ directly.

Orthogonal batch gradients from rank-one examples. For $\theta \in \mathbb { R } ^ { 2 } , J _ { 1 } = ( 1 , 0 )$ and $J _ { 2 } = $ $( 0 , 1 )$ each have rank one, while $J _ { \mathrm { s t a c k } } = I _ { 2 }$ . With losses as in Section 5.3, differentiation at zero gives (1, 1) and $( 1 , - 1 )$ , whose inner product is zero. Positive covariance outputs can be obtained by replacing $f _ { i }$ with $2 + \theta _ { i }$ near zero and shifting the loss targets correspondingly. The example establishes a failure of the proposed inference; it does not purport to reproduce the financial experiment.

A spectral gap without a leading task component. Let $J = \mathrm { d i a g } ( 1 , \varepsilon ) , g _ { 1 } = e _ { 1 }$ and $g _ { 2 } = e _ { 2 }$ where $\varepsilon > 0$ . Their parameter gradients remain orthogonal for every ε, although the singular-value ratio tends to zero. This is why the leading-component condition in Theorem 1 cannot be omitted. $\mathbf { A } \mathbf { t } \varepsilon = 0$ , the second parameter gradient vanishes, so its cosine is undefined.

## A.3 Proof of Proposition 2

Proof. We prove the gradient alignment bound in three steps.

Step 1: Sparsity of the task gradient. Under the hard-K formulation, the QP solution $\mathbf { w } ^ { * } \in$ $\mathbb { R } ^ { K }$ depends on $\hat { \Sigma }$ through two quantities: the selected submatrix $\hat { \Sigma } _ { S S } \in \mathbb { R } ^ { K \times K }$ and the crosscovariance vector $\hat { \pmb { \sigma } } _ { \mathrm { i d x } , S } = \hat { \Sigma } _ { S , : } \mathbf { w } _ { \mathrm { i d x } } \in \mathbb { R } ^ { K }$ . Entry $( i , j )$ of $\hat { \Sigma }$ contributes to the cross-covariance whenever $i \in S$ (regardless of $j ,$ , since $\mathbf { w } _ { \mathrm { i d x } }$ has support on all N assets), and to the sub-matrix whenever both $i , j \in S$ . For any pair $( i , j )$ with $i \not \in S$ and $j \not \in { \mathcal { S } }$ , the entry $\hat { \Sigma } _ { i j }$ does not appear in either quantity. By the chain rule through the KKT conditions (Amos and Kolter, 2017), $( \mathbf { g } _ { \mathrm { t a s k } } ) _ { i j } =$ 0 for all $( i , j ) \in \bar { \mathcal { T } }$

The task-relevant support is $\mathcal { T } = \{ ( i , j ) : i \in \mathcal { S } \mathrm { o r } j \in \mathcal { S } \}$ . Its cardinality is $| \mathcal { T } | = N ^ { 2 } - ( N -$ $K ) ^ { 2 } = 2 N K - K ^ { 2 }$

Step 2: Cosine similarity bound via Cauchy–Schwarz. Since $\mathbf { g } _ { \mathrm { t a s k } }$ is supported on $\tau$ , we can write:

$$
\begin{array} { r l r } { \langle { \bf g } _ { \mathrm { M S E } } , { \bf g } _ { \mathrm { t a s k } } \rangle = \displaystyle \sum _ { ( i , j ) \in \mathcal { T } } ( { \bf g } _ { \mathrm { M S E } } ) _ { i j } ( { \bf g } _ { \mathrm { t a s k } } ) _ { i j } } & { } & \\ & { } & { = \langle { \bf g } _ { \mathrm { M S E } } | { \boldsymbol \tau } , { \bf g } _ { \mathrm { t a s k } } \rangle , \quad } \end{array}\tag{11}
$$

where $\mathbf { g } _ { \mathrm { M S E } } | \boldsymbol { \tau }$ denotes the restriction of $\mathbf { g } _ { \mathrm { M S E } }$ to the support T . By Cauchy–Schwarz:

$$
| \langle \mathbf { g } _ { \mathrm { M S E } } , \mathbf { g } _ { \mathrm { t a s k } } \rangle | ^ { 2 } \leq \| \mathbf { g } _ { \mathrm { M S E } } | _ { T } \| ^ { 2 } \cdot \| \mathbf { g } _ { \mathrm { t a s k } } \| ^ { 2 } .\tag{12}
$$

Dividing both sides by $\| \mathbf { g } _ { \mathrm { M S E } } \| ^ { 2 } \cdot \| \mathbf { g } _ { \mathrm { t a s k } } \| ^ { 2 } .$

$$
\cos ^ { 2 } ( \mathbf { g } _ { \mathrm { M S E } } , \mathbf { g } _ { \mathrm { t a s k } } ) \leq \frac { \Vert \mathbf { g } _ { \mathrm { M S E } } \vert \tau \Vert ^ { 2 } } { \Vert \mathbf { g } _ { \mathrm { M S E } } \Vert ^ { 2 } } .\tag{13}
$$

Step 3: Isotropy assumption. For a fixed support, a sufficient assumption is isotropy of the normalized energy: $\mathbb { E } [ g _ { i j } ^ { 2 } / \lVert \mathbf { g } \rVert ^ { 2 } ] = 1 / N ^ { 2 }$ for every coordinate. For data-dependent selection the corresponding conditional assumption is needed. Under this condition,

$$
\mathbb { E } \bigg [ \frac { \| \mathbf { g } _ { \mathrm { M S E } } \| \tau \| ^ { 2 } } { \| \mathbf { g } _ { \mathrm { M S E } } \| ^ { 2 } } \bigg ] = \frac { | \mathcal { T } | } { N ^ { 2 } } = \frac { 2 N K - K ^ { 2 } } { N ^ { 2 } } = \frac { 2 K } { N } - \frac { K ^ { 2 } } { N ^ { 2 } } .\tag{14}
$$

Combining with Step 2 and applying Jensen’s inequality:

$$
\mathbb { E } [ | \cos ( \mathbf { g } _ { \mathrm { M S E } } , \mathbf { g } _ { \mathrm { t a s k } } ) | ] \le \sqrt { \frac { 2 K } { N } - \frac { K ^ { 2 } } { N ^ { 2 } } } .\tag{15}
$$

For $K \ll N$ , the $K ^ { 2 } / N ^ { 2 }$ term is negligible, yielding E| cos $( \mathbf { g } _ { \mathrm { M S E } } , \mathbf { g } _ { \mathrm { t a s k } } ) \vert \lesssim \sqrt { 2 K / N }$ under the same assumption. □

Scope of the energy assumption. Equal unnormalized second moments do not suffice: expectation of a ratio is not the ratio of expectations. Normalized-energy isotropy is an explicit sufficient condition, not a property of every covariance estimator. Selection can concentrate error energy; then the deterministic support-energy ratio in Step 2 remains valid, while the numerical $\sqrt { 2 K / N }$ bound need not hold.

## A.4 Sharpness of the Support-Only Inequality

Proposition 6 (Ambient-vector sharpness). For a support T of size m in $\mathbb { R } ^ { N ^ { 2 } }$ , there are nonzero vectors $\mathbf { g } ,$ h with h supported on T and uniform squared coordinates in $\mathbf { g } ,$ such that $\cos ( \mathbf { g } , \mathbf { h } ) =$ $\sqrt { m / N ^ { 2 } }$

Proof. Set every coordinate of g to one and let h be the indicator of T. Their inner product is $m ,$ and their norms are N and $\sqrt { m }$ . Their cosine is therefore $\sqrt { m } / N$ , attaining the support-energy bound. □

This proves sharpness of the linear-algebra inequality only. The earlier equicorrelation construction did not establish a nonzero tracking gradient with the asserted structure. In fact, adding $\epsilon \mathbf { 1 1 } ^ { \top }$ leaves the fully invested tracking objective unchanged because $\mathbf { 1 } ^ { \top } ( \mathbf { w } - \mathbf { w } _ { \mathrm { i d x } } ) = 0$ . We do not claim a matching lower bound realized by the tracking QP.

## A.5 Proof of Proposition 3

Proof. Fix a selection set S with $| S | = K$ . For any portfolio w with $w _ { i } = 0$ for $i \not \in { \mathcal { S } }$ , define $\mathbf { d } = \mathbf { w } - \mathbf { w } _ { \mathrm { i d x } } .$ . Then $d _ { i } = w _ { i } - w _ { \mathrm { i d x } , i }$ for $i \in S$ and $d _ { i } = - w _ { \mathrm { i d x } , i }$ for $i \not \in { \mathcal { S } }$ , where the latter is fixed. The QP objective decomposes as:

$$
\mathbf { d } ^ { \top } \hat { \Sigma } \mathbf { d } = \mathbf { d } _ { S } ^ { \top } \hat { \Sigma } _ { S S } \mathbf { d } _ { S } + 2 \mathbf { d } _ { S } ^ { \top } \hat { \Sigma } _ { S \bar { S } } \mathbf { d } _ { \bar { S } } + \mathbf { d } _ { \bar { S } } ^ { \top } \hat { \Sigma } _ { \bar { S } \bar { S } } \mathbf { d } _ { \bar { S } } .\tag{16}
$$

Since $\mathbf { d } _ { \bar { S } } = - \mathbf { w } _ { \mathrm { i d x } , \bar { S } }$ is fixed, the third term is a constant that does not affect the argmin, and the second term involves only the K-vector $\hat { \Sigma } _ { S \bar { S } } \mathbf { w } _ { \mathrm { i d x } , \bar { S } }$ . Thus, two estimates $\hat { \Sigma } _ { 1 } , \hat { \Sigma } _ { 2 }$ with $( \hat { \Sigma } _ { 1 } ) _ { S S } =$ $( \hat { \Sigma } _ { 2 } ) s s$ and $( \hat { \Sigma } _ { 1 } ) _ { S \bar { S } } { \bf w } _ { \mathrm { i d x } , \bar { S } } = ( \hat { \Sigma } _ { 2 } ) _ { S \bar { S } } { \bf w } _ { \mathrm { i d x } , \bar { S } } $ yield identical objectives up to a constant, hence $\mathbf { w } ^ { * } ( \hat { \Sigma } _ { 1 } ) = \mathbf { w } ^ { * } ( \hat { \Sigma } _ { 2 } )$

The sufficient condition $( \hat { \Sigma } _ { 1 } ) _ { \mathcal { T } } = ( \hat { \Sigma } _ { 2 } ) _ { \mathcal { T } }$ (equality on the full task-relevant support) is stronger than needed—it implies both block-equality and cross-product equality—but matches the gradient support from Proposition 2. □

## A.6 Task Regret and Local Perturbation

Proposition 7 (Off-support invariance of regret). Fix the selected set and the true covariance used to evaluate decisions. Changing only the estimated covariance entries in $\bar { \cal S } \times \bar { \cal S }$ leaves the $Q P$ solution and its evaluated task regret unchanged (with a common tie-breaking rule ifnecessary).

Proof. Such changes preserve $Q = \hat { \Sigma } _ { S S }$ and $b = \hat { \Sigma } _ { S , : } w _ { \mathrm { i d x } }$ , so they preserve the entire QP objective and feasible set. They therefore preserve its decision and any fixed evaluation of that decision.

A local quantitative bound additionally requires stability assumptions. Let $Q \ = \ \Sigma _ { S S } \ \succ \ 0$ denote the true block, with $\lambda = \lambda _ { \operatorname* { m i n } } ( Q ) > 0$ . Write $\hat { Q } = Q + \Delta _ { Q } , \hat { b } = b + \Delta _ { b }$ . Suppose $\| \Delta _ { Q } \| _ { \mathrm { o p } } \leq \lambda / 2$ and both solutions have the same active set. With $\delta w = \hat { w } - w ^ { * }$ , subtracting the two KKT stationarity equations and multiplying by δw gives

$$
\delta \boldsymbol { w } ^ { \top } \boldsymbol { Q } \delta \boldsymbol { w } = \delta \boldsymbol { w } ^ { \top } ( \Delta _ { b } - \Delta _ { \boldsymbol { Q } } \hat { w } ) ,
$$

because $\mathbf { 1 } ^ { \top } \delta w = 0$ . Hence

$$
\| \delta w \| \leq \frac { 2 } { \lambda } ( \| \Delta _ { Q } \| _ { \mathrm { o p } } \| w ^ { * } \| + \| \Delta _ { b } \| ) .
$$

For the tracking-variance objective $V ( w ) = ( w - w _ { \mathrm { i d x } } ) ^ { \top } \Sigma ( w - w _ { \mathrm { i d x } } )$ , KKT stationarity on the common free face makes the linear term in $V ( \hat { w } ) - V ( w ^ { * } )$ exactly zero. Consequently,

$$
0 \leq V ( \hat { w } ) - V ( w ^ { * } ) \leq \frac { 4 \| \Sigma \| _ { \mathrm { o p } } } { \lambda ^ { 2 } } ( \| \Delta _ { Q } \| _ { \mathrm { o p } } \| w ^ { * } \| + \| \Delta _ { b } \| ) ^ { 2 } .
$$

This is a local bound on tracking variance, not a global bound on annualized tracking-error standard deviation across active-set changes. The two perturbations involve only the selected block and index cross-covariance; they do not depend on estimated off-support entries.

## A.7 Selection-Robustness Motivation and Its Limits

For any covariance error matrix E and selected subset S′, $\| E _ { S ^ { \prime } S ^ { \prime } } \| _ { \mathrm { o p } } \leq \| E \| _ { F }$ and $\| E _ { S ^ { \prime } , : } \mathbf { w _ { \mathrm { i d x } } } \| \leq$ $\| E \| _ { F } \| \mathbf { w } _ { \mathrm { i d x } } \|$ . Thus full-matrix error control also controls each selected block and index crosscovariance term. Combined with a local QP perturbation bound under a stable active set, this supplies a conservative reason to regularize covariance predictions when selection changes.

An earlier argument inferred a worst-case coefficient $O ( r ^ { 2 } / N ^ { 2 } )$ from the fraction of entries exposed by r swaps. That inference is not valid without a bound on error concentration: one newly selected row can contain nearly all error energy. Nor can the oracle objective for one subset be replaced by that for another without a separate bound. We therefore make no minimax-equivalence or universal coefficient claim. BD-DFL is the empirical regularized objective in Eq. (3); its reported gains do not require that derivation. A theorem linking swap radius to a sharp regularization weight remains open.

## B Protocols, Provenance, and Computational Cost

## B.1 Evidence Provenance and Reproduction Scope

The source bundle compiles the paper and includes its figures. A separate reproduction bundle contains saved result files, figure generators, checks, and a snapshot of the local training source. The code snapshot does not reconstruct every historical run. The new controlled extension comprises 480 candidate fits and 160 selected models; the sensitivity study adds 720 optimization trajectories and 480 selected models. The fixed-class study adds 720 candidate fits and 240 selected models, bringing the new synthetic total to 1,920 fits and 880 selected models. The scalar financial target control saves 70 annual portfolio evaluations. The matched neural forward-target study adds 120 candidate trajectories and 60 selected models, with identical per-loss tuning budgets. These new financial controls use separate protocols and do not rerun the historical neural experiments. Historical spatial training was not rerun.

Current equity comparisons. The primary neural files identify the corrected rebalance harness. Each contains both a common naive baseline and a trained model, so selection must use model, loss, cardinality and fold jointly. For K = 20, there are exactly nine neural rows per loss and 2,096 test-return observations. Recomputed sample-standard-deviation TE matches the saved metric in every selected fold. Earlier risk aggregation mixed in naive rows; the present risk table excludes them. Saved configurations and evaluation windows should be checked before combining other equity runs.

Evaluation-harness corrections. An earlier implementation held each fold’s final rebalance to the end of the dataset instead of its test window. It also used calendar month-end labels: weekend month ends could be skipped, and labels beyond a window could extend holdings outside it. In validation, such extensions could reach test dates. The corrected harness uses the last trading day within each window and terminates holdings at that window’s end. Current equity comparisons use the saved corrected runs; the archived diagnostic record does not.

These changes affected interpretation as well as magnitude. The corrected N = 451, K = 10 comparison favors pure DFL, contrary to an earlier overfitting description. For the nine-fold neural comparison at K = 20, the corrected mean TE reduction is 16.5%. Source files, test windows and model identities are retained so that these comparisons can be distinguished from older evaluations.

Archived predictions. The 33-case diagnostic record predates the corrected equity evaluation. It contains 31 committed predictions with 27 correct outcomes and two abstentions. Timestamps and calibration separation have not been independently certified in this release. Threshold sensitivity and measured-h re-scoring are post-hoc analyses and cannot be used as independent prospective validation.

Spatial and synthetic evidence. The synthetic heterogeneity sweep contains 18 settings with five folds each. NOAA and EPA provide only one or two evaluation windows per configuration, limiting inference about cross-domain replication. The listed sensor experiment has ten configurations with five trials. Hypothetical facility-location and resource-allocation applications are not additional experiments.

Combinatorial evidence. The custom shortest-path and PyEPO experiments have their own seeds, generators and training protocols. The custom shortest-path sweep measures input Jacobians. PyEPO rank estimates use a sampled stacked parameter Jacobian, whereas the equity bound experiment uses pointwise parameter Jacobians. The controlled extension uses all-output parameter probes. Full-capacity hyperparameter selection can favor that regime; unstable and negative cases are reported rather than removed by the rank argument.

What the checks establish. The numerical checkers compare selected table entries and figures with saved files. They do not certify data-provider revisions, every historical training configuration, independent market sampling, or the correctness of all scientific interpretations. The geometry script tests the stated identities on generated matrices and explicit counterexamples. The manuscript’s mathematical arguments and experiment-design limitations remain necessary for interpretation.

Training-target limitation. The inspected equity trainer uses a trailing covariance reconstruction target for its MSE branch, while the task branch uses a forward return window. Consequently, these experiments compare specific implemented objectives rather than an optimal future-covariance forecasting baseline. For scalar shrinkage, reconstructing its own sample-covariance input is especially restrictive. Separate validation-tuned shrinkage and statistical baselines partly broaden the compar ison, but a matched future-target forecasting experiment is still needed.

## B.2 Training Details

Algorithm 1 Accumulated-gradient training for sparse index tracking   
Require: Predictor $f _ { \theta } ,$ , training dates D, index weights, K, accumulation size $B = 4$   
1: for each epoch do   
2: Zero accumulated parameter gradients; set $c = 0$ and $w _ { \mathrm { p r e v } }$ unset   
3: for each valid date $t \in \mathcal { D }$ do   
4: Form trailing features and covariance target using data before t   
5: Predict $\hat { \Sigma } _ { t } = f _ { \theta } ( x _ { t } )$ ; select a detached subset $S _ { t }$   
6: Solve the differentiable QP for $w _ { t } ^ { \ast } ;$ record and skip solver failures   
7: Evaluate Eq. (2); add Eq. (3)’s penalty when $\beta > 0$   
8: Accumulate $\nabla _ { \boldsymbol { \theta } } ( \boldsymbol { \mathcal { L } } _ { t } /$ min $( B , | \mathcal { D } | ) )$ ; set $c \gets c + 1$   
9: Set $w _ { \mathrm { p r e v } } $ stopgrad $( w _ { t } ^ { * } )$   
10: if c is a multiple of B then   
11: Clip accumulated gradients; take an Adam step; zero gradients   
12: end if   
13: end for   
14: Flush a nonempty partial accumulation with the same scaling   
15: Evaluate validation $\mathrm { T E } ,$ update early stopping, and advance the scheduler   
16: end for

The MSE-only branch uses its reconstruction loss without solving the QP during the backward pass. The pseudocode states the single-step task-training path and the accumulation convention in the inspected source. Partial accumulations retain the same denominator; they are not reweighted to an exact partial-batch mean. Previous holdings are detached, so the basic path does not differentiate through the full sequence of past decisions. Experiment-specific runners override some settings; saved configurations take precedence over the defaults below.

The default equity trainer uses Adam $( \mathrm { l r } = 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 } )$ , cosine annealing, gradient clipping at norm 1.0, and gradient accumulation over 4 rebalancing dates. Early stopping with patience 10 monitors validation tracking error. Training uses weekly rebalancing; evaluation uses monthly rebalancing. The QP is solved via cvxpylayers (Agrawal et al., 2019) for backward passes and SCS for test-time inference. DFL training takes ∼10 min/fold on Apple M1 (N=100, 50 epochs).

## B.3 Computational Cost Comparison

Table 4 compares measured wall-clock cost and tracking error in one N = 100, K = 20 setting. The reported signed-rank comparisons do not detect a difference from POET; they are not equivalence tests. Timing and accuracy must be assessed jointly for the intended implementation, hardware and tolerance.

Table 4: Computational cost per fold (N=100, K=20). Tracking errors are regenerated from the result files by scripts/emit\_table\_cost.py; an earlier version of this table carried pre-correction values (0.0285–0.0292) that disagreed with Table 1 for the same quantities. Timings are wall-clock for one fold, measured by scripts/measure\_training\_cost.py and omitted where not measured. The POET figure is its full per-fold cost—the 3×3 validation grid over (factors, threshold) plus a refit and QP solve at every rebalance date, 76 fits in all—not a single estimate. Measured on one machine (Apple MPS); both absolute times and ratios depend on implementation and hardware.
<table><tr><td>Method</td><td>Params</td><td>Time</td><td>TE</td></tr><tr><td> $\mathrm { P O E T + Q P }$ </td><td>0 (tuned)</td><td>0.23 s</td><td>0.0267</td></tr><tr><td>GLasso + QP</td><td>0 (CV)</td><td></td><td>0.0269</td></tr><tr><td>ValTuned</td><td>1</td><td></td><td>0.0269</td></tr><tr><td>SPO+</td><td>1</td><td></td><td>0.0269</td></tr><tr><td>LODL</td><td>12</td><td></td><td>0.0270</td></tr><tr><td>Shrink-DFL</td><td>1</td><td>13 min</td><td>0.0268</td></tr><tr><td>Struct-DFL</td><td>12</td><td>3min</td><td>0.0267</td></tr></table>

POET’s reported per-fold cost includes the validation grid and refits. The saved timing record gives a roughly 3,329× ratio for one-parameter DFL and 804× for structured DFL; the displayed times are rounded. These measurements motivate a cost comparison, but do not imply that spending additional compute can never help a one-parameter predictor or that the ratios transfer to other implementations.

## C Equity Comparisons and Robustness

## C.1 DFL Gains across Sparsity Levels

We sweep the cardinality budget K ∈ {5, 10, 20, 30, 50} using the neural model under hard-K selection.

Table 5: Neural model: DFL vs. MSE across sparsity levels. 95% CI from 10,000-sample bootstrap; p-values from Wilcoxon signed-rank.
<table><tr><td>K</td><td>MSE TE</td><td>DFL TE</td><td>Gain</td><td>95% CI</td><td>p-value</td><td>Wins</td></tr><tr><td>5</td><td>0.0859</td><td>0.0784</td><td>-8.8%</td><td> $[ - 1 3 . 9 , - 3 . 6 ]$ </td><td>0.014</td><td>8/9</td></tr><tr><td>10</td><td>0.0510</td><td>0.0462</td><td>-9.4%</td><td> $[ - 1 3 . 1 , - 5 . 1 ]$ </td><td>0.004</td><td>8/9</td></tr><tr><td>20</td><td>0.0331</td><td>0.0276</td><td>-16.5%</td><td> $[ - 2 6 . 1 , - 7 . 1 ]$ </td><td>0.002</td><td>9/9</td></tr><tr><td>30</td><td>0.0208</td><td>0.0193</td><td>-7.0%</td><td> $[ - 1 5 . 1 , - 0 . 7 ]$ </td><td>0.004</td><td>8/9</td></tr><tr><td>50</td><td>0.0096</td><td>0.0096</td><td>-0.2%</td><td> $[ - 0 . 5 , + 0 . 1 ]$ </td><td>0.213</td><td>5/9</td></tr></table>

Table 5 and Figure 9(a) show neural DFL reductions of 7.0–16.5% at the tested $K \leq 3 0$ , with one-sided Wilcoxon $p \leq 0 . 0 1 4 .$ At K = 50, the estimated reduction is 0.2% (p = 0.213, 5/9 wins). The largest reduction occurs at $K = 2 0$ rather than the smallest portfolio. These finite comparisons support a dependence on the experimental setting, not a universal monotone law in $K / N$

Tracking-risk summaries. For the same nine neural-model folds, worst-fold TE decreases from 0.0564 to 0.0360 and the standard deviation of fold TE decreases from 0.0098 to 0.0045. Recomputed daily-tail loss and arithmetic tracking drawdown decrease by 14.3% and 15.5%, respectively (Appendix C.13). These are descriptive summaries of the evaluated folds, not prospective risk guarantees.

(a)  
![](images/28fd80cc730b115296db9ccd9738286b081bbced9c8b39a3436aa10c4bbaf426.jpg)

(b)  
![](images/cd55c0e49b246c859755329e2c76d878e20f9e4d23d9b44057f069177c0d36c0.jpg)

(c)  
![](images/623f79c529e4eb9ce0340cb12cf682fa5cbe8f420b7d7cc2e65c90de37fa6eba.jpg)  
Figure 8: Loss choice, model capacity, and selection stability. (a)–(b) Tracking error relative to POET; (c) relative to MSE (lower is better). At N=100, the three fitted one-parameter estimators improve on Ledoit– Wolf but meet the empirical equivalence criterion in Table 1. At $N { = } 4 7 8$ , estimator choice changes the ordering. Under dynamic selection, BD-DFL reduces the extreme errors of unregularized neural $\mathrm { D F L } ;$ dots show individual folds.

![](images/7ceb299dfabcf02c64f693f02d3b079c920f7971868e4fe45b41e5115747c230.jpg)

(b)  
![](images/ae1f816fac50da784a81e4a4d8653b81b23871ac903e2c88e028cf960fe1ac63.jpg)  
Figure 9: Tracking-error reduction over nine paired folds. Bars show $1 0 0 ( 1 - \overline { { \mathrm { T E } } } _ { \mathrm { D F L } } / \overline { { \mathrm { T E } } } _ { \mathrm { b a s e } } )$ ; whiskers are ±1 paired delta-method SE of this ratio, a descriptive summary across folds. Unadjusted one-sided Wilcoxon: $^ { * } p < 0 . 0 5$ $^ { * * } p < 0 . 0 1$ -2 $^ { * * * } p < 0 . 0 0 1$ . (a) Neural DFL vs. MSE: reductions of 7.0–16.5% at $K \leq 3 0$ , and 0.2% at $K = 5 0$ . (b) Shrinkage DFL vs. Ledoit–Wolf: reductions of 7.1–19.1%; comparisons with other fitted one-parameter estimators are in Table 1.

Figure 10 shows the corrected test-return series. Each 63-day window lies within one fold; windows crossing train/test boundaries are excluded.

## C.2 Per-Fold Results

Table 6 reports per-fold tracking errors for Experiment 1 (neural model, K=20), demonstrating consistency across diverse market regimes.

## C.3 Sparsity vs. Model Capacity Interaction

Our central hypothesis predicted that DFL gains depend on both sparsity (K) and model misspecification $( K _ { f } )$ . To test this, we run a full $K \times K _ { f }$ interaction grid: factor model with $K _ { f } \in$ {3, 5, 10, 20} and $K \in \{ 5 , 1 0 , 2 0 , 5 0 \}$ , yielding 16 cells × 2 losses × 9 folds = 288 experimental

runs.  
![](images/33e114d40bf3005e5a550763f50d95e9db7286508567880d6f0ef50d46fe4d2d.jpg)  
Figure 10: Corrected rolling tracking error for the neural model at $K { = } 2 0 \colon 2 { , } 0 9 6$ observations across nine test folds. Top: annualized 63-day standard deviation of tracking returns. Bottom: relative TE reduction from DFL. Dotted lines mark fold boundaries; gaps exclude each fold’s first 62 observations.

Table 6: Per-fold TE for neural model at K=20 (Experiment 1). DFL wins 9/9 folds, with the largest gains concentrated in the high-volatility regimes of 2021–2023 (−40.0% and −26.6%) and the smallest in the calmest year, 2017–2018 (−0.3%).
<table><tr><td>Fold</td><td>Test period</td><td>MSE TE</td><td>DFL TE</td><td>Gain</td></tr><tr><td>0</td><td>2016-2017</td><td>0.0260</td><td>0.0223</td><td>-14.3%</td></tr><tr><td>1</td><td>2017-2018</td><td>0.0224</td><td>0.0223</td><td>-0.3%</td></tr><tr><td>2</td><td>2018-2019</td><td>0.0306</td><td>0.0287</td><td>-6.2%</td></tr><tr><td>3</td><td>2019-2020</td><td>0.0389</td><td>0.0360</td><td>-7.5%</td></tr><tr><td>4</td><td>2020-2021</td><td>0.0279</td><td>0.0260</td><td>-6.7%</td></tr><tr><td>5</td><td>2021-2022</td><td>0.0564</td><td>0.0339</td><td>-40.0%</td></tr><tr><td>6</td><td>2022–2023</td><td>0.0394</td><td>0.0289</td><td>-26.6%</td></tr><tr><td>7</td><td>2023-2024</td><td>0.0264</td><td>0.0242</td><td>-8.4%</td></tr><tr><td>8</td><td>2024–2025</td><td>0.0299</td><td>0.0263</td><td>-11.8%</td></tr></table>

Table 7: DFL gain (%) over MSE: full $K \times K _ { f }$ interaction grid (factor model, nine paired folds per cell). Unadjusted one-sided Wilcoxon: $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ , matching Figure 11.
<table><tr><td> $K _ { f } \backslash K$  一</td><td>5</td><td>10</td><td>20</td><td>50</td></tr><tr><td>3</td><td> $- 8 . 8 \% ^ { * * }$ </td><td> $- 1 0 . 2 \% ^ { \ast \ast }$ </td><td> $- 1 5 . 9 \% ^ { \ast \ast }$ </td><td>-0.4%</td></tr><tr><td>5</td><td> $- 8 . 6 \% ^ { * * }$ </td><td> $- 9 . 8 \% ^ { * * }$ </td><td> $- 1 4 . 9 \% ^ { \ast \ast }$ </td><td>-2.1%</td></tr><tr><td>10</td><td> $- 9 . 2 \% ^ { * * }$ </td><td> $- 1 0 . 2 \% ^ { \ast \ast }$ </td><td> $- 1 5 . 1 \% ^ { \ast \ast }$ </td><td>-2.5%</td></tr><tr><td>20</td><td> $- 8 . 6 \% ^ { * }$ </td><td> $- 9 . 7 \% ^ { * * }$ </td><td> $- 1 3 . 9 \% ^ { \ast \ast }$ </td><td> $- 1 . 1 \%$ </td></tr></table>

Table 7 and Figure 11 show that relative TE changes vary more across portfolio cardinalities K than across factor counts $K _ { f }$ in this experiment. At $K = 2 0$ , DFL reduces TE by 13.9–15.9% across factor counts; at $K = 5 0 ,$ , reductions are 0.4–2.5%. Within each fixed-K column, the spread across factor counts is at most 2.1 percentage points.

This pattern is consistent with a role for decision sparsity in the difference between the two objectives. The grid alone does not isolate sparsity from predictor capacity or establish a causal mechanism. Similar relative gains across factor counts also do not imply similar absolute MSE baselines. We therefore interpret the grid as an empirical comparison alongside the conditional

geometric analysis in the main text.

![](images/f850bb38030197f7095525bcf1f04874944f1ea2f63239df277bb02f2f37cc09.jpg)  
Figure 11: Task training across factor counts and portfolio cardinalities (nine paired folds per cell). Values are relative TE changes, $1 0 0 ( \overline { { \mathrm { T E } } } _ { \mathrm { D F L } } / \overline { { \mathrm { T E } } } _ { \mathrm { M S E } } - 1 )$ ; negative values indicate improvement. In this grid, changes vary more across K than across $K _ { f }$ , and improvements are smaller at $K = 5 0$ . Unadjusted one-sided Wilcoxon: $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ . The heatmap does not identify a causal effect of either axis.

## C.4 DFL with Richer Model Classes

We introduce two richer shrinkage models: (i) Conditional shrinkage (∼385 params): $\begin{array} { r l } { \alpha _ { t } } & { { } = } \end{array}$ M $\mathbf { P } ( \mathbf { z } _ { t } )$ with market-level features, and (ii) Structured shrinkage (12 params): $\hat { \Sigma } = ( 1 - \alpha ) S +$ $\alpha T$ with learnable per-sector variance targets. For baseline comparison, we use MSE training and random search (200 trials for conditional, 500 for structured).

Table 8: Richer model classes: MSE vs. DFL, 9 folds, one-sided Wilcoxon $( ^ { * } p { < } 0 . 0 5 , \ ^ { * * } p { < } 0 . 0 1 )$ . The random-search rows of an earlier version are withheld pending a re-run: those runs stored no return series, so they cannot be placed on the corrected evaluation window that every row here uses.
<table><tr><td>K</td><td>Method</td><td>TE</td><td>TO</td><td>p</td><td>Wins</td></tr><tr><td>10</td><td>Cond. Shrinkage + MSE Cond. Shrinkage + DFL</td><td>0.0462 0.0459</td><td>0.034 0.040</td><td> $0 . 0 1 0 ^ { * * }$ </td><td>8/9</td></tr><tr><td></td><td>Struct. Shrinkage + MSE</td><td>0.0459</td><td>0.041</td><td></td><td></td></tr><tr><td></td><td>Struct. Shrinkage + DFL Cond. Shrinkage + MSE</td><td>0.0453 0.0273</td><td>0.043</td><td> $0 . 0 1 4 ^ { * }$ </td><td>8/9</td></tr><tr><td>20</td><td>Cond. Shrinkage + DFL</td><td>0.0270</td><td>0.036 0.040</td><td> $0 . 0 2 0 ^ { * }$ </td><td>7/9</td></tr><tr><td></td><td>Struct. Shrinkage + MSE</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Struct. Shrinkage + DFL</td><td>0.0270 0.0267</td><td>0.045 0.046</td><td> $0 . 0 1 0 ^ { * * }$ </td><td>8/9</td></tr></table>

Table 8 shows that DFL helps both richer model classes, and helps the smaller one more. The 12-parameter structured model gains 1.2% at K=10 and 1.6% at $K { = } 2 0 \left( p \mathrm { = } 0 . 0 0 2 , 9 / 9 \mathrm { f o l d s } \right)$ while the 385-parameter conditional model gains only 0.7% and 1.0%. Parameter count alone does not predict the relative gain. The conditional model also trains less stably (validation-to-training loss ratios range from 0.64 to 3.42 across folds). These results favor the structured model in this comparison, but do not isolate the reason: input-dependent batch geometry, optimization and model specification can all affect the ordering.

![](images/f8f1852e2bf1a7d4913294fa1923eb6cac4aedf79d5efbe4d01f965b67463a58.jpg)  
Figure 12: DFL gains for three model classes at $N \ = \ 1 0 0 ,$ using nine paired folds. Baselines are validation-tuned shrinkage (one parameter) and MSE training (structured and conditional models). Bars show relative reduction in mean TE; whiskers are ±1 paired delta-method SE. Unadjusted one-sided Wilcoxon: $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$ . The 12-parameter model has the largest observed reduction here; the ordering differs at $N = 4 7 8 \left( { \mathrm { T a b l e } } 2 \right)$ . Architecture changes alongside parameter count.

## C.5 Cross-Market Dim-1 Validation

Table 9: Cross-market dim-1 validation: the 1-parameter shrinkage sweep across three sparsity levels, re-run under the repaired harness. DFL gain stays at or below 1.76% across 5 markets, 3 sparsity levels $( K / N$ from 0.02 to 0.43) and 120 paired folds, and is unordered in $K / N ,$ , an empirical observation separate from Proposition 4 and Corollary 1. Even where gains are statistically significant (e.g., Hang Seng $K = 1 0 , p { = } 0 . 0 1 2 )$ their value depends on the application tolerance and computational budget.
<table><tr><td>Market</td><td>N</td><td>K</td><td>MSE TE</td><td>DFL TE</td><td>Shrink ∆%</td><td>p</td></tr><tr><td rowspan="3">FTSE 100</td><td>91</td><td>5</td><td>0.0896</td><td>0.0896</td><td>-0.02</td><td>0.770</td></tr><tr><td>91</td><td>10</td><td>0.0516</td><td>0.0516</td><td>+0.01</td><td>0.473</td></tr><tr><td>91</td><td>20</td><td>0.0261</td><td>0.0260</td><td>+0.17</td><td>0.039</td></tr><tr><td rowspan="3">Nikkei 225</td><td>218</td><td>5</td><td>0.1106</td><td>0.1101</td><td>+0.46</td><td>0.125</td></tr><tr><td>218</td><td>10</td><td>0.0823</td><td>0.0821</td><td>+0.17</td><td>0.039</td></tr><tr><td>218</td><td>20</td><td>0.0511</td><td>0.0511</td><td>+0.02</td><td>0.578</td></tr><tr><td rowspan="3">Euro Stoxx 50</td><td>47</td><td>5</td><td>0.0731</td><td>0.0731</td><td>+0.06</td><td>0.074</td></tr><tr><td>47</td><td>10</td><td>0.0331</td><td>0.0330</td><td>+0.29</td><td>0.055</td></tr><tr><td>47</td><td>20</td><td>0.0140</td><td>0.0140</td><td>+0.08</td><td>0.191</td></tr><tr><td rowspan="3">ASX 200</td><td>159</td><td>5</td><td>0.0778</td><td>0.0769</td><td>+1.21</td><td>0.004</td></tr><tr><td>159</td><td>10</td><td>0.0441</td><td>0.0437</td><td>+0.98</td><td>0.004</td></tr><tr><td>159</td><td>20</td><td>0.0274</td><td>0.0273</td><td>+0.38</td><td>0.004</td></tr><tr><td rowspan="3">Hang Seng</td><td>62</td><td>5</td><td>0.0803</td><td>0.0793</td><td>+1.28</td><td>0.027</td></tr><tr><td>62</td><td>10</td><td>0.0441</td><td>0.0434</td><td>+1.76</td><td>0.012</td></tr><tr><td>62</td><td>20</td><td>0.0223</td><td>0.0223</td><td>+0.42</td><td>0.055</td></tr></table>

The expanded cross-market results reveal a consistent pattern: shrinkage (1p) DFL gains stay at or below 1.76% across all 15 market-K combinations (120 paired folds), at $K / N$ from 0.02 to 0.43; only three clear $1 \% .$ . Even where Wilcoxon $p < 0 . 0 5$ (ASX $K { = } 1 0 \colon p { = } 0 . 0 0 4$ ; Hang Seng $K { = } 1 0 \colon p { = } 0 . 0 1 2 )$ , the absolute gains (+0.98%, +1.76%) are small in relative terms; the 1.76% case exceeds the diagnostic’s 1% cutoff. The largest gain (Hang Seng +1.76%) is not the largest $K / N$ in the sweep: Euro Stoxx at $K { = } 2 0$ reaches $K / N { = } 0 . 4 3$ and $\mathrm { g a i n s + 0 . 0 8 \% }$ . Within this table the gain is unordered in $K / N$ , which is not implied by Corollary 1. The runtime comparison is confined to Appendix B.3; no market-wide cost-benefit threshold is established.

Shared-run comparison with the structured model. Table 10 reports a later, self-contained run over four of these markets that trains both the 1-parameter and the 12-parameter model on the current data pull, at $K \in \{ 1 0 , 2 0 \}$ , 8 folds each. It supplies the structured-model column of Table 3, and the contrast within each row is the point: on the same folds and the same data, the two model classes have different relative changes on the same folds; the ordering and magnitude vary across markets and K.

We flag one thing an earlier draft claimed and this one does not. When both tables ran on different data pulls, their shrinkage columns constituted an independent replication, and we reported it as such. Re-running everything under the repaired harness put both on the current pull, the same splitter and the same per-fold seeds, and their overlapping cells $( K \in \{ 1 0 , 2 0 \}$ , four markets) are now identical to eight decimal places. That is a determinism check, not independent evidence, and we no longer describe it as the latter. Table 9’s remaining independent contribution is its $K { = } 5$ column and its fifth market.

Table 10: Four-market run training both model classes on one data pull (8 folds each). The shrinkage columns share a configuration with Table 9 and match it exactly where they overlap; the structured columns supply Table 3. TE is the MSE-trained baseline; ∆% is the DFL gain over it. ∗p<0.05, $^ { * * } p { < } 0 . 0 1$ , one-sided Wilcoxon.
<table><tr><td></td><td></td><td></td><td colspan="3">Shrinkage (1p)</td><td colspan="3">Structured (12p)</td></tr><tr><td>Market</td><td>N</td><td>K</td><td>TE</td><td>∆%</td><td>p</td><td>TE</td><td>∆%</td><td>p</td></tr><tr><td rowspan="2">Nikkei 225</td><td>218</td><td>10</td><td>0.0823</td><td>+0.17</td><td>0.039*</td><td>0.0826</td><td>+0.87</td><td>0.027*</td></tr><tr><td></td><td>20</td><td>0.0511</td><td>+0.02</td><td>0.578</td><td>0.0511</td><td>+0.20</td><td>0.055</td></tr><tr><td rowspan="2">Euro Stoxx 50</td><td>47</td><td>10</td><td>0.0331</td><td>+0.29</td><td>0.055</td><td>0.0329</td><td>+0.95</td><td>0.008**</td></tr><tr><td></td><td>20</td><td>0.0140</td><td>+0.08</td><td>0.191</td><td>0.0141</td><td>+0.15</td><td>0.230</td></tr><tr><td rowspan="2">ASX 200</td><td>159</td><td>10</td><td>0.0441</td><td>+0.98</td><td>0.004**</td><td>0.0436</td><td>+2.95</td><td>0.004**</td></tr><tr><td></td><td>20</td><td>0.0274</td><td>+0.38</td><td>0.004**</td><td>0.0273</td><td>+1.27</td><td>0.004**</td></tr><tr><td rowspan="2">Hang Seng</td><td>62</td><td>10</td><td>0.0441</td><td>+1.76</td><td>0.012*</td><td>0.0433</td><td>+2.77</td><td>0.008**</td></tr><tr><td></td><td>20</td><td>0.0223</td><td>+0.42</td><td>0.055</td><td>0.0223</td><td>+0.57</td><td>0.004**</td></tr></table>

## C.6 Cross-Benchmark Validation (Russell 2000 / Tech Sector)

To test whether DFL’s $K / N$ theory generalizes beyond the S&P 500, we evaluate the 1-parameter shrinkage model on two additional benchmarks: Russell 2000 (proxied by S&P 600, N=100, tracked against IWM) and S&P 500 Information Technology sector $( N { = } 4 8 .$ , tracked against XLK). Table 11 reports results.

Table 11: Cross-benchmark validation (1-parameter shrinkage). Every gain is under 0.3% across a small-cap index and a single sector, including at $K / N { = } 0 . 4 2 ;$ these rows extend the observed small-gain pattern to additional tested universes.
<table><tr><td>Benchmark</td><td>K</td><td> $K / N$ </td><td>MSE TE</td><td>DFL TE</td><td>Gain</td></tr><tr><td>S&amp;P 500 (N=100)</td><td>20</td><td>0.206</td><td>0.0269</td><td>0.0268</td><td>-0.3%</td></tr><tr><td>Russell 2000 (N=100)</td><td>10</td><td>0.100</td><td>0.0797</td><td>0.0797</td><td>-0.0%</td></tr><tr><td>Russell 2000 (N=100)</td><td>20</td><td>0.200</td><td>0.0492</td><td>0.0491</td><td>-0.2%</td></tr><tr><td>Tech Sector (N=48)</td><td>5</td><td>0.104</td><td>0.0874</td><td>0.0873</td><td>-0.1%</td></tr><tr><td>Tech Sector (N=48)</td><td>10</td><td>0.208</td><td>0.0524</td><td>0.0522</td><td>-0.3%</td></tr><tr><td>Tech Sector (N=48)</td><td>20</td><td>0.417</td><td>0.0222</td><td>0.0221</td><td>-0.2%</td></tr></table>

Across three indices spanning large-cap, small-cap and sector portfolios, the 1-parameter shrinkage model shows near-zero DFL advantage—every gain under 0.3%, including the Tech Sector at $K / N { = } 0 . 4 2 –$ —consistent with the low-capacity pattern observed here; collinearity alone does not imply redundancy.

## C.7 BD-DFL Detailed Results

Table 12: Dynamic selection at $N { = } 1 0 0$ (nine paired folds). Entries below the baseline are relative changes in mean TE; negative is better. Best $\beta$ denotes the best tested setting, an exploratory comparison.
<table><tr><td></td><td colspan="2"> $K { = } 1 0$ </td><td colspan="2"> $K { = } 2 0$ </td></tr><tr><td>Method</td><td>Shrink</td><td>Neural</td><td>Shrink</td><td>Neural</td></tr><tr><td>MSE baseline</td><td>0.0629</td><td>0.0512</td><td>0.0377</td><td>0.0327</td></tr><tr><td>Pure DFL</td><td>-5.4%</td><td>+8.6%</td><td>-0.9%</td><td>+6.7%</td></tr><tr><td>BD-DFL (best tested)</td><td>-10.3%</td><td>-8.2%</td><td>-7.0%</td><td>-11.6%</td></tr></table>

Table 13: Dynamic selection at $N { = } 4 7 8$ (five paired folds). MSE TE / pure-DFL change / best-tested BD-DFL change. Best-tested comparisons are exploratory.
<table><tr><td>Model</td><td> $K { = } 1 0$ </td><td>K=20</td></tr><tr><td>Shrinkage</td><td>一</td><td> $0 . 0 5 8 6 / - 0 . 8 \% / - 8 . 3 \%$ </td></tr><tr><td>Structured</td><td></td><td> $0 . 0 5 5 5 / - 2 . 4 \% / - 5 . 5 \%$ </td></tr><tr><td>Neural</td><td> $0 . 0 6 9 8 / + 0 . 7 \% / - 7 . 4 \%$ </td><td> $0 . 0 5 2 7 / - 0 . 1 \% / + 0 . 5 \%$ </td></tr></table>

Table 14: Corrected neural BD-DFL results at $N { = } 4 5 1$ (11 paired folds, dynamic selection). Changes use the ratio of fold means; $p$ is one-sided Wilcoxon against MSE.
<table><tr><td> $K$ </td><td>Method</td><td>Mean TE</td><td>vs. MSE</td><td>Wins</td><td> $p$ </td></tr><tr><td rowspan="3">10</td><td>MSE baseline</td><td>0.0734</td><td></td><td></td><td></td></tr><tr><td>Pure DFL</td><td>0.0638</td><td>-13.0%</td><td>8/11</td><td>0.0093</td></tr><tr><td>BD-DFL  $( \beta = 0 . 0 0 1 )$ </td><td>0.0620</td><td>-15.6%</td><td>9/11</td><td>0.0024</td></tr><tr><td rowspan="3">20</td><td>MSE baseline</td><td>0.0516</td><td></td><td></td><td></td></tr><tr><td>Pure DFL</td><td>0.0539</td><td>+4.5%</td><td>5/11</td><td>0.6499</td></tr><tr><td> $\mathrm { B D - D F L } \left( \beta = 0 . 0 0 1 \right)$ </td><td>0.0487</td><td>-5.5%</td><td>8/11</td><td>0.1030</td></tr></table>

The corrected $N = 4 5 1$ results differ from the earlier harness: unregularized DFL improves at $K = 1 0$ , while regularization adds a smaller further reduction. At $K \ : = \ : 2 0$ its direction is favorable but not significant at 5%. The choice $\beta = 0 . 0 0 1$ is empirical; proximity to $r ^ { 2 } / N ^ { 2 }$ would not establish a minimax law.

## C.8 Turnover Penalty Ablation

DFL methods produce higher turnover than MSE. A natural concern is that DFL’s TE advantage is an artifact of excessive trading. We ablate the turnover penalty $\gamma \in \{ 0 , 0 . 0 1 , 0 . 1 , 1 . 0 \}$ for the neural DFL model at $K = 2 0$

Table 15 and Figure 14 show the trade-off at the tested coefficients. $\mathrm { A t } \gamma = 0 . 0 1$ , mean TE improves by 12.8%, while turnover remains higher than MSE (0.005 vs. 0.003). Larger penalties reduce turnover and erode the observed TE advantage. These measurements do not establish dominance at every turnover level or a universal optimal penalty.

(a)  
![](images/26b8e5d5dd5aaf9e91477ca724cee90ab7104b25c210833a1a622087860a513d.jpg)

(b)  
![](images/917d602be8a2eedd5097d1ad75cc24d7f7d55110089d3ec2d96276b5a78fe8e5.jpg)  
Figure 13: Regularization under dynamic selection. (a) Shrinkage at $K = 2 0$ across the tested coefficients; dashed segments bridge untested settings. (b) Best-tested coefficient for each model and sparsity at $N = 4 7 8$ These exploratory comparisons do not establish that regularization gains grow monotonically with universe size.

Table 15: Turnover-TE Pareto frontier (Neural, K = 20).
<table><tr><td>Method</td><td>γ</td><td>TE</td><td>Turnover</td><td>vs MSE</td></tr><tr><td>MSE baseline</td><td></td><td>0.0332</td><td>0.003</td><td></td></tr><tr><td>DFL</td><td>0</td><td>0.0275</td><td>0.030</td><td>-17.1%</td></tr><tr><td>DFL</td><td>0.01</td><td>0.0290</td><td>0.005</td><td>-12.8%</td></tr><tr><td>DFL</td><td>0.1</td><td>0.0313</td><td>0.000</td><td>-5.7%</td></tr><tr><td>DFL</td><td>1.0</td><td>0.0332</td><td>0.000</td><td>-0.1%</td></tr></table>

## C.9 Transaction Cost-Aware Optimization

We augment the QP with an explicit transaction cost penalty: min $\mathsf { i } _ { \mathbf { w } } \mathbf { w } ^ { \top } \hat { \Sigma } \mathbf { w } - 2 \mathbf { w } ^ { \top } \pmb { \sigma } _ { \mathrm { i d x } } + c \| \mathbf { w } - \|$ $\mathbf { w } _ { \mathrm { p r e v } } \| _ { 1 }$ , where $\mathbf { w } _ { \mathrm { p r e v } }$ is the previous portfolio and c is the cost coefficient.

Table 16 shows that DFL’s tracking-error advantage survives transaction costs. DFL trades roughly $1 5 \times \mathrm { a s }$ much as the MSE baseline (turnover 0.030–0.046 vs. 0.002–0.003), which is the natural objection to any task-trained policy. But the cost of that turnover is small next to the TE it buys: even at $c = 1 0 0$ bps it adds only 4 bps of net cost at $K { = } 1 0$ and 3 bps at $K { = } 2 0$ , against a TE reduction of 49 and 57 bps respectively. DFL therefore wins on net cost at both cost levels and both sparsity settings, and the margin narrows only slightly as c rises.

## C.10 Design Choices and Robustness

Hard-K vs. $\ell _ { 1 }$ relaxation. Since $\mathbf { w } \geq 0$ and $\mathbf { 1 } ^ { \top } \mathbf { w } = 1$ imply $\| \mathbf { w } \| _ { 1 } = 1$ , an $\ell _ { 1 }$ penalty cannot induce sparsity in long-only portfolios. With elastic-net $\lambda \| \mathbf { w } \| _ { 2 } ^ { 2 } \mathrm { a t } \lambda = 0 . 0 1$ , the portfolio retains 97 of 100 stocks $( \mathrm { T E } = 0 . 0 0 6 5 )$ , making fair comparison with hard-K $( K { = } 2 0 , \thinspace \mathrm { T E } = \thinspace 0 . 0 3 3 2 )$ impossible. All experiments therefore use hard-K.

Fixed vs. dynamic stock selection. Dynamic selection (re-selecting stocks at each rebalance by tracking score) combined with DFL hurts performance: $\mathrm { T E } = 0 . 0 3 4 6$ (vs. 0.0287 for fixed) with turnover 6× higher. This comparison changes the selection rule as well as the resulting holdings. It suggests a stability cost for the tested dynamic rule, but does not isolate gradient feedback or establish that detachment causes the difference. Differentiation remains conditional on the selected subset in the stated protocol.

![](images/48f62a17f9f9a5faea5e786daa9866fe73ad01df22a956e89f3aff49bf8d5da8.jpg)  
Figure 14: Tracking-error–turnover trade-off for the tested neural configurations at $K = 2 0$ . Points are fold means; connectors join the sampled regularization coefficients and do not certify a continuous Pareto frontier. The MSE reference is shown separately. $\mathrm { A t } \gamma = 0 . 0 1$ , mean TE is lower than MSE while turnover is higher (0.005 vs. 0.003; Table 15); the preferred trade-off depends on trading costs.

Table 16: Transaction cost experiment. Net $\mathrm { c o s t } = \mathrm { T E } + c \times$ TO. All rows use the TC-aware QP. Val-tuned grid search is omitted: no stored run for it survives, and comparing a window-corrected DFL row against an uncorrected baseline would be invalid.
<table><tr><td>K</td><td>Method</td><td>c (bps)</td><td>TE</td><td>TO</td><td>Net Cost</td></tr><tr><td rowspan="3">10</td><td>Neural MSE + TC</td><td>10</td><td>0.0510</td><td>0.003</td><td>0.0510</td></tr><tr><td>Neural DFL + TC</td><td>10</td><td>0.0462</td><td>0.041</td><td>0.0462</td></tr><tr><td>Neural MSE + TC Neural DFL + TC</td><td>100 100</td><td>0.0510 0.0462</td><td>0.001</td><td>0.0511</td></tr><tr><td rowspan="3">20</td><td></td><td></td><td></td><td>0.041</td><td>0.0466</td></tr><tr><td>Neural MSE + TC Neural DFL + TC</td><td>10 10</td><td>0.0331 0.0277</td><td>0.002 0.026</td><td>0.0331 0.0278</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Neural MSE + TC Neural DFL + TC</td><td>100 100</td><td>0.0331 0.0277</td><td>0.000 0.025</td><td>0.0331 0.0280</td></tr></table>

Market regime analysis. Table 17 reports regime-conditional tracking errors.

Table 17: Regime analysis: Neural DFL vs. MSE at $K = 2 0$

<table><tr><td>Regime</td><td>MSE TE</td><td>DFL TE</td><td>Gain</td><td>p-value</td></tr><tr><td>High-volatility</td><td>0.0343</td><td>0.0300</td><td>-12.7%</td><td>0.057</td></tr><tr><td>Low-volatility</td><td>0.0308</td><td>0.0276</td><td>-10.5%</td><td>0.008</td></tr></table>

Covariance window sensitivity. We ablate the lookback window for realized covariance estimation across {21, 42, 63, 126} trading days (neural model, K=20, 10 rolling folds with 2-year training window). DFL’s gain is stable across all windows: +12.8% for the 21-, 42- and 63-day windows and +13.9% for the 126-day window (all $p \leq 0 . 0 0 2 )$ . The MSE baseline sits at $\mathrm { T E } = 0 . 0 3 2 6$ regardless of window length—the neural model learns its own covariance representation—while DFL reaches 0.0281–0.0284. The improvement persists across these archived lookback settings; these runs use a different fold protocol from the primary comparison.

Rebalancing frequency sensitivity. We ablate the test-time rebalancing frequency at weekly, biweekly and monthly intervals (neural model, $K { = } 2 0 , 9$ rolling folds; training uses weekly rebalancing in all cases). The tested rebalancing schedules give similar relative gains: +14.9% weekly (MSE 0.0325 → DFL 0.0277), +14.4% biweekly $( 0 . 0 3 2 5 ~  ~ 0 . 0 2 7 8 )$ and +14.7% monthly $( 0 . 0 3 3 1  0 . 0 2 8 2 )$ , all at $p = 0 . 0 0 2$ (Wilcoxon). The three gains agree to within half a percentage point, an empirical comparison within these three schedules. Turnover per rebalance increases as the interval lengthens (DFL: 0.013 weekly, 0.019 biweekly, 0.024 monthly), reflecting that each rebalance absorbs a larger drift when intervals are longer. The improvement is observed under all three tested schedules.

Position size constraints. Our main experiments impose $w _ { i } \geq 0$ and $\textstyle \sum w _ { i } = 1$ but no upper bound. To test robustness under realistic portfolio constraints, we add $w _ { i } \leq w _ { \mathrm { m a x } }$ to the QP and re-evaluate neural DFL vs. MSE at $K { = } 2 0$ . With $w _ { \mathrm { m a x } } = 1 0 \%$ , DFL gain is +12.7% $( p = 0 . 0 0 2$ 9/9 folds); with $w _ { \mathrm { m a x } } = 2 0 \%$ it is +14.7% $( p = 0 . 0 0 2 , 9 / 9 )$ , matching the unconstrained case (+14.7%) to the decimal. The tighter $w _ { \mathrm { m a x } } { = } 1 0 \%$ constraint raises both MSE and DFL tracking errors because it forces the QP away from the optimal solution, but the relative DFL advantage is preserved. The relative improvement persists under both tested position caps.

## C.11 Negative Results: Multi-Step Lookahead and Curriculum Learning

We evaluated two training extensions that did not improve performance:

Multi-step lookahead. Chaining 2–3 consecutive $\mathrm { Q P }$ solves per training step (backpropagating through the trajectory $\mathbf { w } _ { t _ { 1 } } ^ { * }  \mathbf { w } _ { t _ { 2 } } ^ { * }  \mathbf { w } _ { t _ { 3 } } ^ { * } )$ did not reduce tracking error beyond single-step DFL. We tested lookahead depths of 2 and 3 with both shrinkage and conditional shrinkage models at $K \in \{ 1 0 , 2 0 \}$ . In all cases, multi-step TE was within 0.1% of the single-step baseline. This finite ablation does not identify why lookahead failed to help; dependence across dates and optimizer settings remain possible factors.

Curriculum learning (MSE → task). We trained with a curriculum that begins with pure MSE loss, linearly transitions to task loss over 5–10 epochs, then continues with pure task loss. The intuition is that MSE provides a smoother loss landscape for initial optimization. However, the curriculum produced identical final TE to direct task-loss training. The task loss is not proved convex in the shrinkage parameter, and the ablation does not identify a basin-of-attraction mechanism.

## C.12 Value of the QP Structure

An alternative to predict-then-optimize is to directly predict portfolio weights from features, bypassing the QP entirely. We test this with an MLP (same architecture as the neural model) that outputs K-sparse softmax weights, trained end-to-end on tracking error. At $K { = } 2 0$ , direct weight prediction achieves $\mathrm { T E } = 0 . 0 5 8 9$ , more than double the DFL pipeline’s 0.0276; at $K { = } 1 0$ the gap is 0.0806 vs. 0.0462. Both a normalized softmax and the QP can enforce budget and non-negativity constraints. The observed comparison favors the tested QP pipeline, but also changes the parameterization and optimization problem. It therefore does not isolate which aspect of the architecture causes the difference.

## C.13 Risk Metrics and Domain-Specific Analysis

Tracking error summarizes average dispersion but can conceal tail losses and variation across folds.   
We therefore examine additional tracking-risk measures on the same matched neural-model folds;

these retrospective measures do not establish prospective investment performance.

Comprehensive risk metrics. Table 18 reports risk metrics at K=20 computed from daily portfolio returns. The reported reductions describe variability of tracking returns, not expected monetary savings.

Table 18: Tracking-risk metrics at K = 20 from exactly nine neural-model folds. Standard deviations use the sample convention; daily tail loss and arithmetic tracking drawdown are computed within folds. Negative relative change denotes a reduction. The common naive baseline is excluded.
<table><tr><td>Metric</td><td>MSE</td><td>DFL</td><td>Improvement</td></tr><tr><td>Annualized TE</td><td>0.0331</td><td>0.0276</td><td>-16.5%</td></tr><tr><td>21-day block TE (ann.)</td><td>0.0296</td><td>0.0258</td><td>-12.7%</td></tr><tr><td>Worst 21-day TE</td><td>0.0538</td><td>0.0451</td><td>-16.1%</td></tr><tr><td>Max tracking drawdown</td><td>0.0362</td><td>0.0306</td><td>-15.5%</td></tr><tr><td> $\mathrm { C V a R _ { 5 \% } }$  (daily)</td><td>0.0044</td><td>0.0038</td><td>-14.3%</td></tr><tr><td>Worst-fold TE</td><td>0.0564</td><td>0.0360</td><td>-36.3%</td></tr><tr><td>TE std across folds</td><td>0.0098</td><td>0.0045</td><td>-54.0%</td></tr></table>

Per-year regime breakdown. Each fold’s test window is one year, so Table 6 is already a per-year breakdown; we do not repeat it here. Reading it as a regime series, DFL’s gain peaks during volatile regime shifts (2021–2023: 26.6–40.0%) and is smallest in the calmest stretch (2017–2018: 0.3%, a small change), This descriptive variation does not identify covariance stability as its cause.

Factor exposure. A factor-neutrality claim requires the factor construction, aligned return series, regression specification and uncertainty estimates. The released results do not provide a complete auditable regression artifact, so we do not claim that the tracking improvement is factor-neutral.

## D Capacity Diagnostics and Spatial Transfer

## D.1 Practitioner Diagnostic

Definition 1 (Heterogeneity-based capacity proxy). For a structured model with d trainable parameters and a specified partition into R regions, let $m _ { r }$ be the target index-weight mass in region r. Define $h = \operatorname { s d } ( m _ { 1 } , \ldots , m _ { R } ) / \operatorname { m e a n } ( m _ { 1 } , \ldots , m _ { R } )$ and $d _ { \mathrm { p r o x y } } = d h$ when the mean is positive. This proxy depends on the partition and is neither a Jacobian rank nor bounded by d. Archived equity predictions used $h = 1$ by convention; measured heterogeneity and the resulting post-hoc re-scoring are reported separately in Appendix D.1.

The archived diagnostic is a hypothesis-generating rule. A practical assessment should distinguish direct gradient measurements from proxy values:

1. Specify the input, parameterization, loss and aggregation unit. Measure output-space support energy and, when feasible, the pointwise and stacked Jacobians or the actual aggregate gradient angle.

2. Use $\sqrt { 2 K / N }$ only as an isotropic reference; it is not a measured cosine. The deterministic support-energy ratio is valid without that isotropy assumption.

3. Compare inexpensive baselines and task-trained models on held-out folds with paired settings. An observed small gain and a non-significant test do not establish equal minima.

![](images/0cc08a5cb7a121a371fa0d42873b2445c163b9330535a1216c9e6a60ca25b308.jpg)

![](images/eb87a65361bc09078425f12353663963b1c1b35d842effbf75333990997dd22e.jpg)  
Finance 1p Finance 12p NOAA EPA Incorrect prediction  
Figure 15: Archived blind-evaluation record (33 matched cases). Colour shows DFL gain: red $> 1 \% .$ grey within ±1%, blue $< - 1 \%$ . (a) Four incorrect predictions are ringed; overlapping finance points are counted. (b) Equal spacing denotes the four sampled categories, not a continuous sweep; horizontal marks show means. This historical snapshot includes two abstentions.

4. Treat the proxy thresholds 1.5 and 4 as historical calibration choices. Neither the dh proxy nor a measured spectral rank establishes a universal performance threshold. Recalibration requires separate validation data.

5. Under changing selection, tune covariance regularization on validation data and retain unstable runs in the report. The best tested coefficient on evaluation outcomes is an exploratory result, not a deployable selection rule.

Selected standard estimators. The released script probes six chosen covariance parameterizations on 252 days of returns for N=97 assets (scripts/measure\_deployed\_deff.py). The reported quantity is the entropy rank of a 24-row Gaussian projection of the Jacobian, rather than its full singular spectrum. It is distinct from the proxy $d \cdot h ;$ the sketch introduces additional approximation error. This sample is not a survey of deployment prevalence. Discrete hyperparameters contribute no differentiable Jacobian direction.

Table 19: Spectral-rank estimates from 24 Gaussian Jacobian probes for six selected parameterizations. “Low” and “higher” describe this sketch, not a theorem about DFL performance. Zero denotes no trainable parameter. The sample does not establish deployment prevalence.
<table><tr><td>Estimator</td><td>d</td><td> $r _ { \mathrm { e f f } }$  (sketch)</td><td>Sketch class</td></tr><tr><td>Sample covariance</td><td>0</td><td>0.00</td><td>low</td></tr><tr><td>Ledoit-Wolf</td><td>1</td><td>1.00</td><td>low</td></tr><tr><td>RiskMetrics (EWMA)</td><td>1</td><td>1.00</td><td>low</td></tr><tr><td>Constant correlation</td><td>1</td><td>1.00</td><td>low</td></tr><tr><td>Two-target shrinkage</td><td>2</td><td>1.29</td><td>low</td></tr><tr><td>Structured shrinkage (12p)</td><td>12</td><td>8.03</td><td>higher</td></tr></table>

The three measured one-parameter families have rank-one Jacobians, so their nonzero parameter gradients satisfy Proposition 4. This geometric fact does not establish equal task optima or a universal cost advantage. The runtime comparison in Appendix B.3 concerns the tested implementation and configuration.

The archived scoring file contains 33 cases. The revised rule makes 31 committed predictions, of which 27 are correct, and abstains on two cases. Its historical 29/33 score counts both abstentions as correct and should not be interpreted as accuracy on 33 committed predictions. These archived outcomes predate the corrected equity harness.

Threshold sensitivity. The diagnostic thresholds $( d _ { \mathrm { p r o x y } } \leq 1 . 5 , d _ { \mathrm { p r o x y } } \geq 4 _ { \cdot }$ , alignment $< 0 . 4 )$ were examined post hoc: sweeping $d _ { \mathrm { l o w } } \in [ 1 . 0 , 3 . 0 ] , d _ { \mathrm { h i g h } } \in [ 3 . 0 , 7 . 0 ]$ , alignment $\in [ 0 . 2 5 , 0 . 5 0 ]$ at 0.5/0.5/0.05 steps, using the archived convention that counts abstentions as correct, 73.3% of all 270 parameter combinations achieve ≥ 85% accuracy, and accuracy never falls below 78.8% anywhere on the grid (median 87.9%, max 97.0%). Setting $d _ { \mathrm { l o w } } = 1 . 0$ yields 93.9% but by classifying more cases as “ambiguous” (non-committal); our threshold of 1.5 makes the diagnostic more informative by issuing substantive predictions for low-heterogeneity domains $( d _ { \mathrm { p r o x y } } = 1 . 4$ for weather).

Is h doing any work on the equity corpus? A fair objection to $d _ { \mathrm { p r o x y } } = d \cdot h$ is that our equity experiments set h=1 by convention, so that on the corpus carrying most of the paper $d _ { \mathrm { p r o x y } }$ reduces to d and the extra factor explains nothing. We test this directly. Applying the same definition the synthetic sweep uses—h is the coefficient of variation of per-region index-weight mass, with regions given by the structured model’s sector partition—to the actual index weights gives Table 20. No equity universe is anywhere near h=1: the measured values span 0.31 to 0.71, bracketing EPA (0.61) and sitting well above NOAA (0.14).

Table 20: Measured index-weight heterogeneity h per universe (scripts/measure\_heterogeneity.py). The h=1 used in the main results is a conservative convention, not a measurement; every universe is in fact less heterogeneous than that.
<table><tr><td>Universe N</td><td>R</td><td>measured h</td></tr><tr><td>S&amp;P 500 (N=451)</td><td>451 11</td><td>0.314</td></tr><tr><td>S&amp;P 500 (N=478)</td><td>478 11</td><td>0.323</td></tr><tr><td>S&amp;P 500 (N=465)</td><td>465 11</td><td>0.342</td></tr><tr><td>S&amp;P 500 20y</td><td>100 11</td><td>0.378</td></tr><tr><td>Nikkei 225</td><td>218 11</td><td>0.391</td></tr><tr><td>FTSE 100</td><td>91 11</td><td>0.407</td></tr><tr><td>S&amp;P 500 top-100</td><td>97 11</td><td>0.477</td></tr><tr><td>ASX 200</td><td>162 11</td><td>0.505</td></tr><tr><td>Euro Stoxx 50</td><td>47 9</td><td>0.636</td></tr><tr><td>Hang Seng</td><td>62 11</td><td>0.713</td></tr><tr><td colspan="3">NOAA weather (reference) EPA PM2.5 (reference)</td><td>0.144 0.606</td></tr></table>

Post-hoc re-scoring with measured heterogeneity gives 25/28 correct committed predictions (89.3%), compared with 27/31 (87.1%) under the archived convention. The committed case sets differ, so the percentages are not a paired estimate of improved diagnostic accuracy. The revised rule abstains more often and was examined after outcomes were available. Measured h is reproducible from weights and a region partition, but these data do not establish that it explains the performance differences.

Three of the 38 one-parameter configurations clear a 1% gain, and they come from two markets: Hang Seng (+1.76% at K=10, +1.28% at K=5) and ASX 200 (+1.21% at K=5). We have no measurement that explains them, and we list what we ruled out rather than offer a mechanism we cannot support. Measured h does not: Euro Stoxx carries the second-largest h in the corpus (0.636, against Hang Seng’s 0.713 and ASX’s 0.505) and its three gains are +0.06%, +0.29% and +0.08%. Sparsity does not: ASX at K=5 sits at $K / N { = } 0 . 0 3 1$ , among the smallest ratios we test.

Universe size does not: Euro Stoxx is the smallest universe at $N { = } 4 7$ . An earlier draft attributed these gains to h on the strength of Hang Seng alone; with the sweep re-run and ASX also above 1%, that attribution no longer survives its own table. What does survive is the conclusion that matters here: every one of these gains is under 1.8%, far below what a 100–1000× training overhead could justify.

## D.2 Testing the Spectral-Gap Bound Directly

The saved experiment directly evaluates Theorem 1 from the predictor Jacobian and the two output gradients. The computation is pointwise in the rebalance input.

For each (model, fold, rebalance date) we form $\mathbf { J } = \partial \mathrm { v e c } ( \hat { \Sigma } ) / \partial \pmb { \theta }$ by forward-mode differ entiation $( d \ll N ^ { 2 }$ , so this costs $O ( d )$ passes), take its SVD, compute g<sub>MSE</sub> analytically and $\mathbf { g } _ { \mathrm { t a s k } }$ through the same differentiable QP the experiments train with, and compare the realised $| \cos ( \mathbf { J } ^ { \top } \mathbf { g } _ { \mathrm { M S E } } , \mathbf { J } ^ { \top } \mathbf { g } _ { \mathrm { t a s k } } ) |$ against cos Θ. Across 162 points on the $N { = } 9 7$ universe there are no recorded violations within numerical precision. The bound is informative at 108 rank-one or nearrank-one points and vacuous at all 54 structured-model points; vacuous cases do not provide a quantitative validation.

The rank-one endpoint. For the d=1 shrinkage model the bound evaluates to 1 and the realised alignment is 1.000000 at all 54 points, with zero slack. This is the theorem’s endpoint and it is reproduced to the precision of the arithmetic.

Pointwise rank one with 385 parameters. The conditional model has median singular-value ratio approximately $1 . 2 \times 1 0 ^ { - 6 }$ and absolute alignment rounding to one at all 54 sampled points. It maps each input to a scalar shrinkage intensity, so its covariance Jacobian is rank at most one for that input. Across inputs, the gradient of the intensity can rotate in parameter space. Consequently, this measurement does not identify the stacked rank or explain the ordering of test gains between the conditional and structured models. It establishes that parameter count is not pointwise rank; Proposition 5 describes the additional information needed for a training-level interpretation.

The near-rank-one bound is vacuous where it would be useful. For the $d { = } 1 2$ structured model the gap is genuinely open (median 0.089) and the alignment is high but not unity (median 0.869, minimum 0.625)—the regime the theorem’s second part is meant to describe. There $\Theta \ge \pi / 2$ at every one of the 54 points, so the bound returns no information. The cause is that $\rho _ { \mathrm { t a s k } }$ saturates: g<sub>task</sub> sits almost orthogonal to the leading singular direction $\mathbf { u } _ { 1 } \left( \left| \cos { \right| } \lesssim \sigma _ { 2 } / \sigma _ { 1 } \right)$ , while g<sub>MSE</sub> does not. The bound assumes both gradients retain non-trivial alignment with u<sub>1</sub>, and the task gradient does not.

We report this rather than omit it. The theorem is not contradicted—no point violates it—but its quantitative content is confined to the rank-one endpoint, and the intermediate regime is carried by the empirical diagnostic rather than by the bound. Sharpening it for gradients that are nearorthogonal to $\mathbf { u } _ { 1 }$ is the obvious next piece of theory.

## D.3 Synthetic $d _ { \mathrm { p r o x y } }$ Sweep

Table 21 and Figure 16 report a controlled heterogeneity intervention; changing index weights also changes task difficulty. We fix N=100, K=20, 10 regions, 5 folds, and vary only the weight heterogeneity h (via Dirichlet sampling of region weights), sweeping $d _ { \mathrm { p r o x y } }$ from 0.5 to 20 across 18 settings.

Synthetic heterogeneity sweep (N= 100, K = 20, 18 settings 5 folds)  
![](images/bfdf032eb77b5c7dc4c7cc1ed0dccbfda562b41c7d5b33a1dd4e35672b856b27.jpg)  
Figure 16: Synthetic heterogeneity sweep at $N = 1 0 0 , K = 2 0 \colon$ 18 settings, five folds each. Faint dots show every fold; larger markers show mean per-fold TE reduction and ±1 SEM. The ±1.6% band is a descriptive reference, not a confidence interval; individual folds can lie outside it. Mean gains are larger at several settings above the empirical split at $d _ { \mathrm { p r o x y } } = 9 . 5$ , but fall to approximately 1.0% at the final setting. The split is not a validated universal threshold.

The first twelve settings $( d _ { \mathrm { p r o x y } } ~ \leq ~ 9 )$ have mean gains of approximately −0.1–1.6%. Five of the next six settings have means of 2.5–4.7%, while the final setting returns to approximately 1.0%. Baseline TE also changes with heterogeneity, so these measurements do not isolate effective dimension from task difficulty or establish a sharp universal transition.

## D.4 Generalization beyond Portfolio Optimization

The deterministic support-energy inequality applies whenever the downstream loss depends on a restricted set of output coordinates. Its numerical isotropy bound requires an additional assumption and does not follow from selection alone. We examine one matching spatial QP and discuss other potential applications.

Sensor placement (empirical validation). Sensor placement is a canonical spatial selection problem (Krause et al., 2008; Joshi and Boyd, 2009). We validate on a synthetic sensor network: N sensors on a unit square with exponential spatial covariance, selecting K sensors to track the field average (identical QP formulation). A structured shrinkage model with 10 parameters (9 spatial regions + global α) is trained under MSE and DFL. Table 22 shows results across 10 configurations with 5 trials each.

Across the ten displayed configurations, median | cos | is approximately 0.346 at $K / N = 0 . 0 5$ and 0.606 at $K / N = 0 . 5 0$ . All ten rows report a lower TE under DFL, but two measured cosines exceed the displayed isotropic reference. These observations do not establish the energy assumption or a universal sparsity law; the deterministic support-energy inequality remains the appropriate statement without that assumption.

Facility location. Selecting K facilities from N sites given predicted demand covariance. The allocation QP depends on demand correlations among selected sites, but MSE prediction does not distinguish selected from unselected.

Sparse resource allocation. Any optimization that distributes resources across $K \ll N$ options based on a predicted parameter matrix (e.g., sparse Markowitz with general objectives, sparse experimental design, or sparse regression with downstream optimization).

The key structural requirement is: (i) the optimization depends on a K-dimensional sub-problem of an N-dimensional prediction, (ii) the sub-problem selection is fixed or detached, and (iii) the optimization admits efficient differentiation (e.g., KKT conditions). For each application, the actual support, differentiability, and stability assumptions must be checked separately. We do not establish a general grid-search sufficiency theorem or a universal regret guarantee for these applications.

Table 21: Synthetic heterogeneity sweep (18 settings, five folds each). Mean gains are larger at several high-heterogeneity settings, with an exception at the final setting. This exploratory pattern does not by itself establish a universal phase transition.
<table><tr><td> $h$ </td><td> $d _ { \mathrm { p r o x y } }$ </td><td> $\mathrm { T E } _ { \mathrm { M S E } }$ </td><td> $\mathrm { T E } _ { \mathrm { D F L } }$ </td><td>Gain</td><td> $\sigma _ { \mathrm { g a i n } }$ </td></tr><tr><td>0.05</td><td>0.51</td><td>4.296</td><td>4.286</td><td>+0.2%</td><td>0.9%</td></tr><tr><td>0.12</td><td>1.20</td><td>4.251</td><td>4.246</td><td>+0.1%</td><td>1.2%</td></tr><tr><td>0.16</td><td>1.58</td><td>4.145</td><td>4.108</td><td>+0.9%</td><td>0.8%</td></tr><tr><td>0.21</td><td>2.13</td><td>4.169</td><td>4.175</td><td>-0.1%</td><td>0.9%</td></tr><tr><td>0.24</td><td>2.41</td><td>3.805</td><td>3.758</td><td>+1.2%</td><td>1.1%</td></tr><tr><td>0.28</td><td>2.84</td><td>3.962</td><td>3.925</td><td>+0.9%</td><td>1.2%</td></tr><tr><td>0.41</td><td>4.08</td><td>3.766</td><td>3.704</td><td>+1.6%</td><td>2.0%</td></tr><tr><td>0.51</td><td>5.08</td><td>3.457</td><td>3.442</td><td>+0.4%</td><td>0.5%</td></tr><tr><td>0.59</td><td>5.94</td><td>3.348</td><td>3.295</td><td>+1.5%</td><td>1.9%</td></tr><tr><td>0.68</td><td>6.80</td><td>3.474</td><td>3.477</td><td>-0.1%</td><td>1.1%</td></tr><tr><td>0.79</td><td>7.89</td><td>2.410</td><td>2.375</td><td>+1.5%</td><td>1.2%</td></tr><tr><td>0.89</td><td>8.88</td><td>3.309</td><td>3.282</td><td>+0.8%</td><td>2.2%</td></tr><tr><td>1.00</td><td>9.98</td><td>2.098</td><td>2.017</td><td>+3.9%</td><td>1.4%</td></tr><tr><td>1.19</td><td>11.85</td><td>1.975</td><td>1.926</td><td>+2.5%</td><td>0.3%</td></tr><tr><td>1.40</td><td>14.02</td><td>1.784</td><td>1.725</td><td> $+ \mathbf { 3 . 3 \% }$ </td><td>1.4%</td></tr><tr><td>1.60</td><td>15.98</td><td>1.297</td><td>1.254</td><td> $+ \mathbf { 3 . 3 \% }$ </td><td>1.9%</td></tr><tr><td>1.82</td><td>18.16</td><td>0.852</td><td>0.811</td><td>+4.7%</td><td>3.2%</td></tr><tr><td>2.02</td><td>20.17</td><td>1.101</td><td>1.090</td><td>+1.0%</td><td>1.3%</td></tr></table>

NOAA weather stations vs. EPA air quality (empirical validation). Both use the same 10- parameter structured spatial model—the domains also differ in observations, targets, and indexweight heterogeneity h. NOAA stations $( N { \in } \{ 1 0 0 , 1 8 9 , 4 1 5 \}$ , temperature tracking): uniform coverage yields $h { = } 0 . 1 4 , d _ { \mathrm { p r o x y } } { = } 1 . 4$ . EPA PM2.5 monitors $( N { \in } \{  1 2 4 , 1 3 2 \}$ , pollution index tracking): urban concentration yields $h { = } 0 . 6 1 , d _ { \mathrm { p r o x y } } { = } 6 . 1$ Table 23 reports the exploratory contrast: The observed changes are negative for NOAA and positive for EPA; the small number of windows precludes a strong inferential claim.

Table 22: Sensor placement: gradient alignment and DFL gain across $K / N _ { ☉ }$ . The column $\sqrt { 2 K / N }$ is a reference under the energy assumption, not a universal bound on these measured cosines.
<table><tr><td>N</td><td>K</td><td> $K / N$ </td><td>Reference</td><td>|cos |</td><td>DFL Gap</td></tr><tr><td>200</td><td>10</td><td>0.05</td><td>0.316</td><td>0.297</td><td>-2.7%</td></tr><tr><td>100</td><td>5</td><td>0.05</td><td>0.316</td><td>0.394</td><td>-0.9%</td></tr><tr><td>50</td><td>5</td><td>0.10</td><td>0.447</td><td>0.295</td><td>-1.1%</td></tr><tr><td>100</td><td>10</td><td>0.10</td><td>0.447</td><td>0.474</td><td>-1.5%</td></tr><tr><td>50</td><td>10</td><td>0.20</td><td>0.632</td><td>0.349</td><td>-0.9%</td></tr><tr><td>100</td><td>20</td><td>0.20</td><td>0.632</td><td>0.499</td><td>-0.8%</td></tr><tr><td>200</td><td>40</td><td>0.20</td><td>0.632</td><td>0.617</td><td>-3.2%</td></tr><tr><td>50</td><td>25</td><td>0.50</td><td>1.000</td><td>0.507</td><td>-0.4%</td></tr><tr><td>100</td><td>50</td><td>0.50</td><td>1.000</td><td>0.606</td><td>-2.2%</td></tr><tr><td>200</td><td>100</td><td>0.50</td><td>1.000</td><td>0.670</td><td>-0.3%</td></tr></table>

Table 23: Exploratory spatial comparisons with the $d _ { \mathrm { p r o x y } }$ heuristic. Same 10-parameter structured model, different heterogeneity. NOAA $( d _ { \mathrm { p r o x y } } { = } 1 . 4 )$ : DFL hurts. EPA $( d _ { \mathrm { p r o x y } } { = } 6 . 1 )$ : DFL helps. The n column is the point of caution: these domains provide one evaluation window each (two for EPA at N=124), so each row has only one or two evaluation windows. We report them as the direction the diagnostic predicts, not as significance, and the equity results carry the statistical weight.
<table><tr><td>Domain</td><td> $d _ { \mathrm { p r o x y } }$ </td><td>N</td><td>K</td><td>MSE</td><td>DFL</td><td>∆%</td><td>n</td></tr><tr><td>NOAA</td><td>1.4</td><td>100</td><td>20</td><td>20.03</td><td>20.76</td><td>-3.6%</td><td>1</td></tr><tr><td>NOAA</td><td>1.4</td><td>189</td><td>20</td><td>14.87</td><td>15.14</td><td>-1.8%</td><td>1</td></tr><tr><td>NOAA</td><td>1.4</td><td>415</td><td>20</td><td>14.41</td><td>14.76</td><td>-2.5%</td><td>1</td></tr><tr><td>EPA</td><td>6.1</td><td>132</td><td>10</td><td>33.94</td><td>31.98</td><td>+5.8%</td><td>1</td></tr><tr><td>EPA</td><td>6.1</td><td>132</td><td>50</td><td>24.39</td><td>22.09</td><td>+9.4%</td><td>1</td></tr><tr><td>EPA</td><td>6.1</td><td>124</td><td>50</td><td>13.83</td><td>13.27</td><td>+4.0%</td><td>2</td></tr></table>

## E Combinatorial Experiments

## E.1 Shortest Path Experiment Details

Setup. We construct an $8 \times 8$ grid graph (N=64 vertices) with 8-connected edges. Each instance consists of features $\boldsymbol { x } \in \mathbb { R } ^ { 2 0 }$ drawn i.i.d. from $\mathcal { N } ( 0 , I )$ , mapped to positive vertex costs via a fixed ground-truth network: $c = \mathrm { s o f t p l u s } ( W x + b + \varepsilon )$ where $W \in \mathbb { R } ^ { 6 4 \times 2 0 }$ and $\varepsilon \sim \mathcal { N } ( 0 , 0 . 0 2 ^ { 2 } I )$ ). The optimal path $z ^ { * } = \arg \operatorname* { m i n } _ { z \in \mathcal { P } } c ^ { \top } z$ is solved by Dijkstra’s algorithm (source: vertex 0, target: vertex 63; edge cost = the entered vertex cost). We generate 8K training and 1.5K test instances.

Bottleneck architecture. Each model maps x 7→ cˆ through a linear bottleneck of width k: $\hat { c } =$ softplus(Dec(Enc(x))) where Enc : $\mathbb { R } ^ { 2 0 } \overset { \cdot } { \to } \mathbb { R } ^ { k }$ and Dec : $\mathbb { R } ^ { k } \to \mathbb { R } ^ { 6 4 }$ . The bottleneck width k controls the Jacobian rank: rank $( \partial \hat { c } / \partial x ) \le k$ by construction, bounding, but not fixing, the input spectral effective rank $r _ { \mathrm { i n } }$ . We test $k \in \{ 1 , 2 , 5 , 1 0 , 2 0 \}$ plus a full-rank linear $( k { = } 2 0$ , no bottleneck) and a 2-hidden-layer MLP (128 units, ∼27K parameters). All use the softplus activation on the output to ensure positive cost predictions.

Training. Models are trained with Adam $( \mathrm { l r } = 1 0 ^ { - 3 }$ for MSE, $3 \times 1 0 ^ { - 4 }$ for SPO+, weight decay $1 0 ^ { - 5 } )$ , cosine annealing over 80 epochs, batch size 128, gradient clipping at norm 5.0. SPO+ loss uses the correct minimization formulation: $\ell _ { \mathrm { S P O + } } ( \hat { c } , c , z ^ { * } ) = ( 2 \hat { c } - c ) ^ { \top } ( z ^ { * } - z _ { \mathrm { s p o } } )$ where $z _ { \mathrm { s p } 0 } =$ arg $\begin{array} { r } { \operatorname* { m i n } _ { z \in \mathcal { P } } ( 2 \hat { c } - c ) ^ { \top } z } \end{array}$ .

Table 24: Archived shortest-path results on an $8 \times 8$ grid. $r _ { \mathrm { i n } }$ measures input sensitivity, not parametergradient rank. Regret is excess true path cost divided by optimal true path cost. $\Delta$ is MSE minus SPO+ regret in percentage points, averaged over five seeds.
<table><tr><td>Model</td><td>Params</td><td> $r _ { \mathrm { i n } }$ </td><td>MSE Reg.%</td><td> $\mathrm { S P O + R e g . \% }$ </td><td> $\Delta$ </td></tr><tr><td>k=1</td><td>149</td><td>1.0</td><td>53.8%</td><td>54.2%</td><td>-0.4%</td></tr><tr><td>k=2</td><td>234</td><td>2.0</td><td>58.4%</td><td>56.1%</td><td>+2.3%</td></tr><tr><td>k=5</td><td>489</td><td>4.8</td><td>63.8%</td><td>64.0%</td><td>-0.2%</td></tr><tr><td>k=10</td><td>914</td><td>9.1</td><td>72.5%</td><td>70.0%</td><td> $+ 2 . 6 \%$ </td></tr><tr><td>k=20</td><td>1764</td><td>15.3</td><td>77.4%</td><td>74.2%</td><td> $+ 3 . 2 \%$ </td></tr><tr><td>Full linear</td><td>1344</td><td>15.3</td><td>77.4%</td><td>73.3%</td><td> $+ 4 . 1 \%$ </td></tr><tr><td>Full MLP</td><td>27K</td><td>15.1</td><td>77.4%</td><td>75.9%</td><td> $+ 1 . 5 \%$ </td></tr></table>

(a)  
![](images/88487acbb69777e9b5e816bb6a1468352bebbd17c1f7efb123924bd6392f0ecb.jpg)

(b)  
![](images/f5dc80bd36b2da1437cca6473afc1b1ed3a834d2524b7429f6062f403105faff.jpg)  
Figure 17: Shortest-path experiment on an $8 \times 8$ grid with five paired seeds. (a) Mean SPO+ advantage in percentage points, $\Delta = \mathrm { r e g r e t } _ { \mathrm { M S E } } - \mathrm { r e g r e t } _ { \mathrm { S P O } + } ,$ with ±1 SEM. Bottleneck width k bounds input-Jacobian rank; full linear and MLP models also change architecture. The low-rank point has a small negative estimate, while intermediate ranks are mixed. (b) Mean normalized regret for the same paired runs; labels give the SPO+ change in percentage points. Shading describes sampled rank ranges, not a theorem about test regret.

Spectral effective rank measurement. For each trained model, we compute $r _ { \mathrm { i n } }$ of the input-space Jacobian $J _ { x } = \partial \hat { c } / \partial x \in \mathbb { R } ^ { 6 4 \times 2 0 }$ (not the parameter-space Jacobian, for computational tractability). This bounds input sensitivity only. In particular, the trainable decoder bias supplies one parameter direction per output before numerical saturation; even a width-one bottleneck does not impose parameter-Jacobian rank one. We compute $J _ { x }$ via autograd for 30 test inputs and report the mean $r _ { \mathrm { i n } }$

Evaluation metrics. Normalized regret: $( c ^ { \top } z _ { \mathrm { p r e d } } - c ^ { \top } z ^ { * } ) / c ^ { \top } z ^ { * }$ , where $z _ { \mathrm { p r e d } }$ is the path obtained by running Dijkstra on the predicted costs cˆ. Accuracy: fraction of test instances where $z _ { \mathrm { p r e d } } = z ^ { * }$ (exact path match). Results are averaged over 5 random seeds.

Prediction error and regret. Wider bottlenecks can give lower prediction error but higher decision regret in these saved results. The observation does not by itself identify overfitting or a gradient mechanism: architecture, optimization and path sensitivity change together. It should not be treated as evidence that a parameter-space no-go theorem extends to these input-rank measurements.

Scaling to $1 2 \times 1 2$ grid. To test scale dependence, we replicate the bottleneck experiment on a 12×12 grid $( N { = } 1 4 4$ vertices, 8-connected edges). We generate 5K training and 1K test instances with the same feature dimension $( n _ { \mathrm { f e a t u r e s } } { = } 2 0 )$ and training protocol (60 epochs, 5-fold cross-validation). Table 25 and Figure 18 report results.

Table 25: Shortest path on $1 2 \times 1 2$ grid (N=144). The SPO+ advantage $\Delta$ increases monotonically with $r _ { \mathrm { i n } } ,$ reaching +16.4% for the full linear model—a 4× amplification over the 8×8 grid (Table 24). Results averaged over 5 folds.
<table><tr><td>Model</td><td>Params</td><td> $r _ { \mathrm { i n } }$ </td><td>MSE Reg.%</td><td> $\mathrm { S P O + R e g . } \%$ </td><td> $\Delta$ </td></tr><tr><td>k=1</td><td>309</td><td>1.0</td><td>77.0%</td><td>74.6%</td><td>+2.4%</td></tr><tr><td>k=2</td><td>474</td><td>1.9</td><td>89.3%</td><td>83.7%</td><td>+5.6%</td></tr><tr><td>k=5</td><td>969</td><td>4.6</td><td>106.6%</td><td>99.5%</td><td>+7.1%</td></tr><tr><td>k=10</td><td>1794</td><td>8.7</td><td>118.5%</td><td>109.6%</td><td>+8.9%</td></tr><tr><td>k=20</td><td>3444</td><td>15.2</td><td>129.1%</td><td>118.1%</td><td>+11.1%</td></tr><tr><td>Full linear</td><td>3024</td><td>20.0</td><td>126.0%</td><td>109.6%</td><td>+16.4%</td></tr></table>

Advantage amplifies with grid size  
![](images/09a6237b43825defdc58b1b0d6fc950792a8a03c822cd51f54b4f4c4ac3bad61.jpg)  
Figure 18: SPO+ advantage in percentage points at two grid sizes; positive values favor SPO+. Whiskers show ±1 SEM across five paired seeds. Filled markers join the controlled bottleneck settings; open diamonds denote full-capacity architectures. The larger grid has greater observed advantage at the sampled settings. A SEM bar crossing zero is not a hypothesis test; paired inferential results are reported separately in the corresponding table.

The larger grid has greater measured SPO+ advantage at the sampled settings, but grid size changes the problem as well as the cost-space dimension; the experiment does not isolate the mechanism. The number of paired seeds remains five. In the $1 2 \times 1 2$ experiment, one-sided Wilcoxon gives $p = 0 . 3 1 2$ at $k = 1$ and $p = 0 . 0 3 1$ at the tested $k \geq 1 0$ . With five pairs, 0.03125 is the smallest possible exact one-sided signed-rank p-value; these tests are unadjusted across settings.

## E.2 External Validation on PyEPO

We complement the custom tracking and path implementations with archived experiments using PyEPO benchmark generators and decision-loss implementations. This separates implementation provenance from the financial setting, while retaining the protocol limitations described below.

PyEPO (Tang and Khalil, 2024) supplies the benchmark generator and optimization layers. The reported $r _ { \mathrm { e f f } }$ is measured on a sampled stacked Jacobian: the script takes up to 32 examples and the first eight output coordinates per example. It is an entropy-rank estimate for that probe matrix, not a

full pointwise spectrum or the dh proxy. We use its shortest-path generator, its optimization model and its SPO+ implementation unmodified; the only thing we supply is the predictor.

A controlled parameterization sweep. We hold the task, the data, the optimizer and the training budget fixed and vary only the number of trainable parameters. The predictor is a fixed base map plus d fixed directions with trainable coefficients,

$$
\begin{array} { r } { \mathbf { c } ( \pmb { \theta } ) = W _ { 0 } \mathbf { x } + \sum _ { k = 1 } ^ { d } \theta _ { k } ( A _ { k } \mathbf { x } ) , } \end{array}\tag{17}
$$

so $\partial { \mathbf { c } } / \partial \theta _ { k } = A _ { k } { \mathbf { x } }$ and the Jacobian has exactly d columns. $W _ { 0 }$ is MSE-pretrained so that the restricted update starts from a useful predictor. This is a design choice for the comparison, not an implication of the one-parameter geometry.

Two variants we tried first do not control $r _ { \mathrm { e f f } }$ , and we record them because both look reasonable. Constraining $W = U V ^ { \top }$ to low matrix rank leaves $\partial \mathbf { c } / \partial ( U , V )$ high-rank—measured $r _ { \mathrm { e f f } }$ was 12.7 at matrix rank one. Keeping a trainable bias adds one free direction per output coordinate, putting $d { = } 1$ at $r _ { \mathrm { e f f } } { = } 5 . 9 5$ on a 40-edge grid. Only the form above makes measured $r _ { \mathrm { e f f } }$ track d.

Table 26: PyEPO shortest path, 5×5 grid. Only the predictor’s parameter count varies; task, data, solver and budget are fixed. Regret is PyEPO’s normalised decision regret (lower is better); gain is the SPO+ reduction over two-stage MSE training. $^ { * } p { < } 0 . 0 5$ , one-sided Wilcoxon across seeds.
<table><tr><td>d</td><td> $r _ { \mathrm { e f f } }$ </td><td>MSE regret</td><td>SPO+ regret</td><td>Gain ∆%</td><td>p</td><td>wins</td></tr><tr><td>1</td><td>1.00</td><td>0.0884</td><td>0.0887</td><td>-0.25</td><td>0.754</td><td>4/10</td></tr><tr><td>2</td><td>1.94</td><td>0.0887</td><td>0.0890</td><td>-0.28</td><td>0.722</td><td>5/10</td></tr><tr><td>4</td><td>3.73</td><td>0.0882</td><td>0.0874</td><td>+0.91</td><td>0.042*</td><td>6/10</td></tr><tr><td>8</td><td>7.02</td><td>0.0882</td><td>0.0876</td><td>+0.72</td><td>0.246</td><td>6/10</td></tr><tr><td>16</td><td>12.70</td><td>0.0887</td><td>0.0870</td><td>+1.92</td><td>0.065</td><td>7/10</td></tr><tr><td>32</td><td>20.39</td><td>0.0886</td><td>0.0845</td><td>+4.66</td><td>0.002**</td><td>9/10</td></tr><tr><td>full</td><td>44.78</td><td>0.0886</td><td>0.0771</td><td>+12.99</td><td> $0 . 0 0 1 ^ { * * }$ </td><td>10/10</td></tr></table>

What it shows, and what it does not. The gain rises with measured $r _ { \mathrm { e f f } }$ across a 45× range, from −0.25% at $r _ { \mathrm { e f f } } { = } 1 . 0 0 \ \mathrm { t o + } 1 2 . 9 9 \%$ at full rank, with Spearman $\rho { = } { + } 0 . 9 3 \ : ( p { = } 0 . 0 0 2 5 )$ . At the lowest measured rank, the estimated gain is slightly negative, on a benchmark built by other authors and an SPO+ implementation we did not write.

The $r _ { \mathrm { e f f } } \ge 4$ threshold does not transfer. At $r _ { \mathrm { e f f } } { = } 7 . 0 2$ the gain is $+ 0 . 7 2 \%$ , below the 1% the finance calibration would predict, and the two rows either side of it (+0.91% at $3 . 7 3 , + 1 . 9 2 \%$ at 12.70) win on only 6/10 and 7/10 seeds. What survives is the ordering, not the cut point: $r _ { \mathrm { e f f } }$ ranks configurations by how much DFL can buy, while the numeric thresholds are calibrated on equity data and should be re-calibrated per domain. We report this because an earlier run at five seeds did show both thresholds transferring, and the apparent agreement did not survive ten.

The positive control. The full-rank row recovers SPO+’s known advantage over two-stage training (+12.99%, p=0.001, 10/10 seeds) on the library’s own benchmark, through the same code path that produces the nulls at low $r _ { \mathrm { e f f } }$ . The positive full-capacity result shows that the implementation can improve this benchmark under at least one tested configuration; it does not validate every lower-capacity optimization setting.

(a)

![](images/2ff3cceb7fe7d03ecfe8540717e8eb0c28142b398684636164119e0690b4270c.jpg)

![](images/4cc30eb13c120806dc9455860063b30dad870f892b553722bc80f677ebc60e7c.jpg)  
Figure 19: Two empirical axes, with a shared vertical scale. (a) Thirty-eight one-parameter equity configurations across six markets and a 54× range of $K / N \colon$ gains remain below 1.8%, with no detected monotone association $( \rho = - 0 . 1 6 , p = 0 . 3 5 )$ . The shaded ±1% band is a reference, not an envelope containing all points. (b) PyEPO capacity sweep: relative reduction in mean regret, with ±1 paired delta-method SE across ten seeds. Dotted thresholds were calibrated on equities; one transfer failure is marked. The two panel concern different tasks and support association rather than a causal comparison.

Does rank collapse kill every DFL method, or only the one we tested? Proposition 4 is a statement about the Jacobian, not about a particular decision loss: at rank(J)=1 any loss reaching θ through $\mathbf { J } ^ { \top }$ produces a gradient collinear with the MSE gradient. This statement constrains local directions for each loss; it does not predict that their trained solutions or performance coincide. We tested that with four further PyEPO losses spanning unrelated mechanisms.

Method hyperparameters were selected using the full-capacity stage and then frozen for the reduced-capacity sweep (scripts/tune\_pyepo\_methods.py). This conditions the comparison on settings that work at full capacity; it does not give each reduced-capacity model its best validation-tuned configuration. The selection record does not establish a fully independent evaluation of the tuning procedure. We therefore report this as a transfer-of-hyperparameters experiment.

The black-box method did not improve the full-capacity baseline in the six tested configurations (best reported gain −4.84%). It remains a negative result, rather than being removed from the scope of the conclusion. Other runs exceeding 1.5× baseline regret are marked unstable. Such runs are relevant performance outcomes; rank-one collinearity does not rule them out or make them inadmissible counterevidence.

Table 27: PyEPO decision losses at low and full measured capacity, with hyperparameters frozen from the full-capacity stage. Values are relative regret reductions over two-stage training. “Unstable” denotes regret exceeding 1.5× baseline, a reported outcome rather than evidence of equivalence. Non-significant signedrank comparisons do not establish equality. $^ { * } p < 0 . 0 5$ , two-sided Wilcoxon over ten seeds.
<table><tr><td>Decision loss</td><td>Frozen hyper-parameters</td><td> $r _ { \mathrm { e f f } } { = } 1 . 0 0$ </td><td>full rank (44.8)</td></tr><tr><td>SPO+</td><td>lr=0.01</td><td>-0.25</td><td> $+ 1 2 . 9 9 ^ { * }$ </td></tr><tr><td>perturbedOpt</td><td> $\mathrm { l r } { = } 0 . 0 1 , n { = } 2 0 , \sigma { = } 0 . 5$ </td><td>-2.32</td><td>+4.98</td></tr><tr><td>negativeIdentity</td><td>lr=0.01</td><td>unstable</td><td>+2.24</td></tr><tr><td>Perturbed Fenchel-Young</td><td> $\operatorname { l r } [ 0 . 0 1 , n { = } 1 0 , \sigma { = } 0 . 5$ </td><td>-0.57</td><td> $+ 1 5 . 8 0 ^ { * }$ </td></tr></table>

At the lowest capacity, the finite results for SPO+, perturbed optimization and the perturbed Fenchel–Young loss are not significantly different from two-stage training under the reported tests. This does not establish equivalence. The negative-identity method is unstable at reduced capacity. Perturbed optimization has substantial negative gains at intermediate capacities, even though its fullcapacity result is positive. These failures show that rank or a fixed hyperparameter configuration

alone does not determine reliable optimization.

Multiple knapsack. The archived five-seed knapsack sweep does not show a detected monotone relationship between gain and measured capacity $( \rho ~ = ~ 0 . 3 7 , ~ p ~ = ~ 0 . 4 7 )$ . At full capacity the estimated gain is 1.85% $( p ~ = ~ 0 . 4 1$ , two of five wins), and several lower-capacity results favor MSE. Limited replication leaves substantial uncertainty, but the benchmark is still evidence of a setting where the proposed ordering did not appear. It should not be discarded solely because the full-capacity control failed to improve.

Relation to the main text. The equity sweep varies sparsity within one-parameter models, whereas this archived experiment varies an update subspace on a fixed generated task. The fullcapacity endpoint also changes initialization, and hyperparameters are transferred across capacities. These experiments are complementary observations rather than a joint identification of the causal effect of rank. The following extension aligns initialization and validation budgets.

## E.3 Validation-Tuned Controlled Extension

Purpose and relation to earlier experiments. This extension addresses two limitations of the archived PyEPO sweep: transferring hyperparameters from full capacity, and starting the fullcapacity comparator differently from restricted models. It uses a new protocol and smaller exactenumeration problems. It does not overwrite the earlier knapsack null result or establish that tuning caused the difference.

Tasks, splits and exact oracles. We use PyEPO’s synthetic feature–cost generators (Tang and Khalil, 2024) with five Gaussian features, degree four and multiplicative noise width 0.5. Ten seeds generate independent datasets per task, each split into 512 training, 256 validation and 512 test examples. The shortest-path task is a directed 4 × 4 grid with 24 edges and 20 feasible monotone paths. The two-dimensional knapsack has 12 items; each capacity is 35% of the corresponding total item weight. All feasible subsets are enumerated. Knapsack values are negated to express both tasks as minimization. Exact enumeration is checked against dynamic programming for signed shortestpath costs and SciPy’s MILP solver for knapsack instances. Costs are scaled using training data only.

Matched prediction families and selection. Let $\tilde { \boldsymbol { x } } = ( x ^ { \top } , 1 ) ^ { \top }$ . A ridge predictor $P _ { 0 }$ is fitted only on training data, using penalty $1 0 ^ { - 3 }$ . Every loss and capacity starts from $P _ { 0 }$ and uses

$$
\hat { c } ( \theta ; x ) = \left( P _ { 0 } + \sum _ { k = 1 } ^ { d } \theta _ { k } A _ { k } \right) \tilde { x } , \qquad \theta _ { 0 } = 0 .
$$

The vectorized directions $A _ { k }$ are a nested orthonormal basis, fixed within each dataset. We test $d = 1 , 2 , $ 8 and the complete basis $( d \ = \ 1 4 4$ for path, 72 for knapsack). Adam runs for 20 epochs with batch size 64. Each loss at each capacity independently selects among learning rates $0 . 0 0 3 , 0 . 0 1 , 0 . 0 3$ using validation decision regret at the final epoch. Initialization and minibatch order are paired. This gives 480 candidate fits and 160 selected models; test regret is evaluated for selected candidates only. The shared ridge baseline is evaluated separately. This initial protocol has one optimization seed per generated dataset. Appendix E.4 adds a crossed batch-order analysis on fresh datasets.

Table 28: Controlled extension with independent validation at each capacity. Spectral ranks and batch alignment are measured at selected MSE predictors; means are over ten datasets. Gain is relative test-regret reduction in percent. $p _ { H }$ is Holm-adjusted across eight comparisons. Full capacity differs by task.
<table><tr><td>Task</td><td> $d$ </td><td> $\mathrm { P o i n t } ~ r _ { \mathrm { e f f } }$ </td><td>Stack  $r _ { \mathrm { e f f } }$ </td><td>Batch | cos |</td><td>Gain</td><td>pH</td></tr><tr><td>Path</td><td>1</td><td>1.00</td><td>1.00</td><td>1.00</td><td>+0.24</td><td>1.0000</td></tr><tr><td>Path</td><td>2</td><td>1.93</td><td>2.00</td><td>0.73</td><td>+1.10</td><td>1.0000</td></tr><tr><td>Path</td><td>8</td><td>6.85</td><td>7.96</td><td>0.41</td><td>+1.89</td><td>1.0000</td></tr><tr><td>Path</td><td>144</td><td>24.00</td><td>130.27</td><td>0.22</td><td>+11.59</td><td>0.0684</td></tr><tr><td>Knapsack</td><td>1</td><td>1.00</td><td>1.00</td><td>1.00</td><td>-0.61</td><td>1.0000</td></tr><tr><td>Knapsack</td><td>2</td><td>1.88</td><td>1.99</td><td>0.83</td><td>-0.51</td><td>1.0000</td></tr><tr><td>Knapsack</td><td>8</td><td>5.82</td><td>7.90</td><td>0.55</td><td>+0.19</td><td>1.0000</td></tr><tr><td>Knapsack</td><td>72</td><td>12.00</td><td>65.03</td><td>0.42</td><td>+10.59</td><td>0.0156</td></tr></table>

Geometry, outcome and inference. We compute Jacobians for every output at the first 32 test examples after model selection. Spectral rank uses squared singular values, as in the main text; pointwise ranks are averaged, whereas stacked rank is computed from the full probe matrix. Gradients compare MSE with the SPO+ surrogate, not a derivative of discrete regret. The affine parameterization makes these Jacobians constant in θ, while gradient alignment still depends on the selected predictor and data. Test regret is mean excess objective divided by mean absolute optimal objective. Reported gain is the reduction in mean regret across the ten paired datasets. Bootstrap intervals resample dataset pairs 10,000 times; they are pointwise, not simultaneous. Two-sided Wilcoxon tests receive Holm correction over all eight comparisons.

What transfers, and what remains unresolved. The one-direction models have collinear batch gradients, because all examples share one parameter direction. Full models have pointwise spectral ranks 24 and 12, but mean stacked ranks approximately 130 and 65; treating these measurements as interchangeable would hide the batch distinction. Low-capacity mean gains are small, but non-significance does not prove equivalence. Full-capacity pointwise bootstrap intervals are [5.49, 16.81]% for shortest path and [7.30, 13.81]% for knapsack. The former’s adjusted $p = 0 . 0 6 8 4$ does not pass 0.05, despite its interval excluding zero, because the interval is unadjusted and the tests differ. All ten knapsack dataset pairs favor full-capacity SPO+; eight of ten do so on shortest path.

Changing capacity changes the attainable predictor family as well as its geometry. This intervention therefore does not identify rank as the sole cause of the gain. Neither task reproduces financial covariance estimation, detached asset selection, or transaction costs. The experiments test the broader predict–then–optimize question; the finance-specific support bound requires its own assumptions. Saved selected models and held-out arrays permit direct recomputation of decisions and gradients.

## E.4 Training Budget and Batch-Order Sensitivity

Fresh data and crossed randomness. We retain the previous study and add ten new datasets per task, using disjoint generator seeds. Task sizes, noise and the 512/256/512 split remain unchanged. Within each dataset, the ridge initializer and update basis are fixed while three independently seeded minibatch orders vary. We test scalar and full-capacity updates at 20 and 80 epochs. Each 80- epoch optimization trajectory supplies both checkpoints, so training-budget comparisons share their initial 20 epochs. The same three learning rates are compared independently at each budget and for each loss. This entails 720 optimization trajectories, 1,440 trained candidate checkpoints and 480 selected models.

Table 29: Fresh-data sensitivity study with three minibatch orders per dataset and a shared no-update candidate. Gain is relative test-regret reduction over validation-selected MSE. Intervals resample ten independent dataset pairs after averaging the three training seeds. $p _ { H }$ is Holm-adjusted across the eight follow-up comparisons. Full means 144 path or 72 knapsack update directions.
<table><tr><td>Task</td><td>Epochs</td><td>Capacity</td><td>Gain (%)</td><td>95% interval</td><td>pH</td></tr><tr><td>Path</td><td>20</td><td>1</td><td>+0.47</td><td>[+0.08, +1.00]</td><td>0.1250</td></tr><tr><td>Path</td><td>20</td><td>Full</td><td>+12.12</td><td>[+5.22, +19.68]</td><td>0.0820</td></tr><tr><td>Path</td><td>80</td><td>1</td><td>+0.52</td><td>[+0.02, +1.16]</td><td>0.2812</td></tr><tr><td>Path</td><td>80</td><td>Full</td><td>+12.54</td><td>[+5.22, +20.34]</td><td>0.0977</td></tr><tr><td>Knapsack</td><td>20</td><td>1</td><td>+0.11</td><td>[−0.31, +0.54]</td><td>0.7422</td></tr><tr><td>Knapsack</td><td>20</td><td>Full</td><td>+13.67</td><td>[+9.55, +17.83]</td><td>0.0156</td></tr><tr><td>Knapsack</td><td>80</td><td>1</td><td>+0.27</td><td>[−0.06, +0.67]</td><td>0.5938</td></tr><tr><td>Knapsack</td><td>80</td><td>Full</td><td>+13.76</td><td>[+9.42, +18.11]</td><td>0.0273</td></tr></table>

A common no-update option. Both MSE and SPO+ may additionally select the unchanged ridge predictor using validation regret. Ties favor no update, then the smaller learning rate. Thus the comparison does not force the prediction-trained baseline away from an already useful initializer. At full capacity, MSE selects no update in 11/30 path runs at each budget and 8/30, 6/30 knapsack runs at 20, 80 epochs. SPO+ selects no update in 1/30, 2/30 path runs and 3/30 knapsack runs at each budget. These are selection frequencies, not independent significance tests.

Independent units and uncertainty. For each task, capacity and budget, the three test regrets are first averaged within each dataset. The ten dataset-level pairs are then used for the relative reduction, paired bootstrap interval and two-sided Wilcoxon test. A separate Holm family covers the eight follow-up comparisons. This avoids treating optimization repetitions as additional independently generated tasks; bootstrap intervals remain pointwise. The study was designed after inspecting the first extension and is reported as a sensitivity analysis, not as a preregistered confirmation.

Interpretation. The full-capacity mean changes little between the two budgets on these datasets: path 12.12% to 12.54% and knapsack 13.67% to 13.76%. This is evidence of limited sensitivity over the tested budgets, not proof of convergence. Knapsack passes the adjusted 0.05 threshold at both budgets, while path does not. Scalar changes remain below 0.6%; their lack of adjusted significance is not an equivalence result. This follow-up reduces concern about one minibatch ordering or a forced update baseline, but retains the small synthetic task sizes, linear predictor family and restricted learning-rate grid. Full selected parameters, data arrays and validation records are saved for audit.

## F Demonstration Controls

## F.1 Invertible Coordinates at Fixed Expressivity

Question and protocol. Does the capacity contrast merely reflect lost expressivity? We use fresh datasets (seed offset 270927) from the same degree-four, noise-0.5 generators as Appendix E.3: ten datasets per task, 512 training, 256 validation and 512 test examples. A train-only ridge predictor $P _ { 0 }$ initializes every run. We retain the full affine function class and scale its output coordinates by $D _ { \varepsilon } = \operatorname { d i a g } ( 1 , \varepsilon , \dots , \varepsilon )$ for $\varepsilon \in \{ 1 , 0 . 1 , 0 . 0 1 \}$ }. Exact pointwise Jacobian rank stays at 24 for path and 12 for knapsack. Spectral rank is the exponential entropy of normalized squared singular values; the reported values are pointwise, not stacked ranks.

Table 30: Fixed-class coordinate control. Gain is mean test-regret reduction relative to matched MSE; positive favors SPO+. Exact rank is unchanged. All compensated rows within a task recover the same predictor outcomes.
<table><tr><td>Task</td><td>Update</td><td>ε</td><td>Spectral rank</td><td>Gain (%)</td><td>95% interval</td></tr><tr><td>Path</td><td>Ordinary</td><td>1</td><td>24.00</td><td>11.79</td><td>[7.51, 15.76]</td></tr><tr><td>Path</td><td>Ordinary</td><td>0.1</td><td>2.91</td><td>3.12</td><td>[0.85, 5.27]</td></tr><tr><td>Path</td><td>Ordinary</td><td>0.01</td><td>1.02</td><td>1.19</td><td>[−0.50, 2.75]</td></tr><tr><td>Path</td><td>Compensated</td><td>1</td><td>24.00</td><td>11.79</td><td>[7.51, 15.76]</td></tr><tr><td>Path</td><td>Compensated</td><td>0.1</td><td>2.91</td><td>11.79</td><td>[7.51, 15.76]</td></tr><tr><td>Path</td><td>Compensated</td><td>0.01</td><td>1.02</td><td>11.79</td><td>[7.51, 15.76]</td></tr><tr><td>Knapsack</td><td>Ordinary</td><td>1</td><td>12.00</td><td>10.17</td><td>[5.73, 14.44]</td></tr><tr><td>Knapsack</td><td>Ordinary</td><td>0.1</td><td>1.75</td><td>2.90</td><td>[1.30, 4.34]</td></tr><tr><td>Knapsack</td><td>Ordinary</td><td>0.01</td><td>1.01</td><td>0.52</td><td>[-0.28, 1.29]</td></tr><tr><td>Knapsack</td><td>Compensated</td><td>1</td><td>12.00</td><td>10.17</td><td>[5.73, 14.44]</td></tr><tr><td>Knapsack</td><td>Compensated</td><td>0.1</td><td>1.75</td><td>10.17</td><td>[5.73, 14.44]</td></tr><tr><td>Knapsack</td><td>Compensated</td><td>0.01</td><td>1.01</td><td>10.17</td><td>[5.73, 14.44]</td></tr></table>

Why compensation is an exact control. Write $G = \nabla _ { P } L$ . Ordinary SGD in Θ gives

$$
\nabla \Theta L = D _ { \varepsilon } G , \qquad \Delta P = - \eta D _ { \varepsilon } ^ { 2 } G .\tag{18}
$$

The compensated step $\Delta \Theta = - \eta D _ { \varepsilon } ^ { - 2 } \nabla _ { \Theta } L$ instead gives $\Delta P = - \eta G$ . With the same initializer, batch order and learning rate, induction therefore gives the same predictor trajectory as identitycoordinate SGD, for either loss. This equivalence uses invertibility and plain SGD; it is not a claim about arbitrary adaptive optimizers. The largest final parameter discrepancy in double precision was $6 . 7 \times 1 0 ^ { - 1 6 }$

Each loss and coordinate setting independently selects among learning rates {0.003, 0.01, 0.03} and the unchanged initializer by validation decision regret, with ties favoring no update. We train 40 epochs with batch size 64 and matched minibatch order. There are 720 candidate fits and 240 selected models. The audit recomputes validation and test regret from saved predictors and exact decision oracles. Intervals below bootstrap the ten paired datasets 10,000 times; they are pointwise and no new multiplicity-adjusted significance claim is made.

Scope. The control rules out a change of function class as the explanation within this experiment. It does not isolate spectral rank from conditioning: both vary with the same coordinate transformation, and the unscaled first coordinate is fixed rather than randomized across orientations. The finite learning-rate grid and budget do not establish convergence. Exact rank collapse, spectral concentration and expressivity must therefore remain distinct concepts.

## F.2 Chronological Forward-Target Financial Baselines

Question and data. We test whether fitting future covariance changes the conclusion drawn from a reconstruction baseline. This is a separate protocol using a frozen 100-stock universe and observed index returns from the cached 2005–2025 series (5,281 return observations). It is survivorshipbiased and does not reconstruct point-in-time membership. Ten annual test folds cover 2016–2025, each preceded by six validation months and 36 training months. Shared histories make the annual folds unsuitable for an independence-based significance claim here.

At each month’s first trading day, all estimators use strictly preceding returns. A common $K = 2 0$ support is fixed for each fold using squared stock–index correlations from the 252 returns preceding validation. The input $S _ { t }$ is the trailing 63-day joint covariance of stocks and the observed index. A long-only, unit-sum QP uses its selected stock block and stock–index cross-covariance.

Table 31: Future-target control over ten annual test folds. TE is mean annualized tracking error (lower is better). Covariance error is mean relative squared Frobenius error against future 21-day covariance, using only labels contained in the test year. Wins count years with lower TE than reconstruction. Changes are descriptive, not significance tests.
<table><tr><td>Estimator</td><td>TE (%)</td><td>TE change (%)</td><td>Cov. error</td><td>Wins / 10</td></tr><tr><td>Reconstruction</td><td>5.700</td><td>+0.00</td><td>1.636</td><td></td></tr><tr><td>Future MSE: 21 days</td><td>5.945</td><td>+4.31</td><td>1.026</td><td>6</td></tr><tr><td>Future MSE: 63 days</td><td>6.083</td><td>+6.73</td><td>0.983</td><td>6</td></tr><tr><td>Future MSE: selected</td><td>5.935</td><td>+4.13</td><td>1.027</td><td>6</td></tr><tr><td>Task validation</td><td>5.616</td><td>-1.46</td><td>1.453</td><td>7</td></tr><tr><td>Ledoit-Wolf</td><td>5.660</td><td>-0.69</td><td>1.261</td><td>6</td></tr><tr><td>EWMA: selected</td><td>5.708</td><td>+0.14</td><td>1.527</td><td>2</td></tr></table>

![](images/e11071cc8bb4bbbdabaee2ead227ed0a34c5496db0d0e68dfbd16663b123cead.jpg)

(b) Forecast fit and task error differ  
![](images/b29ecf2492bce23a847558f2a1f273f4fcf72391c40d537e54e7f4739941623d.jpg)  
Figure 20: Improved covariance fit need not improve tracking. Left: annual held-out TE reveals the uneven cost of future-target fitting across market years. Right: changes in two distinct metrics relative to reconstruction; negative is better for either metric. Their percentages have different denominators and are not a common utility scale.

The weights are held numerically constant across daily return evaluations until the next monthly estimate; this corresponds to daily rebalancing to target weights. Transaction costs are omitted in this control.

Predictive and task baselines. The common scalar family is $M _ { \alpha } = ( 1 - \alpha ) S + \alpha \mathrm { t r } ( S ) I / 1 0 1$ Reconstruction uses $\alpha = 0$ . Future-target variants fit a single $\alpha \in [ 0 , 1 ]$ by closed-form Frobenius MSE against future 21- or 63-day covariance at monthly training origins. Every forward label ends before validation begins. Future-selected MSE chooses its horizon by validation tracking error. Task validation chooses among 21 uniformly spaced α values by the same metric; this is grid selection, not end-to-end DFL. Additional comparators are Ledoit–Wolf and EWMA with decay selected from {0.94, 0.97, 0.99}. Support, QP constraints and evaluation dates are matched across methods.

Interpretation and remaining gap. The selected future-target estimator improves mean relative covariance error by 37.25%, yet worsens mean TE by 4.13%; it beats reconstruction in six years, illustrating why win counts alone miss the magnitude of failures. Task validation improves mean TE by 1.46%, and Ledoit–Wolf by 0.69%. These results do not show that reconstruction is the strongest predictive baseline or that DFL dominates forecasting. They demonstrate target sensitivity within one scalar family. The neural comparison in Appendix F.3 extends this control beyond a scalar family. Point-in-time stock membership, transaction costs and broader forecast architectures remain needed. The release saves all 70 portfolios, daily returns, support selections and chronology; its independent audit recomputes TE, forward-label boundaries, fitted shrinkage and validation

selections.

## F.3 Matched Neural Forward-Target Comparison

Design. To address the scalar family’s limited expressivity, we compare MSE and DFL using the same residual neural covariance predictor. This is a new matched experiment, not a rerun of the historical neural architecture. We reuse the chronology, frozen universe, training-only $K = 2 0$ support and daily target-weight evaluation of Appendix F.2. Both objectives use the same monthly training origins with 21-day forward labels ending before validation. Ten annual test folds cover 2016–2025. Each fold uses three paired initialization seeds, a 36-month training interval and six validation months.

Architecture and objectives. For each of the 20 selected stocks and the index, six trailing features describe 21-/63-day means divided by 63-day standard deviation, log 21-/63-day volatility, correlation with the index, and the five-day mean divided by 63-day standard deviation. Feature means and scales are fitted only on training origins. A shared 6 → 16 tanh layer and $1 6  5$ linear head contain 197 parameters. If its outputs are $( z _ { i } , f _ { i } )$ , the normalized covariance is

$$
\begin{array} { r } { \widehat { M } = D \widetilde { S } D + F F ^ { \top } + 1 0 ^ { - 4 } I , \quad D _ { i i } = \exp \left( \frac { 1 } { 2 } \operatorname { t a n h } z _ { i } \right) , \quad F _ { i } = 0 . 1 f _ { i } . } \end{array}\tag{19}
$$

Here $\widetilde { S }$ is trailing 63-day joint covariance divided by the mean training covariance trace per asset. eThe rank-four residual does not define the predictor’s Jacobian rank. Small head weights initialize the network close to trailing covariance. The diagonal scaling is bounded and the residual is positive semidefinite, so this is one restricted neural family rather than an unrestricted covariance forecaster.

MSE minimizes mean squared entries against centered future 21-day sample covariance. DFL minimizes mean squared tracking returns on those same 21 days, using the exact long-only, unitsum QP against the predicted joint stock–index covariance. DFL’s training objective is a second moment, while reported TE uses centered tracking-return dispersion. This control omits turnover penalties and transaction costs.

Matched optimization and selection. Both losses use full-batch Adam with gradient norm clipped at one, learning rates {0.001, 0.003}, and 40 epochs. Validation TE selects among checkpoints at 20 and 40 epochs and the unchanged initializer; both methods receive the same candidate budget. The 120 candidate trajectories yield 60 selected models. We determine the QP active set numerically and differentiate its equality-constrained solution; this derivative is local to a stable active set. Twelve double-precision directional finite-difference checks had maximum absolute discrepancy $2 . 1 \times 1 0 ^ { - 1 1 }$ . Independent saved-model checks reconstruct neural outputs in NumPy, verify KKT conditions and chronological boundaries, and recompute tracking returns and selected validation scores.

Results and limits. Mean TE is 5.711% for future-target MSE and 5.715% for DFL; the latter is 0.07% higher in relative terms. DFL improves six of ten years, but its mean advantage is absent. Validation selects no update for 17/30 MSE models and 12/30 DFL models. This finding is a descriptive negative result, not evidence of statistical equivalence or proof that neural DFL is ineffective. It addresses a matched forward-target comparison for one 197-parameter family. The narrow learning-rate grid, 40-epoch budget, small number of monthly training origins and initialization near historical covariance warrant further sensitivity analysis. The result also cannot attribute differences from historical financial gains to the target alone: architecture, asset support and evaluation protocol differ. Broader architectures, more training and point-in-time membership remain unresolved.

Table 32: Matched neural future-target comparison. TE is annualized and averaged over three initialization seeds within each year. Positive gain denotes a reduction from MSE to DFL. The last row compares the two means over years, rather than averaging yearly percentage gains. Annual folds share history; results are descriptive.
<table><tr><td>Test year</td><td>Future MSE TE (%)</td><td>DFL TE (%)</td><td>Gain (%)</td></tr><tr><td>2016</td><td>6.090</td><td>6.127</td><td>-0.60</td></tr><tr><td>2017</td><td>4.106</td><td>4.039</td><td>+1.64</td></tr><tr><td>2018</td><td>5.463</td><td>5.454</td><td>+0.17</td></tr><tr><td>2019</td><td>4.885</td><td>4.899</td><td>-0.28</td></tr><tr><td>2020</td><td>6.789</td><td>6.836</td><td>-0.69</td></tr><tr><td>2021</td><td>4.366</td><td>4.463</td><td>-2.21</td></tr><tr><td>2022</td><td>6.828</td><td>6.795</td><td>+0.48</td></tr><tr><td>2023</td><td>6.085</td><td>6.081</td><td>+0.08</td></tr><tr><td>2024</td><td>4.929</td><td>4.897</td><td>+0.64</td></tr><tr><td>2025</td><td>7.569</td><td>7.560</td><td>+0.12</td></tr><tr><td>Mean</td><td>5.711</td><td>5.715</td><td>-0.07</td></tr></table>

![](images/89d592cf515065f9247fe13ac9869c1e7c7db56ba46c3198a9e9d566df6c4949.jpg)  
Figure 21: A matched neural control without an aggregate DFL gain. (a) Bars compare seed-averaged TE within each year; dots show the three paired initialization results and are not additional independent datasets. Positive values favor DFL. (b) Validation-selected checkpoints across 30 models per loss. Frequent no-update selections limit conclusions about trained-model superiority.

## G Open Questions and Limitations

## G.1 Unresolved Empirical Findings

The following observations delimit the empirical interpretation. They distinguish failures of a proposed ordering from experiments that remain too limited to identify a mechanism. In particular, the new neural control cannot explain why its aggregate result differs from historical financial comparisons.

1. Two markets exceed 1% at d=1, and no measured quantity predicts which. Three of the 38 one-parameter configurations clear a 1% gain: Hang Seng at $K = 1 0 ( 1 . 7 6 \% )$ and $K = 5 ( 1 . 2 8 \% )$ and ASX 200 at $K = 5 ( 1 . 2 1 \% )$ . No simple ordering by heterogeneity, sparsity or universe size explains these cases. Euro Stoxx has the second-largest measured heterogeneity $( h = 0 . 6 3 6 )$ , versus 0.713 for Hang Seng and 0.505 for ASX), yet its three gains are 0.06%, 0.29% and 0.08%. ASX at $K = 5$ has a small support ratio $( K / N = 0 . 0 3 1 )$ ), while Euro Stoxx has the smallest universe $( N = 4 7 )$ . These comparisons do not identify which market characteristics cause the differences.

2. The upper threshold does not transfer across domains. On PyEPO shortest path, SPO+ at sampled stacked rank $r _ { \mathrm { e f f } } = 7 . 0 2$ gains 0.72%, below the equity rule’s 1% benefit threshold (Appendix E.2). The rank–gain ordering is strong $( \rho = 0 . 9 3 )$ , but it does not validate a shared cut point. The equity proxy and sampled stacked rank also measure different quantities. A five-seed result suggesting threshold transfer did not persist with ten seeds.

3. Architecture orders differently across equity universes. At N = 100, the largest displayed gain occurs for the structured model; at $N = 4 7 8$ , the conditional model has the largest relative reduction among these three classes. Its pointwise rank-one Jacobian does not resolve this difference, because input-dependent Jacobian directions can span a larger batch subspace. Changes in architecture, data, initialization and optimization also remain potential explanations.

4. perturbedOpt is significantly worse than two-stage through the middle of the capacity range. With hyper-parameters frozen from the full-capacity stage, it loses to MSE at every intermediate $r _ { \mathrm { e f f } } \mathrm { - } 9 . 2 \%$ at 1.94, −7.9% at 3.73, −20.8% at 7.02, −30.3% at $1 2 . 7 0 , - 1 3 . 2 \%$ at 20.39, all $p \leq 0 . 0 2 $ —then recovers to +4.98% at full rank. At the lowest rank, its estimated change is −2.32% $( p = 0 . 1 9 )$ ; this does not establish equivalence. Capacity-specific tuning is needed to separate optimization effects from geometry.

5. Negative-identity training is unstable at reduced capacity. The full-capacity configuration gives a 2.24% gain, while the reduced-capacity runs exceed 1.5× baseline regret. The instability could reflect the loss approximation, learning-rate transfer, or the restricted parameterization. It is retained as an experimental failure rather than excluded by the geometric argument.

6. The knapsack sweep does not support the proposed ordering. In the archived protocol, the full-capacity gain is imprecise $( 1 . 8 5 \% , p = 0 . 4 1$ , five seeds), and gain is not detectably monotone in measured capacity. More repetitions and separate tuning are needed to distinguish uncertainty from a task-specific failure of the hypothesis. The new validation-tuned extension in Appendix E.3 gives a different result on a smaller task; changes in problem size, initialization, tuning and data prevent attributing that difference to one cause.

7. Spatial replication is limited. NOAA and EPA provide one evaluation window per configuration, except EPA at N = 124, which provides two. Table 23 therefore supports only a descriptive contrast. The synthetic sweep has five folds at each of 18 settings, but cannot establish that heterogeneity explains the real-domain difference. Measured equity heterogeneity is reported separately in Appendix D.1.

8. The neural forward-target control does not reproduce the historical gain. The matched residual-network experiment gives mean TE of 5.711% for MSE and 5.715% for DFL, whereas the historical nine-fold neural comparison favors DFL. Architecture, support, targets and evaluation protocol differ, so this contrast cannot identify which change matters. Moreover, validation frequently retains the initializer. Longer training, a wider learning-rate search and alternative covariance architectures are needed before attributing the result to objective choice or predictor geometry. Shared annual histories and the absence of an equivalence test further limit inference (Appendix F.3).

Implication for use. For a new task, first establish a predictive baseline with labels aligned to the intended forecasting horizon. Compare objectives within the same architecture, initialization distribution and validation budget, allowing both to retain an unchanged baseline. Interpret geometry alongside effect sizes and held-out outcomes; rank one alone does not justify skipping task training. The small financial gains and negative neural control reported here motivate these comparisons, but do not prescribe a universal choice of objective.