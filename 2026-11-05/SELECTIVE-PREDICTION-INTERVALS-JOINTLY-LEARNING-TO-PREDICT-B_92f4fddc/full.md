# SELECTIVE PREDICTION INTERVALS: JOINTLY LEARNING TO PREDICT, BOUND, AND ABSTAIN

Lidor Rubi   
Faculty of Computer and Information Science   
Ben-Gurion University of the Negev   
Beer-Sheva, Israel   
rubil@post.bgu.ac.il   
Gilad Katz   
Faculty of Computer and Information Science   
Ben-Gurion University of the Negev   
Beer-Sheva, Israel   
giladkz@bgu.ac.il

## ABSTRACT

Many high-stakes regression decisions ask three things of a model at once: a point estimate to act on, an interval that bounds the risk of acting, and the option to abstain when neither can be trusted. Deep learning provides these in isolation: prediction-interval methods return an interval, and at best a value estimate inside it, for every input, while selective-prediction methods learn when to abstain, but only for point predictions and with no statement of how far the truth may lie from the estimate. We introduce SelPI, which trains interval, value and abstention heads jointly under a single loss with an explicit acceptance-rate budget. A natural objection is that interval width already tells us when to abstain. We show that it does not: across 18 tabular benchmarks, width predicts large point errors well (AUROC 0.72) but is anti-correlated with interval misses (AUROC 0.38), so abstaining on wide intervals discards precisely the inputs the interval got right. SelPI therefore selects with two gates, one trained on the interval loss and one on the value loss, coupled by a Fréchet penalty that ties their separate budgets to the joint acceptance rate they deliver. With the base predictor held fixed, the learned gate beats width thresholding on a proper interval score on all 18 tabular datasets, closing a median 87% of the gap to an oracle selector, and it also beats selection by deep-ensemble disagreement on every dataset. The advantage grows with the heteroscedasticity of the task, and we report the settings in which it disappears.

## 1 INTRODUCTION

Consider a pediatric radiologist using a model to estimate skeletal age from a hand radiograph. A bare point estimate is not enough: the clinical question is whether bone age is consistent with chronological age, which requires a plausible range. A range alone is not enough either, because the report must state a value. And for some radiographs—unusual poses, occlusions, rare conditions— the honest output is a referral for manual reading. The same triad appears wherever a regression model feeds a decision: these are components of a single decision, and a model that provides only some of them pushes the missing one onto the user.

Two lines of research supply these components separately. Prediction-interval (PI) methods learn a lower and an upper bound around the target (Khosravi et al., 2010; Pearce et al., 2018); IPIV (Simhayev et al., 2022) additionally learns where inside the interval the target is likely to be, yielding a point estimate coherent with its own bounds. These methods answer every query. Selective-prediction methods, exemplified by SelectiveNet (Geifman & El-Yaniv, 2019), learn a gate that abstains on a budgeted fraction of inputs so that the risk on the accepted remainder falls; they return only a point estimate. Neither family delivers the full triad: the accepted set of a selective point predictor carries no statement of risk, and the intervals of a non-selective predictor are only as trustworthy as the model's worst inputs.

The obvious way to close the gap is post hoc: train an interval model and abstain when the interval is wide. Understanding why this baseline falls short is the core of this paper. Width is an aleatoric signal, largely blind to epistemic failure—sparsely covered regions, distribution shift, intervals that are confidently wrong. And when the point estimate lies inside the interval by construction, point error is mechanically bounded by width on every input the interval captures (§3). The failures that matter are therefore the two that width cannot bound: the interval misses the target, or it captures the target but places the point estimate at the wrong end of a wide interval.

We propose SelPI (Selective Prediction Intervals), which trains IPIV's interval and value heads jointly with SelectiveNet-style gates under a single selective interval loss with an explicit acceptance budget. Selection is split across two $\mathrm { g a t e s - } g _ { \mathrm { P I } } .$ , trained on the selective interval loss and therefore driven by miscoverage, and $g _ { \mathrm { v a l } }$ , trained on the selective value loss and therefore driven by valueposition error—which must both admit an input for it to be answered. Each gate has its own budget, and a Fréchet penalty ties the two budgets to the joint acceptance rate they actually deliver.

Because no existing method produces the same output, the evaluation straddles two fields, and we organize it as three questions (§5, Table 1): does learning to abstain damage the interval model (the PI field's question); is the result a competitive selective regressor (the selective-prediction field's question); and, holding the predictor fixed, does a learned gate select better than the predictor's own uncertainty (the new question). The answer to the third is a clear yes; the answers to the first two are “mostly", and we report the exceptions. Our contributions are:

1. SelPI, the first method to jointly learn prediction intervals, an in-interval value estimate, and a budgeted abstention rule. We identify the structural coupling between an in-interval value estimate and interval width, and derive from it a dual-gate selector whose heads specialize to the two failure modes width cannot detect (§4).

2. An evaluation protocol for selective interval prediction: selective risk curves for a proper interval score, a same-predictor comparison that isolates the selector from the base model, and predictorindependent gate diagnostics (§5).

3. A characterization of when learned selection helps: the gain over width thresholding grows with residual heteroscedasticity and vanishes on near-deterministic problems, and each gate's rejections are enriched for the failure mode it was trained on (§7).

## 2 RELATED WORK

Selective prediction. Learning to abstain dates to Chow's error-reject trade-off (Chow, 1970). El-Yaniv & Wiener (2010) introduced the risk–coverage curve, and Cortes et al. (2016) gave the problem a learning-theoretic treatment. Thresholding the softmax response is a strong post-hoc selector for deep classifiers (Geifman & El-Yaniv, 2017). SelectiveNet (Geifman & El-Yaniv, 2019) instead trains the selector jointly with the predictor under an explicit coverage budget, with an auxiliary head that keeps the representation from collapsing onto the accepted subset. Later work recasts abstention as betting (Liu et al., 2019), uses a Gumbel-softmax gate (Salem et al., 2022), adds entropy regularizers (Feng et al., 2022), or calibrates the accepted set (Fisch et al., 2022). The excess-AURC of Geifman et al. (2018) isolates selector quality from base-model quality and underlies our main statistic. Selective regression is comparatively unexplored. Apart from a few regression experiments in SelectiveNet and its Gumbel-softmax variant, existing methods select post hoc, by thresholding an uncertainty score of a trained model. That score is the conditional variance, which is the optimal rule for squared loss (Zaoui et al., 2020), a blended variance with a risk guarantee (Jiang et al., 2020), model-agnostic bootstrap uncertainty (Pugnana et al., 2023), a conformal confidence score (Sokol et al., 2026), or the CRPS entropy of a distributional forecast (Zaoui & Dombry, 2025). Cheng et al. (2023) study rejection under an explicit cost and Shah et al. (2022) under fairness constraints. All of these select on a single scalar, and none learns an interval jointly with the selector; our widththreshold and ensemble-disagreement baselines are this family applied to an interval model.

Prediction intervals for deep regression. Distribution-free interval learning began with LUBE (Khosravi et al., 2010), whose coverage-width objective QD (Pearce et al., 2018) made differentiable. QD+ (Salem et al., 2020) added loss terms for a point prediction and a split-normal mixture for aggregation. IPIV (Simhayev et al., 2022) obtains the point prediction from a value head instead, yielding a coherent (point, interval) pair without extra loss terms. Quantile regression (Koenker & Bassett Jr, 1978; Tagasovska & Lopez-Paz, 2019; Chung et al., 2021) and heteroscedastic likelihoods (Nix & Weigend, 1994; Kendall & Gal, 2017) are alternative routes to intervals. Epistemic uncertainty is usually estimated with MC dropout (Gal & Ghahramani, 2016) or deep ensembles (Lakshminarayanan et al., 2017), and Hüllermeier & Waegeman (2021) review the aleatoric/epistemic distinction that organizes our analysis. Conformal prediction (Vovk et al., 2005; Romano et al., 2019; Angelopoulos & Bates, 2023) wraps any of these with a marginal finite-sample guarantee, and recent work calibrates prediction sets on a post-hoc selected subset (Xu et al., 2025; Kumar et al., 2025). All of these either answer every query or take the selector as given. We learn the selector jointly with the intervals (extended discussion in Appendix A).

## 3 BACKGROUND

Setup and terminology. We observe $( x , y ) \ \in \ \mathcal { X } \times \mathbb { R }$ drawn i.i.d. from an unknown $P . \mathrm { ~ \bf ~ A ~ }$ prediction interval at nominal miscoverage α is a pair $L , U ~ : ~ { \mathcal { X } } ~ \to ~ \mathbb { R }$ intended to satisfy $\begin{array} { r } { \mathrm { P r } [ L ( x ) \leq y \leq U ( x ) ] \geq 1 - \alpha ; } \end{array}$ its empirical quality is summarized by the coverage probability $\begin{array} { r } { \mathrm { \ddot { P I C P } } = \frac { 1 } { n } \sum _ { i } k _ { i } } \end{array}$ with $k _ { i } = 1 [ L ( x _ { i } ) \overset { \cdot } { \leq } y _ { i } \leq \bar { U } ( x _ { i } ) ]$ the capture indicator, and the mean width $\begin{array} { r } { \mathrm { M P I W } = \frac { 1 } { n } \sum _ { i } ( U ( x _ { i } ) - L ( x _ { i } ) ) } \end{array}$ . A selective predictor is a pair $( f , g )$ with gate $g : \mathcal { X }  [ 0 , 1 ]$ that answers $f ( x )$ when $g ( x ) \geq \tau$ and abstains otherwise. For a loss l, its selective risk (El-Yaniv & Wiener, 2010) is the average loss over the accepted subset,

$$
R _ { \ell } ( f , g ; \tau ) ~ = ~ \frac { \sum _ { i } \ell ( f ( x _ { i } ) , y _ { i } ) \mathbb { 1 } [ g ( x _ { i } ) \geq \tau ] } { \sum _ { i } \mathbb { 1 } [ g ( x _ { i } ) \geq \tau ] } .\tag{1}
$$

“Coverage" means accepted inputs in the selective literature and captured targets in the PI literature, so we never use the word unqualified: the former is the acceptance rate φ, the latter PICP. $\tau _ { \phi }$ is the threshold that admits a fraction $\phi$ of inputs, and $R _ { \ell } ( \phi )$ abbreviates $R _ { \ell } ( f , g ; \tau _ { \phi } )$

SelectiveNet. SelectiveNet (Geifman & El-Yaniv, 2019) attaches a predictor $f ,$ a soft gate g and an auxiliary predictor h to a shared backbone, and minimizes a gate-weighted risk under a soft acceptance constraint,

$$
{ \mathcal { L } } _ { ( f , g ) } = { \frac { { \frac { 1 } { n } } \sum _ { i } \ell ( f ( x _ { i } ) , y _ { i } ) g ( x _ { i } ) } { { \hat { \phi } } ( g ) } } + \lambda \Psi { \big ( } \phi _ { \mathrm { t } } - { \hat { \phi } } ( g ) { \big ) } , \qquad { \hat { \phi } } ( g ) = { \frac { 1 } { n } } \sum _ { i } g ( x _ { i } ) ,\tag{2}
$$

with $\phi _ { \mathrm { t } }$ the target acceptance rate and $\Psi ( a ) = \operatorname* { m a x } ( 0 , a ) ^ { 2 }$ . The auxiliary head is trained on all inputs with the plain loss $\mathcal { L } _ { h }$ and discarded at test time; the objective $\gamma \mathcal { L } _ { ( f , g ) } + ( 1 - \gamma ) \mathcal { L } _ { h }$ keeps the backbone from overfitting the accepted subset. One model is trained per φt.

IPIV. IPIV (Simhayev et al., 2022) outputs bounds $U ( x ) , L ( x )$ and a mixing coefficient $v ( x ) \in$ (0, 1), which we call the value head,1 and predicts

$$
\hat { y } ( x ) = v ( x ) U ( x ) + \left( 1 - v ( x ) \right) L ( x ) .\tag{3}
$$

The bounds are trained with the quality-driven objective of QD (Pearce et al., 2018): minimize the width of the intervals that capture their target, $\mathrm { M P \bar { I } W _ { c a p t } } = \bar { \sum _ { i } { k _ { i } ( U _ { i } - L _ { i } ) } } / { \sum _ { i } { k _ { i } } }$ , under a penalty that activates when coverage falls below nominal,

$$
{ \mathcal { L } } _ { \mathrm { P I } } = \mathrm { M P I W } _ { \mathrm { c a p t } } + { \sqrt { n } } \lambda \Psi { \big ( } ( 1 - \alpha ) - { \widetilde { \mathrm { P I C P } } } { \big ) } ,\tag{4}
$$

where $\widetilde { \mathrm { P I C P } }$ replaces the hard capture indicator with the sigmoid relaxation ${ \tilde { k } } _ { i } ~ = ~ \sigma \big ( s ( U _ { i } ~ -$ $y _ { i } ) \big ) \sigma \big ( s ( y _ { i } - L _ { i } ) \big )$ , and n is the batch size (Appendix B). The value head is trained with a regression loss on equation $3 , \mathcal { L } _ { v }$ (squared error here), on all inputs—captured or not—and the total loss is $\beta \mathcal { L } _ { \mathrm { P I } } + \left( 1 - \beta \right) \mathcal { L } _ { v }$ . IPIV motivates this design in the same way SelectiveNet motivates its auxiliary head: a loss that only sees the captured subset overfits to it.

A structural coupling. Equation 3 implies $\hat { y } ( x ) ~ \in ~ [ L ( x ) , U ( x ) ]$ , so on any captured input $| \hat { y } _ { i } - y _ { i } | \leq U _ { i } - L _ { i } { : }$ point error is bounded by interval width wherever the interval is correct. Two consequences shape the rest of the paper. First, abstaining on wide intervals reduces selective point error automatically, so a width threshold is an unusually strong baseline on point-error curves. Second, a learned gate can genuinely separate from width only on the events width does not bound: a coverage failure $\mathsf { \bar { ( } } y _ { i } \notin [ \bar { L _ { i } } , U _ { i } ] )$ and a value failure $( y _ { i }$ captured, but $v _ { i }$ far from the target's true relative position $v _ { i } ^ { \star } = \dot { ( } y _ { i } - L _ { i } \dot { ) } / ( U _ { i } - L _ { i } ) )$ . Section 4 builds the selector around exactly these two events.

![](images/e2b48f10d33a33c060b1f542cbc66860e35216b539abd897a591a3bfcc2b8dea.jpg)  
Figure 1: SelPI in one picture. Left: one backbone feeds the interval and value heads, two selection gates, and auxiliary heads that are trained on all inputs and dropped at inference; a prediction is released only if both gates accept. Right: the cases a width threshold confuses (bar: predicted interval; dot: point estimate $\hat { y } ;$ star: true y).

## 4 SELPI: SELECTIVE PREDICTION INTERVALS

## 4.1 ARCHITECTURE

SelPI attaches three groups of outputs to a shared backbone (Figure 1): the IPIV triple $\big ( L ( x ) , U ( x ) , v ( x ) \big )$ with ${ \hat { y } } ( x )$ as in equation 3; two gates $g _ { \mathrm { P I } } ( x ) , g _ { \mathrm { v a l } } ( \bar { x } ) \in [ 0 , 1 ]$ ; and an auxiliary triple $\big ( L _ { h } ( x ) , U _ { h } ( x ) , v _ { h } ( x ) \big )$ that is trained on all inputs and discarded at inference. The backbone and the $( L , U , v )$ heads are IPIV's for the task, the auxiliary triple duplicates the heads, and each gate is SelectiveNet's selection head (Appendix D). The auxiliary triple plays the role of SelectiveNet's auxiliary head (§3): it is trained on every input, so the shared backbone keeps learning from the inputs the gates reject rather than overfitting the accepted subset. Its loss is given in §4.2 and its contribution is measured in §7. SelPI therefore adds only the gates and the auxiliary triple to an IPIV model, which is what makes the full-acceptance comparison in $\ S 6$ a fair test of what the additions cost.

## 4.2 SELECTIVE INTERVAL AND VALUE LOSSES

We instantiate SelectiveNet's gate-weighted risk equation 2 not with a point loss but with IPIV's two objectives, so that both the interval and the value estimate are optimized on the inputs the model will actually answer. For a gate $g$ with $g _ { i } = g ( x _ { i } )$ , define the gate-weighted capture rate and captured width,

$$
\mathrm { P I C P } _ { g } = \frac { \sum _ { i } g _ { i } \tilde { k } _ { i } } { S _ { g } } , \qquad \mathrm { M P I W } _ { g } = \frac { \sum _ { i } g _ { i } k _ { i } \left( U _ { i } - L _ { i } \right) } { \sum _ { i } g _ { i } k _ { i } } , \qquad S _ { g } = \sum _ { i } g _ { i } , \qquad \hat { \phi } ( g ) = S _ { g } / n ,\tag{5}
$$

where $S _ { g }$ is the effective number of selected samples. The selective interval loss carries both constraints—interval coverage and acceptance rate—and the selective value loss is the gate-weighted squared error:

$$
\mathcal { L } _ { \mathrm { P I } } ^ { \mathrm { s e l } } ( g ) = \mathrm { M P I W } _ { g } + \sqrt { S _ { g } } \lambda _ { \mathrm { P I } } \Psi \bigl ( ( 1 - \alpha ) - \mathrm { P I C P } _ { g } \bigr ) + \lambda _ { \mathrm { s e l } } \Psi \bigl ( \phi _ { \mathrm { t } } - \hat { \phi } ( g ) \bigr ) ,\tag{6}
$$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { v a l } } ^ { \mathrm { s e l } } ( g ) = \sum _ { i } g _ { i } \left( \hat { y } _ { i } - y _ { i } \right) ^ { 2 } \Big / \sum _ { i } g _ { i } . } \end{array}\tag{7}
$$

Relative to equation $^ { 4 , }$ every sample is weighted by the gate, and the coverage penalty is scaled by the selected count $\sqrt { S _ { g } }$ rather than the batch size, because that is the amount of data backing the $\mathrm { P I C P } _ { g }$ estimate (Appendix B). The auxiliary triple is trained with the plain IPIV loss on all samples, $\mathcal { L } _ { \mathrm { a u x } } = \beta _ { h } \mathcal { L } _ { \mathrm { P I } } ^ { ( h ) } + \left( 1 - \beta _ { h } \right) \mathcal { L } _ { v } ^ { ( h ) }$ ; a full triple, rather than one auxiliary regressor, keeps interval-relevant as well as point-relevant features alive for the inputs the gates reject.

## 4.3 TWO FAILURE MODES, TWO GATES

If a single gate weights both losses, it receives gradient from two sources that reward different behavior: the coverage penalty pushes $g$ down where the interval is likely to miss, the value loss where the point estimate is likely to be far from the target. By the coupling of §3 the second signal is bounded by width on captured inputs and the first is not, so one scalar must encode two different predictors, with the balance set implicitly by $\lambda _ { \mathrm { P I } }$ rather than by the decision-maker. SelPI therefore uses two heads, each weighting the loss whose failures it predicts and each with its own budget $\phi _ { \mathrm { t } } ^ { \mathrm { P I } } , \phi _ { \mathrm { t } } ^ { \mathrm { v a l } }$ . An input is accepted only if both gates admit it,

Table 1: Evaluation design. SelPI has no like-for-like competitor, so each question is asked on the home ground of the field it comes from, with that field's comparator and metric.
<table><tr><td>Question (origin)</td><td></td><td>Comparator Metric</td></tr><tr><td></td><td>the interval model? (PI field)</td><td>Q1 Does learning to abstain damage IPIV, all inputs answered, PICP, MPIW, RMSE at IPIV&#x27;s protocol and datasets  $\phi = 1$ </td></tr><tr><td></td><td>Q2 Is it a competitive selective regres- SelectiveNet-regression, sor? (selective-prediction field) </td><td>its selective RMSE / MAE vs. own grid and settings  $\phi$ </td></tr><tr><td></td><td>learned gate beat the model&#x27;s own / by width  $U - L ~ /$  uncertainty? (new) cle; a deep ensemble ranked by ciency</td><td>Q3 Given an interval model, does a same predictor, ranked by gate selective interval score and by an ora- RMSE vs. φ; selection effi-</td></tr></table>

$$
\mathrm { a c c e p t } x \iff g _ { \mathrm { P I } } ( x ) \geq \tau _ { \mathrm { P I } } \land g _ { \mathrm { v a l } } ( x ) \geq \tau _ { \mathrm { v a l } } ,\tag{8}
$$

with the two thresholds calibrated independently on validation data, each to its own budget. Figure 1 (right) shows the three cases this separates: a wide interval that is correct, which both gates accept and a width rule rejects; a narrow interval that misses, which only $g _ { \mathrm { P I } }$ rejects; and a captured target with the value estimate at the wrong end, which $g _ { \mathrm { v a l } }$ rejects.

Two gates with two budgets leave the quantity that actually matters, the joint acceptance rate, uncontrolled. Since $( 1 - g _ { \mathrm { P I } } ) ( 1 - g _ { \mathrm { v a l } } ) \ge 0$ we have $g _ { \mathrm { P I } } g _ { \mathrm { v a l } } \geq g _ { \mathrm { P I } } + g _ { \mathrm { v a l } } - 1$ pointwise, so the joint rate obeys the Fréchet bound

$$
\begin{array} { r } { \hat { \phi } _ { \mathrm { j o i n t } } ~ = ~ \frac { 1 } { n } \sum _ { i } g _ { \mathrm { P I } } ( x _ { i } ) g _ { \mathrm { v a l } } ( x _ { i } ) ~ \ge ~ \phi _ { \mathrm { t } } ^ { \mathrm { P I } } + \phi _ { \mathrm { t } } ^ { \mathrm { v a l } } - 1 . } \end{array}\tag{9}
$$

We make equation 9 a trained constraint: a quadratic penalty fires whenever the gates drift so far apart that their joint mass falls below the floor their budgets imply,

$$
\mathcal { L } _ { \mathrm { f r } } = \lambda _ { \mathrm { f r } } \Psi \big ( \phi _ { \mathrm { t } } ^ { \mathrm { P I } } + \phi _ { \mathrm { t } } ^ { \mathrm { v a l } } - 1 - \hat { \phi } _ { \mathrm { j o i n t } } \big ) .\tag{10}
$$

Without it the two heads are free to reject disjoint sets, and the intersection collapses well below either budget. With $\mathcal { L } _ { g _ { \mathrm { P I } } } = \beta _ { \mathrm { P I } } \mathcal { L } _ { \mathrm { P I } } ^ { \mathrm { s e l } } ( g _ { \mathrm { P I } } ) + ( 1 - \beta _ { \mathrm { P I } } ) \mathcal { L } _ { \mathrm { f r } }$ , the full objective mixes the two selective losses as IPIV mixes its losses, and the result with the auxiliary loss as SelectiveNet does:

$$
\mathcal { L } _ { \mathrm { 2 G } } = \gamma \Big [ \beta \mathcal { L } _ { g _ { \mathrm { P I } } } + ( 1 - \beta ) \mathcal { L } _ { \mathrm { v a l } } ^ { \mathrm { s e l } } ( g _ { \mathrm { v a l } } ) \Big ] + ( 1 - \gamma ) \mathcal { L } _ { \mathrm { a u x } } .\tag{11}
$$

We fix $\gamma = \beta = \beta _ { h } = 0 . 5$ , the parents' defaults; $\lambda _ { \mathrm { f r } }$ and $\beta _ { \mathrm { P I } }$ , which have no counterpart in either parent, are tuned per dataset. Tying the gates $( g _ { \mathrm { P I } } \equiv g _ { \mathrm { v a l } } \equiv g , \lambda _ { \mathrm { f r } } = 0 )$ gives the single-gate variant $\mathrm { S e l P I _ { 1 G } }$ , which we keep as an ablation; the full model is $\mathrm { S e l P I _ { 2 G } }$ . The budget pair $( \mathrm { e . g . \ 0 . 8 5 }$ each guarantees a joint rate of at least 0.70) also exposes a frontier that trades selective interval quality against selective point accuracy at a fixed joint rate (Appendix I), which is why we do not tie the two budgets to one joint target by formula.

## 4.4 TRAINING AND INFERENCE

All heads are trained end to end with Adam; depth, epochs, λPI and initialization are inherited unchanged from IPIV's published per-dataset configuration, except the learning rate, which we cap at $5 \times \mathrm { \overline { { 1 0 } } ^ { - 3 } }$ because IPIV's larger values destabilize the multi-term objective. All penalties are active from the first epoch: a warm start that delays the coverage penalty lets the two bounds cross and never recover (Appendix B). One model is trained per target acceptance rate; at inference the thresholds are set on a validation split to accept the budgeted fraction of inputs, and every baseline's threshold is calibrated on the same split. The model returns $( \hat { y } ( x ) , [ L ( x ) , \hat { U } ( x ) ] )$ on accepted inputs, from a single forward pass, and abstains otherwise.

## 5 EVALUATION DESIGN

Rationale. SelPI produces an output—interval, value and abstention decision—that no existing method produces, so no single comparison is like-for-like. PI methods answer every query and are scored on the (PICP, MPIW) pair; selective methods abstain and are scored on risk-coverage curves of a point loss. Neither convention transfers on its own. Under selection the (PICP, MPIW) pair is no longer a fair summary, because a selector can buy narrow accepted intervals by keeping the misses, or perfect coverage by keeping only easy inputs. A point-loss risk-coverage curve ignores the interval altogether and mechanically favors any width-based selector (§3). We therefore ask three questions, each on the home ground of the field it comes from (Table 1). Q1 and Q2 are the checks each parent field would make first, under that field's protocol. Q3 is the contribution, and it needs two instruments neither field supplies: a same-predictor comparison, which ranks the test points of one trained model by the gate, by interval width and by an oracle, so that differences are attributable to the selector and not to the base model; and a proper interval score, which charges for a miss rather than only for width and yields one number per (dataset, φ) cell.

Datasets. We use 20 regression tasks (Appendix C). (A) The UCI PI suite, the ten datasets of QD and IPIV under the protocol of Hernández-Lobato & Adams (2015), and (B) IPIV's two image tasks, RSNA bone age (Halabi et al., 2019) and IMDB-WIKI apparent age (Rothe et al., 2018), are the PI field's home ground (Q1). Several UCI sets are too small to evaluate selection (at $\phi = 0 . 3$ Yacht accepts 9 test points per split), so we add (C) eight larger OpenML regression tasks (Fischer et al., 2023; Grinsztajn et al., 2022) with 20k–582k samples, for statistical power. Bone age has a single 200-image test split on which validation-calibrated thresholds transfer poorly, so we treat it as illustrative; IMDB-WIKI (5-fold CV, ～172k images) is the reliable image benchmark.

Baselines. All baselines share splits and seeds, and every threshold is calibrated on validation data.

• IPIV + width threshold, the post-hoc rule the central claim is stated against. We evaluate it in two forms: same-predictor (ranking SelPI's own intervals by width; used for Q3 and for selection efficiency) and cross-pipeline (a separately trained IPIV ranked by its own width; Appendix G).

• Deep-ensemble disagreement, the strongest post-hoc rival and the textbook epistemic signal: M=5 IPIV models per split, combined with the aggregation IPIV's own paper reports (adapted from Lakshminarayanan et al., 2017) and ranked by the variance of the members' point predictions (point disagreement) or the summed spread of their bounds (bound disagreement), each against the ensemble's own width threshold and oracle. The comparison is cross-pipeline: one SelPI model against a five-model ensemble, at a fifth of the training and inference cost.

• SelectiveNet-regression, learned selection on a point regressor; it produces no interval and appears only in point-error comparisons (Q2).

• Reference points: IPIV at full acceptance, and an oracle that ranks by realized loss and upperbounds any selector on the same predictor. The non-selective methods of the PI literature (QD, QD+, deep ensembles) are compared at full acceptance in Appendix N.

Two asymmetries should be stated: $\mathrm { S e l P I _ { 2 G } } ?$ s budgets are tuned by search whereas SelectiveNet runs at its published settings, so we report the median search trial, not the best; and SelectiveNet uses its own backbone (Appendix D).

Protocol and metrics. One SelPI model is trained per target acceptance rate on the grid $\phi \in \mathbf { \Xi }$ $\{ 0 . 3 0 , 0 . 4 0 , \ldots , 0 . 7 0 , 0 . 7 5 , \ldots , 0 . 9 5 \}$ , which contains SelectiveNet's regression grid, with $1 - \alpha =$ 0.95 throughout; the post-hoc selectors are swept. Tabular results average 5 random 90/10 splits (the provider's split for Year-MSD). Every metric is a selective risk equation 1 as a function of $\phi .$ The headline is the selective interval score, the accepted-set average of the Winkler score (Winkler, 1972),

$$
\begin{array} { r } { \mathrm { I S } _ { \alpha } ( L , U ; y ) = ( U - L ) + \frac { 2 } { \alpha } ( L - y ) \Im [ y < L ] + \frac { 2 } { \alpha } ( y - U ) \Im [ y > U ] , } \end{array}\tag{12}
$$

a strictly proper scoring rule for interval forecasts (Gneiting & Raftery, 2007). We also report its components (selective PICP, MPIW), selective RMSE/MAE, and the value-position error $| v _ { i } - v _ { i } ^ { \star } |$ on captured inputs, which $g _ { \mathrm { v a l } }$ should select on. Curves are summarized by AURC, the normalized area under the selective-risk curve over $\phi \in [ 0 . 3 , 1 ]$ , and by selection efficiency $\mathrm { S E } = ( \mathrm { A U R C _ { w i d t h } - A U R C _ { g a t e } } ) / ( \mathrm { A U R C _ { w i d t h } - A U R C _ { o r a c l e } } )$ , the fraction of the headroom between the same-predictor width threshold and the oracle that the gate captures; SE is the excess-AURC of Geifman et al. (2018) made unit-free so that it can be aggregated across datasets. Gate diagnostics (AUROC for each failure mode, Spearman $\rho$ with width) do not depend on the base predictor. The unit of statistical analysis is the per-dataset AURC, compared with a Wilcoxon signed-rank test (Demšar, 2006) and always accompanied by win/loss counts (Appendices E and H).

Table 2: Per-dataset results, 18 tabular datasets, $1 { - } \alpha = 0 . 9 5 , \mathrm { S e l P I _ { 2 G } }$ median search trial throughout. Left (Q2): RMSE of IPIV answering every query; RMSE reduction of $\mathrm { S e l P I _ { 2 G } }$ at two realized acceptance levels; SelectiveNet-regression vs. $\mathrm { S e l P I _ { 2 G } }$ at $\phi = 0 . 7 \left( \Delta > 0 \right.$ favors SelPI). Right (Q3): learned gate vs. width threshold on the same predictor at the realized acceptance nearest 0.7; lower $\mathrm { I S } _ { \alpha }$ is better.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">IPIV φ=1</td><td colspan="2">SelPI2G RMSE cut vs. IPIV</td><td colspan="3">sel. RMSE at φ = 0.7</td><td colspan="2"> $\mathrm { S e l P I _ { 2 G } } ~ \mathrm { s e l . } ~ \mathrm { P I C P } ~ \mathrm { a t } ~ \phi { \approx } 0 . 7$ </td><td colspan="2"> $\mathrm { S e l P I _ { 2 G } \ s e l . \ I S _ { \alpha } \ a t } \ \phi { \approx } 0 . 7$ </td></tr><tr><td>φ≈0.55</td><td>φ≈0.75</td><td>Sel.Net</td><td>SelPI2G</td><td>∆</td><td>gate</td><td>width</td><td>gate</td><td>width</td></tr><tr><td>yacht</td><td>1.04</td><td>63.3%</td><td>56.1%</td><td>1.62</td><td>0.35</td><td>+78.3%</td><td>0.975</td><td>0.893</td><td>0.10</td><td>0.45</td></tr><tr><td>boston</td><td>2.64</td><td>10.5%</td><td>7.5%</td><td>2.42</td><td>2.39</td><td>+1.3%</td><td>0.934</td><td>0.869</td><td>1.72</td><td>2.16</td></tr><tr><td>energy</td><td>1.63</td><td>59.4%</td><td>50.5%</td><td>1.17</td><td>0.82</td><td>+30.2%</td><td>0.945</td><td>0.855</td><td>0.45</td><td>1.41</td></tr><tr><td>concrete</td><td>6.17</td><td>33.5%</td><td>22.2%</td><td>4.66</td><td>4.59</td><td>+1.5%</td><td>0.850</td><td>0.809</td><td>1.66</td><td>2.03</td></tr><tr><td>wine</td><td>0.66</td><td>16.2%</td><td>10.1%</td><td>0.56</td><td>0.56</td><td>0.0%</td><td>0.812</td><td>0.701</td><td>5.39</td><td>9.29</td></tr><tr><td>kin8nm</td><td>0.08</td><td>11.0%</td><td>1.5%</td><td>0.07</td><td>0.08</td><td>-15.5%</td><td>0.880</td><td>0.786</td><td>1.68</td><td>3.78</td></tr><tr><td>power-plant</td><td>4.10</td><td>7.5%</td><td>3.2%</td><td>3.86</td><td>3.92</td><td>-1.5%</td><td>0.937</td><td>0.924</td><td>1.14</td><td>1.20</td></tr><tr><td>naval</td><td>0.0010</td><td>15.9%</td><td>3.0%</td><td>0.0006</td><td>0.0011</td><td>-83.6%</td><td>0.956</td><td>0.812</td><td>0.30</td><td>2.47</td></tr><tr><td>protein</td><td>4.41</td><td>28.3%</td><td>14.4%</td><td>3.93</td><td>3.56</td><td>+9.4%</td><td>0.947</td><td>0.861</td><td>2.87</td><td>5.19</td></tr><tr><td>Year-MSD</td><td>9.01</td><td>43.1%</td><td>28.9%</td><td>6.13</td><td>5.96</td><td>+2.8%</td><td>0.932</td><td>0.916</td><td>3.35</td><td>4.38</td></tr><tr><td>superconduct</td><td>13.05</td><td>56.9%</td><td>39.5%</td><td>8.63</td><td>7.40</td><td>+14.3%</td><td>0.941</td><td>0.887</td><td>1.11</td><td>2.22</td></tr><tr><td>sarcos</td><td>3.23</td><td>51.5%</td><td>35.3%</td><td>2.40</td><td>1.91</td><td>+20.3%</td><td>0.955</td><td>0.929</td><td>0.46</td><td>0.60</td></tr><tr><td>diamonds</td><td>563.4</td><td>73.3%</td><td>56.6%</td><td>237.6</td><td>232.5</td><td>+2.2%</td><td>0.973</td><td>0.890</td><td>0.26</td><td>2.28</td></tr><tr><td>video_transcoding</td><td>2.83</td><td>87.5%</td><td>80.1%</td><td>0.79</td><td>0.50</td><td>+37.4%</td><td>0.982</td><td>0.974</td><td>0.16</td><td>0.23</td></tr><tr><td>medical_charges</td><td>0.08</td><td>30.9%</td><td>19.6%</td><td>0.07</td><td>0.06</td><td>+10.2%</td><td>0.961</td><td>0.955</td><td>0.60</td><td>0.66</td></tr><tr><td>black_friday</td><td>0.39 0.39</td><td>8.1% 29.7%</td><td>4.9%</td><td>0.38 0.32</td><td>0.37 0.30</td><td>+3.5% +4.1%</td><td>0.950 0.943</td><td>0.856 0.935</td><td>3.55 2.32</td><td>9.25</td></tr><tr><td>particulate-matter</td><td>0.41</td><td>31.7%</td><td>19.1% 21.8%</td><td>0.34</td><td>0.31</td><td>+9.6%</td><td>0.958</td><td>0.887</td><td>2.98</td><td>2.51 7.20</td></tr><tr><td>nyc-taxi-green</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>median [IQR] wins / ties / losses</td><td></td><td>31% [16–56] 18/0/0</td><td>21% [8–38] 18/0/0</td><td></td><td colspan="2">+3.8% [+1.4, +13.3]</td><td>0.946</td><td>0.887 18/0/0</td><td>1.40</td><td>2.25 18/0/0</td></tr></table>

## 6 RESULTS

Q1: the gates are nearly free; the selective objective is not. $\mathbf { A } \mathbf { t } ~ \phi = 1$ the gates are inactive, so SelPI is an IPIV model trained with the selective objective and carrying extra heads. Across the 18 tabular datasets it deviates from plain IPIV by a median —0.013 in PICP (a small loss of coverage), —6.9% in MPIW (narrower intervals, an improvement) and +4.4% in RMSE (a loss); per-dataset values are in Table 12. Two effects are mixed here; setting the acceptance target to 1 separates them, because the selective losses then reduce to $\mathrm { I P I V } \mathbf { s }$ own and only the architecture differs: the extra heads alone cost a median +0.6% in RMSE, -0.008 in PICP and -4.2% in MPIW, and improve on IPIV on 7 of 18 datasets. The rest of the RMSE deviation comes from the selective objective, which optimizes the model for the inputs it will accept; that is the effect the method exists to exploit, and a price only if one trains to select and then declines to. The architecture costs more than 10% in RMSE on four datasets (kin8nm, naval, video\_transcoding, black\_friday), with no relation to heteroscedasticity $( \rho = - 0 . 2 2 , p = 0 . 3 9 )$ . The answer to Q1: adding the abstention machinery leaves the interval model essentially intact, with about one point of coverage lost, narrower intervals, and a few percent of RMSE lost mostly to the selective objective rather than to the added heads.

Q2: intervals are usually not paid for in point accuracy. Abstaining pays in absolute terms: at a realized acceptance of \~0.55, SelPI2G cuts RMSE relative to IPIV on 18 of 18 datasets, by a median 31% (Table 2) and by up to 88% where error is concentrated in a minority of inputs (video\_transcoding, diamonds); black\_friday and naval gain least and are the weak cases of every later analysis. The informative comparison is with SelectiveNet-regression, a method with no interval to maintain. At $\phi = 0 . 7$ the median $\mathrm { S e l P I _ { 2 G } }$ trial has the lower selective RMSE on 14 of 18 tabular datasets, ties on two (wine, power-plant) and loses on two (kin8nm, naval), both neardeterministic tasks with little uncertainty for an interval method to characterize. The image tasks are less favorable (Appendix F). On IMDB-WIKI SelectiveNet wins wherever the two overlap, by 1.3–1.5 years of MAE (6.29 vs. 7.59 at $\phi \in [ 0 . 7 , 0 . 8 ) )$ ); on bone age the median $\mathrm { S e l P I _ { 2 G } }$ trial is slightly behind (16.2 vs. 15.9 months at $[ 0 . 7 , 0 . 8 ) \}$ , though the best trial wins every bucket. We attribute the IMDB-WIKI result to headroom: IPIV is already tight there, little error sits in a rejectable minority, and a single-purpose objective extracts what remains more efficiently than our joint one (§7). Part of any point-error advantage is, moreover, the structural coupling of $\mathring { \ S 3 } .$ , which is why Q3, not Q2, is our evidence that the gates learn something.

Q3: a learned gate beats width thresholding, most clearly on interval quality. Figure 2 shows three datasets; per-dataset values for all 20 are in Figure 9 and Table 15. With the base predictor held fixed, the learned gate attains a lower selective-RMSE AURC than the width threshold on all 20 datasets (Wilcoxon $p = 2 \times 1 0 ^ { - 6 } )$ , with a median selection efficiency of 0.59 (0.65 on the 18 tabular datasets, IQR 0.48–0.82). On the interval score the margin is far larger: 20 of 20, median SE 0.87 (IQR 0.74–0.96). The smallest gain is on IMDB-WIKI (SE 0.08: interval score 51.95 against 52.99 for width, oracle 38.20), consistent with its tight, near-homoscedastic predictor (§7). An oracle gap remains everywhere; the gate does not close it, and is closest on the most heteroscedastic datasets.

![](images/425aed91783477957969cc10c08dc6b63061f0c1ee58d74caa05c651e540142b.jpg)

![](images/582b8846502cc0fe25339f54a5450aae2b2da14429e054a7b0231d9393f1fa84.jpg)

![](images/ae0072fa6fd4e55d522da844356371e246df2a46bf1540c17c3cce05443a512d.jpg)  
Figure 2: Q3: selective RMSE vs. acceptance rate on a fixed predictor (single-gate $\mathrm { S e l P I _ { 1 G } }$ , one model per acceptance rate) for Concrete, NYC Taxi and bone age (illustrative only, §5). Curves: learned gate, width threshold, oracle, and deep-ensemble point disagreement on its own five-model predictor; bands: $\pm 1 \ : \mathrm { s . d . }$ over runs; lower is better.

Table 3: What each signal sees, and how well it selects. AUROC for coverage failure (cov.) and value failure (val.; captured, but error above the captured-set median), and median SE against each selector's own predictor's width threshold and oracle (<0: worse than width). The SE on the $g _ { \mathrm { P I } } / g _ { \mathrm { v a l } }$ rows is that of the Q3 gate $\scriptstyle ( \mathrm { S e l P I _ { 1 G } } ) ;$ Pt./Bd. disagr. is the deep-ensemble baseline on its own five-model predictor. Medians over 18 tabular datasets.
<table><tr><td rowspan="2">Signal</td><td colspan="2">AUROC</td><td colspan="2">selection efficiency</td></tr><tr><td>cov.↑</td><td>val.↑</td><td>sel. RMSE</td><td> $\mathrm { I S } _ { \alpha }$ </td></tr><tr><td>Width  $U - L$ </td><td>0.378</td><td>0.717</td><td>0 (reference)</td><td></td></tr><tr><td>gPI</td><td>0.790</td><td>0.633</td><td>0.59</td><td>0.87</td></tr><tr><td> $g _ { \mathrm { v a l } }$ </td><td>0.710</td><td>0.647</td><td></td><td></td></tr><tr><td>Pt. disagr.</td><td>0.584</td><td>0.606</td><td>-0.34</td><td>-0.62</td></tr><tr><td>Bd. disagr.</td><td>0.591</td><td>0.581</td><td>-0.37</td><td>-0.55</td></tr></table>

The gap between the two metrics is informative: a width threshold is a tolerable point-error selector but a poor interval selector. Rejecting wide intervals removes the inputs the model covered, so the accepted set retains the misses, and each miss costs $2 / \alpha = 4 0$ times its distance outside the interval. Decomposing $\mathrm { I S } _ { \alpha }$ at $\phi \approx 0 . 7$ confirms the mechanism (right block of Table 2): the gate accepts intervals wider than the width threshold's by a median 9.3% and still wins, because it cuts the miss-penalty term by a median 71.0% (miss rate 0.054 against 0.113). Width buys sharpness on the accepted set and pays in misses; the gate makes the opposite trade, and under a proper scoring rule that is the better one.

Q3, continued: the gate also beats ensemble disagreement. Width is not the only post-hoc alternative. The deep-ensemble baseline of §5 ranks a five-model IPIV ensemble by point or bound disagreement across members, the textbook epistemic signal, and Table 3 (right columns) compares it with the gate, each selector measured against its own predictor's width threshold and oracle. Comparing selection efficiency across the two pipelines, the gate wins on every dataset against both signals: 20 of 20 on selective RMSE (Wilcoxon $p = 2 \times \mathsf { \bar { 1 0 ^ { - 6 } } } )$ and 19 of 19 on the interval score $( p = 4 \times 1 0 ^ { - 6 } )$ . The disagreement selectors mostly fail to beat even their own ensemble's width threshold: on selective RMSE, point disagreement loses to width on 18 of 20 datasets and bound disagreement on 18 of 20; on the interval score, 13 of 19 and 14 of 19 respectively (yacht excluded). The left columns of Table 3 show why: both signals sit well below gPı on coverage failure. Per-dataset values are in Appendix O. Variance across independently trained networks does not substitute for a gate trained jointly with the predictor. (Yacht is excluded from the ensemble's $\mathrm { I S } _ { \alpha }$ column, where its SE is undefined; Appendix E.)

Two gates vs. one. At matched acceptance $\mathrm { S e l P I _ { 2 G } }$ beats $\mathrm { S e l P I _ { 1 G } }$ in selective RMSE on all 18 tabular datasets while producing wider intervals on 16 of them—the opposite of what an “it just picks easy points" explanation predicts—and it reduces the value-position error $\left| v - v ^ { \star } \right|$ on its accepted set on 18 of 18 (median 10. $6 \% )$ , so $g _ { \mathrm { v a l } }$ selects on the quantity it is trained for rather than on width by proxy. The two-budget frontier (Appendix I) shows the mechanism: shifting budget between the gates at a fixed joint rate trades interval quality against point accuracy, which one scalar gate cannot express. On images the picture is weaker: $\mathrm { S e l P I _ { 2 G } }$ wins on bone age, but on IMDB-WIKI the single gate is better at three of five comparable acceptance rates.

## 7 ANALYSIS: WHAT THE GATES LEARN THAT WIDTH DOES NOT

Complementarity and specialization. Treating “did the interval miss?"and “was the value estimate misplaced?" as binary labels, Table 3 asks how well each signal ranks them. Width predicts value failure well (0.717), as the coupling of $\ S 3$ implies, but on coverage failure it scores 0.378— below chance, and therefore anti-predictive: wider intervals miss less often, so a reject-if-wide rule systematically discards the inputs whose intervals were correct. $g _ { \mathrm { P I } }$ reaches 0.790 on that same event. The $\mathrm { g a t e s } ^ { \prime }$ rank correlation with width stays moderate (per-dataset means from —0.77 to $+ 0 . 2 1 ) \colon$ width is a valid aleatoric signal, but reproducing it would be learning nothing. Splitting rejections by which gate alone is responsible confirms the specialization: $g _ { \mathrm { v a l } } \mathbf { \dot { s } }$ unique rejections are enriched for value failure on 17 of 18 datasets, and $\boldsymbol { g _ { \mathrm { P I } } } ^ { \prime } \mathbf { s }$ for coverage failure on 13 of the 17 where it rejects anything uniquely (Table 11).

When does learned selection help? We relate each dataset's selection efficiency to three task properties; with 20 datasets this is descriptive, not inferential (Appendix J). Selection efficiency rises with residual heteroscedasticity (Spearman $\rho = 0 . 5 4 , p = 0 . 0 1 3 )$ , as predicted. The deep-ensemble selectors of Table 3 show the opposite pattern (point disagreement $\rho = - 0 . 8 7$ , bound disagreement $\rho = - 0 . 8 3$ , both $p < 1 0 ^ { - 4 } )$ , so the relationship is specific to a gate trained jointly with the predictor. Target kurtosis and full-acceptance miscoverage show no usable relationship (Appendix J). The extremes fit: video\_transcoding and diamonds sit near 0.99, while particulate-matter, power-plant and IMDB-WIKI—tight, near-homoscedastic predictors—sit below 0.16, where a width threshold is already close to optimal.

Shift-like subpopulations. Width is aleatoric and shift is epistemic, so shift is where a learned gate should most clearly beat width. Without retraining, we take for each dataset the covariate that best tracks realized error, treat its top quintile as a shifted tail, and ask how much each selector's abstention rate rises there at its $\phi = 0 . 7$ threshold. The gate's abstention rises by a median +0.193 against +0.046 for width, and the gate reacts more strongly on 15 of 18 datasets. The deep-ensemble selectors react more than width here too (point disagreement +0.131, bound disagreement +0.165, on the 15 datasets where the tail is defined for all selectors), but less than the gate: ensembling captures only part of the epistemic signal the gate exploits. The tail is chosen with knowledge of the error, so this is weaker evidence than an injected corruption, but the direction is clear (Appendix K).

Ablations and cost. Joint training is essential (Table 13, Appendix L): a gate trained on a frozen IPIV backbone is a median 5.6% worse and loses on 15 of 18 datasets, by up to 54%; the gates shape the representation rather than merely reading it. The auxiliary triple matters, though less $( + 2 . 0 \%$ worse on 14 of 18). The $\sqrt { S _ { g } }$ scaling does not pay for itself empirically (+0.2%, 9 worse and 9 better); we retain it on principle (§4.2). Each gate adds $1 6 h + 6 5$ parameters for hidden width h (44% on a small tabular model, \~0.2% on the image backbones); training takes a median 1.63× IPIV's wall-clock per model. Inference cost is unchanged: one forward pass (Appendix P).

## 8 CONCLUSION

A regression model can learn to predict, bound and abstain jointly. The extra heads cost almost nothing at full acceptance; the resulting selective regressor matches or beats SelectiveNet on most tabular tasks but not on IMDB-WIKI; and on a fixed predictor a learned gate beats width thresholding everywhere on a proper interval score, because width is anti-predictive of the misses that score charges for. It also beats deep-ensemble disagreement on every dataset. Two limitations remain. The accepted set carries no finite-sample coverage guarantee, since abstention breaks the exchangeability that split-conformal calibration relies on; calibrating conditional on acceptance would restore one and make SelPI the joint-training counterpart of selective conformal risk control (Xu et al., 2025). And our gates abstain on hard inputs, whereas some applications care whether acting is expected to lose; retargeting $g _ { \mathrm { v a l } }$ to a downstream utility is a natural extension.

## AI USE STATEMENT

Large language model (LLM) tools were used solely to assist with the writing of this manuscript, specifically for language polishing, including improving grammar, clarity, phrasing, and overall readability. These tools were not used to generate research ideas, design or conduct experiments, analyze results, or produce scientific content. All AI-assisted edits were reviewed and verified by the authors, who take full responsibility for the content of this paper.

## REPRODUCIBILITY STATEMENT

Everything needed to reproduce our results is specified in the appendices. Appendix C lists every dataset with its source, size, target statistics, preprocessing, and split protocol (five random 90/10 splits with fixed seeds for the tabular datasets, the provider's split for Year-MSD, five-fold crossvalidation for IMDB-WIKI, and the fixed 200-image test set for bone age). Appendix B states the training objective in full, including the hard and soft capture indicators, the $\sqrt { S _ { g } }$ scaling, and the absence of a warm start. Appendix D gives the architecture and every hyperparameter of SelPI and of each baseline, the values inherited unchanged from IPIV and SelectiveNet, the per-dataset search spaces and tuning budgets, and the rule by which the median search trial is reported. Appendix E defines every metric, the validation-based threshold calibration used for all selectors, and the selection-efficiency normalization; Appendix H describes the statistical tests behind every reported p-value and win/loss count. The deep-ensemble baseline (number of members, aggregation, and ranking signals) is specified in §5 and Appendix O, and compute and hardware in Appendix P. Full per-dataset results for every experiment summarized in the main text are reported in Appendices F–O.

## ETHICS STATEMENT

This work uses only publicly available benchmark datasets under their original licenses: the UCI and OpenML regression tasks, the RSNA pediatric bone age challenge data (de-identified radiographs released for research), and IMDB-WIKI (publicly posted celebrity photographs with age labels). No new data involving human subjects were collected, and no attempt was made to identify individuals. Age estimation from faces and radiographs is used here as a benchmark for interval prediction with abstention, not as a deployable system; both datasets are known to be demographically unbalanced, and a selector trained on them may abstain at different rates across groups, which we did not audit. More broadly, the accepted set of SelPI carries no finite-sample coverage guarantee (§8), so an abstention decision should be read as a learned signal of predictive difficulty, not as a certification of safety, and any high-stakes use would require domain-specific validation and human oversight. The authors declare no competing interests.

## REFERENCES

Anastasios N Angelopoulos and Stephen Bates. Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4):494–591, 2023.

Xin Cheng, Yuzhou Cao, Haobo Wang, Hongxin Wei, Bo An, and Lei Feng. Regression with costbased rejection. Advances in Neural Information Processing Systems, 36:45172–45196, 2023.

Chao Chow. On optimum recognition error and reject tradeoff. IEEE Transactions on information theory, 16(1):41–46, 1970.

Youngseog Chung, Willie Neiswanger, Ian Char, and Jeff Schneider. Beyond pinball loss: Quantile methods for calibrated uncertainty quantification. Advances in Neural Information Processing Systems, 34:10971–10984, 2021.

Corinna Cortes, Giulia DeSalvo, and Mehryar Mohri. Learning with rejection. In International conference on algorithmic learning theory, pp. 67–82. Springer, 2016.

Janez Demšar. Statistical comparisons of classifiers over multiple data sets. Journal of Machine learning research, 7(Jan):1–30, 2006.

Ran El-Yaniv and Yair Wiener. On the foundations of noise-free selective classification. Journal of Machine Learning Research, 11(53):1605–1641, 2010. URL http: // jmlr.org/papers/ v11/el-yaniv10a.html.

Leo Feng, Mohamed Osama Ahmed, Hossein Hajimirsadeghi, and Amir Abdi. Towards better selective classification. arXiv preprint arXiv:2206.09034, 2022.

Adam Fisch, Tommi Jaakkola, and Regina Barzilay. Calibrated selective classification. arXiv preprint arXiv:2208.12084, 2022.

Sebastian Felix Fischer, Matthias Feurer, and Bernd Bischl. Openml-ctr23-a curated tabular regression benchmarking suite. In AutoML Conference 2023 (Workshop), 2023.

Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In international conference on machine learning, pp. 1050–1059. PMLR, 2016.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. Advances in neural information processing systems, 30, 2017.

Yonatan Geifman and Ran El-Yaniv. Selectivenet: A deep neural network with an integrated reject option. In International conference on machine learning, pp. 2151–2159. PMLR, 2019.

Yonatan Geifman, Guy Uziel, and Ran El-Yaniv. Bias-reduced uncertainty estimation for deep neural classifiers. arXiv preprint arXiv:1805.08206, 2018.

Isaac Gibbs and Emmanuel Candes. Adaptive conformal inference under distribution shift. Advances in Neural Information Processing Systems, 34:1660–1672, 2021.

Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American statistical Association, 102(477):359–378, 2007.

Léo Grinsztajn, Edouard Oyallon, and Gaël Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? Advances in neural information processing systems, 35: 507–520, 2022.

Safwan S Halabi, Luciano M Prevedello, Jayashree Kalpathy-Cramer, Artem B Mamonov, Alexander Bilbily, Mark Cicero, Ian Pan, Lucas Araújo Pereira, Rafael Teixeira Sousa, Nitamar Abdala, et al. The rsna pediatric bone age machine learning challenge. Radiology, 290(2):498–503, 2019.

José Miguel Hernández-Lobato and Ryan Adams. Probabilistic backpropagation for scalable learning of bayesian neural networks. In International conference on machine learning, pp. 1861– 1869. PMLR, 2015.

Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. Machine learning, 110(3):457–506, 2021.

Wenming Jiang, Ying Zhao, and Zehan Wang. Risk-controlled selective prediction for regression deep neural network models. In 2020 international joint conference on neural networks (IJCNN), pp. 1–8. IEEE, 2020.

Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? Advances in neural information processing systems, 30, 2017.

Abbas Khosravi, Saeid Nahavandi, Doug Creighton, and Amir F Atiya. Lower upper bound estimation method for construction of neural network-based prediction intervals. IEEE transactions on neural networks, 22(3):337–346, 2010.

Roger Koenker and Gilbert Bassett Jr. Regression quantiles. Econometrica: journal of the Econometric Society, pp. 33–50, 1978.

Divake Kumar, Nastaran Darabi, Sina Tayebati, and Amit Ranjan Trivedi. Beyond confidence: Adaptive abstention in dual-threshold conformal prediction for autonomous system perception. arXiv preprint arXiv:2502.07255, 2025.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. Advances in neural information processing systems, 30, 2017.

Ziyin Liu, Zhikang Wang, Paul Pu Liang, Russ R Salakhutdinov, Louis-Philippe Morency, and Masahito Ueda. Deep gamblers: Learning to abstain with portfolio theory. Advances in Neural Information Processing Systems, 32, 2019.

David A Nix and Andreas S Weigend. Estimating the mean and variance of the target probability distribution. In Proceedings of 1994 ieee international conference on neural networks (ICNN'94), volume 1, pp. 55–60. IEEE, 1994.

Tim Pearce, Alexandra Brintrup, Mohamed Zaki, and Andy Neely. High-quality prediction intervals for deep learning: A distribution-free, ensembled approach. In International conference on machine learning, pp. 4075–4084. PMLR, 2018.

Andrea Pugnana, Carlos Mougan, and Dan Saattrup Nielsen. Model agnostic explainable selective regression via uncertainty estimation. arXiv preprint arXiv:2311.09145, 2023.

Yaniv Romano, Evan Patterson, and Emmanuel Candes. Conformalized quantile regression. Advances in neural information processing systems, 32, 2019.

Rasmus Rothe, Radu Timofte, and Luc Van Gool. Deep expectation of real and apparent age from a single image without facial landmarks. International Journal of Computer Vision, 126(2):144– 157, 2018.

Mahmoud Salem, Mohamed Osama Ahmed, Frederick Tung, and Gabriel Oliveira. Gumbel-softmax selective networks. arXiv preprint arXiv:2211.10564, 2022.

Tárik S Salem, Helge Langseth, and Heri Ramampiaro. Prediction intervals: Split normal mixture from quality-driven deep ensembles. In Conference on uncertainty in artificial intelligence, pp. 1179–1187. PMLR, 2020.

Abhin Shah, Yuheng Bu, Joshua K Lee, Subhro Das, Rameswar Panda, Prasanna Sattigeri, and Gregory W Wornell. Selective regression under fairness criteria. In International Conference on Machine Learning, pp. 19598–19615. PMLR, 2022.

Eli Simhayev, Gilad Katz, and Lior Rokach. Integrated prediction intervals and specific value predictions for regression problems using neural networks. Knowledge-Based Systems, 247:108685, 2022.

Anna Sokol, Nuno Moniz, and Nitesh V Chawla. Conformalized selective regression. Discover Data, 4(1):14, 2026.

Natasa Tagasovska and David Lopez-Paz. Single-model uncertainties for deep learning. Advances in neural information processing systems, 32, 2019.

Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic learning in a random world. Springer, 2005.

Robert L Winkler. A decision-theoretic approach to interval estimation. Journal of the American Statistical Association, 67(337):187–191, 1972.

Yunpeng Xu, Wenge Guo, and Zhi Wei. Selective conformal risk control. arXiv preprint arXiv:2512.12844, 2025.

Ahmed Zaoui and Clément Dombry. Distributional regression with reject option. arXiv preprint arXiv:2503.23782, 2025.

Ahmed Zaoui, Christophe Denis, and Mohamed Hebiri. Regression with reject option and application to knn. Advances in Neural Information Processing Systems, 33:20073–20082, 2020.

## A EXTENDED RELATED WORK: SELECTION WITH CONFORMAL GUARANTEES

Selective conformal risk control (Xu et al., 2025) selects post hoc and then calibrates prediction sets for classification on the accepted subset; dual-threshold conformal abstention (Kumar et al., 2025) does the same for perception models, and adaptive conformal inference (Gibbs & Candes, 2021) tracks coverage online under drift. Conformalized selective regression (Sokol et al., 2026) goes the other way and uses a conformal score as the confidence function of a post-hoc selective regressor. All of these take either the predictor or the selector as given and add a calibration layer; we learn the selector jointly with the intervals, on regression intervals rather than label sets, and view a conformal wrapper calibrated conditional on acceptance as a natural extension (§8). SelectiveNet's selection machinery is loss-agnostic, but $\mathrm { I P I V } \mathbf { s }$ heads are regression-specific; a classification analogue would target prediction sets and is left to future work.

## B LOSS DETAILS

Hard and soft capture in the IPIV loss. In equation 4 the hard indicator $k _ { i }$ has zero gradient almost everywhere, so the coverage term uses the softened capture $\tilde { k } _ { i } = \sigma \big ( s \left( U _ { i } - y _ { i } \right) \big ) \sigma \big ( s \left( y _ { i } - L _ { i } \right) \big )$ with $s = 1 6 0$ . The two indicators are not interchangeable across the two terms: the soft $\tilde { k }$ enters only the coverage penalty, while $\mathrm { M P I W _ { c a p t } }$ uses the hard $k ,$ so that an interval which misses its target contributes exactly zero to the width objective and is never encouraged to shrink further. The $\sqrt { n }$ factor is the batch size entering as a confidence weight: a larger sample makes the estimate of PICP more trustworthy and a shortfall correspondingly more costly. SelPI preserves this split exactly in equation 5: k in $\mathrm { P I C P } _ { g } ,$ k in $\mathrm { M P I W } _ { g }$

Why $\sqrt { S _ { g } }$ and not $\sqrt { n }$ . When coverage is estimated only on the (softly) accepted subset, the amount of data backing the estimate is $S _ { g } = \textstyle \sum _ { i } g _ { i }$ , not n. Keeping $\sqrt { n }$ over-weights the coverage constraint by a factor $\sqrt { n / S _ { g } } \mathrm { - } 1 . 8 \times$ at $\phi _ { \mathrm { t } } = 0 . 3$ , and more early in training, when g sits near 0.5 everywhere. Empirically the choice makes little difference (Table 13); we keep $\sqrt { S _ { g } }$ because it is the consistent generalization of equation 4.

Why there is no warm start. A warm start that holds $\lambda _ { \mathrm { P I } } = 0$ for the first ≈20% of epochs, letting the gates and the value head settle before interval pressure is applied, seems natural but does not work: with the coverage penalty off nothing stops the two bound heads from crossing, and a nontrivial fraction of runs end the warm-up with $U \bar { ( \boldsymbol { x } ) } < L ( \boldsymbol { x } )$ and never recover once the penalty is switched on. The coverage penalty is not a constraint on an already-valid interval—it is what makes the interval well ordered in the first place—so we train with all penalties active from the first epoch.

## C DATASET DETAILS

Inputs and targets are standardized to zero mean and unit variance; all interval widths and the PICP figures below are therefore in standardized units, while point errors are reported on the original scale. d counts features after one-hot expansion of categoricals for the group-C sets. Naval ships two regression targets and we keep the first, following the convention of the QD and IPIV evaluations. For IMDB-WIKI the two epoch budgets are 90 for SelPI and 29 for IPIV and SelectiveNet-regression; the baselines converge within their shorter budget. Real per-epoch training curves confirm this is measured, not assumed: IPIV and SelectiveNet-regression plateau by epoch \~15–20, while SelPI's harder joint objective — interval, value, selection and auxiliary heads trained together — was still improving at epoch 29, confirmed directly by retraining one fold to 90 epochs, where selective MAE fell from ${ \sim } 8 . 5 ~ \mathrm { t o } \sim 5 . 0 $ . The 90-epoch budget is not an inflated allowance chosen to favor SelPI: it matches the original IPIV paper's own IMDB-WIKI training length exactly.

## D HYPERPARAMETERS AND TUNING BUDGET

Method-level constants, fixed across all 20 datasets. $\gamma = \beta = \beta _ { h } = 0 . 5$ (the convex-combination weights of equation 11, inherited from SelectiveNet's α and IPIV's β respectively); nominal interval

Table 4: The 20 datasets. n and d are read directly from the data files as loaded by our pipeline. “Test/split" is the held-out count per repetition under the protocol in the last column. Epochs are the per-dataset budgets inherited from IPIV's published configuration for the UCI sets and sized by timing probes for the larger ones. Every tabular dataset uses the same 10-point acceptance grid φ ∈ {0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95 }, one model trained per point.
<table><tr><td>Dataset</td><td>n</td><td>d</td><td>Test/split</td><td>Epochs</td><td>Reps</td><td>Split protocol</td></tr><tr><td colspan="7">(A) UCI PI suite — the standard benchmark of QD and IPIV</td></tr><tr><td>Yacht</td><td>308</td><td>6</td><td>31</td><td>2000</td><td>5</td><td>90/10 random</td></tr><tr><td>Boston</td><td>506</td><td>13</td><td>51</td><td>300</td><td>5</td><td>90/10 random</td></tr><tr><td>Energy</td><td>768</td><td>8</td><td>77</td><td>1200</td><td>5</td><td>90/10 random</td></tr><tr><td>Concrete</td><td>1,030</td><td>8</td><td>103</td><td>800</td><td>5</td><td>90/10 random</td></tr><tr><td>Wine</td><td>1,599</td><td>11</td><td>160</td><td>1000</td><td>5</td><td>90/10 random</td></tr><tr><td>Kin8nm</td><td>8,192</td><td>8</td><td>819</td><td>800</td><td>5</td><td>90/10 random</td></tr><tr><td>Power-plant</td><td>9,568</td><td>4</td><td>957</td><td>500</td><td>5</td><td>90/10 random</td></tr><tr><td>Naval</td><td>11,934</td><td>16</td><td>1,193</td><td>1000</td><td>5</td><td>90/10 random</td></tr><tr><td>Protein</td><td>45,730</td><td>9</td><td>4,573</td><td>600</td><td>5</td><td>90/10 random</td></tr><tr><td>Year-MSD</td><td>515,345</td><td>90</td><td>51,630</td><td>300</td><td>1</td><td>fixed provider split</td></tr><tr><td colspan="7">(C) Larger tabular regression — added for statistical power at low φ</td></tr><tr><td>superconduct</td><td>21,263</td><td>79</td><td>2,126</td><td>500</td><td>5</td><td>90/10 random</td></tr><tr><td>sarcos</td><td>48,933</td><td>21</td><td>4,894</td><td>400</td><td>5</td><td>90/10 random</td></tr><tr><td>diamonds</td><td>53,940</td><td>26</td><td>5,394</td><td>400</td><td>5</td><td>90/10 random</td></tr><tr><td>video_transcoding</td><td>68,784</td><td>24</td><td>6,879</td><td>350</td><td>5</td><td>90/10 random</td></tr><tr><td>medical_charges</td><td>163,065</td><td>3</td><td>16,307</td><td>200</td><td>5</td><td>90/10 random</td></tr><tr><td>black_friday</td><td>166,821</td><td>23</td><td>16,682</td><td>200</td><td>5</td><td>90/10 random</td></tr><tr><td>particulate-matter</td><td>394,299</td><td>26</td><td>39,430</td><td>120</td><td>5</td><td>90/10 random</td></tr><tr><td>nyc-taxi-green</td><td>581,835</td><td>9</td><td>58,184</td><td>100</td><td>5</td><td>90/10 random</td></tr><tr><td colspan="7">(B) Image regression — IPIV&#x27;s own two image tasks</td></tr><tr><td>RSNA bone age (VGG-16)</td><td></td><td></td><td>200</td><td>50</td><td>1</td><td>fixed split</td></tr><tr><td>IMDB-WIKI (DenseNet)</td><td> ${ \sim } 1 7 1 , 8 5 2 \mathrm { ~  ~ { ~ \longrightarrow ~ } ~ } { \sim } 3 4 , 3 7 0$ </td><td></td><td></td><td>90/29</td><td>5</td><td>5-fold CV</td></tr></table>

level $1 - \alpha = 0 . 9 5 ;$ softening factor $s = 1 6 0$ and acceptance-penalty weight $\lambda _ { \mathrm { s e l } } = 3 2 ,$ taken unchanged from IPIV and SelectiveNet; gate hidden width 16 with batch normalization; warm-start fraction 0 (§4.4); weight decay 0, matching IPIV.

Table 5: Per-dataset settings. Everything except the learning rate is inherited unchanged from IPIV's published configuration for that dataset. The learning rate is capped at $5 \times 1 0 ^ { - 3 }$ because IPIV's larger values destabilize the multi-term selective objective (§4.4); where the cap binds, the decay rate is raised to compensate.
<table><tr><td>Dataset</td><td>Hidden</td><td> $\lambda _ { \mathrm { P I } }$ </td><td> $\sigma _ { \mathrm { i n i t } }$ </td><td>Learning rate</td><td>Decay</td></tr><tr><td>Boston</td><td>50</td><td>15.0</td><td>0.100</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>0.990</td></tr><tr><td>Concrete</td><td>50</td><td>15.0</td><td>0.200</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>0.990</td></tr><tr><td>Energy</td><td>50</td><td>17.0</td><td>0.350</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>0.990</td></tr><tr><td>Kin8nm</td><td>50</td><td>12.0</td><td>0.436</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>0.990</td></tr><tr><td>Naval</td><td>50</td><td>4.0</td><td>0.100</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>0.990</td></tr><tr><td>Power-plant</td><td>50</td><td>15.7</td><td>0.257</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>0.990</td></tr><tr><td>Wine</td><td>50</td><td>40.0</td><td>0.200</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>0.990</td></tr><tr><td>Yacht</td><td>50</td><td>6.0</td><td>0.200</td><td> $5 \times 1 0 ^ { - 3 }$ </td><td>0.980</td></tr><tr><td>Protein</td><td>100</td><td>40.0</td><td>0.200</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td>0.999</td></tr><tr><td>Year-MSD and all 8 group-C sets</td><td>100</td><td>40.0</td><td>0.200</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td>0.999</td></tr></table>

Batch size is 100 for the UCI sets and 256 for Year-MSD, medical\_charges, black\_friday, particulatematter and nyc-taxi-green. Optimization is Adam with exponential decay throughout.

Dual-gate search. The two acceptance budgets and the four weights with no counterpart in either parent method are tuned per dataset by Optuna (TPE), minimizing selective RMSE at a nominal joint acceptance rate of 0.7, three seeds per trial. The search space is $\phi _ { \mathrm { t } } ^ { \mathrm { P I } } , \phi _ { \mathrm { t } } ^ { \mathrm { v a l } } \in [ 0 . 5 , 0 . 9 ]$ uniform; $\mathbf { \check { \lambda } } _ { \mathrm { s e l } } ^ { \mathrm { P I } } , \lambda _ { \mathrm { s e l } } ^ { \mathrm { v a l } } \mathbf { \check { \lambda } } \in \left[ 8 , 6 4 \right]$ log-uniform; $\lambda _ { \mathrm { f r } } \in [ \bar { 1 } , 3 2 ]$ log-uniform; $\bar { \beta _ { \mathrm { P I } } } \in [ 0 , 1 ]$ uniform. Because the joint acceptance rate is measured rather than targeted (§4.3), the TPE population clusters in the middle of the range, so each dataset's main search is supplemented by dedicated low- and high-band batches that constrain the two budgets to [0.50, 0.58] and [0.82, 0.90] (and later [0.90, 0.97]) respectively. Trial counts after all batches are 149–287 per tabular dataset (180–287 on the ten UCI sets, 149–165 on the eight larger ones), 177 for bone age and 100 for IMDB-WIKI. Reported trials are filtered to those that are well calibrated $( \mathrm { P I C P _ { s e l } } \overset { \cdot } { \geq } 0 . 8 5 )$ and non-degenerate; results are then bucketed by realized joint acceptance rate in 0.1-wide bins, and only buckets with at least 5 trials are reported.

SelectiveNet-regression baseline. Its own canonical configuration, untuned: one hidden layer of 64 ReLU units with batch normalization, selection head of 16 units, learning rate $5 \times 1 0 ^ { - \dot { 4 } }$ , 800 epochs, 5 runs, over its own published coverage grid. It brings its own backbone rather than sharing IPIV's, which is a real confound on the small nonlinear sets: its full-acceptance RMSE on Yacht is 4.03 against IPIV's 1.04, so a large part of the SelPI margin there is backbone rather than selection.

## E METRIC DEFINITIONS

Write $A _ { \phi } = \{ i : x _ { i }$ accepted at rate $\phi \}$ for the accepted set, $S = \vert A _ { \phi } \vert$ , and $C = \{ i \in A _ { \phi } : k _ { i } = 1 \}$ for its captured subset. Every metric below is the instantiation of equation 1 with the stated persample loss.

• Selective PICP: $\begin{array} { r } { \operatorname { P I C P } ( \phi ) = \frac { 1 } { S } \sum _ { i \in A _ { \phi } } k _ { i } } \end{array}$ , reported as the deviation $| \mathrm { P I C P } ( \phi ) - ( 1 - \alpha ) |$ , since over- and under-coverage are both failures

• Selective MPIW: $\begin{array} { r } { \frac { 1 } { S } \sum _ { i \in A _ { \phi } } \left( U _ { i } - L _ { i } \right) } \end{array}$ . Targets are standardized, so this is already in units of the target standard deviation.

• Selective RMSE / MAE: $\begin{array} { r } { \left( \frac { 1 } { S } \sum _ { i \in A _ { \phi } } ( \hat { y } _ { i } - y _ { i } ) ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$ and $\begin{array} { r } { \frac { 1 } { S } \sum _ { i \in A _ { \phi } } \left| \hat { y } _ { i } - y _ { i } \right| } \end{array}$ , on the original target scale. MAE is used for the two image tasks, following their convention; RMSE elsewhere.

• Selective interval score: $\begin{array} { r } { \frac { 1 } { S } \sum _ { i \in A _ { \phi } } \operatorname { I S } _ { \alpha } ( L _ { i } , U _ { i } ; y _ { i } ) } \end{array}$ with $\mathrm { I S } _ { \alpha }$ as in equation 12.

• Value-position error: $\textstyle { \frac { 1 } { | C | } } \sum _ { i \in C } | v _ { i } - v _ { i } ^ { \star } | , v _ { i } ^ { \star } = ( y _ { i } - L _ { i } ) / ( U _ { i } - L _ { i } )$ , defined only on captured inputs since $v ^ { \star } \notin [ 0 , 1 ]$ otherwise; we report $| C | / S$ alongside it.

• AURC: the trapezoidal area of any of the above against φ over the grid [0.3, 1], divided by 0.7 so it reads as an average rather than an area. E-AURC subtracts the oracle computed on the same base predictor.

• Selection efficiency $\mathrm { S E } = \mathrm { ( A U R C _ { w i d t h } - A U R C _ { m e t h o d } ) / ( A U R C _ { w i d t h } - A U R C _ { o r a c l e } ) }$ : computed against the same-predictor width threshold and oracle, never a separately trained IPIV's. $\mathrm { S } \bar { \mathrm { E } } = 1$ is the oracle, $\mathrm { S E } ^ { \mathrm { - } } = 0$ the width threshold, and $\mathrm { S E } < 0$ a selector that is worse than a width threshold on its own predictor; SE is undefined when the width and oracle AURC coincide (yacht, for the ensemble's interval score). For the deep-ensemble selectors the “own predictor" is the fivemodel ensemble of §5, and width is its aggregated interval width, computed the same way on all three groups: IPIV's published aggregate (mean bounds widened by $z \sigma / { \sqrt { M } }$ across members). The disagreement signals’ failure-mode AUROC (Table 3, and the corresponding columns of Table 15) is reported on the 18 tabular datasets only, matching the scope of the diagnostic it extends; Appendix O's image rows report selection efficiency only, marked $^ { 6 6 } - ^ { 5 5 }$ in the AUROC columns.

Two events are used as binary labels for the gate diagnostics. A coverage failure is $k _ { i } = 0 \colon$ the interval missed. A value failure is $k _ { i } = 1$ together with $| \hat { y } _ { i } - y _ { i }$ above the median absolute error among captured inputs: the interval was right but the value head was misplaced within it. The median is taken per dataset, so a value failure is by construction a within-dataset relative event, and roughly half of all captured inputs qualify. The pipeline standardizes targets before training, so widths are already in s.d. units, while point errors are converted back to the original scale — width is therefore comparable across datasets and error is not, which is why selection efficiency, not raw RMSE, is the cross-dataset statistic.

## F IMAGE REGRESSION RESULTS

Table 6: Full image-regression grids, every trained point rather than the three buckets of Table 7. Top: bone age, single gate, one run over 200 test images — note the severe drift between target and realized acceptance rate, which is why Table 7 is keyed by the realized rate. Middle: IMDB-WIKI, single gate then the symmetric dual-gate grid, 5-fold CV at 90 epochs. Bottom: SelectiveNetregression on IMDB-WIKI over its own published coverage grid.
<table><tr><td>Target φ</td><td>Realized φ</td><td> $\mathrm { P I C P _ { s e l } }$ </td><td> $\mathrm { M P I W _ { s e l } }$ </td><td> $\mathbf { M A E } _ { \mathrm { s e l } }$ </td></tr><tr><td colspan="5">bone age, single gate — IPIV at φ = 1: 0.915, 2.189, 20.71</td></tr><tr><td>0.30</td><td>0.000</td><td></td><td></td><td></td></tr><tr><td>0.40</td><td>0.070</td><td>0.929</td><td>1.589</td><td>10.901</td></tr><tr><td>0.50</td><td>0.075</td><td>0.933</td><td>1.729</td><td>16.505</td></tr><tr><td>0.60</td><td>0.190</td><td>1.000</td><td>2.063</td><td>14.099</td></tr><tr><td>0.70</td><td>0.225</td><td>0.889</td><td>1.889</td><td>15.371</td></tr><tr><td>0.75</td><td>0.480</td><td>0.917</td><td>1.775</td><td>16.381</td></tr><tr><td>0.80</td><td>0.505</td><td>0.950</td><td>1.946</td><td>14.526</td></tr><tr><td>0.85</td><td>0.625</td><td>0.904</td><td>1.869</td><td>16.198</td></tr><tr><td>0.90</td><td>0.680</td><td>0.963</td><td>2.075</td><td>14.548</td></tr><tr><td>0.95</td><td>0.865</td><td>0.879</td><td>1.933</td><td>18.027</td></tr><tr><td colspan="5">IMDB-WIKI, single gate, 90 epochs — IPIV at φ = 1: 0.933, 2.835, 7.683</td></tr><tr><td>0.30</td><td>0.302</td><td>0.921</td><td>2.480</td><td>7.261</td></tr><tr><td>0.40</td><td>0.401</td><td>0.943</td><td>2.854</td><td>7.691</td></tr><tr><td>0.50</td><td>0.503</td><td>0.934</td><td>2.847</td><td>7.834</td></tr><tr><td>0.60</td><td>0.602</td><td>0.935</td><td>2.857</td><td>7.826</td></tr><tr><td>0.70</td><td>0.702</td><td>0.940</td><td>2.942</td><td>7.796</td></tr><tr><td>0.75</td><td>0.750</td><td>0.942</td><td>3.019</td><td>7.975</td></tr><tr><td>0.80</td><td>0.799</td><td>0.937</td><td>3.113</td><td>8.455</td></tr><tr><td>0.85</td><td>0.851</td><td>0.938</td><td>3.137</td><td>8.416</td></tr><tr><td>0.90</td><td>0.900</td><td>0.948</td><td>3.326</td><td>8.433</td></tr><tr><td>0.95</td><td>0.959</td><td>0.926</td><td>3.259</td><td>9.113</td></tr><tr><td colspan="5">IMDB-WIKI, dual gate, symmetric budgets; second column is the realized joint rate</td></tr><tr><td>0.5</td><td>0.490</td><td>0.942</td><td>2.884</td><td>7.641</td></tr><tr><td>0.6</td><td>0.591</td><td>0.942</td><td>3.142</td><td>8.252</td></tr><tr><td>0.7</td><td>0.694</td><td>0.945</td><td>3.217</td><td>8.446</td></tr><tr><td>0.8</td><td>0.795</td><td>0.939</td><td>3.159</td><td>8.376</td></tr><tr><td>0.9</td><td>0.898</td><td>0.932</td><td>3.201</td><td>8.841</td></tr><tr><td colspan="5">IMDB-WIKI, SelectiveNet-regression — cols. are realized φ, sel. MAE, full MAE, aux. MAE</td></tr><tr><td>0.70</td><td>0.700</td><td>6.294</td><td>7.257</td><td>7.256</td></tr><tr><td>0.75</td><td>0.751</td><td>6.486</td><td>7.314</td><td>7.303</td></tr><tr><td>0.80</td><td>0.799</td><td>6.506</td><td>7.202</td><td>7.204</td></tr><tr><td>0.85</td><td>0.851</td><td>6.755</td><td>7.328</td><td>7.338</td></tr><tr><td>0.90</td><td>0.901</td><td>6.847</td><td>7.242</td><td>7.242</td></tr><tr><td>0.95</td><td>0.951</td><td>6.999</td><td>7.214</td><td>7.213</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/411a02622c023959ab0e828cf4e06efc1e5015fdb70a847e020109ebb9872cb0.jpg)

![](images/5538227242f0d58d4c213a9c76ff7021aa045893cec0151a322acd93c1ab3244.jpg)  
Figure 3: All four comparisons on the two image tasks, at $1 - \alpha = 0 . 9 5 \colon$ single-gate SelPI (orange), dual-gate SelPI (blue: every Optuna trial faintly, best-per-bucket as the line), plain IPIV at full acceptance (gray dotted), and SelectiveNet-regression (red). Left, bone age: both SelPI variants beat IPIV everywhere and the dual gate beats SelectiveNet at every comparable acceptance rate. $R i g h t ,$ IMDB-WIKI: the dual gate beats IPIV over the 0.5–0.8 range but SelectiveNet wins clearly and consistently, by 0.7–1.6 years of MAE. IMDB-WIKI is the largest dataset in the study and this is the paper's clearest negative result (§6).

![](images/e3c1bd9daf458cdb391366b202de7060d00fd05b2e46beb78346cddb3832b14b.jpg)  
Figure 4: Risk-coverage curves for the two image tasks: selective PICP, MPIW and MAE against target acceptance rate, with plain IPIV at full acceptance as the hollow diamond. Note the contrast in calibration transfer: IMDB-WIKI's realized acceptance rate tracks its target almost exactly $( 0 . 7 0  0 . 7 0 0 , 0 . 9 5  0 . 9 5 9 )$ , whereas bone age drifts severely (target 0.90 → realized 0.68; target $0 . 3 0  \mathrm { r e a l i z e d } \ 0 . 0 0 )$ . The cause is sample size — thresholds calibrated on validation transfer cleanly to IMDB-WIKI's \~34k-sample folds and poorly to bone age's 200-image test set. Bone age's “target coverage"axis should therefore be read as a training knob, not as a description of what fraction of test inputs was accepted.

Table 7: Image regression, selective MAE at $1 - \alpha = 0 . 9 5$ (bone age in months, IMDB-WIKI in years), grouped by realized acceptance rate. Plain IPIV at full acceptance is 20.71 (bone age) and 7.68 (IMDB-WIKI). For the dual gate we give the best and the median trial in each bucket, because its budgets are tuned by search while SelectiveNet is run once at its published settings — the median is the fair comparison. Bone age: every SelPI variant beats IPIV by a wide margin, and the dual gate's best trial beats SelectiveNet in all three buckets. IMDB-WIKI: the dual gate beats IPIV in the two lower buckets, but SelectiveNet wins decisively wherever the two overlap, by 1.3–1.5 years. Rows are keyed by realized rather than target acceptance rate, which matters most for bone age, whose drift is severe (target 0.90 gives realized 0.68) because a threshold calibrated on validation transfers poorly to a 200-image test set; IMDB-WIKI has no such problem and is the more trustworthy of the two.
<table><tr><td rowspan="2">Realized acceptance rate</td><td colspan="3">RSNA bone age</td><td colspan="3">IMDB-WIKI</td></tr><tr><td>[0.5, 0.6)</td><td>[0.7, 0.8)</td><td>[0.8, 0.9)</td><td>[0.5, 0.6)</td><td>[0.7, 0.8)</td><td>[0.8, 0.9)</td></tr><tr><td>IPIV (φ = 1, no selection)</td><td></td><td>20.71</td><td></td><td></td><td>7.68</td><td></td></tr><tr><td>SelectiveNet-regression</td><td>14.96</td><td>15.94</td><td>17.01</td><td></td><td>6.29</td><td>6.76</td></tr><tr><td> $\mathrm { S e l P I _ { 1 G } }$  single gate</td><td>14.53</td><td></td><td>18.03</td><td>7.83</td><td>7.80</td><td>8.42</td></tr><tr><td> $\mathrm { S e l P I _ { 2 G } }$  dual gate (median trial)</td><td>15.07</td><td>16.23</td><td>17.83</td><td>7.92</td><td>8.16</td><td>8.21</td></tr><tr><td> $\mathrm { S e l P I _ { 2 G } }$  dual gate (best trial)</td><td>13.41</td><td>15.11</td><td>15.50</td><td>7.36</td><td>7.59</td><td>7.94</td></tr></table>

## G FULL PER-DATASET RESULTS

Table 8: Dual gate vs. plain IPIV by acceptance bucket, median search trial, reduction in selective RMSE; buckets are the realized joint acceptance rate. The gain decays as acceptance rises on every dataset, as it must: as φ → 1 the accepted set converges on the full test set and there is nothing left to select.
<table><tr><td>Dataset</td><td>[0.5, 0.6)</td><td>[0.6, 0.7)</td><td>[0.7, 0.8)</td><td>[0.8, 0.9)</td></tr><tr><td>yacht</td><td>63.3%</td><td>61.6%</td><td>56.1%</td><td>36.5%</td></tr><tr><td>boston</td><td>10.5%</td><td>8.5%</td><td>7.5%</td><td>4.2%</td></tr><tr><td>energy</td><td>59.4%</td><td>51.0%</td><td>50.5%</td><td>42.3%</td></tr><tr><td>concrete</td><td>33.5%</td><td>25.5%</td><td>22.2%</td><td>16.6%</td></tr><tr><td>wine</td><td>16.2%</td><td>14.8%</td><td>10.1%</td><td>8.1%</td></tr><tr><td>kin8nm</td><td>11.0%</td><td>7.1%</td><td>1.5%</td><td>-6.0%</td></tr><tr><td>power-plant</td><td>7.5%</td><td>4.7%</td><td>3.2%</td><td>1.7%</td></tr><tr><td>naval</td><td>15.9%</td><td>14.7%</td><td>3.0%</td><td>-6.4%</td></tr><tr><td>protein</td><td>28.3%</td><td>22.1%</td><td>14.4%</td><td>11.0%</td></tr><tr><td>Year-MSD</td><td>43.1%</td><td>35.3%</td><td>28.9%</td><td>21.2%</td></tr><tr><td>superconduct</td><td>56.9%</td><td>47.3%</td><td>39.5%</td><td>29.5%</td></tr><tr><td>sarcos</td><td>51.5%</td><td>44.7%</td><td>35.3%</td><td>29.4%</td></tr><tr><td>diamonds</td><td>73.3%</td><td>65.0%</td><td>56.6%</td><td>44.9%</td></tr><tr><td>video_transcoding</td><td>87.5%</td><td>84.8%</td><td>80.1%</td><td>73.1%</td></tr><tr><td>medical_charges</td><td>30.9%</td><td>25.7%</td><td>19.6%</td><td>16.1%</td></tr><tr><td>black_friday</td><td>8.1%</td><td>5.4%</td><td>4.9%</td><td>4.0%</td></tr><tr><td>particulate-matter</td><td>29.7%</td><td>24.7%</td><td>19.1%</td><td>15.1%</td></tr><tr><td>nyc-taxi-green</td><td>31.7%</td><td>28.0%</td><td>21.8%</td><td>18.5%</td></tr></table>

Table 9: What abstention buys. Single-gate $\mathrm { S e l P I _ { 1 G } }$ , one model trained per acceptance rate (not a searched trial population). Selective RMSE at four acceptance rates, against plain IPIV answering every query; 18 tabular datasets, $1 - \alpha = 0 . 9 5$ . The last column is the reduction at $\phi \ : = \ : 0 . 5 .$ Naval's full-acceptance error is 0.001, so its percentage is a small-denominator artifact rather than a meaningful loss.
<table><tr><td></td><td>IPIV</td><td colspan="4"> $\mathrm { S e l P I } _ { \mathrm { 1 G } }$  selective RMSE at  $\phi =$ </td><td>Cut at</td></tr><tr><td>Dataset</td><td> $( \phi = 1 )$ </td><td>0.3</td><td>0.5</td><td>0.7</td><td>0.9</td><td> $\phi = 0 . 5$ </td></tr><tr><td>yacht</td><td>1.04</td><td>0.45</td><td>0.44</td><td>0.55</td><td>1.07</td><td>57.7%</td></tr><tr><td>boston</td><td>2.64</td><td>2.23</td><td>2.18</td><td>2.63</td><td>2.55</td><td>17.5%</td></tr><tr><td>energy</td><td>1.63</td><td>0.61</td><td>0.91</td><td>0.84</td><td>1.10</td><td>44.4%</td></tr><tr><td>concrete</td><td>6.17</td><td>3.81</td><td>4.31</td><td>5.03</td><td>5.60</td><td>30.1%</td></tr><tr><td>wine</td><td>0.66</td><td>0.47</td><td>0.52</td><td>0.54</td><td>0.60</td><td>21.2%</td></tr><tr><td>kin8nm</td><td>0.08</td><td>0.07</td><td>0.07</td><td>0.08</td><td>0.09</td><td>13.6%</td></tr><tr><td>power-plant</td><td>4.10</td><td>3.53</td><td>3.73</td><td>3.78</td><td>3.99</td><td>9.1%</td></tr><tr><td>naval</td><td>0.001</td><td>0.002</td><td>0.002</td><td>0.001</td><td>0.002</td><td>-27.5%</td></tr><tr><td>protein</td><td>4.41</td><td>2.99</td><td>3.21</td><td>3.66</td><td>4.09</td><td>27.3%</td></tr><tr><td>Year-MSD</td><td>9.01</td><td>4.26</td><td>5.10</td><td>6.14</td><td>7.77</td><td>43.4%</td></tr><tr><td>superconduct</td><td>13.05</td><td>3.37</td><td>5.50</td><td>7.74</td><td>10.76</td><td>57.9%</td></tr><tr><td>sarcos</td><td>3.23</td><td>1.10</td><td>1.42</td><td>1.90</td><td>2.59</td><td>55.9%</td></tr><tr><td>diamonds</td><td>563.40</td><td>103.29</td><td>137.61</td><td>221.58</td><td>350.09</td><td>75.6%</td></tr><tr><td>video_transcoding</td><td>2.83</td><td>0.27</td><td>0.33</td><td>0.49</td><td>1.02</td><td>88.3%</td></tr><tr><td>medical_charges</td><td>0.08</td><td>0.05</td><td>0.06</td><td>0.06</td><td>0.07</td><td>30.2%</td></tr><tr><td>black_friday</td><td>0.39</td><td>0.34</td><td>0.35</td><td>0.37</td><td>0.38</td><td>8.4%</td></tr><tr><td>particulate-matter</td><td>0.39</td><td>0.22</td><td>0.27</td><td>0.31</td><td>0.35</td><td>31.2%</td></tr><tr><td>nyc-taxi-green</td><td>0.41</td><td>0.25</td><td>0.27</td><td>0.31</td><td>0.35</td><td>34.3%</td></tr></table>

Table 10: Dual gate vs. SelectiveNet-regression, selective RMSE at $7 0 \%$ acceptance. $\mathrm { \ " { \bf B } e s t \mathrm { \mathrm { \Omega } } }$ is the strongest configuration the search found, “median" the typical one; the median column is the fair comparison, because SelectiveNet is run once at its published settings while the dual gate is tuned (§5). Positive percentages favor SelPI. Naval and kin8nm are the two clear losses, and both are near-deterministic tasks with little uncertainty for an interval method to characterize.
<table><tr><td>Dataset</td><td>SelectiveNet</td><td> $\mathrm { S e l P I _ { 2 G } }$  best  $\mathrm { S e l P I } _ { \mathrm { 2 G } }$ </td><td>med.</td><td>Best</td><td>Median</td></tr><tr><td>yacht</td><td>1.62</td><td>0.27</td><td>0.35</td><td>83.1%</td><td>78.3%</td></tr><tr><td>boston</td><td>2.42</td><td>2.20</td><td>2.39</td><td>9.2%</td><td>1.3%</td></tr><tr><td>energy</td><td>1.17</td><td>0.63</td><td>0.82</td><td>46.2%</td><td>30.2%</td></tr><tr><td>concrete</td><td>4.66</td><td>4.24</td><td>4.59</td><td>9.1%</td><td>1.5%</td></tr><tr><td>wine</td><td>0.56</td><td>0.53</td><td>0.56</td><td>5.6%</td><td>0.0%</td></tr><tr><td>kin8nm</td><td>0.07</td><td>0.08</td><td>0.08</td><td>-9.0%</td><td>-15.5%</td></tr><tr><td>power-plant</td><td>3.86</td><td>3.81</td><td>3.92</td><td>1.3%</td><td>-1.5%</td></tr><tr><td>naval</td><td>0.0006</td><td>0.0009</td><td>0.0011</td><td>-51.3%</td><td>-83.6%</td></tr><tr><td>protein</td><td>3.93</td><td>3.37</td><td>3.56</td><td>14.3%</td><td>9.4%</td></tr><tr><td>Year-MSD</td><td>6.13</td><td>5.71</td><td>5.96</td><td>6.8%</td><td>2.8%</td></tr><tr><td>superconduct</td><td>8.63</td><td>6.81</td><td>7.40</td><td>21.1%</td><td>14.3%</td></tr><tr><td>sarcos</td><td>2.40</td><td>1.76</td><td>1.91</td><td>26.4%</td><td>20.3%</td></tr><tr><td>diamonds</td><td>237.56</td><td>205.32</td><td>232.45</td><td>13.6%</td><td>2.2%</td></tr><tr><td>video_transcoding</td><td>0.79</td><td>0.43</td><td>0.50</td><td>45.3%</td><td>37.4%</td></tr><tr><td>medical_charges</td><td>0.07</td><td>0.06</td><td>0.06</td><td>12.2%</td><td>10.2%</td></tr><tr><td>black_friday</td><td>0.38</td><td>0.35</td><td>0.37</td><td>6.2%</td><td>3.5%</td></tr><tr><td>particulate-matter</td><td>0.32</td><td>0.30</td><td>0.30</td><td>6.3%</td><td>4.1%</td></tr><tr><td>nyc-taxi-green</td><td>0.34</td><td>0.28</td><td>0.31</td><td>17.0%</td><td>9.6%</td></tr></table>

Gate specialisation: does each gate's rejection set match its intended failure mode? (lift > 1 = that failure type is over-represented among that gate's unique rejections)  
![](images/0d8452940f2f761e18c86f73e816b42c5fd6e39b2de568854cc8cc2049589379.jpg)  
Figure 5: Gate specialization, all 18 tabular datasets. For each dataset, the two bars show how over-represented each failure type is among the inputs that only gpı rejects and the inputs that only $g _ { \mathrm { v a l } }$ rejects, relative to that dataset's own base rate (dashed line at 1). The signature of successful specialization is a tall green bar on the left pair and a tall orange bar on the right pair. $g _ { \mathrm { v a l } }$ shows it on 17 of $1 8 ; g _ { \mathrm { P I } }$ on a clear majority, with the exceptions discussed in $^ { \ S 7 }$

Table 11: Each gate's unique rejections are enriched for the failure mode it was trained on. For inputs rejected by only one gate, how over-represented each failure type is relative to that dataset's base rate; 1.0 means no enrichment. $g _ { \mathrm { v a l } } \mathbf { \ ' s }$ unique rejections are enriched for value failure on 17 of 18 datasets; gp1's are enriched for coverage failure on 13 of the 17 where it has any unique rejections at all (on naval it has none, its rejection set being a subset of $g _ { \mathrm { v a l } } \mathbf { \dot { s } } )$ . The value gate is therefore the more consistently load-bearing of the two, which the aggregate results do not reveal.
<table><tr><td rowspan="2">Dataset</td><td colspan="3">rejected by gPI only</td><td colspan="3">rejected by  $g _ { \mathrm { v a l } }$  only</td></tr><tr><td>n</td><td>cov. lift</td><td>val. lift</td><td>n</td><td>val. lift</td><td>cov. lift</td></tr><tr><td>yacht</td><td>6</td><td>3.91</td><td>0.42</td><td>21</td><td>1.69</td><td>0.22</td></tr><tr><td>boston</td><td>16</td><td>1.20</td><td>1.10</td><td>13</td><td>0.78</td><td>1.11</td></tr><tr><td>energy</td><td>24</td><td>2.38</td><td>0.42</td><td>77</td><td>1.51</td><td>0.43</td></tr><tr><td>concrete</td><td>3</td><td>1.70</td><td>1.66</td><td>116</td><td>1.29</td><td>0.88</td></tr><tr><td>wine</td><td>15</td><td>2.81</td><td>0.51</td><td>253</td><td>1.40</td><td>0.78</td></tr><tr><td>kin8nm</td><td>63</td><td>1.88</td><td>0.24</td><td>974</td><td>1.32</td><td>0.66</td></tr><tr><td>power-plant</td><td>258</td><td>1.25</td><td>0.77</td><td>400</td><td>1.23</td><td>0.55</td></tr><tr><td>naval</td><td>0</td><td></td><td></td><td>846</td><td>1.48</td><td>0.29</td></tr><tr><td>protein</td><td>67</td><td>3.49</td><td>0.89</td><td>8912</td><td>1.56</td><td>0.85</td></tr><tr><td>Year-MSD</td><td>1293</td><td>0.69</td><td>0.96</td><td>3486</td><td>1.18</td><td>0.69</td></tr><tr><td>superconduct</td><td>39</td><td>1.43</td><td>0.78</td><td>797</td><td>2.10</td><td>0.30</td></tr><tr><td>sarcos</td><td>589</td><td>0.89</td><td>1.03</td><td>795</td><td>1.30</td><td>0.38</td></tr><tr><td>diamonds</td><td>360</td><td>3.67</td><td>0.12</td><td>1694</td><td>1.63</td><td>0.25</td></tr><tr><td>video_transcoding</td><td>969</td><td>1.01</td><td>0.71</td><td>2727</td><td>1.74</td><td>0.12</td></tr><tr><td>medical_charges</td><td>5203</td><td>0.96</td><td>1.11</td><td>2395</td><td>1.12</td><td>0.88</td></tr><tr><td>black_friday</td><td>164</td><td>0.21</td><td>1.12</td><td>28429</td><td>1.25</td><td>0.62</td></tr><tr><td>particulate-matter</td><td>1804</td><td>1.07</td><td>1.08</td><td>12221</td><td>1.10</td><td>0.95</td></tr><tr><td>nyc-taxi-green</td><td>3873</td><td>1.39</td><td>0.79</td><td>85594</td><td>1.25</td><td>0.90</td></tr></table>

## H STATISTICAL TESTS

Two comparisons are kept separate because only the second isolates the selector. The learned gate against a width threshold on the same predictor: 20 wins, 0 losses, $p = 2 \times 1 0 ^ { - 6 }$ , median selection efficiency 0.587. The whole pipeline against a separately trained IPIV + width: 16 wins, 3 losses, $p = 5 . 2 \times 1 0 ^ { - 4 }$ . On the interval score the same-predictor comparison gives 20 wins, 0 losses, median 0.872. The gate against deep-ensemble disagreement (a cross-pipeline comparison of selection efficiency): 20 wins, 0 losses $( p = 2 \times 1 0 ^ { - 6 } )$ on selective RMSE and 19 wins, 0 losses $( p = 4 \times 1 0 ^ { - 6 } )$ on the interval score, for both disagreement signals. Coverage within width quintiles of the accepted set (Angelopoulos & Bates, 2023) is nearly uniform on the six largest datasets $( \mathrm { g a p } \leq 0 . 0 1 7 )$ but not on the small UCI sets (up to 0.375 on yacht), where accepted-set calibration is partly carried by one easy sub-population. Friedman and Nemenyi tests are not applicable, since they rank a field of methods and each of our comparisons has two.

## I TWO-BUDGET FRONTIER

## J “WHEN DOES IT HELP": REGRESSION DETAILS

The figure supports only one of the three relationships §7 predicts. Heteroscedasticity: confirmed, $\rho = 0 . 5 4 , p = 0 . 0 1 3 , n = 2 0$ , where the heteroscedasticity statistic is Spearman $\rho ( U - L , | y - \hat { y } | )$ on the plain IPIV baseline. Excess kurtosis: no relationship, $\rho = - 0 . 0 3 , p = 0 . 9 1$ . Full-acceptance miscoverage: weakly negative and not significant, $\rho = - 0 . 3 2 , p = 0 . 2 0 -$ if anything the gate does worse where there is more miscoverage, the opposite of the intuitive direction.

## K DISTRIBUTION-SHIFT PROTOCOLS AND RESULTS

For each dataset and run we regenerate the exact test split the model saw, identify the covariate that best tracks realized error, and treat its top quintile as a shifted tail; no retraining is needed. Holding each selector's threshold fixed at its $\phi = 0 . 7$ value, we measure how much its abstention rate rises concrete — Two-Budget Frontier: Dual-Gate vs Single-Gate SelPI

ae e at p et  e o aed :(- .   -− 1 spe e   pe ser es-e : am -si   roet   de et es   e s e ee :e +l ie   t e aie ne cois  s i -s   s s sit   t is ss l it  e eds  e  se  ooy cae o t o ro  e  o pet   e  t t  e t :e ( to     e-non m  o  — matt  e  t  otc :   =  (t    =  i  em =  s a  no le se od pe oe ee e e oe e e =  e se os e e  s oe e oet se
<table><tr><td></td><td colspan="2">ρ(g, U − L)</td><td colspan="2">vs. IPIV + width</td><td colspan="2">Selection efficiency</td><td colspan="3">φ = 1 vs. IPIV</td><td colspan="3">Ablations (∆RMSEsel)</td></tr><tr><td>Dataset</td><td>mean</td><td>range</td><td>φ = 0.5</td><td>φ = 0.7</td><td>RMSE</td><td>ISα</td><td>∆PICP</td><td>∆MPIW</td><td>△RMSE</td><td>no aux</td><td>frozen</td><td>√n</td></tr><tr><td>yacht</td><td>-0.55</td><td>[-0.76, -0.37]</td><td>-14.9%</td><td>26.7%</td><td>0.833</td><td>0.868</td><td>-0.065</td><td>-13.7</td><td>+36.1</td><td>+41.1</td><td>+19.7</td><td>+20.4</td></tr><tr><td>boston</td><td>-0.18</td><td>[-0.33, -0.04]</td><td>6.3%</td><td>-1.2%</td><td>0.456</td><td>0.766</td><td>+0.024</td><td>+20.2</td><td>+4.2</td><td>-4.3</td><td>-7.7</td><td>-3.1</td></tr><tr><td>energy</td><td>-0.57</td><td>[-0.71, -0.36]</td><td>-55.8%</td><td>-12.9%</td><td>0.732</td><td>0.951</td><td>+0.005</td><td>+45.6</td><td>+21.4</td><td>-36.8</td><td>+6.8</td><td>+8.1</td></tr><tr><td>concrete</td><td>-0.13</td><td>[-0.37, 0.25]</td><td>16.0%</td><td>10.4%</td><td>0.572</td><td>0.841</td><td>+0.004</td><td>+5.6</td><td>+4.7</td><td>+1.3</td><td>+6.3</td><td>+1.2</td></tr><tr><td>wine</td><td>0.21</td><td>[0.06, 0.48]</td><td>5.3%</td><td>4.3%</td><td>0.423</td><td>0.795</td><td>-0.011</td><td>-9.9</td><td>-3.6</td><td>-1.1</td><td>-0.1</td><td>-2.6</td></tr><tr><td>kin8nm</td><td>-0.14</td><td>[-0.50, 0.35]</td><td>-13.3%</td><td>-12.6%</td><td>0.710</td><td>0.884</td><td>-0.014</td><td>+38.3</td><td>+51.7</td><td>+1.8</td><td>+11.1</td><td>-0.8</td></tr><tr><td>power-plant</td><td>-0.23</td><td>[-0.47, 0.24]</td><td>3.0%</td><td>0.6%</td><td>0.160</td><td>0.526</td><td>-0.009</td><td>-2.2</td><td>+0.1</td><td>+2.8</td><td>+2.9</td><td>+0.6</td></tr><tr><td>naval</td><td>-0.37</td><td>[-0.70, 0.06]</td><td>-255.0%</td><td>-110.7%</td><td>0.891</td><td>0.972</td><td>-0.063</td><td>+91.0</td><td>+103.1</td><td>+26.7</td><td>-20.0</td><td>-6.7</td></tr><tr><td>protein</td><td>-0.15</td><td>[-0.39, 0.40]</td><td>12.0%</td><td>8.0%</td><td>0.602</td><td>0.866</td><td>-0.025</td><td>-6.7</td><td>+2.0</td><td>+4.1</td><td>+4.4</td><td>+1.4</td></tr><tr><td>Year-MSD</td><td>-0.56</td><td>[-0.73, -0.01]</td><td>10.0%</td><td>12.3%</td><td>0.567</td><td>0.799</td><td>-0.026</td><td>-11.9</td><td>-0.7</td><td>+1.9</td><td>+3.9</td><td>-0.2</td></tr><tr><td>superconduct</td><td>-0.32</td><td>[-0.67, 0.54]</td><td>9.3%</td><td>14.9%</td><td>0.868</td><td>0.981</td><td>-0.031</td><td>-16.5</td><td>-0.1</td><td>+41.2</td><td>+13.2</td><td>-2.7</td></tr><tr><td>sarcos</td><td>-0.63</td><td>[-0.74, -0.38]</td><td>23.1%</td><td>18.2%</td><td>0.543</td><td>0.876</td><td>-0.011</td><td>-7.1</td><td>+0.7</td><td>+2.4</td><td>+33.3</td><td>-0.0</td></tr><tr><td>diamonds</td><td>-0.37</td><td>[-0.85, 0.44]</td><td>-1.5%</td><td>4.1%</td><td>0.949</td><td>0.997</td><td>-0.012</td><td>-8.5</td><td>+11.0</td><td>+33.3</td><td>+8.2</td><td>-12.2</td></tr><tr><td>video_transcoding</td><td>-0.27</td><td>[-0.70, 0.58]</td><td>53.3%</td><td>40.1%</td><td>0.987</td><td>0.998</td><td>-0.019</td><td>-14.2</td><td>+35.9</td><td>+31.2</td><td>+54.2</td><td>+39.5</td></tr><tr><td>medical_charges</td><td>-0.71</td><td>[-0.88, -0.38]</td><td>3.7%</td><td>1.8%</td><td>0.321</td><td>0.661</td><td>-0.006</td><td>-3.3</td><td>+0.4</td><td>+0.4</td><td>+4.9</td><td>+0.4</td></tr><tr><td>black_friday</td><td>0.19</td><td>[-0.32, 0.53]</td><td>8.2%</td><td>4.1%</td><td>0.694</td><td>0.959</td><td>-0.054</td><td>-18.5</td><td>+28.9</td><td>+0.6</td><td>+3.7</td><td>-0.4</td></tr><tr><td>particulate-matter</td><td>-0.77</td><td>[-0.85, -0.63]</td><td>5.7%</td><td>3.6%</td><td>0.147</td><td>0.337</td><td>-0.004</td><td>-3.6</td><td>-0.6</td><td>-0.1</td><td>+3.7</td><td>+0.7</td></tr><tr><td>nyc-taxi-green</td><td>-0.05</td><td>[-0.26, 0.11]</td><td>34.0%</td><td>22.0%</td><td>0.785</td><td>0.959</td><td>-0.028</td><td>-10.5</td><td>+10.5</td><td>+2.1</td><td>+12.9</td><td>+2.0</td></tr></table>

![](images/09f400805605290cc8e865d6335c53d0f9e840392d311e9d0e47a04dd7c819bf.jpg)  
Figure 6: The two-budget frontier, Concrete. Each dot is one dual-gate configuration, colored by the joint acceptance rate it actually realized; the orange line is the single-gate model traced over its acceptance grid. Down and to the right is better (lower selective error, higher selective interval coverage). The dual-gate cloud sits below and to the right of the single-gate curve at matched acceptance — the trade-off a single scalar gate cannot express. Per-dataset versions for all 18 tabular datasets are in Figure 13.  
When does the learned gate beat a width threshold? Descriptive, not inferential, at this sample size.

![](images/b14ce6483924cac90dc775752971b5162c7beb29c224d3cfee86fade81e3ce51.jpg)

![](images/5f90131d7a5da099caa0407e417a99f5bf94917e36a8e0a13c5079a3220e9327.jpg)

![](images/bab8bb5ea358fb271a60e83017c4598530cb616ba21af0563071721626790c00.jpg)  
Figure 7: When joint selection helps. Per-dataset selection efficiency of the learned gate against (left) the full-acceptance miscoverage rate, (middle) a heteroscedasticity statistic, (right) target excess kurtosis. Each point is one of 20 datasets (18 for the left panel, where the two image tasks lack a full-acceptance miscoverage figure); Spearman $\rho$ and its $p \textmd { - }$ value are in each panel title, with a least-squares line for orientation only. Only the heteroscedasticity relationship is distinguishable from noise at this sample size.

on the tail. The learned gate raises abstention by a median +0.193 against +0.046 for the width threshold, and reacts more strongly on 15 of 18 datasets; on diamonds, black\_friday and nyc-taxigreen the gate abstains on 73–96% of the tail while the width threshold's rate moves by less than 0.3, sometimes in the wrong direction. On two datasets (naval, particulate-matter) both selectors abstain less on the tail, i.e. the chosen covariate isolates an easier region there. Because the tail is constructed from a held-out covariate rather than injected, the protocol does not let us claim that the advantage grows with shift severity, and it does not cover the image tasks.

## L FULL ABLATION RESULTS

The three ablations of Table 13 were run on all 18 tabular datasets; per-dataset values are in Table 12 and Figures 14–16.

Table 13: Ablations: change in selective RMSE at $\phi = 0 . 5$ vs. the full recipe (positive: ablation is worse); 18 tabular datasets. W/B: datasets worse/better.
<table><tr><td>Variant</td><td></td><td>Med. W/B Range (%)</td><td></td></tr><tr><td>no auxiliary heads</td><td> $+ 2 . 0 \%$ </td><td>14/4</td><td> $- 3 7 , + 4 1$ </td></tr><tr><td>frozen IPIV, post-hoc gate</td><td> $+ 5 . 6 \%$ </td><td>15/3</td><td> $- 2 0 , + 5 4$ </td></tr><tr><td> $\sqrt { n }$  for  $\sqrt { S _ { g } }$ </td><td> $+ 0 . 2 \%$ </td><td>9/9</td><td> $- 1 2 , + 3 9$ </td></tr></table>

![](images/4d21eeadeaa2ff9dde33e7a6befcf6a53bc51729e520c18d3be9c6b044b063e2.jpg)

yacht — C2 Ablation: End-to-End vs Frozen-Gate  
![](images/074d9c52bb3256024a30f50ab528efbb67f76f4717ddeecaf8114173605bc913.jpg)

![](images/b670de2ddbc18bf376d67dec2e34052dfddc28d9abf17bcd84f0dfeed491ee65.jpg)  
Figure 8: Ablations on Yacht, the dataset where the effects are largest: end-to-end SelPI against a gate trained on a frozen IPIV model, and against removing the auxiliary head. Selective PICP, MPIW and RMSE against target acceptance rate; shaded bands are ±1 s.d. over 5 runs. End-toend training dominates on both interval width and selective error across the whole grid. Equivalent figures exist for all 18 tabular datasets.

## M PER-DATASET FIGURES

![](images/c692c46c675b1b50956270e2d65f746fe1fc4bc2ef1c20762deedffbf9a5f60e.jpg)  
Figure 9: Selection efficiency per dataset (single-gate $\mathrm { { S e l P I } _ { 1 G } ) } \mathrm { { : } }$ the share of the gap between the same-predictor width threshold $( \mathrm { S E } = 0 )$ and the oracle $( \mathrm { S E } = 1 )$ that the learned gate closes, on selective RMSE and on the selective interval score. The gate beats the width threshold on every dataset on both metrics, and by more on the interval score on all 20. The two image tasks are illustrative (§5); SE is scale-free, so their real-unit widths do not affect comparability.

Each grid shows the same plot for all 18 tabular datasets, in the order used throughout the appendix: the nine UCI sets, then the nine larger ones

![](images/c0c8784b047715e06cd8738acb85950aefe1a4c1948f599c34936b3ca4d2bbcc.jpg)

![](images/0592b58d7c9d42ceed518879a1efdb07be22789c4d90b17794e225c2b0dc5679.jpg)

![](images/963fa34b47a3fa890fc4eda481246c8b997dd5964b58cf452832b084113a7915.jpg)

![](images/f9071cd37a2261fde7b831b8a33ee03779b97509ad0e59fa323d3827b2f86d7e.jpg)

![](images/d97db6fe170b98a320dff0284b0b3bf55f837a78f58d03aeb4b534840dfb39e4.jpg)

![](images/190f4ab5d2923cde58e6d8b41ae1c2a7d77770dab4bae0ee09c3cea284614c7e.jpg)

![](images/f5973c4bb7c32e9e38dbdf10f9d944d5a8eaf0dc4fe86e2929b2002ba418d34c.jpg)

![](images/e9bf7b10c6fc85669e2952ab05e496228af3bfb6c8f13fdd11f51490c2498b5a.jpg)

![](images/1ed8e1ee3935f491a408c0f6303aed0aff0d59127c5881db4c755a568c0f47a3.jpg)

![](images/e4c01e34e9b9391ce41945afec302e1946606278384f6625d7db9e3ccafc0e03.jpg)

![](images/62712ccec16e0db32aeb3d3ff9ed3ca254e80d71a6a57a48e8b91535e4bcb00c.jpg)

![](images/7953c238af005f5c7ac94c2039abdf23aad1893c197cba035361f0ead6eb5268.jpg)

![](images/62f85bee50a576941ec9c7772932be36584cf235988dada7003c7f4e9a458552.jpg)

![](images/07b74ec97ab45d34f9bb65ead794b85a4f9af57688ac11c6abc41f2634cd8d70.jpg)

![](images/f915a778aa25d6832a945284c207d4c19fbd670407c545123623a23d71a2438e.jpg)

![](images/bef6bf17e7ac4899d7d01b546d80d76bed1e0a91cc2c7ebd08bb86f70cd12d7d.jpg)

![](images/2e199aa54a0f4205d09ae1a0a7acbf3472912bc2e4d265222374937b9691812e.jpg)

![](images/4ae3435cfb0032e588cbf5b0984421cf7d93df342d8dfbc559befdf8b96d1f33.jpg)

![](images/0c1c82c8bbe13785cf6188d9158fd60790dbfa23d4e6319b68b6004e59aa644e.jpg)

![](images/f98003d33185e4d140a2cb007acdb117dde8491ca9a6a9ee43762403ee2a44c3.jpg)

![](images/1d1c900dbb423143b23b7bbf3798d375a734add3c330239017a7fea9ba42e5b7.jpg)

![](images/62968b1bb6fea63dc9a450d51901080c3f2213e3724fd0ef69f42c4a13dc99bc.jpg)

![](images/571d5fd06e48218865b3efb2b8804f19430db607f44fbba9fc6227d10fc7c3ba.jpg)

![](images/dfe25641e0e5d505c658b9432c562da691a5241145f0264d944c527122d703be.jpg)

![](images/53ba06e5d26b6f3d698ae37f4580394de44fa615a959c587cf40f6e71af2c92d.jpg)

![](images/c61797fb929d6fbd68c9be2df0ce534ec2d657db2017bb64910d1970c914953a.jpg)

![](images/3147f73ea852f012ff6b6f71785a4aa14bbb1225a85f9e5cc333fdcef6a813e7.jpg)

![](images/1bb001c7cdabe60b9d574e443bbaac02492c937dea42e4202070eac0411dfe0b.jpg)

![](images/d915c73fb0423b350e8dc00cde4e3e1449d3a86cc472d2a3ca264534541ace87.jpg)

![](images/bb2d3215365571eb9a9660855628b63c5b08adf5109dca6796115c693e1960c8.jpg)

![](images/9801c3aabfafcd12bac7657487217c02dc5748afdae8612d3958ca4e586da010.jpg)

![](images/2f1b505bba37fbf253e4d40b860116cac0ca2383e1fe274a8f6db731c6219958.jpg)

![](images/9498fe35439694b5396cad3ca96e66f1c9fc565d7ee1beb0d3aa459547fdfad4.jpg)

![](images/98321423d606988411c21634659f5e205d69eb69c0bb4c00b70ecfd5dd71acc2.jpg)

![](images/db1d0024e4c747777acc41fb265ed84f341d11c96ddf3ea5afd34a877b4e3cb2.jpg)

![](images/2744d4d5cb7f5d6f6a5705fb2f84b2dba2877c3d90f1770e75a930f0353c5465.jpg)

![](images/66d2d41d245b7fbd8ad411994feebe56e2d8e37173d22dd5c886cacf1297a5ae.jpg)

![](images/66dfeea38166a70745ea1581a48709234db39370ebccedd38124fe06d41d476d.jpg)

![](images/3c7be3b190e38fde5d7f61e8085e0ffda6847031db7d042e1a007e166bf81092.jpg)

![](images/fad67786d229cead9a9b31659062a346f3f139a4209c5040af3e3d354c526809.jpg)

![](images/adb99d94d5d800233ca47e102784ad38a6ba25715056dac8c3a974db1247e5ad.jpg)

![](images/a698259a277427cc3e763eb0b6fa76f619fb7b8644ee84b2cd9ba25237df7a97.jpg)

![](images/7e73af9c6060623be63bad4054f01958758acfe3511cdc89e7021df3c19a557b.jpg)

![](images/4e7cdc1863c9968f8c640c0b35420e774ad5387327a2683b8e8ff46801dfa1dd.jpg)

![](images/e78196b633b506910584fc4be1e0d5385726c29b99a36e59d2a5819485aa993a.jpg)

![](images/131f50686a705792342756149415334a233178ac460169f4ffd40c6cd299b7bb.jpg)

![](images/2ac58d02d5519a4417cd7ddcf48585870c69b4e2f3c6d61364a1cbf4aad1c8df.jpg)

![](images/b00be62d7455fd2172e486fb3fb521dfc7077eddfea62771984e52c171b17b5b.jpg)

![](images/46826661605ec5fa2cacd9bcbb1bc06fa0e4c4f82d37b1c3e0d38e89f572523a.jpg)

![](images/6203e1164f101b745177b5b915c80df5f8ee260403aaffcb829b1f1e81082bfb.jpg)

![](images/7a81275cb3c0dd96d7fa19b6fc303e95322eca03239088187a617bb6c0edb071.jpg)

![](images/333321aa869af6b5779b7cded020bb3efeea91f8d226c975e4e9218a1c72574c.jpg)

![](images/b1453fd007601e858f4e28b11bdba77ab37b7c6d1a687cdfc17994fff2223524.jpg)

![](images/e21a2f8de57ec844baffa7e3bf6775a6250aa2c35ead035aa2cd38cce0c36382.jpg)  
Figure 10: Risk-coverage curves: selective PICP, MPIW and RMSE against target acceptance rate, with plain IPIV at full acceptance as the hollow diamond. Bands are ±1 s.d. over runs.

![](images/32cf5ed6130c4ee621df0c3d4fd8f86c070a8a1a81e937c963498575d23cf1f6.jpg)

![](images/d24747a3e58218ad56d6728fa3e7e02dd7ccfb3841add911a43d9656d56fd879.jpg)

![](images/8d126a9baa9887e31e3b0a291ad01e8beeba13cd8a7c9aaae11f81de07cece8c.jpg)

![](images/d6ac2be7a0b87b823f290f5a3f8db65f47d81f912c1ff107ce9a26d9d703312f.jpg)

![](images/f70360a9982740b124f5c569977b7f4da81efc9f1c1f9be72fec21cd540efc9d.jpg)

![](images/743b95a2f6a34640d1eac2f07b58237ed42aef357df75d645bece7cc132be5fb.jpg)

![](images/773938bb49133a9cec7e7837cc68efca44413ef85fd5fa2fa0a1cdf0705beeb9.jpg)

![](images/3b16b21ae826c459387852e67dacc8697ee237a4f5db3d43187a358255a2fe37.jpg)

![](images/a251404343b1f97c96c7b4e4fb82151c05d78ceca36e4bb9488c6522d88b5ae2.jpg)

![](images/e44b6da64fa1b552144cfd1ae9a899052e7e55bd8655df7bea635fac05b2610f.jpg)

![](images/877132ded0e913092e7004504c7f0d0fab5f77a98ea394326c22b7ee0d268344.jpg)

![](images/f9425a4303591767e7872ff8ea76362346d15d67bd248957a863b36de34619ea.jpg)

![](images/756eaa0c7df7d7411f47a8844bb2ca7847c5bd1457764031dffebb2c974959b9.jpg)

![](images/4bd1b0a23d2aa1c4448717490c0929a2fb1eb90b0748932b072cd1af24ce2dcb.jpg)

![](images/9e4bc32dc7665303fad12e015fda727e2d25c2aef9e9b0d13d3827a94a750d4b.jpg)

![](images/6aa647ee06aa63dfde7860d81700cdef0c28ba2ba1c36b4261e8b661b06826e1.jpg)

![](images/0bfde316e823a0be4a30f5d424dd77f759b1b0c05b3590cd4f612cad4b4ed315.jpg)

![](images/084cfcccd196704c3726e4f142dd29cf5cc5da2c07a0650c91145448159eaac0.jpg)  
Figure 11: Selector comparison: the learned gate against IPIV + width threshold and against an oracle ranking by realized error. The persistent oracle gap is the headroom that remains on every dataset.

![](images/3d521167a2fa2596c66fe3361be1de3611aa3543474885e75bc2fe58efb1b488.jpg)  
Figure 12: Dual gate vs. SelectiveNet-regression: point accuracy, then the interval reliability and sharpness SelectiveNet cannot produce at all. Shading between best and median trial is the tuning advantage of §5.

![](images/1d928c67c2517ce3f2161ddc92ec4dc9ae3e432c6cfd50e286dc8be684bc1d7f.jpg)

![](images/8993621c8a3b02565f338d2c5ca5ab06563bda0b5e6133e33f3a7a52b6d53258.jpg)

![](images/3a5ac91cde5442f1fee422de1328480f7b4bbd8c9e3ce1cad2a33049318e83fd.jpg)

![](images/cce131bb0d1db0d68ea895b41155ae21ffa726daa749569764bfc498a059eb41.jpg)

![](images/412393be45241e0b7fa8cc46367ef97641f6cc87fbb1e443327090e8431ef6ad.jpg)

![](images/247e3e28528ecd9e20379d144d76069742ee13a72d6849cd944f7945db260f6b.jpg)

![](images/72dc87a5bfdba74737f619ee9c707d0e6051017ea8bf9ad6c4fb19b659335b0a.jpg)

![](images/0f52efa3c437b3e5dd0411f61c7e925366495d69d54229918b3716e664ac021b.jpg)

![](images/c96d97e9a2b1e1860ef41c53e11edf9b87642278df69f6a67315da5020a8f728.jpg)

![](images/c1a45915e04a77f27fa26e889cdd83628d8d4def6a14edee20646d138c24644a.jpg)

![](images/db7d6092ece6139ed31ad41c43604124746f43138030e79a0e26add56692f394.jpg)

![](images/e54b3fe802514338d87973b5ed6160e3e0cd1cf1c4aaf37e7be0d032803b250e.jpg)

![](images/d5ac4a0a921e3b3ccc3da54aecdc2acfad0c15183bd0bb35a18409f5bbdad273.jpg)

![](images/bcad2ce005633e127d90ec31fb5f2e77db226edaa6f67b03ab08af5da9e0b452.jpg)

![](images/c1ac8d1be38cf71658db7d5494a60329c74c4030cb8cb63fc301d06913eebea7.jpg)

![](images/e2e2fd821bf79d3a88f8de783fe3e692b343d171fa8a733273bacae026f992d1.jpg)

![](images/7c85fae483c5533c684c5525b97697992fa2be6e0be5b3a222e539c523496606.jpg)

![](images/f55047bc7210b3bec8254670af7b7e9ca40a7dd9ff625014da994fe80d22338d.jpg)  
Figure 13: Two-budget frontier: every dual-gate configuration, colored by realized joint acceptance rate, against the single-gate curve. Down and to the right is better.

![](images/bec652617be602a9e6e7f8e64335c6249d772d792c3a2aaa8def739fa1f98d47.jpg)

![](images/134d2708c4c8bb93a49a5ba651d77cc842c6d1397b1103a0bc81d9b79bf6f0bf.jpg)

![](images/8b66cc182112d896a5baf37f013ea020d4ea5e10c39e2fa7dce73519b8db41ef.jpg)

![](images/49711c65771c8c7748371f8f98950bab7010a53b3f375da0ec1a0cf028a188c6.jpg)

![](images/9da2ca9ca35f233187edd20cbd18530e026f114cb302c50cd3ea2249953dac8a.jpg)

![](images/05f0da62916b2f1daae1a1cc48cc17b87acac8eb09dbb9ee52893dee20f86670.jpg)

![](images/c4af0d63628d3a78f8119b89489bfa8d6522b1bffd3a96b0f8e2f1e1ab29791b.jpg)

![](images/f600d4ef2495f5791ea798645fe213be918f9368964056a889d7206d244b9cc8.jpg)

![](images/ef3492bef65487bf47948c930247f49d4d16b3679300be4718b2b1afdc975617.jpg)

![](images/dbf52d8a8fd156c937fc2f7b0b9d59b4e1db5684dd8576c319288a87e2154bae.jpg)

![](images/3261a3e728aad5cca691eb2082b9c7f3fe816cf6c230093f2dc5143ffffcd541.jpg)

![](images/d63b02d067bb1be28a5d0f02f0a3ce13501a63a6e9866ecdb55a969e4c109d7b.jpg)

![](images/fe7459f5272c6b141c5d82686989aead1d0f3aef626542336c2f575af80b6538.jpg)

![](images/517ddc72fd30163cedbdb42b23517556a12282226b4569ef65487e1ee27fc09d.jpg)

![](images/12a5c0a3aa6e98a89f3a29066929a0d4a8aeac243d16190c6b0eefeee45d5c5a.jpg)

![](images/bd706c4718a8ce914ff9523cd61925b090bb1bd2d946f00a388a21b788efd472.jpg)

![](images/4a37e14626368a7888c5f9c5344917aa3916bd88f4ed66121e93621d139b1050.jpg)

![](images/487500ad1c4d3673d29a91259c7f8015fdb4f1bfd67158039f8e7bb725d25c68.jpg)

![](images/e1e81b40d3de90fa467b4692c867f348542780dc2a79e94ca4a39b7511133dca.jpg)

![](images/14317d846f0b51f52844c80ca855f106b3e5df48c1893360a168838c1e1ae513.jpg)

![](images/e9c4c8187525c867eef8c4391785ccb244d51410c3f6aac20f42860e74afd7e3.jpg)

![](images/7fbd46b504b342054fe31aef1731a7a86daeb13149907142785140976f51cba5.jpg)

![](images/5e9c03ec80ecdcd95ed4c8e3187ca29911d24d6fae675afa5dc56b4ff4408946.jpg)

![](images/dbef97e39daa674dea20a006c630e0fb9c48664a5b9e1e0ce75eab4e7ada9697.jpg)

![](images/75ffe3ce5816c79de2a8eebbcb5b6dbea29af4ab9983b9d7cccc391f78e6f4bf.jpg)

![](images/e8b1347d9a8c0a374b53c276fc57c5fab5176d4910a8a0bd750ec6b7cd90459b.jpg)

![](images/5856272377b66ff946b787f5c221952a3cf4c9a6c3b63dd352d0159b356ba697.jpg)

![](images/246d0717f86b121be63306e5835f5ebe07ae0e0de83e5b9c0d37207c8fa83262.jpg)

![](images/f5081c9a46ca704931cd0000598f4c812cff39a213e93cd9fe4117ad1e6ed13c.jpg)

![](images/e646520afff10f292e4e730102ded8880270e7c814b2d4247e2607ea9d1b9507.jpg)

![](images/4703308dac0c5ea2b1274fe68d7c7b9eea1614a06aa1904ac3a976fd7c3978f9.jpg)

![](images/f4c8a0357e101da8ca7facdb5effef19a19d942b00603a6f9133f56a23f79146.jpg)

![](images/4a777f1659991cbfea55470507bc70e51238453dd8d308c21f0db9c79a6165bf.jpg)

![](images/426c4adc4938cc0f6e5b9429cfcd2397dd8836dce1ed13ef0738a671ce346ce9.jpg)

![](images/bfe71539792af7d636066befe6db7743bfc23933acd73246c2e8a6ff3a7455c3.jpg)

![](images/b001f77dfc8105b1be4b0ff210e1a95c9d7b73a50c4abc1389bb95a3eaf742a9.jpg)

![](images/2957748cfe3fec34de9b2f7a0f06e456c4930b6e9c11350202bab655cfe78d57.jpg)

![](images/459ce2f029507c7a3ff083968ced14542b39ae4c0f69c4203469006d6e67e3c8.jpg)

![](images/864f68e5edd4fe30bd143cf5052077ee890114a6b23e478f163fc218eef2eef0.jpg)

![](images/259a6303cb684a364bd3aefede3f933b26a08c3a759e8c295ea12a4571155c7b.jpg)

![](images/ba06e181bedbccecc208eb12e33bc82755f643ab14aed2f0e5172c3083d7267a.jpg)

![](images/be3d53452b50020d67d12de26815e71279c4ae69aceda694db37c5decdfd7c7c.jpg)

![](images/b8e3a8ddaf53300532233e8e19c7f255da0f79294170ea34d08e8fccc5bd4552.jpg)

![](images/4ac6cda2f5052ece3a83aca52113e7bcd8767b3bccd7a9867bcad40d6e923bb1.jpg)

![](images/462f19a87e208107b832d44e08e569341de9b2f249747133c7b5bcf7995bda46.jpg)

![](images/b3048121fd5e96325d043ef88c35739f8ea99e6b8789127faba50c0241acd64c.jpg)

![](images/7546cad6110a376fe424536f2d3b4e1ae61399779e00a4e33adf37a1cc1e1e90.jpg)

![](images/419de2a06a7b2d50be791c025086eabbe4bc8a427b042c6536b6076f7a235782.jpg)

![](images/f0f3fae62e5933314eca8ecf192564f80e5545a187c27e261ac6591081d296e4.jpg)

![](images/45b17269017a18fe8f6a6fbba7daf0e53785f70bd186c7ddf807efce830dd304.jpg)

![](images/b1a3dc30bfc906d7dbcf8966ccf1e74a8b5490295835bbcc2d97ce8c2988061e.jpg)

![](images/1dfbf2cab09b2e5de1028231b3aee0432bd5ac646ac78f73bfd66ddb19c95fdb.jpg)

![](images/fbf892bd68f1e160995490e23ad79a440479e9ce7decdb3efe196f87c9327d72.jpg)

![](images/28b55174a3b5008fb01127c0fc5db26fc6fa44b68b7f6424717537c8177670fa.jpg)  
Figure 14: Ablation: end-to-end vs. frozen-gate training, with the no-auxiliary-head variant overlaid. Frozen-gate is worse on 15 of 18 datasets (Table 13). This overlay also contains the noauxiliary-head comparison shown on its own in Figure 15 below.

![](images/8ddb4349f2df0fad4e3f1e1b95b5e0dffefc4ae399738a0a0d5071675330324c.jpg)  
Figure 15: Ablation: with vs. without the auxiliary head, isolated into its own panel (also visible as part of the overlay in Figure 14).

## N FULL-ACCEPTANCE BASELINES FROM THE LITERATURE

Our evaluation compares selectors, so it needs baselines that abstain. Neither parent paper does: IPIV and its comparators answer every query. Their published numbers are nonetheless useful in two ways, and we reproduce them here rather than leave the impression that no wider comparison exists. First, they place our reproduction of IPIV — the base predictor every SelPI number is built on — against the numbers its authors report, on the same datasets and protocol. Second, they show where the interval-quality field stood at full acceptance before a reject option is added.

Our IPIV reproduction tracks the published numbers closely on the tabular sets and is somewhat worse on the two image tasks, where we train a single model rather than an ensemble. Our own Ens. column closes most of that gap: on IMDB-WIKI it improves repro's MAE from 7.68 to 7.42, and on bone age from 20.71 to 19.46, tracking the published DE/IPIV figures more closely once the base predictor is itself an ensemble rather than a single model. It still trails the published DE figure on IMDB-WIKI (6.66), which is expected: DE is a different architecture (a Gaussian-NLL ensemble) evaluated over 20 splits, ours is an IPIV ensemble over 5. On the tabular sets the Ens. MPIW column tracks the published IPIV column almost exactly, as it should, since the published IPIV figures are themselves five-model ensembles. This Ens. predictor is also the one behind the deep-ensemble selector baseline of §6–§7, which we compare directly against SelPI's learned gates.

![](images/3a485b99b8d34d69ddb76cf12311823ebd52e0d002c8a7e38112be747945e438.jpg)  
Figure 16: Ablation: $\sqrt { S _ { g } }$ VS. $\sqrt { n }$ penalty normalization. The curves track each other closely on most datasets, which is the visual form of the null result in Table 13.

## O DEEP-ENSEMBLE SELECTOR: PER-DATASET RESULTS

Table 15 gives the per-dataset values behind the selection-efficiency columns of Table 3: the selection efficiency of the learned gate and of the two disagreement selectors on both metrics, each against its own predictor's width threshold and oracle, and the two failure-mode AUROCs of the disagreement signals.

Table 14: Full-acceptance (φ = 1) results as published in the IPIV paper, with our own reproduction of IPIV alongside (the repro columns — this is plain IPIV retrained in our environment, not SelPI). QD, QD+ and DE are that paper's own baselines. Ens. is our own M=5-model IPIV ensemble (§5), the base predictor behind the deep-ensemble selector of §6. Published figures use 20 splits and 5-model ensembles; ours use 5 splits (and, for Ens., 5 members per split), so small differences are expected and the comparison is one of fidelity, not of ranking. RMSE/MAE are on the original target scale; MPIW is in standardized units.
<table><tr><td></td><td colspan="6">RMSE</td><td colspan="6">MPIW</td></tr><tr><td>Dataset</td><td>DE</td><td>QD</td><td>QD+</td><td>IPIV</td><td>repro</td><td>Ens.</td><td>DE</td><td>QD</td><td>QD+</td><td>IPIV</td><td>repro</td><td>Ens.</td></tr><tr><td>Boston</td><td>2.87</td><td>3.39</td><td>6.12</td><td>3.13</td><td>2.64</td><td>2.70</td><td>0.87</td><td>1.15</td><td>4.98</td><td>1.09</td><td>0.88</td><td>1.09</td></tr><tr><td>Concrete</td><td>5.21</td><td>5.88</td><td>11.87</td><td>5.43</td><td>6.17</td><td>5.79</td><td>1.01</td><td>1.08</td><td>3.43</td><td>1.02</td><td>0.86</td><td>1.08</td></tr><tr><td>Energy</td><td>1.68</td><td>2.28</td><td>7.05</td><td>1.65</td><td>1.63</td><td>1.70</td><td>0.49</td><td>0.45</td><td>1.52</td><td>0.42</td><td>0.35</td><td>0.42</td></tr><tr><td>Kin8nm</td><td>0.08</td><td>0.08</td><td>0.19</td><td>0.07</td><td>0.08</td><td>0.08</td><td>1.14</td><td>1.18</td><td>2.64</td><td>1.10</td><td>0.88</td><td>1.10</td></tr><tr><td>Naval</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.31</td><td>0.27</td><td>2.85</td><td>0.24</td><td>0.21</td><td>0.27</td></tr><tr><td>Power</td><td>3.99</td><td>4.14</td><td>12.00</td><td>4.08</td><td>4.10</td><td>4.09</td><td>0.91</td><td>0.86</td><td>1.51</td><td>0.86</td><td>0.81</td><td>0.86</td></tr><tr><td>Protein</td><td>4.36</td><td>4.99</td><td>5.10</td><td>4.35</td><td>4.41</td><td>4.36</td><td>2.68</td><td>2.27</td><td>3.30</td><td>2.26</td><td>2.14</td><td>2.25</td></tr><tr><td>Wine</td><td>0.62</td><td>0.67</td><td>0.70</td><td>0.63</td><td>0.66</td><td>0.62</td><td>2.50</td><td>2.24</td><td>4.89</td><td>2.22</td><td>1.91</td><td>2.20</td></tr><tr><td>Yacht</td><td>1.38</td><td>1.10</td><td>8.19</td><td>0.98</td><td>1.04</td><td>0.95</td><td>0.33</td><td>0.18</td><td>1.25</td><td>0.17</td><td>0.20</td><td>0.28</td></tr><tr><td>Year-MSD</td><td>8.95</td><td>9.30</td><td>10.11</td><td>8.93</td><td>9.01</td><td>8.93</td><td>2.91</td><td>2.45</td><td>4.25</td><td>2.42</td><td>2.25</td><td>2.47</td></tr><tr><td>IMDB-WIKI (MAE)</td><td>6.66</td><td>10.23</td><td>10.08</td><td>7.03</td><td>7.68</td><td>7.42</td><td>2.61</td><td>3.47</td><td>5.01</td><td>2.87</td><td>2.84</td><td>3.28</td></tr><tr><td>Bone age (MAE)</td><td>18.69</td><td>20.24</td><td>25.18</td><td>18.13</td><td>20.71</td><td>19.46</td><td>2.17</td><td>1.99</td><td>5.24</td><td>2.09</td><td>2.19</td><td>3.38</td></tr></table>

Table 15: Per-dataset selection efficiency (SE) on selective RMSE and on the interval score for the learned gate $\mathrm { ( S e l P I _ { 1 G } ) }$ and for the two deep-ensemble disagreement selectors, and AUROC of each disagreement signal for coverage failure (cov.) and value failure (val.). Each SE is computed against its own predictor's width threshold and oracle; “"marks an undefined SE (width and oracle AURC coincide).
<table><tr><td rowspan="2">Dataset</td><td colspan="3">SE, selective RMSE</td><td colspan="3">SE, interval score</td><td colspan="2">AUROC, point disagr.</td><td colspan="2">AUROC, bound disagr.</td></tr><tr><td>gate</td><td>point</td><td>bound</td><td>gate</td><td>point</td><td>bound</td><td>COV.</td><td>val.</td><td>COV.</td><td>val.</td></tr><tr><td>yacht</td><td>0.833</td><td>-0.961</td><td>-1.062</td><td>0.868</td><td></td><td></td><td>0.761</td><td>0.735</td><td>0.840</td><td>0.747</td></tr><tr><td>boston</td><td>0.456</td><td>0.108</td><td>-0.001</td><td>0.766</td><td>0.225</td><td>0.170</td><td>0.603</td><td>0.575</td><td>0.509</td><td>0.572</td></tr><tr><td>energy</td><td>0.732</td><td>-1.456</td><td>-1.471</td><td>0.951</td><td>-1.060</td><td>-0.552</td><td>0.681</td><td>0.795</td><td>0.674</td><td>0.837</td></tr><tr><td>concrete</td><td>0.572</td><td>-0.331</td><td>-0.269</td><td>0.841</td><td>-0.403</td><td>-0.235</td><td>0.605</td><td>0.519</td><td>0.598</td><td>0.527</td></tr><tr><td>wine</td><td>0.423</td><td>-0.088</td><td>-0.158</td><td>0.795</td><td>0.050</td><td>0.047</td><td>0.548</td><td>0.577</td><td>0.583</td><td>0.494</td></tr><tr><td>kin8nm</td><td>0.71</td><td>-0.326</td><td>-0.308</td><td>0.884</td><td>-1.015</td><td>-0.887</td><td>0.486</td><td>0.533</td><td>0.494</td><td>0.537</td></tr><tr><td>power-plant</td><td>0.16</td><td>-0.114</td><td>-0.174</td><td>0.526</td><td>0.147</td><td>0.040</td><td>0.486</td><td>0.465</td><td>0.472</td><td>0.444</td></tr><tr><td>naval</td><td>0.891</td><td>-0.773</td><td>-0.803</td><td>0.972</td><td>-30.2</td><td>-12.7</td><td>0.913</td><td>0.741</td><td>0.958</td><td>0.726</td></tr><tr><td>protein</td><td>0.602</td><td>-0.131</td><td>-0.365</td><td>0.866</td><td>0.230</td><td>0.187</td><td>0.565</td><td>0.620</td><td>0.599</td><td>0.425</td></tr><tr><td>Year-MSD</td><td>0.567</td><td>-0.358</td><td>-0.371</td><td>0.799</td><td>-0.254</td><td>-0.334</td><td>0.546</td><td>0.638</td><td>0.523</td><td>0.626</td></tr><tr><td>superconduct</td><td>0.868</td><td>-0.624</td><td>-0.579</td><td>0.981</td><td>-0.858</td><td>-0.659</td><td>0.693</td><td>0.686</td><td>0.692</td><td>0.666</td></tr><tr><td>sarcos</td><td>0.543</td><td>-0.758</td><td>-0.741</td><td>0.876</td><td>-2.332</td><td>-2.077</td><td>0.555</td><td>0.592</td><td>0.554</td><td>0.591</td></tr><tr><td>diamonds</td><td>0.949</td><td>-0.849</td><td>-0.693</td><td>0.997</td><td>-1.635</td><td>-1.408</td><td>0.746</td><td>0.739</td><td>0.742</td><td>0.744</td></tr><tr><td>video_transcoding</td><td>0.987</td><td>-1.020</td><td>-0.618</td><td>0.998</td><td>-3.189</td><td>-1.632</td><td>0.865</td><td>0.787</td><td>0.876</td><td>0.790</td></tr><tr><td>medical_charges</td><td>0.321</td><td>-0.453</td><td>-0.423</td><td>0.661</td><td>-0.988</td><td>-0.946</td><td>0.508</td><td>0.523</td><td>0.505</td><td>0.544</td></tr><tr><td>black_friday</td><td>0.694</td><td>-0.001</td><td>0.001</td><td>0.959</td><td>0.006</td><td>-0.102</td><td>0.498</td><td>0.524</td><td>0.486</td><td>0.551</td></tr><tr><td>particulate-matter</td><td>0.147</td><td>-0.457</td><td>-0.453</td><td>0.337</td><td>-0.874</td><td>-0.854</td><td>0.523</td><td>0.516</td><td>0.518</td><td>0.509</td></tr><tr><td>nyc-taxi-green</td><td>0.785</td><td>0.264</td><td>0.286</td><td>0.959</td><td>0.571</td><td>0.676</td><td>0.669</td><td>0.633</td><td>0.751</td><td>0.635</td></tr><tr><td>bone age</td><td>0.338</td><td>-0.072</td><td>-0.271</td><td>0.542</td><td>-0.621</td><td>-1.030</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td>IMDB-WIKI</td><td>0.006</td><td>-0.167</td><td>-0.197</td><td>0.082</td><td>-0.396</td><td>-0.470</td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>median (tabular)</td><td>0.648</td><td>-0.34</td><td>-0.37</td><td>0.872</td><td>-0.62</td><td>-0.55</td><td>0.584</td><td>0.606</td><td>0.591</td><td>0.581</td></tr></table>

## P COST

SelPI adds exactly two things to a IPIV model: the gate heads and the auxiliary triple. For a tabular backbone with hidden width h, each gate costs 16h + 65 parameters (16h + 16 for the hidden layer, 32 for batch normalization, 17 for the output unit) and the auxiliary triple costs 3h + 3. On Concrete $( h = 5 0 , d = 8 )$ that takes a 603-parameter IPIV model to 1,621 for the single gate and 2,486 for the dual gate; on superconduct $\bar { ( h \ : = \ : 1 0 0 , d \ : = \ : 7 9 ) }$ it takes 8,303 to 11,936, a 44% increase. The relative cost falls as the backbone grows and is negligible for the image tasks: on the bone-age

VGG-16 stack each gate reads a 1024-unit representation for 16,449 parameters against a backbone of roughly 15M, so the two gates and the auxiliary triple together add about 0.2%.

Training cost is the more meaningful number, and we measure it directly rather than estimate it. Across the nine UCI datasets, training one SelPI model takes a median 1.63× the wall-clock of training one IPIV model on the same data and hardware (range 1.00 × on Yacht to 2.03× on Protein, the largest). The whole-pipeline cost is higher than this ratio suggests, because SelPI trains one model per acceptance rate: a 10-point curve costs roughly 16× a single IPIV fit.

Inference is where the architecture pays off. Both gates are scalar heads on the shared trunk, so an accept/reject decision, an interval and a point estimate all come from one forward pass — the same cost as plain IPIV, and unlike an ensemble or MC-dropout selector, which need $\bar { K }$ or $T$ passes for their uncertainty signal. The deep-ensemble selector of §5 is the concrete case: it costs five IPIV fits per split to train and five forward passes per query at inference, against one SelPI fit per acceptance rate and one pass per query; the comparison in Table 3 (right columns) is made at that fivefold inference cost.

## Q ADDITIONAL METRICS DISCUSSED IN THE LITERATURE

The coverage-width criterion of the LUBE/QD line combines coverage and width into one number, $\mathrm { C W C } = \bar { \mathrm { N M P I W } } \left( 1 + \gamma ( \mathrm { P I C P } ) e ^ { - \eta ( \mathrm { P I C P } - \mu ) } \right)$ , where NMPIW is the mean width divided by the target range, $\mu$ is the nominal level, and γ is 1 only when coverage falls below nominal. We computed it for completeness but do not headline it, for two reasons. It is not a proper scoring rule, so it cannot be used to rank forecasters without the ranking depending on $\eta ;$ and its exponential penalty makes it extremely sensitive near the nominal level, which shows plainly in our numbers — on wine, a drop in selective coverage from 0.838 to 0.791 moves CWC from 86 to 605, a sevenfold change from a five-point coverage difference. The interval score of equation 12 measures the same trade-off with a linear miss penalty and a decision-theoretic justification, which is why it is our headline.