# ORDERS: An Empirical Study of Norm-Rank Aggregation for Personalized Federated Learning

Koffka Khan

Department of Computing and Information Technology, Faculty of Science and Technology, The University of the West Indies, St. Augustine, Trinidad and Tobago

koffka.khan@sta.uwi.edu

## Abstract

Personalized federated learning combines shared representations with client-specific predictors, but the contribution of a server weighting rule can be obscured by local training and evaluation choices. We study ORDERS, a configuration that combines a shared backbone, a private residual adapter and classifier, geometric weights assigned by descending update norm, feature alignment, and private-parameter perturbations. The server computes a weighted sum of updates obtained from the same broadcast model; it does not obtain an additional optimization effect from sequential addition. A fully specified evaluation comprises 80 final runs: eight configurations, two datasets, and five training seeds on one fixed partition per dataset. On two-class-per-client CIFAR-10, ORDERS achieves 80.51 ± 0.79% native mean client accuracy, compared with 79.02 ± 1.42% for FedPer-R1 and 80.27 ± 0.73% for the matched uniform-weight control. After common local fine-tuning, the difference from FedPer-R1 narrows to 0.32 percentage points. On Sent140, ORDERS reaches 74.71 ± 0.49%, only 0.69 points above a post hoc client training-majority diagnostic. Ablations provide limited, endpoint-dependent evidence for norm ranking and alignment, and no clear benefit from perturbations. Parameter-payload savings are 5.47% and 0.78%, respectively. The study supplies auditable results and a precise account of the method, while showing that these records do not establish universal accuracy, fairness, communication-efficiency, or adversarial-robustness advantages.

Keywords: Personalized federated learning, Heterogeneous data, Weighted aggregation, Representation learning, Reproducible evaluation

## 1. Introduction

Federated learning coordinates local training while retaining training examples at clients. FedAvg combines local updates to produce a shared model [1]. When client distributions differ, the representation that benefits collaboration need not determine the best decision rule for each client. Personalized federated learning therefore permits client-specific predictors, including shared bodies with local prediction layers [2–4] and complete personal models coupled to a global model [5].

An additional design choice is how much influence each returned update receives. Sample-size or equal-client averaging supplies a reference; loss-based objectives, similarity-based collaboration, gradient-conflict modification, and learned aggregation offer alternatives [6–11]. These precedents mean that neither a shared/private split nor the combination of personalization and nonuniform aggregation is, by itself, a new principle. A useful empirical question is narrower: what does a particular weighting rule contribute when the architecture, local training procedure, and evaluation endpoint are made explicit?

ORDERS assigns decreasing geometric coefficients to shared-model updates ranked by descending Euclidean norm. The evaluated configuration also contains a private residual feature adapter, a local classifier, an alignment penalty, and occasional private-parameter perturbations. All participating clients begin their shared updates from the same broadcast state. Consequently, the aggregation operator is a normalized weighted sum. Its coefficients depend on rank, but permuting the additions of already weighted vectors creates no additional curvature or conflict-resolution mechanism.

This paper replaces earlier aggregate summaries with the recorded ORDERS-R1 evaluation. It reports eight configurations on CIFAR-10 and Sent140 over five final training seeds, including matched controls that remove rank weighting, alignment, or perturbations individually. We distinguish the native endpoint from a secondary endpoint with a common local fine-tuning budget. We also report client dispersion, lower-tail accuracy, parameter-payload accounting, and a simple local-majority diagnostic for the sentiment task. Predictions, run records, and an independent metric-recomputation script support the numerical tables. The contributions are threefold. First, we give an executable interpretation of the shared/private model and normrank operator, together with mathematical limits on claims about descent and reliability. Second, we provide a controlled, seed-paired empirical evaluation of the complete configuration and three component removals, with explicit baseline adaptations and tuning failures. Third, we show how endpoint choice, client label imbalance, and the actual parameter split change the interpretation of apparent personalization and communication gains. The evidence supports a modest CIFAR-10 benefit under the specified native evaluation, but much weaker conclusions for Sent140 and after common fine-tuning. No new architectural family, universal descent guarantee, or Byzantine defense is claimed.

## 2. Related work

## 2.1. Shared representations and personalized predictors

FedPer retains personalization layers locally while collaborating through base layers [2]. FedRep learns a shared representation and client heads using separate local optimization phases [3]. FedBABU trains the body with a fixed randomly initialized head during federation and adapts the head for personalization afterward [4]. These methods establish the architectural and training precedents for body/head separation. ORDERS-R1 retains a residual adapter as well as a classifier, but this implementation choice does not establish novelty for shared/private partitioning.

FedRoD separates generic and personalized prediction tasks, using a loss designed for heterogeneous label distributions for the generic component and local empirical risk for personalization [12]. FedProto communicates class prototypes and regularizes local training using aggregated prototypes [13]. Thus FedProto is closely related through representation collaboration, although its communication object differs from body parameters. FedProto’s 2021 preprint and 2022 AAAI publication are both prior work; the published reference is by Tan et al. We discuss both methods as important untested comparators, rather than equating all collaborative/local designs with the parameter split studied here.

Ditto trains complete personal models with regularization toward a global model [5]. MOON uses model-level contrastive representation learning to correct local training under heterogeneity [14]; its original formulation primarily targets a global model, rather than the persistent private adapter used here. FedAMP learns personalized collaboration through attentive message passing among clients [6]. These works already connect local objectives with cross-client information exchange. ORDERS differs operationally by using one shared server body and norm-derived scalar coefficients, with squared within-predictor feature alignment instead of contrastive model comparisons or client-specific attentive model mixtures. This distinction is a method specification, not evidence of superiority.

## 2.2. Weighting, fairness, and robust aggregation

FedFV detects gradient conflicts using cosine similarity and modifies gradient directions and magnitudes before aggregation [7]. TERM changes the influence of losses through exponential tilting [8]; q-FFL emphasizes performance allocation through a loss-dependent objective [15]. Agnostic federated learning optimizes against mixtures of client distributions [9], while superquantile aggregation targets the upper tail of client losses [10]. These mechanisms should not all be characterized as equivalent rank heuristics. In particular, norm ranking does not rank losses, identify disadvantaged clients, or minimize a specified tail-risk objective. FedLAW additionally learns aggregation weights and a global shrinking factor [11], whereas ORDERS uses fixed normalized geometric coefficients.

FedCAP combines update calibration, customized aggregation, anomaly detection, and personalization to address heterogeneity and Byzantine attacks [16]. FLTrust uses a small clean server dataset to obtain a reference direction, assigns trust scores from directional agreement, and normalizes update magnitudes [17]. ORDERS uses neither a trusted reference nor a malicious-client detector; its descending norm score can favor a scaled malicious update. We therefore separate personalized predictive performance from adversarial robustness. The present campaign includes neither method as a trained baseline and contains no poisoning experiment.

Table 1. Mechanistic position of ORDERS. Literature comparisons are qualitative; only the R1 adaptations specified in Table 3 were run.
<table><tr><td rowspan=1 colspan=1>Prior approach</td><td rowspan=1 colspan=1>Established mechanism</td><td rowspan=1 colspan=1>ORDERS-R1 distinction</td></tr><tr><td rowspan=1 colspan=1>FedPer, FedRep,FedBABU</td><td rowspan=1 colspan=1>Body/head separation; different local training andadaptation policies</td><td rowspan=1 colspan=1>Private residual adapter and head; norm-weightedbody aggregation</td></tr><tr><td rowspan=1 colspan=1>FedRoD, FedProto</td><td rowspan=1 colspan=1>Decoupled prediction tasks or prototype exchange</td><td rowspan=1 colspan=1>One adapted feature path; shared parameter updates</td></tr><tr><td rowspan=1 colspan=1>MOON, FedAMP, Ditto</td><td rowspan=1 colspan=1>Contrastive correction, attentive collaboration, orpersonal-model regularization</td><td rowspan=1 colspan=1>Squared residual-feature penalty with one sharedbody</td></tr><tr><td rowspan=1 colspan=1>FedFV, TERM, q-FFL,AFL, superquantile FL</td><td rowspan=1 colspan=1>Conflict modification or explicit loss/distributionobjectives</td><td rowspan=1 colspan=1>Displacement-norm ranking; no fairness objective</td></tr><tr><td rowspan=1 colspan=1>FedCAP, FLTrust</td><td rowspan=1 colspan=1>Calibrated personalized defense or trust-basednormalized aggregation</td><td rowspan=1 colspan=1>No detection, clipping, or trusted reference</td></tr></table>

## 3. ORDERS-R1 specification

## 3.1. Predictor and local objective

Client � has training set $D _ { i } , i = 1 , \ldots , N$ . Let $w _ { t } ^ { G }$ denote the shared body at the start of round �, and let $w _ { i , t } ^ { L } =$ $( A _ { i , t } , V _ { i , t } , b _ { i , t } )$ denote its retained private adapter and classifier. For a generic current local state, define

$$
h = \phi ^ { G } ( x ; w ^ { G } ) , \quad \phi _ { i } ^ { L } ( x ; w ^ { G } , w _ { i } ^ { L } ) = h + A _ { i } h , \quad f _ { i } ( x ) = V _ { i } \phi _ { i } ^ { L } ( x ) + b _ { i } .\tag{1}
$$

Both feature maps have dimension 64. The adapter $A _ { i } \in \mathbb { R } ^ { 6 4 \times 6 4 }$ has no bias and is initialized to zero. The classifier produces task logits. The local map depends on the shared body; there are not two independent encoders. We use “shared” and “private” to describe parameter ownership. Local retention alone does not give differential privacy or protection against inference from transmitted updates.

For minibatch $B \subset D _ { i }$ , the ORDERS training loss is

$$
\begin{array} { r } { \mathcal { L } _ { i } ( B ) = \frac { 1 } { | B | } { \sum _ { ( \boldsymbol { x } , \boldsymbol { y } ) \in B } } \left[ \ell _ { \mathrm { C E } } ( f _ { i } ( \boldsymbol { x } ) , \boldsymbol { y } ) + \frac { \beta } { 2 } \| \phi ^ { G } ( \boldsymbol { x } ) - \phi _ { i } ^ { L } ( \boldsymbol { x } ) \| _ { 2 } ^ { 2 } \right] , \quad \beta = 0 . 5 . } \end{array}\tag{2}
$$

The squared norm sums feature coordinates before averaging examples. Both feature paths receive gradients; neither is detached. Since their difference is $- A _ { i } h$ , the penalty is exactly $( \beta / 2 ) \parallel A _ { i } h \parallel ^ { 2 }$ . It constrains the residual representation used for personalization and also affects the body through ℎ. It can therefore discourage a useful local departure. The value 0.5 is fixed in this campaign, not selected by a sweep. The no-alignment control sets $\beta = 0 ;$ it does not establish an optimal positive value or an adaptive per-client rule.

Each selected client performs two local passes using SGD with momentum 0.9 and weight decay $1 0 ^ { - 4 }$ . At the end of each pass, an independent Bernoulli event with probability 0.1 adds independent Gaussian noise of coordinate standard deviation 0.001 to every private tensor, including the adapter, classifier weight, and classifier bias. Shared tensors are not perturbed. Noise and minibatch ordering use separate seeded streams. These perturbations are a training component, not a privacy mechanism or demonstrated robustness defense.

Table 2. One synchronous ORDERS-R1 round. Unselected private states persist unchanged.
<table><tr><td rowspan=1 colspan=1>Step</td><td rowspan=1 colspan=1>Operation</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Sample $m _ { t }$ clients without replacement and broadcast $w _ { t } ^ { G }$ </td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Each selected client combines $w _ { t } ^ { G }$ with its stored $w _ { i , t } ^ { L }$ . Reset its local SGD state and train for two passes with Eq. (2) and thespecified private perturbations.</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Retain the updated private tensors and return $\varDelta _ { i , t } .$ </td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Compute shared-update norms; sort by decreasing norm and then client identifier.</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Assign normalized geometric coefficients and apply Eq. (4).</td></tr></table>

## 3.2. Server operator

For selected clients $S _ { t } ,$ with $m _ { t } = \vert S _ { t } \vert$ , let $\widetilde { w } _ { i , t } ^ { G }$ be the locally trained body and define

$$
\begin{array} { r } { \varDelta _ { i , t } = \widetilde { w } _ { i , t } ^ { G } - w _ { t } ^ { G } , \qquad s _ { i , t } = \parallel \varDelta _ { i , t } \parallel _ { 2 } . } \end{array}\tag{3}
$$

The norm includes every shared-body coordinate. The server sorts descending norms, breaking ties by client identifier, to obtain $\pi _ { t }$ . With $0 < \omega \leq 1$ -，

$$
\begin{array} { r } { \alpha _ { j , t } = \frac { \omega ^ { j - 1 } } { \sum _ { k = 0 } ^ { m _ { t } - 1 } \omega ^ { k } } , \quad d _ { t } = \sum _ { j = 1 } ^ { m _ { t } } \alpha _ { j , t } \ : \varDelta _ { \pi _ { t } ( j ) , t } , \quad w _ { t + 1 } ^ { G } = w _ { t } ^ { G } + \eta _ { s } d _ { t } . } \end{array}\tag{4}
$$

The campaign fixes $\omega = 0 . 9$ and $\eta _ { s } = 1$ . At this server step size, the next body is a convex combination of returned bodies. At $\omega = 1$ , it is equal-client averaging of the same updates. This differs from sample-size weighting when clients have different training-set sizes.

Norm computation and aggregation require $O ( m _ { t } d _ { G } )$ work for $d _ { G }$ shared coordinates, plus $O ( m _ { t } \mathrm { l o g } m _ { t } )$ sorting. A norm reflects displacement, which also depends on the learning rate, local work, curvature, and parameterization. It does not identify update quality, alignment with a target objective, or client trustworthiness.

## 3.3. Schedule concentration and adversarial limitation

Even a fixed decay gives different concentration as participation changes. The top-to-bottom coefficient ratio is $\omega ^ { - ( m _ { t } - 1 ) }$ , and an effective number of weighted contributors is $n _ { \mathrm { e f f } } = 1 / \sum _ { j } \alpha _ { j , t } ^ { 2 }$ . With $\omega = 0 . 9$ , the tenclient CIFAR-10 rounds have top/bottom ratio 2.58 and $n _ { \mathrm { e f f } } = 9 . 1 8$ ; the twenty-client Sent140 rounds have ratio 7.40 and $n _ { \mathrm { e f f } } = 1 4 . 8 8$ . All coefficients remain positive, but lower-ranked legitimate clients can receive substantially less influence. These are algebraic properties of the schedule, not measured advantages or evidence that it is optimal. No automated schedule-selection mechanism is implemented.

A client allowed to submit an arbitrary vector can increase its norm to obtain a top rank. Once largest, an update �� for unit vector � contributes $\alpha _ { 1 , t } M v _ { \mathrm { i } }$ , which is unbounded as � grows when the other updates remain fixed. Its scalar coefficient is 1.54 times the uniform coefficient for ten participants and 2.28 times for twenty. These coefficient ratios are not attack-success measurements; they expose why norm ranking alone cannot control priority hijacking. The evaluated method does not clip updates or reject suspicious clients. A bounded or trust-based alternative would be a different algorithm requiring adaptive-attack evaluation.

## 4. Mathematical interpretation and limits

## 4.1. Weighted addition and a valid descent statement

For fixed updates and coefficients, an accumulator $z _ { 0 } = w _ { t } ^ { G } , z _ { k } = z _ { k - 1 } + \eta _ { s } q _ { \sigma ( k ) } \varDelta _ { \sigma ( k ) , i }$ <sub>�</sub> ends at Eq. (4) for every permutation �, where $q _ { \pi _ { t } ( j ) } = \alpha _ { j , t }$ . This follows by telescoping. Reassigning coefficients can change the

aggregate; rearranging the same weighted additions cannot, apart from floating-point rounding. The client gradients are not re-evaluated at intermediate accumulator states.

The following standard smoothness statement describes what can be asserted for a fixed differentiable objective. It is not a claim that the personalized training process minimizes that objective.

Proposition 1 (Conditional comparison of aggregate steps). Let � have an �-Lipschitz gradient on a convex region containing the relevant segments, let � and � be fixed directions, and let $\eta _ { s } > 0$ . Then

$$
\begin{array} { r } { F ( w + \eta _ { s } d ) - F ( w ) \leq \eta _ { s } \langle \nabla F ( w ) , d \rangle + \frac { L \eta _ { s } ^ { 2 } } { 2 } \parallel d \parallel ^ { 2 } , } \end{array}\tag{5}
$$

$$
\begin{array} { r } { F ( w + \eta _ { s } d ) - F ( w + \eta _ { s } u ) \leq \eta _ { s } \langle \nabla F ( w + \eta _ { s } u ) , d - u \rangle + \frac { L \eta _ { s } ^ { 2 } } { 2 } \parallel d - u \parallel ^ { 2 } . } \end{array}\tag{6}
$$

Thus a strictly negative right-hand side of Eq. (6) is sufficient for a strictly lower objective value from d than from u.

The proof appears in Appendix A. With $\begin{array} { r } { u = m _ { t } ^ { - 1 } \sum _ { i } \varDelta _ { i , t } } \end{array}$ , the condition involves the direction of $d - u$ relative to the objective gradient, not just pairwise update conflicts. Neither that gradient nor a certified smoothness constant was measured in the campaign, so this condition supplies no numerical prediction of the observed accuracy differences. Accuracy differences are not objective-descent estimates.

## 4.2. Why conflict does not imply an advantage

Consider $\begin{array} { r } { F ( w ) = \frac { 1 } { 2 } \parallel w \parallel ^ { 2 } } \end{array}$ and two updates $\varDelta _ { + } = - w + v , \ A _ { - } = - w - v$ with $\parallel \boldsymbol { v } \parallel > \parallel \boldsymbol { w }$ ∥. Their inner product is negative. Uniform averaging with $\eta _ { s } = 1$ reaches zero, the exact minimizer. Assigning coefficients � and $1 - \theta$ instead yields

$$
w ^ { + } = ( 2 \theta - 1 ) v , \quad F ( w ^ { + } ) - F ( 0 ) = 1 / 2 ( 2 \theta - 1 ) ^ { 2 } \parallel v \parallel ^ { 2 } > 0 \quad ( \theta \neq 1 / 2 ) .\tag{7}
$$

This refutes universal superiority of nonuniform weighting in conflicting-update settings. There need be no norm tie: in one dimension, $w = \varepsilon$ and $v = 2 \varepsilon$ give updates � and −3�. Descending-norm geometric weighting incurs loss $2 [ ( 1 - \omega ) / ( 1 + \omega ) ] ^ { 2 } \varepsilon ^ { 2 }$ , while uniform averaging attains zero. The example persists for arbitrarily small �.

The earlier curvature–conflict claim is therefore withdrawn, rather than repaired by a sign change. A term $- c C$ with $c > 0$ and $C < 0$ is positive and cannot be interpreted as additional descent in an upper bound on loss change. Furthermore, positive semidefiniteness does not permit a signed cross-term bound $x ^ { \top } H y \leq \lambda _ { \operatorname* { m i n } } x ^ { \top } y$ Appendix A gives an explicit example and a generic bias-aware inequality. We claim no unconditional convergence rate for ORDERS: norm-derived coefficients depend on the updates, and the private states evolve, so unbiased mean-objective gradients do not follow from unbiased individual client estimators.

## 5. Experimental protocol

## 5.1. Data and partitions

The recorded protocol is ORDERS-R1-2026-09-29-v1. All methods use the same data partition, generated with seed 1729, and final training seeds 101, 202, 303, 404, and 505. These seeds vary training randomness, not the partition. The study contains 80 final training runs $( 8 \times 2 \times 5 )$ , each with 100 rounds; every run has two evaluation endpoints.

CIFAR-10 [18] is divided among 50 clients, each with exactly two classes. Five random label cycles allocate balanced class ownership. The official training/test boundary is preserved, and a stratified 10% of each client’s official training examples is used for validation. Each client has 900 training, 100 validation, and 200 test examples, giving totals of 45,000, 5,000, and 10,000. Images use fixed channel normalization without random augmentation.

Sent140 uses the original source training tweets and user identifiers, following the user-client motivation of LEAF [19], but it uses a fresh split rather than the historical LEAF split. A seeded random order selects eligible users; exact tweet identifiers and normalized text are deduplicated globally over candidate users before final eligibility checks. There are 100 retained users with at least 50 deduplicated tweets and a cap of 1,000 per user. The candidate deduplication removes 420 records. Within-user stratified splitting yields 5,758 training, 1,245 validation, and 1,245 test examples from 8,248 tweets. Test sets contain 7–51 examples per client. A vocabulary capped at 8,000 entries is learned from training examples only; sequences are truncated to 40 tokens and use no pretrained embeddings. Example identifiers are disjoint across the recorded clients and splits. Appendix B details preprocessing and source hashes.

## 5.2. Models and optimization

All methods use the same complete predictor form in Eq. (1). The CIFAR body has four $3 \times 3$ convolutions with channels 32, 32, 64, and 64, ReLU activations, and max-pooling after each pair. Adaptive $2 \times 2$ pooling and a shared 256 → 64 projection produce the feature vector. Sent140 uses a 64-dimensional embedding and a one-layer LSTM with hidden dimension 64; packed sequences supply the final hidden state. Neither model uses batch normalization or dropout. In split methods, the body is shared and the adapter and classifier remain local. The total float32 parameter counts are 86,762 for CIFAR-10 and 541,634 for Sent140, of which 82,016 and 537,408 are shared, respectively.

Each round samples 20% of clients without replacement: ten for CIFAR-10 and twenty for Sent140. Selection and initialization are paired across methods within a seed. The local budget is two full passes through participating clients’ training data. SGD uses batch size 64, momentum 0.9, and weight decay $1 0 ^ { - 4 } ;$ ; optimizer state is reset at each local phase and round. Table 3 specifies how each method allocates those passes. Equal data passes do not imply equal FLOPs, trainable-parameter work, memory, or final adaptation cost.

Ditto-R1 regularizes all personal parameters toward the pre-round broadcast global model with $( \lambda / 2 ) \Sigma _ { k }$ ∥ $w _ { i , k } - w _ { t , k } \parallel ^ { 2 }$ , where $\lambda = 0 . 1$ . Its global and personal passes partition the two-pass budget. FedRep-R1 uses only one private pass per body pass, rather than exploring larger head-training budgets. These choices, the shared residual-adapter architecture, and the limited tuning grid constrain comparisons with the published methods.

## 5.3. Tuning and evaluation endpoints

Each of the five complete methods is tuned over learning rates {0.01, 0.03, 0.1} using 20 rounds and seed 9001. Selection uses final-round native mean client validation accuracy, with a smaller-learning-rate tie break among finite candidates. All complete methods select 0.01 on CIFAR-10 and 0.1 on Sent140. The three ORDERS controls inherit the selected ORDERS learning rate so that they measure fixed-configuration removals. Other hyperparameters are fixed; they are not described as optimal. Four CIFAR-10 candidates at learning rate 0.1 stop with nonfinite losses (FedAvg-FT, FedPer, Ditto, and ORDERS); 26 candidates finish. All failures and finite results are retained in Appendix C. All 80 final runs finish. Final tests use round 100 after tuning is locked, with no test-based checkpoint selection.

At the native endpoint, split methods combine the final server body with each client’s stored private tensors. Ditto uses its persistent complete personal model. FedAvg-FT uses two train-only, full-model fine-tuning passes from the final global state. Thus native performance includes each method’s stated personalization policy. At the common FT endpoint, two task-loss-only full-model passes are applied to each corresponding pre-adaptation predictor. FedAvg-FT reuses its already fine-tuned state and is not adapted twice. The adaptation learning rate remains the selected method/dataset rate; momentum is reset. Common FT equalizes the additional adaptation budget and loss, but does not erase differences in preceding local training.

Table 3. Implemented R1 configurations. “Private” includes both adapter and classifier. All baseline labels retain R1 because these are specified adaptations, not reproductions of the original publications’ benchmark settings.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Local procedure per participation</td><td rowspan=1 colspan=1>Server aggregation</td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT-R1</td><td rowspan=1 colspan=1>Two full-model passes; two final train-only full-model fine-tuningpasses per client</td><td rowspan=1 colspan=1>Sample-weighted complete models</td></tr><tr><td rowspan=1 colspan=1>FedPer-R1</td><td rowspan=1 colspan=1>Two joint body/private passes</td><td rowspan=1 colspan=1>Sample-weighted bodies</td></tr><tr><td rowspan=1 colspan=1>FedRep-R1</td><td rowspan=1 colspan=1>One private pass with body frozen, then one body pass with privateparameters frozen</td><td rowspan=1 colspan=1>Equal-client bodies</td></tr><tr><td rowspan=1 colspan=1>Ditto-R1</td><td rowspan=1 colspan=1>One global-model pass; one persistent personal-model pass withparameter penalty</td><td rowspan=1 colspan=1>Sample-weighted complete globalmodels</td></tr><tr><td rowspan=1 colspan=1>ORDERS-R1</td><td rowspan=1 colspan=1>Two joint passes, β = 0.5 alignment, private perturbations</td><td rowspan=1 colspan=1>Norm-rank bodies, ω = 0.9</td></tr><tr><td rowspan=1 colspan=1>ORDERS-uniform-R1</td><td rowspan=1 colspan=1>Same as ORDERS</td><td rowspan=1 colspan=1>Equal-client bodies</td></tr><tr><td rowspan=1 colspan=1>ORDERS-no-align-R1</td><td rowspan=1 colspan=1>Same as ORDERS except β = 0</td><td rowspan=1 colspan=1>Norm-rank bodies</td></tr><tr><td rowspan=1 colspan=1>ORDERS-no-mutation-R1</td><td rowspan=1 colspan=1>Same as ORDERS except perturbation probability zero</td><td rowspan=1 colspan=1>Norm-rank bodies</td></tr></table>

## 5.4. Metrics and statistical scope

For run �, let $a _ { i , r }$ be client test accuracy in percent. The primary score is the equal-client mean $\begin{array} { r } { A _ { r } = N ^ { - 1 } \sum _ { i } } \end{array}$ $a _ { i , r }$ . Pooled accuracy weights clients by test size and is reported separately in the evidence files. We summarize $A _ { r }$ by its five-seed mean and sample standard deviation (SD, denominator four). Pairing uses corresponding training seeds on the same split. For ORDERS-minus-comparator differences $d _ { r } .$ , descriptive 95% intervals are

$$
{ \overline { { d } } } \pm t _ { 4 , 0 . 9 7 5 } s _ { d } / { \sqrt { 5 } } .\tag{8}
$$

These intervals are unadjusted, have only five paired observations, and describe training variability conditional on one partition. They do not cover partition or client-population uncertainty. Clients are not treated as independent training repetitions.

For transparency, Appendix D also reports exploratory paired t-test probabilities with a Holm correction across all 28 ORDERS-minus-comparator accuracy comparisons (7 comparators, 2 datasets, 2 endpoints). This family was chosen during analysis, not preregistered. The correction is a sensitivity check, not a basis for selecting favorable comparisons. We emphasize effect sizes, endpoint dependence, and uncertainty rather than declaring a universal winner.

We measure client disparity using the population SD within each run, $[ N ^ { - 1 } \textstyle \sum _ { i } ( a _ { i , r } - A _ { r } ) ^ { 2 } ] ^ { 1 / 2 }$ , and the mean accuracy of the lowest ⌈0.1�⌉ clients, then average these measures over seeds. They describe performance dispersion and lower-tail outcomes, not demographic fairness or a constrained optimization guarantee. Communication is accounted for as transmitted parameter bytes, as specified in Section 6.4.

## 6. Results

## 6.1. Accuracy and endpoint dependence

Table 4 contains every final configuration and both endpoints; Figure 1 shows the complete-method comparisons. At the native CIFAR-10 endpoint, ORDERS attains $8 0 . 5 1 \pm 0 . 7 9 \%$ . Its paired difference from FedPer-R1, the strongest native baseline in this set, is 1.494 pp with unadjusted 95% interval [0.294, 2.694]. Differences from FedAvg-FT-R1, FedRep-R1, and Ditto-R1 are 3.616, 4.188, and 7.404 pp, respectively. These are comparisons between complete R1 procedures. They do not attribute all differences to rank weighting.

Common fine-tuning raises the CIFAR-10 means for ORDERS and FedPer-R1 to 81.16% and 80.84%. Their paired gap narrows to 0.318 pp, with interval [−0.142, 0.778]. Consequently, the native gap from FedPer should not be described as a stable advantage under all personalization policies. FedRep-R1 and Ditto-R1 remain lower in this campaign, but their restricted local phase budgets and hyperparameter settings limit extrapolation to optimally tuned implementations.

Sent140 provides weaker differentiation. ORDERS native accuracy is $7 4 . 7 1 \pm 0 . 4 9 \%$ , compared with $7 4 . 6 4 \pm$ 0.33% for FedRep-R1 and $7 4 . 5 5 \pm 0 . 3 0 \%$ for FedPer-R1. The paired ORDERS-minus-FedRep difference is 0.065 pp with interval [−0.449, 0.580]. Under common FT, ORDERS falls to 74.35%, whereas FedRep-R1 reaches 74.73%; their difference is −0.375 pp with interval [−1.207, 0.458]. All Sent140 common-FT comparison intervals include zero. More local training does not uniformly improve this endpoint.

Across the exploratory family of 28 paired comparisons, only CIFAR-10 comparisons with FedAvg-FT-R1, FedRep-R1, and Ditto-R1 retain Holm-adjusted $p < 0 . 0 5$ at both endpoints. Comparisons with FedPer-R1, all component comparisons, and all Sent140 comparisons do not. This does not establish equivalence for the latter comparisons; it limits the strength of superiority claims from this small campaign.

Table 4. Final mean client test accuracy (%), mean ± sample SD across five training seeds. Both datasets use one fixed partition. All rows are R1 configurations; the suffix is omitted here for space. The two endpoint definitions are in Section 5.3.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>CIFAR-10 Native</td><td rowspan=1 colspan=1>CIFAR-10 CommonFT</td><td rowspan=1 colspan=1>Sent140 Native</td><td rowspan=1 colspan=1>Sent140 CommonFT</td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1> $7 6 . 9 0 \pm 0 . 4 8$ </td><td rowspan=1 colspan=1> $7 6 . 9 0 \pm 0 . 4 8$ </td><td rowspan=1 colspan=1> $7 3 . 8 4 \pm 0 . 4 3$ </td><td rowspan=1 colspan=1> $7 3 . 8 4 \pm 0 . 4 3$ </td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1> $7 9 . 0 2 \pm 1 . 4 2$ </td><td rowspan=1 colspan=1> $8 0 . 8 4 \pm 0 . 6 4$ </td><td rowspan=1 colspan=1> $7 4 . 5 5 \pm 0 . 3 0$ </td><td rowspan=1 colspan=1> $7 4 . 3 5 \pm 0 . 2 8$ </td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1> $7 6 . 3 3 \pm 0 . 9 8$ </td><td rowspan=1 colspan=1> $7 5 . 6 5 \pm 1 . 6 7$ </td><td rowspan=1 colspan=1> $7 4 . 6 4 \pm 0 . 3 3$ </td><td rowspan=1 colspan=1> $7 4 . 7 3 \pm 0 . 3 8$ </td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1> $7 3 . 1 1 \pm 1 . 2 5$ </td><td rowspan=1 colspan=1> $7 3 . 8 2 \pm 1 . 1 6$ </td><td rowspan=1 colspan=1> $7 4 . 1 9 \pm 0 . 2 0$ </td><td rowspan=1 colspan=1> $7 4 . 3 3 \pm 0 . 3 2$ </td></tr><tr><td rowspan=1 colspan=1>ORDERS</td><td rowspan=1 colspan=1> $8 0 . 5 1 \pm 0 . 7 9$ </td><td rowspan=1 colspan=1> $8 1 . 1 6 \pm 0 . 6 7$ </td><td rowspan=1 colspan=1> $7 4 . 7 1 \pm 0 . 4 9$ </td><td rowspan=1 colspan=1> $7 4 . 3 5 \pm 0 . 5 6$ </td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1> $8 0 . 2 7 \pm 0 . 7 3$ </td><td rowspan=1 colspan=1> $8 0 . 8 0 \pm 0 . 5 7$ </td><td rowspan=1 colspan=1> $7 4 . 5 7 \pm 0 . 4 1$ </td><td rowspan=1 colspan=1> $7 4 . 1 9 \pm 0 . 5 5$ </td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1> $7 8 . 9 1 \pm 1 . 5 9$ </td><td rowspan=1 colspan=1> $8 0 . 9 1 \pm 0 . 8 3$ </td><td rowspan=1 colspan=1> $7 4 . 6 2 \pm 0 . 4 5$ </td><td rowspan=1 colspan=1> $7 4 . 2 1 \pm 0 . 6 8$ </td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1> $8 0 . 6 0 \pm 0 . 8 1$ </td><td rowspan=1 colspan=1> $8 1 . 1 0 \pm 0 . 6 6$ </td><td rowspan=1 colspan=1> $7 4 . 7 0 \pm 0 . 5 3$ </td><td rowspan=1 colspan=1> $7 4 . 3 3 \pm 0 . 5 8$ </td></tr></table>

![](images/a460fe7bb83e5e289a08622d02ebe63e17bc2581a0b44b53be5f664c049ad0c2.jpg)

![](images/f479ef5a4d77153eff8cc6f766480b186ae56d43a6df7599d28ae6a76a3aeec1.jpg)  
Figure 1. Complete-method accuracy at the two endpoints. Bars indicate one sample SD across five training seeds, not confidence intervals. The dotted Sent140 line is the post hoc per-client training-majority predictor evaluated on test data. Dataset panels use different horizontal scales.

## 6.2. Component ablations

Table 5 gives paired effects for each single-component removal; Table 4 gives their absolute accuracies. Positive effects favor the complete configuration. On native CIFAR-10, replacing uniform weights by the norm-rank rule gives a mean difference of 0.246 pp, with interval [0.062, 0.430]. Removing alignment yields a larger but less precise gap of 1.606 pp, with interval [0.115, 3.097]. Both unadjusted intervals exclude zero, but neither comparison survives the exploratory correction. After common FT, these differences become 0.354 pp and 0.244 pp, with intervals including zero.

Private perturbations have no clearly established benefit. On native CIFAR-10, the no-mutation control reaches 80.60%, slightly above ORDERS at 80.51%; the paired complete-minus-control difference is −0.086 pp with interval [−0.235, 0.063]. Sent140 component effects are all small and all intervals include zero at both endpoints. The records therefore do not justify presenting all three components as necessary for improved performance. The controls hold the ORDERS learning rate fixed; they estimate removal effects at that configuration, not each variant’s best attainable performance. Interactions among removed components are not tested.

Table 5. Ablation effects: ORDERS minus the named control, in pp. Intervals are unadjusted paired 95% t intervals across five seeds. The final column applies the exploratory 28-comparison Holm correction to paired tests. All rows use the corresponding R1 configurations.
<table><tr><td rowspan=1 colspan=1>Endpoint</td><td rowspan=1 colspan=1>Control</td><td rowspan=1 colspan=1>Mean effect</td><td rowspan=1 colspan=1>95% interval</td><td rowspan=1 colspan=1>Holm p</td></tr><tr><td rowspan=1 colspan=1>CIFAR-10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>Native</td><td rowspan=1 colspan=1>Uniform</td><td rowspan=1 colspan=1>0.246</td><td rowspan=1 colspan=1>[0.062, 0.430]</td><td rowspan=1 colspan=1>0.453</td></tr><tr><td rowspan=1 colspan=1>Native</td><td rowspan=1 colspan=1>No alignment</td><td rowspan=1 colspan=1>1.606</td><td rowspan=1 colspan=1>[0.115, 3.097]</td><td rowspan=1 colspan=1>0.765</td></tr><tr><td rowspan=1 colspan=1>Native</td><td rowspan=1 colspan=1>No mutation</td><td rowspan=1 colspan=1>-0.086</td><td rowspan=1 colspan=1>[-0.235, 0.063]</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Common FT</td><td rowspan=1 colspan=1>Uniform</td><td rowspan=1 colspan=1>0.354</td><td rowspan=1 colspan=1>[-0.268, 0.976]</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Common FT</td><td rowspan=1 colspan=1>No alignment</td><td rowspan=1 colspan=1>0.244</td><td rowspan=1 colspan=1>[-0.330, 0.818]</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Common FT</td><td rowspan=1 colspan=1>No mutation</td><td rowspan=1 colspan=1>0.062</td><td rowspan=1 colspan=1>[-0.032, 0.156]</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Sent140</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Native</td><td rowspan=1 colspan=1>Uniform</td><td rowspan=1 colspan=1>0.139</td><td rowspan=1 colspan=1>[-0.122, 0.399]</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Native</td><td rowspan=1 colspan=1>No alignment</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>[-0.186, 0.362]</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Native</td><td rowspan=1 colspan=1>No mutation</td><td rowspan=1 colspan=1>0.015</td><td rowspan=1 colspan=1>[-0.105, 0.135]</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Common FT</td><td rowspan=1 colspan=1>Uniform</td><td rowspan=1 colspan=1>0.166</td><td rowspan=1 colspan=1>[-0.232, 0.564]</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Common FT</td><td rowspan=1 colspan=1>No alignment</td><td rowspan=1 colspan=1>0.139</td><td rowspan=1 colspan=1>[-0.207, 0.486]</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>Common FT</td><td rowspan=1 colspan=1>No mutation</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>[-0.039, 0.084]</td><td rowspan=1 colspan=1>1.000</td></tr></table>

![](images/4e816625f8e0217716b6f0ce322a788a63d5ffc15d59fe1f2e326bd0daca20db.jpg)  
Figure 2. Paired component effects from Table 5. Bars are unadjusted 95% intervals. Positive values favor complete ORDERS. Horizontal scales vary by panel to make the uncertainty visible; none of these comparisons has Holm-adjusted $p < 0 . 0 5$ in the exploratory family.

## 6.3. Client outcomes and the Sent140 majority diagnostic

Table 6 reports native client dispersion and bottom-decile accuracy. On CIFAR-10, ORDERS has lower mean client SD than the four complete baselines and a bottom-decile mean of 57.54%, compared with 56.54% for FedPer-R1 and 57.32% for uniform ORDERS. The paired bottom-decile differences are uncertain: 1.00 pp versus FedPer with interval [−1.10, 3.10], and 0.22 pp versus uniform with interval [−0.76, 1.20]. The nomutation control is slightly better on both displayed CIFAR-10 measures. On Sent140, FedRep-R1 has lower dispersion and higher bottom-decile accuracy than ORDERS. A universal fairness advantage is not supported.

To assess the role of user label imbalance, we constructed a post hoc diagnostic that always predicts each client’s most frequent training label, breaking ties toward the smaller label. This uses no validation or test labels to select the prediction. Its Sent140 mean client accuracy is 74.017%, while its pooled accuracy is 75.984%; the difference illustrates why these averaging conventions cannot be interchanged. ORDERS native mean client accuracy exceeds this diagnostic by only 0.693 pp. Across its five native runs, 61–67 of the 100 clients receive constant test predictions, and 90.20–91.89% of individual predictions agree with the client’s training-majority label. These counts do not prove that representations contain no useful information, but they substantially limit claims of broad language-modeling improvement from the mean accuracy alone. The same local-majority diagnostic is 50% on the balanced two-class CIFAR-10 clients. This diagnostic was not part of tuning and is not an additional stochastic training baseline.

Table 6. Native client-level outcomes, averaged over five seeds. Client SD uses population SD across clients within a run (lower is less dispersed); bottom 10% is the mean accuracy of the lowest five CIFAR-10 or ten Sent140 clients (higher is better). These are distinct from the seed SDs in Table 4.
<table><tr><td rowspan=1 colspan=1>Method (R1)</td><td rowspan=1 colspan=1>CIFAR-10 Client SD(pp)</td><td rowspan=1 colspan=1>CIFAR-10 Bottom10% (%)</td><td rowspan=1 colspan=1>Sent140 Client SD(pp)</td><td rowspan=1 colspan=1>Sent140 Bottom10% (%)</td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>13.80</td><td rowspan=1 colspan=1>53.04</td><td rowspan=1 colspan=1>17.16</td><td rowspan=1 colspan=1>41.07</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>12.98</td><td rowspan=1 colspan=1>56.54</td><td rowspan=1 colspan=1>16.84</td><td rowspan=1 colspan=1>42.03</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>13.67</td><td rowspan=1 colspan=1>53.78</td><td rowspan=1 colspan=1>16.36</td><td rowspan=1 colspan=1>42.91</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>12.72</td><td rowspan=1 colspan=1>51.28</td><td rowspan=1 colspan=1>16.82</td><td rowspan=1 colspan=1>41.39</td></tr><tr><td rowspan=1 colspan=1>ORDERS</td><td rowspan=1 colspan=1>12.34</td><td rowspan=1 colspan=1>57.54</td><td rowspan=1 colspan=1>16.68</td><td rowspan=1 colspan=1>42.36</td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1>12.47</td><td rowspan=1 colspan=1>57.32</td><td rowspan=1 colspan=1>16.63</td><td rowspan=1 colspan=1>42.27</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1>13.12</td><td rowspan=1 colspan=1>55.60</td><td rowspan=1 colspan=1>16.70</td><td rowspan=1 colspan=1>42.53</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1>12.32</td><td rowspan=1 colspan=1>57.72</td><td rowspan=1 colspan=1>16.70</td><td rowspan=1 colspan=1>42.32</td></tr></table>

## 6.4. Communication accounting

Let $B _ { G }$ be shared-body bytes, $B _ { L }$ private bytes, and $B = B _ { G } + B _ { L }$ . A participating split-method client uploads $B _ { G }$ bytes and downloads $B _ { G }$ bytes per training round. For � rounds with � participants each, the accounted payload is $2 T m B _ { G }$ , compared with 2��� for the full-model reference. Their fixed-budget ratio is

$$
\begin{array} { r } { r _ { \mathrm { p a y l o a d } } = \frac { B _ { G } } { B } , \qquad \mathrm { s a v i n g } = 1 0 0 ( 1 - r _ { \mathrm { p a y l o a d } } ) \% . } \end{array}\tag{9}
$$

Table 7 uses the actual float32 tensor inventories. Savings are 5.47% for CIFAR-10 and 0.78% for Sent140, because most parameters belong to the shared body. A two-part architecture does not imply that each part contains half the parameters.

The accounting excludes initialization, protocol messages, serialization, evaluation, and network headers. It is neither measured bandwidth nor measured latency. With equal rounds and participation, the per-round parameter ratio also equals the ratio over that fixed training horizon; no faster-convergence assumption is needed for this identity. Communication to a common target accuracy additionally depends on the required number of rounds and is not evaluated here. Final adaptation is local, and its compute is separate from the training-round payload. There is no $0 . 5 \times$ communication claim in this study.

Table 7. Parameter-payload accounting per seed over 100 training rounds, both directions. Full-model reference is FedAvg-FT-R1; FedPer-R1 and FedRep-R1 share ORDERS’ body-only payload. One MiB is $2 ^ { 2 0 }$ bytes.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1> $B _ { G }$ (bytes)</td><td rowspan=1 colspan=1>B (bytes)</td><td rowspan=1 colspan=1>Ratio</td><td rowspan=1 colspan=1>Saving (%)</td><td rowspan=1 colspan=1>Split (MiB)</td><td rowspan=1 colspan=1>Full (MiB)</td></tr><tr><td rowspan=1 colspan=1>CIFAR-10</td><td rowspan=1 colspan=1>328,064</td><td rowspan=1 colspan=1>347,048</td><td rowspan=1 colspan=1>0.9453</td><td rowspan=1 colspan=1>5.47</td><td rowspan=1 colspan=1>625.73</td><td rowspan=1 colspan=1>661.94</td></tr><tr><td rowspan=1 colspan=1>Sent140</td><td rowspan=1 colspan=1>2,149,632</td><td rowspan=1 colspan=1>2,166,536</td><td rowspan=1 colspan=1>0.9922</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>8200.20</td><td rowspan=1 colspan=1>8264.68</td></tr></table>

## 7. Discussion and limitations

The most defensible positive finding is the performance of the full configuration on the specified native CIFAR-10 task. The matched uniform control suggests that norm weighting has a small effect there, while alignment has a larger but less precise native removal effect. The reduced differences after common finetuning show why the final adaptation policy is part of the comparison. A gain over a baseline with a different local phase allocation is not a causal estimate of a server rule’s value.

The Sent140 results demonstrate a different limitation: personalized label priors can explain much of a high mean local accuracy. The majority diagnostic, small test sets, and limited differences among complete methods counsel against interpreting the two datasets as universal evidence across modalities. This study contains image and sentiment tasks only; it includes no speech evaluation.

Architecture and hyperparameter sensitivity remain unresolved. We evaluate one body/adapter/classifier split in a small CNN and an LSTM. No layer-selection mechanism, Transformer, GNN, split sweep, decay sweep, or positive-β sweep is present. Alignment explicitly limits local residual features; whether a client benefits from a larger departure should be assessed with training/validation data in a separate study. Similarly, the fixed schedule’s concentration grows with participant count. An adaptive weighting rule would need an objective, a selection protocol, and comparisons against learned or loss-aware aggregation; it cannot be inferred from this campaign.

The baseline set is incomplete. FedRoD, FedProto, FedBABU, FedAMP, MOON, FedFV, TERM, q-FFL, agnostic FL, superquantile FL, FedCAP, FLTrust, and FedLAW are discussed but not trained here. The present results establish no ordering relative to them. The R1 adaptations share an adapter architecture and a two-pass local budget; this supports an explicit comparison while differing from the original methods’ preferred architectures and tuning practices. Only learning rate is tuned, using one short tuning seed. Instability at the largest CIFAR-10 rate illustrates sensitivity rather than robustness.

Five training seeds on one partition support only limited uncertainty estimates. The study does not assess alternative client populations, population shifts, dropout, variable local work, adversarial updates, or labelnoise attacks. No global-only prediction endpoint was saved, so a global-to-local personalization gain cannot be computed. Private retention should not be confused with a formal privacy guarantee, nor norm weighting with robust aggregation. The observed fairness measures are descriptive outcomes rather than constraints enforced by the method.

Finally, the evidence audit establishes consistency of saved records and recomputation from saved predictions. It does not independently reacquire raw data or replay inference from model checkpoints. The exported records omit checkpoint tensors, although their hashes are indexed. These boundaries are explicit in the reproducibility statement and do not justify stronger claims about the unobserved training process.

## 8. Conclusion

ORDERS-R1 is a precisely specified combination of norm-rank-weighted body aggregation, persistent private adapters and heads, residual-feature alignment, and private perturbations. An 80-run evaluation finds a modest native CIFAR-10 advantage over the strongest included baseline, but a smaller difference after common finetuning and weak separation on Sent140. The measured rank-weighting effect is small, the alignment effect depends on the endpoint, and perturbations have no clear benefit. Actual shared-parameter sizes yield small payload savings. The contribution is an auditable empirical characterization of this configuration and its limitations, rather than a new shared/private architecture or a guarantee of conflict mitigation, fairness, or adversarial robustness.

## Data and code availability

The evidence package described in this study contains manuscript sources, verified CSV tables, the metricaudit script, the figure/table generator, and Evidence\_Records.zip. The latter contains configurations, environment metadata, dataset manifests and split assignments, captured source, round logs, and saved predictions. Raw images and tweets are not redistributed in that archive. Dataset source locations and fingerprints appear in Appendix B. The records exclude 860 model files totaling 3,325,439,029 bytes, so the package supports saved-prediction metric verification but not checkpoint inference replay. No public repository or permanent data DOI is asserted. Appendix F states the completed checks.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

None.

## Appendix A. Mathematical details

## Appendix A.1. Smoothness and comparison proof

For a vector � whose segment lies in the stated smoothness region,

$$
\begin{array} { c } { { \displaystyle F ( w + v ) - F ( w ) = \int _ { 0 } ^ { 1 } \langle \nabla F ( w + s v ) , v \rangle d s } } \\ { { \displaystyle \le \langle \nabla F ( w ) , v \rangle + \int _ { 0 } ^ { 1 } L s \parallel v \parallel ^ { 2 } d s = \langle \nabla F ( w ) , v \rangle + \frac { L } { 2 } \parallel v \parallel ^ { 2 } . } } \end{array}
$$

Substituting $v = \eta _ { s } d$ proves Eq. (5). Applying the same result from base point $w + \eta _ { s } u$ with displacement $\eta _ { s } ( d - u )$ proves Eq. (6). A negative upper bound in the latter establishes the stated strict comparison. A comparison of two unrelated upper bounds, by itself, would not establish an ordering between their left-hand sides.

For $\begin{array} { r } { d = \sum _ { i } q _ { i } \varDelta _ { i } } \end{array}$ , its quadratic form can contain pairwise inner products through

$$
\| \sum _ { i } q _ { i } \varDelta _ { i } \| ^ { 2 } = \sum _ { i } q _ { i } ^ { 2 } \parallel \varDelta _ { i } \parallel ^ { 2 } + 2 \sum _ { i < j } q _ { i } q _ { j } \langle \varDelta _ { i } , \varDelta _ { j } \rangle .
$$

This identity is part of the ordinary smoothness bound; it is not an extra sequential-path term. Nor does its sign determine the first-order term involving ∇�. For the positive definite matrix $H = \mathrm { d i a g } ( 1 , 3 )$ , vectors $x =$ $( 1 , 1 ) ^ { \top } , y = ( - 2 , 1 ) ^ { \top }$ satisfy $x ^ { \top } y = - 1$ and $x ^ { \top } H y = 1$ . Thus neither $\lambda _ { \operatorname* { m i n } } x ^ { \top } y$ nor $\lambda _ { \operatorname* { m a x } } x ^ { \top } y$ gives the asserted signed upper bound on a cross term. The valid bound $| x ^ { \top } H y | \leq \parallel H \parallel _ { \mathrm { o p } } \parallel x \parallel \parallel y$ ∥ does not preserve the sign of the Euclidean inner product.

## Appendix A.2. The quadratic counterexample and the withdrawn inequality

The scalar construction in Section 4.2 can be realized by one exact gradient step of size one on client losses $\begin{array} { r } { f _ { + } ( z ) = \frac { 1 } { 2 } ( z - v ) ^ { 2 } } \end{array}$ and $\begin{array} { r } { f _ { - } ( z ) = \frac { 1 } { 2 } ( z + v ) ^ { 2 } } \end{array}$ . Their mean is $\textstyle { \frac { 1 } { 2 } } z ^ { 2 } + { \frac { 1 } { 2 } } v ^ { 2 }$ , differing from $\begin{array} { r } { F ( z ) = \frac { 1 } { 2 } z ^ { 2 } } \end{array}$ only by a constant. Uniform averaging therefore reaches the minimizer of the mean objective. With $w = \varepsilon , v = 2 \varepsilon .$ descending norm assigns $1 / ( 1 + \omega )$ to −3� and $\omega / ( 1 + \omega )$ to �. The new state is $- 2 \varepsilon ( 1 - \omega ) / ( 1 + \omega )$ giving the loss stated in the main text.

The earlier inequality also fails under uniform aggregation. With $m = 2 , \eta _ { s } = \lambda _ { \mathrm { m i n } } = 1$ , uniform coefficients, and conflict sum $C = { \textstyle \frac { 1 } { 4 } } ( \varepsilon ) ( - 3 \varepsilon ) = - 3 \varepsilon ^ { 2 } / 4$ , its proposed right-hand side was

$$
- 1 / 2 | \varepsilon - 3 \varepsilon | ^ { 2 } - 1 / 2 C = - 1 3 \varepsilon ^ { 2 } / 8 .
$$

The exact loss change is $- \varepsilon ^ { 2 } / 2$ , and the quadratic has zero third-order remainder. The proposed upper bound is false. This calculation is a correction to the previous claim, not a new benchmark experiment. Equation (5) is exact for the uniform step in this example.

## Appendix A.3. What a bias-aware inequality would require

For completeness, fix private reference states $\overline { { w } } _ { i } ^ { L }$ and a lower-bounded, �-smooth objective $\begin{array} { r } { F ( w ) = N ^ { - 1 } \sum _ { i } } \end{array}$ $F _ { i } ( w , \overline { { w } } _ { i } ^ { L } )$ . Consider a hypothetical recursion $w _ { t + 1 } = w _ { t } - \eta h _ { t }$ , with deterministic $w _ { 0 } , 0 < \eta \leq 1 / L$ , finite relevant expectations, and history $\mathcal { H } _ { t }$ . Define

$$
\begin{array} { r } { \mu _ { t } = \mathbb { E } [ h _ { t } \mid \mathcal { H } _ { t } ] , \quad b _ { t } = \mu _ { t } - \nabla F ( w _ { t } ) , \quad v _ { t } = \mathbb { E } [ \parallel h _ { t } - \mu _ { t } \parallel ^ { 2 } \mid \mathcal { H } _ { t } ] . } \end{array}
$$

Smoothness and conditional expectation give

$$
\mathbb { E } [ F ( w _ { t + 1 } ) \mid \mathcal { H } _ { t } ] \le F ( w _ { t } ) - \eta \langle \nabla F ( w _ { t } ) , \mu _ { t } \rangle + \frac { L \eta ^ { 2 } } { 2 } ( \Vert \mu _ { t } \Vert ^ { 2 } + v _ { t } )
$$

$$
\leq F ( w _ { t } ) - \frac { \eta } { 2 } \parallel \nabla F ( w _ { t } ) \parallel ^ { 2 } + \frac { \eta } { 2 } \parallel b _ { t } \parallel ^ { 2 } + \frac { L \eta ^ { 2 } } { 2 } v _ { t } .
$$

The second line uses $- 2 \langle g , \mu \rangle = - \|  { g } \| ^ { 2 } - \|  { \mu } \| ^ { 2 } + \|  { g } - \mu \| ^ { 2 }$ and drops the nonpositive coefficient of ∥ $\mu _ { t } \parallel ^ { 2 }$ . Taking expectations, summing, and using $F \geq F _ { \mathrm { i n f } }$ yields

$$
\begin{array} { r } { \frac { 1 } { T } \sum _ { t < T } \mathbb { E } \parallel \nabla F ( w _ { t } ) \parallel ^ { 2 } \leq \frac { 2 ( F ( w _ { 0 } ) - F _ { \mathrm { i n f } } ) } { \eta T } + \frac { 1 } { T } \sum _ { t < T } \mathbb { E } \parallel b _ { t } \parallel ^ { 2 } + \frac { L \eta } { T } \sum _ { t < T } \mathbb { E } v _ { t } . } \end{array}\tag{A.1}
$$

This generic inequality retains the bias explicitly. Norm-dependent weights do not make $b _ { t }$ vanish, and the evolving private states in ORDERS do not satisfy the fixed-reference setup automatically. No bounds on these terms are established by the experiments. Equation (A.1) is therefore not an ORDERS convergence rate. Sorting update norms also provides no identity with a loss-ranked spectral-risk gradient.

## Appendix B. Implementation and provenance

## Appendix B.1. Preprocessing and initialization

CIFAR-10 inputs are converted to float32, divided by 255, and normalized by channel means (0.4914, 0.4822, 0.4465) and standard deviations (0.2470, 0.2435, 0.2616). Convolutions use stride one, padding one, and biases. Max-pooling is $2 \times 2$ . The final body projection has a bias and ReLU. The private classifier is linear with a bias. The only zero-initialized learned block is the residual adapter; remaining modules use PyTorch defaults under the paired seed.

The Sent140 loader reads training.1600000.processed.noemoticon.csv in Latin-1 and maps source labels 0 and 4 to binary labels 0 and 1. Candidate users are the first 400 in a seeded permutation of users with at least 50 source records; the first 100 retaining at least 50 records after deduplication are used. Deduplication compares original tweet IDs and lowercased, whitespace-normalized text. Tokenization lowercases text, replaces URLs with urltoken, and user mentions with usertoken; tokens comprise words with optional internal apostrophes, digit runs, and individual non-word non-whitespace characters. Vocabulary ties use lexical order after descending training frequency. Padding is index 0, unknown tokens index 1, and an empty sequence becomes one unknown token. The LSTM uses the final hidden state of the packed sequence. No raw tweet text is included in the evidence archive.

Within-client class lists are shuffled using deterministic derived seeds and split using the recorded stratified routine; the exact resulting memberships are authoritative in the archived manifests and split\_assignments.csv. Splitting is by example, not by label: the same class may legitimately appear in training, validation, and test sets. Participation, local minibatch order, perturbations, and final adaptation have deterministic derived seed streams. Nonparticipating clients keep their previous private or complete personal state. The campaign enables deterministic PyTorch operations and uses four CPU threads.

## Appendix B.2. Source identities and software

The recorded environment is Python 3.12.13 (conda-forge), PyTorch 2.6.0+cpu, NumPy 2.4.6, and SciPy 1.16.3 on Linux. The code identity recorded in environment.json is

$$
\mathtt { e d 2 6 f 2 9 4 c 1 6 1 e 7 b 3 6 c a 9 3 8 1 0 9 7 d d e 0 7 e d 5 1 3 8 4 2 1 d 8 f c 2 5 5 f e 1 9 2 5 2 1 1 c f i e 4 7 c a t b . }
$$

This is the runner’s normalized code-object hash, not the byte hash of the captured source file. The source and environment records are preserved; metric verification does not execute the captured training code.

The source archives and recorded SHA-256 values are:

• CIFAR-10: https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz;

6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce.

• Sent140: https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip;

004a3772c8a7ff9bbfeb875880f47f0679d93fc63e5cf9cff72d54a8a6162e57.

The CIFAR loader additionally checks the known source archive MD5. The recorded Sent140 SHA-256 identifies the acquired bytes; no independently supplied expected Sent140 SHA-256 was configured. The dataset-manifest fingerprints are, respectively,

451d9162dfe51d778bf23b88ddf0f2ac652854e030d3c7251140a27bda924de9

446ca84de394f67f82db188945d517a0f96966e298810aef93496ab949b39f06.

These fingerprints identify the recorded data construction; the audit does not independently establish raw-data authenticity.

## Appendix C. Validation selection and failures

Table C.8. Native validation accuracy (%) at tuning round 20, seed 9001. Each method chooses the best finite candidate. “Nonfinite” denotes an explicitly recorded failure, not an omitted observation. All three ORDERS controls inherit ORDERS’ choice.
<table><tr><td rowspan=1 colspan=1>Method (R1)</td><td rowspan=1 colspan=1>LR 0.01</td><td rowspan=1 colspan=1>LR 0.03</td><td rowspan=1 colspan=1>LR 0.1</td></tr><tr><td rowspan=1 colspan=1>CIFAR-10</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>66.760</td><td rowspan=1 colspan=1>50.100</td><td rowspan=1 colspan=1>Nonfinite</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>66.140</td><td rowspan=1 colspan=1>49.000</td><td rowspan=1 colspan=1>Nonfinite</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>58.500</td><td rowspan=1 colspan=1>49.000</td><td rowspan=1 colspan=1>49.000</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>53.980</td><td rowspan=1 colspan=1>52.320</td><td rowspan=1 colspan=1>Nonfinite</td></tr><tr><td rowspan=1 colspan=1>ORDERS</td><td rowspan=1 colspan=1>70.880</td><td rowspan=1 colspan=1>62.760</td><td rowspan=1 colspan=1>Nonfinite</td></tr><tr><td rowspan=1 colspan=1>Sent140</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>61.292</td><td rowspan=1 colspan=1>62.121</td><td rowspan=1 colspan=1>67.431</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>64.224</td><td rowspan=1 colspan=1>67.156</td><td rowspan=1 colspan=1>72.282</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>61.710</td><td rowspan=1 colspan=1>64.013</td><td rowspan=1 colspan=1>69.031</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>62.097</td><td rowspan=1 colspan=1>65.934</td><td rowspan=1 colspan=1>70.198</td></tr><tr><td rowspan=1 colspan=1>ORDERS</td><td rowspan=1 colspan=1>63.736</td><td rowspan=1 colspan=1>67.058</td><td rowspan=1 colspan=1>72.013</td></tr></table>

The finite-candidate selection leaves 26 completed evaluations and four failed candidates. The failures occur only for the listed CIFAR-10 candidates, not for final runs. The final test results are not substitutions for failed tuning outcomes. Selection on one short tuning seed does not identify a robust optimum over the larger hyperparameter space.

## Appendix D. Complete paired accuracy comparisons

Table D.9 includes all 28 comparisons. The paired differences use unrounded run scores; reported means and intervals are rounded only for display. A positive number favors ORDERS. Probabilities below 0.0001 are reported as < 0.0001; other probabilities are rounded to four decimals. The Holm family includes both endpoints and all comparators; no fairness or majority-diagnostic test is included in that family.

Table D.9. Paired ORDERS-minus-comparator accuracy effects (pp), five seeds.
<table><tr><td rowspan=1 colspan=1>Comparator (R1)</td><td rowspan=1 colspan=1>Mean</td><td rowspan=1 colspan=1>Unadjusted 95% CI</td><td rowspan=1 colspan=1>Rawp</td><td rowspan=1 colspan=1>Holm p</td></tr><tr><td rowspan=1 colspan=5>CIFAR-10 / Native</td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>3.616</td><td rowspan=1 colspan=1>[2.758, 4.474]</td><td rowspan=1 colspan=1>0.0003</td><td rowspan=1 colspan=1>0.0073</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>1.494</td><td rowspan=1 colspan=1>[0.294, 2.694]</td><td rowspan=1 colspan=1>0.0259</td><td rowspan=1 colspan=1>0.5435</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>4.188</td><td rowspan=1 colspan=1>[3.637, 4.739]</td><td rowspan=1 colspan=1>&lt; 0.0001</td><td rowspan=1 colspan=1>0.0008</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>7.404</td><td rowspan=1 colspan=1>[5.676, 9.132]</td><td rowspan=1 colspan=1>0.0003</td><td rowspan=1 colspan=1>0.0072</td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1>0.246</td><td rowspan=1 colspan=1>[0.062, 0.430]</td><td rowspan=1 colspan=1>0.0206</td><td rowspan=1 colspan=1>0.4526</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1>1.606</td><td rowspan=1 colspan=1>[0.115, 3.097]</td><td rowspan=1 colspan=1>0.0403</td><td rowspan=1 colspan=1>0.7654</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1>-0.086</td><td rowspan=1 colspan=1>[-0.235, 0.063]</td><td rowspan=1 colspan=1>0.1853</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>CIFAR-10 / Common FT</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>4.260</td><td rowspan=1 colspan=1>[3.668, 4.852]</td><td rowspan=1 colspan=1>&lt; 0.0001</td><td rowspan=1 colspan=1>0.0010</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>0.318</td><td rowspan=1 colspan=1>[-0.142, 0.778]</td><td rowspan=1 colspan=1>0.1271</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>5.512</td><td rowspan=1 colspan=1>[4.183, 6.841]</td><td rowspan=1 colspan=1>0.0003</td><td rowspan=1 colspan=1>0.0075</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>7.342</td><td rowspan=1 colspan=1>[6.383, 8.301]</td><td rowspan=1 colspan=1>&lt; 0.0001</td><td rowspan=1 colspan=1>0.0008</td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1>0.354</td><td rowspan=1 colspan=1>[-0.268, 0.976]</td><td rowspan=1 colspan=1>0.1895</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1>0.244</td><td rowspan=1 colspan=1>[-0.330, 0.818]</td><td rowspan=1 colspan=1>0.3035</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1>0.062</td><td rowspan=1 colspan=1>[-0.032, 0.156]</td><td rowspan=1 colspan=1>0.1407</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>Sent140 / Native</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>0.868</td><td rowspan=1 colspan=1>[0.131, 1.604]</td><td rowspan=1 colspan=1>0.0308</td><td rowspan=1 colspan=1>0.6151</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>0.159</td><td rowspan=1 colspan=1>[-0.224, 0.542]</td><td rowspan=1 colspan=1>0.3128</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>0.065</td><td rowspan=1 colspan=1>[-0.449, 0.580]</td><td rowspan=1 colspan=1>0.7423</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>0.519</td><td rowspan=1 colspan=1>[0.020, 1.017]</td><td rowspan=1 colspan=1>0.0445</td><td rowspan=1 colspan=1>0.8018</td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1>0.139</td><td rowspan=1 colspan=1>[-0.122, 0.399]</td><td rowspan=1 colspan=1>0.2136</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>[-0.186, 0.362]</td><td rowspan=1 colspan=1>0.4245</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1>0.015</td><td rowspan=1 colspan=1>[-0.105, 0.135]</td><td rowspan=1 colspan=1>0.7505</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>Sent140 / Common FT</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>0.510</td><td rowspan=1 colspan=1>[-0.021, 1.041]</td><td rowspan=1 colspan=1>0.0560</td><td rowspan=1 colspan=1>0.9514</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>0.004</td><td rowspan=1 colspan=1>[-0.596, 0.605]</td><td rowspan=1 colspan=1>0.9859</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>-0.375</td><td rowspan=1 colspan=1>[-1.207, 0.458]</td><td rowspan=1 colspan=1>0.2794</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>0.023</td><td rowspan=1 colspan=1>[-0.426, 0.472]</td><td rowspan=1 colspan=1>0.8924</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1>0.166</td><td rowspan=1 colspan=1>[-0.232, 0.564]</td><td rowspan=1 colspan=1>0.3120</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1>0.139</td><td rowspan=1 colspan=1>[-0.207, 0.486]</td><td rowspan=1 colspan=1>0.3259</td><td rowspan=1 colspan=1>1.0000</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>[-0.039, 0.084]</td><td rowspan=1 colspan=1>0.3739</td><td rowspan=1 colspan=1>1.0000</td></tr></table>

## Appendix E. Run-level accuracy

The five columns in Table E.10 are the independent training-seed replicates conditional on the fixed partition. Values for FedAvg-FT are identical across endpoints because the evaluated fine-tuned state is reused, not because two independent evaluations happened to agree. The evidence CSVs preserve full precision, pooled accuracy, client-level outcomes, and state hashes.

Table E.10. Mean client test accuracy (%) by training seed.
<table><tr><td rowspan=1 colspan=1>Method (R1)</td><td rowspan=1 colspan=1>101</td><td rowspan=1 colspan=1>202</td><td rowspan=1 colspan=1>303</td><td rowspan=1 colspan=1>404</td><td rowspan=1 colspan=1>505</td></tr><tr><td rowspan=1 colspan=1>CIFAR-10 / Native</td><td rowspan=1 colspan=5></td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>76.270</td><td rowspan=1 colspan=1>76.880</td><td rowspan=1 colspan=1>76.600</td><td rowspan=1 colspan=1>77.320</td><td rowspan=1 colspan=1>77.420</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>76.730</td><td rowspan=1 colspan=1>78.590</td><td rowspan=1 colspan=1>80.290</td><td rowspan=1 colspan=1>79.690</td><td rowspan=1 colspan=1>79.800</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>74.850</td><td rowspan=1 colspan=1>77.200</td><td rowspan=1 colspan=1>76.370</td><td rowspan=1 colspan=1>77.210</td><td rowspan=1 colspan=1>76.000</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>73.180</td><td rowspan=1 colspan=1>73.040</td><td rowspan=1 colspan=1>73.790</td><td rowspan=1 colspan=1>74.430</td><td rowspan=1 colspan=1>71.110</td></tr><tr><td rowspan=1 colspan=1>ORDERS</td><td rowspan=1 colspan=1>79.190</td><td rowspan=1 colspan=1>81.210</td><td rowspan=1 colspan=1>80.970</td><td rowspan=1 colspan=1>80.710</td><td rowspan=1 colspan=1>80.490</td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1>79.020</td><td rowspan=1 colspan=1>80.910</td><td rowspan=1 colspan=1>80.490</td><td rowspan=1 colspan=1>80.540</td><td rowspan=1 colspan=1>80.380</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1>76.850</td><td rowspan=1 colspan=1>78.620</td><td rowspan=1 colspan=1>81.150</td><td rowspan=1 colspan=1>78.340</td><td rowspan=1 colspan=1>79.580</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1>79.240</td><td rowspan=1 colspan=1>81.380</td><td rowspan=1 colspan=1>80.860</td><td rowspan=1 colspan=1>80.860</td><td rowspan=1 colspan=1>80.660</td></tr><tr><td rowspan=1 colspan=1>CIFAR-10 / Common FT</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>76.270</td><td rowspan=1 colspan=1>76.880</td><td rowspan=1 colspan=1>76.600</td><td rowspan=1 colspan=1>77.320</td><td rowspan=1 colspan=1>77.420</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>79.760</td><td rowspan=1 colspan=1>80.870</td><td rowspan=1 colspan=1>80.950</td><td rowspan=1 colspan=1>81.400</td><td rowspan=1 colspan=1>81.220</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>73.090</td><td rowspan=1 colspan=1>75.010</td><td rowspan=1 colspan=1>76.910</td><td rowspan=1 colspan=1>77.220</td><td rowspan=1 colspan=1>76.000</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>72.830</td><td rowspan=1 colspan=1>72.920</td><td rowspan=1 colspan=1>75.280</td><td rowspan=1 colspan=1>74.850</td><td rowspan=1 colspan=1>73.200</td></tr><tr><td rowspan=1 colspan=1>ORDERS</td><td rowspan=1 colspan=1>80.340</td><td rowspan=1 colspan=1>80.560</td><td rowspan=1 colspan=1>81.500</td><td rowspan=1 colspan=1>81.890</td><td rowspan=1 colspan=1>81.500</td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1>80.260</td><td rowspan=1 colspan=1>80.420</td><td rowspan=1 colspan=1>81.710</td><td rowspan=1 colspan=1>80.930</td><td rowspan=1 colspan=1>80.700</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1>79.580</td><td rowspan=1 colspan=1>80.870</td><td rowspan=1 colspan=1>80.880</td><td rowspan=1 colspan=1>81.610</td><td rowspan=1 colspan=1>81.630</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1>80.200</td><td rowspan=1 colspan=1>80.600</td><td rowspan=1 colspan=1>81.480</td><td rowspan=1 colspan=1>81.760</td><td rowspan=1 colspan=1>81.440</td></tr><tr><td rowspan=1 colspan=1>Sent140 /Native</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>73.372</td><td rowspan=1 colspan=1>73.609</td><td rowspan=1 colspan=1>73.682</td><td rowspan=1 colspan=1>74.441</td><td rowspan=1 colspan=1>74.108</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>74.069</td><td rowspan=1 colspan=1>74.769</td><td rowspan=1 colspan=1>74.733</td><td rowspan=1 colspan=1>74.728</td><td rowspan=1 colspan=1>74.455</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>74.670</td><td rowspan=1 colspan=1>74.847</td><td rowspan=1 colspan=1>74.319</td><td rowspan=1 colspan=1>75.073</td><td rowspan=1 colspan=1>74.315</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>74.280</td><td rowspan=1 colspan=1>74.103</td><td rowspan=1 colspan=1>74.376</td><td rowspan=1 colspan=1>74.314</td><td rowspan=1 colspan=1>73.884</td></tr><tr><td rowspan=1 colspan=1>ORDERS</td><td rowspan=1 colspan=1>74.300</td><td rowspan=1 colspan=1>75.067</td><td rowspan=1 colspan=1>74.998</td><td rowspan=1 colspan=1>75.112</td><td rowspan=1 colspan=1>74.072</td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1>74.126</td><td rowspan=1 colspan=1>74.819</td><td rowspan=1 colspan=1>75.051</td><td rowspan=1 colspan=1>74.701</td><td rowspan=1 colspan=1>74.161</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1>74.004</td><td rowspan=1 colspan=1>74.999</td><td rowspan=1 colspan=1>74.722</td><td rowspan=1 colspan=1>75.064</td><td rowspan=1 colspan=1>74.322</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1>74.300</td><td rowspan=1 colspan=1>74.956</td><td rowspan=1 colspan=1>75.110</td><td rowspan=1 colspan=1>75.150</td><td rowspan=1 colspan=1>73.961</td></tr><tr><td rowspan=1 colspan=1>Sent140 / Common FT</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>FedAvg-FT</td><td rowspan=1 colspan=1>73.372</td><td rowspan=1 colspan=1>73.609</td><td rowspan=1 colspan=1>73.682</td><td rowspan=1 colspan=1>74.441</td><td rowspan=1 colspan=1>74.108</td></tr><tr><td rowspan=1 colspan=1>FedPer</td><td rowspan=1 colspan=1>73.971</td><td rowspan=1 colspan=1>74.574</td><td rowspan=1 colspan=1>74.135</td><td rowspan=1 colspan=1>74.527</td><td rowspan=1 colspan=1>74.535</td></tr><tr><td rowspan=1 colspan=1>FedRep</td><td rowspan=1 colspan=1>74.745</td><td rowspan=1 colspan=1>75.378</td><td rowspan=1 colspan=1>74.538</td><td rowspan=1 colspan=1>74.487</td><td rowspan=1 colspan=1>74.488</td></tr><tr><td rowspan=1 colspan=1>Ditto</td><td rowspan=1 colspan=1>74.305</td><td rowspan=1 colspan=1>74.664</td><td rowspan=1 colspan=1>74.085</td><td rowspan=1 colspan=1>74.631</td><td rowspan=1 colspan=1>73.961</td></tr><tr><td rowspan=1 colspan=1>ORDERS</td><td rowspan=1 colspan=1>74.083</td><td rowspan=1 colspan=1>74.463</td><td rowspan=1 colspan=1>73.963</td><td rowspan=1 colspan=1>75.280</td><td rowspan=1 colspan=1>73.975</td></tr><tr><td rowspan=1 colspan=1>ORDERS: uniform</td><td rowspan=1 colspan=1>73.825</td><td rowspan=1 colspan=1>74.781</td><td rowspan=1 colspan=1>73.929</td><td rowspan=1 colspan=1>74.769</td><td rowspan=1 colspan=1>73.631</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no alignment</td><td rowspan=1 colspan=1>73.528</td><td rowspan=1 colspan=1>74.449</td><td rowspan=1 colspan=1>73.698</td><td rowspan=1 colspan=1>75.246</td><td rowspan=1 colspan=1>74.145</td></tr><tr><td rowspan=1 colspan=1>ORDERS: no mutation</td><td rowspan=1 colspan=1>74.083</td><td rowspan=1 colspan=1>74.463</td><td rowspan=1 colspan=1>73.852</td><td rowspan=1 colspan=1>75.280</td><td rowspan=1 colspan=1>73.975</td></tr></table>

# Appendix F. Evidence audit and claim changes

The supplied archive has SHA-256

b7e049a8fd3ea90ded4a0bbcd504364c46c076c1eeaf60b7699236ef7a431a69.

All ZIP CRC checks and 9,438 included-file checksums pass. The archive contains 9,439 members, 80 final run records, 8,000 final training-round records, and 160 final evaluation endpoints. Independent parsing recomputes metrics from 973,275 saved prediction records, including tuning and duplicated endpoint records; this count does not represent unique test examples. Recomputed summaries agree with the saved aggregate results. The audit checks recorded participant schedules, coefficient assignments, parameter-payload accounting, split consistency, and completed/failed tuning outcomes. It never executes the archived training source.

The record check is limited by the supplied archive. Its exclusion inventory lists 860 model files totaling 3,325,439,029 bytes. Model inference cannot be independently replayed from those absent files, and the audit did not reacquire raw images or tweets. Hash consistency is evidence of internal record identity, not independent proof of an experimental outcome.

The revision replaces earlier, unaudited accuracy summaries rather than combining them with the R1 campaign. It also removes unsupported global-to-local gain columns, the 0.5× communication interpretation, speech and attack results not present in this evidence, the curvature– conflict guarantee, and an unconditional convergence-rate claim. No “55-point” or historical 19.7-point personalization gain is used as a result of this campaign. Cross-method differences and component-removal effects are stated in percentage points, with their comparator and endpoint identified.

The empirical study reports complete measured ablation tables. It does not imply completion of unrecorded baseline, attack, architecture, or hyperparameter-sweep experiments.

## References

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, B. Agüera y Arcas, Communication-efficient learning of deep networks from decentralized data, in: AISTATS, PMLR 54, 2017, pp. 1273–1282. https://proceedings.mlr.press/v54/mcmahan17a.html.

[2] M. G. Arivazhagan, V. Aggarwal, A. K. Singh, S. Choudhary, Federated learning with personalization layers, arXiv:1912.00818, 2019. https://arxiv.org/abs/1912.00818.

[3] L. Collins, H. Hassani, A. Mokhtari, S. Shakkottai, Exploiting shared representations for personalized federated learning, in: ICML, PMLR 139, 2021, pp. 2089–2099. https://proceedings.mlr.press/v139/collins21a.html.

[4] J. Oh, S. Kim, S.-Y. Yun, FedBABU: Towards enhanced representation for federated image classification, in: ICLR, 2022. https://arxiv.org/abs/2106.06042.

[5] T. Li, S. Hu, A. Beirami, V. Smith, Ditto: Fair and robust federated learning through personalization, in: ICML, PMLR 139, 2021, pp. 6357–6368. https://proceedings.mlr.press/v139/li21h.html.

[6] Y. Huang, L. Chu, Z. Zhou, L. Wang, J. Liu, J. Pei, Y. Zhang, Personalized cross-silo federated learning on non-IID data, in: AAAI 35(9), 2021, pp. 7865–7873. https://doi.org/10.1609/aaai.v35i9.16960.

[7] Z. Wang, X. Fan, J. Qi, C. Wen, C. Wang, R. Yu, Federated learning with fair averaging, in: IJCAI, 2021, pp. 1615– 1623. https://doi.org/10.24963/ijcai.2021/223.

[8] T. Li, A. Beirami, M. Sanjabi, V. Smith, Tilted empirical risk minimization, in: ICLR, 2021. https://arxiv.org/abs/2007.01162.

[9] M. Mohri, G. Sivek, A. T. Suresh, Agnostic federated learning, in: ICML, PMLR 97, 2019, pp. 4615–4625. https://proceedings.mlr.press/v97/mohri19a.html.

[10] K. Pillutla, Y. Laguel, J. Malick, Z. Harchaoui, Federated learning with superquantile aggregation for heterogeneous data, Machine Learning 113, 2024, pp. 2955–3022. https://doi.org/10.1007/s10994-023-06332-x. Preprint: https://arxiv.org/abs/2112.09429.

[11] Z. Li, T. Lin, X. Shang, C. Wu, Revisiting weighted aggregation in federated learning with neural networks, in: ICML, PMLR 202, 2023, pp. 19767–19788. https://proceedings.mlr.press/v202/li23s.html.

[12] H.-Y. Chen, W.-L. Chao, On bridging generic and personalized federated learning for image classification, in: ICLR, 2022. https://arxiv.org/abs/2107.00778.

[13] Y. Tan, G. Long, L. Liu, T. Zhou, Q. Lu, J. Jiang, C. Zhang, FedProto: Federated prototype learning across heterogeneous clients, in: AAAI 36(8), 2022, pp. 8432–8440. https://doi.org/10.1609/aaai.v36i8.20819. Preprint arXiv:2105.00243, 2021.

[14] Q. Li, B. He, D. Song, Model-contrastive federated learning, in: IEEE/CVF CVPR, 2021, pp. 10713–10722. https://arxiv.org/abs/2103.16257.

[15] T. Li, M. Sanjabi, A. Beirami, V. Smith, Fair resource allocation in federated learning, in: ICLR, 2020. https://arxiv.org/abs/1905.10497.

[16] Y. Li, X. Wang, F. Yu, L. Sun, W. Zhang, X. Wang, FedCAP: Robust federated learning via customized aggregation and personalization, in: Annual Computer Security Applications Conference (ACSAC), 2024, pp. 747–760. https://doi.org/10.1109/ACSAC63791.2024.00067. Preprint: https://arxiv.org/abs/2410.13083.

[17] X. Cao, M. Fang, J. Liu, N. Z. Gong, FLTrust: Byzantine-robust federated learning via trust bootstrapping, in: NDSS, 2021. https://arxiv.org/abs/2012.13995 (2020 preprint).

[18] A. Krizhevsky, Learning multiple layers of features from tiny images, Technical Report, University of Toronto, 2009. Dataset: https://www.cs.toronto.edu/\~kriz/cifar.html.

[19] S. Caldas, S. M. K. Duddu, P. Wu, T. Li, J. Konečný, H. B. McMahan, V. Smith, A. Talwalkar, LEAF: A benchmark for federated settings, arXiv:1812.01097, 2018. https://arxiv.org/abs/1812.01097.