# Two Is Enough: Binary Classification Priors Sufice for Multi-Class In-Context Tabular Prediction

An Empirical Study under the TabICLv2 Framework

Zhixiong Cheng, Shiyuan Han, Jingjing Suo<sup>∗</sup>, Zhengli Wang Bairong Inc.AI, Beijing, China

2026-10-20

## Abstract

Tabular foundation models (TFMs) pretrain a transformer on large-scale synthetic priors so that a single forward pass yields zero-shot tabular prediction. TabICLv2 samples the class count of its classification prior during pretraining, and the prevailing configuration caps it at ten. We report a counter-intuitive result: setting this cap directly to two—making every pretrained classification task binary—still yields weights that perform zero-shot K-way classification $( K \gg 2 )$ . On the 38 TabArena classification datasets (200 bootstrap rounds, anchored at default $\mathrm { R F } = 1 0 0 0 )$ , the binary two-stage recipe reaches 1590.1 Elo, above the oficial default at 1568.0 $( \pmb { \Delta } = + 2 \pmb { 2 . 1 } )$ above TabPFN-v2.6 (1577.1) and RealTabPFN-v2.5 (1521.6). A production-scale validation on 60 real-world credit-scoring institutions independently confirms the direction: the same checkpoint improves test AUC on 44 of 60 institutions (paired Wilcoxon $p = 0 . 0 0 0 7$ for AUC, $p = 0 . 0 0 0 2$ for KS) at +1.5% inference cost. Mechanistically, classification in TabICLv2 is carried not by a fixed-cardinality output head but by an in-context comparison operator matching query-row embeddings against class-conditional embeddings, so the cardinality of the output space is set by the labels present in the context, not by the label space seen during pretraining. This makes a binary prior structurally cheaper to generate and cache than a ten-class prior, demotes “number of classes” from an architectural constraint to a tunable hyperparameter, and points to a class-count-agnostic regime of TFM pretraining.

TabArena Elo Top-15 (grouped) — Classification (38 datasets) DefaultTunedTuned + EnsembledTablCLv2-Custom-Binary-Prior (ours)  
![](images/44b177fbdfd5cd9884c16df328d185cc41d92a70613d988bb7beb3190e0f7437.jpg)  
Figure 1: TabArena Elo top-15 leaderboard on the full 38-dataset classification suite (200 bootstrap rounds, anchored at default $\mathrm { R F } = 1 0 0 0 )$ . Light-blue/orange/green grouped bars show the Default, Tuned, and Tuned + Ensembled configurations of each method, and the highlighted yellow bar marks the TabICLv2 oficial default baseline. Our binary two-stage checkpoint (TabICLv2-Custom-Binary-Prior, dark orange) reaches 1590.1 Elo, +22.1 above the oficial default (1568.0).

## 1 Introduction

Tabular foundation models (TFMs), namely TabPFN [5, 6], TabICL [8, 9], TabDPT [15], and TabFM [7], pretrain a transformer on synthetic priors (TabDPT instead uses real data with selfsupervised retrieval) so that it predicts on an unseen table in a single forward pass with no gradient updates [10]. The central mechanism is in-context learning (ICL): training rows and test rows are fed together as a context, and the model produces the test-row labels within the same forward pass.

A question long treated as an architectural constraint is: in classification, what determines how many classes the model can output? In classical supervised learning the answer is immediate: the width of the output head, fixed by the number of classes in the training set, which then bounds the number of classes at inference time. The ICL paradigm breaks this constraint. TabICLv2 samples the class count of its classification prior during pretraining, typically with an upper bound of ten, so a natural assumption is that this “ten” is simultaneously a prior-sampling parameter and a latent capability boundary of the model.

This paper presents a counterexample to that assumption and addresses three questions:

• Q1 (phenomenon). If the classification prior is fixed to two classes, can the fully pretrained weights still perform multi-class classification?

• Q2 (mechanism). If so, what design decouples the cardinality of the output space from the cardinality of the pretraining label space?

• Q3 (cost). What are the benefits and the failure boundaries of doing this?

Our observation is that the answer to Q1 is yes. On the 38 TabArena [16] classification datasets (200-bootstrap-round Elo, anchored at default RF = 1000), binary two-stage weights retrained from scratch for 540K steps (stage 1: 500K + stage 2: 40K; 10K fewer than the oficial three-stage schedule) with bagging inference reach 1590.1 Elo (Figure 1), above the oficial default at 1568.0 (∆ = +22.1; bootstrap intervals overlap, so the gain is directional, not significant). AUC-metric datasets are on par or slightly better; log-loss-metric datasets show a slight calibration degradation (§4.3), as predicted. A production-scale validation on 60 internal credit-scoring institutions (§4.5) makes the advantage statistically significant (paired Wilcoxon p = 0.0007 for AUC and p = 0.0002 for KS).

## Contributions. This paper contributes:

1. An unexpected empirical result. Binary-only prior pretraining does not sacrifice multi-class inference capability (see Table 1): the binary two-stage recipe (540K steps, no stage 3) scores 1590.1 vs. the oficial default’s 1568.0 Elo (550K steps over three stages). This supports treating “ten classes” as a tunable hyperparameter rather than a capability boundary.

2. A mechanistic explanation. Classification in TabICLv2 is carried by an in-context comparison operator, with K set by the labels in the context. Whether this holds depends on the label-encoder design: a shared encoder extrapolates, a per-class embedding table does not (§3.3); we give a code-level verification checklist.

3. A cost–benefit argument. Holding everything else fixed, a binary prior is structurally cheaper to generate and cache than a ten-class prior: it shrinks the per-task label-encoder input distribution, the label-embedding lookup, and the cached prior tensors carrying label-side state. Since priorgeneration cost is clif-shaped rather than linear (§4.4), this reduction buys a disproportionately long additional cache window, expanding the feasible frontier of long-horizon pretraining.

4. A failure boundary. Capping the class count too low is not free: measurable degradation appears when K is large, when classes are severely imbalanced, and when class labels are ordinal. We lay out these failure scenarios and the preliminary evidence for the calibration-related ones (§4.3).

## 2 Related Work

Tabular Foundation Models and In-Context Optimization. Prior-Data Fitted Networks [10] pretrain transformers on synthetic datasets so that a single forward pass approximates Bayesian in-context prediction [11, 12, 13]. TabPFN [5] introduced the paradigm for small classification tables; later work extended it to regression and longer contexts [6, 14] and made attention linear-time via inducing points [8, 9]. TabICLv2 [9] belongs to this lineage, driving both classification and regression by synthetic priors. This paper examines a setting the lineage accepts by default but has never systematically tested: the cardinality of the prior’s label space.

Label Representation in ICL Predictors. In ICL predictors, how labels are represented determines how the model “understands” classes. Two dominant designs:
<table><tr><td>Design</td><td>Mechanism</td><td>Output-space cardinality</td></tr><tr><td>Fixed output head</td><td>with C set by configuration the number of classes at</td><td>A final linear map to C logits, Hard constraint: C must be ≥ inference</td></tr><tr><td>Shared label encoder + com- Labels are mapped to embed- Soft constraint: K is parison decoder</td><td>dings by a shared encoder; determined by the distinct predictions arise from similar- labels in the context, and ity between query embeddings extrapolates</td><td></td></tr></table>

Our core claim is that TabICLv2 belongs to the second family, which is exactly why a binary prior still predicts multi-class. “Class-count extrapolation” has been glimpsed in text and vision ICL, but systematic evidence on tabular TFMs at the scale of a full pretraining run is missing.

Class Count as a Prior Hyperparameter. A classification prior’s class-count distribution afects both task diversity and generation cost. Prior work universally caps it at ten—matching the class counts of common TabArena/OpenML datasets—yet rarely asks what that cap actually constrains. We split the question into its capability and cost halves.

Architecture-Side Fixes for the Class-Count Limit. Two recent lines attack the same ceiling from the model side. Wu and Bergman [1] propose a mixture-block architecture for tasks with an arbitrary number of classes, calling the cap “the class size limitation—a crucial weakness of prior deep tabular zero-shot learners” and patching it architecturally. Bouadi et al. [2] pair a bi-axial attention encoder with a label-aware ICL head that scales to large label spaces via hierarchical decision routing (Orion-Bix). Both treat the cap as a tunable knob but reach that by adding machinery (a routing/mixing block, a label-aware head); ours instead reshapes the prior’s class-count distribution and leaves the TabICLv2 backbone intact. The two are complementary: an architecture-side fix supplies the operators for K-way inference, a prior-side fix reshapes the supervision they see.

## 3 Methodology

## 3.1 Preliminaries: The TabICLv2 Forward Pass

Given $( X _ { \mathrm { t r a i n } } , y _ { \mathrm { t r a i n } } , X _ { \mathrm { t e s t } } )$ , TabICLv2 executes the following pipeline (Figure 2):

X<sub>train</sub> → standardize → feature grouping → column embedding   
y<sub>train</sub> → label injection (label encoder)   
X<sub>test</sub> → standardize → column embedding ↓ column-wise attention / set transformer ↓ CLS token row-wise attention / ICL transformer ↓ query-row embedding   
⟨ query embedding × in-context class-conditional embeddings ⟩ → logits

Figure 2: TabICLv2 forward pass schematic. Standardization, feature grouping, column embedding, label injection, column-wise and row-wise attention, and the final query-to-class comparison step are shown. The two components that govern the question of this paper are the label encoder and the comparison decoder (§3.3).

Two steps matter most:

• Label injection. Training-row labels are encoded and concatenated onto their rows as “worked examples” in the context; the form of the encoder determines whether the class count is bound.

• Comparison decoding. Logits come from comparing the query-row embedding against the class-conditional embeddings (or prototypes) in the context. Its inputs are the classes present in the context, not the classes the configuration declared.

Pretraining must therefore teach how to compare (an operator), not which of ten classes to recognize (a table).

## 3.2 Where Does “Ten Classes” Come From?

The “target class count” appears in three distinct places in the TabICLv2 pipeline, and conflating them is the usual misunderstanding:

<table><tr><td>Location</td><td>Role Hard constraint?</td></tr><tr><td>Prior sampler</td><td>Decides how many classes each syn- × Affects only the thetic task draws, typically uniform over training-task [2, max_classes] distribution</td></tr><tr><td>Label-encoder width max_classes</td><td>If a per-class embedding table, its size equals △ Potentially a hard constraint (the</td></tr><tr><td>Evaluation protocol</td><td>critical checkpoint) The class count of the benchmark dataset × Unrelated to the itself model</td></tr></table>

Our change touches only the first location: we fix the classification prior’s class count to two and leave everything else (lr, optimizer, weight decay, dropout, sequence-length distribution, the regression prior’s pinball configuration) untouched. That location’s parameter is named max\_classes (shared by TabICL’s training scripts and the TabICL class).

Corroboration: max\_classes is the per-forward comparison cap, not the total number of representable classes. When a dataset has more classes than max\_classes, the TabICL family decomposes the K-way task into sub-problems of size ≤ max\_classes via hierarchical classification plus mixed-radix ensembling and recombines their outputs. The output-space cardinality is thus architecturally decoupled from max\_classes, which bounds a single comparison, not the number of classes the model can express. Binary-prior training does not alter this; it merely lowers each sub-problem from 10 to 2. (Whether TabICLv2 retains this verbatim is unverified.)

## 3.3 Why Does It Work? Three Hypotheses

H1 (operator hypothesis). Classification is comparison; the model learns a comparison operator. Binary tasks already supply the “match / mismatch” supervision to learn it, after which K is just how many objects enter the comparison.

H2 (encoder hypothesis). The label encoder is shared across classes (e.g., a shared scalar/vector encoding plus a class-agnostic transform), not a per-class embedding table. This is necessary for direct K-way inference: with a per-class table, out-of-table classes at inference (K > 2) have no embedding and extrapolation must fail in a single forward pass (it can still be rescued by the hierarchical classification + mixed-radix ensembling of §3.2, which decomposes the task into sub-problems of size ≤ max\_classes).

H3 (signal hypothesis). The discriminative signal comes mainly from column- and row-wise attention over features; label injection only supplies an alignment signal, so binary-only pretraining does not weaken the feature-side representation.

Code-level verification checklist (check each against the code before trusting §4):

1. Label encoder: is it nn.Embedding(max\_classes, d) (a per-class table) or a shared encoder with a class-agnostic transform?

2. Decoder: do logits come from a fixed linear head, or from similarity against in-context class embeddings?

3. Use of the class index: is the class index consumed as a positional encoding (extrapolates) or as a learned lookup (does not)?

4. Over-allocation: if a per-class table is over-allocated (e.g., sized for max\_classes = 100 but trained on 2 classes), H2 becomes “capacity over-allocation plus sparse utilization”; the conclusion can still hold, via a diferent mechanism.

## 3.4 Experimental Protocol

<table><tr><td>Dimension</td><td>Setting</td></tr><tr><td>able</td><td>Target independent vari- Classification prior&#x27;s target class count: official default (max_classes=10, three-stage schedule, 550K steps) vs. ours (max_classes=2, from-scratch two-stage retrain, 540K steps)</td></tr><tr><td>Current measured recipe</td><td>Ours is a from-scratch retrain of stage 1 + stage 2 (500K + 40K = 540K) with bagging; the official default adds a 10K-step stage 3 (550K). A strictly single-variable run (class count the</td></tr><tr><td>Controlled variables</td><td>sole difference) is in progress (§4.2, §5) lr, optimizer, weight decay, dropout, micro-batch, grad-accum, sequence-length distribution, recompute True (full diff in Appendix A)</td></tr><tr><td>Prior supply</td><td>Offline prior cache (target: covering all 500K steps) to avoid an order-of-magnitude latency artefact from online generation</td></tr><tr><td>Evaluation suite Metrics</td><td>TabArena classification subset, 38 datasets (binary + multiclass) Elo (Bradley-Terry, 200 bootstrap rounds, anchored to default</td></tr><tr><td></td><td>RF = 1000), AUC / log-loss (assigned per dataset; ± std over three-fold CV)</td></tr><tr><td>Protocol</td><td>Three-fold outer CV per dataset; comparison uses final weights only (no fine-tuning, no post-hoc calibration; bagging enabled in all TabICLv2 configurations)</td></tr></table>

## 4 Experiments

## 4.1 Setup

We evaluate on 38 TabArena [16] classification datasets (binary + multiclass), strictly zero-shot (no gradient updates or post-hoc calibration; bagging enabled throughout). Elo is the Bradley–Terry rating over 200 bootstrap rounds anchored at default RF = 1000; per-dataset metrics are AUC or log-loss, each with ± std over three-fold CV.

## 4.2 Main Results

Table 1: TabArena classification leaderboard (38 datasets, 200 bootstrap rounds, anchored at default RF = 1000). Elo+ / Elo− are bootstrap intervals; Rank is the mean rank; Win rate is the pairwise win ratio. Prior K is the classification prior’s target class count (for TabFM, the cap reported in [7]; for the other reference checkpoints, the paper’s default). Steps gives total pretraining steps: the oficial default runs the full three-stage schedule (500K + 40K + 10K = 550K); our Custom variant is a from-scratch retrain of stage 1 + stage 2 only (540K, no stage 3). The four reference rows (TabFM, TabPFN-v2.6, the TabICLv2 oficial default, RealTabPFN-v2.5) use their publicly reported Elo without rerunning; “—” marks the original published schedule.
<table><tr><td>Configuration</td><td>Prior K</td><td>Steps</td><td>Elo ↑</td><td>Elo+ / Elo-</td><td>Rank↓</td><td></td><td>Win rate</td></tr><tr><td>TabFM (reference)</td><td>10</td><td></td><td>1749.6</td><td>+133.5 / -114.7</td><td></td><td>9.59</td><td>0.901</td></tr><tr><td>TabICLv2-Custom-Binary-</td><td>2</td><td>540K</td><td>1590.1</td><td>+75.2 / -63.3</td><td></td><td>17.91</td><td>0.806</td></tr><tr><td>Prior (ours) TabPFN-v2.6 default (reference)</td><td>10</td><td></td><td>1577.1</td><td>+60.6 / -56.2</td><td></td><td>18.77</td><td>0.796</td></tr><tr><td>TabICLv2 official default (base- line)</td><td>10</td><td>550K</td><td>1568.0</td><td>+76.5 / -66.8</td><td></td><td>19.40</td><td>0.788</td></tr><tr><td>RealTabPFN-v2.5 default (refer- ence)</td><td>10</td><td></td><td>1521.6</td><td>+54.7 / -52.1</td><td></td><td>22.80</td><td>0.749</td></tr></table>

Finding. The binary two-stage recipe scores 1590.1 vs. the oficial default’s 1568.0 (a ∆ = +22.1 directional gain), with rank (17.91 vs. 19.40) and win rate (0.806 vs. 0.788) moving the same way. The bootstrap intervals overlap (+75.2/−63.3 vs. +76.5/−66.8), so this is a directional advantage, not a significant diference. The attribution boundary matters: ours is a from-scratch retrain of stage 1 + stage 2 only (540K, no stage 3) with bagging, whereas the default adds a 10K-step stage 3, and recipe-side hyperparameters were not verified item-by-item, so the binary prior’s contribution remains unisolated (Limitation 2). Even so, cutting the class count to two costs no observable Elo against the oficial default—if anything +22.1 with 10K fewer steps—while saving prior cost (§4.4). Figure 3 puts all methods on one score axis (higher is better): ours (dark orange) directly above the oficial default, with a score margin of +0.013. The credit-scoring evaluation (§4.5) supplies the paired significance the Elo comparison lacks.

![](images/81d570a460d0a1c34e11b5c0805fddcb54d9d12c66ba8bea7e19aae8bd8c11f7.jpg)  
Figure 3: TabArena score on the full 38-dataset classification suite (binary + multiclass; higher is better), covering all evaluated methods. Our binary two-stage checkpoint (TabICLv2-Custom-Binary-Prior, dark orange) scores +0.013 above the TabICLv2 oficial default baseline (yellow); the Default / Tuned / Tuned + Ensembled bars show the per-method configurations reported by TabArena.

4.3 When Does It Break?
<table><tr><td>Scenario</td><td>Expected risk</td><td>Mechanism</td></tr><tr><td>Large K (&gt; cap)</td><td></td><td>Accuracy degradation Crowded class embeddings in context; too few samples per class</td></tr><tr><td>Severe class imbalance</td><td>Lower minority-class recall</td><td>The comparison decoder estimates rare-class &quot;prototypes&quot; with high variance</td></tr><tr><td>Ordinal classes</td><td>haviour</td><td>Changed threshold be- Binary prior only ever sees a single cut point, never multi-cut structure</td></tr><tr><td>required</td><td>fore accuracy</td><td>Well-calibrated probabilities Log-loss degrades be- Narrower prior class distribution → mismatched prior shape on output</td></tr><tr><td>Regression tasks</td><td>No effect (expected)</td><td>probabilities Regression uses a pinball quantile loss, independent of the classification prior&#x27;s class count; regression metrics were unaffected in our runs</td></tr></table>

Preliminary empirical evidence (binary recipe vs. oficial default). Row 4 is already supported by the qualitative pattern: log-loss datasets show slight calibration degradation, while AUCscored datasets show parity or slight improvement. This separation of discrimination (unafected) from calibration (slightly degraded) matches the mechanism: the comparison decoder’s probability shape depends on the prior’s class distribution, while the ranking does not. Calibration-sensitive applications (e.g., credit-scoring probability outputs) need dedicated post-hoc calibration.

## 4.4 Cost Analysis: Why This Matters in Practice

Why a binary prior is cheaper in principle. Prior-generation cost has three structural drivers: (1) the label space (distinct classes per task); (2) the number of tasks drawn from it; and (3) the per-task cost (SCM sampling, label rendering, post-processing). Holding (2) and (3) fixed, lowering (1) from ten to two reduces the distinct label combinations the prior must support. In the TabICLv2 stack this shrinks the label-encoder input distribution, the label-embedding lookup, and the cached prior tensors carrying label-side state—all consequences of declaring max\_classes = 2 at the prior sampler (§3.2), with no extra engineering.

Why this matters in practice: the cache clif. In real TabICLv2 training, prior tensors are precomputed into an ofline cache; once the cache is exhausted the trainer falls back to online generation, which costs roughly an order of magnitude more per step. In a separate (uncontrolled) observation during our 500K-step run, prior\_time jumps from 0 s (cache hit) to $\approx 1 1 . 5 ~ \mathrm { s }$ (cache miss) once the cached window is exceeded, with the rest of the step at ≈ 1.4 s. This is an observed step-time breakdown, not a controlled cap-10 vs. cap-2 comparison. It does establish the engineering asymmetry: because the clif is sharp rather than linear, any prior-cost reduction—however small— buys a disproportionately long additional cache window. Lowering the class count from ten to two is such a structural reduction (item 1), so it lands on the steep part of the clif; that is the engineering meaning of “Two Is Enough.”

## 4.5 Production-Scale Credit Scoring Evaluation (60 Institutions)

Setup. Beyond TabArena, we evaluate two checkpoints on an internal suite of 60 real-world credit-scoring institutions (binary default-risk scoring; 300 features each; training sizes n ≈ 1K– 17K; identifiers omitted for confidentiality): the oficial TabICLv2 default (prior $K = 1 0 ;$ full three-stage schedule) and our binary-prior checkpoint (from-scratch two-stage retrain, 540K steps; the same checkpoint as Table 1). Each institution uses a train/test hold-out, and we report test-set AUC and KS, the standard credit-risk discrimination metrics. As on TabArena the two difer in more than the class cap (ours lacks stage 3, 540K vs. 550K steps), so this comparison shares the attribution boundary of Limitation 2.

Table 2: Production credit-scoring suite: 60 institutions (300 features each), institution-equal-weighted macro averages on the test hold-out. $\Delta$ is ours minus the oficial default. Across institutions, ours wins on test AUC in 44 / ties in 1 / loses in 15 (strict sign); the paired two-sided Wilcoxon signed-rank test (normal approximation with tie-averaged ranks) gives $p = 0 . 0 0 0 7$ on AUC and $p = 0 . 0 0 0 2$ on KS.
<table><tr><td>Checkpoint</td><td>Prior K</td><td> $\mathrm { A U C } _ { \mathrm { t e s t } }$ </td><td> $\mathrm { K S } _ { \mathrm { t e s t } }$ </td><td>Train-test gap</td><td>Cost (s)</td></tr><tr><td>official default (baseline)</td><td>10</td><td>0.7014</td><td>0.3082</td><td>0.2784</td><td>23.94</td></tr><tr><td>ours (binary, 540K)</td><td>2</td><td>0.7033</td><td>0.3134</td><td>0.2770</td><td>24.30</td></tr><tr><td> $\Delta \ : ( \mathrm { o u r s \ : - \ : d e f a u l t } )$ </td><td></td><td>+0.0019</td><td>+0.0053</td><td>-0.0014</td><td>+0.36</td></tr></table>

Findings. Table 2 reports the macro averages and the paired statistics. Three observations:   
1. Significance. Ours improves test AUC on 44 of 60 institutions $( \Delta \mathrm { A U C } = + 0 . 0 0 1 9$ , ∆KS $= + 0 . 0 0 5 3$ macro), paired-significant at $p = 0 . 0 0 0 7$ for AUC and $p = 0 . 0 0 0 2$ for KS (Table 2). Unlike the directional-only TabArena Elo gap (§4.2), the 60-institution paired design has the power to resolve an efect of this size.

2. Where the gain lives. The improvement concentrates where the baseline already performs well: terciles by baseline test AUC give $\Delta \mathrm { A U C } = + 0 . 0 0 3 3$ (top, baseline 0.79), +0.0018 (middle), +0.0006 (bottom, baseline 0.61). Only 3 institutions show $\Delta \mathrm { A U C } > + 0 . 0 1$ (all mid-sized, n = 2.7K– 17K), and the two largest degradations $( - 0 . 0 1 3 , - 0 . 0 2 7 )$ both fall at the two smallest $( n = 9 6 2$ 1,169; baseline $\mathrm { A U C } \approx 0 . 5 9 \ – 0 . 6 1 )$ , where single-split variance dominates. The binary prior thus improves fitting where signal exists rather than creating capability on weak-signal institutions.

3. Overfitting and inference cost. The train−test AUC gap of ours (0.2770) is slightly below the oficial default’s (0.2784): cutting the prior’s class count to two does not widen generalization error on this suite. Inference cost rises by 0.36 s per institution (+1.49% over 23.94 s), i.e., no material architectural overhead at prediction time (inference cost, distinct from the prior-generation cost of §4.4).

## 5 Discussion and Limitations

Why this is “supposed” to be true. If classification were genuinely carried by a K-dimensional output head, then “pretraining saw only two classes” would necessarily lock $K \le 2$ . Observing extrapolation means the causal direction is the opposite: the labels in the context define the output space, and the output head is an implementation detail. This ofers a cleaner mental model of ICL predictors: pretraining teaches how to infer a decision rule from examples, not a fixed classifier.

## Limitations.

1. We establish that “two sufices,” not what is optimal. A cap of 3–5 may be the more robust default.

2. Recipe confounding. The binary configuration (1590.1) is a from-scratch retrain of stage 1 + stage 2 only (540K, no stage 3) with bagging, whereas the oficial default (1568.0) runs the full three-stage schedule (550K); recipe-side hyperparameters were additionally not verified item-byitem (exact optimizer and bagging settings, etc.), so training length, schedule, and recipe all difer alongside the class cap. The data support “the binary prior shows no observable loss against the oficial default,” not “the class cap alone causes the gain.” A strictly single-variable run (scheduleand bagging-matched, class count the sole diference) is in progress.

3. The task-dificulty distribution is not strictly deconfounded. Binary tasks may have a diferent synthetic SCM complexity distribution than ten-class tasks, so “a weaker task mixture trained an equally good model” cannot be excluded without controlling complexity. Two references frame the risk: Chan et al. [3] show that training-data distributional properties (long tail, burstiness, rare classes) drive emergent ICL, so a binary prior also narrows the long-tail/rare-class surface the model sees; Raventós et al. [4] identify a task-diversity threshold below which ICL collapses to a Bayesian estimator mimicking the pretraining distribution, so “a smaller cap does not hurt” requires staying above that threshold under cap-2. We control for neither.

4. Small institutions are noise-dominated. Institutions with n < 2,000 training rows show high single-split variance, and every large degradation in our suite falls there, so per-institution results at that scale are noise-dominated (§4.5).

Future work.

• Class-count curricula: binary first, then progressively more classes; does this beat a fixed cap of 2 or 10?

• Model-adaptive class count, driven by a structural prior rather than configured.

• Schema-aware routes: class semantics (e.g., an ICD hierarchy) injected by an LLM rewriting the pipeline (the TabFM-Auto route [17]) or, conceivably, through the prior’s class-count distribution.

## 6 Conclusion

We split the “target class count cap” of TabICLv2 pretraining—a setting accepted by default—into its capability-boundary and cost-parameter halves, and report an unexpected result: fixing the prior’s target class count to two still yields weights that perform K-way zero-shot classification (1590.1 vs. the oficial default’s 1568.0 Elo, $\pmb { \Delta } = + 2 2 . 1$ , a directional advantage; and a small but statistically significant test-AUC and KS gain on a 60-institution credit-scoring suite, paired Wilcoxon $p = 0 . 0 0 0 7$ for AUC and $p = 0 . 0 0 0 2$ for KS, at +1.5% inference cost), because classification is carried by an in-context comparison operator and the output-space cardinality is set by the context labels rather than the pretraining label space. This demotes “number of classes” from an architectural constraint to a tunable hyperparameter and, in principle, lowers prior generation and cache cost, expanding the feasible frontier of long-horizon pretraining. We also state the conditions under which it holds (label-encoder and decoder design), the attribution boundary of the current +22.1 (ours is a from-scratch 540K two-stage retrain without stage 3, the default adds a 10K-step stage 3 on the same stage 1 + stage 2, and recipe-side hyperparameters were not verified item-by-item, so the gain is not attributable to the class cap alone), and its failure boundaries (large K, strong imbalance, ordinal structure, and calibration, the last already with preliminary evidence).

## Acknowledgments

The authors thank Bairong Inc. for its support of this work, including the compute resources and engineering support for the pretraining runs and the production-scale evaluation.

## A Configuration Dif (Target Independent Variable)

Listing 1: Configuration dif between the oficial default baseline and our binary two-stage recipe. The default runs the full three-stage schedule (500K + 40K + 10K = 550K steps); our Custom variant retrains stage 1 + stage 2 only (540K, no stage 3).

```perl
# official default ( baseline ; 1568.0 Elo; 10- class prior )
-- max_classes 10 # full official schedule : stage 1 (500
K)
# + stage 2 (40K) + stage 3 (10K) =
550K
# ours ( binary two -stage , Custom - Binary - Prior ; 1590.1 Elo )
# from - scratch retrain of stage 1 (500 K) + stage 2 (40K), both
# under the binary prior ; no stage 3
-- max_classes 2
-- bagging <enabled > # bagging inference
# everything else (-- recompute True / -- use_flash_attn3 False / lr /
# optimizer / weight_decay / dropout / micro_batch / grad_accum /
# sequence - length distribution ): intended identical to the official run
```

On the parameter name. The target independent variable is named max\_classes, shared by TabICL’s training scripts (train/run.py, scripts/train\_stage\*.sh) and the TabICL model class.

## References

[1] Y. Wu and D. L. Bergman. Zero-shot meta-learning for tabular prediction tasks with adversarially pre-trained transformer. In ICML, 2025.

[2] M. Bouadi, P. Seth, A. Tanna, and V. K. Sankarapu. Orion-Bix: Bi-axial attention for tabular in-context learning. arXiv:2512.00181, 2025.

[3] S. C. Y. Chan, A. Santoro, A. K. Lampinen, J. B. Wang, A. Singh, P. H. Richemond, J. McClelland, and F. Hill. Data distributional properties drive emergent in-context learning in transformers. In NeurIPS, 2022.

[4] A. Raventós, M. Paul, F. Chen, and S. Ganguli. Pretraining task diversity and the emergence of non-Bayesian in-context learning for regression. In NeurIPS, 2023.

[5] N. Hollmann, S. Müller, K. Hutmacher, and F. Hutter. TabPFN: A transformer that solves small tabular classification problems in a second. In ICLR, 2023.

[6] N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, M. Mohr, and F. Hutter. Accurate predictions on small data with a tabular foundation model. Nature, 2025.

[7] W. Kong et al. TabFM: A zero-shot foundation model for tabular data. Note: distinct from the separate self-evolving pipeline line TabFM-Auto by D. Fu et al. (2026, arXiv:2609.37989, see entry below). arXiv:2609.37959, 2026.

[8] J. Qu, K. Holsheimer, T. Painter, and F. Hutter. TabICL: A tabular foundation model for in-context learning on large data. In ICML, 2025.

[9] J. Qu, D. Holzmüller, G. Varoquaux, and M. Le Morvan. TabICLv2: A better, faster, scalable, and open tabular foundation model. arXiv:2602.11139, 2026.

[10] S. Müller, N. Hollmann, S. P. Arango, J. Grabocka, and F. Hutter. Transformers can do Bayesian inference. In ICLR, 2022.

[11] A. Garg, D. Tsipras, P. Liang, and G. Valiant. What can transformers learn in-context? A case study of simple function classes. In NeurIPS, 2022.

[12] Y. Li, M. Choi, J. Lee, and S. Oh. Transformers can do Bayesian prediction in-context. arXiv preprint arXiv:2301.12178, 2023.

[13] J. Von Oswald, M. Schlegel, A. Meulemans, S. Kobayashi, E. Niklasson, N. Zucchet, N. Scherrer, N. Miller, M. Sandler, B. Agüera y Arcas, M. Vladymyrov, R. Pascanu, and J. Sacramento. Uncovering mesa-optimization algorithms in transformers. arXiv:2309.05858, 2023.

[14] L. Grinsztajn, K. Flöge, O. Key, F. Birkel, P. Jund, B. Roof, M. Manium, S. B. Hoo, M. Bühler, A. Garg, D. Safaric, J. Robertson, B. Jäger, S. Alessi, A. Hayler, V. Moroshan, L. Purucker, P. Singer, A. Arazi, J. Siems, J. H. Metzen, G. Grab, N. Erickson, S. Guo, E. Kalfon, S. Bing, D. Salinas, C. Cornu, L. C. Wehrhahn, D. Kriuchkova, K. Kaya, L. Sidhoum, M. Salmon, J. Chen, M. Hulsebos, Y. LeCun, S. Müller, B. Schölkopf, S. Gambhir, N. Hollmann, and F. Hutter. TabPFN-3: Technical report. arXiv:2605.13986, 2026.

[15] J. Ma, V. Thomas, R. Hosseinzadeh, A. Labach, H. Kamkari, J. C. Cresswell, K. Golestan, G. Yu, M. Volkovs, and A. L. Caterini. TabDPT: Scaling tabular foundation models on real data. In NeurIPS, 2025.

[16] N. Erickson et al. TabArena: A living benchmark for machine learning on tabular data. In NeurIPS Datasets and Benchmarks, 2026.

[17] D. Fu et al. TabFM-Auto: Self-evolving pipelines for tabular foundation models. arXiv:2609.37989, 2026.