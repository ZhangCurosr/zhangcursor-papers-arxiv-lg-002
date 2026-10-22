# The Predictive–Optimization Gap in GuanDan Self-Play

Qicheng Zou, Biao Yuan, Siyu Song, Wentao Liu, and Xiaokun Wang

Abstract—Outcome-predictive signals are often treated as promising optimization targets in reinforcement learning, but predictive validity and optimization utility are different claims. We expose this distinction in GuanDan, a multi-round imperfect information game in which the implication of a round outcome depends on the public match state. Held-out analysis shows that state–gain interactions provide incremental predictive value beyond a constant signed-gain effect, establishing that match progress is predictively meaningful. We then use changes in a frozen match-win predictor, ∆Φ, as a controlled round-level supervision intervention in a matched DanZero-style Monte Carlo self-play protocol. The intervention can succeed in an individual lineage: a selected progress-augmented checkpoint substantially outperforms its matched Baseline on a locked evaluation. Yet the method-level result reverses this impression: none of five preregistered replication seeds is positive, the pooled ten-seed interval includes zero, and across-iteration effects repeatedly change sign without a common late-stage advantage. These results identify a predictive–optimization gap: under the studied learner, a signal can describe outcome-relevant progress while failing to provide a reliably beneficial optimization direction.

Index Terms—GuanDan, optimization reliability, predictive supervision, reinforcement learning, self-play.

## I. INTRODUCTION

Outcome prediction and policy optimization are often coupled by an appealing intuition: if an intermediate signal better predicts terminal success, then exposing that signal to the learner should improve credit assignment. The first implication is statistical, whereas the second is algorithmic. A signal is predictively meaningful when it contains heldout information about eventual outcomes beyond a simpler alternative; it is optimization reliable when using it produces a consistently beneficial policy effect across stochastic training runs. The broader predict-then-optimize literature already shows that minimizing predictive loss and maximizing downstream decision quality are different objectives [1], [2]. Prediction alone does not guarantee optimization reliability because policy learning also depends on credit granularity, function approximation, finite-sample updates, and the feedback loop between a changing policy and its self-generated data; related performative-learning work formalizes how deployed decisions can change the distribution on which later learning occurs [3], [4]. This paper isolates that usually implicit gap.

GuanDan provides a natural test bed for separating these two claims. It is a four-player, team-based, imperfect-information card game in which a complete match spans multiple rounds as the public level state of each team evolves [5]. DouZero established large-scale self-play deep Monte Carlo learning for DouDizhu, while DanZero-family work transferred this foundation to GuanDan and DanZero+ further refined a pretrained DMC policy using PPO [6]–[8]. Unlike settings in which the episode boundary and the principal competitive objective coincide, GuanDan has a temporal hierarchy: actions determine round outcomes, and a sequence of rounds determines the match winner. Consequently, the same one-level gain can have a different implication for eventual victory early in a match than in a close, near-terminal state.

This hierarchy lets us ask two questions that are related but not equivalent. First, does the predictive relationship between a fixed signed round gain and eventual victory depend on the pre-round public state? Second, if that dependence is exposed explicitly as round-level supervision, does it yield a reproducible improvement in self-play optimization? The first question concerns held-out predictive content; the second concerns the distribution of policy effects over independent training lineages. Separating them requires both a predictive test and method-level replication rather than inferring optimization value from predictor quality or from one successful checkpoint.

We construct a controlled probe of this predictive– optimization gap. Following DanZero-style Monte Carlo practice [6], [7], the Baseline combines signed round gain with the eventual match outcome and broadcasts that label to every executed action in the round. We train and freeze a publicstate match-win predictor Φ, then define match progress as the change in predicted winning probability across a completed round, ∆Φ. The progress-augmented target is

$$
G ^ { \mathrm { p r o g } } = G ^ { \mathrm { b a s e } } + \beta \Delta \Phi ,\tag{1}
$$

where $G ^ { \mathrm { b a s e } }$ is the baseline round-level label and $\beta$ controls the contribution of match progress. We use this augmentation as a diagnostic intervention, not as an assumption that a more predictive label must be a better reward. Because both arms share the same initialization, architecture, learner, budget, and evaluation protocol, their difference tests whether predictive match context is converted into reliable policy improvement under this specific round-broadcast, executed-action regression mechanism.

The two stages of evidence lead to a counterintuitive result. On held-out matches, adding pre-round state–gain interactions improves log loss over a constant-gain specification, and the frozen predictor produces semantically consistent, statedependent progress estimates. Thus, the signal is predictively meaningful by the criterion defined above. At optimization time, however, the selected seed-1 Progress-10 checkpoint beats its matched Baseline by 0.143 MeanDelta on a locked 1,000-pair evaluation, while none of the five preregistered replication seeds has a positive effect and the pooled ten-seed interval includes zero. Across-iteration comparisons repeatedly cross zero and show no common late-stage Progress advantage. The central finding is therefore not that match progress is useless, nor merely that one reward variant failed; it is that a verified predictive signal can support a strong selected policy without yielding a reliable optimization effect.

This distinction changes how intermediate supervision should be validated. Predictor accuracy establishes what a signal describes under an observed distribution, while across-run optimization evidence establishes whether a learner can use it consistently as its policy and data distribution co-evolve. A successful checkpoint demonstrates possibility, but it cannot substitute for replication over training randomness. For outcome-potential supervision, these are separate empirical obligations. Accordingly, we do not claim to be the first to observe that prediction and decision objectives can differ. Our contribution is a controlled self-play case study that tests predictor validity, selected-checkpoint efficacy, and methodlevel optimization reliability within one matched intervention, then shows that the three conclusions separate.

Our contributions are summarized as follows:

• We construct a two-stage self-play diagnostic that separately evaluates predictor validity, selection-conditioned checkpoint efficacy, and optimization reliability across independent training runs.

• We establish that GuanDan round gains have statedependent predictive meaning using held-out state–gain interactions and a calibrated, frozen public-state outcome model.

• We document a predictive–optimization gap under roundbroadcast Monte Carlo regression: selected-checkpoint success coexists with failed method-level replication, heterogeneous ten-seed effects, and sign-changing traininghorizon trajectories.

## II. BACKGROUND AND RELATED WORK

## A. Self-Play in Card Games and Training Reliability

Imperfect-information card games provide challenging environments for reinforcement learning because agents must act under hidden information, large and variable action spaces, and mixed competitive–cooperative interaction. DouZero and DanZero-family systems established large-scale self-play learning for DouDizhu and GuanDan [6]–[8], while OpenGuanDan and related work such as Big 2 further demonstrate the applicability of this paradigm [5], [9]. Most existing card-game RL systems primarily address policy learning, representation, action-space complexity, self-play scalability, or imperfectinformation handling. Our work retains an online self-play setting while focusing on how match-progress information is exposed to round-level learning, rather than proposing a new policy architecture or imperfect-information solver.

Reliable evaluation in deep RL requires uncertainty over independent training runs, because extensive evaluation of one frozen policy does not measure how retraining changes the learned policy [10], [11]. Self-play adds an endogenous source of variation: each policy helps generate the opponents and state distribution used for subsequent learning. In nontransitive games, self-play can exhibit strategic cycles rather than monotonic improvement [12]; we do not claim that this specific mechanism is identified in GuanDan, but it motivates treating lineage-dependent trajectories as a method-level object rather than as evaluation noise. Our protocol accordingly separates fixed-checkpoint uncertainty from variability across matched self-play training runs.

## B. Predictive Validity and Decision Utility

Predict-then-optimize and decision-focused learning distinguish statistical prediction loss from the quality of decisions induced downstream. Smart Predict-then-Optimize defines loss through decision error, while decision-focused learning differentiates through an optimization layer, because standard predictive metrics need not be reliable proxies for downstream utility [1], [2]. These works establish the general prediction– decision mismatch that motivates our study; our novelty is not the mismatch itself. Instead, we test how it appears when a validated predictor-derived signal is inserted into an online selfplay regression target and evaluated at three levels: predictor validity, selection-conditioned checkpoint efficacy, and acrossrun optimization reliability.

Feedback makes this distinction sharper in adaptive systems. Performative prediction studies settings in which deployed predictions alter the distribution of later observations, and performative RL extends this idea to policies that affect subsequent rewards or dynamics [3], [4]. Our environment and frozen predictor differ from those formal models, but the common structural point is that freezing a scoring function does not freeze the data distribution induced by decisions. In GuanDan self-play, the progress term changes policy updates, and the updated policy generates later training matches; heldout predictive improvement therefore supplies no monotonic guarantee for the resulting optimization path.

## C. Outcome-Potential and Intermediate Supervision

Temporal credit assignment concerns how delayed outcomes should inform earlier decisions [13]. Reward-redistribution methods such as RUDDER and randomized return decom position move delayed trajectory returns toward more informative intermediate events, often seeking return-equivalent rearrangements of the same total return [14], [15]. Value transport likewise carries long-horizon outcome information to earlier decisions [16]. These approaches motivate intermediate supervision, but they generally operate over primitive timesteps, actions, or trajectory events.

A complementary line constructs intermediate supervision from predictions of eventual task success. Outcome-potential methods evaluate intermediate states by how favorable they appear with respect to the final outcome and use changes in that evaluation as denser feedback; recent examples in long-horizon language-model agent settings include SIOP, TRACE, and TIPS [17]–[19]. Although these methods operate in different domains, they motivate converting outcome predictions into intermediate supervision. We study the same conversion at completed-round boundaries, but test predictive content and optimization reliability separately rather than treating predictor quality as evidence that the learner will benefit.

## D. Policy Guarantees and Causal Credit

Differences between state potentials are closely related in form to potential-based reward shaping (PBRS). In the classical formulation, an additional transition reward $F ( s , a , s ^ { \prime } ) ~ =$ $\gamma \Phi ( s ^ { \prime } ) - \Phi ( s )$ preserves the optimal policy under the corresponding Markov decision process assumptions [20]; related results extend invariance properties to multiplayer stochastic games under appropriate conditions [21]. Our round-boundary signal is only potential-difference inspired: it augments an executed-action Monte Carlo regression label and is not a policy-preserving reward transformation.

Predictive progress is also distinct from causal credit. Counterfactual credit-assignment methods estimate how outcomes would differ under alternative actions or contributions [22], [23], whereas $\Delta \Phi _ { k }$ describes the observed change in public-state win probability after a realized round. It does not isolate the marginal contribution of an executed action, and every action in the round receives the same label. Moreover, adding $\beta \Delta \Phi _ { k }$ does not preserve trajectory return as in return-equivalent redistribution. The absence of policy-invariance, return-equivalence, and causal-credit guarantees makes optimization benefit an empirical question even after predictive meaning has been established.

## III. DIAGNOSTIC INTERVENTION

We construct a diagnostic intervention that connects a validated predictive signal to policy optimization without presuming that the connection will be beneficial. The design has three parts: a Baseline round-level target, a frozen public-state predictor whose held-out value can be tested independently, and a matched augmentation that changes only the supervision target supplied to the self-play learner. This separation lets us ask whether a predictively meaningful state–gain signal becomes an optimization-reliable training signal under a fixed learning protocol.

## A. Problem and Round-Level Targets

We consider GuanDan as a sequence of rounds embedded in a longer match. Let $x _ { k }$ denote the public match state before round $k + 1$ , and let $\tau _ { k } ~ = ~ \{ ( s _ { k , t } , a _ { k , t } ) \} _ { t = 1 } ^ { T _ { k } }$ denote the within-round trajectory of executed actions. The game therefore contains a natural temporal hierarchy, action → round → match. Our work focuses on the second boundary: how information about the eventual match outcome can be assigned to individual rounds, without attempting within-round action-level attribution.

For a team-relative sample, let $y _ { \mathrm { { m a t c h } } } \in \{ - 1 , + 1 \}$ denote the final match outcome, and let $r _ { k } ^ { \mathrm { r a n k } }$ denote the signed level gain from round k. In our implementation, the nonzero round outcomes take magnitudes corresponding to one-, two-, or threelevel gains. Following DanZero-style Monte Carlo practice, the baseline round-level supervision target is

$$
G _ { k , t } ^ { \mathrm { b a s e } } = \boldsymbol { r } _ { k } ^ { \mathrm { r a n k } } + \boldsymbol { y } _ { \mathrm { m a t c h } } ,\tag{2}
$$

Every executed action within round k receives the same label, and $y _ { \mathrm { { m a t c h } } }$ is broadcast to all rounds after the match ends. We call $G _ { k , t } ^ { \mathrm { b a s e } }$ a round-level Monte Carlo supervision target: it is the regression label used in executed-action learning, not a standard return-to-go obtained by accumulating subsequent round rewards. For a fixed realized pair $( r _ { k } ^ { \mathrm { r a n k } } , y _ { \mathrm { m a t c h } } )$ , the scalar label does not vary with the pre-round public match state. For example, two one-level gains that end in the same match outcome receive identical baseline labels even if one occurs early in a match and the other occurs near a terminal level configuration. Because the network observes s and is trained against $y _ { \mathrm { { m a t c h } } }$ , it can in principle learn $\mathbb { E } [ G ^ { \mathrm { b a s e } } \mid s , a ]$ that already varies with match context. An explicit progress term therefore does not supply missing state access; it tests whether presenting the same outcome-relevant dependence as a round-local teacher produces more favorable finite-sample updates than recovering it implicitly from terminal outcomes.

Let $\Phi _ { \psi } ( x )$ denote a frozen public-state estimate of matchwinning probability. For a completed round $k ,$ the matchprogress signal is

$$
\Delta \Phi _ { k } = \Phi _ { \psi } ( x _ { k + 1 } ) - \Phi _ { \psi } ( x _ { k } ) .\tag{3}
$$

A positive value indicates that the public match state became more favorable to the evaluated team, whereas a negative value indicates the opposite. Unlike the raw level gain, $\Delta \Phi _ { k }$ is explicitly state dependent: the same rank result may correspond to different changes in estimated winning probability at different stages of the match. Eq. (3) should not be interpreted as causal attribution; it is a predictive teacher signal under the datagenerating policy distribution.

The progress-augmented target expands $\operatorname { E q . }$ (1) as

$$
G _ { k , t } ^ { \mathrm { p r o g } } = r _ { k } ^ { \mathrm { r a n k } } + y _ { \mathrm { m a t c h } } + \beta \Delta \Phi _ { k } ,\tag{4}
$$

where $\beta$ controls the contribution of match progress and every executed action in round k again shares the same round label. Thus the diagnostic augments a target that already contains local rank feedback and the broadcast final match outcome. The initial optimization probe instantiates two preregistered scales, Progress-10 and Progress-20, with arm-specific target scaling constants locked before policy training from a fixed reference dataset.

## B. Frozen Predictor

We train an independently fitted public-state match-win predictor $\Phi _ { \psi } ( x ) = P _ { \psi } ( y _ { \mathrm { { m a t c h } } } = + 1 \mid x )$ , where x contains only information publicly available at a round boundary. Throughout, $\Phi _ { \psi }$ denotes this frozen outcome model—a policydistribution-conditional win-probability teacher—rather than a policy-training value function or online critic. It is trained on complete-match trajectories, calibrated, and then frozen before all reward-training experiments; it is never updated from subsequent self-play trajectories. Freezing separates matchprogress estimation from policy optimization and prevents the reward annotation itself from becoming a moving target as the self-play policy changes.

The public-state restriction excludes hidden cards, private observations, future round outcomes, and policy-internal repre sentations unavailable to the acting agent. Train/validation/test partitions are constructed at the complete-match level so that different rounds from the same match cannot appear in different splits. The predictor uses a 27-dimensional publicboundary representation encoding team levels and gap, stage indicators, and other publicly known match-state variables. We fit an $\ell _ { 2 } \cdot$ -regularized logistic model and apply scalar Platt calibration on the validation split, which is also used for predictor development; there is no separate calibration partition. All reported predictor metrics are evaluated once on untouched test matches. The frozen artifact stores the feature transformation, coefficients, calibration parameters, clipping rule, and a cryptographic identifier. Opponent predictions are defined by construction as $\Phi _ { \mathrm { o p p } } ( x ) = 1 - \Phi _ { \mathrm { t e a m } } ( x )$ , so the corresponding progress signals are antisymmetric.

## C. Self-Play Optimization

Policy learning follows an iterative online self-play procedure rather than offline reinforcement learning. At iteration i, the current actor $\pi _ { i }$ generates new complete matches; after each match ends, the frozen predictor supplies $\Delta \Phi _ { k }$ and the corresponding round-level supervision targets are materialized. The learner minimizes the executed-action regression objective

$$
\mathcal { L } _ { \mathrm { M C } } ( \theta ) = \frac { 1 } { N } \sum _ { ( s , a , G ) } \left( Q _ { \theta } ( s , a ) - G \right) ^ { 2 } ,\tag{5}
$$

where G is either Eq. (2) or Eq. (4). Only the executed action is supervised by its realized round-level label; no counterfactual target is available for unexecuted legal actions.

Replacing $G ^ { \mathrm { b a s e } }$ by $G ^ { \mathrm { p r o g } } = G ^ { \mathrm { b a s e } } + \beta \Delta \Phi _ { k }$ shifts every within-round residual $ { \mathbf { b y } } - \beta \Delta \Phi _ { k }$ . Because $\Delta \Phi _ { k }$ varies with public match context even when $r _ { k } ^ { \mathrm { r a n k } }$ is held fixed, samples that share the same rank term but occur in different stages receive systematically different residual corrections, changing the finite-sample update direction at round boundaries relative to recovering the same dependence only through the delayed and widely shared $y _ { \mathrm { { m a t c h } } }$ . After locked arm-specific scaling, the progress component is calibrated to a controlled fraction of reference target RMS, so the intervention changes residual structure and teacher locality rather than merely rescaling the objective; we do not claim that this necessarily improves signalto-noise ratio in every stratum.

The Q-network contains 4,961,793 parameters. A residual MLP with LayerNorm maps the 638-dimensional state input to width 512 using two residual blocks; a GRU maps the 94-dimensional action-history sequence to width 256; and a residual MLP maps the 89-dimensional candidate-action input to width 256. Their representations are fused by two width-768 residual blocks followed by a shared linear Q head, with dropout 0.05. Both reward arms use the identical architecture and initialization.

![](images/63ae612f7ae553aa0ad231c8597584786d5aad01076609d0a4bc1db62f8c687e.jpg)  
Fig. 1. One online collect–annotate–train iteration. Each iteration first generates new complete matches with the current policy $\pi _ { i }$ . After a match ends, the frozen predictor Φ annotates each round boundary with $\Delta \Phi = \Phi ( x _ { \mathrm { a f t e r } } )$ $\Phi ( x _ { \mathrm { b e f o r e } } )$ , and the resulting round-level supervision target is broadcast to every executed action within that round. These round-level labels then supervise one epoch of policy training with a fresh optimizer state, yielding $\pi _ { i + 1 }$ for the next collection round. The predictor Φ remains frozen across all policy-training iterations and is not updated with the policy.

TABLE I  
COMPLETE-MATCH DATASET PARTITIONS. BOUNDARY ROWS INCLUDE BOTH TEAM-RELATIVE VIEWS.
<table><tr><td>SPLIT</td><td>MATCHES</td><td>BOUNDARIES</td></tr><tr><td>TRAIN</td><td>1,411</td><td>34,818</td></tr><tr><td>VAL. (PLATT)</td><td>199</td><td>4,886</td></tr><tr><td>TEST</td><td>390</td><td>9,568</td></tr><tr><td>TOTAL</td><td>2,000</td><td>49,272</td></tr></table>

The data lifecycle for iteration i is shown in Figure 1.

Each iteration collects 250 newly generated complete matches and performs one epoch over the resulting executedaction samples with batch size 2048 and learning rate $5 \times 1 0 ^ { - 5 }$ Training runs for eight iterations. ϵ-greedy exploration decreases from 0.05 to 0.01, and optimizer state is reset between iterations. Temporary reinforcement-learning trajectories are discarded after training.

## IV. TESTING PREDICTIVE MEANING AND OPTIMIZATION RELIABILITY

## A. Complete-Match Dataset

A dedicated complete-match collector was constructed to support both the match-outcome predictor and the reward experiment. The dataset used in the formal experiments was generated with the frozen DMC self-play Parent and contains 2,000 complete matches with zero aborted matches. The collection contains 24,636 rounds, 2,398,084 executed decisions, and 49,272 team-relative round-boundary samples. Matches contain 12.3 rounds on average, with a 95th percentile of 17 rounds and a maximum of 24 rounds.

Dataset partitions are defined by match\_id, and no match contributes boundaries to more than one partition. Each boundary record includes the public states before and after the round, signed level gain, final match outcome, and team-relative match information.

B. Stage I: Testing Predictive Meaning of State–Gain Interactions

Before constructing the progress reward, we measure whether the association between a fixed signed level gain and eventual victory changes with the pre-round public match state. A trivial comparison between level gain alone and full match state would be insufficient, since variables such as the current team levels naturally predict the final winner. A positive answer indicates that an explicit state–gain interaction carries additional predictive content that a round-local teacher can expose more directly than a shared terminal label alone.

We compare two nested predictive specifications. The constant-effect model is

$$
\mathrm { l o g i t } P ( y _ { \mathrm { m a t c h } } = 1 | x _ { k } , \Delta _ { k } ) = f ( x _ { k } ) + c \Delta _ { k } ,\tag{6}
$$

where $\Delta _ { k }$ is the signed round-level gain and the coefficient c is shared across all match states. The state-dependent model is

$$
\mathrm { l o g i t } P ( y _ { \mathrm { m a t c h } } = 1 | x _ { k } , \Delta _ { k } ) = f ( x _ { k } ) + c \Delta _ { k } + h ( x _ { k } , \Delta _ { k } ) ,\tag{7}
$$

where h contains preregistered interactions between round gain and pre-round match-state variables. Only variables available before the round are included in $x _ { k } ;$ variables determined by the round outcome are excluded to avoid post-treatment leakage.

We evaluate the two models on held-out complete matches using log loss and Brier score as primary metrics, with AUROC and calibration diagnostics as secondary measures. Confidence intervals are obtained by match-level resampling, keeping teamrelative views from the same match in the same cluster. For a distribution-level audit, we condition on realized signed level gain $\Delta _ { k } \in \{ - 3 , - 2 , - 1 , + 1 , + 2 , + 3 \}$ and stratify by pre-round level gap (leading / close / trailing), with Near-A status retained as an overlapping Boolean slice. Stage contrasts use match-clustered percentile bootstrap intervals. This is an observational stratification of realized outcomes rather than a same-deal intervention. A meaningful held-out improvement of Eq. (7) over Eq. (6) establishes predictive meaning for the state– gain interaction, but does not by itself establish optimization utility.

## C. Constructing the Frozen Predictive Probe

We train $\Phi _ { \psi }$ on the complete-match train split with Platt calibration on validation, without access to the final test matches. Predictor quality is evaluated by log loss, Brier score, reliability, stage-stratified calibration, team-view consistency, and later out-of-support rates under self-play. Model and calibration parameters are then frozen under a fixed artifact hash; out-of-support states use a deterministic fallback and are logged.

All formal reward-training arms are initialized from the same frozen DMC self-play Parent. This policy was originally initialized through supervised learning on human data and then improved in a separate pre-experiment self-play stage using the conventional round-win objective; the resulting iteration-4 deployment champion was frozen before the reward experiment. Human-data replay is disabled during formal training $\omega _ { \mathrm { B C } } =$ $w _ { \mathrm { h u m a n M C } } = 0 )$ , so human supervision enters only through the frozen initialization lineage.

The initial optimization-probe experiment contains one Baseline arm and two preregistered progress scales. Progress-10 and Progress-20 target auxiliary progress components equal to approximately 10% and 20% of the frozen reference target RMS, respectively, with locked constants $c _ { \mathrm { b a s e } } ~ = ~ 2 . 5 5 6 .$ $c _ { \mathrm { P 1 0 } } = 2 . 7 4 7 , c _ { \mathrm { P 2 0 } } = 2 . 9 4 8 , \beta _ { 1 0 } = 2 . 0 0 2 ,$ , and $\beta _ { 2 0 } = 4 . 0 0 5$ All three arms use the same Parent, network, DMC learner, exploration schedule, collection budget, training duration, and evaluation seeds; after training begins, each arm induces its own on-policy state distribution. We evaluate the reward under a DanZero-style DMC learner rather than the full DanZero+ PPO system. Progress-10 is selected only after the initial fiveseed optimization comparison; Progress-20 does not receive a final-holdout evaluation.

a) Training-protocol safeguards.: Preliminary Monte Carlo runs showed that bounded-target mean-squared regression on a preference-oriented scoring head can compress output scale and degrade action ranking. Formal arms therefore share the same Parent and training protocol, with the complete reward target as the principal manipulated treatment; checkpoints from preliminary harmful runs are excluded from subsequent selfplay collection.

Formal reward training proceeds in iterative collect–train cycles under the protocol of Section III-C. Scientific training lineage is maintained separately from deployment promotion. The initial scale-selection experiment uses five independent training seeds per arm and eight iterations of 250 completed matches per seed, for $3 \times 5 \times 8 \times 2 5 0 = 3 0 , 0 0 0$ completed training matches. After Progress-10 was frozen, a preregistered methodlevel replication trained five additional matched Baseline and Progress-10 lineages (seeds 6–10) under the same Parent, learner, hyperparameters, budget, and scientific-checkpoint selection rule, adding $2 \times 5 \times 8 \times 2 5 0 = 2 0 , 0 0 0$ completed matches. The combined training evidence therefore covers ten Baseline–Progress-10 seed pairs and 50,000 completed matches, while Progress-20 remains confined to the initial scale-selection stage.

Checkpoint selection uses 40 paired seeds to reject clearly harmful candidates and 200 disjoint paired seeds for confirmation; the scientific checkpoint is chosen by complete-match paired performance. The final evaluation set remains untouched during training, hyperparameter selection, and checkpoint confirmation. After Progress-10 is selected, the corresponding scientific checkpoint is evaluated on a locked final holdout of 1,000 paired seeds against both the Parent and the matched Baseline.

## E. Paired Team-Swap Evaluation

Playing strength is evaluated using complete matches under a paired team-swap protocol. For each evaluation seed, two matches are played under the same controlled conditions while swapping which team is controlled by the candidate policy. Let $m _ { i , A } , m _ { i , B } \in \{ - 1 , + 1 \}$ denote the candidate’s outcomes in the two team assignments. The paired outcome is

$$
\bar { d } _ { i } = \frac { m _ { i , A } + m _ { i , B } } { 2 } \in \{ - 1 , 0 , + 1 \} ,\tag{8}
$$

so $\bar { d } _ { i } = + 1$ denotes a double win, $\bar { d } _ { i } = - 1$ a double loss, and $\bar { d } _ { i } = 0$ a split pair. Our primary paired metric is

$$
\mathrm { M e a n D e l t a } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \bar { d } _ { i } = \frac { N _ { + } - N _ { - } } { n } ,\tag{9}
$$

where $N _ { + }$ and $N _ { - }$ are the numbers of double-win and double-loss pairs. When every individual match has a binary win/loss outcome, MeanDelta = 2 WinRate −1. Inference is performed over paired seeds because the two team-swapped matches are deliberately coupled by the evaluation design.

## F. Criterion for Optimization Reliability

We define optimization reliability at the level of independent training runs rather than evaluation games from one selected checkpoint. This follows deep-RL evaluation guidance that runto-run variability and finite-run uncertainty must be reported explicitly when comparing learning procedures [10], [11]. Increasing the number of evaluation games tightens uncertainty for a fixed checkpoint, but does not estimate the variability of policies produced by rerunning training. For training seed s, let $\Delta _ { s } ^ { \mathrm { p r o g } }$ and $\Delta _ { s } ^ { \mathrm { b a s e } }$ denote the paired performance of the progress and baseline policies under matched evaluation conditions, and define

$$
D _ { s } = \Delta _ { s } ^ { \mathrm { p r o g } } - \Delta _ { s } ^ { \mathrm { b a s e } } .\tag{10}
$$

We report the individual $D _ { s }$ values, their mean and median, the fraction of positive training seeds, and an uncertainty interval over training seeds. The original seeds 1–5 use the 200-pair confirmation estimates. The preregistered seeds 6–10 use a fresh 1,000-pair common-random-number evaluation pool, and their Student-t interval is the primary method-level replication inference; pooling seeds 1–10 is a prespecified secondary summary.

Following the original confirmation analysis for Progress-10, we conducted two preregistered read-only audits without replacing any frozen scientific checkpoint. First, the five matched Progress-10 and Baseline checkpoints were re-evaluated on a fresh common-random-number pool of 500 paired seeds using both the parent-relative estimand in Eq. (10) and direct Progress-versus-Baseline evaluation. Second, checkpoint selection stability was assessed on another disjoint 500-pair pool among the three confirmation-eligible candidates from each trajectory, using joint paired-seed resampling and five-fold cross-fitting. These audits diagnose replication and selection stability without redefining the frozen scientific checkpoints.

We further conduct a post-hoc fixed-checkpoint horizon diagnostic over all ten matched lineages. For every training seed s and iteration $t \in \{ 0 , \ldots , 7 \}$ , a fresh common-randomnumber pool of 200 paired team-swap seeds estimates the direct effect

TABLE II  
HELD-OUT INCREMENTAL PREDICTIVE-VALUE TEST. M0 USES A CONSTANT GAIN EFFECT AND M1 ADDS STATE–GAIN INTERACTIONS. LOWER IS BETTER FOR LOG LOSS AND BRIER SCORE; HIGHER IS BETTER FOR AUROC.
<table><tr><td>MODEL</td><td>LOG LOSS</td><td>BRIER</td><td>AUROC</td></tr><tr><td>M0</td><td>0.54458</td><td>0.18431</td><td>0.79633</td></tr><tr><td>M1</td><td>0.53983</td><td>0.18370</td><td>0.79572</td></tr></table>

$$
D _ { s , t } = \mathrm { M e a n D e l t a } ( P _ { s , t } , B _ { s , t } ) .\tag{11}
$$

We summarize the mean trajectory, the late contrast $L _ { s } =$ $D _ { s , 7 } - D _ { s , 4 }$ , and a per-seed linear slope over iterations 4–7. This analysis is diagnostic rather than confirmatory and does not select or replace any checkpoint.

Finally, an exploratory five-seed scale/shuffle ablation separates the conventional Baseline $( B _ { 0 } )$ , a scale-matched Baseline using the Progress-10 divisor $( B _ { S } )$ , a semantically shuffled progress control at that scale $( P _ { S } )$ , and the true Progress-10 treatment $( P _ { T } )$ . Its contrasts are reported as mechanism diagnostics with exploratory seed-level intervals, not as equivalence tests or replacements for the preregistered method-level replication.

## G. Diagnostics and Statistical Reporting

We monitor optimization health through Q-value summaries, within-state action-score dispersion, round-target scale, parent divergence, illegal-action checks, and predictor out-of-support rates. These diagnostics help interpret training instability but do not replace complete-match paired evaluation as the primary measure of playing strength. For a fixed checkpoint comparison, we report positive/split/negative paired counts, MeanDelta with a bootstrap confidence interval, and raw match win/loss counts. For the method-level comparison, the independent unit is the training seed rather than the individual evaluation match, so cohort-specific and pooled ten-seed results are reported separately from the uncertainty of a single selected checkpoint.

## V. RESULTS

A. Predictive Meaning: State–Gain Interactions Improve Held-Out Fit

Table II compares the constant-gain and state-dependent specifications on 9,568 team-relative boundaries from 390 held-out matches. Adding pre-round state–gain interactions yields a small but statistically supported improvement in held-out log loss: the match-clustered bootstrap change (M1 minus M0) is −0.00476, with a 95% confidence interval of [−0.00766, −0.00201].

The secondary Brier and AUROC comparisons do not show corresponding reliable improvements. We therefore use predictively meaningful in a deliberately narrow sense: the state–gain interaction adds held-out information under log loss, not that it improves every predictive metric.

CONFIRMATION PROTOCOL AND IS THEREFORE 0 FOR BASELINE BYDEFINITION. PARENT-RELATIVE AGGREGATES ARE REPORTED IN THE TEXT

bars: mean difference: whiskers: match-clustered 95% Cl (near-A is an overlapping slice, not a 4th gap stage)  
![](images/59a2f10557b3483254d554c2d4a5e3664049e38d96536b8aa42b12d61da6b4e4.jpg)  
Fig. 2. Mean ∆Φ vs. signed rank gain by gap stage (top) and near-A contrast (bottom; overlapping slice, not a fourth exclusive stage). Error bars: matchclustered 95% bootstrap CIs. Stage dependence is strongest at larger |gain|.

Figure 2 summarizes mean ∆Φ by signed gain and gap stage, together with the overlapping Near-A contrast. The direction is semantically consistent: every negative-gain stratum has negative mean progress and every positive-gain stratum has positive mean progress. The magnitude is state dependent: 12 of the 18 pairwise gap-stage contrasts have match-clustered 95% intervals excluding zero, with separation concentrated in larger round outcomes $( | \Delta _ { k } | \geq 2 )$ rather than appearing as a uniform offset. All six Near-A contrasts also exclude zero, but their direction depends on gain size and the current gap stage. Empirical match outcomes retain the same stage ordering at fixed large gains, reinforcing that the predictive value of a round outcome depends on the pre-round state.

## B. Measurement Check: The Frozen Predictor Tracks Match Outcomes

On the held-out team-0 view (n = 4,784 boundaries), the calibrated public-state predictor obtains log loss 0.59497, Brier score 0.20462, AUROC 0.75064, and expected calibration error 0.05064 (Table III).

The reliability curve in Figure 3 shows the largest deviations in sparse upper-probability bins. Opponent-view probabilities are defined as complements, so team-symmetry error is zero by construction.

## C. Optimization Probe: Initial Scale Selection

Table IV summarizes the initial five-seed optimization comparison used to select the progress scale before accessing the final holdout. On the confirmation protocol, the aggregate

TABLE III  
FROZEN Φ PERFORMANCE ON THE UNTOUCHED TEST SPLIT.
<table><tr><td>METRIC</td><td>TEST VALUE</td></tr><tr><td>LOG LOSS</td><td>0.59497</td></tr><tr><td>BRIER SCORE</td><td>0.20462</td></tr><tr><td>AUROC</td><td>0.75064</td></tr><tr><td>ECE</td><td>0.05064</td></tr><tr><td>CLIPPING RATE</td><td>0.000209</td></tr><tr><td>FALLBACK RATE</td><td>0</td></tr></table>

![](images/1abe2cdf1298119b395d56b45363c33a8d89c4cd0d635a4833effcb19790e737.jpg)  
Fig. 3. Reliability diagram for frozen Φ on the test split $( n = 4 , 7 8 4 ) ;$ 95% Wilson intervals per bin, with bin counts below. Predictions track the diagonal except in sparse upper bins $( \mathrm { A U R O C } = 0 . 7 5 1 , \mathrm { E C E } = 0 . 0 5 1 )$ . Platt scaling was fit on validation only.

## TABLE IV

FIVE-SEED PROGRESS-SCALE SELECTION BEFORE THE FINAL HOLDOUT.“AGG. $\Delta ^ { \prime \prime }$ IS THE PROGRESS-MINUS-BASELINE MEANDELTA ON THE

(0.120 FOR BASELINE; 0.154 FOR PROGRESS-10). FINAL MARKS ADMISSION TO THE LOCKED HOLDOUT.
<table><tr><td>ARM</td><td>β</td><td> $c _ { a }$ </td><td></td><td>POS. AGG.∆ FINAL</td><td></td></tr><tr><td>BASELINE</td><td></td><td>02.556</td><td></td><td>0</td><td>NO</td></tr><tr><td>P10</td><td>2.002</td><td>2.747</td><td>3/5</td><td>+0.034</td><td>YES</td></tr><tr><td>P20</td><td>4.005</td><td>2.948</td><td>2/5</td><td>+0.004</td><td>NO</td></tr></table>

parent-relative MeanDelta was approximately 0.120 for Baseline and 0.154 for Progress-10. The corresponding Progressminus-Baseline aggregate effect was +0.034 for Progress-10 and only +0.004 for Progress-20.

Progress-10 achieved the larger aggregate Progress-minus Baseline improvement and was positive for three of five matched training seeds, so it was selected before the final holdout under the preregistered rule.

## D. Optimization Reliability Breaks Across Training Seeds

Predictive meaning would translate into optimization reliability only if the selected Progress-10 scale yielded a consistently positive method-level effect across independent training seeds. In the original confirmation analysis, three of five Progress-10 seed-level effects were positive, but their mean was only 0.034 with a 95% bootstrap interval of [−0.080, 0.147]. Thus, the five training seeds already exhibit clear heterogeneity and do not establish a positive expected method-level effect, despite the independently verified predictive content of the injected signal.

TABLE V  
METHOD-LEVEL PROGRESS-10 EFFECT ACROSS TRAINING-SEED COHORTS.THE ORIGINAL INTERVAL IS A SEED BOOTSTRAP; THE PREREGISTEREDREPLICATION AND POOLED INTERVALS USE STUDENT-t INFERENCE OVERTRAINING SEEDS. POSITIVE COUNTS ARE DESCRIPTIVE.
<table><tr><td>COHORT</td><td>MEAN  $D _ { s }$ </td><td>95% CI</td><td>POSITIVE</td></tr><tr><td>SEEDS 1-5</td><td>+0.034</td><td> $\left\lceil - 0 . 0 8 0 , 0 . 1 4 7 \right\rceil$ </td><td> $3 / 5$ </td></tr><tr><td>SEEDS 6-10</td><td>-0.056</td><td> $[ \dot { - } 0 . 0 8 3 , - 0 . 0 3 \dot { 0 } ]$ </td><td> $0 / 5$ </td></tr><tr><td>SEEDS 1-10</td><td>-0.030</td><td> $[ - 0 . 0 8 4 , 0 . 0 2 5 ]$ </td><td> $3 / 1 0$ </td></tr></table>

TABLE VI

SEED-LEVEL PROGRESS-10 EFFECTS UNDER THE ORIGINAL CONFIRMATION ANALYSIS, INDEPENDENT 500-PAIR REPLICATION, AND CROSS-FITTED CHECKPOINT-SELECTION AUDIT. “PARENT-REL.” IS THE DIFFERENCE BETWEEN PROGRESS-VERSUS-PARENT AND BASELINE-VERSUS-PARENT MEANDELTA; “DIRECT” EVALUATES PROGRESS-10 AGAINST ITS MATCHED BASELINE. CROSS-FIT VALUES ARE HELD-OUT ESTIMATES OF THE ORIGINAL THREE-CANDIDATE SELECTION RULE AND DO NOT REPLACE THE FROZEN SCIENTIFIC CHECKPOINTS.
<table><tr><td>SEED</td><td>ORIGINAL</td><td>PARENT-REL.</td><td>DIRECT</td><td>CROSS-FIT</td></tr><tr><td>1</td><td>+0.145</td><td>+0.128</td><td>+0.128</td><td>+0.14</td></tr><tr><td>2</td><td>+0.135</td><td>+0.182</td><td>+0.074</td><td>+0.02</td></tr><tr><td>3</td><td>-0.125</td><td>-0.102</td><td>-0.160</td><td>-0.13</td></tr><tr><td>4</td><td>+0.155</td><td>-0.044</td><td>-0.004</td><td>-0.05</td></tr><tr><td>5</td><td>-0.140</td><td>-0.038</td><td>-0.068</td><td>-0.05</td></tr></table>

The preregistered replication is less favorable. Across seeds 6–10, none of the five Progress-10 effects is positive; the mean is −0.056 with a Student-t 95% confidence interval of [−0.083, −0.030]. This batch therefore rejects a positive mean effect and excludes an improvement of at least +0.05 MeanDelta under the preregistered practical-effect criterion. As a secondary analysis, pooling all ten training seeds gives mean effect −0.030 with 95% confidence interval [−0.084, 0.025]. The pooled interval contains zero, so the combined evidence establishes neither a positive nor a negative population-average effect; importantly, it provides no support for a reliable positive method-level gain. This is the paper’s central predictive– optimization gap: held-out predictive value survives its own test, while a reproducible positive policy effect does not survive method-level replication.

As shown in Table VI, a fresh 500-pair replication preserves the original sign pattern for four of five seeds, and direct Progress-versus-Baseline evaluation confirms both positive and negative lineages. Cross-fitted selection among confirmationeligible candidates preserves the negative effects for seeds 3 and 5, so checkpoint selection does not explain the overall positive-versus-negative structure.

We additionally audited selection stability among the three confirmation-eligible checkpoints from each trajectory. Most selections were stable, with Progress-10 seeds 3–4 and Baseline seed 2 less so; five-fold cross-fitting nonetheless preserved the negative Progress-10 effects for seeds 3 and 5.

TABLE VII  
LOCKED PAIRED TEAM-SWAP EVALUATION OF THE SELECTED PROGRESS-10 CHECKPOINT ON 1,000 PAIRS. W/S/L ARE DOUBLE-WIN, SPLIT, AND DOUBLE-LOSS COUNTS; MD IS MEANDELTA.
<table><tr><td>COMPARISON</td><td>W/S/L</td><td>MD</td><td>95% CI</td></tr><tr><td>VS. PARENT</td><td>369/516/115</td><td>0.254</td><td>[0.214, 0.294]</td></tr><tr><td>VS. BASELINE</td><td>293/557/150</td><td>0.143</td><td>[0.103, 0.183]</td></tr></table>

## E. A Successful Checkpoint Does Not Establish Reliability

Against this heterogeneous ten-seed background, the locked holdout result for the selected Progress-10 scientific checkpoint provides an instructive counterpoint rather than a methodaverage claim. This frozen checkpoint is stronger than both the common DMC Parent and its matched Baseline on the locked 1000-pair team-swap evaluation (Table VII). Against the matched Baseline, Progress-10 records 293 double wins, 557 split pairs, and 150 double losses, giving MeanDelta 0.143 with 95% confidence interval [0.103, 0.183], corresponding to an approximate single-match win rate of 57.2% over the 2,000 individual matches. Against the Parent, MeanDelta is 0.254 with interval [0.214, 0.294] and raw win rate 0.627. These results establish that the signal can be beneficial in a particular lineage, but they do not show that the intervention is optimization reliable across training randomness.

## F. The Gap Is Not Explained by Selection or Training Horizon

Post-hoc audits did not identify a simple measured mechanism separating positive and negative Progress-10 lineages. Frozen-predictor log loss and Brier score were similar across the groups, and the realized progress-component scale remained close to its nominal 10% contribution. Coarse Q dispersion and top-action margins were also similar. On a fixed 2,000- state bank, matched Progress-10 and Baseline policies retained Spearman action-ranking correlation of approximately 0.94–0.95, while controlled $K \ = \ 8$ branch rollouts with Parent continuation found no systematic advantage for the action disagreements of positive lineages; the corresponding confidence intervals included zero. An offline per-iteration analysis likewise found only isolated differences in topaction divergence, with no persistent consecutive-iteration separation satisfying the preregistered gate. Together with the selection audit, these null results indicate that evaluation noise, checkpoint selection, predictor drift, reward-scale drift, and the measured one-step ranking statistics do not individually explain the seed pattern; they do not exclude unmeasured multi-step self-play interactions. The predictive–optimization gap therefore cannot be reduced to one of these measured artifacts.

The all-checkpoint horizon diagnostic in Figure 4 covers all ten matched Progress-10–Baseline training-seed pairs and does not support insufficient training duration as an explanation for the missing Progress-specific effect. Across these ten pairs, the direct Progress-minus-Baseline trajectories repeatedly cross zero, their peaks occur at different iterations, and intermediate gains are generally not retained at the fixed endpoint. The mean iteration-4-to-7 contrast is −0.0175 $( 9 5 \% \mathrm { ~ C I ~ } [ - 0 . 1 1 6 0 , 0 . 0 9 2 5 ] )$ ), and the mean late-stage slope is −0.0116 per iteration ([−0.0341, 0.0152]). Although the iteration-7 pointwise interval is negative, its simultaneous band over the eight inspected horizons still includes zero. Accordingly, the curves should not be interpreted as proving monotonic deterioration; they show substantial lineagedependent fluctuation and no common late-stage tendency for Progress-10 to overtake matched Baseline training.

<table><tr><td></td><td>individual training seeds across-seed mean</td><td>simultaneous 95% band fixed endpoint (iter. 7)</td></tr></table>

![](images/96a3def02a3203853a87ab69cbc09310dc8f1819119a9d859549a83b156e03bc.jpg)

![](images/f4a52e2b44e3595ca5f81b830dfd029e9723ab2178cd3c4eee500d20b5685083.jpg)

![](images/df547150e53f2aff8e50e3100066cdc11d64870cd09f0777942ac3b9d0af224a.jpg)  
Fig. 4. Training-horizon diagnostics across ten matched self-play seeds. Panels A and B show the paired MeanDelta of Progress-10 and Baseline, respectively, against the frozen Parent; Panel C reports direct Progress-10-versus-Baseline evaluation and is therefore the relevant treatment-effect trajectory. Each matchup uses 200 paired team-swap seeds from the diagnostic common-random-number pool. Thin gray lines denote individual training seeds, thick black lines denote across-seed means, gray regions show simultaneous 95% confidence bands over the eight inspected iterations, and the dashed line marks the fixed endpoint at iteration 7. Both policies improve in absolute strength relative to Parent, whereas their direct comparison fluctuates around zero and shows no common late-stage Progress advantage. This post-hoc diagnostic is not used for checkpoint selection.

The absence of a reliable relative benefit is compatible with continued absolute learning. The upper panels show that both Progress-10 and Baseline improve relative to the frozen Parent after early fluctuations, but the direct comparison in Panel C shows no corresponding late-stage Progress-specific improvement. Longer training may therefore increase absolute policy strength without resolving the absence of a reliable Progress-specific treatment effect.

The exploratory scale/shuffle ablation also fails to isolate a stable semantic benefit. The true-versus-shuffled progress contrast is $P _ { T } - P _ { S } = - 0 . 0 3 2$ with 95% interval $[ - 0 . 1 8 2 , 0 . 1 1 7 ]$ the scale-only contrast is $B _ { S } - B _ { 0 } = + 0 . 0 3 6$ with interval [−0.119, 0.190]. The shuffled-noise contrast $P _ { S } \mathrm { ~ - ~ } B _ { S } \mathrm { ~ = ~ }$ −0.002 and the same-scale true-progress contrast $P _ { T } - B _ { S } =$ −0.006 are both near zero. This separate scale/shuffle ablation uses five seeds per condition and has wide intervals; it detects no stable semantic, scale, or noise advantage and does not establish statistical equivalence among the four conditions.

## VI. DISCUSSION

## A. Predictive Meaning Is Not Optimization Reliability

The main result is an asymmetry between two kinds of evidence. State–gain interactions improve held-out log loss, the frozen predictor tracks eventual match outcomes, and its progress estimates vary coherently with public match context. These results establish predictive meaning under the stated measurement criteria. In contrast, the preregistered replication mean is negative, the pooled ten-seed interval includes zero, and relative training trajectories repeatedly change sign. The same signal therefore fails the stronger criterion of a reproducibly positive optimization effect under the studied DMC protocol.

The successful seed-1 checkpoint is part of this finding rather than an exception to discard. It shows that progress augmentation can produce a substantially stronger policy in one lineage, ruling out the simple interpretation that the signal is uniformly harmful or inert. At the same time, its coexistence with negative and null lineages demonstrates why selected-checkpoint efficacy and method-level reliability answer different questions. The former establishes possibility conditional on a realized training path; the latter asks whether a new training run should be expected to benefit. We therefore interpret the locked result as selection-conditioned efficacy, not as an unconditional estimate of the training procedure’s expected effect.

This gap is plausible because prediction and optimization act on different objects. The predictive test evaluates whether a feature improves outcome estimates under held-out samples from a fixed data-generating distribution, whereas downstream decision quality depends on how prediction errors interact with the optimization objective [1], [2]. The optimization intervention changes finite-sample residuals for executed actions, which changes the policy, which in turn changes subsequent self-play data. This policy–data feedback is structurally related to performative learning, where deployed decisions alter later data or dynamics [3], [4]. Round-broadcast supervision, approximation error, and stochastic trajectory collection can therefore transform a valid descriptive signal into lineagedependent updates. Our diagnostics do not identify a single causal mechanism for that transformation, but they exclude several simpler explanations and show that training longer does not produce a common late-stage advantage.

The practical implication is a two-stage standard for outcomepotential supervision. Predictor validation should establish that the proposed signal contains the intended information, while independent training-seed replication should establish that the learner converts that information into policy gains reliably. Neither predictor metrics nor a locked evaluation of one selected checkpoint can replace the second stage, consistent with established concerns about variability and finite-run uncertainty in deep RL evaluation [10], [11]. This distinction is especially important in online self-play, where optimization changes the distribution on which later supervision is generated.

## B. Scope and Limitations

The conclusion is deliberately bounded to the present construction. First, $\Phi _ { \psi }$ is predictive rather than causal, so $\Delta \Phi$ should not be interpreted as action-level credit. Second, the augmentation has no classical PBRS guarantee, and every executed action within a round shares the same label. Third, the frozen teacher may face distribution shift under later self-play policies. Fourth, ten Baseline–Progress seed pairs remain a modest sample for characterizing a broad training-run population, and the completed diagnostics do not identify which random source determines whether a lineage benefits. Finally, the experiment covers one game and one round-broadcast, executed-action DMC learner; it does not establish that predictive meaning and optimization reliability will diverge under policy-gradient, counterfactual, or policyinvariant constructions.

## VII. CONCLUSION

We used match-progress supervision in GuanDan as a controlled case study of a commonly implicit inference: that an outcome-predictive intermediate signal should also be a reliable optimization signal. The evidence separates these claims. Public match context adds held-out predictive value, and progress augmentation can yield a strong selected checkpoint, yet its policy effect is heterogeneous across ten training seeds and shows no common late-stage advantage. Thus, under the studied self-play protocol, predictive meaning does not imply optimization reliability. Determining which credit-assignment or policy-optimization mechanisms can convert valid predictive information into reproducible gains is the next question, not an assumption that predictor quality has already answered.

## REFERENCES

[1] A. N. Elmachtoub and P. Grigas, “Smart “predict, then optimize”,” Management Science, vol. 68, no. 1, pp. 9–26, 2022.

[2] B. Wilder, B. Dilkina, and M. Tambe, “Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, 2019, pp. 1658–1665.

[3] J. Perdomo, T. Zrnic, C. Mendler-Dunner, and M. Hardt, “Performative¨ prediction,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 7599–7609.

[4] D. Mandal, S. Triantafyllou, and G. Radanovic, “Performative reinforcement learning,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 23 642–23 680.

[5] C. Li, S. Yang, C. Zhan, Z. Ge, Y. Hu, B. Bao, X. Chen, and Y. Gao, “OpenGuanDan: A large-scale imperfect information game benchmark,” arXiv preprint arXiv:2602.00676, 2026.

[6] D. Zha, J. Xie, W. Ma, S. Zhang, X. Lian, X. Hu, and J. Liu, “DouZero: Mastering DouDizhu with self-play deep reinforcement learning,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 12 333–12 344, arXiv:2106.06135.

[7] Y. Lu, J. Zhao, Y. Zhao, W. Zhou, and H. Li, “DanZero: Mastering GuanDan game with reinforcement learning,” in 2023 IEEE Conference on Games (CoG). IEEE, 2023, pp. 1–8, arXiv:2210.17087.

[8] Y. Zhao, Y. Lu, J. Zhao, W. Zhou, and H. Li, “DanZero+: Dominating the GuanDan game through reinforcement learning,” IEEE Transactions on Games, vol. 16, no. 4, pp. 914–926, 2024, arXiv:2312.02561.

[9] A. Patwa, “Self-play reinforcement learning under imperfect information in Big 2,” arXiv preprint arXiv:2605.28863, 2026.

[10] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, “Deep reinforcement learning that matters,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, 2018.

[11] R. Agarwal, M. Schwarzer, P. S. Castro, A. C. Courville, and M. Bellemare, “Deep reinforcement learning at the edge of the statistical precipice,” in Advances in Neural Information Processing Systems, vol. 34, 2021, arXiv:2108.13264.

[12] D. Balduzzi, M. Garnelo, Y. Bachrach, W. Czarnecki, J. Perolat, M. Jaderberg, and T. Graepel, “Open-ended learning in symmetric zerosum games,” in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 97. PMLR, 2019, pp. 434–443.

[13] E. Pignatelli, J. Ferret, M. Geist, T. Mesnard, H. van Hasselt, O. Pietquin, and L. Toni, “A survey of temporal credit assignment in deep reinforcement learning,” Transactions on Machine Learning Research, 2024, arXiv:2312.01072.

[14] J. A. Arjona-Medina, M. Gillhofer, M. Widrich, T. Unterthiner, J. Brandstetter, and S. Hochreiter, “RUDDER: Return decomposition for delayed rewards,” in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 13 566–13 577, arXiv:1806.07857.

[15] Z. Ren, R. Guo, Y. Zhou, and J. Peng, “Learning long-term reward redistribution via randomized return decomposition,” in International Conference on Learning Representations, 2022, arXiv:2111.13485.

[16] C.-C. Hung, T. Lillicrap, J. Abramson, Y. Wu, M. Mirza, F. Carnevale, A. Ahuja, and G. Wayne, “Optimizing agent behavior over long time scales by transporting value,” Nature Communications, vol. 10, no. 1, p. 5223, 2019, arXiv:1810.06721.

[17] S. Hu, Y. Dai, X. Han, Z. Fang, Y. Zhao, S. T. W. Kwong, and Y. Fang, “Self-induced outcome potential: Turn-level credit assignment for agents without verifiers,” arXiv preprint arXiv:2605.04984, 2026.

[18] L. Tao, B. Peng, W. Yao, T. Ge, H. Cheng, M. H. Wang, J. Gao, and S. Li, “TRACE: Turn-level reward assignment via credit estimation for long-horizon agents,” arXiv preprint arXiv:2607.13988, 2026.

[19] Y. Xie, N. Thomas, N. Hansen, Y. Fu, L. E. Li, and X. Wang, “TIPS: Turn-level information-potential reward shaping for search-augmented LLMs,” arXiv preprint arXiv:2603.22293, 2026.

[20] A. Y. Ng, D. Harada, and S. J. Russell, “Policy invariance under reward transformations: Theory and application to reward shaping,” in Proceedings of the Sixteenth International Conference on Machine Learning. Morgan Kaufmann, 1999, pp. 278–287.

[21] X. Lu, H. M. Schwartz, and S. N. Givigi, Jr., “Policy invariance under reward transformations for general-sum stochastic games,” Journal ofArti ficial Intelligence Research, vol. 41, pp. 397–406, 2011, arXiv:1401.3907.

[22] M. Zhou, Z. Liu, P. Sui, Y. Li, and Y. Y. Chung, “Learning implicit credit assignment for cooperative multi-agent reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 33, 2020, arXiv:2007.02529.

[23] T. Mesnard, T. Weber, F. Viola, S. Thakoor, A. Saade, A. Harutyunyan, W. Dabney, T. Stepleton, N. Heess, A. Guez, E. Moulines, M. Hutter,<sup>´</sup> L. Buesing, and R. Munos, “Counterfactual credit assignment in modelfree reinforcement learning,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 139. PMLR, 2021, pp. 7654–7664, arXiv:2011.09464.