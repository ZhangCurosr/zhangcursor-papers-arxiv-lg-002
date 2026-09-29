# MaPP: A Unified Marginalized Posterior-Predictive Framework for Data-Efficient RLVR

Yangyang Ren<sup>1,2∗</sup> Haodong Zhu<sup>1,2∗</sup> Sheng Xu<sup>3†</sup> Yanjing Li<sup>4†</sup> Nikolai Yu. Zolotykh<sup>5</sup> Wentao Zhang<sup>2,6</sup> Baochang Zhang<sup>1,7</sup>

<sup>1</sup>Beihang University <sup>2</sup>Zhongguancun Academy <sup>3</sup>Communication University of China <sup>4</sup>Nanyang Technological University <sup>5</sup>Lobachebsky University <sup>6</sup>Peking University <sup>7</sup>Hangzhou Innovation Institute of Beihang University

## Abstract

Reinforcement learning with verifiable rewards (RLVR) enhances the reasoning capabilities of large language models (LLMs), but at the expense of significant computational overhead due to compute-intensive rollout processes and frequent policy updates. Online prompt selection, a widely adopted strategy for improving training efficiency, maintains per-prompt Bayesian posteriors to predict prompt difficulty and prioritize informative prompts before committing rollout budget. However, these methods assess prompt informativeness without accounting for how reliably learning signal is extracted from sampled responses. In GRPObased RL algorithms, the realized advantage of a response depends not only on its own outcome, but also on the randomly sampled outcomes of its peers through group normalization. Our experimental and theoretical analysis show that the resulting uncertainty in group composition induces composition noise, a non-vanishing variance component that creates an irreducible lower bound on gradient estimation error. Consequently, the standard group-relative advantage fails to faithfully characterize response-level utility, degrading gradient estimation and thereby impairing downstream prompt selection. To address this issue, we propose a unified Marginalized Posterior-Predictive framework (MaPP) for dataefficient RLVR, which first denoises response-level advantage estimation and then improves prompt selection using a shared Beta posterior. Specifically, for each response, MaPP replaces the standard group-relative advantage with a compositioninvariant intrinsic advantage via closed-form Beta–Binomial marginalization. This yields a closed-form posterior-predictive advantage estimator whose error provably diminishes as the posterior concentrates. Then, building on the same posterior, MaPP derives an uncertainty-aware prompt selection score that more faithfully characterizes prompt informativeness, improving data efficiency without additional rollout cost. Experiments across mathematics, planning, and visual geometry on five model backbones show that MaPP consistently outperforms GRPO and strong selection baselines, achieving up to +2.45 average accuracy improvement over the strongest baseline under the same rollout budget, a new state-of-the-art.

## 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) is a key post-training paradigm for enhancing LLM reasoning [Shao et al., 2024, Yu et al., 2025, Guo et al., 2025, Jaech et al., 2024, Yang et al., 2025]. Among RLVR algorithms, GRPO [Shao et al., 2024] is most widely adopted, eliminating a learned value network by estimating advantages via within-group normalization of G responses per prompt. However, RLVR incurs high computation and memory costs due to intensive rollouts for policy evaluation and updates [Zheng et al., 2025c, Lin et al., 2025]. Thus, improving utility per fixed rollout budget is a central challenge in scaling RLVR.

![](images/ed4e0375ad6abb829ecc419d9ff4333a4ec0ac7e1344b5b0da737b7947ded3c5.jpg)  
(a) Test Accuracy vs. Rollout Cost

![](images/6ba59284e8895d8bf25fbfca5ecf92254c8399336ebe8a3efbac6f93e08ec59a.jpg)  
(b) Composition Noise in GRPO Advantages  
Figure 1: On the CountDown dataset using the Qwen3-4B model, (a) MaPP achieves higher accuracy than all selection methods with the same rollout cost, while requiring 80% fewer rollouts than DS to reach comparable performance. (b) Group-composition uncertainty induces composition noise in GRPO (group size $G { = } 8 ;$ mid-training checkpoint of MoPPS; 100 prompts $\times 5 0$ groups): under the same policy checkpoint, identical correct (red) or incorrect (blue) response receives widely varying advantages depending on group composition (±1 std shaded). The noise persists across all non-degenerate difficulty levels, e.g., $\Delta = 2 . 2 2$ at $\gamma \approx 0 . 1 5$ and $\Delta = 1$ .62 at $\gamma \approx 0 . 8$

Existing methods improve training efficiency by selecting prompts for rollout budget. Uniform sampling is inefficient: overly easy or hard prompts cause degenerate groups (all correct or incorrect), wasting rollouts on zero gradient [Bae et al., 2026, Chen et al., 2025, Zeng et al., 2025]. Only intermediate-difficulty prompts provide informative signals. Dynamic Sampling [Yu et al., 2025] oversamples a candidate set, rolls out all, then discards degenerate ones—but the rollouts on discarded prompts are already spent [Zheng et al., 2025c].

To avoid this waste, prediction-based methods [Qu et al., 2026a, Zheng et al., 2025b, Zhou et al., 2026] estimate prompt difficulty before rollout and skip likely degenerate prompts entirely, achieving comparable or superior performance with significantly fewer rollouts, as shown in Fig. 1 (a). However, conventional methods which select prompt before rollout do not by itself determine how reliably learning signal is extracted from the responses sampled afterward. In GRPO-based RLVR [Shao et al., 2024], both prompt selection and policy optimization are driven by group-level statistics formed from the sampled responses. For non-degenerate prompts, these counts are not fixed properties of the prompt itself, but random rewards with uncertainty induced by the current policy’s latent success rate under a finite number of sampled responses. As a result, even for the same response on the same prompt under the same policy, the realized advantage can deviate substantially depending on which peers happen to be drawn. Fig. 1 (b) visualizes this effect: the same correct or incorrect response receives widely varying advantages across different sampled groups, despite sharing the same prompt and policy. We refer to this deviation, induced by uncertainty in group composition, as composition noise. This composition noise degrades gradient estimation and, in turn, limits how faithfully prompt informativeness can be assessed for downstream selection.

Crucially, the information needed to account for this noise already exists in current selection pipelines: the per-prompt Beta posterior maintained over the latent pass rate $\gamma _ { \tau }$ tracks exactly the quantity that governs group-composition uncertainty. To leverage this observation, we propose a Marginalized Posterior-Predictive framework (MaPP), which first denoises response-level advantage estimation and then improves prompt selection using a shared Beta posterior. The key insight is that the GRPO advantage conditions on a single realized group composition, whereas the true learning signal of a response is determined by its correctness and the prompt’s intrinsic difficulty alone. The first component of MaPP, which we call MaPP-AD, denoises the realized GRPO advantage via closedform Beta–Binomial marginalization, recovering a composition-invariant intrinsic advantage. This yields a closed-form posterior-predictive advantage estimator whose error provably diminishes as the posterior concentrates. Building on the same posterior, the second component of MaPP, which we call MaPP-PS, derives a marginalized posterior-predictive, uncertainty-aware prompt selection method that more accurately characterizes prompt informativeness. Both components are modular and can be layered on top of existing selection pipelines. Our contributions are threefold:

![](images/cdf3b5f8d330ec3cfe6b31faaedb82301adfa06ad511a2c56d8283d2e872626e.jpg)  
Figure 2: Overview of the MaPP framework. A shared Beta posterior $( \alpha _ { t } ^ { \tau } , \beta _ { t } ^ { \tau } )$ , updated online from rollout history, serves as the foundation for two closed-form components. (A) MaPP-AD marginalizes $a _ { j } ^ { \star } ( \gamma )$ over a leave-one-out posterior, replacing the composition-dependent GRPO advantage with composition-invariant weights. Both components reuse the same posterior at no additional rollout cost. (B) MaPP-PS marginalizes $V _ { G } ( \gamma )$ over the full posterior for uncertaintyaware prompt selection, replacing the point-estimate scoring used by existing methods.

• We identify composition noise, a within-rollout variance component in the GRPO advantage induced by uncertainty in group composition and show that it degrades gradient estimation and thereby limits downstream prompt selection (§3.2).

• We propose MaPP, a unified posterior-predictive framework with two closed-form components: MaPP-AD, which denoises the realized GRPO advantage via posterior-predictive marginalization, and MaPP-PS, which uses the same shared Beta posterior to derive a more faithful prompt-selection score (§4).

• Experiments across three domains and five model backbones show that MaPP consistently outperforms GRPO and strong selection baselines, achieving up to +2.45 average accuracy improvement over the strongest baseline with the same rollout budget.

## 2 Related Work

## 2.1 RL Finetuning for Large Language Models

RLVR is a dominant post-training paradigm for reasoning LLMs [Guo et al., 2025, Shao et al., 2024, Jaech et al., 2024]. GRPO [Shao et al., 2024] eliminates PPO’s [Schulman et al., 2017] learned value network via within-group normalization, setting the template for many follow-ups. Variants address specific failure modes: Dr.GRPO [Liu et al., 2025] removes length normalization and std scaling, DAPO [Yu et al., 2025] adds asymmetric clipping and dynamic sampling, VAPO [Yue et al., 2025] reintroduces a value model for long CoT. Others include sequence-level importance ratios [Zheng et al., 2025a] and adaptive reweighting [Yang et al., 2026]. These modify what is optimized. MaPP is complementary: it modifies how the GRPO advantage is estimated under group composition uncertainty, while operating within the same GRPO training framework. Conceptually, MaPP Rao–Blackwellizes [Casella and Robert, 1996] the GRPO advantage using a conjugate Beta posterior instead of a learned critic [Liu et al., 2017].

## 2.2 Online Prompt Selection for RLVR

Prompts in RLVR contribute unevenly: overly easy or extremely hard prompts yield vanishing gradients, while intermediate ones provide informative signals [Yu et al., 2025, Bae et al., 2026, Zeng et al., 2025]. This motivates online selection methods [Yu et al., 2025, Qu et al., 2026a, Mao et al., 2026, Zheng et al., 2025b, Zhou et al., 2026, Qu et al., 2026b].

Existing approaches fall into two paradigms [Qu et al., 2026b]: evaluation-based (e.g., Dynamic Sampling [Yu et al., 2025], SPEED-RL [Zhang et al., 2025]) oversample and filter, wasting budget; prediction-based (MoPPS [Qu et al., 2026a], DPS [Mao et al., 2026], GRESO [Zheng et al., 2025b], GPS [Qu et al., 2026b], INSIGHT [Zhou et al., 2026]) estimate difficulty before rollout. These mainly improve budget allocation across prompts, but ignore composition noise from uncertainty in group composition when GRPO assigns response-level advantages.

Unlike these methods, MaPP also addresses signal extraction from sampled responses. Specifically, MaPP-AD denoises the GRPO advantage via posterior-predictive marginalization, and MaPP-PS reuses the shared Beta posterior for a more faithful prompt-selection score. Both are modular and can be layered on existing pipelines.

## 3 Challenge Analysis

## 3.1 Preliminaries

GRPO for RLVR: Given a prompt τ from training dataset $\mathcal { T } = \{ \tau _ { i } \} _ { i = 1 } ^ { N }$ ,where N is the size of the training dataset, and G independent responses $\{ y _ { j } \} _ { j = 1 } ^ { G }$ generated by the policy $\pi _ { \theta } ( \cdot \mid \tau )$ , each receiving a binary reward $r _ { j } ^ { \tau } \in \{ 0 , 1 \}$ verifying correctness [Shao et al., 2024, Guo et al., 2025], the objective of RL finetuning is to maximize the expected return:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { \tau \sim \mathcal { T } , \{ y _ { j } \} _ { j = 1 } ^ { G } \sim \pi _ { \theta } ( \cdot | \tau ) } \left[ \frac { 1 } { G } \sum _ { j = 1 } ^ { G } r _ { j } ^ { \tau } \right] ,\tag{1}
$$

where G denotes the group size. GRPO [Shao et al., 2024] eliminates the need for a learned value network by estimating advantages through within-group normalization. For each prompt τ, the policy generates a group of $G$ independent responses $\{ y _ { j } \} _ { j = 1 } ^ { G }$ , each receiving a binary reward $r _ { j } ^ { \tau } \in \bar { \{ 0 , 1 \} }$ The group-relative advantage for the j-th response is then defined as:

$$
\hat { A } _ { j } ^ { \tau } = \frac { r _ { j } ^ { \tau } - \mathrm { m e a n } ( \{ r _ { k } ^ { \tau } \} _ { k = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ r _ { k } ^ { \tau } \} _ { k = 1 } ^ { G } ) } .\tag{2}
$$

As noted in DAPO [Yu et al., 2025], when all responses to a given prompt are either all correct or all incorrect, they receive identical rewards, yielding $\hat { A } _ { i } ^ { \tau } = 0$ . In this scenario, the prompt is degenerate and contributes no gradient signal [Yu et al., 2025, Zheng et al., 2025b]. To identify such conditions, prior work on prompt selection [Zheng et al., 2025b] introduces a success count variable $S ^ { \tau } \in \{ 0 , G \}$ indicating the degenerate case, i.e., all responses are correct or all are incorrect.

Online Prompt Selection: Prediction-based online prompt selection introduces a success rate $\gamma ^ { \tau } \in [ 0 , 1 ]$ for each prompt τ to estimate prompt difficulty for selection. Since $\gamma$ is not directly observable, prior works [Qu et al., 2026a, Zhou et al., 2026] first model each response reward $r _ { j } ^ { \tau } \in \{ 0 , 1 \}$ generated from prompt τ as an independent Bernoulli trial:

$$
r _ { j } ^ { \tau } \mid \gamma ^ { \tau } \sim \operatorname { B e r n o u l l i } ( \gamma ^ { \tau } ) \quad \forall j \in \{ 1 , \cdot \cdot \cdot G \} .\tag{3}
$$

We define the corresponding success count as $\begin{array} { r } { S ^ { \tau } : = \sum _ { j = 1 } ^ { G } r _ { j } ^ { \tau } } \end{array}$ , which follows a Binomial distribution, $i . e . , S ^ { \tau } \mid \gamma ^ { \tau } \sim$ Binomial $( G , \gamma ^ { \tau } )$

Then, a recursive Bayesian update procedure is applied for efficient posterior inference of the success rates $\gamma ^ { \tau }$ . At the initial stage a Beta prior models the initial success rate formulated as:

$$
\gamma _ { 0 } ^ { \tau } \sim \mathrm { B e t a } ( \alpha _ { 0 } ^ { \tau } , \beta _ { 0 } ^ { \tau } ) ,\tag{4}
$$

where $\alpha _ { 0 } ^ { \tau }$ and $\beta _ { 0 } ^ { \tau }$ reflect prior pseudo-counts of successes and failures, typically set to (1, 1) for a uniform prior. During prompt selection, the optimization history is denoted as $\mathcal { H } _ { t } \dot { = } \{ B _ { t } , \dot { R } _ { t } \}$ , where $B _ { t }$ is the selected prompt batch at step t and $R _ { t } = \{ r _ { j } ^ { \tau } \} _ { \tau \in B _ { t } } ^ { j \in G }$ collects the corresponding rewards. By Bayes rule, the posterior distribution over $\gamma _ { t } ^ { \tau }$ given observations up to step t is:

$$
p ( \gamma _ { t } ^ { \tau } \mid \mathcal { H } _ { t } ) \propto p ( r _ { j } ^ { \tau } \mid \gamma _ { t } ^ { \tau } ) \cdot p ( \gamma _ { t } ^ { \tau } \mid \mathcal { H } _ { t - 1 } ) ,\tag{5}
$$

where $p ( \gamma _ { t } ^ { \tau } \mid \mathcal { H } _ { t - 1 } ) \sim \mathrm { B e t a } ( \alpha _ { t } ^ { \tau } , \beta _ { t } ^ { \tau } )$ represents the conditional prior using the last updated posterior as the proxy. $p ( r _ { j } ^ { \tau } \mid \gamma _ { t } ^ { \tau } )$ is the likelihood of observing the feedback under prompt τ at step t.

With a temporal discounting factor $\lambda \in ( 0 , 1 )$ applied for tracking the evolving policy [Qu et al., 2026a, Zhou et al., 2026], the posterior parameters are updated as:

$$
\alpha _ { t + 1 } ^ { \tau } = \lambda \cdot \alpha _ { t } ^ { \tau } + ( 1 - \lambda ) \cdot \alpha _ { 0 } ^ { \tau } + S _ { t } ^ { \tau } , \qquad \beta _ { t + 1 } ^ { \tau } = \lambda \cdot \beta _ { t } ^ { \tau } + ( 1 - \lambda ) \cdot \beta _ { 0 } ^ { \tau } + G - S _ { t } ^ { \tau } .\tag{6}
$$

Existing selection methods use this posterior to prioritize prompts with intermediate difficulty, as these are most likely to yield non-degenerate gradients [Bae et al., 2026, Qu et al., 2026a,b].

## 3.2 Composition Noise in GRPO

Previous online prompt selection methods model prompt difficulty using success rates and update it via posterior probabilities [Qu et al., 2026a, Mao et al., 2026, Zheng et al., 2025b, Zhou et al., 2026, Qu et al., 2026b], while suffer from composition noise as illustrated in Fig. 1 (b). For an in-depth analysis, we formulate the explicit form of the aforementioned composition noise. The variance of the advantage for each response under the same prompt τ admits a precise decomposition via the law of total variance, expressed as:

$$
\underbrace { \mathrm { V a r } \big ( \hat { A } _ { j } ^ { \tau } \mid \gamma ^ { \tau } \big ) } _ { \mathrm { t o t a l } } = \underbrace { \mathrm { V a r } \big ( \mathbb { E } [ \hat { A } _ { j } ^ { \tau } \mid r _ { j } ^ { \tau } , \gamma ^ { \tau } ] \big | \gamma ^ { \tau } \big ) } _ { \mathrm { c o m p o s i t i o n - f r e e ~ s i g n a l } } + \underbrace { \mathbb { E } \big [ \mathrm { V a r } ( \hat { A } _ { j } ^ { \tau } \mid r _ { j } ^ { \tau } , \gamma ^ { \tau } ) \big | \gamma ^ { \tau } \big ] } _ { \mathrm { c o m p o s i t i o n ~ n o i s e } \xi ( \gamma ^ { \tau } ) } .\tag{7}
$$

The first term describes the variation in advantage among different responses to the same prompt, which distinguishes correct responses from incorrect ones, reflecting the genuine signal. The second term describes that, even when the response is fixed as correct or incorrect, its advantage still varies across different groups — a variation that arises entirely from the random composition of the group. This forms the composition noise term $\xi ( \gamma ^ { \tau } )$ , which is intrinsic and independent to prompt selection. However, $\xi ( \gamma ^ { \tau } )$ is strictly positive for all $\gamma ^ { \tau } \in ( 0 , 1 )$ (Theorem D.1) and cannot be reduced by prompt selection alone. In standard $\mathrm { G R P O }$ , the realized advantage $\hat { A } _ { j } ^ { \tau }$ directly enters the policy gradient, so the composition-noise term in Eq.(7) contributes variance to gradient estimation. Since prompt-selection methods assess prompt informativeness through sampled group rewards generated by the same policy, this noise also makes such assessments less faithful. Therefore, rather than using the noisy realized advantage $\hat { A } _ { i } ^ { \tau }$ , we target its conditional expectation $\mathbb { E } [ \hat { A } _ { j } ^ { \tau } \mid { r } _ { j } ^ { \tau } , { \gamma } ^ { \tau } ]$ , which depends only on the response reward and the prompt’s latent difficulty and removes the composition-noise term by construction.

## 4 Method

The decomposition in Eq. (7) identifies $\mathbb { E } [ \hat { A } _ { j } ^ { \tau } \mid r _ { j } ^ { \tau } , \gamma ^ { \tau } ]$ as the composition-free target. This section formalizes that target (§4.1), recovers it via posterior-predictive marginalization (§4.2), and extends the same principle to prompt selection (§4.3). In this section, we discuss our method with a given prompt τ. Hence, we omit superscript τ for simplicity.

## 4.1 The Intrinsic Advantage

The intrinsic advantage answers the question the standard GRPO advantage fails to: given that this response is correct (or not) on a prompt o $\boldsymbol { \mathscr { f } } d i f f a c u l t y \boldsymbol { \gamma }$ , what advantage should it receive on average? Intuitively, it should depend only on the response reward and the prompt intrinsic difficulty, never on which peers happened to be drawn. Therefore, we first give the definition of the intrinsic advantage $a _ { j } ^ { \star } ( \gamma )$ and its closed form below.

Definition 1 (Intrinsic advantage). For a prompt τ with success rate $\gamma \in ( 0 , 1 )$ , the intrinsic advantage of response j is

$$
a _ { j } ^ { \star } ( \gamma ) : = \mathbb { E } \big [ \hat { A } _ { j } | r _ { j } , \gamma \big ] ,\tag{8}
$$

Proposition 4.1 (Closed form). Under $r _ { j } \mid \gamma \stackrel { \mathrm { i . i . d . } } { \sim }$ Bernoulli(γ),

$$
a _ { j } ^ { \star } ( \gamma ) = \left\{ \begin{array} { l l } { + \mu _ { G } ( \gamma ) , } & { r _ { j } = 1 , } \\ { - \mu _ { G } ( 1 - \gamma ) , } & { r _ { j } = 0 , } \end{array} \right.\tag{9}
$$

where

$$
\mu _ { G } ( p ) = \sum _ { S = 1 } ^ { G - 1 } { \binom { G - 1 } { S - 1 } } p ^ { S - 1 } ( 1 - p ) ^ { G - S } \sqrt { \frac { G - S } { S } } , G \geq 2 .\tag{10}
$$

Note that in the degenerate cases where $S = \{ 0 , G \}$ contribute zero and are omitted from the sum. Applying such intrinsic advantage $a _ { j } ^ { \star } ( \gamma )$ for $\hat { A } _ { j }$ guarantees that $\mathrm { V a r } [ a _ { j } ^ { \star } \mid r _ { j } , \gamma ] = 0$ . Therefore, the total variance in $\operatorname { E q . } ( 7 )$ degenerates into the composition-free signal term alone and intrinsically eliminates the composition-noise term. However, $a _ { j } ^ { \star }$ depends on the latent success rate $\gamma _ { - }$ , which can not be observed or explicitly derived. Therefore, we recover it through closed-form marginalization over the Beta posterior.

## 4.2 Marginalized Posterior-Predictive Advantage Denoising

To avoid each response’s self-influence [Liu et al., 2025], we marginalize the latent success rate γ over a leave-one-out Beta posterior that excludes response $j ^ { \flat }$ s own reward, formulated as:

$$
p ( \gamma \mid \mathcal { H } _ { t } , r _ { t ; - j } ) \ = \ \mathrm { B e t a } ( \alpha _ { t + 1 ; j } ^ { \prime } , \beta _ { t + 1 ; j } ^ { \prime } ) ,\tag{11}
$$

$$
\alpha _ { t + 1 ; j } ^ { \prime } : = \alpha _ { t + 1 } - r _ { t ; j } , \beta _ { t + 1 ; j } ^ { \prime } : = \beta _ { t + 1 } - 1 + r _ { t ; j } ,\tag{12}
$$

where group-shared $\alpha _ { t + 1 }$ and $\beta _ { t + 1 }$ are calculated according to Eq. $( 6 ) . \ r _ { t ; - j }$ denotes the other rewards in the group except for j-th response at t-th iteration. This construction guarantees that the posterior used to estimate the advantage of response j is conditionally independent of $\boldsymbol { r } _ { j } ;$ consequently, the resulting estimator is free from bias induced by the response it evaluates. At each training step t, we marginalize $a _ { j } ^ { \star } ( \gamma )$ over the leave-one-out posterior to obtain a composition-free advantage estimate:

$$
\hat { a } _ { j } ^ { \mathrm { M a P P } } : = \mathbb { E } _ { p ( \gamma | \mathcal { H } _ { t } , r _ { t ; - j } ) } [ a _ { j } ^ { \star } ( \gamma ) ] \ = \ \int _ { 0 } ^ { 1 } a _ { j } ^ { \star } ( \gamma ) \ p ( \gamma | \mathcal { H } _ { t } , r _ { t ; - j } ) \ \mathrm { d } \gamma .\tag{13}
$$

With Eq.(13), we average all plausible values of $\gamma$ , which is weighted by the observed optimization history and forms $\hat { a } _ { i } ^ { \mathrm { M a P P } }$ , called MaPP-AD. Consequently, the closed form of our tractable integral MaPP-AD is given by Beta–Binomial conjugacy.

Proposition 4.2 (Closed form). At training step t, let $\alpha _ { j } ^ { \prime } : = \alpha _ { t + 1 ; j } ^ { \prime }$ and $\beta _ { j } ^ { \prime } : = \beta _ { t + 1 ; j } ^ { \prime }$ for brevity. Then

$$
\hat { a } _ { j } ^ { \mathrm { M a P P } } = \left\{ \begin{array} { l l } { \displaystyle + \sum _ { S = 1 } ^ { G - 1 } w _ { S } ( \alpha _ { j } ^ { \prime } , \beta _ { j } ^ { \prime } ) \sqrt { \frac { G - S } { S } } , } & { \displaystyle r _ { j } = 1 , } \\ { \displaystyle - \sum _ { S = 1 } ^ { G - 1 } w _ { S } ( \beta _ { j } ^ { \prime } , \alpha _ { j } ^ { \prime } ) \sqrt { \frac { G - S } { S } } , } & { \displaystyle r _ { j } = 0 , } \end{array} \right.\tag{14}
$$

where $w _ { S } ( \alpha , \beta ) : = P _ { \mathrm { B B } } ( S - 1 \mid G - 1 , \alpha , \beta ) = { \binom { G - 1 } { S - 1 } } { \frac { B ( \alpha + S - 1 , \beta + G - S ) } { B ( \alpha , \beta ) } }$ is the Beta-Binomial PMF evaluated at S − 1 successes out ofG − 1 trials.

The weight $w _ { S } ( \alpha _ { j } ^ { \prime } , \beta _ { j } ^ { \prime } )$ has a concrete meaning: it is the posterior-predictive probability that a hypothetical fresh group of G−1 responses would contain s−1 correct ones. The estimator evaluates the advantage at every plausible composition s and averages under the posterior belief, replacing the single realized composition with its full predictive distribution. As the posterior concentrates around the true pass rate with more observations, $\hat { a } _ { j } ^ { \mathrm { M a P P } }$ converges to the intrinsic advantage $a _ { j } ^ { \star } ( \gamma )$ We show in Appendix D that $\hat { a } _ { i } ^ { \mathrm { M a P P } }$ achieves provably lower MSE than GRPO once the posterior accumulates a few updates per prompt.

## 4.3 Marginalized Posterior-Predictive Prompt Selection

§4.1–4.2 focused on denoising response-level advantage estimation. We now show how the same posterior-predictive principle also yields a more faithful prompt-selection score. Existing selection methods score prompts via point estimates of difficulty and select the top-k candidates closest to a target difficulty [Qu et al., 2026a, Mao et al., 2026, Zheng et al., 2025b, Zhou et al., 2026]. This point-estimate approach mirrors the same limitation exposed in §3.2: a single value of γ is substituted for the latent quantity, discarding posterior uncertainty.

We propose instead a principled informativeness score derived from the same marginalization principle. Define the probability that a prompt yields a non-degenerate gradient as $\bar { V _ { G } } ( \gamma ) = 1 -$ $( \gamma ) ^ { G } - ( 1 - \gamma ) ^ { G }$ , which peaks at $\gamma = 1 / 2$ and vanishes at the extremes. Since $\gamma$ is latent, we marginalize against the posterior:

Table 1: Evaluation results on CountDown3to4 and mathematical reasoning benchmarks. Bold indicates the best result. CountDown is evaluated after CountDown3to4 training, while all other benchmarks use math-trained models. Rollouts and GPU hours report math-training cost.
<table><tr><td>Models</td><td>Methods</td><td>CountDown</td><td>AMC</td><td>MATH500</td><td>Minerva.</td><td>Olympiad.</td><td>Avg.↑</td><td>Rollouts↓</td><td>GPU Hours↓</td></tr><tr><td rowspan="5">Qwen3-4B</td><td>Random</td><td>80.95</td><td>54.22</td><td>78.63</td><td>28.31</td><td>38.25</td><td>56.07</td><td>563k</td><td>90h</td></tr><tr><td>MoPPS</td><td>83.68</td><td>60.24</td><td>81.04</td><td>28.31</td><td>44.28</td><td>59.51</td><td>563k</td><td>81h</td></tr><tr><td>DPS</td><td>83.42</td><td>59.04</td><td>80.65</td><td>28.68</td><td>43.98</td><td>59.15</td><td>563k</td><td>81h</td></tr><tr><td>DS</td><td>83.83</td><td>60.24</td><td>82.06</td><td>30.88</td><td>45.03</td><td>60.41</td><td>2252k</td><td>209h</td></tr><tr><td>Ours</td><td>85.67</td><td>62.65</td><td>82.66</td><td>31.25</td><td>47.59</td><td>61.96</td><td>563k</td><td>81h</td></tr><tr><td rowspan="5">Qwen3-8B</td><td>Random</td><td>81.29</td><td>55.42</td><td>81.05</td><td>30.51</td><td>42.17</td><td>58.09</td><td>563k</td><td>104h</td></tr><tr><td>MoPPS</td><td>84.69</td><td>62.65</td><td>82.46</td><td>30.15</td><td>50.60</td><td>62.11</td><td>563k</td><td>88h</td></tr><tr><td>DPS</td><td>85.80</td><td>60.24</td><td>81.45</td><td>29.78</td><td>48.49</td><td>61.15</td><td>563k</td><td>89h</td></tr><tr><td>DS</td><td>83.79</td><td>61.45</td><td>82.86</td><td>31.62</td><td>48.34</td><td>61.61</td><td>2252k</td><td>230h</td></tr><tr><td>Ours</td><td>87.51</td><td>63.86</td><td>83.47</td><td>30.88</td><td>52.26</td><td>63.60</td><td>563k</td><td>88h</td></tr><tr><td rowspan="5">R1-Distill-7B</td><td>Random</td><td>78.92</td><td>57.83</td><td>80.24</td><td>27.57</td><td>38.86</td><td>56.68</td><td>563k</td><td>63h</td></tr><tr><td>MoPPS</td><td>80.75</td><td>59.20</td><td>81.45</td><td>26.84</td><td>41.11</td><td>57.87</td><td>563k</td><td>59h</td></tr><tr><td>DPS</td><td>80.03</td><td>59.04</td><td>79.64</td><td>25.74</td><td>40.66</td><td>57.02</td><td>563k</td><td>58h</td></tr><tr><td>DS</td><td>81.52</td><td>62.65</td><td>80.85</td><td>27.81</td><td>41.57</td><td>58.88</td><td>2231k</td><td>152h</td></tr><tr><td>Ours</td><td>82.24</td><td>61.45</td><td>82.26</td><td>28.31</td><td>42.17</td><td>59.29</td><td>563k</td><td>58h</td></tr></table>

$$
V _ { G } ^ { \mathrm { M a P P } } ( \alpha ^ { \tau } , \beta ^ { \tau } ) : = \mathbb { E } _ { \gamma \sim p ( \gamma | \mathcal { H } ^ { \tau } ) } \big [ V _ { G } ( \gamma ) \big ] = 1 - \frac { B ( \alpha ^ { \tau } + G , \beta ^ { \tau } ) } { B ( \alpha ^ { \tau } , \beta ^ { \tau } ) } - \frac { B ( \alpha ^ { \tau } , \beta ^ { \tau } + G ) } { B ( \alpha ^ { \tau } , \beta ^ { \tau } ) } .\tag{15}
$$

Since $V _ { G }$ is concave, Jensen’s inequality gives $V _ { G } ^ { \mathrm { M a P P } } \leq V _ { G } ( \mathbb { E } [ \gamma ^ { \tau } \mid \mathcal { H } ^ { \tau } ] )$ : the marginalized score automatically down-weights prompts whose posterior is wide, avoiding wasted rollouts on prompts that appear informative under a point estimate but whose true difficulty remains uncertain. Batches are drawn by sampling prompts with probability proportional to $V _ { G } ^ { \mathrm { M a P P } } ( \alpha ^ { \tau } , \beta ^ { \tau } )$ ), preserving exploration over prompts whose posterior is still consolidating.

Unified posterior-predictive gradient. Combining posterior-predictive selection with the posteriorpredictive advantage of §4.2, the MaPP gradient takes the form

$$
\nabla J = \mathbb { E } _ { \tau \sim q , \ \{ y _ { j } ^ { \tau } \} _ { j = 1 } ^ { G } \sim \pi _ { \theta } ( \cdot | \tau ) } \left[ \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \hat { a } _ { j } ^ { \mathrm { M a P P } , \tau } \nabla _ { \theta } \log \pi _ { \theta } ( y _ { j } ^ { \tau } \mid \tau ) \right] , \qquad q ( \tau ) \propto V _ { G } ^ { \mathrm { M a P P } } ( \alpha ^ { \tau } , \beta ^ { \tau } ) .\tag{16}
$$

where both the batch distribution $q ( \tau )$ and the per-response weight $a _ { j } ^ { \mathrm { M a P P } }$ are posterior-predictive marginalizations against the same shared Beta posterior. Both components are modular: each can be adopted independently or composed with existing selection strategies, adding $O ( | B | \cdot G ^ { 2 } )$ per-step overhead, negligible compared to rollout and backpropagation. Algorithm 1 gives the full training loop.

## 5 Experiments

## 5.1 Experimental Setup

We evaluate MaPP across three reasoning domains, mathematics, numerical planning, and visual geometry, using diverse model backbones to assess generality.

Tasks. For Mathematics, we train on MATH [Hendrycks et al., 2021] (7,500 problems) and evaluate on AMC23, MATH500 [Lightman et al., 2023], Minerva Math [Lewkowycz et al., 2022], and OlympiadBench [He et al., 2024]. For Numerical planning, we use the Countdown Number Game: train on Countdown-34 [Pan et al., 2025], evaluate on held-out CD-34 and harder CD-4 (four source numbers, larger search space). For Visual geometry, we train and evaluate on Geometry3k [Lu et al., 2021, Hiyouga, 2025], pairing diagrams with multi-step reasoning questions.

![](images/db0483490e074b08ca080dabd71b1102866d4a17f274b623991963041f78b1fd.jpg)  
Figure 3: Performance comparisons of different methods and models on the MATH and Countdown tasks. Our proposed MaPP outperforms the existing SOTA methods and other baselines in both training efficiency and performance.

Models. For mathematics and planning, we use three text-only models: Qwen3-4B-Base, Qwen3- 8B-Base [Yang et al., 2025], and DeepSeek-R1-Distill-Qwen-7B [Guo et al., 2025]. For visual geometry, we adopt Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct [Bai et al., 2025].

Baselines. We compare against four methods spanning online prompt selection paradigms [Qu et al., 2026b]. (1) Random: uniform sampling (standard GRPO baseline) [Shao et al., 2024]. (2) MoPPS [Qu et al., 2026a]: prediction-based; models success rate as Beta variable, uses Thompson sampling for intermediate difficulty. (3) DPS [Mao et al., 2026]: prediction-based; frames solving progress as a 3-state HMM, prioritizes partially-solved prompts. (4) Dynamic Sampling (DS) [Yu et al., 2025]: evaluation-based; oversamples 4× batch, rolls out all, filters zero-variance prompts. Serves as compute-intensive oracle baseline.

## 5.2 Main Results

Mathematics. Tab. 1 reports evaluation results across five mathematics benchmarks for three model backbones. Across all scales, MaPP consistently achieves the highest average accuracy, outperforming both prediction-based methods (MoPPS, DPS) and the compute-intensive oracle baseline DS. On Qwen3-4B, MaPP obtains an average score of 61.96, improving over the strongest prediction-based baseline MoPPS by +2.45 and even surpassing DS by +1.55, while using only 25% of its rollout budget (563k vs. 2252k). Similar trends hold on Qwen3-8B (+1.49 over MoPPS) and R1-Distill-7B (+1.42 over MoPPS), confirming that the improvement is consistent across model families and scales.

Planning and visual geometry. On the Countdown task (Fig. 3, bottom row), MaPP achieves final accuracy gains of +1.84 (Qwen3-4B) and +1.71 (Qwen3-8B) over the strongest baseline at convergence. For visual geometry on Geometry3k (Fig. C(a)), MaPP outperforms the strongest baseline by +1.04 on Qwen2.5-VL-3B-Instruct and +2.40 on Qwen2.5-VL-7B-Instruct. These results demonstrate that MaPP generalizes beyond text-only mathematical reasoning to both planning and multi-modal tasks.

Training efficiency. Fig. 3 plots accuracy against training steps across all six model-task combinations. MaPP converges substantially faster than all baselines: on MATH, it reaches the same accuracy as Random Sample by approximately 2.5–2.8× faster convergence speed, while the acceleration rates range from 1.6× to 2.6× on Countdown dataset. MaPP introduces no additional rollout cost over

Table 2: Sensitivity to the temporal discount λ. Table 3: Robustness to rollout group size G. Trained Trained on Qwen3-4B, Countdown. on Qwen3-4B, Countdown.
<table><tr><td>λ</td><td>CountDown3to4</td><td>CountDown4</td><td>Avg.</td></tr><tr><td>0.0</td><td>83.51</td><td>66.19</td><td>74.85</td></tr><tr><td>0.1</td><td>85.17</td><td>70.04</td><td>77.61</td></tr><tr><td>0.5</td><td>86.76</td><td>72.24</td><td>79.50</td></tr><tr><td>0.7</td><td>85.95</td><td>69.18</td><td>77.57</td></tr><tr><td>0.9</td><td>83.08</td><td>67.33</td><td>75.21</td></tr><tr><td>1.0</td><td>85.82</td><td>70.72</td><td>78.27</td></tr></table>

<table><tr><td colspan="4">G = 4</td><td colspan="2">G = 16</td></tr><tr><td>Method</td><td>CD-34</td><td>CD-4</td><td>#Rollouts</td><td>CD-34 CD-4</td><td>#Rollouts</td></tr><tr><td>Random</td><td>79.39</td><td>60.14</td><td>123k</td><td>82.54 64.01</td><td>492k</td></tr><tr><td>MoPPS</td><td>80.22</td><td>62.70</td><td>123k 86.39</td><td>72.79</td><td>492k</td></tr><tr><td>DPS</td><td>82.17</td><td>63.26</td><td>123k 86.90</td><td>70.18</td><td>492k</td></tr><tr><td>DS</td><td>81.85</td><td>64.11</td><td>490k 83.87</td><td>66.66</td><td>1855k</td></tr><tr><td>Ours</td><td>82.45</td><td>64.44</td><td>123k</td><td>88.33 73.42</td><td>492k</td></tr></table>

prediction-based methods, and requires less than half the runtime of DS across all settings (Tab. 1, last two columns).

Discussion. Several patterns emerge from the results. First, the gains are most pronounced on challenging benchmarks such as OlympiadBench (+3.31 on Qwen3-4B), suggesting that cleaner gradient signals from advantage denoising disproportionately benefit the learning of harder reasoning skills. Second, MaPP significantly improves over MoPPS and DPS despite sharing the same Beta posterior infrastructure, confirming that the gains stem from how the posterior is used, specifically marginalization for both advantage estimation and prompt selection, rather than from a better posterior itself. Third, the improvement persists across three task families and five model backbones, indicating that posterior-predictive marginalization provides a general refinement to GRPO learning signals rather than a task-specific heuristic.

## 5.3 Ablation Analysis

Effect of the two components. MaPP comprises two modules: the posterior-predictive advantage (MaPP-AD) and the posterior-predictive selection score (MaPP-PS). Tab. 4 disentangles the contributions of MaPP-AD and MaPP-PS on Qwen3-4B. Starting from standard GRPO, adding MaPP-PS alone yields +1.62 improvement on MATH, while MaPP-AD alone yields a larger +2.66. Combining both produces +3.44, exceeding the sum of individual gains, suggesting the two modules are synergistic. To verify complementarity with existing methods, we plug MaPP-AD into MoPPS and DPS without modifying their selection logic; it consistently improves both (+1.04 and +1.12 respectively), demonstrating that advantage denoising provides additional gains beyond prompt selection.

Table 4: Ablation on the two components. Trained on MATH, Qwen3-4B. MaPP-AD: posterior-predictive advantage (§ 4.2). MaPP-PS: posterior-predictive selection (§ 4.3).
<table><tr><td>Method</td><td>MATH</td><td>Olympiad.</td><td>Avg.</td></tr><tr><td>GRPO</td><td>79.41</td><td>38.25</td><td>58.83</td></tr><tr><td>+ MaPP-PS</td><td>81.03</td><td>43.17</td><td>62.10</td></tr><tr><td>+ MaPP-AD</td><td>82.07</td><td>44.52</td><td>63.30</td></tr><tr><td>+ AD + PS</td><td>82.85</td><td>47.59</td><td>65.22</td></tr><tr><td>MoPPS</td><td>82.17</td><td>44.28</td><td>63.23</td></tr><tr><td>+ MaPP-AD</td><td>82.63</td><td>45.90</td><td>64.27</td></tr><tr><td>DPS</td><td>82.24</td><td>43.98</td><td></td></tr><tr><td>+ MaPP-AD</td><td>82.71</td><td>45.75</td><td>63.11 64.23</td></tr></table>

Sensitivity to the temporal discount λ. The temporal discount λ controls the trade-off between tracking speed and posterior stability. Tab. 2 and Fig. C(b) reports MaPP performance on Countdown under varying λ. Performance peaks at λ=0.5 (79.50 avg.) and degrades at both extremes: λ=0.0 (74.85) forgets all history and reduces the posterior to single-step evidence, while λ=0.9 (75.21) retains stale observations that lag behind the evolving policy. The moderate sensitivity suggests that a reasonable balance between recency and stability is sufficient, and the optimal value does not require fine-grained tuning. We use λ=0.5 for all other experiments.

Robustness to group size. Tab. 3 and Fig. B evaluate MaPP on the Countdown benchmark under group sizes G=4 and G=16, complementing the default G=8 used on in main experiment. MaPP achieves the highest accuracy across both settings: at G=4, it outperforms the best baseline by +0.28 (CD-34) and +0.33 (CD-4); at G=16, the margin widens to +1.43 (CD-34) and +0.63 (CD-4). The wider margin at larger G is expected: more responses per group provide richer per-step evidence for the posterior, enabling more precise advantage corrections that outweigh the averaging effect of larger groups. This confirms that posterior-predictive marginalization scales favorably with group size.

## 6 Conclusion

We identified composition noise in the GRPO advantage due to uncertainty in group composition, which degrades gradient estimation and prompt selection. To address this, we proposed MaPP: a unified framework that extends the per-prompt Beta posterior to a closed-form posterior-predictive advantage estimator with provably diminishing MSE and an uncertainty-aware selection score, addressing both data-efficiency losses through a single shared posterior. Experiments on mathematics, planning, and visual geometry with five backbones show consistent gains in accuracy and convergence over GRPO and strong baselines, with no additional rollout overhead.

## References

S. Bae, J. Hong, M. Y. Lee, H. Kim, J. Nam, and D. Kwak. Online difficulty filtering for reasoning oriented reinforcement learning. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 700–719, 2026.

S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Xu, M. Yan, J. Lin, D. Liu, B. Yu, B. Hui, B. Li, J. Zhou, J. Lin, C. Zhou, J. Zhou, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

G. Casella and C. P. Robert. Rao-blackwellisation of sampling schemes. Biometrika, 83(1):81–94, 1996.

X. Chen, J. Lu, M. Kim, D. Zhang, J. Tang, A. Piché, N. Gontier, Y. Bengio, and E. Kamalloo. Self-evolving curriculum for llm reasoning. arXiv preprint arXiv:2505.14970, 2025.

D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

C. He, R. Luo, Y. Bai, S. Hu, Z. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, 2024.

D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Hiyouga. Geometry3k: A large-scale multi-modal geometry reasoning dataset. https://huggingface.co/ datasets/hiyouga/geometry3k, 2025.

A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

A. Lewkowycz, A. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.

H. Lightman, V. Kosaraju, Y. Burda, H. Edwards, B. Baker, T. Lee, J. Leike, J. Schulman, I. Sutskever, and K. Cobbe. Let’s verify step by step. In The twelfth international conference on learning representations, 2023.

Z. Lin, M. Lin, Y. Xie, and R. Ji. Cppo: Accelerating the training of group relative policy optimization-based reasoning models. arXiv preprint arXiv:2503.22342, 2025.

H. Liu, Y. Feng, Y. Mao, D. Zhou, J. Peng, and Q. Liu. Action-depedent control variates for policy optimization via stein’s identity. arXiv preprint arXiv:1710.11198, 2017.

Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

I. Loshchilov and F. Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

P. Lu, R. Gong, S. Jiang, L. Qiu, S. Huang, X. Liang, and S.-C. Zhu. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. In The Joint Conference ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021), 2021.

Y. Mao, Y. Qu, Q. Wang, H. Zou, and X. Ji. Dynamics-predictive sampling for active rl finetuning of large reasoning models. arXiv preprint arXiv:2603.10887, 2026.

J. Pan, J. Zhang, X. Wang, L. Yuan, H. Peng, and A. Suhr. Tinyzero. https://github.com/Jiayi-Pan/ TinyZero, 2025. Accessed: 2025-01-24.

Y. Qu, Q. Wang, Y. Mao, V. T. Hu, B. Ommer, and X. Ji. Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models? In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 1240–1250, 2026a.

Y. Qu, Q. Wang, Y. Mao, H. Zou, Y. Jiang, W. Liu, C. Bai, K. Yang, Y. Chen, S. Yang, et al. Small generalizable prompt predictive models can steer efficient rl post-training of large reasoning models. arXiv preprint arXiv:2602.01970, 2026b.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings ofthe Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

F. Yang, Z. Chen, X. Wang, X. Lu, J. Chai, G. Yin, W. Lin, S. Ma, F. Zhuang, D. Wang, et al. Your group-relative advantage is biased. arXiv preprint arXiv:2601.08521, 2026.

Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Y. Yue, Y. Yuan, Q. Yu, X. Zuo, R. Zhu, W. Xu, J. Chen, C. Wang, T. Fan, Z. Du, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.

Y. Zeng, Z. Sun, B. Ji, E. Min, H. Cai, S. Wang, D. Yin, H. Zhang, X. Chen, and J. Wang. Cures: From gradient analysis to efficient curriculum learning for reasoning llms. arXiv preprint arXiv:2510.01037, 2025.

R. Zhang, D. Arora, S. Mei, and A. Zanette. Speed-rl: Faster training of reasoning models via online curriculum learning. arXiv preprint arXiv:2506.09016, 2025.

C. Zheng, S. Liu, M. Li, X.-H. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, et al. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025a.

H. Zheng, Y. Zhou, B. R. Bartoldson, B. Kailkhura, F. Lai, J. Zhao, and B. Chen. Act only when it pays: Efficient reinforcement learning for llm reasoning via selective rollouts. arXiv preprint arXiv:2506.02177, 2025b.

H. Zheng, Y. Zhou, B. R. Bartoldson, B. Kailkhura, F. Lai, J. Zhao, and B. Chen. Act only when it pays: Efficient reinforcement learning for llm reasoning via selective rollouts. arXiv preprint arXiv:2506.02177, 2025c.

X. Zhou, B. Zhu, H. Zhang, H. Wang, and Z. Guo. Efficient rlvr training via weighted mutual information data selection. arXiv preprint arXiv:2603.01907, 2026.

## Appendix

## A Algorithm

Algorithm 1 summarizes the complete MaPP training loop. At each step, prompts are first selected by sampling proportionally to the posterior-predictive score $V _ { G } ^ { \mathrm { M a P P } }$ (§4.3). After rollout, the posteriorpredictive advantage $\hat { a } _ { i } ^ { \mathrm { M a P P } }$ replaces the standard GRPO advantage for policy update (§4.2). The shared Beta posterior is then updated from the new observations. The entire procedure adds only closed-form computations on top of standard GRPO, with no additional rollouts or model forward passes.

Algorithm 1 MAPP: Marginalized Posterior-Predictive Framework   
Require: Prompt pool $\tau _ { \mathrm { { i } } }$ ; policy π<sub>θ</sub>; group size $G ;$ batch size $| B | ;$ ; discount $\lambda \in ( 0 , 1 )$ ; training steps   
$\bar { \boldsymbol { T } }$   
Ensure: Updated policy $\pi _ { \theta }$   
1: Initialize $( \alpha _ { 0 } ^ { \tau } , \dot { \beta } _ { 0 } ^ { \tau } ) \dot {  } ( 1 , 1 )$ for all $\tau \in \mathcal { T }$ ▷ uninformative Beta prior   
2: for $t = 0 , \ldots , \check { T } - 1$ do   
▷ Posterior-predictive prompt selection (§4.3)   
3: for $\tau \in { \bar { \mathcal { T } } }$ do   
4: $\begin{array} { r } { V _ { G } ^ { \mathrm { M a P P } } ( \alpha _ { t } ^ { \tau } , \beta _ { t } ^ { \tau } ) \gets 1 - \frac { B ( \alpha _ { t } ^ { \tau } + G , \beta _ { t } ^ { \tau } ) } { B ( \alpha _ { t } ^ { \tau } , \beta _ { t } ^ { \tau } ) } - \frac { B ( \alpha _ { t } ^ { \tau } , \beta _ { t } ^ { \tau } + G ) } { B ( \alpha _ { t } ^ { \tau } , \beta _ { t } ^ { \tau } ) } } \end{array}$ ▷ Eq. (15)   
5: end for   
6: Sample batch $\begin{array} { r } { \mathsf { B } _ { t } \sim q ( \tau ) \propto V _ { G } ^ { \mathrm { M a P P } } ( \alpha _ { t } ^ { \tau } , \beta _ { t } ^ { \tau } ) } \end{array}$   
▷ Rollout and reward collection   
7: for $\tau \in B _ { t }$ do   
8: Draw $\{ y _ { j } ^ { \tau } \} _ { j = 1 } ^ { G } \sim \pi _ { \theta } ( \cdot \mid \tau ) ;$ ; observe rewards $\{ r _ { t ; j } ^ { \tau } \} _ { j = 1 } ^ { G }$   
9: $\begin{array} { r } { S _ { t } ^ { \tau } \gets \sum _ { j = 1 } ^ { G } r _ { t ; j } ^ { \tau } } \end{array}$   
10: end for   
▷ Posterior-predictive advantage (§4.2)   
11: for $\tau \in \dot { \mathcal { B } } _ { t } , j = 1 , \dots , G$ do   
12: $\alpha _ { t + 1 ; j } ^ { \prime \tau } \stackrel { \triangledown } {  } \lambda \cdot \alpha _ { t } ^ { \tau } + ( 1 - \lambda ) \cdot \alpha _ { 0 } ^ { \tau } + S _ { t } ^ { \tau } - r _ { t ; j } ^ { \tau }$ ▷ LOO posterior, Eq. (12)   
13: $\beta _ { t + \perp ; j } ^ { \prime \prime }  \lambda \cdot \beta _ { t } ^ { \tau } + ( 1 - \lambda ) \cdot \beta _ { 0 } ^ { \tau } + G - S _ { t } ^ { \tau } - 1 + r _ { t ; j } ^ { \tau }$   
14: if $\dot { r } _ { t ; j } ^ { \tau } = 1$ then   
15: $\begin{array} { r } { \hat { a } _ { t ; j } ^ { \mathrm { M a P P } , \tau } \gets + \sum _ { S = 1 } ^ { G - 1 } w _ { S } ( \alpha _ { t + 1 ; j } ^ { \prime \tau } , \beta _ { t + 1 ; j } ^ { \prime \tau } ) \sqrt { ( G - S ) / S } } \end{array}$ ▷ Eq. (14)   
16: else   
17: $\begin{array} { r } { \hat { a } _ { t ; j } ^ { \mathrm { M a P P } , \tau } \gets - \sum _ { S = 1 } ^ { G - 1 } w _ { S } \big ( \beta _ { t + 1 ; j } ^ { \prime \tau } , \alpha _ { t + 1 ; j } ^ { \prime \tau } \big ) \sqrt { ( G - S ) / S } } \end{array}$   
18: end if   
19: end for   
▷ Policy update (Eq. (16))   
20: $\theta  \theta + \eta \nabla _ { \theta } \mathcal { I } ( \theta )$ with $\begin{array} { r } { \nabla _ { \theta } \mathcal { I } = \frac { 1 } { | \mathcal { B } _ { t } | } \sum _ { \tau \in \mathcal { B } _ { t } } } \end{array}$ aˆ<sup>MaPP,τ</sup><sub>t;j</sub> ∇<sub>θ</sub> log π<sub>θ</sub>(y<sup>τ</sup><sub>j</sub> | τ )   
▷ Posterior update (Eq. (6))   
21: for $\tau \in \bar { B } _ { t }$ do   
22: $\alpha _ { t + 1 } ^ { \tau } \stackrel { \cdot } {  } \lambda \cdot \alpha _ { t } ^ { \tau } + ( 1 - \lambda ) \cdot \alpha _ { 0 } ^ { \tau } + S _ { t } ^ { \tau }$   
23: $\beta _ { t + 1 } ^ { \check { \tau } ^ { \mathrm { ~ \tiny ~ \cdot ~ } } } \gets \lambda \cdot \beta _ { t } ^ { \check { \tau } } + \left( 1 - \lambda \right) \cdot \beta _ { 0 } ^ { \check { \tau } } + G ^ { - } - S _ { t } ^ { \tau }$   
24: end for   
25: end for

## B Implementation Details

## B.1 Tasks and Datasets

Mathematics. We train on the MATH dataset [Hendrycks et al., 2021], which contains 7,500 competition-level problems spanning algebra, geometry, number theory, and combinatorics. Following prior work [Mao et al., 2026, Qu et al., 2026a,b], we evaluate on four benchmarks: AMC23, MATH500 [Lightman et al., 2023], Minerva Math [Lewkowycz et al., 2022], and OlympiadBench [He et al., 2024], using the datasets hosted by MOPPS [Qu et al., 2026a]. We adopt a binary reward function following the default configuration in verl [Sheng et al., 2025]: a reward of 1 for correct and 0 otherwise.

Numerical planning. We adopt the Countdown Number Game [Pan et al., 2025], which requires combining given numbers using basic arithmetic operations to reach a target value. Training is conducted on a 2,000-problem subset of the Countdown-34 dataset. Evaluation uses two benchmarks: a 512-problem held-out split from Countdown-34 (CD-34), and a 512-problem subset from Countdown-4 (CD-4), a harder variant that provides four source numbers per problem and substantially enlarges the search space.

Visual geometry. We train on the 2,101-problem training split of the Geometry3k dataset [Lu et al., 2021, Hiyouga, 2025], which pairs geometric diagrams with multi-step reasoning questions. Evaluation is conducted on the official 601-problem test split.

## B.2 Training Details

We adopt GRPO [Shao et al., 2024] as the default RLVR algorithm, implemented within the verl framework [Sheng et al., 2025]. At each training step, we sample $G { = } 5$ responses per prompt for math tasks and $G { = } 8$ responses per prompt for Countdown and Geometry, using temperature 1.0 and top-p=1.0 for rollout generation. We disable the KL penalty by setting $\beta { = } 0 .$ , consistent with Yu et al. [2025]. Training batch sizes are set to 256 for MATH and Countdown, with mini-batch sizes of 128 and 64 respectively, and 512 for Geometry3k with a mini-batch size of 256. The maximum response length is 1024 tokens for all tasks. Optimization is performed using AdamW [Loshchilov and Hutter, 2017] with learning rate $1 \times 1 0 ^ { - 6 } , \dot { ( } \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . \dot { 9 } , 0 . 9 9 9 )$ , and weight decay 0.01. We apply the Clip-Higher strategy from DAPO [Yu et al., 2025], which decouples clipping ranges with $\epsilon _ { \mathrm { l o w } } { = } 0 . 2$ and $\epsilon _ { \mathrm { h i g h } } { = } 0 . 2 8$ . All experiments are conducted on 8 NVIDIA H100 GPUs.

For all prediction-based prompt selection methods, the candidate pool size is set to $\hat { M } { = } 8 \times$ the training batch size B. For MoPPS [Qu et al., 2026a] and DPS [Mao et al., 2026], we follow their original top-B selection protocol, where prompts are ranked by the corresponding scoring criterion and the top-B prompts are selected. In contrast, MaPP does not perform deterministic top-B selection. Instead, each candidate prompt is assigned a retention probability $q ( \tau ) \propto V _ { G } ^ { \mathrm { M a P P } } ( \alpha ^ { \tau ^ { \cdot } } , \beta ^ { \tau } )$ , and the training batch is sampled according to these probabilities. For MaPP, the Beta prior is initialized as $( \alpha _ { 0 } , \beta _ { 0 } ) = ( 1 , 1 )$ and the temporal discount is set to $\lambda { = } 0 . 5$ across all tasks. For MoPPS [Qu et al., 2026a], we follow the original configuration with Beta prior $( \alpha _ { 0 } , \beta _ { 0 } ) = ( 1 , 1 )$ , target success probability $\gamma ^ { * } { = } 0 . 5 ,$ , and decay factor λ=0.5. For DPS [Mao et al., 2026], we adopt the default threestate HMM with Dirichlet prior $\alpha _ { 0 } = ( 1 , 1 , 1 )$ and decay ratio $\lambda { = } 0 . 5$ . For Dynamic Sampling [Yu et al., 2025], we use the verl implementation, which over-samples a candidate batch 4× the training batch size and post-hoc filters out prompts with zero reward standard deviation.

## C Additional Experiments

## C.1 Empirical Calibration of MaPP-AD and MaPP-PS

We verify that $a _ { j } ^ { \star } ( \gamma )$ and $V _ { G } ^ { M a P P } ( \gamma )$ faithfully reflect empirical training dynamics. For MaPP-AD, we take a mid-training checkpoint on Countdown (Qwen3-4B, G=8), sample 100 prompts and draw 50 independent groups per prompt; Figure $\mathbf { A } ( \mathbf { a } )$ shows that the theoretical intrinsic advantage closely tracks the empirical mean advantage, with nearly all points falling along the ideal calibration line, and the ±1 std band visualizes the composition noise that MaPP-AD eliminates. For MaPP-PS, we collect the predicted $V _ { G } ^ { M a P P } ( \hat { \gamma } )$ and the empirical non-degenerate ratio at each training step throughout training, bin them by the theoretical score, and compare; Figure A(b) shows that $V _ { G } ( \hat { \gamma } )$ tracks the empirical ratio within $\mathbf { a } \pm 0 . 0 5$ band across the full range, confirming reliable difficulty estimation. Together, both plots validate the Bernoulli–Beta modeling assumption that MaPP builds upon.

## C.2 Training Dynamics

Robustness to group size. Figure B shows accuracy versus training steps under group sizes G=4 and G=16 on Countdown. MaPP reaches the same accuracy as the strongest baseline by approximately 2.0–2.4× faster convergence speed at G=4, while the acceleration increases to 2.6–

![](images/907e230ab283c48622fed665da7f1c13454bb6ee219db7432aa1c8aea8d50247.jpg)

![](images/f6edbe9cfa32d564c99b41e173a7e029b197692ea6b5a5d07b9e46257adf136a.jpg)  
Figure A: (a) Calibration of the intrinsic advantage: the theoretical prediction $a _ { i } ^ { \star } ( \gamma )$ (Proposition 4.1) closely matches the empirical mean advantage across 100 prompts $\times 5 0$ groups $\left( G { = } 8 \right)$ . The $\pm 1$ std band reflects the composition noise that MaPP eliminates via marginalization. (b) Calibration of the selection score: the posterior-predictive non-zero-variance probability $V _ { G } ^ { \mathrm { M a P P } }$ (Eq. (15)) tracks the empirical effective ratio within a ±0.05 band.

![](images/1615a147b5816afddb031c93488c54be7f25b97350f81c9bf8d4dfd3d27dec94.jpg)  
Figure B: Ablation experiments of our proposed MaPP method under different numbers of rollouts $( n = 4$ and $n = 1 6$ , with $n = 8$ evaluated in the main experiments). The results show that $\mathbf { M a P P }$ performs consistently well across different rollout group sizes.

2.8× at $G { = } 1 6$ . The more pronounced speedup at larger G is consistent with the observation in §5.3 that richer per-step evidence enables sharper advantage corrections.

Visual geometry and sensitivity to λ. Figure C(a) shows training curves for the visual geometry task on Qwen2.5-VL-3B and Qwen2.5-VL-7B. MaPP outperforms the strongest baseline by +1.04 and +2.40 respectively at convergence, reaching comparable accuracy by approximately $1 . { \dot { 4 } } \mathrm { - } 1 . 5 \times$ faster convergence speed, confirming that MaPP generalizes to multi-modal settings. Figure C(b) shows training curves under varying λ on Countdown. λ=0.5 achieves the best final accuracy and convergence speed, while extreme values $( \lambda { = } 0 . 0$ and $\lambda { = } 1 . 0 )$ converge noticeably slower.

## D Theoretical Analysis

All proofs in this section are for a single prompt τ at a fixed training step t; we drop the superscript τ for brevity and write $\alpha _ { j } ^ { \prime } : = \alpha _ { t + 1 ; j } ^ { \prime } , \bar { \beta } _ { j } ^ { \prime } : = \bar { \beta _ { t + 1 ; j } ^ { \prime } }$ following Proposition 4.2. The latent pass rate $\gamma \in ( 0 , 1 )$ is treated as a fixed (but unknown) parameter; expectations $\mathbb { E } [ \cdot \mid \gamma ]$ are taken over the rollout randomness only, while Bayesian quantities are computed under the leave-one-out posterior $\gamma \mid \mathcal { H } _ { t } , r _ { t ; - j } \sim \mathrm { B e t a } ( \bar { \alpha _ { j } ^ { \prime } } , \beta _ { j } ^ { \prime } )$ . Under binary rewards, the GRPO advantage reduces to

$$
\hat { A } _ { j } = h ( r _ { j } , S ) = \left\{ \begin{array} { l l } { + \sqrt { ( G - S ) / S } , } & { r _ { j } = 1 , } \\ { - \sqrt { S / ( G - S ) } , } & { r _ { j } = 0 , } \end{array} \right.\tag{17}
$$

with $\hat { A } _ { j } : = 0$ when $S \in \{ 0 , G \}$ ; correspondingly, both $a _ { j } ^ { \star } ( \gamma )$ and $\hat { A } _ { j }$ drop the degenerate terms, so $\mathbb { E } [ \hat { A } _ { j } \mid r _ { j } , \gamma ] = a _ { j } ^ { \star } ( \gamma )$ for $\gamma \in ( 0 , 1 )$ .

Theorem D.1 (Composition-noise lower bound). For any $G \geq 2$ and $\gamma \in ( 0 , 1 )$

$$
\mathbb { E } \big [ ( \hat { A } _ { j } - a _ { j } ^ { \star } ( \gamma ) ) ^ { 2 } \big | \gamma \big ] = \xi ( \gamma ) > 0 ,\tag{18}
$$

where $\xi ( \gamma )$ depends only on $( \gamma , G )$ and is independent of the posterior or training history.

Proof. By Definition 1, $a _ { j } ^ { \star } ( \gamma ) = \mathbb { E } [ \hat { A } _ { j } \mid r _ { j } , \gamma ]$ . Conditioning on $r _ { j }$ and $\gamma$ , the only remaining randomness comes from the other $G - 1$ responses, so

$$
\mathbb { E } \big [ ( \hat { A } _ { j } - a _ { j } ^ { \star } ( \gamma ) ) ^ { 2 } \mid \gamma \big ] = \mathbb { E } _ { r _ { j } } \big [ \operatorname { V a r } ( \hat { A } _ { j } \mid r _ { j } , \gamma ) \mid \gamma \big ] = : \xi ( \gamma ) .\tag{19}
$$

Letting $k \sim \mathrm { B i n } ( G { - } 1 , \gamma )$ denote the number of correct responses among the other $G - 1$

$$
\begin{array} { r } { \xi ( \gamma ) = \gamma \operatorname { V a r } _ { k } \left[ \sqrt { \frac { G - 1 - k } { k + 1 } } \right] + ( 1 - \gamma ) \operatorname { V a r } _ { k } \left[ \sqrt { \frac { k } { G - k } } \right] . } \end{array}\tag{20}
$$

The map $S \mapsto { \sqrt { ( G - S ) / S } }$ is strictly monotone on $\{ 1 , \ldots , G - 1 \}$ and the binomial is nondegenerate for $\gamma \in ( 0 , 1 )$ , hence $\xi ( \gamma ) > 0$ . Since $\xi$ depends only on $( \gamma , G )$ , this lower bound is structural and cannot be reduced by any prompt-level posterior or selection scheme. □

Theorem D.2 (Frequentist MSE bound for MaPP). $F i x \gamma \in ( 0 , 1 )$ and $G \geq 2 . L e t \tilde { \gamma } : = \alpha _ { j } ^ { \prime } / ( \alpha _ { j } ^ { \prime } + \beta _ { j } ^ { \prime } )$ and $n : = \alpha _ { j } ^ { \prime } + \beta _ { j } ^ { \prime }$ . There exist constants $C _ { 1 } ( \gamma , G ) , C _ { 2 } ( \gamma , G )$ depending only on $( \gamma , G )$ such that

$$
\begin{array} { r } { \mathbb { E } \bigl [ ( \hat { a } _ { j } ^ { \mathrm { M a P P } } - a _ { j } ^ { \star } ( \gamma ) ) ^ { 2 } \big | \gamma \bigr ] \ \leq \ \underbrace { C _ { 1 } ( \tilde { \gamma } - \gamma ) ^ { 2 } } _ { c a l i b r a t i o n \ : b i a s } + \ \underbrace { \frac { C _ { 2 } } { n + 1 } } _ { = O ( G / n ) } \ . } \end{array}\tag{21}
$$

Proof. Decompose the frequentist MSE under the rollout history $\mathcal { H } _ { t }$ into squared bias and variance:

$$
\begin{array} { r } { \mathbb { E } \big [ ( \hat { a } _ { j } ^ { \mathrm { M a P P } } - a _ { j } ^ { \star } ( \gamma ) ) ^ { 2 } \mid \gamma \big ] \ = \ \underbrace { \big ( \mathbb { E } _ { \mathcal { H } _ { t } } [ \hat { a } _ { j } ^ { \mathrm { M a P P } } \mid \gamma ] - a _ { j } ^ { \star } ( \gamma ) \big ) ^ { 2 } } _ { \mathrm { b i a s ^ { 2 } } } + \underbrace { \mathrm { V a r } _ { \mathcal { H } _ { t } } ( \hat { a } _ { j } ^ { \mathrm { M a P P } } \mid \gamma ) } _ { \mathrm { v a r i a n c e } } . } \end{array}\tag{22}
$$

Since $a _ { j } ^ { \star } = \pm \mu _ { G }$ is a degree-(G−1) polynomial with binomial coefficients, its first two derivatives are bounded on compact subsets of $( 0 , 1 )$

$$
L _ { G } : = \operatorname* { s u p } _ { \gamma \in K } \left| ( a _ { j } ^ { \star } ) ^ { \prime } ( \gamma ) \right| = O ( G ) , \qquad M _ { G } : = \operatorname* { s u p } _ { \gamma \in K } \left| ( a _ { j } ^ { \star } ) ^ { \prime \prime } ( \gamma ) \right| = O ( G ^ { 2 } ) , \qquad K \in ( 0 , 1 ) ,\tag{23}
$$

where the constants depend on the distance of $K$ to {0, 1} and are bounded over the non-degenerate regime enforced by MaPP-PS via $V _ { G } ^ { \mathrm { M a P P } }$ . A second-order Taylor expansion of $a _ { j } ^ { \star }$ around γ gives, for $\gamma ^ { \prime }$ in a neighborhood of $\gamma$

$$
\begin{array} { r } { a _ { j } ^ { \star } ( \gamma ^ { \prime } ) = a _ { j } ^ { \star } ( \gamma ) + ( a _ { j } ^ { \star } ) ^ { \prime } ( \gamma ) ( \gamma ^ { \prime } - \gamma ) + R ( \gamma ^ { \prime } , \gamma ) , \qquad | R ( \gamma ^ { \prime } , \gamma ) | \leq \frac { 1 } { 2 } M _ { G } ( \gamma ^ { \prime } - \gamma ) ^ { 2 } . } \end{array}\tag{24}
$$

Taking expectation under the leave-one-out posterior $\gamma ^ { \prime } \sim \mathrm { B e t a } ( \alpha _ { j } ^ { \prime } , \beta _ { j } ^ { \prime } )$ at fixed $\mathcal { H } _ { t }$ , and using $\mathbb { E } _ { \gamma ^ { \prime } } [ \gamma ^ { \prime } ] = \tilde { \gamma }$

$$
\hat { a } _ { j } ^ { \mathrm { M a P P } } = a _ { j } ^ { \star } ( \gamma ) + ( a _ { j } ^ { \star } ) ^ { \prime } ( \gamma ) ( \tilde { \gamma } - \gamma ) + \mathbb { E } _ { \gamma ^ { \prime } } [ R ( \gamma ^ { \prime } , \gamma ) ] .\tag{25}
$$

Bias. Taking expectation of $\operatorname { E q . }$ (25) over $\mathcal { H } _ { t }$ at fixed $\gamma$ and using $\begin{array} { r } { | \mathbb { E } _ { \gamma ^ { \prime } } [ R ] | \leq \frac { 1 } { 2 } M _ { G } \mathbb { E } _ { \gamma ^ { \prime } } [ ( \gamma ^ { \prime } - \gamma ) ^ { 2 } ] } \end{array}$

$$
\begin{array} { r } { \big | \mathbb { E } _ { \mathcal { H } _ { t } } \big [ \hat { a } _ { j } ^ { \mathrm { M a P P } } \mid \gamma \big ] - a _ { j } ^ { \star } ( \gamma ) \big | \ \leq \ L _ { G } \big | \mathbb { E } _ { \mathcal { H } _ { t } } [ \tilde { \gamma } \mid \gamma ] - \gamma \big | + O \big ( M _ { G } / n \big ) , } \end{array}\tag{26}
$$

where the last term absorbs both $\mathbb { E } _ { \gamma ^ { \prime } } [ ( \gamma ^ { \prime } - \tilde { \gamma } ) ^ { 2 } ] = \mathrm { V a r } _ { \gamma ^ { \prime } } ( \gamma ^ { \prime } ) = O ( 1 / n )$ and $( \tilde { \gamma } - \gamma ) ^ { 2 }$ . Squaring, the bias contributes $C _ { 1 } ( \tilde { \gamma } - \gamma ) ^ { 2 } + \overset { \cdot } { O } ( G ^ { 2 } / n ^ { 2 } )$ with $C _ { 1 } \stackrel { . } { = } L _ { G } ^ { 2 } = O ( G ^ { 2 } )$ ; the higher-order term is dominated by the variance term below for $n \geq 1$

Variance. The variance term equals the posterior variance of $a _ { j } ^ { \star } ( \gamma ^ { \prime } )$ under $\gamma ^ { \prime } \sim \mathrm { B e t a } ( \alpha _ { j } ^ { \prime } , \beta _ { j } ^ { \prime } )$ Substituting Eq. (24) and using $\mathrm { V a r } _ { \gamma ^ { \prime } } ( \mathrm { c o n s t a n t s } ) = 0$

$$
\mathrm { V a r } _ { \gamma ^ { \prime } } \big ( a _ { j } ^ { \star } ( \gamma ^ { \prime } ) \big ) \ \leq \ \big [ ( a _ { j } ^ { \star } ) ^ { \prime } ( \gamma ) \big ] ^ { 2 } \mathrm { V a r } _ { \gamma ^ { \prime } } ( \gamma ^ { \prime } ) + O \big ( M _ { G } ^ { 2 } / n ^ { 2 } \big ) \ \leq \ \frac { L _ { G } ^ { 2 } } { 4 ( n + 1 ) } + O ( 1 / n ^ { 2 } ) ,\tag{27}
$$

using $\mathrm { V a r } _ { \gamma ^ { \prime } } ( \gamma ^ { \prime } ) = \alpha _ { j } ^ { \prime } \beta _ { j } ^ { \prime } / [ n ^ { 2 } ( n + 1 ) ] \leq 1 / [ 4 ( n + 1 ) ]$ ]. This gives $C _ { 2 } = L _ { G } ^ { 2 } / 4 = O ( G )$

Combining bias and variance yields Eq. (21).

![](images/414b1d3b73323cf1bf8f70cd00e534d42edd85f4f66dc64b5933e28732987a33.jpg)

![](images/6cfa2fe7726c4ebec749d0571fdae2323ffbc842e6d8ac4b05d713d284844cde.jpg)  
(a) Training on Geometry

![](images/1e127ea748eb577444d91fd6c03d29e3549fb800ab2475adbcda479a27f7ad78.jpg)

(b) Posterior Tracking Discount  
![](images/58588ce29550cec7590bbfa3bb0f8d8b98a67270e6a1ffa8164383559e24dc46.jpg)  
Figure C: (a): Performance of our method trained on the Qwen2.5-VL-Instruct series models and evaluated on the Geometry test set. (b): Self-ablation of MaPP under different posterior tracking discounts. The best performance is achieved at $\lambda = 0 . 5$ (Ours).

Corollary D.3 (Crossover). Combining Theorems D.1 and D.2, there exists a threshold

$$
n ^ { \star } ( \gamma , G ) \leq \frac { C _ { 2 } } { \xi ( \gamma ) - C _ { 1 } ( \tilde { \gamma } - \gamma ) ^ { 2 } } - 1 ,\tag{28}
$$

such thatfor all $n \geq n ^ { \star }$

$$
\begin{array} { r } { \mathbb { E } \big [ ( \hat { a } _ { j } ^ { \mathrm { M a P P } } - a _ { j } ^ { \star } ( \gamma ) ) ^ { 2 } \big | \gamma \big ] \leq \xi ( \gamma ) = \mathbb { E } \big [ ( \hat { A } _ { j } - a _ { j } ^ { \star } ( \gamma ) ) ^ { 2 } \big | \gamma \big ] , } \end{array}\tag{29}
$$

i.e., MaPP strictly improves over $G R P O$ at the per-response level. In practice, this threshold is small under standard initialization $( \alpha _ { 0 } , \beta _ { 0 } ) = ( 1 , \bar { 1 } )$ and is reached after a few posterior updates per prompt, ensuring that MaPP outperforms GRPO throughout the main training phase.

Connection to gradient estimation. The per-response MSE bounds in Theorems D.1 and D.2 transfer to the policy gradient under a standard score-function moment condition. Let $s _ { j } : = \nabla _ { \theta } \log \pi _ { \theta } ( y _ { j } \mid$ τ ) and let $\begin{array} { r l r } { g : = } & { { } \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \hat { a } _ { j } s _ { j } , g ^ { \star } : = } & { \frac { 1 } { G } \sum _ { j = 1 } ^ { G } a _ { j } ^ { \star } ( \gamma ) s _ { j } } \end{array}$ denote the policy gradient and its oracle counterpart, where $\hat { a } _ { j }$ denotes either $\hat { A } _ { j }$ (GRPO) or $\hat { a } _ { j } ^ { \mathrm { M a P P } }$ (MaPP), with per-response error $\epsilon _ { j } : = \hat { a } _ { j } - a _ { j } ^ { \star } ( \gamma )$ . Suppose $\mathbb { E } [ \| s _ { j } \| ^ { 2 } \mid \tau ] \le B ^ { 2 }$ , as is routinely assumed in policy-gradient analysis. A direct application of Cauchy–Schwarz to $\begin{array} { r } { g - g ^ { \star } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } \epsilon _ { j } s _ { j } } \end{array}$ , together with symmetry across responses, yields

$$
\begin{array} { r } { \mathbb { E } \big [ \| g - g ^ { \star } \| ^ { 2 } \big | \gamma \big ] \ \leq \ B ^ { 2 } \cdot \mathbb { E } \big [ \epsilon _ { 1 } ^ { 2 } \big | \gamma \big ] . } \end{array}\tag{30}
$$

Since this bound is monotone in the per-response MSE, the bounds of Theorems D.1 and D.2 immediately yield corresponding gradient-level bounds, and Corollary D.3 likewise transfers: MaPP achieves a smaller gradient MSE upper bound than GRPO whenever n $\geq n ^ { \star }$ . Empirically, Fig. 1(b) and Fig. A show that composition noise produces visible variability in realized advantages across groups, which MaPP suppresses.

## E Derivation of Closed Forms

All derivations in this section are for a single prompt τ ; we drop the superscript τ for brevity and write $\alpha _ { j } ^ { \prime } : = \alpha _ { t + 1 ; j } ^ { \prime } , \beta _ { j } ^ { \prime } : = \beta _ { t + 1 ; j } ^ { \prime }$ following Proposition 4.2.

Intrinsic advantage (Proposition 4.1). Consider $r _ { i } ~ = ~ 1$ ; the case $r _ { j } ~ = ~ 0$ follows by symmetry. Conditional on $r _ { i } = 1$ and γ, the remaining $G - 1$ responses each follow independent Bernoulli(γ) trials, so the total success count satisfies $S \in \{ 1 , \ldots , G \}$ with $\mathbb { P } ( S = s \mid r _ { j } = 1 , \gamma ) =$ $\binom { G - 1 } { s - 1 } \gamma ^ { s - 1 } ( 1 - \gamma ) ^ { G - s }$ . Since $\hat { A } _ { j } = 0$ when $S = G$ , taking expectation gives

$$
a _ { j } ^ { \star } ( \gamma ) = \mathbb { E } \bigl [ \hat { A } _ { j } \mid r _ { j } = 1 , \gamma \bigr ] = \sum _ { S = 1 } ^ { G - 1 } { \binom { G - 1 } { S - 1 } } \gamma ^ { S - 1 } ( 1 - \gamma ) ^ { G - S } \sqrt { \frac { G - S } { S } } = \mu _ { G } ( \gamma ) .\tag{31}
$$

For $r _ { j } = 0$ , by symmetry $a _ { j } ^ { \star } ( \gamma ) = - \mu _ { G } ( 1 - \gamma )$

Posterior-predictive advantage (Proposition 4.2). Marginalizing $a _ { j } ^ { \star } ( \gamma ) = \mu _ { G } ( \gamma )$ against the leave-one-out posterior $\gamma \sim \mathrm { B e t a } ( \alpha _ { j } ^ { \prime } , \beta _ { j } ^ { \prime } )$ and exchanging summation with integration:

$$
\begin{array} { l } { { \hat { a } _ { j } ^ { \mathrm { M a P P } } = \displaystyle \int _ { 0 } ^ { 1 } \mu _ { G } ( \gamma ) ~ { \frac { \gamma ^ { \alpha _ { j } ^ { \prime } - 1 } ( 1 - \gamma ) ^ { \beta _ { j } ^ { \prime } - 1 } } { B ( \alpha _ { j } ^ { \prime } , \beta _ { j } ^ { \prime } ) } } \mathrm { d } \gamma } } \\ { { { } } = \displaystyle \sum _ { S = 1 } ^ { G - 1 } { \binom { G - 1 } { S - 1 } } \sqrt { \frac { G - S } { S } } { \frac { 1 } { B ( \alpha _ { j } ^ { \prime } , \beta _ { j } ^ { \prime } ) } } \underbrace { \int _ { 0 } ^ { 1 } \gamma ^ { \alpha _ { j } ^ { \prime } + S - 2 } ( 1 - \gamma ) ^ { \beta _ { j } ^ { \prime } + G - S - 1 } \mathrm { d } \gamma } _ { = B ( \alpha _ { j } ^ { \prime } + S - 1 , \beta _ { j } ^ { \prime } + G - S ) } } \\ { { { } } = \displaystyle \sum _ { S = 1 } ^ { G - 1 } { \binom { G - 1 } { S - 1 } } { \frac { B ( \alpha _ { j } ^ { \prime } + S - 1 , \beta _ { j } ^ { \prime } + G - S ) } { B ( \alpha _ { j } ^ { \prime } , \beta _ { j } ^ { \prime } ) } } \sqrt { \frac { G - S } { S } } . } \end{array}\tag{32}
$$

The case $r _ { j } = 0$ follows by the change of variable $\gamma ^ { \prime } = 1 - \gamma$ , which swaps $\alpha _ { j } ^ { \prime }$ and $\beta _ { j } ^ { \prime }$

Selection score (Eq. (15)). The selection score operates across prompts, so we restore the superscript τ . By linearity, $V _ { G } ^ { \mathrm { M a P P } } = 1 - \mathbb { E } _ { \gamma \sim \mathrm { B e t a } ( \alpha ^ { \tau } , \beta ^ { \tau } ) } [ \gamma ^ { G } ] - \mathbb { E } [ ( 1 - \dot { \gamma } ) ^ { G } ]$ . The first moment evaluates as

$$
\mathbb { E } [ \gamma ^ { G } ] = \frac { 1 } { B ( \alpha ^ { \tau } , \beta ^ { \tau } ) } \int _ { 0 } ^ { 1 } \gamma ^ { \alpha ^ { \tau } + G - 1 } ( 1 - \gamma ) ^ { \beta ^ { \tau } - 1 } \mathrm { d } \gamma = \frac { B ( \alpha ^ { \tau } + G , \beta ^ { \tau } ) } { B ( \alpha ^ { \tau } , \beta ^ { \tau } ) } ,\tag{33}
$$

and $\mathbb { E } [ ( 1 - \gamma ) ^ { G } ]$ follows by symmetry, giving Eq. (15).

## F Extension to Multi-Level Rewards

The main text presents MaPP under binary rewards for notational clarity, but the posterior-predictive marginalization principle extends naturally to discrete multi-level rewards $r _ { i } ^ { \tau } \bar { \in } \{ v _ { 1 } , \ldots , v _ { L } \}$ by replacing the Beta–Binomial conjugate pair with the Dirichlet–Multinomial pair: the latent difficulty becomes a probability vector $\gamma ^ { \check { \tau } } \in \Delta ^ { L - 1 }$ with a Dirichlet posterior, and the intrinsic advantage is obtained by enumerating over leave-one-out count vectors weighted by the Dirichlet–Multinomial predictive probability. For moderate $G$ and $L ,$ this involves $\overset { - } { \left( { \begin{array} { l } { G + L - \overline { { 2 } } } \\ { L - 1 } \end{array} } \right) }$ terms (e.g., 36 for $G { = } 8 ,$ $L { = } 3 )$ , which is computationally negligible. In this work, all experiments are conducted under binary rewards: the standard {0, 1} setting is used directly, and the {0, 0.1, 1} reward scheme is also treated as binary by merging 0 and 0.1 into the negative class, following MOPPS [Qu et al., 2026a] and related methods. A systematic evaluation of the multi-level extension is left for future work.

## G Data Examples

We provide illustrative data examples for each task below. Prompt templates for MATH and Geometry3k are adopted from the verl framework [Sheng et al., 2025], and the Countdown template follows Pan et al. [2025].

## MATH Data Example

## Prompt:

You have seven bags of gold coins. Each bag has the same number of gold coins. One day, you find a bag of 53 coins. You decide to redistribute the number of coins you have so that all eight bags you hold have the same number of coins. You successfully manage to redistribute all the coins, and you also note that you have more than 200 coins. What is the smallest number of coins you could have had before finding the bag of 53 coins?

Let’s think step by step and output the final answer within \boxed{}.

Ground-Truth Answer:

203

## Countdown Data Example

## Prompt:

A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.

User: Using the numbers [79, 8, 27, 47], create an equation that equals 91. You can use basic arithmetic operations $( + , \dot { - } , \times , \dot { + } )$ , and each number can only be used once. Show your work in <think> </think> tags, and return the final answer in <answer> </answer> tags, for example <answer>(1 + 2)/3</answer>.

Assistant: Let me solve this step by step. <think>

Ground-Truth Answer:

79 + 47 − 27 − 8 = 91

## Geometry3K Data Example

## Prompt:

![](images/88bd18902d99feb0859d20830a37b8cd1b76cd7030436bcb443b2a54e27ca866.jpg)

Find x. Round to the nearest tenth if necessary. Assume that segments that appear to be tangent are tangent.

You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}.