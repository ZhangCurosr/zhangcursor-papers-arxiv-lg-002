# POLICY-ALIGNED PREDICTIVE VERIFICATION FOR ADAPTIVE VLA ACTION-CHUNK EXECUTION

Minkyu Kim<sup>1</sup> Junhoo Lee<sup>2</sup> Seungyeon Kim<sup>1</sup> Suhyun Jeon<sup>1</sup> Baekseung Kim<sup>1</sup> Donghoon Han<sup>3</sup> Nojun Kwak<sup>1</sup> <sup>1</sup>Seoul National University <sup>2</sup>KAIST <sup>3</sup>Dnotitia

## ABSTRACT

Vision-language-action (VLA) policies generate action sequences that reduce repeated inference, but deciding how much of each sequence to execute is critical to task success. Existing adaptive methods make this decision using generation stability or prediction agreement, yet these signals do not establish whether executing a candidate prefix preserves the ability to complete the task. We introduce a policyaligned predictive verifier (PAV) that anticipates the consequences of candidate action prefixes and evaluates their compatibility with successful continuation before execution. Rather than using future prediction to generate actions, our method evaluates existing proposals in the VLA’s internal task-conditioned representation space. To learn local judgments from terminal task outcomes, we combine trajectory-level survival supervision with a localization prior that preserves useful behavior within failed rollouts. The resulting validity scores guide executionhorizon selection. Evaluations with π<sub>0</sub> and π<sub>0.5</sub> across LIBERO, RoboTwin 2.0, and real-world manipulation show improved success–query trade-offs. These findings point toward a different role for future prediction in robotics, using it to assess how far to trust a policy’s proposed actions rather than to generate new ones.

## 1 INTRODUCTION

Vision-Language-Action (VLA) models have shown strong performance in generating robot actions from visual observations and language instructions (Brohan et al., 2023; Kim et al., 2024; Black et al., 2024; Miao et al., 2026). Yet querying a large VLA policy at every control step is costly, while robot behavior requires coordinated motion over multiple steps. Action chunking addresses both demands by generating a sequence of future actions in one query, supporting temporal coherence with fewer policy calls. However, executing an entire chunk without new observations limits responsiveness to environmental changes and allows execution errors to accumulate. Many systems therefore use a shorter execution horizon to execute only an initial prefix of each chunk before replanning from new observations. Adapting this horizon is thus

![](images/c656b01b1e7f65660d1b8625e55fea8782eac223814332f911ee99b5bfe07d73.jpg)  
Figure 1: Generation stability (a) and prediction agreement (b) can remain high despite a missed grasp and task failure. Proposal validity (c) guides earlier replanning, leading to success in this example.

central to retaining the efficiency of action chunking while remaining responsive to feedback.

Adaptive execution seeks the longest prefix of each action chunk that still allows the policy to complete the task successfully. Identifying this prefix requires anticipating how the proposed actions will affect the policy’s ability to complete the remaining task. Because the effects of a candidate prefix on task success are not directly observable before execution, existing approaches rely on indirect signals to guide execution. Some methods use generation stability as this indirect signal, drawing on internal policy information to select an execution horizon (Wang et al., 2026; Feng et al., 2026; Chen et al., 2026). Others use prediction agreement, comparing action-conditioned predictions with incoming observations to guide correction or replanning during execution (Pan et al., 2026; Liu et al., 2026). Generation stability and prediction agreement alone, however, do not establish whether a candidate prefix preserves compatibility with successful task completion. As illustrated in Figure 1, an action proposal can be consistently generated and its consequences accurately predicted, yet executing it can still lead to a missed grasp and task failure.

To assess this compatibility directly, we introduce proposal validity, which estimates the probability that the policy can successfully complete the task after executing a candidate prefix. This probability depends on the state reached after executing the prefix, because the policy must complete the remaining task from that state. We therefore predict this future state to evaluate whether it supports successful continuation under the policy. World models and World Action Models have demonstrated this capability, building on pretrained visual representations or video models to predict future observations or latent states for planning and action generation (Seo et al., 2022; Du et al., 2023; Zhou et al., 2024; Assran et al., 2025; Ye et al., 2026; Sun et al., 2026). However, because VLA policies already generate effective action proposals, we use this predictive capability to evaluate those proposals rather than to generate new actions. To ground this evaluation in the same information used for action generation, we predict the effects of each prefix in the VLA’s task-conditioned representation space, which integrates visual observations and language instructions.

We therefore propose a policy-aligned predictive verifier (PAV) that assesses each action extension’s compatibility with successful continuation from predicted VLA representations. These extensionlevel assessments provide local validity estimates used to select execution horizons. To learn these local judgments, however, we must rely on rollouts that provide only terminal task outcomes. These outcomes do not reveal the validity of individual extensions, since even a failed rollout may contain valid action prefixes. We therefore treat extension-level validity as latent and supervise its accumulated product using the terminal outcome. However, the model can fit a failed rollout by slightly lowering validity scores across the rollout, including those of valid extensions. We therefore introduce a localization prior that encourages the model to concentrate these reductions in fewer extensions. At inference time, we combine the local validity scores to evaluate each candidate prefix, accounting for its length. We use these prefix-level scores to decide how many actions to execute before obtaining a new observation and querying the policy again.

Our experiments with $\pi _ { 0 }$ and $\pi _ { 0 . 5 }$ on LIBERO, RoboTwin 2.0, and real-world tasks assess whether proposal validity selects execution horizons that improve success with fewer policy queries. To examine the basis for these decisions, prediction tests assess action sensitivity, while ablations examine future prediction, policy alignment, and the contributions of individual loss terms. Predicting future VLA representations achieves higher success with fewer queries than direct verification or prediction in generic visual features.

## 2 RELATED WORK

## 2.1 ADAPTIVE EXECUTION OF ACTION CHUNKS

Action-sequence policies such as ACT and $\pi _ { 0 }$ generate future action chunks (Zhao et al., 2023; Black et al., 2024). A fixed execution horizon uses the same prefix length across proposals and states, trading fewer queries for later feedback. Adaptive execution instead adjusts when to request a new proposal. One line of work uses internal policy signals to assess generation stability and guide horizon selection. AutoHorizon estimates an execution limit from action self-attention (Wang et al., 2026), while $A ^ { 3 }$ selects an execution horizon using action consensus and re-decoding stability (Chen et al., 2026). Another line of work assesses prediction agreement from observations during execution. VLA-Corrector truncates and replans when predicted and observed visual dynamics persistently diverge, whereas CheckVLA relates prediction-observation discrepancies to failure risk and latency-aware suffix repair (Pan et al., 2026; Liu et al., 2026). These approaches use generation stability or prediction agreement to guide execution, whereas our verifier predicts the consequences of candidate prefixes and evaluates their compatibility with successful continuation before execution. To our knowledge, we are the first to combine prefix-conditioned future prediction with proposalvalidity estimation for adaptive action-chunk execution.

## 2.2 WORLD MODELS AND PREDICTIVE REPRESENTATIONS

World models capture action-conditioned dynamics over future observations or latent states (Chua et al., 2018; Hafner et al., 2019b; Hansen et al., 2022; 2024). Dreamer learns behavior from imagined latent trajectories (Hafner et al., 2019a), DINO-WM predicts pretrained visual features for goal-directed planning (Zhou et al., 2024), and World Action Models such as DreamZero jointly predict future video and actions (Ye et al., 2026). Value Prediction Networks evaluate imagined futures through rewards and values (Oh et al., 2017), while Diffuser generates trajectories for planning (Janner et al., 2022). Beyond these uses, prediction can support proposal verification. Given a VLA proposal, future prediction anticipates the consequences of executing candidate prefixes. Assessing these consequences requires representations that retain information relevant to control. Deep-MDP and bisimulation-based learning connect representations to rewards and transitions (Gelada et al., 2019; Zhang et al., 2020), while VIP learns visual representations through goal-conditioned value learning (Ma et al., 2022). We therefore predict prefix-induced futures in the action-generating policy’s internal representation space, retaining the joint observation and instruction context for outcome-based proposal verification.

## 2.3 LEARNING FROM TRAJECTORY-LEVEL OUTCOMES

Learning prefix-level validity from terminal outcomes is related to multiple-instance learning, where collection-level labels leave individual instance labels unobserved (Dietterich et al., 1997; Ilse et al., 2018). Here, rollout outcomes supervise temporally ordered prefix extensions without identifying which extension caused a failure. Reward redistribution similarly derives local learning signals from delayed returns or demonstrations (Arjona-Medina et al., 2019; Ren et al., 2021; Patil et al., 2020). A related credit-assignment problem arises in language-model reinforcement learning (Chan et al., 2024), where RED derives token-level rewards from holistic response feedback (Li et al., 2025). These approaches recover local information from global outcomes (Yuan et al., 2024), but we use this supervision to verify proposed execution and select horizons rather than optimize the actiongenerating policy.

## 3 METHOD

## 3.1 PRELIMINARIES

Action-chunk execution. Adaptive action-chunk execution aims to select the longest prefix likely to preserve compatibility with successful continuation. Given observation $o _ { t }$ and instruction ℓ at decision time t, policy π generates a length-H chunk (Zhao et al., 2023; Black et al., 2024). With $\mathbf { a } _ { t , j }$ its j-th action, choosing an execution horizon h from $[ H ] \triangleq \{ 1 , \dots , H \}$ commits to the prefix

$$
\mathbf { A } _ { t } \sim \pi ( \cdot \mid o _ { t } , \ell ) , \qquad \mathbf { A } _ { t } ^ { 1 : h } = ( \mathbf { a } _ { t , 1 } , \ldots , \mathbf { a } _ { t , h } ) , \qquad h \in [ H ] .\tag{1}
$$

After h control steps, the policy receives $o _ { t + h }$ and replans. Choosing h therefore balances inference cost and temporal consistency against responsiveness (Wang et al., 2026).

Generation stability as an execution criterion. Because future execution outcomes are unavailable when choosing a horizon, $A ^ { 3 }$ (Chen et al., 2026) tests whether proposed actions remain stable when other actions in the chunk are fixed as context. Let $\widetilde { \mathbf { a } } _ { t , j }$ denote the action regenerated under such a context. This generation-stability signal can be expressed conceptually through the discrepancy

$$
\begin{array} { r } { e _ { t } ( j ) = \| \widetilde { \mathbf { a } } _ { t , j } - \mathbf { a } _ { t , j } \| _ { 2 } ^ { 2 } . } \end{array}\tag{2}
$$

A small discrepancy indicates that the policy reproduces the proposed action under the imposed context. $A ^ { 3 }$ therefore accepts the longest prefix that passes its conditional consistency checks. However, consistently generated actions can still prevent later task completion, so generation stability does not establish compatibility with successful continuation.

Prediction agreement as an execution criterion. Prediction agreement provides evidence for intervention by comparing action-conditioned predictions with observations collected during execution (Pan et al., 2026; Liu et al., 2026). To express this signal generically, let $x _ { t ^ { \prime } }$ denote an observed feature statistic at control step $t ^ { \prime } > t , \hat { x } _ { t ^ { \prime } }$ its earlier prediction, and d a discrepancy measure, giving

$$
d _ { t ^ { \prime } } = d ( \hat { x } _ { t ^ { \prime } } , x _ { t ^ { \prime } } ) .\tag{3}
$$

A larger discrepancy indicates weaker prediction agreement and provides evidence for correction or replanning during execution. Our decision instead requires evaluating the compatibility of a candi date prefix before the corresponding observations become available.

## 3.2 FORMULATING PROPOSAL VALIDITY

Defining the decision target. We therefore evaluate candidate prefixes by their compatibility with successful continuation rather than by generation stability or prediction agreement alone. Proposal validity quantifies this compatibility as the probability that successful continuation remains possible throughout the proposed execution. To assess the whole prefix, let $\tau _ { t } ^ { 1 : h }$ denote the trajectory induced by the first h actions of a fixed proposal $\mathbf { A } _ { t } ,$ , and let $\mathcal { T } _ { \ell } ^ { \pi , + }$ contain trajectories from which successful continuation under π remains possible after every extension. We include task progress and remaining time in the state to assess that possibility, and hold the continuation protocol constant to isolate the consequences of the candidate prefixes. Formally,

$$
V _ { t } ( h ) : = \mathrm { P r } \big ( \tau _ { t } ^ { 1 : h } \in { \mathscr { T } _ { \ell } ^ { \pi , + } } \mid o _ { t } , \ell , \mathbf { A } _ { t } \big ) .\tag{4}
$$

Requiring compatibility throughout execution makes $\mathcal T _ { \ell } ^ { \pi , + }$ prefix-closed, so a compatible trajectory has compatible shorter prefixes. This gives $V _ { t } ( h )$ the cumulative structure needed to judge how far execution can proceed while preserving compatibility.

To reduce queries while preserving compatibility, an oracle would select the longest prefix with sufficiently high validity. Let ${ \mathcal { H } } \subseteq [ H ]$ denote the candidate execution horizons and $\epsilon \in ( 0 , 1 )$ a small tolerance. The target horizon is

$$
h _ { t } ^ { \mathrm { t a r g e t } } = \operatorname* { m a x } ( \{ \operatorname* { m i n } \mathcal { H } \} \cup \{ h \in \mathcal { H } \mid V _ { t } ( h ) \geq 1 - \epsilon \} ) .\tag{5}
$$

Decomposing proposal validity. To estimate validity before executing progressively longer prefixes, we separate the conditional compatibility of each additional step from the validity accumulated over preceding steps. Since a compatible prefix has compatible predecessors,

$$
V _ { t } ( h ) = V _ { t } ( h - 1 ) \operatorname* { P r } \bigl ( \tau _ { t } ^ { 1 : h } \in \mathscr { T } _ { \ell } ^ { \pi , + } \big | \tau _ { t } ^ { 1 : h - 1 } \in \mathcal { T } _ { \ell } ^ { \pi , + } , o _ { t } , \ell , \mathbf { A } _ { t } \bigr ) .\tag{6}
$$

We set $V _ { t } ( 0 ) = 1$ because no proposed action has yet been executed, and consider the conditional factor where $V _ { t } ( h - 1 ) > 0$ . Assessing this factor before execution requires anticipating the extension’s consequence. We therefore marginalize over its future policy representation $z _ { t + h }$ to separate prediction from verification,

$$
\begin{array} { r l } & { V _ { t } ( h ) = V _ { t } ( h - 1 ) \int \underbrace { p \big ( z \mid \tau _ { t } ^ { 1 : h - 1 } \in \mathcal { T } _ { \ell } ^ { \pi , + } , o _ { t } , \ell , \mathbf { A } _ { t } \big ) } _ { \mathrm { p r e d i c i o n } } } \\ & { \qquad \times \underbrace { \mathrm { P r } \big ( \tau _ { t } ^ { 1 : h } \in \mathcal { T } _ { \ell } ^ { \pi , + } \mid z _ { t + h } = z , \tau _ { t } ^ { 1 : h - 1 } \in \mathcal { T } _ { \ell } ^ { \pi , + } , o _ { t } , \ell , \mathbf { A } _ { t } \big ) } _ { \mathrm { v e r f i f c a t i o n } } \mathrm { d } z . } \end{array}\tag{7}
$$

Here $p$ denotes the conditional density of $z _ { t + h }$ . This decomposition motivates a predictor trained on observed futures and a verifier trained on task outcomes.

## 3.3 LEARNING PREDICTION AND VERIFICATION

Predicting action consequences in policy space. To approximate the prediction factor in Equation 7, we use a learned point prediction $\hat { z } _ { t + h | t }$ for verification, as illustrated in Figure 2. To retain the task context used for action selection, we predict in the policy’s representation space, with $z _ { t } = E _ { \pi } ( o _ { t } , \ell )$ denoting the features extracted by $E _ { \pi }$ from the observation and instruction. Encoding observations after executed prefixes supplies targets for the predictor $f _ { \theta }$

$$
z _ { t + h } = E _ { \pi } ( o _ { t + h } , \ell ) , \qquad \hat { z } _ { t + h | t } = f _ { \theta } ( z _ { t } , \mathbf { A } _ { t } ^ { 1 : h } ) .\tag{8}
$$

Following the use of observed future representations as prediction targets (Schwarzer et al., 2020), we align our predictions with targets from executed transitions $( o _ { t } , \ell , \mathbf { A } _ { t } ^ { 1 : h } , o _ { t + h } )$ collected in $\mathcal { D } _ { \mathrm { t r a n s } } .$ . We use cosine distance $d _ { \mathrm { c o s } } ( u , v ) = 1 - \mathrm { s i m } ( u , v )$ , where sim is cosine similarity. Alignment alone may allow the predictor to ignore which actions produced the future. We therefore pair transitions with alternative prefixes $\tilde { \mathbf { A } } _ { t } ^ { 1 : h }$ in $\mathcal { D } _ { \mathrm { s w a p } } ,$ giving $\tilde { z } _ { t + h | t } = f _ { \theta } ( z _ { t } , \tilde { \mathbf { A } } _ { t } ^ { 1 : h } )$ ). Requiring the executed prefix’s predicted representation to be closer to the target by margin $m > 0$ encourages action sensitivity without assigning the alternative a failure label. With nonnegative weights $\lambda _ { \mathrm { d y n } } , \lambda _ { \mathrm { s w a p } }$ and $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ , these requirements give

![](images/831059df7058c93ea86fbfdf181d5ca259e99365626c9f8e97d390ebb4d15c1f.jpg)  
Figure 2: Policy-aligned predictive verification. The predictor uses the VLA representation and proposed actions to predict future policy representations. The verifier estimates local validity, which is accumulated and normalized to select an execution horizon. The robot executes the selected prefix and replans.

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p r e d } } ( \theta ) = \lambda _ { \mathrm { d y n } } \mathbb { E } _ { \mathcal { D } _ { \mathrm { t r a n s } } } [ d _ { \cos } ( \hat { z } _ { t + h | t } , z _ { t + h } ) ] } \\ & { \qquad +  \lambda _ { \mathrm { s w a p } } \mathbb { E } _ { \mathcal { D } _ { \mathrm { s w a p } } } [ [ m + d _ { \cos } ( \hat { z } _ { t + h | t } , z _ { t + h } ) - d _ { \cos } ( \hat { z } _ { t + h | t } , z _ { t + h } ) ] _ { + } ] . } \end{array}\tag{9}
$$

Learning local validity from trajectory outcomes. Having predicted each prefix’s consequence, we assess its compatibility with successful continuation through the verification factor in Equation $^ { 7 . }$ We use the predicted representation to summarize the context for this conditional judgment and train a verifier $q _ { \phi }$ from task outcomes. Its output $q _ { t } ( h ) \in ( 0 , 1 )$ at $\hat { z } _ { t + h | t }$ estimates local validity. To accumulate these conditional estimates into a prefix score, we follow Equation $^ { 6 , }$ starting from $\hat { S } _ { t } ( 0 ) = 1$

$$
q _ { t } ( h ) = \frac { \hat { S } _ { t } ( h ) } { \hat { S } _ { t } ( h - 1 ) } = q _ { \phi } ( \hat { z } _ { t + h | t } ) , \hat { S } _ { t } ( h ) = \prod _ { j = 1 } ^ { h } q _ { t } ( j ) ,\tag{10}
$$

Thus $q _ { \phi } \big ( f _ { \theta } \big ( z _ { t } , \mathbf { A } _ { t } ^ { 1 : h } \big ) \big )$ implements the prediction–verification approximation, and its accumulated product estimates $V _ { t } ( h )$ S̄) without assuming independent extensions.

Terminal outcomes provide trajectory-level evidence rather than direct labels for local compatibility. We therefore extend score accumulation across policy queries and use these outcomes as surrogate supervision, leaving extension-level validity latent. Let $\mathcal { D } ^ { + }$ and $\mathcal { D } ^ { - }$ collect successful and failed rollouts, with $\mathcal { D } = \mathbf { \bar { \mathcal { D } } ^ { + } } \cup \mathcal { D } ^ { - }$ . For rollout $\tau ,$ let $\mathcal { T } _ { \tau }$ index its executed extensions, $n _ { \tau } = | \mathcal { I } _ { \tau } | .$ , and $q _ { i }$ denote the corresponding $q _ { t } ( h )$ , giving $\begin{array} { r } { \hat { S } _ { \tau } = \prod _ { i \in \mathcal { I } _ { \tau } } q _ { i } } \end{array}$ . Success supports validity throughout, so we raise every factor, averaging over $n _ { \tau }$ to avoid overweighting long episodes. For failed rollouts, we lower only the product so that the model can retain high scores for compatible extensions without assigning a failure label to each one. This trajectory-level supervision yields

$$
\mathcal { L } _ { \mathrm { t r a j } } ( \tau ) = \left\{ \begin{array} { l l } { \displaystyle - \frac { 1 } { n _ { \tau } } \log \hat { S } _ { \tau } , } & { \tau \in \mathcal { D } ^ { + } , } \\ { \displaystyle - \log \big ( 1 - \hat { S } _ { \tau } \big ) , } & { \tau \in \mathcal { D } ^ { - } . } \end{array} \right.\tag{11}
$$

Leaving failure attribution latent avoids labeling every extension as a failure, but the product can still decrease through small validity reductions across the rollout, including at valid extensions. We

Table 1: LIBERO success rates (%) and mean VLA queries per episode. Our method and fixedhorizon baselines use 50 trials per task. <sup>†</sup> denotes published reference results with potentially different protocols. Bold and underlining indicate the highest and second-highest success rates within each policy setting, including ties. Dashes indicate unreported results.
<table><tr><td colspan="7"> $\pi _ { 0 }$  (H = 50)</td></tr><tr><td>Method</td><td>Spatial ↑</td><td>Object ↑</td><td>Goal ↑</td><td>Long ↑</td><td>Avg. ↑</td><td>Queries↓</td></tr><tr><td>h = 5 (Baseline)</td><td>97.4</td><td>97.2</td><td>91.0</td><td>82.8</td><td>92.10</td><td>34.6</td></tr><tr><td>h = 10</td><td>96.0</td><td>98.4</td><td>93.6</td><td>84.2</td><td>93.05</td><td>17.2</td></tr><tr><td>h = 30</td><td>89.4</td><td>94.4</td><td>90.2</td><td>86.6</td><td>90.15</td><td>6.3</td></tr><tr><td>h = 50</td><td>76.6</td><td>75.4</td><td>83.4</td><td>68.8</td><td>76.05</td><td>4.4</td></tr><tr><td>AutoHorizon</td><td>95.4</td><td>98.4</td><td>94.8</td><td>87.6</td><td>94.05</td><td>10.6</td></tr><tr><td>DVAC†</td><td>65.0</td><td>76.0</td><td>76.0</td><td>38.0</td><td>63.75</td><td>6.7</td></tr><tr><td>PAV (Ours)</td><td>96.4</td><td>98.8</td><td>94.8</td><td>87.2</td><td>94.30</td><td>12.3</td></tr><tr><td colspan="7"> $\pi _ { 0 . 5 }$  (H = 10)</td></tr><tr><td>Method</td><td>Spatial ↑</td><td>Object ↑</td><td>Goal ↑</td><td>Long ↑</td><td>Avg. ↑</td><td>Queries ↓</td></tr><tr><td>h = 5 (Baseline)</td><td>98.6</td><td>99.0</td><td>98.2</td><td>91.4</td><td>96.80</td><td>32.3</td></tr><tr><td>h = 10 AutoHorizon</td><td>98.0</td><td>99.6</td><td>98.2</td><td>94.2</td><td>97.50</td><td>16.1</td></tr><tr><td>DVAC†</td><td>99.8</td><td>99.6</td><td>97.2</td><td>94.6</td><td>97.8</td><td>27.8</td></tr><tr><td> $A ^ { 3 }$ </td><td>99.0</td><td>98.0</td><td>98.0</td><td>97.0</td><td>98.00</td><td>18.6</td></tr><tr><td>VLA-Corrector†</td><td>99.2</td><td>99.6</td><td>97.4</td><td>94.0</td><td>97.55</td><td>16.4</td></tr><tr><td></td><td>100.0</td><td>99.8</td><td>98.0</td><td>93.4</td><td>97.80</td><td></td></tr><tr><td>PAV (Ours)</td><td>99.0</td><td>99.4</td><td>98.4</td><td>95.6</td><td>98.10</td><td>16.8</td></tr><tr><td colspan="7"></td></tr><tr><td>Method</td><td>Spatial ↑</td><td> $\pi _ { 0 . 5 }$  Object ↑</td><td>(H = 50) Goal ↑</td><td>Long ↑</td><td>Avg. ↑</td><td>Queries ↓</td></tr><tr><td> $\overline { { h = 5 } }$ </td><td>91.8</td><td>94.8</td><td>84.8</td><td>77.2</td><td>87.15</td><td>38.1</td></tr><tr><td>h = 10</td><td>89.8</td><td>92.8</td><td>88.0</td><td>79.2</td><td>87.45</td><td>19.2</td></tr><tr><td> $h = 3 0$ </td><td>68.4</td><td>72.4</td><td>79.6</td><td>69.8</td><td>72.55</td><td>7.8</td></tr><tr><td> $h = 5 0$ </td><td>53.2</td><td>55.0</td><td>72.4</td><td>46.6</td><td>56.80</td><td>5.4</td></tr><tr><td>AutoHorizon</td><td>95.6</td><td>97.2</td><td>92.4</td><td>92.6</td><td>94.45</td><td>12.7</td></tr><tr><td>PAV (Ours)</td><td>96.8</td><td>98.8</td><td>91.8</td><td>91.2</td><td>94.65</td><td>14.0</td></tr></table>

therefore penalize $\textstyle \sum _ { i } ( 1 - q _ { i } )$ on failed rollouts. At a fixed product, minimizing this sum favors concentrating the decrease in fewer extensions, allowing high validity elsewhere. With nonnegative weights $\lambda _ { \mathrm { t r a j } }$ and $\lambda _ { \mathrm { l o c } }$ for outcome supervision and localization, the verification objective becomes

$$
\mathcal { L } _ { \mathrm { v e r } } ( \theta , \phi ) = \lambda _ { \mathrm { t r a j } } \mathbb { E } _ { \tau \sim \mathcal { D } } [ \mathcal { L } _ { \mathrm { t r a j } } ( \tau ) ] + \lambda _ { \mathrm { l o c } } \mathbb { E } _ { \tau \sim \mathcal { D } ^ { - } } \left[ \frac { 1 } { n _ { \tau } } \sum _ { i \in \mathcal { I } _ { \tau } } ( 1 - q _ { i } ) \right] .\tag{12}
$$

Joint optimization. To make predicted representations informative for verification as well as faithful to execution, we jointly minimize $\mathcal { L } _ { \mathrm { p r e d } } ( \theta ) + \mathcal { L } _ { \mathrm { v e r } } ( \theta , \phi )$ . Outcome supervision thus reaches $f _ { \theta }$ through the predicted representations used by q<sub>ϕ</sub>. Appendix A.2 details the implementation, including label smoothing, minibatch weighting, and auxiliary terminal supervision.

## 3.4 EXECUTION-HORIZON SELECTION

We apply length normalization to the accumulated validity score to moderate its dependence on prefix length, obtaining $\bar { S } _ { t } ( h )$ for horizon selection. Appendix C.5 provides the formulation. We check candidate prefixes in ascending order and stop extending the horizon at the first score below $\beta \in ( 0 , 1 )$ , giving the execution rule

$$
h _ { t } ^ { \star } = \operatorname* { m a x } \Bigl ( \{ \operatorname* { m i n } \mathcal { H } \} \cup \Big \{ h \in \mathcal { H } \ \Big | \ \operatorname* { m i n } _ { j \in \mathcal { H } , j \leq h } \bar { S } _ { t } ( j ) \geq \beta \Big \} \Bigr ) .\tag{13}
$$

The robot executes the selected prefix and queries the policy from the resulting observation.

Table 2: RoboTwin 2.0 with $\pi _ { 0 . 5 }$ $H = 3 2$ (baseline $h =$ 32). Success (%) and mean queries cover DVAC’s Selected 16 and All 50 tasks. † marks published results, and dashes mark unreported values.
<table><tr><td></td><td colspan="2">Selected 16</td><td colspan="2">All 50</td></tr><tr><td>Method</td><td>Success ↑</td><td>Queries ↓</td><td>Success ↑</td><td>Queries↓</td></tr><tr><td> $h = 4$ </td><td>17.38</td><td>181.3</td><td>27.40</td><td>136.5</td></tr><tr><td>h = 8</td><td>46.75</td><td>75.0</td><td>55.76</td><td>54.5</td></tr><tr><td> $h = 2 0$ </td><td>47.88</td><td>28.2</td><td>55.16</td><td>20.8</td></tr><tr><td> $h = 3 2$ </td><td>35.38</td><td>19.7</td><td>43.64</td><td>14.6</td></tr><tr><td>DVAC†</td><td>41.6</td><td></td><td></td><td></td></tr><tr><td>PAV (Ours)</td><td>54.13</td><td>59.8</td><td>58.64</td><td>58.5</td></tr></table>

Table 3: LIBERO ablations with $\pi _ { 0 . 5 }$ $H = 5 0 .$ Success is in %, with overall mean queries. PAV matches the setting and results in Table 1.
<table><tr><td>Suite / metric</td><td>Direct</td><td>DINOv2</td><td>PAV</td></tr><tr><td>Spatial</td><td>83.6</td><td>85.4</td><td>96.8</td></tr><tr><td>Object</td><td>91.8</td><td>94.2</td><td>98.8</td></tr><tr><td>Goal</td><td>85.0</td><td>89.8</td><td>91.8</td></tr><tr><td>Long</td><td>77.8</td><td>78.6</td><td>91.2</td></tr><tr><td>Avg. success ↑</td><td>84.55</td><td>87.00</td><td>94.65</td></tr><tr><td>Avg. queries ↓</td><td>22.3</td><td>22.2</td><td>14.0</td></tr></table>

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Models and Benchmarks. We evaluate two representative flow-based VLAs, $\pi _ { 0 }$ and $\pi _ { 0 . 5 } ,$ to test the execution criterion across actiongenerating policies (Black et al., 2024; Intelligence et al., 2025). On LIBERO, we use the Spatial, Object, Goal, and Long suites to compare spatial relations, object identity, goal variation, and long-horizon composition in singlearm manipulation (Liu et al., 2023), excluding LIBERO-90. We evaluate $\pi _ { 0 }$ with $H = 5 0$ and $\pi _ { 0 . 5 }$ with $H = 1 0$ . To examine a wider range of execution lengths, we also use a $\pi _ { 0 . 5 }$ checkpoint fine-tuned with $H = 5 0$ , although checkpoint differences prevent isolating the effect of chunk length. We then evaluate $\pi _ { 0 . 5 }$ with its default $H = 3 2$ on all 50 RoboTwin 2.0 tasks to test dual-arm manipulation with varied objects and scenes (Chen et al., 2025). Finally, taskfine-tuned $\pi _ { 0 . 5 }$ policies with $H = 5 0$ test physical placement, stacking, and contact tasks. Together, these settings assess the criterion across task structures and manipulation setups (Figure

![](images/95a04e77892faf52be4da91e26c71da86150a6411ff8840b057c772d831fb6c4.jpg)

![](images/62ef8609845adcf69d9b73e717175d83d09ab198b7da7745c22c7f81768a5070.jpg)

![](images/dbad993ed9f6fd7dafd8ce1dba301ce4d60ff8e06ac76222370bc68da1966443.jpg)

(a) Real-world tasks  
![](images/91799361c3ffe6ca36d4c8897fdc582ab73abb18577f9b02a48402e3a16b4db2.jpg)

![](images/6c4c41c34a908c2906a578f635d1922326b307d2ebac533409524ad0607d012c.jpg)  
(b) Simulation tasks  
Figure 3: Experimental settings. (a) Successful real-world task states for Press Button, Stack Cubes, and Place Cube in Tray, shown from left to right. (b) Representative simulation environments from LIBERO and RoboTwin 2.0, covering single-arm tabletop and dual-arm manipulation.

Training and implementation details. To expose the verifier to varied execution lengths, we collect rollouts with randomly sampled horizons and balance successful and failed outcomes during training. In simulation, one verifier is shared across tasks within each policy configuration and benchmark, with evaluation trajectories excluded from training. To isolate method differences, corresponding ablations reuse the same training data, while controlled comparisons share the policy checkpoint and action-generation settings. Appendix A provides the architecture, data collection and splits, optimization settings, and execution parameters.

Metrics. We report success over 50 trials per simulation task and 30 per real-world task alongside mean VLA queries per episode to assess performance relative to inference cost. Each action-chunk generation counts as one query. Selected-horizon distributions characterize execution choices, while verifier computation time measures the added cost of horizon selection.

Table 4: Future-prediction cosine similarity on held-out $\pi _ { 0 . 5 } ,$ H = 50 LIBERO rollouts, with the context and target held fixed.
<table><tr><td>Prediction input</td><td>Cosine ↑</td></tr><tr><td>Executed actions</td><td>0.9401</td></tr><tr><td>Same-task swapped actions</td><td>0.5304</td></tr><tr><td>Reversed actions</td><td>0.4816</td></tr><tr><td>Zero actions</td><td>0.4103</td></tr></table>

Table 5: Action-generation and verifier GPU compute times in ms. Parentheses give verifier time relative to generation.
<table><tr><td>Setting</td><td>Generation</td><td>Verifier</td></tr><tr><td> $\pi _ { 0 } .$  LIBERO  $H = 5 0$ </td><td>180.34</td><td>2.24 (1.24%)</td></tr><tr><td> $\pi _ { 0 . }$  5, LIBERO H = 10</td><td>201.19</td><td>0.90 (0.45%)</td></tr><tr><td> $\pi _ { 0 . 5 } ,$  LIBERO H = 50</td><td>216.95</td><td>2.49 (1.15%)</td></tr><tr><td> $\pi _ { 0 . 5 } ,$  RoboTwin  $H = 3 2$ </td><td>192.20</td><td>2.08 (1.08%)</td></tr></table>

## 4.2 SIMULATION EXPERIMENTS

## 4.2.1 LIBERO

Table 1 compares our method with fixed horizons, AutoHorizon, and $A ^ { 3 }$ using their public implementations (Wang et al., 2026; Chen et al., 2026). Published DVAC and VLA-Corrector results, marked with †, provide additional references for generation stability and prediction agreement (Feng et al., 2026; Pan et al., 2026).

Our method achieves the highest average success across all three policy–chunk settings. The clearest gain over fixed execution occurs with $\pi _ { 0 . 5 }$ at $H = 5 0$ , where it improves success by 7.50 percentage points over $h = 5$ while reducing queries by 63.3%. At $H = \mathrm { { i 0 } }$ , it raises success from 97.50% to 98.10% with similar query counts, showing benefits even with fewer execution choices. Appendices C.2 and C.5 examine comparisons at similar query counts and the effect of length normalization, respectively.

## 4.2.2 ROBOTWIN 2.0

Table 2 compares our method with fixed horizons $h \in \{ 4 , 8 , 2 0 , 3 2 \}$ across all 50 tasks and DVAC’s 16-task subset for comparison with its published results (Feng et al., 2026). Appendix B provides per-task results for the Selected 16 tasks.

Our method achieves 54.13% success on the selected 16 tasks, exceeding the best fixed horizon (47.88%) and DVAC (41.6%). This advantage extends to all 50 tasks, where it reaches 58.64% versus 55.76% for the best fixed horizon $( h = 8 )$ , with similar query counts (58.5 versus 54.5).

## 4.2.3 ANALYSIS OF PREDICTION AND POLICY ALIGNMENT

We first test whether the predictor captures the effects of proposed actions, then compare our method against verification without future prediction (Direct) and prediction in DINOv2 feature space to assess their effects on task success and policy-query cost (Table 3). We conduct these experiments on the four LIBERO suites using $\pi _ { 0 . 5 }$ with $\bar { H } = 5 0$ , where ten candidate execution lengths allow us to examine horizon selection over a wider range than the two choices available with $\bar { H _ { \mathrm { } } } = 1 0$

Does the predictor capture action consequences? For a predicted representation to inform proposal evaluation, it must capture the effects of the proposed actions. On held-out rollouts, we therefore compare predicted and observed future latents while holding the current input and observed target fixed. Replacing the executed actions with another same-task sequence tests sensitivity to action content. Table 4 compares predictions conditioned on executed, same-task swapped, reversed, and zero actions. The prediction conditioned on the executed actions achieves a cosine similarity of 0.9401, compared with 0.5304 for swapped actions.

Does explicit prediction improve execution? To test whether accurate prediction translates into better execution, we compare against Direct, which scores prefixes from current policy representations and actions without predicting future representations. Rollout data, outcome supervision, score accumulation, and horizon selection are held constant. Explicit prediction raises success from 84.55% to 94.65% while reducing mean queries from 22.3 to 14.0 (Table 3), supporting its role in execution decisions.

Table 6: Real-world success rates (%), average replanning counts, and average execution horizons. Avg. success is the mean across the three tasks. Bold and underlining indicate the highest and second-highest success rates, including ties.
<table><tr><td>Method</td><td>Place Cube in Tray ↑</td><td>Stack Cubes ↑</td><td>Press Button ↑</td><td> $\operatorname { A v g } .$  success ↑</td><td> $\operatorname { A v g } .$  replans ↓</td><td>Avg. horizon</td></tr><tr><td> $h = 5$ </td><td>16.7</td><td>10.0</td><td>23.3</td><td>16.7</td><td>69.58</td><td>5.00</td></tr><tr><td> $h = 1 0$ </td><td>66.7</td><td>16.7</td><td>40.0</td><td>41.1</td><td>36.54</td><td>10.00</td></tr><tr><td> $h = 3 0$ </td><td>23.3</td><td>43.3</td><td>53.3</td><td>40.0</td><td>11.97</td><td>30.00</td></tr><tr><td> $h = 5 0$ </td><td>26.7</td><td>33.3</td><td>50.0</td><td>36.7</td><td>6.91</td><td>50.00</td></tr><tr><td>PAV (Ours)</td><td>80.0</td><td>66.7</td><td>76.7</td><td>74.4</td><td>12.87</td><td>30.22</td></tr></table>

Does the prediction space matter? To test whether policy alignment contributes beyond future prediction, we replace policy representations with DINOv2 features while retaining the prediction– verification pipeline, action-generating policy, and evaluation conditions. DINOv2 achieves 87.00% success with 22.2 queries per episode (Table 3). The higher success and lower query cost in policy space support retaining task-conditioned information for proposal evaluation.

Computational overhead. These gains must also be considered alongside the computation added by predictive verification. Across the four policy and chunk-length settings in Table 5, verifier GPU computation takes 0.90–2.49 ms, or 0.45–1.24% of action-generation time. These measurements quantify GPU computation rather than end-to-end control-loop latency.

## 4.3 REAL-WORLD EXPERIMENTS

## 4.3.1 EXPERIMENTAL SETUP

Environment. We use a 7-DoF Franka Panda with front, wrist, and right-side RGB observations to provide scene-level and near-gripper views. For each task, we fine-tune $\pi _ { 0 . 5 }$ from its base checkpoint on task-specific teleoperated demonstrations, then collect robot rollouts to train the verifier under the common protocol. Appendix E provides further hardware and training details.

Tasks. The three tasks vary the precision required after transport or approach, allowing us to examine execution under different interaction demands (Figure 3(a)). Place Cube in Tray tests transport and placement of an orange cube, while Stack Cubes requires placing it on a green cube and makes terminal alignment more restrictive. Press Button tests targeted end-effector contact with a red emergency button. To evaluate beyond a single starting arrangement, object positions are randomized within a 50 cm × 50 cm workspace while remaining visible from the wrist camera.

## 4.3.2 REAL-WORLD RESULTS

Our method achieves the highest success on all three real-world tasks, averaging 74.4% compared with 41.1% for the strongest fixed-horizon baseline, h = 10 (Table 6). It also reduces mean replanning counts from 36.54 to 12.87 per episode. Compared with h = 30, which has a similar replanning count of 11.97, adaptive execution improves average success from 40.0% to 74.4%, showing the benefit of varying execution lengths rather than using a constant horizon.

## 5 CONCLUSION

Effective action-chunk execution depends not only on generating coherent actions, but on deciding how long to commit to them. Our approach uses future prediction to inform this commitment, evaluating proposed execution in the same task-conditioned representation space that guides action generation. Across simulated and real-world manipulation, the results support learning execution decisions from task outcomes rather than relying on consistency alone. More broadly, predictive verification offers a way to improve the use of an existing VLA policy by anticipating the consequences of its proposals before acting on them.

## AI USE STATEMENT

In this work, we used generative AI tools to assist with the implementation of methods after the methodology had been independently developed by the authors. We also used generative AI tools to draft parts of the manuscript, improve its readability and presentation, and check grammar. We did not use generative AI tools to develop the research methodology, formulate hypotheses, or interpret experimental results. All AI-assisted text and code were reviewed, revised, and verified by the authors. We take responsibility for the final content of this work, including all text, claims, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

To support reproducibility, we will submit our implementation as supplementary material. The Method section details the proposed approach and training objectives, while the appendix provides the architecture, experimental configurations, and implementation details needed to reproduce our experiments.

## REFERENCES

Jose A Arjona-Medina, Michael Gillhofer, Michael Widrich, Thomas Unterthiner, Johannes Brandstetter, and Sepp Hochreiter. Rudder: Return decomposition for delayed rewards. Advances in Neural Information Processing Systems, 32, 2019.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

Alex J Chan, Hao Sun, Samuel Holt, and Mihaela Van Der Schaar. Dense reward for free in reinforcement learning from human feedback. arXiv preprint arXiv:2402.00782, 2024.

Feng Chen, Xianghui Wang, Yuxuan Chen, Boying Li, Yefei He, Zeyu Zhang, and Yicheng Wu. Dynamic execution commitment of vision-language-action models. arXiv preprint arXiv:2605.11567, 2026.

Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li, Qiwei Liang, Xianliang Lin, Yiheng Ge, Zhenyu Gu, et al. Robotwin 2.0: A scalable data generator and benchmark with strong domain randomization for robust bimanual robotic manipulation. arXiv preprint arXiv:2506.18088, 2025.

Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. Advances in neural information processing systems, 31, 2018.

Thomas G Dietterich, Richard H Lathrop, and Tomas Lozano-P´ erez. Solving the multiple instance´ problem with axis-parallel rectangles. Artificial intelligence, 89(1-2):31–71, 1997.

Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning universal policies via text-guided video generation. Advances in neural information processing systems, 36:9156–9172, 2023.

Xiangdong Feng, Yuxuan Cheng, Chen Shi, Boyao Han, Yuxuan Yan, Yitong Hong, Zhuotao Tian, and Li Jiang. Denoising tells when to replan: Denoising-variance adaptive chunking for flowbased robot policies. arXiv preprint arXiv:2606.03847, 2026.

Carles Gelada, Saurabh Kumar, Jacob Buckman, Ofir Nachum, and Marc G Bellemare. Deepmdp: Learning continuous latent space models for representation learning. In International conference on machine learning, pp. 2170–2179. PMLR, 2019.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019a.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In International conference on machine learning, pp. 2555–2565. PMLR, 2019b.

Nick Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, volume 2024, pp. 47376– 47405, 2024.

Nicklas Hansen, Xiaolong Wang, and Hao Su. Temporal difference learning for model predictive control. arXiv preprint arXiv:2203.04955, 2022.

Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based deep multiple instance learning. In International conference on machine learning, pp. 2127–2136. Pmlr, 2018.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. π<sub>0.5</sub>: a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054, 2025.

Michael Janner, Yilun Du, Joshua B Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis. arXiv preprint arXiv:2205.09991, 2022.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source visionlanguage-action model. arXiv preprint arXiv:2406.09246, 2024.

Jiahui Li, Lin Li, Tai-Wei Chang, Kun Kuang, Long Chen, Jun Zhou, and Cheng Yang. Red: Unleashing token-level rewards from holistic feedback via reward redistribution. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 4993–5022, 2025.

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

Yushan Liu, Peibo Sun, Xintao Chao, Zhenyang Yang, Yifan Xie, Lingfeng Zhang, Shoujie Li, Chenyu Tang, Fang Chen, Xiao-Ping Zhang, et al. Checkvla: Execution-time verification with action-conditioned world model for long-horizon mobile manipulation. arXiv preprint arXiv:2607.26789, 2026.

Yecheng Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar, and Amy Zhang. Vip: Towards universal visual reward and representation via value-implicit pre-training. arXiv preprint arXiv:2210.00030, 2022.

Shangchen Miao, Ningya Feng, Jialong Wu, Ye Lin, Xu He, Dong Li, and Mingsheng Long. Jepavla: Video predictive embedding is needed for vla models. arXiv preprint arXiv:2602.11832, 2026.

Junhyuk Oh, Satinder Singh, and Honglak Lee. Value prediction network. Advances in neural information processing systems, 30, 2017.

Yi Pan, Miao Pan, Qi Lu, Jiaming Huang, Man Zhang, Siteng Huang, Xin Li, Jie Zhang, Yongliang Shen, Xuhong Zhang, et al. Vla-corrector: Lightweight detect-and-correct inference for adaptive action horizon. arXiv preprint arXiv:2607.01804, 2026.

Vihang P Patil, Markus Hofmarcher, Marius-Constantin Dinu, Matthias Dorfer, Patrick M Blies, Johannes Brandstetter, Jose A Arjona-Medina, and Sepp Hochreiter. Align-rudder: Learning from few demonstrations by reward redistribution. arXiv preprint arXiv:2009.14108, 2020.

Zhizhou Ren, Ruihan Guo, Yuan Zhou, and Jian Peng. Learning long-term reward redistribution via randomized return decomposition. arXiv preprint arXiv:2111.13485, 2021.

Max Schwarzer, Ankesh Anand, Rishab Goel, R Devon Hjelm, Aaron Courville, and Philip Bachman. Data-efficient reinforcement learning with self-predictive representations. arXiv preprint arXiv:2007.05929, 2020.

Younggyo Seo, Kimin Lee, Stephen L James, and Pieter Abbeel. Reinforcement learning with action-free pre-training from videos. In International Conference on Machine Learning, pp. 19561–19579. PMLR, 2022.

Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. Vla-jepa: Enhancing vision-language-action model with latent world model. In European Conference on Computer Vision, pp. 478–497. Springer, 2026.

Haoxuan Wang, Gengyu Zhang, Yan Yan, Ramana Rao Kompella, and Gaowen Liu. Vla knows its limits: Adaptive execution horizons for robot policies. In European Conference on Computer Vision, pp. 607–623. Springer, 2026.

Seonghyeon Ye, Yunhao Ge, Kaiyuan Zheng, Shenyuan Gao, Sihyun Yu, George Kurian, Suneel Indupuru, You Liang Tan, Chuning Zhu, Jiannan Xiang, et al. World action models are zero-shot policies. arXiv preprint arXiv:2602.15922, 2026.

Lifan Yuan, Wendi Li, Huayu Chen, Ganqu Cui, Ning Ding, Kaiyan Zhang, Bowen Zhou, Zhiyuan Liu, and Hao Peng. Free process rewards without process labels. arXiv preprint arXiv:2412.01981, 2024.

Amy Zhang, Rowan McAllister, Roberto Calandra, Yarin Gal, and Sergey Levine. Learning invariant representations for reinforcement learning without reconstruction. arXiv preprint arXiv:2006.10742, 2020.

Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983, 2024.

## A IMPLEMENTATION AND CONFIGURATION

We detail the $\pi _ { 0 . 5 }$ LIBERO configuration with a 50-action chunk, no separate text or task embedding, and acceptance threshold $\beta = 0 . 9$ . These settings describe the implementation used for the corresponding analysis setting, rather than assuming an identical architecture or configuration for every policy in the main experiments.

## A.1 ARCHITECTURE OF THE PREDICTIVE VERIFIER

Policy representation. To retain the context used to generate a proposal, we extract the final prefix hidden states from the policy’s PaliGemma language model, before action denoising. Masked mean pooling, masked max pooling, and the last non-padding token each provide a 2,048-dimensional vector. Concatenating these vectors with the policy-preprocessed 32-dimensional state yields a 6,176- dimensional context. The VLA parameters remain unchanged, and the instruction enters through its language tokens rather than through a separate verifier input. In particular, disabling the task embedding removes the additional learned task-ID lookup, not the policy’s instruction conditioning.

Predicting this context directly would require a substantially wider dynamics model. We therefore standardize it using training-set statistics and learn a projector from 6,176 to 128 dimensions, followed by $\ell _ { 2 }$ normalization. The predicted and target latents in the method are instantiated in this projected policy-feature space. Current and observed future contexts share the projector, while the future target is detached in the prediction losses. Thus, the policy feature extractor is unchanged, but its downstream projection is learned jointly with prediction and verification.

Action-conditioned prediction. To evaluate all ten candidate horizons in one recurrent rollout, the predictor divides the proposed 50 × 7 action sequence into ten blocks of five actions. Each block is standardized using training-set action statistics, flattened to 35 dimensions, and embedded into 128 dimensions. Starting from the projected current context and a zero-initialized recurrent hidden state, a GRU cell combines the current latent with the next action embedding. A residual transition head maps its 192-dimensional hidden state to a 128-dimensional update, which is added to the current latent and normalized to obtain the next prediction. This prediction becomes the input latent for the following block. Teacher forcing is disabled during both training and deployment.

Local verification. The local verifier reads only the predicted 128-dimensional future latent and applies a sigmoid to its scalar output to obtain $q .$ It receives neither a separate action embedding nor the GRU hidden state, so action information reaches this score through the predicted consequence. Table 7 specifies the modules. An additional completion head reads the projected observed termina context during training and is not used to select horizons.

Table 7: Predictive-verifier architecture for $\pi _ { 0 . 5 }$ on LIBERO with $H = 5 0$ . Widths denote linearlayer dimensions. LN denotes LayerNorm. Dropout is zero throughout.
<table><tr><td>Module</td><td>Layers</td><td>Parameters</td></tr><tr><td>Context projector</td><td>LN, 6176 → 256, GELU, 256 → 128, l2 normalization</td><td>1,626,560</td></tr><tr><td>Action encoder</td><td>LN, 35 → 128, GELU, 128 → 128, GELU</td><td>21,190</td></tr><tr><td>Recurrent transition</td><td>GRUCell, input 256, hidden 192</td><td>259,200</td></tr><tr><td>Residual update</td><td>LN, 192 → 192, GELU, 192→ 128</td><td>62,144</td></tr><tr><td>Local verifier</td><td>LN, 128 → 192, GELU, 192 → 1, sigmoid</td><td>25,217</td></tr><tr><td>Completion head</td><td>LN, 128 → 128, GELU, 128 → 1</td><td>16,897</td></tr><tr><td>Total</td><td>Including the training-only completion head</td><td>2,011,208</td></tr></table>

## A.2 TRAINING DATA AND OPTIMIZATION

Rollout collection and splitting. To expose the verifier to different execution commitments, we collect 150 rollouts for each of the 40 tasks across the four LIBERO suites, sampling uniformly from $\{ 5 , 1 0 , \ldots , 5 0 \}$ at every policy query. The policy checkpoint is the H = 50 LIBERO finetuned $\pi _ { 0 . 5 }$ checkpoint at step 30,000. Observations after each completed five-action block provide future targets, while the terminal task outcome labels the rollout. Unexecuted blocks and incomplete final blocks are masked out rather than assigned local labels. Future-context extraction uses the same policy prefix encoder as the current context.

To prevent repeated initializations from crossing data partitions, we split initial-state IDs separately within each task and keep all rollouts from an assigned initialization together. With seed 7, this yields 4,200 training, 840 validation, and 960 held-out test rollouts from the 6,000 collected episodes. Normalization statistics are computed only from the training partition. Context statistics include its current, observed future, and terminal contexts, while action statistics are computed per action dimension. Standard deviations below $1 0 ^ { - 5 }$ are replaced by one. Training samples episodes with replacement using inverse class-frequency weights, so success and failure have equal sampling mass.

Prediction and outcome supervision. Future alignment compares each predicted latent with the detached projection of the corresponding observed future. To require sensitivity to the proposed actions, action-swap ranking cyclically permutes action chunks among decision records from the same task within a minibatch. Records without another same-task record are excluded from this comparison. Both predicted representations are compared against the original record’s future target, with margin 0.1. The substituted chunk receives no task-failure label.

The trajectory objective treats successful and failed rollouts differently to retain useful local behavior within failures. In this implementation, a successful rollout uses the mean blockwise binary cross-entropy with a smoothed positive target of 0.95, while a failed rollout uses the negative logprobability of at least one incompatible extension. The minibatch objective averages the success and failure class means when both classes are present. Localization penalizes 1 − q over all observed blocks from failed rollouts in the minibatch. This implementation uses a block-weighted average, rather than first averaging within each failed rollout and then weighting rollouts equally.

Auxiliary terminal supervision. Because future alignment uses a learned projection, an auxiliary completion objective also trains that representation to retain outcome-relevant information. The completion head takes the projected observed terminal context and predicts the rollout’s binary success label using class-balanced binary cross-entropy. Its loss is weighted by 0.25 and added to the prediction and verification objectives during joint optimization. It does not predict continuation success from candidate prefixes, and neither its output nor an additional completion factor enters the execution score. The future-contrastive and factorized-completion objectives available in the code are disabled.

Table 8: Training settings for the no-task-embedding $\pi _ { 0 . 5 }$ LIBERO $H = 5 0$ verifier.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Collection and training seed</td><td>7</td></tr><tr><td>Rollouts per task / total</td><td>150 / 6,000</td></tr><tr><td>Train / validation / test rollouts</td><td>4,200 / 840 / 960</td></tr><tr><td>Batch size</td><td>8 episodes, including their decision records</td></tr><tr><td>Optimizer</td><td>AdamW, constant learning rate  $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Training epochs / selected epoch</td><td>100 / 7</td></tr><tr><td>Checkpoint selection</td><td>Lowest validation trajectory loss</td></tr><tr><td>Weight decay / gradient norm limit</td><td>10−4 /5</td></tr><tr><td>Alignment / action-swap weights</td><td>1/1</td></tr><tr><td>Action-swap margin</td><td>0.1</td></tr><tr><td>Trajectory / localization weights Auxiliary completion weight</td><td>1 / 0.01</td></tr><tr><td>Successful-block training target</td><td>0.25</td></tr><tr><td>Dropout / teacher-forcing probability</td><td>0.95</td></tr><tr><td></td><td>0/0</td></tr><tr><td>Separate text or task embedding</td><td>None</td></tr></table>

## A.3 CANDIDATE HORIZONS AND EXECUTION

To span each chunk with a regular set of execution choices, we use {5, 10} for $H ~ = ~ 1 0 ,$ {4, 8, 12, 16, 20, 24, 28, 32} for $\bar { H } = 3 2$ , and {5, 10, 15, 20, 25, 30, 35, 40, 45, 50} for $H = 5 0 .$

The five-action prediction stride gives candidate horizons $h _ { k } ~ = ~ 5 k$ for $k \in \{ 1 , \ldots , 1 0 \}$ . Here $k$ counts predicted extensions, whereas $h _ { k }$ counts robot control steps. To prevent the accumulated score from favoring short prefixes solely through repeated multiplication, deployment applies square-root length normalization to the number of blocks. Writing $q _ { t , j }$ for the local score of block $j ,$ the score checked at $h _ { k }$ is

$$
\widetilde { S } _ { t } ( h _ { k } ) = \exp \left( \frac { 1 } { \sqrt { k } } \sum _ { j = 1 } ^ { k } \log q _ { t , j } \right) .\tag{14}
$$

The implementation clips probabilities to $[ 1 0 ^ { - 1 2 } , 1 ]$ before taking logarithms. It scans candidates in ascending order and stops extending the execution horizon at the first score below 0.9. This intermediate-prefix check prevents acceptance beyond an earlier violation, since the normalized score need not be monotone.

The checkpoint’s post-hoc bag calibration is not applied to these local scores, and the alternative score obtained by dividing each q by the training target 0.95 is disabled. Table 9 summarizes the resulting deployment configuration.

Table 9: Deployment settings for the H = 50 LIBERO configuration.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Generated action chunk</td><td> $5 0 \times 7$ </td></tr><tr><td>Prediction stride</td><td>5 control steps</td></tr><tr><td>Candidate horizons</td><td>5, 10, 15, 20, 25, 30, 35, 40, 45, 50</td></tr><tr><td>Acceptance threshold</td><td> $\beta = 0 . 9$ </td></tr><tr><td>Length normalization</td><td>Square root of the block count</td></tr><tr><td>Candidate check</td><td>Stop at the first threshold violation</td></tr><tr><td>Bag calibration / target normalization</td><td>Disabled / disabled</td></tr></table>

## B DETAILED ROBOTWIN RESULTS

To show how the aggregate results vary across tasks, Table 10 expands the Selected 16 comparison in Table 2 to the complete task list reported in DVAC Table 9 (Feng et al., 2026). Our fixed-horizon baselines and verifier use $\pi _ { 0 . 5 }$ with $H = 3 2$ , with 50 evaluation episodes per task. The RoboTwin verifier uses $\beta = 0 . 9 1$ and includes a task embedding, unlike the LIBERO verifier configuration detailed above. The $h \ = \ 3 2$ baseline comes from a separate evaluation, so episode-level initial conditions are not guaranteed to be paired with the other methods. DVAC is included as an external reference under its published $H = 5 0$ policy and evaluation protocol.

## C ADDITIONAL ABLATIONS AND ANALYSIS

The detailed analysis is organized around $\pi _ { 0 . 5 }$ with H = 50 on LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, and LIBERO-Long. With our candidate sets, $H = 1 0$ allows only $h \in \{ 5 , 1 0 \}$ whereas $H = 5 0$ offers ten choices from 5 to 50 steps. We use this broader range to examine how the verifier changes execution length and how each component affects that choice. Only the overhead measurements span multiple policy settings. Comparisons use the same policy checkpoint, rollout dataset, task set, and evaluation protocol. These controlled ablations characterize the components in the H = 50 LIBERO setting, complementing the broader evaluation of the complete method.

## C.1 FUTURE PREDICTION AND REPRESENTATION ABLATIONS

Table 3 in the main paper reports the complete suite-level comparison of Direct, DINOv2, and PAV. Success and VLA-query counts are shown together to distinguish performance improvements from changes in replanning frequency. Individual loss ablations below further isolate the contribution of the training objectives.

Table 10: Per-task success rates (%) on all 16 RoboTwin tasks in the Selected 16 subset. $\mathrm { D V A C } ^ { \dagger }$ reproduces the $\alpha = 2$ results from its Table 9 (Feng et al., 2026), including the reported average rounded to one decimal place. Other averages are computed over 800 episodes. Bold indicates the highest reported value in each row, including ties.
<table><tr><td>Task</td><td> $h = 4$ </td><td> $h = 8$ </td><td> $h = 2 0$ </td><td> $h = 3 2$ </td><td> $\mathrm { D V A C } ^ { \dagger }$ </td><td>PAV (Ours)</td></tr><tr><td>Blocks Ranking RGB</td><td>0</td><td>40</td><td>62</td><td>38</td><td>45</td><td>64</td></tr><tr><td>Blocks Ranking Size</td><td>2</td><td>34</td><td>26</td><td>14</td><td>37</td><td>50</td></tr><tr><td>Hanging Mug</td><td>0</td><td>6</td><td>16</td><td>6</td><td>1</td><td>8</td></tr><tr><td>Move Stapler Pad</td><td>14</td><td>30</td><td>22</td><td>14</td><td>11</td><td>42</td></tr><tr><td>Open Laptop</td><td>44</td><td>62</td><td>84</td><td>60</td><td>90</td><td>70</td></tr><tr><td>Open Microwave</td><td>4</td><td>26</td><td>46</td><td>58</td><td>67</td><td>42</td></tr><tr><td>Pick Diverse Bottles</td><td>40</td><td>62</td><td>38</td><td>36</td><td>55</td><td>44</td></tr><tr><td>Place A2B Left</td><td>34</td><td>62</td><td>64</td><td>64</td><td>42</td><td>74</td></tr><tr><td>Place A2B Right</td><td>22</td><td>74</td><td>70</td><td>56</td><td>30</td><td>84</td></tr><tr><td>Place Burger Fries</td><td>44</td><td>76</td><td>56</td><td>58</td><td>58</td><td>76</td></tr><tr><td>Place Dual Shoes</td><td>16</td><td>48</td><td>48</td><td>24</td><td>27</td><td>56</td></tr><tr><td>Place Empty Cup</td><td>26</td><td>50</td><td>54</td><td>58</td><td>68</td><td>56</td></tr><tr><td>Place Object Basket</td><td>26</td><td>60</td><td>68</td><td>28</td><td>51</td><td>58</td></tr><tr><td>Scan Object</td><td>4</td><td>40</td><td>34</td><td>12</td><td>20</td><td>42</td></tr><tr><td>Stack Blocks Three</td><td>0</td><td>40</td><td>22</td><td>10</td><td>13</td><td>48</td></tr><tr><td>Stack Bowls Three</td><td>2</td><td>38</td><td>56</td><td>30</td><td>50</td><td>52</td></tr><tr><td>Selected 16</td><td>17.38</td><td>46.75</td><td>47.88</td><td>35.38</td><td>41.6</td><td>54.13</td></tr></table>

## C.2 ACCEPTANCE-THRESHOLD SWEEP

The threshold changes how much evidence is required before execution and therefore affects both success and query cost. To separate this operating-point choice from representation learning, the sweep holds the trained checkpoint fixed and varies only β. We evaluate $\beta \in$ {0.88, 0.89, 0.90, 0.91, 0.92} around the default value of 0.90.

To compare adaptive and constant-horizon execution at similar query counts, we relate Table 11 to the $\pi _ { 0 . 5 } , H = 5 0$ baselines in Table 1. At $\beta ~ = ~ 0 . 9 1$ , our method achieves 89.45% average success with 19.7 VLA queries per episode, compared with 87.45% and 19.2 queries for $h = 1 0$ This 2.00-percentage-point gain supports an advantage at a similar query budget, complementing the default-setting comparison.

Table 11: Threshold sweep with a fixed trained verifier on $\pi _ { 0 . 5 }$ with $H = 5 0$ on LIBERO. Success rates are reported per suite and averaged equally across the four suites. Queries are the total histogram selection count divided by 2,000 evaluation episodes; mean horizon weights each selection equally. The default row uses the PAV results from the main paper.
<table><tr><td rowspan="2">β</td><td colspan="5">Success (%) ↑</td><td colspan="2"></td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td> $\operatorname { A v g } .$ </td><td>Queries ↓</td><td>Mean h</td></tr><tr><td>0.88</td><td>79.4</td><td>87.2</td><td>84.6</td><td>79.8</td><td>82.75</td><td>13.8</td><td>15.200</td></tr><tr><td>0.89</td><td>81.8</td><td>90.0</td><td>85.4</td><td>85.4</td><td>85.65</td><td>15.1</td><td>13.443</td></tr><tr><td>Default (0.90)</td><td>96.8</td><td>98.8</td><td>91.8</td><td>91.2</td><td>94.65</td><td>14.0</td><td>12.783</td></tr><tr><td>0.91</td><td>89.4</td><td>94.6</td><td>87.6</td><td>86.2</td><td>89.45</td><td>19.7</td><td>9.685</td></tr><tr><td>0.92</td><td>90.4</td><td>94.8</td><td>86.8</td><td>81.2</td><td>88.30</td><td>25.1</td><td>7.614</td></tr></table>

To explain how the threshold changes query demand, Figure 4 compares the selected-horizon distributions across the five sweep settings. We normalize the pooled counts within each setting so that differences in the total number of decisions do not obscure the shift in execution length. At the default $\beta = 0 . 9 0 , h = 1 5$ is the most frequent choice, accounting for 33.0% of selections. The two highest thresholds concentrate execution more strongly on short prefixes, with $h \leq 1 0$ accounting

for 72.6% of selections at 0.91 and 94.7% at 0.92. The default threshold achieves the highest success among the tested settings (Table 11), while higher thresholds shift selections toward shorter prefixes.  
![](images/815785adaa2904936237eefeae11bdab2ab164eba4fbe427b2fba26945502d98.jpg)  
Figure 4: Selected-horizon distributions for $\pi _ { 0 . 5 }$ with $H = 5 0$ on LIBERO. Each panel shows one threshold, with counts pooled across the four suites and normalized within that setting. Panels share the same axes, with one bar per candidate horizon. The default setting is highlighted in dark blue.

## C.3 REMOVING INDIVIDUAL LOSS TERMS

To determine which training constraints contribute to useful execution decisions, each variant removes one term while retaining the architecture, data, and remaining coefficients. Removing future alignment sets $\lambda _ { \mathrm { d y n } } ~ = ~ 0$ to isolate the contribution of observed future targets. Removing swap ranking sets $\lambda _ { \mathrm { s w a p } } = 0$ and tests whether alignment encourages sufficient action sensitivity. Removing localization sets $\lambda _ { \mathrm { l o c } } = 0$ to evaluate how the localization prior contributes to execution performance.

The trajectory objective is retained to provide a common source of outcome supervision. Each trained variant is evaluated with the default threshold, so the comparison measures the contribu tion of each term under the same execution rule.

Table 12: Loss-term ablations on $\pi _ { 0 . 5 }$ with $H = 5 0$ on LIBERO. Suite columns and the average report success (%). Queries are the mean number of VLA generations per episode across the four suites.
<table><tr><td>Training variant</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Avg.</td><td>Queries</td></tr><tr><td>Full objective</td><td>96.8</td><td>98.8</td><td>91.8</td><td>91.2</td><td>94.65</td><td>14.0</td></tr><tr><td>Without future alignment</td><td>83.2</td><td>91.0</td><td>86.8</td><td>80.2</td><td>85.30</td><td>23.99</td></tr><tr><td>Without swap ranking</td><td>84.6</td><td>93.4</td><td>88.2</td><td>76.6</td><td>85.70</td><td>24.26</td></tr><tr><td>Without localization</td><td>81.8</td><td>92.2</td><td>86.8</td><td>73.8</td><td>83.65</td><td>25.60</td></tr></table>

The full objective improves success by 8.95–11.00 percentage points over the three ablated variants while reducing mean queries from 23.99–25.60 to 14.0 (Table 12). These results support retaining future alignment, action-swap ranking, and localization in the evaluated configuration.

## C.4 SENSITIVITY TO ACTION SWAPS

Table 4 evaluates the trained predictor with executed and same-task swapped actions, alongside reversed and zero-action inputs. We further examine how the effect of replacing the actions varies across prediction horizons. This input-sensitivity diagnostic complements the separately trained loss ablations in Table 12.

A predictor may align with future observations by exploiting scene regularities while responding weakly to the proposed actions. To test this possibility, held-out transitions are evaluated with both the executed prefix and a different prefix of the same length. The current representation and observed future target are held fixed. Each donor comes from a different held-out episode of the same task with the same number of observed blocks. We retain only recorded future targets with an eligible donor, yielding 36,843 observed-block pairs across 960 test episodes. Each pair receives equal weight.

Using the cosine distance from Equation 9, define the prediction-distance gap

$$
\Delta _ { t } ( h ) = d _ { \cos } ( \tilde { z } _ { t + h | t } , z _ { t + h } ) - d _ { \cos } ( \hat { z } _ { t + h | t } , z _ { t + h } ) .\tag{15}
$$

A positive gap means that the executed prefix predicts its observed consequence more accurately than the alternative. We report the gap by horizon together with the fractions satisfying $\Delta _ { t } ( h ) > 0$ and $\Delta _ { t } ( h ) \geq m$ . Both inputs are evaluated against the same recorded future, so the comparison measures action sensitivity relative to that target. Across horizons, the executed prefix gives a closer prediction in 89.3–95.2% of pairs, supporting action sensitivity over the tested prediction range.

Table 13: Horizon-wise action-swap sensitivity of the full predictor on held-out transitions. n is the number of observed-block pairs, and the ranking margin is $m = 0 . 1$
<table><tr><td>Horizon</td><td>n</td><td>Mean  $\Delta _ { t } ( h )$ </td><td> $\Delta _ { t } ( h ) > 0 ( \% )$ </td><td> $\Delta _ { t } ( h ) \geq m \left( \% \right)$ </td></tr><tr><td>5</td><td>7,276</td><td>0.1628</td><td>89.3</td><td>53.5</td></tr><tr><td>10</td><td>6,367</td><td>0.3292</td><td>92.5</td><td>70.7</td></tr><tr><td>15</td><td>5,537</td><td>0.4257</td><td>93.6</td><td>76.2</td></tr><tr><td>20</td><td>4,683</td><td>0.4898</td><td>94.6</td><td>79.6</td></tr><tr><td>25</td><td>3,883</td><td>0.5287</td><td>94.7</td><td>81.0</td></tr><tr><td>30</td><td>3,160</td><td>0.5537</td><td>95.2</td><td>83.0</td></tr><tr><td>35</td><td>2,470</td><td>0.5619</td><td>94.7</td><td>82.7</td></tr><tr><td>40</td><td>1,781</td><td>0.5697</td><td>94.6</td><td>81.9</td></tr><tr><td>45</td><td>1,140</td><td>0.5636</td><td>94.0</td><td>81.8</td></tr><tr><td>50</td><td>546</td><td>0.5792</td><td>95.2</td><td>84.6</td></tr></table>

## C.5 EFFECT OF SQUARE-ROOT LENGTH NORMALIZATION

Following Appendix A.3, let k count five-action blocks and $h _ { k } = 5 k$ denote the execution length. The accumulated score is $\begin{array} { r } { \hat { S } _ { t } ( h _ { k } ) = \prod _ { j = 1 } ^ { k } q _ { t , j } } \end{array}$ , which becomes $q ^ { k }$ when local validity is constant. To permit longer prefixes with consistently high local scores while retaining a length penalty, we moderate this decay to $q ^ { \sqrt { k } }$ using

$$
\begin{array} { r } { \bar { S } _ { t } ( h _ { k } ) = \hat { S } _ { t } ( h _ { k } ) ^ { 1 / \sqrt { k } } . } \end{array}\tag{16}
$$

This normalized score is used for horizon selection rather than as a calibrated probability. Because normalization can raise later scores after an earlier decrease, the execution rule checks candidate prefixes in ascending order and stops at the first threshold violation. A later increase therefore cannot override an earlier violation.

This ablation tests whether square-root normalization enables longer execution while preserving or improving task success on $\pi _ { 0 . 5 }$ with $H = 5 0$ across the four LIBERO suites. Because it acts on the accumulated score at inference rather than as a training regularizer, we hold the trained predictor and verifier fixed and change only the normalization rule. We compare raw accumulation, square-root normalization, and geometric-mean normalization, written as

$$
S _ { t } ^ { ( \eta ) } ( h _ { k } ) = \hat { S } _ { t } ( h _ { k } ) ^ { 1 / k ^ { \eta } } , \qquad \eta \in \{ 0 , { \frac { 1 } { 2 } } , 1 \} .\tag{17}
$$

Here $\eta = 0$ gives the unnormalized product, $\begin{array} { r } { \eta = \frac { 1 } { 2 } } \end{array}$ gives our rule, and $\eta = 1$ gives the geometric mean. For constant local validity $q ,$ these scores reduce to $q ^ { k ^ { 1 - \eta } }$ . Raw accumulation therefore decays exponentially in k, while the geometric mean removes length dependence in this constant-score case. Square-root normalization retains a length penalty but moderates its growth.

All three variants retain the intermediate-prefix acceptance check and the same candidate horizons $h \in \{ 5 , 1 0 , \ldots , 5 0 \}$ . Table 14 compares suite-level success, overall query count, and mean horizon at the common default threshold. This comparison evaluates all three rules at the shared operating point.

To explain how normalization changes execution beyond its mean length, Figure 5 shows one selected-horizon histogram per rule, pooling decisions across the four LIBERO suites and nor malizing within each variant. Raw accumulation selects only $h = 5 \mathrm { o r } h = 1 0 .$ , whereas squareroot normalization shifts the mode to $h = 1 5$ and spreads selections across intermediate horizons. Geometric-mean normalization instead concentrates selections at both $h = 5$ and $h = 5 0$ . At the shared threshold, square-root normalization achieves the highest task success and favors intermedi ate execution lengths, supporting its use as the default normalization in this setting (Table 14).

Table 14: Inference-time normalization ablation with the trained predictor and verifier held fixed. Success rates are reported for each LIBERO suite and averaged equally across the four suites. Queries are the total histogram selection count divided by 2,000 evaluation episodes; mean hori zon weights each selection equally. Entries correspond to the common default threshold.
<table><tr><td rowspan="2">Normalization</td><td colspan="5">Success (%) ↑</td><td colspan="2"></td></tr><tr><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td> $\operatorname { A v g } .$ </td><td>Queries ↓</td><td>Mean h</td></tr><tr><td>Raw product  $( \eta = 0 )$ </td><td>92.0</td><td>90.8</td><td>91.6</td><td>80.0</td><td>88.6</td><td>30.9</td><td>6.104</td></tr><tr><td>Square root  $( \eta = \textstyle { \frac { 1 } { 2 } } )$ </td><td>96.8</td><td>98.8</td><td>91.8</td><td>91.2</td><td>94.65</td><td>14.0</td><td>12.783</td></tr><tr><td>Geometric mean  $( \eta = 1 )$ </td><td>53.6</td><td>53.0</td><td>71.8</td><td>50.4</td><td>57.2</td><td>11.3</td><td>22.645</td></tr></table>

![](images/a9ddf338f8455330401eb6b3089b94166aa916be3baed9c0b26a0c4737e7894b.jpg)  
Figure 5: Horizon selection under three normalization rules at $\beta = 0 . 9 0$ on $\pi _ { 0 . 5 } .$ , LIBERO $H = 5 0$ Each panel uses identical axes and independently normalized counts pooled across suites. Square root normalization favors intermediate horizons. Success and query costs appear in Table 14.

## D QUALITATIVE DEMONSTRATIONS

To relate horizon selection to physical interactions, Figure 6 pairs observed rollout frames with execution-horizon traces from LIBERO-Object and LIBERO-Goal. Both examples contain a missed grasp followed by a successful retry and placement. Shorter horizons around the retry provide more frequent opportunities to incorporate feedback, while longer horizons during approach and transfer sustain motion with fewer policy queries. The traces illustrate how execution length changes within a rollout rather than merely differing across tasks.

These examples illustrate shorter commitments during grasp recovery and longer commitments during transfer, complementing the aggregate success and query results. The images show observed rollout frames, and the traces report selected execution horizons.

![](images/e31d4e94d094a9e89d2926802dace37d755f21b6a83f547f22b22215b003ad76.jpg)

![](images/8c5b5f8851d0adaaddbb52feace0bf77c43519cb054f8f4c657ac3d6e7ea46dd.jpg)

(b) LIBERO-Goal · Recovery after a missed grasp  
![](images/ea2acb90cec5fe70fdacaf0d61339ca5a780a2a29beb5dcd9033cc6999a413b3.jpg)

![](images/d9cba914f1dce5293bac54e9c57b71355fe3a06ec8404f5043cacfd0b87c5460.jpg)  
Figure 6: Adaptive execution during grasp recovery in two LIBERO rollouts. (a) In LIBERO-Object, the robot retries an empty grasp before transferring the soup can to the basket. (b) In LIBERO-Goal, it reattempts the bowl grasp and places the bowl on the stove. Observed frames are sampled every 19 and 15 control steps, respectively. The traces show selected execution horizons, with dots marking policy queries. Short horizons accompany grasp retries, while longer horizons occur during approach and transfer. Shaded intervals identify the annotated interaction phases, not validity labels or training supervision.

## E REAL-WORLD SCENE AND DEMONSTRATIONS

## E.1 HARDWARE, VIEWS, AND TASK LAYOUT

The real-world setup uses a 7-DoF Franka Panda with front, wrist, and right-side RGB observations. Figure 3(a) shows the gripper and tabletop task layouts at successful task states. Object placements vary within the 50 cm × 50 cm region described in the main text, while remaining visible from the wrist camera at initialization. Policy and verifier inference run on an NVIDIA RTX 4090. Each RGB view is resized with aspect-ratio-preserving padding to $2 2 4 \times 2 2 4$ pixels. Actions are executed at a nominal frequency of 15 Hz. The policy interface outputs seven absolute joint-position targets in radians and one gripper command. During training, joint targets are represented as offsets from the current joint configuration and converted back to absolute targets for execution.

Task definitions. Place Cube in Tray places an orange cube in a tray, Stack Cubes places an orange cube on top of a green cube, and Press Button activates a red emergency button. A human evaluator judges task success, and each trial has a maximum budget of 500 action steps. Trials terminated early without achieving task success are counted as failures and retained in the success-rate denominator.

Training and evaluation records. We fully fine-tune a separate $\pi _ { 0 . 5 }$ base policy for each task using task-specific teleoperated demonstrations. The deployed policies use the checkpoints obtained after 10,000 optimizer updates, with $H = 5 0$ . Each task-specific verifier is trained on 50 rollouts for 60 epochs. Deployment uses candidate horizons $\{ 5 , 1 0 , \dot { ~ } . ~ . . , 5 0 \}$ , a common threshold $\beta = 0 . 9 0$ across all three tasks, and square-root block-count normalization. Each decision evaluates a single generated action chunk and executes its selected prefix without regenerating the proposal. Evaluation uses 30 trials per task, separate from the training data. To match initial conditions across methods, object placements are randomized when constructing the evaluation sequence and then held fixed by episode index, so each method encounters the same initial placement for the corresponding episode.

## E.2 REAL-WORLD EXECUTION EXAMPLES

Figure 7 compares adaptive execution with fixed horizons across three real-world tasks. Adaptive execution combines corrective feedback near interactions with sustained progress afterward. Short execution can interrupt coherent progress, while long execution can continue despite an unsuccessful interaction. Each row shows observations from one rollout; the examples illustrate these behaviors rather than a measured trace of individual policy-query events.

Metric aggregation. Replanning counts include the initial action-chunk generation and therefore count total VLA queries per episode. Mean execution horizon is computed over all selected prefixes pooled across evaluation episodes, giving each selection equal weight rather than averaging episodelevel means.

![](images/e639096e38f16932599304c6f4787351555038272affa0ae26a59b20a1474751.jpg)  
Figure 7: Real-world qualitative comparisons. (a) Press Button: PAV recovers and presses. $h = 5$ oscillates, while $h = 5 0$ retains an off-target alignment. (b) Stack Cubes: PAV realigns and stacks. $h = 5$ grasps but stalls before transfer, while $h = 5 0$ continues after a missed grasp. (c) Place Cube in Tray: PAV returns and retries successfully. $h = 5$ drifts away, while $h = 5 0$ continues empty before returning later without success. Red annotations highlight key behaviors. All rows start at t = 0. Video timestamps are nonuniform and not synchronized across rows.