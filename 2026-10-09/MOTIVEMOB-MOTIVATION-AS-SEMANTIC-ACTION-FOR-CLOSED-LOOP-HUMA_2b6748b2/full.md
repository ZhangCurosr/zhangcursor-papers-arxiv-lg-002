# MOTIVEMOB: MOTIVATION AS SEMANTIC ACTION FOR CLOSED-LOOP HUMAN MOBILITY GENERATION

Mengkun Gao<sup>1</sup>, Zengqing Wu<sup>1</sup>, Renhe Jiang<sup>2</sup>, Jiawei Wang<sup>3</sup>, Yusong Wang Chuang Yang<sup>2</sup>, Shuyuan Zheng<sup>1</sup>, Makoto Onizuka<sup>1</sup>, Chuan Xiao<sup>1†</sup>

<sup>1</sup>The University of Osaka, <sup>2</sup>The University of Tokyo,

<sup>3</sup>The Hong Kong University of Science and Technology, <sup>4</sup>Institute of Science Tokyo   
mengkun.gao@ist.osaka-u.ac.jp, wuzengqing@outlook.com   
jiangrh@csis.u-tokyo.ac.jp, wangyi@lr.pi.titech.ac.jp   
chuang.yang@csis.u-tokyo.ac.jp   
{zheng,onizuka,chuanx}@ist.osaka-u.ac.jp

## ABSTRACT

Human mobility generation, as an important task in urban system research, synthesizes trajectory data that can be used for urban planning, transportation management, etc. Human mobility can be characterized as a “why–where–when” decision process: people form an intention to move and then determine where and when the corresponding activity will take place. Trajectory generation under user-level and temporal distribution shifts may benefit from explicitly modeling this decision structure. However, many existing human mobility generation methods either represent behavioral intent at a coarse granularity, such as a daily plan or a trajectory-level description, or directly predict future locations without explicitly reasoning about a possible motivation for each movement step. We introduce MotiveMob, a motivation-driven autoregressive framework for human mobility generation that first forms a hypothesis about why the next movement may occur and then jointly generates where and when it may occur. At each step, a motivation predictor conditions on the current mobility state, a long-term behavioral report, and the mobility history to infer a plausible motivation or determine whether the trajectory should terminate. Given the hypothesized motivation, a state predictor grounds it in a candidate next location and arrival time. Before being fed back as the current mobility state for the next “why–where–when” decision, the candidate is subject to speed-feasibility and repetition checks. We evaluate MotiveMob under distribution shifts involving unseen users and unseen temporal periods, including seasonal changes and the substantial behavioral disruption caused by the COVID-19 pandemic. Experiments show that MotiveMob consistently achieves better distributional fidelity than competitive pretraining-based and prompting-based methods under user-level and temporal distribution shifts, demonstrating robust generalization to out-of-distribution mobility patterns.

## 1 INTRODUCTION

Human mobility trajectory generation aims to synthesize realistic sequences of visits, including the locations individuals visit and the corresponding timestamps (Luca et al., 2021). High-quality synthetic trajectories support a wide range of applications, including urban planning, transportation management, and epidemic modeling (Zhu et al., 2023a; Lopez et al., 2018; Chang et al., 2021). Recent advances in large language models (LLMs) have inspired new approaches to this task. Some methods fine-tune LLMs for next-point-of-interest (POI) prediction, while others prompt LLMs to generate daily mobility plans from semantic descriptions and historical observations (Li et al., 2024; Wang et al., 2025; 2024; 2026).

From a behavioral perspective, human mobility can be viewed as a motivation-first decision process: people form an intention to move and then determine both where to go and when to arrive. This factorization is particularly important for human mobility generation because similar mobility states may lead to different next movements depending on the latent motivation. However, many existing methods either predict future locations directly from observed trajectories or represent intent only at the coarse granularity of a daily plan, without explicitly modeling the step-level motivation associated with each transition. As a result, they may conflate behaviorally distinct transitions and struggle to maintain consistent activity and temporal patterns over extended generation horizons. This limitation becomes particularly consequential under user-level and temporal distribution shifts, as mobility patterns vary across individuals and evolve with seasonal changes and disruptions such as the COVID-19 pandemic. Generalization under these shifts may benefit from representations of users’ long-term behavioral patterns and recent activities, together with a transition model that maps hypothesized motivations to plausible next states. Although prior work suggests that LLMs can encode latent environment dynamics and support implicit world modeling (Abdou et al., 2021; Li et al., 2022; Zhang et al., 2025), mobility data record only realized transitions and provide no direct evidence of what would have occurred under alternative motivations, making such dynamics difficult to learn from observed trajectories alone.

Seeing this limitation in existing studies, we introduce MotiveMob, an LLM-based autoregressive framework that generates daily mobility trajectories through step-wise “why–where–when” decisions. At each step, a motivation predictor $P _ { M }$ conditions on the current mobility state, a long-term behavioral report R, and a recent mobility history $\mathcal { H }$ to predict the motivation for the next movement or determine whether the trajectory should terminate. A state predictor $P _ { S }$ then conditions on the same user context and treats the predicted motivation as a semantic action, grounding it in a feasible next location and arrival time. To train $P _ { S }$ beyond observed transitions, we construct pseudo alternative transitions by pairing observed states with alternative motivation labels, generating candidate next states with a lightweight predictor, and using an LLM-based judge to select the most plausible candidate. The observed and alternative transitions are then used to train $P _ { S }$ as a motivation-conditioned transition model, which serves as the dynamics component of an implicit world model (IWM). During inference, $P _ { M }$ and $P _ { S }$ operate alternately. Speed-infeasible candidates are regenerated, whereas repeated states terminate the rollout. Each accepted state then becomes the current mobility state for the next decision.

We evaluate MotiveMob under both user-level and temporal distribution shifts, covering seen and unseen users across an observed period, a regular unseen period, and a COVID-19 period characterized by substantial changes in mobility behavior. Across these settings, MotiveMob consistently achieves better distributional fidelity than competitive pretraining-based and prompting-based methods. The advantage is particularly evident in time–activity distribution fidelity: averaged across the three temporal periods, MotiveMob reduces the corresponding divergence by 10.7% for seen users and 18.3% for unseen users relative to the best-performing baseline in each setting. Compared with the variant without $P _ { M }$ , the corresponding reductions are 21.6% and 29.1%, respectively, supporting the benefit of reasoning about why a movement may occur before generating where and when it occurs. In addition, the evaluation analyzes the roles and metric-specific trade-offs of implicit world modeling, user-specific context in trajectory generation, and inference-time rollout validation.

Our contributions are summarized as follows:

• We formulate daily mobility trajectory generation as a step-level, motivation-first “why– where–when” process and introduce MotiveMob, which separates potential motivation prediction from the joint generation of the next location and arrival time.

• We develop an offline alternative transition augmentation procedure for training the state predictor as an IWM. A lightweight candidate generator proposes outcomes for hypothetical motivations, while an LLM-based judge filters plausible transitions, enabling the model to learn beyond observed state–motivation pairs without requiring online environment interaction.

• We conduct a systematic evaluation under user-level and temporal distribution shifts, covering unseen users, regular seasonal variation, and pandemic-induced behavioral disruption. MotiveMob consistently achieves better distributional fidelity than competitive pretrainingbased and prompting-based baselines.

## 2 RELATED WORK

## 2.1 HUMAN MOBILITY MODELING

Human mobility modeling encompasses next-location prediction and multi-step trajectory generation (Luca et al., 2021; Kong et al., 2023). Next-location prediction infers a user’s next destination from historical visits and contextual information. Early approaches used probabilistic transition models (Gambs et al., 2012; Lu et al., 2013), followed by recurrent and attention-based architectures that capture long-term dependencies and individual preferences (Feng et al., 2018; Luo et al., 2021). Recent LLM-based methods employ time-aware prompting or zero-shot candidate ranking (Wang et al., 2023; Feng et al., 2024), while others adapt LLMs through task-specific fine-tuning (Li et al., 2024; Wang et al., 2025); Mobility-LLM further learns intention- and preference-aware representations for predictive check-in tasks (Gong et al., 2024). These methods primarily target onestep or related predictive objectives. Trajectory generation instead synthesizes complete location–time sequences and aims to reproduce population-level spatiotemporal regularities. Existing generators employ variational autoencoders (Huang et al., 2019), adversarial imitation learning (Zhang et al., 2020; Choi et al., 2021), and diffusion models (Zhu et al., 2023b; Chu et al., 2024; Long et al., 2026). More recent LLM-based methods formulate trajectories as conditional sequences or agentgenerated plans (Ohara et al., 2024; Zhang et al., 2024; Li et al., 2025; Wang et al., 2026; Li et al., 2026), as summarized in a recent survey (Gao & Wu, 2026). Particularly relevant activity-centric approaches generate activity–time chains before assigning spatial locations (Liao et al., 2026), while retrieval-augmented methods synthesize household-coordinated activity chains from aggregate statistics and sociodemographic profiles (Liu et al., 2025). LLMob also infers a daily-level motivation to guide whole-day trajectory generation (Wang et al., 2024). However, existing activity-centric generators typically introduce semantic conditioning at the level of whole-day motivations or activity chains, rather than explicitly modeling the motivation associated with each individual transition. Moreover, how such models generalize to unseen users and future or disrupted periods also remains underexplored.

## 2.2 WORLD MODELING WITH LLMS

In reinforcement learning (RL), it is often desirable for an agent to have knowledge of the environment dynamics, enabling it to model state transitions and predict future states and rewards, as in modelbased RL (Luo et al., 2024). Traditionally, such knowledge is acquired by learning a world model from observed state transitions, ranging from simple lookup-based models (Sutton, 1991) to deep generative models (Ha & Schmidhuber, 2018). With the emergence of LLMs, agents have begun to employ LLMs as world models to support planning and decision making (Gu et al., 2024; Feng et al., 2025). However, most prior approaches treat the world model as an explicit and separate simulator, echoing classical control pipelines that rely on planning over imagined rollouts. In contrast, Zhang et al. (2025) propose an early experience paradigm in which the policy is trained to predict its own future states from an IWM, thereby internalizing environment dynamics without constructing a standalone simulator. Learning action-conditioned dynamics, however, requires transition data that cover the consequences of different actions under comparable states. Human mobility records contain only real transitions and do not reveal what might have occurred under alternative motivations. This lack of counterfactual observations, together with the absence of an interactive mobility environment, presents a key challenge for learning IWMs in trajectory generation.

## 3 PRELIMINARIES

## 3.1 DAILY TRAJECTORY GENERATION

For a user u and a target date $d ,$ we represent a daily trajectory as an ordered sequence of location–time pairs,

$$
\tau _ { u , d } = \left. ( l _ { 0 } , t _ { 0 } ) , ( l _ { 1 } , t _ { 1 } ) , \dots , ( l _ { n _ { u , d } - 1 } , t _ { n _ { u , d } - 1 } ) \right. ,\tag{1}
$$

where $l _ { i }$ denotes the POI identifier of the i-th check-in and $t _ { i }$ denotes its arrival timestamp, $( l _ { i } , t _ { i } )$ denotes the user’s i-th check-in, and $n _ { u , d }$ denotes the number of check-ins in the trajectory. Given the user’s long-term behavioral report $\mathcal { R } _ { u } ,$ recent mobility history $\mathcal { H } _ { u , d } ,$ and target date d, our objective is to generate $\tau _ { u , d } .$ , describing the sequence of locations visited by the user and their corresponding visit times during the target mobility day. The recent mobility history $H _ { u , d }$ always precedes the target date, whereas $R _ { u }$ is constructed from a query-disjoint support set from the seen period. The construction and textual representations of $\mathcal { R } _ { u }$ and $\mathcal { H } _ { u , d }$ are detailed in Appendix A.

## 3.2 MOBILITY DYNAMICS AS A WORLD MODEL

From a world model perspective, each check-in is represented as a state $s _ { i } = ( l _ { i } , t _ { i } ) \in S$ , and each transition from $s _ { i } ~ \mathrm { t o } ~ s _ { i + 1 }$ is conditioned on a semantic action $m _ { i } \in { \mathcal { M } } .$ . We interpret m<sub>i</sub> as a transition-level latent variable representing a plausible motivation for the user’s next movement, such as work, shopping, or healthcare. Because psychological intent is not directly observed in mobility traces, we weakly supervise $m _ { i }$ using labels derived from destination categories, which serve as operational proxies rather than ground-truth annotations of human intent; the full taxonomy is provided in Appendix B.For a user u and a target date d, the corresponding mobility dynamics can be expressed as

$$
s _ { i + 1 } \sim \mathcal { T } \left( \cdot \mid s _ { i } , m _ { i } , \mathcal { R } _ { u } , \mathcal { H } _ { u , d } , d \right) .\tag{2}
$$

which characterizes the distribution of the next location and arrival time resulting from a motivation under the user’s behavioral context.

## 4 METHODOLOGY

As illustrated in Figure 1, we introduce MotiveMob, an LLM-based framework that generates daily mobility trajectories by repeatedly making “why–where–when” decisions. Unlike existing approaches that reason globally about an entire trajectory or daily plan (Wang et al., 2024; 2026), MotiveMob decomposes mobility generation into a sequence of motivation-conditioned check-in transitions. At each step, the motivation predictor $P _ { M }$ conditions on the current mobility state, the user’s long-term behavioral report $\mathcal { R } _ { u } .$ , and recent mobility history $\mathcal { H } _ { u , d }$ to infer the latent motivation for the next move or determine whether the trajectory should terminate. The state predictor $P _ { S }$ then treats the inferred motivation as a semantic action and grounds it in a candidate next location and arrival time, after which the candidate transition undergoes inference-time validation. The two predictors are trained separately: $P _ { S }$ learns from observed and alternative transitions, whereas $P _ { M }$ is fine-tuned with step-level motivation labels. During inference, each accepted state becomes the current mobility state for the next decision.

## 4.1 STATE PREDICTOR

## 4.1.1 ALTERNATIVE TRANSITION CONSTRUCTION

Because human mobility records contain only real transitions and do not reveal the outcomes of alternative motivations, we augment the observed data with model-synthesized alternative transitions constructed through a two-stage pipeline. Let $\mathcal { D } _ { \mathrm { o b s } } = \{ ( s _ { i } , m _ { i } , s _ { i + 1 } \bar { ) } \} _ { i = 1 } ^ { N }$ denote the set of observed transitions. For each observed transition in $\mathcal { D } _ { \mathrm { o b s } } ,$ we sample three alternative motivations $\{ m _ { i } ^ { j } \} _ { j = 1 } ^ { 3 } \subset$ $\mathcal { M } \setminus \{ m _ { i } \}$ , each representing a motivation that the user could have adopted from the same state $s _ { i }$ . In the first stage, a lightweight Transformer-based candidate generator (Vaswani et al., 2017) conditions on the trajectory history, the current state $s _ { i } ,$ and an alternative motivation $m _ { i } ^ { j }$ to produce a set of candidate next states $\mathcal { C } _ { i } ^ { j }$ . In the second stage, an LLM-based judge evaluates these candidates using the user’s historical information $\mathcal { H } _ { i }$ , the specified motivation $m _ { i } ^ { j }$ , and their spatial and temporal distances from the current state, denoted by $\delta _ { i } ^ { j }$ . It then selects the candidate most consistent with the user’s behavioral profile, mobility context, and the alternative motivation:

$$
\boldsymbol { \tilde { s } } _ { i + 1 } ^ { j } = \operatorname { J u d g e } _ { \operatorname { L L M } } \left( \boldsymbol { \mathcal { C } } _ { i } ^ { j } , \boldsymbol { \mathcal { H } } _ { i } , \boldsymbol { m } _ { i } ^ { j } , \boldsymbol { \delta } _ { i } ^ { j } \right) .\tag{3}
$$

The selected states form a pool of pseudo-labeled alternative transitions, denoted by $\mathcal { D } _ { \mathrm { a l t } }$ . To balance observed and alternative supervision during training, we uniformly retain a fraction $\rho$ of this pool, yielding $\mathcal { D } _ { \mathrm { a l t } } ^ { ( \rho ) } \subseteq \mathcal { D } _ { \mathrm { a l t } }$ . The augmented training set is then defined as $\mathcal { D } _ { \mathrm { a u g } } = \mathcal { D } _ { \mathrm { o b s } } \cup \mathcal { D } _ { \mathrm { a l t } } ^ { ( \rho ) }$

![](images/7a563b7d320fd40801108170eba57741e80ac3837136ac880b222f9bda963987.jpg)  
Figure 1: Overview of the MotiveMob framework. Given the user context, MotiveMob generates a daily trajectory step by step, first predicting why $( P _ { M } )$ and then where and when $( P _ { S } )$ . The implicit world model $P _ { S }$ is trained using observed and alternative transitions. The $\underline { { \widehat { ( \mu ) } } }$ marker denotes rollout validation, which performs speed-feasibility checking and repetition-based termination, as described in Section 4.3.

## 4.1.2 IMPLICIT WORLD MODELING

Using the augmented transition set $\mathcal { D } _ { \mathrm { a u g } } .$ , we train the state predictor $P _ { S }$ as an IWM of motivationconditioned mobility dynamics. Let ${ c _ { i } } ^ { \top } = ( s _ { i } , \mathcal { R } _ { u } , \mathcal { H } _ { u , d } , d )$ denote the context available at step i, including the current mobility state, the behavioral report of user u, the recent mobility history preceding the target date, and the target date $d .$ Given a context $c _ { i }$ and a motivation m treated as a semantic action, $P _ { S }$ models the resulting next-state distribution as $p _ { \theta _ { S } } ( s ^ { \prime } \mid c _ { i } , m )$ . For training, we associate each transition with its corresponding context and represent each instance in $\mathcal { D } _ { \mathrm { a u g } }$ as $( c _ { i } , m , s ^ { \prime } )$ . We train $P _ { S }$ by minimizing the average negative log-likelihood:

$$
\mathcal { L } _ { S } = - \frac { 1 } { | \mathcal { D } _ { \mathrm { a u g } } | } \sum _ { ( c _ { i } , m , s ^ { \prime } ) \in \mathcal { D } _ { \mathrm { a u g } } } \log p _ { \theta _ { S } } ( s ^ { \prime } \mid c _ { i } , m ) .\tag{4}
$$

This yields an offline, reward-free world modeling objective that combines factual grounding with broader coverage of motivation-conditioned mobility dynamics.

## 4.2 MOTIVATION PREDICTOR

The state predictor $P _ { S }$ grounds a given motivation in a next mobility state but does not determine which motivation should drive the next move. We therefore train a dedicated motivation predictor $P _ { M }$ to infer a step-level potential motivation, or predict trajectory termination, from the context $c _ { i } = ( s _ { i } , \mathcal { R } _ { u } , \mathcal { H } _ { i } )$ . To prevent future information leakage, $c _ { i }$ contains only information available up to step i and excludes the next state $s _ { i + 1 }$ and all subsequent states. Besides, for each observed trajectory, we append a terminal training instance whose label is END. We construct the motivation prediction dataset as $\mathcal { D } _ { M } = \{ ( c _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { M } }$ , where $y _ { i } = m _ { i }$ for an observed transition and $y _ { i } = \mathtt { E N D }$ when the trajectory should terminate. We fine-tune $P _ { M }$ by minimizing the average negative log-likelihood:

$$
\mathcal { L } _ { M } = - \frac { 1 } { \vert \mathcal { D } _ { M } \vert } \sum _ { ( c _ { i } , y _ { i } ) \in \mathcal { D } _ { M } } \log p _ { \theta _ { M } } ( y _ { i } \mid c _ { i } ) .\tag{5}
$$

This supervised objective enables $P _ { M }$ to infer the behavioral intent underlying the next move or determine that no further transition should be generated. During generation, a predicted motivation is passed to $P _ { S }$ for state grounding, whereas END terminates the trajectory.

## 4.3 MOTIVATION-GUIDED TRAJECTORY GENERATION

At inference time, $P _ { M }$ and $P _ { S }$ operate sequentially to generate a daily trajectory through steplevel“why–where–when” decisions. Generation begins with a START token in the textual context. Given this context, $P _ { M }$ predicts the first motivation and $P _ { S }$ generates the initial physical state $s _ { 0 }$ Speed validation is applied from the subsequent transition onward. At each step $i , P _ { M }$ greedily predicts mˆ <sub>i</sub> = arg max<sub>y∈M∪{END}</sub> $p _ { \theta _ { M } } ( y \mid c _ { i } )$ . If $\hat { m } _ { i } = \mathbb { E } \mathbb { N } \mathbb { D }$ , generation terminates. Otherwise, $P _ { S }$ conditions on $c _ { i }$ and $\hat { m } _ { i }$ to sample a provisional next state $\tilde { s } _ { i + 1 } \sim p _ { \theta _ { S } } ( \cdot \mid c _ { i } , \hat { m } _ { i } )$ , where $\tilde { s } _ { i + 1 }$ specifies the next location and arrival time. Once accepted, $\tilde { s } _ { i + 1 }$ becomes the current state in $c _ { i + 1 }$ and the same procedure is repeated for the next transition.

Inference-Time Rollout Validation. Before a provisional state is added to the trajectory, it is validated for temporal and travel feasibility. Specifically, its arrival time must not precede the current time, and its implied travel speed must not exceed $v _ { \mathrm { m a x } } = 5 0$ km/h. The selection of this threshold is discussed in Appendix C. An invalid proposal is discarded and resampled from $P _ { S }$ If no proposal satisfies the constraints within a fixed number of attempts, the last proposal is regrounded to a feasible location of the same activity category using the user’s observed mobility information (Li et al., 2026). If no such feasible location is available, the final model proposal is retained unchanged. We additionally apply repetition-aware termination once the generated trajectory reaches a minimum length: the rollout terminates if the candidate-augmented location sequence contains a short consecutively repeating pattern. Generation also terminates when the prescribed maximum trajectory length is reached.

Through this iterative process, $P _ { M }$ determines why the next move occurs, while $P _ { S }$ grounds the predicted motivation in where and when it occurs. Additional implementation details are provided in the appendices. Appendix D provides implementation details for alternative transition construction, predictor training, and autoregressive trajectory inference; Appendix E provides the prompts used for training and inference; and Appendix F presents pseudocode for inference-time trajectory generation.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

Dataset. All experiments are conducted on a human mobility dataset collected in Tokyo through the Twitter and Foursquare APIs. The original dataset consists of individual check-ins with location and timestamp information. We transform consecutive check-ins from the same user into transitions, each comprising the origin and destination states and one of 11 behavioral motivation labels derived from the category of the destination venue. We additionally append an END label after the final check-in to mark trajectory termination. For the seen-user, seen-time setting, we randomly hold out 20% of each user’s trajectories for evaluation and use the remaining 80% for training.

Distribution-Shift Settings. We evaluate generalization along both user and temporal dimensions. The training set contains trajectories from 25 seen users collected between January and October 2019. For user-level generalization, we evaluate exclusively on an additional 50 users whose trajectories are not included in training. For temporal generalization, we consider two unseen periods: November 2019 to February 2020, representing a regular seasonal shift, and April to October 2020, representing a pandemic-induced mobility shift following the onset of COVID-19 restrictions. Detailed split construction and leakage prevention are provided in Appendix G.

Baseline Methods. We compare MotiveMob with two categories of baselines: (1) pretrainingbased models, including LSTM (Hochreiter & Schmidhuber, 1997), MHSA (Vaswani et al., 2017), DeepMove (Feng et al., 2018), GetNext (Yang et al., 2022), TrajGAIL (Choi et al., 2021), Diff-Traj (Zhu et al., 2023b), GNPR-SID (Wang et al., 2025), and LLM4POI (Li et al., 2024); and (2) prompting-based models, including LLM-Mob (Wang et al., 2023), LLMMove (Feng et al., 2024), LLM-ZS (Beneduce et al., 2025), LLMob (Wang et al., 2024), and ELLMob (Wang et al., 2026).

Evaluation Metrics. Following Yuan et al. (2022) and Wang et al. (2024), we evaluate generated trajectories using four distributional metrics: Step Distance (SD), the geographic distance between consecutive locations; Step Interval (SI), the elapsed time between consecutive check-ins; Daily Activity Routine Distribution (DARD), the joint distribution of discretized time slot t and activity category c; and Spatiotemporal Visit Distribution (STVD), the joint distribution of time and location, represented by (t, lat, lng). For each metric, we compute the Jensen–Shannon divergence (JSD) between the generated and real distributions, with lower values indicating better performance.

## 5.2 RESULTS AND ANALYSIS

Tables 1a and 1b report the results for seen and unseen users across the three temporal settings. All methods use the same validation and termination rules. MotiveMob achieves lower JSD than all baselines across both user groups and all three temporal settings. Its advantage persists for users excluded from training, indicating that its motivation-conditioned modeling transfers beyond observed user identities. Although temporal shifts generally increase divergence, with the COVID-19 period producing the largest mismatch in most dimensions, MotiveMob maintains its advantage under both regular seasonal and pandemic-induced shifts. In contrast, the baselines exhibit metric-specific strengths but do not perform consistently well across mobility dimensions. To further examine MotiveMob across both urban contexts and backbone choices, Appendix H reports additional result for Osaka, while Appendix I evaluates MotiveMob with different LLM backbones.

## 5.3 ANALYSIS OF PREDICTION MODULES

Step-level Motivation Predictor. The w/o $P _ { M }$ rows in Tables 1a and 1b evaluate a direct predictor that generates the next mobility state from the current context without first inferring a step-level motivation. This variant retains a learned termination decision by predicting either the next state or END; the comparison therefore isolates the effect of motivation-mediated state prediction rather than removing trajectory termination.

Removing $P _ { M }$ increases both SI and DARD in all six evaluation settings. The mean relative increases in SI and DARD are 44.9% and 30.1% for seen users and 44.0% and 40.8% for unseen users, respectively. In contrast, the effects of $P _ { M }$ on the spatial metrics are mixed: the mean absolute relative change in STVD is only 2.6%, while $P _ { M }$ reduces SD divergence in two of the three seen-user settings but increases it in all three unseen-user settings. These improvements in SI and DARD are not simply explained by termination behavior or average trajectory length. As detailed in Appendix J, the direct predictor generates 7.52 states per day on average, compared with 8.08 for MotiveMob and 6.82 in the ground truth. Although the direct predictor is closer to the ground-truth mean length, it still performs worse on SI and DARD. These results suggest that step-level motivation primarily helps preserve temporal and activity semantics by narrowing spatially plausible alternatives according to behavioral intent, which helps determine when the next move should occur and how it fits into the user’s daily activity sequence. To visualize the effect of motivation predictor, Appendix K presents representative trajectory-level case studies comparing MotiveMob w/ and w/o motivation predictor.

IWM for State Prediction. We compare the IWM-based state predictor, trained on both observed and alternative transitions, with two alternatives while keeping the motivation predictor $P _ { M }$ fixed and inference-time rollout validation disabled. The positive-only predictor is trained solely on observed transitions, whereas the rule-based predictor retrieves observed mobility transitions compatible with each predicted motivation instead of learning a generative transition model.

As shown in Table 2, the IWM-based predictor achieves lower SD than the positive-only predictor in all six settings and lower STVD in four, including all three unseen-user settings, whereas the positive-only predictor achieves lower SI and DARD in five settings. Relative to the rule-based predictor, the IWM-based predictor achieves lower STVD in all six settings and lower DARD in four, whereas the rule-based predictor achieves lower SD in four and lower SI in all six. These results highlight a trade-off between matching individual mobility marginals and preserving userconditioned spatiotemporal structure. The clearest advantage of IWM training lies in modeling the joint time–location distribution, particularly for unseen users, suggesting that alternative transitions broaden the motivation-conditioned state distribution and improve spatiotemporal generalization beyond observed-transition learning or rule-based retrieval alone. Appendix L further shows that IWM initialization provides no consistent final improvement for the motivation predictor.

Table 1: Evaluation results for seen and unseen users across the three temporal settings. Panels (a) and (b) report results for seen and unseen users, respectively. In each panel, the upper and lower baseline sections correspond to pretraining-based and prompting-based methods, respectively. All values are JSD; lower is better. The best result in each column of each panel is shown in bold, and the runner-up is underlined. Speed and Rep. denote speed validation and repetition-aware termination, respectively. Best results are in bold, and runner-up results are underlined.  
(a) Seen users
<table><tr><td rowspan="2">Method</td><td colspan="4">Seen period (908 trajectories)</td><td colspan="4">Regular unseen period (690 trajectories)</td><td colspan="4">COVID-19 unseen period (392 trajectories)</td></tr><tr><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td></tr><tr><td>LSTM</td><td>0.0915</td><td>0.1042</td><td>0.2510</td><td>0.4079</td><td>0.1014</td><td>0.1530</td><td>0.3046</td><td>0.5159</td><td>0.0823</td><td>0.1600</td><td>0.3713</td><td>0.6267</td></tr><tr><td>MHSA</td><td>0.0319</td><td>0.0507</td><td>0.1898</td><td>0.4049</td><td>0.0393</td><td>0.0590</td><td>0.2157</td><td>0.4753</td><td>0.0424</td><td>0.0621</td><td>0.2638</td><td>0.6201</td></tr><tr><td>DeepMove</td><td>0.0830</td><td>0.0990</td><td>0.2334</td><td>0.4991</td><td>0.0593</td><td>0.1305</td><td>0.3592</td><td>0.5331</td><td>0.0911</td><td>0.1372</td><td>0.4313</td><td>0.6422</td></tr><tr><td>GetNext</td><td>0.0440</td><td>0.2414</td><td>0.2126</td><td>0.4917</td><td>0.0559</td><td>0.2387</td><td>0.2114</td><td>0.5659</td><td>0.0773</td><td>0.2046</td><td>0.1991</td><td>0.6613</td></tr><tr><td>TrajGAIL</td><td>0.1079</td><td>0.1382</td><td>0.1140</td><td>0.5910</td><td>0.1183</td><td>0.1295</td><td>0.1360</td><td>0.6367</td><td>0.1311</td><td>0.1363</td><td>0.2136</td><td>0.6570</td></tr><tr><td>DiffTraj</td><td>0.2000</td><td>0.1204</td><td>0.4777</td><td>0.6931</td><td>0.2193</td><td>0.1125</td><td>0.5157</td><td>0.6898</td><td>0.2595</td><td>0.1052</td><td>0.5716</td><td>0.6903</td></tr><tr><td>GNPR-SID</td><td>0.0171</td><td>0.0163</td><td>0.1006</td><td>0.4167</td><td>0.0145</td><td>0.0207</td><td>0.1103</td><td>0.4799</td><td>0.0491</td><td>0.0310</td><td>0.1445</td><td>0.5723</td></tr><tr><td>LLM4POI</td><td>0.0098</td><td>0.0236</td><td>0.1090</td><td>0.3788</td><td>0.0138</td><td>0.0340</td><td>0.1329</td><td>0.4622</td><td>0.0452</td><td>0.0217</td><td>0.1405</td><td>0.5750</td></tr><tr><td>LLMMove</td><td>0.0839</td><td>0.1164</td><td>0.1906</td><td>0.5637</td><td>0.0830</td><td>0.1091</td><td>0.1901</td><td>0.5607</td><td>0.1137</td><td>0.0937</td><td>0.2572</td><td>0.6041</td></tr><tr><td>LLM-ZS</td><td>0.0306</td><td>0.1034</td><td>0.2369</td><td>0.5936</td><td>0.0441</td><td>0.1016</td><td>0.2505</td><td>0.6014</td><td>0.0732</td><td>0.0813</td><td>0.2817</td><td>0.6295</td></tr><tr><td>LLM-Mob</td><td>0.0951</td><td>0.1171</td><td>0.3242</td><td>0.5639</td><td>0.1166</td><td>0.1135</td><td>0.3336</td><td>0.5856</td><td>0.1328</td><td>0.0941</td><td>0.3501</td><td>0.6201</td></tr><tr><td>LLMob</td><td>0.0919</td><td>0.2496</td><td>0.1819</td><td>0.5461</td><td>0.0820</td><td>0.2422</td><td>0.2353</td><td>0.5972</td><td>0.0982</td><td>0.2610</td><td>0.2013</td><td>0.6229</td></tr><tr><td>ELLMob</td><td>0.0501</td><td>0.1053</td><td>0.2791</td><td>0.4598</td><td>0.0476</td><td>0.1085</td><td>0.2863</td><td>0.4937</td><td>0.0583</td><td>0.1245</td><td>0.3309</td><td>0.5836</td></tr><tr><td>MotiveMob</td><td>0.0062</td><td>0.0100</td><td>0.0765</td><td>0.3658</td><td>0.0102</td><td>0.0161</td><td>0.0995</td><td>0.4301</td><td>0.0312</td><td>0.0147</td><td>0.1377</td><td>0.5487</td></tr><tr><td>MotiveMob w/o  $P _ { M }$ </td><td>0.0106</td><td>0.0168</td><td>0.1092</td><td>0.3757</td><td>0.0179</td><td>0.0174</td><td>0.1300</td><td>0.4541</td><td>0.0253</td><td>0.0233</td><td>0.1610</td><td>0.5530</td></tr><tr><td>MotiveMob w/o Ru</td><td>0.0168</td><td>0.0219</td><td>0.0912</td><td>0.5142</td><td>0.0185</td><td>0.0259</td><td>0.1030</td><td>0.5583</td><td>0.0525</td><td>0.0193</td><td>0.1266</td><td>0.5836</td></tr><tr><td>MotiveMob w/o  $\mathcal { H } _ { u , d }$ </td><td>0.0120</td><td>0.0291</td><td>0.1422</td><td>0.4291</td><td>0.0142</td><td>0.0343</td><td>0.1760</td><td>0.4887</td><td>0.0805</td><td>0.0299</td><td>0.1875</td><td>0.5945</td></tr><tr><td>MotiveMob w/o Speed</td><td>0.0276</td><td>0.0109</td><td>0.0735</td><td>0.3762</td><td>0.0250 0.0125</td><td>0.0152 0.0235</td><td>0.0951 0.1114</td><td>0.4351</td><td>0.0878 0.0281</td><td>0.0241</td><td>0.1300</td><td>0.5548</td></tr><tr><td>MotiveMob w/o Rep.</td><td>0.0074</td><td>0.0161</td><td>0.0891</td><td>0.3749</td><td></td><td></td><td></td><td>0.4423</td><td></td><td>0.0245</td><td>0.1461</td><td>0.5578</td></tr></table>

(b) Unseen users
<table><tr><td rowspan="2">Method</td><td colspan="4">Seen period (1345 trajectories)</td><td colspan="4">Regular unseen period (795 trajectories)</td><td colspan="4">COVID-19 unseen period (541 trajectories)</td></tr><tr><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td></tr><tr><td>LSTM</td><td>0.3607</td><td>0.3354</td><td>0.5257</td><td>0.6193</td><td>0.3610</td><td>0.3556</td><td>0.5601</td><td>0.6515</td><td>0.3456</td><td>0.3554</td><td>0.5648</td><td>0.6786</td></tr><tr><td>MHSA</td><td>0.1779</td><td>0.0733</td><td>0.3335</td><td>0.5796</td><td>0.1510</td><td>0.0895</td><td>0.3538</td><td>0.6122</td><td>0.1557</td><td>0.0918</td><td>0.3639</td><td>0.6716</td></tr><tr><td>DeepMove</td><td>0.2247</td><td>0.1621</td><td>0.2977</td><td>0.6660</td><td>0.2517</td><td>0.1896</td><td>0.4468</td><td>0.6883</td><td>0.2393</td><td>0.1889</td><td>0.4370</td><td>0.6897</td></tr><tr><td>GetNext</td><td>0.0715</td><td>0.1930</td><td>0.2303</td><td>0.6223</td><td>0.0627</td><td>0.1818</td><td>0.2419</td><td>0.6561</td><td>0.0760</td><td>0.1931</td><td>0.2246</td><td>0.6703</td></tr><tr><td>TrajGAIL</td><td>0.0999</td><td>0.0801</td><td>0.1051</td><td>0.5941</td><td>0.1237</td><td>0.0808</td><td>0.1273</td><td>0.6222</td><td>0.1079</td><td>0.0866</td><td>0.1540</td><td>0.6638</td></tr><tr><td>DiffTraj</td><td>0.2176</td><td>0.0671</td><td>0.5589</td><td>0.6915</td><td>0.2082</td><td>0.0708</td><td>0.5681</td><td>0.6914</td><td>0.2327</td><td>0.0773</td><td>0.5459</td><td>0.6921</td></tr><tr><td>GNPR-SID</td><td>0.0270</td><td>0.0488</td><td>0.1428</td><td>0.4762</td><td>0.0263</td><td>0.0508</td><td>0.1514</td><td>0.5356</td><td>0.0388</td><td>0.0543</td><td>0.1706</td><td>0.6269</td></tr><tr><td>LLM4POI</td><td>0.0469</td><td>0.0329</td><td>0.1152</td><td>0.5391</td><td>0.0516</td><td>0.0339</td><td>0.1452</td><td>0.5873</td><td>0.0553</td><td>0.0369</td><td>0.1502</td><td>0.6386</td></tr><tr><td>LLMMove</td><td>0.0914</td><td>0.1376</td><td>0.1523</td><td>0.5303</td><td>0.0787</td><td>0.1330</td><td>0.1612</td><td>0.5921</td><td>0.1044</td><td>0.1557</td><td>0.1802</td><td>0.6253</td></tr><tr><td>LLM-ZS</td><td>0.0313</td><td>0.1049</td><td>0.2171</td><td>0.5713</td><td>0.0373</td><td>0.1221</td><td>0.2394</td><td>0.6212</td><td>0.0442</td><td>0.1122</td><td>0.2449</td><td>0.6473</td></tr><tr><td>LLM-Mob</td><td>0.0948</td><td>0.0991</td><td>0.3200</td><td>0.5314</td><td>0.1007</td><td>0.1015</td><td>0.3284</td><td>0.5846</td><td>0.1196</td><td>0.1091</td><td>0.3540</td><td>0.6271</td></tr><tr><td>LLMob</td><td>0.0555</td><td>0.1943</td><td>0.1647</td><td>0.5449</td><td>0.0677</td><td>0.1849</td><td>0.1860</td><td>0.5851</td><td>0.0445</td><td>0.1830</td><td>0.2048</td><td>0.6422</td></tr><tr><td>ELLMob</td><td>0.0437</td><td>0.0751</td><td>0.2159</td><td>0.4830</td><td>0.0401</td><td>0.0750</td><td>0.2341</td><td>0.5428</td><td>0.0364</td><td>0.0861</td><td>0.2875</td><td>0.6336</td></tr><tr><td>MotiveMob</td><td>0.0213</td><td>0.0204</td><td>0.0868</td><td>0.4498</td><td>0.0256</td><td>0.0273</td><td>0.1103</td><td>0.5137</td><td>0.0273</td><td>0.0303</td><td>0.1153</td><td>0.6060</td></tr><tr><td>MotiveMob w/o  $P _ { M }$ </td><td>0.0201</td><td>0.0311</td><td>0.1219</td><td>0.4627</td><td>0.0200</td><td>0.0354</td><td>0.1471</td><td>0.5153</td><td>0.0225</td><td>0.0454</td><td>0.1714</td><td>0.6253</td></tr><tr><td>MotiveMob w/o  $\mathcal { R } _ { u }$ </td><td>0.0142</td><td>0.0385</td><td>0.1023</td><td>0.5082</td><td>0.0146</td><td>0.0357</td><td>0.1243</td><td>0.5700</td><td>0.0154</td><td>0.0445</td><td>0.1485</td><td>0.6149</td></tr><tr><td>MotiveMob w/o  $\mathcal { H } _ { u , d }$ </td><td>0.0289</td><td>0.0258</td><td>0.1193</td><td>0.4777</td><td>0.0351</td><td>0.0309 0.0373</td><td>0.1443 0.1145</td><td>0.5394</td><td>0.0306</td><td>0.0495</td><td>0.1385</td><td>0.6272</td></tr><tr><td>MotiveMob w/o Speed</td><td>0.0633</td><td>0.0287</td><td>0.0870</td><td>0.4527</td><td>0.0698 0.0227</td><td>0.0329</td><td>0.1079</td><td>0.5230 0.5216</td><td>0.0775</td><td>0.0454</td><td>0.1164</td><td>0.6163</td></tr><tr><td>MotiveMob w/o Rep.</td><td>0.0181</td><td>0.0262</td><td>0.0925</td><td>0.4535</td><td></td><td></td><td></td><td></td><td>0.0210</td><td>0.0326</td><td>0.1122</td><td>0.6082</td></tr></table>

## 5.4 ABLATION STUDIES

User-specific information. Removing the behavioral report $\mathcal { R } _ { u }$ worsens SI and STVD in all six settings, confirming its role as a long-term prior over when and where a user typically performs an activity. Its effect on DARD is weaker for seen users but more pronounced for unseen users. The mixed SD results further suggest that displacement patterns encoded in the report are less reliably calibrated under user-level shifts. Recent mobility history provides more immediate conditioning: removing $\mathcal { H } _ { u , d }$ worsens SI in all settings and substantially increases DARD and SD for both user groups. Its effect on STVD is smaller than that of $\mathcal { R } _ { u } .$ , indicating that recent history mainly locates the user within the current routine and constrains the next activity, its timing, and the associated displacement.

Table 2: State predictor comparison without rollout validation for seen and unseen users across the three temporal settings. All values are JSD; lower is better. The best result in each column of each panel is shown in bold, and the runner-up is underlined.
<table><tr><td rowspan="2">State predictor  $P _ { S }$ </td><td colspan="4">Seen period</td><td colspan="4">Regular unseen period</td><td colspan="4">COVID-19 unseen period</td></tr><tr><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td></tr><tr><td colspan="10">Seen users</td><td colspan="3"></td></tr><tr><td>MotiveMob (IWM  $P _ { S } )$ </td><td>0.0241</td><td>0.0159</td><td>0.0857</td><td>0.3841</td><td>0.0307</td><td>0.0212</td><td>0.1054</td><td>0.4558</td><td>0.0880</td><td>0.0309</td><td>0.1632</td><td>0.5745</td></tr><tr><td>w/ positive-only  $P _ { S }$ </td><td>0.0328</td><td>0.0115</td><td>0.0707</td><td>0.3870</td><td>0.0376</td><td>0.0185</td><td>0.0943</td><td>0.4469</td><td>0.0914</td><td>0.0390</td><td>0.1586</td><td>0.5602</td></tr><tr><td>w/ rule-based  $P _ { S }$ </td><td>0.0374</td><td>0.0101</td><td>0.0992</td><td>0.4286</td><td>0.0405</td><td>0.0165</td><td>0.1111</td><td>0.4932</td><td>0.0687</td><td>0.0198</td><td>0.1576</td><td>0.6230</td></tr><tr><td colspan="10">Unseen users</td><td colspan="3"></td></tr><tr><td>MotiveMob (IWM  $P _ { S } )$ </td><td>0.0460</td><td>0.0356</td><td>0.0906</td><td>0.4559</td><td>0.0473</td><td>0.0422</td><td>0.1120</td><td>0.5240</td><td>0.0597</td><td>0.0359</td><td>0.1209</td><td>0.6109</td></tr><tr><td>w/ positive-only  $P _ { S }$ </td><td>0.0713</td><td>0.0244</td><td>0.0878</td><td>0.4875</td><td>0.0876</td><td>0.0300</td><td>0.1181</td><td>0.5417</td><td>0.1045</td><td>0.0220</td><td>0.1165</td><td>0.6440</td></tr><tr><td>w/ rule-based  $P _ { S }$ </td><td>0.0388</td><td>0.0333</td><td>0.0916</td><td>0.5663</td><td>0.0375</td><td>0.0341</td><td>0.1097</td><td>0.5926</td><td>0.0434</td><td>0.0317</td><td>0.1399</td><td>0.6478</td></tr></table>

![](images/49d5fa0edfe41aeeacd08c64a801148afd6fc1834b998f6551e9925e60299150.jpg)  
Figure 2: Ground-truth and predicted spatial mobility distributions for unseen users across two unseen periods in Tokyo. Darker red indicates higher visit density.

Inference-time rollout validation. Removing speed validation increases SD and STVD in all six settings, whereas its effects on SI and DARD are smaller and less uniform. Speed validation therefore primarily prevents physically implausible spatial transitions. Repetition-aware termination serves a different function: disabling it consistently worsens SI and STVD but produces smaller and more mixed changes in SD and DARD. Its primary role is therefore to prevent repeated location cycles from extending trajectories to the maximum length, rather than driving MotiveMob’s overall distributional gains. Detailed resampling, correction, and termination statistics are reported in Appendix M.

## 5.5 SPATIAL DISTRIBUTION ALIGNMENT UNDER USER AND TEMPORAL SHIFTS

As shown in Figure 2, all three methods recover the primary activity centers in the ground-truth distribution. Compared with LLM4POI, MotiveMob forms more compact hotspots and exhibits less peripheral dispersion, thereby better preserving the concentrated structure of the ground-truth distribution. Compared with ELLMob, MotiveMob more closely matches the ground-truth allocation of visit mass across activity regions, particularly under the COVID-19 shift. These results suggest that MotiveMob combines compact hotspot structure with more robust spatial alignment under joint user and temporal shifts. Additional spatial-distribution results are provided in Appendix N.

## 6 CONCLUSIONS AND LIMITATIONS

By combining step-level prediction of motivation proxies, motivation-conditioned state generation, and inference-time rollout validation, our motivation-first framework, MotiveMob, achieved the strongest overall distributional fidelity across the evaluated user groups and temporal periods. Spatial analysis further showed that MotiveMob better preserves core activity regions for unseen users under temporal shifts. Additional analyses indicate that step-level motivation prediction primarily improves temporal and activity fidelity, while IWM training with alternative transitions improves joint spatiotemporal generalization, particularly for unseen users. Together, these findings support motivation proxies as useful intermediate representations for mobility trajectory generation.

As for limitations, MotiveMob relies on a predefined motivation taxonomy and pseudo-labeled alternative transitions whose correspondence to unobserved user intentions cannot be directly verified. The current IWM augmentation also exhibits a metric-specific trade-off: improvements in the joint time–location distribution do not consistently coincide with improvements in the step-distance distribution, suggesting that the generated alternative transitions may not simultaneously preserve local movement geometry and global spatiotemporal structure. Moreover, the step-level training objectives do not explicitly optimize trajectory-level coherence or capture the delayed effects of earlier decisions. Future work could explore adaptive motivation representations, improved augmentation strategies that better preserve both local and global mobility patterns, and hierarchical sequential objectives with trajectory-level feedback or reinforcement learning.

## ACKNOWLEDGMENTS

This work was financially supported by JST CREST JPMJCR22M2, JSPS KAKENHI JP23K17456, JP23K28096, JP24K02996, JP25H01117, JP25K21207, JP26K03246.

## REFERENCES

Mostafa Abdou, Artur Kulmizev, Daniel Hershcovich, Stella Frank, Ellie Pavlick, and Anders Søgaard. Can language models encode perceptual structure without grounding? a case study in color. In Proceedings of the 25th conference on computational natural language learning, pp. 109–132, 2021.

Ciro Beneduce, Bruno Lepri, and Massimiliano Luca. Large language models are zero-shot next location predictors. IEEE Access, 13:77456–77467, 2025.

Serina Chang, Emma Pierson, Pang Wei Koh, Jaline Gerardin, Beth Redbird, David Grusky, and Jure Leskovec. Mobility network models of covid-19 explain inequities and inform reopening. Nature, 589(7840):82–87, 2021.

Seongjin Choi, Jiwon Kim, and Hwasoo Yeo. Trajgail: Generating urban vehicle trajectories using generative adversarial imitation learning. Transportation Research Part C: Emerging Technologies, 128:103091, 2021.

Chen Chu, Hengcai Zhang, Peixiao Wang, and Feng Lu. Simulating human mobility with a trajectory generation framework based on diffusion model. International Journal ofGeographical Information Science, 38(5):847–878, 2024.

Jichen Feng, Yifan Zhang, Chenggong Zhang, Yifu Lu, Shilong Liu, and Mengdi Wang. Web world models. arXiv preprint arXiv:2512.23676, 2025.

Jie Feng, Yong Li, Chao Zhang, Funing Sun, Fanchao Meng, Ang Guo, and Depeng Jin. Deepmove: Predicting human mobility with attentional recurrent networks. In Proceedings of the 2018 world wide web conference, pp. 1459–1468, 2018.

Shanshan Feng, Haoming Lyu, Fan Li, Zhu Sun, and Caishun Chen. Where to move next: Zeroshot generalization of llms for next poi recommendation. In 2024 ieee conference on artificial intelligence (cai), pp. 1530–1535. IEEE, 2024.

Sebastien Gambs, Marc-Olivier Killijian, and Miguel N ´ u´nez del Prado Cortez. Next place prediction˜ using mobility markov chains. In Proceedings of the first workshop on measurement, privacy, and mobility, pp. 1–6, 2012.

Jie Gao and Yaoxin Wu. LLMs for human mobility: Opportunities, challenges, and future directions, 2026. URL https://arxiv.org/abs/2603.12420.

Letian Gong, Yan Lin, Xinyue Zhang, Yiwen Lu, Xuedi Han, Yichen Liu, Shengnan Guo, Youfang Lin, and Huaiyu Wan. Mobility-LLM: Learning visiting intentions and travel preference from human mobility data with large language models. In Advances in Neural Information

Processing Systems, volume 37, pp. 36185–36217, 2024. doi: 10.52202/079017-1141. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ 3fb6c52aeb11e09053c16eabee74dd7b-Abstract-Conference.html.

Yu Gu, Kai Zhang, Yuting Ning, Boyuan Zheng, Boyu Gou, Tianci Xue, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, et al. Is your llm secretly a world model of the internet? modelbased planning for web agents. arXiv preprint arXiv:2411.06559, 2024.

David Ha and Jurgen Schmidhuber. Recurrent world models facilitate policy evolution.¨ Advances in neural information processing systems, 31, 2018.

Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨ Neural computation, 9(8): 1735–1780, 1997.

Dou Huang, Xuan Song, Zipei Fan, Renhe Jiang, Ryosuke Shibasaki, Yu Zhang, Haizhong Wang, and Yugo Kato. A variational autoencoder based generative model of urban human mobility. In 2019 IEEE conference on multimedia information processing and retrieval (MIPR), pp. 425–430. IEEE, 2019.

Xiangjie Kong, Qiao Chen, Mingliang Hou, Hui Wang, and Feng Xia. Mobility trajectory generation: A survey. Artificial Intelligence Review, 56(Suppl 3):3057–3098, 2023. doi: 10.1007/ s10462-023-10598-x. URL https://doi.org/10.1007/s10462-023-10598-x.

Kenneth Li, Aspen K Hopkins, David Bau, Fernanda Viegas, Hanspeter Pfister, and Martin Watten-´ berg. Emergent world representations: Exploring a sequence model trained on a synthetic task. arXiv preprint arXiv:2210.13382, 2022.

Peibo Li, Maarten De Rijke, Hao Xue, Shuang Ao, Yang Song, and Flora D Salim. Large language models for next point-of-interest recommendation. In Proceedings of the 47th international ACM SIGIR conference on research and development in information retrieval, pp. 1463–1472, 2024.

Siyu Li, Toan Tran, Haowen Lin, John Krumm, Cyrus Shahabi, Lingyi Zhao, Khurram Shafique, and Li Xiong. Geo-llama: Leveraging llms for human mobility trajectory generation with constraints. In 2025 26th IEEE International Conference on Mobile Data Management (MDM), pp. 20–31. IEEE, 2025.

Siyu Li, Toan Tran, Lingyi Zhao, Khurram Shafique, and Li Xiong. TrajGenAgent: A hierarchical LLM agent for human mobility trajectory generation. In Proceedings ofthe 27th IEEE International Conference on Mobile Data Management (MDM), 2026.

Xishun Liao, Qinhua Jiang, Brian Yueshuai He, Yifan Liu, Chenchen Kuai, and Jiaqi Ma. Deep activity model: A generative deep learning approach for human mobility pattern synthesis. IEEE Transactions on Intelligent Transportation Systems, 27(6):6519–6536, 2026. doi: 10.1109/TITS. 2026.3678542. URL https://doi.org/10.1109/TITS.2026.3678542.

Yifan Liu, Xishun Liao, Haoxuan Ma, Brian Yueshuai He, Chris Stanford, and Jiaqi Ma. Human mobility modeling with household coordination activities under limited information via retrievalaugmented LLMs. In 2025 IEEE 28th International Conference on Intelligent Transportation Systems, pp. 951–958. IEEE, 2025. doi: 10.1109/ITSC60802.2025.11423539. URL https: //doi.org/10.1109/ITSC60802.2025.11423539.

Qingyue Long, Can Rong, Tong Li, and Yong Li. Dynamic population distribution aware human trajectory generation with diffusion model. ACM Transactions on Intelligent Systems and Technology, 17(3):69:1–69:23, 2026. doi: 10.1145/3787459. URL https://doi.org/10.1145/ 3787459.

Pablo Alvarez Lopez, Michael Behrisch, Laura Bieker-Walz, Jakob Erdmann, Yun-Pang Flotter ¨ od,¨ Robert Hilbrich, Leonhard Lucken, Johannes Rummel, Peter Wagner, and Evamarie Wießner.¨ Microscopic traffic simulation using sumo. In 2018 21st international conference on intelligent transportation systems (ITSC), pp. 2575–2582. Ieee, 2018.

Xin Lu, Erik Wetter, Nita Bharti, Andrew J Tatem, and Linus Bengtsson. Approaching the limit of predictability in human mobility. Scientific reports, 3(1):2923, 2013.

Massimiliano Luca, Gianni Barlacchi, Bruno Lepri, and Luca Pappalardo. A survey on deep learning for human mobility. ACM Computing Surveys (CSUR), 55(1):1–44, 2021.

Fan-Ming Luo, Tian Xu, Hang Lai, Xiong-Hui Chen, Weinan Zhang, and Yang Yu. A survey on model-based reinforcement learning. Science China Information Sciences, 67(2):121101, 2024.

Yingtao Luo, Qiang Liu, and Zhaocheng Liu. Stan: Spatio-temporal attention network for next location recommendation. In Proceedings of the web conference 2021, pp. 2177–2185, 2021.

Ryotaro Ohara, Hideaki Uchida, Yohei Yamaguchi, Shinya Yoshizawa, Katsuya Sakai, and Yoshiyuki Shimoda. Generating individual trajectories for synthetic populations with a large language model. In Proceedings of the 5th Asia Conference of the International Building Performance Simulation Association, pp. 1044–1051, 2024. doi: 10.69357/asim2024.1291. URL https: //doi.org/10.69357/asim2024.1291.

Richard S Sutton. Dyna, an integrated architecture for learning, planning, and reacting. ACM Sigart Bulletin, 2(4):160–163, 1991.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Dongsheng Wang, Yuxi Huang, Shen Gao, Yifan Wang, Chengrui Huang, and Shuo Shang. Generative next poi recommendation with semantic id. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 2904–2914, 2025.

Jiawei Wang, Renhe Jiang, Chuang Yang, Zengqing Wu, Makoto Onizuka, Ryosuke Shibasaki, Noboru Koshizuka, and Chuan Xiao. Large language models as urban residents: An llm agent framework for personal mobility generation. Advances in Neural Information Processing Systems, 37:124547–124574, 2024.

Xinglei Wang, Meng Fang, Zichao Zeng, and Tao Cheng. Where would i go next? large language models as human mobility predictors. arXiv preprint arXiv:2308.15197, 2023.

Yusong Wang, Chuang Yang, Jiawei Wang, Xiaohang Xu, Jiayi Xu, Dongyuan Li, Chuan Xiao, and Renhe Jiang. Ellmob: Event-driven human mobility generation with self-aligned llm framework. arXiv preprint arXiv:2603.07946, 2026.

Song Yang, Jiamou Liu, and Kaiqi Zhao. Getnext: Trajectory flow map enhanced transformer for next poi recommendation. In Proceedings ofthe 45th International ACM SIGIR Conference on research and development in information retrieval, pp. 1144–1153, 2022.

Yuan Yuan, Jingtao Ding, Huandong Wang, Depeng Jin, and Yong Li. Activity trajectory generation via modeling spatiotemporal dynamics. In Proceedings ofthe 28th ACM SIGKDD conference on knowledge discovery and data mining, pp. 4752–4762, 2022.

Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Zeyi Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun Chen, Xiaohan Fu, et al. Agent learning via early experience. arXiv preprint arXiv:2510.08558, 2025.

Kunyi Zhang, Yanbo Pang, Yurong Zhang, and Yoshihide Sekimoto. Mobglm: A large language model for synthetic human mobility generation. In Proceedings of the 32nd ACM International Conference on Advances in Geographic Information Systems, pp. 629–632, 2024.

Xin Zhang, Yanhua Li, Xun Zhou, Ziming Zhang, and Jun Luo. Trajgail: Trajectory generative adversarial imitation learning for long-term decision analysis. In 2020 IEEE International Conference on Data Mining (ICDM), pp. 801–810. IEEE, 2020.

Yuanshao Zhu, Yongchao Ye, Ying Wu, Xiangyu Zhao, and James Yu. Synmob: Creating highfidelity synthetic gps trajectory dataset for urban mobility analysis. Advances in Neural Information Processing Systems, 36:22961–22977, 2023a.

Yuanshao Zhu, Yongchao Ye, Shiyao Zhang, Xiangyu Zhao, and James Yu. Difftraj: Generating gps trajectory with diffusion probabilistic model. Advances in Neural Information Processing Systems, 36:65168–65188, 2023b.

## A USER-SPECIFIC CONTEXT REPRESENTATION

MotiveMob conditions both predictors on two sources of user-specific information: a long-term behavioral report $\mathcal { R } _ { u }$ and a recent mobility history H . This section specifies their textual representations.

## A.1 BEHAVIORAL REPORT

The behavioral report summarizes recurring mobility regularities derived from the user’s training trajectories. It contains the typical starting and ending check-ins on weekdays and weekends, together with frequently visited POIs and their representative visit times. The report supplied to the predictors follows the template below.

During weekday, you usually begin your daily trip at   
<weekday-start-POI> at <HH:MM:SS> and end your daily trip at   
<weekday-end-POI> at <HH:MM:SS>.   
During weekend, you usually begin your daily trip at   
<weekend-start-POI> at <HH:MM:SS> and end your daily trip at   
<weekend-end-POI> at <HH:MM:SS>.   
You usually visit:   
<frequent-POI-1> at <HH:MM:SS>   
<frequent-POI-2> at <HH:MM:SS>

Each POI is represented as <category>#<ID>, where the ID distinguishes venues belonging to the same category.

## A.2 RECENT MOBILITY HISTORY

The mobility history contains up to the 30 most recent check-ins, retained in chronological order, from the three active days immediately preceding the target day:

<POI-1>#<ID> at <HH:MM:SS>   
<POI-2>#<ID> at <HH:MM:SS>   
<POI-K>#<ID> at <HH:MM:SS>

where K ≤ 30.

## B MOBILITY MOTIVATION TAXONOMY

Table 3 summarizes the motivation taxonomy used in our experiments. Each behavioral motivation is deterministically assigned according to the category of the destination venue. The END label is added separately after the final check-in of each trajectory.

Table 3: Mobility motivation taxonomy and its mapping from destination venue categories. The listed venues summarize the category groups associated with each motivation.
<table><tr><td>Motivation</td><td>Behavioral interpretation</td><td>Destination venue categories</td></tr><tr><td>Return</td><td>Returning to a residence or temporary ac- commodation</td><td>Residential venues, hotels, hostels, inns, resorts, and other lodging venues</td></tr><tr><td>Personal Service</td><td>Obtaining a personal, household, financial, repair, or travel-related service</td><td>Salons, banks, repair shops, rental services, post offices, travel agencies, and related service venues</td></tr><tr><td>Entertainment</td><td>Participating in leisure, cultural, or recre- ational activities</td><td>Arts and entertainment, landmarks and out- doors, sports and recreation, cruises, audi-</td></tr><tr><td>Dining</td><td>Eating or drinking</td><td>toriums, and related venues Dining and drinking venues, food courts, and cafeterias</td></tr><tr><td>Transit</td><td>Traveling or transferring between places</td><td>Airports, stations, public transport facili- ties, roads, and other transportation venues</td></tr><tr><td>Religious</td><td>Participating in religious or spiritual activi- ties</td><td>Shrines, temples, churches, mosques, monasteries, and other religious venues</td></tr><tr><td>Education</td><td>Attending educational or learning activi- ties</td><td>Schools, universities, libraries, classrooms laboratories, and training institutions</td></tr><tr><td>Work</td><td>Engaging in occupational or professional activities</td><td>Offices, government facilities, administra- tive buildings, and other professional work-</td></tr><tr><td>Social</td><td>Participating in social, civic, or community activities</td><td>places Event venues, wedding halls, funeral homes, and community or civic venues</td></tr><tr><td>Shopping</td><td>Purchasing goods</td><td>Retail venues and food-and-beverage re- tailers</td></tr><tr><td>Healthcare</td><td>Receiving medical or health-related care</td><td>Hospitals, clinics, pharmacies, and other</td></tr><tr><td>END</td><td>Terminating the daily trajectory</td><td>health and medicine venues Added after the final check-in; not associ- ated with a destination venue</td></tr></table>

## C CALIBRATION OF THE SPEED-FEASIBILITY THRESHOLD

We calibrate the speed threshold using only the observed transitions of the 25 seen users from January 1 to October 31, 2019. After removing one START row from each of the 3,559 training trajectories, the data contain 27,504 consecutive check-in transitions. For a transition from $\left( l _ { i } , t _ { i } \right) \mathrm { t o } \left( l _ { i + 1 } , t _ { i + 1 } \right)$ its implied speed is computed as

$$
v _ { i } = \frac { d _ { \mathrm { h a v } } ( l _ { i } , l _ { i + 1 } ) } { \operatorname* { m a x } ( \Delta q _ { i } , 1 ) \times 1 0 / 6 0 } ,\tag{6}
$$

where $d _ { \mathrm { h a v } }$ is the Haversine distance in kilometers and $\Delta q _ { i }$ is the difference between the corresponding 10-minute time slots. This definition is identical to that used by the speed-feasibility check during inference.

We exclude 14 transitions involving non-spatial states, such as Home (private), and 418 transitions with displacements exceeding 100 km, leaving 27,072 transitions for calibration. The 100 km criterion prevents a small number of extreme long-distance transitions from dominating the threshold estimate and is applied only during calibration, not as a distance constraint during inference.

Because the resulting speed distribution remains strongly right-skewed, we apply the transformation

$$
y _ { i } = \log \left( 1 + { \frac { v _ { i } } { v _ { 0 } } } \right) , \qquad v _ { 0 } = 1 \mathrm { k m / h } ,\tag{7}
$$

which retains zero-speed transitions while compressing the upper tail.

As shown in Figure 3, the transformation substantially reduces the skewness of the speed distribution, although the Q–Q plot still shows deviations from exact Gaussianity.

$$
\mu _ { y } = 1 . 4 9 2 7 , \qquad \sigma _ { y } = 1 . 2 1 9 6 ,\tag{8}
$$

![](images/d52b16e869ba18830c80a1287f128cb01f31666e269f5b39c3cd1f3de3f0fc02.jpg)

![](images/ca72564c17f5fab0c71511d729027c6b998993ce787310f02f63b1f3477476cc.jpg)  
Figure 3: Distributional diagnostics for the log-transformed transition speeds used to calibrate $v _ { \mathrm { m a x } }$ (a) Empirical distribution and a Gaussian distribution with the same mean and standard deviation. The red line marks the two-standard-deviation upper bound. (b) Normal $\mathrm { Q - Q }$ plot showing the remaining deviations from Gaussianity, particularly in the tails.

giving the following upper bound on the original speed scale:

$$
v _ { 0 } \left[ \exp ( \mu _ { y } + 2 \sigma _ { y } ) - 1 \right] = 5 0 . 0 0 3 \mathrm { k m / h } .\tag{9}
$$

This value closely matches the empirical 97.5th percentile,

$$
Q _ { 0 . 9 7 5 } ( \{ v _ { i } \} ) = 4 9 . 5 6 8 \mathrm { k m / h } .\tag{10}
$$

We therefore set $v _ { \mathrm { m a x } } = 5 0$ km/h, which covers 97.606% of the cleaned training transitions. The empirical percentile provides the primary calibration criterion, while the log-space two-standarddeviation bound serves as a consistent distributional check. The resulting threshold is fixed across all user groups and temporal periods without using any unseen-user or unseen-period data.

## D IMPLEMENTATION DETAILS AND CONFIGURATIONS

We use Llama-3.1-8B-Instruct as the base model for the LLM judge, the state predictor $P _ { S }$ , and the motivation predictor $P _ { M }$ . The LLM judge remains frozen during alternative transition construction, whereas $P _ { S }$ and $P _ { M }$ are independently adapted from the base model using QLoRA. This appendix provides implementation details and configurations for three stages: alternative transition construction, predictor training, and autoregressive trajectory inference.

## D.1 ALTERNATIVE TRANSITION CONSTRUCTION

Table 4 summarizes the data, candidate-generator, and LLM-judge configurations used to construct alternative transitions. For each observed transition, we sample three distinct motivations other than the observed one. The candidate generator first proposes possible next states for each alternative motivation, after which the LLM judge selects the most plausible candidate under the user’s behavioral context.

The judge receives the target date and weekend indicator, the user’s long-term behavioral report, recent mobility history, earlier check-ins on the target day, the current location and time, the alternative motivation, and the candidate states. Each candidate is represented by its location, coordinates, arrival time, activity category, distance from the current state, and candidate-generator score. The judge is instructed to return only the index of the candidate that is most consistent with temporal validity, spatial feasibility, the specified motivation, and the user’s historical behavior.

## D.2 PREDICTOR TRAINING

The state predictor and motivation predictor are trained independently. The state predictor is trained on a balanced mixture of observed and judged alternative transitions, preventing the larger set of

Table 4: Configuration for alternative-transition construction.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Construction data</td><td>25 seen users from January–October 2019, containing 3,559 daily trajectories and 31,063 observed transition instances</td></tr><tr><td>Alternative motivations</td><td>Three distinct motivations per observed transition, sampled without replacement while excluding the observed motivation</td></tr><tr><td>Prepared alternative cases</td><td>93,189 cases before candidate generation and judging; 31,063 judged alternative transitions are used for balanced state-predictor training</td></tr><tr><td>Candidate-generator architecture</td><td>Three-layer pre-layer-normalized Transformer encoder with hidden dimension 128, 4 attention heads, feed-forward dimension 256, and dropout 0.1</td></tr><tr><td>Prediction heads</td><td>Separate classification heads for location subcategory, spatial grid, and arrival-time bin</td></tr><tr><td>Candidate-generator objective</td><td>Equal-weighted cross-entropy losses for the subcategory, spatial-grid, and time-bin prediction heads</td></tr><tr><td>Candidate-generator split</td><td>90% training and 10% validation dates within each user, split by complete user-date groups with seed 42</td></tr><tr><td>Candidate-generator optimization</td><td>AdamW with learning rate  $1 \times 1 0 ^ { - 3 }$  , weight decay  $1 \times 1 0 ^ { - 4 }$  , batch size 2,048, and 30 training epochs</td></tr><tr><td>Learning-rate schedule</td><td>Two warm-up epochs followed by cosine decay with a minimum learning-rate ratio of 0.05</td></tr><tr><td>Additional regularization</td><td>Gradient clipping at 1.0 and label smoothing of 0.05</td></tr><tr><td>Checkpoint selection</td><td>The checkpoint with the highest mean top-1 validation accuracy over the subcategory, spatial-grid, and time-bin heads</td></tr><tr><td>Candidate enumeration</td><td>Top 8 subcategories, top 8 spatial grids, and top 8 time bins are combined into at most 50 preliminary triples, from which at most 10 candidates are</td></tr><tr><td>Candidate-scoring weights</td><td>retained for judging  $\alpha _ { \mathrm { s u b } } = 0 . 8 , \alpha _ { \mathrm { g r i d } } = 0 . 9 , \alpha _ { \mathrm { t i m e } } = 0 . 5 5 , \alpha _ { \mathrm { u s e r } } = 1 . 0$ </td></tr><tr><td>Candidate filtering</td><td>Motivation-strict filtering and behavioral-report-based distance capping are disabled; equal-time alternative transitions are not allowed</td></tr><tr><td>Candidate context</td><td>The previous three active mobility days, the preceding check-ins on the target day, the current state, the alternative motivation, and the user&#x27;s</td></tr><tr><td>LLM judge</td><td>long-term behavioral report Frozen Llama-3.1-8B-Instruct loaded with 4-bit NF4 quantization, double</td></tr><tr><td>Judge decoding</td><td>quantization, and bfloat16 computation when supported Greedy decoding with at most 8 newly generated tokens</td></tr><tr><td>Judge decision</td><td>The judge must select one candidate when the candidate set is nonempty. An invalid or unparsable output defaults to the highest-ranked candidate; no alternative transition is retained when the candidate set is empty</td></tr><tr><td>ρ</td><td>0.33</td></tr><tr><td>Construction seed</td><td>3407</td></tr></table>

initially constructed alternatives from dominating its training objective. The motivation predictor is trained only on observed step-level motivations and trajectory-termination labels. Table 5 reports their training configurations.

Table 5: Training configurations for the state predictor $P _ { S }$ and motivation predictor $P _ { M }$
<table><tr><td>Parameter</td><td>State predictor  $P _ { S }$ </td><td>Motivation predictor  $P _ { M }$ </td></tr><tr><td>Initialization</td><td>Llama-3.1-8B-Instruct</td><td>Independently initialized from Llama-3.1-8B-Instruct</td></tr><tr><td>Training data</td><td>31,063 observed transitions and 31,063 judged alternative transitions</td><td>34,622 observed instances, comprising 31,063 step-level motivations and 3,559 END labels</td></tr><tr><td>Observed-to- alternative ratio</td><td>1:1</td><td>Not applicable</td></tr><tr><td>Training objective</td><td>Assistant-token-only causal language-modeling loss</td><td>Assistant-token-only causal language-modeling loss</td></tr><tr><td>Quantization</td><td>4-bit NF4 with double quantization and bfloat16 computation</td><td>4-bit NF4 with double quantization and bfloat16 computation</td></tr><tr><td>QLoRA configuration</td><td>Rank r = 16, scaling factor α = 16, dropout 0, and no bias</td><td>Rank r = 16, scaling factor α = 16, dropout 0, and no bias</td></tr><tr><td>LoRA target modules</td><td>q-proj, k-proj, v-proj, o-proj, gate-proj, up-proj, and</td><td>q-proj, k-proj, v-proj, o-proj, gate-proj, up-proj, and</td></tr><tr><td>Optimizer</td><td>down-proj 8-bit paged AdamW</td><td>down-proj 8-bit paged AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td>0.01</td><td>0.01</td></tr><tr><td>Learning-rate schedule</td><td>Linear decay with 200 warm-up steps</td><td>Linear decay with 200 warm-up steps</td></tr><tr><td>Batch size</td><td>2 per GPU with 8 gradient-accumulation 2 per GPU with 8 gradient-accumulation steps on 8 GPUs, giving an effective batch size of 128</td><td>steps on 8 GPUs, giving an effective batch size of 128</td></tr><tr><td>Training budget</td><td>4,000 optimization steps</td><td>12 epochs; the checkpoint at step 3,084 is used for inference</td></tr><tr><td>Maximum sequence length</td><td>2,048 tokens</td><td>2,048 tokens</td></tr><tr><td>Mobility-history context</td><td>Up to 30 check-ins from the three active days preceding the target day</td><td>Up to 30 check-ins from the three active days preceding the target day</td></tr><tr><td>Training seed</td><td>3407</td><td>3407</td></tr></table>

## D.3 TRAJECTORY INFERENCE

Trajectory generation starts from a START state and alternates between motivation prediction and state prediction. The motivation predictor either produces the next behavioral motivation or returns END. Given a non-terminal motivation, the state predictor generates a candidate location and arrival time. The candidate is then validated before being appended to the trajectory. Table 6 summarizes the inference configuration.

Table 6: Configuration for autoregressive trajectory inference.
<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Initial state</td><td>A START state; the first motivation and check-in are generated without applying a travel-speed constraint because no previous spatial state is available</td></tr><tr><td>Mobility-history context</td><td>Up to 30 check-ins from the three active days preceding the target day</td></tr><tr><td>Motivation decoding</td><td>Greedy decoding with at most 8 newly generated tokens</td></tr><tr><td>State decoding</td><td>Sampling with temperature 0.2, top-p 0.9, and at most 32 newly generated tokens</td></tr><tr><td>State-generation attempts</td><td>At most 8 proposals per step, consisting of one initial proposal and up to seven retries</td></tr><tr><td>Retry decoding</td><td>Temperature increases from 0.3 to 0.9 in increments of 0.1; top-p remains 0.9 and the maximum output length remains 32 tokens</td></tr><tr><td>Output validation</td><td>The generated output must be parseable as a valid location-time state with a recognized location identifier and timestamp</td></tr><tr><td>Temporal validation</td><td>The generated arrival time must not precede the current time; generation terminates when the predicted state crosses into the next mobility day</td></tr><tr><td>Travel-speed calculation</td><td>Haversine distance divided by the elapsed time between consecutive states, with time represented using 10-minute slots</td></tr><tr><td>Speed threshold</td><td>vmax = 50 km/h; its selection is described in Appendix C</td></tr><tr><td>Speed-based retry</td><td>A proposal exceeding vmax is rejected and regenerated using the retry schedule above</td></tr><tr><td>Re-grounding after retries</td><td>If all eight proposals violate the speed constraint, the final proposal is re-grounded, when possible, to a location of the same broad activity category that appears in the user&#x27;s history or behavioral report and satisfies vmax; feasible locations are ranked by time-of-day similarity with historical frequency used as a tie-breaker</td></tr><tr><td>Re-grounding failure</td><td>If no feasible location is available for re-grounding, the final state-predictor proposal is retained</td></tr><tr><td>Additional fallbacks</td><td>The legacy distance-based fallback and history-trajectory replacement fallback are disabled</td></tr><tr><td>Repetition termination</td><td>Activated after 15 accepted check-ins. Generation terminates before appending a candidate if the candidate-augmented location sequence contains a block of length 1–4 repeated for at least three consecutive rounds</td></tr><tr><td>Online repetition guard</td><td>Disabled; repetition is handled only by the termination rule above</td></tr></table>

## E PROMPT TEMPLATES

All prompts are rendered using the chat template of Llama-3.1-8B-Instruct. Text enclosed by angle brackets below denotes sample-specific content.

## E.1 LLM JUDGE FOR ALTERNATIVE TRANSITION CONSTRUCTION

For each alternative motivation, the lightweight Transformer produces a candidate set of next states.   
The LLM judge selects one candidate without generating a new location or timestamp.

## System prompt

You are a careful urban mobility candidate judge.   
You must choose exactly one candidate next check-in from a provided candidate list.   
Do not invent a new place or time.

## User prompt

<INPUT 0> Behavior Pattern:

<BEHAVIORAL\_REPORT>   
<INPUT 1> Recent History Before Today:   
<RECENT\_HISTORY\_OR\_NONE>   
<INPUT 2> Earlier Check-ins Today:   
<EARLIER\_CHECKINS\_OR\_NONE>   
<INPUT 3> Current Location & Time:   
<CURRENT\_STATE> [lat=<LATITUDE>, lng=<LONGITUDE>]   
<INPUT 4> Target Motivation for the NEXT transition:   
<ALTERNATIVE\_MOTIVATION>   
<INPUT 5> Date and Weekend Flag:   
date=<YYYY-MM-DD>, is\_weekend=<0\_OR\_1>   
<CANDIDATES>   
1. <LOCATION>#<ID> at <HH:MM:SS>   
[transformer\_score=<SCORE>;   
top\_category=<CATEGORY>;   
candidate\_motivation=<MOTIVATION>;   
lat=<LATITUDE>, lng=<LONGITUDE>;   
distance\_from\_current\_km=<DISTANCE>]   
K. <LOCATION>#<ID> at <HH:MM:SS> [<METADATA>]   
Task:   
Choose the single best candidate for the next check-in. Prefer candidates   
that fit the target motivation, the user’s recent behavior, the current   
state, and temporal order.   
Output format:   
Output ONLY the candidate number, such as 1 or 7. Do not output any other text.

## E.2 STATE PREDICTOR

The same state-prediction prompt is used for observed and alternative transitions during training. At inference time, the target motivation is the output of $P _ { M }$

## System prompt

You are an urban mobility transition model.   
Given a behavior pattern, recent history checkins, current location and   
time, current motivation, today’s date, and is\_weekend, predict the next   
location and time.   
Follow the required output format exactly.

## User prompt

<INPUT 0> (Behavior Pattern - long-term prior):   
<BEHAVIORAL\_REPORT>   
<INPUT 1> (Recent History Checkins):   
<RECENT\_HISTORY>   
<INPUT 2> (Current Location & Time):   
<CURRENT\_STATE>   
<INPUT 3> (Target Motivation for the NEXT transition):   
<MOTIVATION>   
<INPUT 4> (Today’s Date):   
<YYYY-MM-DD>   
<INPUT 5> (Is Weekend, 1=yes 0=no):   
<0\_OR\_1>   
Task:   
Predict the NEXT location and time.   
Rules:   
- Recent history and current state are more important than the behavior pattern.   
- Use the behavior pattern only as a general long-term reference.   
- The next timestamp must not be earlier than the current timestamp under   
the day definition starting at 04:00:00.

Output format (IMPORTANT):   
<Location>#<ID> at HH:MM:SS   
Do NOT output any extra text.   
Example:   
Park#12 at 09:10:00

## E.3 MOTIVATION PREDICTOR

The motivation predictor does not observe the next state or any subsequent state. It predicts either the motivation underlying the next transition or the termination label END.

## System prompt

You are an urban mobility intention inference model.   
Given a behavior pattern, recent history Checkins, the current location   
and time, today’s date, and is\_weekend, infer the motivation for the next   
transition from the current state.   
Output exactly one label from the provided list.

## User prompt

<INPUT 0> (Behavior Pattern - long-term prior):   
<BEHAVIORAL\_REPORT>   
<INPUT 1> (Recent History Checkins):   
<RECENT\_HISTORY>   
<INPUT 2> (Current Location & Time):   
<CURRENT\_STATE>   
<INPUT 3> (Today’s Date):   
<YYYY-MM-DD>   
<INPUT 4> (Is Weekend, 1=yes 0=no):   
<0\_OR\_1>   
Task:   
Infer the motivation for the next transition.   
Rules:   
- Recent history is more important than the behavior pattern.   
Allowed labels (choose exactly ONE):   
- Return   
Personal Service   
Entertainment   
Dining   
Transit   
Religious   
Education   
Work   
Social   
Shopping   
Healthcare   
END   
Output format (IMPORTANT):   
Output EXACTLY ONE line.   
The line MUST be exactly one of the allowed labels.   
Do NOT output any extra text.

## F MOTIVATION-GUIDED TRAJECTORY GENERATION PSEUDO CODES

This section presents the pseudocode for the complete inference-time rollout, including feasibilityaware resampling, repetition-based termination, and mobility-grounded recovery.

Algorithm 1 Motivation-guided trajectory generation   
Require: Models $P _ { M }$ and $P _ { S } ;$ user context $x _ { u } ;$ observed locations $\mathcal { O } _ { u }$   
Require: $v _ { \mathrm { m a x } } = 5 0$ km/h; $R = 8 ; L _ { \mathrm { m i n } } = 1 5 ; L _ { \mathrm { m a x } } = 2 5$   
Ensure: Generated trajectory $\tau$   
1: $\tau  [ ]$   
2: c ← BUILDCONTEXT(x , START, T)   
3: for $i = 0 , \ldots , L _ { \mathrm { m a x } } - 1$ do   
4: mˆ <sub>i</sub> ← arg max p<sub>θM</sub> (y | c<sub>i</sub>)   
y∈M∪{END}   
5: if $\hat { m } _ { i } = \mathbb { E } \mathbb { N } \mathbb { D }$ then   
6: return $\tau$   
7: end if   
8: $s ^ { \star }  \emptyset$   
9: $\tilde { s } ^ { \mathrm { l a s t } }  \emptyset$   
10: for $r = 1 , \ldots , R$ do   
11: if $r = 1$ then   
12: $T _ { r } \gets 0 . 2$   
13: else   
14: $T _ { r } \gets 0 . 3 + 0 . 6 ( r - 2 ) / ( R - 2 )$   
15: end if   
16: $\tilde { s } \sim p _ { \theta _ { S } } ^ { ( T _ { r } , ~ p = 0 . 9 ) } ( \cdot \mid c _ { i } , \hat { m } _ { i } )$   
17: s˜ ← PARSESTATE(˜s)   
18: if $\tilde { s } = \emptyset$ then   
19: continue   
20: end if   
21: $\tilde { s } ^ { \mathrm { l a s t } } \gets \tilde { s }$   
22: if $\cdot _ { i } > 0$ and ARRIVALTIME(˜s) < ARRIVALTIME $\left( { { s _ { i - 1 } } } \right)$ then   
23: continue   
24: end if   
25: if $i > 0$ and IMPLIEDSPEED $( s _ { i - 1 } , \tilde { s } ) > v _ { \mathrm { m a x } }$ then   
26: continue   
27: end if   
28: $s ^ { \star }  \tilde { s }$   
29: break   
30: end for   
31: if $s ^ { \star } = \emptyset$ then   
32: if $\tilde { s } ^ { \mathrm { l a s t } } = \emptyset$ then   
33: return $\tau$   
34: end if   
35: $s ^ { \star } \gets \mathrm { R E G R O U N D } \big ( \tilde { s } ^ { \mathrm { l a s t } } , s _ { i - 1 } , \mathcal { O } _ { u } , v _ { \mathrm { m a x } } \big )$   
36: if $s ^ { \star } = \emptyset$ then   
37: return $\tau$   
38: end if   
39: end if   
40: $\mathcal { L } ^ { + }  \mathrm { L }$ OCATIONS(T ∥ [s<sup>⋆</sup>])   
41: if $| \mathcal { T } | \geq L _ { \operatorname* { m i n } }$ and REPEATEDCYCLE(L<sup>+</sup>, 4, 3, {1, 2, 3, 4}) then   
42: return $\tau$   
43: end if   
44: $\mathcal { T }  \mathcal { T } \parallel [ s ^ { \star } ]$   
45: $s _ { i }  s ^ { \star }$   
46: c<sub>i+1</sub> ← BUILDCONTEXT $( x _ { u } , s _ { i } , T )$   
47: end for   
48: return T

Algorithm 2 Repetition-aware termination   
1: function REPEATEDCYCLE(L, B, K, P)   
2: n ← |L|   
3: for b = 1, . . . , min $\{ B , \lfloor n / K \rfloor \}$ do   
4: for $a = 1 , \ldots , n - K b + 1$ do   
5: W ← L[a : a + Kb − 1]   
6: $Q  W [ 1 : b ]$   
7: if $W = \dot { Q } ^ { K }$ and PRIMITIVEPERIOD(W) ∈ P then   
8: return true   
9: end if   
10: end for   
11: end for   
12: return false   
13: end function

Algorithm 3 Mobility-grounded recovery after retry exhaustion   
1: function REGROUND(s, s ˜ <sub>prev</sub>, O<sub>u</sub>, v<sub>max</sub>)   
2: (<sup>˜</sup>ℓ, t<sup>˜</sup>) ← SPLITSTATE(˜s)   
3: a ← ACTIVITYCATEGORY(<sup>˜</sup>ℓ)   
4: C ← OBSERVEDCANDIDATES(O , a)   
5: F ← ∅   
6: for all $( \ell , t ^ { \mathrm { o b s } } ) \in \mathcal { C }$ do   
7: $s ^ { \prime } \gets ( \ell , \tilde { t } )$   
8: if $s _ { \mathrm { p r e v } } = \mathbf { S } \mathrm { T A R T }$ or IMPLIEDSPEED(s<sub>prev</sub>, s<sup>′</sup>) ≤ v<sub>max</sub> then   
9: ${ \bf \dot { \mathcal { F } } } \gets \mathcal { F } \cup \{ ( \ell , t ^ { \mathrm { o b s } } ) \}$   
10: end if   
11: end for   
12: if F = ∅ then   
13: return ∅   
14: end if   
15: (ℓ<sup>⋆</sup>, t<sup>⋆</sup>) ← arg min  <sup></sup><sub></sub>TIMESLOT(t<sup>obs</sup>) − TIMESLOT(t<sup>˜</sup>)<sup></sup><sub></sub> , −VISITFREQUENCY(u, ℓ)   
(ℓ,t<sup>obs</sup>)∈F   
16: return (ℓ<sup>⋆</sup>, t<sup>˜</sup>)   
17: end function

## G EVALUATION-SPLIT CONSTRUCTION AND LEAKAGE PREVENTION

We distinguish three roles of mobility trajectories in our experiments. Parameter-training trajectories are used to optimize the candidate generator, the state predictor $P _ { S }$ , and the motivation predictor $P _ { M }$ . Support trajectories are used only to construct user-specific context at inference time. Query trajectories are reserved exclusively as evaluation targets. In particular, support trajectories from unseen users never participate in parameter optimization.

## G.1 SEEN USERS IN THE SEEN PERIOD

For each of the 25 seen users, we randomly divide the daily trajectories collected between January and October 2019 into disjoint training and query subsets using an approximately 80/20 ratio. After peruser rounding, this produces 3,559 parameter-training trajectories and 908 held-out query trajectories. We verify that the two subsets contain no common (user, date) pair.

Both $P _ { S }$ and $P _ { M }$ are trained exclusively from the 3,559 training trajectories. Each trajectory contributes one END instance to the training data of $P _ { M } ;$ ; therefore, the 3,559 END instances reported in Table 5 correspond exactly to the parameter-training trajectories and do not include any of the 908 evaluation trajectories.

The candidate generator further divides the same 3,559 trajectories into 90% training dates and 10% validation dates for checkpoint selection. This is an internal training/validation division and is independent of the 908-trajectory final evaluation set. We therefore treat the seen-user, seen-period setting as an in-distribution held-out evaluation rather than a user- or time-shift generalization setting.

## G.2 INFERENCE CONTEXT FOR UNSEEN USERS

The 50 unseen users are disjoint from the 25 seen users, and none of their trajectories is used to optimize the candidate generator, $P _ { S }$ , or $P _ { M }$ . For the unseen-user, seen-period evaluation, their January–October 2019 trajectories are divided into 5,431 support trajectories and 1,345 held-out query trajectories. These subsets contain no common (user, date) pair.

For an unseen user $u ,$ the behavioral report $R _ { u }$ is constructed only from that user’s support trajectories. For a query trajectory $\tau _ { u , d } ,$ the recent mobility history $H _ { u , d }$ contains up to 30 check-ins from the three most recent active support days strictly preceding the target date $\dot { d } .$ The target trajectory $\tau _ { u , d }$ and all query trajectories are excluded from both report and history construction.

Thus, “unseen user” means unseen during parameter learning, rather than a zero-history coldstart setting. The corresponding experiment evaluates generalization to new users given a limited, inference-only mobility context.

## G.3 CONTEXT USED IN THE TEMPORAL-SHIFT SETTINGS

For the regular unseen period, spanning November 2019 through February 2020, the behavioral report is constructed from January–October 2019 support trajectories. The report window therefore precedes the complete evaluation period. The recent history $H _ { u , d }$ is selected from a support pool disjoint from the query trajectories and contains only observations strictly preceding d.

For the COVID-19 period, spanning April through October 2020, the behavioral report is updated using only mobility observations from March 2020. The recent history is again restricted to support trajectories strictly preceding the target date. Consequently, neither the target-day trajectory nor a future trajectory enters $R _ { u }$ or $H _ { u , d }$ in either temporal-shift evaluation.

The two seen-period settings use an IID trajectory holdout rather than a chronological forecasting split. In these settings, $R _ { u }$ is a static profile constructed from the user’s complete January–October support subset. Accordingly, it may summarize a support trajectory dated after a particular query date, but it never contains the query trajectory itself or any other trajectory from the query subset. We therefore do not interpret the seen-period results as causal future forecasting. By contrast, $H _ { u , d }$ is always restricted to support observations preceding d.

Table 7: Data separation and information available under the evaluation protocol.
<table><tr><td>Protocol component</td><td>Data usage and restriction</td></tr><tr><td>Predictor training</td><td>Trajectories from 25 seen users between January and October 2019. For the seen-user/seen-period evaluation, 20% of user-date trajectories are held out and excluded from predictor training.</td></tr><tr><td>User-level shift</td><td>The 50 unseen users are disjoint from the 25 users used to train  $P _ { M }$  and  $P _ { S } .$  The regular unseen period (November 2019–February 2020) and the COVID-19</td></tr><tr><td>Temporal shift</td><td>period (April–October 2020) do not overlap with the predictor-training period.</td></tr><tr><td>Recent history  $\mathcal { H } _ { u , d }$ </td><td>Contains at most 30 check-ins from the three active days strictly preceding target date d. Target-day and future observations are never included.</td></tr><tr><td>Behavioral report  $R _ { u }$ </td><td>In the temporal-shift settings, the report is constructed only from observations preceding the evaluation period. In the seen-period settings, it is a static aggre- gate profile constructed from the query-disjoint support subset and may include support dates later than the target date.</td></tr><tr><td>COVID-19 update</td><td>For targets in April–October 2020, the report is updated using observations available through March 2020, without using any check-ins from the target period.</td></tr><tr><td>Re-grounding</td><td>Candidate locations are drawn only from the same  $R _ { u }$  and  $H _ { u , d }$  provided to the predictors. Re-grounding does not access the query trajectory or any additional trajectory database.</td></tr></table>

## G.4 INFORMATION AVAILABLE TO RE-GROUNDING

The inference-time re-grounding procedure does not access the complete trajectory database or the ground-truth query trajectory. Its candidate pool contains only POI–time pairs extracted from the same behavioral report $R _ { u }$ and recent mobility history $H _ { u , d }$ supplied to $P _ { S }$ and $P _ { M }$ . Re-grounding therefore introduces no additional user information beyond the context already visible to the two predictors.

The two seen-period settings use non-chronological held-out splits under the observed-period distribution. In both settings, $R _ { u }$ is constructed exclusively from the user’s support subset, but it may aggregate behavior from a support trajectory dated after a particular query date. Importantly, $R _ { u }$ contains only aggregate behavioral regularities—such as typical starting and ending activities and frequently visited POIs with representative visit times—rather than any complete day-level trajectory, and it never contains the target query trajectory or any other trajectory from the query subset. We therefore interpret the seen-user/seen-period setting as an in-distribution held-out evaluation and the unseen-user/seen-period setting as a user-level transfer evaluation under the observed-period distribution; neither is intended as causal future forecasting. By contrast, $H _ { u , d }$ is always restricted to support observations strictly preceding d.

## H OSAKA RESULTS

We report the evaluation results on the Osaka human check-in dataset, following the same user-level and temporal splitting protocols used for the Tokyo dataset. To evaluate user-level generalization, we use trajectories from 25 seen users to train the base model and reserve a disjoint set of 50 additional users exclusively for evaluation. No mobility records from these unseen users are available during training. To evaluate temporal generalization, check-ins collected from January to October 2019 constitute the seen training period. We further consider two unseen periods: November 2019 to February 2020, which captures a regular seasonal shift, and April to October 2020, which captures the pandemic-induced mobility shift following the onset of COVID-19 restrictions.

Across all 24 combinations of user group, time period, and metric, MotiveMob consistently achieves the best performance. For seen users, it reduces the four divergences by an average of 36.3% relative to the corresponding runner-up, with particularly large gains on SD, SI, and DARD. The improvements on STVD are smaller (0.7–2.8%), but remain consistent across all three periods. For unseen users, MotiveMob retains the best results in every setting and yields an average relative error reduction of 31.1% over the runner-up. Although its advantages on SI and STVD narrow under the

Table 8: Evaluation results on seen users in Osaka across seen and unseen time periods. The upper and lower baseline sections correspond to pretraining-based and prompting-based methods, respectively. Lower values are better; the best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="4">Seen period (1186 trajectories)</td><td colspan="4">Regular unseen period (466 trajectories)</td><td colspan="4">COVID-19 unseen period (689 trajectories)</td></tr><tr><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td></tr><tr><td>LSTM</td><td>0.0914</td><td>0.1375</td><td>0.2673</td><td>0.4161</td><td>0.1109</td><td>0.1698</td><td>0.3230</td><td>0.5779</td><td>0.1259</td><td>0.2073</td><td>0.3556</td><td>0.5725</td></tr><tr><td>MHSA</td><td>0.0908</td><td>0.1357</td><td>0.2468</td><td>0.4286</td><td>0.1210</td><td>0.1817</td><td>0.2706</td><td>0.5803</td><td>0.1262</td><td>0.2152</td><td>0.3044</td><td>0.5552</td></tr><tr><td>DeepMove</td><td>0.1777</td><td>0.1024</td><td>0.2348</td><td>0.4933</td><td>0.1645</td><td>0.1267</td><td>0.3180</td><td>0.6065</td><td>0.2308</td><td>0.1595</td><td>0.3340</td><td>0.6029</td></tr><tr><td>GetNext</td><td>0.0395</td><td>0.2820</td><td>0.2149</td><td>0.4635</td><td>0.0433</td><td>0.2834</td><td>0.2057</td><td>0.5601</td><td>0.0455</td><td>0.2952</td><td>0.1953</td><td>0.5294</td></tr><tr><td>TrajGAIL</td><td>0.2251</td><td>0.1256</td><td>0.0799</td><td>0.5836</td><td>0.2458</td><td>0.1227</td><td>0.1250</td><td>0.6576</td><td>0.2494</td><td>0.1118</td><td>0.1137</td><td>0.6424</td></tr><tr><td>DiffTraj</td><td>0.2002</td><td>0.1217</td><td>0.1407</td><td>0.6874</td><td>0.2092</td><td>0.1165</td><td>0.1662</td><td>0.6924</td><td>0.1869</td><td>0.1025</td><td>0.1907</td><td>0.6904</td></tr><tr><td>GNPR-SID</td><td>0.1121</td><td>0.0212</td><td>0.0988</td><td>0.4642</td><td>0.1319</td><td>0.0346</td><td>0.1262</td><td>0.5786</td><td>0.1889</td><td>0.0434</td><td>0.1711</td><td>0.5956</td></tr><tr><td>LLM4POI</td><td>0.0323</td><td>0.0192</td><td>0.0900</td><td>0.4093</td><td>0.0463</td><td>0.0340</td><td>0.1309</td><td>0.5408</td><td>0.0424</td><td>0.0512</td><td>0.1278</td><td>0.5139</td></tr><tr><td>LLMMove</td><td>0.0440</td><td>0.0453</td><td>0.1102</td><td>0.5022</td><td>0.0528</td><td>0.0518</td><td>0.1866</td><td>0.5907</td><td>0.0455</td><td>0.0630</td><td>0.1556</td><td>0.5856</td></tr><tr><td>LLM-ZS</td><td>0.0352</td><td>0.0521</td><td>0.1881</td><td>0.5124</td><td>0.0300</td><td>0.0534</td><td>0.2272</td><td>0.5880</td><td>0.0274</td><td>0.0648</td><td>0.1958</td><td>0.5596</td></tr><tr><td>LLM-Mob</td><td>0.0338</td><td>0.0554</td><td>0.2117</td><td>0.4977</td><td>0.0309</td><td>0.0615</td><td>0.2741</td><td>0.5901</td><td>0.0382</td><td>0.0656</td><td>0.2819</td><td>0.5777</td></tr><tr><td>LLMob</td><td>0.1153</td><td>0.2542</td><td>0.1435</td><td>0.5276</td><td>0.0537</td><td>0.0980</td><td>0.1334</td><td>0.5827</td><td>0.0530</td><td>0.0858</td><td>0.1225</td><td>0.5433</td></tr><tr><td>ELLMob</td><td>0.0160</td><td>0.0274</td><td>0.0939</td><td>0.4127</td><td>0.0205</td><td>0.0281</td><td>0.1496</td><td>0.5302</td><td>0.0209</td><td>0.0334</td><td>0.1442</td><td>0.5405</td></tr><tr><td>MotiveMob</td><td>0.0038</td><td>0.0078</td><td>0.0340</td><td>0.4035</td><td>0.0097</td><td>0.0165</td><td>0.0884</td><td>0.5153</td><td>0.0125</td><td>0.0207</td><td>0.0723</td><td>0.5102</td></tr></table>

Table 9: Evaluation results on unseen users in Osaka across seen and unseen time periods. The upper and lower baseline sections correspond to pretraining-based and prompting-based methods, respectively. Lower values are better; the best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="4">Seen period (2804 trajectories)</td><td colspan="4">Regular unseen period (1655 trajectories)</td><td colspan="4">COVID-19 unseen period (1301 trajectories)</td></tr><tr><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td></tr><tr><td>LSTM</td><td>0.1489</td><td>0.2437</td><td>0.3872</td><td>0.6207</td><td>0.1740</td><td>0.2420</td><td>0.3810</td><td>0.6569</td><td>0.1493</td><td>0.2645</td><td>0.4358</td><td>0.6551</td></tr><tr><td>MHSA</td><td>0.1897</td><td>0.1965</td><td>0.3578</td><td>0.5961</td><td>0.1715</td><td>0.2352</td><td>0.3762</td><td>0.6466</td><td>0.1582</td><td>0.2500</td><td>0.4182</td><td>0.6589</td></tr><tr><td>DeepMove</td><td>0.2754</td><td>0.2267</td><td>0.3548</td><td>0.6510</td><td>0.2746</td><td>0.2257</td><td>0.3876</td><td>0.6714</td><td>0.2709</td><td>0.2338</td><td>0.3918</td><td>0.6704</td></tr><tr><td>GetNext</td><td>0.0832</td><td>0.3220</td><td>0.2568</td><td>0.5752</td><td>0.0752</td><td>0.3215</td><td>0.2376</td><td>0.6408</td><td>0.0994</td><td>0.3351</td><td>0.2511</td><td>0.6438</td></tr><tr><td>TrajGAIL</td><td>0.1737</td><td>0.0825</td><td>0.1287</td><td>0.5750</td><td>0.1954</td><td>0.0912</td><td>0.1578</td><td>0.6513</td><td>0.1915</td><td>0.0943</td><td>0.1427</td><td>0.6532</td></tr><tr><td>DiffTraj</td><td>0.1977</td><td>0.0719</td><td>0.0932</td><td>0.6894</td><td>0.1944</td><td>0.0832</td><td>0.1283</td><td>0.6919</td><td>0.2159</td><td>0.0815</td><td>0.1405</td><td>0.6899</td></tr><tr><td>GNPR-SID</td><td>0.1367</td><td>0.0706</td><td>0.1290</td><td>0.4673</td><td>0.1353</td><td>0.0626</td><td>0.1272</td><td>0.5632</td><td>0.1481</td><td>0.0643</td><td>0.1325</td><td>0.5925</td></tr><tr><td>LLM4POI</td><td>0.0334</td><td>0.0699</td><td>0.1282</td><td>0.4435</td><td>0.0382</td><td>0.0763</td><td>0.1477</td><td>0.5682</td><td>0.0535</td><td>0.0859</td><td>0.1595</td><td>0.5663</td></tr><tr><td>LLMMove</td><td>0.0467</td><td>0.0602</td><td>0.1255</td><td>0.4674</td><td>0.0504</td><td>0.0663</td><td>0.1608</td><td>0.5834</td><td>0.0427</td><td>0.0737</td><td>0.1615</td><td>0.5831</td></tr><tr><td>LLM-ZS</td><td>0.0226</td><td>0.0709</td><td>0.1762</td><td>0.5005</td><td>0.0345</td><td>0.0909</td><td>0.2069</td><td>0.5759</td><td>0.0328</td><td>0.0875</td><td>0.1733</td><td>0.5910</td></tr><tr><td>LLM-Mob</td><td>0.0734</td><td>0.0625</td><td>0.2746</td><td>0.4667</td><td>0.0715</td><td>0.0576</td><td>0.2992</td><td>0.5824</td><td>0.0821</td><td>0.0761</td><td>0.3141</td><td>0.5747</td></tr><tr><td>LLMob</td><td>0.0583</td><td>0.0948</td><td>0.0902</td><td>0.4756</td><td>0.0702</td><td>0.1130</td><td>0.1337</td><td>0.5766</td><td>0.0623</td><td>0.1243</td><td>0.1415</td><td>0.5865</td></tr><tr><td>ELLMob</td><td>0.0227</td><td>0.0366</td><td>0.1226</td><td>0.4255</td><td>0.0260</td><td>0.0383</td><td>0.1820</td><td>0.5467</td><td>0.0264</td><td>0.0408</td><td>0.1657</td><td>0.5514</td></tr><tr><td>MotiveMob</td><td>0.0075</td><td>0.0251</td><td>0.0363</td><td>0.3872</td><td>0.0143</td><td>0.0268</td><td>0.0696</td><td>0.5149</td><td>0.0214</td><td>0.0385</td><td>0.0665</td><td>0.5219</td></tr></table>

COVID-19 shift, MotiveMob remains strongest overall, demonstrating robust generalization to both unseen users and temporal distribution shifts. Beyond user groups and temporal shifts, MotiveMob also performs consistently well across cities, achieving competitive results on both Tokyo and Osaka.

## I MOTIVEMOB RESULTS ACROSS LLM BACKBONES

Table 10 shows that MotiveMob achieves consistently strong results across different LLM backbones, indicating that its effectiveness is not dependent on a particular backbone model.

Table 10: MotiveMob results with different LLM backbones for seen and unseen users across the three temporal settings. All values are JSD; lower is better.
<table><tr><td rowspan="2">Backbone</td><td colspan="4">Seen period</td><td colspan="4">Regular unseen period</td><td colspan="4">COVID-19 unseen period</td></tr><tr><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td></tr><tr><td>Seen users</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-9B</td><td>0.0080</td><td>0.0088</td><td>0.0671</td><td>0.3552</td><td>0.0096</td><td>0.0116</td><td>0.0902</td><td>0.4464</td><td>0.0452</td><td>0.0173</td><td>0.1396</td><td>0.5728</td></tr><tr><td>Gemma-4-E4B</td><td>0.0081</td><td>0.0191</td><td>0.0828</td><td>0.4041</td><td>0.0114</td><td>0.0192</td><td>0.0953</td><td>0.4731</td><td>0.0425</td><td>0.0200</td><td>0.1300</td><td>0.5431</td></tr><tr><td>Gemma-2-9B</td><td>0.0099</td><td>0.0064</td><td>0.0641</td><td>0.3760</td><td>0.0097</td><td>0.0118</td><td>0.0889</td><td>0.4296</td><td>0.0482</td><td>0.0138</td><td>0.1319</td><td>0.5696</td></tr><tr><td>Unseen users</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-9B</td><td>0.0253</td><td>0.0268</td><td>0.0663</td><td>0.4798</td><td>0.0358</td><td>0.0288</td><td>0.0913</td><td>0.5411</td><td>0.0308</td><td>0.0278</td><td>0.1198</td><td>0.6431</td></tr><tr><td>Gemma-4-E4B</td><td>0.0166</td><td>0.0373</td><td>0.0857</td><td>0.4913</td><td>0.0184</td><td>0.0382</td><td>0.1022</td><td>0.5514</td><td>0.0242</td><td>0.0299</td><td>0.1121</td><td>0.6267</td></tr><tr><td>Gemma-2-9B</td><td>0.0362</td><td>0.0153</td><td>0.0727</td><td>0.4675</td><td>0.0366</td><td>0.0152</td><td>0.0918</td><td>0.5348</td><td>0.0244</td><td>0.0216</td><td>0.1184</td><td>0.6308</td></tr></table>

## J ANALYSIS OF TRAJECTORY TERMINATION

In MotiveMob, the motivation predictor $P _ { M }$ has two related responsibilities. At each generation step, it either predicts a semantic motivation for the next transition or emits END to terminate the trajectory. The state predictor $P _ { S }$ is invoked only when a non-terminal motivation is produced. In the direct ablation without $P _ { M }$ , a single predictor instead outputs either the next state or END. Thus, all configurations retain a learned termination decision, but differ in whether termination is separated from state generation through the motivation interface.

Since the timing of END determines the temporal span and number of generated visits, it may affect both SI and DARD. We therefore compare the trajectory lengths and termination behavior of the direct model and three variants sharing the same motivation predictor.

Table 11: Trajectory-length and termination statistics for the motivation-mediation ablation. Mean length and termination rates are aggregated over all $^ { 4 , 6 7 1 }$ test days from the six evaluation splits. SI and DARD are macro averages over the six splits. The maximum rollout length is 25 states.
<table><tr><td>State generation</td><td>END predictor</td><td>Mean length</td><td>Max-step rate</td><td>SI↓</td><td>DARD↓</td></tr><tr><td>Direct, w/o  $P _ { M }$  (IWM)</td><td>Direct predictor</td><td>7.52</td><td>1.48%</td><td>0.0282</td><td>0.1401</td></tr><tr><td> $P _ { S } \ ( \mathrm { I W M } )$ </td><td> $P _ { M }$ </td><td>8.08</td><td>2.27%</td><td>0.0205</td><td>0.1087</td></tr><tr><td> $P _ { S } ( \mathrm { p o s i t i v e - o n l y } )$ </td><td> $P _ { M }$ </td><td>7.42</td><td>2.06%</td><td>0.0176</td><td>0.1132</td></tr><tr><td>Rule-based state predictor</td><td> $P _ { M }$ </td><td>9.06</td><td>1.76%</td><td>0.0215</td><td>0.1163</td></tr><tr><td>Ground truth</td><td></td><td>6.82</td><td>一</td><td>一</td><td>1</td></tr></table>

Table 11 shows that the improvements in SI and DARD cannot be explained solely by matching the average trajectory length. The direct model generates 7.52 states per day on average, which is closer to the ground-truth mean of 6.82 than the 8.08 states generated by the motivation-guided model with the IWM state predictor. Nevertheless, the direct model obtains substantially worse SI and DARD (0.0282 and 0.1401, respectively) than its motivation-guided counterpart (0.0205 and 0.1087). This result suggests that the semantic interface provided by $P _ { M }$ contributes information beyond a simple control of trajectory length.

Reaching the rollout limit is also uncommon: only 1.48–2.27% of trajectories reach the 25-state limit across the learned predictors. Consequently, differences in the number of capped trajectories are unlikely to be the main cause of the SI and DARD improvements. The same $P _ { M }$ paired with different state predictors still produces noticeable metric variation. In particular, the positive-only state predictor achieves the best SI, whereas the IWM predictor achieves the best DARD. We therefore interpret motivation predictor as an important contributor of the observed improvements.

## K TRAJECTORY-LEVEL EFFECTS OF THE MOTIVATION PREDICTOR

Figure 4 provides three examples comparing MotiveMob with and without the trained motivation predictor. The cases are selected from the same evaluation inputs used in the quantitative experiments. Both variants use the same trained state predictor, so the comparison isolates the contribution of motivation prediction. Lines connect consecutive visits and indicate visit order.

![](images/6b225d52574501ad9dfb9ce2f099b63ea2fa3ec08b4ddb92249c8d5a78d32209.jpg)  
Figure 4: Trajectory comparisons with and without the trained motivation predictor. From left to right, the columns show the ground-truth trajectory, MotiveMob with the motivation predictor, and the corresponding ablation without it. Numbered markers indicate visit order, and labels show the visit time and POI.

Case 1: Station-centered return pattern. The ground-truth trajectory revisits Rail Station#1 from 17:20 to 20:50. With the motivation predictor, MotiveMob generates Rail Station#1 at 17:00, 17:50, and 20:30, retaining both the station-centered activity and its evening timing. Without the predictor, the trajectory becomes Rail Station#1 → Hotel#1952 → Rail Station#1 → Hotel#2867 → Rail Station#1 → Hotel#2867, introducing repeated hotel detours.

Case 2: Dining activity consistency. The reference trajectory moves from Japanese Restaurant#416 at 12:10 to Sake Bar#1466 at 18:30. MotiveMob with the predictor generates Ramen Restaurant#1972 at 12:10, preserving the dining motivation and the time of the first activity, although it produces a shorter trajectory. The ablation instead alternates between Children’s Clothing Store#20 and Supermarket#1289 six times, shifting the trajectory from dining to retail and adding repetitive visits.

Case 3: Commute pattern preservation. The ground truth contains Metro Station#472 followed by Platform#360. With the predictor, MotiveMob retains this ordered transit pair and adds one nearby visit to Shopping Mall#1254. Without the predictor, the sequence repeatedly moves between Platform#360 and the distant Community Center#281 before ending at Apartment or Condo#86, producing a longer and less coherent chain.

Overall, these examples show that the motivation predictor helps preserve coherent activity timing and semantics, whereas removing it more often produces unrelated POIs, repetitive transitions, and implausible activity sequences. This pattern is consistent with the ablation results, where removing $P _ { M }$ increases both SI and DARD across all six evaluation settings. Although some cases also exhibit fewer spatial detours with $P _ { M }$ , its primary contribution is improved temporal and activity-semantic coherence rather than a uniform improvement in spatial distribution.

## L MOTIVATION PREDICTOR SENSITIVITY TO EARLY-EXPERIENCE SUPERVISION

Following the early-experience training procedure Zhang et al. (2025), we investigate that whether one epoch of implicit world modeling (IWM) helps the subsequent motivation predictor.

We hold the state predictor fixed, using the IWM-trained LLM state engine in every experiment. We compare three initialization schemes for the motivation predictor: (1) direct imitation learning (IL) without IWM pretraining; (2) one epoch of IWM pretraining on a balanced mixture of observed and alternative transitions $( p _ { + } = 0 . 5 )$ ; and (3) one epoch of IWM pretraining on observed transitions only $( p _ { + } = 1 . 0 )$ . For each initialization scheme, we subsequently train the motivation predictor for 4, 8, or 12 epochs. All other inference settings are held constant.

Figure 5 reports the four JSD metrics across the six evaluation splits. The horizontal axis indicates the motivation predictor initialization scheme, while the vertical axis indicates the number of motivation predictor training epochs. Each cell displays the corresponding raw JSD value.

The benefit of one epoch of IWM is most visible after 8 epochs of motivation predictor training. At this checkpoint, the $p _ { + } = 0 . 5$ and $p _ { + } = 1 . 0$ models outperform the epoch-matched IL baseline in 15 and 16 of the 24 test–metric comparisons, respectively. In particular, $p _ { + } = 0 . 5$ reduces the mean SI from 0.0208 to 0.0192 and the mean DARD from 0.1200 to 0.1134. However, its SD and STVD are slightly higher. Therefore, IWM initialization helps the motivation predictor learn SI and DARD more quickly, but does not improve every metric.

At epoch 12, the IWM advantage is no longer consistent. The selected IL model obtains the best overall mean rank (3.81) and the lowest mean DARD (0.1043), whereas the $p _ { + } = 1 . 0$ model obtains the lowest mean SI (0.0169). Thus, one epoch of IWM provides an intermediate-training advantage rather than a uniform improvement at the final checkpoint.

![](images/8d27ab0f87f5fb8f028500c62591aac254e48c45275ee8b8737e8d07d4c4ba4b.jpg)

![](images/82ba21acc3c91e4fc2afbf75b8a7b2c6ffa94aa97a4870163002c92ac2ea88e5.jpg)

![](images/b47a8071bdfc6c608e7abcf7c6b84c05efee62ac0fcb8f4e8fa6e436c8201b58.jpg)

![](images/903a9ecec47a17889feb2476e5d52d69981991aca252689b4ba3da428447fc2d.jpg)  
Figure 5: Motivation-predictor sensitivity to IWM initialization and training duration. Each evaluation split contains four $3 \times 3$ matrices corresponding to SD, SI, DARD, and STVD. Rows show results after 4, 8, and 12 epochs of motivation-predictor training, while columns show direct IL, balanced IWM initialization $( p _ { + } = 0 . 5 )$ , and observed-only IWM initialization $( p _ { + } = 1 . 0 )$ . Cell annotations report raw JSD values; lower values and lighter shading indicate better distributional alignment.

Table 12: Mean JSD over the six evaluation splits. All metrics are lower-is-better. Mean rank is calculated by ranking the nine configurations separately for every test split and metric and then averaging the resulting 24 ranks.
<table><tr><td>Initialization</td><td>Epoch</td><td>SD</td><td>SI</td><td>DARD</td><td>STVD</td><td>Mean rank</td></tr><tr><td>IL</td><td>4</td><td>0.0200</td><td>0.0372</td><td>0.1550</td><td>0.4771</td><td>5.42</td></tr><tr><td>IL</td><td>8</td><td>0.0204</td><td>0.0208</td><td>0.1200</td><td>0.4898</td><td>5.04</td></tr><tr><td>IL</td><td>12</td><td>0.0203</td><td>0.0198</td><td>0.1043</td><td>0.4857</td><td>3.81</td></tr><tr><td> $\mathrm { I W M } , p _ { + } = 0 . 5$ </td><td>4</td><td>0.0209</td><td>0.0378</td><td>0.1616</td><td>0.4742</td><td>5.42</td></tr><tr><td> $\mathrm { I W M } , p _ { + } = 0 . 5$ </td><td>8</td><td>0.0213</td><td>0.0192</td><td>0.1134</td><td>0.4949</td><td>4.92</td></tr><tr><td> $\mathrm { I W M } , p _ { + } = 0 . 5$ </td><td>12</td><td>0.0204</td><td>0.0207</td><td>0.1109</td><td>0.4969</td><td>5.02</td></tr><tr><td> $\mathrm { I W M } , p _ { + } = 1 . 0$ </td><td>4</td><td>0.0213</td><td>0.0391</td><td>0.1658</td><td>0.4807</td><td>6.33</td></tr><tr><td> $\mathrm { I W M } , p _ { + } = 1 . 0$ </td><td>8</td><td>0.0235</td><td>0.0196</td><td>0.1118</td><td>0.4934</td><td>4.92</td></tr><tr><td> $\mathrm { I W M } , p _ { + } = 1 . 0$ </td><td>12</td><td>0.0225</td><td>0.0169</td><td>0.1115</td><td>0.4927</td><td>4.13</td></tr></table>

## M INFERENCE-TIME VALIDATION AND TERMINATION STATISTICS

We report the operational statistics of MotiveMob’s inference-time validation and termination mechanisms. Table 13 reports the intervention statistics for the speed constraint. A step requiring resampling is a transition for which at least one state proposal exceeds the speed limit. A step is considered resolved by resampling if a valid proposal is obtained within the attempt budget. Otherwise, the samecategory feasibility correction is invoked. We separately report whether the correction successfully identifies a feasible substitute.

We additionally apply repetition-aware termination after the generated trajectory reaches a minimum length of 15 states. Generation terminates when appending a candidate would extend a consecutively repeating location pattern. This operation preserves the generated prefix and is equivalent to emitting END before the repeated candidate.

Table 13: Inference-time validation and termination statistics. All percentages are normalized by generated states, except repetition termination, which is normalized by generated trajectories.
<table><tr><td>User status</td><td>Period</td><td>Days</td><td>States</td><td>Resampling required</td><td>Resampling resolved</td><td>Correction invoked</td><td>Feasibility corrected</td><td>No feasible substitute</td><td>Repetition terminated</td></tr><tr><td>Seen</td><td>Seen</td><td>908</td><td>7,647</td><td>932 (12.2%)</td><td>528 (6.9%)</td><td>404 (5.3%)</td><td>231 (3.0%)</td><td>173 (2.3%)</td><td>121 (13.3%)</td></tr><tr><td>Seen</td><td>Regular unseen</td><td>690</td><td>5,446</td><td>686 (12.6%)</td><td>415 (7.6%)</td><td>271 (5.0%)</td><td>134 (2.5%)</td><td>137 (2.5%)</td><td>88 (12.8%)</td></tr><tr><td>Seen</td><td>COVID-19 unseen</td><td>392</td><td>3,218</td><td>830 (25.8%)</td><td>459 (14.3%)</td><td>371 (11.5%)</td><td>237 (7.4%)</td><td>134 (4.2%)</td><td>77 (19.6%)</td></tr><tr><td>Unseen</td><td>Seen</td><td>1,345</td><td>10,066</td><td>2,582 (25.7%)</td><td>1,534 (15.2%)</td><td>1,048 (10.4%)</td><td>629 (6.2%)</td><td>419 (4.2%)</td><td>207 (15.4%)</td></tr><tr><td>Unseen</td><td>Regular unseen</td><td>795</td><td>6,016</td><td>1,610 (26.8%)</td><td>929 (15.4%)</td><td>681 (11.3%)</td><td>432 (7.2%)</td><td>249 (4.1%)</td><td>150 (18.9%)</td></tr><tr><td>Unseen</td><td>COVID-19 unseen</td><td>541</td><td>4,294</td><td>1,224 (28.5%)</td><td>716 (16.7%)</td><td>508 (11.8%)</td><td>260 (6.1%)</td><td>248 (5.8%)</td><td>97 (17.9%)</td></tr><tr><td>All splits</td><td></td><td>4,671</td><td>36,687</td><td>7,864 (21.4%)</td><td>4,581 (12.5%)</td><td>3,283 (8.9%)</td><td>1,923 (5.2%)</td><td>1,360 (3.7%)</td><td>740 (15.8%)</td></tr></table>

## N ADDITIONAL SPATIAL DISTRIBUTION ALIGNMENT RESULTS

Figure 6 presents the four evaluation settings omitted from the main spatial-distribution analysis. All three methods broadly reproduce the major ground-truth activity centers, with relatively modest differences when users and temporal patterns are familiar. Compared with LLM4POI, MotiveMob generally forms more compact hotspots with less peripheral dispersion. MotiveMob more consistently preserves the allocation of visit mass across the main activity regions, particularly under the COVID-19 shift. Overall, these results support the main finding that MotiveMob maintains robust spatial alignment as the user or temporal setting changes.

![](images/bce11a9f0370bbd698517df3397c6941fcf6bdc0efc6dbfe7e33f241916be0db.jpg)  
Figure 6: Ground-truth and predicted spatial mobility distributions for four additional evaluation settings. From top to bottom, the rows show seen users in the seen period, seen users in the regular unseen period, seen users in the COVID-19 unseen period, and unseen users in the seen period. Darker red indicates higher visit density.