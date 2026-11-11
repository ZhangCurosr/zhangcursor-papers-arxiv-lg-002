# Uncertainty-Aware Hybrid Offline Reinforcement Learning with Online Action Selection Enabled by Sparse Gaussian-Process Critics

Marco Chan\*, Christos Evripidou\*, Liqiu Dong, Benoit Chachuat, and Mehmet Mercangoz¨

Abstract— Start-ups and grade changes remain largely operator-driven, while developing and maintaining dynamic models for model-based control requires substantial engineering effort. Hybrid Offline Learning and Online Optimisation (HOFLON) uses one-step offline reinforcement learning to select actions from historical transition data. In its original implementation, the online objective combines the predicted return from a tree-ensemble critic with an adversarial-autoencoder (AAE) penalty for unfamiliar state–action pairs. We examine the effect of this penalty and replace the critic and AAE with a sparse variational Gaussian-process (sGP) critic. Actions maximise µ−κσ, an uncertainty-adjusted estimate of the return from one action followed by the historical operating practice. On the HOFLON polymerisation-reactor benchmark, removing the AAE from HOFLON led to 4 failed start-ups in 15 runs over five critics. At its published weight, the variation in the AAE penalty is four orders of magnitude smaller than the variation in the tree-ensemble critic. Nevertheless, it changes actions when the predicted returns of competing candidates are nearly equal, and paired runs with and without the penalty diverged from the first move. With a Matern-5/2 sGP critic,´ κ = 0 led to failure in 13 of 15 runs, whereas with κ = 3 none of 15 runs (five critics, three noise realisations) failed. The median and lowest returns were 13,464 and 13,398 (HOFLON: 13,393 and 12,411). A squared-exponential sGP critic with learned inducing inputs needed κ ≥ 3 to remain within the empirical data support in steady state, and then settled with a temperature offset of 2.0 K.

Index Terms— Offline reinforcement learning, Gaussian processes, process control.

## I. INTRODUCTION

Start-ups, shutdowns and grade changes in process plants remain largely human operator-driven, even where routine operation is automated [1], [2]. These transitions can produce off-specification material and take the plant through operating regions where disturbances are more difficult to manage [3]. Process historians contain records of repeated transitions carried out by different operators or under different controller tunings.

Nonlinear model predictive control can automate these transitions, but requires a dynamic model that remains valid throughout the operating range. Developing and maintaining such a model as equipment, feedstocks and operating conditions change involves substantial plant-specific work [4]. Model-free offline reinforcement learning (RL) instead learns from recorded transitions without requiring a process model or exploratory actions on the plant [5].

Distribution shift remains a central difficulty in offline RL. Maximising a value function fitted to historical data can favour state–action pairs that are poorly represented in the training set [6], [7]. Lee, Kaisare and Lee [8] addressed this problem in approximate dynamic programming with a data-density penalty. Other approaches penalise value uncertainty [9]–[12] or correct candidate actions before application [13]. One-step methods estimate the value of the historical behaviour and perform a policy-improvement step without bootstrapping on unobserved actions [14].

HOFLON (Hybrid Offline Learning and Online Optimi sation) [3] follows this one-step approach (Fig. 1, top). In its original implementation, a critic is fitted to discounted returns from historical transitions, while an adversarial autoencoder (AAE) [15] is trained on logged state–action pairs. At each sampling instant, an optimisation problem selects the action with the highest predicted return, subject to penalties on AAE reconstruction error and the change in the manipulated variables. On a polymerisation-reactor startup and a paper-machine grade change, HOFLON achieved higher returns than implicit Q-learning [16] and the best historical runs [3]. The AAE requires separate training and introduces a penalty weight whose effective scale depends on the critic and the data.

We evaluate the effect of the AAE on HOFLON’s closedloop behaviour and test a sparse Gaussian-process (GP) critic as an alternative. GP critics provide a predictive mean and standard deviation [17], [18]; the latter can be used to penalise uncertain actions without a separate autoencoder (Fig. 1, bottom). Sparse GPs were also identified as a possible critic in [3]. A related idea is used in real-time optimisation, where Gaussian processes model the plant– model mismatch and their uncertainty enters the optimisation through acquisition functions and trust regions [19], [20]. In contrast to the optimistic acquisition functions used there to encourage exploration, such as upper confidence bounds, a lower confidence bound is used here to avoid poorly supported actions, within the existing one-step framework [9], [24]. The contributions are:

1) an ablation of the AAE over five independently trained HOFLON critics, including an analysis of the AAE penalty relative to the critic values;

2) HOFLON-sGP, which replaces the tree-ensemble critic and AAE with a sparse variational GP critic and uses a lower confidence bound for one-step action selection;

3) a comparison of two sGP critics, with a Matern-5/2 kernel´ and fixed inducing inputs and with a squared-exponential kernel and learned inducing inputs, over several uncertainty multipliers κ, evaluated through closed-loop return,

temperature, actuator activity and empirical data support. Section II reviews HOFLON, Section III presents the sGP formulation, and Sections IV–VI describe the benchmark, results and discussion. The case study is made available at https://github.com/AISL-at-Imperial-Col lege-London/HOFLON-sGP, released as Python code for the simulator, accompanied by the offline dataset, the trained critics, the offline training and online control scripts, and the closed-loop results.

## II. HOFLON

## A. Offline learning

At each sampling instant k the controller receives a measured state $s _ { k }$ and applies an action $a _ { k } \in [ 0 , 1 ] ^ { m }$ , the vector of scaled manipulated variables. The dataset D contains $N _ { r }$ historical transitions of H steps, recorded under unknown and possibly different behaviour policies, for example operators or controllers with different tunings. Every step has a reward $r _ { k }$ and is labelled with its discounted return

$$
G _ { k } = \sum _ { j = 0 } ^ { H - 1 - k } \gamma ^ { j } r _ { k + j } , \qquad \gamma \in ( 0 , 1 ) .\tag{1}
$$

The critic $\hat { Q }$ is a regression model fitted by least squares to $G _ { k }$ from the features $\boldsymbol { z } _ { k } = [ s _ { k } , e _ { k } , \iota _ { k } , a _ { k } ]$ , where $e _ { k }$ is the set-point error and $\iota _ { k }$ its running integral. It estimates the value $Q _ { \beta } ( s , a )$ of applying a in state s and following the historical behaviour afterwards. Because the runs use different controller tunings, β denotes the aggregate historical behaviour represented by the dataset. In [3], Q<sup>ˆ</sup> is a gradientboosted tree ensemble (XGBoost [21]).

The critic may extrapolate poorly to state–action pairs that are absent from the training set. HOFLON therefore also trains an AAE on $\boldsymbol { x } = [ s , e , a ]$ . Its encoder E maps x to a low-dimensional latent code, and its decoder D reconstructs x from the code. The reconstruction error

$$
\begin{array} { r } { \phi _ { \mathrm { l a t } } \big ( s , a \big ) = \frac { 1 } { n _ { x } } \big \| x - \eta ^ { - 1 } \big ( D ( E ( \eta ( x ) ) ) \big ) \big \| _ { 2 } ^ { 2 } , } \end{array}\tag{2}
$$

where $\eta ( \cdot )$ denotes feature-wise scaling, is used as a penalty for state–action combinations that are poorly reconstructed by the AAE.

## B. Online optimisation

At instant k the measured state is fixed and the controller applies

$$
a _ { k } ^ { \star } = \arg \operatorname* { m a x } _ { a \in [ 0 , 1 ] ^ { m } } \hat { Q } ( z _ { k } ( a ) ) - \lambda _ { 1 } \phi _ { \mathrm { l a t } } ( s _ { k } , a ) - \lambda _ { 2 } \| a - a _ { k - 1 } \| _ { 2 } ^ { 2 } .\tag{3}
$$

The objective combines predicted return, AAE reconstruction error and a penalty on changes in the manipulated variables. Problem (3) has m decision variables, and its objective is not smooth because a tree ensemble is piecewise constant. Three derivative-free methods are used: Nelder–Mead (250 iterations), bounded Powell (250 iterations) and DIRECT (40 iterations). The first two start at $a _ { k - 1 } ;$ the candidate with the highest objective value is applied. The integrators ι are frozen while a move is clipped at its bounds (anti-windup) [3].

![](images/1c4c1bd5ecd366b09127eb57de3f5635708ddde59a51be9b1233f21192650b8a.jpg)  
Fig. 1. Structure of HOFLON [3] (top), with a tree-ensemble critic and an AAE, and of HOFLON-sGP (bottom), with one sparse GP critic. In HOFLON-sGP the predictive standard deviation of the critic is the support term.

C. Relation to offline RL and approximate dynamic programming

HOFLON is a one-step offline RL method [14]. The critic is a Monte-Carlo estimate of the value of the behaviour policy, obtained without Bellman backups or iterations. Problem (3) is a single greedy policy-improvement step with respect to this estimate. It is restricted by the support term and solved at run time, so no actor network is trained. In the terminology of approximate dynamic programming, this is one iteration of approximate policy iteration, or rollout with the historical operating practice as base policy and with Q-factors estimated from logged data [22]. The method of [8] iterates value iteration on a state cost-to-go and uses a process model for the one-step lookahead; the Qfactor formulation of HOFLON requires no model. The term “online” refers to the numerical optimisation of the critic at run time. No learning takes place on the plant.

The one-step approach evaluates each candidate action under the assumption that historical behaviour continues afterwards. It therefore does not optimise a sequence of new actions or directly estimate the optimal policy. Repeating this choice at successive states can combine actions from different historical runs and may achieve returns above those observed in the dataset.

## III. HOFLON-SGP

HOFLON-sGP retains the online optimisation in HOFLON but uses a single GP critic to estimate both expected return and predictive uncertainty (Fig. 1).

## A. Sparse variational GP critic

We place a GP prior on the return function $f ( x ) , f \sim$ $\mathcal { G P } ( m , k _ { \theta } )$ , which is conditioned on the data. We use a constant prior mean m equal to the mean training return and an automatic-relevance-determination Matern-5/2 kernel´ $k _ { \theta } ( x , x ^ { \prime } ) = \sigma _ { t } ^ { 2 } \bigl ( 1 + \sqrt { 5 } \rho + { \textstyle { \frac { 5 } { 3 } } } \rho ^ { 2 } \bigr ) e ^ { - \sqrt { 5 } \rho }$ with $\rho ^ { 2 } = \textstyle \sum _ { i } ( x _ { i } -$ $x _ { i } ^ { \prime } ) ^ { 2 } / \ell _ { i } ^ { 2 }$ on standardised inputs. The squared-exponential kernel $\sigma _ { f } ^ { 2 } e ^ { - \rho ^ { 2 } / 2 }$ is used for comparison (Section V-D). The length scales $\ell _ { i }$ determine how fast the return may change along each input. Exact GP regression involves an $N \times N$ matrix factorisation, which is costly for the approximately $2 \times 1 0 ^ { 4 }$ records used here. We therefore use $M \ \ll \ N$ inducing inputs $Z$ to obtain a sparse approximation. We use the variational free-energy (VFE) approximation of Titsias [23]. The inducing inputs are the k-means centres of the standardised training inputs and are kept fixed. Fixed inducing inputs keep the region where σ is small tied to the data; in a preliminary study, learning them did not improve the closed-loop behaviour. The hyperparameters θ and the noise variance $\sigma _ { n } ^ { 2 }$ maximise the collapsed bound

$$
\begin{array} { r l } { \log \mathcal { N } \big ( y | m \mathbf { 1 } , Q _ { n n } + \sigma _ { n } ^ { 2 } I \big ) - \frac { 1 } { 2 \sigma _ { n } ^ { 2 } } \mathrm { t r } \big ( K _ { n n } - Q _ { n n } \big ) , } \end{array}\tag{4}
$$

with $\begin{array} { r l r } { Q _ { n n } } & { { } = } & { K _ { n u } K _ { u u } ^ { - 1 } K _ { u n } } \end{array}$ . With $\begin{array} { r l r } { \Sigma } & { { } = } & { ( K _ { u u } \ + } \end{array}$ $\sigma _ { n } ^ { - 2 } K _ { u n } K _ { n u } ) ^ { - 1 }$ and $k _ { u } ( x ) = k _ { \theta } ( Z , x )$ , the predictive mean and the latent (epistemic) standard deviation of the return are

$$
\mu ( x ) = m + \sigma _ { n } ^ { - 2 } k _ { u } ( x ) ^ { \top } \Sigma K _ { u n } ( y - m { \bf 1 } ) ,\tag{5}
$$

$$
\sigma ^ { 2 } ( x ) = k _ { \theta } ( x , x ) - k _ { u } ( x ) ^ { \top } \big ( K _ { u u } ^ { - 1 } - \Sigma \big ) k _ { u } ( x ) .\tag{6}
$$

The latent standard deviation excludes the observation noise $\sigma _ { n }$ and describes uncertainty in the expected return rather than variability between individual returns. It is generally smaller near well-represented inputs. Far from the inducing inputs, $\sigma$ approaches the prior standard deviation $\sigma _ { f }$ and $\mu$ approaches $m ;$ the prior mean therefore influences action selection outside the training region. The calibration of $\sigma$ can be checked on held-out logged transitions. Away from the data, $\sigma$ follows from the kernel and its hyperparameters and is a model-based approximation of the epistemic uncertainty. After precomputation, the mean and variance require $\mathcal O ( M )$ and $\mathcal { O } ( M ^ { 2 } )$ operations per query, respectively. Their evaluation does not require automatic differentiation at run time.

## B. Pessimistic action selection

HOFLON-sGP replaces (3) by

$$
a _ { k } ^ { \star } = \arg \operatorname* { m a x } _ { a \in [ 0 , 1 ] ^ { m } } \mu ( x _ { k } ( a ) ) - \kappa \sigma ( x _ { k } ( a ) ) - \lambda _ { 2 } \| a - a _ { k - 1 } \| _ { 2 } ^ { 2 } ,\tag{7}
$$

using the same move penalty and solvers, but no AAE. The critic estimates $Q _ { \beta }$ , so the first two terms form a lower confidence bound on the expected return of applying a and then following the historical practice. For a fixed action and a calibrated GP posterior, the posterior probability that the expected return exceeds $\mu - \kappa \sigma$ is $\Phi ( \kappa )$ , or 99.87 % for $\kappa = 3$ . This is a pointwise statement. The online solvers evaluate several hundred actions and select the maximiser of $\mu - \kappa \sigma _ { \mathrm { : } }$ so the pointwise probability does not provide a simultaneous confidence level for the candidates or a guarantee along the closed-loop trajectory. The objective is the pessimistic counterpart of the upper confidence bound used for exploration in GP bandits [24]. Like supportconstrained policy-improvement methods [25], it discourages actions with weak data support, but it does not impose a hard support constraint.

The penalties in (3) and (7) have different scales and interpretations. The GP term κσ is expressed in return units, with a dimensionless multiplier $\kappa .$ The AAE term requires a weight $\lambda _ { 1 }$ to convert reconstruction error into the scale of the objective. The GP standard deviation is determined by the kernel and the training inputs, whereas the AAE penalty depends on reconstruction of each state–action pair. Neither is a direct measure of data density. Fig. 2 shows the terms at a logged start-up state. Near the logged actions both support terms are small. Over the action set, the AAE term varies by 0.024 return units, the critic by 247 and κσ by 32–1,411.

![](images/76d2f162e7de42795a4c57f6b853cfcbc32fc934ee1e9155e53c0991b1c184aa.jpg)

![](images/64857a6b2c2b9e433d6598e720273da8d68ad03f1d22fc31c82aa9f9797784ab.jpg)

![](images/2a3da658676a9cf72646c2b0992c8db334903ce62c813885d03f04443448f369.jpg)

![](images/824c9996fbed9acdd53f8fb8bb39b514feb95344d403c8c58782bc9d91f62934.jpg)  
Fig. 2. Terms of the online objectives over the action set at a logged start-up state (historical run with median return, $t = 6 \mathrm { h } , P = 6 5 \mathrm { k g } \mathrm { m } ^ { - 3 } ) .$ Dots: actions of all 100 historical runs at this time. Stars: maximisers of (3) in (a) and of (7) with $\kappa = 3$ in (c). The colour scales differ between panels: the AAE term (b) spans 0.024 return units, the critic (a) 247 and the sGP term (d) 32–1,411.

## C. Relation to the historical operating practice

The following remark relates the performance of (7) to that of the historical practice. Let $V _ { \beta } ( s ) = \mathbb { E } _ { a \sim \beta ( \cdot | s ) } Q _ { \beta } ( s , a )$ and $A _ { \beta } ( s , a ) = Q _ { \beta } ( s , a ) - V _ { \beta } ( s )$

Remark 1: Let π be the policy defined by (7), with $a k _ { - 1 }$ included in the state $s ,$ and let $\delta ( s , a ) = \lambda _ { 2 } \lVert a - a _ { k - 1 } \rVert _ { 2 } ^ { 2 } .$ Assume that $| Q _ { \beta } ( s , a ) - \mu ( s , a ) | \ \leq \ \kappa \sigma ( s , a )$ holds at the states visited by π, for $a = \pi ( s )$ and for the actions of $\beta .$ Since $\delta \geq 0$ and $\pi ( s )$ maximises $\mu - \kappa \sigma - \delta ,$

$$
\begin{array} { l l } { { Q _ { \beta } ( s , \pi ( s ) ) \geq \mu ( s , \pi ( s ) ) - \kappa \sigma ( s , \pi ( s ) ) - \delta ( s , \pi ( s ) ) } } \\ { { \qquad \geq \mu ( s , a ) - \kappa \sigma ( s , a ) - \delta ( s , a ) } } \\ { { \qquad \geq Q _ { \beta } ( s , a ) - 2 \kappa \sigma ( s , a ) - \delta ( s , a ) } } \end{array}
$$

for every action a. Averaging over $\begin{array} { r } {  { \boldsymbol { a } } \sim \beta ( \cdot |  { \boldsymbol { s } } ) } \end{array}$ gives $A _ { \beta } ( s , \pi ( s ) ) ~ \geq ~ - \mathbb { E } _ { a \sim \beta ( \cdot | s ) } [ 2 \kappa \sigma ( s , a ) + \delta ( s , a ) ]$ . With the performance difference lemma [26, Lemma 6.1],

$$
V _ { \pi } ( s _ { 0 } ) \geq V _ { \beta } ( s _ { 0 } ) - \frac { \mathbb { E } _ { s \sim d _ { \pi } } \mathbb { E } _ { a \sim \beta ( \cdot | s ) } \left[ 2 \kappa \sigma + \delta \right] ( s , a ) } { 1 - \gamma } ,\tag{8}
$$

where $d _ { \pi }$ is the normalised discounted state distribution of $\pi$ from $s _ { 0 }$ . The same bound holds with $s _ { 0 }$ replaced by any state of the trajectory.

Under the assumptions of Remark 1, bound (8) relates the loss relative to historical operation to predictive uncertainty and the move penalty at historical actions, evaluated at states visited by π. The σ part is small if these states lie in the support of the data and large otherwise. The share of offsupport steps reported in Section V is an empirical indicator of this condition. The remark is a heuristic justification of (7) and not a performance guarantee for the implemented controller. It assumes that the bounds hold at all visited states, which is most restrictive at $a = \pi ( s )$ because $\pi ( s )$ maximises $\mu - \kappa \sigma$ , that $\beta$ is a stationary Markov policy in s, and that the returns are not truncated. The historical PI controllers depend on their tuning and integrator state, and the returns in (1) end with each run.

## D. Critic features

The critic of [3] also receives the integrated errors $\iota _ { k }$ . In the historical data the PI controllers saturate during the startup and their integrators remain frozen. Under the learned controller the integrators continue to integrate, so large integrator values early in a run are outside the data support. The resulting integrated-error features differ between the logged runs and the learned-controller trajectories. A tree ensemble extrapolates such queries with a constant value, and a GP returns its prior. HOFLON-sGP therefore uses the input of the AAE, $\mathit { x } ~ = ~ [ s , e , a ]$ , without integrators. The integrators are part of the state of the historical PI controllers, so this choice also removes information about the continuation policy from the critic. The comparison between HOFLON and HOFLON-sGP changes the critic model, the support term and the critic features together.

## IV. CASE STUDY AND PROTOCOL

## A. Polymerisation-reactor start-up

We use the benchmark of [3]: a continuous stirred-tank reactor for free-radical polymerisation based on [27], with lumped initiation, propagation and termination kinetics and an energy balance with jacket cooling. The reactor starts at 350 K without initiator, radicals or polymer. The controller has to bring the polymer concentration $P ~ { \mathrm { t o ~ 1 0 0 ~ k g ~ m ^ { - 3 } } }$ and hold the temperature $T$ at 350 K. It manipulates the initiator feed $u _ { I }$ (scaled from $0 { - } 2 . 5 \mathrm { k g } \mathrm { h } ^ { - 1 } )$ and the coolant temperature $u _ { T _ { c } }$ (scaled from 280–400 K). The sampling time is 30 min and each run lasts 100 h, i.e. 200 steps. Monomer, polymer and temperature measurements have Gaussian noise with standard deviations of $2 \mathrm { k g } \mathrm { m } ^ { - 3 } , 0 . 5 \mathrm { k g } \mathrm { m } ^ { - 3 }$ and 0.2 K. With $\varepsilon _ { k } = e _ { P , k } ^ { 2 } + e _ { T , k } ^ { 2 }$ computed from the scaled errors $( P / 1 0 0$ and $T / 3 5 0 )$ , the reward is

$$
r _ { k } = 5 0 \big ( e ^ { - \varepsilon _ { k } / 0 . 0 5 ^ { 2 } } - \varepsilon _ { k } \big ) + 5 0 0 0 \big ( \varepsilon _ { k - 1 } - \varepsilon _ { k } \big ) .\tag{9}
$$

The first term rewards operation near the set-points; the second rewards progress towards them. The reward contains no actuation cost, so actuator activity is assessed separately. The dataset consists of 100 start-ups under PI control with randomly drawn gains [3]. The total reward is $1 0 , 4 1 4 \pm 1 , 8 8 4$ (mean ± standard deviation; lowest 5,752), and the coolant temperature remains within 340.1–344.4 K in all of them.

## B. Critics and weights

• XGBoost critics use the settings of [3]: 1600 trees of depth 8, learning rate 0.03, subsampling $0 . 9 , \gamma = 0 . 9 $ , 11 features including the integrators, and a random 80/20 row split, which places rows of the same run in both sets. Five critics differ only in the boosting seed. Their validation $R ^ { 2 }$ is 0.9926–0.9927.

$s G P$ critics of HOFLON-sGP use the Matern-5/2 kernel,´ $M = 5 0 0$ fixed inducing inputs and $\gamma = 0 . 9$ . They are trained on runs 0–94 by maximising the collapsed bound with Adam (learning rate 0.02, 500 iterations, final iterate) and evaluated in the closed form (5)–(6). Five fits differ only in the k-means seed. On the held-out runs 95–99 their $R ^ { 2 }$ is $0 . 9 7 8 ,$ , their 95 % predictive intervals contain 94 % of the held-out returns, the median latent $\sigma$ is 4.3, and $\sigma _ { f } \approx 4 8 1$

• Squared-exponential sGP critics are trained on the same runs with GPyTorch [28], with the inducing inputs initialised by k-means and learned together with the hyperparameters (200 Adam iterations). Five fits differ only in the training seed. On the held-out runs all five have $R ^ { 2 } = 0 . 9 7 9$ , 94 % coverage of the 95 % intervals, a median latent σ of 3.6 and $\sigma _ { f } \approx 3 8 0$ . The GP and XGBoost $R ^ { 2 }$ values use different validation splits and should not be compared directly.

• The AAE is the trained network of [3], with an encoder of layers 9–64–16–4 with tanh activations and a mirrored decoder.

• The weights of the critic, AAE and move terms in the repository of [3] are 6, 0.5 and 0.025, i.e. $\lambda _ { 1 } ~ = ~ 0 . 0 8 3$ and $\lambda _ { 2 } = 0 . 0 0 4 2$ in (3). The value of λ is used for all controllers.

## C. Controllers compared

• HOFLON: XGBoost critic with the AAE at the published weight, objective (3).

• AAE ablations: the same XGBoost critics without the AAE $( \lambda _ { 1 } = 0 )$ and with $\lambda _ { 1 }$ multiplied by $1 0 ^ { 4 }$

• HOFLON-sGP: Matern-5/2 sGP critic, objective (7), with´ $\kappa \in \{ 0 , 0 . 5 , 1 , 2 , 3 \} ; \kappa = 3$ is the proposed setting.

• Squared-exponential sGP critic: objective (7) with $\kappa \in$ $\{ 0 , 0 . 2 5 , 0 . 5 , 1 , 1 . 5 , 2 , 2 . 5 , 3 , 4 , 5 \}$ , and $\kappa ~ = ~ 0$ with the AAE term of (3) at the published weight and with the weight multiplied by $1 0 ^ { 4 }$

## D. Evaluation

All controllers are evaluated in closed loop on the simulator, with the same initial state and plant parameters. The only random element is the measurement noise. A seed of the random-number generator fixes one realisation of this noise over the 200 steps, so runs with the same seed receive the same noise sequence. Because the controller acts on noisy measurements, different realisations can lead to different actions and trajectories.

A run is one controller with one of its five critics on one noise realisation. Differences across critics reflect sensitivity to model fitting, and differences across seeds reflect sensitivity to measurement noise; neither reflects process-model uncertainty. All runs with the same seed are paired, i.e. every controller receives the same noise. Each controller is run with each of its five critics on the same three noise realisations (seeds 11–13), which gives 5 critics × 3 seeds = 15 runs per controller. All results in Section V, including Table I and Figs. 3–5, use these runs. The confidence level κ = 3 was selected on these seeds with the squared-exponential critics, as the smallest tested value for which almost all steps after 20 h were within the data support. The Matern-´ 5/2 kernel, the fixed inducing inputs and the number of training iterations were taken from a preliminary study with a separate implementation and other noise realisations.

![](images/4083efed2be5e4351ffb990f447d7899324f26dbb021f8c6b59b9e6186ce517e.jpg)  
Fig. 3. One representative run per controller, selected as in [3]. For each learned controller, the run shown has the median return of its 15 runs (five critics × three noise realisations, seeds 11–13; Section IV-D). For Historical, the run shown has the median return of the 100 logged runs. Dashed: set-points. Shaded band: range of all logged coolant temperatures.

The following quantities are reported for each run:

• the cumulative reward R of a run. A run is counted as failed if R < 6000. No historical run has R below 5,752, and no failed run in this study has R above 5,437. A low return can result from an offset, an overshoot or a thermal runaway, which are distinguished by the peak temperature and the steady-state values;

• the peak reactor temperature;

• the coolant move, defined as the median |∆T<sub>c</sub>| per step after 20 h, as a measure of actuator activity in steady state;

• the steady-state offsets |P<sup>¯</sup> − 100| and |T<sup>¯</sup> − 350| of the mean values after 50 h;

• the off-support share, defined as the fraction of steps during the start-up (0–20 h) and after 20 h at which $x _ { k }$ is outside the data support. A step counts as off-support if the mean standardised distance of $x _ { k }$ to its ten nearest training records exceeds the 99th percentile of the same distance on held-out runs. This is an empirical indicator of support;

• the solve time of the online optimisation per step.

## V. RESULTS

Table I summarises all controllers, Fig. 4 shows the distributions over runs, and Fig. 3 shows the run with the median return of each controller.

## A. AAE in HOFLON

With the published weight, HOFLON completed all 15 runs (median 13,393, worst 12,411). Without the AAE, 4 of 15 runs failed and the worst return was −2,782. One critic failed at every seed. In the failed runs the polymer concentration overshot its set-point, with mean values of $1 1 4 { - } 1 6 3 \mathrm { k g } \mathrm { m } ^ { - 3 }$ after 50 h and peak temperatures of 367– 375 K. With another critic, the run with seed 12 failed without the AAE (4,673) and reached 13,495 with it. The median peak temperature was 362.6 K without the AAE and 355.8 K with it, and 8 and 1 of the 15 runs, respectively, exceeded 360 K. With one critic, the AAE lowered the median return from 13,020 to 12,521.

At the published weight, the AAE term varies by at most 0.024 return units over the action set and the critic by 247 (Fig. 2). At the applied actions, the AAE term had a median of $2 . 9 \times 1 0 ^ { - 5 }$ and a maximum of 0.01. At this scale, the AAE can influence action selection mainly when candidate actions have nearly equal critic values. A tree ensemble has piecewise-constant predictions, so the reconstruction error can distinguish candidate actions within a flat region of the objective. Paired runs with and without the AAE differed from the first move onwards. With the AAE weight multiplied by 10<sup>4</sup>, the AAE term spans a range similar to that of the critic. No run failed, the worst return was 13,013, and the peak temperature remained below 356.8 K.

$\kappa = 0 .$ . With the GP mean alone, 13 of 15 runs failed (median return −261,594). In these runs the controller set the coolant temperature to its upper bound of 400 K after a median of 10 h, where $\mu$ equals the prior mean and σ the prior standard deviation, and the reactor ran away thermally (peak temperatures of 553–568 K). The other 2 runs, with one critic, did not fail and reached peak temperatures of 374– 375 K.

κ = 3. On seeds 11–13, the median return was 13,464 (HOFLON: 13,393), the lowest return 13,398 (HOFLON: 12,411), and the peak temperature did not exceed 355.7 K (Fig. 4, boxes 3 and 5). None of the 15 runs failed. During the first 20 h, 7.5 % of the steps were off-support (HOFLON: 35 %), and after 20 h 0.0 %. After 20 h the median coolant move was 0.02 K per step (HOFLON: 0.02 K). In steady state, the initiator feed was $1 . 7 2 \mathrm { k g } \mathrm { h } ^ { - 1 } ~ ( \mathrm { P I } ; ~ 1 . 7 7 \mathrm { k g } \mathrm { h } ^ { - 1 } )$ and the offsets were $0 . 4 1 \mathrm { k g } \mathrm { m } ^ { - 3 }$ and 0.20 K (Fig. 3).

## C. Confidence level

With the Matern-5/2 critics and´ $\kappa \in \{ 0 . 5 , 1 , 2 , 3 \}$ , none of the 60 runs on seeds 11–13 failed, and the median returns were 13,464–13,523 (Fig. 5). No step after 20 h was offsupport in any of these runs, the coolant temperature after 20 h remained within 341.0–342.3 K, and the median coolant move was at most 0.04 K per step. During the first 20 h, the median share of off-support steps was 42 % at $\kappa = 0 . 5$ , 32 % at $\kappa = 1$ , 20 % at $\kappa = 2$ and 7.5 % at $\kappa = 3$

## D. Squared-exponential kernel

$\kappa = 0 .$ . All 15 runs failed (median 4,267). Within about 8 h, the controller reached states where $\mu$ and σ were effectively constant across the action set, close to their prior values. The online objective was then dominated by the move penalty, so the warm-started searches retained the previous move and the polymer concentration settled near $8 4 \mathrm { k g } \mathrm { m } ^ { - 3 }$ . With the AAE at the published weight, the paired runs separated after a median of 7 steps. At the states where the $\mathrm { G P }$ is constant, the AAE term is the only term of the objective that depends on the action, and it selects actions by reconstruction error for states outside the data. 15 of 15 runs ended in thermal runaway, with peak temperatures of 386–553 K. With the AAE weight multiplied by $1 0 ^ { 4 }$ , 13 of 15 runs failed.

$\kappa > 0 .$ . For $\kappa \leq 0 . 2 5$ most runs failed (12 of 15 at $\kappa =$ 0.25). For $0 . 5 \leq \kappa \leq 2$ the median share of off-support steps after 20 h was $100 \%$ , and at $\kappa = 2 . 5$ it was $79 \%$ . For $\kappa \leq 1$ the coolant temperature changed by several kelvin per step (4.31 K at $\kappa = 1 )$ , and for $1 . 5 \le \kappa \le 2 . 5$ the inputs were smooth and the polymer offset reached $4 . 7 \mathrm { k g } \mathrm { m } ^ { - 3 }$ (median return $^ { 8 , 6 4 8 }$ at $\kappa = 1 . 5 )$ . With $\kappa = 3 .$ , none of the 15 runs of the squared-exponential controller failed (median 13,378, lowest 13,327), and all but 0.6 % of the steps after 20 h were within the data support. This controller is not shown in Fig. 3. It settled on a branch with an initiator feed of $1 . 3 2 \mathrm { k g } \mathrm { h } ^ { - 1 }$ $( { \mathrm { P I } } ; 1 . 7 7 { \mathrm { k g } } { \mathrm { h } } ^ { - 1 } )$ and a reactor temperature 2.0 K above the set-point (Fig. 4, box 6). At steady-state states visited with $\kappa = 0 . 5$ , the maximum of $\mu$ was 200–400 return units above µ at the logged actions, and σ at the maximiser was about 200 compared with less than 30 at the logged actions. The maximiser of $\mu - \kappa \sigma$ lay in the logged coolant range only for κ between 3 and 5.

## E. Computation

HOFLON-sGP does not require the AAE evaluation during online optimisation, and its solve time per step was shorter than that of HOFLON. All computations were run on one core of an Intel Core Ultra 5 135U processor. For both controllers, the solve time per step was far below the 30 min sampling interval.

## VI. DISCUSSION

Support terms. The AAE reconstruction error and the GP standard deviation can both penalise poorly represented state–action pairs, although neither is necessarily monotonic in distance to the training set. The GP term is expressed in return units, with κ a number of predictive standard deviations; a multiple of the GP standard deviation is used in the same way to tighten constraints in [29]. The effective strength of $\lambda _ { 1 }$ depends on the reconstruction scale, the reward scale and the critic. At the published value, the AAE term matters mainly where the critic is flat, which for a tree ensemble arises from its piecewise-constant predictions, including outside the training domain. Scaling it by $1 0 ^ { 4 }$ removed the failures of the tree critic but not those of the squared-exponential GP mean.

Actuator activity. The reward contains no actuation cost. With $\lambda _ { 2 } = 0 . 0 0 4 2$ , a change of 0.05 in $u _ { T _ { c } }$ costs $\lambda _ { 2 } \cdot 0 . 0 0 2 5$ which is small compared with differences of hundreds of return units between critic values. With the squared exponential critic and $\kappa ~ \leq ~ 1$ , the returns were high and the coolant temperature changed by several kelvin per step outside the logged range. In HOFLON with the XGBoost critic, smooth steady-state actuation results from the constant extrapolation of the tree ensemble. With the sGP critics, it was obtained for all tested $\kappa ~ \geq ~ 0 . 5$ with the Matern-5/2´ kernel and for $\kappa \geq 1 . 5$ with the squared-exponential kernel. The two kernels had similar accuracy on the held-out logged runs and different closed-loop behaviour for $\kappa < 3$

One-step improvement. Both controllers select actions through a one-step improvement objective relative to historical behaviour (Section II-C). At suitable settings, they can achieve returns above the best recorded historical run (12,327), but this does not hold for every critic or penalty setting. Further improvement would require the evaluation of the new policy and additional improvement steps, for example by fitted Q-evaluation with σ restricting the updates to the data support. This reintroduces bootstrapping.

Use in autonomous operation. HOFLON-sGP uses a single learned critic, which could be retrained when additional transitions become available. Its predictive standard deviation is also available online and could provide a supervisory indication of poorly supported actions [1], [2].

Limitations. The study uses one simulator with a fixed initial state, and the seeds vary only the measurement noise.

TABLE I  
CLOSED-LOOP RESULTS FOR FIVE CRITICS AND THREE NOISE REALISATIONS (SEEDS 11–13; 15 PAIRED RUNS PER CONFIGURATION). A RUN FAILS IF R < 6000. COOLANT MOVE: MEDIAN $| \Delta T _ { c } |$ PER STEP AFTER 20 H. OFFSETS: MEDIAN $| \bar { P }$ − 100| AND $| \bar { T }$ − 350| AFTER 50 H. WORST: LOWEST RETURN OVER THE RUNS. OFF-SUPPORT: MEDIAN SHARE OF STEPS OUTSIDE THE DATA SUPPORT DURING THE FIRST 20 H AND AFTER 20 H. THE HISTORICAL ROW DESCRIBES THE TRAINING DATA.
<table><tr><td></td><td>Fig. 4 Support term in the online objective</td><td>Failed</td><td>Median</td><td>Worst</td><td>Max. peak</td><td>Coolant</td><td colspan="2">Steady-state offset</td><td colspan="2">Off-support [%]</td></tr><tr><td>box</td><td></td><td>runs</td><td>R</td><td>R</td><td> $T \ [ \mathrm { K } ]$ </td><td>move [K]</td><td> $P \ [ \mathrm { k g } \mathrm { m } ^ { - 3 } ]$ </td><td>T [K]</td><td>0-20 h</td><td>&gt;20h</td></tr><tr><td>none</td><td colspan="8">HOFLON [3]: XGBoost critic  $\hat { Q } ,$  objective (3)</td><td></td><td></td><td></td></tr><tr><td>2 3</td><td></td><td>4/15</td><td>13,020</td><td>-2,782</td><td>374.7</td><td>0.02</td><td>0.79</td><td>0.05</td><td>45</td><td>0.0</td></tr><tr><td>4</td><td> $\mathrm { A A E , } \lambda _ { 1 } = 0 . 0 8 3 ( \mathrm { p u b l i s h e d , H O F L O N } )$ </td><td>0/15</td><td>13,393</td><td>12,411</td><td>363.6</td><td>0.02</td><td>0.58</td><td>0.04 0.04</td><td>35 22</td><td>0.0</td></tr><tr><td></td><td> $\mathrm { A A E } , \lambda _ { 1 } \times 1 0 ^ { 4 }$ </td><td>0/15</td><td>13,372</td><td>13,013</td><td>356.8</td><td>0.01</td><td>0.58</td><td></td><td></td><td>0.0</td></tr><tr><td colspan="9">HOFLON-sGP: sparse-GP critic with Matérn-5/2 kernel, objective (7)</td><td></td></tr><tr><td>5</td><td>κ = 0 (GP mean only)</td><td>13/15</td><td>-261,594</td><td>-261,752</td><td>567.7</td><td>0.00</td><td>435.03</td><td>83.58</td><td>80</td><td>100</td></tr><tr><td></td><td> $\kappa = 3 \ ( \mathrm { p r o p o s e d } )$ </td><td>0/15</td><td>13,464</td><td>13,398</td><td>355.7</td><td>0.02</td><td>0.41</td><td>0.20</td><td>7.5</td><td>0.0</td></tr><tr><td colspan="9">Sparse-GP critic with squared-exponential kernel, objective (7) with optional AAE term</td><td></td></tr><tr><td></td><td>κ = 0 (GP mean only)</td><td>15/15</td><td>4,267</td><td>3,703</td><td>359.6</td><td>5.86</td><td>16.42</td><td>1.49</td><td>80</td><td>100</td></tr><tr><td></td><td> $\kappa = 0 + \mathrm { A A E } , \lambda _ { 1 } = \stackrel { \cdot } { 0 } . 0 8 3$ </td><td>15/15</td><td>-3,351</td><td>-38,964</td><td>552.6</td><td>3.76</td><td>16.04</td><td>7.21</td><td>80</td><td>100</td></tr><tr><td>6</td><td> $\kappa = 0 + \mathrm { A A E } , \lambda _ { 1 } \times 1 0 ^ { 4 }$ </td><td>13/15</td><td>4,678</td><td>2,932</td><td>377.9</td><td>3.07</td><td>8.03</td><td>0.74</td><td>80</td><td>100</td></tr><tr><td></td><td>κ = 3</td><td>0/15</td><td>13,378</td><td>13,327</td><td>355.8</td><td>0.02</td><td>0.59</td><td>1.96</td><td>48</td><td>0.6</td></tr><tr><td>1</td><td>Historical PI (100 logged runs)</td><td>4/100</td><td>11,017</td><td>5,752</td><td>360.3</td><td>0.01</td><td>0.29</td><td>0.08</td><td>0</td><td>0</td></tr></table>

![](images/1c72975c4052cf9c727b2d9959e3c97d436bd7fa06a0f282ba8fb00dd253820d.jpg)  
Fig. 4. Per-run distributions: five critics × seeds 11–13 per controller (15 runs) and the 100 logged PI start-ups. Boxes span the quartiles, with the median as a black line and the mean as a triangle; whiskers reach 1.5 interquartile ranges. Red arrows count runs outside the plotted range. RMSEs use scaled errors; the dashed line marks 360 K.

Robustness to plant parameters and disturbances - especially nonstationary disturbances - has not been tested. The XG-Boost critics use the integrators and the sGP critics do not, and an XGBoost critic without integrators was not evaluated. Without integrated-error features, the action-selection law has no explicit integral compensation, and steady-state offsets can persist. The confidence level was selected with the squared-exponential critics on the same three noise seeds on which all results are reported. The two kernel implementations also use different inducing-input treatments and training schedules, so their closed-loop differences cannot be attributed to the kernel alone. The paper-machine benchmark of [3], with four manipulated variables and no measurements of internal states, remains to be evaluated; it would also provide a test of the required number of inducing points.

## VII. CONCLUSION

HOFLON is a one-step offline RL method, and in this study its tree-ensemble critic and AAE were replaced by one sparse GP critic with a lower-confidence-bound objective. With a Matern-5/2 critic and´ $\kappa = 3 .$ , all 15 runs over five critics and three noise realisations completed without failure. The median return was comparable to that of HOFLON and the lowest return higher, with smooth actuation and peak temperatures below 356 K. Without a support term, a large share of runs failed with both the tree-ensemble critic and the GP mean; in HOFLON-sGP, the σ term prevents these failures without a separate model. In HOFLON, the AAE term at its published weight changes the selected action mainly where the tree-ensemble critic is nearly constant.

![](images/0d9be29efaca54060657affe8590aad74602b599062df9caf034a271adf98a54.jpg)  
Fig. 5. Effect of the confidence level κ (five critics per kernel, seeds 11–13; line: median, band: range; stars: $\kappa = 3 )$ . Matern-5/2 critics for´ $\kappa \geq 0 . 5 ;$ $\kappa = 0$ in Section V-B. Coolant move: median $| \Delta T _ { c } |$ per step after 20 h. Polymer offset: $| \bar { P } - 1 0 0 |$ after 50 h. Off-support: share of steps after 20 h. Dashed: HOFLON on the same runs.

## ACKNOWLEDGMENT

The authors used Claude Opus 5.5 (Anthropic) to draft and revise parts of the text, mainly in the abstract, Section IV-D and Sections V–VI, and to check the consistency of the reported results. All content was reviewed and edited by the authors, who take full responsibility for the paper.

## REFERENCES

[1] F. Borghesan, M. Zagorowska, and M. Mercangoz, “Unmanned and¨ autonomous systems: Future of automation in process and energy industries,” IFAC-PapersOnLine, vol. 55, no. 7, pp. 875–882, 2022.

[2] M. Baldea, A. T. Georgiou, B. Gopaluni, M. Mercangoz, C. C. Pan-¨ telides, K. Sheth, V. M. Zavala, and C. Georgakis, “From automated to autonomous process operations,” Comput. Chem. Eng., vol. 196, p. 109064, 2025.

[3] A. Durkin, J. Stolte, and M. Mercangoz, “HOFLON: Hybrid offline¨ learning and online optimization for process start-up and gradetransition control,” Comput. Chem. Eng., vol. 207, p. 109566, 2026, arXiv:2510.03830.

[4] A. Ahmed, E. A. del R´ıo-Chanona, and M. Mercangoz, “Comparative ¨ study of machine learning and system identification for process systems engineering dynamics,” Ind. Eng. Chem. Res., vol. 64, no. 8, pp. 4450–4478, 2025.

[5] S. Levine, A. Kumar, G. Tucker, and J. Fu, “Offline reinforcement learning: Tutorial, review, and perspectives on open problems,” arXiv:2005.01643, 2020.

[6] A. Kumar, J. Fu, M. Soh, G. Tucker, and S. Levine, “Stabilizing offpolicy Q-learning via bootstrapping error reduction,” in Advances in Neural Information Processing Systems, vol. 32, 2019.

[7] S. Fujimoto, D. Meger, and D. Precup, “Off-policy deep reinforcement learning without exploration,” in Proc. Int. Conf. Machine Learning (ICML), 2019, pp. 2052–2062.

[8] J. M. Lee, N. S. Kaisare, and J. H. Lee, “Choice of approximator and design of penalty function for an approximate dynamic programming based control approach,” J. Process Control, vol. 16, no. 2, pp. 135– 156, 2006.

[9] Y. Jin, Z. Yang, and Z. Wang, “Is pessimism provably efficient for offline RL?” in Proc. Int. Conf. Machine Learning (ICML), 2021, pp. 5084–5096.

[10] T. Yu, G. Thomas, L. Yu, S. Ermon, J. Zou, S. Levine, C. Finn, and T. Ma, “MOPO: Model-based offline policy optimization,” in Advances in Neural Information Processing Systems, vol. 33, 2020.

[11] G. An, S. Moon, J.-H. Kim, and H. O. Song, “Uncertainty-based offline reinforcement learning with diversified Q-ensemble,” in Advances in Neural Information Processing Systems, vol. 34, 2021.

[12] S. K. S. Ghasemipour, S. S. Gu, and O. Nachum, “Why so pessimistic? Estimating uncertainties for offline RL through ensembles, and why their independence matters,” in Advances in Neural Information Processing Systems, vol. 35, 2022.

[13] A. Durkin, J. Stolte, M. Jones, R. Pitchumani, B. Li, C. Michler, and M. Mercangoz, “Safe deployment of offline reinforcement learning via¨ input convex action correction,” arXiv:2507.22640, 2025.

[14] D. Brandfonbrener, W. F. Whitney, R. Ranganath, and J. Bruna, “Offline RL without off-policy evaluation,” in Advances in Neural Information Processing Systems, vol. 34, 2021.

[15] A. Makhzani, J. Shlens, N. Jaitly, I. Goodfellow, and B. Frey, “Adversarial autoencoders,” arXiv:1511.05644, 2015.

[16] I. Kostrikov, A. Nair, and S. Levine, “Offline reinforcement learning with implicit Q-learning,” in Proc. Int. Conf. Learning Representations (ICLR), 2022.

[17] Y. Engel, S. Mannor, and R. Meir, “Reinforcement learning with Gaussian processes,” in Proc. Int. Conf. Machine Learning (ICML), 2005, pp. 201–208.

[18] T. Savage, D. Zhang, M. Mowbray, and E. A. del Rio Chanona, “Model-free safe reinforcement learning for chemical processes using Gaussian processes,” IFAC-PapersOnLine, vol. 54, no. 3, pp. 504–509, 2021.

[19] E. A. del Rio Chanona, J. E. Alves Graciano, E. Bradford, and B. Chachuat, “Modifier-adaptation schemes employing Gaussian processes and trust regions for real-time optimization,” IFAC-PapersOnLine, vol. 52, no. 1, pp. 52–57, 2019.

[20] E. A. del Rio Chanona, P. Petsagkourakis, E. Bradford, J. E. Alves Graciano, and B. Chachuat, “Real-time optimization meets Bayesian optimization and derivative-free optimization: A tale of modifier adaptation,” Comput. Chem. Eng., vol. 147, p. 107249, 2021.

[21] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016, pp. 785–794.

[22] D. P. Bertsekas, Reinforcement Learning and Optimal Control. Belmont, MA: Athena Scientific, 2019.

[23] M. Titsias, “Variational learning of inducing variables in sparse Gaussian processes,” in Proc. Int. Conf. Artificial Intelligence and Statistics (AISTATS), 2009, pp. 567–574.

[24] N. Srinivas, A. Krause, S. M. Kakade, and M. Seeger, “Gaussian process optimization in the bandit setting: No regret and experimental design,” in Proc. Int. Conf. Machine Learning (ICML), 2010, pp. 1015–1022.

[25] R. Laroche, P. Trichelair, and R. Tachet des Combes, “Safe policy improvement with baseline bootstrapping,” in Proc. Int. Conf. Machine Learning (ICML), 2019, pp. 3652–3661.

[26] S. Kakade and J. Langford, “Approximately optimal approximate reinforcement learning,” in Proc. Int. Conf. Machine Learning (ICML), 2002, pp. 267–274.

[27] B. R. Maner and F. J. Doyle III, “Polymerization reactor control using autoregressive-plus Volterra-based MPC,” AIChE J., vol. 43, no. 7, pp. 1763–1784, 1997.

[28] J. R. Gardner, G. Pleiss, D. Bindel, K. Q. Weinberger, and A. G. Wilson, “GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration,” in Advances in Neural Information Processing Systems, vol. 31, 2018.

[29] L. Hewing, J. Kabzan, and M. N. Zeilinger, “Cautious model predictive control using Gaussian process regression,” IEEE Trans. Control Syst. Technol., vol. 28, no. 6, pp. 2736–2743, 2020.