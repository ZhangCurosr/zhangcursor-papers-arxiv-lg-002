# Simulation Priors for Rapid Adaptation in Model-Based RL for Autonomous Racing

Adrian Remonda<sup>1,2,†</sup>, Jiajun Xi<sup>2</sup>, Nicklas Hansen<sup>2</sup>, Marcus Greiff<sup>1</sup>, Thomas Lew<sup>1</sup>, John Talbot<sup>1</sup>, John Subosits<sup>1</sup>, and Xiaolong Wang<sup>2</sup>

![](images/150b560fb60069a049984d69e03865f9111ae230313e16d5d5e944af6c4349df.jpg)  
Fig. 1: Prior-Informed Adaptation (PIA) pretrains a world model on heterogeneous simulation priors, then performs few-shot online adaptation in the target domain. Across racing tasks, PIA adapts faster than baselines and deploys on a vehicle despite sim-to-real mismatch.

Abstract— Reinforcement learning enables learning highperformance control policies, but its reliance on extensive interaction limits deployment on physical systems where data collection is costly, and failures are unsafe. We introduce Prior-Informed Adaptation (PIA), an offline-to-online model-based RL framework that uses structured simulation priors for rapid adaptation. PIA pretrains a world model on heterogeneous lowand high-fidelity simulated trajectories, then adapts it with limited target-domain interaction using uncertainty-regularized planning. Rather than requiring an accurate simulator for policy transfer, PIA treats simulated trajectories as biased priors whose residual mismatch is corrected during online adaptation. We evaluate PIA in autonomous racing using high-fidelity simulation and on a real full-size autonomous race car. In simulation, PIA adapts faster across track and surface shifts than strong baselines, achieving lower lap times and fewer failures. On hardware, PIA achieves reliable few-shot adaptation with less than one hour of on-vehicle interaction.

## I. INTRODUCTION

Reinforcement learning (RL) has emerged as an effective data-driven framework for learning high-performance control policies, but its strongest successes often rely on large quantities of interaction data [1, 2]. For example, [3] demonstrated that an RL policy can outrun the world’s best human drivers in Gran Turismo, but only after 45,000 hours of online interaction. This data requirement limits RL in settings where interaction is costly, failures are unsafe, or tasks change over time, such as robotics and real-world driving [4].

As a result, RL practitioners often face a tradeoff between dataset diversity and interaction budget. Offline RL is attractive due to its reliance on existing data, but may lead to (unsafe) extrapolation errors when the policy encounters out-of-distribution states [5, 6]. Such errors can be corrected with additional data collected by online RL [7–10], but that too can be unsafe. In domains such as high-speed racing, where mistakes can be costly, this tradeoff often produces one of two undesirable outcomes: systems that require thousands of laps to achieve stable performance, which is unrealistic for real-world racing, or controllers that rely on highly accurate physics models, which limit flexibility and may potentially lead to suboptimal policies. The challenge is to exploit structured but imperfect prior data while adapting to new geometries or dynamics under strict interaction budgets.

Prior work, such as Dyna-Q [11–13], has shown that synthetic rollouts can significantly improve sample efficiency in model-based RL (MBRL) relative to model-free RL methods such as SAC [14]. However, when rollouts are generated from learned dynamics, compounding prediction error can make long trajectories unreliable, which in practice restricts their usefulness to short rollouts. As a result, many MBRL methods rely on frequent replanning or conservative objectives to remain reliable. High-fidelity simulators used in control and robotics, such as MuJoCo [15] and NVIDIA Isaac Gym [16], provide a complementary source of synthetic experience: they can cheaply generate structured long-horizon trajectories, but depending on the simulator fidelity, may limit transfer to the target system. This motivates viewing data augmentation not simply as adding more data, but as simulation priors: structured data generators that complement learned world models rather than replacing them.

![](images/03988c004624471367be3d73b791f5f7a567b2180f415d073df6f24e6e79f064.jpg)  
Fig. 2: Sequences from real-world deployments. After adaptation (bottom), PIA completes the oval racing task.

In this work, we study how heterogeneous simulated trajectories can initialize modern MBRL when they do not exactly match deployment. We introduce Prior-Informed Adaptation (PIA), an offline-to-online MBRL framework that pretrains a world model on structured simulator data and then adapts using limited target-domain interaction. As shown in Figure 1, a mixed replay retains broad prior coverage while updating the model with newly collected target data, and uncertaintyregularized planning discourages predictions associated with high epistemic uncertainty. Simulation is therefore used as an imperfect data prior rather than as an environment from which a policy transfers directly. This avoids the need for an exact simulator or a domain-randomized distribution [17] that fully covers the target domain.

We evaluate PIA in an autonomous racing task, where track geometry is typically available before deployment and can be simulated, while the vehicle dynamics remain difficult to reproduce accurately due to effects such as actuator latency, sensing/estimation mismatch, engine dynamics, changing weather, and tire variation. The Assetto Corsa racing simulator (ACGym) [18], shown in Figure 3, provides a controlled highfidelity proxy for real-world racing, allowing us to study rapid adaptation across unseen tracks and surface changes before hardware deployment. We then deploy the same offline-toonline pipeline on a full-size autonomous race car, where simulated trajectories are combined with a small amount of limited real hardware interaction to anchor the learned world model (see Figure 2). Extensive evaluations show that our proposed PIA method requires significantly fewer trials than other MBRL methods. Our main contributions are:

• A simulation-prior regime for offline-to-online MBRL: PIA uses heterogeneous trajectories as an imperfect offline initialization, then updates the learned world model with limited target-domain interaction rather than requiring direct policy transfer from an accurate simulator.

• Adaptation rate and robustness in simulation: We conduct experiments in ACGym demonstrating that PIA improves adaptation rate, with lower lap times and fewer failures than offline-only and offline-to-online baselines.

• Full-size autonomous racing deployment: To our knowl edge, we provide the first demonstration of offline-to-online adaptation of a simulation-pretrained world model on a fullsize autonomous race car, using less than one hour of real-car interaction.

• Reproducible benchmark: We release code and datasets covering 15 tracks and multiple surface conditions, including more than 740 hours of Assetto Corsa driving data and low-fidelity simulated rollouts, enabling future research on few-shot and sim-to-real adaptation.

Overall, our results show that structured simulated trajectories can substantially reduce the amount of target-domain interaction required for online adaptation.

## II. RELATED WORK

Model-Based Augmentation. Since Dyna-Q [11], synthetic rollouts from learned dynamics have been widely explored (e.g., MBPO [12], MOPO [13], Dreamer [19], and TD-MPC2 [20]), and have shown that model-based rollouts can significantly improve sample efficiency. However, learned rollouts are prone to compounding error, so they are typically kept short to remain reliable. Physics-informed models such as PhIHP [21] mitigate this by combining analytic priors with residual corrections, but are still constrained to short rollouts. AnyCar [22] explores the use of transformers for world model generation and zero-shot deployment. While impressive, learned-model approaches remain tied to model fidelity when used for direct transfer.

Offline RL and Adaptation. Offline RL methods such as CQL and IQL [5, 6, 23] mitigate extrapolation error through conservative or implicit constraints on value learning, but this can limit generalization. H2O [24] combines a fixed offline real-world dataset with online rollouts from a simulator using dynamics-aware value regularization and corrected Bellman updates. PIA considers the complementary direction: it pretrains from fixed simulator trajectories and adapts through limited online interaction with the target environment. [7, 9, 10]. FOWM [8] pretrains a world model offline on targetdomain data and finetunes it with limited online interaction using uncertainty-regularized planning. To mitigate distribution shift when learning, FOWM augments TD-MPC with a V-AWR objective that biases learning toward actions supported by the offline data. PIA follows the same offline-to-online template, but targets a different transfer setting: its offline initialization is constructed from heterogeneous simulation priors. Rather than using the V-AWR offline objective, PIA retains the TD-MPC2 training objective and conditions the model on task context to distinguish heterogeneous dynamics. Thus, the central problem is not only offline-to-online fine tuning under task or policy shift, but cross-domain correction of the simulator’s bias with limited target-domain interaction. Sim-to-real Adaptation. Domain randomization trains policies over randomized simulator dynamics to improve transfer under simulator mismatch [17]. Rapid motor adaptation instead trains a base policy and adaptation module in simulation, enabling fast online adaptation to unseen deployment conditions [25]. These methods are most effective when the simulated variation covers the relevant target-domain variation. More broadly, sim-to-real systems may incorporate real data for system identification, residual modeling, or validation; Kaufmann et al. [26] improve drone racing by augmenting simulation with data-driven residual perception and dynamics models identified from real-world experience. PIA is complementary: simulated trajectories are treated as biased replay priors rather than as the final training environment, and their residual mismatch is corrected with target-domain interaction during offline-to-online MBRL.

Autonomous Racing. Model Predictive Control (MPC) delivers strong performance on full-size autonomous race cars but requires carefully engineered models [4, 27–30]. At the limits of handling, however, small residual mismatches in tire grip or actuator response can cause failures, motivating controllers that can update from physical interaction [31]. RL systems have achieved strong autonomous-racing results in simulation with large interaction budgets, including GT Sophy in Gran Turismo [3, 32], Formula RL [33], and AssettoCorsaGym [18], but these works do not address fewshot physical adaptation on a full-size race car. Closest to our hardware setting, Djeumou et al. [34] train RL policies in a neural-SDE vehicle model learned from real driving data and deploy them zero-shot for a drifting task. PIA instead adapts a simulation-pretrained world model using limited physical interaction. To our knowledge, none of the cited literature has demonstrated this offline-to-online adaptation regime on a full-size autonomous race car.

## III. PRELIMINARIES

In this section, we introduce notation and terminology that will appear throughout the paper. We formulate the autonomous racing problem as a Partially Observable Markov Decision Process (POMDP) $\langle s , \mathcal { A } , \mathcal { O } , \mathcal { T } , r , \gamma \rangle$ , where $s$ denotes the state space, $\mathcal { A }$ is the action space, O is the observation space, $\mathcal { T } \colon \mathcal { S } \times \mathcal { A } \mapsto \mathcal { S }$ is the unknown transition model, $r \colon S \times A \mapsto$ R is the reward function, and $\gamma \in [ 0 , 1 )$ is the discount factor [35, 36]. We approximate states as a sequence of the k most recent observations. A policy π selects actions $a _ { t } \sim \pi ( \cdot | s _ { t } )$ at each time step t to maximize the expected discounted return $\begin{array} { r } { \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) \right] } \end{array}$ ]. We assume that trajectories from the source and target environment share a compatible action space and observation interface, or can be mapped into one.

Model-Based RL. In MBRL, the agent learns a predictive model of environment dynamics and leverages it for planning. This improves sample efficiency over model-free methods and enables sim-to-real transfer. One such example is TD-MPC2 [20], which learns a latent world model from sequential data and performs short-horizon planning with MPC. Its end-to-end training and latent-space planning make it wellsuited for multitask learning and diverse datasets. Its world model consists of an encoder $h _ { \theta } .$ , latent dynamics $d _ { \theta }$ , reward predictor $R _ { \theta }$ , and value predictor $Q _ { \theta } .$ Given trajectory segments sampled from the replay buffer $B ,$ , these components are jointly optimized according to

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } ) = \mathbb { E } _ { \tau \sim p _ { \mathcal { B } } } \left[ \ell _ { \mathrm { T D - M P C 2 } } ( \boldsymbol { \theta } ; \tau ) \right] , } \end{array}\tag{1}
$$

where $\tau = \{ ( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } ) \} _ { t = 0 } ^ { T - 1 }$ is a trajectory segment of length $T$ sampled from the replay distribution $p _ { B }$ , and $\ell _ { \mathrm { T D - M P C 2 } }$ comprises latent consistency, reward prediction, and temporal-difference value prediction losses. The policy prior π<sub>θ</sub> is optimized separately from $\left( h _ { \theta } , d _ { \theta } , R _ { \theta } , Q _ { \theta } \right)$ . For a candidate action sequence $a _ { 0 : H - 1 : }$ , latent dynamics $z _ { h + 1 } =$ $d _ { \theta } \big ( z _ { h } , a _ { h } \big )$ are predicted over a horizon $h = 0 , \ldots , H - 1$ with $z _ { 0 } ~ = ~ h _ { \theta } ( s _ { t } )$ . The planner scores each rollout using predicted rewards and a terminal value $Q _ { \theta } \big ( z _ { H } , a _ { H } \big )$ , where $a _ { H } = \pi _ { \theta } ( z _ { H } )$ is provided by the learned policy prior.

Simulation Priors. We study rapid adaptation in a target environment where online interaction is limited, and failures may be costly. Before adaptation, the agent has access to offline trajectories from source simulators that provide useful structure and coverage, but are not assumed to match the target dynamics. We refer to these sources as simulation priors. The goal is to use these biased but structured trajectories to enable more rapid adaptation with limited target-domain interaction. Offline RL and Adaptation. Given a dataset $\mathcal { D } _ { \mathrm { o f f } } .$ , offline RL seeks to learn $\pi$ without additional interaction [23]. The main challenge is out-of-distribution generalization when π reaches unseen states. A common strategy to mitigate this issue is offline pretraining followed by online adaptation, where a world model is initialized on $\mathcal { D } _ { \mathrm { o f f } }$ and refined with limited new interaction dataset $\mathcal { D } _ { \mathrm { o n } }$ (see [8]).

ODE Model. For low-fidelity simulations, we use a simple vehicle model described by ordinary differential equations $\dot { \mathbf { x } } = g ( \mathbf { x } , \mathbf { u } )$ , where x is the vehicle state, u the control inputs. We use a dynamic bicycle model with aerodynamics loads, drivetrain dynamics, and tire forces. [37]

Assetto Corsa Gym. For high-fidelity simulation, we use ACGym [18], an open-source framework built on the Assetto Corsa simulator. It provides more realistic vehicle dynamics and geometries across several cars and laser-scanned tracks.

## IV. PIA: PRIOR-INFORMED ADAPTATION

Simulation data is cheap to generate, whereas real-world data is costly and potentially dangerous to collect. This motivates using simulation as a prior. PIA treats simulation priors as useful but mismatched data for initializing an MBRL world model: simulated trajectories supply broad structured coverage, limited target-domain interaction corrects their residual bias, and uncertainty-regularized planning keeps control conservative while that correction is still incomplete. PIA has two components:

1) mixed replay, context-aware, offline-to-online training, which updates the world model from a mixture of simulation-prior and target-domain transitions;

2) uncertainty-regularized latent planning, which discourages the controller from exploiting predictions in regions poorly supported by target-domain data

While each component has prior work, their combination enables PIA’s efficient offline-to-online performance.

Mixed Replay, Context-Aware, Offline-to-Online Training. First, PIA trains the policy from replay buffers containing heterogeneous trajectories from multiple sources. In our experiments, the simulation-prior data consist of low-fidelity ODE rollouts and high-fidelity Assetto Corsa trajectories.

Second, all data is mapped to a common transition format and labeled with task and source context, denoted by $c ,$ to distinguish data sources such as ODE rollouts, Assetto Corsa simulation, and real-car interaction, as well as the corresponding track or deployment setting. When a channel is unavailable in one source, we estimate it from available variables when possible, and otherwise set it to zero.

Third, PIA consists of training on a mixed replay buffer containing offline and online data (see Alg. 1). The offline buffer $B _ { \mathrm { o f f } }$ contains simulation-prior trajectories and any target-domain data available before adaptation. The online buffer $B _ { \mathrm { o n } }$ contains target-domain data collected online and used for adaptation. After collecting an episode, PIA samples from a mixture $q _ { \alpha } = \alpha p _ { \mathrm { o f f } } + ( 1 - \alpha ) p _ { \mathrm { o n } }$ , where $p _ { \mathrm { o f f } }$ and $p _ { \mathrm { o n } }$ denote uniform distributions over $B _ { \mathrm { o f f } }$ and $B _ { \mathrm { o n } } \ [ 8 - 1 0 ]$ The TD-MPC2 model is trained on context-aware trajectory segments sampled from the mixed replay distribution,

$$
\begin{array} { r } { \mathcal { L } _ { \alpha } ( \theta ) = \mathbb { E } _ { ( \tau , c ) \sim q _ { \alpha } } \left[ \ell _ { \mathrm { T D - M P C 2 } } ( \theta ; \tau , c ) \right] , } \end{array}\tag{2}
$$

where c denotes the context associated with trajectory segment τ, encoding its task and data source. The policy prior is optimized separately as in TD-MPC2, but with additional context conditioning. For pretraining, we let $\alpha = 1$ , which recovers the TD-MPC2 formulation. For online adaptation, we anneal α from 1 to 0.5 following a linear decay schedule. This combination of simulation-prior data and target-domain data ensures training a model that benefits from both high data coverage and limited but high-quality target-domain data. Uncertainty-Regularized Latent Planning. While collecting target-domain data, the agent must act while this corrective data is still incomplete. Early on during adaptation, value function estimates may reflect simulator-prior bias. To mitigate this, PIA uses FOWM-style uncertaintyregularized latent planning, with epistemic uncertainty estimated from Q-ensemble disagreement [8]. We extend the planner to be context-aware. Given state $s _ { t }$ and context c the encoder becomes $z _ { 0 } ~ = ~ h _ { \theta } ( s _ { t } , c )$ and the terminal action is $a _ { H } ~ = ~ \pi _ { \theta } ( z _ { H } , c )$ . Following [8], define $u _ { h } \ = \ \mathrm { S t d } ( \{ Q _ { \theta } ^ { ( i ) } ( z _ { h } , a _ { h } , c ) \} _ { i = 1 } ^ { K } )$ , where $\bar { Q } _ { \theta } ( z _ { h } , a _ { h } , c ) =$ $\begin{array} { r } { \frac { 1 } { K } \sum _ { i = 1 } ^ { K } Q _ { \theta } ^ { ( i ) } ( z _ { h } , a _ { h } , c ) } \end{array}$ denotes the ensemble mean. The planner scores each rollout by predicted rewards, terminal value, and uncertainty,

```latex
Algorithm 1 PIA: Offline-to-Online Adaptation
Inputs: offline data $\mathcal { D } _ { \mathrm { o f f } } .$ , target environment E, pretraining steps $N _ { \mathrm { p r e } } ,$
online episodes $N _ { \mathrm { e p } } .$ , batch size b, replay ratio α
$B _ { \mathrm { o f f } }  \dot { \mathcal { D } } _ { \mathrm { o f f } } ; \quad B _ { \mathrm { o n } } ^ { \ast }  \emptyset$
Offline pretraining
for $t = 1$ to $N _ { \mathrm { p r e } }$ do
$B $ sample $( \boldsymbol { B } _ { \mathrm { o f f } } , \boldsymbol { b } )$
Update θ by minimizing $\mathcal { L } _ { \alpha } ( \theta )$ over B
end for
Online Adaptation
for $e = 1$ to $N _ { \mathrm { e p } }$ do
Sample $\tau _ { e } = \stackrel { \triangledown ^ { - } } { \{ ( s _ { t } , a _ { t } , r _ { t } , s _ { t + 1 } , c ) \} } _ { t = 0 } ^ { T _ { e } - 1 }$ from E
$\alpha  \alpha - ( 2 \hat { N _ { \mathrm { e p } } } ) ^ { - 1 }$
$B _ { \mathrm { o n } }  B _ { \mathrm { o n } } \cup \dot { \{ \tau _ { e } \} }$
$B \gets$ sample $( \hat { { \cal B } _ { \mathrm { o f f } } } , \lfloor \alpha b \rfloor ) \lfloor$ ∪ sample $( \boldsymbol { B } _ { \mathrm { o n } } , \boldsymbol { b } - \lfloor \alpha \boldsymbol { b } \rfloor )$
Update θ by minimizing $\dot { \mathcal { L } } _ { \alpha } ( \theta )$ over B
end for
```

$$
\begin{array} { l } { { \displaystyle { \cal G } _ { \theta } ( z _ { 0 } , a _ { 0 : H - 1 } , c ) = \sum _ { h = 0 } ^ { H - 1 } \gamma ^ { h } \left( \hat { r } _ { \theta } ( z _ { h } , a _ { h } , c ) - \lambda u _ { h } \right) } } \\ { { \displaystyle ~ + \gamma ^ { H } \left( \bar { Q } _ { \theta } ( z _ { H } , a _ { H } , c ) - \lambda u _ { H } \right) . } } \end{array}\tag{3}
$$

where λ is the uncertainty penalty coefficient.

## V. EXPERIMENTS IN SIMULATION

First, we extensively evaluate PIA for autonomous racing in simulation using Assetto Corsa as the high-fidelity simulator and the simple ODE vehicle model as the low-fidelity simulation prior. We test whether simulation-prior pretraining improves performance and safety on unseen tracks and remains robust under surface-grip shifts.

We evaluate in ACGym on 9 tracks total: 6 Assetto Corsa training tracks and three held-out test tracks, Indianapolis (IND), Red Bull Ring (RBR), and Silverstone (SIL). We distinguish two settings. PIA-AC is pretrained only on AC trajectories from source tracks, and therefore tests transfer to held-out track geometry. PIA-ODE additionally receives low-fidelity ODE trajectories generated on the deploymenttrack geometry, but no high-fidelity AC trajectories from that track; it therefore tests cross-fidelity adaptation with known geometry and mismatched target dynamics. Surface experiments keep the SIL geometry fixed and shift grip among optimal, varying, and low grip conditions. Episodes last up to 15k environment steps and terminate early if the car stalls or leaves the track. We report the best lap time. Safety and stability are captured by the number of failures and miles per failure. An incident is an off-track excursion or lowspeed stall, and miles per incident is the total driving distance divided by the number of incidents.

TABLE I: Performance. Best lap time (s). IQL, SAC scratch, FOWM, PIA: 3 seeds. Avg. only if all tasks completed.
<table><tr><td>Task</td><td>IQL</td><td colspan="3">SAC FOWM PIA-AC PIA-ODE</td></tr><tr><td>IND</td><td>159.33</td><td>60.00</td><td>60.19 59.97</td><td>60.00</td></tr><tr><td>RBR</td><td>Fail</td><td>255.64</td><td>87.83 85.85</td><td>84.34</td></tr><tr><td>SIL</td><td>209.95</td><td>Fail</td><td>118.88 114.16</td><td>112.17</td></tr><tr><td>SILvarying</td><td>Fail</td><td>Fail</td><td>122.82 114.94</td><td>112.81</td></tr><tr><td>SILlow</td><td>Fail</td><td>Fail</td><td>125.85 119.79</td><td>117.61</td></tr><tr><td>Average</td><td>一</td><td></td><td>103.11 98.94</td><td>97.39</td></tr></table>

![](images/a6fa02976cb4cbe9108d5088e88e50884d6b32cd455bd7c925e644ead912e33e.jpg)  
Fig. 3: Tracks. Left: The F317 car and Assetto Corsa. Right: Tracks considered in the ACgym experiments.

Datasets. We construct two sources of pretraining data. (1) AC (high-fidelity). We train SAC from scratch for 6M steps per track in ACGym to generate expert-level trajectories. These datasets cover all training splits (1T–6T). (2) ODE (lowfidelity). We introduce a lightweight ODE environment that shares the same observation and action interface as ACGym. We run the same SAC agent used for AC, but replace the physics step with the ODE model while preserving geometry and localization signals. The target-track ODE trajectories are intentional prior information in PIA-ODE; they provide geometry and approximate dynamics. PIA-ODE is a knowngeometry cross-fidelity setting, whereas PIA-AC is the heldout-geometry setting.

Baselines. We compare online SAC [14], offline IQL [6], offline-to-online FOWM [8], and our PIA variants, PIA-AC and PIA-ODE. FOWM and PIA-AC use identical offline buffers and update budgets. We include FOWM as an end-toend baseline using its original TD-MPC formulation. Because FOWM and PIA differ in both learning objective and backbone, this comparison measures system-level performance rather than isolating an individual component. We evaluate two variants. PIA-AC is pretrained on six AC training tracks, and tests transfer to tracks absent from pretraining. PIA-ODE uses the same AC data and adds ODE rollouts on the three tracks held out from AC pretraining, matching realistic racing workflows in which the circuit is rehearsed in simulation before deployment while the target dynamics are uncertain.

TABLE II: Miles per failures. Results across tracks for SAC scratch, FOWM, and PIA. 3 seeds. Higher is better.
<table><tr><td>Task</td><td colspan="3">SAC FOWM PIA-AC PIA-ODE</td></tr><tr><td>IND</td><td>80.73 714.68</td><td>1194.29</td><td>1233.25</td></tr><tr><td>RBR</td><td>0.39 39.14</td><td>26.40</td><td>34.48</td></tr><tr><td>SIL</td><td>0.28 32.06</td><td>34.62</td><td>68.80</td></tr><tr><td>SILvarying</td><td>0.02 19.54</td><td>42.68</td><td>46.57</td></tr><tr><td>SILlow</td><td>0.02 16.50</td><td>16.48</td><td>25.57</td></tr><tr><td>Average</td><td>16.29 164.38</td><td>262.09</td><td>281.73</td></tr></table>

a) Adaptation: Tables I and II show that IQL and SAC struggle on unseen tasks, while FOWM adapts but remains slower and less safe than PIA. PIA-ODE achieves the best average lap time and miles per incident. The improvements are most evident on complex tracks and low-grip surfaces. Even on simpler tasks such as IND with small lap-time gains, the low-fidelity ODE data augmentation improves safety by increasing the miles per failure metric.

b) Robustness to Domain Shift: To assess robustness, we fine-tune on varying and low grip surfaces on the SIL track. These conditions change grip, while ODE rollouts use nominal vehicle parameters without surface-specific friction variation. This is critical in racing, where grip varies with weather, and controllers that overfit to a single surface fail when conditions change. As shown in Tables II and I and Figure 4, all methods degrade relative to optimal-surface evaluation; SAC fails, while PIA-ODE maintains lower lap times and higher miles per incident than AC-only and FOWM on both low-grip variants. This suggests that nominal ODE rollouts provide transferable structure, although varying ODE friction parameters may further improve robustness.

c) Additional analyses and ablations: Appendix VIII-.2.a separates the roles of ODE and AC data. The ODEsignal ablation in Appendix VIII-.1 directly addresses the known-geometry setting: retaining only explicit geometry channels improves safety but not adaptation speed, while masking those channels and retaining dynamics signals does not improve performance; combining both produces the strongest result. Removing uncertainty regularization leads to substantially worse safety, and task and context identifiers improve generalization when mixing heterogeneous data.

## VI. VEHICLE DEPLOYMENT

The previous experiments isolate the role of augmentation in a controlled sim-to-sim transfer setting. We now evaluate the same adaptation pipeline on a full-size autonomous race car. We conduct a progressive real-car adaptation study on an oval race track (see Figure 6). The goal is to drive the vehicle at the handling limit while respecting a strict out-oftrack (OOT) constraint. At deployment stage k, the agent is initialized from synthetic physics rollouts, Assetto Corsa data, and real-car trajectories collected before stage k. The resulting checkpoint is deployed using the same offline/online mixing protocol as in simulation. Data collected during stage k is used for online fine-tuning during that stage and is added to the offline buffer for subsequent stages. The initial 11 minutes of real-car seed data were collected using predefined maneuvers, including full-throttle acceleration and circular driving at different speeds. We use a deterministic MPC controller as a performance reference [4]. We measure episodes to first success, success rate, best valid time under the OOT, average maximum OOT distance, and average speed. Failure-mode counts and adaptation progress are shown in Figure 5.

![](images/daf313c1a12b75b540d16c410bdbd05af5e0212006cdec04cb6d3f85a5042523.jpg)  
Fig. 4: Adaptation performance. Best valid lap time by episode. Mean and ±1 standard deviation over three seeds.

TABLE III: Progressive real-car adaptation. Each deployment stage is initialized from simulation priors plus accumulated real-car data collected before that stage. Real data is the amount of real-car data (HH:MM) available before that stage; Real % is its percentage of the pretraining buffer; Succ. % is the episode index of the first episode classified as successful; Best time is the fastest valid sector time with max OOT < 1m and no Did-not-finish (DNF).
<table><tr><td>Run</td><td>data</td><td>Real Real %</td><td>Ep. to 1st success</td><td>Succ. %</td><td>Best time (s) OOT (m)</td><td>Mean max OOT (m)</td><td>Mean speed (m/s)</td></tr><tr><td>s1</td><td>00:11 0.28</td><td></td><td>5</td><td>37.5 (3/8)</td><td>FAIL (−)</td><td>1.53 ± 0.09 10.99 ± 3.68</td><td></td></tr><tr><td>s2</td><td>00:180.46</td><td></td><td>2</td><td></td><td></td><td>62.5 (5/8) 51.43 (0.13) 0.30 ± 0.29 14.63 ± 0.98</td><td></td></tr><tr><td>s3</td><td>00:391.02</td><td></td><td>1</td><td></td><td></td><td>75.0 (6/8) 50.18 (0.17) 0.26 ± 0.11 13.57 ± 5.63</td><td></td></tr><tr><td>s4</td><td>00:441.13</td><td></td><td>1</td><td>100 (8/8)</td><td>49.55 (0.30) 0.32 ± 0.29 15.38 ± 0.94</td><td></td><td></td></tr><tr><td>MPC</td><td>1</td><td>1</td><td>1</td><td>100.0</td><td>52.47 (0.87)</td><td>0.87</td><td>14.30</td></tr></table>

Table III shows a clear real-car adaptation progression. With 11 minutes of accumulated real-car data, s1 does not yet produce a valid OOT-constrained time and succeeds in only 37.5% of evaluation episodes. After adding additional realcar data, s2 achieves its first success in episode 2, compared

![](images/3a7b180515792237b64241315d2ebf7289a012903c6d09c655031a5f7534ec8c.jpg)  
Fig. 5: Episode outcomes. Distribution of successful episodes and failures during real-car adaptation.

with episode 5 for s1, and reaches a valid time comparable to the baseline MPC. Stages s3 and s4 succeed on their first episode; by s4, the policy reaches 100% success and improves both the best valid time and average speed. The overall progression shows that just a small 45-minute realcar anchor is sufficient to align the synthetic data replay buffer for reliable physical deployment. Figure 5 shows how the outcome distribution changes across deployment stages. As more real-car data is added, failures decrease, and successful evaluations become more frequent. In addition, we tested PIA using only real-car data, without simulation priors. Deployment exhibited erratic control and was terminated for safety before completion. This highlights the practical value of PIA: under a limited real-data budget, simulation priors provide the structured initialization needed to make few-shot adaptation feasible.

## VII. DISCUSSION AND CONCLUSION

Simulation priors provide a practical offline initialization for model-based RL under strict target-interaction budgets. In ACGym, PIA-AC transfers to held-out track geometry, while target-geometry ODE trajectories further accelerate adaptation under high-fidelity dynamics and grip mismatch. ODE-only pretraining is useful, but the strongest results arise when low-fidelity coverage complements high-fidelity source data. On hardware, the same simulation-pretrained worldmodel pipeline supports progressive online adaptation on a full-size autonomous race car. This full-scale demonstration establishes that the approach can operate under real-time and physical-safety constraints. The central result is not only final lap performance, but that imperfect simulation priors enable faster and safer adaptation as limited target data are collected.

## VIII. LIMITATIONS

PIA assumes that a model or simulator exists, and that synthetic and target data can be represented with a shared observation/action interface, making it more challenging to use for methods using complex sensor modes (image/LiDAR) as inputs, as this must be produced by the offline simulation environment. Our current experiments focus on settings where dynamics mismatch is the main challenge, not settings with strong exogenous stochasticity, such as opponents on track. The real-car study is limited to deployments on a single physical track and surface. The observed failure modes are: early real-car deployments can fail when the real-data anchor is too small, and safety-envelope violations remain possible during adaptation.

![](images/56479ce2f3b6d599e615eefc62bf5d192cdc282843ae8e59716b6f5b36ab7dc7.jpg)  
(a) s1

![](images/2f5daa052ef352d98bf63f8e4661711f7651aab3faeb87bbadca9aa3a3dc6ad2.jpg)  
(b) s2

![](images/7213fef9814f2b2e4fe38bb5e7bc14a3bf4689267dad629416accf229dc208d9.jpg)  
(c) s3

![](images/b0cb481518aaa9f17ff04847dbda543d9a2c4ea19843f209bf01536ec148e016.jpg)  
(d) s4  
Fig. 6: Vehicle XY traces. All episode trajectories for deployment sessions s1–s4.  
APPENDIX

TABLE IV: Synthetic-only pretraining. Pretraining on synthetic rollouts alone enables competitive adaptation. Adding a small amount of AC data improves over ODE-only, while the best lap time and safety are achieved by LOW ODE, where a small amount of ODE data complements a larger AC buffer.

<table><tr><td>Model</td><td>ODE (M)</td><td>AC (M)</td><td>Best lap (s)</td><td># Ep. to 1st success</td><td># Failures</td><td>Miles per Failure</td></tr><tr><td>SCRATCH</td><td>0.0</td><td>0.0</td><td>Fail</td><td>一</td><td>50</td><td>0.03</td></tr><tr><td>ODE ONLY</td><td>39.0</td><td>0.0</td><td>115.90</td><td>36</td><td>46</td><td>3.98</td></tr><tr><td>AC ONLY</td><td>0.0</td><td>36.0</td><td>114.16</td><td>8</td><td>18</td><td>34.62</td></tr><tr><td>LOW AC</td><td>39.0</td><td>1.0</td><td>112.90</td><td>6</td><td>19</td><td>33.46</td></tr><tr><td>LOW ODE</td><td>3.0</td><td>36.0</td><td>112.17</td><td>5</td><td>11</td><td>68.80</td></tr></table>

TABLE V: Ablations. ODE data (partitioning Geometry and Dynamics), sub-task ID, and uncertainty penalization.
<table><tr><td>Experiment</td><td>Best Lap (s)</td><td># Failures</td><td>Miles per Incident</td></tr><tr><td>No ODE</td><td>114.16</td><td>18</td><td>34.62</td></tr><tr><td>ODE G only</td><td>113.39</td><td>14</td><td>49.72</td></tr><tr><td>ODE D only</td><td>114.43</td><td>27</td><td>17.46</td></tr><tr><td>ODE G+D</td><td>112.17</td><td>11</td><td>68.80</td></tr><tr><td>W/o context</td><td>112.98</td><td>29</td><td>19.12</td></tr><tr><td>W/ context</td><td>112.17</td><td>11</td><td>68.80</td></tr><tr><td>W/o uncertainty in (3)</td><td>114.54</td><td>20</td><td>8.32</td></tr><tr><td>W/ uncertainty in (3)</td><td>113.16</td><td>9</td><td>37.62</td></tr></table>

TABLE VI: Domain-randomized ODE priors. All methods are fine-tuned and evaluated on the Silverstone track.
<table><tr><td>Offline data</td><td>Best lap (s) ↓</td><td>Miles per Failure ↑</td></tr><tr><td>ODEDR,SIL + ACBRN</td><td>113.37</td><td>13.31</td></tr><tr><td>ODESIL + ACBRN</td><td>113.37</td><td>11.26</td></tr><tr><td>ACBRN</td><td>115.46</td><td>5.20</td></tr><tr><td>ODEDR,SIL</td><td>115.54</td><td>3.43</td></tr><tr><td>ODEDR,BRN</td><td>117.01</td><td>2.62</td></tr></table>

1) Synthetic-only pretraining: We evaluate whether synthetic ODE rollouts can support adaptation when AC data is limited. This experiment asks whether policies pretrained exclusively on synthetic rollouts can achieve competitive adaptation, and how ODE and AC data trade off under different pretraining mixtures. We compare SCRATCH with no pretraining, ODE ONLY pretrained solely on ODE rollouts, AC ONLY pretrained solely on simulator rollouts, LOW AC with 39M ODE steps and 1M AC steps, and LOW ODE with 3M ODE steps and 36M AC steps. Table IV shows that synthetic-only pretraining is viable when high-fidelity data is unavailable, but low-fidelity data is most effective as complementary coverage rather than as a full replacement.

## 2) Ablations:

a) ODE Signal ablations: We seek to determine whether ODE pretraining helps via geometry (spatial layout: track borders, curvature look-ahead), dynamics (proprioceptive signals), or both. We compare pretraining settings to isolate their contributions. We pretrain with: No-ODE (PIA-MBRL AC), ODE (G+D) (PIA-ODE), ODE G-only (mask dynamics on ODE samples), ODE D-only (mask geometry on ODE sam ples) and fine-tune on SIL. Geometry-only reduces failures relative to No-ODE, confirming that spatial priors improve exploration safety, but has a slower lap time. Dynamics-only underperforms on both metrics, while G+D is best on both.

b) Planning Uncertainty Penalty: We ablate the planner regularization mechanism described in Section IV. Removing it leads to unstable adaptation and more incidents. Regularization stabilizes early adaptation.

c) Context Ablation: We ablate conditioning on the context c. Removing the context leads to worse generalization and noisier adaptation, confirming that explicit context helps disambiguate heterogeneous datasets.

d) Domain-Randomized ODE Prior: We additionally test whether domain randomization can replace or complement heterogeneous simulation priors. We randomize tire grip, vehicle mass, engine torque, and yaw inertia in the ODE generator, then fine-tune and evaluate on SIL. We observe that Domain randomization alone does not improve ODEonly pretraining. When combined with the same AC data, however, it preserves the best lap time and improves miles per failure. These results indicate that domain randomization is complementary to, rather than a replacement for, combining heterogeneous priors with target-domain adaptation.

## REFERENCES

[1] C. Berner, G. Brockman, B. Chan, V. Cheung, P. Debiak, C. Dennison, D. Farhi, Q. Fischer, S. Hashme, C. Hesse et al., “Dota 2 with large scale deep reinforcement learning,” arXiv preprint arXiv:1912.06680, 2019.

[2] J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart, D. Hassabis, T. Graepel et al., “Mastering atari, go, chess and shogi by planning with a learned model,” Nature, 2020.

[3] P. R. Wurman, S. Barrett, K. Kawamoto, and et al., “Outracing champion gran turismo drivers with deep reinforcement learning,” Nature, 2022.

[4] T. Lew, M. Greiff, F. Djeumou, M. Suminaka, M. Thompson, and J. Subosits, “Risk-averse model predictive control for racing in adverse conditions,” arXiv preprint arXiv:2410.17183, 2024.

[5] A. Kumar, A. Zhou, G. Tucker, and S. Levine, “Conservative Q-learning for offline reinforcement learning,” arXiv preprint arXiv:2006.04779, 2020.

[6] I. Kostrikov, A. Nair, and S. Levine, “Offline reinforcement learning with implicit Q-learning,” preprint arXiv:2110.06169, 2021.

[7] A. Nair, A. Gupta, M. Dalal, and S. Levine, “Awac: Accelerating online reinforcement learning with offline datasets,” arXiv preprint arXiv:2006.09359, 2021.

[8] Y. Feng, N. Hansen, Z. Xiong, C. Rajagopalan, and X. Wang, “Finetuning offline world models in the real world,” in Conf. on Robot Learning, 2023.

[9] P. J. Ball, L. Smith, I. Kostrikov, and S. Levine, “Efficient online reinforcement learning with offline data,” in International Conference on Machine Learning, 2023.

[10] S. Lee, Y. Seo, K. Lee, P. Abbeel, and J. Shin, “Offline-to-online reinforcement learning via balanced replay and pessimistic Q-ensemble,” arXiv preprint arXiv:2107.00591, 2021.

[11] R. S. Sutton, “Dyna, an integrated architecture for learning, planning, and reacting,” SIGART Bull., 1990.

[12] M. Janner, J. Fu, M. Zhang, and S. Levine, “When to trust your model: Model-based policy optimization,” arXiv preprint arXiv:1906.08253, 2019.

[13] T. Yu, G. Thomas, L. Yu, S. Ermon, J. Y. Zou, S. Levine, C. Finn, and T. Ma, “Mopo: Model-based offline policy optimization,” arXiv preprint arXiv:2005.13239, 2020.

[14] T. Haarnoja, A. Zhou, K. Hartikainen, G. Tucker, S. Ha, J. Tan, V. Kumar, H. Zhu, A. Gupta, P. Abbeel, and S. Levine, “Soft actorcritic algorithms and applications,” arXiv preprint arXiv:1812.05905, 2018.

[15] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012.

[16] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State, “Isaac gym: High performance gpu-based physics simulation for robot learning,” arXiv preprint arXiv:2108.10470, 2021.

[17] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, “Sim-toreal transfer of robotic control with dynamics randomization,” in 2018 IEEE International Conference on Robotics and Automation (ICRA), 2018.

[18] A. Remonda, N. Hansen, A. Raji, N. Musiu, M. Bertogna, E. E. Veas, and X. Wang, “A simulation benchmark for autonomous racing with large-scale human data,” arXiv preprint arXiv:2407.16680, 2024.

[19] D. Hafner, T. P. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” arXiv preprint arXiv:1912.01603, 2019.

[20] N. Hansen, H. Su, and X. Wang, “TD-MPC2: Scalable, robust world models for continuous control,” in Int. Conference on Learning Representations, 2024.

[21] Z. E. Asri, O. Sigaud, and N. Thome, “Physics-informed model and hybrid planning for efficient dyna-style reinforcement learning,” RLJ, 2024.

[22] W. Xiao, H. Xue, T. Tao, D. Kalaria, J. M. Dolan, and G. Shi, “Anycar to anywhere: Learning universal dynamics model for agile and adaptive mobility,” International Conference on Robotics and Automation (ICRA), 2025.

[23] S. Levine, A. Kumar, G. Tucker, and J. Fu, “Offline reinforcement learning: Tutorial, review, and perspectives on open problems,” preprint arXiv:2005.01643, 2020.

[24] H. Niu, S. Sharma, Y. Qiu, M. Li, G. Zhou, J. Hu, and X. Zhan, “When to trust your simulator: Dynamics-aware hybrid offline-and-online reinforcement learning,” arXiv preprint arXiv:2206.13464, 2022.

[25] A. Kumar, Z. Fu, D. Pathak, and J. Malik, “Rma: Rapid motor adaptation for legged robots,” arXiv preprint arXiv:2107.04034, 2021.

[26] E. Kaufmann, L. Bauersfeld, A. Loquercio, M. Muller, V. Koltun, and¨ D. Scaramuzza, “Champion-level drone racing using deep reinforcement learning,” Nature, 2023.

[27] J. Betz, T. Betz, F. Fent, M. Geisslinger, A. Heilmeier, L. Hermansdorfer, T. Herrmann, S. Huch, P. Karle, M. Lienkamp, B. Lohmann, F. Nobis, L. Ogretmen, M. Rowold, F. Sauerbeck, T. Stahl, R. Trauth,<sup>¨</sup> F. Werner, and A. Wischnewski, “TUM autonomous motorsport: An autonomous racing software for the Indy autonomous challenge,” Journal of Field Robotics, 2023.

[28] A. Raji, A. Liniger, A. Giove, A. Toschi, N. Musiu, D. Morra, M. Verucchi, D. Caporale, and M. Bertogna, “Motion planning and control for multi vehicle autonomous racing at high speeds,” in Int. Conference on Intelligent Transportation Systems (ITSC), 2022.

[29] A. Raji, D. Caporale, F. Gatti, A. Giove, M. Verucchi, D. Malatesta,

N. Musiu, A. Toschi, S. R. Popitanu, F. Bagni, M. Bosi, A. Liniger, M. Bertogna, D. Morra, F. Amerotti, L. Bartoli, F. Martello, and R. Porta, “er.autopilot 1.0: The full autonomous stack for oval racing at high speeds,” Field Robotics, 2024.

[30] A. Wischnewski, T. Herrmann, F. Werner, and B. Lohmann, “A tube-MPC approach to autonomous multi-vehicle racing on high-speed ovals,” IEEE Transactions on Intelligent Vehicles, 2023.

[31] A. Davydov, F. Djeumou, M. Greiff, M. Suminaka, M. Thompson, J. Subosits, and T. Lew, “First, learn what you don’t know: Active information gathering for driving at the limits of handling,” Robotics and Automation Letters, 2025.

[32] F. Fuchs, Y. Song, E. Kaufmann, D. Scaramuzza, and P. Durr, “Superhuman performance in gran turismo sport using deep reinforcement learning,” IEEE Robotics and Automation Letters, 2021.

[33] A. Remonda, S. Krebs, E. Veas, G. Luzhnica, and R. Kern, “Formula rl: Deep reinforcement learning for autonomous racing using telemetry data,” arXiv preprint arXiv:2104.11106, 2022.

[34] F. Djeumou, M. Thompson, M. Suminaka, and J. Subosits, “Referencefree formula drift with reinforcement learning: From driving data to tire energy-inspired, real-world policies,” 2025 IEEE International Conference on Robotics and Automation (ICRA), 2024.

[35] R. Bellman, “A Markovian decision process,” Indiana Univ. Math. J., 1957.

[36] L. P. Kaelbling, M. L. Littman, and A. R. Cassandra, “Planning and acting in partially observable stochastic domains,” Artificial Intelligence, 1998.

[37] A. Raji, N. Musiu, A. Toschi, F. Prignoli, E. Mascaro, P. Musso, F. Amerotti, A. Liniger, S. Sorrentino, and M. Bertogna, “A tricycle model to accurately control an autonomous racecar with locked differential,” in IEEE International Conference on Systems and Control, 2023.