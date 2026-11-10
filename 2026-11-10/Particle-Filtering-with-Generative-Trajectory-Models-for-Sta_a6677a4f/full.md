# Particle Filtering with Generative Trajectory Models for State Estimation and Control

Lukas C. Lao Beyer<sup>1</sup>, Samvit Das<sup>1</sup> and Sertac Karaman<sup>1</sup>

Abstract— Generative models can support both state estimation and control in robotics. For control, a common approach is to use the model as a policy that directly samples actions conditioned on observation histories. However, models that jointly represent states, actions, and observations can also predict how the robot state and its environment evolve. We investigate whether these predictions can improve state estimation and control through a change in inference procedure, without modifying the pretrained model. To this end, we use particle filtering to maintain competing hypotheses about state histories. A single pretrained diffusion model generates these hypotheses along with their predicted observations and associated action proposals. As new measurements arrive, particles are reweighted according to how well their predicted observations match the measurements. While particle filtering conventionally uses these weights only for state estimation, we also use them to guide action selection, favoring proposals associated with hypotheses that better explain the observations. Simulation experiments in partially observed navigation and vision-based manipulation demonstrate improvements in both state estimation and task performance compared to action generation without filtering, using the same pretrained model and no additional test-time information.

Index Terms— Diffusion Models, Particle Filtering, Partial Observability, Inference-Time Computation

## I. INTRODUCTION

Diffusion- and flow-based generative models have become powerful tools for robot learning and control. They have been used as policies that directly generate actions conditioned on observations, as world models that predict the evolution of the environment, and, more recently, as unified world-action models that jointly model states, observations, and actions over time. Such joint trajectory models can support a variety of conditional generation tasks, depending on which variables are provided as context and which are generated. Their usefulness therefore depends not only on the distributions they learn, but also on how those distributions are used at inference time.

For control, a natural approach is to directly generate actions conditioned on the available observation-action history. In principle, if this procedure produces an optimal policy, no alternative inference strategy with access to the same information can achieve better expected task performance. In practice, however, learned policies are generally imperfect due to limited training data, finite model capacity, and distribution shifts between training and deployment. Moreover, a joint generative model may capture useful relationships between states, observations, and actions that are not fully exploited by direct action generation. This creates an opportunity to improve control through additional inferencetime computation, without modifying the learned model. This question is particularly relevant under partial observability, where successful control depends on inferring the underlying state from observations accumulated over time.

![](images/4b1b2f5c4259b061aec4c4e9104db659312ee76b01ce3a212ee615320b470144.jpg)  
Fig. 1. Two inference procedures using the same pretrained model p . (a) Direct action generation. (b) Particle filtering maintains weighted state histories and generates joint state-action-observation proposals. Weights guide action selection. Agreement between predicted and measured observations updates the belief for the next step. Stacked blocks denote per-particle computations.

In this work, we explore particle filtering as an alternative inference-time approach to state estimation and control with pretrained world-action models (Fig. 1). Under partial observability, several different state histories may be consistent with the observations available so far. Maintaining a diverse set of plausible histories allows later observations to help resolve this ambiguity. While traditional particle filters maintain a distribution over possible states of the environment, our approach also associates each state hypothesis with candidate future actions, generated jointly by the same model. As a result, evidence about the underlying state can directly influence action selection, allowing particle filtering to serve both state estimation and control within a unified framework. Importantly, this approach requires no additional filteringspecific training or separately learned dynamics, observation, or policy models. We compare particle filtering against direct action generation using the same pretrained model.

We investigate the benefits and limitations of this approach through experiments in partially observed navigation and visuomotor robotic manipulation. Navigation provides a controlled setting in which we systematically study the effects of training data, distribution shift, and different particle filtering configurations. We further evaluate the approach on manip ulation tasks with high-dimensional RGB observations and substantially different action spaces. Our results demonstrate that inference-time particle filtering can improve state estimation and control without additional model training, while highlighting the conditions under which these improvements can be expected.

## II. RELATED WORK

a) Generative policies, trajectory models, and worldaction models: Diffusion models have emerged as expressive models for both robot policies and trajectory-level decision making. Diffusion Policy represents multimodal visuomotor action distributions through conditional denoising [1], while Diffuser and Decision Diffuser formulate planning and control as conditional trajectory generation [2, 3]. More recently, world-action models jointly represent world evolution and executable robot actions within a unified generative model. Unified World Models, for example, couples video and action diffusion such that a single model can serve as a policy, forward or inverse dynamics model, or observation generator [4]. Diffusion Forcing provides a complementary mechanism for flexible conditioning over arbitrary subsets of sequential variables [5]. We build on these flexible conditioning capabilities by using a pretrained joint state-actionobservation model as the primitive in an inference-time particle filtering procedure.

b) Particle filtering with learned models: Prior learned particle-filter approaches replace hand-specified filtering models with learnable dynamics and observation components embedded in a differentiable particle-filter structure, allowing the resulting estimator to be optimized end-to-end [6, 7]. More recently, DiffPF uses a conditional diffusion model to generate filtering-posterior particles [8]. These methods use learned models to improve estimation, but their particles are used to represent state hypotheses rather than candidate decisions. In our setting, a pretrained generative model jointly produces states, observations, and actions, so each particle simultaneously represents a state hypothesis and an associated control rollout. Moreover, our generative model is trained independently of the filtering procedure and without filtering-specific training or learned components.

c) Particle beliefs for control: Particle beliefs have also been used for decision making under partial observability. Classical online POMDP methods represent uncertainty with particles and couple this belief representation to online planning, typically using repeated forward simulation and tree search to evaluate candidate actions [9, 10]. These approaches demonstrate the value of particle-based beliefs for control, but generally require a separate test-time planning procedure on top of the filter. In our setting, each particle instead carries an action rollout generated jointly with its state and observation hypotheses. Observation-based particle weights can therefore be used directly for action selection, without a separate belief-space planning or search procedure.

d) Particle methods for diffusion inference: Particlebased methods have also been used within diffusion inference to improve sampling or steer generation toward task-specific objectives. Such approaches maintain and resample candidate diffusion samples according to guidance or reward potentials [11, 12]. This use of particles is orthogonal to ours: their particles exist within a single generative inference process, whereas our particles persist across environment timesteps as hypotheses over latent state and associated future trajectories, and are reweighted using newly observed sensor data.

## III. PRELIMINARIES

We consider filtering and control under partial observability, with state $\mathbf { x } _ { t } .$ , action $\mathbf { a } _ { t } .$ , observation $\mathbf { o } _ { t }$ , and available history $\mathcal { H } _ { t } = ( \mathbf { o } _ { 0 : t } , \mathbf { a } _ { 0 : t - 1 } )$ . Filtering estimates $p ( \mathbf { x } _ { 0 : t } \mid \mathcal { H } _ { t } )$

a) Particle filtering: A particle filter represents this posterior by weighted state trajectories,

$$
p ( \mathbf { x } _ { 0 : t } \mid \mathcal { H } _ { t } ) \approx \sum _ { i = 1 } ^ { N } w _ { t } ^ { ( i ) } \delta _ { \mathbf { x } _ { 0 : t } ^ { ( i ) } } ( \mathbf { x } _ { 0 : t } ) ,\tag{1}
$$

with $\begin{array} { r } { \sum _ { i } w _ { t } ^ { ( i ) } = 1 } \end{array}$ . In a bootstrap filter [13], each trajectory is extended using the dynamics model,

$$
\mathbf { x } _ { t } ^ { ( i ) } \sim p _ { \mathrm { d y n } } ( \mathbf { x } _ { t } \mid \mathbf { x } _ { t - 1 } ^ { ( i ) } , \mathbf { a } _ { t - 1 } ) .\tag{2}
$$

The arriving observation reweights particles by their likelihood $p _ { \mathrm { o b s } } ( \mathbf { \bar { o } } _ { t } \mid \mathbf { x } _ { t } ^ { ( i ) } )$ , favoring hypotheses that better explain the measurement.

b) Diffusion forcing: Diffusion models generate samples by starting from noise and iteratively denoising toward structured outputs. Diffusion forcing [5] extends this procedure to sequences by sampling a separate noise level for each token during training. At inference, known tokens can be held at zero or low noise levels, while queried tokens are denoised fully starting from pure noise. This enables the same model to condition on different subsets of sequential variables and modalities [14].

## IV. METHOD

We combine particle filtering with a learned trajectory diffusion model. The central difference from a bootstrap particle filter is the proposal distribution: instead of advancing each particle with a one-step dynamics kernel, we use a learned generative model to propose future states, observations, and actions jointly. The resulting particles are therefore both state-estimation hypotheses and candidate control rollouts.

## A. Flexible Conditional Trajectory Sampler

Our method exposes the diffusion model through a flexible conditioning interface over state, action, and observation tokens. Each model query fixes a subset of trajectory variables and samples the remaining variables jointly. Let $q _ { x } , q _ { a } , q _ { o }$ denote query index sets to be generated, and let $h _ { x } , h _ { a } , h _ { o }$ denote conditioning index sets supplied to the model. We write the resulting conditional sampler as

$$
\begin{array} { r } { \big ( \bar { \bf x } _ { q _ { x } } , \bar { \bf a } _ { q _ { a } } , \bar { \bf o } _ { q _ { o } } \big ) \sim p _ { \theta } \big ( { \bf x } _ { q _ { x } } , { \bf a } _ { q _ { a } } , { \bf o } _ { q _ { o } } \big \vert \ { \bf x } _ { h _ { x } } , { \bf a } _ { h _ { a } } , { \bf o } _ { h _ { o } } \big ) . } \end{array}\tag{3}
$$

This construction uses the diffusion model as a flexible proposal distribution $p _ { \theta }$ . By changing which variables are conditioned on and which are queried, the same model can act as a state-transition proposal, an observation predictor, a future-action generator, or a joint trajectory model.

## B. Particle Filter

We now describe how this flexible sampling interface enables particle filtering for both state estimation and decision making at inference time. Each iteration uses the new observation to reweight competing state-history hypotheses, then generates an action proposal conditioned on each history. The updated weights determine how these proposals influence the executed action.

1) Particle State: At the current timestep t, each particle carries a hypothesized state history $\bar { \mathbf { x } } _ { 0 : t - 1 } ^ { ( i ) }$ and a corresponding weight $w _ { t - 1 } ^ { ( i ) }$ . In the first iteration, particle weights are initialized uniformly and the algorithm proceeds directly to action proposal.

2) State and Observation Prediction: Before incorporating the observation at the current time t, we predict the current state and observation for each particle, conditioned only on its state history hypothesis, past observations and actions $\mathscr { H } _ { t - 1 } = \left( \mathbf { o } _ { 0 : t - 1 } , \mathbf { a } _ { 0 : t - 2 } \right)$ , and the action commanded at the preceding step $\mathbf { a } _ { t - 1 } .$

$$
( \bar { \mathbf { x } } _ { t } ^ { ( i ) } , \mathbf { , } \bar { \mathbf { o } } _ { t } ^ { ( i ) } ) \sim p _ { \theta } ( \cdot \mid \bar { \mathbf { x } } _ { 0 : t - 1 } ^ { ( i ) } , \mathbf { a } _ { t - 1 } , \mathcal { H } _ { t - 1 } ) .\tag{4}
$$

3) Observation-Consistent Weight Updates: We update each particle’s weight according to how well its prediction agrees with the new observation $\mathbf { o } _ { t } .$ , favoring histories that better explain the observed sensor stream. Let $\overline { { \ell } } ( \mathbf { o } _ { t } ; \bar { \mathbf { o } } _ { t } ^ { ( i ) } , \bar { \mathbf { x } } _ { t } ^ { ( i ) } )$ denote this nonnegative observation score, which need not be normalized. It may be an explicit sensor likelihood $p _ { \mathrm { o b s } } ( \mathbf { o } _ { t } \mid$ $\bar { \mathbf { x } } _ { t } ^ { ( i ) } )$ or another model-based compatibility score. We may also avoid sensor-specific design by using an observationconsistency score directly comparing the realized observation $\mathbf { o } _ { t }$ to the previously predicted observation $\bar { \mathbf { o } } _ { t } ^ { ( i ) }$ . Our implementation compares predicted and measured observations using cosine similarity in a learned latent observation space, as described in Section VI. The particle weights are then updated as

$$
\bar { w } _ { t } ^ { ( i ) } = w _ { t - 1 } ^ { ( i ) } \ell \big ( \mathbf { o } _ { t } ; \bar { \mathbf { o } } _ { t } ^ { ( i ) } , \bar { \mathbf { x } } _ { t } ^ { ( i ) } \big ) ,\tag{5}
$$

$$
w _ { t } ^ { ( i ) } = \frac { \bar { w } _ { t } ^ { ( i ) } } { \sum _ { j = 1 } ^ { N } \bar { w } _ { t } ^ { ( j ) } } .\tag{6}
$$

Resampling is applied using the updated weights if the effective sample size becomes small.

4) Action Proposal: We sample current and future states and future actions for each particle, sharing the commanded action and observation history $\mathcal { H } _ { t } ~ = ~ \left( \mathbf { o } _ { 0 : t } , \mathbf { a } _ { 0 : t - 1 } \right)$ across particles:

$$
( \bar { \mathbf { x } } _ { t : T } ^ { ( i ) } , \bar { \mathbf { a } } _ { t : T - 1 } ^ { ( i ) } , . ) \sim p _ { \theta } ( \cdot \mid \bar { \mathbf { x } } _ { 0 : t - 1 } ^ { ( i ) } , \mathcal { H } _ { t } ) ,\tag{7}
$$

where $T - t$ is the planning horizon. The newly sampled current state incorporates the current measurement $\mathbf { o } _ { t }$ . It replaces the prediction used for scoring and is appended to the particle’s state history.

5) Action Aggregation: We use the particle weights to choose the next action from the proposals, evaluating two action-selection rules. Best-particle action selection executes the proposal associated with the highest-weight particle.

Weighted action averaging instead combines the action proposals using the weighted average

$$
\mathbf { a } _ { t } = \sum _ { i = 1 } ^ { N } w _ { t } ^ { ( i ) } \bar { \mathbf { a } } _ { t } ^ { ( i ) } .\tag{8}
$$

The selected action is then executed and the algorithm proceeds to the next iteration.

## V. WHEN CAN PARTICLE FILTERING HELP?

Particle filtering and direct action generation both operate using the same observation-action history. If direct action generation already realizes an optimal policy with exact sampling, filtering cannot improve expected task performance. Its potential benefit lies in exploiting learned relationships among states, observations, and actions that an imperfect direct policy may underuse. Maintaining and reweighting explicit state hypotheses could expose this structure to action selection. We hypothesize that this improves control when three conditions hold:

1) State-action alignment. The generative model must capture meaningful relationships between states and actions, such that more accurate or task-relevant state hypotheses tend to produce higher-quality action proposals.

2) Observation-state alignment. The generative model must capture meaningful relationships between states and observations, such that compatibility between predicted and realized observations provides useful evidence about the quality of the corresponding state hypotheses.

3) Sufficient particle coverage and diversity. The generative model must assign sufficient probability to high-quality state-action hypotheses, and the particle population must retain enough diversity to represent these hypotheses until observations can distinguish between them.

These conditions allow observation-based particle weighting to favor state hypotheses associated with better control proposals. In Section VII, we test whether each of the conditions holds in practice through a series of experiments.

## VI. IMPLEMENTATION

## A. Latent Observation World-Action Model

Raw sensor observations, such as RGB images, are highdimensional and may contain substantial task-irrelevant variation. This can make defining an observation-consistency metric ℓ directly in the raw observation space challenging. We therefore jointly learn an observation encoder $\mathcal { E } _ { \mathrm { o b s } }$ and a diffusion model over states, actions, and latent observations,

$$
{ \mathbf { o } } _ { t } = \mathcal { E } _ { \mathrm { o b s } } \left( { \mathbf { o } } _ { t } ^ { \mathrm { r a w } } \right) .\tag{9}
$$

The resulting latent observations provide a compact representation for both trajectory modeling and the observationconsistency score ℓ.

Directly backpropagating the latent-observation prediction loss into the parameters of $\mathcal { E } _ { \mathrm { o b s } }$ may encourage a degenerate

representation that is easy to predict but uninformative. We therefore train the encoder only through the prediction of task variables. Let

$$
\widetilde { \mathbf { x } } \sim q _ { \tau _ { x } } ( \cdot \mid \mathbf { x } ) , \quad \widetilde { \mathbf { a } } \sim q _ { \tau _ { a } } ( \cdot \mid \mathbf { a } ) , \quad \mathrm { a n d } \quad \widetilde { \mathbf { o } } \sim q _ { \tau _ { o } } ( \cdot \mid \mathbf { o } )\tag{10}
$$

denote states, actions and latent observations $\mathbf { o } = \mathcal { E } _ { \mathrm { o b s } } \left( \mathbf { o } ^ { \mathrm { r a w } } \right)$ corrupted by the diffusion forward process $q _ { \tau }$ using diffusion noise levels τ which are sampled independently for each timestep and each input modality. Let $f _ { \theta }$ denote the denoising network, conditioned on the diffusion noise level of each token (we suppress this conditioning below for readability) and compute

$$
\widehat { { \bf x } } , \widehat { { \bf a } } , _ { - } = f ( \widetilde { { \bf x } } , \widetilde { { \bf a } } , \widetilde { { \bf o } } ) ,\tag{11}
$$

where gradients are allowed to flow through oe to $\mathcal { E } _ { \mathrm { o b s } }$ We then compute the observation prediction using detached latent observations,

$$
\mathbf { \Sigma } _ { - , - , \widehat { \mathbf { o } } } = f ( \widetilde { \mathbf { x } } , \widetilde { \mathbf { a } } , \mathrm { s g } ( \widetilde { \mathbf { o } } ) ) ,\tag{12}
$$

where sg denotes the stop-gradient operator. Both passes share model parameters and use the same corrupted training example, differing only in the gradient flow to the observation encoder.

Using the clean-data-prediction parameterization for notational simplicity, the training objective is

$$
\begin{array} { r } { \mathcal { L } = \left\| \widehat { \mathbf { x } } - \mathbf { x } \right\| _ { 2 } ^ { 2 } + \left\| \widehat { \mathbf { a } } - \mathbf { a } \right\| _ { 2 } ^ { 2 } + \lambda _ { o } \left\| \widehat { \mathbf { o } } - \mathrm { s g } ( \mathbf { o } ) \right\| _ { 2 } ^ { 2 } . } \end{array}\tag{13}
$$

Thus, the observation encoder is optimized to retain information useful for predicting task variables, while the diffusion model is simultaneously trained to model the resulting latent observation sequence.

## B. Latent Cosine Similarity Observation-Consistency Metric

Our choice of observation-consistency metric used to compare predicted and realized observations in the particle filter uses cosine similarity of latent observations:

$$
\ell _ { \mathrm { c o s } } \big ( { \bf o } _ { t } ; \bar { \mathbf { o } } _ { t } ^ { ( i ) } \big ) = \exp \left[ \frac { { \bf o } _ { t } ^ { \top } \bar { \mathbf { o } } _ { t } ^ { ( i ) } } { \gamma \left\| { \bf o } _ { t } \right\| \left\| \bar { \mathbf { o } } _ { t } ^ { ( i ) } \right\| } \right] ,\tag{14}
$$

where $\gamma$ is a temperature scaling hyperparameter.

## VII. EXPERIMENTS

We evaluate our approach in two domains: goal-directed navigation in partially observed 2D environments and visuomotor robotic manipulation. We primarily use navigation as a controlled testbed, where synthetic data generation and access to ground-truth state enable systematic studies of model quality, observation consistency, and particle filtering behavior. We then evaluate our method on robotic manipulation tasks with high-dimensional RGB observations and substantially different action spaces, demonstrating its applicability beyond the navigation setting.

For each task and training-dataset size, all inference variants use the same trained trajectory model and observation encoder, with identical parameters. The baseline denoted Diffusion Policy performs direct action generation using this model without particle filtering. At inference, conditioning histories are restricted to a rolling window that fits the model’s context length. Navigation executes one action per update, while manipulation executes eight actions openloop. For chunked execution, state and observation prediction conditions on the full previously commanded action chunk, and the observation scores over that chunk are multiplied in a single weight update.

## A. Experimental Setup (Navigation)

Our navigation task requires collision-free traversal of an unknown 2D environment towards a goal specified in the frame of the robot’s initial pose. At each step, the robot receives a 2D LiDAR observation consisting of $N _ { \mathrm { r a y s } }$ distance measurements and executes a $[ \Delta x , \Delta y ] \in \mathbb { R } ^ { 2 }$ control action, specified in the robot’s body frame. To ensure that LiDAR-based localization becomes essential, we add zeromean Gaussian noise to executed actions with a standard deviation of 40% of the commanded step length. Heading θ is not controlled by the policy, instead evolving according to an exogenous random walk process. At each timestep, the state is the 4D vector $[ g _ { x } ^ { \mathrm { b o d y } } , g _ { y } ^ { \mathrm { b o d y } } , \sin ( \Delta \theta ) , \cos ( \Delta \theta ) ]$ combining the goal’s position in the agent’s current body frame with its heading relative to the trajectory’s initial pose. This representation is invariant to global translation and rotation, while the sine-cosine encoding avoids angular discontinuities.

1) Dataset: We generate a synthetic dataset containing 1000 examples with randomly placed circular and rectangular obstacles and randomly selected start and goal locations. Each demonstration trajectory is created by first planning the shortest collision-free path between the start and goal. Then, the pure-pursuit controller [15] is used to simulate tracking of the planned reference under noisy action execution, and heading is assigned according to a correlated random walk.

2) Model & Training: We implement the observation encoder $\mathcal { E } _ { \mathrm { o b s } }$ using circular convolutions, producing one or more latent observation tokens per timestep. The denoising network f is implemented as a diffusion transformer [16] which receives noised latent observation, state, and action tokens. A block-triangular attention mask is used to implement temporal causal masking. Goal conditioning is achieved through in-context conditioning in which the first timestep’s state is set to the relative goal position. Since the goal is provided only through the initial state, the navigation Diffusion Policy baseline retains its sampled state history to carry goal information forward as the context window advances.

3) Evaluation Metrics and Protocol: We measure taskrelevant performance using the deviation between the goal position and the position at termination. State-estimation metrics use weighted mean particle positions, independently of action selection. We report ADE (mean absolute position error) and translational RPE (mean relative-translation error over consecutive intervals of distance 0.1). Unless otherwise stated, navigation experiments use 200 held-out evaluation scenarios. Each experiment is repeated eight times using different random seeds. Tables report mean metrics and paired 95% bootstrap confidence intervals for percentage changes in mean performance relative to Diffusion Policy. We average repeated runs within each scenario and resample scenarios, using the same sampled scenarios for each method and its baseline. When applicable, 32 candidates or particles are used, unless noted otherwise. Unless otherwise stated, navigation particle-filter experiments use best-particle action selection.

TABLE I  
FULL PARTICLE FILTERING OUTPERFORMS NA¨IVE ACTION SELECTION UNDER THE POSE-ORACLE OBSERVATION MODEL.
<table><tr><td>Method</td><td>Retained histories</td><td>Accum. weights</td><td>Final dist ↓</td><td>ADE↓</td><td>RPE ↓</td></tr><tr><td>Diffusion Policy</td><td>1</td><td></td><td>0.898</td><td>0.494</td><td>0.116</td></tr><tr><td>One-step selection</td><td>1</td><td>No</td><td>0.528 [−44%,-38%]</td><td>0.253 [-51%,-47%]</td><td>0.067 [-43%,-40%]</td></tr><tr><td>Multi-history sel.</td><td>32</td><td>No</td><td>0.589 [-37%,-32%]</td><td>0.282 [-46%,-40%]</td><td>0.092 [-24%,-15%]</td></tr><tr><td>Full PF</td><td>32</td><td>Yes</td><td>0.426[-56%, -49%]</td><td>0.195[-62%, -59%]</td><td>0.061[-49%, -45%]</td></tr></table>

## B. Comparison To Other Candidate-Selection Methods

We begin by investigating how particle filtering compares to other methods that use selection from among several candidates.

1) Ground-truth pose oracle likelihood: To evaluate stateaction alignment independently of observation model accuracy, we first use a privileged pose oracle score that assigns higher likelihood to particles whose predicted poses agree more closely with ground truth:

$$
\begin{array} { r } { \ell _ { \mathrm { o r a c l e } } ^ { ( i ) } ( \mathbf { x } _ { t } ; \bar { \mathbf { x } } _ { t } ^ { ( i ) } ) = \exp \left[ - \Delta \bar { \mathbf { p } } _ { t } ^ { ( i ) } / \gamma _ { p } - \Delta \bar { \theta } _ { t } ^ { ( i ) } / \gamma _ { \theta } \right] . } \end{array}\tag{15}
$$

Here, $\Delta \bar { \mathbf { p } } _ { t } ^ { ( i ) } \ = \ \| \bar { \mathbf { p } } _ { t } ^ { ( i ) } \ - \mathbf { p } _ { t } \|$ refers to the position error (positions p share the initial-pose frame), and each particle’s heading error $\Delta \bar { \theta } _ { t } ^ { ( i ) }$ is defined analogously. The scales $\gamma _ { p }$ and γ are hyperparameters.

2) Baselines: Our first baseline, one-step selection, maintains a single state trajectory history. At each step, it samples and executes an action conditioned on that history and the latest observation. It then compares N candidate extensions of the history, selecting the one that most accurately predicts the resulting ground-truth pose. The selected history is retained for the next step.

Multi-history selection instead maintains N distinct state histories. At each update, it selects the action proposal associated with the highest current pose-oracle score. Compared to full particle filtering, selection uses only the latest prediction error, without accumulating scores over time.

3) Results (Table I): We find that both baselines are able to leverage ground-truth pose-informed state selection to achieve improved task performance and state estimation accuracy compared to diffusion policy, indicating that the model has indeed learned significant state-action alignment, providing evidence for Condition 1. However, simple onestep selection that does not persist diverse histories and multi-history selection that does not accumulate weights across timesteps both perform significantly worse than full particle filtering.

TABLE II  
OBSERVATION-CONSISTENCY OBSERVATION MODEL ENABLES  
IMPROVED TASK PERFORMANCE AND ABSOLUTE STATE ESTIMATION.
<table><tr><td>Method</td><td colspan="2">Final dist ↓</td><td colspan="2">ADE↓</td><td colspan="2">RPE↓</td></tr><tr><td>Baselines:</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Diffusion Policy</td><td>0.898</td><td></td><td>0.4941</td><td></td><td>0.1155</td><td></td></tr><tr><td>Ensemble</td><td>0.940</td><td> $[ - 1 \% , + 1 0 \% ]$ </td><td>0.4516</td><td> $[ - 1 6 \% , - 2 \% ]$ </td><td></td><td>0.0658 [-44%,-40%]</td></tr><tr><td>PF w/ shuffled weights 0.874</td><td></td><td> $[ - 6 \% , + 1 \% ]$ </td><td></td><td>0.4383 [-15%,-8%]</td><td></td><td>0.0703 [-40%,-37%]</td></tr><tr><td>PF (weighted mean)</td><td>0.885</td><td> $[ - 7 \% , + 4 \% ]$ </td><td>0.4196</td><td>[-22%,-9%]</td><td></td><td>0.0653[-44%,-41%]</td></tr><tr><td>With privileged information:</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PF w/ known map</td><td></td><td>0.633 [-33%,-26%]</td><td></td><td>0.3143 [-40%,-33%]</td><td></td><td>0.0677 [-43%,-38%]</td></tr><tr><td>PF w/ pose oracle</td><td></td><td>0.426 [-56%,-49%]</td><td></td><td>0.1947 [-62%,-59%]</td><td></td><td>0.0609 [-49%,-45%]</td></tr><tr><td>Particle Filter</td><td></td><td>0.812 [−13%,-6%]</td><td></td><td>0.4087[-20%,-14%]</td><td></td><td>0.0697 [-41%,-37%]</td></tr></table>

## C. Observation-Consistency Observation Model

In the previous section, we have demonstrated that particle filtering provides a principled way of incorporating scores assigned to multiple candidates. However, in practice, access to an oracle may not be available. We therefore investigate the effectiveness of the proposed cosine similarity observationconsistency score (eq. 14) as an observation model.

To distinguish the benefits of particle filtering from those of generating and combining multiple proposals, we evaluate two additional baselines. The ensemble baseline uses the same pretrained model to generate N proposals and uniformly averages their state estimates and proposed actions, providing a comparison with similar proposal-generation cost. Each ensemble member retains its own sampled state history across planning iterations. Particle filtering with shuffled weights randomly permutes the observation-model scores among particles at each update, before applying them to the accumulated weights. This disrupts the correspondence between each particle hypothesis and its observationconsistency score.

1) Known-map observation model: To establish that cosine-similarity observation-consistency is a suitable score, we must first verify that the latent observation representation learned by the model supports meaningful comparison via cosine similarity. To this end, we consider a known-map observation model

$$
\ell _ { \mathrm { m a p } } \big ( { \mathbf { o } } _ { t } ; \overline { { \mathbf { x } } } _ { t } ^ { ( i ) } \big ) = \ell _ { \mathrm { c o s } } \big ( { \mathbf { o } } _ { t } ; \mathcal { E } _ { \mathrm { o b s } } \big ( \mathtt { r a y c a s t } \big ( \overline { { \mathbf { x } } } _ { t } ^ { ( i ) } \big ) \big ) \big ) ,\tag{16}
$$

where raycast(x) denotes the result of computing a LiDAR observation by raycasting from the pose x in the known map. As shown in Table II, this known-map observation model enables a substantial improvement across both task performance and state estimation metrics, demonstrating that the latent observation representation is indeed suitable for use with the proposed cosine-similarity-based observation model.

2) Predicted observation consistency: We further verify whether the model has learned to predict latent observations that are consistent and responsive to each particle’s state history. As shown in Table II, using the cosine observation model (eq. 14) to compare each particle’s predicted latent observation with the latent representation of the measured observation significantly outperforms the Diffusion Policy baseline, supporting Condition 2. The ensemble and shuffledweight baselines also improve state estimation, but neither yields a clear improvement in task performance. In particular, the comparison with shuffled scores indicates that observation-consistent weighting helps identify state-action hypotheses that lead to better control. In Fig. 2, we show an example of particle filtering using the predicted-observation consistency model, alongside the Diffusion Policy baseline.

![](images/2a73e7905555883a74b45ee06cbd142581234a19a09f057cea80fae66d0fe473.jpg)

![](images/3e61ee3957a012d95961b6f6ca456e174bb2255c97d2829879b4a61e47144aaa.jpg)

![](images/dac8b23100d03cce2fc80b4b3cf4786ce9498e83100899c75f4052e437bc75a0.jpg)

![](images/c9953224b0f44c35b9655ce68d84b7a57f3a5923e8aaf56569aec75e55b2d13c.jpg)  
Fig. 2. Closed-loop navigation with Diffusion Policy (top) and particle filtering (bottom). Magenta: estimated path; black: executed path; green/red crosses: goal/start. Particle brightness indicates weight. Diffusion Policy stops at t = 41, particle filter stops at t = 30.

![](images/4add2754c8f42878dd129443fb9dae79d0cce54eceb3f76000cfd3e36338e6f8.jpg)  
Fig. 3. Particle-count scaling with pose-oracle (orange), known-map (green), and predicted-observation (blue) weighting.

## D. Action Selection Rule

Table II additionally compares best-particle action selection with weighted action averaging, using the same particlefiltering procedure and observation-consistency score. Weighted action averaging achieves similar ADE and lower RPE than best-particle selection, yet only best-particle selection yields a clear improvement in final goal distance over Diffusion Policy. These results suggest that particle filtering’s estimation benefits are robust to the choice between these two rules, while using the resulting belief effectively for control requires an appropriate action-selection rule.

## E. Particle Count Scaling

Increasing the particle count tests whether the model supplies sufficient hypothesis coverage (Condition 3). Figure 3 shows improvements in task performance and state estimation under all three observation models. Predictedobservation weighting exhibits diminishing returns beyond 16 particles, while the pose oracle continues to improve, supporting both coverage and state-action alignment. Knownmap weighting achieves intermediate task performance and ADE. The remaining gap to the pose oracle may reflect both ambiguity in the observations and limitations of the learned representation and cosine similarity score.

TABLE III  
FINAL DISTANCE ACHIEVED BY NAVIGATION MODELS TRAINED ON DATASETS OF INCREASING SIZE.
<table><tr><td></td><td colspan="4">Training Dataset Size</td></tr><tr><td>Method</td><td colspan="2">1,000</td><td colspan="2">10,000</td></tr><tr><td>Diffusion Policy</td><td>0.898</td><td>0.454</td><td></td><td>0.369</td></tr><tr><td>Particle Filter</td><td>0.812 [−13%,−6%]</td><td></td><td>0.415 [−13%,-4%]</td><td>0.395 [+4%,+11%]</td></tr><tr><td>PF w/ GT oracle</td><td>0.426 [-56%,-49%]</td><td></td><td>0.237 [-52%,-44%]</td><td>0.231 [-42%,-33%]</td></tr></table>

TABLE IV

ON OOD SCENARIOS, PARTICLE FILTERING HELPS EVEN USING THE MODEL TRAINED ON 100,000 SCENARIOS.
<table><tr><td>Method</td><td colspan="2">Final dist ↓</td><td colspan="2">ADE ↓</td><td colspan="2">RPE↓</td></tr><tr><td>Diffusion Policy</td><td>0.863</td><td></td><td>0.4772</td><td></td><td>0.0515</td><td></td></tr><tr><td>Particle Filter</td><td></td><td>0.792 [-11%,-5%]</td><td></td><td>0.4501 [-8%,-4%]</td><td></td><td>0.0444[−15%,-13%]</td></tr><tr><td>PF w/ pose oracle 0.490 [−48%,-38%]</td><td></td><td></td><td></td><td>0.2750 [-46%,-39%]</td><td></td><td>0.0364 [-31%,-27%]</td></tr></table>

## F. Dataset Scaling

We view particle filtering as a way to exploit learned stateaction-observation relationships that direct action generation may not fully utilize, particularly when training data is limited. We therefore test increasing training data from 1,000 to 100,000 demonstrations, which substantially improves direct action generation (Table III). Observation-consistency filtering helps at 1,000 and 10,000 demonstrations but performs slightly worse at 100,000. Pose-oracle filtering retains a substantial advantage throughout, indicating that the model continues to produce state-action hypotheses supporting better control, and showing that particle filtering can remain an effective way of incorporating additional known information.

We additionally evaluate the 100,000-demonstration model on 100 out-of-distribution scenarios with eight random seeds (Table IV). While training scenes contain multiple randomly placed obstacles and randomly selected endpoints, OOD scenes contain a single rectangular or circular obstacle (dimensions and endpoints remain randomized). Filtering improves both task performance and state estimation under this distribution shift, demonstrating that additional inference can help even where direct action generation performs better in-distribution.

TABLE V  
PARTICLE FILTERING IMPROVES SUCCESS RATE (SR) AND STATE ESTIMATES ON CHALLENGING WRIST-CAMERA-ONLY MANIPULATION TASKS.
<table><tr><td></td><td colspan="3">Method Details</td><td colspan="3">Wrist-Camera-Only Transport</td><td colspan="3">Wrist-Camera-Only ToolHang</td></tr><tr><td>Method</td><td>Action selection</td><td>Num. particles Persistent state?</td><td></td><td>SR (%) ↑</td><td colspan="2">State RMSE↓</td><td colspan="2">SR (%) ↑</td><td>State RMSE↓</td></tr><tr><td>Diffusion Policy (DP)</td><td></td><td>1</td><td></td><td>88.4</td><td>0.282</td><td></td><td>22.8</td><td></td><td>0.042</td></tr><tr><td>DP w/ persistent state</td><td>一</td><td>1</td><td>√</td><td>85.0 [−7.2,+0.2]</td><td>0.324 [+7%,+24%]</td><td></td><td>23.4 [−4.2,+5.4]</td><td>0.050</td><td> $[ + 7 \% , + 2 8 \% ]$ </td></tr><tr><td>DP ensemble</td><td>uniform mean</td><td>32</td><td></td><td>88.6 [−5.4,+5.0]</td><td>0.216 [-29%,-19%]</td><td></td><td>5.4 [−22.0,−13.2]</td><td></td><td>0.032 [-37%,-12%]</td></tr><tr><td>DP ensemble w/ persistent state</td><td>uniform mean</td><td>32</td><td>√</td><td>88.8 [−4.8,+5.0]</td><td>0.224 [-28%,-13%]</td><td></td><td>8.8 [-19.2,-8.8]</td><td></td><td>0.034 [-38%,-2%]</td></tr><tr><td>PF w/ weighted action averaging</td><td>weighted mean</td><td>32</td><td>√</td><td> $\mathbf { 9 2 . 0 \ } [ - 0 . 2 , + 7 . 2 ]$ </td><td> $0 . 1 8 7 \left[ - 4 1 \% , - 2 8 \% \right]$ </td><td></td><td> $1 1 . 2 [ - 1 6 . 0 , - 7 . 0 ]$ </td><td></td><td>0.031 [-38%,-15%]</td></tr><tr><td>PF w/ best-particle action selection</td><td>best particle</td><td>32</td><td>√</td><td> $\mathbf { 9 2 . 0 \ } [ + 0 . 4 , + 7 . 0 ]$ </td><td> $0 . 1 9 4 \ [ - 3 8 \% , - 2 5 \% ]$ </td><td></td><td> $\mathbf { 2 8 . 6 } _ { \ [ + 1 . 0 , + 1 0 . 6 ] }$ </td><td></td><td> $0 . 0 3 3 \ [ - 3 0 \% , - 1 5 \% ]$ </td></tr></table>

![](images/472a9d6c3197803c1910db240daf532ca0bea6a460809ddfba6fe39cdfce818e.jpg)  
Fig. 4. Camera observations from a validation set episode of the ToolHang task. All of our manipulation experiments withhold access to the shoulder camera, so the policy must infer actions and state solely from the wrist camera’s view.

## G. Extension to Visuomotor Manipulation

1) Datasets: We consider two datasets based on the Transport and ToolHang tasks from Robomimic [17]. States concatenate object positions and 6D rotations: three object poses for Transport (27 dimensions) and two for ToolHang (18 dimensions). The original dataset includes two available camera views per manipulator: one fixed to the end effector (wrist camera) and another at a static external location (shoulder camera). Since standard diffusion policies already perform strongly when all cameras are available, we instead remove the static cameras, keeping only the wrist camera. As illustrated in Fig. 4, this makes the tasks substantially harder, since the shoulder camera shows task-relevant state in almost every individual frame, while the wrist camera sees only changing and partial views of the scene.

2) Model & Training: The observation encoder $\mathcal { E } _ { \mathrm { o b s } }$ uses the same convolutional image encoder architecture as Diffusion Policy [1]. A block-causal transformer is used for the denoising network, just like for the navigation task.

3) Baselines: For manipulation, we evaluate both stateless and persistent variants of Diffusion Policy and the ensemble baseline. Stateless variants regenerate all state tokens within the current context window at each planning iteration, including those corresponding to past observations, conditioned on the observation-action history. Persistent variants retain previously sampled states and feed them back as conditioning at subsequent iterations. This comparison is possible because manipulation has no explicit goal condition to preserve, while in navigation, persistent states carry goal information forward after the initial goal token leaves the context window. The persistent ensemble retains a separate state history for each member and uses uniform observation scores and uniform action averaging. As in navigation, we compare best-particle action selection with weighted action averaging for particle filtering.

4) Results (Table V): We evaluate each method on 50 scenarios with 10 different random seeds, totaling 500 closedloop simulations per task. State estimates are computed as the componentwise weighted mean, and RMSE is computed across state components and timesteps. State estimation errors are reported only for successful rollouts to reduce confounding from differences in task progress and failure modes. We again report paired 95% confidence intervals for changes from Diffusion Policy.

Using the same trained model, particle filtering with bestparticle action selection improves success and reduces state RMSE relative to Diffusion Policy on both tasks. These results extend the benefits of inference-time filtering to RGBbased manipulation under restricted visual observations.

Retaining sampled state histories does not automatically provide useful memory for control. Single-particle persistence offers no clear success benefit, suggesting that carrying forward an uncorrected hypothesis can be insufficient under partial observability. Maintaining multiple histories with uniform averaging likewise provides no reliable control improvement, even though it lowers state-estimation error. This highlights the value of maintaining competing hypotheses and using observation evidence to determine their influence on control.

ToolHang highlights a potential limitation of averaging action proposals under ambiguity. Proposals associated with different plausible behaviors may each be reasonable individually, yet yield ineffective actions when averaged. Bestparticle selection preserves an individual proposed action sequence and achieves better task performance despite slightly higher reported state RMSE. The stronger dependence on the choice of action selection rule in ToolHang (as well as in our earlier navigation experiments) suggests that the consequences of action aggregation vary across tasks.

## VIII. CONCLUSION

We presented an inference-time approach that uses a joint generative model of states, actions, and observations for both particle filtering and control. Each particle maintains a state-history hypothesis and an associated action proposal, while agreement between predicted and measured observations updates its weight. These weights guide action selection, allowing observation evidence to influence control through persistent hypotheses. The procedure uses the same trained model as direct action generation, without additional filtering-specific training or separately learned dynamics, observation, or policy models.

Experiments in partially observed navigation and manipulation demonstrate that this approach can improve both state estimation and task performance. Controlled studies demonstrate the importance of useful state-action hypotheses, observation-consistent weighting, particle coverage, and the choice of action-selection rule. In navigation, filtering improves control with limited training data and remains useful even with a large synthetic training dataset when obstacle layouts differ from those seen during training. For manipulation, where collecting demonstrations is costly, filtering improves control without requiring more training data.

Particle filtering also provides a natural way to incorporate information available at deployment, such as known maps or explicit sensor models, by using it to assess generated hypotheses without retraining the model. More broadly, it provides a principled framework for using additional inference-time computation with pretrained joint trajectory models.

## REFERENCES

[1] C. Chi, S. Feng, Y. Du, Z. Xu, E. Cousineau, B. Burchfiel, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proceedings of Robotics: Science and Systems (RSS), 2023.

[2] M. Janner, Y. Du, J. Tenenbaum, and S. Levine, “Planning with diffusion for flexible behavior synthesis,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 162. PMLR, 2022, pp. 9902– 9915.

[3] A. Ajay, Y. Du, A. Gupta, J. B. Tenenbaum, T. S. Jaakkola, and P. Agrawal, “Is conditional generative modeling all you need for decision making?” in The Eleventh International Conference on Learning Representations, 2023.

[4] C. Zhu, R. Yu, S. Feng, B. Burchfiel, P. Shah, and A. Gupta, “Unified world models: Coupling video and action diffusion for pretraining on large robotic datasets,” in Proceedings of Robotics: Science and Systems, 2025.

[5] B. Chen, D. Mart´ı Monso, Y. Du, M. Simchowitz,´ R. Tedrake, and V. Sitzmann, “Diffusion forcing: Next-token prediction meets full-sequence diffusion,”

Advances in Neural Information Processing Systems, vol. 37, pp. 24 081–24 125, 2024.

[6] R. Jonschkowski, D. Rastogi, and O. Brock, “Differentiable particle filters: End-to-end learning with algorithmic priors,” in Proceedings ofRobotics: Science and Systems, Pittsburgh, Pennsylvania, Jun. 2018.

[7] P. Karkus, D. Hsu, and W. S. Lee, “Particle filter networks with application to visual localization,” in Proceedings of the 2nd Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 87. PMLR, 2018, pp. 169–178.

[8] Z. Wan and L. Zhao, “DiffPF: Differentiable particle filtering with generative sampling via conditional diffusion models,” IEEE Robotics and Automation Letters, vol. 11, no. 3, pp. 3166–3173, 2026.

[9] D. Silver and J. Veness, “Monte-carlo planning in large pomdps,” in Advances in Neural Information Processing Systems, vol. 23, 2010, pp. 2164–2172.

[10] Z. N. Sunberg and M. J. Kochenderfer, “Online algorithms for pomdps with continuous state, action, and observation spaces,” in Proceedings of the Twenty-Eighth International Conference on Automated Planning and Scheduling, 2018, pp. 259–263.

[11] M. Du and S. Song, “DynaGuide: Steering diffusion polices with active dynamic guidance,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[12] R. Singhal, Z. Horvitz, R. Teehan, M. Ren, Z. Yu, K. McKeown, and R. Ranganath, “A general framework for inference-time scaling and steering of diffusion models,” in Forty-second International Conference on Machine Learning, 2025.

[13] N. J. Gordon, D. J. Salmond, and A. F. Smith, “Novel approach to nonlinear/non-gaussian bayesian state estimation,” in IEE Proceedings F (Radar and Signal Processing), vol. 140, no. 2. IET, 1993, pp. 107–113.

[14] K. Song, B. Chen, M. Simchowitz, Y. Du, R. Tedrake, and V. Sitzmann, “History-guided video diffusion,” in Forty-second International Conference on Machine Learning, 2025.

[15] R. C. Coulter, “Implementation of the pure pursuit path tracking algorithm,” The Robotics Institute, Carnegie Mellon University, Pittsburgh, PA, Tech. Rep. CMU-RI-TR-92-01, Jan. 1992.

[16] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in 2023 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2023, pp. 4172–4182.

[17] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın, “What matters in learning from offline human demonstrations for robot manipulation,” in Conference on Robot Learning (CoRL), 2021.