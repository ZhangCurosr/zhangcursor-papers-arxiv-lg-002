# Role-Adaptive Policy Optimization for Offline Reinforcement Learning

Seonvin Cho Soohyun Choi Songnam Hong

Department of Electronic Engineering, Hanyang University Seoul, Republic of Korea

seonbin0319@hanyang.ac.kr petersun0221@hanyang.ac.kr snhong@hanyang.ac.kr

September 2026

## Abstract

Policy regularization in offline reinforcement learning balances policy improvement against reliance on uncertain value estimates. This balance can differ between selecting actions for execution and supplying actions for critic bootstrapping, yet methods such as TD3+BC couple these roles through a shared policy. We propose Role-Adaptive Policy Optimization (RAPO), which adapts policy-update coefficients according to their roles in value learning and execution. RAPO learns these coefficients by differentiating through candidate policy updates formed using the base algorithm’s actor objective. For TD3+BC, RAPO separates bootstrap and execution actors and adapts their coefficients independently: the bootstrap objective penalizes policy-induced changes in target values, while the execution objective evaluates a local policy-improvement surrogate. For IQL, whose value learning is already independent of the execution actor, RAPO preserves the original value updates and adapts only the inverse temperature in advantage-weighted policy extraction. Experiments on D4RL locomotion and AntMaze tasks show improvements over both base algorithms, with larger gains for TD3+BC, whose RAPO instantiation outperforms baselines on average.

## 1 Introduction

Many offline reinforcement learning (RL) methods improve a policy by balancing critic-guided updates against a reference derived from recorded behavior [Fujimoto and Gu, 2021, Tarasov et al., 2023]. The critic is learned from a fixed dataset, so its estimates can be unreliable for actions poorly covered by that dataset [Levine et al., 2020]. The regularization coefficient controls how strongly an update follows these estimates. A setting that is useful at one stage of training may become overly restrictive or permissive as the actor and critic evolve.

The appropriate balance also depends on the role of the policy. In TD3+BC, one actor selects actions for execution and, through its target copy, supplies next-state actions for critic bootstrapping. An update toward higher estimated values can improve the execution policy while changing the critic’s regression targets. Conversely, limiting target movement can restrict useful execution updates. These two uses therefore need not favor the same regularization, even when they share a critic.

IQL learns values without using its extracted actor, and methods with separate target and evaluation policies allow different constraint strengths [Kostrikov et al., 2022, Xu et al., 2024]. Policy extraction can substantially affect performance for a given value function [Hansen-Estruch et al., 2023, Park et al., 2024]. Adaptive-constraint methods can learn regularization coefficients through differentiable actor updates [Tan et al., 2026]. The evaluation criterion for such an update should reflect whether the policy supplies execution actions or critic targets.

We propose Role-Adaptive Policy Optimization (RAPO), which learns regularization coefficients by evaluating candidate policy updates according to how their actions are used. Execution updates are assessed for local policy improvement, while bootstrap updates are assessed for the stability of the critic targets they produce. RAPO differentiates through a candidate update to choose its coefficient and retains the base algorithm’s actor objective for the actual update.

We instantiate this procedure in TD3+BC and IQL. TD3+RAPO separates execution and bootstrap actors and learns their coefficients independently. IQL learns Q and V from dataset actions without using its execution actor in either update [Kostrikov et al., 2022]. IQL+RAPO therefore adapts only the inverse temperature of advantage-weighted policy extraction. An explicit bootstrap actor is used when the base method obtains critic targets from policy actions.

The contribution is a role-specific criterion for learning each policy-update coefficient while retaining the base actor losses. Across fifteen D4RL tasks, both instantiations improve the domain averages of their base algorithms, and TD3+RAPO has the highest overall average among the compared methods. Matched TD3 controls examine the execution score and bootstrap adaptation; a fixed-coefficient control tests whether learned endpoint values reproduce adaptation during training.

## 2 Background

## 2.1 Offline Learning Setup

The training experience in offline RL is a fixed dataset $\boldsymbol { \mathcal { D } } = \{ ( s , a , r , s ^ { \prime } ) \}$ collected by behavior policies [Levine et al., 2020]. Each tuple records a state $s \in S$ , action $a \in { \mathcal { A } }$ , reward $r ,$ and successor state $s ^ { \prime } .$ . Learning uses these records without further environment interaction.

We seek a policy $\pi$ that maximizes $J ( \pi ) = \mathbb { E } _ { \rho _ { 0 } , P , \pi } [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r ( s _ { t } , a _ { t } ) ]$ , where $\rho _ { 0 }$ is the initialstate distribution, $P$ is the environment transition kernel, r is the reward function, and $\gamma \in [ 0 , 1 )$ is the discount factor. The actor-critic methods below represent the policy by $\pi _ { \theta }$ and estimate action and state values with $Q _ { \phi }$ and $V _ { \psi }$ , respectively.

## 2.2 Base Actor-Critic Updates

TD3+BC and IQL differ in their actor objectives and in the actions used for critic bootstrapping. Let $\bar { Q } _ { j }$ denote target critic $j$ and $\bar { Q } _ { \mathrm { m i n } } = \operatorname* { m i n } _ { j = 1 , 2 } \bar { Q } _ { j }$ their pointwise minimum. Bars consistently mark target critics. Let sg denote the stop-gradient operator. We write the losses for nonterminal transitions; implementation details are given in Appendix B.2.

TD3+BC. TD3+BC combines Q maximization with behavior cloning [Fujimoto and Gu, 2021]. In the loss convention used by RAPO, its actor minimizes

$$
\mathcal { L } _ { \pi } ^ { \mathrm { T D 3 + B C } } ( \theta ; \alpha ) = \mathbb { E } _ { ( s , a ) \sim \mathcal { D } } \left[ - \frac { Q _ { \phi _ { 1 } } ( s , \pi _ { \theta } ( s ) ) } { S } + \frac { \| \pi _ { \theta } ( s ) - a \| _ { 2 } ^ { 2 } } { \alpha d _ { a } } \right] ,\tag{1}
$$

where $\alpha > 0 , d _ { a }$ is the action dimension, and $S = \mathrm { s g } ( \operatorname* { m a x } \{ \mathbb { E } _ { s \sim \mathcal { D } } \vert Q _ { \phi _ { 1 } } ( s , \pi _ { \theta } ( s ) ) \vert , \epsilon \} )$ for a small $\epsilon > 0$ . Increasing α weakens behavior regularization relative to $\mathrm { Q }$ maximization. The critic loss is

$$
\mathcal { L } _ { Q } ^ { \mathrm { T D 3 } } ( \phi ) = \sum _ { j = 1 } ^ { 2 } \mathbb { E } _ { ( s , a , r , s ^ { \prime } ) \sim \mathcal { D } } \left[ \left( Q _ { \phi _ { j } } ( s , a ) - \mathrm { s g } \big ( r + \gamma \bar { Q } _ { \mathrm { m i n } } ( s ^ { \prime } , \bar { \pi } _ { \theta } ( s ^ { \prime } ) ) \big ) \right) ^ { 2 } \right] .\tag{2}
$$

In TD3+BC, the actor in Equation (1) selects execution actions; its target copy in Equation (2) selects the next-state actions used for critic bootstrapping. RAPO assigns these two uses to separate actors.

IQL. IQL learns values from dataset actions independently of its actor [Kostrikov et al., 2022]:

$$
\mathcal { L } _ { V } ( \psi ) = \mathbb { E } _ { ( s , a ) \sim \mathcal { D } } \ell _ { \tau } \left( \mathrm { s g } ( \bar { Q } _ { \mathrm { m i n } } ( s , a ) ) - V _ { \psi } ( s ) \right) ,\tag{3}
$$

$$
\mathcal { L } _ { Q } ^ { \mathrm { I Q L } } ( \phi ) = \frac { 1 } { 2 } \sum _ { j = 1 } ^ { 2 } \mathbb { E } _ { ( s , a , r , s ^ { \prime } ) \sim \mathcal { D } } \left[ \left( Q _ { \phi _ { j } } ( s , a ) - \mathrm { s g } ( r + \gamma V _ { \psi } ( s ^ { \prime } ) ) \right) ^ { 2 } \right] ,\tag{4}
$$

where $\ell _ { \tau } ( u ) = | \tau - { \bf 1 } \{ u < 0 \} | u ^ { 2 }$ and $\tau \in [ 1 / 2 , 1 )$ select an upper expectile (the mean at $\tau = 1 / 2 )$ Given the detached advantage $A ( s , a ) = \mathrm { s g } ( \bar { Q } _ { \mathrm { m i n } } ( s , a ) - V _ { \psi } ( s ) )$ , the actor minimizes

$$
\mathcal { L } _ { \pi } ^ { \mathrm { I Q L } } ( \theta ; \beta ) = - \mathbb { E } _ { ( s , a ) \sim \mathcal { D } } \left[ \exp ( \beta A ( s , a ) ) \log \pi _ { \theta } ( a \mid s ) \right] .\tag{5}
$$

The inverse temperature $\beta > 0$ controls how strongly policy extraction favors higher-advantage actions.

## 2.3 Meta-Gradient Learning

Meta-gradient learning trains a coefficient by differentiating through the update it controls [Xu et al., 2018]. For an actor loss $\mathcal { L } _ { \mathrm { i n n e r } }$ and dataset minibatch $B _ { \mathrm { i n } }$ , an inner update changes θ with α fixed. A gradient step with learning rate η gives

$$
\widetilde { \theta } ( \alpha ) = \theta - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { i n n e r } } ( \theta ; \alpha , B _ { \mathrm { i n } } ) .
$$

An outer loss $\mathcal { L } _ { \mathrm { o u t e r } }$ scores the updated actor on another minibatch $ { { B _ { \mathrm { o u t } } } }$ . Holding the starting parameters θ fixed, the chain rule gives

$$
\frac { \mathrm { d } \mathcal { L } _ { \mathrm { o u t e r } } ( \widetilde { \theta } ( \alpha ) ; \mathcal { B } _ { \mathrm { o u t } } ) } { \mathrm { d } \alpha } = \left( \frac { \partial \widetilde { \theta } } { \partial \alpha } \right) ^ { \top } \nabla _ { \widetilde { \theta } } \mathcal { L } _ { \mathrm { o u t e r } } .\tag{6}
$$

Gradient descent on the outer loss trains α: the inner loss trains the policy, and the outer loss trains its update coefficient.

## 3 Method

RAPO applies the meta-gradient procedure in Section 2.3 with outer losses tailored to execution and critic bootstrapping (Figure 1).

## 3.1 Policy Roles and Adaptive Coefficients

The execution policy $\pi _ { E }$ selects actions in the environment. When value learning requires a policy to supply next-state actions, RAPO assigns that task to a separate bootstrap policy $\pi _ { B }$ . For each active role i, write $c _ { i }$ for its method-specific positive actor-loss coefficient:

<table><tr><td>Action use</td><td>Actor</td><td>Coefficient</td><td>Outer criterion</td></tr><tr><td>Execution</td><td> $\pi _ { E } = \pi _ { \theta _ { E } }$ </td><td> $c _ { E }$ </td><td>Local value improvement</td></tr><tr><td>Critic-target actions</td><td> $\pi _ { B } = \pi _ { \theta _ { B } }$ </td><td> $c _ { B }$ </td><td>Target stability</td></tr></table>

TD3+RAPO uses both actors with shared critics, with $c _ { E } = \alpha _ { E }$ and $c _ { B } = \alpha _ { B } \colon \pi _ { E }$ is executed, while $\pi _ { B }$ supplies the actions in Equation (2). IQL’s target in Equation (4) is actor-independent, so its only required role is execution, with $c _ { E } = \beta _ { E }$

For each active role i, the inner loss $\mathcal { L } _ { \mathrm { i n n e r } , i }$ is the corresponding actor objective from Section 2.2: Equation (1) with $\alpha = c _ { i }$ for TD3+BC, or Equation (5) with $\beta = c _ { E }$ for IQL.

Let $\mathcal { U } _ { i } ( \theta _ { i } ; c _ { i } , B _ { \mathrm { i n } } )$ denote one differentiable optimizer step on $\mathcal { L } _ { \mathrm { i n n e r } , i }$ from $\theta _ { i } ,$ using coefficient $c _ { i }$ and minibatch $\scriptstyle B _ { \mathrm { i n } }$ . The resulting candidate parameters and policy are

$$
\widetilde { \theta } _ { i } ( c _ { i } ) = \mathcal { U } _ { i } ( \theta _ { i } ; c _ { i } , \mathcal { B } _ { \mathrm { i n } } ) , \qquad \widetilde { \pi } _ { i } = \pi _ { \widetilde { \theta } _ { i } ( c _ { i } ) } .\tag{7}
$$

The outer losses evaluate $\widetilde { \pi } _ { i }$ on an independently drawn minibatch $ { { B _ { \mathrm { o u t } } } }$ and update $c _ { i }$ through Equation (6). Positive, detached scales $\bar { S } _ { i }$ normalize the scores by mean absolute critic values at candidate actions. Appendix B.1 specifies the scales and stop-gradient operations; Appendix B.2 gives the optimizers and schedules.

![](images/b397be9c015b023258310b4348c640d70b181f58d1ee10aa57cb68a9dbeb2130.jpg)  
Figure 1: Overview of RAPO. $\pi _ { \beta }$ denotes the behavior represented by the offline data. (a) TD3+RAPO separates the bootstrap actor $\pi _ { B }$ from the execution actor π and adapts their coefficients $\alpha _ { B }$ and $\alpha _ { E }$ using target-stability and local-improvement objectives, respectively. (b) IQL+RAPO retains actor-independent Q/V learning from dataset transitions and adapts only the execution policy’s inverse temperature $\beta _ { E }$

## 3.2 Execution Objective: Local Policy Improvement

The performance-difference identity relates return improvement to the candidate policy’s action-value gain under $Q ^ { \pi _ { E } }$ , averaged over its discounted state distribution (Appendix A.2). RAPO approximates this criterion using a learned critic and dataset states, and scores the local change from the current action to the candidate action.

Write $\bar { Q } _ { E }$ for the target critic used to score execution: $\bar { Q } _ { E } = \bar { Q } _ { 1 }$ <sub>1</sub> in TD3+RAPO and $\bar { Q } _ { E } = \bar { Q } _ { \operatorname* { m i n } }$ in IQL+RAPO. It is held fixed while scoring an update. Here π denotes the action map used in evaluation, including the mean for Gaussian actors.

For a state s, write $a _ { 0 } = \pi _ { E } ( s ) , a _ { + } = \widetilde \pi _ { E } ( s )$ , and $\delta a = a _ { + } - a _ { 0 }$ . A standard smoothness bound motivates an update term that rewards first-order improvement and a conservatism term that accounts for curvature.

Lemma 1 (Local critic improvement). Fix s. Suppose $\bar { Q } _ { E } ( s , \cdot )$ is differentiable on a neighborhood of the segment from $a _ { 0 }$ to $a _ { + }$ and its action gradient is L<sub>s</sub>-Lipschitz on that segment. Then

$$
\bar { Q } _ { E } ( s , a _ { + } ) - \bar { Q } _ { E } ( s , a _ { 0 } ) \geq \nabla _ { a } \bar { Q } _ { E } ( s , a _ { 0 } ) ^ { \top } \delta a - \frac { L _ { s } } { 2 } \| \delta a \| _ { 2 } ^ { 2 } .\tag{8}
$$

To make the curvature penalty computable, define the detached endpoint gradients

$$
g _ { 0 } = \mathrm { s g } ( \nabla _ { a } \bar { Q } _ { E } ( s , a _ { 0 } ) ) , \qquad g _ { + } = \mathrm { s g } ( \nabla _ { a } \bar { Q } _ { E } ( s , a _ { + } ) ) .
$$

Under a local quadratic approximation with Hessian $H _ { s }$ , the endpoint-gradient difference satisfies $g _ { + } - g _ { 0 } = H _ { s } \delta a$ . Hence,

$$
\left| \frac { 1 } { 2 } \delta \boldsymbol { a } ^ { \top } H _ { s } \delta \boldsymbol { a } \right| \leq \frac { 1 } { 2 } \| \boldsymbol { g } _ { + } - \boldsymbol { g } _ { 0 } \| _ { 2 } \| \delta \boldsymbol { a } \| _ { 2 } .
$$

RAPO uses this computable penalty to discount the first-order update term, requiring two action gradients without forming a Hessian. This motivates the following surrogate score and execution

loss:

$$
\widehat { B } _ { \pi } ( s ) = \underbrace { g _ { 0 } ^ { \top } \delta a } _ { \mathrm { u p d a t e } } - \underbrace { \frac { 1 } { 2 } \| g _ { + } - g _ { 0 } \| _ { 2 } \| \delta a \| _ { 2 } } _ { \mathrm { c o n s e r v a t i s m } } ,\tag{9}
$$

$$
\mathcal { L } _ { E } = - \frac { \mathbb { E } _ { s \sim B _ { \mathrm { o u t } } } \widehat { B } _ { \pi } ( s ) } { \bar { S } _ { E } } .\tag{10}
$$

Minimizing $\mathcal { L } _ { E }$ favors coefficients whose candidate updates have large first-order gains and small endpoint-gradient penalties. With $g _ { 0 } , g _ { + }$ , and $\bar { S } _ { E }$ detached, the coefficient gradient passes through $\delta a .$ . The penalty bounds the magnitude of a quadratic correction under the stated approximation; endpoint gradients alone do not establish a bound for a general critic. Appendix A.2 identifies the additional critic and state-distribution errors involved in relating this score to return improvement.

## 3.3 Bootstrap Objective: Target Stability

A bootstrap-policy update can move the critic’s regression targets by changing the next-state actions in Equation (2). To isolate the effect of the policy update, we evaluate the current and candidate actions using the same fixed target critics.

For a transition $\boldsymbol { x } = ( s , a , r , s ^ { \prime } , d )$ , let d = 1 indicate termination and define

$$
y ( \pi ; x ) = r + \gamma ( 1 - d ) \bar { Q } _ { \mathrm { m i n } } \bigl ( s ^ { \prime } , \pi ( s ^ { \prime } ) \bigr ) .
$$

The candidate-induced target change is

$$
\begin{array} { r l } & { \Delta y _ { B } ( x ) = y ( \widetilde { \pi } _ { B } ; x ) - \mathrm { s g } ( y ( { \pi } _ { B } ; x ) ) } \\ & { \qquad = \gamma ( 1 - d ) \left[ \bar { Q } _ { \operatorname* { m i n } } ( s ^ { \prime } , \widetilde { \pi } _ { B } ( s ^ { \prime } ) ) - \mathrm { s g } ( \bar { Q } _ { \operatorname* { m i n } } ( s ^ { \prime } , { \pi } _ { B } ( s ^ { \prime } ) ) ) \right] . } \end{array}\tag{11}
$$

We penalize its root-mean-square magnitude, so positive and negative target changes cannot cancel:

$$
\mathcal { L } _ { B } = \mathrm { s g } ( \alpha _ { B } ) \sqrt { \mathbb { E } _ { x \sim \mathcal { B } _ { \mathrm { o u t } } } \left[ \left( \frac { \Delta y _ { B } ( x ) } { \bar { S } _ { B } } \right) ^ { 2 } \right] } .\tag{12}
$$

The critic loss in Equation (2) depends on values at bootstrap actions. With the target critics fixed, $\Delta y _ { B } ( x )$ measures how an unsmoothed Bellman target for the same transition responds to a candidate update of the online bootstrap actor. Penalizing its RMS favors updates with smaller target sensitivity while the inner TD3+BC loss still trains the actor for policy improvement. The actual training target uses a smoothed target actor (Appendix B.2), so this objective is a proxy for target stability. Algorithm 1 combines the outer objectives with the base updates.

```latex
Algorithm 1 RAPO (optimizer timing and schedules in Appendix B.2)
Input: Dataset $\mathcal { D } ;$ base method $m \in \{ \mathrm { T D 3 + B C , I Q L } \}$ ; active roles ${ \mathcal { T } } ( m ) = \{ E , B \} { \mathrm { ~ o r ~ } } \{ E \}$
respectively.
Initialize actors, coefficients, and value/target networks.
for each training step do
Sample $B _ { \mathrm { i n } } ;$ apply the value updates of m from Section 2.2.
if coefficient update is due then
Sample independent $B _ { \mathrm { o u t } } ;$ form each role’s candidate via $( 7 ) .$
Update each $c _ { i }$ via (6), using $\mathcal { L } _ { E }$ in (10) or $\mathcal { L } _ { B }$ in (12).
Apply the scheduled actor and target updates of m.
end for
Output: Trained value networks and actors $\{ \pi _ { i } : i \in \mathcal { T } ( m ) \}$ ; execute $\pi _ { E } .$
```

## 4 Experimental Results

## 4.1 Evaluation Protocol

We evaluate TD3+RAPO and IQL+RAPO against TD3+BC, IQL, wPC [Peng et al., 2023], A2PR [Liu et al., 2024], and ASPC [Tan et al., 2026] on fifteen D4RL tasks [Fu et al., 2020]: nine Locomotion tasks spanning HalfCheetah, Hopper, and Walker2d with medium, medium-replay, and mediumexpert data, and six AntMaze tasks. The base actor-critic updates follow the CORL implementations [Tarasov et al., 2022], with network architectures and training settings detailed in Appendix B.

All final scores are measured at one million training steps: each of four independent training runs is evaluated over 50 episodes, and its episode mean gives one seed score. We report means and sample standard deviations across these four scores. Table 1 reports normalized returns for each task; domain and overall averages weight environments equally.

Table 1: D4RL performance at one million training steps, reported as mean ± sample standard deviation across four training seeds. Initial coefficients are fixed at five for both RAPO methods. Hyperparameters are reported in Appendix B.2. Average rows give equal weight to environments; the final row averages all fifteen tasks. Bold marks the highest mean in each row, including ties.
<table><tr><td>Dataset</td><td>TD3+BC</td><td>IQL</td><td>wPC</td><td>A2PR</td><td>ASPC</td><td>TD3+RAPO IQL+RAPO</td><td></td></tr><tr><td>HalfCheetah-m</td><td>50.0±0.7</td><td>48.4±0.2</td><td> $5 4 . 7 { \pm } 0 . 4 $ </td><td>55.9±0.6</td><td>57.8±0.7</td><td>58.2±0.5</td><td>48.6±0.2</td></tr><tr><td>HalfCheetah-mr</td><td>46.2±0.5</td><td>44.2±0.4</td><td> $4 8 . 2 { \pm } 0 . 4 $ </td><td> $4 8 . 0 { \pm } 1 . 1 $ </td><td>50.1±1.3</td><td>49.7±0.7</td><td>44.6±0.5</td></tr><tr><td>HalfCheetah-me</td><td>99.5±1.0</td><td>89.5±3.9</td><td>100.0±0.5</td><td>98.9±1.7</td><td>92.2±12.8</td><td>100.6±3.0</td><td>92.8±0.6</td></tr><tr><td>Hopper-m</td><td>61.5±2.7</td><td>63.0±2.5</td><td>83.9±6.8</td><td>83.4±2.5</td><td>81.9±5.2</td><td>101.8±0.2</td><td>64.4±5.8</td></tr><tr><td>Hopper-mr</td><td>78.8±34.2</td><td>87.6±8.4</td><td>101.2±0.6</td><td>100.1±1.1</td><td>99.6±4.0</td><td>101.5±0.6</td><td>94.1±15.4</td></tr><tr><td>Hopper-me</td><td>111.4±2.0 99.8±24.0</td><td></td><td></td><td>103.5±7.4 112.6±0.5</td><td>109.2±2.7</td><td> $1 1 1 . 0 { \pm } 1 . 3 $ </td><td>108.3±5.1</td></tr><tr><td>Walker2d-m</td><td>83.7±2.8</td><td>81.4±3.5</td><td>88.9±1.4</td><td>89.3±1.3</td><td>89.5±15.4</td><td>93.8±1.4</td><td>83.2±2.2</td></tr><tr><td>Walker2d-mr</td><td>86.7±5.6</td><td>81.5±4.5</td><td>93.5±3.5</td><td>91.5±5.1</td><td>90.6±9.4</td><td>98.0±1.4</td><td> $8 2 . 4 { \pm } 6 . 5$ </td></tr><tr><td>Walker2d-me</td><td>109.7±0.7 111.8±1.2</td><td></td><td>110.3±0.5</td><td>112.0±0.3</td><td>110.5±0.2</td><td>111.1±1.2</td><td>112.5±0.8</td></tr><tr><td>Locomotion average</td><td>80.8</td><td>78.6</td><td>87.1</td><td>88.0</td><td>86.8</td><td>91.7</td><td>81.2</td></tr><tr><td>Ant-umaze</td><td>90.0±8.2</td><td>76.0±14.0</td><td>92.5±9.6</td><td>92.5±5.0</td><td>88.0±3.7</td><td>92.5±6.4</td><td>77.5±10.8</td></tr><tr><td>Ant-umaze-diverse</td><td>87.5±5.0</td><td>58.0±5.9</td><td>80.0±8.2</td><td>55.0±41.2</td><td>82.5±4.7</td><td>88.0±6.9</td><td>63.0±7.0</td></tr><tr><td>Ant-medium-play</td><td>2.5±5.0</td><td>63.5±6.2</td><td>70.0±14.1</td><td>45.0±30.0</td><td>76.0±12.4</td><td>81.5±3.0</td><td>73.0±6.8</td></tr><tr><td>Ant-medium-diverse</td><td>10.0±8.2</td><td>66.0±13.1</td><td>60.0±14.1</td><td>47.5±9.6</td><td>49.5±22.3</td><td>59.0±17.2</td><td>73.0±11.8</td></tr><tr><td>Ant-large-play</td><td>0.0±0.0</td><td>35.0±10.4</td><td>32.5±5.0</td><td>15.0±12.9</td><td> ${ \bf 5 3 . 5 \pm 1 9 . 7 }$ </td><td>48.0±6.9</td><td>45.5±5.7</td></tr><tr><td>Ant-large-diverse</td><td>0.0±0.0</td><td>34.0±11.0</td><td>45.0±17.3</td><td> $1 5 . 0 { \pm } 1 7 . 3 $ </td><td> $3 2 . 5 { \pm } 2 0 . 9 \ $ </td><td>51.0±5.3</td><td>40.5±4.4</td></tr><tr><td>AntMaze average</td><td>31.7</td><td>55.4</td><td>63.3</td><td>45.0</td><td>63.7</td><td>70.0</td><td>62.1</td></tr><tr><td>Overall average</td><td>61.2</td><td>69.3</td><td>77.6</td><td>70.8</td><td>77.6</td><td>83.0</td><td>73.5</td></tr></table>

## 4.2 Benchmark Performance

TD3+RAPO averages 91.7 on Locomotion and 70.0 on AntMaze, compared with 80.8 and 31.7 for TD3+BC. IQL+RAPO averages 81.2 and 62.1, compared with 78.6 and 55.4 for IQL. Both RAPO variants exceed their respective baselines in the two domain averages, and TD3+RAPO has the highest overall average among the compared methods.

![](images/f1be8ef4bfdc881e000370cc081d690cd9881d50c016a12214f8ced04d4ef46f.jpg)  
Figure 2: Learned coefficients on three Hopper datasets and AntMaze-medium-play. The top row shows $\mathrm { T D } 3 { + } \mathrm { R A P O } ^ { \cdot } \mathrm { s } \alpha _ { B }$ and $\alpha _ { E } ;$ the bottom row shows $\operatorname { I Q L + R A P O ^ { \prime } s } \beta _ { E }$ . Curves and shaded bands show means and sample standard deviations across seeds. The dotted line marks the common initial value of five.

Learned coefficient trajectories. Figure 2 shows the coefficients learned from equal initial values. On Hopper-medium, Hopper-medium-replay, and AntMaze-medium-play, $\mathrm { T D } 3 { + } \mathrm { R A P O } ^ { * } \mathrm { s } \alpha _ { E }$ rises above $\alpha _ { B } ;$ on Hopper-medium-expert they remain close. In IQL+RAPO, $\beta _ { E }$ decreases on the three Hopper datasets and increases on AntMaze-medium-play. Larger $\alpha _ { i }$ in TD3+RAPO weakens cloning relative to Q maximization, whereas larger $\beta _ { E }$ in IQL+RAPO increases the weight on higher-advantage dataset actions before clipping; their numerical values represent different scales.

## 4.3 Execution-Objective Ablation

With the actor structure and bootstrap objective held fixed, we compare ${ \widehat { B } } _ { \pi }$ with two simpler scores to test the criterion used for execution-coefficient adaptation.

Q uses the direct change in the fixed execution critic defined in Section $3 . 2 , f _ { Q } ( s ) = \bar { Q } _ { E } ( s , a _ { + } ) -$ $\mathrm { s g } ( \bar { Q } _ { E } ( s , a _ { 0 } ) )$ ), for the same candidate actions $a _ { 0 } , a _ { + }$

Linear uses only the first-order gain from Equation $( 9 ) , f _ { \mathrm { L i n e a r } } ( s ) = g _ { 0 } ^ { \top } \delta a$ , omitting the endpoint gradient penalty.

Each score defines an execution loss $- \mathbb { E } { \boldsymbol { B } } _ { \mathrm { o u t } } [ f ( \boldsymbol { s } ) ] / \bar { \boldsymbol { S } } _ { E }$ , using the normalization and detachment rules in Appendix B.1.

Q and Linear change only the execution score; they retain the two actors, RMS bootstrap objective, and coefficient learning rate of TD3+RAPO in each environment. All three conditions cover the fifteen benchmark tasks.

Table 2 shows that the full score $\widehat { B } _ { \pi }$ in $\mathcal { L } _ { E }$ has the highest means in both domains and overall. Its overall mean is 83.0, compared with 78.8 for Q and 79.4 for Linear. The Linear control isolates the endpoint-gradient penalty; its lower averages suggest that this term helps select execution-coefficient updates. The full score also exceeds direct Q change under the same actor structure and bootstrap objective. These comparisons support $\mathcal { L } _ { E }$ as an effective criterion for evaluating candidate execution updates in this setting.

Table 2: Execution-score ablation at the selected coefficient learning rates. Q uses target-critic value change; Linear uses the first-order gain. Entries are equal-weight task means within each domain and overall. Bold marks the highest mean.
<table><tr><td>Domain</td><td>Q</td><td>Linear</td><td> $\widehat { B } _ { \pi } \left( \mathrm { O u r s } \right)$ </td></tr><tr><td>Locomotion</td><td>87.0</td><td>88.7</td><td>91.7</td></tr><tr><td>AntMaze</td><td>66.6</td><td>65.5</td><td>70.0</td></tr><tr><td>Overall</td><td>78.8</td><td>79.4</td><td>83.0</td></tr></table>

## 4.4 Adaptation During Training

Bootstrap adaptation. To isolate bootstrap adaptation, we compare adapting $\alpha _ { B }$ from five with fixing it at five. Both conditions retain two actors, adapt $\alpha _ { E }$ from five, and use the same actor and critic settings and coefficient learning rate. Table 3 shows higher averages with bootstrap adaptation in both domains, particularly on AntMaze (55.8 to 70.0). The overall mean increases from 75.9 to 83.0. Gains are particularly large on AntMaze-large-play (20.0 to 48.0) and AntMaze-large-diverse (13.0 to 51.0). Appendix B.5 reports the per-task results and a fixed-α = 1 control.

Table 3: Bootstrap-coefficient adaptation with $\alpha _ { E }$ adapted in both conditions. The control fixes $\alpha _ { B } = 5 ;$ TD3+RAPO adapts it from five. Entries are equal-weight task means within each domain and overall. Per-task results appear in Table 8.
<table><tr><td>Domain</td><td>Fixed  $\alpha _ { B } = 5$ </td><td>TD3+RAPO</td></tr><tr><td>Locomotion</td><td>89.3</td><td>91.7</td></tr><tr><td>AntMaze</td><td>55.8</td><td>70.0</td></tr><tr><td>Overall</td><td>75.9</td><td>83.0</td></tr></table>

Fixed learned endpoints. We next test whether fixed coefficients reproduce adaptation throughout training. For each environment, we retrain a two-actor agent with both coefficients fixed to the corresponding four-seed means of TD3+RAPO’s final coefficients.

Table 4 shows a higher overall mean with adaptation (83.0 versus 78.4), with higher means in both domains. The per-task results in Appendix B.5 show that the Locomotion difference is concentrated in Hopper-medium-expert, while four of six AntMaze tasks improve. Learned final coefficient values therefore do not reproduce the adaptive domain averages when fixed from the start.

Table 4: Fixing learned endpoint coefficients from the start versus adapting from five. Both conditions retain separate TD3 actors. Scores are equal-weight task means; fixed constants and per-task results appear in Table 9.
<table><tr><td>Domain</td><td>Fixed endpoints</td><td>TD3+RAPO</td></tr><tr><td>Locomotion</td><td>88.7</td><td>91.7</td></tr><tr><td>AntMaze</td><td>62.9</td><td>70.0</td></tr><tr><td>Overall</td><td>78.4</td><td>83.0</td></tr></table>

## 5 Related Work

Behavior regularization. Offline RL must balance exploiting learned values against selecting actions outside dataset coverage [Levine et al., 2020]. TD3+BC applies a quadratic BC penalty to the TD3 actor [Fujimoto et al., 2018, Fujimoto and Gu, 2021], while CQL regularizes value estimates toward pessimism [Kumar et al., 2020]. ReBRAC combines architectural changes with separate regularization coefficients for actor learning and critic targets [Tarasov et al., 2023]. These interventions act at different points in learning. RAPO retains the base value-loss form and adapts actor-loss coefficients, including the coefficient of the actor supplying bootstrap actions.

Adaptive constraints and meta-gradients. Weighted Policy Constraints favor desirable dataset actions [Peng et al., 2023], A2PR constructs an advantage-guided behavior reference [Liu et al., 2024], and selective regularization varies constraint strength across states [Luo et al., 2025]. Meta-gradients learn update parameters by differentiating through optimization [Xu et al., 2018, Franceschi et al., 2018]. ASPC uses this mechanism for policy-constraint scales, including IQL’s inverse temperature [Tan et al., 2026]. Its outer loss includes the squared change in mean $\mathrm { Q } , ( \mathbb { E } \Delta Q ) ^ { 2 }$ . RAPO instead assigns local improvement to execution and per-transition target RMS, $\sqrt { \mathbb { E } ( \Delta y ) ^ { 2 } }$ , to bootstrapping. The latter prevents opposite-signed target changes from canceling. The distinction is the role-specific evaluation criterion, not the use of meta-gradients or value-change regularization itself.

Decoupling value learning and execution. Separating the policies involved in value learning and deployment has several precedents. One-step RL estimates the behavior policy’s value and then performs a regularized improvement step without repeatedly evaluating the improved policy [Brandfonbrener et al., 2021]. IQL and XQL learn values from dataset actions without requiring the extracted actor in their offline Bellman updates [Kostrikov et al., 2022, Garg et al., 2023]. MCEP retains an explicit target policy and trains a separate, more mildly constrained evaluation policy [Xu et al., 2024]. RAPO learns how strongly to regularize each actor by evaluating candidate updates according to the actor’s role in execution or value learning.

Value learning and policy extraction. The extraction procedure determines how learned values are converted into executable actions. IDQL relates IQL’s value objective to an implicit actor and uses critic-weighted samples from a diffusion behavior model for extraction [Hansen-Estruch et al., 2023]. Experiments that vary value learning and extraction independently show that the extraction objective can substantially change performance even with the same value function [Park et al., 2024]. Other decoupled methods rank behavior-model proposals at inference time [Lin et al., 2026]. RAPO instead learns an explicit execution actor during training. Its IQL instantiation retains advantage-weighted regression and adapts its inverse temperature using a local critic-improvement score; it does not change the policy class or introduce inference-time search.

Local policy improvement. Performance-difference and trust-region analyses connect action-value improvement to return change under assumptions on the critic and state distribution [Schulman et al., 2015]. RAPO uses a single candidate update to assess a coefficient, retaining the base actor objective. Its endpoint-gradient score is a local surrogate for this assessment; the conditional smoothness bound and the additional errors separating that score from return improvement are stated in Section 3.2 and Appendix A.

## 6 Conclusion

RAPO learns a policy-update coefficient by testing a candidate update against a criterion for the policy’s use. In TD3+BC, this gives separate execution and bootstrap actors: the former is evaluated by a local critic-improvement surrogate, the latter by its effect on critic targets. In IQL, value updates already use dataset actions, so only the execution policy’s extraction coefficient is adapted.

Both instantiations improve over their respective baselines in domain-average return on the fifteen D4RL tasks. Within the two-actor TD3 setting, the full execution score and adapting the bootstrap coefficient achieve higher domain averages than the tested controls; fixed learned endpoint coefficients also yield lower averages than continued adaptation. These findings motivate choosing coefficient-learning criteria according to how policy actions enter the algorithm. The execution score remains a local surrogate, and its relation to return improvement depends on the learned critic and the state distribution.

## AI use statement

Generative AI tools assisted with discovering relevant literature; drafting, editing, and polishing portions of the manuscript; and developing and checking mathematical arguments, including the local critic-improvement lemma and its proof in Appendix A.1. The authors reviewed the AI-assisted work and take responsibility for the final text, citations, proofs, technical claims, and results.

## References

David Brandfonbrener, Will Whitney, Rajesh Ranganath, and Joan Bruna. Offline RL Without Off-Policy Evaluation. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 274a10ffa06e434f2a94df765cac6bf4-Abstract.html.

Luca Franceschi, Paolo Frasconi, Saverio Salzo, Riccardo Grazzi, and Massimiliano Pontil. Bilevel Programming for Hyperparameter Optimization and Meta-Learning. In ICML, 2018. URL https://proceedings.mlr.press/v80/franceschi18a.html.

Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4RL: Datasets for Deep Data-Driven Reinforcement Learning. arXiv preprint arXiv:2004.07219, 2020. URL https://arxiv.org/abs/2004.07219.

Scott Fujimoto and Shixiang Shane Gu. A Minimalist Approach to Offline Reinforcement Learning. In NeurIPS, 2021. URL https://arxiv.org/abs/2106.06860.

Scott Fujimoto, Herke van Hoof, and David Meger. Addressing Function Approximation Error in Actor-Critic Methods. In ICML, 2018. URL https://proceedings.mlr.press/v80/ fujimoto18a.html.

Divyansh Garg, Joey Hejna, Matthieu Geist, and Stefano Ermon. Extreme Q-Learning: MaxEnt RL without Entropy. In ICLR, 2023. URL https://arxiv.org/abs/2301.02328.

Philippe Hansen-Estruch, Ilya Kostrikov, Michael Janner, Jakub Grudzien Kuba, and Sergey Levine. IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies. arXiv preprint arXiv:2304.10573, 2023. URL https://arxiv.org/abs/2304.10573.

Ilya Kostrikov, Ashvin Nair, and Sergey Levine. Offline Reinforcement Learning with Implicit Q-Learning. In ICLR, 2022. URL https://arxiv.org/abs/2110.06169.

Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. Conservative Q-Learning for Offline Reinforcement Learning. In NeurIPS, 2020. URL https://arxiv.org/abs/2006. 04779.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. arXiv preprint arXiv:2005.01643, 2020. URL https://arxiv.org/abs/2005.01643.

Xuyao Lin, Yixiang Shan, Jinru Duan, Tao Yang, Xinyu Zhao, Runyu Lei, Yiming Zhao, Jiaxin Fan, Zongbao Feng, and Peng Jia. Decoupling Policy Extraction for Offline Reinforcement Learning. arXiv preprint arXiv:2608.20909, 2026. URL https://arxiv.org/abs/2608.20909.

Tenglong Liu, Yang Li, Yixing Lan, Hao Gao, Wei Pan, and Xin Xu. Adaptive Advantage-Guided Policy Regularization for Offline Reinforcement Learning. In ICML, 2024. URL https:// proceedings.mlr.press/v235/liu24ai.html.

Qin-Wen Luo, Ming-Kun Xie, Ye-Wen Wang, and Sheng-Jun Huang. Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 41450–41467, 2025. URL https://proceedings.mlr.press/v267/ luo25p.html.

Seohong Park, Kevin Frans, Sergey Levine, and Aviral Kumar. Is Value Learning Really the Main Bottleneck in Offline RL? In NeurIPS, 2024. URL https://arxiv.org/abs/2406. 09329.

Zhiyong Peng, Changlin Han, Yadong Liu, and Zongtan Zhou. Weighted Policy Constraints for Offline Reinforcement Learning. In AAAI, 2023. URL https://ojs.aaai.org/index. php/AAAI/article/view/26130.

John Schulman, Sergey Levine, Pieter Abbeel, Michael I. Jordan, and Philipp Moritz. Trust Region Policy Optimization. In ICML, 2015. URL https://proceedings.mlr.press/v37/ schulman15.html.

Jing Tan, Xiaorui Li, Chao Yao, Xiaojuan Ban, Yuetong Fang, Renjing Xu, and Zhaolin Yuan. Adaptive Scaling of Policy Constraints for Offline Reinforcement Learning. In ICLR, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 713939ee5a7b14a6fecf69e692ec3799-Abstract-Conference.html.

Denis Tarasov, Alexander Nikulin, Dmitry Akimov, Vladislav Kurenkov, and Sergey Kolesnikov. CORL: Research-oriented deep offline reinforcement learning library. In 3rd Offline RL Workshop: Offline RL as a ”Launchpad”, 2022. URL https://openreview.net/forum?id= SyAS49bBcv.

Denis Tarasov, Vladislav Kurenkov, Alexander Nikulin, and Sergey Kolesnikov. Revisiting the Minimalist Approach to Offline Reinforcement Learning. In NeurIPS, 2023. URL https: //arxiv.org/abs/2305.09836.

Linjie Xu, Zhengyao Jiang, Jinyu Wang, Lei Song, and Jiang Bian. Mildly Constrained Evaluation Policy for Offline Reinforcement Learning. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=imAROs79Pb.

Zhongwen Xu, Hado van Hasselt, and David Silver. Meta-Gradient Reinforcement Learning. In NeurIPS, 2018. URL https://arxiv.org/abs/1805.09801.

## A Analysis of the Execution Objective

## A.1 Proof of Lemma 1

Proof. Fix s and write $f ( a ) = { \bar { Q } } _ { E } ( s , a )$ . The fundamental theorem of calculus and the Lipschitz condition give

$$
\begin{array} { r l } & { \displaystyle f ( \boldsymbol { a } _ { + } ) - f ( \boldsymbol { a } _ { 0 } ) - \nabla f ( \boldsymbol { a } _ { 0 } ) ^ { \top } \delta \boldsymbol { a } = \int _ { 0 } ^ { 1 } \big [ \nabla f ( \boldsymbol { a } _ { 0 } + t \delta \boldsymbol { a } ) - \nabla f ( \boldsymbol { a } _ { 0 } ) \big ] ^ { \top } \delta \boldsymbol { a } \mathrm { d } t } \\ & { \qquad \geq - \displaystyle \int _ { 0 } ^ { 1 } L _ { s } t \| \delta \boldsymbol { a } \| _ { 2 } ^ { 2 } \mathrm { d } t = - \frac { L _ { s } } { 2 } \| \delta \boldsymbol { a } \| _ { 2 } ^ { 2 } . } \end{array}
$$

The smoothness premise can fail at activation boundaries $^ { \mathrm { o r , } }$ for IQL’s minimum critic, where the minimizing Q-network switches.

Quadratic interpretation of the conservatism term. If $\bar { Q } _ { E } ( s , a )$ is quadratic in a with symmetric Hessian $H _ { s }$ , then $g _ { + } - g _ { 0 } = H _ { s } \delta a$ and

$$
\begin{array} { r l } & { \bar { Q } _ { E } ( s , a _ { + } ) - \bar { Q } _ { E } ( s , a _ { 0 } ) = g _ { 0 } ^ { \top } \delta a + \frac { 1 } { 2 } \delta a ^ { \top } H _ { s } \delta a , } \\ & { \qquad \quad \big | \frac { 1 } { 2 } \delta a ^ { \top } H _ { s } \delta a \big | \leq \frac { 1 } { 2 } \| H _ { s } \delta a \| _ { 2 } \| \delta a \| _ { 2 } = \frac { 1 } { 2 } \| g _ { + } - g _ { 0 } \| _ { 2 } \| \delta a \| _ { 2 } . } \end{array}
$$

Thus Equation (9) subtracts an upper bound on the magnitude of the quadratic correction in this special case. For a general critic, endpoint variation need not control interior curvature; the score is not a certified lower bound. This curvature penalty also does not quantify uncertainty in the critic.

## A.2 Relation to Return Improvement

For deterministic policies π and $\pi ^ { + }$ with a common initial-state distribution, the performancedifference identity is [Schulman et al., 2015]

$$
J ( \pi ^ { + } ) - J ( \pi ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { s \sim d ^ { \pi ^ { + } } } \left[ Q ^ { \pi } ( s , \pi ^ { + } ( s ) ) - Q ^ { \pi } ( s , \pi ( s ) ) \right] ,\tag{13}
$$

where $d ^ { \pi ^ { + } }$ is normalized discounted state occupancy. To relate this identity to RAPO, set $\pi = \pi _ { E }$ $\pi ^ { + } = \widetilde { \pi } _ { E }$ , let ν be the dataset state distribution, and define $\Delta _ { E } ( s ) = \bar { Q } _ { E } ( s , \pi ^ { + } ( s ) ) - \bar { Q } _ { E } ( s , \pi ( s ) )$ Assuming integrability, adding and subtracting $\Delta _ { E }$ gives the exact decomposition

$$
\begin{array} { r l } & { ( 1 - \gamma ) \big [ J ( \pi ^ { + } ) - J ( \pi ) \big ] = \mathbb { E } _ { \nu } \widehat { B } _ { \pi } + \mathcal { R } _ { \mathrm { l o c a l } } + \mathcal { R } _ { \mathrm { c r i t i c } } + \mathcal { R } _ { \mathrm { s t a t e } } , } \\ & { \qquad \mathcal { R } _ { \mathrm { l o c a l } } = \mathbb { E } _ { \nu } \Big [ \Delta _ { E } - \widehat { B } _ { \pi } \Big ] , } \\ & { \qquad \mathcal { R } _ { \mathrm { c r i t i c } } = \mathbb { E } _ { d ^ { \pi ^ { + } } } \big [ Q ^ { \pi } ( s , \pi ^ { + } ( s ) ) - Q ^ { \pi } ( s , \pi ( s ) ) - \Delta _ { E } ( s ) \big ] , } \\ & { \qquad \mathcal { R } _ { \mathrm { s t a t e } } = \mathbb { E } _ { d ^ { \pi ^ { + } } } \Delta _ { E } - \mathbb { E } _ { \nu } \Delta _ { E } . } \end{array}\tag{14}
$$

These residuals capture the endpoint approximation, critic error relative to $Q ^ { \pi }$ (including referencepolicy mismatch), and the use of dataset states. RAPO does not bound them. Even if the learned critic is smooth, endpoint gradients need not control interior curvature; hence the execution score does not guarantee return improvement.

## B Algorithm Instantiations and Implementation Details

## B.1 Differentiation and Normalization

The optimizer updates an unconstrained scalar $\rho _ { i }$ for each active role, with the coefficient $c _ { i }$ from Section 3.1:

$$
\begin{array} { r l r l r } & { \mathrm { T D 3 + R A P O } ; } & { c _ { i } = \alpha _ { i } = \mathrm { s o f t p l u s } ( \rho _ { i } ) , } & { i \in \{ B , E \} , } & { \quad } & { \underline { { \mathrm { d } } } \mathcal { L } _ { i } = \underline { { \mathrm { d } } } c _ { i } \underline { { \mathrm { d } } } \mathcal { L } _ { i } } \\ & { \mathrm { I Q L + R A P O } ; } & { c _ { E } = \beta _ { E } = \exp ( \rho _ { E } ) , } & & { } & { \mathrm { d } \rho _ { i } = \mathrm { d } \rho _ { i } \overline { { \mathrm { d } } } c _ { i } } \end{array} .
$$

Each $\rho _ { i }$ has a separate Adam state. The second factor follows Equation (6) with the specified stop-gradient operations.

The candidate step leaves real actor parameters and optimizer state unchanged. Starting actor parameters, critic parameters, dataset-action references, and incoming optimizer states are constants for coefficient differentiation; new gradients and optimizer moments remain differentiable in $c _ { i }$ Detached quantities are recomputed at each evaluation.

Using the critics from Sections 3.2 and 3.3, the detached value scales are defined as follows, with $\epsilon = 1 0 ^ { - 6 }$

$$
\bar { S } _ { E } = \mathrm { s g } \big ( \operatorname* { m a x } \big \{ \mathbb { E } _ { \mathcal { B } _ { \mathrm { o u t } } } \vert \bar { Q } _ { E } ( s , \widetilde { \pi } _ { E } ( s ) ) \vert , \epsilon \big \} \big ) ,\tag{15}
$$

$$
\bar { S } _ { B } = \mathrm { s g } \big ( \mathbb { E } _ { \boldsymbol { B } _ { \mathrm { o u t } } } | \bar { Q } _ { \mathrm { m i n } } ( \boldsymbol { s } , \widetilde { \pi } _ { B } ( \boldsymbol { s } ) ) | \big ) + \epsilon .\tag{16}
$$

The bootstrap prefactor $\displaystyle \mathrm { s g } ( \alpha _ { B } )$ scales the coefficient gradient without adding a direct derivative path; dependence on $\alpha _ { B }$ enters through the candidate policy. The bootstrap RMS includes terminal transitions as zeros and has no additive epsilon inside the square root; its value and chosen subgradient are zero when all inputs are zero.

## B.2 Training Settings

The actor-critic implementations follow CORL [Tarasov et al., 2022]. For the TD3+BC, wPC, and ASPC comparisons, we use the same Q-network architecture as $\mathrm { T D } 3 { + } \mathrm { R A } \mathrm { P O }$ three hidden layers of width 256 with ReLU activations and LayerNorm in each hidden layer.

Both actual and candidate TD3 actor updates use the loss scale in Equation (1); multiplying it by $\alpha _ { i }$ changes finite updates and their meta-gradients. The training critic uses the target bootstrap actor with clipped Gaussian action noise $( \sigma = 0 . 2 , c = 0 . 5 )$ , whereas Equation (11) compares unsmoothed online actors. Both methods mask the next-state value by $1 - d ,$ where d indicates termination.

IQL clips $\exp ( \beta _ { E } A )$ at 100. Gaussian actor means are used in the execution score and at evaluation.

Table 5 summarizes RAPO’s training settings. At initialization five, we select each method’s coefficient learning rate from $\{ 3 \times 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 2 \times 1 0 ^ { - 3 } \}$ by its four-seed mean in each environment. Table 6 lists the selected rates.

Table 5: Training settings. Intervals are measured in value-update steps; $\bar { Q }  ( 1 - \tau _ { Q } ) \bar { Q } + \tau _ { Q } Q$ defines the target-update rate.
<table><tr><td>Setting</td><td>TD3+RAPO</td><td>IQL+RAPO</td></tr><tr><td>Actor hidden layers</td><td> $2 \times 2 5 6$ </td><td> $2 \times 2 5 6$ </td></tr><tr><td>Value hidden layers</td><td> $\mathrm { Q } \mathrm { : 3 \times 2 5 6 }$  , LayerNorm</td><td>Q and  $\mathrm { V } ; 2 \times 2 5 6 , \mathrm { n o }$  LayerNorm</td></tr><tr><td>Network optimizer / initial rate</td><td> $\mathrm { A d a m / 3 \times 1 0 ^ { - 4 } }$ </td><td>Adam  $/ 3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Actor learning-rate schedule</td><td>Constant</td><td>Cosine decay</td></tr><tr><td>Candidate optimizer</td><td>E: current-state Adam; B: SGD</td><td>E: current-state Adam</td></tr><tr><td>Actual actor optimizer / interval</td><td>Adam / 2</td><td>Adam / 1</td></tr><tr><td>Coefficient initialization</td><td> $\begin{array} { c } { { \alpha _ { B } = \alpha _ { E } = 5 } } \\ { { 2 0 } } \end{array}$ </td><td> $\beta _ { E } = 5$ </td></tr><tr><td>Coefficient update interval Coefficient rate schedule</td><td></td><td>20, after 100,000 steps</td></tr><tr><td></td><td>Exponential to 0.01 of initial rate</td><td>Constant</td></tr><tr><td>Coefficient range</td><td> $\alpha _ { i } > 0$ </td><td> $\beta _ { E } \in [ 0 . 0 5 , 1 0 0 ]$ </td></tr><tr><td>Target-update rate  $\tau _ { Q }$ </td><td>0.005</td><td>0.005</td></tr></table>

Candidate execution steps use the actor’s current Adam moments and learning rate. The bootstrap candidate uses SGD at the actor learning rate; the actual bootstrap update uses Adam. Network Adam uses moments (0.9, 0.999) and epsilon $1 0 ^ { - 8 }$ . TD3 coefficient Adam uses the same moments; IQL coefficient Adam uses (0, 0.999), with $\rho _ { E }$ projected to [log 0.05, log 100] after each update without resetting its moments.

Table 6: Main-selected coefficient learning rates with initialization five. TD3 uses the listed rate for both coefficients; its fixed-bootstrap control reuses the same rate for $\alpha _ { E } .$ . Fixed-endpoint TD3 has no active coefficient update.
<table><tr><td>Environment</td><td> $\mathrm { T D 3 + R A P O }$ </td><td> $\mathrm { I Q L { + } R A P O }$ </td></tr><tr><td>HalfCheetah-medium</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>HalfCheetah-medium-replay</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>HalfCheetah-medium-expert</td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Hopper-medium</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Hopper-medium-replay</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Hopper-medium-expert</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td> $\mathrm { W a l k e r 2 d \mathrm { - } m e d i u m }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $\mathrm { W a l k e r 2 d - m e d i u m - r e p l a y }$ </td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Walker2d-medium-expert</td><td> $2 \times 1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td> $\scriptstyle \mathbf { A n t M a z e - u m a z e }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td> $\mathrm { A n t M a z e - u m a z e - d i v e r s e }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> ${ \mathrm { A n t M a z e } } { \mathrm { - m e d i u m } } { \mathrm { - p l a y } }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td> $\mathrm { A n t M a z e  – m e d i u m \mathrm { - } d i v e r s e }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td> $_ { \mathrm { A n t M a z e - l a r g e - p l a y } }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td> ${ \mathrm { A n t M a z e  – l a r g e  – d i v e r s e } }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $1 0 ^ { - 3 }$ </td></tr></table>

For both instantiations, the actual execution update uses the coefficient cached before that step’s meta-update. TD3 first updates its critics, then, on scheduled steps, updates $\rho _ { E }$ and $\rho _ { B }$ , applies both real actor updates (using the new $\alpha _ { B } )$ , and updates its target networks. The TD3 bootstrap target actor uses ${ \bar { \theta } } _ { B }  ( 1 - \tau _ { Q } ) { \bar { \theta } } _ { B } + \tau _ { Q } \theta _ { B }$ with the same rate as its target critics. IQL caches the advantage and next-state value, updates $\mathrm { v } , \mathrm { Q }$ , and target Q, performs the scheduled execution meta-update, and applies its real execution update.

Both instantiations use batch size 256, discount 0.99, normalized observations, and unchanged Locomotion rewards. TD3’s smoothed target actions are clipped to [−1, 1].

IQL+RAPO uses Gaussian actors and an expectile of $0 . 7$ on Locomotion and 0.9 on AntMaze. AntMaze rewards are shifted by −1. Actor means use tanh, and log standard deviations are stateindependent and bounded in [−20, 2].

IQL baseline. The benchmark baseline uses the same actor family, target-update rate, expectiles, and reward handling, with fixed $\beta = 3$ on Locomotion and $\beta = 1 0$ on AntMaze.

## B.3 Evaluation and Configuration Selection

The final-score and uncertainty protocol is defined in Section 4.1. For domain uncertainty, environments are averaged within each seed before computing the sample standard deviation across seeds.

## B.4 Coefficient Learning-Rate Sensitivity

Table 7 tests TD3+RAPO with each initial coefficient learning rate shared across environments; other settings follow Table 5. It exceeds TD3+BC’s domain averages at all three rates, though task-level performance varies. $\mathrm { A t 3 \times 1 0 ^ { - 4 } }$ , its Locomotion and AntMaze means are 87.4 and $6 4 . 2 ,$ compared with 80.8 and 31.7 for TD3+BC.

Table 7: TD3+RAPO sensitivity to the initial coefficient learning rate, with coefficient initialization five. Average rows weight environments equally.
<table><tr><td>Environment</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 3 }$ </td><td> $2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>HalfCheetah-medium</td><td> $5 4 . 3 \pm 0 . 6$ </td><td> $5 6 . 4 \pm 0 . 7$ </td><td> $5 8 . 2 \pm 0 . 5$ </td></tr><tr><td>HalfCheetah-medium-replay</td><td> $4 7 . 9 \pm 0 . 5$ </td><td> $4 9 . 3 \pm 0 . 3$ </td><td> $4 9 . 7 \pm 0 . 7$ </td></tr><tr><td>HalfCheetah-medium-expert</td><td> $9 8 . 1 \pm 2 . 8 $ </td><td> $1 0 0 . 6 \pm 3 . 0$ </td><td> $9 8 . 0 \pm 3 . 9$ </td></tr><tr><td>Hopper-medium</td><td> $8 3 . 9 \pm 1 1 . 2$ </td><td> $9 6 . 4 \pm 5 . 7$ </td><td> $1 0 1 . 8 \pm 0 . 2$ </td></tr><tr><td>Hopper-medium-replay</td><td> $9 6 . 7 \pm 6 . 4$ </td><td> $9 9 . 8 \pm 2 . 6 $ </td><td> $1 0 1 . 5 \pm 0 . 6$ </td></tr><tr><td>Hopper-medium-expert</td><td> $1 1 1 . 0 \pm 1 . 3$ </td><td> $7 9 . 1 \pm 2 0 . 3$ </td><td> $4 5 . 4 \pm 5 . 5$ </td></tr><tr><td>Walker2d-medium</td><td> $8 9 . 6 \pm 1 . 7$ </td><td> $9 3 . 8 \pm 1 . 4$ </td><td> $8 8 . 7 \pm 1 2 . 7$ </td></tr><tr><td>Walker2d-medium-replay</td><td> $9 5 . 0 \pm 2 . 5$ </td><td> $9 4 . 0 \pm 4 . 5$ </td><td> $9 8 . 0 \pm 1 . 4$ </td></tr><tr><td>Walker2d-medium-expert</td><td> $1 1 0 . 0 \pm 0 . 5$ </td><td> $1 1 0 . 8 \pm 0 . 7$ </td><td> $1 1 1 . 0 \pm 1 . 2$ </td></tr><tr><td>Locomotion  $\operatorname { A v g } .$ </td><td>87.4</td><td>86.7</td><td>83.6</td></tr><tr><td>AntMaze-umaze</td><td> $9 2 . 5 \pm 6 . 4$ </td><td> $8 8 . 0 \pm 4 . 3 $ </td><td> $8 5 . 0 \pm 7 . 4$ </td></tr><tr><td>AntMaze-umaze-diverse</td><td> $8 8 . 0 \pm 6 . 9$ </td><td> $3 5 . 5 \pm 2 9 . 9$ </td><td> $7 7 . 0 \pm 1 2 . 7$ </td></tr><tr><td>AntMaze-medium-play</td><td> $7 2 . 5 \pm 1 4 . 9$ </td><td> $8 1 . 5 \pm 3 . 0$ </td><td> $7 1 . 0 \pm 9 . 5$ </td></tr><tr><td>AntMaze-medium-diverse</td><td> $5 9 . 0 \pm 1 7 . 2$ </td><td> $4 3 . 0 \pm 2 0 . 0$ </td><td> $2 6 . 5 \pm 1 9 . 8$ </td></tr><tr><td>AntMaze-large-play</td><td> $2 9 . 5 \pm 1 3 . 4$ </td><td> $4 8 . 0 \pm 6 . 9$ </td><td> $4 6 . 5 \pm 1 2 . 6$ </td></tr><tr><td>AntMaze-large-diverse</td><td> $4 3 . 5 \pm 5 . 3$ </td><td> $5 1 . 0 \pm 5 . 3$ </td><td> $3 0 . 0 \pm 1 4 . 6$ </td></tr><tr><td>AntMaze Avg.</td><td>64.2</td><td>57.8</td><td>56.0</td></tr><tr><td>Overall  $\operatorname { A v g } .$ </td><td>78.1</td><td>75.1</td><td>72.6</td></tr></table>

## B.5 Outer-Objective and Coefficient Ablations

Bootstrap adaptation at main-selected settings. Table 8 extends the fixed- $x _ { B } = 5$ comparison to fixed and adaptive values of one. All four conditions retain two actors, adapt $\alpha _ { E }$ from five, and use TD3+RAPO’s actor and critic settings and selected coefficient learning rate.

Figure 3 extends the domain summary in Table 3. Adapting $\alpha _ { B }$ from one raises the Locomotion mean from 88.2 to 90.9 and the AntMaze mean from 23.8 to 38.5 relative to fixing it at one. Bootstrap adaptation therefore improves both domain averages at either initial value. Initialization still matters, especially on AntMaze, where adaptation from five reaches 70.0 compared with 38.5 from one.

![](images/6d53d6e7b123d4b84ef269e43ab4a9fda1dbc40ea0559edafcd618a551d61a5b.jpg)  
Figure 3: Fixed and adaptive bootstrap regularization at $\mathrm { T D 3 + R A P O ^ { \circ } s }$ selected learning rates. Bars show domain or overall task averages; error bars show standard deviations across seed-wise task averages. All conditions adapt $\alpha _ { E }$ from five; adaptive $\alpha _ { B }$ starts at one or five. Per-task results appear in Table 8.

Table 8: Fixed and adaptive bootstrap regularization at the selected learning rates. All conditions retain two actors and adapt $\alpha _ { E }$ from five. Under Fixed, column numbers are constant $\alpha _ { B }$ values; under Adaptive, they are initial values.
<table><tr><td rowspan="2">Environment</td><td colspan="2">Fixed  $\alpha _ { B }$ </td><td colspan="2">Adaptive  $\alpha _ { B }$ </td></tr><tr><td>1</td><td>5</td><td>1</td><td>5</td></tr><tr><td>HalfCheetah-medium</td><td> $5 6 . 6 \pm 0 . 9$ </td><td> $5 9 . 1 \pm 0 . 6$ </td><td> $5 8 . 1 \pm 1 . 0$ </td><td> $5 8 . 2 \pm 0 . 5$ </td></tr><tr><td>HalfCheetah-medium-replay</td><td> $4 8 . 7 \pm 2 . 4$ </td><td> $5 0 . 0 \pm 0 . 6$ </td><td> $4 9 . 1 \pm 0 . 5$ </td><td> $4 9 . 7 \pm 0 . 7$ </td></tr><tr><td>HalfCheetah-medium-expert</td><td> $1 0 3 . 1 \pm 3 . 6 $ </td><td> $9 8 . 8 \pm 3 . 7$ </td><td> $9 5 . 3 \pm 5 . 7$ </td><td> $1 0 0 . 6 \pm 3 . 0$ </td></tr><tr><td>Hopper-medium</td><td> $8 8 . 5 \pm 2 4 . 2 $ </td><td> $1 0 1 . 7 \pm 0 . 5$ </td><td> $1 0 1 . 9 \pm 0 . 7$ </td><td> $1 0 1 . 8 \pm 0 . 2$ </td></tr><tr><td>Hopper-medium-replay</td><td> $9 8 . 7 \pm 1 . 3 $ </td><td> $1 0 1 . 3 \pm 1 . 2$ </td><td> $1 0 0 . 5 \pm 1 . 5$ </td><td> $1 0 1 . 5 \pm 0 . 6$ </td></tr><tr><td>Hopper-medium-expert</td><td> $1 1 1 . 0 \pm 2 . 3$ </td><td> $9 9 . 1 \pm 1 0 . 1 $ </td><td> $1 1 2 . 5 \pm 0 . 2$ </td><td> $1 1 1 . 0 \pm 1 . 3$ </td></tr><tr><td>Walker2d-medium</td><td> $8 5 . 0 \pm 0 . 2$ </td><td> $9 0 . 3 \pm 3 . 1$ </td><td> $9 3 . 2 \pm { 1 . 8 }$ </td><td> $9 3 . 8 \pm 1 . 4$ </td></tr><tr><td>Walker2d-medium-replay</td><td> $9 1 . 9 \pm 1 . 8$ </td><td> $9 2 . 6 \pm 9 . 7$ </td><td> $9 6 . 5 \pm 3 . 6 $ </td><td> $9 8 . 0 \pm 1 . 4$ </td></tr><tr><td>Walker2d-medium-expert</td><td> $1 1 0 . 2 \pm { 0 . 8 }$ </td><td> $1 1 0 . 6 \pm 0 . 8$ </td><td> $1 1 0 . 9 \pm 0 . 8$ </td><td> $1 1 1 . 1 \pm 1 . 2$ </td></tr><tr><td>Locomotion average</td><td>88.2</td><td>89.3</td><td>90.9</td><td>91.7</td></tr><tr><td>AntMaze-umaze</td><td> $8 4 . 5 \pm 4 . 4$ </td><td> $9 0 . 0 \pm 4 . 3 $ </td><td> $9 0 . 0 \pm 1 . 6$ </td><td> $9 2 . 5 \pm 6 . 4$ </td></tr><tr><td>AntMaze-umaze-diverse</td><td> $5 8 . 5 \pm 2 6 . 3$ </td><td> $8 8 . 5 \pm 1 2 . 0$ </td><td> $5 7 . 5 \pm 3 9 . 5$ </td><td> $8 8 . 0 \pm 6 . 9$ </td></tr><tr><td>AntMaze-medium-play</td><td> $0 . 0 \pm 0 . 0$ </td><td> $6 2 . 0 \pm 5 . 4$ </td><td> $4 6 . 5 \pm 1 7 . 0$ </td><td> $8 1 . 5 \pm 3 . 0$ </td></tr><tr><td>AntMaze-medium-diverse</td><td> $0 . 0 \pm 0 . 0$ </td><td> $6 1 . 5 \pm 1 6 . 3$ </td><td> $6 . 5 \pm 6 . 0$ </td><td> $5 9 . 0 \pm 1 7 . 2$ </td></tr><tr><td>AntMaze-large-play</td><td> $0 . 0 \pm 0 . 0$ </td><td> $2 0 . 0 \pm 4 . 3$ </td><td> $2 4 . 5 \pm 1 5 . 3$ </td><td> $4 8 . 0 \pm 6 . 9$ </td></tr><tr><td>AntMaze-large-diverse</td><td> $0 . 0 \pm 0 . 0$ </td><td> $1 3 . 0 \pm 8 . 9$ </td><td> $6 . 0 \pm 9 . 5$ </td><td> $5 1 . 0 \pm 5 . 3$ </td></tr><tr><td>AntMaze average</td><td>23.8</td><td>55.8</td><td>38.5</td><td>70.0</td></tr><tr><td>Overall average</td><td>62.4</td><td>75.9</td><td>69.9</td><td>83.0</td></tr></table>

Fixed learned endpoints. For each environment, we use the mean final coefficients of the four TD3+RAPO seeds as a fixed pair, rounded to six decimal places for training. We train new actors and critics with coefficient updates disabled. Table 9 displays the constants and per-task results; the adaptive column uses the main benchmark.

Table 9: Endpoint-fixed TD3 control. Each coefficient pair averages the final values of the main runs and remains fixed throughout the control runs. Constants are displayed to one decimal place; training uses six decimal places. TD3+RAPO uses the main benchmark runs.
<table><tr><td>Environment</td><td>Fixed  $\alpha _ { E }$ </td><td>Fixed  $\alpha _ { B }$ </td><td>Fixed endpoints</td><td> $\mathrm { T D 3 + R A P O }$ </td></tr><tr><td>HalfCheetah-medium</td><td>19.4</td><td>13.4</td><td> $5 8 . 3 \pm 0 . 4$ </td><td> $5 8 . 2 \pm 0 . 5$ </td></tr><tr><td>HalfCheetah-medium-replay</td><td>18.8</td><td>17.4</td><td> $4 9 . 9 \pm 0 . 9$ </td><td> $4 9 . 7 \pm 0 . 7$ </td></tr><tr><td>HalfCheetah-medium-expert</td><td>12.6</td><td>10.2</td><td> $9 8 . 8 \pm 5 . 4$ </td><td> $1 0 0 . 6 \pm 3 . 0$ </td></tr><tr><td>Hopper-medium</td><td>19.6</td><td>17.0</td><td> $1 0 1 . 9 \pm 0 . 4$ </td><td> $1 0 1 . 8 \pm 0 . 2$ </td></tr><tr><td>Hopper-medium-replay</td><td>23.3</td><td>16.7</td><td> $1 0 1 . 0 \pm 1 . 1$ </td><td> $1 0 1 . 5 \pm 0 . 6$ </td></tr><tr><td>Hopper-medium-expert</td><td>7.1</td><td>7.1</td><td> $8 3 . 8 \pm 2 0 . 3$ </td><td> $1 1 1 . 0 \pm 1 . 3$ </td></tr><tr><td>Walker2d-medium</td><td>11.8</td><td>9.8</td><td> $9 5 . 5 \pm 6 . 8$ </td><td> $9 3 . 8 \pm 1 . 4$ </td></tr><tr><td>Walker2d-medium-replay</td><td>20.7</td><td>18.2</td><td> $9 8 . 2 \pm 2 . 6 $ </td><td> $9 8 . 0 \pm 1 . 4$ </td></tr><tr><td>Walker2d-medium-expert</td><td>19.3</td><td>12.8</td><td> $1 1 1 . 1 \pm 0 . 5$ </td><td> $1 1 1 . 1 \pm 1 . 2$ </td></tr><tr><td>AntMaze-umaze</td><td>7.8</td><td>6.7</td><td> $9 5 . 5 \pm 3 . 4$ </td><td> $9 2 . 5 \pm 6 . 4$ </td></tr><tr><td>AntMaze-umaze-diverse</td><td>7.5</td><td>6.3</td><td> $6 1 . 5 \pm 3 5 . 9$ </td><td> $8 8 . 0 \pm 6 . 9$ </td></tr><tr><td>AntMaze-medium-play</td><td>13.7</td><td>9.0</td><td> $7 5 . 5 \pm 9 . 1$ </td><td> $8 1 . 5 \pm 3 . 0$ </td></tr><tr><td>AntMaze-medium-diverse</td><td>7.7</td><td>6.0</td><td> $4 9 . 0 \pm 1 7 . 3$ </td><td> $5 9 . 0 \pm 1 7 . 2$ </td></tr><tr><td> $_ { \mathrm { A n t M a z e - l a r g e - p l a y } }$ </td><td>13.5</td><td>9.2</td><td> $5 3 . 0 \pm 6 . 8$ </td><td> $4 8 . 0 \pm 6 . 9$ </td></tr><tr><td>AntMaze-large-diverse</td><td>13.7</td><td>9.3</td><td> $4 3 . 0 \pm 1 3 . 2$ </td><td> $5 1 . 0 \pm 5 . 3$ </td></tr></table>