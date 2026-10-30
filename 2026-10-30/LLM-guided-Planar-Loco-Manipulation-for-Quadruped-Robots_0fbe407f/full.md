# LLM-guided Planar Loco-Manipulation for Quadruped Robots

Georges Jetti<sup>∗</sup>, Michael Khayyat, Jean-Pierre Sleiman, Stefano Arrigoni, and Francesco Braghin

Abstract—Robotic navigation typically assumes a collisionfree route to the goal exists; in cluttered environments with movable obstacles, however, reaching the goal may require physical interaction. Classical solutions encode these behaviors through geometric rules and task-specific logic, while learningbased approaches often constrain the decision space to lowdimensional motion commands or predefined action sets. We instead consider a broader decision space, jointly selecting the skill, relevant object, and geometric target. Our languageconditioned task planner, fine-tuned with LoRA on synthetic data, maps natural-language tasks and a structured scene representation to variable-length sequences of parameterized actions, executed by low-level reinforcement-learning policies. The planner learns the geometric relations underlying interactive navigation and outperforms a classical planner and a prompted frontier model on interactive navigation tasks. Comparisons with models trained from scratch indicate that composition does not require pretraining when the representation is structured, whereas pretraining yields more reliable geometric targets. At the low level, we introduce an end-to-end whole-body planar pushing policy that achieves a 93.3% success rate in simulation, versus 83.5% for a hierarchical baseline, while reducing completion time by more than 60%. On hardware, the hierarchy yields the correct skill composition in 44/45 trials and end-to-end success in 34/45. Project website: https://anonymousprojects-dev.github. io/LLM-guided-LocoManipulation-Quadrupeds/.

## I. INTRODUCTION

Conventional robotic navigation pipelines typically assume the existence of a feasible path to the goal and that the robot’s repertoire of navigation skills is sufficient to cope with structural constraints, such as large obstacles. In practice, these assumptions often fail. Instead, when no feasible collisionfree path exists in the current environment configuration, the robot must reason about the scene and generate coordinated sequences of locomotion and manipulation actions to modify the environment and create a feasible route to the objective, analogous to how biological agents actively alter their surroundings to achieve task objectives. This class of problems is known as navigation among movable obstacles (NAMO) in the literature and, more broadly, interactive navigation, where navigation and manipulation actions must be coordinated to reach otherwise inaccessible goals [1]–[10].

Conventional NAMO approaches effectively combine geometric planning and explicit interaction logic [1]–[4]; however, accommodating different interaction cases generally requires extending the associated geometric rules, heuristics, or taskspecific decision logic, increasing the engineering burden as the range of task configurations expands. Learning-based methods provide an alternative paradigm by acquiring interac tion strategies directly from experience [5]–[8]. To manage the resulting learning problem, existing approaches adopt different task-specific abstractions of the decision space. The authors in [6] directly learn a policy over forward and angular velocities for both local navigation and interaction. Although effective, this formulation does not explicitly represent alternative manipulation modes, and the object interaction is induced implicitly through the velocity policy, providing less control over how the environment should be modified. Moreover, as investigated in Sec. IV-A, coupling locomotion and planar manipulation through a common controller constrains the manipulation behavior, leading to lower success and higher task completion time under object variability. In [7], the policy instead emits navigation and manipulation actions and learns to predict their effects on the surrounding objects; however, the action set has predefined geometric targets, limiting the range of interactions that can be performed. In [8], a hierarchical policy is employed in which the high-level controller generates a compact pushing command specifying the contact location, relative contact orientation, and base velocity. Nonetheless, the problem is formulated around following a preplanned global path, limiting the set of interactions with the environment. The method in [5] jointly learns which segment of the robot to command and subsequently a continuous sub-goal. However, the evaluated tasks contain a limited set of task-level decisions: the relevant object to manipulate is effectively determined by the task structure rather than selected among multiple candidate objects based on their relation to the task objective and the environment. In contrast, we consider a planning problem in which the planner must jointly determine which skill to execute, which object to interact with (when applicable), and the geometric target associated with that skill. The cases considered in this work can admit multiple feasible task-level plans, while alternative choices of skill, object, or geometric target can result in infeasible plans. The planner must therefore reason over the relationship between the task objective and the scene geometry, rather than follow a prescribed interaction sequence. The resulting variable-length sequence of parame terized navigation and manipulation actions provides a flexible interface for composing different loco-manipulation strategies.

Pretrained foundation models provide a natural means of reasoning over such variable task-level decisions. Prior work employing LLMs and VLMs to coordinate locomotion and manipulation skills for quadrupedal robots commonly relies on prompt engineering and in-context learning (ICL [11]) [9], [12]–[17]. These approaches demonstrate that pretrained models can decompose long-horizon objectives, select among available skills, and, in some cases, infer continuous execution parameters. Nevertheless, the required planning behavior is elicited through task descriptions, skill specifications, and examples of how tasks can be solved, increasing the input context and inference latency during deployment. Moreover, pretrained LLMs are not inherently grounded in the robotspecific action space or operational constraints required for execution. Task-specific finetuning provides a means of adapting these pretrained representations to robotic planning and action generation [10], [18]–[26], alleviating the need for large input prompts while grounding the model’s outputs with respect to the robot’s capabilities and environment. Taskspecific finetuning, however, introduces its own costs: adapting large pretrained models can require updating a large number of parameters, while collecting sufficiently diverse expert demonstrations can be expensive.

Closely related to our formulation, [10] fine-tune a Vision Language Model (VLM) for interactive navigation to determine when environmental interaction is required and to select the corresponding interaction skill and object. The geometric outcome of the interaction, however, is determined downstream by the skill-execution pipeline. Moreover, its counterfactual reasoning evaluates the removal of a candidate object, whereas in cluttered scenes the object must be relocated to a pose that itself remains feasible with respect to the surrounding objects. In contrast, our planner predicts the geometric target associated with each action. For manipulation, this explicitly specifies the desired object configuration, allowing the task planner to reason not only about which object to manipulate, but also where it should be moved. By decoupling the manipulation skill from a task-specific geometric outcome, this parameterization provides direct control over how the environment is modified and enables the same goalconditioned manipulation skill to be reused across environment configurations.

The effectiveness of a high-level planner additionally depends on the capabilities of the low-level skills available to the robot. In particular, successful interactive navigation requires a manipulation policy that can reliably reposition obstacles while maintaining a stable behavior. Typically, quadruped manipulation is solved by mounting a multi-DOF arm on the quadruped body [27], [28]. An alternative paradigm for the manipulation of heavy objects is to leverage the base of the quadruped. Although limited to non-prehensile manipulation, it has become a viable option to push objects of various sizes, shapes, and masses both in a planar setting [9], [12], [29], on slopes [30], and collaboratively with multiple robots [31], [32]. Nevertheless, existing solutions constrain the manipulation controller by defining a specific gait cycle through a scheduler [30], [32], or more commonly by formulating a hierarchical loco-manipulation framework [9], [12], [29], [31] whereby a high-level planner feeds velocity commands to a fixed locomotion policy with prescribed gait, and body orientation. Such decoupling fails to produce an overall policy specialized for manipulation; the locomotion-centric priors bias contact scheduling and base motion toward an optimal behavior for locomotion, not manipulation, thereby limiting efficiency and adaptability.

To address the aforementioned limitations, we finetune a Qwen2.5-1.5B-Instruct model [33] using LoRA [34]. Inspired by recent work on robot learning [18], [26], [35], [36], we finetune our planner on synthetic data. The proposed data generation pipeline enables rapid collection and curation of diverse scene configurations and demonstrations without requiring physical teleoperation [19], or continuous instantiation of a larger model to generate the trajectories [18], [26], [37]. Given a natural-language task description and a structured geometric scene representation, the planner generates a variablelength sequence of parameterized navigation and manipulation actions, including the selected skill, object when applicable, and the associated geometric target for the selected skill. Additionally, unlike hierarchical whole-body planar manipulation frameworks [9], [12], [29], we devise an end-to-end manipulation policy that discovers an efficient planar pushing behavior. The resulting policy improves manipulation success while reducing task completion time by more than 60% compared to the conventional hierarchical method. We evaluate the complete framework across simulated and real-world locomanipulation tasks, including held-out in-distribution trajectories and a shifted task structure excluded from training that requires learned navigation and manipulation behaviors to be composed into a new multi-step strategy. Figure 1 illustrates the deployment of the proposed framework on the quadruped platform.

The main contributions of this work are summarized as follows:

• A language-conditioned task planner for interactive navigation, finetuned with LoRA on synthetic demonstrations, that predicts the remaining sequence of parameterized actions by jointly selecting the skill, the relevant object, and the geometric target. The planner learns the geometric relations underlying interactive navigation without handcrafted rules or large prompts. Comparisons with models trained from scratch indicate that skill composition does not require pretraining when the representation is structured, whereas pretraining yields more reliable geometric targets.

• An end-to-end whole-body planar manipulation policy that enables specializing the robot’s behavior to planar object pushing. The resulting policy improves manipulation success and reduces task-completion time by more than 60% relative to the common hierarchical manipulation baseline.

• Repeated hardware evaluation of the complete framework, coupling the language-conditioned planner with learned navigation and manipulation skills across diverse task structures, including the task structure excluded from training.

## II. PRELIMINARIES

For the planner, we assume access to robot and object pose observations, and 2D object geometry, thus focusing on the planning capabilities of the model. These observations can be obtained from a motion-capture system, an external vision system, or an onboard perception and mapping pipeline. In our hardware experiments, these quantities are provided by a motion-capture system. The manipulation policy has no prior knowledge of the objects’ dynamics or geometry. We define a loco-manipulation policy as one that is trained to both locomote to an object and manipulate it, while a manipulation policy is one that can only manipulate.

![](images/723ff1eee599caad96e81ffbf249910de6cdc7e7e3ec867d57e1545ff899c4c9.jpg)  
Fig. 1: Deployment pipeline of the proposed framework on the quadruped robot. The planner receives observation $o _ { t } ^ { \prime }$ consisting of a task instruction specified by the user, scene states such as robot and object pose, along with object parameters, and the previous skill that was applied on the environment. The planner, trained on predicting the remaining action sequence to achieve the task objective, applies only the first action in the sequence, and is queried again once that action is complete. Thus, our task planner is an event-driven task planner. Blue robot instances signify that a manipulation policy was used in that stage, whereas Red signifies a navigation policy. Dashed lines represent data flow, whereas solid lines represent decision flow.

We consider two goal-conditioned motor skills for our lowlevel skill library: navigation $( s = n )$ and planar manipulation $( s \ : = \ : m )$ . Each skill is modeled as a finite-horizon Markov Decision Process (MDP) $\mathcal { M } _ { s } = ( \mathcal { X } _ { s } , \mathcal { A } _ { s } , \mathcal { G } _ { s } , p _ { s } , r _ { s } , \gamma )$ ; where $\mathcal { X } _ { s }$ and $A _ { s }$ denote the state and action spaces, $p _ { s } ( x _ { t + 1 }$ $x _ { t } , a _ { t } )$ is the transition dynamics, $r _ { s } ( x _ { t } , a _ { t } , g _ { s } )$ is the goalconditioned reward with $g _ { s } \in \mathcal { G } _ { s } ,$ , and $\gamma \in ( 0 , 1 )$ is the discount factor. A goal-conditioned policy parameterized by $\theta _ { s } ,$ denoted by $\pi _ { s } ( a _ { t } \mid x _ { t } , g _ { s } )$ for simplicity, induces trajectories $\tau ~ = ~ ( x _ { 0 } , a _ { 0 } , \ldots , x _ { T } )$ under the dynamics $p _ { s }$ . The optimal policy maximizes the expected discounted return:

$$
\pi _ { s } ^ { * } \in \arg \operatorname* { m a x } _ { \pi _ { s } } \mathbb { E } _ { \tau \sim p _ { s } , \pi _ { s } } \left[ \sum _ { t = 0 } ^ { T } \gamma ^ { t } r _ { s } ( x _ { t } , a _ { t } , g _ { s } ) \right] .\tag{1}
$$

Both low-level policies, $\pi _ { n }$ and $\pi _ { m } ,$ are trained with Proximal Policy Optimization (PPO) [38] in Isaac Lab [39] using domain randomization to enable zero-shot sim-to-real transfer.

At discrete high-level decision epochs k, the languageconditioned planner maps a natural-language task $G _ { \mathrm { N I } }$ <sub>L</sub> and structured scene representation to a variable-length sequence of parameterized actions denoted by $\mathbf { a } ^ { \mathbf { H L } } { } _ { k : k + H _ { k } }$ where,

$$
\begin{array} { r } { \mathbf { a } ^ { \mathbf { H L } } { } _ { k : k + H _ { k } } = \left( a _ { k } ^ { H L } , a _ { k + 1 } ^ { H L } , \ldots , a _ { k + H _ { k } } ^ { H L } \right) , \qquad a _ { j } ^ { H L } = \left( s _ { j } , i _ { j } , g _ { j } \right) } \end{array}\tag{2}
$$

where $H _ { k }$ denotes the predicted remaining skill horizon and $j ~ \in ~ \{ k , \ldots , k + H _ { k } \}$ indexes the predicted skills. Here, $s _ { j } \in \{ n , m \}$ denotes the selected skill, $i _ { j }$ the relevant object identifier when applicable, and $g _ { j }$ its geometric target. Although the planner predicts the complete remaining action sequence $\mathbf { a } ^ { \mathbf { H L } } _ { k : k + H _ { k } } ^ { - }$ , only $a _ { k } ^ { H L }$ is executed before the planner is queried again at the next high-level decision using the updated scene state.

## III. METHODOLOGY

## A. Planar Manipulation Policy

a) Observation and Action space: We use an asymmetric actor-critic architecture [40]. The actor receives observations $\mathbf { o } _ { \mathbf { t } } = ( \mathbf { x } _ { \mathbf { t } } , \mathbf { g } _ { \mathbf { t } } )$ , whereas the critic additionally receives noiseless, privileged information $\mathbf { o _ { t } ^ { \mathrm { p r } } }$ that is unavailable to the robot in deployment allowing it to better estimate the value function under object diversity. The list of observations is documented in Table I. The action is a vector of desired joint-angle offsets $\Delta { \bf q } _ { \mathrm { d e s } } \in \mathbb { R } ^ { 1 2 }$ , tracked by a joint-space impedance controller to generate torques τ.

b) Rewards: To avoid weight-tuning between translation and yaw errors, we define a keypoint set K on the object, similar to [29], [41], [42], to embed both errors into the same space. Let ${ \cal W } _ { \bf p } _ { \cal O } ^ { { ( i ) } } , \ { \dot { W } } _ { \bf p } _ { \cal G } ^ { { ( i ) } }$ be the world positions of keypoint i on the current object pose and the goal pose, respectively. The primary task reward is

$$
r e w _ { \mathrm { k p } } = \exp \left( - \sum _ { i \in \mathcal { K } } \ell _ { \mathrm { k p } } ^ { i } / \sigma _ { \mathrm { k p } } ^ { 2 } \right)\tag{3}
$$

TABLE I: Observations for the actor $( o _ { m } )$ and critic $( o _ { m } , o _ { m } ^ { \mathrm { p r } } )$ for the manipulation policy.
<table><tr><td>Quantity</td><td>Description</td><td>Dim</td><td>Noise</td></tr><tr><td colspan="4"></td></tr><tr><td> ${ \mathit { R } } _ { \mathbf { p } _ { R G } }$ </td><td>Goal-Robot relative position w.r.t. Robot</td><td>3</td><td></td></tr><tr><td> $\psi _ { R G }$ </td><td>Goal-Robot relative heading</td><td>1</td><td></td></tr><tr><td> $\Delta \mathbf { q } _ { j }$ </td><td>Joint position offsets from default</td><td>12</td><td> $\mathcal { U } ( - 0 . 0 1 , 0 . 0 1 )$ </td></tr><tr><td> $\Delta \dot { \mathbf q } _ { j }$ </td><td>Joint velocities from default</td><td>12</td><td> $\mathcal { U } ( - 1 . 0 , 1 . 0 )$ </td></tr><tr><td> ${ \bf a } t - 1$ </td><td>Previous actions</td><td>12</td><td></td></tr><tr><td> $\omega _ { R }$ </td><td>Robot base angular velocity</td><td>3</td><td> $\mathcal { U } ( - 0 . 1 0 , 0 . 1 0 )$ </td></tr><tr><td> $\| \mathbf { p } _ { O G } \|$ </td><td>Goal-Object distance</td><td>1</td><td> $\mathcal { U } ( - 0 . 0 5 , 0 . 0 5 )$ </td></tr><tr><td> $\psi _ { G O }$ </td><td>Object-Goal relative heading</td><td>1</td><td> $\mathcal { U } ( - 0 . 0 2 , 0 . 0 2 )$ </td></tr><tr><td> ${ \bf \Pi } ^ { R } { \bf { p } } _ { R O }$ </td><td>Object-Robot relative position w.r.t Robot</td><td>3</td><td> $\mathcal { U } ( - 0 . 0 5 , 0 . 0 5 )$ </td></tr><tr><td> $\psi _ { R O }$ </td><td>Object-Robot relative heading</td><td>1</td><td> $\mathcal { U } ( - 0 . 0 2 , 0 . 0 2 )$ </td></tr><tr><td> ${ { \bf { \mathit { \mathbf { \mathit { R } } } } } _ { { \bf { \bar { q } } } _ { R O } } }$ </td><td>Object-Robot relative orientation (quaternion) w.r.t Robot</td><td>4</td><td> $\mathcal { U } ( - 0 . 0 5 , 0 . 0 5 )$ </td></tr><tr><td colspan="4"> $o _ { t } ^ { \mathrm { p r } }$ </td></tr><tr><td> $\lambda$ </td><td>Robot-object contact state</td><td>1</td><td></td></tr><tr><td> $\mathbf { v } _ { R }$ </td><td>Robot base linear velocity</td><td>3</td><td></td></tr><tr><td> $\mathbf { g _ { R } }$ </td><td>Projected Gravity on the Robot frame</td><td>3</td><td></td></tr><tr><td> ${ } ^ { R } { } _ { \mathbf { p } { C O M } }$ </td><td>Object CoM position w.r.t. Robot</td><td>3</td><td></td></tr><tr><td> $m$ </td><td>Object mass</td><td>1</td><td></td></tr><tr><td> $\mathbf { d }$ </td><td>Object bounding-box dimensions</td><td>3</td><td></td></tr><tr><td> $\mathbf { I } _ { O }$ </td><td>Object principal moments of inertia</td><td>3</td><td></td></tr><tr><td> $\mathbf { v } _ { O }$ </td><td>Object linear velocity</td><td>3</td><td></td></tr><tr><td> $\omega _ { O }$ </td><td>Object angular velocity</td><td>3</td><td></td></tr><tr><td> $\kappa _ { T }$ </td><td>Object type</td><td>1</td><td></td></tr><tr><td> $\pmb { \mu } _ { s d }$ </td><td>Object dynamic and static friction coefficients</td><td>2</td><td></td></tr></table>

where

$$
\ell _ { \mathbf { k } \mathbf { p } } ^ { i } = \left\| ^ { W } \mathbf { p } _ { O } ^ { ( i ) } - { W } \mathbf { p } _ { G } ^ { ( i ) } \right\| ^ { 2 }\tag{4}
$$

To guide early learning over long distances, we formulate a velocity command $\mathbf { v } _ { c m d } ,$ correlated to the target vector ${ } ^ { R } \mathbf { p } _ { O G } ,$ and reward for small error between object velocity $\mathbf { v } _ { O }$ and $\mathbf { v } _ { c m d } .$ Previous planar manipulation work [29], [43] reward for object motion inline with the target vector. Although our reward adds a tuning parameter $K .$ , we empirically find that it offers better guidance for the agent, and emerges with a behavior that gradually decreases object velocity throughout the task. The velocity command is formulated as follows:

$$
r e w _ { v } = \exp \left( - \frac { | | \mathbf { v } _ { \mathrm { c m d } } - \mathbf { v } _ { O } | | ^ { 2 } } { \sigma ^ { 2 } } \right)\tag{5}
$$

where

$$
{ \bf v } _ { \mathrm { c m d } } = { } \hat { { \bf p } } _ { O G } \ \operatorname* { m i n } ( K \| { } ^ { R } { \bf p } _ { O G } \| , \ 0 . 7 5 )\tag{6}
$$

We find that including a reward for positive robot linear velocity along its x-axis , i.e. $v _ { R , x } > 0 \mathrm { { ; } }$ , improves the stability of the robot. To obtain a smooth behavior, we penalize joint acceleration ${ \ddot { \mathbf { q } } } ,$ joint torques $\tau ,$ action rate $\Delta \mathbf { a _ { t - 1 , t } } ,$ and object roll/pitch velocities $\omega _ { O , x y } .$ . We also find that penalizing feet acceleration helps reduce impact forces with the ground, decreasing the energy expenditure throughout the task. The list of rewards and their weights is provided in Appendix B-C.

c) Training Details: We train the policy on cubes and cylinders, with randomized physical and geometric parameters, as detailed in Table II. At the beginning of each episode, we randomize robot yaw and goal pose to enhance generalization. Object pose is randomized; however, we ensure small initial distance between the robot and the object. To further promote robustness, external disturbances are applied to the robot base at randomized intervals during training, implemented as base twist perturbations. Episode termination occurs under any of the following conditions: (i) object flip over (i.e. the roll/pitch exceed a predefined threshold), (ii) robot flip over, or (iii) Episode Timeout (not failure).

TABLE II: Training parameters in simulation.
<table><tr><td>Parameters</td><td>Training Range</td></tr><tr><td>Mass (kg)</td><td>U(1, 6)</td></tr><tr><td>∆ CoM position (cm)</td><td>U(−10.0, 10.0)</td></tr><tr><td>Static Friction</td><td>U(0.5, 1.2)</td></tr><tr><td>Dynamic Friction</td><td>U(0.6, 1.0)</td></tr><tr><td>Object type</td><td>[Box, Cylinder]</td></tr><tr><td>Object diameter (m)</td><td>U(0.3, 0.6)</td></tr><tr><td>Object width (m)</td><td>U(0.3, 0.6)</td></tr><tr><td>Object depth (m)</td><td>U(0.3, 0.6)</td></tr><tr><td>Object height (m)</td><td>U(0.3, 0.6)</td></tr></table>

## B. Navigation Policy

We train a hierarchical navigation policy $\pi _ { n }$ to track planar pose commands $\mathbf { c _ { t } } ~ \in ~ \mathbb { R } ^ { 3 }$ (x, y, yaw) expressed in the base frame of the robot, while avoiding obstacles. We adopt a navigation policy rather than a pure point-to-point locomotion policy, as this removes the need for the LLM to generate intermediate locomotion waypoints, thereby reducing output sequence length and simplifying action generation. The lowlevel policy is trained to track velocity commands $( v _ { x } , v _ { y } , \psi )$ and outputs desired joint angles w.r.t the default configuration, $\Delta q _ { \mathrm { d e s i r e d } }$ . For additional details on the navigation policy, please refer to the Appendix B-B. The main task rewards for the navigation planner are expressed as follows:

$$
r e w _ { t } = \exp \left( - \frac { \| { \bf \nabla } ^ { W } { \bf p } _ { G } - { \bf \nabla } ^ { W } { \bf p } _ { R } \| ^ { 2 } } { \sigma _ { t } ^ { 2 } } \right)
$$

$$
r e w _ { r } = \exp \left( - \frac { \| \psi _ { \mathrm { G } } - \psi _ { R } \| ^ { 2 } } { \sigma _ { r } ^ { 2 } } \right)\tag{7}
$$

(8)

## C. High-level LLM Task Planner

a) Data Generation and Augmentation Strategy: To systematically structure the training distribution, we introduce a taxonomy of task cases that partitions the dataset into mutually exclusive categories, ensuring systematic coverage of the problem space. These cases correspond directly to the task families summarized in Table XII. For robot-centric objectives, we define the following cases:

• Case 1: No objects in the scene.

• Case 2: One or more objects are present but do not obstruct the robot’s path to the goal.

• Case 3: Two or more objects are arranged such that they obstruct direct navigation to the goal.

Cases 1 and 2 correspond to the Direct Navigation task family in Table XII, and can be completed through direct navigation. We additionally define an object-centric case, denoted Case 2.5, in which the robot is randomly positioned away from a single target object and must first navigate toward it before manipulation. The goal of this task is to facilitate the learning of the underlying manipulation skill. This corresponds to the Object Manipulation task family. Case

3 provides the scene structure underlying the interactive task families, where successful completion requires manipulation and navigation to be combined. Depending on the initial robotobject configuration, the training data contains Interactive Navigation trajectories with composition $M  N$ and Long-Horizon Interactive Navigation trajectories with composition $N \ \to \ M \ \to \ N$ . All data is generated in a custom 2D simulation environment implemented in Pygame, where the scene is a room of dimensions $3 \times 5 ~ [ \mathrm { m } ^ { 2 } ]$ emulating the space in the lab. Scenes are procedurally constructed and evaluated at discrete state transitions, where each action deterministically updates the environment state. We record scene states at each discrete decision step (i.e., skill-level transitions). This yields an event-driven trajectory representation

$$
\mathcal { D } _ { i } = \{ ( z _ { i , 0 } , a _ { i , 0 } ^ { H L } ) , \dots , ( z _ { i , k } , a _ { i , k } ^ { H L } ) , \dots , ( z _ { i , N _ { k } } , a _ { i , N _ { k } } ^ { H L } ) \} ,
$$

where i denotes the trajectory index and $N _ { k }$ is the number of skill transitions in trajectory i. Here $z _ { i , k }$ represents the symbolic scene states and parameters at decision step k, and $a _ { i , k } ^ { H L }$ denotes the selected high-level action. Each scene encodes the robot pose, object poses, object geometry, a binary flag indicating object mobility, and a unique identifier assigned to each object. The number of manually-generated trajectories is dependent on the difficulty of the introduced cases. For instance, cases that require skill switching require more emphasis on manually generated data to ensure adequate coverage of the state-action pairs. Once the manual generation phase is done, we augment the trajectories using a set of simple rules to automate the enrichment of the dataset. For example, we mirror the scene and goal w.r.t each axis, we randomize robot, goal and object pose (and identifier), we insert irrelevant objects, etc.

This multi-stage data generation combines manual structural design with controlled stochastic augmentation, enabling scalable dataset expansion while preserving the semantic integrity of each task case. After augmentation, the resulting dataset contains approximately 8K trajectories, corresponding to around 16K training samples. For additional information on data generation, and augmentation, we refer the reader to Appendix E-A.

b) Model Finetuning: We fine-tune the Qwen2.5-1.5B-Instruct model, a decoder-only causal transformer architecture, using Low-Rank Adaptation (LoRA). We fine-tune the model using standard cross-entropy loss for next-token prediction in a causal language modeling setting. Given an input sequence consisting of (i) the task instruction , (ii) the symbolic scene states and parameters $z _ { k , j }$ , including robot pose and object attributes, (iii) the skill executed at the previous step, and (iv) a Done flag indicating trajectory termination, the model autoregressively predicts a structured action output. Rather than predicting only the immediate next high-level action, we train the model to predict a sequence of high-level actions within the trajectory, commonly referred to as action chunking (AC) [44], applied in a receding-horizon manner. In our case, the target is the remaining action sequence to reach the goal. Formally, we model

$$
p \left( a _ { i , k : k + H } \mid z _ { i , k } \right)
$$

![](images/7646a4d0173b5e80badcdee3720e0001dd774c76f603359896255513f22bf666.jpg)  
(a)

![](images/bda33b674e9ed2ef815c7b3df2a0cd82c0e9736bc0471ab56ab8a31442e4a698.jpg)  
(b)

Fig. 2: Emergent Gait Cycle for (a) HRL, and (b) E2E Manipulation Policies. Results were obtained from simulation.  
![](images/cc90424427c7e5f91fd77d97cc992fe486886fc54fe4971e0141934271d77162.jpg)  
(a)

![](images/a126f33f9449c8db82f6e7d2fc47c13e7c467abc4061cd5afca85e96d474e24e.jpg)  
(b)  
Fig. 3: Comparison of body orientation, (a) Roll, and (b) Pitch for a run where the robot is distant from the object. Blue and orange dashed lines denote the first instant of contact with the object for E2E and HRL, respectively. The manipulate object is a box with an edge length of 40 [cm].

where H denotes the remaining skill horizon in our case. LoRA is applied to all linear layers. The LoRA rank $( r = 1 2 8 )$ is selected on a validation set of shifted-task scenes that is disjoint from the test scenes reported in Section IV-C2. We use a learning rate of $\eta = 2 \times 1 0 ^ { - 5 }$ . Additional information on finetuning and our design choices are detailed in Appendix E-B.

## IV. EXPERIMENTS AND RESULTS

We evaluate the proposed framework at both levels of the hierarchy. Low-level manipulation is evaluated in physics simulation and through repeated hardware trials to assess task performance and sim-to-real transfer. High-level planning is evaluated separately in the procedural geometric environment described in III-C. Finally, repeated hardware experiments evaluate the complete hierarchy by coupling the pretrained planner with the learned navigation and manipulation policies. We use the Unitree Go2 robot in our experiments.

In this section, all end-to-end (E2E) policies run at 50 Hz while the impedance controller at the joints runs at 200 Hz. The planner of the hierarchical (HRL) policies runs at 10 Hz, while its low-level policy and joint controller run at 50 Hz and 200 Hz, respectively. All real-world experiments reported in this work are inferred on an RTX 5080 GPU, where LLM inference runs at approximately 1 Hz. Additional information on training is provided in Appendix B-A.

## A. Preliminary Experiments

We begin by demonstrating a set of experiments comparing HRL and E2E loco-manipulation policies, which motivates the design of our low-level skill library. To properly demonstrate the differences between these two policies, and ensure a fair comparison, we train both loco-manipulation policies from scratch under the same environment configuration. To train the HRL loco-manipulation policy, we follow the pipeline of [29], excluding the adaptation module, to isolate the influence of the control architecture. The E2E policy, also without the adaptation module, is composed of a single MLP. To train that policy, we first reward the robot for reaching the object and activate the manipulation reward after L training steps. For additional details on the training of these policies, we refer the reader to Appendix C. Figure 2 shows the emergent gait cycle for HRL and E2E loco-manipulation policies, respectively. Although the difference in training scheme may seem minimal, the HRL policy is intrinsically limited by the locomotion priors of the velocity-tracking policy. Thus, the planner of the HRL policy, the only module that has access to object pose, is optimized under a constrained setting. Instead, we see that the E2E policy learns an asynchronous gait cycle, and exhibits more base oscillations as seen in figure 3, since it is free to optimize its entire behavior in order to maximize the expected rewards. The emergent gait cycle can be connected to the emergent body-orientation of the robot. The robot pitches its body, shifting its weight to the front. This shift in configuration makes it harder to lift the hind feet without losing stability. The emergent behavior allows the robot to position more of its inertia into the object allowing for more efficient transfer of momentum, and the lower point of contact aligns better with the CoM of the object. One caveat of the E2E policy is that it consumes more power. The mean power consumption for the E2E policy is 127.23 [W] compared to 66.03 [W] for the HRL policy.

While the emergent behavior of the E2E policy is effective for object-centric tasks, it is unnecessary for locomotion: increased base oscillations raise the probability of falling over, while the overall behavior results in higher energy consumption. We provide additional discussions and experiments on energy efficiency in Appendix C.

In summary, these results motivate our decomposition of planar loco-manipulation into specialized navigation and manipulation policies, where each segment can be independently solved by its own specialized policy instead of optimizing for both objectives in a single policy, which can lead to suboptimal results [45], [46]. Additionally, as we will see in Sec. IV-B, since the E2E policy is not constrained by any priors, and exhibits a broader range of adaptation towards diverse objects.

## B. Low-Level Manipulation Evaluation

In this section, we present the results of our low-level manipulation policy in simulation and on hardware. Each baseline follows its corresponding training procedure described in IV-A, with the exception of initializing the robot near the object for the HRL manipulation policy. All policies in this section, including ours, share the same observations, object distribution, and domain randomization (Table II). We tune the reward weights of each policy to ensure a fair comparison.

Additionally, we ablate two reward terms from our proposed setting in order to better understand their effect on planar manipulation and provide the results in Appendix D.

1) Simulation Comparison: The proposed policy is compared against the following baselines: (1) E2E policy trained for loco-manipulation, and (2) HRL policy trained for manipulation. The simulation comparison isolates two design choices. Comparing our policy with E2E loco-manipulation evaluates the benefit of specialization for manipulation, whereas comparison with the HRL manipulation policy evaluates the effect of optimizing planar manipulation end-to-end rather than through a fixed locomotion policy.

We generate 1000 random environments following the training randomization and evaluate the aforementioned baselines. The average initial distance between the object and the goal is $2 . 5 \ \pm \ 1$ m. We consider a run to be successful, if the pose error is less than 15 cm and $1 5 ^ { \circ }$ . The episode length is set to 15 s. We report the following quantities: Success Rate (SR), average Episode Time (ET), Fall Over (FO) rate, and Energy Consumption (EC) for successful runs only expressed as $\begin{array} { r } { \sum _ { t } \sum _ { j } | \tau _ { j } \dot { q } _ { j } | \Delta t } \end{array}$ . Comparison results are presented in Table III.

We find that an E2E loco-manipulation policy shows similar success rate to that of the HRL manipulation policy, however with significantly smaller (ET) due to the unconstrained behavior discussed in IV-A. Although this formulation still exceeds the HRL solution in efficiency, the conflicting objectives during training do not enable the model to be optimized properly for manipulation, consequently the model achieves a lower success rate than our method. We highlight the difference in efficiency between the HRL policy and ours. Higher success rate, coupled with lower mean and standard deviation of (ET) indicates that our method enables the robot to effectively adapt to different object size and weight, resulting in a more robust manipulation behavior. Nonetheless, the HRL policy emerges with the lowest (FO) rate, only slightly lower than our method, mainly due to the constrained body orientation and gait.

In summary, the two design choices are complementary. Combining manipulation-specific training with an end-to-end whole-body architecture yields the highest success rate while retaining low task completion time.

2) Hardware Experiments: We conduct repeated hardware trials, for both the HRL and the proposed E2E manipulation policies, on three objects with different geometries and masses: (i) A cardboard box of edge length 40 [cm] and weight of approx. 2 [kg], (ii) A plastic cylinder of height 70 [cm] and diameter of 55 [cm] containing a backpack of 4 [kg] at the bottom, and (iii) a plastic rectangular prism containing the box of the Go2 with the battery and charger, weighing approx 8 [kg]. The average initial object-goal distance is approximately 2.0[m], 1.4[m], and 1.25[m]; whereas the initial object-goal yaw error is 90<sup>◦</sup>, 45<sup>◦</sup>, and $0 ^ { \circ }$ for the box, cylinder and prism, respectively. The results are presented in Table IV. We provide a video to qualitatively demonstrate the behavioral difference between the runs for both controllers for all three objects in Fig. 4.

The cardboard box is within the training distribution seen in training, whereas the plastic cylinder is outside of the distribution in terms of object height, while the rectangular prism is outside of the distribution in terms of object weight.

Across all three objects, the proposed policy achieves lower mean pose error and shorter completion time than the HRL baseline. The reduction in completion time is consistent across trials for every object, whereas for the box, which lies within the training distribution, the error distributions of both policies overlap. The gap is most pronounced for the rectangular prism, which represents a shift in object mass relative to the training distribution: position error decreases from 19.44 to 10.27 [cm] and completion time from 11.18 to 2.52 [s]. These results are consistent with the simulation evaluation and indicate that the proposed policy transfers to hardware.

TABLE III: Comparison with Baselines. Compared to the HRL baseline, our method improves Success Rate and Episode Time. Compared to the E2E loco-manipulation baseline, our policy improves success rate while retaining low Episode Time.
<table><tr><td>Method</td><td>Success Rate (%)</td><td>Episode Time (s)</td><td>Fall Over (%)</td><td>Energy (J)</td></tr><tr><td>E2E loco-manipulation</td><td>84.50</td><td> $4 . 1 5 \pm 1 . 4 6$ </td><td>3.5</td><td> $1 7 3 7 . 5 5 \pm 7 0 4 . 0 1$ </td></tr><tr><td>HRL manipulation</td><td>83.50</td><td> $1 0 . 8 0 \pm 4 . 0 2$ </td><td>0.3</td><td> $1 8 1 3 . 6 1 \pm 1 0 6 0 . 3 1$ </td></tr><tr><td>Ours</td><td>93.30</td><td> ${ \bf 4 . 0 9 \pm 1 . 3 1 }$ </td><td>0.8</td><td> $\mathbf { 1 6 8 0 . 8 0 \pm 6 7 8 . 1 2 }$ </td></tr></table>

![](images/b9c993ada2d3c5f1cb93410b37b4e05034c16fed059c1a455262b3dc0bd7f4b1.jpg)  
Fig. 4: Hardware comparison of the manipulation policies.

TABLE IV: Hardware evaluation of the manipulation policies. Values are reported as mean ± standard deviation over five trials per object.
<table><tr><td>Object</td><td></td><td>Controller Position Err. [cm] Yaw Err. [deg]</td><td></td><td>Time [s]</td></tr><tr><td rowspan="2">Box</td><td>HRL</td><td> $7 . 5 3 \pm 3 . 1 2$ </td><td> $1 4 . 2 0 \pm 7 . 4 7$ </td><td> $4 . 5 7 \pm 0 . 3 9$ </td></tr><tr><td>Ours</td><td> ${ \bf 5 . 3 4 \pm 1 . 5 3 }$ </td><td> ${ \bf 1 3 . 1 2 \pm 5 . 1 0 }$ </td><td> ${ \bf 3 . 0 8 \pm 0 . 1 8 }$ </td></tr><tr><td rowspan="2">Cylinder</td><td>HRL</td><td> $1 3 . 4 3 \pm 5 . 2 2$ </td><td> $1 9 . 1 8 \pm 5 . 9 2$ </td><td> $3 . 0 6 \pm 0 . 1 3$ </td></tr><tr><td>Ours</td><td> ${ \bf 1 0 . 3 7 \pm 2 . 7 3 }$ </td><td> ${ \bf 4 . 9 9 } \pm 2 . 2 { \bf 0 }$ </td><td> ${ \bf 1 . 8 8 \pm 0 . 1 4 }$ </td></tr><tr><td rowspan="2">Prism</td><td>HRL</td><td> $1 9 . 4 4 \pm 9 . 6 1$ </td><td> $1 2 . 6 4 \pm 5 . 8 7$ </td><td> $1 1 . 1 8 \pm 2 . 0 9$ </td></tr><tr><td>Ours</td><td> ${ \bf 1 0 . 2 7 \pm 2 . 6 8 }$ </td><td> ${ \bf 5 . 7 6 \pm 2 . 3 1 }$ </td><td> $\mathbf { \ } 2 . 5 2 \pm \mathbf { \ 0 . 4 4 }$ </td></tr></table>

## C. Language-Conditioned Task Planner Evaluation

We evaluate the high-level planner in the procedural geometric planning environment described in III-C, and subsequently evaluate the complete planning and control hierarchy on hardware. The geometric benchmark isolates semantic and geometric decision-making by providing each baseline with the same structured scene representation, while the hardware experiments assess whether the generated skill plans remain executable when coupled with the learned motor skills.

Our planner evaluation aims to address the following questions:

• (Q1) How does the finetuned planner compare with a task-specific classical planner and in-context learning baselines?

• (Q2) Can the planner compose a novel skill sequence for a shifted task structure?

• (Q3) What role does the pretrained backbone play in skill composition and geometric target prediction?

• (Q4) Does the resulting planner, trained on synthetic data, transfer to real-world loco-manipulation tasks?

We additionally analyze how LoRA rank and action chunking affect both the ID and shifted task families in Appendix F-B.

1) Evaluation Protocol and Baselines:

a) Task families: We evaluate the planner across task families that differ in both the required interaction with the environment and the number of high-level skills needed for task completion. We denote navigation and manipulation skills by (N) and (M), respectively. The in-distribution (ID) evaluation contains task structures represented in the training distribution. In addition, we evaluate the planner on a shifted task structure whose skill composition is not represented during training. The task families and their skill compositions are summarized in Table XII.

b) Baselines: We compare against a classical planner, two in-context learning (ICL) baselines, one using the same pretrained Qwen backbone and one using a frontier model (GPT-5.5), a task-specific Transformer, and a Qwen2.5- architecture model trained from scratch on the same text-based inputs and outputs.

Classical planner: We implement a task-specific classical planner based on a state machine, a visibility graph, Dijkstra’s algorithm for geometric path planning, and a sampling-based heuristic for identifying the blocking object to reposition and the relocation target. The classical planner receives the same structured geometric scene information as the LLM planner. Its parameters are tuned jointly across all task families, and a single parameter set is used for all scenes. In addition to the branches covering the training task families, we design a dedicated branch encoding the $\mathrm { ~ M ~ } \to \mathrm { ~ N ~ } \to \mathrm { ~ M ~ }$ strategy of the shifted task. The classical planner therefore has explicit knowledge of a task structure that is excluded from our planner’s training data. The choice of this baseline is motivated by the ability of explicitly designing rules and conditions that the plan must meet in order to complete the required task.

ICL baselines: We compare against two ICL baselines; one using the same pretrained Qwen backbone without any gradient updates, and a frontier model (GPT-5.5). The models receive a fixed prompt describing the available skills, the workspace bounds, the proximity condition for manipulation, and the admissible plan structures, together with one demonstration for each in-distribution task family and for the shifted task family. We additionally include a counterexample for direct navigation in a scene where a free path to the goal is absent. At inference time, the ICL baselines receive the same natural-language task and structured scene states and parameters as the finetuned planner, and generate the complete skill sequence from the initial state, which is then executed open-loop. These baselines evaluate whether the required semantic and geometric planning behavior can be elicited from the same small pretrained Qwen model and a substantially larger model through prompting alone. For the evaluation of GPT-5.5 on the shifted task, the description and demonstration of the shifted structure are removed from the prompt, so that, like our planner, it is not informed of it. As for the Qwen ICL baseline, since it fails on the shifted task even with this information, it is reported with the full prompt.

Task-specific Transformer: To assess whether the task structures in our formulation can be learned without pretrained representations, we train a decoder-only Transformer from scratch on the same synthetic dataset. The task objective, robot state, previous skill, and individual scene objects are embedded separately. The model autoregressively predicts the skill, the relevant object through a pointer over the contextualized object representations, and the geometric target $( x , y , \psi )$ , which is quantized into bins of 1 [cm] and 1 [<sup>◦</sup>] and trained with a cross-entropy loss. As with our planner, the model is trained to predict the remaining action sequence, and only the first action is executed before re-planning. Architecture and training details are provided in Appendix F-C.

Additional information on the baselines is provided in Appendix F.

c) Plan-success criterion: Many scene configurations admit multiple valid solutions, thus we evaluate the generated plan according to task-level feasibility. A plan is considered successful if its execution in the geometric environment completes the requested task without violating the planner’s operational constraints. Specifically, (i) navigation and manipulation must remain collision-free with respect to unintended objects and within the workspace boundaries; (ii) manipulation may only be applied to movable objects and only when the robot satisfies a required proximity condition represented by a center-to-center distance of at most 1.2 [m] ; and (iii) the resulting sequence of parametrized actions must achieve the task objective. Any violation of these constraints is counted as a planning failure.

2) Simulation Comparison: We evaluate our planner and the baselines on 200 held-out trajectories spanning the indistribution task families described in IV-C1, and 30 trajectories from the shifted task structure. The results are presented in Table V.

## (Q1) How does the finetuned planner compare with a task-specific classical planner and in-context learning baselines?

Our planner shows its advantage once successful task completion of the ID task families requires composition of navigation and manipulation skills. Our planner solves 109/120 interactive navigation trajectories, compared with 86/120 for the classical planner. A similar trend is observed for longhorizon interactive navigation, where our planner solves 29/30 trajectories, compared with 22/30 for the classical planner.

Of the eleven failures of our planner on interactivenavigation tasks, six correspond to incorrectly issuing a directnavigation skill despite the path being obstructed. Two other failures result from an incorrect geometric target and an incorrect object identifier, respectively. In the remaining three failures, the planner generates a task plan with composition $\mathrm { ~ N ~ } \to \mathrm { ~ M ~ } \to \mathrm { ~ N ~ } .$ The composition is not inherently faulty, in fact, the planner attempted to align the robot with the object to manipulate. Nevertheless, the manipulation target fails to create a free path, preventing successful completion of the task. The classical planner remains a competitive baseline, but its performance is sensitive to the geometric heuristics used to determine goal reachability and object relocation. Of its 34 interactive-navigation failures, six occur because no suitable relocation target is found, and the remainder because the visibility graph returns a path that is not physically feasible. More conservative obstacle inflation reduces such failures but triggers unnecessary manipulation in direct-navigation scenes, illustrating the trade-off of a single fixed parameterization.

The Qwen ICL baseline outputs a single action mirroring the task instruction in every scene, and therefore solves none of the task families that require combining navigation and manipulation. GPT-5.5 narrows this gap, but our planner matches or exceeds it on all ID task families, most clearly on interactive navigation (109/120 vs. 94/120) and long-horizon interactive navigation (29/30 vs. 18/30), where its failures are predominantly geometric, at roughly 15× the per-action latency (Appendix F).

In summary, the finetuned planner learns the geometric relations underlying interactive navigation without handcrafted rules or large prompts, and exceeds both the classical planner and the prompted frontier model on interactive navigation tasks, solving 109/120 interactive-navigation and 29/30 longhorizon trajectories, consistently across training runs (Appendix F-E).

## (Q2) Can the planner compose a novel skill sequence for a shifted task structure?

We evaluate the planner on the shifted interactive manipulation task, whose task structure (M → N → M) is excluded from the training distribution. The training distribution contains obstacle-clearing behavior through interactive-navigation tasks and object-centric manipulation in scenes where only the target object and the robot are present. The interactive manipulation task requires these behaviors to be recombined into a new sequence: the planner must first manipulate a blocking object, navigate to the specified target object, and subsequently manipulate it to the goal. Our planner completes 12/30 shifted-task trajectories, compared with 20/30 for the classical planner, which solves this task through a branch designed specifically for this structure, and 2/30 for GPT-5.5, whose prompt, like our training data, neither describes nor demonstrates this structure. When the structure is described and demonstrated in its prompt, GPT-5.5 completes 22/30 with the correct composition in all 30 trajectories (Appendix F).

Our planner generates the intended skill composition in 24/30 trajectories (16/30 and 20/30 for the two additional training runs; Appendix F-E). The six failures result from reverting to the familiar object-manipulation sequence seen during training. In the remaining 12 failures, the correct skill sequence is recovered, but the navigation target following manipulation of the blocker does not place the robot sufficiently close to the target object to satisfy the proximity constraint. This behavior is particularly apparent when a large number of objects is present in scene.

TABLE V: Planner baseline results, represented as (# of successes)/(# of trajectories) for each task family. Results of our planner and the task-specific Transformer are from a single training run; results of additional runs with different training seed are reported in Appendix F-E.
<table><tr><td>Task Family</td><td>Ours</td><td>ICL (Qwen2.5-1.5B)</td><td>ICL (GPT-5.5)</td><td>Transformer (scratch)</td><td>Classical planner</td></tr><tr><td>Direct navigation</td><td>38/40</td><td>40/40</td><td>36/40</td><td>35/40</td><td>40/40</td></tr><tr><td>Object manipulation</td><td>10/10</td><td>0/10</td><td>10/10</td><td>3/10</td><td>10/10</td></tr><tr><td>Interactive navigation</td><td>109/120</td><td>0/120</td><td>94/120</td><td>101/120</td><td>86/120</td></tr><tr><td>Long-horizon interactive navigation</td><td>29/30</td><td>0/30</td><td>18/30</td><td>22/30</td><td>22/30</td></tr><tr><td>Interactive manipulation (shifted)</td><td>12/30</td><td>0/30</td><td>2/30</td><td>0/30</td><td>20/30†</td></tr><tr><td>Per-action Latency (ms)</td><td>472.4 ± 15.5</td><td>953.5 ± 97.1</td><td>7011 ± 3400</td><td>26.9 ± 6.2</td><td>318.4 ± 598.0*</td></tr></table>

<sup>†</sup>The classical planner uses a dedicated branch designed for the shifted task structure; its skill composition is therefore correct by construction. <sup>‡</sup>The Qwen ICL prompt describes and demonstrates the shifted task structure, which is excluded from the training data of our planner. <sup>§</sup>The model outputs a single action that mirrors the verb of the task instruction in all scenes; success on direct navigation therefore does not reflect planning. <sup>¶</sup>Queried via the OpenAI API; includes network overhead (see Appendix F). <sup>∗</sup>The classical planner exhibits a highly skewed latency distribution (first quartile: 1.2 ms; median: 102.1 ms; third quartile: 276.1 ms). The larger variance arises primarily in interactive tasks, where the planner samples candidate object relocations and repeatedl recomputes geometric paths.

We further probe a longer interactive-manipulation task requiring N → M → N → M, in which the blocker is initially out of reach. For the planner run reported in Table V, only 1/30 plans satisfy all constraints, yet the planner generates the intended skill sequence in 22/30 trajectories while selecting the correct objects to manipulate; failures are dominated by geometric targets that violate reachability, do not match the task target, or fail to create a free path. A detailed failure breakdown is provided in Appendix F-A.

In summary, the learned planner can extend the discrete skill composition beyond the structures represented during training and correctly selects the objects to manipulate, but the geometric targets increasingly violate constraints that are only implicitly represented in the training data.

## (Q3) What role does the pretrained backbone play in skill composition and geometric target prediction?

To separate the contribution of the planning formulation from that of the pretrained backbone, we compare our planner against the task-specific Transformer. On the shifted task, the Transformer produces the intended M → N → M composition in all 30 trajectories but completes none (27 and 30 compositions and at most 2/30 completions across training runs in Appendix F-E). Its failures are dominated by geometric targets: after manipulating the blocker, the predicted navigation target directs the robot toward the final task goal rather than to the target object. Although its navigation target resembles that of the interactive-navigation task seen during training, the Transformer does not reduce the shifted task to interactive navigation since it subsequently emits the final manipulation of the correct target object, completing the M→N→M composition. A similar pattern appears on the ID families, where the Transformer trails our planner on all four, most notably object manipulation (3/10 vs. 10/10), in which it predicts the expected N → M sequence but places the robot too far from the object to manipulate it. A Qwen2.5- architecture model trained from scratch on the same text-based inputs and outputs performs considerably worse, reaching 63/120 on interactive navigation and composing none of the shifted-task trajectories (Appendix F-D).

In summary, these results indicate that skill composition does not require pretraining when the input and output representation is structured, whereas pretraining compensates for a generic text interface and additionally yields more reliable geometric targets from limited synthetic data, at the cost of higher inference latency.

As the models trained from scratch also differ in size from the pretrained backbone, this comparison does not fully isolate the effect of pretraining.

## 3) Hardware experiments: (Q4) Does the resulting planner, trained on synthetic data, transfer to real-world locomanipulation tasks?

All hardware experiments use the planner run reported in Table V.

We evaluate the complete pipeline on hardware on the following task families: (a) 1 direct navigation case; (b) 1 object manipulation case; (c) 3 interactive navigation tasks, of which 2 qualitatively demonstrate the reasoning capability of the model in choosing the correct object to manipulate; (d) 2 long-horizon interactive navigation tasks; (e) 2 shifted interactive manipulation tasks. Each experiment is repeated 5 times, amounting to a total of 45 runs across 9 scenarios. To highlight at which level the failure occurs, we report the following quantities: (i) correct skill composition, and (ii) end-to-end success. Correct skill composition measures the model’s ability in emitting the correct skill sequence. End-toend success additionally requires correct action parameters and their successful completion on hardware. We note that failure in skill composition is propagated to end-to-end failure. The results are presented in Table VI. We additionally provide a qualitative analysis of the plans on the scenes discussed above in Figures 5, 6, and 7.

During repeated hardware experiments, manual replacement of the robot and objects introduces small variations in the initial scene configuration. Across the in-distribution task families, the planner nevertheless produces the correct task composition in 34/35 trials. The only compositional error occurs in IN-1, where the planner selects direct navigation to the goal despite the path between the objects being insufficient for collision-free navigation. In this trial, manual resetting of the scene resulted in a wider gap between the objects than in the nominal configuration; however, the gap remained slightly narrower than the robot’s width.

TABLE VI: Hardware evaluation of the complete loco-manipulation pipeline. Each scenario is repeated five times. Correct composition indicates whether the planner selects the appropriate ordered sequence of skills. End-to-end success require successful completion of the full physical task.
<table><tr><td>Family</td><td>Case</td><td>Evaluation focus</td><td>Expected composition</td><td>Trials</td><td>Correct skill composition</td><td>E2E success</td></tr><tr><td>Direct navigation</td><td>N-1</td><td>Simple Navigation</td><td>N</td><td>5</td><td>5/5</td><td>5/5</td></tr><tr><td>Object manipulation</td><td>OM-1</td><td>Simple Manipulation</td><td>N → M</td><td>5</td><td>5/5</td><td>5/5</td></tr><tr><td rowspan="3">Interactive navigation</td><td>IN-1</td><td>Task composition</td><td>M → N</td><td>555</td><td>4/5</td><td>4/5</td></tr><tr><td>IN-2</td><td>Object selection</td><td>M → N</td><td></td><td>5/5</td><td>5/5</td></tr><tr><td>IN-3</td><td>Object selection</td><td>M →N</td><td></td><td>5/5</td><td>5/5</td></tr><tr><td rowspan="2">Long-horizon interactive navigation</td><td>LH-1</td><td>Long-horizon composition</td><td>N → M → N</td><td>55</td><td>5/5</td><td>3/5</td></tr><tr><td>LH-2</td><td>Long-horizon composition</td><td>N → M → N</td><td></td><td>5/5</td><td>3/5</td></tr><tr><td rowspan="2">Shifted interactive manipulation</td><td>IM-1</td><td>Shifted composition</td><td>M → N → M</td><td>55</td><td>5/5</td><td>3/5</td></tr><tr><td>IM-2</td><td>Shifted composition</td><td>M → N → M</td><td></td><td>5/5</td><td>1/5</td></tr><tr><td></td><td></td><td></td><td>Completed experiments</td><td>45</td><td>44/45</td><td>34/45</td></tr></table>

![](images/822fa39be34684424a8459787541def6b0b818622bdd0dd69c65f27611b3771e.jpg)

![](images/0fc2ed7c4b379c8e658a66a232e4cdf61062de1d132dfa3a8ff7a20b697227bd.jpg)  
(a) Scene IN-2

![](images/47ebc48077c90b01cf485099615230c27b7f46ed99cc952a56ac7e387258b164.jpg)

![](images/e69acd89d5d9e408214b0aa17e4d63ac60629be8281204435a627a662771373c.jpg)

![](images/2f8adf51c60150f1e3f324fa7ebb4df5f9df214be502ffe5a1e4766266e8c5a8.jpg)

![](images/cd8b30358970e667221b179ab0d8d945df561594943f042bce3368bd926e2143.jpg)  
(b) Scene IN-3

Fig. 5: Qualitative hardware examples evaluating object selection. In both scenes, the planner must identify and manipulate the appropriate blocking object before navigating toward the goal.  
![](images/81ee25a4609ec858bf7aeda89768a92ca2fdd5783ae6cdcf236fbb9e8b3eebf6.jpg)  
Fig. 6: Long horizon task execution for LH-1. The robot appropriately navigates to the only movable object in the scene, manipulates it to create enough space to then navigate to the goal.

At the level of complete physical execution, the pipeline successfully completes 30/35 in-distribution trials. Apart from the skill composition error in IN-1, the remaining failures arise either due to incorrect geometric target selection or low-level execution. In the long horizon case LH-1, one failure results from the manipulation policy not reaching the prescribed terminal accuracy threshold (i.e., the planner was not called for inference at the next decision state), while the other is due to the manipulation target generated by the planner placing the object close to the final target, thus the subsequent navigation stage does not admit a collision-free path to the goal. In LH-2, both failures occur due to the manipulation policy driving the robot outside the room boundary.

We additionally evaluate the planner on two shifted interactive manipulation cases requiring the $\mathrm { ~ M ~ } \to \mathrm { ~ N ~ } \to \mathrm { ~ M ~ }$ skill composition. The planner predicts the correct composition in 10/10 trials, while the complete pipeline succeeds end-toend in 4/10 trials. At the skill level, the planner consistently identifies that the blocking object must first be manipulated before navigating toward and manipulating the target object. We further inspect the token probabilities associated with the first manipulation decision and object selection. The model assigns high probability to the manipulation prefix and to the selected blocking object, with an average token probability of $P _ { a v g } = 0 . 9 2 3$ across the aggregated tokens; however, at the final token related to the generated skill name, the probability is split between \_to and \_object. The latter would produce manipulate\_object, which is not part of the defined skill vocabulary. Thus, while the model exhibits some uncertainty in the exact output representation, the probability mass remains associated with manipulation rather than direct navigation to the target object.

The deployment failures in the shifted task IM-1 were associated with navigation targets that placed the robot at object-relative distances outside those observed during training of the manipulation policy, resulting in inconsistent low-level execution. In IM-2, the errors were instead related to the yaw command of the navigation target. These failures can be interpreted from both the planner and low-level policy perspectives. On the low-level side, the manipulation policy learns to manipulate the object predominantly with its left side, irrespective of the relative pose between the robot and the object. Thus, when a navigation action repositions the robot with the object to its right, the resulting state lies outside of the manipulation policy’s training distribution, leading to inconsistent execution. At the higher level, the underlying issue is that the planner generates geometric targets without knowledge of the execution capabilities of the parametrized actions. Consequently, correcting such failures only at the low level would require post-hoc adaptation or retraining of the manipulation policy to accommodate planner-generated states. A tighter coupling between high-level target generation and low-level policy dynamics would instead allow the planner to avoid issuing commands that are difficult for the downstream policies to execute reliably.

![](images/7997f2293ec1e4e0c6906de42b96ef9b57fc24af977c8c196763419d71a0b77a.jpg)

![](images/f180b03304d73feb9c79ba992e2e6e1556fdafc1175b22e24b269e8654b7cb6d.jpg)  
(a) Scene IN-1

![](images/41f78ad7c8ed805a4f3b399bccd5f931dc859b4ca88da4e3da942eac4fddc93d.jpg)

![](images/6169832b383aeb1312717158e95708ba7ac30eac6a28b2170e8c4a102660f80e.jpg)

![](images/e139d46776515e32c227d0efa7d084cede9747b596c6e78d67c906e6c380918c.jpg)

![](images/6c43c66092521055c43a23e727a62fb8009c5c65ddedeb643ca238fe90956c5b.jpg)  
(b) Scene OM-1

![](images/c00c491f432c75851e73f945fc1ae64427bd9cc4fd854e81db6a7397dfde35dc.jpg)

![](images/7cb5357761fc367d2354df4dbaf30060b5a9bae0025061f98b608b1dfda9b1a6.jpg)  
(c) Scene IM-2

![](images/b852b8780a67b397ed3604c6583e2c900db83000203efadd35baf424661bbcbc.jpg)

![](images/8813d8436ef7404fae99d1dfcaff8cf4b2e983cef84b68362f3cf46c614a9c4e.jpg)  
Fig. 7: Qualitative hardware examples of skill composition. Scene IN-1 requires M → N, whereas Scene OM-1 require N → M. In Scene IM-2, the planner composes these into the sequence M → N → M, which is excluded from training.

In summary, the planner trained on synthetic data transfers to hardware with the correct skill composition in 44/45 trials, while end-to-end success is limited by the decoupling between planner-generated targets and low-level execution, particularly for long-horizon and shifted tasks.

Finally, the examples in Figures 5, 6, and 7 illustrate the different task-level decisions evaluated above. In Figure 5, the two objects near the robot are movable obstacles, whereas the third is immovable. Between IN-2 and IN-3, the immovable obstacle is placed behind a different movable object, changing which object must be displaced to create a feasible route to the goal. The planner correctly changes its object selection according to this geometric relation. The LH-1 example in Figure 6 illustrates the longer-horizon N → M → N composition, where the robot must first navigate to the blocker, manipulate it to clear the path, and subsequently navigate to the final goal. Finally, Fig. 7 contrasts task structures represented during training with a successful execution of the shifted IM-2 case, in which previously learned navigation and manipulation behaviors are recombined into the novel M → N → M sequence.

## D. Limitations and Future Work

Although the proposed framework offers a quick and deployable recipe for finetuning LLMs for loco-manipulation tasks, one limitation is that the planner is only inferenced at discrete transition steps. Future work should investigate more frequent inference, enabling better plan adaptation under changes in the environment. Additionally, our evaluation is restricted to a single room configuration; future work should extend the framework across different room layouts. Another limitation of the framework is that the planner has no explicit model of the execution feasibility of its parameterized skills, leading to end-to-end failures on hardware. Finally, a natural extension of this project is to rely on local information (i.e. perception) rather than assuming that the pose of the entities in the scene is available.

## V. CONCLUSION

In this work, we proposed a hierarchical framework for quadruped planar loco-manipulation in cluttered environments. The high-level planner, a language-conditioned task planner finetuned with LoRA on synthetic data, jointly selects the skill, the relevant object, and a geometric target, and learns the geometric relations underlying interactive navigation without handcrafted rules or large prompts. Comparisons with models trained from scratch indicate that skill composition, including into a task structure excluded from training, does not require pretraining when the representation is structured, whereas pretraining yields more reliable geometric targets, which remain the main limitation of full-task success. At the low level, the proposed end-to-end whole-body manipulation policy adapts its behavior to a wide range of object properties and exhibits an efficient planar pushing behavior, improving success and reducing task-completion time by more than 60% compared with the commonly used hierarchical approach. On hardware, the complete framework produces the correct skill composition in 44/45 trials and succeeds end-to-end in 34/45.

## REFERENCES

[1] C. Dai et al., “Interactive navigation with adaptive non-prehensile mobile manipulation,” 2024. [Online]. Available: https://arxiv.org/abs/ 2410.13418

[2] Z. Bi et al., “Interactive navigation for legged manipulators with learned arm-pushing controller,” 2025. [Online]. Available: https://arxiv.org/abs/2503.01474

[3] K. Zhang et al., “Navigation among movable obstacles using machine learning based total time cost optimization,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 11 321–11 327.

[4] B. He et al., “Interactive-far:interactive, fast and adaptable routing for navigation among movable obstacles in complex unknown environments,” 2024. [Online]. Available: https://arxiv.org/abs/2404. 07447

[5] C. Li et al., “Hrl4in: Hierarchical reinforcement learning for interactive navigation with mobile manipulators,” 2019. [Online]. Available: https://arxiv.org/abs/1910.11432

[6] L. Yao et al., “Local path planning among pushable objects based on reinforcement learning,” 2024. [Online]. Available: https: //arxiv.org/abs/2303.02407

[7] K.-H. Zeng et al., “Pushing it out of the way: Interactive visual navigation,” 2021. [Online]. Available: https://arxiv.org/abs/2104.14040

[8] T. Yang et al., “Efficient navigation among movable obstacles using a mobile manipulator via hierarchical policy learning,” 2025. [Online]. Available: https://arxiv.org/abs/2506.15380

[9] K. Zhou et al., “Adaptive interactive navigation of quadruped robots using large language models,” 2025. [Online]. Available: https://arxiv.org/abs/2503.22942

[10] ——, “Coin: Interactive navigation with counterfactual reasoning via vision–language models,” IEEE Transactions on Automation Science and Engineering, vol. 23, pp. 14 881–14 899, 2026.

[11] T. B. Brown et al., “Language models are few-shot learners,” 2020. [Online]. Available: https://arxiv.org/abs/2005.14165

[12] Y. Ouyang et al., “Long-horizon locomotion and manipulation on a quadrupedal robot with large language models,” 2025. [Online]. Available: https://arxiv.org/abs/2404.05291

[13] R.-Z. Qiu et al., “Wildlma: Long horizon loco-manipulation in the wild,” 2025. [Online]. Available: https://arxiv.org/abs/2411.15131

[14] Y. Mei et al., “Quadrupedgpt: Towards a versatile quadruped agent in open-ended worlds,” 2024. [Online]. Available: https: //arxiv.org/abs/2406.16578

[15] A. S. Chen et al., “Commonsense reasoning for legged robot adaptation with vision-language models,” 2024. [Online]. Available: https://arxiv.org/abs/2407.02666

[16] K. Qu et al., “A pragmatist robot: Learning to plan tasks by experiencing the real world,” 2026. [Online]. Available: https: //arxiv.org/abs/2507.16713

[17] K. Wang et al., “Odyssey: Open-world quadrupeds exploration and manipulation for long-horizon tasks,” arXiv preprint arXiv:2508.08240, 2025.

[18] Z. Ravichandran et al., “Distilling on-device language models for robot planning with minimal human intervention,” 2025. [Online]. Available: https://arxiv.org/abs/2506.17486

[19] M. J. Kim et al., “Openvla: An open-source vision-language-action model,” 2024. [Online]. Available: https://arxiv.org/abs/2406.09246

[20] M. J. Kim, C. Finn, and P. Liang, “Fine-tuning vision-language-action models: Optimizing speed and success,” 2025. [Online]. Available: https://arxiv.org/abs/2502.19645

[21] S. Nasrat et al., “Rdmm: Fine-tuned llm models for on-device robotic decision making with enhanced contextual awareness in specific domains,” 2025. [Online]. Available: https://arxiv.org/abs/2501.16899

[22] A. Brohan et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” 2023. [Online]. Available: https://arxiv.org/abs/2307.15818

[23] S. Wei et al., “ψ<sub>0</sub>: An open foundation model towards universal humanoid loco-manipulation,” 2026. [Online]. Available: https://arxiv. org/abs/2603.12263

[24] K. Black et al., “π<sub>0</sub>: A vision-language-action flow model for general robot control,” 2026. [Online]. Available: https://arxiv.org/abs/2410. 24164

[25] P. Intelligence et al., “π<sub>0.5</sub>: a vision-language-action model with open-world generalization,” 2025. [Online]. Available: https://arxiv.org/ abs/2504.16054

[26] NVIDIA et al., “Gr00t n1: An open foundation model for generalist humanoid robots,” 2025. [Online]. Available: https://arxiv.org/abs/2503. 14734

[27] J.-P. Sleiman et al., “A unified mpc framework for whole-body dynamic locomotion and manipulation,” 2021. [Online]. Available: https://arxiv.org/abs/2103.00946

[28] J.-P. Sleiman, F. Farshidian, and M. Hutter, “Versatile multicontact planning and control for legged loco-manipulation,” Science Robotics, vol. 8, no. 81, p. eadg5014, 2023. [Online]. Available: https: //www.science.org/doi/abs/10.1126/scirobotics.adg5014

[29] S. Jeon et al., “Learning whole-body manipulation for quadrupedal robot,” 2023. [Online]. Available: https://arxiv.org/abs/2308.16820

[30] M. Sombolestan and Q. Nguyen, “Hierarchical adaptive locomanipulation control for quadruped robots,” 2023. [Online]. Available: https://arxiv.org/abs/2209.13145

[31] Y. Feng et al., “Learning multi-agent loco-manipulation for longhorizon quadrupedal pushing,” 2025. [Online]. Available: https: //arxiv.org/abs/2411.07104

[32] M. Sombolestan and Q. Nguyen, “Hierarchical adaptive control for collaborative manipulation of a rigid object by quadrupedal robots,” 2023. [Online]. Available: https://arxiv.org/abs/2303.06741

[33] Qwen et al., “Qwen2.5 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2412.15115

[34] E. J. Hu et al., “Lora: Low-rank adaptation of large language models,” 2021. [Online]. Available: https://arxiv.org/abs/2106.09685

[35] A. Mandlekar et al., “Mimicgen: A data generation system for scalable robot learning using human demonstrations,” 2023. [Online]. Available: https://arxiv.org/abs/2310.17596

[36] Z. Jiang et al., “Dexmimicgen: Automated data generation for bimanual dexterous manipulation via imitation learning,” 2025. [Online]. Available: https://arxiv.org/abs/2410.24185

[37] S. Choudhury and P. Sodhi, “Better than your teacher: Llm agents that learn from privileged ai feedback,” 2024. [Online]. Available: https://arxiv.org/abs/2410.05434

[38] J. Schulman et al., “Proximal policy optimization algorithms,” 2017. [Online]. Available: https://arxiv.org/abs/1707.06347

[39] M. Mittal et al., “Orbit: A unified simulation framework for interactive robot learning environments,” IEEE Robotics and Automation Letters, vol. 8, no. 6, p. 3740–3747, Jun. 2023. [Online]. Available: http://dx.doi.org/10.1109/LRA.2023.3270034

[40] L. Pinto et al., “Asymmetric actor critic for image-based robot learning,” 2017. [Online]. Available: https://arxiv.org/abs/1710.06542

[41] A. Petrenko et al., “Dexpbt: Scaling up dexterous manipulation for hand-arm systems with population based training,” 2023. [Online]. Available: https://arxiv.org/abs/2305.12127

[42] A. Allshire et al., “Transferring dexterous manipulation from gpu simulation to a remote real-world trifinger,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, Oct. 2022. [Online]. Available: http://dx.doi.org/10.1109/IROS47612.2022.9981458

[43] I. Dadiotis et al., “Dynamic object goal pushing with mobile manipulators through model-free constrained reinforcement learning,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, May 2025, p. 13363–13369. [Online]. Available: http://dx.doi.org/10.1109/ICRA55743.2025.11128166

[44] T. Z. Zhao et al., “Learning fine-grained bimanual manipulation with low-cost hardware,” 2023. [Online]. Available: https://arxiv.org/abs/ 2304.13705

[45] J. Xu et al., “Prediction-guided multi-objective reinforcement learning for continuous robot control,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, H. D. III and A. Singh, Eds., vol. 119. PMLR, 13–18 Jul 2020, pp. 10 607–10 616. [Online]. Available: https://proceedings.mlr.press/v119/xu20h.html

[46] H. Munn et al., “Scalable multi-objective robot reinforcement learning through gradient conflict resolution,” 2025. [Online]. Available: https://arxiv.org/abs/2509.14816

[47] N. Rudin et al., “Advanced skills by learning locomotion and local navigation end-to-end,” 2022. [Online]. Available: https://arxiv.org/abs/ 2209.12827

[48] S. Choi et al., “Learning quadrupedal locomotion on deformable terrain,” Science Robotics, vol. 8, 01 2023.

[49] Y. Liu et al., “Bidirectional decoding: Improving action chunking via guided test-time sampling,” 2025. [Online]. Available: https: //arxiv.org/abs/2408.17355

[50] K. Black et al., “Training-time action conditioning for efficient real-time chunking,” 2025. [Online]. Available: https://arxiv.org/abs/2512.05964

## APPENDIX A NOTATION

General: Unless otherwise noted, boldface denotes vectors, and plain symbols denote scalars. We use the following reference frames: $W$ (world), R (robot/base), O (object), and $G$ (goal). For any quantity $\bigsqcup , { } ^ { F } \bigsqcup _ { A B }$ denotes the quantity of $B$ relative to $A ,$ expressed in frame $F ; \mathrm { { e . g . } }$ , the position of the goal relative to the object, expressed in the robot frame is denoted as ${ } ^ { R } { \bf p } _ { O G }$ . The unit normalized vector is aˆ, and ∥·∥ denotes the Euclidean $( \ell _ { 2 } )$ norm. Yaw angles are denoted by ψ $\in ( - \pi , \pi ]$ (a coordinate on SO(2)). The actor observations are denoted by $o _ { s }$ where $s = [ n , m ]$ refers to navigation and manipulation skills, respectively.

Robot-related quantities: The joint configuration is $\textbf { q } \in$ $\mathbb { R } ^ { 1 2 }$ with the default (nominal) configuration ${ \bf q } _ { 0 }$ and offsets $\Delta \mathbf { q } = \mathbf { q } - \mathbf { q } _ { 0 } ;$ joint velocities and torques are ${ \dot { \mathbf { q } } } ,$ and $\tau ,$ respectively. The base (body) pose is given by $w _ { \mathbf { p } _ { W R } } \in \mathbb { R } ^ { 3 }$ (position of R in W), and yaw $\psi _ { R }$ (rotation about the $W \ z -$ axis). The base twist comprises linear and angular velocities expressed in R, denoted by ${ \bf v } _ { R } \in \mathbb { R } ^ { 3 }$ and $\omega _ { R } \in \mathbb { R } ^ { 3 }$ . v<sub>R,i</sub> denotes the i-th component of the velocity vector of the robot $\mathbf { v } _ { R } .$

Object-related quantities: the object pose is ${ } ^ { W } { \bf p } _ { O } \in \mathbb { R } ^ { 3 }$ (position of O in W) with yaw ψ<sub>O</sub>; the object twist is $\mathbf { v } _ { O } \in \mathbb { R } ^ { 3 }$ and $\omega _ { O } \in \mathbb { R } ^ { 3 }$ (linear and angular velocities of O). The goal pose is ${ \cal W } _ { { \bf p } _ { G } } \in \mathbb { R } ^ { 3 }$ with desired yaw ψ<sub>G</sub>.

## APPENDIX B

## TRAINING DETAILS

## A. Training and platform details

Policy training was performed on a workstation equipped with an Intel<sup>®</sup> Core<sup>™</sup> Ultra 9 275HX CPU (24 cores) and an NVIDIA RTX 5080 GPU. The manipulation policy $\pi _ { m }$ was trained for 8000 iterations, while the high-level navigation planner $\pi _ { n }$ was trained for 6000 iterations. Manipulation and locomotion policies were trained using 4096 and 1024 parallel environments, respectively, with 24 environment steps per iteration. Training required approximately 3 hours for the manipulation policy and 4 hours for the locomotion policy.

LLM fine-tuning was performed on two NVIDIA RTX 5090 GPUs and required approximately 45 minutes. During deployment, action inference runs at 1 Hz on an RTX 5080 GPU.

## B. Navigation policy

The inputs to the navigation policy are proprioception information $o _ { n , t } \in \mathbb { R } ^ { 2 7 }$ (joint positions w.r.t. default configuration, joint velocities and base angular velocity), previous actions $\mathbf { a _ { t - 1 } } \in \mathbb { R } ^ { 3 }$ , pose commands $\mathbf { c _ { t } } ,$ and depth data $\mathbf { h } _ { \mathbf { t } } \in \mathbb { R } ^ { 7 }$ coming from the on-board LiDAR. h<sub>t</sub> represents the maximum depth from the sensor to the ray hit location for seven discrete ray angles $\in [ - \pi / 2 , \pi / 2 ]$ . This choice of environment sensing is due to the built-in L1 sensor of the Unitree Go2, which has a non-repetitive scan pattern and produces sparser depth readings than other commonly used sensors.

The terrain consists of a varying number of randomly generated rectangular obstacles with varying widths and heights, as done in [47], positioned in a square of dimension $4 \times 4$ $[ \mathrm { m } ^ { 2 } ]$ centered at the origin of the environment. An area of dimension $1 \times 1 ~ [ \mathrm { m ^ { 2 } } ]$ is left clear at the center of the environment, and serves as the robot’s spawning platform. Position commands are sampled in polar coordinates outside of the obstacle area, while goal and robot yaw are sampled to cover the entire space $\in [ - \pi , \pi ]$

TABLE VII: Reward terms used for training the manipulation policy.
<table><tr><td>Reward term</td><td>Description</td><td>Weight</td></tr><tr><td> $r e w _ { \mathrm { k p } }$ </td><td>Object-goal keypoint tracking</td><td>20.0</td></tr><tr><td> $r e w _ { v }$ </td><td>Object velocity tracking</td><td>15.0</td></tr><tr><td> $r e w _ { { \boldsymbol { v } } _ { R , x } }$ </td><td>Positive robot forward velocity</td><td>0.5</td></tr><tr><td> $| | \ddot { \mathbf { q } } | | ^ { 2 }$ </td><td>Joint acceleration penalty</td><td> $- 2 . 5 * 1 0 ^ { - 7 }$ </td></tr><tr><td> $\| \tau \| ^ { 2 }$ </td><td>Joint torque penalty</td><td> $- 2 * 1 0 ^ { - 4 }$ </td></tr><tr><td> $\| \Delta \mathbf { a } _ { t - 1 , t } \| ^ { 2 }$ </td><td>Action-rate penalty</td><td>-0.4</td></tr><tr><td> $\| \omega _ { O , x y } \| ^ { 2 }$ </td><td>Object roll/pitch velocity penalty</td><td>-2.0</td></tr><tr><td> $\| \omega _ { R , x y } \| ^ { 2 }$ </td><td>Robot roll/pitch velocity penalty</td><td>-0.2</td></tr><tr><td> $\| \mathbf { a } _ { \mathrm { f e e t } } \| ^ { 2 }$ </td><td>Feet acceleration penalty</td><td> $- 8 * 1 0 ^ { - 3 }$ </td></tr><tr><td> $\| \mathbf { q } - \mathbf { q } _ { 0 } \| ^ { 2 }$ </td><td>Joint deviation from default configuration</td><td>-0.5</td></tr></table>

TABLE VIII: PPO and network hyperparameters used for training the low-level policies with RSL-RL.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Rollout steps per update Action clipping Actor hidden layers Critic hidden layers Activation function</td><td>24 5.0 [512, 256, 128]</td></tr><tr><td>Initial action noise std.</td><td>[512, 256, 128] ELU</td></tr><tr><td>Value loss coefficient</td><td>1.0 1.0</td></tr><tr><td>Clipped value loss PPO clipping parameter Entropy coefficient</td><td>True 0.2</td></tr><tr><td></td><td></td></tr><tr><td>Learning epochs per update</td><td>0.005 5</td></tr><tr><td>Mini-batches per update</td><td>4</td></tr><tr><td>Learning rate Learning-rate schedule</td><td> $1 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Discount factor γ GAE parameter λ</td><td>Adaptive 0.99</td></tr></table>

## C. Manipulation policy

The complete list of rewards is provided in Table VII. PPO and network hyperparameters are provided in Table VIII.

The total reward r follows,

$$
r = \sum _ { i = 1 } ^ { N } w _ { i } r e w _ { i }\tag{9}
$$

where rew<sub>i</sub> denotes each reward term i, and $w _ { i }$ its associated weight.

We use an action scale of $a _ { s c a l e } = 0 . 2 5$ , and PD gains of $K _ { p } = 2 5$ and $K _ { d } = 0 . 5$ . We do not normalize the observations before feeding them to the policy.

## APPENDIX C

## HRL VS E2E LOCO-MANIPULATION POLICIES

As discussed in IV-A, the HRL loco-manipulation policy was trained following the pipeline of [29], excluding the adaptation module. Additionally, the low-level velocity tracking policy is slightly simpler than the one implemented (based on [48]) and does not require any state estimation. Although using the low-level policy from [48] could lead to a slightly more performant loco-manipulation policy, it does not solve the imposed constraint on the policy’s behavior. We found that training the E2E loco-manipulation policy is trickier than training the HRL policy, similar to what was discussed in [29]; however, the following method worked consistently well: We first reward the policy for approaching the object with a heading inline with the target vector ${ w _ { \mathbf { p } } } _ { O G }$ , this stage can be seen as a ”warm start” to the policy as it learns a ”base” stable behavior. When the robot has learned to approach the object, it will collide with it inducing a motion towards the goal. At this stage, the reward for reposing the object is activated. The initial learned behavior of colliding with the object offers a strong signal for the manipulation phase. The number of training iterations for the first phase is important as activating the manipulation reward too late will cause big variations in policy update. On the other hand, activating it too early and the robot doesn’t learn as well due to the conflicting objectives, as demonstrated in figure 8.

![](images/b78fe15b1744d094d5e1a649587df6095d444d91bda5f249e60759a3dff861ee.jpg)

![](images/60763ff88e3e0da652b3a5a13c89e2196ce4243989f0158b82df35392c28f580.jpg)  
(a) Manipulation reward for the three E2E locomanipulation policies over training steps.  
(b) Action standard deviation of the three E2E locomanipulation policies over training steps.

Both policies are trained for the same environment configuration, and randomization. Reward tuning was done separately to ensure a good policy checkpoint has been reached.

Figure 9 shows the energy consumption for both methods for the following two runs: (a) The object is distant from the robot, (b) the object is close to the robot. In both cases, we see that the E2E locomanipulation policy is significantly faster than the HRL approach in reposing the object to the goal. However, it is only more energy efficient for case b. Table IX shows the per-phase energy consumption for each case. We note that the HRL policy consumes less energy in the manipulation phase of the first run, even with a 50% longer episode time. The E2E approach is rather more energetically efficient for cases where the robot is close to the object, and more notably for a long manipulation phase. We note that the E2E policy shown in this experiment was trained without an explicit penalty on foot accelerations. Incorporating this penalty reduces impact forces on hardware, resulting in lower impact noise and reduced kinetic energy dissipation.

Fig. 8: End-to-end loco-manipulation policy learning dynamics. Manipulation reward was activated at iteration 0 for the orange curve, iteration 400 for the cyan curve, and iteration 1000 for the red curve.  
![](images/6f4070c5f38150d63705375c9db39f6f3f1e1fcc7631f55e39cec9b20c895939.jpg)  
(a)

![](images/25fabd3aa2e853724e14954beb0d59327f61aed775a42caeb2f13591734a39e0.jpg)  
(b)  
Fig. 9: Comparison of Energy Consumption between HRL and E2E Loco-manipulation policies. (a) represents the case in which the object distant from the robot, and (b) represents the case in which the object close to the robot. Blue and Orange dashed lines represent the first instant in which the robot is in contact with the object for E2E and HRL, respectively. In these two runs, the object is a cube with an edge length of 40 [cm], weight and CoM position are inside the training distribution. We provide a video showing the behavioral difference between both methods for the first case.

TABLE IX: Energy in [J] for locomotion and manipulation phases, for both cases.
<table><tr><td></td><td colspan="2">Run 1</td><td colspan="2">Run 2</td></tr><tr><td>Method</td><td>Locomotion</td><td>Manipulation</td><td>Locomotion</td><td>Manipulation</td></tr><tr><td>E2E</td><td>359.56</td><td>684.28</td><td>136.7</td><td>903.3</td></tr><tr><td>HRL</td><td>218.63</td><td>337.42</td><td>133.9</td><td>984.0</td></tr></table>

We note that the reason behind the lower completion time of our method, or end-to-end approach in general, is not due to a higher base velocity. This was shown in section IV-A, where we compare both loco-manipulation approaches and figure 3 show that both policies achieve the locomotion phase in approximately similar durations. Thus, demonstrating that the end-to-end approach is not faster in terms of robot base velocity, it is only better suited for manipulation.

## APPENDIX D

## MANIPULATION POLICY ABLATION

The ablation study was carried out on the same environment configuration used for the comparison in Section IV-B. Results are presented in Table X. In the case where we omitted the positive velocity reward, we observe that (FO) increases by an order of magnitude when compared to our solution, and success rate drops to 20.7%. The average error for (TO) runs is 0.28 [m] and $4 0 ^ { \circ }$ , indicating that the robot is falling over around the end of the episode. Upon closer inspection, the cases related to Fall Over were mostly due to the robot flipping over. Without this reward, the robot chooses to manipulate the object with its lateral side. The problem arises once the robot has to detach from the object. Once no opposing force is present to stabilize the configuration of the robot, it either (i) falls over due to the CoM being outside the support polygon, or (ii) collides with the object, pushing it away from the goal, and then re-stabilizes itself. Left with little time to re-adjust the pose of the object, 73.3% of the runs end in Time Out (TO). Our method leads to significantly higher success rate since, due to the geometry of the robot, the range of stable pitch angles is bigger than the range of stable roll angles. Also, within the successful runs, we notice that our method emerges with a lower (ET). The robot can displace the object at higher velocities, in part due to the increased stability and the alignment of the calf and thigh motors with the direction of motion of the robot. Instead, when pushing to the side, the motors on the hip joints constitute the main movers. By carefully modulating this reward, we are also able to emerge with a dual effect, where the robot pushes with a forward heading, until it comes close to the goal position, where it rotates the object from the side while having a more stable configuration.

TABLE X: Ablation Results.
<table><tr><td>Method</td><td>Success Rate (%)</td><td>Episode Time (s)</td><td>Fall Over (%)</td><td>Energy (J)</td></tr><tr><td>w/o Positive Velocity</td><td>20.70</td><td> $5 . 8 6 \pm 1 . 9 9$ </td><td>6.0</td><td> $2 4 4 6 . 1 9 \pm 6 8 7 . 4 4$ </td></tr><tr><td>w/o rv</td><td>87.30</td><td> ${ \bf 3 . 7 7 \pm 1 . 0 9 }$ </td><td>7.5</td><td> $2 0 2 8 . 2 8 \pm 7 8 9 . 6 7$ </td></tr><tr><td>Ours</td><td>93.30</td><td> $4 . 0 9 \pm 1 . 3 1$ </td><td>0.8</td><td> $\mathbf { 1 6 8 0 . 8 0 \pm 6 7 8 . 1 2 }$ </td></tr></table>

Removing $r _ { v }$ does not affect the success rate drastically, however we notice a ten-fold increase in (FO) rate when compared to ours, mainly attributed to the object falling over. Excluding our $r _ { v }$ reward leads to a faster, more aggressive behavior towards the end of the episode, also reflected by the lower (ET).

## APPENDIX E

## LLM TASK PLANNER

## A. Data generation and augmentation

The cases are ordered by increasing difficulty, where difficulty is defined as a function of (i) the number of required skill switches and (ii) the degree of spatial reasoning necessary to resolve the task objective. This structured decomposition induces a curriculum over task complexity and simplifies scalable data generation and augmentation. The proposed case progression exposes the model to increasingly complex spatial configurations, transitioning from direct goal reaching to scenarios that require explicit reasoning over object placement and skill switching. At this stage, we note that updates are purely deterministic (e.g., navigation sets the robot to the target pose; manipulation places the object at the desired pose). Trajectory generation, therefore, operates at a symbolic, tasklevel abstraction.

Within Case 2, task difficulty is further modulated by controlling object count and spatial arrangement. Simpler instances include a single object or multiple objects that are wellseparated from the robot’s path. More challenging instances involve multiple objects arranged to create narrow yet navigable corridors. This structured variation discourages unnecessary manipulation by exposing the model to a broad distribution of non-blocking configurations, thereby reinforcing correct skill selection when navigation suffices. We additionally introduce Case 2.5, which corresponds to object-centric commands (e.g., manipulating object ”A” to a specified pose). This case isolates the structure of manipulation instructions and their associated scene dependencies (analogous to Case 1 for navigation).

Data generation and augmentation are performed in a casedependent manner to ensure controlled coverage of increasing task complexity, imposing a deliberate bias towards more challenging scenarios. The following describes in detail data generation and initial augmentation.

Easy Cases (Case 1 and simple Case 2): For lowcomplexity scenarios, a small set of base environments is manually constructed to define representative spatial layouts. These base templates are then augmented by sampling new robot poses, goal poses, and (when present) object poses from predefined spatial distributions. The sampling ranges are chosen to preserve the defining characteristics of each case (e.g., maintaining non-blocking configurations in Case 2).

Medium-Complexity Cases (Case 2 – Narrow Corridors): For more challenging navigation-only scenarios, we manually design a diverse set of narrow but traversable passage configurations. These templates are augmented using the same pose-randomization strategy, but under a different spatial distribution that preserves corridor geometry while varying robot, goal, and object placements. This ensures variability while maintaining the requirement that navigation remains feasible without manipulation.

Hard Cases (Case 3 – Obstructed Scenes): For obstruction scenarios requiring skill switching (i.e., Interactive Navigation tasks), manual generation plays a larger role to ensure broad semantic coverage. We vary both movable and immovable blocking objects, as this directly affects the required action sequence. Augmentation is again performed via controlled pose randomization, using distributions that preserve the blocking constraint. Ensuring wide coverage in this case is critical; insufficient variation would fail to expose the model to the diversity of manipulation–navigation transitions necessary for robust planning.

Additionally, we apply the following structure-preserving augmentations:

• Object ID and type randomization: Object identifiers are uniformly sampled within a fixed range (up to six objects per scene) to prevent overfitting to index-based correlations. We also randomize the type of the object (circle or rectangle while keeping the surface area equal).

• Transition-level state randomization: Since the transitions in our synthetic dataset are purely deterministic (which is not representative of the real world), we insert a uniform noise to the initial robot and object pose (ensuring no semantic change has occurred), and to the state transitions caused by the actions. This is different from the initial augmentation done where the goal was to cover as much of the state space for each entity.

• Symmetry transformations: Scenes are mirrored across spatial quadrants, yielding up to four equivalent geometric variants per configuration.

• Irrelevant object injection: One or two irrelevant objects are inserted at non-interfering positions to increase robustness to irrelevant objects without altering the groundtruth action sequence.

Once the augmented trajectories are obtained, they undergo a feasibility verification step to ensure collision-free execution and consistency with the intended task constraints. Any invalid configurations are discarded. Approximately 85% of the generated trajectories were automatically generated.

## B. LLM finetuning

We choose to finetune the model with LoRA , since [19] report minor improvements w.r.t full finetuning at a fraction of the cost. While in our setting the predicted action chunks are considerably shorter compared to VLA models for robotic manipulators [20], [23], [44], [49], [50] due to our task-level planner design, we find that predicting the complete action sequence enables the model to compose the behaviors learned during training to form novel skill sequences.

Although the aforementioned studies have shown that LLM planners, through AC (and its more intricate variants such as RTC [50]), can output consistent long sequences of actions, we opt to train the model on discrete skill-transition actions. This design choice presents a natural mitigation strategy for compounding errors and is motivated by the inherent complexity in the deployment of legged robots, and the task we are solving.

Specifically, the LLM planner would have to output sequences of geometric targets that are aligned with the low-level capabilities of the robot. Such a geometric output is outside the current capabilities of the model since it was finetuned on synthetic data. This also motivates the choice behind inferring the planner once the previous skill is done, and the robot has acquired a stable configuration. We would like to note that the current model could (in theory) be inferenced continuously rather than waiting for previous actions to be completed; however, the LLM does not have access to robot proprioception information that would allow it to infer the motion state of the robot (stationary or in motion). The problem arises when the robot has a non-zero base velocity. To enable continuous inference, the LLM has to first have access to such information and must be able to output idle actions. For example if the LLM gives a conservative manipulation target (which is representative of the expert data we generated) to make enough space to navigate, and we continuously inference the model, the skill (and consequently its target) might change given that the model reasons that now the robot has enough navigation space. This would lead to a big Sim2Real gap, since (1) the LiDAR data would effectively identify the contact as a collision, and (2) the robot has to change its behavior from manipulation to locomotion in a very short time span. Ideally, the LLM would have access to proprioception information onboard the robot, and output an idle target that is close to the current object position. This would ensure that the transition between the different skills is smooth and does not lead the robot to fall over. Unfortunately, the current planner is unable to infer such actions. We leave this step for future work.

TABLE XI: LLM fine-tuning details.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Dataset size</td><td>16K I/O pairs (8096 trajectories)</td></tr><tr><td>Validation dataset size</td><td>5% of total dataset</td></tr><tr><td>LoRA rank</td><td>128</td></tr><tr><td>LoRA alpha</td><td>256</td></tr><tr><td>LoRA dropout</td><td>0.05</td></tr><tr><td>Learning rate</td><td>2e-5</td></tr><tr><td>LR schedule</td><td>Cosine with warmup</td></tr><tr><td>Warmup ratio</td><td>0.02</td></tr><tr><td>Batch size</td><td>16</td></tr></table>

Table XI summarizes the hyperparameters used for training the LLM. We note that the batch size is smaller than in prior work, leading to increased gradient noise and partial GPU under-utilization. However, with appropriate learning rate tuning, the model converges reliably. This batch size was selected as it yields the best empirical performance among the tested values [8, 16, 32, 64, 128]. A batch size of 128 represents the largest batch size achievable on our hardware without encountering out-of-memory errors. We note that we split the trajectories at the trajectory level to avoid leakage between subsequent decision states of a trajectory. Held-out ID trajectories may share the same task-case representation as training trajectories but contain independently perturbed robot, goal, and object configurations.

## C. Inference Procedure

During inference, the language-conditioned planner receives the current scene state from the OptiTrack system and predicts the complete remaining sequence of actions. However, only the first action is executed by sending its target to the corresponding low-level policy (navigation or manipulation). An action is considered complete once the relevant controlled entity reaches a predefined positional accuracy threshold of 15 [cm] and 15 ◦ with respect to the commanded target and the robot base velocity, estimated from a temporal sequence of robot-pose observations, falls below 0.01 m/s. At this point, the planner is queried again using the updated scene state, with the previously executed skill additionally provided as input.

If the active low-level policy fails to satisfy the completion criterion, the planner is not queried for the next action and does not intervene by issuing an alternative command. The corresponding run is therefore counted as an end-to-end failure.

## APPENDIX F PLANNER EVALUATION DETAILS

Throughout, per-action latency is the query time divided by the number of generated actions.

a) Classical planner: We design separate branches for each task, i.e. navigating to a goal pose or manipulating an object to a goal pose. When the navigation goal is unreachable, the planner considers movable objects as potential blockers, samples candidate relocation targets, and counter-factually updates the scene to determine whether each manipulation restores a feasible path. Manipulation is permitted only when the robot is sufficiently close to the selected object; otherwise, the planner first generates a navigation target that aligns the robot with the intended object-motion direction. An analogous branching strategy is used for object-manipulation tasks when access to the target object is obstructed. Planner parameters are tuned jointly across all task families, and a single parameter set is used for all evaluated scenes. The branch handling obstructed access to the target object corresponds to the shifted task structure, which is therefore explicitly encoded in the classical planner.

b) Qwen ICL baseline: The prompt explains the planner’s task, the workspace bounds, the object format, and the two skills with the 1.2 [m] proximity condition for manipulation. It lists the five admissible plan structures, including the shifted interactive-manipulation structure, followed by one demonstration per task family in Table XII and one counterexample of an interactive-navigation scene solved incorrectly by direct navigation. The Qwen2.5-1.5B ICL baseline receives this prompt for all evaluations, including the shifted task: since it fails on the shifted task even when its structure is described and demonstrated, no reduced prompt is required. In all evaluated scenes, it outputs a single action whose skill mirrors the verb of the task instruction.

c) Frontier-model ICL baseline: The frontier ICL baseline uses gpt-5.5-2026-04-23 with reasoning effort low, queried via the OpenAI API. On the ID task families, it receives the same prompt as the Qwen ICL baseline. For the evaluation on the shifted task, the shifted structure and its demonstration are removed from the prompt, leaving four plan structures and four demonstrations, so that, like our planner, the model is not informed of the shifted structure. In both cases, it generates the complete plan from the initial state in a single query, which is executed open-loop. Peraction latency is computed as the query time divided by the number of planned actions and includes network overhead; this metric was chosen due to the observed linear relationship between query latency and the number of actions generated per query. Across all held-out queries, the per-action latency is $7 . 0 \pm 3 . 4 \mathrm { ~ s ~ }$ (median: 5.9 s), and a full-plan query takes $1 4 . 0 \pm 6 . 8 \mathrm { ~ s . ~ A ~ }$ query uses on average 2,728 prompt tokens and 737 output tokens, of which 628 are reasoning tokens; variation in reasoning length accounts for most of the latency spread.

Our planner matches or exceeds GPT-5.5 on all indistribution task families and on the shifted task. Of its 26 interactive-navigation failures, 20 stem from geometric targets, five from selecting the wrong object, and one from insufficient clearance created by the manipulation; on longhorizon interactive navigation, eight of its 12 failures are geometric and four are skill-selection errors. On direct navigation, it tends to manipulate objects lying along the path even when sufficient clearance is available, in some cases with manipulation targets that overlap other objects. On the shifted task, it produces the correct composition in only 3/30 trajectories and completes 2/30. When the prompt additionally describes and demonstrates the shifted structure, it completes 22/30 with the correct composition in all 30, and its eight failures stem from geometric targets.

TABLE XII: Planner evaluation task families. (N) and (M) denote navigation and manipulation actions, respectively. ID denotes task structures represented in the training distribution, while Shifted denotes the task composition used to evaluate the planner’s ability to compose a novel sequence of actions.
<table><tr><td>Task family</td><td>Distribution</td><td>Task composition</td></tr><tr><td>Direct navigation</td><td>ID</td><td>N</td></tr><tr><td>Object manipulation</td><td>ID</td><td> $\mathrm { N }  \mathrm { M }$ </td></tr><tr><td>Interactive navigation</td><td>ID</td><td> $\mathrm { M } \to \mathrm { N }$ </td></tr><tr><td>Long-horizon interactive navigation</td><td>ID</td><td> $\mathrm { N }  \mathrm { M }  \mathrm { N }$ </td></tr><tr><td>Interactive manipulation</td><td>Shifted</td><td> $\mathrm { M } \to \mathrm { N } \to \mathrm { M }$ </td></tr></table>

## A. Long-Horizon Interactive Manipulation

As described in IV-C2, we further evaluate the planner on a longer interactive-manipulation stress test requiring the composition $\mathrm { N }  \mathrm { M }  \mathrm { N }  \mathrm { M }$ for Run 1. The purpose of this experiment is to probe how far the planner can extend the learned navigation and manipulation behaviors to a longer task structure. Across 30 trajectories, the planner completes the task only once. In six additional trajectories, it selects the correct skill sequence and satisfies the task constraints at each intermediate stage, but fails at the final manipulation where the predicted geometric target does not match the task target.

A recurrent failure pattern appears in 12 trajectories. In these cases, the planner selects the correct skill sequence, $\mathrm { ~ N ~ } \to \mathrm { ~ M ~ } \to \mathrm { ~ N ~ } \to \mathrm { ~ M ~ }$ , and the correct object at each manipulation stage, but the geometric parameters are invalid. Specifically, the initial navigation target is unreachable under the task constraints. Rather than subsequently selecting a feasible navigation action to the target object, the planner consistently predicts a manipulation that moves the blocking object in a pulling-like behavior. This pattern is not limited to re-planning at later decision states: the same sequence is already present in the complete plan generated from the initial state. Thus, although the discrete skill and object composition is correct in these cases, the associated geometric targets make the plan infeasible.

The remaining failures consist of three trajectories in which the composed sequence is correct; however, the intermediate manipulation target does not clear a navigable path, two trajectories with invalid action format, and six trajectories in which the generated sequence instead follows the structure of the Object manipulation task family.

Based on the results of this experiment, we find that the planner is able to compose learned manipulation and navigation behaviors at the skill-level; however, to a certain extent. Failures become more frequent, and are particularly concentrated around violating action and environment constraints that are implicitly represented in the training data. Future work should investigate how to compose these learned behaviors while complying with the constraints present in the training data and across various scenes.

## B. Ablation and Model-Selection Analysis

How do adaptation capacity, and action chunking affect ID and shifted-task performance?

TABLE XIII: Planner ablation and model-selection results, reported as (# of successes)/(# of trajectories). The shifted row reports results on the 30 validation scenes used for rank selection.
<table><tr><td>Task Family</td><td colspan="5">LoRA Rank (r)</td><td colspan="2">No AC</td></tr><tr><td></td><td> $r = 3 2$ </td><td> $r = 6 4$ </td><td> $r = 1 2 8$ </td><td> $r = 2 5 6$ </td><td> $r = 5 1 2$ </td><td> $r = 5 1 2$ </td><td> $r = 1 2 8$ </td></tr><tr><td>Direct navigation</td><td>37/40</td><td>37/40</td><td>38/40</td><td>38/40</td><td>39/40</td><td>37/40</td><td>39/40</td></tr><tr><td>Object manipulation</td><td>10/10</td><td>10/10</td><td>10/10</td><td>10/10</td><td>10/10</td><td>10/10</td><td>10/10</td></tr><tr><td>Interactive navigation</td><td>98/120</td><td>107/120</td><td>109/120</td><td>109/120</td><td>112/120</td><td>107/120</td><td>105/120</td></tr><tr><td>Long-horizon interactive navigation</td><td>25/30</td><td>29/30</td><td>29/30</td><td>29/30</td><td>29/30</td><td>28/30</td><td>27/30</td></tr><tr><td>Interactive manipulation (shifted)</td><td>0/30</td><td>0/30</td><td>14/30</td><td>6/30</td><td>3/30</td><td>3/30</td><td>1/30</td></tr></table>

To that end, we examine model selection across LoRA adaptation rank and action chunking. For each run, we select the checkpoint with the lowest validation loss computed on the held-out split from which the ID test trajectories are drawn; selection is based on the loss only, not on plan success. The LoRA rank is then selected based on success on a validation set of 30 shifted-task scenes, which is disjoint from the 30 shifted-task test scenes reported in Table V, to avoid leakage between model selection and evaluation.

First, we analyze the relationship between model success and adaptation rank across both ID and shifted trajectories. For the ID task families, performance improves up to $r = 6 4$ and saturates thereafter. For example, the $r \ = \ 5 1 2$ model resolves three of the eleven interactive-navigation failures made by the selected $r \ = \ 1 2 8$ model, corresponding to an incorrect object selection, an incorrect geometric target, and an erroneous direct-navigation decision. In contrast, shifted-task performance does not improve monotonically with increasing rank, peaking at $r \ = \ 1 2 8$ with 14/30 successful trajectories and decreasing for both lower and higher ranks. Composition still emerges at higher ranks (11/30 and 10/30 at $r = 2 5 6$ and $r = 5 1 2 )$ , but most remaining shifted-task trajectories revert to the object-manipulation sequence seen during training, whereas lower ranks additionally suffer from faulty geometric targets and erroneous output format.

These results suggest a possible trade-off between fitting the ID task distribution and the compositional capability to the shifted task structure, although experiments across additional random seeds would be required to determine whether this behavior is systematic.

The model trained without action chunking achieves competitive performance on the ID trajectories; however, its performance drops sharply on the shifted interactive-manipulation task, where failures correspond to treating the scene as the familiar object-manipulation case and failing to account for the blocking object. We hypothesize that predicting the complete remaining action sequence provides a training signal that helps the planner stitch together skills observed during training into a novel composition. The same benefit is not observed at $r = 5 1 2 .$ , where both AC and No-AC achieve 3/30, indicating that the effect also depends on the adaptation configuration.

## C. Comparison with a Task-Specific Transformer

This section complements the comparison in Section IV-C2. Results are reported in Table V.

The task-specific Transformer uses a causal Transformer backbone with the same architectural components as Qwen2 (grouped-query attention, rotary position embeddings, RM-SNorm, and SwiGLU feed-forward layers), with a hidden dimension of 512, 8 layers, 8 attention heads with 2 K/V heads, and a feed-forward dimension of 3072, but no pretrained weights and no text tokenizer. Instead, each semantic quantity is mapped directly into the hidden space and summed with a learned role embedding. The input prefix consists of the task (skill, object, and goal x, y, ψ), the robot pose, one token per scene object, and the previously executed skill. Each object token is the sum of embeddings of its identifier, type, mobility flag, size, and pose. Input geometry is continuous, with yaw encoded as (sin ψ, cos ψ); only the output space is discretized. Each predicted action is represented by five tokens, generated in the order skill, object, x, y, ψ, and a separate continuation head predicts whether another action follows, which determines the plan length. Position and yaw targets are quantized at resolutions of 1 [cm] and 1 [<sup>◦</sup>], resulting in a comparable number of output classes across the geometric variables, and are trained with a cross-entropy loss. All outputs are decoded greedily.

Object pointer: For each predicted action $j ,$ if $s _ { j } ~ = ~ n$ (navigation), no object is selected $( i _ { j } = - 1 )$ and no pointer loss is applied. If $s _ { i } = m$ (manipulation), the hidden state of the skill token, $\mathbf { h } _ { j } ^ { \mathrm { s k i l l } }$ , is projected to a query, and the hidden state of each scene-object slot o, $\mathbf { h } _ { o } ^ { \mathrm { o b j } }$ , to a key:

$$
{ \bf q } _ { j } = { \bf W } _ { q } { \bf h } _ { j } ^ { \mathrm { s k i l l } } , \qquad { \bf k } _ { o } = { \bf W } _ { k } { \bf h } _ { o } ^ { \mathrm { o b j } } .\tag{10}
$$

The alignment between the query and each key is scored by a scaled dot product, normalized over the set O of valid object slots (padding slots are masked):

$$
\alpha _ { j o } = \frac { \mathbf { q } _ { j } ^ { \top } \mathbf { k } _ { o } } { \sqrt { d } } , \qquad p ( i _ { j } = o ) = \frac { \exp ( \alpha _ { j o } ) } { \sum _ { o ^ { \prime } \in \mathcal { O } } \exp ( \alpha _ { j o ^ { \prime } } ) } ,\tag{11}
$$

where $d = 5 1 2 .$ The pointer is trained with a cross-entropy loss over manipulation actions,

$$
\mathcal { L } _ { \mathrm { o b j } } = - \sum _ { j : s _ { j } = m } \log p ( i _ { j } = i _ { j } ^ { \star } ) ,\tag{12}
$$

where $i _ { j } ^ { \star }$ is the ground-truth object, and the object with the highest probability is selected at inference.

On interactive navigation, the Transformer performs closest to our planner (101/120 vs. 109/120), with failures mainly associated with incorrect manipulation targets. On long-horizon interactive navigation (22/30), failures arise from immediately manipulating the blocking object instead of first navigating to it, i.e., an incorrect skill composition. The largest difference is observed on object manipulation (3/10): the model predicts the expected N → M sequence, but the navigation target places the robot either too far from the object or close to the final task goal. One possible explanation is the comparatively limited amount of object-manipulation data in the training set. On the shifted task, the Transformer generates the intended M → N → M composition in all 30 test trajectories (along with the correct object to manipulate) but completes none, with the same failure mode: after the blocker is manipulated, the navigation target directs the robot toward the final task goal rather than to the target object.

The Transformer infers at 26.9 ± 6.2 ms per action, compared with 472.4±15.5 ms for our planner, owing to its smaller embedding dimension and specialized tokenization.

Results of additional training runs are reported in Table XV.

## D. Qwen Architecture Trained from Scratch

To isolate the effect of pretraining from the architecture, input representation, and target discretization of the taskspecific Transformer, we additionally train a model with the Qwen2.5 architecture and tokenizer from random initialization, using the same text-based inputs, outputs, and data split as our planner. Given the dataset size, we use a reduced configuration (hidden size 768, 10 layers, 12 attention heads with 2 K/V heads, feed-forward dimension 3072; approximately 201M parameters), trained for 8 epochs with a learning rate of 10<sup>−4</sup> and otherwise the same training hyperparameters as our planner. The configuration was selected by validation loss among the tested configurations. Table XIV compares it with our planner and the task-specific Transformer.

On the ID task families, the model trained from scratch falls well behind our planner on interactive navigation (63/120 vs. 109/120) and long-horizon interactive navigation (23/30 vs. 29/30). On the shifted task, it does not produce the correct composition in any of the 30 trajectories; instead, it manipulates the blocking object and then navigates to the manipulation goal, following the interactive-navigation structure seen during training.

This gap is expected; without pretraining, the model must learn the text-based input and output format from limited data, without prior knowledge of the vocabulary it operates on. The comparison is therefore intended to isolate the contribution of pretraining for a text-based planner rather than to serve as a competitive baseline.

TABLE XIV: Comparison of our planner with the two models trained from scratch, represented as (# of successes)/(# of trajectories). For the shifted task, the last row reports the number of trajectories with the correct skill composition.

<table><tr><td></td><td>Ours</td><td>Qwen (scratch)</td><td>Transformer (scratch)</td></tr><tr><td>Pretrained</td><td></td><td></td><td></td></tr><tr><td>Input/output</td><td>Text</td><td>Text</td><td>Structured</td></tr><tr><td>Direct navigation</td><td>38/40</td><td>40/40</td><td>35/40</td></tr><tr><td>Object manipulation</td><td>10/10</td><td>8/10</td><td>3/10</td></tr><tr><td>Interactive navigation</td><td>109/120</td><td>63/120</td><td>101/120</td></tr><tr><td>Long-horizon interactive nav.</td><td>29/30</td><td>23/30</td><td>22/30</td></tr><tr><td>Interactive manipulation (shifted)</td><td>12/30</td><td>0/30</td><td>0/30</td></tr><tr><td>correct composition</td><td>24/30</td><td>0/30</td><td>30/30</td></tr></table>

## E. Additional Training Seeds

To assess sensitivity to training stochasticity, we train our planner (r = 128) and the task-specific Transformer with two additional seeds. All runs use the same trajectory split; only weight initialization, data order, and dropout differ. Table XV reports the results of each run on the same evaluation trajectories as Table V.

Success varies moderately across runs, whereas the correct skill composition is observed consistently in every run.

TABLE XV: Planner results per training run, represented as (# of successes)/(# of trajectories). Runs differ only in the training seed; Run 1 corresponds to the models reported in Table V. For the shifted task, the last row reports the number of trajectories with the correct skill composition.

<table><tr><td></td><td>Ours</td><td></td><td></td><td colspan="3">Transformer (scratch)</td></tr><tr><td>Task Family</td><td>Run 1</td><td>Run 2</td><td>Run 3</td><td>Run 1</td><td>Run 2</td><td>Run 3</td></tr><tr><td>Direct navigation</td><td>38/40</td><td>40/40</td><td>40/40</td><td>35/40</td><td>34/40</td><td>33/40</td></tr><tr><td>Object manipulation</td><td>10/10</td><td>10/10</td><td>10/10</td><td>3/10</td><td>4/10</td><td>5/10</td></tr><tr><td>Interactive navigation</td><td>109/120</td><td>104/120</td><td>103/120</td><td>101/120</td><td>97/120</td><td>91/120</td></tr><tr><td>Long-horizon interactive nav.</td><td>29/30</td><td>30/30</td><td>29/30</td><td>22/30</td><td>23/30</td><td>22/30</td></tr><tr><td>Interactive manipulation (shifted)</td><td>12/30</td><td>9/30</td><td>10/30</td><td>0/30</td><td>0/30</td><td>2/30</td></tr><tr><td>correct composition</td><td>24/30</td><td>16/30</td><td>20/30</td><td>30/30</td><td>30/30</td><td>27/30</td></tr></table>