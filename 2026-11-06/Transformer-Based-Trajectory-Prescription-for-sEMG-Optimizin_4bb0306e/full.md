# Transformer-Based Trajectory Prescription for sEMG-Optimizing Robot-Mediated Rehabilitation

Gabriel Parra<sup>1,2</sup>, Kimia Khoshnami<sup>1</sup>, Daniel S. Brown<sup>2</sup>, and Laura A. Hallock<sup>1,2</sup> <sup>1</sup>Department of Mechanical Engineering and <sup>2</sup>Kahlert School of Computing, University of Utah {u1289824,kimia.khoshnami,daniel.s.brown,laura.hallock}@utah.edu

## Abstract:

Robot-mediated rehabilitation has the potential to deliver high-intensity, individualized therapy, but existing systems (which have not consistently improved outcomes) typically prescribe kinematic trajectories rather than explicitly targeting the neuromotor activations thought to support recovery. Prescribing motions that elicit desired activity is challenging because muscle activity is hysteretic and stochastic, noninvasive measurement methods (e.g., surface electromyography, or sEMG) are noisy and sensor-location-dependent, and interventional datasets are costly to collect, inhibiting model development. To address this gap, and ultimately enable activation-optimizing rehabilitation robot behavior, we present a personalizable transformer-based model mapping individuals’ muscle activations (as measured via sEMG) to prescribed contact force trajectories when interacting with an end-effector rehabilitation robot. The model uses a transformer encoder– decoder leveraging rotary positional embeddings (RoPE) to generate a latent representation of trajectory-relevant and user-specific neuromotor factors, and augments trajectory prediction with flow matching to produce a distribution over plausible prescriptions biased toward the training distribution. Using data from 16 healthy participants completing 15 isometric trajectory tracking tasks, we evaluate generalization across held-out tasks and users, as well as the feasibility of userspecific fine-tuning within a therapy-session timescale. The model achieves mean trajectory prediction errors of <8% on held out tasks and users, with effectively instantaneous deterministic inference and sub-second sampling of trajectory distributions. User-specific fine-tuning further reduces error using only minutes of additional data collection and training. An initial test using the model during live rehabilitation tasks demonstrates that, on average, participants reach the target neuromotor activations with just 15% error accumulated throughout the task. These results provide a proof of concept for data-driven, activation-targeted trajectory prescription in robot-mediated rehabilitation and establish a foundation for future validation in clinical populations. To enable further development by scientific and user communities, data and code have been open-sourced as part of the Open-RobotRehab project on SimTK (simtk.org/projects/openrobotrehab).

Keywords: personalized rehabilitation, sequence modeling, human–robot interaction

## 1 Introduction

Robot-mediated therapy is a promising treatment for neuromotor injury (e.g., stroke), but existing systems (which largely rely on impedance-style error correction to desired kinematic trajectories) have not yet demonstrated meaningful improvement in rehabilitation outcomes over standard of care [1]. Neuroscientists posit that improving such outcomes requires systems to reinforce healthy neuromotor pathways — i.e., induce healthy patterns of underlying muscle activation, not just guide the user through desired kinematic motions [2, 3].

An ideal robotic rehabilitation system (of the type illustrated in Figure 1) would allow a new patient to arrive at a clinic, interact with the robot briefly to calibrate the system, then perform a set of therapy tasks that induce healthy activations for that individual. (Note that what constitutes “healthy” motion is an active area of investigation [4, 5]; in this work, we consider this to be a therapist- or expert-specified time series of muscle activations.) However, there do not yet exist models relating prescribed therapy activities (i.e., trajectories) to neuromotor activations of a given individual: even the healthy human neuromotor system exhibits complex (hysteretic and probabilistic) behavior, noninvasive measurement methods (e.g., surface electromyography, or sEMG) are noisy and dependent on sensor location, and large-scale data of patient populations are prohibitively expensive to collect, inhibiting both physics-based and data-driven model development.

In this work, we leverage the robust modeling capabilities of transformers [6] and flow matching [7] to construct a trajectory prescription model with the following properties: (1) given a target sEMGmeasured time series of muscle activations, predict the required target trajectory to induce these activations; (2) require minimal per-user calibration accomplishable within a single therapy session (i.e., taking only a few minutes); (3) once calibrated, predict trajectories in a sufficiently short time (i.e., a few seconds maximum) to enable a therapeutically meaningful number of repetitions and distinct motions within a session; and (4) provide well-calibrated uncertainty modeling to flag outof-distribution target muscle activations, supporting decisions about when prescribed trajectories should (and should not) be trusted. Specifically, we present the following contributions:

• Construction and training of a flow-matching-augmented, transformer-based trajectory prescription model mapping desired sEMG-measured muscle activations of the arm to the associated end-effector (hand) force target trajectory, quickly fine-tunable to new users.

• Demonstration in an exemplar cohort of 16 healthy users performing 15 trajectory tracking tasks that our model achieves an average of <8% trajectory prediction error on held-out users and tasks (with the bulk of that error attributable to the conservative, training-distribution-seeking behavior of the flow matching algorithm), and that user-specific calibration can substantially reduce that error within a few minutes’ data collection and training with quantifiable time–performance tradeoffs.

• Deployment of the model during live rehabilitation tasks for 5 healthy users, demonstrating an average task-accumulated error between model-prescribed and sEMG-measured user neuromotor activations of 15%.

• The open-source release of all data and modeling code as part of the OpenRobotRehab project on SimTK (simtk.org/projects/openrobotrehab), enabling both additional insights on neuromotor behavior and further model augmentation and population-specific refinement by the wider rehabilitation and machine learning communities.

We present this work as a first-of-its-kind, proof-of-concept system: our preliminary cohort establishes the feasibility of activation-targeted trajectory prescription, laying the foundation for expansions toward numerous clinical applications (e.g., post-injury muscle-specific strength training, post-stroke muscle coordination training).

## 2 Related Work

Principles of experience-dependent plasticity and motor learning suggest that effective neuromotor therapy should reinforce healthy patterns of neuromotor engagement — i.e., time series muscle activations — rather than merely guiding the limb through desired motions [2, 3]. While even healthy motor behavior is heterogeneous [5], properties of desirable and undesirable activation patterns can be characterized across a wide range of therapy applications: in stroke, for example, a therapeutic target is often to move outside of abnormal muscle synergies and avoid excessive co-contraction [8], and in sports medicine, increasing activation of under-recruited muscles and improving neuromotor coordination can be used to avoid injury [9]. At the same time, clinical rehabilitation rarely incorporates these insights directly, and remains largely empirical: in practice, therapists select movements, assistance, resistance, and perturbations based on experience, without a systematic way to choose tasks that induce a desired pattern of neuromotor activation.

Robot-mediated rehabilitation, alongside neuromechanical sensing of muscle activations, kinematics, and contact forces, is a promising paradigm to systematize the prescription of therapy tasks to best induce such activations. In the literature, if not in practice, sEMG biofeedback has long been used to expose muscle activity to patients and clinicians, supporting efforts to increase weak muscle recruitment, reduce inappropriate activation, or encourage task-oriented retraining [10, 11]. Recent work (e.g., Lee et al. [12]) has begun to connect such muscle-level targets to prescribed task-level actions, demonstrating significant therapeutic effects in pilot cohorts that support the feasibility of such an approach. At the same time, these systems typically remain reliant on explicit pathology- or task-specific models, inhibiting translation to new populations and motions. Those that aim to model task-agnostic behavior at the neuromuscular level generally employ neuromusculoskeletal simulators (e.g., OpenSim [13], MyoSuite [14]) that are expensive to calibrate and may not generalize across individuals, sensor placements, impairments, or contact conditions.

The heterogeneity and stochastic nature of neuromotor behavior makes machine learning a natural approach to close this modeling gap, and ML-based models have been widely applied to sEMG for gesture recognition, prosthetic and orthotic control, intent inference, torque estimation, and human–robot interaction [15, 16, 17], though never (to our knowledge) to the problem of target trajectory generation during robot-mediated rehabilitation. To construct such a model, we employ a transformer encoder–decoder architecture [6] (detailed in Section 3.3), motivated by its strong performance in related sEMG inference tasks (e.g., [18, 19, 20]). This architectural choice is wellsuited to sEMG, which is a multichannel temporal signal whose interpretation depends on activation history, co-activation patterns, and task phase. Self-attention enables the model to capture dependencies across an entire time window without requiring hand-engineered temporal features or strictly recurrent state updates [6], and transformer-based representation learning has proven effective for multivariate time-series modeling [21]. We use rotary positional embeddings (RoPE) [22] to preserve temporal ordering while allowing attention to depend explicitly on relative time offsets, which is important when mapping activation trajectories rather than isolated samples. In designing our architecture, we also acknowledge that the desired inverse map is not a deterministic function of observed sEMG alone: neuromechanical redundancy, user-specific pursuit strategies, electrode placement, fatigue, and impairment introduce unobserved factors that can change the relationship between a displayed trajectory and the resulting muscle activity. This motivates a compact latentvariable representation that separates target-associated trajectory information from latent neuromotor behavior, following the broader use of continuous latent representations for scalable inference in partially observed systems [23] and aligning with neuromechanical understanding of redundant muscle activation and individual variation [4, 24]. Finally, because multiple force or velocity trajectories may plausibly induce similar desired activation patterns, we model a conditional distribution over latent trajectory codes rather than only a point estimate. Flow matching provides an efficient simulation-free objective for learning continuous generative flows [7], including conditional variants suitable for sampling from learned conditional distributions [25]; in our setting, these samples can be decoded into a distribution over prescribed trajectories, providing an uncertainty estimate that can inform when additional user-specific calibration may be needed. Flow matching also tends to bias prescribed trajectories toward the distribution of the training data set — a useful (though not guaranteed) property, as out-of-distribution trajectories may be unsafe to execute on a robot. Section 3.3 describes how each of these design choices is realized in our architecture.

## 3 System Design

This section presents our trajectory prescription model (Section 3.3) for a robot-mediated rehabilitation platform (Section 3.1), trained and tested using data collected from 16 participants (Section 3.2).

## 3.1 Robot-Mediated Rehabilitation Platform

Figure 1 illustrates our end-effector robot-mediated rehabilitation platform, an extension of Anand et al.’s OpenRobotRehab platform [26].

![](images/c5649eb704d2bf7c975f45852f86c62f8b4976c379764ee6e2727fb22c021ee3.jpg)  
Figure 1: End-effector robot-mediated rehabilitation platform. Users exert forces on a LBR iiwa 14 R820 7-degree-of-freedom cobot (KUKA AG, Augsburg, Germany; a) via a custom handle attached to a SensONE 6-axis force torque sensor (Bota Systems AG, Zurich, Switzerland; b). Users’ forces along each positional axis¨ are mapped to the corresponding x/y/z velocity of an on-screen avatar ball (c), where the z axis into and out of the screen is represented by a decrease and increase in avatar size, respectively. Users are instructed to follow a target ball along a displayed trajectory. Force and game trajectory data (of both avatar and target) are collected at 1000 Hz and 60 Hz, respectively. During this process, two Trigno Quattro 4-channel sensor motes (d), controlled through a Trigno Base Station (Delsys, Inc., Natick, MA, USA) collect surface electromyography (sEMG) data at a rate of 2.22 kHz from 8 upper limb muscles (e). Electrode placements are as follows: anterior deltoid (AD), middle deltoid (MD), posterior deltoid (PD), and biceps brachii (BB), grounded at the acromion (G1); and triceps brachii (long head, TR), brachioradialis (BR), wrist flexors (FL), and wrist extensors (EX), grounded at the olecranon (G2). All data are collected and synchronized via Redis (MongoDB, Inc., New York, NY, USA). Full implementation details are included with code release.

## 3.2 Data Collection

Data were collected from 16 healthy participants (P01–P16; 5 female, 11 male; age: $2 8 . 4 \pm 7 . 0$ (mean ± standard deviation, min 19, max 47); 5 left-handed, 11 right-handed), none of whom had prior experience performing this study’s tasks. Three additional participants (P17, P18, P19) did not complete initial data collection but participated in preliminary closed-loop trajectory prescription tests discussed in Section 4.1.<sup>1</sup>

Each participant executed 15 isometric force trajectories (see Figure 1 (d)), all with their (sEMGinstrumented) right arm regardless of handedness, designed to traverse the user’s $x / y / z$ force workspace, in the order enumerated in Table 1<sup>2</sup>. Tasks included single axis motions (T01–T12) and stochastically generated trajectories surrounding the x and y axes generated via a smoothed Brownian motion (T14–T15).

Trajectories were performed at 2 distinct force levels and both with and without non-targeted axes “locked” for single axis tasks (e.g., T01 and T04 represented the same x-axis traversal task, but for T01, the avatar moved in all dimensions such that the user must maintain y and z force values at 0 for accurate task completion, while for T04, the avatar was locked to the x-axis such that these y and z force values could vary freely without impacting task performance).<sup>3</sup> Visuals for each trajectory are available in Appendix D.

Table 1: Trajectory Tracking Tasks
<table><tr><td rowspan=1 colspan=1>TaskNo.</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>Otheraxeslocked?</td><td rowspan=1 colspan=1>Force required forunit velocity $x / y / z$ motion (N)</td><td rowspan=1 colspan=1>Repe-ti-tions</td><td rowspan=1 colspan=1>Dura-tion(s)</td><td rowspan=1 colspan=1>Trajectorydis-played?</td><td rowspan=1 colspan=1>Trai-ningset?</td><td rowspan=1 colspan=1>Testset?</td></tr><tr><td rowspan=1 colspan=1>T01</td><td rowspan=1 colspan=1>x-axis</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>15/15/36</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>√</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T02</td><td rowspan=1 colspan=1>y-axis</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>15/15/36</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=2 colspan=1>VV</td><td rowspan=7 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T03</td><td rowspan=1 colspan=1>z-axis</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>15/15/36</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>T04</td><td rowspan=1 colspan=1>x-axis</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>15/-/-</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>V</td></tr><tr><td rowspan=1 colspan=1>T05</td><td rowspan=1 colspan=1>y-axis</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>-/15/-</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>V</td></tr><tr><td rowspan=1 colspan=1>T06</td><td rowspan=1 colspan=1>z-axis</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>-/-/36</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>V</td></tr><tr><td rowspan=1 colspan=1>T07</td><td rowspan=1 colspan=1>x-axis</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>30/30/72</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>V</td></tr><tr><td rowspan=1 colspan=1>T08</td><td rowspan=1 colspan=1>y-axis</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>30/30/72</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>V</td></tr><tr><td rowspan=1 colspan=1>T09</td><td rowspan=1 colspan=1>z-axis</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>30/30/72</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T10</td><td rowspan=1 colspan=1>x-axis</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>30/-/-</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T11</td><td rowspan=1 colspan=1>y-axis</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>-/30/-</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>V</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>T12</td><td rowspan=1 colspan=1>z-axis</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>-/-/72</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>V</td></tr><tr><td rowspan=1 colspan=1>T14</td><td rowspan=1 colspan=1>stochastic-x</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>20/20/48</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>√</td></tr><tr><td rowspan=1 colspan=1>T15</td><td rowspan=1 colspan=1>stochastic-y</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>20/20/48</td><td rowspan=1 colspan=1>N/A</td><td rowspan=1 colspan=1>20</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>V</td></tr></table>

## 3.3 Trajectory Prescription Model

Recall that we aim to prescribe a target trajectory to display to a given user to induce a specified pattern of sEMG-measured muscle activations. Mathematically, this means that we seek to identify some function mapping desired sEMG muscle activation time series $\mathbf { a } = \{ \mathbf { a } _ { t } \in \mathbb { R } ^ { 8 } \} _ { t = 1 } ^ { T }$ over $T$ time steps to target $x / y / z$ velocity trajectory $\mathbf { v } _ { \mathrm { t a r g e t } } = \{ \mathbf { v } _ { \mathrm { t a r g e t } , t } \in \mathbb { R } ^ { 3 } \} _ { t = 1 } ^ { T }$ that will induce these activations. We know, however, that target trajectory is not determined by sEMG content alone: even if a user completes a trajectory perfectly, neuromechanical redundancy [24] means that a user may engage a continuum of possible muscles, and in practice, users will complete the tracking task with varying skill levels, and employ varying pursuit strategies [5]. These strategies vary by user, as do noise characteristics of the sEMG signal due to variations in sensor placement, attenuation due to adipose tissue, and fatigue. We therefore model the activation–target relationship $f$ as

$$
\mathbf { a } = f ( \mathbf { v } _ { \mathrm { t a r g e t } } , U )
$$

for some unknown latent variable $U$ encompassing this variation in neuromechanical behavior. Our task of trajectory prescription can then be written as: model some function $g \approx f ^ { - 1 }$ , such that we can prescribe some trajectory $\hat { \mathbf { v } } _ { \mathrm { t a r g e t } }$ via

$$
\hat { \mathbf { v } } _ { \mathrm { t a r g e t } } = g ( \mathbf { a } ) = g ( f ( \mathbf { v } _ { \mathrm { t a r g e t } } , U ) )
$$

given some desired activation a.

In addition to accounting for this unknown latent variable $U ,$ the function g must be sufficiently expressive to account for the time-dependent nature of the sEMG signal. (An instantaneous sEMG sample is meaningless alone, and a function of both prior and future sEMG data.) Practically, any model for $g$ must also admit rapid retraining to accommodate out-of-distribution users, and, ideally, provide a measure of uncertainty on trajectory predictions to inform decision making on when model updates are required.

Consistent with these design constraints, we present the following transformer–autoencoder architecture model for $^ { g , }$ illustrated in Figure 2. Activations a are encoded via a Rotary Position Em bedding (RoPE) [22] transformer into a latent representation $\textbf { z } \in \mathbb { R } ^ { 6 4 }$ consisting of concatenated vectors $\mathbf { z _ { v } } \in \mathbb { R } ^ { 3 2 }$ and $\mathbf { z } _ { U } \in \mathbb { R } ^ { 3 2 }$ compressing information corresponding to the target trajectory and unknown neuromechanical behavior, respectively. This partition mirrors the structure of $f$ above (and the motivation in Section 2): the activations a user generates are a function of both the prescribed trajectory and that user’s specific motor strategy, so we allocate one latent to each factor.

Separate transformer decoders then reconstruct the prescribed target trajectory $\hat { \mathbf { v } } _ { \mathrm { t a r g e t } }$ from $\mathbf { z _ { v } }$ (the primary model output) and estimated associated activations ˆa from $\mathbf { z _ { v } }$ and $\mathbf { z } _ { U }$ (autoencoder only used in training). To model uncertainty over $\hat { \mathbf { v } } _ { \mathrm { t a r g e t } }$ , a flow-matching diffusion model [7] is also trained to generate conditional probability distribution $p ( \mathbf { z } _ { \mathbf { v } } | \mathbf { z } _ { U } )$ An arbitrary number of samples $\mathbf { Z _ { v } }$ can then be generated from this distribution and passed through the target trajectory decoder to generate a distribution $\hat { \mathbf { V } } _ { \mathrm { t a r g e t } }$ over $\hat { \mathbf { v } } _ { \mathrm { t a r g e t } }$ . In addition to providing a measure of prediction uncertainty, this flow matching formulation tends to bias samples of $\hat { \mathbf { V } } _ { \mathrm { t a r g e t } }$ toward the distribution of the training set, with uncertainty generally rising for inputs sparsely represented or absent in the

![](images/c770fe212043869871a944ba720eb5a0212bb3091e989c0b7c7b576d3a312403.jpg)  
Figure 2: Architecture of trajectory prescription model. Latent representation z of both target-associated information $\bf ( z _ { v } )$ and unknown neuromotor behavior $\left( { { \bf { z } } _ { U } } \right)$ is generated using a RoPE [22] transformer encoder (blue flow), then decoded separately to generate prescribed trajectory $\hat { \mathbf { v } } _ { \mathrm { t a r g e t } }$ (green flow) and estimated activation ˆa (turquoise flow). A flow-matching diffusion model [7] is also trained to generate a conditional distribution of $\mathbf { z _ { v } }$ given z<sub>U</sub> whose samples can be passed through the target trajectory decoder to generate a distribution over $\hat { \mathbf { v } } _ { \mathrm { t a r g e t } }$ (orange flow).

training data: while this may curtail generalization outside this distribution, we consider it a usefu property for our application, as prescribed trajectories far outside of distribution may be unsafe to execute. We emphasize, however, that this behavior is not a theoretical safety guarantee. Instead, we envision flow matching uncertainty as a decision-support tool within the therapy workflow: if the model exhibits high uncertainty for a requested set of target muscle activations, the system (or supervising therapist) should defer targeting those activations until more information about the user has been gathered, e.g., via the fine-tuning procedure of Section 4.2.

The system is then jointly trained to minimize loss function

$$
\mathcal { L } = \lambda _ { 1 } \left\| \mathbf { v _ { t a r g e t } } - \hat { \mathbf { v } } _ { \mathrm { t a r g e t } } \right\| _ { 2 } ^ { 2 } + \lambda _ { 2 } \left\| \mathbf { a } - \hat { \mathbf { a } } \right\| _ { 2 } ^ { 2 } + \lambda _ { 3 } \left\| \mathbf { z _ { v } } - \mathbf { Z _ { v } } \right\| _ { 2 } ^ { 2 }\tag{1}
$$

such that primary trajectory prediction $\hat { \mathbf { v } } _ { \mathrm { t a r g e t } }$ is regularized by activation decoder and flowmatching terms to encourage a structured latent representation and uncertainty-aware trajectory prescription, respectively.

## 4 Experiments

Motivated by our intended use case — a new user briefly interacts with the platform, after which the system prescribes trajectories based on expert-specified desired sEMG patterns — we present the following evaluations of model performance in terms of both prescribed trajectory accuracy and required training time.

## 4.1 Baseline Model Training & Evaluation

We first characterize model performance absent any user-specific retraining.

Training and evaluation splits. The baseline model was trained using data from participants P01– P10, tasks T01–T12, as noted in Table 1, and evaluated on unseen tasks for known users (P01–P10, T14–T15), known tasks for unseen users (P11–P16, T01–T12), and unseen tasks for unseen users (P11–P16, T14–T15). Note that this division results in a training set that contains highly stereotyped, single-axis motions, and unseen-task test sets that are stochastic in nature and thus contain trajectory regions both in and out of distribution of the test set, allowing exploration of performance across these conditions. The final data set contained 6480 training windows and 4562 test windows.

Model training. Our (Figure 2) model was trained to minimize the loss function in Equation 1. Our implementation used PyTorch with AdamW using a learning rate of 0.001, batch size of 64, and was trained for 100 epochs. Each training example consisted of a 10 s window (generated with a 1 s stride across each 60 s trial) of processed sEMG <sup>4</sup> activation a paired with corresponding target trajectory $\mathbf { v } _ { \mathrm { t a r g e t } }$ . Windows were shuffled and dropped (dropout $\scriptstyle { p = 0 . 1 ) }$ during training. Flow matching time steps were sampled from a uniform distribution, with base noise sampled from the standard normal distribution and 10 samples drawn per training example. Hyperparameter selection is documented in Appendix F. Training was performed on a NVIDIA RTX 4070 with 12 GB of VRAM. A full baseline training run required ≈1.5 h. At inference time, generating one prescribed trajectory was effectively instantaneous (5 ms), and generating 250 flow matching samples required <1 s.

Trajectory prediction performance metrics. We quantify model performance as mean squared prediction error for both our deterministic transformer-decoded target trajectory $( \mathrm { M S E _ { T D } ) }$ and our flow matching distribution model $( \mathrm { M S E _ { F M } } )$ , defined as

$$
\begin{array} { r } { \mathrm { M S E } _ { \mathrm { T D } } = \frac { 1 } { n _ { \mathrm { t r i a l s } } n _ { \mathrm { w i n d o w s } } } \displaystyle \sum _ { i = 1 } ^ { n _ { \mathrm { t r i a l s } } n _ { \mathrm { w i n d o w s } } } \left. \mathbf { v } _ { \mathrm { t a r g e t } } - \hat { \mathbf { v } } _ { \mathrm { t a r g e t } } \right. _ { 2 _ { i , j } } ^ { 2 } } \\ { \mathrm { M S E } _ { \mathrm { F M } } = \frac { 1 } { n _ { \mathrm { t r i a l s } } n _ { \mathrm { w i n d o w s } } } \displaystyle \sum _ { i = 1 } ^ { n _ { \mathrm { t r i a l s } } n _ { \mathrm { w i n d o w s } } } \left. \mathbf { v } _ { \mathrm { t a r g e t } } - \mathbb { E } [ \hat { \mathbf { V } } _ { \mathrm { t a r g e t } } ] \right. _ { 2 _ { i , j } } ^ { 2 } } \end{array}
$$

where $\mathbb { E } [ \hat { \mathbf { V } } _ { \mathrm { t a r g e t } } ]$ is computed as the sample mean across 25 trajectories per activation window, each decoded from an independent flow matching sample of $\mathbf { Z _ { v } }$ . (See code release for implementation details.) We selected 25 samples per window to generate a variety of candidate trajectories while remaining within our GPU’s memory limits, enabling rigorous evaluation across all test windows.

Trajectory prediction performance results & ablations. Table 2 aggregates prediction performance of the “Full” model across test sets, alongside naive baselines (MLP and LSTM) and ablated models, and Figure 3 illustrates exemplar Full model performance. A complete description of all tested architectures can be found in Appendix G. The Full model reliably achieves ≈8.5% deterministic transformer-decoded error, with flow matching providing a performance boost of ≈1–2%, across all test sets, demonstrating superior generalization performance as compared with all ablations. The naive MLP baseline is uniformly less performant than the Full model on all test sets. The LSTM baseline achieves deterministic error comparable to our Full model’s $\mathrm { M S E _ { F M } }$ on new tasks performed by known users and on new tasks performed by new users, but generalizes poorly to new users performing known tasks, and also performs poorly on the training set (a meaningful deficiency, as rehabilitation often consists of patients returning to complete similar tasks). For almost all models and ablations, standard deviation is substantially higher than mean error, and Figure 3 presents exemplar data illustrating the source of this variance in our Full model: portions of the trajectory within the distribution of the test set exhibit substantially higher prediction accuracy than out-of-distribution areas, which flow matching tends to pull toward the training distribution.

Preliminary trajectory prescription performance. To show proof of concept of intended model usage — closed-loop prescription of trajectories to induce specified muscle activations — we selected an exemplar set of muscle activations from an arbitrary system user and task (P07, T08) and generated a new trajectory from those muscle activations using our most performant (Full) model (expected trajectory from 250 samples of the flow-matching model; Figure 4, Left). Five participants (P08 and P13 from the original cohort, and P17–P19 new) were then instructed to complete this new trajectory. Participants were instructed to follow the trajectory as closely as possible, and were granted multiple chances to repeat the trajectory to fix their mistakes if requested. We only report on these participants’ final (not best) trajectory repetition. Figure 4 $( R i g h t )$ shows exemplar prescription performance for new participant P17, illustrating the reliable qualitative similarity between target and user-generated muscle activations.<sup>5</sup> Quantitatively, activation error (measured via RMSE) was consistent across all 5 participants, with mean 0.155 and standard deviation 0.026.

Table 2: Trajectory Prediction Performance of Baselines, Ablated Models, and Full Model
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Model</td><td colspan="3">Components</td><td colspan="4">Error (mean ± standard deviation)</td></tr><tr><td>Traj. Dec.</td><td>Act. Dec.</td><td>Flow Mat.</td><td>Training Set</td><td>New Tasks</td><td>New Users</td><td>New Tasks &amp; Users</td></tr><tr><td rowspan="6"> $\mathrm { M S E } _ { \mathrm { T D } }$ </td><td>Naive MLP</td><td>√</td><td></td><td></td><td>0.106 ± 0.273</td><td>0.097 ± 0.233</td><td>0.238 ± 0.462</td><td>0.106 ± 0.204</td></tr><tr><td>LSTM</td><td>√</td><td></td><td></td><td>0.155 ± 0.308</td><td>0.068 ± 0.143 0.108 ±</td><td>0.191 ± 0.354</td><td>0.063 ± 0.128</td></tr><tr><td>Minimal</td><td>√</td><td></td><td></td><td>0.016 ± 0.109</td><td>0.211</td><td>0.099 ± 0.256</td><td>0.115 ± 0.213</td></tr><tr><td>+ Act. Dec.</td><td>√</td><td>√</td><td></td><td>0.005 ± 0.074</td><td>0.117 ± 0.267</td><td>0.088 ± 0.252</td><td>0.110 ± 0.228</td></tr><tr><td>+ Flow Mat.</td><td>√</td><td></td><td>√</td><td>0.069 ± 0.216</td><td>0.115 ± 0.239</td><td>0.095 ± 0.255</td><td>0.106 ± 0.185</td></tr><tr><td>Full</td><td>V</td><td>√</td><td>V √</td><td>0.009 ± 0.110</td><td>0.084 ± 0.227 0.114 ±</td><td>0.086 ± 0.252</td><td>0.082 ± 0.211 0.106 ±</td></tr><tr><td>MSEFM</td><td>+ Flow Mat. Full</td><td>√ V</td><td>√</td><td>V</td><td>0.076 ± 0.213 0.010 ± 0.094</td><td>0.241 0.067 ± 0.190</td><td>0.098 ± 0.246 0.077 ± 0.211</td><td>0.196 0.062 ± 0.174</td></tr></table>

![](images/2ab6d9ecfec4e5f3b1373e8e975d8b8118984d69d89ee9955d9cd10cb6f4e92b.jpg)

![](images/f07bb3e02430f27ddf42995f2d200fa306f5976a89a8c3c13ed3a1056acabedf.jpg)

![](images/cc56efa89240341c19aebb125be5ada2fe1597c8066bc606f1d35280221616f3.jpg)

Figure 3: Exemplar trajectory prediction accuracy $\hat { \mathbf { V } } _ { \mathrm { t a r g e t } }$ distributions for unseen tasks and unseen users. Left: Known participant (P09) performing unknown task (T14). For regions within the distribution of tasks observed during training $( 0 . 3 \ : \bar { < } \ : t < 0 . 6 )$ , trajectory mean is accurately predicted. For regions outside this distribution $( t < 0 . 3 , t > 0 . 9 )$ , flow matching tends to pull predicted trajectories toward in-distribution regions, which are more likely to be safe. Some regions $( 0 . 6 \dot { ~ } < t { ~ } < 0 . 9 )$ exhibit a mix of this behavior across axes. Center: Unseen participant (P11) performing known task (T07). Trajectory predictions are accurate aside from artifacts (e.g., at t ≈ 0.15) due to the length of windows that were seen by the transformer at training time. Right: Unseen participant (P16) performing unseen task (T14). Prediction is qualitatively comparable to unseen tasks performed by known users, with slightly higher overall error rates.  
![](images/2cdd908fa99c01d9a8b82407d3f5ed7dd82a9b7d4da7ba315acce5efa7578126.jpg)  
Figure 4: When model-prescribed trajectory (left), designed to induce target muscle activations (right, top) is followed by new users, they generate qualitatively and qualitatively similar activations (right, bottom for exemplar new user P17).

Discussion. Our trained model’s rapid inference time affirms its suitability for real-time trajectory prescription applications in therapy settings, with solid prediction error for unseen tasks and users; desirable conservative, training-distribution-seeking behavior when encountering out-of-distribution data; and promising proof-of-concept performance inducing target activations through prescribed trajectories. Moving forward, we aim to characterize first, the prediction accuracy required to induce therapeutically meaningful muscle activations, and second, model performance in cohorts with specific neuromotor pathologies.

## 4.2 User-Specific Fine-Tuning & Evaluation

In clinical settings, patients are likely to exhibit out-of-distribution behavior; while flow matching biases our prescribed trajectories toward those observed in training (which are more likely to be safe, assuming trajectories executed by one user are safe for another), they may not be sufficiently expressive. We present the following method to calibrate our baseline model to new users and remark on the trade-offs between calibration time and model performance.

Fine-tuning procedures. To emulate the process of a new user entering a therapy session, performing a calibration task, and engaging with the fine-tuned model, we retrain our baseline model with one or more tasks greedily selected from the new user’s data set via an algorithm detailed in Appendix I. We evaluate loss L (as specified in Equation 1), an upper bound on expected trajectory prediction error, for retrained models incorporating as few as 1 and as many as 12 user-specific tasks, on all remaining test tasks, for each test user.

Fine-tuning performance & time requirements. Figure 5 illustrates the loss achieved for each test user after training with incrementing numbers of test tasks. Training performance improved for all users with the incorporation of increasing amounts of user-specific data, with variable effect sizes ranging from ≈2–12%. The time required to train all 12 models for each participant was ≈7 min, equally distributed across models, such that each generated model represents ≈35 s in training time. This means that the total calibration time, accounting for both additional data collection time and training, can be interpreted as $n _ { \mathrm { t r i a l s n e w } } \times \{ 6 0 , 2 0 \}$ + 35 s (for 60 and 20 s tasks, respectively).

![](images/1e0417624e6871eb8d294d03947b545a51e68052a7ce3ad9d4ec0c8e0116fc78.jpg)  
Figure 5: Training loss L upper-bounding trajectory prediction error after re-training using an incrementing number of each user’s trials via our greedy re-training algorithm. All users exhibit decreasing error with additional training, characterizing the trade-off between calibration time and prediction accuracy.

Discussion. Short training times (a few minutes in the context of a ≈30–60 min therapy appointment) and modest-to-substantial performance improvements in our pilot cohort demonstrate the fea sibility of meaningfully fine-tuning our model on clinical timelines.

## 5 Limitations

This paper presents the first proof-of-concept usage of our trajectory prescription model. Testing with target populations will be needed to establish therapeutic utility. Specific therapy tasks used in training and testing have also not been optimized; we hypothesize that performance could be substantially enhanced by training on more diverse trajectories rather than stereotyped motions along each axis at limited force levels, and determining how to select such trajectories optimally for most efficient training is a rich area for future exploration. Now that prescription feasibility is established, future work includes model training with larger, pathology-specific cohorts, including combination with a pathology- or user-specific residual model, and consideration of additional factors known to influence neuromotor behavior, such as handedness (5 of our 16 participants were left-handed, but all completed tasks with their right arm). The behavior of our flow matching algorithm near the edges of the training distribution has also only been examined qualitatively, and its uncertainty estimates should not be interpreted as a safety guarantee; for more robust safety in deployment, our model can be combined with multiple layers of safety guarantees, such as hardware joint/speed limits and control barrier function and/or reachability-based control layers. In addition, hyperparameter values were largely selected by intuition, and a more systematic hyperparameter search could likely also improve performance. Lastly, all interactions were isometric, and we aim to expand this model in the future to prescribe dynamic end effector trajectories.

## 6 Conclusions

We have demonstrated a first-of-its-kind activation-optimizing trajectory prescription model shown to generalize to unseen users and tasks, run fast enough for real-time therapy use, and be fine-tuned with brief user-specific calibration. These results support the feasibility of activation-targeted robotmediated rehabilitation, and we have open-sourced all data and code to facilitate such development.

## Acknowledgments

The authors thank Ajay Anand and Jono Jenkens for their contributions to the development of the system that enabled data collection for this study. We also thank the reviewers for their careful reading of the manuscript and their constructive, detailed feedback, which led to meaningful improvements, including the live demonstration of our model’s closed-loop capabilities and a more rigorous description of the limitations of our model.

## References

[1] J. M. Park, H. J. Park, S. Y. Yoon, Y. W. Kim, J. I. Shin, and S. C. Lee. Effects of robot-assisted therapy for upper limb rehabilitation after stroke: An umbrella review of systematic reviews. Stroke, 2024.

[2] J. A. Kleim and T. A. Jones. Principles of experience-dependent neural plasticity: implications for rehabilitation after brain damage. Journal of Speech, Language, and Hearing Research, 51 (1):S225–S239, 2008.

[3] M. Maier, B. R. Ballester, and P. F. Verschure. Principles of neurorehabilitation after stroke based on motor learning and brain plasticity mechanisms. Frontiers in Systems Neuroscience, 13:74, 2019.

[4] L. H. Ting, S. A. Chvatal, S. A. Safavynia, and J. Lucas McKay. Review and perspective: neuromechanical considerations for predicting muscle activation patterns for movement. International Journal for Numerical Methods in Biomedical Engineering, 28(10):1003–1014, 2012.

[5] A. Anand, G. Parra, C. A. Berghoff, and L. A. Hallock. A case study characterizing healthy & post-stroke neuromotor behavior during 6D upper-limb isometric gaming: Implications for design of end-effector rehabilitation robot interfaces. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2026. doi:10.1109/TNSRE.2026.3737040.

[6] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need, 2023. URL https://arxiv.org/abs/1706.03762.

[7] Y. Lipman, M. Havasi, P. Holderrieth, N. Shaul, M. Le, B. Karrer, R. T. Q. Chen, D. Lopez-Paz, H. Ben-Hamu, and I. Gat. Flow matching guide and code, 2024. URL https://arxiv. org/abs/2412.06264.

[8] L. Dipietro, H. I. Krebs, S. E. Fasoli, B. T. Volpe, J. Stein, C. Bever, and N. Hogan. Changing motor synergies in chronic stroke. Journal of Neurophysiology, 2007.

[9] L. Y. E. Griffin. Neuromuscular training and injury prevention in sports. Clinical Orthopaedics and Related Research, 409:53–60, 2003.

[10] H. Huang, S. L. Wolf, and J. He. Recent developments in biofeedback for neuromotor rehabilitation. Journal ofNeuroEngineering and Rehabilitation, 3(1):11, 2006.

[11] O. M. Giggins, U. M. Persson, and B. Caulfield. Biofeedback in rehabilitation. Journal of NeuroEngineering and Rehabilitation, 10(1):60, 2013.

[12] H. Lee, J.-H. Park, J.-H. Shin, J. Roh, and H.-S. Park. Muscle-to-action mapping for intuitive training of muscle synergies in post-stroke upper-limb rehabilitation. Journal of NeuroEngineering and Rehabilitation, 22(1):99, 2025.

[13] S. L. Delp, F. C. Anderson, A. S. Arnold, P. Loan, A. Habib, C. T. John, E. Guendelman, and D. G. Thelen. OpenSim: open-source software to create and analyze dynamic simulations of movement. IEEE Transactions on Biomedical Engineering, 54(11):1940–1950, 2007.

[14] V. Caggiano, H. Wang, G. Durandau, M. Sartori, and V. Kumar. MyoSuite–A contact-rich simulation suite for musculoskeletal motor control. arXiv preprint arXiv:2205.13600, 2022.

[15] P. Tamilvanan, V. K. Elumalai, and B. Elumalai. A comprehensive survey of machine learning algorithms in hand gesture recognition for myoelectric prosthetic control: Current trends, challenges and future directions. Archives of Computational Methods in Engineering, pages 1–30, 2025.

[16] M. Zheng, M. S. Crouch, and M. S. Eggleston. Surface electromyography as a natural human– machine interface: A review. IEEE Sensors Journal, 22(10):9198–9214, 2022.

[17] S. M. Sid’El Moctar, I. Rida, and S. Boudaoud. Comprehensive review of feature extraction techniques for semg signal classification: From handcrafted features to deep learning approaches. IRBM, 45(6):100866, 2024.

[18] J. Xu, R. Wang, S. Shang, A. Chen, L. Winterbottom, T.-L. Hsu, W. Chen, K. Ahmed, P. L. La Rotta, X. Zhu, et al. ChatEMG: Synthetic data generation to control a robotic hand orthosis for stroke. IEEE Robotics and Automation Letters, 10(2):907–914, 2024.

[19] Z. Wang, J. Yao, M. Xu, M. Jiang, and J. Su. Transformer-based network with temporal depthwise convolutions for sEMG recognition. Pattern Recognition, 145:109967, 2024.

[20] W. Zhang, T. Zhao, J. Zhang, and Y. Wang. LST-EMG-Net: Long short-term transformer feature fusion network for sEMG gesture recognition. Frontiers in Neurorobotics, 17:1127338, 2023.

[21] G. Zerveas, S. Jayaraman, D. Patel, A. Bhamidipaty, and C. Eickhoff. A transformer-based framework for multivariate time series representation learning. In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, pages 2114–2124, 2021.

[22] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu. RoFormer: Enhanced transformer with rotary position embedding, 2023. URL https://arxiv.org/abs/2104.09864.

[23] D. P. Kingma and M. Welling. Auto-encoding variational bayes, 2022. URL https://arxiv. org/abs/1312.6114.

[24] D. Stanev and K. Moustakas. Modeling musculoskeletal kinematic and dynamic redundancy using null space projection. PloS One, 14(1):e0209171, 2019.

[25] A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport, 2024. URL https://arxiv.org/abs/2302.00482.

[26] A. Anand, C. A. Berghoff\*, C. J. Wynn\*, E. C. Falconer\*, G. Parra\*, J. Jenkens\*, C. J. Thomson, W. C. Hamrick, J. A. George, and L. A. Hallock. An extensible platform for measurement and modification of muscle engagement during upper-limb robot-facilitated rehabilitation. In IEEE RAS/EMBS International Conference on Rehabilitation Robotics (ICORR), pages 1556– 1563. IEEE, 2025. \*Equal contribution. doi:10.1109/ICORR66766.2025.11062974.

## Appendices

## A Participant Demographics

Table 3 contains all participant demographic data. Note that all participants completed tasks with their right arm regardless of handedness.

Table 3: Participant Demographics
<table><tr><td>ID</td><td>Age</td><td>Gender</td><td>Handedness</td></tr><tr><td>P01</td><td>21</td><td>M</td><td>Left</td></tr><tr><td>P02</td><td>24</td><td>M</td><td>Right</td></tr><tr><td>P03</td><td>24</td><td>M</td><td>Right</td></tr><tr><td>P04</td><td>19</td><td>M</td><td>Right</td></tr><tr><td>P05</td><td>27</td><td>M</td><td>Left</td></tr><tr><td>P06</td><td>47</td><td>F</td><td>Right</td></tr><tr><td>P07</td><td>31</td><td>F</td><td>Right</td></tr><tr><td>P08</td><td>33</td><td>F</td><td>Right</td></tr><tr><td>P09</td><td>34</td><td>M</td><td>Right</td></tr><tr><td>P10</td><td>30</td><td>F</td><td>Right</td></tr><tr><td>P11</td><td>37</td><td>F</td><td>Right</td></tr><tr><td>P12</td><td>23</td><td>M</td><td>Left</td></tr><tr><td>P13</td><td>25</td><td>M</td><td>Right</td></tr><tr><td>P14</td><td>29</td><td>M</td><td>Right</td></tr><tr><td>P15</td><td>23</td><td>M</td><td>Left</td></tr><tr><td>P16</td><td>28</td><td>M</td><td>Left</td></tr><tr><td>P17</td><td>23</td><td>F</td><td>Left</td></tr><tr><td>P18</td><td>28</td><td>M</td><td>Left</td></tr><tr><td>P19</td><td>29</td><td>M</td><td>Right</td></tr></table>

## B Protocol Deviations

Our first participant, P01, completed tasks at lower and higher force values $( 1 5  5 , 3 6  2 0 ,$ $3 0  5 0 , 7 2  1 2 0$ N) which were determined to be artificially low and unsustainably high, respectively. These protocol deviations reflect improvements made during system development: P01 provided critical feedback on required force levels, which were adjusted for subsequent participants to provide a more comfortable user experience. Similarly, T14 and T15 were added after the first two participants to develop a more robust test set, such that P01 and P02 did not complete them. We elected not to exclude these early-protocol data from model training and evaluation, granting our model the widest distribution possible.

## C Locked & Unlocked Task Motivation & Performance

Each task was undertaken in both “unlocked” and “locked” versions, where users’ force inputs were not and were constrained to the task’s relevant axis, respectively. As illustrated in Figure 6, these tasks resulted in qualitatively different behavior along both prescribed and non-prescribed axes: in unlocked tasks, users oscillated about the target values in all 3 axes, while for locked tasks, users motions were smoother, but maintained an offset in non-prescribed axes that differed across users. Both locked and unlocked task specifications are used in therapy, and with no consensus on which may perform most effectively, we elect to include both. A comprehensive examination of model performance across both conditions is beyond the scope of this work, but in practice, the model performed with comparable overall error on both types of tasks.

![](images/c901fbcc54f27c9f8952ad5b5407c951062fb89005a1750b736430ff47752193.jpg)

![](images/49c89eb35189a7b346c34e35ceb2836073ed3eb3e421bf34a66d4a1db242f32c.jpg)  
Figure 6: Example of user force generation behavior f about target v on unlocked x-axis task T01 (top) and its locked counterpart T02 (bottom). Due to the challenge of maintaining appropriate values in all axes during unlocked tasks, users frequently correct one axis at a time, resulting in oscillating signals. On locked tasks, users’ signals are smoother, but have bias in unspecified dimensions that varies from user to user.

## D Task Trajectories

Tasks T01–T12 (the training set), illustrated in Figures 7–12, consisted of a dominant axis that users were asked to follow. Users were shown the trajectory that their target would follow, a target ball traversing that trajectory at the prescribed pace, and an avatar ball whose velocity they controlled via force exertion. “Unlocked” and “locked” versions of the tasks shared trajectories; for locked versions, the avatar ball only deviated in appearance along the prescribed axis. For tasks along the z-axis, target z-coordinate position was indicated by increasing and diminishing size of the target ball (shown in Figures 11–12). Tasks T14–T15 (the test set), illustrated in Figures 13–14, consisted of stochastically generated trajectories along the x and y axes. These trajectories were generated via a Brownian motion diffusion process that was then smoothed by a cubic spline. Trajectories were generated at collection time such that they varied for each participant, and were never seen in the training set.

![](images/e480c4aa1f006b7e2e711ab16c01cb8b44d96883ade2b8c8174d9747b32e2f4c.jpg)

![](images/c1c442f16a4668b55cba994126dd4baa508b1323481ab512883a9b3e56d93292.jpg)  
Figure 7: T01 and T04

![](images/9ee50a92019b1550d7ea106386c9ffe621e6983d0e99ae5037ffbf177bfc22c1.jpg)

![](images/1c30b1041adca681fd8c23d80a72752d79e90349ba04382e958a5c7a58d6a9c8.jpg)  
Figure 8: T07 and T10

![](images/d12f96c7d0d51841573e82926a34f8ffdc78975f99503f170413269964217488.jpg)

![](images/19456a3346126f3963fac4b539caf062ba052f8477944da5c2e80a78e1001eae.jpg)  
Figure 9: T02 and T05

![](images/7109dcd32b50c8a7b1b4ff48f1761902eeb72ee495b13be07a2c5e725e9232f1.jpg)

![](images/f7a7d732306adabc994d9ccc1c94aa0eb9496cb537e59a6bfa9214a2d9620928.jpg)  
Figure 10: T08 and T11

![](images/0222c624178296b2656dddaeae8c4c8f3f895542dddf4b42942bd6240f8af454.jpg)

![](images/f8297056b99d1fe3030f9c7afc2cac6f5993c728a056a6b7dbba765d799f7ede.jpg)  
Figure 11: T03 and T06

![](images/1858121408443ea82ba3bef0d2577ce0a422daf6544bcd20bb0fb378edd107e7.jpg)

![](images/17ddadd80507e6827a20958ddacb6138f3a3d55973df0ca0650f74759a6ac23a.jpg)  
Figure 12: T09 and T12

![](images/7d508d3522d196288fc004195ebeca8ad0992ea6e65ab338b13391889ae102f0.jpg)

![](images/1c84099e6bb1b5dfb74d9665c7e9d339de03846dfd3ca54b2f07b0467f53f09d.jpg)  
Figure 13: T14 (exemplar)

![](images/1298879fcb7a7d2097ab30d6fe1ea2388a18ab3484ddadb149494ae8f1aa2829.jpg)

![](images/d39749eba7061bba42120fd72bb63756418ba5fa57b4e21dc085f1d741cc4427.jpg)  
Figure 14: T15 (exemplar)

## E sEMG Processing Protocol

Raw 8-channel sEMG were processed by a butterworth band-pass filter (20–500 Hz), a second-order iir notch filter (to remove power line frequencies), rectified, and smoothed by RMS with window size 200. Processed sEMG were resampled or aligned to 1000 Hz to match the sample rate of the captured forces.

## F Hyperparameters

Table 4 contains hyperparameter values used during training. These values were selected empiri cally; a systematic hyperparameter search could likely improve model performance.

Table 4: Hyperparameters
<table><tr><td>Component</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="9">Transformer Encoder/Decoder</td><td>Model width</td><td>128</td></tr><tr><td>Attention heads</td><td>4</td></tr><tr><td>Feedforward width</td><td>512</td></tr><tr><td>Encoder layers</td><td>3</td></tr><tr><td>Decoder layers</td><td>3</td></tr><tr><td>Latent dimension</td><td>64</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Activation loss weight (λ₁)</td><td>1.0</td></tr><tr><td>Velocity Trajectory loss weight  $\left( \lambda _ { 2 } \right)$ </td><td>2.0</td></tr><tr><td></td><td>Flow Matching loss weight (λ3) Learning rate</td><td>1.0 10-3</td></tr><tr><td rowspan="3">Flow Matching</td><td></td><td></td></tr><tr><td>Latent dimension (dz) Vector field hidden width</td><td>64</td></tr><tr><td>ODE solver steps</td><td>256 20</td></tr><tr><td rowspan="6">LSTM (baseline)</td><td>Hidden width</td><td>128</td></tr><tr><td>Encoder layers</td><td>3</td></tr><tr><td>Decoder layers</td><td>3</td></tr><tr><td>Latent dimension</td><td>32</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Learning rate</td><td>10-3</td></tr><tr><td rowspan="6">Naive MLP (baseline)</td><td>Hidden width</td><td>128</td></tr><tr><td>Encoder layers</td><td>3</td></tr><tr><td>Decoder layers</td><td>3</td></tr><tr><td>Latent dimension</td><td>32</td></tr><tr><td>Activation function</td><td>ReLU</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td></td><td>Learning rate</td><td>10-3</td></tr></table>

## G Baseline & Ablations Details

Ablations to the model’s architecture were carried out to validate if the complexity of our model was justified, and our model was compared against two naive baselines. The following models were tested:

• Naive MLP: Encoder-Decoder style MLP that maps activation window a directly to target trajectory $\hat { \mathbf { v } } _ { \mathrm { t a r g e t } }$

• LSTM: Long short-term memory recurrent network mapping activation window a directly to target trajectory $\hat { \mathbf { v } } _ { \mathrm { t a r g e t } }$

• Minimal: Target trajectory reconstruction only (Figure 2 blue and green flows only).

• Minimal + Act. Dec.: Minimal model plus activation decoder (Figure 2 blue, green, and turquoise flows only).

• Minimal + Flow Mat.: Minimal model plus flow matching (Figure 2 blue, green, and orange flows only).

• Full: Full model (Figure 2 all).

## H Trajectory Prescription Performance Plots

![](images/d2c55a55b77fd6b5b07cd16f08ba19068e35888465d6c0dbf76578a71073f9cc.jpg)  
Figure 15: Target (top) and user-generated (bottom) muscle activations for participant P08.

![](images/c594b69b5bc113cdd18587524ba1053fa2f11df382074ab63b6980fe64d8c8d3.jpg)  
Figure 16: Target (top) and user-generated (bottom) muscle activations for participant P13.

![](images/67c045eb0a861ef82ab37f945b24446fdf4eacb5bbb076f1a95ee39d98d207e1.jpg)  
Figure 17: Target (top) and user-generated (bottom) muscle activations for participant P18.

![](images/648268394aa194f69f31c2fa06874cdc3846f12fa974905303d707989f0a2222.jpg)  
Figure 18: Target (top) and user-generated (bottom) muscle activations for participant P19.

## I User-Specific Fine-Tuning Algorithm

To identify which tasks were most informative for fine-tuning on unseen users, the greedy algorithm detailed in Algorithm 1 was run on user data whose tasks were withheld from the training data. For each held-out user, we loaded all of their task data, loaded the trained weights into our model, trained on each task individually, tested on each task individually, and added the task with the best test loss to a list. This process was repeated until a single task remained, which we retained for our final test set. Empirically, we did not observe substantial difference in performance based on which tasks were selected for training — more tasks improved performance regardless — but our small number of tested users (6) means that this remains a preliminary observation.

Algorithm 1 GreedyFineTune   
Input: participant id   
Output: ordered sets ${ { \mathcal { T } } ^ { * } } , { \mathcal { L } }$   
1: $\mathbf { \dot { \tau } } ^ { * } \gets [ ]$ ▷ Best task execution order for fine-tuning   
2: $\mathcal { L }  [ ] ^ { \lfloor }$ ▷ Best test loss after each task completion   
3: $\mathcal { T } \gets \{ 1 , 2 , \dots , 1 5 \} _ { i d }$   
4: while $\lvert \mathcal { T } ^ { * } \rvert < \lvert \mathcal { T } \rvert - 1$ do   
5: $S \gets \emptyset$ ▷ Candidates for best data to use for fine-tuning   
6: for each task $t \in \mathcal T$ do   
7: ${ \mathcal { F } } \gets \mathrm { L o } { \mathfrak { s } }$ ADPRETRAINEDMODEL() ▷ Load baseline model   
8: i $\mathbf { f } \ t \in \tau ^ { * }$ then   
9: continue   
10: end if   
11: ${ \mathcal { T } } _ { \operatorname { t r a i n } } \gets { \mathcal { T } } ^ { * } \cup \{ t \}$   
12: $\mathcal { T } _ { \mathrm { t e s t } }  \mathcal { T } \backslash \mathcal { T } _ { \mathrm { t r a i n } }$   
13: TRAINMODEL(T , F) ▷ Re-train model using the new training set   
14: $\boldsymbol { \ell } \gets \mathrm { T E S T M O D E L } ( \mathcal { T } _ { \mathrm { t e s t } } , \mathcal { F } )$ ▷ Re-test model using remaining test set   
15: $S \gets S \cup \{ ( t , \ell ) \}$   
16: end for   
17: $( t ^ { * } , \ell ^ { * } ) \gets \arg \operatorname* { m i n } _ { \boldsymbol { \rho } } S$   
18: ${ \mathcal { T } } ^ { * } \gets { \mathcal { T } } ^ { * } \cup [ t ^ { * } ]$   
19: ${ \mathcal { L } } \gets { \mathcal { L } } \cup [ \ell ^ { * } ]$   
20: end while   
21: $\mathcal { T } _ { \mathrm { r e m } }  \mathcal { T } \backslash \mathcal { T } ^ { * }$   
22: return ${ \mathcal { T } } ^ { * } { \dot { \cup } } { \mathcal { T } } _ { \mathrm { r e m } } , { \mathcal { L } }$