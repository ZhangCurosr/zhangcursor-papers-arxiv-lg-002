# Preserving Unstable Modes Through Inverse Dynamics in JEPA World Models

Leonardo F. Toso\*1, Yann LeCun3, James Anderson¹, and Oumayma Bounou²

1Columbia University 2New York University 3New York University, AMI Labs

Website Code

## Abstract

Robotic systems often exhibit unstable modes, along which small perturbations and disturbances can cause unbounded growth unless corrected through feedback. Controlling such systems from high-dimensional visual observations requires representations that preserve these modes. Joint-embedding predictive architectures (JEPAs) provide a natural framework for learning such representations and their dynamics from visual data. However, we demonstrate that next step prediction combined with anti-collapse regularization does not guarantee that controllable unstable modes are preserved: the training loss can be minimized while these modes are collapsed, making stabilization from the learned representation impossible. To address this, we augment world-model training with an action reconstruction objective (i.e., an inverse dynamics loss) that encourages control-aware representations, namely, visual representations that preserve crucial features for control. We prove that exact action reconstruction makes the encoder injective on the finite-horizon reachable subspace. Thus, the encoder cannot discard any state direction reachable by an action sequence within H steps. Moreover, we show that, as H grows, the dominant eigenspace of the finite-horizon controllability Gramian converges to the controllable unstable subspace. We establish our theoretical results for linear systems and demonstrate empirically that our findings extend to nonlinear visual control tasks (CartPole, Walker2D, and PointMaze), highlighting the benefits of control-aware representation learning.

## 1 Introduction

Quadrotors and legged robots operate around unstable equilibria where small perturbations can grow catastrophically without corrective feedback (Mellinger and Kumar, 2011; Ames et al., 2014). Controlling such systems requires actions that not only minimize a cost (e.g., goal reaching), but also stabilize the physical dynamics; otherwise, incorrect feedback can rapidly drive the system from its intended operating regime and compromise safe deployment (Ames et al., 2016; Marco et al., 2021).

Latent world models have found their way into robotics as a computationally efficient solution for learning dynamics and designing control actions from high-dimensional sensory observations. They compress pixel observations into a lower-dimensional representation, learn action-conditioned latent dynamics, and optimize actions by rolling out these dynamics, with limited interaction with the physical system (Ha and Schmidhuber, 2018; Hafner et al., 2019, 2020, 2021, 2023). Joint-embedding predictive architectures (JEPAs) follow this recipe without pixel-level reconstruction (LeCun, 2022; Sobal et al., 2025; Balestriero and LeCun, 2025): an encoder extracts a latent representation, a predictor forecasts its evolution, and anti-collapse mechanisms (e.g., SIGReg (Balestriero and LeCun, 2025), EMA (Assran et al., 2023), and others (Bardes et al., 2021)) prevent representation collapse. Planning can then be performed efficiently directly in this learned latent space (Zhou et al., 2024; Toso et al., 2026; Wang et al. 2026a). However, deploying JEPAs on unstable systems remains largely unexplored.

![](images/654477f2ac2ccbb7a265bedc06f79af093792c25bd193eadf37f2341bbc67b35.jpg)  
Figure 1: Architecture of our proposed world model. At each step, an encoder (E) maps observation $y _ { t }$ to a latent state $z _ { t } ,$ and a predictor (P) rolls out latent predictions $\hat { z } _ { t + 1 } , \dotsc , \hat { z } _ { t + H }$ conditioned on actions. Training losses are applied at each predicted latent: 1SP/MSP (single- or multi-step prediction) aligns $\hat { z } _ { t + k }$ with the encoded ground-truth $z _ { t + k }$ . The EP-IDM estimates the intermediate actions $\hat { \mathbf { a } } _ { t , H } =$ $\left( \hat { a } _ { t } , \hat { a } _ { t + 1 } , \dots , \hat { a } _ { t + H - 1 } \right)$ from $z _ { t }$ and $z _ { t + H }$

When actions are designed using the learned latent representation and dynamics, the representation must preserve the physical information required for feedback control. For open-loop unstable systems, this requires preserving the unstable modes that feedback must observe and correct (Lemma 1) (Hu et al., 2022; Werner and Peherstorfer, 2024; Toso et al., 2025; Lutkus et al., 2025). That is, if a controller cannot “see" an unstable mode, it cannot stabilize it.

However, learning to accurately predict latent dynamics while preventing representation collapse does not guarantee that this control-theoretic information is preserved. In fact, the objective combining next-step prediction and anti-collapse regularization, the standard recipe for JEPAs, can be minimized while the encoder discards every unstable mode (Lemma 2). To preserve these modes, we augment the JEPA predictive learning objective with an “endpoint inverse-dynamics" (EP-IDM) loss, which reconstructs the underlying action sequence from the initial and final latent states (see Figure 1), so that the encoder cannot discard directions the actions can reach (Mhammedi et al., 2020; Ivashkov et al., 2026). Our contributions are summarized below.

• Prediction does not guarantee stabilization. We show that a controller acting on the latent states can stabilize the system if and only if the encoder keeps every unstable mode (Lemma 1). We then show, with SIGReg as the anti-collapse regularizer, that the training objective can reach its minimum while the encoder discards all unstable modes (Lemma 2).

• Inverse dynamics preserves unstable modes. We introduce EP-IDM, which reconstructs the action sequence from the initial and final latent states. We prove that, with sufficient action excitation, exact reconstruction keeps every direction reachable within H steps, and hence every unstable mode reachable within H steps (Theorem 1).

• Unstable subspace recovery. We prove that, as H grows, the dominant eigenspace of the finite-horizon controllability Gramian converges to the unstable subspace (Theorem 2).

• Numerical validation. We show on synthetic linear systems and a linearized CartPole that EP-IDM preserves the unstable direction and enables latent stabilization, while SIGReg does not. We further validate the benefits of EP-IDM on nonlinear visual control tasks (CartPole, Walker2D, and PointMaze), where representations and dynamics are learned from images and proprioceptive measurements.

## 2 Related Work

Our work lies at the intersection of joint-embedding predictive world models and latent feedback stabilization.

Joint embedding predictive world models. JEPAs learn representations by predicting future latent states rather than reconstructing pixels (LeCun, 2022; Assran et al., 2023). Recent work applies this recipe to action-conditioned prediction and planning using pre-trained visual encoders (Zhou et al., 2024; Toso et al., 2026), end-to-end anti-collapse mechanisms (Balestriero and LeCun, 2025; Maes et al., 2026; Kuang et al., 2026), or objectives that shape latent dynamics and geometry (Sobal et al., 2025; Parthasarathy et al., 2025; Wang et al., 2026a; Zhang et al., 2026; Nath et al., 2026). While this body of work primarily addresses total representation collapse and planning, we demonstrate that standard JEPA objectives can selectively collapse directions required for feedback stabilization.

Most closely related, Ivashkov et al. (2026) use one-step inverse dynamics to prevent collapse. We share the principle that actions provide a control-relevant learning signal but we study feedback stabilization of open-loop unstable systems. Our EP-IDM loss reconstructs the entire action sequence from the endpoint latent states. We prove that exact reconstruction preserves the finite-horizon reachable subspace and characterize its convergence to the controllable unstable subspace, connecting action reconstruction to detectability and latent stabilizability.

Latent feedback stabilization. Control from visual observations requires retaining the state information needed for stabilization. Prior work studies stabilization of linear systems under unknown nonlinear observation maps (Mhammedi et al., 2020), characterizes the sample complexity of learning to stabilize (Hu et al., 2022; Werner and Peherstorfer, 2024), and learns low-dimensional unstable subspace representations (Toso et al., 2025; Lutkus et al., 2025). In particular, preserving only the unstable subspace can suffice, since stable modes already decay without correction (Toso et al., 2025). More than just constructing those sufficient representations, we also ask whether standard JEPA training can learn them in the first place. In particular, we connect latent world model training objectives to the control-theoretic information that their representations must preserve.

## 3 Setup

We introduce the dynamical system, latent world model, and training objectives considered throughout the paper.

Learning an encoder and a predictor. We consider the discrete-time dynamical system

$$
x _ { t + 1 } = f ( x _ { t } , a _ { t } ) , \qquad y _ { t } = g ( x _ { t } ) \in \mathcal { V } \subset \mathbb { R } ^ { p } ,\tag{1}
$$

for all $t = 0 , 1 , 2 , \ldots ,$ where $x _ { t } \in \mathcal { X } \subset \mathbb { R } ^ { n }$ is the state and $a _ { t } \in \mathcal { A } \subset \mathbb { R } ^ { m }$ is the applied action, at time t. The transition map $f : \mathcal { X } \times \mathcal { A }  \mathcal { X }$ describes the system dynamics, while the observation map $g : \mathcal { X }  \mathcal { Y }$ maps the state to the observation $y _ { t }$ . This observation may contain images, proprioceptive measurements $( { \mathrm { i . e . } }$ , part or all of the state $x _ { t } )$ , both, or other modalities (Wang et al., 2026b; Huang et al., 2026). Examples of such a mapping are provided in Appendix A.5. We call Eq. (1) the physical system.

Our goal is to learn an observation encoder $E _ { \theta } : \mathcal { V }  \mathcal { Z }$ , an action encoder $G _ { \theta } : { \mathcal { A } }  { \mathcal { U } } _ { \theta }$ and a latent dynamics model (predictor) $P _ { \theta } : \mathcal { Z } \times \mathcal { U }  \mathcal { Z }$ such that

$$
z _ { t + 1 } = P _ { \theta } { \left( z _ { t } , G _ { \theta } ( a _ { t } ) \right) } \mathrm { ~ w i t h ~ l a t e n t ~ s t a t e ~ } z _ { t } = E _ { \theta } ( y _ { t } ) ,\tag{2}
$$

where $\mathcal { Z } \subset \mathbb { R } ^ { d }$ is the latent space, with $d \ll p .$ For notation simplicity, we denote by $\theta \in \mathbb { R } ^ { d _ { \theta } }$ the trainable parameters of the observation encoder, action encoder, and predictor.

Standard training objective. Let D denote the distribution of finite-horizon trajectory windows collected from the system in Eq. (1). Each sampled window $\tau \sim \mathcal { D }$ starts at a trajectorydependent time $t _ { \tau } \geq 0$ and is given by

$$
\tau : = ( y _ { t _ { \tau } - 1 : t _ { \tau } + H } , s _ { t _ { \tau } : t _ { \tau } + H } , a _ { t _ { \tau } : t _ { \tau } + H - 1 } ) ,\tag{3}
$$

where $y _ { t } , \ s _ { t } .$ and $a _ { t }$ denote the image observation, proprioceptive measurement, and action at time $t ,$ respectively. Hence, each window has $H + 2$ images, H + 1 proprioceptive measurements, and H actions. The preceding image $y _ { t _ { \tau } - 1 }$ is used only to construct the first encoder input resulting in $H + 1$ latent states and H prediction transitions. In our experiments, the encoder uses two consecutive images, their difference to capture motion, and the current proprioceptive measurement:

$$
z _ { t _ { \tau } + k } = E _ { \theta } \left( y _ { t _ { \tau } + k - 1 } , y _ { t _ { \tau } + k } , y _ { t _ { \tau } + k } - y _ { t _ { \tau } + k - 1 } , s _ { t _ { \tau } + k } \right) , \qquad k = 0 , \dots , H .\tag{4}
$$

We train the model using the next-step prediction objective

$$
\mathcal { L } _ { \mathrm { p r e d } } ( \theta ) = \mathbb { E } _ { \tau \sim \mathcal { D } } \left[ \frac { 1 } { H } \sum _ { k = 0 } ^ { H - 1 } \big \lVert P _ { \theta } \big ( z _ { t _ { \tau } + k } , G _ { \theta } ( a _ { t _ { \tau } + k } ) \big ) - z _ { t _ { \tau } + k + 1 } \big \rVert _ { 2 } ^ { 2 } \right] .\tag{5}
$$

We combine the prediction loss with SIGReg (Balestriero and LeCun, 2025) to prevent total representation collapse:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } _ { \mathrm { p r e d } } ( \theta ) + \lambda _ { \mathrm { r e g } } \mathcal { L } _ { \mathrm { r e g } } ( \theta ) ,\tag{6}
$$

where $\lambda _ { \mathrm { r e g } } > 0$ controls the strength of the regularizer.

Total representation collapse. The representation collapse in JEPAs occurs when the encoder $E _ { \theta }$ maps almost all observations to the same latent representation, i.e., $E _ { \theta } ( y ) = c$ almost surely for some constant $^ { c , }$ so that the latent distribution has zero covariance. In particular, SIGReg (Balestriero and LeCun, 2025) (also defined in Appendix A.3.4 for completeness) avoids such a degenerate solution by encouraging the aggregate latent distribution to remain dispersed and approximately isotropic. However, this condition does not require the encoder to preserve any particular physical direction: the representation can keep sufficient variability while selectively discarding the information (i.e., unstable modes) required for feedback stabilization. We formalize this collapse of the unstable modes in Section 4.

Planning in the latent space. 1 Given an initial observation $y _ { 0 }$ and a goal observation $y _ { \mathrm { g } } ,$ we encode $z _ { 0 } = E _ { \theta } ( y _ { 0 } )$ and $z _ { \mathrm { g } } = E _ { \theta } ( y _ { \mathrm { g } } )$ and optimize the action sequence through the learned latent dynamics. For a planning horizon $T .$ we solve

$$
\operatorname* { m i n } _ { a } \quad \| z _ { T } - z _ { \mathsf { g } } \| _ { 2 } ^ { 2 } \qquad \mathrm { s u b j e c t ~ t o } \qquad z _ { t + 1 } = P _ { \theta } \big ( z _ { t } , G _ { \theta } ( a _ { t } ) \big ) , \qquad t = 0 , \ldots , T - 1 ,\tag{7}
$$

where $\pmb { a } = ( a _ { 0 } , \ldots , a _ { T - 1 } )$ is the designed action sequence. In practice, we use a receding-horizon implementation. More details in Appendix $\mathrm { A . 3 }$

Open-loop stability. Let $x _ { \mathrm { e } }$ be an equilibrium under zero input, i.e., $x _ { \mathrm { e } } = f ( x _ { \mathrm { e } } , 0 )$ . The system is locally open-loop stable around $x _ { \mathrm { e } }$ if trajectories initialized sufficiently close to $x _ { \mathrm { e } }$ remain close to it when $a _ { t } = 0$ for all t. It is locally asymptotically stable if these trajectories additionally converge to $x _ { \mathrm { e } }$ as $t \to \infty$ , and it is open-loop unstable if the trajectories diverge from $x _ { \mathrm { e } }$

For a linear system $x _ { t + 1 } = A x _ { t }$ , open-loop asymptotic stability is equivalent to A being Schur stable, $\operatorname { i . e . , } \rho ( A ) < 1$ . Eigenvalues satisfying $| \lambda | > 1$ correspond to unstable modes, while those with $| \lambda | = 1$ are marginally stable. We emphasize that Schur stability prevents $\| x _ { t } \|  \infty$ as $t \to \infty$ . Likewise, stabilization allows us to design the feedback that sends $\| x _ { t } \|$ to zero.

Stabilization and unstable modes. For an open-loop unstable system, i.e., $x _ { t } = A x _ { t }$ with $\rho ( A ) > 1$ , the goal is to design a feedback controller that selects actions from the available observations and makes the desired equilibrium asymptotically stable. Under a static linear feedback controller $a _ { t } ~ = ~ K x _ { t }$ , this amounts to finding K such that the closed-loop matrix $A + B K$ is Schur stable. When the controller acts only on the learned latent state, this is possible only if the encoder preserves every unstable or marginally stable direction that must be corrected through feedback (Toso et al., 2025). In this work, we demonstrate that this requirement is not implied by an accurate latent next-step predictor alone.

Next, we formalize necessary and sufficient conditions for latent stabilization and demonstrate that the objective above in Eq. (6) does not necessarily satisfy them.

## 4 Prediction Does Not Guarantee Stabilization

An encoder $E _ { \theta }$ can discard unstable modes while still learning predictable, non-constant representations. In particular, there may exist an initial condition and a controller that stabilizes the learned latent dynamics, so that $\| z _ { t } \| _ { 2 } \to 0$ , while the corresponding physical trajectory satisfies $\| \boldsymbol { x } _ { t } \| _ { 2 } \to \infty$ because an unstable mode lies in the kernel of the encoder. In the linear setting, we characterize the conditions that rule out this failure and demonstrate that next step prediction combined with SIGReg does not necessarily satisfy them. We illustrate these findings at the end of this section with two examples (Fig. 2) and further validate them in Section 6, with additional details also provided in Appendix A.5

Linear setting. We consider a linear dynamical system (LDS) obtained by linearizing Eq.(1) around an equilibrium point $x ^ { \star }$ , and we assume $x ^ { \star } = 0$ without loss of generality:

$$
x _ { t + 1 } = A x _ { t } + B a _ { t } , y _ { t } = C x _ { t } ,\tag{8}
$$

for all $t \in \mathbb { Z } _ { + }$ , where $x _ { t } \in \mathbb { R } ^ { n } , a _ { t } \in \mathbb { R } ^ { m } , y _ { t } \in \mathbb { R } ^ { p }$ , and A, B, and C are the transition, control, and observation matrices, respectively. We consider an open-loop unstable system $( \operatorname { i . e . , } \rho ( A ) > 1 )$

Definition 1 (Stabilizability, observability, and detectability). The pair $( A , B )$ is stabilizable if there exists a gain K such that $A + B K$ is Schur stable under $a _ { t } = K x _ { t }$ . Given $q _ { t } = M x _ { t }$ , the pair $( A , M )$ is observable if x0 can be recovered from nitely many observations, and detectable if every unobservable mode is Schur stable. By the Popov-Belevitch-Hautus (PBH) test (Hautus, 1969), detectability is also equivalent to

$$
\operatorname { r a n k } { \left[ \begin{array} { l } { \lambda I - A } \\ { M } \end{array} \right] } = n \quad f o r \ e v e r y \ \lambda \in \operatorname { s p e c } ( A ) \ s u c h \ t h a t \ | \lambda | \geq 1 ,\tag{9}
$$

where spec(A) denotes the set of eigenvalues of A.

Assumption 1. For the system $( \boldsymbol { \vartheta } )$ , the pair $( A , B )$ is stabilizable and $( A , C )$ is observable.

We consider an affine observation encoder and a linear predictor:

$$
E _ { \theta } ( y ) = W y + b = z \in \mathcal { Z } , \qquad P _ { \theta } ( z , a ) = A _ { z } z + B _ { z } a \in \mathcal { Z } .\tag{10}
$$

Here d $< n$ . The vector $\theta = ( W , b , A _ { z } , B _ { z } )$ collects all trainable parameters. We write the linear predictor directly in terms of the actions, as a linear action encoder can be absorbed into $B _ { z }$ without loss of expressivity.

## 4.1 Necessary and Sufficient Conditions for Stabilization of LDS

We first characterize the conditions under which a controller acting only on the learned latent state can stabilize the physical system. We defer the proof of Lemma 1 to Appendix A.6.2.

Lemma 1. Let $F = W C$ denote the state-to-latent map and consider the centered latent state $\tilde { z } _ { t } = z _ { t } - b = F x _ { t }$ . Suppose that Assumption 1 holds. Then there exists a causal dynamic controller that uses only the centered latent states $\{ \tilde { z } _ { \tau } \} _ { \tau = 0 } ^ { t }$ and asymptotically stabilizes $E q . \ ( 8 )$ if and only $i f \left( A , F \right)$ is detectable. Equivalently, every unstable or marginally stable mode of A is observable through F, or, by the PBH test, there does not exist a nonzero vector $v \in \mathbb { C } ^ { n }$ such that

$$
A v = \lambda v , \qquad F v = 0 , \qquad | \lambda | \geq 1 .\tag{11}
$$

Lemma 1 demonstrates that unstable and marginally stable modes in (8) need to remain observable through the encoder $E _ { \theta } , \mathrm { i . e . , } E _ { \theta }$ needs to preserve them. In the next subsection, we demonstrate that minimizing next step prediction loss together with SIGReg does not guarantee this property.

## 4.2 The Collapse of Unstable Modes

Let $\Phi _ { \geq 1 }$ denote the unstable subspace of A associated with eigenvalues satisfying $| \lambda | \geq 1$ . We next demonstrate that standard predictive training objectives $\left( \mathrm { e . g . , ~ E q . ~ ( 6 ) } \right)$ can discard the entire subspace $\Phi _ { > 1 }$ while still reaching their minimum.

Lemma 2. There exist open-loop unstable linear systems satisfying Assumption 1 and data distributions for which the prediction loss $\mathcal { L } _ { \mathrm { p r e d } }$ and SIGReg objective are simultaneously minimized by an encoder satisfying $\Phi _ { \geq 1 } \subset \ker ( F )$

Proof. We begin our proof by setting the encoder bias to zero and consider coordinates in which

$$
A = \left[ \begin{array} { l l } { A _ { u } } & { \Delta } \\ { 0 } & { A _ { s } } \end{array} \right] , B = \left[ \begin{array} { l } { B _ { u } } \\ { B _ { s } } \end{array} \right] , \mathrm { a n d } x _ { t } = \left[ \begin{array} { l } { x _ { u , t } } \\ { x _ { s , t } } \end{array} \right] .\tag{12}
$$

Here, we note that $A _ { u }$ is open-loop unstable and $A _ { s }$ is stable. We choose $F = [ 0 ~ F _ { s } ]$ , where $F _ { s }$ is invertible on the stable coordinates. Then, we have

$$
z _ { t + 1 } = F _ { s } A _ { s } F _ { s } ^ { - 1 } z _ { t } + F _ { s } B _ { s } u _ { t } .\tag{13}
$$

Therefore, choosing $A _ { z } = F _ { s } A _ { s } F _ { s } ^ { - 1 }$ and $\begin{array} { r } { B _ { z } = F _ { s } B _ { s } } \end{array}$ implies zero one-step error, while every unstable direction lies in ker(F).

Note that SIGReg (Balestriero and LeCun, 2025) does not rule out this degenerate solution. Let $\boldsymbol { x } _ { s } \sim \mathcal { N } ( 0 , \Sigma _ { s } )$ , with $\Sigma _ { s } \succ 0$ , and take $\boldsymbol { F _ { s } } = \boldsymbol { Q } \boldsymbol { \Sigma } _ { s } ^ { - 1 / 2 }$ for any orthogonal matrix $Q .$ Then $z _ { t }$ is distributed according to a standard Gaussian. We then note that when we use population SIGReg objective in Eq. (6) the regularization term $\mathcal { L } _ { \mathrm { r e g } }$ can be minimized despite the unstable directions being absent in the latent space. □

![](images/ef7ab675a8430d0184e525ad1aec52b8b3d68edbbf5611dda6c0ff6bc6c44deb.jpg)  
Figure 2: SIG vs. EP-IDM across two examples. Each row compares a representation trained with SIGReg (SIG) against one trained with the endpoint inverse-dynamics (action-reconstruction) loss (EP-IDM). Top row: synthetic open-loop unstable LDS. Bottom row: linearized CartPole. Left: true unstable $( v _ { u } , \mathrm { g r e e n } )$ and stable $( v _ { s } , { \mathrm { g r e y } } )$ eigenvectors versus the learned predictor's dominant eigenvector, for SIG (blue) and EP-IDM (dashed orange). Center: closed-loop trajectories under the latent LQR controller from four initial states (circles) for the model trained with SIGReg. Right: closed-loop trajectories under the latent LQR controller for the model trained with the inverse dynamics loss (EP-IDM). For the bottom row $x _ { 3 } .$ and $x _ { 4 }$ corresponds to the pole angle and angular velocity, respectively. The latent dimension is one for the example in the top row and three for the example in the bottom row.

The constructed encoder $E _ { \theta }$ minimizes both terms of Eq. (6) and therefore minimizes Eq. (6) for every $\lambda _ { \mathrm { r e g } } \geq 0$ , while discarding the entire unstable subspace. By Lemma 1, the physical system cannot be stabilized from the resulting latent state. Our linear analysis removes representation capacity as a confounding factor: even with a linear observation-to-latent map and little compression, standard predictive objectives can discard precisely the state directions required for stabilization.

We illustrate Lemma 2 with a synthetic open-loop unstable system (top row of Figure 2) and a linearized CartPole system (bottom row). Appendix A.5 provides details and an additional open-loop stable example. The left panels compare the learned unstable eigenvector, mapped back to the state space, with the ground-truth unstable eigenvector. The center panels show closed-loop trajectories for the physical systems under the latent linear quadratic regulator (LQR) trained with prediction and SIGReg through Eq. (6). In both systems, SIG learns a substantially misaligned unstable direction, and the resulting controller fails to stabilize the physical system.2

## 5 Preserving Unstable Directions Through Inverse Dynamics

To encourage the learned representation to preserve unstable modes, we augment the worldmodel training objective with an inverse dynamics loss (IDM). In particular, our endpoint inverse dynamics (EP-IDM) loss reconstructs the entire action sequence over the prediction horizon from the initial and final latent states (see Fig. 1 for the complete architecture).

## 5.1 Endpoint Inverse Dynamics

We train an inverse dynamics model3 $D _ { \theta } : \mathcal { Z } \times \mathcal { Z } \to \mathcal { A } ^ { H }$ to reconstruct the action sequence from the initial and final latent states $z _ { t _ { \tau } }$ and $\boldsymbol { z } _ { t _ { \tau } + H } { } ^ { 4 }$

$$
\mathcal { L } _ { \mathrm { E P - I D M } } ( \theta ) = \mathbb { E } _ { \tau \sim \mathcal { D } } \left[ \frac { 1 } { H } \sum _ { t = 0 } ^ { H - 1 } \big \lVert \left[ D _ { \theta } ( z _ { t _ { \tau } } , z _ { t _ { \tau } + H } ) \right] _ { t } - a _ { t _ { \tau } + t } \big \rVert _ { 2 } ^ { 2 } \right] ,\tag{14}
$$

where $[ D _ { \theta } ( z _ { t _ { \tau } } , z _ { t _ { \tau } + H } ) ] _ { \scriptstyle \mathrm { . } }$ t denotes the predicted action at timestep t. We jointly optimize the encoders, predictor, and inverse dynamics model solving the optimization problem in (6) with $\mathcal { L } _ { \mathrm { E P - I D M } } ( \theta )$ in lieu of ${ \mathcal { L } } _ { \mathrm { r e g } } ( \theta )$

## 5.2 Preservation of Reachable Directions

We show that exact endpoint action reconstruction requires the encoder to preserve every direction reachable within H steps. For the linear analysis, we consider an affine inverse dynamics model

$$
D _ { \theta } ( z _ { 0 } , z _ { H } ) = W _ { H } [ z _ { 0 } z _ { H } ] + b _ { H } \mathrm { ~ w i t h ~ } W _ { H } \in \mathbb { R } ^ { m H \times 2 d } \mathrm { ~ a n d ~ } b _ { H } \in \mathbb { R } ^ { m H } .\tag{15}
$$

Let $\mathcal { C } _ { H } = \left\lceil A ^ { H - 1 } B \quad A ^ { H - 2 } B \quad \cdot \cdot \cdot \ B \right\rceil$ denote the finite-horizon controllability matrix and let $\begin{array} { r } { \mathcal { R } _ { H } = \mathrm { r a n g e } ( \mathcal { C } _ { H } ) } \end{array}$ be the H-step reachable subspace.

We require the training data to contain sufficiently rich action excitation. For a fixed initial state $\scriptstyle { \boldsymbol { x } } _ { t _ { 7 } }$ , let $\operatorname { s u p p } ( \mathbf { a } _ { \tau , H } \mid x _ { t _ { \tau } } )$ denote the support of the conditional distribution of the stacked action sequence $\begin{array} { r } { \mathbf { a } _ { \tau , H } : = \left[ a _ { t _ { \tau } } ^ { \top } \quad \cdot \cdot \quad a _ { t _ { \tau } + H - 1 } ^ { \top } \right] ^ { \top } \in \mathbb { R } ^ { m H } } \end{array}$

Theorem 1. Consider the linear system in $E q . \ ( 8 )$ and suppose that, for almost every initial state $\boldsymbol { x } _ { t _ { \tau } }$ , the conditional support $\operatorname { s u p p } ( \mathbf { a } _ { \tau , H } \mid x _ { t _ { \tau } } )$ contains a nonempty open subset of $\mathbb { R } ^ { m H }$ . If an afine inverse dynamics model achieves $\mathcal { L } _ { \mathrm { E P - I D M } } = 0$ , then

$$
\ker ( F ) \cap { \mathcal { R } } _ { H } = \{ 0 \} .\tag{16}
$$

Equivalently, the encoder is injective on the H-step reachable subspace: $i f v \in \mathcal { R } _ { H }$ and $F v = 0$ 2 then $v = 0$ . Thus, no state direction that can be generated by an action sequence within H steps is discarded by the encoder. In particular, if $\Phi _ { \geq 1 } \subseteq \mathcal { R } _ { H }$ , then ker $\left( F \right) \cap \Phi _ { \geq 1 } = \left\{ 0 \right\}$ . Exact reconstruction $\mathcal { L } _ { \mathrm { E P - I D M } } = 0$ holds only if rank $( F \mathcal { C } _ { H } ) = m H \le \operatorname* { m i n } \{ d , n \}$

The proof of Theorem 1 is provided in Appendix A.6.3. We note that under Assumption 1 and a sufficiently large horizon H, we have that $\Phi _ { \geq 1 } \subseteq \mathcal { R } _ { H }$ . Therefore, the theorem above ensures that all unstable and marginally unstable modes remain observable through the representation, satisfying the detectability condition in Lemma 1.

Theorem 1 describes the idealized exact-reconstruction regime. We note that empirically, we use $\mathcal { L } _ { \mathrm { E P . } }$ -IDM as a soft inductive bias and do not require it to be zero. Indeed, when m $H > d .$

exact recovery of an arbitrary action sequence is not feasible. Nevertheless, minimizing the loss can still encourage the representation to preserve the action-induced directions required for control (Section 6).

As illustrated in Fig. 2, training the latent world model with the endpoint inverse dynamics loss (EP-IDM) preserves the unstable direction of the physical system in the latent space. The learned latent dynamics capture the unstable mode that must be corrected through feedback. enabling latent stabilization, as shown in the right panels of Fig. 2. In addition, our result in Theorem 1 motivates the question we address next: Which controllable directions preserved by the action reconstruction are the most identifiable ones?

## 5.3 Reachability and Unstable Dynamics

We consider the finite-horizon controllability Gramian given by $\begin{array} { r } { \Pi _ { H } = \sum _ { k = 0 } ^ { H - 1 } A ^ { k } B B ^ { \top } ( A ^ { k } ) ^ { \top } } \end{array}$

For simplicity, we exclude eigenvalues on the unit circle, so $\Phi _ { \geq 1 }$ coincides with the strictly unstable subspace Φ. Let ∥ sin $\Theta ( \hat { \Phi } , \Phi ) \vert \vert _ { 2 }$ denote their subspace distance, as defined in Definition 2.

Theorem 2. Under the mild assumptions given in Appendix A.6.4, let $\hat { \Phi }$ be the span of the top $r = \dim ( \Phi )$ eigenvectors of $\Pi _ { H }$ . Then, we have

$$
\| \sin \Theta ( \hat { \Phi } , \Phi ) \| _ { 2 } \to 0 \qquad a s \ H \to \infty .\tag{17}
$$

The convergence rates and proof are provided in Appendix A.6.4.

We note that although range $( \Pi _ { H } ) = \operatorname { s p a n } ( B , A B , \dotsc , A ^ { H - 1 } B )$ does not acquire new directions once H exceeds the state dimension, the relative weighting of these directions continues to change with H. Theorem 2 demonstrates that the effects of unstable modes grow exponentially and eventually dominate the stable ones, making the leading eigenspace of $\Pi _ { H }$ to converge to the unstable subspace Φ.

Moreover, Assumption 3 in Theorem 2 requires $\lambda _ { \operatorname* { m i n } } ( \Pi _ { u , H } ) ~ \geq ~ c _ { u } \alpha ^ { 2 H }$ , where $\Pi _ { u , H }$ denotes the unstable block of the finite-horizon controllability Gramian. We emphasize that this condition strengthens controllability by requiring every unit unstable direction to exhibit an action-induced response growing at least as $\alpha ^ { 2 H }$ . It follows when $( A _ { u } , B _ { u } )$ is controllable and $\sigma _ { \operatorname* { m i n } } ( A _ { u } ) \geq \alpha > 1$ , as discussed in Appendix A.6.4. For $\Delta \neq 0$ , Assumption 3 further requires that the indirect input response transmitted through the stable coordinates and the coupling $\Delta$ does not cancel the direct unstable response generated through $B _ { u }$ . Therefore,

$$
\lambda _ { \operatorname* { m i n } } ( \Pi _ { u , H } ) = \operatorname* { m i n } _ { \| v \| _ { 2 } = 1 } v ^ { \top } \Pi _ { u , H } v \geq c _ { u } \alpha ^ { 2 H } ,\tag{18}
$$

or equivalently,

$$
v ^ { \top } \Pi _ { u , H } v \geq c _ { u } \alpha ^ { 2 H } \qquad { \mathrm { f o r ~ e v e r y ~ } } \| v \| _ { 2 } = 1\tag{19}
$$

requires that the finite-horizon controllability energy grows at the same exponential scale dictated by the unstable modes, which strengthens the condition of all unstable directions being reachable.

Therefore, taken together, Theorems 1 and 2 demonstrate that action reconstruction, in particular EP-IDM, prevents the encoder from discarding finite-horizon controllable directions. The controllability Gramian further explains why the dominant action-induced controllable directions increasingly align with the unstable subspace as H grows, i.e., why the unstable directions are the most identifiable ones.

We also note that, for simplicity, Theorem 2 excludes eigenvalues on the unit circle. Therefore, its instantiation to the linearized CartPole example in Figure 2 concerns only the strictly unstable mode depicted in the left panel.

![](images/820ca929cd366b26c73e915ae866aeeb9cea71cdfd115c2dd6f9fcbb111ddb00.jpg)  
Figure 3: CartPole, Walker2D, and PointMaze visual-control tasks used in our evaluation. Each row depicts ten frames from a successful trial (initial state: blue border, final state: green border). Cart-Pole: a continuous-action inverted-pendulum balancing task controlled via LQR. Walker2D: a bipedal robot locomotion task controlled via iCEM (Pinneri et al., 2021). PointMaze: a 2-D point-mass robot navigation task planned via CEM.

## 6 Experiments

We now validate⁵ our approach on nonlinear visual-control tasks in MuJoCo (Todorov et al., 2012). Our experiments address three questions: (i) whether standard predictive JEPA objectives (i.e., solving Eq. (6) with SIGReg as a regularizer) preserve the information required for feedback stabilization, (ii) whether EP-IDM preserves that information, and (iii) whether the resulting representations remain useful for stable systems. We consider CartPole, Walker2D, and PointMaze systems (See Fig. 3 for the visualizations of successful trials for CartPole, Walker2D, and PointMaze). Additional experiments and ablations are provided in Appendix A.1

## 6.1 Setup

We consider both encoders trained from scratch and frozen pretrained encoders (DINOv2 (Oquab et al., 2023) and iBOT (Zhou et al., 2021), with results reported in Appendix A.2), for which we train a projector on top. Our training objectives combine either onestep prediction (1SP) or autoregressive multi-step predic-

![](images/7011bd0ece3119089f47dbf403e85ab67cbcdc321fcc7290d29d68c5ee7d7397.jpg)

![](images/4b11a7fddb381915b919deefa1ac263bd92802d3b47d02b99059645bcb1902ae.jpg)  
Figure 4: CartPole state norm $\| x _ { t } \| _ { 2 }$ under the designed latent LQR controller over 300 steps.

tion (MSP) with either SIGReg (SIG) or an inverse-dynamics objective (IDM).

Our primary objective is endpoint action reconstruction (EP-IDM), for which the decoder reconstructs the complete action sequence from the first and last latent states. We additionally evaluate one-step action reconstruction (IDM), which reconstructs each action from consecutive latent states, and multi-step action reconstruction (MS-IDM), which reconstructs the complete action sequence from the corresponding latent trajectory. Further details are provided in Appendix A.3.5. Architecture, optimization, and dataset details are provided in Appendix A.3.

![](images/ea5c61ef16d1aaf8bba40b704ec814542aa12a003b687ab2661925ff7f4c2db6.jpg)

![](images/10d2f950a4fc9e3a27b55227bdfe800111369205f2df0e0f0c446dc98b36f54a.jpg)

![](images/9b0cebda4c4aa01921cc8e34d383d1c52102915356dfa37b4fae27e4fe9bc14a.jpg)

![](images/2fa9f47b07b4a4b94577495e4ce6a09350202fc097414048a6c4b4dd36a68aad.jpg)  
Figure 5: Walker2D. Phase portraits of right-hip angle vs. angular velocity for GT dynamics, 1SP+EP-IDM, 1SP+SIG, and MSP+SIG. Each trajectory is a 500-step closed-loop rollout: GT uses real MuJoCo physics, model panels roll out in latent space under an MLP-decoded state, with a SAC (Huang et al. 2022; Haarnoja et al., 2018) policy acting on the decoded observation.

Systems. CartPole requires stabilizing an open-loop unstable upright equilibrium, where small angular perturbations can make the pole fall. Walker2D is also open-loop unstable because maintaining a walking gait requires continuous feedback. PointMaze instead evaluates goal reaching in a U-shaped maze with open-loop stable point-mass dynamics.

Controllers and planners. We evaluate three uses of the learned world model. Latent LQR linearizes the predictor at the equilibrium and provides our most direct diagnostic: a collapsed unstable mode cannot be identified or corrected by the resulting feedback controller. CEM searches over sampled open-loop rollouts with repeated replanning, whereas GBP differentiates the terminal latent cost through the predicted rollout, both planners are implemented as receding-horizon MPC and periodically replan. Unlike LQR, neither requires the local linearization to capture the unstable modes. We report success over ten trials and provide all hyperparameters in Appendix A.3.

Metrics. We report success rate (SR). For CartPole, success requires $\| x _ { T } \| _ { 2 } \leq 0 . 7$ after the 300-step rollout, and the mean fraction of steps (MFS) satisfying $\| x _ { t } \| _ { 2 } \leq 0 . 7$ measures how consistently the controller remains near the upright equilibrium. For PointMaze, success requires $\| p _ { T } - p _ { \mathrm { g } } \| _ { 2 } \leq 0 . 5$ , where $p _ { T }$ and $p _ { \mathrm { g } }$ are the final and goal positions. For Walker2D, we evaluate whether the learned latent dynamics preserve the limit cycle induced by the locomotion policy used to collect the training data. We also do planning through the learned latent dynamics for Walker2D with iCEM (Pinneri et al., 2021), where success requires keeping forward locomotion without falling over the evaluation horizon. Additional details are provided in Appendix A.3.

## 6.2 CartPole: Stabilization Around an Unstable Equilibrium

Table 1 shows the effect of action reconstruction on stabilization performance when the encoder and predictor are trained from scratch. Prediction with SIGReg yields zero latent-LQR success and low MFS, whereas adding EP-IDM achieves 100% success. This confirms that standard prediction and anti-collapse regularization can discard controllable unstable directions, while action reconstruction encourages the representation to preserve the finite-horizon reachable directions containing them.

CEM and GBP succeed even with SIGReg models (Table 1). While LQR relies on a local linearization and directly verifies whether the learned model supports local feedback, CEM and GBP optimize finite action sequences through nonlinear predictor rollouts. Their success indicates that SIGReg models retain enough information for nonlinear replanning, even when their local latent linearizations do not yield stabilizing LQR controllers. Figure 4 makes the failure of the standard JEPA recipe more explicit: under 1SP+SIG and $\mathrm { M S P { + } S I G }$ , the state norm grows without bound, whereas EP-IDM enables the latent LQR controller to drive the state toward the equilibrium and keep it there.

Table 1: CartPole stabilization success rate (%).
<table><tr><td>Model</td><td>LQR (SR)</td><td>LQR (MFS)</td><td>CEM (SR)</td><td>GBP (SR)</td></tr><tr><td>Ground truth (GT)</td><td>100%</td><td>1.000</td><td>100%</td><td>100%</td></tr><tr><td> $1 \mathrm { S P } + \mathrm { S I G }$ </td><td>0%</td><td>0.202</td><td>100%</td><td>90%</td></tr><tr><td> $\mathrm { M S P } + \mathrm { S I G }$ </td><td>0%</td><td>0.247</td><td>100%</td><td>100%</td></tr><tr><td> $\mathrm { 1 S P + E P \mathrm { - } I D M }$ </td><td>100%</td><td>0.999</td><td>100%</td><td>100%</td></tr><tr><td> $\mathrm { M S P + E P \mathrm { { - } I D M + S I G } }$ </td><td>100%</td><td>0.996</td><td>100%</td><td>100%</td></tr></table>

![](images/96a6134eee344308fe6e443685c0d08bda53bd0db3ba1794dc52bdf15d2103b1.jpg)  
Figure 6: Trajectory frames for Walker2D. Each row shows ten frames uniformly sampled from a 500-step closed-loop rollout under the SAC policy (Haarnoja et al., 2018), decoded from the trained model's latent state via an MLP decoder.

## 6.3 Walker2D: Preserving a Walking Gait

For Walker2D, we evaluate whether action reconstruction preserves control-relevant dynamics beyond a single unstable equilibrium. A walking gait forms a limit cycle whose deviations must be continuously corrected. Figure 5 compares the right-hip phase portrait from the groundtruth MuJoCo simulator with latent rollouts from 1SP+EP-IDM, MSP+SIG, and 1SP+SIG. EP-IDM preserves the cyclic structure and range of joint angles and angular velocities, whereas 1SP+SIG and MSP+SIG fail to capture the limit cycle induced by the locomotion policy. Hence, we confirm empirically that action reconstruction also preserves control-relevant information associated with a locomotion orbit which goes beyond the scope of preserving unstable equilibria.

Figure 6 complements Fig. 5 where we visualize the decoded states directly in pixel space. GT dynamics (top row) produce a periodic bipedal gait throughout the rollout. 1SP+EP-IDM (second row) also keeps coherent walking motion on the episode, confirming that the model captures the structure of the walking limit cycle. 1SP+SIG (third row) and MSP+SIG (bottom row) both fail to keep a physically plausible locomotion: the decoded states are erratic and unnatural over the course of the rollout, which is not consistent with a periodic bipedal walking gait as also depicted in Fig. 5.

Table 2 and Fig. 7 evaluate the closed-loop control performance of the learned models under the latent iCEM (Pinneri et al., 2021) across ten trials. The ground-truth planner achieves 3.61 m/s and 14.43 m of forward displacement. Among the learned models, 1SP+EP-IDM reaches 2.91 m/s and 9.15 m, substantially closer to the ground truth behavior than the models trained with SIGReg, which produce near-zero forward displacement. The planning trajectories in Fig. 7 confirm that, with 1SP+EP-IDM, we are able to keep a coherent bipedal gait throughout the episode, while 1SP+SIG and MSP+SIG cannot keep such walking gait.

Table 2: Walker2D locomotion under latent iCEM (Pinneri et al., 2021). All metrics are averaged over ten trials. Final height is the torso height (m) at the end of the episode.
<table><tr><td>Model</td><td>Avg Velocity (m/s)</td><td>Displacement (m)</td><td>Final Height (m)</td></tr><tr><td>Ground truth (GT)</td><td>3.61</td><td>14.43</td><td>1.21</td></tr><tr><td> $1 \mathrm { S P } + \mathrm { S I G }$ </td><td>0.08</td><td>0.06</td><td>0.98</td></tr><tr><td> $\mathrm { M S P } + \mathrm { S I G }$ </td><td>0.46</td><td>0.66</td><td>0.85</td></tr><tr><td> $\mathrm { 1 S P + E P \mathrm { - } I D M }$ </td><td>2.91</td><td>9.15</td><td>0.92</td></tr></table>

![](images/2844449adfa6cb48ed949f8ec7d9fdcf540ebcffa964345fc8501589aa84b38a.jpg)  
Figure 7: Planning trajectory frames for Walker2D. Each row shows ten frames uniformly sampled from a 500-step closed-loop rollout under latent iCEM planning for the trained models. We select the best trial, with respect to the forward displacement, across the ten trials for the trained models.

## 6.4 Diagnostics

Figure 8 compares the local dynamics learned by 1SP+EP-IDM and MSP+SIG. 1SP+EP-IDM reproduces the ground-truth vector field near the upright equilibrium with low threestep prediction error, whereas MSP+SIG captures only parts of the field and exhibits larger, anisotropic error. Both models produce a latent goal cost with a minimum near the equilibrium, which CEM and GBP can exploit through finite-horizon optimization, while LQR requires an accurate local linearization.

Figure 9 shows the empirical region of attraction (ROA) and Lyapunov difference under the latent LQR controllers. The controller obtained from 1SP+EP-IDM stabilizes almost every evaluated initial condition and yields $\Delta V = V ( x _ { t + 1 } ) - V ( x _ { t } ) < 0$ around the equilibrium, providing an empirical certificate of local closed-loop stability. In contrast, the MSP+SIG controller stabilizes only a small neighborhood around the equilibrium.

Although MSP+SIG provides a terminal cost suitable for nonlinear planning (Figure 8), its local representation and predictor do not yield a latent stabilizing controller. Action reconstruction instead learns local dynamics from which a stabilizing controller can be designed, consistent with Section 5.

We evaluate whether action reconstruction remains effective for systems with no unstable modes using PointMaze (Table 3). We note that all action-reconstruction variants trained from scratch reach at least 90% CEM success and at least 70% LQR success.

![](images/7a69c634125f50f974912f43cf517ae09213d33d29b005ebf0766620189468d2.jpg)

![](images/3c7bd74ff66b26791d03c098775d07a530af641fd15735829cb10baace080817.jpg)

![](images/6dcdc69cec98e3ba225c848232dea9eecb03cca74239d933c1b306ecb2710057.jpg)

![](images/a203f18af16d113562550efdf804c6d106844660ef767ab366a48b2bb3b67af9.jpg)

![](images/c6523dc9c4e663c2e3e25a6f03fc4c10bc401c93175b631251f328670a611e9b.jpg)

![](images/d67f07d5dcda1d2ab70fd908ea95c26af599e7395ee160422191e9e83a3a1bb3.jpg)  
Figure 8: CartPole diagnostics. Left: Phase portraits of ground-truth and learned vector fields. Center: H = 3 open-loop prediction error across (θ, θ). Right: Zero-action planning cost.

![](images/e840794bd38fb8babf168060d12dcc495e17e64bd1fd364e21f93627f09e4df3.jpg)

![](images/293f30be7cfd39c3f09e8e09753337c1e1f39c886f420f446a11ecca5a9634c6.jpg)

![](images/3282ee797990e9241f1632ba16c2bedda73118270474ca275c8fee77dd621f01.jpg)

![](images/0d0ba49df724088439b6b9f01a09767ea4613ec90acea163a59289b8a0f07c14.jpg)  
Figure 9: Local stability on CartPole. First two panels: Empirical region of attraction of the learned LQR controller when deployed in the ground-truth system. The dashed line indicates the ground-truth LQR boundary. Last two panels: Lyapunov decrease $\Delta V = V ( x ^ { \prime } ) - V ( x )$ under the latent LQR controller.

## 7 Conclusion and Future Work

In this work, we demonstrated a fundamental limitation of next-step prediction combined with standard anti-collapse regularization (e.g., SIGReg) in JEPAs: these objectives do not necessarily preserve the control-theoretic information required to stabilize an open-loop unstable system. They can be minimized despite the encoder discarding controllable unstable modes. Therefore, the latent representation may not collapse on the training distribution, yet be insufficient for designing a stabilizing latent feedback controller.

To overcome this limitation, we proposed endpoint inverse dynamics (EP-IDM), which reconstructs the action sequence from the initial and final latent representations. For linear systems, we proved that exact reconstruction preserves the finite-horizon reachable subspace and prevents the collapse of reachable unstable modes, while the dominant eigenspace of the controllability Gramian converges to the controllable unstable subspace as the prediction horizon grows. Empirically, EP-IDM increases latent-LQR success from 0% to 100% on nonlinear CartPole, preserves the Walker2D limit cycle and supports successful planning with 9.15 m of average forward displacement over 500 time steps, and achieves strong planning performance on PointMaze.

Table 3: PointMaze navigation success rate (%).
<table><tr><td>Encoder</td><td>Objective</td><td>CEM (SR)</td><td>LQR (SR)</td><td>GBP (SR)</td></tr><tr><td>GT</td><td></td><td>100%</td><td>80%</td><td>90.0%*</td></tr><tr><td rowspan="6">From scratch</td><td>1SP + IDM</td><td>100%</td><td>70%</td><td>100%</td></tr><tr><td>MSP + MS-IDM</td><td>100%</td><td>80%</td><td>90.0%</td></tr><tr><td>MSP + EP-IDM</td><td>90.0%</td><td>80%</td><td>90.0%</td></tr><tr><td> $\mathrm { 1 S P + E P \mathrm { - } I D M }$ </td><td>100%</td><td>80%</td><td>80.0%</td></tr><tr><td> $\mathrm { M S P } + \mathrm { S I G }$ </td><td>90.0%</td><td>60%</td><td>100%</td></tr><tr><td> $\mathrm { 1 S P + I D M + M S \mathrm { - } I D M }$ </td><td>90.0%</td><td>80%</td><td>90.0%</td></tr><tr><td>DINOv2</td><td>MSP</td><td>50.0%</td><td>20%</td><td>80%</td></tr><tr><td>iBOT</td><td>MSP</td><td>100%</td><td>80%</td><td>80%</td></tr><tr><td>DINO-WM</td><td></td><td>80%</td><td>80%</td><td>60%</td></tr></table>

Because MuJoCo is non-differentiable, we estimate gradients with SPSA using two rollouts per optimization step (Spall, 1998).

Future work could combine our control-aware representation with latent-geometry objectives for efficient and robust planning, including bisimulation (Toso et al., 2026) and temporal straightening (Wang et al., 2026a).

## 8 Acknowledgments

Leonardo F. Toso and James Anderson thank Paul Lutkus and Professor Stephen Tu for insightful discussions during the early stages of this work. The authors also thank Professor Jean Ponce for his detailed comments and feedback on an earlier version of this manuscript. Leonardo F. Toso is funded by the Center for AI and Responsible Financial Innovation (CAIRFI) Fellowship and the Columbia Presidential Fellowship. James Anderson is partially funded by NSF grants EECS 2144634 and CNS 2535097 and the Center of AI Technology (CAIT) in collaboration with Amazon. This work was also supported in part by AFOSR under grant FA95502310139.

## References

Ames, A. D., Galloway, K., Sreenath, K., and Grizzle, J. W. (2014). Rapidly exponentially stabilizing control lyapunov functions and hybrid zero dynamics. IEEE Transactions on Automatic Control, 59(4):876–891.

Ames, A. D., Xu, X., Grizzle, J. W., and Tabuada, P. (2016). Control barrier function based quadratic programs for safety critical systems. IEEE transactions on automatic control, 62(8):3861–3876.

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629.

Balestriero, R. and LeCun, Y. (2025). Lejepa: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544.

Bardes, A., Ponce, J., and LeCun, Y. (2021). Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906.

Davis, C. and Kahan, W. M. (1970). The rotation of eigenvectors by a perturbation. iii. SIAM Journal on Numerical Analysis, 7(1):1–46.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. (2009). Imagenet: A largescale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee.

Ha, D. and Schmidhuber, J. (2018). World models. arXiv preprint arXiv:1803.10122, 2(3).

Haarnoja, T., Zhou, A., Abbeel, P., and Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. Pmlr.

Hafner, D., Lillicrap, T., Ba, J., and Norouzi, M. (2020). Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations.

Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., and Davidson, J. (2019). Learning latent dynamics for planning from pixels. In International conference on machine learning, pages 2555–2565. PMLR.

Hafner, D., Lillicrap, T. P., Norouzi, M., and Ba, J. (2021). Mastering atari with discrete world models. In International Conference on Learning Representations.

Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. (2023). Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104.

Hautus, M. (1969). Controllability and observability conditions of linear autonomous systems. Indagationes Mathematicae (Proceedings), 72(5):443–448.

Hu, Y., Wierman, A., and Qu, G. (2022). On the sample complexity of stabilizing lti systems on a single trajectory. Advances in Neural Information Processing Systems, 35:16989–17002.

Huang, H., LeCun, Y., and Balestriero, R. (2026). Llm-jepa: Large language models meet joint embedding predictive architectures. In International Conference on Learning Representations, volume 2026, pages 105717–105737.

Huang, S., Dossa, R. F. J., Ye, C., Braga, J., Chakraborty, D., Mehta, K., and Araújo, J. G. (2022). Cleanrl: High-quality single-file implementations of deep reinforcement learning algorithms. Journal of Machine Learning Research, 23(274):1–18.

Ivashkov, P., Balestriero, R., and Schölkopf, B. (2026). Sensorimotor world models: Perception for action via inverse dynamics. arXiv preprint arXiv:2606.20104.

Kuang, Y., Dagade, Y., Lidec, Q. L., Maes, L., Balestriero, R., and LeCun, Y. (2026). Lpwm: A case for sparse representations in world models. arXiv preprint arXiv:2608.22764.

LeCun, Y. (2022). A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62.

Levine, S., Kumar, A., Tucker, G., and Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643.

Lutkus, P., Wang, K., Lindemann, L., and Tu, S. (2025). Latent representations for control design with provable stability and safety guarantees. In 2025 IEEE 64th Conference on Decision and Control (CDC), pages 2937–2944. IEEE.

Maes, L., Lidec, Q. L., Scieur, D., LeCun, Y., and Balestriero, R. (2026). Leworldmodel: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312.

Marco, A., Baumann, D., Khadiv, M., Hennig, P., Righetti, L., and Trimpe, S. (2021). Robot learning with crash constraints. IEEE Robotics and Automation Letters, 6(2):1439–1446.

Mellinger, D. and Kumar, V. (2011). Minimum snap trajectory generation and control for quadrotors. In 2011 IEEE international conference on robotics and automation, pages 2520– 2525. Ieee.

Mhammedi, Z., Foster, D. J., Simchowitz, M., Misra, D., Sun, W., Krishnamurthy, A., Rakhlin, A., and Langford, J. (2020). Learning the linear quadratic regulator from nonlinear observations. Advances in Neural Information Processing Systems, 33:14532–14543.

Nath, D., Srinivasan, A., Yin, H., Jiang, R., Fang, J., and Chou, G. (2026). Pixels to proofs: Probabilistically-safe latent world model control via parallel conformal robust mpc. arXiv preprint arXiv:2606.15594.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al. (2023). Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193.

Parthasarathy, A., Kalra, N., Agrawal, R., LeCun, Y., Bounou, O., Izmailov, P., and Goldblum, M. (2025). Closing the train-test gap in world models for gradient-based planning. arXiv preprint arXiv:2512.09929.

Pinneri, C., Sawant, S., Blaes, S., Achterhold, J., Stueckler, J., Rolinek, M., and Martius, G. (2021). Sample-efficient cross-entropy method for real-time planning. In Conference on Robot Learning, pages 1049–1065. PMLR.

Sobal, V., Zhang, W., Cho, K., Balestriero, R., Rudner, T. G., and LeCun, Y. (2025). Learning from reward-free offline data: A case for planning with latent dynamics models. arXiv preprint arXiv:2502.14819.

Spall, J. C. (1998). An overview of the simultaneous perturbation method for efficient optimization. Johns Hopkins apl technical digest, 19(4):482–492.

Todorov, E., Erez, T., and Tassa, Y. (2012). Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026 5033. IEEE.

Toso, L. F., Shadunts, D., Lu, Y., Sharma, N., Zhan, D., Nguyen, N. H., and Anderson, J. (2026). Learning invariant visual representations for planning with joint-embedding predictive world models. arXiv preprint arXiv:2602.18639.

Toso, L. F., Ye, L., and Anderson, J. (2025). Learning stabilizing policies via an unstable subspace representation. In 2025 IEEE 64th Conference on Decision and Control (CDC), pages 7543–7550. IEEE

Wang, Y., Bounou, O., Zhou, G., Balestriero, R., Rudner, T. G., LeCun, Y., and Ren, M. (2026a). Temporal straightening for latent planning. arXiv preprint arXiv:2603.12231.

Wang, Z., Fang, K., and LeCun, Y. (2026b). Music-jepa: Learning a world model of sound from action. arXiv preprint arXiv:2607.22000.

Werner, S. W. and Peherstorfer, B. (2024). On the sample complexity of stabilizing linear dynamical systems from data. Foundations of Computational Mathematics, 24(3):955–987.

Zhang, W., Terver, B., Zholus, A., Chitnis, S., Sutaria, H., Assran, M., Balestriero, R., Bar, A., Bardes, A., LeCun, Y., et al. (2026). Hierarchical planning with latent world models. arXiv preprint arXiv:2604.03208.

Zhou, G., Pan, H., LeCun, Y., and Pinto, L. (2024). Dino-wm: World models on pre-trained visual features enable zero-shot planning. arXiv preprint arXiv:2411.04983.

Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., and Kong, T. (2021). ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832.

## A Appendix

This appendix is organized as follows. Appendix A.1 supplements Section 6 with additional experimental results and diagnostics. Appendix A.2 evaluates the trained world models using frozen pretrained encoders. Appendix A.3 provides the model architectures, datasets, optimization parameters, controller implementations, and additional action-reconstruction objectives used in our experiments. Appendix A.4 presents ablations on the action reconstruction, action-context length, and observation inputs. Appendix A.5 provides the construction and physical parameters of the linear examples from Section 4. Finally, Appendix A.6.1 states the supporting technical results, and Appendices A.6.2-A.6.4 provide the proofs of our theoretical guarantees.

## A.1 Additional Experimental Validation (Extending Section 6)

We provide additional evidence and illustration supporting the performance results in Section 6. We first examine representative CartPole and PointMaze trajectories and relate their closedloop performance to the geometry and local dynamics learned by each model. We then evaluate long-horizon prediction along feedback trajectories to distinguish accurate rollout prediction from the preservation of the unstable directions required for stabilization.

## A.1.1 CartPole

In Fig. 10 we complement the success rates in Section 6 with the sequences of frames in a randomly selected trial trajectory under the latent LQR controllers. The models trained with EP-IDM, both from scratch and with the frozen iBOT encoder, keep the pole upright throughout the rollout. In contrast, the prediction combined with SIGReg models, frozen iBOT without EP-IDM, and DINO-WM fail to stabilize the system.

Moreover, Figs. 11-12 show that the model trained from scratch with MSP+EP-IDM+SIG recovers the local vector field around the upright equilibrium, yields low prediction error around the equilibrium, and leads to a broad region of attraction with the expected decrease in the Lyapunov certificate. Figs. 13-14 depict similar performance when EP-IDM is applied through a learned projector on top of the pretrained iBOT. By comparison, the iBOT models without EP-IDM and DINO-WM have less accurate local dynamics, even when their terminal latent costs is minimized near the goal (see Figs. 15–18).

Table 4 reports the diagnostics for the CEM experiments. The inverse squared stabilization length highlights models that lose stability early from those that remain close to the goal over the planning horizon, while the final error measures terminal accuracy. These results emphasize that CEM can compensate for imperfect local dynamics through repeated nonlinear replanning, whereas LQR directly depends on the local model around the equilibrium.

## A.1.2 PointMaze

PointMaze is open-loop stable and therefore does not require recovery of an unstable subspace. We, nevertheless, check whether IDM retains task-relevant directions without degrading nonlinear planning for open-loop stable tasks. Fig. 19 shows that both EP-IDM models and MSP+SIG reach the goal under CEM, whereas DINO-WM fails on the displayed rollout.

The latent goal-distance fields in Fig. 20 explain this behavior. The models trained from scratch produce smooth fields that decrease toward the goal and broadly respect the maze geometry, providing CEM with an informative terminal cost. The DINO-WM field is less aligned with the navigation geometry, which can direct the planner toward states that appear

Table 4: CartPole CEM diagnostic metrics (action context 1). $1 / L ^ { 2 } \colon$ inverse squared mean stabilization length. FE: final error $\Vert { x } _ { T } \Vert$ . PR denotes a learned projector on top of the frozen encoder.
<table><tr><td>Encoder</td><td>Objective</td><td>CEM  $( 1 / L ^ { 2 } )$ </td><td>CEM (FE)</td></tr><tr><td colspan="2">GT</td><td>0.0003</td><td>0.30202</td></tr><tr><td rowspan="5">From scratch</td><td> $1 \mathrm { S P } + \mathrm { S I G }$ </td><td>0.0003</td><td>0.19956</td></tr><tr><td> $\mathrm { 1 S P + I D M + M S \mathrm { - } I D M }$ </td><td>0.0003</td><td>0.43608</td></tr><tr><td> $\mathrm { 1 S P + E P \mathrm { - } I D M }$ </td><td>0.0003</td><td>0.13683</td></tr><tr><td> $\mathrm { M S P } + \mathrm { S I G }$ </td><td>0.0003</td><td>0.07667</td></tr><tr><td> $\mathrm { M S P } + \mathrm { E P - I D M } + \mathrm { S I G }$ </td><td>0.0003</td><td>0.11985</td></tr><tr><td rowspan="5">DINOv2</td><td>1SP</td><td>0.0038</td><td>4.60613</td></tr><tr><td>MSP</td><td>0.0031</td><td>5.09216</td></tr><tr><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { S I G }$ </td><td>0.0003</td><td>0.19056</td></tr><tr><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { I D M }$ </td><td>0.0331</td><td>4.68120</td></tr><tr><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { E P } { \cdot } \mathrm { I D M }$ </td><td>0.0004</td><td>1.57507</td></tr><tr><td rowspan="6">iBOT</td><td>1SP</td><td>0.0003</td><td>0.09306</td></tr><tr><td> $\mathrm { 1 S P + P R \mathrm { - } E P \mathrm { - } I D M }$ </td><td>0.0004</td><td>0.47275</td></tr><tr><td>MSP</td><td>0.0073</td><td>3.49471</td></tr><tr><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { S I G }$ </td><td>0.0004</td><td>1.42928</td></tr><tr><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { I D M }$ </td><td>0.0567</td><td>5.11192</td></tr><tr><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { E P } { \cdot } \mathrm { I D M }$ </td><td>0.0003</td><td>0.62596</td></tr><tr><td colspan="2">DINO-WM</td><td>0.0005</td><td>0.9777</td></tr></table>

close in latent space but are not connected by a feasible path. Importantly, on this open-loop stable task, the goal is to shape the representation to obtain a smooth planning latent geometry rather than the recovery of unstable modes.

## A.1.3 Closed-Loop Prediction Error

Figure 21 compares prediction errors over long closedloop rollouts. The 1SP+EP-IDM model is close to the ground-truth trajectory over the entire horizon. The error of $\mathrm { M S P + E P - I D M }$ grows steadily, whereas combining multi-step prediction and EP-IDM with SIGReg keeps the error bounded. Hence, multistep training alone is not able to alleviate the rollout drift in an unstable system, i.e., small model errors can still be amplified by the dynamics. On the other hand, the representation regularization SIGReg im-

![](images/fe4b7a62d4830da724ed31d4eb63ce3b94c6ff01f46b74d5f42393cc8c8b67b1.jpg)  
Figure 21: k-step closed-loop prediction error along trajectories generated by the ground-truth LQR controller. Curves show the physical-state error between the ground-truth trajectory and the trajectory decoded from each latent rollout.

Table 5: CartPole control performance with frozen pretrained encoders. SR: success rate. MFS: mean fraction of steps within the success threshold. PR denotes a learned projector. DINO-WM (Zhou et al., 2024) is an external baseline.
<table><tr><td>Encoder</td><td>Objective</td><td>LQR (SR)</td><td>LQR (MFS)</td><td>CEM (SR)</td><td>GBP (SR)</td></tr><tr><td rowspan="5">DINOv2</td><td>1SP</td><td>0%</td><td>0.301</td><td>0%</td><td>0%</td></tr><tr><td>MSP</td><td>0%</td><td>0.232</td><td>0%</td><td>0%</td></tr><tr><td>1SP + PR-EP-IDM</td><td>0%</td><td>0.025</td><td>20%</td><td>0%</td></tr><tr><td>MSP + PR-SIG</td><td>0%</td><td>0.460</td><td>100%</td><td>0%</td></tr><tr><td>MSP + PR-IDM</td><td>0%</td><td>0.455</td><td>0%</td><td>0%</td></tr><tr><td rowspan="5">iBOT</td><td>MSP + PR-EP-IDM</td><td>0%</td><td>0.513</td><td>40%</td><td>10%</td></tr><tr><td>1SP</td><td>0% 0%</td><td>0.249</td><td>100%</td><td>80% 0%</td></tr><tr><td>MSP</td><td>0%</td><td>0.228</td><td>0% 100%</td><td></td></tr><tr><td>1SP + PR-EP-IDM</td><td>50%</td><td>0.263</td><td></td><td>80%</td></tr><tr><td>MSP + PR-SIG MSP + PR-IDM</td><td></td><td>0.571</td><td>80%</td><td>80%</td></tr><tr><td rowspan="4">DINO-WM</td><td></td><td>0%</td><td>0.405</td><td>0%</td><td>0%</td></tr><tr><td>MSP + PR-EP-IDM</td><td>100%</td><td>0.998</td><td>100%</td><td>0%</td></tr><tr><td></td><td>0%</td><td>0.019</td><td>40%</td><td>10%</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

proves the conditioning of the representation. This is complementary to the stability results in Section 6. The low rollout error in the landscape diagnosis supports control, but does not by itself guarantee that the representation preserves the unstable directions required by LQR.

## A.2 Pre-trained encoders

We evaluate whether IDM can preserve control-relevant information when the visual encoder is frozen and only a lightweight projector (PR) and the latent dynamics are trained.

Table 5 shows that the benefit of EP-IDM depends on the information and local geometry already present in the frozen latents. In particular, with iBOT, MSP+PR-EP-IDM achieves 100% LQR and CEM success. In contrast, none of the DINOv2 configurations yields a stabilizing LQR controller, although MSP+PR-SIG supports CEM. Hence, IDM can shape the projector to preserve control-relevant information captured by the frozen backbone, but cannot recover information that the backbone has discarded.

Figure 22 depicts the closed-loop LQR trajectories. For every DINOv2-based model, the CartPole state norm crosses the stabilization threshold and grows without bound, except for MSP for which $\| x _ { t } \| _ { 2 } \leq 1 0$ . In contrast, the latent controller obtained with MSP+iBOT+PR-EP-IDM drives the state norm below the threshold and keeps it near the upright equilibrium for the full 300-step rollout.

## A.3 Additional Details on the Implementation of our Experiments

In this section, we provide additional implementation details needed to reproduce the nonlinear visual-control experiments from Section 6. We first detail our world-model architecture, then specify the datasets and control parameters used for each task.

## A.3.1 Model Architecture

We next describe the encoder, latent predictor, and action decoders used across our experiments.

Visual encoder. For models trained from scratch, each RGB observation is divided into non-overlapping patches and processed by a ViT-Tiny encoder with four transformer blocks, three attention heads, and embedding dimension 192, similar to Ivashkov et al. (2026). The resulting class token is concatenated with the proprioceptive measurements and mapped to a 192-dimensional latent state by a two-layer projection head. Unless stated otherwise, we encode the difference between consecutive frames together with the proprioceptive measurements to emphasize motion (i.e., velocity) of the physical system in the latent space.

For the frozen visual encoder, we use either the DINOv2 ViT-S/14 or iBOT ViT-S/16 class token. The backbone parameters are frozen and a trainable two-layer projector maps the concatenated visual and proprioceptive features to the common 192-dimensional latent space. This construction follows the use of frozen visual features in DINO-WM (Zhou et al., 2024).

Latent predictor and action decoders. The latent predictor is an action-conditioned causal transformer with six blocks, 16 attention heads, hidden dimension 192, MLP dimension 2048, and dropout 0.1, also similar to the architecture adopted in Ivashkov et al. (2026). Actions are embedded by a two-layer MLP and modulate each transformer block through adaptive layer normalization. The prediction head is a two-layer MLP with hidden dimension 2048 and output dimension 192.

The one-step and multi-step action decoders are MLPs with two hidden layers of width 256 and ReLU nonlinearities. The endpoint decoder receives the initial and terminal latent states and uses the same hidden dimensions to reconstruct the complete length-H action sequence. Then, all action-reconstruction variants differ only in the latent information provided to the decoder.

Table 6: Task-specific observation and action dimensions. The frame skip is the number of simulator steps represented by one model transition.
<table><tr><td>Task</td><td></td><td>Image size Proprioception Action Frame skip</td><td></td><td></td><td>Training trajectories</td></tr><tr><td>CartPole</td><td> $1 2 8 \times 1 2 8$ </td><td>4</td><td>1</td><td>5</td><td>787</td></tr><tr><td>Walker2D</td><td> $6 4 \times 6 4$ </td><td>17</td><td>6</td><td>5</td><td>3,200</td></tr><tr><td>PointMaze</td><td> $6 4 \times 6 4$ </td><td>4</td><td>2</td><td>5</td><td>1,600</td></tr></table>

Optimization. We train all trainable modules jointly with AdamW. The learning rate is linearly warmed up during the first five epochs and then follows a cosine schedule. Gradients are clipped to unit norm. The coefficients of the active prediction, regularization, and actionreconstruction objectives are set to one. Table 7 reports the shared hyperparameters. All reported configurations use the same optimization budget.

Table 7: World-model training hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>Epochs</td><td>200</td></tr><tr><td>Batch size</td><td>64</td><td>Prediction horizon H</td><td>3</td></tr><tr><td>Initial learning rate</td><td>10-4</td><td>Minimum learning rate</td><td>10-6</td></tr><tr><td>Weight decay</td><td>10-3</td><td>Warm-up epochs</td><td>5</td></tr><tr><td>Gradient-norm threshold</td><td>1</td><td>Latent dimension</td><td>192</td></tr><tr><td>Random seed</td><td>42</td><td>Active loss weights</td><td>1</td></tr></table>

## A.3.2 Task and Dataset Details

We next describe the data collection and evaluation protocols for each task.

CartPole. We render the nonlinear CartPole system from a fixed camera and pair each image with the four-dimensional physical state. The dataset contains 985 trajectories, split into 787/99/99 training, validation, and test trajectories. To cover both uncontrolled and stabilizing behaviors, 75% of the trajectories are collected using passive $( \mathrm { i . e . , } a _ { t } = 0$ for all t) or uniformly random inputs, and the remaining 25% use LQR-based policies and reference-angle sweeps. During evaluation, each method is tested from ten initial conditions for 300 control steps. A trial is successful when the norm of the physical state remains below 0.7 after T steps. Actions are clipped to [-10, 10].

Walker2D. We use the Walker2D-v4 environment with 64 × 64 RGB observations, the 17- dimensional proprioceptive measurements, and six-dimensional actions. The training data combine equal numbers of trajectories generated by a random policy and a pretrained SAC policy (Huang et al., 2022; Haarnoja et al., 2018), yielding 3, 200 training, 400 validation, and 400 test trajectories after the split. Evaluation is conducted over ten trials of 600 steps.

PointMaze. We use the U-shaped PointMaze environment with 64 × 64 RGB observations, four-dimensional proprioception, and two-dimensional actions. The dataset contains 2, 000 trajectories of length 100, split into 1, 600/200/200 training, validation, and test trajectories. For each of ten trials, the point robot is initialized at a sampled start state and controlled to reach a trial-specific goal. Success happens when the planar distance to the goal is at most 0.5.

## A.3.3 Planning and Control Details

All planners optimize a terminal latent-space objective: the predicted terminal representation is matched to the representation of the desired goal observation. As in DINO-WM (Zhou et al., 2024), CEM keeps a Gaussian distribution over action sequences, evaluates sampled sequences with the learned predictor, and select the lowest-cost elite actions, and refines the distribution. We use the iCEM variant in Pinneri et al. (2021) for Walker2D with a SAC policy warm initialization for the distribution of actions. Table 8 provides the task-specific sampling budgets.

Table 8: CEM and improved-CEM planning hyperparameters. Here, “execute" denotes the number of actions applied before replanning.
<table><tr><td>Task</td><td>Planner</td><td>Horizon</td><td>Execute</td><td>Population</td><td>Elites</td><td>Iterations</td></tr><tr><td>CartPole</td><td>CEM</td><td>10</td><td>1</td><td>300</td><td>30</td><td>10</td></tr><tr><td>Walker2D</td><td>iCEM</td><td>3</td><td>1</td><td>200</td><td>20</td><td>5</td></tr><tr><td>PointMaze</td><td>CEM</td><td>25</td><td>25</td><td>300</td><td>30</td><td>10</td></tr></table>

For gradient-based planning (GBP), the action sequence is optimized directly through the differentiable latent rollout. We use Adam for 50 optimization steps at each replanning instant, with learning rate 0.1 and Gaussian action perturbations of standard deviation 0.05. The CartPole and PointMaze horizons are 10 and 25, respectively.

For latent LQR, we linearize the learned latent dynamics around the desired equilibrium by automatic differentiation and solve the corresponding discrete algebraic Riccati equation. We use $Q = I _ { 1 9 2 }$ in both tasks, with R = 1 for CartPole and $R = I _ { 2 }$ for PointMaze.

## A.3.4 Sketched Isotropic Gaussian Regularization (SIGReg)

For completeness, we define the SIGReg objective used to prevent total representation collapse by enforcing random one-dimensional projections of the latent distribution to match a standard isotropic Gaussian (Balestriero and LeCun, 2025).

$$
L _ { \mathrm { S I G } } ( \theta ) = \frac { 1 } { K } \sum _ { \ell = 1 } ^ { K } \left( \frac { \sum _ { j = 1 } ^ { n _ { \omega } } g ( \omega _ { j } ) \Big [ \left( \operatorname { R e } \hat { \varphi } _ { \ell } ( \omega _ { j } ) - \varphi _ { N } ( \omega _ { j } ) \right) ^ { 2 } + \left( \operatorname { I m } \hat { \varphi } _ { \ell } ( \omega _ { j } ) \right) ^ { 2 } \Big ] } { \sum _ { j = 1 } ^ { n _ { \omega } } g ( \omega _ { j } ) } \right) .
$$

where

$$
\hat { \varphi } _ { \ell } ( \omega ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } e ^ { \mathrm { i } \omega ( w _ { \ell } ^ { \top } z _ { i } ) } , \qquad \varphi _ { N } ( \omega ) = e ^ { - \omega ^ { 2 } / 2 } = : g ( \omega ) , \qquad w _ { \ell } ^ { \mathrm { ~ i . i . d . } } \mathrm { U n i f } \big ( \mathbb { S } ^ { d - 1 } \big ) .
$$

Here, K is the number of random projection directions, and $w _ { \ell } \in \mathbb { S } ^ { d - 1 }$ is the l-th random unit direction in latent space. Each $w _ { \ell }$ is obtained by drawing an i.i.d. Gaussian vector in $\mathbb { R } ^ { d }$ and normalizing it. Moreover, $z _ { i } = E _ { \theta } ( y _ { i } ) \in \mathbb { R } ^ { d }$ is the i-th latent vector in the current minibatch, obtained by pooling every “frame" from every window, for $i = 1 , \dots , M$ . The total number of pooled latents is $M = N ( H { + } 1 )$ , corresponding to all $H + 1$ frames from each of the N windows. Lastly, $\omega \in \mathbb { R }$ denotes the characteristic-function argument, and $\{ \omega _ { j } \} _ { j = 1 } ^ { n _ { \omega } }$ is a grid of $n _ { \omega }$ points evenly spaced in $[ - \omega _ { \mathrm { m a x } } , \omega _ { \mathrm { m a x } } ]$

## A.3.5 Additional Action Reconstruction Losses

In addition to the endpoint action reconstruction loss studied in Section 5, we empirically consider two alternative inverse-dynamics objectives. These objectives differ only in the latent information made available to the action decoder.

Multi-step action reconstruction. The multi-step decoder receives the complete latent trajectory and jointly reconstructs all actions applied within the window:

$$
D _ { \mathrm { M S - I D M } , \theta } ( z _ { t _ { \tau } } , \ L , \ L . . , z _ { t _ { \tau } + H } ) = \hat { \mathbf { a } } _ { \tau , H } \in \mathbb { R } ^ { m H } ,\tag{20}
$$

where $\hat { \mathbf { a } } _ { \tau , H } = [ \hat { a } _ { t _ { \tau } } ^ { \top } ~ \cdot \cdot ~ \hat { a } _ { t _ { \tau } + H - 1 } ^ { \top } ] ^ { \top }$ . In the pixel-based experiments, $D _ { \mathrm { M S - I D M } , \theta }$ is a multilayer perceptron that first concatenates the $H + 1$ latent states and then predicts the entire length-H action sequence in a single forward pass. In the linear-system examples, we use a linear decoder. The corresponding loss is

$$
\mathcal { L } _ { \mathrm { M S - I D M } } ( \theta ) = \mathbb { E } _ { \tau \sim \mathcal { D } } \left[ \frac { 1 } { H } \sum _ { t = 0 } ^ { H - 1 } \left. \left[ D _ { \mathrm { M S - I D M } , \theta } \left( z _ { t \tau } , \dots , z _ { t _ { \tau } + H } \right) \right] _ { t } - a _ { t _ { \tau } + t } \right. _ { 2 } ^ { 2 } \right] .\tag{21}
$$

In contrast to the endpoint action reconstruction, this decoder can use the intermediate representations to infer each action from local changes along the trajectory.

One-step action reconstruction. The one-step decoder is shared across time and reconstructs each action from a pair of consecutive latent states:

$$
D _ { \mathrm { I D M } , \theta } \big ( z _ { t _ { \tau } + t } , z _ { t _ { \tau } + t + 1 } \big ) = \hat { a } _ { t _ { \tau } + t } \in \mathbb { R } ^ { m } .\tag{22}
$$

Table 9: CartPole ablation over inverse-dynamics formulations for an encoder trained from scratch.
<table><tr><td>Objective</td><td>LQR (SR)</td><td>LQR (MFS)</td><td>CEM (SR)</td><td>GBP (SR)</td></tr><tr><td> $\mathrm { 1 S P + I D M + M S \mathrm { - } I D M }$ </td><td>100%</td><td>0.999</td><td>100%</td><td>100%</td></tr><tr><td> $\mathrm { 1 S P + E P \mathrm { - } I D M }$ </td><td>100%</td><td>0.999</td><td>100%</td><td>100%</td></tr><tr><td> $\mathrm { M S P + E P \mathrm { { - } I D M + S I G } }$ </td><td>100%</td><td>0.996</td><td>100%</td><td>100%</td></tr></table>

In our implementation, $D _ { \mathrm { I D M } , \theta }$ is a multilayer perceptron (MLP) applied independently to every adjacent latent pair. Its loss is

$$
\mathcal { L } _ { \mathrm { I D M } } ( \theta ) = \mathbb { E } _ { \tau \sim \mathcal { D } } \left[ \frac { 1 } { H } \sum _ { t = 0 } ^ { H - 1 } \left. D _ { \mathrm { I D M } , \theta } \left( z _ { t _ { \tau } + t } , z _ { t _ { \tau } + t + 1 } \right) - a _ { t _ { \tau } + t } \right. _ { 2 } ^ { 2 } \right] .\tag{23}
$$

This objective provides a local inverse-dynamics signal at each transition and uses the same decoder parameters for all k. This one-step action-reconstruction objective is the empirical finite-sample counterpart of the inverse-dynamics regularizer proposed by Ivashkov et al. (2026).

We emphasize that, empirically, we consider endpoint (EP-IDM), multi-step (MS-IDM), and one-step (IDM) action reconstruction both separately and jointly. A common loss encompassing all variants of action reconstruction considered in this work is given by

$$
\begin{array} { r } { \bar { \mathcal { L } } _ { \mathrm { I D M } } ( \theta ) = \lambda _ { 1 } \mathcal { L } _ { \mathrm { E P - I D M } } ( \theta ) + \lambda _ { 2 } \mathcal { L } _ { \mathrm { M S - I D M } } ( \theta ) + \lambda _ { 3 } \mathcal { L } _ { \mathrm { I D M } } ( \theta ) . } \end{array}\tag{24}
$$

The theoretical guarantees in Section 5 leverage the endpoint loss $\mathcal { L } _ { \mathrm { E P - I D M } }$ , whereas the two alternatives above are also included as empirical comparisons.

## A.4 Ablations

We ablate the action reconstruction formulation, action-context length, and observation inputs to identify the effect of these components in the obtained control performance.

Different inverse dynamics loss formulations. In Table 9, we compare the different variants for action reconstruction, i.e., EP-IDM, IDM, and MS-IDM, used in the main experiments (see Appendix A.3.5 for additional details). We note that EP-IDM and the joint one-step/multistep objective both produce perfect CartPole success with all three controllers. Adding SIGReg to multi-step prediction and endpoint reconstruction does not degrade performance.

Action context. Moreover, in Tables 10 and 11, we compare training with different action context lengths. Here, by action context, we mean the number of consecutive actions provided to the latent predictor when predicting the next latent. Here we consider action context one (the standard in our experiments) and five (the ablation) while holding the encoder and objective fixed. The trained-from-scratch action-reconstruction models do not benefit from more actions in the context length. However, the models trained with pretrained encoders indicate that a longer action context can actually help shape the representation for a better planning when using iBOT as the frozen encoder.

Observation inputs. In Table 12 we summarize the results when we remove the pixel observations (i.e., images) and compares MLP and transformer predictors using only the physical state (proprioceptive measurements). We note that EP-IDM succeeds with the transformer but not with the MLP under LQR, showing that representation sufficiency does not replace the need to fit an accurate local predictor. Moreover, in Table 13, we have the results when we remove the proprioceptive measurements (states) from the frozen-iBOT models. We note that performance drops to zero across all planners, indicating that the present image encoder and short temporal context do not reliably recover velocity information from pixels alone.

Table 10: CartPole success rates (%) with action contexts one and five. PR denotes a learned projector on top of the frozen encoder.
<table><tr><td></td><td></td><td colspan="2">LQR (SR)</td><td colspan="2">CEM (SR)</td><td colspan="2">GBP (SR)</td></tr><tr><td>Encoder</td><td>Objective</td><td></td><td>Context 1 Context 5</td><td>Context 1</td><td>Context 5</td><td></td><td>Context 1 Context 5</td></tr><tr><td>From scratch</td><td> $1 \mathrm { S P } + \mathrm { S I G }$ </td><td>0</td><td>0</td><td>100</td><td>100</td><td>90</td><td>90</td></tr><tr><td></td><td> $\mathrm { 1 S P + I D M + M S \mathrm { - } I D M }$ </td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td></td><td> $\mathrm { 1 S P + E P \mathrm { - } I D M }$ </td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td></td><td> $\mathrm { M S P } + \mathrm { S I G }$ </td><td>0</td><td>40</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>DINOv2</td><td>MSP</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>DINOv2</td><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { S I G }$ </td><td>0</td><td>0</td><td>100</td><td>60</td><td>0</td><td>0</td></tr><tr><td>iBOT</td><td>MSP</td><td>0</td><td>0</td><td>0</td><td>100</td><td>0</td><td>60</td></tr><tr><td></td><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { S I G }$ </td><td>50</td><td>40</td><td>80</td><td>90</td><td>80</td><td>60</td></tr></table>

Table 11: PointMaze success rates (%) with action contexts one and five.
<table><tr><td></td><td></td><td colspan="2">LQR (SR)</td><td colspan="2">CEM (SR)</td><td colspan="2">GBP (SR)</td></tr><tr><td>Encoder</td><td>Objective</td><td></td><td>Context 1 Context 5</td><td>Context 1</td><td>Context 5</td><td></td><td>Context 1 Context 5</td></tr><tr><td>From scratch</td><td> $\mathrm { 1 S P + I D M }$ </td><td>70</td><td>80</td><td>100</td><td>90</td><td>100</td><td>80</td></tr><tr><td></td><td> $\mathrm { 1 S P + E P \mathrm { - } I D M }$ </td><td>80</td><td>80</td><td>100</td><td>100</td><td>80</td><td>80</td></tr><tr><td></td><td> $\mathrm { 1 S P + I D M + M S \mathrm { - } I D M }$ </td><td>80</td><td>70</td><td>90</td><td>100</td><td>90</td><td>90</td></tr><tr><td></td><td> $\mathrm { M S P } + \mathrm { S I G }$ </td><td>60</td><td>80</td><td>90</td><td>70</td><td>100</td><td>60</td></tr><tr><td></td><td> $\mathrm { M S P + M S \mathrm { - } I D M }$ </td><td>80</td><td>80</td><td>100</td><td>100</td><td>90</td><td>90</td></tr><tr><td></td><td> $\mathrm { M S P + E P - I D M }$ </td><td>80</td><td>80</td><td>90</td><td>90</td><td>90</td><td>100</td></tr><tr><td>DINOv2</td><td>MSP</td><td>20</td><td>10</td><td>50</td><td>60</td><td>80</td><td>50</td></tr><tr><td>iBOT</td><td>MSP</td><td>80</td><td>80</td><td>100</td><td>80</td><td>80</td><td>60</td></tr></table>

Table 12: CartPole ablation: Action context one, only proprioceptive information (state).
<table><tr><td>Predictor</td><td>Objective</td><td>LQR (SR)</td><td>LQR (MFS)</td><td>CEM (SR)</td><td>CEM (held)</td><td>GBP (SR/held)</td></tr><tr><td rowspan="2">MLP</td><td> $\mathrm { M S P } + \mathrm { S I G }$ </td><td>20%</td><td>0.571</td><td>20%</td><td>20%</td><td>0% / 0%</td></tr><tr><td> $\mathrm { 1 S P + E P - I D M }$ </td><td>0%</td><td>0.204</td><td>100%</td><td>100%</td><td>0% / 0%</td></tr><tr><td rowspan="2">Transformer</td><td> $\mathrm { M S P } + \mathrm { S I G }$ </td><td>20%</td><td>0.461</td><td>0%</td><td>0%</td><td>0% / 0%</td></tr><tr><td> $\mathrm { 1 S P + E P - I D M }$ </td><td>80%</td><td>0.869</td><td>100%</td><td>100%</td><td>100% / 100%</td></tr></table>

Fixed-point consistency and encoder sensitivity. We also evaluate how the two frozen encoders respond to small physical perturbations in order to understand the reason why MSP+PR-EP-IDM with iBOT succeeds and with DINOv2 fails. Table 14 reports the ratio between consecutive latent and physical displacements for zero-action rollouts near the upright equilibrium. iBOT has a gain close to one across the evaluated perturbations, whereas DINOv2 amplifies the same perturbations by approximately 22-26 times. This is also consistent with the fixed-point residuals in Table 15, namely, the predictor has difficulty representing the upright equilibrium state as a stationary latent point when small physical perturbations lead to disproportionately large latent perturbations

We also note that DINOv2 (Oquab et al., 2023) is trained on ImageNet (Deng et al., 2009)

Table 13: CartPole ablation with a frozen iBOT encoder and action context five, with and without proprioceptive inputs. PR denotes the learned projector. SR: success rate, held: fraction of trials in which stabilization is preserved.
<table><tr><td>Objective</td><td colspan="5">Proprioception LQR (SR) CEM (SR) CEM (held) GBP (SR) GBP (held)</td></tr><tr><td rowspan="2">MSP</td><td>Yes 0%</td><td>100%</td><td>100%</td><td>60%</td><td>40%</td></tr><tr><td>No</td><td>0% 0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td rowspan="2">MSP + PR-SIG</td><td>Yes</td><td>40%</td><td>90% 0%</td><td>90%</td><td>60% 60%</td></tr><tr><td>No</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td rowspan="2">MSP + PR-IDM</td><td>Yes</td><td>100% 80%</td><td>80%</td><td>70%</td><td>70%</td></tr><tr><td>No</td><td>0%</td><td>0% 0%</td><td>0%</td><td>0%</td></tr></table>

to maximally separate image representations, that is, it amplifies fine visual perturbations. In particular, a 0.02-unit physical perturbation of the CartPole state changes the image by only a few pixels, but DINOv2's CLS token will completely separate it in the latent space. On the other hand, iBOT (Zhou et al., 2021), trained with masked reconstruction, learns a smoother representation where nearby images map to close latents.

Table 14: Ablation on the encoder gain $\| \Delta z _ { t } \| / \| \Delta x _ { t } \|$ as a function of initial-state perturbation magnitude ‖xo]. We start from a perturbed upright equilibrium, the cart pole is stepped with zero action for 60 steps. We measure the mean latent displacement $\| \Delta z _ { t } \| = \| z _ { t } - z _ { t - 1 } \|$ divided by the mean physicalstate displacement $\| \Delta x _ { t } \| = \| x _ { t } - x _ { t - 1 } \|$
<table><tr><td>Encoder ||x₀|=0.005</td><td></td><td>∥|x₀||=0.01 ||x₀||=0.02</td><td>||x₀||=0.05</td><td>||x₀|=0.10</td><td>||x₀||=0.20</td></tr><tr><td>iBOT</td><td>1.07×</td><td>1.11×</td><td>1.07× 1.07×</td><td>1.02×</td><td>1.03×</td></tr><tr><td>DINOv2</td><td>25.45×</td><td>26.07×</td><td>26.38× 25.67×</td><td>21.83×</td><td>25.12×</td></tr></table>

## A.5 Additional Details on Examples of Section 4

This section provides details on the system matrices, physical parameters, and linear camera maps used in the examples of Section 4. In addition, we also provide here an additional motivating example of a synthetic open-loop stable system.

Linear camera maps. In all three examples, we observe the physical state through a fixed linear perceptual “sensor" or camera map $y _ { t } = C x _ { t }$ . It lifts the low-dimensional physical state into a redundant observation space and mixes its coordinates, so that dynamically meaningful directions are not presented to the encoder in an axis-aligned basis. For a state dimension n and observation dimension $p ,$ we draw $\bar { C } \in \mathbb { R } ^ { p \times n }$ using random seed zero, with $\bar { C } _ { i j } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , 1 )$ 2 and normalize each column, i.e., $C _ { : , j } = \bar { C } _ { : , j } / \| \bar { C } _ { : , j } \| _ { 2 }$

Synthetic, open-loop unstable system (Example 1). This example isolates the collapse of unstable modes in the smallest possible setting: a two-dimensional, single-input system with one stable mode and one controllable but open-loop unstable mode that must be retained by the learned representation for feedback stabilization. We first define

$$
\bar { A } = \mathrm { d i a g } ( 1 . 2 5 , 0 . 8 5 ) \mathrm { ~ a n d ~ } \bar { B } = \left[ \begin{array} { c } { { 1 } } \\ { { 0 . 6 } } \end{array} \right] .\tag{25}
$$

Both modes are therefore actuated by the single input. To avoid aligning the modal coordinates

Table 15: Fixed-point residual of the learned predictor at the encoded upright CartPole equilibrium under zero action. The residual measures the one-step drift from this encoded equilibrium. PR denotes a learned projector on top of the frozen encoder.
<table><tr><td>Encoder Objective</td><td> $\| P ( z _ { \mathrm { e q } } , 0 ) - z _ { \mathrm { e q } } \|$ </td></tr><tr><td>DINOv2 1SP</td><td>5.569</td></tr><tr><td>MSP</td><td>7.215</td></tr><tr><td> $1 \mathrm { S P } + \mathrm { P R } \mathrm { - } \mathrm { E P } \mathrm { - } \mathrm { I D M }$ </td><td>0.4026</td></tr><tr><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { S I G }$ </td><td>2.515</td></tr><tr><td> $\mathrm { M S P + P R \mathrm { \mathrm { - } E P \mathrm { - } I D M } }$ </td><td>0.343</td></tr><tr><td>iBOT 1SP</td><td>0.059</td></tr><tr><td>MSP</td><td>0.101</td></tr><tr><td> $1 \mathrm { S P } + \mathrm { P R } \mathrm { - } \mathrm { E P } \mathrm { - } \mathrm { I D M }$ </td><td>0.091</td></tr><tr><td> $\mathrm { M S P } + \mathrm { P R } { \cdot } \mathrm { S I G }$ </td><td>0.641</td></tr><tr><td> $\mathrm { M S P + P R \mathrm { \mathrm { - } E P \mathrm { - } I D M } }$ </td><td>0.129</td></tr></table>

with the physical state coordinates, we apply the similarity transformation

$$
T _ { \vartheta } = [ \cos \vartheta \ :  - \sin \vartheta ] \ : , \ : \vartheta = 0 . 6 , \ : A = T _ { \vartheta } \bar { A } T _ { \vartheta } ^ { - 1 } , \ : \mathrm { a n d } \ : B = T _ { \vartheta } \bar { B } .\tag{26}
$$

This yields the following system matrices:

$$
A = { \left[ \begin{array} { l l } { 1 . 1 2 2 } & { 0 . 1 8 6 } \\ { 0 . 1 8 6 } & { 0 . 9 7 8 } \end{array} \right] } \in \mathbb { R } ^ { 2 \times 2 } { \mathrm { ~ a n d ~ } } B = { \left[ \begin{array} { l } { 0 . 4 8 7 } \\ { 1 . 0 6 0 } \end{array} \right] } \in \mathbb { R } ^ { 2 \times 1 } .\tag{27}
$$

The six-dimensional linear camera map is

$$
C = \left[ \begin{array} { l l } { 0 . 0 6 9 } & { - 0 . 0 8 1 } \\ { 0 . 3 5 3 } & { 0 . 0 6 4 } \\ { - 0 . 2 9 5 } & { 0 . 2 2 2 } \\ { 0 . 7 1 8 } & { 0 . 5 8 1 } \\ { - 0 . 3 8 8 } & { - 0 . 7 7 6 } \\ { - 0 . 3 4 3 } & { 0 . 0 2 5 } \end{array} \right] \in \mathbb { R } ^ { 6 \times 2 } .\tag{28}
$$

This system has one unstable and one stable mode with eigenvalues and eigenvectors as follows:

$$
\lambda _ { u } = 1 . 2 5 \ ( \mathbf { u n s t a b l e } ) , \ v _ { u } = \left[ 0 . 8 2 5 \right] \ \mathrm { a n d } \ \lambda _ { s } = 0 . 8 5 \ ( \mathbf { s t a b l e } ) , \ v _ { s } = \left[ { - 0 . 5 6 5 } \right] .\tag{29}
$$

The initial condition is given by $x _ { 0 } = \alpha _ { u } v _ { u } + \alpha _ { s } v _ { s } .$ where $\alpha _ { u } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \sigma _ { u } ^ { 2 } )$ with $\sigma _ { u } = 0 . 0 1$ and $\alpha _ { s } \stackrel { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \sigma _ { s } ^ { 2 } )$ with $\sigma _ { s } = 0 . 3$ . We use an asymmetric initial-condition distribution, with larger variance along the stable mode and smaller variance along the unstable mode. We note that this choice reflects realistic data collection near an unstable equilibrium point (Levine et al., 2020). That is, a policy that avoids catastrophic failure will naturally spend most of its time exploring near-equilibrium conditions or stable-mode excursions and will therefore rarely encounter the large deviations that reveal unstable directions.

Each rollout has length $T = 5$ . With probability 0.3, an episode is passive, meaning that $a _ { t } \equiv 0$ . Otherwise, the actions are sampled independently according to $a _ { t } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( 0 , \sigma _ { a } ^ { 2 } )$ , where $\sigma _ { a } = 0 . 0 2$

Synthetic, open-loop stable system (Example 2). This example provides an open-loop stable control case for the previous example: it preserves the same observation, actuation, and data asymmetry as in the previous example while replacing the unstable mode by a stable mode, thereby separating representation collapse from its consequences for closed-loop stability.

$$
\lambda _ { s 1 } = 0 . 2 5 \mathrm { ~ ( s t a b l e ~ 1 ) } , \ v _ { s 1 } = \left[ 0 . 8 2 5 \right] \ \mathrm { a n d } \ \lambda _ { s 2 } = 0 . 8 5 \mathrm { ~ ( s t a b l e ~ 2 ) } , \ v _ { s 2 } = \left[ { - 0 . 5 6 5 } \right] .\tag{30}
$$

The initial-condition distribution, episode rollout length, per-step action distribution, and all training hyperparameters are kept the same as in the previous example.

In particular, we construct this example using the same ${ \bar { B } } ,$ rotation $T _ { \vartheta }$ , and linear camera map as in the previous example, but replace the eigenvalue 1.25 by 0.25 in the diagonal of A. We then have

$$
\bar { A } = \mathrm { d i a g } ( 0 . 2 5 , 0 . 8 5 ) \ \mathrm { a n d } \ A = T _ { \vartheta } \bar { A } T _ { \vartheta } ^ { - 1 } = \left[ \begin{array} { c c } { { 0 . 4 4 1 } } & { { - 0 . 2 8 0 } } \\ { { - 0 . 2 8 0 } } & { { 0 . 6 5 9 } } \end{array} \right] ,\tag{31}
$$

while $B = [ 0 . 4 8 7 ~ 1 . 0 6 0 ] ^ { \top }$ and C is the $6 \times 2$ matrix reported above. Hence, the two examples differ only in whether the low-variance mode is unstable or contractive.

Figure 23 provides an open-loop stable system baseline to compare with the examples in Fig. 2, where all modes decay even without corrective feedback (i.e., they are all stable modes) and preserving a stable direction for latent stabilization is unnecessary. Hence, the standard latent world-model objective (i.e., solving Eq. (6) with SIGReg) can still work. For this open-loop stable system in Fig. 23, EP-IDM recovers the dominant stable direction.

Linearized CartPole (Example 3). Let p denote the cart position, θ the pole angle measured from the upright equilibrium, and a the horizontal force applied to the cart. For the state $\boldsymbol { x } = [ p , \dot { p } , \theta , \dot { \theta } ] ^ { \top }$ , the linearized continuous-time dynamics are

$$
\dot { x } = A _ { c } x + B _ { c } a ,\tag{32}
$$

where

$$
A _ { c } = \left[ \begin{array} { c c c c } { 0 } & { 1 } & { 0 } & { 0 } \\ { 0 } & { 0 } & { - \frac { m g } { M } } & { 0 } \\ { 0 } & { 0 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { \frac { \left( M + m \right) g } { M \ell } } & { 0 } \end{array} \right] \mathrm { ~ a n d ~ } B _ { c } = \left[ \begin{array} { c } { 0 } \\ { \frac { 1 } { M } } \\ { 0 } \\ { - \frac { 1 } { M \ell } } \end{array} \right] .\tag{33}
$$

We use cart mass $M = 1 . 0 \AA$ pole mass $m = 0 . 1$ , pole-length parameter $\ell = 0 . 5$ and gravitational acceleration $g \ : = \ : 9 . 8 1$ . The continuous-time spectrum is $\{ 0 , 0 , + \omega , - \omega \}$ , where $\omega =$ $\sqrt { ( M + m ) g / ( M \ell ) }$ . We then discretize $( A _ { c } , B _ { c } )$ using an exact zero-order hold with sampling time $d _ { t } = 0 . 0 2$ , which gives

$$
A = { \left[ \begin{array} { l l l l } { 1 . 0 0 0 } & { 0 . 0 2 0 } & { - 0 . 0 0 0 2 } & { - 0 . 0 0 0 0 } \\ { 0 . 0 0 0 } & { 1 . 0 0 0 } & { - 0 . 0 1 9 6 } & { - 0 . 0 0 0 2 } \\ { 0 . 0 0 0 } & { 0 . 0 0 0 } & { 1 . 0 0 4 3 } & { 0 . 0 2 0 0 } \\ { 0 . 0 0 0 } & { 0 . 0 0 0 } & { 0 . 4 3 2 3 } & { 1 . 0 0 4 3 } \end{array} \right] } \in \mathbb { R } ^ { 4 \times 4 } { \mathrm { ~ a n d ~ } } B = { \left[ \begin{array} { l } { 0 . 0 0 0 2 } \\ { 0 . 0 2 0 0 } \\ { - 0 . 0 0 0 4 } \\ { - 0 . 0 4 0 1 } \end{array} \right] } \in \mathbb { R } ^ { 4 \times 1 } .\tag{34}
$$

We note that the underlying repeated eigenvalue at one corresponds to the cart's free-integrator block, while the upright pole contributes one stable and one unstable eigenvalue. Lastly, the eight-dimensional linear camera map is given by

$$
C = \left[ \begin{array} { c c c c c } { 0 . 0 4 6 } & { - 0 . 0 6 8 } & { 0 . 2 5 6 } & { 0 . 0 5 6 } \\ { - 0 . 1 9 5 } & { 0 . 1 8 5 } & { 0 . 5 2 2 } & { 0 . 5 0 3 } \\ { - 0 . 2 5 6 } & { - 0 . 6 4 7 } & { - 0 . 2 5 0 } & { 0 . 0 2 2 } \\ { - 0 . 8 4 7 } & { - 0 . 1 1 2 } & { - 0 . 4 9 9 } & { - 0 . 3 8 9 } \\ { - 0 . 1 9 8 } & { - 0 . 1 6 2 } & { 0 . 1 6 5 } & { 0 . 5 5 3 } \\ { - 0 . 0 4 7 } & { 0 . 6 9 9 } & { - 0 . 2 6 6 } & { 0 . 1 8 7 } \\ { 0 . 3 2 9 } & { 0 . 0 4 8 } & { - 0 . 2 9 8 } & { - 0 . 4 8 9 } \\ { - 0 . 1 6 7 } & { 0 . 1 1 3 } & { - 0 . 4 0 4 } & { - 0 . 1 1 1 } \end{array} \right] \in \mathbb { R } ^ { 8 \times 4 } .\tag{35}
$$

Fig. 24 complements the closed-loop control performance from Fig. 2 with the Lyapunov certificate guarantee. For both unstable systems, the controllers learned with SIGReg produce regions in which $\Delta V > 0$ , showing that the ground-truth Lyapunov function increases after one closed-loop step under the learned latent LQR controller. This arises because the learned representations do not preserve the unstable dynamics required to design an accurate stabilizing feedback controller. In contrast, the controllers learned with EP-IDM yield $\Delta V < 0$ throughout the evaluated region, except at the equilibrium where $\Delta V = 0$

## A.6 Proofs of Our Theoretical Guarantees

This section provides the complete proofs of the theoretical results presented in the main text. Notation. Given a matrix $M \in \mathbb { R } ^ { n \times n }$ , we denote the spectral norm as $\| M \| _ { 2 }$ , the smallest singular value as $\sigma _ { \operatorname* { m i n } } ( M )$ , the spectral radius as $\rho ( M )$ , and the spectrum of M as spec(M). The smallest and largest eigenvalues are $\lambda _ { \operatorname* { m i n } } ( M )$ and $\lambda _ { \operatorname* { m a x } } ( M )$ . For a subspace S, Ps denotes its orthogonal projector and $\mathcal { S } ^ { \perp }$ its orthogonal complement. We write $h \lesssim g$ when $h \leq C g$ for a constant $C > 0$

## A.6.1 Supporting Lemmas

We first introduce the auxiliary results used throughout the subsequent proofs.

Lemma 2.1. Let T be a linear map and suppose that the support of a random vector u contains an open subset of its ambient space. If an affine map D satisfies $D ( T u + c ) = u$ almost surely, then T has full column rank.

Proof. Write $D ( v ) = L v + b$ . Exact reconstruction gives

$$
( L T - I ) u + ( L c + b ) = 0\tag{36}
$$

on an open set. An affine function that vanishes on an open set is identically zero, and hence $L T = I$ . Therefore, T admits a left inverse and has full column rank. □

Theorem 3 (Davis-Kahan theorem (Davis and Kahan, 1970)). Let M and $\hat { M } = M + E$ be symmetric matrices. Let Φ be a subspace of M associated with an eigenvalue cluster separated from the remaining spectrum of M by a gap $\delta > 0$ , and let $\hat { \Phi }$ be the corresponding subspace of M. Then, $i f \parallel E \parallel _ { 2 } \leq \delta / 2$ , it holds that

$$
\| \sin \Theta ( \hat { \Phi } , \Phi ) \| _ { 2 } \leq \frac { 2 \| E \| _ { 2 } } { \delta } .\tag{37}
$$

## A.6.2 Proof of Lemma 1

Suppose first that $A v = \lambda v , | \lambda | \geq 1$ , and $F v = 0$ . We consider two initial conditions that differ by v. As their initial representations agree, any deterministic controller using only the latent history applies the same first action to both systems. Inductively, the two latent histories, and hence the applied actions, remain identical because

$$
F A ^ { t } v = \lambda ^ { t } F v = 0 .\tag{38}
$$

Their state difference is therefore $A ^ { t } v = \lambda ^ { t } v$ , which cannot converge to zero, unless $v = 0$ . Thus, a single latent controller cannot stabilize both initial conditions. This proves necessity and is equivalent to the stated Popov-Belevitch-Hautus rank condition in (9).

On the other hand, suppose that (A, F) is detectable. By Assumption 1, (A, B) is stabilizable, so there exists a feedback gain K such that $A + B K$ is Schur stable. Moreover, the detectability guarantees the existence of an observer gain L such that $A - L F$ is Schur stable. We consider the observer-based latent controller

$$
\begin{array} { r } { \hat { x } _ { t + 1 } = A \hat { x } _ { t } + B a _ { t } + L \big ( \tilde { z } _ { t } - F \hat { x } _ { t } \big ) , } \\ { a _ { t } = K \hat { x } _ { t } . \qquad } \end{array}\tag{39}
$$

For the estimation error $e _ { t } : = x _ { t } - \hat { x } _ { t }$ , we have $e _ { t + 1 } = ( A - L F ) e _ { t }$ . Moreover,

$$
x _ { t + 1 } = ( A + B K ) x _ { t } - B K e _ { t } .\tag{40}
$$

Thus, the joint dynamics of $( x _ { t } , e _ { t } )$ are block triangular with Schur-stable diagonal blocks A + BK and $A - L F$ . Hence, both $e _ { t }$ and $x _ { t }$ converge to zero. This demonstrates sufficiency and completes the proof.

## A.6.3 Proof of Theorem 1

By unrolling the dynamics for H steps, we obtain

$$
x _ { t + H } = A ^ { H } x _ { t } + \mathcal { C } _ { H } \mathbf { a } _ { t }\tag{41}
$$

where we recall that $\mathbf { a } _ { t , H } = [ a _ { t } ^ { \top } ~ \cdot \cdot ~ a _ { t + H - 1 } ^ { \top } ] ^ { \top }$ . Hence, we have

$$
z _ { t + H } = F A ^ { H } x _ { t } + F \mathcal { C } _ { H } \mathbf { a } _ { t , H } + b .\tag{42}
$$

Fix an initial state. The initial representation and the autonomous contribution to the final representation are then fixed $( \mathrm { i . e . , } F A ^ { H } x _ { t } + b )$ , while the only endpoint variation caused by the actions is $F \mathcal { C } _ { H } \mathbf { a } _ { t , H }$ . As the linear decoder reconstructs $\mathbf { a } _ { t , H }$ exactly on a conditional support containing an open set, Lemma 2.1 implies that $F { \mathcal { C } } _ { H }$ has full column rank.

Now we take $v \in \ker ( F ) \cap \mathcal { R } _ { H }$ . Therefore, there exists a vector q such that $v = \mathcal { C } _ { H } q$ . Hence, we have that

$$
\begin{array} { r } { F \mathcal { C } _ { H } q = F v = 0 . } \end{array}\tag{43}
$$

As $F { \mathcal { C } } _ { H }$ has full column rank, $q = 0$ , and therefore $v = 0$ . This proves (16). Therefore, if $\Phi _ { > 1 } \subseteq { \mathcal { R } } _ { H }$ , the same conclusion holds for every vector in $\Phi _ { \geq 1 }$ . Finally, full column rank of $F { \mathcal { C } } _ { H } \in \mathbb { R } ^ { d \times m H }$ requires m ${ \cal H } \le \mathrm { m i n } \{ d , n \}$

## A.6.4 Proof of Theorem 2

For simplicity, we consider systems with no eigenvalues on the unit circle. Thus, $\Phi _ { > 1 }$ is the strictly unstable subspace $\Phi , \mathrm { i . e . , } \Phi _ { > 1 } : = \Phi$ . In an orthonormal basis adapted to the right unstable invariant subspace Φ and its orthogonal complement $\Phi ^ { \perp }$ , we write

$$
A = \left[ \begin{array} { l l } { A _ { u } } & { \Delta } \\ { 0 } & { A _ { s } } \end{array} \right] , \qquad B = \left[ \begin{array} { l } { B _ { u } } \\ { B _ { s } } \end{array} \right] .\tag{44}
$$

We make the following assumptions. For each $j \geq 0$ , let $\Gamma _ { j }$ denote the unstable block of $A ^ { j } B ,$ so that

$$
A ^ { j } B = \left[ \underset { A _ { s } ^ { j } B _ { s } } { \Gamma _ { j } } \right] .\tag{45}
$$

Assumption 2. There exist constants $\alpha > 1 , \beta \in ( \rho ( A _ { s } ) , 1 ) , \gamma \geq \alpha$ , and $C _ { 1 } , C _ { 2 } , C _ { 3 } \geq 1$ such that

$$
\sigma _ { \operatorname* { m i n } } ( A _ { u } ) \geq \alpha , \ \| A _ { s } ^ { j } \| _ { 2 } \leq C _ { 1 } \beta ^ { j } , \ \| A _ { u } ^ { j } \| _ { 2 } \leq C _ { 2 } \gamma ^ { j } , \ \| \Gamma _ { j } \| _ { 2 } \leq C _ { 3 } \gamma ^ { j } , \forall j \geq 0 , a n d \gamma \beta < \alpha ^ { 2 } .\tag{46}
$$

We emphasize that under Assumptions 2, there exists $\alpha > 1 , \beta \in ( \rho ( A _ { s } ) , 1 ) , \gamma \geq \alpha$ , such that every direction in $\Phi$ expands by at least a factor α at each open-loop step, whereas the stable dynamics contract at rate $\beta .$ The condition $\gamma \beta < \alpha ^ { 2 }$ ensures that the coupling between the unstable and stable responses grows strictly slower than the unstable contribution.

Assumption 3. The pair $\left( A _ { u } , B _ { u } \right)$ is controllable, and the action transmitted through the coupling ∆ does not cancel the direct unstable input response. In particular, there exist $c _ { u } > 0$ and $H _ { 0 } \geq 1$ such that

$$
\lambda _ { \operatorname* { m i n } } ( \Pi _ { u , H } ) \geq c _ { u } \alpha ^ { 2 H } , \ f o r \ H \geq H _ { 0 } .\tag{47}
$$

For block-diagonal dynamics, $i . e . , \Delta \equiv 0$ , this lower bound follows directly from controllability $o f \left( A _ { u } , B _ { u } \right)$ and Assumption 2.

The constants $c _ { u } , c _ { s }$ , and $c _ { u s }$ have a direct control-theoretic interpretation. The constant $c _ { u }$ measures the smallest action-induced variation generated along any unstable direction, i.e., a larger $c _ { u }$ means that even the least excitable unstable direction becomes distinguishable over a shorter horizon. The constant $c _ { s }$ bounds the total action-induced variation that can accumulate within the stable subspace, which remains bounded because the stable dynamics contract. Finally, $c _ { u s }$ quantifies the interaction between the action-induced unstable and stable responses. A larger $c _ { u s }$ allows stronger mixing between these responses at finite horizons, but Assumption 2 ensures that this interaction grows more slowly than the unstable contribution $c _ { u } \alpha ^ { 2 H }$

Moreover, under Assumptions 2 and 3, there exist constants $c _ { s } , c _ { u s } > 0$ such that, for all sufficiently large H, we have

$$
\lambda _ { \operatorname* { m i n } } ( \Pi _ { u , H } ) \geq c _ { u } \alpha ^ { 2 H } , \| \Pi _ { s , H } \| _ { 2 } \leq c _ { s } , \mathrm { ~ a n d ~ } \| \Pi _ { u s , H } \| _ { 2 } \leq c _ { u s } \sum _ { j = 0 } ^ { H - 1 } ( \gamma \beta ) ^ { j } ,\tag{48}
$$

where $\Pi _ { u , H } , \Pi _ { s , H }$ , and $\Pi _ { u s , H }$ are the unstable, stable, and cross Gramian blocks, respectively.

Lemma 2.2. Suppose that the assumptions preceding Theorem 2 hold. Then, the Gramian blocks satisfy

$$
\begin{array} { r } { \lambda _ { \operatorname* { m i n } } ( \Pi _ { u , H } ) \geq c _ { u } \alpha ^ { 2 H } , } \end{array}
$$

$$
\begin{array} { r c l } { { } } & { { } } & { { \| \Pi _ { s , H } \| _ { 2 } \le c _ { s } , } } \\ { { } } & { { } } & { { } } \\ { { } } & { { } } & { { \| \Pi _ { u s , H } \| _ { 2 } \le c _ { u s } \displaystyle \sum _ { j = 0 } ^ { H - 1 } ( \gamma \beta ) ^ { j } . } } \end{array}\tag{49}
$$

Therefore, the unstable block grows exponentially, the stable block remains bounded, and the cross block grows strictly more slowly than the unstable block whenever $\gamma \beta < \alpha ^ { 2 }$

Proof. The first inequality follows from Assumption 3. The remaining bounds follow from $\| \Gamma _ { j } \| _ { 2 } \le C _ { 3 } \gamma ^ { j }$ and $\| A _ { s } ^ { j } \| _ { 2 } \leq C _ { 1 } \beta ^ { j }$ . In particular

$$
\| \Pi _ { s , H } \| _ { 2 } \leq \sum _ { j = 0 } ^ { H - 1 } \| A _ { s } ^ { j } B _ { s } \| _ { 2 } ^ { 2 } \leq C _ { 1 } ^ { 2 } \| B _ { s } \| _ { 2 } ^ { 2 } \sum _ { j = 0 } ^ { \infty } \beta ^ { 2 j } = \frac { C _ { 1 } ^ { 2 } \| B _ { s } \| _ { 2 } ^ { 2 } } { 1 - \beta ^ { 2 } } ,
$$

$$
\| \Pi _ { u s , H } \| _ { 2 } \leq \sum _ { j = 0 } ^ { H - 1 } \| \Gamma _ { j } \| _ { 2 } \| A _ { s } ^ { j } B _ { s } \| _ { 2 } \leq C _ { 3 } C _ { 1 } \| B _ { s } \| _ { 2 } \sum _ { j = 0 } ^ { H - 1 } ( \gamma \beta ) ^ { j } .\tag{50}
$$

The proof is completed when we absorb the constant factors into $c _ { s }$ and $c _ { u s }$

Definition 2 (Subspace distance). Let $\Xi , \hat { \Xi } \in \mathbb { R } ^ { n \times r }$ be orthonormal bases for two r-dimensional subspaces Φ and $\hat { \Phi }$ , respectively. Their principal angles $\theta _ { 1 } , \ldots , \theta _ { r } \in [ 0 , \pi / 2 ]$ are defined by

$$
\sigma _ { i } ( \Xi ^ { \top } \hat { \Xi } ) = \cos ( \theta _ { i } ) , \ f o r \ a l l \ i = 1 , \dots , r ,\tag{51}
$$

where $\sigma _ { i } ( \cdot )$ denotes the i-th singular value. We define

$$
\sin \Theta ( \hat { \Phi } , \Phi ) : = \mathrm { d i a g } \big ( \sin ( \theta _ { 1 } ) , \dots , \sin ( \theta _ { r } ) \big ) .\tag{52}
$$

The spectral-norm distance between the two subspaces is then

$$
\big \| \mathrm { s i n } \Theta ( \hat { \Phi } , \Phi ) \big \| _ { 2 } : = \operatorname* { m a x } _ { i = 1 , \ldots , r } \mathrm { s i n } ( \theta _ { i } ) = \big \| P _ { \hat { \Phi } } - P _ { \Phi } \big \| _ { 2 } ,\tag{53}
$$

where $P _ { \hat { \Phi } }$ and $P _ { \Phi }$ are the orthogonal projectors onto $\hat { \Phi }$ and Φ, respectively.

We proceed by writing

$$
\Pi _ { H } = \underbrace { \left[ \Pi _ { u , H } 0 \right] } _ { M _ { 1 } } + \underbrace { \left[ \Pi _ { u s , H } ^ { \top } \Pi _ { s , H } \right] } _ { M _ { 2 } } .\tag{54}
$$

The top r-dimensional eigenspace of $M _ { 1 }$ is exactly Φ, and its eigengap is at least $c _ { u } \alpha ^ { 2 H }$ . Moreover, Lemma 2.2 yields

$$
\| M _ { 2 } \| _ { 2 } \leq \| \Pi _ { u s , H } \| _ { 2 } + \| \Pi _ { s , H } \| _ { 2 } \leq c _ { u s } \sum _ { j = 0 } ^ { H - 1 } ( \gamma \beta ) ^ { j } + c _ { s } .\tag{55}
$$

As $\gamma \beta < \alpha ^ { 2 }$ , the ratio $\| M _ { 2 } \| _ { 2 } / ( c _ { u } \alpha ^ { 2 H } )$ converges to zero. It is therefore smaller than $1 / 2$ for all sufficiently large H. We then proceed, by applying Theorem 3 to write

$$
\| \sin \Theta ( \hat { \Phi } , \Phi ) \| _ { 2 } \leq \frac { 2 \| M _ { 2 } \| _ { 2 } } { c _ { u } \alpha ^ { 2 H } } \leq \frac { 2 \left( c _ { u s } \sum _ { j = 0 } ^ { H - 1 } ( \gamma \beta ) ^ { j } + c _ { s } \right) } { c _ { u } \alpha ^ { 2 H } } .\tag{56}
$$

It remains to make the convergence rate explicit. We then set $q = \gamma \beta$ . Under Assumptions 2 and 3, for all sufficiently large $H$ we can write

$$
\big \| \sin \Theta ( \hat { \Phi } , \Phi ) \big \| _ { 2 } \leq 2 \left( c _ { u s } \sum _ { j = 0 } ^ { H - 1 } ( \gamma \beta ) ^ { j } + c _ { s } \right) / \left( c _ { u } \alpha ^ { 2 H } \right) .\tag{57}
$$

The convergence rate is then given by the following three cases:

• If $q > 1$ , then it holds that

$$
\begin{array} { r } { \left\| \sin \Theta ( \hat { \Phi } , \Phi ) \right\| _ { 2 } \lesssim ( q ^ { H } + 1 ) / \alpha ^ { 2 H } . } \end{array}\tag{58}
$$

• If $q = 1$ , then it holds that

$$
\begin{array} { r } { \big \| \sin \Theta ( \hat { \Phi } , \Phi ) \big \| _ { 2 } \lesssim H / \alpha ^ { 2 H } . } \end{array}\tag{59}
$$

• If $q < 1$ , then it holds that

$$
\begin{array} { r } { \big \| \sin \Theta ( \hat { \Phi } , \Phi ) \big \| _ { 2 } \lesssim 1 / \alpha ^ { 2 H } . } \end{array}\tag{60}
$$

As $q < \alpha ^ { 2 }$ , each of these bounds converges to zero as $H \to \infty$ . Then, we have

$$
\big \| \sin \Theta ( \hat { \Phi } , \Phi ) \big \| _ { 2 } \to 0 \mathrm { ~ a s ~ } H \to \infty ,\tag{61}
$$

which completes the proof.

![](images/c22c88987215b8a1274f25ba22f7b6eb5a9119b48c25d093ac31244ec0fed56a.jpg)  
Figure 10: CartPole trajectory visualization for the trained models under the latent LQR controller

![](images/0e3deff3e321e3c709d38a0d52307ea0b2ba89d84e3ebbafe2aa742a8fd519b6.jpg)

![](images/819ee3871ada1964fef850e17450a2791e32eec503afa7508555b7b4bcd448df.jpg)

![](images/cd93e3567adaeff7d9eec4fccaf71ae7e7706c28dd52efe5483d7f2d7f371824.jpg)  
Figure 11: MSP+EP-IDM+SIG model on CartPole. Left: Phase portrait of ground-truth vs. learned vector fields. Center: H = 3 open-loop prediction error. Right: Zero-action planning cost.

![](images/354038cf17522b73548ac2745270087a1015216e8f22823f8708910cd3c7df8f.jpg)

![](images/65e0a52ad4e405cd152a9af8877b726a4e58294989abc5477d31bfb35b0f5a51.jpg)  
Figure 12: Local stability of MSP+EP-IDM+SIG on CartPole. Left: Empirical region of attraction (ROA) of the learned LQR policy in the GT environment. Dashed line: GT LQR boundary. Right: Lyapunov decrease $\Delta V = V ( x ^ { \prime } ) - V ( x )$ under the learned policy using the GT DARE solution.

![](images/c6f3ddd7f0ee06d9cddd324e49498ff416603e9004755d6a266ae3d19f513a58.jpg)

![](images/17bb3054bfe0687d3610d1282d625fcc4b64073c2d97e6dfd34ec509d1c771f8.jpg)

![](images/9361bd8d6107bf7b8ac6f7b06c4c5a27b80f81f553acdae8be37582d5e6e942e.jpg)  
Figure 13: MSP+IBOT+PR-EP-IDM model on CartPole. Panel descriptions are in Fig. 11.

![](images/fdb111020b9b5d8b8df02ffaa23d6bb41d591089ba17e553ca5fcbd0773df902.jpg)

![](images/85bccae2aab337dfe93ee8b54d6e2d950185952ee8c4f3432a4e89d30b64c62a.jpg)  
Figure 14: Local stability of MSP+IBOT+PR-EP-IDM on CartPole. Panel descriptions are the same as those in Fig. 12.

![](images/1053aaed48955661d79855319ad4d9497bbc334dc97173540356b823f6b70635.jpg)

![](images/0df2d33bdd95b027640f26be5308212c1abd26dcb545fe09ecad0ead31102a75.jpg)

![](images/972796898ae0b57b8bf8d8535f114aa1ae9c601d0ecd7bbf0361475a1449eb3b.jpg)  
Figure 15: MSP+IBOT+PR-SIG model on CartPole. Panel descriptions are the same as those in Fig. 11.

![](images/7cbe9926b4936695e6cfc4418e00fa701a4bff29c42fc43e244a9e14e5aea3f8.jpg)

![](images/93b4f4de2f7dd4b47feef52dcaa3eacb633b7f0931da4ef6fec93f475fb444fd.jpg)

![](images/e25b4963f12ec3a412d1ef80085dce0f19043f2c9a0f87165388e00cb530b834.jpg)  
Figure 16: MSP+IBOT model on CartPole. Panel descriptions are the same as those in Fig. 11.

![](images/7ea61a2b03a6aff1eeb18f84060b6bad904b5e9d5804795749ba258089de80b2.jpg)

![](images/f4e9db7ea649ff63012de3a64291712f1df97770c78483351af20b85451b7350.jpg)  
Figure 17: Local stability of MSP+IBOT on CartPole. Left: Empirical region of attraction (ROA) of the learned LQR policy in the GT environment. Panel descriptions are as those in Fig. 12.

![](images/8e86314806573de3973a564da6a196d6e2e246d3599df60e38d7b0b3bdbe7f15.jpg)

![](images/13d0601ad4f7898be50b27981c22bfc41d4d67edee4a2b2a3b893f1b0b39c16c.jpg)

![](images/a23c415aa2c49ffccb0b7d2bc819e0db1285f35831ee10fe621e22fbda29d9c3.jpg)  
Figure 18: DINO-WM model on CartPole. Panel descriptions are the same as those in Fig. 11.

![](images/18e3ff9935a757f5a980432e0c14f14862e71807ccc7b7a319b2d0c75f097ad5.jpg)  
Figure 19: PointMaze visualization for our trained models under CEM planning.

![](images/d4659246b9be020209a7415b03a3c8de40bb2c0a1f1eff42eb567d088b0653ae.jpg)  
Figure 20: Latent goal-distance fields $\| z - z _ { \mathrm { g } } \|$ across the PointMaze U-maze for four trained world models.

![](images/c2d7b9bd42ac5be2fd7c72f79be1ac2e67636f42af63594777a58c04e6d11003.jpg)  
MSP + DINOv2 MSP + DINOv2 + PR-IDM 1SP + DINOv2 MSP + DINOv2 + PR-EP-IDM MSP + DINOv2 + PR-SIG DINO-WM

![](images/6f3d3e17415aa9478b8623617cd73650f64d86b87aae81c60ccf89d8c6c81a54.jpg)  
Figure 22: CartPole state norm∥xt2 under the learned latent LQR controller over 300 steps. Left: DINOv2. Right: iBOT.

![](images/6af72d514864f31b09a42778a4656660d8bed448444508192bf1db663ff9d9cc.jpg)

![](images/6bcc53870e5d9a12166019f3f8717815c535b85729e8a5985e2fe601298f4fab.jpg)

![](images/c7ea91a15c275c3a63139a7586ba3cdfc35ad4e81e2051bf20c8ce327ddc798c.jpg)  
Figure 23: Synthetic open-loop stable LDS. This example considers a two-dimensional open-loop stable system. Panels descriptions are the same as those in Fig. 2.

![](images/f9c4402dcde8a1d5ba42df1d90d8332dfe6eb8fecb7b9b4a021c9ed9f45f11f0.jpg)

![](images/32bcd957e57a714cdc99475fc1d99fb9fb510c1f4081b6e9d0acdb9e583cce9a.jpg)

![](images/041b007c1f8cd6b2054db744fef1064c13f7de92ee7622e7eac685272b6a2f5b.jpg)

![](images/f00804df1e07eb736c0caf03f178331b2f8a28157f94d150268494aac1b607e3.jpg)  
Figure 24: Sign of $\Delta V = V ( x ^ { \prime } ) - V ( x )$ after one closed-loop step under the learned latent LQR controller, where $V ( x ) = x ^ { \top } S x$ and $S \succ 0$ is the ground-truth Lyapunov function matrix. We also depict the $\Delta V = 0$ contour (separatrix), i.e., the boundary separating states where the learned controller decreases V (blue) from those where it does not (red).