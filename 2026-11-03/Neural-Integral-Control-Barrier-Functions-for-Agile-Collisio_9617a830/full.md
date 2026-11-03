# Neural Integral Control Barrier Functions for Agile Collision Avoidance

Dasari Raj Vamsi Indian Institute ofInformation Technology and Management, Gwalior

Vrushabh Zinage California Institute of Technology

Efstathios Bakolas The University of Texas at Austin

Rohan Chandra University of Virginia

IMT 2022035@IIITM.AC.IN

VRUSHABH.ZINAGE@CALTECH.EDU

BAKOLAS@AUSTIN.UTEXAS.EDU

ROHANCHANDRA@VIRGINIA.EDU

## Abstract

Control barrier functions (CBFs) are a standard way to keep a planner inside a safe set, but they assume control affine dynamics, a barrier function given in closed form, and a known nominal controller. Integral control barrier functions (ICBFs) remove the first assumption and absorb input constraints, yet they still assume a given nominal controller and a barrier function. In this work, we present an approach to learn all three from data, using neural ordinary differential equations (ODEs) for the dynamics, imitation for the nominal controller, and supervised regression for the barrier function. The learned dynamics, nominal controller, and the barrier function are used to build the ICBF filter constraint analytically. Furthermore, the barrier is trained on viability rather than geometry, meaning that a state counts as safe only if some emergency maneuver can still avoid the constraint under the actuator limits, a correction that geometric labels miss entirely and that grows with speed. Actuator limits then enter the online program as hard bounds while the learned barrier rows may be relaxed, which keeps the program solvable at every state and the actuator limits exact, and we bound how far the input it returns can drift from the one the true models would give. On agile racing with a 6-D dynamic bicycle, the filter removes all 12 unsafe episodes from 200 held out layouts at an unchanged success rate and above 100 Hz on one CPU core, holds the input constraint on every episode of a benchmark where a jointly trained baseline holds it on 53.38%, and keeps that advantage under large errors in the plant friction.

## 1. Introduction

Autonomous vehicles (Hussain and Zeadally, 2018), factory robots, and multi robot teams (Beltrame et al., 2018) must respect hard state and actuator constraints while a performance objective drives them toward the limits of the hardware (Wabersich et al., 2023). The prevailing architecture for this is the safety filter: a nominal planner proposes an input, and a small optimization problem projects that input onto the set of inputs that keep a barrier function nonnegative. With CBFs this projection is a quadratic program (QP) that can be solved at control rates (Ames et al., 2019).

The CBF-QP is normally stated under three assumptions. First, the dynamics must be control affine, since otherwise the filter becomes a sequence of nonlinear programs that are expensive to solve and difficult to analyze (Son and Nguyen, 2019; Seo et al., 2022; Agrawal and Sreenath, 2017). For example, a vehicle with saturating tire forces, a steering slew limit and motor lag is not control affine. Second, a valid barrier must be known in closed form. For an obstacle field it is usually taken to be a signed distance, but the set of states from which a collision can still be avoided is fixed by the dynamics and the actuator limits rather than by the geometry of the obstacles. Learned barriers such as Srinivasan et al. (2020) relax this requirement but remain restricted to control affine systems without input constraints. Third, a stabilizing feedback law $k ( { \pmb x } )$ for the full nonlinear model is assumed, which is meant to be similar to the original problem. Input constraints compound all three, since the QP need not stay feasible once u is confined to a compact set. Feasibility (Zeng et al., 2021; Xiao et al., 2023a,b; Liu et al., 2023a) and input constraints (Agrawal and Panagou, 2021; Breeden and Panagou, 2021, 2023; Black and Panagou, 2023) have both been treated, but under the above three assumptions.

ICBFs (Ames et al., 2020) resolve the first difficulty together with the input constraint. The input is lifted into the state, driven by an integral law $\dot { \pmb u } = \phi ( { \pmb x } , { \pmb u } ) + { \pmb v }$ , and the QP solves for the auxiliary input v. The construction applies to general nonlinear systems and encodes state and input constraints in one scalar function. Current ICBF theory, however, assumes that $\phi$ (assembled from a nominal controller), the dynamics, and the barrier are defined apriori. The open question is: can ICBF-based safety be ensured for systems with unknown dynamics, unknown ICBF formulation, and an unknown nominal controller?

This paper proposes a modular approach that learns all three from data while keeping the ICBF construction analytic. We learn the state transition dynamics $F _ { \theta }$ with a Neural ODE (Chen et al., 2018), a nominal controller $\pi \theta$ by imitation of an unconstrained planner, and a state barrier $h _ { \theta } ( { \pmb x } )$ by supervised regression. Everything the filter needs is then differentiated out of those three networks, so the online cost is one small quadratic program per control step (Section 4), and no part of the safety requirement is learned as a penalty term. What decides whether such a filter is usable is not the learning itself but two choices inside the construction, which we study in detail.

At high speeds, a state outside an obstacle can also be unsafe, because braking distance and steering authority, not the shape of the obstacle, decide whether a collision is still avoidable. We therefore call a state safe only if at least one emergency maneuver, simulated through $F _ { \theta }$ with the actuator limits in the loop, keeps the vehicle safe. Furthermore, actuator limits are physics and cannot be violated, and so, actuator range and rate limits become bounds on the decision variable.

## Main Contributions.

1. A complete learning based realization of the ICBF safety filter for systems whose dynamics, nominal controller and barrier are all unknown, with every ICBF quantity obtained analytically from the three learned models rather than assumed (Section 4).

2. The online program is feasible and input invariant (Proposition 1). An error bound (Theorem 6) further shows that the input computed from the learned models approaches the one the same construction would produce from the true dynamics, controller, and barrier function, and identifies how much authority the input retains over the barrier as the quantity that governs the remaining gap.

We evaluate on a kinematic vehicle and a quadrotor carrying a suspended pendulum, against SABLAS (Qin et al., 2022) on its own double integrator benchmark, and on agile racing with a 6-D dynamic single track model with F1TENTH parameters (O’Kelly et al., 2020). An analytical HOCBF-QP (Xiao and Belta, 2022) written for a kinematic model that the vehicle does not obey is unsafe on 29 to 35% of episodes, because the barrier it enforces is certified for dynamics other than the ones in force. Filtering an MPPI planner (Williams et al., 2016) with the learned ICBF at 4.0 m/s instead leaves 0 of 200 held out layouts unsafe against 12 for the planner alone $( p = 0 . 0 0 0 5$ , paired exact McNemar test), at an unchanged success rate and with every parameter frozen before those layouts were generated.

## 2. Related work

Input constraints and ICBFs. Once u is confined to a compact set, a CBF-QP can lose feasibility. Input constrained CBFs restrict the safe set to the part that bounded inputs can hold (Agrawal and Panagou, 2021), with extensions to high relative degree and robustness (Breeden and Panagou, 2021, 2023), online adaptation (Black and Panagou, 2023) and learned feasibility constraints (Xiao et al., 2023b), but all keep u as the decision variable of a control affine model with a barrier in closed form. ICBFs (Ames et al., 2020) instead lift the input into the state through the integral law $\pmb { \dot { u } } = \phi ( \pmb { x } , \pmb { u } ) + \pmb { v }$ , so that state and input constraints become constraints on one augmented system, the dynamics need not be control affine, and the program is solved for v. Later work made the construction robust to disturbances and extended it to higher relative degree through a disturbance observer (Zinage et al., 2023), and handled input delay through predictor feedback with a closed form feasibility condition for joint state and input constraints (Kiss et al., 2026). Throughout that line of work the dynamics, the barrier and the nominal controller inside $\phi$ are given in closed form, whereas we keep the ICBF construction unchanged and supply all three from data.

Neural ICBFs. Learning has entered ICBF based control only recently. Decentralized neural ICBFs for multi agent navigation (Zinage et al., 2024) learn $h _ { \theta } ( { \pmb x } , { \pmb u } )$ jointly with a policy by penalizing violations of the ICBF condition in the training loss, recover safety with an embedded MPC based ICBF, and assume that the agent dynamics and a nominal controller defining $\phi$ (an LQR or PID law) are known a priori. Neural CBFs for input limited systems outside the integral formulation, trained against saturating counterexamples (Liu et al., 2023b) or as the value function of a nominal policy (So et al., 2024), likewise assume a known control affine model. Our work differs from these in assuming none of the dynamics, the barrier or the nominal controller, in training the barrier on viability through the learned dynamics rather than on geometry or on a penalized barrier condition, and in enforcing actuator limits as hard bounds of the online program while only the learned barrier rows may be relaxed.

Learned barriers, and what they are trained on. Model based synthesis (Ames et al., 2019; Zinage et al., 2023) requires the constraint set in closed form. Data driven alternatives replace it with a network trained from sampled states (Dawson et al., 2022; Taylor et al., 2020a; Zinage and Bakolas, 2023; Qin et al., 2021a; Ahmadi et al., 2019; Yu et al., 2023), from expert trajectories (Robey et al., 2020; Boffi et al., 2021), or jointly with a policy (Qin et al., 2021b; Meng et al., 2021; Wang et al., 2021). Across this literature the labels are geometric, so the learned set is a smoothed obstacle indicator rather than a set from which the constraint can still be met, and that distinction is what viability labeling addresses. Episodic CBF learning (Cheng et al., 2019; Wang et al., 2018; Taylor et al., 2020b), networks trained to emulate a CBF controller (Yaghoubi et al., 2020) and BarrierNet (Xiao et al., 2021) all start from a valid CBF, and the latter two from known dynamics.

Where safety is enforced. SABLAS (Qin et al., 2022) jointly learns black box dynamics and a control affine safety filter, with safety and input compliance expressed as penalty terms in the training loss, where the optimizer trades them against the task objective so that neither is met exactly. We keep the three learning problems separate and move safety into the online program, which makes each component independently verifiable, makes the error propagation of Theorem 6 tractable, and removes the trade off between tracking and constraint satisfaction, a choice whose cost as well as its benefit is quantified in Section 6.1. Imitation learning (Kostrikov et al., 2019; Dadashi et al., 2020; Zheng et al., 2022; Torabi et al., 2018; Karnan et al., 2022) supplies the nominal controller but cannot encode state or input constraints, which is why the filter is needed at all.

## 3. Preliminaries and Problem Statement

Consider the general nonlinear controlled system

$$
{ \pmb { \dot { x } } } = F ( { \pmb x } , { \pmb u } ) , \qquad { \pmb x } ( 0 ) = { \pmb x } ^ { 0 }\tag{1}
$$

where $\pmb { x } \in \mathcal { X } \subset \mathbb { R } ^ { n } , \pmb { u } \in \mathcal { U } \subset \mathbb { R } ^ { m } , F : \mathcal { X } \times \mathcal { U }  \mathbb { R } ^ { n }$ is continuously differentiable, and X and U are compact. Let $b : \mathcal { X }  \mathbb { R }$ satisfy $b ( { \pmb x } ) > 0$ on $\mathrm { i n t } ( S _ { x } ) , b ( { \pmb x } ) = 0 \mathrm { o n } \delta S _ { \pmb x }$ and $b ( { \pmb x } ) < 0$ on $\mathcal { X } \setminus \mathcal { S } _ { x }$ , where $s _ { x } \subset \mathcal { X }$ is the safe set.

Definition 1 A function b as above is a Control Barrier Function (CBF) if there exists a $\kappa _ { \infty }$ func  
tion γ such that inf $\begin{array} { r } { \frac { \partial b } { \partial \pmb { x } } ( \pmb { x } ) F ( \pmb { x } , \pmb { u } ) \geq - \gamma ( b ( \pmb { x } ) ) } \end{array}$ for every $\mathbf { \boldsymbol { x } } \in \mathcal { X } .$ u∈U

If (1) is control affine, ${ \cal F } = f ( { \pmb x } ) + g ( { \pmb x } ) { \pmb u } .$ , safe inputs follow from the CBF-QP argmin ∥u − $k ( { \pmb x } ) \| ^ { 2 }$ subject to $L _ { f } b + L _ { g } b \mathbf { \delta } \geq - \gamma ( b )$ , with $k ( { \pmb x } )$ a stabilizing feedback controller. That program assumes control affine dynamics and a known k, and need not remain feasible under input constraints, both of which restrictions the integral formulation below removes.

## 3.1. Integral Control Barrier Functions (ICBFs)

Consider the integral feedback law $\dot { \pmb u } = \phi ( \pmb x , \pmb u )$ with $\pmb { u } ( 0 ) : = \pmb { u } ^ { 0 }$ , where $\phi : \mathcal { X } \times \mathcal { U }  \mathbb { R } ^ { m }$ is continuously differentiable. Combined with (1) this gives

$$
[ \dot { { \pmb x } } ^ { \top } , \dot { { \pmb u } } ^ { \top } ] = [ F ( { \pmb x } , { \pmb u } ) ^ { \top } , \phi ( { \pmb x } , { \pmb u } ) ^ { \top } ] , { \pmb x } ( 0 ) = { \pmb x } _ { 0 } , { \pmb u } ( 0 ) = { \pmb u } _ { 0 } .\tag{2}
$$

Let $\pmb { z } : = [ \pmb { x } ^ { \top } , \pmb { u } ^ { \top } ] ^ { \top }$ for the augmented state and $S = \{ ( \pmb { x } , \pmb { u } ) : \ h ( \pmb { x } , \pmb { u } ) \geq 0 \}$ for a safety set covering both state and input constraints. Forward invariance of S requires $\dot { h } ( z ) + \gamma ( h ( z ) ) \ge 0$ along every trajectory, which is enforced through an auxiliary input v in $\begin{array} { r } { \dot { \pmb { u } } = \phi ( \pmb { x } , \pmb { u } ) + \pmb { v } \ : ( \mathrm { A m e s } } \end{array}$ et al., 2020).

Definition 2 Let γ be a class K function. A continuously differentiable $h : \mathcal { X } \times \mathcal { U } \to \mathbb { R }$ with $S \ = \ \{ ( { \pmb x } , { \pmb u } ) \ : \quad h ( { \pmb x } , { \pmb u } ) \ \geq \ 0 \}$ is an Integral Control Barrier Function (ICBF) if, for every $( { \pmb x } , { \pmb u } ) \in \mathcal { S } , q ( { \pmb x } , { \pmb u } ) \le 0$ whenever $p ( { \pmb x } , { \pmb u } ) = 0$ , where $\begin{array} { r } { p ( \pmb { x } , \pmb { u } ) : = \left( \frac { \partial h ( \pmb { x } , \pmb { u } ) } { \partial \pmb { u } } \right) ^ { \top } } \end{array}$ and $q ( { \pmb x } , { \pmb u } ) : =$ $\begin{array} { r } { - \left( \frac { \partial h ( x , u ) } { \partial x } F ( { \pmb x } , { \pmb u } ) + \frac { \partial h ( { \pmb x } , { \pmb u } ) } { \partial { \pmb u } } \phi ( { \pmb x } , { \pmb u } ) + \gamma ( h ( { \pmb x } , { \pmb u } ) ) \right) } \end{array}$

Theorem 3 (Ames et al., 2020) Let ϕ(x, u) be an integralfeedback controller and let S be defined by an ICBF $h ( { \pmb x } , { \pmb u } )$ in the sense of Definition 2. Modify the integral controller to $\begin{array} { r } { \dot { \pmb u } = \phi ( \pmb { x } , \pmb u ) . } \end{array}$ + ${ \pmb v } ^ { \star } ( { \pmb x } , { \pmb u } )$ , where

$$
\pmb { v } ^ { \star } ( \pmb { x } , \pmb { u } ) = \underset { \pmb { v } \in \mathbb { R } ^ { m } } { \mathrm { a r g m i n } } \| \pmb { v } \| ^ { 2 } \quad s . t . \quad p ( \pmb { x } , \pmb { u } ) ^ { \top } \pmb { v } \geq q ( \pmb { x } , \pmb { u } ) .\tag{3}
$$

Ifthe resulting closed loop admits a solution on the interval considered, then S isforward invariant along itfor every initial condition $( { \pmb x } ^ { 0 } , { \pmb u } ^ { 0 } ) \in \mathcal { S }$

Following Ames et al. (2020), we do not learn a barrier over the joint state and input space. We learn a state barrier $h _ { \theta } ( { \pmb x } )$ , form the extended barrier $h _ { e , \theta } ( { \pmb x } , { \pmb u } ) = \dot { h } _ { \theta } ( { \pmb x } , { \pmb u } ) + \gamma _ { x } ( h _ { \theta } ( { \pmb x } ) )$ , and treat the actuator limits separately. The set $S _ { e } = \{ ( \pmb { x } , \pmb { u } ) : h _ { e , \theta } ( \pmb { x } , \pmb { u } ) \geq 0 \}$ is therefore not a Cartesian product $S _ { x } \times \mathcal { U } \colon$ it couples x and u through the dynamics and depends on how a given input moves the state. We now state the overall objective of this paper:

Problem 1 Given that the dynamics F of (1), a valid ICBF, and the nominal controller from which the integral law ϕ is assembled are all unknown a priori, design an integralfeedback law that enforces safety for (1) together with satisfaction of the input constraint $\mathbf { \pmb { u } } \in \mathcal { U } .$

## 4. Proposed approach

We learn, in order, (i) the dynamics $F ( { \pmb x } , { \pmb u } )$ using neural ODE learning techniques, (ii) a nominal controller in place of $k ( { \pmb x } )$ using imitation learning, and (iii) a state barrier $h _ { \theta } ( { \pmb x } )$ , and combine them into the safe controller (11), as summarized in Algorithm 1 in the appendix. More specifically, we learn the parameters for each module, $\Theta _ { F } , \Theta _ { \pi }$ and $\Theta _ { h }$ . The parameter sets are learned sequentially in this order as each learned parameter set is needed in the subsequent module. For instance, the barrier labels are generated by simulating backup maneuvers through the learned dynamics, which must therefore exist first. Throughout, we will denote $\theta : = ( \Theta _ { F } , \Theta _ { \pi } , \Theta _ { h } )$ as the collective parameter set. $F _ { \theta } , \pi _ { \theta }$ and $h _ { \theta }$ always denote the corresponding module after training, frozen. Objects assembled from those modules inherit the subscript, so the integral law built from $\pi _ { \theta }$ and $F _ { \theta }$ is written $\phi _ { \theta }$ and the extended barrier is written $h _ { e , \theta }$

Remark 4 (on the sequential nature of the learning approach) Each component is verifiable on its own against a measurable criterion, the error each contributes propagates analytically (Theorem $\delta ) ,$ , and the training objectives do not compete, because safety is a constraint of the online program rather than a term inside a differentiable loss.

## 4.1. Nonlinear system identification

System identification refers to estimating or deriving, analytically or from data, a functional form for the given system such that it can be differentiated or placed inside an optimizer.

We use Neural ODEs (Chen et al., 2018) to learn F in (1) from a dataset $\mathcal { D } _ { F } = \{ \Gamma _ { 1 } , . . . , \Gamma _ { N _ { \tau } } \}$ where each trajectory $\Gamma _ { j }$ is a sequence $\left\{ ( { \pmb x } _ { 0 } , { \pmb u } _ { 0 } ) , \dots , ( { \pmb x } _ { N } , { \pmb u } _ { N } ) \right\}$ with $\pmb { x } _ { k } = \pmb { x } ( t _ { k } ) , \pmb { u } _ { k } = \pmb { u } ( t _ { k } )$ The dataset is produced by open loop excitation rather than by logging a controller: initial states are drawn uniformly, actuator values are drawn uniformly from an admissible set and held piecewise constant with small jitter, and the resulting short trajectories are recorded at the control period $\Delta t$ Exciting the input box directly is what makes $\partial F / \partial \mathbf { { u } }$ identifiable.

Our simulation studies read the labels x˙ from the plant directly, whereas on hardware they would be differenced from the recorded states or the first term of (4) dropped in exchange for the second, which needs only states sampled at $\Delta t$ . The recorded data form two sets, isolated transitions $( { \pmb x } , { \pmb u } , \dot { { \pmb x } } )$ and short trajectories $\Gamma _ { j }$ , and we fit $\Theta _ { F }$ by minimizing

$$
\mathcal { L } _ { F } ( \boldsymbol { \Theta } _ { F } ) = \sum _ { ( \pmb { x } , \pmb { u } ) } \left\| F _ { \Theta _ { F } } ( \pmb { x } , \pmb { u } ) - \dot { \pmb { x } } \right\| _ { W _ { 1 } } ^ { 2 } + \lambda \sum _ { j = 1 } ^ { N _ { \tau } } \sum _ { k = 1 } ^ { K } \left\| \hat { \pmb x } _ { k } ^ { j } - { \pmb x } _ { k } ^ { j } \right\| _ { W _ { 2 } } ^ { 2 } ,\tag{4}
$$

where $\hat { \pmb x } _ { k } ^ { j } = \Phi _ { \Theta _ { F } } \big ( \hat { \pmb x } _ { k - 1 } ^ { j } , { \pmb u } _ { k - 1 } ^ { j } , \Delta t \big )$ with $\hat { \pmb x } _ { 0 } ^ { j } = { \pmb x } _ { 0 } ^ { j }$ , diagonal normalizations $W _ { 1 } , W _ { 2 }$ and weight $\lambda > 0$ , where $\Phi _ { \Theta _ { F } }$ is the one step transition map obtained by integrating $F _ { \Theta _ { F } }$ over $\Delta t$ with a fourth order Runge-Kutta scheme. The second term feeds the model its own predictions, so it penalizes drift accumulated over K steps rather than one step error. The transition map is used twice in the paper, here and in Section 4.3, where it is the simulator through which backup maneuvers are rolled out to generate the barrier labels.

## 4.2. Imitation learning for the nominal controller

Several expert nominal controllers exist but most unfortunately are intractable to run in realtime. It is much easier to learn a lighter weight controller that serves a realistic copy of the expert.

We start by generating a dataset $\mathcal { D } _ { c } : = \{ ( \boldsymbol { x } _ { i } ^ { \star } , \boldsymbol { u } _ { i } ^ { \star } ) \} _ { i = 1 } ^ { N _ { c } }$ by rolling the expert out on an obstacle free instance of the task from randomized initial states, with a fixed fraction of the executed actions perturbed so that the recorded states leave the expert’s own closed loop distribution. That perturbation matters here more than in ordinary behavior cloning, because the filter will itself drive the system off that distribution whenever it intervenes.

We then fit $\Theta _ { \pi }$ by minimizing $\begin{array} { r } { \mathcal { L } _ { c } ( \Theta _ { \pi } ) = \mathbb { E } _ { ( \pmb { x } ^ { \star } , \pmb { u } ^ { \star } ) \sim \mathcal { D } _ { c } } \big [ \ell \big ( \pi _ { \Theta _ { \pi } } ( \pmb { x } ^ { \star } ) , \pmb { u } ^ { \star } \big ) \big ] } \end{array}$ , where ℓ is a per sample discrepancy between the predicted and the demonstrated action, in our case a squared error normalized by the actuator ranges. Cloning an unconstrained expert also avoids fitting the sharp corners of a saturated policy, which feedforward networks approximate poorly, and the filter would be required in unseen states in any case. The nominal controller is moreover consumed through its Jacobian rather than its output, since $\partial \pi _ { \boldsymbol { \theta } } / \partial \boldsymbol { x }$ is what builds the integral law. Following Remark 3 of Ames et al. (2020) with $\pi _ { \theta }$ in the role of $k ,$ that law is

$$
\phi _ { \theta } ( { \pmb x } , { \pmb u } ) = \frac { \partial \pi _ { \theta } } { \partial { \pmb x } } ( { \pmb x } ) F _ { \theta } ( { \pmb x } , { \pmb u } ) + \frac { \alpha _ { \phi } } { 2 } ( \pi _ { \theta } ( { \pmb x } ) - { \pmb u } ) ,\tag{5}
$$

with gain $\alpha _ { \phi } > 0$ , the Jacobian being evaluated by automatic differentiation through $\pi _ { \theta }$ . Note that $\phi _ { \theta }$ depends on two of the three blocks, $\Theta _ { \pi }$ through $\pi _ { \theta }$ and $\Theta _ { F }$ through $F _ { \theta }$ . An external planner may be substituted for $\pi \theta \cdot$ , in which case $\begin{array} { r } { \phi _ { \theta } = \frac { \alpha _ { \phi } } { 2 } ( { \pmb u } _ { \mathrm { n o m } } - { \pmb u } ) } \end{array}$ , and Section 6.3 reports both.

Lemma 5 (Tracking error dynamics induced by ϕ) Let $F _ { \theta }$ and $\pi _ { \theta }$ be continuously differentiable and consider the nominalflow ${ \dot { \pmb x } } = F _ { \boldsymbol \theta } ( { \pmb x } , { \pmb u } )$ under the unfiltered integral law $\dot { \pmb u } = \phi _ { \theta } ( { \pmb x } , { \pmb u } ) o f \left( 5 \right)$ Then ${ \mathbf e } : = { \pmb u } - \pi _ { \theta } ( { \pmb x } )$ satisfies $\dot { \mathbf { e } } = - \frac { \alpha _ { \phi } } { 2 } \mathbf { e } ,$ so $\| \mathbf { e } ( t ) \| = \| \mathbf { e } ( 0 ) \| e ^ { - \alpha _ { \phi } t / 2 }$ . Under the filtered law $\dot { \pmb { u } } = \phi _ { \theta } + \pmb { v } ^ { \star }$ the same cancellation leaves $\begin{array} { r } { \dot { \mathbf { e } } = - \frac { \alpha _ { \phi } } { 2 } \mathbf { e } + \pmb { v } ^ { \star } } \end{array}$

Proof $\begin{array} { r } { \dot { \mathbf { e } } = \dot { \pmb { u } } - \frac { \partial \pi _ { \theta } } { \partial \pmb { x } } \dot { \pmb { x } } = \frac { \partial \pi _ { \theta } } { \partial \pmb { x } } F _ { \theta } ( \pmb { x } , \pmb { u } ) + \frac { \alpha _ { \phi } } { 2 } ( \pi _ { \theta } ( \pmb { x } ) - \pmb { u } ) - \frac { \partial \pi _ { \theta } } { \partial \pmb { x } } F _ { \theta } ( \pmb { x } , \pmb { u } ) = - \frac { \alpha _ { \phi } } { 2 } \mathbf { e } } \end{array}$ , since the Jacobian terms cancel. The cancellation is exact under $F _ { \theta }$ , which is the model (5) is built from, and holds for the true plant only up to the mismatch $\| F - F _ { \theta } \|$

## 4.3. Viability labeled Neural ICBFs

We learn a state barrier $h _ { \theta } ( \pmb { x } ) : \mathcal { X }  \mathbb { R }$ and construct the extended barrier in its general, non affine form,

$$
h _ { e , \theta } ( { \pmb x } , { \pmb u } ) = \frac { \partial h _ { \theta } } { \partial { \pmb x } } ( { \pmb x } ) F _ { \theta } ( { \pmb x } , { \pmb u } ) + \alpha h _ { \theta } ( { \pmb x } ) , \qquad p _ { \theta } ( { \pmb x } , { \pmb u } ) ^ { \top } = \frac { \partial h _ { e , \theta } } { \partial { \pmb u } } = \frac { \partial h _ { \theta } } { \partial { \pmb x } } \frac { \partial F _ { \theta } } { \partial { \pmb u } } ,\tag{6}
$$

which reduces to $\begin{array} { r } { p _ { \theta } ^ { \top } = \frac { \partial h _ { \theta } } { \partial \mathbfit { x } } g ( \pmb { x } ) } \end{array}$ when the dynamics happen to be control affine. Both derivatives in (6) are taken by automatic differentiation through the learned models, so no closed form for $F$ or h is required anywhere. The network is trained by regression on labeled states,

$$
\mathcal { L } _ { h } ( \Theta _ { h } ) = \sum _ { i } w _ { i } \left( h _ { \Theta _ { h } } ( x _ { i } ) - y _ { i } \right) ^ { 2 } ,\tag{7}
$$

and everything turns on how $y _ { i }$ is defined. A geometric label, $y _ { i } = + 1$ outside the obstacles and −1 inside, describes where the system is at an instant rather than whether the constraint can still be met, so we replace it with a viability label defined as follows. Fix a finite library M of backup maneuvers and a horizon $T .$ Simulate each $\mu \in \mathcal { M }$ from $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ through the learned dynamics $F _ { \theta }$ with the actuator model in the loop, so that steering slew and motor lag are respected, and let $c _ { \mu } ( \pmb { x } _ { i } )$ be the minimum clearance to the constraint along that rollout. The label is

$$
y _ { i } = \operatorname { t a n h } \Bigl ( { \textstyle { \frac { 1 } { \kappa } } } \operatorname* { m a x } _ { \mu \in { \mathcal { M } } } c _ { \mu } ( { \pmb x } _ { i } ) \Bigr ) ,\tag{8}
$$

with scale $\kappa > 0$ . A state is safe exactly when some admissible maneuver keeps the system clear for $T$ seconds, which is a finite horizon, finite library inner approximation of the viability kernel. Two properties follow immediately and distinguish (8) from a worst case test over a single fixed policy. Enlarging M can only increase the maximum, so the labeled safe set grows monotonically toward the true viability kernel and the statistic converges as $| { \mathcal { M } } |$ grows, and a state inside the constraint has $c _ { \mu } < 0$ for every $\mu ,$ so that no geometrically unsafe state is ever labeled safe. Both properties are verified empirically in Section 6.2, where they also serve to fix the horizon and the library size.

Enforcement margin. Because $h _ { \theta }$ is fitted from finite data, we enforce $h _ { \theta } - \eta \geq 0$ for a margin $\eta > 0$ rather than $h _ { \theta } \geq 0$ , and choose $\eta$ as the smallest value at which the measured false safe rate on held out labeled states is zero. Additive disturbances can be handled with the DO-ICBF of Zinage et al. (2023).

## 4.4. The Resultant Program

Let $h _ { \theta , 1 } , \hdots , h _ { \theta , M }$ be the active state barriers, one per nearby constraint, and let $p _ { i } , q _ { i }$ be the corresponding ICBF row and right hand side built from (6). The safe input follows the integral law $\dot { \pmb u } = \phi _ { \theta } ( \pmb x , \pmb u ) + \pmb v ^ { \star } ( \pmb x , \pmb u )$ with

$$
( { \pmb v } ^ { \star } , { \pmb s } ^ { \star } ) = \operatorname * { a r g m i n } _ { { \pmb v } \in \mathbb { R } ^ { m } , { \pmb s } \in \mathbb { R } _ { \geq 0 } ^ { M } } \| { \pmb W } { \pmb v } \| ^ { 2 } + \rho \| { \pmb s } \| ^ { 2 } \quad \mathrm { s . t . } \quad P { \pmb v } + { \pmb s } \geq Q , \qquad { \pmb v } \in \mathcal { V } ( { \pmb x } , { \pmb u } ) ,\tag{9}
$$

where P stacks the rows $p _ { i } ^ { \top } , Q$ the values $q _ { i } , W \succ 0$ weights the input channels by their physical rates, $\rho > 0$ penalizes slack, and $\mathcal { V } ( \pmb { x } , \pmb { u } )$ is the space of auxiliary inputs that keep every channel within both its rate limit and its actuator range, written out in (12).

In $P v + s \geq Q$ , each row, $\begin{array} { r } { p _ { i } ^ { \top } = \frac { \partial h _ { \theta , i } } { \partial \mathbfit { x } } \frac { \partial F _ { \theta } } { \partial \mathbfit { u } } } \end{array}$ is a barrier gradient contracted with the input Jacobian of the dynamics, involving $h _ { \theta }$ and $F _ { \theta }$ . Substituting $h _ { e , \theta , i }$ for h in Definition 2 gives,

$$
q _ { i } = - \Big ( \frac { \partial h _ { e , \theta , i } } { \partial \pmb { x } } F _ { \theta } + p _ { i } ^ { \top } \phi _ { \theta } + \gamma h _ { e , \theta , i } \Big ) ,\tag{10}
$$

with $\gamma > 0$ a linear class K gain on the extended barrier, distinct from the α of (6) and from the actuator gain $\gamma _ { u } .$ . It adds the drift of the extended barrier along the learned dynamics to the term $p _ { i } ^ { \top } \phi _ { \theta }$ , which is the only place where the nominal controller acts, since $\phi _ { \theta }$ is assembled from $\partial \pi _ { \boldsymbol { \theta } } / \partial x$ by (5). The barrier and the nominal controller are therefore read through their derivatives alone, while the dynamics enter both directly, in the drift, and through the input Jacobian that makes the row nonzero.

What (9) returns is a rate rather than an input, because the ICBF construction lifted u into the state of the augmented system (2). The command the actuators receive is obtained by integrating that rate across one control period,

$$
\begin{array} { r } { { \pmb u } ( t + \Delta t ) = { \pmb u } ( t ) + \big ( \phi _ { \theta } ( { \pmb x } , { \pmb u } ) + { \pmb v } ^ { \star } \big ) \Delta t , \quad { \pmb u } ( 0 ) = \pi _ { \theta } ( { \pmb x } ^ { 0 } ) , } \end{array}\tag{11}
$$

and $\pi _ { \theta }$ supplies an admissible starting input because its output layer is bounded and scaled to the actuator ranges.

## 5. Feasibility, Invariance, and Error Analysis

Forward invariance is enabled assuming an admissible start provided by ${ \pmb u } ( 0 ) = \pi _ { \theta } ( { \pmb x } ^ { 0 } )$ . The construction also recovers from a start that is outside U but not far outside it, as part (ii) of Proposition 1 makes precise.

Proposition 1 (Feasibility and input invariance) Let $\gamma _ { u } > 0$ be the actuator gain and, for each channel j, let $R _ { j } > 0$ be its rate limit and $u _ { j } ^ { \operatorname* { m a x } } > 0$ its range, so that the input constraints of (9) are carried by

$$
\begin{array} { r } { \mathcal V ( { \pmb x } , { \pmb u } ) = \big \lbrace { \pmb v } : \underline { { \upsilon } } _ { j } \leq v _ { j } \leq \bar { v } _ { j } \forall j \big \rbrace , \quad \frac { v _ { j } = \operatorname* { m a x } \{ - R _ { j } , - \gamma _ { u } ( u _ { j } + u _ { j } ^ { \operatorname* { m a x } } ) \} - \phi _ { \theta , j } , } { \bar { v } _ { j } = \operatorname* { m i n } \{ R _ { j } , \gamma _ { u } ( u _ { j } ^ { \operatorname* { m a x } } - u _ { j } ) \} - \phi _ { \theta , j } , } } \end{array}\tag{12}
$$

whose second arguments are the ICBF conditionsfor the actuator range barriers $h _ { u , j } ^ { \pm } = u _ { j } ^ { \operatorname* { m a x } } \mp u _ { j }$ Write $\mathcal { U } _ { + } ~ = ~ \{ \pmb { u } ~ : ~ | \boldsymbol { u } _ { j } | ~ \leq ~ \boldsymbol { u } _ { j } ^ { \operatorname* { m a x } } + R _ { j } / \gamma _ { \boldsymbol { u } } ~ f o r \boldsymbol { a l l } ~ j \} ~ \supseteq ~ \mathcal { U } ,$ , and suppose the closed loop ${ \dot { \textbf { \em u } } } =$ $\phi _ { \theta } + { \pmb v } ^ { \star } ( { \pmb x } , { \pmb u } )$ admits a solution on the interval considered. Then, for every $\mathbf { { \boldsymbol { x } } } \in { \mathcal { X } } ,$ , every $\pmb { u } \in \mathcal { U } _ { + }$ and every finite set of barrier rows, the following three statements hold. (i) The set $\nu ( { \pmb x } , { \pmb u } )$ is nonempty, the feasible set of (9) is nonempty, and the minimizer is unique. (ii) The realized rate $\dot { \pmb { u } } = \phi _ { \theta } + \pmb { v } ^ { \star }$ respects every actuator rate limit, $| \dot { u } _ { j } | \le R _ { j }$ , and renders both U and $\ d \mathcal { U } _ { + }$ forward invariant. On $\ d \mathcal { U } _ { + } \backslash \mathcal { U }$ it returns every violating channel to its range, because $u _ { j } > u _ { j } ^ { \mathrm { m a x } }$ forces $\dot { u } _ { j } \le \gamma _ { u } \big ( u _ { j } ^ { \operatorname* { m a x } } - u _ { j } \big ) < 0$ . (iii) No input constraint is relaxed, because s appears only in the barrier rows.

The full argument is given in Appendix A.1. Proposition 1 is what separates this filter from a soft penalty formulation: state safety is approximate, because $h _ { \theta }$ is learned, but input feasibility is exact and is never purchased at the expense of a barrier row. Section 6.1 shows the empirical counterpart, and Section 6.3 reports that the program was feasible at all 74,590 control steps of the racing evaluation.

We now bound the gap between the input produced from $( F _ { \theta } , \pi _ { \theta } , h _ { \theta } )$ and the input the same construction would produce from $( F , k , h )$ . Write $\varepsilon _ { F } , \varepsilon _ { F ^ { \prime } }$ for the sup norm errors of $F _ { \theta }$ and of $\partial F _ { \theta } / \partial \pmb { u }$ $\varepsilon _ { k } , \varepsilon _ { k ^ { \prime } }$ for those of $\pi _ { \theta }$ and $\partial \pi _ { \boldsymbol { \theta } } / \partial \boldsymbol { x }$ , and $\varepsilon _ { p } , \varepsilon _ { q }$ for the induced errors in the ICBF row and right hand side.

Assumption 1 On a compact operating set $\Omega \subset \mathcal { X } \times \mathcal { U }$ the functions F, k and $F _ { \theta } , \pi _ { \theta }$ are continuously differentiable and $h , h _ { \theta }$ are twice continuously differentiable, so that $h _ { e }$ and $h _ { e , \theta }$ are continuously differentiable, with $\lVert \boldsymbol { F } \rVert \le B _ { F } , \lVert \boldsymbol { \phi } \rVert , \lVert \boldsymbol { \phi } _ { \theta } \rVert \le B _ { \phi }$ and $\| \partial h _ { e , \theta } / \partial \pmb { x } \| \le B _ { h ^ { \prime } }$ , the policy $\pi _ { \theta }$ is $L _ { \pi } L i p s c h i t z ,$ , the class Kfunction γ ofDefinition $2 i s L _ { \gamma }$ Lipschitz, and the ICBF is regular in the sense that $0 < \sigma \leq \| p \| , \| p _ { \theta } \| \leq \bar { p }$ with $[ q ] _ { + } , [ q _ { \theta } ] _ { + } \leq \bar { q } .$

Theorem 6 Let u<sup>⋆</sup> and $\boldsymbol { \mathscr { u } } ^ { \star }$ be obtained by one Euler step of the integral law from the same $( { \pmb x } , { \pmb u } ) \in$ Ω, the first built from $( F _ { \theta } , \pi _ { \theta } , h _ { \theta } )$ and the secondfrom $( F , k , h )$ , with $v ^ { \star }$ in each case the solution ofthe one row ICBF program (3). Under Assumption 1,

$$
\begin{array} { r } { \| u _ { \theta } ^ { \star } - u ^ { \star } \| \leq \Delta t \Big [ \underbrace { B _ { F } \varepsilon _ { k ^ { \prime } } + L _ { \pi } \varepsilon _ { F } + \frac { \alpha _ { \phi } } { 2 } \varepsilon _ { k } } _ { i n t e g r a l l a w } + \underbrace { \frac { \bar { p } } { 2 } \varepsilon _ { q } + \left( \frac { \bar { q } } { \sigma ^ { 2 } } + \frac { 2 \bar { p } ^ { 2 } \bar { q } } { \sigma ^ { 4 } } \right) \varepsilon _ { p } } _ { Q P s o l u t i o n m a p } \Big ] , } \end{array}\tag{13}
$$

and the row errors obey $\begin{array} { r } { \varepsilon _ { p } \leq \left\| \frac { \partial F } { \partial u } \right\| \varepsilon _ { h ^ { \prime } } + \left\| \nabla h _ { \theta } \right\| \varepsilon _ { F ^ { \prime } } a n d \varepsilon _ { q } \leq L _ { \gamma } \varepsilon _ { h _ { e } } + B _ { h ^ { \prime } } \varepsilon _ { F } + B _ { F } \varepsilon _ { h _ { e } ^ { \prime } } + \bar { p } \varepsilon _ { \phi } + } \end{array}$ $B _ { \phi } \varepsilon _ { p } ,$ where $\varepsilon _ { h ^ { \prime } }$ and $\varepsilon _ { h _ { e } ^ { \prime } }$ are the sup norm gradient errors of $h _ { \theta }$ and $h _ { e , \theta } ,$ , and $\varepsilon _ { h _ { e } }$ and $\varepsilon _ { \phi }$ the sup norm errors of $\cdot h _ { e , \theta }$ and ϕ<sub>θ</sub> on Ω.

The full proof is given in Appendix A.2. The theorem is stated for the one row program (3) of Theorem 3, which is the setting in which σ has its meaning. The deployed program (9) adds the input box, the slack variables and one row per nearby constraint, and although its solution map remains Lipschitz in the problem data because the objective is strongly convex, the governing constant is then a Hoffman constant of the active system rather than $\sigma ,$ , and it is not uniform across changes of active set. We therefore do not claim the explicit constants of (13) for that program.

## 6. Results

We report four studies, beginning in Section 6.1 with two low dimensional systems and the comparison with SABLAS, continuing in Section 6.2 with viability labeling in isolation, and turning in Section 6.3 to the agile racing evaluation that the method targets and to its behaviour when the plant departs from the friction the models were trained on. All code is implemented in PyTorch and all filter timings are on a single CPU core.

## 6.1. On low dimensional systems

The same pipeline was first validated on two low dimensional systems, a control affine kinematic bicycle and a quadrotor carrying a suspended pendulum (Beard, 2008), each avoiding M obstacles on the way to a goal under a nonlinear MPC expert. On both it stays within 1% of the nonlinear MPC cost at roughly a seventh of its runtime, because the program (9) deviates minimally from the unconstrained solution. The full comparison, together with two multi agent navigation methods included as runtime reference points, is given in the supplement. The remainder of this section concerns the choice that separates this framework from a jointly trained filter.

Table 1 compares against SABLAS (Qin et al., 2022) on its own double integrator environment and makes the cost of the two architectures explicit. SABLAS reaches the goal more often and at lower cost, which is what joint training of the policy and the certificate enables. It satisfies the input constraint on 53.38% of episodes, because that constraint is a penalty term competing with

<table><tr><td>Method</td><td>Safety</td><td>Goal</td><td>Input compliance</td><td>Cost</td></tr><tr><td>Ours</td><td>100%</td><td>71%</td><td>100%</td><td>16758</td></tr><tr><td>SABLAS</td><td>98.49%</td><td>89%</td><td>53.38%</td><td>11474</td></tr></table>

Table 1: Double integrator benchmark of Qin et al. (2022), 100 random test episodes. Input compliance is the fraction of episodes on which the actuator constraint held throughout.

∆x [m]

![](images/fc8e4bfa10a6457894135fe0ddad58b3027409de8618eb82ac30460c636ac4e5.jpg)

![](images/3fae978a0a3c96629a940d48cb7c3edeb7082a25933266abcddc4d10fd61ed16.jpg)

![](images/b09fb02b2697ad1c2dd40ed644e9e658a09dbc4625baf5987419a0f118783685.jpg)  
Figure 1: What the viability labeled barrier represents. (left, middle) The learned obstacle barrier $h _ { \theta }$ in the body frame at two speeds, with its zero level set and the geometric obstacle. At 1 m/s the two nearly coincide, whereas at 5 m/s the unsafe set extends roughly 1.5 m along the approach direction, covering states clear of the obstacle that can no longer avoid it under the actuator limits. (c) Fraction of geometrically safe states that viability relabels unsafe, by speed: the signature of a viability set rather than a smoothed obstacle indicator.

the task objective. Our filter satisfies it on every

episode, for the structural reason given in Proposition 1: the input constraint is a bound on the decision variable, not a term in an objective, so there is nothing for the optimizer to trade it against. However, it comes at a cost of reduction in goal rate while maintaining higher safety with input satisfaction.

## 6.2. Viability labeling

We sample 12,000 states around an obstacle, with heading, speed, slip and yaw rate drawn over the operating envelope, and compare the geometric label with the viability label (8) computed through $F _ { \theta }$ with a library of $| { \mathcal { M } } | = 2 2$ backup maneuvers (three brake levels crossed with six steering policies, plus four full throttle swerves) over $T = 1 . 5 \ : \mathrm { s }$

Convergence of the labels. Both limits that define the label have converged at the deployed setting. Lengthening the horizon past 0.5 s does not change the labels. The relabeled fraction is 2.52% at 0.3 s and 2.60% from 0.5 s onward, because every backup maneuver has brought the vehicle to rest by then. Enlarging the library moves the fraction monotonically downward, from 6.53% with a single maneuver to 2.61% at eighteen, beyond which it stops changing, so the labeled safe set is stable well before the library is exhausted. No geometrically unsafe state was ever labeled safe, in agreement with the argument in Section 4.3. A worst case test over a single fixed policy has no such limit, since lengthening the horizon or widening the heading sample can only add unsafe states, so the fraction it reports is an artifact of where the sweep was truncated.

Size of the correction, and its dependence on speed. Over the whole sample, geometry marks 8.57% of states unsafe and viability marks 10.95%, so 2.60% of the geometrically safe states are already committed. Fig. 1(c) shows that this correction is concentrated at speed: it is 0.34% below 1 m/s, where the vehicle stops within its own footprint, and rises monotonically to 4.37% above 4 m/s, a factor of thirteen. A barrier trained on geometric labels is therefore not merely conservative at racing speed but wrong, and it is most wrong exactly where the consequences are worst.

Enforcement margin. We enforce $h _ { \theta } \geq \eta$ and pick $\eta$ by the false safe rate on held out labeled states, the fraction of unsafe states the network scores above the threshold. That rate is 1.70%, 0.076% and 0.00% at $\eta = 0 , 0 . 1$ and 0.2 for the obstacle barrier, and 1.81%, 0.038% and 0.00% for the wall, so we deploy $\eta = 0 . 2$ , the smallest tested value at which no held out unsafe state is scored safe.

## 6.3. Agile racing with a 6-D dynamic bicycle

Setup. We apply the same pipeline to a non control-affine 6-D dynamic single track model with F1TENTH parameters (O’Kelly et al., 2020), $\pmb { x } = ( x , y , \psi , v _ { x } , v _ { y } , r )$ and $\pmb { u } = \left( a _ { x } , \delta \right)$ , with saturating lateral tire forces, a steering slew limit and first order motor lag $( \Delta t = 0 . 0 2 \mathrm { s } , \mathrm { R K } 4 )$ . The car races through a cluttered 50 m × 4 m track at 3.5 and 4.0 m/s with $\mu = 1 . 0$ . Learned components. $F _ { \theta } ,$ , trained with (4) over 10 step RK4 rollouts, has a median 1 s open loop position error of 8.5 cm, or 14.1 cm at the 95th percentile. More importantly for the filter, its input Jacobian $\partial F _ { \theta } / \partial \pmb { u }$ has median cosine similarity 0.99998 to the true Jacobian and median relative error 8.4%. The expert is a GPU MPPI planner (Williams et al., 2016) on an empty track, and $\pi _ { \theta }$ takes the target speed as an additional input. Separate barriers are learned for obstacles and for the wall, the opposite wall following by mirror symmetry, with the viability labels (8), reaching 99.7% and 99.4% sign accuracy. Filter. Program (9) carries one row per nearby obstacle and wall, a grip row $( 0 . 8 \mu g ) ^ { 2 } - a _ { y } ( { \pmb x } , { \pmb u } ) ^ { 2 } \geq 0$ on the lateral acceleration predicted by $F _ { \theta } .$ , treated as a state-input ICBF in the sense of Definition 2, together with the input box (12).

Baselines and protocol. We compare MPPI alone with $K = 5 0$ samples, an analytical HOCBF-QP filter (Xiao and Belta, 2022) on MPPI, and the Neural ICBF with either $\pi _ { \theta }$ or MPPI as nominal, on 200 layouts never used for tuning and with every parameter frozen beforehand. Unsafe means collision or leaving the track.

Results. Table 2 summarizes the outcomes. At 4.0 m/s MPPI alone is unsafe in 12 of 200 episodes, while the Neural ICBF is unsafe in 1 with $\pi _ { \theta }$ as nominal and in 0 with MPPI as nominal, the latter at an unchanged success rate of 84.5% against 84.0%. At 3.5 m/s MPPI alone is already nearly crash free, 1 of 200, and the filter matches it. The analytical HOCBF-QP is unsafe in 29 to 35% of episodes because it enforces a barrier certified for a kinematic model, and the vehicle obeys a dynamic model with tire saturation and actuator lag, so the certificate it enforces is void under the dynamics actually in force. Our learned filter avoids this because its barrier rows are differentiated through $F _ { \theta }$ , which was fitted to

<table><tr><td>u (m/s)</td><td>Method</td><td>Success</td><td>Collision</td><td>Out of bounds</td><td>Unsafe</td><td>Stall</td><td>Timeout</td></tr><tr><td rowspan="5">3.5</td><td>MPPI</td><td>172</td><td>0</td><td>1</td><td>1</td><td>27</td><td>0</td></tr><tr><td>HOCBF-QP</td><td>122</td><td>58</td><td>0</td><td>58</td><td>14</td><td>6</td></tr><tr><td>Ours, πθ</td><td>162</td><td>3</td><td>0</td><td>3</td><td>35</td><td>0</td></tr><tr><td>Ours, MPPI</td><td>166</td><td>0</td><td>0</td><td>0</td><td>34</td><td>0</td></tr><tr><td>MPPI</td><td>168</td><td>7</td><td>5</td><td>12</td><td>20</td><td>0</td></tr><tr><td rowspan="4">4.0</td><td>HOCBF-QP</td><td>55</td><td>69</td><td>1</td><td>70</td><td>18</td><td>57</td></tr><tr><td>Ours, πθ</td><td>155</td><td>1</td><td>0</td><td>1</td><td>44</td><td>0</td></tr><tr><td>Ours, MPPI</td><td>169</td><td>0</td><td>0</td><td>0</td><td>31</td><td>0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Agile racing on 200 held out layouts, in episode counts, with every row summing to 200. MPPI is the planner alone, HOCBF-QP is the analytical filter of Xiao and Belta (2022) applied to $\mathrm { i t , }$ and Ours is the Neural ICBF with either $\pi _ { \theta }$ or MPPI as the nominal controller. Unsafe is the sum of the collision and out of bounds columns.

The filter runs in 7.2 to 7.9 ms per step on a single uncontended CPU core, or 8.7 to 9.2 ms at the 95th percentile, so it closes the loop above 100 Hz without a GPU even though every row is differentiated through two networks at run time. Across the 120 episodes and 74,590 control steps on which solver diagnostics were recorded, (9) was feasible at every step, consistent with Proposition 1.

Note that success is 3 to 6 points lower with $\pi _ { \theta }$ as the nominal controller (due to the vehicle stalling), which is expected of a policy trained on an empty track and obstacle unaware by construction. The configuration we would deploy is therefore the filter on top of an obstacle aware planner (e.g. MPPI).

Scope. labeled states are still required, so the framework reduces rather than removes the prior safety knowledge needed, and state safety is enforced only up to the penalized slack in (9), in contrast to the input guarantee of Proposition 1. The method targets a sub limit regime in which MPPI requests more than 90% of the available grip on only 4 to 5% of steps. At 6 m/s the front tire saturates, the steering gradient of the wall barrier changes sign, the margin σ of Assumption 1 collapses as the $\sigma ^ { - 4 }$ term of Theorem 6 anticipates, and the one step program steers into the wall. Removing that failure calls for a predictive formulation rather than a larger library or a finer barrier.

## 7. Conclusion

We gave a complete learning based realization of the ICBF safety filter for systems whose dynamics, nominal controller and barrier are all unknown, resting on viability labeling through the learned dynamics and on an online program in which actuator limits are hard and learned barrier rows are relaxable. That program is feasible at every state and keeps the actuator set forward invariant, and the filter built on it removes every unsafe episode from an agile racing task at an unchanged success rate and above 100 Hz on one CPU core. The friction limit, deadlock in front of obstacles, and transfer to hardware remain open.

## References

Ayush Agrawal and Koushil Sreenath. Discrete control barrier functions for safety-critical control of discrete systems with application to bipedal robot navigation. In Robotics: Science and Systems, volume 13, pages 1–10. Cambridge, MA, USA, 2017.

Devansh R Agrawal and Dimitra Panagou. Safe control synthesis via input constrained control barrier functions. In 2021 60th IEEE Conference on Decision and Control (CDC), pages 6113– 6118. IEEE, 2021.

Mohamadreza Ahmadi, Andrew Singletary, Joel W Burdick, and Aaron D Ames. Safe policy synthesis in multi-agent pomdps via discrete-time barrier functions. In 2019 IEEE 58th Conference on Decision and Control (CDC), pages 4797–4803. IEEE, 2019.

Aaron D Ames, Samuel Coogan, Magnus Egerstedt, Gennaro Notomista, Koushil Sreenath, and Paulo Tabuada. Control barrier functions: Theory and applications. In 2019 18th European control conference (ECC), pages 3420–3431. IEEE, 2019.

Aaron D Ames, Gennaro Notomista, Yorai Wardi, and Magnus Egerstedt. Integral control barrier functions for dynamically defined control laws. IEEE control systems letters, 5(3):887–892, 2020.

Randal W Beard. Quadrotor dynamics and control. Brigham Young University, 19(3):46–56, 2008.

Giovanni Beltrame, Ettore Merlo, Jacopo Panerati, and Carlo Pinciroli. Engineering safety in swarm robotics. In Proceedings of the 1st International Workshop on Robotics Software Engineering, pages 36–39, 2018.

Mitchell Black and Dimitra Panagou. Consolidated control barrier functions: Synthesis and online verification via adaptation under input constraints. arXiv preprint arXiv:2304.01815, 2023.

Nicholas Boffi, Stephen Tu, Nikolai Matni, Jean-Jacques Slotine, and Vikas Sindhwani. Learning stability certificates from data. In Conference on Robot Learning, pages 1341–1350. PMLR, 2021.

Joseph Breeden and Dimitra Panagou. High relative degree control barrier functions under input constraints. In 2021 60th IEEE Conference on Decision and Control (CDC), pages 6119–6124. IEEE, 2021.

Joseph Breeden and Dimitra Panagou. Robust control barrier functions under high relative degree and input constraints for satellite trajectories. Automatica, 155:111109, 2023.

Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.

Yu Fan Chen, Miao Liu, Michael Everett, and Jonathan P How. Decentralized non-communicating multiagent collision avoidance with deep reinforcement learning. In 2017 IEEE international conference on robotics and automation (ICRA), pages 285–292. IEEE, 2017.

Richard Cheng, Gabor Orosz, Richard M Murray, and Joel W Burdick. End-to-end safe reinforce-´ ment learning through barrier functions for safety-critical continuous control tasks. In Proceedings ofthe AAAI conference on artificial intelligence, volume 33, pages 3387–3395, 2019.

Rahman Dadashi, Loic Hussenot, Mathieu Geist, and Olivier Pietquin. Primal wasserstein imitation learning. CoRR, abs/2006.04678, 2020.

Charles Dawson, Zengyi Qin, Sicun Gao, and Chuchu Fan. Safe nonlinear control using robust neural lyapunov-barrier functions. In Conference on Robot Learning, pages 1724–1735. PMLR, 2022.

Stepan Dergachev and Konstantin Yakovlev. Distributed multi-agent navigation based on reciprocal collision avoidance and locally confined multi-agent path finding. In 2021 IEEE 17th International Conference on Automation Science and Engineering (CASE), pages 1489–1494. IEEE, 2021.

Rasheed Hussain and Sherali Zeadally. Autonomous cars: Research results, issues, and future challenges. IEEE Communications Surveys & Tutorials, 21(2):1275–1313, 2018.

Haresh Karnan, Faraz Torabi, Garrett Warnell, and Peter Stone. Adversarial imitation learning from video using a state observer. In 2022 International Conference on Robotics and Automation (ICRA), pages 2452–2458. IEEE, 2022.

Adam K. Kiss, Ersin Das, Tamas G. Molnar, and Aaron D. Ames. Integral control barrier functions with input delay: Prediction, feasibility, and robustness. arXiv preprint arXiv:2603.24566, 2026.

Ilya Kostrikov, Ofir Nachum, and Jonathan Tompson. Imitation learning via off-policy distribution matching, 2019.

Shuo Liu, Wei Xiao, and Calin A Belta. Feasibility-guaranteed safety critical control with applications to heterogeneous platoons. arXiv preprint arXiv:2310.00238, 2023a.

Simin Liu, Changliu Liu, and John Dolan. Safe control under input limits with neural control barrier functions. In Proceedings ofThe 6th Conference on Robot Learning, volume 205 of Proceedings of Machine Learning Research, pages 1970–1980. PMLR, 2023b.

Yue Meng, Zengyi Qin, and Chuchu Fan. Reactive and safe road user simulations using neural barrier certificates. In 2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 6299–6306. IEEE, 2021.

Matthew O’Kelly, Hongrui Zheng, Dhruv Karthik, and Rahul Mangharam. F1TENTH: An opensource evaluation environment for continuous control and reinforcement learning. In NeurIPS 2019 Competition and Demonstration Track, pages 77–89. PMLR, 2020.

Zengyi Qin, Kaiqing Zhang, Yuxiao Chen, Jingkai Chen, and Chuchu Fan. Learning safe multiagent control with decentralized neural barrier certificates. arXiv preprint arXiv:2101.05436, 2021a.

Zengyi Qin, Kaiqing Zhang, Yuxiao Chen, Jingkai Chen, and Chuchu Fan. Learning safe multiagent control with decentralized neural barrier certificates. arXiv preprint arXiv:2101.05436, 2021b.

Zengyi Qin, Dawei Sun, and Chuchu Fan. Sablas: Learning safe control for black-box dynamical systems. IEEE Control Systems Letters, 6:2527–2532, 2022.

Alexander Robey, Haimin Hu, Lars Lindemann, Hanwen Zhang, Dimos V Dimarogonas, Stephen Tu, and Nikolai Matni. Learning control barrier functions from expert demonstrations. In 2020 59th IEEE Conference on Decision and Control (CDC), pages 3717–3724. IEEE, 2020.

Joohwan Seo, Joonho Lee, Eunkyu Baek, Roberto Horowitz, and Jongeun Choi. Safety-critical control with nonaffine control inputs via a relaxed control barrier function for an autonomous vehicle. IEEE Robotics and Automation Letters, 7(2):1944–1951, 2022.

Oswin So, Zachary Serlin, Makai Mann, Jake Gonzales, Kwesi Rutledge, Nicholas Roy, and Chuchu Fan. How to train your neural control barrier function: Learning safety filters for complex input-constrained systems. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 11532–11539. IEEE, 2024.

Tong Duy Son and Quan Nguyen. Safety-critical control for non-affine nonlinear systems with application on autonomous vehicle. In 2019 IEEE 58th Conference on Decision and Control (CDC), pages 7623–7628. IEEE, 2019.

Mohit Srinivasan, Amogh Dabholkar, Samuel Coogan, and Patricio A Vela. Synthesis of control barrier functions using a supervised machine learning approach. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 7139–7145. IEEE, 2020.

Andrew Taylor, Andrew Singletary, Yisong Yue, and Aaron Ames. Learning for safety-critical control with control barrier functions. In Learning for Dynamics and Control, pages 708–717. PMLR, 2020a.

Andrew J Taylor, Andrew Singletary, Yisong Yue, and Aaron D Ames. A control barrier perspective on episodic learning via projection-to-state safety. IEEE Control Systems Letters, 5(3):1019– 1024, 2020b.

Faraz Torabi, Garrett Warnell, and Peter Stone. Behavioral cloning from observation. arXiv preprint arXiv:1805.01954, 2018.

Kim P Wabersich, Andrew J Taylor, Jason J Choi, Koushil Sreenath, Claire J Tomlin, Aaron D Ames, and Melanie N Zeilinger. Data-driven safety filters: Hamilton-jacobi reachability, control barrier functions, and predictive methods for uncertain systems. IEEE Control Systems Magazine, 43(5):137–177, 2023.

Chuanzheng Wang, Yiming Meng, Yinan Li, Stephen L Smith, and Jun Liu. Learning control barrier functions with high relative degree for safety-critical control. In 2021 European Control Conference (ECC), pages 1459–1464. IEEE, 2021.

Li Wang, Evangelos A Theodorou, and Magnus Egerstedt. Safe learning of quadrotor dynamics using barrier certificates. In 2018 IEEE International Conference on Robotics and Automation (ICRA), pages 2460–2465. IEEE, 2018.

Grady Williams, Paul Drews, Brian Goldfain, James M Rehg, and Evangelos A Theodorou. Aggressive driving with model predictive path integral control. In 2016 IEEE International Conference on Robotics and Automation (ICRA), pages 1433–1440. IEEE, 2016.

Wei Xiao and Calin Belta. High-order control barrier functions. IEEE Transactions on Automatic Control, 67(7):3655–3662, 2022.

Wei Xiao, Ramin Hasani, Xiao Li, and Daniela Rus. Barriernet: A safety-guaranteed layer for neural networks. arXiv preprint arXiv:2111.11277, 2021.

Wei Xiao, Christos G Cassandras, and Calin Belta. Feasibility for cbf-based optimal control using machine learning. In Safe Autonomy with Control Barrier Functions, pages 53–72. Springer, 2023a.

Wei Xiao, Christos G Cassandras, and Calin A Belta. Learning feasibility constraints for control barrier functions. arXiv preprint arXiv:2303.09403, 2023b.

Shakiba Yaghoubi, Georgios Fainekos, and Sriram Sankaranarayanan. Training neural network controllers using control barrier functions in the presence of disturbances. In 2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC), pages 1–6. IEEE, 2020.

Chenning Yu, Hongzhan Yu, and Sicun Gao. Learning control admissibility models with graph neural networks for multi-agent navigation. In Conference on Robot Learning, pages 934–945. PMLR, 2023.

Jun Zeng, Zhongyu Li, and Koushil Sreenath. Enhancing feasibility and safety of nonlinear model predictive control with discrete-time control barrier functions. In 2021 60th IEEE Conference on Decision and Control (CDC), pages 6137–6144. IEEE, 2021.

Boyuan Zheng, Sunny Verma, Jianlong Zhou, Ivor W Tsang, and Fang Chen. Imitation learning: Progress, taxonomies and challenges. IEEE Transactions on Neural Networks and Learning Systems, (99):1–16, 2022.

Vrushabh Zinage and Efstathios Bakolas. Neural koopman control barrier functions for safetycritical control of unknown nonlinear systems. In 2023 American Control Conference (ACC), pages 3442–3447. IEEE, 2023.

Vrushabh Zinage, Rohan Chandra, and Efstathios Bakolas. Disturbance observer-based robust integral control barrier functions for nonlinear systems with high relative degree. arXiv preprint arXiv:2309.16945, 2023.

Vrushabh Zinage, Abhishek Jha, Rohan Chandra, and Efstathios Bakolas. Decentralized safe and scalable multi-agent control under limited actuation. arXiv preprint arXiv:2409.09573, 2024.

## Appendix A. Supplementary material

## A.1. Proof of Proposition 1

The three parts are proved in turn. Part (i) is a statement about the geometry of the box $\mathcal { V } ( \pmb { x } , \pmb { u } )$ of (12) and about convexity of the program, part (ii) identifies the two bounds defining that box as rate limits and as barrier conditions for the actuator range, and part (iii) reads off the role of the slack variable. Throughout, j indexes a single input channel, and all statements are made channel by channel because both V and U are products over j.

Proof [Proof of Proposition 1]

Part (i), the box V is nonempty exactly on U<sub>+</sub>. By (12) the channel j bounds are

$$
\begin{array} { r } { \underline { { v } } _ { j } = \operatorname* { m a x } \{ \underbrace { - R _ { j } } _ { \mathrm { r a t e } } , \underbrace { - \gamma _ { u } ( u _ { j } + u _ { j } ^ { \operatorname* { m a x } } ) } _ { \mathrm { r a n g e } } \} - \phi _ { \theta , j } , \quad \overline { { v } } _ { j } = \operatorname* { m i n } \{ \underbrace { R _ { j } } _ { \mathrm { r a t e } } , \underbrace { \gamma _ { u } ( u _ { j } ^ { \operatorname* { m a x } } - u _ { j } ) } _ { \mathrm { r a p g e } } \} - \phi _ { \theta , j } , } \end{array}
$$

so the interval $[ \underline { { v } } _ { j } , \bar { v } _ { j } ]$ is nonempty if and only if $\underline { { { v } } } _ { j } \leq \bar { v } _ { j }$ . The term $- \phi _ { \theta , j }$ is common to both bounds and cancels, which leaves

$$
\operatorname* { m a x } \{ - R _ { j } , - \gamma _ { u } ( u _ { j } + u _ { j } ^ { \operatorname* { m a x } } ) \} \ \leq \ \operatorname* { m i n } \{ R _ { j } , \gamma _ { u } ( u _ { j } ^ { \operatorname* { m a x } } - u _ { j } ) \} .
$$

For real numbers, max $\left\{ a _ { 1 } , a _ { 2 } \right\} \leq \operatorname* { m i n } \{ b _ { 1 } , b _ { 2 } \}$ holds if and only if $a _ { i } \leq b _ { j }$ for all four pairs, so the display above is equivalent to the following four inequalities, which we take one at a time.

(a) rate against rate:

$$
- R _ { j } \leq R _ { j } \Longleftrightarrow 0 \leq 2 R _ { j } ,
$$

(b) range against range: $- \gamma _ { u } \big ( u _ { j } + u _ { j } ^ { \operatorname* { m a x } } \big ) \leq \gamma _ { u } \big ( u _ { j } ^ { \operatorname* { m a x } } - u _ { j } \big ) \quad \Longleftrightarrow \quad - u _ { j } ^ { \operatorname* { m a x } } \leq u _ { j } ^ { \operatorname* { m a x } } ,$

(c) rate against range:

$$
- R _ { j } \leq \gamma _ { u } \bigl ( u _ { j } ^ { \operatorname* { m a x } } - u _ { j } \bigr ) \quad \Longleftrightarrow \quad u _ { j } \leq u _ { j } ^ { \operatorname* { m a x } } + R _ { j } / \gamma _ { u } ,
$$

(d) range against rate:

$$
- \gamma _ { u } \bigl ( u _ { j } + u _ { j } ^ { \operatorname* { m a x } } \bigr ) \leq R _ { j } \qquad \iff \quad - u _ { j } \leq u _ { j } ^ { \operatorname* { m a x } } + R _ { j } / \gamma _ { u } .
$$

Inequalities (a) and (b) hold for every ${ \mathbf { } } ^ { \mathbf { } } \mathbf { \Delta } ^ { \mathbf { } } \mathbf { u } ,$ since $R _ { j } > 0$ and $u _ { j } ^ { \operatorname* { m a x } } > 0$ by hypothesis. The remaining two are the only ones that constrain u, and together they say precisely that $| u _ { j } | \leq u _ { j } ^ { \operatorname* { m a x } } + R _ { j } / \gamma _ { u } ,$ which is membership of $\ d \mathcal { U } _ { + }$ . Since each step above is an equivalence, the conclusion runs in both directions: $\mathcal { V } ( \pmb { x } , \pmb { u } )$ is a nonempty box for every $\textbf { \em u } \in \mathcal { U } _ { + }$ , and it is empty as soon as $| u _ { j } | ~ >$ $u _ { j } ^ { \operatorname* { m a x } } + R _ { j } / \gamma _ { u }$ for some channel. Hence $\mathcal { U } _ { + }$ is not merely sufficient but is the largest set over which the claim can be made. The box is also bounded and closed, being a finite product of compact intervals, so it is compact.

Part (i) continued, the program is feasible with a unique minimizer. Feasibility of (9) needs a pair $( { \pmb v } , { \pmb s } )$ , not a v alone. Take any ${ \pmb v } \in \mathcal { V } ( { \pmb x } , { \pmb u } )$ , which part (i) has just supplied, and set

$$
\displaystyle \boldsymbol { s } : = \left[ Q - P v \right] _ { + } ,
$$

the positive part taken componentwise. Then $s \geq 0$ by construction, and row by row either $Q _ { i } -$ $( P v ) _ { i } \leq 0$ , in which case $( P \pmb { v } ) _ { i } + \pmb { s } _ { i } = ( P \pmb { v } ) _ { i } \geq Q _ { i } , \operatorname { o r } Q _ { i } - ( P \pmb { v } ) _ { i } > 0$ , in which case $( P { \pmb v } ) _ { i } + { \pmb s } _ { i } =$ $Q _ { i }$ . Either way $P v + s \geq Q$ , so $( { \pmb v } , { \pmb s } )$ is feasible and the feasible set is nonempty. It is also closed and convex, being the intersection of a box with finitely many halfspaces.

For uniqueness, the objective $\| W \pmb { v } \| ^ { 2 } + \rho \| \pmb { s } \| ^ { 2 }$ is a quadratic form in the stacked variable $( { \pmb v } , { \pmb s } )$ with Hessian

$$
\nabla ^ { 2 } \binom { \pmb { v } } { \pmb { s } } = \mathrm { b l k d i a g } \big ( 2 W ^ { \top } W , ~ 2 \rho I \big ) .
$$

Here $W \succ 0$ gives $W ^ { \top } W \succ 0$ , and $\rho > 0$ gives $\rho I \succ 0$ , so the Hessian is positive definite and the objective is strictly convex. A strictly convex function on a nonempty closed convex set attains its minimum at exactly one point, which proves the claim. Note that nothing in this argument referred to $P \thinspace \mathrm { o r } \thinspace Q$ , so it is independent of how many barrier rows there are and of whether the learned barrier is accurate.

Part (ii), the realized rate respects the rate limits and keeps U and $\ d \mathcal { U } _ { + }$ invariant. The minimizer satisfies ${ \pmb v } ^ { \star } \in \mathcal { V } ( { \pmb x } , { \pmb u } )$ , that is $\underline { { v } } _ { j } \le v _ { j } ^ { \star } \le \bar { v } _ { j }$ . Adding $\phi _ { \theta , j }$ throughout and using $\dot { u } _ { j } = \phi _ { \theta , j } + v _ { j } ^ { \star }$ cancels the − $- \phi _ { \theta , j }$ in the bounds and gives the two sided estimate

$$
\begin{array} { r } { \operatorname* { m a x } \{ - R _ { j } , - \gamma _ { u } \big ( u _ { j } + u _ { j } ^ { \operatorname* { m a x } } \big ) \} \ \leq \ \dot { u } _ { j } \ \leq \ \operatorname* { m i n } \{ R _ { j } , \gamma _ { u } \big ( u _ { j } ^ { \operatorname* { m a x } } - u _ { j } \big ) \} . } \end{array}
$$

This single display carries both conclusions, obtained by keeping one entry of the max and of the min at a time.

Keeping the rate entries gives $- R _ { j } \leq \dot { u } _ { j } \leq R _ { j }$ , which is the rate limit $| \dot { u } _ { j } | \le R _ { j }$

Keeping the range entries gives the barrier conditions. Write the two actuator range barriers as

$$
h _ { u , j } ^ { + } : = u _ { j } ^ { \operatorname* { m a x } } - u _ { j } , \qquad h _ { u , j } ^ { - } : = u _ { j } ^ { \operatorname* { m a x } } + u _ { j } ,
$$

so that $\mathcal { U } = \{ \pmb { u } : h _ { u , j } ^ { + } \geq 0$ and $h _ { u , j } ^ { - } \geq 0$ for every j}. Differentiating and substituting the range entries, 7J

$$
\begin{array} { r l } & { { \dot { h } } _ { u , j } ^ { + } = - { \dot { u } } _ { j } \ \geq \ - \gamma _ { u } \big ( { u } _ { j } ^ { \mathrm { m a x } } - { u } _ { j } \big ) \ = \ - \gamma _ { u } h _ { u , j } ^ { + } , } \\ & { { \dot { h } } _ { u , j } ^ { - } = \ \ \ { \dot { u } } _ { j } \ \geq \ - \gamma _ { u } \big ( { u } _ { j } + { u } _ { j } ^ { \mathrm { m a x } } \big ) \ = \ - \gamma _ { u } h _ { u , j } ^ { - } . } \end{array}
$$

Both are the differential condition $\dot { h } \ge - \gamma _ { u } h$ for a barrier function with the linear class K function $\gamma _ { u }$ . The comparison lemma applied to $\dot { h } \ge - \gamma _ { u } h$ yields $h ( t ) \geq h ( 0 ) e ^ { - \gamma _ { u } t }$ , so $h ( 0 ) \geq 0$ forces $h ( t ) \geq 0$ for all t in the interval considered. Applying this to $h _ { u , j } ^ { \pm }$ for every j gives forward invariance of $u .$

For $\ d \mathcal { U } _ { + }$ the same argument runs with the enlarged barriers $H _ { u , j } ^ { \pm } : = u _ { j } ^ { \operatorname* { m a x } } + R _ { j } / \gamma _ { u } \mp u _ { j }$ , whose zero superlevel sets intersect to $\ d \mathcal { U } _ { + }$ . Using $H _ { u , j } ^ { + } = h _ { u , j } ^ { + } + R _ { j } / \gamma _ { u }$ and the range entry once more,

$$
\dot { H } _ { u , j } ^ { + } = - \dot { u } _ { j } \geq - \gamma _ { u } h _ { u , j } ^ { + } = - \gamma _ { u } \bigl ( H _ { u , j } ^ { + } - R _ { j } / \gamma _ { u } \bigr ) = - \gamma _ { u } H _ { u , j } ^ { + } + R _ { j } \geq - \gamma _ { u } H _ { u , j } ^ { + } ,
$$

since $R _ { j } > 0$ , and symmetrically for $H _ { u , j } ^ { - }$ . So $\ d \mathcal { U } _ { + }$ is forward invariant as well.

Finally, suppose the input starts in $u _ { + } \backslash u$ , so that some channel has $u _ { j } > u _ { j } ^ { \operatorname* { m a x } }$ . The range entry of the upper bound then reads

$$
\dot { u } _ { j } \ \leq \ \gamma _ { u } ( u _ { j } ^ { \operatorname* { m a x } } - u _ { j } ) \ < \ 0 ,
$$

strictly negative because $u _ { i } ^ { \operatorname* { m a x } } - u _ { j } < 0$ and $\gamma _ { u } > 0$ , so that channel decreases, and it continues to decrease until $u _ { j } \leq u _ { j } ^ { \operatorname* { m a x } }$ . The case $u _ { j } < - u _ { j } ^ { \operatorname* { m a x } }$ is symmetric, with the lower bound giving $\dot { u } _ { j } \ge - \gamma _ { u } ( u _ { j } + u _ { j } ^ { \operatorname* { m a x } } ) > 0$ . Every violating channel is therefore returned to its range, which is the recovery claim.

Part (iii), no input constraint is relaxed. The slack s enters (9) in exactly two places, the barrier rows $P { \pmb v } + { \pmb s } \ge Q$ and its own sign constraint $s \geq 0$ , together with the penalty $\rho \| s \| ^ { 2 }$ in the objective. The input constraint is the separate requirement ${ \pmb v } \in \mathcal { V } ( { \pmb x } , { \pmb u } )$ , in which s does not appear. Consequently no value of s, however large, can enlarge the set of admissible $^ { v , }$ and the bounds used in part (ii) hold for the minimizer whatever the barrier rows demand. Relaxation is available to the learned part of the problem and is unavailable to the physical part.

## A.2. Proof of Theorem 6

The proof has four steps. Step 1 splits the deviation into a part caused by the integral law and a part caused by the quadratic program, so that the two sources of error can be bounded separately. Step 2 bounds the integral law part, Step 3 bounds the program part by exploiting the closed form of the one row program, and Step 4 traces the row errors $\varepsilon _ { p }$ and $\varepsilon _ { q }$ used in Step 3 back to the errors of the three networks. Every step uses the same elementary device, namely inserting a term that is a hybrid of the true and the learned object so that a difference of products becomes a sum of differences, each with one factor held fixed.

Proof [Proof of Theorem 6]

Step 1, splitting the deviation. Both inputs are produced from the same $( { \pmb x } , { \pmb u } )$ by one Euler step of the integral law, the learned one using $( \phi _ { \theta } , v _ { \theta } ^ { \star } )$ and the true one using $( \phi , v ^ { \star } )$

$$
\begin{array} { r } { \pmb { u } _ { \theta } ^ { \star } = \pmb { u } + \Delta t \big ( \phi _ { \theta } + \pmb { v } _ { \theta } ^ { \star } \big ) , \qquad \pmb { u } ^ { \star } = \pmb { u } + \Delta t \big ( \phi + \pmb { v } ^ { \star } \big ) . } \end{array}
$$

The common u cancels on subtraction, and the triangle inequality separates the two sources,

$$
\begin{array} { r } { \| u _ { \theta } ^ { \star } - u ^ { \star } \| = \Delta t \| \big ( \phi _ { \theta } - \phi \big ) + \big ( v _ { \theta } ^ { \star } - v ^ { \star } \big ) \| \leq \Delta t \Big ( \underbrace { \| \phi _ { \theta } - \phi \| } _ { \mathrm { S t e p } 2 } + \underbrace { \| v _ { \theta } ^ { \star } - v ^ { \star } \| } _ { \mathrm { S t e p } 3 } \Big ) . } \end{array}
$$

The factor $\Delta t$ in front is the reason the statement is a one step result. It is also why both bracketed terms must be bounded by quantities that vanish with the model errors, which is what the next two steps establish.

Step 2, the integral law term. By (5) the two integral laws are the same expression evaluated on different objects,

$$
\phi = \frac { \partial k } { \partial { \pmb x } } { \cal F } + \frac { \alpha _ { \phi } } { 2 } ( k - { \pmb u } ) , \qquad \phi _ { \theta } = \frac { \partial \pi _ { \theta } } { \partial { \pmb x } } { \cal F } _ { \theta } + \frac { \alpha _ { \phi } } { 2 } ( \pi _ { \theta } - { \pmb u } ) .
$$

The difficulty is the leading term, in which both factors differ. Inserting the hybrid ${ \frac { \partial \pi _ { \theta } } { \partial { \boldsymbol { x } } } } F$ , which uses the learned Jacobian against the true dynamics, turns that product difference into two terms each having one factor fixed,

$$
\phi _ { \theta } - \phi = \underbrace { \frac { \partial \pi _ { \theta } } { \partial x } \bigl ( F _ { \theta } - F \bigl ) } _ { \mathrm { d y n a m i c s ~ e r r o r } } + \underbrace { \Bigl ( \frac { \partial \pi _ { \theta } } { \partial x } - \frac { \partial k } { \partial x } \Bigr ) F } _ { \mathrm { J a c o b i a n ~ e r r o r } } + \underbrace { \frac { \alpha _ { \phi } } { 2 } \bigl ( \pi _ { \theta } - k \bigl ) } _ { \mathrm { p o l i c y ~ e r r o r } } .
$$

Taking norms and applying Assumption 1 to each factor in turn, with $\| \partial \pi _ { \theta } / \partial x \| \leq L _ { \pi }$ because $\pi _ { \theta }$ is $L _ { \pi }$ Lipschitz and $\| F \| \le B _ { F }$

$$
\begin{array} { r } { \| \phi _ { \theta } - \phi \| \leq L _ { \pi } \underbrace { \| F _ { \theta } - F \| } _ { \leq \varepsilon _ { F } } + B _ { F } \underbrace { \left\| \frac { \partial \pi _ { \theta } } { \partial x } - \frac { \partial k } { \partial x } \right\| } _ { \leq \varepsilon _ { k ^ { \prime } } } + \frac { \alpha _ { \phi } } { 2 } \underbrace { \| \pi _ { \theta } - k \| } _ { \leq \varepsilon _ { k } } \leq B _ { F } \varepsilon _ { k ^ { \prime } } + L _ { \pi } \varepsilon _ { F } + \frac { \alpha _ { \phi } } { 2 } \varepsilon _ { k } , } \end{array}
$$

which is the integral law group of (13). Each of its three terms is attributable to one learned object, so a reader can see which network is responsible for which part of the bound.

Step 3, the quadratic program term. The one row program (3) minimizes $\| \pmb { v } \| ^ { 2 }$ subject to the single constraint $p ^ { \top } { \pmb v } \geq q ,$ and this problem can be solved in closed form. If $q \leq 0$ the unconstrained minimizer $v = 0$ is already feasible, and if $q > 0$ the constraint is active and the minimum norm solution is the projection onto the constraint boundary along $p .$ . The two cases combine into

$$
v ^ { \star } = \lambda p , \qquad \lambda = \frac { [ q ] _ { + } } { \| p \| ^ { 2 } } ,
$$

and likewise $\pmb { v } _ { \theta } ^ { \star } = \lambda _ { \theta } p _ { \theta }$ with $\lambda _ { \theta } = [ q _ { \theta } ] _ { + } / \lVert p _ { \theta } \rVert ^ { 2 }$ . The deviation is again a difference of products, so we insert the hybrid $\lambda p _ { \theta }$

$$
\begin{array} { r } { v _ { \theta } ^ { \star } - v ^ { \star } = \lambda _ { \theta } p _ { \theta } - \lambda p = ( \lambda _ { \theta } - \lambda ) p _ { \theta } + \lambda ( p _ { \theta } - p ) , \qquad \mathrm { h e n c e } \qquad \| v _ { \theta } ^ { \star } - v ^ { \star } \| \leq | \lambda _ { \theta } - \lambda | \| p _ { \theta } \| + \lambda \| p _ { \theta } - p \| . } \end{array}
$$

It remains to bound the scalar multiplier gap $| \lambda _ { \theta } - \lambda |$ . Splitting it so that the numerator and the denominator vary one at a time,

$$
\lambda _ { \theta } - \lambda = \frac { [ q _ { \theta } ] _ { + } - [ q ] _ { + } } { \| p _ { \theta } \| ^ { 2 } } + [ q ] _ { + } \Big ( \frac { 1 } { \| p _ { \theta } \| ^ { 2 } } - \frac { 1 } { \| p \| ^ { 2 } } \Big ) .
$$

For the first term, $t \mapsto [ t ] _ { + }$ is 1 Lipschitz, so $\left| [ q _ { \theta } ] _ { + } - [ q ] _ { + } \right| \leq | q _ { \theta } - q | \leq \varepsilon _ { q }$ , and $\| p _ { \theta } \| \geq \sigma$ by the regularity part of Assumption 1. For the second, $[ q ] _ { + } \leq \bar { q }$ and

$$
\Big | \frac { 1 } { \| p _ { \theta } \| ^ { 2 } } - \frac { 1 } { \| p \| ^ { 2 } } \Big | = \frac { \big | \| p \| ^ { 2 } - \| p _ { \theta } \| ^ { 2 } \big | } { \| p _ { \theta } \| ^ { 2 } \| p \| ^ { 2 } } = \frac { \big | \| p \| - \| p _ { \theta } \| \big | \left( \| p \| + \| p _ { \theta } \| \right) } { \| p _ { \theta } \| ^ { 2 } \| p \| ^ { 2 } } \le \frac { \varepsilon _ { p } \cdot 2 \bar { p } } { \sigma ^ { 4 } } ,
$$

where the numerator uses the reverse triangle inequality $| | p | | - | | p _ { \theta } | | \big | \leq | | p - p _ { \theta } | | \leq \varepsilon _ { p }$ together with $\| p \| , \| p _ { \theta } \| \leq { \bar { p } } .$ , and the denominator uses $\| p \| , \| p _ { \boldsymbol { \theta } } \| \dot { \geq } \sigma$ . Combining the two terms,

$$
| \lambda _ { \theta } - \lambda | \leq \frac { \varepsilon _ { q } } { \sigma ^ { 2 } } + \frac { 2 \bar { p } \bar { q } } { \sigma ^ { 4 } } \varepsilon _ { p } .
$$

Substituting this into the earlier estimate, and using $\| p _ { \theta } \| \leq { \bar { p } }$ on the first term together with $\lambda =$ $[ q ] _ { + } / \lVert p \rVert ^ { 2 } \leq \bar { q } / \sigma ^ { 2 }$ and $\| p _ { \theta } - p \| \leq \varepsilon _ { p }$ on the second,

$$
\| { \pmb v } _ { \theta } ^ { \star } - { \pmb v } ^ { \star } \| \leq \bar { p } \Big ( \frac { \varepsilon _ { q } } { \sigma ^ { 2 } } + \frac { 2 \bar { p } \bar { q } } { \sigma ^ { 4 } } \varepsilon _ { p } \Big ) + \frac { \bar { q } } { \sigma ^ { 2 } } \varepsilon _ { p } = \frac { \bar { p } } { \sigma ^ { 2 } } \varepsilon _ { q } + \Big ( \frac { \bar { q } } { \sigma ^ { 2 } } + \frac { 2 \bar { p } ^ { 2 } \bar { q } } { \sigma ^ { 4 } } \Big ) \varepsilon _ { p } ,
$$

which is the QP solution map group of (13). Adding this to Step 2 and multiplying by $\Delta t$ proves (13). The two appearances of $\sigma$ in the denominators arise from different places, one from inverting $\| \boldsymbol { p } \| ^ { 2 }$ and one from differentiating that inverse, which is why the worst dependence is $\sigma ^ { - 4 }$ rather than $\sigma ^ { - 2 }$

Step 4, the row errors. Step 3 treated $\varepsilon _ { p }$ and $\varepsilon _ { q }$ as given. We now bound them by the network errors, so that the whole estimate rests on measurable quantities.

For the row, (6) gives $\begin{array} { r } { p ^ { \top } = \nabla h ^ { \top } \frac { \partial F } { \partial { \bf u } } } \end{array}$ , a product in which both factors are learned. Inserting the hybrid $\nabla h _ { \theta } ^ { \top } \frac { \partial F } { \partial u }$

$$
p _ { \theta } - p = \Big ( \frac { \partial F _ { \theta } } { \partial u } - \frac { \partial F } { \partial u } \Big ) ^ { \top } \nabla h _ { \theta } + \Big ( \frac { \partial F } { \partial u } \Big ) ^ { \top } \big ( \nabla h _ { \theta } - \nabla h \big ) , \qquad \mathrm { s o } \qquad \varepsilon _ { p } \leq \left\| \frac { \partial F } { \partial u } \right\| \varepsilon _ { h ^ { \prime } } + \left\| \nabla h _ { \theta } \right\| \varepsilon _ { F ^ { \prime } } .
$$

Only Jacobian errors appear here, not prediction errors, which is the reason Section 6 validates $\partial F _ { \theta } / \partial \pmb { u }$ directly rather than reporting rollout accuracy alone.

For the right hand side, Definition 2 applied to $h _ { e }$ gives $\begin{array} { r } { q = - \gamma ( h _ { e } ) - \frac { \partial h _ { e } } { \partial x } F - p ^ { \top } \phi } \end{array}$ , and we bound the three terms separately.

• The class K term is Lipschitz by Assumption 1, so $| \gamma ( h _ { e , \theta } ) - \gamma ( h _ { e } ) | \leq L _ { \gamma } \big | h _ { e , \theta } - h _ { e } \big | \leq$ $L _ { \gamma } \varepsilon _ { h _ { e } }$

• The drift term is a product in which both factors are learned. Inserting the hybrid ${ \frac { \partial h _ { e , \theta } } { \partial { \boldsymbol { x } } } } F ,$

$$
\frac { \partial h _ { e , \theta } } { \partial { \boldsymbol x } } F _ { \theta } - \frac { \partial h _ { e } } { \partial { \boldsymbol x } } F = \frac { \partial h _ { e , \theta } } { \partial { \boldsymbol x } } \big ( F _ { \theta } - F \big ) + \Big ( \frac { \partial h _ { e , \theta } } { \partial { \boldsymbol x } } - \frac { \partial h _ { e } } { \partial { \boldsymbol x } } \Big ) F ,
$$

whose norm is at most $B _ { h ^ { \prime } } \varepsilon _ { F } + B _ { F } \varepsilon _ { h _ { e } ^ { \prime } }$ by the bounds $\| \partial h _ { e , \theta } / \partial \pmb { x } \| \le B _ { h ^ { \prime } }$ and $\| F \| \le B _ { F }$

• The nominal term is handled the same way, inserting the hybrid $p _ { \theta } ^ { \top } \phi$

$$
\boldsymbol { p } _ { \theta } ^ { \intercal } \phi _ { \theta } - \boldsymbol { p } ^ { \intercal } \phi = \boldsymbol { p } _ { \theta } ^ { \intercal } \left( \phi _ { \theta } - \phi \right) + \left( \boldsymbol { p } _ { \theta } - \boldsymbol { p } \right) ^ { \intercal } \phi ,
$$

whose norm is at most $\bar { p } \varepsilon _ { \phi } + B _ { \phi } \varepsilon _ { p }$ by $\| p _ { \theta } \| \leq { \bar { p } }$ and $\| \phi \| \leq B _ { \phi }$ Summing the three contributions,

$$
{ \varepsilon } _ { q } \ \leq \ L _ { \gamma } { \varepsilon } _ { h _ { e } } + B _ { h ^ { \prime } } { \varepsilon } _ { F } + B _ { F } { \varepsilon } _ { h _ { e } ^ { \prime } } + \bar { p } { \varepsilon } _ { \phi } + B _ { \phi } { \varepsilon } _ { p } ,
$$

as claimed. The two barrier errors on the right are themselves controlled by the smoothness granted in Assumption 1, since $h _ { e , \theta }$ is assembled from $h _ { \theta }$ and $F _ { \theta }$ by (6), so that $\varepsilon _ { h _ { e } }$ is controlled by the $C ^ { 1 }$ error of $F _ { \theta }$ together with the $C ^ { 1 }$ error of $h _ { \theta } .$ , and $\varepsilon _ { h _ { e } ^ { \prime } }$ by those errors at one order higher, which is where the $C ^ { 2 }$ requirement on $h _ { \theta }$ is used. Every term in (13) is therefore proportional to an error of one of the three networks, and the bound vanishes when all three are exact.

## A.3. Low dimensional systems

The vehicle is a control affine kinematic bicycle, ${ \dot { \pmb x } } = f _ { 1 } ( { \pmb x } ) { \pmb u }$ , and must reach a goal while avoiding M circular obstacles, while the quadrotor with a suspended pendulum uses the dynamics of Beard (2008) with M spherical obstacles. Expert trajectories come from nonlinear MPC solved with do-mpc. Table 3 reports the average time to reach the goal from 100 sampled initial states together with the average closed loop quadratic cost in state and input. Nonlinear MPC attains the lowest cost at the highest runtime, since it solves non convex programs online, whereas our filter stays within 1% of that optimum at a fraction of the runtime, because the program (9) deviates minimally from the unconstrained solution. ORCA-MAPF (Dergachev and Yakovlev, 2021) and CADRL (Chen et al., 2017) serve as runtime reference points only, since both were designed for reciprocal multi agent collision avoidance rather than as safety filters, so the cost column should not be read as a like for like comparison against them.

## A.4. Ablations

Ablations. Table 4 isolates three choices on the held out layouts at 4.0 m/s on which MPPI alone is unsafe. Twelve layouts cannot separate outcome counts, so the discriminating quantities are the clearance reached and the use made of the relaxation. Every configuration kept the program feasible at every step, including those that degrade safety, which is Proposition 1 at work: feasibility follows from the structure of (9) rather than from the quality of $h _ { \theta }$ or the choice of gains. Of the three, the grip row matters most, since it is the only state-input row and without it the filter may command lateral accelerations the tires cannot deliver. The smallest obstacle barrier value reached falls from +0.29 to −0.05, so the vehicle leaves the learned safe set, and slack becomes active on 2.77% of steps against 0.16% with the row present. The filter stays feasible and still intervenes, but it is enforcing a barrier through a model of the input response that the plant no longer obeys. The margin and the gain act on that same clearance from opposite directions. Removing the margin, $\eta = 0 ,$ , reduces the smallest barrier value reached to +0.20, which is what the margin was introduced to buy and matches the open loop false safe rates of Section 6.2, while raising the gain from $\alpha = 5$ to $\alpha = 1 0$ halves it $\mathrm { t o \ t o . 1 4 }$ , because a larger α lets the extended barrier tolerate a faster approach while $h _ { \theta }$ is still positive. Neither changes the outcome counts on these layouts.

<table><tr><td></td><td colspan="2">Vehicle</td><td colspan="2">Quadrotor-pendulum</td></tr><tr><td>Method</td><td>Time (s)</td><td>Cost</td><td>Time (s)</td><td>Cost</td></tr><tr><td>NMPC ORCA-MAPF CADRL</td><td>5.562 1.373 0.837</td><td>4.583 6.839 5.683</td><td>10.242 2.328 2.181</td><td>15.232 21.281 19.283</td></tr></table>

Table 3: Time to goal and average closed loop cost on the two low dimensional systems. ORCA-MAPF and CADRL are runtime reference points rather than safety filters, so the cost column is not a like for like comparison against them.

<table><tr><td>Configuration</td><td>Success</td><td>Unsafe</td><td>min hθ</td><td>QP feasible</td><td>Slack active</td><td>Interventions</td></tr><tr><td>Default (η = 0.2, α = 5, grip row)</td><td>8/12</td><td>0/12</td><td>+0.29</td><td>100%</td><td>0.16%</td><td>44%</td></tr><tr><td>No enforcement margin (η = 0)</td><td>9/12</td><td>0/12</td><td>+0.20</td><td>100%</td><td>0.13%</td><td>45%</td></tr><tr><td>Barrier gain α = 10</td><td>9/12</td><td>0/12</td><td>+0.14</td><td>100%</td><td>0.15%</td><td>45%</td></tr><tr><td>No grip row</td><td>7/12</td><td>0/12</td><td>-0.05</td><td>100%</td><td>2.77%</td><td>31%</td></tr></table>

## A.5. Behavior under plant mismatch

Behavior under plant mismatch. Every learned component assumes the friction its training data was generated at, $\mu = 1 . 0 { \mathrm { : } }$ , so lowering the friction of the plant alone probes the regime that the split in (9) was designed for. On 16 held out layouts at 4.0 m/s with all three models left unchanged, the planner alone is unsafe on 3 layouts at $\mu = 0 . 8 5$ and on 7 at $\mu = 0 . 7 0$ , against 0 and 1 for the filtered system, and it reaches the

Table 4: Ablations on the 12 held out layouts at 4.0 m/s on which MPPI alone is unsafe, with MPPI as the nominal controller. min $h _ { \theta }$ is the smallest obstacle barrier value reached in closed loop, before the enforcement margin is subtracted, so a negative entry means the learned safe set was left. Interventions is the fraction of steps at which $\pmb { v } ^ { \star } \neq 0$

goal on 10 and 6 against 13 and 11. Both gaps widen as the plant departs from the models, because under mismatch it is the planner and not the filter that degrades, and at $\mu = 0 . 7 0$ the safety difference approaches significance on 16 layouts alone $( p = 0 . 0 7 )$ . The mechanism is visible in the solver: slack activity, 0.16% of steps at matched friction, rises by roughly an order of magnitude to 1.4% and 1.0%, while (9) stays feasible at every step and no actuator limit is ever relaxed. The relaxation absorbs the model error exactly where Proposition 1 says it must.

## A.6. Algorithm

Input: expert $\pi _ { \mathrm { e x p } } .$ , sampling distributions, backup library M, horizon $T _ { \cdot }$ , gains $\alpha , \alpha _ { \phi } , \gamma , \gamma _ { u }$ , margin $\eta$

Output: $F _ { \theta } , \pi _ { \theta } , h _ { \theta }$ and the online filter

Stage 1 (dynamics). Excite the plant, collect ${ \mathcal { D } } _ { F } .$ , fit $F _ { \theta }$ by (4) and report the $\partial F _ { \theta } / \partial \pmb { u }$ error Stage 2 (nominal). Roll out $\pi _ { \mathrm { e x p } }$ on the unconstrained task, collect $\mathcal { D } _ { c } .$ , fit $\pi _ { \theta }$ Stage 3 (barrier). foreach sampled state $\mathbf { \mathcal { x } } _ { i }$ do

foreach $\mu \in \mathcal { M }$ do   
simulate $\mu$ from $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ through $F _ { \theta }$ with the actuator model for T and record $c _ { \mu } ( \pmb { x } _ { i } )$   
end   
y<sub>i</sub> ← tanh $\left( \operatorname* { m a x } _ { \mu } c _ { \mu } ( \pmb { x } _ { i } ) / \kappa \right)$

end

Fit $h _ { \theta }$ by (7), then pick η as the smallest margin with zero false safe rate on held out states Online. while running do

assemble $p _ { i } , q _ { i }$ from (6) by automatic differentiation through $h _ { \theta }$ and $F _ { \theta }$ build $\mathcal { V } ( \pmb { x } , \pmb { u } )$ from the actuator ranges and rates by (12) solve (9) for $v ^ { \star }$ , then set u $ { } { \pmb u } + ( \phi _ { \theta } + { \pmb v } ^ { \star } ) \Delta t$ and apply it

end

Algorithm 1: Neural ICBF safety filter: offline training and online use

## A.7. Network architectures and training

$F _ { \theta }$ is a three hidden layer network of width 256 with tanh activations, taking (cos ψ, sin ψ, $v _ { x } , v _ { y } , r , a _ { x } , \delta )$ and returning x˙ , where smooth activations are required because the filter differentiates through it twice. $\pi _ { \theta }$ is a four hidden layer network of width 256 with a tanh output scaled to the actuator ranges. Both barriers are three hidden layer networks of width 256 with a tanh output, taking the relative state and the obstacle radius. Training uses AdamW with cosine annealing from $1 0 ^ { - 3 }$ to $1 0 ^ { - 5 }$ . Unsafe states are weighted five times in (7).