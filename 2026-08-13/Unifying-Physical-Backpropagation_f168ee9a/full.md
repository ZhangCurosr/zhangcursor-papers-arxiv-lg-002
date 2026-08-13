# Unifying Physical Backpropagation

Cyrill B¨osch<sup>∗1</sup>, Yigithan Gediz<sup>2</sup>, and Hakan T¨ureci<sup>2</sup>

<sup>1</sup>Department of Computer Science, Princeton University, Princeton, NJ 08540, USA <sup>2</sup>Department of Electrical and Computer Engineering, Princeton University, Princeton, NJ 08540, USA

August 13, 2026

## Abstract

Physical computing systems exploit device dynamics for computation, but their gradient-based op timization is challenging: backpropagation through a digital twin sufers from model-reality gap. Ondevice gradient computation could resolve this issue, and a handful of theoretical and experimental studies have proposed ways to achieve it. Yet a unifying theory identifying when a physical system can compute the gradient of its own performance has been missing. Here we develop such a unification, based on the adjoint method: we identify suficient conditions under which the adjoint field required for formally exact gradients can be generated on the same hardware that performs the computation. Linear and nonlinear systems obey fundamentally diferent conditions: for linear systems damping or gain is admissible provided reciprocity is preserved. For nonlinear trajectory systems the suficient conditions are reciprocity of the linearized system and the existence of a time-reversal mirror. Algorithmically, the nonlinear case requires infinitesimal nudging, whereas linear systems admit a finite-amplitude experiment. We recover Equilibrium Propagation, Hamiltonian echo backpropagation, fully forward mode training and in situ gradient methods in integrated-photonic and free-space-optical systems. We further show that reciprocity is only the simplest instance of a more general intertwining condition, which extends exact on-device gradient computation to a class of non-Hermitian, non-reciprocal systems. Fur ther generalizations include time-dependent parameters, Onsager-reciprocal dynamics and nonlinear, PT-symmetric Schr¨odinger equations. Our work provides a unified theoretical basis for formally exact physical learning algorithms and a template for constructing them across a range of physical systems.

## Contents

1 Introduction 2

2 Introducing the adjoint method 3

3 What is computed physically, under what conditions and how: an overview 6

4 Local learning rules and “knowing the model” vs. “knowing the parameters” 7

5 Onsager reciprocity 9

6 Second-order real-valued systems 10

7 First-order real-valued and overdamped systems 14

8 Schr¨odinger-type complex systems 15   
9 Stationary problems 19   
10 Spatial symmetries and twisted reciprocity 22   
11 Conclusions 26   
Appendix 33

## 1 Introduction

Physical computing uses the controlled dynamics or steady-state response of a material or device for information processing [1–3]. Inputs may be encoded in initial or boundary conditions, external drives, or programmable parameters, and outputs are measured from selected physical degrees of freedom. Representative platforms include mechanical and elastic networks [4], analog electrical circuits [5], spintronic and memristive devices [6, 7], integrated photonic circuits [8], nonlinear multimode fibers [9], free-space optical systems [10–12] and programmable wave guide systems [13, 14]. These platforms span transient, fixed-point, and frequency-domain computations.

A common special case is Physical Reservoir Computing [15], in which the internal physical dynamics is fixed and only a linear read-out layer is trained. Here, instead, we address the optimization of parameters that enter the physical dynamics itself, which remains a major challenge [16]. Training only a digital twin [17] is vulnerable to a model–reality gap. Physics-aware training, in which the forward pass is performed in the physical device but diferentiation is performed using a digital model, can reduce this gap, but still requires a diferentiable surrogate for the backward pass and communication between the physical and digital domains [18, 19]. Over the past decade, in situ (equivalently, on-device) training methods, in which the physical system participates in both the forward computation and the generation of the learning signal, have emerged as a promising alternative [20]. These approaches range from heuristic or approximate rules to methods that compute formally exact gradients. Approximate rules can be simple and local, but their updates need not coincide with the true gradient, and their convergence and performance can therefore be system dependent [16]. This motivates formally exact physical-gradient constructions, which have been developed for free-space optics [21,22], integrated photonics [23], mechanics [24], resistive electrical networks [25], and Hamiltonian systems [26], among other platforms.

In this article, we unify these formally exact physical backpropagation methods using the adjoint method from PDE-constrained optimization [27–29]. The main contribution of this work is twofold. First, we show that seemingly disparate physical backpropagation algorithms are instances of a single adjoint construction: platform-specific methods become one family in a common language that lets those algorithms be compared and extended. Second, we identify suficient conditions under which the required adjoint field—and hence the cost function gradient—can be obtained from a physical experiment that runs in the same system. The only allowed modifications between the forward and adjoint experiments are changes of initial conditions, external forces, and reversal of explicit time dependences.

We show that linear and nonlinear systems give rise to diferent physical experiments and diferent conditions. Linear reciprocal systems admit direct, finite-amplitude adjoint experiments, and damping or gain is compatible with exact physical gradients as long as reciprocity is preserved. For example, we show for the first time that even in the overdamped, reciprocal case, the gradient of trajectories can be obtained physically, whereas in the nonlinear case trajectory learning is obstructed. Instead, under certain conditions, Equilibrium Propagation [30] for fixed point engineering emerges.

Nonlinear trajectory systems are more restrictive: they require infinitesimal nudging, multiple experiments, a particular form of time reversal symmetry and reciprocity of the linearized dynamics—the latter two are independent conditions. In the nonlinear regime, ordinary damping or gain is generically obstructive, and even the ability to experimentally swap damping and gain does not in general restore a same-hardware adjoint construction.

We show that the adjoint-based theory recovers a broad range of existing exact physical-backpropagation algorithms: Equilibrium Propagation [30], Hamiltonian echo backpropagation and its recurrent variants [26, 31], the Fully Forward Mode (FFM) algorithm [32], early reciprocity-based recurrent learning [20], and physical gradient algorithms from integrated photonics [23, 33], free-space and difractive optics [21, 22, 34–36], linear wave scattering [29, 37], static and topological mechanics [24, 38], and the projector-based analytical gradient for passive resistor networks [25]. In particular, we identify FFM training as a special case of a more general condition, in which the forward operator and its transpose are related by a constant invertible transformation. This viewpoint extends FFM from static optics to timedependent trajectory learning and yields exact physical gradients even in a class of non-Hermitian and non-reciprocal systems—which we call “twisted reciprocal”—illustrated on a Hatano–Nelson chain [39]. Furthermore, we provide various generalizations to non-autonomous, Parity-time symmetric and Onsager reciprocal systems.

Disclaimer. In this article we do not cover coupled [5, 40–44] and heuristic [45, 46] physical learning rules. We also do not discuss forward-forward and related local-learning schemes for physical systems, which forgo gradients of a global loss function in favor of layer-local objectives [47–49]. Furthermore, we do not cover probabilistic physical computing machines. Boltzmann machines [50, 51] and score-based generative models [52] have been shown to give rise to exact (unbiased) local learning rules, although these do not derive from the adjoint method. We consider a setup where the gradient is calculated from physically observed forward and adjoint fields. We are agnostic to how the parameter updates are then applied and assume for simplicity that an external teacher does so. This means we do not cover the problem of designing a physical system that adapts by itself [5, 26, 34, 42].

Structure. The structure of the article is as follows: In Sec. 2 we formally define the optimization problem and introduce the adjoint method exemplified in a second-order real-valued system. In Sec. 3 we precisely define which part of the gradient computation is performed by the physical system and provide a summary of our results across various systems. In Sec. 4 we introduce the notion of local learning rules and discuss the diference between knowing the model and knowing the parameters. In Sec. 5 we discuss reciprocity and the generalization to Onsager reciprocity. Then, in Sections 6–9 we present the main results: suficiency conditions for obtaining the adjoint fields physically in second- (Sec. 6) and firstorder overdamped (Sec. 7) real-valued systems, in Schr¨odinger-type complex-valued systems (Sec. 8), and in stationary problems, where Equilibrium Propagation and the Helmholtz-type constructions live side by side (Sec. 9). At the end of each section we identify which existing physical learning frameworks are recovered by the presented adjoint formalism. Finally, in Sec. 10 we show that reciprocity is the simplest instance of a more general intertwining condition, recover fully forward mode training as a special case, extend it to time-dependent trajectory learning, and obtain exact physical gradients in a class of non-Hermitian, non-reciprocal systems, exemplified by a Hatano–Nelson chain.

We also provide an extensive and self-contained appendix that derives and discusses all the diferent cases in more detail.

## 2 Introducing the adjoint method

Here we consider N-dimensional discretized systems governed by ODEs, though the theory developed also applies to fields which are governed by PDEs. If the ODE resulted from discretizing a continuous PDE through, e.g., the Finite Element method, then, in the language of PDE-constrained optimization, this would correspond to the principle of “discretize-then-optimize” [28]. We denote the physical state by ${ \bf u } ( t ) \in \mathbb { R } ^ { N }$ and the trainable parameters by $\mathbf { p } \in \mathbb { R } ^ { M }$ . Inputs may be encoded traditionally in initia conditions, external drives [53], or directly in parameters. The latter endows linear systems with nonlinear computing capabilities [37, 54, 55]. Suppressing explicit encoders, decoders, and batch averages, we write the cost as

![](images/e143d86ffa34a40399ed1cf3645b01027b3bfe370284790bbae7930cb1295c9e.jpg)  
Figure 1: Conceptual overview of optimizing a physical system with the adjoint method. The physical state u(t) evolves under the forward dynamics $\mathbf { M } \ddot { \mathbf { u } } + \mathbf { D } \dot { \mathbf { u } } + \mathbf { F } [ \mathbf { u } , \mathbf { p } ] = \mathbf { f } ( t )$ , with mass matrix M, damping matrix D, internal forces F carrying the trainable parameters p, external drive f, and initial conditions ${ \bf u } ( 0 ) = { \bf u } _ { 0 } , \dot { \bf u } ( 0 ) = { \bf v } _ { 0 }$ . The recorded trajectory and terminal state induce the cost $\chi [ \mathbf { u } ] = \psi [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) ] \cdot \mathsf { I }$ $\begin{array} { r l } { \int _ { 0 } ^ { T } \theta [ \mathbb { P } \mathbf { u } ( t ) ] d t } \end{array}$ , where P projects onto the measured output degrees of freedom and ψ and θ are the terminal and trajectory losses. Evaluating the derivatives of the losses provides the drive and initial conditions for the adjoint field ${ \bf a } ( t )$ . The adjoint state equation involves the transposes of the mass and damping matrices and the linearized internal forces $\mathbf { F } _ { \mathbf { u } } ^ { T }$ along the time-reversed forward trajectory $\mathbf { u } ( T - t )$ ; in general it therefore corresponds to a diferent physical system than the one at hand and must, in most cases, be simulated digitally. If the forward pass is obtained by a physical experiment and the adjoint pass via a digital twin, this is known as physics-aware training [18,19]. Finally, the gradient $d \chi / d \mathbf { p }$ is evaluated from overlap integrals of the recorded forward and adjoint fields, and the physical parameters are updated.

$$
\chi [ \mathbf { u } ] = \int _ { 0 } ^ { T } \theta \left[ \mathbb { P } \mathbf { u } ( t ) \right] d t + \psi \left[ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) \right] ,\tag{1}
$$

where $\mathbb { P } = \mathbb { P } ^ { T } = \mathbb { P } ^ { 2 }$ projects onto the measured or output degrees of freedom (DOFs) which encode the solution to the computation. The first term is a trajectory loss and the second a final-state loss.

We now introduce the adjoint method [27, 56–63] that we will use throughout the text. The constructions used here are rooted in PDE constrained optimization, where an objective is minimized subject to a governing equation enforced as a constraint through Lagrange multipliers [64]. A full self-contained derivation can be found in the Appendix A.

For concreteness, consider a second-order real-valued system

$$
\mathbf { M } ( \mathbf { p } _ { M } ) \ddot { \mathbf { u } } ( t ) + \mathbf { D } ( \mathbf { p } _ { D } ) \dot { \mathbf { u } } ( t ) + \mathbf { F } \big [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t \big ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) = \mathbf { 0 } , \qquad \mathbf { u } ( 0 ) = \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) , \qquad \dot { \mathbf { u } } ( 0 ) = \mathbf { v } _ { 0 } ( \mathbf { p } _ { v _ { 0 } } ) .\tag{2}
$$

Here M is the invertible mass matrix, D the damping or gain matrix, F the internal force, which carries the constitutive law of the system and is in general nonlinear in u and explicitly time dependent, and f the external drive; $\mathbf { u } _ { 0 }$ and $\mathbf { v } _ { 0 }$ are the initial state and its rate of change. We use mechanical language throughout, but the same three operators appear as the inertial, dissipative, and restoring terms of electrical, optical, and other physical platforms. Each operator carries its own block of trainable parameters, collected into $\mathbf { p } = \left( \mathbf { p } _ { M } , \mathbf { p } _ { D } , \mathbf { p } _ { F } , \mathbf { p } _ { f } , \mathbf { p } _ { u _ { 0 } } , \mathbf { p } _ { v _ { 0 } } \right)$ . The constrained optimization problem is

$$
\operatorname* { m i n } _ { \mathbf { p } } \ \chi [ \mathbf { u } ] \qquad \mathrm { s u b j e c t ~ t o } \qquad ( 2 ) .\tag{3}
$$

The adjoint method replaces the diferentiation of the full trajectory $\mathbf { u } ( t ; \mathbf { p } )$ with respect to every parameter by a single adjoint solve. Written in forward time, the adjoint field ${ \bf a } ( t )$ satisfies (see Appendix A)

$$
\mathbf { M } ^ { T } \ddot { \mathbf { a } } ( t ) + \mathbf { D } ^ { T } \dot { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } \big [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \mathbf { a } ( t ) + \mathbf { f } _ { \mathbf { a d j } } ( t ) = \mathbf { 0 } ,\tag{4}
$$

where $( \cdot ) _ { \mathbf { x } }$ denotes a partial derivative, i.e. $\mathbf { F } _ { \mathbf { u } } : = \partial \mathbf { F } / \partial \mathbf { u }$ (in the parameters, $\mathbf { p } _ { X }$ , the subscripts are labels). The adjoint source is given by

$$
\mathbf { f } _ { \mathrm { a d j } } ( t ) = \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ]\tag{5}
$$

and the adjoint initial conditions are

$$
\mathbf { M } ^ { T } \mathbf { a } ( 0 ) = - \mathbb { P } \psi _ { \dot { \mathbf { u } } } \big [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) \big ] ,\tag{6}
$$

$$
\mathbf { M } ^ { T } \dot { \mathbf { a } } ( 0 ) = - \mathbf { D } ^ { T } \mathbf { a } ( 0 ) - \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) \big ] .\tag{7}
$$

The derivative of the trajectory loss, $\mathbb { P } \theta _ { \mathbf { u } } ,$ is therefore a forcing term in the adjoint experiment. The derivatives of the final-state loss, $\mathbb { P } \psi _ { \mathbf { u } }$ and $\mathbb { P } \psi _ { \dot { \mathbf { u } } }$ , set the adjoint initial displacement and velocity. If there is no trajectory loss, the adjoint is unforced; if there is no terminal loss, the adjoint starts from zero.

Once ${ \bf a } ( t )$ is known, the gradient is obtained from overlap integrals (see Appendix A.9)

$$
\frac { d \chi } { d \mathbf { p } _ { M } } = \int _ { 0 } ^ { T } \langle \mathbf { a } ( T - t ) , \mathbf { M } _ { \mathbf { p } _ { M } } \ddot { \mathbf { u } } ( t ) \rangle d t ,\tag{8}
$$

$$
\frac { d \chi } { d \mathbf { p } _ { D } } = \int _ { 0 } ^ { T } \langle \mathbf { a } ( T - t ) , \mathbf { D } _ { \mathbf { p } _ { D } } \dot { \mathbf { u } } ( t ) \rangle d t ,\tag{9}
$$

$$
\frac { d \boldsymbol { \chi } } { d { \bf p } _ { F } } = \int _ { 0 } ^ { T } \left. { \bf a } ( T - t ) , { \bf F } _ { { \bf p } _ { F } } \left[ { \bf u } ( t ) , { \bf p } _ { F } , t \right] \right. d t ,\tag{10}
$$

$$
\frac { d \chi } { d { \bf p } _ { f } } = - \int _ { 0 } ^ { T } \left. { \bf a } ( T - t ) , { \bf f _ { p } } _ { f } ( t , { \bf p } _ { f } ) \right. d t ,\tag{11}
$$

$$
\frac { d \boldsymbol { \chi } } { d \mathbf { p } _ { u _ { 0 } } } = - \mathbf { \nabla } \left. \left[ \mathbf { M } ^ { T } \dot { \mathbf { a } } ( T ) + \mathbf { D } ^ { T } \mathbf { a } ( T ) \right] , \mathbf { u } _ { 0 , \mathbf { p } _ { u _ { 0 } } } \right. ,\tag{12}
$$

$$
\frac { d \chi } { d \mathbf { p } _ { v _ { 0 } } } = - \mathbf { \nabla } \left. \mathbf { M } ^ { T } \mathbf { a } ( T ) , \mathbf { v } _ { 0 , \mathbf { p } _ { v _ { 0 } } } \right. ,\tag{13}
$$

where $\langle \mathbf { x } , \mathbf { y } \rangle = \mathbf { x } ^ { T } \mathbf { y }$ is the Euclidean inner product. Thus the gradient with respect to dynamical parameters is an overlap between the adjoint field and the parameter sensitivity of the respective forward operators. The gradient with respect to initial-condition parameters is an endpoint overlap. In particular, if the initial displacement and velocity themselves are trainable, then

$$
\frac { d \boldsymbol \chi } { d \mathbf { u } _ { 0 } } = - \mathbf { M } ^ { T } \dot { \mathbf { a } } ( T ) - \mathbf { D } ^ { T } \mathbf { a } ( T ) , \qquad \frac { d \boldsymbol \chi } { d \mathbf { v } _ { 0 } } = - \mathbf { M } ^ { T } \mathbf { a } ( T ) .\tag{14}
$$

Interpretation. The overlap integrals show that the time-reversed adjoint field ${ \bf { a } } ( T - t )$ encodes the sensitivity of the loss with respect to a force applied at time t during the forward evolution. All parameters act in this way. Writing the state equation as $\mathbf { M } \ddot { \mathbf { u } } + \mathbf { D } \dot { \mathbf { u } } + \mathbf { F } = \mathbf { f }$ , a parameter change $\delta \mathbf { p }$ perturbs the net force acting on the system along the forward trajectory by

$$
\delta { \bf f } ( t ) = { \bf f } _ { { \bf p } _ { f } } \delta { \bf p } _ { f } - { \bf M } _ { { \bf p } _ { M } } \ddot { \bf u } ( t ) \delta { \bf p } _ { M } - { \bf D } _ { { \bf p } _ { D } } \dot { \bf u } ( t ) \delta { \bf p } _ { D } - { \bf F } _ { { \bf p } _ { F } } \left[ { \bf u } ( t ) , { \bf p } _ { F } , t \right] \delta { \bf p } _ { F } ,\tag{15}
$$

the internal contributions entering with the opposite sign because they act against the drive. The trajectory gradients (8)–(11) then collapse into the single statement $\begin{array} { r } { \delta \chi = - \int _ { 0 } ^ { T } \langle \mathbf { a } ( T - t ) , \delta \mathbf { f } ( t ) \rangle } \end{array}$ dt. If the force change aligns with the adjoint field, the cost therefore decreases at the largest rate per unit of forcing, while a force orthogonal to it leaves the cost unchanged. The endpoint overlaps (12) and (13) read the same way for the initial conditions.

The arrow of time. As discussed in Appendix A.10, the adjoint state as it emerges from the derivation evolves backward in time. The equations above already incorporate the change of variable $t \mapsto T - t$ that recasts this evolution in forward time; this will become crucial for the physical realization of the adjoint field in Sec. 6.

Algorithm. The algorithm is summarized in Fig. 1. One first solves the forward problem for the forward field ${ \bf \delta u } ( t )$ , either digitally or, in physics-aware training, on the physical system itself. Next, the adjoint problem—which in general difers from the forward problem through the transposes and the linearization— is solved digitally for ${ \bf a } ( t )$ . The gradient then follows from the overlap integrals in (8)–(13), and the parameters are finally updated according to a chosen optimization algorithm.

A brief note on computational complexity. The advantage of the adjoint method in digital optimization is that the cost of this gradient is essentially independent of the number of parameters [28,56,57]. Finite-diference methods and forward sensitivity analysis require a separate solve for every entry of p, so their computational cost grows linearly with the parameter count dim(p). The adjoint method, by contrast, requires only two trajectory solves regardless of how many parameters are trained: one forward solve for u(t) and one adjoint solve for a(t). All gradient components are then read of from the overlap integrals (8)–(13), which are cheap because the only ingredients they require are the parameter sensitivities of the forward operators, such as $\mathbf { M } _ { \mathbf { p } _ { M } } , \mathbf { D } _ { \mathbf { p } _ { D } } ,$ and $\mathbf { F } _ { \mathbf { p } _ { F } }$ which are typically sparse, structured, and available in closed form from the model itself (see Sec. 4).

How to test the gradient? To test any adjoint gradient implementation one can employ a Taylor test.   
We detail this procedure in Appendix I.

Forward and adjoint systems are diferent. The adjoint construction is exact and fully general: it applies to any physical system, linear or nonlinear, conservative or dissipative. The adjoint problem (4) involves transposes and derivatives of the internal forces, and is hence in general diferent from the physical system that evolves with the forward state equation (2).

## 3 What is computed physically, under what conditions and how: an overview

The question we ask in this article is: under what conditions and how can the adjoint field be obtained from a physical experiment in the same system? By the same system we mean the same hardware and trainable parameters; we may only change initial conditions and applied sources, and reverse explicit time dependences.

All constructions below have the same form: run the forward dynamics and record u(t); construct adjoint sources $\left( \mathrm { E q . \ ( 5 ) } \right)$ and initial conditions (Eqs. (6), (7)) from the loss derivatives; obtain the adjoint field ${ \bf a } ( t )$ from a second physical experiment; then evaluate the gradient by an overlap between the measured forward/adjoint fields and known parameter sensitivities (Eqs. (8)–(13)).

The distinction between linear and nonlinear systems is the central organizing principle of the paper. Across all systems we consider, the conditions and algorithms can be summarized on a high level as:

## Suficient physical adjoint conditions and constructions.

• Linear systems: Reciprocity, or Onsager reciprocity in an appropriate metric, is suficient; Hermiticity is not required, so loss and gain are permitted. The adjoint field is obtained by a direct finite-amplitude run of the same physical system with adjoint sources and initial conditions and any explicit time dependence played in reverse.

• Nonlinear trajectory systems: The forward evolution must be recoverable by $\mathrm { ~ a ~ } ( \mathcal { P T } - )$ timereversal operation solely on the final state and the linearized dynamics around the reversed trajectory must be self-adjoint in the sense the adjoint equation demands. The adjoint field is then obtained as the infinitesimal response of a nudged time-reversed trajectory. Explicit time dependences are played in reverse.

Stationary linear problems. In linear (Onsager) reciprocal fixed-point or Helmholtz-type systems the adjoint field can be obtained directly with a single finite-amplitude experiment.

Stationary nonlinear problems. In nonlinear stationary systems, the exact adjoint is obtained in the infinitesimal limit from the diference between a free solution and a nudged solution, provided the system relaxes to the steady state and the tangent operators there are self-adjoint in the same sense. This is Equilibrium Propagation [30].

For real fields this self-adjointness is reciprocity, $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } } .$ For complex fields the two Wirtinger tangent operators are constrained separately. In Sec. 10 we show that in linear systems the reciprocity condition can be relaxed to a more general intertwining condition that allows us to extend physical backpropagation to certain types of “twisted-reciprocal” non-Hermitian systems.

The conditions formulated here are suficient, but not necessary. In particular, there may be special cases in which an additional transformation or rescaling makes the adjoint field physically accessible in the same device, even when the stated conditions are not strictly satisfied. One example thereof is developed in Sec. 10, where a spatial symmetry restores physical access to the adjoint field in a class of linear systems that are not reciprocal at all.

Table 1 organizes the cases treated in the remainder of the article along these lines, and records for each of them whether the adjoint field is available on the device and which experiment or proposal in the literature realizes it.

## 4 Local learning rules and “knowing the model” vs. “knowing the parameters”

If the adjoint field can be observed physically and the parameter sensitivity is local, the remaining overlap gives a local learning rule. Consider two coupled Dufing oscillators with internal force

$$
\mathbf { F } ( \mathbf { u } , \mathbf { p } ) = \left[ \begin{array} { l } { k _ { 1 } u _ { 1 } + \beta _ { 1 } u _ { 1 } ^ { 3 } + c ( u _ { 1 } - u _ { 2 } ) } \\ { k _ { 2 } u _ { 2 } + \beta _ { 2 } u _ { 2 } ^ { 3 } + c ( u _ { 2 } - u _ { 1 } ) } \end{array} \right] , \qquad \mathbf { p } = ( k _ { 1 } , k _ { 2 } , \beta _ { 1 } , \beta _ { 2 } , c ) .\tag{16}
$$

For a parameter in the first oscillator, for example $\beta _ { 1 }$ , the parameter sensitivity is given by

$$
\mathbf { F } _ { \beta _ { 1 } } ( \mathbf { u } ) = \left[ \begin{array} { l } { u _ { 1 } ^ { 3 } } \\ { 0 } \end{array} \right] .\tag{17}
$$

The corresponding gradient contribution is therefore

$$
\frac { d \chi } { d \beta _ { 1 } } = \int _ { 0 } ^ { T } a _ { 1 } ( T - t ) u _ { 1 } ( t ) ^ { 3 } d t .\tag{18}
$$

Table 1 : The cases t reat ed in t his art icle . The first split is linear versus nonlinear t he second is t he st ruct ure t hat makes t he adj oint field physically accessible . $^ { 6 6 } \mathrm { O n }$ device<sup>”</sup> means the adj oint field or its complex conj ugate is produced by the hardware that ran the forward problem . A ⋆ marks a case available on device for which we are not aware of any in the literature . In all traj ectory cases K and F may carry an explicit externally programmable time dependence : the st ated conditions must then hold at every inst ant and the schedule is replayed in reverse during the adj oint run . Onsager reciprocity comes on top of this tree rather than branching it : every row h<sub>o</sub>ld<sub>s</sub> i<sub>n</sub> th<sub>e</sub> V-<sub>me</sub>t<sub>r</sub>i<sub>c w</sub>ith <sub>sources an</sub>d <sub>nu</sub>d<sub>ges pre</sub>-<sub>mu</sub>lti<sub>p</sub>li<sub>e</sub>d b<sub>y</sub> V <sub>an</sub>d $\mathbf { a } = \mathbf { V } ^ { - 1 } \mathbf { d } \ ( \mathrm { S e c . \ 5 } )$
<table><tr><td>System</td><td>Sufficient condition</td><td>Adjoint construction</td><td>Sec.</td><td>On device</td><td>Representative realization</td></tr><tr><td colspan="6">Reciprocal Second order, real</td></tr><tr><td></td><td> $\mathbf { M } ^ { T } = \mathbf { M }$  and  $\begin{array} { r } { \mathbf { D } ^ { T } = \mathbf { D } , } \end{array}$  both constant in time, and  $\mathbf { K } ^ { T } = \mathbf { K } ;$  damping is no obstacle</td><td>Forward drive and initial conditions (ICs) swapped for adjoint source and ICs and applied at full amplitude</td><td>6.1</td><td>yes</td><td></td></tr><tr><td>First order, real</td><td> ${ \bf K } ^ { T } = { \bf K }$ </td><td>As above</td><td>7.1</td><td>yes</td><td>★ none; the trajectory rule of [65] is approximate</td></tr><tr><td>Schrödinger type</td><td> $\mathbf { K } ^ { T } = \mathbf { K } ;$  reciprocity and not Hermiticity, so gain and loss are admissible</td><td>As above; what propagates is the con- jugate adjoint  $\mathbf { c } = \mathbf { \overline { { a } } }$ </td><td>8.1</td><td>yes</td><td>In situ optical backpropagation [21]; holographic networks [34]</td></tr><tr><td>Iinsear syems Stationary, Helmholtz</td><td> ${ \mathbf K } ^ { T } = { \mathbf K }$ </td><td>Forward drive swapped for the adjoint source; applied at full amplitude</td><td>9.1</td><td>yes</td><td>Photonic meshes [23, 33]; wave scatter- ing [29,37]</td></tr><tr><td>Reciprocal, with a hardware symmetry</td><td> ${ \bf K } ^ { T } = { \bf K }$  and  $\mathbf { S } \mathbf { K } \mathbf { S } ^ { - 1 } = \mathbf { K }$  with  $\mathbf { S } \mathbb { P } \mathbf { S } ^ { - 1 } =$   $\mathbb { P } ^ { \mathrm { i n } }$ </td><td>The error signal enters where the data does and propagates forward; the ad- joint follows from  $\mathbf { a } = \mathbf { S } ^ { - 1 } \mathbf { d }$ </td><td>10</td><td>yes</td><td>Fully forward mode training [32], in static optics</td></tr><tr><td>Twisted reciprocal Non-reciprocal K</td><td>Orthogonal S with  $\mathbf { S K } _ { s } \mathbf { S } ^ { - 1 } \mathbf { \Phi } = \mathbf { \Phi } \mathbf { K } _ { s } ,$   $\mathbf { S K } _ { a } \mathbf { S } ^ { - 1 } = - \mathbf { K } _ { a } ; \mathbf { e . g . }$  a mirror-even Hatano- Nelson chain</td><td>One run with relocated sources; the adjoint follows from  $\mathbf { a } = \mathbf { S } ^ { - 1 } \mathbf { d }$ </td><td>10</td><td>yes</td><td>★ none</td></tr><tr><td colspan="6">Trajectories Second order, real,  $\mathbf { D } = \mathbf { 0 } ,$ </td></tr><tr><td>undamped</td><td> $\mathbf { M } ^ { T } = \mathbf { M }$  with  $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } }$  along the reversed trajectory</td><td>Infinitesimal response of a nudged time-reversed trajectory</td><td>6.2</td><td>yes</td><td>Hamiltonian echo backpropagation [26, 31]</td></tr><tr><td>Second order, real, damped</td><td>None: time reversal turns damping into gain but the adjoint requires damping</td><td>Obstructed</td><td>6.2</td><td>no</td><td>Contrastive trajectory rule [66], ap- proximate</td></tr><tr><td>First order, real, over- damped</td><td>None: reversal requires  $\mathbf { F } \mapsto - \mathbf { F } ,$  the adjoint  $+ \mathbf { F } _ { \mathbf { u } } ^ { T }$ </td><td>Obstructed; only the steady state sur- vives</td><td>7.2</td><td>no</td><td>Rayleighian trajectory rule [65], ap- proximate</td></tr><tr><td>Schrödinger, PT-TRM</td><td>PT symmetry of F and the parity-pulled- back tangent conditions; e.g. a Kerr medium then needs K Hermitian with  $\begin{array} { r } { \pmb { \Pi } \mathbf { K } \pmb { \Pi } = \mathbf { K } ^ { T } , } \end{array}$  admitting complex couplings.  $\pi = \textbf { I i s }$ </td><td>Infinitesimal response of a nudged PT- reversed trajectory, with the parity undone; gives  $\mathbf { c } = \overline { { \mathbf { a } } }$ </td><td>8.2</td><td>yes</td><td>★ none for ⅡI ≠ I; at  $\boldsymbol { \Pi } = \mathbf { I _ { \begin{array} { l } { [ 2 6 ] } } \end{array} }$ </td></tr><tr><td>Stationary problems</td><td>ordinary time reversal with real symmetric K</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="6">Real fixed point (Equi- librium Propagation)</td></tr><tr><td></td><td>A stable fixed point, and  $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } }$  there</td><td>Difference between the free and the nudged equilibrium</td><td>9.2</td><td>yes</td><td>Equilibrium Propagation (EP) [30]</td></tr><tr><td>Complex steady state (Complex-valued EP)</td><td>A stable steady state, with  $\mathbf { F _ { u } }$  Hermitian and  $\mathbf { F } _ { \overline { { \mathbf { u } } } }$  symmetric</td><td>As above</td><td>9.2</td><td>yes</td><td>★ none; [67] is holomorphic, [68, 69] approximate</td></tr><tr><td>Non-reciprocal tangent</td><td>The condition fails, so the nudged response no longer solves the adjoint equation</td><td>Obstructed on the unmodified device</td><td>9.2</td><td>no</td><td>Approximate extensions  $[ 7 0 , 7 1 ] ; [ 7 2 ]$  modifies the learning dynamics</td></tr></table>

Only the forward and adjoint trajectories of the first DOF are needed. Similarly, $\begin{array} { r } { d \chi / d k _ { 1 } = \int _ { 0 } ^ { T } a _ { 1 } ( T } \end{array}$ $t ) u _ { 1 } ( t ) d t$ . The coupling parameter requires both oscillators:

$$
\frac { d \chi } { d c } = \int _ { 0 } ^ { T } \left( a _ { 1 } ( T - t ) - a _ { 2 } ( T - t ) \right) \left( u _ { 1 } ( t ) - u _ { 2 } ( t ) \right) d t .\tag{19}
$$

These examples illustrate what we mean by a local learning rule (e.g. [5,30,41]): the gradient with respect to a given parameter depends only on the forward and adjoint dynamics of the degree(s) of freedom that this parameter couples to. This locality follows directly from the parameter sensitivity of the internal force $\mathbf { F } _ { ( \cdot ) }$ —and likewise of the mass, damping, and drive operators—being supported only on those degrees of freedom, so that the gradient overlap integral collapses onto them.

Not knowing the parameters. A further simplification arises when the parameters enter the forward operators linearly, as they do in Eq. (16). The sensitivity $\mathbf { F } _ { \mathbf { p } _ { F } }$ is then independent of $\mathbf { p } _ { F }$ , so the overlap integrals (18) and (19) are assembled entirely from the measured forward and adjoint trajectories and never reference the parameter values themselves. In an experiment the gradient is thus evaluated at whatever parameter values the device happens to sit at, so slow drift, aging, and unknown calibration ofsets are absorbed automatically instead of biasing the gradient estimate.

But knowing the model. Knowing which observable to record, and which overlap integral belongs to which parameter, is a statement about the structure of F and therefore still requires a model, in contrast to self-learning machines, which need none [5, 26, 73].

## 5 Onsager reciprocity

We show that the gradient can be obtained physically if the forward dynamics or its linearization are reciprocal. Formally, reciprocity means that the forward operators (or their linearization) are symmetric. However, there is a more general notion of reciprocity under which gradients remain physically realizable: as a self-adjointness condition in the inner product natural to irreversible dynamics. Let ${ \bf V } = { \bf V } ^ { T } \succsim 0$ (or Hermitian if complex-valued) be a constant Onsager mobility [74, 75], and define

$$
\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathbf { V } ^ { - 1 } } = \mathbf { x } ^ { T } \mathbf { V } ^ { - 1 } \mathbf { y } .\tag{20}
$$

The adjoint of a linear operator A in this inner product is

$$
\mathbf { A } ^ { \dagger } \mathbf { v } ^ { - 1 } = \mathbf { V } \mathbf { A } ^ { T } \mathbf { V } ^ { - 1 } ,\tag{21}
$$

since $\langle \mathbf { A } \mathbf { x } , \mathbf { y } \rangle _ { \mathbf { V } ^ { - 1 } } = \langle \mathbf { x } , \mathbf { A } ^ { \dagger } \mathbf { v } ^ { - 1 } \mathbf { y } \rangle _ { \mathbf { V } ^ { - 1 } }$ . We call A V-reciprocal if it is self-adjoint in this Onsager metric,

$$
\mathbf { V } \mathbf { A } ^ { T } \mathbf { V } ^ { - 1 } = \mathbf { A } .\tag{22}
$$

With $\mathbf { V } = \mathbf { I }$ , this reduces to the usual Euclidean reciprocity condition $\mathbf { A } ^ { T } = \mathbf { A }$ . Applied to the forward operators of a linear system, Eq. (22) is precisely Onsager reciprocity, and we call the linearized dynamics of a nonlinear system Onsager reciprocal if $\mathbf { F ( u ) } = \mathbf { V } E _ { \mathbf { u } } ( \mathbf { u } )$

Example: resistor–capacitor networks with heterogeneous capacitances. By Kirchhof’s current law, the node voltages u(t) of a passive resistor network with node capacitances and external current injections j(t) evolve according to

$$
\mathbf { C } \dot { \mathbf { u } } ( t ) + \mathbf { G } \mathbf { u } ( t ) - \mathbf { j } ( t ) = \mathbf { 0 } ,\tag{23}
$$

where ${ \bf G } = { \bf G } ^ { T }$ is the conductance matrix and C is the diagonal, positive matrix of node capacitances. Written in the first-order form of Sec. 7, $\dot { \bf u } + { \bf K } { \bf u } - { \bf C } ^ { - 1 } { \bf j } = { \bf 0 }$ with $\mathbf { K } = \mathbf { C } ^ { - 1 } \mathbf { G }$ . Unless all capacitances are identical, K is not Euclidean-symmetric, ${ \mathbf K } ^ { T } \neq { \mathbf K }$ . It is, however, exactly V-reciprocal with the constant mobility ${ \bf V } = { \bf C } ^ { - 1 }$ : Eq. (22) holds because $\mathbf { V } \mathbf { K } ^ { T } \mathbf { V } ^ { - 1 } = \mathbf { C } ^ { - 1 } \mathbf { G } ^ { T } = \mathbf { C } ^ { - 1 } \mathbf { G } = \mathbf { K }$ ; equivalently, $\mathbf { K u } = \mathbf { V } E _ { \mathbf { u } }$ derives from the energy $\begin{array} { r } { E ( \mathbf { u } ) = \frac { 1 } { 2 } \mathbf { u } ^ { T } } \end{array}$ Gu through the mobility $\mathbf { V } = \mathbf { C } ^ { - 1 }$ . If the linear resistors are replaced by diodes, transistors or memristive elements whose constitutive laws derive from a co-content function $\mathcal { E } ( \mathbf { u } )$ [42], the internal force $\mathbf { F } ( \mathbf { u } ) \ = \ \mathbf { C } ^ { - 1 } \boldsymbol { \mathcal { E } } _ { \mathbf { u } } ( \mathbf { u } ) \ = \ \mathbf { V } \boldsymbol { \mathcal { E } } _ { \mathbf { u } } ( \mathbf { u } )$ retains the constant-mobility gradient-flow structure, and the nonlinear fixed-point construction of Sec. 9.2 (Equilibrium Propagation) applies.

For the simplicity of the presentation, we set V = I throughout the main text. If $\mathbf { V } \neq \mathbf { I } ,$ the hardware propagates the V-weighted adjoint field instead. We show in Sec. 10 that this is an instance of a general intertwining construction, and that in practice the V-weighting is largely applied by the hardware itself (see also Appendix D). In Sec. 10 we also discuss the linear case in more detail and show that neither the symmetry nor the positivity of V is required: for linear systems, any constant, known, invertible intertwiner gives physical access to the adjoint field. In the nonlinear setting the two properties play distinct roles. Symmetry is what makes the linearized dynamics of the gradient flow $\mathbf { F } = \mathbf { V } E _ { \mathbf { u } }$ V-reciprocal. Positivity is used only in the fixed-point case, where it makes the relaxation a gradient descent (Sec. 9.2).

## 6 Second-order real-valued systems

Consider a second-order, real-valued and non-autonomous system introduced in Sec. 2. The explicit time dependence is assumed to be externally programmable, so that it can be replayed as $t \mapsto T - t$ during the adjoint experiment. We wish to shape its trajectory such that it minimizes the cost function (1). The adjoint field required to compute the loss function’s gradients Eqs. (8)–(13) can be obtained on the same system under the following suficient conditions:

Theorem 6.1 (Suficient conditions for physical adjoints for second-order real-valued systems). The adjoint field can be obtained on the same hardware in the following cases:

• Linear: Reciprocity. Suppose

$$
{ \bf F } [ { \bf u } , { \bf p } _ { F } , t ] = { \bf K } ( { \bf p } _ { K } , t ) { \bf u } ,
$$

and reciprocity holds, i.e.,

$$
\mathbf { M } ^ { T } = \mathbf { M } , \qquad \mathbf { D } ^ { T } = \mathbf { D } , \qquad \mathbf { K } ( \mathbf { p } _ { K } , t ) ^ { T } = \mathbf { K } ( \mathbf { p } _ { K } , t ) ,\tag{24}
$$

then the adjoint is propagated by the same linear hardware, with the explicit time dependence replayed in reverse and with adjoint source (5) and initial conditions (6), (7).

• Nonlinear: Existence of a Time Reversal Mirror (TRM) and reciprocity of the linearized dynamics. Suppose the time-reversed trajectory can be generated on the same hardware by an operation on the final state only (applying the TRM), together with the reversal of explicit time dependences. This means initializing with $\mathbf { u } ( T )$ and $- \dot { \mathbf { u } } ( T )$ . If, in addition,

$$
{ \bf D } = { \bf 0 } , \qquad { \bf M } ^ { T } = { \bf M } , \qquad { \bf F _ { u } } \big [ { \bf u } ( t ) , { \bf p } _ { F } , t \big ] ^ { T } = { \bf F _ { u } } \big [ { \bf u } ( t ) , { \bf p } _ { F } , t \big ] \quad f o r \ a l l \ t \in [ 0 , T ] ,\tag{25}
$$

then the adjoint is obtained as the infinitesimal response of a nudged time-reversed trajectory.

PT -TRM. The existence of TRM can be generalized if instead of time-reversal symmetry one has PT symmetry. We discuss this in Theorem 8.1 for nonlinear Schr¨odinger equations; for second-order realvalued systems the same logic covers undamped gyroscopic dynamics, $\mathbf { M } \ddot { \mathbf { u } } + \mathbf { G } \dot { \mathbf { u } } + \mathbf { F } [ \mathbf { u } ] = \mathbf { f }$ with ${ \bf G } ^ { T } =$

−G, as realized by Coriolis forces in rotating frames, Lorentz forces on charged lattices, and gyroscopic metamaterials [76]. A real orthogonal mirror $\mathbf { I I } = \mathbf { I I } ^ { T } = \mathbf { I I } ^ { - 1 }$ that anticommutes with G and leaves the remaining dynamics invariant along the trajectory,

$$
\mathbf { I G I I } = - \mathbf { G } , \qquad \mathbf { I I M I I } = \mathbf { M } , \qquad \mathbf { F } \big [ \mathbf { I I u } ( t ) , \mathbf { p } _ { F } , t \big ] = \mathbf { I I F } \big [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t \big ] ,
$$

implements the TRM as $\mathbf { w } ( t ) = \pmb { \Pi } \mathbf { u } ( T - t )$ , with the drive replayed as $\mathbf { f } ( t ) \mapsto \Pi \mathbf { f } ( T - t )$ . Theorem 6.1 then applies with the reciprocity condition pulled back by Π, $\Pi \mathbf { F _ { u } } [ \mathbf { \Pi } \mathbf { I } \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] \mathbf { H } = \mathbf { F _ { u } } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] ^ { T } . ^ { 1 }$

Onsager reciprocity. The extension to Onsager reciprocal systems as introduced in Sec. 5 can be found in Appendix D.

Non-autonomous dynamics and reciprocity. In Theorem 6.1 we allow the internal force F to be a function of time, i.e. the parameters may now specify a time-dependent schedule rather than being static. The reciprocity condition is instantaneous, i.e. it must hold at every time step, and the time evolution must be played backwards in time during the adjoint experiment. On the other hand, M and D need to be constant in time because otherwise the adjoint state equation will include time derivatives of those matrices and so the forward and adjoint systems will be fundamentally diferent.

## 6.1 Linear

The forward dynamics is

$$
\mathbf { M } \ddot { \mathbf { u } } ( t ) + \mathbf { D } \dot { \mathbf { u } } ( t ) + \mathbf { K } ( t ) \mathbf { u } ( t ) - \mathbf { f } ( t ) = \mathbf { 0 } .\tag{26}
$$

In the reciprocal case where $\mathbf { M } ^ { T } = \mathbf { M } , \mathbf { D } ^ { T } = \mathbf { D }$ , and $\mathbf { K } ( t ) ^ { T } = \mathbf { K } ( t )$ , the adjoint equation (4) becomes

$$
\mathbf { M } \ddot { \mathbf { a } } ( t ) + \mathbf { D } \dot { \mathbf { a } } ( t ) + \mathbf { K } ( T - t ) \mathbf { a } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } .\tag{27}
$$

Thus the adjoint experiment has exactly the same damped second-order form as the forward experiment. This proves the linear part of Theorem 6.1. □

Physical adjoint algorithm. One reuses the same hardware, replaces the external source by the adjoint source, replaces the initial displacement and velocity by the cost-derived adjoint initial conditions (no nudging required), and replays any explicit time dependence as $t \mapsto T - t$ . In this linear reciprocal regime the same damped hardware propagates the adjoint field (see Fig. 2, a).

Remark. If the adjoint source or the initial conditions are too strong, it may excite the system into a large amplitude regime where the linear equations no longer hold. In that case, the adjoint source and initial conditions can simply be rescaled by a constant, which rescales the norm of the gradient by the same constant.

Literature. In numerical wave-based inverse problems it is well known that the same operator governs the forward and the adjoint problem for reciprocal problems [28]. In physical learning, reciprocity has been used for backpropagation in an early work by Hermans et al. [20]. Berneman and Hexner [66] study a contrastive trajectory rule for damped reciprocal systems. Their algorithm relies on nudged trajectories and applies to periodic steady states or resting initial conditions. The adjoint framework computes the same gradients without nudging in the linear reciprocal case, and extends to arbitrary initial conditions, including gradients with respect to them. In Sec. 8 and Sec. 9 we discuss further references that harness reciprocity for same-device gradient calculation in Schr¨odinger (including free space optics) and stationary (including Helmholtz) problems, respectively.

a)  
![](images/88872914e18ffbe5f7e34538fd6e7175d8821fa6227161ca00e90fa0e708137e.jpg)

![](images/bdebb998ef72902b5fe9924859840b3355bed85565a070829c066e689657b55b.jpg)

![](images/7c5ec73c8cec17fcf03cb6a96b22dc7dc9bef465d7fe2273986c887635ff61bd.jpg)

![](images/167a9616e60876d44a7b924c6ab046c5345cb7acf52a53453d3a18ca73d4f1bc.jpg)  
Figure 2: A schematic illustration of the experiments to obtain the adjoint field physically. For the linear devices (a), reciprocity allows the same device to be used, with the drive and ICs adjusted for the forward and adjoint field propagations. Damping and gain are allowed, so long as reciprocity is preserved. In the nonlinear case (b), the existence of a time reversal mirror (TRM) [78] together with reciprocity of the linearized system provide indirect access to the adjoint field in the same hardware. To obtain the adjoint field physically, simultaneously apply the TRM to the final state of the forward run and perturb the timereversed trajectory with the adjoint drive and ICs. The adjoint field is then obtained as the (infinitesimal) diferential response of the reverse trajectory to the nudge.

## 6.2 Nonlinear

The nonlinear case is more restrictive because the experiment must reproduce the reversed forward trajectory before the infinitesimal nudge can reveal the adjoint. To see the obstruction, set $\mathbf { w } ( t ) = \mathbf { u } ( T - t )$ Evaluating (2) at $T - t$ gives

$$
\mathbf { M } \ddot { \mathbf { w } } ( t ) - \mathbf { D } \dot { \mathbf { w } } ( t ) + \mathbf { F } \big [ \mathbf { w } ( t ) , \mathbf { p } _ { F } , T - t \big ] - \mathbf { f } \big ( T - t , \mathbf { p } _ { f } \big ) = \mathbf { 0 } .\tag{28}
$$

The damping term has the wrong sign for the same hardware. Reproducing w would therefore require either $\mathbf { D = 0 }$ or an experimental replacement $\mathbf { D } \mapsto - \mathbf { D }$ , i.e. turning damping into gain. Even granting such a gain-flipped run does not fix the adjoint problem. A perturbation δw due to a small nudge into the direction of the adjoint source and around the gain-flipped reversed trajectory obeys (see Appendix B)

$$
\mathbf { M } \delta \ddot { \mathbf { w } } ( t ) - \mathbf { D } \delta \dot { \mathbf { w } } ( t ) + \mathbf { F } _ { \mathbf { u } } \big [ \mathbf { w } ( t ) , \mathbf { p } _ { F } , T - t \big ] \delta \mathbf { w } ( t ) + \epsilon \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] + \mathcal { O } ( \epsilon ^ { 2 } ) = \mathbf { 0 } , \quad | \epsilon | \ll 1 .\tag{29}
$$

The adjoint equation, however, propagates with $+ \mathbf { D } ^ { T } \dot { \mathbf { a } }$ . For reciprocal damping, $\mathbf { D } ^ { T } = \mathbf { D }$ , the two tangent operators difer by 2Da˙ . The same hardware cannot simultaneously reproduce the reversed nonlinear background trajectory and propagate the correct adjoint perturbation around it, except in the undamped case.

In the case of mass proportional damping, if one is able to flip damping (gain) to gain (damping) then the adjoint field can be obtained physically and the gradient can be calculated by exponential time weighting with the known damping coeficient [79] (Appendix C). This, however, falls outside our setting, since flipping the sign of the damping changes the hardware between the forward and the adjoint run.

If $\epsilon \to 0$ and $\mathbf { D = 0 }$ the perturbation dynamics (29) recovers the adjoint state equation (4) if $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } } -$ reciprocity along the forward trajectory. Then $\delta \mathbf { w } ( t )$ is proportional to the adjoint field, which can therefore be extracted from the diference of the reversed and reversed nudged dynamics (see Eq. (31)). This proves the nonlinear part of Theorem 6.1. □

Remark: only the tangent condition is used. We require $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } }$ only along the visited trajectory; no potential is needed for the construction. If this symmetry holds on an open neighborhood, the Poincar´e lemma gives $\mathbf { F } = E _ { \mathbf { u } }$ locally there [80]; if it holds throughout a simply connected state space, a single global potential exists. The same distinction applies in the complex case of Sec. 9.2: on an open set, condition (70) is equivalent to closedness of $2 \mathrm { R e } \langle { \bf F } , d { \bf u } \rangle _ { \mathbb { C } }$ and hence locally to (77).

Physical adjoint algorithm. One first runs the forward dynamics and records ${ \bf \delta u } ( t )$ ; the time-reversed trajectory $\mathbf { w } ( t ) = \mathbf { u } ( T - t )$ is then available directly from this recording and need not be regenerated physically. The adjoint requires only one further experiment, in which the TRM and the nudge are applied together: On the same (undamped) hardware, initialize at the final state with the velocity flipped and both initial conditions perturbed in the directions set by the terminal-loss derivatives,

$$
\begin{array} { r } { { \mathbf w } ^ { \epsilon } ( 0 ) = { \mathbf u } ( T ) - \epsilon { \mathbf M } ^ { - 1 } \mathbb { P } { \boldsymbol \psi } _ { \dot { \mathbf u } } , \qquad \dot { \mathbf { w } } ^ { \epsilon } ( 0 ) = - \dot { { \mathbf u } } ( T ) - \epsilon { \mathbf M } ^ { - 1 } \mathbb { P } { \boldsymbol \psi } _ { \mathbf u } , \qquad | \epsilon | \ll 1 , } \end{array}\tag{30}
$$

drive the system with the reversed external drive superimposed with the adjoint source, ${ \bf f } ( T - t ) -$ $\epsilon \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ]$ , and evolve it forward while replaying the explicit time dependence of the internal forces in reverse, $\mathbf { F } [ \cdot , \bar { \mathbf { p } _ { F } } , t ] \mapsto \mathbf { F } [ \cdot , \mathbf { p } _ { F } , T - t ]$ . Recording $\mathbf { w } ^ { \epsilon } ( t )$ , the adjoint field is recovered as the infinitesimal diferential response

$$
\mathbf { a } ( t ) = \operatorname* { l i m } _ { \epsilon  0 } \frac { \mathbf { w } ^ { \epsilon } ( t ) - \mathbf { w } ( t ) } { \epsilon } , \qquad \mathbf { w } ( t ) = \mathbf { u } ( T - t ) ,\tag{31}
$$

and the gradient is evaluated digitally from the overlap integrals (8)–(13) (recall $\mathbf { D } = \mathbf { 0 } ;$ see Fig. 2, b).

Symmetric nudging. At finite ϵ the extraction (31) is a forward diference, so the measured field is the adjoint up to an $\mathcal { O } ( \epsilon )$ bias from the second-order term of the perturbation expansion. Running instead two nudged experiments with opposite signs and forming the central diference,

$$
\mathbf { a } ( t ) = \operatorname* { l i m } _ { \epsilon  0 } \frac { \mathbf { w } ^ { + \epsilon } ( t ) - \mathbf { w } ^ { - \epsilon } ( t ) } { 2 \epsilon } ,\tag{32}
$$

leaves an $\mathcal { O } ( \epsilon ^ { 2 } )$ bias, at the cost of one additional experiment [31, 81].

Literature. For Hamiltonian dynamics with time-independent parameters, the construction above coincides with Hamiltonian echo backpropagation [26] and its recurrent extension by Pourcel and Ernoult [31], the latter shown to be equivalent to the continuous adjoint method; the self-adaptation of the physical parameters proposed by L´opez-Pastor and Marquardt [26], in which the parameters themselves become dynamical, is outside our scope. Our formulation additionally covers time dependent parameters of the internal force and generalizes to Onsager-weighted gradient flows $\mathbf { F } = \mathbf { V } E _ { \mathbf { u } }$ (Appendix D). A complementary, variational route recasts the learning signal through the classical action: Massar [82] and Pourcel et al. [79] generalize Equilibrium Propagation [30] from energy-based fixed points to full trajectories (Lagrangian Equilibrium Propagation), replacing its two-state energy contrast with a contrast of Lagrangian parameter-derivatives integrated over the trajectory. Because this contrast follows from an integration by parts, it carries boundary terms, so the resulting rule depends on the imposed boundary conditions which either vanish by construction (e.g. fixed or periodic endpoints) or persist as generally intractable residuals. For time-reversible systems, Pourcel et al. [79] identify a Parametric Final Value Problem boundary choice that renders these residuals tractable and recovers the recurrent Hamiltonian echo rule [31] as the physically practical instantiation of this variational construction.

## 6.3 Summary: linear and nonlinear systems behave fundamentally diferently

We have shown that linear systems give rise to fundamentally diferent, weaker conditions and simpler finite amplitude physical adjoint field constructions (Fig. 2). The simplicity of physical adjoint experiments for linear systems makes a strong case for parameter encoding schemes (structural nonlinearities) [53–55], where the linear system is endowed with nonlinear computational capabilities. Yet physical gradients are readily accessible with finite amplitude constructions and in the realistic damped setting [37]. Furthermore, damping and gain parameters, whose gradients are also available once the adjoint field is obtained (Eq. (9)) can also be harnessed to increase the expressivity of the physical device.

## 7 First-order real-valued and overdamped systems

We next consider first-order, real-valued, overdamped and non-autonomous dynamics

$$
\dot { \mathbf { u } } ( t ) + \mathbf { F } \big [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t \big ] - \mathbf { f } \big ( t , \mathbf { p } _ { f } \big ) = \mathbf { 0 } .\tag{33}
$$

The corresponding adjoint equation, derived in Appendix E, is

$$
\dot { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } \big [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \mathbf { a } ( t ) + \mathbf { f } _ { \mathrm { a d j } } ( t ) = \mathbf { 0 } ,\tag{34}
$$

with adjoint source and initial conditions given by

$$
\mathbf { f } _ { \mathrm { a d j } } ( t ) = \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] , \qquad \mathbf { a } ( 0 ) = - \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T ) \big ] .\tag{35}
$$

Again, we assume external control of the explicit time dependences of F and $\mathbf { f } ,$ so that it can be replayed as $t \mapsto T - t$ during the adjoint experiment.

Theorem 7.1 (Suficient conditions for physical adjoints for first-order real-valued systems). The adjoint field can be obtained on the same hardware in the following cases:

• Linear: Reciprocity. Suppose

$$
\mathbf { F } [ \mathbf { u } , \mathbf { p } _ { F } , t ] = \mathbf { K } ( \mathbf { p } _ { K } , t ) \mathbf { u }
$$

and reciprocity holds, i.e.,

$$
\mathbf { K } ( \mathbf { p } _ { K } , t ) ^ { T } = \mathbf { K } ( \mathbf { p } _ { K } , t ) ,\tag{36}
$$

then the adjoint is propagated by the same linear hardware, with the explicit time dependence replayed in reverse and with the adjoint source and initial conditions (35).

• Nonlinear, overdamped: trajectories are obstructed, the steady state is not.

## 7.1 Linear

The forward dynamics is

$$
\dot { \mathbf { u } } ( t ) + \mathbf { K } ( t ) \mathbf { u } ( t ) - \mathbf { f } ( t ) = \mathbf { 0 } .\tag{37}
$$

If the operator is reciprocal, $\mathbf { K } ( t ) ^ { T } = \mathbf { K } ( t )$ , the adjoint equation (34) becomes

$$
\dot { \mathbf { a } } ( t ) + \mathbf { K } ( T - t ) \mathbf { a } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } .\tag{38}
$$

Thus the adjoint experiment has exactly the same overdamped first-order form as the forward experiment and can hence be obtained in the same hardware. □

Physical adjoint algorithm. One reuses the same hardware, replaces the external source by the adjoint source, replaces the initial condition by the loss-derived adjoint initial condition, and replays any explicit time dependence as $t \mapsto T - t$

Literature. To the best of the authors’ knowledge this is the first theoretical demonstration that in linear, reciprocal but overdamped dynamics not only fixed points but also full trajectory gradients can be obtained physically. The Rayleighian trajectory rule of Berneman and Hexner [65] is approximate for trajectories.

## 7.2 Nonlinear

The first-order ODE of this section is the large-damping limit of the second-order ODE (Sec. 6). The nonlinear adjoint construction is therefore obstructed for the same reason discussed there: time reversal requires flipping damping into gain (here $\mathbf { F } \  \ - \mathbf { F } )$ while the adjoint requires damping. Trajectory gradients of nonlinear overdamped dynamics are therefore not accessible on the same hardware.

What remains accessible is the steady state. If F and f are time-independent and the dynamics converges to a linearly exponentially stable fixed point, the loss depends on the parameters only through that fixed point and the problem becomes stationary. That case is Equilibrium Propagation, treated in Sec. 9.2; its relation to the trajectory adjoint of this section is derived in Appendix H.3.

## 8 Schr¨odinger-type complex systems

We now consider complex fields governed by

$$
i \dot { \mathbf { u } } ( t ) + \mathbf { F } \big [ \mathbf { u } ( t ) , \overline { { \mathbf { u } } } ( t ) , \mathbf { p } _ { F } , t \big ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) = \mathbf { 0 } , \qquad \mathbf { u } ( 0 ) = \mathbf { u } _ { 0 } \big ( \mathbf { p } _ { u _ { 0 } } \big ) .\tag{39}
$$

Throughout this section derivatives are Wirtinger derivatives [83,84]: for $\begin{array} { r } { z = x + i y , \partial _ { z } = \frac { 1 } { 2 } ( \partial _ { x } - i \partial _ { y } ) } \end{array}$ and $\begin{array} { r } { \partial _ { \overline { { z } } } = \frac { 1 } { 2 } ( \partial _ { x } + i \partial _ { y } ) } \end{array}$ , and for vector fields we treat u and its complex conjugate u as formally independent variables. Thus $\mathbf { F _ { u } }$ and $\mathbf { F } _ { \overline { { \mathbf { u } } } }$ denote the two Wirtinger Jacobians, and for the real-valued cost functions the corresponding derivatives satisfy $\theta _ { \mathbf { u } } = \overline { { \theta _ { \overline { { { \mathbf { u } } } } } } }$ and $\psi _ { \mathbf { u } } = \overline { { \psi _ { \overline { { \mathbf { u } } } } } }$ and similarly for $\chi .$ As we will see, the physical adjoint construction delivers the complex conjugate of the adjoint field. Writing $\mathbf { c } ( t ) = \overline { { \mathbf { b } ( T - t ) } }$ , the forward-time complex conjugate adjoint equation derived in Appendix F is

$$
i \dot { \mathbf { c } } ( \boldsymbol { t } ) + \mathbf { F } _ { \mathbf { u } } ^ { T } \left[ \mathbf { u } ( T - \boldsymbol { t } ) , \overline { { \mathbf { u } } } ( T - \boldsymbol { t } ) , \mathbf { p } _ { F } , T - \boldsymbol { t } \right] \mathbf { c } ( \boldsymbol { t } ) + \overline { { \mathbf { F } _ { \mathbf { u } } \left[ \mathbf { u } ( T - \boldsymbol { t } ) , \overline { { \mathbf { u } } } ( T - \boldsymbol { t } ) , \mathbf { p } _ { F } , T - \boldsymbol { t } \right] } } ^ { T } \overline { { \mathbf { c } } } ( \boldsymbol { t } ) + \mathbf { f } _ { \mathrm { a d j } } ( \boldsymbol { t } ) = \mathbf { 0 } ,\tag{40}
$$

where ${ \overline { { ( \cdot ) } } } ^ { T }$ is the conjugate transpose, and with source and initial conditions

$$
\mathbf { f } _ { \mathrm { a d j } } ( t ) = \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) , \mathbb { P } \overline { { \mathbf { u } } } ( T - t ) \big ] , \qquad \mathbf { c } ( 0 ) = i \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \overline { { \mathbf { u } } } ( T ) \big ] .\tag{41}
$$

Complex conjugating Eq. (40) flips the signs in front of the Wirtinger Jacobians, which is why $\mathbf { c } ,$ rather than b, can be obtained physically in the same hardware.

Inner products now include complex conjugation, $\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathbb { C } } = \overline { { \mathbf { x } } } ^ { T } \mathbf { y }$ . The total variation of $\chi$ is therefore $\delta \chi = \langle d \chi / d \overline { { \mathbf { u } } } , \delta \mathbf { u } \rangle _ { \mathbb { C } } + \langle d \chi / d \mathbf { u } , \delta \overline { { \mathbf { u } } } \rangle _ { \mathbb { C } } = 2 \operatorname { R e } \langle d \chi / d \overline { { \mathbf { u } } } , \delta \mathbf { u } \rangle _ { \mathbb { C } }$ . The gradient overlap integrals expressed in terms of c then become (Appendix F):

$$
\frac { d \chi } { d \mathbf { p } _ { F } } = 2 \operatorname { R e } \int _ { 0 } ^ { T } \left. \overline { { \mathbf { c } ( T - t ) } } , \mathbf { F } _ { \mathbf { p } _ { F } } \left[ \mathbf { u } ( t ) , \overline { { \mathbf { u } } } ( t ) , \mathbf { p } _ { F } , t \right] \right. _ { \mathbb { C } } d t ,\tag{42}
$$

$$
\frac { d \chi } { d \mathbf { p } _ { f } } = - 2 \operatorname { R e } \int _ { 0 } ^ { T } \left. \overline { { \mathbf { c } ( T - t ) } } , \mathbf { f } _ { \mathbf { p } _ { f } } ( t ) \right. _ { \mathbb { C } } d t ,\tag{43}
$$

$$
\frac { d \chi } { d \mathbf { p } _ { u _ { 0 } } } = - 2 \operatorname { R e } \left. \overline { { i \mathbf { c } ( T ) } } , \mathbf { u } _ { 0 , \mathbf { p } _ { u _ { 0 } } } \right. _ { \mathbb { C } } .\tag{44}
$$

The suficient conditions under which c can be obtained in the same hardware are as follows:

Theorem 8.1 (Suficient conditions for physical adjoints for Schr¨odinger-type systems). The complex conjugate of the adjoint field can be obtained on the same hardware in the following cases:

• Linear: Reciprocity. Suppose

$$
\mathbf { F } [ \mathbf { u } , \overline { { \mathbf { u } } } , \mathbf { p } _ { F } , t ] = \mathbf { K } ( \mathbf { p } _ { K } , t ) \mathbf { u }
$$

and reciprocity holds, i.e.,

$$
\mathbf { K } ( \mathbf { p } _ { K } , t ) ^ { T } = \mathbf { K } ( \mathbf { p } _ { K } , t )
$$

then the complex conjugate of the adjoint is propagated by the same linear hardware, with the explicit time dependence replayed in reverse and with the adjoint source and initial conditions (41).

• Nonlinear: Existence of a PT-TRM and reciprocity of the parity-pulled-back linearized dynamics. Let Π be a real orthogonal parity involution, $ { \boldsymbol { \Pi } } ^ { - 1 } =  { \boldsymbol { \Pi } } ^ { T } =  { \boldsymbol { \Pi } }$ , and let

$$
\mathbf { w } ( t ) : = \pmb { \Pi } \overline { { \mathbf { u } ( T - t ) } }
$$

be the $\mathcal { P T }$ -reversed trajectory. Suppose the dynamics are $\mathcal { P T }$ -symmetric along the forward trajectory,

$$
\mathbf { F } \big [ \mathbf { I } \overline { { \mathbf { u } ( t ) } } , \mathbf { I } \mathbf { u } ( t ) , \mathbf { p } _ { F } , t \big ] = \mathbf { I } \mathbf { F } \big [ \mathbf { u } ( t ) , \overline { { \mathbf { u } ( t ) } } , \mathbf { p } _ { F } , t \big ] , \qquad t \in [ 0 , T ] ,\tag{45}
$$

so that $\boldsymbol { \Theta _ { \mathcal { P T } } } = \mathbf { I I } \boldsymbol { \mathcal { K } }$ implements a TRM: w is generated on the same hardware by applying $\Theta _ { \mathcal { P T } }$ to the final state $\mathbf { u } ( T )$ , reversing the explicit time dependences, and transforming the external drive as $\mathbf { f } ( t ) \mapsto \Pi \overline { { \mathbf { f } ( T - t ) } }$

Suppose, independently, that the Wirtinger Jacobians satisfy

$$
\Pi \mathbf { F _ { u } } \big [ \mathbf { w } ( T - t ) , \overline { { \mathbf { w } ( T - t ) } } , \mathbf { p } _ { F } , t \big ] \Pi = \mathbf { F _ { u } } \big [ \mathbf { u } ( t ) , \overline { { \mathbf { u } ( t ) } } , \mathbf { p } _ { F } , t \big ] ^ { T } ,\tag{46}
$$

$$
\mathbf { I I F } _ { \overline { { \mathbf { u } } } } \big [ \mathbf { w } ( T - t ) , \overline { { \mathbf { w } ( T - t ) } } , \mathbf { p } _ { F } , t \big ] \mathbf { I I } = \overline { { \mathbf { F } _ { \overline { { \mathbf { u } } } } \big [ \mathbf { u } ( t ) , \overline { { \mathbf { u } ( t ) } } , \mathbf { p } _ { F } , t \big ] } } ^ { T } ,\tag{47}
$$

for all $t \in [ 0 , T ]$ . Then the complex-conjugate adjoint is obtained, up to the parity pull-back Π, as the infinitesimal response of a nudged PT -reversed trajectory.

## 8.1 Linear

The linear case requires symmetry in the reciprocal sense, not Hermiticity: loss and gain are compatible with the physical adjoint as long as reciprocity is preserved. For a complex-linear system, the forward dynamics is

$$
i \dot { \mathbf { u } } ( t ) + \mathbf { K } ( \mathbf { p } _ { K } , t ) \mathbf { u } ( t ) - \mathbf { f } ( t , \mathbf { p } _ { f } ) = \mathbf { 0 } ,\tag{48}
$$

where $\mathbf { F } _ { \overline { { \mathbf { u } } } } = 0$ by complex linearity and $\mathbf { F } _ { \mathbf { u } } = \mathbf { K } ( \mathbf { p } _ { K } , t )$ independent of the forward state. Then (40) reads

$$
i \dot { \mathbf { c } } ( t ) + \mathbf { K } ( \mathbf { p } _ { K } , T - t ) ^ { T } \mathbf { c } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) , \mathbb { P } \overline { { \mathbf { u } } } ( T - t ) \big ] = \mathbf { 0 } .\tag{49}
$$

Hence, if ${ \bf K } ^ { T } = { \bf K }$ for all $t ,$ one can propagate both fields in the same device by swapping the external drive and initial conditions of the forward problem for those of the adjoint and by replaying the explicit time dependence in reverse. □

Free-space difractive optical systems. In Appendix G we show that replacing time t by the propagation coordinate z recovers free-space difractive optical systems.

Holographic vs. intensity measurements. The physical algorithm above requires measuring both ${ \bf \delta u } ( t )$ and $\mathbf { c } ( t )$ holographically (phase and amplitude), so that the gradient overlap integrals (42)–(44) can be evaluated. Extracting the gradient from intensity measurements alone would be experimentally simpler. In [23, 33] this was achieved for Helmholtz systems by exploiting time reversal symmetry in addition to reciprocity. The same construction applies to linear Schr¨odinger problems:

For two complex numbers x and y, $2 \mathrm { R e } ( \overline { { x } } y ) = | x + y | ^ { 2 } - | x | ^ { 2 } - | y | ^ { 2 }$ , so $\operatorname { R e } ( { \overline { { x } } } y )$ follows from three intensity measurements. Eqs. (42)–(44) are instead of the form $\mathrm { R e } ( x y )$ , with x not conjugated, which intensities do not give; one needs b physically in the system rather than $\mathbf { c } ( t ) = \overline { { \mathbf { b } ( T - t ) } }$ This can be achieved if the system is time reversal symmetric and reciprocal: (i) evolve the forward state u from 0 to $T$ and measure the intensity $I _ { \mathbf { u } } ( t ) = | \mathbf { u } ( t ) | ^ { 2 }$ and (ii) physically evolve c from 0 to $T$ and measure the intensity $I _ { \mathbf { c } } ( t ) = | \mathbf { c } ( t ) | ^ { 2 }$ . Then, (iii), apply the TRM of complex conjugating $\mathbf { c } ( T ) \to { \overline { { \mathbf { c } ( T ) } } }$ , time reverse the adjoint source and add the forward sources and initial conditions so that the system now evolves the forward state plus the actual adjoint state ${ \mathbf u } ( t ) + \overline { { { \mathbf c } ( T - t ) } } = { \mathbf u } ( t ) + { \mathbf b } ( t )$ ; and record $I _ { \mathbf { u + b } } ( t ) = | \mathbf { u } ( t ) + \mathbf { b } ( t ) | ^ { 2 }$ . The instantaneous overlap can then be constructed as $I _ { \mathbf { u + b } } ( t ) - I _ { \mathbf { u } } ( t ) - I _ { \mathbf { c } } ( T - t )$ . The construction requires time reversal symmetry, so that complex conjugating $\mathbf { c } ( T )$ re-generates the adjoint state evolution, and linearity, since step (iii) uses the superposition principle.

What it delivers is site-wise, $2 \mathrm { R e } [ \overline { { b _ { i } } } ( t ) u _ { i } ( t ) ]$ , and time reversal symmetry makes K, and with it $\mathbf { K _ { p } } .$ real. Eq. (42) therefore follows when $\mathbf { K _ { p } }$ is diagonal in the measurement basis, as for an on-site potential, but not for a coupling coeficient, which needs the cross-site overlaps $2 \mathrm { R e } [ \overline { { b _ { i } } } u _ { j } ]$ . The source and initial-condition gradients (43)–(44) pair b with a known vector rather than with u and remain holographic.

Literature. For phase-only spatial light modulators (SLMs), the framework reduces to the in situ optical backpropagation rule theorized by Zhou et al. [21]. An early proposal by Wagner and Psaltis implemented the same layerwise adjoint structure in a holographic free-space optical network: reciprocal volume holograms propagated the transpose of each linear interconnection, while a weak counterpropagating probe through nonlinear Fabry-P´erot neurons was designed to approximate multiplication by the local activation derivative [34]. This is a direct small-signal implementation of the nonlinear tangent rather than a contrastive nudging construction; because the probe transmission only approximates the required derivative, the resulting physical gradient is approximate through the nonlinear units. The same reciprocal linearadjoint construction covers the linear optical layers proposed in Guo et al. [35] and experimentally realized in Spall et al. [36]; in both cases, the saturable-absorber response only approximates the nonlinear tangent required by backpropagation, so the construction is exact for the ideal linear layers but approximate through the nonlinearity. In Mididoddi et al. [22] physical realization of the adjoint field was implemented experimentally to optimize the input field in order to suppress output fluctuations in a dynamically varying medium. Finally, we recover the gradient rule in Dal Cin et al. [68] in the linear limit (their nonlinear extension is approximate). However, we note that in the linear case, the adjoint source can be applied at full amplitude and no nudging or diference of trajectories is required.

## 8.2 Nonlinear

Nonlinear Schr¨odinger systems require both a time-reversal mirror (TRM) for the background trajectory and reciprocity of the parity-pulled-back full Wirtinger linearized system. Let

$$
\Theta _ { \mathcal { P T } } = \mathbf { I I } \mathcal { K } ,\tag{50}
$$

where Π is the parity involution introduced in Theorem 8.1 and K denotes complex conjugation. The special case $\mathbf { I I } = \mathbf { I }$ recovers spinless time reversal implemented by complex conjugation alone.

Applying $\Theta \tau$ to the forward state at $t = T$ generates the PT -reversed background

$$
\mathbf { w } ( t ) : = \Theta _ { \mathcal { P T } } \mathbf { u } ( T - t ) = \Pi \overline { { \mathbf { u } ( T - t ) } } .\tag{51}
$$

Evaluating Eq. (39) at $T - t .$ , complex conjugating it, and applying Π gives

$$
\begin{array} { r l } & { i \dot { \mathbf { w } } ( t ) + \mathbf { I I } \overline { { \mathbf { F } \big [ \mathbf { u } ( T - t ) , \overline { { \mathbf { u } ( T - t ) } } , \mathbf { p } _ { F } , T - t \big ] } } } \\ & { ~ - \mathbf { I I } \overline { { \mathbf { f } ( T - t , \mathbf { p } _ { f } ) } } = \mathbf { 0 } , \qquad \mathbf { w } ( 0 ) = \mathbf { I I } \overline { { \mathbf { u } ( T ) } } . } \end{array}\tag{52}
$$

Therefore, if the $\mathcal { P T } .$ -symmetry condition (45) holds, the same hardware reproduces the PT -reversed background according to

$$
\begin{array} { r l r } & { } & { i \dot { \mathbf { w } } ( t ) + \mathbf { F } \big [ \mathbf { w } ( t ) , \overline { { \mathbf { w } ( t ) } } , \mathbf { p } _ { F } , T - t \big ] \qquad } \\ & { } & { - \Pi \overline { { \mathbf { f } ( T - t , \mathbf { p } _ { f } ) } } = \mathbf { 0 } , \qquad \mathbf { w } ( 0 ) = \Pi \overline { { \mathbf { u } ( T ) } } . } \end{array}\tag{53}
$$

Thus the reversed background is generated by applying $\Theta _ { \mathcal { P T } }$ to the final state, reversing all explicit time dependences, and transforming the external drive as

$$
\mathbf { f } ( t ) \mapsto \Pi \overline { { \mathbf { f } ( T - t ) } } .
$$

As in the real nonlinear case, the adjoint is revealed by an infinitesimal nudge. Run a copy $\mathbf { w } ^ { \epsilon }$ on the same hardware with the adjoint source transformed by Π and, when a terminal loss is present, with initia condition

$$
\mathbf { w } ^ { \epsilon } ( 0 ) = \mathbf { w } ( 0 ) + \epsilon i \pmb { \Pi } \mathbb { P } \psi _ { \mathbf { u } } \bigl [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \overline { { \mathbf { u } } } ( T ) \bigr ] .\tag{54}
$$

The diferential field

$$
\delta \mathbf { w } ( t ) : = \mathbf { w } ^ { \epsilon } ( t ) - \mathbf { w } ( t )\tag{55}
$$

then obeys, for $| \epsilon | \ll 1$

$$
\begin{array} { r l } & { i \delta \dot { \mathbf { w } } ( t ) + \mathbf { F _ { u } } \big [ \mathbf { w } ( t ) , \overline { { \mathbf { w } ( t ) } } , \mathbf { p } _ { F } , T - t \big ] \delta \mathbf { w } ( t ) } \\ & { \qquad + \mathbf { F _ { \overline { { u } } } } \big [ \mathbf { w } ( t ) , \overline { { \mathbf { w } ( t ) } } , \mathbf { p } _ { F } , T - t \big ] \overline { { \delta \mathbf { w } ( t ) } } } \\ & { \qquad + \epsilon \mathbf { I I } \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) , \mathbb { P } \overline { { \mathbf { u } ( T - t ) } } \big ] + \mathcal { O } ( \epsilon ^ { 2 } ) = \mathbf { 0 } . } \end{array}\tag{56}
$$

To compare this response with the complex-conjugate adjoint, undo the parity transformation and define

$$
\delta \mathbf { v } ( t ) : = \mathbf { I I } \delta \mathbf { w } ( t ) .
$$

Since Π is real and $\pmb { \Pi } ^ { - 1 } = \pmb { \Pi }$ , multiplying the perturbation equation by Π gives

$$
\begin{array} { r l } & { i \delta \dot { \mathbf { v } } ( t ) + \pmb { \Pi } \mathbf { F _ { u } } \big [ \mathbf { w } ( t ) , \overline { { \mathbf { w } ( t ) } } , \mathbf { p } _ { F } , T - t \big ] \mathbf { H } \delta \mathbf { v } ( t ) } \\ & { \qquad + \mathbf { \Pi } \mathbf { H } \mathbf { F } _ { \overline { { \mathbf { u } } } } \big [ \mathbf { w } ( t ) , \overline { { \mathbf { w } ( t ) } } , \mathbf { p } _ { F } , T - t \big ] \mathbf { H } \overline { { \delta \mathbf { v } ( t ) } } } \\ & { \qquad + \epsilon \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) , \mathbb { P } \overline { { \mathbf { u } ( T - t ) } } \big ] + \mathcal { O } ( \epsilon ^ { 2 } ) = \mathbf { 0 } . } \end{array}\tag{57}
$$

Comparing Eq. (57) term by term with the complex-conjugate adjoint equation (40), the parity-pulledback response $\delta \mathbf { v } / \epsilon$ coincides with c as $\epsilon  0$ , precisely when the independent tangent-adjoint matching conditions $( 4 6 )  { - } ( 4 7 )$ hold.

Thus the two conditions play distinct roles: $\mathcal { P T }$ symmetry supplies the correct reversed background trajectory, while reciprocity of the parity-pulled-back tangent operators ensures that the infinitesimal nudg propagates as the physical adjoint. The complex-conjugate adjoint is therefore

$$
\mathbf { c } ( t ) = \operatorname* { l i m } _ { \epsilon \to 0 } \frac { \delta \mathbf { v } ( t ) } { \epsilon } = \Pi \operatorname* { l i m } _ { \epsilon \to 0 } \frac { \mathbf { w } ^ { \epsilon } ( t ) - \mathbf { w } ( t ) } { \epsilon } ,\tag{58}
$$

and the raw Wirtinger adjoint follows from

$$
\mathbf { b } ( t ) = \overline { { \mathbf { c } ( T - t ) } } .\tag{59}
$$

For $\Pi = \mathbf { I } .$ the parity pull-back is trivial and the construction reduces to conjugation-time reversal with $\begin{array} { r } { \mathbf { c } = \operatorname* { l i m } _ { \epsilon \to 0 } \delta \mathbf { w } / \epsilon } \end{array}$

Example: local Kerr nonlinearity. For a permutation parity Π, consider

$$
\mathbf { F } [ \mathbf { u } , { \overline { { \mathbf { u } } } } ] = \mathbf { K } \mathbf { u } + g | \mathbf { u } | ^ { 2 } \odot \mathbf { u } , \qquad g \in \mathbb { R } .\tag{60}
$$

If

$$
\mathbf { K } = \mathbf { \Pi } \overline { { \mathbf { I } \mathbf { K } } } \mathbf { \Pi } \mathbf { I } \mathbf { I } ,\tag{61}
$$

the nonlinear dynamics are $\mathcal { P T }$ -symmetric. The Wirtinger Jacobians are

$$
{ \bf F _ { u } } = { \bf K } + 2 g \ \mathrm { d i a g } ( | { \bf u } | ^ { 2 } ) , \qquad { \bf F _ { \overline { { { u } } } } } = g \ \mathrm { d i a g } ( { \bf u } ^ { 2 } ) .\tag{62}
$$

The local Kerr contributions satisfy the parity-pulled-back tangent conditions automatically, but the linear part must additionally satisfy

$$
\mathbf { I I K I I } = \mathbf { K } ^ { T } .\tag{63}
$$

Together with Eq. (61) this implies that K needs to be Hermitian. For Π = I, these conditions reduce to a real symmetric K with a real local Kerr coeficient. Thus, a non-trivial Π allows for complex couplings.

Literature. For autonomous (time-independent) Hamiltonian systems with a terminal-value cost, the nudged time-reversed (‘echo’) dynamics of L´opez-Pastor and Marquardt [26] transports the output perturbation backward to yield the loss gradient equivalent to the continuous adjoint field. That work goes a step further and constructs a self-adapting system, in which the learnable parameters are themselves physical degrees of freedom updated by the same dynamics, with no external gradient computation. To the best of the authors’ knowledge, the extension to parity-time symmetry is new.

## 9 Stationary problems

In the preceding sections the computation was carried by a trajectory. We now turn to stationary problems, in which the result is read from a steady state such as the fixed point that a relaxational dynamics settles into, or the steady state of a harmonically driven system. We allow complex amplitudes $\mathbf { u } \in \mathbb { C } ^ { N }$ . The stationary problem considered reads

$$
\mathbf { F } \big [ \mathbf { u } ^ { * } , \overline { { \mathbf { u } ^ { * } } } , \mathbf { p } _ { F } \big ] - \mathbf { f } ( \mathbf { p } _ { f } ) = \mathbf { 0 } ,\tag{64}
$$

where $\mathbf { u } ^ { * }$ denotes the steady state, and the cost function is defined on it as

$$
\chi = \psi \left[ \mathbb { P } \mathbf { u } ^ { * } , \mathbb { P } \overline { { \mathbf { u } ^ { * } } } \right] .\tag{65}
$$

The adjoint field $\mathbf { a } \in \mathbb { C } ^ { N }$ associated with Eq. (64) satisfies the adjoint equation (see Appendix H), in which the two Wirtinger Jacobians are evaluated at the steady state,

$$
\overline { { \mathbf { F } _ { \mathbf { u } } } } ^ { T } \mathbf { a } + \mathbf { F } _ { \mathbf { \overline { { u } } } } ^ { T } \mathbf { \overline { { a } } } + \mathbf { f } _ { \mathrm { a d j } } = \mathbf { 0 } ,\tag{66}
$$

where the adjoint source is given by

$$
\mathbf { f } _ { \mathrm { a d j } } = \mathbb { P } \psi _ { \overline { { \mathbf { u } } } } \big [ \mathbb { P } \mathbf { u } ^ { * } , \mathbb { P } \overline { { \mathbf { u } ^ { * } } } \big ] .\tag{67}
$$

Once a is known, the gradients are a single overlap of the two static fields,

$$
\frac { d \chi } { d \mathbf { p } _ { F } } = 2 \operatorname { R e } \left. \mathbf { a } , \mathbf { F } _ { \mathbf { p } _ { F } } \left[ \mathbf { u } ^ { * } , \overline { { \mathbf { u } ^ { * } } } , \mathbf { p } _ { F } \right] \right. _ { \mathbb { C } } ,\tag{68}
$$

$$
\frac { d \chi } { d { \bf p } _ { f } } = - 2 \operatorname { R e } \left. { \bf a } , { \bf f } _ { { \bf p } _ { f } } ( { \bf p } _ { f } ) \right. _ { \mathbb { C } } .\tag{69}
$$

The suficient conditions mirror the trajectory case, the nonlinear construction now nudging the steady state instead of a time-reversed trajectory:

Theorem 9.1 (Suficient conditions for physical adjoints in stationary problems). The adjoint field, or its complex conjugate, can be obtained on the same hardware in the following cases:

• Linear: Reciprocity. Suppose

$$
{ \displaystyle { \bf F } \big [ { \bf u } , { \bf \overline { { u } } } , { \bf p } _ { F } \big ] = { \bf K } ( { \bf p } _ { \bf K } ) { \bf u } , }
$$

and reciprocity holds, i.e.,

$$
\mathbf { K } ( \mathbf { p } _ { \mathbf { K } } ) ^ { T } = \mathbf { K } ( \mathbf { p } _ { \mathbf { K } } ) ,
$$

then the complex conjugate of the adjoint field, $\mathbf { c } = { \overline { { \mathbf { a } } } }$ , is generated by solving the same stationary problem with the forward source f replaced by the adjoint source − $- \mathbb { P } \psi _ { \mathbf { u } }$ , in a single finite-amplitude experiment and with no nudging.

• Nonlinear: Reciprocity of the tangent operators at the steady state. Suppose the steady state $\mathbf { u } ^ { * }$ is a linearly exponentially stable fixed point of a relaxation realized by the same hardware, and that the full real-linear tangent operator of the stationary equation at $\mathbf { u } ^ { * }$ is invertible. If the Wirtinger tangent operators there satisfy

$$
\begin{array} { r } { \overline { { \mathbf { F } _ { \mathbf { u } } \left[ \mathbf { u } ^ { * } , \mathbf { p } _ { F } \right] } } ^ { T } = \mathbf { F } _ { \mathbf { u } } \left[ \mathbf { u } ^ { * } , \mathbf { p } _ { F } \right] , \qquad \mathbf { F } _ { \mathbf { \overline { { u } } } } \left[ \mathbf { u } ^ { * } , \mathbf { p } _ { F } \right] ^ { T } = \mathbf { F } _ { \mathbf { \overline { { u } } } } \left[ \mathbf { u } ^ { * } , \mathbf { p } _ { F } \right] , } \end{array}\tag{70}
$$

then the adjoint field is $\begin{array} { r } { \mathbf { a } = \left. \frac { d \mathbf { u } ^ { * , \epsilon } } { d \epsilon } \right| _ { \epsilon = 0 } , } \end{array}$ where $\mathbf { u } ^ { * , \epsilon }$ is the nearby steady state under an adjointsource nudge of amplitude ϵ.

## 9.1 Linear

For a linear stationary problem

$$
\mathbf { K ( p _ { K } ) u ^ { * } = f ( p _ { f } ) } ,\tag{71}
$$

with K invertible, the internal force ${ \bf F } [ { \bf u } , \overline { { { \bf u } } } , { \bf p } _ { F } ] = { \bf K } ( { \bf p _ { K } } ) { \bf u }$ is complex-linear, so its Wirtinger Jacobians are constant, $\mathbf { F _ { u } } = \mathbf { K } ( \mathbf { p _ { K } } )$ and $\mathbf { F } _ { \overline { { \mathbf { u } } } } = \mathbf { 0 }$ . An example is a harmonically driven system at frequency $\omega ,$ for which $\mathbf { K } = - \omega ^ { 2 } \mathbf { M } + i \omega \mathbf { D } + \mathbf { K } _ { \mathrm { s t a t } }$ . The adjoint problem then becomes the Hermitian-adjoint system

$$
\overline { { \mathbf { K } ( \mathbf { p } _ { \mathbf { K } } ) ^ { T } } } \mathbf { a } \ = \ - \mathbb { P } \psi _ { \overline { { \mathbf { u } } } } \big [ \mathbb { P } \mathbf { u } ^ { * } , \mathbb { P } \overline { { \mathbf { u } ^ { * } } } \big ] ,\tag{72}
$$

whose conjugate reads

$$
\mathbf { K } ( \mathbf { p } _ { \mathbf { K } } ) ^ { T } \mathbf { c } \ = \ - \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ^ { * } , \mathbb { P } \overline { { \mathbf { u } ^ { * } } } \big ] , \qquad \mathbf { c } = \overline { { \mathbf { a } } } .\tag{73}
$$

This is the same system if $\mathbf { K } ( \mathbf { p } _ { \mathbf { K } } ) ^ { T } = \mathbf { K } ( \mathbf { p } _ { \mathbf { K } } )$ . Hence, the adjoint field can be obtained by swapping the external drive for the adjoint source. □

Comparison with EP. This is the linear, stationary (and complex-valued) counterpart of Equilibrium Propagation (Sec. 9.2). Because the system is linear, no nudging is required, the adjoint source is applied at finite amplitude in a single experiment, and damping or gain are permissible: reciprocity, not Hermiticity, is what is asked of the hardware.

Holographic vs. intensity measurements. For complex amplitudes, holographic measurements are required. Two distinct cases allow the gradients to be reconstructed from intensity measurements.

(i) If K is real-valued (and hence reciprocity implies Hermiticity), injecting $- \mathbb { P } \psi _ { \overline { { \mathbf { u } } } }$ in place of $- \mathbb { P } \psi _ { \mathbf { u } }$ propagates a itself rather than $\mathbf { c } = \overline { { \mathbf { a } } }$ . Three intensity distributions, $I _ { \mathbf { u ^ { * } } } , \ I _ { \mathbf { a } }$ and $I _ { \mathbf { u ^ { * } + a } }$ , the last with the forward drive f and the adjoint source applied simultaneously, then deliver the site-wise overlaps $\begin{array} { r } { 2 \operatorname { R e } [ \overline { { a _ { i } } } u _ { i } ^ { * } ] = I _ { \mathbf { u } ^ { * } + \mathbf { a } , i } - I _ { \mathbf { u } ^ { * } , i } - I _ { \mathbf { a } , i } } \end{array}$ . A real K has a real $\mathbf { K _ { p } } ,$ so these evaluate (68) and (69) whenever $\mathbf { K _ { p } }$ is diagonal in the measurement basis, as for a local permittivity; a non-local $\mathbf { K _ { p } }$ also needs the cross-site overlaps $2 \operatorname { R e } [ \overline { { a _ { i } } } u _ { j } ^ { * } ]$ , which intensity distributions alone do not provide.

(ii) If K is complex-valued and its associated transfer matrix is unitary, the analogous procedure of Sec. 8, paragraph “Holographic vs. intensity measurements,” applies. It was developed in Hughes et al. [33] and realized in Pai et al. [23] in silicon photonic meshes of Mach–Zehnder interferometers and programmable phase shifters. The conjugated adjoint field c is propagated from the outputs to the inputs, measured holographically, and re-encoded as complex-conjugated input sources, so that a further forward pass carries a rather than $\mathbf { c } ;$ superimposing it on the forward field gives $I _ { \mathbf { u } ^ { * } + \mathbf { a } } .$ . Here $\mathbf { K _ { p } }$ may be complex on the diagonal, as for a phase shifter, which also requires Im $[ \overline { { a _ { i } } } u _ { i } ^ { * } ]$ . Re-encoding with an extra factor i carries ia, for which $\mathrm { R e } [ \overline { { i a _ { i } } } u _ { i } ^ { * } ] = \mathrm { I m } [ \overline { { a _ { i } } } u _ { i } ^ { * } ]$ and $I _ { i \mathbf { a } } = I _ { \mathbf { a } } .$ , so a single further distribution $I _ { \mathbf { u } ^ { * } + i \mathbf { a } }$ sufices.

Literature. Physical backpropagation in Helmholtz systems was theorized for feed-forward photonic neural networks by Hughes et al. [33] and experimentally demonstrated by Pai et al. [23] and more recently by Ashtiani et al. [85]. In the optical–electronic–optical (OEO) architectures considered there, the optical output of each layer is detected, transformed by the electronic nonlinearity, and re-encoded for the next layer. During backpropagation, the field returned by each optical layer is multiplied electronically by the derivative of the intervening nonlinearity and re-encoded as the adjoint input to the preceding layer, allowing the single-layer protocol to be chained across layers (see Appendix H.4). For general wavescattering systems, the in situ physical adjoint computing protocol of Guillamon et al. [29] is equivalent to the holographic formulation presented here; a mechanical equivalent was developed by Li and Mao [24,38]. Closely related identities were derived by Wanjura and Marquardt [37] for neuromorphic systems based on linear wave scattering. As shown in Appendix H.5, their scattering-matrix gradient formulas follow from the present adjoint identity.

## 9.2 Nonlinear: Equilibrium Propagation

Let $\mathbf { u } ^ { * \epsilon }$ be the steady state reached with the adjoint source added to the drive,

$$
\mathbf { F } \big [ \mathbf { u } ^ { * \epsilon } , \overline { { \mathbf { u } ^ { * \epsilon } } } , \mathbf { p } _ { F } \big ] - \mathbf { f } ( \mathbf { p } _ { f } ) + \epsilon \mathbb { P } \psi _ { \overline { { \mathbf { u } } } } \big [ \mathbb { P } \mathbf { u } ^ { * \epsilon } , \mathbb { P } \overline { { \mathbf { u } ^ { * \epsilon } } } \big ] = \mathbf { 0 } , \qquad | \epsilon | \ll 1 .\tag{74}
$$

Subtracting the free steady state and linearizing in $\delta \mathbf { u } = \mathbf { u } ^ { * \epsilon } - \mathbf { u } ^ { * }$ gives

$$
\mathbf { F } _ { \mathbf { u } } \delta \mathbf { u } + \mathbf { F } _ { \overline { { \mathbf { u } } } } \overline { { \delta \mathbf { u } } } + \epsilon \mathbb { P } \psi _ { \overline { { \mathbf { u } } } } = \mathcal { O } ( \epsilon ^ { 2 } ) .\tag{75}
$$

Divide by ϵ and let $\epsilon \to 0$ . The rescaled response then obeys the adjoint equation (66) precisely when the tangent operators satisfy condition (70), and by uniqueness

$$
\mathbf { a } = \operatorname* { l i m } _ { \epsilon \to 0 } \frac { \mathbf { u } ^ { * \epsilon } - \mathbf { u } ^ { * } } { \epsilon } .\tag{76}
$$

This proves the nonlinear part of Theorem 9.1.

Physical adjoint algorithm. The device is run twice and the two steady states are subtracted. (i) Free phase: let the system settle under the forward drive alone, record $\mathbf { u } ^ { * }$ , and evaluate the loss derivative from the measured output. (ii) Nudged phase: let it settle again with the adjoint source added to the drive at small amplitude $\epsilon ,$ and record $\mathbf { u } ^ { * \epsilon }$ to extract the adjoint field via $\operatorname { E q . }$ (76). The free and nudged phases must settle onto the same branch: ϵ must be small enough that the nudge neither crosses a basin boundary nor drives the system through a bifurcation.

Complex-valued example. Condition (70) holds identically whenever the force is the complex gradient of a real energy,

$$
\mathbf { F } = E _ { \overline { { \mathbf { u } } } } , \qquad E \in \mathbb { R } ,\tag{77}
$$

the complex counterpart of $\mathbf { F } = E _ { \mathbf { u } }$ : reality of E makes $\mathbf { F } _ { \mathbf { u } } = E _ { \overline { { \mathbf { u } } } \mathbf { u } }$ Hermitian, and commuting partial derivatives make $\mathbf { F } _ { \overline { { \mathbf { u } } } } = E _ { \overline { { \mathbf { u } } } \overline { { \mathbf { u } } } }$ symmetric. A concrete case is a lattice with a Kerr nonlinearity,

$$
\mathbf { F } = \mathbf { K } \mathbf { u } + g | \mathbf { u } | ^ { 2 } \odot \mathbf { u } , \qquad E = \langle \mathbf { u } , \mathbf { K } \mathbf { u } \rangle _ { \mathbb { C } } + \frac { g } { 2 } \sum _ { i } | u _ { i } | ^ { 4 } ,\tag{78}
$$

with K Hermitian and g real, for which $\mathbf { F _ { u } } = \mathbf { K } + 2 g \mathrm { d i a g } ( | \mathbf { u } | ^ { 2 } )$ and $\mathbf { F } _ { \overline { { \mathbf { u } } } } = g \mathrm { d i a g } ( \mathbf { u } ^ { 2 } )$

Comparison with the trajectory case. Section 8 asks instead for reciprocity of the tangent operators, because its time-reversal mirror $\boldsymbol { \Theta _ { \mathcal { P T } } } = \mathbf { I I } \boldsymbol { \mathcal { K } }$ is antiunitary and already carries the conjugation. Here there is no background to reverse, so the condition is weaker: the same Kerr medium needs K Hermitian and $\mathbf { \Pi } \mathbf { \Pi } \mathbf { \ K } \mathbf { \Pi } \mathbf { I } = \mathbf { K } ^ { T }$ there (K real symmetric for $\boldsymbol { \Pi } = \boldsymbol { \Pi } )$ , but Hermiticity alone here.

The real-valued case. For real fields $\mathbf { F } _ { \overline { { \mathbf { u } } } }$ is absent and Hermiticity is symmetry, so condition (70) reduces to $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } } ,$ that is, $\mathbf { F } = E _ { \mathbf { u } }$ for a potential E defined locally (Sec. 6, Remark). Adjoint and gradient are then real,

$$
\mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { a } + \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ^ { * } \big ] = \mathbf { 0 } , \qquad \frac { d \boldsymbol { \chi } } { d \mathbf { p } } = \langle \mathbf { a } , \mathbf { F } _ { \mathbf { p } _ { F } } \big [ \mathbf { u } ^ { * } , \mathbf { p } _ { F } \big ] - \mathbf { f } _ { \mathbf { p } _ { f } } ( \mathbf { p } _ { f } ) \rangle ,\tag{79}
$$

with $\psi _ { \mathbf { u } }$ the ordinary real gradient; the factor 2 Re disappears together with the Wirtinger convention. The construction is then Equilibrium Propagation [30]; Appendix H.3 derives this static adjoint and relates it to the trajectory adjoint of Sec. 7.

Onsager reciprocity. If the force is an Onsager-weighted gradient flow, $\mathbf { F } = \mathbf { V } E _ { \mathbf { u } }$ with a constant symmetric mobility (Sec. 5), the construction holds with the V-weighted nudge $\epsilon \mathbf { V } \mathbb { P } \psi _ { \mathbf { u } }$ and the adjoint recovered as $\begin{array} { r } { \mathbf { a } = \mathbf { V } ^ { - 1 } \operatorname* { l i m } _ { \epsilon \to 0 } ( \mathbf { u } ^ { * \epsilon } - \mathbf { u } ^ { * } ) / \epsilon } \end{array}$ (Appendix D). Positivity of V enters only here: with the Vweighted nudge both the free and the nudged dynamics remain gradient flows in the $\mathbf { V } ^ { - 1 }$ metric, descending E tilted by the drive and, in the nudged phase, by ϵψ. A local minimum is then a stable operating point, as Theorem 9.1 assumes.

Literature. EP was first introduced by Scellier and Bengio [30], theoretically applied to nonlinear resistive networks by Kendall et al. [86] and experimentally implemented on an Ising machine by Laydevant et al. [87]. It was later extended by Laborieux and Zenke [67] to holomorphic networks, where exact gradients can be recovered from finite-amplitude complex oscillations; there it is the nudge amplitude that is complexified and the dynamics must be holomorphic, $\mathbf { F } _ { \overline { { \mathbf { u } } } } = \mathbf { 0 }$ . A broad literature has clarified the relation between EP and recurrent backpropagation [88, 89], reduced finite-nudge bias via symmetric nudging and scaled EP to deep convolutional networks [81], introduced frequency propagation as a related frequency-multiplexed learning rule [90], studied EP in strongly multistable systems [91] and developed approximate extensions beyond reciprocal energy-based dynamics [70, 71]; in particular, Scurria et al. [72] generalize to non-reciprocal systems by modifying the learning dynamics. Lin et al. [25] propose a generalized Equilibrium Propagation that organizes two-phase rules by the perturbative order of the nudge at the free equilibrium, recovering $\mathrm { E P }$ at first order and Coupled Learning at second, and apply it to linear resistor networks. What separates the exact from the approximate case is condition (70): both physical schemes proposed for complex fields, by Dal Cin et al. [68] and by Sajnok and Matuszewski [69], miss the Hermiticity of $\mathbf { F } _ { \mathbf { u } }$ and are accurate only to first order, in the nonlinearity strength and in the dissipation respectively.

## 10 Spatial symmetries and twisted reciprocity

In all linear constructions of the preceding sections, the adjoint field propagated under the transposed forward operator, with the loss derivatives injected as sources and initial conditions at the output DOFs through the projector P. Reciprocity, Euclidean or Onsager (Sec. 5), allowed the same hardware to realize this transposed propagation directly. In this section we show that reciprocity is only the simplest instance of a more general condition: it sufices that the forward operator and its transpose are related by a constant, known, invertible transformation S, in practice a symmetry operation of the hardware. This single condition recovers fully forward mode (FFM) training [32] for reciprocal systems and extends exact physical backpropagation to a class of non-reciprocal systems.

For concreteness we work with linear first-order dynamics, $\dot { \mathbf { u } } ( t ) + \mathbf { K } ( \mathbf { p } _ { K } , t ) \mathbf { u } ( t ) - \mathbf { f } ( t ) = \mathbf { 0 }$ , whose adjoint field obeys (recall Sec. 7)

$$
\dot { \mathbf { a } } ( t ) + \mathbf { K } ( \mathbf { p } _ { K } , T - t ) ^ { T } \mathbf { a } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } , \qquad \mathbf { a } ( 0 ) = - \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T ) \big ] .\tag{80}
$$

Everything carries over verbatim to the second-order (Sec. 6), Schr¨odinger (Sec. 8; the construction then acts on the conjugated adjoint field c) and stationary (Sec. 9) cases.

Theorem 10.1 (Suficient conditions for physical adjoints in linear systems: intertwined reciprocity). Suppose there exists a constant invertible matrix S, independent of the trainable parameters and known to the experimenter, such that

$$
\mathbf { S } \mathbf { K } ( \mathbf { p } _ { K } , t ) ^ { T } \mathbf { S } ^ { - 1 } = \mathbf { K } ( \mathbf { p } _ { K } , t )\tag{81}
$$

holds for all times t and for every parameter value p<sub>K</sub> visited during training; for second-order systems we require in addition $\mathbf { S } \mathbf { M } ^ { T } \mathbf { S } ^ { - 1 } = \mathbf { M }$ and $\mathbf { S } \mathbf { D } ^ { T } \mathbf { S } ^ { - 1 } = \mathbf { D }$ . Reciprocity of K is not required. Then the transformed adjoint field $\mathbf { d } ( t ) = \mathbf { S } \mathbf { a } ( t )$ is propagated by the same hardware in a single finite-amplitude experiment, in which the adjoint source and initial conditions are pre-multiplied by S, and any explicit time dependence is replayed in reverse. The adjoint field is recovered digitally as $\mathbf { a } ( t ) = \mathbf { S } ^ { - 1 } \mathbf { d } ( t )$

To see this, substitute $\mathbf { d } ( t ) = \mathbf { S } \mathbf { a } ( t )$ into Eq. (80) and multiply from the left by S. Using condition (81),

$$
\dot { \mathbf { d } } ( t ) + \mathbf { K } ( T - t ) \mathbf { d } ( t ) + \mathbf { S } \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } , \qquad \mathbf { d } ( 0 ) = - \mathbf { S } \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T ) \big ] .\tag{82}
$$

This is the forward hardware, driven by the transformed adjoint source $\mathbf { S P } \theta _ { \mathbf { u } } .$ , a known linear recombination of the loss derivatives. □

Which S qualify, and at what cost? For a single fixed K an invertible S always exists, since every square matrix is similar to its transpose. The requirement is that one known S serve the entire trainable family ${ \bf K } ( { \bf p } _ { K } , t )$ ; a parameter-dependent $\mathbf { S } ( \mathbf { p } _ { K } )$ in closed form also qualifies, at the price of recomputing it and re-adjusting the source and measurement weights after every update. The remaining cost is that of applying the transformed sources, and it depends on the kind of matrix S is. A signed permutation covers the typical symmetry operations of the hardware (a mirror, a rotation by π, a point inversion, an exchange of two subsystems, a sign flip diag(±1) on a subset of DOFs). Then $\mathbb { P } ^ { \mathbf { S } } = \mathbf { S } \mathbb { P } \mathbf { S } ^ { - 1 }$ again projects onto a set of DOFs, and the adjoint source and initial conditions are injected unchanged at relocated ports (most such operations satisfy $\mathbf { S } ^ { 2 } = \mathbf { I } .$ , so $\mathbf { a } = \mathbf { S } \mathbf { d } )$ . An orthogonal S preserves amplitudes, so no dynamic range is lost, but the sources become coherent combinations across DOFs. A merely invertible S puts weights on the sources and on the recovered adjoint, and its condition number sets the cost in dynamic range.

Who applies the transformation and relation to Onsager reciprocity? For a symmetric positive $\mathbf { S } = \mathbf { V }$ , condition (81) is precisely the V-reciprocity condition (22) of Sec. 5, whose linear statement is thereby a corollary of the theorem. The symmetry and Onsager instances, however, difer in where the transformation resides. A symmetry operation S appears nowhere in the equations of motion: the experimenter implements it by relocating the sources and undoes it in the gradient overlaps. An Onsager reciprocity, by contrast, is part of the dynamics itself. This has two consequences. First, condition (81) holds automatically for the gradient-flow structure K = VH with symmetric H, since $\mathbf { V } \mathbf { K } ^ { T } \mathbf { V } ^ { - 1 } = \mathbf { V } \mathbf { H } \mathbf { V } \mathbf { V } ^ { - 1 } = \mathbf { K }$ . Second, the V-weighting is largely applied by the physics. On the source side, external drives typically couple through the mobility: in the resistor–capacitor network of Sec. 5, the forward dynamics is $\dot { \mathbf { u } } + \mathbf { C } ^ { - 1 } \mathbf { G } \mathbf { u } = \mathbf { C } ^ { - 1 } \mathbf { j }$ , so injecting the raw error currents $\mathbf { j } _ { \mathrm { a d j } } = - \mathbb { P } \theta _ { \mathbf { u } }$ realizes the required source $\mathbf { V } \mathbb { P } \theta _ { \mathbf { u } }$ with ${ \bf V } = { \bf C } ^ { - 1 }$ ; likewise, initializing by depositing the raw error charges $- \mathbb { P } \psi _ { \mathbf { u } }$ realizes $\mathbf { d } ( 0 ) = - \mathbf { V } \mathbb { P } \psi _ { \mathbf { u } }$ . On the recovery side, if the trainable parameters sit in the symmetric part H, the transformation cancels identically in the overlaps,

$$
\langle \mathbf { a } , \mathbf { K } _ { \mathbf { p } } \mathbf { u } \rangle = \mathbf { a } ^ { T } \mathbf { V } \mathbf { H } _ { \mathbf { p } } \mathbf { u } = ( \mathbf { V } \mathbf { a } ) ^ { T } \mathbf { H } _ { \mathbf { p } } \mathbf { u } = \mathbf { d } ^ { T } \mathbf { H } _ { \mathbf { p } } \mathbf { u } ,\tag{83}
$$

using only $\mathbf { V } = \mathbf { V } ^ { T }$ . For the resistor network this gives, for the conductance $g _ { e }$ of the edge connecting nodes m and $n _ { \mathrm { : } }$

$$
\frac { d \chi } { d g _ { e } } = \int _ { 0 } ^ { T } \left( d _ { m } ( T - t ) - d _ { n } ( T - t ) \right) \left( u _ { m } ( t ) - u _ { n } ( t ) \right) d t .\tag{84}
$$

Hence, the measured voltages of the forward and adjoint runs pair directly, with no capacitance weights anywhere. The two cases also difer in what must be known. In the symmetry case the experimenter must know S explicitly and applies it twice: once physically, by injecting the adjoint source and initial conditions at the relocated ports, and once digitally, by relabeling indices in the overlaps to extract $\mathbf { a } = \mathbf { S } ^ { - 1 } \mathbf { d } ;$ both steps are pure routing. In the Onsager case the numerical values of V need never be known: it sufices to know, structurally, that the dynamics is a gradient flow with a constant, parameter-independent mobility and that the trainable parameters sit in H. Sources and initial conditions are specified in the natural current and charge variables, where the hardware applies V, and V cancels not only in the overlaps (83) but also in gradients with respect to drive parameters, $\begin{array} { r } { d \chi / d { \bf p } _ { j } = - \int _ { 0 } ^ { T } \mathbf { d } ( T - t ) ^ { T } \mathbf { j } _ { \bf p } _ { j } } \end{array}$ dt for $\mathbf f = \mathbf V \mathbf j$ , and, if the initial state is set by deposited charges $\mathbf { q } _ { 0 } = \mathbf { C } \mathbf { u } _ { 0 }$ , in the initial-condition gradient $d \chi / d \mathbf { q } _ { 0 } = - \mathbf { d } ( T )$

Reciprocal systems and fully forward mode training. For reciprocal hardware, ${ \bf K } ^ { T } = { \bf K } .$ the identity S = I satisfies condition (81), and Theorem 10.1 reduces to the linear cases of Theorems 6.1–9.1, with the adjoint sources injected at the output DOFs. Suppose the hardware additionally possesses a spatial symmetry, $\mathbf { S K } ( t ) \mathbf { S } ^ { - 1 } = \mathbf { K } ( t )$ , that maps the output onto the input DOFs, SPS $^ { - 1 } = { \mathbb { P } } ^ { \mathrm { i n } }$ . Then S satisfies condition (81) as well, and the adjoint experiment may instead be driven at the inputs: the adjoint source and initial conditions enter the device where the data enters and propagates forward. This is fully forward mode training [32]. Reciprocity does the existence work and the symmetry performs the relocation of the injection ports to the experimentally convenient side of the device. The price is that parameter updates must preserve the symmetry, so symmetry-related parameters are tied and trained jointly. The original derivation of Ref. [32] is phrased in terms of the Green’s functions of static, Helmholtz-type propagation (cf. Sec. 9.1), with Lorentz reciprocity entering as the symmetry of the Green’s function. The formulation above extends it from the static setting to full time-dependent trajectory learning.

## 10.1 Non-reciprocal systems: twisted reciprocity

If there exists an orthogonal transformation, S, such that condition (81) holds for a non-reciprocal K we call the system twisted reciprocal. Decomposing $\mathbf { K } = \mathbf { K } _ { s } + \mathbf { K } _ { a }$ into its symmetric and antisymmetric parts (conjugation by an orthogonal S preserves both), condition (81) holds if and only if

$$
{ \bf S } { \bf K } _ { s } { \bf S } ^ { - 1 } = { \bf K } _ { s } , { \bf S } { \bf K } _ { a } { \bf S } ^ { - 1 } = - { \bf K } _ { a } ,\tag{85}
$$

a reciprocal backbone that is even under the symmetry, carrying a non-reciprocal part that is odd. Physically, the non-reciprocity must therefore originate from a pseudoscalar or time-odd quantity mounted on a symmetric structure: a uniform hopping bias [39, 92, 93], a rotation sense [76, 94, 95], or an in-plane magnetization, with the symmetry supplying the field reversal required by the Onsager–Casimir relation $\mathbf { K } ( \mathbf { B } ) ^ { T } = \mathbf { K } ( - \mathbf { B } )$ [96]. The $\operatorname { s y }$ mmetry certifies that the violation of reciprocity is exactly compensated between symmetry-transformed pairs of DOFs, while the operator itself stays non-reciprocal, ${ \bf K } ^ { T } \ne { \bf K }$ Condition (81) is the transpose-type symmetry of the non-Hermitian symmetry classification [97], here repurposed as a trainability condition.

![](images/91702f2358c0c3ffad14c834066edc9f0874a0afdfef1c56e8f4d6df8924ea93.jpg)  
Figure 3: The Hatano–Nelson chain of Eq. (86) for $N \ = \ 4 { : }$ sites with on-site potentials $\varepsilon _ { n } .$ , coupled by asymmetric nearest-neighbor hoppings, amplified to the right $( J e ^ { + h }$ , blue) and attenuated to the left $( J e ^ { - h }$ , red), so that ${ \bf K } ^ { T } \ne { \bf K }$ and the chain transmits with one-way gain. Transposition reverses the arrow of every hop; the spatial inversion $n  N + 1 - n$ about the marked inversion center reverses it again. If the on-site potential is mirror-symmetric, $\varepsilon _ { n } = \varepsilon _ { N + 1 - n } ,$ the chain is therefore twisted reciprocal, $\mathbf { S } \mathbf { K } ^ { T } \mathbf { S } ^ { - 1 } = \mathbf { K }$ , and by Theorem 10.1 the adjoint field is obtained in the same hardware, with the same bias h, by injecting the spatially flipped adjoint source and initial conditions.

Example: a Hatano–Nelson chain. Consider N sites on an open chain with asymmetric nearestneighbor hoppings and a real on-site potential [39],

$$
K _ { n + 1 , n } = J e ^ { + h } , \qquad K _ { n , n + 1 } = J e ^ { - h } , \qquad K _ { n n } = \varepsilon _ { n } , \qquad J , h , \varepsilon _ { n } \in \mathbb { R } .\tag{86}
$$

For $N = 4$ , the operator and the spatial inversion of the chain, $S _ { m n } = \delta _ { m , N + 1 - n } ,$ read

$$
\mathbf { K } = \left[ \begin{array} { c c c c } { \varepsilon _ { 1 } } & { J e ^ { - h } } & { 0 } & { 0 } \\ { J e ^ { + h } } & { \varepsilon _ { 2 } } & { J e ^ { - h } } & { 0 } \\ { 0 } & { J e ^ { + h } } & { \varepsilon _ { 3 } } & { J e ^ { - h } } \\ { 0 } & { 0 } & { J e ^ { + h } } & { \varepsilon _ { 4 } } \end{array} \right] , \qquad \mathbf { S } = \left[ \begin{array} { c c c c } { 0 } & { 0 } & { 0 } & { 1 } \\ { 0 } & { 0 } & { 1 } & { 0 } \\ { 0 } & { 1 } & { 0 } & { 0 } \\ { 1 } & { 0 } & { 0 } & { 0 } \end{array} \right] .\tag{87}
$$

The dynamics may be taken Schr¨odinger-type, iu˙ $+ \mathbf { K } \mathbf { u } - \mathbf { f } = \mathbf { 0 }$ , or first-order dissipative, u˙ $+ \mathbf { K } \mathbf { u } - \mathbf { f } = \mathbf { 0 }$ For $h \neq 0$ rightward hops are amplified and leftward hops attenuated. Hence, ${ \bf K } ^ { T } \ne { \bf K }$ , and the chain transmits with exponential one-way gain. This class is the linearized description of the standard platforms of the non-Hermitian skin efect [97], including feedback-biased active metamaterials [98,99], non-reciprocal topolectrical circuits [92], and non-reciprocal photonic and acoustic lattices [93, 94]. The adjoint field (80) must propagate under ${ \mathbf K } ^ { T }$ , the attenuating direction, which the hardware does not provide.

The spatial inversion resolves this: transposition reverses the arrow of every hop and the inversion reverses it again, so condition (81) holds, provided the on-site potential is mirror-symmetric, $\varepsilon _ { n } = \varepsilon _ { N + 1 - n } .$ By Theorem 10.1 the adjoint is then obtained from a single finite-amplitude run of the same chain, with the same bias h. If the loss is measured at the right edge, the spatially flipped adjoint source and initial conditions are injected at the left edge, and the gradient overlaps are evaluated with $\mathbf { a } ( t ) = \mathbf { S } \mathbf { d } ( t )$

Heterogeneous couplings are admissible provided their profiles are mirror-even, $J _ { n } = J _ { N - n } , h _ { n } = h _ { N - n } ,$ $\varepsilon _ { n } = \varepsilon _ { N + 1 - n }$ . Upon optimizing the chain, the symmetry must be preserved, halving the number of design parameters.

## 11 Conclusions

We have unified a broad range of formally exact physical gradient methods [20–24, 26, 29–33, 37, 38] using the adjoint method from PDE-constrained optimization. We have clarified that linear and nonlinear systems behave fundamentally diferently and are governed by diferent suficient conditions for obtaining the adjoint field on the same hardware. For linear systems reciprocity is suficient and the adjoint follows from a single finite-amplitude experiment, whereas nonlinear trajectory systems additionally require a time-reversal mirror, reciprocity of the linearized dynamics, and infinitesimal nudging. For linear systems we have further shown that reciprocity is only the simplest instance of a more general condition: it sufices that the forward operator and its transpose are related by a constant, known, invertible transformation, in practice a symmetry operation of the hardware. This intertwined reciprocity yields exact physical gradients in a class of non-Hermitian, non-reciprocal systems, exemplified by the Hatano–Nelson chain [39]. We have also introduced several generalizations, covering non-autonomous, Onsager-reciprocal, and PT-symmetric dynamics.

Because on-device gradients are far easier to obtain in linear than in nonlinear hardware, encoding the inputs in the parameters of a linear system, a scheme known as structural nonlinearity [53–55], is a particularly attractive route to physical computing: the device computes nonlinearly in its inputs, yet its adjoint field, and with it the gradient, remains accessible at finite amplitude and with damping present [37]. The non-autonomous generalization makes the temporal modulation of the internal forces itself a trainable resource and may thereby enable novel ways of computing with physical systems.

As discussed in Sec. 4, when the parameters enter the forward operators linearly, the gradient overlaps are assembled entirely from measured trajectories and never reference the parameter values, so slow drift, aging, and calibration ofsets are absorbed automatically. What must still be known is the structure of the model: which measured signals to pair, and where the trainable parameters sit in the equations of motion. How accurately that structure must be known, and how the gradients degrade when it is wrong, remains open.

Acknowledgments We thank Dr. Sam Dillavou, Dr. Fatih Dinc and Andy Davis for helpful discussions and feedback on the manuscript. C.B. was supported by the Swiss National Science Foundation (SNSF) through a Postdoc.Mobility fellowship (P500PT 217673/1). YG and HT were sponsored by AFOSR MURI Grant No. FA955022-1-0203 and by the Army Research Ofice under Grant No. W911NF-25-1-0261. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the oficial policies, either expressed or implied, of the Army Research Ofice or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein. This work was performed in part at Aspen Center for Physics, which is supported by National Science Foundation grant PHY-2210452.

## References

[1] Herbert Jaeger, Beatriz Noheda, and Wilfred G Van Der Wiel. Toward a formal theory for computing machines made out of whatever physics ofers. Nature Communications, 14(1):4911, 2023.

[2] Tyler W Hughes, Ian AD Williamson, Momchil Minkov, and Shanhui Fan. Wave physics as an analog recurrent neural network. Science Advances, 5(12):eaay6946, 2019.

[3] Farzad Zangeneh-Nejad, Dimitrios L Sounas, Andrea Al\`u, and Romain Fleury. Analogue computing with metamaterials. Nature Reviews Materials, 6(3):207–225, 2021.

[4] Tena Dubˇcek, Daniel Moreno-Garcia, Thomas Haag, Parisa Omidvar, Henrik R Thomsen, Theodor S Becker, Lars Gebraad, Christoph B¨arlocher, Fredrik Andersson, Sebastian D Huber, et al. Insensor passive speech classification with phononic metamaterials. Advanced Functional Materials, 34(17):2311877, 2024.

[5] Sam Dillavou, Menachem Stern, Andrea J Liu, and Douglas J Durian. Demonstration of decentralized physics-driven learning. Physical Review Applied, 18(1):014040, 2022.

[6] Abu Sebastian, Manuel Le Gallo, Riduan Khaddam-Aljameh, and Evangelos Eleftheriou. Memory devices and applications for in-memory computing. Nature Nanotechnology, 15(7):529–544, 2020.

[7] Danijela Markovi´c, Alice Mizrahi, Damien Querlioz, and Julie Grollier. Physics for neuromorphic computing. Nature Reviews Physics, 2(9):499–510, 2020.

[8] Bhavin J Shastri, Alexander N Tait, Thomas Ferreira de Lima, Wolfram HP Pernice, Harish Bhaskaran, C David Wright, and Paul R Prucnal. Photonics for artificial intelligence and neuromorphic computing. Nature Photonics, 15(2):102–114, 2021.

[9] U˘gur Te˘gin, Mustafa Yıldırım, <sup>˙</sup>Ilker O˘guz, Christophe Moser, and Demetri Psaltis. Scalable optical learning operator. Nature Computational Science, 1(8):542–549, 2021.

[10] Gordon Wetzstein, Aydogan Ozcan, Sylvain Gigan, Shanhui Fan, Dirk Englund, Marin Soljaˇci´c, Cornelia Denz, David AB Miller, and Demetri Psaltis. Inference in artificial intelligence with deep optics and photonics. Nature, 588(7836):39–47, 2020.

[11] Xing Lin, Yair Rivenson, Nezih T Yardimci, Muhammed Veli, Yi Luo, Mona Jarrahi, and Aydogan Ozcan. All-optical machine learning using difractive deep neural networks. Science, 361(6406):1004– 1008, 2018.

[12] Peter L McMahon. The physics of optical computing. Nature Reviews Physics, 5(12):717–734, 2023.

[13] Tatsuhiro Onodera, Martin M Stein, Benjamin A Ash, Mandar M Sohoni, Melissa Bosch, Ryotatsu Yanagimoto, Marc Jankowski, Timothy P McKenna, Tianyu Wang, Gennady Shvets, Maxim R Shcherbakov, Logan G Wright, and Peter L McMahon. Arbitrary control over multimode wave propagation for machine learning. Nature Physics, 22:164–171, 2026.

[14] Ryotatsu Yanagimoto, Benjamin A Ash, Mandar M Sohoni, Martin M Stein, Yiqi Zhao, Federico Presutti, Marc Jankowski, Logan G Wright, Tatsuhiro Onodera, and Peter L McMahon. Programmable on-chip nonlinear photonics. Nature, 649(8096):330–337, 2026.

[15] Gouhei Tanaka, Toshiyuki Yamane, Jean Benoit H´eroux, Ryosho Nakane, Naoki Kanazawa, Seiji Takeda, Hidetoshi Numata, Daiju Nakano, and Akira Hirose. Recent advances in physical reservoir computing: A review. Neural Networks, 115:100–123, 2019.

[16] Ali Momeni, Babak Rahmani, Benjamin Scellier, Logan G Wright, Peter L McMahon, Clara C Wanjura, Yuhang Li, Anas Skalli, Natalia G Berlof, Tatsuhiro Onodera, et al. Training of physica neural networks. Nature, 645(8079):53–61, 2025.

[17] Vahid Nikkhah, Ali Pirmoradi, Farshid Ashtiani, Brian Edwards, Firooz Aflatouni, and Nader Engheta. Inverse-designed low-index-contrast structures on a silicon photonics platform for vector– matrix multiplication. Nature Photonics, 18(5):501–508, 2024.

[18] Logan G Wright, Tatsuhiro Onodera, Martin M Stein, Tianyu Wang, Darren T Schachter, Zoey Hu, and Peter L McMahon. Deep physical neural networks trained with backpropagation. Nature, 601(7894):549–555, 2022.

[19] Ilker Oguz, Niyazi Ulas Dinc, Mustafa Yildirim, Junjie Ke, Innfarn Yoo, Qifei Wang, Feng Yang, Christophe Moser, and Demetri Psaltis. Optical difusion models for image generation. Advances in Neural Information Processing Systems, 37:59150–59173, 2024.

[20] Michiel Hermans, Micha¨el Burm, Thomas Van Vaerenbergh, Joni Dambre, and Peter Bienstman. Trainable hardware for dynamical computing using error backpropagation through physical media. Nature Communications, 6(1):6729, 2015.

[21] Tiankuang Zhou, Lu Fang, Tao Yan, Jiamin Wu, Yipeng Li, Jingtao Fan, Huaqiang Wu, Xing Lin, and Qionghai Dai. In situ optical backpropagation training of difractive optical neural networks. Photonics Research, 8(6):940–953, 2020.

[22] Chaitanya K Mididoddi, Robert J Kilpatrick, Christina Sharp, Philipp Del Hougne, Simon AR Horsley, and David B Phillips. Threading light through dynamic complex media. Nature Photonics, 19(4):434–440, 2025.

[23] Sunil Pai, Zhanghao Sun, Tyler W Hughes, Taewon Park, Ben Bartlett, Ian AD Williamson, Momchil Minkov, Maziyar Milanizadeh, Nathnael Abebe, Francesco Morichetti, et al. Experimentally realized in situ backpropagation for deep learning in photonic neural networks. Science, 380(6643):398–404, 2023.

[24] Shuaifeng Li and Xiaoming Mao. Training all-mechanical neural networks for task learning through in situ backpropagation. Nature Communications, 15(1):10528, 2024.

[25] Jonathan Lin, Aman Desai, Frank Barrows, and Francesco Caravelli. How to train your resistive network: Generalized equilibrium propagation and analytical learning. arXiv preprint arXiv:2602.03546, 2026.

[26] V´ıctor L´opez-Pastor and Florian Marquardt. Self-learning machines based on Hamiltonian echo backpropagation. Physical Review X, 13(3):031020, 2023.

[27] Lev Semenovich Pontryagin. Mathematical theory of optimal processes. Routledge, 2018.

[28] Andreas Fichtner. Full seismic waveform modelling and inversion. Springer Science & Business Media, 2010.

[29] John Guillamon, Cheng-Zhen Wang, Zin Lin, and Tsampikos Kottos. In-situ physical adjoint computing in multiple-scattering electromagnetic environments for wave control. Nature Communications, 16(1):11466, 2025.

[30] Benjamin Scellier and Yoshua Bengio. Equilibrium propagation: Bridging the gap between energybased models and backpropagation. Frontiers in Computational Neuroscience, 11:24, 2017.

[31] Guillaume Pourcel and Maxence Ernoult. Learning long range dependencies through time reversal symmetry breaking. arXiv preprint arXiv:2506.05259, 2025.

[32] Zhiwei Xue, Tiankuang Zhou, Zhihao Xu, Shaoliang Yu, Qionghai Dai, and Lu Fang. Fully forward mode training for optical neural networks. Nature, 632(8024):280–286, 2024.

[33] Tyler W Hughes, Momchil Minkov, Yu Shi, and Shanhui Fan. Training of photonic neural networks through in situ backpropagation and gradient measurement. Optica, 5(7):864–871, 2018.

[34] Kelvin Wagner and Demetri Psaltis. Multilayer optical learning networks. Applied Optics, 26(23):5061–5076, 1987.

[35] Xianxin Guo, Thomas D Barrett, Zhiming M Wang, and Alexander I Lvovsky. Backpropagation through nonlinear units for the all-optical training of neural networks. Photonics Research, 9(3):B71– B80, 2021.

[36] James Spall, Xianxin Guo, and Alexander I Lvovsky. Training neural networks with end-to-end optical backpropagation. Advanced Photonics, 7(1):016004, 2025.

[37] Clara C Wanjura and Florian Marquardt. Fully nonlinear neuromorphic computing with linear wave scattering. Nature Physics, 20(9):1434–1440, 2024.

[38] Shuaifeng Li and Xiaoming Mao. Topological mechanical neural networks as classifiers through in situ backpropagation learning. arXiv preprint arXiv:2503.07796, 2025.

[39] Naomichi Hatano and David R Nelson. Localization transitions in non-Hermitian quantum mechanics. Physical Review Letters, 77(3):570–573, 1996.

[40] Menachem Stern, Daniel Hexner, Jason W Rocks, and Andrea J Liu. Supervised learning in physical networks: From machine learning to learning machines. Physical Review X, 11(2):021045, 2021.

[41] Menachem Stern and Arvind Murugan. Learning without neurons in physical systems. Annual Review of Condensed Matter Physics, 14(1):417–441, 2023.

[42] Sam Dillavou, Benjamin D Beyer, Menachem Stern, Andrea J Liu, Marc Z Miskin, and Douglas J Durian. Machine learning without a processor: Emergent learning in a nonlinear analog network. Proceedings of the National Academy of Sciences, 121(28):e2319718121, 2024.

[43] Menachem Stern, Sam Dillavou, Dinesh Jayaraman, Douglas J Durian, and Andrea J Liu. Training self-learning circuits for power-eficient solutions. APL Machine Learning, 2(1):016114, 2024.

[44] Joshua A McGinnis, Xinbo Li, and Yoichiro Mori. Coercivity and local convergence of physical learning in linear circuits. arXiv preprint arXiv:2606.15443, 2026.

[45] Junwei Cheng, Chaoran Huang, Jialong Zhang, Bo Wu, Wenkai Zhang, Xinyu Liu, Jiahui Zhang, Yiyi Tang, Hailong Zhou, Qiming Zhang, Min Gu, Jianji Dong, and Xinliang Zhang. Multimodal deep learning using on-chip difractive optics with in situ training capability. Nature Communications, 15(1):6189, 2024.

[46] Daan de Bos and Marc Serra-Garcia. Multimodal oscillator networks learn to solve a classification problem. npj Metamaterials, 2(1):3, 2026.

[47] Ilker Oguz, Junjie Ke, Qifei Weng, Feng Yang, Mustafa Yildirim, Niyazi Ulas Dinc, Jih-Liang Hsieh, Christophe Moser, and Demetri Psaltis. Forward–forward training of an optical neural network. Optics Letters, 48(20):5249–5252, 2023.

[48] Ali Momeni, Babak Rahmani, Matthieu Mall´ejac, Philipp Del Hougne, and Romain Fleury. Backpropagation-free training of deep physical neural networks. Science, 382(6676):1297–1303, 2023.

[49] Hao Wang, Ziao Wang, Xiangpeng Liang, Han Zhao, Jianqi Hu, Junjie Jiang, Xing Fu, Jianshi Tang, Huaqiang Wu, Sylvain Gigan, et al. Training deep physical neural networks with local physica information bottleneck. arXiv preprint arXiv:2602.09569, 2026.

[50] Maxence Ernoult, Julie Grollier, and Damien Querlioz. Using memristors for robust local learning of hardware restricted Boltzmann machines. Scientific Reports, 9(1):1851, 2019.

[51] Marcelo Guzman, Simone Ciarella, and Andrea J Liu. Unsupervised and probabilistic learning with contrastive local learning networks: The restricted Kirchhof machine. Proceedings of the National Academy of Sciences, 123(21):e2525792123, 2026.

[52] Cyrill B¨osch, Geofrey Roeder, Marc Serra-Garcia, and Ryan P Adams. Local learning rules for out-of-equilibrium physical generative models. arXiv preprint arXiv:2506.19136, 2025.

[53] Nick Richardson, Cyrill B¨osch, and Ryan P Adams. Nonlinear computation with linear optics via source-position encoding. arXiv preprint arXiv:2504.20401, 2025.

[54] Fei Xia, Kyungduk Kim, Yaniv Eliezer, SeungYun Han, Liam Shaughnessy, Sylvain Gigan, and Hui Cao. Nonlinear optical encoding enabled by recurrent linear scattering. Nature Photonics, 18(10):1067–1075, 2024.

[55] Mustafa Yildirim, Niyazi Ulas Dinc, Ilker Oguz, Demetri Psaltis, and Christophe Moser. Nonlinear processing with linear optics. Nature Photonics, 18(10):1076–1082, 2024.

[56] Michael B. Giles and Niles A. Pierce. An introduction to the adjoint approach to design. Flow, Turbulence and Combustion, 65(3-4):393–415, December 2000.

[57] R.-E. Plessix. A review of the adjoint-state method for computing the gradient of a functional with geophysical applications. Geophysical Journal International, 167(2):495–503, November 2006.

[58] Georgios Veronis, Robert W. Dutton, and Shanhui Fan. Method for sensitivity analysis of photonic crystal devices. Optics Letters, 29(19):2288–2290, Oct 2004.

[59] Jan Werschnik and E. K. U. Gross. Quantum optimal control theory. Journal of Physics B: Atomic, Molecular and Optical Physics, 40(18):R175–R211, 2007.

[60] Mar´ıa Soledad Aronna, Joseph Fr´ed´eric Bonnans, and Axel Kr¨oner. Optimal control of PDEs in a complex space setting: Application to the Schr¨odinger equation. SIAM Journal on Control and Optimization, 57(2):1390–1412, 2019.

[61] D. H. Brandwood. A complex gradient operator and its application in adaptive array theory. IEE Proceedings F - Communications, Radar and Signal Processing, 130(1):11–16, 1983.

[62] Are Hjørungnes and David Gesbert. Complex-valued matrix diferentiation: Techniques and key results. IEEE Transactions on Signal Processing, 55(6):2740–2746, 2007.

[63] Laurent Sorber, Marc Van Barel, and Lieven De Lathauwer. Unconstrained optimization of real functions in complex variables. SIAM Journal on Optimization, 22(3):879–898, 2012.

[64] Michael Hinze, Ren´e Pinnau, Michael Ulbrich, and Stefan Ulbrich. Optimization with PDE Constraints. Springer Netherlands, 2009.

[65] Marc Berneman and Daniel Hexner. Training overdamped dynamics. arXiv preprint arXiv:2602.19122, 2026.

[66] Marc Berneman and Daniel Hexner. Equilibrium propagation for dissipative dynamics. arXiv preprint arXiv:2506.20402, 2025.

[67] Axel Laborieux and Friedemann Zenke. Holomorphic equilibrium propagation computes exact gradients through finite size oscillations. Advances in Neural Information Processing Systems, 35:12950– 12963, 2022.

[68] Nicola Dal Cin, Florian Marquardt, and Clara C Wanjura. Training nonlinear optical neural networks with scattering backpropagation. arXiv preprint arXiv:2508.11750, 2025.

[69] Karol Sajnok and Micha l Matuszewski. Near-equilibrium propagation training in nonlinear wave systems. arXiv preprint arXiv:2510.16084, 2025.

[70] Benjamin Scellier, Anirudh Goyal, Jonathan Binas, Thomas Mesnard, and Yoshua Bengio. Generalization of equilibrium propagation to vector field dynamics. arXiv preprint arXiv:1808.04873, 2018.

[71] Axel Laborieux and Friedemann Zenke. Improving equilibrium propagation without weight symmetry through Jacobian homeostasis. In International Conference on Learning Representations, 2024.

[72] Antonino Emanuele Scurria, Dimitri Vanden Abeele, Bortolo Matteo Mognetti, and Serge Massar. Equilibrium propagation for non-conservative systems. arXiv preprint arXiv:2602.03670, 2026.

[73] Benjamin Scellier, Siddhartha Mishra, Yoshua Bengio, and Yann Ollivier. Agnostic physics-driven deep learning. arXiv preprint arXiv:2205.15021, 2022.

[74] Lars Onsager. Reciprocal relations in irreversible processes. I. Physical Review, 37(4):405–426, 1931.

[75] Lars Onsager and Stefan Machlup. Fluctuations and irreversible processes. Physical Review, 91(6):1505–1512, 1953.

[76] Lisa M Nash, Dustin Kleckner, Alismari Read, Vincenzo Vitelli, Ari M Turner, and William T M Irvine. Topological mechanics of gyroscopic metamaterials. Proceedings of the National Academy of Sciences, 112(47):14495–14500, 2015.

[77] Carl M Bender, Bjorn K Berntson, David Parker, and E Samuel. Observation of PT phase transition in a simple mechanical system. American Journal of Physics, 81(3):173–179, 2013.

[78] Mathias Fink. Time reversal of ultrasonic fields. I. Basic principles. IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 39(5):555–566, 1992.

[79] Guillaume Pourcel, Debabrota Basu, Maxence Ernoult, and Aditya Gilra. Lagrangian-based equilibrium propagation: generalisation to arbitrary boundary conditions & equivalence with Hamiltonian echo learning. arXiv preprint arXiv:2506.06248, 2025.

[80] Michael Spivak. Calculus on Manifolds: A Modern Approach to Classical Theorems of Advanced Calculus. W. A. Benjamin, 1965.

[81] Axel Laborieux, Maxence Ernoult, Benjamin Scellier, Yoshua Bengio, Julie Grollier, and Damien Querlioz. Scaling equilibrium propagation to deep convnets by drastically reducing its gradient estimator bias. Frontiers in Neuroscience, 15:633674, 2021.

[82] Serge Massar. Equilibrium propagation for learning in Lagrangian dynamical systems. Physical Review E, 112(3):035304, 2025.

[83] Reinhold Remmert. Theory of complex functions, volume 122. Springer Science & Business Media, 1991.

[84] Kelvin Koor, Yixian Qiu, Leong Chuan Kwek, and Patrick Rebentrost. A short tutorial on Wirtinger calculus with applications in quantum information. arXiv preprint arXiv:2312.04858, 2023.

[85] Farshid Ashtiani, Mohamad Hossein Idjadi, and Kwangwoong Kim. Integrated photonic neural network with on-chip backpropagation training. Nature, 651(8107):927–932, 2026.

[86] Jack Kendall, Ross Pantone, Kalpana Manickavasagam, Yoshua Bengio, and Benjamin Scellier. Training end-to-end analog neural networks with equilibrium propagation. arXiv preprint arXiv:2006.01981, 2020.

[87] J´er´emie Laydevant, Danijela Markovi´c, and Julie Grollier. Training an Ising machine with equilibrium propagation. Nature Communications, 15(1):3671, 2024.

[88] Benjamin Scellier and Yoshua Bengio. Equivalence of equilibrium propagation and recurrent backpropagation. Neural Computation, 31(2):312–329, 2019.

[89] Maxence Ernoult, Julie Grollier, Damien Querlioz, Yoshua Bengio, and Benjamin Scellier. Updates of equilibrium prop match gradients of backprop through time in an RNN with static input. Advances in Neural Information Processing Systems, 32, 2019.

[90] Vidyesh Rao Anisetti, Ananth Kandala, Benjamin Scellier, and JM Schwarz. Frequency propagation: Multimechanism learning in nonlinear physical networks. Neural Computation, 36(4):596–620, 2024.

[91] Qingshan Wang, Clara C Wanjura, and Florian Marquardt. Training coupled phase oscillators as a neuromorphic platform using equilibrium propagation. Neuromorphic Computing and Engineering, 4(3):034014, 2024.

[92] Tobias Helbig, Tobias Hofmann, Stefan Imhof, Mohamed Abdelghany, Tobias Kiessling, Laurens W Molenkamp, Ching Hua Lee, Alexander Szameit, Martin Greiter, and Ronny Thomale. Generalized bulk–boundary correspondence in non-Hermitian topolectrical circuits. Nature Physics, 16(7):747– 750, 2020.

[93] Sebastian Weidemann, Mark Kremer, Tobias Helbig, Tobias Hofmann, Alexander Stegmaier, Martin Greiter, Ronny Thomale, and Alexander Szameit. Topological funneling of light. Science, 368(6488):311–314, 2020.

[94] Romain Fleury, Dimitrios L Sounas, Caleb F Sieck, Michael R Haberman, and Andrea Al\`u. Sound isolation and giant linear nonreciprocity in a compact acoustic circulator. Science, 343(6170):516–519, 2014.

[95] Colin Scheibner, Anton Souslov, Debarghya Banerjee, Piotr Sur´owka, William T M Irvine, and Vincenzo Vitelli. Odd elasticity. Nature Physics, 16(4):475–480, 2020.

[96] Hendrik B G Casimir. On Onsager’s principle of microscopic reversibility. Reviews of Modern Physics, 17(2-3):343–350, 1945.

[97] Kohei Kawabata, Ken Shiozaki, Masahito Ueda, and Masatoshi Sato. Symmetry and topology in non-Hermitian physics. Physical Review X, 9(4):041015, 2019.

[98] Martin Brandenbourger, Xander Locsin, Edan Lerner, and Corentin Coulais. Non-reciprocal robotic metamaterials. Nature Communications, 10(1):4608, 2019.

[99] Ananya Ghatak, Martin Brandenbourger, Jasper van Wezel, and Corentin Coulais. Observation of non-Hermitian topology and its bulk–edge correspondence in an active mechanical metamaterial. Proceedings of the National Academy of Sciences, 117(47):29561–29568, 2020.

[100] Cyrill B¨osch, Marc Serra-Garcia, Christian B¨ohm, and Andreas Fichtner. Adjoint computation of Berry phase gradients. Journal of Sound and Vibration, 619:119357, 2025.

[101] Stefano Longhi. Quantum-optical analogies using photonic structures. Laser & Photonics Reviews, 3(3):243–261, 2009.

[102] M. D. Feit and J. A. Fleck, Jr. Light propagation in graded-index optical fibers. Applied Optics, 17(24):3990–3998, 1978.

[103] Sia Ghelichkhan, Angus Gibson, D Rhodri Davies, Stephan C Kramer, and David A Ham. Automatic adjoint-based inversion schemes for geodynamics: reconstructing the evolution of Earth’s mantle in space and time. Geoscientific Model Development, 17(13):5057–5086, 2024.

[104] Patrick E Farrell, David A Ham, Simon W Funke, and Marie E Rognes. Automated derivation of the adjoint of high-level transient finite element programs. SIAM Journal on Scientific Computing, 35(4):C369–C393, 2013.

## Contents of the appendices

A Derivation of the adjoint state equation and the gradient formula 34   
A.1 State equation 34   
A.2 Cost function 34   
A.3 Diferentiability assumptions 35   
A.4 Constrained optimization problem 35   
A.5 The Lagrangian . 35   
A.6 Stationarity with respect to the multipliers: the forward problem 36   
A.7 Stationarity turns the partial derivative into the gradient 36   
A.8 Stationarity with respect to the state: the adjoint problem 37   
A.9 The gradient . 38   
A.10 Integrating the adjoints forward in time 39   
B Physical adjoint for nonlinear second-order systems: time reversal and the damping   
obstruction 40   
B.1 Time reversal requires removing or flipping the damping 40   
B.2 The nudged response on the gain-flipped device . 40   
B.3 Comparison with the adjoint equation: the damping obstruction 41   
B.4 The undamped reciprocal case 41   
C Mass-proportional damping 42   
C.1 Exponential weighting interchanges damping and gain 42   
C.2 Initial conditions, recovery, and the physical algorithm 43   
D General Onsager reciprocity for second-order systems 43   
D.1 Assumptions 43   
D.2 The V-weighted adjoint propagates under the forward operators 44   
D.3 Physical realization and gradient recovery 44   
D.4 Generalization to all other systems 45   
E Adjoint state equation and gradient for first-order systems 45   
E.1 Constrained optimization problem 45   
E.2 The Lagrangian . 46   
E.3 Stationarity with respect to the multipliers: the forward problem 46   
E.4 Stationarity turns the partial derivative into the gradient 46   
E.5 Stationarity with respect to the state: the adjoint problem 46   
E.6 The gradient . 47   
E.7 Integrating the adjoint forward in time 48   
F Wirtinger derivation of the Schr¨odinger adjoint 48   
F.1 Problem and conventions 49   
F.2 Augmented Lagrangian and the Wirtinger adjoint 49   
F.3 Forward-time conjugated adjoint 51   
G Free-space optics 51   
H Details for stationary problems 53   
H.1 The stationary adjoint . 53   
H.2 Linear systems . . . 54   
H.3 Real fields: the static adjoint at a fixed point 54   
H.4 Layered photonic networks . 56   
H.5 Equivalence with scattering-matrix learning rules . 56

I How to test a gradient: the Taylor test

## A Derivation of the adjoint state equation and the gradient formula

In this appendix we derive the adjoint state equation and the gradient formula through the Lagrangian method of PDE-constrained optimization [56, 64], in the Lagrange-multiplier form that is standard for time-dependent inverse problems [57]. The derivation is organized around a single object, the Lagrangian $\mathcal { L } [ \mathbf { u } , \mathbf { b } , \lambda , \mu , \mathbf { p } ]$ , viewed as a functional of independent arguments: the state trajectory $\mathbf { u } ,$ the multipliers b, λ, µ, and the parameters p. Every ingredient of the adjoint method is one of its partial derivatives:

• stationarity with respect to the multipliers, $\mathcal { L } _ { \mathbf { b } } = 0 , \mathcal { L } _ { \lambda } = 0 , \mathcal { L } _ { \mu } = 0$ , recovers the forward problem (state equation and initial conditions);

• stationarity with respect to the state trajectory, ${ \mathcal { L } } _ { \mathbf { u } } = 0 .$ , defines the adjoint problem;

• at a point where all four conditions hold, the remaining partial derivative ${ \mathcal { L } } _ { \mathbf { p } }$ equals the total derivative $d \chi / d \mathbf { p }$ of the reduced cost, and evaluating it yields the gradient as forward–adjoint overlap integrals.

## A.1 State equation

Let V be a real Hilbert space with inner product $\langle \cdot , \cdot \rangle$ . Here $V = \mathbb { R } ^ { N }$ (a discrete system with N degrees of freedom),

$$
\langle \mathbf { x } , \mathbf { y } \rangle = \sum _ { n = 1 } ^ { N } x _ { n } y _ { n } = \mathbf { x } ^ { T } \mathbf { y } .\tag{88}
$$

Let u be a path

$$
\mathbf { u } \in C ^ { 2 } ( [ 0 , T ] , V ) ,\tag{89}
$$

so that each $\mathbf { u } ( t ) \in V$ . The state equation is

$$
\mathbf { M } ( \mathbf { p } _ { M } ) \ddot { \mathbf { u } } ( t ) + \mathbf { D } ( \mathbf { p } _ { D } ) \dot { \mathbf { u } } ( t ) + \mathbf { F } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) = 0 , \qquad t \in [ 0 , T ] ,\tag{90}
$$

with parametrized initial conditions $\mathbf { u } ( 0 ) = \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) , \dot { \mathbf { u } } ( 0 ) = \mathbf { v } _ { 0 } ( \mathbf { p } _ { v _ { 0 } } )$ . Here $\mathbf { M } ( \mathbf { p } _ { M } ) , \mathbf { D } ( \mathbf { p } _ { D } ) : V  V$ are linear operators, $\mathbf { F } [ \cdot , \mathbf { p } _ { F } , t ] : V  V$ is a possibly nonlinear operator, and $\mathbf { f } : [ 0 , T ] \times \mathbb { R } ^ { M _ { f } }  V$ . We assume that $\mathbf { M } ( \mathbf { p } _ { M } )$ is invertible for all visited $\mathbf { p } _ { M }$ . Each term carries its own parameters $\mathbf { p } _ { x } \in \mathbb { R } ^ { M _ { x } }$ for $x \in$ $\{ M , D , F , f , u _ { 0 } , v _ { 0 } \}$ , collected into a single vector $\mathbf { p } = [ \mathbf { p } _ { M } , \mathbf { p } _ { D } , \mathbf { p } _ { F } , \mathbf { p } _ { f } , \mathbf { p } _ { u _ { 0 } } , \mathbf { p } _ { v _ { 0 } } ] \in \mathbb { R } ^ { M }$ with $\begin{array} { r } { M = \sum _ { x } M _ { x } } \end{array}$

We use round brackets for explicit parameter dependence, e.g. $\mathbf { M } ( \mathbf { p } _ { M } )$ , and square brackets for functional dependence, e.g. $\mathbf { F } [ \mathbf { u } ]$

## A.2 Cost function

Given a path $\mathbf { u } \in C ^ { 2 } ( [ 0 , T ] , V )$ , we define the cost $\chi : C ^ { 2 } ( [ 0 , T ] , V ) \to \mathbb { R }$ to be

$$
\chi [ \mathbf { u } ] : = \int _ { 0 } ^ { T } \theta [ \mathbb { P } \mathbf { u } ( t ) ] d t + \psi [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) ] ,\tag{91}
$$

where P is an orthogonal projector of the state onto output degrees of freedom, i.e. $\mathbb { P } = \mathbb { P } ^ { T } = \mathbb { P } ^ { 2 }$ , and $\theta : V  \mathbb { R }$ and $\psi : V \times V \to \mathbb { R }$ are possibly nonlinear functionals.

## A.3 Diferentiability assumptions

The dynamics may be non-autonomous, so F carries an explicit time argument alongside its state and parameter arguments. We require F to be $C ^ { 2 }$ in the latter two and continuous in t, and f to be $C ^ { 2 }$ in p. The derivatives below are taken at fixed t, which we suppress in this subsection. Concretely, for each fixed p the map $\mathbf { u } \mapsto \mathbf { F } [ \mathbf { u } , \mathbf { p } ]$ has a Fr´echet derivative $\mathbf { F } _ { \mathbf { u } } : V \to V$ , a bounded linear operator satisfying

$$
\mathbf { F } [ \mathbf { u } + \varepsilon \mathbf { h } , \mathbf { p } ] = \mathbf { F } [ \mathbf { u } , \mathbf { p } ] + \varepsilon \mathbf { F } _ { \mathbf { u } } [ \mathbf { u } , \mathbf { p } ] \mathbf { h } + { \mathcal { O } } ( \varepsilon ^ { 2 } ) \qquad \forall \mathbf { h } \in V ,\tag{92}
$$

where $\mathbf { F _ { u } } [ \mathbf { u } , \mathbf { p } ] \in \mathbb { R } ^ { N \times N }$ is the Jacobian with entries $\left( \frac { \partial F _ { i } } { \partial u _ { j } } \right)$ . Its adjoint $\mathbf { F } _ { \mathbf { u } } ^ { T } : V \to V$ is defined by

$$
\langle \mathbf { x } , \mathbf { F } _ { \mathbf { u } } \mathbf { h } \rangle = \langle \mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { x } , \mathbf { h } \rangle \qquad \forall \mathbf { x } , \mathbf { h } \in V .\tag{93}
$$

This identity is purely algebraic in V, so no boundary terms arise.

Analogously, for each fixed u the map $\mathbf { p } \mapsto \mathbf { F } [ \mathbf { u } , \mathbf { p } ]$ has a derivative $\mathbf { F _ { p } } ( \mathbf { u } , \mathbf { p } ) : \mathbb { R } ^ { M }  V$ (the $N \times M$ matrix $\left( \frac { \partial F _ { i } } { \partial p _ { j } } \right) )$ , and for each fixed t the map $\mathbf p \mapsto \mathbf f ( t , \mathbf p )$ has a derivative $\mathbf { f _ { p } } ( t , \mathbf { p } ) : \mathbb { R } ^ { M }  V$ . All these derivatives exist and depend continuously on their arguments. $\mathrm { B y }$ the Picard–Lindel¨of theorem, the state equation (90) then has a unique solution $\mathbf { u } ( \cdot ; \mathbf { p } )$ for each p, and this solution depends smoothly on p.

## A.4 Constrained optimization problem

The constrained optimization problem is the following.

Problem A.1 (Constrained optimization of a second-order ODE). Find

$$
\operatorname* { m i n } _ { \mathbf { p \in \mathbb { R } ^ { M } } } \chi [ \mathbf { u } ] \qquad \mathrm { s u b j e c t ~ t o } \qquad \left\{ \begin{array} { l l } { \mathbf { M ( p _ { \cal { M } } ) } \ddot { \mathbf { u } } ( t ) + \mathbf { D } ( \mathbf { p } _ { \cal { D } } ) \dot { \mathbf { u } } ( t ) + \mathbf { F } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] = \mathbf { f } ( t , \mathbf { p } _ { f } ) , \quad t \in [ 0 , T ] , } \\ { \mathbf { u } ( 0 ) = \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) , } \\ { \dot { \mathbf { u } } ( 0 ) = \mathbf { v } _ { 0 } ( \mathbf { p } _ { v _ { 0 } } ) . } \end{array} \right.\tag{94}
$$

The adjoint method computes the gradient of the cost function, $d \chi / d \mathbf { p }$ , at the cost of one additional ODE solve, independent of the number of parameters M. Once the gradient is obtained, the optimization problem can be solved, e.g., via gradient descent:

$$
\mathbf { p }  \mathbf { p } - \alpha \frac { d \chi } { d \mathbf { p } } ,\tag{95}
$$

where $\alpha > 0$ is the learning rate.

## A.5 The Lagrangian

We enforce the state equation with a multiplier path $\mathbf { b } \in C ^ { 2 } ( [ 0 , T ] , V )$ and the two initial conditions with multiplier vectors $\lambda , \mu \in V$ , and define

$$
\begin{array} { r l } & { \mathcal { L } [ { \bf u } , { \bf b } , \boldsymbol { \lambda } , \mu , { \bf p } ] : = \displaystyle \int _ { 0 } ^ { T } \langle { \bf b } ( t ) , \ { \bf M } ( { \bf p } _ { M } ) \ddot { \bf u } ( t ) + { \bf D } ( { \bf p } _ { D } ) \dot { \bf u } ( t ) + { \bf F } [ { \bf u } ( t ) , { \bf p } _ { F } , t ] - { \bf f } ( t , { \bf p } _ { f } ) \rangle d t } \\ & { \qquad +   { \boldsymbol { \lambda } } , \ { \bf u } ( 0 ) - { \bf u } _ { 0 } ( { \bf p } _ { u _ { 0 } } )  +  { \boldsymbol { \mu } } , \ \dot { \bf u } ( 0 ) - { \bf v } _ { 0 } ( { \bf p } _ { v _ { 0 } } )  + \chi [ { \bf u } ] . } \end{array}\tag{96}
$$

This is a functional $\mathcal { L } : C ^ { 2 } ( [ 0 , T ] , V ) \times C ^ { 2 } ( [ 0 , T ] , V ) \times V \times V \times \mathbb { R } ^ { M } \to \mathbb { R } .$

The five arguments of $\mathcal { L }$ are independent variables: L is defined on the full, unconstrained product space. In particular, u is an arbitrary path and is not assumed to solve the state equation, while p enters only through the explicit dependencies ${ \bf M } ( { \bf p } _ { M } ) , ~ { \bf D } ( { \bf p } _ { D } ) , ~ { \bf F } [ \cdot , { \bf p } _ { F } , \cdot ] , ~ { \bf f } ( \cdot , { \bf p } _ { f } ) , ~ { \bf u } _ { 0 } ( { \bf p } _ { u _ { 0 } } ) , ~ { \bf v } _ { 0 } ( { \bf p } _ { v _ { 0 } } )$ . We may therefore diferentiate $\mathcal { L }$ with respect to each argument separately, holding the other four fixed. These partial (Fr´echet) derivatives are denoted $\mathcal { L } _ { \mathbf { b } } , \mathcal { L } _ { \lambda } , \mathcal { L } _ { \mu } , \mathcal { L } _ { \mathbf { u } }$ , and ${ \mathcal { L } } _ { \mathbf { p } }$ . The remainder of the derivation consists of computing each of them in turn.

## A.6 Stationarity with respect to the multipliers: the forward problem

The Lagrangian is afine in b, λ, and $\mu ,$ so these three derivatives can be read of directly. Identifying them with elements of the respective spaces through the inner product,

$$
\mathcal { L } _ { \mathbf { b } } ( t ) = \mathbf { M } \ddot { \mathbf { u } } ( t ) + \mathbf { D } \dot { \mathbf { u } } ( t ) + \mathbf { F } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) ,\tag{97}
$$

and

$$
{ \mathcal { L } } _ { \lambda } = \mathbf { u } ( 0 ) - \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) , \qquad { \mathcal { L } } _ { \mu } = { \dot { \mathbf { u } } } ( 0 ) - \mathbf { v } _ { 0 } ( \mathbf { p } _ { v _ { 0 } } ) ,\tag{98}
$$

so $\mathcal { L } _ { \mathbf { b } } = 0$ holds if and only if u satisfies the state equation (90) on $[ 0 , T ]$ , and $\mathcal { L } _ { \lambda } = 0 , \mathcal { L } _ { \mu } = 0$ if and only if u satisfies the initial conditions.

Stationarity with respect to a multiplier thus recovers the constraint that the multiplier enforces, which is the generic mechanism of Lagrangian relaxation. The three conditions together state that $\mathbf { u } = \mathbf { u } ( \cdot ; \mathbf { p } )$ is the forward solution.

## A.7 Stationarity turns the partial derivative into the gradient

Fix p and evaluate $\mathcal { L }$ at the forward solution $\mathbf { u } = \mathbf { u } ( \cdot ; \mathbf { p } )$ . The constraint terms in (96) vanish, so

$$
\chi ( \mathbf { p } ) = { \mathcal { L } } [ \mathbf { u ( \cdot p ) } , \mathbf { b } , \lambda , \mu , \mathbf { p } ] \qquad \forall \mathbf { b } , \lambda , \mu .\tag{99}
$$

Since the multipliers are arbitrary in (99), we may let them, too, depend on p: pick any assignments $\mathbf { b } ( \cdot ; \mathbf { p } )$ $\lambda ( \mathbf { p } ) , \mu ( \mathbf { p } )$ that depend diferentiably on p (the adjoint state constructed below does). Diferentiating (99) with respect to $\mathbf { p }$ by the chain rule, one term per argument of ${ \mathcal { L } } ,$ gives

$$
\frac { d \chi } { d \mathbf { p } } = \mathcal { L } _ { \mathbf { u } } \frac { d \mathbf { u } } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { b } } \frac { d \mathbf { b } } { d \mathbf { p } } + \mathcal { L } _ { \lambda } \frac { d \lambda } { d \mathbf { p } } + \mathcal { L } _ { \mu } \frac { d \mu } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { p } } ,\tag{100}
$$

with every partial derivative evaluated at the point $( { \mathbf { u } } ( \cdot ; { \mathbf { p } } ) , { \mathbf { b } } ( \cdot ; { \mathbf { p } } ) , \lambda ( { \mathbf { p } } ) , \mu ( { \mathbf { p } } ) , { \mathbf { p } } )$ . On the left stands the total derivative of the reduced cost. On the right stand partial derivatives of $\mathcal { L }$ multiplying the sensitivities of its arguments.

Now inspect (100) term by term. The three multiplier terms vanish because $\mathbf { u } ( \cdot ; \mathbf { p } )$ is the forward solution: by Sec. $\mathrm { A . 6 , ~ } \mathcal { L } _ { \mathbf { b } } = 0 , \mathcal { L } _ { \lambda } = 0 , \mathcal { L } _ { \mu } = 0$ at this point, whatever the sensitivities $d \mathbf { b } / d \mathbf { p } , \ d \lambda / d \mathbf { p }$ $d \pmb { \mu } / d \mathbf { p }$ may be. The remaining obstruction is the first term. The sensitivity $d \mathbf { u } / d \mathbf { p }$ is the expensive object: computing it amounts to diferentiating through the diferential equation solver, one linearized solve per parameter, M in total, which is in general prohibitively expensive. But it enters (100) only multiplied by ${ \mathcal { L } } _ { \mathbf { u } }$ , and the multipliers $\mathbf { b } , \mathbf {  { \lambda } } \lambda ,  { \mu }$ are still free. If we choose them such that

$$
\begin{array} { r } { \mathcal { L } _ { \mathbf { u } } = 0 \qquad \mathrm { a t } ~ \mathbf { u } = \mathbf { u } ( \cdot ; \mathbf { p } ) , } \end{array}\tag{101}
$$

then $d \mathbf { u } / d \mathbf { p }$ is never needed, and

$$
\frac { d \chi } { d \mathbf { p } } = \mathcal { L } _ { \mathbf { p } } :\tag{102}
$$

the total derivative of the cost equals the partial derivative of the Lagrangian, which involves only the explicit and cheap parameter dependence of M, D, F, f, u , v . The condition (101) is a linear equation for $( \mathbf { b } , \lambda , \mu )$ . We now show that it has exactly one solution, the adjoint state.

## A.8 Stationarity with respect to the state: the adjoint problem

Let h $\in C ^ { 2 } ( [ 0 , T ] , V )$ be an arbitrary test path with no constraints at either endpoint, and compute

$$
\mathcal { L } _ { \mathbf { u } } \mathbf { h } : = \operatorname* { l i m } _ { \varepsilon \to 0 } \frac { \mathcal { L } [ \mathbf { u } + \varepsilon \mathbf { h } , \mathbf { b } , \boldsymbol { \lambda } , \mu , \mathbf { p } ] - \mathcal { L } [ \mathbf { u } , \mathbf { b } , \boldsymbol { \lambda } , \mu , \mathbf { p } ] } { \varepsilon } .\tag{103}
$$

The Lagrangian (96) has three groups of terms. We vary each in turn.

(i) The constraint integral. Replacing $\mathbf { u }  \mathbf { u } + \varepsilon \mathbf { h }$ and using linearity of M, D and the diferentiability (92) of F (the drive f has no u-dependence and does not contribute), the variation of the first term of (96) is

$$
\int _ { 0 } ^ { T } \langle \mathbf { b } , \ \mathbf { M } \ddot { \mathbf { h } } + \mathbf { D } \dot { \mathbf { h } } + \mathbf { F } _ { \mathbf { u } } \mathbf { h } \rangle d t .\tag{104}
$$

To read of conditions on b, all terms must appear as $\langle \cdot , \mathbf { h } \rangle$ : we move the operators across the inner product using $\langle \mathbf { x } , \mathbf { M } \mathbf { y } \rangle = \langle \mathbf { M } ^ { T } \mathbf { x } , \mathbf { y } \rangle$ (and (93) for ${ { \bf { F } } _ { \bf { u } } } )$ , and move the time derivatives of h and onto b via integration by parts, twice for the mass term and once for the damping term:

$$
\int _ { 0 } ^ { T } \langle \mathbf { M } ^ { T } \mathbf { b } , \ddot { \mathbf { h } } \rangle d t = \int _ { 0 } ^ { T } \langle \mathbf { M } ^ { T } \ddot { \mathbf { b } } , \mathbf { h } \rangle d t + \Big [ \langle \mathbf { M } ^ { T } \mathbf { b } , \dot { \mathbf { h } } \rangle - \langle \mathbf { M } ^ { T } \dot { \mathbf { b } } , \mathbf { h } \rangle \Big ] _ { 0 } ^ { T } ,\tag{105}
$$

$$
\int _ { 0 } ^ { T } \langle \mathbf { D } ^ { T } \mathbf { b } , \dot { \mathbf { h } } \rangle d t = - \int _ { 0 } ^ { T } \langle \mathbf { D } ^ { T } \dot { \mathbf { b } } , \mathbf { h } \rangle d t + \Big [ \langle \mathbf { D } ^ { T } \mathbf { b } , \mathbf { h } \rangle \Big ] _ { 0 } ^ { T } .\tag{106}
$$

Combining, the variation of the constraint integral is

$$
\int _ { 0 } ^ { T } \langle \mathbf { M } ^ { T } \ddot { \mathbf { b } } - \mathbf { D } ^ { T } \dot { \mathbf { b } } + \mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { b } , \ \mathbf { h } \rangle d t + \Big [ \langle \mathbf { D } ^ { T } \mathbf { b } - \mathbf { M } ^ { T } \dot { \mathbf { b } } , \ \mathbf { h } \rangle + \langle \mathbf { M } ^ { T } \mathbf { b } , \ \dot { \mathbf { h } } \rangle \Big ] _ { 0 } ^ { T } .\tag{107}
$$

(ii) The initial-condition multiplier terms. Under $\mathbf { u }  \mathbf { u } + \varepsilon \mathbf { h }$ , the variation is simply

$$
\langle \lambda , { \bf h } ( 0 ) \rangle + \langle { \pmb \mu } , { \bf \dot { h } } ( 0 ) \rangle .\tag{108}
$$

(iii) The cost. Using the diferentiability of θ and $\psi ,$ , and $\mathbb { P } ^ { T } = \mathbb { P }$

$$
\int _ { 0 } ^ { T } \langle \mathbb { P } \theta _ { \mathbf { u } } , \ \mathbf { h } \rangle d t + \langle \mathbb { P } \psi _ { \mathbf { u } } , \ \mathbf { h } ( T ) \rangle + \langle \mathbb { P } \psi _ { \mathbf { \dot { u } } } , { \dot { \mathbf { h } } } ( T ) \rangle .\tag{109}
$$

The full variation. Adding Eqs. (107), (108), (109) and grouping by where h and h<sup>˙</sup> are evaluated:

$$
\begin{array}{c} \mathcal { L } _ { \mathbf { u } } \mathbf { h } = \int _ { 0 } ^ { T } \langle \mathbf { M } ^ { T } \dot { \mathbf { b } } - \mathbf { D } ^ { T } \dot { \mathbf { b } } + \mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { b } + \mathbb { P } \theta _ { \mathbf { u } } , \mathbf { \Phi } \mathbf { h } \rangle d t  \\ { + \langle \mathbf { D } ^ { T } \mathbf { b } ( T ) - \mathbf { M } ^ { T } \dot { \mathbf { b } } ( T ) + \mathbb { P } \psi _ { \mathbf { u } } , \mathbf { \Phi } \mathbf { h } ( T ) \rangle + \langle \mathbf { M } ^ { T } \mathbf { b } ( T ) + \mathbb { P } \psi _ { \mathbf { u } } , \dot { \mathbf { h } } ( T ) \rangle } \\ { + \langle \mathbf { M } ^ { T } \dot { \mathbf { b } } ( 0 ) - \mathbf { D } ^ { T } \mathbf { b } ( 0 ) + \lambda , \mathbf { \Phi } \mathbf { h } ( 0 ) \rangle + \langle - \mathbf { M } ^ { T } \mathbf { b } ( 0 ) + \mu , \dot { \mathbf { h } } ( 0 ) \rangle . } \end{array}\tag{110}
$$

Reading of the conditions. Since h is arbitrary and unconstrained at both endpoints, the values ${ \bf h } ( t ) , { \bf h } ( 0 ) , \dot { { \bf h } } ( 0 ) , { \bf h } ( T ) , \dot { { \bf h } } ( T )$ can be chosen independently, so $\mathcal { L } _ { \bf u } = 0$ requires each inner product in (110) to vanish separately. The bulk term and the two terms at $t = T$ involve only b. Together they form a terminal-value problem that determines b uniquely and defines the adjoint state.

Definition A.2 (Adjoint state for the second-order problem). The adjoint state $\mathbf { b } ( t ) \in V$ associated with Problem A.1 is the solution of the adjoint ODE

$$
\mathbf { M } ^ { T } \ddot { \mathbf { b } } ( t ) - \mathbf { D } ^ { T } \dot { \mathbf { b } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] \mathbf { b } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } [ \mathbb { P } \mathbf { u } ( t ) ] = 0 , \qquad t \in [ 0 , T ] ,\tag{111}
$$

a second-order ODE requiring two conditions, with the terminal conditions

$$
\mathbf { M } ^ { T } \mathbf { b } ( T ) = - \mathbb { P } \psi _ { \dot { \mathbf { u } } } [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) ] ,\tag{112}
$$

$$
\mathbf { M } ^ { T } \dot { \mathbf { b } } ( T ) = \mathbf { D } ^ { T } \mathbf { b } ( T ) + \mathbb { P } \psi _ { \mathbf { u } } [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) ] .\tag{113}
$$

We solve the adjoint ODE (111) backward from T to 0 using these terminal conditions. This produces b(0) and b<sup>˙</sup> (0), which in turn determine the multipliers via the two terms at $t = 0$ in (110):

$$
\begin{array} { r } { \pmb { \lambda } = - \mathbf { M } ^ { T } \dot { \mathbf { b } } ( 0 ) + \mathbf { D } ^ { T } \mathbf { b } ( 0 ) , } \end{array}\tag{114}
$$

$$
\begin{array} { r } { \pmb { \mu } = \mathbf { M } ^ { T } \mathbf { b } ( 0 ) . } \end{array}\tag{115}
$$

The adjoint ODE is linear in b (the nonlinear forces enter only through their Jacobian, evaluated along the forward trajectory), its coeficients depend continuously on p, and (114)–(115) are explicit. Hence $( \mathbf { b } , \lambda , \mu )$ is the unique solution of $\mathcal { L } _ { \bf u } = 0$ and depends smoothly on p, as required in Sec. A.7.

## A.9 The gradient

All partial derivatives of L are now in place, and the gradient follows by evaluating the last one, ${ \mathcal { L } } _ { \mathbf { p } } .$

Theorem A.3 (Adjoint gradient for the second-order problem). Let $\mathbf { u } = \mathbf { u } ( \cdot ; \mathbf { p } )$ solve the forward problem of Problem A.1 and let b be the adjoint state of Definition A.2. Then the gradient of the reduced cost with respect to p is

$$
\begin{array} { l } { \displaystyle \frac { d \boldsymbol { \chi } } { d { \bf p } } = \int _ { 0 } ^ { T } \langle { \bf b } ( t ) , ~ { \bf M _ { p } } _ { M } \ddot { \bf u } ( t ) + { \bf D _ { p } } _ { D } \dot { \bf u } ( t ) + { \bf F _ { p } } _ { F } [ { \bf u } ( t ) , { \bf p } _ { F } , t ] - { \bf f _ { p } } _ { f } ( t , { \bf p } ) \rangle d t } \\ { \displaystyle \qquad + \langle { \bf M } ^ { T } \dot { \bf b } ( 0 ) - { \bf D } ^ { T } { \bf b } ( 0 ) , \left. \frac { d { \bf u } _ { 0 } } { d { \bf p } _ { u _ { 0 } } } \right. - \langle { \bf M } ^ { T } { \bf b } ( 0 ) , \left. \frac { d { \bf v } _ { 0 } } { d { \bf p } _ { v _ { 0 } } } \right. . } \end{array}\tag{116}
$$

Proof. With $( \mathbf { b } , \lambda , \mu )$ chosen as in Definition A.2 and Eqs. (114)–(115), we have ${ \mathcal { L } } _ { \mathbf { u } } = 0 .$ and the key identity (102) gives $d \chi / d \mathbf { p } = \mathcal { L } _ { \mathbf { p } }$ . The explicit p-dependence of L enters through M(p ), D(p ), F[u, p , t], $\mathbf { f } \left( t , \mathbf { p } _ { f } \right)$ in the constraint integral and through ${ \bf u } _ { 0 } ( { \bf p } _ { u _ { 0 } } ) , { \bf v } _ { 0 } ( { \bf p } _ { v _ { 0 } } )$ in the initial-condition terms. The cost χ has none. Diferentiating each at fixed u, b, λ, µ:

$$
\mathcal { L } _ { \mathbf { p } } = \int _ { 0 } ^ { T } \langle \mathbf { b } ( t ) , \mathbf { M } _ { \mathbf { p } , u } \ddot { \mathbf { u } } ( t ) + \mathbf { D } _ { \mathbf { p } , D } \dot { \mathbf { u } } ( t ) + \mathbf { F } _ { \mathbf { p } _ { F } } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } _ { \mathbf { p } _ { f } } ( t , \mathbf { p } ) \rangle d t - \langle \boldsymbol { \lambda } , \frac { d \mathbf { u } _ { 0 } } { d \mathbf { p } _ { u _ { 0 } } } \rangle - \langle \mu , \frac { d \mathbf { v } _ { 0 } } { d \mathbf { p } _ { v _ { 0 } } } \rangle .\tag{117}
$$

Substituting the expressions (114)–(115) for λ and µ gives (116).

Every quantity on the right-hand side of (116) is known: u(t) from the forward solve, b(t) from the backward solve, and the partial derivatives $\mathbf { M _ { p } } _ { M } , \ \mathbf { D } _ { \mathbf { p } _ { D } } , \ \mathbf { F } _ { \mathbf { p } _ { F } } , \ \mathbf { f _ { p } } _ { f } , \ d \mathbf { u } _ { 0 } / d \mathbf { p } _ { u _ { 0 } } , \ d \mathbf { v } _ { 0 } / d \mathbf { p } _ { v _ { 0 } }$ from the model. The cost of evaluating (116) is one forward solve plus one backward solve, independent of the number of parameters M.

If M and D do not depend on p and the initial conditions are p-independent, Eq. (116) reduces to

$$
{ \frac { d { \boldsymbol { \chi } } } { d \mathbf { p } } } = \int _ { 0 } ^ { T } \langle \mathbf { b } ( t ) , \mathbf { F } _ { \mathbf { p } } ( \mathbf { u } ( t ) , \mathbf { p } ) - \mathbf { f } _ { \mathbf { p } } ( t , \mathbf { p } ) \rangle d t .\tag{118}
$$

Interpreting $\langle \mathbf { b } , \mathbf { F _ { p } } \rangle$ . Recall that $\mathbf { F _ { p } } : \mathbb { R } ^ { M }  V$ is a linear operator, not an element of V, so the symbol $\langle \mathbf { b } , \mathbf { F _ { p } } \rangle$ is a shorthand for an M-component vector, to be read componentwise: for each parameter $p _ { j }$ , the partial $\partial \mathbf { F } / \partial p _ { j } \in V$ is a genuine vector, and the j-th component of the gradient is the scalar inner product

$$
\frac { d \chi } { d p _ { j } } = \int _ { 0 } ^ { T } \left. { \bf b } ( t ) , ~ \frac { \partial { \bf F } } { \partial p _ { j } } [ { \bf u } ( t ) , { \bf p } ] - \frac { \partial { \bf f } } { \partial p _ { j } } ( t , { \bf p } ) \right. d t .\tag{119}
$$

Equivalently, using the adjoint operator $\mathbf { F } _ { \mathbf { p } } ^ { T } : V \to \mathbb { R } ^ { M }$ defined by $\langle \mathbf { b } , \mathbf { F } _ { \mathbf { p } } \mathbf { q } \rangle _ { V } = \langle \mathbf { F } _ { \mathbf { p } } ^ { T } \mathbf { b } , \mathbf { q } \rangle _ { \mathbb { R } ^ { M } }$ for all $\mathbf { q } \in \mathbb { R } ^ { M }$ the compact form is

$$
\frac { d \chi } { d \mathbf { p } } = \int _ { 0 } ^ { T } \left( \mathbf { F } _ { \mathbf { p } } ^ { T } \mathbf { b } ( t ) - \mathbf { f } _ { \mathbf { p } } ^ { T } \mathbf { b } ( t ) \right) d t \ \in \ \mathbb { R } ^ { M } .\tag{120}
$$

Hence $\mathbf { F _ { p } ^ { \it T } } \mathbf { b }$ is the M-vector obtained by multiplying the transpose of the $N \times M$ Jacobian by the N-vector b. The same interpretation applies to the ${ \bf F } _ { { \bf p } _ { F } } , { \bf M } _ { { \bf p } _ { M } } , { \bf D } _ { { \bf p } _ { D } }$ , and $\mathbf { f } _ { \mathbf { p } _ { f } }$ terms in the full gradient (116).

Direct optimization of initial conditions. Suppose a subvector of p encodes the initial state itself, i.e. $\mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) = \mathbf { p } _ { u _ { 0 } }$ with $\mathbf { p } _ { u _ { 0 } } \in V$ . Then $d { \bf u } _ { 0 } / d { \bf p } _ { u _ { 0 } } = I _ { V }$ and $d \mathbf { v } _ { 0 } / d \mathbf { p } _ { u _ { 0 } } = 0$ , so the $\mathbf { u } _ { \mathrm { 0 } } \mathrm { - b l o c k }$ of Eq. (116) identifies the gradient directly:

$$
\frac { d \boldsymbol { \chi } } { d { \bf u } _ { 0 } } = { \bf M } ^ { T } \dot { \bf b } ( 0 ) - { \bf D } ^ { T } { \bf b } ( 0 ) .\tag{121}
$$

Analogously, if a subvector of p equals $\mathbf { v } _ { 0 }$ , then $d { \bf v } _ { 0 } / d { \bf p } _ { v _ { 0 } } = I _ { V } , d { \bf u } _ { 0 } / d { \bf p } _ { v _ { 0 } } = 0$ , and

$$
\frac { d \chi } { d \mathbf { v } _ { 0 } } = - \mathbf { M } ^ { T } \mathbf { b } ( 0 ) .\tag{122}
$$

## A.10 Integrating the adjoints forward in time

The adjoint problem of Definition A.2 is posed backward in time: the data (112)–(113) are imposed at the final time and the ODE (111) is integrated from T down to 0. Physical hardware, by contrast, runs forward. The change of variable $t \mapsto T - t$ removes the mismatch.

Define the forward-time adjoint field

$$
\mathbf { a } ( t ) : = \mathbf { b } ( T - t ) , \qquad \dot { \mathbf { a } } ( t ) = - \dot { \mathbf { b } } ( T - t ) , \qquad \ddot { \mathbf { a } } ( t ) = \ddot { \mathbf { b } } ( T - t ) ,\tag{123}
$$

so that odd derivatives flip sign under the reversal and even ones do not. Evaluating the adjoint ODE (111) at time $T - t$ and substituting (123) gives the forward-time adjoint equation

$$
\mathbf { M } ^ { T } \ddot { \mathbf { a } } ( t ) + \mathbf { D } ^ { T } \dot { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t ] \mathbf { a } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } [ \mathbb { P } \mathbf { u } ( T - t ) ] = 0 , \qquad t \in [ 0 , T ] .\tag{124}
$$

Among the operators only the damping changes, $- \mathbf { D } ^ { T }  + \mathbf { D } ^ { T }$ , with mass and stifness untouched. This sign flip is the entire algebraic efect of time reversal on the dissipative term, and it is the origin of the obstruction to a physical realization in dissipative systems studied in Appendix B. The linearization and the source are now read along the reversed trajectory $\mathbf { u } ( T - t )$

The terminal data sit at the reversed time $t = 0$ . With ${ \bf b } ( T ) = { \bf a } ( 0 )$ and $\dot { \mathbf { b } } ( T ) = - \dot { \mathbf { a } } ( 0 )$ , the terminal conditions (112)–(113) become genuine initial conditions,

$$
\mathbf { M } ^ { T } \mathbf { a } ( 0 ) = - \mathbb { P } \psi _ { \dot { \mathbf { u } } } [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) ] ,\tag{125}
$$

$$
\mathbf { M } ^ { T } \dot { \mathbf { a } } ( 0 ) = - \mathbf { D } ^ { T } \mathbf { a } ( 0 ) - \mathbb { P } \psi _ { \mathbf { u } } [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) ] .\tag{126}
$$

Likewise ${ \bf b } ( 0 ) = { \bf a } ( T )$ and $\dot { \mathbf { b } } ( 0 ) = - \dot { \mathbf { a } } ( T )$ , so substituting ${ \mathbf { b } ( t ) = \mathbf { a } ( T - t ) }$ throughout the gradient (116) gives

$$
\frac { d \chi } { d \mathbf { p } } = \int _ { 0 } ^ { T } \left. \mathbf { a } ( T - t ) , \ \mathbf { M _ { p } } _ { M } \ddot { \mathbf { u } } ( t ) + \mathbf { D _ { p } } _ { D } \ddot { \mathbf { u } } ( t ) + \mathbf { F _ { p } } _ { F } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f _ { p } } _ { f } ( t , \mathbf { p } ) \right. d t
$$

$$
- \left. \mathbf { M } ^ { T } \dot { \mathbf { a } } ( T ) + \mathbf { D } ^ { T } \mathbf { a } ( T ) , \ \frac { d \mathbf { u } _ { 0 } } { d \mathbf { p } _ { u _ { 0 } } } \right. - \left. \mathbf { M } ^ { T } \mathbf { a } ( T ) , \ \frac { d \mathbf { v } _ { 0 } } { d \mathbf { p } _ { v _ { 0 } } } \right. .\tag{127}
$$

Passing to forward time therefore does three things: it flips the sign of the damping operator, it reverses the trajectory argument ${ \mathbf u } ( t ) \mapsto { \mathbf u } ( T - t )$ feeding the linearization and the source, and it turns the terminal data at $t = T$ into initial data at $t = 0$

Eqs. (124)–(127) are the adjoint problem and gradient formula quoted in the main text.

## B Physical adjoint for nonlinear second-order systems: time reversal and the damping obstruction

This appendix derives the nudged time-reversed dynamics quoted in Sec. 6, in particular the gain-flipped perturbation equation (29), and proves the nonlinear part of Theorem 6.1. Throughout we use the forwardtime adjoint equation (4), which propagates with a damping term $+ \mathbf { D } ^ { T } \dot { \mathbf { a } }$ , an internal-force linearization $\mathbf { F } _ { \mathbf { u } } ^ { T }$ evaluated along the time-reversed trajectory $\mathbf { u } ( T - t )$ , the source (5), and initial conditions (6)–(7) set by the terminal-loss derivatives $\mathbb { P } \psi _ { \mathbf { u } }$ and $\mathbb { P } \psi _ { \dot { \mathbf { u } } }$

## B.1 Time reversal requires removing or flipping the damping

Unlike the linear case, in the nonlinear adjoint (4) the field a propagates under $\mathbf { F } _ { \mathbf { u } } ^ { T }$ evaluated along the reversed trajectory $\mathbf { u } ( T - t )$ . To access this linearization physically one must first reproduce that reversed trajectory on the device. Define

$$
\mathbf { w } ( t ) : = \mathbf { u } ( T - t ) , \qquad \dot { \mathbf { w } } ( t ) = - \dot { \mathbf { u } } ( T - t ) , \qquad \ddot { \mathbf { w } } ( t ) = \ddot { \mathbf { u } } ( T - t ) .\tag{128}
$$

Evaluating the forward state equation (2) at $T - t$ and using (128) gives the reversed state equation (28),

$$
\begin{array} { r } { \mathbf { M } \ddot { \mathbf { w } } ( t ) - \mathbf { D } \dot { \mathbf { w } } ( t ) + \mathbf { F } \big [ \mathbf { w } ( t ) , \mathbf { p } _ { F } , T - t \big ] - \mathbf { f } \big ( T - t , \mathbf { p } _ { f } \big ) = \mathbf { 0 } , } \end{array}\tag{129}
$$

with initial conditions

$$
\mathbf { w } ( 0 ) = \mathbf { u } ( T ) , \qquad \dot { \mathbf { w } } ( 0 ) = - \dot { \mathbf { u } } ( T ) .\tag{130}
$$

The only diference from the forward equation (2) is the sign of the damping term: time reversal sends Du˙ $\mapsto - \mathbf { D } \dot { \mathbf { w } }$ , because the first derivative is odd under $t \mapsto T - t$ while the second derivative is even. Reproducing w on the same hardware therefore requires either $\mathbf { D = 0 }$ or an experimental replacement $\mathbf { D } \mapsto - \mathbf { D }$ that turns damping into gain. We grant the latter possibility in the next subsection and show that it still fails to deliver the adjoint.

## B.2 The nudged response on the gain-flipped device

Suppose the gain flip D 7→ −D can be realized, so that (129) genuinely runs on the device and produces the reversed background w. On this same gain-flipped device we superimpose an infinitesimal nudge: a source in the direction of the adjoint source (5) and a perturbation of the initial conditions in the directions of the adjoint terminal data (6)–(7). The nudged field $\mathbf { w } ^ { \epsilon }$ obeys

$$
\mathbf { M } \ddot { \mathbf { w } } ^ { \epsilon } ( t ) - \mathbf { D } \dot { \mathbf { w } } ^ { \epsilon } ( t ) + \mathbf { F } \big [ \mathbf { w } ^ { \epsilon } ( t ) , \mathbf { p } _ { F } , T - t \big ] - \mathbf { f } ( T - t , \mathbf { p } _ { f } ) + \epsilon \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } ,\tag{131}
$$

with initial conditions

$$
\mathbf { w } ^ { \epsilon } ( 0 ) = \mathbf { u } ( T ) + \epsilon \mathbf { a } ( 0 ) , \qquad \dot { \mathbf { w } } ^ { \epsilon } ( 0 ) = - \dot { \mathbf { u } } ( T ) + \epsilon \dot { \mathbf { a } } ( 0 ) , \qquad | \epsilon | \ll 1 ,\tag{132}
$$

where ${ \bf a } ( 0 )$ and $\dot { \mathbf { a } } ( 0 )$ are the adjoint initial data (6)–(7) solved for the leading derivative,

$$
\mathbf { a } ( 0 ) = - \mathbf { M } ^ { - 1 } \mathbb { P } \psi _ { \dot { \mathbf { u } } } ,\tag{133}
$$

$$
\dot { { \mathbf a } } ( 0 ) = - { \mathbf M } ^ { - 1 } \big [ { \mathbf D } ^ { T } { \mathbf a } ( 0 ) + \mathbb { P } \psi _ { \mathbf { u } } \big ] = { \mathbf M } ^ { - 1 } \big [ { \mathbf D } ^ { T } { \mathbf M } ^ { - 1 } \mathbb { P } \psi _ { \dot { \mathbf { u } } } - \mathbb { P } \psi _ { \mathbf { u } } \big ] ,\tag{134}
$$

where we set $\mathbf { M } ^ { T } = \mathbf { M }$ . The velocity nudge carries the damping contribution $\mathbf { D } ^ { T } \mathbf { a } ( 0 )$ demanded by (7), which drops out when $\mathbf { D = 0 }$ . Write the diferential response

$$
\delta \mathbf { w } ( t ) : = \mathbf { w } ^ { \epsilon } ( t ) - \mathbf { w } ( t ) = { \mathcal { O } } ( \epsilon ) .\tag{135}
$$

Expanding the internal force to first order,

$$
\mathbf { F } \big [ \mathbf { w } ^ { \epsilon } , \mathbf { p } _ { F } , T - t \big ] = \mathbf { F } \big [ \mathbf { w } , \mathbf { p } _ { F } , T - t \big ] + \mathbf { F _ { u } } \big [ \mathbf { w } ( t ) , \mathbf { p } _ { F } , T - t \big ] \delta \mathbf { w } ( t ) + \mathcal { O } \big ( \| \delta \mathbf { w } \| ^ { 2 } \big ) ,\tag{136}
$$

and subtracting the reversed state equation (129) from (131) gives the gain-flipped perturbation equation (29),

$$
\mathbf { M } \delta \ddot { \mathbf { w } } ( t ) - \mathbf { D } \delta \dot { \mathbf { w } } ( t ) + \mathbf { F } _ { \mathbf { u } } \big [ \mathbf { w } ( t ) , \mathbf { p } _ { F } , T - t \big ] \delta \mathbf { w } ( t ) + \epsilon \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] + \mathcal { O } ( \epsilon ^ { 2 } ) = \mathbf { 0 } ,\tag{137}
$$

with initial conditions obtained by subtracting (130) from (132),

$$
\delta { \bf w } ( 0 ) = \epsilon { \bf a } ( 0 ) , \qquad \delta \dot { \bf w } ( 0 ) = \epsilon \dot { \bf a } ( 0 ) .\tag{138}
$$

Because $\mathbf { w } ( t ) = \mathbf { u } ( T - t )$ , the linearization ${ \bf F _ { u } } [ { \bf w } ( t ) , { \bf p } _ { F } , T - t ]$ in (137) is precisely the time-reversed Jacobian $\mathbf { F _ { u } } [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t ]$ appearing in the adjoint equation (4).

## B.3 Comparison with the adjoint equation: the damping obstruction

Introduce the rescaled response $\widetilde { \mathbf { a } } ( t ) : = \delta \mathbf { w } ( t ) / \epsilon$ , which is $\mathcal { O } ( 1 )$ . Dividing (137)–(138) by ϵ and letting $\epsilon  0$

$$
\begin{array} { r } { \mathbf { M } \tilde { \mathbf { a } } ( t ) - \mathbf { D } \tilde { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } \big [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \tilde { \mathbf { a } } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } , } \end{array}\tag{139}
$$

with $\widetilde { \mathbf { a } } ( 0 ) = \mathbf { a } ( 0 )$ and $\dot { \widetilde { \mathbf { a } } } ( 0 ) = \dot { \mathbf { a } } ( 0 )$ , given by (133)–(134). We compare this term by term with the forwardtime adjoint equation (4),

$$
\mathbf { M } ^ { T } \ddot { \mathbf { a } } ( t ) + \mathbf { D } ^ { T } \dot { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } \big [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \mathbf { a } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } ,\tag{140}
$$

whose initial conditions $( 6 )  { - } ( 7 )$ the nudge (132) reproduces by construction. What remains is the propagator, and matching (139) to (140) requires three things:

• Mass. $\mathbf { M } ^ { T } = \mathbf { M } .$ so the inertial operators agree.

• Stifness. $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } }$ along the reversed trajectory, so the two internal-force linearizations agree, i.e. the tangent dynamics is reciprocal.

• Damping. The rescaled response carries $- \mathbf { D } \dot { \widetilde { \mathbf { a } } }$ , whereas the adjoint carries $+ \mathbf { D } ^ { T } \dot { \mathbf { a } }$ . For reciprocal damping $\mathbf { D } ^ { T } = \mathbf { D }$ the two terms difer by 2Da˙ , and they coincide only when $\mathbf { D = 0 }$

Even after granting the gain flip needed to reproduce the reversed background, the damping enters the nudged perturbation with the sign −D, the opposite of the $+ \mathbf { D } ^ { T }$ demanded by the adjoint. A discrepancy in the propagator cannot be repaired by any choice of source or initial condition, so the same hardware cannot simultaneously carry the reversed background and propagate the correct adjoint perturbation around it unless the damping vanishes.

## B.4 The undamped reciprocal case

Set $\mathrm { ~ \bf ~ D ~ } = \mathrm { ~ \bf ~ 0 ~ }$ and assume $\mathbf { M } ^ { T } = \mathbf { M }$ together with $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } }$ along the trajectory. Then the reversed background (129) runs on the same hardware, and the rescaled perturbation equation (139) coincides with the adjoint equation (140), whose initial conditions the nudge already carries. Here (133)–(134) reduce to

$\mathbf { a } ( 0 ) = - \mathbf { M } ^ { - 1 } \mathbb { P } \psi _ { \dot { \mathbf { u } } }$ and $\dot { \bf a } ( 0 ) = - { \bf M } ^ { - 1 } { \mathbb { P } } \psi _ { \bf u }$ , the nudge quoted in the main text. By uniqueness of solutions, $\widetilde { \mathbf { a } } = \mathbf { a }$ , i.e.

$$
\mathbf { a } ( t ) = \operatorname* { l i m } _ { \epsilon  0 } { \frac { \mathbf { w } ^ { \epsilon } ( t ) - \mathbf { w } ( t ) } { \epsilon } } , \qquad \mathbf { w } ( t ) = \mathbf { u } ( T - t ) .\tag{141}
$$

The adjoint field is therefore the infinitesimal diferential response of the nudged time-reversed trajectory, and the gradient follows from the overlap integrals (8)–(13). This proves the nonlinear part of Theorem 6.1. □

## C Mass-proportional damping

The obstruction established in Appendix B forbids damping in the nonlinear case because the nudged response on the gain-flipped device propagates with <sup>−</sup>D <sup>˙</sup>ae, while the adjoint requires $+ \mathbf { D } ^ { T } \dot { \mathbf { a } }$ . One structurally special case reconciles the two: mass-proportional damping. The construction still flips damping to gain, so the adjoint propagates on a device other than the one that ran the forward pass, which places it outside the same-device constructions of this article. The mechanism is instructive nonetheless.

Mass-proportional damping means

$$
{ \bf D } = \gamma { \bf M } , \qquad \gamma \in \mathbb { R } \textrm { a k n o w n } \mathrm { s c a l a r } .\tag{142}
$$

Because the damping operator is now a scalar multiple of the inertia, a single scalar exponential weighting in time interchanges the damped and anti-damped equations while leaving M and the stifness untouched. We assume, as in Appendix B, reciprocity along the reversed trajectory, $\mathbf { M } ^ { T } = \mathbf { M }$ and $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } }$ . Note that (142) makes D automatically reciprocal, $\mathbf { D } ^ { T } = \gamma \mathbf { M } ^ { T } = \mathbf { D }$

## C.1 Exponential weighting interchanges damping and gain

With $\mathbf { D } = \gamma \mathbf { M }$ the forward-time adjoint equation (4) reads

$$
\mathbf { M } \ddot { \mathbf { a } } ( t ) + \gamma \mathbf { M } \dot { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } \big [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \mathbf { a } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } ,\tag{143}
$$

i.e. a damped linear oscillator for a. The gain-flipped device of Appendix B, by contrast, can only realize the anti-damped perturbation dynamics

$$
\mathbf { M } \ddot { \mathbf { a } } ( t ) - \gamma \mathbf { M } \dot { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } \big [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \tilde { \mathbf { a } } ( t ) + \pmb { \sigma } ( t ) = \mathbf { 0 } ,\tag{144}
$$

where $\sigma$ is the source we are free to apply. Substitute the exponential ansatz

$$
\mathbf { a } ( t ) = e ^ { - \gamma t } \widetilde { \mathbf { a } } ( t )\tag{145}
$$

into (143), using $\dot { \mathbf { a } } = e ^ { - \gamma t } ( \dot { \widetilde { \mathbf { a } } } - \gamma \widetilde { \mathbf { a } } )$ and $\ddot { \mathbf { a } } = e ^ { - \gamma t } ( \ddot { \tilde { \mathbf { a } } } - 2 \gamma \dot { \tilde { \mathbf { a } } } + \gamma ^ { 2 } \widetilde { \mathbf { a } } )$ . The $\gamma ^ { 2 } \mathbf { M }$ contributions cancel and the coeficient of $\overset { \cdot } { \mathbf { a } }$ flips from +γ to −γ, leaving

$$
e ^ { - \gamma t } \Big [ \mathbf { M } \ddot { \mathbf { a } } - \gamma \mathbf { M } \dot { \mathbf { a } } + \mathbf { F } _ { \mathbf { u } } \mathbf { \widetilde { a } } \Big ] + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } .\tag{146}
$$

Multiplying by $e ^ { \gamma t }$ shows that $\widetilde { \mathbf { a } }$ obeys the anti-damped equation (144), provided the applied source is the exponentially weighted adjoint source

$$
\pmb { \sigma } ( t ) = e ^ { \gamma t } \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] .\tag{147}
$$

Crucially, the mass and stifness operators in (144) are identical to those of the device: only the sign of the damping has flipped, which the gain flip supplies, and the source carries the known scalar weight $e ^ { \gamma t }$ . For a terminal-loss-only objective $( \theta \equiv 0 )$ the source vanishes and no weighting of the drive is needed at all.

## C.2 Initial conditions, recovery, and the physical algorithm

The ansatz (145) also maps the adjoint initial conditions $( 6 ) – ( 7 ) \ ( \mathrm { w i t h \ } \mathbf { D } = \gamma \mathbf { M } )$ onto undamped-type conditions for ae. From $\mathbf { a } ( 0 ) = \widetilde \mathbf { \mathbf { a } } ( 0 )$ and $\dot { \mathbf { a } } ( 0 ) = \dot { \widetilde { \mathbf { a } } } ( 0 ) - \gamma \widetilde { \mathbf { a } } ( 0 )$ , the terms proportional to $\gamma$ cancel and

$$
\mathbf { M } \mathbf { \widetilde { a } } ( 0 ) = - \mathbb { P } \psi _ { \dot { \mathbf { u } } } , \qquad \mathbf { M } \mathbf { \dot { \widetilde { a } } } ( 0 ) = - \mathbb { P } \psi _ { \mathbf { u } } ,\tag{148}
$$

i.e. the same nudge directions as in the undamped case of $\mathrm { A p p e n d i x }$ B.4. The physically measured field $\widetilde { \mathbf { a } }$ is therefore obtained as in the undamped nonlinear construction, with the single modification that the adjoint source is applied with the weight $e ^ { \gamma t }$ . The true adjoint is recovered by undoing the weighting,

$$
\mathbf { a } ( t ) = e ^ { - \gamma t } { \widetilde { \mathbf { a } } } ( t ) = e ^ { - \gamma t } \operatorname* { l i m } _ { \epsilon \to 0 } { \frac { \mathbf { w } ^ { \epsilon } ( t ) - \mathbf { w } ( t ) } { \epsilon } } , \qquad \mathbf { w } ( t ) = \mathbf { u } ( T - t ) ,\tag{149}
$$

before substitution into the gradient overlaps (8)–(13). Equivalently, the weight can be carried directly into the overlaps: since $\mathbf { a } ( T - t ) = e ^ { - \gamma ( T - t ) } \mathbf { \widetilde { a } } ( T - t )$ , each integrand acquires the known scalar factor $e ^ { - \gamma ( T - t ) }$ e.g.

$$
\frac { d \chi } { d \mathbf { p } _ { F } } = \int _ { 0 } ^ { T } e ^ { - \gamma ( T - t ) } \left. \widetilde { \mathbf { a } } ( T - t ) , \mathbf { F } _ { \mathbf { p } _ { F } } \big [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t \big ] \right. d t ,\tag{150}
$$

which realizes the exponential time weighting referenced in Sec. 6. Concretely:

1. Run the forward dynamics and record u(t), then form $\mathbf { w } ( t ) = \mathbf { u } ( T - t )$

2. Flip the damping to gain, $\mathbf { D } = \gamma \mathbf { M } \mapsto - \gamma \mathbf { M }$ , so the device reproduces the reversed background $\mathbf { w } .$

3. On the same gain-flipped device, apply the TRM and nudge together: initialize with ${ \bf w } ^ { \epsilon } ( 0 ) = { \bf u } ( T ) -$ $\epsilon \mathbf { M } ^ { - 1 } \mathbb { P } \psi _ { \dot { \mathbf { u } } }$ and $\dot { \mathbf { w } } ^ { \epsilon } ( 0 ) = - \dot { \mathbf { u } } ( T ) - \epsilon \mathbf { M } ^ { - 1 } \mathbb { P } \psi _ { \mathbf { u } }$ , drive with $\mathbf { f } ( T - t ) - \epsilon e ^ { \gamma t } \mathbb { P } \theta _ { \mathbf { u } } [ \mathbb { P } \mathbf { u } ( T - t ) ]$ , and reverse the explicit time dependence of F.

4. Measure $\begin{array} { r } { \widetilde { \mathbf { a } } ( t ) = \operatorname* { l i m } _ { \epsilon  0 } [ \mathbf { w } ^ { \epsilon } ( t ) - \mathbf { w } ( t ) ] / \epsilon } \end{array}$ and form the adjoint $\mathbf { a } ( t ) = e ^ { - \gamma t } \widetilde { \mathbf { a } } ( t )$

5. Evaluate the gradient from (8)–(13), equivalently (150).

Two remarks. First, the construction hinges on $\gamma$ being a known scalar : only then is $e ^ { - \gamma t }$ a scalar weight that commutes with M and $\mathbf { F } _ { \mathbf { u } }$ and leaves the stifness untouched in (146). A damping D that is not a scalar multiple of M would require a matrix exponential $e ^ { - \mathbf { M } ^ { - 1 } \mathbf { D } t }$ that does not, in general, commute with $\mathbf { F } _ { \mathbf { u } }$ , and the obstruction of Appendix B returns. Second, as in the undamped case, this still requires the gain flip to reproduce the reversed background. Mass-proportional damping only makes the subsequent adjoint extraction exact, through the known exponential reweighting [79].

## D General Onsager reciprocity for second-order systems

In the main text $\mathrm { ( S e c . ~ 5 ) }$ we set the Onsager mobility to $\mathbf { V } = \mathbf { I }$ , so that reciprocity reduces to Euclidean symmetry. Here we prove the general claim made there: when the forward operators are V-reciprocal in the sense of Eq. (22), the physical hardware propagates the V-weighted adjoint field $\mathbf { d } = \mathbf { V } \mathbf { a }$ , and the Euclidean adjoint $\mathbf { a } = \mathbf { V } ^ { - 1 } \mathbf { d }$ that enters the gradient formulas is recovered by applying $\mathbf { V } ^ { - 1 }$ . We work with the second-order real-valued system of Sec. 2. As shown at the end, the argument is a similarity transformation by V and carries over verbatim to all the other systems.

## D.1 Assumptions

Let ${ \bf V } = { \bf V } ^ { T } \succsim 0$ be a constant, invertible Onsager mobility. We assume that all three forward operators are V-reciprocal,

$$
\mathbf { V } \mathbf { M } ^ { T } \mathbf { V } ^ { - 1 } = \mathbf { M } , \qquad \mathbf { V } \mathbf { D } ^ { T } \mathbf { V } ^ { - 1 } = \mathbf { D } , \qquad \mathbf { V } \mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { V } ^ { - 1 } = \mathbf { F } _ { \mathbf { u } } ,\tag{151}
$$

the last evaluated along the (time-reversed) forward trajectory. In the common symmetric case $\mathbf { M } ^ { T } = \mathbf { M }$ and $\mathbf { D } ^ { T } = \mathbf { D }$ , the first two conditions are the commutation relations MV = VM and $\mathbf { D } \mathbf { V } = \mathbf { V } \mathbf { D }$

The condition on $\mathbf { F } _ { \mathbf { u } }$ is the linearized form of an Onsager-symmetric gradient flow: if the internal force derives from a scalar energy through the mobility,

$$
\mathbf { F } ( \mathbf { u } , \mathbf { p } _ { F } ) = \mathbf { V } E _ { \mathbf { u } } ( \mathbf { u } , \mathbf { p } _ { F } ) ,\tag{152}
$$

then $\mathbf { F } _ { \mathbf { u } } = \mathbf { V } E _ { \mathbf { u u } }$ with the Hessian $E _ { \mathbf { u u } }$ symmetric, so $\mathbf { V } \mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { V } ^ { - 1 } = \mathbf { V } E _ { \mathbf { u u } } ^ { T } \mathbf { V } ^ { T } \mathbf { V } ^ { - 1 } = \mathbf { V } E _ { \mathbf { u u } } = \mathbf { F } _ { \mathbf { u } }$ , as required. For a quadratic energy $\begin{array} { r } { E = \frac { 1 } { 2 } \langle \mathbf { u } , \mathbf { H } \mathbf { u } \rangle } \end{array}$ with $\mathbf { H } ^ { T } = \bar { \mathbf { H } }$ this is the linear Onsager-reciprocal case $\mathbf { F } = \mathbf { V } \mathbf { H } \mathbf { u }$ . For nonquadratic E the dynamics is nonlinear, but the mobility is still constant and reciprocal.

## D.2 The V-weighted adjoint propagates under the forward operators

Recall the forward-time adjoint equation (4),

$$
\mathbf { M } ^ { T } \ddot { \mathbf { a } } ( t ) + \mathbf { D } ^ { T } \dot { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } \big [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \mathbf { a } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } .\tag{153}
$$

Define the transformed field

$$
\mathbf { d } ( t ) : = \mathbf { V } \mathbf { a } ( t ) , \qquad { \mathrm { e q u i v a l e n t l y } } \qquad \mathbf { a } ( t ) = \mathbf { V } ^ { - 1 } \mathbf { d } ( t ) .\tag{154}
$$

Substituting $\mathbf { a } = \mathbf { V } ^ { - 1 } \mathbf { d }$ into (153) and multiplying on the left by V,

$$
\begin{array} { r } { \left( \mathbf { V } \mathbf { M } ^ { T } \mathbf { V } ^ { - 1 } \right) \ddot { \mathbf { d } } + \left( \mathbf { V } \mathbf { D } ^ { T } \mathbf { V } ^ { - 1 } \right) \dot { \mathbf { d } } + \left( \mathbf { V } \mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { V } ^ { - 1 } \right) \mathbf { d } + \mathbf { V } \mathbb { P } \theta _ { \mathbf { u } } \left[ \mathbb { P } \mathbf { u } ( T - t ) \right] = \mathbf { 0 } . } \end{array}\tag{155}
$$

By the V-reciprocity assumptions (151), the three conjugated operators collapse to the forward operators $\mathbf { M } , \mathbf { D } , \mathbf { F _ { u } }$ , leaving

$$
\begin{array} { r } { \mathbf { M } \ddot { \mathbf { d } } ( t ) + \mathbf { D } \dot { \mathbf { d } } ( t ) + \mathbf { F } _ { \mathbf { u } } \big [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \mathbf { d } ( t ) + \mathbf { V } \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } . } \end{array}\tag{156}
$$

This is governed by the same linearized operator that the physical device realizes around the (time-reversed) forward trajectory, as in the Euclidean case $\mathbf { V } = \mathbf { I }$ , but with the adjoint source now carrying the weight V.

The adjoint initial conditions (6)–(7) transform the same way. Substituting $\mathbf { a } = \mathbf { V } ^ { - 1 } \mathbf { d }$ and multiplying by V,

$$
\mathbf { M d } ( 0 ) = - \mathbf { V } \mathbb { P } \psi _ { \dot { \mathbf { u } } } \big [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) \big ] ,\tag{157}
$$

$$
\begin{array} { r } { \mathbf { M } \dot { \mathbf { d } } ( 0 ) = - \mathbf { D } \mathbf { d } ( 0 ) - \mathbf { V } \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \dot { \mathbf { u } } ( T ) \big ] , } \end{array}\tag{158}
$$

again using (151). Thus d solves the forward-operator problem (156)–(158), with every adjoint source and initial-condition kick simply pre-multiplied by V.

## D.3 Physical realization and gradient recovery

Eq. (156) has the structure analyzed for V = I: it propagates under the forward M, D, $\mathbf { F } _ { \mathbf { u } }$ . The physical constructions therefore apply unchanged to d:

• Linear V-reciprocal (F = VHu, with V-reciprocal M and D): d is obtained by a direct finiteamplitude run of the same hardware, driven by the V-weighted adjoint source and initialized by (157)– (158). As in the Euclidean case, V-reciprocal damping is permitted.

• Nonlinear gradient flow $( \mathbf { F } = \mathbf { V } E _ { \mathbf { u } } , \ \mathbf { D } = \mathbf { 0 } )$ : d is obtained as the infinitesimal response of the nudged time-reversed trajectory (Appendix B), now with the source and initial-condition nudges pre-multiplied by V. If instead $\mathbf { D } = \gamma \mathbf { M }$ , the exponential reweighting of Appendix C applies on top.

In both cases the measured physical field is $\mathbf { d } = \mathbf { V } \mathbf { a }$ . The Euclidean adjoint required by the gradient overlaps (8)–(13) is recovered by a single application of $\mathbf { V } ^ { - 1 }$

$$
\mathbf { a } ( t ) = \mathbf { V } ^ { - 1 } \mathbf { d } ( t ) ,\tag{159}
$$

after which, for instance,

$$
\frac { d \chi } { d \mathbf { p } _ { F } } = \int _ { 0 } ^ { T } \langle \mathbf { V } ^ { - 1 } \mathbf { d } ( T - t ) , \mathbf { F } _ { \mathbf { p } _ { F } } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] \rangle d t .\tag{160}
$$

This proves the claim of Sec. 5 and the second-order, V-reciprocal generalization of Theorem 6.1.

## D.4 Generalization to all other systems

The proof used a single structural fact: conjugation by V turns each transposed (Euclidean-adjoint) operator into the corresponding forward operator, $\mathbf { V } ( \mathbf { \cdot } ) ^ { T } \mathbf { V } ^ { - 1 } = \mathbf { \rho } ( \mathbf { \cdot } )$ . Nothing else about the second-order form entered. The same $\mathbf { d } = \mathbf { V } \mathbf { a }$ substitution therefore carries over to every other system treated in this article:

• First-order real systems are the special case $\mathbf { M } = \mathbf { 0 } , \mathbf { D } = \mathbf { I }$ of (156): the same calculation gives d˙ $+ \mathbf { F } _ { \mathbf { u } } [ \mathbf { u } ( T - t ) ] \mathbf { d } + \mathbf { V } \mathbb { P } \theta _ { \mathbf { u } } = \mathbf { 0 } { \mathrm { ~ w i t h ~ } } \mathbf { d } ( 0 ) = - \mathbf { V } \mathbb { P } \psi _ { \mathbf { u } }$

• Complex Schr¨odinger- and Helmholtz-type systems admit the analogous similarity transformation in their Hermitian inner product.

In every case the hardware propagates the V-weighted field and the Euclidean adjoint is recovered by applying $\mathbf { V } ^ { - 1 }$ before the gradient overlap. □

## E Adjoint state equation and gradient for first-order systems

We derive the first-order adjoint equation (34) and the associated gradient formula quoted in Sec. 7. The derivation follows Appendix A step for step: the Lagrangian is again a functional of independent arguments, and each of its partial derivatives supplies one ingredient of the method. Stationarity in the multipliers returns the forward problem, stationarity in the state defines the adjoint problem, and the remaining explicit parameter dependence is the gradient. Two things are simpler here. The first-order problem carries a single initial condition, hence a single initial-condition multiplier, and its constraint integral needs only one integration by parts. We reuse the diferentiability assumptions of Appendix A throughout.

## E.1 Constrained optimization problem

The forward dynamics is the first-order system (33),

$$
\begin{array} { r } { \dot { \mathbf { u } } ( t ) + \mathbf { F } \big [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t \big ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) = \mathbf { 0 } , \qquad t \in [ 0 , T ] , \qquad \mathbf { u } ( 0 ) = \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) , } \end{array}\tag{161}
$$

with $\mathbf { u } \in C ^ { 1 } ( [ 0 , T ] , V )$ . The cost retains a trajectory and a terminal part, the latter now depending on $\mathbf { u } ( T )$ alone, since the full state of a first-order system is the position itself,

$$
\chi [ \mathbf { u } ] = \int _ { 0 } ^ { T } \theta \left[ \mathbb { P } \mathbf { u } ( t ) \right] d t + \psi \left[ \mathbb { P } \mathbf { u } ( T ) \right] .\tag{162}
$$

The cost evaluated on the forward solution defines the reduced cost $\chi ( \mathbf { p } ) : = \chi [ \mathbf { u } ( \cdot ; \mathbf { p } ) ]$ , and the optimization problem is the following.

Problem E.1 (Constrained optimization of a first-order ODE). Find

$$
\operatorname* { m i n } _ { \mathbf { p } \in \mathbb { R } ^ { M } } \chi [ \mathbf { u } ] \qquad \mathrm { s u b j e c t ~ t o } \qquad \left\{ \begin{array} { l l } { \dot { \mathbf { u } } ( t ) + \mathbf { F } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) = \mathbf { 0 } , \quad t \in [ 0 , T ] , } \\ { \mathbf { u } ( 0 ) = \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) . } \end{array} \right.\tag{163}
$$

## E.2 The Lagrangian

We enforce the state equation with a multiplier path b $\in C ^ { 1 } ( [ 0 , T ] , V )$ and the initial condition with a multiplier vector $\lambda \in V$ , and define

$$
\mathcal { L } [ \mathbf { u } , \mathbf { b } , \boldsymbol { \lambda } , \mathbf { p } ] : = \int _ { 0 } ^ { T } \langle \mathbf { b } ( t ) , \dot { \mathbf { u } } ( t ) + \mathbf { F } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) \rangle d t + \left. \boldsymbol { \lambda } , \mathbf { u } ( 0 ) - \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) \right. + \chi [ \mathbf { u } ] .\tag{164}
$$

As in Appendix A, the four arguments of $\mathcal { L }$ are independent: u is an arbitrary path, not assumed to solve the state equation, and p enters only through the explicit dependencies $\mathbf { F } [ \cdot , \mathbf { p } _ { F } , \cdot ] , \mathbf { f } ( \cdot , \mathbf { p } _ { f } ) , \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } )$ We may therefore diferentiate with respect to each argument separately, and the derivation consists of computing the four partial derivatives $\mathcal { L } _ { \bf b } , \mathcal { L } _ { \lambda } , \mathcal { L } _ { \bf u } , \mathcal { L } _ { \bf p }$ in turn.

## E.3 Stationarity with respect to the multipliers: the forward problem

The Lagrangian is afine in b and λ, so both derivatives can be read of directly. Identifying them with elements of the respective spaces through the inner product,

$$
\begin{array} { r } { \mathcal { L } _ { \mathbf { b } } ( t ) = \dot { \mathbf { u } } ( t ) + \mathbf { F } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) , \qquad \mathcal { L } _ { \lambda } = \mathbf { u } ( 0 ) - \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) , } \end{array}\tag{165}
$$

so $\mathcal { L } _ { \mathbf { b } } = 0$ holds if and only if u satisfies the state equation (161) on $[ 0 , T ]$ , and $\mathcal { L } _ { \lambda } = 0$ if and only if u satisfies the initial condition. Stationarity with respect to a multiplier again recovers the constraint that the multiplier enforces. Together the two conditions state that $\mathbf { u } = \mathbf { u } ( \cdot ; \mathbf { p } )$ is the forward solution.

## E.4 Stationarity turns the partial derivative into the gradient

Evaluated at the forward solution the constraint terms in (164) vanish, so $\chi ( \mathbf { p } ) = \mathcal { L } [ \mathbf { u } ( \cdot ; \mathbf { p } ) , \mathbf { b } , \lambda , \mathbf { p } ]$ for every b and λ. Letting the multipliers depend diferentiably on p as well and diferentiating, one term per argument of ${ \mathcal { L } } ,$ , gives

$$
\frac { d \chi } { d \mathbf { p } } = \mathcal { L } _ { \mathbf { u } } \frac { d \mathbf { u } } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { b } } \frac { d \mathbf { b } } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { \lambda } } \frac { d \lambda } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { p } } .\tag{166}
$$

The two multiplier terms vanish by Sec. E.3, whatever the sensitivities $d \mathbf { b } / d \mathbf { p }$ and $d \lambda / d \mathbf { p }$ may be. The expensive sensitivity $d \mathbf { u } / d \mathbf { p }$ enters only multiplied by ${ \mathcal { L } } _ { \mathbf { u } } .$ , and b, λ are still free. Choosing them such that $\mathcal { L } _ { \bf u } = 0$ therefore leaves the identity

$$
\frac { d \chi } { d \mathbf { p } } = \mathcal { L } _ { \mathbf { p } } ,\tag{167}
$$

in which only the cheap explicit parameter dependence of F, f and $\mathbf { u } _ { 0 }$ appears.

## E.5 Stationarity with respect to the state: the adjoint problem

Let h $\in C ^ { 1 } ( [ 0 , T ] , V )$ be an arbitrary test path with no constraints at either endpoint. Varying $\mathbf { u } \to \mathbf { u } + \varepsilon \mathbf { h }$ and noting that the drive f has no u-dependence and does not contribute,

$$
\mathcal { L } _ { \mathbf { u } } \mathbf { h } = \int _ { 0 } ^ { T } \langle \mathbf { b } , \dot { \mathbf { h } } + \mathbf { F } _ { \mathbf { u } } \mathbf { h } \rangle d t + \big \langle \lambda , \mathbf { h } ( 0 ) \big \rangle + \int _ { 0 } ^ { T } \langle \mathbb { P } \theta _ { \mathbf { u } } , \mathbf { h } \rangle d t + \big \langle \mathbb { P } \psi _ { \mathbf { u } } , \mathbf { h } ( T ) \big \rangle .\tag{168}
$$

Where the second-order case needed two integrations by parts, a single one here moves the derivative of h,

$$
\int _ { 0 } ^ { T } \langle \mathbf { b } , { \dot { \mathbf { h } } } \rangle d t = - \int _ { 0 } ^ { T } \langle { \dot { \mathbf { b } } } , \mathbf { h } \rangle d t + \langle \mathbf { b } ( T ) , \mathbf { h } ( T ) \rangle - \langle \mathbf { b } ( 0 ) , \mathbf { h } ( 0 ) \rangle ,\tag{169}
$$

and moving $\mathbf { F } _ { \mathbf { u } }$ across the inner product gives $\langle \mathbf { b } , \mathbf { F } _ { \mathbf { u } } \mathbf { h } \rangle = \langle \mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { b } , \mathbf { h } \rangle$ . Grouping by where h is evaluated,

$$
\mathcal { L } _ { \mathbf { u } } \mathbf { h } = \int _ { 0 } ^ { T } \langle - \dot { \mathbf { b } } + \mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { b } + \mathbb { P } \theta _ { \mathbf { u } } , \mathbf { h } \rangle d t + \langle \mathbf { b } ( T ) + \mathbb { P } \psi _ { \mathbf { u } } , \mathbf { h } ( T ) \rangle + \langle - \mathbf { b } ( 0 ) + \lambda , \mathbf { h } ( 0 ) \rangle .\tag{170}
$$

Since h is arbitrary and unconstrained at both endpoints, $\mathcal { L } _ { \bf u } = 0$ requires each inner product to vanish separately. The bulk term and the term at $t = T$ involve only b. Together they form a terminal-value problem that determines b uniquely and defines the adjoint state.

Definition E.2 (Adjoint state for the first-order problem). The adjoint state $\mathbf { b } ( t ) \in V$ associated with Problem E.1 is the solution of the adjoint ODE

$$
- \dot { \mathbf { b } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } \big [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t \big ] \mathbf { b } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( t ) \big ] = \mathbf { 0 } , \qquad t \in [ 0 , T ] ,\tag{171}
$$

a first-order ODE requiring one condition, with the terminal condition

$$
\mathbf { b } ( T ) = - \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T ) \big ] .\tag{172}
$$

We solve the adjoint ODE (171) backward from T to 0 using this terminal condition. This produces ${ \bf b } ( 0 )$ which in turn determines the multiplier via the term at $t = 0$ in (170),

$$
\begin{array} { r } { \lambda = \mathbf { b } ( 0 ) . } \end{array}\tag{173}
$$

The adjoint ODE is linear in b, its coeficients depend continuously on p, and (173) is explicit. Hence $( \mathbf { b } , \pmb { \lambda } )$ is the unique solution of $\mathcal { L } _ { \bf u } = 0$ and depends smoothly on p, as required in Sec. E.4.

## E.6 The gradient

Both remaining partial derivatives are now in place, and evaluating ${ \mathcal { L } } _ { \mathbf { p } }$ gives the gradient.

Theorem E.3 (Adjoint gradient for the first-order problem). Let $\mathbf { u } = \mathbf { u } ( \cdot ; \mathbf { p } )$ solve the forward problem of Problem E.1 and let b be the adjoint state of Definition E.2. Then the gradient of the reduced cost with respect to p is

$$
\frac { d \boldsymbol { \chi } } { d \mathbf { p } } = \int _ { 0 } ^ { T } \langle \mathbf { b } ( t ) , \mathbf { F } _ { \mathbf { p } _ { F } } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } _ { \mathbf { p } _ { f } } ( t , \mathbf { p } _ { f } ) \rangle d t - \langle \mathbf { b } ( 0 ) , \frac { d \mathbf { u } _ { 0 } } { d \mathbf { p } _ { u _ { 0 } } } \rangle .\tag{174}
$$

Proof. With $( \mathbf { b } , \pmb { \lambda } )$ chosen as in Definition E.2 and Eq. (173), we have $\mathcal { L } _ { \bf u } = 0$ , and the key identity (167) gives $d \chi / d \mathbf { p } = \mathcal { L } _ { \mathbf { p } }$ . The explicit p-dependence of $\mathcal { L }$ enters through ${ \bf F } [ { \bf u } , { \bf p } _ { F } , t ]$ and $\mathbf { f } \left( t , \mathbf { p } _ { f } \right)$ in the constraint integral and through ${ \bf u } _ { 0 } ( { \bf p } _ { u _ { 0 } } )$ in the initial-condition term. The cost $\chi$ has none. Diferentiating each at fixed u, b, λ and substituting $\lambda = \mathbf { b } ( 0 )$ gives (174). □

As in the second-order case, every quantity on the right-hand side is known: ${ \bf \delta u } ( t )$ from the forward solve, b(t) from the backward solve, and the parameter sensitivities from the model. The cost is one forward solve plus one backward solve, independent of the number of parameters M.

Direct optimization of the initial condition. Suppose a subvector of p encodes the initial state itself, i.e. $\mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) = \mathbf { p } _ { u _ { 0 } }$ with $\mathbf { p } _ { u _ { 0 } } \in V$ . Then $d { \bf u } _ { 0 } / d { \bf p } _ { u _ { 0 } } = I _ { V }$ , and the $\mathbf { u } _ { 0 } .$ -block of Eq. (174) identifies the gradient directly:

$$
\frac { d \chi } { d { \bf u } _ { 0 } } = - { \bf b } ( 0 ) ,\tag{175}
$$

that is, minus the initial-condition multiplier, which is the M = 0, D = I case of Eq. (121).

## E.7 Integrating the adjoint forward in time

The adjoint problem of Definition E.2 is posed backward in time, while physical hardware runs forward. $\mathrm { A s }$ in Sec. A.10, the change of variable $t \mapsto T - t$ removes the mismatch. Define the forward-time adjoint field

$$
\mathbf { a } ( t ) : = \mathbf { b } ( T - t ) , \qquad \dot { \mathbf { a } } ( t ) = - \dot { \mathbf { b } } ( T - t ) .\tag{176}
$$

Evaluating the adjoint ODE (171) at time $T - t$ and substituting (176) gives the forward-time adjoint equation

$$
\dot { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } \big [ \mathbf { u } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \mathbf { a } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) \big ] = \mathbf { 0 } , \qquad t \in [ 0 , T ] .\tag{177}
$$

Under the reversal the derivative term changes sign, $- \dot { \mathbf { b } }  + \dot { \mathbf { a } }$ . This is the $\mathbf { M } = \mathbf { 0 } , \mathbf { D } = \mathbf { I }$ instance of the $\mathrm { { H i p } } - \mathbf { D } ^ { T }  + \mathbf { D } ^ { T }$ of Sec. A.10, and it gives Eq. (177) the same sign pattern as the state equation (161). Apart from this, the reversal only means that the linearization and the source are read along the reversed trajectory $\mathbf { u } ( T - t )$ , with the explicit time dependence of F replayed as $t \mapsto T - t$ . The terminal condition (172) sits at the reversed time $t = 0$ and becomes a genuine initial condition,

$$
\mathbf { a } ( 0 ) = - \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T ) \big ] .\tag{178}
$$

Likewise ${ \bf b } ( 0 ) = { \bf a } ( T )$ , so substituting ${ \mathbf { b } ( t ) = \mathbf { a } ( T - t ) }$ throughout the gradient (174) gives

$$
\frac { d \boldsymbol { \chi } } { d \mathbf { p } } = \int _ { 0 } ^ { T } \langle \mathbf { a } ( T - t ) , \mathbf { F } _ { \mathbf { p } _ { F } } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } _ { \mathbf { p } _ { f } } ( t , \mathbf { p } _ { f } ) \rangle d t - \langle \mathbf { a } ( T ) , \frac { d \mathbf { u } _ { 0 } } { d \mathbf { p } _ { u _ { 0 } } } \rangle .\tag{179}
$$

Eqs. (177) and (178) are the adjoint equation (34) with the source and initial condition (35) quoted in the main text. They are likewise the first-order analogues of the second-order results of Appendix A, and follow formally from them by the overdamped reduction $\mathbf { M } = \mathbf { 0 } , \mathbf { D } = \mathbf { I }$

## F Wirtinger derivation of the Schr¨odinger adjoint

We derive the Wirtinger adjoint state, gradient, and forward-time conjugated field used in Sec. 8 from an augmented Lagrangian in which u and u are treated as independent variables [100]. The derivation parallels the real first-order case of Appendix E. The only new ingredient is Wirtinger calculus [84]. The use of Wirtinger variables is natural here because real-valued objectives of complex fields are generically nonholomorphic: the variation of the cost depends on both δu and δu. This viewpoint is standard in complex optimization, where one treats a complex variable and its conjugate as formally independent variables to obtain stationary conditions and descent directions [61–63]. For time-dependent complex dynamics, the same variational structure appears in quantum optimal control, where the Schr¨odinger equation is imposed by a complex Lagrange multiplier and stationarity of the augmented functional yields a terminal-value adjoint equation [59]. A rigorous PDE-constrained optimization formulation in complex Banach spaces, including semigroup Schr¨odinger equations and first- and second-order optimality conditions, was developed in [60]. Our derivation follows this complex Lagrangian/Wirtinger viewpoint, but is specialized to the adjoint field and forward-adjoint overlap gradients needed for the physical backpropagation constructions in Sec. 8.

## F.1 Problem and conventions

The state is the Schr¨odinger-type system (39),

$$
i \dot { \mathbf { u } } ( t ) + \mathbf { F } \big [ \mathbf { u } ( t ) , \overline { { \mathbf { u } } } ( t ) , \mathbf { p } _ { F } , t \big ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) = \mathbf { 0 } , \qquad \mathbf { u } ( 0 ) = \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) ,\tag{180}
$$

with $\mathbf { u } \in \mathbb { C } ^ { N }$ , real parameters p, and a real-valued cost

$$
\chi [ \mathbf { u } ] = \int _ { 0 } ^ { T } \theta \left[ \mathbb { P } \mathbf { u } ( t ) , \mathbb { P } \overline { { \mathbf { u } } } ( t ) \right] d t + \psi \left[ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \overline { { \mathbf { u } } } ( T ) \right] .\tag{181}
$$

We use the Hermitian inner product $\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathbb { C } } = \overline { { \mathbf { x } } } ^ { T } \mathbf { y }$ , treat u and u as independent, and write the two Wirtinger Jacobians $\mathbf { F _ { u } }$ and $\mathbf { F } _ { \overline { { \mathbf { u } } } } .$ Since θ and ψ are real, $\theta _ { \mathbf { u } } = \overline { { \theta _ { \overline { { { \mathbf { u } } } } } } }$ and $\psi _ { \mathbf { u } } = { \overline { { \psi _ { \mathbf { u } } } } } .$ and, for an arbitrary variation h of u, the first variation of the (real) cost is

$$
\delta \chi = 2 \operatorname { R e } \Big [ \int _ { 0 } ^ { T } \Big \langle \mathbb { P } \theta _ { \mathbf { \overline { { u } } } } , \mathbf { h } \Big \rangle _ { \mathbb { C } } d t + \Big \langle \mathbb { P } \psi _ { \mathbf { \overline { { u } } } } , \mathbf { h } ( T ) \Big \rangle _ { \mathbb { C } } \Big ] .\tag{182}
$$

Problem F.1 (Constrained optimization of a Schr¨odinger-type system). Find

$$
\operatorname* { m i n } _ { \mathbf { p } } \ \chi [ \mathbf { u } ] \qquad \mathrm { s u b j e c t ~ t o } \qquad \left\{ { \begin{array} { l } { i \dot { \mathbf { u } } ( t ) + \mathbf { F } [ \mathbf { u } ( t ) , \mathbf { \bar { u } } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) = \mathbf { 0 } , \quad t \in [ 0 , T ] , } \\ { \mathbf { u } ( 0 ) = \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) . } \end{array} } \right.\tag{183}
$$

## F.2 Augmented Lagrangian and the Wirtinger adjoint

Introduce a complex adjoint field $\mathbf { b } \in C ^ { 1 } ( [ 0 , T ] , \mathbb { C } ^ { N } )$ and an initial-condition multiplier $\pmb { \lambda } \in \mathbb { C } ^ { N }$ , and pair the complex constraint against them through 2 Re so that the Lagrangian is real,

$$
\mathcal { L } = \chi + 2 \operatorname { R e } \int _ { 0 } ^ { T } \langle \mathbf { b } ( t ) , i \mathbf { i } + \mathbf { F } - \mathbf { f } \rangle _ { \mathbb { C } } d t + 2 \operatorname { R e } \langle \lambda , \mathbf { u } ( 0 ) - \mathbf { u } _ { 0 } \rangle _ { \mathbb { C } } .\tag{184}
$$

The Lagrangian is afine in b and λ, which enter only through the real pairing, so those two derivatives can be read of as the residuals they multiply,

$$
\begin{array} { r } { \mathcal { L } _ { \mathbf { b } } ( t ) = i \dot { \mathbf { u } } ( t ) + \mathbf { F } \big [ \mathbf { u } ( t ) , \overline { { \mathbf { u } } } ( t ) , \mathbf { p } _ { F } , t \big ] - \mathbf { f } ( t , \mathbf { p } _ { f } ) , \qquad \mathcal { L } _ { \lambda } = \mathbf { u } ( 0 ) - \mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } ) . } \end{array}\tag{185}
$$

A real pairing $2 \operatorname { R e } \langle \cdot , \mathbf { X } \rangle _ { \mathbb { C } }$ that vanishes for every complex direction forces $\mathbf { X } = \mathbf { 0 } .$ , since a direction and its multiple by i recover the real and the imaginary part separately. Hence $\mathcal { L } _ { \mathbf { b } } = 0$ holds if and only if u satisfies the state equation (180) on $[ 0 , T ]$ , and $\mathcal { L } _ { \lambda } = 0$ if and only if it satisfies the initial condition. Stationarity in the multipliers returns the forward problem, as in the real case.

Evaluated on that forward solution the two constraint pairings vanish and ${ \mathcal { L } } = \chi ( \mathbf { p } )$ . Letting the multipliers depend diferentiably on p as well and diferentiating, with one term per argument of $\mathcal { L }$ and each product denoting the real pairing of a derivative with a sensitivity,

$$
\frac { d \chi } { d \mathbf { p } } = \mathcal { L } _ { \mathbf { u } } \frac { d \mathbf { u } } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { b } } \frac { d \mathbf { b } } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { \lambda } } \frac { d \lambda } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { p } } .\tag{186}
$$

Here ${ \mathcal { L } } _ { \mathbf { u } }$ is the state derivative, the object whose real pairing with a variation h gives the first variation computed below. Because that pairing is taken under 2 Re, it already carries the u and the u contributions together, and no separate condition in u is needed. The two multiplier terms vanish by the previous

paragraph, whatever the sensitivities $d \mathbf { b } / d \mathbf { p }$ and $d \lambda / d \mathbf { p }$ may be, and the expensive $d \mathbf { u } / d \mathbf { p }$ enters only against ${ \mathcal { L } } _ { \mathbf { u } }$ . Choosing b and λ so that $\mathcal { L } _ { \bf u } = 0$ therefore leaves the key identity

$$
\frac { d \chi } { d { \bf p } } = \mathcal { L } _ { \bf p } .\tag{187}
$$

We make L stationary by varying $\mathbf { u }  \mathbf { u } + \varepsilon \mathbf { h }$ (with the accompanying conjugate variation h), the first variation $\delta \mathcal { L }$ being the coeficient of ε. The linearized residual is ih<sup>˙</sup> $+ \mathbf { F } _ { \mathbf { u } } \mathbf { h } + \mathbf { F } _ { \overline { { \mathbf { u } } } } \overline { { \mathbf { h } } }$ , so together with the cost variation (182),

$$
\delta \mathcal { L } = 2 \operatorname { R e } \int _ { 0 } ^ { T } \langle \mathbf { b } , i \dot { \mathbf { h } } + \mathbf { F } _ { \mathbf { u } } \mathbf { h } + \mathbf { F } _ { \mathbf { \overline { { u } } } } \mathbf { \overline { { h } } } \rangle _ { \mathbb { C } } d t + 2 \operatorname { R e } \langle \pmb { \lambda } , \mathbf { h } ( 0 ) \rangle _ { \mathbb { C } } + \delta \chi .\tag{188}
$$

To read of the stationarity conditions we move every term onto h. The $\overline { { \mathbf { h } } }$ contribution is folded onto h under the real part,

$$
2 \mathrm { R e } \big \langle \mathbf { b } , \mathbf { B } \overline { { \mathbf { h } } } \big \rangle _ { \mathbb { C } } = 2 \mathrm { R e } \big \langle \mathbf { B } ^ { T } \overline { { \mathbf { b } } } , \mathbf { h } \big \rangle _ { \mathbb { C } } ,\tag{189}
$$

applied with $\mathbf { B } = \mathbf { F } _ { \overline { { \mathbf { u } } } } .$ , while the holomorphic term moves through the Hermitian adjoint, $2 \mathrm { R e } \langle \mathbf { b } , \mathbf { F } _ { \mathbf { u } } \mathbf { h } \rangle _ { \mathbb { C } } =$ $2 \mathrm { R e } \langle \overline { { \mathbf { F } _ { \mathbf { u } } } } ^ { T } \mathbf { b } , \mathbf { h } \rangle _ { \mathbb { C } }$ . The only time derivative is integrated by parts. Writing $\langle \mathbf { b } , i \dot { \mathbf { h } } \rangle _ { \mathbb { C } } = \langle - i \mathbf { b } , \dot { \mathbf { h } } \rangle _ { \mathbb { C } } .$

$$
2 \operatorname { R e } \int _ { 0 } ^ { T } \langle \mathbf { b } , i \mathbf { \dot { h } } \rangle _ { \mathbb { C } } d t = 2 \operatorname { R e } \int _ { 0 } ^ { T } \langle i \mathbf { \dot { b } } , \mathbf { h } \rangle _ { \mathbb { C } } d t + 2 \operatorname { R e } \Big [ \langle - i \mathbf { b } ( T ) , \mathbf { h } ( T ) \rangle _ { \mathbb { C } } - \langle - i \mathbf { b } ( 0 ) , \mathbf { h } ( 0 ) \rangle _ { \mathbb { C } } \Big ] .\tag{190}
$$

Every surviving factor of i originates from this step. Substituting (182)–(190) into (188) and grouping by where h is evaluated,

$$
\begin{array} { r l } & { \delta \mathcal { L } = 2 \mathrm { R e } \int _ { 0 } ^ { T } \langle i \dot { \mathbf { b } } + \mathbf { \overline { { F } } _ { u } } ^ { T } \mathbf { b } + \mathbf { F } _ { \mathbf { u } } ^ { T } \mathbf { \overline { { b } } } + \mathbb { F } \theta _ { \mathbf { \overline { { u } } } } , \mathbf { h } \rangle _ { \mathbb { C } } d t } \\ & { \qquad +  2 \mathrm { R e } \langle - i \mathbf { b } ( T ) + \mathbb { P } \psi _ { \mathbf { \overline { { u } } } } , \mathbf { h } ( T ) \rangle _ { \mathbb { C } } + 2 \mathrm { R e } \langle i \mathbf { b } ( 0 ) + \lambda , \mathbf { h } ( 0 ) \rangle _ { \mathbb { C } } . } \end{array}\tag{191}
$$

Since h is arbitrary and unconstrained at both endpoints, $\mathcal { L } _ { \bf u } = 0$ requires each bracket to vanish separately.   
The $t = 0$ term fixes the multiplier $\pmb { \lambda } = - i \mathbf { b } ( 0 )$ , while the bulk and $t = T$ terms define the adjoint state.

Definition F.2 (Wirtinger adjoint state). The Wirtinger adjoint state ${ \mathbf b } ( t ) \in { \mathbb C } ^ { N }$ associated with Problem F.1 is the solution of the backward-time ODE

$$
\begin{array} { r } { i \dot { \mathbf { b } } ( t ) + \overline { { \mathbf { F } _ { \mathbf { u } } } } ^ { T } \mathbf { b } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } \overline { { \mathbf { b } } } ( t ) + \mathbb { P } \theta _ { \overline { { \mathbf { u } } } } \big [ \mathbb { P } \mathbf { u } ( t ) , \mathbb { P } \overline { { \mathbf { u } } } ( t ) \big ] = \mathbf { 0 } , } \end{array}\tag{192}
$$

with the Wirtinger Jacobians evaluated at $( { \bf u } ( t ) , \overline { { { \bf u } } } ( t ) )$ , and the terminal condition

$$
\mathbf { b } ( T ) = - i \mathbb { P } \psi _ { \overline { { \mathbf { u } } } } \big [ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \overline { { \mathbf { u } } } ( T ) \big ] .\tag{193}
$$

Theorem F.3 (Wirtinger adjoint gradient). Let u solve Problem F.1 and b be the adjoint state of Definition F.2. For real parameters p, the gradient of the cost is

$$
\frac { d \boldsymbol { \chi } } { d \mathbf { p } } = 2 \mathrm { R e } \int _ { 0 } ^ { T } \left. \mathbf { b } ( t ) , \mathbf { F } _ { \mathbf { p } _ { F } } [ \mathbf { u } ( t ) , \mathbf { \overline { { u } } } ( t ) , \mathbf { p } _ { F } , t ] - \mathbf { f } _ { \mathbf { p } _ { f } } ( t , \mathbf { p } _ { f } ) \right. _ { \mathbb { C } } d t + 2 \mathrm { R e } \left. i \mathbf { b } ( 0 ) , \mathbf { u } _ { 0 , \mathbf { p } _ { u _ { 0 } } } \right. _ { \mathbb { C } } .\tag{194}
$$

Proof. With b and λ chosen as in Definition F.2 we have $\mathcal { L } _ { \bf u } = 0$ , and the key identity (187) gives $d \chi / d \mathbf { p } = \mathcal { L } _ { \mathbf { p } }$ . The explicit parameter dependence of (184) enters only through $\mathbf { F } ( \mathbf { p } _ { F } )$ and $\mathbf { f } \left( \mathbf { p } _ { f } \right)$ in the constraint pairing and through $\mathbf { u } _ { 0 } ( \mathbf { p } _ { u _ { 0 } } )$ in the initial-condition pairing, giving $\begin{array} { r } { \mathcal { L } _ { \mathbf { p } } = 2 \mathrm { R e } \int _ { 0 } ^ { T } \langle \mathbf { b } , \mathbf { F } _ { \mathbf { p } _ { F } } \ - } \end{array}$ $\mathbf { f } _ { \mathbf { p } _ { f } } \rangle _ { \mathbb { C } } d t - 2 \operatorname { R e } \langle \lambda , \mathbf { u } _ { 0 , \mathbf { p } _ { u _ { 0 } } } \rangle _ { \mathbb { C } }$ . Substituting $\pmb { \lambda } = - i \mathbf { b } ( 0 )$ yields (194). The external drive enters the residual $\mathrm { a s - } \mathbf { f } .$ , so its gradient carries the opposite sign to the internal force, and the initial-condition term carries the explicit factor i inherited from $\pmb { \lambda } = - i \mathbf { b } ( 0 )$ □

## F.3 Forward-time conjugated adjoint

The adjoint of Definition F.2 runs backward. For a physical realization we pass to the conjugated timereversed field $\mathbf { c } ( t ) : = \overline { { \mathbf { b } ( T - t ) } }$ . Evaluating (192) at $T - t .$ , using $\dot { \mathbf { b } } ( T - t ) = - \overline { { \dot { \mathbf { c } } ( t ) } }$ , and complex conjugating the entire equation gives the forward-time equation (40),

$$
\begin{array} { r l } & { i \dot { \mathbf { c } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } \big [ \mathbf { u } ( T - t ) , \mathbf { \overline { { u } } } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] \mathbf { c } ( t ) } \\ & { \qquad + \frac { 1 } { \mathbf { F } _ { \overline { { \mathbf { u } } } } \big [ \mathbf { u } ( T - t ) , \mathbf { \overline { { u } } } ( T - t ) , \mathbf { p } _ { F } , T - t \big ] ^ { T } } \bar { \mathbf { c } } ( t ) + \mathbb { P } \theta _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ( T - t ) , \mathbb { P } \mathbf { \overline { { u } } } ( T - t ) \big ] = \mathbf { 0 } , } \end{array}\tag{195}
$$

where we used $\overline { { \theta _ { \overline { { { \bf u } } } } } } = \theta _ { { \bf u } } .$ , with initial condition

$$
\mathbf { c } ( 0 ) = \overline { { \mathbf { b } ( T ) } } = i \mathbb { P } \psi _ { \mathbf { u } } \left[ \mathbb { P } \mathbf { u } ( T ) , \mathbb { P } \overline { { \mathbf { u } } } ( T ) \right] .\tag{196}
$$

Conjugating term by term flips the signs in front of the Wirtinger Jacobians, which is why the conjugated field c, rather than b, can be propagated on the same hardware. The raw adjoint is recovered by $\mathbf { b } ( t ) =$ $\overline { { \mathbf { c } ( T - t ) } }$ . Substituting into (194) gives the gradient overlaps quoted in Sec. 8 in terms of the measured field,

$$
\frac { d \chi } { d \mathbf { p } _ { F } } = 2 \operatorname { R e } \int _ { 0 } ^ { T } \langle \overline { { \mathbf { c } ( T - t ) } } , \mathbf { F } _ { \mathbf { p } _ { F } } [ \mathbf { u } ( t ) , \mathbf { \overline { { u } } } ( t ) , \mathbf { p } _ { F } , t ] \rangle _ { \mathbb { C } } d t ,\tag{197}
$$

$$
\frac { d \chi } { d \mathbf { p } _ { f } } = - 2 \operatorname { R e } \int _ { 0 } ^ { T } \langle \overline { { \mathbf { c } ( T - t ) } } , \mathbf { f } _ { \mathbf { p } _ { f } } ( t , \mathbf { p } _ { f } ) \rangle _ { \mathbb { C } } d t ,\tag{198}
$$

$$
\frac { d \chi } { d { \bf p } _ { u _ { 0 } } } = - 2 \mathrm { R e } \langle i \overline { { { { \bf c } ( T ) } } } , { \bf u } _ { 0 , { \bf p } _ { u _ { 0 } } } \rangle _ { \mathbb { C } } .\tag{199}
$$

## G Free-space optics

Free-space difractive optics. In the scalar, monochromatic, paraxial, and thin-element approximation, a free-space difractive optical system is a linear Schr¨odinger-type system of the form considered in Sec. 8, with the propagation coordinate z playing the role of time [101]. Starting from the scalar Helmholtz equation and writing the field as a slowly varying envelope,

$$
\left( \nabla _ { \perp } ^ { 2 } + \partial _ { z } ^ { 2 } + k _ { 0 } ^ { 2 } n ^ { 2 } ( \mathbf { r } _ { \perp } , z ) \right) E ( \mathbf { r } _ { \perp } , z ) = 0 , \qquad E ( \mathbf { r } _ { \perp } , z ) = u ( \mathbf { r } _ { \perp } , z ) e ^ { i k z } , \qquad k = k _ { 0 } n _ { 0 } ,\tag{200}
$$

the paraxial approximation neglects $\partial _ { z } ^ { 2 } u$ relative to $2 i k \partial _ { z } u$ , giving

$$
i \partial _ { z } u + \left[ \frac { 1 } { 2 k } \nabla _ { \perp } ^ { 2 } + k _ { 0 } \Delta n ( \mathbf { r } _ { \perp } , z ) \right] u = 0 .\tag{201}
$$

An ideal SLM plane at $z = z _ { j }$ is modeled as a thin local transmission

$$
u ( z _ { j } ^ { + } ) = m _ { j } ( \mathbf { r } _ { \perp } ) u ( z _ { j } ^ { - } ) , \qquad m _ { j } ( \mathbf { r } _ { \perp } ) = \rho _ { j } ( \mathbf { r } _ { \perp } ) e ^ { i \phi _ { j } ( \mathbf { r } _ { \perp } ) } = e ^ { i \alpha _ { j } ( \mathbf { r } _ { \perp } ) } ,\tag{202}
$$

or equivalently as a delta-function potential, the form underlying split-step beam propagation [102],

$$
i \partial _ { z } u + \left[ \frac { 1 } { 2 k } \nabla _ { \perp } ^ { 2 } + \sum _ { j } \alpha _ { j } ( { \bf r } _ { \perp } ) \delta ( z - z _ { j } ) \right] u = 0 .\tag{203}
$$

After transverse discretization this has the form treated in Sec. 8,

$$
i \dot { \mathbf { u } } ( z ) + \mathbf { F } ( \mathbf { u } , \mathbf { \overline { { u } } } , \mathbf { p } _ { F } ) = 0 , \qquad \mathbf { F } ( \mathbf { u } , \mathbf { \overline { { u } } } , \mathbf { p } _ { F } ) = \mathbf { K } ( z ; \mathbf { p } _ { K } ) \mathbf { u } ,\tag{204}
$$

with

$$
\mathbf { K } ( z ; \mathbf { p } _ { K } ) = { \frac { 1 } { 2 k } } \mathbf { L } _ { \perp } + \sum _ { j } \mathbf { A } _ { j } ( \mathbf { p } _ { j } ) \delta ( z - z _ { j } ) , \qquad \mathbf { D } _ { j } = e ^ { i \mathbf { A } _ { j } } = \mathrm { d i a g } ( m _ { j 1 } , \ldots , m _ { j N } ) .\tag{205}
$$

The internal force (204) is linear and holomorphic: it carries no u dependence, so $\mathbf { F } _ { \overline { { \mathbf { u } } } } = \mathbf { 0 }$ and the second Wirtinger Jacobian drops out of every formula in Appendix F. Equivalently, the input–output map is the usual free-space/SLM cascade [11]

$$
\mathbf { U } = \mathbf { P } _ { m } \mathbf { D } _ { m } \mathbf { P } _ { m - 1 } \mathbf { D } _ { m - 1 } \cdot \cdot \cdot \mathbf { P } _ { 1 } \mathbf { D } _ { 1 } \mathbf { P } _ { 0 } , \qquad \mathbf { P } _ { \ell } = \exp \left( i \frac { \Delta z _ { \ell } } { 2 k } \mathbf { L } _ { \perp } \right) .\tag{206}
$$

For a reciprocal scalar discretization,

$$
\mathbf { L } _ { \perp } ^ { T } = \mathbf { L } _ { \perp } , \qquad \mathbf { A } _ { j } ^ { T } = \mathbf { A } _ { j } , \qquad \mathbf { D } _ { j } ^ { T } = \mathbf { D } _ { j } .\tag{207}
$$

Reciprocity therefore requires complex symmetry, not Hermiticity or unitarity. In particular, lossy or linear-gain pixels with $| m _ { j i } | \neq 1$ are admissible even though generally

$$
\overline { { \mathbf { D } } } _ { j } ^ { T } \mathbf { D } _ { j } \neq \mathbf { I } .\tag{208}
$$

Accordingly, the physical adjoint experiment is the c-adjoint run of Appendix F, not a direct integration of the backward Wirtinger adjoint b. In the linear reciprocal case the c-field obeys

$$
i \dot { \mathbf { c } } ( z ) + \mathbf { K } ^ { T } ( L - z ; \mathbf { p } _ { K } ) \mathbf { c } ( z ) + \mathbb { P } \theta _ { \mathbf { u } } [ \mathbb { P } \mathbf { u } ( L - z ) , \mathbb { P } \overline { { \mathbf { u } } } ( L - z ) ] = 0 ,\tag{209}
$$

with initial condition

$$
\mathbf { c } ( 0 ) = i \mathbb { P } \psi _ { \mathbf { u } } [ \mathbb { P } \mathbf { u } ( L ) , \mathbb { P } \overline { { \mathbf { u } } } ( L ) ] .\tag{210}
$$

Since ${ \mathbf { K } } ^ { T } = { \mathbf { K } }$ , this is implemented by the same reciprocal optical system traversed in the reverse direction, with the adjoint source and initial condition applied at the output side. Note that ${ \bf K } ^ { T } ( L - z )$ places the modulator planes at $L - z _ { j } ,$ that is, in reversed element order, which is what reverse traversal physically realizes. The backward adjoint is then recovered as

$$
\mathbf { b } ( z ) = { \overline { { \mathbf { c } ( L - z ) } } } .\tag{211}
$$

For parameters that enter only through the optical operator, the gradient of Theorem F.3 becomes

$$
\frac { d \chi } { d p } = 2 \operatorname { R e } \int _ { 0 } ^ { L } \left. \overline { { \mathbf { c } ( L - z ) } } , \mathbf { K } _ { p } ( z ; \mathbf { p } _ { K } ) \mathbf { u } ( z ) \right. _ { \mathbb { C } } d z .\tag{212}
$$

Pixel gradients. Each SLM pixel carries two trainable parameters, a phase and a log-amplitude,

$$
m _ { j i } = e ^ { \gamma _ { j i } + i \phi _ { j i } } = \rho _ { j i } e ^ { i \phi _ { j i } } , \qquad \rho _ { j i } = e ^ { \gamma _ { j i } } , \qquad \alpha _ { j i } = \phi _ { j i } - i \gamma _ { j i } ,\tag{213}
$$

so that $\gamma _ { j i } < 0$ is an attenuating pixel, $\gamma _ { j i } > 0$ a linear-gain pixel, and a phase-only device is the special case $\gamma _ { j i } \equiv 0$ . Both gradients are obtained from the same pair of runs. We use the cascade to derive the gradient because it exhibits the pixel dependence directly [21]. Letting $\mathbf { E } _ { i }$ be diagonal with a single nonzero entry at pixel i,

$$
\frac { \partial { \bf U } } { \partial \phi _ { j i } } = { \bf P } _ { m } \cdot \cdot \cdot { \bf P } _ { j } \left( i m _ { j i } { \bf E } _ { i } \right) { \bf P } _ { j - 1 } \cdot \cdot \cdot { \bf P } _ { 0 } ,\tag{214}
$$

and since $\mathbf { E } _ { i } \mathbf { D } _ { j } = \mathbf { D } _ { j } \mathbf { E } _ { i }$ the phase and amplitude gradients are the two parts of one complex overlap at plane $j ,$

$$
\frac { \partial \chi } { \partial \phi _ { j i } } = 2 \mathrm { R e } \bigl [ \overline { { b _ { i } } } u _ { i } \bigr ] _ { z _ { j } ^ { + } } ,\tag{215}
$$

$$
\frac { \partial \chi } { \partial \gamma _ { j i } } = 2 \mathrm { I m } \left[ \overline { { { b _ { i } } } } u _ { i } \right] _ { z _ { j } ^ { + } } .\tag{216}
$$

Both fields must be evaluated on the same side of plane $j ,$ here the output side $z _ { j } ^ { + }$ . The assignment of the real and imaginary parts follows from the factor −i in $\mathbf { b } ( L ) = - i \mathbb { P } \psi _ { \overline { { \mathbf { u } } } }$ of Definition F.2, which rotates the overlap by a quarter turn.

## H Details for stationary problems

This appendix gives the derivations behind Sec. 9. The setting is the stationary problem

$$
\mathbf { F } \big [ \mathbf { u } , \overline { { \mathbf { u } } } , \mathbf { p } _ { F } \big ] - \mathbf { f } ( \mathbf { p } _ { f } ) = \mathbf { 0 } ,\tag{217}
$$

with an invertible tangent operator at the solution, covering static linear elasticity, linear resistor networks, and frequency-domain scattering at a fixed drive frequency alike. We keep the complex-valued notation throughout. Real problems are the special case treated in the last subsection. The loss is a real-valued function of the measured field,

$$
\chi = \psi \left[ \mathbb { P } \mathbf { u } , \mathbb { P } \overline { { \mathbf { u } } } \right] .\tag{218}
$$

Problem H.1 (Constrained optimization of a stationary system). Find

$$
\operatorname* { m i n } _ { \mathbf { p } } \psi \big [ \mathbb { P } \mathbf { u } , \mathbb { P } \overline { { \mathbf { u } } } \big ] \qquad \mathrm { s u b j e c t ~ t o } \qquad \mathbf { F } \big [ \mathbf { u } , \overline { { \mathbf { u } } } , \mathbf { p } _ { F } \big ] = \mathbf { f } ( \mathbf { p } _ { f } ) ,\tag{219}
$$

over the parameters $ { \mathbf { p } } = [  { \mathbf { p } } _ { F } ,  { \mathbf { p } } _ { \mathbf { f } } ] \in \mathbb { R } ^ { M }$ .

The derivation is the static instance of the template used in the preceding appendices: the field equation is imposed as a constraint with an adjoint multiplier, stationarity with respect to the multiplier returns the field equation, stationarity with respect to the field yields the adjoint equation, and the remaining explicit parameter dependence gives the gradient as a forward–adjoint overlap. For frequency-domain problems this is the standard adjoint-variable sensitivity construction used for Maxwell and Helmholtz systems [58].

## H.1 The stationary adjoint

There is no time, hence no integration by parts and no boundary term, and the single algebraic constraint carries a single multiplier. Pairing the residual through 2 Re so that the Lagrangian is real,

$$
\begin{array} { r } { \mathcal { L } [ { \bf u } , \bar { \bf u } , { \bf a } , \bar { \bf a } , { \bf p } ] = \boldsymbol { \chi } + 2 \mathrm { R e } \big \langle { \bf a } , { \bf F } \big [ { \bf u } , \bar { \bf u } , { \bf p } _ { F } \big ] - { \bf f } ( { \bf p } _ { f } ) \big \rangle _ { \mathbb { C } } , } \end{array}\tag{220}
$$

a real-valued function of independent arguments in which u is an arbitrary field and is not assumed to solve (217). The loss sits inside ${ \mathcal { L } } ,$ so the partial derivatives below already carry its contribution.

Stationarity in the multiplier returns the constraint,

$$
\mathcal { L } _ { \mathbf { a } } = \mathbf { F } \big [ \mathbf { u } , \overline { { \mathbf { u } } } , \mathbf { p } _ { F } \big ] - \mathbf { f } \big ( \mathbf { p } _ { f } \big ) ,\tag{221}
$$

so $\mathcal { L } _ { \mathbf { a } } = 0$ holds if and only if u solves (217), since a real pairing that vanishes in every complex direction forces the residual to vanish, as in Appendix F. Evaluated there the constraint term drops and ${ \mathcal { L } } = \chi ( \mathbf { p } )$ , so diferentiating with one term per argument of $\mathcal { L }$ gives

$$
\frac { d \chi } { d \mathbf { p } } = \mathcal { L } _ { \mathbf { u } } \frac { d \mathbf { u } } { d \mathbf { p } } + \mathcal { L } _ { \bar { \mathbf { u } } } \frac { d \bar { \mathbf { u } } } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { a } } \frac { d \mathbf { a } } { d \mathbf { p } } + \mathcal { L } _ { \bar { \mathbf { a } } } \frac { d \bar { \mathbf { a } } } { d \mathbf { p } } + \mathcal { L } _ { \mathbf { p } } .\tag{222}
$$

The two multiplier terms vanish by (221), whatever the sensitivities of a may be. The expensive object is $d \mathbf { u } / d \mathbf { p }$ , which would otherwise cost one stationary solve per parameter, and it enters only against ${ \mathcal { L } } _ { \mathbf { u } }$ and its conjugate. Choosing a so that ${ \mathcal { L } } _ { \overline { { \mathbf { u } } } } = 0$ , which by conjugacy also gives ${ \mathcal { L } } _ { \mathbf { u } } = 0$ , therefore leaves the key identity $d \chi / d \mathbf { p } = \mathcal { L } _ { \mathbf { p } }$ , and that condition is what defines the adjoint. Writing the real pairing out,

$$
2 \mathrm { R e } \big \langle \mathbf { a } , \mathbf { F } - \mathbf { f } \big \rangle _ { \mathbb { C } } = \overline { { \mathbf { a } } } ^ { T } ( \mathbf { F } - \mathbf { f } ) + \mathbf { a } ^ { T } ( \overline { { \mathbf { F } } } - \overline { { \mathbf { f } } } ) ,\tag{223}
$$

and collecting the terms that carry u, through the Wirtinger Jacobian $\mathbf { F } _ { \overline { { \mathbf { u } } } }$ in the first term and $\overline { { \mathbf { F } } } _ { \overline { { \mathbf { u } } } } = \overline { { \mathbf { F } _ { \mathbf { u } } } }$ in the second,

$$
\mathcal { L } _ { \overline { { \mathbf { u } } } } = \mathbb { P } \psi _ { \overline { { \mathbf { u } } } } \big [ \mathbb { P } \mathbf { u } , \mathbb { P } \overline { { \mathbf { u } } } \big ] + \overline { { \mathbf { F } _ { \mathbf { u } } } } ^ { T } \mathbf { a } + \mathbf { F } _ { \overline { { \mathbf { u } } } } ^ { T } \mathbf { \overline { { a } } } .\tag{224}
$$

Setting it to zero gives the adjoint.

Definition H.2 (Stationary adjoint state). The adjoint state a $\in \mathbb { C } ^ { N }$ associated with Problem H.1 is the solution of ${ \mathcal { L } } _ { \overline { { \mathbf { u } } } } = 0$ , that is

$$
\overline { { \mathbf { F } _ { \mathbf { u } } } } ^ { T } \mathbf { a } + \mathbf { F } _ { \overline { { \mathbf { u } } } } ^ { T } \overline { { \mathbf { a } } } = - \mathbb { P } \psi _ { \overline { { \mathbf { u } } } } \big [ \mathbb { P } \mathbf { u } , \mathbb { P } \overline { { \mathbf { u } } } \big ] ,\tag{225}
$$

with both Wirtinger Jacobians evaluated at the solution of (217).

Theorem H.3 (Stationary adjoint gradient). Let u solve Problem H.1 and a be the adjoint state of Definition H.2. Then, for each parameter $p _ { m }$

$$
\frac { d \chi } { d p _ { m } } = 2 \operatorname { R e } \langle \mathbf { a } , \mathbf { F } _ { p _ { m } } - \mathbf { f } _ { p _ { m } } \rangle _ { \mathbb { C } } .\tag{226}
$$

Proof. With a as in Definition H.2 we have ${ \mathcal { L } } _ { \overline { { \mathbf { u } } } } = 0$ and hence $\mathcal { L } _ { \bf u } = 0$ , so (222) leaves $d \chi / d \mathbf { p } = \mathcal { L } _ { \mathbf { p } }$ , and the explicit parameter dependence of the Lagrangian (220) sits in F and f alone. □

## H.2 Linear systems

For $\mathbf { F } = \mathbf { K } ( \mathbf { p _ { K } } ) \mathbf { u }$ with invertible K, the Wirtinger Jacobians are ${ \bf F } _ { \bf u } = { \bf K }$ and ${ \bf { F } } _ { \overline { { { \bf { u } } } } } = { \bf { 0 } }$ , and the adjoint equation (225) becomes the Hermitian-adjoint system

$$
\mathbf { K } ( \mathbf { p } \mathbf { \mathbf { \mathbf { \mathbf { K } } } } ) ^ { \dagger } \mathbf { a } = - \mathbb { P } \psi _ { \bar { \mathbf { u } } } [ \mathbb { P } \mathbf { u } , \mathbb { P } \bar { \mathbf { u } } ] ,\tag{227}
$$

with the gradients of Theorem H.3 reading

$$
\begin{array} { r l } { \frac { d \chi } { d \mathbf { p } _ { \bf K } } = } & { { } 2 \mathrm { R e } \langle \mathbf { a } , \mathbf { K } _ { \mathbf { p } _ { \bf K } } \mathbf { u } \rangle _ { \mathbb { C } } , } \end{array}\tag{228}
$$

$$
\frac { d \chi } { d \mathbf { p } \mathbf { f } } = - 2 \operatorname { R e } \langle \mathbf { a } , \mathbf { f } _ { \mathbf { p } \mathbf { f } } \rangle _ { \mathbb { C } } .\tag{229}
$$

The conjugated equation (225) becomes

$$
\mathbf { K } ( \mathbf { p } _ { \mathbf { K } } ) ^ { T } \mathbf { c } = - \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } , \mathbb { P } \overline { { \mathbf { u } } } \big ] .\tag{230}
$$

For a reciprocal system, ${ \bf K } ^ { T } = { \bf K }$ , so c is obtained by solving the same forward problem with the source replaced by $- \mathbb { P } \psi _ { \mathbf { u } }$ , applied at the measurement degrees of freedom. In terms of that physically generated field the gradients read

$$
\begin{array} { r l } { \frac { d \chi } { d \mathbf { p } _ { \mathbf { K } } } = } & { { } 2 \operatorname { R e } \left[ \mathbf { c } ^ { T } \mathbf { K } _ { \mathbf { p } _ { \mathbf { K } } } \mathbf { u } \right] , } \end{array}\tag{231}
$$

$$
\frac { d \chi } { d \mathbf { p } \mathbf { f } } = - 2 \operatorname { R e } \bigl [ \mathbf { c } ^ { T } \mathbf { f } _ { \mathbf { p } \mathbf { f } } \bigr ] .\tag{232}
$$

Eq. (230) is Eq. (73) of the main text, and (231)–(232) are the gradients (68)–(69) specialized to $\mathbf { F } = \mathbf { K } \mathbf { u }$

## H.3 Real fields: the static adjoint at a fixed point

When F is real and nonlinear, the trajectory-based construction is obstructed (Theorem 7.1 and Sec. 7.2). What remains accessible is the fixed point of the relaxation

$$
\begin{array} { r } { \dot { \mathbf { u } } ( t ) + \mathbf { F } [ \mathbf { u } ( t ) , \mathbf { p } _ { F } ] - \mathbf { f } ( \mathbf { p } _ { f } ) = \mathbf { 0 } , } \end{array}
$$

with a time-independent drive and a terminal-cost-only objective $( \theta \equiv 0 )$ , assumed to converge to a linearly exponentially stable fixed point $\mathbf { u } ^ { * } ( \mathbf { p } )$ , on which the loss $\psi [ \mathbb { P } { \mathbf { u } } ^ { * } ]$ is read after settling. The relevant parameters are those entering the fixed-point equation. Parameters afecting only the initial condition leave $\mathbf { u } ^ { * }$ unchanged as long as the trajectory stays in the same basin of attraction. For real fields the conjugations in the derivation above drop and the factor 2 Re becomes the Euclidean inner product, so th adjoint equation (225) reduces to the following.

Definition H.4 (Static adjoint state). The static adjoint ${ \bf a } ^ { * } ( { \bf p } )$ is the solution of

$$
\mathbf { F } _ { \mathbf { u } } ^ { T } \big [ \mathbf { u } ^ { * } ( \mathbf { p } ) , \mathbf { p } _ { F } \big ] \mathbf { a } ^ { * } + \mathbb { P } \psi _ { \mathbf { u } } \big [ \mathbb { P } \mathbf { u } ^ { * } ( \mathbf { p } ) \big ] = \mathbf { 0 } .\tag{233}
$$

No separate invertibility hypothesis is needed. Linearizing the forward dynamics about the fixed point gives $\delta \dot { \mathbf { u } } = - \mathbf { F } _ { \mathbf { u } } \delta \mathbf { u } .$ , so linear exponential stability of $\mathbf { u } ^ { * }$ places the spectrum of $\mathbf { F } _ { \mathbf { u } } [ \mathbf { u } ^ { * } , \mathbf { p } _ { F } ]$ in the open right half-plane, Re $\lambda ( { \bf { F } } _ { \bf { u } } ) > 0$ . In particular $\mathbf { F } _ { \mathbf { u } }$ is invertible, $\mathbf { a } ^ { * }$ is the unique solution of (233), and $\mathbf { u } ^ { * } ( \mathbf { p } )$ , $\mathbf { a } ^ { * } ( \mathbf { p } )$ depend smoothly on p by the implicit function theorem.

Theorem H.5 (Static adjoint gradient). Let $\mathbf { u } ^ { * }$ be the linearly exponentially stable fixed point above, so that $\mathbf { F } _ { \mathbf { u } } [ \mathbf { u } ^ { * } , \mathbf { p } _ { F } ]$ is invertible, and $\mathbf { a } ^ { * }$ the static adjoint of Definition $H . 4 .$ . Then

$$
\frac { d \chi } { d \mathbf { p } } = \bigl \langle \mathbf { a } ^ { * } , \mathbf { F } _ { \mathbf { p } _ { F } } [ \mathbf { u } ^ { * } ( \mathbf { p } ) , \mathbf { p } _ { F } ] - \mathbf { f } _ { \mathbf { p } _ { f } } ( \mathbf { p } _ { f } ) \bigr \rangle .\tag{234}
$$

Proof. The real instance of Theorem H.3.

Relation to the time-dependent adjoint. The static adjoint is the time integral of the forward-time adjoint of Appendix E. Take the forward system to be settled, $\mathbf { u } ( t ) \equiv \mathbf { u } ^ { * }$ on the whole window, which is the free phase of the protocol below. The linearization is then constant along the trajectory, and with $\theta \equiv 0$ the running source vanishes, so (177) is the homogeneous relaxation

$$
\begin{array} { r } { \dot { \mathbf { a } } ( t ) + \mathbf { F } _ { \mathbf { u } } ^ { T } [ \mathbf { u } ^ { * } , \mathbf { p } _ { F } ] \mathbf { a } ( t ) = \mathbf { 0 } , \qquad \mathbf { a } ( 0 ) = - \mathbb { P } { \boldsymbol \psi } _ { \mathbf { u } } [ \mathbb { P } \mathbf { u } ^ { * } ] , \qquad \mathbf { a } ( t ) = e ^ { - \mathbf { F } _ { \mathbf { u } } ^ { T } t } \mathbf { a } ( 0 ) . } \end{array}\tag{235}
$$

The spectrum of $\mathbf { F _ { u } }$ lies in the open right half-plane, so a decays to zero and its late-time value carries no information. Its time integral does:

$$
\int _ { 0 } ^ { \infty } \mathbf { a } ( t ) d t = \mathbf { F } _ { \mathbf { u } } ^ { - T } \mathbf { a } ( 0 ) = - \mathbf { F } _ { \mathbf { u } } ^ { - T } \mathbb { P } \psi _ { \mathbf { u } } = \mathbf { a } ^ { * } ,\tag{236}
$$

which is (233). Over a window long compared with the decay time, truncating the integral to [0, T] makes no diference. The two gradient formulas agree for the same reason: on the settled trajectory the second factor in the integrand of (179) is the constant $\mathbf { F } _ { \mathbf { p } _ { F } } [ \mathbf { u } ^ { * } , \mathbf { p } _ { F } ] - \mathbf { f } _ { \mathbf { p } _ { f } } ( \mathbf { p } _ { f } )$ , so the time integral falls entirely on the adjoint and returns $\mathbf { a } ^ { * }$ , recovering (234). The initial-condition term drops out because ${ \mathbf a } ( T )$ has decayed, matching the remark above that parameters entering only $\mathbf { u } _ { 0 }$ leave $\mathbf { u } ^ { * }$ unchanged.

Physical realization: Equilibrium Propagation. The static adjoint (233) requires the transpose $\mathbf { F } _ { \mathbf { u } } ^ { T } \mathrm { . }$ which is generally not realizable on the same device. Reciprocity removes it: for $\mathbf { F } _ { \mathbf { u } } ^ { T } = \mathbf { F } _ { \mathbf { u } } , \mathrm { ~ i . e . ~ } \mathbf { F } = E _ { \mathbf { u } }$ the static adjoint is the infinitesimal diference between the free fixed point and one nudged by the adjoint source, $\begin{array} { r } { \mathbf { a } ^ { * } = \operatorname* { l i m } _ { \epsilon \to 0 } \bigl ( \mathbf { u } ^ { * \epsilon } - \mathbf { u } ^ { * } \bigr ) / \epsilon } \end{array}$ , the real-valued instance of Eq. (76). Both relaxations descend the energy $E$ tilted by the drive, the second additionally by $\epsilon \psi$ , the two-phase energy contrast of Ref. [30]. Substituting the diference of equilibria into Theorem H.5 gives the learning rule.

## H.4 Layered photonic networks

Integrated photonic neural networks can be described as a sequence of frequency-domain optical solves and elementwise activations. For $\ell = 1 , \ldots , L$ , write

$$
\mathbf { K } _ { \ell } ( \mathbf { p } ) \mathbf { u } _ { \ell } = \mathbf { B } _ { \ell } \mathbf { X } _ { \ell - 1 } ,\tag{237a}
$$

$$
\mathbf { Z } _ { \ell } = \mathbf { P } _ { \ell } \mathbf { u } _ { \ell } ,\tag{237b}
$$

$$
\begin{array} { r } { \mathbf { X } _ { \ell } = f _ { \ell } \big ( \mathbf { Z } _ { \ell } , \overline { { \mathbf { Z } } } _ { \ell } \big ) . } \end{array}\tag{237c}
$$

Here $\mathbf { K } _ { \ell }$ is the reciprocal frequency-domain operator of layer ℓ, $\mathbf { B } _ { \ell }$ injects the incoming modal amplitudes into the device, and $\mathbf { P } _ { \ell }$ projects the internal field onto output modes. Eliminating the field gives the usual layer transfer matrix

$$
\begin{array} { r } { \mathbf { Z } _ { \ell } = \widehat { \mathbf { W } } _ { \ell } \mathbf { X } _ { \ell - 1 } , \qquad \widehat { \mathbf { W } } _ { \ell } : = \mathbf { P } _ { \ell } \mathbf { K } _ { \ell } ^ { - 1 } \mathbf { B } _ { \ell } . } \end{array}\tag{238}
$$

Applying the Lagrangian construction to the three constraints in Eq. (237) gives three adjoint variables: the optical adjoint field $\mathbf { a } _ { \ell } ,$ the pre-activation error $\delta _ { \ell } ,$ , and the output error $\gamma _ { \ell }$ . Here we pair each residual bilinearly, through $2 \mathrm { R e } [ \mathbf { a } ^ { T } ( \cdot ) ]$ rather than through the Hermitian $2 \operatorname { R e } \langle \mathbf { a } , \cdot \rangle _ { \mathbb { C } }$ used above. The two difer only by conjugating the multiplier, and the bilinear choice is the convenient one here because it produces transposed rather than Hermitian-adjoint operators, the combination a reciprocal device realizes. It is the layered counterpart of passing to $\mathbf { c } = \mathbf { \overline { { a } } }$ in Eq. (230). The backward equations are

$$
{ \bf K } _ { \ell } ^ { T } { \bf a } _ { \ell } = { \bf P } _ { \ell } ^ { T } \delta _ { \ell } ,\tag{239}
$$

$$
\delta _ { \ell } = ( f _ { \ell } ) _ { \mathbf { Z } } ^ { T } \gamma _ { \ell } + \overline { { ( f _ { \ell } ) _ { \mathbf { Z } } } } ^ { T } \overline { { \gamma } } _ { \ell } ,\tag{240}
$$

$$
\begin{array} { r } { \gamma _ { \ell - 1 } = \mathbf { B } _ { \ell } ^ { T } \mathbf { a } _ { \ell } , \qquad \gamma _ { L } = - \psi _ { \mathbf { X } _ { L } } . } \end{array}\tag{241}
$$

Taken from $\ell = L$ down to 1, these equations form the physical backpropagation [23, 34]: begin at the output error $\gamma _ { L } = - \psi _ { \mathbf { X } _ { L } } .$ , pull it through the activation (240) to obtain $\delta _ { \ell } ,$ solve the adjoint field equation (239) for $\mathbf { a } _ { \ell }$ , and read of the preceding layer’s output error $\boldsymbol { \gamma } _ { \ell - 1 } = \mathbf { B } _ { \ell } ^ { T } \mathbf { a } _ { \ell }$

If the optical layer is reciprocal, ${ \mathbf K } _ { \ell } ^ { T } = { \mathbf K } _ { \ell } .$ , then Eq. (239) is a physical solve in the same device, excited from the output side with source $\mathbf { P } _ { \ell } ^ { T } \dot { \delta _ { \ell } }$ . Combining Eqs. (239) and (241) gives

$$
\boldsymbol { \gamma } _ { \ell - 1 } = \mathbf { B } _ { \ell } ^ { T } \mathbf { K } _ { \ell } ^ { - T } \mathbf { P } _ { \ell } ^ { T } \delta _ { \ell } = \widehat { \mathbf { W } } _ { \ell } ^ { T } \delta _ { \ell } .\tag{242}
$$

Thus the physical adjoint solve implements the transpose of the forward optical map.

The parameter gradient follows from the explicit parameter dependence of the layer residuals:

$$
\frac { d \boldsymbol { \chi } } { d p _ { m } } = 2 \mathrm { \normalfont ~ R e } \sum _ { \ell = 1 } ^ { L } \left[ { \mathbf a } _ { \ell } ^ { T } \left( ( { \mathbf K } _ { \ell } ) _ { p _ { m } } { \mathbf u } _ { \ell } - ( { \mathbf B } _ { \ell } ) _ { p _ { m } } { \mathbf X } _ { \ell - 1 } \right) - \delta _ { \ell } ^ { T } ( { \mathbf P } _ { \ell } ) _ { p _ { m } } { \mathbf u } _ { \ell } - \gamma _ { \ell } ^ { T } ( f _ { \ell } ) _ { p _ { m } } \right] .\tag{243}
$$

When the trainable parameters live only in the optical operators $\mathbf { K } _ { \ell } ,$ this reduces to the local field overlap

$$
\frac { d \chi } { d p _ { m } } = 2 \ \mathrm { R e } \sum _ { \ell = 1 } ^ { L } \mathbf { a } _ { \ell } ^ { T } ( \mathbf { K } _ { \ell } ) _ { p _ { m } } \mathbf { u } _ { \ell } .\tag{244}
$$

For a local permittivity perturbation in a phase shifter region, $( \mathbf { K } _ { \ell } ) _ { \epsilon _ { m } }$ is supported only on that region, so Eq. (244) is measured by a local product of the forward and adjoint optical fields.

## H.5 Equivalence with scattering-matrix learning rules

The learning rules derived for linear wave-scattering networks by Wanjura and Marquardt [37] express each parameter gradient as a product of two measured scattering responses. We show that these two responses

are the forward and the adjoint field of Appendix H.2, so that their rule is the adjoint identity read at single ports.

Their neuron modes obey the coupled-mode and input–output equations

$$
{ \bf K } { \bf u } = - i { \bf { \Gamma } } { \bf q } , \qquad { \bf q } ^ { \mathrm { o u t } } = { \bf S } { \bf q } , \qquad { \bf S } = { \bf I } + { \bf { \Gamma } } \mathcal { G } { \bf { \Gamma } } ,\tag{245}
$$

with q the probe amplitudes, $\mathbf { r } = \mathrm { d i a g } ( \sqrt { \kappa _ { 1 } } , \ldots , \sqrt { \kappa _ { N } } )$ the couplings to the probe waveguides, $\mathbf { K } = \omega \mathbf { I } - \mathbf { H }$ and $\mathcal { G } = - i \mathbf { K } ^ { - 1 }$ . The dynamical matrix H collects the detunings and couplings together with the decay rates $- { \frac { i } { 2 } } \mathbf { { \cal T } } ^ { 2 }$ , so K is symmetric without being Hermitian: the network is reciprocal, ${ \bf K } ^ { T } = { \bf K }$ and $\mathcal { G } ^ { T } = \mathcal { G }$ and the conjugated adjoint equation (230) applies.

Both fields are probe responses. Probing port p with unit amplitude, $\mathbf { q } = \mathbf { e } _ { p } ,$ , the forward field is the p-th column of $\mathcal { G }$

$$
\mathbf { u } = \sqrt { \kappa _ { p } } \mathcal { G } \mathbf { e } _ { p } .\tag{246}
$$

The network output is a quadrature of the scattering amplitude recorded at port r $\cdot , \ y = \ \operatorname { R e } ( e ^ { i \phi } S _ { r , p } )$ and $S _ { r , p } = \delta _ { r , p } + \sqrt { \kappa _ { r } } u _ { r } .$ , so with the real weight $\lambda : = \partial \chi / \partial y$ the loss derivative entering (230) is $\psi _ { \bf u } = $ $\frac { \lambda } { 2 } e ^ { i \phi } \sqrt { \kappa _ { r } } \mathbf { e } _ { r }$ . Reciprocity then delivers the adjoint field as

$$
\mathbf { c } = - \mathbf { K } ^ { - 1 } \psi _ { \mathbf { u } } = - \frac { i \lambda } { 2 } e ^ { i \phi } \sqrt { \kappa _ { r } } \mathcal { G } \mathbf { e } _ { r } ,\tag{247}
$$

the r-th column of the same ${ \mathcal { G } } .$ . The adjoint field is the device’s response to a probe at the readout port, so both factors of the gradient are scattering measurements on the same hardware.

The gradient is their product rule. Let $\theta _ { m }$ be a trainable parameter of H, so that ${ \bf K } _ { \theta _ { m } } = - { \bf H } _ { \theta _ { m } }$ Substituting (246) and (247) into the gradient (231) and using $\mathcal { G } ^ { T } = \mathcal { G }$ 2

$$
\frac { d \chi } { d \theta _ { m } } = 2 \operatorname { R e } \left[ \mathbf { c } ^ { T } \mathbf { K } _ { \theta _ { m } } \mathbf { u } \right] = \lambda \operatorname { R e } \left[ e ^ { i \phi } i \sqrt { \kappa _ { p } \kappa _ { r } } \left( \mathcal { G } \mathbf { H } _ { \theta _ { m } } \mathcal { G } \right) _ { r , p } \right] .\tag{248}
$$

The bracket is the sensitivity of the scattering matrix itself: diferentiating $\begin{array} { r } { \mathbf { S } = \mathbf { I } + \mathbf { T } \mathcal { G } \mathbf { I } } \end{array}$ with $\partial _ { \theta _ { m } } { \bf K } ^ { - 1 } =$ $\mathbf { K } ^ { - 1 } \mathbf { H } _ { \theta _ { m } } \mathbf { K } ^ { - 1 }$ gives

$$
\frac { \partial S _ { r , p } } { \partial \theta _ { m } } = i \sqrt { \kappa _ { p } \kappa _ { r } } \left( \mathcal { G } \mathbf { H } _ { \theta _ { m } } \mathcal { G } \right) _ { r , p } ,\tag{249}
$$

so (248) is the chain rule $d \chi / d \theta _ { m } = ( \partial \chi / \partial y ) ( \partial y / \partial \theta _ { m } )$ , with several outputs simply summed.

Read at single ports. Inverting the input–output relation gives $\mathcal { G } _ { m , n } = ( S _ { m , n } - \delta _ { m , n } ) / \sqrt { \kappa _ { m } \kappa _ { n } } \mathrm { ~ . ~ }$ so the overlap in (249) is a product of measured amplitudes. Writing $\mathbf { E } _ { j , \ell }$ for the matrix whose only nonzero entry is a unit entry at $( j , \ell )$ , an on-site detuning has $\mathbf { H } _ { \Delta _ { j } } = \mathbf { E } _ { j , j }$ and

$$
\frac { \partial S _ { r , p } } { \partial \Delta _ { j } } = \frac { i } { \kappa _ { j } } ( S _ { r , j } - \delta _ { r , j } ) ( S _ { j , p } - \delta _ { j , p } ) ,\tag{250}
$$

the response from the probe port to the perturbed site times the response from that site to the readout port, while a reciprocal coupling has $\mathbf { H } _ { J _ { j , \ell } } = \mathbf { E } _ { j , \ell } + \mathbf { E } _ { \ell , j }$ and

$$
\frac { \partial S _ { r , p } } { \partial J _ { j , \ell } } = \frac { i } { \sqrt { \kappa _ { j } \kappa _ { \ell } } } \big [ ( S _ { r , j } - \delta _ { r , j } ) ( S _ { \ell , p } - \delta _ { \ell , p } ) + ( S _ { r , \ell } - \delta _ { r , \ell } ) ( S _ { j , p } - \delta _ { j , p } ) \big ] ,\tag{251}
$$

the sum over the two paths through the coupled pair. These are the learning rules of Ref. [37], written here in terms of the measured amplitudes $S _ { m , n } - \delta _ { m , n }$ . Their rule is therefore an instance of the adjoint method, in which the drive and the measurement each address a single port and the backward pass is a second scattering experiment on the same device.

## I How to test a gradient: the Taylor test

Every implementation of an adjoint method requires a simple, model-agnostic way of checking that it is correct. The standard tool, used widely in the PDE-constrained optimization literature, is the Taylor test [103, 104].

Let $\mathbf { g } : = d \chi / d \mathbf { p }$ denote the gradient produced by the adjoint procedure, and let $\mathbf { v } \in \mathbb { R } ^ { M }$ be a random direction (e.g. a Gaussian vector). For a small step size $h > 0$ , define the zeroth- and first-order remainders

$$
R _ { 0 } ( h ) \ : = \ \big | \chi ( \mathbf { p } + h \mathbf { v } ) \ - \ \chi ( \mathbf { p } ) \big | ,\tag{252}
$$

$$
R _ { 1 } ( h ) : = { \mathbf { \left| \nabla \chi ( p + h { \mathbf { v } } ) \right. } - \mathbf { \nabla } \chi ( \mathbf { p } ) \mathbf { \nabla } - \mathbf { \nabla } h \left. \mathbf { g } , \mathbf { v } \right. \mathbf { \nabla } \big | } .\tag{253}
$$

A Taylor expansion of $\chi$ about p gives

$$
R _ { 0 } ( h ) \ = \ h | \langle { \bf g } , { \bf v } \rangle | \ + \ { \mathcal O } ( h ^ { 2 } ) , \qquad R _ { 1 } ( h ) \ = \ { \mathcal O } ( h ^ { 2 } ) .\tag{254}
$$

The diagnostic is a log–log plot of $R _ { 0 } ( h )$ and $R _ { 1 } ( h )$ as h is swept through several decades:

$R _ { 0 }$ should decrease with slope 1 in log h.

$R _ { 1 }$ should decrease with slope 2, provided g is the true gradient.

• If $\mathbf { g }$ is wrong, $R _ { 1 }$ instead decreases with slope 1, and the curve typically transitions from a slope of 1 at large h to a noise-dominated floor at small h, producing the characteristic “hockey-stick” shape.

For very small $h , R _ { 1 }$ becomes dominated by floating-point round-of in the forward evaluation. The Taylor test should therefore be read by reporting the slope over the regime where $R _ { 1 }$ still decreases monotonically.